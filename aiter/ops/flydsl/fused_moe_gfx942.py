# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    flydsl_quant_per_token,
    per_token_quant_supported,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import launch


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool
    down_path: str = "default"
    down_output_padding_bytes: int = 0
    use_batch1_algorithm: bool = False

    def __post_init__(self):
        if self.down_path not in (
            "default",
            "1x4_64x256",
            "8x1_compact",
            "m128",
            "m128p",
        ):
            raise ValueError(f"Unsupported Down path: {self.down_path}")
        if self.down_output_padding_bytes not in (0, 32, 64, 128):
            raise ValueError("Down output padding must be 0, 32, 64 or 128 bytes")
        if (
            self.down_path in ("default", "m128", "m128p")
            and self.down_output_padding_bytes != 0
        ):
            raise ValueError(f"The {self.down_path} Down path does not support output padding")
        if self.down_path != "default" and not self.use_prefill:
            raise ValueError("Specialized Down paths require prefill")
        if self.use_batch1_algorithm and (
            self.use_prefill or self.down_path != "default"
        ):
            raise ValueError(
                "Direct decoding cannot use prefill or specialized Down paths"
            )

    def to_string(self):
        # Only direct decoding adds the fifth tile field, so existing strings are unchanged.
        batch1 = "_True" if self.use_batch1_algorithm else ""
        return (
            f"{self.BLOCK_M}_{self.BLOCK_N}_{self.BLOCK_K}_{self.use_prefill}{batch1}:"
            f"{self.down_path}:{self.down_output_padding_bytes}"
        )

    @classmethod
    def from_string(cls, data: str):
        parts = data.split(":")
        # Configs tuned before the Down suffix existed use the default Down path.
        if len(parts) == 1:
            parts += ["default", "0"]
        tile, down_path, padding = parts
        fields = tile.split("_")
        if len(fields) == 4:
            fields.append("False")
        block_m, block_n, block_k, use_prefill, use_batch1_algorithm = fields
        if use_prefill not in ("True", "False") or use_batch1_algorithm not in (
            "True",
            "False",
        ):
            raise ValueError(f"Invalid algorithm flags: {tile}")
        return cls(
            int(block_m),
            int(block_n),
            int(block_k),
            use_prefill == "True",
            down_path,
            int(padding),
            use_batch1_algorithm=use_batch1_algorithm == "True",
        )

    @property
    def four_wave_gateup(self) -> bool:
        # BLOCK_M=256 prefill selects the gfx950 four-wave 256x256 FP8 Gate/Up; BLOCK_M=128
        # runs it on half-height tiles (for blocks of at most 128 routes).
        return self.use_prefill and self.BLOCK_M in (128, 256)

    def unsupported_reason(self, problem: "_Problem") -> str | None:
        if self.use_batch1_algorithm and not 2 <= problem.batch <= 8:
            return "direct decoding requires 2 to 8 tokens"
        # Batch 1 runs the batch-1 kernels for any tile; tune it only as 16_16_16.
        if not self.use_prefill and problem.batch == 1 and self.BLOCK_N != 16:
            return "batch 1 uses the 16_16_16 config"
        if self.four_wave_gateup:
            reason = _four_wave_gateup_reason(self, problem)
            if reason is not None:
                return reason
        if self.down_path in ("m128", "m128p"):
            from aiter.jit.utils.chip_info import get_gfx_runtime
            from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm2_4x1 import (
                m128_supported,
            )

            if get_gfx_runtime() != "gfx950":
                return "M128 Down requires gfx950"
            # The M128 Down reads 256-row sort blocks; Gate/Up walks them in BLOCK_M tiles.
            if 256 % self.BLOCK_M != 0:
                return "M128 Down needs a Gate/Up BLOCK_M that divides 256"
            if problem.quant_type not in ("", "ptpc"):
                return "M128 Down requires per-token FP8 activations and weights"
            if not m128_supported(problem.model_dim, problem.inter_dim):
                return "M128 Down needs inter_dim % 64 == 0 in [128, 640] and N128-pair column slices"
        elif self.down_path != "default":
            if self.BLOCK_M != 64 and not self.four_wave_gateup:
                return "specialized Down paths require unchanged M64 Gate/Up metadata"
            # The native tuner's empty quant_type is a shape-only probe. The
            # runtime wrapper checks actual FP8 weight and activation support.
            if problem.quant_type not in ("", "ptpc", "per_tensor"):
                return "specialized Down paths require FP8 weights"
            if self.down_path == "8x1_compact":
                if problem.inter_dim not in (192, 256, 384, 512, 640):
                    return "compact Down requires inter_dim in {192,256,384,512,640}"
                if problem.model_dim < 512:
                    return "compact Down requires model_dim >= 512"
                if not 0 < problem.experts <= 2048:
                    return "compact Down requires 1 to 2048 experts"
            scale_bytes = 0 if problem.quant_type == "per_tensor" else 256 * 4
            if 64 * problem.inter_dim + scale_bytes + 4 * 16 * 64 * 2 > 64 * 1024:
                return "specialized M64 Down exceeds the 64 KiB LDS capacity"
        if self.use_prefill and problem.quant_type == "no":
            if self.BLOCK_K not in (64, 128):
                return "BF16 prefill requires Gate/Up BLOCK_K=64 or 128"
            if 2 * self.BLOCK_M * self.BLOCK_K * 2 > 64 * 1024:
                return "BF16 Gate/Up ping-pong exceeds the 64 KiB LDS capacity"
            if self.BLOCK_M * problem.inter_dim * 2 > 64 * 1024:
                return "BF16 default Down exceeds the 64 KiB LDS capacity"
        if (
            self.use_prefill
            and not self.four_wave_gateup
            and problem.gateup_dim % self.BLOCK_N != 0
        ):
            return (
                f"gateup_dim={problem.gateup_dim} is not divisible by "
                f"BLOCK_N={self.BLOCK_N}"
            )
        if self.use_prefill and problem.inter_dim % 64 != 0:
            return (
                f"inter_dim={problem.inter_dim} is not divisible by the "
                "down BLOCK_K=64"
            )
        if self.use_prefill and problem.hidden_dim % (2 * self.BLOCK_K) != 0:
            return (
                f"hidden_dim={problem.hidden_dim} is not divisible by "
                f"2*BLOCK_K={2 * self.BLOCK_K} for the gateup pipeline"
            )
        if self.use_prefill and problem.model_dim % 256 != 0:
            return (
                f"model_dim={problem.model_dim} is not divisible by 256; "
                "the prefill down pipeline processes paired 128-wide tiles"
            )
        return None


@dataclass(frozen=True)
class _Problem:
    batch: int
    experts: int
    gateup_dim: int
    hidden_dim: int
    model_dim: int
    inter_dim: int
    topk: int
    quant_type: str

    @classmethod
    def from_inputs(
        cls,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_type: QuantType,
    ):
        experts, gateup_dim, hidden_dim = w1.shape
        model_dim, inter_dim = w2.shape[1], w2.shape[2]
        assert gateup_dim == 2 * inter_dim
        return cls(
            batch=int(hidden_states.shape[0]),
            experts=experts,
            gateup_dim=gateup_dim,
            hidden_dim=hidden_dim,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            quant_type={
                QuantType.No: "no",
                QuantType.per_Token: "ptpc",
                QuantType.per_Tensor: "per_tensor",
            }[quant_type],
        )


def get_tune_space():
    prefill = [(256, 128), (128, 256), (128, 128)]
    configs = [
        Config(16, 16, 16, False),
        Config(16, 16, 16, False, use_batch1_algorithm=True),
        Config(16, 128, 64, False),
    ]
    configs.extend(Config(64, block_n, block_k, True) for block_n, block_k in prefill)
    configs.extend(
        Config(
            64, block_n, block_k, True, down_path=path, down_output_padding_bytes=128
        )
        for path in ("1x4_64x256", "8x1_compact")
        for block_n, block_k in prefill
    )
    # M64 Gate/Up on the M128 Down's 256-row sort blocks (gfx950).
    configs.extend(
        Config(64, block_n, block_k, True, down_path=path)
        for path in ("m128", "m128p")
        for block_n, block_k in prefill
    )
    # gfx950 four-wave 256x256 Gate/Up (full and half-height tiles); unsupported_reason
    # filters other targets.
    configs.extend(
        Config(block_m, 256, 128, True, down_path=path, down_output_padding_bytes=pad)
        for block_m in (256, 128)
        for path, pad in (
            ("default", 0),
            ("1x4_64x256", 128),
            ("8x1_compact", 128),
            ("m128", 0),
            ("m128p", 0),
        )
    )
    return [config.to_string() for config in configs]


def _four_wave_gateup_reason(config: Config, problem: "_Problem") -> str | None:
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm1_2x2 import (
        gateup_4w_supported,
    )

    if (config.BLOCK_N, config.BLOCK_K) != (256, 128):
        return "four-wave Gate/Up uses the {256,128}_256_128 tiles"
    if get_gfx_runtime() != "gfx950":
        return "four-wave Gate/Up requires gfx950"
    if problem.quant_type not in ("", "ptpc", "per_tensor"):
        return "four-wave Gate/Up requires FP8 weights"
    if not gateup_4w_supported(problem.gateup_dim, problem.hidden_dim):
        return "four-wave Gate/Up needs inter_dim % 16 == 0 and hidden_dim % 256 == 0 >= 512"
    return None


@cache
def _get_gateup_4w_kernel(N, K, TOPK, quant_type_str, activation_str, swiglu_limit, rows=256):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm1_2x2 import (
        build_moe_gateup_4w,
    )

    if activation_str == "swiglu":
        swiglu_limit = float(swiglu_limit) if swiglu_limit else 7.0
    return build_moe_gateup_4w(
        N=N,
        K=K,
        TOPK=TOPK,
        activation=activation_str,
        swiglu_limit=swiglu_limit,
        act_quant_type=quant_type_str,
        weight_quant_type=quant_type_str,
        rows=rows,
    )


@cache
def _get_compiled_kernel(
    N,
    K,
    weight_dtype_str,
    quant_type_str,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage,
    alg,
    E,
    act_quant_type_str=None,
    BLOCK_TILE_SIZE_K=None,
    activation_str="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=0,
    down_n_splits=1,
    fused_down_clear=False,
    sort_block_m=None,
):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import compile_gemm

    return compile_gemm(
        N=N,
        K=K,
        weight_dtype=weight_dtype_str,
        weight_quant_type=quant_type_str,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        tile_k=BLOCK_TILE_SIZE_K,
        stage=stage,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=True,
        act_quant_type=act_quant_type_str,
        activation=activation_str,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        down_n_splits=down_n_splits,
        fused_down_clear=fused_down_clear,
        sort_block_m=sort_block_m,
    )


def _launch(kernel_fn, *args):
    launch(kernel_fn, *args)


@cache
def _parse_config(config_string: str) -> Config:
    return Config.from_string(config_string)


@lru_cache(maxsize=4096)
def _unsupported_reason(config_string: str, problem: "_Problem") -> str | None:
    return _parse_config(config_string).unsupported_reason(problem)


def _quant_per_tensor(x, scale=None, quant_dtype=torch.float8_e4m3fn, num_rows=None):
    assert scale is None
    assert num_rows is None

    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    xq = torch.empty_like(x, dtype=quant_dtype)
    flydsl_absmax()(x, amax)
    flydsl_quant_per_tensor(quant_dtype)(x, amax, xq)
    fmax = torch.finfo(quant_dtype).max
    xs = amax / fmax
    xs = xs.reshape(1).to(torch.float32)

    return xq, xs


def _empty_scale(device):
    return torch.empty(0, device=device)


def _down_n_splits(problem: _Problem, down_path: str) -> int:
    """Split each 1x4_64x256 Down task's N until every CU has about six tasks."""
    if down_path != "1x4_64x256":
        return 1
    rows = problem.batch * problem.topk
    # Each active expert pads half an M64 block on average.
    tasks = rows / 64 + min(problem.experts, rows) / 2
    tiles = problem.model_dim // 256
    for splits in range(1, tiles):
        if tiles % splits == 0 and tasks * splits >= 6 * get_cu_num():
            return splits
    return tiles


_DOWN_QUEUE_HEADS = {}


def _m128_oc_splits(problem: _Problem, merge_halves: bool) -> int | None:
    """OC split of an M128 Down above K256 (None keeps the builder default).

    Every OC slice re-reads its block's K-wide A rows, so slices stay at least K
    columns wide; the narrowest split giving eight tasks per worker wins.
    """
    n, k = problem.model_dim, problem.inter_dim
    if k <= 256:
        return None
    rows = problem.batch * problem.topk
    block_rows = 256 if merge_halves else 128
    # Each active expert pads half a block on average.
    blocks = rows / block_rows + min(problem.experts, rows) / 2
    valid = [oc for oc in (1, 2, 4, 8, 16) if n % oc == 0 and n // oc % 128 == 0 and n // oc >= k]
    for oc in valid:
        if blocks * oc >= 8 * 2 * get_cu_num():
            return oc
    return valid[-1] if valid else None


def _down_queue_heads(device):
    # Persistent M128 Down task queues; each launch leaves the heads at zero.
    if device not in _DOWN_QUEUE_HEADS:
        _DOWN_QUEUE_HEADS[device] = torch.zeros(256, dtype=torch.int32, device=device)
    return _DOWN_QUEUE_HEADS[device]


def _gateup_output(hidden_states: torch.Tensor, problem: _Problem):
    return torch.empty(
        [problem.batch, problem.topk, problem.inter_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


def _run_prefill(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type: QuantType,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
):
    m128_down = config.down_path in ("m128", "m128p")
    # The M128 Down and the four-wave Gate/Up read 256-row sort blocks; M64 Gate/Up
    # walks them in BLOCK_M tiles.
    sort_block_m = 256 if m128_down or config.four_wave_gateup else config.BLOCK_M
    needs_resort = config.four_wave_gateup and not m128_down
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        cur_out,
        *sort_aux,
    ) = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        sort_block_m,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
        accumulate=False,
        output_aux=False if needs_resort else AUX_SORT_OPUS,
    )
    reverse_sorted = None
    if not needs_resort:
        _, reverse_sorted = sort_aux
    if cur_out.numel() == 0:
        cur_out = torch.empty(
            [problem.batch, problem.model_dim],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    act_quant_type_str = problem.quant_type
    quant_func = (
        aiter.get_hip_quant(aiter.QuantType.per_Token)
        if quant_type == QuantType.per_Token
        else _quant_per_tensor
    )

    if weight_dtype_str == "fp8":
        gateup_in, a_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            num_rows=None,
        )
        a_scale = a_scale.to(torch.float32).contiguous()
    else:
        gateup_in = hidden_states
        a_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    gemm1_out = _gateup_output(hidden_states, problem)
    if config.four_wave_gateup:
        gateup_kernel = _get_gateup_4w_kernel(
            problem.gateup_dim,
            problem.hidden_dim,
            problem.topk,
            problem.quant_type,
            activation_str,
            swiglu_limit,
            rows=config.BLOCK_M,
        )
    else:
        gateup_kernel = _get_compiled_kernel(
            N=problem.gateup_dim,
            K=problem.hidden_dim,
            weight_dtype_str=weight_dtype_str,
            quant_type_str=problem.quant_type,
            TOPK=problem.topk,
            BLOCK_TILE_SIZE_M=config.BLOCK_M,
            BLOCK_TILE_SIZE_N=config.BLOCK_N,
            BLOCK_TILE_SIZE_K=config.BLOCK_K,
            stage="gateup",
            alg="prefill_1x4",
            E=problem.experts,
            act_quant_type_str=act_quant_type_str,
            activation_str=activation_str,
            swiglu_limit=swiglu_limit,
            sort_block_m=sort_block_m,
        )
    task_num = int(sorted_expert_ids.shape[0])
    if config.four_wave_gateup:
        from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm1_2x2 import (
            gateup_4w_workspace,
        )

        stream_k_args = gateup_4w_workspace(hidden_states.device)
    else:
        stream_k_args = ()
    _launch(
        gateup_kernel,
        gateup_in,
        w1,
        gemm1_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        a_scale,
        *stream_k_args,
        problem.batch,
        task_num * (sort_block_m // config.BLOCK_M),
    )

    if weight_dtype_str == "fp8":
        down_rows = gemm1_out.view(problem.batch * problem.topk, -1)
        if (
            quant_type == QuantType.per_Token
            and down_rows.dtype == torch.bfloat16
            and per_token_quant_supported(problem.inter_dim)
        ):
            down_in = torch.empty_like(down_rows, dtype=w2.dtype)
            down_in_scale = torch.empty(
                [down_rows.shape[0], 1],
                dtype=torch.float32,
                device=hidden_states.device,
            )
            flydsl_quant_per_token(w2.dtype, problem.inter_dim)(
                down_rows, down_in, down_in_scale
            )
        else:
            down_in, down_in_scale = quant_func(
                down_rows,
                scale=None,
                quant_dtype=w2.dtype,
                num_rows=None,
            )
    else:
        down_in = gemm1_out
        down_in_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    w2_scale_arg = (
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device)
    )
    if m128_down:
        from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm2_4x1 import (
            build_moe_down_m128,
            make_packed_sum,
        )

        # Packed [block128, N64, row128, col64] routes, summed per token below.
        packed = torch.empty(
            [task_num * sort_block_m, problem.model_dim],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        merge_halves = problem.batch * problem.topk < 128 * problem.experts
        down_kernel = build_moe_down_m128(
            N=problem.model_dim,
            K=problem.inter_dim,
            TOPK=problem.topk,
            sort_block_m=sort_block_m,
            persistent=config.down_path == "m128p",
            # Under 128 routes per expert on average most second halves are empty.
            merge_halves=merge_halves,
            oc_splits=_m128_oc_splits(problem, merge_halves),
        )
        _launch(
            down_kernel,
            down_in,
            w2,
            packed,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            w2_scale_arg,
            down_in_scale,
            _down_queue_heads(hidden_states.device),
            problem.batch,
            task_num,
        )
        loc_ids = reverse_sorted.view(problem.batch, problem.topk)
        _launch(
            make_packed_sum(N=problem.model_dim, TOPK=problem.topk),
            cur_out,
            packed,
            loc_ids,
            sorted_weights,
            down_in_scale,
            problem.batch,
            packed.shape[0],
        )
        return cur_out

    down_block_m = config.BLOCK_M
    if needs_resort:
        # The M64 Down kernels index 64-row sorted blocks.
        down_block_m = 64
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            _,
            _,
            reverse_sorted,
        ) = moe_sorting(
            topk_ids,
            topk_weight,
            problem.experts,
            problem.model_dim,
            hidden_states.dtype,
            down_block_m,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            accumulate=False,
            output_aux=AUX_SORT_OPUS,
        )
        task_num = int(sorted_expert_ids.shape[0])

    output_row_size = (
        problem.model_dim
        + config.down_output_padding_bytes // hidden_states.element_size()
    )
    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * down_block_m, output_row_size],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=down_block_m,
        BLOCK_TILE_SIZE_N=128,
        stage="down",
        alg="prefill_1x4",
        E=problem.experts,
        act_quant_type_str=act_quant_type_str,
        down_path=config.down_path,
        down_output_padding_bytes=config.down_output_padding_bytes,
        down_n_splits=_down_n_splits(problem, config.down_path),
    )
    down_args = (
        down_in,
        w2,
        gemm2_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale_arg,
        down_in_scale,
        problem.batch,
        task_num,
    )
    if config.down_path == "8x1_compact":
        from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm2_8x1_compact import (
            allocate_task_buffers,
        )

        # The split policy and its buffer capacities must use the same CU count.
        cu_count = get_cu_num()
        full_tasks, tail_tasks, counts = allocate_task_buffers(
            sorted_expert_ids, problem.experts, cu_count
        )
        _launch(
            down_kernel,
            *down_args,
            full_tasks,
            tail_tasks,
            counts,
            full_tasks.shape[0],
            tail_tasks.shape[0],
            cu_count,
        )
    else:
        _launch(down_kernel, *down_args)

    loc_ids = reverse_sorted.view(problem.batch, problem.topk)
    sorted_sum(problem.topk, problem.model_dim, config.down_output_padding_bytes)(
        loc_ids, gemm2_out, cur_out, problem.batch
    )
    return cur_out


def _run_batch1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
):
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    # Gate/Up zeroes the output itself when the output row matches its input row.
    fused_down_clear = problem.model_dim == problem.hidden_dim
    allocate_output = torch.empty if fused_down_clear else torch.zeros
    cur_out = allocate_output(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    route_count = problem.batch * problem.topk
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32,
        stage="gateup",
        alg="batch1",
        E=None,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
        fused_down_clear=fused_down_clear,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        cur_out,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        route_count,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64,
        stage="down",
        alg="batch1",
        E=None,
    )
    _launch(
        down_kernel,
        gemm1_out,
        w2,
        cur_out,
        topk_ids,
        topk_weight,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        route_count,
    )
    return cur_out


def _run_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    grid = int(sorted_expert_ids.shape[0])
    if problem.batch * problem.topk <= problem.experts:
        grid = problem.batch * problem.topk

    block_n = 64 if config.BLOCK_N == 16 else config.BLOCK_N
    gemm1_out = _gateup_output(hidden_states, problem)
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="gateup",
        alg="splitk",
        E=problem.experts,
        activation_str=activation_str,
        swiglu_limit=swiglu_limit,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        grid,
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=block_n,
        stage="down",
        alg="splitk",
        E=problem.experts,
    )
    _launch(
        down_kernel,
        gemm1_out,
        w2,
        cur_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        grid,
    )
    return cur_out


def run_flydsl_moe_gfx942(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: ActivationType,
    quant_type: QuantType,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    expert_mask: Any,
    num_local_tokens: Any,
    moe_sorting_dispatch_policy: int,
    config_string: str,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if num_local_tokens is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support num_local_tokens"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("w1", w1),
        ("w2", w2),
        ("topk_weight", topk_weight),
        ("topk_ids", topk_ids),
        ("w1_scale", w1_scale),
        ("w2_scale", w2_scale),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise NotImplementedError(
                f"gfx942 FlyDSL whole-graph backend requires contiguous {name}"
            )
    config = _parse_config(config_string)
    is_bf16 = (
        w1.dtype == torch.bfloat16
        and w2.dtype == torch.bfloat16
        and quant_type == QuantType.No
    )
    is_fp8 = (
        w1.dtype == dtypes.fp8
        and w2.dtype == dtypes.fp8
        and quant_type in (QuantType.per_Token, QuantType.per_Tensor)
    )
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or activation not in (ActivationType.Silu, ActivationType.Swiglu)
        or not (is_bf16 or is_fp8)
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if is_fp8 and (w1_scale is None or w2_scale is None):
        raise ValueError("FP8 weights require both w1_scale and w2_scale")
    if is_bf16 and (w1_scale is not None or w2_scale is not None):
        raise NotImplementedError("BF16 weights do not support weight scales")

    activation_str = "swiglu" if activation == ActivationType.Swiglu else "silu"
    problem = _Problem.from_inputs(hidden_states, w1, w2, topk_ids, quant_type)
    unsupported_reason = _unsupported_reason(config_string, problem)
    if unsupported_reason is not None:
        raise RuntimeError(
            f"Unsupported gfx942 FlyDSL MoE config {config_string!r}: "
            f"{unsupported_reason}"
        )
    if config.use_prefill:
        return _run_prefill(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant_type,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            config,
            problem,
            activation_str,
            swiglu_limit,
        )
    if problem.batch == 1 or config.use_batch1_algorithm:
        return _run_batch1(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale,
            w2_scale,
            problem,
            activation_str,
            swiglu_limit,
        )
    if 2 <= problem.batch <= 256:
        return _run_decode(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            w1_scale,
            w2_scale,
            expert_mask,
            num_local_tokens,
            moe_sorting_dispatch_policy,
            config,
            problem,
            activation_str,
            swiglu_limit,
        )
    raise RuntimeError(f"Unsupported batch-size {problem.batch}")


def run_flydsl_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
    config = Config.from_string(config_string)
    if not (
        getattr(request.w1, "is_shuffled", False)
        and getattr(request.w2, "is_shuffled", False)
    ):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend requires preshuffled weights"
        )
    if request.bias1 is not None or request.bias2 is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support per-expert bias"
        )
    if request.doweight_stage1:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support doweight_stage1=True"
        )
    if request.a1_scale is not None or request.a2_scale is not None:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support prequantized activations"
        )
    if request.hidden_pad or request.intermediate_pad:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support hidden/intermediate padding"
        )
    if request.gate_mode not in (None, "separated"):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend only supports separated gate weights"
        )
    if request.dtype not in (None, request.hidden_states.dtype):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support output dtype conversion"
        )
    if request.block_size_m not in (None, config.BLOCK_M):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support overriding block_size_m"
        )
    if request.ksplit != 0:
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph backend does not support split-K"
        )
    quant_dtype = torch.bfloat16 if request.w1.dtype == torch.bfloat16 else dtypes.fp8
    if request.q_dtype_a not in (None, quant_dtype):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph activation dtype must match the weight mode"
        )
    if request.q_dtype_w not in (None, quant_dtype):
        raise NotImplementedError(
            "gfx942 FlyDSL whole-graph weight dtype must match the weight mode"
        )
    return run_flydsl_moe_gfx942(
        request.hidden_states,
        request.w1,
        request.w2,
        request.topk_weight,
        request.topk_ids,
        request.activation,
        request.quant_type,
        request.w1_scale,
        request.w2_scale,
        request.expert_mask,
        request.num_local_tokens,
        request.moe_sorting_dispatch_policy,
        config.to_string(),
        request.swiglu_limit,
    )
