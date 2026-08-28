# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
from functools import cache
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    invert_sorted_ids,
    sorted_sum,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.moe_common import GateMode


@dataclass
class Config:
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    use_prefill: bool
    use_batch1_algorithm: bool = False

    def to_string(self):
        return (
            str(self.BLOCK_M)
            + "_"
            + str(self.BLOCK_N)
            + "_"
            + str(self.BLOCK_K)
            + "_"
            + str(self.use_prefill)
            + "_"
            + str(self.use_batch1_algorithm)
        )

    @classmethod
    def from_string(cls, data: str):
        parts = data.split("_")
        if len(parts) not in (4, 5):
            raise ValueError(f"Invalid config string: {data}")

        def parse_bool(value: str) -> bool:
            if value == "True":
                return True
            if value == "False":
                return False
            raise ValueError(f"Invalid boolean value in config string: {value}")

        return cls(
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            parse_bool(parts[3]),
            parse_bool(parts[4]) if len(parts) == 5 else False,
        )


def _uses_batch1_path(config: Config, batch: int) -> bool:
    return not config.use_prefill and (
        batch == 1 or (config.use_batch1_algorithm and 2 <= batch <= 8)
    )


def _supports_mxfp4_activation_request(
    q_dtype_a: torch.dtype,
    activation: Any,
    batch: int,
    config: Config,
) -> bool:
    # This backend always computes BF16-A x MXFP4-W. An FP4 q_dtype_a is only a
    # dispatch label allowing A4W4 SiTUv2 to select the small-batch fast path.
    return q_dtype_a == torch.bfloat16 or (
        q_dtype_a == torch.float4_e2m1fn_x2
        and activation == ActivationType.Situv2
        and _uses_batch1_path(config, batch)
    )


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
        if w1.dtype == torch.float4_e2m1fn_x2:
            hidden_dim *= 2
            inter_dim *= 2
        assert gateup_dim == 2 * inter_dim
        if quant_type == QuantType.per_1x32:
            quant_type_str = "mxfp4"
        elif quant_type == QuantType.per_Token:
            quant_type_str = "ptpc"
        else:
            quant_type_str = "per_tensor"
        return cls(
            batch=int(hidden_states.shape[0]),
            experts=experts,
            gateup_dim=gateup_dim,
            hidden_dim=hidden_dim,
            model_dim=model_dim,
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            quant_type=quant_type_str,
        )


def get_tune_space(batch: int | None = None, *, include_prefill: bool = True):
    configs = [
        # Decode split-K ignores BLOCK_N/BLOCK_K. Small batches also tune the
        # direct route-wise algorithm normally used for batch 1.
        Config(16, 16, 16, False, False).to_string(),
    ]
    if batch is not None and 2 <= batch <= 8:
        configs.insert(1, Config(16, 16, 16, False, True).to_string())
    if include_prefill:
        configs.extend(
            [
                Config(64, 256, 128, True, False).to_string(),
                Config(64, 128, 256, True, False).to_string(),
                Config(64, 128, 128, True, False).to_string(),
            ]
        )
    return configs


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
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
):
    """Cache-compiled flydsl kernel via compile_gemm."""
    from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import compile_gemm

    if weight_dtype_str != "fp4":
        mxfp4_gate_up_interleaved = False
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
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
    )


_TORCH_TO_FX = {
    torch.uint8: fx.Uint8,
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
    torch.float8_e8m0fnu: fx.Uint8,
    torch.float4_e2m1fn_x2: fx.Uint8,
}


def _ptr(t):
    return flyc.from_c_void_p(_TORCH_TO_FX[t.dtype], t.data_ptr())


def _launch(kernel_fn, *args):
    """Launch a FlyDSL JIT kernel on the current stream."""
    stream = torch.cuda.current_stream()
    prepared_args = [
        _ptr(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    ]
    _run_compiled(kernel_fn, *prepared_args, stream)


def _quant_per_tensor(x, scale=None, quant_dtype=torch.float8_e4m3fn, num_rows=None):
    assert scale is None
    assert num_rows is None

    amax = torch.empty(1, dtype=torch.float32, device=x.device)
    xq = torch.empty_like(x, dtype=quant_dtype)
    flydsl_absmax()(x, amax)
    flydsl_quant_per_tensor(quant_dtype)(x, amax, xq)
    fmax = torch.finfo(quant_dtype).max
    xs = amax / fmax
    xs = xs.reshape(1).to(torch.float32)

    return xq, xs


def _empty_scale(device):
    return torch.empty(0, device=device)


def _activation_scalars(
    activation: str,
    situ_beta: float,
    situ_linear_beta: float,
    swiglu_limit: float | None,
):
    if activation == "situv2":
        beta = float(situ_beta)
        linear_beta = float(situ_linear_beta)
        if beta <= 0.0 or linear_beta <= 0.0:
            raise ValueError(
                "situ_beta and situ_linear_beta must be positive, "
                f"got {beta}/{linear_beta}"
            )
    else:
        beta = linear_beta = 1.0
    if activation == "swiglu":
        limit = float(swiglu_limit) if swiglu_limit else 7.0
    else:
        limit = float(swiglu_limit) if swiglu_limit else float("inf")
    return beta, 1.0 / beta, linear_beta, 1.0 / linear_beta, limit


def _gateup_output(hidden_states: torch.Tensor, problem: _Problem):
    return torch.empty(
        [problem.batch, problem.topk, problem.inter_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


def precompile_flydsl_moe(
    *,
    config_string: str,
    batch: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    weight_dtype: str,
    quant_type: str,
    activation: str,
):
    """Compile the whole-graph launchers selected by a tuned config."""
    config = Config.from_string(config_string)
    is_mxfp4 = weight_dtype == "fp4"
    if is_mxfp4 and config.use_prefill:
        raise ValueError("MXFP4 does not support the prefill algorithm")

    device = torch.device("cpu")
    bf16 = torch.zeros(1, dtype=torch.bfloat16, device=device)
    byte = torch.zeros(1, dtype=torch.uint8, device=device)
    int32 = torch.zeros(1, dtype=torch.int32, device=device)
    float32 = torch.zeros(1, dtype=torch.float32, device=device)
    weight_scale = byte if is_mxfp4 else float32
    activation_scalars = _activation_scalars(activation, 1.0, 1.0, None)

    def compile_kernel(*, stage, alg, block_m, block_n, **kwargs):
        return _get_compiled_kernel(
            N=(2 * inter_dim if stage == "gateup" else model_dim),
            K=(model_dim if stage == "gateup" else inter_dim),
            weight_dtype_str=weight_dtype,
            quant_type_str=quant_type,
            TOPK=topk,
            BLOCK_TILE_SIZE_M=block_m,
            BLOCK_TILE_SIZE_N=block_n,
            stage=stage,
            alg=alg,
            E=(None if alg == "batch1" else experts),
            activation_str=(activation if stage == "gateup" else "silu"),
            **kwargs,
        )

    if config.use_prefill:
        gateup = compile_kernel(
            stage="gateup",
            alg="prefill_1x4",
            block_m=config.BLOCK_M,
            block_n=config.BLOCK_N,
            act_quant_type_str="ptpc",
        )
        _run_compiled(
            gateup,
            _ptr(byte),
            _ptr(byte),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            _ptr(float32),
            batch,
            1,
            *activation_scalars,
            0,
        )
        down = compile_kernel(
            stage="down",
            alg="prefill_1x4",
            block_m=config.BLOCK_M,
            block_n=128,
        )
        _run_compiled(
            down,
            _ptr(byte),
            _ptr(byte),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            _ptr(float32),
            batch,
            1,
            *activation_scalars,
            0,
        )
        return

    use_batch1_algorithm = batch == 1 or (
        config.use_batch1_algorithm and 2 <= batch <= 8
    )
    if config.use_batch1_algorithm and not 2 <= batch <= 8:
        raise ValueError(f"The batch-1 algorithm is not valid for tuned batch {batch}")

    if use_batch1_algorithm:
        force_batch1_path = batch > 1
        fused_down_clear = is_mxfp4 and force_batch1_path
        gate_layouts = (False, True) if is_mxfp4 else (True,)
        gate_block_ns = (
            (32, 64)
            if is_mxfp4 and batch == 4
            else (64 if is_mxfp4 and batch >= 4 else 32,)
        )
        for block_n in gate_block_ns:
            for gate_up_interleaved in gate_layouts:
                gateup = compile_kernel(
                    stage="gateup",
                    alg="batch1",
                    block_m=16,
                    block_n=block_n,
                    mxfp4_gate_up_interleaved=gate_up_interleaved,
                    fused_down_clear=fused_down_clear,
                )
                _run_compiled(
                    gateup,
                    _ptr(bf16),
                    _ptr(byte),
                    _ptr(bf16),
                    _ptr(int32),
                    _ptr(bf16 if fused_down_clear else float32),
                    _ptr(weight_scale),
                    batch,
                    *activation_scalars,
                    0,
                )
        down = compile_kernel(
            stage="down",
            alg="batch1",
            block_m=16,
            block_n=(32 if is_mxfp4 and force_batch1_path else 64),
        )
        _run_compiled(
            down,
            _ptr(bf16),
            _ptr(byte),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(weight_scale),
            batch,
            *activation_scalars,
            0,
        )
        return

    gate_layouts = (False, True) if is_mxfp4 else (True,)
    for gate_up_interleaved in gate_layouts:
        gateup = compile_kernel(
            stage="gateup",
            alg="splitk",
            block_m=config.BLOCK_M,
            block_n=64,
            mxfp4_gate_up_interleaved=gate_up_interleaved,
        )
        _run_compiled(
            gateup,
            _ptr(bf16),
            _ptr(byte),
            _ptr(bf16),
            _ptr(int32),
            _ptr(float32),
            _ptr(int32),
            _ptr(int32),
            _ptr(weight_scale),
            batch,
            1,
            *activation_scalars,
            0,
        )
    down = compile_kernel(
        stage="down",
        alg="splitk",
        block_m=config.BLOCK_M,
        block_n=64,
    )
    _run_compiled(
        down,
        _ptr(bf16),
        _ptr(byte),
        _ptr(bf16),
        _ptr(int32),
        _ptr(float32),
        _ptr(int32),
        _ptr(int32),
        _ptr(weight_scale),
        batch,
        1,
        *activation_scalars,
        0,
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
    config: Config,
    problem: _Problem,
    activation_str: str,
    swiglu_limit: float | None,
    situ_beta: float,
    situ_linear_beta: float,
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.model_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        None,
        None,
        0,
    )
    weight_dtype_str = "bf16" if w1.dtype == torch.bfloat16 else "fp8"
    act_quant_type_str = "ptpc"
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
        if quant_type == QuantType.per_Tensor:
            a_scale = a_scale.repeat(problem.batch, 1).contiguous()
        a_scale = a_scale.to(torch.float32).contiguous()
    else:
        gateup_in = hidden_states
        a_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    gemm1_out = _gateup_output(hidden_states, problem)
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=config.BLOCK_N,
        BLOCK_TILE_SIZE_K=None,
        stage="gateup",
        alg="prefill_1x4",
        E=problem.experts,
        act_quant_type_str=act_quant_type_str,
        activation_str=activation_str,
    )
    task_num = int(sorted_expert_ids.shape[0])
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
        problem.batch,
        task_num,
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
    )

    if weight_dtype_str == "fp8":
        down_in, down_in_scale = quant_func(
            gemm1_out.view(problem.batch * problem.topk, -1),
            scale=None,
            quant_dtype=w2.dtype,
            num_rows=None,
        )
    else:
        down_in = gemm1_out
        down_in_scale = torch.empty(1, dtype=torch.float32, device=hidden_states.device)

    gemm2_out = torch.empty(
        [sorted_expert_ids.shape[0] * config.BLOCK_M, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=128,
        stage="down",
        alg="prefill_1x4",
        E=problem.experts,
    )
    _launch(
        down_kernel,
        down_in,
        w2,
        gemm2_out,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        w2_scale if w2_scale is not None else _empty_scale(hidden_states.device),
        down_in_scale,
        problem.batch,
        task_num,
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
    )

    loc_ids = torch.empty(
        [problem.batch, problem.topk],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    invert_sorted_ids(problem.topk)(
        sorted_ids,
        loc_ids,
        num_valid_ids,
        sorted_ids.shape[0],
        problem.batch,
    )
    sorted_sum(problem.topk, problem.model_dim)(
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
    situ_beta: float,
    situ_linear_beta: float,
    mxfp4_gate_up_interleaved: bool,
):
    topk_weight = (
        topk_weight if topk_weight.dtype == torch.float32 else topk_weight.float()
    )
    gemm1_out = _gateup_output(hidden_states, problem)
    is_mxfp4 = w1.dtype == torch.float4_e2m1fn_x2
    force_batch1_path = problem.batch > 1
    fused_down_clear = is_mxfp4 and force_batch1_path
    output_factory = torch.empty if fused_down_clear else torch.zeros
    cur_out = output_factory(
        [problem.batch, problem.model_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str="fp4" if is_mxfp4 else "fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=64 if is_mxfp4 and problem.batch >= 4 else 32,
        stage="gateup",
        alg="batch1",
        E=None,
        activation_str=activation_str,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
    )
    _launch(
        gateup_kernel,
        hidden_states,
        w1,
        gemm1_out,
        topk_ids,
        cur_out if fused_down_clear else topk_weight,
        w1_scale if w1_scale is not None else _empty_scale(hidden_states.device),
        problem.batch,
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str="fp4" if is_mxfp4 else "fp8",
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=16,
        BLOCK_TILE_SIZE_N=32 if is_mxfp4 and force_batch1_path else 64,
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
        problem.batch,
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
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
    situ_beta: float,
    situ_linear_beta: float,
    mxfp4_gate_up_interleaved: bool,
):
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, cur_out = moe_sorting(
        topk_ids,
        topk_weight,
        problem.experts,
        problem.hidden_dim,
        hidden_states.dtype,
        config.BLOCK_M,
        expert_mask,
        num_local_tokens,
        moe_sorting_dispatch_policy,
    )
    grid = int(sorted_expert_ids.shape[0])
    if problem.batch * problem.topk <= problem.experts:
        grid = problem.batch * problem.topk

    gemm1_out = _gateup_output(hidden_states, problem)
    weight_dtype_str = "fp4" if w1.dtype == torch.float4_e2m1fn_x2 else "fp8"
    gateup_kernel = _get_compiled_kernel(
        N=problem.gateup_dim,
        K=problem.hidden_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=64,
        stage="gateup",
        alg="splitk",
        E=problem.experts,
        activation_str=activation_str,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
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
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
    )

    down_kernel = _get_compiled_kernel(
        N=problem.model_dim,
        K=problem.inter_dim,
        weight_dtype_str=weight_dtype_str,
        quant_type_str=problem.quant_type,
        TOPK=problem.topk,
        BLOCK_TILE_SIZE_M=config.BLOCK_M,
        BLOCK_TILE_SIZE_N=64,
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
        *_activation_scalars(activation_str, situ_beta, situ_linear_beta, swiglu_limit),
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
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    gate_mode: GateMode | str = GateMode.SEPARATED,
) -> torch.Tensor:
    config = Config.from_string(config_string)
    gate_mode = GateMode(gate_mode)
    is_fp8 = (
        w1.dtype == torch.float8_e4m3fnuz
        and w2.dtype == torch.float8_e4m3fnuz
        and quant_type in (QuantType.per_Token, QuantType.per_Tensor)
        and activation in (ActivationType.Silu, ActivationType.Swiglu)
    )
    is_mxfp4 = (
        w1.dtype == torch.float4_e2m1fn_x2
        and w2.dtype == torch.float4_e2m1fn_x2
        and quant_type == QuantType.per_1x32
        and activation
        in (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)
        and get_gfx() == "gfx950"
    )
    if (
        hidden_states.dtype != torch.bfloat16
        or expert_mask is not None
        or not (is_fp8 or is_mxfp4)
    ):
        raise RuntimeError("Unsupported input for the gfx942 FlyDSL MoE backend")
    if w1_scale is None or w2_scale is None:
        raise ValueError("Quantized weights require both w1_scale and w2_scale")
    if config.use_prefill and is_mxfp4:
        raise RuntimeError("MXFP4 does not support the prefill algorithm")
    problem_batch = int(hidden_states.shape[0])
    if config.use_batch1_algorithm and not 2 <= problem_batch <= 8:
        raise RuntimeError(
            "The direct algorithm requires an actual batch from 2 through 8, "
            f"got {problem_batch}"
        )

    activation_str = (
        "situv2"
        if activation == ActivationType.Situv2
        else "swiglu" if activation == ActivationType.Swiglu else "silu"
    )
    problem = _Problem.from_inputs(hidden_states, w1, w2, topk_ids, quant_type)
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
            config,
            problem,
            activation_str,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
        )
    if _uses_batch1_path(config, problem.batch):
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
            situ_beta,
            situ_linear_beta,
            gate_mode == GateMode.INTERLEAVE,
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
            situ_beta,
            situ_linear_beta,
            gate_mode == GateMode.INTERLEAVE,
        )
    raise RuntimeError(f"Unsupported batch-size {problem.batch}")


def run_flydsl_moe_gfx942_impl(
    request: FusedMoeRequest,
    config_string: str,
) -> torch.Tensor:
    if (
        request.doweight_stage1
        or request.bias1 is not None
        or request.bias2 is not None
    ):
        raise RuntimeError(
            "The FlyDSL whole-graph backend does not support bias or doweight_stage1"
        )
    if request.hidden_pad or request.intermediate_pad:
        raise RuntimeError(
            "The FlyDSL whole-graph backend does not support padded dimensions"
        )
    config = Config.from_string(config_string)
    if (
        request.q_dtype_w == torch.float4_e2m1fn_x2
        and not _supports_mxfp4_activation_request(
            request.q_dtype_a,
            request.activation,
            int(request.hidden_states.shape[0]),
            config,
        )
    ):
        raise RuntimeError(
            "The MXFP4 whole-graph backend requires BF16 activations; "
            "FP4-tagged requests are supported only by the SiTUv2 Batch1 path"
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
        config_string,
        request.swiglu_limit,
        1.0 if request.beta is None else float(request.beta),
        1.0 if request.linear_beta is None else float(request.linear_beta),
        request.gate_mode,
    )
