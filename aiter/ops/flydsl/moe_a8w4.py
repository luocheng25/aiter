# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 A8W4 MoE (FlyDSL): FP8 per-token activations x MXFP4 weights, BF16 in and out.

``prepare_moe_a8w4_ptpc_weight`` converts standard MXFP4 MoE weights once (at load time) into the private layout
of ``kernels/moe_gemm_2stage/gfx9_fp8/common_a8w4.py``. ``flydsl_moe_a8w4_ptpc`` runs the whole MoE with the
prefill kernels: Opus ``moe_sorting``, per-row FP8 quantization of the input (the same kernel sets ``loc_ids`` to
-1), the A8W4 Gate/Up kernel (SiLU(gate) * up -> f32, optionally with the DeepSeek-V4.1 SwiGLU limit), per-row
FP8 quantization of the intermediate, the A8W4 Down kernel (routing weight applied; it also records each route's
sorted row in ``loc_ids``) and ``sorted_sum``. The calls issue no host synchronization.
"""

import functools
import math
import numbers
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.moe_gemm_2stage.gfx9_fp8 import common_a8w4 as layout
from .kernels.moe_gemm_2stage.gfx9_fp8 import gemm2_a8w4
from .kernels.moe_gemm_2stage.gfx9_fp8.common import _TORCH_TO_FX
from .kernels.moe_gemm_2stage.gfx9_fp8.gemm1_a8w4 import compile_a8w4_gemm1
from .kernels.moe_gemm_2stage.gfx9_fp8.moe_reduce import sorted_sum
from .kernels.moe_gemm_2stage.gfx9_fp8.quant_a8w4 import compile_row_quant_fp8
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

__all__ = [
    "MoeA8W4PtpcWeight",
    "flydsl_moe_a8w4_ptpc",
    "prepare_moe_a8w4_ptpc_weight",
]

SUPPORTED_GFX = ("gfx942",)
_PAD_I = 64
_FP4_DTYPES = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
_E8M0_DTYPES = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
_E8M0_NAN = 255
# The kernels address activations, the intermediate and its quantization with 32-bit byte offsets.
_MAX_BYTES = 1 << 31
# The prefill Down splits the hidden dimension over 2 or 4 workgroups per sorted block while the
# estimated blocks (E * ceil(M * topk / E / block_m), uniform routing) times the split stay below this:
# few blocks (M ~2048) otherwise leave the last round of workgroups mostly idle (MiMo M2048 Down -10%).
# Padded I below 256 (short Down steps) gains nothing from it and stays unsplit.
DOWN_MIN_WGS = 1000
# Sorted blocks per XCD chunk of the prefill Gate/Up / Down from hidden 5120 on with at least XCD_MIN_ROUTES_H5120
# routes (XCD_CHUNK of gemm1_a8w4 / gemm2_a8w4; 0: the 2-D grid): a block's N tiles / hidden parts and neighbouring
# blocks of an expert run on one XCD and share its L2 (DeepSeek TP4 M4096 / 16384 / 32768: 0.991x / 0.982x / 0.961x
# the time; M2048 0.997x; with few blocks, M <= 32, some XCDs get no work: 1.01x).
GATEUP_XCD_CHUNK_H5120 = 1
DOWN_XCD_CHUNK_H5120 = 4
XCD_MIN_ROUTES_H5120 = 16384
# The prefill Down stages a large activation tile in two passes (gemm2_a8w4 TWO_PASS) only for grids of more
# workgroups (estimated blocks times the split) than one per CU.
DOWN_TWO_PASS_MIN_WGS = 80

# Kernels launched by the most recent ``flydsl_moe_a8w4_ptpc`` call, in order (host-side record; tests and the
# benchmark read it).
last_path = []


@dataclass(frozen=True)
class MoeA8W4PtpcWeight:
    """MoE weights in the A8W4 layout; build them with ``prepare_moe_a8w4_ptpc_weight``.

    Flat per expert (see ``common_a8w4``): ``w1`` / ``w2`` uint8 FP4 codes [E, bytes], ``w1_delta`` /
    ``w2_delta`` uint8 [E, bytes], ``w1_scale`` / ``w2_scale`` float32 [E, N] per-channel factors. ``inter`` is
    the checkpoint's intermediate size I, ``inter_padded`` I padded to a multiple of 64.
    """

    w1: torch.Tensor
    w1_delta: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_delta: torch.Tensor
    w2_scale: torch.Tensor
    experts: int
    hidden: int
    inter: int
    inter_padded: int


def _require_supported_gfx():
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_GFX:
        raise RuntimeError(
            f"FlyDSL A8W4 MoE supports {', '.join(SUPPORTED_GFX)} only, got {arch!r}"
        )


def _codes(scale, shape):
    if scale.dtype not in _E8M0_DTYPES:
        raise TypeError(
            f"MXFP4 scales must be float8_e8m0fnu or uint8, got {scale.dtype}"
        )
    if scale.numel() != math.prod(shape):
        raise ValueError(
            f"expected MXFP4 scales of shape {shape}, got {tuple(scale.shape)}"
        )
    codes = scale.view(torch.uint8).reshape(shape)
    if bool((codes == _E8M0_NAN).any()):
        raise ValueError("E8M0 scale code 255 (NaN) is not supported")
    return codes


def _packed(weight, shape):
    if weight.dtype not in _FP4_DTYPES:
        raise TypeError(
            f"MXFP4 weights must be float4_e2m1fn_x2 or uint8, got {weight.dtype}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"expected MXFP4 weight shape {shape}, got {tuple(weight.shape)}"
        )
    return weight.view(torch.uint8)


def _to_fd2_layout(q: torch.Tensor, codes: torch.Tensor):
    """FP4 bytes [n, k/2] + E8M0 codes [n, k/32] -> (q, d, s) of ``layout.py``."""
    n, half_k = q.shape
    k = half_k * 2
    kb = k // 64
    kq = layout.delta_dwords(k)
    codes = codes.to(torch.int32)
    cmax = codes.amax(dim=-1)
    diff = cmax[:, None] - codes
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).view(n, k // 32, 32)
    # -0 is stored as +0; groups more than MAX_DELTA codes below the row maximum flush to 0.
    nib = nib.masked_fill(nib == 8, 0).masked_fill(
        (diff > layout.MAX_DELTA)[..., None], 0
    )
    nib = nib.view(n, k // 8, 8)
    out_q = (nib[..., :4] | (nib[..., 4:] << 4)).view(n // 16, 16, kb, 4, 8)
    out_q = out_q.permute(0, 2, 3, 1, 4).reshape(-1)
    delta = (diff.clamp(max=layout.MAX_DELTA) * 8).to(torch.uint8).view(n, kb, 2)
    delta = F.pad(delta, (0, 0, 0, kq * 4 - kb))
    out_d = delta.view(n // 16, 16, kq, 4, 2).permute(0, 2, 4, 1, 3).reshape(-1)
    scale = torch.ldexp(torch.ones_like(cmax, dtype=torch.float64), cmax - 132)
    return out_q.contiguous(), out_d.contiguous(), scale.float().contiguous()


def _convert(q, codes, chunk=32):
    """[E, N, K/2] FP4 bytes + [E, N, K/32] E8M0 codes -> ``(q, delta, channel scale)`` [E, ...], converted
    ``chunk`` experts at a time (one expert's 16-row blocks follow the previous expert's, so each expert's layout
    is the dense layout of its own weight)."""
    E, _, half_k = q.shape
    parts = [
        _to_fd2_layout(
            q[e : e + chunk].reshape(-1, half_k),
            codes[e : e + chunk].reshape(-1, codes.shape[2]),
        )
        for e in range(0, E, chunk)
    ]
    return tuple(torch.cat(p).view(E, -1) for p in zip(*parts))


def prepare_moe_a8w4_ptpc_weight(w1, w1_scale, w2, w2_scale) -> MoeA8W4PtpcWeight:
    """Convert standard MXFP4 MoE weights to the A8W4 layout (run once, at load time).

    Inputs: ``w1`` [E, 2I, H/2] FP4x2 (``torch.float4_e2m1fn_x2`` or uint8; rows ``[0, I)`` gate,
    ``[I, 2I)`` up; low nibble = even K), ``w1_scale`` [E, 2I, H/32] E8M0 (``float8_e8m0fnu`` or
    uint8), ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32], all on one GPU. Requires ``H % 512 == 0`` and
    ``I % 32 == 0``. I is zero-padded to a multiple of 64 (160 -> 192): padded Gate/Up rows and
    Down columns are zero, so results equal the unpadded MoE.

    Precision: each 32-K group is decoded to FP8 relative to its row's largest E8M0 code; groups
    more than 11 codes (2**11) below that maximum are treated as zero (``common_a8w4.MAX_DELTA``); an E8M0
    code 255 (NaN) is rejected. One result serves every token count of ``flydsl_moe_a8w4_ptpc``.
    Synchronizes with the host once per scale tensor. Raises ``RuntimeError`` off gfx942.
    """
    _require_supported_gfx()
    if w1.dim() != 3:
        raise ValueError(f"w1 must be [E, 2I, H/2], got shape {tuple(w1.shape)}")
    E, n1, half_h = w1.shape
    I, H = n1 // 2, half_h * 2
    if n1 % 2 or H % 512 or I % 32:
        raise ValueError(f"unsupported shape: H={H} (needs % 512), I={I} (needs % 32)")
    devices = {t.device for t in (w1, w1_scale, w2, w2_scale)}
    if len(devices) != 1 or w1.device.type != "cuda":
        raise ValueError(
            f"w1, w1_scale, w2, w2_scale must be on one GPU, got {sorted(map(str, devices))}"
        )
    q1 = _packed(w1, (E, 2 * I, H // 2))
    q2 = _packed(w2, (E, H, I // 2))
    c1 = _codes(w1_scale, (E, 2 * I, H // 32))
    c2 = _codes(w2_scale, (E, H, I // 32))
    ip = -(-I // _PAD_I) * _PAD_I
    if ip != I:
        pad = ip - I
        q1 = torch.cat(
            [
                q1[:, :I],
                q1.new_zeros(E, pad, H // 2),
                q1[:, I:],
                q1.new_zeros(E, pad, H // 2),
            ],
            1,
        )
        c1 = torch.cat(
            [
                c1[:, :I],
                c1.new_full((E, pad, H // 32), 127),
                c1[:, I:],
                c1.new_full((E, pad, H // 32), 127),
            ],
            1,
        )
        q2 = F.pad(q2, (0, pad // 2))
        c2 = F.pad(c2, (0, pad // 32))
    w1q, w1d, w1s = _convert(q1.contiguous(), c1.contiguous())
    w2q, w2d, w2s = _convert(q2.contiguous(), c2.contiguous())
    return MoeA8W4PtpcWeight(w1q, w1d, w1s, w2q, w2d, w2s, E, H, I, ip)


def _trace_constants():
    """Kernel-module constants the prefill launchers read while being built or traced (XCDs per GPU, the K from
    which Down stages its activations in two passes); part of their cache keys."""
    return layout.XCDS, gemm2_a8w4.A_TWO_PASS_MIN_K


@functools.cache
def _gateup_kernel(
    N, K, topk, block_m, block_n, tile_k, experts, limit, xcd_chunk, device, constants
):
    """Prefill Gate/Up launcher. ``device`` (a compiled handle runs on one GPU) and ``constants``
    (``_trace_constants()``) only key the cache."""
    return compile_a8w4_gemm1(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=block_n,
        TILE_K=tile_k,
        E=experts,
        SWIGLU_LIMIT=limit,
        XCD_CHUNK=xcd_chunk,
    )


@functools.cache
def _down_kernel(
    N, K, topk, block_m, experts, nsplit, two_pass, xcd_chunk, device, constants
):
    """Prefill Down launcher. ``device`` and ``constants`` (``_trace_constants()``) only key the cache."""
    return gemm2_a8w4.compile_a8w4_gemm2(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        E=experts,
        NSPLIT=nsplit,
        TWO_PASS=two_pass,
        XCD_CHUNK=xcd_chunk,
    )


@functools.cache
def _row_quant_kernel(cols, dtype, fill, device):
    """Row quantization launcher (``compile_row_quant_fp8``). ``device`` only keys the cache."""
    return compile_row_quant_fp8(cols, dtype, fill)


def _prefill_down(tokens, topk, experts, hidden, inter, block_m, device):
    """Prefill Down kernel of a call: ``_down_split``, DOWN_TWO_PASS_MIN_WGS, DOWN_XCD_CHUNK_H5120."""
    nsplit = _down_split(tokens, topk, experts, hidden, inter, block_m)
    blocks = min(tokens * topk, experts * -(-tokens * topk // (experts * block_m)))
    return _down_kernel(
        hidden,
        inter,
        topk,
        block_m,
        experts,
        nsplit,
        blocks * nsplit > DOWN_TWO_PASS_MIN_WGS,
        _xcd_chunk(DOWN_XCD_CHUNK_H5120, hidden, tokens * topk),
        device,
        _trace_constants(),
    )


def _xcd_chunk(chunk, hidden, routes):
    """XCD_CHUNK of a prefill kernel (see ``GATEUP_XCD_CHUNK_H5120``)."""
    return chunk if hidden >= 5120 and routes >= XCD_MIN_ROUTES_H5120 else 0


def _down_split(tokens, topk, experts, hidden, inter, block_m):
    """Hidden-dimension split of the prefill Down (see ``DOWN_MIN_WGS``)."""
    if inter < 256:
        return 1
    blocks = experts * -(-tokens * topk // (experts * block_m))
    nsplit = 1
    while (
        nsplit < 4
        and blocks * nsplit < DOWN_MIN_WGS
        and (hidden // 64) % (4 * nsplit) == 0
    ):
        nsplit *= 2
    return nsplit


def _quantize(x, rows, cols, dtype, fill=None):
    """Per-row FP8 quantization of ``x`` [rows, cols]; the same kernel sets int32 ``fill`` [rows, k] to -1."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    kernel = _row_quant_kernel(
        cols, dtype, 0 if fill is None else fill.shape[1], x.device.index
    )
    _launch(
        "a8w4_row_quant_fp8", kernel, x, q, scale, scale if fill is None else fill, rows
    )
    return q, scale


def _arg(t):
    # Pointers typed by dtype as the gfx9_fp8 launchers pass them (uint8 and FP8 as Uint8).
    return ptr_arg(t, _TORCH_TO_FX.get(t.dtype))


def _launch(name, kernel, *args):
    """Launch ``kernel`` on the current stream of the first tensor argument's device; record ``name``."""
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    _run_compiled(
        kernel,
        *(_arg(a) if isinstance(a, torch.Tensor) else a for a in args),
        current_stream_arg(device),
    )
    last_path.append(name)


def _swiglu_limit(value):
    """``swiglu_limit`` of ``flydsl_moe_a8w4_ptpc`` as a float, 0.0 when off."""
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"swiglu_limit must be a real number, got {type(value).__name__}"
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"swiglu_limit must be finite and >= 0 (0 disables it), got {value!r}"
        )
    return float(value) if value > 0 else 0.0


def _check_inputs(hidden_states, weight, topk_weight, topk_ids, output):
    if not isinstance(weight, MoeA8W4PtpcWeight):
        raise TypeError(
            f"weight must come from prepare_moe_a8w4_ptpc_weight, got {type(weight).__name__}"
        )
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise TypeError("hidden_states must be a 2-D bfloat16 tensor")
    M, H = hidden_states.shape
    if H != weight.hidden:
        raise ValueError(
            f"hidden size {H} does not match the weights ({weight.hidden})"
        )
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if (
        topk_ids.dim() != 2
        or topk_ids.shape[0] != M
        or tuple(topk_weight.shape) != tuple(topk_ids.shape)
    ):
        raise ValueError(
            f"topk_ids and topk_weight must be [M={M}, topk], got {tuple(topk_ids.shape)} and "
            f"{tuple(topk_weight.shape)}"
        )
    if (
        topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_weight.is_floating_point()
    ):
        raise TypeError(
            f"topk_ids must be int32 / int64 and topk_weight floating, got {topk_ids.dtype}, "
            f"{topk_weight.dtype}"
        )
    if output is not None:
        if output.dtype != torch.bfloat16 or tuple(output.shape) != (M, H):
            raise TypeError(
                f"output must be a bfloat16 [{M}, {H}] tensor, got {output.dtype} {tuple(output.shape)}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
    tensors = [hidden_states, topk_weight, topk_ids, weight.w1, weight.w2] + (
        [] if output is None else [output]
    )
    devices = {t.device for t in tensors}
    if len(devices) != 1 or hidden_states.device.type != "cuda":
        raise ValueError(
            f"inputs, output and weight must be on one GPU, got {sorted(map(str, devices))}"
        )
    topk = topk_ids.shape[1]
    if max(M * topk * weight.inter_padded * 4, M * H * 2) >= _MAX_BYTES:
        raise ValueError(
            f"M={M} is too large: M * topk * I_padded * 4 and M * H * 2 must be below 2**31 (32-bit byte offsets)"
        )


def flydsl_moe_a8w4_ptpc(
    hidden_states,
    weight,
    topk_weight,
    topk_ids,
    *,
    output=None,
    swiglu_limit=0.0,
    block_m=None,
    block_n=None,
    tile_k=None,
):
    """A8W4 MoE forward: BF16 ``hidden_states`` [M, H] (contiguous) -> BF16 [M, H].

    ``weight`` comes from ``prepare_moe_a8w4_ptpc_weight``; ``topk_weight`` [M, topk] (float32) is applied
    in Down; ``topk_ids`` [M, topk] (int32) in ``[0, E)`` (a fused shared expert is an ordinary
    expert id), distinct within a token (with a repeated id the Opus sort keeps one route); int64 ids,
    other floating weights and non-contiguous routing tensors are converted first. The input
    and the f32 intermediate SiLU(gate) * up are quantized per row to FP8
    E4M3FNUZ with ``scale = amax / 240`` and ``q = RNE(x / scale)``. SiLU only, no expert mask /
    bias; ``M * topk * I_padded * 4`` and ``M * H * 2`` below ``2**31``. ``output`` may be a caller's contiguous
    BF16 [M, H] tensor, ``hidden_states`` itself included. Issues no host synchronization; compile a shape
    outside a CUDA graph capture before capturing it. Raises ``RuntimeError`` off gfx942.

    ``block_m`` / ``block_n`` / ``tile_k`` select the tiles: ``block_m`` (default 64) is the sorting block and
    M tile of both kernels (``block_m * I_padded % 4096 == 0``, e.g. 64 or 128); ``block_n`` (default 128) /
    ``tile_k`` are the Gate/Up N tile (128 or 256, dividing 2 * padded I into whole gate/up halves) and K tile
    (128 or 256; default 128 for M > 2048, where its lower LDS and register use raise occupancy, else 256).

    ``swiglu_limit`` is the SwiGLU limit of DeepSeek-V4.1 (config ``swiglu_limit``, 10 for V4.1-Flash; the
    ``Expert`` of its inference/model.py): when > 0 the Gate/Up kernel computes
    ``SiLU(gate.clamp(max=L)) * up.clamp(-L, L)`` on the dequantized f32 gate / up values (after the weight
    and activation scales). 0 (default) turns it off and runs exactly the kernels without it; otherwise
    the limit is a compile-time constant (kernels compiled per value, names ending in ``_swiglu``). Must
    be finite and >= 0 (``ValueError``).
    """
    from aiter.fused_moe import moe_sorting

    _require_supported_gfx()
    _check_inputs(hidden_states, weight, topk_weight, topk_ids, output)
    limit = _swiglu_limit(swiglu_limit)
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    E, I = weight.experts, weight.inter_padded
    device = hidden_states.device
    if output is None:
        output = torch.empty_like(hidden_states)
    last_path.clear()
    if M == 0:
        return output
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
    block_m = 64 if block_m is None else block_m
    block_n = 128 if block_n is None else block_n
    tile_k = (128 if M > 2048 else 256) if tile_k is None else tile_k

    sorted_ids, sorted_weights, sorted_experts, num_valid = moe_sorting(
        topk_ids, topk_weight, E, H, hidden_states.dtype, block_m, accumulate=False
    )[:4]
    last_path.append("moe_sorting_opus")
    # The sort's inverse output (output_aux) costs more than Down storing it: the input quantization
    # sets loc to -1 (a repeated expert id of a token keeps one route), Down stores each route's sorted row.
    loc_ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    xq, xs = _quantize(hidden_states, M, H, "bf16", fill=loc_ids)
    grid = sorted_experts.numel()
    routing = (sorted_ids, sorted_weights, sorted_experts, num_valid)

    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    gateup = _gateup_kernel(
        2 * I,
        H,
        topk,
        block_m,
        block_n,
        tile_k,
        E,
        limit,
        _xcd_chunk(GATEUP_XCD_CHUNK_H5120, H, M * topk),
        device.index,
        _trace_constants(),
    )
    _launch(
        f"a8w4_gateup_prefill_bm{block_m}_bn{block_n}_tk{tile_k}",
        gateup,
        xq,
        weight.w1,
        weight.w1_delta,
        mid,
        *routing,
        weight.w1_scale,
        xs,
        M,
        grid,
    )
    mq, ms = _quantize(mid, M * topk, I, "f32")
    routes = torch.empty((grid * block_m, H), dtype=torch.bfloat16, device=device)
    down = _prefill_down(M, topk, E, H, I, block_m, device.index)
    _launch(
        f"a8w4_down_prefill_bm{block_m}",
        down,
        mq,
        weight.w2,
        weight.w2_delta,
        routes,
        *routing,
        weight.w2_scale,
        ms,
        loc_ids,
        M,
        grid,
    )
    sorted_sum(topk, H)(loc_ids, routes, output, M)
    last_path.append("sorted_sum")
    return output
