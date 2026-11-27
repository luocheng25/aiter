# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 MoE for gfx942: FP8 per-token activations x MXFP4 weights on FlyDSL two-stage kernels.

``prepare_a8w4_weights`` converts standard MXFP4 weights once (at load time) into the private layout of
``kernels/moe_a8w4/common.py``; ``fused_moe_a8w4`` runs one rank's MoE (no communication). Three paths by the
token count M (kernels in ``kernels/moe_a8w4/``):

* decode, M <= ``DECODE_MAX_M`` (``decode.py``, no sorting): Gate/Up per route, Down per token summing its
  routes in the workgroup and writing BF16; activations quantized inside the kernels or, with more tokens, by a
  row kernel. Hidden sizes in ``DECODE_WIDE_HIDDEN`` deduplicate the Gate/Up by expert from
  ``DECODE_WIDE_DEDUP_MIN_M`` tokens on (``decode_dedup.py``).
* stream, larger M with at most ``_stream_max_rows`` mean rows per expert (M * topk / E; ``stream.py``): one prep
  launch sorts the routes and quantizes the input, the weight-streaming Gate/Up (quantizing the intermediate
  itself for blocks of 32 / 64 at hidden < 4096) and Down (BF16 atomics into the output for padded I < 512,
  else rows summed by ``route_sum``).
* prefill, the rest: sorting (the stream prep, or Opus ``moe_sorting`` and the row kernel of ``quant.py``), the
  Gate/Up (``gemm1.py``: SiLU(gate) * up -> f32, optionally with the SwiGLU limit), per-row FP8 quantization of
  the intermediate, the Down (``gemm2.py``: routing weight applied, records each route's sorted row) and
  ``sorted_sum`` (``moe_gemm_2stage/gfx9_fp8``).

The rules below were measured on MI308X for the shapes ``MegaMoeTPGfx942`` serves (Qwen3-Next-80B TP2 / TP4:
hidden 2560, E 513, topk 11, padded I 320 / 192; DeepSeek-V4.1-Flash TP4: hidden 5120, E 384, topk 6, I 576).
"""

import functools
import math
import numbers
from dataclasses import dataclass

import flydsl.expr as fx
import torch

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_a8w4 import chunks as a8w4_chunks
from aiter.ops.flydsl.kernels.moe_a8w4 import common as a8w4_common
from aiter.ops.flydsl.kernels.moe_a8w4 import decode as a8w4_decode
from aiter.ops.flydsl.kernels.moe_a8w4.decode_dedup import (
    compile_a8w4_gateup_decode_dd,
    compile_a8w4_row_quant_plan,
)
from aiter.ops.flydsl.kernels.moe_a8w4.gemm1 import compile_a8w4_gemm1
from aiter.ops.flydsl.kernels.moe_a8w4.gemm2 import compile_a8w4_gemm2
from aiter.ops.flydsl.kernels.moe_a8w4.quant import compile_row_quant_fp8
from aiter.ops.flydsl.kernels.moe_a8w4.stream import (
    compile_a8w4_down_stream,
    compile_a8w4_gateup_stream,
    compile_route_sum,
    compile_stream_prep,
    down_stream_lds,
    stream_prep_sizes,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import sorted_sum
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _TORCH_TO_FX
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
)

__all__ = ["A8W4Weights", "fused_moe_a8w4", "last_path", "prepare_a8w4_weights"]

MAX_DELTA = 11
_PAD_I = 64
# Token counts up to this use the decode kernels (no sorting, one workgroup per route / token).
DECODE_MAX_M = 32
# The decode path quantizes activations in separate row kernels instead of inside every workgroup:
# Down's intermediate rows from DECODE_PREQUANT_MIN_M tokens on, Gate/Up's token rows from
# DECODE_GATEUP_PREQUANT_MIN_WGS Gate/Up workgroups (M * topk * I / 16) on.
DECODE_PREQUANT_MIN_M = 4
DECODE_GATEUP_PREQUANT_MIN_WGS = 2560
DECODE_DOWN_PQ_WAVES = 3
# Hidden sizes above 4096 that take the decode kernels (the Gate/Up decode then stages the FP8 token row in two
# rounds of 16-byte pieces); measured: DeepSeek-V4.1 TP4 (H5120). Other hidden sizes above 4096 take prefill.
DECODE_WIDE_HIDDEN = (5120,)
# From DECODE_WIDE_DEDUP_MIN_M tokens on, those hidden sizes deduplicate the decode Gate/Up by expert
# (``decode_dedup``, DECODE_WIDE_DEDUP_WAVES waves) and keep the per-token slot-pair Down. Measured at H5120 E384
# topk 6 against the per-route kernels: M4 / 8 / 16 / 32 1.03x / 0.985x / 0.967x / 0.95x the time (8 waves 7-11%
# slower).
DECODE_WIDE_DEDUP_MIN_M = 8
DECODE_WIDE_DEDUP_WAVES = 4
# Routes per dedup group: 2 up to DECODE_DEDUP_PAIR_MAX_M tokens (fewer staged rows; an expert rarely has three
# routes), DECODE_DEDUP_GROUP beyond.
DECODE_DEDUP_GROUP = 3
DECODE_DEDUP_PAIR_MAX_M = 16
# Above DECODE_MAX_M tokens the weight-streaming kernels run while the mean rows per expert
# (M * topk / E) stay at or below this, the prefill kernels beyond: measured crossover
# between 22.0 and 24.7 rows per expert for TP2 and TP4 alike (E513, topk 11: M = 1024 / 1152).
STREAM_MAX_ROWS = 22
# The same with hidden >= 4096: measured in one process against the prefill kernels, 48 rows (M1536) 0.94x /
# 1.00x the time, 64 rows (M2048) 1.04x / 1.03x.
STREAM_MAX_ROWS_WIDE = 48
# The same for hidden 2560 to 4095 (measured at 2560) when the Gate/Up of blocks of 64 quantizes the intermediate
# itself (``_stream_fused_quant``, Qwen TP4 I=192): 33 / 44 / 55 rows (M1536 / 2048 / 2560) 0.96x / 0.92x / 0.98x.
STREAM_MAX_ROWS_FUSED_QUANT = 56
# Most mean rows per expert sorted in blocks of 32 (more: blocks of 64).
STREAM_BM32_ROWS = 28
# 16-row weight blocks the streaming Down keeps in flight.
STREAM_DOWN_PREFETCH = 4
# 64-K steps per workgroup barrier of the light-wave streaming Gate/Up's LDS activation ring (2: about 1% faster;
# the fused-quantization tile keeps 1, Qwen TP4 M1024 +1.5% with 2).
STREAM_GATEUP_SYNC = 2
# 64-K steps of weights the light-wave streaming Gate/Up keeps in flight (5: VGPRs 85 -> 79 at blocks of 32, one
# more 8-wave workgroup per CU; about 1-2% faster).
STREAM_GATEUP_PREFETCH = 5
# Cache policy bits of the streaming kernels' weight loads (0 cached, 2 non-temporal): non-temporal weights make
# the Gate/Up 10-24% faster, the Down no faster below hidden 5120.
STREAM_GATEUP_W_AUX = 2
STREAM_DOWN_W_AUX = 0
# The prefill Down splits the hidden dimension over 2 or 4 workgroups per sorted block while the
# estimated blocks (E * ceil(M * topk / E / block_m), uniform routing) times the split stay below this:
# few blocks (M ~2048) otherwise leave the last round of workgroups mostly idle (Down -10%).
# Padded I below 256 (short Down steps) gains nothing from it and stays unsplit.
DOWN_MIN_WGS = 1000
# The prefill kernels take the streaming path's prep (one launch: sorting, input quantization, loc_ids) instead of
# Opus moe_sorting and the row kernel while M * topk stays at or below this (hidden 2560 and above, H % 512 == 0).
# Each sorting workgroup histograms every route, so the prep grows with the square of the routes: Qwen TP2 M2048
# (22528 routes) 0.979x, M8192 1.02x the time.
PREFILL_PREP_MAX_ROUTES = 24576
# DeepSeek-V4.1-Flash TP4 (hidden 5120, E384, topk 6, I576, swiglu_limit 10), measured in one process against the
# rules above; they apply from hidden 5120 on:
# - the streaming kernels run up to STREAM_MAX_ROWS_H5120 mean rows per expert once M * topk reaches
#   STREAM_H5120_MIN_ROUTES (prefill vs streaming: 12 / 16 / 20 / 24 rows, M768 / 1024 / 1280 / 1536,
#   1.07x / 0.995x / 0.97x / 0.92x the time); fewer routes keep STREAM_MAX_ROWS_WIDE;
# - the streaming path sorts into blocks of 16 at any route count up to 12 rows per expert (``_stream_block_m``);
# - the prefill kernels take the streaming prep up to PREFILL_PREP_MAX_ROUTES_H5120 routes (Opus and the row kernel
#   vs the prep at 7680 / 12288 / 24576 routes, M1280 / 2048 / 4096: 1.003x / 0.998x / 0.988x the time);
# - the streaming Down's weight loads are non-temporal (M48-1536 2-4% faster).
STREAM_MAX_ROWS_H5120 = 16
STREAM_H5120_MIN_ROUTES = 4096
PREFILL_PREP_MAX_ROUTES_H5120 = 12288
STREAM_DOWN_W_AUX_H5120 = 2
# Activation bytes per lane and load of the light-wave streaming Gate/Up from hidden 5120 on (16: one load every 4 /
# 2 steps with blocks of 16 / 32 instead of 4 / 8 bytes every step).
STREAM_GATEUP_ACT_BYTES_H5120 = 16
# Sorted blocks per XCD chunk of the prefill Gate/Up / Down from hidden 5120 on with at least XCD_MIN_ROUTES_H5120
# routes (XCD_CHUNK of gemm1 / gemm2; 0: the 2-D grid): a block's N tiles / hidden parts and neighbouring blocks of
# an expert run on one XCD and share its L2 (M4096 / 16384 / 32768: 0.991x / 0.982x / 0.961x the time).
GATEUP_XCD_CHUNK_H5120 = 1
DOWN_XCD_CHUNK_H5120 = 4
XCD_MIN_ROUTES_H5120 = 16384
# Prefill tiles: sorting block / M tile of both kernels, Gate/Up N tile and K tile.
PREFILL_BLOCK_M = 64
PREFILL_BLOCK_N = 128
PREFILL_TILE_K = 128

# Kernels launched by the most recent ``fused_moe_a8w4`` call, in order (names start with ``a8w4_``: FP8
# activations); tests check the dispatch with it.
last_path = []


@dataclass(frozen=True)
class A8W4Weights:
    """Prepared weights; tensors are flat uint8 / float32 per expert (see ``kernels/moe_a8w4/common.py``)."""

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


def _codes(scale, shape):
    if scale.dtype not in (torch.float8_e8m0fnu, torch.uint8):
        raise TypeError(
            f"MXFP4 scales must be float8_e8m0fnu or uint8, got {scale.dtype}"
        )
    codes = scale.view(torch.uint8).reshape(shape)
    if bool((codes == 255).any()):
        raise ValueError("E8M0 scale 0xFF (NaN) is not supported")
    return codes


def _packed(weight, shape):
    if weight.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise TypeError(
            f"MXFP4 weights must be float4_e2m1fn_x2 or uint8, got {weight.dtype}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"expected MXFP4 weight shape {shape}, got {tuple(weight.shape)}"
        )
    return weight.view(torch.uint8)


def _to_layout(q, codes):
    """[E, N, K/2] FP4 bytes + [E, N, K/32] E8M0 codes -> (q, delta, channel scale) layouts."""
    E, N, half_k = q.shape
    K = half_k * 2
    KB = K // 64
    KQ = (KB + 3) // 4
    codes = codes.to(torch.int32)
    cmax = codes.amax(dim=-1)
    diff = cmax[..., None] - codes
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).view(E, N, K // 32, 32)
    nib = nib.masked_fill(nib == 8, 0).masked_fill((diff > MAX_DELTA)[..., None], 0)
    nib = nib.view(E, N, K // 8, 8)
    out_q = (nib[..., :4] | (nib[..., 4:] << 4)).view(E, N // 16, 16, KB, 4, 8)
    out_q = out_q.permute(0, 1, 3, 4, 2, 5).reshape(E, -1)
    delta = (diff.clamp(max=MAX_DELTA) * 8).to(torch.uint8).view(E, N, KB, 2)
    delta = torch.nn.functional.pad(delta, (0, 0, 0, KQ * 4 - KB))
    out_d = (
        delta.view(E, N // 16, 16, KQ, 4, 2).permute(0, 1, 3, 5, 2, 4).reshape(E, -1)
    )
    scale = torch.ldexp(torch.ones_like(cmax, dtype=torch.float64), cmax - 132).float()
    return out_q, out_d, scale


def _convert(q, codes, chunk=32):
    parts = [
        _to_layout(q[e : e + chunk], codes[e : e + chunk])
        for e in range(0, q.shape[0], chunk)
    ]
    return tuple(torch.cat(p).contiguous() for p in zip(*parts))


def prepare_a8w4_weights(w1, w1_scale, w2, w2_scale):
    """Convert standard MXFP4 MoE weights to the A8W4 private layout (run once at load time).

    Inputs: ``w1`` [E, 2I, H/2] FP4x2 (``torch.float4_e2m1fn_x2`` or uint8; rows ``[0, I)`` gate,
    ``[I, 2I)`` up; low nibble = even K), ``w1_scale`` [E, 2I, H/32] E8M0 (``float8_e8m0fnu`` or
    uint8), ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32]. Requires ``H % 512 == 0`` and
    ``I % 32 == 0``. I is zero-padded to a multiple of 64 (160 -> 192): padded Gate/Up rows and
    Down columns are zero, so results equal the unpadded MoE.

    Precision: each 32-K group is decoded to FP8 relative to its row's largest E8M0 code; groups
    more than 11 codes (2**11) below that maximum are treated as zero (``MAX_DELTA``).
    One result serves every token count of ``fused_moe_a8w4``.
    """
    E, n1, half_h = w1.shape
    inter, H = n1 // 2, half_h * 2
    if n1 % 2 or H % 512 or inter % 32:
        raise ValueError(
            f"unsupported shape: H={H} (needs % 512), I={inter} (needs % 32)"
        )
    q1 = _packed(w1, (E, 2 * inter, H // 2))
    q2 = _packed(w2, (E, H, inter // 2))
    c1 = _codes(w1_scale, (E, 2 * inter, H // 32))
    c2 = _codes(w2_scale, (E, H, inter // 32))
    ip = -(-inter // _PAD_I) * _PAD_I
    if ip != inter:
        pad = ip - inter
        q1 = torch.cat(
            [
                q1[:, :inter],
                q1.new_zeros(E, pad, H // 2),
                q1[:, inter:],
                q1.new_zeros(E, pad, H // 2),
            ],
            1,
        )
        c1 = torch.cat(
            [
                c1[:, :inter],
                c1.new_full((E, pad, H // 32), 127),
                c1[:, inter:],
                c1.new_full((E, pad, H // 32), 127),
            ],
            1,
        )
        q2 = torch.nn.functional.pad(q2, (0, pad // 2))
        c2 = torch.nn.functional.pad(c2, (0, pad // 32))
    w1q, w1d, w1s = _convert(q1.contiguous(), c1.contiguous())
    w2q, w2d, w2s = _convert(q2.contiguous(), c2.contiguous())
    return A8W4Weights(w1q, w1d, w1s, w2q, w2d, w2s, E, H, inter, ip)


# ---------------------------------------------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------------------------------------------
def _arg(t):
    # Pointer element types the launchers were traced with: bytes for uint8 / FP8, else the tensor's type.
    return ptr_arg(t, fx.Uint8 if t.dtype == torch.uint8 else _TORCH_TO_FX[t.dtype])


def _launch(name, kernel, *args):
    # Every argument is a contiguous tensor of the dtype the factory fixes, or a scalar: the compiled handle
    # only needs the device in its key (aiter/ops/flydsl/AGENTS.md).
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    _run_compiled(
        kernel,
        *(_arg(a) if isinstance(a, torch.Tensor) else a for a in args),
        current_stream_arg(device),
        specialization_key=device.index,
    )
    last_path.append(name)


# ---------------------------------------------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------------------------------------------
def _decode_waves(topk, paired):
    """Waves of the Down decode kernel: few tokens (latency bound) one per routing slot; pre-quantized
    rows (``paired``) DECODE_DOWN_PQ_WAVES waves with slot pairs in rounds."""
    return (
        min((topk + 1) // 2, DECODE_DOWN_PQ_WAVES)
        if paired
        else (topk if topk <= 16 else 4)
    )


def _decode_fits(H, inter, topk, paired):
    """Whether the decode kernels support this shape (Gate/Up and the single-slot or slot-pair Down) and it was
    measured to take them (hidden <= 4096 or in DECODE_WIDE_HIDDEN)."""
    return (
        (H <= 4096 or H in DECODE_WIDE_HIDDEN)
        and a8w4_decode.gateup_decode_fits(H)
        and a8w4_decode.down_decode_fits(
            inter, topk, _decode_waves(topk, paired), paired
        )
    )


def _down_pq_fits(H, weights, topk):
    """Whether the slot-pair Down decode kernel (pre-quantized rows) takes this shape: it addresses all experts'
    weights with 32-bit byte offsets; larger weights or shapes it does not support take the single-slot Down.
    """
    small_w2 = weights.w2.numel() * weights.w2.element_size() < 2**32
    return small_w2 and _decode_fits(H, weights.inter_padded, topk, True)


@functools.cache
def _gateup_decode_kernel(N1, H, topk, prequant, limit):
    return a8w4_decode.compile_a8w4_gateup_decode(
        N=N1, K=H, TOPK=topk, PREQUANT=prequant, SWIGLU_LIMIT=limit
    )


@functools.cache
def _down_decode_kernel(H, inter, topk, paired):
    if paired:
        # More tokens: 16-byte loads of slot pairs.
        return a8w4_decode.compile_a8w4_down_decode_pq(
            N=H, K=inter, TOPK=topk, WAVES=_decode_waves(topk, True)
        )
    return a8w4_decode.compile_a8w4_down_decode(
        N=H, K=inter, TOPK=topk, WAVES=_decode_waves(topk, False)
    )


@functools.cache
def _gateup_dd_kernel(N1, H, topk, group, waves, limit):
    return compile_a8w4_gateup_decode_dd(
        N=N1, K=H, TOPK=topk, GROUP=group, WAVES=waves, SWIGLU_LIMIT=limit
    )


@functools.cache
def _row_quant_plan_kernel(cols, topk, group):
    return compile_a8w4_row_quant_plan(
        COLS=cols, TOPK=topk, MAX_M=DECODE_MAX_M, GROUP=group
    )


@functools.cache
def _row_quant_decode_kernel(cols, bf16):
    return a8w4_decode.compile_a8w4_row_quant(COLS=cols, BF16=bf16)


def _quantize_decode(x, rows, cols, bf16):
    """Per-row FP8 quantization with the decode path's vectorized kernel (same arithmetic as ``_quantize``)."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _launch(
        "a8w4_row_quant_decode", _row_quant_decode_kernel(cols, bf16), x, q, scale, rows
    )
    return q, scale


def _fused_moe_decode_wide_dd(
    hidden_states, weights, topk_weight, topk_ids, output, limit, decode_down
):
    """Decode of the DECODE_WIDE_HIDDEN sizes from DECODE_WIDE_DEDUP_MIN_M tokens: one kernel quantizes x and writes
    the dedup plan, Gate/Up is deduplicated by expert (``decode_dedup``), the row kernel quantizes the intermediate
    and the slot-pair Down sums each token's routes."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    inter = weights.inter_padded
    group = 2 if M <= DECODE_DEDUP_PAIR_MAX_M else DECODE_DEDUP_GROUP
    device = hidden_states.device
    plan = torch.empty((M * topk, 4), dtype=torch.int32, device=device)
    mid = torch.empty((M, topk, inter), dtype=torch.float32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    # ``output`` fills the plan kernel's unused pointer argument.
    _launch(
        "a8w4_row_quant_plan_decode",
        _row_quant_plan_kernel(H, topk, group),
        hidden_states,
        xq,
        xs,
        topk_ids,
        plan,
        output,
        M,
    )
    _launch(
        "a8w4_gateup_decode_dd",
        _gateup_dd_kernel(2 * inter, H, topk, group, DECODE_WIDE_DEDUP_WAVES, limit),
        xq,
        xs,
        topk_ids,
        plan,
        weights.w1,
        weights.w1_delta,
        weights.w1_scale,
        mid,
        M,
    )
    mq, ms = _quantize_decode(mid, M * topk, inter, False)
    if decode_down is not None:
        decode_down(True, mq, ms, topk_ids, topk_weight, weights, output)
        return output
    _launch(
        "a8w4_down_decode_pq",
        _down_decode_kernel(H, inter, topk, True),
        mq,
        ms,
        topk_ids,
        topk_weight,
        weights.w2,
        weights.w2_delta,
        weights.w2_scale,
        output,
        M,
    )
    return output


def _fused_moe_decode(
    hidden_states, weights, topk_weight, topk_ids, output, limit, decode_down
):
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    inter = weights.inter_padded
    pq_fits = _down_pq_fits(H, weights, topk)
    if H in DECODE_WIDE_HIDDEN and M >= DECODE_WIDE_DEDUP_MIN_M and pq_fits:
        return _fused_moe_decode_wide_dd(
            hidden_states, weights, topk_weight, topk_ids, output, limit, decode_down
        )
    # Few tokens: each workgroup quantizes its own activation rows; more tokens: quantize once.
    gateup_pq = M * topk * (inter // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS
    down_pq = M >= DECODE_PREQUANT_MIN_M and pq_fits
    mid = torch.empty(
        (M, topk, inter), dtype=torch.float32, device=hidden_states.device
    )
    if gateup_pq:
        xq, xs = _quantize_decode(hidden_states, M, H, True)
    else:
        xq, xs = hidden_states, topk_weight
    # ``mid`` also fills the Gate/Up kernel's two unused pointer arguments.
    _launch(
        "a8w4_gateup_decode" + ("_pq" if gateup_pq else ""),
        _gateup_decode_kernel(2 * inter, H, topk, gateup_pq, limit),
        xq,
        xs,
        topk_ids,
        weights.w1,
        weights.w1_delta,
        weights.w1_scale,
        mid,
        mid,
        mid,
        M,
    )
    if down_pq:
        mq, ms = _quantize_decode(mid, M * topk, inter, False)
    else:
        mq, ms = mid, topk_weight
    if decode_down is not None:
        decode_down(down_pq, mq, ms, topk_ids, topk_weight, weights, output)
        return output
    _launch(
        "a8w4_down_decode" + ("_pq" if down_pq else ""),
        _down_decode_kernel(H, inter, topk, down_pq),
        mq,
        ms,
        topk_ids,
        topk_weight,
        weights.w2,
        weights.w2_delta,
        weights.w2_scale,
        output,
        M,
    )
    return output


# ---------------------------------------------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------------------------------------------
def _stream_fused_quant(inter, block_m, hidden):
    """Whether the streaming Gate/Up kernel quantizes the intermediate itself: four waves (G = I / 64 gate
    blocks each) cover all of I, when their accumulators fit (8 * G * sub-blocks <= 96 VGPRs). From hidden
    4096 on, waves of one gate block plus the row kernel are faster."""
    return block_m > 16 and hidden < 4096 and 8 * (inter // 64) * (block_m // 16) <= 96


@functools.cache
def _stream_gateup_kernel(N, K, topk, block_m, experts, limit):
    # Fused quantization: one workgroup of four waves per block covers all of I and quantizes the intermediate
    # in the epilogue; with up to 4 + 4 row blocks per wave the activations are a large share of the traffic, so
    # the waves share them in LDS. Otherwise many light waves (one gate and one up block each, four per
    # workgroup, eight with larger blocks) sharing the block's activations in LDS.
    if _stream_fused_quant(N // 2, block_m, K):
        G = N // 2 // 64
        tile = {"G": G, "PREFETCH": (4, 3), "QUANT": True, "SHARE": G <= 4}
    else:
        W = 8 if block_m > 16 and (N // 2) % 128 == 0 else 4
        # From hidden 5120 on each lane loads 16 bytes of the shared activations per load, covering several 64-K steps.
        steps = max(STREAM_GATEUP_ACT_BYTES_H5120 * W // block_m, 1) if K >= 5120 else 1
        tile = {
            "G": 1,
            "W": W,
            "PREFETCH": (STREAM_GATEUP_PREFETCH, STREAM_GATEUP_PREFETCH),
            "SYNC": STREAM_GATEUP_SYNC,
            "ACT_STEPS": steps,
        }
    return compile_a8w4_gateup_stream(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        E=experts,
        W_AUX=STREAM_GATEUP_W_AUX,
        SWIGLU_LIMIT=limit,
        **tile,
    )


@functools.cache
def _stream_prep_kernel(hidden, topk, experts, block_m, zero, qw):
    return compile_stream_prep(
        H=hidden, TOPK=topk, E=experts, BM=block_m, ZERO=zero, QW=qw
    )


def _stream_prep(hidden_states, topk_weight, topk_ids, experts, block_m, output, zero):
    """Sorting and input quantization in one launch (``compile_stream_prep``); with ``zero`` the same launch
    zeroes ``output`` for the atomic Down."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device
    n_sort, rows, blocks = stream_prep_sizes(M, topk, experts, block_m)
    sorted_ids = torch.empty((rows,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((rows,), dtype=torch.float32, device=device)
    sorted_experts = torch.empty((blocks,), dtype=torch.int32, device=device)
    num_valid = torch.empty((2,), dtype=torch.int32, device=device)
    loc_ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    # Four waves per input row while the row's 512-column chunks split evenly or rows are few (measured: H 4096
    # M512 13.2 vs 14.5 us; H 2560 M1024 one wave per row 15.4 vs 17.7 us).
    qw = 4 if H % 2048 == 0 or M <= 256 else 1
    _launch(
        "a8w4_stream_prep",
        _stream_prep_kernel(H, topk, experts, block_m, zero, qw),
        topk_ids,
        topk_weight,
        hidden_states,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        loc_ids,
        xq,
        xs,
        output,
        M,
        n_sort,
        rows,
    )
    return sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs


@functools.cache
def _route_sum_kernel(topk, cols):
    return compile_route_sum(topk, cols)


def _stream_down_shared(N, K, block_m):
    """Whether the streaming Down runs four waves of 8 row blocks that share the block's activation rows
    through LDS (``down_stream_lds``)."""
    # Blocks of 32 and 64 rows: four waves of 8 row blocks each share the block's activation rows
    # through LDS when they fit (with a single sub-block the sharing costs more than it saves, except for
    # a wide hidden dimension over a short K).
    wide = N >= 4096 and K <= 256
    return (
        (block_m >= 32 or wide)
        and N % 512 == 0
        and down_stream_lds(K, block_m, W=4) <= a8w4_common.LDS_BYTES
    )


@functools.cache
def _stream_down_kernel(N, K, topk, block_m, experts, atomic):
    tile = {"R": 8, "W": 4} if _stream_down_shared(N, K, block_m) else {}
    w_aux = STREAM_DOWN_W_AUX_H5120 if N >= 5120 else STREAM_DOWN_W_AUX
    return compile_a8w4_down_stream(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        E=experts,
        ATOMIC=atomic,
        PREFETCH=STREAM_DOWN_PREFETCH,
        W_AUX=w_aux,
        **tile,
    )


def _stream_atomic(inter):
    """Whether the streaming Down adds its routes into the output with BF16 atomics (no routes buffer and
    route_sum): measured faster up to I = 320 (Qwen TP2 M512 355 -> 336 us), slower at I = 512 (the atomic
    epilogue's extra registers on top of K = 512 activations)."""
    return inter < 512


def _stream_max_rows(hidden, inter, routes):
    """Most mean rows per expert (M * topk / E) for the streaming kernels (``routes`` = M * topk)."""
    if hidden >= 5120 and routes >= STREAM_H5120_MIN_ROUTES:
        return STREAM_MAX_ROWS_H5120
    if hidden >= 4096:
        return STREAM_MAX_ROWS_WIDE
    # Measured at hidden 2560 only; smaller hidden sizes keep STREAM_MAX_ROWS.
    if hidden >= 2560 and _stream_fused_quant(inter, 64, hidden):
        return STREAM_MAX_ROWS_FUSED_QUANT
    return STREAM_MAX_ROWS


def _stream_block_m(tokens, topk, experts, hidden):
    """Sorting block of the streaming kernels from the mean rows per expert (sub-blocks of 16).

    Up to 12 rows per expert: blocks of 16 once 3 * routes >= 4 * experts or from hidden 5120 on (DeepSeek TP4
    M48 / M64: 0.89x the time of blocks of 32), else 32.
    """
    rows = tokens * topk / experts
    if rows <= 12:
        return 16 if 3 * tokens * topk >= 4 * experts or hidden >= 5120 else 32
    return 32 if rows <= STREAM_BM32_ROWS else 64


def _quantize(x, rows, cols, dtype, fill=None, out=None):
    """Per-row FP8 quantization of ``x`` [rows, cols] (into ``out`` = (q, scale) if given); the same kernel sets
    int32 ``fill`` [rows, k] to -1."""
    if out is not None:
        q, scale = out
    else:
        q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
        scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    kernel = compile_row_quant_fp8(cols, dtype, 0 if fill is None else fill.shape[1])
    _launch(
        "a8w4_row_quant_fp8", kernel, x, q, scale, scale if fill is None else fill, rows
    )
    return q, scale


def _fused_moe_stream(
    hidden_states,
    weights,
    topk_weight,
    topk_ids,
    output,
    limit,
    reduce,
    stream_down=None,
):
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    E, inter = weights.experts, weights.inter_padded
    block_m = _stream_block_m(M, topk, E, H)
    device = hidden_states.device
    atomic = _stream_atomic(inter)
    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs = (
        _stream_prep(hidden_states, topk_weight, topk_ids, E, block_m, output, atomic)
    )
    task_num = sorted_experts.numel()
    gateup = _stream_gateup_kernel(2 * inter, H, topk, block_m, E, limit)
    routing = (sorted_ids, sorted_experts, num_valid)
    if not _stream_fused_quant(inter, block_m, H):
        mid = torch.empty((M, topk, inter), dtype=torch.float32, device=device)
        _launch(
            f"a8w4_gateup_stream_bm{block_m}",
            gateup,
            xq,
            weights.w1,
            weights.w1_delta,
            mid,
            *routing,
            weights.w1_scale,
            xs,
            xs,
            M,
            task_num,
        )
        mq, ms = _quantize(mid, M * topk, inter, "f32")
    else:
        mq = torch.empty((M * topk, inter), dtype=torch.float8_e4m3fnuz, device=device)
        ms = torch.empty((M * topk,), dtype=torch.float32, device=device)
        _launch(
            f"a8w4_gateup_quant_stream_bm{block_m}",
            gateup,
            xq,
            weights.w1,
            weights.w1_delta,
            mq,
            *routing,
            weights.w1_scale,
            xs,
            ms,
            M,
            task_num,
        )
    down_args = (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        weights.w2_scale,
        ms,
        M,
        task_num,
    )
    if stream_down is not None:
        routes = None
        if not atomic:
            routes = torch.empty(
                (task_num * block_m, H), dtype=torch.bfloat16, device=device
            )
        stream_down(
            mq, ms, down_args[:4], loc_ids, routes, output, weights, block_m, task_num
        )
        return output
    down = _stream_down_kernel(H, inter, topk, block_m, E, atomic)
    if atomic:
        # The prep zeroed output: Down adds every route into its token's row with BF16 atomics.
        _launch(
            f"a8w4_down_stream_bm{block_m}",
            down,
            mq,
            weights.w2,
            weights.w2_delta,
            output,
            *down_args,
        )
        if reduce is not None:
            reduce(output, None, None)
        return output
    routes = torch.empty((task_num * block_m, H), dtype=torch.bfloat16, device=device)
    _launch(
        f"a8w4_down_stream_bm{block_m}",
        down,
        mq,
        weights.w2,
        weights.w2_delta,
        routes,
        *down_args,
    )
    if reduce is not None:
        reduce(output, loc_ids, routes)
        return output
    _launch(
        "a8w4_route_sum",
        _route_sum_kernel(topk, H),
        loc_ids,
        routes,
        output,
        M,
        task_num * block_m,
    )
    return output


# ---------------------------------------------------------------------------------------------------------------
# prefill
# ---------------------------------------------------------------------------------------------------------------
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


@functools.cache
def _gateup_prefill_kernel(N, K, topk, experts, limit, xcd_chunk, chunked=False):
    return compile_a8w4_gemm1(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=PREFILL_BLOCK_M,
        BLOCK_TILE_SIZE_N=PREFILL_BLOCK_N,
        TILE_K=PREFILL_TILE_K,
        E=experts,
        SWIGLU_LIMIT=limit,
        XCD_CHUNK=xcd_chunk,
        CHUNKED=chunked,
    )


@functools.cache
def _down_prefill_kernel(N, K, topk, experts, nsplit, xcd_chunk):
    return compile_a8w4_gemm2(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=PREFILL_BLOCK_M,
        E=experts,
        NSPLIT=nsplit,
        XCD_CHUNK=xcd_chunk,
    )


def _fused_moe_prefill(
    hidden_states, weights, topk_weight, topk_ids, output, limit, reduce, down=None
):
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    E, inter = weights.experts, weights.inter_padded
    block_m, routes_n = PREFILL_BLOCK_M, M * topk
    device = hidden_states.device
    prep_routes = (
        PREFILL_PREP_MAX_ROUTES_H5120 if H >= 5120 else PREFILL_PREP_MAX_ROUTES
    )
    chunks = getattr(down, "chunks", None)
    if H >= 2560 and routes_n <= prep_routes and chunks is None:
        sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs = (
            _stream_prep(
                hidden_states, topk_weight, topk_ids, E, block_m, output, False
            )
        )
    else:
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
    mid = torch.empty((M, topk, inter), dtype=torch.float32, device=device)
    gateup_xcd = _xcd_chunk(GATEUP_XCD_CHUNK_H5120, H, routes_n)
    if chunks is not None:
        return _prefill_chunks(
            down,
            _gateup_prefill_kernel(2 * inter, H, topk, E, limit, gateup_xcd, True),
            xq,
            xs,
            weights,
            routing,
            mid,
            loc_ids,
            output,
            grid,
            _xcd_chunk(DOWN_XCD_CHUNK_H5120, H, routes_n),
        )
    gateup = _gateup_prefill_kernel(2 * inter, H, topk, E, limit, gateup_xcd)
    _launch(
        f"a8w4_gateup_prefill_bm{block_m}_bn{PREFILL_BLOCK_N}_tk{PREFILL_TILE_K}",
        gateup,
        xq,
        weights.w1,
        weights.w1_delta,
        mid,
        *routing,
        weights.w1_scale,
        xs,
        M,
        grid,
    )
    mq, ms = _quantize(mid, routes_n, inter, "f32")
    routes = torch.empty((grid * block_m, H), dtype=torch.bfloat16, device=device)
    xcd_chunk = _xcd_chunk(DOWN_XCD_CHUNK_H5120, H, routes_n)
    if down is not None:
        down(mq, ms, routing, loc_ids, routes, output, weights, grid, xcd_chunk)
        return output
    down = _down_prefill_kernel(
        H, inter, topk, E, _down_split(M, topk, E, H, inter, block_m), xcd_chunk
    )
    _launch(
        f"a8w4_down_prefill_bm{block_m}",
        down,
        mq,
        weights.w2,
        weights.w2_delta,
        routes,
        *routing,
        weights.w2_scale,
        ms,
        loc_ids,
        M,
        grid,
    )
    if reduce is not None:
        reduce(output, loc_ids, routes)
        return output
    sorted_sum(topk, H)(loc_ids, routes, output, M)
    last_path.append("sorted_sum")
    return output


def _prefill_chunks(
    down, gateup, xq, xs, weights, routing, mid, loc_ids, output, grid, xcd_chunk
):
    """The prefill path after the sort in token chunks (``prefill_down.chunks``, ``fused_moe_a8w4``): the chunks'
    sorted-block lists (``a8w4_chunk_blocks``, ``kernels/moe_a8w4/chunks.py``), then per chunk the ``CHUNKED``
    Gate/Up of its blocks, the quantization of the routes from the chunk's first token on (the later ones again
    with the later chunks: a sorted block of the chunk may hold routes of later tokens) and ``prefill_down``; the
    kernels of a chunk launch the bound of its blocks (``chunks.chunk_bound``)."""
    starts, chunk_desc = down.chunks
    sorted_ids, sorted_weights, sorted_experts, num_valid = routing
    M, topk, inter = mid.shape
    rows = M * topk
    desc = chunk_desc(grid)
    build = a8w4_chunks.compile_chunk_blocks(
        CHUNKS=len(starts),
        K=-(-grid // (64 * a8w4_chunks.WAVES)),
        BM=PREFILL_BLOCK_M,
        ROW=desc.shape[1],
    )
    _launch("a8w4_chunk_blocks", build, sorted_ids, num_valid, desc)
    mid_rows = mid.view(rows, inter)
    mq = torch.empty((rows, inter), dtype=torch.float8_e4m3fnuz, device=mid.device)
    ms = torch.empty((rows,), dtype=torch.float32, device=mid.device)
    routes = torch.empty(
        (grid * PREFILL_BLOCK_M, xq.shape[1]), dtype=torch.bfloat16, device=mid.device
    )
    for c, (first, end) in enumerate(zip(starts, (*starts[1:], M))):
        bound = a8w4_chunks.chunk_bound(
            end - first, topk, PREFILL_BLOCK_M, weights.experts, grid
        )
        chunk = (sorted_ids, sorted_weights, sorted_experts, desc[c])
        _launch(
            f"a8w4_gateup_prefill_bm{PREFILL_BLOCK_M}_bn{PREFILL_BLOCK_N}_tk{PREFILL_TILE_K}",
            gateup,
            xq,
            weights.w1,
            weights.w1_delta,
            mid,
            *chunk,
            weights.w1_scale,
            xs,
            M,
            bound,
        )
        r0 = first * topk
        _quantize(mid_rows[r0:], rows - r0, inter, "f32", out=(mq[r0:], ms[r0:]))
        down(mq, ms, chunk, loc_ids, routes, output, weights, bound, xcd_chunk, c)
    return output


# ---------------------------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------------------------
def _swiglu_limit(value):
    """``swiglu_limit`` of ``fused_moe_a8w4`` as a float, 0.0 when off."""
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"swiglu_limit must be a real number, got {type(value).__name__}"
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"swiglu_limit must be finite and >= 0 (0 disables it), got {value!r}"
        )
    return float(value) if value > 0 else 0.0


def fused_moe_a8w4(
    hidden_states,
    weights,
    topk_weight,
    topk_ids,
    *,
    output=None,
    swiglu_limit=0.0,
    reduce=None,
    decode_down=None,
    prefill_down=None,
    stream_down=None,
):
    """A8W4 MoE forward of one rank: BF16 ``hidden_states`` [M, H] -> BF16 [M, H] (``output`` if given).

    ``weights`` comes from ``prepare_a8w4_weights``; ``topk_weight`` float32 [M, topk] is applied in Down;
    ``topk_ids`` int32 [M, topk] in ``[0, E)`` (a fused shared expert is an ordinary expert id), distinct within a
    token (with a repeated id the Opus-sorted prefill keeps one route, the other paths every route). The input and
    the f32 intermediate SiLU(gate) * up are quantized per row to FP8 E4M3FNUZ with ``scale = amax / 240`` and
    ``q = RNE(x / scale)``. No expert mask / bias; ``M * topk * I_padded * 4 < 2**31``. Issues no host
    synchronization. The path (decode / stream / prefill) follows from M and the shape (module docstring).

    ``swiglu_limit`` is the SwiGLU limit of DeepSeek-V4.1 (10 for V4.1-Flash): when > 0 every Gate/Up kernel
    computes ``SiLU(gate.clamp(max=L)) * up.clamp(-L, L)`` on the dequantized f32 gate / up values (after the
    weight and activation scales); the limit is a compile-time constant (the Gate/Up kernel symbols end in
    ``_swiglu``; the ``last_path`` names do not change).
    0 (default) runs the kernels without it. Must be finite and >= 0 (``ValueError``).

    ``reduce`` (default None) replaces the last step of the stream and prefill paths, for a caller that fuses
    more work into it (``MegaMoeTPGfx942``: the cross-rank sum): instead of the route sum it is called as
    ``reduce(output, loc_ids, routes)`` -- ``routes`` BF16 [rows, H] by sorted row, ``loc_ids`` int32 [M, topk]
    the sorted row of each route (negative: none), summed in slot order -- and after the atomic streaming Down
    (which leaves the route sum in ``output``) as ``reduce(output, None, None)``; it must launch on the current
    stream and leave the result in ``output``. The decode path never calls it.

    ``decode_down`` (default None) likewise replaces the Down kernel of the decode path: it is called with that
    kernel's inputs as ``decode_down(paired, mid, mid_scale, topk_ids, topk_weight, weights, output)`` --
    ``paired``: the slot-pair kernel, ``mid`` FP8 [M * topk, I] with its row scales ``mid_scale``; else the
    single-slot kernel, ``mid`` f32 [M, topk, I] and ``mid_scale`` the (unused) ``topk_weight`` -- and must launch on
    the current stream and leave the route sum in ``output``. The other paths never call it.

    ``prefill_down`` (default None) likewise replaces the Down kernel and ``reduce`` of the prefill path: it is called
    as ``prefill_down(mid, mid_scale, routing, loc_ids, routes, output, weights, grid, xcd_chunk)`` -- ``mid`` FP8
    [M * topk, I] with its row scales ``mid_scale``, ``routing`` the sorted (ids, weights, experts, valid count),
    ``loc_ids`` as for ``reduce`` (not yet stored by Down), ``routes`` the uninitialized BF16 [grid * 64, H] route
    rows, ``grid`` the sorted blocks and ``xcd_chunk`` the Down kernel's XCD_CHUNK -- and must launch on the current
    stream and leave the result in ``output``. The other paths never call it. If it has an attribute ``chunks`` =
    (``starts``, ``desc``), the prefill path runs in token chunks after the sort: chunk c takes the tokens from
    ``starts[c]`` (``starts[0]`` = 0, increasing) to the next chunk's first one (the last one to M) and the sorted
    blocks whose first route's token is among them (``moe_sorting``, which keeps the token order within an expert,
    is then used at any M); ``desc(grid)`` returns the chunk descriptors on the device, int32 [chunks,
    ``chunks.desc_row_words(grid)``] (``kernels/moe_a8w4/chunks.py``: per chunk its first token, its end and c set
    by the host, the valid count and the chunk's sorted-block list stored by each call). Per chunk, Gate/Up
    (``CHUNKED``) of its blocks, the quantization of the routes from its first token on, then ``prefill_down(...,
    routing, ..., grid, xcd_chunk, c)`` with ``desc[c]`` in place of the valid count in ``routing`` and ``grid`` the
    chunk's block bound (``chunks.chunk_bound``): the routes of the tokens below a chunk's end are complete after its
    call.

    ``stream_down`` (default None) likewise replaces the Down kernel and ``reduce`` of the stream path: it is called
    as ``stream_down(mid, mid_scale, routing, loc_ids, routes, output, weights, block_m, task_num)`` -- ``mid`` FP8
    [M * topk, I] with its row scales ``mid_scale``, ``routing`` the sorted (ids, weights, experts, valid count) of
    ``block_m``-row blocks (``task_num`` of them), ``loc_ids`` as for ``reduce`` (stored), ``routes`` the
    uninitialized BF16 [task_num * block_m, H] route rows, or None for the atomic Down (``output`` zeroed) -- and must
    launch on the current stream and leave the result in ``output``. The other paths never call it.
    """
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise TypeError("hidden_states must be a 2-D bfloat16 tensor")
    M, H = hidden_states.shape
    if H != weights.hidden:
        raise ValueError(
            f"hidden size {H} does not match the weights ({weights.hidden})"
        )
    if (
        topk_ids.dim() != 2
        or topk_ids.shape[0] != M
        or tuple(topk_weight.shape) != tuple(topk_ids.shape)
    ):
        raise ValueError(
            f"topk_ids / topk_weight must both be [{M}, topk], got {tuple(topk_ids.shape)} / "
            f"{tuple(topk_weight.shape)}"
        )
    limit = _swiglu_limit(swiglu_limit)
    if output is None:
        output = torch.empty_like(hidden_states)
    elif (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != (M, H)
        or not output.is_contiguous()
    ):
        raise ValueError(f"output must be a contiguous bfloat16 [{M}, {H}] tensor")
    last_path.clear()
    if M == 0:
        return output
    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weight = topk_weight.to(torch.float32).contiguous()
    topk = topk_ids.shape[1]
    E, inter = weights.experts, weights.inter_padded
    if M <= DECODE_MAX_M and _decode_fits(H, inter, topk, False):
        return _fused_moe_decode(
            hidden_states, weights, topk_weight, topk_ids, output, limit, decode_down
        )
    if M > DECODE_MAX_M and M * topk <= _stream_max_rows(H, inter, M * topk) * E:
        return _fused_moe_stream(
            hidden_states,
            weights,
            topk_weight,
            topk_ids,
            output,
            limit,
            reduce,
            stream_down,
        )
    return _fused_moe_prefill(
        hidden_states,
        weights,
        topk_weight,
        topk_ids,
        output,
        limit,
        reduce,
        prefill_down,
    )
