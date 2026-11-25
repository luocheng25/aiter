# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""1x32 (MXFP8) quantization of the BF16 activation of the gfx942 A8W4 GEMM.

Each token's K values are split into 32-wide blocks; a block gets one E8M0 scale byte and
32 FP8 E4M3FNUZ values. The rounding is the one of DeepSeek-V4.1's reference
``act_quant(x, block_size=32)`` with UE8M0 scales and E4M3FN values (``inference/kernel.py``),
mapped to FNUZ, bit for bit:

* ``amax`` = the largest ``|x|`` of the block (f32), then ``max(amax, 1e-4)``;
* ``e_fn = ceil(log2(amax * f32(1 / 448)))`` (exponent bits of the f32 product, plus one when
  its mantissa is nonzero) and ``q_fn = RNE_e4m3fn(clamp(x / 2**e_fn, -448, 448))``;
* FNUZ: the scale byte is ``e_fn + 128`` (``2**e = 2 * 2**e_fn``) and ``q`` keeps the code of
  ``q_fn``, with -0 stored as +0 (0x00) and NaN as 0x80: ``q = RNE_fnuz(x * 2**-e)``,
  ``|q| <= 224``.

The kernel takes the scale from the bit pattern: ``p2 = (bits(max(amax, 1e-4)) + 0x1FFFFF) &
0xFF800000`` rounds ``amax`` up to a power of two only when its mantissa exceeds 1.75, and the
byte is ``log2(p2) - 7 + 127``. This equals the formula for every finite BF16 ``amax``
(checked for all of them); the bytes are 106 (``amax <= 1e-4``, e.g. an all-zero block) to 248.

Against ``aiter.ops.triton.quant.dynamic_mxfp8_quant(x, quant_dtype=torch.float8_e4m3fnuz)``
(``+ 0x200000``, no floor, scale exponent clamped to 127) a block differs only when its
``amax`` has the mantissa 1.75 exactly (aiter rounds it up: scale byte + 1), when
``amax < 1e-4`` (aiter has no floor) and when ``amax >= 1.7578125 * 2**127`` (BF16
0x7F61-0x7F7F: aiter saturates the byte to 254 and stores zeros); with NaN inputs aiter's
Triton max also depends on its reduction order. Elsewhere the two are bit-identical.

Non-finite inputs (the reference's NaN / Inf handling depends on its CUDA backend): NaN
elements are ignored by ``amax`` (as CUDA's ``fmaxf`` does) and quantize to 0x80, so an
all-NaN block gets the floor byte 106; a block holding +-Inf has an infinite scale in the
reference (E8M0 NaN), here byte 254 with its finite elements quantized to 0x00 and +-Inf to
0x80 (NaN): a GEMM row that reads Inf or NaN is NaN either way. BF16 subnormal inputs are
kept (next to the 1e-4 floor they quantize to 0).

The work split is ``quant.QuantConfig`` (the per-token kernel's): ``lanes`` lanes per row, load
``v`` of lane ``sub`` reads the 16-byte unit ``sub + v * lanes`` (8 BF16), so the four lanes
``4j .. 4j + 3`` of a row hold one 32-block per load and combine their maxima with two DPP
quad permutations. The maximum runs on 16-bit keys of ``|x|`` (two per ``v_pk_max_u16``):
``t = |bits| + 0x7F`` and ``key = min(t, t ^ 0x8000)`` maps the non-NaN values in order to
``[0x7F, 0x7FFF]`` and the NaNs below them, so the largest key is the largest non-NaN ``|x|``
(0 for an all-NaN block). Lane ``4j`` stores the block's scale byte.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor
from .quant import (
    _SC0_SC1,
    _WAVE,
    _WIDE_MAX_ROWS,
    _config_supported,
    _dpp_max,
    _max_u16x2,
    _pack_fp8x4,
    quant_config,
)

BLOCK_K = 32
# Key offset: non-NaN |x| bit patterns [0, 0x7F80] -> [0x7F, 0x7FFF]; NaNs fall below it.
_KEY_BIAS = 0x7F
_BF16_INF = 0x7F80
# f32 1e-4, the reference's amax floor, and the rounding addend of 2**ceil(log2(amax / 448)).
_AMAX_FLOOR_BITS = 0x38D1B717
_ROUND_UP = 0x1FFFFF
# Scale byte of a block holding +-Inf (its finite elements quantize to 0).
_E8M0_INF_BLOCK = 254


def quant_1x32_kernel_name(cols, cfg):
    return format_kernel_name(
        f"gemm_a8w4_ptpc_quant1x32_k{cols}_l{cfg.lanes}_r{cfg.rows}"
    )


def _amax_keys(raw):
    """Four BF16x2 dwords -> one dword of two 16-bit keys (their maxima per half)."""
    keys = []
    for j in range_constexpr(4):
        t = (raw[j] & fx.Uint32(0x7FFF7FFF)) + fx.Uint32(_KEY_BIAS * 0x00010001)
        flipped = t ^ fx.Uint32(0x80008000)
        keys.append(
            fx.min(
                Vec.from_elements([t], fx.Uint32).bitcast(fx.Uint16),
                Vec.from_elements([flipped], fx.Uint32).bitcast(fx.Uint16),
            )
        )
    return _max_u16x2(keys).bitcast(fx.Uint32)[0]


def _block_scale(key):
    """Block key -> (scale byte, f32 multiplier ``2**-e``; 0 for a block holding +-Inf)."""
    amax16 = fx.max(key, fx.Uint32(_KEY_BIAS)) - fx.Uint32(_KEY_BIAS)
    bits = fx.max(amax16 << 16, fx.Uint32(_AMAX_FLOOR_BITS))
    p2 = (bits + fx.Uint32(_ROUND_UP)) & fx.Uint32(0xFF800000)
    inf = amax16 == fx.Uint32(_BF16_INF)
    e8m0 = inf.select(fx.Uint32(_E8M0_INF_BLOCK), (p2 >> 23) - fx.Uint32(7))
    inv_bits = inf.select(fx.Uint32(0), (fx.Uint32(254) - e8m0) << 23)
    return e8m0, inv_bits.bitcast(fx.Float32)


@functools.cache
def compile_quant_x_1x32(cols, device_index, cfg=None):
    """Launcher ``(x, q, scale, rows, stream)`` of the 1x32 quantization.

    BF16 ``x`` [rows, cols] -> FP8 E4M3FNUZ ``q`` [rows, cols] and uint8 E8M0 ``scale``
    [rows, cols / 32] (row-major), passed as raw ``fx.Pointer``; ``cols % 64 == 0``.
    ``cfg`` is a ``quant.QuantConfig`` (default: the large-M one). ``device_index`` only
    keys the cache (a compiled handle launches on one GPU).
    """
    del device_index
    if cfg is None:
        cfg = quant_config(_WIDE_MAX_ROWS + 1, cols)
    if not _config_supported(cols, cfg) or cfg.lanes % 4 != 0:
        raise ValueError(f"cols={cols} must be a multiple of 64 with a valid {cfg}")
    lanes, rows_per_block = cfg
    # Same store policy as the per-token kernel (see quant.compile_quant_x).
    store_policy = _SC0_SC1 if lanes <= _WAVE else 0
    units_per_row = cols // 8
    loads = units_per_row // lanes
    threads = lanes * rows_per_block
    row_bytes = cols * 2
    blocks_per_row = cols // BLOCK_K

    @flyc.kernel(
        name=quant_1x32_kernel_name(cols, cfg), known_block_size=[threads, 1, 1]
    )
    def quant_x_1x32_kernel(
        p_x: fx.Pointer, p_q: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32
    ):
        tid = fx.thread_idx.x
        sub = tid % lanes
        local = tid // lanes
        row0 = fx.block_idx.x * rows_per_block
        # Buffers rebased at the CTA's first row: 32-bit offsets for any row count; rows
        # past ``rows`` read zeros and drop their stores.
        nrows = fx.min(rows - row0, fx.Int32(rows_per_block))
        x_buf = ptr_buf_tensor(
            buf_base_i64(p_x) + fx.Int64(row0) * row_bytes,
            elem=fx.Uint32,
            unit_elems=4,
            num_records_bytes=nrows * row_bytes,
        )
        q_buf = ptr_buf_tensor(
            buf_base_i64(p_q) + fx.Int64(row0) * cols,
            elem=fx.Uint32,
            unit_elems=2,
            num_records_bytes=nrows * cols,
        )
        s_buf = ptr_buf_tensor(
            buf_base_i64(p_scale) + fx.Int64(row0) * blocks_per_row,
            elem=fx.Uint8,
            num_records_bytes=nrows * blocks_per_row,
        )

        unit0 = local * units_per_row + sub
        raw = [
            buf_copy_load(x_buf, unit0 + v * lanes, elem=fx.Uint32, unit_elems=4)
            for v in range_constexpr(loads)
        ]
        scale_bytes = []
        for v in range_constexpr(loads):
            pair = _amax_keys(raw[v])
            key = fx.max(pair & fx.Uint32(0xFFFF), pair >> 16)
            # The block's four lanes (aligned quad): quad_perm [1, 0, 3, 2], [2, 3, 0, 1].
            key = _dpp_max(fx.Int32(key), 0xB1)
            key = fx.Uint32(_dpp_max(key, 0x4E))
            e8m0, inv = _block_scale(key)
            scale_bytes.append(e8m0)
            inv2 = Vec.from_elements([inv, inv], fx.Float32)
            dwords = []
            for d in range_constexpr(2):
                q = []
                for i in range_constexpr(2):
                    w = fx.Uint32(raw[v][2 * d + i])
                    vals = Vec.from_elements(
                        [w << 16, w & fx.Uint32(0xFFFF0000)], fx.Uint32
                    ).bitcast(fx.Float32)
                    q.append(Vec(vals * inv2))
                dwords.append(_pack_fp8x4([q[0][0], q[0][1], q[1][0], q[1][1]]))
            buf_copy_store(
                q_buf,
                unit0 + v * lanes,
                Vec.from_elements(dwords, fx.Uint32),
                elem=fx.Uint32,
                unit_elems=2,
                cache_modifier=store_policy,
            )
        if sub % 4 == 0:
            for v in range_constexpr(loads):
                buf_copy_store(
                    s_buf,
                    local * blocks_per_row + (sub + v * lanes) // 4,
                    fx.Uint8(scale_bytes[v]),
                    elem=fx.Uint8,
                    cache_modifier=store_policy,
                )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        quant_x_1x32_kernel(p_x, p_q, p_scale, rows).launch(
            grid=((rows + rows_per_block - 1) // rows_per_block, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
