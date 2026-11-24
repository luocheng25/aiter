# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-token FP8 E4M3FNUZ quantization of the BF16 activation of the gfx942 A8W4 GEMM.

``scale = amax / 240`` (IEEE f32 division; an all-zero row gets scale 1) and
``q = RNE(x / scale)`` with the IEEE quotient (FNUZ has no -0: it is stored as 0): bit-exact with
``(x.float() / scale).to(torch.float8_e4m3fnuz)``.

A row is split over ``lanes`` lanes of one wave. Load ``v`` of lane ``sub`` reads the eight
BF16 at 16-byte unit ``sub + v * lanes`` of the row and its FP8 store writes eight bytes at
unit ``sub + v * lanes``: every load and store instruction covers a contiguous run of
``16 * lanes`` / ``8 * lanes`` bytes. ``amax`` is the unsigned maximum of the BF16 bit patterns
with the sign cleared, two per ``v_pk_max_u16``: it orders like the magnitudes and propagates
NaN as torch's ``amax`` does.

The quotient uses ``y = 1 / scale`` (IEEE, once per row) and two FMA corrections
``q += (x - q * scale) * y`` (Markstein): the first makes ``q`` faithful, the second yields
the correctly rounded quotient. Rows with ``scale < 2**-64``, where the residual could
underflow, divide per element instead. No clamp precedes the conversion: ``|x| / scale``
is at most the float just above 240, which ``v_cvt_pk_fp8_f32`` rounds to 240 like the
reference does, and NaN stays NaN.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor

FP8_MAX = 240.0
_SMALL_SCALE = 2.0**-64
_THREADS = 256


def _lanes_per_row(cols):
    for lanes in (64, 32, 16, 8, 4):
        if cols % lanes == 0 and (cols // lanes * 2) % 16 == 0:
            return lanes
    return None


def quant_kernel_name(cols):
    return format_kernel_name(f"gemm_a8w4_ptpc_quant_k{cols}")


def _pack_fp8x4(q):
    """4 f32 (|q| < 248) -> one dword of FP8 E4M3FNUZ, RNE (-0 and values rounding to 0 give 0x00)."""
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8
    # has no LLVM translation); the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
    return fx.Uint32(fx.rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], packed, True))


def _max_u16x2(vals):
    """Pairwise-tree unsigned maximum of 2 x u16 vectors (one v_pk_max_u16 per pair)."""
    while len(vals) > 1:
        tail = [vals[-1]] if len(vals) % 2 else []
        vals = [fx.max(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)] + tail
    return vals[0]


@functools.cache
def compile_quant_x(cols, device_index):
    """Launcher ``(x, q, scale, rows, stream)`` of the per-token quantization.

    BF16 ``x`` [rows, cols] -> FP8 ``q`` [rows, cols] and f32 ``scale`` [rows], passed as raw
    ``fx.Pointer``; ``cols % 64 == 0``. ``device_index`` only keys the cache (a compiled
    handle launches on one GPU).
    """
    del device_index
    lanes = _lanes_per_row(cols)
    if lanes is None:
        raise ValueError(f"cols={cols} must be a multiple of 64")
    per_lane = cols // lanes
    loads = per_lane * 2 // 16
    words = per_lane // 4
    rows_per_block = _THREADS // lanes
    shuffles = [lanes >> (i + 1) for i in range(lanes.bit_length() - 1)]
    row_bytes = cols * 2

    @flyc.kernel(name=quant_kernel_name(cols), known_block_size=[_THREADS, 1, 1])
    def quant_x_kernel(
        p_x: fx.Pointer, p_q: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32
    ):
        tid = fx.thread_idx.x
        sub = tid % lanes
        local = tid // lanes
        row0 = fx.block_idx.x * rows_per_block
        # Each block rebases its buffers at its first row: offsets stay 32-bit for any
        # row count, and rows past ``rows`` read zeros and drop their stores.
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
            buf_base_i64(p_scale) + fx.Int64(row0) * 4,
            elem=fx.Float32,
            num_records_bytes=nrows * 4,
        )

        values = []
        raw_words = []
        unit0 = local * (row_bytes // 16) + sub
        for v in range_constexpr(loads):
            raw = buf_copy_load(x_buf, unit0 + v * lanes, elem=fx.Uint32, unit_elems=4)
            for j in range_constexpr(4):
                word = raw[j]
                raw_words.append(word)
                values.append((word << 16).bitcast(fx.Float32))
                values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
        # amax on the BF16 bit patterns: |x| as unsigned 16-bit integers orders like the
        # values and puts NaN above Inf (NaN propagates, as torch's amax does); two per
        # dword (v_pk_max_u16) instead of a NaN-propagating float max (three VALU each).
        packed = _max_u16x2(
            [
                Vec.from_elements([w & fx.Uint32(0x7FFF7FFF)], fx.Uint32).bitcast(fx.Uint16)
                for w in raw_words
            ]
        ).bitcast(fx.Uint32)[0]
        amax_bits = fx.max(packed & fx.Uint32(0xFFFF), packed >> 16)
        for i in range_constexpr(len(shuffles)):
            amax_bits = fx.max(
                amax_bits,
                fx.Uint32(
                    fx.Int32(amax_bits).shuffle_xor(fx.Int32(shuffles[i]), fx.Int32(64))
                ),
            )
        amax = (amax_bits << 16).bitcast(fx.Float32)
        scale = amax / FP8_MAX
        scale = (scale == 0.0).select(fx.Float32(1.0), scale)
        if sub == 0:
            buf_copy_store(s_buf, local, scale, elem=fx.Float32)

        q_unit0 = local * (cols // 8) + sub

        def emit(quotient):
            dwords = []
            for w in range_constexpr(words):
                q = [quotient(values[4 * w + i]) for i in range_constexpr(4)]
                dwords.append(_pack_fp8x4(q))
            for w in range_constexpr(words // 2):
                buf_copy_store(
                    q_buf,
                    q_unit0 + w * lanes,
                    Vec.from_elements([dwords[2 * w], dwords[2 * w + 1]], fx.Uint32),
                    elem=fx.Uint32,
                    unit_elems=2,
                )

        if scale >= fx.Float32(_SMALL_SCALE):
            y = 1.0 / scale

            def markstein(x):
                q = x * y
                q = fx.Float32(fx.fma(fx.fma(-q, scale, x), y, q))
                return fx.Float32(fx.fma(fx.fma(-q, scale, x), y, q))

            emit(markstein)
        else:
            emit(lambda x: x / scale)

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        quant_x_kernel(p_x, p_q, p_scale, rows).launch(
            grid=((rows + rows_per_block - 1) // rows_per_block, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch
