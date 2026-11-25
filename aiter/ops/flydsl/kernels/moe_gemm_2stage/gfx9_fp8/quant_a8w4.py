# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-row FP8 E4M3FNUZ quantization for the A8W4 MoE: ``scale = amax / 240``, ``q = RNE(x / scale)``.

Matches ``torch`` exactly (IEEE quotient, round-to-nearest-even conversion, ``-0 -> +0``), so the
activation quantization points agree bit for bit with the fp32 reference of
``op_tests/test_flydsl_moe_a8w4.py``.

A row is split into 16-byte chunks shared by ``L`` lanes: lane ``l`` takes chunks ``l, l + L, ...``
(at least 4 when the row allows), so every load / store instruction covers contiguous memory. The quotient
``x / scale`` uses ``y = 1 / scale`` (IEEE, once per row) and two FMA corrections
``q += (x - q * scale) * y``: the first makes ``q`` faithful, the second then yields the correctly
rounded quotient (Markstein). Rows with ``scale < 2**-64``, where the residual could underflow,
divide per element instead. No clamp to +-240 is needed: ``|x| <= amax`` and ``scale = RN(amax / 240)``
bound the quotient by ``240 * (1 + 2**-23)``, which FP8 rounds to 240. Inputs must be finite (the
row maximum ignores NaN).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh

_ESIZE = {"bf16": 2, "f32": 4}
_SMALL_SCALE = 2.0**-64


def _lanes_per_row(chunks):
    lanes = 64
    while lanes > 4 and (chunks % lanes or chunks // lanes < 4):
        lanes //= 2
    return lanes


def _buffer_rsrc(ptr, nbytes):
    view = fx.make_view(
        fx.recast_iter(fx.Uint8, fxh._as_ptr(ptr)), fx.make_layout(1, 1)
    )
    tensor = fx.rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=nbytes)
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))


def _store(rsrc, value, offset):
    """Raw buffer store; bytes past the buffer end are dropped."""
    # Raw buffer store: the typed fx.copy store takes its offset from a view layout; callers
    # compute byte offsets and send the lanes that must not write past the buffer end, where
    # the hardware range check drops the store (no branch).
    rocdl_dialect.RawPtrBufferStoreOp(
        _raw(value),
        rsrc,
        offset.ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    )


def _pack_fp8x4(q):
    """4 f32 (|q| < 248) -> one dword of FP8 E4M3FNUZ (RNE); 0x80 (-0, NaN in FNUZ) becomes 0."""
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8
    # has no LLVM translation); the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    packed = rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], packed, True))
    low = packed & fx.Uint32(0x7F7F7F7F)
    # Bit 7 of each byte of low + 0x7F is set iff its low 7 bits are not all zero (no carries between bytes).
    return ((low + fx.Uint32(0x7F7F7F7F)) & packed) | low


def compile_row_quant_fp8(cols, in_dtype, fill=0):
    """Launcher ``(x, q, scale, aux, rows, stream)``: x ``in_dtype`` [rows, cols] -> q FP8 [rows, cols], scale f32 [rows].

    ``in_dtype`` is ``"bf16"`` or ``"f32"``; ``cols % 64 == 0``. All-zero rows get scale 1. With ``fill`` > 0
    the kernel also sets int32 ``aux`` [rows, fill] to -1 (otherwise ``aux`` is not accessed).
    Not cached here: ``aiter.ops.flydsl.moe_a8w4`` caches the launcher per device.
    """
    assert cols % 64 == 0, f"cols={cols} must be a multiple of 64"
    esize = _ESIZE[in_dtype]
    per_chunk = 16 // esize
    chunks = cols // per_chunk
    lanes = _lanes_per_row(chunks)
    steps = chunks // lanes
    rows_per_block = 256 // lanes
    shuffles = [lanes >> (i + 1) for i in range(lanes.bit_length() - 1)]

    @flyc.kernel
    def row_quant_fp8(
        p_in: fx.Pointer,
        p_out: fx.Pointer,
        p_scale: fx.Pointer,
        p_aux: fx.Pointer,
        rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        sub = tid % lanes
        row = fx.Int32(gpu.block_idx.x) * rows_per_block + tid // lanes
        in_rsrc = _buffer_rsrc(p_in, fx.Int64(rows) * (cols * esize))
        out_rsrc = _buffer_rsrc(p_out, fx.Int64(rows) * cols)
        scale_rsrc = _buffer_rsrc(p_scale, fx.Int64(rows) * 4)
        if const_expr(fill):
            aux_rsrc = _buffer_rsrc(p_aux, fx.Int64(rows) * (fill * 4))
            for i in range_constexpr((fill + lanes - 1) // lanes):
                j = sub + i * lanes
                _store(
                    aux_rsrc,
                    fx.Int32(-1),
                    (j < fill).select((row * fill + j) * 4, fx.Int32(0x7FFFFFF0)),
                )
        in_base = row * (cols * esize) + sub * 16
        values = []
        for c in range_constexpr(steps):
            # Raw 16-byte buffer load at a per-lane byte offset (see _store); rows past ``rows``
            # read zeros from the range check.
            raw = Vec(
                rocdl_dialect.RawPtrBufferLoadOp(
                    ir.VectorType.get([4], fx.Int32.ir_type),
                    in_rsrc,
                    (in_base + c * (lanes * 16)).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
                ).result
            ).bitcast(fx.Uint32)
            for j in range_constexpr(4):
                word = raw[j]
                if const_expr(in_dtype == "bf16"):
                    values.append((word << 16).bitcast(fx.Float32))
                    values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
                else:
                    values.append(word.bitcast(fx.Float32))
        amax = abs(values[0])
        for j in range_constexpr(1, steps * per_chunk):
            amax = fx.maxnumf(amax, abs(values[j]))
        for i in range_constexpr(len(shuffles)):
            amax = fx.maxnumf(amax, amax.shuffle_xor(shuffles[i], 64))
        scale = amax / 240.0
        scale = (scale == 0.0).select(fx.Float32(1.0), scale)
        out_base = row * cols + sub * per_chunk
        _store(scale_rsrc, scale, (sub == 0).select(row * 4, fx.Int32(0x7FFFFFFF)))

        def emit(quotient):
            for c in range_constexpr(steps):
                dwords = []
                for w in range_constexpr(per_chunk // 4):
                    first = c * per_chunk + 4 * w
                    q = [quotient(values[first + i]) for i in range_constexpr(4)]
                    dwords.append(_pack_fp8x4(q))
                offset = out_base + c * (lanes * per_chunk)
                if const_expr(per_chunk == 8):
                    _store(out_rsrc, Vec.from_elements(dwords, fx.Uint32), offset)
                else:
                    _store(out_rsrc, dwords[0], offset)

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
        p_in: fx.Pointer,
        p_out: fx.Pointer,
        p_scale: fx.Pointer,
        p_aux: fx.Pointer,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        row_quant_fp8(p_in, p_out, p_scale, p_aux, rows).launch(
            grid=((rows + rows_per_block - 1) // rows_per_block, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
