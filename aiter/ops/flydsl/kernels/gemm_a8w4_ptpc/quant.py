# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-token FP8 E4M3FNUZ quantization of the BF16 activation of the gfx942 A8W4 GEMM.

``scale = amax / 240`` (IEEE f32 division; an all-zero row gets scale 1) and
``q = RNE(x / scale)`` with the IEEE quotient (FNUZ has no -0: it is stored as 0): bit-exact with
``(x.float() / scale).to(torch.float8_e4m3fnuz)``, except in rows that contain +-Inf. There
``scale`` is Inf and the residual of the correction below is NaN, so every element is written
as NaN (0x80), while torch gives 0 for the row's finite elements (both give NaN for the Inf
elements, so a GEMM output row is NaN either way).

``QuantConfig`` splits the work: a CTA quantizes ``rows`` rows and ``lanes`` lanes share a
row. Load ``v`` of lane ``sub`` reads the eight BF16 at 16-byte unit ``sub + v * lanes`` of
the row and its FP8 store writes eight bytes at unit ``sub + v * lanes``: every load and
store instruction covers a contiguous run of ``16 * lanes`` / ``8 * lanes`` bytes. With
``lanes <= 64`` a wave holds ``64 / lanes`` rows; with ``lanes`` a multiple of 64 a row spans
``lanes / 64`` waves, which combine their maxima through LDS behind one barrier.
``quant_config`` runs up to 128 rows one row per CTA over ``K / 8`` lanes (one load per lane:
the shortest chain from the load to the store) and more rows one wave per CTA over 64 lanes
per row (several loads in flight per lane, no barrier, stores written through L2).

``amax`` is the unsigned maximum of the BF16 bit patterns with the sign cleared, two per
``v_pk_max_u16``: it orders like the magnitudes and propagates NaN as torch's ``amax`` does.
The lanes of a row combine theirs with DPP moves (quad permutations, row mirrors and row
broadcasts) and ``v_readlane``, without an LDS round trip per step.

The quotient uses ``y = 1 / scale`` (IEEE, once per row) and one FMA correction
``q += (x - q * scale) * y`` (Markstein), as packed f32 math (two values per instruction).
Checked for every BF16 ``amax`` and every BF16 ``x`` with ``|x| <= amax``: the f32 result
differs from the IEEE quotient only below 2**-50, with subnormal operands, and rounds to the
same FP8 everywhere. Rows with ``scale < 2**-64``, where the residual could underflow, divide
per element instead. No clamp precedes the conversion: ``|x| / scale`` is at most the float
just above 240 for a normal ``scale``, about 240.003 for a subnormal one and about 240.06
when ``amax`` itself is subnormal (``scale`` keeps fewer bits), and ``v_cvt_pk_fp8_f32``
rounds anything below 248 to 240 like the reference does; NaN stays NaN.
"""

import functools
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..dpp_utils import update_dpp_i32
from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor

FP8_MAX = 240.0
_SMALL_SCALE = 2.0**-64
# CDNA3 buffer cache policy sc0 | sc1: system-scope (write-through) stores.
_SC0_SC1 = 0x11
_WAVE = 64
_MAX_THREADS = 1024
# Rows up to this run one row per CTA over K / 8 lanes (when K % 512 == 0).
_WIDE_MAX_ROWS = 128


class QuantConfig(NamedTuple):
    """``lanes`` lanes per row (a power of two up to 64 or a multiple of 64), ``rows`` per CTA."""

    lanes: int = 64
    rows: int = 4


def _narrow_lanes(cols):
    for lanes in (64, 32, 16, 8):
        if cols % (8 * lanes) == 0:
            return lanes
    return None


def quant_config(rows, cols):
    """The ``QuantConfig`` of a BF16 [rows, cols] activation, ``cols % 64 == 0``."""
    units = cols // 8
    if rows <= _WIDE_MAX_ROWS and units % _WAVE == 0 and units <= _MAX_THREADS:
        return QuantConfig(units, 1)
    lanes = _narrow_lanes(cols)
    return QuantConfig(lanes, _WAVE // lanes)


def _config_supported(cols, cfg):
    lanes, rows = cfg
    units = cols // 8
    threads = lanes * rows
    return (
        cols % 64 == 0
        and lanes > 0
        and units % lanes == 0
        and (lanes in (8, 16, 32, 64) or (lanes % _WAVE == 0 and rows == 1))
        and threads % _WAVE == 0
        and threads <= _MAX_THREADS
    )


def quant_kernel_name(cols, cfg):
    return format_kernel_name(f"gemm_a8w4_ptpc_quant_k{cols}_l{cfg.lanes}_r{cfg.rows}")


def _pack_fp8x4(q):
    """4 f32 (|q| < 248) -> one dword of FP8 E4M3FNUZ, RNE (-0 and values rounding to 0 give 0x00)."""
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8
    # has no LLVM translation); the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
    return fx.Uint32(fx.rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], packed, True))


def _dpp_max(v, ctrl, row_mask=0xF):
    return fx.max(v, fx.Int32(update_dpp_i32(v, v, ctrl, row_mask, 0xF, False)))


def _group_max(v, lanes):
    """Max of the non-negative i32 ``v`` over each aligned group of ``lanes`` lanes (8 .. 64).

    ``shuffle_xor`` lowers to ds_swizzle / ds_bpermute, an LDS round trip per step (ISA);
    DPP moves and readlane (raw rocdl: FlyDSL 0.5.3 has no typed DPP or wave reduction)
    take 0.05-0.15 us off a quantization of up to 256 rows (measured on MI308X).
    """
    v = _dpp_max(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    v = _dpp_max(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    v = _dpp_max(v, 0x141)  # row_half_mirror
    if const_expr(lanes >= 16):
        v = _dpp_max(v, 0x140)  # row_mirror
    if const_expr(lanes >= 32):
        v = _dpp_max(v, 0x142, 0xA)  # row_bcast:15 into rows 1, 3
    if const_expr(lanes == 32):
        lo = fx.Int32(fx.rocdl.readlane(T.i32, v.ir_value(), 31))
        hi = fx.Int32(fx.rocdl.readlane(T.i32, v.ir_value(), 63))
        v = (fx.thread_idx.x % 64 < 32).select(lo, hi)
    if const_expr(lanes == 64):
        v = _dpp_max(v, 0x143, 0xC)  # row_bcast:31 into rows 2, 3
        v = fx.Int32(fx.rocdl.readlane(T.i32, v.ir_value(), 63))
    return v


def _max_u16x2(vals):
    """Pairwise-tree unsigned maximum of 2 x u16 vectors (one v_pk_max_u16 per pair)."""
    while len(vals) > 1:
        tail = [vals[-1]] if len(vals) % 2 else []
        vals = [fx.max(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)] + tail
    return vals[0]


@functools.cache
def compile_quant_x(cols, device_index, cfg=None):
    """Launcher ``(x, q, scale, rows, stream)`` of the per-token quantization.

    BF16 ``x`` [rows, cols] -> FP8 ``q`` [rows, cols] and f32 ``scale`` [rows], passed as raw
    ``fx.Pointer``; ``cols % 64 == 0``. ``cfg`` is a ``QuantConfig`` (default: the large-M
    one). ``device_index`` only keys the cache (a compiled handle launches on one GPU).
    """
    del device_index
    if cfg is None:
        cfg = quant_config(_WIDE_MAX_ROWS + 1, cols)
    if not _config_supported(cols, cfg):
        raise ValueError(f"cols={cols} must be a multiple of 64 with a valid {cfg}")
    lanes, rows_per_block = cfg
    # The large-M configuration (a wave or less per row) writes through L2: a GEMM leaves
    # the previous call's FP8 rows in several XCDs' L2s, and cached stores to them are slow
    # (tp4_qkvz M = 2048: 10.4 -> 7.1 us). For one row per CTA the cached stores are
    # 0.05-0.1 us faster (measured on MI308X).
    store_policy = _SC0_SC1 if lanes <= _WAVE else 0
    units_per_row = cols // 8
    loads = units_per_row // lanes
    threads = lanes * rows_per_block
    row_waves = lanes // _WAVE
    wave_lanes = min(lanes, _WAVE)
    row_bytes = cols * 2
    n_waves = threads // _WAVE

    @fx.struct
    class SharedStorage:
        wave_max: fx.Array[fx.Uint32, max(n_waves, 4), 16]

    @flyc.kernel(name=quant_kernel_name(cols, cfg), known_block_size=[threads, 1, 1])
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

        unit0 = local * units_per_row + sub
        raw = [
            buf_copy_load(x_buf, unit0 + v * lanes, elem=fx.Uint32, unit_elems=4)
            for v in range_constexpr(loads)
        ]
        # amax on the BF16 bit patterns: |x| as unsigned 16-bit integers orders like the
        # values and puts NaN above Inf (NaN propagates, as torch's amax does); two per
        # dword (v_pk_max_u16) instead of a NaN-propagating float max (three VALU each).
        packed = _max_u16x2(
            [
                Vec.from_elements(
                    [raw[v][j] & fx.Uint32(0x7FFF7FFF)], fx.Uint32
                ).bitcast(fx.Uint16)
                for v in range_constexpr(loads)
                for j in range_constexpr(4)
            ]
        ).bitcast(fx.Uint32)[0]
        amax_bits = fx.max(packed & fx.Uint32(0xFFFF), packed >> 16)
        amax_bits = fx.Uint32(_group_max(fx.Int32(amax_bits), wave_lanes))
        if const_expr(row_waves > 1):
            # The row's waves publish their maxima and take the largest.
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            wave_max = fx.recast_iter(fx.Uint32, lds.wave_max.ptr)
            if tid % _WAVE == 0:
                fx.ptr_store(amax_bits, wave_max + tid // _WAVE)
            gpu.barrier()
            first = local * row_waves
            for i in range_constexpr(row_waves):
                amax_bits = fx.max(
                    amax_bits, fx.Uint32(fx.ptr_load(wave_max + first + i))
                )
        amax = (amax_bits << 16).bitcast(fx.Float32)
        scale = amax / FP8_MAX
        scale = (scale == 0.0).select(fx.Float32(1.0), scale)
        if sub == 0:
            buf_copy_store(s_buf, local, scale, elem=fx.Float32)

        q_unit0 = local * units_per_row + sub

        def emit(quotient):
            # quotient: 2 f32 -> 2 f32 (packed math).
            for v in range_constexpr(loads):
                dwords = []
                for d in range_constexpr(2):
                    q = []
                    for i in range_constexpr(2):
                        w = fx.Uint32(raw[v][2 * d + i])
                        pair = Vec.from_elements(
                            [w << 16, w & fx.Uint32(0xFFFF0000)], fx.Uint32
                        ).bitcast(fx.Float32)
                        q.append(quotient(pair))
                    dwords.append(_pack_fp8x4([q[0][0], q[0][1], q[1][0], q[1][1]]))
                buf_copy_store(
                    q_buf,
                    q_unit0 + v * lanes,
                    Vec.from_elements(dwords, fx.Uint32),
                    elem=fx.Uint32,
                    unit_elems=2,
                    cache_modifier=store_policy,
                )

        if scale >= fx.Float32(_SMALL_SCALE):
            y = 1.0 / scale
            y2 = Vec.from_elements([y, y], fx.Float32)
            s2 = Vec.from_elements([scale, scale], fx.Float32)

            def markstein(v):
                q = v * y2
                return Vec(fx.fma(fx.fma(-q, s2, v), y2, q))

            emit(markstein)
        else:
            s2 = Vec.from_elements([scale, scale], fx.Float32)
            emit(lambda v: v / s2)

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
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
