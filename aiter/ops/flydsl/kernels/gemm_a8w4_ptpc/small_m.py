# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 small-M dense A8W4 GEMM: BF16 X x fd2 MXFP4 W -> BF16 Y = X W^T, M <= 32, one launch.

Each CTA quantizes the M token rows itself (per token, FP8 E4M3FNUZ, ``scale = amax / 240``,
bit-exact with the generic path's ``_quant_per_token``) into LDS, so there is no separate
quantization kernel. Decode-style after the pyhip A8W4 MoE decode kernels (``a8w4_decode.py``,
tag moe-ptpc-a8w4-r2-M11), made dense: a CTA reads its weight rows once for all M tokens, and
its waves split K and add their partial sums in LDS.

Three kernels; ``small_m_config`` picks one per (M, N, K), K % 512 == 0:

* decode (M <= 8): block pairs. One 16-byte load per lane covers two adjacent 64-K blocks;
  MFMA 16x16x32 FP8 takes X as A (8 tokens, in rows t and 8 + t) and W as B, and lanes l and
  l ^ 40 hold the two K halves of a channel (``compile_gemm_a8w4_ptpc_decode``).
* dec2 (widest shapes, M <= 4): the decode mapping with split roles. One CTA per CU; its
  first ``xw`` waves only quantize X and stage the deltas, publishing them through an LDS
  counter, and the weight waves wait for that counter instead of a CTA barrier, so no
  weight wave sits behind a barrier while its loads are still queued
  (``compile_gemm_a8w4_ptpc_decode2``).
* tile (M <= 32): MFMA 16x16x32 FP8 takes W as A (16 channels) and X as B (16 tokens per
  token tile, ``tq`` <= 2 tiles); 8-byte weight loads and one LUT per 128-K step
  (``compile_gemm_a8w4_ptpc_tile``). The FP8 token rows go to LDS in ``kp`` K windows so
  32 rows fit.

decode and dec2 hold the FP8 token rows of all of K in LDS: where that and the CTA's deltas
and K-part sums exceed the 64 KiB LDS (M 5..8 from K = 7168, or 5120 on the widest N; M <= 4
from K = 9216), the next kernel that fits runs instead (dec2 -> decode -> tile -> generic path).

Tile: CTA ``i`` of ``num_wg`` owns the 16-row blocks ``[i nb / num_wg, (i + 1) nb / num_wg)``
(at most ``rb``, ``nb = N / 16``) x all M tokens x all of K. Its ``rb * ks`` waves each take
one row block slot and ``1 / ks`` of K (of every K window); a slot past the CTA's run computes
on a neighbouring block and stores nothing.

Schedule: grid ``num_wg``, at most about one CTA per CU on the measured part; no split-K
across CTAs.

Steps (decode, tile): nothing is pipelined. Each wave issues all its loads first, then:

    phase     VMEM                LDS                       VALU / MFMA
    prologue  X rows, D, W, s     -                         -
    quant     -                   X (FP8), x_scale, D       amax, X / scale -> FP8
    barrier
    step s    -                   D(s), X(s)                LUT(s); decode W(s); acc += W X
    epilogue  -                   K-part sums               sum, scale, BF16 store

(the tile kernel with ``kp`` > 1 repeats quant / barrier / steps per K window, behind a
barrier before each window's LDS writes; in dec2 the X waves do prologue and quant, the weight
waves the steps, and the LDS counter replaces the first barrier).

Naming: M = tokens, N = channels, K = reduction. ``rb`` row blocks per CTA, ``ks`` K parts,
``mcap`` decode token capacity, ``tq`` 16-token tiles, ``kp`` K windows, ``s`` 128-K step,
``h`` half of a step, ``c`` MFMA of a half; ``raws`` FP4 dwords, ``deltas`` delta dwords,
``part`` the LDS partial sums.

Rules, to keep when editing a kernel body:
1. Loads are issued in consumption order, separated by ``sched_barrier``: ``vmcnt`` is in
   order, so moving one makes the first steps wait for later loads.
2. Every wave reaches every barrier: only wave-uniform conditions, around LDS writes and
   stores; no early exit (dec2's roles reach the same number of barriers).
3. ``part`` is summed in K-part order: the result does not depend on wave timing.
4. X rows at or past M are not written to LDS; MFMA outputs of those tokens are never stored.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..communication_ops_utils import wave_uniform_i32
from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import layout

MAX_M = 32
FP8_MAX = 240.0
# Below this scale the quotient's FMA residual could underflow: divide per element instead.
_SMALL_SCALE = 2.0**-64
DECODE_MAX_M = 8
_LDS_BYTES = 65536
# CU count of the part the configurations were measured on (gfx942 MI308X).
_CUS = 80


def decode_kernel_name(rb, ks, mcap, num_wg):
    return format_kernel_name(f"gemm_a8w4_ptpc_decode_rb{rb}_ks{ks}_m{mcap}_g{num_wg}")


def tile_kernel_name(rb, ks, tq, kp, num_wg):
    return format_kernel_name(f"gemm_a8w4_ptpc_tile_rb{rb}_ks{ks}_t{tq}_kp{kp}_g{num_wg}")


def _tile_windows(k, kp):
    """512-K units of each of the ``kp`` K windows of the tile kernel."""
    units = k // 512
    return [units // kp + (1 if p < units % kp else 0) for p in range(kp)]


def _tile_lds(n, k, rb, ks, tq, kp):
    """(token rows, deltas, K-part sums, token scales) LDS bytes of the tile kernel."""
    del n
    x = 16 * tq * (max(_tile_windows(k, kp)) * 512 + 16)
    delta = rb * layout.delta_dwords(k) * layout.D_BLOCK_BYTES
    return x, delta, rb * (ks - 1) * tq * 64 * 16, 16 * tq * 4


def _decode_lds(k, rb, ks, mcap, xw=0):
    """LDS bytes of the decode kernel's ``SharedStorage`` (``xw`` = 0) or of dec2's (``xw`` X
    waves): token rows of all of K, deltas, K-part sums, token scales and, in dec2, the
    publish counter, each field 16-byte aligned."""
    tq = (mcap + 7) // 8
    fields = [
        mcap * (k + 16),
        rb * layout.delta_dwords(k) * layout.D_BLOCK_BYTES,
        4 * max(1, rb * (ks - 1) * 16 * 2 * tq * 4),
        4 * max(4, mcap),
    ]
    if xw:
        fields.append(16)
    return sum(-(-b // 16) * 16 for b in fields)


def small_m_config(m, n, k):
    """Small-M kernel for BF16 ``x`` [m, k] x weight [n, k], or None (the generic path).

    ``("decode", rb, ks, mcap, 1, num_wg)`` for M <= 8 (M rounded up to ``mcap``),
    ``("dec2", rb, ks, mcap, xw, num_wg)`` for the widest shapes at M <= 4 (``xw`` X waves),
    ``("tile", rb, ks, tq, kp, num_wg)`` for M <= 32 (``tq`` 16-token tiles, ``kp`` K windows).
    All quantize X in the CTA, so K % 512 == 0. ``rb`` row blocks x ``ks`` K parts per CTA
    and ``num_wg`` CTAs, measured on the target shapes. A kernel whose LDS would exceed
    64 KiB is skipped for the next one: dec2, decode, tile, then None.
    """
    if not 0 < m <= MAX_M or n % 64 or k % 512:
        return None
    blocks = n // 16
    if m <= DECODE_MAX_M:
        mcap = next(c for c in (1, 2, 4, 8) if m <= c)
        # Wide shapes: one CTA per CU, one X wave per token row next to two K parts per row
        # block, as long as that fits in 16 waves.
        rb = -(-blocks // _CUS)
        if (
            blocks > 4 * _CUS
            and 2 * rb + mcap <= 16
            and mcap <= 4
            and _decode_lds(k, rb, 2, mcap, mcap) <= _LDS_BYTES
        ):
            return "dec2", rb, 2, mcap, mcap, _CUS
        # Every CTA quantizes all M rows and reads them for its weights, so wider CTAs pay
        # off as M grows; CTAs beyond one per CU only pay where two fit on a CU.
        if blocks > 6 * _CUS:
            rb, ks, num_wg = (8, 2, -(-blocks // 8)) if mcap == 8 else (4, 2, -(-blocks // 4))
        elif blocks > 4 * _CUS:
            rb, ks, num_wg = 6, 2, _CUS
        elif blocks > 3 * _CUS:
            rb, ks, num_wg = 4, 4 if mcap == 8 else 2, -(-blocks // 4)
        elif blocks > 2 * _CUS:
            rb, ks, num_wg = 3, 4, _CUS
        else:
            rb, ks, num_wg = 2, 4, -(-blocks // 2)
        if _decode_lds(k, rb, ks, mcap) <= _LDS_BYTES:
            return "decode", rb, ks, mcap, 1, num_wg
    tq = 1 if m <= 16 else 2
    if blocks > 4 * _CUS:
        rb, ks = 8, 2
    elif blocks > 2 * _CUS:
        rb, ks = 4, 4
    else:
        rb, ks = 2, 4
    for kp in range(1, k // 512 + 1):
        x, delta, part, xscale = _tile_lds(n, k, rb, ks, tq, kp)
        if x + delta + xscale <= _LDS_BYTES and part <= max(x, _LDS_BYTES - x - delta - xscale):
            return "tile", rb, ks, tq, kp, -(-blocks // rb)
    return None


def small_m_kernel_name_of(config):
    kind, rb, ks, tokens, kp, num_wg = config
    if kind == "decode":
        return decode_kernel_name(rb, ks, tokens, num_wg)
    if kind == "dec2":
        return decode2_kernel_name(rb, ks, kp, tokens, num_wg)
    return tile_kernel_name(rb, ks, tokens, kp, num_wg)


def compile_small_m(config, n, k, device_index):
    """Launcher ``(x, wq, wd, ws, y, M, stream)`` of ``small_m_config``'s kernel."""
    kind, rb, ks, tokens, kp, num_wg = config
    if kind == "decode":
        return compile_gemm_a8w4_ptpc_decode(n, k, rb, ks, tokens, num_wg, device_index)
    if kind == "dec2":
        return compile_gemm_a8w4_ptpc_decode2(n, k, rb, ks, kp, tokens, num_wg, device_index)
    return compile_gemm_a8w4_ptpc_tile(n, k, rb, ks, tokens, kp, num_wg, device_index)


def _lds_bytes(base, byte_off, width):
    """``width`` bytes of LDS at byte offset ``byte_off`` from ``base`` (any element type)."""
    return fx.Tensor(
        fx.make_view(
            fx.add_offset(fx.recast_iter(fx.Uint8, base), fx.make_int_tuple(byte_off)),
            fx.make_layout(width, 1),
        )
    )


def _decode_pair(raw, lut):
    """The two fd2 dwords of a lane (K 0..7, 8..15) -> the FP8 operands of its two MFMAs.

    The nibble shifts of both dwords are single 64-bit shifts (bits crossing the dword
    boundary are masked off), as ``layout.fp4x8_to_fp8`` per dword otherwise.
    """
    lo, hi = lut
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    wide = Vec.from_elements([raw[0], raw[1]], fx.Uint32).bitcast(fx.Uint64)[0]
    right = Vec.from_elements([wide >> 4], fx.Uint64).bitcast(fx.Uint32)
    left = Vec.from_elements([wide << 4], fx.Uint64).bitcast(fx.Uint32)
    ops = []
    for c in range_constexpr(2):
        r = fx.Uint32(raw[c])
        even = (fx.Uint32(left[c]) & sign) | layout._perm(hi, lo, r & mag)
        odd = (r & sign) | layout._perm(hi, lo, fx.Uint32(right[c]) & mag)
        ops.append(Vec.from_elements([even, odd], fx.Uint32).bitcast(fx.Float8E4M3FNUZ))
    return ops


def _pack_fp8x4(q):
    """4 f32 -> one dword of FP8 E4M3FNUZ, RNE (-0 and values rounding to 0 give 0x00)."""
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8
    # has no LLVM translation); the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, q[0], q[1], fx.Int32(0), False)
    return fx.Uint32(fx.rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], packed, True))


def _row_scale(raw_units):
    """Per-token scale of one BF16 row held by a wave (``raw_units``: the lane's 16-byte units).

    Same arithmetic as the generic path's quantization: ``scale = amax / 240`` (1 for an
    all-zero row). ``amax`` is the integer maximum of the BF16 bit patterns with the sign
    cleared, which orders like ``|x|`` and keeps NaN.
    """
    words = [fx.Uint32(u[j]) for u in raw_units for j in range(4)]

    def mag16(w):
        return Vec.from_elements([w & fx.Uint32(0x7FFF7FFF)], fx.Uint32).bitcast(fx.Uint16)

    acc = mag16(words[0])
    for w in words[1:]:
        acc = fx.max(acc, mag16(w))
    amax_bits = fx.Uint32(fx.max(acc[0], acc[1])) << 16
    for sh in (32, 16, 8, 4, 2, 1):
        amax_bits = fx.max(amax_bits, gpu.shuffle_xor(amax_bits, sh, 64))
    scale = amax_bits.bitcast(fx.Float32) / FP8_MAX
    return (scale == 0.0).select(fx.Float32(1.0), scale)


def _quotient(scale):
    """``v -> v / scale`` (IEEE) on 2 f32 for ``scale >= _SMALL_SCALE``: reciprocal and two
    FMA corrections, as packed f32 math."""
    y = 1.0 / scale
    y2 = Vec.from_elements([y, y], fx.Float32)
    s2 = Vec.from_elements([scale, scale], fx.Float32)

    def quotient(v):
        q = v * y2
        q = Vec(fx.fma(fx.fma(-q, s2, v), y2, q))
        return Vec(fx.fma(fx.fma(-q, s2, v), y2, q))

    return quotient


def _quant_units(raw_units, units, quotient):
    """FP8 dword pairs ``RNE(quotient(x))`` of the lane's 16-byte BF16 ``units`` (8 values each).

    No clamp is needed: ``|x| / scale <= 240 (1 + 2**-8)`` rounds to at most 240.
    """
    out = []
    for u in units:
        dws = []
        for d in range(2):
            q = []
            for i in range(2):
                w = fx.Uint32(raw_units[u][2 * d + i])
                v = Vec.from_elements([w << 16, w & fx.Uint32(0xFFFF0000)], fx.Uint32)
                q.append(quotient(v.bitcast(fx.Float32)))
            dws.append(_pack_fp8x4([q[0][0], q[0][1], q[1][0], q[1][1]]))
        out.append(Vec.from_elements(dws, fx.Uint32))
    return out


@functools.cache
def compile_gemm_a8w4_ptpc_decode(n, k, rb, ks, mcap, num_wg, device_index):
    """Launcher ``(x, wq, wd, ws, y, M, stream)`` of the decode kernel, M <= mcap.

    ``x`` BF16 [M, k], ``wq`` / ``wd`` / ``ws`` the fd2 layout, ``y`` BF16 [M, n]; raw
    ``fx.Pointer``. The ``num_wg`` CTAs split the N / 16 row blocks into contiguous runs of
    ``rb`` or fewer (CTA ``i`` owns ``[i nb / num_wg, (i + 1) nb / num_wg)``); the ``rb * ks``
    waves of a CTA each take one row block slot and ``1 / ks`` of K. The CTA quantizes the
    token rows into LDS (FP8) and stages its delta bytes there.

    Block pairs: one 16-byte load per lane covers two adjacent 64-K blocks of the row block;
    lane ``l`` holds rows ``2 (l % 8) + h`` of lane group ``b = (l % 32) // 8`` of block
    ``l // 32`` and is MFMA operand B (column ``l % 16``, K chunk ``l // 16``). Operand A of
    lane ``l`` is token ``8 q + l % 8`` at the K of the same block and lane group, so output
    row ``t`` pairs with the columns ``j`` of ``j // 8 == t // 8``: rows ``t`` and ``8 + t``
    both hold token ``t``, columns ``j`` and ``8 + j`` the two halves of the K groups of
    channel ``2 j + h``. Lane ``l`` (``l % 16 < 8``, ``l < 32``) adds lane ``l ^ 40`` and holds
    tokens ``4 (l // 16) + v`` of channel ``2 (l % 8) + h``.
    """
    del device_index
    pairs = k // 128
    waves = rb * ks
    nb = n // 16
    if (
        n % 16
        or k % 256
        or pairs % ks
        or waves > 16
        or k % 512
        or not 0 < num_wg <= nb
        or -(-nb // num_wg) > rb
    ):
        raise ValueError(
            f"decode A8W4 GEMM: unsupported N={n} K={k} rb={rb} ks={ks} mcap={mcap} "
            f"num_wg={num_wg}"
        )
    if _decode_lds(k, rb, ks, mcap) > _LDS_BYTES:
        raise ValueError(f"decode A8W4 GEMM: LDS too small for K={k} rb={rb} ks={ks} mcap={mcap}")
    fp8 = fx.Float8E4M3FNUZ
    ppw = pairs // ks
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    threads = 64 * waves
    tq = (mcap + 7) // 8
    # LDS token rows: the first mcap tokens (rows >= M are not written). Lanes of tokens
    # >= mcap read row mcap - 1: an MFMA output row depends only on its own A row, and the
    # rows of tokens >= M are never stored.
    rows = mcap
    x_stride = k + 16
    # In-CTA quantization: wave w quantizes token rows w + waves * j, lane l holding the
    # 16-byte BF16 units l + 64 u of the row.
    q_rows = (mcap + waves - 1) // waves
    q_units = k // 512
    d_units = rb * kq_total * layout.D_BLOCK_BYTES // 16
    d_loads = (d_units + threads - 1) // threads
    # K-part partial sums: 16 lanes x (2 channels x tq token sets x 4 tokens) per wave.
    part_floats = 16 * 2 * tq * 4
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Uint8, rows * x_stride, 16]
        delta: fx.Array[fx.Uint8, d_units * 16, 16]
        part: fx.Array[fx.Float32, max(1, rb * (ks - 1) * part_floats), 16]
        xscale: fx.Array[fx.Float32, max(4, mcap), 16]

    @flyc.kernel(name=decode_kernel_name(rb, ks, mcap, num_wg), known_block_size=[threads, 1, 1])
    def gemm_a8w4_ptpc_decode_kernel(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = wave_uniform_i32(tid // 64)
        r_local = wave % rb
        k_part = wave // rb
        blk = fx.block_idx.x
        rb0 = (blk * nb) // num_wg
        valid = r_local < ((blk + 1) * nb) // num_wg - rb0
        rbg = rb0 + r_local
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        x_buf = ptr_buf_tensor(p_x, elem=fx.Uint32, unit_elems=4, num_records_bytes=M * (k * 2))
        w_buf = ptr_buf_tensor(p_wq, elem=fx.Uint32, unit_elems=4, num_records_bytes=w_bytes)
        d_buf = ptr_buf_tensor(p_wd, elem=fx.Uint32, unit_elems=4, num_records_bytes=d_bytes)

        # ======== Prologue: token rows and deltas of the CTA, then the wave's weights ========
        q_vals = [
            [
                buf_copy_load(
                    x_buf,
                    (wave + waves * j) * (k // 8) + lane + 64 * u,
                    elem=fx.Uint32,
                    unit_elems=4,
                )
                for u in range(q_units)
            ]
            for j in range(q_rows)
        ]
        d_vals = [
            buf_copy_load(
                d_buf,
                rb0 * (kq_total * layout.D_BLOCK_BYTES // 16) + tid + threads * i,
                elem=fx.Uint32,
                unit_elems=4,
            )
            for i in range(d_loads)
        ]
        rocdl.sched_barrier(0)
        s0 = k_part * ppw
        raws = [
            buf_copy_load(
                w_buf, (rbg * kb_total + 2 * (s0 + i)) * 32 + lane, elem=fx.Uint32, unit_elems=4
            )
            for i in range(ppw)
        ]
        rocdl.sched_barrier(0)
        # Epilogue operands: channels 2 (l % 8) + h, tokens 8 q + 4 ((l // 16) % 2) + v.
        s_buf = ptr_buf_tensor(p_ws, elem=fx.Float32, unit_elems=2, num_records_bytes=n * 4)
        w_scale = buf_copy_load(s_buf, rbg * 8 + lane % 8, elem=fx.Float32, unit_elems=2)
        rocdl.sched_barrier(0)

        # ======== In-CTA per-token quantization: BF16 rows -> FP8 rows in LDS ========
        for j in range_constexpr(q_rows):
            row = wave + waves * j
            if row < M:
                scale = _row_scale(q_vals[j])
                if lane == 0:
                    _lds_bytes(lds.xscale.ptr, row * 4, 4).store(
                        Vec.from_elements([scale], fx.Float32).bitcast(fx.Uint8)
                    )
                if scale >= fx.Float32(_SMALL_SCALE):
                    dws = _quant_units(q_vals[j], range(q_units), _quotient(scale))
                    for u in range_constexpr(q_units):
                        _lds_bytes(lds.x.ptr, row * x_stride + (lane + 64 * u) * 8, 8).store(
                            dws[u].bitcast(fx.Uint8)
                        )
                else:
                    dws = _quant_units(q_vals[j], range(q_units), lambda v: v / scale)
                    for u in range_constexpr(q_units):
                        _lds_bytes(lds.x.ptr, row * x_stride + (lane + 64 * u) * 8, 8).store(
                            dws[u].bitcast(fx.Uint8)
                        )

        for i in range_constexpr(d_loads):
            u = tid + threads * i
            if const_expr(threads * (i + 1) <= d_units):
                _lds_bytes(lds.delta.ptr, u * 16, 16).store(d_vals[i].bitcast(fx.Uint8))
            else:
                if u < d_units:
                    _lds_bytes(lds.delta.ptr, u * 16, 16).store(d_vals[i].bitcast(fx.Uint8))
        gpu.barrier()
        x_scales = [
            Vec(
                _lds_bytes(
                    lds.xscale.ptr,
                    fx.min(fx.Int32(q * 8) + ((lane // 16) % 2) * 4, fx.Int32(max(4, mcap) - 4))
                    * 4,
                    16,
                ).load()
            ).bitcast(fx.Float32)
            for q in range(tq)
        ]

        # ======== Block pairs: decode W(s), acc += X(s) W(s) ========
        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        accs = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range(tq)] for _ in range(2)]
        for h in range_constexpr(2):
            for q in range_constexpr(tq):
                accs[h][q].store(Vec.filled(4, 0.0, fx.Float32))
        x_lane = (lane // 32) * 64 + ((lane % 32) // 8) * 16
        x_rows = [fx.min(fx.Int32(q * 8) + lane % 8, fx.Int32(rows - 1)) for q in range(tq)]
        d_lane = (r_local * kq_total) * layout.D_BLOCK_BYTES + ((lane % 32) // 16) * 64 + (
            lane % 8
        ) * 8
        # LDS reads of every pair first (X operands, delta dwords), so the per-pair chain is
        # only weight load -> decode -> MFMA.
        xa = [
            [
                Vec(
                    _lds_bytes(
                        lds.x.ptr, x_rows[q] * x_stride + x_lane + (s0 + i) * 128, 16
                    ).load()
                ).bitcast(fx.Uint32)
                for i in range(ppw)
            ]
            for q in range(tq)
        ]
        deltas = [
            Vec(
                _lds_bytes(
                    lds.delta.ptr, d_lane + ((s0 + i) // 2) * layout.D_BLOCK_BYTES, 8
                ).load()
            ).bitcast(fx.Uint32)
            for i in range(ppw)
        ]
        # Delta byte 2 (s % 2) + l // 32 of the dword, broadcast by the LUT's byte permute.
        sel_beta = fx.Uint32((lane // 32) * 0x01010101)
        for i in range_constexpr(ppw):
            s = s0 + i
            sel = sel_beta + fx.Uint32(s % 2) * fx.Uint32(0x02020202)
            a_ops = []
            for q in range_constexpr(tq):
                v = xa[q][i]
                per_c = []
                for c in range_constexpr(2):
                    frag = fx.make_rmem_tensor(8, fp8)
                    frag.store(
                        Vec.from_elements([v[2 * c], v[2 * c + 1]], fx.Uint32).bitcast(fp8)
                    )
                    per_c.append(frag)
                a_ops.append(per_c)
            raw = raws[i]
            for h in range_constexpr(2):
                lut = layout.fp8_lut(deltas[i][h], sel)
                ops = _decode_pair(
                    Vec.from_elements([raw[2 * h], raw[2 * h + 1]], fx.Uint32), lut
                )
                for c in range_constexpr(2):
                    b = fx.make_rmem_tensor(8, fp8)
                    b.store(ops[c])
                    for q in range_constexpr(tq):
                        fx.gemm(mma, accs[h][q], a_ops[q][c], b, accs[h][q])

        # ======== Epilogue: join the K halves (lane l ^ 40), add the K parts, BF16 ========
        sums = []
        for h in range_constexpr(2):
            for q in range_constexpr(tq):
                acc = Vec(accs[h][q].load())
                for v in range_constexpr(4):
                    sums.append(acc[v] + gpu.shuffle_xor(acc[v], 40, 64))
        useful = ((lane % 16) < 8) & (lane < 32)
        if const_expr(ks > 1):
            slot = (lane // 16) * 8 + lane % 8
            if (k_part > 0) & useful & valid:
                base = ((r_local * (ks - 1) + k_part - 1) * 16 + slot) * (2 * tq * 4) * 4
                for j in range_constexpr(2 * tq * 4):
                    _lds_bytes(lds.part.ptr, base + j * 4, 4).store(
                        Vec.from_elements([sums[j]], fx.Float32).bitcast(fx.Uint8)
                    )
            gpu.barrier()
            for p in range_constexpr(1, ks):
                base = ((r_local * (ks - 1) + p - 1) * 16 + slot) * (2 * tq * 4) * 4
                for j in range_constexpr(2 * tq * 4):
                    sums[j] = sums[j] + Vec(
                        _lds_bytes(lds.part.ptr, base + j * 4, 4).load()
                    ).bitcast(fx.Float32)[0]
        if (k_part == 0) & useful & valid:
            y_buf = ptr_buf_tensor(
                p_y, elem=fx.BFloat16, unit_elems=2, num_records_bytes=M * (n * 2)
            )
            for q in range_constexpr(tq):
                for v in range_constexpr(4):
                    token = q * 8 + (lane // 16) * 4 + v
                    out = Vec.from_elements(
                        [
                            (sums[(h * tq + q) * 4 + v] * w_scale[h] * x_scales[q][v]).to(
                                fx.BFloat16
                            )
                            for h in range_constexpr(2)
                        ],
                        fx.BFloat16,
                    )
                    buf_copy_store(
                        y_buf,
                        token * (n // 2) + rbg * 8 + lane % 8,
                        out,
                        elem=fx.BFloat16,
                        unit_elems=2,
                    )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_a8w4_ptpc_decode_kernel(p_x, p_wq, p_wd, p_ws, p_y, M).launch(
            grid=(num_wg, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def decode2_kernel_name(rb, ks, xw, mcap, num_wg):
    return format_kernel_name(f"gemm_a8w4_ptpc_dec_rb{rb}_ks{ks}_x{xw}_m{mcap}_g{num_wg}")


@functools.cache
def compile_gemm_a8w4_ptpc_decode2(n, k, rb, ks, xw, mcap, num_wg, device_index):
    """Launcher ``(x, wq, wd, ws, y, M, stream)`` of the role-split decode kernel, M <= mcap.

    Same block-pair MFMA mapping as ``compile_gemm_a8w4_ptpc_decode``. The first ``xw`` waves
    of a CTA load the token rows and the CTA's deltas, quantize the rows into LDS, stage the
    deltas there and publish both through an LDS counter; the other ``rb * ks`` waves issue
    all their weight loads at once and only wait for that counter before their first block
    pair, so no weight wave waits behind a CTA barrier while its loads are still queued.
    """
    del device_index
    pairs = k // 128
    wwaves = rb * ks
    waves = xw + wwaves
    nb = n // 16
    if (
        n % 16
        or k % 512
        or pairs % ks
        or waves > 16
        or not 0 < xw <= mcap
        or not 0 < num_wg <= nb
        or -(-nb // num_wg) > rb
    ):
        raise ValueError(
            f"decode A8W4 GEMM: unsupported N={n} K={k} rb={rb} ks={ks} xw={xw} mcap={mcap} "
            f"num_wg={num_wg}"
        )
    if _decode_lds(k, rb, ks, mcap, xw) > _LDS_BYTES:
        raise ValueError(
            f"decode A8W4 GEMM: LDS too small for K={k} rb={rb} ks={ks} xw={xw} mcap={mcap}"
        )
    fp8 = fx.Float8E4M3FNUZ
    ppw = pairs // ks
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    threads = 64 * waves
    tq = (mcap + 7) // 8
    rows = mcap
    x_stride = k + 16
    q_rows = (mcap + xw - 1) // xw
    q_units = k // 512
    d_units = rb * kq_total * layout.D_BLOCK_BYTES // 16
    d_loads = (d_units + 64 * xw - 1) // (64 * xw)
    part_floats = 16 * 2 * tq * 4
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Uint8, rows * x_stride, 16]
        delta: fx.Array[fx.Uint8, d_units * 16, 16]
        part: fx.Array[fx.Float32, max(1, rb * (ks - 1) * part_floats), 16]
        xscale: fx.Array[fx.Float32, max(4, mcap), 16]
        ready: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=decode2_kernel_name(rb, ks, xw, mcap, num_wg), known_block_size=[threads, 1, 1])
    def gemm_a8w4_ptpc_dec_kernel(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = wave_uniform_i32(tid // 64)
        blk = fx.block_idx.x
        rb0 = (blk * nb) // num_wg
        nrb = ((blk + 1) * nb) // num_wg - rb0
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ready_ptr = lds.ready.ptr
        # LDS starts undefined: zero the publish counter before any wave can read or add to it.
        if tid == 0:
            fx.generic_store(ready_ptr, fx.Int32(0))
        gpu.barrier()

        if wave < xw:
            # ======== X waves: token rows and the CTA's deltas -> LDS, then publish ========
            x_buf = ptr_buf_tensor(
                p_x, elem=fx.Uint32, unit_elems=4, num_records_bytes=M * (k * 2)
            )
            d_buf = ptr_buf_tensor(
                p_wd, elem=fx.Uint32, unit_elems=4, num_records_bytes=d_bytes
            )
            q_vals = [
                [
                    buf_copy_load(
                        x_buf,
                        (wave + xw * j) * (k // 8) + lane + 64 * u,
                        elem=fx.Uint32,
                        unit_elems=4,
                    )
                    for u in range(q_units)
                ]
                for j in range(q_rows)
            ]
            d_vals = [
                buf_copy_load(
                    d_buf,
                    rb0 * (kq_total * layout.D_BLOCK_BYTES // 16) + tid + 64 * xw * i,
                    elem=fx.Uint32,
                    unit_elems=4,
                )
                for i in range(d_loads)
            ]
            rocdl.sched_barrier(0)
            for j in range_constexpr(q_rows):
                row = wave + xw * j
                if row < M:
                    scale = _row_scale(q_vals[j])
                    if lane == 0:
                        _lds_bytes(lds.xscale.ptr, row * 4, 4).store(
                            Vec.from_elements([scale], fx.Float32).bitcast(fx.Uint8)
                        )
                    if scale >= fx.Float32(_SMALL_SCALE):
                        dws = _quant_units(q_vals[j], range(q_units), _quotient(scale))
                        for u in range_constexpr(q_units):
                            _lds_bytes(
                                lds.x.ptr, row * x_stride + (lane + 64 * u) * 8, 8
                            ).store(dws[u].bitcast(fx.Uint8))
                    else:
                        dws = _quant_units(q_vals[j], range(q_units), lambda v: v / scale)
                        for u in range_constexpr(q_units):
                            _lds_bytes(
                                lds.x.ptr, row * x_stride + (lane + 64 * u) * 8, 8
                            ).store(dws[u].bitcast(fx.Uint8))
            for i in range_constexpr(d_loads):
                u = tid + 64 * xw * i
                if const_expr(64 * xw * (i + 1) <= d_units):
                    _lds_bytes(lds.delta.ptr, u * 16, 16).store(d_vals[i].bitcast(fx.Uint8))
                else:
                    if u < d_units:
                        _lds_bytes(lds.delta.ptr, u * 16, 16).store(
                            d_vals[i].bitcast(fx.Uint8)
                        )
            if lane == 0:
                fx.atomic_add(
                    ready_ptr,
                    fx.Int32(1),
                    syncscope="workgroup",
                    ordering=fx.AtomicOrdering.Release,
                )
            gpu.barrier()
        else:
            ww = wave - xw
            r_local = ww % rb
            k_part = ww // rb
            valid = r_local < nrb
            rbg = rb0 + r_local
            # A slot past the CTA's run loads nothing (zero-sized buffer) and stores nothing.
            w_buf = ptr_buf_tensor(
                p_wq,
                elem=fx.Uint32,
                unit_elems=4,
                num_records_bytes=valid.select(fx.Int32(w_bytes), fx.Int32(0)),
            )
            s0 = k_part * ppw
            # Epilogue operands: channels 2 (l % 8) + h, tokens 8 q + 4 ((l // 16) % 2) + v.
            s_buf = ptr_buf_tensor(p_ws, elem=fx.Float32, unit_elems=2, num_records_bytes=n * 4)
            w_scale = buf_copy_load(s_buf, rbg * 8 + lane % 8, elem=fx.Float32, unit_elems=2)

            def load_pair(i):
                return buf_copy_load(
                    w_buf,
                    (rbg * kb_total + 2 * (s0 + i)) * 32 + lane,
                    elem=fx.Uint32,
                    unit_elems=4,
                )

            raws = [load_pair(i) for i in range(ppw)]
            rocdl.sched_barrier(0)

            # ======== Wait for the token rows and deltas in LDS ========
            published = fx.Int32(
                fx.generic_load(
                    ready_ptr,
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="workgroup",
                    volatile=True,
                )
            )
            while published < fx.Int32(xw):
                rocdl.s_sleep(1)
                published = fx.Int32(
                    fx.generic_load(
                        ready_ptr,
                        memory_order=fx.AtomicOrdering.Acquire,
                        syncscope="workgroup",
                        volatile=True,
                    )
                )
            x_scales = [
                Vec(
                    _lds_bytes(
                        lds.xscale.ptr,
                        fx.min(
                            fx.Int32(q * 8) + ((lane // 16) % 2) * 4, fx.Int32(max(4, mcap) - 4)
                        )
                        * 4,
                        16,
                    ).load()
                ).bitcast(fx.Float32)
                for q in range(tq)
            ]

            # ======== Block pairs: decode W(s), acc += X(s) W(s) ========
            mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
            accs = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range(tq)] for _ in range(2)]
            for h in range_constexpr(2):
                for q in range_constexpr(tq):
                    accs[h][q].store(Vec.filled(4, 0.0, fx.Float32))
            x_lane = (lane // 32) * 64 + ((lane % 32) // 8) * 16
            x_rows = [fx.min(fx.Int32(q * 8) + lane % 8, fx.Int32(rows - 1)) for q in range(tq)]
            d_lane = (r_local * kq_total) * layout.D_BLOCK_BYTES + ((lane % 32) // 16) * 64 + (
                lane % 8
            ) * 8
            xa = [
                [
                    Vec(
                        _lds_bytes(
                            lds.x.ptr, x_rows[q] * x_stride + x_lane + (s0 + i) * 128, 16
                        ).load()
                    ).bitcast(fx.Uint32)
                    for i in range(ppw)
                ]
                for q in range(tq)
            ]
            deltas = [
                Vec(
                    _lds_bytes(
                        lds.delta.ptr, d_lane + ((s0 + i) // 2) * layout.D_BLOCK_BYTES, 8
                    ).load()
                ).bitcast(fx.Uint32)
                for i in range(ppw)
            ]
            # Delta byte 2 (s % 2) + l // 32 of the dword, broadcast by the LUT's byte permute.
            sel_beta = fx.Uint32((lane // 32) * 0x01010101)
            for i in range_constexpr(ppw):
                s = s0 + i
                sel = sel_beta + fx.Uint32(s % 2) * fx.Uint32(0x02020202)
                a_ops = []
                for q in range_constexpr(tq):
                    v = xa[q][i]
                    per_c = []
                    for c in range_constexpr(2):
                        frag = fx.make_rmem_tensor(8, fp8)
                        frag.store(
                            Vec.from_elements([v[2 * c], v[2 * c + 1]], fx.Uint32).bitcast(fp8)
                        )
                        per_c.append(frag)
                    a_ops.append(per_c)
                raw = raws[i]
                for h in range_constexpr(2):
                    lut = layout.fp8_lut(deltas[i][h], sel)
                    ops = _decode_pair(
                        Vec.from_elements([raw[2 * h], raw[2 * h + 1]], fx.Uint32), lut
                    )
                    for c in range_constexpr(2):
                        b = fx.make_rmem_tensor(8, fp8)
                        b.store(ops[c])
                        for q in range_constexpr(tq):
                            fx.gemm(mma, accs[h][q], a_ops[q][c], b, accs[h][q])

            # ======== Epilogue: join the K halves (lane l ^ 40), add the K parts, BF16 ========
            sums = []
            for h in range_constexpr(2):
                for q in range_constexpr(tq):
                    acc = Vec(accs[h][q].load())
                    for v in range_constexpr(4):
                        sums.append(acc[v] + gpu.shuffle_xor(acc[v], 40, 64))
            useful = ((lane % 16) < 8) & (lane < 32)
            slot = (lane // 16) * 8 + lane % 8
            if const_expr(ks > 1):
                if (k_part > 0) & useful & valid:
                    base = ((r_local * (ks - 1) + k_part - 1) * 16 + slot) * (2 * tq * 4) * 4
                    for j in range_constexpr(2 * tq * 4):
                        _lds_bytes(lds.part.ptr, base + j * 4, 4).store(
                            Vec.from_elements([sums[j]], fx.Float32).bitcast(fx.Uint8)
                        )
            gpu.barrier()
            if const_expr(ks > 1):
                for p in range_constexpr(1, ks):
                    base = ((r_local * (ks - 1) + p - 1) * 16 + slot) * (2 * tq * 4) * 4
                    for j in range_constexpr(2 * tq * 4):
                        sums[j] = sums[j] + Vec(
                            _lds_bytes(lds.part.ptr, base + j * 4, 4).load()
                        ).bitcast(fx.Float32)[0]
            if (k_part == 0) & useful & valid:
                y_buf = ptr_buf_tensor(
                    p_y, elem=fx.BFloat16, unit_elems=2, num_records_bytes=M * (n * 2)
                )
                for q in range_constexpr(tq):
                    for v in range_constexpr(4):
                        token = q * 8 + (lane // 16) * 4 + v
                        out = Vec.from_elements(
                            [
                                (sums[(h * tq + q) * 4 + v] * w_scale[h] * x_scales[q][v]).to(
                                    fx.BFloat16
                                )
                                for h in range_constexpr(2)
                            ],
                            fx.BFloat16,
                        )
                        buf_copy_store(
                            y_buf,
                            token * (n // 2) + rbg * 8 + lane % 8,
                            out,
                            elem=fx.BFloat16,
                            unit_elems=2,
                        )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_a8w4_ptpc_dec_kernel(p_x, p_wq, p_wd, p_ws, p_y, M).launch(
            grid=(num_wg, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


@functools.cache
def compile_gemm_a8w4_ptpc_tile(n, k, rb, ks, tq, kp, num_wg, device_index):
    """Launcher ``(x, wq, wd, ws, y, M, stream)``: M <= 16 tq tokens, X quantized in the CTA.

    ``x`` BF16 [M, k], ``wq`` / ``wd`` / ``ws`` the fd2 layout, ``y`` BF16 [M, n]; raw
    ``fx.Pointer``. CTA ``i`` of the ``num_wg`` owns the row blocks ``[i nb / num_wg, (i + 1)
    nb / num_wg)`` (at most ``rb``) and all tokens; its ``rb * ks`` waves each take one row
    block slot and ``1 / ks`` of every K window. The CTA quantizes the token rows into LDS one
    K window at a time: ``kp`` windows of whole 512-K units, so ``16 tq`` FP8 rows fit in LDS.

    MFMA 16x16x32 FP8 with W as A (16 channels) and X as B (16 tokens per token tile): lane
    ``l`` = (``r`` = l % 16, ``g`` = l // 16) reads, in half ``h`` of 128-K step ``s``, the
    8-byte fd2 piece of row ``r``, block ``2 s + g // 2``, lane group ``2 (g % 2) + h`` (stored
    K ``128 s + 32 g + 16 h + 8 c + v`` for MFMA ``c``, all in 32-K group ``4 s + g``: one LUT
    per step), and per token tile ``t`` the 32 bytes of token ``16 t + r`` at ``128 s + 32 g``.
    The accumulator of tile ``t`` holds channels ``4 g + v`` of token ``16 t + r``.
    """
    del device_index
    waves = rb * ks
    nb = n // 16
    q_units = k // 512
    win_units = _tile_windows(k, kp)
    if (
        n % 16
        or k % 512
        or not 0 < kp <= q_units
        or any(4 * u % ks for u in win_units)
        or waves > 16
        or tq not in (1, 2)
        or not 0 < num_wg <= nb
        or -(-nb // num_wg) > rb
    ):
        raise ValueError(
            f"A8W4 tile GEMM: unsupported N={n} K={k} rb={rb} ks={ks} num_wg={num_wg} "
            f"tq={tq} kp={kp}"
        )
    fp8 = fx.Float8E4M3FNUZ
    tokens = 16 * tq
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    threads = 64 * waves
    # Window p: 512-K units [u0[p], u0[p] + win_units[p]); wave k_part takes spw[p] of its steps.
    u0 = [sum(win_units[:p]) for p in range(kp)]
    spw = [4 * u // ks for u in win_units]
    x_stride = max(win_units) * 512 + 16
    x_bytes = tokens * x_stride
    q_rows = (tokens + waves - 1) // waves
    part_bytes = rb * (ks - 1) * tq * 64 * 16
    # The K-part sums reuse the token rows when separate buffers would exceed the LDS.
    d_units = rb * kq_total * layout.D_BLOCK_BYTES // 16
    d_loads = (d_units + threads - 1) // threads
    part_alias = x_bytes + d_units * 16 + part_bytes + tokens * 4 > 65536
    if (part_alias and part_bytes > x_bytes) or x_bytes + d_units * 16 + tokens * 4 > 65536:
        raise ValueError(f"A8W4 tile GEMM: LDS too small for N={n} K={k} rb={rb} ks={ks}")
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)

    @fx.struct
    class SharedStorage:
        x: fx.Array[fx.Uint8, x_bytes, 16]
        delta: fx.Array[fx.Uint8, d_units * 16, 16]
        part: fx.Array[fx.Uint8, 16 if part_alias else max(16, part_bytes), 16]
        xscale: fx.Array[fx.Float32, tokens, 16]

    @flyc.kernel(name=tile_kernel_name(rb, ks, tq, kp, num_wg), known_block_size=[threads, 1, 1])
    def gemm_a8w4_ptpc_tile_kernel(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        row = lane % 16
        g = lane // 16
        wave = wave_uniform_i32(tid // 64)
        r_local = wave % rb
        k_part = wave // rb
        blk = fx.block_idx.x
        rb0 = (blk * nb) // num_wg
        valid = r_local < ((blk + 1) * nb) // num_wg - rb0
        rbg = rb0 + r_local
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        x_buf = ptr_buf_tensor(p_x, elem=fx.Uint32, unit_elems=4, num_records_bytes=M * (k * 2))
        w_buf = ptr_buf_tensor(p_wq, elem=fx.Uint32, unit_elems=2, num_records_bytes=w_bytes)
        d_buf = ptr_buf_tensor(p_wd, elem=fx.Uint32, unit_elems=4, num_records_bytes=d_bytes)

        # ======== Prologue: token rows and deltas of the CTA, then the wave's weights ========
        # Wave w quantizes token rows w + waves * j; lane l holds the 16-byte BF16 units
        # l + 64 u of the row (rows >= M read as zero).
        q_vals = [
            [
                buf_copy_load(
                    x_buf,
                    (wave + waves * j) * (k // 8) + lane + 64 * u,
                    elem=fx.Uint32,
                    unit_elems=4,
                )
                for u in range(q_units)
            ]
            for j in range(q_rows)
        ]
        d_vals = [
            buf_copy_load(
                d_buf,
                rb0 * (kq_total * layout.D_BLOCK_BYTES // 16) + tid + threads * i,
                elem=fx.Uint32,
                unit_elems=4,
            )
            for i in range(d_loads)
        ]
        rocdl.sched_barrier(0)
        # 8-byte unit of (step s, half h): (rbg * kb_total + 2 s) * 64 + w_lane + 16 h.
        w_lane = (g // 2) * 64 + (g % 2) * 32 + row
        win_s0 = [4 * u0[p] + k_part * spw[p] for p in range(kp)]
        raws = [
            [
                [
                    buf_copy_load(
                        w_buf,
                        (rbg * kb_total + 2 * (win_s0[p] + i)) * 64 + w_lane + 16 * h,
                        elem=fx.Uint32,
                        unit_elems=2,
                    )
                    for h in range(2)
                ]
                for i in range(spw[p])
            ]
            for p in range(kp)
        ]
        rocdl.sched_barrier(0)
        s_buf = ptr_buf_tensor(p_ws, elem=fx.Float32, unit_elems=4, num_records_bytes=n * 4)
        w_scale = buf_copy_load(s_buf, rbg * 4 + g, elem=fx.Float32, unit_elems=4)
        rocdl.sched_barrier(0)

        # ======== Token scales (whole rows) and the CTA's deltas to LDS ========
        scales = []
        for j in range_constexpr(q_rows):
            qrow = wave + waves * j
            scale = _row_scale(q_vals[j])
            scales.append(scale)
            if (lane == 0) & (qrow < M):
                _lds_bytes(lds.xscale.ptr, qrow * 4, 4).store(
                    Vec.from_elements([scale], fx.Float32).bitcast(fx.Uint8)
                )
        for i in range_constexpr(d_loads):
            u = tid + threads * i
            if const_expr(threads * (i + 1) <= d_units):
                _lds_bytes(lds.delta.ptr, u * 16, 16).store(d_vals[i].bitcast(fx.Uint8))
            else:
                if u < d_units:
                    _lds_bytes(lds.delta.ptr, u * 16, 16).store(d_vals[i].bitcast(fx.Uint8))

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        accs = [fx.make_rmem_tensor(4, fx.Float32) for _ in range(tq)]
        for t in range_constexpr(tq):
            accs[t].store(Vec.filled(4, 0.0, fx.Float32))
        x_lane = row * x_stride + g * 32
        d_lane = (r_local * kq_total) * layout.D_BLOCK_BYTES + (g % 2) * 64 + row * 4
        # Delta byte 2 (s % 2) + g // 2 of the dword, broadcast by the LUT's byte permute.
        sel_g = fx.Uint32((g // 2) * 0x01010101)
        for p in range_constexpr(kp):
            # ======== Window p: the token rows, FP8, to LDS ========
            if const_expr(p > 0):
                gpu.barrier()
            units = range(u0[p], u0[p] + win_units[p])
            for j in range_constexpr(q_rows):
                qrow = wave + waves * j
                if qrow < M:
                    if scales[j] >= fx.Float32(_SMALL_SCALE):
                        dws = _quant_units(q_vals[j], units, _quotient(scales[j]))
                        for i, u in enumerate(units):
                            off = qrow * x_stride + (lane + 64 * (u - u0[p])) * 8
                            _lds_bytes(lds.x.ptr, off, 8).store(dws[i].bitcast(fx.Uint8))
                    else:
                        dws = _quant_units(q_vals[j], units, lambda v, sc=scales[j]: v / sc)
                        for i, u in enumerate(units):
                            off = qrow * x_stride + (lane + 64 * (u - u0[p])) * 8
                            _lds_bytes(lds.x.ptr, off, 8).store(dws[i].bitcast(fx.Uint8))
            gpu.barrier()

            # ======== Window p steps: decode W(s), acc += W(s) X(s) ========
            for i in range_constexpr(spw[p]):
                s = win_s0[p] + i
                delta = Vec(
                    _lds_bytes(lds.delta.ptr, d_lane + (s // 2) * layout.D_BLOCK_BYTES, 4).load()
                ).bitcast(fx.Uint32)[0]
                if const_expr(spw[p] % 2 == 0):
                    sel = sel_g + fx.Uint32((i % 2) * 0x02020202)
                else:
                    sel = sel_g + fx.Uint32(s % 2) * fx.Uint32(0x02020202)
                lut = layout.fp8_lut(delta, sel)
                xb = [
                    Vec(
                        _lds_bytes(
                            lds.x.ptr,
                            x_lane + 16 * t * x_stride + (s - 4 * u0[p]) * 128,
                            32,
                        ).load()
                    ).bitcast(fx.Uint32)
                    for t in range(tq)
                ]
                for h in range_constexpr(2):
                    ops = _decode_pair(raws[p][i][h], lut)
                    for c in range_constexpr(2):
                        a = fx.make_rmem_tensor(8, fp8)
                        a.store(ops[c])
                        for t in range_constexpr(tq):
                            b = fx.make_rmem_tensor(8, fp8)
                            b.store(
                                Vec.from_elements(
                                    [xb[t][4 * h + 2 * c], xb[t][4 * h + 2 * c + 1]], fx.Uint32
                                ).bitcast(fp8)
                            )
                            fx.gemm(mma, accs[t], a, b, accs[t])

        # ======== Epilogue: add the K parts in order, scale, BF16 ========
        totals = [Vec(accs[t].load()) for t in range(tq)]
        if const_expr(ks > 1):
            part_ptr = lds.x.ptr if const_expr(part_alias) else lds.part.ptr
            if const_expr(part_alias):
                gpu.barrier()
            if (k_part > 0) & valid:
                for t in range_constexpr(tq):
                    base = (((r_local * (ks - 1) + k_part - 1) * tq + t) * 64 + lane) * 16
                    _lds_bytes(part_ptr, base, 16).store(totals[t].bitcast(fx.Uint8))
            gpu.barrier()
            for kp_i in range_constexpr(1, ks):
                for t in range_constexpr(tq):
                    base = (((r_local * (ks - 1) + kp_i - 1) * tq + t) * 64 + lane) * 16
                    totals[t] = totals[t] + Vec(
                        _lds_bytes(part_ptr, base, 16).load()
                    ).bitcast(fx.Float32)
        y_buf = ptr_buf_tensor(
            p_y, elem=fx.BFloat16, unit_elems=4, num_records_bytes=M * (n * 2)
        )
        for t in range_constexpr(tq):
            token = 16 * t + row
            x_scale = Vec(_lds_bytes(lds.xscale.ptr, token * 4, 4).load()).bitcast(fx.Float32)[0]
            if (k_part == 0) & valid & (token < M):
                out = Vec.from_elements(
                    [
                        (totals[t][v] * w_scale[v] * x_scale).to(fx.BFloat16)
                        for v in range_constexpr(4)
                    ],
                    fx.BFloat16,
                )
                buf_copy_store(
                    y_buf, token * (n // 4) + rbg * 4 + g, out, elem=fx.BFloat16, unit_elems=4
                )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_a8w4_ptpc_tile_kernel(p_x, p_wq, p_wd, p_ws, p_y, M).launch(
            grid=(num_wg, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch
