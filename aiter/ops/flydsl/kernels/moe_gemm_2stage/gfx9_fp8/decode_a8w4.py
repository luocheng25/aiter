# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 MoE decode kernels for small token counts (gfx942): no moe_sorting, one workgroup per route / token.

``aiter.ops.flydsl.moe_a8w4`` runs them up to ``DECODE_MAX_M`` tokens. Both kernels read the ``common_a8w4``
weight layout, decode MXFP4 to FP8 in registers and run MFMA 16x16x32 FP8 with the activation of one token
broadcast to all 16 rows of the A operand.

* Gate/Up (``compile_a8w4_gateup_decode``): grid ``(I / 16, M * TOPK)``. A workgroup owns 16
  channels of one route ``(token, slot)``: wave ``w`` reduces 16 gate (``w`` even) or up rows over
  one half of K. The token row is quantized per row to FP8 inside the workgroup (``scale =
  amax / 240``, IEEE division, round to nearest even, as ``quant_a8w4``); the per-channel factor and
  the activation scale are applied in the epilogue, which writes ``SiLU(gate) * up`` as float32
  (with a SwiGLU limit, gate and up clamped first: ``common_a8w4.swiglu_clamp``).
* Down (``compile_a8w4_down_decode``): grid ``(H / 16, M)``. A workgroup owns 16 output channels of
  one token and sums all its ``TOPK`` experts: wave ``w`` takes slots ``w, w + WAVES, ...``,
  quantizes those intermediate rows to FP8 (same arithmetic) and scales by the channel factor, the
  row scale and the routing weight. The waves' partial sums are added in LDS and written as BF16,
  so the output needs no clearing and no separate reduction. With pre-quantized rows
  (``compile_a8w4_down_decode_pq``, more tokens) the waves take slot pairs (see below).

Block pairs (Gate/Up): one 16-byte load per lane covers two adjacent 64-K blocks, which issues half
as many memory instructions as 8-byte loads of single blocks (the decode kernels are bound by
these loads). Lane ``l`` then holds layout lanes ``2l`` and ``2l + 1`` of block ``l // 32`` of the
pair: rows ``2 (l % 8)`` and ``2 (l % 8) + 1`` with K group ``b = (l % 32) // 8``. Each row goes to
its own MFMA, and the A operand of lane ``l`` is the activation of its own K group, so output row
``m`` (A of lanes ``16 g + m``) pairs with the B columns ``n`` of the same K groups, ``n // 8 == m // 8``:
columns 0..7 are read from output row 0 and columns 8..15 from row 8. Column ``n`` sums K groups
``{n // 8, n // 8 + 2}`` of both blocks for row ``2 (n % 8) (+ 1)``; columns ``n`` and ``n + 8``
together hold the full K.

Slot pairs (Down): the 16-byte loads cover the same block of two routing slots instead (lanes
0..31 slot ``2p``, lanes 32..63 slot ``2p + 1``, per-lane expert offsets into the whole weight
tensor), with the lane mapping above. The two slots need separate sums (their channel factors
differ), so the A operand of lane ``l`` is zero unless ``l % 2`` equals its slot's half: output rows
0 / 8 then sum the lower slot and rows 1 / 9 the upper one, from one MFMA.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh
from . import common_a8w4 as a8w4

FP8_MAX = 240.0
_LOG2E = 1.4426950408889634
_MAX_DELTA = (
    11  # largest delta stored by prepare_moe_a8w4_ptpc_weight (common_a8w4.MAX_DELTA)
)
_LUT_BYTES = 8 * (_MAX_DELTA + 1)


def gateup_decode_fits(K):
    """Whether ``compile_a8w4_gateup_decode`` supports hidden size ``K`` (4 waves; the threads stage the FP8 token
    row's 16-byte pieces in as many rounds as needed)."""
    return K % 512 == 0


def down_decode_fits(K, TOPK, WAVES, paired):
    """Whether the Down decode kernel (``paired``: ``compile_a8w4_down_decode_pq``) supports padded I
    ``K`` with ``WAVES`` waves: the staged intermediate rows fit in LDS (``SharedStorage`` of the
    kernel) and, for slot pairs, the 16-byte pieces of one row fit in the waves' lanes.
    """
    if paired:
        rows = 2 * WAVES * (((TOPK + 1) // 2 + WAVES - 1) // WAVES)
        lds, pieces_ok = (rows + 1) * K + WAVES * 32 * 4, K // 16 <= 64 * WAVES
    else:
        lds, pieces_ok = (
            WAVES * ((TOPK + WAVES - 1) // WAVES) * K + WAVES * 16 * 4,
            True,
        )
    return K % 64 == 0 and pieces_ok and lds + _LUT_BYTES <= a8w4.LDS_BYTES


def _lds_ptr(base, byte_off, elem, align):
    p = fx.add_offset(base, fx.make_int_tuple(byte_off))
    return fx.recast_iter(
        fx.PointerType.get(elem.ir_type, fx.AddressSpace.Shared, align), p
    )


def _global_ptr(p, elem, align):
    return fx.recast_iter(
        fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, align), fxh._as_ptr(p)
    )


def _lds_load(base, byte_off, elem, n, align):
    return fx.ptr_load(
        _lds_ptr(base, byte_off, elem, align), result_type=Vec.make_type(n, elem)
    )


def _lds_store(base, byte_off, vec, elem, align):
    fx.ptr_store(vec, _lds_ptr(base, byte_off, elem, align))


def _wave_max(value):
    """Max over the 64 lanes of a wave (non-negative inputs), returned wave-uniform."""
    # Raw DPP moves and readlane: FlyDSL 0.5.3 has no typed DPP operand (the quad / row permutations below
    # are free modifiers of the max, where fx.gpu.shuffle_xor issues a separate cross-lane permute per step),
    # and no typed readlane to take the four row maxima into scalar registers.
    # DPP controls: quad_perm xor 1, quad_perm xor 2, row_ror 4, row_ror 8.
    for ctrl in (0xB1, 0x4E, 0x124, 0x128):
        bits = value.bitcast(fx.Int32)
        moved = fx.rocdl.update_dpp(
            fx.Int32.ir_type, _raw(bits), _raw(bits), ctrl, 0xF, 0xF, False
        )
        value = fx.maxnumf(value, fx.Int32(moved).bitcast(fx.Float32))
    bits = value.bitcast(fx.Int32)
    rows = [
        fx.Int32(rocdl.readlane(T.i32, _raw(bits), 16 * r)).bitcast(fx.Float32)
        for r in range_constexpr(4)
    ]
    return fx.maxnumf(fx.maxnumf(rows[0], rows[1]), fx.maxnumf(rows[2], rows[3]))


def _abs_max(values):
    amax = abs(values[0])
    for v in range_constexpr(1, len(values)):
        amax = fx.maxnumf(amax, abs(values[v]))
    return amax


def _tree_sum(values):
    """Pairwise sum ``((v0 + v1) + (v2 + v3)) + ...`` (trace-time recursion)."""
    if len(values) == 1:
        return values[0]
    half = (len(values) + 1) // 2
    return _tree_sum(values[:half]) + _tree_sum(values[half:])


def _row_scale(amax):
    scale = amax / FP8_MAX
    return (scale == 0.0).select(fx.Float32(1.0), scale)


def _quant(value, scale):
    return fx.minnumf(
        fx.maxnumf(value / scale, fx.Float32(-FP8_MAX)), fx.Float32(FP8_MAX)
    )


def _fp8_pair(v0, v1, scale):
    """``RNE(v / scale)`` of two values as FP8 E4M3FNUZ bytes 0 and 1 (``-0`` stored as ``+0``), as ``quant_a8w4``."""
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8 has no LLVM
    # translation); the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    packed = fx.Uint32(
        rocdl.cvt_pk_fp8_f32(
            T.i32, _quant(v0, scale), _quant(v1, scale), fx.Int32(0), False
        )
    )
    mag = packed & fx.Uint32(0x7F7F)
    keep = (mag + fx.Uint32(0x7F7F)) & fx.Uint32(0x8080)
    return mag | (packed & keep)


def _fp8_words(values, scale):
    """FP8 bytes of ``values`` (a multiple of 4) packed little-endian into Uint32 words."""
    return Vec.from_elements(
        [
            _fp8_pair(values[4 * w], values[4 * w + 1], scale)
            | (_fp8_pair(values[4 * w + 2], values[4 * w + 3], scale) << 16)
            for w in range_constexpr(len(values) // 4)
        ],
        fx.Uint32,
    )


@flyc.jit
def _lut_table_init(base, tid):
    """LDS table of the ``fp8_lut`` pairs: the entry of delta ``d`` sits at byte ``8 * d`` (the stored delta byte)."""
    if tid <= _MAX_DELTA:
        d = fx.Uint32(tid * 8) * fx.Uint32(0x01010101)
        lo = (fx.Uint32(0x6C686080) - d) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(0x7C787470) - d
        _lds_store(base, tid * 8, Vec.from_elements([lo, hi], fx.Uint32), fx.Uint32, 8)


def _lut_at(base, deltas, shift):
    """LUT pair of the delta byte at bit ``shift`` (a per-lane value) of the lane's delta dword."""
    d = (fx.Uint32(deltas) >> shift) & fx.Uint32(0xFF)
    pair = _lds_load(base, fx.Int32(d), fx.Uint32, 2, 8)
    return pair[0], pair[1]


def _load_dwordx4(buffer, voffset_bytes, soffset_bytes):
    """Raw 16-byte buffer load at (VGPR, SGPR) byte offsets -> 4 x Uint32."""
    packet = fxh.BufferTensor(fx.make_view(fx.get_iter(buffer), fx.make_layout(16, 1)))
    return packet.load(
        voffset_bytes=voffset_bytes, soffset_bytes=soffset_bytes
    ).bitcast(fx.Uint32)


def _all_bytes(p):
    """Buffer over a whole tensor (descriptor size 0xFFFFFFFF, no range check): per-lane offsets."""
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(fx.recast_iter(fx.Uint8, fxh._as_ptr(p)), fx.make_layout(1, 1))
    )


def _decode_block(raw, lut):
    """The two layout dwords of a 64-K block (16 FP4) -> the FP8 operands of its two MFMAs as i64.

    The nibble shifts of both dwords are single 64-bit shifts (bits crossing the dword boundary
    are masked off), which saves two VALU per block.
    """
    lo, hi = lut
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    wide = Vec(raw).bitcast(fx.Uint64)[0]
    right = Vec.from_elements([wide >> 4], fx.Uint64).bitcast(fx.Uint32)
    left = Vec.from_elements([wide << 4], fx.Uint64).bitcast(fx.Uint32)
    ops = []
    for c in range_constexpr(2):
        r = fx.Uint32(raw[c])
        even = (fx.Uint32(left[c]) & sign) | a8w4._perm(hi, lo, r & mag)
        odd = (r & sign) | a8w4._perm(hi, lo, fx.Uint32(right[c]) & mag)
        ops.append(Vec.from_elements([even, odd], fx.Uint32).bitcast(fx.Int64)[0])
    return ops


def _mfma(a, b, acc):
    # Raw MFMA on register values: the typed MMA atom (fx.make_mma_atom with fx.gemm) takes fragments described
    # by layouts, while each operand here is one i64 built in registers (a decoded weight block, a broadcast
    # LDS read of the token row) and used by the next MFMA.
    return fx.rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a, b, acc, 0, 0, 0])


def _wave_index(tid):
    # Raw readfirstlane: FlyDSL 0.5.3 has no typed wave-uniform broadcast (fx.gpu offers only the shuffle
    # family, whose result is a VGPR); the wave index stays scalar for the offsets and branches built on it.
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(tid // 64)))


def compile_a8w4_gateup_decode(
    *, N, K, TOPK, PREQUANT=False, PLAN=None, SWIGLU_LIMIT=0.0
):
    """Launcher ``(x, x_scale, topk_ids, w, w_delta, w_scale, out, plan, y, M, stream)``.

    ``N`` = 2 * I (padded I, ``I % 16 == 0``; rows ``[0, I)`` gate), ``K`` = hidden (``K % 512 == 0``).
    x BF16 [M, K] quantized per row inside the kernel (``x_scale`` unused), or with ``PREQUANT``
    FP8 [M, K] with x_scale f32 [M]; topk_ids int32 [M, TOPK]; w / w_delta in the ``common_a8w4``
    layout; w_scale f32 [E, N]; out f32 [M, TOPK, I]. ``PLAN = (GROUP, MAX_M, ZERO_COLS)``: the workgroups
    of channel block 0 also write the expert-dedup plan entry of their route (``decode_dedup_a8w4``, plan
    int32 [M * TOPK, 4], ``M <= MAX_M``) and every workgroup zeroes its share of its token's row of y BF16
    [M, ZERO_COLS], for ``decode_dedup_a8w4.compile_a8w4_down_decode_dd``; without ``PLAN`` plan / y are unused.
    ``SWIGLU_LIMIT`` > 0 clamps gate and up before SiLU (``common_a8w4.swiglu_clamp``).
    """
    I = N // 2
    assert I % 16 == 0 and K % 512 == 0, f"unsupported Gate/Up decode shape N={N} K={K}"
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    KB_HALF = KB // 2
    PAIRS = KB_HALF // 2
    NJ = I // 16
    X_DWORDS = K // 2 // 256  # bf16 pairs per thread
    X_CHUNKS = K // 16  # 16-byte pieces of an FP8 row
    X_LOADS = (X_CHUNKS + 255) // 256  # of them per thread (pieces tid + 256 u)
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    if PLAN is not None:
        from .decode_dedup_a8w4 import _address, _plan_route, _rsrc

        GROUP, MAX_M, ZERO_COLS = PLAN
        SEGS = (MAX_M * TOPK + 63) // 64
        ZERO_PIECES = ZERO_COLS // 8  # 16-byte pieces of a y row
        ZPW = (ZERO_PIECES + NJ * TOPK - 1) // (
            NJ * TOPK
        )  # of them zeroed per workgroup
        assert ZERO_COLS % 8 == 0 and ZPW <= 256

    @fx.struct
    class SharedStorage:
        xq: fx.Array[fx.Uint8, K, 16]
        part: fx.Array[fx.Float32, 128, 16]
        red: fx.Array[fx.Float32, 4, 16]
        lut: fx.Array[fx.Uint8, _LUT_BYTES, 16]

    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_gateup_decode", SWIGLU_LIMIT))
    def a8w4_gateup_decode(
        p_x: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        p_plan: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = _wave_index(tid)
        j = gpu.block_idx.x
        route = gpu.block_idx.y
        token = route // TOPK

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xq_base = fx.recast_iter(fx.Uint8, lds.xq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        red_base = fx.recast_iter(fx.Uint8, lds.red.ptr)
        lut_base = fx.recast_iter(fx.Uint8, lds.lut.ptr)
        _lut_table_init(lut_base, tid)

        expert = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))[route]
        rb = (wave & 1) * NJ + j
        kb0 = (wave >> 1) * KB_HALF
        # 16-byte loads of block pairs (see "Block pairs" in the module docstring): lane l holds rows
        # 2 (l % 8) and 2 (l % 8) + 1, K group (l % 32) // 8 of block l // 32 of the pair.
        x_off = (lane // 32) * 64 + ((lane % 32) // 8) * 16
        # The token row does not depend on the expert id: issue it first.
        if const_expr(PREQUANT):
            # Coalesced 16-byte pieces tid + 256 u, staged in LDS (MFMA operands are broadcast reads);
            # threads past the row issue no load (whole waves skip it).
            x_row = _global_ptr(p_x, fx.Int64, 16) + fx.Int64(token) * (K // 8)
            x_pieces = []
            for u in range_constexpr(X_LOADS):
                piece = tid
                if const_expr(u > 0):
                    piece = tid + 256 * u
                x_piece = fx.full(2, 0, fx.Int64)
                if piece < X_CHUNKS:
                    x_piece = fx.ptr_load(
                        x_row + piece * 2, result_type=Vec.make_type(2, fx.Int64)
                    )
                x_pieces.append(x_piece)
            scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_x_scale))[token]
        else:
            x_row = fx.make_view(
                _global_ptr(p_x, fx.Uint32, 4) + fx.Int64(token) * (K // 2),
                fx.make_layout(K // 2, 1),
            )
            xbits = [fx.Uint32(x_row[tid + 256 * m]) for m in range_constexpr(X_DWORDS)]
        w_buf = a8w4.expert_bytes(p_weight, expert, W_BYTES)
        d_buf = a8w4.expert_bytes(p_w_delta, expert, D_BYTES)
        w_voff = fx.Int32(lane * 16)
        d_voff = fx.Int32(((lane % 32) // 16) * 64 + (lane % 8) * 8)
        d_shift = fx.Uint32((lane // 32) * 8)
        # Each delta load (both rows, 4 blocks) right before the two block pairs it covers (the
        # scheduler would otherwise group it after all weight loads), so the first pairs do not wait
        # for every load.
        raws, deltas = [], []
        for p in range_constexpr(PAIRS):
            if const_expr(p % 2 == 0):
                rocdl.sched_barrier(0)
                deltas.append(
                    a8w4.load_dwordx2(
                        d_buf,
                        d_voff,
                        fx.Int32((rb * KQ + kb0 // 4 + p // 2) * a8w4.D_BLOCK_BYTES),
                    )
                )
            raws.append(
                _load_dwordx4(
                    w_buf,
                    w_voff,
                    fx.Int32((rb * KB + kb0 + 2 * p) * a8w4.W_BLOCK_BYTES),
                )
            )
        rocdl.sched_barrier(0)
        # Channel factors of the epilogue (thread t < 16 owns channel j * 16 + t), loaded now so that
        # the epilogue does not wait for another memory round trip; only wave 0 issues these loads.
        channel = j * 16 + tid
        ws_gate = fx.Float32(0.0)
        ws_up = fx.Float32(0.0)
        if tid < 16:
            ws = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N,
                fx.make_layout(N, 1),
            )
            ws_gate = ws[channel]
            ws_up = ws[I + channel]
        rocdl.sched_barrier(0)

        if const_expr(PREQUANT):
            for u in range_constexpr(X_LOADS):
                piece = tid
                if const_expr(u > 0):
                    piece = tid + 256 * u
                if piece < X_CHUNKS:
                    _lds_store(xq_base, piece * 16, x_pieces[u], fx.Int64, 16)
            gpu.barrier()
        else:
            # Quantize the token row: thread `tid` owns bf16 pairs tid + 256 * m.
            values = []
            for bits in xbits:
                values.append((bits << 16).bitcast(fx.Float32))
                values.append((bits & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
            amax = _wave_max(_abs_max(values))
            if lane == 0:
                _lds_store(
                    red_base,
                    wave * 4,
                    Vec.from_elements([amax], fx.Float32),
                    fx.Float32,
                    4,
                )
            gpu.barrier()
            red = _lds_load(red_base, 0, fx.Float32, 4, 16)
            scale = _row_scale(
                fx.maxnumf(fx.maxnumf(red[0], red[1]), fx.maxnumf(red[2], red[3]))
            )
            for m in range_constexpr(X_DWORDS):
                pair = _fp8_pair(values[2 * m], values[2 * m + 1], scale).to(fx.Uint16)
                _lds_store(
                    xq_base,
                    (tid + 256 * m) * 2,
                    Vec.from_elements([pair], fx.Uint16),
                    fx.Uint16,
                    2,
                )
            gpu.barrier()

        # accs[h]: rows 2m + h of the block in columns m and m + 8 (m = 0..7), each with half of the K groups.
        accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
        for p in range_constexpr(PAIRS):
            xop = _lds_load(xq_base, (kb0 + 2 * p) * 64 + x_off, fx.Int64, 2, 16)
            shift = d_shift + fx.Uint32(16 * (p % 2))
            ops = [
                _decode_block(
                    Vec.from_elements([raws[p][2 * h], raws[p][2 * h + 1]], fx.Uint32),
                    _lut_at(lut_base, deltas[p // 2][h], shift),
                )
                for h in range_constexpr(2)
            ]
            for c in range_constexpr(2):
                for h in range_constexpr(2):
                    accs[h] = _mfma(xop[c], ops[h][c], accs[h])

        # Output row m multiplies the A operand of lanes 16 g + m, whose K group matches the B
        # columns n with n // 8 == m // 8: columns 0..7 are read from row 0 (lane n), columns 8..15
        # from row 8 (lane 32 + n).
        if (lane < 8) | ((lane >= 40) & (lane < 48)):
            for h in range_constexpr(2):
                _lds_store(
                    part_base,
                    ((wave * 2 + h) * 16 + lane % 16) * 4,
                    Vec.from_elements([Vec(accs[h])[0]], fx.Float32),
                    fx.Float32,
                    4,
                )
        gpu.barrier()
        if tid < 16:
            # Channel row tid = 2m + h: columns m and m + 8 of accumulator h.
            col = (tid & 1) * 16 + tid // 2
            parts = [
                _lds_load(part_base, (w * 32 + col) * 4, fx.Float32, 1, 4)[0]
                + _lds_load(part_base, (w * 32 + col + 8) * 4, fx.Float32, 1, 4)[0]
                for w in range_constexpr(4)
            ]
            gate = (parts[0] + parts[2]) * ws_gate * scale
            up = (parts[1] + parts[3]) * ws_up * scale
            gate, up = a8w4.swiglu_clamp(gate, up, SWIGLU_LIMIT)
            # Bare v_exp_f32 / v_rcp_f32 (raw rocdl): fx.math.exp2 and the f32 division lower to the
            # denormal-scaled exp and the correctly rounded divide, several VALU ops more per element; the
            # SiLU tolerance is met with the approximations.
            tmp = rocdl.exp2(T.f32, _raw(gate * (-_LOG2E)))
            out = fx.make_view(
                fxh._as_ptr(p_out, fx.Float32) + fx.Int64(route) * I,
                fx.make_layout(I, 1),
            )
            out[channel] = (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up
        if const_expr(PLAN is not None):
            # This workgroup's share of the token's y row, then (channel block 0, wave 0) the route's plan entry.
            y_row = _global_ptr(p_y, fx.Int64, 16) + fx.Int64(token) * (ZERO_COLS // 4)
            piece = (fx.Int32(route % TOPK) * NJ + fx.Int32(j)) * ZPW + fx.Int32(tid)
            if (fx.Int32(tid) < ZPW) & (piece < ZERO_PIECES):
                fx.ptr_store(fx.full(2, 0, fx.Int64), y_row + piece * 2)
            if (fx.Int32(j) == 0) & (wave == 0):
                routes = M * TOPK
                _plan_route(
                    fx.Int32(route),
                    fx.Int32(lane),
                    routes,
                    _rsrc(_address(p_ids), routes * 4),
                    fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids)),
                    _rsrc(_address(p_plan), routes * 16),
                    SEGS,
                    GROUP,
                )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        p_plan: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_gateup_decode(
            p_x,
            p_x_scale,
            p_ids,
            p_weight,
            p_w_delta,
            p_w_scale,
            p_out,
            p_plan,
            p_y,
            M,
        ).launch(grid=(NJ, M * TOPK, 1), block=(256, 1, 1), stream=stream)

    return launch


def compile_a8w4_down_decode(*, N, K, TOPK, WAVES=None):
    """Launcher ``(mid, mid_scale, topk_ids, topk_weight, w, w_delta, w_scale, out, M, stream)``.

    ``N`` = hidden (``N % 16 == 0``), ``K`` = padded I (``K % 64 == 0``). mid f32 [M, TOPK, K] is
    quantized per row inside the kernel (``mid_scale`` unused); topk_ids int32 / topk_weight f32
    [M, TOPK]; w_scale f32 [E, N]; out BF16 [M, N]. ``WAVES`` waves per workgroup (default one per
    slot), slots in rounds.
    """
    assert N % 16 == 0 and K % 64 == 0, f"unsupported Down decode shape N={N} K={K}"
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    NB = N // 16
    WAVES = TOPK if WAVES is None else WAVES
    ROUNDS = (TOPK + WAVES - 1) // WAVES
    K_PER_LANE = K // 64
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)

    @fx.struct
    class SharedStorage:
        mq: fx.Array[fx.Uint8, WAVES * ROUNDS * K, 16]
        part: fx.Array[fx.Float32, WAVES * 16, 16]
        lut: fx.Array[fx.Uint8, _LUT_BYTES, 16]

    @flyc.kernel
    def a8w4_down_decode(
        p_mid: fx.Pointer,
        p_mid_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = _wave_index(tid)
        j = gpu.block_idx.x
        token = gpu.block_idx.y

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mq_base = fx.recast_iter(fx.Uint8, lds.mq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        lut_base = fx.recast_iter(fx.Uint8, lds.lut.ptr)
        _lut_table_init(lut_base, tid)

        ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
        tws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weight))
        w_voff = fx.Int32(lane * 8)
        d_voff = fx.Int32((lane // 32) * 64 + (lane % 16) * 4)
        channel = j * 16 + lane % 16

        slots, valid = [], []
        for r in range_constexpr(ROUNDS):
            slot = wave + WAVES * r
            ok = slot < TOPK
            slots.append(ok.select(slot, fx.Int32(TOPK - 1)))
            valid.append(ok)
        experts = [ids[token * TOPK + slots[r]] for r in range_constexpr(ROUNDS)]
        weights = [tws[token * TOPK + slots[r]] for r in range_constexpr(ROUNDS)]
        x_off = (lane // 16) * 16
        # Intermediate rows do not depend on the expert ids: issue them first.
        mids = []
        for r in range_constexpr(ROUNDS):
            mid = fx.make_view(
                fxh._as_ptr(p_mid, fx.Float32) + fx.Int64(token * TOPK + slots[r]) * K,
                fx.make_layout(K, 1),
            )
            mids.append([mid[lane + 64 * u] for u in range_constexpr(K_PER_LANE)])
        raws, deltas, chan_scale = [], [], []
        for r in range_constexpr(ROUNDS):
            w_buf = a8w4.expert_bytes(p_weight, experts[r], W_BYTES)
            d_buf = a8w4.expert_bytes(p_w_delta, experts[r], D_BYTES)
            # Each delta dword right before the weight blocks it covers.
            raws.append([])
            deltas.append([])
            for i in range_constexpr(KB):
                if const_expr(i % 4 == 0):
                    deltas[r].append(
                        a8w4.load_dword(
                            d_buf,
                            d_voff,
                            fx.Int32((j * KQ + i // 4) * a8w4.D_BLOCK_BYTES),
                        )[0]
                    )
                raws[r].append(
                    a8w4.load_dwordx2(
                        w_buf, w_voff, fx.Int32((j * KB + i) * a8w4.W_BLOCK_BYTES)
                    )
                )
            ws = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(experts[r]) * N,
                fx.make_layout(N, 1),
            )
            chan_scale.append(ws[channel])
        rocdl.sched_barrier(0)

        # Quantize this wave's intermediate rows into its LDS slots.
        row_scale = []
        for r in range_constexpr(ROUNDS):
            values = mids[r] + ([fx.Float32(0.0)] if K_PER_LANE % 2 else [])
            scale = _row_scale(_wave_max(_abs_max(mids[r])))
            row_scale.append(scale)
            base = (wave * ROUNDS + r) * K
            for u in range_constexpr(0, K_PER_LANE, 2):
                pair = _fp8_pair(values[u], values[u + 1], scale)
                _lds_store(
                    mq_base,
                    base + lane + 64 * u,
                    Vec.from_elements([pair.to(fx.Uint8)], fx.Uint8),
                    fx.Uint8,
                    1,
                )
                if u + 1 < K_PER_LANE:
                    _lds_store(
                        mq_base,
                        base + lane + 64 * (u + 1),
                        Vec.from_elements([(pair >> 8).to(fx.Uint8)], fx.Uint8),
                        fx.Uint8,
                        1,
                    )
        gpu.barrier()

        total = fx.Float32(0.0)
        for r in range_constexpr(ROUNDS):
            base = (wave * ROUNDS + r) * K
            acc = fx.full(4, 0.0, fx.Float32)
            for i in range_constexpr(KB):
                xop = _lds_load(mq_base, base + i * 64 + x_off, fx.Int64, 2, 16)
                ops = _decode_block(
                    raws[r][i],
                    _lut_at(lut_base, deltas[r][i // 4], fx.Uint32(8 * (i % 4))),
                )
                for c in range_constexpr(2):
                    acc = _mfma(xop[c], ops[c], acc)
            contrib = Vec(acc)[0] * chan_scale[r] * row_scale[r] * weights[r]
            total = total + valid[r].select(contrib, fx.Float32(0.0))

        if lane < 16:
            _lds_store(
                part_base,
                (wave * 16 + lane) * 4,
                Vec.from_elements([total], fx.Float32),
                fx.Float32,
                4,
            )
        gpu.barrier()
        if tid < 16:
            parts = [
                _lds_load(part_base, (w * 16 + tid) * 4, fx.Float32, 1, 4)[0]
                for w in range_constexpr(WAVES)
            ]
            y = _tree_sum(parts)
            out = fx.make_view(
                fxh._as_ptr(p_out, fx.BFloat16) + fx.Int64(token) * N,
                fx.make_layout(N, 1),
            )
            out[j * 16 + tid] = fxh._f32_to_bf16(y)

    @flyc.jit
    def launch(
        p_mid: fx.Pointer,
        p_mid_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_down_decode(
            p_mid,
            p_mid_scale,
            p_ids,
            p_topk_weight,
            p_weight,
            p_w_delta,
            p_w_scale,
            p_out,
            M,
        ).launch(grid=(NB, M, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch


def compile_a8w4_down_decode_pq(*, N, K, TOPK, WAVES=None):
    """Launcher ``(mid, mid_scale, topk_ids, topk_weight, w, w_delta, w_scale, out, M, stream)``.

    Down of ``compile_a8w4_down_decode`` for pre-quantized rows (mid FP8 [M, TOPK, K], mid_scale
    f32 [M, TOPK]) with 16-byte loads of slot pairs (see "Slot pairs" in the module docstring).
    ``WAVES`` waves per workgroup (default one per slot pair), pairs in rounds.
    """
    assert N % 16 == 0 and K % 64 == 0, f"unsupported Down decode shape N={N} K={K}"
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    NB = N // 16
    PAIRS = (TOPK + 1) // 2
    WAVES = PAIRS if WAVES is None else WAVES
    ROUNDS = (PAIRS + WAVES - 1) // WAVES
    ROWS = 2 * WAVES * ROUNDS  # staged intermediate rows, followed by one all-zero row
    PIECES = K // 16  # 16-byte pieces of an FP8 row
    PIECE_LOADS = (PIECES + 31) // 32
    assert PIECES <= 64 * WAVES
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)

    @fx.struct
    class SharedStorage:
        mq: fx.Array[fx.Uint8, (ROWS + 1) * K, 16]
        part: fx.Array[fx.Float32, WAVES * 32, 16]
        lut: fx.Array[fx.Uint8, _LUT_BYTES, 16]

    @flyc.kernel
    def a8w4_down_decode_pq(
        p_mid: fx.Pointer,
        p_mid_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = _wave_index(tid)
        j = gpu.block_idx.x
        token = gpu.block_idx.y

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mq_base = fx.recast_iter(fx.Uint8, lds.mq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        lut_base = fx.recast_iter(fx.Uint8, lds.lut.ptr)
        _lut_table_init(lut_base, tid)
        if tid < PIECES:
            # The zero row: A operand of the masked lanes (see "Slot pairs" in the module docstring).
            _lds_store(
                mq_base, ROWS * K + tid * 16, fx.full(2, 0, fx.Int64), fx.Int64, 16
            )

        ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
        tws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weight))
        # Round r: slot pair wave + WAVES * r, slot 2 p in lanes 0..31 and slot 2 p + 1 in lanes 32..63.
        upper = lane >= 32
        slots, valid = [], []
        for r in range_constexpr(ROUNDS):
            for s in range_constexpr(2):
                slot = (wave + WAVES * r) * 2 + s
                ok = slot < TOPK
                slots.append(ok.select(slot, fx.Int32(TOPK - 1)))
                valid.append(ok)
        # Raw readfirstlane (no typed wave-uniform broadcast in FlyDSL 0.5.3, see _wave_index): the expert ids
        # stay scalar for the per-expert channel-factor offsets.
        experts = [
            fx.Int32(rocdl.readfirstlane(T.i32, _raw(ids[token * TOPK + slots[q]])))
            for q in range_constexpr(2 * ROUNDS)
        ]
        weights = [tws[token * TOPK + slots[q]] for q in range_constexpr(2 * ROUNDS)]
        # Intermediate rows do not depend on the expert ids: issue them first.
        pieces, row_scale = [], []
        for r in range_constexpr(ROUNDS):
            # Lane l loads the 16-byte pieces l % 32 (+ 32 u) of its slot's row, staged in LDS.
            lane_slot = upper.select(slots[2 * r + 1], slots[2 * r])
            row = _global_ptr(p_mid, fx.Int64, 16) + fx.Int64(
                token * TOPK + lane_slot
            ) * (K // 8)
            for u in range_constexpr(PIECE_LOADS):
                chunk = lane % 32 + 32 * u
                if const_expr(32 * (u + 1) > PIECES):
                    chunk = (chunk < PIECES).select(chunk, fx.Int32(PIECES - 1))
                pieces.append(
                    fx.ptr_load(row + chunk * 2, result_type=Vec.make_type(2, fx.Int64))
                )
            for s in range_constexpr(2):
                row_scale.append(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_mid_scale))[
                        token * TOPK + slots[2 * r + s]
                    ]
                )
        # Whole-tensor buffers with per-lane expert offsets (the two halves read different experts).
        w_all = _all_bytes(p_weight)
        d_all = _all_bytes(p_w_delta)
        s_all = _all_bytes(p_w_scale)
        w_lane = fx.Int32((lane % 32) * 16)
        d_lane = fx.Int32(((lane % 32) // 16) * 64 + (lane % 8) * 8)
        s_lane = fx.Int32((lane % 8) * 8)
        raws, deltas, chan_scale = [], [], []
        for r in range_constexpr(ROUNDS):
            lane_expert = upper.select(experts[2 * r + 1], experts[2 * r])
            w_voff = lane_expert * W_BYTES + w_lane
            d_voff = lane_expert * D_BYTES + d_lane
            # Each delta load (both rows, 4 blocks) right before the weight blocks it covers.
            raws.append([])
            deltas.append([])
            for i in range_constexpr(KB):
                if const_expr(i % 4 == 0):
                    deltas[r].append(
                        a8w4.load_dwordx2(
                            d_all,
                            d_voff,
                            fx.Int32((j * KQ + i // 4) * a8w4.D_BLOCK_BYTES),
                        )
                    )
                raws[r].append(
                    _load_dwordx4(
                        w_all, w_voff, fx.Int32((j * KB + i) * a8w4.W_BLOCK_BYTES)
                    )
                )
            for s in range_constexpr(2):
                # Channel factors of the lane's rows 2 (l % 8) and 2 (l % 8) + 1.
                ws = a8w4.load_dwordx2(
                    s_all, s_lane, experts[2 * r + s] * (N * 4) + j * 64
                )
                chan_scale.append(
                    [fx.Uint32(ws[h]).bitcast(fx.Float32) for h in range_constexpr(2)]
                )
        rocdl.sched_barrier(0)

        for r in range_constexpr(ROUNDS):
            for u in range_constexpr(PIECE_LOADS):
                chunk = lane % 32 + 32 * u
                dst = ((wave * ROUNDS + r) * 2 + lane // 32) * K + chunk * 16
                if const_expr(32 * (u + 1) <= PIECES):
                    _lds_store(mq_base, dst, pieces[r * PIECE_LOADS + u], fx.Int64, 16)
                else:
                    if chunk < PIECES:
                        _lds_store(
                            mq_base, dst, pieces[r * PIECE_LOADS + u], fx.Int64, 16
                        )
        gpu.barrier()

        # Lane l holds rows 2 (l % 8) + h, K group b = (l % 32) // 8 of its slot's block. Its A operand
        # is the activation of that slot and K group if l % 2 equals the slot's half, else zero: output
        # row 0 / 8 then sums the lower slot, row 1 / 9 the upper one.
        b = (lane % 32) // 8
        keep = (lane % 2) == (lane // 32)
        vals = [fx.Float32(0.0), fx.Float32(0.0)]
        for r in range_constexpr(ROUNDS):
            a_off = keep.select(
                ((wave * ROUNDS + r) * 2 + lane // 32) * K + b * 16, ROWS * K + b * 16
            )
            accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
            for i in range_constexpr(KB):
                xop = _lds_load(mq_base, a_off + i * 64, fx.Int64, 2, 16)
                ops = [
                    _decode_block(
                        Vec.from_elements(
                            [raws[r][i][2 * h], raws[r][i][2 * h + 1]], fx.Uint32
                        ),
                        _lut_at(lut_base, deltas[r][i // 4][h], fx.Uint32(8 * (i % 4))),
                    )
                    for h in range_constexpr(2)
                ]
                for c in range_constexpr(2):
                    for h in range_constexpr(2):
                        accs[h] = _mfma(xop[c], ops[h][c], accs[h])
            for h in range_constexpr(2):
                acc = Vec(accs[h])
                for s in range_constexpr(2):
                    q = 2 * r + s
                    contrib = acc[s] * chan_scale[q][h] * row_scale[q] * weights[q]
                    vals[h] = vals[h] + valid[q].select(contrib, fx.Float32(0.0))

        # Columns 0..7 from row 0 / 1 (lane n), columns 8..15 from row 8 / 9 (lane 32 + n).
        if (lane < 8) | ((lane >= 40) & (lane < 48)):
            for h in range_constexpr(2):
                _lds_store(
                    part_base,
                    ((wave * 2 + h) * 16 + lane % 16) * 4,
                    Vec.from_elements([vals[h]], fx.Float32),
                    fx.Float32,
                    4,
                )
        gpu.barrier()
        if tid < 16:
            # Channel row tid = 2m + h: columns m and m + 8 of the waves' accumulators h.
            col = (tid & 1) * 16 + tid // 2
            parts = [
                _lds_load(part_base, (w * 32 + col) * 4, fx.Float32, 1, 4)[0]
                + _lds_load(part_base, (w * 32 + col + 8) * 4, fx.Float32, 1, 4)[0]
                for w in range_constexpr(WAVES)
            ]
            y = _tree_sum(parts)
            out = fx.make_view(
                fxh._as_ptr(p_out, fx.BFloat16) + fx.Int64(token) * N,
                fx.make_layout(N, 1),
            )
            out[j * 16 + tid] = fxh._f32_to_bf16(y)

    @flyc.jit
    def launch(
        p_mid: fx.Pointer,
        p_mid_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_down_decode_pq(
            p_mid,
            p_mid_scale,
            p_ids,
            p_topk_weight,
            p_weight,
            p_w_delta,
            p_w_scale,
            p_out,
            M,
        ).launch(grid=(NB, M, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch


def compile_a8w4_row_quant(*, COLS, BF16):
    """Launcher ``(x, q, scale, rows, stream)``: per-row FP8 quantization for the decode kernels.

    x BF16 (``BF16``) or f32 [rows, COLS] -> q FP8 E4M3FNUZ [rows, COLS] and scale f32 [rows], with
    the arithmetic of ``quant_a8w4`` (``scale = amax / 240``, IEEE division, round to nearest even,
    ``-0`` stored as ``+0``; all-zero rows get scale 1). One wave per row; lane ``l`` owns the
    16-byte input pieces ``l + 64 m``, so each wave issues a few 16-byte loads and 8- / 4-byte stores
    instead of one access per element.
    """
    VEC = 8 if BF16 else 4  # elements per 16-byte input piece
    assert COLS % VEC == 0, f"unsupported row quantization COLS={COLS}"
    PIECES = COLS // VEC
    LOADS = (PIECES + 63) // 64
    ROWS = 4  # waves (rows) per workgroup

    @flyc.kernel
    def a8w4_row_quant(
        p_in: fx.Pointer, p_q: fx.Pointer, p_scale: fx.Pointer, rows: fx.Int32
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        row = gpu.block_idx.x * ROWS + _wave_index(tid)
        if row < rows:
            src = _global_ptr(p_in, fx.Uint32, 16) + fx.Int64(row) * (
                COLS * (2 if BF16 else 4) // 4
            )
            pieces, values = [], []
            for m in range_constexpr(LOADS):
                piece = lane + 64 * m
                if const_expr(64 * (m + 1) > PIECES):
                    piece = (piece < PIECES).select(
                        piece, fx.Int32(PIECES - 1)
                    )  # duplicates leave amax unchanged
                pieces.append(piece)
                words = fx.ptr_load(
                    src + piece * 4, result_type=Vec.make_type(4, fx.Uint32)
                )
                vals = []
                for w in range_constexpr(4):
                    word = fx.Uint32(words[w])
                    if const_expr(BF16):
                        vals.append((word << 16).bitcast(fx.Float32))
                        vals.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
                    else:
                        vals.append(word.bitcast(fx.Float32))
                values.append(vals)
            scale = _row_scale(
                _wave_max(_abs_max([v for vals in values for v in vals]))
            )
            dst = _global_ptr(p_q, fx.Uint32, VEC) + fx.Int64(row) * (COLS // 4)
            for m in range_constexpr(LOADS):
                stored = _fp8_words(values[m], scale)
                if const_expr(64 * (m + 1) <= PIECES):
                    fx.ptr_store(stored, dst + pieces[m] * (VEC // 4))
                else:
                    if lane + 64 * m < PIECES:
                        fx.ptr_store(stored, dst + pieces[m] * (VEC // 4))
            if lane == 0:
                scale_out = fx.make_view(
                    fxh._as_ptr(p_scale, fx.Float32) + fx.Int64(row),
                    fx.make_layout(1, 1),
                )
                scale_out[0] = scale

    @flyc.jit
    def launch(
        p_in: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_row_quant(p_in, p_q, p_scale, rows).launch(
            grid=((rows + ROWS - 1) // ROWS, 1, 1),
            block=(64 * ROWS, 1, 1),
            stream=stream,
        )

    return launch
