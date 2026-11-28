# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 MoE decode kernels for small token counts (gfx942): no moe_sorting, one workgroup per route / token.

``aiter.ops.flydsl.moe_a8w4`` runs them in its decode tier (``_decode_range``). The kernels read the ``common_a8w4``
weight layout, decode MXFP4 to FP8 in registers and run MFMA 16x16x32 FP8 with the activation of one token
broadcast to all 16 rows of the A operand. One Gate/Up kernel and two Down kernels:

* Gate/Up (``compile_a8w4_gateup_decode_mixed``, kernel ``a8w4_gateup_decode``): grid ``(I / 16, M * SLOTS)``. A
  workgroup owns 16 channels of one routing column of one token: for a routed expert wave ``w`` reduces 16 gate
  (``w`` even) or up rows over one half of K. The token row is quantized per row to FP8 inside the workgroup
  (``scale = amax / 240``, round to nearest even, the FP8 bytes of ``quant_a8w4``'s IEEE division: one reciprocal per
  row and one FMA correction per value, ``_quant_fast``) or read pre-quantized (``PREQUANT``, more tokens); the
  per-channel factor and the activation scale are applied in the epilogue, which writes ``SiLU(gate) * up`` as float32
  (with a SwiGLU limit, gate and up clamped first: ``common_a8w4.swiglu_clamp``). With ``SHARED`` the last column is
  a shared FP8 expert's (DeepSeek-V4.1, the "mixed" decode below), else every column is routed (SLOTS = TOPK).
* Down that quantizes its rows (``compile_a8w4_down_decode_mixed``, kernel ``a8w4_down_decode``): grid ``(H / 32, M)``.
  A workgroup owns 32 output channels (a row-block pair, see below) of one token and sums all its routing columns,
  one wave each: the wave quantizes its intermediate row to FP8 (same bytes, two corrections) and scales by the channel
  factor, the row scale and the routing weight. The waves' partial sums are added in LDS and written as BF16, so the
  output needs no clearing and no separate reduction. With ``SHARED`` the last wave runs the shared FP8 expert.
* Down on pre-quantized rows (``compile_a8w4_down_decode_pq``, kernel ``a8w4_down_decode_pq``, more tokens and the
  folded shared expert): the waves take slot pairs (see below).

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

Row-block pairs (the routed columns of the Down that quantizes its rows): the 16-byte loads cover
the same 64-K block of two neighbouring 16-row blocks of one slot. Lane ``16 g + n`` loads the 16
bytes of rows ``2 (n % 8)`` and ``2 (n % 8) + 1``, K group ``g``, of 16-row block ``n // 8`` of the
pair (layout lanes ``16 g + 2 (n % 8)`` and ``+ 1``), so lane ``16 g + n`` sits in MFMA lane group
``g`` with the K group that the broadcast activation of group ``g`` multiplies: MMA ``c`` of row
``h`` (``h`` = 0, 1) sums, in column ``n``, exactly the 32 products that the 8-byte load of row
``2 (n % 8) + h`` alone gave in column ``2 (n % 8) + h``, in the same order of blocks and MMAs. Each
output channel therefore gets the bits of 8-byte loads of single blocks, from half as many load
instructions, and a workgroup quantizes each intermediate row once for 32 channels.

Masked routes (id -1): no kernel reads their weights, codes, deltas or channel factors. Gate/Up and the Down that
quantizes its rows load through zero-byte buffer views of the expert (the hardware range check, which counts the SGPR
offset too, returns 0 without a memory access) and skip the factor loads, so a masked
route's intermediate row is 0; the slot-pair Down's whole-tensor views end one expert's bytes below 2**32, where the
32-bit offsets of expert -1 fall. Down drops a masked slot's product with a select (whatever its routing weight and
intermediate row), so a token whose slots are all masked gets BF16 +0.
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


# Block pairs per wave whose loads the Gate/Up decode kernel issues before it quantizes the token row itself (without
# PREQUANT); each further pair's loads go out as the MFMA loop consumes one. Fewer registers in flight (85 VGPRs instead
# of 153 at H 5120: 5 waves per SIMD instead of 3) let more waves hide the loads: measured per launch (eager) at
# DeepSeek-V4.1 TP4 (H 5120, I 576, topk 6), per-route kernel M 1 / 2 / 4 0.88x / 0.88x / 0.93x the time of all 20
# pairs first; Qwen TP2 (H 2560, 10 pairs) 0.97x / 0.94x / 0.97x. With PREQUANT it gains nothing (Qwen TP2 M 16 / 32
# within 1%) and keeps all pairs first; decode_dedup_a8w4's deduplicated kernel takes it where
# moe_a8w4.DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I says.
GATEUP_PREFETCH_PAIRS = 6


# 64-K blocks (per wave, over its rounds in order) whose loads the slot-pair Down issues before staging its
# intermediate rows; each further block's loads go out as the MFMA loop consumes one. Measured per launch (eager) at
# DeepSeek-V4.1 TP4 (H 5120, I 576, topk 6) M 4 / 8 / 16 / 32 0.99x / 0.99x / 0.94x / 0.95x the time of all blocks
# first, Qwen TP2 (I 320, two rounds of 5 blocks) M 4 / 8 / 16 / 32 0.97x / 0.95x / 0.95x / 0.98x. (The Down that
# quantizes its rows itself: ``compile_a8w4_down_decode_mixed``'s PREFETCH.)
DOWN_PQ_PREFETCH_BLOCKS = 6


def gateup_decode_fits(K):
    """Whether the Gate/Up decode kernel (``compile_a8w4_gateup_decode_mixed``) supports hidden size ``K`` (4 waves; the
    threads stage the FP8 token row's 16-byte pieces in as many rounds as needed)."""
    return K % 512 == 0


def _down_pq_waves(TOPK):
    """Waves of the slot-pair Down by default: one per slot pair, at most 3 (pairs in rounds). Measured in one process
    against 2 / 4 waves: Qwen TP2 / TP4 (H 2560, topk 11) 1.008-1.022x / 1.056-1.071x the time, DeepSeek-V4.1 TP4
    (H 5120, topk 6) 1.060-1.077x / 0.995-1.000x, the DeepSeek-V4.1 main TP4 layer under CUDA graph replay
    1.073-1.074x / 1.008-1.032x."""
    return min((TOPK + 1) // 2, 3)


def down_decode_pq_fits(K, TOPK):
    """Whether the slot-pair Down (``compile_a8w4_down_decode_pq`` with its default waves) supports padded I ``K``: the
    staged intermediate rows fit in LDS (``SharedStorage`` of the kernel) and the 16-byte pieces of one row fit in the
    waves' lanes.
    """
    WAVES = _down_pq_waves(TOPK)
    rows = 2 * WAVES * (((TOPK + 1) // 2 + WAVES - 1) // WAVES)
    lds = (rows + 1) * K + WAVES * 32 * 4
    return K % 64 == 0 and K // 16 <= 64 * WAVES and lds + _LUT_BYTES <= a8w4.LDS_BYTES


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


def _plus_shared(value, shared):
    """``value`` plus the BF16 ``shared`` in f32, or ``value`` for None (no shared expert; trace-time choice outside the
    AST rewriting of a kernel's dynamic branches)."""
    return value if shared is None else value + shared.to(fx.Float32)


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


def _all_bytes(p, unit):
    """Buffer over a whole tensor addressed by per-lane 32-bit offsets ``expert * unit + (offset within the expert)``:
    the range check ends ``unit`` bytes before 2**32, so a masked route (expert -1, offsets in the last ``unit`` bytes
    of the 32-bit range) loads 0 without reading memory; the tensor's ``E * unit`` bytes must fit below that.
    """
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(fx.recast_iter(fx.Uint8, fxh._as_ptr(p)), fx.make_layout(1, 1)),
        max_size=False,
        num_records_bytes=(1 << 32) - unit,
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


def _row_block_pair_offsets(lane, KB, KQ):
    """Per-lane byte offsets ``(weight, delta, channel factor)`` of the row-block pairs (see "Row-block pairs" in the
    module docstring): lane ``16 g + n`` reads rows ``2 (n % 8)`` and ``2 (n % 8) + 1`` of the pair's 16-row block
    ``n // 8``, K group ``g``; the caller adds the pair's first block through the scalar offsets."""
    n = lane % 16
    half = n // 8
    pair = n % 8
    w = half * (KB * a8w4.W_BLOCK_BYTES) + (lane // 16) * 128 + pair * 16
    d = half * (KQ * a8w4.D_BLOCK_BYTES) + (lane // 32) * 64 + pair * 8
    s = half * 64 + pair * 8
    return fx.Int32(w), fx.Int32(d), fx.Int32(s)


def _row_block_pair_mfmas(xop, raw, deltas, lut_base, shift, accs):
    """The four MFMAs of one 64-K block of a row-block pair (``raw``: the lane's 16-byte load, ``deltas``: its rows'
    delta dwords): accumulator ``h`` takes rows ``2 (n % 8) + h``, MMA ``c`` of the block after MMA ``c - 1``, as the
    single-block Down chains them."""
    ops = [
        _decode_block(
            Vec.from_elements([raw[2 * h], raw[2 * h + 1]], fx.Uint32),
            _lut_at(lut_base, deltas[h], shift),
        )
        for h in range_constexpr(2)
    ]
    for c in range_constexpr(2):
        for h in range_constexpr(2):
            accs[h] = _mfma(xop[c], ops[h][c], accs[h])
    return accs


def _pair_channel(lane, h):
    """Channel (of the workgroup's 32) of accumulator ``h`` of lane ``n < 16``: row ``2 (n % 8) + h`` of the pair's 16-row
    block ``n // 8`` (output row 0, column ``n``)."""
    return (lane // 8) * 16 + (lane % 8) * 2 + h


def compile_a8w4_down_decode_pq(
    *, N, K, TOPK, WAVES=None, SHARED=False, ID_COLS=None, PREFETCH=None
):
    """Launcher ``(mid, mid_scale, topk_ids, topk_weight, w, w_delta, w_scale, out, M, stream)``.

    The Down decode kernel for pre-quantized rows (mid FP8 [M, TOPK, K], mid_scale f32 [M, TOPK]; topk_ids int32 /
    topk_weight f32 [M, TOPK], -1 masked; w_scale f32 [E, N]; out BF16 [M, N]) with 16-byte loads of slot pairs (see
    "Slot pairs" in the module docstring): grid ``(N / 16, M)``, a workgroup sums 16 output channels of one token over
    all its routes. ``WAVES`` waves per workgroup (default ``_down_pq_waves``), pairs in rounds. Every expert's w2 bytes
    plus one expert's must fit in 2**32 (32-bit offsets, the range check ending below expert -1's). With ``SHARED``
    ``out`` holds BF16 values on entry (a fused shared expert's output, ``moe_a8w4._SharedFold``) and each output
    element is the f32 sum of the routes plus that value, rounded to BF16 once (each workgroup reads its 16 elements
    with its first loads). topk_ids / topk_weight have ``ID_COLS`` (default TOPK) columns per token, those past TOPK
    unread (``common_a8w4.id_index``).
    """
    assert N % 16 == 0 and K % 64 == 0, f"unsupported Down decode shape N={N} K={K}"
    ID_COLS = TOPK if ID_COLS is None else ID_COLS
    assert ID_COLS >= TOPK
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    NB = N // 16
    PAIRS = (TOPK + 1) // 2
    WAVES = _down_pq_waves(TOPK) if WAVES is None else WAVES
    ROUNDS = (PAIRS + WAVES - 1) // WAVES
    ROWS = 2 * WAVES * ROUNDS  # staged intermediate rows, followed by one all-zero row
    PIECES = K // 16  # 16-byte pieces of an FP8 row
    PIECE_LOADS = (PIECES + 31) // 32
    assert PIECES <= 64 * WAVES
    PREFETCH = min(ROUNDS * KB, DOWN_PQ_PREFETCH_BLOCKS if PREFETCH is None else PREFETCH)
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
            fx.Int32(rocdl.readfirstlane(T.i32, _raw(ids[token * ID_COLS + slots[q]])))
            for q in range_constexpr(2 * ROUNDS)
        ]
        weights = [tws[token * ID_COLS + slots[q]] for q in range_constexpr(2 * ROUNDS)]
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
        w_all = _all_bytes(p_weight, W_BYTES)
        d_all = _all_bytes(p_w_delta, D_BYTES)
        s_all = _all_bytes(p_w_scale, N * 4)
        w_lane = fx.Int32((lane % 32) * 16)
        d_lane = fx.Int32(((lane % 32) // 16) * 64 + (lane % 8) * 8)
        s_lane = fx.Int32((lane % 8) * 8)
        lane_experts = [
            upper.select(experts[2 * r + 1], experts[2 * r]) for r in range_constexpr(ROUNDS)
        ]
        raws = [[] for _ in range_constexpr(ROUNDS)]
        deltas = [[] for _ in range_constexpr(ROUNDS)]

        def issue(step):
            # Trace-time helper: the loads of block step % KB of round step // KB, each delta load (both rows, 4
            # blocks) right before the weight blocks it covers.
            r, i = divmod(step, KB)
            if const_expr(i % 4 == 0):
                deltas[r].append(
                    a8w4.load_dwordx2(
                        d_all,
                        lane_experts[r] * D_BYTES + d_lane,
                        fx.Int32((j * KQ + i // 4) * a8w4.D_BLOCK_BYTES),
                    )
                )
            raws[r].append(
                _load_dwordx4(
                    w_all,
                    lane_experts[r] * W_BYTES + w_lane,
                    fx.Int32((j * KB + i) * a8w4.W_BLOCK_BYTES),
                )
            )

        # The first PREFETCH blocks now, each further one as the MFMA loop consumes one (DOWN_PQ_PREFETCH_BLOCKS).
        for step in range_constexpr(PREFETCH):
            issue(step)
        chan_scale = []
        for r in range_constexpr(ROUNDS):
            for s in range_constexpr(2):
                # Channel factors of the lane's rows 2 (l % 8) and 2 (l % 8) + 1.
                ws = a8w4.load_dwordx2(
                    s_all, s_lane, experts[2 * r + s] * (N * 4) + j * 64
                )
                chan_scale.append(
                    [fx.Uint32(ws[h]).bitcast(fx.Float32) for h in range_constexpr(2)]
                )
        shared_value = None
        if const_expr(SHARED):
            # The shared expert's value of this thread's output channel (threads from 16 on load a duplicate).
            shared_row = fx.make_view(
                fxh._as_ptr(p_out, fx.BFloat16) + fx.Int64(token) * N,
                fx.make_layout(N, 1),
            )
            shared_value = shared_row[j * 16 + tid % 16]
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
                step = r * KB + i
                if const_expr(0 < step and step + PREFETCH - 1 < ROUNDS * KB):
                    issue(step + PREFETCH - 1)
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
                    # A masked route (id -1) adds nothing (its weight may be anything); its loads above fell past
                    # the range of _all_bytes.
                    live = valid[q] & (experts[q] >= 0)
                    vals[h] = vals[h] + live.select(contrib, fx.Float32(0.0))

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
            y = _plus_shared(_tree_sum(parts), shared_value)
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
    the bytes of ``quant_a8w4`` (``scale = amax / 240``, IEEE division, round to nearest even,
    ``-0`` stored as ``+0``; all-zero rows get scale 1): one reciprocal per row and FMA corrections per value
    (``_quant_fast``: one for BF16, two for f32), rows outside ``_fast_div_ok`` divide. One wave per row; lane ``l``
    owns the 16-byte input pieces ``l + 64 m``, so each wave issues a few 16-byte loads and 8- / 4-byte stores
    instead of one access per element.
    """
    VEC = 8 if BF16 else 4  # elements per 16-byte input piece
    assert COLS % VEC == 0, f"unsupported row quantization COLS={COLS}"
    PIECES = COLS // VEC
    LOADS = (PIECES + 63) // 64
    ROWS = 4  # waves (rows) per workgroup
    STEPS = 1 if BF16 else 2  # FMA corrections of _quant_fast (exact at the FP8 level)

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
            amax = _wave_max(_abs_max([v for vals in values for v in vals]))
            scale = _row_scale(amax)
            dst = _global_ptr(p_q, fx.Uint32, VEC) + fx.Int64(row) * (COLS // 4)
            if _fast_div_ok(amax):
                inv = fx.Float32(1.0) / scale
                for m in range_constexpr(LOADS):
                    stored = _fp8_words_fast(values[m], scale, inv, STEPS)
                    if const_expr(64 * (m + 1) <= PIECES):
                        fx.ptr_store(stored, dst + pieces[m] * (VEC // 4))
                    else:
                        if lane + 64 * m < PIECES:
                            fx.ptr_store(stored, dst + pieces[m] * (VEC // 4))
            else:
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


# -- Gate/Up and the Down that quantizes its rows; with SHARED a shared FP8 expert (DeepSeek-V4.1) as one more column --
# compile_a8w4_gateup_decode_mixed / compile_a8w4_down_decode_mixed with SHARED (the mixed decode) run a shared FP8
# expert with per-row scales as one more routing column of the two decode launches. Its weights are FP8 E4M3FNUZ in
# 16-row x 64-K blocks of 1024 bytes ([N / 16][K / 64][4][16 rows][16 B], twice a routed block's bytes, K in order: the
# layout aiter.ops.shuffle.shuffle_weight(layout=(16, 16)) gives an FP8 [N, K] with K % 64 == 0), so lane 16 g + n of a
# wave loads K group g of row n with one 16-byte load per block and feeds the bytes to the MFMA with no delta and no
# LUT; the row scale takes the place of the channel factor. Gate/Up gives the shared expert two workgroups per channel
# block (8 gate and the same 8 up rows each, K split over the 4 waves), so that every workgroup reads as many bytes with
# as many loads as a routed one; Down gives it the last of its TOPK waves.
# Both kernels are bound by the VALU work of their waves (with two or more workgroups per CU), a third of which was the
# IEEE division of the row quantization (one per value). They divide once per row (the reciprocal) and correct each
# product with FMAs (``_quant_fast``): the same FP8 bytes as the division (rows outside ``_fast_div_ok`` keep the
# division).


def _silu_mul(gate, up):
    """``SiLU(gate) * up`` as the Gate/Up decode epilogue computes it: bare v_exp_f32 / v_rcp_f32 (raw rocdl;
    fx.math.exp2 and the f32 division lower to the denormal-scaled exp and the correctly rounded divide, several VALU
    ops more per element; the SiLU tolerance is met with the approximations)."""
    tmp = rocdl.exp2(T.f32, _raw(gate * (-_LOG2E)))
    return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up


# Row abs-max range of the fast row quantization (``_quant_fast``): with it every quotient that can reach the FP8 grid
# (|v / scale| >= 2**-12) has a normal residual, and the result equals the IEEE division's at the FP8 level.
_FAST_DIV_MIN = 2.0**-60
_FAST_DIV_MAX = 2.0**127


def _fast_div_ok(amax):
    """Whether a row with abs-max ``amax`` takes the fast quantization (0 included: the quotients are all 0)."""
    return (amax < _FAST_DIV_MAX) & ((amax >= _FAST_DIV_MIN) | (amax == 0.0))


def _quant_fast(value, scale, inv, steps):
    """``_quant`` through the reciprocal ``inv = RN(1 / scale)`` (one IEEE division per row instead of one per value):
    ``q = RN(value * inv)`` and ``steps`` FMA corrections ``q += (value - q * scale) * inv`` (Markstein). Two steps
    give the correctly rounded quotient (the first makes it faithful, the second exact); one step was checked equal to
    the IEEE division at the FP8 level for every BF16 value against every BF16 row abs-max in ``_fast_div_ok``.
    """
    q = value * inv
    for _ in range(steps):
        q = fx.math.fma(fx.math.fma(-q, scale, value), inv, q)
    return fx.minnumf(fx.maxnumf(q, fx.Float32(-FP8_MAX)), fx.Float32(FP8_MAX))


def _fp8_pair_q(q0, q1):
    """FP8 E4M3FNUZ bytes 0 and 1 of two clamped quotients (``-0`` stored as ``+0``), as ``_fp8_pair``."""
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False))
    mag = packed & fx.Uint32(0x7F7F)
    keep = (mag + fx.Uint32(0x7F7F)) & fx.Uint32(0x8080)
    return mag | (packed & keep)


def _fp8_word_q(q0, q1, q2, q3):
    """FP8 E4M3FNUZ bytes of four clamped quotients in one dword (``q0`` in byte 0, ``-0`` stored as ``+0``): the
    second conversion writes the high half of the first one's result, and one ``-0`` fix covers the four bytes (a
    byte's magnitude plus 0x7F stays below 0x100)."""
    lo = rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, q2, q3, lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    keep = (mag + fx.Uint32(0x7F7F7F7F)) & fx.Uint32(0x80808080)
    return mag | (packed & keep)


def _fp8_pair_fast(v0, v1, scale, inv, steps):
    """``_fp8_pair`` with ``_quant_fast``."""
    return _fp8_pair_q(
        _quant_fast(v0, scale, inv, steps), _quant_fast(v1, scale, inv, steps)
    )


def _fp8_words_fast(values, scale, inv, steps):
    """``_fp8_words`` with ``_quant_fast``, four values per conversion dword (``_fp8_word_q``)."""
    q = [_quant_fast(v, scale, inv, steps) for v in values]
    return Vec.from_elements(
        [
            _fp8_word_q(q[4 * w], q[4 * w + 1], q[4 * w + 2], q[4 * w + 3])
            for w in range_constexpr(len(values) // 4)
        ],
        fx.Uint32,
    )


def _row_amax_scale(values):
    """Wave-uniform abs-max of a wave's f32 row (lane ``l`` holds elements ``l + 64 u``) and its FP8 row scale."""
    amax = _wave_max(_abs_max(values))
    return amax, _row_scale(amax)


def _stage_row(mq_base, base, lane, values, pair_of):
    """Stores a wave's f32 intermediate row (lane ``l`` holds elements ``l + 64 u``) as FP8 bytes into LDS at byte
    ``base``; ``pair_of(v0, v1)`` gives the FP8 bytes of two values (``_fp8_pair`` / ``_fp8_pair_fast``)."""
    count = len(values)
    padded = values + ([fx.Float32(0.0)] if count % 2 else [])
    for u in range(0, count, 2):
        pair = pair_of(padded[u], padded[u + 1])
        _lds_store(
            mq_base,
            base + lane + 64 * u,
            Vec.from_elements([pair.to(fx.Uint8)], fx.Uint8),
            fx.Uint8,
            1,
        )
        if u + 1 < count:
            _lds_store(
                mq_base,
                base + lane + 64 * (u + 1),
                Vec.from_elements([(pair >> 8).to(fx.Uint8)], fx.Uint8),
                fx.Uint8,
                1,
            )


def _stage_row_ieee(mq_base, base, lane, values, scale):
    """``_stage_row`` with the IEEE division of the Down decode kernel (``_fp8_pair``)."""
    _stage_row(mq_base, base, lane, values, lambda v0, v1: _fp8_pair(v0, v1, scale))


def _stage_row_fast(mq_base, base, lane, values, scale, steps):
    """``_stage_row`` with ``_quant_fast`` (``steps`` corrections; the row's abs-max within ``_fast_div_ok``)."""
    inv = fx.Float32(1.0) / scale
    _stage_row(
        mq_base,
        base,
        lane,
        values,
        lambda v0, v1: _fp8_pair_fast(v0, v1, scale, inv, steps),
    )


def _stage_x(xq_base, tid, values, quant):
    """Stores the FP8 bytes of a token row's values (thread ``tid`` holds pairs ``tid + 256 m``) into LDS;
    ``quant(v)`` gives a value's clamped quotient. Two pairs share a conversion dword (``_fp8_word_q``)."""

    def store(m, half):
        _lds_store(
            xq_base,
            (tid + 256 * m) * 2,
            Vec.from_elements([half.to(fx.Uint16)], fx.Uint16),
            fx.Uint16,
            2,
        )

    q = [quant(v) for v in values]
    pairs = len(values) // 2
    m = 0
    while m + 1 < pairs:
        word = _fp8_word_q(q[2 * m], q[2 * m + 1], q[2 * m + 2], q[2 * m + 3])
        store(m, word)
        store(m + 1, word >> 16)
        m += 2
    for r in range(m, pairs):
        store(r, _fp8_pair_q(q[2 * r], q[2 * r + 1]))


def _stage_x_ieee(xq_base, tid, values, scale):
    """``_stage_x`` with the IEEE division of the Gate/Up decode kernel (``_quant``)."""
    _stage_x(xq_base, tid, values, lambda v: _quant(v, scale))


def _stage_x_fast(xq_base, tid, values, scale, steps):
    """``_stage_x`` with ``_quant_fast`` (``steps`` corrections; the row's abs-max within ``_fast_div_ok``)."""
    inv = fx.Float32(1.0) / scale
    _stage_x(xq_base, tid, values, lambda v: _quant_fast(v, scale, inv, steps))


def _wave_fence():
    # LDS bytes written by other lanes of the same wave are read next: a wavefront-scope fence keeps the compiler from
    # moving the reads above the writes and emits no instruction (a wave performs its LDS operations in order).
    fx.memory_fence(
        syncscope=rocdl.SyncScope.Wavefront, ordering=fx.AtomicOrdering.AcqRel
    )


def down_decode_mixed_fits(K, TOPK, SHARED=True):
    """Whether ``compile_a8w4_down_decode_mixed`` supports padded I ``K`` with ``TOPK`` routing columns (one wave
    each, at most 16; at least 2 with ``SHARED``): every wave's intermediate row, partial sums and LUT copy fit in LDS.
    """
    return (
        K % 64 == 0
        and (2 if SHARED else 1) <= TOPK <= 16
        and TOPK * (K + 32 * 4 + _LUT_BYTES) <= a8w4.LDS_BYTES
    )


def _never():
    # A condition that is constant false once traced: the kernel variant without a shared expert keeps the shared
    # expert's branches out (they fold away), so its code is the routed columns' alone.
    return fx.Int32(1) == fx.Int32(0)


def compile_a8w4_gateup_decode_mixed(
    *,
    N,
    K,
    TOPK,
    E=None,
    SWIGLU_LIMIT=0.0,
    SHARED=True,
    PREQUANT=False,
    ID_COLS=None,
):
    """The Gate/Up decode kernel: grid ``(I / 16, M * SLOTS)``, a workgroup per 16 channels of one routing column of
    one token. ``N`` = 2 * I (padded I, ``I % 16 == 0``; rows ``[0, I)`` gate), ``K`` = hidden (``K % 512 == 0``);
    w / w_delta in the ``common_a8w4`` layout, w_scale f32 [E, N]; out f32 [M, TOPK, I]. ``SWIGLU_LIMIT`` > 0 clamps
    gate and up before SiLU (``common_a8w4.swiglu_clamp``).

    Without ``SHARED`` (the per-route Gate/Up) every column is a routed expert's and the launcher is ``(x, x_scale,
    topk_ids, w, w_delta, w_scale, out, M, stream)``, grid rows ``M * TOPK`` (row = route = token * TOPK + slot). x BF16 [M, K] is quantized per row inside every workgroup (``x_scale`` unused), or with ``PREQUANT`` x is
    FP8 [M, K] with x_scale f32 [M] (the row kernel's) and every block pair's loads go out first. topk_ids int32
    [M, TOPK] (-1: a masked route, its out row 0, no weight read), or [M, ID_COLS] with ``ID_COLS`` > TOPK, the columns
    past TOPK unread (a folded shared expert's, see ``common_a8w4.id_index``).

    With ``SHARED`` (the mixed Gate/Up) the launcher is ``(x, topk_ids, w, w_delta, w_scale, shared_w, shared_scale,
    out, M, stream)`` (x BF16, quantized in the kernel): a shared FP8 expert in the last of the ``TOPK`` routing columns
    (``topk_ids`` / out rows [M, TOPK]). The ``TOPK - 1`` routed columns run as above. The shared expert, id ``E``, is
    live when the last column holds ``E`` and the token's first id is not -1 (ATOM's rule; otherwise it reads no weight
    and its out row is 0). It takes two more workgroups per channel block and token, 8 of the 16 channels each, so that
    every workgroup reads the bytes of a routed one: shared_w FP8 E4M3FNUZ [N, K] in the 16-row x 64-K block layout
    ``[N / 16][K / 64][4][16 rows][16 B]`` (K in order within the 16 bytes; ``aiter.ops.shuffle.shuffle_weight(layout=
    (16, 16))`` of an FP8 [N, K] with K % 64 == 0) and shared_scale f32 [N] per row, applied in place of the channel
    factor. Lane ``16 g + n`` loads K group ``g`` of gate row ``8 h + n`` (``n < 8``) or up row ``8 h + n - 8`` of the
    half ``h``, wave ``w`` reduces K quarter ``w`` (the FP8 bytes feed the MFMA directly), and the epilogue writes
    SiLU(gate) * up (clamped as above) of the 8 channels. The routed workgroups' loads are the same instructions as the
    shared ones' (the descriptor and offsets selected per workgroup). Grid rows ``M * (TOPK + 1)``: row
    ``t * (TOPK + 1) + s`` is the route ``(t, s)`` for ``s < TOPK - 1``, then the shared expert's halves.

    A routed workgroup: wave ``w`` reduces 16 gate (``w`` even) or up rows over one half of K with 16-byte loads of
    block pairs (see "Block pairs" in the module docstring). A workgroup that quantizes its token row does it with
    ``_quant_fast`` (one correction: the FP8 bytes of ``quant_a8w4``'s IEEE division).
    """
    from .stream_a8w4 import _address, _load, _rsrc

    I = N // 2
    ID_COLS = TOPK if ID_COLS is None else ID_COLS
    assert (
        I % 16 == 0 and K % 512 == 0 and TOPK >= (2 if SHARED else 1)
    ), f"unsupported {'mixed ' if SHARED else ''}Gate/Up decode shape N={N} K={K} TOPK={TOPK}"
    assert ID_COLS >= TOPK
    # The mixed kernel quantizes its token rows itself and reads ids of TOPK columns.
    assert not SHARED or (E is not None and not PREQUANT and ID_COLS == TOPK)
    E = 0 if E is None else E  # traced in the folded-away shared branches only
    R = TOPK - 1 if SHARED else TOPK  # routed columns
    SLOTS = TOPK + 1 if SHARED else TOPK  # grid rows per token
    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    KB_HALF = KB // 2
    # 16-byte loads per lane: the routed block pairs of a K half, the shared blocks of a K quarter.
    PAIRS = KB_HALF // 2
    NJ = I // 16
    X_DWORDS = K // 2 // 256  # bf16 pairs per thread
    X_CHUNKS = K // 16  # 16-byte pieces of an FP8 row
    X_LOADS = (X_CHUNKS + 255) // 256  # of them per thread (pieces tid + 256 u)
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    S_BLOCK = 2 * a8w4.W_BLOCK_BYTES  # one 16-row x 64-K FP8 block
    S_UP = NJ * KB * S_BLOCK  # from a gate row's block to the same up row's block
    PREFETCH = PAIRS if PREQUANT else min(PAIRS, GATEUP_PREFETCH_PAIRS)

    @fx.struct
    class SharedStorage:
        xq: fx.Array[fx.Uint8, K, 16]
        part: fx.Array[fx.Float32, 128, 16]
        red: fx.Array[fx.Float32, 4, 16]
        lut: fx.Array[fx.Uint8, _LUT_BYTES, 16]

    # The pointers a variant reads lie at the start of the kernel arguments (one scalar load fetches them): the shared
    # FP8 expert's weight and scales (SHARED) or x_scale (without SHARED) take the slots p_aux0 / p_aux1.
    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_gateup_decode", SWIGLU_LIMIT))
    def a8w4_gateup_decode(
        p_x: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        p_aux0: fx.Pointer,
        p_aux1: fx.Pointer,
        M: fx.Int32,
    ):
        if const_expr(SHARED):
            p_shared, p_shared_scale = p_aux0, p_aux1
            p_x_scale = p_aux0  # unread
        else:
            p_x_scale = p_aux0
            p_shared, p_shared_scale = p_weight, p_w_scale  # traced in the folded-away shared branches only
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = _wave_index(tid)
        j = fx.Int32(gpu.block_idx.x)
        y = fx.Int32(gpu.block_idx.y)
        token = y // SLOTS
        s = y - token * SLOTS
        shared = (s >= R) if SHARED else _never()
        half = s - R

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xq_base = fx.recast_iter(fx.Uint8, lds.xq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        red_base = fx.recast_iter(fx.Uint8, lds.red.ptr)
        lut_base = fx.recast_iter(fx.Uint8, lds.lut.ptr)
        _lut_table_init(lut_base, tid)

        # The token row does not depend on the ids: issue it first.
        if const_expr(PREQUANT):
            # Coalesced 16-byte pieces tid + 256 u of the FP8 row, staged in LDS (MFMA operands are broadcast reads);
            # threads past the row issue no load (whole waves skip it).
            x_prow = _global_ptr(p_x, fx.Int64, 16) + fx.Int64(token) * (K // 8)
            x_pieces = []
            for u in range_constexpr(X_LOADS):
                piece = tid
                if const_expr(u > 0):
                    piece = tid + 256 * u
                x_piece = fx.full(2, 0, fx.Int64)
                if piece < X_CHUNKS:
                    x_piece = fx.ptr_load(
                        x_prow + piece * 2, result_type=Vec.make_type(2, fx.Int64)
                    )
                x_pieces.append(x_piece)
            scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_x_scale))[token]
        else:
            x_row = fx.make_view(
                _global_ptr(p_x, fx.Uint32, 4) + fx.Int64(token) * (K // 2),
                fx.make_layout(K // 2, 1),
            )
            xbits = [
                fx.Uint32(x_row[tid + 256 * m]) for m in range_constexpr(X_DWORDS)
            ]
        ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
        row = token * TOPK
        # The weight, delta and factor bytes of the workgroup's route: the routed expert's codes (the id zero-extended,
        # as in common_a8w4.expert_bytes) or the shared expert's FP8 rows; zero bytes when masked (the range check
        # returns 0 without reading memory). A branch per kind, so that a routed workgroup waits for its own id only and
        # a shared one issues both of its ids before waiting.
        w_addr = fx.Int64(0)
        w_records = fx.Int64(0)
        d_addr = fx.Int64(0)
        d_records = fx.Int64(0)
        f_addr = fx.Int64(0)
        alive = fx.Int32(0)
        if shared:
            expert = ids[row + R]
            first = ids[row]
            ok = (expert == E) & (first >= 0)
            w_addr = _address(p_shared)
            w_records = ok.select(fx.Int64(N * K), fx.Int64(0))
            f_addr = _address(p_shared_scale) + fx.Int64(j * 16 + half * 8) * 4
            alive = ok.select(fx.Int32(1), fx.Int32(0))
        else:
            expert = ids[token * ID_COLS + s]
            ok = expert >= 0
            expert64 = fx.Int64(fx.Uint32(expert))
            w_addr = _address(p_weight) + expert64 * W_BYTES
            w_records = ok.select(fx.Int64(W_BYTES), fx.Int64(0))
            d_addr = _address(p_w_delta) + expert64 * D_BYTES
            d_records = ok.select(fx.Int64(D_BYTES), fx.Int64(0))
            f_addr = _address(p_w_scale) + (expert64 * N + fx.Int64(j * 16)) * 4
            alive = ok.select(fx.Int32(1), fx.Int32(0))
        live = alive != 0
        w_rsrc = _rsrc(w_addr, w_records)
        d_rsrc = _rsrc(d_addr, d_records)
        rb = (wave & 1) * NJ + j
        kb0 = (wave >> 1) * KB_HALF
        n = lane % 16
        w_voff = shared.select(
            (n >= 8).select(fx.Int32(S_UP), fx.Int32(0))
            + (lane // 16) * 256
            + (n % 8) * 16,
            lane * 16,
        )
        w_soff = shared.select(
            (j * KB + wave * PAIRS) * S_BLOCK + half * 128,
            (rb * KB + kb0) * a8w4.W_BLOCK_BYTES,
        )
        d_voff = fx.Int32(((lane % 32) // 16) * 64 + (lane % 8) * 8)
        d_soff = (rb * KQ + kb0 // 4) * a8w4.D_BLOCK_BYTES
        # Each delta load right before the two block pairs it covers (the scheduler would otherwise group it after all
        # weight loads, so that the first pairs wait for every load); the first PREFETCH pairs now, each further one as
        # the MFMA loop consumes one (GATEUP_PREFETCH_PAIRS).
        raws, deltas = [], []

        def issue(p, raws, deltas):
            # Trace-time helper: the loads of 16-byte step p (and of its delta dwords for an even p, unless the
            # caller passes no delta list: a shared workgroup's branch, whose deltas are never read).
            if const_expr(p % 2 == 0):
                rocdl.sched_barrier(0)
                if const_expr(deltas is not None):
                    deltas.append(
                        _load(d_rsrc, d_voff, d_soff + (p // 2) * a8w4.D_BLOCK_BYTES, 2)
                    )
            raws.append(_load(w_rsrc, w_voff, w_soff + p * S_BLOCK, 4))

        for p in range_constexpr(PREFETCH):
            issue(p, raws, deltas)
        rocdl.sched_barrier(0)
        # Channel factors of the epilogue (thread t < 16 owns channel j * 16 + t), or the shared expert's row scales
        # (t < 8 owns channel j * 16 + 8 h + t); a masked route skips them (factors 0, its out row 0).
        ws_gate = fx.Float32(0.0)
        ws_up = fx.Float32(0.0)
        if (fx.Int32(tid) < shared.select(fx.Int32(8), fx.Int32(16))) & live:
            f_rsrc = _rsrc(f_addr, N * 4)
            ws_gate = _load(f_rsrc, tid * 4, 0, 1)[0].bitcast(fx.Float32)
            ws_up = _load(f_rsrc, tid * 4, I * 4, 1)[0].bitcast(fx.Float32)
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
            row_max = fx.maxnumf(fx.maxnumf(red[0], red[1]), fx.maxnumf(red[2], red[3]))
            scale = _row_scale(row_max)
            # One correction: exact at the FP8 level for BF16 values (see _quant_fast).
            if _fast_div_ok(row_max):
                _stage_x_fast(xq_base, tid, values, scale, 1)
            else:
                _stage_x_ieee(xq_base, tid, values, scale)
            gpu.barrier()

        accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
        if shared:
            # Every output row holds column n's sum over the K group of all four lane groups.
            g_off = (lane // 16) * 16
            s_raws = list(raws)
            for p in range_constexpr(PAIRS):
                if const_expr(0 < p and p + PREFETCH - 1 < PAIRS):
                    issue(p + PREFETCH - 1, s_raws, None)
                xop = _lds_load(
                    xq_base, (wave * PAIRS + p) * 64 + g_off, fx.Int64, 2, 16
                )
                weight = Vec.from_elements(s_raws[p], fx.Uint32).bitcast(fx.Int64)
                for c in range_constexpr(2):
                    accs[0] = _mfma(xop[c], weight[c], accs[0])
        else:
            # The routed loop (block pairs, accs[h] rows 2 m + h).
            x_off = (lane // 32) * 64 + ((lane % 32) // 8) * 16
            d_shift = fx.Uint32((lane // 32) * 8)
            r_raws, r_deltas = list(raws), list(deltas)
            for p in range_constexpr(PAIRS):
                if const_expr(0 < p and p + PREFETCH - 1 < PAIRS):
                    issue(p + PREFETCH - 1, r_raws, r_deltas)
                xop = _lds_load(xq_base, (kb0 + 2 * p) * 64 + x_off, fx.Int64, 2, 16)
                shift = d_shift + fx.Uint32(16 * (p % 2))
                ops = [
                    _decode_block(
                        Vec.from_elements(
                            [r_raws[p][2 * h], r_raws[p][2 * h + 1]], fx.Uint32
                        ),
                        _lut_at(lut_base, r_deltas[p // 2][h], shift),
                    )
                    for h in range_constexpr(2)
                ]
                for c in range_constexpr(2):
                    for h in range_constexpr(2):
                        accs[h] = _mfma(xop[c], ops[h][c], accs[h])

        if shared:
            if lane < 16:
                _lds_store(
                    part_base,
                    (wave * 16 + lane) * 4,
                    Vec.from_elements([Vec(accs[0])[0]], fx.Float32),
                    fx.Float32,
                    4,
                )
        else:
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
        if shared:
            if tid < 8:
                # Gate column tid and up column 8 + tid of the four K quarters.
                gate = _tree_sum(
                    [
                        _lds_load(part_base, (w * 16 + tid) * 4, fx.Float32, 1, 4)[0]
                        for w in range_constexpr(4)
                    ]
                )
                up = _tree_sum(
                    [
                        _lds_load(part_base, (w * 16 + 8 + tid) * 4, fx.Float32, 1, 4)[
                            0
                        ]
                        for w in range_constexpr(4)
                    ]
                )
                gate, up = a8w4.swiglu_clamp(
                    gate * ws_gate * scale, up * ws_up * scale, SWIGLU_LIMIT
                )
                out = fx.make_view(
                    fxh._as_ptr(p_out, fx.Float32) + fx.Int64(row + R) * I,
                    fx.make_layout(I, 1),
                )
                out[j * 16 + half * 8 + tid] = _silu_mul(gate, up)
        else:
            if tid < 16:
                # Channel row tid = 2m + h: columns m and m + 8 of accumulator h.
                pcol = (tid & 1) * 16 + tid // 2
                parts = [
                    _lds_load(part_base, (w * 32 + pcol) * 4, fx.Float32, 1, 4)[0]
                    + _lds_load(part_base, (w * 32 + pcol + 8) * 4, fx.Float32, 1, 4)[0]
                    for w in range_constexpr(4)
                ]
                gate = (parts[0] + parts[2]) * ws_gate * scale
                up = (parts[1] + parts[3]) * ws_up * scale
                gate, up = a8w4.swiglu_clamp(gate, up, SWIGLU_LIMIT)
                out = fx.make_view(
                    fxh._as_ptr(p_out, fx.Float32) + fx.Int64(row + s) * I,
                    fx.make_layout(I, 1),
                )
                out[j * 16 + tid] = _silu_mul(gate, up)

    if SHARED:

        @flyc.jit
        def launch(
            p_x: fx.Pointer,
            p_ids: fx.Pointer,
            p_weight: fx.Pointer,
            p_w_delta: fx.Pointer,
            p_w_scale: fx.Pointer,
            p_shared: fx.Pointer,
            p_shared_scale: fx.Pointer,
            p_out: fx.Pointer,
            M: fx.Int32,
            stream: fx.Stream,
        ):
            CompilationContext.get_current()
            a8w4_gateup_decode(
                p_x,
                p_ids,
                p_weight,
                p_w_delta,
                p_w_scale,
                p_out,
                p_shared,
                p_shared_scale,
                M,
            ).launch(grid=(NJ, M * SLOTS, 1), block=(256, 1, 1), stream=stream)

    else:

        @flyc.jit
        def launch(
            p_x: fx.Pointer,
            p_x_scale: fx.Pointer,
            p_ids: fx.Pointer,
            p_weight: fx.Pointer,
            p_w_delta: fx.Pointer,
            p_w_scale: fx.Pointer,
            p_out: fx.Pointer,
            M: fx.Int32,
            stream: fx.Stream,
        ):
            CompilationContext.get_current()
            a8w4_gateup_decode(
                p_x,
                p_ids,
                p_weight,
                p_w_delta,
                p_w_scale,
                p_out,
                p_x_scale,
                p_out,  # unread
                M,
            ).launch(grid=(NJ, M * SLOTS, 1), block=(256, 1, 1), stream=stream)

    return launch


def compile_a8w4_down_decode_mixed(
    *, N, K, KP=None, TOPK, E=None, SHARED=True, PREFETCH=None
):
    """The Down decode kernel that quantizes its intermediate rows itself: one wave per routing column (``TOPK``
    waves, at most 16), grid ``(N / 32, M)``, 32 output channels of one token per workgroup (two 16-row blocks read with
    16-byte loads, see "Row-block pairs" in the module docstring). ``N`` = hidden (``N % 32 == 0``), ``K`` = padded I
    (``K % 64 == 0``); mid f32 [M, TOPK, K]; topk_ids int32 / topk_weight f32 [M, TOPK]; w_scale f32 [E, N]; out BF16
    [M, N].

    Without ``SHARED`` (the per-route Down of few tokens) every column is a routed expert's and the launcher is ``(mid,
    mid_scale, topk_ids, topk_weight, w, w_delta, w_scale, out, M, stream)`` (``mid_scale`` unused). With ``SHARED``
    (the mixed Down) the launcher is ``(mid, topk_ids, topk_weight, w, w_delta, w_scale, shared_w, shared_scale, out, M,
    stream)`` and the last column is a shared FP8 expert (``compile_a8w4_gateup_decode_mixed``'s rule: it adds only when
    the last id is ``E`` and the token's first id is not -1), whose wave reads its FP8 Down weight directly: shared_w
    FP8 E4M3FNUZ [N, KP] (K zero-padded to ``KP``, a multiple of 64) in the 16-row x 64-K block layout of the Gate/Up
    (lane ``16 g + n`` holds K group ``g`` of row ``n``, one 16-byte load per block and 16-row block, both blocks of the
    workgroup), scaled by shared_scale f32 [N] per row.

    A routed column's wave quantizes its intermediate row to FP8 (``_quant_fast``, two corrections: the bytes of the
    correctly rounded division), decodes its expert's MXFP4 codes through its LUT copy and scales by the channel
    factor, the row scale and the routing weight (-1 masked: no weight read, nothing added). Every wave runs its column
    on its own (its own LUT copy and LDS row, a wavefront fence between staging and its MFMAs), so all of them issue
    their weight loads first and the workgroup meets only before the sum of their partials (``_tree_sum`` over the
    columns), which is written as BF16, so the output needs no clearing and no separate reduction. ``PREFETCH``: the
    64-K blocks whose loads each wave issues before staging its row (default: all with ``SHARED``, else 3), each
    further one as the MFMA loop consumes one.
    """
    from .stream_a8w4 import _address, _load, _rsrc

    KP = K if KP is None else KP
    assert (
        N % 32 == 0
        and KP % 64 == 0
        and KP >= K
        and down_decode_mixed_fits(K, TOPK, SHARED)
        and (E is not None or not SHARED)
    ), f"unsupported {'mixed ' if SHARED else ''}Down decode shape N={N} K={K} KP={KP} TOPK={TOPK}"
    E = 0 if E is None else E  # traced in the folded-away shared branch only
    KB = K // 64
    KBP = KP // 64
    KQ = a8w4.delta_dwords(K)
    NB2 = N // 32
    R = TOPK - 1 if SHARED else TOPK  # routed columns
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    S_BLOCK = 2 * a8w4.W_BLOCK_BYTES
    if PREFETCH is None:
        # The mixed Down issues all blocks first: measured in one process under CUDA graph replay at the
        # DeepSeek-V4.1 layers M 1-4 (9 blocks), 3 first took 1.00-1.03x the time. The per-route Down issues 3 first:
        # measured per launch (eager) at DeepSeek-V4.1 TP4 (H 5120, I 576, topk 6), M 1 / 2 / 3 0.92x / 0.86x / 0.83x
        # the time of all 9 first (under graph replay 1.01x / 0.97x / 1.00x; Qwen TP2 / TP4, MiMo TP8 / TP4 M 1-3
        # 0.99-1.02x).
        PREFETCH = KB if SHARED else 3
    PREFETCH = min(KB, PREFETCH)

    @fx.struct
    class SharedStorage:
        mq: fx.Array[fx.Uint8, TOPK * K, 16]
        part: fx.Array[fx.Float32, TOPK * 32, 16]
        lut: fx.Array[fx.Uint8, TOPK * _LUT_BYTES, 16]

    # The pointers the waves read first lie at the start of the kernel arguments (one scalar load fetches them); the
    # output, written last, follows.
    @flyc.kernel
    def a8w4_down_decode(
        p_mid: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_shared: fx.Pointer,
        p_shared_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = _wave_index(tid)
        j = gpu.block_idx.x  # 16-row blocks 2 j and 2 j + 1
        token = gpu.block_idx.y

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mq_base = fx.recast_iter(fx.Uint8, lds.mq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        lut_base = fx.add_offset(
            fx.recast_iter(fx.Uint8, lds.lut.ptr),
            fx.make_int_tuple(wave * _LUT_BYTES),
        )
        _lut_table_init(lut_base, lane)

        ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
        tws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_topk_weight))
        row = token * TOPK
        expert = ids[row + wave]
        weight = tws[row + wave]
        first = ids[row]
        x_off = (lane // 16) * 16
        # Each branch loads its own intermediate row (first: it does not depend on the expert id) and the shared
        # expert's wave addresses its row with constants: code common to both branches would be hoisted above the
        # branch, where its wait for the intermediate row would hold back the weight loads.
        shared = (wave == R) if SHARED else _never()
        if shared:
            mid = fx.make_view(
                fxh._as_ptr(p_mid, fx.Float32) + fx.Int64(row + R) * K,
                fx.make_layout(K, 1),
            )
            mids = [mid[lane + 64 * u] for u in range_constexpr(KB)]
            live = (expert == E) & (first >= 0)
            w_rsrc = _rsrc(
                _address(p_shared), live.select(fx.Int64(N * KP), fx.Int64(0))
            )
            s_rsrc = _rsrc(
                _address(p_shared_scale), live.select(fx.Int64(N * 4), fx.Int64(0))
            )
            # raws[b][i]: block i of 16-row block 2 j + b; the first PREFETCH blocks of both now, the rest as the MFMA
            # loop consumes them.
            raws = [[], []]

            def issue_shared(i):
                for b in range_constexpr(2):
                    raws[b].append(
                        _load(w_rsrc, lane * 16, ((2 * j + b) * KBP + i) * S_BLOCK, 4)
                    )

            for i in range_constexpr(PREFETCH):
                issue_shared(i)
            factors = [
                _load(s_rsrc, (lane % 16) * 4, (2 * j + b) * 64, 1)[0].bitcast(
                    fx.Float32
                )
                for b in range_constexpr(2)
            ]
            rocdl.sched_barrier(0)
            amax, row_scale = _row_amax_scale(mids)
            if _fast_div_ok(amax):
                _stage_row_fast(mq_base, R * K, lane, mids, row_scale, 2)
            else:
                _stage_row_ieee(mq_base, R * K, lane, mids, row_scale)
            _wave_fence()
            accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
            for i in range_constexpr(KB):
                if const_expr(0 < i and i + PREFETCH - 1 < KB):
                    issue_shared(i + PREFETCH - 1)
                xop = _lds_load(mq_base, R * K + i * 64 + x_off, fx.Int64, 2, 16)
                for b in range_constexpr(2):
                    w = Vec.from_elements(raws[b][i], fx.Uint32).bitcast(fx.Int64)
                    for c in range_constexpr(2):
                        accs[b] = _mfma(xop[c], w[c], accs[b])
            totals = [
                live.select(
                    Vec(accs[b])[0] * factors[b] * row_scale * weight, fx.Float32(0.0)
                )
                for b in range_constexpr(2)
            ]
            if lane < 16:
                for b in range_constexpr(2):
                    _lds_store(
                        part_base,
                        (R * 32 + b * 16 + lane) * 4,
                        Vec.from_elements([totals[b]], fx.Float32),
                        fx.Float32,
                        4,
                    )
        else:
            base = wave * K
            mid = fx.make_view(
                fxh._as_ptr(p_mid, fx.Float32) + fx.Int64(row + wave) * K,
                fx.make_layout(K, 1),
            )
            mids = [mid[lane + 64 * u] for u in range_constexpr(KB)]
            w_buf = a8w4.expert_bytes(p_weight, expert, W_BYTES, masked=True)
            d_buf = a8w4.expert_bytes(p_w_delta, expert, D_BYTES, masked=True)
            w_voff, d_voff, s_voff = _row_block_pair_offsets(lane, KB, KQ)
            # Each delta load (both rows, 4 blocks) right before the weight blocks it covers; the first PREFETCH
            # blocks now, the rest as the MFMA loop consumes them.
            raws, deltas = [], []

            def issue_routed(i):
                if const_expr(i % 4 == 0):
                    deltas.append(
                        a8w4.load_dwordx2(
                            d_buf,
                            d_voff,
                            fx.Int32((2 * j * KQ + i // 4) * a8w4.D_BLOCK_BYTES),
                        )
                    )
                raws.append(
                    _load_dwordx4(
                        w_buf, w_voff, fx.Int32((2 * j * KB + i) * a8w4.W_BLOCK_BYTES)
                    )
                )

            for i in range_constexpr(PREFETCH):
                issue_routed(i)
            s_buf = a8w4.expert_bytes(p_w_scale, expert, N * 4, masked=True)
            factors = a8w4.load_dwordx2(s_buf, s_voff, fx.Int32(j * 128))
            rocdl.sched_barrier(0)
            amax, row_scale = _row_amax_scale(mids)
            if _fast_div_ok(amax):
                _stage_row_fast(mq_base, base, lane, mids, row_scale, 2)
            else:
                _stage_row_ieee(mq_base, base, lane, mids, row_scale)
            _wave_fence()
            accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
            for i in range_constexpr(KB):
                if const_expr(0 < i and i + PREFETCH - 1 < KB):
                    issue_routed(i + PREFETCH - 1)
                xop = _lds_load(mq_base, base + i * 64 + x_off, fx.Int64, 2, 16)
                accs = _row_block_pair_mfmas(
                    xop, raws[i], deltas[i // 4], lut_base, fx.Uint32(8 * (i % 4)), accs
                )
            # A masked route (id -1) adds nothing (its weight may be anything) and read no weights above.
            totals = [
                (expert >= 0).select(
                    Vec(accs[h])[0]
                    * fx.Uint32(factors[h]).bitcast(fx.Float32)
                    * row_scale
                    * weight,
                    fx.Float32(0.0),
                )
                for h in range_constexpr(2)
            ]
            if lane < 16:
                for h in range_constexpr(2):
                    _lds_store(
                        part_base,
                        (wave * 32 + _pair_channel(lane, h)) * 4,
                        Vec.from_elements([totals[h]], fx.Float32),
                        fx.Float32,
                        4,
                    )
        gpu.barrier()
        if tid < 32:
            parts = [
                _lds_load(part_base, (w * 32 + tid) * 4, fx.Float32, 1, 4)[0]
                for w in range_constexpr(TOPK)
            ]
            y = _tree_sum(parts)
            out = fx.make_view(
                fxh._as_ptr(p_out, fx.BFloat16) + fx.Int64(token) * N,
                fx.make_layout(N, 1),
            )
            out[j * 32 + tid] = fxh._f32_to_bf16(y)

    if SHARED:

        @flyc.jit
        def launch(
            p_mid: fx.Pointer,
            p_ids: fx.Pointer,
            p_topk_weight: fx.Pointer,
            p_weight: fx.Pointer,
            p_w_delta: fx.Pointer,
            p_w_scale: fx.Pointer,
            p_shared: fx.Pointer,
            p_shared_scale: fx.Pointer,
            p_out: fx.Pointer,
            M: fx.Int32,
            stream: fx.Stream,
        ):
            CompilationContext.get_current()
            a8w4_down_decode(
                p_mid,
                p_ids,
                p_topk_weight,
                p_weight,
                p_w_delta,
                p_w_scale,
                p_shared,
                p_shared_scale,
                p_out,
                M,
            ).launch(grid=(NB2, M, 1), block=(64 * TOPK, 1, 1), stream=stream)

    else:

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
            # mid_scale: unread (the routed rows are quantized in the kernel); the shared pointers: unread.
            a8w4_down_decode(
                p_mid,
                p_ids,
                p_topk_weight,
                p_weight,
                p_w_delta,
                p_w_scale,
                p_weight,
                p_w_scale,
                p_out,
                M,
            ).launch(grid=(NB2, M, 1), block=(64 * TOPK, 1, 1), stream=stream)

    return launch
