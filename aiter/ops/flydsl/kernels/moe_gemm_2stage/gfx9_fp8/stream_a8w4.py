# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 MoE weight-streaming Gate/Up and Down kernels (gfx942) for the memory-bound token range.

``aiter.ops.flydsl.moe_a8w4`` uses them above the decode range while each routed expert sees at most a few dozen
tokens on average (``STREAM_MAX_ROWS`` rows per expert there), where the cost is reading the MXFP4 weights once.
Both kernels walk moe_sorting blocks of ``BLOCK_M`` (16/32/64) rows in sub-blocks of 16
and use MFMA 16x16x32 FP8 with the weights as A and 16 sorted rows as B. Every wave owns a few 16-row
weight blocks of the block's expert and reads them straight into VGPRs in the ``common_a8w4`` layout
(several 64-K steps in flight; the weight loads can carry the non-temporal cache bit, ``W_AUX``),
decodes FP4 -> FP8 with the per-group LUT and multiplies. Lane
``16b + t`` of the B operand holds bytes ``16b .. 16b+15`` of row ``t`` in each 64-K block, which is
the K permutation of the weight layout; the rows come straight from global memory or, with ``SHARE``
and several waves per workgroup, are loaded once per workgroup into LDS (in Gate/Up a ring of
``2 * SYNC`` 64-K steps: step k + SYNC is written during step k, one barrier every ``SYNC`` steps; in Down
the block's whole rows).

Row pairs: the weights are read 16 bytes per lane, two 16-row blocks per load. Lane ``16b + i`` loads layout lanes
``16b + 2(i & 7)`` and ``+ 1`` (two adjacent rows, K group b) of the first block (``i < 8``) or of the second
(``i >= 8``), so every quad of lanes reads 64 contiguous bytes, and the two rows' delta dwords are adjacent (one
8-byte load). MFMA chain p takes row p of every lane's pair: row R of the 16 x 16 tile is channel ``2(R & 7) + p`` of
the first (R < 8) or second block, and each lane of the C layout holds 8 consecutive channels of one block. The pairs
are Gate/Up's gate block and the up block of the same channels (lanes l and l ^ 32 then trade half of the values,
so each lane holds gate and up of 4 channels for the epilogue) and Down's blocks 2u and 2u + 1 of the wave (lanes hold
8 consecutive channels for a 16-byte BF16 store, or with ``ATOMIC`` a channel pair per row). The decode reads the LUT
pair of each row and step from a table that every wave writes into LDS (``_lut_table``: one VALU and an 8-byte LDS
read instead of four VALU) and shifts the nibbles of a row's two dwords with 64-bit shifts (``_decode_row``); the
bytes are those of ``common_a8w4.fp8_lut`` / ``fp4x8_to_fp8``, and every channel's MFMAs run in the same order as
with one block per 8-byte load, so the results are unchanged bit for bit.

Gate/Up: wave ``w`` of workgroup ``(g, block)`` owns gate 16-row blocks ``(g * W + w) * G + j`` and
the matching up blocks, accumulates over the whole K and writes SiLU(gate) * up as float32 by
(token, slot), the same epilogue arithmetic as ``gemm1_a8w4``; with ``QUANT`` (the waves cover all of
I) it quantizes those rows to FP8 itself, exchanging the row maxima through LDS. Down: the wave keeps
its rows' FP8 activations (all of K = I) in VGPRs (with ``LDS_ACTS`` it reads them from the shared LDS rows
at every 64-K step), stages its ``R * 16`` channel factors in LDS and
streams ``R`` 16-row blocks of the hidden dimension, writing BF16 rows by sorted position (padding
rows are not written), or, with ``ATOMIC``, adding each route into its token's output row with BF16x2 atomics.
``compile_stream_prep`` (sorting, input quantization and the zeroed output in one launch) and ``compile_route_sum``
(sum of a token's routes) replace the prefill path's Opus sorting, input quantization and ``sorted_sum``.

Raw operations: these kernels address global memory through raw buffer descriptors built from 64-bit addresses
(``_rsrc`` / ``_load`` / ``_store``: per-lane byte offsets, ``_DROP`` past every buffer instead of a branch), run
MFMA on register operands built in the kernel and keep LDS as a flat byte array; each raw ``rocdl`` / ``llvm`` /
``scf`` call names the stable FlyDSL 0.5.3 API it lacks at the call (``aiter/ops/flydsl/AGENTS.md``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith as farith
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common_a8w4 as a8w4

# Byte offset past every buffer used here: loads return 0, stores are dropped.
_DROP = 0x7FFFFF00
_LOG2_EXP1 = -1.4426950408889634


def _address(p):
    return fx.Int64(fx.ptrtoint(p))


def _rsrc(address, nbytes):
    # Raw descriptor (llvm.inttoptr + rocdl.make_buffer_rsrc) for the raw loads / stores of _load / _store, which
    # have no typed form (see there). The flags 0x27000 are those of the typed fx.rocdl.make_buffer_ptr on CDNA;
    # the base is a 64-bit address computed in the kernel (an expert's weights or channel factors, routing tables,
    # activation rows) and the byte count depends on the runtime token count.
    pointer = llvm.inttoptr(ir.Type.parse("!llvm.ptr"), fx.Int64(address).ir_value())
    return rocdl.make_buffer_rsrc(
        ir.Type.parse("!llvm.ptr<8>"),
        pointer,
        fx.Int16(0).ir_value(),
        fx.Int64(nbytes).ir_value(),
        fx.Int32(0x27000).ir_value(),
    )


def _load(rsrc, voffset, soffset, dwords, aux=0):
    """Raw buffer load of ``dwords`` dwords -> list of Uint32 (``aux``: cache policy bits, 2 = non-temporal)."""
    # Raw buffer load: the typed fx.copy with a BufferCopy atom addresses through a view layout and fills a vector
    # fragment, while these loads take a per-lane VGPR byte offset (_DROP for lanes past the data: the range
    # check returns 0, no branch) plus an SGPR one and the cache policy bits, and return the dwords as scalars.
    if dwords == 1:
        value = rocdl.raw_ptr_buffer_load(
            fx.Int32.ir_type,
            rsrc,
            fx.Int32(voffset).ir_value(),
            fx.Int32(soffset).ir_value(),
            aux=aux,
        )
        return [fx.Uint32(value)]
    value = rocdl.raw_ptr_buffer_load(
        ir.VectorType.get([dwords], fx.Int32.ir_type),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=aux,
    )
    vec = Vec(value).bitcast(fx.Uint32)
    return [vec[i] for i in range(dwords)]


def _store(rsrc, words, voffset):
    # Raw buffer store: the typed fx.copy store takes its offset from a view layout; here a per-lane byte offset,
    # _DROP drops the store in the hardware range check (no branch).
    words = list(words)
    # A one-element vector store cannot be scalarized by the backend: store a plain i32.
    value = (
        fx.Int32(words[0])
        if len(words) == 1
        else Vec.from_elements(words, fx.Uint32).bitcast(fx.Int32)
    )
    rocdl.raw_ptr_buffer_store(
        value.ir_value(),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    )


def _uniform(value):
    # Raw readfirstlane: FlyDSL 0.5.3 has no typed wave-uniform broadcast (fx.gpu offers only the shuffle family,
    # whose result is a VGPR); the value stays scalar for the offsets and branches built on it.
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _atomic_add_bf16x2_if(cond, base, offset, word):
    """``*(base + offset) += word`` (two BF16, one agent-scope atomic, no return) on the lanes where ``cond``."""
    # Raw scf.IfOp: FlyDSL 0.5.3 lowers a dynamic `if` only in the AST of a @flyc.kernel / @flyc.jit function and
    # has no typed if builder for a plain helper. Raw llvm.inttoptr + llvm.AtomicRMWOp (fadd on <2 x bf16>, no
    # return: global_atomic_pk_add_bf16): the typed fx.inttoptr + fx.atomic_add take the alignment from an
    # fx.PointerType, the address here is a per-lane 64-bit integer with the 4-byte alignment given per access.
    loc = ir.Location.unknown()
    if_op = scf.IfOp(_raw(cond), [], has_else=False, loc=loc)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        pointer = llvm.inttoptr(
            ir.Type.parse("!llvm.ptr<1>"), (base + fx.Int64(offset)).ir_value()
        )
        pair = Vec.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.BFloat16)
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.fadd,
            pointer,
            _raw(pair),
            llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        )
        scf.YieldOp([], loc=loc)


def _f32(word):
    return fx.Uint32(word).bitcast(fx.Float32)


def _pack64(lo, hi):
    return Vec.from_elements([lo, hi], fx.Uint32).bitcast(fx.Int64)[0]


def _decode(raw, lut):
    """One layout dword (8 FP4) -> 8 FP8 E4M3FNUZ as one i64 MFMA operand (``common_a8w4`` order)."""
    lo, hi = lut
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    even = ((raw << 4) & sign) | a8w4._perm(hi, lo, raw & mag)
    odd = (raw & sign) | a8w4._perm(hi, lo, (raw >> 4) & mag)
    return _pack64(even, odd)


def _decode_row(raw, lut):
    """The two layout dwords of one row of a 64-K block (16 FP4) -> the i64 operands of its two MFMAs, the bytes of
    ``_decode``: the nibble shifts of both dwords are single 64-bit shifts (the bits crossing the dword boundary are
    masked off), two VALU fewer per row."""
    lo, hi = lut
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    wide = Vec.from_elements(list(raw), fx.Uint32).bitcast(fx.Uint64)[0]
    right = Vec.from_elements([wide >> 4], fx.Uint64).bitcast(fx.Uint32)
    left = Vec.from_elements([wide << 4], fx.Uint64).bitcast(fx.Uint32)
    ops = []
    for c in range(2):
        r = fx.Uint32(raw[c])
        even = (fx.Uint32(left[c]) & sign) | a8w4._perm(hi, lo, r & mag)
        odd = (r & sign) | a8w4._perm(hi, lo, fx.Uint32(right[c]) & mag)
        ops.append(_pack64(even, odd))
    return ops


def _mfma(a, b, acc):
    # Raw MFMA on register values: the typed MMA atom (fx.make_mma_atom with fx.gemm) takes fragments described by
    # layouts, while each operand here is one i64 built in registers (a decoded weight dword pair, 16 bytes of an
    # activation row) and used by the next MFMA.
    return Vec(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            ir.VectorType.get([4], fx.Float32.ir_type), [a, b, acc, 0, 0, 0]
        )
    )


def _zeros():
    return Vec.filled(4, 0.0, fx.Float32)


def _lds(address):
    # LDS by byte address (raw llvm.inttoptr to !llvm.ptr<3>, then llvm.load / store / atomicrmw at the callers):
    # the typed fx.inttoptr + fx.generic_load / generic_store / atomic_add take the alignment from an
    # fx.PointerType, while these accesses compute per-lane byte addresses (XOR swizzles, ring and scratch slots)
    # into one flat allocation and give the alignment (4 or 16 bytes) per access.
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_base(nbytes):
    storage = fx.SharedAllocator().allocate(fx.Array[fx.Int8, nbytes, 16]).peek()
    return fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(nbytes, 1)))))


def _fp8_raw(q):
    """``quant_a8w4`` rounding of ``x / scale`` before -0 -> +0: clamp to +-240, RNE to FP8 E4M3FNUZ (a byte)."""
    q = q.maximumf(fx.Float32(-240.0)).minimumf(fx.Float32(240.0))
    # FlyDSL 0.5.3 cannot lower the typed `.to(fx.Float8E4M3FNUZ)` (arith.truncf to FP8 has no LLVM translation);
    # the rocdl op is v_cvt_pk_fp8_f32, two values per op.
    return fx.Uint32(
        rocdl.cvt_pk_fp8_f32(T.i32, q, fx.Float32(0.0), fx.Int32(0), False)
    ) & fx.Uint32(0xFF)


def _fp8_byte(q):
    """``quant_a8w4`` rounding of ``x / scale``: clamp to +-240, RNE to FP8 E4M3FNUZ, -0 -> +0 (as a byte)."""
    byte = _fp8_raw(q)
    return (byte == fx.Uint32(0x80)).select(fx.Uint32(0), byte)


def _clamp240(q):
    return fx.Float32(
        farith.minnumf(farith.maxnumf(q, fx.Float32(-240.0)), fx.Float32(240.0))
    )


def _uniform_if(cond, then_fn, else_words):
    """``then_fn()`` (dwords) when the wave-uniform ``cond`` holds, else ``else_words``, as one scf.if."""
    # Raw scf.IfOp with results: no typed if builder for a plain helper (see _atomic_add_bf16x2_if).
    loc = ir.Location.unknown()
    if_op = scf.IfOp(_raw(cond), [T.i32] * len(else_words), has_else=True, loc=loc)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        scf.YieldOp([fx.Uint32(w).ir_value() for w in then_fn()], loc=loc)
    if len(if_op.regions[1].blocks) == 0:
        if_op.regions[1].blocks.append(*[])
    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
        scf.YieldOp([fx.Uint32(w).ir_value() for w in else_words], loc=loc)
    return [fx.Uint32(r) for r in if_op.results]


# Relative bracket of the reciprocal quotient: 2^-20, far above its error of under 3 ulp (2^-21).
_BRACKET = 2.0**-20


def _fp8_bytes(values, scale):
    """FP8 bytes of ``values / scale`` exactly as ``quant_a8w4``: IEEE division, clamp +-240, RNE, -0 -> +0.

    ``v * rcp(scale)`` is within 3 ulp of ``v / scale``; when the FP8 roundings of the quotient scaled by
    1 -+ 2^-20 agree, no FP8 rounding boundary lies within the error of either quotient and that byte
    is exact (rounding is monotonic). Otherwise (some lane near a boundary, or a scale that is not a
    finite normal number) the wave redoes the IEEE divisions.
    """
    # Bare v_rcp_f32 (raw rocdl.rcp): FlyDSL 0.5.3 has no typed approximate reciprocal (the f32 division lowers to
    # the correctly rounded divide, several VALU ops per element); with the bracket below the bytes stay exact.
    y = fx.Float32(rocdl.rcp(T.f32, _raw(scale)))
    y_lo, y_hi = y * (1.0 - _BRACKET), y * (1.0 + _BRACKET)
    # scale != scale: the NaN test of the device value (an fx.Float32 comparison, not a Python self-comparison).
    miss = (scale < 2.0**-126) | (scale > 3.0e38) | (scale != scale)  # noqa: PLR0124
    fast = []
    for v in values:
        # v_cvt_pk_fp8_f32 of both bracket ends (no typed FP8 conversion, see _fp8_raw).
        pair = fx.Uint32(
            rocdl.cvt_pk_fp8_f32(
                T.i32, _clamp240(v * y_lo), _clamp240(v * y_hi), fx.Int32(0), False
            )
        )
        byte = pair & fx.Uint32(0xFF)
        miss = miss | (byte != ((pair >> 8) & fx.Uint32(0xFF)))
        fast.append(byte)
    # rocdl.ballot: FlyDSL 0.5.3 has no typed wave vote (fx.gpu offers only the shuffle family); the 64-lane mask
    # makes the redo branch wave-uniform.
    redo = fx.Int64(rocdl.ballot(T.i64, _raw(miss))) != fx.Int64(0)
    exact = _uniform_if(redo, lambda: [_fp8_raw(v / scale) for v in values], fast)
    return [(b == fx.Uint32(0x80)).select(fx.Uint32(0), b) for b in exact]


def _pack_bytes(bytes4):
    word = bytes4[0]
    for v in range(1, 4):
        word = word | (bytes4[v] << (8 * v))
    return word


def _umax(a, b):
    return (a > b).select(a, b)


def _abs_bits_max(values):
    """Bits of max |v| (non-negative floats order as unsigned integers; NaN beats Inf, so it propagates)."""
    peak = values[0].bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF)
    for v in values[1:]:
        peak = _umax(peak, v.bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF))
    return peak


def _lanes_max(peak, steps):
    for step in steps:
        other = (
            fx.Float32(peak.bitcast(fx.Float32))
            .shuffle_xor(step, 64)
            .bitcast(fx.Uint32)
        )
        peak = _umax(peak, other)
    return peak


def _bf16_pair(lo, hi):
    """Two f32 -> one dword of BF16 (``lo`` in the low half), round to nearest even (finite values and Inf)."""
    words = []
    for value in (lo, hi):
        bits = value.bitcast(fx.Uint32)
        words.append(bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1)))
    return a8w4._perm(words[1], words[0], fx.Uint32(0x07060302))


def _lds_store(address, words):
    words = list(words)
    value = (
        fx.Int32(words[0])
        if len(words) == 1
        else Vec.from_elements(words, fx.Uint32).bitcast(fx.Int32)
    )
    # Raw llvm.store at an LDS byte address (see _lds).
    llvm.store(value.ir_value(), _lds(address), alignment=4 * len(words))


def _lds_load4(address):
    # Raw 16-byte llvm.load at an LDS byte address (see _lds).
    return Vec(
        llvm.load(ir.VectorType.get([4], fx.Int32.ir_type), _lds(address), alignment=16)
    ).bitcast(fx.Uint32)


# LDS bytes of the FP8 LUT table (``_lut_table``): 16 entries of 8 bytes, of which deltas 0 .. MAX_DELTA are read.
_LUT_LDS = 16 * 8


def _lut_table(lut_lds, lane):
    """The ``common_a8w4.fp8_lut`` pairs of every delta in LDS, the entry of the stored delta byte B (= 8 * delta) at
    byte B: lane l writes entry l & 15. Every wave writes the whole (identical) table, so a wave reads its own writes
    (LDS accesses of a wave execute in order) and no barrier is needed."""
    d = fx.Uint32((lane & 15) * 8) * fx.Uint32(0x01010101)
    lo = (fx.Uint32(0x6C686080) - d) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - d
    _lds_store(lut_lds + (lane & 15) * 8, [lo, hi])


def _lut_at(lut_lds, deltas, j):
    """``common_a8w4.fp8_lut(deltas, j * 0x01010101)`` from the LDS table: one VALU (the byte) and one 8-byte LDS read
    instead of four VALU."""
    byte = (fx.Uint32(deltas) >> (8 * j)) & fx.Uint32(0xFF)
    # Raw 8-byte llvm.load at an LDS byte address (see _lds).
    pair = Vec(
        llvm.load(
            ir.VectorType.get([2], fx.Int32.ir_type),
            _lds(lut_lds + fx.Int32(byte)),
            alignment=8,
        )
    ).bitcast(fx.Uint32)
    return pair[0], pair[1]


def _lds_barrier():
    """Wait for this wave's LDS accesses, then the workgroup barrier (no wait on global loads)."""
    # Raw s_waitcnt(lgkmcnt=0) + s_barrier, no memory fence: FlyDSL 0.5.3's gpu.barrier lowers to fence release /
    # s_barrier / fence acquire at workgroup scope over every address space, which also orders the global weight
    # loads in flight (several 64-K steps ahead) against the barrier; it has no typed barrier without the fences.
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def _row_scale(peak):
    scale = peak / 240.0
    return (scale == 0.0).select(fx.Float32(1.0), scale)


class _Block:
    """Per-wave view of the 16-row sub-blocks of one moe_sorting block (B rows), with the block's expert."""

    def __init__(self, position, token, slot, expert):
        self.position, self.token, self.slot, self.expert = (
            position,
            token,
            slot,
            expert,
        )


def _first(block, subs):
    """The first ``subs`` sub-blocks (a plain function: the kernel AST rewriter treats a method call
    inside a dynamic ``if`` as mutating its receiver)."""
    return _Block(
        block.position[:subs], block.token[:subs], block.slot[:subs], block.expert
    )


def _block_head(
    blk, block_m, M, lane, p_sorted_ids, p_sorted_experts, p_num_valid, rows
):
    """``(live, block)`` with every load in one round trip: the uniform count of the block's 16-row
    sub-blocks holding a real row (0 past the sorted range), its expert and this lane's sorted ids.

    Real rows of an expert come before its padding, so the live sub-blocks are a prefix.
    """
    num_valid = _uniform(fx.Int32(_load(_rsrc(_address(p_num_valid), 4), 0, 0, 1)[0]))
    expert = _uniform(
        fx.Int32(
            _load(
                _rsrc(_address(p_sorted_experts), rows // block_m * 4), blk * 4, 0, 1
            )[0]
        )
    )
    sorted_rsrc = _rsrc(_address(p_sorted_ids), rows * 4)
    position = [blk * block_m + 16 * s + (lane & 15) for s in range(block_m // 16)]
    ids = [_load(sorted_rsrc, p * 4, 0, 1)[0] for p in position]
    token = [fx.Int32(i & fx.Uint32(0xFFFFFF)) for i in ids]
    slot = [fx.Int32((i >> 24) & fx.Uint32(0xFF)) for i in ids]
    count = fx.Int32(0)
    for t in token:
        # Raw readlane: FlyDSL 0.5.3 has no typed readlane (fx.gpu offers only the shuffle family, whose result
        # is a VGPR); lane 0's token of each sub-block stays scalar for the uniform count.
        head = fx.Int32(
            rocdl.readlane(fx.Int32.ir_type, t.ir_value(), fx.Int32(0).ir_value())
        )
        count = count + (head < M).select(fx.Int32(1), fx.Int32(0))
    live = (blk * block_m < num_valid).select(count, fx.Int32(0))
    return live, _Block(position, token, slot, expert)


def compile_a8w4_gateup_stream(
    *,
    N,
    K,
    TOPK,
    BLOCK_M,
    G=4,
    W=1,
    PREFETCH=(6, 4),
    QUANT=False,
    SHARE=True,
    E=None,
    SYNC=1,
    W_AUX=0,
    SWIGLU_LIMIT=0.0,
    ACT_STEPS=1,
):
    """Build the streaming A8W4 Gate/Up launcher.

    ``N`` = 2 * I (gate rows ``[0, I)``, up rows ``[I, 2I)``, ``I % (16 * G * W) == 0``), ``K`` = hidden
    (``K % 64 == 0``), ``BLOCK_M`` the sorting block (16, 32 or 64). Launch arguments: x FP8 [M, K],
    w and w_delta (``common_a8w4`` layout, uint8), out, sorted_ids, sorted_expert_ids, num_valid_ids,
    w_scale f32 [E, N], x_scale f32 [M], out_scale, then M, task_num, stream. Without ``QUANT`` out is
    f32 [M, TOPK, I] and out_scale unused; with ``QUANT`` one workgroup covers all I of its block
    (``W = I / (16 * G)``) and quantizes each (token, slot) row like ``quant_a8w4`` (amax over I
    exchanged through LDS): out FP8 [M * TOPK, I], out_scale f32 [M * TOPK]. With ``SHARE`` and ``W > 1``
    the waves of a workgroup load the block's activations once into LDS (ring of ``2 * SYNC`` 64-K steps,
    step k + SYNC written during step k, one barrier every ``SYNC`` steps) instead of each wave reading all
    of them. ``ACT_STEPS`` > 1 (with ``SHARE``): each lane loads 16 (or 8) bytes covering ``ACT_STEPS`` 64-K steps
    of one row, one load per lane every ``ACT_STEPS`` steps instead of every step (fewer vector memory
    instructions); ``SYNC`` becomes ``ACT_STEPS``, the loads run two groups ahead and a group is written at the
    end of the group before it. ``SWIGLU_LIMIT`` > 0 clamps gate and up before SiLU (``common_a8w4.swiglu_clamp``).
    """
    I = N // 2
    if QUANT:
        W = I // (16 * G)
    assert I % (16 * G * W) == 0, f"I={I} must be a multiple of {16 * G * W}"
    assert K % 64 == 0, f"K={K} must be a multiple of 64"
    assert BLOCK_M in (16, 32, 64), f"BLOCK_M={BLOCK_M} must be 16, 32 or 64"
    KB, KQ = K // 64, a8w4.delta_dwords(K)
    W_BYTES, D_BYTES = a8w4.weight_bytes(N, K), a8w4.delta_bytes(N, K)
    UP = I // 16
    # Several waves share the block's activations through LDS: each lane of the first LOADERS waves loads
    # LOADER bytes per 64-K step; with more waves the rest write a scratch slot that nobody reads.
    SHARE = SHARE and W > 1
    SLOT = BLOCK_M * 64
    if ACT_STEPS > 1:
        assert (
            SHARE and not QUANT
        ), "ACT_STEPS > 1 needs the light-wave tile with shared activations"
        assert (
            ACT_STEPS & (ACT_STEPS - 1) == 0 and KB % ACT_STEPS == 0
        ), f"K={K} with ACT_STEPS={ACT_STEPS}"
        SYNC = ACT_STEPS
        LOADER = ACT_STEPS * SLOT // (64 * W)
        assert (
            LOADER * 64 * W == ACT_STEPS * SLOT
        ), f"BLOCK_M={BLOCK_M}, W={W}: {ACT_STEPS} steps per load"
        LOADERS = W
    else:
        LOADER = max(4, SLOT // (64 * W))
        LOADERS = SLOT // (64 * LOADER)
    assert not SHARE or LOADER in (
        4,
        8,
        16,
    ), f"BLOCK_M={BLOCK_M} with W={W} waves: {LOADER} bytes per lane"
    assert SYNC <= min(
        PREFETCH
    ), f"SYNC={SYNC}: the activations of step k + SYNC must be in flight at step k"
    RING = 2 * SYNC
    ACT_LDS = RING * SLOT + (64 * LOADER if LOADERS < W else 0) if SHARE else 0

    def silu_mul(S, block, acc, rbs, chan_off, expert, M, p_w_scale, p_x_scale):
        """SiLU(gate) * up per [gate block j][sub-block s] (4 channels ``chan_off`` .. + 3 of one row per lane), as
        ``gemm1_a8w4``."""
        ws_rsrc = _rsrc(_address(p_w_scale) + fx.Int64(expert) * (N * 4), N * 4)
        xs_rsrc = _rsrc(_address(p_x_scale), M * 4)
        x_scale = [_f32(_load(xs_rsrc, t * 4, 0, 1)[0]) for t in block.token]
        channels = [rbs[j] * 16 + chan_off for j in range(G)]
        values = []
        for j in range(G):
            gate_scale = [_f32(v) for v in _load(ws_rsrc, channels[j] * 4, 0, 4)]
            up_scale = [_f32(v) for v in _load(ws_rsrc, (channels[j] + I) * 4, 0, 4)]
            per_sub = []
            for s in range(S):
                gate, up = acc[j][s], acc[G + j][s]
                row = []
                for v in range(4):
                    g = gate[v] * gate_scale[v] * x_scale[s]
                    u = up[v] * up_scale[v] * x_scale[s]
                    g, u = a8w4.swiglu_clamp(g, u, SWIGLU_LIMIT)
                    # Bare v_exp_f32 / v_rcp_f32 (raw rocdl): fx.math.exp2 and the f32 division lower to the
                    # denormal-scaled exp and the correctly rounded divide, several VALU ops more per element; the
                    # SiLU tolerance is met with the approximations (the arithmetic of gemm1_a8w4's epilogue).
                    t = rocdl.exp2(T.f32, _raw(g * _LOG2_EXP1))
                    row.append(fx.Float32((g * rocdl.rcp(T.f32, 1.0 + t)) * u))
                per_sub.append(row)
            values.append(per_sub)
        return channels, values

    def store_f32(S, block, channels, values, M, p_out):
        out_rsrc = _rsrc(_address(p_out), M * (TOPK * I * 4))
        for j in range(G):
            for s in range(S):
                token = block.token[s]
                offset = ((token * TOPK + block.slot[s]) * I + channels[j]) * 4
                offset = (token < M).select(offset, fx.Int32(_DROP))
                _store(
                    out_rsrc,
                    Vec.from_elements(values[j][s], fx.Float32).bitcast(fx.Uint32),
                    offset,
                )

    def store_fp8(
        S, block, channels, values, lane, wave, M, amax_lds, p_out, p_out_scale
    ):
        """Row amax over this wave's channels -> LDS -> amax over the workgroup -> FP8 bytes + scale."""
        out_rsrc = _rsrc(_address(p_out), M * (TOPK * I))
        scale_rsrc = _rsrc(_address(p_out_scale), M * (TOPK * 4))
        row_lane = lane & 15
        for s in range(S):
            peak = _lanes_max(
                _abs_bits_max([values[j][s][v] for j in range(G) for v in range(4)]),
                (16, 32),
            )
            # Raw llvm.store / llvm.load of the row maxima at LDS byte addresses (see _lds).
            llvm.store(
                fx.Int32(peak).ir_value(),
                _lds(amax_lds + ((s * 16 + row_lane) * W + wave) * 4),
                alignment=4,
            )
        gpu.barrier()
        for s in range(S):
            slots = [
                fx.Uint32(
                    llvm.load(
                        fx.Int32.ir_type,
                        _lds(amax_lds + ((s * 16 + row_lane) * W + w) * 4),
                        alignment=4,
                    )
                )
                for w in range(W)
            ]
            peak = slots[0]
            for other in slots[1:]:
                peak = _umax(peak, other)
            scale = _row_scale(peak.bitcast(fx.Float32))
            token = block.token[s]
            route = token * TOPK + block.slot[s]
            live = token < M
            q = _fp8_bytes([values[j][s][v] for j in range(G) for v in range(4)], scale)
            for j in range(G):
                _store(
                    out_rsrc,
                    [_pack_bytes(q[4 * j : 4 * j + 4])],
                    live.select(route * I + channels[j], fx.Int32(_DROP)),
                )
            first = live & (lane < 16) & (wave == 0)
            _store(
                scale_rsrc,
                [scale.bitcast(fx.Uint32)],
                first.select(route * 4, fx.Int32(_DROP)),
            )

    def body(
        S,
        lut_lds,
        act_lds,
        amax_lds,
        block,
        lane,
        wave,
        group,
        M,
        p_x,
        p_w,
        p_d,
        p_out,
        p_w_scale,
        p_x_scale,
        p_out_scale,
    ):
        expert = block.expert
        w_rsrc = _rsrc(_address(p_w) + fx.Int64(expert) * W_BYTES, W_BYTES)
        d_rsrc = _rsrc(_address(p_d) + fx.Int64(expert) * D_BYTES, D_BYTES)
        x_rsrc = _rsrc(_address(p_x), M * K)
        base = (group * W + wave) * G
        rbs = [base + j for j in range(G)]
        # Row pairs: lane 16b + i loads layout lanes 16b + 2(i & 7) and + 1 (rows 2(i & 7) and + 1, K group b) of
        # gate block j (i < 8) or of the up block j + UP (i >= 8): one 16-byte load per lane and 64-K step covers a
        # gate and an up block (each quad of lanes 64 contiguous bytes), and the rows' two delta dwords are adjacent
        # (one 8-byte load every four steps).
        up_lane = (lane >> 3) & 1
        w_voff = (
            (lane >> 4) * 128
            + (lane & 7) * 16
            + up_lane * (UP * KB * a8w4.W_BLOCK_BYTES)
        )
        d_voff = (
            (lane >> 5) * 64 + (lane & 7) * 8 + up_lane * (UP * KQ * a8w4.D_BLOCK_BYTES)
        )

        def load_w(kb):
            w = [
                _load(w_rsrc, w_voff, (rb * KB + kb) * a8w4.W_BLOCK_BYTES, 4, W_AUX)
                for rb in rbs
            ]
            d = None
            if kb % 4 == 0:
                d = [
                    _load(
                        d_rsrc,
                        d_voff,
                        (rb * KQ + kb // 4) * a8w4.D_BLOCK_BYTES,
                        2,
                        W_AUX,
                    )
                    for rb in rbs
                ]
            return w, d

        if SHARE and ACT_STEPS > 1:
            # The same ring, filled ACT_STEPS steps at a time: of the block's rows of ACT_STEPS * 64 bytes, lane L
            # loads bytes L * LOADER .. + LOADER - 1 (all in one row and one 64-K step) and writes them to that step.
            span_log2 = (ACT_STEPS * 64).bit_length() - 1
            idx = (wave * 64 + lane) * LOADER
            ld_sub = idx >> (span_log2 + 4)
            ld_row = (idx >> span_log2) & 15
            off = idx & ((1 << span_log2) - 1)
            within = off & 63
            token = block.token[0]
            for s in range(1, S):
                token = (ld_sub == s).select(block.token[s], token)
            # Raw ds_bpermute (byte address = 4 * source lane): fx.gpu.shuffle_idx scales the lane and range-checks
            # it on every call. Each loader lane fetches the token of the row it loads.
            token = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(ld_row * 4), _raw(token)))
            ld_voff = (ld_sub < S).select(token * K + off, fx.Int32(_DROP))
            ld_lds = (
                act_lds
                + (off >> 6) * SLOT
                + ld_sub * 1024
                + ld_row * 64
                + (((within >> 4) ^ ((ld_row >> 1) & 3)) << 4)
                + (within & 15)
            )
            row = lane & 15
            rd_lds = act_lds + row * 64 + (((lane >> 4) ^ ((row >> 1) & 3)) << 4)

            def load_a(kb):
                return _load(x_rsrc, ld_voff, kb * 64, LOADER // 4)

            def write_a(kb, words):
                _lds_store(ld_lds + (kb % RING) * SLOT, words)

            def read_a(kb):
                xs = [
                    _lds_load4(rd_lds + (kb % RING) * SLOT + s * 1024) for s in range(S)
                ]
                return [(_pack64(x[0], x[1]), _pack64(x[2], x[3])) for x in xs]

        elif SHARE:
            # The block's FP8 rows of one 64-K step, [sub-block][row][64 B] with the 16-byte chunks of row t
            # XOR-swizzled by (t >> 1) & 3, staged in a ring of two LDS slots. Each lane of the workgroup
            # loads LOADER bytes of it; lane 16b + t then reads its 16 bytes of row t.
            within = (wave * (64 * LOADER) + lane * LOADER) & 1023
            ld_sub = (wave * (64 * LOADER)) >> 10
            ld_row = within >> 6
            ld_chunk = (within >> 4) & 3
            token = block.token[0]
            for s in range(1, S):
                token = (ld_sub == s).select(block.token[s], token)
            # Raw ds_bpermute: the loader lane's row token (see the ACT_STEPS branch above).
            token = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(ld_row * 4), _raw(token)))
            ld_voff = (ld_sub < S).select(token * K + (within & 63), fx.Int32(_DROP))
            ld_lds = (
                act_lds
                + ld_sub * 1024
                + ld_row * 64
                + ((ld_chunk ^ ((ld_row >> 1) & 3)) << 4)
                + (within & 15)
            )
            ld_slots = [ld_lds + p * SLOT for p in range(RING)]
            if LOADERS < W:
                scratch = act_lds + RING * SLOT + lane * LOADER
                ld_slots = [(wave < LOADERS).select(a, scratch) for a in ld_slots]
            row = lane & 15
            rd_lds = act_lds + row * 64 + (((lane >> 4) ^ ((row >> 1) & 3)) << 4)

            def load_a(kb):
                return _load(x_rsrc, ld_voff, kb * 64, LOADER // 4)

            def write_a(kb, words):
                _lds_store(ld_slots[kb % RING], words)

            def read_a(kb):
                xs = [
                    _lds_load4(rd_lds + (kb % RING) * SLOT + s * 1024) for s in range(S)
                ]
                return [(_pack64(x[0], x[1]), _pack64(x[2], x[3])) for x in xs]

        else:
            a_voff = [t * K + (lane >> 4) * 16 for t in block.token]

            def load_a(kb):
                return [_load(x_rsrc, v, kb * 64, 4) for v in a_voff]

            def read_a(kb):
                return None

        # 64-K steps in flight (fewer with more sub-blocks to bound VGPRs). Activations are issued
        # first in each step so waiting for them never waits for later weights.
        P = min(PREFETCH[0] if S == 1 else PREFETCH[1], KB)
        if ACT_STEPS > 1:
            # Groups 0 and 1 of ACT_STEPS steps; group g + 2 is issued at the start of group g.
            acts = {
                kb: load_a(kb) for kb in range(0, min(2 * ACT_STEPS, KB), ACT_STEPS)
            }
        else:
            acts = {kb: load_a(kb) for kb in range(P)}
        stages = [load_w(kb) for kb in range(P)]
        if SHARE:
            for kb in range(0, min(SYNC, KB), ACT_STEPS):
                write_a(kb, acts.pop(kb))
            _lds_barrier()
        acc = [[[_zeros() for _ in range(S)] for _ in range(2)] for _ in rbs]
        deltas = None
        for kb in range(KB):
            if ACT_STEPS > 1:
                if kb % ACT_STEPS == 0 and kb + 2 * ACT_STEPS < KB:
                    acts[kb + 2 * ACT_STEPS] = load_a(kb + 2 * ACT_STEPS)
                if kb + P < KB:
                    stages.append(load_w(kb + P))
            elif kb + P < KB:
                acts[kb + P] = load_a(kb + P)
                stages.append(load_w(kb + P))
            w, d = stages[kb]
            stages[kb] = None
            if d is not None:
                deltas = d
            if SHARE:
                # Step k + SYNC goes into the slot of step k - SYNC, read before the last barrier.
                if ACT_STEPS == 1 and kb + SYNC < KB:
                    write_a(kb + SYNC, acts.pop(kb + SYNC))
                b = read_a(kb)
            else:
                b = [(_pack64(x[0], x[1]), _pack64(x[2], x[3])) for x in acts.pop(kb)]
            for j in range(len(rbs)):
                # Chain p takes row p of every lane's pair (dwords 2p, 2p + 1 of its load).
                for p in range(2):
                    lut = _lut_at(lut_lds, deltas[j][p], kb % 4)
                    ops = _decode_row(w[j][2 * p : 2 * p + 2], lut)
                    for s in range(S):
                        for c in range(2):
                            acc[j][p][s] = _mfma(ops[c], b[s][c], acc[j][p][s])
            if ACT_STEPS > 1 and (kb + 1) % ACT_STEPS == 0 and kb + 1 < KB:
                # The next group goes into the slots of the previous one, read before the last barrier.
                write_a(kb + 1, acts.pop(kb + 1))
            if SHARE and kb + 1 < KB and (kb + 1) % SYNC == 0:
                _lds_barrier()
            rocdl.sched_barrier(0)

        # MFMA row R of chain p is channel 2(R & 7) + p of the gate (R < 8) or up block: lane 16q + t holds 8 gate
        # (q < 2) or up (q >= 2) channels 8(q & 1) .. + 7. Lanes l and l ^ 32 trade half of them, so that each holds
        # gate and up of the 4 channels 8(q & 1) + 4(q >> 1) .. + 3 (``chan_off``).
        is_up = lane >= 32
        partner = (lane ^ 32) * 4
        gate_up = [[None] * S for _ in range(2 * G)]
        for j in range(G):
            for s in range(S):
                even, odd = acc[j][0][s], acc[j][1][s]
                vals = [x for v in range(4) for x in (even[v], odd[v])]
                gate, up = [], []
                for k in range(4):
                    keep = is_up.select(vals[4 + k], vals[k])
                    send = is_up.select(vals[k], vals[4 + k])
                    # Raw ds_bpermute with the fixed partner byte address (see the loaders above).
                    got = fx.Int32(
                        rocdl.ds_bpermute(
                            T.i32, _raw(partner), _raw(send.bitcast(fx.Int32))
                        )
                    ).bitcast(fx.Float32)
                    gate.append(is_up.select(got, keep))
                    up.append(is_up.select(keep, got))
                gate_up[j][s], gate_up[G + j][s] = gate, up
        chan_off = ((lane >> 4) & 1) * 8 + (lane >> 5) * 4
        channels, values = silu_mul(
            S, block, gate_up, rbs, chan_off, expert, M, p_w_scale, p_x_scale
        )
        if QUANT:
            store_fp8(
                S, block, channels, values, lane, wave, M, amax_lds, p_out, p_out_scale
            )
        else:
            store_f32(S, block, channels, values, M, p_out)

    @flyc.kernel(
        known_block_size=[64 * W, 1, 1],
        name=a8w4.swiglu_kernel_name("a8w4_gateup_stream", SWIGLU_LIMIT),
    )
    def a8w4_gateup_stream(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_d: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_experts: fx.Pointer,
        p_num_valid: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_out_scale: fx.Pointer,
        M: fx.Int32,
        rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        group = fx.Int32(gpu.block_id("x"))
        blk = fx.Int32(gpu.block_id("y"))
        lds = _lds_base(_LUT_LDS + ACT_LDS + (BLOCK_M * W * 4 if QUANT else 0))
        lut_lds = lds
        _lut_table(lut_lds, lane)
        act_lds = lds + _LUT_LDS if SHARE else None
        amax_lds = lds + _LUT_LDS + ACT_LDS if QUANT else None
        live, block = _block_head(
            blk, BLOCK_M, M, lane, p_sorted_ids, p_sorted_experts, p_num_valid, rows
        )
        for subs in range_constexpr(1, BLOCK_M // 16 + 1):
            if live == subs:
                body(
                    subs,
                    lut_lds,
                    act_lds,
                    amax_lds,
                    _first(block, subs),
                    lane,
                    wave,
                    group,
                    M,
                    p_x,
                    p_w,
                    p_d,
                    p_out,
                    p_w_scale,
                    p_x_scale,
                    p_out_scale,
                )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_d: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_experts: fx.Pointer,
        p_num_valid: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_out_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        rows = task_num * BLOCK_M
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        a8w4_gateup_stream(
            p_x,
            p_w,
            p_d,
            p_out,
            p_sorted_ids,
            p_sorted_experts,
            p_num_valid,
            p_w_scale,
            p_x_scale,
            p_out_scale,
            M,
            rows,
        ).launch(
            grid=(I // (16 * G * W), task_num, 1), block=(64 * W, 1, 1), stream=stream
        )

    return launch


def compile_route_sum(topk, cols):
    """Launcher ``(loc_ids, routes, out, tokens, rows, stream)``: BF16 ``out[t] = sum_k routes[loc_ids[t, k]]``.

    ``loc_ids`` int32 [tokens, topk] (negative: no route), ``routes`` BF16 [rows, cols]. Same arithmetic as
    ``sorted_sum``: f32 sum in slot order (missing routes add +0), round to nearest even. One wave per
    1 KB of an output row, so all ``topk`` 16-byte loads of a lane are in flight together
    (``cols % 512 == 0``).
    """
    assert cols % 512 == 0, f"cols={cols} must be a multiple of 512"
    chunks = cols // 512

    def body(p_loc, p_routes, p_out, tokens, rows):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        item = _uniform(fx.Int32(gpu.block_id("x")) * 4 + (tid >> 6))
        token = item // chunks
        live = token < tokens
        column = (item - token * chunks) * 1024 + lane * 16
        loc_rsrc = _rsrc(_address(p_loc), tokens * (topk * 4))
        routes_rsrc = _rsrc(_address(p_routes), rows * (cols * 2))
        out_rsrc = _rsrc(_address(p_out), tokens * (cols * 2))
        mine = fx.Int32(
            _load(
                loc_rsrc,
                (live & (lane < topk)).select(
                    token * (topk * 4) + lane * 4, fx.Int32(_DROP)
                ),
                0,
                1,
            )[0]
        )
        words = []
        for k in range(topk):
            # Raw readlane (no typed readlane in FlyDSL 0.5.3, see _block_head): route k's sorted row, uniform.
            loc = fx.Int32(
                rocdl.readlane(
                    fx.Int32.ir_type, mine.ir_value(), fx.Int32(k).ir_value()
                )
            )
            words.append(
                _load(
                    routes_rsrc,
                    (live & (loc >= 0)).select(
                        loc * (cols * 2) + column, fx.Int32(_DROP)
                    ),
                    0,
                    4,
                )
            )
        total = []
        for k, row in enumerate(words):
            values = []
            for word in row:
                values.append((word << 16).bitcast(fx.Float32))
                values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
            total = values if k == 0 else [a + b for a, b in zip(total, values)]
        out = [_bf16_pair(total[2 * i], total[2 * i + 1]) for i in range(4)]
        _store(out_rsrc, out, live.select(token * (cols * 2) + column, fx.Int32(_DROP)))

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a8w4_route_sum(
        p_loc: fx.Pointer,
        p_routes: fx.Pointer,
        p_out: fx.Pointer,
        tokens: fx.Int32,
        rows: fx.Int32,
    ):
        body(p_loc, p_routes, p_out, tokens, rows)

    @flyc.jit
    def launch(
        p_loc: fx.Pointer,
        p_routes: fx.Pointer,
        p_out: fx.Pointer,
        tokens: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_route_sum(p_loc, p_routes, p_out, tokens, rows).launch(
            grid=((tokens * chunks + 3) // 4, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


# Routes placed by each sorting workgroup of ``compile_stream_prep`` (one per thread) and routes histogrammed per
# round trip (four 16-byte loads of ids per thread).
_PREP_RANGE = 256
_PREP_ROUND = 4096


def stream_prep_sizes(M, TOPK, E, BM):
    """``(sorting workgroups, sorted-row capacity, block capacity)`` of ``compile_stream_prep`` for M tokens: an
    expert with c routes takes ceil(c / BM) <= (c + BM - 1) / BM blocks, over at most min(E, routes) experts.
    """
    routes = M * TOPK
    blocks = (routes + min(E, routes) * (BM - 1)) // BM
    return -(-routes // _PREP_RANGE), blocks * BM, blocks


def _lds_word(address):
    # Raw llvm.load / llvm.store / atomicrmw at LDS byte addresses (see _lds).
    return fx.Int32(llvm.load(fx.Int32.ir_type, _lds(address), alignment=4))


def _lds_put(address, value):
    llvm.store(fx.Int32(value).ir_value(), _lds(address), alignment=4)


def _lds_inc(address):
    """LDS ``atomicAdd(address, 1)``, returning the previous value."""
    # Raw llvm.AtomicRMWOp at an LDS byte address (ds_add_rtn_u32; the typed fx.atomic_add takes an fx.Pointer,
    # see _lds).
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            _lds(address),
            fx.Int32(1).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def compile_stream_prep(*, H, TOPK, E, BM, ZERO=False, QW=4, SHARED=False):
    """Build the streaming path's prep launcher: the moe_sorting of ``BM``-row blocks and the FP8 input rows (with
    ``ZERO`` also a zeroed output) in one launch.

    Launch arguments: topk_ids int32 [M, TOPK], topk_weight f32 [M, TOPK], x BF16 [M, H] (``H % 512 == 0``),
    sorted_ids int32 [rows], sorted_weights f32 [rows], sorted_expert_ids int32 [rows / BM], num_valid int32 [2],
    loc_ids int32 [M, TOPK], xq FP8 [M, H], xs f32 [M], out BF16 [M, H], then M, the sorting workgroups and rows
    of ``stream_prep_sizes``, stream.

    Sorting (workgroups ``[0, n)``), in aiter moe_sorting's format: sorted_ids ``token | slot << 24`` grouped by
    expert in ascending order, each expert padded to whole blocks with rows of token M and weight 0;
    sorted_expert_ids per block; num_valid ``[padded rows, M]``; loc_ids the sorted row of each route (-1 for an
    expert id outside ``[0, E)``). Every sorting workgroup histograms all routes in LDS, and separately those
    before its own ``_PREP_RANGE`` routes, so it knows every expert's offset and its own base without talking to
    the others; it places its routes with LDS atomics (the order within an expert is not deterministic) and
    writes the padding rows and block experts of experts ``e % n == workgroup``. Every route of a token is kept,
    repeated expert ids included. Quantization (the other workgroups), the arithmetic of ``quant_a8w4``: with
    ``QW == 4`` one row per workgroup, the four waves splitting its 512-column chunks and exchanging the amax
    through LDS (``_fp8_bytes``); with ``QW == 1`` one row per wave (IEEE division per element; fewer workgroups
    for many rows). With ``ZERO`` the waves also zero their rows of out. With ``SHARED`` the last of the TOPK columns
    is a fused shared expert's (``moe_a8w4._SharedFold``): it is never sorted, whatever its id, and its loc_ids is -1.
    """
    assert H % 512 == 0, f"H={H} must be a multiple of 512"
    assert BM & (BM - 1) == 0, f"BM={BM} must be a power of two"
    assert QW in (1, 4)
    chunks = H // 512
    LOG_BM = BM.bit_length() - 1
    EPT = -(-E // 256)  # consecutive experts per thread in the scan
    ES = 256 * EPT
    # LDS words: counts, counts before this workgroup's range and cursors, each followed by one dummy slot per
    # lane (routes that are masked off count there), then the four wave totals of the scan.
    CNT, BEF, CUR = 0, ES + 64, 2 * (ES + 64)
    WTOT = 3 * (ES + 64)
    ZERO_ROUNDS = -(-2 * (ES + 64) // 256)
    LDS_WORDS = max(WTOT + 4, 256 * ZERO_ROUNDS)

    def histogram(lds, base, tid, lane, R, start, ids_rsrc):
        first = [base + (tid + 256 * u) * 4 for u in range(_PREP_ROUND // 1024)]
        words = [_load(ids_rsrc, i * 4, 0, 4) for i in first]
        for i0, w4 in zip(first, words):
            for v in range(4):
                i = i0 + v
                e = fx.Int32(w4[v])
                ok = (i < R) & (e >= 0) & (e < E)
                if const_expr(SHARED):
                    ok = ok & (i % TOPK != TOPK - 1)
                _lds_inc(lds + ok.select(CNT + e, CNT + ES + lane) * 4)
                _lds_inc(lds + (ok & (i < start)).select(BEF + e, BEF + ES + lane) * 4)

    def scan(lds, tid, lane, wave):
        """Per expert of this thread: (routes, padded rows, first sorted row); and all padded rows."""
        counts = [_lds_word(lds + (CNT + tid * EPT + k) * 4) for k in range(EPT)]
        padded = [((c + (BM - 1)) >> LOG_BM) << LOG_BM for c in counts]
        local = padded[0]
        for p in padded[1:]:
            local = local + p
        incl = local
        for d in (1, 2, 4, 8, 16, 32):
            # Raw ds_bpermute from lane - d (byte address 4 * lane; see the Gate/Up loaders): inclusive wave scan.
            other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw((lane - d) * 4), _raw(incl)))
            incl = incl + (lane >= d).select(other, fx.Int32(0))
        _lds_put(lds + (lane == 63).select(WTOT + wave, CNT + ES + lane) * 4, incl)
        gpu.barrier()
        totals = [_lds_word(lds + (WTOT + w) * 4) for w in range(4)]
        offset = incl - local
        for w in range(3):
            offset = offset + (wave > w).select(totals[w], fx.Int32(0))
        offsets = []
        for k in range(EPT):
            offsets.append(offset)
            offset = offset + padded[k]
            e = tid * EPT + k
            _lds_put(lds + (CUR + e) * 4, offsets[k] + _lds_word(lds + (BEF + e) * 4))
        return counts, padded, offsets, totals[0] + totals[1] + totals[2] + totals[3]

    def scatter(lds, tid, lane, R, start, ids, weights, sid_rsrc, sw_rsrc, loc_rsrc):
        i = start + tid
        e = fx.Int32(ids)
        ok = (i < R) & (e >= 0) & (e < E)
        if const_expr(SHARED):
            ok = ok & (i % TOPK != TOPK - 1)
        pos = _lds_inc(lds + ok.select(CUR + e, CUR + ES + lane) * 4)
        token = i // TOPK
        row = ok.select(pos * 4, fx.Int32(_DROP))
        _store(sid_rsrc, [fx.Uint32(token | ((i - token * TOPK) << 24))], row)
        _store(sw_rsrc, [weights], row)
        _store(
            loc_rsrc,
            [fx.Uint32(ok.select(pos, fx.Int32(-1)))],
            (i < R).select(i * 4, fx.Int32(_DROP)),
        )

    def quant(lds, q, lane, wave, M, p_x, p_xq, p_xs, p_out):
        """Workgroup ``q`` of the quantization: row q split over the four waves (``QW == 4``, amax through LDS),
        or rows 4q .. 4q+3, one per wave."""
        in_rsrc = _rsrc(_address(p_x), M * (H * 2))
        if QW == 4:
            row, first = q, (wave == 0) & (lane == 0)
            mine = [
                (wave + 4 * j, wave + 4 * j < chunks) for j in range(-(-chunks // 4))
            ]
        else:
            row = q * 4 + wave
            first = (row < M) & (lane == 0)
            mine = [(c, row < M) for c in range(chunks)]
        values = []
        for c, ok in mine:
            for word in _load(
                in_rsrc,
                ok.select(row * (H * 2) + (c * 64 + lane) * 16, fx.Int32(_DROP)),
                0,
                4,
            ):
                values.append((word << 16).bitcast(fx.Float32))
                values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
        peak = _lanes_max(_abs_bits_max(values), (32, 16, 8, 4, 2, 1))
        if QW == 4:
            _lds_put(lds + wave * 4, peak)
            gpu.barrier()
            peaks = [fx.Uint32(_lds_word(lds + w * 4)) for w in range(4)]
            peak = _umax(_umax(peaks[0], peaks[1]), _umax(peaks[2], peaks[3]))
        scale = _row_scale(peak.bitcast(fx.Float32))
        q8 = (
            _fp8_bytes(values, scale)
            if QW == 4
            else [_fp8_byte(v / scale) for v in values]
        )
        out_rsrc = _rsrc(_address(p_xq), M * H)
        zero_rsrc = _rsrc(_address(p_out), M * (H * 2))
        for j, (c, ok) in enumerate(mine):
            words = [
                _pack_bytes(q8[j * 8 + h * 4 : j * 8 + h * 4 + 4]) for h in range(2)
            ]
            _store(
                out_rsrc,
                words,
                ok.select(row * H + (c * 64 + lane) * 8, fx.Int32(_DROP)),
            )
            if ZERO:
                _store(
                    zero_rsrc,
                    [fx.Uint32(0)] * 4,
                    ok.select(row * (H * 2) + (c * 64 + lane) * 16, fx.Int32(_DROP)),
                )
        _store(
            _rsrc(_address(p_xs), M * 4),
            [scale.bitcast(fx.Uint32)],
            first.select(row * 4, fx.Int32(_DROP)),
        )

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a8w4_stream_prep(
        p_ids: fx.Pointer,
        p_w: fx.Pointer,
        p_x: fx.Pointer,
        p_sid: fx.Pointer,
        p_sw: fx.Pointer,
        p_se: fx.Pointer,
        p_nv: fx.Pointer,
        p_loc: fx.Pointer,
        p_xq: fx.Pointer,
        p_xs: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        n_sort: fx.Int32,
        rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        wg = fx.Int32(gpu.block_id("x"))
        lds = _lds_base(LDS_WORDS * 4)
        if wg < n_sort:
            R = M * TOPK
            start = wg * _PREP_RANGE
            ids_rsrc = _rsrc(_address(p_ids), (R * 4 + 15) & -16)
            w_rsrc = _rsrc(_address(p_w), (R * 4 + 15) & -16)
            sid_rsrc = _rsrc(_address(p_sid), rows * 4)
            sw_rsrc = _rsrc(_address(p_sw), rows * 4)
            se_rsrc = _rsrc(_address(p_se), (rows >> LOG_BM) * 4)
            own_ids = _load(ids_rsrc, (start + tid) * 4, 0, 1)[0]
            own_weights = _load(w_rsrc, (start + tid) * 4, 0, 1)[0]
            for j in range_constexpr(ZERO_ROUNDS):
                _lds_put(lds + (tid + 256 * j) * 4, 0)
            gpu.barrier()
            for base in range(fx.Int32(0), R, fx.Int32(_PREP_ROUND)):
                histogram(lds, base, tid, lane, R, start, ids_rsrc)
            gpu.barrier()
            counts, padded, offsets, total = scan(lds, tid, lane, wave)
            gpu.barrier()
            scatter(
                lds,
                tid,
                lane,
                R,
                start,
                own_ids,
                own_weights,
                sid_rsrc,
                sw_rsrc,
                _rsrc(_address(p_loc), R * 4),
            )
            for k in range_constexpr(EPT):
                expert = tid * EPT + k
                owned = (expert < E) & (expert - (expert // n_sort) * n_sort == wg)
                b0 = offsets[k] >> LOG_BM
                b1 = b0 + owned.select(padded[k] >> LOG_BM, fx.Int32(0))
                for b in range(b0, b1):
                    _store(se_rsrc, [fx.Uint32(expert)], b * 4)
                p0 = offsets[k] + counts[k]
                p1 = owned.select(offsets[k] + padded[k], p0)
                for p in range(p0, p1):
                    _store(sid_rsrc, [fx.Uint32(M)], p * 4)
                    _store(sw_rsrc, [fx.Uint32(0)], p * 4)
            _store(
                _rsrc(_address(p_nv), 8),
                [fx.Uint32(total), fx.Uint32(M)],
                ((wg == 0) & (tid == 0)).select(fx.Int32(0), fx.Int32(_DROP)),
            )
        else:
            quant(lds, wg - n_sort, lane, wave, M, p_x, p_xq, p_xs, p_out)

    @flyc.jit
    def launch(
        p_ids: fx.Pointer,
        p_w: fx.Pointer,
        p_x: fx.Pointer,
        p_sid: fx.Pointer,
        p_sw: fx.Pointer,
        p_se: fx.Pointer,
        p_nv: fx.Pointer,
        p_loc: fx.Pointer,
        p_xq: fx.Pointer,
        p_xs: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        n_sort: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_stream_prep(
            p_ids,
            p_w,
            p_x,
            p_sid,
            p_sw,
            p_se,
            p_nv,
            p_loc,
            p_xq,
            p_xs,
            p_out,
            M,
            n_sort,
            rows,
        ).launch(
            grid=(n_sort + (M if QW == 4 else (M + 3) // 4), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


def _down_scratch(K, BLOCK_M, W):
    """Whether some loader lane of the shared Down falls past the block's rows (S * K bytes over W * 64 lanes of
    16 bytes) for some live sub-block count S and needs the scratch slot."""
    return any((s * K) % (W * 64) for s in range(1, BLOCK_M // 16 + 1))


def down_stream_lds(K, BLOCK_M, W=1, SHARE=True):
    """LDS bytes of ``compile_a8w4_down_stream``: the FP8 LUT table (``_LUT_LDS``) and 1 KiB of channel factors per
    wave, plus with ``SHARE`` and ``W > 1`` the block's activation rows (stride ``K + 16``) and, when some loader lane
    falls past the rows, one 16-byte slot per lane."""
    if not (SHARE and W > 1):
        return _LUT_LDS + W * 1024
    return (
        _LUT_LDS
        + W * 1024
        + BLOCK_M * (K + 16)
        + (W * 64 * 16 if _down_scratch(K, BLOCK_M, W) else 0)
    )


def compile_a8w4_down_stream(
    *,
    N,
    K,
    TOPK,
    BLOCK_M,
    R=16,
    W=1,
    PREFETCH=4,
    SHARE=True,
    E=None,
    ATOMIC=False,
    LDS_ACTS=False,
    W_AUX=0,
):
    """Build the streaming A8W4 Down launcher.

    ``N`` = hidden (``N % (16 * R * W) == 0``, ``R`` even and ``<= 16``), ``K`` = padded I (``K % 64 == 0``),
    ``BLOCK_M`` the sorting block. Launch arguments: x FP8 [M, TOPK, K], w and w_delta (``common_a8w4``
    layout, uint8), out BF16 [task_num * BLOCK_M, N] by sorted row (padding rows untouched), sorted_ids,
    sorted_weights, sorted_expert_ids, num_valid_ids, w_scale f32 [E, N], x_scale f32 [M, TOPK], then
    M, task_num, stream. The wave's ``R * 16`` channel factors are staged in LDS with one 16-byte load
    per lane. The wave takes its 16-row blocks in pairs: one 16-byte load per lane and 64-K step reads two
    adjacent rows of either block (lanes ``16b .. 16b+7`` the first, ``16b+8 .. 16b+15`` the second), MFMA chain p
    takes row p of every lane's pair, and each sub-block accumulates the pair in four chains (row p x the two MMAs
    of every 64-K step, the latter summed at the end as one block's two chains were). With
    ``SHARE`` and ``W > 1`` the waves of a workgroup load the block's activation rows once into LDS; with
    ``LDS_ACTS`` as well they stay there and each 64-K step reads them again (fewer VGPRs for many rows / long K).
    Without ``ATOMIC`` lane ``16q + t`` then holds channels ``8q .. 8q+7`` of the pair for sorted row t, one
    16-byte BF16 store. With ``ATOMIC`` out is BF16 [M, N], zeroed by the caller, and each route is added into its
    token's row with agent-scope BF16x2 atomics: the MFMA takes the rows as A, so lane ``16b + j`` holds
    channels ``2j`` and ``2j + 1`` of the pair (chains 0 and 1) of rows ``4b .. 4b+3``, and one atomic instruction
    covers 64 contiguous bytes of four rows.
    """
    assert K % 64 == 0, f"Down K={K} must be a multiple of 64"
    assert N % (16 * R * W) == 0, f"Down N={N} must be a multiple of {16 * R * W}"
    assert BLOCK_M in (16, 32, 64), f"BLOCK_M={BLOCK_M} must be 16, 32 or 64"
    assert (
        R <= 16
    ), "one 16-byte load per lane stages at most 16 blocks of channel factors"
    assert R % 2 == 0, "the 16-byte weight loads cover pairs of 16-row blocks"
    KB, KQ = K // 64, a8w4.delta_dwords(K)
    P = min(PREFETCH, R)
    # Block pairs of the wave and pairs in flight (P blocks).
    PAIRS, PP = R // 2, max(P // 2, 1)
    W_BYTES, D_BYTES = a8w4.weight_bytes(N, K), a8w4.delta_bytes(N, K)
    REGION = 1024
    SHARE = SHARE and W > 1
    LDS_ACTS = LDS_ACTS and SHARE
    KP = K + 16  # LDS row stride: 16-byte reads of 16 rows hit distinct banks
    SUBS = BLOCK_M // 16
    # Activation rows, then (if some loader lane can fall past the rows) one 16-byte scratch slot per lane.
    ACT_LDS = (
        SUBS * 16 * KP + (W * 64 * 16 if _down_scratch(K, BLOCK_M, W) else 0)
        if SHARE
        else 0
    )
    LDS = _LUT_LDS + W * REGION + ACT_LDS
    assert LDS == down_stream_lds(K, BLOCK_M, W, SHARE)
    assert LDS <= a8w4.LDS_BYTES, f"Down needs {LDS} B of LDS (K={K})"

    def body(
        S,
        lut_lds,
        scale_lds,
        act_lds,
        block,
        lane,
        wave,
        group,
        M,
        rows,
        p_x,
        p_w,
        p_d,
        p_out,
        p_sorted_weights,
        p_w_scale,
        p_x_scale,
    ):
        expert = block.expert
        w_rsrc = _rsrc(_address(p_w) + fx.Int64(expert) * W_BYTES, W_BYTES)
        d_rsrc = _rsrc(_address(p_d) + fx.Int64(expert) * D_BYTES, D_BYTES)
        ws_rsrc = _rsrc(_address(p_w_scale) + fx.Int64(expert) * (N * 4), N * 4)
        x_rsrc = _rsrc(_address(p_x), M * (TOPK * K))
        xs_rsrc = _rsrc(_address(p_x_scale), M * (TOPK * 4))
        sw_rsrc = _rsrc(_address(p_sorted_weights), rows * 4)
        out_rsrc = _rsrc(_address(p_out), rows * (N * 2))
        rb0 = (group * W + wave) * R
        # Block pairs: pair u of the wave is its 16-row blocks 2u and 2u + 1. Lane 16b + i loads layout lanes
        # 16b + 2(i & 7) and + 1 (rows 2(i & 7) and + 1, K group b) of block 2u (i < 8) or 2u + 1 (i >= 8): one 16-byte
        # load per lane and 64-K block covers both blocks (each quad of lanes reads 64 contiguous bytes), and the rows'
        # two delta dwords are adjacent (8-byte loads). MFMA row R of chain p (row p of every lane's pair) is channel
        # 2(R & 7) + p of block 2u + (R >> 3), so the C layout puts channels 8q .. 8q + 7 of the pair in lane 16q + t.
        second = (lane >> 3) & 1
        w_voff = (
            (lane >> 4) * 128 + (lane & 7) * 16 + second * (KB * a8w4.W_BLOCK_BYTES)
        )
        d_voff = (lane >> 5) * 64 + (lane & 7) * 8 + second * (KQ * a8w4.D_BLOCK_BYTES)
        # Lanes 16b + i and 16(b ^ 1) + i need the same delta dwords, so one load serves two K groups:
        # lanes with odd b fetch the odd group of each pair and the halves swap them across 16 lanes.
        odd = (lane >> 4) & 1
        d_voff_pair = d_voff + odd * a8w4.D_BLOCK_BYTES

        def load_w(u, kb):
            return _load(
                w_rsrc, w_voff, ((rb0 + 2 * u) * KB + kb) * a8w4.W_BLOCK_BYTES, 4, W_AUX
            )

        def load_d(u):
            return [
                _load(
                    d_rsrc,
                    d_voff_pair if q + 1 < KQ else d_voff,
                    ((rb0 + 2 * u) * KQ + q) * a8w4.D_BLOCK_BYTES,
                    2,
                    W_AUX,
                )
                for q in range(0, KQ, 2)
            ]

        def resolve(loads):
            """The delta dwords of the lane's two rows per 256-K group."""
            groups = []
            for i, words in enumerate(loads):
                if 2 * i + 1 < KQ:
                    other = [
                        fx.Float32(word.bitcast(fx.Float32))
                        .shuffle_xor(16, 64)
                        .bitcast(fx.Uint32)
                        for word in words
                    ]
                    groups.append(
                        [(odd == 0).select(w, o) for w, o in zip(words, other)]
                    )
                    groups.append(
                        [(odd == 0).select(o, w) for w, o in zip(words, other)]
                    )
                else:
                    groups.append(list(words))
            return groups

        route = [t * TOPK + s for t, s in zip(block.token, block.slot)]
        region = scale_lds + wave * REGION
        if SHARE:
            # The block's S * 16 rows of K bytes as 16-byte chunks spread over the workgroup's lanes; loads
            # are issued before the weights so waiting for them never waits for weights.
            chunks = K // 16
            total = S * 16 * chunks
            pending = []
            for j in range(-(-total // (W * 64))):
                i = wave * 64 + lane + j * (W * 64)
                sub = i // (16 * chunks)
                row = (i // chunks) & 15
                chunk = i - (i // chunks) * chunks
                src = fx.Int32(0)
                for s in range(S):
                    # Raw ds_bpermute (byte address = 4 * source lane, see the Gate/Up loaders): the loaded row's route.
                    other = fx.Int32(
                        rocdl.ds_bpermute(T.i32, _raw(row * 4), _raw(route[s]))
                    )
                    src = (sub == s).select(other, src)
                live = i < total
                words = _load(
                    x_rsrc, live.select(src * K + chunk * 16, fx.Int32(_DROP)), 0, 4
                )
                pending.append(
                    (
                        live.select(
                            act_lds + sub * (16 * KP) + row * KP + chunk * 16,
                            act_lds + SUBS * 16 * KP + (wave * 64 + lane) * 16,
                        ),
                        words,
                    )
                )
        factors = _load(
            ws_rsrc,
            (lane * 16 < R * 64).select(rb0 * 64 + lane * 16, fx.Int32(_DROP)),
            0,
            4,
        )
        # The first PP pairs in flight; pair u + PP is issued during pair u, one 64-K block per 64-K block.
        weights = {u: [load_w(u, kb) for kb in range(KB)] for u in range(PP)}
        delta_loads = {u: load_d(u) for u in range(PP)}
        # Raw llvm.store / llvm.load of the channel factors at LDS byte addresses (see _lds).
        llvm.store(
            Vec.from_elements(factors, fx.Uint32).bitcast(fx.Int32).ir_value(),
            _lds(region + lane * 16),
            alignment=16,
        )
        acts = []
        if SHARE:
            for address, words in pending:
                _lds_store(address, words)
            _lds_barrier()
            act_rd = act_lds + (lane & 15) * KP + (lane >> 4) * 16

            def read_act(s, kb):
                x = _lds_load4(act_rd + s * (16 * KP) + kb * 64)
                return _pack64(x[0], x[1]), _pack64(x[2], x[3])

            if not LDS_ACTS:
                acts = [[read_act(s, kb) for kb in range(KB)] for s in range(S)]
        else:
            for r in route:
                words = [
                    _load(x_rsrc, r * K + (lane >> 4) * 16, kb * 64, 4)
                    for kb in range(KB)
                ]
                acts.append([(_pack64(x[0], x[1]), _pack64(x[2], x[3])) for x in words])
        row_scale = []
        for r, p in zip(route, block.position):
            a_scale = _f32(_load(xs_rsrc, r * 4, 0, 1)[0])
            row_scale.append(a_scale * _f32(_load(sw_rsrc, p * 4, 0, 1)[0]))
        if ATOMIC:
            # Rows 4b .. 4b+3 of lane 16b + j (the MFMA's C layout with the rows as A). Raw ds_bpermute (byte
            # address = 4 * source lane, see the Gate/Up loaders).
            src = [((lane >> 4) * 4 + v) * 4 for v in range(4)]
            row_token = [
                [
                    fx.Int32(
                        rocdl.ds_bpermute(T.i32, _raw(src[v]), _raw(block.token[s]))
                    )
                    for v in range(4)
                ]
                for s in range(S)
            ]
            row_factor = [
                [
                    fx.Uint32(
                        rocdl.ds_bpermute(
                            T.i32, _raw(src[v]), _raw(row_scale[s].bitcast(fx.Uint32))
                        )
                    ).bitcast(fx.Float32)
                    for v in range(4)
                ]
                for s in range(S)
            ]
            out_base = _address(p_out)
        deltas = {0: resolve(delta_loads.pop(0))}
        for u in range(PAIRS):
            ahead = u + PP < PAIRS
            if ahead:
                delta_loads[u + PP] = load_d(u + PP)
                weights[u + PP] = []
            if u + 1 < PAIRS:
                # One pair ahead: the swap's latency stays hidden.
                deltas[u + 1] = resolve(delta_loads.pop(u + 1))
            w = weights.pop(u)
            d = deltas.pop(u)
            # Chains [p][c]: row p of the lanes' pairs, MMA c of every 64-K block (summed at the end).
            acc = [[[_zeros() for _ in range(S)] for _ in range(2)] for _ in range(2)]
            for kb in range(KB):
                if ahead:
                    weights[u + PP].append(load_w(u + PP, kb))
                b = (
                    [read_act(s, kb) for s in range(S)]
                    if LDS_ACTS
                    else [acts[s][kb] for s in range(S)]
                )
                for p in range(2):
                    lut = _lut_at(lut_lds, d[kb // 4][p], kb % 4)
                    ops = _decode_row(w[kb][2 * p : 2 * p + 2], lut)
                    for c in range(2):
                        for s in range(S):
                            if ATOMIC:
                                acc[p][c][s] = _mfma(b[s][c], ops[c], acc[p][c][s])
                            else:
                                acc[p][c][s] = _mfma(ops[c], b[s][c], acc[p][c][s])
            if ATOMIC:
                # Lane 16b + j holds channels 2j and 2j + 1 of the pair (chains 0 and 1) for rows 4b .. 4b + 3: one
                # BF16x2 atomic per row adds a channel pair, 64 contiguous bytes of four rows per instruction.
                ws = Vec(
                    llvm.load(
                        ir.VectorType.get([2], fx.Float32.ir_type),
                        _lds(region + (u * 32 + (lane & 15) * 2) * 4),
                        alignment=8,
                    )
                )
                channel = (rb0 + 2 * u) * 16 + (lane & 15) * 2
                for s in range(S):
                    for v in range(4):
                        pair = [
                            ((acc[p][0][s][v] + acc[p][1][s][v]) * ws[p])
                            * row_factor[s][v]
                            for p in range(2)
                        ]
                        _atomic_add_bf16x2_if(
                            row_token[s][v] < M,
                            out_base,
                            (row_token[s][v] * N + channel) * 2,
                            _bf16_pair(pair[0], pair[1]),
                        )
                rocdl.sched_barrier(0)
                continue
            # Lane 16q + t holds channels 8q .. 8q + 7 of the pair (chain p the ones of parity p) for row t: one
            # 16-byte BF16 store per sub-block.
            ws = [
                Vec(
                    llvm.load(
                        ir.VectorType.get([4], fx.Float32.ir_type),
                        _lds(region + (u * 32 + (lane >> 4) * 8 + 4 * h) * 4),
                        alignment=16,
                    )
                )
                for h in range(2)
            ]
            channel = (rb0 + 2 * u) * 16 + (lane >> 4) * 8
            for s in range(S):
                values = []
                for v in range(4):
                    for p in range(2):
                        total = acc[p][0][s][v] + acc[p][1][s][v]
                        values.append(
                            (total * ws[v // 2][2 * (v % 2) + p]) * row_scale[s]
                        )
                offset = (block.position[s] * N + channel) * 2
                offset = (block.token[s] < M).select(offset, fx.Int32(_DROP))
                _store(
                    out_rsrc,
                    [_bf16_pair(values[2 * v], values[2 * v + 1]) for v in range(4)],
                    offset,
                )
            rocdl.sched_barrier(0)

    @flyc.kernel(known_block_size=[64 * W, 1, 1])
    def a8w4_down_stream(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_d: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_experts: fx.Pointer,
        p_num_valid: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_x_scale: fx.Pointer,
        M: fx.Int32,
        rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        group = fx.Int32(gpu.block_id("x"))
        blk = fx.Int32(gpu.block_id("y"))
        lut_lds = _lds_base(LDS)
        scale_lds = lut_lds + _LUT_LDS
        _lut_table(lut_lds, lane)
        act_lds = scale_lds + W * REGION
        live, block = _block_head(
            blk, BLOCK_M, M, lane, p_sorted_ids, p_sorted_experts, p_num_valid, rows
        )
        for subs in range_constexpr(1, BLOCK_M // 16 + 1):
            if live == subs:
                body(
                    subs,
                    lut_lds,
                    scale_lds,
                    act_lds,
                    _first(block, subs),
                    lane,
                    wave,
                    group,
                    M,
                    rows,
                    p_x,
                    p_w,
                    p_d,
                    p_out,
                    p_sorted_weights,
                    p_w_scale,
                    p_x_scale,
                )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_w: fx.Pointer,
        p_d: fx.Pointer,
        p_out: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_experts: fx.Pointer,
        p_num_valid: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_x_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        rows = task_num * BLOCK_M
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        a8w4_down_stream(
            p_x,
            p_w,
            p_d,
            p_out,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_experts,
            p_num_valid,
            p_w_scale,
            p_x_scale,
            M,
            rows,
        ).launch(
            grid=(N // (16 * R * W), task_num, 1), block=(64 * W, 1, 1), stream=stream
        )

    return launch
