# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 multi-query/multi-head QSA sparse attention with exact membership.

A task is 128 (query, head) rows of a union tile against its BN = 64-key tiles: four waves
of 32 rows (one per SIMD) with 32x32x8 BF16 MFMAs, K and V staged in LDS through registers
(see the 32-row section below). A CTA runs its tasks back to back and loads the next
task's first operands under the current one's epilogue.

Split plans (short prefills): K3 gives a long task one order entry per key-range part, two
(sparse_prefill_prepare.SPLIT_PARTS), part p from key tile ceil(p * tiles / 2) on
(_split_entry_layout). Each part stores its O^T fragments, m and l (system-scope stores, any
XCD may merge) and counts on the task's slot; the part counting last merges both records in
part order (O = O_0 2^(m_0 - m) + O_1 2^(m_1 - m), l likewise) and writes the rows like an
unsplit task.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import rocdl

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    SC_SYSTEM,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fma,
    _keep_vgpr,
    _lazy,
    _load_lds,
    _maximum,
    _min,
    _pin_i32,
    _stage_end,
    _uniform,
    _write_address,
)
from .sparse_prefill_prepare import PARTIAL_FLOATS, SPLIT_PARTS, SPLIT_SLOTS

__all__ = ["launcher", "run"]

BM, BN, D, LDS_BYTES = 128, 64, 256, 65536
# A split part's record (sparse_prefill_prepare.PARTIAL_FLOATS): O^T fragments (16 bytes per
# lane and register quad, wave-major), then m and l of the task's 128 rows.
PARTIAL_BYTES = PARTIAL_FLOATS * 4
PARTIAL_ML = BM * D * 4
# A slot table's pool of at most this many bytes is read through one descriptor (32-bit
# offsets: a block past the list starts at the span, its tokens stay below 2^32).
POOL_LIMIT = 2**32 - 2**20
# A split task's merge keeps MERGE_AHEAD_WORDS record words in flight ahead of its folds.
MERGE_AHEAD_WORDS = 64


def _split_entry_layout(split):
    """(part mask, slot shift, task shift) of a split plan's task-order entries (K3 writes
    them): bit 0 marks a task split into `split` key ranges, then its part, its slot (of
    SPLIT_SLOTS) and the task; two parts: task << 8 | slot << 2 | part << 1 | 1."""
    assert split == SPLIT_PARTS
    part_bits = (split - 1).bit_length()
    slot_shift = 1 + part_bits
    return (1 << part_bits) - 1, slot_shift, slot_shift + (SPLIT_SLOTS - 1).bit_length()


def _split_i64(value):
    """(low, high) Int32 words of an Int64."""
    return fx.Int32(value & 0xFFFFFFFF), fx.Int32(value >> 32)


def _join_i64(low, high):
    return (fx.Int64(high) << 32) | (fx.Int64(low) & 0xFFFFFFFF)


# ---- 32 rows per wave: four waves (one per SIMD) of 32x32x8 BF16 MFMAs ----
#
# A task's 128 rows run on four waves of 32 rows: S^T = K Q^T and O^T += V^T P^T per 64-key
# tile, so each K or V operand read from LDS feeds 32 rows (half the LDS reads per MFMA of
# 16-row waves with 16x16x16 MFMAs). Lane l: n = l % 32, h = l // 32. 32x32x8 operands: A row
# m = n with k = 4h + 0..3, B column n with k = 4h + 0..3, C column n and rows
# 8 (r // 4) + 4h + r % 4 of register r.
#
# QK: M = keys (blocks kb of 32), N = the wave's rows, K = dims. Step pair p takes dims
# 16p .. 16p + 15: lane half h reads dims 16p + 8h .. + 7 of its key (one 16-byte read, two
# MFMAs), and Q holds the same dims of row n. M position m of block kb is key 32kb +
# swap23(m) (bits 2 and 3 swapped), so S^T register r is key 32kb + (r & 7) + 8h +
# 16 (r >> 3), and the four registers of PV chunk c are keys 16 (c >> 1) + 8h + 4 (c & 1) +
# 0..3: a chunk pair's eight keys are consecutive.
# PV: M = dims (block b: M row m is dim 8m + b), N = rows, K = keys. P^T comes straight from
# S^T's registers (B operand). V is kept transposed in LDS (_static32's vt_wbase and
# vt_rbase, read by _vt_read: per dim, the tile's 64 keys in eight 16-byte octets, XOR-
# swizzled), so one 16-byte read holds the A operands of dim 8n + b for both MFMAs of a chunk
# pair. O^T register r of block b is row n, dim 8 (8 (r // 4) + 4h + r % 4) + b: per r a
# lane holds 8 consecutive dims.
#
# K and V reach LDS through registers: wave w loads keys 16w .. 16w + 15 of a tile and writes
# them one part later. K: two keys per buffer_load_dwordx4 (16 bytes per lane), written by
# ds_write_b128 (16-byte chunks XOR-swizzled by key & 7). V: one key per
# buffer_load_dwordx2 (dims 4l .. 4l + 3 in lane l); v_perm pairs the keys of each of those
# dims into octets, one ds_write_b128 per dim and octet.
#
# Phase t: Part 1 runs QK(t) (K reads KPRE ahead) with P(t - 1) = exp2(S * c - m), its row
# sums and BF16 B operands, the V(t - 1) writes and the K(t + 1) loads under the MFMAs;
# after QK MFMA BAR1 every K read and V write has landed: barrier, then the first V reads.
# Part 2 runs PV(t - 1) (V^T reads a group ahead, used as A operands as read) with the rest
# of P(t - 1), S(t)'s membership masks, row max and lazy max m', the K(t + 1) writes and the
# V(t) loads; after PV MFMA BAR2 every V read and K write has landed: barrier, then the
# first K reads of QK(t + 1). When some row's max moved, O and l rescale after the last PV
# MFMA. The helpers are plain Python run at trace time (not rewritten): instruction
# streams are generators that _part32 steps between MFMAs by deadline and estimated issue
# cycles; a prim.LgkmLedger keeps the partial lgkmcnt waits of their LDS operations.

WAVES32 = 4
THREADS32 = 64 * WAVES32
NEG_INF_BITS = -8388608  # FP32 -inf
# Scheduling constants below: the issue-cycle estimates (SLOT32, C_*) are gfx942 ISA issue
# costs; the MFMA positions (KPRE, BAR*, release/deadline pairs, GAP_*, AFTER_BAR) were tuned
# on MI308X by timing sweeps and checked against ATT traces of the phase loop.
#
# K reads issued KPRE reads (2 KPRE MFMAs) ahead; the barriers after QK MFMA BAR1 and PV
# MFMA BAR2.
KPRE, BAR1, BAR2 = 10, 48, 48
# Issue cycles per MFMA slot; stream (release MFMA, deadline MFMA, estimated slots) per part.
SLOT32 = 30
P1_EXP, P1_VWRITE, P1_KLOAD = (0, 63, 21), (0, BAR1 - 2, 24), (0, 56, 48)
P1_META, P1_VLOAD0 = (0, 12, 2), (0, 60, 48)
P2_SOFT, P2_KWRITE, P2_VLOAD = (2, BAR2 - 8, 21), (6, BAR2 - 2, 16), (0, 62, 48)
SOFT_DUE = 63  # deadline of the softmax steps after the row-max shuffle
# Estimated issue cycles of stream steps; MFMA gaps after VMEM loads and LDS writes.
C_EXP, C_VALU, C_LDS_W, C_LDS_R, C_VMEM = 19, 5, 26, 5, 6
GAP_KLOAD, GAP_VLOAD, GAP_WRITE = 6, 3, 2
AFTER_BAR = 14  # MFMAs to finish the streams after a barrier
PV_PREFETCH = (0, 46, 34)  # the next task's Q staging, ids and mask under PV(last)
QK_ORDER = tuple((p, e, kb) for p in range(16) for e in range(2) for kb in range(2))
PV_GROUPS = ((0, 0), (0, 1), (1, 0), (1, 1))  # (kb, q2): chunk pairs of PV(t - 1)

EXP_CHUNKS = tuple((kb, c) for kb in range(2) for c in range(4))


def _first_pv32(kb, c):
    """The first PV MFMA using P chunk (kb, c)."""
    return 16 * (2 * kb + (c >> 1)) + 8 * (c & 1)


# P chunks (kb, c) in EXP_CHUNKS order: the first P1_CHUNKS under QK, the rest under the
# PV MFMAs before their first use.
P1_CHUNKS, P2_EXP = 5, (0, _first_pv32(*EXP_CHUNKS[5]) - 2, 13)


class _Env32:
    """Trace-time values of one task, shared by the helpers below."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _swap23(n):
    """n with bits 2 and 3 swapped: the key of QK M position n in a 32-key block."""
    return (n & -13) | ((n & 4) << 1) | ((n & 8) >> 1)


def _mfma32(a, b, acc):
    """One 32x32x8 BF16 MFMA: a and b hold 4 BF16 (two words) per lane, acc 16 FP32."""
    frag_a = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_b = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_c = fx.make_rmem_tensor(16, fx.Float32)
    frag_a.store(fx.Vector.from_elements(list(a), fx.Int32).bitcast(fx.BFloat16))
    frag_b.store(fx.Vector.from_elements(list(b), fx.Int32).bitcast(fx.BFloat16))
    frag_c.store(fx.Vector(acc))
    mma = fx.make_mma_atom(rocdl.MFMA(32, 32, 8, fx.BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _pin32():
    prim.sched_barrier(0)


def _step32(gen, count=1):
    """count steps of an instruction stream, each fenced."""
    for _ in range(count):
        next(gen, None)
        prim.sched_barrier(0)


def _drain32(gen):
    for _ in gen:
        prim.sched_barrier(0)


def _chain32(*gens):
    for gen in gens:
        yield from gen


class _Stream32:
    """An instruction stream stepped between the MFMAs of a part from MFMA `release` on and
    finished by MFMA `deadline`. Its generator yields each step's estimated issue cycles, or
    (cycles, gap): the next step waits gap MFMAs (spreads VMEM and LDS writes), or
    (cycles, gap, deadline): the remaining steps get a new deadline."""

    def __init__(self, gen, release, deadline, slots=0):
        self.gen, self.deadline, self.done = gen, deadline, False
        self.ready, self.slots = release, slots

    def laxity(self, i):
        """Spare MFMA slots before the deadline (slots: the estimated slots still needed)."""
        return self.deadline - i - self.slots

    def step(self, i):
        cost = next(self.gen, None)
        prim.sched_barrier(0)
        if cost is None:
            self.done = True
            return 0
        gap = 0
        if isinstance(cost, tuple):
            if len(cost) == 3:
                self.deadline = cost[2]
            cost, gap = cost[:2]
            self.ready = i + gap
        self.slots -= max(gap, cost / SLOT32)
        return cost

    def drain(self, i=None):
        """Every remaining step, or with i the steps due by MFMA i."""
        while not self.done and (i is None or self.deadline <= i):
            self.step(0 if i is None else i)


def _part32(count, issue, streams, bar, after_bar, lgkm):
    """count MFMAs, issue(i) emitting MFMA i with its own operand reads and returning their
    estimated issue cycles. Between MFMAs the ready streams step, least laxity first,
    until the MFMA's SLOT32 cycles are filled (an overrun is carried, idle time is not).
    After MFMA bar every LDS access lands (lgkmcnt 0), the CTA meets, and after_bar()'s
    stream (deadline: AFTER_BAR MFMAs later) is released."""
    streams = list(streams)
    budget = 0
    for i in range(count):
        budget = min(budget, 0) + SLOT32 - issue(i)
        _pin32()
        for stream in streams:
            stream.drain(i)
        while budget > 0:
            ready = [x for x in streams if not x.done and x.ready <= i]
            if not ready:
                break
            budget -= min(ready, key=lambda x: x.laxity(i)).step(i)
        if i == bar:
            for stream in streams:
                stream.drain(bar)
            rocdl.s_waitcnt(lgkmcnt=0)
            lgkm.all_done()
            _stage_end()
            streams.append(_Stream32(after_bar(), i + 1, i + AFTER_BAR, 10))
    for stream in streams:
        stream.drain()


def _k32_bases(n, h):
    """K read addresses of step pairs p & 3 = 0..3 (key swap23(n), 16-byte chunk 2p + h
    XOR-swizzled by n & 7, the key's phase (key & 3) | (key >> 3 & 1) * 4); kb and p >> 2
    go into the immediate."""
    key = _swap23(n)
    return tuple(key * 512 + (((2 * c + h) ^ (n & 7)) << 4) for c in range(4))


def _k32_read(env, kb, p):
    """(words, ledger index) of step pair p's K read of key block kb."""
    words = _load_lds(env.k_lds, env.kbases[p & 3], 16384 * kb + 128 * (p >> 2))
    return words, env.lgkm.issue()


def _vt_read(env, g, b):
    """Keys 32kb + 16q2 + 8h + 0..7 of dim 8n + b (group g = (kb, q2)): words 0-1 are the A
    operand of the chunk pair's first MFMA, words 2-3 of its second; with its ledger
    index."""
    words = _load_lds(env.v_lds, env.vt_rbase[g][b >> 2], 128 * b)
    return words, env.lgkm.issue()


def _keep32(words, kb, r):
    """0 / -1 membership of S^T register r of block kb. The planner's 64 bits per query and
    tile (see _masks in sparse_prefill_prepare) hold keys 8j .. 8j + 7 at bits 16j .. 16j + 7
    and keys 32 + 8j .. at bits 16j + 8 .. 16j + 15; register r's key 32kb + 8h + (r & 7) +
    16 (r >> 3) is bit 8kb + (r & 7) of word r >> 3 shifted right by 16h."""
    position = 8 * kb + (r & 7)
    return (fx.Int32(words[r >> 3]) << (31 - position)) >> 31


def _masked32(value, keep):
    bits = _as_i32(value)
    return _as_f32((bits & keep) | ((~keep) & fx.Int32(NEG_INF_BITS)))


def _read_ids32(env, tile):
    """Block ids of the wave's four blocks of a key tile (lanes 0..3; clamped tiles, whose
    blocks _rows32 replaces)."""
    # A scalar load would share lgkmcnt with the pipeline's LDS reads: their next wait
    # becomes lgkmcnt(0) and waits for the scalar load (~300 cycles). The descriptor spans
    # the task's list (slots past count read 0): a VALU offset per tile.
    resource = _buffer(fx.make_view(env.blocks, fx.make_layout(16, 1)), env.count * 4)
    offset = env.ids_offset + _min(tile, env.last_tile) * 64
    return _buffer_words(resource, offset, 1)[0]


def _read_mask32(env, tile):
    """The lane's query's 64 membership bits of a key tile (see _keep32)."""
    words = _buffer_words(
        env.gmask, env.mask_offset + _min(tile, env.last_tile) * (env.bq * 8), 2
    )
    return fx.Vector.from_elements([words[0], words[1]], fx.Int32)


def _gen_rows_steps32(env, ids, tile, out):
    """_rows32 one block per step."""
    rows = []
    for slot in range(4):
        rows.append(_row32(env, ids, tile, slot))
        yield 30, 1
    out.append(fx.Vector.from_elements(rows, fx.Int32))


def _row32(env, ids, tile, slot):
    block = fx.Int32(prim.readlane(fx.Int32(ids), slot))
    past = fx.Int32(tile) * 16 + (env.wave * 4 + slot) >= env.count
    if env.slotted == 2:
        return past.select(fx.Int32(-1), block)
    row = fx.Int32(fx.Uint32(block) * (4 * env.hk * D * 2))
    return past.select(env.oob_row, row)


def _rows32(env, ids, tile):
    """The wave's four blocks of key tile `tile` (ids: lanes 0..3): byte offsets of their
    first tokens (in the request's K/V, or the pool's with a slot table), or for SLOTTED 2
    pool blocks. A block past the list reads zeros: offset past the K/V extent, pool block
    -1 (no records)."""
    return fx.Vector.from_elements(
        [_row32(env, ids, tile, slot) for slot in range(4)], fx.Int32
    )


def _load32(env, resource, rows, j, offset):
    """Dims 8n .. 8n + 7 of key 16 wave + 2j + h of a key tile (block j >> 1, token
    2 (j & 1) + h), 16 bytes per lane at VOFFSET offset (_koffset32); tokens past the
    request read zeros (SLOTTED 1: the pool rows after them, which no membership bit
    selects)."""
    if env.slotted == 2:
        base, final, partial, full = resource
        block = fx.Int32(rows[j >> 1])
        records = (block == final).select(partial, full)
        records = (block < 0).select(fx.Int32(0), records)
        resource = _buffer_at(
            base + fx.Int64(fx.Uint32(block)) * (4 * env.hk * D * 2), records
        )
    return _buffer_words(resource, offset)


def _store32(env, j, words):
    """Load j's 16 bytes into the K tile in LDS (chunks XOR-swizzled by the key's phase, see
    _k32_bases)."""
    _write_address(env.k_wbase[(j & 1) | ((j >> 2) << 1)] + 1024 * j, words)
    env.lgkm.issue()


def _koffset32(env, rows, j):
    if env.slotted == 2:
        return env.token_off[j & 1]
    return fx.Int32(rows[j >> 1]) + env.token_off[j & 1]


def _gen_loads32(env, resource, rows, out, gap=GAP_KLOAD):
    # Each load's VOFFSET is added one load ahead (no VALU-to-VMEM dependency stall).
    offset = _koffset32(env, rows, 0)
    for j in range(8):
        out.append(_load32(env, resource, rows, j, offset))
        if j + 1 < 8:
            offset = _koffset32(env, rows, j + 1)
        yield C_VMEM + C_VALU, gap


def _gen_stores32(env, data):
    for j in range(8):
        _store32(env, j, data[j])
        yield C_LDS_W, GAP_WRITE


def _vbase32(env, rows, block):
    if env.slotted == 2:
        return env.lane8
    row = fx.Int32(rows[block])
    if env.slotted:
        # The partial last block reads through its own descriptor (_vload32).
        row = (row == env.final_row).select(fx.Int32(0), row)
    return row + env.lane8


def _vload32(env, resource, rows, j, base):
    """Dims 4l .. 4l + 3 of key 16 wave + j (block j >> 2, token j & 3), 8 bytes per lane
    at VOFFSET base (_vbase32) + token; tokens past the request read zeros."""
    step = (j & 3) * env.hk * D * 2
    if env.slotted == 1:
        pool, span, last, records = env.gv_final
        final = fx.Int32(rows[j >> 2]) == env.final_row
        resource = _buffer_at(
            final.select(last, pool), final.select(fx.Uint32(records), fx.Uint32(span))
        )
    elif env.slotted:
        start, final, partial, full = resource
        block = fx.Int32(rows[j >> 2])
        records = (block == final).select(partial, full)
        records = (block < 0).select(fx.Int32(0), records)
        resource = _buffer_at(
            start + fx.Int64(fx.Uint32(block)) * (4 * env.hk * D * 2), records
        )
    return _buffer_words(resource, base + step, 2)


def _gen_vloads32(env, resource, rows, out):
    # A block's VOFFSET base is added during the previous block's loads.
    base = _vbase32(env, rows, 0)
    for j in range(16):
        out.append(_vload32(env, resource, rows, j, base))
        if j % 4 == 3 and j + 1 < 16:
            base = _vbase32(env, rows, (j + 1) >> 2)
        yield C_VMEM, GAP_VLOAD


def _gen_vstores32(env, data):
    """V^T octets (keys 16 wave + 8o .. + 7) of the lane's dims 4l + d from the 16 loads."""
    for o in range(2):
        # One vmcnt wait for the octet's eight loads instead of one per v_perm.
        _keep_vgpr(_flat32(data[8 * o : 8 * o + 8]))
        for d in range(4):
            selector = 0x07060302 if d % 2 else 0x05040100
            words = [
                fx.Int32(
                    prim.perm_b32(
                        data[8 * o + 2 * i + 1][d // 2],
                        data[8 * o + 2 * i][d // 2],
                        selector,
                    )
                )
                for i in range(4)
            ]
            yield 4 * C_VALU, 1
            _write_address(
                env.vt_wbase[o] + 128 * d, fx.Vector.from_elements(words, fx.Int32)
            )
            env.lgkm.issue()
            yield C_LDS_W, GAP_WRITE


def _gen_meta32(env, t, out):
    out["ids2"] = _read_ids32(env, t + 2)
    yield 24
    out["mask1"] = _read_mask32(env, t + 1)
    yield 16


def _gen_rows32(env, t, meta, out):
    """The wave's rows of key tile t + 2 (block ids loaded by _gen_meta32)."""
    yield from _gen_rows_steps32(env, meta["ids2"], t + 2, out)


def _bf16_pair(a, b):
    """BF16 a (low half) and b (high half), rounded to nearest even like _pack_bf16."""
    ra, rb = (_as_i32(x) + 0x7FFF + ((_as_i32(x) >> 16) & 1) for x in (a, b))
    return fx.Int32(prim.perm_b32(rb, ra, 0x07060302))


def _gen_exp32(x_prev, l_pair, out, chunks=EXP_CHUNKS, due=None):
    """P(t - 1) = exp2(x) of the (kb, c) chunks (x = S * c - m, see _gen_softmax32), row
    sums (two partial sums, from l_pair or l_pair() when the stream starts) and the BF16
    PV B operands per chunk. With due (under PV), each chunk is due two MFMAs before its
    first PV MFMA."""
    if callable(l_pair):
        l_pair = l_pair()
    l_lo, l_hi = fx.Float32(l_pair[0]), fx.Float32(l_pair[1])
    for index, (kb, c) in enumerate(chunks):
        if due is not None and index:
            yield 0, 0, _first_pv32(kb, c) - 2
        values = []
        for i in range(4):
            values.append(_exp(fx.Float32(x_prev[kb * 16 + 4 * c + i])))
            yield C_EXP
        words = []
        for k in range(2):
            l_lo = l_lo + values[2 * k]
            l_hi = l_hi + values[2 * k + 1]
            words.append(_bf16_pair(values[2 * k], values[2 * k + 1]))
            yield 7 * C_VALU
        out[(kb, c)] = tuple(words)
    out["l"] = (l_lo, l_hi)


HUGE32 = 2.0**15  # |m| from which S * c - m rounds S * c first (see _gen_softmax32)


def _flag32(m):
    """Ballot of the rows whose m is at least HUGE32 in magnitude (rows with no key so far
    keep m = _MAX_FLOOR: their P is 0 either way)."""
    magnitude = _as_f32(_as_i32(m) & 0x7FFFFFFF)
    return prim.ballot((magnitude >= HUGE32) & (m != fx.Float32(_MAX_FLOOR)))


def _gen_softmax32(env, s_acc, mask, m_run, out, first_tile, robust=None):
    """S(t): membership masks, row max, lazy max m' and x = S * c - m' (P(t) = exp2(x)) as
    one FMA: exact S * c, at most half an ulp of S * c from the reference's rounded
    product, which is negligible while |m'| < HUGE32. Once some row's m' reaches it
    (logits near 1e11 have ulps of 1e5 in log2 units: exp2(ulp / 2) is inf), the wave
    takes x = RN(S * c) - m' instead (sticky; m' moves only where the lazy max triggers,
    so _rescale32 sets the flag and recomputes x from the masked S), so equal logits give
    equal x at any magnitude. The first tile's flag is set here; later tiles' ballot (the
    rescale branch's condition) includes every lane once the wave is flagged."""
    shift = fx.Uint32(env.mask_shift)
    words = [fx.Int32(fx.Uint32(mask[w]) >> shift) for w in range(2)]
    yield 2 * C_VALU
    values = []
    for kb in range(2):
        for r in range(16):
            value = fx.Float32(s_acc[kb][r])
            values.append(_masked32(value, _keep32(words, kb, r)))
            if r % 2:
                yield 4 * C_VALU
    level = values
    while len(level) > 1:
        level = [_maximum(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        if len(level) % 4 == 0 or len(level) <= 2:
            yield max(len(level), 2) * C_VALU
    local = level[0]
    other = fx.gpu.shuffle_xor(local, 32, 64)
    shuffled = env.lgkm.issue()
    # ds_bpermute: the next step waits for it, so let other streams run first; the steps
    # up to here are due before the Part 2 barrier (its lgkmcnt(0) would wait for it).
    yield 3 * C_VALU, 3, SOFT_DUE
    env.lgkm.wait(shuffled)
    tile_max = _maximum(local, other) * env.c_scale
    if first_tile:
        m_new = _maximum(tile_max, fx.Float32(_MAX_FLOOR)) + 1.0
        ballot = fx.Int64(0)
    else:
        bound = fx.Float32(m_run) + 7.0
        trigger = tile_max > bound
        # The rescale branch's ballot, flagged: every lane with a key (x = -inf without
        # one either way). One compare (an OR with the flag here stalls on the VALU-written
        # mask, and at the branch takes 8 SALU).
        flagged = (fx.Int32(robust) != 0).select(fx.Float32(_MAX_FLOOR), bound)
        ballot = prim.ballot(tile_max > flagged)
        m_new = trigger.select(tile_max + 1.0, fx.Float32(m_run))
    yield 6 * C_VALU
    neg = fx.Float32(0.0) - m_new
    x = []
    for k in range(32):
        x.append(_fma(values[k], env.c_scale, neg))
        if k % 2:
            # x is used only by the next phase: keep it from sinking past the rescale
            # branch.
            prim.hold(x[k - 1], x[k])
            yield 2 * C_VALU
    out.update(
        x=fx.Vector.from_elements(x, fx.Float32), s=values, m=m_new, ballot=ballot
    )
    if first_tile:
        out["robust"] = (_flag32(m_new) != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))


def _qk32(env, kpre, streams, after_bar):
    """QK(t) MFMAs, K reads KPRE ahead; see _part32."""
    s_acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
    reads = list(kpre) + [None] * (32 - len(kpre))

    def read(r):
        reads[r] = _k32_read(env, r % 2, r // 2)
        return C_LDS_R

    def issue(i):
        p, e, kb = QK_ORDER[i]
        cost = 4
        if i == 0:
            cost += read(KPRE)
            _pin32()
        if i % 4 == 0:
            env.lgkm.wait(reads[2 * p + 1][1])  # both reads of step pair p
        words = reads[2 * p + kb][0]
        s_acc[kb] = _mfma32(
            (words[2 * e], words[2 * e + 1]),
            (env.q[p][2 * e], env.q[p][2 * e + 1]),
            s_acc[kb],
        )
        # Read KPRE + j (first used by MFMA 2j + 2 KPRE) right after MFMA 2j - 1, ahead
        # of the slot's LDS writes.
        if i % 2 == 1 and (i + 1) // 2 + KPRE < 32:
            _pin32()
            cost += read((i + 1) // 2 + KPRE)
        return cost

    _part32(64, issue, streams, BAR1, after_bar, env.lgkm)
    return s_acc


def _gen_v0_32(env, raw):
    """After the Part 1 barrier: group 0's V reads, one per MFMA."""
    for b in range(8):
        raw.append(_vt_read(env, 0, b))
        yield C_LDS_R, 1


def _pv32(env, o, pw, raw0, streams, after_bar):
    """PV(t - 1) MFMAs, V^T reads one group ahead (during the group's first half); see
    _part32. raw0: group 0's reads."""
    o = list(o)
    raw = {0: raw0}

    def issue(i):
        g, rest = divmod(i, 16)
        s, b = divmod(rest, 8)
        kb, q2 = PV_GROUPS[g]
        cost = 4
        if rest == 0:
            env.lgkm.wait(raw[g][7][1])  # the group's eight reads
        words = raw[g][b][0]
        o[b] = _mfma32((words[2 * s], words[2 * s + 1]), pw[(kb, 2 * q2 + s)], o[b])
        # The next group's reads right after the MFMA, ahead of the slot's LDS writes.
        if g + 1 < 4 and rest < 8:
            _pin32()
            raw.setdefault(g + 1, []).append(_vt_read(env, g + 1, rest))
            cost += C_LDS_R
        return cost

    _part32(64, issue, streams, BAR2, after_bar, env.lgkm)
    return o


def _pv_last32(env, o, pw, streams):
    """PV(last) block-major, block b's eight MFMAs consecutive (the per-block order of
    _pv32, so the same sums), V^T reads two blocks ahead and waited once per block; no
    barrier (see _part32). Updates the list o in place: block b is final after MFMA
    8b + 7."""
    reads = {(g, b): _vt_read(env, g, b) for b in range(2) for g in range(4)}

    def issue(i):
        b, rest = divmod(i, 8)
        g, s = divmod(rest, 2)
        kb, q2 = PV_GROUPS[g]
        cost = 4
        if rest == 0:
            env.lgkm.wait(reads[(3, b)][1])  # the block's four reads
        words = reads[(g, b)][0]
        o[b] = _mfma32((words[2 * s], words[2 * s + 1]), pw[(kb, 2 * q2 + s)], o[b])
        if s == 1 and b + 2 < 8:
            _pin32()
            reads[(g, b + 2)] = _vt_read(env, g, b + 2)
            cost += C_LDS_R
        return cost

    _part32(64, issue, streams, 64, lambda: iter(()), env.lgkm)


def _gen_out_pair32(o, inv, words, i):
    """BF16 O / l of blocks 2i and 2i + 1 (words[r][i] per register r), once final."""
    for r in range(16):
        words[r][i] = _bf16_pair(
            fx.Float32(o[2 * i][r]) * inv, fx.Float32(o[2 * i + 1][r]) * inv
        )
        yield 7 * C_VALU


def _gen_kpre32(env, out, gap=1):
    for r in range(KPRE):
        out.append(_k32_read(env, r % 2, r // 2))
        yield C_LDS_R, gap


def _robust_x32(env, soft, x, robust):
    """x, or RN(S * c) - m' when the wave is flagged (see _gen_softmax32)."""
    flagged = robust != 0
    x = fx.Vector(x)
    return fx.Vector.from_elements(
        [
            flagged.select(
                fx.Float32(value) * env.c_scale - fx.Float32(soft["m"]),
                fx.Float32(x[k]),
            )
            for k, value in enumerate(soft["s"])
        ],
        fx.Float32,
    )


def _rescale32(env, o, l_pair, m_old, soft, robust):
    """When some row's max moved (lazy max) or the wave is flagged (see _gen_softmax32):
    O and l times 2^(m_old - m'), the flag updated from m' and, once flagged, x recomputed
    as RN(S * c) - m'. Rare, and exact no-ops where they do not apply."""

    def update(*values):
        corr = _exp(fx.Float32(m_old) - fx.Float32(soft["m"]))
        flagged = (values[11] != 0) | (_flag32(soft["m"]) != fx.Int64(0))
        flag = flagged.select(fx.Int32(1), fx.Int32(0))
        return (
            *(fx.Vector(v) * corr for v in values[:8]),
            *(fx.Float32(v) * corr for v in values[8:10]),
            _robust_x32(env, soft, values[10], flag),
            flag,
        )

    values = _lazy(soft["ballot"], (*o, *l_pair, soft["x"], fx.Int32(robust)), update)
    return list(values[:8]), tuple(values[8:10]), values[10], values[11]


def _gout32(a, env):
    """The output buffer and extent of env's task (descriptor in SGPRs)."""
    q0, hkv = _uniform(env.q0_pinned), _uniform(env.hkv_pinned)
    H, HK = a.H, a.HK
    extent = fx.Uint32(a.NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    view = fx.make_view(
        fx.get_iter(a.OUT) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D,
        fx.make_layout(a.NQ * H * D, 1),
    )
    return _buffer(view, extent), extent


def _inverse32(total):
    return (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))


def _output32(env, o, total, stream=None):
    """O / l as BF16 (see _output_words32)."""
    inv = _inverse32(total)
    words = []
    for r in range(16):
        values = [fx.Float32(o[b][r]) * inv for b in range(8)]
        words.append([_bf16_pair(values[2 * i], values[2 * i + 1]) for i in range(4)])
    _output_words32(env, words, stream)


def _output_words32(env, words, stream=None):
    """BF16 O / l (words[r]: 8 consecutive dims of register r) through the wave's 16 KiB
    of LDS (no V or K read is pending): per register r a lane writes 16 bytes of its row
    n; then each store covers two whole rows (512 contiguous bytes each)."""
    lane = _output_lane32(env)
    for r in range(16):
        _output_stage32(env, lane, r, words[r])
        if stream is not None and r % 2:
            _step32(stream)
    _output_rows32(env, lane, stream)


def _output_lane32(env):
    # With a split, the lane from a pin here: LLVM keeps every per-row address of the
    # output in registers across the kernel, and the split variant's VGPRs are short.
    return _pin_i32(env.lane) if env.split > 1 else env.lane


def _output_stage32(env, lane, r, words):
    """Register r's 16 bytes of the lane's row in the wave's LDS (_output_words32)."""
    n, h = lane & 31, lane >> 5
    position = 8 * (r // 4) + 4 * h + (r % 4)
    _write_address(
        env.o_shared + n * 512 + ((position ^ (n & 7)) << 4),
        fx.Vector.from_elements(list(words), fx.Int32),
    )


def _output_rows32(env, lane, stream=None):
    """The staged rows back from LDS and their stores (_output_words32)."""
    n, h = lane & 31, lane >> 5
    chunk = n
    # One KV head (GP == H): the task's rows are contiguous 512-byte rows of the output.
    contiguous = env.gp == env.h_all
    rows_valid = _min(env.qvalid, fx.Int32(env.bq)) * env.gp
    rows = []
    for k in range(16):
        row = 2 * k + h
        rows.append(
            _load_lds(env.o_lds, env.o_offset + row * 512 + ((chunk ^ (row & 7)) << 4))
        )
    # One full wait (LLVM would add partial ones per store, outside the ledger).
    rocdl.s_waitcnt(lgkmcnt=0)
    for k, data in enumerate(rows):
        row = 2 * k + h
        task_row = env.row_offset + env.wave * 32 + row
        if contiguous:
            valid = task_row < rows_valid
            offset = fx.Uint32(task_row * (D * 2) + chunk * 16)
        else:
            query, head = task_row // env.gp, task_row % env.gp
            valid = (query < env.qvalid) & (query < env.bq) & (head < env.heads)
            offset = fx.Uint32((query * env.h_all * D + head * D) * 2 + chunk * 16)
        _buffer_store_words(data, env.gout, valid.select(offset, env.gout_extent))
        if stream is not None and k % 2:
            _step32(stream)
    if stream is not None:
        _drain32(stream)


def _partial_o32(env, base, b, r4):
    """Byte offset of O^T registers 4 r4 .. 4 r4 + 3 of block b in a split part's record:
    each (b, r4) of a wave is 1 KiB, 16 bytes per lane (coalesced stores and loads)."""
    return base + (env.wave * 32 + b * 4 + r4) * 1024 + env.lane * 16


def _store_partial32(env, records, base, o, m_run, total):
    """A split part's O^T registers, m and l (system scope: the merging CTA may run on
    another XCD); both lanes of a row store its m and l alike."""
    for b in range(8):
        for r4 in range(4):
            words = [_as_i32(fx.Vector(o[b])[4 * r4 + i]) for i in range(4)]
            _buffer_store_words(
                fx.Vector.from_elements(words, fx.Int32),
                records,
                _partial_o32(env, base, b, r4),
                SC_SYSTEM,
            )
    row = (env.wave * 32 + env.n) * 4
    for offset, value in ((0, m_run), (BM * 4, total)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            records,
            base + PARTIAL_ML + offset + row,
            SC_SYSTEM,
        )


def _merge_output32(env, records, first_record, o, m_run, total, part, parts):
    """Merge of a task's `parts` key-range parts and _output32 for this part's index `part`,
    known at trace time (the merging wave's branch: no per-value selects). In part order,
    so the bits do not depend on which part merges: m = max m_p, w_p = 2^(m_p - m), O =
    sum O_p w_p and l = sum l_p w_p, fused from part 0 on (two parts: O = O_1 w_1 + O_0
    w_0); this part's terms come from its registers, the others' from their records (from
    first_record on, in part order). A part without a key of the row has m = FLOOR: weight
    0. Streamed: per unit (register quad r4, O^T block pair 2i, 2i + 1) the others' two
    quads load MERGE_AHEAD_WORDS words ahead of the folds; each value folds, scales by 1 /
    l and packs to BF16 as it comes, and the wave stages a register's 16 output bytes once
    its four block pairs are done."""
    row = (env.wave * 32 + env.n) * 4
    others = [
        first_record + (q + (1 if q >= part else 0)) * PARTIAL_BYTES
        for q in range(parts - 1)
    ]

    def term(p, mine, theirs):
        if p == part:
            return mine
        return theirs[p] if p < part else theirs[p - 1]

    def stat(base, offset):
        return _as_f32(
            _buffer_words(records, base + PARTIAL_ML + offset + row, 1, SC_SYSTEM)[0]
        )

    units = [(r4, i) for r4 in range(4) for i in range(4)]
    ahead = max(1, MERGE_AHEAD_WORDS // (8 * (parts - 1))) - 1

    def issue(u):
        r4, i = units[u]
        return [
            [
                _buffer_words(records, _partial_o32(env, base, b, r4), 4, SC_SYSTEM)
                for b in (2 * i, 2 * i + 1)
            ]
            for base in others
        ]

    stats = [[stat(base, offset) for base in others] for offset in (0, BM * 4)]
    # The first units' loads before the statistics' math (it waits for m and l only).
    pending = [issue(u) for u in range(min(ahead, len(units)))]
    maxima = [term(p, m_run, stats[0]) for p in range(parts)]
    sums = [term(p, total, stats[1]) for p in range(parts)]
    top = maxima[0]
    for m in maxima[1:]:
        top = _maximum(top, m)
    weights = [_exp(m - top) for m in maxima]
    merged_l = sums[0] * weights[0]
    for p in range(1, parts):
        merged_l = _fma(sums[p], weights[p], merged_l)
    inv = _inverse32(merged_l)
    lane = _output_lane32(env)
    words = [[None] * 4 for _ in range(16)]
    for u, (r4, i) in enumerate(units):
        if u + ahead < len(units):
            pending.append(issue(u + ahead))
        loads = pending[u]
        scaled = []
        for k, b in enumerate((2 * i, 2 * i + 1)):
            mine = fx.Vector(o[b])
            column = []
            for j in range(4):
                theirs = [_as_f32(load[k][j]) for load in loads]
                value = term(0, fx.Float32(mine[4 * r4 + j]), theirs) * weights[0]
                for p in range(1, parts):
                    t = term(p, fx.Float32(mine[4 * r4 + j]), theirs)
                    value = _fma(t, weights[p], value)
                column.append(value * inv)
            scaled.append(column)
        for j in range(4):
            words[4 * r4 + j][i] = _bf16_pair(scaled[0][j], scaled[1][j])
        if i == 3:
            for j in range(4):
                _output_stage32(env, lane, 4 * r4 + j, words[4 * r4 + j])
    _output_rows32(env, lane)


def _flat32(data):
    return fx.Vector.from_elements([w for part in data for w in part], fx.Int32)


def _unflat32(vector, count, words=4):
    vector = fx.Vector(vector)
    return [
        fx.Vector.from_elements([vector[words * j + i] for i in range(words)], fx.Int32)
        for j in range(count)
    ]


def _epilogue_p32(env, x_last, l_pair, v_data, pw, loads):
    """P(last) with the V(last) writes and the stream `loads` (the next task's Q loads)
    interleaved (no MFMAs to hide under)."""
    exps = _gen_exp32(fx.Vector(x_last), l_pair, pw)
    stores = _gen_vstores32(env, _unflat32(v_data, 16, 2))
    for i in range(16):
        _step32(stores)
        _step32(exps, 3)
        # The scalars' loads were issued just before P(last): give them time to land.
        if i >= 6:
            _step32(loads, 2)
    _drain32(exps)
    _drain32(loads)


# ---- Task scalars, per-task values and the cross-task prefetch ----
#
# A CTA runs its tasks back to back. During a task's epilogue (P(last), PV(last), output)
# the next task's scalars, Q, first block ids and mask, and K(0) are loaded, so its prologue
# starts with the K(0) writes. A task's context (8 scalars: mapped, enabled, q0, qvalid, k0,
# kv_len, count, tail) and prefetched registers (Q words (64), ids(0), ids(1), mask(0),
# rows(0), K(0) (32 words)) travel through the task loop.


class _Args32:
    """The kernel arguments and lane values shared by every task of a CTA."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _entry_task(a, mapped):
    """The task of an order entry (see _split_entry_layout)."""
    if a.SPLIT > 1:
        return mapped >> _split_entry_layout(a.SPLIT)[2]
    return mapped


def _lookahead_order32(a, work):
    """Lane k loads the ORDER entry of work item work + (k + 1) GRID (a raw VGPR value)."""
    item = work + (a.lane + 1) * a.GRID
    inside = item < a.limit
    return inside.select(a.ORDER[_min(item, a.limit - 1)], fx.Int32(-1))


def _lookahead_active32(a, mapped):
    """The ACTIVE flag of lane k's ORDER entry's tile (a raw VGPR value)."""
    task = _entry_task(a, mapped)
    tile = (mapped >= 0).select(task // (a.HK * a.slices), fx.Int32(0))
    return (mapped >= 0).select(a.ACTIVE[tile], fx.Int32(0))


def _lookahead_loads32(a, work):
    """Lane k loads the ORDER entry of work item work + (k + 1) GRID and its tile's ACTIVE
    flag (raw VGPR values; see _lookahead32)."""
    mapped = _lookahead_order32(a, work)
    return mapped, _lookahead_active32(a, mapped)


def _lookahead32(a, work, loads):
    """(work item, ORDER entry) of the first enabled task among the 64 work items after
    `work`; without one, (work + 64 GRID, -1)."""
    mapped, active = loads
    found = prim.ballot(active != 0)
    first = _min(fx.Int32(fx.math.cttz(found)), fx.Int32(64))
    entry = prim.readlane(mapped, _min(first, fx.Int32(63)))
    return work + (first + 1) * a.GRID, (first < 64).select(entry, fx.Int32(-1))


def _ctx_loads32(a, mapped):
    """Raw loads (VGPR values) of task entry mapped's ACTIVE, META, COUNTS and TAILS."""
    task = _entry_task(a, mapped)
    tile = (mapped >= 0).select(task // (a.HK * a.slices), fx.Int32(0))
    raw = [a.ACTIVE[tile]] + [a.META[tile * 5 + i] for i in range(4)]
    raw.append(a.COUNTS[tile])
    raw.append(a.TAILS[tile] if a.SLOTTED else fx.Int32(-1))
    return raw


def _ctx32(mapped, raw):
    """The task's scalars (SGPRs) once its raw loads landed. Without an enabled task no
    query row or block is valid, so its prefetch loads read nothing from memory."""
    values = [_uniform(v) for v in raw]
    enabled = (mapped >= 0).select(values[0], fx.Int32(0))
    live = enabled != 0
    q0, qvalid, k0, kv_len, count, tail = values[1:]
    qvalid = live.select(qvalid, fx.Int32(0))
    count = live.select(count, fx.Int32(0))
    return (mapped, enabled, q0, qvalid, k0, kv_len, count, tail)


def _task32(a, ctx):
    """Per-task values of task context ctx (see _Env32 users)."""
    mapped, _, q0, qvalid, k0, kv_len, count, tail = ctx
    H, HK, GP, BQ, CAP = a.H, a.HK, a.GP, a.BQ, a.CAP
    task = _entry_task(a, mapped)
    task = (mapped >= 0).select(task, fx.Int32(0))
    tile_id, hkv = task // (HK * a.slices), task % HK
    row_offset = ((task // HK) % a.slices) * 128
    first = fx.Int32(0)
    blocks = fx.get_iter(a.BLOCKS) + tile_id * CAP
    if a.SPLIT > 1:
        # Part p of SPLIT runs N64 steps [ceil(p steps / SPLIT), ceil((p + 1) steps / SPLIT)).
        parted = (mapped & 1) != 0
        part = (mapped >> 1) & _split_entry_layout(a.SPLIT)[0]
        steps = (count + 15) >> 4

        def bound(p):
            ceiling = p * steps + (a.SPLIT - 1)
            if a.SPLIT & (a.SPLIT - 1) == 0:
                return ceiling >> (a.SPLIT.bit_length() - 1)
            return ceiling // a.SPLIT

        first = parted.select(bound(part), fx.Int32(0))
        end = parted.select(bound(part + 1), steps)
        count = _min(count, end * 16) - first * 16
        blocks = fx.get_iter(a.BLOCKS) + (tile_id * CAP + first * 16)
    masks = fx.make_view(
        fx.get_iter(a.MEMBERS) + (tile_id * (CAP // 16) + first) * BQ * 4,
        fx.make_layout((CAP // 16) * BQ * 4, 1),
    )
    row = row_offset + a.wave * 32 + a.n
    query, head = row // GP, row % GP
    valid_row = (query < qvalid) & (query < BQ) & (head < (H // HK))
    # Byte offsets are unsigned 32-bit: Q and each request's K/V may span up to 4 GiB.
    qextent = fx.Uint32(a.NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    qptr = fx.get_iter(a.Q) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D
    gq = _buffer(fx.make_view(qptr, fx.make_layout(a.NQ * H * D, 1)), qextent)
    qoffset = valid_row.select(fx.Uint32((query * H * D + head * D) * 2), qextent)
    row_bytes = HK * D * 2
    oob_row = fx.Int32(0)
    final_row, gv_final = fx.Int32(-1), None
    if a.SLOTTED == 1:
        # One descriptor over the pool; a block past the list starts at the span. Only V's
        # partial last block (tokens past the request, which may be NaN) needs its own
        # bound: K's are masked out of S.
        extent = fx.Uint32(a.POOL) - hkv * D * 2
        oob_row = fx.Int32(_min(extent, fx.Uint32(0xFFFFFFFF - 4 * row_bytes)))
        gk, gv = (
            _buffer(
                fx.make_view(
                    fx.get_iter(tensor) + hkv * D, fx.make_layout(extent // 2, 1)
                ),
                extent,
            )
            for tensor in (a.K, a.V)
        )
        final = (tail >= 0).select(tail >> 2, fx.Int32(-1))
        final_row = (tail >= 0).select(
            fx.Int32(fx.Uint32(final) * (4 * row_bytes)), fx.Int32(-1)
        )
        v_base = fx.Int64(fx.ptrtoint(fx.get_iter(a.V))) + fx.Int64(hkv * D * 2)
        # (pool base, span; the partial last block's base, its valid tokens' bytes)
        gv_final = (
            v_base,
            extent,
            v_base + fx.Int64(fx.Uint32(final_row)),
            (tail >= 0).select((kv_len & 3) * row_bytes, fx.Int32(0)),
        )
    elif a.SLOTTED:
        final = (tail >= 0).select(tail >> 2, fx.Int32(-1))
        partial = (kv_len & 3) * row_bytes
        full = fx.Int32(4 * row_bytes)
        offset = fx.Int64(hkv * D * 2)
        gk = (fx.Int64(fx.ptrtoint(fx.get_iter(a.K))) + offset, final, partial, full)
        gv = (fx.Int64(fx.ptrtoint(fx.get_iter(a.V))) + offset, final, partial, full)
    else:
        extent = (fx.Uint32(kv_len) * HK - hkv) * D * 2
        # A block past the list starts at the extent (requests below 4 GiB - 16 KiB).
        oob_row = fx.Int32(_min(extent, fx.Uint32(0xFFFFFFFF - 4 * row_bytes)))
        gk, gv = (
            _buffer(
                fx.make_view(
                    fx.get_iter(tensor) + (fx.Int64(k0) * HK + hkv) * D,
                    fx.make_layout(extent // 2, 1),
                ),
                extent,
            )
            for tensor in (a.K, a.V)
        )
    key_tiles = (count * 4 + BN - 1) // BN
    return {
        "blocks": blocks,
        "count": count,
        "gmask": _buffer(masks, ((CAP // 16) - first) * BQ * 8),
        "mask_offset": _min(query, fx.Int32(BQ - 1)) * 8,
        "last_tile": key_tiles - 1,
        "key_tiles": key_tiles,
        "oob_row": oob_row,
        "gk": gk,
        "gv": gv,
        "final_row": final_row,
        "gv_final": gv_final,
        "gq": gq,
        "qoffset": qoffset,
        # In VGPRs across the key loop: the output's descriptor is rebuilt from them.
        "q0_pinned": _pin_i32(q0),
        "hkv_pinned": _pin_i32(hkv),
        "row_offset": row_offset,
        "qvalid": qvalid,
        "qextent": qextent,
    }


def _static32(a, k_lds, v_lds, scale_log2):
    """The lane and LDS values shared by every task (an _Env32 without per-task fields)."""
    # In a VGPR: the phase loop needs every SGPR (see _body).
    scale_bits = _pin_i32(fx.Float32(scale_log2).bitcast(fx.Int32))
    wave, lane, n, h = a.wave, a.lane, a.n, a.h
    row_bytes = a.HK * D * 2
    k_shared = fx.Int32(fx.ptrtoint(fx.get_iter(k_lds)))
    v_shared = fx.Int32(fx.ptrtoint(fx.get_iter(v_lds)))
    write_row = (wave * 16 + h) * 512
    # Output staging: waves 0-1 in the K tile, 2-3 in the V tile, 16 KiB each.
    o_shared = (wave < 2).select(k_shared, v_shared - 32768) + wave * 16384
    return _Env32(
        k_lds=k_lds,
        v_lds=v_lds,
        wave=wave,
        lane=lane,
        h=h,
        hk=a.HK,
        bq=a.BQ,
        slotted=a.SLOTTED,
        split=a.SPLIT,
        c_scale=scale_bits.bitcast(fx.Float32),
        ids_offset=(wave * 4 + (lane & 3)) * 4,
        mask_shift=h * 16,
        token_off=tuple(
            _pin_i32((2 * odd + h) * row_bytes + n * 16) for odd in range(2)
        ),
        k_wbase=tuple(
            k_shared + write_row + ((n ^ ((2 * (p & 1) + h) | ((p >> 1) << 2))) << 4)
            for p in range(4)
        ),
        kbases=_k32_bases(n, h),
        lane8=_pin_i32(lane * 8),
        vt_wbase=tuple(
            v_shared
            + lane * 512
            + (((wave * 2 + octet) ^ ((lane ^ (lane >> 1)) & 7)) << 4)
            for octet in range(2)
        ),
        vt_rbase=tuple(
            tuple(
                n * 1024 + (((4 * kb + 2 * q2 + h) ^ ((n ^ (2 * n + half)) & 7)) << 4)
                for half in range(2)
            )
            for kb, q2 in PV_GROUPS
        ),
        o_lds=k_lds,
        o_offset=o_shared - k_shared,
        o_shared=o_shared,
        q_shared=v_shared + wave * 8192,
        qk_shared=k_shared + wave * 8192,
        q_offset=wave * 8192,
        n=n,
        gp=a.GP,
        heads=a.H // a.HK,
        h_all=a.H,
    )


def _task_env32(a, base, ctx):
    """base (_static32) with task ctx's per-task values."""
    env = _Env32(**base.__dict__)
    env.__dict__.update(_task32(a, ctx))
    return env


def _q_offsets32(env):
    """Byte offsets of the wave's rows 4k + l // 16 (k = 0..7) for the coalesced Q loads:
    each load takes 256 contiguous bytes of four rows (rows past the task: the extent).
    """
    offsets = []
    for k in range(8):
        task_row = env.row_offset + env.wave * 32 + 4 * k + (env.lane >> 4)
        query, head = task_row // env.gp, task_row % env.gp
        valid = (query < env.qvalid) & (query < env.bq) & (head < env.heads)
        offset = fx.Uint32(
            (query * env.h_all * D + head * D) * 2 + (env.lane & 15) * 16
        )
        offsets.append(valid.select(offset, env.qextent))
    return offsets


def _q_loads32(env):
    """The wave's 32 Q rows in 16 coalesced loads: load 8r + k holds dims 128r + 8 (l % 16)
    .. + 7 of row 4k + l // 16."""
    offsets = _q_offsets32(env)
    return [
        _buffer_words(env.gq, offsets[k] + 256 * r) for r in range(2) for k in range(8)
    ]


def _gen_q_stage32(env, loads, q, in_k=False):
    """Q B operands (q[p]: dims 16p + 8h .. + 7 of row n) from the coalesced loads, through
    the wave's 8 KiB of the V tile (free until V(0)'s writes) or, with in_k, of the K tile
    (free during the previous task's PV(last)), half the dims per round. Its LDS operations
    count in env.lgkm when in_k (no wait: the Part 2 barrier's lgkmcnt(0) covers them).
    """
    lds, shared, offset = (
        (env.k_lds, env.qk_shared, env.q_offset)
        if in_k
        else (env.v_lds, env.q_shared, env.q_offset)
    )
    q.extend([None] * 16)
    for r in range(2):
        for k in range(8):
            row = 4 * k + (env.lane >> 4)
            chunk = env.lane & 15
            _write_address(
                shared + row * 256 + ((chunk ^ (row & 7)) << 4), loads[8 * r + k]
            )
            if in_k:
                env.lgkm.issue()
            yield C_LDS_W, GAP_WRITE
        for p in range(8 * r, 8 * r + 8):
            chunk = 2 * p + env.h - 16 * r
            words = _load_lds(lds, offset + env.n * 256 + ((chunk ^ (env.n & 7)) << 4))
            if in_k:
                env.lgkm.issue()
            q[p] = tuple(words[i] for i in range(4))
            yield C_LDS_R


def _q_stage32(env, loads):
    """_gen_q_stage32 through the V tile, without overlap."""
    q = []
    _drain32(_gen_q_stage32(env, loads, q))
    rocdl.s_waitcnt(lgkmcnt=0)
    return q


def _prefetch_issue32(env):
    """Q (coalesced), ids(0), ids(1) and mask(0) loads of env's task."""
    q = _q_loads32(env)
    ids = [_read_ids32(env, fx.Int32(t)) for t in range(2)]
    return q, ids[0], ids[1], _read_mask32(env, fx.Int32(0))


def _gen_prefetch_q32(a, base, mapped, raw, out):
    """Under P(last): the next task's scalars (raw: _ctx_loads32), then its Q loads
    (coalesced), into out."""
    ctx = _ctx32(mapped, raw)
    env = _task_env32(a, base, ctx)
    out.update(ctx=ctx, env=env)
    yield 60
    offsets = _q_offsets32(env)
    yield 60
    q = []
    for r in range(2):
        for k in range(8):
            q.append(_buffer_words(env.gq, offsets[k] + 256 * r))
            yield C_VMEM
    out["q_loads"] = q


def _gen_prefetch_ids32(env, out):
    """Under PV(last): the next task's Q through the K tile (its loads landed during
    P(last)), then its ids(0), ids(1) and mask(0) loads, into out (env: this task's,
    whose ledger counts the staging)."""
    nxt = out["env"]
    q = []
    yield from _gen_q_stage32(env, out["q_loads"], q, True)
    out["q"] = q
    out["ids0"] = _read_ids32(nxt, fx.Int32(0))
    yield 2 * C_VMEM
    out["ids1"] = _read_ids32(nxt, fx.Int32(1))
    yield 2 * C_VMEM
    out["mask0"] = _read_mask32(nxt, fx.Int32(0))
    yield 2 * C_VMEM


def _gen_prefetch_k32(env, ids0, out):
    """rows(0) and the K(0) loads of env's task (out: rows, then the 8 loads)."""
    rows = []
    yield from _gen_rows_steps32(env, ids0, fx.Int32(0), rows)
    out.append(rows[0])
    data = []
    yield from _gen_loads32(env, env.gk, rows[0], data, 2)
    out.append(data)


def _pre_pack32(q, ids1, mask0, rows0, kdata):
    return (
        _flat32(q),
        fx.Int32(ids1),
        fx.Vector(mask0),
        fx.Vector(rows0),
        _flat32(kdata),
    )


def _pre_now32(a, base, ctx):
    """The prefetch of task ctx without overlap (first task, after a disabled one)."""
    env = _task_env32(a, base, ctx)
    q, ids0, ids1, mask0 = _prefetch_issue32(env)
    out = []
    _drain32(_gen_prefetch_k32(env, ids0, out))
    return _pre_pack32(_q_stage32(env, q), ids1, mask0, out[0], out[1])


def _ctx_now32(a, work):
    """(work item, task scalars) of the first enabled task after `work`, without overlap."""
    work, mapped = _lookahead32(a, work, _lookahead_loads32(a, work))
    return work, _ctx32(mapped, _ctx_loads32(a, mapped))


def _args32(
    Q,
    K,
    V,
    OUT,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    TAILS,
    ACTIVE,
    ORDER,
    H,
    HK,
    NQ,
    CAP,
    BQ,
    GP,
    TASKS,
    GRID,
    SLOTTED,
    SPLIT,
    POOL,
):
    tid = fx.thread_idx.x
    lane = tid & 63
    return _Args32(
        Q=Q,
        K=K,
        V=V,
        OUT=OUT,
        META=META,
        BLOCKS=BLOCKS,
        MEMBERS=MEMBERS,
        COUNTS=COUNTS,
        TAILS=TAILS,
        ACTIVE=ACTIVE,
        ORDER=ORDER,
        H=H,
        HK=HK,
        NQ=NQ,
        CAP=CAP,
        BQ=BQ,
        GP=GP,
        GRID=GRID,
        limit=((TASKS + GRID - 1) // GRID) * GRID,
        slices=(BQ * GP + 127) // 128,
        SLOTTED=SLOTTED,
        SPLIT=SPLIT,
        POOL=POOL,
        wave=_uniform(tid >> 6),
        lane=lane,
        n=lane & 31,
        h=lane >> 5,
    )


def _phase32(
    env,
    gk,
    gv,
    o,
    x_prev,
    m_run,
    l_lo,
    l_hi,
    kpre,
    rows_cur,
    rows_next,
    mask_cur,
    v_data,
    robust,
    t,
):
    """Phase t of _body32's key loop: QK(t) with P(t - 1), the V(t - 1) writes and K(t + 1)
    loads; PV(t - 1) with S(t)'s softmax, the K(t + 1) writes and V(t) loads."""
    pw, meta, k_data, raw0, rows = {}, {}, [], [], []
    env.lgkm = prim.LgkmLedger(KPRE)
    s_acc = _qk32(
        env,
        list(zip(_unflat32(kpre, KPRE), range(KPRE))),
        [
            _Stream32(
                _gen_exp32(fx.Vector(x_prev), (l_lo, l_hi), pw, EXP_CHUNKS[:P1_CHUNKS]),
                *P1_EXP,
            ),
            _Stream32(_gen_vstores32(env, _unflat32(v_data, 16, 2)), *P1_VWRITE),
            _Stream32(_gen_loads32(env, gk, rows_next, k_data), *P1_KLOAD),
            _Stream32(_gen_meta32(env, t, meta), *P1_META),
        ],
        lambda: _chain32(_gen_rows32(env, t, meta, rows), _gen_v0_32(env, raw0)),
    )
    soft, kpre_next, v_next = {}, [], []
    o = _pv32(
        env,
        list(o),
        pw,
        raw0,
        [
            _Stream32(
                _gen_exp32(
                    fx.Vector(x_prev), lambda: pw["l"], pw, EXP_CHUNKS[P1_CHUNKS:], True
                ),
                *P2_EXP,
            ),
            _Stream32(
                _gen_softmax32(env, s_acc, mask_cur, m_run, soft, False, robust),
                *P2_SOFT,
            ),
            _Stream32(_gen_stores32(env, k_data), *P2_KWRITE),
            _Stream32(_gen_vloads32(env, gv, rows_cur, v_next), *P2_VLOAD),
        ],
        lambda: _gen_kpre32(env, kpre_next),
    )
    assert env.lgkm.issued - env.lgkm.done == KPRE
    o, l_pair, s_next, robust = _rescale32(env, o, pw["l"], m_run, soft, robust)
    return (
        *o,
        s_next,
        soft["m"],
        *l_pair,
        _flat32([words for words, _ in kpre_next]),
        rows_next,
        rows[0],
        meta["mask1"],
        _flat32(v_next),
        robust,
    )


@flyc.jit
def _body32(
    Q,
    K,
    V,
    OUT,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    TAILS,
    ACTIVE,
    ORDER,
    PARTIALS,
    COUNTERS,
    k_lds,
    v_lds,
    work,
    c_mapped,
    c_enabled,
    c_q0,
    c_qvalid,
    c_k0,
    c_kv_len,
    c_count,
    c_tail,
    p_q,
    p_ids1,
    p_mask0,
    p_rows0,
    p_kdata,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE_LOG2: fx.Float32,
    SLOTTED: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    POOL: fx.Int32,
):
    """One task on four waves of 32 rows (see the section comment); returns the next task's
    context and prefetched registers.

    Softmax in log2 units: S' = S * SCALE_LOG2 - m, P = exp2(S'), l = sum(P), O = sum(P V),
    output O / l. m starts at the first row max + 1 and moves to a later row max + 1 only
    when that exceeds m + 7, rescaling O and l.
    """
    a = _args32(
        Q,
        K,
        V,
        OUT,
        META,
        BLOCKS,
        MEMBERS,
        COUNTS,
        TAILS,
        ACTIVE,
        ORDER,
        H,
        HK,
        NQ,
        CAP,
        BQ,
        GP,
        TASKS,
        GRID,
        SLOTTED,
        SPLIT,
        POOL,
    )
    base = _static32(a, k_lds, v_lds, SCALE_LOG2)
    ctx = (c_mapped, c_enabled, c_q0, c_qvalid, c_k0, c_kv_len, c_count, c_tail)
    env = _task_env32(a, base, ctx)
    # The next work item: ORDER entries at the task start, their ACTIVE flags after the
    # first barrier, the lookahead before the key loop (all land during this task).
    order_next = _lookahead_order32(a, work)
    env.q = [tuple(w[i] for i in range(4)) for w in _unflat32(p_q, 16)]
    gk, gv, key_tiles = env.gk, env.gv, env.key_tiles
    split_bases = None
    if fx.const_expr(SPLIT > 1):
        # In VGPRs across the key loop (see _kernel's split_bases).
        split_bases = tuple(
            _pin_i32(word)
            for tensor in (PARTIALS, COUNTERS)
            for word in _split_i64(fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
        )

    # Prologue: K(0) (prefetched) to LDS, QK(0) with the K(1) / V(0) loads, the K(1) writes
    # after its barrier, S'(0) and m; barrier, K(1)'s first reads.
    ids1, mask0, rows0 = p_ids1, fx.Vector(p_mask0), fx.Vector(p_rows0)
    rows1 = _rows32(env, ids1, fx.Int32(1))
    env.lgkm = prim.LgkmLedger()
    _drain32(_gen_stores32(env, _unflat32(p_kdata, 8)))
    rocdl.s_waitcnt(lgkmcnt=0)
    env.lgkm.all_done()
    _stage_end()
    active_next = _lookahead_active32(a, order_next)
    kpre, meta, k1_data, v0_data, soft = [], {}, [], [], {}
    _drain32(_gen_kpre32(env, kpre))
    s_acc = _qk32(
        env,
        kpre,
        [
            _Stream32(_gen_loads32(env, gk, rows1, k1_data), *P1_KLOAD),
            _Stream32(_gen_vloads32(env, gv, rows0, v0_data), *P1_VLOAD0),
            _Stream32(_gen_meta32(env, fx.Int32(0), meta), *P1_META),
        ],
        lambda: _gen_stores32(env, k1_data),
    )
    _drain32(_gen_softmax32(env, s_acc, mask0, fx.Float32(_MAX_FLOOR), soft, True))
    rows2 = _rows32(env, meta["ids2"], fx.Int32(2))
    rocdl.s_waitcnt(lgkmcnt=0)
    env.lgkm.all_done()
    _stage_end()
    kpre = []
    _drain32(_gen_kpre32(env, kpre))
    # Each phase starts with exactly these KPRE reads pending (ledger indices 0..KPRE-1).
    assert env.lgkm.issued - env.lgkm.done == KPRE
    # Its scalars load at the epilogue. In VGPRs across the key loop (SGPRs are short),
    # but in SGPRs with a split (its VGPRs are short: the split bases).
    work_next, mapped_next = _lookahead32(a, work, (order_next, active_next))
    if fx.const_expr(SPLIT == 1):
        work_next, mapped_next = _pin_i32(work_next), _pin_i32(mapped_next)
    state = (
        *[fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(8)],
        _robust_x32(env, soft, soft["x"], soft["robust"]),
        soft["m"],
        fx.Float32(0.0),
        fx.Float32(0.0),
        _flat32([words for words, _ in kpre]),
        rows1,
        rows2,
        meta["mask1"],
        _flat32(v0_data),
        soft["robust"],
    )

    @flyc.jit
    def phase(
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        x_prev,
        m_run,
        l_lo,
        l_hi,
        kpre,
        rows_cur,
        rows_next,
        mask_cur,
        v_data,
        robust,
        t,
    ):
        return _phase32(
            env,
            gk,
            gv,
            (o0, o1, o2, o3, o4, o5, o6, o7),
            x_prev,
            m_run,
            l_lo,
            l_hi,
            kpre,
            rows_cur,
            rows_next,
            mask_cur,
            v_data,
            robust,
            t,
        )

    for t in range(fx.Int32(1), key_tiles, fx.Int32(1)):
        state = phase(*state, t)
    # Epilogue: the next task's scalars and Q load under P(last) and the V(last) writes,
    # its Q goes through the K tile and its first ids and mask load under PV(last), its
    # rows(0) and K(0) under the output.
    if fx.const_expr(SPLIT == 1):
        work_next, mapped_next = _uniform(work_next), _uniform(mapped_next)
    raw_next = _ctx_loads32(a, mapped_next)
    nxt = {}
    pw, raw0 = {}, []
    env.lgkm = prim.LgkmLedger(KPRE)
    _epilogue_p32(
        env,
        state[8],
        state[10:12],
        state[16],
        pw,
        _gen_prefetch_q32(a, base, mapped_next, raw_next, nxt),
    )
    rocdl.s_waitcnt(lgkmcnt=0)
    env.lgkm.all_done()
    _stage_end()
    local = fx.Float32(pw["l"][0]) + fx.Float32(pw["l"][1])
    k_next = []
    if fx.const_expr(SPLIT > 1):
        _drain32(_gen_v0_32(env, raw0))
        o = _pv32(
            env,
            list(state[:8]),
            pw,
            raw0,
            [_Stream32(_gen_prefetch_ids32(env, nxt), *PV_PREFETCH)],
            lambda: iter(()),
        )
        total = local + fx.gpu.shuffle_xor(local, 32, 64)
        env.gout, env.gout_extent = _gout32(a, env)
        _drain32(_gen_prefetch_k32(nxt["env"], nxt["ids0"], k_next))
        m_run = fx.Float32(state[9])
        entry = c_mapped
        if (entry & 1) != 0:
            part_mask, slot_shift, _ = _split_entry_layout(SPLIT)
            part = _uniform((entry >> 1) & part_mask)
            partials, counters = (
                _join_i64(_uniform(split_bases[i]), _uniform(split_bases[i + 1]))
                for i in (0, 2)
            )
            records = _buffer_at(partials, SPLIT_SLOTS * SPLIT * PARTIAL_BYTES)
            slot = (entry >> slot_shift) & (SPLIT_SLOTS - 1)
            _store_partial32(
                env,
                records,
                (slot * SPLIT + part) * PARTIAL_BYTES,
                o,
                m_run,
                total,
            )
            # The wave's record is out before its rows count: each wave counts its 32
            # rows in its byte of the slot's word (lane 0's buffer atomic, the other
            # lanes out of range), and the wave counting last merges them (no barrier:
            # each wave stages its output in its own LDS).
            rocdl.s_waitcnt(vmcnt=0)
            shift = env.wave * 8
            counted = prim.buffer_atomic(
                "add",
                _buffer_at(counters, SPLIT_SLOTS * 4),
                (env.lane == 0).select(slot * 4, fx.Int32(SPLIT_SLOTS * 4)),
                fx.Int32(1) << shift,
            )
            done = _uniform((prim.readlane(counted, 0) >> shift) & 0xFF)
            if done == SPLIT - 1:
                for k in fx.range_constexpr(SPLIT):
                    if part == k:
                        _merge_output32(
                            env,
                            records,
                            slot * SPLIT * PARTIAL_BYTES,
                            o,
                            m_run,
                            total,
                            k,
                            SPLIT,
                        )
        else:
            _output32(env, o, total)
    else:
        # 1 / l before PV(last); each block pair's BF16 output under the MFMAs left.
        other = fx.gpu.shuffle_xor(local, 32, 64)
        env.lgkm.wait(env.lgkm.issue())
        inv = _inverse32(local + other)
        o = list(state[:8])
        words = [[None] * 4 for _ in range(16)]
        _pv_last32(
            env,
            o,
            pw,
            [_Stream32(_gen_prefetch_ids32(env, nxt), *PV_PREFETCH)]
            + [
                _Stream32(_gen_out_pair32(o, inv, words, i), 16 * i + 16, 63, 13)
                for i in range(4)
            ],
        )
        # Every wave's V and Q-staging reads are done before the output staging.
        rocdl.s_waitcnt(lgkmcnt=0)
        env.lgkm.all_done()
        _stage_end()
        env.gout, env.gout_extent = _gout32(a, env)
        _output_words32(env, words, _gen_prefetch_k32(nxt["env"], nxt["ids0"], k_next))
    # The next task's K(0) writes overwrite the LDS.
    _stage_end()
    return (
        work_next,
        *nxt["ctx"],
        *_pre_pack32(nxt["q"], nxt["ids1"], nxt["mask0"], k_next[0], k_next[1]),
    )


@flyc.kernel(name="sparse_prefill_union_bf16_d256", known_block_size=[THREADS32, 1, 1])
def _kernel32(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE_LOG2: fx.Float32,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    SLOTTED: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    POOL: fx.Int32,
):
    allocator = fx.SharedAllocator()
    k_lds, v_lds = (
        allocator.allocate(fx.Array[fx.Int8, LDS_BYTES // 2, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES // 2, 1))
        for _ in range(2)
    )
    a = _args32(
        Q,
        K,
        V,
        OUT,
        META,
        BLOCKS,
        MEMBERS,
        COUNTS,
        TAILS,
        ACTIVE,
        ORDER,
        H,
        HK,
        NQ,
        CAP,
        BQ,
        GP,
        TASKS,
        GRID,
        SLOTTED,
        SPLIT,
        POOL,
    )
    base = _static32(a, k_lds, v_lds, SCALE_LOG2)
    # Work items whose tile is not active are skipped: each task looks ahead for the next.
    work, ctx = _ctx_now32(a, fx.block_idx.x - GRID)
    carry = (work, *ctx, *_pre_now32(a, base, ctx))
    while carry[0] < a.limit:
        if carry[2] != 0:
            carry = _body32(
                Q,
                K,
                V,
                OUT,
                META,
                BLOCKS,
                MEMBERS,
                COUNTS,
                TAILS,
                ACTIVE,
                ORDER,
                PARTIALS,
                COUNTERS,
                k_lds,
                v_lds,
                *carry,
                H,
                HK,
                NQ,
                CAP,
                BQ,
                GP,
                TASKS,
                GRID,
                SCALE_LOG2,
                SLOTTED,
                SPLIT,
                POOL,
            )
        else:
            work, ctx = _ctx_now32(a, carry[0])
            carry = (work, *ctx, *_pre_now32(a, base, ctx))


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SLOTTED: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    POOL: fx.Int32,
    stream: fx.Stream,
):
    if GRID > 0:
        _kernel32(
            Q,
            K,
            V,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            TAILS,
            ACTIVE,
            ORDER,
            H,
            HK,
            NQ,
            CAP,
            BQ,
            GP,
            TASKS,
            GRID,
            SCALE_LOG2,
            PARTIALS,
            COUNTERS,
            SLOTTED,
            SPLIT,
            POOL,
            value_attrs=prim.kernel_attrs(),
        ).launch(grid=(GRID, 1, 1), block=(THREADS32, 1, 1), stream=stream)


_COMPILED = {}


def _args(inputs, plan, out, stream):
    # Slot-table plans read K/V through K3's pool block table, addressed from the pool's base.
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        inputs.scale_log2,
        plan.metadata.view(-1),
        (plan.pool_blocks if plan.slotted else plan.blocks).view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.tails if plan.slotted else plan.counts,
        plan.active,
        plan.task_order,
        plan.partials.view(-1) if plan.split > 1 else plan.counts,
        plan.split_counters if plan.split > 1 else plan.counts,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        plan.block_capacity,
        plan.query_tile,
        plan.group_padded,
        # Order entries (tasks times key-range parts).
        plan.num_tiles
        * inputs.k.shape[1]
        * ((plan.query_tile * plan.group_padded + 127) // 128)
        * plan.split,
        plan.grid,
        _slotted(inputs, plan),
        plan.split,
        _pool_span(inputs, plan),
        stream,
    )


def _pool_span(inputs, plan):
    """SLOTTED 1: the pool's byte span (an unsigned 32-bit kernel argument), else 0."""
    if _slotted(inputs, plan) != 1:
        return 0
    span = inputs.k.numel() * inputs.k.element_size()
    return span - 2**32 if span >= 2**31 else span


def _slotted(inputs, plan):
    """0 without a slot table; with one, 1 when the pool's byte offsets fit 32 bits (one
    descriptor), else 2 (each block's own 64-bit base)."""
    if not plan.slotted:
        return 0
    return 1 if inputs.k.numel() * inputs.k.element_size() <= POOL_LIMIT else 2


def _compiled(inputs, args, stream):
    # H, HK, BQ, GP, SLOTTED, SPLIT; the softmax scale (and the pool span) are arguments.
    key = (inputs.q.device, *args[14:16], *args[18:20], *args[22:24])
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            # Four waves and 64 KiB LDS per CTA: one wave per SIMD, 512 VGPRs.
            compiled = _COMPILED[key] = flyc.compile[{"waves_per_eu": 1}](
                _launch, *args[:21], 0, *args[22:]
            )
    return compiled


def run(*, inputs, plan, out):
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    with torch.cuda.device(inputs.q.device):
        _compiled(inputs, args, stream)(*args)


def launcher(*, inputs, plan, out):
    """launch(q, k, v, out, slots, scale_log2) with flat Q/out, unflattened K/V and the
    FP32 softmax_scale * log2(e); other arguments prebuilt. Union reads pool rows through
    K3's table, so the slot table is unused here.

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[5:]
    return lambda q, k, v, o, slots, scale_log2: compiled(q, k, v, o, scale_log2, *tail)
