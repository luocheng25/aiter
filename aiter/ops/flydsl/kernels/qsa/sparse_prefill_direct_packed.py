# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-call block-packed KV for QSA direct attention, packing included in the launch.

Four-token KV blocks are stored in the native QK/PV operand layouts. One query
wave then needs no K-data bpermutes or PV byte permutations. Original public
KV tensors remain unchanged; private scratch is refreshed on every call.

_DirectLoop is the key loop both direct kernels run; _Packed gives it the packed
layout, sparse_prefill_direct._Raw the raw K/V one. The packed launch runs a one-wave
CTA per work item (see _kernel): a (row, KV head) slot, or a key-tile part of one of
the last slots, merged by the part counting last. Plans of equal rows that fill a wave
per SIMD run persistent instead: one WG of PERSIST_WAVES one-row waves per CU takes the
same work items from a queue (see _persistent_kernel), with the same outputs.
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx

from . import sparse_prefill_primitives as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    QUEUE_WORDS,
    SC_SYSTEM,
    _as_f32,
    _as_i32,
    _behind_prio,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fma,
    _lds_load,
    _lds_relaxed_load,
    _lds_relaxed_store,
    _lds_store,
    _maximum,
    _min,
    _pack_bf16,
    _pv,
    _qk,
    _queue_exit,
    _queue_take,
    _read_q_g2r,
    _rescale,
    _stage_end,
    _tiles,
    _untile,
    _write_address,
)


def _enabled(row, rows, first, query_tiles, active, gated, nq):
    valid = row < rows
    if gated:
        # Padded waves read the tile's last row; valid already excludes them.
        tile = fx.Int32(query_tiles[first + _min(row, rows - 1)])
        valid = valid & (fx.Int32(active[tile]) == 0)
    return valid


def _output(
    o0,
    o1,
    inv,
    buffer,
    storage,
    tid,
    h,
    g,
    rows,
    extent,
    threads,
    first,
    query_tiles,
    active,
    gated,
    nq,
):
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        acc = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [acc[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = _pack_bf16(values)
            col = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(shared + ((row * 256 + col) ^ ((row & 7) * 8)) * 2)
            _write_address(address, words)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + part * threads * 8
        parts.append(
            prim.load_lds(storage, (element ^ (((element // 256) & 7) * 8)) * 2)
        )
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    prim.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * threads * 8
        r, col = element // 256, element % 256
        query, head = r // 16, r % 16
        enabled = _enabled(query, rows, first, query_tiles, active, gated, nq) & (
            head < g
        )
        offset = enabled.select(
            fx.Uint32(query * h * 256 + head * 256 + col), extent // 2
        )
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        fx.copy(
            fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16),
            fragment,
            fx.make_view(fx.get_iter(buffer) + offset, fx.make_layout(8, 1)),
        )


def _pack_source(sources, pair, packed, hk):
    """(block, head, valid, first context token) of a (block, KV head) pair."""
    block, head = pair // hk, pair % hk
    valid = pair < packed * hk
    return block, head, valid, fx.Int32(sources[_min(block, packed - 1)])


def _pack_loads(kr, vr, source, half, lane, hk, source_extent, slots, slotted):
    """The lane's 16 bytes of K and of V of a pair (token lane // 16, dims half * 128 +
    (lane % 16) * 8 ..)."""
    _, head, valid, start = source
    token, chunk = lane >> 4, lane & 15
    dim = half * 128 + chunk * 8
    if slotted:
        # A wave packs one block, four consecutive pool rows from the slot table's first:
        # give it its own 64-bit base, so the pool may exceed 4 GiB. kr/vr are the pool
        # addresses here.
        row = prim.uniform(fx.Int32(slots[prim.uniform(start)]))
        base = (fx.Int64(row) * hk + prim.uniform(head)) * 512
        kr = _buffer_at(kr + base, 4 * hk * 512)
        vr = _buffer_at(vr + base, 4 * hk * 512)
        offset = valid.select(
            fx.Uint32(token * hk * 512 + dim * 2), fx.Uint32(4 * hk * 512)
        )
    else:
        # Tokens past a request's end are masked by the direct kernel.
        offset = valid.select(
            fx.Uint32((start + token) * hk + head) * 512 + dim * 2, source_extent
        )
    return _buffer_words(kr, offset), _buffer_words(vr, offset)


def _pack_stores(pk, pv, source, data, half, lane, hk, extent):
    """Store a pair's K and V (_pack_loads) in the packed QK / PV operand layouts."""
    block, head, valid, _ = source
    key, vector = data
    token, chunk = lane >> 4, lane & 15
    dim = half * 128 + chunk * 8
    target = (
        fx.Uint32(
            ((block * 4 + dim // 64) * hk + head) * 256
            + (dim % 64 // 32) * 128
            + (dim % 32 // 8) * 32
            + token * 8
        )
        * 2
    )
    target = valid.select(target, extent)
    _buffer_store_words(key, pk, target)
    words = [vector[i] for i in range(4)]
    # Exchange the two token bits with the two register-index bits. Each wave
    # owns all four tokens, so the native DS operations need no CTA barrier.
    for bit in range(2):
        parity = (lane & (16 << bit)) != 0
        source_lane = fx.Int32((lane ^ (16 << bit)) * 4)
        next_words = list(words)
        for first in range(4):
            if first & (1 << bit):
                continue
            second = first | (1 << bit)
            kept = parity.select(words[second], words[first])
            sent = parity.select(words[first], words[second])
            received = prim.bpermute(source_lane, sent)
            next_words[first] = parity.select(received, kept)
            next_words[second] = parity.select(kept, received)
        words = next_words
    output = fx.Vector.from_elements(
        [
            (words[0] & 65535) | (words[1] << 16),
            (words[2] & 65535) | (words[3] << 16),
            ((words[0] >> 16) & 65535) | (words[1] & -65536),
            ((words[2] >> 16) & 65535) | (words[3] & -65536),
        ],
        fx.Int32,
    )
    rank = half * 512 + (lane >> 4) * 128 + chunk * 8
    target = valid.select(
        fx.Uint32((block * 4 + rank // 256) * hk + head) * 512 + rank % 256 * 2,
        extent,
    )
    _buffer_store_words(output, pv, target)


@flyc.jit
def _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, block):
    tid = fx.thread_idx.x
    lane, half = tid & 63, (tid >> 6) & 1
    first = block * 8 + (tid >> 7)
    source_extent = fx.Uint32(N) * HK * 512
    extent = fx.Uint32(PACKED) * 4 * HK * 512
    if fx.const_expr(SLOTTED):
        kr = fx.Int64(fx.ptrtoint(fx.get_iter(K)))
        vr = fx.Int64(fx.ptrtoint(fx.get_iter(V)))
    else:
        kr, vr = _buffer(K, source_extent), _buffer(V, source_extent)
    pk, pv = _buffer(PK, extent), _buffer(PV, extent)
    slots = fx.rocdl.make_buffer_tensor(SLOTS)
    # The wave's four pairs: every load before any store, so their latencies overlap (LLVM
    # keeps buffer loads after earlier buffer stores).
    pairs = [_pack_source(SOURCES, first + u * 2, PACKED, HK) for u in range(4)]
    data = [
        _pack_loads(kr, vr, pair, half, lane, HK, source_extent, slots, SLOTTED)
        for pair in pairs
    ]
    for pair, words in zip(pairs, data):
        _pack_stores(pk, pv, pair, words, half, lane, HK, extent)


@flyc.kernel(name="sparse_prefill_pack_kv_bf16_d256")
def _pack(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    SLOTS: fx.Tensor,
    COUNTS: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    CTAS: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[bool],
):
    if fx.const_expr(SPLIT):
        # A first CTA zeroes the direct launch's part counts.
        if fx.block_idx.x == 0:
            _zero_counts(COUNTS, CTAS)
        else:
            block = fx.block_idx.x - 1
            _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, block)
    else:
        _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, fx.block_idx.x)


@flyc.kernel(name="sparse_prefill_pack_kv_gated_bf16_d256")
def _pack_gated(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    ACTIVE: fx.Tensor,
    UNION_META: fx.Tensor,
    TILE_LIST: fx.Tensor,
    LISTED: fx.Tensor,
    COUNTS: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    TILES: fx.Int32,
    CTAS: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[bool],
):
    # A first CTA lists the direct (inactive) union tiles for the direct launch.
    if prim.uniform(DIRECT_FLAG[0]) != 0:
        if fx.block_idx.x == 0:
            _list_tiles(ACTIVE, UNION_META, TILE_LIST, LISTED, TILES)
            if fx.const_expr(SPLIT):
                _zero_counts(COUNTS, CTAS)
        else:
            block = fx.block_idx.x - 1
            _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, block)


def _key_offset(cached, n, visible, extent, hk, topk):
    complete = _min(visible >> 2, fx.Int32(topk))
    width = complete * 4 + (visible & 3)
    selector = fx.Int32((_min(n >> 2, fx.Int32(topk - 1)) & 63) * 4)
    block = prim.bpermute(selector, fx.Int32(cached))
    base = (n < complete * 4).select(
        fx.Uint32(block) * (hk * 2048), fx.Uint32(visible & -4) * (hk * 512)
    )
    return (n < width).select(base + (n & 3) * 16, extent)


def _key_load(resource, offset, lane, hk):
    base = offset + (lane >> 4) * 64
    words = []
    for part in range(8):
        # Invalid offset is already descriptor extent. Nonnegative stripe
        # offsets remain OOB; the host eligibility bound prevents wraparound.
        address = base + (part // 2) * hk * 512 + (part % 2) * 256
        piece = _buffer_words(resource, address)
        words.extend(piece[i] for i in range(4))
    return fx.Vector.from_elements(words, fx.Int32)


@flyc.jit
def _mask_value_tail(value, tile, segment, lane, visible, TOPK: fx.Constexpr[int]):
    value = fx.Vector(value)
    width = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    if tile * 32 + 32 > width:
        first = tile * 32 + segment * 16 + (lane >> 4) * 4
        mask0 = (first < width).select(fx.Int32(65535), fx.Int32(0)) | (
            first + 1 < width
        ).select(fx.Int32(-65536), fx.Int32(0))
        mask1 = (first + 2 < width).select(fx.Int32(65535), fx.Int32(0)) | (
            first + 3 < width
        ).select(fx.Int32(-65536), fx.Int32(0))
        value = fx.Vector.from_elements(
            [value[i] & (mask1 if i % 2 else mask0) for i in range(16)], fx.Int32
        )
    return value


def _value_load(resource, source_base, tile, segment, half, lane, visible, hk, topk):
    base = fx.Uint32(source_base) + half * 2 * hk * 512 + (lane & 15) * 16
    parts = [
        _buffer_words(resource, base + (part // 2) * hk * 512 + (part % 2) * 256)
        for part in range(4)
    ]
    value = fx.Vector.from_elements(
        [part[i] for part in parts for i in range(4)], fx.Int32
    )
    # A packed block may contain later tokens with NaNs; zero those BF16
    # elements before MFMA rather than relying on zero probabilities.
    return _mask_value_tail(value, tile, segment, lane, visible, topk)


def _reduce(value, maximum):
    a = fx.gpu.shuffle_xor(value, 16, 64)
    b = fx.gpu.shuffle_xor(value, 32, 64)
    c = fx.gpu.shuffle_xor(value, 48, 64)
    prim.sched_barrier(0)
    if maximum:
        return _maximum(_maximum(value, a), _maximum(b, c))
    return (value + a) + (b + c)


@flyc.jit
def _scale_scores(s0, s1, count, tile, lane, scale):
    first, second = fx.Vector(s0) * scale, fx.Vector(s1) * scale
    if tile * 32 + 32 > count:
        first = fx.Vector.from_elements(
            [
                (tile * 32 + (lane >> 4) * 4 + i < count).select(
                    first[i], fx.Float32(float("-inf"))
                )
                for i in range(4)
            ],
            fx.Float32,
        )
        second = fx.Vector.from_elements(
            [
                (tile * 32 + 16 + (lane >> 4) * 4 + i < count).select(
                    second[i], fx.Float32(float("-inf"))
                )
                for i in range(4)
            ],
            fx.Float32,
        )
    return first, second


class _DirectLoop:
    """The direct kernels' key loop: one query wave over its own selected keys in BN32
    tiles (two 16-key MFMA steps), online softmax per tile. A subclass gives the K/V
    layout (_Packed here, sparse_prefill_direct._Raw): where V rows come from, how K/V
    and block ids load, how scores are masked and reduced, how O rescales, and PV.

    Each tile waits for K(t) at its QK and leaves V(t) segment 0 in flight; K(t + 1)
    loads during PV. The waits count a subclass's loads: values() issues 4 VMEM
    requests (8 per segment), load_key() 8 (16 for both K halves).
    """

    @flyc.jit
    def attend(
        self, q, k0, k1, offset0, offset1, ids, maximum, total, o0, o1, tiles, start=0
    ):
        next_ids = ids
        lane, count = self.lane, self.count
        # The lane holding a four-key block's first K offset gives its V row: no second
        # dependent block-table load. A tile's rows shuffle under the previous tile's PV.
        row0 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset0))
        row1 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset1))
        for t in range(fx.Int32(start), tiles, fx.Int32(1)):
            self.pace(t)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            prim.sched_barrier(0)
            values0 = self.values(row0, t, 0, 0)
            values1 = self.values(row0, t, 0, 1)
            if ((t + 1) & 7) == 0:
                next_ids = self.next_ids(t)
            prim.sched_barrier(0)
            # The prior tile's sixteen K requests precede these eight V requests: wait
            # for K at its consumer, leaving V in flight for QK. A block-id prefetch
            # is younger still; vmcnt(8) then also waits for the oldest V request.
            fx.rocdl.s_waitcnt(vmcnt=8)
            prim.sched_barrier(0)
            scores = self.scores(_qk(q, *self.key_operands(k0, k1)), t)
            candidate = fx.Float32(_MAX_FLOOR)
            for score in scores:
                candidate = _maximum(candidate, score)
            candidate = self.reduce(candidate, True)
            updated = _maximum(maximum, candidate)
            alpha = _exp(maximum - updated)
            probabilities = fx.Vector.from_elements(
                [_exp(score - updated) for score in scores], fx.Float32
            )
            current = fx.Float32(0.0)
            for i in fx.range_constexpr(8):
                current = current + probabilities[i]
            current = self.reduce(current, False)
            total = total * alpha + current
            o0, o1 = self.rescale(o0, o1, alpha)
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            prim.sched_barrier(0)
            fx.rocdl.s_waitcnt(vmcnt=0)
            prim.sched_barrier(0)
            if ((t + 1) & 7) == 0:
                ids = next_ids
            following0 = _min((t + 1) * 32, (((count + 15) >> 4) - 1) * 16)
            following1 = _min((t + 1) * 32 + 16, (((count + 15) >> 4) - 1) * 16)
            offset0 = self.key_offset(ids, following0)
            offset1 = self.key_offset(ids, following1)
            p0 = _pack_bf16(
                fx.Vector.from_elements(
                    [probabilities[i] for i in range(4)], fx.Float32
                )
            ).bitcast(fx.Int16)
            p1 = _pack_bf16(
                fx.Vector.from_elements(
                    [probabilities[4 + i] for i in range(4)], fx.Float32
                )
            ).bitcast(fx.Int16)
            next_values0 = self.values(row1, t, 1, 0)
            next_values1 = self.values(row1, t, 1, 1)
            prim.sched_barrier(0)
            o0 = self.pv(p0, values0, o0)
            prim.sched_barrier(0)
            k0 = self.load_key(offset0)
            prim.sched_barrier(0)
            o1 = self.pv(p0, values1, o1)
            prim.sched_barrier(0)
            k1 = self.load_key(offset1)
            prim.sched_barrier(0)
            row0 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset0))
            row1 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset1))
            prim.sched_barrier(0)
            # Eight V loads precede sixteen K loads: consume V while K stays in flight.
            fx.rocdl.s_waitcnt(vmcnt=16)
            prim.sched_barrier(0)
            o0 = self.pv(p1, next_values0, o0)
            o1 = self.pv(p1, next_values1, o1)
            maximum = updated
        return maximum, total, o0, o1

    def pace(self, t):
        """Hook at the start of tile t (the persistent launch keeps sibling waves in step)."""


class _Packed(_DirectLoop):
    """Packed K/V: each four-token block stored in the QK/PV operand layouts."""

    # Lane 4j holds token 4j's K offset.
    row_lane = 16

    def __init__(
        self, kr, vr, blocks, row, visible, extent, count, lane, hk, topk, scale
    ):
        self.kr, self.vr, self.blocks, self.row = kr, vr, blocks, row
        self.visible, self.extent, self.count, self.lane = visible, extent, count, lane
        self.hk, self.topk, self.scale = hk, topk, scale

    def values(self, row, t, segment, half):
        return _value_load(
            self.vr, row, t, segment, half, self.lane, self.visible, self.hk, self.topk
        )

    def next_ids(self, t):
        chunk = _min((t + 1) >> 3, fx.Int32(self.topk // 64 - 1)) * 64
        return fx.Int32(self.blocks[self.row * self.topk + chunk + self.lane])

    def key_operands(self, k0, k1):
        return k0, k1

    def scores(self, s, t):
        s = fx.Vector(s)
        first, second = _scale_scores(
            fx.Vector.from_elements([s[i] for i in range(4)], fx.Float32),
            fx.Vector.from_elements([s[4 + i] for i in range(4)], fx.Float32),
            self.count,
            t,
            self.lane,
            self.scale,
        )
        return [first[i] for i in range(4)] + [second[i] for i in range(4)]

    def reduce(self, value, maximum):
        return _reduce(value, maximum)

    def rescale(self, o0, o1, alpha):
        changed = prim.ballot(alpha != fx.Float32(1.0))
        return _rescale((o0, o1), changed, lambda: alpha)

    def key_offset(self, ids, position):
        return _key_offset(
            ids,
            position + (self.lane & 15),
            self.visible,
            self.extent,
            self.hk,
            self.topk,
        )

    def load_key(self, offset):
        return _key_load(self.kr, offset, self.lane, self.hk)

    def pv(self, p, values, output):
        prim.s_setprio(2)
        acc = self.pv_tiles(p, values, output)
        prim.s_setprio(0)
        return _untile(acc)

    def pv_tiles(self, p, values, output):
        """PV's MFMA accumulator tiles (pv without its issue priority)."""
        values = fx.Vector(values)
        return _pv(
            p,
            lambda n: fx.Vector.from_elements(
                [values[n * 2 + i] for i in range(2)], fx.Int32
            ).bitcast(fx.Int16),
            _tiles(output),
        )


# Work items: the launch runs a one-wave CTA per (row, KV head) slot, gated of the inactive
# union tiles the pack launch lists in tile order (CTAs past them exit after the working
# ones), else of the call; the hardware dispatches them in slot order as CTAs finish. A last
# round of a few slots would leave most CTAs idle for a whole slot's time (1953 slots on
# 640 CTAs: a last round of 33), so unless batch_invariant the last slots split into
# key-tile parts (_kernel): each part stores its O, m and l (system scope: any XCD may
# merge) and counts on its slot; the part counting last merges the records in part order
# (the bits do not depend on which part merges) and writes the rows.
DIRECT_WAVES_PER_CU = 8  # resident CTAs per CU: two waves per SIMD, 8 KiB of LDS each
DIRECT_PARTS = 4


def split_extra(slots, ctas):
    """The parts past their slots' first that _kernel's split adds to `slots` work items with
    room for every part (an ungated launch's extra CTAs)."""
    rows = min(slots, slots % ctas + ctas // 16)
    if rows * 4 <= ctas:
        return rows * 3
    return slots if slots * 2 <= ctas else 0


def record_bytes(group):
    """A split part's record: O (group heads x 256 FP32), then m and l per head."""
    return group * ((256 + 2) * 4)


@flyc.jit
def _attend_part(ins, tile, hkv, part, split, shift, lane, pace=None):
    """One wave's online softmax over its row's key tiles (a split part: its share of
    the 2^shift parts). Returns (m, l, O registers, first row, rows, out extent). ins: the
    kernel's inputs and constants (_kernel, _persistent_kernel). pace: a persistent wave's
    (LDS word, sibling's LDS word, tiles before this row); its loop keeps in step
    (_PackedPaced) and it ends without the one-wave CTA's barrier."""
    Q, K, V, META = ins.Q, ins.K, ins.V, ins.META
    H, HK, NQ, TOPK = ins.H, ins.HK, ins.NQ, ins.TOPK
    first, rows = prim.uniform(META[tile * 5]), prim.uniform(META[tile * 5 + 1])
    k0, kv_len = prim.uniform(META[tile * 5 + 2]), prim.uniform(META[tile * 5 + 3])
    position0 = prim.uniform(META[tile * 5 + 4])
    valid_wave = rows > 0
    visible = position0 + 1
    count = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    tiles = valid_wave.select((count + 31) >> 5, fx.Int32(0))
    begin = split.select((part * tiles) >> shift, fx.Int32(0))
    end = split.select(((part + 1) * tiles) >> shift, tiles)
    qe = fx.Uint32(NQ - first) * H * 512 - hkv * (H // HK) * 512
    qptr = fx.get_iter(Q) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    qr = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * 256, 1)), qe)
    qoffset = (valid_wave & ((lane & 15) < H // HK)).select(
        fx.Uint32((lane & 15) * 512), qe
    )
    q = _read_q_g2r(qr, qoffset, lane)
    extent = (fx.Uint32(kv_len) * HK - hkv) * 512
    kr = _buffer(
        fx.make_view(
            fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(extent // 2, 1),
        ),
        extent,
    )
    vr = _buffer(
        fx.make_view(
            fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(extent // 2, 1),
        ),
        extent,
    )
    row = _min(first, fx.Int32(NQ - 1))
    blocks = fx.rocdl.make_buffer_tensor(ins.BLOCKS)
    maximum, total = fx.Float32(_MAX_FLOOR), fx.Float32(0.0)
    o0, o1 = prim.pin(fx.Vector.filled(32, 0.0, fx.Float32)), prim.pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    # A part starts in the 64-block id chunk holding its first tile.
    chunk = _min(begin >> 3, fx.Int32(TOPK // 64 - 1)) * 64
    cached = fx.Int32(blocks[row * TOPK + chunk + lane])
    fx.rocdl.s_waitcnt(vmcnt=0)
    offset0 = _key_offset(cached, begin * 32 + (lane & 15), visible, extent, HK, TOPK)
    offset0 = valid_wave.select(offset0, extent)
    kval0 = _key_load(kr, offset0, lane, HK)
    offset1 = _key_offset(
        cached,
        _min(begin * 32 + 16, (((count + 15) >> 4) - 1) * 16) + (lane & 15),
        visible,
        extent,
        HK,
        TOPK,
    )
    offset1 = valid_wave.select(offset1, extent)
    kval1 = _key_load(kr, offset1, lane, HK)
    # K(0) stays in flight with the first tile's V loads (its vmcnt(8) waits for it).
    prim.sched_barrier(0)
    path = _packed_path(
        pace,
        kr,
        vr,
        blocks,
        row,
        visible,
        extent,
        count,
        lane,
        HK,
        TOPK,
        ins.SCALE_LOG2,
    )
    maximum, total, o0, o1 = path.attend(
        q, kval0, kval1, offset0, offset1, cached, maximum, total, o0, o1, end, begin
    )
    if fx.const_expr(pace is None):
        fx.rocdl.s_waitcnt(vmcnt=0)
        _stage_end()
    return maximum, total, o0, o1, first, rows, qe


def _write_rows(ins, storage, o0, o1, total, first, rows, qe, hkv, lane):
    H, HK, NQ = ins.H, ins.HK, ins.NQ
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(ins.OUT) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = fx.rocdl.make_buffer_tensor(
        fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), num_records_bytes=qe
    )
    _output(
        o0,
        o1,
        inv,
        output,
        storage,
        lane,
        H,
        H // HK,
        rows,
        qe,
        64,
        first,
        None,
        None,
        False,
        NQ,
    )


def _count_part(COUNTS, index, lane, CTAS):
    """Count one part of split item `index` done: lane 0's buffer atomic add at the item's
    word, the other lanes out of range (one request, no divergent branch). Returns the parts
    counted before this one, wave-uniform."""
    extent = CTAS // 2 * 4
    counted = prim.buffer_atomic(
        "add",
        _buffer(COUNTS, extent),
        (lane == 0).select(index * 4, fx.Int32(extent)),
        fx.Int32(1),
    )
    return prim.readlane(counted, 0)


def _record_o(base, k, lane, group):
    # Coalesced: store k of the valid lanes (head < group) covers 4 * group * 16 bytes.
    return base + ((k * 4 + (lane >> 4)) * group + (lane & 15)) * 16


def _store_record(records, base, o0, o1, maximum, total, lane, group, extent):
    head = lane & 15
    for k in range(16):
        acc = fx.Vector(o1 if k >= 8 else o0)
        words = fx.Vector.from_elements(
            [_as_i32(acc[(k % 8) * 4 + i]) for i in range(4)], fx.Int32
        )
        _buffer_store_words(
            words,
            records,
            (head < group).select(_record_o(base, k, lane, group), extent),
            SC_SYSTEM,
        )
    _buffer_store_words(
        fx.Vector.from_elements([_as_i32(maximum), _as_i32(total)], fx.Int32),
        records,
        ((head < group) & (lane < 16)).select(base + group * 1024 + head * 8, extent),
        SC_SYSTEM,
    )


def _merge_records(
    records, first_record, part, parts, o0, o1, maximum, total, lane, group, extent
):
    """O and l of a split slot's parts in part order (the bits do not depend on which part
    merges): m = max m_p, w_p = 2^(m_p - m), O = sum O_p w_p, l = sum l_p w_p. Parts past
    `parts` add zeros (m = floor, l = O = 0: out-of-range loads); this part's own terms
    come from its registers. The other records load four register quads per round trip
    (m and l with the first), each drained by vmcnt(0): the direct loop's vmcnt(8) and
    vmcnt(16) stay the kernel's only partial waits."""
    head = lane & 15
    valid = head < group

    def other(p, offset):
        return (valid & (p < parts) & (part != p)).select(offset, extent)

    def quads(first):
        return [
            [
                _buffer_words(
                    records,
                    other(
                        p,
                        _record_o(
                            first_record + p * record_bytes(group), k, lane, group
                        ),
                    ),
                    4,
                    SC_SYSTEM,
                )
                for p in range(DIRECT_PARTS)
            ]
            for k in range(first, first + 4)
        ]

    words = [
        _buffer_words(
            records,
            other(p, first_record + p * record_bytes(group) + group * 1024 + head * 8),
            2,
            SC_SYSTEM,
        )
        for p in range(DIRECT_PARTS)
    ]
    pending = quads(0)
    prim.sched_barrier(0)
    fx.rocdl.s_waitcnt(vmcnt=0)
    prim.sched_barrier(0)
    stats = []
    for p in range(DIRECT_PARTS):
        mine = part == p
        floor = (p < parts).select(_as_f32(words[p][0]), fx.Float32(_MAX_FLOOR))
        stats.append(
            (mine.select(maximum, floor), mine.select(total, _as_f32(words[p][1])))
        )
    top = stats[0][0]
    for m, _ in stats[1:]:
        top = _maximum(top, m)
    weights = [_exp(m - top) for m, _ in stats]
    merged_l = stats[0][1] * weights[0]
    for p in range(1, DIRECT_PARTS):
        merged_l = _fma(stats[p][1], weights[p], merged_l)
    acc = [fx.Vector(o0), fx.Vector(o1)]
    values = []
    for first in range(0, 16, 4):
        current = pending
        if first > 0:
            current = quads(first)
            prim.sched_barrier(0)
            fx.rocdl.s_waitcnt(vmcnt=0)
        prim.sched_barrier(0)
        for k in range(first, first + 4):
            loads = current[k - first]
            for i in range(4):
                mine = fx.Float32(acc[k // 8][(k % 8) * 4 + i])
                terms = [
                    (part == p).select(mine, _as_f32(loads[p][i]))
                    for p in range(DIRECT_PARTS)
                ]
                value = terms[0] * weights[0]
                for p in range(1, DIRECT_PARTS):
                    value = _fma(terms[p], weights[p], value)
                values.append(value)
        prim.sched_barrier(0)
    return (
        fx.Vector.from_elements(values[:32], fx.Float32),
        fx.Vector.from_elements(values[32:], fx.Float32),
        merged_l,
    )


# The direct kernels' parameters the work-item helpers read, gathered once per kernel.
_INPUTS = (
    "Q",
    "K",
    "V",
    "OUT",
    "BLOCKS",
    "META",
    "TILE_LIST",
    "COUNTS",
    "RECORDS",
    "H",
    "HK",
    "NQ",
    "GATED",
    "SCALE_LOG2",
    "TOPK",
    "BQ",
    "CTAS",
    "SPLIT",
)


def _inputs(params):
    """The kernel's _INPUTS (params: its locals() on entry) as one namespace (ins)."""
    return SimpleNamespace(**{name: params[name] for name in _INPUTS})


@flyc.jit
def _finish(ins, write, o0, o1, maximum, total, split, part, shift, index, lane):
    """A work item's result (m, l and the O registers), written by write(o0, o1, l); a part
    of split item `index` (2^shift parts) instead stores its record and counts, and the
    part counting last merges every record in part order and writes the rows."""
    if fx.const_expr(ins.SPLIT):
        if split:
            group = ins.H // ins.HK
            extent = ins.CTAS * record_bytes(group)
            records = _buffer(ins.RECORDS, extent)
            first_record = (index << shift) * record_bytes(group)
            _store_record(
                records,
                first_record + part * record_bytes(group),
                o0,
                o1,
                maximum,
                total,
                lane,
                group,
                extent,
            )
            # Every record store is out before the part counts.
            fx.rocdl.s_waitcnt(vmcnt=0)
            done = _count_part(ins.COUNTS, index, lane, ins.CTAS)
            parts = fx.Int32(1) << shift
            if done == parts - 1:
                o0, o1, total = _merge_records(
                    records,
                    first_record,
                    part,
                    parts,
                    o0,
                    o1,
                    maximum,
                    total,
                    lane,
                    group,
                    extent,
                )
                write(o0, o1, total)
        else:
            write(o0, o1, total)
    else:
        write(o0, o1, total)


@flyc.jit
def _item(ins, storage, item, slots, split_rows, shift, listed):
    """Work item `item`: a whole (row, KV head) slot, or one of the 2^shift parts of one of
    the last split_rows slots. listed (gated): (first row, rows) of its tile and the tile
    and slot in it as a whole slot."""
    HK = ins.HK
    lane = fx.thread_idx.x & 63
    full = slots - split_rows
    split = item >= full
    index = (item - full) >> shift
    slot = split.select(full + index, item)
    part = split.select((item - full) & ((fx.Int32(1) << shift) - 1), fx.Int32(0))
    if fx.const_expr(ins.GATED):
        first, count, k, rest = listed
        if split:
            per_tile = ins.BQ * HK
            k = slot // per_tile
            rest = slot - k * per_tile
            first = prim.uniform(ins.TILE_LIST[k * 2])
            count = prim.uniform(ins.TILE_LIST[k * 2 + 1])
        local, hkv = rest // HK, rest % HK
        valid = local < count
        row = first + local
    else:
        row, hkv, valid = slot // HK, slot % HK, slot >= 0
    if valid:
        maximum, total, o0, o1, first_row, rows, qe = _attend_part(
            ins, row, hkv, part, split, shift, lane
        )
        _finish(
            ins,
            lambda o0, o1, total: _write_rows(
                ins, storage, o0, o1, total, first_row, rows, qe, hkv, lane
            ),
            o0,
            o1,
            maximum,
            total,
            split,
            part,
            shift,
            index,
            lane,
        )


@flyc.kernel(name="sparse_prefill_direct_packed_bf16_d256")
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    TILE_LIST: fx.Tensor,
    LISTED: fx.Tensor,
    COUNTS: fx.Tensor,
    RECORDS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    BQ: fx.Int32,
    TASKS: fx.Int32,
    TILES: fx.Int32,
    EXTRA: fx.Int32,
    CTAS: fx.Constexpr[int],
    SPLIT: fx.Constexpr[bool],
):
    ins = _inputs(locals())
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 8192, 16])
        .peek()
        .view(fx.make_layout(8192, 1))
    )
    listed = None
    if fx.const_expr(GATED):
        # Grid (tile slot, listed tile): one round of loads, K3's direct flag (none: no
        # list, the pack launch skipped it), the listed tile count and this item's tile as
        # a whole slot.
        item = fx.Int32(fx.block_idx.y) * (BQ * HK) + fx.Int32(fx.block_idx.x)
        launched = TILES * (BQ * HK)
        entry = _min(fx.Int32(fx.block_idx.y), TILES - 1) * 2
        listed = (
            prim.uniform(TILE_LIST[entry]),
            prim.uniform(TILE_LIST[entry + 1]),
            fx.Int32(fx.block_idx.y),
            fx.Int32(fx.block_idx.x),
        )
        slots = (prim.uniform(DIRECT_FLAG[0]) != 0).select(
            prim.uniform(LISTED[0]) * (BQ * HK), fx.Int32(0)
        )
    else:
        item = fx.Int32(fx.block_idx.x)
        slots = TASKS
        launched = TASKS + EXTRA
    # Past the slots and the most parts a split adds (3/4 CTAS), exit before the split
    # rule (most gated CTAs).
    if item < slots + (3 * CTAS // 4 if SPLIT else 0):
        _work(ins, storage, item, slots, launched, listed)


def _split_rule(slots, launched, CTAS):
    """(split_rows, shift): the last split_rows of `slots` work items split in 2^shift
    key-tile parts. The slots past the last whole round of CTAS, and CTAS / 16 more, split
    in four parts when those fit a quarter round (records and counts fit CTAS parts); in
    two when all the slots fit half a round (split_extra). A longer last round (say 276 of
    640) loses less than the parts' setup and records cost. The extra parts take
    the `launched` CTAs past the slots (gated: those of the active tiles' slots; ungated:
    split_extra's), as many slots as they cover."""
    rows = _min(slots, slots - slots // CTAS * CTAS + (CTAS >> 4))
    shift = (rows * 4 <= CTAS).select(
        fx.Int32(2), (slots * 2 <= CTAS).select(fx.Int32(1), fx.Int32(0))
    )
    spare = launched - slots
    rows = _min(rows, (shift == 2).select(spare // 3, spare))
    return ((shift > 0) & (rows > 0)).select(rows, fx.Int32(0)), shift


@flyc.jit
def _work(ins, storage, item, slots, launched, listed):
    split_rows, shift = fx.Int32(0), fx.Int32(0)
    if fx.const_expr(ins.SPLIT):
        split_rows, shift = _split_rule(slots, launched, ins.CTAS)
    if item < slots + split_rows * ((fx.Int32(1) << shift) - 1):
        _item(ins, storage, item, slots, split_rows, shift, listed)


# Persistent launch: plans of equal rows (every row selects topk complete blocks), at least a
# wave per SIMD, run one WG of PERSIST_WAVES one-row waves per CU (waves w and w ^ 4 share a
# SIMD). Rows started together sweep their sorted blocks together, so the K/V one wave reads
# stays in the L2 for the others. The older wave of a SIMD issues first, which would pull the
# two apart: each tile publishes the wave's tile count in LDS, and a wave more than
# PRIO_MARGIN tiles behind its sibling runs at s_setprio(PRIO_BEHIND) (PV then keeps that
# priority; see common's persistent launch section). Later items come from the launch queue
# while they fill whole rounds, then the last round's spread one per SIMD, on every CU, its
# split items' parts included (the one-wave launch's split: one part per wave); the last
# wave out zeroes the queue for the next launch (graph replays included). Every row and
# part runs the one-wave CTA's code: outputs are unchanged.
PERSIST_WAVES = DIRECT_WAVES_PER_CU
PRIO_BEHIND = 1
PRIO_MARGIN = 0


class _PackedPaced(_Packed):
    """_Packed of a persistent wave: each tile keeps it in step with its SIMD sibling, and
    PV keeps the pace's priority."""

    def __init__(self, *args, pace):
        super().__init__(*args)
        self.mine, self.sibling, self.base = pace

    def pace(self, t):
        count = self.base + t
        _behind_prio(
            count,
            prim.uniform(_lds_relaxed_load(self.sibling)),
            PRIO_MARGIN,
            PRIO_BEHIND,
        )
        _lds_relaxed_store(self.mine, count + 1)

    def pv(self, p, values, output):
        return _untile(self.pv_tiles(p, values, output))


def _packed_path(pace, *args):
    return _Packed(*args) if pace is None else _PackedPaced(*args, pace=pace)


def _store_rows(OUT, o0, o1, total, first, rows, qe, hkv, lane, H, HK, NQ):
    """_write_rows without LDS: each lane stores its eight 16-byte runs of BF16 O / l (the
    persistent WG's waves share no barrier)."""
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(OUT) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = _buffer(fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), qe)
    head = lane & 15
    valid = (head < H // HK) & (rows > 0)
    for half in range(2):
        acc = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [acc[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            col = (((lane >> 4) & 3) * 4 + i) * 8 + half * 128
            offset = fx.Uint32((head * 256 + col) * 2)
            _buffer_store_words(_pack_bf16(values), output, valid.select(offset, qe))


def _row_tiles(META, row, TOPK):
    rows, position0 = prim.uniform(META[row * 5 + 1]), prim.uniform(META[row * 5 + 4])
    visible = position0 + 1
    count = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    return (rows > 0).select((count + 31) >> 5, fx.Int32(0))


@flyc.jit
def _persistent_row(ins, item, part, shift, index, lane, pace):
    """Work item `item` (_kernel's slots) of a persistent wave, or (shift > 0) its key-tile
    part `part` of 2^shift, the parts of item `index` of the launch counting on
    COUNTS[index] and merging as _item's; returns its key tiles (0: a slot past its tile's
    rows). pace: (LDS word, sibling's LDS word, tiles before this item)."""
    H, HK, NQ = ins.H, ins.HK, ins.NQ
    if fx.const_expr(ins.GATED):
        k = item // (ins.BQ * HK)
        rest = item - k * (ins.BQ * HK)
        local, hkv = rest // HK, rest % HK
        valid = local < prim.uniform(ins.TILE_LIST[k * 2 + 1])
        row = prim.uniform(ins.TILE_LIST[k * 2]) + local
    else:
        row, hkv, valid = item // HK, item % HK, item >= 0
    tiles = fx.Int32(0)
    if valid:
        split = shift > 0
        row_tiles = _row_tiles(ins.META, row, ins.TOPK)
        begin = (part * row_tiles) >> shift
        mine, sibling, done = pace
        maximum, total, o0, o1, first, rows, qe = _attend_part(
            ins, row, hkv, part, split, shift, lane, (mine, sibling, done - begin)
        )
        _finish(
            ins,
            lambda o0, o1, total: _store_rows(
                ins.OUT, o0, o1, total, first, rows, qe, hkv, lane, H, HK, NQ
            ),
            o0,
            o1,
            maximum,
            total,
            split,
            part,
            shift,
            index,
            lane,
        )
        tiles = (((part + 1) * row_tiles) >> shift) - begin
    return tiles


def _persistent_slots(GATED, DIRECT_FLAG, LISTED, BQ, HK, TASKS, TILES, EXTRA):
    """(work items, the CTAs _kernel would launch for them)."""
    if GATED:
        listed = prim.uniform(LISTED[0]) * (BQ * HK)
        slots = (prim.uniform(DIRECT_FLAG[0]) != 0).select(listed, fx.Int32(0))
        return slots, TILES * (BQ * HK)
    return TASKS, TASKS + EXTRA


@flyc.kernel(
    name="sparse_prefill_direct_packed_bf16_d256",
    known_block_size=[64 * PERSIST_WAVES, 1, 1],
)
def _persistent_kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    TILE_LIST: fx.Tensor,
    LISTED: fx.Tensor,
    QUEUE: fx.Tensor,
    COUNTS: fx.Tensor,
    RECORDS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    BQ: fx.Int32,
    TASKS: fx.Int32,
    TILES: fx.Int32,
    EXTRA: fx.Int32,
    CUS: fx.Int32,
    CTAS: fx.Constexpr[int],
    SPLIT: fx.Constexpr[bool],
):
    """Persistent launch (see PERSIST_WAVES): wave w of WG b starts on item b *
    PERSIST_WAVES + w; after each row lane 0 takes the queue's next index (buffer atomic,
    the other lanes out of range: no divergent branch)."""
    ins = _inputs(locals())
    counts = fx.SharedAllocator().allocate(fx.Array[fx.Int32, PERSIST_WAVES, 16]).peek()
    paces = fx.Int32(
        fx.ptrtoint(fx.get_iter(counts.view(fx.make_layout(PERSIST_WAVES, 1))))
    )
    lane = fx.Int32(fx.thread_idx.x) & 63
    wave = prim.uniform(fx.Int32(fx.thread_idx.x) >> 6)
    _lds_relaxed_store(paces + wave * 4, fx.Int32(0))
    fx.gpu.barrier()
    slots, launched = _persistent_slots(
        GATED, DIRECT_FLAG, LISTED, BQ, HK, TASKS, TILES, EXTRA
    )
    queue = _buffer(QUEUE, QUEUE_WORDS * 4)
    waves = CUS * PERSIST_WAVES
    # Unless batch_invariant, the last split_rows items split their key tiles in 2^shift
    # parts as _kernel's (_split_rule; at most CTAS == waves parts). The items before the
    # last round and the split items come first and from the queue; then wave w of WG b
    # runs late position w * CUS + b (one per SIMD, on every CU): the unsplit items left
    # of the last round, then part p of split item i at p * split_rows + i.
    split_rows, shift = fx.Int32(0), fx.Int32(0)
    if fx.const_expr(SPLIT):
        split_rows, shift = _split_rule(slots, launched, CTAS)
    full = slots - split_rows
    edge = _min(slots // waves * waves, full)
    position = wave * CUS + fx.Int32(fx.block_idx.x)
    spread = position - (full - edge)
    rows = (split_rows > 0).select(split_rows, fx.Int32(1))
    late = (spread < 0).select(
        edge + position,
        (spread < (split_rows << shift)).select(full + spread % rows, slots),
    )
    part = spread // rows
    first = fx.Int32(fx.block_idx.x) * PERSIST_WAVES + wave
    item = (first < edge).select(first, late)
    done = fx.Int32(0)
    while item < slots:
        pace = (paces + wave * 4, paces + (wave ^ 4) * 4, done)
        split = item >= full
        # The lane through a pin: the row's lane-derived offsets are not hoisted out of
        # the row loop, which would keep them live through every key loop.
        tiles = _persistent_row(
            ins,
            item,
            split.select(part, fx.Int32(0)),
            split.select(shift, fx.Int32(0)),
            item - full,
            prim.pin_i32(lane),
            pace,
        )
        done = done + tiles
        # Taken after the row (not live through its key loop).
        nxt = prim.readlane(_queue_take(queue, lane), 0) + waves
        item = (item >= edge).select(slots, (nxt < edge).select(nxt, late))
    _queue_exit(queue, lane, waves)


def _lanes_sum(value, bits):
    """(exclusive prefix over lower lanes, wave total) of a nonnegative value < 2^bits:
    one ballot per bit."""
    below, total = fx.Int32(0), fx.Int32(0)
    for b in range(bits):
        mask = prim.ballot(((value >> b) & 1) != 0)
        below = below + (prim.lanes_below(mask) << b)
        total = total + (fx.Int32(fx.ctpop(mask)) << b)
    return below, prim.uniform(total)


LIST_TILES = 8  # tiles per thread and pass of the tile list


@flyc.jit
def _list_tiles(ACTIVE, UNION_META, TILE_LIST, LISTED, TILES):
    """One 256-thread CTA: the first row and row count of every inactive (direct) union tile,
    in tile order, into TILE_LIST, and their number into LISTED[0]."""
    tid = fx.thread_idx.x
    wave = prim.uniform(tid >> 6)
    shared = fx.SharedAllocator().allocate(fx.Array[fx.Int32, 8, 16]).peek()
    address = fx.Int32(fx.ptrtoint(fx.get_iter(shared.view(fx.make_layout(8, 1)))))
    extent = fx.Uint32(TILES) * 8
    tile_list = _buffer(TILE_LIST, extent)
    base = fx.Int32(0)
    for start in range(fx.Int32(0), TILES, fx.Int32(256 * LIST_TILES)):
        first = start + tid * LIST_TILES
        # Every load of the pass before its scan (one round trip).
        tiles = [first + i for i in fx.range_constexpr(LIST_TILES)]
        flags = [fx.Int32(ACTIVE[_min(tile, TILES - 1)]) for tile in tiles]
        meta = [
            [
                fx.Int32(UNION_META[_min(tile, TILES - 1) * 5 + j])
                for j in fx.range_constexpr(2)
            ]
            for tile in tiles
        ]
        inactive = [(tile < TILES) & (flag == 0) for tile, flag in zip(tiles, flags)]
        count = fx.Int32(0)
        for direct in inactive:
            count = count + direct.select(fx.Int32(1), fx.Int32(0))
        below, total = _lanes_sum(count, 4)
        _lds_store(address + wave * 4, total)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        _stage_end()
        totals = [
            prim.uniform(_lds_load(address + w * 4)) for w in fx.range_constexpr(4)
        ]
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        _stage_end()
        position = base + below
        for w in fx.range_constexpr(3):
            position = position + (fx.Int32(w) < wave).select(totals[w], fx.Int32(0))
        for direct, words in zip(inactive, meta):
            _buffer_store_words(
                fx.Vector.from_elements(words, fx.Int32),
                tile_list,
                direct.select(fx.Uint32(position) * 8, extent),
            )
            position = position + direct.select(fx.Int32(1), fx.Int32(0))
        base = base + totals[0] + totals[1] + totals[2] + totals[3]
    _buffer_store_words(
        fx.Vector.from_elements([base], fx.Int32),
        _buffer(LISTED, 4),
        (tid == 0).select(fx.Int32(0), fx.Int32(4)),
    )


@flyc.jit
def _zero_counts(COUNTS, CTAS: fx.Constexpr[int]):
    """The split slots' part counts (at most CTAS / 2 slots: two parts or more fit a round)."""
    counts = _buffer(COUNTS, CTAS // 2 * 4)
    for index in fx.range_constexpr(0, CTAS // 2, 256):
        index = index + fx.thread_idx.x
        _buffer_store_words(
            fx.Vector.from_elements([fx.Int32(0)], fx.Int32), counts, index * 4
        )


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    UNION_META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    NK: fx.Int32,
    PACKED: fx.Int32,
    BQ: fx.Int32,
    TILES: fx.Int32,
    TASKS: fx.Int32,
    GATED: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    TILE_LIST: fx.Tensor,
    LISTED: fx.Tensor,
    COUNTS: fx.Tensor,
    RECORDS: fx.Tensor,
    EXTRA: fx.Int32,
    CTAS: fx.Constexpr[int],
    SPLIT: fx.Constexpr[bool],
    PERSIST: fx.Constexpr[bool],
    QUEUE: fx.Tensor,
    CUS: fx.Int32,
    stream: fx.Stream,
):
    if TASKS > 0:
        packs = (PACKED * HK + 7) // 8
        if fx.const_expr(GATED):
            _pack_gated(
                K,
                V,
                PK,
                PV,
                SOURCES,
                SLOTS,
                DIRECT_FLAG,
                ACTIVE,
                UNION_META,
                TILE_LIST,
                LISTED,
                COUNTS,
                NK,
                PACKED,
                TILES,
                CTAS,
                HK,
                SLOTTED,
                SPLIT,
            ).launch(grid=(packs + 1, 1, 1), block=(256, 1, 1), stream=stream)
        else:
            _pack(
                K,
                V,
                PK,
                PV,
                SOURCES,
                SLOTS,
                COUNTS,
                NK,
                PACKED,
                CTAS,
                HK,
                SLOTTED,
                SPLIT,
            ).launch(
                grid=(packs + (1 if SPLIT else 0), 1, 1),
                block=(256, 1, 1),
                stream=stream,
            )
        if fx.const_expr(PERSIST):
            _persistent_kernel(
                Q,
                PK,
                PV,
                OUT,
                BLOCKS,
                META,
                DIRECT_FLAG,
                TILE_LIST,
                LISTED,
                QUEUE,
                COUNTS,
                RECORDS,
                H,
                HK,
                NQ,
                GATED,
                SCALE_LOG2,
                TOPK,
                BQ,
                TASKS,
                TILES,
                EXTRA,
                CUS,
                CTAS,
                SPLIT,
                value_attrs=prim.kernel_attrs(),
            ).launch(grid=(CUS, 1, 1), block=(64 * PERSIST_WAVES, 1, 1), stream=stream)
        else:
            # One CTA per work item: the (row, KV head) slots of the listed tiles (gated: as
            # many CTAs as every tile's slots, those past the work exit after it) or of the
            # call, then the split slots' parts past their first.
            kernel = _kernel(
                Q,
                PK,
                PV,
                OUT,
                BLOCKS,
                META,
                DIRECT_FLAG,
                TILE_LIST,
                LISTED,
                COUNTS,
                RECORDS,
                H,
                HK,
                NQ,
                GATED,
                SCALE_LOG2,
                TOPK,
                BQ,
                TASKS,
                TILES,
                EXTRA,
                CTAS,
                SPLIT,
                value_attrs=prim.kernel_attrs(),
            )
            if fx.const_expr(GATED):
                kernel.launch(grid=(BQ * HK, TILES, 1), block=(64, 1, 1), stream=stream)
            else:
                kernel.launch(
                    grid=(TASKS + EXTRA, 1, 1), block=(64, 1, 1), stream=stream
                )


def launch_args(inputs, prepared, out, tasks, stream):
    # A raw plan compiles this path ahead of time with its public KV as stand-ins. The fifth
    # argument is the slot table (a stand-in without one), the sixth the FP32
    # softmax_scale * log2(e).
    packed_key = inputs.k if prepared.packed_key is None else prepared.packed_key
    packed_value = inputs.v if prepared.packed_value is None else prepared.packed_value
    sources = (
        prepared.metadata.view(-1)
        if prepared.pack_sources is None
        else prepared.pack_sources
    )
    slots = inputs.kv_slots if prepared.slotted else prepared.metadata.view(-1)
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        slots,
        inputs.scale_log2,
        packed_key,
        packed_value,
        sources,
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.union_meta.view(-1),
        prepared.direct_flag,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        inputs.k.shape[0],
        packed_key.shape[0] // 4,
        prepared.union_tile,
        prepared.union_tiles,
        tasks,
        prepared.gated,
        inputs.topk,
        prepared.slotted,
        *(
            prepared.metadata.view(-1) if tensor is None else tensor
            for tensor in (
                prepared.tile_list,
                prepared.listed,
                prepared.split_counts,
                prepared.records,
            )
        ),
        (
            split_extra(tasks, prepared.ctas)
            if prepared.split and not prepared.gated and prepared.packed_key is not None
            else 0
        ),
        prepared.ctas,
        prepared.split,
        prepared.persistent,
        prepared.metadata.view(-1) if prepared.queue is None else prepared.queue,
        prepared.ctas // DIRECT_WAVES_PER_CU,
        stream,
    )


def compile_launch(inputs, prepared, out, stream):
    return flyc.compile(_launch, *launch_args(inputs, prepared, out, 0, stream))
