# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA direct attention over FP8 (E4M3FNUZ) K/V: one query wave over its own
selected blocks, FP8 MFMAs.

While the pack fits the platform's budget (the rule of sparse_prefill_direct), each call
packs every request's K/V into four-token blocks in FP8 MFMA operand order, so the direct
kernel's loads are whole, coalesced operands (no LDS, no permutes); otherwise the direct
kernel reads the raw K/V (RAW, below). The pack launch's CTAs past the blocks quantize Q
(sparse_prefill_common.quantize_rows) for both FP8 kernels. With a union plan (the FP8 union
kernel reads the raw K/V) the pack runs only when K3 routes some tile to direct
(direct_flag[0]):
  PK[block][i * 256 + (g * 4 + token) * 16 + 0..15]
      = K[token][64i + 8g + 0..7] ++ K[token][64i + 32 + 8g + 0..7]
  PV[block][p * 256 + d * 16 + (m - 4p) * 4 + token] = V[token][16d + m]   (p = m >> 2)
Per-token scales are stored per block as one 64-byte record of k_scale, log2(v_scale) and
1 / v_scale (clamped as in the union kernel); tokens past a request's end read 0 (a slot
table's later pool rows too). A pack wave reads its block from the block's own 64-bit base,
so K/V (or a pool) may exceed 4 GiB; the pack is [blocks * HK, 1024] bytes (FlyDSL passes
tensor dims as int32) and below 4 GiB.

A wave runs one query row and KV head: its G <= 16 Q heads are the columns of 16x16x32 FP8
MFMAs. Its key list (complete blocks in block_indices order, then the causal tail block)
runs in 32-key chunks of eight blocks: QK(kg) for blocks 8t + 4kg + jj; S lane (head, g)
holds blocks 8t + g and 8t + 4 + g, which are also its PV K-dim, so P needs no exchange.
PV MFMA m writes dims 16d + m: a lane (head, g) ends with dims 64g .. 64g + 63. Softmax:
the FP8 lazy max (sparse_prefill_common._lazy_max: S'' = S c - m, a rescale only when some
S'' passes LAZY or a fresh row sees its first key, the row max then at S'' = P_BIAS),
uniform per row across its four lanes; c = q_scale * SCALE_LOG2 (times k_scale per tensor,
per key otherwise), SCALE_LOG2 the runtime FP32 softmax_scale * log2(e). Q8 and its row
scales come from the pack launch.

Loads: chunk t + 1's words issue during chunk t, spread over its QK (the first K half and
per-token scale words first, the other K half between QK's halves, V after QK); its block ids
come one chunk earlier by scalar loads, so the K/V loads stay in flight (vector loads
complete in order). The loop runs two chunks per iteration, alternating two register sets,
and stops before the last chunk, which runs apart without loads and alone masks keys past the
key list (every earlier chunk holds eight complete blocks); the last chunk's padding slots
read zeros without a memory request. Per token the 24 scale terms of a 16-lane row's blocks
(its g) take two dword loads per lane and reach their users by DPP row broadcasts.

Plain plans (no core/residual tiles) run PLAIN_WAVES rows per WG, one wave each, heaviest
rows first (fewer rows than two such WGs per CU: one row per WG, spread over the CUs):
waves of rows the union serves exit at once, and every wave of a launch whose tiles all run
in the union exits on K3's direct flag before any row load.

RAW: a wave first maps its key list's slots to rows in LDS (the slot table's pool row of a
block's first token, else the request row), which each chunk then reads per lane. A lane
loads 16-byte K quarters of its token's rows (lanes g and g ^ 1 swap the halves the other
one needs, ds_swizzle) and its V rows (transposed as the pack does), so the operands and
per-token scale terms (computed as the pack does) are the pack's, bit for bit. K/V of a
request without a slot table read zero past its end (and for its last chunk's padding
slots); through a slot table its last chunk's masked keys read zero, or for a pool past 4 GiB
(WIDE, 64-bit loads) the request's first row.

Core/residual union plans (prepare.CORE_RATIOS): the pack launch also runs one CTA per tile,
dispatched first: on a union tile K3b (prepare.residual_tile) moves the residual blocks of
each row of an active core tile to the front of its block_indices row. Those rows then run
here over their residual blocks only (no causal tail) and merge into the union output by
log-sum-exp (the union kernel writes each row's lse); a residual row copies its union output
to LDS when it starts. A residual count is known only then, so the same CTA lists the tile's
direct rows in chunk-count buckets, and the direct waves take the rows from that list,
heaviest first: the launch does not end on a long row, and a wave reads its row's parameters
in one entry.
"""

import math
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    MAX_ROW_PACKED_KV_BYTES,
    QROWS,
    SC_DEVICE,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fp8_word,
    _lazy_max,
    _lds_load,
    _lds_store,
    _maximum,
    _mfma_fp8,
    _min,
    _scaled_bf16,
    _uniform,
    _v_scale_terms,
    platform,
    quantize_rows,
)
from .sparse_prefill_direct import _global_words
from .sparse_prefill_direct_packed import _enabled
from .sparse_prefill_prepare import (
    RESIDUAL_WORDS,
    allocate_scratch,
    packed_block_starts,
    residual_tile,
    upload,
)

__all__ = ["launchers", "prepare", "run", "scratch_specs"]

D = 256
BLOCK_BYTES = 1024  # one packed four-token block of one KV head
CHUNK = 8  # blocks per 32-key chunk
# Byte base of a padding key slot: its loads (base + < 1 KiB, scale records at (base >> 10) * 64)
# lie past every pack (PK/PV below 4 GiB) and read zeros without a memory request.
PAD_BASE = -1024
# Core/residual plans run the direct rows heaviest first: the pack launch lists every row in
# one of LIST_BUCKETS chunk-count buckets (bucket 0 the longest rows), each split in
# LIST_PARTS parts by tile (the atomics of one counter serialize). Slot s = bucket *
# LIST_PARTS + part. Schedule scratch (Int32 words): word b the rows of buckets 0 .. b, word
# LIST_BUCKETS + s the rows of slots 0 .. s, word SCHED_WHOLE 1 when no listed tile is active
# (every row runs whole), then from word SCHED_HEADER list q * LIST_PARTS + part (q < 4) of
# `cap` entries, bucket 2q of the part filling it from the front and bucket 2q + 1 from the
# back, one 16-byte entry per row: row | residual << 31, its visible keys (residual: 4 x its
# residual blocks), its request's first packed block (RAW: K row or slot-table entry) and
# length.
LIST_BUCKETS = 8
LIST_PARTS = 8
LIST_SLOTS = LIST_BUCKETS * LIST_PARTS
SCHED_WHOLE = LIST_BUCKETS + LIST_SLOTS
SCHED_HEADER = SCHED_WHOLE + 4
# Persistent counters (zero between pack launches), each on its own 256-byte line: tiles
# listed per part, parts listed, then one per slot. A part's count adds CORE_COUNT per active
# tile, the parts' count CORE_COUNT per part with one (a tile count in the low half).
COUNTER_STRIDE = 64
COUNTER_SLOTS = LIST_PARTS + 1
CORE_COUNT = 1 << 16
# Plain plans' rows per WG (one wave each): four waves fill a CU's SIMDs at once, so a launch
# of equal rows refills a CU in a quarter of the WG dispatches, and a launch the union serves
# dispatches a quarter of the (empty) WGs.
PLAIN_WAVES = 4
# Residual merges of groups of at most COMPACT_GROUP heads run on 8 lanes per head (_merge_o).
COMPACT_GROUP = 8


def _transpose4(w):
    """w[t] holds bytes (d0..d3) of token t; returns o[d] holding bytes (t0..t3)."""

    def perm(hi, lo, selector):
        return fx.Int32(prim.perm_b32(hi, lo, selector))

    x_lo = perm(w[1], w[0], 0x05010400)
    x_hi = perm(w[1], w[0], 0x07030602)
    y_lo = perm(w[3], w[2], 0x05010400)
    y_hi = perm(w[3], w[2], 0x07030602)
    return [
        perm(y_lo, x_lo, 0x05040100),
        perm(y_lo, x_lo, 0x07060302),
        perm(y_hi, x_hi, 0x05040100),
        perm(y_hi, x_hi, 0x07060302),
    ]


# ---- Pack ----


def _pack_v(vr, pv, target, start, valid, head, lane, hk, oob):
    """V^T piece of lane p * 16 + d: dims 16d + 4p .. + 3 of the four tokens, transposed."""
    p, d = lane >> 4, lane & 15
    words = []
    for t in range(4):
        source = ((start + t) * hk + head) * D + 16 * d + 4 * p
        source = (fx.Int32(t) < valid).select(source, oob)
        words.append(_buffer_words(vr, source, 1)[0])
    _buffer_store_words(
        fx.Vector.from_elements(_transpose4(words), fx.Int32), pv, target
    )


def _pack_scales(KS, VS, PSC, start, valid, block, head, lane, inside, n, packed, hk):
    """Scale record of the block: k_scale, log2(v_scale), 1 / v_scale of its four tokens
    (lanes 0..3), 64 bytes."""
    scale_extent = fx.Uint32(n) * hk * 4
    ks, vs = _buffer(KS, scale_extent), _buffer(VS, scale_extent)
    t = lane & 3
    source = (t < valid).select(((start + t) * hk + head) * 4, fx.Int32(scale_extent))
    sk = _as_f32(_buffer_words(ks, source, 1)[0])
    lsv, rsv = _v_scale_terms(_as_f32(_buffer_words(vs, source, 1)[0]))
    record_extent = fx.Uint32(packed) * hk * 64
    record = (inside & (lane < 4)).select(
        (block * hk + head) * 64 + t * 4, fx.Int32(record_extent)
    )
    records = _buffer(PSC, record_extent)
    for part, value in enumerate((sk, lsv, rsv)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            records,
            record + 16 * part,
        )


@flyc.jit
def _pack_body(
    K,
    V,
    KS,
    VS,
    PK,
    PV,
    PSC,
    SOURCES,
    VALID,
    SLOTS,
    pair,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
):
    lane = fx.Int32(fx.thread_idx.x) & 63
    block, head = pair // HK, pair % HK
    inside = pair < PACKED * HK
    start = _uniform(fx.Int32(SOURCES[_min(block, PACKED - 1)]))
    valid = inside.select(
        _uniform(fx.Int32(VALID[_min(block, PACKED - 1)])), fx.Int32(0)
    )
    if fx.const_expr(SLOTTED):
        # A block occupies four consecutive pool rows from its first token's slot.
        start = _uniform(fx.Int32(SLOTS[start]))
    # Each wave packs one block: its own 64-bit base, so K/V (or a pool) may exceed 4 GiB.
    base = (fx.Int64(start) * HK + head) * D
    kr = _buffer_at(fx.Int64(fx.ptrtoint(fx.get_iter(K))) + base, 4 * HK * D)
    vr = _buffer_at(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + base, 4 * HK * D)
    oob = fx.Int32(4 * HK * D)
    out_extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    pk, pv = _buffer(PK, out_extent), _buffer(PV, out_extent)
    target = inside.select(
        (block * HK + head) * BLOCK_BYTES + lane * 16, fx.Int32(out_extent)
    )
    # K: piece (i, token, g) of lane i * 16 + token * 4 + g (reads stay contiguous), stored
    # token-minor so each 4-lane quad of a direct K load reads 64 contiguous bytes.
    i, token, g = lane >> 4, (lane >> 2) & 3, lane & 3
    row = (token < valid).select(token * HK * D, oob)
    lo = _buffer_words(kr, row + 64 * i + 8 * g, 2)
    hi = _buffer_words(kr, row + 64 * i + 32 + 8 * g, 2)
    k_target = inside.select(
        (block * HK + head) * BLOCK_BYTES + i * 256 + (g * 4 + token) * 16,
        fx.Int32(out_extent),
    )
    _buffer_store_words(
        fx.Vector.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int32),
        pk,
        k_target,
    )
    _pack_v(vr, pv, target, 0, valid, 0, lane, HK, oob)
    if fx.const_expr(PER_TOKEN):
        _pack_scales(
            KS,
            VS,
            PSC,
            start,
            valid,
            block,
            head,
            lane,
            inside,
            N,
            PACKED,
            HK,
        )


def _bucket_edges(topk):
    """Chunk counts bounding the list buckets, longest first: 3/4, 1/2, 3/8, 1/4, 3/16, 1/8
    and 1/16 of a whole row's chunks."""
    whole = (topk + CHUNK) // CHUNK
    return [max(1, whole * k // 16) for k in (12, 8, 6, 4, 3, 2, 1)]


@flyc.jit
def _list_tile(
    SCHED,
    COUNTERS,
    TILE_META,
    TILE_ACTIVE,
    RESIDUAL_ROWS,
    DIRECT_META,
    rows,
    tiles,
    cap,
    tile,
    TOPK,
):
    """List the tile's direct rows (all of an inactive tile; an active one's with residual
    blocks, after K3b counted them) in their chunk-count buckets; the last tile listed
    publishes the slot counts (and SCHED_WHOLE) and zeroes the counters for the next
    launch."""
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & 63
    part = tile & (LIST_PARTS - 1)
    first_row = _uniform(fx.Int32(TILE_META[tile * 5]))
    count_rows = _uniform(fx.Int32(TILE_META[tile * 5 + 1]))
    position = _uniform(fx.Int32(TILE_META[tile * 5 + 4]))
    active = _uniform(fx.Int32(TILE_ACTIVE[tile])) != 0
    # K3b's residual counts (stored by lane 0 of each wave) are in L2 before wave 0 reads them
    # back past its L1.
    rocdl.s_waitcnt(vmcnt=0)
    fx.gpu.barrier()
    if _uniform(tid >> 6) == 0:
        row = first_row + _min(lane, count_rows - 1)
        counts = _buffer(RESIDUAL_ROWS, fx.Uint32(rows) * 4)
        # An inactive tile's rows run whole (their counts are stale): no load.
        count_at = active.select(row * 4, fx.Int32(rows * 4))
        count = _buffer_words(counts, count_at, 1, SC_DEVICE)[0]
        listed = (lane < count_rows) & (~active | (count > 0))
        visible = active.select(count * 4, position + lane + 1)
        complete = _min(visible >> 2, fx.Int32(TOPK))
        blocks = complete + ((visible & 3) != 0).select(fx.Int32(1), fx.Int32(0))
        chunks = (blocks + CHUNK - 1) // CHUNK
        bucket = fx.Int32(0)
        for edge in _bucket_edges(TOPK):
            bucket = bucket + (chunks < edge).select(fx.Int32(1), fx.Int32(0))
        rank, size = fx.Int32(0), fx.Int32(0)
        for b in fx.range_constexpr(LIST_BUCKETS):
            mine = listed & (bucket == b)
            mask = prim.ballot(mine)
            rank = mine.select(prim.lanes_below(mask), rank)
            size = (lane == b).select(fx.Int32(fx.ctpop(mask)), size)
        # Lane b reserves the tile's rows of bucket b in its slot; each row takes its bucket's
        # base from that lane's register (v_readlane, whatever the lanes' order of execution).
        counters = fx.get_iter(COUNTERS)
        base = fx.Int32(0)
        if (lane < LIST_BUCKETS) & (size > 0):
            slot = COUNTER_SLOTS + lane * LIST_PARTS + part
            base = fx.atomic_add(
                counters + COUNTER_STRIDE * slot, size, syncscope="agent"
            )
        index = rank
        for b in fx.range_constexpr(LIST_BUCKETS):
            index = (bucket == b).select(prim.readlane(base, b) + rank, index)
        where = ((bucket & 1) == 1).select(cap - 1 - index, index)
        entry = ((bucket >> 1) * LIST_PARTS + part) * cap + where
        flag = active.select(fx.Int32(-(2**31)), fx.Int32(0))
        extent = (fx.Uint32(cap) * (LIST_SLOTS // 2) * 4 + SCHED_HEADER) * 4
        sched = _buffer(SCHED, extent)
        # The request's first packed block (RAW: K row or slot-table entry) and length.
        p_start = fx.Int32(DIRECT_META[row * 5 + 2])
        length = fx.Int32(DIRECT_META[row * 5 + 3])
        _buffer_store_words(
            fx.Vector.from_elements([row | flag, visible, p_start, length], fx.Int32),
            sched,
            listed.select((SCHED_HEADER + entry * 4) * 4, fx.Int32(extent)),
        )
        # The reservations are performed (their results arrived) before the tile counts. Lane 0
        # counts the tile (then the part), an active tile CORE_COUNT more; the wave reads its
        # result by v_readlane.
        rocdl.s_waitcnt(vmcnt=0)
        cores = active.select(fx.Int32(CORE_COUNT), fx.Int32(0))
        counted = fx.Int32(0)
        if lane == 0:
            counted = fx.atomic_add(
                counters + COUNTER_STRIDE * part, fx.Int32(1) + cores, syncscope="agent"
            )
        counted = prim.readlane(counted, 0)
        part_tiles = (tiles - part + LIST_PARTS - 1) // LIST_PARTS
        if (counted & (CORE_COUNT - 1)) == part_tiles - 1:
            # The part's last tile counts the part, CORE_COUNT more when one of its tiles is
            # active.
            any_core = ((counted & -CORE_COUNT) + cores) != 0
            cores = any_core.select(fx.Int32(CORE_COUNT), fx.Int32(0))
            counted = fx.Int32(0)
            if lane == 0:
                counted = fx.atomic_add(
                    counters + COUNTER_STRIDE * LIST_PARTS,
                    fx.Int32(1) + cores,
                    syncscope="agent",
                )
            counted = prim.readlane(counted, 0)
            parts = _min(tiles, fx.Int32(LIST_PARTS))
            if (counted & (CORE_COUNT - 1)) == parts - 1:
                # Every tile is listed: lane s takes slot s's row count (zeroing it) and
                # publishes the rows of slots 0 .. s; lane 0 whether no tile is active.
                cores = (counted & -CORE_COUNT) + cores
                total = fx.atomic_xchg(
                    counters + COUNTER_STRIDE * (COUNTER_SLOTS + lane),
                    fx.Int32(0),
                    syncscope="agent",
                )
                for offset in (1, 2, 4, 8, 16, 32):
                    other = prim.bpermute(((lane - offset) & 63) * 4, total)
                    total = (lane >= offset).select(total + other, total)
                words = fx.Vector.from_elements([total], fx.Int32)
                _buffer_store_words(words, sched, (LIST_BUCKETS + lane) * 4)
                last_part = (lane & (LIST_PARTS - 1)) == LIST_PARTS - 1
                bucket_end = last_part.select(
                    (lane // LIST_PARTS) * 4, fx.Int32(extent)
                )
                _buffer_store_words(words, sched, bucket_end)
                whole = (cores == 0).select(fx.Int32(1), fx.Int32(0))
                _buffer_store_words(
                    fx.Vector.from_elements([whole], fx.Int32),
                    sched,
                    (lane == 0).select(fx.Int32(SCHED_WHOLE * 4), fx.Int32(extent)),
                )
                if lane < COUNTER_SLOTS:
                    fx.atomic_xchg(
                        counters + COUNTER_STRIDE * lane, fx.Int32(0), syncscope="agent"
                    )


@flyc.kernel(name="sparse_prefill_pack_kv_fp8_d256", known_block_size=[256, 1, 1])
def _pack_kernel(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    Q: fx.Tensor,
    Q8: fx.Tensor,
    QS: fx.Tensor,
    BLOCK_ROWS: fx.Tensor,
    TILE_COUNTS: fx.Tensor,
    TILE_BLOCKS: fx.Tensor,
    TILE_META: fx.Tensor,
    TILE_ACTIVE: fx.Tensor,
    TILE_RESIDUAL: fx.Tensor,
    RESIDUAL_ROWS: fx.Tensor,
    SCHED: fx.Tensor,
    COUNTERS: fx.Tensor,
    DIRECT_META: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    ROWS: fx.Int32,
    CAPACITY: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    TILES: fx.Int32,
    LISTED: fx.Int32,
    DIRECT_ROWS: fx.Int32,
    CAP: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
):
    index = fx.Int32(fx.block_idx.x)
    pack_ctas = (PACKED * HK + 3) // 4
    quant_end = pack_ctas + (ROWS + QROWS - 1) // QROWS
    # The list CTAs (latency-bound; K3b first on union tiles) are dispatched first: the
    # bandwidth-bound K/V and Q CTAs overlap them instead of the launch ending on them.
    listing = LISTED if fx.const_expr(RESIDUAL) else fx.Int32(0)
    cta = (index < listing).select(index + quant_end, index - listing)
    if fx.const_expr(RESIDUAL):
        storage = (
            fx.SharedAllocator().allocate(fx.Array[fx.Int32, RESIDUAL_WORDS, 16]).peek()
        )
        bits = fx.Int32(
            fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(RESIDUAL_WORDS, 1))))
        )
    if cta >= quant_end:
        # One CTA per tile: on a union tile K3b moves its rows' residual blocks to the front
        # of their block_indices rows; then the tile's direct rows are listed by cost.
        if fx.const_expr(RESIDUAL):
            tile = cta - quant_end
            # K3 sets direct_flag[0] when some tile goes direct or keeps residual blocks:
            # otherwise no row is listed, and the direct launch exits on the flag (the
            # schedule and the counters stay as they are). Read before K3b's latency.
            listed = _uniform(fx.Int32(DIRECT_FLAG[0])) != 0
            if tile < TILES:
                residual_tile(
                    BLOCK_ROWS,
                    TILE_COUNTS,
                    TILE_BLOCKS,
                    TILE_META,
                    TILE_ACTIVE,
                    TILE_RESIDUAL,
                    RESIDUAL_ROWS,
                    CAPACITY,
                    MAX_BLOCKS,
                    tile,
                    bits,
                    TOPK,
                )
            if listed:
                _list_tile(
                    SCHED,
                    COUNTERS,
                    TILE_META,
                    TILE_ACTIVE,
                    RESIDUAL_ROWS,
                    DIRECT_META,
                    DIRECT_ROWS,
                    LISTED,
                    CAP,
                    tile,
                    TOPK,
                )
    elif cta >= pack_ctas:
        # CTAs past the blocks quantize Q for both FP8 kernels (one launch fewer).
        quantize_rows(Q, Q8, QS, ROWS, cta - pack_ctas)
    else:
        pair = cta * 4 + _uniform(fx.Int32(fx.thread_idx.x) >> 6)
        args = (K, V, KS, VS, PK, PV, PSC, SOURCES, VALID, SLOTS, pair)
        if fx.const_expr(GATED):
            # With a union plan only the direct kernel reads the pack: K3 sets
            # direct_flag[0] when some tile goes direct (or keeps residual blocks).
            if _uniform(fx.Int32(DIRECT_FLAG[0])) != 0:
                _pack_body(*args, N, PACKED, HK, PER_TOKEN, SLOTTED)
        else:
            _pack_body(*args, N, PACKED, HK, PER_TOKEN, SLOTTED)


@flyc.jit
def _pack_launch(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    Q: fx.Tensor,
    Q8: fx.Tensor,
    QS: fx.Tensor,
    BLOCK_ROWS: fx.Tensor,
    TILE_COUNTS: fx.Tensor,
    TILE_BLOCKS: fx.Tensor,
    TILE_META: fx.Tensor,
    TILE_ACTIVE: fx.Tensor,
    TILE_RESIDUAL: fx.Tensor,
    RESIDUAL_ROWS: fx.Tensor,
    SCHED: fx.Tensor,
    COUNTERS: fx.Tensor,
    DIRECT_META: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    ROWS: fx.Int32,
    CAPACITY: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    TILES: fx.Int32,
    LISTED: fx.Int32,
    DIRECT_ROWS: fx.Int32,
    CAP: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if ROWS > 0:
        _pack_kernel(
            K,
            V,
            KS,
            VS,
            PK,
            PV,
            PSC,
            SOURCES,
            VALID,
            SLOTS,
            DIRECT_FLAG,
            Q,
            Q8,
            QS,
            BLOCK_ROWS,
            TILE_COUNTS,
            TILE_BLOCKS,
            TILE_META,
            TILE_ACTIVE,
            TILE_RESIDUAL,
            RESIDUAL_ROWS,
            SCHED,
            COUNTERS,
            DIRECT_META,
            N,
            PACKED,
            ROWS,
            CAPACITY,
            MAX_BLOCKS,
            TILES,
            LISTED,
            DIRECT_ROWS,
            CAP,
            HK,
            PER_TOKEN,
            SLOTTED,
            GATED,
            RESIDUAL,
            TOPK,
            value_attrs=prim.kernel_attrs(),
        ).launch(
            grid=(
                (PACKED * HK + 3) // 4
                + (ROWS + QROWS - 1) // QROWS
                + (LISTED if RESIDUAL else 0),
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )


# ---- Direct ----


def _raw_env(env, K, V, KS, VS, SLOTS, length, nk, topk, slotted, wide):
    """RAW: env of the raw K/V (and per token their scales) from META's start, the request's
    first K row or slot-table entry: descriptors over the request's rows, or through the slot
    table over the pool's (WIDE: 64-bit K/V addresses) with the masked-key limits of
    _raw_sources (slot s, token r is visible while s < complete + (r < tail))."""
    lane, hk, hkv = env.lane, env.hk, env.hkv
    n, g = lane & 15, lane >> 4
    env.slotted, env.wide = slotted, wide
    env.table_words = topk // 64 + 1
    env.token = lane & 3
    # Row-heads before (row 0 of the pool, or the request's first row, KV head hkv) and from it.
    first = fx.Int64(hkv) if slotted else fx.Int64(env.p_start) * hk + hkv
    count = fx.Uint32(nk if slotted else length) * hk - hkv
    k_base = fx.Int64(fx.ptrtoint(fx.get_iter(K))) + first * D
    v_base = fx.Int64(fx.ptrtoint(fx.get_iter(V))) + first * D
    # K: lane g reads quarter g // 2 (g even) or 2 + g // 2 (g odd) of each 64 bytes of its
    # token's row, the 16 bytes of the pack's operand coming from lanes g and g ^ 1.
    quarter = (g >> 1) * 16 + (g & 1) * 32
    env.k_even = (g & 1) == 0
    if wide:
        env.kbase, env.vbase = k_base + fx.Int64(quarter), v_base + fx.Int64(16 * n)
        env.gk = env.gv = None
    else:
        env.gk, env.gv = _buffer_at(k_base, count * D), _buffer_at(v_base, count * D)
        env.oob = fx.Int32(count * D)
        env.k_lane, env.v_lane = (lane & 3) * (hk * D) + quarter, n * 16
    if env.per_token:
        layout = fx.make_layout(count, 1)
        env.gks = _buffer(fx.make_view(fx.get_iter(KS) + first, layout), count * 4)
        env.gvs = _buffer(fx.make_view(fx.get_iter(VS) + first, layout), count * 4)
        env.sc_oob = fx.Int32(count * 4)
        env.sc_r = n & 3
        env.sc_k, env.sc_log = (n % 12) < 4, ((n % 12) >> 2) == 1
        env.sc_up_log = (n & 7) < 4
    if slotted:
        env.gslots = _buffer(
            fx.make_view(fx.get_iter(SLOTS) + env.p_start, fx.make_layout(length, 1)),
            fx.Uint32(length) * 4,
        )

        def one(predicate):
            return predicate.select(fx.Int32(1), fx.Int32(0))

        env.k_limit = env.complete + one((lane & 3) < env.tail)
        env.v_limits = [env.complete + one(fx.Int32(r) < env.tail) for r in range(4)]
        if env.per_token:
            limit = env.complete + one(env.sc_r < env.tail)
            env.sc_limits = [
                limit - env.sc_hi.select(fx.Int32(4), fx.Int32(0)),
                limit - 4,
            ]
    if wide:
        env.fallback = _uniform(fx.Int32(_buffer_words(env.gslots, fx.Int32(0), 1)[0]))


def _row_table(env):
    """RAW: the rows of the key list's slots into the wave's LDS table (slot s at word s): the
    complete blocks in block_indices order, the causal tail block, then padding (block 0), each
    as its first token's row (the slot table's pool row, else the request row). Lane l maps
    slots l + 64j, all its loads in flight."""
    slots = [env.lane + 64 * j for j in range(env.table_words)]
    ids = [_buffer_words(env.gblocks, (env.ids_row + s) * 4, 1)[0] for s in slots]
    rows = []
    for s, listed in zip(slots, ids):
        block = (s < env.complete).select(
            fx.Int32(listed), (s == env.complete).select(env.tail_block, fx.Int32(0))
        )
        rows.append(block * 4)
    if env.slotted:
        # Request-wide: past the request's end (a masked tail block) reads row 0.
        rows = [_buffer_words(env.gslots, row * 4, 1)[0] for row in rows]
    for s, row in zip(slots, rows):
        _lds_store(env.table + s * 4, row)


def _chunk_ids(env, t):
    """Listed block ids of key-list slots 8t .. 8t + 7 by two wave-uniform scalar loads (SMEM,
    lgkmcnt): they do not wait behind the K/V loads in the in-order vector memory counter.
    Slots past the row's list read another row's ids or zero (unused). RAW: the rows of the
    lane's blocks 8t + jj, 8t + 4 + jj (K) and 8t + g, 8t + 4 + g (V) from LDS."""
    if env.raw:
        k_at = env.table + (t * CHUNK + env.jj) * 4
        v_at = env.table + (t * CHUNK + env.g) * 4
        return [fx.Int32(_lds_load(at)) for at in (k_at, k_at + 16, v_at, v_at + 16)]
    offset = env.ids_row + t * CHUNK
    words = [
        fx.rocdl.s_buffer_load(env.gblocks, offset + 4 * h, count=4) for h in range(2)
    ]
    return [_uniform(fx.Vector(w)[i]) for w in words for i in range(4)]


def _chunk_sources(env, ids, t):
    """Where chunk t's words come from: _chunk_bases, RAW _raw_sources."""
    if env.raw:
        return _raw_sources(env, ids, t)
    return _chunk_bases(env, ids, t)


@flyc.jit
def _chunk_bases(env, ids, t):
    """(K bases of blocks 8t + jj, 8t + 4 + jj; V/scale bases of blocks 8t + g, 8t + 4 + g):
    complete blocks in block_indices order, the causal tail block, then padding slots at
    PAD_BASE (masked; their loads read zeros without a memory request). Only a row's last
    chunk can hold the tail block or padding: a chunk of eight complete blocks takes the
    short path."""
    first = t * CHUNK
    full = [
        ((env.p_start + fx.Int32(ids[s])) * env.hk + env.hkv) * BLOCK_BYTES
        for s in range(CHUNK)
    ]
    b0, b1, b2, b3, b4, b5, b6, b7 = full
    if first + CHUNK > env.complete:
        bases = []
        for s in fx.range_constexpr(CHUNK):
            slot = first + s
            block = (slot < env.complete).select(fx.Int32(ids[s]), env.tail_block)
            base = ((env.p_start + block) * env.hk + env.hkv) * BLOCK_BYTES
            bases.append((slot < env.blocks).select(base, fx.Int32(PAD_BASE)))
        b0, b1, b2, b3, b4, b5, b6, b7 = bases
    bases = [b0, b1, b2, b3, b4, b5, b6, b7]

    def pick(sel, lo):
        low = (sel & 1) == 1
        a = low.select(bases[lo + 1], bases[lo])
        b = low.select(bases[lo + 3], bases[lo + 2])
        return ((sel & 2) == 2).select(b, a)

    return [pick(env.jj, 0), pick(env.jj, 4), pick(env.g, 0), pick(env.g, 4)]


def _raw_offsets(env, rows, t, masked):
    """_raw_sources' twelve values, masked (the row's last chunk) or not."""
    k_slot, v_slot = t * CHUNK + env.jj, t * CHUNK + env.g
    stride = env.hk * D
    if masked:
        if env.slotted:
            k_valid = [k_slot + 4 * kg < env.k_limit for kg in range(2)]
            v_valid = [
                v_slot + 4 * b < env.v_limits[tt] for b in range(2) for tt in range(4)
            ]
            if env.per_token:
                sc_valid = [v_slot < env.sc_limits[w] for w in range(2)]
        else:
            # Padding slots: past the key list.
            k_valid = [k_slot + 4 * kg < env.blocks for kg in range(2)]
            v_valid = [v_slot + 4 * b < env.blocks for b in range(2) for _ in range(4)]
            sc_valid = [
                v_slot + env.sc_hi.select(fx.Int32(4), fx.Int32(0)) < env.blocks,
                v_slot + 4 < env.blocks,
            ]
    if env.wide:
        k_rows = [rows[kg] + env.token for kg in range(2)]
        v_rows = [rows[2 + b] + tt for b in range(2) for tt in range(4)]
        if masked:
            k_rows = [k_valid[kg].select(k_rows[kg], env.fallback) for kg in range(2)]
            v_rows = [v_valid[i].select(v_rows[i], env.fallback) for i in range(8)]
        k = [env.kbase + fx.Int64(row) * stride for row in k_rows]
        v = [env.vbase + fx.Int64(row) * stride for row in v_rows]
    else:
        k = [rows[kg] * stride + env.k_lane for kg in range(2)]
        v = [rows[2 + b] * stride + env.v_lane for b in range(2) for _ in range(4)]
        if masked:
            k = [k_valid[kg].select(k[kg], env.oob) for kg in range(2)]
            v = [v_valid[i].select(v[i], env.oob) for i in range(8)]
        v = [v[i] + (i % 4) * stride for i in range(8)]
    scales = [fx.Int32(0), fx.Int32(0)]
    if env.per_token:
        # Rows of values n (block n >= 12) and 16 + n (block 1), token n % 4.
        scales = [
            (env.sc_hi.select(rows[3], rows[2]) + env.sc_r) * (env.hk * 4),
            (rows[3] + env.sc_r) * (env.hk * 4),
        ]
        if masked:
            scales = [sc_valid[w].select(scales[w], env.sc_oob) for w in range(2)]
    return k + v + scales


@flyc.jit
def _raw_sources(env, rows, t):
    """RAW: (K sources of the lane's token in blocks 8t + jj, 8t + 4 + jj; V sources of the
    four tokens of blocks 8t + g, 8t + 4 + g; scale offsets of _scale_loads) from the rows of
    _chunk_ids: byte offsets, WIDE 64-bit K/V addresses. Only a row's last chunk holds keys
    past its key list: there through a slot table masked keys read zero (WIDE: the request's
    first row), as pool rows past a request's end may hold anything, and otherwise padding
    slots read zero without a memory request (the request's own descriptors read zero past
    its end)."""
    first = t * CHUNK
    k0, k1, v0, v1, v2, v3, v4, v5, v6, v7, c0, c1 = _raw_offsets(env, rows, t, False)
    if first + CHUNK > env.complete:
        masked = _raw_offsets(env, rows, t, True)
        k0, k1, v0, v1, v2, v3, v4, v5, v6, v7, c0, c1 = masked
    return [k0, k1], [v0, v1, v2, v3, v4, v5, v6, v7], [c0, c1]


def _raw_load(env, source, buffer, offset=0):
    if env.wide:
        return _global_words(source + offset)
    return _buffer_words(buffer, source + offset)


def _k_loads(env, sources, kg):
    """K words of QK half kg of a chunk (blocks 8t + 4kg + jj): 4 b128 loads (RAW: quarters
    of the token's K row, _raw_env)."""
    if env.raw:
        return [_raw_load(env, sources[0][kg], env.gk, 64 * i) for i in range(4)]
    return [
        _buffer_words(env.gk, sources[kg] + 256 * i + env.k_lane, 4) for i in range(4)
    ]


def _v_loads(env, sources):
    """V words of a chunk: 8 b128 loads (RAW: dims 16n .. 16n + 15 of each token)."""
    if env.raw:
        return [_raw_load(env, source, env.gv) for source in sources[1]]
    return [
        _buffer_words(env.gv, sources[2 + b] + 256 * p + env.v_lane, 4)
        for b in range(2)
        for p in range(4)
    ]


def _scale_loads(env, sources, zero4):
    """Per token the chunk's scale words (_scale_words), else zeros."""
    if not env.per_token:
        return fx.Vector.from_elements([zero4[0], zero4[0]], fx.Int32)
    if env.raw:
        # k_scale and v_scale of value n, v_scale of value 16 + n.
        offsets = sources[2]
        scales = [
            _buffer_words(env.gks, offsets[0], 1)[0],
            _buffer_words(env.gvs, offsets[0], 1)[0],
            _buffer_words(env.gvs, offsets[1], 1)[0],
        ]
        return fx.Vector.from_elements(scales, fx.Int32)
    # The 24 scale terms of the row's blocks 8t + g, 8t + 4 + g (records: k_scale,
    # log2(v_scale), 1 / v_scale of four tokens) spread over its 16 lanes: two dword
    # loads instead of six 16-byte loads per lane. Bases are unsigned: past 2 GiB of
    # packed K/V an arithmetic shift would read zeros past the records.
    rec = [fx.Int32(fx.Uint32(sources[2 + b]) >> 10) * 64 for b in range(2)]
    scales = [
        _buffer_words(env.gsc, env.sc_hi.select(rec[1], rec[0]) + env.sc_lo, 1)[0],
        _buffer_words(env.gsc, rec[1] + env.sc_up, 1)[0],
    ]
    return fx.Vector.from_elements(scales, fx.Int32)


def _chunk_words(env, sources, zero4):
    """K, V words and scale words of a chunk at once (the row's first)."""
    k = _k_loads(env, sources, 0) + _k_loads(env, sources, 1)
    sc = _scale_loads(env, sources, zero4)
    return _flatten(k), _flatten(_v_loads(env, sources)), sc


def _scale_term(sc, b, field, r):
    """Per-token scale term (field 0 k_scale, 1 log2(v_scale), 2 1 / v_scale) of token r of
    the lane's block b: value 12b + 4 field + r of its 16-lane row (g), word 0 holding values
    0..15 and word 1 values 16..23 in lanes 0..7, read by a DPP row broadcast."""
    v = 12 * b + 4 * field + r
    return prim.row_broadcast(_as_f32(sc[v >> 4]), v & 15)


def _probabilities(vals, sc, per_token):
    """FP8 P' words of the lane's 8 keys and its row-sum term (P' / v_scale per token)."""
    p = [_exp(vals[i]) for i in range(8)]
    terms = (
        [p[i] * _scale_term(sc, i // 4, 2, i % 4) for i in range(8)] if per_token else p
    )
    total = terms[0]
    for x in terms[1:]:
        total = total + x
    return (_fp8_word(p[0:4]), _fp8_word(p[4:8])), total


def _pv16(o, v, pw):
    """O^T += V^T P for the 16 PV MFMAs (MFMA m writes dims 16d + m)."""
    o = fx.Vector(o)
    out = []
    for m in range(16):
        a = (v[m >> 2][m & 3], v[4 + (m >> 2)][m & 3])
        acc = fx.Vector.from_elements([o[4 * m + r] for r in range(4)], fx.Float32)
        acc = _mfma_fp8(a, pw, acc)
        out += [acc[r] for r in range(4)]
    return fx.Vector.from_elements(out, fx.Float32)


def _store_o(env, o, inv):
    """O / l as BF16: lane (head, g) holds dims 64g .. 64g + 63 (dim 64g + 16r + m in
    o[4m + r])."""
    words = []
    for r in range(4):
        values = [o[4 * m + r] for m in range(16)]
        for k in range(4):
            words += _scaled_bf16(values[4 * k : 4 * k + 4], inv)
    for s in range(8):
        _buffer_store_words(
            fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
            env.gout,
            env.out_lane + 16 * s,
        )


def _from_lower(own, lower):
    """own in lanes 0 .. 7 of each 16-lane row, lane i - 8's lower in lanes 8 .. 15 (one
    v_mov_b32_dpp: row_shr:8 into banks 2 and 3)."""
    return prim.dpp_banks(own, lower, 0x118, 0xC)


def _merge_o(env, o, inv, m_run, total):
    """Residual row: O_d = o * inv merged with the row's union output O_u (BF16, already
    normalized) by their log-sum-exps (log2 units), then stored like _store_o, a half (32
    dims) at a time. O_u and the LSE were loaded at the row's start (_union_words). A group
    of at most COMPACT_GROUP heads leaves lanes n >= 8 idle: lane n + 8 merges the upper
    half of lane n's head (one DPP move per value), so each lane merges one half."""
    lse_u = _as_f32(env.lse_word)
    lse_d = fx.Float32(m_run) + prim.log2(total)
    top = _maximum(lse_u, lse_d)
    w_u, w_d = _exp(lse_u - top), _exp(lse_d - top)
    norm = prim.rcp(w_u + w_d)
    w_u, w_d = w_u * norm, w_d * norm * fx.Float32(inv)
    # The LDS copy of O_u landed long ago; LLVM waits for it before the first read.
    rocdl.s_waitcnt(vmcnt=0)
    if env.group <= COMPACT_GROUP:
        w_u, w_d = _from_lower(w_u, w_u), _from_lower(w_d, w_d)
        halves = [
            (
                env.union_compact,
                env.out_compact,
                lambda m, r: _from_lower(o[4 * m + r], o[4 * m + r + 2]),
            )
        ]
    else:
        halves = [
            (
                env.union_lane + 64 * half,
                env.out_lane + 64 * half,
                lambda m, r, half=half: o[4 * m + r + 2 * half],
            )
            for half in range(2)
        ]
    for union_at, out_at, value in halves:
        union = [prim._load_lds(env.union, union_at + 16 * s) for s in range(4)]
        words = []
        for r in range(2):
            for k in range(4):
                values = []
                for j in range(4):
                    m = 4 * k + j
                    # Dim 16r + m of the half: BF16 half m & 1 of word 8r + m // 2.
                    w = 8 * r + m // 2
                    bits = fx.Int32(fx.Vector(union[w // 4])[w % 4])
                    bits = (bits & fx.Int32(-65536)) if m & 1 else (bits << 16)
                    values.append(
                        fx.Float32(fx.fma(value(m, r), w_d, _as_f32(bits) * w_u))
                    )
                words += _scaled_bf16(values, fx.Float32(1.0))
        for s in range(4):
            _buffer_store_words(
                fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
                env.gout,
                out_at + 16 * s,
            )


def _union_words(env, row, hkv, lane):
    """A residual row's union output O_u (its group heads, contiguous BF16 rows of 512 bytes)
    copied to the wave's LDS (LDS DMA, no registers held across the chunk loop) and the
    lane's LSE word, issued at the row's start; _merge_o reads them."""
    q_rows = fx.Uint32(env.nq) * env.heads
    first = (row * env.heads + hkv * env.group) * (D * 2)
    resource = rocdl.get_buffer_rsrc(env.gout)
    for j in range(2 * env.group):
        prim.buffer_load_lds(
            resource,
            fx.to_llvm_ptr(fx.get_iter(env.union) + j * 256),
            4,
            fx.Int32(first + j * 256 + lane * 4),
            fx.Uint32(0),
            0,
        )
    # out_lane is unsigned (Q may pass 2 GiB).
    lse_lane = fx.Int32(fx.Uint32(env.out_lane) >> 9) << 2
    return _buffer_words(_buffer(env.lse, q_rows * 4), lse_lane, 1)[0]


def _flatten(vectors, dtype=fx.Int32):
    return fx.Vector.from_elements([v[i] for v in vectors for i in range(4)], dtype)


def _split(vector, count):
    vector = fx.Vector(vector)
    return [
        fx.Vector.from_elements([vector[4 * j + i] for i in range(4)], vector.dtype)
        for j in range(count)
    ]


def _k_half(env, kw, kg, got=None):
    """The A operands of QK half kg (blocks 8t + 4kg + jj) from the chunk's K words. RAW: lanes
    g and g ^ 1 (16 apart) swap the half of their K quarters the other one needs (got:
    _k_swaps), giving the pack's K[64i + 8g ..] ++ K[64i + 32 + 8g ..]."""
    k = _split(kw, 8)[4 * kg : 4 * kg + 4]
    if not env.raw:
        return k
    even = env.k_even
    k = [fx.Vector(x) for x in k]
    return [
        fx.Vector.from_elements(
            [even.select(x[j], r[j]) for j in range(2)]
            + [even.select(r[j], x[2 + j]) for j in range(2)],
            fx.Int32,
        )
        for x, r in zip(k, got[4 * kg : 4 * kg + 4])
    ]


def _k_swaps(env, kw):
    """RAW: the K words lanes g ^ 1 send this lane (ds_swizzle), both QK halves' at once."""
    if not env.raw:
        return None
    even = env.k_even
    k = [fx.Vector(x) for x in _split(kw, 8)]
    sent = [[even.select(x[2 + j], x[j]) for j in range(2)] for x in k]
    return [[fx.Int32(fx.gpu.shuffle_xor(w, 16, 64)) for w in pair] for pair in sent]


def _v_operands(env, vw):
    """The V^T operands of a chunk from its V words. RAW: each block's V rows transpose as in
    _pack_v."""
    v = _split(vw, 8)
    if not env.raw:
        return v
    return [
        fx.Vector.from_elements(
            _transpose4([fx.Vector(v[4 * b + tt])[p] for tt in range(4)]), fx.Int32
        )
        for b in range(2)
        for p in range(4)
    ]


def _scale_words(env, scw):
    """The two words _scale_term reads: RAW per token, the record's k_scale, log2(v_scale) or
    1 / v_scale from the loaded scales, as _pack_scales computes them."""
    scw = fx.Vector(scw)
    if not (env.raw and env.per_token):
        return [scw[0], scw[1]]
    lsv, rsv = _v_scale_terms(_as_f32(scw[1]))
    lsv_up, rsv_up = _v_scale_terms(_as_f32(scw[2]))
    log = env.sc_log.select(_as_i32(lsv), _as_i32(rsv))
    return [
        env.sc_k.select(fx.Int32(scw[0]), log),
        env.sc_up_log.select(_as_i32(lsv_up), _as_i32(rsv_up)),
    ]


def _qk_half_fp8(env, k):
    """One QK half of a chunk: S of blocks 8t + 4kg + jj from its A operands k (_k_half),
    eight 16x16x32 FP8 MFMAs (sparse_prefill_common._qk is the BF16 direct kernels' step).
    """
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for c in range(8):
        w = k[c >> 1]
        q = env.q[c]
        acc = _mfma_fp8((w[2 * (c & 1)], w[2 * (c & 1) + 1]), (q[0], q[1]), acc)
    return acc


def _scores(env, s, sc, t, m_run, last):
    """S''(t) of the lane's 8 keys (blocks 8t + g, 8t + 4 + g; tokens 0..3) from QK's halves
    s, with the lane-local max and the lazy-max ballot; in the row's last chunk (last) keys
    past the key list are masked."""
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    neg = fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run)
    vals = []
    for b in range(2):
        for r in range(4):
            x = s[b][r]
            if env.per_token:
                x = x * _scale_term(sc, b, 0, r)
            x = fx.Float32(fx.fma(x, env.c_scale, neg))
            if env.per_token:
                x = x + _scale_term(sc, b, 1, r)
            vals.append(x)
    if last:
        vals = _key_mask(env, vals, t)
    local = vals[0]
    for x in vals[1:]:
        local = _maximum(local, x)
    trigger = (local > LAZY) | (fresh & (local > fx.Float32(-math.inf)))
    return fx.Vector.from_elements(vals, fx.Float32), local, prim.ballot(trigger)


@flyc.jit
def _key_mask(env, vals, t):
    """S'' of the lane's keys past the key list set to -inf; a chunk of eight complete blocks
    has none."""
    v0, v1, v2, v3, v4, v5, v6, v7 = vals
    partial = (t + 1) * CHUNK > env.complete
    if fx.const_expr(env.per_token):
        # Per token the last chunk masks unconditionally: the branch costs that variant
        # more registers than it saves.
        partial = fx.Boolean(True)
    if partial:
        masked = []
        for b in fx.range_constexpr(2):
            slot = t * CHUNK + 4 * b + env.g
            for r in fx.range_constexpr(4):
                keep = (slot < env.complete) | (
                    (slot == env.complete) & (fx.Int32(r) < env.tail)
                )
                masked.append(keep.select(vals[4 * b + r], fx.Float32(-math.inf)))
        v0, v1, v2, v3, v4, v5, v6, v7 = masked
    return [v0, v1, v2, v3, v4, v5, v6, v7]


def _row_max(local):
    """The row max of the lane-local maxima over the row's four lanes (g)."""

    def reduce():
        value = fx.Float32(local)
        value = _maximum(value, fx.gpu.shuffle_xor(value, 16, 64))
        return _maximum(value, fx.gpu.shuffle_xor(value, 32, 64))

    return reduce


@flyc.jit
def _direct_row(ins, row, hkv, p_start, visible, residual, on, length=None, table=None):
    """The wave's query row and KV head: residual (a core/residual row) merges into the
    union output; visible keys of the row (residual: 4 x its residual blocks). RAW: length
    the request's, table the wave's LDS row table (TOPK + 64 words)."""
    H, HK, TOPK, PER_TOKEN = ins.H, ins.HK, ins.TOPK, ins.PER_TOKEN
    group = H // HK
    lane = fx.Int32(fx.thread_idx.x) & 63
    n, g = lane & 15, lane >> 4
    complete = _min(visible >> 2, fx.Int32(TOPK))
    tail = visible & 3
    tail_block = visible >> 2
    blocks = complete + (tail != 0).select(fx.Int32(1), fx.Int32(0))
    chunks = on.select((blocks + CHUNK - 1) // CHUNK, fx.Int32(0))
    q_rows = fx.Uint32(ins.NQ) * H
    head_ok = on & (n < group)
    qrow = row * H + hkv * group + n
    extent = fx.Uint32(ins.PACKED) * HK * BLOCK_BYTES
    scale_extent = fx.Uint32(ins.PACKED) * HK * 64
    gq8 = _buffer(ins.Q8, q_rows * D)
    q_base = head_ok.select(qrow * D + 8 * g, fx.Int32(q_rows * D))
    qs_offset = head_ok.select(qrow * 4, fx.Int32(q_rows * 4))
    qs = _as_f32(_buffer_words(_buffer(ins.QS, q_rows * 4), qs_offset, 1)[0])
    c_scale = qs * ins.SCALE_LOG2
    v_scale = None
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(ins.KS[0])
        # Read with k_scale (a scalar load): past the loop's memory fences it would be a
        # vector load the epilogue waits for.
        v_scale = fx.Float32(ins.VS[0])
    env = SimpleNamespace(
        raw=ins.RAW,
        lane=lane,
        hkv=hkv,
        hk=HK,
        p_start=p_start,
        complete=complete,
        tail=tail,
        tail_block=tail_block,
        blocks=blocks,
        residual=residual,
        jj=(lane >> 2) & 3,
        g=g,
        per_token=PER_TOKEN,
        c_scale=c_scale,
        gblocks=_buffer(ins.BLOCKS, fx.Uint32(ins.ROWS) * TOPK * 4),
        # Int32 offset of the row's key list (row 0 for a disabled row, which runs no chunk).
        ids_row=on.select(row * TOPK, fx.Int32(0)),
        gk=_buffer(ins.PK, extent),
        gv=_buffer(ins.PV, extent),
        gsc=_buffer(ins.PSC, scale_extent),
        k_lane=(g * 4 + (lane & 3)) * 16,
        v_lane=n * 16,
        # Scale value n (word 0: block n >= 12, field (n % 12) // 4) and 16 + n (word 1,
        # lanes 0..7: block 1, field 1 + n // 4), token n % 4.
        sc_hi=n >= 12,
        sc_lo=((n % 12) >> 2) * 16 + (n & 3) * 4,
        sc_up=(1 + ((n & 7) >> 2)) * 16 + (n & 3) * 4,
        gout=_buffer(ins.OUT, q_rows * D * 2),
        out_lane=head_ok.select(qrow * (D * 2) + g * 128, fx.Int32(q_rows * D * 2)),
        lse=ins.LSE,
        nq=ins.NQ,
        heads=H,
        group=group,
    )
    if fx.const_expr(ins.RAW):
        env.table = table
        _raw_env(
            env,
            ins.PK,
            ins.PV,
            ins.KS,
            ins.VS,
            ins.SLOTS,
            length,
            ins.NK,
            TOPK,
            ins.SLOTTED,
            ins.WIDE,
        )
    if fx.const_expr(ins.RESIDUAL):
        # The row's union output O_u for the merge: group heads of 512 bytes. With at most
        # COMPACT_GROUP heads lane n merges dims 64g + 32 (n >> 3) .. + 31 of head n & 7.
        env.union = ins.union
        env.union_lane = n * 512 + g * 128
        head = n & 7
        env.union_compact = (
            _min(head, fx.Int32(group - 1)) * 512 + g * 128 + (n >> 3) * 64
        )
        env.out_compact = (on & (head < group)).select(
            (row * H + hkv * group + head) * (D * 2) + g * 128 + (n >> 3) * 64,
            fx.Int32(q_rows * D * 2),
        )

    @flyc.jit
    def run():
        # Q (B operand): head n, dims 32c + 8g .. + 7 for MFMA c.
        env.q = [_buffer_words(gq8, q_base + 32 * c, 2) for c in range(8)]
        zero4 = fx.Vector.filled(4, 0, fx.Int32)
        if fx.const_expr(ins.RAW):
            _row_table(env)
        ids0 = _chunk_ids(env, fx.Int32(0))
        first_k, first_v, first_sc = _chunk_words(
            env, _chunk_sources(env, ids0, fx.Int32(0)), zero4
        )
        if fx.const_expr(ins.RESIDUAL):
            lse_word = fx.Int32(0)
            if env.residual:
                lse_word = _union_words(env, row, hkv, lane)
            env.lse_word = lse_word
        state = (
            fx.Vector.filled(64, 0.0, fx.Float32),
            fx.Float32(_MAX_FLOOR),
            fx.Float32(0.0),
            first_k,
            first_v,
            first_sc,
            *_chunk_ids(env, fx.Int32(1)),
        )

        def softmax_pv(o, m_run, l_run, v, sc, scored):
            """(O, running max, running sum) after a chunk from its scores."""
            vals, local, ballot = scored
            (o,), l_run, vals, m_run = _lazy_max(
                (o,), l_run, vals, m_run, ballot, _row_max(local)
            )
            pw, total = _probabilities(fx.Vector(vals), sc, PER_TOKEN)
            return _pv16(o, v, pw), m_run, fx.Float32(l_run) + total

        @flyc.jit
        def step(o, m_run, l_run, kw, vw, scw, *rest):
            """Chunk t from its prefetched words. Chunk t + 1's loads (block ids from the
            previous step) issue spread over QK(t): its first K half and scale words (with
            chunk t + 2's id loads) first, its other K half between QK(t)'s halves, its V
            after QK(t). Issued at once, they queued in the CU's vector memory path and
            stalled the wave's issue longer."""
            *ids, t = rest
            t = fx.Int32(t)
            sources = _chunk_sources(env, ids, t + 1)
            k_next = _k_loads(env, sources, 0)
            sc_next = _scale_loads(env, sources, zero4)
            ids_next = _chunk_ids(env, t + 2)
            swaps = _k_swaps(env, kw)
            # The scheduling barriers keep each group of loads in its place among QK(t)'s
            # MFMAs, which wait for chunk t's K words.
            prim.sched_barrier(0)
            s0 = _qk_half_fp8(env, _k_half(env, kw, 0, swaps))
            prim.sched_barrier(0)
            k_next = k_next + _k_loads(env, sources, 1)
            prim.sched_barrier(0)
            s1 = _qk_half_fp8(env, _k_half(env, kw, 1, swaps))
            sc = _scale_words(env, scw)
            scored = _scores(env, (s0, s1), sc, t, fx.Float32(m_run), False)
            prim.sched_barrier(0)
            v_next = _v_loads(env, sources)
            # MachineSink cannot move the loads above the fence past the rescale branch into
            # the block that first uses their words (the next step's).
            prim.compiler_fence()
            prim.sched_barrier(0)
            return (
                *softmax_pv(o, m_run, l_run, _v_operands(env, vw), sc, scored),
                _flatten(k_next),
                _flatten(v_next),
                sc_next,
                *ids_next,
            )

        # Two chunks per iteration: chunk t + 1's words land in the registers chunk t - 1
        # used, so the loop carries them without copies (a copy would wait for the loads at
        # the loop end). The last chunk runs apart: it issues no loads (past the key list)
        # and alone masks keys (every earlier chunk holds eight complete blocks).
        for t in range(fx.Int32(0), chunks - 2, fx.Int32(2)):
            state = step(*state, t)
            state = step(*state, fx.Int32(t) + 1)
        if (chunks & 1) == 0:
            state = step(*state, chunks - 2)
        o, m_run, l_run, kw, vw, scw = state[:6]
        swaps = _k_swaps(env, kw)
        sc = _scale_words(env, scw)
        s0 = _qk_half_fp8(env, _k_half(env, kw, 0, swaps))
        s1 = _qk_half_fp8(env, _k_half(env, kw, 1, swaps))
        scored = _scores(env, (s0, s1), sc, chunks - 1, fx.Float32(m_run), True)
        state = softmax_pv(o, m_run, l_run, _v_operands(env, vw), sc, scored)
        o, l_run = fx.Vector(state[0]), fx.Float32(state[2])
        total = l_run + fx.gpu.shuffle_xor(l_run, 16, 64)
        total = total + fx.gpu.shuffle_xor(total, 32, 64)
        inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
        if fx.const_expr(not PER_TOKEN):
            inv = inv * v_scale
        if fx.const_expr(ins.RESIDUAL):
            if env.residual:
                _merge_o(env, [o[i] for i in range(64)], inv, state[1], total)
            else:
                _store_o(env, [o[i] for i in range(64)], inv)
        else:
            _store_o(env, [o[i] for i in range(64)], inv)

    if fx.const_expr(ins.listed):
        # A listed row runs at least one chunk.
        run()
    else:
        # Disabled waves (rows another kernel serves) exit without loads.
        if chunks > 0:
            run()


def _rows_before(sched, word):
    """Eight wave-uniform schedule words from word on (scalar loads)."""
    return [
        _uniform(fx.Vector(rocdl.s_buffer_load(sched, word + 4 * h, count=4))[i])
        for h in range(2)
        for i in range(4)
    ]


@flyc.jit
def _list_entry(sched, ends, j, cap):
    """The list entry (row | residual << 31, visible keys, p_start, length) at position j of
    the pack launch's cost order (ends[b]: the rows of buckets 0 .. b)."""
    bucket, first = fx.Int32(0), fx.Int32(0)
    for b in fx.range_constexpr(1, LIST_BUCKETS):
        later = j >= ends[b - 1]
        bucket = later.select(fx.Int32(b), bucket)
        first = later.select(ends[b - 1], first)
    slots = _rows_before(sched, LIST_BUCKETS + bucket * LIST_PARTS)
    part = fx.Int32(0)
    for p in fx.range_constexpr(1, LIST_PARTS):
        later = j >= slots[p - 1]
        part = later.select(fx.Int32(p), part)
        first = later.select(slots[p - 1], first)
    index = j - first
    where = ((bucket & 1) == 1).select(cap - 1 - index, index)
    entry = ((bucket >> 1) * LIST_PARTS + part) * cap + where
    words = fx.Vector(rocdl.s_buffer_load(sched, SCHED_HEADER + entry * 4, count=4))
    return [_uniform(words[i]) for i in range(4)]


def _lds_words(allocator, words):
    """Byte address of words Int32 words of LDS."""
    array = allocator.allocate(fx.Array[fx.Int32, words, 16]).peek()
    return fx.Int32(fx.ptrtoint(fx.get_iter(array.view(fx.make_layout(words, 1)))))


def _listed(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    BLOCKS: fx.Tensor,
    LSE: fx.Tensor,
    FLAG: fx.Tensor,
    SCHED: fx.Tensor,
    CAP: fx.Int32,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    NK: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    RAW: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    WIDE: fx.Constexpr[bool],
):
    """Core/residual plans: the pack launch's list of direct rows, heaviest first (a
    cost-sorted dispatch keeps the launch from ending on a long row). Work item i is list
    position i // HK, KV head i % HK; items past the listed rows exit, as every item does
    when K3 routes no row direct (FLAG[0] clear: the pack launch listed nothing)."""
    ins = SimpleNamespace(
        Q8=Q8,
        QS=QS,
        PK=PK,
        PV=PV,
        PSC=PSC,
        KS=KS,
        VS=VS,
        OUT=OUT,
        SLOTS=SLOTS,
        BLOCKS=BLOCKS,
        LSE=LSE,
        H=H,
        HK=HK,
        NQ=NQ,
        ROWS=ROWS,
        PACKED=PACKED,
        NK=NK,
        SCALE_LOG2=SCALE_LOG2,
        TOPK=TOPK,
        PER_TOKEN=PER_TOKEN,
        RESIDUAL=True,
        RAW=RAW,
        SLOTTED=SLOTTED,
        WIDE=WIDE,
        listed=True,
    )
    group = H // HK
    allocator = fx.SharedAllocator()
    # The row's O_u (_union_words, group heads of 512 bytes).
    ins.union = (
        allocator.allocate(fx.Array[fx.Int8, group * 512, 16])
        .peek()
        .view(fx.make_layout(group * 512, 1))
    )
    table = _lds_words(allocator, TOPK + 64) if fx.const_expr(RAW) else None
    item = fx.Int32(fx.block_idx.x)
    sched = _buffer(SCHED, (fx.Uint32(CAP) * (LIST_SLOTS // 2) * 4 + SCHED_HEADER) * 4)
    ends = _rows_before(sched, fx.Int32(0))
    listed = (_uniform(fx.Int32(FLAG[0])) != 0).select(
        ends[LIST_BUCKETS - 1] * HK, fx.Int32(0)
    )
    if item < listed:
        word, visible, p_start, length = _list_entry(sched, ends, item // HK, CAP)
        row, residual = word & 0x7FFFFFFF, word < 0
        _direct_row(
            ins,
            row,
            item % HK,
            p_start,
            visible,
            residual,
            fx.Boolean(True),
            length,
            table,
        )


_direct_kernel = flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64, 1, 1]
)(_listed)


def _plain(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    FLAG: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    NK: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    RAW: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    WIDE: fx.Constexpr[bool],
    WAVES: fx.Constexpr[int],
):
    """Plans without core/residual tiles: WAVES rows per WG, one wave each."""
    ins = SimpleNamespace(
        Q8=Q8,
        QS=QS,
        PK=PK,
        PV=PV,
        PSC=PSC,
        KS=KS,
        VS=VS,
        OUT=OUT,
        SLOTS=SLOTS,
        BLOCKS=BLOCKS,
        LSE=OUT,
        H=H,
        HK=HK,
        NQ=NQ,
        ROWS=ROWS,
        PACKED=PACKED,
        NK=NK,
        SCALE_LOG2=SCALE_LOG2,
        TOPK=TOPK,
        PER_TOKEN=PER_TOKEN,
        RESIDUAL=False,
        RAW=RAW,
        SLOTTED=SLOTTED,
        WIDE=WIDE,
        listed=False,
    )
    wave = _uniform(fx.Int32(fx.thread_idx.x) >> 6)
    table = None
    if fx.const_expr(RAW):
        tables = _lds_words(fx.SharedAllocator(), WAVES * (TOPK + 64))
        table = tables + wave * ((TOPK + 64) * 4)
    # Rows run in reverse: later rows list the most blocks (causal-prefix rows the fewest),
    # so the heaviest waves start first and the light or empty ones fill the tail.
    work = ROWS * HK - 1 - (fx.Int32(fx.block_idx.x) * WAVES + wave)
    if fx.const_expr(GATED):
        # K3 raises flag[0] when some tile goes direct: otherwise every wave exits here.
        if _uniform(fx.Int32(FLAG[0])) != 0:
            _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED, table)
    else:
        _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED, table)


_plain_kernel = flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64 * PLAIN_WAVES, 1, 1]
)(_plain)
# Fewer rows than two PLAIN_WAVES WGs per CU: one row per WG, so that they spread over the CUs
# (PLAIN_WAVES rows per WG would leave most CUs idle).
_plain_spread_kernel = flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64, 1, 1]
)(_plain)


def _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED, table):
    """Work item `work` of a plain launch: query row work // HK, KV head work % HK (the last
    WG's spare waves, work < 0, exit). A plain plan's metadata row i is query row i, so the
    union test does not wait for META."""
    inside = work >= 0
    work = inside.select(work, fx.Int32(0))
    row, hkv = work // ins.HK, work % ins.HK
    p_start, position = _uniform(META[row * 5 + 2]), _uniform(META[row * 5 + 4])
    length = _uniform(META[row * 5 + 3]) if fx.const_expr(ins.RAW) else None
    on = _enabled(fx.Int32(0), fx.Int32(1), row, QUERY_TILES, ACTIVE, GATED, ins.NQ)
    _direct_row(
        ins,
        row,
        hkv,
        p_start,
        position + 1,
        fx.Boolean(False),
        on & inside,
        length,
        table,
    )


@flyc.jit
def _direct_launch(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    LSE: fx.Tensor,
    FLAG: fx.Tensor,
    SCHED: fx.Tensor,
    CAP: fx.Int32,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    NK: fx.Int32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    RAW: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    WIDE: fx.Constexpr[bool],
    WAVES: fx.Constexpr[int],
    GRID: fx.Int32,
    stream: fx.Stream,
):
    # FLAG: K3's direct flag; SCHED: the pack launch's schedule (core/residual plans).
    if GRID > 0:
        if fx.const_expr(RESIDUAL):
            _direct_kernel(
                Q8,
                QS,
                PK,
                PV,
                PSC,
                KS,
                VS,
                OUT,
                SLOTS,
                BLOCKS,
                LSE,
                FLAG,
                SCHED,
                CAP,
                H,
                HK,
                NQ,
                ROWS,
                PACKED,
                NK,
                SCALE_LOG2,
                TOPK,
                PER_TOKEN,
                RAW,
                SLOTTED,
                WIDE,
                value_attrs=prim.kernel_attrs(),
            ).launch(grid=(GRID, 1, 1), block=(64, 1, 1), stream=stream)
        else:
            (_plain_kernel if WAVES == PLAIN_WAVES else _plain_spread_kernel)(
                Q8,
                QS,
                PK,
                PV,
                PSC,
                KS,
                VS,
                OUT,
                SLOTS,
                BLOCKS,
                META,
                ACTIVE,
                QUERY_TILES,
                FLAG,
                H,
                HK,
                NQ,
                ROWS,
                PACKED,
                NK,
                SCALE_LOG2,
                TOPK,
                GATED,
                PER_TOKEN,
                RAW,
                SLOTTED,
                WIDE,
                WAVES,
                value_attrs=prim.kernel_attrs(),
            ).launch(
                grid=((GRID + WAVES - 1) // WAVES, 1, 1),
                block=(64 * WAVES, 1, 1),
                stream=stream,
            )


# ---- Host ----


class DirectPlan:
    """FP8 direct plan: one wave per (query row, KV head) over packed or raw K/V."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def scratch_specs(prepared):
    # [blocks * HK, 1024]: FlyDSL passes int32 dims and a packed context may exceed 2**31 bytes.
    rows = prepared.packed_blocks * prepared.hk
    specs = []
    if not prepared.raw:
        specs.append(("packed_key", (rows, BLOCK_BYTES), torch.uint8))
    if rows:
        specs.append(("packed_value", (rows, BLOCK_BYTES), torch.uint8))
    if prepared.per_token and not prepared.raw:
        specs.append(("scale_records", (rows * 16,), torch.float32))
    if prepared.residual:
        # The direct rows' cost order (written by the pack launch's list CTAs).
        words = SCHED_HEADER + prepared.list_cap * (LIST_SLOTS // 2) * 4
        specs.append(("schedule", (words,), torch.int32))
    return specs


def prepare(*, inputs, union=None, per_token, scratch=True):
    """Plan the FP8 direct path, one wave per row. As BF16 direct (sparse_prefill_direct), the
    call packs every request's context while the pack fits the platform's budget; otherwise
    the direct kernel reads the raw K/V (raw)."""
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    k_starts = np.cumsum(lengths) - lengths
    p_starts = packed_block_starts(inputs.query_lens, inputs.prefix_lens)
    config = platform("token" if per_token else "tensor")
    total = int(blocks.sum())
    # PK + PV, per token with a 64-byte scale record per block and KV head. The direct kernel
    # reads PK/PV at unsigned 32-bit byte offsets (the pack reads each block at its own base).
    pack_bytes = total * hk * (2 * BLOCK_BYTES + (64 if per_token else 0))
    raw = not (
        (
            pack_bytes <= config.max_packed_kv_bytes
            or (
                total * 4 <= config.pack_tokens_per_row * int(q_lens.sum())
                and pack_bytes <= MAX_ROW_PACKED_KV_BYTES
            )
        )
        and total * hk * BLOCK_BYTES < 2**32
    )
    packed_blocks = 0 if raw else total
    request = np.repeat(np.arange(len(q_lens)), q_lens)
    local = np.arange(len(request)) - np.repeat(np.cumsum(q_lens) - q_lens, q_lens)
    rows = len(request)
    metadata = np.stack(
        (
            np.arange(rows),
            np.ones(rows, dtype=np.int64),
            # The request's first packed block, or (raw) K row / slot-table entry.
            (k_starts if raw else p_starts)[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    device = inputs.q.device
    metadata = upload(metadata, device)
    sources = valid = metadata.view(-1)  # stand-ins when nothing is packed
    if packed_blocks:
        owner = np.repeat(np.arange(len(blocks)), blocks)
        index = np.arange(len(owner)) - p_starts[owner]
        sources = upload(k_starts[owner] + 4 * index, device)
        valid = upload(np.minimum(lengths[owner] - 4 * index, 4), device)
    standin = inputs.query_positions
    prepared = DirectPlan(
        metadata=metadata,
        rows=rows,
        hk=hk,
        raw=raw,
        packed_blocks=packed_blocks,
        pack_sources=sources,
        pack_valid=valid,
        per_token=per_token,
        slotted=inputs.kv_slots is not None,
        # Raw through a slot table: the pool spans more than 4 GiB (64-bit loads).
        wide=raw and inputs.wide,
        gated=union is not None,
        direct_flag=standin if union is None else union.direct_flag,
        active=standin if union is None else union.active,
        query_tiles=standin if union is None else union.query_tiles,
        source_blocks=inputs.block_indices,
        # Core/residual union plans: residual rows (union.residual_rows, union.lse) merge.
        union=union,
        residual=union is not None and union.core_ratio > 0,
    )
    # A plain launch of fewer rows than two of its WGs per CU spreads one row per WG.
    prepared.waves = PLAIN_WAVES if rows * hk >= 2 * PLAIN_WAVES * get_cu_num() else 1
    if prepared.residual:
        if union.query_tile > 64:
            raise ValueError("FP8 direct lists at most 64 rows per union tile")
        if union.num_tiles >= LIST_PARTS * CORE_COUNT:
            raise ValueError(
                f"FP8 direct lists fewer than {LIST_PARTS * CORE_COUNT} tiles"
            )
        # Rows of one part's tiles: a list's length.
        prepared.list_cap = -(-union.num_tiles // LIST_PARTS) * union.query_tile
        # The list CTAs' counters, zero between pack launches (the last tile listed zeroes
        # them): not arena scratch, which other plans overwrite.
        prepared.counters = torch.zeros(
            COUNTER_STRIDE * (COUNTER_SLOTS + LIST_SLOTS),
            dtype=torch.int32,
            device=device,
        )
    if scratch:
        allocate_scratch(prepared, scratch_specs(prepared), device)
    return prepared


_PACK, _DIRECT = {}, {}


def _standin(prepared):
    """A small FP32 tensor for the arguments a specialization does not read."""
    return prepared.metadata.view(-1)[:16].view(torch.float32)


def _pack_args(inputs, prepared, stream):
    """_pack_launch's arguments by parameter name, in its order."""
    empty = _standin(prepared)
    union, residual, raw = prepared.union, prepared.residual, prepared.raw
    standin = prepared.pack_valid
    # A raw call packs nothing: byte stand-ins of PK/PV's rank, so it shares the variant.
    blank = empty.view(torch.uint8).view(1, -1)
    return {
        "K": inputs.k,
        "V": inputs.v,
        "KS": inputs.k_scale.view(-1),
        "VS": inputs.v_scale.view(-1),
        "PK": blank if raw else prepared.packed_key,
        "PV": blank if raw else prepared.packed_value,
        "PSC": prepared.scale_records if prepared.per_token and not raw else empty,
        "SOURCES": prepared.pack_sources,
        "VALID": prepared.pack_valid,
        "SLOTS": inputs.kv_slots if prepared.slotted else prepared.pack_sources,
        "DIRECT_FLAG": prepared.direct_flag,
        "Q": inputs.q.view(-1),
        "Q8": inputs.q8.view(-1),
        "QS": inputs.qs.view(-1),
        # K3b and list CTAs of a core/residual union plan (stand-ins otherwise).
        "BLOCK_ROWS": inputs.block_indices.view(-1) if residual else standin,
        "TILE_COUNTS": union.counts if residual else standin,
        "TILE_BLOCKS": union.blocks.view(-1) if residual else standin,
        "TILE_META": union.metadata.view(-1) if residual else standin,
        "TILE_ACTIVE": union.active if residual else standin,
        "TILE_RESIDUAL": union.residual if residual else standin,
        "RESIDUAL_ROWS": union.residual_rows if residual else standin,
        "SCHED": prepared.schedule if residual else standin,
        "COUNTERS": prepared.counters if residual else standin,
        "DIRECT_META": prepared.metadata.view(-1),
        "N": inputs.k.shape[0],
        "PACKED": prepared.packed_blocks,
        "ROWS": inputs.q.shape[0] * inputs.q.shape[1],
        "CAPACITY": union.block_capacity if residual else 0,
        "MAX_BLOCKS": union.max_blocks if residual else 0,
        "TILES": union.union_tiles if residual else 0,
        "LISTED": union.num_tiles if residual else 0,
        "DIRECT_ROWS": prepared.rows,
        "CAP": prepared.list_cap if residual else 0,
        "HK": prepared.hk,
        "PER_TOKEN": prepared.per_token,
        "SLOTTED": prepared.slotted,
        "GATED": prepared.gated,
        "RESIDUAL": residual,
        "TOPK": inputs.topk if residual else 0,
        "stream": stream,
    }


def _direct_args(inputs, prepared, out, stream):
    """_direct_launch's arguments by parameter name, in its order."""
    empty = _standin(prepared)
    raw, residual = prepared.raw, prepared.residual
    return {
        "Q8": inputs.q8.view(-1),
        "QS": inputs.qs.view(-1),
        "PK": inputs.k if raw else prepared.packed_key,
        "PV": inputs.v if raw else prepared.packed_value,
        "PSC": prepared.scale_records if prepared.per_token and not raw else empty,
        "KS": inputs.k_scale.view(-1),
        "VS": inputs.v_scale.view(-1),
        "OUT": out.view(-1),
        "SLOTS": (
            inputs.kv_slots if raw and prepared.slotted else prepared.metadata.view(-1)
        ),
        "SCALE_LOG2": inputs.scale_log2,
        "BLOCKS": prepared.source_blocks.view(-1),
        "META": prepared.metadata.view(-1),
        "ACTIVE": prepared.active,
        "QUERY_TILES": prepared.query_tiles,
        "LSE": prepared.union.lse.view(-1) if residual else empty,
        "FLAG": prepared.direct_flag,
        "SCHED": prepared.schedule if residual else empty,
        "CAP": prepared.list_cap if residual else 0,
        "H": inputs.q.shape[1],
        "HK": prepared.hk,
        "NQ": inputs.q.shape[0],
        "ROWS": prepared.rows,
        "PACKED": prepared.packed_blocks,
        "NK": inputs.k.shape[0],
        "TOPK": inputs.topk,
        "GATED": prepared.gated,
        "PER_TOKEN": prepared.per_token,
        "RESIDUAL": residual,
        "RAW": raw,
        "SLOTTED": raw and prepared.slotted,
        "WIDE": prepared.wide,
        "WAVES": prepared.waves,
        "GRID": prepared.rows * prepared.hk,
        "stream": stream,
    }


def _compile(cache, launch, args, key, empty, device, stream, hints=None):
    """Compile once per specialization: the launcher's constexpr arguments named by key.
    Compiling runs the launcher, so the argument named empty (the pack's Q rows, the direct
    grid) is 0 and it launches nothing: neither kernel is idempotent (the pack's K3b CTAs
    compact block_indices in place, residual rows merge into out). args: by name, in the
    launcher's order."""
    specialization = (device, *(args[name] for name in key))
    compiled = cache.get(specialization)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        args = {**args, empty: 0}
        compile_ = flyc.compile if hints is None else flyc.compile[hints]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            compiled = cache[specialization] = compile_(launch, *args.values())
    return compiled


def _compiled(inputs, prepared, out, stream):
    """The pack and direct launchers' arguments (by name, in their order) and compiled
    launchers. The softmax scale is a kernel argument, not part of a specialization."""
    device = inputs.q.device
    pack = _pack_args(inputs, prepared, stream)
    direct = _direct_args(inputs, prepared, out, stream)
    pack_fn = _compile(
        _PACK,
        _pack_launch,
        pack,
        ("HK", "PER_TOKEN", "SLOTTED", "GATED", "RESIDUAL", "TOPK"),
        "ROWS",
        device,
        stream,
    )
    # Two direct waves per SIMD.
    direct_fn = _compile(
        _DIRECT,
        _direct_launch,
        direct,
        (
            "H",
            "HK",
            "TOPK",
            "GATED",
            "PER_TOKEN",
            "RESIDUAL",
            "RAW",
            "SLOTTED",
            "WIDE",
            "WAVES",
        ),
        "GRID",
        device,
        stream,
        {"waves_per_eu": 2},
    )
    return pack, direct, pack_fn, direct_fn


def run(*, inputs, prepared, out, stage="both"):
    """Pack the context and quantize Q into inputs.q8/qs (stage "pack"; the FP8 union kernel
    reads its Q8), then the direct waves ("direct"). inputs carries q8/qs, k_scale/v_scale
    and scale_log2 like the union kernel."""
    if prepared.rows == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    with torch.cuda.device(inputs.q.device):
        pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
        if stage in ("pack", "both"):
            pack_fn(*pack.values())
        if stage in ("direct", "both"):
            direct_fn(*direct.values())


def launchers(*, inputs, prepared, out):
    """(pack(q, k, v, k_scale, v_scale, slots), direct(q8, qs, k, v, out, k_scale, v_scale,
    slots, scale_log2)) with flat Q/Q8/QS/out/scales and unflattened K/V, other arguments
    prebuilt (the pack writes the bound inputs.q8/qs); None when empty.
    """
    if prepared.rows == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
    slotted, raw = prepared.slotted, prepared.raw
    # Bind only fixed arguments: the cached workspace must not keep this call's Q, K/V,
    # scales, slot table or out alive.
    names, values = tuple(pack), tuple(pack.values())
    at_slots, at_q = names.index("SLOTS"), names.index("Q")
    pack_mid, pack_table = values[4:at_slots], values[at_slots]
    pack_between, pack_tail = values[at_slots + 1 : at_q], values[at_q + 1 :]
    if slotted:
        pack_table = None
    names, values = tuple(direct), tuple(direct.values())
    at_slots = names.index("SLOTS")
    packed_kv, records = values[2:4], values[4]
    table_fixed, direct_tail = values[at_slots], values[at_slots + 2 :]
    if raw:
        packed_kv = None
        if slotted:
            table_fixed = None
    del pack, direct, values

    def pack_launch(q, k, v, ks, vs, slots):
        table = slots if slotted else pack_table
        pack_fn(k, v, ks, vs, *pack_mid, table, *pack_between, q, *pack_tail)

    def direct_launch(q8, qs, k, v, o, ks, vs, slots, scale_log2):
        kv = (k, v) if raw else packed_kv
        table = slots if raw and slotted else table_fixed
        direct_fn(q8, qs, *kv, records, ks, vs, o, table, scale_log2, *direct_tail)

    return pack_launch, direct_launch
