# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA direct attention over FP8 (E4M3FNUZ) K/V: one query wave over its own
selected blocks, FP8 MFMAs.

Each call packs every request's K/V into four-token blocks in FP8 MFMA operand order, so the
direct kernel's loads are whole, coalesced operands (no LDS, no permutes); the same launch's
CTAs past the blocks quantize Q (sparse_prefill_common.quantize_rows) for both FP8 kernels.
With a union plan the pack runs before the FP8 union kernel, which reads its V^T (and per
token the scale records) for every tile; K is packed only when K3 routes some tile to direct
(direct_flag[0]):
  PK[block][i * 256 + (g * 4 + token) * 16 + 0..15]
      = K[token][64i + 8g + 0..7] ++ K[token][64i + 32 + 8g + 0..7]
  PV[block][p * 256 + d * 16 + (m - 4p) * 4 + token] = V[token][16d + m]   (p = m >> 2)
Per-token scales are stored per block as one 64-byte record of four 16-byte token entries
(k_scale, log2(v_scale), 1 / v_scale, 0), the V terms clamped as in the union kernel, which
loads a token's entry at once; tokens past a request's end read 0 (a slot table's later pool
rows too). The pack is [blocks * HK, 1024] bytes: FlyDSL passes tensor dims as int32, and a
packed context may span up to 4 GiB.

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
per-token scale terms first, the other K half between QK's halves, V after QK); its block ids
come one chunk earlier by scalar loads, so the K/V loads stay in flight (vector loads
complete in order). The loop runs two chunks per iteration, alternating two register sets,
and stops before the last chunk, which runs apart without loads and alone masks keys past
the key list (every earlier chunk holds eight complete blocks). Per token the 24 scale terms
of a 16-lane row's blocks (its g) take two dword loads per lane and reach their users by DPP
row broadcasts. Plain plans (no core/residual tiles) run PLAIN_WAVES rows per WG, one wave
each, heaviest rows first: waves of rows the union serves exit at once, and every wave of a
launch whose tiles all run in the union exits on K3's direct flag before any row load. A plain
plan whose rows all select topk blocks (equal rows, at least a wave per SIMD) runs persistent
instead (PLAIN_PERSIST): the launch's waves start their rows together and, kept at one pace,
sweep their sorted key lists in step, so the K/V they read at a time stay in the L2. Its WGs
first list the tiles K3 routes direct (the union's plan): the waves take only those tiles'
rows, however few. So does a core/residual plan of such rows whose union keeps no tile (the
pack launch's list says so on the GPU; its listed launch then exits).

Core/residual union plans (prepare.CORE_RATIOS): the pack launch also runs one K3b CTA per
union tile (prepare.residual_tile), which moves the residual blocks of each row of an active
core tile to the front of its block_indices row. Those rows then run here over their residual
blocks only (no causal tail) and merge into the union output by log-sum-exp (the union kernel
writes each row's lse). A residual count is known only then, so the same CTA (one per tile)
lists the tile's direct rows in chunk-count buckets, and the direct waves take the rows from
that list, heaviest first: the launch does not end on a long row, and a wave reads its row's
parameters in one entry.
"""

import math
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch

from aiter.jit.utils.chip_info import get_cu_num

from . import sparse_prefill_primitives as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    QROWS,
    QUEUE_WORDS,
    SC_DEVICE,
    _as_f32,
    _as_i32,
    _behind_prio,
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fp8_word,
    _lazy_max,
    _lds_relaxed_load,
    _lds_relaxed_store,
    _maximum,
    _mfma_fp8,
    _min,
    _queue_exit,
    _queue_take,
    _scaled_bf16,
    _v_scale_terms,
    quantize_rows,
)
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
# lie past every pack (< 4 GiB - 1 MiB of K or V) and read zeros without a memory request.
PAD_BASE = -1024
# Core/residual plans run the direct rows heaviest first: the pack launch lists every row in
# one of LIST_BUCKETS chunk-count buckets (bucket 0 the longest rows), each split in
# LIST_PARTS parts by tile (the atomics of one counter serialize). Slot s = bucket *
# LIST_PARTS + part. Schedule scratch (Int32 words): word b the rows of buckets 0 .. b, word
# LIST_BUCKETS + s the rows of slots 0 .. s, word SCHED_WHOLE 1 when no listed tile is active
# (every row runs whole), then from word SCHED_HEADER list q * LIST_PARTS + part (q < 4) of
# `cap` entries, bucket 2q of the part filling it from the front and bucket 2q + 1 from the
# back, one 16-byte entry per row: row | residual << 31, its visible keys (residual: 4 x its
# residual blocks), its request's first packed block, 0.
LIST_BUCKETS = 8
LIST_PARTS = 8
SLOTS = LIST_BUCKETS * LIST_PARTS
SCHED_WHOLE = LIST_BUCKETS + SLOTS
SCHED_HEADER = SCHED_WHOLE + 4
# Plain plans' rows per WG (one wave each): four waves fill a CU's SIMDs at once, so a launch
# of equal rows (all selecting topk blocks) refills a CU in a quarter of the WG dispatches, and a
# launch the union serves dispatches a quarter of the (empty) WGs.
PLAIN_WAVES = 4
# Persistent plain launch: one WG of PLAIN_PERSIST waves per CU (waves w and w ^ 4 share a
# SIMD). Rows started together sweep their sorted blocks together (the first rows of a plain
# launch ran at about 520 chunks/us, later ones near 470), but the older wave of a SIMD issues
# first: a wave more than PRIO_MARGIN chunks behind its sibling runs at s_setprio(PRIO_BEHIND)
# so the two keep pace. Rows past the whole rounds (one row per wave) run one per SIMD, on
# different CUs.
PLAIN_PERSIST = 8
PRIO_BEHIND = 1
PRIO_MARGIN = 0
# LDS a persistent WG keeps besides its pace words, output slots and tile list.
PERSIST_SPARE = 1024
# Residual merges of groups of at most COMPACT_GROUP heads run on 8 lanes per head (_merge_o).
COMPACT_GROUP = 8
# Persistent and listed rows stage their output in LDS, a head per SLOT_HEAD bytes (512 + 16:
# on both the write and the read side lanes 8k .. 8k + 7 hit different banks) (_store_o).
SLOT_HEAD = 528
# Persistent counters (zero between pack launches), each on its own 256-byte line: tiles
# listed per part, parts listed, then one per slot. A part's count adds CORE_COUNT per active
# tile, the parts' count CORE_COUNT per part with one (a tile count in the low half).
COUNTER_STRIDE = 64
COUNTER_SLOTS = LIST_PARTS + 1
CORE_COUNT = 1 << 16


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
    """Scale record of the block: per token (lanes 0..3) 16 bytes of k_scale, log2(v_scale),
    1 / v_scale and 0."""
    scale_extent = fx.Uint32(n) * hk * 4
    ks, vs = _buffer(KS, scale_extent), _buffer(VS, scale_extent)
    t = lane & 3
    source = (t < valid).select(((start + t) * hk + head) * 4, fx.Int32(scale_extent))
    sk = _as_f32(_buffer_words(ks, source, 1)[0])
    lsv, rsv = _v_scale_terms(_as_f32(_buffer_words(vs, source, 1)[0]))
    record_extent = fx.Uint32(packed) * hk * 64
    record = (inside & (lane < 4)).select(
        (block * hk + head) * 64 + t * 16, fx.Int32(record_extent)
    )
    _buffer_store_words(
        fx.Vector.from_elements(
            [_as_i32(sk), _as_i32(lsv), _as_i32(rsv), fx.Int32(0)], fx.Int32
        ),
        _buffer(PSC, record_extent),
        record,
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
    with_k,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
):
    lane = fx.Int32(fx.thread_idx.x) & 63
    block, head = pair // HK, pair % HK
    inside = pair < PACKED * HK
    start = prim.uniform(fx.Int32(SOURCES[_min(block, PACKED - 1)]))
    valid = inside.select(
        prim.uniform(fx.Int32(VALID[_min(block, PACKED - 1)])), fx.Int32(0)
    )
    if fx.const_expr(SLOTTED):
        # A block occupies four consecutive pool rows from its first token's slot.
        start = prim.uniform(fx.Int32(SLOTS[start]))
    extent = fx.Uint32(N) * HK * D
    kr, vr = _buffer(K, extent), _buffer(V, extent)
    oob = fx.Int32(extent)
    out_extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    pk, pv = _buffer(PK, out_extent), _buffer(PV, out_extent)
    target = inside.select(
        (block * HK + head) * BLOCK_BYTES + lane * 16, fx.Int32(out_extent)
    )
    if with_k != 0:
        # K: piece (i, token, g) of lane i * 16 + token * 4 + g (reads stay contiguous), stored
        # token-minor so each 4-lane quad of a direct K load reads 64 contiguous bytes.
        i, token, g = lane >> 4, (lane >> 2) & 3, lane & 3
        row = ((start + token) * HK + head) * D
        row = (token < valid).select(row, oob)
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
    _pack_v(vr, pv, target, start, valid, head, lane, HK, oob)
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
    publishes the slot counts and zeroes the counters for the next launch."""
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & 63
    part = tile & (LIST_PARTS - 1)
    first_row = prim.uniform(fx.Int32(TILE_META[tile * 5]))
    count_rows = prim.uniform(fx.Int32(TILE_META[tile * 5 + 1]))
    position = prim.uniform(fx.Int32(TILE_META[tile * 5 + 4]))
    active = prim.uniform(fx.Int32(TILE_ACTIVE[tile])) != 0
    # K3b's residual counts (stored by lane 0 of each wave) are in L2 before wave 0 reads them
    # back past its L1.
    fx.rocdl.s_waitcnt(vmcnt=0)
    fx.gpu.barrier()
    if prim.uniform(tid >> 6) == 0:
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
        extent = (fx.Uint32(cap) * (SLOTS // 2) * 4 + SCHED_HEADER) * 4
        sched = _buffer(SCHED, extent)
        p_start = fx.Int32(DIRECT_META[row * 5 + 2])
        _buffer_store_words(
            fx.Vector.from_elements(
                [row | flag, visible, p_start, fx.Int32(0)], fx.Int32
            ),
            sched,
            listed.select((SCHED_HEADER + entry * 4) * 4, fx.Int32(extent)),
        )
        # The reservations are performed (their results arrived) before the tile counts. Lane 0
        # counts the tile (then the part), an active tile CORE_COUNT more; the wave reads its
        # result by v_readlane.
        fx.rocdl.s_waitcnt(vmcnt=0)
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
    raw = fx.Int32(fx.block_idx.x)
    pack_ctas = (PACKED * HK + 3) // 4
    quant_end = pack_ctas + (ROWS + QROWS - 1) // QROWS
    # The list CTAs (latency-bound; K3b first on union tiles) are dispatched first: the
    # bandwidth-bound K/V and Q CTAs overlap them instead of the launch ending on them.
    listing = LISTED if fx.const_expr(RESIDUAL) else fx.Int32(0)
    cta = (raw < listing).select(raw + quant_end, raw - listing)
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
        pair = cta * 4 + prim.uniform(fx.Int32(fx.thread_idx.x) >> 6)
        args = (K, V, KS, VS, PK, PV, PSC, SOURCES, VALID, SLOTS, pair)
        if fx.const_expr(GATED):
            # With a union plan the FP8 union kernel reads PV (and the scale records); K3
            # sets direct_flag[0] when some tile goes direct, the only reader of PK.
            with_k = prim.uniform(fx.Int32(DIRECT_FLAG[0]))
            _pack_body(*args, with_k, N, PACKED, HK, PER_TOKEN, SLOTTED)
        else:
            _pack_body(*args, fx.Int32(1), N, PACKED, HK, PER_TOKEN, SLOTTED)


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
    if PACKED > 0:
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


def _chunk_ids(env, t):
    """Listed block ids of key-list slots 8t .. 8t + 7 by two wave-uniform scalar loads (SMEM,
    lgkmcnt): they do not wait behind the K/V loads in the in-order vector memory counter.
    Slots past the row's list read another row's ids or zero (unused)."""
    offset = env.ids_row + t * CHUNK
    words = [
        fx.rocdl.s_buffer_load(env.gblocks, offset + 4 * h, count=4) for h in range(2)
    ]
    return [prim.uniform(fx.Vector(w)[i]) for w in words for i in range(4)]


@flyc.jit
def _chunk_bases(env, ids, t):
    """(K bases of blocks 8t + jj, 8t + 4 + jj; V/scale bases of blocks 8t + g, 8t + 4 + g):
    complete blocks in block_indices order, the causal tail block, then padding slots at
    PAD_BASE (masked; their loads read zeros without a memory request). A chunk of eight
    complete blocks takes the short path."""
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


def _k_loads(env, bases, kg):
    """K words of QK half kg of a chunk (blocks 8t + 4kg + jj): 4 b128 loads."""
    return [
        _buffer_words(env.gk, bases[kg] + 256 * i + env.k_lane, 4) for i in range(4)
    ]


def _v_loads(env, bases):
    """V words of a chunk: 8 b128 loads."""
    return [
        _buffer_words(env.gv, bases[2 + b] + 256 * p + env.v_lane, 4)
        for b in range(2)
        for p in range(4)
    ]


def _scale_words(env, bases, zero4):
    """Per token the chunk's scale terms (two dword loads), else zeros."""
    if not env.per_token:
        return fx.Vector.from_elements([zero4[0], zero4[0]], fx.Int32)
    # The scale records of the row's blocks 8t + g, 8t + 4 + g (per token k_scale,
    # log2(v_scale), 1 / v_scale, 0) over its 16 lanes: a dword load per block and lane
    # instead of six 16-byte loads per lane. Bases are unsigned: past 2 GiB of packed K/V
    # an arithmetic shift would read zeros past the records.
    rec = [fx.Int32(fx.Uint32(bases[2 + b]) >> 10) * 64 for b in range(2)]
    scales = [_buffer_words(env.gsc, rec[b] + env.sc_lane, 1)[0] for b in range(2)]
    return fx.Vector.from_elements(scales, fx.Int32)


def _chunk_words(env, bases, zero4):
    """K, V words and scale terms of a chunk at once (the row's first)."""
    k = _k_loads(env, bases, 0) + _k_loads(env, bases, 1)
    v = _v_loads(env, bases)
    return _flatten(k), _flatten(v), _scale_words(env, bases, zero4)


def _scale_term(sc, b, field, r):
    """Per-token scale term (field 0 k_scale, 1 log2(v_scale), 2 1 / v_scale) of token r of
    the lane's block b: word b of lane 4r + field of its 16-lane row (g), read by a DPP row
    broadcast."""
    return prim.row_broadcast(_as_f32(sc[b]), 4 * r + field)


def _probabilities(vals, sc, per_token):
    """FP8 P' words of the lane's 8 keys and its row-sum term (P' / v_scale per token). The
    sum adds the FP8 values PV multiplies (decoded back), so O and l weigh each key alike.
    """
    p = [_exp(vals[i]) for i in range(8)]
    words = (_fp8_word(p[0:4]), _fp8_word(p[4:8]))
    q = [x for w in words for x in prim.fp8_pair(w, False) + prim.fp8_pair(w, True)]
    terms = (
        [q[i] * _scale_term(sc, i // 4, 2, i % 4) for i in range(8)] if per_token else q
    )
    total = terms[0]
    for x in terms[1:]:
        total = total + x
    return words, total


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
    o[4m + r]), 16-byte piece p of them in words 4p .. 4p + 3. Stored from the lanes, each
    store writes a piece of 4 x group lines. A persistent or listed row (env.stage) passes
    them through its wave's LDS slot instead, so that store s writes the group's bytes
    1024 s .. + 1023: whole lines instead of 16-byte pieces of lines."""
    words = []
    for r in range(4):
        values = [o[4 * m + r] for m in range(16)]
        for k in range(4):
            words += _scaled_bf16(values[4 * k : 4 * k + 4], inv)
    if env.stage is None:
        for s in range(8):
            _buffer_store_words(
                fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
                env.gout,
                env.out_lane + 16 * s,
            )
        return
    for p in range(8):
        prim.store_lds(env.stage, env.stage_write, words[4 * p : 4 * p + 4], 16 * p)
    _store_staged(env)


def _store_staged(env):
    """The group's output from the wave's LDS slot (_store_o): lane i of store s stores the
    group's 16-byte piece 64 s + i."""
    stores = (env.group + 1) // 2
    # The wave's LDS operations run in order: the reads see the other lanes' writes. All
    # reads first, then the stores back to back.
    pieces = [
        prim.load_lds(env.stage, env.stage_read, 2 * SLOT_HEAD * s)
        for s in range(stores)
    ]
    lane = fx.Int32(fx.thread_idx.x) & 63
    for s in range(stores):
        at = env.stage_out + 1024 * s
        if 2 * s + 1 == env.group:
            # An odd group's last store: its upper half reads the spare slot.
            at = (lane < 32).select(at, fx.Int32(env.stage_end))
        _buffer_store_words(pieces[s], env.gout, at)


def _from_lower(own, lower):
    """own in lanes 0 .. 7 of each 16-lane row, lane i - 8's lower in lanes 8 .. 15 (one
    v_mov_b32_dpp: row_shr:8 into banks 2 and 3)."""
    return prim.dpp_banks(own, lower, 0x118, 0xC)


def _merge_o(env, o, inv, m_run, total):
    """Residual row: O_d = o * inv merged with the row's union output O_u (BF16, already
    normalized) by their log-sum-exps (log2 units), then stored like _store_o, a half (32
    dims) at a time. O_u and the LSE were loaded at the row's start (_union_words). A group
    of at most COMPACT_GROUP heads leaves lanes n >= 8 idle: lane n + 8 merges the upper
    half of lane n's head (one DPP move per value), so each lane merges one half. Staged
    (env.stage: the slot overlaps O_u's), the halves are written to it after every read of
    O_u."""
    lse_u = _as_f32(env.lse_word)
    lse_d = fx.Float32(m_run) + prim.log2(total)
    top = _maximum(lse_u, lse_d)
    w_u, w_d = _exp(lse_u - top), _exp(lse_d - top)
    norm = prim.rcp(w_u + w_d)
    w_u, w_d = w_u * norm, w_d * norm * fx.Float32(inv)
    # The LDS copy of O_u landed long ago; LLVM waits for it before the first read.
    fx.rocdl.s_waitcnt(vmcnt=0)
    if env.group <= COMPACT_GROUP:
        w_u, w_d = _from_lower(w_u, w_u), _from_lower(w_d, w_d)
        halves = [
            (
                env.union_compact,
                env.out_compact,
                env.stage_compact if env.stage is not None else None,
                lambda m, r: _from_lower(o[4 * m + r], o[4 * m + r + 2]),
            )
        ]
    else:
        halves = [
            (
                env.union_lane + 64 * half,
                env.out_lane + 64 * half,
                env.stage_write + 64 * half if env.stage is not None else None,
                lambda m, r, half=half: o[4 * m + r + 2 * half],
            )
            for half in range(2)
        ]
    staged = []
    for union_at, out_at, stage_at, value in halves:
        union = [prim.load_lds(env.union, union_at + 16 * s) for s in range(4)]
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
        if env.stage is not None:
            staged.append((stage_at, words))
            continue
        for s in range(4):
            _buffer_store_words(
                fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
                env.gout,
                out_at + 16 * s,
            )
    if env.stage is not None:
        for stage_at, words in staged:
            for p in range(4):
                prim.store_lds(env.stage, stage_at, words[4 * p : 4 * p + 4], 16 * p)
        _store_staged(env)


def _union_words(env, row, hkv, lane):
    """A residual row's union output O_u (its group heads, contiguous BF16 rows of 512 bytes)
    copied to the wave's LDS (LDS DMA, no registers held across the chunk loop) and the
    lane's LSE word, issued at the row's start; _merge_o reads them."""
    q_rows = fx.Uint32(env.nq) * env.heads
    first = (row * env.heads + hkv * env.group) * (D * 2)
    resource = fx.rocdl.get_buffer_rsrc(env.gout)
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


def _qk_half_fp8(env, k, kg):
    """QK half kg of a chunk: S of blocks 8t + 4kg + jj from its K words k (eight 16x16x32 FP8
    MFMAs; sparse_prefill_common._qk is the BF16 direct kernels' 16x16x16 step)."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for c in range(8):
        w = k[4 * kg + (c >> 1)]
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
def _direct_row(ins, row, hkv, p_start, visible, residual, on, pace=None, take=None):
    """The wave's query row and KV head: residual (a core/residual row) merges into the
    union output; visible keys of the row (residual: 4 x its residual blocks). Persistent
    plain waves: pace (the LDS words of the wave's and its SIMD sibling's chunk counts, the
    wave's count before this row) and take (the queue's next index, taken in the last chunk).
    Returns (chunks, the take) for plain rows."""
    group = ins.H // ins.HK
    lane = fx.Int32(fx.thread_idx.x) & 63
    n, g = lane & 15, lane >> 4
    complete = _min(visible >> 2, fx.Int32(ins.TOPK))
    tail = visible & 3
    tail_block = visible >> 2
    blocks = complete + (tail != 0).select(fx.Int32(1), fx.Int32(0))
    chunks = on.select((blocks + CHUNK - 1) // CHUNK, fx.Int32(0))
    q_rows = fx.Uint32(ins.NQ) * ins.H
    head_ok = on & (n < group)
    qrow = row * ins.H + hkv * group + n
    extent = fx.Uint32(ins.PACKED) * ins.HK * BLOCK_BYTES
    scale_extent = fx.Uint32(ins.PACKED) * ins.HK * 64
    gq8 = _buffer(ins.Q8, q_rows * D)
    q_base = head_ok.select(qrow * D + 8 * g, fx.Int32(q_rows * D))
    qs_offset = head_ok.select(qrow * 4, fx.Int32(q_rows * 4))
    qs = _as_f32(_buffer_words(_buffer(ins.QS, q_rows * 4), qs_offset, 1)[0])
    c_scale = qs * ins.SCALE_LOG2
    v_scale = None
    if fx.const_expr(not ins.PER_TOKEN):
        c_scale = c_scale * fx.Float32(ins.KS[0])
        # Read with k_scale (a scalar load): past the loop's memory fences it would be a
        # vector load the epilogue waits for.
        v_scale = fx.Float32(ins.VS[0])
    env = SimpleNamespace(
        hkv=hkv,
        hk=ins.HK,
        p_start=p_start,
        complete=complete,
        tail=tail,
        tail_block=tail_block,
        blocks=blocks,
        residual=residual,
        jj=(lane >> 2) & 3,
        g=g,
        per_token=ins.PER_TOKEN,
        c_scale=c_scale,
        gblocks=_buffer(ins.BLOCKS, fx.Uint32(ins.ROWS) * ins.TOPK * 4),
        # Int32 offset of the row's key list (row 0 for a disabled tile, which runs no chunk).
        ids_row=on.select(row * ins.TOPK, fx.Int32(0)),
        gk=_buffer(ins.PK, extent),
        gv=_buffer(ins.PV, extent),
        gsc=_buffer(ins.PSC, scale_extent),
        k_lane=(g * 4 + (lane & 3)) * 16,
        v_lane=n * 16,
        # Scale record dword n of each block: token n // 4, field n % 4.
        sc_lane=n * 4,
        gout=_buffer(ins.OUT, q_rows * D * 2),
        out_lane=head_ok.select(qrow * (D * 2) + g * 128, fx.Int32(q_rows * D * 2)),
        lse=ins.LSE,
        nq=ins.NQ,
        heads=ins.H,
        group=group,
        stage=None,
    )
    if fx.const_expr(ins.stage is not None):
        # The output through the wave's LDS slot (_store_o): lane (n, g) writes head n's
        # dims 64g .. at n * SLOT_HEAD + 128g (lanes past the group into the spare slot
        # `group`), lane i of store s reads the group's piece 64 s + i.
        slot = pace.stage if pace is not None else 0
        env.stage = ins.stage
        env.stage_write = slot + _min(n, fx.Int32(group)) * SLOT_HEAD + g * 128
        env.stage_read = (
            slot + (lane >> 5) * SLOT_HEAD + ((lane >> 3) & 3) * 128 + (lane & 7) * 16
        )
        env.stage_out = on.select(
            (row * ins.H + hkv * group) * (D * 2) + 16 * lane, fx.Int32(q_rows * D * 2)
        )
        env.stage_end = q_rows * D * 2
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
            (row * ins.H + hkv * group + head) * (D * 2) + g * 128 + (n >> 3) * 64,
            fx.Int32(q_rows * D * 2),
        )
        if fx.const_expr(ins.stage is not None):
            env.stage_compact = (
                _min(head, fx.Int32(group)) * SLOT_HEAD + g * 128 + (n >> 3) * 64
            )

    @flyc.jit
    def run():
        # Q (B operand): head n, dims 32c + 8g .. + 7 for MFMA c.
        env.q = [_buffer_words(gq8, q_base + 32 * c, 2) for c in range(8)]
        zero4 = fx.Vector.filled(4, 0, fx.Int32)
        ids0 = _chunk_ids(env, fx.Int32(0))
        first_k, first_v, first_sc = _chunk_words(
            env, _chunk_bases(env, ids0, fx.Int32(0)), zero4
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
            _lds_relaxed_load(pace.sibling) if pace is not None else fx.Int32(0),
        )

        def softmax_pv(o, m_run, l_run, v, sc, scored):
            """(O, running max, running sum) after a chunk from its scores."""
            vals, local, ballot = scored
            (o,), l_run, vals, m_run = _lazy_max(
                (o,), l_run, vals, m_run, ballot, _row_max(local)
            )
            pw, total = _probabilities(fx.Vector(vals), sc, ins.PER_TOKEN)
            return _pv16(o, v, pw), m_run, fx.Float32(l_run) + total

        @flyc.jit
        def step(o, m_run, l_run, kw, vw, scw, i0, i1, i2, i3, i4, i5, i6, i7, sib, t):
            """Chunk t from its prefetched words. Chunk t + 1's loads (block ids from the
            previous step) issue spread over QK(t): its first K half and scale terms (with
            chunk t + 2's id loads) first, its other K half between QK(t)'s halves, its V
            after QK(t). Issued at once, they queued in the CU's vector memory path and
            stalled the wave's issue longer (most where its K/V miss the L2). sib: the
            sibling's chunk count (persistent plain waves), read one step earlier."""
            t = fx.Int32(t)
            if fx.const_expr(pace is not None):
                # The sibling's count landed with chunk t + 1's ids, which the step waits for
                # first; publish this wave's count and read the sibling's for the next step.
                count = pace.base + t
                _behind_prio(count, prim.uniform(sib), PRIO_MARGIN, PRIO_BEHIND)
                _lds_relaxed_store(pace.mine, count + 1)
                sib = _lds_relaxed_load(pace.sibling)
            bases = _chunk_bases(env, (i0, i1, i2, i3, i4, i5, i6, i7), t + 1)
            k_next = _k_loads(env, bases, 0)
            sc_next = _scale_words(env, bases, zero4)
            ids_next = _chunk_ids(env, t + 2)
            # The scheduling barriers keep each group of loads in its place among QK(t)'s
            # MFMAs, which wait for chunk t's K words.
            prim.sched_barrier(0)
            k, v = _split(kw, 8), _split(vw, 8)
            s0 = _qk_half_fp8(env, k, 0)
            prim.sched_barrier(0)
            k_next = k_next + _k_loads(env, bases, 1)
            prim.sched_barrier(0)
            s1 = _qk_half_fp8(env, k, 1)
            sc = [fx.Vector(scw)[0], fx.Vector(scw)[1]]
            scored = _scores(env, (s0, s1), sc, t, fx.Float32(m_run), False)
            prim.sched_barrier(0)
            v_next = _v_loads(env, bases)
            # MachineSink cannot move the loads above the fence past the rescale branch into
            # the block that first uses their words (the next step's).
            prim.compiler_fence()
            prim.sched_barrier(0)
            return (
                *softmax_pv(o, m_run, l_run, v, sc, scored),
                _flatten(k_next),
                _flatten(v_next),
                sc_next,
                *ids_next,
                sib,
            )

        # Two chunks per iteration: chunk t + 1's words land in the registers chunk t - 1
        # used, so the loop carries them without copies (a copy would wait for the loads at
        # the loop end). The last chunk runs apart: it issues no loads (past the key list).
        for t in range(fx.Int32(0), chunks - 2, fx.Int32(2)):
            state = step(*state, t)
            state = step(*state, fx.Int32(t) + 1)
        if (chunks & 1) == 0:
            state = step(*state, chunks - 2)
        # The next row's index: the atomic's latency hides behind this row's last chunk.
        nxt = take() if take is not None else fx.Int32(0)
        o, m_run, l_run, kw, vw, scw = state[:6]
        k, v = _split(kw, 8), _split(vw, 8)
        sc = [fx.Vector(scw)[0], fx.Vector(scw)[1]]
        scored = _scores(
            env,
            (_qk_half_fp8(env, k, 0), _qk_half_fp8(env, k, 1)),
            sc,
            chunks - 1,
            fx.Float32(m_run),
            True,
        )
        state = softmax_pv(o, m_run, l_run, v, sc, scored)
        o, l_run = fx.Vector(state[0]), fx.Float32(state[2])
        total = l_run + fx.gpu.shuffle_xor(l_run, 16, 64)
        total = total + fx.gpu.shuffle_xor(total, 32, 64)
        inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
        if fx.const_expr(not ins.PER_TOKEN):
            inv = inv * v_scale
        if fx.const_expr(ins.RESIDUAL):
            if env.residual:
                _merge_o(env, [o[i] for i in range(64)], inv, state[1], total)
            else:
                _store_o(env, [o[i] for i in range(64)], inv)
        else:
            _store_o(env, [o[i] for i in range(64)], inv)
        return nxt

    if fx.const_expr(ins.listed):
        run()
    else:
        # Disabled waves (rows another kernel serves) exit without loads.
        nxt = fx.Int32(0)
        if chunks > 0:
            nxt = run()
        else:
            nxt = take() if take is not None else fx.Int32(0)
        return chunks, nxt


def _rows_before(sched, word):
    """Eight wave-uniform schedule words from word on (scalar loads)."""
    return [
        prim.uniform(fx.Vector(fx.rocdl.s_buffer_load(sched, word + 4 * h, count=4))[i])
        for h in range(2)
        for i in range(4)
    ]


@flyc.jit
def _list_entry(sched, ends, j, cap):
    """The list entry (row | residual << 31, visible keys, p_start, 0) at position j of the
    pack launch's cost order (ends[b]: the rows of buckets 0 .. b)."""
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
    words = fx.Vector(fx.rocdl.s_buffer_load(sched, SCHED_HEADER + entry * 4, count=4))
    return [prim.uniform(words[i]) for i in range(3)]


def _listed(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    LSE: fx.Tensor,
    SCHED: fx.Tensor,
    CAP: fx.Int32,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    WHOLE_EXIT: fx.Constexpr[bool],
):
    ins = SimpleNamespace(
        Q8=Q8,
        QS=QS,
        PK=PK,
        PV=PV,
        PSC=PSC,
        KS=KS,
        VS=VS,
        OUT=OUT,
        BLOCKS=BLOCKS,
        LSE=LSE,
        H=H,
        HK=HK,
        NQ=NQ,
        ROWS=ROWS,
        PACKED=PACKED,
        SCALE_LOG2=SCALE_LOG2,
        TOPK=TOPK,
        PER_TOKEN=PER_TOKEN,
        RESIDUAL=RESIDUAL,
        listed=RESIDUAL,
    )
    # Core/residual plans only (plain plans launch _plain_kernel). The row's O_u
    # (_union_words, group heads of 512 bytes), then its output's staging slot (_store_o,
    # group + 1 heads of SLOT_HEAD bytes).
    group = H // HK
    lds_bytes = (group + 1) * SLOT_HEAD
    ins.union = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, lds_bytes, 16])
        .peek()
        .view(fx.make_layout(lds_bytes, 1))
    )
    ins.stage = ins.union
    # The rows run in the pack launch's cost order, heaviest first (a cost-sorted
    # dispatch keeps the launch from ending on a long row): work item i is list position
    # i // HK, KV head i % HK. Positions past the listed rows exit.
    item = fx.Int32(fx.block_idx.x)
    sched = _buffer(SCHED, (fx.Uint32(CAP) * (SLOTS // 2) * 4 + SCHED_HEADER) * 4)
    ends = _rows_before(sched, fx.Int32(0))
    listed = ends[LIST_BUCKETS - 1] * HK
    if fx.const_expr(WHOLE_EXIT):
        # A list of whole rows ran in the persistent launch before this one.
        whole = fx.Vector(fx.rocdl.s_buffer_load(sched, SCHED_WHOLE, count=4))[0]
        listed = (prim.uniform(whole) != 0).select(fx.Int32(0), listed)
    if item < listed:
        word, visible, p_start = _list_entry(sched, ends, item // HK, CAP)
        row, residual = word & 0x7FFFFFFF, word < 0
        _direct_row(ins, row, item % HK, p_start, visible, residual, fx.Boolean(True))


_direct_kernel = flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64, 1, 1]
)(_listed)
# The listed launch after a persistent one (WHOLE_EXIT), apart in a profile: it runs the
# rows of a list with an active tile, the persistent launch those of a whole-row list.
_direct_whole_exit_kernel = flyc.kernel(
    name="sparse_prefill_direct_fp8_listed_d256", known_block_size=[64, 1, 1]
)(_listed)


@flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64 * PLAIN_WAVES, 1, 1]
)
def _plain_kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
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
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
):
    ins = SimpleNamespace(
        Q8=Q8,
        QS=QS,
        PK=PK,
        PV=PV,
        PSC=PSC,
        KS=KS,
        VS=VS,
        OUT=OUT,
        BLOCKS=BLOCKS,
        LSE=OUT,
        H=H,
        HK=HK,
        NQ=NQ,
        ROWS=ROWS,
        PACKED=PACKED,
        SCALE_LOG2=SCALE_LOG2,
        TOPK=TOPK,
        PER_TOKEN=PER_TOKEN,
        RESIDUAL=False,
        listed=False,
        stage=None,
    )
    # Rows run in reverse: later rows list the most blocks (causal-prefix rows the fewest),
    # so the heaviest waves start first and the light or empty ones fill the tail.
    wave = prim.uniform(fx.Int32(fx.thread_idx.x) >> 6)
    work = ROWS * HK - 1 - (fx.Int32(fx.block_idx.x) * PLAIN_WAVES + wave)
    if fx.const_expr(GATED):
        # K3 raises flag[0] when some tile goes direct: otherwise every wave exits here.
        if prim.uniform(fx.Int32(FLAG[0])) != 0:
            _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED)
    else:
        _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED)


def _plain_row(ins, work, META, QUERY_TILES, ACTIVE, GATED, pace=None, take=None):
    """Work item `work` of a plain launch: query row work // HK, KV head work % HK (the last
    WG's spare waves, work < 0, exit). A plain plan's metadata row i is query row i, so the
    union test does not wait for META. Returns (chunks, the take) (_direct_row)."""
    inside = work >= 0
    work = inside.select(work, fx.Int32(0))
    row, hkv = work // ins.HK, work % ins.HK
    p_start, position = prim.uniform(META[row * 5 + 2]), prim.uniform(META[row * 5 + 4])
    on = _enabled(fx.Int32(0), fx.Int32(1), row, QUERY_TILES, ACTIVE, GATED, ins.NQ)
    return _direct_row(
        ins, row, hkv, p_start, position + 1, fx.Boolean(False), on & inside, pace, take
    )


@flyc.kernel(
    name="sparse_prefill_direct_fp8_d256", known_block_size=[64 * PLAIN_PERSIST, 1, 1]
)
def _plain_persistent_kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    FLAG: fx.Tensor,
    QUEUE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    CUS: fx.Int32,
    FLAG_WORD: fx.Constexpr[int],
    TILE_META: fx.Tensor,
    TILES: fx.Int32,
    TILE_ROWS: fx.Int32,
):
    """Plans of equal whole rows: one WG of PLAIN_PERSIST waves per CU. Wave b *
    PLAIN_PERSIST + w starts on that work item (the plain order, heaviest first) and takes
    later ones from the launch's queue (QUEUE[0]) while they are in the whole rounds (CUS *
    PLAIN_PERSIST items each), then item whole + w * CUS + b of the last round if any (one per
    SIMD, on different CUs). The last wave out zeroes the queue (QUEUE[64] counts the waves
    out) for the next launch. A gated plain plan's items are the rows of the tiles K3 routes
    direct (_tile_list): TILE_ROWS * HK items per listed tile, those past its rows exit.
    """
    ins = SimpleNamespace(
        Q8=Q8,
        QS=QS,
        PK=PK,
        PV=PV,
        PSC=PSC,
        KS=KS,
        VS=VS,
        OUT=OUT,
        BLOCKS=BLOCKS,
        LSE=OUT,
        H=H,
        HK=HK,
        NQ=NQ,
        ROWS=ROWS,
        PACKED=PACKED,
        SCALE_LOG2=SCALE_LOG2,
        TOPK=TOPK,
        PER_TOKEN=PER_TOKEN,
        RESIDUAL=False,
        listed=False,
    )
    allocator = fx.SharedAllocator()
    counts = allocator.allocate(fx.Array[fx.Int32, PLAIN_PERSIST, 16]).peek()
    # A gated plain plan's direct tiles (first row | rows << 24), a word past them for the
    # unlisted lanes' stores, and the waves' tile counts of two listing steps.
    compact = GATED and FLAG_WORD == 0
    capacity = _tile_capacity(H // HK) if compact else 0
    if fx.const_expr(compact):
        listing = allocator.allocate(fx.Array[fx.Int32, capacity + 4, 16]).peek()
        tally = allocator.allocate(fx.Array[fx.Int32, 2 * PLAIN_PERSIST, 16]).peek()
    # Each wave's output staging slot (_store_o): group + 1 heads, while the CU's 64 KiB
    # of LDS hold them (groups of at most 14 heads).
    slot = (H // HK + 1) * SLOT_HEAD
    ins.stage = None
    if fx.const_expr(_staged(H // HK)):
        ins.stage = (
            allocator.allocate(fx.Array[fx.Int8, PLAIN_PERSIST * slot, 16])
            .peek()
            .view(fx.make_layout(PLAIN_PERSIST * slot, 1))
        )
    paces = fx.Int32(
        fx.ptrtoint(fx.get_iter(counts.view(fx.make_layout(PLAIN_PERSIST, 1))))
    )
    lane = fx.Int32(fx.thread_idx.x) & 63
    wave = prim.uniform(fx.Int32(fx.thread_idx.x) >> 6)
    go = fx.Boolean(True)
    if fx.const_expr(GATED):
        # K3 raises flag[0] when some tile goes direct (a core/residual plan: the pack launch
        # raises the schedule's SCHED_WHOLE when no listed tile is active): otherwise every
        # wave exits here.
        go = prim.uniform(fx.Int32(FLAG[FLAG_WORD])) != 0
    if go:
        items = ROWS * HK
        if fx.const_expr(compact):
            tile_words = _lds_address(listing, capacity + 4)
            listed = _tile_list(
                ACTIVE,
                TILE_META,
                TILES,
                tile_words,
                _lds_address(tally, 2 * PLAIN_PERSIST),
                capacity,
                wave,
                lane,
            )
            items = prim.uniform(listed) * TILE_ROWS * HK
        # (This barrier also ends the listing: every wave reads all of it.)
        _lds_relaxed_store(paces + wave * 4, fx.Int32(0))
        fx.gpu.barrier()
        queue = _buffer(QUEUE, QUEUE_WORDS * 4)
        waves = CUS * PLAIN_PERSIST
        whole = (items // waves) * waves
        spread = whole + wave * CUS + fx.Int32(fx.block_idx.x)
        late = (spread < items).select(spread, items)
        first = fx.Int32(fx.block_idx.x) * PLAIN_PERSIST + wave
        item = (first < whole).select(first, late)
        done = fx.Int32(0)

        def take():
            return _queue_take(queue, lane)

        while item < items:
            pace = SimpleNamespace(
                mine=paces + wave * 4,
                sibling=paces + (wave ^ 4) * 4,
                base=done,
                stage=wave * slot,
            )
            if fx.const_expr(compact):
                chunks, got = _tile_row(
                    ins, items - 1 - item, META, tile_words, TILE_ROWS, pace, take
                )
            else:
                # A whole-row list (FLAG_WORD) has no row the union serves.
                chunks, got = _plain_row(
                    ins, items - 1 - item, META, QUERY_TILES, ACTIVE, False, pace, take
                )
            done = done + chunks
            nxt = prim.readlane(got, 0) + waves
            item = (item >= whole).select(items, (nxt < whole).select(nxt, late))
        _queue_exit(queue, lane, waves)


def _staged(group):
    """The persistent WG stages its waves' output rows (_store_o) in LDS."""
    return PLAIN_PERSIST * (group + 1) * SLOT_HEAD + PERSIST_SPARE <= 65536


def _tile_capacity(group):
    """Direct tiles a gated plain persistent WG lists: the LDS its output slots leave."""
    staged = PLAIN_PERSIST * (group + 1) * SLOT_HEAD if _staged(group) else 0
    free = 65536 - PERSIST_SPARE - staged - 2 * PLAIN_PERSIST * 4
    return (free // 4 - 4) // 64 * 64


def _lds_address(array, words):
    return fx.Int32(fx.ptrtoint(fx.get_iter(array.view(fx.make_layout(words, 1)))))


@flyc.jit
def _tile_list(ACTIVE, TILE_META, TILES, listing, tally, capacity, wave, lane):
    """The WG lists the plan's tiles K3 routes direct (ACTIVE clear), in tile order, as
    first row | rows << 24 words from listing on: a step's waves take 64 tiles each, count
    theirs (tally, alternate halves by step) and store them past the earlier waves' and steps'
    (the other lanes store past the list). Returns the list's length (lane values)."""
    active = _buffer(ACTIVE, fx.Uint32(TILES) * 4)
    meta = _buffer(TILE_META, fx.Uint32(TILES) * 20)
    listed = fx.Int32(0)
    for base in range(fx.Int32(0), TILES, fx.Int32(64 * PLAIN_PERSIST)):
        tile = base + wave * 64 + lane
        kept = fx.Int32(_buffer_words(active, tile * 4, 1)[0])
        first = fx.Int32(_buffer_words(meta, tile * 20, 1)[0])
        rows = fx.Int32(_buffer_words(meta, tile * 20 + 4, 1)[0])
        direct = (tile < TILES) & (kept == fx.Int32(0))
        mask = prim.ballot(direct)
        half = tally + ((base // (64 * PLAIN_PERSIST)) & 1) * (PLAIN_PERSIST * 4)
        _lds_relaxed_store(half + wave * 4, fx.Int32(fx.math.ctpop(mask)))
        fx.gpu.barrier()
        before, total = fx.Int32(0), fx.Int32(0)
        for w in fx.range_constexpr(PLAIN_PERSIST):
            count = _lds_relaxed_load(half + w * 4)
            before = before + (fx.Int32(w) < wave).select(count, fx.Int32(0))
            total = total + count
        at = listed + before + prim.lanes_below(mask)
        _lds_relaxed_store(
            direct.select(listing + at * 4, listing + capacity * 4),
            first | (rows << 24),
        )
        listed = listed + total
    return listed


def _tile_row(ins, work, META, listing, TILE_ROWS, pace, take):
    """Work item `work` of a gated plain persistent launch: row work // HK % TILE_ROWS of
    listed tile work // (TILE_ROWS * HK), KV head work % HK (past the tile's rows: exits).
    Returns (chunks, the take) (_direct_row)."""
    span = TILE_ROWS * ins.HK
    j = work // span
    local = work - j * span
    r, hkv = local // ins.HK, local % ins.HK
    entry = prim.uniform(_lds_relaxed_load(listing + j * 4))
    on = r < (entry >> 24)
    row = (entry & 0xFFFFFF) + on.select(r, fx.Int32(0))
    p_start, position = prim.uniform(META[row * 5 + 2]), prim.uniform(META[row * 5 + 4])
    return _direct_row(
        ins, row, hkv, p_start, position + 1, fx.Boolean(False), on, pace, take
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
    SCALE_LOG2: fx.Float32,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    LSE: fx.Tensor,
    SCHED: fx.Tensor,
    CAP: fx.Int32,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    GRID: fx.Int32,
    PERSIST: fx.Constexpr[bool],
    QUEUE: fx.Tensor,
    CUS: fx.Int32,
    TILE_META: fx.Tensor,
    TILES: fx.Int32,
    TILE_ROWS: fx.Int32,
    stream: fx.Stream,
):
    # SCHED: the pack launch's schedule (core/residual plans) or K3's direct flag (plain).
    # A core/residual plan of equal rows launches the persistent kernel for a list of whole
    # rows (no active tile, known on the GPU), then the listed kernel for any other list.
    if GRID > 0:
        if fx.const_expr(PERSIST):
            _plain_persistent_kernel(
                Q8,
                QS,
                PK,
                PV,
                PSC,
                KS,
                VS,
                OUT,
                BLOCKS,
                META,
                ACTIVE,
                QUERY_TILES,
                SCHED,
                QUEUE,
                H,
                HK,
                NQ,
                ROWS,
                PACKED,
                SCALE_LOG2,
                TOPK,
                GATED,
                PER_TOKEN,
                CUS,
                SCHED_WHOLE if RESIDUAL else 0,
                TILE_META,
                TILES,
                TILE_ROWS,
                value_attrs=prim.kernel_attrs(),
            ).launch(grid=(CUS, 1, 1), block=(64 * PLAIN_PERSIST, 1, 1), stream=stream)
        if fx.const_expr(not RESIDUAL and not PERSIST):
            _plain_kernel(
                Q8,
                QS,
                PK,
                PV,
                PSC,
                KS,
                VS,
                OUT,
                BLOCKS,
                META,
                ACTIVE,
                QUERY_TILES,
                SCHED,
                H,
                HK,
                NQ,
                ROWS,
                PACKED,
                SCALE_LOG2,
                TOPK,
                GATED,
                PER_TOKEN,
                value_attrs=prim.kernel_attrs(),
            ).launch(
                grid=((GRID + PLAIN_WAVES - 1) // PLAIN_WAVES, 1, 1),
                block=(64 * PLAIN_WAVES, 1, 1),
                stream=stream,
            )
        if fx.const_expr(RESIDUAL):
            (_direct_whole_exit_kernel if PERSIST else _direct_kernel)(
                Q8,
                QS,
                PK,
                PV,
                PSC,
                KS,
                VS,
                OUT,
                BLOCKS,
                LSE,
                SCHED,
                CAP,
                H,
                HK,
                NQ,
                ROWS,
                PACKED,
                SCALE_LOG2,
                TOPK,
                PER_TOKEN,
                RESIDUAL,
                PERSIST,
                value_attrs=prim.kernel_attrs(),
            ).launch(grid=(GRID, 1, 1), block=(64, 1, 1), stream=stream)


# ---- Host ----


class DirectPlan:
    """Packed-KV FP8 direct plan: one wave per (query row, KV head)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def scratch_specs(prepared):
    # [blocks * HK, 1024]: FlyDSL passes int32 dims and a packed context may exceed 2**31 bytes.
    rows = prepared.packed_blocks * prepared.hk
    specs = [
        ("packed_key", (rows, BLOCK_BYTES), torch.uint8),
        ("packed_value", (rows, BLOCK_BYTES), torch.uint8),
    ]
    if prepared.per_token:
        specs.append(("scale_records", (rows * 16,), torch.float32))
    if prepared.residual:
        # The direct rows' cost order (written by the pack launch's list CTAs).
        words = SCHED_HEADER + prepared.list_cap * (SLOTS // 2) * 4
        specs.append(("schedule", (words,), torch.int32))
    return specs


def prepare(*, inputs, union=None, per_token, scratch=True):
    """Plan the FP8 direct path: every request's context packed per call, one wave per row."""
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    k_starts = np.cumsum(lengths) - lengths
    p_starts = packed_block_starts(inputs.query_lens, inputs.prefix_lens)
    request = np.repeat(np.arange(len(q_lens)), q_lens)
    local = np.arange(len(request)) - np.repeat(np.cumsum(q_lens) - q_lens, q_lens)
    rows = len(request)
    metadata = np.stack(
        (
            np.arange(rows),
            np.ones(rows, dtype=np.int64),
            p_starts[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    owner = np.repeat(np.arange(len(blocks)), blocks)
    index = np.arange(len(owner)) - p_starts[owner]
    sources = k_starts[owner] + 4 * index
    valid = np.minimum(lengths[owner] - 4 * index, 4)
    device = inputs.q.device
    standin = inputs.query_positions
    prepared = DirectPlan(
        metadata=upload(metadata, device),
        rows=rows,
        hk=hk,
        packed_blocks=int(blocks.sum()),
        pack_sources=upload(sources, device),
        pack_valid=upload(valid, device),
        per_token=per_token,
        slotted=inputs.kv_slots is not None,
        gated=union is not None,
        direct_flag=standin if union is None else union.direct_flag,
        active=standin if union is None else union.active,
        query_tiles=standin if union is None else union.query_tiles,
        source_blocks=inputs.block_indices,
        # Core/residual union plans: residual rows (union.residual_rows, union.lse) merge.
        union=union,
        residual=union is not None and union.core_ratio > 0,
    )
    # A plan of equal rows (every row selects topk blocks), at least a wave per SIMD, runs
    # persistent (PLAIN_PERSIST); its queue words stay zero between launches. A core/residual
    # plan's rows run whole there when the union keeps no tile (no active tile: the pack
    # launch's SCHED_WHOLE), else they run listed. A gated plain plan's WGs list its direct
    # tiles in LDS (_tile_list), so it runs persistent while they hold its tiles.
    prepared.cus = get_cu_num()
    prepared.persistent = (
        rows * hk >= prepared.cus * PLAIN_PERSIST
        and int(((metadata[:, 4] + 1) >> 2).min()) >= inputs.topk
        and (
            union is None
            or prepared.residual
            or union.num_tiles <= _tile_capacity(inputs.q.shape[1] // hk)
        )
    )
    if prepared.persistent:
        prepared.queue = torch.zeros(QUEUE_WORDS, dtype=torch.int32, device=device)
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
            COUNTER_STRIDE * (COUNTER_SLOTS + SLOTS), dtype=torch.int32, device=device
        )
    if scratch:
        allocate_scratch(prepared, scratch_specs(prepared), device)
    return prepared


_PACK, _DIRECT = {}, {}


def _standin(prepared):
    """A small FP32 tensor for the arguments a specialization does not read."""
    return prepared.packed_key.view(-1)[:16].view(torch.float32)


def _pack_args(inputs, prepared, stream):
    """_pack_launch's arguments by parameter name, in its order."""
    empty = _standin(prepared)
    union, residual = prepared.union, prepared.residual
    standin = prepared.pack_valid
    return {
        "K": inputs.k,
        "V": inputs.v,
        "KS": inputs.k_scale.view(-1),
        "VS": inputs.v_scale.view(-1),
        "PK": prepared.packed_key,
        "PV": prepared.packed_value,
        "PSC": prepared.scale_records if prepared.per_token else empty,
        "SOURCES": prepared.pack_sources,
        "VALID": prepared.pack_valid,
        "SLOTS": inputs.kv_slots if prepared.slotted else prepared.pack_sources,
        "DIRECT_FLAG": prepared.direct_flag,
        "Q": inputs.q.view(-1),
        "Q8": inputs.q8.view(-1),
        "QS": inputs.qs.view(-1),
        # K3b CTAs of a core/residual union plan (stand-ins otherwise).
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
    return {
        "Q8": inputs.q8.view(-1),
        "QS": inputs.qs.view(-1),
        "PK": prepared.packed_key,
        "PV": prepared.packed_value,
        "PSC": prepared.scale_records if prepared.per_token else empty,
        "KS": inputs.k_scale.view(-1),
        "VS": inputs.v_scale.view(-1),
        "OUT": out.view(-1),
        "SCALE_LOG2": inputs.scale_log2,
        "BLOCKS": prepared.source_blocks.view(-1),
        "META": prepared.metadata.view(-1),
        "ACTIVE": prepared.active,
        "QUERY_TILES": prepared.query_tiles,
        "LSE": prepared.union.lse.view(-1) if prepared.residual else empty,
        "SCHED": prepared.schedule if prepared.residual else prepared.direct_flag,
        "CAP": prepared.list_cap if prepared.residual else 0,
        "H": inputs.q.shape[1],
        "HK": prepared.hk,
        "NQ": inputs.q.shape[0],
        "ROWS": prepared.rows,
        "PACKED": prepared.packed_blocks,
        "TOPK": inputs.topk,
        "GATED": prepared.gated,
        "PER_TOKEN": prepared.per_token,
        "RESIDUAL": prepared.residual,
        "GRID": prepared.rows * prepared.hk,
        "PERSIST": prepared.persistent,
        "QUEUE": prepared.queue if prepared.persistent else empty,
        "CUS": prepared.cus,
        "TILE_META": prepared.union.metadata.view(-1) if prepared.gated else empty,
        "TILES": prepared.union.num_tiles if prepared.gated else 0,
        "TILE_ROWS": prepared.union.query_tile if prepared.gated else 0,
        "stream": stream,
    }


def _compile(cache, launch, args, key, empty, device, stream, hints=None):
    """Compile once per specialization: the launcher's constexpr arguments named by key.
    Compiling runs the launcher, so the argument named empty (the pack's block count, the
    direct grid) is 0 and it launches nothing: neither kernel is idempotent (the pack's K3b
    CTAs compact block_indices in place, residual rows merge into out). args: by name, in the
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
    """The pack and direct launchers' arguments (tuples, in their order) and compiled
    launchers. The softmax scale is a kernel argument, not part of a specialization."""
    device = inputs.q.device
    pack = _pack_args(inputs, prepared, stream)
    direct = _direct_args(inputs, prepared, out, stream)
    pack_fn = _compile(
        _PACK,
        _pack_launch,
        pack,
        ("HK", "PER_TOKEN", "SLOTTED", "GATED", "RESIDUAL", "TOPK"),
        "PACKED",
        device,
        stream,
    )
    # Two direct waves per SIMD.
    direct_fn = _compile(
        _DIRECT,
        _direct_launch,
        direct,
        ("H", "HK", "TOPK", "GATED", "PER_TOKEN", "RESIDUAL", "PERSIST"),
        "GRID",
        device,
        stream,
        {"waves_per_eu": 2},
    )
    return tuple(pack.values()), tuple(direct.values()), pack_fn, direct_fn


def run(*, inputs, prepared, out, stage="both"):
    """Pack the context and quantize Q into inputs.q8/qs (stage "pack"; the FP8 union kernel
    reads its V^T and Q8), then the direct waves ("direct"). inputs carries q8/qs,
    k_scale/v_scale and scale_log2 like the union kernel."""
    if prepared.rows == 0 or prepared.packed_blocks == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    with torch.cuda.device(inputs.q.device):
        pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
        if stage in ("pack", "both"):
            pack_fn(*pack)
        if stage in ("direct", "both"):
            direct_fn(*direct)


def launchers(*, inputs, prepared, out):
    """(pack(q, k, v, k_scale, v_scale, slots), direct(q8, qs, out, k_scale, v_scale,
    scale_log2)) with flat Q/Q8/QS/out/scales and unflattened K/V, other arguments prebuilt
    (the pack writes the bound inputs.q8/qs); None when empty.
    """
    if prepared.rows == 0 or prepared.packed_blocks == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
    slotted = prepared.slotted

    def pack_launch(q, k, v, ks, vs, slots):
        pack_fn(
            k,
            v,
            ks,
            vs,
            *pack[4:9],
            slots if slotted else pack[9],
            pack[10],
            q,
            *pack[12:],
        )

    def direct_launch(q8, qs, o, ks, vs, scale_log2):
        direct_fn(q8, qs, *direct[2:5], ks, vs, o, scale_log2, *direct[9:])

    return pack_launch, direct_launch
