# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact QSA attention selection, private union scratch and fused preparation launches.

These three FlyDSL kernels (K1-K3) are the private preparation step of the
FlyDSL QSA sparse attention; they are not a public op.

Recovery does not validate the selection layout; it writes each row's sorted complete blocks.
Compaction builds each union candidate tile's block union and membership from those rows, in LDS.
The next launch sorts tasks and builds disjoint mask partitions; with a slot table it also maps
the union blocks to pool blocks. Plans may split a short prefill's long union tasks into key
ranges; FP8 plans may also keep only a core of a union and leave each row's other blocks to
the direct kernel (CORE_RATIOS); residual_tile lists them, run by the FP8 pack launch. Plans
use one stream's scratch and are pinned for graphs.
"""

import math
import operator
from dataclasses import dataclass
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch

from aiter.jit.utils.chip_info import get_cu_num

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _lds_load,
    _lds_pointer,
    _lds_store,
    _min,
    _uniform,
    platform,
)

THREADS = 256
INT_MAX = 2**31 - 1
LIMITS = 8  # union step limits the route compares: max * 2**(-j/2) for j < 7, then none
ROUTE_UNROLL = 2  # tiles per thread whose plan words one route pass loads at once
# The route's first pass (two words a tile) loads up to 2048 tiles' words in one round trip.
ROUTE_FIRST_UNROLL = 8
ORDER_LOADS = 8  # task entries per thread whose plan words load at once
# K3 holds every task-entry width (entries per thread) and picks one per launch, so task
# counts never compile a new variant.
SORT_WIDTHS = (1, 2, 4, 8, 16)
EXCHANGE_WORDS = 1024  # K3 LDS: the sort's bins, then route partials
ORDER_WORDS = EXCHANGE_WORDS + 8 + 4 * LIMITS
# Union plans of fewer tasks than CUs (a short prefill leaves CUs idle while its longest
# causal tiles run): K3 splits a task of more N64 steps than the even load (steps +
# SPLIT_OVERHEAD per task over the CUs) into SPLIT_PARTS key ranges, run as separate CTA slots,
# for at most SPLIT_SLOTS tasks; the part finishing last merges their partial O / m / l records
# (PARTIAL_FLOATS each: 128 rows x 256 FP32, then m and l of 256 lanes, or of 128 rows). BF16
# union tasks run about twice as long per step, and its plans split below twice the CUs (a
# 2048-token prefill selecting every causal block, H6: -7%), as many tasks as their parts
# fit a round or pair with tasks of at most half the longest's steps (8 extends of 256
# tokens after 2k-16k, H3: 64 equal tasks, all split before, union -11%). FP8 plans of a
# task per CU or more do not: the split saves less than it costs a reused plan, whose split
# counters a separate launch zeroes before each call (the same prefill, every call reusing
# its plan: +1.3%).
SPLIT_PARTS = 2
SPLIT_SLOTS = 64
SPLIT_OVERHEAD = 6.0
PARTIAL_FLOATS = 128 * 256 + 2 * 256
# K3 LDS past ORDER_WORDS for the split policy: reduction / scan partials, then (from
# SPLIT_TABLE) the slot of each task (so at most SPLIT_TASKS tasks per split-capable plan).
SPLIT_TASKS = 1024
SPLIT_TABLE = 16
SPLIT_WORDS = SPLIT_TABLE + SPLIT_TASKS
# Persistent union CTAs take their tasks a round at a time (a round: one task per CTA, tasks by
# decreasing N64 steps). A union task costs its steps plus 2.5-4 steps of fixed work, so
# dealing every round in a snake leaves CTAs up to 7% apart. An order of one chunk (up to
# THREADS * SORT_WIDTHS[-1] tasks, GRID up to BALANCE_GRID) of a routed plan (not a
# batch-invariant one) snakes the rounds but the last BALANCE_ROUNDS that hold a runnable
# task, then balances that partial deal and those rounds by balanced largest differencing:
# twice the partial deals of largest spread merge, the most loaded CTA of one taking the least
# loaded of the other, until one deal remains (predicted loads in quarter steps,
# TASK_OVERHEADS per task).
BALANCE_ROUNDS = 4
# From this many rounds on the last three suffice (the predicted makespans of 15-round plans
# match), and the order CTA saves a sorted merge.
BALANCE_LONG = 12
BALANCE_GRID = 80
# _rank_loads keys a load as load << 7 | 127 - CTA.
assert BALANCE_GRID <= 128
# The merges pair whole deals by rank, which leaves a partial last round's empty slots on the
# CTAs its merge pairs them with (a 12000-token prefill at H3, 3.6 tasks per CTA: 3% over the
# even load). Plans of DESCENT_ROUNDS rounds instead deal the rounds in a snake and re-deal
# each against the others' loads, DESCENT_SWEEPS times, last round first: the CTA least loaded
# without it takes its largest slot (its CTAs within 1% of the even load).
DESCENT_ROUNDS = BALANCE_ROUNDS
DESCENT_SWEEPS = 3
# The re-deals rank unique keys (_key) three threads a key, each comparing it with RANK_SPAN of
# the keys (a multiple of 4 words: aligned vector reads).
RANK_SPAN = 28
assert 3 * RANK_SPAN >= BALANCE_GRID and RANK_SPAN % 4 == 0
# A union task's cost past its N64 steps, in quarter steps, by union kernel (s_memtime per task
# against its steps: FP8 3000 cycles a step plus 11.1-11.7k a task, BF16 4600 plus 11.1-11.3k).
TASK_OVERHEADS = {"fp8": 15, "bf16": 10}
# The order CTA sorts task entries by decreasing N64 steps with a counting sort over
# SORT_BINS bins in LDS (then the per-wave scan totals).
SORT_BINS = 1024
SORT_HIST = 0
SORT_TOTALS = SORT_HIST + SORT_BINS
# FP8 union tiles may keep only a core of their union (CORE_RATIOS by query heads per KV head;
# others keep whole unions): the blocks at least ceil(ratio * rows) of the tile's rows select,
# plus its diagonal (blocks from the first row's causal tail block on). Each row's other complete
# blocks are its residual, which the direct kernel attends and merges into the union output by
# log-sum-exp. K2 keeps the core when its union steps plus the residual direct work cost less
# than the whole union. Residual direct work in BN32 direct steps, per (row, block) pair and
# per row by query heads per KV head (fitted on the residual direct of SGLang-captured 12k
# selections, like FP8_ROUTES).
CORE_RATIOS = {12: 0.5, 6: 0.2}
RESIDUAL_PAIR = 0.112
RESIDUAL_ROW = {12: 3.08, 6: 3.08}
# K2 weighs a core's residual direct work by CORE_MARGINS when it decides the core (the residual
# steps K3 routes with are not scaled).
CORE_MARGINS = {12: 1.0, 6: 1.0}
# Past 2**DIRECT_SPAN_KNEE tokens of context the BF16 route terms, fitted below 64k tokens,
# favour one side by 1 + DIRECT_SPAN_SLOPE per doubling, depending on the K/V bytes the direct
# kernel reads per query head and element:
# - at least a third (group <= DIRECT_SPAN_GROUPS x 2 bytes: H3 / H6): a direct step costs more
#   once the context outgrows the caches, and K2's budget weighs it by the tile's context;
# - less (H12): the union takes those tiles slower than the direct does (weighing the direct
#   side instead, a 16k-token chunk after a 1M-token prefix, 12 layers: +1.6%), and K3's route
#   weighs the union step by the plan's longest context.
# FP8 plans keep the unweighted terms that FP8_ROUTES were fitted with.
DIRECT_SPAN_KNEE = 17
DIRECT_SPAN_SLOPE = 0.2
DIRECT_SPAN_GROUPS = 3


@dataclass(kw_only=True)
class SparsePlan:
    metadata: torch.Tensor
    blocks: torch.Tensor
    membership: torch.Tensor
    score_masks: torch.Tensor
    counts: torch.Tensor
    costs: torch.Tensor
    active: torch.Tensor
    task_order: torch.Tensor
    query_tiles: torch.Tensor
    direct_flag: torch.Tensor
    query_tile: int
    group_padded: int
    block_capacity: int
    max_blocks: int
    num_tiles: int
    # Tiles 0 .. union_tiles - 1 are union candidates (K2 compacts them, scratch holds them).
    # A batch-invariant plan follows them with direct-only tiles of the requests below the
    # union planning rows: their counts and costs stay 0, so K3 publishes them direct.
    union_tiles: int
    grid: int
    # Runtime route terms: union step share per tile, direct cost per step, fixed direct cost,
    # union cost per step (the calibration's, weighted past the knee: DIRECT_SPAN_KNEE).
    route_costs: tuple
    route_model: tuple  # platform union_fixed, union_step, row_step, route_margin
    union_ratio: float
    # K2's direct budget weight per doubling of a tile's context past the knee (0: none).
    span_slope: float
    # The union kernel's per-task overhead in quarter N64 steps (TASK_OVERHEADS by the plan's K/V
    # dtype), which K3's balance adds to each task's predicted load.
    task_overhead: int
    routing: bool = True
    # Slot-table launches (K/V read from a token pool through kv_slots): K3 writes each union block's
    # pool block id and each tile's final partial block pool row (-1: none) for the union kernel.
    slotted: bool = False
    pool_blocks: torch.Tensor | None = None
    tails: torch.Tensor | None = None
    # Gated raw direct (set by its plan): K3 lists the row groups holding a direct row, so the
    # direct grid runs them first. direct_flag[1] counts them. K3 also ranks the direct tiles
    # in tile order (tile_ranks; direct_flag[2] counts them): raw direct splits the slots the
    # packed direct launch would.
    direct_meta: torch.Tensor | None = None
    direct_list: torch.Tensor | None = None
    direct_groups: int = 0
    tile_ranks: torch.Tensor | None = None
    # Key-range parts per task (SPLIT_PARTS for a split-capable plan, else 1); with
    # split > 1 task_order entries encode (task, slot, part, split) and the union kernel
    # writes / merges partial records (partials, split_counters).
    split: int = 1
    # BF16 split plans: the parts past a round take at most the short tasks they pair with
    # (K3's _order; FP8 plans split as their kernels were calibrated).
    split_pairs: bool = False
    partials: torch.Tensor | None = None
    split_counters: torch.Tensor | None = None
    # Core/residual tiles (core_ratio > 0): K2 writes each tile's residual direct steps (0: whole
    # union); for active core tiles the residual launch rewrites each row's block_indices to its
    # residual blocks and residual_rows to their count, and the union kernel writes each union
    # row's log-sum-exp (lse, log2 units) for the direct merge.
    core_ratio: float = 0.0
    residual: torch.Tensor | None = None
    residual_rows: torch.Tensor | None = None
    lse: torch.Tensor | None = None
    rows: int = 0
    heads: int = 0


def _ballot(pred):
    return prim.ballot(pred)


# gfx9 DPP controls: quad_perm xor 1 / 2 / 3, row_half_mirror (xor 7), row_mirror (xor 15).
_DPP_XOR = {1: (0xB1,), 2: (0x4E,), 4: (0x141, 0x1B), 8: (0x140, 0x141)}


def _lane_xor(value, mask):
    """The value of lane ^ mask; every lane of the wave must be active."""
    if isinstance(value, fx.Int64):
        halves = fx.Vector.from_elements([value], fx.Int64).bitcast(fx.Int32)
        return fx.Vector.from_elements(
            [_lane_xor(halves[0], mask), _lane_xor(halves[1], mask)], fx.Int32
        ).bitcast(fx.Int64)[0]
    if mask not in _DPP_XOR:
        return fx.gpu.shuffle_xor(value, mask, 64)
    # Row-local DPP moves instead of LDS-crossbar shuffles cut the latency-bound K1 by
    # 10-14% on small calls.
    for control in _DPP_XOR[mask]:
        value = prim.dpp_move(value, control)
    return value


def _wave_reduce(value, op):
    for mask in (32, 16, 8, 4, 2, 1):
        value = op(value, _lane_xor(value, mask))
    return value


def _bitonic(keys, lane, descending):
    """Bitonic sort of one wave's keys.

    Lane t holds the keys t * w .. t * w + w - 1, w = len(keys) a power of two.
    """
    keys = list(keys)
    width = len(keys)
    log_width = width.bit_length() - 1
    log_size = log_width + 6

    def index_bit(bit):
        return ((lane >> (bit - log_width)) & 1) != 0

    for stage in range(1, log_size + 1):
        # Bit `stage` of the key index reverses its run's order; the last merge has none.
        flip = index_bit(stage) if log_width <= stage < log_size else None
        for bit in range(stage - 1, -1, -1):
            if bit < log_width:
                step = 1 << bit
                for e in range(width):
                    if e & step:
                        continue
                    first = fx.min(keys[e], keys[e | step])
                    second = fx.max(keys[e], keys[e | step])
                    if descending != (stage < log_width and ((e >> stage) & 1) == 1):
                        first, second = second, first
                    if flip is None:
                        keys[e], keys[e | step] = first, second
                    else:
                        keys[e] = flip.select(second, first)
                        keys[e | step] = flip.select(first, second)
                continue
            upper = index_bit(bit) if flip is None else index_bit(bit) ^ flip
            peers = [_lane_xor(key, 1 << (bit - log_width)) for key in keys]
            for e, peer in enumerate(peers):
                low, high = fx.min(keys[e], peer), fx.max(keys[e], peer)
                if descending:
                    low, high = high, low
                # The upper key of a pair takes the high one (the low one descending).
                keys[e] = upper.select(high, low)
    return keys


def _sort_columns(topk):
    """K1 lanes hold topk / 64 columns; the sort pads them to a power of two."""
    return 1 << (topk // 64 - 1).bit_length()


@flyc.jit
def _recover_row(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    ROWS: fx.Int32,
    row,
    lane,
    scratch,
    slot,
    TOPK: fx.Constexpr[int],
):
    # An indices row holds TOPK four-token blocks, then 3 tail tokens.
    row_bytes = (4 * TOPK + 3) * 4
    columns, width = TOPK // 64, _sort_columns(TOPK)
    visible = _uniform(POSITIONS[row]) + 1
    complete = _min(visible >> 2, fx.Int32(TOPK))
    indices = _buffer(INDICES, ROWS * row_bytes)
    # Lane holds columns 64 * k + lane: each load, store and scatter spans 64 adjacent ones.
    keys = []
    for k in fx.range_constexpr(columns):
        column = k * 64 + lane
        inside = column < complete
        # Columns past the complete blocks load out of bounds: no memory traffic.
        offset = inside.select(row * row_bytes + column * 16, ROWS * row_bytes)
        block = _buffer_words(indices, offset, 1)[0] >> 2
        keys.append(inside.select(block, fx.Int32(INT_MAX)))
    # Rows whose complete blocks already ascend skip the sort: complete causal prefixes and the
    # indexer's ascending top-k. Lane 63 takes the next column from lane 0's next key.
    next_lane = ((lane + 1) & 63) * 4
    unsorted = fx.Boolean(False)
    for k in fx.range_constexpr(columns):
        column = k * 64 + lane
        source = (
            keys[k] if k + 1 == columns else (lane == 0).select(keys[k + 1], keys[k])
        )
        following = prim.bpermute(next_lane, source)
        unsorted = unsorted | ((column + 1 < complete) & (keys[k] > following))
    if _ballot(unsorted) != 0:
        # The sort holds keys width * lane .. width * lane + width - 1: transpose through the
        # wave's LDS, INT_MAX past the last column.
        padded = keys + [fx.Int32(INT_MAX)] * (width - columns)
        for k in fx.range_constexpr(width):
            scratch[slot + k * 64 + lane] = padded[k]
        ordered = [scratch[slot + lane * width + r] for r in range(width)]
        ordered = _bitonic(ordered, lane, False)
        for r in fx.range_constexpr(width):
            scratch[slot + lane * width + r] = ordered[r]
        keys = [scratch[slot + k * 64 + lane] for k in range(columns)]
    for k in fx.range_constexpr(columns):
        column = k * 64 + lane
        BLOCKS[row * TOPK + column] = (column < complete).select(keys[k], fx.Int32(-1))


@flyc.kernel(name="sparse_prefill_recover_scatter", known_block_size=[THREADS, 1, 1])
def _recover_kernel(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    ROWS: fx.Int32,
    TOPK: fx.Constexpr[int],
):
    words = 64 * _sort_columns(TOPK)  # one wave's sort transpose
    scratch = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, 4 * words, 16])
        .peek()
        .view(fx.make_layout(4 * words, 1))
    )
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    # One wave per row, last rows first: in a causal request they reach the most blocks.
    row = ROWS - 1 - (fx.block_idx.x * 4 + wave)
    if row >= 0:
        _recover_row(
            INDICES,
            POSITIONS,
            BLOCKS,
            ROWS,
            row,
            tid & 63,
            scratch,
            wave * words,
            TOPK,
        )


@flyc.jit
def _recover_launch(
    INDICES: fx.Tensor,
    POSITIONS: fx.Tensor,
    BLOCKS: fx.Tensor,
    ROWS: fx.Int32,
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if ROWS > 0:
        _recover_kernel(
            INDICES,
            POSITIONS,
            BLOCKS,
            ROWS,
            TOPK,
            value_attrs=prim.kernel_attrs(),
        ).launch(grid=((ROWS + 3) // 4, 1, 1), block=(THREADS, 1, 1), stream=stream)


# K2: each union candidate tile's exact block union, from the rows' sorted complete blocks
# (K1's block_indices) and their causal tail blocks. One CTA per tile, last tile first; LDS is
# sized per launch. A tile whose causal range fits the LDS ORs each row's bit into a per-block
# membership word and compacts the nonzero words in block order; one up to DENSE_WINDOWS times
# that long, as equal such windows. Longer ranges go a
# WINDOW_BLOCKS-block window at a time: blocks set bits of a presence bitmap, a CTA-wide prefix
# over its words ranks every present block, and membership accumulates in LDS by union index,
# a chunk at a time. A tile whose padded union exceeds `ratio` times its direct work is
# routed direct (its count is then partial). Work grows with the rows' selections, plus one
# bitmap word per 32 blocks of a long range; no scratch grows with tiles x KV blocks.

# CTA sizes: tiles of more rows, and plans whose longest range passes the dense limit, take
# more waves (_launches).
COMPACT_THREADS = (256, 512)
# Block ids loaded in flight per thread in a pass over the rows' blocks: a launch takes a
# full tile's ids per thread in the fewest passes of at most UNROLL_MAX (_launches), so most
# tiles visit their blocks in one pass, one memory round trip (BQ10 at 512 threads: 10 ids,
# not 16 with 6 out of range; BQ21: 21, not 16 + 5); more in flight cost more than the round
# trips they save.
UNROLL_MAX = 24
WINDOW_BLOCKS = 65536
WORDS = WINDOW_BLOCKS // 32
# A range up to twice the dense limit goes in two equal dense windows instead, one of 64-bit
# membership (half the dense limit, half the windowed chunk, so more windowed rank passes) up
# to DENSE_WINDOWS times it. With TOPK <= REGISTER_TOPK each wave keeps its rows' block ids in
# registers (rows wave + waves * j, j < REGISTER_ROWS: 4 waves hold up to 24 rows, 8 waves
# 48) and every window reads them there; otherwise a window rereads them, and only
# two windows pay off. Each window zeroes and compacts its words: with 32-bit membership more
# than two cost more than the windowed passes (a 16k-token chunk after a 112k-token prefix, H6:
# 134 -> 206 us).
DENSE_WINDOWS = 8
REGISTER_TOPK = 512
REGISTER_ROWS = 6
# LDS the lower bound's marked bytes leave below the scan words: a spare byte per lane (the
# store of an id past the limit) and a word, as the cleared words round the limit up.
MARK_SPARE_BYTES = 64 + 4
# Rows per wave whose blocks bound the union before the windows (_windows_over), and the
# LDS bytes of the row overlap probe that decides whether to count it (_shares_few).
OVER_ROWS = 2
PROBE_BYTES = 4096
# LDS of a launch whose longest range does not fit densely: two CTAs per CU.
LDS_BUDGET = 32768
# Windowed LDS: presence words, their exclusive prefix counts in the window, then a membership
# chunk up to the waves' scan totals, which always take the last scan bytes: a total per wave
# (SCAN_BYTES), three on a core/residual plan (a dense tile's union, core and residual pair
# counts).
SCAN_WAVES = max(COMPACT_THREADS) // 64
SCAN_BYTES = SCAN_WAVES * 4
PRESENCE, PREFIX, MEMBER = 0, WORDS * 4, 2 * WORDS * 4


def _scan_bytes(core):
    return (3 if core else 1) * SCAN_BYTES


def _resource(tensor, size_bytes, offset=None):
    """Byte-addressed buffer over tensor from element offset on."""
    pointer = fx.get_iter(tensor)
    if offset is not None:
        pointer = pointer + offset
    return _buffer(fx.make_view(pointer, fx.make_layout(1, 1)), size_bytes)


def _word(buffer, offset):
    return _buffer_words(buffer, offset, 1)[0]


def _put(buffer, index, value):
    words = fx.Vector.from_elements([fx.Int32(value)], fx.Int32)
    _buffer_store_words(words, buffer, fx.Int32(index) * 4)


def _cta_sync():
    fx.gpu.barrier()


def _inclusive_scan(value, lane):
    for offset in (1, 2, 4, 8, 16, 32):
        source = ((lane - offset) & 63) * 4
        other = prim.bpermute(source, value)
        value = (lane >= offset).select(value + other, value)
    return value


def _popcount(value):
    return fx.Int32(fx.ctpop(value))


def _entry_bytes(wide):
    return 8 if wide else 4


def _entry_type(wide):
    return fx.Int64 if wide else fx.Int32


def _or(address, value, wide):
    fx.atomic_or(_lds_pointer(address, _entry_type(wide)), value, syncscope="workgroup")


def _member_bit(row, wide):
    return (fx.Int64(1) << fx.Int64(row)) if wide else (fx.Int32(1) << row)


def _load_entry(address, wide):
    return _lds_load(address, _entry_type(wide))


def _store_entry(address, value, wide):
    _lds_store(address, value, _entry_type(wide))


def _put_entry(members, index, value, wide):
    words = fx.Vector.from_elements([value], _entry_type(wide)).bitcast(fx.Int32)
    _buffer_store_words(words, members, index * _entry_bytes(wide))


def _zero(wide):
    return fx.Int64(0) if wide else fx.Int32(0)


def _rank(shared, offset):
    """Union index within the window of the present block `offset` blocks into it."""
    word = offset >> 5
    below = (fx.Int32(1) << (offset & 31)) - 1
    return _lds_load(shared + PREFIX + word * 4) + _popcount(
        _lds_load(shared + PRESENCE + word * 4) & below
    )


@flyc.jit
def _dense_block(shared, block, row, w0, w1, wide):
    if (block >= w0) & (block < w1):
        _or(shared + (block - w0) * _entry_bytes(wide), _member_bit(row, wide), wide)


@flyc.jit
def _mark_block(shared, block, w0, w1):
    if (block >= w0) & (block < w1):
        offset = block - w0
        _or(shared + PRESENCE + (offset >> 5) * 4, fx.Int32(1) << (offset & 31), False)


@flyc.jit
def _member_block(shared, block, row, w0, w1, c0, entries, wide):
    if (block >= w0) & (block < w1):
        index = _rank(shared, block - w0) - c0
        if (index >= 0) & (index < entries):
            address = shared + MEMBER + index * _entry_bytes(wide)
            _or(address, _member_bit(row, wide), wide)


def _visit(cta, block, row, mode, w0, w1, c0, entries):
    """Mode 0: OR the row's bit into the block's dense word (blocks [w0, w1)); 1: mark the
    block present;
    2: add the row's bit to the block's union entry in the membership chunk."""
    if mode == 0:
        _dense_block(cta.shared, block, row, w0, w1, cta.wide)
    elif mode == 1:
        _mark_block(cta.shared, block, w0, w1)
    else:
        _member_block(cta.shared, block, row, w0, w1, c0, entries, cta.wide)


def _chunks(cta, first, total):
    """cta.unroll block ids at elements first + k * threads (all loads issued before any use;
    a wave's lanes hold consecutive elements), as (element, id), id -1 past the tile's rows.
    """
    loads = []
    for k in range(cta.unroll):
        element = first + k * cta.threads
        loads.append((element, _word(cta.rows, element * 4)))
    return [
        (element, (element < total).select(block, fx.Int32(-1)))
        for element, block in loads
    ]


def _visit_chunks(cta, chunks, mode, w0, w1, c0, entries):
    topk = cta.topk
    for element, block in chunks:
        # Row of the element: TOPK block ids per row.
        if topk & (topk - 1) == 0:
            row = element >> (topk.bit_length() - 1)
        else:
            row = element // topk
        _visit(cta, block, row, mode, w0, w1, c0, entries)


@flyc.jit
def _blocks(cta, mode, w0, w1, c0, entries):
    """Visit every complete block of the tile's rows, then each row's causal tail block
    (visible // 4 holds its last visible % 4 tokens)."""
    total = cta.nrows * cta.topk
    for start in range(cta.tid, total, fx.Int32(cta.threads * cta.unroll)):
        chunks = _chunks(cta, fx.Int32(start), total)
        _visit_chunks(cta, chunks, mode, w0, w1, c0, entries)
    visible = cta.position + cta.tid + 1
    if (cta.tid < cta.nrows) & ((visible & 3) != 0) & ((visible >> 2) < cta.max_blocks):
        _visit(cta, visible >> 2, cta.tid, mode, w0, w1, c0, entries)


def _prefix_base(cta, inclusive, local):
    """(this thread's exclusive prefix over the CTA, CTA total) from the waves' inclusive scans."""
    totals = [_lds_load(cta.scan + w * 4) for w in range(cta.threads // 64)]
    running = inclusive - local
    for w, total in enumerate(totals):
        running = running + (fx.Int32(w) < cta.wave).select(total, fx.Int32(0))
    present = totals[0]
    for total in totals[1:]:
        present = present + total
    return running, present


@flyc.jit
def _cta_prefix(cta, local):
    inclusive = _inclusive_scan(local, cta.lane)
    if cta.lane == 63:
        _lds_store(cta.scan + cta.wave * 4, inclusive)
    _cta_sync()
    running, present = _prefix_base(cta, inclusive, local)
    return running, present


def _over(count, budget):
    padded = (16 * ((count + 15) >> 4)).to(fx.Float32)
    return (padded > budget).select(fx.Int32(1), fx.Int32(0))


def _owned(cta, count):
    """This thread's contiguous share [lo, hi) of `count` items."""
    per = (count + cta.threads - 1) // cta.threads
    lo = _min(cta.tid * per, count)
    return lo, _min(lo + per, count)


def _below(mask):
    return prim.lanes_below(mask)


def _count(mask):
    return fx.Int32(fx.ctpop(mask))


@flyc.jit
def _wave_bases(cta, total):
    """(this wave's exclusive prefix over the CTA, CTA total) of the waves' uniform totals."""
    if cta.lane == 0:
        _lds_store(cta.scan + cta.wave * 4, total)
    _cta_sync()
    base, present = _prefix_base(cta, fx.Int32(0), fx.Int32(0))
    return base, present


@flyc.jit
def _put_word(cta, index, word, base, mask, take):
    if take:
        destination = base + _below(mask)
        _put(cta.output, destination, index)
        _put_entry(cta.members, destination, word, cta.wide)


def _kept(cta, word, index):
    """(present, core, rows) of a dense membership word: core when at least cta.threshold of
    the tile's rows select the block or it lies on the tile's diagonal."""
    present = word != _zero(cta.wide)
    rows = fx.Int32(fx.ctpop(word))
    core = present & ((rows >= cta.threshold) | (index >= cta.diagonal))
    return present, core, rows


@flyc.jit
def _put_wave_totals(cta, total, core, pairs):
    if cta.lane == 0:
        _lds_store(cta.scan + cta.wave * 4, total)
        _lds_store(cta.scan + (SCAN_WAVES + cta.wave) * 4, core)
        _lds_store(cta.scan + (2 * SCAN_WAVES + cta.wave) * 4, pairs)
    _cta_sync()


def _wave_sums(cta, region):
    """(this wave's exclusive prefix, CTA total) of the waves' totals in scan region `region`."""
    running, total = fx.Int32(0), fx.Int32(0)
    for w in range(cta.threads // 64):
        value = _lds_load(cta.scan + (region * SCAN_WAVES + w) * 4)
        running = running + (fx.Int32(w) < cta.wave).select(value, fx.Int32(0))
        total = total + value
    return running, total


@flyc.jit
def _row_ids(cta):
    """This wave's rows' block ids (_dense_windows), lane l holding columns 64 * k + l; -1
    past the tile's rows (K1 fills a row past its complete blocks with -1)."""
    waves = cta.threads // 64
    ids = []
    for j in fx.range_constexpr(REGISTER_ROWS):
        row = cta.wave + waves * j
        live = row < cta.nrows
        for k in fx.range_constexpr(cta.topk // 64):
            element = live.select(row * cta.topk + k * 64 + cta.lane, fx.Int32(0))
            ids.append(live.select(_word(cta.rows, element * 4), fx.Int32(-1)))
    return ids


@flyc.jit
def _dense_words(cta, w0, end, ids=None):
    """One word per block of [w0, w0 + end) in LDS, each row's bit ORed in (with ids, from
    _row_ids); returns this wave's contiguous share [lo, hi) of the words, a multiple of 64.
    """
    size = _entry_bytes(cta.wide)
    zero = _zero(cta.wide)
    for index in range(cta.tid, end, fx.Int32(cta.threads)):
        _store_entry(cta.shared + fx.Int32(index) * size, zero, cta.wide)
    _cta_sync()
    if fx.const_expr(ids is None):
        _blocks(cta, 0, w0, w0 + end, fx.Int32(0), fx.Int32(0))
    else:
        waves = cta.threads // 64
        columns = cta.topk // 64
        for j in fx.range_constexpr(REGISTER_ROWS):
            row = cta.wave + waves * j
            for k in fx.range_constexpr(columns):
                _dense_block(
                    cta.shared, ids[j * columns + k], row, w0, w0 + end, cta.wide
                )
        visible = cta.position + cta.tid + 1
        if (cta.tid < cta.nrows) & ((visible & 3) != 0):
            _dense_block(cta.shared, visible >> 2, cta.tid, w0, w0 + end, cta.wide)
    _cta_sync()
    waves = cta.threads // 64
    per = ((end + waves - 1) // waves + 63) & ~63
    lo = _min(cta.wave * per, end)
    hi = _min(lo + per, end)
    return lo, hi


@flyc.jit
def _dense(cta, w0, end, count, budget, ids=None):
    """(union size, whether over budget) of the blocks [w0, w0 + end) of a tile, which fit
    the LDS one word per block, appended at union index `count` (over: count plus them
    exceed the budget, nothing appended). Each wave compacts its contiguous share 64 words
    at a time, so the outputs coalesce."""
    size = _entry_bytes(cta.wide)
    zero = _zero(cta.wide)
    lo, hi = _dense_words(cta, w0, end, ids)
    total = fx.Int32(0)
    for start in range(lo, hi, fx.Int32(64)):
        index = fx.Int32(start) + cta.lane
        address = cta.shared + _min(index, end - 1) * size
        word = (index < hi).select(_load_entry(address, cta.wide), zero)
        total = total + _count(_ballot(word != zero))
    base, present = _wave_bases(cta, total)
    over = _over(count + present, budget)
    if over == 0:
        base = base + count
        for start in range(lo, hi, fx.Int32(64)):
            index = fx.Int32(start) + cta.lane
            address = cta.shared + _min(index, end - 1) * size
            word = (index < hi).select(_load_entry(address, cta.wide), zero)
            mask = _ballot(word != zero)
            _put_word(cta, w0 + index, word, base, mask, word != zero)
            base = base + _count(mask)
    return present, over


@flyc.jit
def _dense_core(cta, end, budget):
    """_dense of a core/residual plan: (union size, whether over budget, residual direct
    steps). The union keeps only its core words when that costs less (union steps plus the
    residual direct work); the residual is 0 otherwise."""
    size = _entry_bytes(cta.wide)
    zero = _zero(cta.wide)
    lo, hi = _dense_words(cta, fx.Int32(0), end)
    total, core, pairs = fx.Int32(0), fx.Int32(0), fx.Int32(0)
    for start in range(lo, hi, fx.Int32(64)):
        index = fx.Int32(start) + cta.lane
        address = cta.shared + _min(index, end - 1) * size
        word = (index < hi).select(_load_entry(address, cta.wide), zero)
        listed, kept, rows = _kept(cta, word, index)
        total = total + _count(_ballot(listed))
        core = core + _count(_ballot(kept))
        pairs = pairs + (listed & ~kept).select(rows, fx.Int32(0))
    _put_wave_totals(cta, total, core, _wave_reduce(pairs, operator.add))
    base_all, total_all = _wave_sums(cta, 0)
    base_core, core_all = _wave_sums(cta, 1)
    _, pairs_all = _wave_sums(cta, 2)
    residual = fx.Float32(RESIDUAL_PAIR) * pairs_all.to(fx.Float32) + cta.row_cost * (
        fx.Float32(cta.nrows)
    )
    whole = (16 * ((total_all + 15) >> 4)).to(fx.Float32)
    split_cost = (16 * ((core_all + 15) >> 4)).to(fx.Float32) + cta.ratio * (
        cta.core_margin * residual
    )
    use = (pairs_all > 0) & (split_cost < whole)
    present = use.select(core_all, total_all)
    base = use.select(base_core, base_all)
    over = (use.select(split_cost, whole) > budget).select(fx.Int32(1), fx.Int32(0))
    if over == 0:
        for start in range(lo, hi, fx.Int32(64)):
            index = fx.Int32(start) + cta.lane
            address = cta.shared + _min(index, end - 1) * size
            word = (index < hi).select(_load_entry(address, cta.wide), zero)
            listed, kept, _ = _kept(cta, word, index)
            take = listed & (kept | ~use)
            mask = _ballot(take)
            _put_word(cta, index, word, base, mask, take)
            base = base + _count(mask)
    steps = use.select(fx.Int32(residual + fx.Float32(0.5)), fx.Int32(0))
    return present, over, steps


@flyc.jit
def _windows_over(cta, ids, end, windows, budget):
    """Whether a range of more than two dense windows is over budget, from a lower bound
    of its union counted before any window: the blocks of the first OVER_ROWS rows of each
    wave and every row's tail. Random selections already exceed the budget with them, so
    such a tile costs one short pass, not windows until the union outgrows the budget; real
    ones fall short and the windows decide. A byte per block, set by plain stores: rows
    selecting the same block store the same byte, no atomic serializes them."""
    over = fx.Int32(0)
    if windows > 2:
        over = _shares_few(cta, ids)
    if over != 0:
        # Blocks past the LDS left for the spare bytes only lower the bound.
        limit = _min(end, cta.scan - cta.shared - MARK_SPARE_BYTES)
        words = (limit + 3) >> 2
        for index in range(cta.tid, words, fx.Int32(cta.threads)):
            _lds_store(cta.shared + fx.Int32(index) * 4, fx.Int32(0))
        _cta_sync()
        columns = cta.topk // 64
        spare = cta.shared + words * 4 + cta.lane
        for j in fx.range_constexpr(OVER_ROWS):
            for k in fx.range_constexpr(columns):
                _mark_byte(cta.shared, ids[j * columns + k], limit, spare)
        visible = cta.position + cta.tid + 1
        tail = (cta.tid < cta.nrows) & ((visible & 3) != 0)
        _mark_byte(cta.shared, tail.select(visible >> 2, fx.Int32(-1)), limit, spare)
        _cta_sync()
        # Each byte is 0 or 1: a word's set bits count its present blocks.
        local = fx.Int32(0)
        for index in range(cta.tid, words, fx.Int32(cta.threads)):
            local = local + _popcount(_lds_load(cta.shared + fx.Int32(index) * 4))
        _, total = _cta_prefix(cta, local)
        # The windows reuse the bytes and the scan totals.
        _cta_sync()
        over = _over(total, budget)
    return over


@flyc.jit
def _shares_few(cta, ids):
    """1 when the tile's rows look random: fewer than a quarter of the blocks of wave 0's
    second row (row `waves`) are among its first row's (a byte per block hash in LDS, wave 0
    alone, then one barrier). Rows of real selections share 40-80% of their blocks (a 16k-token
    chunk after a 64k-token prefix, H3, rows 0 and 8 of a tile), random ones ~2% (with hash
    collisions ~12%), so only random tiles take the lower bound; the windows decide the
    others. Only a trigger: either way the union and the route are exact."""
    waves = cta.threads // 64
    columns = cta.topk // 64
    # The flag sits in the scan words, which the next pass writes only behind a barrier: the
    # LDS it clears first (from cta.shared on) may be reached by a wave before another reads it.
    flag = cta.scan
    if cta.wave == 0:
        for i in fx.range_constexpr(PROBE_BYTES // 256):
            _lds_store(cta.shared + (i * 64 + cta.lane) * 4, fx.Int32(0))
        spare = cta.shared + PROBE_BYTES + cta.lane
        for k in fx.range_constexpr(columns):
            block = ids[k]
            hashed = cta.shared + (block & (PROBE_BYTES - 1))
            _lds_store((block >= 0).select(hashed, spare), fx.Int8(1), fx.Int8)
        hits, listed = fx.Int32(0), fx.Int32(0)
        for k in fx.range_constexpr(columns):
            block = ids[columns + k]
            seen = _lds_load(cta.shared + (block & (PROBE_BYTES - 1)), fx.Int8)
            valid = block >= 0
            hits = hits + _count(_ballot(valid & (fx.Int32(seen) != 0)))
            listed = listed + _count(_ballot(valid))
        if cta.lane == 0:
            few = (4 * hits < listed) | (listed == 0) | (cta.nrows <= waves)
            _lds_store(flag, few.select(fx.Int32(1), fx.Int32(0)))
    _cta_sync()
    return _uniform(_lds_load(flag))


def _mark_byte(shared, block, limit, spare):
    """Set the byte of a block in [0, limit); any other id stores to the lane's spare byte
    (no branch)."""
    inside = (block >= 0) & (block < limit)
    _lds_store(inside.select(shared + block, spare), fx.Int8(1), fx.Int8)


def _dense_windows_limit(cta, dense_limit):
    windows = DENSE_WINDOWS if cta.wide and cta.topk <= REGISTER_TOPK else 2
    return windows * dense_limit


@flyc.jit
def _dense_windows(cta, end, dense_limit, budget):
    """(union size, whether over budget) of a tile's causal range in equal dense windows,
    each appended after the previous ones: the same union and membership in block order as
    the windowed way (over budget appends nothing, count 0, as its one window of a range this
    short), without a rank per block visit."""
    ids = None
    windows = (end + dense_limit - 1) // dense_limit
    over = fx.Int32(0)
    if fx.const_expr(cta.topk <= REGISTER_TOPK):
        ids = _row_ids(cta)
        over = _windows_over(cta, ids, end, windows, budget)
    size = (end + windows - 1) // windows
    w0, count = fx.Int32(0), fx.Int32(0)
    while (w0 < end) & (over == 0):
        w1 = _min(w0 + size, end)
        present, over = _dense(cta, w0, w1 - w0, count, budget, ids)
        _cta_sync()
        count = count + present
        w0 = w1
    return over.select(fx.Int32(0), count), over


@flyc.jit
def _emit_blocks(cta, base, w0, lo, hi):
    # Present blocks of this thread's words in order, from union index base + prefix.
    for index in range(lo, hi, fx.Int32(1)):
        word = fx.Int32(index)
        bits = _lds_load(cta.shared + PRESENCE + word * 4)
        union = base + _lds_load(cta.shared + PREFIX + word * 4)
        while bits != 0:
            bit = fx.cttz(bits)
            _put(cta.output, union, w0 + word * 32 + bit)
            union = union + 1
            bits = bits & (bits - 1)


@flyc.jit
def _window(cta, w0, w1, count, budget):
    """Union of blocks [w0, w1) appended at `count`; returns (new count, whether over budget)."""
    words = (w1 - w0 + 31) >> 5
    for index in range(cta.tid, words, fx.Int32(cta.threads)):
        _lds_store(cta.shared + PRESENCE + fx.Int32(index) * 4, 0)
    _cta_sync()
    zero = fx.Int32(0)
    _blocks(cta, 1, w0, w1, zero, zero)
    _cta_sync()
    # Thread t owns words [lo, hi); a CTA-wide exclusive scan of their set bits gives each
    # word's prefix.
    lo, hi = _owned(cta, words)
    local = fx.Int32(0)
    for index in range(lo, hi, fx.Int32(1)):
        local = local + _popcount(
            _lds_load(cta.shared + PRESENCE + fx.Int32(index) * 4)
        )
    running, present = _cta_prefix(cta, local)
    for index in range(lo, hi, fx.Int32(1)):
        word = fx.Int32(index)
        _lds_store(cta.shared + PREFIX + word * 4, running)
        running = running + _popcount(_lds_load(cta.shared + PRESENCE + word * 4))
    over = _over(count + present, budget)
    if over == 0:
        _emit_blocks(cta, count, w0, lo, hi)
        size = _entry_bytes(cta.wide)
        # The membership chunk ends where the scan totals start.
        entries = (LDS_BUDGET - _scan_bytes(cta.core) - MEMBER) // size
        for start in range(fx.Int32(0), present, fx.Int32(entries)):
            c0 = fx.Int32(start)
            # Union indices c0 .. c0 + filled - 1 of this chunk.
            filled = _min(present - c0, fx.Int32(entries))
            for index in range(cta.tid, filled, fx.Int32(cta.threads)):
                address = cta.shared + MEMBER + fx.Int32(index) * size
                _store_entry(address, _zero(cta.wide), cta.wide)
            _cta_sync()
            _blocks(cta, 2, w0, w1, c0, fx.Int32(entries))
            _cta_sync()
            for index in range(cta.tid, filled, fx.Int32(cta.threads)):
                slot = fx.Int32(index)
                entry = _load_entry(cta.shared + MEMBER + slot * size, cta.wide)
                _put_entry(cta.members, count + c0 + slot, entry, cta.wide)
            _cta_sync()
    _cta_sync()
    return count + over.select(fx.Int32(0), present), over


@flyc.jit
def _tile(
    ROWS,
    META,
    BLOCKS,
    MEMBERSHIP,
    COUNTS,
    COSTS,
    RESIDUAL,
    DIRECT_FLAG,
    max_blocks,
    capacity,
    ratio,
    core_ratio,
    row_cost,
    core_margin,
    span_slope,
    dense_limit,
    lds_bytes,
    wide,
    core,
    threads,
    topk,
    unroll,
):
    storage = (
        fx.SharedAllocator(static=False).allocate(fx.Array[fx.Int8, 16, 16]).peek()
    )
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(16, 1)))))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    program = fx.block_idx.x
    # Last tile first: in a causal request it holds the most blocks.
    tile = fx.grid_dim.x - 1 - program
    first_row, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    position = _uniform(META[tile * 5 + 4])
    # Union pads blocks to N64 steps; one step equals sixteen BN32 direct steps of a row.
    visible = position + lane + 1
    tokens = _min(visible >> 2, fx.Int32(topk)) * 4 + (visible & 3)
    steps = (lane < rows).select((tokens + 31) >> 5, fx.Int32(0))
    direct_steps = _wave_reduce(steps, operator.add)
    row_steps = _wave_reduce(steps, fx.max)
    # The tile's context in 1/256 doublings past the knee, in scalar registers: the leading
    # one's position, then the next 8 bits interpolate log2 linearly.
    context = position + rows
    lead = 31 - fx.ctlz(context)
    span = (lead - DIRECT_SPAN_KNEE) * 256 + (((context << (31 - lead)) >> 23) & 255)
    weight = fx.Float32(1.0) + span_slope * fx.max(span, fx.Int32(0)).to(fx.Float32)
    budget = ratio * weight * direct_steps.to(fx.Float32)
    end = _min(max_blocks, (position + rows + 3) >> 2)
    size = _entry_bytes(wide)
    # MEMBERSHIP is an int32 view; 64-bit entries take two words.
    cta = SimpleNamespace(
        shared=shared,
        scan=shared + lds_bytes - _scan_bytes(core),
        tid=tid,
        wave=wave,
        lane=lane,
        threads=threads,
        topk=topk,
        unroll=unroll,
        wide=wide,
        core=core,
        nrows=rows,
        position=position,
        max_blocks=max_blocks,
        rows=_resource(ROWS, rows * (topk * 4), fx.Int64(first_row) * topk),
        output=_resource(BLOCKS, capacity * 4, fx.Int64(tile) * fx.Int64(capacity)),
        members=_resource(
            MEMBERSHIP,
            capacity * size,
            fx.Int64(tile) * fx.Int64(capacity * (size // 4)),
        ),
    )
    if fx.const_expr(core):
        cta.ratio = ratio
        cta.row_cost = row_cost
        cta.core_margin = core_margin
        # ceil(core_ratio * rows), at least 1 (a whole union).
        cta.threshold = fx.max(
            fx.Int32(fx.Float32(rows) * core_ratio + fx.Float32(0.999)), fx.Int32(1)
        )
        cta.diagonal = (position + 1) >> 2
    count, over, residual = fx.Int32(0), fx.Int32(0), fx.Int32(0)
    if end <= dense_limit:
        if fx.const_expr(core):
            count, over, residual = _dense_core(cta, end, budget)
        else:
            count, over = _dense(cta, fx.Int32(0), end, fx.Int32(0), budget)
    elif end <= _dense_windows_limit(cta, dense_limit):
        count, over = _dense_windows(cta, end, dense_limit, budget)
    else:
        w0 = fx.Int32(0)
        while (w0 < end) & (over == 0):
            w1 = _min(w0 + WINDOW_BLOCKS, end)
            count, over = _window(cta, w0, w1, count, budget)
            w0 = w1
    if tid == 0:
        _put(_resource(COUNTS, 4, fx.Int64(tile)), fx.Int32(0), count)
        costs = _resource(COSTS, 12, fx.Int64(tile) * 3)
        _put(costs, fx.Int32(0), direct_steps)
        _put(costs, fx.Int32(1), row_steps)
        _put(costs, fx.Int32(2), (over == 0).select(fx.Int32(1), fx.Int32(0)))
        if fx.const_expr(core):
            _put(
                _resource(RESIDUAL, 4, fx.Int64(tile)),
                fx.Int32(0),
                (over == 0).select(residual, fx.Int32(0)),
            )
        if program == 0:
            flag = _resource(DIRECT_FLAG, 8)
            _put(flag, fx.Int32(0), fx.Int32(0))
            _put(flag, fx.Int32(1), fx.Int32(0))


@flyc.kernel(name="sparse_prefill_compact", known_block_size=[COMPACT_THREADS[0], 1, 1])
def _compact_kernel(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    CORE_RATIO: fx.Float32,
    ROW_COST: fx.Float32,
    CORE_MARGIN: fx.Float32,
    SPAN_SLOPE: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    UNROLL: fx.Constexpr[int],
):
    _tile(
        ROWS,
        META,
        BLOCKS,
        MEMBERSHIP,
        COUNTS,
        COSTS,
        RESIDUAL,
        DIRECT_FLAG,
        MAX_BLOCKS,
        CAPACITY,
        RATIO,
        CORE_RATIO,
        ROW_COST,
        CORE_MARGIN,
        SPAN_SLOPE,
        DENSE_LIMIT,
        LDS_BYTES,
        MEMBERS64,
        CORE,
        COMPACT_THREADS[0],
        TOPK,
        UNROLL,
    )


@flyc.kernel(name="sparse_prefill_compact", known_block_size=[COMPACT_THREADS[1], 1, 1])
def _compact_kernel_large(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    CORE_RATIO: fx.Float32,
    ROW_COST: fx.Float32,
    CORE_MARGIN: fx.Float32,
    SPAN_SLOPE: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    UNROLL: fx.Constexpr[int],
):
    _tile(
        ROWS,
        META,
        BLOCKS,
        MEMBERSHIP,
        COUNTS,
        COSTS,
        RESIDUAL,
        DIRECT_FLAG,
        MAX_BLOCKS,
        CAPACITY,
        RATIO,
        CORE_RATIO,
        ROW_COST,
        CORE_MARGIN,
        SPAN_SLOPE,
        DENSE_LIMIT,
        LDS_BYTES,
        MEMBERS64,
        CORE,
        COMPACT_THREADS[1],
        TOPK,
        UNROLL,
    )


@flyc.jit
def _compact_launch(
    ROWS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    MAX_BLOCKS: fx.Int32,
    CAPACITY: fx.Int32,
    RATIO: fx.Float32,
    CORE_RATIO: fx.Float32,
    ROW_COST: fx.Float32,
    CORE_MARGIN: fx.Float32,
    SPAN_SLOPE: fx.Float32,
    DENSE_LIMIT: fx.Int32,
    LDS_BYTES: fx.Int32,
    TILES: fx.Int32,
    MEMBERS64: fx.Constexpr[bool],
    LARGE: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    UNROLL: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TILES > 0:
        args = (
            ROWS,
            META,
            BLOCKS,
            MEMBERSHIP,
            COUNTS,
            COSTS,
            RESIDUAL,
            DIRECT_FLAG,
            MAX_BLOCKS,
            CAPACITY,
            RATIO,
            CORE_RATIO,
            ROW_COST,
            CORE_MARGIN,
            SPAN_SLOPE,
            DENSE_LIMIT,
            LDS_BYTES,
            MEMBERS64,
            CORE,
            TOPK,
            UNROLL,
        )
        # Without prim.kernel_attrs(), unlike K1 and K3: "-packed-fp32-ops" changes much more of
        # K2's code than its few packed FP32 ops, and K2 measures slower with it (MI308X: 1-3 us
        # more per call, 56 us more at a 1M-token context).
        if fx.const_expr(LARGE):
            _compact_kernel_large(*args).launch(
                grid=(TILES, 1, 1),
                block=(COMPACT_THREADS[1], 1, 1),
                smem=LDS_BYTES,
                stream=stream,
            )
        else:
            _compact_kernel(*args).launch(
                grid=(TILES, 1, 1),
                block=(COMPACT_THREADS[0], 1, 1),
                smem=LDS_BYTES,
                stream=stream,
            )


def lds_layout(max_blocks, wide, core=False):
    """(LDS bytes, longest causal range in blocks compacted densely) of a K2 launch: one
    word per block when the plan's longest range fits LDS_BUDGET, else LDS_BUDGET (ranges
    up to it still dense, longer ones windowed). core: a core/residual plan's scan totals.
    """
    scan = _scan_bytes(core)
    dense = (-(-max_blocks // 64) * 64) * _entry_bytes(wide) + scan
    if dense <= LDS_BUDGET:
        return dense, max_blocks
    return LDS_BUDGET, (LDS_BUDGET - scan) // _entry_bytes(wide)


def _block_reduce(values, ops, scratch, offset, wave):
    """CTA-wide reductions; every lane stores its wave's (equal) partial."""
    for i, (value, op) in enumerate(zip(values, ops)):
        scratch[offset + i * 4 + wave] = _wave_reduce(value, op)
    fx.gpu.barrier()
    totals = []
    for i, op in enumerate(ops):
        total = scratch[offset + i * 4]
        for w in range(1, 4):
            total = op(total, scratch[offset + i * 4 + w])
        totals.append(total)
    return totals


# Row-local DPP moves to lane l ^ 1, l ^ 2, then the mirrored half row (l ^ 7) and row (l ^ 15):
# combining each with the value leaves every lane of a 16-lane row its row's reduction.
ROW_MOVES = (0xB1, 0x4E, 0x141, 0x140)
ROUTE_ROWS = THREADS // 16


def _row_reduce(value, op):
    for control in ROW_MOVES:
        value = op(value, prim.dpp_move(value, control))
    return value


@flyc.jit
def _pair_totals(values, scratch, partials, totals, tid):
    """CTA-wide totals of alternating sums and maxima (16k values) without crossbar
    shuffles: each 16-lane row reduces them by DPP moves, lane i of a row keeps the row's
    partials of values i, i + 16, ... and stores them at `partials` (ROUTE_ROWS rows of
    len(values) + 1 words), thread i combines value i's into totals + i, then every thread
    reads the totals. (The route's 32 wave reductions, two LDS-crossbar shuffles each, cost
    its order CTA 2-3 us.)"""
    stride = len(values) + 1
    lane = tid & 15
    words = [fx.Int32(0)] * (len(values) // 16)
    for i in fx.range_constexpr(len(values)):
        row = _row_reduce(values[i], fx.max if i % 2 else operator.add)
        # Consumed at once, so the reductions add few live registers.
        words[i // 16] = (lane == i % 16).select(row, words[i // 16])
    for k in fx.range_constexpr(len(words)):
        scratch[partials + (tid >> 4) * stride + 16 * k + lane] = words[k]
    fx.gpu.barrier()
    if tid < len(values):
        loads = [scratch[partials + r * stride + tid] for r in range(ROUTE_ROWS)]
        total, peak = loads[0], loads[0]
        for value in loads[1:]:
            total, peak = total + value, fx.max(peak, value)
        scratch[totals + tid] = ((tid & 1) != 0).select(peak, total)
    fx.gpu.barrier()
    return [scratch[totals + i] for i in range(len(values))]


def _route_cost(
    union_sum,
    union_max,
    direct_sum,
    direct_max,
    task_share,
    step,
    fixed,
    union_step,
    model,
):
    # Separate multiplies and adds, as in the fitted model.
    union_fixed, _, row_step, _ = model
    union = fx.Float32(union_fixed) + union_step * fx.max(
        fx.Float32(union_max), fx.Float32(union_sum) * task_share
    )
    direct = fixed + fx.max(
        fx.Float32(row_step) * fx.Float32(direct_max), step * fx.Float32(direct_sum)
    )
    zero = fx.Float32(0.0)
    return (union_sum > 0).select(union, zero) + (direct_sum > 0).select(direct, zero)


def _route_choice(totals, limits, task_share, step, fixed, union_step, model):
    """The limit of the cheapest candidate; totals hold 4 sums per candidate, which keep ever
    fewer union tiles. Each candidate must cost route_margin less than the one it replaces:
    within the model's error the plan keeps more of its union, whose FP8 rows run closer to
    the FP8 query reference than direct ones (rows of few keys most: an H3 1000-row request
    of up to 1003 keys moved to direct for a predicted 1.8% read +10% fp8 query error).
    """
    keep = fx.Float32(1.0 - model[3])
    for j in range(LIMITS):
        cost = _route_cost(
            *totals[4 * j : 4 * j + 4], task_share, step, fixed, union_step, model
        )
        if j == 0:
            best_cost, best = cost * keep, limits[0]
        else:
            better = cost < best_cost
            best_cost = better.select(cost * keep, best_cost)
            best = better.select(limits[j], best)
    return best


@flyc.jit
def _route(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    scratch,
    TILES: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    UNION_STEP: fx.Float32,
    ROUTE_MODEL: fx.Constexpr[tuple],
    CORE: fx.Constexpr[bool],
):
    # Union and direct run back to back, so one long union task or a few direct rows set a
    # latency floor the per-tile rule cannot see. Return the most N64 steps a union tile keeps:
    # every proposal-1 tile, those within max * 2**(-j/2) for j = 1..6, or none (-1).
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    zero = fx.Int32(0)
    union_sum, union_max = zero, zero
    for first in range(tid, TILES, THREADS * ROUTE_FIRST_UNROLL):
        tiles = [fx.Int32(first) + u * THREADS for u in range(ROUTE_FIRST_UNROLL)]
        # All loads of a pass go out before their uses: one memory round trip per pass.
        loads = [
            (COSTS[_min(t, TILES - 1) * 3 + 2], COUNTS[_min(t, TILES - 1)])
            for t in tiles
        ]
        for tile, (route, count) in zip(tiles, loads):
            union = (tile < TILES) & (route == 1)
            steps = union.select((count + 15) >> 4, zero)
            union_sum = union_sum + steps
            union_max = fx.max(union_max, steps)
    union_sum, union_max = _block_reduce(
        [union_sum, union_max], [operator.add, fx.max], scratch, EXCHANGE_WORDS, wave
    )
    limits = [
        fx.Int32(fx.Float32(union_max) * fx.Float32(2.0 ** (-0.5 * j)))
        for j in range(LIMITS - 1)
    ] + [fx.Int32(-1)]
    threshold = union_max
    # Only a union phase set by its longest task, not by throughput, gains from a shorter one.
    if fx.Float32(union_max) > fx.Float32(union_sum) * TASK_SHARE:
        kept_sum, kept_max = [zero] * LIMITS, [zero] * LIMITS
        direct_sum, direct_max = [zero] * LIMITS, [zero] * LIMITS
        for first in range(tid, TILES, THREADS * ROUTE_UNROLL):
            tiles = [fx.Int32(first) + u * THREADS for u in range(ROUTE_UNROLL)]
            loads = []
            for t in tiles:
                at = _min(t, TILES - 1)
                loads.append(
                    (
                        COUNTS[at],
                        COSTS[at * 3],
                        COSTS[at * 3 + 1],
                        COSTS[at * 3 + 2],
                        RESIDUAL[at] if CORE else zero,
                    )
                )
            for tile, (count, direct, row, route, residual) in zip(tiles, loads):
                inside = tile < TILES
                steps = (count + 15) >> 4
                union = route == 1
                # A kept core tile's rows attend their residual blocks in the direct phase.
                direct = inside.select(direct, zero)
                row = inside.select(row, zero)
                residual = inside.select(residual, zero)
                for j in fx.range_constexpr(LIMITS):
                    keep = inside & union & (steps <= limits[j])
                    kept_sum[j] = kept_sum[j] + keep.select(steps, zero)
                    kept_max[j] = fx.max(kept_max[j], keep.select(steps, zero))
                    direct_sum[j] = direct_sum[j] + keep.select(residual, direct)
                    direct_max[j] = fx.max(direct_max[j], keep.select(zero, row))
        # The sort's bins hold the row partials (idle until the sort).
        totals = _pair_totals(
            [
                value
                for j in range(LIMITS)
                for value in (kept_sum[j], kept_max[j], direct_sum[j], direct_max[j])
            ],
            scratch,
            SORT_HIST,
            EXCHANGE_WORDS + 8,
            fx.Int32(tid),
        )
        threshold = _route_choice(
            totals,
            limits,
            TASK_SHARE,
            DIRECT_STEP,
            DIRECT_FIXED,
            UNION_STEP,
            ROUTE_MODEL,
        )
    return threshold


# Order-CTA LDS of the balance, after the sort (it reuses the exchange and route words): each
# partial deal's CTA loads (largest first), each leaf round's slot in its deal, then the rank
# partials, the merged loads, the slot permutation, the snake head's loads and a reduction.
BALANCE_LOADS = 0
BALANCE_SLOTS = (BALANCE_ROUNDS + 1) * BALANCE_GRID
BALANCE_FINAL = 2 * BALANCE_SLOTS
BALANCE_RANKS = 3 * BALANCE_SLOTS
BALANCE_MERGED = BALANCE_RANKS + 4 * BALANCE_GRID
# The re-deals' keys (their rank counts take the first three quarters of BALANCE_RANKS), then
# words of -1 up to 3 * RANK_SPAN in BALANCE_MERGED, free by then.
BALANCE_KEYS = BALANCE_RANKS + 3 * BALANCE_GRID
BALANCE_HEAD = BALANCE_MERGED + BALANCE_GRID
BALANCE_WORDS = BALANCE_HEAD + BALANCE_GRID


@flyc.jit
def _rank_loads(scratch, values, tid, wave):
    """Rank of the load scratch[values + tid] (tid < BALANCE_GRID) among the BALANCE_GRID loads,
    largest first, ties by index (-1 past the grid's loads ranks last). Wave w compares every
    load with a quarter of them as unique keys load << 7 | 127 - k, branch-free; the partial
    ranks meet in LDS behind one barrier (the caller's next barrier frees them)."""
    lane = tid & 63
    quarter = BALANCE_GRID // 4
    keys, ranks = [], []
    for half in fx.range_constexpr(2):
        k = _min(lane + 64 * half, fx.Int32(BALANCE_GRID - 1))
        keys.append((scratch[values + k] << 7) | (127 - k))
        ranks.append(fx.Int32(0))
    for i in fx.range_constexpr(quarter):
        j = wave * quarter + i
        other = (scratch[values + j] << 7) | (127 - j)
        for half in fx.range_constexpr(2):
            # Minus one when the other key is larger (no overflow: loads < 2**24).
            ranks[half] = ranks[half] + ((keys[half] - other) >> 31)
    for half in fx.range_constexpr(2):
        k = lane + 64 * half
        if k < BALANCE_GRID:
            scratch[BALANCE_RANKS + wave * BALANCE_GRID + k] = ranks[half]
    fx.gpu.barrier()
    k = _min(tid, fx.Int32(BALANCE_GRID - 1))
    rank = fx.Int32(0)
    for w in fx.range_constexpr(4):
        rank = rank - scratch[BALANCE_RANKS + w * BALANCE_GRID + k]
    return rank


def _key(load, k):
    """CTA k's rank key of a load (< 2**24): unique, larger loads first, of equal loads the
    higher CTA first."""
    return (load << 7) | k


@flyc.jit
def _rank_keys(scratch, tid):
    """Rank, largest first, of the key BALANCE_KEYS[tid] (tid < BALANCE_GRID) among the
    BALANCE_GRID keys there (-1 for no CTA ranks last; the words past them up to 3 * RANK_SPAN
    hold -1). Three threads a key count the larger ones among RANK_SPAN keys each,
    branch-free; the counts meet in BALANCE_RANKS behind one barrier (the caller's next barrier
    frees them)."""
    part = tid // BALANCE_GRID
    key = scratch[BALANCE_KEYS + tid - part * BALANCE_GRID]
    first = BALANCE_KEYS + _min(part, fx.Int32(2)) * RANK_SPAN
    count = fx.Int32(0)
    for i in fx.range_constexpr(RANK_SPAN):
        # Minus one when the other key is larger (no overflow: keys < 2**31 - 1).
        count = count - ((key - scratch[first + i]) >> 31)
    if part < 3:
        scratch[BALANCE_RANKS + tid] = count
    fx.gpu.barrier()
    k = _min(tid, fx.Int32(BALANCE_GRID - 1))
    return (
        scratch[BALANCE_RANKS + k]
        + scratch[BALANCE_RANKS + BALANCE_GRID + k]
        + scratch[BALANCE_RANKS + 2 * BALANCE_GRID + k]
    )


def _snake(row, column, grid):
    return ((row & 1) == 0).select(column, grid - 1 - column)


def _bin(cost):
    """Counting-sort bin of a cost: larger costs first."""
    return SORT_BINS - 1 - _min(fx.max(cost, fx.Int32(0)), fx.Int32(SORT_BINS - 1))


@flyc.jit
def _sort_positions(costs, scratch, tid, wave, stable):
    """Each entry's position in decreasing cost, and the number of entries and of those of
    cost > 0 (cost -1: no entry). Costs past SORT_BINS - 2 share the first bin. stable (a
    split plan, one entry per thread: position = tid): ties keep the entries' order, since
    which of equal tasks take the split slots and CTAs changes the union kernel's time by a
    few percent from run to run; otherwise ties come in no fixed order."""
    zero = fx.Int32(0)
    base = fx.Int32(fx.ptrtoint(fx.get_iter(scratch)))
    single = stable and len(costs) == 1
    for i in fx.range_constexpr(SORT_BINS // THREADS):
        scratch[SORT_HIST + tid + i * THREADS] = zero
    fx.gpu.barrier()
    within = []
    if fx.const_expr(single):
        # Each wave counts in its own byte of the bin word: the LDS returns a wave's lanes'
        # adds in a fixed order, so a lane's rank among its wave's equal bins is its byte of
        # the old word (other waves' bytes may or may not have counted yet, unused).
        cost = costs[0]
        valid = cost >= 0
        # No entry: add 0 to the lane's own bin word (no lane waits on another's).
        address = base + (SORT_HIST + valid.select(_bin(cost), tid & 63)) * 4
        old = fx.atomic_add(
            _lds_pointer(address, fx.Int32),
            valid.select(fx.Int32(1) << (8 * wave), zero),
            syncscope="workgroup",
        )
        within.append((old >> (8 * wave)) & 255)
    else:
        for cost in costs:
            address = base + (SORT_HIST + _bin(cost)) * 4
            old = zero
            if cost >= 0:
                old = fx.atomic_add(
                    _lds_pointer(address, fx.Int32), fx.Int32(1), syncscope="workgroup"
                )
            within.append(old)
    fx.gpu.barrier()
    # Exclusive scan over the bins, SORT_BINS // THREADS consecutive ones per thread.
    per = SORT_BINS // THREADS
    words = [scratch[SORT_HIST + tid * per + i] for i in range(per)]
    counts = words
    if fx.const_expr(single):
        # Bytes summed in two 16-bit halves.
        halves = [(w & 0x00FF00FF) + ((w >> 8) & 0x00FF00FF) for w in words]
        counts = [(h & 0xFFFF) + (h >> 16) for h in halves]
    local = counts[0]
    for count in counts[1:]:
        local = local + count
    inclusive = _inclusive_scan(local, tid & 63)
    if (tid & 63) == 63:
        scratch[SORT_TOTALS + wave] = inclusive
    fx.gpu.barrier()
    running = inclusive - local
    total = zero
    for w in fx.range_constexpr(THREADS // 64):
        value = scratch[SORT_TOTALS + w]
        running = running + (fx.Int32(w) < wave).select(value, zero)
        total = total + value
    for i in fx.range_constexpr(per):
        start = running
        if fx.const_expr(single):
            # Waves 0-2's counts keep their bytes, the bin's start takes wave 3's (an
            # occupied bin starts below THREADS; the last bin's start goes on the side).
            start = (words[i] & 0xFFFFFF) | (running << 24)
            if fx.const_expr(i == per - 1):
                # The last thread's goes to the side, the others' to the word they rewrite.
                last = tid == THREADS - 1
                scratch[last.select(SORT_TOTALS + 4, SORT_HIST + tid * per + i)] = (
                    running
                )
        scratch[SORT_HIST + tid * per + i] = start
        running = running + counts[i]
    fx.gpu.barrier()
    if fx.const_expr(single):
        word = scratch[SORT_HIST + _bin(costs[0])]
        lower = zero
        for w in fx.range_constexpr(THREADS // 64 - 1):
            lower = lower + (fx.Int32(w) < wave).select((word >> (8 * w)) & 255, zero)
        positions = [((word >> 24) & 255) + lower + within[0]]
        runnable = scratch[SORT_TOTALS + 4]
    else:
        positions = [scratch[SORT_HIST + _bin(c)] + w for c, w in zip(costs, within)]
        runnable = scratch[SORT_HIST + SORT_BINS - 1]
    # Entries of cost > 0 come before the last (cost 0) bin.
    return positions, total, runnable


@flyc.jit
def _rank_merged(scratch, merged, inside, last, a, m, tid, wave, GRID):
    """Sort a merged deal (not the last merge) into deal a, keep its permutation as merge m's;
    returns its spread."""
    spread = fx.Int32(0)
    if ~last:
        if tid < BALANCE_GRID:
            scratch[BALANCE_MERGED + tid] = inside.select(merged, fx.Int32(-1))
        fx.gpu.barrier()
        rank = _rank_loads(scratch, BALANCE_MERGED, tid, wave)
        if inside:
            scratch[BALANCE_LOADS + a * BALANCE_GRID + rank] = merged
            scratch[BALANCE_SLOTS + (m + 1) * BALANCE_GRID + tid] = rank
        fx.gpu.barrier()
        deal = BALANCE_LOADS + a * BALANCE_GRID
        spread = _uniform(scratch[deal]) - _uniform(scratch[deal + GRID - 1])
    return spread


@flyc.jit
def _balance(
    scratch, costs, places, runnable, tid, wave, TASKS, GRID, OVERHEAD, DESCENTS
):
    """Balance the rounds of the sorted entries (see BALANCE_ROUNDS): BALANCE_FINAL holds each
    leaf deal's CTA by slot; returns (head, rounds) for _balanced_cta. OVERHEAD: a task's
    cost past its steps, in quarter steps. A plan of DESCENT_ROUNDS rounds of its TASKS has
    its leaf deals, its rounds, re-dealt by _descend instead of merged: a uniform branch on
    the kernel arguments, not a variant of its own, which a process would trace and compile
    when it first meets such a plan (often long after its warm-up). DESCENTS: whether the
    chunk can hold such a plan (else no _descend)."""
    zero = fx.Int32(0)
    loads = [(cost > 0).select(4 * cost + OVERHEAD, zero) for cost in costs]
    rounds = (runnable + GRID - 1) // GRID
    # Up to three rounds the snake already pairs the longest tasks with the shortest.
    rounds = (rounds > 3).select(rounds, zero)
    descend = fx.Boolean(False)
    if fx.const_expr(DESCENTS):
        # ceil(TASKS / GRID) == DESCENT_ROUNDS; then the rounds are DESCENT_ROUNDS unless
        # some tasks route direct (then fewer).
        above = TASKS > (DESCENT_ROUNDS - 1) * GRID
        descend = above & (TASKS <= DESCENT_ROUNDS * GRID)
        rounds = (descend & (rounds != DESCENT_ROUNDS)).select(zero, rounds)
    tail = (rounds >= BALANCE_LONG).select(
        fx.Int32(BALANCE_ROUNDS - 1), fx.Int32(BALANCE_ROUNDS)
    )
    # (0 for a descent plan: DESCENT_ROUNDS = BALANCE_ROUNDS rounds or none.)
    head = fx.max(rounds - tail, zero)
    if rounds > 0:
        if fx.const_expr(DESCENTS):
            if descend:
                _descend(scratch, loads, places, tid, GRID)
            else:
                _deal(scratch, loads, places, rounds, head, tid, wave, GRID)
        else:
            _deal(scratch, loads, places, rounds, head, tid, wave, GRID)
    return head, rounds


@flyc.jit
def _deal(scratch, loads, places, rounds, head, tid, wave, GRID):
    """_balance's merges: BALANCE_FINAL by leaf deal and slot."""
    zero = fx.Int32(0)
    base = fx.Int32(fx.ptrtoint(fx.get_iter(scratch)))
    fx.gpu.barrier()
    if tid < BALANCE_GRID:
        scratch[BALANCE_HEAD + tid] = (tid < GRID).select(zero, fx.Int32(-1))
    # Tail rounds hold every column (positions past the entries cost nothing).
    for i in fx.range_constexpr(BALANCE_ROUNDS * BALANCE_GRID // THREADS + 1):
        index = tid + i * THREADS
        if index < BALANCE_ROUNDS * BALANCE_GRID:
            scratch[BALANCE_LOADS + BALANCE_GRID + index] = zero
    fx.gpu.barrier()
    # Leaf deals: 0 the snake head (its CTA loads by snake column), 1.. one tail round each.
    program, size, positions, inverse = places
    for load, position in zip(loads, positions):
        row, column = _place(program, size, position, inverse, GRID)
        if row < head:
            snake = _snake(row, column, GRID)
            address = base + (BALANCE_HEAD + snake) * 4
            fx.atomic_add(_lds_pointer(address, fx.Int32), load, syncscope="workgroup")
        elif row < rounds:
            scratch[BALANCE_LOADS + (row - head + 1) * BALANCE_GRID + column] = load
    fx.gpu.barrier()
    inside = tid < GRID
    if head > 0:
        # Sort the head deal: snake column c's slot (the first step of its final CTA).
        rank = _rank_loads(scratch, BALANCE_HEAD, tid, wave)
        load = scratch[BALANCE_HEAD + _min(tid, GRID - 1)]
        fx.gpu.barrier()
        if inside:
            scratch[BALANCE_LOADS + rank] = load
            scratch[BALANCE_SLOTS + tid] = rank
        fx.gpu.barrier()
    deals = BALANCE_ROUNDS + 1
    alive = [head > 0] + [j <= rounds - head for j in range(1, deals)]
    spread = []
    for d in fx.range_constexpr(deals):
        deal = BALANCE_LOADS + d * BALANCE_GRID
        width = _uniform(scratch[deal]) - _uniform(scratch[deal + GRID - 1])
        spread.append(alive[d].select(width, fx.Int32(-1)))
    owner = [fx.Int32(j) for j in range(deals)]
    live = alive[0].select(fx.Int32(1), zero) + (rounds - head)
    history = []
    for m in fx.range_constexpr(BALANCE_ROUNDS):
        # Merge the two live deals of largest spread: deal a's slot k with deal b's GRID-1-k.
        a, best, b, second = zero, fx.Int32(-1), zero, fx.Int32(-1)
        for d in fx.range_constexpr(deals):
            first = spread[d] > best
            take = ~first & (spread[d] > second)
            b = first.select(a, take.select(fx.Int32(d), b))
            second = first.select(best, take.select(spread[d], second))
            a = first.select(fx.Int32(d), a)
            best = first.select(spread[d], best)
        merge = live > 1
        # The last merge's pairs are the CTAs: slot k of the merged deal, no sort.
        last = live == 2
        next_spread = zero
        if merge:
            pair = _min(tid, GRID - 1)
            merged = (
                scratch[BALANCE_LOADS + a * BALANCE_GRID + pair]
                + scratch[BALANCE_LOADS + b * BALANCE_GRID + GRID - 1 - pair]
            )
            if fx.const_expr(m + 1 < BALANCE_ROUNDS):
                next_spread = _rank_merged(
                    scratch, merged, inside, last, a, m, tid, wave, GRID
                )
        history.append((merge, last, a, b, list(owner)))
        for d in fx.range_constexpr(deals):
            spread[d] = (merge & (fx.Int32(d) == b)).select(fx.Int32(-1), spread[d])
            spread[d] = (merge & (fx.Int32(d) == a)).select(next_spread, spread[d])
            owner[d] = (merge & (owner[d] == b)).select(a, owner[d])
        live = live - merge.select(fx.Int32(1), zero)
    # Each leaf deal's final CTA by slot (head: by snake column), followed through the merges.
    for i in fx.range_constexpr(deals * BALANCE_GRID // THREADS + 1):
        index = tid + i * THREADS
        leaf = _min(index // BALANCE_GRID, fx.Int32(deals - 1))
        column = index - leaf * BALANCE_GRID
        slot = _min(column, GRID - 1)
        slot = (leaf == 0).select(scratch[BALANCE_SLOTS + slot], slot)
        for m, (merge, last, a, b, owners) in enumerate(history):
            deal = owners[0]
            for j in fx.range_constexpr(1, deals):
                deal = (leaf == j).select(owners[j], deal)
            slot = (merge & (deal == b)).select(GRID - 1 - slot, slot)
            moved = merge & ~last & ((deal == a) | (deal == b))
            at = BALANCE_SLOTS + (m + 1) * BALANCE_GRID + _min(slot, GRID - 1)
            slot = moved.select(scratch[at], slot)
        if index < deals * BALANCE_GRID:
            scratch[BALANCE_FINAL + index] = slot
    fx.gpu.barrier()


@flyc.jit
def _descend(scratch, loads, places, tid, GRID):
    """_balance of DESCENT_ROUNDS rounds (head 0, leaf deal r + 1 round r): deal the rounds in a
    snake, then re-deal each, last first, DESCENT_SWEEPS times; BALANCE_FINAL by leaf deal and
    slot. BALANCE_LOADS holds each round's loads by slot (largest first); CTA c's thread c
    keeps its slot and load of each round in registers."""
    zero = fx.Int32(0)
    inside = tid < GRID
    cta = _min(tid, GRID - 1)
    fx.gpu.barrier()
    # Rounds hold every column (positions past the entries cost nothing).
    for i in fx.range_constexpr(DESCENT_ROUNDS * BALANCE_GRID // THREADS + 1):
        index = tid + i * THREADS
        if index < DESCENT_ROUNDS * BALANCE_GRID:
            scratch[BALANCE_LOADS + BALANCE_GRID + index] = zero
    if tid < 3 * RANK_SPAN - BALANCE_GRID:
        scratch[BALANCE_KEYS + BALANCE_GRID + tid] = fx.Int32(-1)
    fx.gpu.barrier()
    program, size, positions, inverse = places
    for load, position in zip(loads, positions):
        row, column = _place(program, size, position, inverse, GRID)
        if row < DESCENT_ROUNDS:
            scratch[BALANCE_LOADS + (row + 1) * BALANCE_GRID + column] = load
    fx.gpu.barrier()
    # The snake: CTA c holds slot c of the even rounds, GRID - 1 - c of the odd ones.
    slots = [cta if r % 2 == 0 else GRID - 1 - cta for r in range(DESCENT_ROUNDS)]
    owns = [
        scratch[BALANCE_LOADS + (r + 1) * BALANCE_GRID + slots[r]]
        for r in range(DESCENT_ROUNDS)
    ]
    total = owns[0]
    for own in owns[1:]:
        total = total + own
    # The sweeps loop (their rounds unrolled): the narrow sorts carry this beside _deal, and
    # unrolled it lengthens K3's code and its trace (a 286-task plan's K3: +5% with K3's code
    # cold, as in a call).
    for _ in range(fx.Int32(0), fx.Int32(DESCENT_SWEEPS), fx.Int32(1)):
        for r in fx.range_constexpr(DESCENT_ROUNDS - 1, -1, -1):
            others = total - owns[r]
            if tid < BALANCE_GRID:
                scratch[BALANCE_KEYS + tid] = inside.select(
                    _key(others, tid), fx.Int32(-1)
                )
            fx.gpu.barrier()
            # Least loaded first: slot GRID - 1 - rank of the others' loads, of equal ones the
            # lower CTA first (after the snake many are equal; this order keeps its pairs).
            rank = _rank_keys(scratch, tid)
            slots[r] = _min(fx.max(GRID - 1 - rank, zero), GRID - 1)
            owns[r] = scratch[BALANCE_LOADS + (r + 1) * BALANCE_GRID + slots[r]]
            total = others + owns[r]
    if inside:
        for r in fx.range_constexpr(DESCENT_ROUNDS):
            scratch[BALANCE_FINAL + (r + 1) * BALANCE_GRID + slots[r]] = tid
    fx.gpu.barrier()


def _balanced_cta(scratch, state, row, column, grid):
    """The CTA of sorted position (row, column); rounds past the last runnable task go in a
    snake."""
    head, rounds = state
    snake = _snake(row, column, grid)
    tail = row >= head
    leaf = tail.select(row - head + 1, fx.Int32(0))
    slot = tail.select(column, snake)
    index = _min(leaf, fx.Int32(BALANCE_ROUNDS)) * BALANCE_GRID + slot
    return (row < rounds).select(scratch[BALANCE_FINAL + index], snake)


def _entry(program, SIZE, tid, WIDTH, e, TASKS, SPLIT, HK, SLICES):
    """(rank, valid, task, part, tile) of this thread's entry e of the chunk (recomputed where
    needed, so no entry keeps them in registers). Entry e of thread t is position
    e * THREADS + t: a wave's entries are consecutive tasks, so their plan-word loads and
    ACTIVE stores coalesce and meet the route's first pass's loads in cache. A split plan's
    threads hold consecutive positions instead: its slot scan runs in task order."""
    position = e * THREADS + tid
    if fx.const_expr(SPLIT > 1):
        position = tid * WIDTH + e
    rank = program * SIZE + position
    valid = (position < SIZE) & (rank < TASKS * SPLIT)
    task, part = rank, fx.Int32(0)
    if fx.const_expr(SPLIT > 1):
        task, part = rank % TASKS, rank // TASKS
    if fx.const_expr(HK == 1):
        # Past the last entry any tile serves: the entry is invalid.
        tile = _min(task, TASKS - 1) // SLICES
    else:
        tile = (task % (TASKS // HK)) // SLICES
    return rank, valid, task, part, tile


def _place(program, SIZE, position, inverse, GRID):
    """(ORDER row, column) of a chunk's sorted position, by a reciprocal and one correction."""
    place = program * SIZE + position
    row = fx.Int32(place.to(fx.Float32) * inverse)
    row = row - (row * GRID > place).select(fx.Int32(1), fx.Int32(0))
    row = row + ((row + 1) * GRID <= place).select(fx.Int32(1), fx.Int32(0))
    return row, place - row * GRID


@flyc.jit
def _order(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    COUNTERS: fx.Tensor,
    scratch,
    program,
    threshold,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    WIDTH: fx.Constexpr[int],
    SPLIT: fx.Constexpr[int],
    SPLIT_PAIRS: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    BALANCE: fx.Constexpr[int],
    DESCENTS: fx.Constexpr[bool],
    CHUNKS,
    runs,
    idles,
):
    """Order chunk `program` (SIZE entries) of the order CTA. One chunk places its sorted
    entries at the start of ORDER; with several, each chunk's runnable entries go after the
    previous chunks' (runs placed so far) and its idle ones before theirs at ORDER's end
    (idles), so each CTA still meets its runnable tasks first. Returns the new runs, idles.
    """
    tid = fx.thread_idx.x
    lane, wave = tid & 63, _uniform(tid >> 6)
    per_head = TASKS // HK
    zero = fx.Int32(0)

    # Entry rank r: task r % TASKS (hkv-major), key-range part r // TASKS.
    def entry(e):
        return _entry(program, SIZE, tid, WIDTH, e, TASKS, SPLIT, HK, SLICES)

    # Entries' plan words load ORDER_LOADS at a time before their ACTIVE stores (loads may not
    # pass a store that could alias them): a memory round trip per group, not per entry.
    # Each entry keeps its cost: N64 steps if it runs union, 0 if routed direct, -1 past the
    # chunk's entries (a split plan also its steps and route).
    costs, info = [], []
    raised = fx.Boolean(False)
    for first in fx.range_constexpr(0, WIDTH, ORDER_LOADS):
        group = range(first, min(first + ORDER_LOADS, WIDTH))
        words = []
        for e in group:
            tile = entry(e)[4]
            words.append(
                (COUNTS[tile], COSTS[tile * 3 + 2], RESIDUAL[tile] if CORE else None)
            )
        for e, (count, route, residual) in zip(group, words):
            _, valid, task, part, tile = entry(e)
            steps = (valid.select(count, zero) + 15) >> 4
            union = valid.select(route, zero) == 1
            active = union & (steps <= threshold)
            # The first head's first slice of each tile publishes the route; any direct tile,
            # or union tile with residual blocks, raises the flag compact cleared, which gated
            # pack/direct launches read (one store per thread, after the route stores).
            publish = valid & (part == 0) & (task < per_head) & (task % SLICES == 0)
            direct = ~active
            if fx.const_expr(CORE):
                direct = direct | (residual > 0)
            raised = raised | (publish & direct)
            if publish:
                ACTIVE[tile] = active.select(fx.Int32(1), zero)
            costs.append(valid.select(active.select(steps, zero), fx.Int32(-1)))
            if fx.const_expr(SPLIT > 1):
                info.append((valid, part, steps, active))
    if raised:
        DIRECT_FLAG[0] = 1
    if fx.const_expr(SPLIT > 1):
        # Even per-CTA load of the active tasks (steps plus a task overhead), then the tasks
        # above it take a slot (task order, at most SPLIT_SLOTS) and SPLIT key ranges.
        # SPLIT_PAIRS: a round deals an item per CTA and the next pairs the rest with the
        # shortest ones, so the parts past a round take at most the tasks of at most half
        # the longest's steps (64 tasks of 33 steps on 80 CTAs split 16: a pair of parts
        # would run as long as a task, the parts' records and merges added).
        base = ORDER_WORDS
        table = ORDER_WORDS + SPLIT_TABLE
        total, count, longest = zero, zero, zero
        for valid, part, steps, active in info:
            first = valid & (part == 0) & active
            total = total + first.select(steps, zero)
            count = count + first.select(fx.Int32(1), zero)
            longest = fx.max(longest, first.select(steps, zero))
        slots = fx.Int32(SPLIT_SLOTS)
        if fx.const_expr(SPLIT_PAIRS):
            total, count, longest = _block_reduce(
                [total, count, longest],
                [operator.add, operator.add, fx.max],
                scratch,
                base,
                wave,
            )
            short = zero
            for valid, part, steps, active in info:
                first = valid & (part == 0) & active
                short = short + (first & (2 * steps <= longest)).select(
                    fx.Int32(1), zero
                )
            (short,) = _block_reduce([short], [operator.add], scratch, base + 12, wave)
            slots = _min(slots, GRID + short - count)
        else:
            total, count = _block_reduce(
                [total, count], [operator.add, operator.add], scratch, base, wave
            )
        even = (
            total.to(fx.Float32) + fx.Float32(SPLIT_OVERHEAD) * count.to(fx.Float32)
        ) / GRID.to(fx.Float32)
        wanted = []
        for valid, part, steps, active in info:
            wanted.append(
                valid
                & (part == 0)
                & active
                & (steps >= 2)
                & (steps.to(fx.Float32) > even)
            )
        local = zero
        for w in wanted:
            local = local + w.select(fx.Int32(1), zero)
        fx.gpu.barrier()
        inclusive = _inclusive_scan(local, lane)
        if lane == 63:
            scratch[base + wave] = inclusive
        fx.gpu.barrier()
        running = inclusive - local
        for w in fx.range_constexpr(THREADS // 64):
            running = running + (fx.Int32(w) < wave).select(scratch[base + w], zero)
        for e in fx.range_constexpr(WIDTH):
            valid, part, steps, active = info[e]
            task = entry(e)[2]
            taken = wanted[e] & (running < slots)
            if valid & (part == 0):
                scratch[table + task] = taken.select(running, fx.Int32(-1))
            running = running + wanted[e].select(fx.Int32(1), zero)
        fx.gpu.barrier()
        keys = []
        for e in fx.range_constexpr(WIDTH):
            valid, part, steps, active = info[e]
            owner = valid.select(scratch[table + entry(e)[2]], fx.Int32(-1))
            split = owner >= 0
            # Part 0 of a split task runs the first ceil(steps / 2) N64 steps; part 1 of a
            # task not split is no entry.
            cost = split.select((steps + 1 - part) >> 1, costs[e])
            costs[e] = (valid & ((part == 0) | split)).select(cost, fx.Int32(-1))
            # Sort key 2 * steps + 1 for a split task's parts: of equal steps they deal
            # first, since a part costs more than a whole task of its steps (its record, and
            # the merge of the part counting last), and the snake's second round then pairs it
            # with a shorter task (a 2048-token prefill of every causal block, H6: BF16 union
            # -2.2%).
            keys.append(
                (costs[e] >= 0).select(
                    2 * costs[e] + split.select(fx.Int32(1), zero), fx.Int32(-1)
                )
            )
        if tid < SPLIT_SLOTS:
            COUNTERS[tid] = 0
    positions, count, runnable = _sort_positions(
        keys if SPLIT > 1 else costs, scratch, tid, wave, SPLIT > 1
    )
    inverse = fx.Float32(1.0) / GRID.to(fx.Float32)
    state = None
    if fx.const_expr(BALANCE):
        places = (program, SIZE, positions, inverse)
        state = _balance(
            scratch, costs, places, runnable, tid, wave, TASKS, GRID, BALANCE, DESCENTS
        )

    def cta(row, column):
        if fx.const_expr(BALANCE):
            return _balanced_cta(scratch, state, row, column, GRID)
        return _snake(row, column, GRID)

    # Publish the tile-major task the union kernel runs at each sorted entry's place, idle
    # entries past them.
    rounded = (TASKS * SPLIT + GRID - 1) // GRID * GRID
    multi = CHUNKS > 1
    for e in fx.range_constexpr(WIDTH):
        task, part = entry(e)[2], entry(e)[3]
        if fx.const_expr(HK == 1):
            value = task
        else:
            value = (task % per_head) * HK + task // per_head
        if fx.const_expr(SPLIT > 1):
            # (task, slot, part, split): the slot table outlived the sort.
            owner = scratch[
                ORDER_WORDS + SPLIT_TABLE + _min(task, fx.Int32(SPLIT_TASKS - 1))
            ]
            value = (
                (value << 8)
                | ((owner >= 0).select(owner, zero) << 2)
                | (part << 1)
                | (owner >= 0).select(fx.Int32(1), zero)
            )
        position = positions[e]
        if fx.const_expr(not BALANCE):
            position = multi.select(
                (position < runnable).select(
                    runs + position, rounded - 1 - idles - (position - runnable)
                ),
                program * SIZE + position,
            )
        else:
            position = program * SIZE + position
        row, column = _place(zero, SIZE, position, inverse, GRID)
        destination = row * GRID + cta(row, column)
        if (costs[e] >= 0) & (destination < rounded):
            ORDER[destination] = value
    if ~multi:
        for spare in range(count + tid, SIZE, THREADS):
            idle_row, idle_column = _place(
                program, SIZE, fx.Int32(spare), inverse, GRID
            )
            idle = idle_row * GRID + cta(idle_row, idle_column)
            if idle < rounded:
                ORDER[idle] = -1
    # The scratch the next chunk reuses: every thread has read its bins and loads.
    fx.gpu.barrier()
    return runs + runnable, idles + (count - runnable)


def _claim(flag, count):
    """Atomically add count to flag[1]; returns the previous value."""
    return fx.atomic_add(fx.get_iter(flag) + 1, fx.Int32(count), syncscope="agent")


@flyc.jit
def _direct_groups(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    scratch,
    chunk,
    threshold,
    GROUPS: fx.Int32,
):
    """Append this CTA's raw direct row groups that hold a row of a direct tile to GROUP_LIST
    (count in DIRECT_FLAG[1]), in group order within the CTA."""
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    group = chunk * THREADS + tid
    index = _min(group, GROUPS - 1)
    first, rows = GROUP_META[index * 5], GROUP_META[index * 5 + 1]
    direct = fx.Int32(0)
    # A raw group holds up to four rows; the route matches _order's.
    for local in fx.range_constexpr(4):
        tile = QUERY_TILES[first + _min(fx.Int32(local), rows - 1)]
        steps = (COUNTS[tile] + 15) >> 4
        active = (COSTS[tile * 3 + 2] == 1) & (steps <= threshold)
        direct = direct | active.select(fx.Int32(0), fx.Int32(1))
    listed = (group < GROUPS) & (direct != 0)
    mask = _ballot(listed)
    scratch[wave] = _count(mask)
    fx.gpu.barrier()
    totals = [scratch[w] for w in range(THREADS // 64)]
    total = totals[0]
    for w in fx.range_constexpr(1, THREADS // 64):
        total = total + totals[w]
    if tid == 0:
        scratch[THREADS // 64] = _claim(DIRECT_FLAG, total)
    fx.gpu.barrier()
    base = scratch[THREADS // 64]
    for w in fx.range_constexpr(THREADS // 64 - 1):
        base = base + (fx.Int32(w) < wave).select(totals[w], fx.Int32(0))
    if listed:
        GROUP_LIST[base + _below(mask)] = group


RANK_TILES = 8  # tiles per thread and pass of _rank_direct_tiles


@flyc.jit
def _rank_direct_tiles(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    TILE_RANKS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    scratch,
    threshold,
    UNION_TILES: fx.Int32,
    TILES: fx.Int32,
):
    """Each direct tile's rank among them in tile order into TILE_RANKS, their number into
    DIRECT_FLAG[2]: the order of the packed direct launch's slots (its pack lists the tiles
    _order publishes inactive), whose split raw direct repeats."""
    tid = fx.thread_idx.x
    wave = _uniform(tid >> 6)
    zero = fx.Int32(0)
    base = zero
    # The scratch _route read is free.
    fx.gpu.barrier()
    for start in range(zero, TILES, fx.Int32(THREADS * RANK_TILES)):
        tiles = [start + tid * RANK_TILES + i for i in fx.range_constexpr(RANK_TILES)]
        words = [
            (
                COUNTS[_min(tile, UNION_TILES - 1)],
                COSTS[_min(tile, UNION_TILES - 1) * 3 + 2],
            )
            for tile in tiles
        ]
        direct = []
        for tile, (count, route) in zip(tiles, words):
            # _order's route; tiles past the union candidates are direct.
            active = (
                (tile < UNION_TILES) & (route == 1) & (((count + 15) >> 4) <= threshold)
            )
            direct.append((tile < TILES) & ~active)
        below, total = zero, zero
        for flag in direct:
            mask = _ballot(flag)
            below = below + _below(mask)
            total = total + _count(mask)
        scratch[wave] = total
        fx.gpu.barrier()
        totals = [scratch[w] for w in range(THREADS // 64)]
        fx.gpu.barrier()
        position = base + below
        for w in fx.range_constexpr(THREADS // 64 - 1):
            position = position + (fx.Int32(w) < wave).select(totals[w], zero)
        for tile, flag in zip(tiles, direct):
            if flag:
                TILE_RANKS[tile] = position
            position = position + flag.select(fx.Int32(1), zero)
        for w in fx.range_constexpr(THREADS // 64):
            base = base + totals[w]
    if tid == 0:
        DIRECT_FLAG[2] = base


# Mask entry layout: slot s of a group sits in quarter (s % 8) // 2, nibble (s // 8) * 2 + s % 2;
# quarters 0, 1 form the entry's first word, 2, 3 its second.
MASK_WORD = tuple(((s % 8) // 2) >> 1 for s in range(16))
MASK_SHIFT = tuple(
    16 * (((s % 8) // 2) & 1) + 4 * ((s // 8) * 2 + s % 2) for s in range(16)
)
MASK_BITS = 0x11111111
# A mask CTA stages THREADS union slots (MASK_GROUPS groups of 16) at a time in LDS, two
# buffers of: per group its membership words (64-bit: the 16 low words, then the 16 high),
# padded by a word so the groups fall in different banks; per group its block ids, padded
# likewise; per group whether some block of it is not wholly visible to the tile's first query.
MASK_GROUPS = THREADS // 16
MASK_MEMBERS = 33
MASK_BLOCKS = MASK_GROUPS * MASK_MEMBERS
MASK_LATE = MASK_BLOCKS + MASK_GROUPS * 17
MASK_BUFFER = MASK_LATE + MASK_GROUPS
MASK_WORDS = 2 * MASK_BUFFER
# CTAs that share a wide (BQ21/BQ42) tile's mask chunks when they all fit one round (eight
# K3 CTAs a CU): its heaviest tiles otherwise end K3 alone (a 12000-token prefill at H3: one
# 13 us CTA); with more tiles the extra CTAs' fixed latency costs more (the same at H6: +4 us).
MASK_PARTS = 2


@flyc.jit
def _mask_stage(members, blocks, count, cap, slot, members64):
    """(membership words, block id) of a union slot; zero past the tile's union (loads past a
    buffer's end return zero)."""
    inside = slot < count
    if fx.const_expr(members64):
        pair = _buffer_words(members, inside.select(slot * 8, cap * 8), 2)
        words = [pair[0], pair[1]]
    else:
        words = [_word(members, inside.select(slot * 4, cap * 4))]
    block = _word(blocks, inside.select(slot * 4, cap * 4))
    return words, block


@flyc.jit
def _pool_block(SLOTS, POOL_BLOCKS, kv_start, block, index, live):
    """Pool block of a live union slot (POOL_BLOCKS[index]): its block's four tokens occupy
    four aligned pool rows."""
    if live:
        POOL_BLOCKS[index] = fx.Int32(SLOTS[kv_start + block * 4]) >> 2


# Nibble j of MASK_VISIBLE >> 4 * (d + 3) (-3 <= d <= 4) is (1 << clamp(d + j, 0, 4)) - 1: the
# tokens of a block that query j sees when query 0 sees d of them (none below -3, all past 4).
MASK_VISIBLE = sum(((1 << min(max(i - 3, 0), 4)) - 1) << (4 * i) for i in range(11))


@flyc.jit
def _mask_late(scratch, buffer, group, entries, first, position):
    """_mask_quad's entries of a group with a block not wholly visible to the tile's first
    query: each slot's nibble keeps the tokens of its block query first + j sees (a valid
    row's causal window ends before its request's)."""
    zero = fx.Int32(0)
    visible = position + first + 1
    seen = [[zero, zero] for _ in range(4)]
    # A slot's four queries' nibbles from one shift of MASK_VISIBLE (slot by slot: each block
    # start used at once).
    for s in fx.range_constexpr(16):
        start = scratch[buffer + MASK_BLOCKS + group * 17 + s] * 4
        shift = (fx.min(fx.max(visible - start, fx.Int32(-3)), fx.Int32(4)) + 3) * 4
        nibbles = fx.Int32(
            (fx.Int64(MASK_VISIBLE) >> fx.Int64(shift)) & fx.Int64(0xFFFF)
        )
        for j in fx.range_constexpr(4):
            word = MASK_WORD[s]
            nibble = (nibbles >> (4 * j)) & 15
            seen[j][word] = seen[j][word] | (nibble << MASK_SHIFT[s])
    return [entries[2 * j + w] & seen[j][w] for j in range(4) for w in range(2)]


@flyc.jit
def _mask_quad(scratch, buffer, group, quad, rows, position, members64):
    """The mask entries (two words each) of queries 4 * quad + j (j < 4) of a staged group.
    Each slot's four member bits of those queries go to its nibble once; entry j then takes
    bit j of every nibble (times 0xF: the block is wholly visible)."""
    first = quad * 4
    zero = fx.Int32(0)
    base = buffer + group * MASK_MEMBERS
    if fx.const_expr(members64):
        base = base + (first >= 32).select(fx.Int32(16), zero)
    shift = first & 31
    nibbles = [zero, zero]
    for s in fx.range_constexpr(16):
        nibble = (scratch[base + s] >> shift) & 15
        nibbles[MASK_WORD[s]] = nibbles[MASK_WORD[s]] | (nibble << MASK_SHIFT[s])
    e = []
    for j in fx.range_constexpr(4):
        ok = first + j < rows
        for w in fx.range_constexpr(2):
            bits = (nibbles[w] >> j) & MASK_BITS
            e.append(ok.select((bits << 4) - bits, zero))
    e0, e1, e2, e3, e4, e5, e6, e7 = e
    if scratch[buffer + MASK_LATE + group] != 0:
        e0, e1, e2, e3, e4, e5, e6, e7 = _mask_late(
            scratch, buffer, group, e, first, position
        )
    return e0, e1, e2, e3, e4, e5, e6, e7


@flyc.jit
def _masks(
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    MASKS: fx.Tensor,
    SLOTS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    scratch,
    tile,
    part,
    QB: fx.Int32,
    CAP: fx.Int32,
    SLOTTED: fx.Constexpr[bool],
    MEMBERS64: fx.Constexpr[bool],
    PARTS: fx.Constexpr[int],
):
    # Entry (nt, query) holds 4 bits for each slot nt * 16 + s (MASK_WORD / MASK_SHIFT). The
    # CTA stages THREADS slots at a time in LDS (coalesced, the next chunk's loads in flight
    # while this one is built). A thread then builds the entries of four queries of a group
    # (16 member loads for four entries) and stores them as 32 contiguous bytes next to the
    # other queries'. With a slot table the staging threads also map their blocks to pool
    # blocks (a four-token block occupies four aligned pool rows).
    tid = fx.Int32(fx.thread_idx.x)
    if _uniform(COSTS[tile * 3 + 2]) != 0:
        count = _uniform(COUNTS[tile])
        rows = _uniform(META[tile * 5 + 1])
        length = _uniform(META[tile * 5 + 3])
        position = _uniform(META[tile * 5 + 4])
        size = 8 if MEMBERS64 else 4
        blocks = _resource(BLOCKS, CAP * 4, fx.Int64(tile) * fx.Int64(CAP))
        members = _resource(MEMBERSHIP, CAP * size, fx.Int64(tile) * fx.Int64(CAP))
        entries = (CAP >> 4) * QB
        masks = _resource(MASKS, entries * 8, fx.Int64(tile) * fx.Int64(entries * 4))
        kv_start = fx.Int32(0)
        if fx.const_expr(SLOTTED):
            kv_start = _uniform(META[tile * 5 + 2])
            if (tid == 0) & (part == 0):
                # Pool row of the final partial block's first token, -1 without one.
                tail = fx.Int32(SLOTS[kv_start + (((length - 1) >> 2) << 2)])
                TAILS[tile] = ((length & 3) != 0).select(tail, fx.Int32(-1))
        groups = (count + 15) >> 4
        chunks = (count + THREADS - 1) // THREADS
        # At most 16 quads per group (QB <= 64): one (group, quad) item per thread a chunk.
        quads = _uniform((QB + 3) >> 2)
        group = tid // quads
        quad = tid - group * quads
        lane = tid & 63
        stage_group, stage_slot = tid >> 4, tid & 15
        # Part p of a tile's CTAs takes chunks p, p + PARTS, ...
        words, block = _mask_stage(
            members, blocks, count, CAP, part * THREADS + tid, MEMBERS64
        )
        for chunk in range(part, chunks, fx.Int32(PARTS)):
            chunk = fx.Int32(chunk)
            buffer = (((chunk - part) // PARTS) & 1) * MASK_BUFFER
            slot = chunk * THREADS + tid
            column = buffer + stage_group * MASK_MEMBERS + stage_slot
            for w in fx.range_constexpr(len(words)):
                scratch[column + 16 * w] = words[w]
            scratch[buffer + MASK_BLOCKS + stage_group * 17 + stage_slot] = block
            late = _ballot((slot < count) & (block * 4 + 4 > position + 1))
            if stage_slot == 0:
                bits = fx.Int32(late >> fx.Int64(lane)) & 0xFFFF
                scratch[buffer + MASK_LATE + stage_group] = bits
            if fx.const_expr(SLOTTED):
                live = slot < count
                _pool_block(
                    SLOTS, POOL_BLOCKS, kv_start, block, tile * CAP + slot, live
                )
            fx.gpu.barrier()
            words, block = _mask_stage(
                members, blocks, count, CAP, slot + THREADS * PARTS, MEMBERS64
            )
            first = chunk * MASK_GROUPS
            if group < _min(groups - first, fx.Int32(MASK_GROUPS)):
                e = _mask_quad(scratch, buffer, group, quad, rows, position, MEMBERS64)
                offset = ((first + group) * QB + quad * 4) * 8
                for pair in fx.range_constexpr(2):
                    query = quad * 4 + 2 * pair
                    if query + 1 < QB:
                        _buffer_store_words(
                            fx.Vector.from_elements(
                                list(e[4 * pair : 4 * pair + 4]), fx.Int32
                            ),
                            masks,
                            offset + 16 * pair,
                        )
                    elif query < QB:
                        _buffer_store_words(
                            fx.Vector.from_elements(
                                list(e[4 * pair : 4 * pair + 2]), fx.Int32
                            ),
                            masks,
                            offset + 16 * pair,
                        )


@flyc.jit
def _order_chunks(
    COUNTS,
    COSTS,
    RESIDUAL,
    ACTIVE,
    ORDER,
    DIRECT_FLAG,
    COUNTERS,
    scratch,
    threshold,
    TASKS,
    GRID,
    SIZE,
    CHUNKS,
    HK,
    SLICES,
    WIDTH,
    SPLIT,
    SPLIT_PAIRS,
    CORE,
    BALANCE,
    DESCENTS,
):
    """The order CTA: every chunk of the task entries in turn (see _order), then idle
    entries between the runnable and the idle ones of several chunks."""
    runs, idles = fx.Int32(0), fx.Int32(0)
    if fx.const_expr(BALANCE):
        # One chunk (no loop: the balance's bookkeeping lives in scalar registers).
        _order(
            COUNTS,
            COSTS,
            RESIDUAL,
            ACTIVE,
            ORDER,
            DIRECT_FLAG,
            COUNTERS,
            scratch,
            fx.Int32(0),
            threshold,
            TASKS,
            GRID,
            SIZE,
            HK,
            SLICES,
            WIDTH,
            SPLIT,
            SPLIT_PAIRS,
            CORE,
            BALANCE,
            DESCENTS,
            CHUNKS,
            runs,
            idles,
        )
        return
    for chunk in range(fx.Int32(0), CHUNKS, fx.Int32(1)):
        runs, idles = _order(
            COUNTS,
            COSTS,
            RESIDUAL,
            ACTIVE,
            ORDER,
            DIRECT_FLAG,
            COUNTERS,
            scratch,
            fx.Int32(chunk),
            threshold,
            TASKS,
            GRID,
            SIZE,
            HK,
            SLICES,
            WIDTH,
            SPLIT,
            SPLIT_PAIRS,
            CORE,
            BALANCE,
            DESCENTS,
            CHUNKS,
            runs,
            idles,
        )
    if CHUNKS > 1:
        rounded = (TASKS * SPLIT + GRID - 1) // GRID * GRID
        for gap in range(runs + fx.thread_idx.x, rounded - idles, THREADS):
            gap = fx.Int32(gap)
            row = gap // GRID
            column = gap - row * GRID
            ORDER[row * GRID + _snake(row, column, GRID)] = -1


@flyc.kernel(name="sparse_prefill_order_masks", known_block_size=[THREADS, 1, 1])
def _order_masks_kernel(
    SLOTS: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    COUNTERS: fx.Tensor,
    TILE_RANKS: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    CHUNKS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    GROUPS: fx.Int32,
    RANKS: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    UNION_STEP: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    SPLIT_PAIRS: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    BALANCE: fx.Constexpr[int],
    MEMBERS64: fx.Constexpr[bool],
    MAX_WIDTH: fx.Constexpr[int],
    MASK_PARTS: fx.Constexpr[int],
    ORDER_PRIORITY: fx.Constexpr[bool],
):
    # The order CTA's balance reuses its sort and route words; mask CTAs share the allocation.
    words = max(
        ORDER_WORDS + (SPLIT_WORDS if SPLIT > 1 else 0),
        BALANCE_WORDS,
        SORT_TOTALS + 5,
        MASK_WORDS,
    )
    scratch = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int32, words, 16])
        .peek()
        .view(fx.make_layout(words, 1))
    )
    program = fx.block_idx.x
    # The order CTA, then MASK_PARTS mask CTAs per tile, then the raw direct group-list CTAs
    # and one CTA ranking the direct tiles (RANKS).
    mask_ctas = TILES * MASK_PARTS
    if fx.const_expr(ORDER_PRIORITY):  # noqa: SIM102 - a compile-time branch
        if program == 0:
            prim.s_setprio(3)
    if (program == 0) | (program > mask_ctas):
        threshold = fx.Int32(INT_MAX)
        if fx.const_expr(ROUTE):
            threshold = _route(
                COUNTS,
                COSTS,
                RESIDUAL,
                scratch,
                TILES,
                TASK_SHARE,
                DIRECT_STEP,
                DIRECT_FIXED,
                UNION_STEP,
                ROUTE_MODEL,
                CORE,
            )
        if program == 0:
            # The variant leaves out the wider sorts its plan never runs: they set the
            # registers of every CTA of the launch (see _sort_width_bound).
            widths = tuple(w for w in SORT_WIDTHS if w <= MAX_WIDTH)
            for width in fx.range_constexpr(len(widths)):
                width = widths[width]
                lower = 0 if width == 1 else THREADS * width // 2
                # A plan of DESCENT_ROUNDS rounds has at most DESCENT_ROUNDS * BALANCE_GRID
                # entries: its variant sorts up to SORT_WIDTHS[-2] (_sort_width_bound) and its
                # chunk holds more than lower. Only those sorts carry _descend (in every sort
                # it took K3 from 63 to 70 VGPRs, 7 waves per SIMD instead of 8).
                descents = (
                    MAX_WIDTH < SORT_WIDTHS[-1]
                    and lower < DESCENT_ROUNDS * BALANCE_GRID
                )
                if (SIZE > lower) & (SIZE <= THREADS * width):
                    _order_chunks(
                        COUNTS,
                        COSTS,
                        RESIDUAL,
                        ACTIVE,
                        ORDER,
                        DIRECT_FLAG,
                        COUNTERS,
                        scratch,
                        threshold,
                        TASKS,
                        GRID,
                        SIZE,
                        CHUNKS,
                        HK,
                        SLICES,
                        width,
                        SPLIT,
                        SPLIT_PAIRS,
                        CORE,
                        BALANCE,
                        descents,
                    )
        elif program - 1 - mask_ctas < (GROUPS + THREADS - 1) // THREADS:
            _direct_groups(
                COUNTS,
                COSTS,
                QUERY_TILES,
                GROUP_META,
                GROUP_LIST,
                DIRECT_FLAG,
                scratch,
                program - 1 - mask_ctas,
                threshold,
                GROUPS,
            )
        else:
            _rank_direct_tiles(
                COUNTS,
                COSTS,
                TILE_RANKS,
                DIRECT_FLAG,
                scratch,
                threshold,
                TILES,
                TASKS // (HK * SLICES),
            )
    else:
        # MASK_PARTS mask CTAs per union tile, last tile first.
        item = program - 1
        _masks(
            COUNTS,
            COSTS,
            META,
            BLOCKS,
            MEMBERSHIP,
            MASKS,
            SLOTS,
            POOL_BLOCKS,
            TAILS,
            scratch,
            TILES - 1 - item // MASK_PARTS,
            item % MASK_PARTS,
            QB,
            CAP,
            SLOTTED,
            MEMBERS64,
            MASK_PARTS,
        )


@flyc.jit
def _order_masks_launch(
    SLOTS: fx.Tensor,
    COUNTS: fx.Tensor,
    COSTS: fx.Tensor,
    RESIDUAL: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERSHIP: fx.Tensor,
    META: fx.Tensor,
    MASKS: fx.Tensor,
    POOL_BLOCKS: fx.Tensor,
    TAILS: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    GROUP_META: fx.Tensor,
    GROUP_LIST: fx.Tensor,
    COUNTERS: fx.Tensor,
    TILE_RANKS: fx.Tensor,
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SIZE: fx.Int32,
    CHUNKS: fx.Int32,
    QB: fx.Int32,
    CAP: fx.Int32,
    TILES: fx.Int32,
    GROUPS: fx.Int32,
    RANKS: fx.Int32,
    TASK_SHARE: fx.Float32,
    DIRECT_STEP: fx.Float32,
    DIRECT_FIXED: fx.Float32,
    UNION_STEP: fx.Float32,
    HK: fx.Constexpr[int],
    SLICES: fx.Constexpr[int],
    ROUTE: fx.Constexpr[bool],
    ROUTE_MODEL: fx.Constexpr[tuple],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    SPLIT_PAIRS: fx.Constexpr[bool],
    CORE: fx.Constexpr[bool],
    BALANCE: fx.Constexpr[int],
    MEMBERS64: fx.Constexpr[bool],
    MAX_WIDTH: fx.Constexpr[int],
    MASK_PARTS: fx.Constexpr[int],
    ORDER_PRIORITY: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if TILES > 0:
        _order_masks_kernel(
            SLOTS,
            COUNTS,
            COSTS,
            RESIDUAL,
            ACTIVE,
            ORDER,
            DIRECT_FLAG,
            BLOCKS,
            MEMBERSHIP,
            META,
            MASKS,
            POOL_BLOCKS,
            TAILS,
            QUERY_TILES,
            GROUP_META,
            GROUP_LIST,
            COUNTERS,
            TILE_RANKS,
            TASKS,
            GRID,
            SIZE,
            CHUNKS,
            QB,
            CAP,
            TILES,
            GROUPS,
            RANKS,
            TASK_SHARE,
            DIRECT_STEP,
            DIRECT_FIXED,
            UNION_STEP,
            HK,
            SLICES,
            ROUTE,
            ROUTE_MODEL,
            SLOTTED,
            SPLIT,
            SPLIT_PAIRS,
            CORE,
            BALANCE,
            MEMBERS64,
            MAX_WIDTH,
            MASK_PARTS,
            ORDER_PRIORITY,
            value_attrs=prim.kernel_attrs(),
        ).launch(
            grid=(
                1 + TILES * MASK_PARTS + (GROUPS + THREADS - 1) // THREADS + RANKS,
                1,
                1,
            ),
            block=(THREADS, 1, 1),
            stream=stream,
        )


# K3b (CTAs of the FP8 pack launch, after K3): each row of an active core tile keeps only its
# residual blocks (complete blocks outside the tile's core, ascending) at the front of its
# block_indices row and their count in residual_rows; rows of other active tiles get count 0.
# One CTA per tile: a bitmap of the core over the tile's causal blocks in LDS (RESIDUAL_WORDS
# words cover the longest dense range), then a wave per row at a time, one bit test per block.
RESIDUAL_WORDS = -(-((LDS_BUDGET - _scan_bytes(True)) // 4) // 32)


@flyc.jit
def _residual_row(ROWS, RESIDUAL_ROWS, shared, words, row, visible, lane, topk):
    blocks = _resource(ROWS, topk * 4, fx.Int64(row) * topk)
    complete = _min(visible >> 2, fx.Int32(topk))
    ids = [_word(blocks, (k * 64 + lane) * 4) for k in range(topk // 64)]
    keep = []
    for k in fx.range_constexpr(topk // 64):
        inside = k * 64 + lane < complete
        word = _lds_load(shared + inside.select(_min(ids[k] >> 5, words - 1), 0) * 4)
        core = ((word >> (ids[k] & 31)) & 1) != 0
        keep.append(inside & ~core)
    base = fx.Int32(0)
    for k in fx.range_constexpr(topk // 64):
        mask = _ballot(keep[k])
        # Entries move only toward the front, after every lane loaded the row.
        _put(blocks, keep[k].select(base + _below(mask), fx.Int32(topk)), ids[k])
        base = base + _count(mask)
    if lane == 0:
        RESIDUAL_ROWS[row] = base


@flyc.jit
def residual_tile(
    ROWS,
    COUNTS,
    UNION,
    META,
    ACTIVE,
    RESIDUAL,
    RESIDUAL_ROWS,
    CAPACITY,
    MAX_BLOCKS,
    tile,
    shared,
    TOPK: fx.Constexpr[int],
):
    """K3b for one tile (a 256-thread CTA; shared: RESIDUAL_WORDS words of LDS, byte address)."""
    tid = fx.Int32(fx.thread_idx.x)
    wave, lane = _uniform(tid >> 6), tid & 63
    first_row, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    position = _uniform(META[tile * 5 + 4])
    active = _uniform(ACTIVE[tile]) != 0
    if active & (_uniform(RESIDUAL[tile]) > 0):
        count = _uniform(COUNTS[tile])
        union = _resource(UNION, CAPACITY * 4, fx.Int64(tile) * fx.Int64(CAPACITY))
        # Core blocks lie below the tile's causal end (K2's dense range).
        words = (_min(MAX_BLOCKS, (position + rows + 3) >> 2) + 31) >> 5
        for i in range(tid, words, fx.Int32(THREADS)):
            _lds_store(shared + fx.Int32(i) * 4, fx.Int32(0))
        _cta_sync()
        for i in range(tid, count, fx.Int32(THREADS * 4)):
            # Four core entries in flight per thread.
            entries = [fx.Int32(i) + j * THREADS for j in range(4)]
            loaded = [_word(union, _min(e, count - 1) * 4) for e in entries]
            for j in fx.range_constexpr(4):
                if entries[j] < count:
                    block = loaded[j]
                    _or(shared + (block >> 5) * 4, fx.Int32(1) << (block & 31), False)
        _cta_sync()
        for r in range(wave, rows, fx.Int32(THREADS // 64)):
            r = fx.Int32(r)
            _residual_row(
                ROWS,
                RESIDUAL_ROWS,
                shared,
                words,
                first_row + r,
                position + r + 1,
                lane,
                TOPK,
            )
    elif active:
        for r in range(tid, rows, fx.Int32(THREADS)):
            RESIDUAL_ROWS[first_row + fx.Int32(r)] = 0


def _cdiv(a, b):
    return -(-a // b)


def upload(array, device) -> torch.Tensor:
    # Pinned and asynchronous: building a new layout never drains the stream.
    host = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
    return host.pin_memory().to(device, non_blocking=True)


def scratch_specs(plan: SparsePlan) -> list:
    """(field, shape, dtype) of the per-call scratch; every launch rewrites what it reads."""
    tiles, capacity = plan.union_tiles, plan.block_capacity
    members = torch.int64 if plan.query_tile > 32 else torch.int32
    specs = [
        ("blocks", (tiles, capacity), torch.int32),
        ("membership", (tiles, capacity), members),
        ("score_masks", (tiles, capacity // 16, plan.query_tile, 4), torch.int16),
    ]
    if plan.slotted:
        specs += [
            ("pool_blocks", (tiles, capacity), torch.int32),
            ("tails", (tiles,), torch.int32),
        ]
    if plan.split > 1:
        specs += [
            ("partials", (SPLIT_SLOTS * SPLIT_PARTS, PARTIAL_FLOATS), torch.float32),
            ("split_counters", (SPLIT_SLOTS,), torch.int32),
        ]
    if plan.core_ratio > 0:
        specs += [
            ("residual_rows", (plan.rows,), torch.int32),
            ("lse", (plan.rows, plan.heads), torch.float32),
        ]
    return specs


def allocate_scratch(owner, specs, device) -> None:
    for name, shape, dtype in specs:
        setattr(owner, name, torch.empty(shape, dtype=dtype, device=device))


def wide_tile_rows(query_tile: int) -> int:
    """Fewest union rows that take the BQ21/BQ42 tile."""
    return math.ceil(query_tile * get_cu_num() * platform().wide_tiles_per_cu)


def _tile_rows(group, query_tile, rows, one_head):
    """Union tile rows for `rows` query rows; one_head: one KV head."""
    # H6/H3 tiles of 21/42 queries fill 126/128 MFMA rows (BQ42 with 64-bit membership), with or
    # without a prefix. Layouts too small to fill the persistent grid keep BQ16/BQ32.
    if group == 6 and one_head and query_tile >= 21 and rows >= wide_tile_rows(21):
        return 21
    if group == 3 and one_head and query_tile == 32 and rows >= wide_tile_rows(42):
        return 42
    return min(
        query_tile,
        128 // group if group == 12 else 1 << ((128 // group).bit_length() - 1),
    )


def _batch_tile_rows(group, query_tile, q_lens, one_head):
    """Union tile rows of a batch whose decisions span it: wide tiles by its total rows,
    while they take fewer tiles (short requests fill a narrow tile as well)."""
    narrow = _tile_rows(group, query_tile, 0, one_head)
    rows = _tile_rows(group, query_tile, int(q_lens.sum()), one_head)
    if rows > narrow and (-(-q_lens // rows)).sum() >= (-(-q_lens // narrow)).sum():
        return narrow
    return rows


def packed_block_starts(query_lens, prefix_lens):
    """Each request's first block in the FP8 pack's four-token blocks of every request with
    queries (a request's ceil(length / 4) blocks, in request order)."""
    q_lens = np.asarray(query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    return np.cumsum(blocks) - blocks


def fixed_selection(query_lens, prefix_lens, topk) -> bool:
    """Whether the indices layout fixes every selection: no row's causal window holds more
    than topk complete blocks, so each row selects all of them (each request spans at most
    4 * topk + 3 tokens)."""
    q_lens = np.asarray(query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(prefix_lens, dtype=np.int64)
    return bool(np.all(lengths[q_lens > 0] // 4 <= topk))


def allocate_plan(
    *,
    inputs,
    query_tile: int,
    scratch: bool = True,
    slotted: bool = False,
    min_request_rows: int | None = None,
    fp8=None,
) -> SparsePlan:
    """Union plan of a layout. By default its decisions span the batch: every request's tiles
    are union candidates, the batch's rows decide whether H6/H3 tiles take BQ21/BQ42 (one KV
    head, with or without prefixes), K3 routes over all tiles and short prefills split their
    long union tasks into key ranges (not with min_request_rows: a split depends on the whole
    batch's load). min_request_rows makes the decisions per request (batch-invariant):
    requests of at least that many rows take union tiles of their own rows (the kernels
    compile for the widest), the others direct-only tiles after them, and K3 does not route.

    fp8 ("tensor" / "token") plans for the FP8 kernels: their route calibration, BQ21/BQ42
    tiles for full prefills only and core/residual tiles (H12/H6 without a split, nor a fixed
    selection).
    """
    group = inputs.q.shape[1] // inputs.k.shape[1]
    if query_tile < 1 or query_tile > 32:
        raise ValueError("Require 1<=BQ<=32")
    config = platform(fp8)
    num_cus = get_cu_num()
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    prefix = np.asarray(inputs.prefix_lens, dtype=np.int64)
    lengths = q_lens + prefix
    # FP8 plans keep BQ21/BQ42 tiles to full prefills (a single one by default): the FP8
    # kernels' route terms and errors were measured with those tiles only.
    if min_request_rows is None:
        if fp8 is None:
            query_tile = _batch_tile_rows(group, query_tile, q_lens, hk == 1)
        else:
            full = hk == 1 and len(q_lens) == 1 and prefix[0] == 0
            query_tile = _tile_rows(group, query_tile, int(q_lens[0]), full)
        union = np.ones(len(q_lens), dtype=bool)
        rows = np.full(len(q_lens), query_tile)
        max_blocks = _cdiv(inputs.max_seqlen_k, 4)
    else:
        union = q_lens >= min_request_rows
        rows = np.array(
            [
                _tile_rows(
                    group, query_tile, int(n), hk == 1 and (fp8 is None or p == 0)
                )
                for n, p in zip(q_lens, prefix)
            ],
            dtype=np.int64,
        )
        query_tile = int(rows[union].max())
        rows[~union] = query_tile
        max_blocks = _cdiv(int(lengths[union].max()), 4)
    # Equal tiles per request, larger first: a tiny last tile would route its few rows direct, after union.
    # Union candidate requests' tiles come first.
    order = np.concatenate((np.flatnonzero(union), np.flatnonzero(~union)))
    counts = -(-q_lens // rows)
    request = np.repeat(order, counts[order])
    index = np.arange(len(request)) - np.repeat(
        np.cumsum(counts[order]) - counts[order], counts[order]
    )
    base = (q_lens // np.maximum(counts, 1))[request]
    extra = (q_lens % np.maximum(counts, 1))[request]
    local = index * base + np.minimum(index, extra)
    sizes = base + (index < extra)
    metadata = np.stack(
        (
            (np.cumsum(q_lens) - q_lens)[request] + local,
            sizes,
            (np.cumsum(lengths) - lengths)[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    tiles = len(request)
    union_tiles = int(counts[union].sum())
    query_tiles = np.empty(int(sizes.sum()), dtype=np.int64)
    within = np.arange(len(query_tiles)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    query_tiles[np.repeat(metadata[:, 0], sizes) + within] = np.repeat(
        np.arange(tiles), sizes
    )
    # A query adds at most topk blocks and its tail block.
    capacity = _cdiv(min(max_blocks, query_tile * (inputs.topk + 1)), 16) * 16
    slices = _cdiv(query_tile * group, 128)
    tasks = tiles * hk * slices
    # A union CTA (a wave per SIMD, 64 KiB LDS) fills a CU: a larger grid only adds dispatch
    # rounds.
    grid = min(union_tiles * hk * slices, num_cus)
    # BF16 between the CUs and twice the CUs: K3 splits a task of more steps than the even
    # load, and causal tasks run about half the longest (capacity // 16 steps) on average;
    # where none can exceed that load a split plan only adds its fixed costs (K3's split
    # pass, twice the union slots: a 1024-token prefill at H12, +2%).
    bf16_split = (
        tasks < num_cus
        or (capacity // 16) * (num_cus - tasks / 2) > SPLIT_OVERHEAD * tasks
    )
    split = (
        SPLIT_PARTS
        if min_request_rows is None
        and 0 < tasks < min(num_cus if fp8 is not None else 2 * num_cus, SPLIT_TASKS)
        and (fp8 is not None or bf16_split)
        and _cdiv(tasks * SPLIT_PARTS, num_cus) * num_cus
        <= THREADS * _split_sort_width(num_cus)
        else 1
    )
    # Split plans run every CU (parts of a task are separate slots).
    launch_grid = num_cus if split > 1 else grid
    packed = int((-(-lengths // 4))[q_lens > 0].sum())
    # BF16 plans: K2 weighs direct steps past the knee where they read at least a third of a
    # K/V byte per query head (DIRECT_SPAN_GROUPS), K3 the union step in the other plans, at
    # the plan's longest context.
    weighted = fp8 is None
    span_slope = (
        DIRECT_SPAN_SLOPE if weighted and group <= DIRECT_SPAN_GROUPS * 2 else 0.0
    )
    doublings = math.log2(max(int(lengths.max()), 1)) - DIRECT_SPAN_KNEE
    union_weight = (
        1.0 + DIRECT_SPAN_SLOPE * max(doublings, 0.0)
        if weighted and not span_slope
        else 1.0
    )
    route_costs = (
        hk * slices / grid if grid else 0.0,
        config.direct_cu_step * hk / num_cus,
        config.direct_fixed + config.pack_block * packed * hk,
        config.union_step * union_weight,
    )
    # Core/residual tiles need the FP8 direct merge; KV-split (short) plans keep whole unions,
    # and so do fixed selections (fixed_selection): their rows select every block below a
    # tile's diagonal, so a core is the whole union, and their plans run again without K1,
    # whose block_indices the residual K3b rewrites in place (the entry's plan reuse).
    fixed = fixed_selection(inputs.query_lens, inputs.prefix_lens, inputs.topk)
    core_ratio = (
        CORE_RATIOS.get(group, 0.0)
        if fp8 is not None and split == 1 and not fixed
        else 0.0
    )
    kwargs = {"dtype": torch.int32, "device": inputs.q.device}
    plan = SparsePlan(
        metadata=upload(metadata, inputs.q.device),
        blocks=None,
        membership=None,
        score_masks=None,
        counts=torch.zeros(tiles, **kwargs),
        costs=torch.zeros((tiles, 3), **kwargs),
        active=torch.zeros(tiles, **kwargs),
        task_order=torch.empty(
            _cdiv(tasks * split, launch_grid) * launch_grid if grid else 0, **kwargs
        ),
        direct_flag=torch.empty(3, **kwargs),
        query_tiles=upload(query_tiles, inputs.q.device),
        query_tile=query_tile,
        group_padded=group,
        block_capacity=capacity,
        max_blocks=max_blocks,
        num_tiles=tiles,
        union_tiles=union_tiles,
        grid=launch_grid,
        route_costs=route_costs,
        route_model=(
            config.union_fixed,
            config.union_step,
            config.row_step,
            config.route_margin,
        ),
        union_ratio=config.union_ratio,
        span_slope=span_slope,
        task_overhead=TASK_OVERHEADS["bf16" if fp8 is None else "fp8"],
        routing=min_request_rows is None,
        slotted=slotted,
        split=split,
        split_pairs=split > 1 and fp8 is None,
        core_ratio=core_ratio,
        residual=torch.zeros(tiles, **kwargs),
        rows=inputs.q.shape[0],
        heads=inputs.q.shape[1],
    )
    if scratch:
        allocate_scratch(plan, scratch_specs(plan), inputs.q.device)
    return plan


_COMPILED = {}


def _split_sort_width(num_cus):
    """The widest K3 sort (entries per thread) a split plan on num_cus CUs runs: its task
    order holds fewer than SPLIT_PARTS * 2 * num_cus entries (plans split below twice the
    CUs' tasks at most); 2 on 80 CUs."""
    entries = SPLIT_PARTS * 2 * num_cus
    return next((w for w in SORT_WIDTHS if THREADS * w >= entries), SORT_WIDTHS[-1])


def _sort_width_bound(plan, size):
    """The widest K3 sort a plan's variant compiles: a split plan's (a constant per device);
    else the widest one only for a chunk of more than THREADS * 8 entries. The widest sort
    takes K3 from 64 to 83 VGPRs, and the launch's mask CTAs share them: 5 per CU instead of
    8 (two variants, by the task order's size class)."""
    if plan.split > 1:
        return _split_sort_width(plan.grid)
    return SORT_WIDTHS[-1] if size > THREADS * SORT_WIDTHS[-2] else SORT_WIDTHS[-2]


def _compiled(launch, args, grid, constants):
    """Compile once per device and constexpr values. Compiling runs the launcher, so it gets
    an empty grid (args[grid] = 0) and launches nothing."""
    device = args[0].device
    key = (launch, device, *constants)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        empty = list(args)
        empty[grid] = 0
        with torch.cuda.device(device):
            compiled = _COMPILED[key] = flyc.compile(launch, *empty)
    return compiled


def _launches(inputs, plan, stream):
    """(compiled, args) of K1-K3 in launch order; K1's first argument is the indices, K3's
    the slot table (a stand-in without one)."""
    args = (
        inputs.indices,
        inputs.query_positions,
        inputs.block_indices.view(-1),
        inputs.q.shape[0],
        inputs.topk,
        stream,
    )
    yield _compiled(_recover_launch, args, 3, (inputs.topk,)), args
    if plan is None or plan.num_tiles == 0:
        return
    members64 = plan.query_tile > 32
    core = plan.core_ratio > 0
    lds_bytes, dense_limit = lds_layout(plan.max_blocks, members64, core)
    # BQ > 21 tiles take the larger compact CTA, and so do all tiles of a plan whose longest
    # range passes the dense limit: its CTAs fill LDS_BUDGET, two per CU, and twice the waves
    # keep twice the block id loads and window passes in flight (a 16k-token chunk after a
    # 112k-token prefix, H6: 131 -> 101 us; ranges that fit stay faster with the smaller CTA).
    large = plan.query_tile > 21 or plan.max_blocks > dense_limit
    waves = COMPACT_THREADS[large] // 64
    assert -(-plan.query_tile // waves) <= REGISTER_ROWS
    # A full tile's ids per thread, in the fewest passes of at most UNROLL_MAX, evened out.
    per_thread = _cdiv(plan.query_tile * inputs.topk, COMPACT_THREADS[large])
    unroll = _cdiv(per_thread, _cdiv(per_thread, UNROLL_MAX))
    args = (
        inputs.block_indices.view(-1),
        plan.metadata.view(-1),
        plan.blocks.view(-1),
        plan.membership.view(torch.int32).view(-1),
        plan.counts,
        plan.costs.view(-1),
        plan.residual,
        plan.direct_flag,
        plan.max_blocks,
        plan.block_capacity,
        plan.union_ratio,
        plan.core_ratio,
        RESIDUAL_ROW.get(plan.group_padded, 0.0),
        CORE_MARGINS.get(plan.group_padded, 1.0),
        plan.span_slope / 256,
        dense_limit,
        lds_bytes,
        plan.union_tiles,
        members64,
        large,
        core,
        inputs.topk,
        unroll,
        stream,
    )
    constants = (members64, large, core, inputs.topk, unroll)
    yield _compiled(_compact_launch, args, 17, constants), args
    hk = inputs.k.shape[1]
    slices = _cdiv(plan.query_tile * plan.group_padded, 128)
    # Order CTAs cover every tile (a direct-only tile publishes 0 and raises the direct flag);
    # mask CTAs only the union tiles.
    tasks = plan.num_tiles * hk * slices
    numel = plan.task_order.numel()
    # The order CTA sorts the task order a chunk of the widest variant at a time (a split plan
    # always fits one).
    size = min(THREADS * SORT_WIDTHS[-1], numel)
    # One chunk's rounds are balanced (a split plan's two are dealt in a snake either way);
    # balance: the task overhead of the plan's union kernel, 0 when not balanced. A
    # batch-invariant plan (no routing) keeps the snake: the balance deals by the call's loads.
    balance = (
        plan.task_overhead
        if plan.routing
        and plan.split == 1
        and numel == size
        and plan.grid <= BALANCE_GRID
        else 0
    )
    standin = plan.counts  # slot-table tensors K3 does not read without a slot table
    max_width = _sort_width_bound(plan, size)
    mask_parts = (
        MASK_PARTS
        if plan.query_tile >= 21 and MASK_PARTS * plan.union_tiles <= 8 * plan.grid
        else 1
    )
    # Narrow tiles' plans end K3 with the order CTA (its serial phases): its waves then win
    # the SIMDs it shares with mask CTAs; wide tiles' end with mask CTAs.
    order_priority = plan.query_tile < 21
    args = (
        inputs.kv_slots if plan.slotted else standin,
        plan.counts,
        plan.costs.view(-1),
        plan.residual,
        plan.active,
        plan.task_order,
        plan.direct_flag,
        plan.blocks.view(-1),
        plan.membership.view(-1),
        plan.metadata.view(-1),
        plan.score_masks.view(-1),
        plan.pool_blocks.view(-1) if plan.slotted else standin,
        plan.tails if plan.slotted else standin,
        plan.query_tiles,
        plan.direct_meta.view(-1) if plan.direct_groups else standin,
        plan.direct_list if plan.direct_groups else standin,
        plan.split_counters if plan.split > 1 else standin,
        plan.tile_ranks if plan.direct_groups else standin,
        tasks,
        plan.grid,
        size,
        _cdiv(numel, size),
        plan.query_tile,
        plan.block_capacity,
        plan.union_tiles,
        plan.direct_groups,
        1 if plan.direct_groups else 0,
        *plan.route_costs,
        hk,
        slices,
        plan.routing,
        plan.route_model,
        plan.slotted,
        plan.split,
        plan.split_pairs,
        core,
        balance,
        members64,
        max_width,
        mask_parts,
        order_priority,
        stream,
    )
    constants = (
        hk,
        slices,
        plan.routing,
        plan.route_model,
        members64,
        plan.slotted,
        plan.split,
        plan.split_pairs,
        core,
        balance,
        max_width,
        mask_parts,
        order_priority,
    )
    yield _compiled(_order_masks_launch, args, 24, constants), args


def run(*, inputs, plan):
    stream = torch.cuda.current_stream(inputs.q.device)
    for compiled, args in _launches(inputs, plan, stream):
        compiled(*args)


def launchers(*, inputs, plan):
    """K1-K3 launches with this binding's other arguments prebuilt. K1 takes the indices,
    K2 and K3 the slot table (read by K3 of a slot-table plan, ignored otherwise).

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    stream = torch.cuda.current_stream(inputs.q.device)
    (recover, args), *rest = _launches(inputs, plan, stream)
    tail = args[1:]
    result = [lambda indices: recover(indices, *tail)]
    if rest:
        (compact, compact_args), (masks, mask_args) = rest
        mask_tail = mask_args[1:]
        result.append(lambda slots: compact(*compact_args))
        if plan.slotted:
            result.append(lambda slots: masks(slots, *mask_tail))
        else:
            result.append(lambda slots: masks(*mask_args))
    return result
