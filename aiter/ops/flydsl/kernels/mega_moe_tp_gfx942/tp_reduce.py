# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-rank reduction fused into the last kernel of ``flydsl_moe_a8w4_ptpc`` (``MegaMoeTPGfx942``, M > 32), gfx942.

``compile_tp_reduce`` replaces the route sum of one rank's A8W4 MoE (``a8w4_route_sum`` / ``sorted_sum``, or
nothing after the atomic streaming Down) **and** the all-reduce after it: every rank enters with the partial
output of its slice of the intermediate dimension and leaves with the sum over the ranks, bitwise equal on every
rank. Ported from the mega-tp-a8w4 S2 prototype (``a8w4_tp_comm.py``, proto-b-overlap) with: any M (a masked
last tile), an inbox of ``WORLD - 1`` source regions, a landing buffer per rank (the output is not symmetric
memory: the owner writes peers' rows there and every rank copies them into its output), 64-bit route row bases
(a prefill's routes can exceed 2 GiB) and waits bounded in time rather than in polls.

Tile: ``TT`` tokens; tile ``i`` covers tokens ``i * TT ..`` (the last one masked at M) and is owned by rank
``i % WORLD`` (tiles in flight together use every link). A workgroup of 4 waves takes the tiles ``w, w + nwg,
..``; a wave takes the (token, 512-column chunk) items ``wave, wave + 4, ..`` of a tile, 8 columns per lane.

Owners (``OWNER``): ``OWN_TILE`` (the all-reduce layer) as above; ``OWN_HOME`` (data parallelism): the batch is
``M = WORLD * M_r`` tokens, rank r's being ``[r * M_r, (r + 1) * M_r)``, tiled per rank (``tpr = ceil(M_r / TT)``
tiles each: tile ``r + WORLD * i`` holds the tokens ``r * M_r + i * TT ..`` masked at ``(r + 1) * M_r``, inbox row
``i * TT + t``, ``rps = tpr * TT``) and owned by its tokens' rank, the owners interleaved as above. ``AG`` = 0
(``OWN_HOME`` only): the owner stores the sums into its own ``out`` only (BF16 [M_r, H], its tokens), nothing into
``landing``; it still publishes the AG flags (they tell the peers that it has read their partials, rule 3), which
the second loop waits for, copying nothing. As in ``down_tp`` (its Owners).

* partial: SUM mode (``TOPK > 0``): ``sum_k routes[loc[t, k]]`` in f32 in slot order (missing routes add +0),
  rounded to BF16 -- the arithmetic of ``a8w4_route_sum`` / ``sorted_sum``; COPY mode (``TOPK == 0``): the BF16
  row of ``src`` (the atomic Down's output, possibly ``out`` itself).
* reduce-scatter (non-owner): the partial goes to the owner's inbox, region ``(rank - owner - 1) % WORLD``, row
  ``(i // WORLD) * TT + t - i * TT``; then ``RS[i][rank] = epoch`` on the owner.
* sum + all-gather (owner): waits for the ``WORLD - 1`` RS flags, adds the BF16 partials of ranks 0 .. WORLD-1 in
  f32 in that order (its own one at its rank's place), rounds to BF16 once, stores the row into its ``out`` and
  every peer's ``landing``, then ``AG[i] = epoch`` on every peer (``POISON`` set if the RS wait gave up, rule 5).
* copy (second loop, the tiles this rank does not own): waits for ``AG[i]``, copies ``landing`` rows into ``out``.

Schedule (one workgroup, per tile of the first loop): items' partials (all ``TOPK`` row loads of an item in flight
together, ``sched_barrier`` between items) -> owner: wait RS, ``WORLD - 1`` inbox loads per item, sum, stores ->
``s_waitcnt vmcnt(0)``, barrier, flags; non-owner: inbox stores -> ``s_waitcnt vmcnt(0)``, barrier, RS flag. Then
the second loop over the same tiles: wait AG, load all items, store all items.

Naming (int32 index unless noted): ``tab`` int64 [3, WORLD]: per rank the address of ``landing`` (BF16 [M_max,
H]), ``inbox`` (BF16 [(WORLD - 1) * rps, H], ``rps = ceil(tiles / WORLD) * TT``) and ``flags`` (int32
``[slots * (WORLD + 1)]``: ``RS[i][s]`` at ``i * WORLD + s``, ``AG[i]`` at ``slots * WORLD + i``); ``epoch``
int32 [slots] (this rank's, plain device memory); ``err`` int32 [1].

Rules:

1. Epochs live on the device: a tile's epoch is ``epoch[i] + 1``, stored back by the workgroup that took the tile
   once it is done with it. Every rank makes the same calls, so the epochs agree without a host counter (CUDA
   graph replays advance them). Flags hold the last epoch, so they are never cleared between calls.
2. Release: payload stores (system scope, written through), ``s_waitcnt vmcnt(0)``, workgroup barrier, then the
   flag store (system scope). Acquire: wave 0 polls the flag with system-scope loads, workgroup barrier, then
   the payload loads (system scope: past the caches).
3. Reuse across calls: a non-owner overwrites the owner's inbox tile only in a later call, whose RS needs the
   AG of this call, i.e. the owner's inbox loads of this call. The owner overwrites a peer's landing tile only
   after that peer's RS of the later call, i.e. after the peer's kernel of this call (copies included) ended.
4. No deadlock: workgroup ``w`` only waits for workgroup ``w`` of the peers, which takes the same tiles in the same
   order; workgroups are dispatched in order on every XCD, so the lowest unfinished one is always resident.
5. Bounded waits: a wait gives up after ``limit * 1024`` ticks of the 100 MHz timer (``limit`` is a launch
   argument), ORs ``ERR_TIMEOUT`` into ``err`` and goes on: wrong output, never a hang. A wait that finds ``err``
   already set gives up after its first poll (one dead peer costs one timeout, not one per tile). An owner whose
   RS wait gave up still publishes the tile's AG flags, with ``POISON`` set; a copy that finds it ORs
   ``ERR_TIMEOUT`` too, so a rank that comes late (after the others gave up) does not take an incomplete sum
   silently: every rank with a wrong output has the bit. AG flags compare the epoch's low 30 bits.
6. Bitwise equal outputs: only the owner sums a row; every rank stores the owner's BF16 bits.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..communication_ops_utils import atomic_or_global_at, memrealtime
from ..moe_gemm_2stage.gfx9_fp8.stream_a8w4 import (
    _DROP,
    _address,
    _bf16_pair,
    _load,
    _pack64,
    _rsrc,
    _uniform,
)

# Cache policy sc0 | sc1 (system scope): stores are written through to the target's memory, loads go past the
# caches. The typed BufferCopy atom documents only 0 (cached) and 2 (nt) on CDNA3 (flydsl/expr/rocdl/cdna3.py),
# so the cross-rank accesses below use the raw buffer ops with this aux field, as the prototype did (verified on
# MI308X: S2-overlap report).
SYS = 17
# Bit of ``err`` set by a wait of this kernel that gave up, or by a copy of a poisoned tile.
ERR_TIMEOUT = 1
# AG flag bit of a tile whose owner gave up waiting for a partial (rule 5); the other bits hold the epoch.
POISON = 1 << 30
_EPOCH_BITS = POISON - 1
# Timer ticks (100 MHz) per unit of the ``limit`` launch argument.
_LIMIT_SHIFT = 10
LIMIT_TICKS = 1 << _LIMIT_SHIFT
# Owner maps (module docstring, Owners), down_tp's too.
OWN_TILE, OWN_HOME = 0, 1


def _store(rsrc, words, voffset, aux=0):
    words = list(words)
    value = (
        fx.Int32(words[0])
        if len(words) == 1
        else Vec.from_elements(words, fx.Uint32).bitcast(fx.Int32)
    )
    # Raw buffer store: the typed BufferCopy atom has no system-scope policy on CDNA3 (see SYS).
    rocdl.raw_ptr_buffer_store(
        value.ir_value(),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, aux),
    )


def _when(cond, fn):
    """``fn()`` under one scf.if on ``cond``."""
    # Raw scf.if: the body is emitted by a Python closure outside the kernel function, which FlyDSL's AST
    # rewriter does not convert (same as stream._atomic_add_bf16x2_if).
    loc = ir.Location.unknown()
    if_op = scf.IfOp(_raw(cond), [], has_else=False, loc=loc)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        fn()
        scf.YieldOp([], loc=loc)


def _when_int(cond, fn):
    """``fn()`` (an Int32) under one scf.if on ``cond``; 0 where ``cond`` is false."""
    # Raw scf.if with a result: see _when.
    loc = ir.Location.unknown()
    if_op = scf.IfOp(_raw(cond), [fx.Int32.ir_type], has_else=True, loc=loc)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        scf.YieldOp([fx.Int32(fn()).ir_value()], loc=loc)
    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
        scf.YieldOp([fx.Int32(0).ir_value()], loc=loc)
    return fx.Int32(if_op.results[0])


def _loop(lb, ub, step, fn):
    """``fn(i)`` for ``i`` in ``range(lb, ub, step)`` as one scf.for (run-time i32 bounds)."""
    # Raw scf.for: run-time bounds in a closure outside the kernel function (see _when).
    loc = ir.Location.unknown()
    for_op = scf.ForOp(
        fx.Int32(lb).ir_value(),
        fx.Int32(ub).ir_value(),
        fx.Int32(step).ir_value(),
        loc=loc,
    )
    with ir.InsertionPoint(for_op.body):
        fn(fx.Int32(for_op.induction_variable))
        scf.YieldOp([], loc=loc)


def _spin(poll, limit):
    """Poll until ``poll()`` (wave-uniform i1: still pending) is false or ``limit * LIMIT_TICKS`` timer ticks
    have passed; returns whether it gave up (wave-uniform i1)."""
    # Raw scf.while: a loop-carried result in a closure outside the kernel function (see _when).
    loc = ir.Location.unknown()
    i32 = fx.Int32.ir_type
    start = memrealtime()
    loop = scf.WhileOp([i32], [fx.Int32(0).ir_value()], loc=loc)
    loop.regions[0].blocks.append(i32, arg_locs=[loc])
    loop.regions[1].blocks.append(i32, arg_locs=[loc])
    with ir.InsertionPoint(loop.regions[0].blocks[0]):
        pending = poll()
        elapsed = (memrealtime() - start) >> _LIMIT_SHIFT
        go = pending & (elapsed < fx.Int64(limit))
        scf.ConditionOp(
            _raw(go), [pending.select(fx.Int32(1), fx.Int32(0)).ir_value()], loc=loc
        )
    with ir.InsertionPoint(loop.regions[1].blocks[0]):
        rocdl.s_sleep(2)
        scf.YieldOp([loop.regions[1].blocks[0].arguments[0]], loc=loc)
    return fx.Int32(loop.results[0]) != 0


def _ballot_any(cond):
    return fx.Int64(rocdl.ballot(T.i64, _raw(cond))) != fx.Int64(0)


def _f32_pairs(words):
    values = []
    for word in words:
        values.append((word << 16).bitcast(fx.Float32))
        values.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
    return values


def _sum_bf16(rows):
    """f32 sum of BF16 rows (lists of 4 words: 8 columns) in list order, rounded once -> 4 words."""
    total = None
    for words in rows:
        values = _f32_pairs(words)
        total = values if total is None else [a + b for a, b in zip(total, values)]
    return [_bf16_pair(total[2 * i], total[2 * i + 1]) for i in range(4)]


@functools.cache
def compile_tp_reduce(*, WORLD, H, TOPK, TT, OWNER=OWN_TILE, AG=1):
    """Launcher ``(loc, src, out, tab, epoch, err, M, rank, slots, limit, nwg, stream)`` (module docstring).

    ``WORLD`` ranks (2, 4 or 8), hidden size ``H`` (``% 512 == 0``), ``TT`` tokens per tile (1, 2 or 4).
    ``TOPK > 0``: ``src`` = routes BF16 by sorted row, ``loc`` int32 [M, TOPK] (negative: no route);
    ``TOPK == 0``: ``src`` = the partial BF16 [M, H] (may be ``out``), ``loc`` unused (any int32 tensor).
    ``tab``: int64 [3, WORLD] as int32; ``epoch`` int32 [slots] with ``tiles <= slots`` (``ceil(M / TT)``;
    ``OWN_HOME``: ``WORLD * ceil(M / WORLD / TT)``); ``err`` int32 [1]. ``OWNER`` / ``AG``: the owner map and
    whether the owners send the sums to every rank (module docstring, Owners; ``AG`` = 0: ``out`` BF16 [M / WORLD,
    H], not ``src``). ``limit``: wait bound in units of ``LIMIT_TICKS`` timer ticks. Grid: ``nwg`` workgroups of
    256 threads.
    """
    assert WORLD in (2, 4, 8), WORLD
    assert H % 512 == 0, H
    assert TT in (1, 2, 4), TT
    assert 0 <= TOPK <= 64, TOPK
    assert OWNER in (OWN_TILE, OWN_HOME), OWNER
    assert AG in (0, 1) and (AG or OWNER == OWN_HOME), (AG, OWNER)
    ROW = 2 * H  # bytes of a BF16 row
    CH = H // 512  # 512-column chunks of a row
    ITEMS = TT * CH  # (token, chunk) items of a tile
    IPW = -(-ITEMS // 4)  # items per wave

    def body(p_loc, p_src, p_out, p_tab, p_epoch, p_err, M, rank, slots, limit, nwg):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        first = fx.Int32(gpu.block_id("x"))
        if OWNER == OWN_TILE:
            tiles = (M + (TT - 1)) // TT
            rps = (tiles + (WORLD - 1)) // WORLD * TT
        else:
            home = M // WORLD
            tpr = (home + (TT - 1)) // TT
            tiles = tpr * WORLD
            rps = tpr * TT
        out_bytes = M * ROW
        inbox_bytes = rps * ((WORLD - 1) * ROW)
        flag_bytes = slots * ((WORLD + 1) * 4)
        ag0 = slots * WORLD
        tab = _rsrc(_address(p_tab), 3 * WORLD * 8)

        def entry(row, p):
            words = _load(tab, (row * WORLD + p) * 8, 0, 2)
            return _pack64(fx.Uint32(_uniform(words[0])), fx.Uint32(_uniform(words[1])))

        landing = [_rsrc(entry(0, p), out_bytes) for p in range(WORLD)]
        inbox_addr = [entry(1, p) for p in range(WORLD)]
        flags = [_rsrc(entry(2, p), flag_bytes) for p in range(WORLD)]
        own_landing = _rsrc(entry(0, rank), out_bytes)
        own_inbox = _rsrc(entry(1, rank), inbox_bytes)
        own_flags = _rsrc(entry(2, rank), flag_bytes)
        out_rsrc = _rsrc(_address(p_out), out_bytes if AG else home * ROW)
        ep_rsrc = _rsrc(_address(p_epoch), slots * 4)
        err_rsrc = _rsrc(_address(p_err), 4)
        src_base = _address(p_src)
        if TOPK > 0:
            loc_rsrc = _rsrc(_address(p_loc), M * (TOPK * 4))
        else:
            src_rsrc = _rsrc(src_base, out_bytes)

        def place(tile):
            """``OWN_HOME``: owner, first token and end of the tokens (exclusive) of ``tile`` (module docstring,
            Owners)."""
            owner = _uniform(tile % WORLD)
            t0 = owner * home + (tile // WORLD) * TT
            return owner, t0, owner * home + home

        def item(t0, j, end):
            """Token, byte offset in a row-major [M, H] BF16 buffer and liveness of this wave's item ``j`` (tokens
            before ``end``)."""
            idx = wave + 4 * j
            token = t0 + idx // CH
            col = (idx % CH) * 512 + lane * 8
            ok = token < end
            if 4 * j + 3 >= ITEMS:
                ok = ok & (idx < ITEMS)
            return token, col * 2, ok

        def wait(poll):
            """Wave 0 polls (bounded, rule 5), then a workgroup barrier; nonzero in wave 0 if it gave up."""

            def wave0():
                failed = fx.Int32(_load(err_rsrc, 0, 0, 1, aux=SYS)[0]) != 0
                late = _spin(poll, failed.select(fx.Int32(0), limit))
                _when(
                    late & (lane == 0),
                    lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
                )
                return late.select(fx.Int32(1), fx.Int32(0))

            gave_up = _when_int(wave == 0, wave0)
            gpu.barrier()
            return gave_up

        def epoch_of(tile):
            return fx.Int32(_uniform(_load(ep_rsrc, tile * 4, 0, 1)[0])) + 1

        def tile_body(tile):
            if OWNER == OWN_TILE:
                t0 = tile * TT
                owner = _uniform(tile % WORLD)
                end = M
                # Inbox row of token t of this tile within a source region.
                base = (tile // WORLD) * TT - t0
            else:
                owner, t0, end = place(tile)
                base = (tile // WORLD) * TT - t0
            epoch = epoch_of(tile)

            items = []
            for j in range(IPW):
                token, col2, ok = item(t0, j, end)
                if TOPK > 0:
                    mine = fx.Int32(
                        _load(
                            loc_rsrc,
                            (ok & (lane < TOPK)).select(
                                token * (TOPK * 4) + lane * 4, fx.Int32(_DROP)
                            ),
                            0,
                            1,
                        )[0]
                    )
                    rows = []
                    for k in range(TOPK):
                        loc = fx.Int32(
                            rocdl.readlane(
                                fx.Int32.ir_type,
                                mine.ir_value(),
                                fx.Int32(k).ir_value(),
                            )
                        )
                        live = ok & (loc >= 0)
                        # One row per descriptor from a 64-bit base: routes may exceed the 2 GiB voffset range.
                        row = _rsrc(src_base + fx.Int64(live.select(loc, 0)) * ROW, ROW)
                        rows.append(
                            _load(row, live.select(col2, fx.Int32(_DROP)), 0, 4)
                        )
                    partial = _sum_bf16(rows)
                else:
                    partial = _load(
                        src_rsrc, ok.select(token * ROW + col2, fx.Int32(_DROP)), 0, 4
                    )
                items.append((partial, token, col2, ok))
                rocdl.sched_barrier(0)

            def owner_part():
                peer = (lane < WORLD) & (lane != rank)

                def pending():
                    seen = fx.Int32(
                        _load(
                            own_flags,
                            peer.select((tile * WORLD + lane) * 4, fx.Int32(_DROP)),
                            0,
                            1,
                            aux=SYS,
                        )[0]
                    )
                    return _ballot_any(peer & (seen != epoch))

                # Thread 0 (wave 0) poisons the AG flags if a peer's partial is missing (rule 5).
                poison = (wait(pending) != 0).select(fx.Int32(POISON), fx.Int32(0))
                for partial, token, col2, ok in items:
                    parts = []
                    for s in range(WORLD):
                        region = (s + (WORLD - 1) - rank) & (WORLD - 1)
                        row = region * rps + base + token
                        got = _load(
                            own_inbox,
                            (ok & (rank != s)).select(
                                row * ROW + col2, fx.Int32(_DROP)
                            ),
                            0,
                            4,
                            aux=SYS,
                        )
                        parts.append(
                            [(rank == s).select(w, g) for w, g in zip(partial, got)]
                        )
                    total = _sum_bf16(parts)
                    # AG = 0: out holds this rank's tokens only (from t0 - its first tile's offset).
                    row_out = token if AG else token - owner * home
                    _store(
                        out_rsrc,
                        total,
                        ok.select(row_out * ROW + col2, fx.Int32(_DROP)),
                    )
                    if AG:
                        for p in range(WORLD):
                            _store(
                                landing[p],
                                total,
                                (ok & (rank != p)).select(
                                    token * ROW + col2, fx.Int32(_DROP)
                                ),
                                SYS,
                            )
                    rocdl.sched_barrier(0)
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                ag = (epoch & _EPOCH_BITS) | poison
                for p in range(WORLD):
                    _store(
                        flags[p],
                        [ag],
                        ((tid == 0) & (rank != p)).select(
                            (ag0 + tile) * 4, fx.Int32(_DROP)
                        ),
                        SYS,
                    )
                _store(ep_rsrc, [epoch], (tid == 0).select(tile * 4, fx.Int32(_DROP)))

            def peer_part():
                address = inbox_addr[0]
                for p in range(1, WORLD):
                    address = (owner == p).select(inbox_addr[p], address)
                owner_inbox = _rsrc(address, inbox_bytes)
                region = (rank + (WORLD - 1) - owner) & (WORLD - 1)
                for partial, token, col2, ok in items:
                    row = region * rps + base + token
                    _store(
                        owner_inbox,
                        partial,
                        ok.select(row * ROW + col2, fx.Int32(_DROP)),
                        SYS,
                    )
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                for p in range(WORLD):
                    _store(
                        flags[p],
                        [epoch],
                        ((tid == 0) & (owner == p)).select(
                            (tile * WORLD + rank) * 4, fx.Int32(_DROP)
                        ),
                        SYS,
                    )

            _when(owner == rank, owner_part)
            _when(owner != rank, peer_part)

        def copy_body(tile):
            if OWNER == OWN_HOME:
                owner, home_t0, home_end = place(tile)

            def copy():
                if OWNER == OWN_TILE:
                    t0, end = tile * TT, M
                else:
                    t0, end = home_t0, home_end
                epoch = epoch_of(tile)
                want = epoch & _EPOCH_BITS

                def pending():
                    seen = fx.Int32(
                        _load(own_flags, (ag0 + tile) * 4, 0, 1, aux=SYS)[0]
                    )
                    _when(
                        (lane == 0) & (seen == (want | POISON)),
                        lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
                    )
                    return _ballot_any((seen & _EPOCH_BITS) != want)

                wait(pending)
                rows = []
                for j in range(IPW if AG else 0):
                    token, col2, ok = item(t0, j, end)
                    offset = ok.select(token * ROW + col2, fx.Int32(_DROP))
                    rows.append((_load(own_landing, offset, 0, 4, aux=SYS), offset))
                for words, offset in rows:
                    _store(out_rsrc, words, offset)
                _store(ep_rsrc, [epoch], (tid == 0).select(tile * 4, fx.Int32(_DROP)))

            if OWNER == OWN_TILE:
                owner = _uniform(tile % WORLD)
            _when(owner != rank, copy)

        _loop(first, tiles, nwg, tile_body)
        _loop(first, tiles, nwg, copy_body)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a8w4_tp_reduce(
        p_loc: fx.Pointer,
        p_src: fx.Pointer,
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        slots: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
    ):
        body(p_loc, p_src, p_out, p_tab, p_epoch, p_err, M, rank, slots, limit, nwg)

    @flyc.jit
    def launch(
        p_loc: fx.Pointer,
        p_src: fx.Pointer,
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        slots: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_tp_reduce(
            p_loc, p_src, p_out, p_tab, p_epoch, p_err, M, rank, slots, limit, nwg
        ).launch(grid=(nwg, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
