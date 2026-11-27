# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Down with the cross-rank sum streamed out of it (``MegaMoeTPGfx942``, fused batches), gfx942.

``DownTP`` makes the Down kernel -- the prefill one (``moe_a8w4.gemm2``, ``TP=``, ``compile_down_tp``) or the
streaming one (``moe_a8w4.stream``, ``TP=``, ``compile_down_stream_tp``) -- the first of the two kernels of the
fused cross-rank sum, which it hides under the Down: while the compute workgroups produce the hidden dimension one
column block after the other, workgroups of the same launch (inserted between them) reduce the finished blocks over
the routes, send the partials to their owners (reduce-scatter) and, on the owner, sum them and push the result to
every rank (all-gather). The second kernel, ``compile_tp_gather`` (launched as ``a8w4_tp_reduce``), copies the rows
the other ranks summed from ``landing`` into ``out``. Same arithmetic, buffers and flag protocol as ``tp_reduce``
(its rules 1-6; ``SYS``, ``POISON``, ``ERR_TIMEOUT``): with route rows the output is bitwise that of ``tp_reduce``;
with ``COPY`` (the atomic streaming Down has summed the routes into this rank's ``out`` itself, in the order its
atomics land) it is that of the atomic Down followed by ``tp_reduce``'s copy mode up to that order. With ``OS``
(two ranks, rules 11-12) the prefill Down runs as token chunks without owners: the reduce workgroups store this
rank's partials into ``out``, an SDMA queue pushes each finished block of them to the peer while the next chunk
computes, and the second kernel adds the peer's partials from ``landing`` (bitwise the same output).

Tile: column block ``q``: hidden columns ``[q * CWD, (q + 1) * CWD)``, ``CWD = H / NSPLIT`` (a multiple of 512). Item
``(q, tile)``: the ``TT`` tokens ``tile * TT ..`` (the last tile masked at M) of block ``q``, owned by rank
``tile % WORLD``; its flag / epoch slot is ``v = q * tpq + tile`` (``tpq``: tiles of the largest batch). Unit: 512
columns of one token of an item, one wave (8 BF16 per lane); wave w of a workgroup takes the units ``w, w + 4, ..``
of an item (``UPW`` per wave).

Schedule (1-D grid, workgroups start in id order; ``DownTP.segments``, ``S = NSPLIT``, ``a`` / ``b`` = ``AX`` /
``AY`` eighths of ``task_pad``, rounded down to multiples of 4)::

    C(0) | C(1)[:a] X(0) C(1)[a:b] Y(0) C(1)[b:] | .. | C(S-1)[:a] X(S-2) C(S-1)[a:b] Y(S-2) C(S-1)[b:] | X(S-1) Y(S-1)

* C(s): ``task_pad`` compute workgroups, the Down of the ``CWD`` columns of block s of one sorted block (``e_idx``)
  each; route rows and ``loc`` (the latter by block 0 only) are stored written through (``ROUTE_AUX``; ``COPY``:
  agent-scope BF16 atomic adds into this rank's ``routes`` instead, no ``loc``); then ``s_waitcnt vmcnt(0)``,
  barrier, ``ctr[s] += 1`` (agent-scope atomic). Every compute workgroup counts, empty blocks included, so block
  s is complete when ``ctr[s] == task_pad``.
* X(q): ``NX`` reduce workgroups (``NXT`` for the last block); workgroup j waits until ``ctr[0]`` and ``ctr[q]``
  reach ``task_pad``, then for the tiles ``j, j + NX, ..``: this rank's partial (the item's ``TOPK`` route rows
  summed in f32 in slot order, missing routes +0, rounded to BF16; ``COPY``: the token's row of ``routes``) into
  the owner's inbox region ``(rank - owner - 1) % WORLD`` (an owner's own partial: its region ``WORLD - 1``), row
  ``(tile // WORLD) * TT + t - tile * TT``;
  every ``GX`` tiles ``s_waitcnt vmcnt(0)``, barrier, ``RS[v][rank] = epoch`` on the owner of each.
* Y(q): ``NY`` owner workgroups (``NYT`` for the last block); workgroup j takes groups of ``GY`` owned tiles ``rank
  + WORLD * i``, ``i`` from ``GY * (j + NY * m)``: waits for ``RS[v][s] == epoch`` of every rank s (its own
  included) and tile of the group (one poll), adds the ``WORLD`` partials in rank order in f32, rounds once,
  stores the rows into ``out`` and every peer's ``landing``; ``s_waitcnt vmcnt(0)``, barrier, ``AG[v] = epoch``
  (``POISON`` if the wait gave up) on every peer and ``epoch[v] = epoch`` for the group.
* gather (second kernel, ``G`` consecutive items per workgroup, block-major): for the items of the other ranks:
  waits for ``AG[v]`` (one poll for the ``G`` items), copies the ``landing`` rows into ``out`` (every load of the
  ``G`` items issued before the stores), ``epoch[v] = epoch``; workgroup 0 zeroes ``ctr``.

Steps of the X / Y loops (one step: what is summed, then the loads issued before its stores)::

    role  step                                    loads issued before the step's stores
    X     TOPK route rows of UX units (per wave)  the next step's routes (after the last step of tile i: tile
                                                  i + 1's first, and loc of tile i + 2)
    Y     WORLD partials of one owned tile        the next tile's partials (within the group)

Naming: ``tab`` int64 [3, WORLD] as int32: per rank the address of ``landing`` (BF16 [M_max, H]), ``inbox`` (BF16
``[WORLD * rps, H]``, ``rps = ceil(tiles / WORLD) * TT``) and ``flags`` (int32 ``[slots * (WORLD + 1)]``, ``slots =
NSPLIT_MAX * tpq`` whatever ``NSPLIT`` a call uses: ``RS[v][s]`` at ``v * WORLD + s``, ``AG[v]`` at ``slots * WORLD +
v``); ``epoch`` int32 [slots] and ``ctr`` int32 [NSPLIT] (this rank's, plain device memory, ``ctr`` zero between
calls); ``err`` int32 [1]. C / X / Y: compute / reduce / owner workgroups (``ROLE_*``). Token chunks of the one-shot
sum (``OS``, rules 11-12): ``tab`` [4, 2] (row 3 [0]: the address of this rank's ``sdma.SdmaQueues.desc``),
``flags`` ``MAX_CHUNKS * CHUNK_WORDS`` more words (``flag_words``; chunk c's at ``slots * 3 + c * CHUNK_WORDS``:
``CEP[o]`` / ``PSN[o]`` at ``CEP`` / ``PSN + o``, per column block q ``TRG[q]`` / ``ARR[q]`` / ``SNT[q]`` /
``XCN[q]`` at ``TRG`` / ``ARR`` / ``SNT`` / ``XCN + q``; ``RDY[o]`` at ``RDY + o`` of chunk 0's), ``ctr``
``ctr_words(True)`` (chunk c's counters at ``c * NSPLIT_MAX``).

Rules (in addition to tp_reduce's):

7. Coherence across the XCDs of a rank: whatever one workgroup stores for another one of the same launch (routes,
   ``loc``, partials) is written through (``SYS``) and read past the caches (``SYS``), behind the same release /
   acquire as the cross-rank flags (``ctr`` is updated by agent-scope atomics, polled with ``SYS`` loads).
8. No deadlock: compute workgroups never wait; X(q) waits for C(0) and C(q), before it in the grid; Y(q) for X(q)
   of every rank, before it in every rank's grid (``AX <= AY``); the gather for Y. Workgroups start in id order,
   so every wait ends (a rank that never comes: rule 5).
9. Reuse across calls: a rank writes partials (any path) only after its previous call ended, whose gather waited
   for the AG of every item it sent a partial for (the owner's reads came first); an owner's own region is read
   by its own Y in the same launch. Landing as in tp_reduce rule 3 (one-shot: rule 12).
10. X / Y loops (the table above): the loop carries the words loaded for the next step and nothing else is in
    flight when it starts (``s_waitcnt vmcnt(0)``), so its waits count only the previous step's memory operations;
    a group's flags are stored after ``s_waitcnt vmcnt(0)`` and a barrier, i.e. after the partial / sum stores of
    all of its tiles by every wave (the next step's loads are waited for there too).
11. Token chunks (``OS``; ``moe_a8w4.fused_moe_a8w4``, ``prefill_down.chunks``): chunk c, the tokens ``[t_lo,
    t_hi)`` (multiples of ``WORLD * TT`` but the call's end), is a Down launch of its own after chunk c - 1's, of
    the sorted blocks whose first route's token is in the chunk (``CHUNKED``): every route of its tokens is then
    complete (the sort keeps the token order within an expert; a block's routes of later tokens are computed with
    it). Its X take the chunk's tiles only; its compute workgroups count in its own ``ctr`` words (the final kernel
    zeroes all).
12. One-shot sum (``OS``, two ranks, ``MegaMoeTPGfx942.CHUNK_SPEC``; every chunk the same; peer ``o = 1 - rank``):
    no owner workgroups and no inbox. X stores this rank's partials into its own ``out`` (token rows, written
    through, ``SYS``); every X workgroup at its end (``s_waitcnt vmcnt(0)``, barrier) adds 1 to the chunk's
    ``XCN[q]``, the last of block q zeroes it and stores ``TRG[q] = CEP[rank] + 1`` (``| POISON`` if ``err`` is set,
    after ``PSN[rank] = (CEP[rank] + 1) | POISON`` on o). X(0) workgroup 0 first, wave 0: advances ``CEP[o]``;
    chunk 0 stores ``RDY[rank] = CEP[rank] + 1`` on o; it submits to SDMA queue 0 (``sdma.py``): chunk 0 POLLs until
    o's ``RDY[o]`` here reaches ``CEP[o]``, then per column block q: POLL until ``TRG[q] & _EPOCH_BITS == CEP[rank]
    + 1``, one 2-D copy of the block's columns of the chunk's rows (``2 * CWD`` bytes, ``2 * H`` apart) from ``out``
    into o's ``landing`` at the same place, FENCE ``ARR[q]`` on o and FENCE ``SNT[q]`` here (both ``CEP[rank] +
    1``); a FENCE on the queue's completion word. Chunk 0's submission needs the queue idle (else it submits
    nothing and sets ``ERR_TIMEOUT``: a wait of an earlier call gave up, ``reset()`` recreates the queues); later
    chunks queue behind it. The final kernel (``compile_tp_gather(OS=1)``) takes every item (all ranks' tiles): it waits
    (bounded, one poll for its ``G`` items) until ``ARR[q]`` (o's partials are in ``landing``) and ``SNT[q]`` (the
    queue has read this rank's ``out``) of the item's chunk have reached ``CEP[o]`` (``ERR_TIMEOUT`` if it gave up
    or ``PSN[o] == CEP[o] | POISON``), then stores ``out`` = the f32 sum of ``out`` and ``landing`` rounded once
    (the two partials' sum in rank order: bitwise the owner's of the other modes, on both ranks). At its end
    workgroup 0, wave 0, waits (bounded) for the queue's completion fence; if it gave up, it releases the queue's
    POLLs (``TRG`` of every chunk to ``CEP[rank] + 1 | POISON``, after ``PSN[rank]`` on o, and ``RDY[o]`` here),
    waits again and sets ``ERR_TIMEOUT``; then it advances ``CEP[rank]`` of every chunk. Between the Downs and the
    final kernel's end ``CEP[o] = CEP[rank] + 1``, so the final kernel reads ``CEP[o]`` only. Reuse: o's queue
    writes this rank's ``landing`` only after this rank's ``RDY`` of the call, stored when its first Down started,
    i.e. after its previous final kernel ended (whatever mode it was: the other modes' writers into ``landing``,
    o's owner workgroups, wait for this rank's partials of the same call); this rank's queue reads ``out`` before
    ``SNT``, which the final kernel waits for before storing the sums. The queues are the device's (shared by its
    kernels on one stream, ``sdma.py`` rule 1).
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import Vector as Vec

from ..communication_ops_utils import (
    atomic_add_global_at,
    atomic_or_global_at,
)
from ..moe_a8w4.gemm2 import compile_a8w4_gemm2
from ..moe_a8w4.stream import (
    _DROP,
    _address,
    _load,
    _pack64,
    _rsrc,
    _uniform,
    compile_a8w4_down_stream,
)
from . import sdma
from .tp_reduce import (
    _EPOCH_BITS,
    ERR_TIMEOUT,
    POISON,
    SYS,
    _ballot_any,
    _loop,
    _spin,
    _store,
    _sum_bf16,
    _when,
    _when_int,
)

# Workgroup roles of the streamed Down grid (module docstring).
ROLE_COMPUTE, ROLE_REDUCE, ROLE_OWNER = 0, 1, 2
# ``SYS`` with the non-temporal hint (nt): route rows written through as a stream (``DownTP.ROUTE_AUX``), the same
# scope. The Down alone 5.6 % faster at Qwen TP4 M 32768 than with ``SYS`` (20261126-213259-dtax-run-0510), but the
# reduce workgroups' reads of the rows miss more: a gain at large batches only (mega_moe_tp_gfx942.DOWN_TUNE).
SYS_NT = SYS | 2
# Column blocks the flag / epoch layout has room for (fixed, so calls with another NSPLIT find the same slots).
NSPLIT_MAX = 16
# Tile sizes (tokens) DownTP takes; the workspace sizes the slots and inbox rows for all of them.
TILE_TOKENS = (4, 8, 16, 32, 64)
# Token chunks of the one-shot sum (``DownTP.OS``, rules 11-12): chunks of a call, and the words of chunk c's region
# at ``flags`` word ``slots * (WORLD + 1) + c * CHUNK_WORDS``: CEP[o] / PSN[o] at CEP / PSN + o (o < 4), RDY[o] at
# RDY + o, and per column block q < NSPLIT_MAX TRG[q] / ARR[q] / SNT[q] / XCN[q] at TRG / ARR / SNT / XCN + q.
MAX_CHUNKS = 4
CHUNK_WORDS, CEP, PSN, RDY, TRG, ARR, SNT, XCN = 128, 4, 12, 16, 32, 48, 64, 80


def _os_submit_bytes(nsplit, stamps=0):
    """Bytes of one-shot SDMA submission of a chunk (``DownTP.OS``): the RDY POLL, per column block POLL, COPY2D
    and two FENCEs, the completion FENCE (``stamps``: ``DownTP.STAMP``, a TIMESTAMP after the RDY POLL and after
    every POLL and COPY2D), NOPs up to a multiple of 64 bytes."""
    return -(-(4 * (6 + 34 * nsplit + 4 + (3 * (1 + 2 * nsplit) if stamps else 0))) // 64) * 64


# DownTP.STAMP (timing analysis of rule 12) from ctr word ctr_words(True): per chunk a row of STAMP_SLOTS 64-byte
# slots (0: _plan_os, 1: SDMA past the RDY POLL, 2 + 2q / 3 + 2q: SDMA past the POLL of TRG[q] / its COPY2D,
# 12 + q: TRG[q] stored, 17 + q / 22 + q: reduce workgroup 0 of block q past / before its wait), then per final
# kernel workgroup (< STAMP_WGS) 32 bytes: its start, past its last wait, its end.
STAMP_SLOTS = 32
STAMP_WGS = 1024
STAMP_BYTES = MAX_CHUNKS * STAMP_SLOTS * 64 + STAMP_WGS * 32


def flag_words(slots, world, chunks):
    """Words of ``flags`` (module docstring, Naming) for ``slots`` slots, ``chunks``: room for the token chunks."""
    return slots * (world + 1) + (MAX_CHUNKS * CHUNK_WORDS if chunks else 0)


def ctr_words(chunks):
    """Words of ``ctr`` before the ``TRACE`` timestamps."""
    return NSPLIT_MAX * (MAX_CHUNKS if chunks else 1)


def _entry(tab, world, row, p):
    words = _load(tab, (row * world + p) * 8, 0, 2)
    return _pack64(fx.Uint32(_uniform(words[0])), fx.Uint32(_uniform(words[1])))


def _wait(p_err, limit, poll):
    """Wave 0 polls (bounded, tp_reduce rule 5), then a workgroup barrier; nonzero in wave 0 if it gave up."""
    tid = fx.Int32(gpu.thread_id("x"))
    lane = tid & 63
    wave = _uniform(tid >> 6)
    err_rsrc = _rsrc(_address(p_err), 4)

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


def _stamp():
    """``s_memrealtime`` (100 MHz) as Int64 (``DownTP.TRACE``)."""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.typing import T

    return fx.Int64(
        _llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
    )


def _stamp_at(p_ctr, offset, live):
    """``DownTP.STAMP``: stores ``s_memrealtime`` at byte ``offset`` of the stamp area where ``live``."""
    pair = Vec.from_elements([_stamp()], fx.Int64).bitcast(fx.Uint32)
    base = 4 * ctr_words(True)
    _store(
        _rsrc(_address(p_ctr), base + STAMP_BYTES),
        [pair[0], pair[1]],
        live.select(base + offset, fx.Int32(_DROP)),
        SYS,
    )


def _loop_carry(lb, ub, step, init, fn):
    """``state = fn(i, state)`` for ``i`` in ``range(lb, ub, step)`` as one scf.for carrying ``state`` (a list of
    32-bit values, passed to ``fn`` as Int32); returns the final state."""
    # Raw scf.for with iter_args: run-time bounds in a closure outside the kernel function (see tp_reduce._when).
    loc = ir.Location.unknown()
    for_op = scf.ForOp(
        fx.Int32(lb).ir_value(),
        fx.Int32(ub).ir_value(),
        fx.Int32(step).ir_value(),
        [fx.Int32(v).ir_value() for v in init],
        loc=loc,
    )
    with ir.InsertionPoint(for_op.body):
        state = [fx.Int32(v) for v in for_op.inner_iter_args]
        out = fn(fx.Int32(for_op.induction_variable), state)
        scf.YieldOp([fx.Int32(v).ir_value() for v in out], loc=loc)
    return [fx.Int32(v) for v in for_op.results]


@dataclass(frozen=True)
class DownTP:
    """Streamed cross-rank sum of the prefill Down (module docstring): ``WORLD`` ranks, hidden ``H``, ``TOPK``
    routes per token, ``TT`` tokens per tile, ``NSPLIT`` column blocks, ``NX`` / ``NY`` reduce / owner workgroups
    per block but the last (``NXT`` / ``NYT`` for the last, 0: ``NX`` / ``NY``), inserted after ``AX`` / ``AY``
    eighths of the compute workgroups of the next block (module docstring, Schedule). ``GX`` / ``GY``: tiles per
    group of a reduce / owner workgroup (one ``s_waitcnt vmcnt(0)``, barrier and flag store per group); ``UX``:
    units per wave of a reduce step (0: all of the tile's, ``UPW``). ``ROUTE_AUX``: cache policy of the compute
    workgroups' route and ``loc`` stores. ``COPY``: the Down has summed the routes of every token itself (the atomic
    streaming Down, into ``routes`` BF16 [M, H]): a reduce workgroup reads the token's row of ``routes`` in place of
    its ``TOPK`` route rows (``loc`` unused).

    Timing analysis only: ``DRY`` (wrong output) 1 = the Down alone (reduce / owner workgroups and the gather do
    nothing), 2 = the Down and this rank's reduction over the routes into ``out``, nothing sent, 3 = as 1 without
    the compute workgroups' counting (no ``s_waitcnt vmcnt(0)`` before they end), 4 = the Down and the reduce
    workgroups' reduce-scatter (owner workgroups and the gather do nothing). ``TRACE``: every workgroup stores its
    start and end ``s_memrealtime`` (int64 pair) at ``ctr`` word ``ctr_words(OS)`` + 4 * wg. ``STAMP`` (with
    ``OS``): the one-shot sum's timestamps (``STAMP_SLOTS``).

    ``OS`` (two ranks, no owner workgroups; the ``CHUNKED`` prefill Down of one token chunk, ``TPArgs.t_lo`` /
    ``t_hi`` / ``chunk``, rule 11): the one-shot sum, the reduce workgroups take the chunk's tiles only and store
    this rank's partials into ``out``, the SDMA queue pushes them to the peer (rule 12); the chunk's block counters
    are ``ctr`` words ``chunk * NSPLIT_MAX`` on.

    Compute workgroups' schedule (the prefill Down, ``compile_a8w4_gemm2``; the results are the same bits): ``PRO``
    issues the loads of the first two weight blocks, of the row scales and of ``loc``'s sorted row before the
    activations are gathered; ``SB`` puts a scheduling barrier after the middle barrier of the block loop (else the
    compiler hoists the next half's dequantization and MFMAs above it, next to their weight loads, whose wait then
    also waits for the route stores issued before them: ``vmcnt`` counts loads and stores in issue order); ``LAS``
    issues each half's weight loads before its route stores. ``WPE``: the kernel's ``waves_per_eu`` compile hint
    (0: none); ``SB`` / ``LAS`` raise the VGPRs: with ``WPE`` they keep the occupancy."""
    WORLD: int
    H: int
    TOPK: int
    TT: int
    NSPLIT: int
    NX: int
    NY: int
    AX: int = 2
    AY: int = 6
    NXT: int = 256
    NYT: int = 128
    GX: int = 4
    GY: int = 2
    UX: int = 0
    ROUTE_AUX: int = SYS
    COPY: int = 0
    DRY: int = 0
    TRACE: int = 0
    WPE: int = 0
    OS: int = 0
    STAMP: int = 0
    PRO: int = 0
    SB: int = 0
    LAS: int = 0

    def __post_init__(self):
        assert self.WORLD in (2, 4, 8), self.WORLD
        assert self.H % (512 * self.NSPLIT) == 0, (self.H, self.NSPLIT)
        assert self.NSPLIT <= NSPLIT_MAX, self.NSPLIT
        assert self.TT in TILE_TOKENS, self.TT
        assert 1 <= self.TOPK <= 64, self.TOPK
        for n in (self.NX, self.NY, self.NXT, self.NYT):
            assert n % 4 == 0, (self.NX, self.NY, self.NXT, self.NYT)
        # Rule 8: the owner workgroups of a block start after its reduce workgroups.
        assert 0 <= self.AX <= self.AY <= 8, (self.AX, self.AY)
        assert 1 <= self.GX <= 8 and 1 <= self.GY <= 8, (self.GX, self.GY)
        # _reduce loads a tile's slots of loc with one load per wave; _owner polls a group's flags in one.
        assert self._upw() * self._rows() <= 64, (self.TT, self.TOPK, self.COPY)
        assert self.UX >= 0 and self._upw() % (self.UX or self._upw()) == 0, self.UX
        assert self.GY * self.WORLD <= 64, (self.GY, self.WORLD)
        # OS: two ranks (one SDMA queue), no owner workgroups (rule 12).
        assert not self.OS or (self.WORLD == 2 and self.NY == 0 and self.NYT == 0 and not self.COPY), (
            self.OS,
            self.WORLD,
            self.NY,
            self.NYT,
            self.COPY,
        )
        assert not self.STAMP or (self.OS and not self.TRACE), (self.STAMP, self.OS, self.TRACE)

    def _rows(self):
        """Rows a reduce workgroup sums per token: the ``TOPK`` routes, one with ``COPY``."""
        return 1 if self.COPY else self.TOPK

    def _upw(self):
        """Units (``_units``) per wave of a tile."""
        return -(-self.TT * (self.H // self.NSPLIT // 512) // 4)

    def _counts(self, q):
        """Reduce / owner workgroups of block ``q`` (Python int: static; else run-time Int32)."""
        S, NX, NY = self.NSPLIT, self.NX, self.NY
        nxt, nyt = self.NXT or NX, self.NYT or NY
        if isinstance(q, int):
            return (nxt, nyt) if q == S - 1 else (NX, NY)
        last = q == S - 1
        pick = lambda tail, n: n if tail == n else last.select(fx.Int32(tail), fx.Int32(n))
        return pick(nxt, NX), pick(nyt, NY)

    def segments(self, task_pad):
        """The grid in launch order: (role, column block, index of the first workgroup within the role's block,
        count) per segment (``task_pad``: int or run-time Int32; the counts are multiples of 4)."""
        S, C, X, Y = self.NSPLIT, ROLE_COMPUTE, ROLE_REDUCE, ROLE_OWNER
        n = self._counts
        a = task_pad * self.AX // 32 * 4
        b = task_pad * self.AY // 32 * 4
        segs = [(C, 0, 0, task_pad)]
        for s in range(1, S):
            segs += [
                (C, s, 0, a),
                (X, s - 1, 0, n(s - 1)[0]),
                (C, s, a, b - a),
                (Y, s - 1, 0, n(s - 1)[1]),
                (C, s, b, task_pad - b),
            ]
        return segs + [(X, S - 1, 0, n(S - 1)[0]), (Y, S - 1, 0, n(S - 1)[1])]

    def grid(self, task_pad):
        nx, ny = self._counts(self.NSPLIT - 1)
        return self.NSPLIT * task_pad + (self.NSPLIT - 1) * (self.NX + self.NY) + nx + ny

    def roles(self, wg, task_pad):
        """(role, index within its role and block, column block) of workgroup ``wg`` (all wave-uniform)."""
        role, idx, blk = fx.Int32(ROLE_COMPUTE), wg, fx.Int32(0)
        start = fx.Int32(0)
        for r, q, lo, count in self.segments(task_pad):
            here = wg >= start
            role = here.select(fx.Int32(r), role)
            idx = here.select(wg - start + lo, idx)
            blk = here.select(fx.Int32(q), blk)
            start = start + count
        return role, idx, blk

    def trace(self, p_ctr, task_pad, at):
        """``TRACE``: thread 0 stores ``s_memrealtime`` at byte ``at`` (0: start, 8: end) of its workgroup's pair."""
        pair = Vec.from_elements([_stamp()], fx.Int64).bitcast(fx.Uint32)
        wg = fx.Int32(gpu.block_id("x"))
        base = 4 * ctr_words(self.OS)
        _store(
            _rsrc(_address(p_ctr), base + 16 * self.grid(task_pad)),
            [pair[0], pair[1]],
            (fx.Int32(gpu.thread_id("x")) == 0).select(base + wg * 16 + at, fx.Int32(_DROP)),
            SYS,
        )

    @staticmethod
    def args(**kw):
        return TPArgs(**kw)

    def finish(self, role, idx, blk, a):
        """Everything of a workgroup after its Down block (``a``: the kernel's TP arguments, ``TPArgs``)."""
        if self.DRY != 3:
            _when(role == ROLE_COMPUTE, lambda: self._signal(a, blk))
        if self.DRY not in (1, 3):
            _when(role == ROLE_REDUCE, lambda: self._reduce(a, blk, idx))
        if not self.DRY and not self.OS:
            _when(role == ROLE_OWNER, lambda: self._owner(a, blk, idx))
        if self.TRACE:
            self.trace(a.p_ctr, a.task_pad, 8)

    # -- roles ------------------------------------------------------------------------------------------------------
    def _signal(self, a, part):
        tid = fx.Int32(gpu.thread_id("x"))
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        word = part + a.chunk * NSPLIT_MAX if self.OS else part
        _when(
            tid == 0,
            lambda: atomic_add_global_at(
                _address(a.p_ctr) + fx.Int64(word * 4),
                1,
                syncscope=fx.rocdl.SyncScope.Agent,
            ),
        )

    def _geometry(self, a):
        W, TT = self.WORLD, self.TT
        ROW = 2 * self.H
        tiles = (a.M + (TT - 1)) // TT
        rps = (tiles + (W - 1)) // W * TT
        slots = a.tpq * NSPLIT_MAX
        tab = _rsrc(_address(a.p_tab), (4 if self.OS else 3) * W * 8)
        g = dict(
            ROW=ROW,
            tiles=tiles,
            rps=rps,
            slots=slots,
            tab=tab,
            inbox_bytes=rps * (W * ROW),
            flag_bytes=slots * ((W + 1) * 4),
            out_bytes=a.M * ROW,
            tile_lo=0,
            tile_hi=tiles,
        )
        if self.OS:
            # The chunk's tiles (its bounds are multiples of WORLD * TT but the call's end).
            g.update(
                flag_bytes=flag_words(slots, W, True) * 4,
                tile_lo=a.t_lo // TT,
                tile_hi=(a.t_hi + (TT - 1)) // TT,
                region=slots * (W + 1) + a.chunk * CHUNK_WORDS,
                t_lo=a.t_lo,
                t_hi=a.t_hi,
                chunk=a.chunk,
            )
        return g

    def _units(self, t0, q, M):
        """(token, byte offset in a row, liveness) of this wave's units of a tile in block ``q``."""
        CWD = self.H // self.NSPLIT
        CH = CWD // 512
        UNITS = self.TT * CH
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        units = []
        for u in range(-(-UNITS // 4)):
            idx = wave + 4 * u
            token = t0 + idx // CH
            col = q * CWD + (idx % CH) * 512 + lane * 8
            ok = token < M
            if 4 * u + 3 >= UNITS:
                ok = ok & (idx < UNITS)
            units.append((token, col * 2, ok))
        return units

    def _reduce(self, a, q, j):
        """X workgroup ``j`` of block ``q`` (module docstring): tiles ``j, j + n, ..`` (``n`` reduce workgroups of the
        block), a tile's units ``UX`` at a time (a step), the route loads of the next step (of the next tile after
        the last step, ``loc`` of the tile after it loaded meanwhile) issued before the partials of this one are
        stored (rule 10); the RS flags are released ``GX`` tiles at a time. ``COPY``: one row per token, its own
        row of ``routes`` (no ``loc``, nothing summed)."""
        W, TT, TOPK, G = self.WORLD, self.TT, self._rows(), self.GX
        CWD = self.H // self.NSPLIT
        CH = CWD // 512
        UNITS = TT * CH
        UPW = self._upw()
        UX = self.UX or UPW
        STEPS = UPW // UX
        g = self._geometry(a)
        ROW, rps, tab, tiles = g["ROW"], g["rps"], g["tab"], g["tiles"]
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        rank, M = a.rank, a.M
        inbox_addr = [] if self.OS else [_entry(tab, W, 1, p) for p in range(W)]
        flags = [] if self.OS else [_rsrc(_entry(tab, W, 2, p), g["flag_bytes"]) for p in range(W)]
        ep_rsrc = _rsrc(_address(a.p_epoch), g["slots"] * 4)
        if not self.COPY:
            loc_rsrc = _rsrc(_address(a.p_loc), M * (TOPK * 4))
        if self.OS:
            ctr_rsrc = _rsrc(_address(a.p_ctr), ctr_words(True) * 4)
            cbase = a.chunk * NSPLIT_MAX
        else:
            ctr_rsrc = _rsrc(_address(a.p_ctr), self.NSPLIT * 4)
            cbase = 0
        src_base = _address(a.p_routes)
        src_rsrc = _rsrc(src_base, g["out_bytes"]) if self.COPY else None
        out_rsrc = _rsrc(_address(a.p_out), g["out_bytes"])
        cnt = self._counts(q)[0]
        lo = g["tile_lo"]
        if self.OS:
            n = (g["tile_hi"] - lo - j + (cnt - 1)) // cnt
        else:
            n = (tiles - j + (cnt - 1)) // cnt

        def tile_of(i):
            return lo + j + cnt * i if self.OS else j + cnt * i

        if self.OS and not self.DRY:
            _when((q == 0) & (j == 0), lambda: self._plan_os(a, g))

        def pending():
            if self.OS:
                at = (lane == 0).select(cbase, cbase + q) * 4
            else:
                at = (lane == 0).select(fx.Int32(0), q * 4)
            seen = fx.Int32(
                _load(ctr_rsrc, (lane < 2).select(at, fx.Int32(_DROP)), 0, 1, aux=SYS)[
                    0
                ]
            )
            return _ballot_any((lane < 2) & (seen < a.task_pad))

        if self.STAMP:
            _stamp_at(a.p_ctr, (g["chunk"] * STAMP_SLOTS + 22 + q) * 64, (j == 0) & (tid == 0))
        _wait(a.p_err, a.limit, pending)
        if self.STAMP:
            _stamp_at(a.p_ctr, (g["chunk"] * STAMP_SLOTS + 17 + q) * 64, (j == 0) & (tid == 0))
        # Lane l < UPW * TOPK holds slot l % TOPK of the token of unit l // TOPK (one load per tile).
        lu = lane // TOPK
        lk = lane - lu * TOPK

        def unit(i, u):
            idx = wave + 4 * u
            token = tile_of(i) * TT + idx // CH
            ok = (i < n) & (token < M)
            if 4 * u + 3 >= UNITS:
                ok = ok & (idx < UNITS)
            return token, (q * CWD + (idx % CH) * 512 + lane * 8) * 2, ok

        def loc_load(i):
            if self.COPY:
                return fx.Int32(0)
            idx = wave + 4 * lu
            token = tile_of(i) * TT + idx // CH
            ok = (lane < UPW * TOPK) & (i < n) & (token < M) & (idx < UNITS)
            offset = ok.select(token * (TOPK * 4) + lk * 4, fx.Int32(_DROP))
            return fx.Int32(_load(loc_rsrc, offset, 0, 1, aux=SYS)[0])

        def step_loads(i, s, slots):
            words = []
            for u in range(s * UX, (s + 1) * UX):
                token, col2, ok = unit(i, u)
                if self.COPY:
                    words += _load(src_rsrc, ok.select(token * ROW + col2, fx.Int32(_DROP)), 0, 4, aux=SYS)
                    continue
                for k in range(TOPK):
                    loc = fx.Int32(
                        rocdl.readlane(
                            fx.Int32.ir_type,
                            slots.ir_value(),
                            fx.Int32(u * TOPK + k).ir_value(),
                        )
                    )
                    live = ok & (loc >= 0)
                    # One row per descriptor from a 64-bit base: routes may exceed the 2 GiB voffset range.
                    row = _rsrc(src_base + fx.Int64(live.select(loc, 0)) * ROW, ROW)
                    words += _load(row, live.select(col2, fx.Int32(_DROP)), 0, 4, aux=SYS)
            return words

        def release(i):
            """RS flags of the group of tiles ending at tile i (lanes 0 .. GX - 1 of wave 0)."""
            first = i - i % G
            tile = tile_of(first + tid)
            ok = (tid < G) & (first + tid <= i) & (tile < tiles)
            v = q * a.tpq + tile
            epoch = fx.Int32(_load(ep_rsrc, ok.select(v * 4, fx.Int32(_DROP)), 0, 1)[0]) + 1
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            owner = tile % W
            for p in range(W):
                _store(
                    flags[p],
                    [epoch],
                    (ok & (owner == p)).select((v * W + rank) * 4, fx.Int32(_DROP)),
                    SYS,
                )

        def body(i, state):
            words, loc_i, loc_next = state[:-2], state[-2], state[-1]
            tile = tile_of(i)
            if self.OS:
                # One-shot: this rank's partials into its own out, written through (its SDMA queue reads them).
                dest, base = out_rsrc, 0
            else:
                owner = _uniform(tile % W)
                region = (rank + (W - 1) - owner) & (W - 1)
                base = region * rps + (tile // W) * TT - tile * TT
                address = inbox_addr[0]
                for p in range(1, W):
                    address = (owner == p).select(inbox_addr[p], address)
                dest = _rsrc(address, g["inbox_bytes"])
            loc_far = None
            for s in range(STEPS):
                rows = [[fx.Uint32(w) for w in words[4 * m : 4 * m + 4]] for m in range(UX * TOPK)]
                if self.COPY:
                    partials = rows
                else:
                    partials = [_sum_bf16(rows[v * TOPK : (v + 1) * TOPK]) for v in range(UX)]
                if s + 1 < STEPS:
                    words = step_loads(i, s + 1, loc_i)  # next step of this tile
                else:
                    words = step_loads(i + 1, 0, loc_next)  # first step of the next tile
                    loc_far = loc_load(i + 2)
                for v in range(UX):
                    token, col2, ok = unit(i, s * UX + v)
                    if self.DRY == 2:
                        _store(out_rsrc, partials[v], ok.select(token * ROW + col2, fx.Int32(_DROP)))
                    else:
                        _store(
                            dest,
                            partials[v],
                            ok.select((base + token) * ROW + col2, fx.Int32(_DROP)),
                            SYS,
                        )
            if self.DRY != 2 and not self.OS:
                _when((i % G == G - 1) | (i == n - 1), lambda: release(i))
            return words + [loc_next, loc_far]

        loc0 = loc_load(fx.Int32(0))
        start = step_loads(fx.Int32(0), 0, loc0) + [loc0, loc_load(fx.Int32(1))]
        # Nothing in flight into the loop: its waits count only the previous iteration's memory operations.
        rocdl.s_waitcnt(vmcnt=0)
        _loop_carry(0, n, 1, start, body)
        if self.OS and self.DRY != 2:
            self._xdone(a, g, q, cnt)

    def _owner(self, a, q, j):
        """Y workgroup ``j`` of block ``q`` (module docstring): ``GY`` owned tiles at a time, one poll for the group's
        RS flags, the partial loads of the next tile issued before the sums of this one are stored (rule 10), one
        ``s_waitcnt vmcnt(0)`` and barrier before the group's AG flags (``POISON`` on all of them if the wait gave
        up)."""
        W, TT, G = self.WORLD, self.TT, self.GY
        UPW = self._upw()
        g = self._geometry(a)
        ROW, rps, tab = g["ROW"], g["rps"], g["tab"]
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        rank, M = a.rank, a.M
        landing = [_rsrc(_entry(tab, W, 0, p), g["out_bytes"]) for p in range(W)]
        flags = [_rsrc(_entry(tab, W, 2, p), g["flag_bytes"]) for p in range(W)]
        own_inbox = _rsrc(_entry(tab, W, 1, rank), g["inbox_bytes"])
        own_flags = _rsrc(_entry(tab, W, 2, rank), g["flag_bytes"])
        out_rsrc = _rsrc(_address(a.p_out), g["out_bytes"])
        ep_rsrc = _rsrc(_address(a.p_epoch), g["slots"] * 4)
        ag0 = g["slots"] * W
        owned = (g["tiles"] - rank + (W - 1)) // W
        cnt = self._counts(q)[1]
        n = (owned - G * j + (G * cnt - 1)) // (G * cnt)

        def loads(i):
            tile = rank + W * i
            t0 = tile * TT
            parts = []
            for token, col2, ok in self._units(t0, q, M):
                for s in range(W):
                    row = ((s + (W - 1) - rank) & (W - 1)) * rps + i * TT - t0 + token
                    parts.append(
                        _load(
                            own_inbox,
                            (ok & (i < owned)).select(row * ROW + col2, fx.Int32(_DROP)),
                            0,
                            4,
                            aux=SYS,
                        )
                    )
            return parts

        def group(m):
            first = G * (j + cnt * m)
            # Lane l < GY * WORLD polls RS[v][l % WORLD] of owned tile first + l // WORLD.
            il = first + lane // W
            ok_l = (lane < G * W) & (il < owned)
            v_l = q * a.tpq + rank + W * il
            want = fx.Int32(_load(ep_rsrc, ok_l.select(v_l * 4, fx.Int32(_DROP)), 0, 1)[0]) + 1

            def pending():
                at = ok_l.select((v_l * W + (lane & (W - 1))) * 4, fx.Int32(_DROP))
                seen = fx.Int32(_load(own_flags, at, 0, 1, aux=SYS)[0])
                return _ballot_any(ok_l & (seen != want))

            # Wave 0 poisons the group's AG flags if a partial is missing (tp_reduce rule 5).
            poison = (_wait(a.p_err, a.limit, pending) != 0).select(
                fx.Int32(POISON), fx.Int32(0)
            )
            parts = loads(first)
            for k in range(G):
                i = first + k
                units = self._units((rank + W * i) * TT, q, M)
                totals = [_sum_bf16(parts[u * W : (u + 1) * W]) for u in range(UPW)]
                if k + 1 < G:
                    parts = loads(i + 1)
                for total, (token, col2, ok) in zip(totals, units):
                    ok = ok & (i < owned)
                    offset = token * ROW + col2
                    _store(out_rsrc, total, ok.select(offset, fx.Int32(_DROP)))
                    for p in range(W):
                        _store(
                            landing[p],
                            total,
                            (ok & (rank != p)).select(offset, fx.Int32(_DROP)),
                            SYS,
                        )
            ok = (tid < G) & (first + tid < owned)
            v = q * a.tpq + rank + W * (first + tid)
            epoch = fx.Int32(_load(ep_rsrc, ok.select(v * 4, fx.Int32(_DROP)), 0, 1)[0]) + 1
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            ag = (epoch & _EPOCH_BITS) | poison
            for p in range(W):
                _store(
                    flags[p],
                    [ag],
                    (ok & (rank != p)).select((ag0 + v) * 4, fx.Int32(_DROP)),
                    SYS,
                )
            _store(ep_rsrc, [epoch], ok.select(v * 4, fx.Int32(_DROP)))

        _loop(0, n, 1, group)

    # -- one-shot: this rank's partials pushed to the peer by SDMA (OS, rule 12) ---------------------------------
    def _plan_os(self, a, g):
        """Reduce workgroup 0 of block 0 before it waits, wave 0, peer o = 1 - rank: advances CEP[o]; chunk 0 stores
        this rank's RDY flag on o (this call has started, its landing is free); SDMA queue 0 then (chunk 0: once
        o's RDY flag here holds o's epoch), per column block q: once TRG[q] holds the chunk epoch, one 2-D copy of
        the block's columns of the chunk's rows from ``out`` into o's ``landing``, FENCEs storing the epoch as
        ARR[q] on o and SNT[q] here; a FENCE on the queue's completion word (rule 12)."""
        W, S = self.WORLD, self.NSPLIT
        ROW = g["ROW"]
        BLOCK = 2 * (self.H // S)
        NBYTES = _os_submit_bytes(S, self.STAMP)
        assert MAX_CHUNKS * NBYTES <= sdma.RING_BYTES // 2, NBYTES  # a call's submissions (sdma.py rule 1)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        rank, region, chunk = a.rank, g["region"], g["chunk"]
        o = 1 - rank
        base = g["slots"] * (W + 1)
        drop = fx.Int32(_DROP)
        own_address = _entry(g["tab"], W, 2, rank)
        own_flags = _rsrc(own_address, g["flag_bytes"])
        peer_address = _entry(g["tab"], W, 2, o)
        peer_flags = _rsrc(peer_address, g["flag_bytes"])
        land = _entry(g["tab"], W, 0, o)
        out_address = fx.Int64(_address(a.p_out))
        stamp_address = fx.Int64(_address(a.p_ctr)) + 4 * ctr_words(True)

        def stamp(slot):
            if not self.STAMP:
                return []
            return sdma.stamp(stamp_address + fx.Int64((chunk * STAMP_SLOTS + slot) * 64))

        def queue():
            if self.STAMP:
                _stamp_at(a.p_ctr, chunk * (STAMP_SLOTS * 64), lane == 0)
            ring = sdma.Ring(_entry(g["tab"], W, 3, 0), 0, lane)
            wptr = ring.wptr()
            done = ring.done()
            # The call's first submission needs the previous call's complete (its final kernel waited for them);
            # later chunks queue behind it.
            go = (chunk != 0) | (done == fx.Int32(wptr))
            stuck = (chunk == 0) & (done != fx.Int32(wptr))
            offset, pad, new = sdma.place(wptr, NBYTES)
            at = (region + CEP + o) * 4
            e = fx.Int32(_uniform(_load(own_flags, at, 0, 1, aux=SYS)[0])) + 1
            _store(own_flags, [e], (lane == 0).select(at, drop), SYS)
            mine = fx.Int32(_uniform(_load(own_flags, (region + CEP + rank) * 4, 0, 1, aux=SYS)[0])) + 1
            mine = mine & _EPOCH_BITS
            first = chunk == 0
            _store(peer_flags, [mine], (first & (lane == 0)).select((base + RDY + rank) * 4, drop), SYS)
            ready = own_address + fx.Int64((base + RDY + o) * 4)
            words = sdma.select(first, sdma.poll(ready, e & _EPOCH_BITS, _EPOCH_BITS), sdma.nops(6))
            words += stamp(1)
            rows = g["t_hi"] - g["t_lo"]
            start = fx.Int64(g["t_lo"]) * ROW
            for q in range(S):
                at_q = start + q * BLOCK
                words += sdma.poll(own_address + fx.Int64((region + TRG + q) * 4), mine, _EPOCH_BITS)
                words += stamp(2 + 2 * q)
                words += sdma.copy2d(out_address + at_q, land + at_q, ROW, BLOCK, rows)
                words += stamp(3 + 2 * q)
                words += sdma.fence(peer_address + fx.Int64((region + ARR + q) * 4), mine)
                words += sdma.fence(own_address + fx.Int64((region + SNT + q) * 4), mine)
            words += sdma.fence(ring.done_address, fx.Int32(new))
            words += sdma.nops(NBYTES // 4 - len(words))

            def submit():
                ring.pad(wptr, pad)
                ring.store(offset, words, lane == 0)
                ring.commit(new)

            _when(go, submit)
            # The previous call's copies are not complete (a wait that gave up): no copies this call.
            _when(stuck & (lane == 0), lambda: atomic_or_global_at(_address(a.p_err), ERR_TIMEOUT))

        _when(wave == 0, queue)

    def _xdone(self, a, g, q, total):
        """Every reduce workgroup at its end (``OS``): once its partials are stored, counts itself in the chunk's
        XCN[q] (``total`` reduce workgroups of block q); the last one zeroes XCN[q] and stores TRG[q] = the chunk
        epoch, the release of the block's SDMA push (``| POISON`` if a wait of this rank gave up, ``err``: after
        PSN[rank] = epoch | ``POISON`` on the peer), rule 12."""
        W = self.WORLD
        tid = fx.Int32(gpu.thread_id("x"))
        rank, region = a.rank, g["region"]
        own_address = _entry(g["tab"], W, 2, rank)
        own_flags = _rsrc(own_address, g["flag_bytes"])
        peer_flags = _rsrc(_entry(g["tab"], W, 2, 1 - rank), g["flag_bytes"])
        drop = fx.Int32(_DROP)
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()

        def count():
            old = fx.Int32(
                atomic_add_global_at(
                    own_address + fx.Int64((region + XCN + q) * 4),
                    1,
                    syncscope=fx.rocdl.SyncScope.Agent,
                )
            )

            def fire():
                e = fx.Int32(_load(own_flags, (region + CEP + rank) * 4, 0, 1, aux=SYS)[0]) + 1
                e = e & _EPOCH_BITS
                failed = fx.Int32(_load(_rsrc(_address(a.p_err), 4), 0, 0, 1, aux=SYS)[0]) != 0
                poison = failed.select(fx.Int32(POISON), fx.Int32(0))
                _store(peer_flags, [e | POISON], failed.select((region + PSN + rank) * 4, drop), SYS)
                rocdl.s_waitcnt(vmcnt=0)
                _store(own_flags, [fx.Int32(0)], (region + XCN + q) * 4, SYS)
                _store(own_flags, [e | poison], (region + TRG + q) * 4, SYS)
                if self.STAMP:
                    _stamp_at(a.p_ctr, (g["chunk"] * STAMP_SLOTS + 12 + q) * 64, tid == 0)

            _when(old == total - 1, fire)

        _when(tid == 0, count)


@dataclass
class TPArgs:
    """The streamed Down kernel's arguments its TP roles use (``DownTP.finish``)."""

    p_loc: object
    p_routes: object
    p_out: object
    p_ctr: object
    p_tab: object
    p_epoch: object
    p_err: object
    M: object
    rank: object
    tpq: object
    limit: object
    task_pad: object
    t_lo: object = None
    t_hi: object = None
    chunk: object = None


@functools.cache
def compile_down_tp(*, N, K, TOPK, BLOCK_TILE_SIZE_M, E, XCD_CHUNK, tp):
    """The streamed prefill Down launcher (``compile_a8w4_gemm2`` with ``TP=tp``, ``CHUNKED`` with ``tp.OS``): the
    arguments of the prefill Down up to ``loc``, then ``out``, ``ctr``, ``tab``, ``epoch``, ``err``, M, rank, tpq,
    limit, task_num, stream."""
    return compile_a8w4_gemm2(
        N=N,
        K=K,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        E=E,
        NSPLIT=tp.NSPLIT,
        XCD_CHUNK=XCD_CHUNK,
        TP=tp,
        CHUNKED=bool(tp.OS),
    )


@functools.cache
def compile_down_stream_tp(*, N, K, TOPK, BLOCK_M, E, ATOMIC, R, SHARE, PREFETCH, W_AUX, tp):
    """The streamed streaming-tier Down launcher (``compile_a8w4_down_stream`` with four waves and ``TP=tp``): the
    arguments of the streaming Down up to ``task_num``, then ``loc``, ``out``, ``ctr``, ``tab``, ``epoch``, ``err``,
    rank, tpq, limit, stream. With ``ATOMIC`` (``tp.COPY``) the Down adds the routes into its ``out`` argument (the
    result ``out``, zeroed by the caller)."""
    return compile_a8w4_down_stream(
        N=N,
        K=K,
        TOPK=TOPK,
        BLOCK_M=BLOCK_M,
        R=R,
        W=4,
        PREFETCH=PREFETCH,
        SHARE=SHARE,
        E=E,
        ATOMIC=ATOMIC,
        W_AUX=W_AUX,
        TP=tp,
    )


def _sdma_finish_os(W, g, rank, p_err, limit, nags, S):
    """The one-shot final kernel's workgroup 0 after its items (``OS``), wave 0: waits (bounded) until SDMA queue 0
    is idle; if it gave up, releases the queue's polls (this rank's TRG flags of each chunk ``< nags`` to the chunk
    epoch | ``POISON``, its PSN flag on the peer first, and the peer's RDY flag here), waits once more and sets
    ``ERR_TIMEOUT``. Then thread 0 advances ``CEP[rank]`` of those chunks (rule 12)."""
    tid = fx.Int32(gpu.thread_id("x"))
    lane = tid & 63
    wave = _uniform(tid >> 6)
    o = 1 - rank
    own_flags = _rsrc(_entry(g["tab"], W, 2, rank), g["flag_bytes"])
    peer_flags = _rsrc(_entry(g["tab"], W, 2, o), g["flag_bytes"])
    base = g["slots"] * (W + 1)
    drop = fx.Int32(_DROP)

    def queue():
        ring = sdma.Ring(_entry(g["tab"], W, 3, 0), 0, lane)
        wptr = fx.Int32(ring.wptr())

        def busy():
            return ring.done() != wptr

        def release():
            def chunk(c):
                region = base + c * CHUNK_WORDS
                # The chunk epoch: CEP[o] (advanced by this call's _plan_os) = CEP[rank] + 1.
                e = fx.Int32(_uniform(_load(own_flags, (region + CEP + o) * 4, 0, 1, aux=SYS)[0])) & _EPOCH_BITS
                _store(peer_flags, [e | POISON], (lane == 0).select((region + PSN + rank) * 4, drop), SYS)
                rocdl.s_waitcnt(vmcnt=0)
                _store(own_flags, [e | POISON], (lane < S).select((region + TRG + lane) * 4, drop), SYS)
                _store(own_flags, [e], ((lane == 0) & (c == 0)).select((base + RDY + o) * 4, drop), SYS)

            _loop(0, nags, 1, chunk)
            _spin(busy, limit)
            _when(lane == 0, lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT))

        _when(_spin(busy, limit), release)

    _when(wave == 0, queue)
    gpu.barrier()

    def advance(c):
        at = (base + c * CHUNK_WORDS + CEP + rank) * 4
        e = fx.Int32(_uniform(_load(own_flags, at, 0, 1, aux=SYS)[0])) + 1
        _store(own_flags, [e], (tid == 0).select(at, drop), SYS)

    _loop(0, nags, 1, advance)


def _os_gather(spec, G):
    """The body of the one-shot final kernel (``compile_tp_gather(OS=1)``, rule 12)."""
    WORLD, TT, NSPLIT = spec.WORLD, spec.TT, spec.NSPLIT

    def body(p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg, s1, s2, s3, nags):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        first = fx.Int32(gpu.block_id("x"))
        a = TPArgs(
            p_loc=None,
            p_routes=None,
            p_out=p_out,
            p_ctr=p_ctr,
            p_tab=p_tab,
            p_epoch=p_epoch,
            p_err=p_err,
            M=M,
            rank=rank,
            tpq=tpq,
            limit=limit,
            task_pad=None,
            t_lo=fx.Int32(0),
            t_hi=M,
            chunk=fx.Int32(0),
        )
        g = spec._geometry(a)
        ROW = g["ROW"]
        own_landing = _rsrc(_entry(g["tab"], WORLD, 0, rank), g["out_bytes"])
        own_flags = _rsrc(_entry(g["tab"], WORLD, 2, rank), g["flag_bytes"])
        out_rsrc = _rsrc(_address(p_out), g["out_bytes"])
        drop = fx.Int32(_DROP)
        # The Downs have ended: every chunk's block counters restart from zero for the next call.
        words = ctr_words(True)
        _store(
            _rsrc(_address(p_ctr), words * 4),
            [fx.Int32(0)],
            ((first == 0) & (tid < words)).select(tid * 4, drop),
            SYS,
        )
        tiles = g["tiles"]
        total = NSPLIT * tiles
        base = g["slots"] * (WORLD + 1)
        o = 1 - rank
        record = MAX_CHUNKS * STAMP_SLOTS * 64 + first * 32
        recorded = (tid == 0) & (first < STAMP_WGS)
        if spec.STAMP:
            _stamp_at(p_ctr, record, recorded)

        def group(i):
            # Lane l < G (and 32 + l): ARR (SNT) of item f0 + l's chunk and column block; items f = q * tiles + tile.
            f0 = i * G
            arr = lane < 32
            f = f0 + (lane & 31)
            q = f // tiles
            tile = f - q * tiles
            live = ((lane & 31) < G) & (f < total)
            c = (
                (tile >= s1).select(fx.Int32(1), fx.Int32(0))
                + (tile >= s2).select(fx.Int32(1), fx.Int32(0))
                + (tile >= s3).select(fx.Int32(1), fx.Int32(0))
            )
            region = base + c * CHUNK_WORDS
            # The chunk epoch: CEP[o], advanced by this call's _plan_os (= CEP[rank] + 1 until this kernel ends).
            want = fx.Int32(_load(own_flags, live.select((region + CEP + o) * 4, drop), 0, 1, aux=SYS)[0])
            want = want & _EPOCH_BITS
            at = live.select((region + arr.select(fx.Int32(ARR), fx.Int32(SNT)) + q) * 4, drop)

            def pending():
                seen = fx.Int32(_load(own_flags, at, 0, 1, aux=SYS)[0])
                # Reached: epochs modulo _EPOCH_BITS + 1.
                return _ballot_any(live & (((seen - want) & _EPOCH_BITS) >= (POISON >> 1)))

            _wait(p_err, limit, pending)
            if spec.STAMP:
                _stamp_at(p_ctr, record + 8, recorded)
            check = live & arr & (tid < 64)
            psn = fx.Int32(_load(own_flags, check.select((region + PSN + o) * 4, drop), 0, 1, aux=SYS)[0])
            _when(
                check & (psn == (want | POISON)),
                lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
            )
            rows = []
            for k in range(G):
                fk = f0 + k
                qk = fk // tiles
                tk = fk - qk * tiles
                inside = fk < total
                for token, col2, ok in spec._units(tk * TT, qk, M):
                    offset = (ok & inside).select(token * ROW + col2, drop)
                    rows.append(
                        (_load(out_rsrc, offset, 0, 4), _load(own_landing, offset, 0, 4, aux=SYS), offset)
                    )
            # Two partials: their f32 sum is that of the ranks' order (one rounding, as the owner's in tp_reduce).
            for mine, peer, offset in rows:
                pair = [[fx.Uint32(w) for w in mine], [fx.Uint32(w) for w in peer]]
                _store(out_rsrc, _sum_bf16(pair), offset)

        if not spec.DRY:
            _loop(first, (total + (G - 1)) // G, nwg, group)
            _when(first == 0, lambda: _sdma_finish_os(WORLD, g, rank, p_err, limit, nags, NSPLIT))
        if spec.STAMP:
            _stamp_at(p_ctr, record + 16, recorded)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a8w4_tp_reduce(
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_ctr: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
        s1: fx.Int32,
        s2: fx.Int32,
        s3: fx.Int32,
        nags: fx.Int32,
    ):
        body(p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg, s1, s2, s3, nags)

    @flyc.jit
    def launch_os(
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_ctr: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
        s1: fx.Int32,
        s2: fx.Int32,
        s3: fx.Int32,
        nags: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_tp_reduce(
            p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg, s1, s2, s3, nags
        ).launch(grid=(nwg, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_os


@functools.cache
def compile_tp_gather(*, WORLD, H, TT, NSPLIT, DRY=0, G=1, OS=0, STAMP=0):
    """Launcher ``(out, tab, epoch, ctr, err, M, rank, tpq, limit, nwg, stream)`` of the second kernel (module
    docstring); grid: ``nwg`` (``ceil(NSPLIT * ceil(M / TT) / G)``) workgroups of 256 threads, ``G`` consecutive
    items each (one poll of their ``AG`` flags, lane g of wave 0 the item's g; then the loads of all of their rows
    before the stores). ``DRY``: ``DownTP.DRY`` (only ``ctr`` is zeroed).

    ``OS`` (after the one-shot Downs of token chunks, ``DownTP.OS``, rules 11-12; ``tab`` [4, 2]): four arguments
    follow ``nwg``: ``s1``, ``s2``, ``s3`` (the first tiles of chunks 1 .. 3, the call's tiles for the chunks it does
    not have) and ``nags`` (its chunks). Every item (all ranks') sums this rank's partial in ``out`` and the peer's
    in ``landing`` into ``out`` once both SDMA pushes of its chunk and column block are complete (one bounded poll of
    their ARR and SNT flags for the ``G`` items); workgroup 0 at its end waits for this rank's queue (bounded) and
    advances the chunks' ``CEP[rank]``. ``STAMP`` (with ``OS``): ``DownTP.STAMP``'s per-workgroup stamps."""
    assert 1 <= G <= (32 if OS else 64), G
    assert not STAMP or OS, (STAMP, OS)
    if OS:
        spec = DownTP(WORLD=WORLD, H=H, TOPK=1, TT=TT, NSPLIT=NSPLIT, NX=4, NY=0, NYT=0, DRY=DRY, OS=1, STAMP=STAMP)
        return _os_gather(spec, G)
    spec = DownTP(WORLD=WORLD, H=H, TOPK=1, TT=TT, NSPLIT=NSPLIT, NX=4, NY=4, DRY=DRY)

    def body(p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        first = fx.Int32(gpu.block_id("x"))
        a = TPArgs(
            p_loc=None,
            p_routes=None,
            p_out=p_out,
            p_ctr=p_ctr,
            p_tab=p_tab,
            p_epoch=p_epoch,
            p_err=p_err,
            M=M,
            rank=rank,
            tpq=tpq,
            limit=limit,
            task_pad=None,
        )
        g = spec._geometry(a)
        ROW, tiles = g["ROW"], g["tiles"]
        own_landing = _rsrc(_entry(g["tab"], WORLD, 0, rank), g["out_bytes"])
        own_flags = _rsrc(_entry(g["tab"], WORLD, 2, rank), g["flag_bytes"])
        out_rsrc = _rsrc(_address(p_out), g["out_bytes"])
        ep_rsrc = _rsrc(_address(p_epoch), g["slots"] * 4)
        ag0 = g["slots"] * WORLD
        # The streamed Down has ended: its counters restart from zero for the next call.
        _store(
            _rsrc(_address(p_ctr), NSPLIT * 4),
            [fx.Int32(0)],
            ((first == 0) & (tid < NSPLIT)).select(tid * 4, fx.Int32(_DROP)),
            SYS,
        )

        total = NSPLIT * tiles
        drop = fx.Int32(_DROP)

        def item(f):
            q = f // tiles
            tile = f - q * tiles

            def copy():
                v = q * tpq + tile
                epoch = fx.Int32(_uniform(_load(ep_rsrc, v * 4, 0, 1)[0])) + 1
                want = epoch & _EPOCH_BITS

                def pending():
                    seen = fx.Int32(_load(own_flags, (ag0 + v) * 4, 0, 1, aux=SYS)[0])
                    _when(
                        (lane == 0) & (seen == (want | POISON)),
                        lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
                    )
                    return _ballot_any((seen & _EPOCH_BITS) != want)

                _wait(p_err, limit, pending)
                rows = []
                for token, col2, ok in spec._units(tile * TT, q, M):
                    offset = ok.select(token * ROW + col2, drop)
                    rows.append((_load(own_landing, offset, 0, 4, aux=SYS), offset))
                for words, offset in rows:
                    _store(out_rsrc, words, offset)
                _store(ep_rsrc, [epoch], (tid == 0).select(v * 4, drop))

            _when(_uniform(tile % WORLD) != rank, copy)

        def group(i):
            # Lane g < G: item f0 + g (its epoch slot and AG flag); items of this rank or past the end: none.
            f0 = i * G
            f = f0 + lane
            q = f // tiles
            tile = f - q * tiles
            live = (lane < G) & (f < total) & ((tile % WORLD) != rank)
            v = q * tpq + tile
            epoch = fx.Int32(_load(ep_rsrc, live.select(v * 4, drop), 0, 1)[0]) + 1
            want = epoch & _EPOCH_BITS

            def pending():
                seen = fx.Int32(
                    _load(own_flags, live.select((ag0 + v) * 4, drop), 0, 1, aux=SYS)[0]
                )
                _when(
                    live & (seen == (want | POISON)),
                    lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
                )
                return _ballot_any(live & ((seen & _EPOCH_BITS) != want))

            _wait(p_err, limit, pending)
            rows = []
            for g in range(G):
                fg = f0 + g
                qg = fg // tiles
                tg = fg - qg * tiles
                mine = (fg < total) & ((tg % WORLD) != rank)
                for token, col2, ok in spec._units(tg * TT, qg, M):
                    offset = (ok & mine).select(token * ROW + col2, drop)
                    rows.append((_load(own_landing, offset, 0, 4, aux=SYS), offset))
            for words, offset in rows:
                _store(out_rsrc, words, offset)
            _store(ep_rsrc, [epoch], (live & (tid < 64)).select(v * 4, drop))

        if not DRY:
            if G == 1:
                _loop(first, total, nwg, item)
            else:
                _loop(first, (total + (G - 1)) // G, nwg, group)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def a8w4_tp_reduce(
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_ctr: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
    ):
        body(p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg)

    @flyc.jit
    def launch(
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_ctr: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_tp_reduce(
            p_out, p_tab, p_epoch, p_ctr, p_err, M, rank, tpq, limit, nwg
        ).launch(grid=(nwg, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
