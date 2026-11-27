# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route sum and cross-rank sum in one kernel for the TP4 batches between the decode path and the fused Down
(``MegaMoeTPGfx942``: more than ``DECODE_TP_MAX_M`` rows, BF16 output below ``FUSED_MIN_BYTES``) whose op ends in a
route sum (``a8w4_route_sum``: the non-atomic streaming / prefill Down), gfx942.

``compile_sum_tp`` replaces the op's route sum **and** the all-reduce after it (``DEST`` "dp": the reduce-scatter):
every rank enters with the routes of its slice of the intermediate dimension and leaves with the sum over the ranks.
On these sizes the custom all-reduce (aiter's Lamport kernels) beats ``tp_reduce`` / ``down_tp``, whose separate
flags cost a cross-GPU trip of their own; this kernel keeps the Lamport protocol -- push only, the data its own flag,
no fence, no barrier -- and adds the route sum in front of it, so the route sum and the all-reduce are one launch.

Units: unit ``u = t * CB + c`` is the 512 columns ``c * 512 ..`` of token ``t`` (``CB = H / 512``). A wave takes one
unit, a lane one 16-byte pack (8 BF16 columns); workgroup ``w`` (4 waves) takes units ``4 w .. 4 w + 3``.

Partial: ``sum_k routes[loc[t, k]]`` in f32 in slot order (missing routes add +0), rounded to BF16 -- the arithmetic
of ``a8w4_route_sum``.

Destinations (``DEST``, compile time):

* "ar2" (two-shot all-reduce): the unit's owner ``u % WORLD`` receives the peers' partials (reduce-scatter), adds the
  ranks' partials, stores the BF16 sum into its ``out`` and every peer's landing (all-gather); a peer waits for it in
  its landing and copies it into its ``out``.
* "dp" (reduce-scatter, for a data-parallel caller whose rank r holds the tokens ``[r * M_r, (r + 1) * M_r)``,
  ``M = WORLD * M_r``): the owner is the token's home rank ``t // M_r``, which stores the sum into row ``t - r * M_r``
  of its ``out`` [M_r, H]; the other ranks only send. Kernel names end in ``_rs``.

The sum adds the ranks' BF16 partials in f32 in rank order 0..WORLD-1 and rounds to BF16 once (the custom
all-reduce's and ``tp_reduce``'s arithmetic).

Buffers: inbox, symmetric (same layout on every rank) uncached int32 ``[2][WORLD][UNITS][256]``: the 16 bytes
``[b][s][u][4 l .. 4 l + 3]`` hold rank s's partial of lane l's pack of unit u in buffer b; landing ("ar2"), symmetric
uncached int32 ``[2][UNITS][256]``: the owner's sum. Every half-word rests at the sentinel 0xFFFF (a BF16 NaN no
kernel stores: a computed 0xFFFF is sent, and with "ar2" stored, as 0x7FFF, also a NaN), so a pack has arrived when
none of its 8 half-words is the sentinel (a pack may land in pieces: each half-word is checked).

Schedule (one wave, one unit): scalar loads of the unit's buffer ``b = parity[u]``, ``err`` and the address table
(under the partial's loads) -> partial -> sends (system scope, written through) -> wait: poll the expected packs of
the own inbox / landing (all of them per poll, system scope) until none is the sentinel -> put the sentinels back ->
sum, stores -> lane 0 flips ``parity[u]``.

Naming: ``tab`` int64 [2, WORLD] (as int32): the ranks' inbox, then landing addresses; ``parity`` int32 [UNITS]
(zeroed once, this rank's, plain device memory); ``err`` int32 [1]; ``limit`` the wait bound in
``tp_reduce.LIMIT_TICKS`` timer ticks.

Rules:

1. No deadlock: a wave only waits for the waves of its own unit on the peers, which send before they wait ("ar2": the
   owner sends the sum once the peers' partials, sent before their waits, are in); workgroups are dispatched in index
   order on each XCD, so the smallest unit anybody waits for belongs to running workgroups everywhere (decode_tp,
   rule 1).
2. Two buffers: a rank writes a pack of a peer's buffer b again two calls using the unit later. In the call in
   between it received data from that peer (every rank receives from every peer in every call when every rank owns
   a unit: "ar2" ``M * CB >= WORLD``, "dp" ``M >= WORLD``), which the peer sent after
   its kernel of the earlier call -- sentinel resets included -- completed. The parity of a unit flips once per call
   that uses it, on every rank alike (every rank makes the same calls with the same M): no host state, so CUDA graph
   replays advance it too.
3. Bounded waits: a wait gives up after ``limit * LIMIT_TICKS`` ticks of the 100 MHz timer (after its first poll if
   ``err`` is already set), ORs ``ERR_TIMEOUT`` into ``err`` and goes on: wrong output, never a hang. A timeout can
   leave the buffers dirty (a late peer's pack): later calls are wrong until they are refilled with 0xFF and the
   parities zeroed (``MegaMoeTPGfx942.reset``).
4. Bitwise equal outputs ("ar2"): only the owner sums, every rank stores its bits.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, rocdl
from flydsl.expr.utils.arith import _to_raw as _raw

from ..communication_ops_utils import atomic_or_global_at, memrealtime
from ..moe_gemm_2stage.gfx9_fp8.stream_a8w4 import (
    _DROP,
    _address,
    _load,
    _pack64,
    _rsrc,
    _uniform,
)
from .decode_tp import _home, _keep, _pick, _s_load
from .tp_reduce import (
    LIMIT_TICKS,
    SYS,
    _ballot_any,
    _store,
    _sum_bf16,
    _when,
)

# Bit of ``err`` set by a wait of this kernel that gave up (tp_reduce: 1, decode_tp: 2).
ERR_TIMEOUT = 4
SENTINEL = -1  # 0xFFFFFFFF: two sentinel half-words
DESTS = ("ar2", "dp")
_LIMIT_SHIFT = LIMIT_TICKS.bit_length() - 1
_UNIT_BYTES = 64 * 16  # one wave, a 16-byte pack per lane


def inbox_words(UNITS, WORLD):
    """int32 words of the symmetric inbox of ``compile_sum_tp``."""
    return 2 * WORLD * UNITS * (_UNIT_BYTES // 4)


def landing_words(UNITS):
    """int32 words of the symmetric landing of ``compile_sum_tp`` (``DEST`` "ar2")."""
    return 2 * UNITS * (_UNIT_BYTES // 4)


def kernel_name(DEST):
    """``a8w4_route_sum_tp``, ``_rs`` appended for ``DEST`` "dp"."""
    return "a8w4_route_sum_tp" + ("_rs" if DEST == "dp" else "")


def _missing(words):
    """Per lane: whether a half-word of ``words`` (Uint32) is the sentinel."""
    ones = fx.Uint32(0xFFFFFFFF)
    hit = None
    for w in words:
        h = ((w | fx.Uint32(0xFFFF0000)) == ones) | ((w | fx.Uint32(0xFFFF)) == ones)
        hit = h if hit is None else hit | h
    return hit


def _canonical(words):
    """``words`` (Uint32) with every sentinel half-word 0xFFFF turned into 0x7FFF."""
    out = []
    for w in words:
        w = ((w | fx.Uint32(0xFFFF0000)) == fx.Uint32(0xFFFFFFFF)).select(
            w ^ fx.Uint32(0x8000), w
        )
        w = ((w | fx.Uint32(0xFFFF)) == fx.Uint32(0xFFFFFFFF)).select(
            w ^ fx.Uint32(0x80000000), w
        )
        out.append(w)
    return out


def _spin_words(poll, count, limit):
    """Poll until ``poll()`` -> (pending: wave-uniform i1, ``count`` Uint32 words) is not pending or
    ``limit * LIMIT_TICKS`` timer ticks have passed; returns (gave up: wave-uniform i1, the last words).
    """
    # Raw scf.while: loop-carried values in a helper outside the kernel function, which FlyDSL's AST rewriter does
    # not convert (as decode_tp._spin_value).
    loc = ir.Location.unknown()
    i32 = fx.Int32.ir_type
    n = count + 1
    start = memrealtime()
    zero = fx.Int32(0).ir_value()
    loop = scf.WhileOp([i32] * n, [zero] * n, loc=loc)
    loop.regions[0].blocks.append(*([i32] * n), arg_locs=[loc] * n)
    loop.regions[1].blocks.append(*([i32] * n), arg_locs=[loc] * n)
    after = loop.regions[1].blocks[0]
    with ir.InsertionPoint(loop.regions[0].blocks[0]):
        pending, words = poll()
        elapsed = (memrealtime() - start) >> _LIMIT_SHIFT
        go = pending & (elapsed < fx.Int64(limit))
        scf.ConditionOp(
            _raw(go),
            [pending.select(fx.Int32(1), fx.Int32(0)).ir_value()]
            + [fx.Int32(w).ir_value() for w in words],
            loc=loc,
        )
    with ir.InsertionPoint(after):
        rocdl.s_sleep(2)
        scf.YieldOp(list(after.arguments), loc=loc)
    return fx.Int32(loop.results[0]) != 0, [
        fx.Uint32(loop.results[i]) for i in range(1, n)
    ]


@functools.cache
def compile_sum_tp(*, WORLD, H, TOPK, UNITS, DEST):
    """Launcher ``(loc, src, out, tab, parity, err, M, rank, limit, nwg, stream)`` (module docstring).

    ``WORLD`` ranks (2 or 4), hidden size ``H`` (``% 512 == 0``), buffers of ``UNITS`` units (``M * H / 512 <=
    UNITS``). ``src`` = routes BF16 by sorted row, ``loc`` int32 [M, TOPK] (negative: no route). ``out`` BF16 [M, H]
    ("dp": [M / WORLD, H]). Grid: ``nwg = ceil(M * H / 512 / 4)`` workgroups of 256 threads.
    """
    assert WORLD in (2, 4), WORLD
    assert H % 512 == 0, H
    assert 0 < TOPK <= 64, TOPK
    assert DEST in DESTS, DEST
    CB = H // 512
    ROW = 2 * H
    INBOX = inbox_words(UNITS, WORLD) * 4
    LANDING = landing_words(UNITS) * 4
    TABLES = 2 if DEST == "ar2" else 1
    ONES = [fx.Uint32(0xFFFFFFFF)] * 4

    def body(p_loc, p_src, p_out, p_tab, p_parity, p_err, M, rank, limit):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        u = _uniform(fx.Int32(gpu.block_id("x")) * 4 + (tid >> 6))

        def unit():
            t = u // CB
            col2 = (u - t * CB) * 1024 + lane * 16
            pack = u * _UNIT_BYTES + lane * 16
            state = [_s_load(p_parity, u), _s_load(p_err, 0)]
            for q in range(0, 2 * WORLD * TABLES, 4):
                row = _s_load(p_tab, q, count=4)
                state += [fx.Int32(row[i]) for i in range(4)]
            src_base = _address(p_src)
            mine = fx.Int32(
                _load(
                    _rsrc(_address(p_loc), M * (TOPK * 4)),
                    (lane < TOPK).select(t * (TOPK * 4) + lane * 4, fx.Int32(_DROP)),
                    0,
                    1,
                )[0]
            )
            rows = []
            for k in range(TOPK):
                loc = fx.Int32(
                    rocdl.readlane(
                        fx.Int32.ir_type, mine.ir_value(), fx.Int32(k).ir_value()
                    )
                )
                live = loc >= 0
                # One row per descriptor from a 64-bit base (as tp_reduce).
                row = _rsrc(src_base + fx.Int64(live.select(loc, 0)) * ROW, ROW)
                rows.append(_load(row, live.select(col2, fx.Int32(_DROP)), 0, 4))
            # The scalar loads issued under the partial's loads (decode_tp._keep).
            _keep(state)
            partial = _sum_bf16(rows)
            # The partial as sent (and, for bitwise equal outputs, as added by every rank).
            sent = _canonical(partial)
            buf, err = state[0], state[1]
            bases = [
                _pack64(fx.Uint32(state[2 + 2 * i]), fx.Uint32(state[3 + 2 * i]))
                for i in range(WORLD * TABLES)
            ]
            own_inbox = _rsrc(_pick(bases[:WORLD], rank), INBOX)
            wait_limit = (err != fx.Int32(0)).select(fx.Int32(0), limit)

            def slot(s):
                """Byte offset of this lane's pack [buf][s][u] in an inbox."""
                return (buf * WORLD + s) * (UNITS * _UNIT_BYTES) + pack

            def gave_up(late):
                _when(
                    late & (lane == 0),
                    lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT),
                )

            def receive():
                """Wait for the peers' partials in the own inbox, put the sentinels back; the ranks' partials in
                rank order (the own one from registers)."""

                def poll():
                    words, miss = [], None
                    for s in range(WORLD):
                        got = _load(
                            own_inbox,
                            (rank != s).select(slot(s), fx.Int32(_DROP)),
                            0,
                            4,
                            aux=SYS,
                        )
                        words += got
                        m = _missing(got)
                        miss = m if miss is None else miss | m
                    return _ballot_any(miss), words

                late, words = _spin_words(poll, 4 * WORLD, wait_limit)
                for s in range(WORLD):
                    _store(
                        own_inbox,
                        ONES,
                        (rank != s).select(slot(s), fx.Int32(_DROP)),
                        SYS,
                    )
                gave_up(late)
                return [
                    [(rank == s).select(w, g) for w, g in zip(sent, words[4 * s :])]
                    for s in range(WORLD)
                ]

            if DEST == "ar2":
                owner = u & (WORLD - 1)
                landing = [_rsrc(b, LANDING) for b in bases[WORLD:]]
                own_landing = _rsrc(_pick(bases[WORLD:], rank), LANDING)
                at = buf * (UNITS * _UNIT_BYTES) + pack
                out = _rsrc(_address(p_out), M * ROW)

                def as_owner():
                    total = _canonical(_sum_bf16(receive()))
                    _store(out, total, t * ROW + col2)
                    for p in range(WORLD):
                        _store(
                            landing[p],
                            total,
                            (rank != p).select(at, fx.Int32(_DROP)),
                            SYS,
                        )

                def as_peer():
                    _store(
                        _rsrc(_pick(bases[:WORLD], owner), INBOX), sent, slot(rank), SYS
                    )

                    def poll():
                        got = _load(own_landing, at, 0, 4, aux=SYS)
                        return _ballot_any(_missing(got)), got

                    late, got = _spin_words(poll, 4, wait_limit)
                    _store(own_landing, ONES, at, SYS)
                    _store(out, got, t * ROW + col2)
                    gave_up(late)

                _when(owner == rank, as_owner)
                _when(owner != rank, as_peer)
            else:
                owner = _home(t, M, WORLD)
                local = M >> (WORLD.bit_length() - 1)

                def as_owner():
                    total = _sum_bf16(receive())
                    _store(
                        _rsrc(_address(p_out), local * ROW),
                        total,
                        (t - rank * local) * ROW + col2,
                    )

                def as_peer():
                    _store(
                        _rsrc(_pick(bases[:WORLD], owner), INBOX), sent, slot(rank), SYS
                    )

                _when(owner == rank, as_owner)
                _when(owner != rank, as_peer)
            _store(
                _rsrc(_address(p_parity), UNITS * 4),
                [buf ^ fx.Int32(1)],
                (lane == 0).select(u * 4, fx.Int32(_DROP)),
            )

        _when(u < M * CB, unit)

    @flyc.kernel(name=kernel_name(DEST), known_block_size=[256, 1, 1])
    def a8w4_sum_tp(
        p_loc: fx.Pointer,
        p_src: fx.Pointer,
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_parity: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        limit: fx.Int32,
    ):
        body(p_loc, p_src, p_out, p_tab, p_parity, p_err, M, rank, limit)

    @flyc.jit
    def launch(
        p_loc: fx.Pointer,
        p_src: fx.Pointer,
        p_out: fx.Pointer,
        p_tab: fx.Pointer,
        p_parity: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        limit: fx.Int32,
        nwg: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_sum_tp(p_loc, p_src, p_out, p_tab, p_parity, p_err, M, rank, limit).launch(
            grid=(nwg, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch
