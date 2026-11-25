# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Down decode kernels with the cross-rank sum in their epilogue (``MegaMoeTPGfx942``, M <= 16 rows), gfx942.

``compile_down_decode_tp`` builds the single-slot (``a8w4_down_decode_tp``) or slot-pair (``a8w4_down_decode_pq_tp``)
Down decode kernel of ``moe_a8w4/decode.py`` -- the same body (``down_decode_partials`` / ``down_decode_pq_partials``)
and grid ``(N / 16, M)`` -- whose last step, instead of writing the 16 BF16 channels of its block, exchanges this
rank's f32 partial sums of the block with the other ranks and writes the sum over the ranks. It replaces the Down
kernel **and** the all-reduce after it. Ported from the mega-tp-a8w4 S2 prototype (``a8w4_decode_tp.py``,
proto-b-decode, its plain variant: no lag, no prefetched poll -- the fastest measured) with the body shared with the
single-GPU kernels, waits bounded in time and scalar loads of the exchange state.

Inbox: symmetric (same layout on every rank) uncached int32 ``[2][WORLD][MAX_M * NB][16]`` (``NB = N / 16``): word
``[b][s][blk][c]`` holds rank s's partial of channel c of block blk in buffer b. Every word rests at the sentinel
0xFFFFFFFF, an f32 NaN no kernel stores (a computed 0xFFFFFFFF is sent as 0x7FFFFFFF), so the data is its own flag:
no separate flag, no fence.

Schedule: workgroup ``(j, token)`` owns block ``blk = token * NB + j``. After the waves' partials are in LDS, wave 0
alone goes on: it loads (scalar memory, one batch under the LDS reads of the partial) the block's buffer
``b = parity[blk]``, ``err`` and the ranks' inbox addresses (loaded in the body's head instead, under its main loads,
they raise the slot-pair kernel's VGPRs past an occupancy step on some shapes -- K=192: 62 -> 76 -- for no measured
gain), then lane ``l = 16 s + c`` (rank s, channel c; lanes of ranks >= WORLD idle):

1. send: for a peer s, this rank's partial of channel c into word ``[b][rank][blk][c]`` of rank s's inbox (system
   scope, written through: one 64-byte write per peer);
2. wait: poll the words ``[b][s][blk][c]`` of the own inbox until no peer's lane sees the sentinel;
3. put the sentinels back, add the ranks' partials in rank order 0..WORLD-1 in f32 (the own one from registers),
   lanes 0..15 store the BF16 channels, lane 0 flips ``parity[blk]``.

Naming: ``tab`` int64 [WORLD] the ranks' inbox addresses (as int32); ``parity`` int32 [MAX_M * NB] (zeroed once,
this rank's, plain device memory); ``err`` int32 [1]; ``limit`` the wait bound in ``LIMIT_TICKS`` timer ticks.

Rules:

1. No deadlock: a workgroup only waits for the peers' workgroups of its own block, which send before they wait, and
   workgroups are dispatched in index order on each XCD: the smallest block anybody waits for belongs to a running
   workgroup (it sends) or to one not dispatched although every slot of its XCD holds a smaller block waiting for an
   even smaller one -- a contradiction.
2. Two buffers: a peer writes word ``[b][s][blk][c]`` again two calls using blk later, after it received this rank's
   data of the call in between, sent after this rank's kernel of this call (sentinel resets included) completed. The
   parity of a block flips once per call that uses it, on every rank alike (every rank makes the same calls): no host
   state, so CUDA graph replays advance it too.
3. Bounded waits: a wait gives up after ``limit * LIMIT_TICKS`` ticks of the 100 MHz timer (after its first poll if
   ``err`` is already set), ORs ``ERR_TIMEOUT`` into ``err`` and goes on: wrong output, never a hang. A timeout can
   leave the inbox dirty (a late peer's word): later calls are wrong until the inboxes are refilled with 0xFF and the
   parities zeroed (``MegaMoeTPGfx942.reset``).
4. Bitwise equal outputs: every rank adds the same f32 partials in the same order and rounds once.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.utils.arith import _to_raw as _raw

from ..communication_ops_utils import (
    atomic_or_global_at,
    load_i32_global_system,
    memrealtime,
    store_i32_global_system_monotonic,
)
from ..moe_a8w4 import decode as dec
from ..moe_a8w4.stream import _address, _pack64
from ..moe_gemm_2stage.gfx9_fp8 import common as fxh
from .tp_reduce import LIMIT_TICKS, _ballot_any, _when

# Bit of ``err`` set by a wait of these kernels that gave up (tp_reduce.ERR_TIMEOUT is 1).
ERR_TIMEOUT = 2
SENTINEL = -1  # 0xFFFFFFFF
_LIMIT_SHIFT = LIMIT_TICKS.bit_length() - 1


def inbox_words(N, WORLD, MAX_M):
    """int32 words of the symmetric inbox of ``compile_down_decode_tp``."""
    return 2 * WORLD * MAX_M * N


def _s_load(p, offset, count=1):
    """Scalar (SMEM) load of ``count`` (1 or 4) int32 words at word ``offset`` (wave-uniform) of the tensor at ``p``."""
    ptr = rocdl.make_buffer_ptr(fxh._as_ptr(p, fx.Int32))
    return rocdl.s_buffer_load(ptr, offset, count=count)


def _exchange_state(NB, WORLD, p_tab, p_parity, p_err):
    """Scalar loads of the block's buffer, ``err`` and the ranks' inbox addresses (int32 words)."""
    blk = gpu.block_idx.y * NB + gpu.block_idx.x
    words = [_s_load(p_parity, blk), _s_load(p_err, 0)]
    for q in range(0, 2 * WORLD, 4):
        row = _s_load(p_tab, q, count=4)
        words += [fx.Int32(row[i]) for i in range(4)]
    return words


def _keep(values):
    """Empty side-effecting asm reading the scalar ``values`` (Int32 / Int64): pins the loads producing them before
    this point. LLVM would otherwise sink these loads (no memory effects) to their first use, apart from each other
    and behind the exchange's branches: their latencies would add up in the tail."""
    # Raw llvm.inline_asm: FlyDSL has no opaque use of a value (as kernels_common.hold).
    from flydsl._mlir.dialects import llvm

    operands = [v.ir_value() for v in values]
    llvm.inline_asm(
        None, operands, "", ",".join(["s"] * len(operands)), has_side_effects=True
    )


def _pick(values, idx):
    """``values[idx]`` by selects (``idx`` per lane)."""
    v = values[0]
    for s in range(1, len(values)):
        v = (idx == fx.Int32(s)).select(values[s], v)
    return v


def _spin_value(poll, limit):
    """Poll until ``poll()`` -> (pending: wave-uniform i1, value: i32) is not pending or ``limit * LIMIT_TICKS``
    timer ticks have passed; returns (gave up: wave-uniform i1, the last value)."""
    # Raw scf.while: a loop-carried value in a helper outside the kernel function, which FlyDSL's AST rewriter does
    # not convert (as tp_reduce._spin).
    loc = ir.Location.unknown()
    i32 = fx.Int32.ir_type
    start = memrealtime()
    zero = fx.Int32(0).ir_value()
    loop = scf.WhileOp([i32, i32], [zero, zero], loc=loc)
    loop.regions[0].blocks.append(i32, i32, arg_locs=[loc, loc])
    loop.regions[1].blocks.append(i32, i32, arg_locs=[loc, loc])
    after = loop.regions[1].blocks[0]
    with ir.InsertionPoint(loop.regions[0].blocks[0]):
        pending, value = poll()
        elapsed = (memrealtime() - start) >> _LIMIT_SHIFT
        go = pending & (elapsed < fx.Int64(limit))
        scf.ConditionOp(
            _raw(go),
            [
                pending.select(fx.Int32(1), fx.Int32(0)).ir_value(),
                fx.Int32(value).ir_value(),
            ],
            loc=loc,
        )
    with ir.InsertionPoint(after):
        rocdl.s_sleep(2)
        scf.YieldOp(list(after.arguments), loc=loc)
    return fx.Int32(loop.results[0]) != 0, fx.Int32(loop.results[1])


def _exchange(
    partial,
    lane,
    j,
    token,
    p_out,
    p_tab,
    p_parity,
    p_err,
    rank,
    limit,
    N,
    NB,
    BLKS,
    WORLD,
):
    """Wave 0 after the body (module docstring, steps 1-3); ``partial()`` is this rank's partial of channel
    ``lane % 16``."""
    state = _exchange_state(NB, WORLD, p_tab, p_parity, p_err)
    y = partial()
    # Issued together, under the partial's LDS reads: one wait (else sunk to each first use, behind branches).
    _keep(state)
    buf, err = state[0], state[1]
    bases = [
        _pack64(fx.Uint32(state[2 + 2 * s]), fx.Uint32(state[3 + 2 * s]))
        for s in range(WORLD)
    ]
    src = lane // 16
    c = lane % 16
    blk = token * NB + j
    peer = (src < WORLD) & (src != rank)
    slot_rank = (src < WORLD).select(src, rank)

    def word(base, s):
        """Address of word [buf][s][blk][c] of the inbox at ``base``."""
        return base + fx.Int64((((buf * WORLD + s) * BLKS + blk) * 16 + c) * 4)

    bits = y.bitcast(fx.Int32)
    bits = (bits == fx.Int32(SENTINEL)).select(fx.Int32(0x7FFFFFFF), bits)
    _when(
        peer,
        lambda: store_i32_global_system_monotonic(
            word(_pick(bases, slot_rank), rank), bits
        ),
    )

    own = word(_pick(bases, rank), slot_rank)

    def poll():
        seen = fx.Int32(load_i32_global_system(own))
        return _ballot_any(peer & (seen == fx.Int32(SENTINEL))), seen

    late, seen = _spin_value(poll, (err != fx.Int32(0)).select(fx.Int32(0), limit))
    _when(peer, lambda: store_i32_global_system_monotonic(own, fx.Int32(SENTINEL)))

    total = None
    for s in range(WORLD):
        v = fx.Int32(gpu.shuffle_idx(seen, fx.Int32(16 * s) + c, 64)).bitcast(
            fx.Float32
        )
        v = (rank == fx.Int32(s)).select(y, v)
        total = v if total is None else total + v

    def store_out():
        out = fx.make_view(
            fxh._as_ptr(p_out, fx.BFloat16) + fx.Int64(token) * N, fx.make_layout(N, 1)
        )
        out[j * 16 + c] = fxh._f32_to_bf16(total)

    def flip():
        fx.recast_iter(fx.Int32, fxh._as_ptr(p_parity))[blk] = buf ^ fx.Int32(1)

    _when(lane < 16, store_out)
    _when(lane == 0, flip)
    _when(late & (lane == 0), lambda: atomic_or_global_at(_address(p_err), ERR_TIMEOUT))


@functools.cache
def compile_down_decode_tp(*, N, K, TOPK, WAVES, WORLD, MAX_M, PAIRED):
    """Launcher ``(mid, mid_scale, topk_ids, topk_weight, w, w_delta, w_scale, out, M, tab, parity, err, rank, limit,
    stream)`` (module docstring): the arguments of ``decode.compile_a8w4_down_decode`` (``PAIRED``:
    ``compile_a8w4_down_decode_pq``) with ``N``, ``K``, ``TOPK``, ``WAVES``, then the exchange state of ``WORLD``
    (2 or 4) ranks for batches of up to ``MAX_M`` rows."""
    assert WORLD in (2, 4), WORLD  # lanes 16 s + c of one wave cover the ranks
    assert N % 16 == 0 and K % 64 == 0, f"unsupported Down decode shape N={N} K={K}"
    NB = N // 16
    BLKS = MAX_M * NB
    if PAIRED:
        assert K // 16 <= 64 * WAVES
        Storage = dec.down_decode_pq_storage(K, TOPK, WAVES)
        partials, channel_sum = dec.down_decode_pq_partials, dec.down_decode_pq_sum
    else:
        Storage = dec.down_decode_storage(K, TOPK, WAVES)
        partials, channel_sum = dec.down_decode_partials, dec.down_decode_sum

    @flyc.kernel(name="a8w4_down_decode_pq_tp" if PAIRED else "a8w4_down_decode_tp")
    def a8w4_down_decode_tp(
        p_mid: fx.Pointer,
        p_mid_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_topk_weight: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        p_tab: fx.Pointer,
        p_parity: fx.Pointer,
        p_err: fx.Pointer,
        rank: fx.Int32,
        limit: fx.Int32,
    ):
        if const_expr(PAIRED):
            tid, j, token, part_base = partials(
                N,
                K,
                TOPK,
                WAVES,
                Storage,
                p_mid,
                p_mid_scale,
                p_ids,
                p_topk_weight,
                p_weight,
                p_w_delta,
                p_w_scale,
            )
        else:
            tid, j, token, part_base = partials(
                N,
                K,
                TOPK,
                WAVES,
                Storage,
                p_mid,
                p_ids,
                p_topk_weight,
                p_weight,
                p_w_delta,
                p_w_scale,
            )
        gpu.barrier()
        if tid < 64:
            _exchange(
                lambda: channel_sum(WAVES, part_base, tid % 16),
                tid,
                j,
                token,
                p_out,
                p_tab,
                p_parity,
                p_err,
                rank,
                limit,
                N,
                NB,
                BLKS,
                WORLD,
            )

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
        p_tab: fx.Pointer,
        p_parity: fx.Pointer,
        p_err: fx.Pointer,
        rank: fx.Int32,
        limit: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_down_decode_tp(
            p_mid,
            p_mid_scale,
            p_ids,
            p_topk_weight,
            p_weight,
            p_w_delta,
            p_w_scale,
            p_out,
            M,
            p_tab,
            p_parity,
            p_err,
            rank,
            limit,
        ).launch(grid=(NB, M, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch
