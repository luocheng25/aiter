# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 decode Gate/Up and Down that read each expert's weights once per group of routes (expert deduplication).

``compile_a8w4_row_quant_plan`` is the decode path's per-row FP8 quantization of the token rows
(``decode.compile_a8w4_row_quant``, BF16 input) that also writes a dedup plan for the routes of its
row's token: wave ``t`` loads all ``M * TOPK`` expert ids and, for each route ``r = t * TOPK + s``, ranks
it among the routes of the same expert (route order) with ballots. ``plan[r] = [count, r, r1, ...]``:
the route of rank ``GROUP * g`` leads a group of ``count`` routes (itself and the next ``count - 1``
routes of the expert), every other route gets ``count = 0``.

``compile_a8w4_gateup_decode_dd`` keeps the grid of ``decode.compile_a8w4_gateup_decode`` (``(I / 16,
M * TOPK)``, workgroup ``(j, route)``) and its block-pair weight loads; a workgroup reads its plan entry
and exits unless it leads. A leader stages the group's FP8 token rows in LDS, reads the expert's
16-channel block once and runs the MFMAs with route ``m`` of the group in A rows ``m`` and ``8 + m``
(see "Block pairs" in ``decode``), so a fused shared expert chosen by every token is read
``ceil(M / GROUP)`` times instead of ``M`` times. Outputs are ``SiLU(gate) * up`` as float32 per route,
bit-identical to the per-route kernel (with the same SwiGLU limit). The Down of this path is the slot-pair
``decode.compile_a8w4_down_decode_pq``.

The leader check depends on scalar loads (plan entry, expert id, kernel arguments); ``_pin`` makes them issue
together before the branch instead of one round trip after another.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..moe_gemm_2stage.gfx9_fp8 import common as fxh
from . import common as a8w4
from .decode import (
    _LOG2E,
    _LUT_BYTES,
    _abs_max,
    _decode_block,
    _fp8_words,
    _global_ptr,
    _lds_load,
    _lds_store,
    _load_dwordx4,
    _lut_at,
    _lut_table_init,
    _mfma,
    _row_scale,
    _tree_sum,
    _wave_index,
    _wave_max,
)
from .stream import _DROP, _address, _load, _rsrc, _store

_NONE = 0x7FFFFFFF  # "no further route" in a plan entry


def _pin(values):
    """Empty asm reading the uniform ``values`` in SGPRs: their scalar loads issue together before this point
    (LLVM would otherwise sink each into the branch that uses it, one memory round trip after another).
    """
    ops = [_raw(v) for v in values]
    # Inline asm: FlyDSL has no op that keeps scalar loads ahead of the branches using them.
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        ops,
        "",
        ",".join(["s"] * len(ops)),
        has_side_effects=True,
    )


def _ballot64(pred):
    return fx.Int64(rocdl.ballot(T.i64, _raw(pred)))


def _halves(x):
    halves = Vec.from_elements([x], fx.Int64).bitcast(fx.Int32)
    return fx.Int32(halves[0]), fx.Int32(halves[1])


def _popcount(lo, hi):
    return fx.Int32(fx.math.ctpop(_raw(lo))) + fx.Int32(fx.math.ctpop(_raw(hi)))


def _lowest_lane(lo, hi):
    """Lowest set lane of a ballot (64 when empty)."""
    tz_lo = fx.Int32(fx.math.cttz(_raw(lo)))
    tz_hi = fx.Int32(fx.math.cttz(_raw(hi)))
    return (lo != 0).select(tz_lo, (hi != 0).select(tz_hi + 32, fx.Int32(64)))


def _low_mask(n):
    """64-bit lane mask of the lanes below ``n`` (uniform, clamped to 0..64)."""
    n = (n < 0).select(fx.Int32(0), (n > 64).select(fx.Int32(64), n))
    ones = (fx.Int64(1) << fx.Int64(n & 63)) - fx.Int64(1)
    return (n >= 64).select(fx.Int64(-1), ones)


def _plan_route(r, lane, routes, ids_rsrc, ids_ptr, plan_rsrc, SEGS, GROUP):
    """Plan entry of route ``r`` (one wave): lane l compares the ids of routes ``l + 64 k`` with route r's
    expert; one ballot per segment of 64 routes gives the rank, the expert's route count and the next members.
    """
    qs = [lane + 64 * k for k in range(SEGS)]
    vals = [
        (q < routes).select(
            fx.Int32(
                _load(ids_rsrc, (q < routes).select(q * 4, fx.Int32(_DROP)), 0, 1)[0]
            ),
            fx.Int32(-1),
        )
        for q in qs
    ]
    expert = fx.Int32(ids_ptr[r])
    balls = [_ballot64(v == expert) for v in vals]
    rank = fx.Int32(0)
    total = fx.Int32(0)
    for k in range(SEGS):
        rank = rank + _popcount(*_halves(balls[k] & _low_mask(r - 64 * k)))
        total = total + _popcount(*_halves(balls[k]))
    words = [fx.Int32(0), r]
    prev = r
    for _ in range(1, GROUP):
        nxt = fx.Int32(_NONE)
        for k in reversed(range(SEGS)):
            low = _lowest_lane(
                *_halves(balls[k] & (_low_mask(prev + 1 - 64 * k) ^ fx.Int64(-1)))
            )
            nxt = (low < 64).select(low + 64 * k, nxt)
        words.append(nxt)
        prev = nxt
    while len(words) < 4:
        words.append(fx.Int32(_NONE))
    left = total - rank
    count = (left < GROUP).select(left, fx.Int32(GROUP))
    words[0] = ((rank % GROUP) == 0).select(count, fx.Int32(0))
    _store(
        plan_rsrc,
        [fx.Uint32(w) for w in words],
        (lane == 0).select(r * 16, fx.Int32(_DROP)),
    )


def compile_a8w4_row_quant_plan(*, COLS, TOPK, MAX_M, GROUP=2):
    """Launcher ``(x, q, scale, topk_ids, plan, pad, rows, stream)``: ``decode.compile_a8w4_row_quant(BF16=True)``
    (x BF16 [rows, COLS] -> q FP8 [rows, COLS], scale f32 [rows], same arithmetic) plus the dedup plan
    int32 [rows * TOPK, 4] of the routes in topk_ids int32 [rows, TOPK] (``rows <= MAX_M``). Four-wave
    workgroups: the first ``rows`` quantize a row each, the next ``rows * ceil(TOPK / 4)`` write the plan
    entries of one token's routes, a wave per route. ``pad`` is an unused pointer that keeps the kernel-argument
    layout, and with it the ISA, of the measured kernel."""
    assert COLS % 8 == 0 and 1 <= GROUP <= 3
    PIECES = COLS // 8
    LOADS = (PIECES + 255) // 256
    SEGS = (MAX_M * TOPK + 63) // 64
    PW = (TOPK + 3) // 4  # plan workgroups per token

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, 4, 16]

    @flyc.kernel
    def a8w4_row_quant_plan(
        p_in: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_plan: fx.Pointer,
        p_pad: fx.Pointer,
        rows: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % 64
        wave = _wave_index(tid)
        block = fx.Int32(gpu.block_idx.x)
        routes = rows * TOPK
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red_base = fx.recast_iter(fx.Uint8, lds.red.ptr)
        if block >= rows:
            k = block - rows
            slot = (k % PW) * 4 + wave
            if slot < TOPK:
                _plan_route(
                    (k // PW) * TOPK + slot,
                    lane,
                    routes,
                    _rsrc(_address(p_ids), routes * 4),
                    fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids)),
                    _rsrc(_address(p_plan), routes * 16),
                    SEGS,
                    GROUP,
                )
        if block < rows:
            row = block
            src = _global_ptr(p_in, fx.Uint32, 16) + fx.Int64(row) * (COLS // 2)
            pieces, values = [], []
            for m in range_constexpr(LOADS):
                piece = tid + 256 * m
                if const_expr(256 * (m + 1) > PIECES):
                    piece = (piece < PIECES).select(piece, fx.Int32(PIECES - 1))
                pieces.append(piece)
                words = fx.ptr_load(
                    src + piece * 4, result_type=Vec.make_type(4, fx.Uint32)
                )
                vals = []
                for w in range_constexpr(4):
                    word = fx.Uint32(words[w])
                    vals.append((word << 16).bitcast(fx.Float32))
                    vals.append((word & fx.Uint32(0xFFFF0000)).bitcast(fx.Float32))
                values.append(vals)
            amax = _wave_max(_abs_max([v for vals in values for v in vals]))
            if lane == 0:
                _lds_store(
                    red_base,
                    wave * 4,
                    Vec.from_elements([amax], fx.Float32),
                    fx.Float32,
                    4,
                )
            gpu.barrier()
            red = _lds_load(red_base, 0, fx.Float32, 4, 16)
            scale = _row_scale(
                fx.maxnumf(fx.maxnumf(red[0], red[1]), fx.maxnumf(red[2], red[3]))
            )
            dst = _global_ptr(p_q, fx.Uint32, 8) + fx.Int64(row) * (COLS // 4)
            for m in range_constexpr(LOADS):
                stored = _fp8_words(values[m], scale)
                if const_expr(256 * (m + 1) <= PIECES):
                    fx.ptr_store(stored, dst + pieces[m] * 2)
                else:
                    if tid + 256 * m < PIECES:
                        fx.ptr_store(stored, dst + pieces[m] * 2)
            if tid == 0:
                scale_out = fx.make_view(
                    fxh._as_ptr(p_scale, fx.Float32) + fx.Int64(row),
                    fx.make_layout(1, 1),
                )
                scale_out[0] = scale

    @flyc.jit
    def launch(
        p_in: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_plan: fx.Pointer,
        p_pad: fx.Pointer,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_row_quant_plan(p_in, p_q, p_scale, p_ids, p_plan, p_pad, rows).launch(
            grid=(rows * (1 + PW), 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def _pick(values, index):
    """``values[index]`` for a per-lane ``index`` into a short list of uniform values."""
    out = values[0]
    for g in range(1, len(values)):
        out = (index == g).select(values[g], out)
    return out


def _gateup_group(
    c,
    tid,
    lane,
    wave,
    j,
    count,
    members,
    expert,
    M,
    p_x,
    p_x_scale,
    p_weight,
    p_w_delta,
    p_w_scale,
    p_out,
    bases,
):
    """Leader workgroup: Gate/Up of channel block ``j`` for the ``count`` routes ``members``."""
    xq_base, part_base, lut_base = bases
    G, K, TOPK, INTER, N, NJ, KB, KQ, KB_PART, PAIRS, XS, X_PIECES, X_LOADS, WAVES = (
        c["GROUP"],
        c["K"],
        c["TOPK"],
        c["I"],
        c["N"],
        c["NJ"],
        c["KB"],
        c["KQ"],
        c["KB_PART"],
        c["PAIRS"],
        c["XS"],
        c["X_PIECES"],
        c["X_LOADS"],
        c["WAVES"],
    )
    # Token rows of the group first (they do not wait for the weights): 16-byte pieces, rows past the
    # group read as zero.
    x_rsrc = _rsrc(_address(p_x), M * K)
    pieces, dsts = [], []
    for u in range(X_LOADS):
        g = tid + 64 * WAVES * u
        m = g // X_PIECES
        piece = g - m * X_PIECES
        ok = (m < count) & (g < G * X_PIECES)
        tok = _pick(members, m) // TOPK
        pieces.append(
            _load(x_rsrc, ok.select(tok * K + piece * 16, fx.Int32(_DROP)), 0, 4)
        )
        dsts.append((g < G * X_PIECES).select(m * XS + piece * 16, fx.Int32(G * XS)))
    rocdl.sched_barrier(0)
    rb = (wave & 1) * NJ + j
    kb0 = (wave >> 1) * KB_PART
    w_buf = a8w4.expert_bytes(p_weight, expert, c["W_BYTES"])
    d_buf = a8w4.expert_bytes(p_w_delta, expert, c["D_BYTES"])
    w_voff = fx.Int32(lane * 16)
    d_voff = fx.Int32(((lane % 32) // 16) * 64 + (lane % 8) * 8)
    d_shift = fx.Uint32((lane // 32) * 8)
    raws, deltas = [], []
    for p in range(PAIRS):
        if p % 2 == 0:
            rocdl.sched_barrier(0)
            deltas.append(
                a8w4.load_dwordx2(
                    d_buf,
                    d_voff,
                    fx.Int32((rb * KQ + kb0 // 4 + p // 2) * a8w4.D_BLOCK_BYTES),
                )
            )
        raws.append(
            _load_dwordx4(
                w_buf, w_voff, fx.Int32((rb * KB + kb0 + 2 * p) * a8w4.W_BLOCK_BYTES)
            )
        )
    rocdl.sched_barrier(0)
    # Epilogue factors: thread t < GROUP * 16 owns channel j * 16 + t % 16 of member t // 16.
    em = tid // 16
    e_ok = em < count
    e_route = _pick(members, em)
    xs_rsrc = _rsrc(_address(p_x_scale), M * 4)
    x_scale = fx.Uint32(
        _load(xs_rsrc, e_ok.select((e_route // TOPK) * 4, fx.Int32(_DROP)), 0, 1)[0]
    ).bitcast(fx.Float32)
    ws_rsrc = _rsrc(_address(p_w_scale) + fx.Int64(expert) * (N * 4), N * 4)
    channel = j * 16 + tid % 16
    ws_gate = fx.Uint32(_load(ws_rsrc, channel * 4, 0, 1)[0]).bitcast(fx.Float32)
    ws_up = fx.Uint32(_load(ws_rsrc, (channel + INTER) * 4, 0, 1)[0]).bitcast(
        fx.Float32
    )
    rocdl.sched_barrier(0)
    for u in range(X_LOADS):
        _lds_store(
            xq_base, dsts[u], Vec.from_elements(pieces[u], fx.Uint32), fx.Uint32, 16
        )
    gpu.barrier()

    # A operand of lane l: member l % 8 (clamped into the staged rows), K group (l % 32) // 8 of block l // 32.
    x_off = ((lane % 8) % G) * XS + (lane // 32) * 64 + ((lane % 32) // 8) * 16
    accs = [fx.full(4, 0.0, fx.Float32), fx.full(4, 0.0, fx.Float32)]
    for p in range(PAIRS):
        xop = _lds_load(xq_base, x_off + (kb0 + 2 * p) * 64, fx.Int64, 2, 16)
        shift = d_shift + fx.Uint32(16 * (p % 2))
        ops = [
            _decode_block(
                Vec.from_elements([raws[p][2 * h], raws[p][2 * h + 1]], fx.Uint32),
                _lut_at(lut_base, deltas[p // 2][h], shift),
            )
            for h in range(2)
        ]
        for cc in range(2):
            for h in range(2):
                accs[h] = _mfma(xop[cc], ops[h][cc], accs[h])

    # Lane l holds rows 4 (l // 16) + i of column l % 16: member 4 ((l // 16) & 1) + i, valid for
    # columns 0..7 in rows 0..7 (lanes below 32) and columns 8..15 in rows 8..15.
    col = lane % 16
    mbase = ((lane // 16) & 1) * 4
    lane_ok = (col < 8) == (lane < 32)
    for h in range(2):
        acc = Vec(accs[h])
        for i in range(4):
            ok = lane_ok & (mbase + i < G)
            dst = ok.select(
                (((wave * 2 + h) * G + mbase + i) * 16 + col) * 4,
                fx.Int32(WAVES * 2 * G * 16 * 4),
            )
            _lds_store(
                part_base, dst, Vec.from_elements([acc[i]], fx.Float32), fx.Float32, 4
            )
    gpu.barrier()
    # Thread t < GROUP * 16: member t // 16, channel row t % 16 = 2 m' + h (columns m' and m' + 8 of accumulator h).
    cr = tid % 16
    h = cr & 1
    cm = cr >> 1
    sums = []
    for w in range(WAVES):
        base = ((w * 2 + h) * G + e_ok.select(em, fx.Int32(0))) * 16
        sums.append(
            _lds_load(part_base, (base + cm) * 4, fx.Float32, 1, 4)[0]
            + _lds_load(part_base, (base + cm + 8) * 4, fx.Float32, 1, 4)[0]
        )
    gate = _tree_sum(sums[0::2]) * ws_gate * x_scale
    up = _tree_sum(sums[1::2]) * ws_up * x_scale
    gate, up = a8w4.swiglu_clamp(gate, up, c["SWIGLU_LIMIT"])
    tmp = rocdl.exp2(T.f32, _raw(gate * (-_LOG2E)))
    value = fx.Float32((gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up)
    out_rsrc = _rsrc(_address(p_out), M * (TOPK * INTER * 4))
    _store(
        out_rsrc,
        [value.bitcast(fx.Uint32)],
        e_ok.select((e_route * INTER + j * 16 + cr) * 4, fx.Int32(_DROP)),
    )


def compile_a8w4_gateup_decode_dd(*, N, K, TOPK, GROUP=2, WAVES=4, SWIGLU_LIMIT=0.0):
    """Launcher ``(x, x_scale, topk_ids, plan, w, w_delta, w_scale, out, M, stream)``.

    Arguments and output of ``compile_a8w4_gateup_decode(PREQUANT=True)`` plus ``plan`` from
    ``compile_a8w4_row_quant_plan`` (same ``GROUP``): x FP8 [M, K] with x_scale f32 [M]; topk_ids int32
    [M, TOPK]; out f32 [M, TOPK, I]. ``WAVES`` (4 or 8) waves split gate / up and ``WAVES / 2`` parts of K.
    ``SWIGLU_LIMIT`` > 0 clamps gate and up before SiLU (``common.swiglu_clamp``).
    """
    INTER = N // 2
    assert (
        INTER % 16 == 0 and K % (128 * WAVES) == 0
    ), f"unsupported Gate/Up decode shape N={N} K={K}"
    assert 1 <= GROUP <= 3 and WAVES in (4, 8)
    KB = K // 64
    KB_PART = KB // (WAVES // 2)
    X_PIECES = K // 16
    c = {
        "GROUP": GROUP,
        "K": K,
        "TOPK": TOPK,
        "I": INTER,
        "N": N,
        "NJ": INTER // 16,
        "KB": KB,
        "KQ": a8w4.delta_dwords(K),
        "KB_PART": KB_PART,
        "PAIRS": KB_PART // 2,
        "XS": K + 16,
        "X_PIECES": X_PIECES,
        "WAVES": WAVES,
        "X_LOADS": (GROUP * X_PIECES + 64 * WAVES - 1) // (64 * WAVES),
        "W_BYTES": a8w4.weight_bytes(N, K),
        "D_BYTES": a8w4.delta_bytes(N, K),
        "SWIGLU_LIMIT": SWIGLU_LIMIT,
    }
    NJ = c["NJ"]

    @fx.struct
    class SharedStorage:
        xq: fx.Array[fx.Uint8, GROUP * (K + 16) + 16, 16]
        part: fx.Array[fx.Float32, WAVES * 2 * GROUP * 16 + 4, 16]
        lut: fx.Array[fx.Uint8, _LUT_BYTES, 16]

    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_gateup_decode_dd", SWIGLU_LIMIT))
    def a8w4_gateup_decode_dd(
        p_x: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_plan: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % 64
        wave = _wave_index(tid)
        j = fx.Int32(gpu.block_idx.x)
        route = fx.Int32(gpu.block_idx.y)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        xq_base = fx.recast_iter(fx.Uint8, lds.xq.ptr)
        part_base = fx.recast_iter(fx.Uint8, lds.part.ptr)
        lut_base = fx.recast_iter(fx.Uint8, lds.lut.ptr)
        _lut_table_init(lut_base, tid)

        # Uniform addresses: scalar loads (the vector memory queue is busy with other waves' weight loads).
        plan_ptr = fx.recast_iter(fx.Int32, fxh._as_ptr(p_plan))
        expert = fx.Int32(fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))[route])
        count = fx.Int32(plan_ptr[route * 4])
        members = []
        for g in range_constexpr(GROUP):
            members.append(fx.Int32(plan_ptr[route * 4 + 1 + g]))
        _pin(
            [count, expert, M]
            + members
            + [
                _address(p)
                for p in (p_x, p_x_scale, p_weight, p_w_delta, p_w_scale, p_out)
            ]
        )
        if count > 0:
            _gateup_group(
                c,
                tid,
                lane,
                wave,
                j,
                count,
                members,
                expert,
                M,
                p_x,
                p_x_scale,
                p_weight,
                p_w_delta,
                p_w_scale,
                p_out,
                (xq_base, part_base, lut_base),
            )

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_x_scale: fx.Pointer,
        p_ids: fx.Pointer,
        p_plan: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_gateup_decode_dd(
            p_x,
            p_x_scale,
            p_ids,
            p_plan,
            p_weight,
            p_w_delta,
            p_w_scale,
            p_out,
            M,
        ).launch(grid=(NJ, M * TOPK, 1), block=(64 * WAVES, 1, 1), stream=stream)

    return launch
