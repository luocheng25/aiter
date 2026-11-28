# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The A8W4 MoE's shared FP8 expert kernels (gfx942): the step between its two dense GEMMs and its Down GEMM.

``aiter.ops.flydsl.moe_a8w4`` runs a shared expert in FP8 with per-row scales (DeepSeek-V4.1-Flash, chosen by every
token) as two FP8 GEMMs on the routed path's quantized input and folds the second one's BF16 output into the routed
path's last reduction (``_SharedFold``): the Gate/Up GEMM is aiter's preshuffle GEMM (``moe_a8w4._shared_gemm``), the
Down GEMM ``compile_shared_down`` from enough rows on (else the preshuffle GEMM, ``moe_a8w4._shared_down``; the same
values bit for bit). ``compile_shared_mid`` sits between them: it
reads the Gate/Up GEMM's BF16 [M, 2I], computes SiLU(gate) * up in f32 (with DeepSeek-V4.1's SwiGLU limit), quantizes
each row to FP8 for the Down GEMM (zero-padded to the K of its weight, so the GEMM needs no padding copy) and folds the
shared expert's routing weight into the row scale: the Down GEMM then writes ``weight * shared(x)`` directly. A token
whose shared expert is dropped (ATOM's rule: first routed id -1, or the last id not E) gets an all-zero row and scale 0,
so the Down GEMM writes +0. Optionally it also sets the shared expert's column of ``loc_ids`` (the row of the routes
buffer that ``a8w4_route_sum`` adds for the prefill tier).

Raw operations: the Down kernel addresses global memory through the raw buffer descriptors of ``stream_a8w4``
(``_rsrc`` / ``_load``, a raw store with cache-policy bits here) and runs MFMA on register operands; the mid kernel's
row maximum uses raw DPP moves and its quantization a raw wave vote; each names the stable FlyDSL 0.5.3 API it lacks at
the call (``aiter/ops/flydsl/AGENTS.md``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh
from . import common_a8w4 as a8w4
from .decode_a8w4 import (
    _FAST_DIV_MAX,
    _FAST_DIV_MIN,
    _LOG2E,
    _abs_max,
    _fp8_words,
    _fp8_words_fast,
    _global_ptr,
    _row_scale,
    _wave_max,
)
from .stream_a8w4 import (
    _DROP,
    _address,
    _bf16_pair,
    _f32,
    _lds_barrier,
    _lds_base,
    _lds_load4,
    _lds_store,
    _load,
    _mfma,
    _pack64,
    _rsrc,
    _uniform,
    _uniform_if,
    _zeros,
)

# DPP controls of the max over a row's lanes (shared_fold_mid): quad_perm [1, 0, 3, 2], quad_perm [2, 3, 0, 1],
# row_half_mirror (lane i <-> 7 - i of each 8 lanes), row_mirror (lane i <-> 15 - i of each 16 lanes).
_ROW_DPP = (0xB1, 0x4E, 0x141, 0x140)


def mid_row_lanes(I):
    """Lanes per row of ``compile_shared_mid`` for many rows of ``I`` values: 8 up to 40 pieces of 8 values
    (I <= 320), else 16 (at most 5 pieces per lane for I <= 640; DeepSeek-V4.1's I 320 and 576 then leave no lane / 8
    of 80 slots idle). Few rows take 64 (one row per wave, the shortest latency; ``moe_a8w4``'s
    SHARED_MID_WAVE_ROW_MAX_M)."""
    return 8 if I // 8 <= 40 else 16


def _row_max(value, lanes):
    """Max over the ``lanes`` (8 / 16 / 64) consecutive lanes that hold one row (wave-uniform for 64)."""
    if lanes == 64:
        return _wave_max(value)
    # Raw DPP moves: FlyDSL 0.5.3 has no typed DPP operand (the permutations are free modifiers of the max, where
    # fx.gpu.shuffle_xor issues a separate cross-lane permute per step).
    for ctrl in _ROW_DPP[: 3 if lanes == 8 else 4]:
        bits = value.bitcast(fx.Int32)
        moved = fx.rocdl.update_dpp(
            fx.Int32.ir_type, _raw(bits), _raw(bits), ctrl, 0xF, 0xF, False
        )
        value = fx.maxnumf(value, fx.Int32(moved).bitcast(fx.Float32))
    return value


def _row_fp8_words(values, amax, scale):
    """FP8 bytes of ``values / scale`` (a multiple of 4 values; ``amax`` the row's abs-max and ``scale`` its FP8 scale,
    per lane) exactly as ``quant_a8w4`` (IEEE division, clamp +-240, RNE, -0 -> +0), four per Uint32 word: through
    the reciprocal with two corrections (the correctly rounded quotient, ``decode_a8w4._quant_fast``), or the IEEE
    division where some lane's row abs-max is outside ``_fast_div_ok`` (a wave-uniform branch, taken only then).
    """
    inv = fx.Float32(1.0) / scale
    fast = _fp8_words_fast(values, scale, inv, 2)
    n = len(values) // 4
    fast = [fx.Uint32(fast[i]) for i in range(n)]
    off = (amax >= fx.Float32(_FAST_DIV_MAX)) | (
        (amax < fx.Float32(_FAST_DIV_MIN)) & (amax != 0.0)
    )
    # rocdl.ballot: FlyDSL 0.5.3 has no typed wave vote (fx.gpu offers only the shuffle family); the 64-lane mask
    # makes the fallback branch wave-uniform.
    redo = fx.Int64(rocdl.ballot(T.i64, _raw(off))) != fx.Int64(0)

    def ieee():
        words = _fp8_words(values, scale)
        return [fx.Uint32(words[i]) for i in range(n)]

    return _uniform_if(redo, ieee, fast)


def compile_shared_mid(*, I, KP, COLS, E, SWIGLU_LIMIT=0.0, LOC=False, LANES=0):
    """Launcher ``(gate_up, topk_ids, topk_weight, q, scale, loc, rows, base, stream)``.

    gate_up BF16 [rows, 2 * I] (gate columns [0, I), up columns [I, 2I); ``I % 8 == 0``) -> q FP8 E4M3FNUZ [rows, KP]
    (``KP >= I``, ``KP % 8 == 0``; columns [I, KP) zero) and scale f32 [rows]. Per row: g, u in f32 (with
    ``SWIGLU_LIMIT`` L > 0: ``g = min(g, L)``, ``u = clamp(u, -L, L)``), h = SiLU(g) * u, quantized with the
    arithmetic of ``quant_a8w4`` (``s = amax / 240``, an all-zero row 1, ``q = RNE(h / s)``, -0 stored as +0); the
    stored scale is ``s * w``, w the row's shared-expert weight: ``topk_weight[t, COLS - 1]`` where
    ``topk_ids[t, 0] >= 0`` and ``topk_ids[t, COLS - 1] == E`` (int32 / f32 [rows, COLS], contiguous), else 0 with h
    taken as 0 (an all-zero q row: the Down GEMM's product is +0 whatever its weight). With ``LOC`` int32 ``loc``
    [rows, COLS] gets ``base + t`` in column COLS - 1 of row t where the shared expert is live, -1 elsewhere (``loc``
    and ``base`` are unused without it). Each row is held by ``LANES`` (8 / 16 / 64; 0: ``mid_row_lanes``)
    consecutive lanes: lane r of the row loads the 16-byte pieces (8 values) r, r + LANES, ... of gate and up (the
    lanes of a row read contiguous bytes per load), the row maximum is a DPP max over them, and each piece's 8 FP8
    bytes are one 8-byte store; 256 threads take 1024 / LANES rows. With 8 / 16 lanes the quotients are the correctly
    rounded ones through the reciprocal (``_row_fp8_words``), with 64 the per-value IEEE divisions: the same bytes.
    """
    assert I % 8 == 0 and KP % 8 == 0 and KP >= I, f"unsupported I={I} KP={KP}"
    assert COLS >= 2
    PIECES = I // 8
    G = LANES or mid_row_lanes(I)
    assert G in (8, 16, 64), f"unsupported LANES={LANES}"
    PER_LANE = (PIECES + G - 1) // G
    PAD = (KP - I) // 8  # zero pieces of 8 bytes past the row's I columns
    PAD_PER_LANE = (PAD + G - 1) // G
    WAVES = 4
    ROWS = WAVES * (64 // G)  # rows per workgroup
    HALF = I // 2  # Uint32 words of the gate half of a row

    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_shared_mid", SWIGLU_LIMIT))
    def a8w4_shared_mid(
        p_in: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_loc: fx.Pointer,
        rows: fx.Int32,
        base: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane_r = tid % G
        row = gpu.block_idx.x * ROWS + tid // G
        if row < rows:
            ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
            tws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_weight))
            live = (ids[row * COLS] >= 0) & (ids[row * COLS + (COLS - 1)] == E)
            weight = live.select(tws[row * COLS + (COLS - 1)], fx.Float32(0.0))
            # 2I BF16 = I Uint32 words per row; the up half starts HALF words in.
            src = _global_ptr(p_in, fx.Uint32, 16) + fx.Int64(row) * I
            pieces, values = [], []
            for k in range_constexpr(PER_LANE):
                piece = lane_r + G * k
                if const_expr(G * (k + 1) > PIECES):
                    # Duplicates of the last piece leave the row maximum unchanged and are not stored.
                    piece = (piece < PIECES).select(piece, fx.Int32(PIECES - 1))
                pieces.append(piece)
                gate_words = fx.ptr_load(
                    src + piece * 4, result_type=Vec.make_type(4, fx.Uint32)
                )
                up_words = fx.ptr_load(
                    src + (HALF + piece * 4), result_type=Vec.make_type(4, fx.Uint32)
                )
                for w in range_constexpr(4):
                    gw, uw = fx.Uint32(gate_words[w]), fx.Uint32(up_words[w])
                    halves = (
                        ((gw << 16), (uw << 16)),
                        (gw & fx.Uint32(0xFFFF0000), uw & fx.Uint32(0xFFFF0000)),
                    )
                    for g_bits, u_bits in halves:
                        gate, up = a8w4.swiglu_clamp(
                            g_bits.bitcast(fx.Float32),
                            u_bits.bitcast(fx.Float32),
                            SWIGLU_LIMIT,
                        )
                        # Bare v_exp_f32 / v_rcp_f32 (raw rocdl), as the decode Gate/Up's epilogue: fx.math.exp2 and
                        # the f32 division lower to the denormal-scaled exp and the correctly rounded divide.
                        tmp = rocdl.exp2(T.f32, _raw(gate * (-_LOG2E)))
                        h = (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up
                        values.append(live.select(h, fx.Float32(0.0)))
            amax = _row_max(_abs_max(values), G)
            scale = _row_scale(amax)
            if const_expr(G == 64):
                # One row per wave (few rows, latency-bound): the per-value IEEE division, no wave vote.
                words = _fp8_words(values, scale)
            else:
                words = _row_fp8_words(values, amax, scale)
            dst = _global_ptr(p_q, fx.Uint32, 8) + fx.Int64(row) * (KP // 4)
            for k in range_constexpr(PER_LANE):
                stored = Vec.from_elements([words[2 * k], words[2 * k + 1]], fx.Uint32)
                if const_expr(G * (k + 1) <= PIECES):
                    fx.ptr_store(stored, dst + pieces[k] * 2)
                else:
                    if lane_r + G * k < PIECES:
                        fx.ptr_store(stored, dst + pieces[k] * 2)
            for k in range_constexpr(PAD_PER_LANE):
                if lane_r + G * k < PAD:
                    fx.ptr_store(
                        Vec.from_elements([fx.Uint32(0), fx.Uint32(0)], fx.Uint32),
                        dst + (PIECES + lane_r + G * k) * 2,
                    )
            if lane_r == 0:
                scale_out = fx.make_view(
                    fxh._as_ptr(p_scale, fx.Float32) + fx.Int64(row),
                    fx.make_layout(1, 1),
                )
                scale_out[0] = scale * weight
                if const_expr(LOC):
                    loc_out = fx.make_view(
                        fxh._as_ptr(p_loc, fx.Int32)
                        + fx.Int64(row * COLS + (COLS - 1)),
                        fx.make_layout(1, 1),
                    )
                    loc_out[0] = live.select(base + row, fx.Int32(-1))

    @flyc.jit
    def launch(
        p_in: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_loc: fx.Pointer,
        rows: fx.Int32,
        base: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_shared_mid(p_in, p_ids, p_weight, p_q, p_scale, p_loc, rows, base).launch(
            grid=((rows + ROWS - 1) // ROWS, 1, 1),
            block=(64 * WAVES, 1, 1),
            stream=stream,
        )

    return launch


def _store_dwordx4(rsrc, words, voffset, aux):
    """Raw 16-byte buffer store of four dwords at a per-lane byte offset (``aux``: cache policy bits)."""
    # Raw buffer store: the typed fx.copy store takes its offset from a view layout and has no cache-policy bits; here
    # a per-lane byte offset (rows past the token count land past the descriptor's range, where the hardware range
    # check drops them, no branch) and the policy bits.
    rocdl.raw_ptr_buffer_store(
        Vec.from_elements(list(words), fx.Uint32).bitcast(fx.Int32).ir_value(),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, aux),
    )


def compile_shared_down(*, N, K, KP, PAIRS=4, OUT_AUX=0):
    """Launcher ``(a, a_scale, w, w_scale, out, M, stream)`` of the shared expert's Down GEMM:
    out BF16 [M, N] = (a[:, :K] @ w[:, :K].T) * a_scale * w_scale.T, each value rounded once (RNE) from
    ``(acc * a_scale[m]) * w_scale[n]``: bit for bit the preshuffle GEMM's output on the same operands.

    a FP8 E4M3FNUZ [M, KP] (row stride KP; columns [K, KP) are not read), a_scale f32 [M], w FP8 [N, KP] in
    ``shuffle_weight(layout=(16, 16))`` (16-row blocks of 16 * KP bytes; in each, 64-K blocks of 1 KiB: K group q of
    the 16 rows at 256 q, 16 bytes per row), w_scale f32 [N], out BF16 [M, N] (16-byte aligned). ``K`` (the useful
    reduction) and ``KP`` (the weight's padded K and the row stride of a) are multiples of 64; ``N % (128 * PAIRS)
    == 0``. Rows past M are neither read (0) nor written (dropped by the buffer range checks). ``OUT_AUX``: the cache
    policy bits of the output stores (2: non-temporal).

    Tile: a workgroup (4 waves) takes BM = 64 rows of a (S = 4 sub-blocks of 16) and NG = 128 * PAIRS output
    columns; wave w owns PAIRS pairs of 16-row weight blocks (32 columns each). MFMA 16x16x32 FP8, the weights as A,
    16 rows of a as B. A pair runs two chains t: MFMA row R of chain t is column 8 (R >> 2) + 4 t + (R & 3) of the
    pair (block R >> 3), so the C layout gives lane 16 q + j the eight consecutive columns 8 q .. 8 q + 7 of the pair
    for row j of a sub-block: one 16-byte BF16 store per sub-block and pair. Accumulators: 2 chains x S x 4 f32 per
    lane (two sets: a pair's epilogue overlaps the next pair). The activations of all of K stay in VGPRs (S x K / 64
    x 16 bytes per lane: lane 16 q + j holds bytes 16 q .. 16 q + 15 of each 64-K block of row j, the weight
    layout's K order). LDS: the block's 64 rows of a (row stride K + 16) and the workgroup's NG channel factors, both
    loaded once per workgroup in the prologue (64 (K + 16) + 4 NG bytes: K 320 / NG 2560 31 KiB, two workgroups per
    CU; K 576 47 KiB, one, as its 324 VGPRs allow anyway).
    Schedule: grid (N / NG, ceil(M / 64)), the column groups of a row block adjacent.
    Prologue: global -> LDS of the rows and factors (16-byte loads spread over the 256 lanes), the row scales and
    pair 0's weights from global, a barrier, then the rows from LDS into VGPRs.
    K step (64-K block kb of pair u; no DMA):

        step    MFMA                         VMEM                      LDS
        kb      C_t[s] += W_t(u, kb) A_s(kb) W_0 / W_1(u + 1, kb)                (2 chains x 2 MMAs x 4 sub-blocks)
        start                                                          factors of pair u + 1 (two 16-byte reads)
        0       (after the MFMAs) epilogue of pair u - 1: (C * a_scale) * w_scale -> BF16 -> 4 stores
    Naming: q = lane >> 4 (K group / column octet), j = lane & 15 (row of a sub-block, MFMA row of the A operand),
    s = sub-block, t = chain, u = pair, kb = 64-K block; ``rb_wave`` = the wave's first 16-row weight block.
    Pipeline rules, to keep when editing the kernel body:
    1. Every accumulator chain runs kb = 0 .. K / 64 - 1 and the two MMAs of a block in order, as the preshuffle GEMM
       does (its zero K tail adds +0), and the epilogue multiplies by a_scale first: the output bits equal its output.
    2. Pair u + 1's weights are issued during pair u, block by block, so each load has a whole pair of MFMAs to land;
       VMEM loads and stores share one counter on gfx942, so pair u - 1's stores come after pair u + 1's first loads
       and a wait for weights never waits for the stores just issued.
    3. A sched_barrier closes every pair: the scheduler keeps the pair's loads and MFMAs together.
    """
    assert K % 64 == 0 and KP % 64 == 0 and 0 < K <= KP, f"unsupported K={K} KP={KP}"
    WAVES = 4
    S = 4  # 16-row sub-blocks of the workgroup's 64 rows
    BM = 16 * S
    KB = K // 64
    BLOCK_BYTES = 16 * KP  # one 16-row block of the shuffled weight
    NG = 32 * WAVES * PAIRS
    assert N % NG == 0, f"N={N} must be a multiple of {NG}"
    PP = 1  # pairs of weights in flight
    # LDS: the block's 64 rows of a (row stride K + 16: the 16-byte reads of 16 rows spread over the banks), then the
    # workgroup's NG channel factors.
    A_STRIDE = K + 16
    ACT_LDS = BM * A_STRIDE
    WCH = NG // 4  # 16-byte chunks of the workgroup's channel factors

    @flyc.kernel(known_block_size=[64 * WAVES, 1, 1])
    def a8w4_shared_down(
        p_a: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_w: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid & 63
        wave = _uniform(tid >> 6)
        group = fx.Int32(gpu.block_id("x"))
        row0 = fx.Int32(gpu.block_id("y")) * BM
        rows = M - row0
        # Descriptors start at the block's first row: a row past M has a byte offset past the range (0 / dropped).
        a_rsrc = _rsrc(_address(p_a) + fx.Int64(row0) * KP, fx.Int64(rows) * KP)
        as_rsrc = _rsrc(_address(p_a_scale) + fx.Int64(row0) * 4, fx.Int64(rows) * 4)
        out_rsrc = _rsrc(
            _address(p_out) + fx.Int64(row0) * (2 * N), fx.Int64(rows) * (2 * N)
        )
        w_rsrc = _rsrc(_address(p_w), N * KP)
        quad = lane >> 4
        j = lane & 15
        # First 16-row weight block of the wave (wave-uniform).
        rb_wave = (group * WAVES + wave) * (2 * PAIRS)
        # MFMA row j of chain t is column 8 (j >> 2) + 4 t + (j & 3) of the pair: block j >> 3, row 8 ((j >> 2) & 1)
        # + 4 t + (j & 3) of it; lane 16 q + j reads K group q (bytes 16 q .. 16 q + 15) of each 64-K block.
        w_voff = (
            (j >> 3) * BLOCK_BYTES + quad * 256 + (((j >> 2) & 1) * 8 + (j & 3)) * 16
        )

        def load_w(u, kb):
            """The two chains' 16-byte weight loads of block kb of pair u."""
            base = (rb_wave + 2 * u) * BLOCK_BYTES + kb * 1024
            return [_load(w_rsrc, w_voff + 64 * t, base, 4) for t in range(2)]

        def load_ws(u):
            """Channel factors of columns 8 q .. 8 q + 7 of pair u (from the workgroup's LDS copy)."""
            at = ws_lds + ((wave * PAIRS + u) * 32 + quad * 8) * 4
            words = [_lds_load4(at + 16 * h) for h in range_constexpr(2)]
            return [_f32(x[i]) for x in words for i in range_constexpr(4)]

        # ======== Prologue: activations, row scales, the first pair's weights and factors ========
        # The block's rows of a and the workgroup's channel factors go through LDS, each byte loaded once per
        # workgroup (each wave loading its own copy filled the VMEM queue at the start).
        lds = _lds_base(ACT_LDS + WCH * 16)
        ws_lds = lds + ACT_LDS
        wsg_rsrc = _rsrc(_address(p_w_scale) + fx.Int64(group) * (NG * 4), NG * 4)
        CH = K // 16  # 16-byte chunks of a row
        staged = []
        for i in range_constexpr(BM * CH // 256):
            c = tid + 256 * i
            r = c // CH
            col = c - r * CH
            staged.append(
                (lds + r * A_STRIDE + col * 16, _load(a_rsrc, r * KP + col * 16, 0, 4))
            )
        ws_tail = []
        for i in range_constexpr((WCH + 255) // 256):
            c = tid + 256 * i
            if const_expr(256 * (i + 1) <= WCH):
                staged.append((ws_lds + c * 16, _load(wsg_rsrc, c * 16, 0, 4)))
            else:
                # Lanes past the factors load nothing (offset past the range) and store nothing.
                off = (c < WCH).select(c * 16, fx.Int32(_DROP))
                ws_tail.append((c, _load(wsg_rsrc, off, 0, 4)))
        row_scale = [
            _f32(_load(as_rsrc, (s * 16 + j) * 4, 0, 1)[0]) for s in range_constexpr(S)
        ]
        weights = {u: [load_w(u, kb) for kb in range(KB)] for u in range(PP)}
        for i in range_constexpr(len(staged)):
            _lds_store(staged[i][0], staged[i][1])
        for i in range_constexpr(len(ws_tail)):
            c, words = ws_tail[i]
            if c < WCH:
                _lds_store(ws_lds + c * 16, words)
        _lds_barrier()
        a_rd = lds + j * A_STRIDE + quad * 16
        acts = []
        for s in range_constexpr(S):
            blocks = []
            for kb in range_constexpr(KB):
                x = _lds_load4(a_rd + s * (16 * A_STRIDE) + kb * 64)
                blocks.append((_pack64(x[0], x[1]), _pack64(x[2], x[3])))
            acts.append(blocks)
        factors = {u: load_ws(u) for u in range(PP)}
        out_voff = j * (2 * N) + quad * 16

        def epilogue(acc, ws, u):
            """Scale, BF16 and one 16-byte store per sub-block of pair u."""
            col = (rb_wave + 2 * u) * 32
            for s in range_constexpr(S):
                vals = [
                    (acc[t][s][v] * row_scale[s]) * ws[4 * t + v]
                    for t in range_constexpr(2)
                    for v in range_constexpr(4)
                ]
                words = [
                    _bf16_pair(vals[2 * i], vals[2 * i + 1]) for i in range_constexpr(4)
                ]
                _store_dwordx4(
                    out_rsrc, words, out_voff + s * (16 * 2 * N) + col, OUT_AUX
                )

        # ======== Main loop over the wave's pairs (unrolled) ========
        # The epilogue of pair u - 1 runs inside pair u's MFMA stream (after its first 64-K block), on the previous
        # pair's accumulators: its VALU work and stores issue while the matrix pipe is busy.
        pending = None
        for u in range_constexpr(PAIRS):
            ahead = u + PP < PAIRS
            if const_expr(ahead):
                factors[u + PP] = load_ws(u + PP)
                weights[u + PP] = []
            w = weights.pop(u)
            acc = [[_zeros() for _ in range(S)] for _ in range(2)]
            for kb in range_constexpr(KB):
                if const_expr(ahead):
                    weights[u + PP].append(load_w(u + PP, kb))  # weights of pair u + PP
                ops = [(_pack64(x[0], x[1]), _pack64(x[2], x[3])) for x in w[kb]]
                for c in range_constexpr(2):
                    for t in range_constexpr(2):
                        for s in range_constexpr(S):
                            acc[t][s] = _mfma(ops[t][c], acts[s][kb][c], acc[t][s])
                if const_expr(pending is not None and kb == 0):
                    epilogue(*pending)  # pair u - 1
                    pending = None
            pending = (acc, factors.pop(u), u)
            rocdl.sched_barrier(0)
        # ======== Epilogue of the last pair ========
        epilogue(*pending)

    @flyc.jit
    def launch(
        p_a: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_w: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_out: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_shared_down(p_a, p_a_scale, p_w, p_w_scale, p_out, M).launch(
            grid=(N // NG, (M + BM - 1) // BM, 1),
            block=(64 * WAVES, 1, 1),
            stream=stream,
        )

    return launch
