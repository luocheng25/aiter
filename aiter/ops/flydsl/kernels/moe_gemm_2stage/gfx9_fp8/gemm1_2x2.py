# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 four-wave 256x256 FP8 Gate/Up for prefill.

Ported from pyhip's four-wave FP8 GEMM (tests/flydsl/fp8_gemm_950) and its MoE
gate/up variant. A tile multiplies one 256-row sorted block by 128 gate and 128 up
rows of one expert with 16x16x128 MFMAs. Each wave owns a 64x64 slice of the four
128x128 quadrants (A top/bottom x gate/up), so 256 accumulators per lane stay in
AGPRs. A rows are gathered by sorted token ID and B comes from the (16,16)
preshuffled weights, both through LDS DMA into ping-pong buffers.

The kernel is persistent: each CTA walks a strided tile list, and the last K pair
of a tile prefetches the first two K tiles of the next one, so the epilogue
(per-token x per-channel or per-tensor scales, SiLU/SwiGLU, [tokens, TOPK, N/2]
stores) overlaps the next tile's DMA latency.

With stream_k and a partial last round of at most num_ctas/8 tiles, whole-tile
rounds stop one round early: that round and the partial one are split evenly over
all CTAs in K-pair units. A tile split between slots s and s+1 is finished by s,
which adds the fp32 partial that s+1 publishes first (workspace slot s, ready flag
s, reset after use).
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_cu_num

from . import common as fxh
from .common import _f32_to_bf16

SORT_BLOCK_M = 256
BLOCK = 128  # BLOCK_M = BLOCK_N = BLOCK_K
# A LDS: 8-row groups padded to break bank conflicts (pyhip A8W8 layout).
A_GROUP = 8 * BLOCK + 32
A_GROUP8 = 8 * BLOCK + 16
A_GROUP16 = 2 * A_GROUP
A_LDS_BYTES = (BLOCK // 8) * A_GROUP
B_LDS_BYTES = BLOCK * BLOCK
B_K_STRIDE = 16 * BLOCK  # preshuffled bytes per 128-K step of a 16-row block
PARTIAL_BYTES = 4 * (2 * BLOCK) * (2 * BLOCK)  # fp32 256x256 stream-K partial
_LOG2E = 1.4426950408889634
_WORKSPACE = {}


def gateup_4w_workspace(device, num_ctas=None):
    """Stream-K partials and ready flags. Every launch leaves the flags at zero;
    launches sharing them must not run concurrently."""
    num_ctas = num_ctas or get_cu_num()
    key = (torch.device(device), num_ctas)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = (
            torch.empty(num_ctas * PARTIAL_BYTES // 4, dtype=torch.float32, device=device),
            torch.zeros(num_ctas, dtype=torch.int32, device=device),
        )
    return _WORKSPACE[key]


def gateup_4w_supported(N, K):
    # A partial last N tile reads past the gate rows into valid up rows (or past the
    # expert, which the buffer range zero-fills) and masks those output columns.
    return (N // 2) % 16 == 0 and K % (2 * BLOCK) == 0 and K >= 4 * BLOCK


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


def _rsrc(ptr, num_records):
    view = fx.make_view(ptr, fx.make_layout(num_records, 1))
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(fx.rocdl.make_buffer_tensor(view, False)))


def _lds_address(array):
    return _scalar(fx.Int32(fx.ptrtoint(array.ptr)))


def _wait_groups(groups):
    """Wait until at most `groups` DMA groups are in flight, then sync the CTA."""
    rocdl.wait_asyncmark(groups)
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _schedule_compute(group_id, dsrd_ops, vmem_ops):
    """One next-tile DMA, then one LDS read, per MFMA of a quadrant's 16.

    The DMAs precede the reads in program order: the compiler cannot reorder LDS
    DMA writes and LDS reads it assumes may alias.
    """
    rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(vmem_ops):
        rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    for _ in range_constexpr(dsrd_ops):
        rocdl.sched_group_barrier(rocdl.mask_dsrd, 1, group_id)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    rocdl.sched_group_barrier(rocdl.mask_mfma, 16 - 1 - dsrd_ops - vmem_ops, group_id)


def _schedule_dense(group_id, dsrd_ops, vmem_ops):
    """Like _schedule_compute, but spreads more than 15 DMAs and reads over the MFMAs."""
    ops = [rocdl.mask_vmem_rd] * vmem_ops + [rocdl.mask_dsrd] * dsrd_ops
    rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)
    done = 0
    for slot in range_constexpr(15):
        take = (len(ops) - done + 14 - slot) // (15 - slot)
        for op in ops[done : done + take]:
            rocdl.sched_group_barrier(op, 1, group_id)
        done += take
        rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group_id)


def _swap_pack(words0, words1):
    """permlane16_swap two BF16x2 word pairs into one 16-byte store packet."""
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    lo = rocdl.permlane16_swap(pair_type, words0[0], words1[0], False, False)
    hi = rocdl.permlane16_swap(pair_type, words0[1], words1[1], False, False)
    return fx.Vector.from_elements(
        [
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])),
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])),
            fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1])),
        ],
        fx.Int32,
    )


@functools.cache
def build_moe_gateup_4w(
    *,
    N,
    K,
    TOPK,
    activation="silu",
    swiglu_limit=None,
    act_quant_type="ptpc",
    weight_quant_type="ptpc",
    group_m=4,
    num_ctas=None,
    stream_k=True,
    tail_max=4,
    rows=256,
):
    """FP8 Gate/Up on gfx950; prefill Gate/Up ABI plus the stream-K workspace pair.

    rows=128 runs half tiles: every sort block's top 128 rows on the usual tile
    schedule, preceded by the bottom halves that hold routes (most are empty below
    ~128 routes per expert).
    """
    assert gateup_4w_supported(N, K), f"4-wave Gate/Up needs N/2 % 16 == 0, K % 256 == 0: {N=} {K=}"
    assert act_quant_type in ("ptpc", "per_tensor")
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert activation in ("silu", "swiglu")
    assert rows in (128, 256)
    halves = rows // BLOCK
    num_ctas = num_ctas or get_cu_num()
    assert num_ctas % 8 == 0, "CTA slots are interleaved over eight XCDs"
    inter = N // 2
    n_tiles = (inter + BLOCK - 1) // BLOCK
    n_tail = inter % BLOCK != 0
    num_k_tiles = K // BLOCK
    num_pairs = num_k_tiles // 2
    max_f = max(1, min(tail_max, num_pairs))
    in_group = group_m * n_tiles
    fp8 = fx.Float8E4M3FN
    # Per quadrant: four 16-byte DMAs (one group) per operand, eight ds_read_b128 per
    # fragment. Each phase waits for the operand it reads; five groups stay in flight
    # (three for half tiles, which issue three groups per K tile).
    a_vmem = b_vmem = 4
    a_dsrd = b_dsrd = 8
    in_flight = 5 if halves == 2 else 3

    @fx.struct
    class Storage:
        a_t0: fx.Array[fp8, A_LDS_BYTES, 16]
        a_t1: fx.Array[fp8, A_LDS_BYTES, 16]
        a_b0: fx.Array[fp8, A_LDS_BYTES, 16]
        a_b1: fx.Array[fp8, A_LDS_BYTES, 16]
        b_l0: fx.Array[fp8, B_LDS_BYTES, 16]
        b_l1: fx.Array[fp8, B_LDS_BYTES, 16]
        b_r0: fx.Array[fp8, B_LDS_BYTES, 16]
        b_r1: fx.Array[fp8, B_LDS_BYTES, 16]

    def _activate(gate, up):
        if const_expr(activation == "swiglu"):
            neg_limit = fx.Float32(-swiglu_limit)
            gate = -((-gate).maximumf(neg_limit))
            up = (-((-up).maximumf(neg_limit))).maximumf(neg_limit)
            tmp = rocdl.exp2(T.f32, _raw(gate * (-1.702 * _LOG2E)))
            return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * (up + 1.0)
        tmp = rocdl.exp2(T.f32, _raw(gate * -_LOG2E))
        return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_gateup_4w(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_partials: fx.Pointer,
        p_flags: fx.Pointer,
        tokens: fx.Int32,
    ):
        tid = fx.thread_idx.x
        tid_i = fx.Int32(tid)
        lane = tid_i % 64
        wave = _scalar(tid_i // 64)
        sorted_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))
        expert_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))
        valid_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids))
        n_blocks = (_scalar(valid_ids[0]) + SORT_BLOCK_M - 1) // SORT_BLOCK_M
        total = n_blocks * n_tiles
        # CTA c runs on XCD c % 8; its tiles are slot + i * num_ctas, so the tiles an
        # XCD runs concurrently are consecutive and share A blocks and expert weights.
        cta = fx.Int32(fx.block_idx.x)
        slot = cta % 8 * (num_ctas // 8) + cta // 8
        rounds = total // num_ctas
        rest = total - rounds * num_ctas
        count = (slot < total).select(
            (total - slot + num_ctas - 1) // num_ctas, fx.Int32(0)
        )
        tail, chunk = slot < 0, fx.Int32(0)
        if const_expr(stream_k):
            # A short remainder (rest <= num_ctas / 8) splits the last num_ctas + rest
            # tiles evenly: slot s owns K pairs [g0, g1). Its ranges drift by
            # rest/num_ctas tiles and desynchronize CTAs sharing A/B tiles, so a longer
            # one keeps whole rounds data-parallel and splits each remainder tile into
            # f <= tail_max K ranges, chunk j of tile t on slot t * f + j.
            split = (rounds > 0) & (rest > 0) & (rest * 8 <= num_ctas)
            sk_work = (num_ctas + rest) * num_pairs
            g0 = slot * sk_work // num_ctas
            g1 = (slot + 1) * sk_work // num_ctas
            u0 = g0 // num_pairs
            n_dp = split.select(rounds - 1, count)
            count = n_dp + split.select((g1 - 1) // num_pairs - u0 + 1, fx.Int32(0))
            f = (rest > 0).select(
                fx.Int32(num_ctas) // (rest > 0).select(rest, fx.Int32(1)), fx.Int32(1)
            )
            f = (f < max_f).select(f, fx.Int32(max_f))
            tail = (~split) & (f > 1)
            count = tail.select(rounds + (slot < rest * f).select(fx.Int32(1), fx.Int32(0)), count)
            chunk = slot % f

        zero = fx.Int32(0)

        def coords(logical):
            """Logical tile -> (sorted block, N tile); group_m blocks per N sweep."""
            first_m = logical // in_group * group_m
            rest_m = n_blocks - first_m
            size_m = (rest_m < group_m).select(rest_m, fx.Int32(group_m))
            return first_m + logical % in_group % size_m, logical % in_group // size_m

        # Half tiles: bottom-half candidates j are the CTA's tiles slot + j * num_ctas.
        bottom_slots = (slot < total).select((total - slot + num_ctas - 1) // num_ctas, zero)

        def bottom_live(j):
            ok = j < bottom_slots
            block, _ = coords(ok.select(j * num_ctas + slot, zero))
            head = sorted_ids[block * SORT_BLOCK_M + BLOCK].bitcast(fx.Uint32)
            return ok & ((head & 0xFFFFFF) < fx.Uint32(tokens)) & ((head >> 24) < TOPK)

        n_live_b = zero
        if const_expr(halves == 1):
            # A single carried value comes back unwrapped after the loop.
            for j, live_state in range(0, bottom_slots, 1, init=[zero]):
                live_next = live_state[0] + bottom_live(fx.Int32(j)).select(fx.Int32(1), zero)
                live_count = yield [live_next]
            n_live_b = live_count

        if count + n_live_b > 0:
            lds = fx.SharedAllocator().allocate(Storage).peek()
            a_rsrc = _rsrc(fxh._as_ptr(p_input, fx.Uint8), tokens * K)
            out_bytes = tokens * (TOPK * inter * 2)
            o_rsrc = _rsrc(fxh._as_ptr(p_output, fx.Uint8), out_bytes)
            lg = lane // 16
            wave_m = wave // 2
            wave_n = wave % 2

            def job(i):
                """Job i of this CTA -> (logical tile, first K pair, end K pair)."""
                logical = i * num_ctas + slot
                if const_expr(not stream_k):
                    return logical, zero, fx.Int32(num_pairs)
                in_sk = i >= n_dp
                u = u0 + i - n_dp
                start = g0 - u * num_pairs
                stop = g1 - u * num_pairs
                first = in_sk.select((start > 0).select(start, zero), zero)
                end = in_sk.select((stop < num_pairs).select(stop, fx.Int32(num_pairs)), fx.Int32(num_pairs))
                logical = in_sk.select((rounds - 1) * num_ctas + u, logical)
                in_tail = tail & (i >= rounds)
                first = in_tail.select(chunk * num_pairs // f, first)
                end = in_tail.select((chunk + 1) * num_pairs // f, end)
                return in_tail.select(rounds * num_ctas + slot // f, logical), first, end

            def pick(i, jb):
                """Job i -> (logical tile, first K pair, end K pair, half); live bottom
                halves (candidate jb) come first, then the top-half schedule."""
                logical, first, end = job(i - n_live_b)
                if const_expr(halves == 2):
                    return logical, first, end, None
                is_b = i < n_live_b
                return (
                    is_b.select(jb * num_ctas + slot, logical),
                    is_b.select(zero, first),
                    is_b.select(fx.Int32(num_pairs), end),
                    is_b.select(fx.Int32(1), zero),
                )

            def a_voffsets(block_m, half=None):
                offsets = []
                for h in range_constexpr(halves):
                    base = block_m * SORT_BLOCK_M + h * BLOCK
                    if const_expr(half is not None):
                        base = base + half * BLOCK
                    for c in range_constexpr(4):
                        row = base + wave + (lane // 8) * 16 + c * 4
                        token = sorted_ids[row] & 0xFFFFFF
                        offsets.append(token * K + (lane % 8) * 16)
                return offsets

            def weight_rsrc(block_m):
                expert = _scalar(expert_ids[block_m])
                base = fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(expert) * (N * K)
                return expert, _rsrc(base, N * K)

            # ---- G2S: A gathered by token, B from the preshuffled expert weights ----
            # Preshuffled rows are 16-row blocks of 16 * K bytes; a tile's first row
            # and the K step ride in soffset.
            b_slot = [
                (tid_i // 128 + 2 * c) * (16 * K) + (tid_i % 128) * 16
                for c in range_constexpr(4)
            ]
            a_lds_wave = (wave % 2) * A_GROUP8 + (wave // 2) * A_GROUP16
            a_t_dma = [_lds_address(arr) + a_lds_wave for arr in (lds.a_t0, lds.a_t1)]
            a_b_dma = [_lds_address(arr) + a_lds_wave for arr in (lds.a_b0, lds.a_b1)]
            b_l_dma = [_lds_address(arr) + wave * 1024 for arr in (lds.b_l0, lds.b_l1)]
            b_r_dma = [_lds_address(arr) + wave * 1024 for arr in (lds.b_r0, lds.b_r1)]
            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

            def dma(rsrc, address, voffset, soffset):
                rocdl.raw_ptr_buffer_load_async_lds(
                    rsrc,
                    llvm.inttoptr(lds_ptr_type, address.ir_value()),
                    fx.Int32(16).ir_value(),
                    voffset.ir_value(),
                    soffset.ir_value(),
                    zero.ir_value(),
                )

            def g2s_a(kk, bases, voffsets):
                soffset = kk * BLOCK
                for c in range_constexpr(4):
                    dma(a_rsrc, bases + c * 4 * A_GROUP, voffsets[c], soffset)
                rocdl.asyncmark()

            def g2s_b(rsrc, kk, bases, first_row):
                soffset = first_row * K + kk * B_K_STRIDE
                for c in range_constexpr(4):
                    dma(rsrc, bases + c * 4096, b_slot[c], soffset)
                rocdl.asyncmark()

            # ---- S2R and MMA (weights are MFMA operand A, activations operand B) ----
            mma_atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fp8))
            k_perm = fx.make_layout((32, 4), (1, 32))
            tiled_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)), (None, None, k_perm)
            )
            thr_mma = tiled_mma.thr_slice(tid)
            a_read = fx.make_layout(
                ((2, BLOCK // 16, 8), (32, BLOCK // 32)),
                ((A_GROUP8, A_GROUP16, BLOCK), (1, 32)),
            )
            b_read = fx.make_layout(
                ((16, BLOCK // 16), (16, BLOCK // 16)), ((16, 2048), (1, 256))
            )
            sA_t = [fx.make_view(arr.ptr, a_read) for arr in (lds.a_t0, lds.a_t1)]
            sA_b = [fx.make_view(arr.ptr, a_read) for arr in (lds.a_b0, lds.a_b1)]
            sB_l = [fx.make_view(arr.ptr, b_read) for arr in (lds.b_l0, lds.b_l1)]
            sB_r = [fx.make_view(arr.ptr, b_read) for arr in (lds.b_r0, lds.b_r1)]
            copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
            copy_a = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
            copy_b = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
            src_A_t = [copy_a.partition_S(v) for v in sA_t]
            src_A_b = [copy_a.partition_S(v) for v in sA_b]
            src_B_l = [copy_b.partition_S(v) for v in sB_l]
            src_B_r = [copy_b.partition_S(v) for v in sB_r]
            frag_A_t = thr_mma.make_fragment_B(sA_t[0])
            frag_A_b = thr_mma.make_fragment_B(sA_b[0])
            frag_B_l = thr_mma.make_fragment_A(sB_l[0])
            frag_B_r = thr_mma.make_fragment_A(sB_r[0])
            dst_A_t = copy_a.retile(frag_A_t)
            dst_A_b = copy_a.retile(frag_A_b)
            dst_B_l = copy_b.retile(frag_B_l)
            dst_B_r = copy_b.retile(frag_B_r)
            # Half tiles read the next K tile's A while the current one is in use.
            frag_A = [frag_A_t, thr_mma.make_fragment_B(sA_t[0])]
            dst_A = [dst_A_t, copy_a.retile(frag_A[1])]

            def load(src, dst):
                fx.copy(copy_atom, src, dst, pred=None)

            def gemm(c_frag, b_frag, a_frag):
                fx.gemm(mma_atom, c_frag, b_frag, a_frag, c_frag)

            c_tile = fx.make_rmem_tensor(
                fx.make_ordered_layout((BLOCK, BLOCK), (1, 0)), fx.Float32
            )
            frag_C_tl = thr_mma.make_fragment_C(c_tile)
            frag_C_tr = thr_mma.make_fragment_C(c_tile)
            frag_C_bl = thr_mma.make_fragment_C(c_tile)
            frag_C_br = thr_mma.make_fragment_C(c_tile)
            frags_C = (frag_C_tl, frag_C_tr, frag_C_bl, frag_C_br)[: 2 * halves]

            def run_pair(kk, a_vo, b_rsrc, n_tile):
                """Compute the two K tiles in LDS; prefetch K tiles kk, kk+1 of a tile."""

                def quadrant(group, read, mma, dma, dsrd, vmem):
                    _wait_groups(in_flight)
                    dma()
                    load(*read)
                    gemm(*mma)
                    _schedule_compute(group, dsrd, vmem)
                    rocdl.sched_barrier(0)

                if const_expr(halves == 1):
                    for buf in range_constexpr(2):
                        nxt = 1 - buf
                        k_pf = kk + buf
                        # tl: refill B_l and A_t of this buffer, read B_r.
                        _wait_groups(in_flight)
                        g2s_b(b_rsrc, k_pf, b_l_dma[buf], n_tile * BLOCK)
                        g2s_a(k_pf, a_t_dma[buf], a_vo[0:4])
                        load(src_B_r[buf], dst_B_r)
                        gemm(frag_C_tl, frag_B_l, frag_A[buf])
                        _schedule_dense(2 * buf, b_dsrd, 2 * b_vmem)
                        rocdl.sched_barrier(0)
                        # tr: refill B_r, read the next K tile's B_l and A_t.
                        _wait_groups(in_flight)
                        g2s_b(b_rsrc, k_pf, b_r_dma[buf], n_tile * BLOCK + inter)
                        load(src_B_l[nxt], dst_B_l)
                        load(src_A_t[nxt], dst_A[nxt])
                        gemm(frag_C_tr, frag_B_r, frag_A[buf])
                        _schedule_dense(2 * buf + 1, b_dsrd + a_dsrd, b_vmem)
                        rocdl.sched_barrier(0)
                    return

                for buf in range_constexpr(2):
                    nxt = 1 - buf
                    k_pf = kk + buf
                    # Quadrant order: tl(A_t.B_l), bl(A_b.B_l), tr(A_t.B_r), br(A_b.B_r).
                    quadrant(
                        4 * buf + 0,
                        (src_A_b[buf], dst_A_b),
                        (frag_C_tl, frag_B_l, frag_A_t),
                        lambda: g2s_b(b_rsrc, k_pf, b_l_dma[buf], n_tile * BLOCK),
                        a_dsrd,
                        b_vmem,
                    )
                    quadrant(
                        4 * buf + 1,
                        (src_B_r[buf], dst_B_r),
                        (frag_C_bl, frag_B_l, frag_A_b),
                        lambda: g2s_a(k_pf, a_t_dma[buf], a_vo[0:4]),
                        b_dsrd,
                        a_vmem,
                    )
                    quadrant(
                        4 * buf + 2,
                        (src_B_l[nxt], dst_B_l),
                        (frag_C_tr, frag_B_r, frag_A_t),
                        lambda: g2s_a(k_pf, a_b_dma[buf], a_vo[4:8]),
                        b_dsrd,
                        a_vmem,
                    )
                    quadrant(
                        4 * buf + 3,
                        (src_A_t[nxt], dst_A_t),
                        (frag_C_br, frag_B_r, frag_A_b),
                        lambda: g2s_b(b_rsrc, k_pf, b_r_dma[buf], n_tile * BLOCK + inter),
                        a_dsrd,
                        b_vmem,
                    )

            def epilogue(row_ids, row_scales, ws_gate, ws_up, expert_scale, n_tile, partial=None,
                         enable=None):
                """Dequant + activation; permlane-pack eight columns per 16-byte store.

                partial: (first workspace slot, n, live) adds the fp32 partials of n
                consecutive slots; slots past the dynamic `live` count read as zero.
                enable: dynamic store predicate.
                """
                # Fragment (cr, rr) holds columns cr*32 + wave_n*16 + lg*4 + e and row
                # rr*32 + wave_m*16 + lane%16 of its quadrant.
                for qm in range_constexpr(halves):
                    gate_frag = frags_C[2 * qm]
                    up_frag = frags_C[2 * qm + 1]
                    for rr in range_constexpr(4):
                        encoded = row_ids[qm * 4 + rr]
                        token, slot_id = encoded & 0xFFFFFF, encoded >> 24
                        valid = (token < fx.Uint32(tokens)) & (slot_id < TOPK)
                        if const_expr(enable is not None):
                            valid = valid & enable
                        out_row = fx.Int32(token) * TOPK + fx.Int32(slot_id)
                        a_sc = row_scales[qm * 4 + rr]
                        words = []
                        for cr in range_constexpr(4):
                            gate = gate_frag[None, cr, rr].load()
                            up = up_frag[None, cr, rr].load()
                            if const_expr(partial is not None):
                                base, n_parts, live = partial
                                for k in range_constexpr(n_parts):
                                    ok = None if live is None else live > k
                                    p_gate = load_partial(base + k, 2 * qm, cr, rr, ok)
                                    p_up = load_partial(base + k, 2 * qm + 1, cr, rr, ok)
                                    gate = [gate[e] + p_gate[e] for e in range_constexpr(4)]
                                    up = [up[e] + p_up[e] for e in range_constexpr(4)]
                            values = []
                            for e in range_constexpr(4):
                                if const_expr(weight_quant_type == "ptpc"):
                                    sg, su = ws_gate[cr][e], ws_up[cr][e]
                                else:
                                    sg = su = expert_scale
                                values.append(
                                    _activate(gate[e] * sg * a_sc, up[e] * su * a_sc)
                                )
                            bf16 = _f32_to_bf16(fx.Vector.from_elements(values, fx.Float32))
                            packed = bf16.bitcast(fx.Int32)
                            words.append([packed[0].ir_value(), packed[1].ir_value()])
                        for cp in range_constexpr(0, 4, 2):
                            data = _swap_pack(words[cp], words[cp + 1])
                            col = (
                                n_tile * BLOCK
                                + cp * 32
                                + (lg % 2) * 32
                                + wave_n * 16
                                + (lg // 2) * 8
                            )
                            keep = valid & (col < inter) if const_expr(n_tail) else valid
                            offset = keep.select((out_row * inter + col) * 2, out_bytes)
                            rocdl.raw_ptr_buffer_store(
                                data.ir_value(), o_rsrc, offset.ir_value(), zero.ir_value()
                            )

            def epilogue_operands(block_m, expert, n_tile, half=None):
                """Row IDs and scales of a tile, loaded before its last K pair."""
                row_ids = []
                for qm in range_constexpr(halves):
                    for rr in range_constexpr(4):
                        row = block_m * SORT_BLOCK_M + qm * BLOCK + rr * 32 + wave_m * 16 + lane % 16
                        if const_expr(half is not None):
                            row = row + half * BLOCK
                        row_ids.append(sorted_ids[row].bitcast(fx.Uint32))
                if const_expr(act_quant_type == "ptpc"):
                    as_rsrc = _rsrc(fxh._as_ptr(p_a_scale, fx.Float32), tokens)
                    row_scales = []
                    for encoded in row_ids:
                        token = encoded & 0xFFFFFF
                        offset = (token < fx.Uint32(tokens)).select(
                            fx.Int32(token) * 4, fx.Int32(-1)
                        )
                        row_scales.append(
                            fx.Float32(
                                rocdl.raw_ptr_buffer_load(
                                    fx.Float32.ir_type,
                                    as_rsrc,
                                    offset.ir_value(),
                                    zero.ir_value(),
                                )
                            )
                        )
                else:
                    tensor_scale = fx.make_view(
                        fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
                    )[0]
                    row_scales = [tensor_scale] * (4 * halves)
                ws_gate = ws_up = expert_scale = None
                if const_expr(weight_quant_type == "ptpc"):
                    ws_buf = fx.rocdl.make_buffer_tensor(
                        fx.make_view(
                            fxh._as_ptr(p_w_scale, fx.Float32) + expert * N,
                            fx.make_layout(N, 1),
                        ),
                        False,
                    )
                    ws_packet = fxh.BufferTensor(
                        fx.make_view(fx.get_iter(ws_buf), fx.make_layout(4, 1))
                    )
                    ws_col = n_tile * BLOCK + wave_n * 16 + lg * 4
                    ws_gate = [
                        ws_packet.load(voffset_bytes=(ws_col + cr * 32) * 4, soffset_bytes=zero)
                        for cr in range_constexpr(4)
                    ]
                    ws_up = [
                        ws_packet.load(
                            voffset_bytes=(ws_col + inter + cr * 32) * 4, soffset_bytes=zero
                        )
                        for cr in range_constexpr(4)
                    ]
                else:
                    expert_scale = fx.make_view(
                        fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
                    )[0]
                return row_ids, row_scales, ws_gate, ws_up, expert_scale

            # ---- stream-K partial exchange (16-byte chunks, thread-major per slot) ----
            # SC1 (device-scope) stores and loads keep the partials coherent across
            # XCD L2s without the whole-L2 writeback/invalidate of agent fences.
            sc1 = ir.IntegerAttr.get(fx.Int32.ir_type, 16)
            p_rsrc = _rsrc(fxh._as_ptr(p_partials, fx.Uint8), num_ctas * PARTIAL_BYTES)
            flag_base = fx.recast_iter(fx.Int32, fxh._as_ptr(p_flags))
            global_ptr = ir.Type.parse("!llvm.ptr<1>")

            def flag_ptr(index):
                return llvm.inttoptr(global_ptr, fx.ptrtoint(flag_base + index).ir_value())

            def flag_load(index):
                return fx.Int32(
                    llvm.LoadOp(
                        fx.Int32.ir_type,
                        flag_ptr(index),
                        alignment=4,
                        ordering=llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                    ).result
                )

            def flag_store(index, value):
                llvm.StoreOp(
                    fx.Int32(value).ir_value(),
                    flag_ptr(index),
                    alignment=4,
                    ordering=llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                )

            def partial_offset(index, q, cr, rr):
                return index * PARTIAL_BYTES + ((q * 16 + rr * 4 + cr) * 256) * 16 + tid_i * 16

            def publish_partial(index, head):
                """Store this CTA's partial in slot `index`, then count it on `head`'s flag."""
                for q in range_constexpr(len(frags_C)):
                    for rr in range_constexpr(4):
                        for cr in range_constexpr(4):
                            chunk = frags_C[q][None, cr, rr].load()
                            rocdl.raw_ptr_buffer_store(
                                chunk.bitcast(fx.Int32).ir_value(),
                                p_rsrc,
                                partial_offset(index, q, cr, rr).ir_value(),
                                zero.ir_value(),
                                aux=sc1,
                            )
                rocdl.s_waitcnt(vmcnt=0)
                rocdl.s_barrier()
                if tid_i == 0:
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        flag_ptr(head),
                        fx.Int32(1).ir_value(),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                    )

            def load_partial(index, q, cr, rr, ok=None):
                offset = partial_offset(index, q, cr, rr)
                if const_expr(ok is not None):
                    offset = ok.select(offset, fx.Int32(num_ctas * PARTIAL_BYTES))
                return fx.Vector(
                    rocdl.raw_ptr_buffer_load(
                        ir.VectorType.get([4], fx.Float32.ir_type),
                        p_rsrc,
                        offset.ir_value(),
                        zero.ir_value(),
                        aux=sc1,
                    )
                )

            # ---- prologue: first two K tiles of the first job ----
            a_vo = fx.make_rmem_tensor(4 * halves, fx.Int32)
            # Bottom-half candidate of the current job, carried across iterations.
            jb0 = zero
            if const_expr(halves == 1):
                cur_b = fx.make_rmem_tensor(1, fx.Int32)
                while (n_live_b > 0) & ~bottom_live(jb0):
                    jb0 = jb0 + 1
                cur_b.store(fx.Vector.from_elements([jb0], fx.Int32))
            logical0, first0, _, half0 = pick(zero, jb0)
            block0, n_tile0 = coords(logical0)
            a_vo.store(fx.Vector.from_elements(a_voffsets(block0, half0), fx.Int32))
            _, b_rsrc0 = weight_rsrc(block0)
            vo0 = a_vo.load()
            vo0 = [vo0[i] for i in range_constexpr(4 * halves)]
            for kk in range_constexpr(2):
                k0 = 2 * first0 + kk
                g2s_b(b_rsrc0, k0, b_l_dma[kk], n_tile0 * BLOCK)
                rocdl.sched_barrier(0)
                g2s_a(k0, a_t_dma[kk], vo0[0:4])
                rocdl.sched_barrier(0)
                if const_expr(halves == 2):
                    g2s_a(k0, a_b_dma[kk], vo0[4:8])
                    rocdl.sched_barrier(0)
                g2s_b(b_rsrc0, k0, b_r_dma[kk], n_tile0 * BLOCK + inter)
                rocdl.sched_barrier(0)
            # Land the first K tile's B_l and A_t (2 of 2 * (2 + halves) groups).
            _wait_groups(2 + 2 * halves)
            load(src_B_l[0], dst_B_l)
            load(src_A_t[0], dst_A_t)
            rocdl.sched_barrier(0)

            for it in range(0, count + n_live_b, 1):
                i = fx.Int32(it)
                jb = jb_n = zero
                if const_expr(halves == 1):
                    jb = cur_b.load()[0]
                    jb_n = jb + 1
                    while (i + 1 < n_live_b) & ~bottom_live(jb_n):
                        jb_n = jb_n + 1
                    cur_b.store(fx.Vector.from_elements([jb_n], fx.Int32))
                logical, first, end, half = pick(i, jb)
                block_m, n_tile = coords(logical)
                expert, b_rsrc = weight_rsrc(block_m)
                has_next = i + 1 < count + n_live_b
                logical_n, first_n, _, half_n = pick(
                    has_next.select(i + 1, i), has_next.select(jb_n, jb)
                )
                block_n, n_tile_n = coords(logical_n)
                _, b_rsrc_n = weight_rsrc(block_n)
                vo_next = a_voffsets(block_n, half_n)
                vo = a_vo.load()
                vo = [vo[j] for j in range_constexpr(4 * halves)]
                for frag in frags_C:
                    frag.fill(0)
                acc_init = [frag.load() for frag in frags_C]
                for kidx, states in range(2 * first, 2 * end - 2, 2, init=acc_init):
                    for frag, state in zip(frags_C, states):
                        frag.store(state)
                    run_pair(fx.Int32(kidx) + 2, vo, b_rsrc, n_tile)
                    results = yield [frag.load() for frag in frags_C]
                for frag, result in zip(frags_C, results):
                    frag.store(result)
                operands = epilogue_operands(block_m, expert, n_tile, half)
                # The last pair prefetches the next job; the epilogue overlaps its DMAs.
                run_pair(2 * first_n, vo_next, b_rsrc_n, n_tile_n)
                if const_expr(stream_k):
                    # Split jobs skip the stores; a tail (first > 0) publishes its partial
                    # early, a head is always the CTA's last job and is merged below.
                    epilogue(*operands, n_tile, enable=end - first == num_pairs)
                    if first > 0:
                        publish_partial(slot, tail.select(slot - chunk, slot - 1))
                else:
                    epilogue(*operands, n_tile)
                a_vo.store(fx.Vector.from_elements(vo_next, fx.Int32))
            if const_expr(stream_k):
                head_tile, head_first, head_end = job(count - 1)
                if (head_first == 0) & (head_end < num_pairs):
                    head_block, head_n = coords(head_tile)
                    head_operands = epilogue_operands(
                        head_block, _scalar(expert_ids[head_block]), head_n
                    )
                    live = tail.select(f - 1, fx.Int32(1))
                    pending = fx.Int32(1)
                    while pending != 0:
                        pending = (flag_load(slot) < live).select(fx.Int32(1), zero)
                    epilogue(*head_operands, head_n, partial=(slot + 1, max(max_f - 1, 1), live))
                    rocdl.s_barrier()
                    if tid_i == 0:
                        flag_store(slot, 0)
            rocdl.wait_asyncmark(0)
            rocdl.s_waitcnt(vmcnt=0)

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_partials: fx.Pointer,
        p_flags: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        # Persistent: num_ctas CTAs walk the (256-row block, 128-column tile) list.
        moe_gateup_4w(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            p_partials,
            p_flags,
            M,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "passthrough": [["amdgpu-agpr-alloc", "256,256"]],
            },
        ).launch(grid=(num_ctas, 1, 1), block=(256, 1, 1), stream=stream)

    launch.compile_hints["llvm_options"] = {"amdgpu-mfma-vgpr-form": False}
    return launch
