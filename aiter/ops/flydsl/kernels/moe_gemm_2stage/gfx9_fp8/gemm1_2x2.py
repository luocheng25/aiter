# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 four-wave 256x256 FP8 Gate/Up for MoE prefill.

Ported from pyhip's four-wave FP8 GEMM (tests/flydsl/fp8_gemm_950) and its MoE
gate/up variant.

Tile: one 256-row sorted block of activations A (rows gathered by token) times
128 gate and 128 up rows of one expert's (16, 16)-preshuffled weights B. A splits
by rows into A_t / A_b and B into B_l (gate) / B_r (up); each pair makes one
128x128 quadrant of C. Each wave owns a 64x64 slice of all four quadrants, 256
AGPR accumulators per lane. The 16x16x128 MFMAs take the weights as operand A.
Half tiles (rows=128) drop A_b. The epilogue scales C by the A and W scales and
stores SiLU(gate) * up (or SwiGLU) as BF16 [tokens, TOPK, N/2].

Schedule: persistent. CTA c runs on XCD c % 8 and owns the XCD-interleaved slot s
with tiles s, s + C, ..., so the tiles an XCD runs concurrently are consecutive
and share A blocks and weights in L2. Half tiles first run the CTA's bottom halves
that hold routes (most are empty below ~128 routes per expert), then the top
halves on the usual schedule.

Stream-K: a last round of at most C/8 tiles and the round before it split evenly
over all CTAs in K-pair units; a longer remainder keeps whole rounds and splits
each remainder tile into up to tail_max K ranges. The head of a split tile (its
first K range, always its CTA's last job) waits for the other ranges' fp32
partials, merges them in its epilogue and resets the ready flag.

K step: LDS holds two K tiles of every operand (buffer = K tile % 2). A K step is
four regions of one quadrant's 16 MFMAs. Each region also issues a DMA group (4 x
16-byte LDS DMAs per thread), which refills a buffer just consumed with the K tile
two steps ahead, and an LDS read (8 x ds_read_b128) for a later region. A job's
last K pair prefetches the next job's first two K tiles; the job's epilogue
overlaps their latency.

    region  MFMA              LDS -> VGPR   DMA -> LDS
    0       C_tl += B_l A_t   A_b(k)        B_l(k+2)
    1       C_bl += B_l A_b   B_r(k)        A_t(k+2)
    2       C_tr += B_r A_t   B_l(k+1)      A_b(k+2)
    3       C_br += B_r A_b   A_t(k+1)      B_r(k+2)

Half tiles: two regions per K step, A_t double-buffered in VGPRs.

    0       C_tl += B_l A_t(k)  B_r(k)              B_l(k+2), A_t(k+2)
    1       C_tr += B_r A_t(k)  B_l(k+1), A_t(k+1)  B_r(k+2)

Naming: M = sorted rows (routes), N = output columns (weight rows), K = the
reduction. An m / n / k in a name gives its direction (m_block, n_tile, k_pairs,
wave_m); an n_ prefix always means N, never "number of".

    A_t, A_b      top / bottom 128 rows of A (mh = 0 / 1)
    B_l, B_r      gate / up 128 columns of B; C.tl .. C.br their quadrants
    a_t, b_l ..   lowercase: an operand's VGPR fragment (a_t1: the next K tile's
                  A_t, half tiles)
    k, buf        K tile k lives in LDS buffer buf = k % 2; next_buf holds k + 1,
                  and a region's DMA loads k2 = k + 2
    kp0, kp1      a job's K pairs [kp0, kp1); stream-K: w0 / w1 bound a CTA's
                  share of all K pairs, a tail tile splits into `ranges` K ranges
    ni, mi        fragment (ni, mi) of a quadrant: columns ni*32 + ..., rows
                  mi*32 + ... (global_io)
    i, j          job i; DMA j of a four-DMA group
    *_g2s, *_s2r  data movement, g = global, s = LDS, r = VGPRs: load_g2s (LDS DMA
                  in a region), fill_g2s (alone), read_s2r (LDS read)
    DMA, LR       in comments: LDS DMA, LDS read; X(k) is K tile k of operand X
    tile0, k0 ..  suffix 0: job 0's, in the prologue

Pipeline rules, to keep when editing the kernel body:

 1. A region issues its DMA group(s), then its LDS reads, then its MFMAs;
    region() spreads them one per MFMA, DMA first: the compiler keeps LDS reads
    behind LDS DMA writes that may alias them.
 2. DMA groups go out in one order per K tile (B_l, A_t, A_b, B_r; half tiles
    B_l, A_t, B_r), an asyncmark each. A region waits until at most `in_flight`
    groups are pending, the number issued after the group it reads (5, or 3 for
    half tiles; derived at in_flight). Moving a DMA or a read changes the count.
 3. LDS write after read: a DMA refills a buffer no earlier than the region after
    that buffer's last read; the lgkmcnt(0) and s_barrier opening that region
    retire every wave's reads.
 4. VGPR read after write: a fragment read in region r feeds MFMAs from region
    r + 1 on, and a read may overwrite a fragment only after its last MFMA (half
    tiles keep A_t(k+1) in a_t1 while region 1 still multiplies A_t(k)).
 5. Prefetch distance two: K step k loads K tile k + 2 into the buffer k leaves. A
    job's last K pair (its last k_pair) loads the next job's K tiles 0, 1 instead,
    and its last reads take that job's tile 0, so the next job starts with its
    operands ready.
 6. The epilogue's VMEM loads go before the last K pair, whose MFMAs cover them;
    its stores go after it.
 7. After the last job (whose "next job" is itself), stream-K heads merge, then
    wait_asyncmark(0) and vmcnt(0) drain the DMAs.
"""

import contextlib
import functools
from typing import Any, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_cu_num

from . import common as fxh
from .common import (
    Bundle,
    _f32_to_bf16,
    is_route,
    route_token,
    scalar,
    split_route,
    swap_pack,
)

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
DMA_OPS = 4  # 16-byte LDS DMAs per thread in one operand's DMA group
READ_OPS = 8  # ds_read_b128 per operand fragment read
_LOG2E = 1.4426950408889634
_WORKSPACE = {}


class _Job(NamedTuple):
    """Logical tile and its K pairs [kp0, kp1)."""

    logical: Any
    kp0: Any
    kp1: Any


class _Tile(NamedTuple):
    """A job's sorted M block, N tile and K pairs [kp0, kp1); half is 1 for a
    bottom half-tile job."""

    m_block: Any
    n_tile: Any
    kp0: Any
    kp1: Any
    half: Any


class _Source(NamedTuple):
    """Where a tile's K tiles come from: A row offsets, weight resource, N tile."""

    a: list
    b: Any
    n_tile: Any


class _EpilogueIn(NamedTuple):
    """A tile's epilogue inputs, loaded before its last K pair: sorted IDs of the
    lane's rows, their A scales, and the gate / up weight scales (ptpc) or the
    expert's weight scale (per tensor)."""

    row_ids: list
    row_scales: list
    ws_gate: Any
    ws_up: Any
    expert_scale: Any


def gateup_4w_workspace(device, num_ctas=None):
    """Stream-K partials and ready flags. Every launch leaves the flags at zero;
    launches sharing them must not run concurrently."""
    num_ctas = num_ctas or get_cu_num()
    key = (torch.device(device), num_ctas)
    if key not in _WORKSPACE:
        _WORKSPACE[key] = (
            torch.empty(
                num_ctas * PARTIAL_BYTES // 4, dtype=torch.float32, device=device
            ),
            torch.zeros(num_ctas, dtype=torch.int32, device=device),
        )
    return _WORKSPACE[key]


def gateup_4w_supported(N, K):
    # A partial last N tile reads past the gate rows into valid up rows (or past the
    # expert, which the buffer range zero-fills) and masks those output columns.
    return (N // 2) % 16 == 0 and K % (2 * BLOCK) == 0 and K >= 4 * BLOCK


def _rsrc(ptr, num_records):
    view = fx.make_view(ptr, fx.make_layout(num_records, 1))
    return fx.rocdl.get_buffer_rsrc(
        fx.get_iter(fx.rocdl.make_buffer_tensor(view, False))
    )


def _lds_address(array):
    return scalar(fx.Int32(fx.ptrtoint(array.ptr)))


def _regs(rmem, count):
    values = rmem.load()
    return [values[i] for i in range(count)]


def _wait_groups(groups):
    """Wait until at most `groups` DMA groups are in flight, then sync the CTA."""
    rocdl.wait_asyncmark(groups)
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _interleave(group, vmem, dsrd):
    """Spread a region's `vmem` DMAs, then its `dsrd` LDS reads, over its 16 MFMAs."""
    ops = [rocdl.mask_vmem_rd] * vmem + [rocdl.mask_dsrd] * dsrd
    rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
    if len(ops) < 16:
        for op in ops:
            rocdl.sched_group_barrier(op, 1, group)
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
        rocdl.sched_group_barrier(rocdl.mask_mfma, 15 - len(ops), group)
    else:
        done = 0
        for slot in range(15):
            take = (len(ops) - done + 14 - slot) // (15 - slot)
            for op in ops[done : done + take]:
                rocdl.sched_group_barrier(op, 1, group)
            done += take
            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, group)
    rocdl.sched_barrier(0)


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

    rows=128 runs half tiles (module docstring).
    """
    assert gateup_4w_supported(
        N, K
    ), f"4-wave Gate/Up needs N/2 % 16 == 0, K % 256 == 0: {N=} {K=}"
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
    k_tiles = K // BLOCK
    k_pairs = k_tiles // 2
    max_ranges = max(1, min(tail_max, k_pairs))
    in_group = group_m * n_tiles
    fp8 = fx.Float8E4M3FN
    # DMA groups per K tile, issued B_l, A_t, (A_b,) B_r. Region 0 of K step k reads
    # group 3 of K tile k, issued before the rest of K tile k and all of K tile
    # k + 1: 2 * groups - 3 groups. Each region issues as many groups as its reads
    # advance, so every region waits until that many remain.
    groups = 2 + halves
    in_flight = 2 * groups - 3

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

    def activate(gate, up):
        if activation == "swiglu":
            neg_limit = fx.Float32(-swiglu_limit)
            gate = -((-gate).maximumf(neg_limit))
            up = (-((-up).maximumf(neg_limit))).maximumf(neg_limit)
            tmp = rocdl.exp2(T.f32, _raw(gate * (-1.702 * _LOG2E)))
            return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * (up + 1.0)
        tmp = rocdl.exp2(T.f32, _raw(gate * -_LOG2E))
        return (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up

    def kernel_context(p_sorted_ids, p_sorted_expert_ids, tokens):
        """Thread coordinates and the routing views every stage reads."""
        tid = fx.thread_idx.x  # for the MMA / copy thread slices
        tid_i = fx.Int32(tid)  # for arithmetic
        return Bundle(
            tid=tid,
            tid_i=tid_i,
            lane=tid_i % 64,
            wave=scalar(tid_i // 64),
            sorted_ids=fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids)),
            expert_ids=fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids)),
            tokens=tokens,
            zero=fx.Int32(0),
        )

    def tile_schedule(ctx, p_num_valid_ids):
        """This CTA's jobs, in order (logical tiles; coords maps them to M, N):

        1. half tiles only: the bottom halves that hold routes among the CTA's tiles
           slot + j * num_ctas, over all K pairs (bottom_jobs);
        2. tiles slot + j * num_ctas over all K pairs, where stream-K
           (stream_k_plan) replaces the last round(s) by K-pair ranges:
           - even split: after dp_jobs tiles, the tiles that the CTA's share
             [w0, w1) of the last num_ctas + rest tiles' K pairs touches;
           - tail: after `rounds` tiles, K range k_range of tile
             rounds * num_ctas + slot // ranges.
        """
        sorted_ids, tokens, zero = ctx.sorted_ids, ctx.tokens, ctx.zero
        valid_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids))
        m_blocks = (scalar(valid_ids[0]) + SORT_BLOCK_M - 1) // SORT_BLOCK_M
        total = m_blocks * n_tiles
        # CTA c runs on XCD c % 8; its tiles are slot + i * num_ctas, so the tiles an
        # XCD runs concurrently are consecutive and share A blocks and expert weights.
        cta = fx.Int32(fx.block_idx.x)
        slot = cta % 8 * (num_ctas // 8) + cta // 8
        rounds = total // num_ctas
        rest = total - rounds * num_ctas
        count = (slot < total).select(
            (total - slot + num_ctas - 1) // num_ctas, fx.Int32(0)
        )

        def stream_k_plan(count):
            """Stream-K jobs after `count` data-parallel ones. A short remainder
            (rest <= num_ctas / 8) splits the last num_ctas + rest tiles evenly: slot
            s owns K pairs [w0, w1) of their concatenated work. Its ranges drift by
            rest/num_ctas tiles and desynchronize CTAs sharing A/B tiles, so a longer
            one keeps whole rounds data-parallel and splits each remainder tile into
            `ranges` <= tail_max K ranges, K range j of tile t on slot t * ranges + j.
            """
            split = (rounds > 0) & (rest > 0) & (rest * 8 <= num_ctas)
            sk_work = (num_ctas + rest) * k_pairs
            w0 = slot * sk_work // num_ctas
            w1 = (slot + 1) * sk_work // num_ctas
            sk_tile0 = w0 // k_pairs
            dp_jobs = split.select(rounds - 1, count)
            count = dp_jobs + split.select(
                (w1 - 1) // k_pairs - sk_tile0 + 1, fx.Int32(0)
            )
            ranges = (rest > 0).select(
                fx.Int32(num_ctas) // (rest > 0).select(rest, fx.Int32(1)), fx.Int32(1)
            )
            ranges = (ranges < max_ranges).select(ranges, fx.Int32(max_ranges))
            tail = (~split) & (ranges > 1)
            count = tail.select(
                rounds + (slot < rest * ranges).select(fx.Int32(1), fx.Int32(0)), count
            )
            return Bundle(
                count=count,
                dp_jobs=dp_jobs,
                w0=w0,
                w1=w1,
                sk_tile0=sk_tile0,
                ranges=ranges,
                tail=tail,
                k_range=slot % ranges,
            )

        plan = None
        if stream_k:
            plan = stream_k_plan(count)
            count = plan.count

        def coords(logical):
            """Logical tile -> (M block, N tile); group_m M blocks per N sweep."""
            first_m = logical // in_group * group_m
            rest_m = m_blocks - first_m
            size_m = (rest_m < group_m).select(rest_m, fx.Int32(group_m))
            return first_m + logical % in_group % size_m, logical % in_group // size_m

        # Half tiles: bottom-half candidates j are the CTA's tiles slot + j * num_ctas.
        bottom_slots = (slot < total).select(
            (total - slot + num_ctas - 1) // num_ctas, zero
        )

        def bottom_live(j):
            ok = j < bottom_slots
            m_block, _ = coords(ok.select(j * num_ctas + slot, zero))
            head = sorted_ids[m_block * SORT_BLOCK_M + BLOCK].bitcast(fx.Uint32)
            return ok & is_route(head, tokens, TOPK)

        def job(i):
            """Top-half job i."""
            logical = i * num_ctas + slot
            if not stream_k:
                return _Job(logical, zero, fx.Int32(k_pairs))
            in_sk = i >= plan.dp_jobs
            sk_tile = plan.sk_tile0 + i - plan.dp_jobs
            start = plan.w0 - sk_tile * k_pairs
            stop = plan.w1 - sk_tile * k_pairs
            kp0 = in_sk.select((start > 0).select(start, zero), zero)
            stop = (stop < k_pairs).select(stop, fx.Int32(k_pairs))
            kp1 = in_sk.select(stop, fx.Int32(k_pairs))
            logical = in_sk.select((rounds - 1) * num_ctas + sk_tile, logical)
            k_range, ranges = plan.k_range, plan.ranges
            in_tail = plan.tail & (i >= rounds)
            kp0 = in_tail.select(k_range * k_pairs // ranges, kp0)
            kp1 = in_tail.select((k_range + 1) * k_pairs // ranges, kp1)
            logical = in_tail.select(rounds * num_ctas + slot // ranges, logical)
            return _Job(logical, kp0, kp1)

        def tile(i, bottom):
            """Job i; the first bottom_jobs jobs are live bottom halves, `bottom` the
            current one's candidate."""
            logical, kp0, kp1 = job(i - bottom_jobs)
            half = None
            if halves == 1:
                is_bottom = i < bottom_jobs
                logical = is_bottom.select(bottom * num_ctas + slot, logical)
                kp0 = is_bottom.select(zero, kp0)
                kp1 = is_bottom.select(fx.Int32(k_pairs), kp1)
                half = is_bottom.select(fx.Int32(1), zero)
            m_block, n_tile = coords(logical)
            return _Tile(m_block, n_tile, kp0, kp1, half)

        def tile_after(i, bottom, next_bottom):
            """Job i + 1 (bottom candidate next_bottom), else job i again."""
            has_next = i + 1 < jobs
            return tile(has_next.select(i + 1, i), has_next.select(next_bottom, bottom))

        def head_slot():
            """Slot (and flag) of the head this tail K range's partial completes."""
            return plan.tail.select(slot - plan.k_range, slot - 1)

        def head_parts():
            """Number of partials this CTA's head merges."""
            return plan.tail.select(plan.ranges - 1, fx.Int32(1))

        @flyc.jit
        def live_bottoms():
            """Half tiles: how many bottom-half candidates hold routes."""
            for j, state in range(0, bottom_slots, 1, init=[zero]):
                live = bottom_live(fx.Int32(j)).select(fx.Int32(1), zero)
                live_count = yield [state[0] + live]
            return live_count

        # Half tiles run this CTA's bottom halves that hold routes first.
        bottom_jobs = live_bottoms() if halves == 1 else zero
        jobs = count + bottom_jobs
        return Bundle(
            slot=slot,
            count=count,
            bottom_jobs=bottom_jobs,
            jobs=jobs,
            bottom_live=bottom_live,
            coords=coords,
            job=job,
            tile=tile,
            tile_after=tile_after,
            head_slot=head_slot,
            head_parts=head_parts,
        )

    def global_io(ctx, p_input, p_output):
        """Buffer resources of the activations and the output, and the lane's place in
        a C quadrant: fragment (ni, mi) holds columns ni*32 + wave_n*16 + lg*4 + e and
        row mi*32 + wave_m*16 + lane%16."""
        tokens = ctx.tokens
        a_rsrc = _rsrc(fxh._as_ptr(p_input, fx.Uint8), tokens * K)
        out_bytes = tokens * (TOPK * inter * 2)
        o_rsrc = _rsrc(fxh._as_ptr(p_output, fx.Uint8), out_bytes)
        return Bundle(
            a_rsrc=a_rsrc,
            out_bytes=out_bytes,
            o_rsrc=o_rsrc,
            lg=ctx.lane // 16,
            wave_m=ctx.wave // 2,
            wave_n=ctx.wave % 2,
        )

    def mma_operands(ctx, lds, a_rsrc, p_weight):
        """A_t, A_b, B_l, B_r and the accumulators C.

        An operand has two LDS buffers. load_g2s(src, buf, k) DMAs K tile k of a
        source from global memory into buffer buf inside a region, which interleaves
        it; fill_g2s does the same alone, for the prologue. read_s2r(buf) reads buffer
        buf into the operand's VGPR fragment and returns it. A rows are gathered by
        token through per-lane voffsets. B's preshuffled rows are 16-row blocks of
        16 * K bytes; a tile's first row and the K step ride in soffset.
        """
        tid, tid_i, lane, wave, zero = ctx.tid, ctx.tid_i, ctx.lane, ctx.wave, ctx.zero
        b_voffsets = [
            (tid_i // 128 + 2 * j) * (16 * K) + (tid_i % 128) * 16 for j in range(4)
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

        def dma_a(bases, k, voffsets):
            soffset = k * BLOCK
            for j in range(4):
                dma(a_rsrc, bases + j * 4 * A_GROUP, voffsets[j], soffset)
            rocdl.asyncmark()

        def dma_b(bases, k, rsrc, n_first):
            soffset = n_first * K + k * B_K_STRIDE
            for j in range(4):
                dma(rsrc, bases + j * 4096, b_voffsets[j], soffset)
            rocdl.asyncmark()

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
        # Half tiles read the next K tile's A_t into a second fragment.
        frag_A_t1 = thr_mma.make_fragment_B(sA_t[0])
        dst_A_t1 = copy_a.retile(frag_A_t1)
        if halves == 2:
            frag_A_t1, dst_A_t1 = frag_A_t, dst_A_t
        c_tile = fx.make_rmem_tensor(
            fx.make_ordered_layout((BLOCK, BLOCK), (1, 0)), fx.Float32
        )
        C = Bundle(
            tl=thr_mma.make_fragment_C(c_tile),
            tr=thr_mma.make_fragment_C(c_tile),
            bl=thr_mma.make_fragment_C(c_tile),
            br=thr_mma.make_fragment_C(c_tile),
        )

        def operand(srcs, frags, dsts, load_g2s):
            def fill_g2s(src, buf, k):
                load_g2s(src, buf, k)
                rocdl.sched_barrier(0)

            def read_s2r(buf):
                fx.copy(copy_atom, srcs[buf], dsts[buf], pred=None)
                return frags[buf]

            return Bundle(fill_g2s=fill_g2s, load_g2s=load_g2s, read_s2r=read_s2r)

        A_t = operand(
            src_A_t,
            [frag_A_t, frag_A_t1],
            [dst_A_t, dst_A_t1],
            lambda s, buf, k: dma_a(a_t_dma[buf], k, s.a[0:4]),
        )
        A_b = operand(
            src_A_b,
            [frag_A_b] * 2,
            [dst_A_b] * 2,
            lambda s, buf, k: dma_a(a_b_dma[buf], k, s.a[4:8]),
        )
        B_l = operand(
            src_B_l,
            [frag_B_l] * 2,
            [dst_B_l] * 2,
            lambda s, buf, k: dma_b(b_l_dma[buf], k, s.b, s.n_tile * BLOCK),
        )
        B_r = operand(
            src_B_r,
            [frag_B_r] * 2,
            [dst_B_r] * 2,
            lambda s, buf, k: dma_b(b_r_dma[buf], k, s.b, s.n_tile * BLOCK + inter),
        )

        def mma(acc, b, a):
            """acc += b a; the weights are the MFMA's operand A."""
            fx.gemm(mma_atom, acc, b, a, acc)

        def a_offsets(tile):
            """Byte offsets of this lane's A rows: four per half, gathered by token."""
            offsets = []
            for mh in range(halves):
                base = tile.m_block * SORT_BLOCK_M + mh * BLOCK
                if tile.half is not None:
                    base = base + tile.half * BLOCK
                for j in range(4):
                    row = base + wave + (lane // 8) * 16 + j * 4
                    token = route_token(ctx.sorted_ids[row])
                    offsets.append(token * K + (lane % 8) * 16)
            return offsets

        def weights(m_block):
            """(expert, buffer resource of its N x K weights) of a sorted M block."""
            expert = scalar(ctx.expert_ids[m_block])
            base = fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(expert) * (N * K)
            return expert, _rsrc(base, N * K)

        def source(tile, a_reg=None):
            """(expert, DMA source) of a tile; its A offsets are gathered now or come
            from the register `a_reg`."""
            expert, b_rsrc = weights(tile.m_block)
            a = a_offsets(tile) if a_reg is None else _regs(a_reg, 4 * halves)
            return expert, _Source(a, b_rsrc, tile.n_tile)

        def zero_acc():
            """Clear the accumulators; returns their values to start a K loop."""
            for frag in acc:
                frag.fill(0)
            return acc_values()

        def acc_values():
            return [frag.load() for frag in acc]

        def set_acc(values):
            for frag, value in zip(acc, values):
                frag.store(value)

        acc = (C.tl, C.tr, C.bl, C.br)[: 2 * halves]
        return Bundle(
            A_t=A_t,
            A_b=A_b,
            B_l=B_l,
            B_r=B_r,
            C=C,
            acc=acc,
            zero_acc=zero_acc,
            acc_values=acc_values,
            set_acc=set_acc,
            mma=mma,
            a_offsets=a_offsets,
            source=source,
        )

    def stream_k_exchange(ctx, p_partials, p_flags, acc):
        """fp32 partials of split tiles, 16-byte chunks thread-major per workspace slot,
        and a ready flag per slot. SC1 (device-scope) stores and loads keep the partials
        coherent across XCD L2s without the whole-L2 writeback/invalidate of agent
        fences."""
        tid_i, zero = ctx.tid_i, ctx.zero
        sc1 = ir.IntegerAttr.get(fx.Int32.ir_type, 16)
        p_rsrc = _rsrc(fxh._as_ptr(p_partials, fx.Uint8), num_ctas * PARTIAL_BYTES)
        flag_base = fx.recast_iter(fx.Int32, fxh._as_ptr(p_flags))
        global_ptr = ir.Type.parse("!llvm.ptr<1>")

        def flag_ptr(index):
            return llvm.inttoptr(global_ptr, fx.ptrtoint(flag_base + index).ir_value())

        def flag(index):
            return fx.Int32(
                llvm.LoadOp(
                    fx.Int32.ir_type,
                    flag_ptr(index),
                    alignment=4,
                    ordering=llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                ).result
            )

        def waiting(index, parts):
            """1 until `parts` partials have arrived at slot `index`, then 0."""
            return (flag(index) < parts).select(fx.Int32(1), zero)

        def reset(index):
            llvm.StoreOp(
                fx.Int32(0).ir_value(),
                flag_ptr(index),
                alignment=4,
                ordering=llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            )

        def arrive(index):
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                flag_ptr(index),
                fx.Int32(1).ir_value(),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            )

        def offset(index, quad, ni, mi):
            chunk = (quad * 16 + mi * 4 + ni) * 256
            return index * PARTIAL_BYTES + chunk * 16 + tid_i * 16

        def publish(index):
            """Store the accumulators in workspace slot `index` and wait for them."""
            for quad in range(len(acc)):
                for mi in range(4):
                    for ni in range(4):
                        chunk = acc[quad][None, ni, mi].load()
                        rocdl.raw_ptr_buffer_store(
                            chunk.bitcast(fx.Int32).ir_value(),
                            p_rsrc,
                            offset(index, quad, ni, mi).ir_value(),
                            zero.ir_value(),
                            aux=sc1,
                        )
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()

        def load(index, quad, ni, mi, ok=None):
            chunk_offset = offset(index, quad, ni, mi)
            if ok is not None:
                chunk_offset = ok.select(
                    chunk_offset, fx.Int32(num_ctas * PARTIAL_BYTES)
                )
            return fx.Vector(
                rocdl.raw_ptr_buffer_load(
                    ir.VectorType.get([4], fx.Float32.ir_type),
                    p_rsrc,
                    chunk_offset.ir_value(),
                    zero.ir_value(),
                    aux=sc1,
                )
            )

        return Bundle(
            waiting=waiting, reset=reset, arrive=arrive, publish=publish, load=load
        )

    def epilogue(ctx, io, sched, sk, acc, p_a_scale, p_w_scale):
        """Dequant and activation of the accumulators, permlane-packed into 16-byte
        BF16 stores of [tokens, TOPK, N/2]."""
        lane, tokens, zero = ctx.lane, ctx.tokens, ctx.zero
        out_bytes, o_rsrc = io.out_bytes, io.o_rsrc
        lg, wave_m, wave_n = io.lg, io.wave_m, io.wave_n
        load_partial = sk.load

        def prepare(block_m, expert, n_tile, half=None):
            """Row IDs and scales of a tile, loaded before its last K pair."""
            row_ids = []
            for mh in range(halves):
                for mi in range(4):
                    row = block_m * SORT_BLOCK_M + mh * BLOCK + mi * 32
                    row = row + wave_m * 16 + lane % 16
                    if half is not None:
                        row = row + half * BLOCK
                    row_ids.append(ctx.sorted_ids[row].bitcast(fx.Uint32))
            if act_quant_type == "ptpc":
                as_rsrc = _rsrc(fxh._as_ptr(p_a_scale, fx.Float32), tokens)
                row_scales = []
                for encoded in row_ids:
                    token = route_token(encoded)
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
            if weight_quant_type == "ptpc":
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
                    ws_packet.load(
                        voffset_bytes=(ws_col + ni * 32) * 4, soffset_bytes=zero
                    )
                    for ni in range(4)
                ]
                ws_up = [
                    ws_packet.load(
                        voffset_bytes=(ws_col + inter + ni * 32) * 4, soffset_bytes=zero
                    )
                    for ni in range(4)
                ]
            else:
                expert_scale = fx.make_view(
                    fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
                )[0]
            return _EpilogueIn(row_ids, row_scales, ws_gate, ws_up, expert_scale)

        def store(epi_in, n_tile, partial=None, enable=None):
            """partial: (first workspace slot, count, live) adds the fp32 partials of
            count consecutive slots; slots past the dynamic `live` count read as zero.
            enable: dynamic store predicate."""
            row_ids, row_scales, ws_gate, ws_up, expert_scale = epi_in
            # Fragment (ni, mi) holds columns ni*32 + wave_n*16 + lg*4 + e and row
            # mi*32 + wave_m*16 + lane%16 of its quadrant.
            for mh in range(halves):
                gate_frag = acc[2 * mh]
                up_frag = acc[2 * mh + 1]
                for mi in range(4):
                    encoded = row_ids[mh * 4 + mi]
                    token, topk_slot, valid = split_route(encoded, tokens, TOPK)
                    if enable is not None:
                        valid = valid & enable
                    out_row = fx.Int32(token) * TOPK + fx.Int32(topk_slot)
                    a_sc = row_scales[mh * 4 + mi]
                    words = []
                    for ni in range(4):
                        gate = gate_frag[None, ni, mi].load()
                        up = up_frag[None, ni, mi].load()
                        if partial is not None:
                            base, parts, live = partial
                            for j in range(parts):
                                ok = None if live is None else live > j
                                p_gate = load_partial(base + j, 2 * mh, ni, mi, ok)
                                p_up = load_partial(base + j, 2 * mh + 1, ni, mi, ok)
                                gate = [gate[e] + p_gate[e] for e in range(4)]
                                up = [up[e] + p_up[e] for e in range(4)]
                        values = []
                        for e in range(4):
                            if weight_quant_type == "ptpc":
                                sg, su = ws_gate[ni][e], ws_up[ni][e]
                            else:
                                sg = su = expert_scale
                            values.append(
                                activate(gate[e] * sg * a_sc, up[e] * su * a_sc)
                            )
                        bf16 = _f32_to_bf16(fx.Vector.from_elements(values, fx.Float32))
                        packed = bf16.bitcast(fx.Int32)
                        words.append([packed[0].ir_value(), packed[1].ir_value()])
                    for ni0 in range(0, 4, 2):
                        data = swap_pack(words[ni0], words[ni0 + 1])
                        col = (
                            n_tile * BLOCK
                            + ni0 * 32
                            + (lg % 2) * 32
                            + wave_n * 16
                            + (lg // 2) * 8
                        )
                        keep = valid & (col < inter) if n_tail else valid
                        offset = keep.select((out_row * inter + col) * 2, out_bytes)
                        rocdl.raw_ptr_buffer_store(
                            data.ir_value(), o_rsrc, offset.ir_value(), zero.ir_value()
                        )

        @flyc.jit
        def finish(epi_in, tile):
            """A job's stores. Stream-K: split jobs store nothing, a tail K range
            (kp0 > 0) publishes its partial, and the head merges after the jobs."""
            if const_expr(stream_k):
                whole = tile.kp1 - tile.kp0 == k_pairs
                store(epi_in, tile.n_tile, enable=whole)
                if tile.kp0 > 0:
                    head = sched.head_slot()
                    sk.publish(sched.slot)
                    if ctx.tid_i == 0:
                        sk.arrive(head)
            else:
                store(epi_in, tile.n_tile)

        @flyc.jit
        def merge_head():
            """Stream-K: the head of a split tile (the CTA's last job) waits for the
            other K ranges' partials, adds them in its stores and resets the flag."""
            if const_expr(stream_k):
                head_job = sched.job(sched.count - 1)
                if (head_job.kp0 == 0) & (head_job.kp1 < k_pairs):
                    head_m_block, head_n_tile = sched.coords(head_job.logical)
                    head_expert = scalar(ctx.expert_ids[head_m_block])
                    head_in = prepare(head_m_block, head_expert, head_n_tile)
                    parts = sched.head_parts()
                    pending = fx.Int32(1)
                    while pending != 0:
                        pending = sk.waiting(sched.slot, parts)
                    others = (sched.slot + 1, max(max_ranges - 1, 1), parts)
                    store(head_in, head_n_tile, partial=others)
                    rocdl.s_barrier()
                    if ctx.tid_i == 0:
                        sk.reset(sched.slot)

        return Bundle(prepare=prepare, finish=finish, merge_head=merge_head)

    def job_cursor(ctx, sched, operands):
        """This CTA's jobs in order. The current job's A offsets, and for half tiles its
        bottom-half candidate, ride in registers across the job loop."""
        zero, bottom_jobs = ctx.zero, sched.bottom_jobs
        a_rows = fx.make_rmem_tensor(4 * halves, fx.Int32)
        cur_bottom = fx.make_rmem_tensor(1, fx.Int32) if halves == 1 else None

        @flyc.jit
        def first():
            """Job 0: (tile, DMA source); half tiles start at the first live bottom
            half."""
            bottom = zero
            if const_expr(halves == 1):
                while (bottom_jobs > 0) & ~sched.bottom_live(bottom):
                    bottom = bottom + 1
                cur_bottom.store(fx.Vector.from_elements([bottom], fx.Int32))
            tile = sched.tile(zero, bottom)
            a_rows.store(fx.Vector.from_elements(operands.a_offsets(tile), fx.Int32))
            _, src = operands.source(tile, a_rows)
            return tile, src

        @flyc.jit
        def at(i):
            """Job i and job i + 1 (job i again after the last): (tile, expert, DMA
            source, next tile, next DMA source)."""
            bottom = next_bottom = zero
            if const_expr(halves == 1):
                bottom = cur_bottom.load()[0]
                next_bottom = bottom + 1
                while (i + 1 < bottom_jobs) & ~sched.bottom_live(next_bottom):
                    next_bottom = next_bottom + 1
                cur_bottom.store(fx.Vector.from_elements([next_bottom], fx.Int32))
            tile = sched.tile(i, bottom)
            expert, src = operands.source(tile, a_rows)
            nxt = sched.tile_after(i, bottom, next_bottom)
            _, nxt_src = operands.source(nxt)
            return tile, expert, src, nxt, nxt_src

        def advance(nxt_src):
            """Hand the next job's A offsets to the next iteration."""
            a_rows.store(fx.Vector.from_elements(nxt_src.a, fx.Int32))

        return Bundle(first=first, at=at, advance=advance)

    def region(buf, r, dma=1, read=1):
        """Scope of region r of the K step on LDS buffer buf: wait until at most
        `in_flight` DMA groups are pending; on exit, interleave the body's `dma` DMA
        groups and `read` fragment reads with its 16 MFMAs. The body issues them in
        that order, then the MFMAs."""
        _wait_groups(in_flight)
        scope = contextlib.ExitStack()
        group = 2 * halves * buf + r
        scope.callback(_interleave, group, dma * DMA_OPS, read * READ_OPS)
        return scope

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
        ctx = kernel_context(p_sorted_ids, p_sorted_expert_ids, tokens)
        sched = tile_schedule(ctx, p_num_valid_ids)
        if sched.jobs > 0:
            lds = fx.SharedAllocator().allocate(Storage).peek()
            io = global_io(ctx, p_input, p_output)
            operands = mma_operands(ctx, lds, io.a_rsrc, p_weight)
            sk = stream_k_exchange(ctx, p_partials, p_flags, operands.acc)
            epi = epilogue(ctx, io, sched, sk, operands.acc, p_a_scale, p_w_scale)
            cursor = job_cursor(ctx, sched, operands)
            A_t, A_b, B_l, B_r = operands.A_t, operands.A_b, operands.B_l, operands.B_r
            C, mma = operands.C, operands.mma

            # ======== Prologue: job 0's K tiles 0, 1 -> LDS buffers 0, 1 ========
            # Per K tile one DMA group (4 x 16 B per thread) each of B_l, A_t, A_b and
            # B_r (half tiles: no A_b); then LR the two operands region 0 consumes.
            tile0, src0 = cursor.first()
            for buf0 in range_constexpr(2):
                k0 = 2 * tile0.kp0 + buf0
                B_l.fill_g2s(src0, buf0, k0)  # DMA B_l(k0)
                A_t.fill_g2s(src0, buf0, k0)  # DMA A_t(k0)
                if const_expr(halves == 2):
                    A_b.fill_g2s(src0, buf0, k0)  # DMA A_b(k0)
                B_r.fill_g2s(src0, buf0, k0)  # DMA B_r(k0)
            _wait_groups(2 * groups - 2)  # B_l(0), A_t(0) landed
            b_l = B_l.read_s2r(0)  # LR B_l(0)
            a_t = A_t.read_s2r(0)  # LR A_t(0)
            rocdl.sched_barrier(0)

            def k_pair(src, k_load, b_l, a_t):
                """K steps k = k_load - 2 + buf on LDS buffers buf = 0, 1. Step k reads
                buf (next_buf holds K tile k + 1) and loads K tile k2 = k + 2 from src
                into buf. Each region: DMA, LR (8 x ds_read_b128), 16 MFMAs. b_l, a_t:
                the fragments region 0 multiplies, in and out."""
                for buf in range_constexpr(2):
                    k2, next_buf = k_load + buf, 1 - buf
                    if const_expr(halves == 2):
                        with region(buf, 0):  # C_tl += B_l(k) A_t(k)
                            B_l.load_g2s(src, buf, k2)  # DMA B_l(k+2)
                            a_b = A_b.read_s2r(buf)  # LR A_b(k)
                            mma(C.tl, b_l, a_t)
                        with region(buf, 1):  # C_bl += B_l(k) A_b(k)
                            A_t.load_g2s(src, buf, k2)  # DMA A_t(k+2)
                            b_r = B_r.read_s2r(buf)  # LR B_r(k)
                            mma(C.bl, b_l, a_b)
                        with region(buf, 2):  # C_tr += B_r(k) A_t(k)
                            A_b.load_g2s(src, buf, k2)  # DMA A_b(k+2)
                            b_l = B_l.read_s2r(next_buf)  # LR B_l(k+1)
                            mma(C.tr, b_r, a_t)
                        with region(buf, 3):  # C_br += B_r(k) A_b(k)
                            B_r.load_g2s(src, buf, k2)  # DMA B_r(k+2)
                            a_t = A_t.read_s2r(next_buf)  # LR A_t(k+1)
                            mma(C.br, b_r, a_b)
                    else:  # half tiles: A_t(k+1) lands in A_t's second VGPR set
                        with region(buf, 0, dma=2):  # C_tl += B_l(k) A_t(k)
                            B_l.load_g2s(src, buf, k2)  # DMA B_l(k+2)
                            A_t.load_g2s(src, buf, k2)  # DMA A_t(k+2)
                            b_r = B_r.read_s2r(buf)  # LR B_r(k)
                            mma(C.tl, b_l, a_t)
                        with region(buf, 1, read=2):  # C_tr += B_r(k) A_t(k)
                            B_r.load_g2s(src, buf, k2)  # DMA B_r(k+2)
                            b_l = B_l.read_s2r(next_buf)  # LR B_l(k+1)
                            a_t1 = A_t.read_s2r(next_buf)  # LR A_t(k+1)
                            mma(C.tr, b_r, a_t)
                        a_t = a_t1
                return b_l, a_t

            for i in range(0, sched.jobs, 1):
                tile, expert, src, nxt, nxt_src = cursor.at(fx.Int32(i))

                # ======== Main loop: every K pair of the job but the last ========
                # C lives in AGPRs; the loop carries its values (set_acc rebinds them).
                acc0 = operands.zero_acc()
                for kidx, acc in range(2 * tile.kp0, 2 * tile.kp1 - 2, 2, init=acc0):
                    operands.set_acc(acc)
                    b_l, a_t = k_pair(src, fx.Int32(kidx) + 2, b_l, a_t)
                    acc = yield operands.acc_values()
                operands.set_acc(acc)

                # ======== Last K pair: its K tiles k + 2 are the next job's K tiles 0,
                # 1 (so the second step's LRs (k+1) read its tile 0). The epilogue
                # loads (VMEM: row IDs, A and W scales) go first so that these MFMAs
                # cover them. ========
                epi_in = epi.prepare(tile.m_block, expert, tile.n_tile, tile.half)
                b_l, a_t = k_pair(nxt_src, 2 * nxt.kp0, b_l, a_t)

                # ======== Epilogue: C * A and W scales -> SiLU(gate) * up -> BF16,
                # permlane-packed 16-byte VMEM stores (stream-K: see finish) ========
                epi.finish(epi_in, tile)
                cursor.advance(nxt_src)

            # ======== Stream-K: heads of split tiles add the other K ranges ========
            epi.merge_head()
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
