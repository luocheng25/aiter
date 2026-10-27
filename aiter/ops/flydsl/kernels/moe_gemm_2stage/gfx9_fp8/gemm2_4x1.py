# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 M128 Down: four waves of 32 rows, LDS-DMA B packets, packed BF16 output.

Derived from pyhip tests/flydsl/moe_8w_down/moe_multistage_down_m128.py (per-128
block scales) and adapted to per-token activation x per-channel weight scales.

Task: one 128-row sorted block (A, rows gathered by route) times one OC slice of
N/OC columns of its expert's weights (B). Each wave owns 32 rows and keeps A for
all of K in VGPRs (K/8 per lane).

Output: [block128, N64, row128, col64] BF16 without each route's activation
scale and topk weight; ``make_packed_sum`` applies them while it reduces the
routes.

Schedule: a CTA runs one task, or persistent CTAs claim tasks from eight sharded
queues (task_queue). With merge_halves a task covers a 256-row sort block, whose
second half runs only when it holds routes.

Packets: a task walks its slice in N64 x K packets n = 0, 1, ..., starting at a
worker-rotated N128 pair to spread B reads. A packet splits K into SP sub-packets
k of KU K128 steps, and B streams through an LDS ring that LDS DMA keeps
`lookahead` sub-packets ahead. Up to K256 a packet is one sub-packet in a
four-slot ring; above K256 it is KB sub-packets of N64 x K128 in a ring of five
to nine 8 KiB slots, so B stays at 32 VGPRs (_ring_plan).

Stages: two CTAs per CU alternate a memory and a compute stage per sub-packet,
the memory stage at high priority (like the inter-wave Gluon GEMM). Packets run
three deep: packet n's compute stages also run packet n-1's epilogue and packet
n-2's stores. Each stage repeats one pattern:

    stage    pattern
    memory   (prio 3) per j: one 16-byte LDS DMA per thread (4 KiB per CTA) of
             the sub-packet `lookahead` ahead, into the ring slot the previous
             sub-packet freed, then four ds_read_b128 of this one; the packet's
             last sub-packet also reads WS(n)
    compute  (prio 0) per MFMA i: packet n-2's store due at i (store_at), one
             16x16x128 MFMA (two row x four column fragments per K128 step),
             then per_step VALU steps of packet n-1's epilogue (weight scale,
             BF16, permlane pack)

Naming: M = sorted rows (routes), N = output columns (weight rows), K = the
reduction. An m / n / k in a name gives its direction (block_m, n_split,
n16_bytes); an n_ prefix always means N, never "number of".

    n, k          packet n (the n-th N64 x K packet, at packet_index(n) in the
                  slice) and its sub-packet k (a K chunk of KU K128 steps)
    B[s]          sub-packet s = n * SP + k of the stream, in ring slot s % slots
    cur, pf       the sub-packet a memory stage reads / prefetches
    mi, ni        16-row fragment (two per wave) / N16 column group (four per
                  packet)
    kb, kh        K128 step / K64 half of one
    j, i          DMA and LR group j of a memory stage; MFMA i of a packet
    *_g2s, ...    data movement, g = global, s = LDS, r = VGPRs: fill_g2s and
                  load_g2s (LDS DMA), read_s2r (LDS read, in flight until
                  wait_s2r), wait_g2s, rows_g2r (A rows), store_r2g (output)
    B, WS, O      operands and output; lowercase a, b, c, ws: VGPR values
    c1, out2      a digit counts packets back: C(n-1), WS(n-1) and out(n-2)

Pipeline rules, to keep when editing the kernel body:

 1. Program order is the schedule: the compiler keeps LDS reads behind LDS DMA
    writes that may alias them, and sched_barrier fences pin the epilogue steps,
    so the kernel body writes the patterns out in order.
 2. Ring: B[s] lives in slot s % slots. The prologue DMAs B[0 .. lookahead-1]; the
    memory stage of B[s] DMAs B[s + lookahead] into B[s-1]'s slot, after
    wait_g2s, whose barrier follows every wave's wait_s2r of B[s-1].
 3. One async group per sub-packet, in stream order (load_g2s closes it), so
    wait_g2s for B[s] leaves at most min(lookahead, sub-packets left) - 1 pending.
 4. A memory stage pairs DMA j with LR group j (4 * sub_dmas == 8 * KU), and its
    reads land in wait_s2r, before the compute stage.
 5. Packet n-1's epilogue steps stay between sched_barrier fences and end within
    packet n: per_step * 8 * KB >= 44.
 6. Packet n hands C(n), WS(n) and out(n-1) on to packet n + 1 (_Carry); the last
    two packets' stores follow the packets.
 7. The dynamic loop spans whole ring cycles (static slots): it starts at the first
    cycle boundary with n >= 2 (earlier packets lack n-1 / n-2) and stops while
    every sub-packet still prefetches; static head and tail packets wrap it.
 8. Pins (inline asm "=v,0") hold values in VGPRs from where they are emitted:
    B's LDS read addresses after the A loads, b / ws in wait_s2r, C at the
    packet's end.
 9. A task ends with wait_asyncmark(0) and vmcnt / lgkmcnt(0); next_step's barrier
    then hands the LDS arena to the next task.
"""

import functools
import itertools
from typing import Any, NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

from aiter.jit.utils.chip_info import get_cu_num

from . import common as fxh
from .common import Bundle, is_route, scalar, split_route, swap_pack

PACKED_ROWS = 128
_PACKET_BYTES_PER_K = 64  # an N64 packet holds 64 * K FP8 bytes
_DMA_BYTES = 4096  # one 16-byte LDS DMA per thread of the 256-thread CTA
# B in LDS, per N16 x K128 block: lane (lr, lk) reads 16 B at lr * 16 +
# lk * _N16_K16_BYTES, plus _N16_K64_BYTES for the second K64 half.
_N16_K16_BYTES = 256
_N16_K64_BYTES = 1024
_N16_K128_BYTES = 2048
_EPILOGUE_STEPS = 4 * 11  # 4 N32 pairs x (8 scales + 2 BF16 + 1 pack)
_OUT_PACKET_BYTES = PACKED_ROWS * 64 * 2  # 128 rows x N64 BF16
_OUTPUT_AUX = 18  # SC1 | NT output stores
_WEIGHT_AUX = 16  # SC1 weight DMA


def m128_oc_splits(n, merge_halves=False):
    """Largest supported OC split whose column slice is a whole number of N128 pairs.

    Merged-half tasks may run two passes, so they take up to 16 slices of at least
    four N64 packets.
    """
    for oc in (16, 8, 4, 2, 1) if merge_halves else (8, 4, 2, 1):
        if (
            n % oc == 0
            and (n // oc) % 128 == 0
            and (not merge_halves or n // oc >= 256)
        ):
            return oc
    return None


def m128_supported(n, k):
    return k % 64 == 0 and 128 <= k <= 640 and m128_oc_splits(n) is not None


def arith_min(a, b):
    return (a < b).select(a, fx.Int32(b))


def _pin_address(value):
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type, [value.ir_value()], "", "=v,0", has_side_effects=True
        )
    )


def _pin_packet(value):
    raw = value.ir_value()
    return fx.Vector(
        llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True)
    )


def _stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _coalesce_output_pairs(first, second):
    """Exchange a row bit with a column bit for contiguous 128-byte stores."""
    mask, even = fx.Uint64(0xAAAAAAAAAAAAAAAA), fx.Uint64(0x5555555555555555)
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(
        f"v_cndmask_b32_dpp ${i}, ${12 + i}, ${8 + i}, vcc quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf"
        for i in range(4)
    )
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(
        f"v_cndmask_b32_dpp ${4 + i}, ${8 + i}, ${12 + i}, vcc quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf"
        for i in range(4)
    )
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>"),
        [first[i].ir_value() for i in range(4)]
        + [second[i].ir_value() for i in range(4)]
        + [even.ir_value(), mask.ir_value()],
        "\n".join(assembly),
        ",".join(["=&v"] * 8 + ["v"] * 8 + ["s", "s", "~{vcc}"]),
        has_side_effects=True,
    )
    return [
        _pin_packet(
            fx.Vector.from_elements(
                [
                    fx.Int32(
                        llvm.extractvalue(fx.Int32.ir_type, result, [side * 4 + i])
                    )
                    for i in range(4)
                ],
                fx.Int32,
            )
        )
        for side in range(2)
    ]


def _bf16_words(values):
    """Four f32 values -> two BF16x2 words (RNE, as the reference conversion)."""
    return [
        fx.Vector.from_elements(values[2 * w : 2 * w + 2], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        .ir_value()
        for w in range(2)
    ]


class _SubPacket(NamedTuple):
    """Sub-packet k (a K chunk) of packet n, in ring slot `slot`; `left` counts the
    sub-packets from it to the end of the stream (None for a dynamic n)."""

    n: Any
    k: int
    slot: int
    left: Any


class _RingPlan(NamedTuple):
    """How B streams through a task's LDS ring, and the packet loop's bounds."""

    KB: int  # K128 steps, rounded up
    SP: int  # sub-packets per packet
    KU: int  # K128 steps per sub-packet
    n16_bytes: int  # an N16 group of a sub-packet: contiguous preshuffled weights
    sub_bytes: int  # a sub-packet, one ring slot
    sub_dmas: int  # 16-byte LDS DMAs per thread per sub-packet
    slots: int
    slots_per_base: int  # ring slots per pinned ds_read base
    lookahead: int  # sub-packets the DMA runs ahead
    subs: int  # sub-packets in the stream
    body: int  # packets per ring cycle: the main loop's step
    prologue: int  # static head packets
    loop_end: int  # the first static tail packet
    per_step: int  # packet n-1 epilogue steps after each MFMA of packet n
    store_at: dict  # MFMA of packet n -> packet n-2 record stored before it


def _ring_plan(K, packets):
    """The ring and packet-loop geometry of a task, checked against the pipeline
    rules (module docstring)."""
    # K % 128 == 64 pads the last K128 step: its A half past K loads as zero, so the
    # B bytes read past an N16 group (the next group's, or zero) add nothing.
    KB = (K + 127) // 128
    SP = 1 if K <= 256 else KB
    KU = KB // SP
    sub_bytes = 4 * KU * _N16_K128_BYTES
    # At most nine 8 KiB slots keep two CTAs per CU; a ring cycle holds whole packets,
    # so every packet's slots are static.
    slots = 4 if SP == 1 else 9 // SP * SP
    lookahead = slots - 1
    subs = packets * SP
    body = slots // SP
    # The main loop starts on a ring cycle with n >= 2 (packets n-1 and n-2 exist)
    # and ends while every sub-packet still prefetches; static packets wrap it.
    prologue = min(packets, -(-2 // body) * body)
    steady_end = (subs - lookahead) // SP
    loop_end = prologue + max(0, (steady_end - prologue) // body * body)
    plan = _RingPlan(
        KB=KB,
        SP=SP,
        KU=KU,
        n16_bytes=KU * _N16_K128_BYTES,
        sub_bytes=sub_bytes,
        sub_dmas=sub_bytes // _DMA_BYTES,
        slots=slots,
        slots_per_base=1 if SP == 1 else 4,
        lookahead=lookahead,
        subs=subs,
        body=body,
        prologue=prologue,
        loop_end=loop_end,
        per_step=(_EPILOGUE_STEPS + 8 * KB - 1) // (8 * KB),
        # One record mid-way through each quarter of the packet's 8 * KB MFMAs.
        store_at={(2 * r + 1) * KB: r for r in range(4)},
    )
    assert 4 * plan.sub_dmas == 8 * KU, "a memory stage pairs each DMA with 4 LRs"
    assert slots * sub_bytes <= 9 * 8192, "the ring leaves room for two CTAs per CU"
    assert plan.per_step * 8 * KB >= _EPILOGUE_STEPS, "epilogue(n-1) ends in packet n"
    assert loop_end == prologue or (
        prologue % body == 0 and loop_end * SP + lookahead <= subs
    ), "the main loop starts on a ring cycle and every sub-packet in it prefetches"
    return plan


class _Carry(NamedTuple):
    """What packet n hands on: C(n), WS(n) and its packed out(n-1), which packet
    n + 1 sees as c1, ws1 and out2."""

    c1: Any  # None before packet 1
    ws1: Any
    out2: Any  # empty before packet 2

    @property
    def has_out2(self):
        return len(self.out2) > 0

    def flat(self):
        return [*self.c1, *self.out2, *self.ws1]

    @staticmethod
    def unflat(values):
        """flat's inverse: 8 C fragments, 4 packed records, 4 weight-scale vectors."""
        vectors = [fx.Vector(values[i]) for i in range(16)]
        return _Carry(c1=vectors[:8], ws1=vectors[12:], out2=vectors[8:12])


_NO_CARRY = _Carry(c1=None, ws1=None, out2=())


def _zero_acc():
    """C(n) = 0: two row x four column fragments of 4 f32 per lane."""
    c = [fx.make_rmem_tensor(4, fx.Float32) for _ in range(8)]
    for frag in c:
        frag.fill(0)
    return c


def _acc_values(c):
    """C(n) pinned in VGPRs as values: the next packet's C(n-1)."""
    return [_pin_packet(frag.load()) for frag in c]


def _epilogue_row(c, ws, mi, pieces):
    """Row fragment mi of a packet's epilogue: weight scale, BF16 and permlane pack
    of each N32 pair into `pieces`, yielding after every VALU step."""
    for pair in range(2):
        scaled = []
        for ni in (2 * pair, 2 * pair + 1):
            for element in range(4):
                scaled.append(c[mi * 4 + ni][element] * ws[ni][element])
                yield
        words = []  # words 0/1 hold ni even, 2/3 ni odd
        for odd in range(2):
            words.extend(_bf16_words(scaled[4 * odd : 4 * odd + 4]))
            yield
        pieces.append(_pin_packet(swap_pack(words[:2], words[2:])))
        yield


@functools.cache
def build_moe_down_m128(
    *,
    N,
    K,
    TOPK,
    sort_block_m=128,
    persistent=False,
    merge_halves=False,
    oc_splits=None,
):
    """PTPC FP8 Down on gfx950; returns the launcher taking the aiter Down ABI.

    Without ``persistent`` each CTA runs one task. Persistent launches two CTAs per
    CU that claim tasks from eight sharded queues in the caller's zero-initialized
    ``counter`` (eight heads, 128 bytes apart). Each shard's last terminal claimer
    restores its head to zero, so the counter stays zero between launches; launches
    sharing a counter must not run concurrently.

    A task is a 128-row block's OC slice, or with ``merge_halves`` a 256-row sort
    block's OC slice whose second half runs only when it holds routes. Merging
    avoids empty-half tasks while experts average at most 128 routes; with fuller
    blocks the separate halves read each weight slice concurrently and win.
    ``oc_splits`` overrides the default OC split.
    """
    # A holds K/8 VGPRs per lane; past K640 it no longer fits two CTAs per CU.
    assert (
        K % 64 == 0 and 128 <= K <= 640
    ), "M128 Down supports K = 128..640, K % 64 == 0"
    assert sort_block_m in (128, 256)
    if oc_splits is None:
        oc_splits = m128_oc_splits(N, merge_halves)
    assert (
        oc_splits is not None and N % oc_splits == 0 and N // oc_splits % 128 == 0
    ), f"N={N} has no OC split with N128-pair slices"
    n_split = N // oc_splits
    packets = n_split // 64
    pair_count = packets // 2
    packet_bytes = _PACKET_BYTES_PER_K * K
    plan = _ring_plan(K, packets)
    KB, SP, KU, subs = plan.KB, plan.SP, plan.KU, plan.subs
    n16_bytes, sub_bytes, sub_dmas = plan.n16_bytes, plan.sub_bytes, plan.sub_dmas
    slots, slots_per_base = plan.slots, plan.slots_per_base
    lookahead, body = plan.lookahead, plan.body
    prologue, loop_end = plan.prologue, plan.loop_end
    per_step, store_at = plan.per_step, plan.store_at
    blocks_per_sort = sort_block_m // PACKED_ROWS
    halves = blocks_per_sort if merge_halves else 1
    task_rows = PACKED_ROWS * halves
    num_workers = 2 * get_cu_num()
    assert num_workers % 8 == 0
    # CTA w runs on XCD w % 8; a width-W transpose spreads a block's OC slices over
    # 8 / W XCDs that each fetch its 128 x K A rows. Width 8 wins once those are
    # >= 5% of the expert's N x K weights.
    width = 8 if not persistent and N <= 2560 and oc_splits >= 4 else 4
    # LDS words: B ring, the task's weight-scale slice, sorted IDs.
    ring_words = slots * sub_bytes // 4
    wscale_offset = ring_words
    ids_offset = wscale_offset + n_split
    task_offset = ids_offset + PACKED_ROWS
    arena_words = task_offset + 1
    ws_per_thread = (n_split + 255) // 256
    attrs = {
        "rocdl.waves_per_eu": 2,
        "rocdl.flat_work_group_size": "256,256",
        "llvm.passthrough": [
            ["target-features", "-packed-fp32-ops"],
            ["amdgpu-agpr-alloc", "0,0"],
        ],
    }

    @fx.struct
    class Storage:
        arena: fx.Array[fx.Int32, arena_words, 16]

    def kernel_context(p_sorted_ids, p_sorted_expert_ids, p_w_scale, tokens):
        """Thread coordinates, routing and scale views, and the LDS arena."""
        tid = fx.Int32(fx.thread_idx.x)
        return Bundle(
            tid=tid,
            wave=scalar(tid // 64),
            worker=fx.Int32(fx.block_idx.x),
            sorted_ids=fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids)),
            expert_ids=fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids)),
            w_scale=fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale)),
            lds=fx.SharedAllocator().allocate(Storage).peek(),
            tokens=tokens,
        )

    def task_queue(ctx, p_num_valid_ids, p_counter):
        """Task ranks. A one-shot CTA's rank is its task; persistent CTAs claim ranks
        from shard worker % 8, whose rank r is task r * 8 + shard. `claim_word` is
        the LDS word a claim is broadcast through."""
        valid_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids))
        tasks = (scalar(valid_ids[0]) // task_rows) * oc_splits
        claim_word = fx.make_view(ctx.lds.arena.ptr + task_offset, fx.make_layout(1, 1))
        counter = fx.recast_iter(fx.Int32, fxh._as_ptr(p_counter))
        shard = ctx.worker % 8
        chunk = tasks // width

        def claim():
            head = llvm.inttoptr(
                ir.Type.parse("!llvm.ptr<1>"),
                fx.ptrtoint(counter + shard * 32).ir_value(),
            )
            return fx.Int32(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    head,
                    fx.Int32(1).ir_value(),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                ).res
            )

        def task_of(rank):
            if persistent:
                return rank * 8 + shard
            return rank

        def slice_of(task):
            """(block unit, OC slice) of a task: the valid task prefix is transposed,
            the tail keeps identity."""
            task = (task < chunk * width).select(
                task % width * chunk + task // width, task
            )
            return task // oc_splits, task % oc_splits

        def last_rank():
            """Rank of the shard's last claim: one per task, one failing per worker."""
            return (tasks - shard + 7) // 8 + num_workers // 8 - 1

        def reset():
            counter[shard * 32] = fx.Int32(0)

        def next_half_live(half, row_begin):
            """Whether the task's second half runs next: only if it holds routes."""
            more = half + 1 < halves
            if halves > 1:
                nxt = scalar(
                    ctx.sorted_ids[more.select(row_begin + PACKED_ROWS, row_begin)]
                ).bitcast(fx.Uint32)
                more = more & is_route(nxt, ctx.tokens, TOPK)
            return more

        @flyc.jit
        def first_step():
            """The CTA's first step, rank * halves + half; persistent CTAs claim the
            rank and broadcast it through LDS."""
            rank = ctx.worker
            if const_expr(persistent):
                word = claim_word
                if ctx.tid == 0:
                    word[0] = claim()
                fx.barrier()
                rank = scalar(word[0])
            return rank * halves

        @flyc.jit
        def claim_next(half):
            """Persistent CTAs claim the next rank now; its latency hides behind the
            current task."""
            next_rank = fx.Int32(0)
            if const_expr(persistent):  # noqa: SIM102 (compile-time guard)
                if (ctx.tid == 0) & (half == 0):
                    next_rank = claim()
            return next_rank

        @flyc.jit
        def next_step(step, half, row_begin, next_rank):
            """The task's second half if it holds routes, else the next rank; a one-shot
            CTA's next rank is past every task."""
            fx.barrier()  # the next task reuses the LDS arena
            more = next_half_live(half, row_begin)
            if const_expr(persistent):
                word = claim_word
                if (ctx.tid == 0) & (half == 0):
                    word[0] = next_rank
                fx.barrier()
                step = more.select(step + 1, scalar(word[0]) * halves)
            else:
                step = more.select(step + 1, tasks * halves)
            return step

        @flyc.jit
        def drain():
            """Every worker of a shard makes one failing claim; the last proves the
            queue is drained and resets it for the next launch."""
            if const_expr(persistent):
                word = claim_word
                if ctx.tid == 0:  # noqa: SIM102 (only thread 0 reads the word)
                    if word[0] == last_rank():
                        reset()

        return Bundle(
            tasks=tasks,
            task_of=task_of,
            slice_of=slice_of,
            first_step=first_step,
            claim_next=claim_next,
            next_step=next_step,
            drain=drain,
        )

    def locate(ctx, queue, task, half):
        """A task's OC slice, 128-row block and expert, whether the block holds routes,
        the lane's place in it, and the LDS views its metadata is staged in."""
        tokens = ctx.tokens
        unit, oc = queue.slice_of(task)
        block_m = unit * halves + half
        row_begin = block_m * PACKED_ROWS
        # Sorting puts an expert's routes before its padding, so a 128-row half of a
        # 256-row sort block is empty iff its first entry is padding.
        head = scalar(ctx.sorted_ids[row_begin]).bitcast(fx.Uint32)
        live = is_route(head, tokens, TOPK)
        expert = scalar(ctx.expert_ids[block_m // blocks_per_sort])
        rows = tokens * TOPK
        lane = ctx.tid % 64
        lr, lk = lane % 16, lane // 16
        iptr = ctx.lds.arena.ptr
        fptr = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, iptr.memspace, 16), iptr
        )
        ids = fx.make_view(iptr + ids_offset, fx.make_layout(PACKED_ROWS, 1))
        wscales = fx.make_view(fptr + wscale_offset, fx.make_layout(n_split, 1))
        return Bundle(
            oc=oc,
            block_m=block_m,
            row_begin=row_begin,
            live=live,
            expert=expert,
            rows=rows,
            lr=lr,
            lk=lk,
            iptr=iptr,
            fptr=fptr,
            ids=ids,
            wscales=wscales,
            zero=fx.Int32(0).ir_value(),
        )

    def slice_scales(ctx, task):
        """The weight scales this thread stages, one per 256 columns of the OC slice."""
        return [
            ctx.w_scale[
                task.expert * N
                + task.oc * n_split
                + fx.Int32(arith_min(ctx.tid + i * 256, n_split - 1))
            ]
            for i in range(ws_per_thread)
        ]

    @flyc.jit
    def stage_task(ctx, task):
        """VMEM -> LDS: the block's sorted IDs (a thread each for 128 rows) and the
        slice's weight scales (a thread each for 256 columns). Returns the IDs view."""
        ids, wscales = task.ids, task.wscales
        ws_values = slice_scales(ctx, task)
        if ctx.tid < PACKED_ROWS:
            ids[ctx.tid] = ctx.sorted_ids[task.row_begin + ctx.tid]
        for i in range_constexpr(ws_per_thread):
            col = ctx.tid + i * 256
            if col < n_split:
                wscales[col] = ws_values[i]
        fx.barrier()
        return ids

    def b_ring(ctx, task, p_weight):
        """B's LDS ring for the task's weight slice: sub-packet s lands in slot
        s % slots; packets start at a worker-dependent N128 pair. DMA side: fill_g2s,
        load_g2s, wait_g2s; LDS read side: pin_read_bases, read_s2r, wait_s2r."""
        tid = ctx.tid
        wview = fx.make_view(
            fxh._as_ptr(p_weight, fx.Uint8)
            + fx.Int64(task.expert) * (N * K)
            + fx.Int64(task.oc) * (n_split * K),
            fx.make_layout(n_split * K, 1),
        )
        wbuf = fx.rocdl.make_buffer_tensor(wview, False)
        wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
        bptr = task.iptr
        blds = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
        # A wave's 64 lanes DMA 16 B each.
        dma_base = scalar(fx.Int32(fx.ptrtoint(bptr))) + ctx.wave * (64 * 16)
        if SP == 1:
            dma_offset = _pin_address(tid * 16)
        else:
            # Byte tid * 16 of a DMA's 4 KiB lies in N16 group (tid * 16) // n16_bytes.
            dma_offset = _pin_address(
                tid * 16 // n16_bytes * (16 * K) + tid * 16 % n16_bytes
            )
        rotation = ctx.worker // 8 % pair_count

        def packet_index(n):
            """Where packet n sits in the slice, in N64 units."""
            # n < packets, so the rotated pair needs one conditional wrap.
            pair = (fx.Int32(n) >> 1) + rotation
            pair = (pair >= pair_count).select(pair - pair_count, pair)
            return pair * 2 + (fx.Int32(n) & 1)

        def subpacket(n, k, cycle):
            """Sub-packet k of packet n; `cycle` is n's place in the ring cycle."""
            left = subs - n * SP - k if isinstance(n, int) else None
            return _SubPacket(n, k, (cycle * SP + k) % slots, left)

        def prefetch_target(cur):
            """What cur's memory stage prefetches: the sub-packet `lookahead` after
            cur, into the slot of the one before cur; None once the stream has
            ended. Issues nothing (load_g2s does)."""
            if cur.left is not None and cur.left <= lookahead:
                return None
            ahead = cur.k + lookahead
            return _SubPacket(
                cur.n + ahead // SP, ahead % SP, (cur.slot + lookahead) % slots, None
            )

        def wait_g2s(cur):
            """Until the DMA of sub-packet cur has landed, then a CTA barrier, after
            which no wave reads the slot before cur any more."""
            in_flight = lookahead if cur.left is None else min(lookahead, cur.left)
            rocdl.wait_asyncmark(in_flight - 1)
            _stage_end()

        def load_g2s(sp, j):
            """LDS DMA j of sub-packet sp into its slot: 16 B per thread, 4 KiB per
            CTA. The last one closes sp's async group. Nothing for None."""
            if sp is None:
                return
            dst = llvm.inttoptr(
                ir.Type.parse("!llvm.ptr<3>"),
                (dma_base + sp.slot * sub_bytes + j * _DMA_BYTES).ir_value(),
            )
            source = sp.k * n16_bytes + j * _DMA_BYTES // n16_bytes * (16 * K)
            rocdl.raw_ptr_buffer_load_async_lds(
                wrsrc,
                dst,
                fx.Int32(16).ir_value(),
                dma_offset.ir_value(),
                (packet_index(sp.n) * packet_bytes + source).ir_value(),
                task.zero,
                aux=ir.IntegerAttr.get(fx.Int32.ir_type, _WEIGHT_AUX),
            )
            if j == sub_dmas - 1:
                rocdl.asyncmark()

        def fill_g2s(s):
            """DMA sub-packet s of the stream into its slot."""
            sp = _SubPacket(s // SP, s % SP, s % slots, None)
            for j in range(sub_dmas):
                load_g2s(sp, j)

        read_bases = []

        def pin_read_bases():
            """Pin the ds_read bases, one per slot or per four 8 KiB slots."""
            read_bases.extend(
                _pin_address(
                    fx.Int32(fx.ptrtoint(bptr))
                    + task.lr * 16
                    + task.lk * _N16_K16_BYTES
                    + base * slots_per_base * sub_bytes
                )
                for base in range((slots + slots_per_base - 1) // slots_per_base)
            )

        def read_s2r(cur, j):
            """LR j of sub-packet cur, its K128 step j // 2 and N32 half j % 2: four
            ds_read_b128, in flight until wait_s2r."""
            base = read_bases[cur.slot // slots_per_base]
            slot_offset = cur.slot % slots_per_base * sub_bytes
            kb = j // 2
            values = []
            for ni in (2 * (j % 2), 2 * (j % 2) + 1):
                for kh in range(2):
                    offset = (
                        slot_offset
                        + ni * n16_bytes
                        + kb * _N16_K128_BYTES
                        + kh * _N16_K64_BYTES
                    )
                    values.append(blds._at(base, offset).load())
            return values

        def wait_s2r(b, ws):
            """Wait for the memory stage's LDS reads, then pin them in VGPRs: b, a
            sub-packet's reads in (kb, ni, kh) order, becomes its MFMA fragments."""
            rocdl.s_waitcnt(lgkmcnt=0)
            frags = fx.make_rmem_tensor([8, 4, KU], fx.Int32)
            words = fx.make_view(
                fx.get_iter(frags), fx.make_ordered_layout([4, 2, 4, KU], 0)
            )
            for j, value in enumerate(b):
                words[None, j % 2, j % 8 // 2, j // 8].store(_pin_packet(value))
            return frags, None if ws is None else [_pin_packet(w) for w in ws]

        return Bundle(
            packet_index=packet_index,
            subpacket=subpacket,
            prefetch_target=prefetch_target,
            wait_g2s=wait_g2s,
            load_g2s=load_g2s,
            fill_g2s=fill_g2s,
            pin_read_bases=pin_read_bases,
            read_s2r=read_s2r,
            wait_s2r=wait_s2r,
        )

    def rows_g2r(ctx, task, ids, p_input, p_output):
        """VMEM -> VGPR: this lane's two A rows, all of K (K steps past K load as
        zero); and where the rows it stores go in the packed output."""
        wave, tokens = ctx.wave, ctx.tokens
        abuf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(fx.Int32.ir_type, p_input.memspace, 16),
                    p_input,
                ),
                fx.make_layout(task.rows * (K // 4), 1),
            ),
            False,
        )
        apacket = fxh.BufferTensor(
            fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1))
        )
        # Packed rows are addressed per 128-row block so the total may pass 4 GiB.
        obuf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.Uint8)
                + fx.Int64(task.block_m) * (PACKED_ROWS * N * 2),
                fx.make_layout(PACKED_ROWS * N * 2, 1),
            ),
            False,
        )
        orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
        a = fx.make_rmem_tensor([8, 2, KB], fx.Int32)
        awords = fx.make_view(fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, KB], 0))
        lr, lk = task.lr, task.lk
        out_addresses = []
        for mi in range(2):
            row = wave * 32 + mi * 16 + lr
            encoded = ids[row].bitcast(fx.Uint32)
            token, topk_slot, valid = split_route(encoded, tokens, TOPK)
            source = fx.Int32(token) * TOPK + fx.Int32(topk_slot)
            for kb in range(KB):
                for kh in range(2):
                    if kb * 128 + kh * 64 >= K:
                        awords[None, kh, mi, kb].store(fx.Vector.filled(4, 0, fx.Int32))
                    else:
                        offset = valid.select(
                            source * K + lk * 16 + kb * 128 + kh * 64, fx.Int32(-1)
                        )
                        awords[None, kh, mi, kb].store(
                            apacket.load(
                                voffset_bytes=offset, soffset_bytes=fx.Int32(0)
                            )
                        )
            for parity in range(2):
                target = wave * 32 + mi * 16 + lr // 2 * 2 + parity
                encoded = ids[target].bitcast(fx.Uint32)
                _, _, valid = split_route(encoded, tokens, TOPK)
                column = ((lk & 1) * 2 + (lk >> 1)) * 8 + lr % 2 * 32
                offset = (
                    task.oc * (PACKED_ROWS * n_split * 2) + target * 128 + column * 2
                )
                out_addresses.append(
                    _pin_address(valid.select(offset, fx.Int32(PACKED_ROWS * N * 2)))
                )
        return a, Bundle(orsrc=orsrc, addresses=out_addresses)

    def packet_operands(task, B, out):
        """WS (the packets' weight scales in LDS), O (the packed output) and mma."""
        flds = fxh.LdsTensor(fx.make_view(task.fptr, fx.make_layout(4, 1)))
        ws_base = (
            scalar(fx.Int32(fx.ptrtoint(task.fptr + wscale_offset))) + task.lk * 16
        )
        atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        packet_index = B.packet_index

        def read_ws(n):
            """LR of packet n's weight scales, 16 per N16 column group."""
            base = ws_base + packet_index(n) * (64 * 4)
            return [flds._at(base, ni * (16 * 4)).load() for ni in range(4)]

        def store_r2g(packed, n, record=None):
            """VMEM: packed record `record` of packet n (default all four), 16 B per
            lane."""
            for r in range(4) if record is None else (record,):
                rocdl.raw_ptr_buffer_store(
                    packed[r].ir_value(),
                    out.orsrc,
                    out.addresses[r].ir_value(),
                    (packet_index(n) * _OUT_PACKET_BYTES).ir_value(),
                    aux=ir.IntegerAttr.get(fx.Int32.ir_type, _OUTPUT_AUX),
                )

        def mma(c, b, a):
            """c += b a; the weights are the MFMA's operand A."""
            fx.gemm(atom, c, b, a, c)

        return Bundle(read_s2r=read_ws), Bundle(store_r2g=store_r2g), mma

    def epilogue(prev):
        """The epilogue of the packet `prev` (a _Carry) came from: C * WS -> BF16 ->
        permlane pack, as VALU steps. run(count) issues the next count between
        scheduling fences, finish() the rest and returns the four packed records.
        Nothing before packet 1 (no C)."""
        pieces = [[], []]
        rows = []
        if prev.c1 is not None:
            rows = [_epilogue_row(prev.c1, prev.ws1, mi, pieces[mi]) for mi in range(2)]
        steps = itertools.chain(*rows)

        def run(count):
            rocdl.sched_barrier(0)
            for _ in range(count):
                next(steps, None)
            rocdl.sched_barrier(0)

        def finish():
            records = []
            for mi, row in enumerate(rows):
                for _ in row:  # steps not run yet
                    pass
                records.extend(_coalesce_output_pairs(*pieces[mi]))
            return records

        return Bundle(run=run, finish=finish)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_down_m128(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_counter: fx.Pointer,
        tokens: fx.Int32,
    ):
        ctx = kernel_context(p_sorted_ids, p_sorted_expert_ids, p_w_scale, tokens)
        queue = task_queue(ctx, p_num_valid_ids, p_counter)
        step = queue.first_step()  # rank * halves + half
        while queue.task_of(step // halves) < queue.tasks:
            half = step % halves
            task_id = queue.task_of(step // halves)
            next_rank = queue.claim_next(half)
            task = locate(ctx, queue, task_id, half)
            if task.live:
                # ======== Prologue: the task's operands ========
                # VMEM -> LDS: the block's sorted IDs and the slice's weight scales.
                # DMA: B sub-packets 0 .. lookahead-1 into ring slots 0 .. lookahead-1,
                # sub_dmas x 16 B per thread each. VMEM -> VGPR: A, all of K.
                ids = stage_task(ctx, task)
                B = b_ring(ctx, task, p_weight)
                for s in range_constexpr(min(lookahead, subs)):
                    B.fill_g2s(s)  # DMA B[s]
                a, out = rows_g2r(ctx, task, ids, p_input, p_output)  # VMEM A -> VGPR
                B.pin_read_bases()  # after the A loads (rule 8)
                WS, O, mma = packet_operands(task, B, out)
                rocdl.sched_barrier(0)

                # ======== Packet n: C(n) = A B(n), N64 x K ========
                # B(n) is SP sub-packets B[s], s = n * SP + k, of KU K128 steps; B[s]
                # is in ring slot s % slots, DMA'd `lookahead` sub-packets ahead.
                # Packet n's compute stages also run packet n-1's epilogue and packet
                # n-2's stores; prev carries C(n-1), WS(n-1) and out(n-2) (_Carry).
                # Defaults bind this task's operands: the def sits in the task loop.
                def packet(n, cycle, prev, a=a, B=B, WS=WS, O=O, mma=mma):
                    c = _zero_acc()  # C(n) = 0
                    epi = epilogue(prev)  # out(n-1) from C(n-1), WS(n-1)
                    for k in range_constexpr(SP):
                        cur = B.subpacket(n, k, cycle)  # B[s], s = n * SP + k
                        # -- memory stage (prio 3): per j, an LDS DMA, then four LRs --
                        rocdl.s_setprio(3)
                        B.wait_g2s(cur)  # B[s] landed; CTA barrier: B[s-1] fully read
                        pf = B.prefetch_target(cur)  # B[s + lookahead] -> B[s-1]'s slot
                        b = []  # B[s], LDS reads in flight
                        for j in range_constexpr(sub_dmas):
                            B.load_g2s(pf, j)  # DMA j of B[s + lookahead]: 4 KiB
                            b += B.read_s2r(cur, j)  # LR B[s]: K128 j // 2, N32 j % 2
                        ws = WS.read_s2r(n) if k == SP - 1 else None  # LR WS(n)
                        b, ws = B.wait_s2r(b, ws)  # lgkmcnt(0): b, ws in VGPRs
                        # -- compute stage (prio 0): MFMA C(n)[mi, ni] += A[mi, kb]
                        # b[ni, kb], then per_step VALU steps of out(n-1) --
                        rocdl.s_setprio(0)
                        for i in range_constexpr(8 * KU * k, 8 * KU * (k + 1)):
                            kb, mi, ni = i // 8, i % 8 // 4, i % 4  # MFMA i of packet n
                            if const_expr(prev.has_out2 and i in store_at):
                                O.store_r2g(prev.out2, n - 2, store_at[i])  # VMEM
                            mma(c[mi * 4 + ni], b[None, ni, kb % KU], a[None, mi, kb])
                            epi.run(per_step)  # VALU out(n-1)
                    return _Carry(c1=_acc_values(c), ws1=ws, out2=epi.finish())

                # ======== Packets: static head, main loop, static tail ========
                carry = _NO_CARRY
                for n in range_constexpr(prologue):
                    carry = packet(n, n % body, carry)
                if const_expr(loop_end > prologue):
                    # Whole ring cycles of packets, so every slot is static.
                    for n, values in range(
                        fx.Int32(prologue),
                        fx.Int32(loop_end),
                        fx.Int32(body),
                        init=carry.flat(),
                    ):
                        carry = _Carry.unflat(values)
                        for p in range_constexpr(body):
                            carry = packet(n + p, p, carry)
                        final = yield carry.flat()
                    carry = _Carry.unflat(final)
                for n in range_constexpr(loop_end, packets):  # the DMA runs out
                    carry = packet(n, n % body, carry)

                # ======== Epilogue: the last two packets' stores ========
                if const_expr(carry.has_out2):
                    O.store_r2g(carry.out2, packets - 2)
                O.store_r2g(epilogue(carry).finish(), packets - 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            step = queue.next_step(step, half, task.row_begin, next_rank)
        queue.drain()

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
        p_counter: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        # Tasks are (128-row block or merged sort block, OC slice) pairs.
        moe_down_m128(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            p_counter,
            M,
            value_attrs=attrs,
        ).launch(
            grid=(
                (
                    num_workers
                    if const_expr(persistent)
                    else task_num * (blocks_per_sort // halves) * oc_splits
                ),
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def make_packed_sum(*, N, TOPK):
    """Sum each token's TOPK packed routes; inverse holds sorted positions per route.

    Route k of a token is scaled by route_scales[token * TOPK + k] (its activation
    scale) times sorted_weights at its sorted position.
    """
    assert N % 512 == 0 and 0 < TOPK <= 255
    threads = 64 if N % 2048 == 512 else 256
    columns_per_block = threads * 8
    column_blocks = (N + columns_per_block - 1) // columns_per_block

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def moe_packed_sum(
        p_output: fx.Pointer,
        p_source: fx.Pointer,
        p_inverse: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_route_scales: fx.Pointer,
        tokens: fx.Int32,
        source_rows: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x) // column_blocks
        block_col = fx.Int32(fx.block_idx.x) % column_blocks
        inverse = fx.recast_iter(fx.Int32, fxh._as_ptr(p_inverse))
        sorted_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_sorted_weights))
        route_scales = fx.recast_iter(fx.Float32, fxh._as_ptr(p_route_scales))
        target = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.Uint8) + fx.Int64(token) * (N * 2),
                fx.make_layout(N * 2, 1),
            ),
            False,
        )
        drsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(target))
        zero_offset = fx.Int32(0).ir_value()
        vector_type = ir.VectorType.get([4], fx.Int32.ir_type)
        locations = [
            scalar(inverse[token * TOPK + route]) for route in range_constexpr(TOPK)
        ]

        def read_route(location, column):
            valid_row = (location >= 0) & (location < source_rows)
            safe = valid_row.select(location, fx.Int32(0))
            bm, row = safe // PACKED_ROWS, safe % PACKED_ROWS
            block = fx.make_view(
                fxh._as_ptr(p_source, fx.Uint8) + fx.Int64(bm) * (PACKED_ROWS * N * 2),
                fx.make_layout(PACKED_ROWS * N * 2, 1),
            )
            srsrc = fx.rocdl.get_buffer_rsrc(
                fx.get_iter(fx.rocdl.make_buffer_tensor(block, False))
            )
            element = column // 64 * (PACKED_ROWS * 64) + row * 64 + column % 64
            offset = (valid_row & (column < N)).select(element * 2, fx.Int32(-1))
            return fx.Vector(
                rocdl.raw_ptr_buffer_load(
                    vector_type,
                    srsrc,
                    offset.ir_value(),
                    zero_offset,
                    aux=ir.IntegerAttr.get(fx.Int32.ir_type, 2),
                )
            ).bitcast(fx.BFloat16)

        column = block_col * columns_per_block + tid * 8
        fragments = [
            read_route(locations[route], column) for route in range_constexpr(TOPK)
        ]
        factors = []
        for route in range_constexpr(TOPK):
            location = locations[route]
            safe = ((location >= 0) & (location < source_rows)).select(
                location, fx.Int32(0)
            )
            factors.append(sorted_weights[safe] * route_scales[token * TOPK + route])
        accum = fragments[0].to(fx.Float32) * factors[0]
        for route in range_constexpr(1, TOPK):
            accum = accum + fragments[route].to(fx.Float32) * factors[route]
        offset = (column < N).select(column * 2, fx.Int32(-1))
        rocdl.raw_ptr_buffer_store(
            accum.to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
            drsrc,
            offset.ir_value(),
            zero_offset,
            aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
        )

    @flyc.jit
    def launch(
        p_output: fx.Pointer,
        p_source: fx.Pointer,
        p_inverse: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_route_scales: fx.Pointer,
        tokens: fx.Int32,
        source_rows: fx.Int32,
        stream: fx.Stream,
    ):
        moe_packed_sum(
            p_output,
            p_source,
            p_inverse,
            p_sorted_weights,
            p_route_scales,
            tokens,
            source_rows,
        ).launch(
            grid=(tokens * column_blocks, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch
