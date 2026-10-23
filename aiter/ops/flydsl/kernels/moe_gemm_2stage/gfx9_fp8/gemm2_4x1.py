# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 M128 Down: four waves of 32 rows, LDS-DMA B packets, packed BF16 output.

Derived from pyhip tests/flydsl/moe_8w_down/moe_multistage_down_m128.py (per-128
block scales) and adapted to per-token activation x per-channel weight scales.
A task is one 128-row sorted block times N/OC output columns, walked in N64 x K
packets that a four-slot LDS ring prefetches three packets ahead. Above K256 a
packet streams through a ring of five to nine N64 x K128 sub-packets, so B stays
at 32 VGPRs. The output layout is [block128, N64, row128, col64]; reduce it with
``make_packed_sum``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

from aiter.jit.utils.chip_info import get_cu_num

from . import common as fxh

PACKED_ROWS = 128
_PACKET_BYTES_PER_K = 64  # an N64 packet holds 64 * K FP8 bytes
_TURN_BYTES = 4096  # one 16-byte LDS DMA per thread of the 256-thread CTA
_OUTPUT_AUX = 18  # SC1 | NT output stores
_WEIGHT_AUX = 16  # SC1 weight DMA


def m128_oc_splits(n, merge_halves=False):
    """Largest supported OC split whose column slice is a whole number of N128 pairs.

    Merged-half tasks may run two passes, so they take up to 16 slices of at least
    four N64 packets.
    """
    for oc in (16, 8, 4, 2, 1) if merge_halves else (8, 4, 2, 1):
        if n % oc == 0 and (n // oc) % 128 == 0 and (not merge_halves or n // oc >= 256):
            return oc
    return None


def m128_supported(n, k):
    return k % 128 == 0 and k <= 640 and m128_oc_splits(n) is not None


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


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
                    fx.Int32(llvm.extractvalue(fx.Int32.ir_type, result, [side * 4 + i]))
                    for i in range(4)
                ],
                fx.Int32,
            )
        )
        for side in range(2)
    ]


def _swap_pack(words0, words1):
    """permlane16_swap two BF16x2 word pairs into one 16-byte store packet."""
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    lo = rocdl.permlane16_swap(pair_type, words0[0], words1[0], False, False)
    hi = rocdl.permlane16_swap(pair_type, words0[1], words1[1], False, False)
    return _pin_packet(
        fx.Vector.from_elements(
            [
                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [0])),
                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [0])),
                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, lo, [1])),
                fx.Int32(llvm.extractvalue(fx.Int32.ir_type, hi, [1])),
            ],
            fx.Int32,
        )
    )


def _bf16_words(values):
    """Four f32 values -> two BF16x2 words (RNE, as the reference conversion)."""
    return [
        fx.Vector.from_elements(values[2 * w : 2 * w + 2], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        .ir_value()
        for w in range(2)
    ]


@functools.cache
def build_moe_down_m128(
    *, N, K, TOPK, sort_block_m=128, persistent=False, merge_halves=False, oc_splits=None
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
    assert K % 128 == 0 and 128 <= K <= 640, "M128 Down supports K = 128..640"
    assert sort_block_m in (128, 256)
    if oc_splits is None:
        oc_splits = m128_oc_splits(N, merge_halves)
    assert oc_splits is not None and N % oc_splits == 0 and N // oc_splits % 128 == 0, (
        f"N={N} has no OC split with N128-pair slices"
    )
    KB = K // 128
    n_split = N // oc_splits
    packets = n_split // 64
    pair_count = packets // 2
    packet_bytes = _PACKET_BYTES_PER_K * K
    # Ring geometry: SP sub-packets of KU K128 steps per packet; an N16 row group of
    # a sub-packet is one contiguous `piece` of the preshuffled weights.
    SP = 1 if K <= 256 else KB
    KU = KB // SP
    piece = KU * 2048
    sub_bytes = 4 * piece
    sub_turns = sub_bytes // _TURN_BYTES
    # At most nine 8 KiB slots keep two CTAs per CU; the ring cycle spans whole
    # packets so phases stay static.
    slots = 4 if SP == 1 else 9 // SP * SP
    body = slots // SP
    lookahead = slots - 1
    subs = packets * SP
    # Pinned ds_read bases: one per phase, or one per four 8 KiB phases.
    phases_per_base = 1 if SP == 1 else 4
    blocks_per_sort = sort_block_m // PACKED_ROWS
    halves = blocks_per_sort if merge_halves else 1
    task_rows = PACKED_ROWS * halves
    num_workers = 2 * get_cu_num()
    assert num_workers % 8 == 0
    # CTA w runs on XCD w % 8; a width-W transpose spreads a block's OC slices over
    # 8 / W XCDs that each fetch its 128 x K A rows. Width 8 wins once those are
    # >= 5% of the expert's N x K weights.
    width = 8 if not persistent and N <= 2560 and oc_splits >= 4 else 4
    # LDS words: B ring, the task's weight-scale slice, sorted IDs, row factors.
    ring_words = slots * sub_bytes // 4
    wscale_offset = ring_words
    ids_offset = wscale_offset + n_split
    rows_offset = ids_offset + PACKED_ROWS
    task_offset = rows_offset + PACKED_ROWS
    arena_words = task_offset + 1
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
        tid = fx.Int32(fx.thread_idx.x)
        wave = _scalar(tid // 64)
        worker = fx.Int32(fx.block_idx.x)
        sorted_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids))
        sorted_weights = fx.recast_iter(fx.Float32, fxh._as_ptr(p_sorted_weights))
        expert_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))
        valid_ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids))
        w_scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_w_scale))
        a_scale = fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale))
        sort_tasks = (_scalar(valid_ids[0]) // task_rows) * oc_splits
        lds = fx.SharedAllocator().allocate(Storage).peek()
        task_slot = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))
        counter = fx.recast_iter(fx.Int32, fxh._as_ptr(p_counter))
        shard = worker % 8
        chunk = sort_tasks // width

        def claim():
            # Shard worker % 8 covers tasks rank * 8 + shard.
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
            if const_expr(persistent):
                return rank * 8 + shard
            return rank

        rank = worker
        if const_expr(persistent):
            if tid == 0:
                task_slot[0] = claim()
            fx.barrier()
            rank = _scalar(task_slot[0])
        # step = rank * halves + half
        step = rank * halves
        while task_of(step // halves) < sort_tasks:
            half = step % halves
            task = task_of(step // halves)
            # Claim the next task now; its latency hides behind this one.
            next_rank = fx.Int32(0)
            if const_expr(persistent):
                if (tid == 0) & (half == 0):
                    next_rank = claim()
            # Transpose the valid task prefix; the tail keeps identity.
            task = (task < chunk * width).select(
                task % width * chunk + task // width, task
            )
            unit, oc = task // oc_splits, task % oc_splits
            block_m = unit * halves + half
            row_begin = block_m * PACKED_ROWS
            # Sorting puts an expert's routes before its padding, so a 128-row
            # half of a 256-row sort block is empty iff its first entry is padding.
            head = _scalar(sorted_ids[row_begin]).bitcast(fx.Uint32)
            live = ((head & 0xFFFFFF) < fx.Uint32(tokens)) & ((head >> 24) < TOPK)
            expert = _scalar(expert_ids[block_m // blocks_per_sort])
            rows = tokens * TOPK
            lane = tid % 64
            lr, lk = lane % 16, lane // 16
            iptr = lds.arena.ptr
            fptr = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, iptr.memspace, 16), iptr
            )
            ids = fx.make_view(iptr + ids_offset, fx.make_layout(PACKED_ROWS, 1))
            row_factor = fx.make_view(fptr + rows_offset, fx.make_layout(PACKED_ROWS, 1))
            wscales = fx.make_view(fptr + wscale_offset, fx.make_layout(n_split, 1))
            asbuf = fx.rocdl.make_buffer_tensor(
                fx.make_view(a_scale, fx.make_layout(rows, 1)), False
            )
            asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
            zero = fx.Int32(0).ir_value()
            ws_turns = (n_split + 255) // 256
            if live:
                ws_values = [
                    w_scale[expert * N + oc * n_split + fx.Int32(arith_min(tid + turn * 256, n_split - 1))]
                    for turn in range_constexpr(ws_turns)
                ]
                if tid < PACKED_ROWS:
                    encoded = sorted_ids[row_begin + tid].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < TOPK)
                    ids[tid] = encoded.bitcast(fx.Int32)
                    source = fx.Int32(token) * TOPK + fx.Int32(slot)
                    offset = valid.select(source * 4, fx.Int32(-1))
                    act = fx.Float32(
                        rocdl.raw_ptr_buffer_load(
                            fx.Float32.ir_type, asrsrc, offset.ir_value(), zero
                        )
                    )
                    row_factor[tid] = act * sorted_weights[row_begin + tid]
                for turn in range_constexpr(ws_turns):
                    index = tid + turn * 256
                    if index < n_split:
                        wscales[index] = ws_values[turn]
                fx.barrier()
            if live:
                wview = fx.make_view(
                    fxh._as_ptr(p_weight, fx.Uint8)
                    + fx.Int64(expert) * (N * K)
                    + fx.Int64(oc) * (n_split * K),
                    fx.make_layout(n_split * K, 1),
                )
                wbuf = fx.rocdl.make_buffer_tensor(wview, False)
                wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
                bptr = iptr
                dma_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + wave * 1024
                if const_expr(SP == 1):
                    dma_offset = _pin_address(tid * 16)
                else:
                    # Sub-packet byte tid * 16 of a turn: N16 group (tid * 16) // piece.
                    dma_offset = _pin_address(
                        tid * 16 // piece * (16 * K) + tid * 16 % piece
                    )

                def packet_index(q):
                    # Start each task at a worker-dependent N128 pair to spread B reads.
                    rotation = worker // 8
                    return ((fx.Int32(q) // 2 + rotation) % pair_count) * 2 + fx.Int32(q) % 2

                def dma_turn(q, sub, phase, turn):
                    dst = llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<3>"),
                        (dma_base + phase * sub_bytes + turn * _TURN_BYTES).ir_value(),
                    )
                    source = sub * piece + turn * _TURN_BYTES // piece * (16 * K)
                    rocdl.raw_ptr_buffer_load_async_lds(
                        wrsrc,
                        dst,
                        fx.Int32(16).ir_value(),
                        dma_offset.ir_value(),
                        (packet_index(q) * packet_bytes + source).ir_value(),
                        zero,
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, _WEIGHT_AUX),
                    )

                for s in range_constexpr(min(lookahead, subs)):
                    for turn in range_constexpr(sub_turns):
                        dma_turn(s // SP, s % SP, s % slots, turn)
                    rocdl.asyncmark()
                blds = fxh.LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
                flds = fxh.LdsTensor(fx.make_view(fptr, fx.make_layout(4, 1)))
                rlds = fxh.LdsTensor(fx.make_view(fptr, fx.make_layout(1, 1)))
                abuf = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(
                            fx.PointerType.get(fx.Int32.ir_type, p_input.memspace, 16),
                            p_input,
                        ),
                        fx.make_layout(rows * (K // 4), 1),
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
                        + fx.Int64(block_m) * (PACKED_ROWS * N * 2),
                        fx.make_layout(PACKED_ROWS * N * 2, 1),
                    ),
                    False,
                )
                orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
                a = fx.make_rmem_tensor([8, 2, KB], fx.Int32)
                awords = fx.make_view(
                    fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, KB], 0)
                )
                factors = []
                out_addresses = []
                for mi in range_constexpr(2):
                    row = wave * 32 + mi * 16 + lr
                    encoded = ids[row].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < TOPK)
                    source = fx.Int32(token) * TOPK + fx.Int32(slot)
                    for kb in range_constexpr(KB):
                        for part in range_constexpr(2):
                            offset = valid.select(
                                source * K + lk * 16 + kb * 128 + part * 64, fx.Int32(-1)
                            )
                            awords[None, part, mi, kb].store(
                                apacket.load(voffset_bytes=offset, soffset_bytes=fx.Int32(0))
                            )
                    factors.append(
                        fx.Float32(
                            rlds._at(
                                fx.Int32(fx.ptrtoint(fptr + rows_offset)) + row * 4, 0
                            ).load()[0]
                        )
                    )
                    for parity in range_constexpr(2):
                        target = wave * 32 + mi * 16 + lr // 2 * 2 + parity
                        encoded = ids[target].bitcast(fx.Uint32)
                        token, slot = encoded & 0xFFFFFF, encoded >> 24
                        valid = (token < fx.Uint32(tokens)) & (slot < TOPK)
                        column = ((lk & 1) * 2 + (lk >> 1)) * 8 + lr % 2 * 32
                        offset = oc * (PACKED_ROWS * n_split * 2) + target * 128 + column * 2
                        out_addresses.append(
                            _pin_address(valid.select(offset, fx.Int32(PACKED_ROWS * N * 2)))
                        )
                b_addresses = [
                    _pin_address(
                        fx.Int32(fx.ptrtoint(bptr))
                        + lr * 16
                        + lk * 256
                        + base * phases_per_base * sub_bytes
                    )
                    for base in range_constexpr((slots + phases_per_base - 1) // phases_per_base)
                ]
                ws_base = _scalar(fx.Int32(fx.ptrtoint(fptr + wscale_offset))) + lk * 16
                atom = fx.make_mma_atom(rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))

                def store_one(packed, q, record):
                    rocdl.raw_ptr_buffer_store(
                        packed[record].ir_value(),
                        orsrc,
                        out_addresses[record].ir_value(),
                        (packet_index(q) * (PACKED_ROWS * 128)).ir_value(),
                        aux=ir.IntegerAttr.get(fx.Int32.ir_type, _OUTPUT_AUX),
                    )

                def store(packed, q):
                    for record in range_constexpr(4):
                        store_one(packed, q, record)

                def load_wscales(q):
                    base = ws_base + packet_index(q) * 256
                    return [flds._at(base, ni * 64).load() for ni in range_constexpr(4)]

                def memory(q, sub, phase, left=None):
                    """B of sub-packet (q, sub) to VGPRs; left = sub-packets from it to the end
                    when static (prologue/tail), else the steady state."""
                    rocdl.s_setprio(3)
                    rocdl.wait_asyncmark(
                        lookahead - 1 if left is None else min(lookahead, left) - 1
                    )
                    _stage_end()
                    issue = left is None or left > lookahead
                    target_q = q + (sub + lookahead) // SP
                    target_sub = (sub + lookahead) % SP
                    target_phase = (phase + lookahead) % slots
                    b = fx.make_rmem_tensor([8, 4, KU], fx.Int32)
                    words = fx.make_view(
                        fx.get_iter(b), fx.make_ordered_layout([4, 2, 4, KU], 0)
                    )
                    base = b_addresses[phase // phases_per_base]
                    phase_offset = phase % phases_per_base * sub_bytes
                    for kb in range_constexpr(KU):
                        for ni in range_constexpr(4):
                            for part in range_constexpr(2):
                                read = kb * 8 + ni * 2 + part
                                if const_expr(issue and read % 4 == 0):
                                    dma_turn(target_q, target_sub, target_phase, read // 4)
                                words[None, part, ni, kb].store(
                                    blds._at(
                                        base,
                                        phase_offset + ni * piece + kb * 2048 + part * 1024,
                                    ).load()
                                )
                    if const_expr(issue):
                        rocdl.asyncmark()
                    # Packet q's weight scales feed its epilogue in the next packet.
                    ws = load_wscales(q) if const_expr(sub == SP - 1) else None
                    rocdl.s_waitcnt(lgkmcnt=0)
                    for kb in range_constexpr(KU):
                        for ni in range_constexpr(4):
                            for part in range_constexpr(2):
                                view = words[None, part, ni, kb]
                                view.store(_pin_packet(view.load()))
                    return b, None if ws is None else [_pin_packet(w) for w in ws]

                def compute_start(previous, prev_ws, first):
                    """Zero packet accumulators; queue the previous packet's epilogue ops."""
                    accum = [fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(8)]
                    for acc in accum:
                        acc.fill(0)
                    ops = []
                    if const_expr(not first):
                        for pair_id in range_constexpr(4):
                            mi, pair = pair_id // 2, pair_id % 2
                            for op in range_constexpr(8):
                                ops.append(("scale", pair_id, mi, pair * 2 + op // 4, op % 4))
                            for wi in range_constexpr(2):
                                ops.append(("bf16", pair_id, mi, wi, 0))
                            ops.append(("swap", pair_id, mi, 0, 0))
                    return dict(
                        accum=accum,
                        previous=previous,
                        prev_ws=prev_ws,
                        first=first,
                        ops=ops,
                        cursor=0,
                        per_step=(len(ops) + 8 * KB - 1) // (8 * KB),
                        scaled=[[] for _ in range_constexpr(4)],
                        words=[[] for _ in range_constexpr(4)],
                        packed=[[], []],
                    )

                def run_op(st, item):
                    kind, pair_id, mi, x, y = item
                    if const_expr(kind == "scale"):
                        ni, element = x, y
                        value = st["previous"][mi * 4 + ni][element] * st["prev_ws"][ni][element]
                        st["scaled"][pair_id].append(value * factors[mi])
                    elif const_expr(kind == "bf16"):
                        # Words 0/1 hold ni even, 2/3 ni odd of this pair.
                        st["words"][pair_id].extend(
                            _bf16_words(st["scaled"][pair_id][4 * x : 4 * x + 4])
                        )
                    else:
                        w = st["words"][pair_id]
                        st["packed"][mi].append(_swap_pack([w[0], w[1]], [w[2], w[3]]))

                def compute_steps(st, b, q, old_words, kb0):
                    """MFMAs of K128 steps kb0..kb0+KU-1, interleaved with queued epilogue
                    ops and packet q-2's stores."""
                    rocdl.s_setprio(0)
                    steps = 8 * KB
                    accum = st["accum"]
                    for kb_local in range_constexpr(KU):
                        kb = kb0 + kb_local
                        for local in range_constexpr(8):
                            index = kb * 8 + local
                            if const_expr(index % (steps // 4) == steps // 8 and (not isinstance(q, int) or q >= 2)):
                                store_one(old_words, q - 2, index // (steps // 4))
                            mi, ni = local // 4, local % 4
                            fx.gemm(atom, accum[mi * 4 + ni], b[None, ni, kb_local], a[None, mi, kb], accum[mi * 4 + ni])
                            rocdl.sched_barrier(0)
                            for _ in range_constexpr(st["per_step"]):
                                if const_expr(st["cursor"] < len(st["ops"])):
                                    run_op(st, st["ops"][st["cursor"]])
                                    st["cursor"] += 1
                            rocdl.sched_barrier(0)

                def compute_finish(st):
                    while const_expr(st["cursor"] < len(st["ops"])):
                        run_op(st, st["ops"][st["cursor"]])
                        st["cursor"] += 1
                    result = [_pin_packet(st["accum"][i].load()) for i in range_constexpr(8)]
                    output_words = []
                    if const_expr(not st["first"]):
                        for mi in range_constexpr(2):
                            output_words.extend(
                                _coalesce_output_pairs(st["packed"][mi][0], st["packed"][mi][1])
                            )
                    return result, output_words

                def run_packet(q, base_phase, current, cur_ws, packed, left=None):
                    """One N64 packet: SP sub-packets of B; returns (accumulators, the previous
                    packet's output words, this packet's weight scales)."""
                    st = compute_start(current, cur_ws, first=isinstance(q, int) and q == 0)
                    ws = None
                    for sub in range_constexpr(SP):
                        b, sub_ws = memory(
                            q, sub, (base_phase + sub) % slots, None if left is None else left - sub
                        )
                        if const_expr(sub_ws is not None):
                            ws = sub_ws
                        compute_steps(st, b, q, packed, sub * KU)
                    current, packed = compute_finish(st)
                    return current, packed, ws

                rocdl.sched_barrier(0)
                current = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
                cur_ws = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                packed = [fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(4)]
                # The loop carries packed words and stores packet q - 2, so it starts at
                # q >= 2, on a ring-cycle boundary.
                prologue = min(packets, -(-2 // body) * body)
                for q in range_constexpr(prologue):
                    current, packed, cur_ws = run_packet(
                        q, q * SP % slots, current, cur_ws, packed, left=subs - q * SP
                    )
                # Packets before steady_end issue every sub-packet's lookahead DMA.
                steady_end = (subs - lookahead) // SP
                loop_end = prologue + max(0, (steady_end - prologue) // body * body)
                if const_expr(packets > prologue):
                    if const_expr(loop_end > prologue):
                        for q, state in range(
                            fx.Int32(prologue),
                            fx.Int32(loop_end),
                            fx.Int32(body),
                            init=[*current, *packed, *cur_ws],
                        ):
                            current = [fx.Vector(state[i]) for i in range_constexpr(8)]
                            packed = [fx.Vector(state[8 + i]) for i in range_constexpr(4)]
                            cur_ws = [fx.Vector(state[12 + i]) for i in range_constexpr(4)]
                            for p in range_constexpr(body):
                                current, packed, cur_ws = run_packet(
                                    q + p, p * SP % slots, current, cur_ws, packed
                                )
                            final = yield [*current, *packed, *cur_ws]
                        current = [fx.Vector(final[i]) for i in range_constexpr(8)]
                        packed = [fx.Vector(final[8 + i]) for i in range_constexpr(4)]
                        cur_ws = [fx.Vector(final[12 + i]) for i in range_constexpr(4)]
                    for q in range_constexpr(loop_end, packets):
                        current, packed, cur_ws = run_packet(
                            q, q * SP % slots, current, cur_ws, packed, left=subs - q * SP
                        )
                if const_expr(packets >= 2):
                    store(packed, packets - 2)
                final_packed = []
                for mi in range_constexpr(2):
                    pieces = []
                    for pair in range_constexpr(2):
                        values = []
                        for ni in (2 * pair, 2 * pair + 1):
                            for element in range_constexpr(4):
                                value = current[mi * 4 + ni][element] * cur_ws[ni][element]
                                values.append(value * factors[mi])
                        w = _bf16_words(values[:4]) + _bf16_words(values[4:])
                        pieces.append(_swap_pack([w[0], w[1]], [w[2], w[3]]))
                    final_packed.extend(_coalesce_output_pairs(pieces[0], pieces[1]))
                store(final_packed, packets - 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            # The next task reuses the LDS arena.
            fx.barrier()
            # Continue with the next half while it starts with a route.
            more = half + 1 < halves
            if const_expr(halves > 1):
                nxt = _scalar(
                    sorted_ids[more.select(row_begin + PACKED_ROWS, row_begin)]
                ).bitcast(fx.Uint32)
                more = more & ((nxt & 0xFFFFFF) < fx.Uint32(tokens)) & ((nxt >> 24) < TOPK)
            if const_expr(persistent):
                if (tid == 0) & (half == 0):
                    task_slot[0] = next_rank
                fx.barrier()
                step = more.select(step + 1, _scalar(task_slot[0]) * halves)
            else:
                # A one-shot CTA's next rank is past every task.
                step = more.select(step + 1, sort_tasks * halves)
        if const_expr(persistent):
            if tid == 0:
                # Every worker of a shard makes one failing claim; the last proves
                # the queue is drained and resets it for the next launch.
                if task_slot[0] == (sort_tasks - shard + 7) // 8 + num_workers // 8 - 1:
                    counter[shard * 32] = fx.Int32(0)

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
                num_workers
                if const_expr(persistent)
                else task_num * (blocks_per_sort // halves) * oc_splits,
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def make_packed_sum(*, N, TOPK):
    """Sum each token's TOPK packed routes; inverse holds sorted positions per route."""
    assert N % 512 == 0 and 0 < TOPK <= 255
    column_blocks = (N + 2047) // 2048

    @flyc.kernel(known_block_size=[256, 1, 1])
    def moe_packed_sum(
        p_output: fx.Pointer,
        p_source: fx.Pointer,
        p_inverse: fx.Pointer,
        tokens: fx.Int32,
        source_rows: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        token = fx.Int32(fx.block_idx.x) // column_blocks
        block_col = fx.Int32(fx.block_idx.x) % column_blocks
        inverse = fx.recast_iter(fx.Int32, fxh._as_ptr(p_inverse))
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
        locations = [_scalar(inverse[token * TOPK + route]) for route in range_constexpr(TOPK)]

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

        column = block_col * 2048 + tid * 8
        fragments = [read_route(locations[route], column) for route in range_constexpr(TOPK)]
        accum = fragments[0].to(fx.Float32)
        for route in range_constexpr(1, TOPK):
            accum = accum + fragments[route].to(fx.Float32)
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
        tokens: fx.Int32,
        source_rows: fx.Int32,
        stream: fx.Stream,
    ):
        moe_packed_sum(p_output, p_source, p_inverse, tokens, source_rows).launch(
            grid=(tokens * column_blocks, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch
