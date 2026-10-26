# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023
# Helpers run synchronously during FlyDSL tracing, capturing loop SSA by design.
# Binding default arguments changes DSL state.
"""gfx950 M128/4-wave FP8 Down with configurable K and OC splits.

Consume native sort128 or sort256 and write packed BF16 M128/N64 tiles.
K256 retains its PF3/paired pipeline; larger positive multiples of 256 stream
K256 chunks through a fixed double buffer. Persistent queues are caller-owned.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from . import common as fxh
from .common import (
    _ptr,
    asm_mark,
    get_device_cache_key,
    pin_vgpr,
    llvm_extract_as_vector,
)
from .gemm2_common import LdsTensor, BufferTensor
from .a8w8_blockscale_config import m128_down_lds_bytes

M128_QUEUE_SIZE = 256  # Eight heads, spaced by 32 int32 elements; not OC splits.
M128_CONTINUOUS_VMEM_CONFIG = {
    "block_m": 128,
    "block_n": 128,
    "sort_block_m": 128,
    "num_waves": 4,
    "num_oc_splits": 8,
    "xcd_count": 2,
    "weight_cache_policy": 16,
    "output_cache_policy": 18,
    "output_layout": "packed",
    "packed_rows": 128,
}


def _scalar(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, value.ir_value()))


def _coalesce_output_pairs(first, second):
    """Exchange a row bit with a column bit for contiguous 128-byte stores."""
    mask, even = fx.Uint64(0xAAAAAAAAAAAAAAAA), fx.Uint64(0x5555555555555555)
    assembly = ["s_mov_b64 vcc, $16"]
    assembly.extend(
        f"v_cndmask_b32_dpp ${i}, ${12+i}, ${8+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        for i in range(4)
    )
    assembly.append("s_mov_b64 vcc, $17")
    assembly.extend(
        f"v_cndmask_b32_dpp ${4+i}, ${8+i}, ${12+i}, vcc quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
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
        llvm_extract_as_vector(fx.Int32, result, *([side * 4 + i] for i in range(4)))
        for side in range(2)
    ]


def _stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _pack_pair(c0, c1, route):
    words0 = (c0 * route).to(fx.BFloat16).bitcast(fx.Int32)
    words1 = (c1 * route).to(fx.BFloat16).bitcast(fx.Int32)
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    lo = rocdl.permlane16_swap(
        pair_type, words0[0].ir_value(), words1[0].ir_value(), False, False
    )
    hi = rocdl.permlane16_swap(
        pair_type, words0[1].ir_value(), words1[1].ir_value(), False, False
    )
    return llvm_extract_as_vector(fx.Int32, lo, [0], hi, [0], lo, [1], hi, [1])


def _pack_output_steps(previous, cached_rows, packed):
    """Python trace-time generator; each next() emits one pack step, not GPU yield."""
    for mi in range(2):
        for pair in range(2):
            scaled = []
            for ni in range(2):
                for element in range(4):
                    scaled.append(
                        previous[mi * 4 + pair * 2 + ni][element] * cached_rows[mi, 2]
                    )
                    yield

            words = []
            for word in range(4):
                words.append(
                    fx.Vector.from_elements(scaled[word * 2 : word * 2 + 2], fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                yield

            swaps = []
            for word in range(2):
                swaps.append(
                    rocdl.permlane16_swap(
                        ir.Type.parse("!llvm.struct<(i32, i32)>"),
                        words[word].ir_value(),
                        words[word + 2].ir_value(),
                        False,
                        False,
                    )
                )
                if word == 1:
                    lo, hi = swaps
                    packed[mi].append(
                        llvm_extract_as_vector(
                            fx.Int32, lo, [0], hi, [0], lo, [1], hi, [1]
                        )
                    )
                yield


@cache
def _compile_m128_down(
    device_key, n, topk, num_experts, persistent, is_sorted_bm_256, num_oc_splits
):
    if device_key is None or device_key[2].split(":")[0] != "gfx950":
        raise ValueError("M128 blockscale Down requires gfx950")
    m_blocks_per_sort = 2 if is_sorted_bm_256 else 1
    width = 4 if persistent else 2
    n_split = n // num_oc_splits
    packets = n_split // 64
    scale_count = n_split // 128 * 2
    scale_words = (scale_count + 63 + 255) // 256 * 256
    ids_offset = scale_words + 4 * 4096
    rows_offset = ids_offset + 128
    alive_offset = rows_offset + 128 * 3
    task_offset = alive_offset + 2
    arena_words = task_offset + int(persistent)
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
    def moe_down_m128_kernel(
        output: fx.Pointer,
        input_q: fx.Pointer,
        weight: fx.Pointer,
        input_scales: fx.Pointer,
        weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        expert_ids: fx.Pointer,
        valid_ids: fx.Pointer,
        counter: fx.Pointer,
        tokens: fx.Int32,
        capacity_blocks: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = _scalar(tid // 64)
        worker = fx.Int32(fx.block_idx.x)
        virtual = worker
        active_tasks = (_scalar(valid_ids[0]) // 128) * num_oc_splits
        lds = fx.SharedAllocator().allocate(Storage).peek()
        task_slot = fx.make_view(lds.arena.ptr + task_offset, fx.make_layout(1, 1))

        def take_task(slot):
            # Keep one claim site in the while header. Shards are independent
            # of the number of output-channel splits.
            if tid == 0:
                head = counter + worker % 8 * 32
                ptr = llvm.inttoptr(
                    ir.Type.parse("!llvm.ptr<1>"), fx.ptrtoint(head).ir_value()
                )
                slot[0] = fx.Int32(
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        ptr,
                        fx.Int32(1).ir_value(),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                    ).res
                )
            fx.barrier()
            rank = _scalar(slot[0])
            return rank * 8 + worker % 8

        while (
            take_task(task_slot) if const_expr(persistent) else virtual
        ) < active_tasks:
            task = (
                _scalar(task_slot[0]) * 8 + worker % 8
                if const_expr(persistent)
                else virtual
            )
            chunk = active_tasks // width
            mapped = task % width * chunk + task // width
            task = (task < chunk * width).select(mapped, task)
            block_m, oc = task // num_oc_splits, task % num_oc_splits
            row_begin = block_m * 128
            tile_has_rows = fx.Boolean(True)
            if const_expr(is_sorted_bm_256):
                # Native sorting has a valid prefix followed only by padding.
                # All waves read the same first ID; test before metadata/DMA.
                first = _scalar(sorted_ids[row_begin]).bitcast(fx.Uint32)
                tile_has_rows = ((first & 0xFFFFFF) < fx.Uint32(tokens)) & (
                    (first >> 24) < topk
                )
            if tile_has_rows:
                expert = _scalar(expert_ids[block_m // m_blocks_per_sort])
                rows = tokens * topk
                lane = tid % 64
                lr, lk = lane % 16, lane // 16
                bptr = lds.arena.ptr + scale_words
                fptr = fx.recast_iter(
                    fx.PointerType.get(fx.Float32.ir_type, lds.arena.ptr.memspace, 16),
                    lds.arena.ptr,
                )
                ids = fx.make_view(lds.arena.ptr + ids_offset, fx.make_layout(128, 1))
                row_scales = fx.make_view(fptr + rows_offset, fx.make_layout(384, 1))
                bscales = fx.make_view(fptr, fx.make_layout(scale_count, 1))
                alive = fx.make_view(lds.arena.ptr + alive_offset, fx.make_layout(2, 1))
                asbuf = fx.rocdl.make_buffer_tensor(
                    fx.make_view(input_scales, fx.make_layout(rows * 2, 1)), False
                )
                asrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(asbuf))
                zero = fx.Int32(0).ir_value()
                if tid < 128:
                    encoded = sorted_ids[row_begin + tid].bitcast(fx.Uint32)
                    token, slot = encoded & 0xFFFFFF, encoded >> 24
                    valid = (token < fx.Uint32(tokens)) & (slot < topk)
                    ids[tid] = encoded.bitcast(fx.Int32)
                    row_scales[256 + tid] = sorted_weights[row_begin + tid]
                    source = fx.Int32(token) * topk + fx.Int32(slot)
                    for kb in range_constexpr(2):
                        offset = valid.select((kb * rows + source) * 4, fx.Int32(-1))
                        row_scales[kb * 128 + tid] = fx.Float32(
                            rocdl.raw_ptr_buffer_load(
                                fx.Float32.ir_type,
                                asrsrc,
                                offset.ir_value(),
                                zero,
                                aux=fxh.rocdl_aux(0),
                            )
                        )
                    if const_expr(not persistent and not is_sorted_bm_256):
                        mask = fx.Uint64(
                            rocdl.ballot(fx.Uint64.ir_type, valid.ir_value())
                        )
                        if lane == 0:
                            alive[wave] = (mask != fx.Uint64(0)).select(
                                fx.Int32(1), fx.Int32(0)
                            )
                for turn in range_constexpr((scale_count + 255) // 256):
                    index = tid + turn * 256
                    if index < scale_count:
                        bscales[index] = weight_scales[
                            expert * (n // 64) + oc * scale_count + index
                        ]
                fx.barrier()
                has_rows = (
                    fx.Boolean(True)
                    if const_expr(persistent or is_sorted_bm_256)
                    else _scalar(alive[0] | alive[1]) != 0
                )
                if has_rows:
                    wview = fx.make_view(
                        weight
                        + fx.Int64(expert) * (n * 256)
                        + fx.Int64(oc) * (n_split * 256),
                        fx.make_layout(n_split * 256, 1),
                    )
                    wbuf = fx.rocdl.make_buffer_tensor(wview, False)
                    wrsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(wbuf))
                    dma_base = _scalar(fx.Int32(fx.ptrtoint(bptr))) + wave * 1024
                    dma_offset = pin_vgpr(tid * 16)
                    phase_worker = worker
                    if const_expr(persistent):
                        # Re-materialize invariant worker phase per task instead
                        # of keeping all derived N offsets live across the queue.
                        phase_worker = fx.Int32(
                            llvm.inline_asm(
                                fx.Int32.ir_type,
                                [worker.ir_value()],
                                "",
                                "=s,0",
                                has_side_effects=True,
                            )
                        )

                    def packet_index(q):
                        # Resident-worker phases retain B locality without making
                        # all consumers restart each task at the same N phase.
                        rotation = phase_worker // 8
                        return (
                            (fx.Int32(q) // 2 + rotation) % (packets // 2)
                        ) * 2 + fx.Int32(q) % 2

                    def dma_turn(q, phase, turn):
                        dst = llvm.inttoptr(
                            ir.Type.parse("!llvm.ptr<3>"),
                            (dma_base + phase * 16384 + turn * 4096).ir_value(),
                        )
                        rocdl.raw_ptr_buffer_load_async_lds(
                            wrsrc,
                            dst,
                            fx.Int32(16).ir_value(),
                            dma_offset.ir_value(),
                            (packet_index(q) * 16384 + turn * 4096).ir_value(),
                            zero,
                            aux=fxh.rocdl_aux(16),
                        )

                    for q in range_constexpr(min(4 if persistent else 3, packets)):
                        for turn in range_constexpr(4):
                            dma_turn(q, q, turn)
                        rocdl.asyncmark()
                    blds = LdsTensor(fx.make_view(bptr, fx.make_layout(4, 1)))
                    rlds = LdsTensor(
                        fx.make_view(fptr + rows_offset, fx.make_layout(1, 1))
                    )
                    aptr = fx.recast_iter(
                        fx.PointerType.get(fx.Int32.ir_type, input_q.memspace, 16),
                        input_q,
                    )
                    abuf = fx.rocdl.make_buffer_tensor(
                        fx.make_view(aptr, fx.make_layout(rows * 64, 1)), False
                    )
                    apacket = BufferTensor(
                        fx.make_view(fx.get_iter(abuf), fx.make_layout(4, 1))
                    )
                    out_rows = capacity_blocks * 128
                    obuf = fx.rocdl.make_buffer_tensor(
                        fx.make_view(output, fx.make_layout(out_rows * n, 1)), False
                    )
                    orsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(obuf))
                    a = fx.make_rmem_tensor([8, 2, 2], fx.Int32)
                    awords = fx.make_view(
                        fx.get_iter(a), fx.make_ordered_layout([4, 2, 2, 2], 0)
                    )
                    cached_rows = fx.make_rmem_tensor([2, 3], fx.Float32)
                    out_addresses = []
                    for mi in range_constexpr(2):
                        row = wave * 32 + mi * 16 + lr
                        encoded = ids[row].bitcast(fx.Uint32)
                        token, slot = encoded & 0xFFFFFF, encoded >> 24
                        valid = (token < fx.Uint32(tokens)) & (slot < topk)
                        source = fx.Int32(token) * topk + fx.Int32(slot)
                        for kb in range_constexpr(2):
                            for part in range_constexpr(2):
                                offset = valid.select(
                                    source * 256 + lk * 16 + kb * 128 + part * 64,
                                    fx.Int32(-1),
                                )
                                awords[None, part, mi, kb].store(
                                    apacket.load(voffset_bytes=offset)
                                )
                        for kb in range_constexpr(3):
                            address = (
                                fx.Int32(fx.ptrtoint(fptr + rows_offset))
                                + (kb * 128 + row) * 4
                            )
                            cached_rows[mi, kb] = rlds.load(address_bytes=address)[0]
                        if const_expr(persistent):
                            row_valid = valid.select(fx.Int32(1), fx.Int32(0))
                        for parity in range_constexpr(2):
                            target = wave * 32 + mi * 16 + lr // 2 * 2 + parity
                            if const_expr(persistent):
                                # Packed stores exchange a row bit; derive the
                                # exact target validity without rereading IDs.
                                mask = fx.Int32(
                                    rocdl.update_dpp(
                                        fx.Int32.ir_type,
                                        # Full masks and in-quad sources never use old.
                                        llvm.mlir_undef(fx.Int32.ir_type),
                                        row_valid.ir_value(),
                                        dpp_ctrl=0xA0 + parity * 0x55,
                                        row_mask=15,
                                        bank_mask=15,
                                        bound_ctrl=False,
                                    )
                                )
                                valid = mask != 0
                            else:
                                encoded = ids[target].bitcast(fx.Uint32)
                                token, slot = encoded & 0xFFFFFF, encoded >> 24
                                valid = (token < fx.Uint32(tokens)) & (slot < topk)
                            column = ((lk & 1) * 2 + (lk >> 1)) * 8 + lr % 2 * 32
                            offset = (
                                block_m * (128 * n * 2)
                                + oc * (128 * n_split * 2)
                                + target * 128
                                + column * 2
                            )
                            out_addresses.append(
                                pin_vgpr(valid.select(offset, out_rows * n * 2))
                            )
                    b_addresses = [
                        pin_vgpr(
                            fx.Int32(fx.ptrtoint(bptr))
                            + lr * 16
                            + lk * 256
                            + phase * 16384
                        )
                        for phase in range_constexpr(4)
                    ]
                    atom = fx.make_mma_atom(
                        rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
                    )

                    def store_one(packed, q, record):
                        rocdl.raw_ptr_buffer_store(
                            packed[record].ir_value(),
                            orsrc,
                            out_addresses[record].ir_value(),
                            (packet_index(q) * 16384).ir_value(),
                            aux=fxh.rocdl_aux(18),
                        )

                    def store(packed, q):
                        for record in range_constexpr(4):
                            store_one(packed, q, record)

                    pair_scales = fx.make_rmem_tensor(2, fx.Int32)

                    def memory(q, phase, tail_left=4):
                        refill_pair = (
                            persistent
                            and phase % 2 == 0
                            and tail_left > 2
                            and (not isinstance(q, int) or q != 0)
                        )
                        rocdl.s_setprio(3)
                        if const_expr(not persistent or phase % 2 == 0):
                            # Persistent publishes a pair and retires prior-pair
                            # readers before reusing their slots; PF3 publishes one.
                            wait = (
                                (
                                    min(2, packets - 2)
                                    if isinstance(q, int) and q == 0
                                    else 0
                                )
                                if persistent
                                else min(2, tail_left - 1)
                            )
                            rocdl.wait_asyncmark(wait)
                            _stage_end()
                        asm_mark(f"MOE128_MEMORY_BEGIN_{phase}")
                        b = fx.make_rmem_tensor([8, 4, 2], fx.Int32)
                        words = fx.make_view(
                            fx.get_iter(b), fx.make_ordered_layout([4, 2, 4, 2], 0)
                        )
                        for kb in range_constexpr(2):
                            for ni in range_constexpr(4):
                                for part in range_constexpr(2):
                                    if const_expr(refill_pair and part == 0):
                                        turn_id = kb * 4 + ni
                                        dma_turn(
                                            q + 2 + turn_id // 4,
                                            (phase + 2 + turn_id // 4) % 4,
                                            turn_id % 4,
                                        )
                                    elif const_expr(
                                        not persistent
                                        and tail_left > 3
                                        and (kb * 8 + ni * 2 + part) % 4 == 0
                                    ):
                                        dma_turn(
                                            q + 3,
                                            (phase + 3) % 4,
                                            (kb * 8 + ni * 2 + part) // 4,
                                        )
                                    words[None, part, ni, kb].store(
                                        blds.load(
                                            address_bytes=b_addresses[phase],
                                            offset_bytes=ni * 4096
                                            + kb * 2048
                                            + part * 1024,
                                        )
                                    )
                        if const_expr(
                            refill_pair or (not persistent and tail_left > 3)
                        ):
                            rocdl.asyncmark()
                        for kb in range_constexpr(2):
                            if const_expr(not persistent or phase % 2 == 0):
                                pair_scales[kb] = bscales[
                                    packet_index(q) // 2 * 2 + kb
                                ].bitcast(fx.Int32)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        for kb in range_constexpr(2):
                            for ni in range_constexpr(4):
                                for part in range_constexpr(2):
                                    view = words[None, part, ni, kb]
                                    view.store(pin_vgpr(view.load()))
                        asm_mark(f"MOE128_MEMORY_END_{phase}")
                        return b, [pair_scales[kb] for kb in range_constexpr(2)]

                    def compute(b, scales, previous, phase, q, old_words, first=False):
                        rocdl.s_setprio(0)
                        asm_mark(f"MOE128_COMPUTE_BEGIN_{phase}")
                        factors = [
                            [
                                cached_rows[mi, kb]
                                * _scalar(scales[kb]).bitcast(fx.Float32)
                                for mi in range_constexpr(2)
                            ]
                            for kb in range_constexpr(2)
                        ]
                        accum = fx.make_rmem_tensor([4, 2, 4], fx.Float32)
                        packed = [[], []]
                        pack_steps = _pack_output_steps(previous, cached_rows, packed)
                        pending = []

                        def retire(item):
                            partial, mi, ni, kb = item
                            for element in range_constexpr(4):
                                value = partial.load()[element]
                                if const_expr(kb == 0):
                                    accum[element, mi, ni] = value * factors[kb][mi]
                                else:
                                    accum[element, mi, ni] = fx.Float32(
                                        llvm.intr_fma(
                                            value.ir_value(),
                                            factors[kb][mi].ir_value(),
                                            accum[element, mi, ni].ir_value(),
                                        )
                                    )

                        for kb in range_constexpr(2):
                            for mi in range_constexpr(2):
                                for ni in range_constexpr(4):
                                    if const_expr(
                                        ni == 2 and (not isinstance(q, int) or q >= 2)
                                    ):
                                        store_one(old_words, q - 2, kb * 2 + mi)
                                    partial = fx.make_rmem_tensor(4, fx.Float32)
                                    partial.fill(0)
                                    fx.gemm(
                                        atom,
                                        partial,
                                        b[None, ni, kb],
                                        a[None, mi, kb],
                                        partial,
                                    )
                                    rocdl.sched_barrier(0)
                                    pending.append((partial, mi, ni, kb))

                                    # Delay consumption by two MFMAs. Startup emits
                                    # 7 pack steps; steady state emits 4 retire + 3 pack.
                                    pack_count = 7
                                    if const_expr(len(pending) > 2):
                                        retire(pending.pop(0))
                                        pack_count = 3
                                    if const_expr(not first):
                                        for _ in range_constexpr(pack_count):
                                            next(pack_steps)
                                    rocdl.sched_barrier(0)

                        # Non-first packets issue all pack steps: 2*7 + 14*3 = 56.
                        # Only the last two MFMA partials remain to be consumed.
                        for item in pending:
                            retire(item)
                        result = [
                            pin_vgpr(accum[None, mi, ni].load())
                            for mi in range_constexpr(2)
                            for ni in range_constexpr(4)
                        ]
                        output_words = []
                        if const_expr(not first):
                            for mi in range_constexpr(2):
                                output_words.extend(
                                    _coalesce_output_pairs(packed[mi][0], packed[mi][1])
                                )
                        asm_mark(f"MOE128_COMPUTE_END_{phase}")
                        return result, output_words

                    rocdl.sched_barrier(0)
                    current = [
                        fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)
                    ]
                    packed = [
                        fx.Vector.filled(4, 0, fx.Int32) for _ in range_constexpr(4)
                    ]
                    prologue = min(4, packets)
                    for q in range_constexpr(prologue):
                        b, scales = memory(q, q % 4, tail_left=packets - q)
                        current, packed = compute(
                            b, scales, current, q % 4, q, packed, first=q == 0
                        )
                    loop_end = max(prologue, (packets - 3) // 4 * 4)
                    if const_expr(packets > prologue):
                        for q, state in range(
                            fx.Int32(prologue),
                            fx.Int32(loop_end),
                            fx.Int32(4),
                            init=[*current, *packed],
                        ):
                            current = [fx.Vector(state[i]) for i in range_constexpr(8)]
                            packed = [
                                fx.Vector(state[8 + i]) for i in range_constexpr(4)
                            ]
                            for phase in range_constexpr(4):
                                b, scales = memory(q + phase, phase)
                                current, packed = compute(
                                    b, scales, current, phase, q + phase, packed
                                )
                            final = yield [*current, *packed]
                        current = [fx.Vector(final[i]) for i in range_constexpr(8)]
                        packed = [fx.Vector(final[8 + i]) for i in range_constexpr(4)]
                        for q in range_constexpr(loop_end, packets):
                            b, scales = memory(q, q % 4, tail_left=packets - q)
                            current, packed = compute(
                                b, scales, current, q % 4, q, packed
                            )
                    if const_expr(persistent):
                        rocdl.s_setprio(3)
                    store(packed, packets - 2)
                    final_packed = []
                    for mi in range_constexpr(2):
                        pieces = [
                            _pack_pair(
                                current[mi * 4 + pair * 2],
                                current[mi * 4 + pair * 2 + 1],
                                cached_rows[mi, 2],
                            )
                            for pair in range_constexpr(2)
                        ]
                        final_packed.extend(
                            _coalesce_output_pairs(pieces[0], pieces[1])
                        )
                    store(final_packed, packets - 1)
                    rocdl.wait_asyncmark(0)
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            if const_expr(persistent):
                fx.barrier()
            else:
                virtual = active_tasks

        if const_expr(persistent):  # noqa: SIM102 - Compile-time/runtime split.
            if tid == 0:
                # Each shard has 64 workers, but its task count can differ by
                # one when OC splits do not make active_tasks divisible by 8.
                shard_tasks = (active_tasks + 7 - worker % 8) // 8
                if task_slot[0] == shard_tasks + 63:
                    counter[worker % 8 * 32] = fx.Int32(0)

    @flyc.jit
    def launch(
        output: fx.Pointer,
        input_q: fx.Pointer,
        weight: fx.Pointer,
        input_scales: fx.Pointer,
        weight_scales: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        expert_ids: fx.Pointer,
        valid_ids: fx.Pointer,
        counter: fx.Pointer,
        tokens: fx.Int32,
        capacity_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        moe_down_m128_kernel(
            output,
            input_q,
            weight,
            input_scales,
            weight_scales,
            sorted_ids,
            sorted_weights,
            expert_ids,
            valid_ids,
            counter,
            tokens,
            capacity_blocks,
            value_attrs=attrs,
        ).launch(
            grid=(
                512 if const_expr(persistent) else capacity_blocks * num_oc_splits,
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


@cache
def make_m128_down(
    *,
    n,
    k=256,
    topk,
    num_experts,
    persistent=False,
    is_sorted_bm_256=False,
    num_oc_splits=8,
):
    """Build a stateless ten-tensor Down callable; cache code, not queue heads.

    num_oc_splits partitions N, not K. Each split must contain whole N128
    scale blocks (at least two N64 packets). Output is always packed by M128.
    K is any positive multiple of 256 within the buffer-address and LDS bounds. K>256
    streams fixed K256 chunks; scales are A[K/128,rows] and B[E,N/128,K/128].
    With sort256, expert_ids[m//2] serves both halves and a first-ID check skips
    empty tiles; arbitrary holes in the native valid prefix are not supported.

    Persistent callers pass 256 zero-initialized int32 words as counter. The
    eight heads self-reset before completion. Never share this storage across
    concurrent launches; graph capture must retain its own queue allocation.
    Independent calls accept an unused one-element counter and preserve it.
    """
    assert k == 256
    lds_bytes = m128_down_lds_bytes(
        n=n,
        k=k,
        topk=topk,
        num_experts=num_experts,
        persistent=persistent,
        num_oc_splits=num_oc_splits,
    )
    sort_block_m = 256 if is_sorted_bm_256 else 128
    n_split = n // num_oc_splits

    def down(
        output,
        input_q,
        weight,
        input_scales,
        weight_scales,
        sorted_ids,
        sorted_weights,
        expert_ids,
        valid_ids,
        counter,
    ):
        tokens, actual_topk, actual_k = input_q.shape
        tensors = (
            output,
            input_q,
            weight,
            input_scales,
            weight_scales,
            sorted_ids,
            sorted_weights,
            expert_ids,
            valid_ids,
            counter,
        )
        assert 0 < tokens < (1 << 24) and (actual_topk, actual_k) == (topk, k)
        assert weight.shape == (num_experts, n, k) and output.shape == (
            expert_ids.numel() * sort_block_m,
            n,
        )
        assert (
            input_q.dtype == weight.dtype == torch.float8_e4m3fn
            and output.dtype == torch.bfloat16
        )
        assert (
            input_scales.dtype
            == weight_scales.dtype
            == sorted_weights.dtype
            == torch.float32
        )
        assert (
            sorted_ids.dtype
            == expert_ids.dtype
            == valid_ids.dtype
            == counter.dtype
            == torch.int32
        )
        assert input_scales.numel() == tokens * topk * (
            k // 128
        ) and weight_scales.shape == (num_experts, n // 128, k // 128)
        assert (
            sorted_ids.ndim == expert_ids.ndim == 1
            and sorted_ids.shape == sorted_weights.shape
        )
        assert (
            expert_ids.numel()
            == (sorted_ids.numel() + sort_block_m - 1) // sort_block_m
        ), f"native sort{sort_block_m} required"
        assert (
            counter.numel() == (M128_QUEUE_SIZE if persistent else 1)
            and valid_ids.numel() >= 1
        )
        assert all(
            t.is_cuda and t.is_contiguous() and t.device == output.device
            for t in tensors
        )
        # The invalid-row voffset plus the last packet's soffset must not wrap.
        assert (
            output.numel() + 128 * n_split
        ) * 2 < 1 << 32 and input_q.numel() < 1 << 32
        assert input_scales.numel() * 4 < 1 << 32 and n_split * k < 1 << 32
        capacity_blocks = expert_ids.numel() * (sort_block_m // 128)
        with torch.cuda.device(output.device):
            launcher = _compile_m128_down(
                get_device_cache_key(),
                n,
                topk,
                num_experts,
                persistent,
                is_sorted_bm_256,
                num_oc_splits,
            )
            stream = torch.cuda.current_stream(output.device)
            _run_compiled(
                launcher,
                *[_ptr(t) for t in tensors],
                fx.Int32(tokens),
                fx.Int32(capacity_blocks),
                fx.Stream(stream.cuda_stream),
            )
        return output

    down.config = {
        **M128_CONTINUOUS_VMEM_CONFIG,
        "persistent": persistent,
        "k": k,
        "num_oc_splits": num_oc_splits,
        "sort_block_m": sort_block_m,
        "is_sorted_bm_256": is_sorted_bm_256,
        "scheduler": "sharded" if persistent else "independent",
        "persistent_workgroups": 512 if persistent else 0,
        "xcd_count": 4 if persistent else 2,
        "lds_bytes": lds_bytes,
    }
    if persistent:
        down.config["queue_self_reset"] = True
    return down
