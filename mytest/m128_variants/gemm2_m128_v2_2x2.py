# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M128 four-wave Down for gfx942 FP8 on sort128 metadata.

A task is one sorted M128 block times N/OC output columns, swept in N64 packets.
The four waves form a 2x2 grid: each owns 64 rows x 32 columns of every packet,
keeps its A fragments in registers and loads only its half of the weight packet,
so the main loop needs no CTA barrier. The previous packet's scaling, BF16
packing, wave-private LDS shuffle and stores are interleaved between the current
packet's MFMAs. Tasks are assigned in contiguous ranges per XCD so an expert's
weights and each A block are fetched into one L2.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from . import common as fxh
from .common import DOWN_STORE_CACHE_MODIFIER, _f32_to_bf16, get_down_device_config

BLOCK_M = 128
WAVES = 4
PACKET_N = 64
M_TILES = 4  # M16 tiles per wave (64 rows)
N_TILES = 2  # N16 tiles per wave (32 columns)
WAVE_LDS = 64 * 64
# Byte offset that is out of bounds for every buffer descriptor built below.
OOB = 0x7FFFFFF0


def _aux(value):
    return ir.IntegerAttr.get(fx.Int32.ir_type, value)


def _rsrc(pointer, num_bytes):
    return fx.rocdl.get_buffer_rsrc(
        fx.rocdl.make_buffer_ptr(pointer, num_records_bytes=num_bytes)
    )


def _load(rsrc, voffset, soffset, words, aux=0):
    result_type = (
        fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type)
    )
    value = rocdl_dialect.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=_aux(aux),
    ).result
    return fx.Int32(value) if words == 1 else Vec(value)


def _store(rsrc, value, voffset, soffset, aux):
    rocdl_dialect.RawPtrBufferStoreOp(
        value.ir_value(),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=_aux(aux),
    )


def _lds_pointer(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_store(address, value, alignment):
    llvm.store(value.ir_value(), _lds_pointer(address), alignment=alignment)


def _lds_load(address, words):
    vector = ir.VectorType.get([words], fx.Int32.ir_type)
    return Vec(llvm.load(vector, _lds_pointer(address), alignment=4 * words))


def _mfma(weight, activation, acc):
    return Vec(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            ir.VectorType.get([4], fx.Float32.ir_type),
            [weight, activation, acc, 0, 0, 0],
        )
    )


def _halves(packet):
    """Split one 16 B K packet into the two i64 MFMA operands."""
    pair = Vec(packet).bitcast(fx.Int64)
    return pair[0], pair[1]


def _slot(row):
    """LDS row slot: swap row bits 0 and 3 so bit 3 selects the bank half."""
    return (row & 0b110110) | ((row & 1) << 3) | ((row >> 3) & 1)


def m128_oc_splits(down_path):
    """`m128` uses OC4; `m128_ocS` selects S output-column splits per M128 block."""
    if down_path == "m128":
        return 4
    prefix = "m128_oc"
    if not down_path.startswith(prefix) or not down_path[len(prefix) :].isdigit():
        raise ValueError(f"Unsupported M128 Down path: {down_path}")
    return int(down_path[len(prefix) :])


def _build_moe_gemm2_m128(
    *, N, K, weight_quant_type, act_quant_type, TOPK, padding, oc_splits
):
    assert K % 64 == 0 and 64 <= K <= 256, "M128 Down supports K in {64,...,256}"
    assert N % (oc_splits * PACKET_N) == 0, "N/OC must be a multiple of 64"
    assert padding in (0, 32, 64, 128)
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert act_quant_type in ("ptpc", "per_tensor")
    _, xcd_count = get_down_device_config()
    import os

    _DEBUG = os.environ.get("AITER_M128_DEBUG", "")
    _STORE_AUX = int(os.environ.get("AITER_M128_STORE_AUX", DOWN_STORE_CACHE_MODIFIER))
    DEPTH = int(os.environ.get("AITER_M128_DEPTH", "2"))
    KB = K // 64
    N_OC = N // oc_splits
    PACKETS = N_OC // PACKET_N
    STRIDE = (N + padding // 2) * 2
    SCALE_WORDS = N_OC if weight_quant_type == "ptpc" else 0
    assert BLOCK_M * STRIDE < OOB

    @flyc.kernel(known_block_size=[WAVES * 64, 1, 1])
    def moe_2stage_down_m128(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // 64).ir_value()))
        wave_m, wave_n = wave // 2, wave % 2
        lr, lq = lane % 16, lane // 16

        valid_rows = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        tasks = ((fx.Uint32(valid_rows) + (BLOCK_M - 1)) // BLOCK_M) * oc_splits
        wg = fx.Uint32(gpu.block_idx.x)
        chunk = tasks // xcd_count
        mapped = (wg % xcd_count) * chunk + wg // xcd_count
        task = fx.Uint32(arith.select(wg < chunk * xcd_count, mapped, wg))
        if task < tasks:
            m_block = fx.Int32(task // oc_splits)
            n_begin = fx.Int32(task % oc_splits) * N_OC
            row_begin = m_block * BLOCK_M
            expert = fx.Int32(p_sorted_expert_ids[m_block])
            if const_expr("expert0" in _DEBUG):
                expert = fx.Int32(0)

            storage = fx.SharedAllocator().allocate(
                fx.Array[fx.Int32, SCALE_WORDS + WAVES * WAVE_LDS // 4, 16]
            )
            lds_view = storage.peek().view(
                fx.make_layout(SCALE_WORDS + WAVES * WAVE_LDS // 4, 1)
            )
            lds_scales = fx.Int32(fx.ptrtoint(fx.get_iter(lds_view)))
            lds_base = lds_scales + SCALE_WORDS * 4 + wave * WAVE_LDS

            ids_rsrc = _rsrc(
                fxh._as_ptr(p_sorted_ids, fx.Int32) + fx.Int64(row_begin), BLOCK_M * 4
            )
            routing_rsrc = _rsrc(
                fxh._as_ptr(p_sorted_weights, fx.Float32) + fx.Int64(row_begin),
                BLOCK_M * 4,
            )
            routes = M * TOPK
            input_rsrc = _rsrc(fxh._as_ptr(p_input, fx.Uint8), fx.Int64(routes) * K)
            weight_rsrc = _rsrc(
                fxh._as_ptr(p_weight, fx.Uint8) + fx.Int64(expert) * (N * K), N * K
            )
            output_rsrc = _rsrc(
                fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(row_begin) * (STRIDE // 2),
                BLOCK_M * STRIDE,
            )

            if const_expr(weight_quant_type == "ptpc"):
                # Stage this task's N/OC weight scales once; the packets read them from LDS.
                scale_rsrc = _rsrc(
                    fxh._as_ptr(p_w_scale, fx.Float32)
                    + fx.Int64(expert) * N
                    + fx.Int64(n_begin),
                    N_OC * 4,
                )
                for turn in range_constexpr((N_OC // 4 + 255) // 256):
                    index = tid + turn * 256
                    if index < N_OC // 4:
                        _lds_store(
                            lds_scales + index * 16,
                            _load(scale_rsrc, index * 16, 0, 4),
                            16,
                        )
                gpu.barrier()

            def decode(row):
                encoded = _load(ids_rsrc, row * 4, 0, 1).bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(M)) & (slot < TOPK)
                return valid, fx.Int32(token) * TOPK + fx.Int32(slot)

            # A and row scales: tile i holds row 64*wave_m + 16*i + lr.
            if const_expr(act_quant_type == "ptpc"):
                a_scale_rsrc = _rsrc(
                    fxh._as_ptr(p_a_scale, fx.Float32), fx.Int64(routes) * 4
                )
            else:
                a_scale_all = fxh.view_as_torch_tensor(p_a_scale, (1,), fx.Float32)[0]
            if const_expr(weight_quant_type == "per_tensor"):
                w_scale_all = fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert),
                    (1,),
                    fx.Float32,
                )[0]
            a_packets, row_scales = [], []
            for i in range_constexpr(M_TILES):
                row = wave_m * 64 + i * 16 + lr
                valid, route = decode(row)
                base = valid.select(route * K + lq * 16, fx.Int32(OOB))
                for kb in range_constexpr(KB):
                    a_packets.append(_load(input_rsrc, base + kb * 64, 0, 4))
                scale = fx.Float32(
                    _load(routing_rsrc, row * 4, 0, 1).bitcast(fx.Float32)
                )
                if const_expr(act_quant_type == "ptpc"):
                    offset = valid.select(route * 4, fx.Int32(OOB))
                    token_scale = fx.Float32(
                        _load(a_scale_rsrc, offset, 0, 1).bitcast(fx.Float32)
                    )
                else:
                    token_scale = a_scale_all
                scale = token_scale * scale
                if const_expr(weight_quant_type == "per_tensor"):
                    scale = scale * w_scale_all
                row_scales.append(scale)
            a_halves = [_halves(packet) for packet in a_packets]

            # Read-back instruction s covers 16 rows with row bit 0 = s % 2 (so the
            # half swap is static) and bit 5 = s // 2; lanes 4q..4q+3 store one 64 B row.
            q = lane // 4
            store_offsets, read_offsets = [], []
            for s in range_constexpr(4):
                row = (
                    (s % 2)
                    | (((q >> 1) & 1) << 1)
                    | (((q >> 2) & 1) << 2)
                    | ((q & 1) << 3)
                    | (((q >> 3) & 1) << 4)
                    | ((s // 2) << 5)
                )
                valid, _ = decode(wave_m * 64 + row)
                column = wave_n * 64 + (lane % 4) * 16
                store_offsets.append(
                    valid.select((wave_m * 64 + row) * STRIDE + column, fx.Int32(OOB))
                )
                read_offsets.append(
                    lds_base + _slot(row) * 64 + ((lane % 4) ^ ((row >> 1) & 3)) * 16
                )
            # Fragment (i, j) of lane (lq, lr): row 16i + lr, 8 B at chunk 2j + lq // 2.
            write_offsets = []
            for i in range_constexpr(M_TILES):
                row = i * 16 + lr
                for j in range_constexpr(N_TILES):
                    physical = (2 * j + lq // 2) ^ ((row >> 1) & 3)
                    half = (lq % 2) ^ (row & 1)
                    write_offsets.append(lds_base + _slot(row) * 64 + physical * 16 + half * 8)

            weight_lane = lq * 256 + lr * 16

            def load_weights(packet):
                n0 = n_begin + packet * PACKET_N + wave_n * 32
                return [
                    _load(weight_rsrc, weight_lane + kb * 1024, (n0 + 16 * j) * K, 4)
                    for j in range_constexpr(N_TILES)
                    for kb in range_constexpr(KB)
                ]

            def mfma_ops(weights, acc):
                halves = [_halves(packet) for packet in weights]
                ops = []
                for kb in range_constexpr(KB):
                    for h in range_constexpr(2):
                        for j in range_constexpr(N_TILES):
                            for i in range_constexpr(M_TILES):

                                def op(i=i, j=j, kb=kb, h=h):
                                    acc[i][j] = _mfma(
                                        halves[j * KB + kb][h],
                                        a_halves[i * KB + kb][h],
                                        acc[i][j],
                                    )

                                ops.append(op)
                return ops

            def epilogue_ops(packet, acc):
                """Scale, pack and store one finished packet as small schedulable steps."""
                packet = fx.Int32(packet)
                state = {}
                ops = []
                if const_expr(weight_quant_type == "ptpc"):

                    def read_scales():
                        column = packet * PACKET_N + wave_n * 32 + lq * 4
                        state["scales"] = [
                            Vec(_lds_load(lds_scales + (column + 16 * j) * 4, 4)).bitcast(
                                fx.Float32
                            )
                            for j in range_constexpr(N_TILES)
                        ]

                    ops.append(read_scales)
                for i in range_constexpr(M_TILES):
                    for j in range_constexpr(N_TILES):

                        def scale(i=i, j=j):
                            values = acc[i][j]
                            if const_expr(weight_quant_type == "ptpc"):
                                values = values * state["scales"][j]
                            state[(i, j)] = values * row_scales[i]

                        def pack(i=i, j=j):
                            packed = _f32_to_bf16(state.pop((i, j))).bitcast(fx.Int32)
                            _lds_store(write_offsets[i * N_TILES + j], packed, 8)

                        ops += [scale, pack]
                for s in range_constexpr(4):

                    def read(s=s):
                        chunk = _lds_load(read_offsets[s], 4)
                        if const_expr(s % 2 == 1):
                            chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                        state[("out", s)] = chunk

                    ops.append(read)
                for s in range_constexpr(4):

                    def store(s=s):
                        n0 = packet * PACKET_N + n_begin
                        value = state.pop(("out", s))
                        if const_expr("nostore" in _DEBUG):
                            return
                        _store(
                            output_rsrc,
                            value,
                            store_offsets[s],
                            n0 * 2,
                            _STORE_AUX,
                        )

                    ops.append(store)
                return ops

            def zero_acc():
                return [
                    [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(N_TILES)]
                    for _ in range_constexpr(M_TILES)
                ]

            def step(packet, weights, previous, next_packet):
                """MFMAs of `packet`, interleaved with the previous packet's epilogue
                and the next packet's weight loads. Returns (acc, next weights)."""
                acc = zero_acc()
                mfmas = mfma_ops(weights, acc)
                tail = [] if previous is None else epilogue_ops(*previous)
                upcoming = []
                loads = []
                if const_expr(next_packet is not None):
                    n0 = n_begin + fx.Int32(next_packet) * PACKET_N + wave_n * 32
                    for j in range_constexpr(N_TILES):
                        for kb in range_constexpr(KB):

                            def load(j=j, kb=kb):
                                upcoming.append(
                                    _load(
                                        weight_rsrc,
                                        weight_lane + kb * 1024,
                                        (n0 + 16 * j) * K,
                                        4,
                                    )
                                )

                            loads.append(load)
                slots = len(mfmas)
                interval = slots // max(1, len(loads))
                if const_expr("seq" in _DEBUG):
                    for op in loads:
                        op()
                    for op in mfmas:
                        op()
                    for op in tail:
                        op()
                    return acc, upcoming
                for index in range_constexpr(slots):
                    mfmas[index]()
                    if const_expr(loads and index % interval == 0):
                        loads.pop(0)()
                    share = (len(tail) + slots - 1 - index) // (slots - index)
                    for _ in range_constexpr(share):
                        tail.pop(0)()
                    if const_expr("nosched" not in _DEBUG):
                        rocdl.sched_barrier(0)
                for op in loads + tail:
                    op()
                return acc, upcoming

            def flat(acc):
                return [acc[i][j] for i in range_constexpr(M_TILES) for j in range_constexpr(N_TILES)]

            def unflat(values):
                return [
                    [Vec(values[i * N_TILES + j]) for j in range_constexpr(N_TILES)]
                    for i in range_constexpr(M_TILES)
                ]

            # Weights are loaded DEPTH packets ahead. gfx9 vmcnt retires loads and stores
            # in issue order, so a load issued after a store only completes once that store
            # is acknowledged; the extra depth gives the stores time to drain. The loop
            # unrolls DEPTH + 1 steps so the buffer ring returns to the same registers.
            unroll = DEPTH + 1
            ring = [load_weights(fx.Int32(p)) for p in range_constexpr(min(DEPTH, PACKETS))]

            def run_step(p, ring, previous):
                ahead = p + DEPTH
                has_next = isinstance(ahead, int) and ahead < PACKETS or not isinstance(ahead, int)
                acc, upcoming = step(p, ring[0], previous, ahead if has_next else None)
                return acc, ring[1:] + ([upcoming] if has_next else [])

            acc, ring = run_step(0, ring, None)
            iters = max(0, (PACKETS - DEPTH - 1) // unroll)
            if const_expr(iters > 0):
                words = M_TILES * N_TILES
                for it, carried in range(
                    fx.Int32(0),
                    fx.Int32(iters),
                    fx.Int32(1),
                    init=flat(acc) + [v for w in ring for v in w],
                ):
                    acc = unflat(carried[:words])
                    ring = [
                        [Vec(v) for v in carried[words + d * N_TILES * KB : words + (d + 1) * N_TILES * KB]]
                        for d in range_constexpr(DEPTH)
                    ]
                    base = fx.Int32(it) * unroll + 1
                    for u in range_constexpr(unroll):
                        p = base + u
                        acc, ring = run_step(p, ring, (p - 1, acc))
                    carried = yield flat(acc) + [v for w in ring for v in w]
                acc = unflat(carried[:words])
                ring = [
                    [Vec(v) for v in carried[words + d * N_TILES * KB : words + (d + 1) * N_TILES * KB]]
                    for d in range_constexpr(DEPTH)
                ]
            for p in range_constexpr(1 + iters * unroll, PACKETS):
                acc, ring = run_step(p, ring, (p - 1, acc))
            for op in epilogue_ops(fx.Int32(PACKETS - 1), acc):
                op()

    @flyc.jit
    def launch_m128(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        kernel = moe_2stage_down_m128(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(
            grid=(task_num * oc_splits, 1, 1), block=(WAVES * 64, 1, 1), stream=stream
        )

    launch_m128.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_m128
