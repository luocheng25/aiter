# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M128 four-wave Down for gfx942 FP8 on sort128 metadata.

A task is one sorted M128 block times N/OC output columns, swept in N64 x K weight
packets. Each wave owns 32 rows and keeps its A fragments in registers. Every packet
is copied once per CTA into a three-slot LDS ring by async LDS DMA (each wave copies
a quarter, two packets ahead), so the CTA meets at one barrier per packet. The
previous packet's scaling, BF16 packing, wave-private LDS shuffle and stores are
interleaved with the current packet's MFMAs. Tasks are assigned in contiguous ranges
per XCD so an expert's weights and each A block are fetched into one L2.
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
from .common import _f32_to_bf16, get_down_device_config

BLOCK_M = 128
WAVES = 4
PACKET_N = 64
PACKET_BYTES = PACKET_N * 256
SLOTS = 3
SCALE_BYTES = PACKET_N * 4
SLOT_BYTES = PACKET_BYTES + SCALE_BYTES
SHUFFLE_BYTES = 32 * 64  # 32 rows x 32 BF16 columns per wave and half packet
# Byte offset that is out of bounds for every buffer descriptor built below.
OOB = 0x7FFFFFF0


def _aux(value):
    return ir.IntegerAttr.get(fx.Int32.ir_type, value)


def _rsrc(pointer, num_bytes):
    return fx.rocdl.get_buffer_rsrc(
        fx.rocdl.make_buffer_ptr(pointer, num_records_bytes=num_bytes)
    )


def _load(rsrc, voffset, soffset, words):
    result_type = (
        fx.Int32.ir_type if words == 1 else ir.VectorType.get([words], fx.Int32.ir_type)
    )
    value = rocdl_dialect.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=_aux(0),
    ).result
    return fx.Int32(value) if words == 1 else Vec(value)


def _store(rsrc, value, voffset, soffset):
    rocdl_dialect.RawPtrBufferStoreOp(
        value.ir_value(),
        rsrc,
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        aux=_aux(0),
    )


def _lds_pointer(address):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.Int32(address).ir_value())


def _lds_store(address, value, alignment):
    llvm.store(value.ir_value(), _lds_pointer(address), alignment=alignment)


def _lds_load(address, words):
    vector = ir.VectorType.get([words], fx.Int32.ir_type)
    return Vec(llvm.load(vector, _lds_pointer(address), alignment=4 * words))


def _dma(rsrc, lds_address, voffset, soffset):
    """Async 4 B/lane global -> LDS copy of 256 contiguous bytes per wave."""
    rocdl.raw_ptr_buffer_load_async_lds(
        rsrc,
        _lds_pointer(lds_address),
        fx.Int32(4).ir_value(),
        fx.Int32(voffset).ir_value(),
        fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(),
        aux=_aux(0),
    )


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


def _slot_row(row):
    """Shuffle row slot: swap row bits 0 and 3 so bit 3 selects the bank half."""
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
    assert K == 256, "M128 Down copies N64 x K256 weight packets"
    assert N % (oc_splits * PACKET_N) == 0, "N/OC must be a multiple of 64"
    assert padding in (0, 32, 64, 128)
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert act_quant_type in ("ptpc", "per_tensor")
    _, xcd_count = get_down_device_config()
    KB = K // 64
    N_OC = N // oc_splits
    PACKETS = N_OC // PACKET_N
    STRIDE = (N + padding // 2) * 2
    LDS_BYTES = SLOTS * SLOT_BYTES + WAVES * SHUFFLE_BYTES
    assert BLOCK_M * STRIDE < OOB and LDS_BYTES <= 64 * 1024

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

            storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, LDS_BYTES // 4, 16])
            lds_view = storage.peek().view(fx.make_layout(LDS_BYTES // 4, 1))
            lds_base = fx.Int32(fx.ptrtoint(fx.get_iter(lds_view)))
            shuffle_base = lds_base + SLOTS * SLOT_BYTES + wave * SHUFFLE_BYTES

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
                scale_rsrc = _rsrc(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, N * 4
                )

            def issue_packet(packet):
                """This wave's quarter of the packet (and wave 0 the 64 scales)."""
                slot = lds_base + (packet % SLOTS) * SLOT_BYTES
                n0 = n_begin + fx.Int32(packet) * PACKET_N
                for turn in range_constexpr(PACKET_BYTES // WAVES // 256):
                    offset = wave * (PACKET_BYTES // WAVES) + turn * 256
                    _dma(weight_rsrc, slot + offset, lane * 4, n0 * K + offset)
                if const_expr(weight_quant_type == "ptpc"):
                    if wave == 0:
                        _dma(scale_rsrc, slot + PACKET_BYTES, lane * 4, n0 * 4)
                rocdl.asyncmark()

            def decode(row):
                encoded = _load(ids_rsrc, row * 4, 0, 1).bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(M)) & (slot < TOPK)
                return valid, fx.Int32(token) * TOPK + fx.Int32(slot)

            # The first two packets start before the A and row-scale setup.
            issue_packet(0)
            if const_expr(PACKETS > 1):
                issue_packet(1)

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
            # A and row scales: tile i holds row 32*wave + 16*i + lr.
            a_packets, row_scales = [], []
            for i in range_constexpr(2):
                row = wave * 32 + i * 16 + lr
                valid, route = decode(row)
                base = valid.select(route * K + lq * 16, fx.Int32(OOB))
                for kb in range_constexpr(KB):
                    a_packets.append(_load(input_rsrc, base + kb * 64, 0, 4))
                scale = fx.Float32(_load(routing_rsrc, row * 4, 0, 1).bitcast(fx.Float32))
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

            # Read-back s of a half packet covers 16 rows with row bit 0 = s (so the
            # half swap is static); four consecutive lanes store one 64 B row piece.
            q = lane // 4
            store_offsets, read_offsets = [], []
            for s in range_constexpr(2):
                row = s | (((q >> 1) & 1) << 1) | (((q >> 2) & 1) << 2) | ((q & 1) << 3) | (
                    ((q >> 3) & 1) << 4
                )
                valid, _ = decode(wave * 32 + row)
                store_offsets.append(
                    valid.select((wave * 32 + row) * STRIDE + (lane % 4) * 16, fx.Int32(OOB))
                )
                read_offsets.append(
                    shuffle_base + _slot_row(row) * 64 + ((lane % 4) ^ ((row >> 1) & 3)) * 16
                )
            # Fragment (i, j) of lane (lq, lr): row 16i + lr, 8 B at chunk 2(j%2) + lq // 2.
            write_offsets = []
            for i in range_constexpr(2):
                row = i * 16 + lr
                for jj in range_constexpr(2):
                    physical = (2 * jj + lq // 2) ^ ((row >> 1) & 3)
                    half = (lq % 2) ^ (row & 1)
                    write_offsets.append(
                        shuffle_base + _slot_row(row) * 64 + physical * 16 + half * 8
                    )
            weight_lane = lq * 256 + lr * 16

            def read_scales(packet):
                if const_expr(weight_quant_type != "ptpc"):
                    return []
                slot = lds_base + (packet % SLOTS) * SLOT_BYTES + PACKET_BYTES
                return [
                    Vec(_lds_load(slot + (16 * j + 4 * lq) * 4, 4)).bitcast(fx.Float32)
                    for j in range_constexpr(4)
                ]

            def epilogue_ops(packet, acc, scales):
                """Scale, pack, shuffle and store one finished packet in small steps."""
                state = {}
                ops = []
                for half in range_constexpr(2):
                    for i in range_constexpr(2):
                        for jj in range_constexpr(2):
                            j = 2 * half + jj

                            def pack(i=i, j=j, jj=jj):
                                values = acc[i][j]
                                if const_expr(weight_quant_type == "ptpc"):
                                    values = values * scales[j]
                                packed = _f32_to_bf16(values * row_scales[i]).bitcast(fx.Int32)
                                _lds_store(write_offsets[i * 2 + jj], packed, 8)

                            ops.append(pack)
                    for s in range_constexpr(2):

                        def read(s=s, half=half):
                            chunk = _lds_load(read_offsets[s], 4)
                            if const_expr(s == 1):
                                chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                            state[(half, s)] = chunk

                        ops.append(read)
                    for s in range_constexpr(2):

                        def store(s=s, half=half):
                            column = (n_begin + fx.Int32(packet) * PACKET_N + half * 32) * 2
                            _store(output_rsrc, state.pop((half, s)), store_offsets[s], column)

                        ops.append(store)
                return ops

            def step(packet, previous, issue=True, last=False):
                """Compute `packet` from its LDS slot; interleave the previous packet's
                epilogue. Returns (acc, scales) of this packet."""
                packet = fx.Int32(packet)
                # One group (the next packet) may stay in flight except at the end.
                rocdl.wait_asyncmark(0 if last else 1)
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)
                if const_expr(issue):
                    issue_packet(packet + 2)
                slot = lds_base + (packet % SLOTS) * SLOT_BYTES
                scales = read_scales(packet)
                acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                       for _ in range_constexpr(2)]
                tail = [] if previous is None else epilogue_ops(*previous)
                weights = [None] * KB

                def read_weights(kb):
                    weights[kb] = [
                        _halves(_lds_load(slot + weight_lane + j * 4096 + kb * 1024, 4))
                        for j in range_constexpr(4)
                    ]

                read_weights(0)
                slots = KB * 16
                index = 0
                for kb in range_constexpr(KB):
                    if const_expr(kb + 1 < KB):
                        read_weights(kb + 1)
                    for h in range_constexpr(2):
                        for j in range_constexpr(4):
                            for i in range_constexpr(2):
                                acc[i][j] = _mfma(
                                    weights[kb][j][h], a_halves[i * KB + kb][h], acc[i][j]
                                )
                                share = (len(tail) + slots - 1 - index) // (slots - index)
                                for _ in range_constexpr(share):
                                    tail.pop(0)()
                                index += 1
                                rocdl.sched_barrier(0)
                for op in tail:
                    op()
                return acc, scales

            def flat(state):
                acc, scales = state
                return [acc[i][j] for i in range_constexpr(2) for j in range_constexpr(4)] + scales

            def unflat(values):
                acc = [[Vec(values[i * 4 + j]) for j in range_constexpr(4)]
                       for i in range_constexpr(2)]
                return acc, [Vec(v) for v in values[8:]]

            # Packet p waits for its own DMA group, then issues p + 2 into the slot that
            # packet p - 1 just released. The loop covers the packets that still issue
            # a successor; it is unrolled twice so the carried accumulators alternate.
            state = step(0, None, issue=PACKETS > 2, last=PACKETS == 1)
            pairs = max(0, (PACKETS - 3) // 2)
            if const_expr(pairs > 0):
                for pair, carried in range(
                    fx.Int32(0), fx.Int32(pairs), fx.Int32(1), init=flat(state)
                ):
                    first = fx.Int32(pair) * 2 + 1
                    state = step(first, (first - 1, *unflat(carried)))
                    state = step(first + 1, (first, *state))
                    carried = yield flat(state)
                state = unflat(carried)
            for packet in range_constexpr(1 + 2 * pairs, PACKETS):
                state = step(packet, (packet - 1, *state), issue=packet + 2 < PACKETS,
                             last=packet == PACKETS - 1)
            for op in epilogue_ops(PACKETS - 1, *state):
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
