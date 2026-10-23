# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M128 four-wave Down for gfx942 FP8 on sort128 metadata.

A task is one sorted M128 block times N/OC output columns. Each wave owns 32 rows:
its A fragments stay in registers and it streams N64 x K weight packets straight
into registers, so the main loop has no CTA barrier. Results pass through a
wave-private LDS shuffle so eight consecutive lanes store one 128 B row segment.
Tasks are assigned in contiguous ranges per XCD so an expert's weights and each
A block are fetched into one L2.
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
        fx.Int32.ir_type
        if words == 1
        else ir.VectorType.get([words], fx.Int32.ir_type)
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


def _dma(rsrc, lds_address, voffset, soffset):
    rocdl.raw_ptr_buffer_load_async_lds(
        rsrc, _lds_pointer(lds_address), fx.Int32(4).ir_value(),
        fx.Int32(voffset).ir_value(), fx.Int32(soffset).ir_value(),
        fx.Int32(0).ir_value(), aux=_aux(0),
    )


def _mfma(weight, activation, acc):
    return Vec(
        rocdl.mfma_f32_16x16x32_fp8_fp8(
            ir.VectorType.get([4], fx.Float32.ir_type), [weight, activation, acc, 0, 0, 0]
        )
    )


def _halves(packet):
    """Split one 16 B K packet into the two i64 MFMA operands."""
    pair = Vec(packet).bitcast(fx.Int64)
    return pair[0], pair[1]


def m128_oc_splits(down_path):
    """`m128` uses OC4; `m128_ocS` selects S output-column splits per M128 block."""
    if down_path in ("m128", "m128_lds", "m128_regs", "m128_lds_regs", "m128_packed", "m128_pipe"):
        return 4
    prefix = "m128_oc"
    if not down_path.startswith(prefix) or not down_path[len(prefix):].isdigit():
        raise ValueError(f"Unsupported M128 Down path: {down_path}")
    return int(down_path[len(prefix):])


def _build_moe_gemm2_m128(
    *, N, K, weight_quant_type, act_quant_type, TOPK, padding, oc_splits,
    shared_lds=False,
    register_shuffle=False,
    packed_output=False,
    shared_subpacket=False,
):
    assert K % 64 == 0 and 64 <= K <= 256, "M128 Down supports K in {64,...,256}"
    assert N % (oc_splits * PACKET_N) == 0, "N/OC must be a multiple of 64"
    assert padding in (0, 32, 64, 128)
    assert not packed_output or padding == 0
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert act_quant_type in ("ptpc", "per_tensor")
    _, xcd_count = get_down_device_config()
    KB = K // 64
    N_OC = N // oc_splits
    PACKETS = N_OC // PACKET_N
    STRIDE = (N + padding // 2) * 2
    SUB_KB = 2 if K % 128 == 0 else 1
    SUBS_PER_PACKET = KB // SUB_KB
    RING_SLOTS = 3 if SUB_KB == 2 else 4
    half_shuffle = shared_subpacket and SUB_KB == 2
    WAVE_LDS = (16 if half_shuffle else 32) * 128
    packet_bytes = PACKET_N * K
    subpacket_bytes = PACKET_N * 64 * SUB_KB
    ring_bytes = RING_SLOTS * subpacket_bytes
    if shared_subpacket:
        shared_bytes = ring_bytes + WAVES * WAVE_LDS
    elif register_shuffle:
        shared_bytes = 2 * packet_bytes if shared_lds else 16
    else:
        shared_bytes = packet_bytes + WAVES * WAVE_LDS if shared_lds else WAVES * WAVE_LDS
    assert not shared_lds or K == 256
    assert shared_bytes <= 32 * 1024
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

            storage = fx.SharedAllocator().allocate(
                fx.Array[fx.Int32, shared_bytes // 4, 16]
            )
            lds_view = storage.peek().view(fx.make_layout(shared_bytes // 4, 1))
            storage_base = fx.Int32(fx.ptrtoint(fx.get_iter(lds_view)))
            lds_base = storage_base + (ring_bytes if shared_subpacket else packet_bytes if shared_lds else 0) + wave * WAVE_LDS

            ids_rsrc = _rsrc(fxh._as_ptr(p_sorted_ids, fx.Int32) + fx.Int64(row_begin), BLOCK_M * 4)
            routing_rsrc = _rsrc(
                fxh._as_ptr(p_sorted_weights, fx.Float32) + fx.Int64(row_begin), BLOCK_M * 4
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

            def issue_packet(packet):
                slot = storage_base + ((packet % 2) * packet_bytes if register_shuffle else 0)
                start = (n_begin + packet * PACKET_N) * K
                for turn in range_constexpr(packet_bytes // WAVES // 256):
                    offset = wave * (packet_bytes // WAVES) + turn * 256
                    _dma(weight_rsrc, slot + offset, lane * 4, start + offset)
                rocdl.asyncmark()

            def issue_subpacket(subpacket):
                subpacket = fx.Int32(subpacket)
                packet = subpacket // SUBS_PER_PACKET
                kb = (subpacket % SUBS_PER_PACKET) * SUB_KB
                slot = storage_base + (subpacket % RING_SLOTS) * subpacket_bytes
                source = (n_begin + packet * PACKET_N + wave * 16) * K + kb * 1024
                for turn in range_constexpr(4 * SUB_KB):
                    offset = (subpacket < PACKETS * SUBS_PER_PACKET).select(source + turn * 256, fx.Int32(OOB))
                    _dma(weight_rsrc, slot + wave * SUB_KB * 1024 + turn * 256, lane * 4, offset)
                rocdl.asyncmark()

            if const_expr(shared_subpacket):
                for prefetch in range_constexpr(RING_SLOTS - 1):
                    issue_subpacket(fx.Int32(prefetch))
            if const_expr(shared_lds):
                issue_packet(fx.Int32(0))
                if const_expr(register_shuffle and PACKETS > 1):
                    issue_packet(fx.Int32(1))

            def decode(row):
                encoded = _load(ids_rsrc, row * 4, 0, 1).bitcast(fx.Uint32)
                token, slot = encoded & 0xFFFFFF, encoded >> 24
                valid = (token < fx.Uint32(M)) & (slot < TOPK)
                return valid, fx.Int32(token) * TOPK + fx.Int32(slot)

            # A rows of this lane: tile i holds row 32*wave + 16*i + lr.
            a_packets, row_scales = [], []
            if const_expr(act_quant_type == "ptpc"):
                a_scale_rsrc = _rsrc(fxh._as_ptr(p_a_scale, fx.Float32), fx.Int64(routes) * 4)
            else:
                a_scale_all = fxh.view_as_torch_tensor(p_a_scale, (1,), fx.Float32)[0]
            if const_expr(weight_quant_type == "per_tensor"):
                w_scale_all = fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert), (1,), fx.Float32
                )[0]
            for i in range_constexpr(2):
                row = wave * 32 + i * 16 + lr
                valid, route = decode(row)
                base = valid.select(route * K + lq * 16, fx.Int32(OOB))
                for kb in range_constexpr(KB):
                    a_packets.append(_load(input_rsrc, base + kb * 64, 0, 4))
                scale = fx.Float32(_load(routing_rsrc, row * 4, 0, 1).bitcast(fx.Float32))
                if const_expr(act_quant_type == "ptpc"):
                    token_scale = fx.Float32(
                        _load(a_scale_rsrc, valid.select(route * 4, fx.Int32(OOB)), 0, 1).bitcast(
                            fx.Float32
                        )
                    )
                else:
                    token_scale = a_scale_all
                scale = token_scale * scale
                if const_expr(weight_quant_type == "per_tensor"):
                    scale = scale * w_scale_all
                row_scales.append(scale)
            a_halves = [_halves(packet) for packet in a_packets]

            # Store rows: instruction s writes rows 32*wave + 8*s + lane // 8.
            store_offsets = []
            for s in range_constexpr(4):
                row = wave * 32 + s * 8 + lane // 8
                valid, _ = decode(row)
                store_offsets.append(
                    valid.select(row * (128 if packed_output else STRIDE) + (lane % 8) * 16, fx.Int32(OOB))
                )

            # Wave-private CShuffle: chunk c of row r lives at chunk c ^ (r % 8); rows with
            # (r // 8) odd also swap their two 8 B halves, so writes and reads are conflict free.
            write_offsets = []
            for j in range_constexpr(4):
                chunk_id = (2 * j + lq // 2) ^ (lr % 8)
                half = (lq % 2) ^ (lr // 8)
                write_offsets.append(lds_base + lr * 128 + chunk_id * 16 + half * 8)
            read_offset = lds_base + (lane // 8) * 128 + ((lane % 8) ^ (lane // 8)) * 16

            weight_lane = lq * 256 + lr * 16
            if const_expr(weight_quant_type == "ptpc"):
                w_scale_rsrc = _rsrc(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, N * 4
                )

            def load_packet(packet):
                n0 = n_begin + packet * PACKET_N
                if const_expr(shared_lds):
                    slot = storage_base + ((packet % 2) * packet_bytes if register_shuffle else 0)
                    weights = [
                        _lds_load(slot + weight_lane + kb * 1024 + channel * 16 * K, 4)
                        for channel in range_constexpr(4)
                        for kb in range_constexpr(KB)
                    ]
                else:
                    weights = [
                        _load(weight_rsrc, weight_lane + kb * 1024, (n0 + 16 * j) * K, 4)
                        for j in range_constexpr(4)
                        for kb in range_constexpr(KB)
                    ]
                scales = []
                if const_expr(weight_quant_type == "ptpc"):
                    scales = [
                        _load(w_scale_rsrc, lq * 16, (n0 + 16 * j) * 4, 4)
                        for j in range_constexpr(4)
                    ]
                return weights + scales

            def store_acc(packet, acc, scales, shuffle_offset=0):
                n0 = n_begin + packet * PACKET_N
                output_column = n0 * (BLOCK_M if packed_output else 1) * 2
                packed_words = []
                for i in range_constexpr(2):
                    for j in range_constexpr(4):
                        values = acc[i][j]
                        if const_expr(weight_quant_type == "ptpc"):
                            values = values * Vec(scales[j]).bitcast(fx.Float32)
                        packed = _f32_to_bf16(values * row_scales[i]).bitcast(fx.Int32)
                        if const_expr(register_shuffle):
                            packed_words.extend((packed[0], packed[1]))
                        else:
                            _lds_store(write_offsets[j] + (0 if half_shuffle else i * 2048) + shuffle_offset, packed, 8)
                    if const_expr(half_shuffle):
                        for part in range_constexpr(2):
                            chunk = _lds_load(read_offset + part * 1024, 4)
                            if const_expr(part == 1):
                                chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                            _store(output_rsrc, chunk, store_offsets[i * 2 + part], output_column, DOWN_STORE_CACHE_MODIFIER)
                if const_expr(register_shuffle):
                    for lane_mask, register_mask in ((16, 2), (8, 4)):
                        lane_bit = (lane & lane_mask) != 0
                        for register in range_constexpr(16):
                            if const_expr(register & register_mask == 0):
                                low = packed_words[register]
                                high = packed_words[register | register_mask]
                                peer = lane_bit.select(low, high).shuffle_xor(lane_mask, 64)
                                packed_words[register] = lane_bit.select(peer, low)
                                packed_words[register | register_mask] = lane_bit.select(high, peer)
                    source_lane = ((lane >> 3) & 7) | ((lane & 1) << 5) | ((lane & 2) << 3) | ((lane & 4) << 1)
                    reordered = [
                        fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, (source_lane * 4).ir_value(), word.ir_value()))
                        for word in packed_words
                    ]
                    for part in range_constexpr(4):
                        _store(output_rsrc, Vec.from_elements(reordered[4 * part:4 * part + 4], fx.Int32),
                               store_offsets[part], output_column, DOWN_STORE_CACHE_MODIFIER)
                elif const_expr(not half_shuffle):
                    for s in range_constexpr(4):
                        chunk = _lds_load(read_offset + s * 1024 + shuffle_offset, 4)
                        if const_expr(s % 2 == 1):
                            chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                        _store(
                            output_rsrc, chunk, store_offsets[s], output_column,
                            DOWN_STORE_CACHE_MODIFIER,
                        )

            def compute_store(packet, state, shuffle_offset=0):
                weights = [_halves(state[index]) for index in range_constexpr(4 * KB)]
                acc = [
                    [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                    for _ in range_constexpr(2)
                ]
                for kb in range_constexpr(KB):
                    for half in range_constexpr(2):
                        for channel in range_constexpr(4):
                            for height in range_constexpr(2):
                                acc[height][channel] = _mfma(
                                    weights[channel * KB + kb][half],
                                    a_halves[height * KB + kb][half],
                                    acc[height][channel],
                                )
                store_acc(packet, acc, state[4 * KB:], shuffle_offset)

            def store_previous_half(packet, previous, height):
                output_column = (n_begin + packet * PACKET_N) * (BLOCK_M if packed_output else 1) * 2
                for channel in range_constexpr(4):
                    values = previous[height * 4 + channel]
                    if const_expr(weight_quant_type == "ptpc"):
                        values = values * Vec(previous[8 + channel]).bitcast(fx.Float32)
                    packed = _f32_to_bf16(values * row_scales[height]).bitcast(fx.Int32)
                    _lds_store(write_offsets[channel], packed, 8)
                for part in range_constexpr(2):
                    chunk = _lds_load(read_offset + part * 1024, 4)
                    if const_expr(part == 1):
                        chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                    _store(output_rsrc, chunk, store_offsets[height * 2 + part], output_column, DOWN_STORE_CACHE_MODIFIER)

            # Ping-pong in an unrolled pair: the next packet loads while the current one
            # computes, and the carried buffer is reloaded in place, so the loop head never
            # waits for the stores still in flight (gfx9 vmcnt also counts stores).
            if const_expr(shared_subpacket):
                initial = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)]
                if const_expr(weight_quant_type == "ptpc"):
                    initial += [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(4)]
                for packet, previous in range(fx.Int32(0), fx.Int32(PACKETS), fx.Int32(1), init=initial):
                    scales = []
                    if const_expr(weight_quant_type == "ptpc"):
                        scales = [
                            _load(w_scale_rsrc, lq * 16, (n_begin + packet * PACKET_N + 16 * channel) * 4, 4)
                            for channel in range_constexpr(4)
                        ]
                    acc = [
                        [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                        for _ in range_constexpr(2)
                    ]
                    for sub in range_constexpr(SUBS_PER_PACKET):
                        subpacket = packet * SUBS_PER_PACKET + sub
                        rocdl.wait_asyncmark(RING_SLOTS - 2)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        rocdl.s_barrier()
                        slot = storage_base + (subpacket % RING_SLOTS) * subpacket_bytes
                        weights = [
                            _halves(_lds_load(slot + (channel * SUB_KB + inner) * 1024 + weight_lane, 4))
                            for channel in range_constexpr(4)
                            for inner in range_constexpr(SUB_KB)
                        ]
                        issue_subpacket(subpacket + RING_SLOTS - 1)
                        for inner in range_constexpr(SUB_KB):
                            for half in range_constexpr(2):
                                if const_expr(half_shuffle and K == 256 and inner == 1 and half == 0):
                                    if packet > 0:
                                        store_previous_half(packet - 1, previous, sub)
                                for channel in range_constexpr(4):
                                    for height in range_constexpr(2):
                                        acc[height][channel] = _mfma(
                                            weights[channel * SUB_KB + inner][half],
                                            a_halves[height * KB + sub * SUB_KB + inner][half],
                                            acc[height][channel],
                                        )
                    if const_expr(not half_shuffle or K != 256):
                        store_acc(packet, acc, scales)
                    previous = yield [value for row in acc for value in row] + scales
                if const_expr(half_shuffle and K == 256):
                    store_previous_half(fx.Int32(PACKETS - 1), previous, 0)
                    store_previous_half(fx.Int32(PACKETS - 1), previous, 1)
                rocdl.wait_asyncmark(0)
                rocdl.s_waitcnt(lgkmcnt=0)
            elif const_expr(shared_lds):
                for packet in range(fx.Int32(0), fx.Int32(PACKETS), fx.Int32(1)):
                    if const_expr(register_shuffle):
                        if packet + 1 < PACKETS:
                            rocdl.wait_asyncmark(1)
                        else:
                            rocdl.wait_asyncmark(0)
                    else:
                        rocdl.wait_asyncmark(0)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.s_barrier()
                    current = load_packet(packet)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.s_barrier()
                    if const_expr(register_shuffle):
                        if packet + 2 < PACKETS:
                            issue_packet(packet + 2)
                    else:
                        if packet + 1 < PACKETS:
                            issue_packet(packet + 1)
                    compute_store(packet, current)
            else:
                current = load_packet(fx.Int32(0))
                pairs = (PACKETS - 1) // 2
                if const_expr(pairs > 0):
                    for pair, carried in range(
                        fx.Int32(0), fx.Int32(pairs), fx.Int32(1), init=current
                    ):
                        first = pair * 2
                        following = load_packet(first + 1)
                        compute_store(first, carried)
                        preloaded = load_packet(first + 2)
                        compute_store(first + 1, following)
                        carried = yield preloaded
                    current = carried
                if const_expr(PACKETS - 2 * pairs == 2):
                    following = load_packet(fx.Int32(PACKETS - 1))
                    compute_store(fx.Int32(PACKETS - 2), current)
                    compute_store(fx.Int32(PACKETS - 1), following)
                else:
                    compute_store(fx.Int32(PACKETS - 1), current)

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
        kernel.launch(grid=(task_num * oc_splits, 1, 1), block=(WAVES * 64, 1, 1), stream=stream)

    launch_m128.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_m128
