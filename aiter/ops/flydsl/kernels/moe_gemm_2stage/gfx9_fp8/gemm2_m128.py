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
    if down_path == "m128":
        return 4
    prefix = "m128_oc"
    if not down_path.startswith(prefix) or not down_path[len(prefix):].isdigit():
        raise ValueError(f"Unsupported M128 Down path: {down_path}")
    return int(down_path[len(prefix):])


def _build_moe_gemm2_m128(
    *, N, K, weight_quant_type, act_quant_type, TOPK, padding, oc_splits
):
    assert K % 64 == 0 and 64 <= K <= 256, "M128 Down supports K in {64,...,256}"
    assert N % (oc_splits * PACKET_N) == 0, "N/OC must be a multiple of 64"
    assert padding in (0, 32, 64, 128)
    assert weight_quant_type in ("ptpc", "per_tensor")
    assert act_quant_type in ("ptpc", "per_tensor")
    _, xcd_count = get_down_device_config()
    KB = K // 64
    N_OC = N // oc_splits
    PACKETS = N_OC // PACKET_N
    STRIDE = (N + padding // 2) * 2
    WAVE_LDS = 32 * 128
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
                fx.Array[fx.Int32, WAVES * WAVE_LDS // 4, 16]
            )
            lds_view = storage.peek().view(fx.make_layout(WAVES * WAVE_LDS // 4, 1))
            lds_base = fx.Int32(fx.ptrtoint(fx.get_iter(lds_view))) + wave * WAVE_LDS

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
                    valid.select(row * STRIDE + (lane % 8) * 16, fx.Int32(OOB))
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

            def compute_store(packet, state):
                n0 = n_begin + packet * PACKET_N
                weights = [_halves(state[index]) for index in range_constexpr(4 * KB)]
                acc = [
                    [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                    for _ in range_constexpr(2)
                ]
                for kb in range_constexpr(KB):
                    for h in range_constexpr(2):
                        for j in range_constexpr(4):
                            for i in range_constexpr(2):
                                acc[i][j] = _mfma(
                                    weights[j * KB + kb][h],
                                    a_halves[i * KB + kb][h],
                                    acc[i][j],
                                )
                for i in range_constexpr(2):
                    for j in range_constexpr(4):
                        values = acc[i][j]
                        if const_expr(weight_quant_type == "ptpc"):
                            values = values * Vec(state[4 * KB + j]).bitcast(fx.Float32)
                        packed = _f32_to_bf16(values * row_scales[i]).bitcast(fx.Int32)
                        _lds_store(write_offsets[j] + i * 2048, packed, 8)
                for s in range_constexpr(4):
                    chunk = _lds_load(read_offset + s * 1024, 4)
                    if const_expr(s % 2 == 1):
                        chunk = chunk.shuffle(chunk, [2, 3, 0, 1])
                    _store(output_rsrc, chunk, store_offsets[s], n0 * 2,
                           DOWN_STORE_CACHE_MODIFIER)

            # Ping-pong in an unrolled pair: the next packet loads while the current one
            # computes, and the carried buffer is reloaded in place, so the loop head never
            # waits for the stores still in flight (gfx9 vmcnt also counts stores).
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
