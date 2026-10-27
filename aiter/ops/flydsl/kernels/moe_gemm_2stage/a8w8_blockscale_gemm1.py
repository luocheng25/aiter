# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""gfx950 W128x128/A1x128 FP8 GateUp with BF16 routed output.

Kernel synced from the optimized standalone test_fp8_stage1_giu1.py: 8-wave,
256x256, half-M single-FIFO pipeline with optional K rotation. Weights use
Aiter's shuffle_weight(16,16), with gate rows followed by up rows. Scales are
FP32: A is physically [K/128, tokens], B is [E, 2*I/128, K/128]. Each task
consumes 256 sorted rows and produces 128 columns of SiLU(gate) * up, without
clamp or routing weights. A block whose bottom 128 rows hold no route runs a
128-row pipeline instead (same DMA/LDS/DOT helpers, M128 phase order).
A persistent grid (at most one workgroup per CU) walks (block, N tile) tasks
round-robin; the last two K tiles of each task prefetch the next task's first
two through the two-slot LDS ring, so DMA streams across task boundaries.
Sorting capacity may exceed the valid padded prefix: tasks are counted from
num_valid_ids[0] on device and blocks past it are never read.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Float8E4M3FN, Float32, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .common import get_device_cache_key

SORT_BLOCK_M = 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
TOKEN_MASK = 0xFFFFFF
ROUTE_WORDS = SORT_BLOCK_M + 64
BUFFER_LIMIT = 0x7FFFFFFF
# As an unsigned voffset this is past every buffer below 4 GiB, so loads return 0.
OOB_OFFSET = -1


def validate_parameters(tokens, intermediate_size, hidden_size, topk, num_experts):
    if not 0 < tokens <= TOKEN_MASK:
        raise ValueError(f"tokens must be in [1, {TOKEN_MASK}]")
    if intermediate_size <= 0 or intermediate_size % 128:
        raise ValueError("intermediate_size must be a positive multiple of 128")
    if hidden_size < 256 or hidden_size % 256:
        raise ValueError("hidden_size must be a multiple of 256 and at least 256")
    if not 0 < topk < 256 or num_experts < topk:
        raise ValueError("require 0 < topk < 256 and num_experts >= topk")
    # Buffer descriptors hold a 32-bit byte count.
    extents = {
        "a (tokens*K)": tokens * hidden_size,
        "a_scale (tokens*K/128*4)": tokens * (hidden_size // 128) * 4,
        "output (tokens*topk*I*2)": tokens * topk * intermediate_size * 2,
        "weight per expert (2*I*K)": 2 * intermediate_size * hidden_size,
    }
    over = [f"{name}={size:,}" for name, size in extents.items() if size >= 2**32]
    if over:
        raise ValueError("buffer byte extents must be < 4 GiB: " + ", ".join(over))


def _buffer_resource(tensor, num_records_bytes, base_byte_offset=0):
    ptr = fx.add_offset(
        fx.recast_iter(fx.Uint8, fx.get_iter(tensor)),
        fx.make_int_tuple(base_byte_offset),
    )
    view = fx.make_view(ptr, fx.make_layout(1, 1))
    buffer = rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def _lds_byte_ptr(ptr, byte_offset):
    return fx.to_llvm_ptr(
        fx.add_offset(fx.recast_iter(fx.Uint8, ptr), fx.make_int_tuple(byte_offset))
    )


def compile_fp8_stage1_giu1(
    intermediate_size,
    hidden_size,
    topk,
    num_experts,
    *,
    xcd_swizzle=True,
    group_size_m=1,
    k_rotate=True,
):
    """Build a half-M, single-FIFO kernel for AIter-native FP8 weight bytes.

    Routing must contain complete 256-row expert blocks, each assigned to a
    valid local expert. Padding uses an invalid token/slot; each real
    (token, slot) must appear exactly once. No EP expert masking or bias.
    Native B slabs are read coalesced into unpadded LDS. Each DOT phase pairs
    four scalar FP32 FMAs with one MFMA; the launch disables packed FP32 ops to
    keep the FMAs scalar. The retained pipeline is half-M, one FIFO, B lookahead
    and scalar B scales; no full-M, K-unroll, priority-switch or alternative
    loader experiments. Blocks with an empty bottom half (sorted row 128
    invalid) take a 128-row pipeline that skips the bottom A DMAs, MFMAs and
    stores.

    The launch passes num_workers workgroups; worker w runs tasks w, w +
    num_workers, ... (num_workers = task count gives one task per workgroup).
    The next task's routing is loaded at a task's start into an LDS ring and
    its first two K tiles are fetched by the task's last two, in the same
    pipeline slots as steady-state refills: a full tile issues its A bottom one
    tile ahead, and a half task's last tile adds the next task's (out of range
    when that task is half too), so both pipelines keep their steady-state
    vmcnt budgets across any task transition. Output stores are unconditional
    (invalid rows go out of range; half tasks add four placeholders), keeping
    the VMEM count between tasks fixed.
    """
    validate_parameters(1, intermediate_size, hidden_size, topk, num_experts)
    if group_size_m <= 0:
        raise ValueError("group_size_m must be positive")
    kb_count = hidden_size // BLOCK_K
    n_tiles = intermediate_size // BLOCK_N
    # Keep A64 padding: tested row-major XOR layouts were slower on the main
    # gfx950 targets, even with precomputed offsets and zero register spills.
    # Saving 2KiB of LDS did not improve measured end-to-end kernel latency.
    a_group8 = 8 * BLOCK_K + 16
    a_group16 = 2 * a_group8 + 32
    a_slice_rows = BLOCK_M // 2
    a_lds_elems = (a_slice_rows // 16) * a_group16
    b_group16 = 16 * BLOCK_K
    b_lds_elems = (BLOCK_N // 16) * b_group16

    batch_rotation = 37 * group_size_m * n_tiles % 256

    def _get_pids(pid, m_tiles, grid_size):
        if const_expr(xcd_swizzle):
            # Match bounded XCD batches without scattering adjacent experts
            # over the full grid. Leave the incomplete final batch linear.
            full_batches = grid_size - grid_size % 256
            if pid < full_batches:
                # A persistent worker takes one task per batch. Rotating batch b
                # by an odd block count times b varies which blocks (full or
                # half) a worker gets, e.g. when experts alternate them.
                batch = pid // 256
                pid = (
                    batch * 256
                    + ((pid % 8) * 32 + (pid % 256) // 8 + batch * batch_rotation) % 256
                )
        group_id = pid // (group_size_m * n_tiles)
        first_m = group_id * group_size_m
        remaining = m_tiles - first_m
        group_m = (remaining < group_size_m).select(remaining, group_size_m)
        return (
            first_m + (pid % (group_size_m * n_tiles)) % group_m,
            (pid % (group_size_m * n_tiles)) // group_m,
        )

    get_pids = ASTRewriter.transform(_get_pids)

    @fx.struct
    class LDS:
        ##LDS A ping buffer
        a_t0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b0_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]

        ##LDS A pong buffer
        a_t1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_t1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice0: fx.Array[Float8E4M3FN, a_lds_elems, 16]
        a_b1_slice1: fx.Array[Float8E4M3FN, a_lds_elems, 16]

        ##LDS B ping buffer
        b_l0: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r0: fx.Array[Float8E4M3FN, b_lds_elems, 16]

        ##LDS B pong buffer
        b_l1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        b_r1: fx.Array[Float8E4M3FN, b_lds_elems, 16]
        # One DMA per K128 tile: 256 row scales duplicated across 512 lanes.
        # Both compute M slices share these entries; do not split the scales.
        scale_a0: fx.Array[Float32, 512, 4]
        scale_a1: fx.Array[Float32, 512, 4]
        # Task routing ring (task index mod 3): 256 sorted IDs, then 64 copies
        # of the task's expert ID. Task t + 1 is fetched at task t's start into
        # the buffer task t - 2 finished reading a whole task earlier.
        route: fx.Array[fx.Int32, 3 * ROUTE_WORDS, 4]

    @flyc.kernel(known_block_size=[512, 1, 1])
    def fp8_stage1_kernel(
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_sorted_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_c: fx.Tensor,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
        num_workers: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane_id = tid % 64
        wave_id = tid // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        # The launch grid holds num_workers persistent workers. fx.grid_dim would
        # call __ockl_get_num_groups, which cannot be inlined under -packed-fp32-ops
        # (target-feature mismatch); its VGPR result would make every task index
        # divergent and add waterfall loops around the buffer accesses.
        worker = fx.Int32(fx.block_idx.x)
        prefix_rsrc = _buffer_resource(arg_num_valid_ids, 4)
        valid_rows = rocdl.raw_ptr_buffer_load(
            T.i32, prefix_rsrc, fx.Int32(0), fx.Int32(0)
        )
        valid_rows = fx.Int32(rocdl.readfirstlane(T.i32, arith._to_raw(valid_rows)))
        # Aiter passes full-capacity sort buffers: only the valid prefix holds
        # routes and blocks past it may carry arbitrary expert IDs, so tasks are
        # counted from it on device. num_expert_blocks sizes the descriptors.
        valid_blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        valid_blocks = (valid_blocks < num_expert_blocks).select(
            valid_blocks, num_expert_blocks
        )
        num_tasks = valid_blocks * n_tiles
        if worker < num_tasks:
            sorted_rsrc = _buffer_resource(
                arg_sorted_ids,
                arith._to_raw(fx.Int32(num_expert_blocks * SORT_BLOCK_M * 4)),
            )
            expert_rsrc = _buffer_resource(
                arg_expert_ids, arith._to_raw(fx.Int32(num_expert_blocks * 4))
            )
            a_rsrc = _buffer_resource(arg_a, fx.Int64(num_tokens) * hidden_size)
            scale_a_rsrc = _buffer_resource(
                arg_scale_a, fx.Int64(num_tokens) * (kb_count * 4)
            )
            c_rsrc = _buffer_resource(
                arg_c, fx.Int64(num_tokens) * (topk * intermediate_size * 2)
            )
            lds = fx.SharedAllocator().allocate(LDS).peek()
            # Map the low two logical row bits to the padded 16-row groups.
            # The matching producer below preserves a single 8KiB DMA per M slice.
            read_layout = fx.make_layout(
                ((a_slice_rows // 16, 2, 8), (32, BLOCK_K // 32)),
                ((a_group16, a_group8, BLOCK_K), (1, 32)),
            )
            a_top = [
                [fx.make_view(p, read_layout) for p in pair]
                for pair in (
                    (lds.a_t0_slice0.ptr, lds.a_t0_slice1.ptr),
                    (lds.a_t1_slice0.ptr, lds.a_t1_slice1.ptr),
                )
            ]
            a_bottom = [
                [fx.make_view(p, read_layout) for p in pair]
                for pair in (
                    (lds.a_b0_slice0.ptr, lds.a_b0_slice1.ptr),
                    (lds.a_b1_slice0.ptr, lds.a_b1_slice1.ptr),
                )
            ]
            # Keep native N16/K16 bytes in LDS. Move the existing logical-row
            # permutation to the consumer so each producer wave reads 1024B
            # contiguously instead of eight scattered rows from eight N16 groups.
            b_read_layout = fx.make_layout(
                ((2, BLOCK_N // 16, 8), ((16, 2), BLOCK_K // 32)),
                ((16, 32, b_group16), ((1, 256), 512)),
            )
            b_gate = [
                fx.make_view(p, b_read_layout) for p in (lds.b_l0.ptr, lds.b_l1.ptr)
            ]
            b_up = [
                fx.make_view(p, b_read_layout) for p in (lds.b_r0.ptr, lds.b_r1.ptr)
            ]

            # Transposed C, as in blockscale GEMM: actual B is MMA operand A.
            mma_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float8E4M3FN)
            )
            mma_atom = fx.atom_set_value(mma_atom, "scale_a", fx.Int32(0))
            mma_atom = fx.atom_set_value(mma_atom, "scale_b", fx.Int32(0))
            tiled_mma = fx.make_tiled_mma(
                mma_atom,
                fx.make_layout((4, 2, 1), (1, 4, 0)),
                (None, None, fx.make_layout((32, 4), (1, 32))),
            )
            thread_mma = tiled_mma.thr_slice(tid)
            copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), Float8E4M3FN)
            copy_a = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
            copy_b = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)

            frag_a = thread_mma.make_fragment_B(a_top[0][0])
            frag_bl = thread_mma.make_fragment_A(b_gate[0])
            frag_br = thread_mma.make_fragment_A(b_up[0])
            dest_a = copy_a.retile(frag_a)
            dest_bl = copy_b.retile(frag_bl)
            dest_br = copy_b.retile(frag_br)
            c_tile = fx.make_rmem_tensor(
                fx.make_ordered_layout((BLOCK_N, BLOCK_M), (1, 0)), Float32
            )
            c_tl = thread_mma.make_fragment_C(c_tile)
            c_tr = thread_mma.make_fragment_C(c_tile)
            c_bl = thread_mma.make_fragment_C(c_tile)
            c_br = thread_mma.make_fragment_C(c_tile)
            p_tile = fx.flat_divide(c_tile, (BLOCK_N, 64))[None, None, 0, 0]
            frag_p = thread_mma.make_fragment_C(p_tile)

            def route_valid(fused):
                slot = (fused >> 24) & 255
                return ((fused & TOKEN_MASK) < num_tokens) & (slot < topk)

            # Routing of one task (256 sorted IDs, then 64 copies of its expert
            # ID) lives in an LDS ring buffer; route_base is its byte offset. It
            # is fetched with two VGPR loads per wave (waves w and w + 4 load the
            # same rows, so vmcnt budgets stay wave-uniform) and written to LDS
            # once a pipeline wait has retired them: LLVM tracks those VGPRs
            # exactly, while an LDS DMA here would make it wait for vmcnt(0)
            # before every later routing read.
            route_row = wave_id % 4 * 64 + lane_id

            def task_coords(v):
                block_id, n_tile = get_pids(v, valid_blocks, num_tasks)
                return fx.Int32(block_id), fx.Int32(n_tile)

            def load_route(v, valid):
                block_id, _ = task_coords(v)
                return [
                    rocdl.raw_ptr_buffer_load(
                        T.i32,
                        rsrc,
                        valid.select(offset, fx.Int32(OOB_OFFSET)),
                        fx.Int32(0),
                    )
                    for rsrc, offset in (
                        (sorted_rsrc, (block_id * SORT_BLOCK_M + route_row) * 4),
                        (expert_rsrc, block_id * 4),
                    )
                ]

            def store_route(raw, route_base):
                for value, row in zip(raw, (route_row, SORT_BLOCK_M + lane_id)):
                    _llvm.store(
                        arith._to_raw(fx.Int32(value)),
                        _lds_byte_ptr(lds.route.ptr, route_base + row * 4),
                    )

            def route_load(route_base, row):
                return fx.Int32(
                    _llvm.load(
                        T.i32, _lds_byte_ptr(lds.route.ptr, route_base + row * 4)
                    )
                )

            # One cooperative DWORDx4 DMA covers one complete A64xK128 entry.
            # Within an entry r = wave_id//2 + 4*(wave_id%2) + 8*(lane_id//8).
            # This is the inverse of the (4,2,8) consumer's padded row mapping.
            a_local_row = wave_id // 2 + (wave_id % 2) * 4 + (lane_id // 8) * 8

            def task_context(v, route_base, valid=None):
                """Descriptors and DMA offsets of task v from its LDS routing.

                valid=None marks the current task; otherwise every DMA of the
                task is out of range when valid is false.
                """
                block_id, n_tile = task_coords(v)
                expert = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, arith._to_raw(route_load(route_base, SORT_BLOCK_M))
                    )
                )
                # Valid routes are a prefix of every 256-row block, so sorted
                # row 128 tells whether the bottom half holds any route.
                bottom_id = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, arith._to_raw(route_load(route_base, 128))
                    )
                )
                b_slab = BLOCK_N * hidden_size
                b_gate_base = fx.Int64(expert) * (
                    2 * intermediate_size * hidden_size
                ) + (fx.Int64(n_tile) * b_slab)
                ctx = {
                    "valid": valid,
                    "full": route_valid(bottom_id),
                    "n_tile": n_tile,
                    "sorted_base": block_id * SORT_BLOCK_M,
                    "k_off": expert % kb_count,
                    "b_gate_rsrc": _buffer_resource(arg_b, b_slab, b_gate_base),
                    "b_up_rsrc": _buffer_resource(
                        arg_b, b_slab, b_gate_base + intermediate_size * hidden_size
                    ),
                    "scale_b_rsrc": _buffer_resource(
                        arg_scale_b,
                        2 * n_tiles * kb_count * 4,
                        fx.Int64(expert) * (2 * n_tiles * kb_count * 4),
                    ),
                }
                a_voffsets = []
                for bottom in range_constexpr(2):
                    offsets = []
                    for m_slice in range_constexpr(2):
                        fused = route_load(
                            route_base,
                            bottom * 128 + m_slice * a_slice_rows + a_local_row,
                        )
                        row_valid = route_valid(fused)
                        if const_expr(valid is not None):
                            row_valid = row_valid & valid
                        offset = (fused & TOKEN_MASK) * hidden_size + fx.Int32(
                            (tid % 8) * 16
                        )
                        offsets.append(row_valid.select(offset, fx.Int32(OOB_OFFSET)))
                    a_voffsets.append(offsets)
                ctx["a_voffsets"] = a_voffsets
                scale_id = route_load(route_base, tid % SORT_BLOCK_M)
                scale_valid = route_valid(scale_id)
                if const_expr(valid is not None):
                    scale_valid = scale_valid & valid
                ctx["scale_voffset"] = scale_valid.select(
                    (scale_id & TOKEN_MASK) * 4, fx.Int32(OOB_OFFSET)
                )
                return ctx

            wave_dma_base = rocdl.readfirstlane(
                T.i32,
                arith._to_raw(
                    fx.Int32((wave_id % 2) * a_group8 + (wave_id // 2) * a_group16)
                ),
            )
            scale_dma_base = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(tid * 4))
            )

            def operand_dma_ptrs(views):
                return [
                    [_lds_byte_ptr(fx.get_iter(view), wave_dma_base) for view in pair]
                    for pair in views
                ]

            at_dma = operand_dma_ptrs(a_top)
            ab_dma = operand_dma_ptrs(a_bottom)
            b_wave_base = rocdl.readfirstlane(
                T.i32, arith._to_raw(fx.Int32(wave_id * b_group16))
            )

            def b_dma_ptrs(views):
                return [_lds_byte_ptr(fx.get_iter(view), b_wave_base) for view in views]

            bl_dma = b_dma_ptrs(b_gate)
            br_dma = b_dma_ptrs(b_up)
            scale_dma = [
                _lds_byte_ptr(p, scale_dma_base)
                for p in (lds.scale_a0.ptr, lds.scale_a1.ptr)
            ]

            # Native FP8 offset = (n//16)*16*K + (k//16)*256 + (n%16)*16 + k%16.
            # A wave copies an N16/K64 slab, two slabs cover K128. No host-side
            # weight conversion; the matching LDS view preserves MFMA operands.
            b_voffset = fx.Int32(wave_id * (16 * hidden_size) + lane_id * 16)

            def k_tile(ctx, ki):
                if const_expr(not k_rotate):
                    return ki
                t = ctx["k_off"] + ki
                return (t < kb_count).select(t, t - kb_count)

            # Every DMA helper issues a fixed number of VMEM ops (At/Ab: 1 per M
            # slice, Bl/Br: 2, ScaleA: 1) whether or not its rows are valid.
            def raw_g2s_a(ctx, lds_dst, voffset, ki):
                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    lds_dst,
                    fx.Int32(16),
                    voffset,
                    fx.Int32(k_tile(ctx, ki) * BLOCK_K),
                    fx.Int32(0),
                    fx.Int32(0),
                )

            def async_copy_At(ctx, lds_idx, ki, m_slices=(0, 1)):
                for m_slice in range_constexpr(2):
                    if const_expr(m_slice in m_slices):
                        raw_g2s_a(
                            ctx,
                            at_dma[lds_idx][m_slice],
                            ctx["a_voffsets"][0][m_slice],
                            ki,
                        )

            def async_copy_Ab(ctx, lds_idx, ki):
                # A half task's bottom rows are all invalid: these move no data.
                for m_slice in range_constexpr(2):
                    raw_g2s_a(
                        ctx, ab_dma[lds_idx][m_slice], ctx["a_voffsets"][1][m_slice], ki
                    )

            def raw_g2s_b(ctx, rsrc, lds_dst, ki):
                soffset = fx.Int32(k_tile(ctx, ki) * (BLOCK_K * 16))
                if const_expr(ctx["valid"] is not None):
                    soffset = ctx["valid"].select(soffset, fx.Int32(OOB_OFFSET))
                for chunk in range_constexpr(2):
                    # The immediate offset advances both the global and the LDS address.
                    rocdl.raw_ptr_buffer_load_lds(
                        rsrc,
                        lds_dst,
                        fx.Int32(16),
                        b_voffset,
                        soffset,
                        fx.Int32(chunk * 1024),
                        fx.Int32(0),
                    )

            def async_copy_Bl(ctx, lds_idx, ki):
                raw_g2s_b(ctx, ctx["b_gate_rsrc"], bl_dma[lds_idx], ki)

            def async_copy_Br(ctx, lds_idx, ki):
                raw_g2s_b(ctx, ctx["b_up_rsrc"], br_dma[lds_idx], ki)

            def async_copy_scale_a(ctx, lds_idx, ki):
                rocdl.raw_ptr_buffer_load_lds(
                    scale_a_rsrc,
                    scale_dma[lds_idx],
                    fx.Int32(4),
                    ctx["scale_voffset"],
                    fx.Int32(k_tile(ctx, ki) * num_tokens * 4),
                    fx.Int32(0),
                    fx.Int32(0),
                )

            def lds_rd_At(lds_idx, m_slice):
                fx.copy(copy_atom, copy_a.partition_S(a_top[lds_idx][m_slice]), dest_a)

            def lds_rd_Ab(lds_idx, m_slice):
                fx.copy(
                    copy_atom, copy_a.partition_S(a_bottom[lds_idx][m_slice]), dest_a
                )

            def lds_rd_Bl(lds_idx):
                fx.copy(copy_atom, copy_b.partition_S(b_gate[lds_idx]), dest_bl)

            def lds_rd_Br(lds_idx):
                fx.copy(copy_atom, copy_b.partition_S(b_up[lds_idx]), dest_br)

            scale_copy = fx.make_copy_atom(fx.UniversalCopy32b(), Float32)

            def lds_rd_scale_a(lds_idx, bottom, m_slice):
                values = []
                for m0 in range_constexpr(2):
                    index = wave_m * 256 + bottom * 128 + wave_m * 16 + lane_id % 16
                    index = index + (m_slice * 2 + m0) * 32
                    ptr = (
                        lds.scale_a0.ptr
                        if const_expr(lds_idx == 0)
                        else lds.scale_a1.ptr
                    )
                    src = fx.make_view(
                        fx.add_offset(ptr, fx.make_int_tuple(index)),
                        fx.make_layout(1, 1),
                    )
                    frag = fx.make_fragment_like(src)
                    fx.copy(scale_copy, src, frag)
                    values.append(Vec(frag.load())[0])
                return Vec.from_elements(values, Float32)

            # Gate/up ScaleB of this N tile and K tile are wave-uniform: two SMEM
            # loads into SGPRs. Inline s_buffer_load_dword works around the pinned
            # LLVM: SITargetLowering::lowerSBuffer() drops the MachineMemOperand,
            # so the backend cannot see the source buffer and inserts extra
            # waitcnts that cost significant performance.
            def load_scale_b(ctx, ki):
                offset = fx.Int32((ctx["n_tile"] * kb_count + k_tile(ctx, ki)) * 4)
                result = _llvm.inline_asm(
                    ir.Type.parse("!llvm.struct<(f32, f32)>"),
                    [
                        arith._to_raw(ctx["scale_b_rsrc"]),
                        arith._to_raw(offset),
                        arith._to_raw(offset + n_tiles * kb_count * 4),
                    ],
                    "s_buffer_load_dword $0, $2, $3\ns_buffer_load_dword $1, $2, $4",
                    "=&s,=&s,s,s,s,~{memory}",
                    has_side_effects=True,
                )
                return Vec.from_elements(
                    [
                        fx.Float32(_llvm.extractvalue(T.f32, result, [i]))
                        for i in range_constexpr(2)
                    ],
                    Float32,
                )

            def compute(c_frag, b_frag, dq_scale_a, dq_scale_b, dq_m_slice):
                # One DOT phase: retire the pending P FIFO into c_frag with the
                # scales of the phase that produced it, then refill P with this
                # phase's b_frag x frag_a products.
                # Per-lane fragments of a wave's 64(M) x 32(N) share of a C tile:
                #   frag_a [Kval, Mrep, Krep]: (32, 2, 1) FP8 -> 16 dwords
                #   b_frag [Kval, Nrep, Krep]: (32, 2, 1) FP8 -> 16 dwords
                #   c_frag [Cval, Nrep, Mrep]: (4, 2, 4) F32  -> 32 dwords
                #   frag_p [Cval, Nrep, Mrep]: (4, 2, 2) F32  -> 16 dwords
                # frag_a holds one M slice; c_frag keeps both (Mrep 0-1 and 2-3).
                # dq_scale_a has two row scales; dq_scale_b is wave-uniform.
                #
                # for m0 in (Mrep):
                #     dq_scale = dq_scale_a[m0] * dq_scale_b
                #     for n0 in (Nrep):
                #         c_frag[:, n0, dq_m_slice * 2 + m0] += dq_scale * frag_p[:, n0, m0]
                #         frag_p[:, n0, m0] = mfma_16x16x128(b_frag[:, n0], frag_a[:, m0], 0)
                #
                # Compiler-only sched_barrier fences keep 4 scalar FMAs -> 1 MFMA;
                # the launch's -packed-fp32-ops prevents re-packing into
                # v_pk_fma_f32. Retired and new partials are independent, so the
                # FMAs overlap the MFMA pipeline. WG sync stays in the caller.
                # The MFMA is the raw intrinsic with zero scales (the unscaled
                # form): on the pinned FlyDSL, fx.gemm rejects these FP8 register
                # fragments in rmem SSA promotion, and their DWORD views are not
                # contiguous 8-dword operands.
                for m0 in range_constexpr(2):
                    dq_scale = Vec(dq_scale_a)[m0] * dq_scale_b
                    rocdl.sched_barrier(0)
                    for n0 in range_constexpr(2):
                        sub_c = c_frag[None, n0, dq_m_slice * 2 + m0]
                        sub_p = frag_p[None, n0, m0]
                        partial = Vec(sub_p.load())
                        accum = Vec(sub_c.load())
                        sub_c.store(
                            Vec.from_elements(
                                [
                                    fx.fma(partial[i], dq_scale, accum[i])
                                    for i in range_constexpr(4)
                                ],
                                Float32,
                            )
                        )
                        rocdl.sched_barrier(0)
                        sub_p.store(
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.vec(4, T.f32),
                                [
                                    Vec(b_frag[None, n0, 0].load()).bitcast(fx.Int32),
                                    Vec(frag_a[None, m0, 0].load()).bitcast(fx.Int32),
                                    Vec.filled(4, 0.0, Float32),
                                    0,
                                    0,
                                    0,
                                    fx.Int32(0),
                                    0,
                                    fx.Int32(0),
                                ],
                            )
                        )
                        rocdl.sched_barrier(0)

            def begin_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            def end_compute_phase():
                rocdl.sched_barrier(0)
                rocdl.s_barrier()
                rocdl.sched_barrier(0)

            pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")

            def silu_mul(gate, up):
                exponent = rocdl.exp2(T.f32, arith._to_raw(gate * -1.4426950408889634))
                inverse = rocdl.rcp(T.f32, 1.0 + exponent)
                return (gate * inverse) * up

            def read_store_ids(route_base):
                return [
                    route_load(
                        route_base,
                        bottom * 128 + repeat * 32 + wave_m * 16 + lane_id % 16,
                    )
                    for bottom in range_constexpr(2)
                    for repeat in range_constexpr(4)
                ]

            def output_store(packed, offset):
                rocdl.raw_ptr_buffer_store(
                    packed.ir_value(),
                    c_rsrc,
                    offset.ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 0),
                )

            # Every task issues exactly eight output stores (invalid rows go out of
            # range), so the next task's vmcnt budgets see a fixed VMEM count.
            def store_gate_up(ctx, gates, ups, bottom, store_ids):
                for row_repeat in range_constexpr(4):
                    row = bottom * 128 + row_repeat * 32 + wave_m * 16 + lane_id % 16
                    fused = store_ids[bottom * 4 + row_repeat]
                    token, slot = fused & TOKEN_MASK, (fused >> 24) & 255
                    valid = route_valid(fused) & (ctx["sorted_base"] + row < valid_rows)
                    g0, g1 = Vec(gates[None, 0, row_repeat].load()), Vec(
                        gates[None, 1, row_repeat].load()
                    )
                    u0, u1 = Vec(ups[None, 0, row_repeat].load()), Vec(
                        ups[None, 1, row_repeat].load()
                    )
                    v0 = [silu_mul(g0[i], u0[i]) for i in range_constexpr(4)]
                    v1 = [silu_mul(g1[i], u1[i]) for i in range_constexpr(4)]
                    # In the transposed C, lane_id % 16 selects the row and the
                    # lane group (lane_id // 16) four consecutive columns.
                    # aX/bX are output columns of N repeat 0/1 (64 apart),
                    # relative to the wave's N base:
                    # | lane | p0      | p1      | p2      | p3      |
                    # |------|---------|---------|---------|---------|
                    # |    0 | a0,a1   | a2,a3   | b0,b1   | b2,b3   |
                    # |   16 | a4,a5   | a6,a7   | b4,b5   | b6,b7   |
                    # |   32 | a8,a9   | a10,a11 | b8,b9   | b10,b11 |
                    # |   48 | a12,a13 | a14,a15 | b12,b13 | b14,b15 |
                    # permlane16_swap exchanges the odd groups' p0/p1 with the
                    # even groups' p2/p3 (partner lane = lane_id ^ 16), so each
                    # lane then owns eight consecutive columns of its row:
                    # | lane | swap0[0], swap1[0], swap0[1], swap1[1] | columns |
                    # |------|----------------------------------------|---------|
                    # |    0 | a0..a7                                 | 0-7     |
                    # |   16 | b0..b7                                 | 64-71   |
                    # |   32 | a8..a15                                | 8-15    |
                    # |   48 | b8..b15                                | 72-79   |
                    p0 = rocdl.cvt_pk_bf16_f32(v0[0], v0[1])
                    p1 = rocdl.cvt_pk_bf16_f32(v0[2], v0[3])
                    p2 = rocdl.cvt_pk_bf16_f32(v1[0], v1[1])
                    p3 = rocdl.cvt_pk_bf16_f32(v1[2], v1[3])
                    swap0 = rocdl.permlane16_swap(
                        pair_type, arith._to_raw(p0), arith._to_raw(p2), False, False
                    )
                    swap1 = rocdl.permlane16_swap(
                        pair_type, arith._to_raw(p1), arith._to_raw(p3), False, False
                    )
                    packed = Vec.from_elements(
                        [
                            fx.Int32(_llvm.extractvalue(T.i32, swap0, [0])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap1, [0])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap0, [1])),
                            fx.Int32(_llvm.extractvalue(T.i32, swap1, [1])),
                        ],
                        fx.Int32,
                    )
                    lane_group = lane_id // 16
                    col = (
                        ctx["n_tile"] * 128
                        + lane_group % 2 * 64
                        + wave_n * 16
                        + lane_group // 2 * 8
                    )
                    address = (
                        (token * topk + slot) * intermediate_size + fx.Int32(col)
                    ) * 2
                    output_store(packed, valid.select(address, fx.Int32(OOB_OFFSET)))

            def store_placeholders():
                # Stand in for a bottom half's four stores. Distinct offsets keep
                # LLVM from merging them; all stay above any < 4 GiB extent.
                for i in range_constexpr(4):
                    output_store(
                        Vec.filled(4, 0, fx.Int32), fx.Int32(OOB_OFFSET - 16 * i)
                    )

            def full_tile(cur, ki, tick, ab, ref, old_a, old_b, hook=None):
                """Compute BM256 K tile ki of cur from LDS slot tick.

                ab = (task, tile) whose A bottom refills the other slot; ref = (task,
                tile) refilling slot tick.
                | Phase | MFMA   | DQ (VALU)   | LDS read (slot) | AC (slot, tile) / wait |
                |-------|--------|-------------|-----------------|------------------------|
                | TL s0 | A00*B0 | BR s1 (k-1) | A00, SA00 (t)   | A1 (u, ab)             |
                | TR s0 | A00*B1 | TL s0       | B1 (t)          | -                      |
                | BL s0 | A10*B0 | TR s0       | A10, SA10 (t)   | B0 (t, ref)            |
                | BR s0 | A10*B1 | BL s0       | -               | B1 (t, ref)            |
                | TL s1 | A01*B0 | BR s0       | A01, SA01 (t)   | -                      |
                | TR s1 | A01*B1 | TL s1       | -               | A0 (t, ref)            |
                | BL s1 | A11*B0 | TR s1       | A11, SA11 (t)   | vmcnt(6)               |
                | BR s1 | A11*B1 | BL s1       | B0 (u, k+1)     | SA (t, ref)            |
                Ahs = A[half h, M slice s]; B0/B1 = gate/up; SA = ScaleA; t = tick,
                u = 1 - t. Every phase: LR/AC/wait -> lgkmcnt(0) -> s_barrier -> DOT
                -> s_barrier. TL s0 also loads both ScaleB SGPRs (SMEM). BL s1 leaves
                only the six ref At/Bl/Br DMAs in flight, so every wave's next-tile
                operands (the latest being TL s0's A bottom) have landed. The B0
                lookahead of a task's last tile is read at the next task start.
                """
                tock = 1 - tick
                last = ref[0] is not cur and ref[1] == 1
                for m_slice in range_constexpr(2):
                    lds_rd_At(lds_idx=tick, m_slice=m_slice)
                    top_scale = lds_rd_scale_a(lds_idx=tick, bottom=0, m_slice=m_slice)
                    if const_expr(m_slice == 0):
                        b_scale = load_scale_b(cur, ki)
                        async_copy_Ab(ab[0], tock, ab[1])
                    begin_compute_phase()
                    # New TL; consume BR from the preceding M slice/K tile.
                    compute(c_br, frag_bl, old_a, old_b, 1 - m_slice)
                    end_compute_phase()
                    if const_expr(m_slice == 0):
                        lds_rd_Br(lds_idx=tick)
                    else:
                        # Both At[tick] slices were last read two or more phases ago.
                        async_copy_At(ref[0], tick, ref[1])
                    begin_compute_phase()
                    compute(c_tl, frag_br, top_scale, b_scale[0], m_slice)
                    end_compute_phase()
                    lds_rd_Ab(lds_idx=tick, m_slice=m_slice)
                    bottom_scale = lds_rd_scale_a(
                        lds_idx=tick, bottom=1, m_slice=m_slice
                    )
                    if const_expr(m_slice == 0):
                        # B registers persist through slice1; LDS can be reused.
                        async_copy_Bl(ref[0], tick, ref[1])
                    else:
                        rocdl.s_waitcnt(vmcnt=6)
                    begin_compute_phase()
                    compute(c_tr, frag_bl, top_scale, b_scale[1], m_slice)
                    end_compute_phase()
                    if const_expr(m_slice == 0):
                        async_copy_Br(ref[0], tick, ref[1])
                    else:
                        async_copy_scale_a(ref[0], tick, ref[1])
                        if const_expr(not last):
                            lds_rd_Bl(lds_idx=tock)
                        if const_expr(hook is not None):
                            hook()
                    begin_compute_phase()
                    compute(c_bl, frag_br, bottom_scale, b_scale[0], m_slice)
                    end_compute_phase()
                    old_a, old_b = bottom_scale, b_scale[1]
                return old_a, old_b

            def half_tile(
                cur, ki, tick, ab, ref, old_a, old_b, l1_extra=0, r1_extra=0, hook=None
            ):
                """Compute the top 128 rows of K tile ki of cur from LDS slot tick.

                Per K tile k (t = tick, u = 1 - t): L/R = gate/up x top A slice 0/1.
                | Phase | MFMA   | DQ (VALU) | LDS read (slot) | AC (slot, tile) / wait       |
                |-------|--------|-----------|-----------------|------------------------------|
                | L0    | A00*B0 | R1 (k-1)  | A00, SA00 (t)   | [A1 (u, ab)]                 |
                | R0    | A00*B1 | L0        | B1 (t)          | B0, A00 (t, ref)             |
                | L1    | A01*B0 | R0        | A01, SA01 (t)   | vmcnt(8 + l1); B1 (t, ref)   |
                | R1    | A01*B1 | L1        | B0 (u, k+1)     | vmcnt(5 + r1); A01, SA (t, ref) |
                A tile refill issues seven VMEM ops: B0 (2), A00 (1), B1 (2), A01 (1)
                and SA (1). L1's wait lands B0 (k+1) before R1 reads it; R1's lands
                every k+1 operand before the next L0. A task's last tile also fills
                the next task's A bottom (ab), which a full next task reads.
                """
                tock = 1 - tick
                last = ref[0] is not cur and ref[1] == 1
                # L0: gate x A0, retiring the previous up x A1 partial.
                lds_rd_At(lds_idx=tick, m_slice=0)
                scale0 = lds_rd_scale_a(lds_idx=tick, bottom=0, m_slice=0)
                b_scale = load_scale_b(cur, ki)
                if const_expr(ab is not None):
                    async_copy_Ab(ab[0], tock, ab[1])
                begin_compute_phase()
                compute(c_tr, frag_bl, old_a, old_b, 1)
                end_compute_phase()
                # R0: up x A0, retiring gate x A0.
                lds_rd_Br(lds_idx=tick)
                async_copy_Bl(ref[0], tick, ref[1])
                async_copy_At(ref[0], tick, ref[1], m_slices=(0,))
                begin_compute_phase()
                compute(c_tl, frag_br, scale0, b_scale[0], 0)
                end_compute_phase()
                # L1: gate x A1, retiring up x A0.
                rocdl.s_waitcnt(vmcnt=8 + l1_extra)
                if const_expr(hook is not None):
                    hook()
                lds_rd_At(lds_idx=tick, m_slice=1)
                scale1 = lds_rd_scale_a(lds_idx=tick, bottom=0, m_slice=1)
                async_copy_Br(ref[0], tick, ref[1])
                begin_compute_phase()
                compute(c_tr, frag_bl, scale0, b_scale[1], 0)
                end_compute_phase()
                # R1: up x A1, retiring gate x A1.
                rocdl.s_waitcnt(vmcnt=5 + r1_extra)
                async_copy_At(ref[0], tick, ref[1], m_slices=(1,))
                async_copy_scale_a(ref[0], tick, ref[1])
                if const_expr(not last):
                    lds_rd_Bl(lds_idx=tock)
                begin_compute_phase()
                compute(c_tl, frag_br, scale1, b_scale[0], 1)
                end_compute_phase()
                return scale1, b_scale[1]

            def drain(c_frag, old_a, old_b):
                # Retire the final partial (second M slice) into c_frag.
                for m0 in range_constexpr(2):
                    dq_scale = Vec(old_a)[m0] * old_b
                    for n0 in range_constexpr(2):
                        sub_c = c_frag[None, n0, 2 + m0]
                        partial = Vec(frag_p[None, n0, m0].load())
                        accum = Vec(sub_c.load())
                        sub_c.store(
                            Vec.from_elements(
                                [
                                    fx.fma(partial[i], dq_scale, accum[i])
                                    for i in range_constexpr(4)
                                ],
                                Float32,
                            )
                        )

            def tile_refs(cur, nxt, k):
                """(task, tile) of the A-bottom (k+1) and the slot (k+2) refills."""
                ab = (cur, k + 1) if k + 1 < kb_count else (nxt, k + 1 - kb_count)
                ref = (cur, k + 2) if k + 2 < kb_count else (nxt, k + 2 - kb_count)
                return ab, ref

            # A task's first two and last two K tiles are peeled: the half
            # pipeline's tile 0 waits behind the task boundary's VMEM ops and the
            # last two tiles prefetch the next task's tiles 0 and 1 into slots 0
            # and 1. With fewer than four K tiles, tile 0 already prefetches the
            # next task, whose routing must then land first.
            head_tiles = 2 if kb_count >= 4 else 0
            route_bytes = ROUTE_WORDS * 4

            # Prologue: the first task's routing, then its K tiles 0 and 1 in the
            # state a task tail leaves: tile 0 landed, tile 1 in tail order
            # (a full task's tile 1 A bottom comes from its first TL phase).
            store_route(load_route(worker, worker < num_tasks), 0)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            first = task_context(worker, 0)
            async_copy_scale_a(first, 0, 0)
            async_copy_Bl(first, 0, 0)
            async_copy_At(first, 0, 0)
            async_copy_Br(first, 0, 0)
            async_copy_Ab(first, 0, 0)
            rocdl.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            async_copy_Bl(first, 1, 1)
            async_copy_At(first, 1, 1, m_slices=(0,))
            async_copy_Br(first, 1, 1)
            async_copy_At(first, 1, 1, m_slices=(1,))
            async_copy_scale_a(first, 1, 1)
            rocdl.sched_barrier(0)
            # Out-of-range stand-ins for a previous task's eight output stores.
            store_placeholders()
            store_placeholders()
            rocdl.sched_barrier(0)
            # Offset the two groups of four waves by one stage. The lower group
            # closes this unmatched barrier after the last task.
            if wave_id >= 4:
                rocdl.s_barrier()

            # Between tasks the VMEM queue holds the tile-1 refill, eight output
            # stores and the next task's two routing loads, all younger than
            # what the half pipeline's tile 0 waits for.
            boundary_ops = 8 + 2
            my_tasks = (num_tasks - worker + num_workers - 1) // num_workers
            for t_index in range(my_tasks):
                task = worker + t_index * num_workers
                route_base = t_index % 3 * route_bytes
                next_base = (t_index + 1) % 3 * route_bytes
                cur = task_context(task, route_base)
                # B0 of tile 0, read one phase ahead (skipped by the last tile).
                lds_rd_Bl(lds_idx=0)
                upcoming = task + num_workers
                upcoming_valid = upcoming < num_tasks
                upcoming = upcoming_valid.select(upcoming, task)
                rocdl.sched_barrier(0)
                upcoming_route = load_route(upcoming, upcoming_valid)
                rocdl.sched_barrier(0)
                keep_route = functools.partial(store_route, upcoming_route, next_base)

                if const_expr(head_tiles == 0):
                    keep_route()
                    rocdl.s_waitcnt(lgkmcnt=0)
                    rocdl.s_barrier()
                frag_p.fill(0)
                c_tl.fill(0)
                c_tr.fill(0)
                if cur["full"]:
                    c_bl.fill(0)
                    c_br.fill(0)
                    f_old_a = Vec.filled(2, 0.0, Float32)
                    f_old_b = fx.Float32(0)
                    for k in range_constexpr(head_tiles):
                        ab, ref = tile_refs(cur, None, k)
                        # Tile 0's BL s1 wait retires the routing loads.
                        f_old_a, f_old_b = full_tile(
                            cur,
                            k,
                            k,
                            ab,
                            ref,
                            f_old_a,
                            f_old_b,
                            hook=keep_route if k == 0 else None,
                        )
                    if const_expr(kb_count > 4):
                        for f_k in range(2, kb_count - 2, 2):
                            f_ki = fx.Int32(f_k)
                            for unroll_idx in range_constexpr(2):
                                ki = f_ki + unroll_idx
                                f_old_a, f_old_b = full_tile(
                                    cur,
                                    ki,
                                    unroll_idx,
                                    (cur, ki + 1),
                                    (cur, ki + 2),
                                    f_old_a,
                                    f_old_b,
                                )
                    f_nxt = task_context(upcoming, next_base, upcoming_valid)
                    for k in range_constexpr(kb_count - 2, kb_count):
                        ab, ref = tile_refs(cur, f_nxt, k)
                        f_old_a, f_old_b = full_tile(
                            cur, k, k % 2, ab, ref, f_old_a, f_old_b
                        )
                    f_ids = read_store_ids(route_base)
                    drain(c_br, f_old_a, f_old_b)
                    rocdl.sched_barrier(0)
                    store_gate_up(cur, c_tl, c_tr, 0, f_ids)
                    store_gate_up(cur, c_bl, c_br, 1, f_ids)
                    rocdl.sched_barrier(0)
                else:
                    h_old_a = Vec.filled(2, 0.0, Float32)
                    h_old_b = fx.Float32(0)
                    for k in range_constexpr(head_tiles):
                        _, ref = tile_refs(cur, None, k)
                        h_old_a, h_old_b = half_tile(
                            cur,
                            k,
                            k,
                            None,
                            ref,
                            h_old_a,
                            h_old_b,
                            l1_extra=boundary_ops if k == 0 else 0,
                            r1_extra=boundary_ops if k == 0 else 0,
                            # Tile 1's L1 wait retires the routing loads.
                            hook=keep_route if k == 1 else None,
                        )
                    if const_expr(kb_count > 4):
                        for h_k in range(2, kb_count - 2, 2):
                            h_ki = fx.Int32(h_k)
                            for unroll_idx in range_constexpr(2):
                                ki = h_ki + unroll_idx
                                h_old_a, h_old_b = half_tile(
                                    cur,
                                    ki,
                                    unroll_idx,
                                    None,
                                    (cur, ki + 2),
                                    h_old_a,
                                    h_old_b,
                                )
                    h_nxt = task_context(upcoming, next_base, upcoming_valid)
                    for k in range_constexpr(kb_count - 2, kb_count):
                        _, ref = tile_refs(cur, h_nxt, k)
                        last = k == kb_count - 1
                        # The last tile also fills the next task's tile 0 A bottom
                        # (two more VMEM ops younger than L1's B0 target).
                        h_old_a, h_old_b = half_tile(
                            cur,
                            k,
                            k % 2,
                            (h_nxt, 0) if last else None,
                            ref,
                            h_old_a,
                            h_old_b,
                            l1_extra=2 if last else 0,
                        )
                    h_ids = read_store_ids(route_base)
                    drain(c_tr, h_old_a, h_old_b)
                    rocdl.sched_barrier(0)
                    store_gate_up(cur, c_tl, c_tr, 0, h_ids)
                    store_placeholders()
                    rocdl.sched_barrier(0)

            if wave_id < 4:
                rocdl.s_barrier()
            # The last task's next-task prefetches are out of range; retire them.
            rocdl.s_waitcnt(vmcnt=0)

    @flyc.jit
    def launch_stage1(
        a: fx.Tensor,
        b: fx.Tensor,
        scale_a: fx.Tensor,
        scale_b: fx.Tensor,
        sorted_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out: fx.Tensor,
        num_tokens: fx.Int32,
        num_expert_blocks: fx.Int32,
        num_workers: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fp8_stage1_kernel(
            a,
            b,
            scale_a,
            scale_b,
            sorted_ids,
            expert_ids,
            num_valid_ids,
            out,
            num_tokens,
            num_expert_blocks,
            num_workers,
            # Keep the DOT-phase FP32 FMAs scalar: no v_pk_fma_f32 re-packing.
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(grid=(num_workers, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_stage1


@functools.cache
def _cu_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _compile_stage1(
    device_key, intermediate_size, hidden_size, topk, num_experts, k_rotate
):
    if device_key is None or device_key[2].split(":")[0] != "gfx950":
        raise ValueError("FP8 blockscale GateUp requires gfx950")
    launcher = compile_fp8_stage1_giu1(
        intermediate_size, hidden_size, topk, num_experts, k_rotate=k_rotate
    )
    launcher.compile_hints["opt_level"] = 2
    return launcher


def fp8_blockscale_stage1(
    a,
    weight,
    scale_a,
    scale_b,
    sorted_ids,
    expert_ids,
    num_valid_ids,
    output,
    *,
    k_rotate=None,
    persistent=True,
):
    """Write BF16 [tokens, topk, I] on the input device's current stream.

    scale_a must already be physically K-block-major (Aiter transpose_scale=True).
    Valid routes form a 256-row-aligned prefix; padding encodes invalid token/slot
    IDs and each real (token, slot) occurs once. No host read of valid counts.
    k_rotate selects stage1 K-tile rotation; None uses the heuristic below.
    persistent launches at most one worker per CU; each walks (block, N tile)
    tasks round-robin. Otherwise every task gets its own workgroup.
    """
    if a.ndim != 2 or weight.ndim != 3 or output.ndim != 3:
        raise ValueError("expected A[T,H], weight[E,2I,H], output[T,topk,I]")
    tokens, hidden = a.shape
    experts = weight.shape[0]
    topk, intermediate = output.shape[1:]
    validate_parameters(tokens, intermediate, hidden, topk, experts)
    if not getattr(weight, "is_shuffled", False):
        raise ValueError("stage1 weight must use Aiter shuffle_weight(16,16)")
    if (
        tuple(weight.shape) != (experts, 2 * intermediate, hidden)
        or output.shape[0] != tokens
    ):
        raise ValueError("stage1 weight/output shape mismatch")
    tensors = (
        (a, torch.float8_e4m3fn),
        (weight, torch.float8_e4m3fn),
        (scale_a, torch.float32),
        (scale_b, torch.float32),
        (sorted_ids, torch.int32),
        (expert_ids, torch.int32),
        (num_valid_ids, torch.int32),
        (output, torch.bfloat16),
    )
    for tensor, dtype in tensors:
        if (
            tensor.dtype != dtype
            or not tensor.is_cuda
            or tensor.device != a.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "stage1 tensors must be contiguous, correctly typed and on the same GPU"
            )
    if scale_a.numel() != tokens * (hidden // 128) or scale_b.numel() != experts * (
        2 * intermediate // 128
    ) * (hidden // 128):
        raise ValueError("stage1 scale size mismatch")
    if (
        any(t.ndim != 1 for t in (sorted_ids, expert_ids, num_valid_ids))
        or num_valid_ids.numel() < 1
    ):
        raise ValueError("stage1 sorting metadata must be 1D with a valid count")
    if max(sorted_ids.numel(), expert_ids.numel()) * 4 >= BUFFER_LIMIT:
        raise ValueError("stage1 sorting metadata must fit 31-bit byte offsets")
    # The valid prefix is 256-row aligned and never exceeds the sort capacity, so
    # a trailing partial capacity block is always inactive and need not launch.
    blocks = min(expert_ids.numel(), sorted_ids.numel() // SORT_BLOCK_M)
    # The harness enables K rotation for <= 2 valid blocks per expert. The valid
    # count stays on device, so estimate it from the mean routed rows per expert.
    if k_rotate is None:
        k_rotate = tokens * topk <= 2 * SORT_BLOCK_M * experts
    elif type(k_rotate) is not bool:
        raise ValueError("k_rotate must be None or a bool")
    if type(persistent) is not bool:
        raise ValueError("persistent must be a bool")
    tasks = blocks * (intermediate // BLOCK_N)
    workers = min(tasks, _cu_count(a.device)) if persistent else tasks
    with torch.cuda.device(a.device):
        launcher = _compile_stage1(
            get_device_cache_key(), intermediate, hidden, topk, experts, k_rotate
        )
        if blocks:
            # These are views only: do not transpose the already transposed scale bytes.
            _run_compiled(
                launcher,
                a.view(torch.int8),
                weight.view(torch.int8),
                scale_a.view(hidden // 128, tokens),
                scale_b.view(experts, 2 * intermediate // 128, hidden // 128),
                sorted_ids,
                expert_ids,
                num_valid_ids,
                output.view(-1),
                tokens,
                blocks,
                workers,
                torch.cuda.current_stream(a.device),
            )
    return output
