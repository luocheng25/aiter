# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 Down prefill kernel: the default prefill Down kernel of ``gemm2.py`` with MXFP4 weights.

Each workgroup keeps its BM x I FP8 activation tile (gathered by sorted id) in registers (staged through
LDS in one pass, or in two for I > 512 when the tile exceeds 32 KiB, so that two workgroups fit per CU) and
walks the hidden dimension in 64-row blocks; each wave reads its 16 weight rows in the
``common`` layout, decodes them to FP8, runs MFMA 16x16x32 FP8 and scales by the
channel factor, the activation scale and the routing weight. Rows are written by sorted
position as BF16 for ``sorted_sum``; each workgroup also stores the sorted row of every valid
route of its block into ``loc`` [M, TOPK] (the sort's own inverse output costs more). From
``K`` = 256 on (MFMA-heavier steps) packed FP32 VALU ops are disabled, as in the base Down kernels
(``gemm2_8x1``): measured 1–2.5% faster there, 1–3% slower at ``K`` = 192. With ``NSPLIT`` > 1 the
hidden dimension is split over ``NSPLIT`` workgroups per sorted block (grid x; each gathers the
block's activations again): few sorted blocks then still fill the machine in several rounds.

``TP`` (a ``mega_moe_tp_gfx942.down_tp.DownTP``): the streamed Down of ``MegaMoeTPGfx942`` -- a 1-D grid that
computes the ``NSPLIT`` column blocks one after the other (part-major) and runs the cross-rank sum of the finished
blocks in workgroups of the same launch inserted between the compute workgroups (``DownTP.segments``, ``down_tp``
docstring); route and ``loc`` stores take its cache policy.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..moe_gemm_2stage.gfx9_fp8 import common as fxh
from . import common as a8w4
from .quant import _buffer_rsrc, _store

# Larger K stages the activation tile in two passes when it exceeds 32 KiB (see compile_a8w4_gemm2).
A_TWO_PASS_MIN_K = 512


def compile_a8w4_gemm2(
    *, N, K, TOPK, BLOCK_TILE_SIZE_M=64, E=None, NSPLIT=1, XCD_CHUNK=0, TP=None, CHUNKED=False
):
    """Build the A8W4 Down launcher.

    ``N`` = hidden (``N % 256 == 0``), ``K`` = I (``K % 64 == 0``; pad I in the weights);
    ``BLOCK_TILE_SIZE_M * K`` must be a multiple of 4096 and at most 64 KiB.
    Launch arguments: x FP8 [M, TOPK, K], w and w_delta (``common`` layout, uint8),
    out BF16 [task_num * BLOCK_TILE_SIZE_M, N] by sorted row, sorted_ids, sorted_weights,
    sorted_expert_ids, num_valid_ids, w_scale f32 [E, N], x_scale f32 [M, TOPK], loc int32 [M, TOPK]
    (receives the sorted row of each valid route; entries of routes the sort dropped keep their
    value), then M, task_num, stream.
    ``XCD_CHUNK`` > 0: a 1-D grid ordered as in ``gemm1`` (chunks of ``XCD_CHUNK`` consecutive sorted blocks,
    all their hidden-dimension parts, dealt to the XCDs in turn, so blocks of one expert share its weights in L2).
    ``TP``: the streamed Down (module docstring); the launcher then takes ``out``, ``ctr``, ``tab``, ``epoch``,
    ``err``, M, rank, tpq, limit, task_num, stream after ``loc`` (``down_tp.compile_down_tp``).
    ``CHUNKED``: ``num_valid_ids`` is a token chunk's descriptor row (``chunks``) and the sorted-block positions are
    those of its block list (``common.chunk_block``), task_num its bound (``chunks.chunk_bound``); with ``TP`` its
    roles take the chunk's tokens (``TPArgs.t_lo`` / ``t_hi`` / ``chunk``).
    """
    BM = BLOCK_TILE_SIZE_M
    assert (N // 64) % (
        2 * NSPLIT
    ) == 0, f"Down N={N} must split into {NSPLIT} parts of an even number of 64 rows"
    assert K % 64 == 0, f"Down K={K} must be a multiple of 64"
    assert N % 256 == 0, f"Down N={N} must be a multiple of 256"
    assert 16 <= BM <= 256 and BM % 16 == 0
    assert BM * K <= 64 * 1024, f"Down A tile needs {BM * K} B of LDS"
    assert (
        BM * K % (256 * 16) == 0
    ), f"Down A gather needs BLOCK_TILE_SIZE_M * K={BM * K} % 4096 == 0"
    KB = K // 64
    # The BM x K activation tile goes through LDS in one pass, or for K > A_TWO_PASS_MIN_K with BM * K above 32 KiB in
    # two passes of whole 64-column blocks (K = 576: 5 + 4 blocks, 20 KiB of LDS instead of 36 KiB, two workgroups per
    # CU instead of one); each pass's fragments are read before the next pass overwrites the LDS.
    one_pass = K <= A_TWO_PASS_MIN_K or BM * K <= 32 * 1024
    A_BLOCKS = [KB] if one_pass else [-(-KB // 2), KB // 2]
    A_COLS = [64 * b for b in A_BLOCKS]
    A_K0 = [64 * sum(A_BLOCKS[:p]) for p in range(len(A_BLOCKS))]
    A_PART = [p for p, b in enumerate(A_BLOCKS) for _ in range(b)]
    assert all(
        BM * c % (256 * 16) == 0 for c in A_COLS
    ), f"Down A gather passes {A_COLS} need BM * cols % 4096 == 0"
    KQ = a8w4.delta_dwords(K)
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    fp8 = fx.Float8E4M3FNUZ
    flyobj = fxh.FlyObjCache()
    no_packed = K >= 256
    features_text = (
        '#llvm.target_features<["-packed-fp32-ops"]>'
        if no_packed
        else "#llvm.target_features<[]>"
    )
    CHUNK = XCD_CHUNK * NSPLIT  # workgroups of one XCD chunk

    def _part(part):
        """This workgroup's part of the hidden dimension (grid x without XCD_CHUNK and TP)."""
        return (
            part
            if XCD_CHUNK > 0 or TP is not None
            else fx.Int32(fx.gpu.block_idx.x)
        )

    def _chunk_tokens(p_desc):
        """TP roles of a ``CHUNKED`` launch: the chunk's tokens ``[desc[1], desc[2])`` and its index ``desc[3]``."""
        desc = fxh.view_as_torch_tensor(p_desc, (4,), fx.Int32)
        return dict(t_lo=fx.Int32(desc[1]), t_hi=fx.Int32(desc[2]), chunk=fx.Int32(desc[3]))

    def _rne_bits(value):
        bits = value.bitcast(fx.Uint32)
        return bits + ((bits >> 16) & fx.Uint32(1)) + fx.Uint32(0x7FFF)

    def _pack_bf16_rne(values):
        # Finite f32 -> BF16 round-to-nearest-even; one v_perm keeps the high halves of a pair.
        words = [
            a8w4._perm(
                _rne_bits(values[2 * i + 1]), _rne_bits(values[2 * i]), 0x07060302
            )
            for i in range_constexpr(values.numel // 2)
        ]
        return Vec.from_elements(words, fx.Uint32).bitcast(fx.BFloat16)

    @(flyc.kernel if TP is None else flyc.kernel(name="a8w4_down_prefill_tp"))
    def a8w4_down_prefill(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_loc: fx.Pointer,
        M: fx.Int32,
        p_out: fx.Pointer,
        p_ctr: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_err: fx.Pointer,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        task_pad: fx.Int32,
    ):
        # p_out .. task_pad: arguments of TP only.
        tid = fx.gpu.thread_idx.x
        if const_expr(TP is not None):
            if const_expr(TP.TRACE):
                TP.trace(p_ctr, task_pad, 0)
            role, idx, part = TP.roles(fx.Int32(fx.gpu.block_idx.x), task_pad)
            if const_expr(XCD_CHUNK > 0):
                step = idx // a8w4.XCDS
                e_idx = (
                    (step // XCD_CHUNK) * a8w4.XCDS + idx % a8w4.XCDS
                ) * XCD_CHUNK + step % XCD_CHUNK
            else:
                e_idx = idx
        elif const_expr(XCD_CHUNK > 0):
            wg = fx.Int32(fx.gpu.block_idx.x)
            step = wg // a8w4.XCDS
            work = ((step // CHUNK) * a8w4.XCDS + wg % a8w4.XCDS) * CHUNK + step % CHUNK
            part = work % NSPLIT
            e_idx = work // NSPLIT
        else:
            e_idx = fx.gpu.block_idx.y
            part = e_idx  # unused: _part reads grid x
        if const_expr(CHUNKED):
            pos = e_idx
            if const_expr(TP is not None):
                # The reduce / owner workgroups read position 0 (their idx may pass the chunk's bound).
                pos = (role == 0).select(e_idx, fx.Int32(0))
            active, e_idx = a8w4.chunk_block(p_num_valid_ids, pos)
        else:
            max_valid_id = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
            active = e_idx * BM < max_valid_id
        if const_expr(TP is not None):
            active = (role == 0) & active

        if active:
            arg_p_input = fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fp8)
            arg_p_output = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(e_idx) * (BM * N), (BM, N)
            )
            arg_p_sorted_ids = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_ids) + e_idx * BM, (BM,), fx.Int32
            )
            arg_p_sorted_weights = fxh.view_as_torch_tensor(
                fxh._as_ptr(p_sorted_weights) + e_idx * BM, (BM,), fx.Float32
            )
            expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[
                e_idx
            ]
            w_buf = a8w4.expert_bytes(p_weight, expert_id, W_BYTES)
            d_buf = a8w4.expert_bytes(p_w_delta, expert_id, D_BYTES)
            arg_p_output = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)

            BLOCK_N = 64
            BLOCK_K = 64
            swz = fx.SwizzleType.get(3, 4, 3)

            @fx.union
            class SharedStorage:
                A: fx.Array[fp8, BM * max(A_COLS)]
                C: fx.Array[fx.BFloat16, 2 * BM * BLOCK_N]

            lds = fx.SharedAllocator().allocate(SharedStorage)

            def lds_A(cols):
                return lds.A.peek().view(
                    fx.make_composed_layout(fx.static(swz), fxh.torch_layout(BM, cols))
                )

            ldsA0 = lds_A(A_COLS[0])

            arg_p_input = fx.rocdl.make_buffer_tensor(
                arg_p_input,
                max_size=False,
                num_records_bytes=fx.Int64(M) * (TOPK * K),
            )
            cp_atom = flyobj.get_buffer_copy_atom(arg_p_input.dtype, 128)

            def flatten_A(x):
                x = fx.select(x, [1, 0])
                return fx.group(x, 0, -1)

            def gather_A(part, k0):
                """Columns [k0, k0 + width of ``part``) of the block's BM activation rows -> ``part``, then a barrier."""
                cp_ldsA0 = flatten_A(part)
                cp_rows = flatten_A(
                    fxh.make_1d_coord_tensor(part, 0, fx.get_iter(arg_p_sorted_ids))
                )
                cp_cols = flatten_A(
                    fxh.make_1d_coord_tensor(part, 1, fx.make_int_tuple(k0))
                )
                for dst, row, col in fxh.all_copy_atoms(
                    cp_ldsA0, cp_rows, cp_cols, atom_bits=128, num_threads=256
                ):
                    sorted_id = row[0].bitcast(fx.Uint32)
                    atom_A = fxh.atom_tensor(
                        arg_p_input,
                        (sorted_id & 0xFFFFFF, sorted_id >> 24, col[0]),
                        128,
                    )
                    fx.copy(cp_atom, atom_A, dst)
                fx.gpu.barrier()

            gather_A(ldsA0, 0)

            nBN = N // BLOCK_N // NSPLIT
            # First 64-row block of this workgroup's part of the hidden dimension.
            n0 = _part(part) * nBN if NSPLIT > 1 else 0
            # Stride-0 N mode spans one MMA N tile (16): FlyDSL >= 0.5.3 rejects narrower C views.
            arg_w_scale = fx.flat_divide(
                fx.make_view(
                    fxh._as_ptr(p_w_scale) + expert_id * N,
                    fx.make_layout((N, 16), (1, 0)),
                ),
                (BLOCK_N, 16),
            )
            arg_a_scale = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)),
                    fx.make_layout((M, TOPK), (TOPK, 1)),
                ),
                max_size=False,
                num_records_bytes=fx.Int64(M) * TOPK * 4,
            )

            def body(bm, ops):
                # Computes the first bm sorted rows; ops is a fresh builder cache per variant.
                ldsA = fx.flat_divide(ldsA0, (bm, BLOCK_K))
                layoutC = fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((bm, BLOCK_N, 2), (1, 0, 2))
                )
                layoutCt = fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((BLOCK_N, bm, 2), (0, 1, 2))
                )
                ldsC = lds.C.peek().view(layoutC)
                ldsCt = lds.C.peek().view(layoutCt)
                mm = ops.create_thr_mma(fp8, (4, 1, 1))

                c_fake_tensor = fx.make_view(
                    fx.get_iter(arg_p_input),
                    fx.make_ordered_layout((BLOCK_N, bm), (0, 1)),
                )
                fragC = [
                    mm.make_fragment_C(c_fake_tensor),
                    mm.make_fragment_C(c_fake_tensor),
                ]
                fragC_bf16 = fx.make_fragment_like(fragC[0], fx.BFloat16)

                frag_act = [
                    ops.load_tiled_mma_fragB(
                        mm, ldsA, [None, None, 0, None], copy_atom_bits=128
                    )
                ]
                for p in range_constexpr(1, len(A_COLS)):
                    fx.gpu.barrier()  # Every wave has read the previous part before it is overwritten.
                    part = lds_A(A_COLS[p])
                    gather_A(part, A_K0[p])
                    frag_act.append(
                        ops.load_tiled_mma_fragB(
                            mm,
                            fx.flat_divide(part, (bm, BLOCK_K)),
                            [None, None, 0, None],
                            copy_atom_bits=128,
                        )
                    )
                fx.gpu.barrier()  # Finish reading ldsA before ldsC reuses its storage.

                sorted_weights = fx.make_view(
                    fx.get_iter(arg_p_sorted_weights),
                    fx.make_layout((BLOCK_N, bm), (0, 1)),
                )
                frag_sorted_weight = ops.load_tiled_mma_fragC(
                    mm, sorted_weights, copy_atom_bits=32
                )
                sc_atom = ops.get_buffer_copy_atom(p_a_scale.dtype, 32)
                coord_tensor = fx.make_view(
                    fx.get_iter(arg_p_sorted_ids), fx.make_layout((BLOCK_N, bm), (0, 1))
                )
                frag_coord = ops.load_tiled_mma_fragC(
                    mm, coord_tensor, copy_atom_bits=32
                )
                frag_row_scale = mm.make_fragment_C(coord_tensor)
                frag_row_scale_r = ops.get_tiled_mma_retile(
                    mm, frag_row_scale, "C", copy_atom=sc_atom
                )
                for dst, coord in fxh.all_elements(frag_row_scale_r, frag_coord):
                    sorted_id = coord[0].bitcast(fx.Uint32)
                    atom = fxh.atom_tensor(
                        arg_a_scale, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32
                    )
                    fx.copy(sc_atom, atom, dst)
                for frag_pt, frag_sw in fxh.all_elements(
                    frag_row_scale, frag_sorted_weight
                ):
                    frag_pt.store(frag_pt.load() * frag_sw.load())

                # Decoded FP8 weight operand of one 64-row block: (8, 1, 2, KB).
                w_fake = fx.make_view(
                    fx.get_iter(arg_p_input),
                    fx.make_layout((BLOCK_N, BLOCK_K, KB), (K, 1, BLOCK_K)),
                )
                frag_w = mm.make_fragment_A(w_fake)
                w_voff, d_voff = a8w4.lane_offsets(tid, K)
                w_regs = [
                    [
                        fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Uint32)
                        for _ in range(KB)
                    ]
                    for _ in range(2)
                ]
                d_regs = [
                    [
                        fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Uint32)
                        for _ in range(KQ)
                    ]
                    for _ in range(2)
                ]
                frag_pc_scales = [None, None]

                def load_block(n, slot):
                    rb = (n0 + n) * (BLOCK_N // 16)
                    for kb in range_constexpr(KB):
                        w_regs[slot][kb].store(
                            a8w4.load_dwordx2(
                                w_buf,
                                w_voff,
                                fx.Int32((rb * KB + kb) * a8w4.W_BLOCK_BYTES),
                            )
                        )
                    for q in range_constexpr(KQ):
                        d_regs[slot][q].store(
                            a8w4.load_dword(
                                d_buf,
                                d_voff,
                                fx.Int32((rb * KQ + q) * a8w4.D_BLOCK_BYTES),
                            )
                        )
                    if const_expr(frag_pc_scales[slot] is None):
                        frag_pc_scales[slot] = ops.load_tiled_mma_fragC(
                            mm, arg_w_scale, [None, None, n0 + n, 0], copy_atom_bits=128
                        )
                    else:
                        ops.load_tiled_mma_fragC(
                            mm,
                            arg_w_scale,
                            [None, None, n0 + n, 0],
                            frag_pc_scales[slot],
                        )

                def gemm_compute(slot, fragC):
                    for kb in range_constexpr(KB):
                        lut = a8w4.fp8_lut(
                            d_regs[slot][kb // 4].load()[0],
                            fx.Uint32((kb % 4) * 0x01010101),
                        )
                        raw = w_regs[slot][kb].load()
                        for c in range_constexpr(2):
                            frag_w[None, 0, c, kb].store(a8w4.fp4x8_to_fp8(raw[c], lut))
                    fragC.fill(0)
                    for k in range_constexpr(KB):
                        act = frag_act[A_PART[k]][
                            None, None, None, k - A_K0[A_PART[k]] // BLOCK_K
                        ]
                        fx.gemm(mm, fragC, frag_w[None, None, None, k], act, fragC)
                    for fc, fpc in fxh.all_elements(fragC, frag_pc_scales[slot]):
                        fc.store(fc.load() * fpc.load())

                tcopyLDS, cp_ldsc = ops.get_tiled_copy_coalesced_mn(
                    ldsC[None, None, 0], copy_atom_bits=128, num_threads=256
                )
                thrv_ldsC = tcopyLDS.partition_S(ldsC)
                copy_atom_ = ops.get_universal_copy_atom(fragC_bf16.dtype, 64)
                tcopy = ops.get_tiled_mma_copy(copy_atom_, mm, "C")
                fragC_bf16r = ops.get_retile(tcopy, fragC_bf16)
                thrv_ldsCt = ops.get_partition_D(tcopy, ldsCt)

                def postprocess_store2lds(fragC, ldsc_idx):
                    for fc, fsw in fxh.all_elements(fragC, frag_row_scale):
                        fc.store(fc.load() * fsw.load())
                    fragC_bf16.store(_pack_bf16_rne(fragC.load()))
                    fx.copy(
                        copy_atom_, fragC_bf16r, thrv_ldsCt[None, None, None, ldsc_idx]
                    )

                out_tiles = fx.flat_divide(arg_p_output, (bm, BLOCK_N))
                if const_expr(TP is not None):
                    cp_atom_out_128b = fx.make_copy_atom(
                        fx.rocdl.BufferCopy(128, TP.ROUTE_AUX), fx.BFloat16
                    )
                else:
                    cp_atom_out_128b = ops.get_buffer_copy_atom(fx.BFloat16, 128)
                thrv_out = tcopyLDS.partition_D(out_tiles)
                fragOut = fx.make_fragment_like(thrv_ldsC[None, None, None, 0])

                def postprocess_store2vmem(n, ldsc_idx):
                    fx.copy(cp_ldsc, thrv_ldsC[None, None, None, ldsc_idx], fragOut)
                    fx.copy(
                        cp_atom_out_128b, fragOut, thrv_out[None, None, None, 0, n0 + n]
                    )

                load_block(0, 0)
                gemm_compute(0, fragC[0])
                load_block(1, 1)
                postprocess_store2lds(fragC[0], 0)
                fx.gpu.barrier()
                for n, state in range(0, nBN - 2, 2, init=[]):
                    postprocess_store2vmem(n, 0)
                    load_block(n + 2, 0)
                    gemm_compute(1, fragC[1])
                    postprocess_store2lds(fragC[1], 1)
                    fx.gpu.barrier()

                    postprocess_store2vmem(n + 1, 1)
                    load_block(n + 3, 1)
                    gemm_compute(0, fragC[0])
                    postprocess_store2lds(fragC[0], 0)
                    fx.gpu.barrier()

                postprocess_store2vmem(nBN - 2, 0)
                gemm_compute(1, fragC[1])
                postprocess_store2lds(fragC[1], 1)
                fx.gpu.barrier()
                postprocess_store2vmem(nBN - 1, 1)

            if const_expr(BM % 64 == 0):
                # Valid rows lead a sorted block: when row BM/2 is padding, BM/2 rows suffice.
                half_id = fx.Int32(
                    rocdl.readfirstlane(T.i32, arg_p_sorted_ids[BM // 2])
                )
                if (half_id & 0xFFFFFF) >= M:
                    body(BM // 2, fxh.FlyObjCache())
                else:
                    body(BM, fxh.FlyObjCache())
            else:
                body(BM, flyobj)

            # Sorted row of each valid route (padding rows store nothing). Stored after the main loop:
            # at the start the store made the whole kernel ~30% slower (Qwen TP2).
            row = fx.Int32(tid) % BM
            sid = arg_p_sorted_ids[row].bitcast(fx.Uint32)
            token = fx.Int32(sid & fx.Uint32(0xFFFFFF))
            route = (token * TOPK + fx.Int32(sid >> 24)) * 4
            past = fx.Int32(0x7FFF0000)
            first = fx.Int32(tid) < BM
            if const_expr(NSPLIT > 1):
                first = (_part(part) == 0).select(fx.Int32(tid), fx.Int32(BM)) < BM
            _store(
                _buffer_rsrc(p_loc, fx.Int64(M) * (TOPK * 4)),
                fx.Int32(e_idx) * BM + row,
                first.select((token < M).select(route, past), past),
                0 if TP is None else TP.ROUTE_AUX,
            )

        if const_expr(TP is not None):
            TP.finish(
                role,
                idx,
                part,
                TP.args(
                    p_loc=p_loc,
                    p_routes=p_output,
                    p_out=p_out,
                    p_ctr=p_ctr,
                    p_tab=p_tab,
                    p_epoch=p_epoch,
                    p_err=p_err,
                    M=M,
                    rank=rank,
                    tpq=tpq,
                    limit=limit,
                    task_pad=task_pad,
                    **(_chunk_tokens(p_num_valid_ids) if CHUNKED else {}),
                ),
            )

    @flyc.jit
    def launch(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_loc: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        features = ir.Attribute.parse(features_text)
        if const_expr(XCD_CHUNK > 0):
            per = a8w4.XCDS * XCD_CHUNK
            grid = ((task_num + (per - 1)) // per * (a8w4.XCDS * CHUNK), 1, 1)
        else:
            grid = (NSPLIT, task_num, 1)
        a8w4_down_prefill(
            p_input,
            p_weight,
            p_w_delta,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            p_loc,
            M,
            p_loc,
            p_loc,
            p_loc,
            p_loc,
            p_loc,
            M,
            M,
            M,
            M,
            value_attrs={"llvm.target_features": features},
        ).launch(grid=grid, block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_tp(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_w_delta: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        p_loc: fx.Pointer,
        p_out: fx.Pointer,
        p_ctr: fx.Pointer,
        p_tab: fx.Pointer,
        p_epoch: fx.Pointer,
        p_err: fx.Pointer,
        M: fx.Int32,
        rank: fx.Int32,
        tpq: fx.Int32,
        limit: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        features = ir.Attribute.parse(features_text)
        # Compute workgroups per stage: whole XCD chunks (the stage bases stay multiples of the XCD count).
        per = a8w4.XCDS * max(XCD_CHUNK, 1)
        task_pad = (task_num + (per - 1)) // per * per
        a8w4_down_prefill(
            p_input,
            p_weight,
            p_w_delta,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            p_loc,
            M,
            p_out,
            p_ctr,
            p_tab,
            p_epoch,
            p_err,
            rank,
            tpq,
            limit,
            task_pad,
            value_attrs={"llvm.target_features": features},
        ).launch(grid=(TP.grid(task_pad), 1, 1), block=(256, 1, 1), stream=stream)

    if no_packed:
        launch.compile_hints["target_features"] = "-packed-fp32-ops"
        launch_tp.compile_hints["target_features"] = "-packed-fp32-ops"
    if TP is not None and TP.WPE:
        launch_tp.compile_hints["waves_per_eu"] = TP.WPE
    return launch if TP is None else launch_tp
