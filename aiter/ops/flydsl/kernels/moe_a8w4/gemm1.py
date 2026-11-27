# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 Gate/Up prefill kernel: the prefill_1x4 kernel of ``gemm1.py`` with MXFP4 weights.

FP8 activations are gathered by sorted id into double-buffered LDS exactly as in
``gemm1.py``; the weight operand is read from the ``common`` layout 16 bytes per lane (two
adjacent rows of a 16-row block per load), decoded to FP8 in registers and fed to MFMA 16x16x32
FP8. The per-channel factor and the per-token activation scale are applied in the epilogue (then the
SwiGLU limit, if any), then SiLU(gate) * up is written as float32 by (token, slot): the intermediate is
quantized from f32 so its FP8 rounding matches the reference.

Tile: a CTA (4 waves) owns one sorted block: its valid rows lead the block, so it computes the
first TILE_M rows only (BM 64: the valid rows rounded up to 16; other multiples of 64: BM/2 when
row BM/2 is padding, else BM), still loading ``A`` in whole copy passes (32 rows at TILE_K 128).
It owns BN/2 gate + BN/2 up channels; wave w owns 16-row gate and up blocks w + 4m. MFMA M =
channels (operand ``W``, decoded weights), N = rows (operand ``A``, activations from LDS),
accumulators ``c_even`` / ``c_odd`` in VGPRs.

Weight lanes: lane (g, i) reads rows 2 * (i % 8) and + 1 of the wave's gate (i < 8) or up
(i >= 8) block, so lanes 0..7 / 8..15 of a lane group read 128 contiguous bytes each and one
16-byte load replaces a gate and an up 8-byte load. MFMA p takes row p of every lane's pair: its
row R is channel 2 * (R % 8) + p of the gate (R < 8) or up (R >= 8) block. The epilogue trades
accumulators between lanes l and l ^ 32 so each lane holds gate and up of 4 consecutive channels.

Occupancy: BN 128 / TILE_K 128 pins ``rocdl.waves_per_eu`` = 4 (128 VGPRs, a few spilled); left
alone the register allocator settles at 132 VGPRs and 3 waves per SIMD, which is slower.

K step (tile k of TILE_K = 64 * k_iters, ``buf = k % 2``; weights in VGPR slot ``k % 2``, staged
A in slot ``k % A_AHEAD``):

    region  MFMA            LDS -> VGPR   global -> VGPR    decode      VGPR -> LDS
    0       C += W A(k,0)   A(k,0..1)     A(k+A_AHEAD)      W(k,1)
    i       C += W A(k,i)   A(k,i+1)                        W(k,i+1)
    last    C += W A(k,l)                 W(k+2)            W(k+1,0)    A(k+1) -> buf^1

Pipeline rules:
1. ``W(k+2)`` reuses tile k's slot, so it is issued in the last region, after ``W(k,l)`` is
   decoded; the decode of ``W(k+1,0)`` then waits on loads issued a whole stage earlier.
2. ``A_AHEAD`` = 2 for K <= 2560: ``A(k+2)`` is issued first in stage k into A slot k % 2 (``A(k)``
   left it for LDS at the end of stage k-1) and written to LDS at the end of stage k+1; larger K
   keeps one slot (``A(k+1)`` issued first and written at the end of the same stage). ``vmcnt``
   leaves the younger loads in flight; the closing barrier orders that write before the next
   stage's reads of ``buf^1`` and this stage's reads of ``buf`` before their refill.
3. The prologue loads ``A(0)``, ``W(0)``, waits, then issues ``W(1)`` (and ``A(1)``); the last
   ``A_AHEAD`` stages issue no A, the last one writes none and the one before it no ``W(k+2)``.
4. With TILE_K 128 tiles 2j and 2j+1 share one 256-K delta dwordx2 (the lane's two rows): only
   even tiles load it, and issuing ``W(2j+1)`` (a stage after ``W(2j)``, before slot 0 is
   refilled) copies it from slot 0.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..moe_gemm_2stage.gfx9_fp8 import common as fxh
from ..moe_gemm_2stage.gfx9_fp8.common import make_gemm_helpers
from . import common as a8w4


def compile_a8w4_gemm1(
    *,
    N,
    K,
    TOPK,
    BLOCK_TILE_SIZE_M=64,
    BLOCK_TILE_SIZE_N=128,
    TILE_K=256,
    E=None,
    SWIGLU_LIMIT=0.0,
    XCD_CHUNK=0,
    CHUNKED=False,
):
    """Build the A8W4 Gate/Up launcher.

    ``N`` = 2 * I (weight rows ``[0, I)`` gate, ``[I, 2I)`` up; ``I % (BLOCK_TILE_SIZE_N/2) == 0``),
    ``K`` = hidden (``K % (2 * TILE_K) == 0``). ``BLOCK_TILE_SIZE_M`` must equal the sorting block.
    Launch arguments: x FP8 [M, K], w and w_delta (``common`` layout, uint8), out f32
    [M, TOPK, I] (``M * TOPK * I * 4 < 2**31``), sorted_ids, sorted_weights (unused),
    sorted_expert_ids, num_valid_ids, w_scale f32 [E, N], x_scale f32 [M], then M, task_num, stream.
    ``SWIGLU_LIMIT`` > 0 clamps the dequantized gate and up before SiLU (``common.swiglu_clamp``).
    ``XCD_CHUNK`` > 0: a 1-D grid whose workgroup ``L`` (dispatched to XCD ``L % XCDS``) takes the next
    (sorted block, N tile) of chunks of ``XCD_CHUNK`` consecutive blocks dealt to the XCDs in turn, so the N tiles
    of a block (same activation rows) and neighbouring blocks of an expert (same weights) share an L2.
    ``CHUNKED``: ``num_valid_ids`` is a token chunk's descriptor row (``chunks``) and the (sorted block, N tile)
    positions are those of its block list (``common.chunk_block``), task_num its bound (``chunks.chunk_bound``).
    """
    BM, BN = BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N
    assert BN in (128, 256), "BLOCK_TILE_SIZE_N must be 128 or 256"
    assert (N // 2) % (BN // 2) == 0, f"I={N // 2} must be a multiple of {BN // 2}"
    assert TILE_K in (128, 256), f"TILE_K must be 128 or 256, got {TILE_K}"
    assert K % (2 * TILE_K) == 0, f"K={K} needs an even number of {TILE_K}-wide tiles"
    assert (
        32 <= BM <= 256 and BM % 32 == 0
    ), f"BLOCK_TILE_SIZE_M={BM} must be 32..256, step 32"
    a_lds_size = BM * TILE_K
    assert 2 * a_lds_size <= 64 * 1024, f"A ping-pong needs {2 * a_lds_size} B of LDS"

    KB = K // 64
    KQ = a8w4.delta_dwords(K)
    KB_PER_TILE = TILE_K // 64
    A_AHEAD = 2 if K <= 2560 else 1  # rule 2
    WAVES_PER_EU = 4 if BN == 128 and TILE_K == 128 else None  # occupancy
    W_BYTES = a8w4.weight_bytes(N, K)
    D_BYTES = a8w4.delta_bytes(N, K)
    fp8 = fx.Float8E4M3FNUZ
    NUM_N_BLOCKS = (N + BN - 1) // BN
    CHUNK = XCD_CHUNK * NUM_N_BLOCKS  # workgroups of one XCD chunk

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[fp8, a_lds_size, 16]
        a_pong: fx.Array[fp8, a_lds_size, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers

    TensorWithIndex, _read_sorted_index, _ = make_gemm_helpers(K, fp8, BM, TOPK)

    def _chan_base(lane):
        # First of the 4 channels (of the wave's 16) lane group G = 0, 1, 2, 3 holds after _unpair.
        return ((lane // 16) % 2) * 8 + (lane // 32) * 4

    def _store_silu_f32(
        gate_frag, up_frag, tid, blk_n, contiguous_n, asc_idx, M, p_output
    ):
        # Lane holds channels _chan_base + v of token lane % 16 for each (m, n) MMA tile.
        log2_exp1 = -1.4426950408889634
        INTER = N // 2
        lane = tid % 64
        wave = tid // 64
        oob = M * (TOPK * INTER * 4)
        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fx.Float32, fxh._as_ptr(p_output)), fx.make_layout(1, 1)
            ),
            max_size=False,
            num_records_bytes=fx.Int64(oob),
        )
        channel = blk_n * contiguous_n + wave * 16 + _chan_base(lane)
        m_reps = fx.size(fx.get_shape(gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(gate_frag)[2]).to_py_value()
        for n in range_constexpr(n_reps):
            sorted_id = asc_idx[0, n]
            token = sorted_id & 0xFFFFFF
            slot = (sorted_id >> 24) & 0xFF
            row_bytes = (token * TOPK + slot) * (INTER * 4)
            for m in range_constexpr(m_reps):
                gate = gate_frag[None, m, n].load()
                up = up_frag[None, m, n].load()
                acc = []
                for j in range_constexpr(gate.numel):
                    tmp = rocdl.exp2(T.f32, _raw(gate[j] * log2_exp1))
                    acc.append((gate[j] * rocdl.rcp(T.f32, 1.0 + tmp)) * up[j])
                offset = row_bytes + (channel + m * 64) * 4
                # Padding rows carry token id >= M: redirect them past the buffer end.
                offset = (token < M).select(offset, oob)
                a8w4.store_vector(
                    out, Vec.from_elements(acc, fx.Float32), offset, fx.Int32(0)
                )

    def _unpair(c_even, c_odd, tid):
        # Row 4G + v of MFMA p is channel 8 * (G % 2) + 2v + p of the wave's gate (G < 2) or up
        # (G >= 2) block: lanes l and l ^ 32 trade raw accumulators so that the low lane holds gate
        # and up of channels 0..3 of those 8 and the high lane 4..7 (_chan_base).
        lane = tid % 64
        is_up = lane >= 32
        partner = (lane ^ 32) * 4
        m_reps = fx.size(fx.get_shape(c_even)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_even)[2]).to_py_value()
        for m in range_constexpr(m_reps):
            for n in range_constexpr(n_reps):
                ev = c_even[None, m, n].load()
                od = c_odd[None, m, n].load()
                vals = [ev[0], od[0], ev[1], od[1], ev[2], od[2], ev[3], od[3]]
                gate, up = [], []
                for k in range_constexpr(4):
                    keep = is_up.select(vals[4 + k], vals[k])
                    send = is_up.select(vals[k], vals[4 + k])
                    got = rocdl.ds_bpermute(
                        T.i32, _raw(partner), _raw(send.bitcast(fx.Int32))
                    )
                    got = fx.Int32(got).bitcast(fx.Float32)
                    gate.append(is_up.select(got, keep))
                    up.append(is_up.select(keep, got))
                c_even[None, m, n].store(Vec.from_elements(gate, fx.Float32))
                c_odd[None, m, n].store(Vec.from_elements(up, fx.Float32))

    def _make_1x4_tiled_mma():
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_1x4(TILE_M, TILE_N, blk_n: int, arg_p_input: fx.Tensor, w_buf, d_buf, lds):
        tid = gpu.thread_idx.x
        contiguous_n = TILE_N // 2

        a_tensor = fx.rocdl.make_buffer_tensor(arg_p_input, max_size=False)
        mma_atom, tiled_mma = _make_1x4_tiled_mma()

        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        _val_per_thr = 16
        _thrs_k = TILE_K // _val_per_thr
        _thrs_m = 256 // _thrs_k
        # Rows of A loaded and stored to LDS: whole copy passes covering the TILE_M computed rows.
        TILE_M_LD = -(-TILE_M // _thrs_m) * _thrs_m
        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input), fx.make_layout((TILE_M_LD, K), (K, 1))
            ),
            max_size=False,
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(TILE_M_LD, TILE_K))[
            None, None, 0, None
        ]
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(
            buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        _m_per_wave = _thrs_m // 4
        tiled_copy_sortid_a = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = _read_sorted_index(
            tiled_copy_sortid_a, tid, lds.sorted_lds, index_size=TILE_M_LD
        )
        a_idx = TensorWithIndex(
            a_tensor, TILE_M_LD, TILE_K, a_index_frag, a_mem_cp_g2r, tid
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = [
            fx.make_fragment_like(a_mem_thr[None, None, None, 0])
            for _ in range(A_AHEAD)
        ]

        # All indices must be captured before a_ping reuses sorted_lds.
        gpu.barrier()

        swz = fx.SwizzleType.get(3, 4, 3)

        def _tile_view(buf, rows):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz), fx.make_ordered_layout((rows, TILE_K), order=(1, 0))
                ),
            )

        a_ping = _tile_view(lds.gemm.a_ping, TILE_M_LD)
        a_pong = _tile_view(lds.gemm.a_pong, TILE_M_LD)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)
        )
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = [a_r2s.get_slice(tid).retile(frag) for frag in a_cp_frag]

        # MMA reads see K permuted within each 128-K step so that lane group g reads 32-K
        # group g of the step in both 64-K halves h (one FP8 LUT per lane and step):
        # MMA-side K 64*h + 16*g + 8*c + v -> stored K 32*g + 16*h + 8*c + v.
        k_shape = (8, 2, 4, 2, TILE_K // 128)
        k_stride = (1, 8, 32, 16, 128)

        def _read_view(buf):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_layout((TILE_M, k_shape), (TILE_K, k_stride)),
                ),
            )

        a_lds_r = [
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(_read_view(lds.gemm.a_ping)),
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma)
            .get_slice(tid)
            .partition_S(_read_view(lds.gemm.a_pong)),
        ]
        a_frag = tiled_mma.make_fragment_B(_tile_view(lds.gemm.a_ping, TILE_M))
        a_frag_retile = (
            fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag)
        )

        w_fake = fx.make_view(
            fx.get_iter(arg_p_input),
            fx.make_layout((contiguous_n, TILE_K), (TILE_K, 1)),
        )
        # Decoded FP8 weight operands of the current tile per MFMA p, each (8, m_reps, (2, k_iters)).
        w_frag = [tiled_mma.make_fragment_A(w_fake), tiled_mma.make_fragment_A(w_fake)]

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.get_iter(arg_p_input),
                fx.make_layout((contiguous_n, TILE_M), (TILE_M, 1)),
            ),
            max_size=False,
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, TILE_M))[
            None, None, 0, 0
        ]
        c_even = tiled_mma.make_fragment_C(c_fake)
        c_odd = tiled_mma.make_fragment_C(c_fake)
        c_even.fill(0)
        c_odd.fill(0)

        num_tiles = K // TILE_K
        _m_reps = fx.size(fx.get_shape(c_even)[1]).to_py_value()
        _n_reps = fx.size(fx.get_shape(c_even)[2]).to_py_value()
        k_iters = KB_PER_TILE

        # Lane (g, i) of wave w: 64-K block g // 2 of the step, layout lane group 2 * (g % 2) + h for
        # half h (+128 B), rows 2 * (i % 8) and + 1 of the gate (i < 8) or up (i >= 8) block; delta
        # half g % 2 and byte 2 * (step % 2) + g // 2 of each row's dword.
        lane = tid % 64
        lane_group = lane // 16
        wave = tid // 64
        up_lane = (lane % 16) // 8
        w_voff = fx.Int32(
            (lane_group // 2) * a8w4.W_BLOCK_BYTES
            + (lane_group % 2) * 256
            + (lane % 8) * 16
            + up_lane * ((N // 32) * KB * a8w4.W_BLOCK_BYTES)
            + wave * (KB * a8w4.W_BLOCK_BYTES)
        )
        d_voff = fx.Int32(
            (lane_group % 2) * 64
            + (lane % 8) * 8
            + up_lane * ((N // 32) * KQ * a8w4.D_BLOCK_BYTES)
            + wave * (KQ * a8w4.D_BLOCK_BYTES)
        )
        lut_sel = [fx.Uint32((lane_group // 2 + 2 * v) * 0x01010101) for v in range(2)]
        # FP8 LUT of the latest decoded step per [row p][m] (reused by its second half).
        lut_regs = [
            [
                fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Uint32)
                for _ in range(_m_reps)
            ]
            for _ in range(2)
        ]
        # Raw FP4 dwords (row 2j: 0..1, row 2j+1: 2..3) per [slot][m][ki] and the two rows' delta
        # dwords per [slot][m]; slot = tile parity.
        w_regs = [
            [
                [
                    fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Uint32)
                    for _ in range(k_iters)
                ]
                for _ in range(_m_reps)
            ]
            for _ in range(2)
        ]
        d_regs = [
            [
                fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Uint32)
                for _ in range(_m_reps)
            ]
            for _ in range(2)
        ]

        def load16(buffer, voff, soff):
            packet = fxh.BufferTensor(
                fx.make_view(fx.get_iter(buffer), fx.make_layout(16, 1))
            )
            return packet.load(voffset_bytes=voff, soffset_bytes=soff).bitcast(
                fx.Uint32
            )

        def row_block16(m):
            return blk_n * (contiguous_n // 16) + m * 4

        def load_weights(slot, k_tile):
            kb0 = k_tile * KB_PER_TILE
            for m in range_constexpr(_m_reps):
                rb = row_block16(m)
                for ki in range_constexpr(k_iters):
                    # Half ki % 2 of the 128-K step starting at 64-K block kb0 + ki - ki % 2.
                    w_regs[slot][m][ki].store(
                        load16(
                            w_buf,
                            w_voff,
                            fx.Int32(
                                (rb * KB + kb0 + ki - ki % 2) * a8w4.W_BLOCK_BYTES
                                + (ki % 2) * 128
                            ),
                        )
                    )
                if const_expr(KB_PER_TILE == 2 and slot == 1):
                    d_regs[1][m].store(d_regs[0][m].load())
                else:
                    d_regs[slot][m].store(
                        a8w4.load_dwordx2(
                            d_buf,
                            d_voff,
                            fx.Int32((rb * KQ + kb0 // 4) * a8w4.D_BLOCK_BYTES),
                        )
                    )

        def decode_weights(slot, parity, ki):
            # parity = tile index % 2 (static: ping tiles are even); the first half of a
            # 128-K step builds the LUTs, the second half reuses them. Row p feeds MFMA p.
            step_parity = ((parity * KB_PER_TILE + ki) // 2) % 2
            for m in range_constexpr(_m_reps):
                if const_expr(ki % 2 == 0):
                    deltas = d_regs[slot][m].load()
                    luts = [
                        a8w4.fp8_lut(deltas[p], lut_sel[step_parity]) for p in range(2)
                    ]
                    for p in range_constexpr(2):
                        lut_regs[p][m].store(
                            Vec.from_elements(list(luts[p]), fx.Uint32)
                        )
                else:
                    luts = []
                    for p in range_constexpr(2):
                        held = lut_regs[p][m].load()
                        luts.append((held[0], held[1]))
                raw = w_regs[slot][m][ki].load()
                for p in range_constexpr(2):
                    for c in range_constexpr(2):
                        w_frag[p][None, m, (c, ki)].store(
                            a8w4.fp4x8_to_fp8(raw[2 * p + c], luts[p])
                        )

        mfma_per_ki = 4 * _m_reps * _n_reps
        dsrd_per_ki = TILE_M // 16

        def valu_for(ki):
            # Region ki decodes block ki + 1 (or the next tile's block 0); first halves add the LUT.
            return 2 * _m_reps * (20 if (ki + 1) % 2 == 0 else 16)

        a_vmem_cnt = TILE_M_LD * TILE_K // (256 * 16)
        # Per weight slot (rule 4: with TILE_K 128 slot 1 loads no deltas).
        w_vmem_cnt = [
            _m_reps * (k_iters + 1),
            _m_reps * (k_iters + int(KB_PER_TILE != 2)),
        ]

        def hot_loop_scheduler(issue_a, has_next, n_w):
            # Next-tile A loads go out with the first MFMAs, the weight loads of the tile after it
            # with the first MFMAs of the last block; the next 64-K block's LDS reads and FP4
            # decode hide under the current block's MFMAs (the last block decodes the next tile's
            # first block); LDS writes close the stage.
            valu = 0x002
            n_a = a_vmem_cnt * int(issue_a)
            n_dswr = a_vmem_cnt * int(has_next)
            rocdl.sched_dsrd(dsrd_per_ki)
            for ki in range_constexpr(k_iters):
                last = ki + 1 == k_iters
                valu_per_ki = valu_for(ki)
                for j in range_constexpr(mfma_per_ki):
                    rocdl.sched_mfma(1)
                    if const_expr((ki == 0 and j < n_a) or (last and j < n_w)):
                        rocdl.sched_vmem(1)
                    if const_expr(not last or has_next):
                        cnt = (valu_per_ki * (j + 1)) // mfma_per_ki - (
                            valu_per_ki * j
                        ) // mfma_per_ki
                        if const_expr(cnt > 0):
                            rocdl.sched_group_barrier(valu, cnt, 0)
                    if const_expr(not last and j % 2 == 1 and j // 2 < dsrd_per_ki):
                        rocdl.sched_dsrd(1)
                    if const_expr(last and j >= mfma_per_ki - n_dswr):
                        rocdl.sched_dswr(1)

        def pipeline_stage(read_i, k_cur, k_next, has_next, ahead=None):
            # ahead: tile k_cur + 2, whose weights reuse this tile's slot once its last block is
            # decoded, a whole stage before the end of the next stage decodes their first block
            # (with A_AHEAD 2 its A also goes to this tile's A slot, freed when A(k_cur) went to
            # LDS).
            write_i = read_i ^ 1
            issue_a = (ahead is not None) if A_AHEAD == 2 else has_next
            if const_expr(issue_a):
                if const_expr(A_AHEAD == 2):
                    a_idx.copy(buf_cp_atom_r, ahead, a_cp_frag[read_i])
                else:
                    a_idx.copy(buf_cp_atom_r, k_next, a_cp_frag[0])
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                if const_expr(ki > 0):
                    decode_weights(read_i, read_i, ki)
                if const_expr(ki + 1 == k_iters and ahead is not None):
                    load_weights(read_i, ahead)
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
                            fx.mma_atom_call(
                                mma_atom,
                                c_even[None, m, n],
                                w_frag[0][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_even[None, m, n],
                            )
                            fx.mma_atom_call(
                                mma_atom,
                                c_odd[None, m, n],
                                w_frag[1][None, m, (k, ki)],
                                a_frag[None, n, (k, ki)],
                                c_odd[None, m, n],
                            )
            if const_expr(has_next):
                decode_weights(write_i, write_i, 0)
                fx.copy(
                    uni_cp_atom, a_cp_frag_retile[write_i % A_AHEAD], a_lds_w[write_i]
                )
            hot_loop_scheduler(
                issue_a, has_next, w_vmem_cnt[read_i] if ahead is not None else 0
            )
            rocdl.sched_barrier(0)
            gpu.barrier()

        a_idx.copy(buf_cp_atom_r, fx.Int32(0), a_cp_frag[0])
        load_weights(0, fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        load_weights(1, fx.Int32(1))
        fx.copy(uni_cp_atom, a_cp_frag_retile[0], a_lds_w[0])
        if const_expr(A_AHEAD == 2):
            a_idx.copy(buf_cp_atom_r, fx.Int32(1), a_cp_frag[1])
        decode_weights(0, 0, 0)
        gpu.barrier()

        acc_init = [c_even.load(), c_odd.load()]
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            c_even.store(state[0])
            c_odd.store(state[1])
            kb = fx.Int32(iv * 2)
            pipeline_stage(0, kb, kb + 1, True, kb + 2)
            pipeline_stage(1, kb + 1, kb + 2, True, kb + 3)
            results = yield [c_even.load(), c_odd.load()]
        c_even.store(results[0])
        c_odd.store(results[1])
        kb = fx.Int32(num_tiles - 2)
        pipeline_stage(0, kb, kb + 1, True)
        pipeline_stage(1, kb + 1, fx.Int32(0), False)
        return c_even, c_odd

    def _apply_dequant(
        c_gate_frag,
        c_up_frag,
        tid,
        expert_id,
        blk_n,
        contiguous_n,
        asc_idx,
        M,
        p_w_scale,
        p_a_scale,
    ):
        m_reps = fx.size(fx.get_shape(c_gate_frag)[1]).to_py_value()
        n_reps = fx.size(fx.get_shape(c_gate_frag)[2]).to_py_value()
        scale_gate = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N + blk_n * contiguous_n,
            fx.make_layout(contiguous_n, 1),
        )
        scale_up = fx.make_view(
            fxh._as_ptr(p_w_scale) + expert_id * N + N // 2 + blk_n * contiguous_n,
            fx.make_layout(contiguous_n, 1),
        )
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        # Thread (t, G, w) loads channels 16w + _chan_base + v of each 64-channel group.
        scale_copy = fx.make_tiled_copy(
            cp_atom_scale,
            fx.make_layout(((16, (2, 2), 4), 4), ((0, (8, 4), 16), 1)),
            fx.make_tile(64),
        )
        sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
        su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
        gate_scale = fx.make_fragment_like(sg_thr)
        up_scale = fx.make_fragment_like(su_thr)
        fx.copy(cp_atom_scale, sg_thr, gate_scale)
        fx.copy(cp_atom_scale, su_thr, up_scale)

        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fx.Float32, fxh._as_ptr(p_a_scale)), fx.make_layout(M, 1)
            ),
            max_size=False,
        )
        a_sc_n = [
            a_scale_tensor[asc_idx[0, n] & 0xFFFFFF] for n in range_constexpr(n_reps)
        ]
        for m in range_constexpr(m_reps):
            sg_v = gate_scale[None, m].load()
            su_v = up_scale[None, m].load()
            for n in range_constexpr(n_reps):
                a_sc = a_sc_n[n]
                cg = c_gate_frag[None, m, n].load()
                cu = c_up_frag[None, m, n].load()
                cg_items = []
                cu_items = []
                for v in range_constexpr(4):
                    g, u = a8w4.swiglu_clamp(
                        cg[v] * sg_v[v] * a_sc, cu[v] * su_v[v] * a_sc, SWIGLU_LIMIT
                    )
                    cg_items.append(g)
                    cu_items.append(u)
                c_gate_frag[None, m, n].store(Vec.from_elements(cg_items, fx.Float32))
                c_up_frag[None, m, n].store(Vec.from_elements(cu_items, fx.Float32))

    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_gateup_prefill", SWIGLU_LIMIT))
    def a8w4_gateup_prefill(
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
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        if const_expr(XCD_CHUNK > 0):
            wg = fx.Int32(gpu.block_idx.x)
            step = wg // a8w4.XCDS
            work = ((step // CHUNK) * a8w4.XCDS + wg % a8w4.XCDS) * CHUNK + step % CHUNK
            blk_n = work % NUM_N_BLOCKS
            e_idx = work // NUM_N_BLOCKS
        else:
            blk_n = gpu.block_idx.x
            e_idx = gpu.block_idx.y

        in_ptr = fx.recast_iter(fp8, fxh._as_ptr(p_input))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, K), (K, 1)))
        num_valid_buf = fx.make_view(
            fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid_ids)), fx.make_layout(1, 1)
        )
        if const_expr(CHUNKED):
            live, e_idx = a8w4.chunk_block(p_num_valid_ids, e_idx)
        else:
            max_valid_id = num_valid_buf[0]
            live = e_idx * BM < max_valid_id
        if live:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()
            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_ids) + e_idx * BM),
                fx.make_layout(BM, 1),
            )
            expert_id = fx.recast_iter(fx.Int32, fxh._as_ptr(p_sorted_expert_ids))[
                e_idx
            ]
            contiguous_n = BN // 2
            w_buf = a8w4.expert_bytes(p_weight, expert_id, W_BYTES)
            d_buf = a8w4.expert_bytes(p_w_delta, expert_id, D_BYTES)

            sorted_ids_buf = fx.rocdl.make_buffer_tensor(
                arg_p_sorted_ids, max_size=False
            )
            if tid < BM:
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Capture scatter indices before a_ping reuses sorted_lds.
            asc_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                fx.make_tile(16),
            )
            asc_lds = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
            asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
            asc_idx = fx.make_fragment_like(asc_thr)
            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32), asc_thr, asc_idx
            )

            def run(tile_m):
                # c_even / c_odd of the main loop become gate / up in _unpair.
                c_gate_frag, c_up_frag = gemm_1x4(
                    tile_m, BN, blk_n, arg_p_input, w_buf, d_buf, lds
                )
                _unpair(c_gate_frag, c_up_frag, tid)
                _apply_dequant(
                    c_gate_frag,
                    c_up_frag,
                    tid,
                    expert_id,
                    blk_n,
                    contiguous_n,
                    asc_idx,
                    M,
                    p_w_scale,
                    p_a_scale,
                )
                _store_silu_f32(
                    c_gate_frag,
                    c_up_frag,
                    tid,
                    blk_n,
                    contiguous_n,
                    asc_idx,
                    M,
                    p_output,
                )

            ids = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))

            def padding(row):
                return (fx.Int32(rocdl.readfirstlane(T.i32, ids[row])) & 0xFFFFFF) >= M

            # Valid rows lead a sorted block: compute only the rows up to the first padding row.
            if const_expr(BM == 64):
                if padding(32):
                    if padding(16):
                        run(16)
                    else:
                        run(32)
                else:
                    if padding(48):
                        run(48)
                    else:
                        run(64)
            elif const_expr(BM % 64 == 0):
                if padding(BM // 2):
                    run(BM // 2)
                else:
                    run(BM)
            else:
                run(BM)

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
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        num_n_blocks = fxh.div_up(N, BN)
        if const_expr(E is not None) and M * TOPK <= E:
            task_num = M * TOPK
        if const_expr(XCD_CHUNK > 0):
            per = a8w4.XCDS * XCD_CHUNK
            grid = ((task_num + (per - 1)) // per * (a8w4.XCDS * CHUNK), 1, 1)
        else:
            grid = (num_n_blocks, task_num, 1)
        a8w4_gateup_prefill(
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
            M,
            **(
                {"value_attrs": {"rocdl.waves_per_eu": WAVES_PER_EU}}
                if WAVES_PER_EU
                else {}
            ),
        ).launch(grid=grid, block=(256, 1, 1), stream=stream)

    return launch
