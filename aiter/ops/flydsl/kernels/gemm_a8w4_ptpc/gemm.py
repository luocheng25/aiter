# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 four-wave 64x128x128 dense A8W4 GEMM: per-token FP8 X x fd2 MXFP4 W -> BF16 Y = X W^T.

Port of the pyhip A8W4 MoE prefill Gate/Up 1x4 kernel (``a8w4_gemm1.py``, tag
moe-ptpc-a8w4-r2-M11) without the sorted-id gather, the SiLU and the gate/up pairing: X rows
are dense and bounded by the buffer records, the two 64-channel halves are independent output
columns and the epilogue writes BF16.

Tile: a CTA owns ``BLOCK_M`` (64) tokens x ``BLOCK_N`` (128) channels. The channels form two
halves ``m`` = 0, 1 of 64; wave ``w`` owns 16-channel block ``w`` of each half for all 64
tokens (four 16-token columns ``ni``). MFMA 16x16x32 FP8 takes W as A (16 channels) and X as
B (16 tokens); ``acc`` holds 2 x 4 tiles = 32 f32 per lane, lane ``l`` channels
``4 (l // 16) + v`` of token ``l % 16``. Epilogue: ``acc * s[channel] * x_scale[token]`` ->
BF16, one 8-byte store per tile and lane; rows at or past M drop their stores (buffer
bounds), channels past N (``N % 128 == 64``) are masked.

Schedule: grid ``(ceil(N / 128), ceil(M / 64))``, the N block fastest, so CTAs that run
together share their X tile in L2. One tile per CTA, no split-K.

K step: X goes through two LDS buffers (``a_ping`` / ``a_pong``, 64 x 128 B, XOR swizzled).
W stays in VGPRs: raw FP4 / delta dwords in two slots by K-tile parity, decoded into
``w_frag`` one 64-K sub-step before its MFMAs. A 128-K tile ``k`` is one pipeline stage of
two 64-K sub-steps ``ki`` (halves ``h`` of the 128-K step):

    ki   MFMA (16)          LDS -> VGPR  VMEM (first MFMAs)    VALU (between MFMAs)
    0    acc += W0(k) X0(k)  X0(k)        X(k+1) W(k+1) D(k+1)  decode W1(k), same LUT
    1    acc += W1(k) X1(k)  X1(k)        -                     decode W0(k+1), new LUT
    end  X(k+1) VGPR -> LDS[(k+1) % 2]; s_barrier

(``W0`` / ``X0``: the first 64-K half of tile ``k`` in MMA-side K order, ``D``: deltas.)

The runtime loop runs two stages (ping, pong) per iteration; the last two stages are peeled:
the first prefetches the last tile, the second prefetches nothing. Inside a 128-K step lane
group ``g`` reads 32-K group ``g`` in both halves, so one LUT per lane and step decodes its
FP4: MMA-side K ``64 h + 16 g + 8 c + v`` -> stored K ``32 g + 16 h + 8 c + v``;
``_read_view`` permutes the X reads the same way.

Naming: M = tokens, N = channels, K = reduction. ``m`` indexes 64-channel halves (MMA A
rows), ``ni`` 16-token columns (MMA B columns), ``ki`` 64-K sub-steps, ``c`` the two MMAs of
a sub-step; ``lane_group`` / ``row`` split a wave's 64 lanes. ``w_regs`` / ``d_regs``: raw
weight / delta dwords per slot; ``w_frag``: decoded FP8 A operand; ``a_frag``: X B operand;
``a_cp_frag``: the next X tile between its global load and its LDS store.

Pipeline rules, to keep when editing the kernel body:
1. One ``s_barrier`` per stage, after its LDS store: the next stage overwrites the buffer
   this stage read (WAR) and reads the one it wrote (RAW).
2. A stage's global loads only fill slot ``write_i`` and ``a_cp_frag``; slot ``read_i`` was
   decoded by the stage before. Ping stages take even tiles (static parity), so the tile
   count is even: ``K % 256 == 0``.
3. The sub-step that decodes a step's first half builds the LUT; the second half reuses it.
   One delta dword covers two 128-K steps (byte ``2 (step % 2) + g // 2``).
4. The accumulators cross the runtime loop as ``init`` / ``yield`` values; the other loop
   state (weight slots, LUTs, ``w_frag``) lives in register tensors.
5. ``hot_loop_scheduler`` only orders instructions; its VMEM / VALU / LDS counts must match
   what a stage issues.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import layout

BLOCK_M = 64
BLOCK_N = 128
TILE_K = 128
_THREADS = 256
# Weight row (channel) alignment the kernel handles; K needs an even number of K tiles.
N_ALIGN = 64
K_ALIGN = 2 * TILE_K

GEMM_KERNEL_NAME = format_kernel_name(
    f"gemm_a8w4_ptpc_bm{BLOCK_M}_bn{BLOCK_N}_tk{TILE_K}"
)


@functools.cache
def compile_gemm_a8w4_ptpc(n, k, device_index):
    """Launcher ``(xq, x_scale, wq, wd, ws, y, M, stream)`` for weights of ``n`` channels x ``k``.

    ``xq`` FP8 E4M3FNUZ [M, k], ``x_scale`` f32 [M], ``wq`` / ``wd`` / ``ws`` the fd2 layout
    of ``layout.py``, ``y`` BF16 [M, n]; all contiguous, passed as raw ``fx.Pointer``.
    ``device_index`` only keys the cache (a compiled handle launches on one GPU).
    """
    del device_index
    if n <= 0 or n % N_ALIGN or k <= 0 or k % K_ALIGN:
        raise ValueError(
            f"A8W4 GEMM needs N % {N_ALIGN} == 0 and K % {K_ALIGN} == 0, got N={n}, K={k}"
        )
    fp8 = fx.Float8E4M3FNUZ
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    k_iters = TILE_K // 64
    num_tiles = k // TILE_K
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)
    m_reps = BLOCK_N // 64
    n_reps = BLOCK_M // 16
    n_blocks = (n + BLOCK_N - 1) // BLOCK_N
    ragged_n = n % BLOCK_N != 0
    w_block_units = layout.W_BLOCK_BYTES // 8
    d_block_units = layout.D_BLOCK_BYTES // 4

    @fx.struct
    class SharedStorage:
        a_ping: fx.Array[fp8, BLOCK_M * TILE_K, 16]
        a_pong: fx.Array[fp8, BLOCK_M * TILE_K, 16]

    def _make_tiled_mma():
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        # One 64-K MMA tile = (8 values, 4 lane groups, 2 MMAs); lane group b of MMA c
        # holds K 16b + 8c .. 16b + 8c + 7, the order of one fd2 ``q`` lane run.
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((4, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )
        return mma_atom, tiled_mma

    @flyc.jit
    def gemm_tile(x_tile, w_buf, d_buf, blk_n, lds):
        tid = gpu.thread_idx.x
        mma_atom, tiled_mma = _make_tiled_mma()

        # X: global -> VGPR (16 B per lane, 32 rows per pass) -> swizzled LDS.
        a_tile = fx.flat_divide(x_tile, fx.make_tile(BLOCK_M, TILE_K))[
            None, None, 0, None
        ]
        buf_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        val_per_thr = 16
        thrs_k = TILE_K // val_per_thr
        thrs_m = _THREADS // thrs_k
        g2r_tv_layout = fx.make_layout(
            ((thrs_k, thrs_m), (1, val_per_thr)),
            ((thrs_m * val_per_thr, 1), (1, thrs_m)),
        )
        a_g2r = fx.make_tiled_copy(
            buf_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, TILE_K)
        )
        a_mem_thr = a_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])

        swz = fx.SwizzleType.get(3, 4, 3)

        def _lds_tile(buf):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((BLOCK_M, TILE_K), order=(1, 0)),
                ),
            )

        a_ping = _lds_tile(lds.a_ping)
        a_pong = _lds_tile(lds.a_pong)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, TILE_K)
        )
        a_lds_w = [
            a_r2s.get_slice(tid).partition_D(a_ping),
            a_r2s.get_slice(tid).partition_D(a_pong),
        ]
        a_cp_frag_retile = a_r2s.get_slice(tid).retile(a_cp_frag)

        # MMA-side K (v, c, g, h, step) -> stored K v + 8c + 32g + 16h + 128 step.
        k_shape = (8, 2, 4, 2, TILE_K // 128)
        k_stride = (1, 8, 32, 16, 128)

        def _read_view(buf):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_layout((BLOCK_M, k_shape), (TILE_K, k_stride)),
                ),
            )

        a_s2r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid)
        a_lds_r = [
            a_s2r.partition_S(_read_view(lds.a_ping)),
            a_s2r.partition_S(_read_view(lds.a_pong)),
        ]
        a_frag = tiled_mma.make_fragment_B(a_ping)
        a_frag_retile = a_s2r.retile(a_frag)

        # Decoded FP8 weight operand of the current tile, (8, m_reps, (2, k_iters)), and the
        # accumulators, (4, m_reps, n_reps); the two views only give the fragment shapes.
        w_shape = fx.make_view(
            lds.a_ping.ptr, fx.make_layout((BLOCK_N, TILE_K), (TILE_K, 1))
        )
        w_frag = tiled_mma.make_fragment_A(w_shape)
        acc_shape = fx.make_view(
            lds.a_ping.ptr, fx.make_layout((BLOCK_N, BLOCK_M), (BLOCK_M, 1))
        )
        acc = tiled_mma.make_fragment_C(acc_shape)
        acc.fill(0)

        # Lane (g, r) of wave w: 64-K block g // 2 of the step, fd2 lane group
        # 2 * (g % 2) + h for half h (+128 B), delta half g % 2 and byte
        # 2 * (step % 2) + g // 2 of that dword.
        lane = tid % 64
        lane_group = lane // 16
        row = lane % 16
        wave = tid // 64
        w_lane = (lane_group // 2) * w_block_units + (lane_group % 2) * 32 + row
        d_lane = (lane_group % 2) * 16 + row
        lut_sel = [fx.Uint32((lane_group // 2 + 2 * p) * 0x01010101) for p in range(2)]
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        d_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Uint32)
        d_view = fx.logical_divide(d_buf, fx.make_layout(1, 1))
        # FP8 LUT of the latest decoded step per m (reused by its second half).
        lut_regs = [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(m_reps)]
        # Raw FP4 dwords and delta dwords per [slot][m]; slot = K-tile parity.
        w_regs = [
            [
                [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(k_iters)]
                for _ in range(m_reps)
            ]
            for _ in range(2)
        ]
        d_regs = [
            [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(m_reps)] for _ in range(2)
        ]

        def row_block16(m):
            return blk_n * (BLOCK_N // 16) + m * 4 + wave

        def load_weights(slot, k_tile):
            kb0 = k_tile * k_iters
            for m in range_constexpr(m_reps):
                rb = row_block16(m)
                for ki in range_constexpr(k_iters):
                    # Half ki % 2 of the 128-K step starting at 64-K block kb0 + ki - ki % 2.
                    unit = (
                        (rb * kb_total + kb0 + (ki - ki % 2)) * w_block_units
                        + w_lane
                        + (ki % 2) * 16
                    )
                    fx.copy(w_atom, fx.slice(w_buf, (unit, None)), w_regs[slot][m][ki])
                unit = (rb * kq_total + kb0 // 4) * d_block_units + d_lane
                fx.copy(d_atom, fx.slice(d_view, (None, unit)), d_regs[slot][m])

        def decode_weights(slot, parity, ki):
            # parity = K-tile index % 2 (static: ping tiles are even); the first half of a
            # 128-K step builds the LUT, the second half reuses it.
            step_parity = ((parity * k_iters + ki) // 2) % 2
            for m in range_constexpr(m_reps):
                if const_expr(ki % 2 == 0):
                    lut = layout.fp8_lut(
                        d_regs[slot][m].load()[0], lut_sel[step_parity]
                    )
                    lut_regs[m].store(Vec.from_elements(list(lut), fx.Uint32))
                else:
                    held = lut_regs[m].load()
                    lut = (held[0], held[1])
                raw = w_regs[slot][m][ki].load()
                for c in range_constexpr(2):
                    w_frag[None, m, (c, ki)].store(layout.fp4x8_to_fp8(raw[c], lut))

        mfma_per_ki = 2 * m_reps * n_reps
        dsrd_per_ki = BLOCK_M // 16

        def valu_for(ki):
            # Region ki decodes block ki + 1 (or the next tile's block 0); first halves add the LUT.
            return m_reps * (20 if (ki + 1) % 2 == 0 else 16)

        a_vmem_cnt = BLOCK_M * TILE_K // (_THREADS * 16)
        w_vmem_cnt = m_reps * (k_iters + 1)

        def hot_loop_scheduler(do_prefetch):
            # Next-tile loads go out with the first MFMAs; the next 64-K block's LDS reads and
            # FP4 decode hide under the current block's MFMAs (the last block decodes the next
            # tile's first block); LDS writes close the stage.
            n_vmem = (a_vmem_cnt + w_vmem_cnt) * int(do_prefetch)
            n_dswr = a_vmem_cnt * int(do_prefetch)
            rocdl.sched_dsrd(dsrd_per_ki)
            issued = 0
            for ki in range_constexpr(k_iters):
                last = ki + 1 == k_iters
                valu_per_ki = valu_for(ki)
                for j in range_constexpr(mfma_per_ki):
                    rocdl.sched_mfma(1)
                    if const_expr(issued < n_vmem):
                        rocdl.sched_vmem(1)
                        issued += 1
                    if const_expr(not last or do_prefetch):
                        cnt = (valu_per_ki * (j + 1)) // mfma_per_ki - (
                            valu_per_ki * j
                        ) // mfma_per_ki
                        if const_expr(cnt > 0):
                            rocdl.sched_group_barrier("valu", cnt, 0)
                    if const_expr(not last and j % 2 == 1 and j // 2 < dsrd_per_ki):
                        rocdl.sched_dsrd(1)
                    if const_expr(last and j >= mfma_per_ki - n_dswr):
                        rocdl.sched_dswr(1)

        def pipeline_stage(read_i, k_next, do_prefetch):
            write_i = read_i ^ 1
            if const_expr(do_prefetch):
                fx.copy(buf_cp_atom, a_mem_thr[None, None, None, k_next], a_cp_frag)
                load_weights(write_i, k_next)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    a_lds_r[read_i][None, None, ki],
                    a_frag_retile[None, None, ki],
                )
                if const_expr(ki > 0):
                    decode_weights(read_i, read_i, ki)
                fx.gemm(
                    mma_atom,
                    acc,
                    w_frag[None, None, (None, ki)],
                    a_frag[None, None, (None, ki)],
                    acc,
                )
            if const_expr(do_prefetch):
                decode_weights(write_i, write_i, 0)
                fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[write_i])
            hot_loop_scheduler(do_prefetch)
            rocdl.sched_barrier(0)
            gpu.barrier()

        fx.copy(buf_cp_atom, a_mem_thr[None, None, None, 0], a_cp_frag)
        load_weights(0, fx.Int32(0))
        rocdl.s_waitcnt(vmcnt=0)
        fx.copy(uni_cp_atom, a_cp_frag_retile, a_lds_w[0])
        decode_weights(0, 0, 0)
        gpu.barrier()

        acc_init = [acc.load()]
        results = acc_init
        for iv, state in range(0, num_tiles // 2 - 1, 1, init=acc_init):
            acc.store(state[0])
            kt = fx.Int32(iv * 2)
            pipeline_stage(0, kt + 1, True)
            pipeline_stage(1, kt + 2, True)
            results = yield [acc.load()]
        # A loop with one carried value yields it bare, not as a one-element list.
        acc.store(results[0] if isinstance(results, (list, tuple)) else results)
        pipeline_stage(0, fx.Int32(num_tiles - 1), True)
        pipeline_stage(1, fx.Int32(0), False)
        return acc

    @flyc.jit
    def store_tile(acc, blk_n, s_buf, xs_buf, y_buf):
        # Lane l holds channels 4 * (l // 16) + v of token l % 16 in every (m, n) MMA tile.
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tokens = [ni * 16 + lane % 16 for ni in range(n_reps)]
        x_scales = [buf_copy_load(xs_buf, t, elem=fx.Float32) for t in tokens]
        for m in range_constexpr(m_reps):
            ch4 = blk_n * (BLOCK_N // 4) + m * 16 + wave * 4 + lane // 16
            w_scale = buf_copy_load(s_buf, ch4, elem=fx.Float32, unit_elems=4)
            for ni in range_constexpr(n_reps):
                c = acc[None, m, ni].load()
                out = Vec.from_elements(
                    [
                        (c[v] * w_scale[v] * x_scales[ni]).to(fx.BFloat16)
                        for v in range_constexpr(4)
                    ],
                    fx.BFloat16,
                )
                unit = tokens[ni] * (n // 4) + ch4
                if const_expr(ragged_n):
                    if ch4 < n // 4:
                        buf_copy_store(y_buf, unit, out, elem=fx.BFloat16, unit_elems=4)
                else:
                    buf_copy_store(y_buf, unit, out, elem=fx.BFloat16, unit_elems=4)

    @flyc.kernel(name=GEMM_KERNEL_NAME, known_block_size=[_THREADS, 1, 1])
    def gemm_a8w4_ptpc_kernel(
        p_x: fx.Pointer,
        p_xs: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        blk_n = fx.block_idx.x
        row0 = fx.block_idx.y * BLOCK_M
        # Per-workgroup buffers start at the tile's first row: offsets stay 32-bit for
        # any M, and rows at or past M read zeros and drop their stores.
        nrows = fx.min(M - row0, fx.Int32(BLOCK_M))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        x_tile = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(fp8, p_x) + fx.Int64(row0) * k,
                fx.make_layout((BLOCK_M, k), (k, 1)),
            ),
            num_records_bytes=nrows * k,
        )
        w_buf = ptr_buf_tensor(
            p_wq, elem=fx.Uint32, unit_elems=2, num_records_bytes=w_bytes
        )
        d_buf = ptr_buf_tensor(p_wd, elem=fx.Uint32, num_records_bytes=d_bytes)
        acc = gemm_tile(x_tile, w_buf, d_buf, blk_n, lds)

        s_buf = ptr_buf_tensor(
            p_ws, elem=fx.Float32, unit_elems=4, num_records_bytes=n * 4
        )
        xs_buf = ptr_buf_tensor(
            buf_base_i64(p_xs) + fx.Int64(row0) * 4,
            elem=fx.Float32,
            num_records_bytes=nrows * 4,
        )
        y_buf = ptr_buf_tensor(
            buf_base_i64(p_y) + fx.Int64(row0) * (n * 2),
            elem=fx.BFloat16,
            unit_elems=4,
            num_records_bytes=nrows * (n * 2),
        )
        store_tile(acc, blk_n, s_buf, xs_buf, y_buf)

    @flyc.jit
    def launch(
        p_x: fx.Pointer,
        p_xs: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_a8w4_ptpc_kernel(p_x, p_xs, p_wq, p_wd, p_ws, p_y, M).launch(
            grid=(n_blocks, (M + BLOCK_M - 1) // BLOCK_M, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch
