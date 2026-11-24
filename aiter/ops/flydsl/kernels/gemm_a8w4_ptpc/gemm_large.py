# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 eight-wave persistent ping-pong 128x256 (K step 128) dense A8W4 GEMM for large M.

Same operands and arithmetic as ``gemm.py`` (per-token FP8 X x fd2 MXFP4 W -> BF16
Y = X W^T, W decoded to FP8 in VGPRs by ``layout``), for M large enough to give every CU many
tiles (the entry sends M > 2048 with N % 256 == 0 here). Differences from ``gemm.py``: a
128 x 256 tile, a persistent grid, two wave groups in ping-pong, a channel permutation for
16-byte weight loads and output stores, and the output written inside the K loop.

Tile: a CTA owns ``BLOCK_M`` (128) tokens x ``BLOCK_N`` (256) channels. Wave ``w`` (0..7)
owns channels ``32 w .. 32 w + 31`` for all 128 tokens: two MFMA row tiles ``m`` x eight
16-token column tiles ``ni``. MFMA 16x16x32 FP8 takes W as A and X as B, with the K order of
``gemm.py``. Tile ``m`` row ``r`` is channel ``32 w + 2 r + m`` (fd2 block ``2 w + r // 8``,
row ``2 (r % 8) + m``): a lane's rows of both tiles are one 16-byte load, and output lane
group ``g`` holds channels ``8 g .. 8 g + 7`` of a token. ``acc`` holds 2 x 8 tiles x 4 = 64
f32 per lane in VGPRs. Epilogue: ``acc * w_scale * x_scale`` rounded to BF16, one 16-byte
store per lane and 16-token block.

Schedule: persistent, ``G = min(tiles, num_cus)`` CTAs, one per CU (two waves per SIMD).
CTA ``c`` takes tiles ``c, c + G, ...``; tile ``t`` is M block ``t // NB``, N block
``t % NB``, so the CTAs running together share a few M blocks of X. The K loop runs through
tile boundaries without draining.

Ping-pong: waves 0-3 (group 0) and 4-7 (group 1), one of each per SIMD, alternate a memory
phase ``M(j)`` and a compute phase ``C(j)`` per K step ``j``, group 1 one phase behind; a CTA
barrier ends every phase. While one wave of a SIMD issues its 64 MFMAs (``s_setprio 1``),
the other does its loads and LDS traffic:

    slot   group 0   group 1
    2j     M(j)      C(j-1)
    2j+1   C(j)      M(j)

K step ``j``: X in LDS buffer ``x0`` / ``x1`` and raw W in VGPR slot 0 / 1, both by
``j % 2``; W0 / W1 = first / second 64-K half, D = its delta dwords, LUT = decode table:

    phase  MFMA (64 per wave)      LR     VMEM                          LW       VALU
    M(j)   -                       X(j)   X(j+2) W1(j+1) W0(j+2) D(j+2) X(j+1)   -
    C(j)   acc += W0(j) X0(j)      -      -                             -        decode W1(j)
           acc += W1(j) X1(j)                                                    LUT, decode W0(j+1)

Each group writes its own rows of ``X(j+1)`` (``x_stage``, loaded one memory phase earlier):
group 0 in slot ``2j``, group 1 in slot ``2j+1``, both after every read of that buffer
(``X(j-1)``, slots ``2j-2`` / ``2j-1``) and before its first read (slot ``2j+2``).

Tile boundary (``T = K / 128`` steps): ``C(0)`` starts ``acc`` from zero (MFMA C = 0). The
last K pair loads the next tile's first steps (``M(T-2)``: X(0), W0(0); ``M(T-1)``: X(1),
W1(0), W0(1)), and ``M(T-2)`` also the finished tile's scales. ``C(T-1)`` runs its second
half two token blocks at a time (MFMA chains four apart) and scales, rounds and stores the
previous pair while a pair multiplies; the first pair decodes ``W0`` of the next tile instead
and the last pair is stored after the final MFMA. ``acc`` is dead at the tile loop's back edge.

Naming: M = tokens, N = channels, K = reduction. ``m`` indexes the two MFMA row tiles of a
wave, ``ni`` the eight 16-token column tiles, ``ki`` / ``half`` the 64-K halves of a K step,
``slot`` the parity of a K step. ``w_regs`` / ``d_regs``: raw weight / delta dwords per slot;
``w_frag``: decoded FP8 A operand; ``a_frag``: X B operand; ``x_stage``: the rows of
``X(j+1)`` a thread writes; ``xs_regs`` / ``ws_regs``: the epilogue scales. LR / LW = LDS
read / write.

Pipeline rules, to keep when editing the kernel body:
1. ``phase_end`` (one CTA barrier, after ``lgkmcnt(0)``) ends every phase; it orders the LDS
   writes and reads of ``X`` above.
2. A memory phase reads ``X(j)`` first, then writes ``X(j+1)`` from ``x_stage`` and then
   reloads ``x_stage`` with ``X(j+2)``.
3. A raw W slot half is reloaded only after its decode: ``M(j)`` replaces ``W1(j-1)``,
   ``W0(j)`` and ``D(j)``, all decoded by ``C(j-1)``.
4. Compute phases issue MFMAs and the decode (in ``C(T-1)`` also the epilogue) between
   ``s_setprio 1`` and ``s_setprio 0``; ``sched_group_barrier`` spreads the VALU over the
   MFMAs.
5. Every wave passes the same number of barriers: group 1 adds one before the tile loop,
   group 0 one after it.
6. ``T`` is even and at least 4: the buffer and slot of a K step are static in the unrolled
   pair, and the first and last pairs are peeled.
7. ``vmcnt`` counts loads and stores in order on gfx942: the first wait on a load the
   ``C(T-1)`` stores precede (``M(1)`` of the next tile, two phases later) also waits for them.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import layout
from .gemm import K_ALIGN, TILE_K

BLOCK_M = 128
BLOCK_N = 256
_WAVES = 8
_THREADS = 64 * _WAVES
_WAVE_N = BLOCK_N // _WAVES
# The first and last K-step pairs of a tile are peeled.
MIN_K_TILES = 4

KERNEL_NAME = format_kernel_name(
    f"gemm_a8w4_ptpc_pp_bm{BLOCK_M}_bn{BLOCK_N}_tk{TILE_K}"
)


def large_m_supported(n, k):
    """Whether the large-M kernel handles weights of ``n`` channels and reduction ``k``."""
    return n % BLOCK_N == 0 and k % K_ALIGN == 0 and k // TILE_K >= MIN_K_TILES


def _bf16x2_rne(lo, hi):
    """Two f32 -> one dword of two BF16, round to nearest even (lo in the low half).

    ``(u + 0x7FFF + bit16(u)) >> 16`` per value (v_bfe + v_add3) and one v_perm to pack;
    the typed ``.to(BFloat16)`` adds a NaN check and select per value on gfx942. It agrees
    with it everywhere except NaNs whose payload lies only in the low 16 mantissa bits,
    which arithmetic does not produce.
    """
    ru = [
        (v.bitcast(fx.Uint32) + fx.Uint32(0x7FFF) + ((v.bitcast(fx.Uint32) >> 16) & 1))
        for v in (lo, hi)
    ]
    # FlyDSL 0.5.3 has no typed byte permute (see layout._perm): bytes 2, 3 of each.
    return fx.Uint32(fx.rocdl.perm_b32(ru[1], ru[0], fx.Uint32(0x07060302)))


@functools.cache
def compile_gemm_a8w4_ptpc_large(n, k, num_cus, device_index):
    """Launcher ``(xq, x_scale, wq, wd, ws, y, M, stream)`` of the large-M kernel.

    Arguments as ``gemm.compile_gemm_a8w4_ptpc``; ``num_cus`` is the persistent grid size.
    ``device_index`` only keys the cache.
    """
    del device_index
    if not large_m_supported(n, k):
        raise ValueError(
            f"large-M A8W4 GEMM needs N % {BLOCK_N} == 0, K % {K_ALIGN} == 0 and "
            f"K >= {MIN_K_TILES * TILE_K}, got N={n}, K={k}"
        )
    fp8 = fx.Float8E4M3FNUZ
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    k_iters = TILE_K // 64
    num_k_tiles = k // TILE_K
    n_blocks = n // BLOCK_N
    m_reps = _WAVE_N // 16
    n_reps = BLOCK_M // 16
    # The two row tiles of a wave share each 16-byte weight load and output dword.
    assert m_reps == 2
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)
    w_block_units = layout.W_BLOCK_BYTES // 8
    d_block_units = layout.D_BLOCK_BYTES // 4

    @fx.struct
    class SharedStorage:
        x0: fx.Array[fp8, BLOCK_M * TILE_K, 16]
        x1: fx.Array[fp8, BLOCK_M * TILE_K, 16]

    @flyc.kernel(name=KERNEL_NAME, known_block_size=[_THREADS, 1, 1])
    def gemm_a8w4_ptpc_large_kernel(
        p_x: fx.Pointer,
        p_xs: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = wave // 4
        cta = fx.block_idx.x
        m_blocks = (M + BLOCK_M - 1) // BLOCK_M
        num_tiles = m_blocks * n_blocks
        num_mine = (num_tiles - cta + num_cus - 1) // num_cus
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))
        # As gemm.py: lane group b of MMA c holds K 16b + 8c .. 16b + 8c + 7.
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 1, 1), (1, 0, 0)),
            fx.make_tile(None, None, k_perm),
        )

        def tile_coords(i):
            t = cta + i * num_cus
            return t // n_blocks, t % n_blocks

        # ---- X: global -> VGPR (16 B per lane, 64 rows per pass) -> swizzled LDS ----
        buf_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
        val_per_thr = 16
        thrs_k = TILE_K // val_per_thr
        thrs_m = _THREADS // thrs_k
        g2r_tv_layout = fx.make_layout(
            ((thrs_k, thrs_m), (1, val_per_thr)),
            ((thrs_m * val_per_thr, 1), (1, thrs_m)),
        )
        x_g2r = fx.make_tiled_copy(
            buf_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, TILE_K)
        )
        x_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, TILE_K)
        )

        def x_rows(m_blk):
            # Rows of M block m_blk (< m_blocks) below M; buffers start at the block's first
            # row, so offsets stay 32-bit and rows at or past M read zeros / drop stores.
            # (A clamp at 0 here would become a VALU med3 and force waterfall loops.)
            row0 = m_blk * BLOCK_M
            return row0, fx.min(M - row0, fx.Int32(BLOCK_M))

        def x_source(m_blk):
            row0, nrows = x_rows(m_blk)
            x_tile = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(fp8, p_x) + fx.Int64(row0) * k,
                    fx.make_layout((BLOCK_M, k), (k, 1)),
                ),
                num_records_bytes=nrows * k,
            )
            tiles = fx.flat_divide(x_tile, fx.make_tile(BLOCK_M, TILE_K))[
                None, None, 0, None
            ]
            return x_g2r.get_slice(tid).partition_S(tiles)

        swz = fx.SwizzleType.get(3, 4, 3)

        def lds_tile(buf):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((BLOCK_M, TILE_K), order=(1, 0)),
                ),
            )

        x_lds = [lds_tile(lds.x0), lds_tile(lds.x1)]
        x_lds_w = [x_r2s.get_slice(tid).partition_D(v) for v in x_lds]

        # MMA-side K (v, c, g, h, step) -> stored K v + 8c + 32g + 16h + 128 step (gemm.py).
        k_shape = (8, 2, 4, 2, TILE_K // 128)
        k_stride = (1, 8, 32, 16, 128)

        def read_view(buf):
            return fx.make_view(
                buf.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_layout((BLOCK_M, k_shape), (TILE_K, k_stride)),
                ),
            )

        x_s2r = fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(lane)
        x_lds_r = [
            x_s2r.partition_S(read_view(lds.x0)),
            x_s2r.partition_S(read_view(lds.x1)),
        ]
        a_frag = tiled_mma.make_fragment_B(x_lds[0])
        a_frag_retile = x_s2r.retile(a_frag)

        # Decoded FP8 weight operand (8, m_reps, (2, k_iters)) and accumulators
        # (4, m_reps, n_reps); the views only give the fragment shapes.
        w_frag = tiled_mma.make_fragment_A(
            fx.make_view(lds.x0.ptr, fx.make_layout((_WAVE_N, TILE_K), (TILE_K, 1)))
        )
        acc = tiled_mma.make_fragment_C(
            fx.make_view(lds.x0.ptr, fx.make_layout((_WAVE_N, BLOCK_M), (BLOCK_M, 1)))
        )

        # ---- W: fd2 bytes -> VGPR slots (as gemm.py, wave rows 32 w .. 32 w + 31) ----
        w_buf = ptr_buf_tensor(
            p_wq, elem=fx.Uint32, unit_elems=4, num_records_bytes=w_bytes
        )
        d_buf = ptr_buf_tensor(
            p_wd, elem=fx.Uint32, unit_elems=2, num_records_bytes=d_bytes
        )
        s_buf = ptr_buf_tensor(
            p_ws, elem=fx.Float32, unit_elems=4, num_records_bytes=n * 4
        )
        lane_group = lane // 16
        row = lane % 16
        # MFMA tile m row r holds channel 32 w + 2 r + m of the wave's 32: fd2 block
        # 2 w + r // 8, rows 2 (r % 8) + m, so one 16-byte load brings a lane's two tiles
        # (8 full 128-byte lines per load). Lane group g of the output then holds channels
        # 8 g .. 8 g + 7 of each token: one 16-byte store per token.
        fd2_row = 2 * (row % 8)
        lane_blk = row // 8
        # 16-byte units of q (block = 32 units) and 8-byte units of d (block = 16 units).
        w_lane = (
            lane_blk * (kb_total * w_block_units)
            + (lane_group // 2) * w_block_units
            + (lane_group % 2) * 32
            + fd2_row
        ) // 2
        d_lane = (
            lane_blk * (kq_total * d_block_units) + (lane_group % 2) * 16 + fd2_row
        ) // 2
        lut_sel = [fx.Uint32((lane_group // 2 + 2 * p) * 0x01010101) for p in range(2)]
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Uint32)
        d_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        # W slot, 64-K half: dwords 2 m + c (tile m, K 8c .. 8c + 7 of the lane group).
        w_regs = [
            [fx.make_rmem_tensor(4, fx.Uint32) for _ in range(k_iters)] for _ in range(2)
        ]
        d_regs = [fx.make_rmem_tensor(m_reps, fx.Uint32) for _ in range(2)]

        lut_regs = [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(m_reps)]

        def load_weights(slot, n_blk, k_tile, half):
            # half 0: the delta dwords and the first 64-K half; half 1: the second half.
            kb0 = k_tile * k_iters
            rb = n_blk * (BLOCK_N // 16) + wave * 2
            if const_expr(half == 0):
                unit = (rb * kq_total + kb0 // 4) * (d_block_units // 2) + d_lane
                fx.copy(d_atom, fx.slice(d_buf, (unit, None)), d_regs[slot])
            # Half ``half`` of the 128-K step starting at 64-K block kb0.
            unit = (rb * kb_total + kb0) * (w_block_units // 2) + w_lane + half * 8
            fx.copy(w_atom, fx.slice(w_buf, (unit, None)), w_regs[slot][half])

        def build_lut(slot):
            # slot = K-step parity: byte 2 * (step % 2) + g // 2 of the delta dword.
            deltas = d_regs[slot].load()
            for m in range_constexpr(m_reps):
                lut = layout.fp8_lut(deltas[m], lut_sel[slot])
                lut_regs[m].store(Vec.from_elements(list(lut), fx.Uint32))

        def decode_half(slot, half):
            raw = w_regs[slot][half].load()
            for m in range_constexpr(m_reps):
                held = lut_regs[m].load()
                for c in range_constexpr(2):
                    w_frag[None, m, (c, half)].store(
                        layout.fp4x8_to_fp8(raw[2 * m + c], (held[0], held[1]))
                    )

        # ---- epilogue operands: per-token / per-channel scales of the finished tile ----
        xs_regs = fx.make_rmem_tensor(n_reps, fx.Float32)
        ws_regs = [fx.make_rmem_tensor(4, fx.Float32) for _ in range(m_reps)]

        def load_scales(m_blk, n_blk):
            row0, nrows = x_rows(m_blk)
            xs_buf = ptr_buf_tensor(
                buf_base_i64(p_xs) + fx.Int64(row0) * 4,
                elem=fx.Float32,
                num_records_bytes=nrows * 4,
            )
            xs_regs.store(
                Vec.from_elements(
                    [
                        buf_copy_load(xs_buf, ni * 16 + lane % 16, elem=fx.Float32)
                        for ni in range_constexpr(n_reps)
                    ],
                    fx.Float32,
                )
            )
            # Channels 8 g .. 8 g + 7 of the wave; tile m row 4 g + v is channel 8 g + 2 v + m.
            ch4 = n_blk * (BLOCK_N // 4) + wave * 8 + lane_group * 2
            ws8 = [
                Vec(buf_copy_load(s_buf, ch4 + h, elem=fx.Float32, unit_elems=4))[i]
                for h in range_constexpr(2)
                for i in range_constexpr(4)
            ]
            for m in range_constexpr(m_reps):
                ws_regs[m].store(
                    Vec.from_elements([ws8[2 * v + m] for v in range(4)], fx.Float32)
                )

        def out_buffer(m_blk, n_blk):
            # Output rows of the tile; lane l writes 16 bytes: channels 8 (l // 16) .. + 7
            # of the wave's 32 for token l % 16 of every 16-token block.
            row0, nrows = x_rows(m_blk)
            y_buf = ptr_buf_tensor(
                buf_base_i64(p_y) + fx.Int64(row0) * (n * 2),
                elem=fx.Uint32,
                unit_elems=4,
                num_records_bytes=nrows * (n * 2),
            )
            return y_buf, n_blk * (BLOCK_N // 8) + wave * 4 + lane_group

        def store_block(out, ni):
            # Token block ni of the finished tile: scale, round to BF16 and store.
            y_buf, ch8 = out
            x_scale = xs_regs.load()[ni]
            vals = [
                Vec(acc[None, m, ni].load()) * Vec(ws_regs[m].load()) * x_scale
                for m in range_constexpr(m_reps)
            ]
            # Channel 2 v + m: tile 0 in the low half of dword v.
            words = [_bf16x2_rne(vals[0][v], vals[1][v]) for v in range_constexpr(4)]
            buf_copy_store(
                y_buf,
                (ni * 16 + row) * (n // 8) + ch8,
                Vec.from_elements(words, fx.Uint32),
                elem=fx.Uint32,
                unit_elems=4,
            )

        x_stage = fx.make_fragment_like(x_source(0)[None, None, None, 0])
        x_stage_retile = x_r2s.get_slice(tid).retile(x_stage)

        def phase_end():
            rocdl.sched_barrier(0)
            gpu.barrier()
            rocdl.sched_barrier(0)

        mfma_per_half = m_reps * n_reps * 2
        # VALU per 64-K decode half (per m: 2 dwords x 8) and per LUT (per m: 4).
        decode_valu = m_reps * 16
        lut_valu = m_reps * 4

        def memory_phase(slot, x_next, k_x, w1, w2):
            # M(j), j % 2 = slot: LR X(j); LW X(j+1); VMEM X(j+2) (from x_next, K tile k_x),
            # W(j+1) second half into slot ^ 1 (w1 = (n_blk, k_tile)) and W(j+2) first half
            # with its deltas into slot (w2). Both halves they overwrite are decoded already.
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_cp_atom,
                    x_lds_r[slot][None, None, ki],
                    a_frag_retile[None, None, ki],
                )  # LR X(j)
            rocdl.sched_barrier(0)
            fx.copy(uni_cp_atom, x_stage_retile, x_lds_w[slot ^ 1])  # LW X(j+1)
            fx.copy(buf_cp_atom, x_next[None, None, None, k_x], x_stage)  # VMEM X(j+2)
            load_weights(slot ^ 1, w1[0], w1[1], 1)  # VMEM W1(j+1)
            load_weights(slot, w2[0], w2[1], 0)  # VMEM W0(j+2), D(j+2)

        def schedule(n_valu, n_mfma=mfma_per_half):
            # Each MFMA followed by its even share of the VALU.
            for i in range_constexpr(n_mfma):
                rocdl.sched_mfma(1)
                cnt = n_valu * (i + 1) // n_mfma - n_valu * i // n_mfma
                if const_expr(cnt > 0):
                    rocdl.sched_group_barrier("valu", cnt, 0)

        zero_c = tiled_mma.make_fragment_C(
            fx.make_view(lds.x0.ptr, fx.make_layout((_WAVE_N, BLOCK_M), (BLOCK_M, 1)))
        )
        zero_c.fill(0)

        def compute_phase(slot, first=False):
            # C(j): acc += W0(j) X0(j) while decoding W1(j); acc += W1(j) X1(j) while
            # decoding W0(j+1) with its LUT. The decode VALU fills the MFMA issue gaps.
            # A tile's first step starts the accumulators from zero (MFMA C = 0).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            fx.gemm(
                mma_atom,
                acc,
                w_frag[None, None, (None, 0)],
                a_frag[None, None, (None, 0)],
                zero_c if first else acc,
            )
            decode_half(slot, 1)  # W1(j)
            schedule(decode_valu)
            rocdl.sched_barrier(0)
            fx.gemm(
                mma_atom,
                acc,
                w_frag[None, None, (None, 1)],
                a_frag[None, None, (None, 1)],
                acc,
            )
            build_lut(slot ^ 1)
            decode_half(slot ^ 1, 0)  # W0(j+1)
            schedule(decode_valu + lut_valu)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            phase_end()

        # VALU per token block of output, to spread it over the MFMAs (per m: 2 v_pk_mul by
        # w_scale, 4 x (v_bfe + v_add3), 2 v_perm; the x_scale multiplies are not counted).
        store_valu = m_reps * 12

        def compute_phase_last(slot, out):
            # C(T-1): as compute_phase, but the second half runs two token blocks at a time
            # (MFMA chains 4 apart) and scales, rounds and stores the previous pair of the
            # finished tile while a pair multiplies; the first pair decodes W0 of the next
            # tile instead, the last pair is stored after the final MFMA.
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            rocdl.sched_barrier(0)
            fx.gemm(
                mma_atom,
                acc,
                w_frag[None, None, (None, 0)],
                a_frag[None, None, (None, 0)],
                acc,
            )
            decode_half(slot, 1)  # W1(j)
            schedule(decode_valu)
            rocdl.sched_barrier(0)
            for p in range_constexpr(n_reps // 2):
                for kk in range_constexpr(2):
                    for ni in range_constexpr(2 * p, 2 * p + 2):
                        for m in range_constexpr(m_reps):
                            fx.gemm(
                                mma_atom,
                                acc[(None, 0), m, ni],
                                w_frag[None, m, (kk, 1)],
                                a_frag[None, ni, (kk, 1)],
                                acc[(None, 0), m, ni],
                            )
                if const_expr(p == 0):
                    build_lut(slot ^ 1)
                    decode_half(slot ^ 1, 0)  # W0 of the next tile
                    n_valu = decode_valu + lut_valu
                else:
                    store_block(out, 2 * p - 2)
                    store_block(out, 2 * p - 1)
                    n_valu = 2 * store_valu
                schedule(n_valu, 4 * m_reps)
                rocdl.sched_barrier(0)
            store_block(out, n_reps - 2)
            store_block(out, n_reps - 1)
            rocdl.sched_barrier(0)
            rocdl.s_setprio(0)
            phase_end()

        # ======== Prologue: X(0) -> LDS x0, X(1) staged, W(0) / W(1) loaded; M(0) ========
        m_blk0, n_blk0 = tile_coords(0)
        x_first = x_source(m_blk0)
        fx.copy(buf_cp_atom, x_first[None, None, None, 0], x_stage)
        load_weights(0, n_blk0, fx.Int32(0), 0)
        load_weights(0, n_blk0, fx.Int32(0), 1)
        load_weights(1, n_blk0, fx.Int32(1), 0)
        rocdl.s_waitcnt(vmcnt=0)
        fx.copy(uni_cp_atom, x_stage_retile, x_lds_w[0])
        fx.copy(buf_cp_atom, x_first[None, None, None, 1], x_stage)
        build_lut(0)
        decode_half(0, 0)
        phase_end()
        if group == 1:
            gpu.barrier()
        memory_phase(0, x_first, 2, (n_blk0, fx.Int32(1)), (n_blk0, fx.Int32(2)))
        phase_end()

        # ======== Tiles: C(0) .. C(T-1), C(T-1) also storing the tile, then the next
        # tile's M(0). acc is dead at the back edge. ========
        for iv, state in range(0, num_mine, 1, init=[fx.Int32(0)]):
            i = fx.Int32(iv)
            m_blk, n_blk = tile_coords(i)
            m_next, n_next = tile_coords(fx.min(i + 1, num_mine - 1))
            x_cur = x_source(m_blk)
            x_nxt = x_source(m_next)

            compute_phase(0, first=True)
            memory_phase(1, x_cur, 3, (n_blk, fx.Int32(2)), (n_blk, fx.Int32(3)))
            phase_end()
            compute_phase(1)

            pair_init = [acc.load()]
            pair_res = pair_init
            for p, pst in range(1, num_k_tiles // 2 - 1, 1, init=pair_init):
                acc.store(pst[0] if isinstance(pst, (list, tuple)) else pst)
                j = fx.Int32(p) * 2
                memory_phase(0, x_cur, j + 2, (n_blk, j + 1), (n_blk, j + 2))
                phase_end()
                compute_phase(0)
                memory_phase(1, x_cur, j + 3, (n_blk, j + 2), (n_blk, j + 3))
                phase_end()
                compute_phase(1)
                pair_res = yield [acc.load()]
            acc.store(pair_res[0] if isinstance(pair_res, (list, tuple)) else pair_res)

            # ======== Last K pair: prefetch the next tile's first steps and this tile's
            # scales; C(T-1) stores the tile. ========
            last = fx.Int32(num_k_tiles - 1)
            memory_phase(0, x_nxt, 0, (n_blk, last), (n_next, fx.Int32(0)))
            load_scales(m_blk, n_blk)
            phase_end()
            compute_phase(0)
            memory_phase(1, x_nxt, 1, (n_next, fx.Int32(0)), (n_next, fx.Int32(1)))
            phase_end()
            compute_phase_last(1, out_buffer(m_blk, n_blk))
            memory_phase(0, x_nxt, 2, (n_next, fx.Int32(1)), (n_next, fx.Int32(2)))
            phase_end()
            state = yield [state[0] if isinstance(state, (list, tuple)) else state]

        if group == 0:
            gpu.barrier()

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
        num_tiles = (M + BLOCK_M - 1) // BLOCK_M * n_blocks
        gemm_a8w4_ptpc_large_kernel(
            p_x,
            p_xs,
            p_wq,
            p_wd,
            p_ws,
            p_y,
            M,
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(
            grid=(fx.min(num_tiles, fx.Int32(num_cus)), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch
