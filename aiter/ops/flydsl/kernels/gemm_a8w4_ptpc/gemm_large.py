# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 eight-wave persistent ping-pong 128x256 (K step 128) dense A8W4 GEMM for large M.

Compile-time variant ``x_block_scale``: 1x32 FP8 X (one E8M0 scale per token and 32 K).

Same operands and arithmetic as ``gemm.py`` (per-token FP8 X x fd2 MXFP4 W -> BF16
Y = X W^T, W decoded to FP8 in VGPRs by ``layout``), for large M: the entry's
``_gemm_choice`` sends M > 2048 here when ``large_m_supported`` (N % 256 == 0, K >= 512),
and the six (N, K) of its measured ``_MID_TABLE`` from lower M (8192 x 2560 from M = 129,
6656 x 2560 from 513, the other four from 1025); with 1x32 X (``x_quant="1x32"`` and
``flydsl_gemm_a8w4_mxfp8``) the ``x_block_scale`` variant runs at the same M. Differences
from ``gemm.py``: a
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
Plain order: CTA ``c`` takes tiles ``c, c + G, ...``; tile ``t`` is M block ``t // NB``, N
block ``t % NB``, so the CTAs running together share a few M blocks of X. CTAs are dealt
round-robin to the 4 XCDs (``c % 4``, each with its own L2), so XCD ``x`` sees the N blocks
``= x`` mod ``gcd(NB, 4)``. Panel order (``panels`` = 1, 2 or 4, chosen per call by
``xcd_panels``; it needs as many CTAs as tiles on every XCD): the N blocks are cut into
panels of near-equal width; the list of all tiles, panel by panel and M-major inside a
panel, is cut into 4 equal parts, and XCD ``x``'s CTAs take part ``x`` in turn. One panel
puts whole M blocks of X on one XCD; two or four make each XCD hold the weights of about
one panel in its L2 (the outputs are then stored nt). The K loop runs through tile
boundaries without draining.

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

Variant ``x_block_scale`` (1x32 X: ``x_scale`` uint8 E8M0 [M, K / 32]): gfx942 has no
block-scaled MFMA, so every MMA covers one 32-K block and its sum is scaled before it is
accumulated. Tile, schedule, ping-pong, X staging and the W prefetch slots stay; the MMA is
32x32x16 FP8, whose lane ``(g, r)`` (``g = lane // 32``) holds K ``16 g + 8 i .. + 7`` of MMA
``i`` of a block, so a block's two MMAs (chained into ``part``) read 16 contiguous X bytes and
one 8-byte fd2 run (lane group ``2 (b % 2) + g``) per lane: one 16-byte X read and one
8-byte W load per block (16x16x32 MMAs would need 4-byte W loads, 20% slower). Wave ``w``
owns channels ``32 w ..`` as one 32-row tile: row ``r`` is channel
``pi(r) = 16 ((r // 4) % 2) + 4 (r // 8) + r % 4`` (fd2 block ``2 w + (r // 4) % 2``, row
``4 (r // 8) + r % 4``), so output lane ``(g, r)`` holds channels ``16 g .. 16 g + 15`` of
token ``32 ni + r`` (value ``i`` = channel ``16 g + i``: two 16-byte stores per token block);
``acc`` holds 4 token blocks x 16 = 64 f32 per lane. The E8M0 bytes become f32
``2**(e - 127)`` = bits ``max(e << 23, 2**-127)`` (exact for e < 255; e = 0 is the subnormal
``2**-127``, kept), expanded by each wave group into its own two LDS buffers
(``[block][token % 32][token // 32]``, 2 KiB per K step) one memory phase ahead, like X. Per
block and token block ``acc += part * s`` (``= part * s`` for a tile's first block) as 16
FMAs, half as scalar ``v_fmac`` (co-issued with the MMAs) and half as ``v_pk_fma_f32``
(never co-issued: issued after an MMA's shadow is full). The epilogue multiplies by the
channel factor only, loaded in ``M(T-1)``. K step ``j`` (one sched region per MMA, block
``b`` = 32-K block, ``D`` = deltas of both K halves, ``S`` = block scales):

    phase  MFMA (32 per wave)             LR             VMEM                         LW
    M(j)   -                              X(j), S(j) b0  X(j+2) S(j+2) W1(j+1)        X(j+1)
                                                         W0(j+2) D(j+2)               S(j+1)
    C(j)   part = W_b X_b (i = 0, 1),     S(j) b1..b3    -                            -
           acc += part * s; per MMA ~7 scalar VALU (FMA units, then decode W1(j) in
           blocks 0, 1 / W0(j+1) in blocks 2, 3), then a packed FMA unit

Naming: M = tokens, N = channels, K = reduction. ``m`` indexes the two MFMA row tiles of a
wave, ``ni`` the eight 16-token column tiles, ``ki`` / ``half`` the 64-K halves of a K step,
``slot`` the parity of a K step. ``w_regs`` / ``d_regs``: raw weight / delta dwords per slot;
``w_frag``: decoded FP8 A operand; ``a_frag``: X B operand; ``x_stage``: the rows of
``X(j+1)`` a thread writes; ``xs_regs`` / ``ws_regs``: the epilogue scales. LR / LW = LDS
read / write. ``x_block_scale`` only: ``ni`` indexes 4 token blocks of 32, ``i`` the two MMAs
of a block, ``part`` a block's unscaled sum, ``w_regs_bs`` / ``d_regs_bs`` the raw W / delta
dwords, ``d_hand`` the deltas ``M(j)`` replaces, ``s_stage`` / ``s_hold`` the E8M0 dword of
``S(j+2)`` / the scales of block 0, ``ws_bs`` the channel scales; a unit is 4 VALU.

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
8. ``x_block_scale``: ``C(j)`` still decodes ``W1(j)`` with ``D(j)``, which ``M(j)`` replaces
   (rule 3): ``M(j)`` hands it over in ``d_hand``. Each wave group writes ``S(j+1)`` only to
   its own buffers, so its compute phases may read ``S(j)`` without a cross-group barrier.
9. ``x_block_scale``: a token block's FMA units run from the region after its MMA pair and
   before its next block's pair (the unit scheduler ``_take_units`` keeps them first); a
   ``v_pk_fma_f32`` never follows an MMA directly (LLVM unpacks those it finds in an MMA's
   shadow, and a packed op there stalls until the MMA ends); SLP vectorization is off for
   this kernel so the scalar units stay ``v_fmac``.
"""

import functools
import math

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
# XCDs the CTAs are dealt to round-robin by linear id (MI308X). Another count only changes
# L2 locality.
_XCDS = 4
# Bytes of weights or X that one XCD's L2 (MI308X: 4 MiB) holds well, for xcd_panels.
_XCD_L2_BYTES = 3 << 20
# x_block_scale: 32-K blocks per K step and block scales per K step of a tile.
_BLOCKS = TILE_K // 32
_S_STEP = BLOCK_M * _BLOCKS

KERNEL_NAME = format_kernel_name(
    f"gemm_a8w4_ptpc_pp_bm{BLOCK_M}_bn{BLOCK_N}_tk{TILE_K}"
)


def large_kernel_name(x_block_scale=False):
    """Name of the large-M kernel; the 1x32 variant (``x_block_scale``) ends in ``_1x32``."""
    return KERNEL_NAME + ("_1x32" if x_block_scale else "")


def large_m_supported(n, k):
    """Whether the large-M kernel handles weights of ``n`` channels and reduction ``k``."""
    return n % BLOCK_N == 0 and k % K_ALIGN == 0 and k // TILE_K >= MIN_K_TILES


def xcd_panels(m, n, k, num_cus):
    """N panels of the tile order for ``x [m, k]`` and weights ``[n, k]`` (0: plain order).

    The plain order deals tile ``t`` (M block ``t // NB``, N block ``t % NB``) to CTA
    ``t % G``, so XCD ``x`` sees the ``NB / gcd(NB, 4)`` N blocks ``= x`` mod ``gcd(NB, 4)``
    and, per round of ``G`` tiles, ``G / NB`` M blocks, each M block on every XCD. A panel
    order needs as many CTAs as tiles on every XCD (``tiles >= num_cus``, or a multiple of
    4). One panel (whole M blocks per XCD) when the weights fit one L2 (``_XCD_L2_BYTES``),
    or X does: the next call's quantization then rewrites each X row in one L2 only
    (measured on MI308X: the quantization after the plain order is 2-3x slower).
    Otherwise, when ``NB % 4 != 0``: four panels (about ``NB / 4`` N blocks per XCD) when
    the plain order's weights per XCD alone exceed ``_XCD_L2_BYTES`` (they cannot stay in
    the L2 from one round to the next), two (the same N blocks, half the M blocks) when its
    weights and X of a round together do.
    """
    n_blocks = n // BLOCK_N
    m_blocks = (m + BLOCK_M - 1) // BLOCK_M
    tiles = m_blocks * n_blocks
    if num_cus % _XCDS != 0 or (tiles < num_cus and tiles % _XCDS != 0):
        return 0
    x_bytes = m_blocks * BLOCK_M * k
    if layout.weight_bytes(n, k) <= _XCD_L2_BYTES or x_bytes <= _XCD_L2_BYTES:
        return 1
    if n_blocks % _XCDS == 0:
        return 0
    w_xcd = n_blocks // math.gcd(n_blocks, _XCDS) * layout.weight_bytes(BLOCK_N, k)
    if w_xcd > _XCD_L2_BYTES:
        return _XCDS
    x_xcd = num_cus * BLOCK_M * k // n_blocks
    two = math.gcd(n_blocks, _XCDS) == 2 and w_xcd + x_xcd > _XCD_L2_BYTES
    return 2 if two else 0


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


def _take_units(queue, region, cap, flush):
    """Pop the queue's leading units (earliest region, fn, then, deadline) that may run in
    ``region``: at most ``cap`` of them, all when ``flush``; a unit with a deadline must run
    before that region (the x_block_scale compute stream, at trace time)."""
    taken = []
    while queue and queue[0][0] <= region and (flush or len(taken) < cap):
        unit = queue.pop(0)
        assert unit[3] is None or region < unit[3], "decode unit past its deadline"
        taken.append(unit)
    return taken


@functools.cache
def compile_gemm_a8w4_ptpc_large(
    n, k, num_cus, device_index, panels=0, x_block_scale=False
):
    """Launcher ``(xq, x_scale, wq, wd, ws, y, M, stream)`` of the large-M kernel.

    Arguments as ``gemm.compile_gemm_a8w4_ptpc``; ``num_cus`` is the persistent grid size,
    ``panels`` the tile order (``xcd_panels``; any M runs correctly with any order).
    ``x_block_scale``: 1x32 X, ``x_scale`` uint8 E8M0 [M, k / 32] (row-major; byte 255
    gives an undefined result) instead of f32 [M]. ``device_index`` only keys the cache.
    """
    del device_index
    if not large_m_supported(n, k):
        raise ValueError(
            f"large-M A8W4 GEMM needs N % {BLOCK_N} == 0, K % {K_ALIGN} == 0 and "
            f"K >= {MIN_K_TILES * TILE_K}, got N={n}, K={k}"
        )
    block_scale = bool(x_block_scale)
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
    if num_cus % _XCDS != 0 or n_blocks < panels:
        panels = 0
    # Two or four panels store the outputs nt (streaming), so they do not evict weights.
    y_cache = 2 if panels > 1 else 0
    ctas_per_xcd = num_cus // _XCDS
    # Panel p holds N blocks bounds[p] .. bounds[p + 1] - 1 (widths differ by at most one).
    bounds = [(p * n_blocks + panels // 2) // max(panels, 1) for p in range(panels + 1)]
    widths = sorted({bounds[p + 1] - bounds[p] for p in range(panels)})

    if block_scale:
        # x_block_scale: X buffers and per wave group two buffers of a K step's block
        # scales (f32, [block][token % 32][token // 32]).
        @fx.struct
        class SharedStorage:
            x0: fx.Array[fp8, BLOCK_M * TILE_K, 16]
            x1: fx.Array[fp8, BLOCK_M * TILE_K, 16]
            s: fx.Array[fx.Float32, 2 * 2 * _S_STEP, 16]

    else:

        @fx.struct
        class SharedStorage:
            x0: fx.Array[fp8, BLOCK_M * TILE_K, 16]
            x1: fx.Array[fp8, BLOCK_M * TILE_K, 16]

    @flyc.kernel(name=large_kernel_name(block_scale), known_block_size=[_THREADS, 1, 1])
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
        # Plain order: CTA c takes tiles c, c + G, ... of the M-major tile list.
        num_mine = (num_tiles - cta + num_cus - 1) // num_cus
        if const_expr(panels > 0):
            # Panel order, when every XCD gets as many CTAs as tiles (G = num_cus, or
            # G = tiles, a multiple of 4, one tile each; else the plain order): see
            # tile_coords.
            xcd_ok = (num_tiles >= num_cus) | (num_tiles % _XCDS == 0)
            xcd = cta % _XCDS
            local = cta // _XCDS
            part_lo = xcd * num_tiles // _XCDS
            part_hi = (xcd + 1) * num_tiles // _XCDS
            xcd_mine = (part_hi - part_lo - local + ctas_per_xcd - 1) // ctas_per_xcd
            num_mine = xcd_ok.select(xcd_mine, num_mine)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(block_scale):
            # MFMA 32x32x16: lane (g, r) (g = lane // 32) of MMA i holds K 16g + 8i .. + 7
            # of a 32-K block, so the block's two MMAs read 16 contiguous bytes per lane.
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fp8))
            k_perm = fx.make_layout((8, 2, 2), (1, 16, 8))
        else:
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
            m_blk, n_blk = t // n_blocks, t % n_blocks
            if const_expr(panels == 0):
                return m_blk, n_blk
            # Tile u of the panel list: panels in N order, M-major inside a panel.
            u = part_lo + local + i * ctas_per_xcd
            if const_expr(panels == 1):
                return (
                    xcd_ok.select(u // n_blocks, m_blk),
                    xcd_ok.select(u % n_blocks, n_blk),
                )
            # Panel p starts at tile bounds[p] * m_blocks, so u // m_blocks finds it.
            col = u // m_blocks
            p = fx.Int32(0)
            for b in bounds[1:-1]:
                p = p + (col >= b).select(fx.Int32(1), fx.Int32(0))
            b0 = (p * n_blocks + panels // 2) // panels
            width = (p * n_blocks + n_blocks + panels // 2) // panels - b0
            v = u - b0 * m_blocks
            m_x = v // widths[0]
            if const_expr(len(widths) > 1):
                m_x = (width == widths[1]).select(v // widths[1], m_x)
            n_x = b0 + v - m_x * width
            return xcd_ok.select(m_x, m_blk), xcd_ok.select(n_x, n_blk)

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

        # MMA-side K (v, c, g, h, step) -> stored K v + 8c + 32g + 16h + 128 step (gemm.py);
        # with x_block_scale (v, i, g, b) -> v + 8i + 16g + 32b (MMA i of 32-K block b).
        if const_expr(block_scale):
            k_shape = (8, 2, 2, _BLOCKS)
            k_stride = (1, 8, 16, 32)
        else:
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
            [fx.make_rmem_tensor(4, fx.Uint32) for _ in range(k_iters)]
            for _ in range(2)
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

        if const_expr(block_scale):
            # ---- x_block_scale W: 8 bytes per lane and 32-K block (MFMA 32x32x16) ----
            # Lane (g, r) (g = lane // 32, r = lane % 32) supplies A row r, channel
            # pi(r) = 16 ((r // 4) % 2) + 4 (r // 8) + r % 4 of the wave's 32 (so output lane
            # (g, r) holds channels 16 g .. 16 g + 15): fd2 block 2 w + (r // 4) % 2, row
            # 4 (r // 8) + r % 4. For 32-K block b of a K step (64-K block kb = 2 step + b // 2)
            # it loads fd2 lane group 2 (b % 2) + g of that row: dword i is MMA i's K
            # 16 g + 8 i .. + 7. Its delta is byte kb % 4 of the delta dword of K half b % 2.
            w_grp = lane // 32
            w_row = lane % 32
            w_rb = (w_row // 4) % 2
            w_rr = 4 * (w_row // 8) + w_row % 4
            w_view8 = ptr_buf_tensor(
                p_wq, elem=fx.Uint32, unit_elems=2, num_records_bytes=w_bytes
            )
            d_view4 = fx.logical_divide(
                ptr_buf_tensor(p_wd, elem=fx.Uint32, num_records_bytes=d_bytes),
                fx.make_layout(1, 1),
            )
            w_dw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Uint32)
            # 8-byte units of q (16-row x 64-K block = 64 units) and dwords of d (16-row
            # x 256-K block = 32 dwords, K half h at + 16 h).
            w_lane_bs = w_rb * (kb_total * 64) + w_grp * 16 + w_rr
            d_lane_bs = w_rb * (kq_total * 32) + w_rr
            # [slot][half]: byte 2 * (step % 2) + half of a delta dword, step % 2 = slot.
            sel_bs = [
                [fx.Uint32((2 * s + h) * 0x01010101) for h in range(2)]
                for s in range(2)
            ]
            # [slot][half][c] raw dwords (MMA i = 0, 1) of block 2 half + c; [slot][h] the
            # delta dword of K half h; d_hand: the deltas of the step a memory phase
            # replaces, handed to the compute phase that still decodes its second half.
            w_regs_bs = [
                [
                    [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(2)]
                    for _ in range(2)
                ]
                for _ in range(2)
            ]
            d_regs_bs = [
                [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(2)] for _ in range(2)
            ]
            d_hand = [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(2)]

            def load_weights_bs(slot, n_blk, k_tile, half):
                # half 0: the delta dwords of both K halves and blocks 0, 1; half 1: 2, 3.
                kb = k_tile * k_iters + half
                rb = n_blk * (BLOCK_N // 16) + wave * 2
                if const_expr(half == 0):
                    for h in range_constexpr(2):
                        dw = (rb * kq_total + k_tile // 2) * 32 + h * 16 + d_lane_bs
                        fx.copy(
                            w_dw_atom,
                            fx.slice(d_view4, (None, dw)),
                            d_regs_bs[slot][h],
                        )
                base = (rb * kb_total + kb) * 64 + w_lane_bs
                for c in range_constexpr(2):
                    fx.copy(
                        d_atom,
                        fx.slice(w_view8, (base + c * 32, None)),
                        w_regs_bs[slot][half][c],
                    )

            def decode_units(slot, half, c, delta):
                # Decode of block b = 2 half + c of the step in slot (into w_frag (i, b)) as
                # five 4-VALU units: its LUT, then per MMA i the even / odd nibbles.
                st = {}
                b = 2 * half + c

                def lut_unit():
                    st["lut"] = layout.fp8_lut(delta, sel_bs[slot][half])

                def nibble_unit(i, odd):
                    raw = w_regs_bs[slot][half][c].load()[i]
                    st[(i, odd)] = layout.fp4x4_to_fp8(raw, st["lut"], odd)
                    if const_expr(odd):
                        w_frag[None, 0, (i, b)].store(
                            Vec.from_elements(
                                [st[(i, False)], st[(i, True)]], fx.Uint32
                            ).bitcast(fp8)
                        )

                units = [lut_unit]
                for i in range_constexpr(2):
                    units += [
                        functools.partial(nibble_unit, i, False),
                        functools.partial(nibble_unit, i, True),
                    ]
                return units

            # ---- x_block_scale X scales: E8M0 bytes -> f32 2**(e - 127) in LDS ----
            # Each wave group expands all 128 x 4 scales of a K step into its own buffers
            # ([block][token % 32][token // 32]; thread t of the group: token t // 2, blocks
            # 2 (t % 2), 2 (t % 2) + 1), so its compute phases read them without a barrier
            # between the groups.
            s_lds = lds.s.ptr
            s_thr = tid % (_THREADS // 2)
            s_tok = s_thr // 2
            s_pair = s_thr % 2
            s_stage = fx.make_rmem_tensor(1, fx.Uint32)
            vec4 = fx.Vector.make_type(4, fx.Float32)

            def scale_source(m_blk):
                # E8M0 dwords of M block m_blk (one per token and K step); rows past M read 0.
                row0, nrows = x_rows(m_blk)
                return fx.logical_divide(
                    ptr_buf_tensor(
                        buf_base_i64(p_xs) + fx.Int64(row0) * (k // 32),
                        elem=fx.Uint32,
                        num_records_bytes=nrows * (k // 32),
                    ),
                    fx.make_layout(1, 1),
                )

            def load_scale_dword(s_src, k_s):
                fx.copy(
                    w_dw_atom,
                    fx.slice(s_src, (None, s_tok * (k // TILE_K) + k_s)),
                    s_stage,
                )

            def write_scales(buf):
                # 2**(e - 127) as the f32 bits max(e << 23, 2**-127 bits): exact for every
                # e < 255 (e = 0 is the subnormal 2**-127; the kernel keeps f32 denormals).
                word = s_stage.load()[0] >> fx.Uint32(s_pair * 16)
                base = (group * 2 + buf) * _S_STEP + (s_tok % 32) * 4 + s_tok // 32
                for i in range_constexpr(2):
                    bits = fx.max(
                        (word << (23 - 8 * i)) & fx.Uint32(0x7F800000),
                        fx.Uint32(0x00400000),
                    )
                    fx.ptr_store(
                        Vec.from_elements([bits.bitcast(fx.Float32)], fx.Float32),
                        s_lds + base + (2 * s_pair + i) * (_S_STEP // _BLOCKS),
                    )

            def read_scales(buf, b):
                # 2**(e - 127) of block b for the lane's tokens ni * 32 + r, ni = 0 .. 3.
                base = (group * 2 + buf) * _S_STEP + b * (_S_STEP // _BLOCKS)
                v = Vec(fx.ptr_load(s_lds + base + w_row * 4, result_type=vec4))
                return [v[i] for i in range(4)]

        # ---- epilogue operands: per-token / per-channel scales of the finished tile ----
        xs_regs = fx.make_rmem_tensor(n_reps, fx.Float32)
        ws_regs = [fx.make_rmem_tensor(4, fx.Float32) for _ in range(m_reps)]
        if const_expr(block_scale):
            # Channel scales of output lane (g, r): channels 16 g .. 16 g + 15 of the wave.
            ws_bs = fx.make_rmem_tensor(16, fx.Float32)

        def load_scales(m_blk, n_blk):
            if const_expr(block_scale):
                # Channel scales only: the block scales were applied in the K loop.
                ch4 = n_blk * (BLOCK_N // 4) + wave * 8 + w_grp * 4
                ws_bs.store(
                    Vec.from_elements(
                        [
                            Vec(
                                buf_copy_load(
                                    s_buf, ch4 + q, elem=fx.Float32, unit_elems=4
                                )
                            )[i]
                            for q in range_constexpr(4)
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )
                )
                return
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
            # of the wave's 32 for token l % 16 of every 16-token block (x_block_scale:
            # 2 x 16 bytes, channels 16 (l // 32) .. + 15 for token l % 32 of a 32-token
            # block).
            row0, nrows = x_rows(m_blk)
            y_buf = ptr_buf_tensor(
                buf_base_i64(p_y) + fx.Int64(row0) * (n * 2),
                elem=fx.Uint32,
                unit_elems=4,
                num_records_bytes=nrows * (n * 2),
            )
            if const_expr(block_scale):
                return y_buf, n_blk * (BLOCK_N // 8) + wave * 4 + w_grp * 2
            return y_buf, n_blk * (BLOCK_N // 8) + wave * 4 + lane_group

        def store_block(out, ni):
            # Token block ni of the finished tile: scale, round to BF16 and store.
            y_buf, ch8 = out
            if const_expr(block_scale):
                # Value i of the 32x32 tile is channel 16 g + i (pi); the block scales were
                # applied in the K loop.
                vals = Vec(tile16(acc, ni).load()) * Vec(ws_bs.load())
                words = [
                    _bf16x2_rne(vals[2 * q], vals[2 * q + 1])
                    for q in range_constexpr(8)
                ]
                for h in range_constexpr(2):
                    buf_copy_store(
                        y_buf,
                        (ni * 32 + w_row) * (n // 8) + ch8 + h,
                        Vec.from_elements(words[4 * h : 4 * h + 4], fx.Uint32),
                        elem=fx.Uint32,
                        unit_elems=4,
                        cache_modifier=y_cache,
                    )
                return
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
                cache_modifier=y_cache,
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

        def memory_phase(slot, x_next, k_x, w1, w2, s_next=None):
            # M(j), j % 2 = slot: LR X(j); LW X(j+1); VMEM X(j+2) (from x_next, K tile k_x),
            # W(j+1) second half into slot ^ 1 (w1 = (n_blk, k_tile)) and W(j+2) first half
            # with its deltas into slot (w2). Both halves they overwrite are decoded already.
            if const_expr(block_scale):
                memory_phase_bs(slot, x_next, k_x, w1, w2, s_next)
                return
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

        if const_expr(block_scale):
            # [ni]: 2**(e - 127) of block 0 of a step, read by its memory phase.
            s_hold = fx.make_rmem_tensor(BLOCK_M // 32, fx.Float32)

            def memory_phase_bs(slot, x_next, k_x, w1, w2, s_next):
                # M(j) of x_block_scale: as memory_phase, plus the block scales: LR S(j) of
                # block 0, LW S(j+1) (expanded from the staged dword), VMEM S(j+2) (from
                # s_next); D(j), which W0(j+2) replaces, goes to d_hand for C(j).
                fx.copy(uni_cp_atom, x_lds_r[slot], a_frag_retile)  # LR X(j)
                s_hold.store(
                    Vec.from_elements(read_scales(slot, 0), fx.Float32)
                )  # LR S(j) block 0
                rocdl.sched_barrier(0)
                fx.copy(uni_cp_atom, x_stage_retile, x_lds_w[slot ^ 1])  # LW X(j+1)
                write_scales(slot ^ 1)  # LW S(j+1)
                fx.copy(
                    buf_cp_atom, x_next[None, None, None, k_x], x_stage
                )  # VMEM X(j+2)
                load_scale_dword(s_next, k_x)  # VMEM S(j+2)
                load_weights_bs(slot ^ 1, w1[0], w1[1], 1)  # VMEM W1(j+1)
                for h in range_constexpr(2):
                    d_hand[h].store(d_regs_bs[slot][h].load())
                load_weights_bs(slot, w2[0], w2[1], 0)  # VMEM W0(j+2), D(j+2)

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
            if const_expr(block_scale):
                compute_phase_bs(slot, first)
                return
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
            if const_expr(block_scale):
                compute_phase_last_bs(slot, out)
                return
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

        if const_expr(block_scale):
            # One 32-K block's unscaled sum per token block (MMAs i = 0, 1 chained into it).
            part = tiled_mma.make_fragment_C(
                fx.make_view(
                    lds.x0.ptr, fx.make_layout((_WAVE_N, BLOCK_M), (BLOCK_M, 1))
                )
            )
            n_t = BLOCK_M // 32
            unit_valu = 4

            def tile16(frag, ni):
                # Token block ni of a 32x32 C fragment as one rank-1 run of its 16 values
                # (value v + 4 j: row 8 j + 4 (lane // 32) + v), as fx.gemm takes one MMA.
                t = frag[None, 0, ni]
                return fx.make_view(t.iter, fx.coalesce(t.layout))

            # Scalar VALU per MMA region: about what a 32x32x16 MMA (8 passes) hides; the
            # scalar FMA units (half the tiles) and the decode units fill it.
            region_valu = 7
            win = 2 * n_t

            def fma_units(ni, pv, s, first):
                # acc += part * 2**(e - 127) of a block for token block ni (acc = part * s
                # for a tile's first block) as four 4-value units: scalar v_fmac for even ni
                # (co-issued under an MMA), two v_pk_fma_f32 for odd ni (never co-issued:
                # issued after the MMA's shadow is full of scalar work).
                packed = ni % 2 == 1

                def unit(q):
                    # Only this unit's values of acc are loaded and stored.
                    view = fx.logical_divide(tile16(acc, ni), fx.make_layout(4, 1))
                    av = Vec(view[None, q].load())
                    pq = [pv[4 * q + k] for k in range(4)]
                    if const_expr(packed):
                        pq = Vec.from_elements(pq, fx.Float32)
                        sq = Vec.from_elements([s] * 4, fx.Float32)
                        nq = pq * sq if first else fx.fma(pq, sq, av)
                    else:
                        nq = Vec.from_elements(
                            [
                                pq[k] * s if first else fx.fma(pq[k], s, av[k])
                                for k in range(4)
                            ],
                            fx.Float32,
                        )
                    view[None, q].store(nq)

                return packed, [functools.partial(unit, q) for q in range(4)]

            def compute_stream(slot, first, out=None):
                # C(j) of x_block_scale as one software pipeline. MMAs block-major, per token
                # block ni the chained pair (i = 0, 1) into part; per MMA one sched region:
                # the MMA, then about region_valu scalar VALU in 4-VALU units, a token
                # block's scalar FMA units first (from the region after its pair), then
                # decode units; then a sched region with one packed FMA unit. Windows
                # 0, 1 (blocks 0, 1) decode W1(j) (blocks 2, 3; deltas D(j) from d_hand),
                # windows 2, 3 decode W0(j+1) (deltas D(j+1)); block b + 1's scales are read
                # in block b's window, block 0's came with M(j). With out (C(T-1)) each
                # token block is stored after its last FMA. The units left after the last
                # MMA run in a final region.
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                rocdl.sched_barrier(0)
                hold = s_hold.load()
                scales = {0: [hold[ni] for ni in range(n_t)]}
                dh = [d_hand[h].load()[0] for h in range(2)]
                dn = [d_regs_bs[slot ^ 1][h].load()[0] for h in range(2)]
                dec_q = []
                for b in range_constexpr(_BLOCKS):
                    if const_expr(b < 2):
                        units = decode_units(slot, 1, b, dh[b])  # W1(j) block 2 + b
                        deadline = (2 + b) * win
                    else:
                        units = decode_units(slot ^ 1, 0, b - 2, dn[b - 2])  # W0(j+1)
                        deadline = _BLOCKS * win + 1
                    dec_q += [(b * win, u, None, deadline) for u in units]
                # VALU credit: region_valu per region, spent in 4-VALU units.
                credit = [0]
                s_q = []
                p_q = []
                n_mma = _BLOCKS * win
                for r in range_constexpr(n_mma + 1):
                    if const_expr(r < n_mma):
                        b, rest = divmod(r, win)
                        ni, i = divmod(rest, 2)
                        fx.gemm(
                            mma_atom,
                            tile16(part, ni),
                            w_frag[None, 0, (i, b)],
                            a_frag[None, ni, (i, b)],
                            tile16(zero_c if i == 0 else part, ni),
                        )
                        if const_expr(i == 1):
                            pv = Vec(tile16(part, ni).load())
                            packed, units = fma_units(
                                ni, pv, scales[b][ni], first and b == 0
                            )
                            last = b == _BLOCKS - 1 and out is not None
                            for q, u in enumerate(units):
                                then = ni if last and q == len(units) - 1 else None
                                (p_q if packed else s_q).append((r + 1, u, then, None))
                        if const_expr(rest == 1 and b + 1 < _BLOCKS):
                            scales[b + 1] = read_scales(
                                slot, b + 1
                            )  # LR S(j) block b+1
                    flush = r == n_mma
                    credit[0] = min(credit[0] + region_valu, 3 * unit_valu)
                    s_now = _take_units(s_q, r, credit[0] // unit_valu, flush)
                    dec_now = _take_units(
                        dec_q, r, credit[0] // unit_valu - len(s_now), flush
                    )
                    credit[0] -= unit_valu * (len(dec_now) + len(s_now))
                    for item in s_now + dec_now:
                        item[1]()
                    n_s = unit_valu * (len(dec_now) + len(s_now))
                    if const_expr(r < n_mma):
                        rocdl.sched_mfma(1)
                    if const_expr(n_s > 0):
                        rocdl.sched_group_barrier("valu", n_s, 0)
                    rocdl.sched_barrier(0)
                    # Packed FMAs after the shadow's scalar work, in their own sched region.
                    p_now = _take_units(p_q, r, 1, flush)
                    for item in p_now:
                        item[1]()
                    for item in s_now + p_now:
                        if const_expr(item[2] is not None):
                            store_block(out, item[2])
                    rocdl.sched_barrier(0)
                assert not dec_q and not s_q and not p_q
                rocdl.s_setprio(0)
                phase_end()

            def compute_phase_bs(slot, first):
                compute_stream(slot, first)

            def compute_phase_last_bs(slot, out):
                compute_stream(slot, False, out)

        # ======== Prologue: X(0) -> LDS x0, X(1) staged, W(0) / W(1) loaded; M(0) ========
        m_blk0, n_blk0 = tile_coords(0)
        x_first = x_source(m_blk0)
        if const_expr(block_scale):
            # Also S(0) -> LDS (this group's buffer 0) and S(1) staged.
            s_first = scale_source(m_blk0)
            fx.copy(buf_cp_atom, x_first[None, None, None, 0], x_stage)
            load_scale_dword(s_first, 0)
            load_weights_bs(0, n_blk0, fx.Int32(0), 0)
            load_weights_bs(0, n_blk0, fx.Int32(0), 1)
            load_weights_bs(1, n_blk0, fx.Int32(1), 0)
            rocdl.s_waitcnt(vmcnt=0)
            fx.copy(uni_cp_atom, x_stage_retile, x_lds_w[0])
            write_scales(0)
            fx.copy(buf_cp_atom, x_first[None, None, None, 1], x_stage)
            load_scale_dword(s_first, 1)
            for c in range_constexpr(2):
                for u in decode_units(0, 0, c, d_regs_bs[0][c].load()[0]):
                    u()  # W0(0) block c
        else:
            s_first = None
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
        memory_phase(
            0, x_first, 2, (n_blk0, fx.Int32(1)), (n_blk0, fx.Int32(2)), s_first
        )
        phase_end()

        # ======== Tiles: C(0) .. C(T-1), C(T-1) also storing the tile, then the next
        # tile's M(0). acc is dead at the back edge. ========
        for iv, state in range(0, num_mine, 1, init=[fx.Int32(0)]):
            i = fx.Int32(iv)
            m_blk, n_blk = tile_coords(i)
            m_next, n_next = tile_coords(fx.min(i + 1, num_mine - 1))
            x_cur = x_source(m_blk)
            x_nxt = x_source(m_next)
            if const_expr(block_scale):
                s_cur = scale_source(m_blk)
                s_nxt = scale_source(m_next)
            else:
                s_cur = s_nxt = None

            compute_phase(0, first=True)
            memory_phase(1, x_cur, 3, (n_blk, fx.Int32(2)), (n_blk, fx.Int32(3)), s_cur)
            phase_end()
            compute_phase(1)

            pair_init = [acc.load()]
            pair_res = pair_init
            for p, pst in range(1, num_k_tiles // 2 - 1, 1, init=pair_init):
                acc.store(pst[0] if isinstance(pst, (list, tuple)) else pst)
                j = fx.Int32(p) * 2
                memory_phase(0, x_cur, j + 2, (n_blk, j + 1), (n_blk, j + 2), s_cur)
                phase_end()
                compute_phase(0)
                memory_phase(1, x_cur, j + 3, (n_blk, j + 2), (n_blk, j + 3), s_cur)
                phase_end()
                compute_phase(1)
                pair_res = yield [acc.load()]
            acc.store(pair_res[0] if isinstance(pair_res, (list, tuple)) else pair_res)

            # ======== Last K pair: prefetch the next tile's first steps and this tile's
            # scales; C(T-1) stores the tile. ========
            last = fx.Int32(num_k_tiles - 1)
            memory_phase(0, x_nxt, 0, (n_blk, last), (n_next, fx.Int32(0)), s_nxt)
            if const_expr(not block_scale):
                load_scales(m_blk, n_blk)
            phase_end()
            compute_phase(0)
            memory_phase(
                1, x_nxt, 1, (n_next, fx.Int32(0)), (n_next, fx.Int32(1)), s_nxt
            )
            if const_expr(block_scale):
                # Channel scales only, needed late in C(T-1): fewer live VGPRs in C(T-2).
                load_scales(m_blk, n_blk)
            phase_end()
            compute_phase_last(1, out_buffer(m_blk, n_blk))
            memory_phase(
                0, x_nxt, 2, (n_next, fx.Int32(1)), (n_next, fx.Int32(2)), s_nxt
            )
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

    if block_scale:
        # Keep the scalar FMAs of the 1x32 compute stream scalar (v_fmac is co-issued with
        # the MMAs; SLP would pack pairs into v_pk_fma_f32, which never is).
        launch.compile_hints = {"llvm_options": {"vectorize-slp": False}}
    return launch
