# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 dense A8W4 GEMM with a configurable tile: per-token FP8 X x fd2 MXFP4 W -> BF16 Y = X W^T.

Compile-time variant ``x_block_scale``: 1x32 FP8 X (one E8M0 scale per token and 32 K).

Port of the pyhip A8W4 MoE prefill Gate/Up 1x4 kernel (``a8w4_gemm1.py``, tag
moe-ptpc-a8w4-r2-M11) without the sorted-id gather, the SiLU and the gate/up pairing: X rows
are dense and bounded by the buffer records, the 64-channel groups are independent output
columns and the epilogue writes BF16. ``GemmConfig`` picks the tile, the K per stage, the
prefetch depths, the XCD split and the K groups; the entry chooses one per (M, N, K).

Tile: a CTA owns ``block_m`` (16 / 48 with tile_k 256, 32, 64 or 128) tokens x ``block_n``
(64, 128 or 256) channels and has ``k_groups`` groups of four waves; group ``g`` covers the
contiguous K tiles ``[g T / k_groups, (g + 1) T / k_groups)`` (``T = K / tile_k``). The
channels form ``m_reps = block_n / 64`` groups ``m`` of 64; wave ``w`` of a group owns
16-channel block ``w`` of each ``m`` for all tokens (``n_reps = block_m / 16`` 16-token
columns ``ni``). MFMA 16x16x32 FP8 takes W as A (16 channels) and X as B (16 tokens);
``acc`` holds ``m_reps x n_reps`` tiles of 4 f32 per lane, lane ``l`` channels
``4 (l // 16) + v`` of token ``l % 16``. After the K loop the groups add their sums pairwise
through LDS (over the X buffers): groups ``[h, 2h)`` hand theirs to groups ``[0, h)``,
``h`` = k_groups / 2, 1/4 ... (fixed order, deterministic). Epilogue (group 0):
``acc * s[channel] * x_scale[token]`` -> BF16. When the tile fits the LDS allocation
(``_shuffled_store``), each lane writes its 8 bytes per MMA tile to an LDS copy of the
output tile (token rows, pitch ``2 block_n + 16`` B) and, after a barrier, every lane stores
16 bytes of a token row, so a store instruction writes whole 128-byte lines (an 8-byte
store per tile and lane writes 32-byte pieces of 16 rows: tp2_qkv M = 64 GEMM +0.9 us);
otherwise the 8-byte stores go out directly. Rows at or past M drop their stores (buffer
bounds), channels past N (``N % block_n != 0``) are masked.

Schedule: a 1-D grid, one tile per CTA, no split-K across CTAs. A super tile covers
``xcd_m`` M blocks x ``xcd_n = _XCDS / xcd_m`` N blocks and super tiles run M block major;
CTA ``c`` takes tile ``c % _XCDS`` of super tile ``c // _XCDS``. CTAs are dealt round-robin
to the ``_XCDS`` XCDs by linear id, so XCD ``x`` gets the M blocks ``= x // xcd_n`` and the
N blocks ``= x % xcd_n`` (mod ``xcd_m``, ``xcd_n``): an X tile goes through ``xcd_n`` XCD L2s,
a W tile through ``xcd_m``. ``xcd_m = 4`` keeps X in one L2: when X lines sit in several L2s,
the next call's quantization kernel, which rewrites the same buffer, slows down by up to
~15 us. Edge super tiles have CTAs without a tile, which exit.

K step: X goes through two LDS buffers per group (``block_m`` x ``tile_k`` B, XOR swizzled),
staged in VGPRs (``a_cp_frag``, ``x_stages`` sets) between its global load and its LDS
store. W stays in VGPRs: raw FP4 / delta dwords in ``num_slots = prefetch + 1`` slots,
decoded into ``w_frag`` one 64-K sub-step before its MFMAs. A ``tile_k`` (128 or 256) K tile
``k`` is one pipeline stage of ``tile_k / 64`` sub-steps ``ki`` (two per 128-K step, halves
``h``); stage ``k`` reads LDS buffer ``k % 2`` and W slot ``k % num_slots``:

    ki   MFMA (per wave)     LDS -> VGPR  VMEM (between all MFMAs)     VALU (between MFMAs)
    0    acc += W0(k) X0(k)  X0(k)        X(k+XS) W(k+P) D(k+P) ...    decode W1(k), same LUT
    1    acc += W1(k) X1(k)  X1(k)        ... (evenly spread)          decode W2(k) / W0(k+1)
    ..   (tile_k 256: two more sub-steps, a new LUT at ki 2)
    end  X(k+1) VGPR -> LDS[(k+1) % 2]; s_barrier

(``W0`` / ``X0``: the first 64-K block of tile ``k`` in MMA-side K order, ``D``: deltas,
``XS = x_stages``, ``P = prefetch``; ``k`` counts the group's tiles.) A W tile is in flight
for ``P`` stages, an X tile for ``XS``. The stage's loads go out evenly between all its
MFMAs (with ``P = 1`` between the first ones, as W(k+1) is decoded at the end of stage
``k``): issued back to back they fill the CU's vector memory queue, which stalls the MFMA
issue of every wave of the CTA (tp2_qkvz M = 64: GEMM 16.2 -> 14.5 us). The runtime loop runs
``unroll = lcm(num_slots, 2, XS)`` stages per iteration (slots, buffers and staging sets are
static in it); the last ``unroll`` to ``2 unroll - 1`` stages are peeled and stop loading
past the group's last tile. Inside a 128-K step lane group ``g`` reads 32-K group ``g`` in
both halves, so one LUT per lane and step decodes its FP4: MMA-side K
``64 h + 16 g + 8 c + v`` -> stored K ``32 g + 16 h + 8 c + v``; ``_read_view`` permutes the
X reads the same way.

Variant ``x_block_scale`` (1x32 activations: FP8 X with one E8M0 byte per token and 32 K,
``x_scale`` uint8 [M, K / 32]): gfx942 has no block-scaled MFMA, so each MMA must cover one
32-K block and its sum is scaled before it is accumulated. MMA ``c`` of sub-step ``ki`` covers
32-K block ``2 ki + c`` of the tile in natural K order (lane group ``g``: K ``8g .. 8g + 7``
of it): X is read from LDS as two 8-byte pieces per sub-step (stored K
``v + 32c + 8g + 64h + 128 step``) and lane ``(g, r)`` loads dword ``g % 2`` of fd2 lane
group ``2c + g // 2`` per MMA (two 4-byte loads per 64-K block instead of one 8-byte load)
and decodes each with the LUT of its own 32-K group (delta dwords of both halves). Each MMA
starts from zero, ``part = W_c X_c`` (f32), and the lane adds
``part * 2**(e[token, block] - 127)`` to ``acc`` with one FMA per two values; the E8M0 byte
becomes the f32 ``max(e << 23, 2**-127 bits)``, exact for every e < 255 (e = 0 is the
subnormal 2**-127: the kernel keeps f32 denormals, ``.amdhsa_float_denorm_mode_32 3``;
e = 255 gives Inf and an undefined result). ``acc`` holds these sums in decoded-weight units
(the channel factor comes in the epilogue, ``|part| < 2**21``), so it cannot overflow while
every E8M0 byte is at most 226 (block amax < 2**107). The scale dwords of tile ``k``
(``tile_k / 32`` bytes per token) are loaded with W(k) into its slot and expanded at the
start of stage ``k``. The epilogue multiplies by the channel factor only. The stage issues
its loads and MFMAs without ``hot_loop_scheduler`` hints (functional version):

    ki   MFMA (per wave)                         LDS -> VGPR       VMEM            VALU
    0    part = W0c(k) X0c(k); acc += part s(k)  X0(k), 2 x 8 B    X(k+XS) W(k+P)  scales(k),
    1    ... (c = 0, 1: blocks 2 ki + c)         X1(k)             D(k+P) S(k+P)   decode, FMA
    end  X(k+1) VGPR -> LDS[(k+1) % 2]; s_barrier

Naming: M = tokens, N = channels, K = reduction. ``m`` indexes 64-channel groups (MMA A
rows), ``ni`` 16-token columns (MMA B columns), ``ki`` 64-K sub-steps, ``c`` the two MMAs of
a sub-step; ``lane_group`` / ``row`` split a wave's 64 lanes; ``j`` is a stage's position in
the unrolled loop body. ``w_regs`` / ``d_regs``: raw weight / delta dwords per slot;
``w_frag``: decoded FP8 A operand; ``a_frag``: X B operand; ``a_cp_frag``: X tiles between
their global load and their LDS store. ``x_block_scale`` only: ``s_regs`` the E8M0 dwords of
a slot's tile, ``part`` one MMA's unscaled sum, ``W0c`` / ``X0c`` MMA ``c`` of block 0.

Pipeline rules, to keep when editing the kernel body:
1. One ``s_barrier`` per stage, after its LDS store: the next stage overwrites the buffer
   this stage read (WAR) and reads the one it wrote (RAW).
2. Stage ``k`` loads W(k+P) into slot ``(k + P) % num_slots``, the slot of W(k-1), decoded
   by stage k-1; it loads X(k+XS) into staging set ``k % XS``, whose tile X(k) went to LDS at
   the end of stage k-1. Slots, buffers and staging sets are static: the loop body is
   ``unroll`` stages and the tile count of the loop is a multiple of it. ``x_block_scale``:
   the E8M0 dwords of tile k+P go into the same slot, after its deltas; stage k expands
   slot ``k % num_slots`` once its own loads are issued.
3. The sub-step that decodes a step's first half builds the LUT; the second half reuses it.
   One delta dword covers two 128-K steps (byte ``2 (step % 2) + g // 2``): the parity is
   static per stage, flipped at run time for K groups that start on an odd 128-K tile
   (a 256-K tile is one whole delta dword). ``x_block_scale``: every MMA builds its own LUT
   from byte ``kb % 4`` (64-K block ``kb``) of the delta dword of its half ``c``, with the
   same static parity and run-time flip.
4. The accumulators cross the runtime loop as ``init`` / ``yield`` values; the other loop
   state (weight slots, LUTs, ``w_frag``, X staging) lives in register tensors.
5. Loads are issued in consumption order (X before W within a stage, oldest tile first in
   the prologue): ``vmcnt`` counts in order, so the waits before a decode or an LDS store
   leave every younger load in flight.
6. ``hot_loop_scheduler`` only orders instructions; its VMEM / VALU / LDS counts must match
   what a stage issues (per-token stages only; ``x_block_scale`` stages issue no hints).
7. Every wave of the CTA reaches every barrier: all groups run the same number of stages,
   and the K-group exchange (partials over the X buffers, after the last stage's barrier)
   and the epilogue are guarded only by wave-uniform conditions.
8. ``x_block_scale``: the K loop's FMA chain keeps a user in its own basic block. With one K
   group the epilogue is not guarded (every thread is group 0; with more, the K-group
   exchange is that user) and channels past N store out of bounds instead of branching:
   behind a branch instcombine sinks the whole chain into it and every MFMA sum stays live
   to the end (bm32 bn64 tk256: 512 VGPRs and scratch instead of 138 VGPRs).
"""

import functools
import math
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import format_kernel_name
from ..tensor_shim import buf_base_i64, buf_copy_load, buf_copy_store, ptr_buf_tensor
from . import layout

TILE_K = 128
_THREADS = 256
# Weight row (channel) alignment the kernel handles; K needs an even number of K tiles.
N_ALIGN = 64
K_ALIGN = 2 * TILE_K
# XCDs the CTAs are dealt to round-robin (MI308X). Another count only changes L2 locality.
_XCDS = 4


class GemmConfig(NamedTuple):
    """Tile (``block_m`` tokens x ``block_n`` channels), pipeline depths, K groups, XCD split."""

    block_m: int = 64
    block_n: int = 128
    # K tiles a weight tile is loaded ahead / X tiles staged in VGPRs ahead.
    prefetch: int = 1
    x_stages: int = 1
    # M blocks per super tile; each of the _XCDS tiles of a super tile goes to its own XCD.
    xcd_m: int = 1
    # Four-wave groups per CTA, each on its own contiguous K range (summed in LDS).
    k_groups: int = 1
    # K per pipeline stage (128 or 256).
    tile_k: int = TILE_K


DEFAULT_CONFIG = GemmConfig()


def gemm_kernel_name(cfg=DEFAULT_CONFIG, x_block_scale=False):
    return format_kernel_name(
        f"gemm_a8w4_ptpc_bm{cfg.block_m}_bn{cfg.block_n}_tk{cfg.tile_k}"
        f"_p{cfg.prefetch}_x{cfg.x_stages}_g{cfg.xcd_m}_kg{cfg.k_groups}"
        + ("_1x32" if x_block_scale else "")
    )


def _lds_bytes(cfg):
    """(X buffers, K-group partial sums) LDS bytes; the two share one allocation."""
    x = cfg.k_groups * 2 * cfg.block_m * cfg.tile_k
    part = cfg.k_groups // 2 * cfg.block_m * cfg.block_n * 4
    return x, part


def _out_row_bytes(cfg):
    """LDS row pitch of the BF16 output tile (+16 B: rows start in different banks)."""
    return 2 * cfg.block_n + 16


def _shuffled_store(cfg):
    """Whether the epilogue goes through LDS: when the output tile fits the allocation."""
    return cfg.block_m * _out_row_bytes(cfg) <= max(_lds_bytes(cfg))


def gemm_config_supported(n, k, cfg):
    """Whether ``compile_gemm_a8w4_ptpc`` accepts ``cfg`` for a weight [n, k]."""
    tiles = k // cfg.tile_k
    return (
        n > 0
        and n % N_ALIGN == 0
        and k > 0
        and k % K_ALIGN == 0
        and (
            cfg.block_m in (32, 64, 128)
            or (cfg.block_m in (16, 48) and cfg.tile_k == 256)
        )
        and cfg.block_n in (64, 128, 256)
        and cfg.prefetch >= 1
        and cfg.x_stages >= 1
        and _XCDS % cfg.xcd_m == 0
        and cfg.tile_k in (128, 256)
        and k % cfg.tile_k == 0
        and cfg.k_groups in (1, 2, 4)
        and tiles % cfg.k_groups == 0
        and max(_lds_bytes(cfg)) <= 65536
    )


@functools.cache
def compile_gemm_a8w4_ptpc(n, k, device_index, cfg=DEFAULT_CONFIG, x_block_scale=False):
    """Launcher ``(xq, x_scale, wq, wd, ws, y, M, stream)`` for weights of ``n`` channels x ``k``.

    ``xq`` FP8 E4M3FNUZ [M, k], ``x_scale`` f32 [M] (per token) or, with ``x_block_scale``,
    uint8 E8M0 [M, k / 32] (1x32, row-major; byte 255 gives an undefined result),
    ``wq`` / ``wd`` / ``ws`` the fd2 layout of ``layout.py``, ``y`` BF16 [M, n]; all
    contiguous, passed as raw ``fx.Pointer``. ``cfg`` is a ``GemmConfig``
    (``gemm_config_supported``). ``device_index`` only keys the cache (a compiled handle
    launches on one GPU).
    """
    del device_index
    if not gemm_config_supported(n, k, cfg):
        raise ValueError(
            f"A8W4 GEMM needs N % {N_ALIGN} == 0, K % {K_ALIGN} == 0 and a supported "
            f"config, got N={n}, K={k}, {cfg}"
        )
    block_m, block_n, prefetch, x_stages, xcd_m, k_groups, tile_k = cfg
    block_scale = bool(x_block_scale)
    # x_block_scale with one K group: the epilogue runs unguarded (pipeline rule 8).
    unguarded = block_scale and k_groups == 1
    fp8 = fx.Float8E4M3FNUZ
    threads = _THREADS * k_groups
    kb_total = k // 64
    kq_total = layout.delta_dwords(k)
    k_iters = tile_k // 64
    # K tiles per K group; group g owns tiles [g * group_tiles, (g + 1) * group_tiles).
    group_tiles = k // tile_k // k_groups
    w_bytes = layout.weight_bytes(n, k)
    d_bytes = layout.delta_bytes(n, k)
    m_reps = block_n // 64
    n_reps = block_m // 16
    n_blocks = (n + block_n - 1) // block_n
    ragged_n = n % block_n != 0
    w_block_units = layout.W_BLOCK_BYTES // 8
    d_block_units = layout.D_BLOCK_BYTES // 4
    num_slots = prefetch + 1
    unroll = math.lcm(num_slots, 2, x_stages)
    xcd_n = _XCDS // xcd_m
    super_n = (n_blocks + xcd_n - 1) // xcd_n
    # Loop iterations of ``unroll`` stages with every load; the rest is peeled.
    loop_iters = max(group_tiles // unroll - 1, 0)
    tail_stages = group_tiles - loop_iters * unroll
    x_tile_bytes = block_m * tile_k
    num_acc_tiles = m_reps * n_reps
    shuffled = _shuffled_store(cfg)
    # x_block_scale: E8M0 dwords per token and K tile (one byte per 32 K).
    xs_tile_dwords = tile_k // 128

    @fx.struct
    class SharedStorage:
        a: fx.Array[fp8, max(_lds_bytes(cfg)), 16]

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
    def gemm_tile(x_tile, w_buf, d_buf, blk_n, lds, xs_buf=None):
        tid = gpu.thread_idx.x % _THREADS
        group = gpu.thread_idx.x // _THREADS
        mma_atom, tiled_mma = _make_tiled_mma()

        # X: global -> VGPR (16 B per lane, 4096 / tile_k rows per pass) -> swizzled LDS.
        a_tile = fx.flat_divide(x_tile, fx.make_tile(block_m, tile_k))[
            None, None, 0, None
        ]
        buf_cp_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        val_per_thr = 16
        thrs_k = tile_k // val_per_thr
        thrs_m = _THREADS // thrs_k
        g2r_tv_layout = fx.make_layout(
            ((thrs_k, thrs_m), (1, val_per_thr)),
            ((thrs_m * val_per_thr, 1), (1, thrs_m)),
        )
        a_g2r = fx.make_tiled_copy(
            buf_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, tile_k)
        )
        a_mem_thr = a_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = [
            fx.make_fragment_like(a_mem_thr[None, None, None, 0])
            for _ in range(x_stages)
        ]

        # XOR the 16-byte chunk of a row with the row (128-byte rows: 8 chunks, 256: 16).
        swz_bits = (tile_k // 16).bit_length() - 1
        swz = fx.SwizzleType.get(swz_bits, 4, swz_bits)
        # This K group's two X buffers.
        lds_x = [lds.a.ptr + (group * 2 + b) * x_tile_bytes for b in range(2)]

        def _lds_tile(ptr):
            return fx.make_view(
                ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((block_m, tile_k), order=(1, 0)),
                ),
            )

        a_bufs = [_lds_tile(ptr) for ptr in lds_x]
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fp8)
        a_r2s = fx.make_tiled_copy(
            uni_cp_atom, g2r_tv_layout, fx.make_tile(thrs_m, tile_k)
        )
        a_lds_w = [a_r2s.get_slice(tid).partition_D(v) for v in a_bufs]
        a_cp_frag_retile = [a_r2s.get_slice(tid).retile(f) for f in a_cp_frag]

        # MMA-side K (v, c, g, h, step) -> stored K v + 8c + 32g + 16h + 128 step; with
        # x_block_scale v + 32c + 8g + 64h + 128 step (MMA c of half h: one 32-K block).
        k_shape = (8, 2, 4, 2, tile_k // 128)
        if const_expr(block_scale):
            k_stride = (1, 32, 8, 64, 128)
        else:
            k_stride = (1, 8, 32, 16, 128)

        def _read_view(ptr):
            return fx.make_view(
                ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_layout((block_m, k_shape), (tile_k, k_stride)),
                ),
            )

        if const_expr(block_scale):
            # A lane's two MMAs of a sub-step read two 8-byte pieces 32 bytes apart.
            rd_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fp8)
        else:
            rd_atom = uni_cp_atom
        a_s2r = fx.make_tiled_copy_B(rd_atom, tiled_mma).get_slice(tid)
        a_lds_r = [a_s2r.partition_S(_read_view(ptr)) for ptr in lds_x]
        a_frag = tiled_mma.make_fragment_B(a_bufs[0])
        a_frag_retile = a_s2r.retile(a_frag)

        # Decoded FP8 weight operand of the current tile, (8, m_reps, (2, k_iters)), and the
        # accumulators, (4, m_reps, n_reps); the two views only give the fragment shapes.
        w_shape = fx.make_view(
            lds.a.ptr, fx.make_layout((block_n, tile_k), (tile_k, 1))
        )
        w_frag = tiled_mma.make_fragment_A(w_shape)
        acc_shape = fx.make_view(
            lds.a.ptr, fx.make_layout((block_n, block_m), (block_m, 1))
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
        k_base = group * group_tiles
        w_lane = (lane_group // 2) * w_block_units + (lane_group % 2) * 32 + row
        d_lane = (lane_group % 2) * 16 + row
        lut_sel = [fx.Uint32((lane_group // 2 + 2 * p) * 0x01010101) for p in range(2)]
        if const_expr(group_tiles % 2 == 1 and tile_k == 128):
            # Odd groups start on an odd K tile: their step parity is flipped.
            odd = (group % 2) == 1
            lut_sel = [odd.select(lut_sel[1 - p], lut_sel[p]) for p in range(2)]
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Uint32)
        d_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Uint32)
        d_view = fx.logical_divide(d_buf, fx.make_layout(1, 1))
        # FP8 LUT of the latest decoded step per m (reused by its second half).
        lut_regs = [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(m_reps)]
        # Raw FP4 dwords and delta dwords per [slot][m]; tile k uses slot k % num_slots.
        w_regs = [
            [
                [fx.make_rmem_tensor(2, fx.Uint32) for _ in range(k_iters)]
                for _ in range(m_reps)
            ]
            for _ in range(num_slots)
        ]
        d_regs = [
            [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(m_reps)]
            for _ in range(num_slots)
        ]

        def row_block16(m):
            return blk_n * (block_n // 16) + m * 4 + wave

        def load_weights(slot, k_tile):
            # k_tile: the group's K tile; absolute tile k_base + k_tile.
            kb0 = (k_base + k_tile) * k_iters
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

        def load_x(stage, k_tile):
            fx.copy(
                buf_cp_atom,
                a_mem_thr[None, None, None, k_base + k_tile],
                a_cp_frag[stage],
            )

        def decode_weights(slot, parity, ki):
            # parity = group K-tile index % 2 (static); the first half of a 128-K step builds
            # the LUT, the second half reuses it.
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

        if const_expr(block_scale):
            # ======== x_block_scale: per-MMA weight dwords, LUTs and E8M0 scales ========
            # Lane (g, r), MMA c of 64-K block kb: dword g % 2 of fd2 lane group 2c + g // 2
            # (K 8g .. 8g + 7 of 32-K group 2 kb + c); its delta is byte kb % 4 of the delta
            # dword of half c. Tile k of the group starts at block kb0 = (k_base + k) k_iters.
            w_lane_bs = (lane_group // 2) * 32 + row * 2 + lane_group % 2
            w_view = fx.logical_divide(w_buf, fx.make_layout(1, 1))
            xs_view = fx.logical_divide(xs_buf, fx.make_layout(1, 1))
            sel_bs = [
                [
                    fx.Uint32(((p * k_iters + ki) % 4) * 0x01010101)
                    for ki in range(k_iters)
                ]
                for p in range(2)
            ]
            if const_expr(group_tiles % 2 == 1 and tile_k == 128):
                odd_bs = (group % 2) == 1
                sel_bs = [
                    [
                        odd_bs.select(sel_bs[1 - p][ki], sel_bs[p][ki])
                        for ki in range(k_iters)
                    ]
                    for p in range(2)
                ]
            # [slot][m][ki][c] weight dwords, [slot][m][c] delta dwords and [slot][ni][i] the
            # E8M0 dwords of tokens ni * 16 + row (byte b: 32-K block 4 i + b of the tile).
            w_regs_bs = [
                [
                    [
                        [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(2)]
                        for _ in range(k_iters)
                    ]
                    for _ in range(m_reps)
                ]
                for _ in range(num_slots)
            ]
            d_regs_bs = [
                [
                    [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(2)]
                    for _ in range(m_reps)
                ]
                for _ in range(num_slots)
            ]
            s_regs = [
                [
                    [fx.make_rmem_tensor(1, fx.Uint32) for _ in range(xs_tile_dwords)]
                    for _ in range(n_reps)
                ]
                for _ in range(num_slots)
            ]
            # One MMA's unscaled sum and the zero it starts from.
            part = tiled_mma.make_fragment_C(acc_shape)
            zero_c = tiled_mma.make_fragment_C(acc_shape)
            zero_c.fill(0)

            def load_weights_bs(slot, k_tile):
                kb0 = (k_base + k_tile) * k_iters
                for m in range_constexpr(m_reps):
                    rb = row_block16(m)
                    for ki in range_constexpr(k_iters):
                        for c in range_constexpr(2):
                            unit = (
                                (rb * kb_total + kb0 + ki) * (layout.W_BLOCK_BYTES // 4)
                                + w_lane_bs
                                + c * 64
                            )
                            fx.copy(
                                d_atom,
                                fx.slice(w_view, (None, unit)),
                                w_regs_bs[slot][m][ki][c],
                            )
                    for c in range_constexpr(2):
                        unit = (rb * kq_total + kb0 // 4) * d_block_units + c * 16 + row
                        fx.copy(
                            d_atom,
                            fx.slice(d_view, (None, unit)),
                            d_regs_bs[slot][m][c],
                        )
                for ni in range_constexpr(n_reps):
                    for i in range_constexpr(xs_tile_dwords):
                        unit = (
                            (ni * 16 + row) * (k // 128)
                            + (k_base + k_tile) * xs_tile_dwords
                            + i
                        )
                        fx.copy(
                            d_atom, fx.slice(xs_view, (None, unit)), s_regs[slot][ni][i]
                        )

            def decode_weights_bs(slot, parity, ki):
                # parity = group K-tile index % 2 (static): selects byte kb % 4.
                for m in range_constexpr(m_reps):
                    for c in range_constexpr(2):
                        lut = layout.fp8_lut(
                            d_regs_bs[slot][m][c].load()[0], sel_bs[parity][ki]
                        )
                        raw = w_regs_bs[slot][m][ki][c].load()[0]
                        w_frag[None, m, (c, ki)].store(layout.fp4x8_to_fp8(raw, lut))

            def block_scales(slot):
                # [ni][b]: 2**(e - 127) of 32-K block b, as f32 bits max(e << 23, 2**-127):
                # exact for every e < 255 (e = 0 is the subnormal 2**-127).
                scales = []
                for ni in range_constexpr(n_reps):
                    row_scales = []
                    for b in range_constexpr(tile_k // 32):
                        word = s_regs[slot][ni][b // 4].load()[0]
                        shift = 23 - 8 * (b % 4)
                        if const_expr(shift >= 0):
                            bits = word << shift
                        else:
                            bits = word >> -shift
                        bits = fx.max(
                            bits & fx.Uint32(0x7F800000), fx.Uint32(0x00400000)
                        )
                        row_scales.append(bits.bitcast(fx.Float32))
                    scales.append(row_scales)
                return scales

            def read_x_bs(buf, ki):
                # Partitions ((8, 2), n_reps, k_iters): MMA c's 8 bytes are piece c of mode 0.
                fx.copy(
                    rd_atom,
                    a_lds_r[buf][None, None, ki],
                    a_frag_retile[None, None, ki],
                )

            def mma_block_scaled(ki, scales):
                # MMA c covers 32-K block 2 ki + c: part = W_c X_c from zero, then
                # acc += part * 2**(e - 127) (one FMA per two values).
                for c in range_constexpr(2):
                    fx.gemm(
                        mma_atom,
                        part,
                        w_frag[None, None, (c, ki)],
                        a_frag[None, None, (c, ki)],
                        zero_c,
                    )
                    for m in range_constexpr(m_reps):
                        for ni in range_constexpr(n_reps):
                            s = scales[ni][2 * ki + c]
                            acc[None, m, ni].store(
                                fx.fma(
                                    Vec(part[None, m, ni].load()),
                                    Vec.from_elements([s, s, s, s], fx.Float32),
                                    Vec(acc[None, m, ni].load()),
                                )
                            )

            load_weights = load_weights_bs
            decode_weights = decode_weights_bs

        mfma_per_ki = 2 * m_reps * n_reps
        dsrd_per_ki = n_reps

        def valu_for(ki):
            # Region ki decodes block ki + 1 (or the next tile's block 0); first halves add the LUT.
            return m_reps * (20 if (ki + 1) % 2 == 0 else 16)

        a_vmem_cnt = block_m * tile_k // (_THREADS * 16)
        w_vmem_cnt = m_reps * (k_iters + 1)

        # MFMAs the stage's loads are spread over: all of them, or with a one-stage W
        # prefetch the first ones (W(k+1) is decoded at the end of stage k).
        def vmem_span(n_vmem):
            return mfma_per_ki * k_iters if prefetch > 1 else n_vmem

        def hot_loop_scheduler(load_x_, load_w, has_next):
            # This stage's loads go out evenly between its MFMAs (back to back they fill the
            # CU's vector memory queue and stall the MFMA issue of all its waves); the next
            # 64-K block's LDS reads and FP4 decode hide under the current block's MFMAs (the
            # last block decodes the next tile's first block); LDS writes close the stage.
            n_vmem = a_vmem_cnt * int(load_x_) + w_vmem_cnt * int(load_w)
            n_dswr = a_vmem_cnt * int(has_next)
            span = vmem_span(n_vmem)
            rocdl.sched_dsrd(dsrd_per_ki)
            for ki in range_constexpr(k_iters):
                last = ki + 1 == k_iters
                valu_per_ki = valu_for(ki)
                for j in range_constexpr(mfma_per_ki):
                    rocdl.sched_mfma(1)
                    jj = ki * mfma_per_ki + j
                    if const_expr(jj < span):
                        due = (jj + 1) * n_vmem // span - jj * n_vmem // span
                        for _ in range_constexpr(due):
                            rocdl.sched_vmem(1)
                    if const_expr(not last or has_next):
                        cnt = (valu_per_ki * (j + 1)) // mfma_per_ki - (
                            valu_per_ki * j
                        ) // mfma_per_ki
                        if const_expr(cnt > 0):
                            rocdl.sched_group_barrier("valu", cnt, 0)
                    if const_expr(not last and j % 2 == 1 and j // 2 < dsrd_per_ki):
                        rocdl.sched_dsrd(1)
                    if const_expr(last and j >= mfma_per_ki - n_dswr):
                        rocdl.sched_dswr(1)

        def pipeline_stage(j, k_tile, do_x, do_w, has_next):
            # Stage of group K tile k (k % unroll == j, k_tile = k as a runtime value).
            buf = j % 2
            if const_expr(do_x):
                load_x(j % x_stages, k_tile + x_stages)  # VMEM X(k+XS)
            if const_expr(do_w):
                load_weights(
                    (j + prefetch) % num_slots, k_tile + prefetch
                )  # VMEM W(k+P)
            if const_expr(block_scale):
                scales = block_scales(j % num_slots)  # E8M0 of X(k), loaded with W(k)
            for ki in range_constexpr(k_iters):
                if const_expr(block_scale):
                    read_x_bs(buf, ki)  # LR X(k) half ki, two 8-byte pieces
                else:
                    fx.copy(
                        uni_cp_atom,
                        a_lds_r[buf][None, None, ki],
                        a_frag_retile[None, None, ki],
                    )  # LR X(k) half ki
                if const_expr(ki > 0):
                    decode_weights(j % num_slots, j % 2, ki)  # W(k) half ki
                if const_expr(block_scale):
                    mma_block_scaled(ki, scales)
                else:
                    fx.gemm(
                        mma_atom,
                        acc,
                        w_frag[None, None, (None, ki)],
                        a_frag[None, None, (None, ki)],
                        acc,
                    )
            if const_expr(has_next):
                decode_weights((j + 1) % num_slots, (j + 1) % 2, 0)  # W(k+1) half 0
                fx.copy(
                    uni_cp_atom,
                    a_cp_frag_retile[(j + 1) % x_stages],
                    a_lds_w[(j + 1) % 2],
                )  # LW X(k+1)
            if const_expr(not block_scale):
                hot_loop_scheduler(do_x, do_w, has_next)
            rocdl.sched_barrier(0)
            gpu.barrier()

        # ======== Prologue: X(0) -> LDS, X(1 .. XS-1) staged, W(0 .. P-1) in slots ========
        load_x(0, fx.Int32(0))
        load_weights(0, fx.Int32(0))
        for i in range_constexpr(1, max(x_stages, prefetch)):
            if const_expr(i < x_stages and i < group_tiles):
                load_x(i, fx.Int32(i))
            if const_expr(i < prefetch and i < group_tiles):
                load_weights(i, fx.Int32(i))
        fx.copy(uni_cp_atom, a_cp_frag_retile[0], a_lds_w[0])
        decode_weights(0, 0, 0)
        gpu.barrier()

        # ======== Main loop: ``unroll`` stages per iteration, every stage loads ========
        if const_expr(loop_iters > 0):
            acc_init = [acc.load()]
            results = acc_init
            for iv, state in range(0, loop_iters, 1, init=acc_init):
                acc.store(state[0])
                kt = fx.Int32(iv * unroll)
                for j in range_constexpr(unroll):
                    pipeline_stage(j, kt + j, True, True, True)
                results = yield [acc.load()]
            # A loop with one carried value yields it bare, not as a one-element list.
            acc.store(results[0] if isinstance(results, (list, tuple)) else results)

        # ======== Tail: the last stages stop loading past the group's last tile ========
        for j in range_constexpr(tail_stages):
            kk = loop_iters * unroll + j
            pipeline_stage(
                j % unroll,
                fx.Int32(kk),
                kk + x_stages < group_tiles,
                kk + prefetch < group_tiles,
                kk + 1 < group_tiles,
            )

        # ======== K groups: pairwise tree through LDS, group 0 ends with the sum ========
        if const_expr(k_groups > 1):
            # One 16-byte unit per lane and accumulator tile, lane-contiguous (no conflicts).
            part = fx.recast_iter(fx.Float32, lds.a.ptr)

            def part_ptr(slot, t):
                return part + ((slot * 4 + wave) * num_acc_tiles + t) * 256 + lane * 4

            vec4 = fx.Vector.make_type(4, fx.Float32)
            for rnd in range_constexpr(k_groups.bit_length() - 1):
                # Groups [half, active) hand their sums to groups [0, half): fixed order,
                # deterministic. Every wave adds (keeps acc out of branches); the sums of
                # groups past half are not used again.
                active = k_groups >> rnd
                half = active // 2
                if (group >= half) & (group < active):
                    for m in range_constexpr(m_reps):
                        for ni in range_constexpr(n_reps):
                            fx.ptr_store(
                                Vec(acc[None, m, ni].load()),
                                part_ptr(group - half, m * n_reps + ni),
                            )
                gpu.barrier()
                slot = group % half
                for m in range_constexpr(m_reps):
                    for ni in range_constexpr(n_reps):
                        acc[None, m, ni].store(
                            Vec(acc[None, m, ni].load())
                            + Vec(
                                fx.ptr_load(
                                    part_ptr(slot, m * n_reps + ni), result_type=vec4
                                )
                            )
                        )
                if const_expr(half > 1):
                    gpu.barrier()
        return acc

    out_pitch = _out_row_bytes(cfg) // 4
    out_units = block_m * (block_n // 8)

    def load_scales(blk_n, s_buf, xs_buf):
        # Lane l's token scales (token ni * 16 + l % 16; none with x_block_scale, applied in
        # the K loop) and channel scales (channels 4 (l // 16) .. + 3 of its 16-channel block
        # of each m). Loading them before the K loop instead delays its first loads
        # (tp2_qkv M = 32: GEMM +1 us).
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        if block_scale:
            x_scales = None
        else:
            x_scales = [
                buf_copy_load(xs_buf, ni * 16 + lane % 16, elem=fx.Float32)
                for ni in range_constexpr(n_reps)
            ]
        w_scales = [
            buf_copy_load(
                s_buf,
                blk_n * (block_n // 4) + m * 16 + wave * 4 + lane // 16,
                elem=fx.Float32,
                unit_elems=4,
            )
            for m in range_constexpr(m_reps)
        ]
        return x_scales, w_scales

    def bf16_out(c, w_scale, x_scales, ni):
        # One MMA tile's 4 sums -> BF16: times the channel factors and, per token, the
        # token scale (x_block_scale applied its block scales in the K loop).
        if block_scale:
            vals = [(c[v] * w_scale[v]).to(fx.BFloat16) for v in range(4)]
        else:
            vals = [
                (c[v] * w_scale[v] * x_scales[ni]).to(fx.BFloat16) for v in range(4)
            ]
        return Vec.from_elements(vals, fx.BFloat16)

    @flyc.jit
    def stage_out_tile(acc, blk_n, s_buf, xs_buf, lds):
        # The scaled BF16 tile goes to LDS, token-major (pitch out_pitch dwords): lane l
        # writes channels 4 (l // 16) .. + 3 of its 16-channel block for token l % 16.
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        x_scales, w_scales = load_scales(blk_n, s_buf, xs_buf)
        out_lds = fx.recast_iter(fx.Uint32, lds.a.ptr)
        tokens = [ni * 16 + lane % 16 for ni in range(n_reps)]
        for m in range_constexpr(m_reps):
            w_scale = w_scales[m]
            col = (m * 64 + wave * 16 + lane // 16 * 4) // 2
            for ni in range_constexpr(n_reps):
                c = acc[None, m, ni].load()
                out = bf16_out(c, w_scale, x_scales, ni)
                fx.ptr_store(
                    Vec(out).bitcast(fx.Uint32), out_lds + tokens[ni] * out_pitch + col
                )

    @flyc.jit
    def store_out_rows(blk_n, y_buf, lds):
        # Each lane stores 16 B (8 channels) of a token row: a wave covers whole rows.
        tid = gpu.thread_idx.x
        out_lds = fx.recast_iter(fx.Uint32, lds.a.ptr)
        vec4 = fx.Vector.make_type(4, fx.Uint32)
        for it in range_constexpr((out_units + _THREADS - 1) // _THREADS):
            u = tid + it * _THREADS
            row = u // (block_n // 8)
            chunk = u % (block_n // 8)
            data = fx.ptr_load(out_lds + row * out_pitch + chunk * 4, result_type=vec4)
            unit = row * (n // 8) + blk_n * (block_n // 8) + chunk
            if const_expr(ragged_n or out_units % _THREADS != 0):
                ok = u < out_units
                if const_expr(ragged_n):
                    ok = ok & (blk_n * (block_n // 8) + chunk < n // 8)
                if ok:
                    buf_copy_store(y_buf, unit, Vec(data), elem=fx.Uint32, unit_elems=4)
            else:
                buf_copy_store(y_buf, unit, Vec(data), elem=fx.Uint32, unit_elems=4)

    @flyc.jit
    def store_tile(acc, blk_n, s_buf, xs_buf, y_buf):
        # Lane l holds channels 4 * (l // 16) + v of token l % 16 in every (m, n) MMA tile.
        tid = gpu.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        x_scales, w_scales = load_scales(blk_n, s_buf, xs_buf)
        tokens = [ni * 16 + lane % 16 for ni in range(n_reps)]
        for m in range_constexpr(m_reps):
            ch4 = blk_n * (block_n // 4) + m * 16 + wave * 4 + lane // 16
            w_scale = w_scales[m]
            for ni in range_constexpr(n_reps):
                c = acc[None, m, ni].load()
                out = bf16_out(c, w_scale, x_scales, ni)
                unit = tokens[ni] * (n // 4) + ch4
                if const_expr(ragged_n and block_scale):
                    # Channels past N store to a unit past the buffer, which drops it, rather
                    # than branch (pipeline rule 8).
                    unit = (ch4 < n // 4).select(unit, fx.Int32(block_m * (n // 4)))
                    buf_copy_store(y_buf, unit, out, elem=fx.BFloat16, unit_elems=4)
                elif const_expr(ragged_n):
                    if ch4 < n // 4:
                        buf_copy_store(y_buf, unit, out, elem=fx.BFloat16, unit_elems=4)
                else:
                    buf_copy_store(y_buf, unit, out, elem=fx.BFloat16, unit_elems=4)

    @flyc.kernel(
        name=gemm_kernel_name(cfg, block_scale), known_block_size=[threads, 1, 1]
    )
    def gemm_a8w4_ptpc_kernel(
        p_x: fx.Pointer,
        p_xs: fx.Pointer,
        p_wq: fx.Pointer,
        p_wd: fx.Pointer,
        p_ws: fx.Pointer,
        p_y: fx.Pointer,
        M: fx.Int32,
    ):
        # Super tile s = c // _XCDS covers xcd_m M blocks x xcd_n N blocks; CTA c takes its
        # tile c % _XCDS, so XCD x always gets M blocks = x // xcd_n and N blocks = x % xcd_n
        # (mod xcd_m, xcd_n): an X tile is read through xcd_n XCD L2s, a W tile through xcd_m.
        cta = fx.block_idx.x
        xcd = cta % _XCDS
        sup = cta // _XCDS
        m_blk = sup // super_n * xcd_m + xcd // xcd_n
        blk_n = sup % super_n * xcd_n + xcd % xcd_n
        row0 = m_blk * block_m
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        # Edge super tiles have CTAs without a tile (uniform per CTA). Two nested runtime
        # branches (scf.if) on purpose: `and` would emit one branch on the combined condition.
        if blk_n < fx.Int32(n_blocks):  # noqa: SIM102
            if row0 < M:
                # Per-workgroup buffers start at the tile's first row: offsets stay 32-bit
                # for any M, and rows at or past M read zeros and drop their stores.
                nrows = fx.min(M - row0, fx.Int32(block_m))
                x_tile = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        fx.recast_iter(fp8, p_x) + fx.Int64(row0) * k,
                        fx.make_layout((block_m, k), (k, 1)),
                    ),
                    num_records_bytes=nrows * k,
                )
                if const_expr(block_scale):
                    # 4-byte weight units (one dword per MMA) and the E8M0 rows of the tile.
                    w_buf = ptr_buf_tensor(
                        p_wq, elem=fx.Uint32, num_records_bytes=w_bytes
                    )
                else:
                    w_buf = ptr_buf_tensor(
                        p_wq, elem=fx.Uint32, unit_elems=2, num_records_bytes=w_bytes
                    )
                d_buf = ptr_buf_tensor(p_wd, elem=fx.Uint32, num_records_bytes=d_bytes)
                s_buf = ptr_buf_tensor(
                    p_ws, elem=fx.Float32, unit_elems=4, num_records_bytes=n * 4
                )
                if const_expr(block_scale):
                    xs_buf = ptr_buf_tensor(
                        buf_base_i64(p_xs) + fx.Int64(row0) * (k // 32),
                        elem=fx.Uint32,
                        num_records_bytes=nrows * (k // 32),
                    )
                else:
                    xs_buf = ptr_buf_tensor(
                        buf_base_i64(p_xs) + fx.Int64(row0) * 4,
                        elem=fx.Float32,
                        num_records_bytes=nrows * 4,
                    )
                acc = gemm_tile(x_tile, w_buf, d_buf, blk_n, lds, xs_buf)

                if const_expr(shuffled):
                    # Whole 128-byte lines per store instead of 32-byte pieces of 16 rows
                    # (tp2_qkvz M = 64: GEMM -0.5 us). The K loop ends with a barrier; the
                    # K-group sums need one more before the tile overwrites them.
                    y_buf = ptr_buf_tensor(
                        buf_base_i64(p_y) + fx.Int64(row0) * (n * 2),
                        elem=fx.Uint32,
                        unit_elems=4,
                        num_records_bytes=nrows * (n * 2),
                    )
                    if const_expr(k_groups > 1):
                        gpu.barrier()
                    if const_expr(unguarded):
                        stage_out_tile(acc, blk_n, s_buf, xs_buf, lds)
                    else:
                        if gpu.thread_idx.x < _THREADS:
                            stage_out_tile(acc, blk_n, s_buf, xs_buf, lds)
                    gpu.barrier()
                    if gpu.thread_idx.x < _THREADS:
                        store_out_rows(blk_n, y_buf, lds)
                else:
                    y_buf = ptr_buf_tensor(
                        buf_base_i64(p_y) + fx.Int64(row0) * (n * 2),
                        elem=fx.BFloat16,
                        unit_elems=4,
                        num_records_bytes=nrows * (n * 2),
                    )
                    if const_expr(unguarded):
                        store_tile(acc, blk_n, s_buf, xs_buf, y_buf)
                    else:
                        if gpu.thread_idx.x < _THREADS:
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
        num_m_blocks = (M + block_m - 1) // block_m
        gemm_a8w4_ptpc_kernel(p_x, p_xs, p_wq, p_wd, p_ws, p_y, M).launch(
            grid=((num_m_blocks + xcd_m - 1) // xcd_m * super_n * _XCDS, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
