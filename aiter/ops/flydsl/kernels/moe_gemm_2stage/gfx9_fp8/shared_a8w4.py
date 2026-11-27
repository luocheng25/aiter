# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The step between the two dense GEMMs of the A8W4 MoE's shared FP8 expert (gfx942).

``aiter.ops.flydsl.moe_a8w4`` runs a shared expert in FP8 with per-row scales (DeepSeek-V4.1-Flash, chosen by every
token) as two FP8 GEMMs (aiter's FlyDSL preshuffle GEMM, ``moe_a8w4._shared_gemm``) on the routed path's quantized
input and folds the second one's BF16 output into the routed path's last reduction (``_SharedFold``). ``compile_shared_mid`` sits between them: it
reads the Gate/Up GEMM's BF16 [M, 2I], computes SiLU(gate) * up in f32 (with DeepSeek-V4.1's SwiGLU limit), quantizes
each row to FP8 for the Down GEMM (zero-padded to the K of its weight, so the GEMM needs no padding copy) and folds the
shared expert's routing weight into the row scale: the Down GEMM then writes ``weight * shared(x)`` directly. A token
whose shared expert is dropped (ATOM's rule: first routed id -1, or the last id not E) gets an all-zero row and scale 0,
so the Down GEMM writes +0. Optionally it also sets the shared expert's column of ``loc_ids`` (the row of the routes
buffer that ``a8w4_route_sum`` / ``sorted_sum`` add for it).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh
from . import common_a8w4 as a8w4
from .decode_a8w4 import (
    _LOG2E,
    _abs_max,
    _fp8_words,
    _global_ptr,
    _row_scale,
    _wave_index,
    _wave_max,
)


def compile_shared_mid(*, I, KP, COLS, E, SWIGLU_LIMIT=0.0, LOC=False):
    """Launcher ``(gate_up, topk_ids, topk_weight, q, scale, loc, rows, base, stream)``.

    gate_up BF16 [rows, 2 * I] (gate columns [0, I), up columns [I, 2I); ``I % 8 == 0``) -> q FP8 E4M3FNUZ [rows, KP]
    (``KP >= I``, ``KP % 8 == 0``; columns [I, KP) zero) and scale f32 [rows]. Per row: g, u in f32 (with
    ``SWIGLU_LIMIT`` L > 0: ``g = min(g, L)``, ``u = clamp(u, -L, L)``), h = SiLU(g) * u, quantized with the
    arithmetic of ``quant_a8w4`` (``s = amax / 240``, an all-zero row 1, ``q = RNE(h / s)``, -0 stored as +0); the
    stored scale is ``s * w``, w the row's shared-expert weight: ``topk_weight[t, COLS - 1]`` where
    ``topk_ids[t, 0] >= 0`` and ``topk_ids[t, COLS - 1] == E`` (int32 / f32 [rows, COLS], contiguous), else 0 with h
    taken as 0 (an all-zero q row: the Down GEMM's product is +0 whatever its weight). With ``LOC`` int32 ``loc``
    [rows, COLS] gets ``base + t`` in column COLS - 1 of row t where the shared expert is live, -1 elsewhere (``loc``
    and ``base`` are unused without it). One wave per row, four rows per workgroup; each lane loads 16-byte pieces
    (8 values) of gate and up and stores 8 FP8 bytes per piece.
    """
    assert I % 8 == 0 and KP % 8 == 0 and KP >= I, f"unsupported I={I} KP={KP}"
    assert COLS >= 2
    PIECES = I // 8
    LOADS = (PIECES + 63) // 64
    PAD = (KP - I) // 8  # zero pieces of 8 bytes past the row's I columns
    assert PAD <= 64, f"KP={KP} pads I={I} by more than one wave of pieces"
    ROWS = 4  # waves (rows) per workgroup
    HALF = I // 2  # Uint32 words of the gate half of a row

    @flyc.kernel(name=a8w4.swiglu_kernel_name("a8w4_shared_mid", SWIGLU_LIMIT))
    def a8w4_shared_mid(
        p_in: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_loc: fx.Pointer,
        rows: fx.Int32,
        base: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        lane = tid % 64
        row = gpu.block_idx.x * ROWS + _wave_index(tid)
        if row < rows:
            ids = fx.recast_iter(fx.Int32, fxh._as_ptr(p_ids))
            tws = fx.recast_iter(fx.Float32, fxh._as_ptr(p_weight))
            live = (ids[row * COLS] >= 0) & (ids[row * COLS + (COLS - 1)] == E)
            weight = live.select(tws[row * COLS + (COLS - 1)], fx.Float32(0.0))
            # 2I BF16 = I Uint32 words per row; the up half starts HALF words in.
            src = _global_ptr(p_in, fx.Uint32, 16) + fx.Int64(row) * I
            pieces, values = [], []
            for m in range_constexpr(LOADS):
                piece = lane + 64 * m
                if const_expr(64 * (m + 1) > PIECES):
                    # Duplicates of the last piece leave the row maximum unchanged and are not stored.
                    piece = (piece < PIECES).select(piece, fx.Int32(PIECES - 1))
                pieces.append(piece)
                gate_words = fx.ptr_load(
                    src + piece * 4, result_type=Vec.make_type(4, fx.Uint32)
                )
                up_words = fx.ptr_load(
                    src + (HALF + piece * 4), result_type=Vec.make_type(4, fx.Uint32)
                )
                vals = []
                for w in range_constexpr(4):
                    gw, uw = fx.Uint32(gate_words[w]), fx.Uint32(up_words[w])
                    halves = (
                        ((gw << 16), (uw << 16)),
                        (gw & fx.Uint32(0xFFFF0000), uw & fx.Uint32(0xFFFF0000)),
                    )
                    for g_bits, u_bits in halves:
                        gate, up = a8w4.swiglu_clamp(
                            g_bits.bitcast(fx.Float32),
                            u_bits.bitcast(fx.Float32),
                            SWIGLU_LIMIT,
                        )
                        # Bare v_exp_f32 / v_rcp_f32 (raw rocdl), as the decode Gate/Up's epilogue: fx.math.exp2 and the
                        # f32 division lower to the denormal-scaled exp and the correctly rounded divide.
                        tmp = rocdl.exp2(T.f32, _raw(gate * (-_LOG2E)))
                        h = (gate * rocdl.rcp(T.f32, 1.0 + tmp)) * up
                        vals.append(live.select(h, fx.Float32(0.0)))
                values.append(vals)
            scale = _row_scale(
                _wave_max(_abs_max([v for vals in values for v in vals]))
            )
            dst = _global_ptr(p_q, fx.Uint32, 8) + fx.Int64(row) * (KP // 4)
            for m in range_constexpr(LOADS):
                stored = _fp8_words(values[m], scale)
                if const_expr(64 * (m + 1) <= PIECES):
                    fx.ptr_store(stored, dst + pieces[m] * 2)
                else:
                    if lane + 64 * m < PIECES:
                        fx.ptr_store(stored, dst + pieces[m] * 2)
            if const_expr(PAD > 0):  # noqa: SIM102
                if lane < PAD:
                    fx.ptr_store(
                        Vec.from_elements([fx.Uint32(0), fx.Uint32(0)], fx.Uint32),
                        dst + (PIECES + lane) * 2,
                    )
            if lane == 0:
                scale_out = fx.make_view(
                    fxh._as_ptr(p_scale, fx.Float32) + fx.Int64(row),
                    fx.make_layout(1, 1),
                )
                scale_out[0] = scale * weight
                if const_expr(LOC):
                    loc_out = fx.make_view(
                        fxh._as_ptr(p_loc, fx.Int32)
                        + fx.Int64(row * COLS + (COLS - 1)),
                        fx.make_layout(1, 1),
                    )
                    loc_out[0] = live.select(base + row, fx.Int32(-1))

    @flyc.jit
    def launch(
        p_in: fx.Pointer,
        p_ids: fx.Pointer,
        p_weight: fx.Pointer,
        p_q: fx.Pointer,
        p_scale: fx.Pointer,
        p_loc: fx.Pointer,
        rows: fx.Int32,
        base: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()
        a8w4_shared_mid(p_in, p_ids, p_weight, p_q, p_scale, p_loc, rows, base).launch(
            grid=((rows + ROWS - 1) // ROWS, 1, 1),
            block=(64 * ROWS, 1, 1),
            stream=stream,
        )

    return launch
