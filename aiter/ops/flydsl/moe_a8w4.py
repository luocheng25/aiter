# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 A8W4 MoE (FlyDSL): FP8 per-token activations x MXFP4 weights, BF16 in and out.

``prepare_moe_a8w4_ptpc_weight`` converts standard MXFP4 MoE weights once (at load time) into the private layout
of ``kernels/moe_gemm_2stage/gfx9_fp8/common_a8w4.py``. ``flydsl_moe_a8w4_ptpc`` runs the whole MoE. Few tokens
skip the sorting and use the decode kernels of ``gfx9_fp8/decode_a8w4.py`` when they support the shape (Gate/Up per
route, Down per token summing its routes in the workgroup and writing BF16; activations quantized inside the kernels
or by a row kernel): below hidden 4096 while the Gate/Up grid stays below ``DECODE_MAX_WGS`` workgroups, other hidden
sizes up to ``DECODE_MAX_M`` tokens, the ``DECODE_WIDE_HIDDEN`` sizes up to ``DECODE_WIDE_MAX_M`` (their Gate/Up reads
each expert's weights once per pair of its routes from ``DECODE_WIDE_DEDUP_MIN_M`` tokens on,
``gfx9_fp8/decode_dedup_a8w4.py``). Larger token
counts with at most ``STREAM_MAX_ROWS`` rows per expert (M * topk / E; ``STREAM_MAX_ROWS_WIDE`` for hidden >= 4096,
``STREAM_MAX_ROWS_H5120`` for hidden >= 5120 with many routes, ``STREAM_MAX_ROWS_FUSED_QUANT`` when the Gate/Up of
blocks of 64 quantizes the intermediate; at most ``STREAM_BM32_ROWS`` from padded I 512 on) use the
weight-streaming kernels of ``gfx9_fp8/stream_a8w4.py``: one prep launch sorts the routes into small blocks,
quantizes the input and zeroes the output, the Gate/Up writes SiLU(gate) * up (with larger blocks quantized to FP8 in
the same kernel), and the Down adds each route into the output with BF16x2 atomics. The rest run the prefill kernels:
the streaming path's prep (sorting into blocks of ``block_m`` rows, the FP8 input and ``loc_ids``), the A8W4 Gate/Up
kernel (SiLU(gate) * up -> f32, optionally with the DeepSeek-V4.1 SwiGLU limit), per-row FP8 quantization of the
intermediate, the A8W4 Down kernel (routing weight applied; it writes the routes by sorted row and records each
route's sorted row in ``loc_ids``) and ``a8w4_route_sum``, which adds every token's routes. The
calls issue no host synchronization. A routing id of -1 masks its route (ATOM's padding rows): no tier reads its
weights or adds it (the decode kernels load through zero-byte views and leave it out of the deduplication plan, the
prep does not sort it).

``aiter.fused_moe`` is the other entry: ``prepare_moe_a8w4_ptpc_weight`` tags the ``w1`` / ``w2`` it returns with the
attribute ``WEIGHT_TAG`` (as ``shuffle_weight`` sets ``is_shuffled``), and ``fused_moe(hidden_states, weight.w1,
weight.w2, topk_weight, topk_ids, activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
quant_type_a=QuantType.per_Token, swiglu_limit=...)`` calls ``flydsl_moe_a8w4_ptpc`` on them on gfx942 (the kernels and
results of a direct call), before its custom op, its tuned-config lookup and its sorting (``_fused_moe_route`` lists
how every other ``fused_moe`` parameter is treated). Untagged MXFP4 weights with that quantization raise there,
pointing here.

A shared expert in FP8 with per-row scales (DeepSeek-V4.1-Flash) joins the weight as expert E, after the E routed
experts (``prepare_moe_a8w4_ptpc_weight(..., shared_w1=..., shared_w1_scale=..., shared_w2=..., shared_w2_scale=...)``,
the E385 / topk-7 convention of ATOM and aiter's FHMoE): routing tensors then have one more column, the last, that
holds E. Up to ``DECODE_SHARED_MIXED_MAX_M`` tokens, while tokens times routed topk times padded I stay within
``DECODE_SHARED_MIXED_MAX_WORK`` and the routed experts take the decode tier, the shared expert runs as one more route
of two decode launches (``decode_a8w4``'s mixed Gate/Up and Down: its FP8 weights feed the MFMA directly,
``_fused_moe_decode_mixed``). Otherwise the routed experts run as above on the other columns (the tier decided on them
alone; ``_fold_tier``) and the shared expert is folded into their last reduction (``_SharedFold``): two FlyDSL FP8 GEMMs
in chunks of at most 65536 rows (aiter's preshuffle GEMM, ``_shared_gemm``; the Down from SHARED_DOWN_MIN_M_SHORT_K /
SHARED_DOWN_MIN_M rows on ``gfx9_fp8/shared_a8w4.py``'s own, ``_shared_down``) on the routed path's FP8 input with
``shared_a8w4``'s mid kernel between them, on a side stream when the routed experts take the decode or the streaming
tier (on the current stream behind the prefill tier), write its weighted output where ``a8w4_route_sum`` adds it as
one more route (prefill), or into the output that the routed Down then adds onto (the decode tier's slot-pair Down,
the streaming tier's atomic Down).
"""

import contextlib
import functools
import inspect
import math
import numbers
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.moe_gemm_2stage.gfx9_fp8 import common_a8w4 as layout
from .kernels.moe_gemm_2stage.gfx9_fp8 import (
    decode_a8w4,
    decode_dedup_a8w4,
    gemm2_a8w4,
    shared_a8w4,
    stream_a8w4,
)
from .kernels.moe_gemm_2stage.gfx9_fp8.common import _TORCH_TO_FX
from .kernels.moe_gemm_2stage.gfx9_fp8.gemm1_a8w4 import compile_a8w4_gemm1
from .kernels.moe_gemm_2stage.gfx9_fp8.quant_a8w4 import compile_row_quant_fp8
from .kernels.tensor_shim import (
    _run_compiled,
    current_raw_stream,
    current_stream_arg,
    ptr_arg,
)

__all__ = [
    "MoeA8W4PtpcWeight",
    "flydsl_moe_a8w4_ptpc",
    "prepare_moe_a8w4_ptpc_weight",
]

SUPPORTED_GFX = ("gfx942",)
_PAD_I = 64
_FP4_DTYPES = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
_E8M0_DTYPES = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
_E8M0_NAN = 255
# The kernels address activations, the intermediate and its quantization with 32-bit byte offsets.
_MAX_BYTES = 1 << 31
# Hidden sizes below 4096 take the decode kernels (no sorting: a workgroup per route and 16 Gate/Up channels, one per
# token in Down) while the per-route Gate/Up grid, M * topk * padded I / 16 workgroups, stays below DECODE_MAX_WGS
# (Qwen TP2 / TP4, E513 topk 11, I 320 / 192: up to M 11 / 19), the streaming kernels beyond (blocks of 16). Measured in
# one process against the decode kernels (Gate/Up on pre-quantized rows from DECODE_MAX_WGS workgroups on), streaming /
# decode: TP2 M 12 / 16 / 24 / 32 0.95 / 0.95 / 0.90 / 0.85x the routed experts' summed kernel time (CUDA graph replay
# M 12 / 16 / 32 0.90 / 0.89 / 0.77x), TP4 M 20 / 22 / 24 / 32 1.00 / 0.97 / 0.97 / 0.89x (graph M 20 / 24 / 32 0.87 /
# 0.86 / 0.78x); below the limit the decode kernels are faster (TP2 M 8 1.07x, TP4 M 16 0.99-1.03x streamed).
DECODE_MAX_WGS = 2560
# Other hidden sizes (from 4096 on, DECODE_WIDE_HIDDEN aside) take the decode kernels up to DECODE_MAX_M tokens, with
# the Down that quantizes its rows at every M (MiMo TP8 / TP4, H 4096 E256 topk 8, I 256 / 512), the streaming
# kernels beyond. Measured in one process, streaming / decode: TP8 M 8 / 10 / 12 / 16 1.02 / 1.07 / 1.01 / 0.94x the
# summed kernel time but 0.94 / 0.92 / 0.84 / 0.80x under CUDA graph replay, TP4 1.04 / 1.08 / 1.02 / 0.94x and 0.94 /
# 0.96 / 0.91 / 0.83x (from M 8 on graph replay favours streaming more than the kernel times favour decode); at M 6 / 7
# streaming took 1.19 / 1.10x (TP8) and 1.02-1.08 / 1.01x (TP4) the summed kernel time of the decode kernels. The Down
# that quantizes its rows takes 0.84-0.97x the summed kernel time of the slot-pair Down at M 4-7.
DECODE_MAX_M = 7
# The decode path quantizes activations in separate row kernels instead of inside every workgroup: Down's
# intermediate rows from DECODE_PREQUANT_MIN_M tokens on (the slot-pair Down; below hidden 4096 and for the
# DECODE_WIDE_HIDDEN sizes), Gate/Up's token rows from DECODE_MAX_WGS Gate/Up workgroups on.
DECODE_PREQUANT_MIN_M = 4
# Hidden sizes above 4096 that take the decode kernels (the Gate/Up decode then stages the FP8 token row in two
# rounds of 16-byte pieces); measured: DeepSeek-V4.1 TP4 (H5120). Other hidden sizes above 4096 stream (or prefill).
DECODE_WIDE_HIDDEN = (5120,)
# From DECODE_WIDE_DEDUP_MIN_M tokens on, those hidden sizes deduplicate the decode Gate/Up by expert
# (``decode_dedup_a8w4``: pairs of an expert's routes, 4 waves) and keep the per-token slot-pair Down (padded I =
# 576). Measured at H5120 E384 topk 6 against the per-route kernels: M4 / 8 / 16 / 32 1.03x / 0.985x / 0.967x / 0.95x
# the time (8 waves 7-11% slower).
DECODE_WIDE_DEDUP_MIN_M = 8
# From this topk * padded I on the deduplicated Gate/Up issues only its first decode_a8w4.GATEUP_PREFETCH_PAIRS block
# pairs' loads before staging the token rows (each further pair as the MFMA loop consumes one), below it all of them.
# Measured against the other choice (frozen benchmark, same-GPU ABBA, two to three rounds): the main TP4 layer
# (6 * 576) M 8 / 10 and the routed ds_tp4 M 8 take 0.93x / 0.94x / 0.97x the time of all pairs first, the MTP TP4
# layer (3 * 576) M 8 0.94x the time of 6 pairs first (M 10 the same); the main / MTP TP8 layers (6 * 320, 3 * 320)
# M 8 / 10 0.97-1.00x the time with 6 pairs first.
DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I = 3072
# Those hidden sizes take the decode path up to DECODE_WIDE_MAX_M tokens, the streaming kernels beyond.
# Measured in one process at H5120 against the streaming kernels (sorting blocks of 16, 16-byte weight loads), streaming
# / decode: the DeepSeek-V4.1 main TP4 / main TP8 / MTP TP4 / MTP TP8 layers with the folded shared expert (CUDA graph
# replay of 19-24 routings, up to three runs) at M 9 1.03 / 1.05 / 1.02 / 1.01x, M 10 1.00-1.01 / 0.98-0.99 / 1.00 /
# 0.98x, M 11 0.98-0.99 / 0.93-0.95 / 0.96-0.97 / 0.95-0.97x, M 16 0.94-0.96 / 0.89-0.92 / 0.89-0.92 / 0.94-0.97x
# (MTP TP8 M 13 / 14 1.03 / 1.02x), M 32 0.84 / 0.82-0.83 / 0.79-0.81 / 0.79-0.81x; the routed experts alone (E384
# topk 6) at M 10 / 11 / 16 / 32 1.02 / 0.97-0.98 / 0.95 / 0.84x under graph replay, 1.03 / 1.00-1.02 / 1.01-1.02 /
# 0.77x by summed kernel times (the original cases' protocol, three routings; 0.97x from M 18), the frozen benchmark's
# own ds_tp4 M 16 / 32 0.93 / 0.83x (same-GPU ABBA against DECODE_WIDE_MAX_M = 32, three rounds; its decode M 16 is
# bimodal, 150-171 us, the streaming 151 us).
DECODE_WIDE_MAX_M = 10
# Above the decode tier the weight-streaming kernels run while the mean rows per expert
# (M * topk / E) stay at or below this, the prefill kernels (TILE_K 128) beyond: measured crossover
# between 22.0 and 24.7 rows per expert for TP2 and TP4 alike (E513, topk 11: M = 1024 / 1152); with blocks of 64
# Qwen TP2 (I=320) streams 27% slower at 33 and 44 rows.
STREAM_MAX_ROWS = 22
# The same with hidden >= 4096 (MiMo TP8; padded I >= 512 stops at STREAM_BM32_ROWS): measured in one process against
# the prefill kernels, 48 rows (M1536) 0.94x the time, 64 rows (M2048) 1.04x.
STREAM_MAX_ROWS_WIDE = 48
# The same for hidden 2560 to 4095 (measured at 2560) when the Gate/Up of blocks of 64 quantizes the intermediate
# itself (``_stream_fused_quant``, Qwen TP4 I=192): 33 / 44 / 55 rows (M1536 / 2048 / 2560) 0.96x / 0.92x / 0.98x.
STREAM_MAX_ROWS_FUSED_QUANT = 56
# Most mean rows per expert sorted in blocks of 32 (more: blocks of 64), and the most the streaming kernels take from
# padded I 512 on (``_stream_max_rows``): the streaming Down always adds atomically, and with blocks of 64 that costs
# registers from K 512 on (MiMo TP4 M1024: 1.27x the time of the prefill Down and route_sum), so there the prefill
# kernels follow blocks of 32. Measured against streaming blocks of 64 into the prefill Down (same-GPU ABBA): MiMo TP4
# M1024 1.01x the summed kernel time, the MTP TP4 layer (CUDA graph) M1365 0.98x. Below padded I 512 blocks of 64
# stream with the atomic Down (MiMo TP8, Qwen TP4, the MTP TP8 layer up to M1365: the prefill kernels instead 1.06x /
# 1.10x / 1.05x the time).
STREAM_BM32_ROWS = 28
# 16-row weight blocks the streaming Down keeps in flight (it loads them in pairs: half as many block pairs) with
# sorting blocks of 64 rows, and with smaller sorting blocks (STREAM_DOWN_PREFETCH_SMALL: one pair ahead, fewer VGPRs;
# measured against 4 in one process: Qwen TP2 M48 0.81x, MiMo TP8 M512 0.91x, DeepSeek TP4 M64-256 0.98x the Down's
# time, MiMo TP8 M1024 with blocks of 64 1.05x).
STREAM_DOWN_PREFETCH = 4
STREAM_DOWN_PREFETCH_SMALL = 2
# The shared streaming Down reads the activations from LDS at every 64-K step instead of holding them in VGPRs where
# that buys a workgroup per CU: blocks of 32 except 256 < K < 512 (MiMo TP8 M512 94 -> 90 us; DeepSeek I576 M1024
# 2 -> 3 workgroups, 0.89-0.91x the Down's time with and without atomics, lab), blocks of 64 when K <= 256 (the atomic
# Down, 2 -> 3 waves per SIMD: Qwen TP4 M2048 0.89x, MiMo TP8 M1024 with the rows' table in LDS 0.90x); elsewhere
# (K = 320) LDS still bounds the occupancy or the VGPRs saved do not reach the next wave and the extra reads cost 3-4%.
STREAM_DOWN_LDS_ACTS = 1
# 64-K steps per workgroup barrier of the light-wave streaming Gate/Up's LDS activation ring (2: about 1% faster;
# the fused-quantization tile keeps 1, Qwen TP4 M1024 +1.5% with 2).
STREAM_GATEUP_SYNC = 2
# 64-K steps of weights the light-wave streaming Gate/Up keeps in flight (5: VGPRs 85 -> 79 at blocks of 32, one
# more 8-wave workgroup per CU; MiMo M512 about 1-2% faster).
STREAM_GATEUP_PREFETCH = 5
# Cache policy bits of the streaming kernels' weight loads (0 cached, 2 non-temporal): non-temporal weights make
# the Gate/Up 10-24% faster (MiMo TP8 M64 / M512 / M1024 122 / 148 / 191 -> 93 / 126 / 163 us), the Down no faster.
STREAM_GATEUP_W_AUX = 2
STREAM_DOWN_W_AUX = 0
# The prefill Down splits the hidden dimension over 2 or 4 workgroups per sorted block while the
# estimated blocks (E * ceil(M * topk / E / block_m), uniform routing) times the split stay below this:
# few blocks (M ~2048) otherwise leave the last round of workgroups mostly idle (MiMo M2048 Down -10%).
# Padded I below 256 (short Down steps) gains nothing from it and stays unsplit.
DOWN_MIN_WGS = 1000
# DeepSeek-V4.1-Flash TP4 (hidden 5120, E384, topk 6, I576, swiglu_limit 10), measured in one process against the rules
# above; they apply from hidden 5120 on:
# - the streaming kernels run up to STREAM_MAX_ROWS_H5120 mean rows per expert once M * topk reaches
#   STREAM_H5120_MIN_ROUTES; streaming (16-byte weight loads) / prefill (the two-pass Down of gemm2_a8w4): the routed
#   experts alone by summed kernel times at 17 / 20 / 24 / 26 / 28 / 32 rows, M1088 / 1280 / 1536 / 1664 / 1792 /
#   2048, 0.92 / 0.94 / 0.97 / 0.99 / 1.04 / 1.03x the time; the main TP4 / TP8 layers (CUDA graph, the shared expert
#   folded on both paths) at 17 / 24 / 26 / 28 / 30 rows 0.92 / 0.97 / 1.00 / 1.04 / 1.04x and 0.87 / 0.92 / 0.94 /
#   0.97 / 1.10x; fewer routes keep STREAM_MAX_ROWS_WIDE (the MTP layer, E128 topk 3, up to M1365; TP4's padded I
#   576 stops at STREAM_BM32_ROWS: at 24 / 27 / 30 / 32 rows 0.96 / 1.00 / 1.02 / 1.02x at TP4, 0.94 / 0.93 / 1.07 /
#   1.06x at TP8);
# - the streaming Down's weight loads are non-temporal (M48-1536 2-4% faster).
STREAM_MAX_ROWS_H5120 = 24
STREAM_H5120_MIN_ROUTES = 4096
STREAM_DOWN_W_AUX_H5120 = 2
# Activation bytes per lane and load of the light-wave streaming Gate/Up from hidden 5120 on (16: one load every 4 /
# 2 steps with blocks of 16 / 32 instead of 4 / 8 bytes every step).
STREAM_GATEUP_ACT_BYTES_H5120 = 16
# Sorted blocks per XCD chunk of the prefill Gate/Up / Down from hidden 5120 on with at least XCD_MIN_ROUTES_H5120
# routes (XCD_CHUNK of gemm1_a8w4 / gemm2_a8w4; 0: the 2-D grid): a block's N tiles / hidden parts and neighbouring
# blocks of an expert run on one XCD and share its L2 (DeepSeek TP4 M4096 / 16384 / 32768: 0.991x / 0.982x / 0.961x
# the time; M2048 0.997x; with few blocks, M <= 32, some XCDs get no work: 1.01x).
GATEUP_XCD_CHUNK_H5120 = 1
DOWN_XCD_CHUNK_H5120 = 4
XCD_MIN_ROUTES_H5120 = 16384
# From hidden 5120 on with at least GATEUP_FEW_WAVES_MIN_E experts and GATEUP_FEW_WAVES_MIN_ROUTES routes (the
# DeepSeek-V4.1 main layer from M 16384 on) the prefill Gate/Up keeps GATEUP_FEW_WAVES workgroups of 4 waves per CU
# (gemm1_a8w4 WGS_PER_CU: its LDS padded, and the VGPRs this frees hold a second staged activation tile; 0: as many
# as its LDS and VGPRs allow, 4) and deals XCD chunks of GATEUP_FEW_WAVES_XCD_CHUNK sorted blocks: with fewer tiles in
# flight per XCD the experts' weights stay in its L2. Measured in one process against 4 workgroups and chunks of 1
# (Gate/Up time): DeepSeek TP4 M8192 / 16384 / 32768 1.04x / 0.97x / 0.92x, the main TP8 layer's routed experts M8192 /
# 16384 / 32768 1.04x / 0.96x / 0.88x, the main TP4 layer's M16384 0.96x (chunks of 2 alone: 0.99x / 0.98-0.99x /
# 0.93-0.95x); the second activation tile then 0.96-0.97x again (M8192 with both still 1.01x); latency-bound shapes
# are slower with 12 waves per CU (M1: MiMo H4096 M16384 / 32768 1.06x / 1.05x, the MTP layer (E128) M32768 1.03x,
# DeepSeek M4096 1.06x).
GATEUP_FEW_WAVES = 3
GATEUP_FEW_WAVES_XCD_CHUNK = 2
GATEUP_FEW_WAVES_MIN_E = 256
GATEUP_FEW_WAVES_MIN_ROUTES = 98304
# The prefill Down stages a large activation tile in two passes (gemm2_a8w4 TWO_PASS) only for grids of more
# workgroups (estimated blocks times the split) than one per CU.
DOWN_TWO_PASS_MIN_WGS = 80
# Token counts up to this, while M * routed topk * padded I stays within DECODE_SHARED_MIXED_MAX_WORK, run the shared
# expert as one more route of the per-route decode kernels (``decode_a8w4``'s mixed Gate/Up and Down: its FP8 weights
# feed the MFMA directly; two launches, no side stream, no final add) when the routed experts take the decode tier
# (``_mixed_decode``); the others fold it (``_SharedFold``). The DeepSeek-V4.1 layers, where both limits were measured,
# leave the decode tier past M 10 (DECODE_WIDE_MAX_M); the mixed decode is not measured on more tokens.
DECODE_SHARED_MIXED_MAX_M = 10
# The mixed kernels' per-token waves grow with the routed columns and I, while the fold's shared chain runs beside the
# routed kernels: the work M * routed topk * padded I tells which is faster, not M. Measured at the DeepSeek-V4.1
# layers (CUDA graph, PTL on, one process, mixed / folded): up to 10368 the mixed decode is faster (MTP TP8 M5-10
# 0.61-0.94x, MTP TP4 M5 / 6 0.85-0.86x / 0.90-0.96x, main TP8 M5 0.92-0.93x the time), main TP8 M6 (11520) and MTP
# TP4 M7 (12096) take about the same time (0.99-1.04x), main TP4 M4 (13824) and more fold faster (mixed 1.05-1.22x).
DECODE_SHARED_MIXED_MAX_WORK = 10368
# The shared expert's Down GEMM (N = hidden, K = padded shared I, short) runs shared_a8w4.compile_shared_down from
# these token counts on (``_shared_down_config``): SHARED_DOWN_MIN_M_SHORT_K rows where K <= 384, SHARED_DOWN_MIN_M
# above; fewer rows take aiter's preshuffle GEMM (``_shared_gemm``). The kernel stores 16 bytes per lane interleaved
# with the next columns' MFMAs and reads only the useful K, where the preshuffle GEMM writes 2 bytes per lane after
# its main loop and runs K padded to 128. Measured alone (eager, cold operands, PTL on), us, preshuffle GEMM -> this
# kernel: K 320 M1024 19.6 -> 13.2, M2048 34.7 -> 23.0, M4096 62.9 -> 43.8, M16384 245 -> 156, M32768 499 -> 305;
# K 576 M1024 24.6 -> 23.5, M2048 44.4 -> 41.6, M4096 80.3 -> 80.0, M8192 160 -> 148, M16384 325 -> 272, M32768 695
# -> 530. Whole layer under CUDA graph replay (one process, PTL on, exclusive GPU), preshuffle / this kernel: K 320
# MTP M1024 / 1152 / 2048 / 8192 / 32768 0.999 / 0.997 / 1.029 / 1.022 / 1.041x, main M1024 / 1536 / 4096 1.011 /
# 1.019 / 1.012x; K 576 main M1536 / 2048 / 4096 / 8192 / 16384 0.998 / 1.009 / 1.000 / 1.006 / 1.008x, MTP M1152 /
# 2048 / 32768 0.980 / 0.993 / 1.025x; the kernel alone in the layers (bench INFO, eager) at K 576: M4096 92 -> 103 us,
# M8192 177 -> 175, so K 576 takes it from 8192 rows on.
SHARED_DOWN_MIN_M_SHORT_K = 769
SHARED_DOWN_MIN_M = 8192
# Rows from which its stores carry the non-temporal bit (whole layer, without / with it: K 320 MTP M2048 / 8192 /
# 32768 1.002 / 0.996 / 1.008x, main M1536 / 4096 1.008 / 1.001x; at M1024 MTP 0.981x, main 1.016x).
SHARED_DOWN_NT_MIN_M = 2048
# Token counts up to which shared_fold_mid runs one row per wave (per-value IEEE division, as before M10); more rows
# share a wave, 8 / 16 lanes per row (``shared_a8w4.mid_row_lanes``; quotients through the reciprocal, the same
# bytes). Measured alone (eager, cold, PTL on), us, one row per wave / 16 lanes per row at I 576: M256 3.6 / 4.8,
# M1000 6.6 / 5.0, M32768 134.5 (before) / 57; 8 lanes at I 320: M256 3.0 / 4.8, M1000 4.6 / 4.5, M32768 74.9 / 31.
SHARED_MID_WAVE_ROW_MAX_M = 512
# Checks that read routing tensors on the host (a synchronization, so off by default; AITER_MOE_A8W4_DEBUG=1): routed
# ids in [0, E) or -1 (masked), no id repeated within a token apart from -1 and, with a shared expert, its id E in the
# last column of every row (``_check_routing``).
DEBUG_CHECKS = os.environ.get("AITER_MOE_A8W4_DEBUG", "0") == "1"

# Kernels launched by the most recent ``flydsl_moe_a8w4_ptpc`` call, in order (host-side record; tests and the
# benchmark read it). With a shared expert it also holds steps named ``shared*``: the fold's (``shared_fold_*``, with a
# side stream between the markers ``shared_fork`` and ``shared_join``), or ``shared_mixed`` after the two kernels
# of the mixed decode (``a8w4_gateup_decode``, ``a8w4_down_decode``), which runs the shared expert inside them.
last_path = []

# The device kernels of the most recent ``flydsl_moe_a8w4_ptpc`` call made inside ``recording_launches()``, in launch
# order, one ``KernelLaunch`` each (host-side record; every call clears it, as ``last_path``, and only a recording call
# fills it): the steps of ``last_path`` that are one kernel, one entry per kernel of the steps that launch several and
# of the copies and small ops ``last_path`` does not name; its markers (``shared_fork``, ``shared_join``,
# ``shared_mixed``) launch nothing. Only ``DEBUG_CHECKS``' host checks of the routing are left out. The benchmark's
# per-kernel roofline lines read it (the minimal bytes and useful FLOPs of each kernel). Recording reads no GPU data
# and synchronizes nothing, so a call under CUDA graph capture records its kernels too.
last_launches = []
# Whether calls fill ``last_launches`` (``recording_launches``). Off, each launch site only tests it (``if _recording:``
# after the launch) and computes nothing: a call costs the host what it did without the record.
_recording = False


@contextlib.contextmanager
def recording_launches():
    """Context in which ``flydsl_moe_a8w4_ptpc`` calls record their device kernels in ``last_launches`` (each call
    replaces the record of the previous one); yields that list. Outside it calls record nothing (``last_launches``
    stays empty) and pay one flag test per launch for it; inside it each kernel costs a few microseconds of host time
    (its bytes and FLOPs, its stream). Nests; not thread-safe (as ``last_path``)."""
    global _recording
    outer, _recording = _recording, True
    try:
        yield last_launches
    finally:
        _recording = outer


class KernelLaunch(NamedTuple):
    """One device kernel of a ``flydsl_moe_a8w4_ptpc`` call (``last_launches``), recorded on the host as the call
    launches it inside ``recording_launches()``: its stream, minimal bytes and useful FLOPs (the roofline model of the
    kernel's interface).

    - ``name``: the step's name in ``last_path`` for a step of one kernel; ``<step>/<part>`` for each kernel of a step
      that launches several (a shared GEMM in row chunks: ``<step>/chunk<i>``)
      and for kernels of steps ``last_path`` does not name (``routing_copy/*``: routing tensors made contiguous int32 /
      float32).
    - ``stream``: ``"main"``, the stream current at the call, or ``"side"``, the device's side stream of the folded
      shared expert's chain (``_side_stream``).
    - ``expert_bytes``: minimal bytes per distinct routed expert of the call's routing (the caller counts the distinct
      ids >= 0 of the routed columns): the kernel's weights, deltas and channel scales of one expert in storage format
      (I padded), plus 4 bytes of a sorting block's expert id where the kernel writes or reads sorting blocks (at least
      one block per expert).
    - ``shared_bytes``: the shared expert's FP8 weights and per-row scales the kernel reads, in storage format (I padded,
      the Down's K padded to 128; a shared GEMM run in row chunks counts them on its first chunk only, so that a call
      counts each byte once).
    - ``fixed_bytes``: the rest of the kernel's interface traffic, every distinct byte once however often the kernel
      touches it, no cache hit subtracted: activations, intermediates and outputs in their stored dtypes (x BF16, xq
      FP8 and xs f32, mid f32, mq FP8 and ms f32, routes BF16, out BF16; an output the kernel also reads first counts
      as a read and a write, an atomic add into it as one write), the routing tensors' columns it reads (int32 ids,
      f32 weights), the sorting metadata at its minimum (4 bytes per route of sorted ids and of sorted weights, 8 of
      num_valid; no padding rows), loc_ids, the dedup plan.
    - ``flops``: useful FLOPs, 2 * rows * N * K of each GEMM: rows = M * topk routes (masked ids included: the host
      does not read the ids) or M for the shared expert, N and K at the padded I (the shared expert's padded I, its
      Down's K padding to 128 not counted), MXFP4 counted as the FP8 MFMA it is decoded for, sorting-block and MFMA
      padding not counted; 0 for every other kernel. A call sums to 6 * M * H * (topk * I_pad + I_shared) (I_shared the
      shared expert's padded I, 0 without one).
    """

    name: str
    stream: str
    expert_bytes: int
    shared_bytes: int
    fixed_bytes: int
    flops: int

    def bytes(self, experts):
        """Minimal bytes of the kernel when the call's routing selects ``experts`` distinct routed experts."""
        return experts * self.expert_bytes + self.shared_bytes + self.fixed_bytes


class _Cost(NamedTuple):
    """The bytes and FLOPs of a ``KernelLaunch`` (its fields after ``stream``)."""

    expert: int = 0
    shared: int = 0
    fixed: int = 0
    flops: int = 0


def _nbytes(*tensors):
    """Bytes of ``tensors`` (None skipped), each read or written once in full."""
    total = 0
    for t in tensors:
        if t is not None:
            total += t.nbytes
    return total


def _expert_bytes(q, delta, scale):
    """Bytes of one expert of a weight's per-expert tensors [E, ...]: codes, deltas and channel scales."""
    return (
        q.nbytes // q.shape[0]
        + delta.nbytes // delta.shape[0]
        + scale.nbytes // scale.shape[0]
    )


def _gateup_cost(weight, routes, fixed, blocks=False, shared_rows=0):
    """``_Cost`` of a routed Gate/Up kernel of ``routes`` routes: one expert's w1, deltas and channel scales (and a
    sorting block's expert id with ``blocks``), ``fixed`` bytes, 2 * routes * 2I * H FLOPs; with ``shared_rows`` also the
    shared expert's on that many rows (its w1 and scales, 2 * rows * 2I_s * H FLOPs)."""
    H = weight.hidden
    shared = flops = 0
    if shared_rows:
        shared = _nbytes(weight.shared_w1, weight.shared_w1_scale)
        flops = 2 * shared_rows * weight.shared_w1.shape[0] * H
    return _Cost(
        _expert_bytes(weight.w1, weight.w1_delta, weight.w1_scale) + 4 * blocks,
        shared,
        fixed,
        flops + 4 * routes * weight.inter_padded * H,
    )


def _down_cost(weight, routes, fixed, blocks=False, shared_rows=0):
    """``_Cost`` of a routed Down kernel of ``routes`` routes: one expert's w2, deltas and channel scales (and a
    sorting block's expert id with ``blocks``), ``fixed`` bytes, 2 * routes * H * I FLOPs; with ``shared_rows`` also the
    shared expert's on that many rows (its w2 and scales, 2 * rows * H * I_s FLOPs, I_s its padded I).
    """
    H = weight.hidden
    shared = flops = 0
    if shared_rows:
        shared = _nbytes(weight.shared_w2, weight.shared_w2_scale)
        flops = 2 * shared_rows * H * (weight.shared_w1.shape[0] // 2)
    return _Cost(
        _expert_bytes(weight.w2, weight.w2_delta, weight.w2_scale) + 4 * blocks,
        shared,
        fixed,
        flops + 2 * routes * H * weight.inter_padded,
    )


def _record(device, name, cost=None):
    """Append the ``KernelLaunch`` of a kernel just launched on the current stream of ``device`` to ``last_launches``
    (callers check ``_recording`` first). ``cost``: its ``_Cost``, None for zeros (``_launch``, whose caller sets them
    with ``_cost``). The stream is "side" on the device's side stream (``_side_stream``), else "main".
    """
    side = _SIDE_STREAMS.get(device.index)
    label = (
        "side"
        if side is not None and side.cuda_stream == current_raw_stream(device)
        else "main"
    )
    last_launches.append(KernelLaunch(name, label, *(cost or _Cost())))


def _cost(cost):
    """Set the bytes and FLOPs of the kernel recorded last (by ``_launch``) to ``cost``, a ``_Cost``. Callers check
    ``_recording`` first, right after the launch: outside the recording nothing is computed.
    """
    last_launches[-1] = KernelLaunch(*last_launches[-1][:2], *cost)


def _routing_input(t, dtype, name):
    """Routing tensor ``t`` as a contiguous ``dtype`` tensor (``t.to(dtype).contiguous()``; callers take this path only
    when ``t`` is not one already), recording each copy kernel this launches as ``routing_copy/<name>``.
    """
    if t.dtype != dtype:
        src, t = t, t.to(dtype)
        if _recording:
            _record(t.device, f"routing_copy/{name}", _Cost(fixed=_nbytes(src, t)))
    if not t.is_contiguous():
        src, t = t, t.contiguous()
        if _recording:
            _record(t.device, f"routing_copy/{name}", _Cost(fixed=_nbytes(src, t)))
    return t


@dataclass(frozen=True)
class MoeA8W4PtpcWeight:
    """MoE weights in the A8W4 layout; build them with ``prepare_moe_a8w4_ptpc_weight``.

    Flat per expert (see ``common_a8w4``): ``w1`` / ``w2`` uint8 FP4 codes [E, bytes], ``w1_delta`` /
    ``w2_delta`` uint8 [E, bytes], ``w1_scale`` / ``w2_scale`` float32 [E, N] per-channel factors. ``inter`` is
    the checkpoint's intermediate size I, ``inter_padded`` I padded to a multiple of 64.

    An optional shared expert (expert E, see ``prepare_moe_a8w4_ptpc_weight``), all four or none (None): FP8 E4M3FNUZ
    ``shared_w1`` [2 * inter_padded, H] (gate rows first, then up rows, each I zero-padded to ``inter_padded``) and
    ``shared_w2`` [H, inter_padded rounded up to 128] (zero-padded K) in the layout of
    ``aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))`` that ``aiter.gemm_a8w8_bpreshuffle`` reads,
    ``shared_w1_scale`` / ``shared_w2_scale`` float32 [N, 1] per-row scales. For an FP8 [N, K] with K % 64 == 0 that
    layout is also the mixed decode kernels' 16-row x 64-K blocks (``decode_a8w4``), which read the same tensors.
    """

    w1: torch.Tensor
    w1_delta: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_delta: torch.Tensor
    w2_scale: torch.Tensor
    experts: int
    hidden: int
    inter: int
    inter_padded: int
    shared_w1: torch.Tensor | None = None
    shared_w1_scale: torch.Tensor | None = None
    shared_w2: torch.Tensor | None = None
    shared_w2_scale: torch.Tensor | None = None

    def __post_init__(self):
        shared = (
            self.shared_w1,
            self.shared_w1_scale,
            self.shared_w2,
            self.shared_w2_scale,
        )
        if any(t is None for t in shared) and any(t is not None for t in shared):
            raise ValueError(
                "shared_w1, shared_w1_scale, shared_w2 and shared_w2_scale are all given or all None"
            )

    @property
    def has_shared(self):
        """Whether the weight holds a shared expert (expert ``experts``, the last routing column)."""
        return self.shared_w1 is not None


# Attribute of the ``w1`` / ``w2`` tensors that ``prepare_moe_a8w4_ptpc_weight`` returns (as ``shuffle_weight`` sets
# ``is_shuffled``); ``aiter.fused_moe`` sends tensors carrying it to ``_fused_moe_route``.
WEIGHT_TAG = "aiter_moe_a8w4_ptpc"


@dataclass(frozen=True, eq=False)
class _WeightTag:
    """The ``WEIGHT_TAG`` value: ``role`` "w1" or "w2", and ``rest`` = ``(w1_delta, w1_scale, w2_delta, w2_scale,
    experts, hidden, inter, inter_padded, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)`` of the weight (the
    shared ones None without a shared expert), one tuple shared by the two tags of a ``prepare_moe_a8w4_ptpc_weight``
    call. It holds no reference to the tagged tensors (no reference cycle keeps them alive), so it can also be copied
    onto another tensor object of the same storage, e.g. an ``nn.Parameter`` wrapper (views, copies and wrappers do not
    carry it, as ``is_shuffled``).
    """

    role: str
    rest: tuple = field(repr=False)


def _tag_weight(weight):
    rest = (
        weight.w1_delta,
        weight.w1_scale,
        weight.w2_delta,
        weight.w2_scale,
        weight.experts,
        weight.hidden,
        weight.inter,
        weight.inter_padded,
        weight.shared_w1,
        weight.shared_w1_scale,
        weight.shared_w2,
        weight.shared_w2_scale,
    )
    setattr(weight.w1, WEIGHT_TAG, _WeightTag("w1", rest))
    setattr(weight.w2, WEIGHT_TAG, _WeightTag("w2", rest))
    return weight


def _tagged_weight(w1, w2):
    """The ``MoeA8W4PtpcWeight`` of ``w1`` / ``w2`` from one ``prepare_moe_a8w4_ptpc_weight`` call (their tags)."""
    t1, t2 = getattr(w1, WEIGHT_TAG, None), getattr(w2, WEIGHT_TAG, None)
    if not (
        isinstance(t1, _WeightTag)
        and isinstance(t2, _WeightTag)
        and (t1.role, t2.role) == ("w1", "w2")
        and t1.rest is t2.rest
    ):
        raise ValueError(
            "w1 and w2 must be the w1 and w2 of one prepare_moe_a8w4_ptpc_weight result (tensor copies and views do "
            "not carry its tag)"
        )
    w1_delta, w1_scale, w2_delta, w2_scale, E, H, I, ip, *shared = t1.rest
    if (
        w1.dtype != torch.uint8
        or w2.dtype != torch.uint8
        or tuple(w1.shape) != (E, ip * H)
        or tuple(w2.shape) != (E, H * ip // 2)
        or not (w1.is_contiguous() and w2.is_contiguous())
        or {w1.device, w2.device} != {w1_delta.device}
    ):
        raise ValueError(
            f"w1 / w2 do not match their A8W4 tag: expected contiguous uint8 [{E}, {ip * H}] / [{E}, {H * ip // 2}] "
            f"on {w1_delta.device}, got {w1.dtype} {tuple(w1.shape)} / {w2.dtype} {tuple(w2.shape)} on "
            f"{w1.device} / {w2.device}"
        )
    return MoeA8W4PtpcWeight(
        w1, w1_delta, w1_scale, w2, w2_delta, w2_scale, E, H, I, ip, *shared
    )


def _require_supported_gfx():
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_GFX:
        raise RuntimeError(
            f"FlyDSL A8W4 MoE supports {', '.join(SUPPORTED_GFX)} only, got {arch!r}"
        )


def _codes(scale, shape):
    if scale.dtype not in _E8M0_DTYPES:
        raise TypeError(
            f"MXFP4 scales must be float8_e8m0fnu or uint8, got {scale.dtype}"
        )
    if scale.numel() != math.prod(shape):
        raise ValueError(
            f"expected MXFP4 scales of shape {shape}, got {tuple(scale.shape)}"
        )
    codes = scale.view(torch.uint8).reshape(shape)
    if bool((codes == _E8M0_NAN).any()):
        raise ValueError("E8M0 scale code 255 (NaN) is not supported")
    return codes


def _packed(weight, shape):
    if weight.dtype not in _FP4_DTYPES:
        raise TypeError(
            f"MXFP4 weights must be float4_e2m1fn_x2 or uint8, got {weight.dtype}"
        )
    if tuple(weight.shape) != shape:
        raise ValueError(
            f"expected MXFP4 weight shape {shape}, got {tuple(weight.shape)}"
        )
    return weight.view(torch.uint8)


def _to_fd2_layout(q: torch.Tensor, codes: torch.Tensor):
    """FP4 bytes [n, k/2] + E8M0 codes [n, k/32] -> (q, d, s) of ``layout.py``."""
    n, half_k = q.shape
    k = half_k * 2
    kb = k // 64
    kq = layout.delta_dwords(k)
    codes = codes.to(torch.int32)
    cmax = codes.amax(dim=-1)
    diff = cmax[:, None] - codes
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).view(n, k // 32, 32)
    # -0 is stored as +0; groups more than MAX_DELTA codes below the row maximum flush to 0.
    nib = nib.masked_fill(nib == 8, 0).masked_fill(
        (diff > layout.MAX_DELTA)[..., None], 0
    )
    nib = nib.view(n, k // 8, 8)
    out_q = (nib[..., :4] | (nib[..., 4:] << 4)).view(n // 16, 16, kb, 4, 8)
    out_q = out_q.permute(0, 2, 3, 1, 4).reshape(-1)
    delta = (diff.clamp(max=layout.MAX_DELTA) * 8).to(torch.uint8).view(n, kb, 2)
    delta = F.pad(delta, (0, 0, 0, kq * 4 - kb))
    out_d = delta.view(n // 16, 16, kq, 4, 2).permute(0, 2, 4, 1, 3).reshape(-1)
    scale = torch.ldexp(torch.ones_like(cmax, dtype=torch.float64), cmax - 132)
    return out_q.contiguous(), out_d.contiguous(), scale.float().contiguous()


def _convert(q, codes, chunk=32):
    """[E, N, K/2] FP4 bytes + [E, N, K/32] E8M0 codes -> ``(q, delta, channel scale)`` [E, ...], converted
    ``chunk`` experts at a time (one expert's 16-row blocks follow the previous expert's, so each expert's layout
    is the dense layout of its own weight)."""
    E, _, half_k = q.shape
    parts = [
        _to_fd2_layout(
            q[e : e + chunk].reshape(-1, half_k),
            codes[e : e + chunk].reshape(-1, codes.shape[2]),
        )
        for e in range(0, E, chunk)
    ]
    return tuple(torch.cat(p).view(E, -1) for p in zip(*parts))


_FP8_FORMATS = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
_E4M3FN_MAX = 448.0
_SHARED_NAMES = ("shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale")


def _shared_scale_format(name, w, scale):
    """``"row"`` for float32 per-row scales ([N] or [N, 1]) of ``w`` [N, K], ``"block"`` for E8M0 scales of 32 x 32
    blocks ([N / 32, K / 32], ``float8_e8m0fnu`` or uint8 codes); ValueError otherwise.
    """
    rows, cols = w.shape
    shape = tuple(scale.shape)
    if scale.dtype == torch.float32:
        if shape not in ((rows,), (rows, 1)):
            raise ValueError(
                f"{name}_scale: per-row float32 scales of a [{rows}, {cols}] weight are [{rows}] or [{rows}, 1], "
                f"got {list(shape)}"
            )
        return "row"
    if scale.dtype in _E8M0_DTYPES:
        if rows % 32 or cols % 32:
            raise ValueError(
                f"{name}_scale: E8M0 scales of 32x32 blocks need both weight dimensions to be multiples of 32, got "
                f"[{rows}, {cols}]"
            )
        if shape != (rows // 32, cols // 32):
            raise ValueError(
                f"{name}_scale: E8M0 scales of 32x32 blocks of a [{rows}, {cols}] weight are "
                f"[{rows // 32}, {cols // 32}], got {list(shape)}"
            )
        return "block"
    raise ValueError(
        f"{name}_scale must be float32 per-row scales ([N] or [N, 1]) or E8M0 scales of 32x32 blocks ([N/32, K/32], "
        f"float8_e8m0fnu or uint8), got {scale.dtype} {list(shape)}"
    )


def _shared_inputs(tensors, hidden, inter, device):
    """Checks the shared-expert arguments of ``prepare_moe_a8w4_ptpc_weight`` (``tensors`` in ``_SHARED_NAMES``
    order) against the routed experts' H / I; returns their scale format (``_shared_scale_format``), or None without
    a shared expert. Raises ValueError for a partial set, other shapes, dtypes or devices and mixed formats (TypeError
    for an argument that is no tensor)."""
    given = [t is not None for t in tensors]
    if not any(given):
        return None
    if not all(given):
        missing = [n for n, g in zip(_SHARED_NAMES, given) if not g]
        raise ValueError(
            f"a shared expert needs all of {', '.join(_SHARED_NAMES)}; missing {', '.join(missing)}"
        )
    w1, s1, w2, s2 = tensors
    for name, t in zip(_SHARED_NAMES, tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(t).__name__}")
    for name, w in (("shared_w1", w1), ("shared_w2", w2)):
        if w.dtype not in _FP8_FORMATS or w.dim() != 2:
            raise ValueError(
                f"{name} must be a 2-D float8_e4m3fn (the checkpoint's OCP E4M3) or float8_e4m3fnuz tensor, got "
                f"{w.dtype} {list(w.shape)}"
            )
    if w1.dtype != w2.dtype:
        raise ValueError(
            f"shared_w1 and shared_w2 must have one FP8 format, got {w1.dtype} and {w2.dtype}"
        )
    formats = {
        _shared_scale_format("shared_w1", w1, s1),
        _shared_scale_format("shared_w2", w2, s2),
    }
    if len(formats) != 1:
        raise ValueError(
            "shared_w1_scale and shared_w2_scale must have one format (both per-row float32 or both E8M0 32x32 "
            f"blocks), got {s1.dtype} {list(s1.shape)} and {s2.dtype} {list(s2.shape)}"
        )
    want1, want2 = (2 * inter, hidden), (hidden, inter)
    if tuple(w1.shape) != want1 or tuple(w2.shape) != want2:
        raise ValueError(
            f"shared expert shapes must match the routed experts' H = {hidden} and I = {inter}: shared_w1 [2I, H] = "
            f"{list(want1)} (gate rows, then up rows) and shared_w2 [H, I] = {list(want2)}, got "
            f"{list(w1.shape)} and {list(w2.shape)}"
        )
    devices = {t.device for t in tensors}
    if devices != {device}:
        raise ValueError(
            f"the shared expert must be on the routed experts' GPU {device}, got {sorted(map(str, devices))}"
        )
    return formats.pop()


def _e4m3fn_to_fnuz(bits):
    """E4M3FN bytes as E4M3FNUZ: the same bits are half the value (bias 8 instead of 7), so the scale doubles; -0
    (0x80) is NaN in E4M3FNUZ and becomes +0."""
    return bits.masked_fill(bits == 0x80, 0).view(torch.float8_e4m3fnuz)


def _fold_blocks(w, codes):
    """FP8 ``w`` [N, K] with E8M0 ``codes`` of 32x32 blocks -> (E4M3FN [N, K], float32 [N] power-of-two scale per
    row).

    The value of each element is v = w * 2**(code - 127) (exact); the row scale is s = 2**ceil(log2(max|v| / 448))
    (1 for an all-zero row), the smallest power of two that keeps v / s within E4M3FN, and the element RNE(v / s).
    Elements of blocks below the row's largest move down in exponent and round where they become E4M3FN subnormals
    (relative L2 about 1e-7 of the real shared experts); FP4PTPC@97cc9b32 folded its main layers' shared experts this
    way (byte-identical).
    """
    factor = torch.exp2(codes.float() - 127)
    factor = factor.repeat_interleave(32, 0).repeat_interleave(32, 1)
    v = w.float() * factor
    amax = v.abs().amax(dim=1).double()
    a = torch.where(amax > 0, amax, torch.full_like(amax, _E4M3FN_MAX))
    k = torch.ceil(torch.log2(a / _E4M3FN_MAX))
    # log2 is rounded: step to the smallest k with a <= 448 * 2**k (exact in float64).
    k = torch.where(a > _E4M3FN_MAX * torch.exp2(k), k + 1, k)
    k = torch.where(a <= _E4M3FN_MAX * torch.exp2(k - 1), k - 1, k)
    scale = torch.exp2(k).float()
    return (v / scale[:, None]).to(torch.float8_e4m3fn), scale


def _shared_rows(name, w, scale, fmt):
    """One shared-expert matrix [N, K] -> FP8 E4M3FNUZ [N, K] and its float32 [N] per-row scales: per-row input keeps
    its elements (E4M3FN re-encoded by ``_e4m3fn_to_fnuz``, the scale doubled; E4M3FNUZ input as it is, its scale taken
    as already doubled), 32x32 block input is folded to per-row scales first (``_fold_blocks``). NaN elements, scale
    codes or scales raise ValueError (reads them on the host)."""
    w = w.contiguous()
    bits = w.view(torch.uint8)
    fnuz = w.dtype == torch.float8_e4m3fnuz
    nan = (bits == 0x80) if fnuz else ((bits & 0x7F) == 0x7F)
    if bool(nan.any()):
        raise ValueError(f"{name} holds NaN elements")
    if fmt == "row":
        s = scale.reshape(-1).float()
        if not bool(torch.isfinite(s).all()):
            raise ValueError(f"{name}_scale holds non-finite scales")
        if fnuz:
            return w, s.contiguous()
        return _e4m3fn_to_fnuz(bits), (s * 2).contiguous()
    codes = scale.contiguous().view(torch.uint8)
    if bool((codes == _E8M0_NAN).any()):
        raise ValueError(f"{name}_scale: E8M0 code 255 (NaN) is not supported")
    folded, s = _fold_blocks(w, codes)
    return _e4m3fn_to_fnuz(folded.view(torch.uint8)), s * 2


def _prepare_shared(tensors, fmt, inter, inter_padded):
    """The shared-expert fields of ``MoeA8W4PtpcWeight`` (``_SHARED_NAMES`` order) from checked inputs: per-row FP8
    E4M3FNUZ (``_shared_rows``), I zero-padded to ``inter_padded`` (padded gate / up rows and Down columns are zero,
    their gate / up scales 1), then ``shuffle_weight(layout=(16, 16))`` (the Down with ``pad_k_to=128``) and [N, 1]
    scales, the operands of ``aiter.gemm_a8w8_bpreshuffle`` and, as they are (the shuffle of an FP8 [N, K] with
    K % 64 == 0 is their 16-row x 64-K block layout), of the mixed decode kernels."""
    from aiter.ops.shuffle import shuffle_weight

    w1, s1, w2, s2 = tensors
    q1, t1 = _shared_rows("shared_w1", w1, s1, fmt)
    q2, t2 = _shared_rows("shared_w2", w2, s2, fmt)
    pad = inter_padded - inter
    if pad:
        b1 = q1.view(torch.uint8)
        z = b1.new_zeros((pad, b1.shape[1]))
        q1 = torch.cat([b1[:inter], z, b1[inter:], z]).view(torch.float8_e4m3fnuz)
        one = t1.new_ones(pad)
        t1 = torch.cat([t1[:inter], one, t1[inter:], one])
        q2 = F.pad(q2.view(torch.uint8), (0, pad)).view(torch.float8_e4m3fnuz)
    return (
        shuffle_weight(q1.contiguous(), layout=(16, 16)),
        t1.reshape(-1, 1).contiguous(),
        shuffle_weight(q2.contiguous(), layout=(16, 16), pad_k_to=128),
        t2.reshape(-1, 1).contiguous(),
    )


def prepare_moe_a8w4_ptpc_weight(
    w1,
    w1_scale,
    w2,
    w2_scale,
    *,
    shared_w1=None,
    shared_w1_scale=None,
    shared_w2=None,
    shared_w2_scale=None,
) -> MoeA8W4PtpcWeight:
    """Convert standard MXFP4 MoE weights to the A8W4 layout (run once, at load time).

    Inputs: ``w1`` [E, 2I, H/2] FP4x2 (``torch.float4_e2m1fn_x2`` or uint8; rows ``[0, I)`` gate,
    ``[I, 2I)`` up; low nibble = even K), ``w1_scale`` [E, 2I, H/32] E8M0 (``float8_e8m0fnu`` or
    uint8), ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32], all on one GPU. Requires ``H % 512 == 0`` and
    ``I % 32 == 0``. I is zero-padded to a multiple of 64 (160 -> 192): padded Gate/Up rows and
    Down columns are zero, so results equal the unpadded MoE.

    Precision: each 32-K group is decoded to FP8 relative to its row's largest E8M0 code; groups
    more than 11 codes (2**11) below that maximum are treated as zero (``common_a8w4.MAX_DELTA``); an E8M0
    code 255 (NaN) is rejected. One result serves every token count of ``flydsl_moe_a8w4_ptpc``.
    Synchronizes with the host once per scale tensor. Raises ``RuntimeError`` off gfx942.

    Shared expert (optional, all four or none; DeepSeek-V4.1-Flash): FP8 ``shared_w1`` [2I, H] (gate rows, then up
    rows: the checkpoint's w1 and w3 concatenated, ATOM's w13) and ``shared_w2`` [H, I] with the routed experts' H and
    I (a tensor-parallel slice, cut by the caller), both ``torch.float8_e4m3fn`` (the checkpoint's OCP E4M3) or both
    ``torch.float8_e4m3fnuz`` (already converted, scales taken as already doubled). The format follows the scales, the
    same for both: float32 per row ([N] or [N, 1], FP4PTPC checkpoints) or E8M0 of 32x32 blocks ([N/32, K/32],
    ``float8_e8m0fnu`` or uint8 codes; the official checkpoint), which are folded to one power-of-two scale per row
    (``_fold_blocks``: byte-identical to FP4PTPC@97cc9b32's folding of whole rows). E4M3FN becomes E4M3FNUZ (same bits,
    -0 -> 0, scale x2) and I is zero-padded to the routed experts' padded I (exact). It becomes expert E: routing tensors
    of ``flydsl_moe_a8w4_ptpc`` then have topk + 1 columns, the last holding E. Other shapes, dtypes, devices, mixed
    formats, NaN elements, scale codes or scales raise ValueError.

    The returned ``w1`` / ``w2`` carry the attribute ``WEIGHT_TAG`` (values and shapes unchanged), so that
    ``aiter.fused_moe(hidden_states, weight.w1, weight.w2, topk_weight, topk_ids,
    activation=ActivationType.Silu, quant_type=QuantType.per_1x32, quant_type_a=QuantType.per_Token)`` runs this
    op on them (``_fused_moe_route``); the scales and the shared expert travel with the tags (pass no ``w1_scale`` /
    ``w2_scale`` and no ``shared_*`` tensors there).
    """
    _require_supported_gfx()
    if w1.dim() != 3:
        raise ValueError(f"w1 must be [E, 2I, H/2], got shape {tuple(w1.shape)}")
    E, n1, half_h = w1.shape
    I, H = n1 // 2, half_h * 2
    if n1 % 2 or H % 512 or I % 32:
        raise ValueError(f"unsupported shape: H={H} (needs % 512), I={I} (needs % 32)")
    devices = {t.device for t in (w1, w1_scale, w2, w2_scale)}
    if len(devices) != 1 or w1.device.type != "cuda":
        raise ValueError(
            f"w1, w1_scale, w2, w2_scale must be on one GPU, got {sorted(map(str, devices))}"
        )
    shared = (shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)
    shared_format = _shared_inputs(shared, H, I, w1.device)
    q1 = _packed(w1, (E, 2 * I, H // 2))
    q2 = _packed(w2, (E, H, I // 2))
    c1 = _codes(w1_scale, (E, 2 * I, H // 32))
    c2 = _codes(w2_scale, (E, H, I // 32))
    ip = -(-I // _PAD_I) * _PAD_I
    if ip != I:
        pad = ip - I
        q1 = torch.cat(
            [
                q1[:, :I],
                q1.new_zeros(E, pad, H // 2),
                q1[:, I:],
                q1.new_zeros(E, pad, H // 2),
            ],
            1,
        )
        c1 = torch.cat(
            [
                c1[:, :I],
                c1.new_full((E, pad, H // 32), 127),
                c1[:, I:],
                c1.new_full((E, pad, H // 32), 127),
            ],
            1,
        )
        q2 = F.pad(q2, (0, pad // 2))
        c2 = F.pad(c2, (0, pad // 32))
    shared = (
        () if shared_format is None else _prepare_shared(shared, shared_format, I, ip)
    )
    w1q, w1d, w1s = _convert(q1.contiguous(), c1.contiguous())
    w2q, w2d, w2s = _convert(q2.contiguous(), c2.contiguous())
    return _tag_weight(
        MoeA8W4PtpcWeight(w1q, w1d, w1s, w2q, w2d, w2s, E, H, I, ip, *shared)
    )


def _trace_constants():
    """Kernel-module constants the prefill launchers read while being built or traced (XCDs per GPU, the K from
    which Down stages its activations in two passes); part of their cache keys."""
    return layout.XCDS, gemm2_a8w4.A_TWO_PASS_MIN_K


@functools.cache
def _gateup_kernel(
    N,
    K,
    topk,
    block_m,
    block_n,
    tile_k,
    experts,
    limit,
    xcd_chunk,
    device,
    constants,
    *,
    wgs_per_cu=0,
):
    """Prefill Gate/Up launcher. ``device`` (a compiled handle runs on one GPU) and ``constants``
    (``_trace_constants()``) only key the cache; ``wgs_per_cu``: gemm1_a8w4's WGS_PER_CU (``_gateup_grid``)."""
    return compile_a8w4_gemm1(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        BLOCK_TILE_SIZE_N=block_n,
        TILE_K=tile_k,
        E=experts,
        SWIGLU_LIMIT=limit,
        XCD_CHUNK=xcd_chunk,
        WGS_PER_CU=wgs_per_cu,
    )


@functools.cache
def _down_kernel(
    N,
    K,
    topk,
    block_m,
    experts,
    nsplit,
    two_pass,
    xcd_chunk,
    device,
    constants,
    *,
    loc_topk=None,
):
    """Prefill Down launcher. ``device`` and ``constants`` (``_trace_constants()``) only key the cache; ``loc_topk``:
    the columns of loc_ids (default topk; one more with a folded shared expert)."""
    return gemm2_a8w4.compile_a8w4_gemm2(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_TILE_SIZE_M=block_m,
        E=experts,
        NSPLIT=nsplit,
        TWO_PASS=two_pass,
        XCD_CHUNK=xcd_chunk,
        LOC_TOPK=loc_topk,
    )


@functools.cache
def _row_quant_kernel(cols, dtype, device):
    """Row quantization launcher (``compile_row_quant_fp8``). ``device`` only keys the cache."""
    return compile_row_quant_fp8(cols, dtype)


def _prefill_down(tokens, topk, experts, hidden, inter, block_m, device, loc_topk=None):
    """Prefill Down kernel of a call: ``_down_split``, DOWN_TWO_PASS_MIN_WGS, DOWN_XCD_CHUNK_H5120 (``loc_topk``: the
    columns of loc_ids when a folded shared expert adds one)."""
    nsplit = _down_split(tokens, topk, experts, hidden, inter, block_m)
    blocks = min(tokens * topk, experts * -(-tokens * topk // (experts * block_m)))
    extra = {} if loc_topk is None else {"loc_topk": loc_topk}
    return _down_kernel(
        hidden,
        inter,
        topk,
        block_m,
        experts,
        nsplit,
        blocks * nsplit > DOWN_TWO_PASS_MIN_WGS,
        _xcd_chunk(DOWN_XCD_CHUNK_H5120, hidden, tokens * topk),
        device,
        _trace_constants(),
        **extra,
    )


def _xcd_chunk(chunk, hidden, routes):
    """XCD_CHUNK of a prefill kernel (see ``GATEUP_XCD_CHUNK_H5120``)."""
    return chunk if hidden >= 5120 and routes >= XCD_MIN_ROUTES_H5120 else 0


def _gateup_grid(hidden, experts, routes, tile):
    """(XCD_CHUNK, WGS_PER_CU) of the prefill Gate/Up (see ``GATEUP_FEW_WAVES``; ``tile`` = (block_m, block_n,
    tile_k), the measured (64, 128, 128) only)."""
    if (
        hidden >= 5120
        and experts >= GATEUP_FEW_WAVES_MIN_E
        and routes >= GATEUP_FEW_WAVES_MIN_ROUTES
        and tile == (64, 128, 128)
    ):
        return GATEUP_FEW_WAVES_XCD_CHUNK, GATEUP_FEW_WAVES
    return _xcd_chunk(GATEUP_XCD_CHUNK_H5120, hidden, routes), 0


def _down_split(tokens, topk, experts, hidden, inter, block_m):
    """Hidden-dimension split of the prefill Down (see ``DOWN_MIN_WGS``)."""
    if inter < 256:
        return 1
    blocks = experts * -(-tokens * topk // (experts * block_m))
    nsplit = 1
    while (
        nsplit < 4
        and blocks * nsplit < DOWN_MIN_WGS
        and (hidden // 64) % (4 * nsplit) == 0
    ):
        nsplit *= 2
    return nsplit


def _decode_gateup_fits(H):
    """Whether the decode Gate/Up supports hidden size ``H`` and it was measured to take it (hidden <= 4096 or in
    DECODE_WIDE_HIDDEN)."""
    return (H <= 4096 or H in DECODE_WIDE_HIDDEN) and decode_a8w4.gateup_decode_fits(H)


def _decode_fits(H, I, topk, paired):
    """Whether the decode kernels support this shape: the Gate/Up (``_decode_gateup_fits``) and the slot-pair Down
    (``paired``) or the Down that quantizes its rows (one wave per routing slot, topk <= 16)."""
    return _decode_gateup_fits(H) and (
        decode_a8w4.down_decode_pq_fits(I, topk)
        if paired
        else decode_a8w4.down_decode_mixed_fits(I, topk, SHARED=False)
    )


@functools.cache
def _decode_kernels(N1, H, I, topk, gateup_pq, down_pq, limit, device, *, shared=False):
    """Decode Gate/Up and Down launchers of the routed experts: the per-route Gate/Up (x quantized in the kernel, or
    pre-quantized with ``gateup_pq``) and the slot-pair Down (``down_pq``) or the Down that quantizes its rows.
    ``device`` (a compiled handle runs on one GPU) keys the cache; with ``shared`` both read routing tensors of
    topk + 1 columns (the folded shared expert's last, unread) and the slot-pair Down adds the shared expert's output it
    finds in ``out`` (``_SharedFold``).
    """
    extra = {"ID_COLS": topk + 1} if shared else {}
    gateup = decode_a8w4.compile_a8w4_gateup_decode_mixed(
        N=N1,
        K=H,
        TOPK=topk,
        SWIGLU_LIMIT=limit,
        SHARED=False,
        PREQUANT=gateup_pq,
        **extra,
    )
    if down_pq:
        # More tokens: 16-byte loads of slot pairs.
        return gateup, decode_a8w4.compile_a8w4_down_decode_pq(
            N=H, K=I, TOPK=topk, SHARED=shared, **extra
        )
    assert not shared
    return gateup, decode_a8w4.compile_a8w4_down_decode_mixed(
        N=H, K=I, TOPK=topk, SHARED=False
    )


@functools.cache
def _decode_mixed_kernels(N1, H, I, KP, topk, experts, limit, device):
    """Mixed decode Gate/Up and Down launchers (a shared FP8 expert, id ``experts``, in the last of ``topk`` columns;
    its Down weight's K padded to ``KP``). ``device`` only keys the cache."""
    gateup = decode_a8w4.compile_a8w4_gateup_decode_mixed(
        N=N1, K=H, TOPK=topk, E=experts, SWIGLU_LIMIT=limit, SHARED=True
    )
    down = decode_a8w4.compile_a8w4_down_decode_mixed(
        N=H, K=I, KP=KP, TOPK=topk, E=experts, SHARED=True
    )
    return gateup, down


@functools.cache
def _gateup_dd_kernel(N1, H, topk, limit, device, *, shared=False):
    """Deduplicated decode Gate/Up launcher of the DECODE_WIDE_HIDDEN sizes. ``device`` only keys the cache; ``shared``
    as in ``_decode_kernels``. From DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I (topk times padded I) on it issues only its
    first decode_a8w4.GATEUP_PREFETCH_PAIRS block pairs' loads up front."""
    prefetch = (
        decode_a8w4.GATEUP_PREFETCH_PAIRS
        if topk * (N1 // 2) >= DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I
        else None
    )
    return decode_dedup_a8w4.compile_a8w4_gateup_decode_dd(
        N=N1,
        K=H,
        TOPK=topk,
        SWIGLU_LIMIT=limit,
        PREFETCH=prefetch,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


@functools.cache
def _row_quant_plan(cols, topk, max_m, device, *, shared=False):
    """Row quantization of x that also writes the dedup plan of up to ``max_m`` tokens. ``device`` only keys the
    cache; ``shared`` as in ``_decode_kernels``."""
    return decode_dedup_a8w4.compile_a8w4_row_quant_plan(
        COLS=cols,
        TOPK=topk,
        MAX_M=max_m,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


@functools.cache
def _down_pq_kernel(H, I, topk, device, *, shared=False):
    """Slot-pair decode Down of the deduplicated Gate/Up of the DECODE_WIDE_HIDDEN sizes. ``device`` only keys the
    cache; ``shared`` as in ``_decode_kernels``.
    """
    return decode_a8w4.compile_a8w4_down_decode_pq(
        N=H,
        K=I,
        TOPK=topk,
        SHARED=shared,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


def _fused_moe_decode_wide_dd(
    hidden_states, weight, topk_weight, topk_ids, output, limit, shared=None
):
    """Decode of the DECODE_WIDE_HIDDEN sizes from DECODE_WIDE_DEDUP_MIN_M tokens: one kernel quantizes x and writes
    the plan, Gate/Up is deduplicated by expert (``decode_dedup_a8w4``), the row kernel quantizes the intermediate and
    the slot-pair Down sums each token's routes. A folded shared expert (``shared``, ``_SharedFold``) starts on the plan
    kernel's FP8 input, writes ``output`` and the Down adds the routes onto it; the kernels read the routing tensors
    with its column (the last) as they are.
    """
    M, H = hidden_states.shape
    topk = topk_ids.shape[1] - (shared is not None)
    I = weight.inter_padded
    on = {} if shared is None else {"shared": True}
    device = hidden_states.device
    plan = torch.empty((M * topk, 4), dtype=torch.int32, device=device)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    _launch(
        "a8w4_row_quant_plan_decode",
        _row_quant_plan(H, topk, DECODE_WIDE_MAX_M, device.index, **on),
        hidden_states,
        xq,
        xs,
        topk_ids,
        plan,
        M,
    )
    if _recording:
        # The routed columns' ids: 4 bytes per route.
        _cost(_Cost(fixed=_nbytes(hidden_states, xq, xs, plan) + 4 * M * topk))
    if shared is not None:
        shared.start(xq, xs, output)
    _launch(
        "a8w4_gateup_decode_dd",
        _gateup_dd_kernel(2 * I, H, topk, limit, device.index, **on),
        xq,
        xs,
        topk_ids,
        plan,
        weight.w1,
        weight.w1_delta,
        weight.w1_scale,
        mid,
        M,
    )
    if _recording:
        _cost(_gateup_cost(weight, M * topk, _nbytes(xq, xs, plan, mid) + 4 * M * topk))
    if shared is not None:
        shared.fork()
    mq, ms = _quantize_decode(mid, M * topk, I, False)
    if shared is not None:
        shared.join()
    _launch(
        "a8w4_down_decode_pq" + ("" if shared is None else "_shared"),
        _down_pq_kernel(H, I, topk, device.index, **on),
        mq,
        ms,
        topk_ids,
        topk_weight,
        weight.w2,
        weight.w2_delta,
        weight.w2_scale,
        output,
        M,
    )
    if _recording:
        # The SHARED variant reads the shared expert's output from ``output`` before writing it.
        _cost(
            _down_cost(
                weight,
                M * topk,
                _nbytes(mq, ms)
                + 8 * M * topk
                + _nbytes(output) * (1 + (shared is not None)),
            )
        )
    return output


def _slot_pair_offsets_fit(weight):
    """Whether the slot-pair Down (``compile_a8w4_down_decode_pq``) can address all experts' w2 with 32-bit byte
    offsets: its range check ends one expert's bytes below 2**32 (where a masked route's offsets fall).
    """
    w2_bytes = weight.w2.numel() * weight.w2.element_size()
    return w2_bytes + layout.weight_bytes(weight.hidden, weight.inter_padded) <= 2**32


def _paired_down(M, H):
    """Whether the decode tier's routed experts take the slot-pair Down on pre-quantized rows (where both Downs hold
    the shape and no shared expert is folded): from DECODE_PREQUANT_MIN_M tokens on below hidden 4096 and for the
    DECODE_WIDE_HIDDEN sizes; the other hidden sizes keep the Down that quantizes its rows (DECODE_MAX_M).
    """
    return M >= DECODE_PREQUANT_MIN_M and (H < 4096 or H in DECODE_WIDE_HIDDEN)


def _fused_moe_decode(
    hidden_states, weight, topk_weight, topk_ids, output, limit, shared=None
):
    """The decode path: Gate/Up per route into the f32 intermediate, Down per token summing its routes into
    ``output`` (from DECODE_WIDE_DEDUP_MIN_M tokens on the DECODE_WIDE_HIDDEN sizes deduplicate the Gate/Up);
    ``topk_ids`` / ``topk_weight`` are contiguous int32 / float32. A folded shared expert (``shared``, ``_SharedFold``;
    only where ``_decode_fold`` holds) keeps its column in them, the last, which the decode kernels skip; the routed
    experts then take the slot-pair Down at any M (not the Down that quantizes its rows) and, unless the H 5120 Gate/Up
    deduplicates, the per-route Gate/Up on pre-quantized rows. The chain takes the FP8 input of the routed Gate/Up (from
    the plan kernel of the deduplicated Gate/Up, else from the row kernel), writes ``output`` and the slot-pair Down
    adds the routes onto it.
    """
    M, H = hidden_states.shape
    topk = topk_ids.shape[1] - (shared is not None)
    I = weight.inter_padded
    wide = (
        H in DECODE_WIDE_HIDDEN
        and M >= DECODE_WIDE_DEDUP_MIN_M
        and _slot_pair_offsets_fit(weight)
        and _decode_fits(H, I, topk, True)
    )
    if wide:
        return _fused_moe_decode_wide_dd(
            hidden_states, weight, topk_weight, topk_ids, output, limit, shared
        )
    # Few tokens: each workgroup quantizes its own activation rows; more tokens, or a folded shared expert's chain that
    # takes those rows too: quantize once.
    gateup_pq = shared is not None or M * topk * (I // 16) >= DECODE_MAX_WGS
    # The slot-pair Down on pre-quantized rows where ``_paired_down`` says, at any M with a folded shared expert or
    # where only it holds the shape; it addresses all experts' weights with 32-bit byte offsets, so larger weights take
    # the Down that quantizes its rows (one wave per slot) at any M.
    single = shared is None and _decode_fits(H, I, topk, False)
    down_pq = (
        (not single or _paired_down(M, H))
        and _slot_pair_offsets_fit(weight)
        and _decode_fits(H, I, topk, True)
    )
    assert shared is None or down_pq
    if not (down_pq or single):
        # Neither Down holds this shape with this weight (w2 past the slot-pair Down's offsets and more slots than the
        # other takes): the streaming kernels, which take any M and padded I.
        return _fused_moe_stream(
            hidden_states, weight, topk_weight, topk_ids, output, limit
        )
    gateup, down = _decode_kernels(
        2 * I,
        H,
        I,
        topk,
        gateup_pq,
        down_pq,
        limit,
        hidden_states.device.index,
        **({} if shared is None else {"shared": True}),
    )
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    if gateup_pq:
        xq, xs = _quantize_decode(hidden_states, M, H, True)
    else:
        xq, xs = hidden_states, topk_weight
    if shared is not None:
        shared.start(xq, xs, output)
    _launch(
        "a8w4_gateup_decode" + ("_pq" if gateup_pq else ""),
        gateup,
        xq,
        xs,
        topk_ids,
        weight.w1,
        weight.w1_delta,
        weight.w1_scale,
        mid,
        M,
    )
    if _recording:
        # Without pre-quantized rows the kernel reads x (xs is a placeholder); the routed columns' ids.
        _cost(
            _gateup_cost(
                weight,
                M * topk,
                _nbytes(xq, xs if gateup_pq else None, mid) + 4 * M * topk,
            )
        )
    if shared is not None:
        shared.fork()
    if down_pq:
        mq, ms = _quantize_decode(mid, M * topk, I, False)
    else:
        mq, ms = mid, topk_weight
    if shared is not None:
        shared.join()
    _launch(
        "a8w4_down_decode"
        + ("_pq" if down_pq else "")
        + ("" if shared is None else "_shared"),
        down,
        mq,
        ms,
        topk_ids,
        topk_weight,
        weight.w2,
        weight.w2_delta,
        weight.w2_scale,
        output,
        M,
    )
    if _recording:
        # The Down that quantizes its rows reads the f32 intermediate (ms is the routing weight); the SHARED variant
        # reads the shared expert's output from ``output`` before writing it.
        _cost(
            _down_cost(
                weight,
                M * topk,
                _nbytes(mq, ms if down_pq else None)
                + 8 * M * topk
                + _nbytes(output) * (1 + (shared is not None)),
            )
        )
    return output


def _fused_moe_decode_mixed(
    hidden_states, weight, topk_weight, topk_ids, output, limit
):
    """The mixed decode path of a call with a shared expert (``_mixed_decode``): ``topk_ids`` / ``topk_weight`` with
    the shared expert's column last, its FP8 weights read in place (the (16, 16) shuffle of ``shared_w1`` /
    ``shared_w2`` is the decode kernels' 16-row x 64-K block layout), Gate/Up into the f32 intermediate, Down per token
    summing every column into ``output``."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = weight.inter_padded
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = _routing_input(topk_ids, torch.int32, "topk_ids")
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = _routing_input(topk_weight, torch.float32, "topk_weight")
    gateup, down = _decode_mixed_kernels(
        2 * I,
        H,
        I,
        weight.shared_w2.shape[1],
        topk,
        weight.experts,
        limit,
        hidden_states.device.index,
    )
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    _launch(
        "a8w4_gateup_decode",
        gateup,
        hidden_states,
        topk_ids,
        weight.w1,
        weight.w1_delta,
        weight.w1_scale,
        weight.shared_w1,
        weight.shared_w1_scale,
        mid,
        M,
    )
    if _recording:
        # The routed columns' GEMMs plus the shared expert's on the M rows; both kernels read every column.
        _cost(
            _gateup_cost(
                weight,
                M * (topk - 1),
                _nbytes(hidden_states, topk_ids, mid),
                shared_rows=M,
            )
        )
    _launch(
        "a8w4_down_decode",
        down,
        mid,
        topk_ids,
        topk_weight,
        weight.w2,
        weight.w2_delta,
        weight.w2_scale,
        weight.shared_w2,
        weight.shared_w2_scale,
        output,
        M,
    )
    if _recording:
        _cost(
            _down_cost(
                weight,
                M * (topk - 1),
                _nbytes(mid, topk_ids, topk_weight, output),
                shared_rows=M,
            )
        )
    # The shared expert ran inside the two kernels (their SHARED variants, named as the per-route ones): a marker,
    # named as the fold's steps are.
    last_path.append("shared_mixed")
    return output


@functools.cache
def _row_quant_decode(cols, bf16, device):
    """Decode row quantization launcher (``compile_a8w4_row_quant``). ``device`` only keys the cache."""
    return decode_a8w4.compile_a8w4_row_quant(COLS=cols, BF16=bf16)


def _quantize_decode(x, rows, cols, bf16):
    """``_quantize`` with the decode path's vectorized kernel (same arithmetic, bit-identical)."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    _launch(
        "a8w4_row_quant_decode",
        _row_quant_decode(cols, bf16, x.device.index),
        x,
        q,
        scale,
        rows,
    )
    if _recording:
        _cost(_Cost(fixed=_nbytes(x, q, scale)))
    return q, scale


def _stream_fused_quant(inter, block_m, hidden=0):
    """Whether the streaming Gate/Up kernel quantizes the intermediate itself: four waves (G = I / 64 gate
    blocks each) cover all of I, when their accumulators fit (8 * G * sub-blocks <= 96 VGPRs). From hidden
    4096 on, waves of one gate block plus the row kernel are faster (measured: MiMo TP8 M512 191 -> 153 us).
    """
    return block_m > 16 and hidden < 4096 and 8 * (inter // 64) * (block_m // 16) <= 96


def _stream_gateup_constants():
    """Module constants ``_stream_gateup_kernel`` reads while building its launcher; part of its cache key."""
    return (
        STREAM_GATEUP_ACT_BYTES_H5120,
        STREAM_GATEUP_PREFETCH,
        STREAM_GATEUP_SYNC,
        STREAM_GATEUP_W_AUX,
    )


@functools.cache
def _stream_gateup_kernel(N, K, topk, block_m, experts, limit, device, constants):
    """Streaming Gate/Up launcher. ``device`` (a compiled handle runs on one GPU) and ``constants``
    (``_stream_gateup_constants()``) only key the cache."""
    # Fused quantization: one workgroup of four waves per block covers all of I and quantizes the intermediate
    # in the epilogue; with up to 4 + 4 row blocks per wave the activations are a large share of the traffic, so
    # the waves share them in LDS. Otherwise many light waves (one gate and one up block each, four per
    # workgroup, eight with larger blocks) sharing the block's activations in LDS.
    # Blocks of 64: the shared activations two 64-K steps at a time (a row's 128 bytes, whole cache lines, instead
    # of half a line per step: the half-line requests jam the vector memory queue in front of the weight loads) and
    # the LUT by VALU, which keeps 4 waves' workgroups at 16 KiB of LDS (4 per CU instead of 3); lab, kernel alone:
    # MTP TP8 M1365 0.88x, MTP TP4 M1280 / M1365 0.90x, MiMo TP8 / TP4 M1024 0.95x / 0.97x, Qwen TP4 M2048 (fused
    # quantization) 0.94x the time.
    wide = block_m == 64 and K % 128 == 0
    if _stream_fused_quant(N // 2, block_m, K):
        G = N // 2 // 64
        tile = dict(G=G, PREFETCH=(4, 3), QUANT=True, SHARE=G <= 4)  # noqa: C408
        if wide and G <= 4:
            tile.update(ACT_STEPS=2, LUT_LDS=False)
    else:
        W = 8 if block_m > 16 and (N // 2) % 128 == 0 else 4
        # From hidden 5120 on each lane loads 16 bytes of the shared activations per load, covering several 64-K
        # steps.
        steps = max(STREAM_GATEUP_ACT_BYTES_H5120 * W // block_m, 1) if K >= 5120 else 1
        tile = dict(  # noqa: C408
            G=1,
            W=W,
            PREFETCH=(STREAM_GATEUP_PREFETCH, STREAM_GATEUP_PREFETCH),
            SYNC=STREAM_GATEUP_SYNC,
            ACT_STEPS=max(steps, 2) if wide else steps,
            LUT_LDS=not wide,
        )
    return stream_a8w4.compile_a8w4_gateup_stream(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        E=experts,
        W_AUX=STREAM_GATEUP_W_AUX,
        SWIGLU_LIMIT=limit,
        **tile,
    )


@functools.cache
def _stream_prep_kernel(hidden, topk, experts, block_m, zero, device, *, shared=False):
    """Streaming prep launcher (``compile_stream_prep``). ``device`` only keys the cache; ``shared``: the last of the
    ``topk`` columns is a folded shared expert's, never sorted."""
    return stream_a8w4.compile_stream_prep(
        H=hidden, TOPK=topk, E=experts, BM=block_m, ZERO=zero, SHARED=shared
    )


def _stream_prep(
    hidden_states,
    topk_weight,
    topk_ids,
    experts,
    block_m,
    output,
    zero=True,
    routed=None,
):
    """Sorting and input quantization of the streaming path in one launch (``compile_stream_prep``); with
    ``zero`` the same launch zeroes ``output`` for the streaming Down's atomics. ``topk_ids`` / ``topk_weight`` are the
    contiguous int32 / float32 routing of the entry point; with ``routed`` = topk - 1 the last column is a folded
    shared expert's (``_SharedFold``): the prep never sorts it (its loc_ids -1) and sizes the sorted rows and blocks
    for the routed columns."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device
    shared = routed is not None and routed < topk
    n_sort, rows, blocks = stream_a8w4.stream_prep_sizes(M, topk, experts, block_m)
    if shared:
        assert routed == topk - 1
        _, rows, blocks = stream_a8w4.stream_prep_sizes(M, routed, experts, block_m)
    sorted_ids = torch.empty((rows,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((rows,), dtype=torch.float32, device=device)
    sorted_experts = torch.empty((blocks,), dtype=torch.int32, device=device)
    num_valid = torch.empty((2,), dtype=torch.int32, device=device)
    loc_ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    _launch(
        "a8w4_stream_prep",
        _stream_prep_kernel(
            H,
            topk,
            experts,
            block_m,
            zero,
            device.index,
            **({"shared": True} if shared else {}),
        ),
        topk_ids,
        topk_weight,
        hidden_states,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        loc_ids,
        xq,
        xs,
        output,
        M,
        n_sort,
        rows,
    )
    if _recording:
        # Reads the sorted columns' ids and weights and x; writes each route's sorted id and weight, the blocks'
        # experts, num_valid, loc_ids, xq / xs and, with ``zero``, the output.
        _cost(
            _Cost(
                4,
                0,
                16 * M * (routed if shared else topk)
                + 8
                + _nbytes(hidden_states, loc_ids, xq, xs, output if zero else None),
            )
        )
    return sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs


@functools.cache
def _stream_sum_kernel(topk, cols, device):
    """``a8w4_route_sum`` launcher (``compile_route_sum``; the prefill tier's sum of the routes). ``device`` only keys
    the cache."""
    return stream_a8w4.compile_route_sum(topk, cols)


def _stream_down_shared(N, K, block_m, rows_lds=False):
    """Whether the streaming Down runs four waves of 8 row blocks that share the block's activation rows
    through LDS (``down_stream_lds``; ``rows_lds``: plus the atomic Down's table of the rows' offsets and factors)."""
    # Blocks of 32 and 64 rows: four waves of 8 row blocks each share the block's activation rows
    # through LDS when they fit (with a single sub-block the sharing costs more than it saves, except for
    # a wide hidden dimension over a short K: MiMo TP8 M256 103 -> 91 us).
    wide = N >= 4096 and K <= 256
    return (
        (block_m >= 32 or wide)
        and N % 512 == 0
        and stream_a8w4.down_stream_lds(K, block_m, W=4, R=8, ATOMIC=True, ROWS_LDS=rows_lds)
        <= layout.LDS_BYTES
    )


def _stream_down_constants():
    """Module constants ``_stream_down_kernel`` reads while building its launcher (the STREAM_DOWN_* tuning and,
    through ``_stream_down_shared``, ``common_a8w4.LDS_BYTES``); part of its cache key.
    """
    return (
        STREAM_DOWN_PREFETCH,
        STREAM_DOWN_PREFETCH_SMALL,
        STREAM_DOWN_W_AUX,
        STREAM_DOWN_W_AUX_H5120,
        STREAM_DOWN_LDS_ACTS,
        layout.LDS_BYTES,
    )


@functools.cache
def _stream_down_kernel(N, K, topk, block_m, experts, atomic, device, constants):
    """Streaming Down launcher (``atomic``: add the routes into the output, what the streaming tier runs; else write
    them by sorted row). ``device`` (a compiled handle runs on one GPU) and ``constants`` (``_stream_down_constants()``)
    only key the cache."""
    # The atomic Down of blocks of 64 at K = 256 keeps the rows' offsets and factors in LDS: with the activations read
    # from LDS that lifts it from 2 to 3 waves per SIMD (178 -> 134 VGPRs; MiMo TP8 M1024 0.90x the Down's time, lab);
    # K <= 192 already reaches 3 and from K = 320 LDS bounds 2 workgroups, where the extra LDS reads only cost 1-2%.
    rows_lds = atomic and block_m == 64 and K == 256
    tile = dict(R=8, W=4) if _stream_down_shared(N, K, block_m, rows_lds) else {}  # noqa: C408
    w_aux = STREAM_DOWN_W_AUX_H5120 if N >= 5120 else STREAM_DOWN_W_AUX
    return stream_a8w4.compile_a8w4_down_stream(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        E=experts,
        ATOMIC=atomic,
        PREFETCH=STREAM_DOWN_PREFETCH if block_m == 64 else STREAM_DOWN_PREFETCH_SMALL,
        W_AUX=w_aux,
        LDS_ACTS=bool(STREAM_DOWN_LDS_ACTS)
        and (K <= 256 if block_m == 64 else block_m == 32 and not 256 < K < 512),
        ROWS_LDS=rows_lds,
        **tile,
    )


def _stream_max_rows(hidden, inter, routes=0):
    """Most mean rows per expert (M * topk / E) for the streaming kernels (``routes`` = M * topk): at most
    STREAM_BM32_ROWS from padded I 512 on, where the atomic streaming Down of blocks of 64 is
    slow."""
    if hidden >= 5120 and hidden % 512 == 0 and routes >= STREAM_H5120_MIN_ROUTES:
        rows = STREAM_MAX_ROWS_H5120
    elif hidden >= 4096 and hidden % 512 == 0:
        rows = STREAM_MAX_ROWS_WIDE
    # Measured at hidden 2560 only; smaller hidden sizes keep STREAM_MAX_ROWS.
    elif hidden >= 2560 and _stream_fused_quant(inter, 64, hidden):
        rows = STREAM_MAX_ROWS_FUSED_QUANT
    else:
        rows = STREAM_MAX_ROWS
    if inter >= 512:
        return min(rows, STREAM_BM32_ROWS)
    return rows


def _stream_block_m(tokens, topk, experts, hidden=0):
    """Sorting block of the streaming kernels from the mean rows per expert (sub-blocks of 16): 16 up to 12 rows, 32 up
    to STREAM_BM32_ROWS, else 64; the same at every hidden size (``hidden`` is not read).

    Blocks of 16 at any route count, since the streaming path sorts with its own prep: blocks of 32 below 4E/3 routes
    date from aiter's Opus moe_sorting, which there switched blocks of 16 to a separate fused MXFP4 sort, and with about
    one route per expert they compute 32-row tiles for one or two rows. Measured in one process against blocks of 32
    (summed kernel times, three routings): Qwen TP2 / TP4 (E513 topk 11) M33-62 0.76-0.84x / 0.79-0.88x the time (M48
    0.82x / 0.88x); DeepSeek TP4 M48 / M64 0.89x.
    """
    rows = tokens * topk / experts
    if rows <= 12:
        return 16
    return 32 if rows <= STREAM_BM32_ROWS else 64


def _quantize(x, rows, cols, dtype):
    """Per-row FP8 quantization of ``x`` [rows, cols] (``compile_row_quant_fp8``)."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    kernel = _row_quant_kernel(cols, dtype, x.device.index)
    _launch("a8w4_row_quant_fp8", kernel, x, q, scale, rows)
    if _recording:
        _cost(_Cost(fixed=_nbytes(x, q, scale)))
    return q, scale


def _arg(t):
    # Pointers typed by dtype as the gfx9_fp8 launchers pass them (uint8 and FP8 as Uint8).
    return ptr_arg(t, _TORCH_TO_FX.get(t.dtype))


def _launch(name, kernel, *args):
    """Launch ``kernel`` on the current stream of the first tensor argument's device; record ``name`` in ``last_path``
    and, inside ``recording_launches()``, its ``KernelLaunch`` in ``last_launches`` with zero bytes and FLOPs, which
    the caller then sets with ``_cost`` under ``if _recording:`` (so that a call outside the recording computes
    nothing for it)."""
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    _run_compiled(
        kernel,
        *(_arg(a) if isinstance(a, torch.Tensor) else a for a in args),
        current_stream_arg(device),
    )
    last_path.append(name)
    if _recording:
        _record(device, name)


def _fused_moe_stream(
    hidden_states,
    weight,
    topk_weight,
    topk_ids,
    output,
    limit,
    shared=None,
):
    """Streaming path (``stream_a8w4``): the prep sorts the routes into blocks of ``_stream_block_m`` rows,
    quantizes the input and zeroes ``output``; the streaming Gate/Up writes SiLU(gate) * up (quantized to FP8 in its
    epilogue when ``_stream_fused_quant``, else by the row kernel); the Down adds every route into ``output`` with
    BF16x2 atomics (any padded I: no routes buffer and no sum, so no prefill kernel and no entry check either).

    With a folded shared expert (``shared``, ``_SharedFold``) the routing tensors keep its column (the last; the prep
    reads them as they are and drops it), its chain starts on the prep's FP8 input and writes the whole output, which
    the prep then leaves alone and the Down adds the routes onto. The streaming tier also takes the fold that a decode
    tier cannot (``_fold_tier``: the decode tier's few tokens and any padded I the decode kernels take).
    """
    M, H = hidden_states.shape
    cols = topk_ids.shape[1]
    topk = cols - (shared is not None)
    E, I = weight.experts, weight.inter_padded
    block_m = _stream_block_m(M, topk, E, H)
    device = hidden_states.device
    sorted_ids, sorted_weights, sorted_experts, num_valid, _, xq, xs = _stream_prep(
        hidden_states,
        topk_weight,
        topk_ids,
        E,
        block_m,
        output,
        shared is None,
        routed=topk,
    )
    task_num = sorted_experts.numel()
    if shared is not None:
        shared.start(xq, xs, output)
    gateup = _stream_gateup_kernel(
        2 * I, H, topk, block_m, E, limit, device.index, _stream_gateup_constants()
    )
    routing = (sorted_ids, sorted_experts, num_valid)
    if not _stream_fused_quant(I, block_m, H):
        mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
        _launch(
            f"a8w4_gateup_stream_bm{block_m}",
            gateup,
            xq,
            weight.w1,
            weight.w1_delta,
            mid,
            *routing,
            weight.w1_scale,
            xs,
            xs,
            M,
            task_num,
        )
        if _recording:
            # The routes' sorted ids (4 bytes each) and num_valid (8 bytes).
            _cost(
                _gateup_cost(
                    weight,
                    M * topk,
                    _nbytes(xq, xs, mid) + 4 * M * topk + 8,
                    blocks=True,
                )
            )
        if shared is not None:
            shared.fork()
        mq, ms = _quantize(mid, M * topk, I, "f32")
    else:
        mq = torch.empty((M * topk, I), dtype=torch.float8_e4m3fnuz, device=device)
        ms = torch.empty((M * topk,), dtype=torch.float32, device=device)
        _launch(
            f"a8w4_gateup_quant_stream_bm{block_m}",
            gateup,
            xq,
            weight.w1,
            weight.w1_delta,
            mq,
            *routing,
            weight.w1_scale,
            xs,
            ms,
            M,
            task_num,
        )
        if _recording:
            _cost(
                _gateup_cost(
                    weight,
                    M * topk,
                    _nbytes(xq, xs, mq, ms) + 4 * M * topk + 8,
                    blocks=True,
                )
            )
        if shared is not None:
            shared.fork()
    down_args = (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        weight.w2_scale,
        ms,
        M,
        task_num,
    )
    # The prep zeroed output (or the shared expert wrote it): Down adds every route into its token's row with BF16
    # atomics.
    if shared is not None:
        shared.join()
    _launch(
        f"a8w4_down_stream_bm{block_m}",
        _stream_down_kernel(
            H, I, topk, block_m, E, True, device.index, _stream_down_constants()
        ),
        mq,
        weight.w2,
        weight.w2_delta,
        output,
        *down_args,
    )
    if _recording:
        # The routes' sorted ids and weights (8 bytes each) and num_valid; the atomic adds are one write of the
        # output.
        _cost(
            _down_cost(
                weight,
                M * topk,
                _nbytes(mq, ms, output) + 8 * M * topk + 8,
                blocks=True,
            )
        )
    return output


def _swiglu_limit(value):
    """``swiglu_limit`` of ``flydsl_moe_a8w4_ptpc`` as a float, 0.0 when off."""
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"swiglu_limit must be a real number, got {type(value).__name__}"
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"swiglu_limit must be finite and >= 0 (0 disables it), got {value!r}"
        )
    return float(value) if value > 0 else 0.0


def _check_inputs(hidden_states, weight, topk_weight, topk_ids, output):
    if not isinstance(weight, MoeA8W4PtpcWeight):
        raise TypeError(
            f"weight must come from prepare_moe_a8w4_ptpc_weight, got {type(weight).__name__}"
        )
    if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
        raise TypeError("hidden_states must be a 2-D bfloat16 tensor")
    M, H = hidden_states.shape
    if H != weight.hidden:
        raise ValueError(
            f"hidden size {H} does not match the weights ({weight.hidden})"
        )
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if (
        topk_ids.dim() != 2
        or topk_ids.shape[0] != M
        or tuple(topk_weight.shape) != tuple(topk_ids.shape)
    ):
        raise ValueError(
            f"topk_ids and topk_weight must be [M={M}, topk], got {tuple(topk_ids.shape)} and "
            f"{tuple(topk_weight.shape)}"
        )
    if (
        topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_weight.is_floating_point()
    ):
        raise TypeError(
            f"topk_ids must be int32 / int64 and topk_weight floating, got {topk_ids.dtype}, "
            f"{topk_weight.dtype}"
        )
    if output is not None:
        if output.dtype != torch.bfloat16 or tuple(output.shape) != (M, H):
            raise TypeError(
                f"output must be a bfloat16 [{M}, {H}] tensor, got {output.dtype} {tuple(output.shape)}"
            )
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
    tensors = [hidden_states, topk_weight, topk_ids, weight.w1, weight.w2] + (
        [] if output is None else [output]
    )
    if weight.has_shared:
        tensors += [weight.shared_w1, weight.shared_w2]
    devices = {t.device for t in tensors}
    if len(devices) != 1 or hidden_states.device.type != "cuda":
        raise ValueError(
            f"inputs, output and weight must be on one GPU, got {sorted(map(str, devices))}"
        )
    topk = topk_ids.shape[1]
    if weight.has_shared:
        if topk < 2:
            raise ValueError(
                f"with a shared expert topk_ids and topk_weight are [M, topk + 1] (routed experts, then the shared "
                f"expert {weight.experts}) with topk >= 1, got {tuple(topk_ids.shape)}"
            )
        topk -= 1
    if max(M * topk * weight.inter_padded * 4, M * H * 2) >= _MAX_BYTES:
        raise ValueError(
            f"M={M} is too large: M * topk * I_padded * 4 and M * H * 2 must be below 2**31 (32-bit byte offsets)"
        )


def flydsl_moe_a8w4_ptpc(
    hidden_states,
    weight,
    topk_weight,
    topk_ids,
    *,
    output=None,
    swiglu_limit=0.0,
    block_m=None,
    block_n=None,
    tile_k=None,
):
    """A8W4 MoE forward: BF16 ``hidden_states`` [M, H] (contiguous) -> BF16 [M, H].

    ``weight`` comes from ``prepare_moe_a8w4_ptpc_weight``; ``topk_weight`` [M, topk] (float32) is applied
    in Down; ``topk_ids`` [M, topk] (int32) in ``[0, E)`` (a fused shared expert is an ordinary
    expert id) or -1, a masked route (ATOM's padding rows): it reads no weights and adds nothing whatever its weight,
    so a token whose ids are all -1 gets 0. Ids are distinct within a token apart from -1 (every tier runs each slot
    of a repeated id as a route of its own and adds them all); other ids are invalid. Only ``DEBUG_CHECKS``
    (``AITER_MOE_A8W4_DEBUG=1``, a host synchronization) reads the ids to check both rules (``ValueError``). int64
    ids, other floating weights and non-contiguous routing tensors are converted first.
    The input and the f32 intermediate SiLU(gate) * up are quantized per row to FP8
    E4M3FNUZ with ``scale = amax / 240`` and ``q = RNE(x / scale)``. SiLU only, no expert mask /
    bias; ``M * topk * I_padded * 4`` and ``M * H * 2`` below ``2**31``; a call with a tile option (the prefill tier)
    needs ``block_m * I_padded <= 64 KiB`` (I_padded <= 1024 with the default block_m 64; without a tile option such
    shapes take the decode or streaming tier, which take any). Both raise ``ValueError`` before any kernel runs. ``output`` may be a caller's contiguous
    BF16 [M, H] tensor, ``hidden_states`` itself included (one off a 4-byte boundary gets the result through one more
    copy, ``output_copy`` in ``last_path``). Issues no host synchronization; compile a shape
    outside a CUDA graph capture before capturing it. Raises ``RuntimeError`` off gfx942.

    Unless a tile option is given, few tokens run the decode kernels (no sorting) when they support the shape
    (H <= 4096 or in ``DECODE_WIDE_HIDDEN``, the intermediate rows fit in LDS): below hidden 4096 while M * topk *
    I_padded / 16 stays below ``DECODE_MAX_WGS``, other hidden sizes up to ``DECODE_MAX_M`` tokens
    (``DECODE_WIDE_MAX_M`` for the ``DECODE_WIDE_HIDDEN`` sizes); other calls with at most
    ``STREAM_MAX_ROWS`` rows per expert (``STREAM_MAX_ROWS_WIDE`` for hidden >= 4096, ``STREAM_MAX_ROWS_H5120`` for
    hidden >= 5120 with many routes, ``STREAM_MAX_ROWS_FUSED_QUANT`` when the Gate/Up of blocks of 64 quantizes the
    intermediate; at most ``STREAM_BM32_ROWS`` from padded I 512 on) the streaming kernels and the
    rest the prefill kernels (TILE_K 128 when chosen for having more
    rows per expert, else 256). ``block_m`` / ``block_n`` / ``tile_k`` select the prefill kernels' tiles: ``block_m``
    (default 64) is the sorting block and M tile of both kernels (``block_m * I_padded % 4096 == 0``, e.g. 64 or
    128); ``block_n`` (default 128) / ``tile_k`` are the Gate/Up N tile (128 or 256, dividing 2 * padded I into
    whole gate/up halves) and K tile (128 or 256; default 128 for M > 2048, where its lower LDS and register use
    raise occupancy, else 256).

    ``swiglu_limit`` is the SwiGLU limit of DeepSeek-V4.1 (config ``swiglu_limit``, 10 for V4.1-Flash; the
    ``Expert`` of its inference/model.py): when > 0 every Gate/Up kernel computes
    ``SiLU(gate.clamp(max=L)) * up.clamp(-L, L)`` on the dequantized f32 gate / up values (after the weight
    and activation scales). 0 (default) turns it off and runs exactly the kernels without it; otherwise
    the limit is a compile-time constant (kernels compiled per value, names ending in ``_swiglu``). Must
    be finite and >= 0 (``ValueError``).

    With a shared expert in ``weight`` (``prepare_moe_a8w4_ptpc_weight(..., shared_w1=...)``, expert E after the E
    routed ones; ATOM's E385 / topk-7 convention for DeepSeek-V4.1-Flash), ``topk_ids`` / ``topk_weight`` are
    [M, topk + 1]: the routed experts in the first topk columns, run as above, and the shared expert's id E in the last
    column (``DEBUG_CHECKS`` checks it). Its weight is ``topk_weight[:, -1]`` where ``topk_ids[:, 0] >= 0`` and
    ``topk_ids[:, -1] == E``, else 0 (ATOM's rule: a masked first routed id drops the row's shared expert too, so ATOM's
    padding rows, all routed ids -1, get 0; a dropped shared expert reads no weight). Up to
    ``DECODE_SHARED_MIXED_MAX_M`` tokens, while tokens times routed topk times padded I stay within
    ``DECODE_SHARED_MIXED_MAX_WORK``, when the routed experts take the decode tier (no tile option, decided on the
    routed columns as the tier is), it runs as the last route of the mixed decode kernels (two launches, every column
    summed in f32 and rounded to BF16 once). Otherwise it is folded into the routed experts' last reduction
    (``_SharedFold``; the routed tier ``_fold_tier``: their own, with the decode tier's slot-pair Down at any M, or
    streaming where that Down cannot run): two FP8 GEMMs on the routed path's quantized input (in chunks of at most
    65536 rows) with SiLU(gate) * up, the limit, a per-row FP8 quantization and the routing weight between them, on a
    side stream beside the decode and streaming tiers (behind the prefill tier on the current stream), their BF16
    output summed with the routes in f32 and rounded to BF16 once (onto which the streaming tier's atomic Down adds the
    routes in BF16; an ``output`` off a 16-byte boundary gets the GEMM's output through one more copy).
    The streams can be captured in a CUDA graph (compile outside the capture first, as above). Tile options apply to
    the routed experts.

    ``aiter.fused_moe`` calls this function for the tagged ``w1`` / ``w2`` of a prepared weight (``_fused_moe_route``),
    with its ``swiglu_limit`` normalized as on aiter's other SiLU paths and without tile options: it takes only the
    automatic ``block_size_M``, so pinning the prefill tile (``block_m``) needs this function.
    """
    _require_supported_gfx()
    _check_inputs(hidden_states, weight, topk_weight, topk_ids, output)
    limit = _swiglu_limit(swiglu_limit)
    if output is None:
        output = torch.empty_like(hidden_states)
    last_path.clear()
    last_launches.clear()
    if hidden_states.shape[0] == 0:
        return output
    routed_topk = topk_ids.shape[1] - (1 if weight.has_shared else 0)
    _check_prefill_down(
        hidden_states.shape[0], weight, routed_topk, block_m, block_n, tile_k
    )
    if DEBUG_CHECKS:
        _check_routing(topk_ids, weight.experts, weight.has_shared)
    # The BF16x2 atomics of the streaming Down add pairs of output elements at 4-byte boundaries: an output off one (a
    # view starting at an odd element) gets the result from an aligned buffer through one copy.
    dst = output if output.data_ptr() % 4 == 0 else torch.empty_like(output)
    if weight.has_shared:
        _moe_shared(
            hidden_states,
            weight,
            topk_weight,
            topk_ids,
            dst,
            limit,
            (block_m, block_n, tile_k),
        )
    else:
        _moe_routed(
            hidden_states,
            weight,
            topk_weight,
            topk_ids,
            dst,
            limit,
            block_m,
            block_n,
            tile_k,
        )
    if dst is not output:
        output.copy_(dst)
        last_path.append("output_copy")
        if _recording:
            _record(output.device, "output_copy", _Cost(fixed=2 * _nbytes(output)))
    return output


def _side_stream(device):
    """The side stream of ``device`` (created on first use) on which a folded shared expert's chain runs beside the
    routed kernels (``_SharedFold``)."""
    stream = _SIDE_STREAMS.get(device.index)
    if stream is None:
        stream = _SIDE_STREAMS[device.index] = torch.cuda.Stream(device=device)
    return stream


_SIDE_STREAMS = {}


@functools.cache
def _shared_mid_kernel(inter, kp, cols, experts, limit, loc, device, lanes=0):
    """``shared_a8w4.compile_shared_mid`` launcher (``lanes`` per row, see SHARED_MID_WAVE_ROW_MAX_M; 0: the kernel's
    default for many rows). ``device`` only keys the cache."""
    return shared_a8w4.compile_shared_mid(
        I=inter, KP=kp, COLS=cols, E=experts, SWIGLU_LIMIT=limit, LOC=loc, LANES=lanes
    )


def _shared_gemm_tiles(M, N, K):
    """``(tile_m, tile_n, tile_k)`` of aiter's FlyDSL preshuffle FP8 GEMM (``flydsl_preshuffle_gemm_a8``) for the folded
    shared expert's Gate/Up (K = hidden) and Down (K = its weight's padded I) GEMMs of M rows and N columns.

    gfx942 has no tuned FP8 rows for these shapes, where ``gemm_a8w8_bpreshuffle`` runs CK's default kernels. Measured
    at the DeepSeek-V4.1 shapes (H5120, shared I 576 / 320; graph replay, PTL on), us, these tiles against CK:
    Gate/Up N1152: M4 5.5 vs 6.8, M64 7.9 vs 7.3, M128 9.2 vs 11.7, M256 13.0 vs 22.2, M512 17.7 vs 27.0, M4096 118 vs
    151 (N640 M96 / 128: tile_m 16 5.7 vs 32 8.1); Down K640: M4 3.3 vs 3.4, M64 4.2 vs 6.9, M256 8.4 vs 10.2, M512
    13.4 vs 21.8, M4096 80 vs 123 (K384 M4 2.9 vs 5.9; up to M48 now two waves of tile_n 32, as fast, see
    ``_shared_gemm``). ATOM's split Down pads its input to K and runs CKTile: M4 22 us, M512 33 (PTL off).
    """
    if K >= 2048:
        if M <= 64 or (M <= 128 and N <= 640):
            return 16, 64, 512
        if M <= 256:
            return 32, 64, 512
        if 1024 < M <= 1152 or (1152 < M <= 1408 and N % 192):
            # Just past one wave of 64 x 128 tiles (N1152 M1152: 162 tiles on 160 slots): smaller or wider tiles.
            # Whole layer under CUDA graph replay (one process, PTL on) against 64 x 128 x 256: N1152 M1152 0.984x
            # (48 x 192 x 256; M1280 / 1365 / 1536 with 64 x 192 x 256 1.000 / 1.001 / 0.999x, kept), N640 M1152 /
            # 1365 / 1536 0.980 / 0.992 / 1.001x (48 x 128 x 256).
            return (48, 192, 256) if N % 192 == 0 else (48, 128, 256)
        return (64, 128, 512) if M <= 512 else (64, 128, 256)
    if M <= 48:
        # Two waves (tile_n 32): a 16 x 128 FP8 A tile is 2 KiB, one 16-byte load per thread of 128 (see _shared_gemm).
        # Measured against (32, 64, 128) at the layers' M4-48: 0.98-1.00x the layer time.
        return 16, 32, 128
    return (32, 64, 128) if M <= 192 else (64, 128, 128)


def _shared_gemm(xq, wq, x_scale, w_scale, out):
    """``out`` = BF16 (xq @ wq.T) * x_scale * w_scale.T with aiter's FlyDSL preshuffle FP8 GEMM (the operands of
    ``gemm_a8w8_bpreshuffle``: wq in ``shuffle_weight(layout=(16, 16))``, scales [M, 1] / [N, 1]; tiles of
    ``_shared_gemm_tiles``, no split-K) on the current stream; returns the rows of each launch. The GEMM views A and C
    through a layout of at most ``gemm_kernels.PRESHUFFLE_M_MAX`` (65536) rows (``kernels/preshuffle_gemm.py``), so
    more rows take one GEMM per chunk of that many rows (row views of xq, x_scale and out, the last chunk shorter).
    Measured at the DeepSeek-V4.1 layers' M 65537-131072 (CUDA graph, PTL on, one process): the fold with chunked
    GEMMs took 0.82-0.94x the layer time of ATOM's split (aiter's per-token quantization, CK GEMMs, a final add).
    """
    from .gemm_kernels import PRESHUFFLE_M_MAX, flydsl_preshuffle_gemm_a8

    M, K = xq.shape
    if M > PRESHUFFLE_M_MAX:
        rows = []
        for r in range(0, M, PRESHUFFLE_M_MAX):
            chunk = slice(r, min(r + PRESHUFFLE_M_MAX, M))
            rows += _shared_gemm(xq[chunk], wq, x_scale[chunk], w_scale, out[chunk])
        return rows
    tile_m, tile_n, tile_k = _shared_gemm_tiles(M, wq.shape[0], K)
    # The GEMM stages each A tile in LDS with one tiled copy of 16-byte loads, one per thread (128 threads with
    # tile_n 32, else 256) and row: a tile of fewer bytes makes the copy cover rows past tile_m, which land in the
    # other LDS stage while other waves may still read it (kernels/preshuffle_gemm.py, four waves, tile_m 16 with
    # tile_k 128: wrong 8 x 16 output blocks now and then when other kernels share the CUs).
    assert tile_m * tile_k >= (128 if tile_n == 32 else 256) * 16
    flydsl_preshuffle_gemm_a8(xq, wq, x_scale, w_scale, out, tile_m, tile_n, tile_k)
    return [M]


def _shared_down_config(M, N, K):
    """``(PAIRS, OUT_AUX)`` of ``shared_a8w4.compile_shared_down`` for the shared Down GEMM of M rows (one chunk
    of ``_shared_down``), N = hidden columns and useful K (the padded shared I), or None where the preshuffle GEMM
    (``_shared_gemm``) runs: below SHARED_DOWN_MIN_M_SHORT_K / SHARED_DOWN_MIN_M rows, or a shape the kernel is not
    built for (K % 64 or K > 576: its rows of a stay in VGPRs and LDS; N % 512). PAIRS (pairs of 16-row weight
    blocks per wave, 128 * PAIRS columns per workgroup) by M:
    measured alone (eager, cold, PTL on) at the DeepSeek-V4.1 shapes, a workgroup's fixed cost (its activations, the
    first weights) wants many pairs, the grid wants enough workgroups for 80 CUs."""
    if K % 64 or K > 576 or N % 512:
        return None
    short = K <= 384
    if M < (SHARED_DOWN_MIN_M_SHORT_K if short else SHARED_DOWN_MIN_M):
        return None
    if M <= (4096 if short else 1536):
        pairs = 4
    elif M <= 16384 and (short or M > 4096):
        pairs = 10
    else:
        pairs = 20
    if N % (128 * pairs):
        pairs = 4
    return pairs, 2 if M >= SHARED_DOWN_NT_MIN_M else 0


@functools.cache
def _shared_down_kernel(n, k, kp, pairs, aux, device):
    """``shared_a8w4.compile_shared_down`` launcher. ``device`` only keys the cache."""
    return shared_a8w4.compile_shared_down(N=n, K=k, KP=kp, PAIRS=pairs, OUT_AUX=aux)


def _shared_down(mq, wq, m_scale, w_scale, out, inter):
    """The shared expert's Down GEMM ``out`` = BF16 (mq[:, :inter] @ wq[:, :inter].T) * m_scale * w_scale.T (operands
    of ``_shared_gemm``; ``inter`` = the padded shared I, the columns of mq that ``shared_a8w4.compile_shared_mid``
    fills; the rest up to the weight's K is zero) on the current stream, in chunks of at most
    ``gemm_kernels.PRESHUFFLE_M_MAX`` rows as ``_shared_gemm``; each chunk runs ``shared_a8w4.compile_shared_down``
    where ``_shared_down_config`` picks it, else the preshuffle GEMM. Returns the rows of each launch. Both give the
    same values bit for bit (the same MFMA order and epilogue rounding; the zero K tail adds +0)."""
    from .gemm_kernels import PRESHUFFLE_M_MAX

    M, KP = mq.shape
    N = wq.shape[0]
    if M > PRESHUFFLE_M_MAX:
        rows = []
        for r in range(0, M, PRESHUFFLE_M_MAX):
            chunk = slice(r, min(r + PRESHUFFLE_M_MAX, M))
            rows += _shared_down(mq[chunk], wq, m_scale[chunk], w_scale, out[chunk], inter)
        return rows
    config = _shared_down_config(M, N, inter)
    if config is None:
        return _shared_gemm(mq, wq, m_scale, w_scale, out)
    _run_compiled(
        _shared_down_kernel(N, inter, KP, *config, mq.device.index),
        _arg(mq),
        _arg(m_scale),
        _arg(wq),
        _arg(w_scale),
        _arg(out),
        M,
        current_stream_arg(mq.device),
    )
    return [M]


def _record_shared_gemm(device, name, rows, shared, fixed, flops):
    """``last_launches`` entries of a shared GEMM of ``sum(rows)`` rows launched per chunk as ``_shared_gemm`` returned
    ``rows`` (``name``, or ``<name>/chunk<i>`` for several): the first chunk counts the weight (``shared`` bytes; a
    call counts each byte once), every chunk its rows' share of the ``fixed`` bytes and of the ``flops`` of all rows.
    """
    M = sum(rows)
    for i, m in enumerate(rows):
        _record(
            device,
            name if len(rows) == 1 else f"{name}/chunk{i}",
            _Cost(0, shared if i == 0 else 0, fixed * m // M, flops * m // M),
        )


class _SharedFold:
    """The shared expert of a call that folds it into the routed experts' last reduction (``_moe_shared_fold``).

    Its chain: the Gate/Up GEMM (``_shared_gemm``) on the routed path's FP8 input and per-row scale (the prep's of the
    streaming and prefill tiers, the row kernel's of the decode tier, the plan kernel's of the deduplicated decode
    Gate/Up), then ``shared_a8w4.compile_shared_mid`` (SiLU(gate) * up with the SwiGLU limit, per-row FP8 for Down
    with the shared expert's routing weight folded into the row scale, optionally its loc_ids column) and the Down GEMM
    (``_shared_down``: ``shared_a8w4.compile_shared_down`` or the preshuffle GEMM) into ``out``: M rows of the prefill
    tier's routes buffer that ``a8w4_route_sum`` adds as the last column of loc_ids, or the output itself, onto which
    the slot-pair decode Down or the streaming Down's BF16x2 atomics add the routes. The GEMM stores 16 bytes per
    thread: an ``out`` off a 16-byte boundary gets the GEMM's output from an aligned buffer through a copy
    (``shared_fold_copy``, one more kernel on the chain's stream). A token whose shared expert is dropped (ATOM's rule,
    ``shared_a8w4``) gets +0 there. With a side stream (``side``) the side stream waits for the current stream where
    the routed path calls ``start`` (after the kernel that quantizes the input), its kernels are issued by ``fork``,
    which the routed path calls right after its next launch (``_moe_shared_fold``), and joined by ``join`` (before the
    kernel that reads ``out``); without one the chain runs on the current stream at ``start``.
    """

    def __init__(self, hidden_states, weight, topk_weight, topk_ids, limit, side):
        self.weight = weight
        self.topk_weight = topk_weight
        self.topk_ids = topk_ids
        self.limit = limit
        self.side = side
        self.current = torch.cuda.current_stream(hidden_states.device)
        self.pending = None

    def _front(self, xq, xs, loc, base):
        w = self.weight
        M = xq.shape[0]
        device = xq.device
        gate_up = torch.empty(
            (M, w.shared_w1.shape[0]), dtype=torch.bfloat16, device=device
        )
        rows = _shared_gemm(xq, w.shared_w1, xs.view(M, 1), w.shared_w1_scale, gate_up)
        last_path.append("shared_fold_gemm_gateup")
        if _recording:
            _record_shared_gemm(
                device,
                "shared_fold_gemm_gateup",
                rows,
                _nbytes(w.shared_w1, w.shared_w1_scale),
                _nbytes(xq, xs, gate_up),
                2 * M * gate_up.shape[1] * xq.shape[1],
            )
        kp = w.shared_w2.shape[1]
        mq = torch.empty((M, kp), dtype=torch.float8_e4m3fnuz, device=device)
        ms = torch.empty((M, 1), dtype=torch.float32, device=device)
        _launch(
            "shared_fold_mid",
            _shared_mid_kernel(
                w.inter_padded,
                kp,
                self.topk_ids.shape[1],
                w.experts,
                self.limit,
                loc is not None,
                device.index,
                lanes=(
                    64
                    if M <= SHARED_MID_WAVE_ROW_MAX_M
                    else shared_a8w4.mid_row_lanes(w.inter_padded)
                ),
            ),
            gate_up,
            self.topk_ids,
            self.topk_weight,
            mq,
            ms,
            ms if loc is None else loc,
            M,
            base,
        )
        if _recording:
            # Per token: the first and the last id, the last weight (12 bytes) and, with ``loc``, its last column.
            _cost(
                _Cost(
                    fixed=_nbytes(gate_up, mq, ms)
                    + 12 * M
                    + (0 if loc is None else 4 * M)
                )
            )
        return mq, ms

    def _down(self, mq, ms, out):
        w = self.weight
        dst = out if out.data_ptr() % 16 == 0 else torch.empty_like(out)
        rows = _shared_down(
            mq, w.shared_w2, ms, w.shared_w2_scale, dst, w.shared_w1.shape[0] // 2
        )
        last_path.append("shared_fold_gemm_down")
        if _recording:
            # Useful K: the shared expert's padded I, not the weight's K padded to 128.
            _record_shared_gemm(
                mq.device,
                "shared_fold_gemm_down",
                rows,
                _nbytes(w.shared_w2, w.shared_w2_scale),
                _nbytes(mq, ms, dst),
                2 * mq.shape[0] * dst.shape[1] * (w.shared_w1.shape[0] // 2),
            )
        if dst is not out:
            out.copy_(dst)
            last_path.append("shared_fold_copy")
            if _recording:
                _record(out.device, "shared_fold_copy", _Cost(fixed=2 * _nbytes(out)))

    def start(self, xq, xs, out, loc=None, base=0):
        """Run the chain, or with a side stream make the side stream wait for the current stream's kernels so far and
        keep the chain for ``fork``: ``xq`` / ``xs`` FP8 [M, H] / f32 [M] (the routed path's quantized input), ``out``
        BF16 [M, H] (contiguous), ``loc`` int32 [M, topk + 1] whose last column gets ``base + t`` (or -1).
        """
        if self.side is None:
            self._down(*self._front(xq, xs, loc, base), out)
            return
        ready = torch.cuda.Event()
        ready.record(self.current)
        self.pending = (ready, xq, xs, out, loc, base)

    def fork(self):
        """Issue the side stream's chain kept by ``start`` (no-op without one or once issued)."""
        if self.pending is None:
            return
        ready, xq, xs, out, loc, base = self.pending
        self.pending = None
        self.side.wait_event(ready)
        last_path.append("shared_fork")
        with torch.cuda.stream(self.side):
            self._down(*self._front(xq, xs, loc, base), out)
        # Tensors of the current stream that the side stream uses.
        for t in (xq, xs, self.topk_weight, self.topk_ids, out, loc):
            if t is not None:
                t.record_stream(self.side)

    def join(self):
        """The current stream waits for the chain (issued first if ``fork`` has not)."""
        self.fork()
        if self.side is not None:
            self.current.wait_stream(self.side)
            last_path.append("shared_join")


def _decode_fold(weight, topk):
    """Whether the decode tier can fold a shared expert (``topk`` routed columns): its routed experts then take the
    slot-pair Down at any M (``compile_a8w4_down_decode_pq``, whose SHARED variant adds the routes onto the shared
    expert's output), which addresses all experts' w2 with 32-bit offsets and needs the shape within its limits.
    """
    H, I = weight.hidden, weight.inter_padded
    return _slot_pair_offsets_fit(weight) and _decode_fits(H, I, topk, True)


def _fold_tier(M, weight, topk, tiles):
    """The routed tier of a call that folds its shared expert (``_SharedFold``; ``topk`` routed columns, M > 0): the
    routed experts' own (``_routed_tier``: the routed columns decide), except streaming in place of a decode tier that
    cannot fold (``_decode_fold``: weights or a shape past the slot-pair Down's limits; the streaming kernels take any M
    and padded I). Every tier folds at any M: the decode tier's slot-pair Down and the streaming atomic Down add the
    routes onto the shared expert's output, ``a8w4_route_sum`` adds its M rows of the prefill tier's routes buffer as
    one more column.
    """
    H, I = weight.hidden, weight.inter_padded
    tier = _routed_tier(M, H, I, topk, weight.experts, *tiles)
    if tier == "decode" and not _decode_fold(weight, topk):
        return "stream"
    return tier


def _moe_shared_fold(
    hidden_states, weight, topk_weight, topk_ids, output, limit, tiles
):
    """``flydsl_moe_a8w4_ptpc`` with a shared expert folded into the routed experts' last reduction (``_SharedFold``):
    the routed experts on ``_fold_tier`` with the full routing tensors, the chain on a side stream beside the decode
    and streaming tiers (an output in place included: the chain writes the output only after the routed kernels that
    read x) and on the current stream behind the prefill tier.
    """
    # Under graph replay the side stream's kernels run beside the routed ones only where these leave the CUs room, and a
    # fork / join costs about 8 us: measured in one process at the DeepSeek-V4.1 layers (PTL on) against the chain on
    # the current stream, decode and streaming M4-1024 0.97-1.03x the time, streaming M1025-1536 0.96-1.01x, prefill
    # M2048 / 4096 1.00-1.02x. The side stream's kernels are issued right after the routed path's next kernel (the
    # Gate/Up), with the same dependencies (they wait for the kernel that quantizes the input only), so that the routed
    # Gate/Up does not start behind the fork under graph replay (measured in one process at the layers, PTL on, against
    # issuing them first: decode M4-32 0.89-0.98x the time, streaming M64 / 256 0.97-1.00x).
    M = hidden_states.shape[0]
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = _routing_input(topk_ids, torch.int32, "topk_ids")
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = _routing_input(topk_weight, torch.float32, "topk_weight")
    tier = _fold_tier(M, weight, topk_ids.shape[1] - 1, tiles)
    side = _side_stream(hidden_states.device) if tier != "prefill" else None
    fold = _SharedFold(hidden_states, weight, topk_weight, topk_ids, limit, side)
    return _moe_routed(
        hidden_states, weight, topk_weight, topk_ids, output, limit, *tiles, fold, tier
    )


def _mixed_decode(M, weight, topk, tiles):
    """Whether a call with a shared expert (``topk`` routed columns, M > 0) runs the mixed decode kernels
    (``_fused_moe_decode_mixed``): up to DECODE_SHARED_MIXED_MAX_M tokens while M * topk * padded I stays within
    DECODE_SHARED_MIXED_MAX_WORK (so that the mixed Gate/Up, which quantizes its token row in every workgroup, stays
    far below the DECODE_MAX_WGS routed workgroups where the per-route Gate/Up reads pre-quantized
    rows), the mixed Down within its limits and the routed experts on the decode tier (no tile option; the routed
    columns decide it, as they decide the tier)."""
    H, I = weight.hidden, weight.inter_padded
    return (
        M <= DECODE_SHARED_MIXED_MAX_M
        and M * topk * I <= DECODE_SHARED_MIXED_MAX_WORK
        and decode_a8w4.down_decode_mixed_fits(I, topk + 1)
        and _routed_tier(M, H, I, topk, weight.experts, *tiles) == "decode"
    )


def _moe_shared(hidden_states, weight, topk_weight, topk_ids, output, limit, tiles):
    """``flydsl_moe_a8w4_ptpc`` with a shared expert: the mixed decode kernels when ``_mixed_decode`` holds, else the
    shared expert folded into the routed experts' last reduction (``_moe_shared_fold``).
    """
    if _mixed_decode(hidden_states.shape[0], weight, topk_ids.shape[1] - 1, tiles):
        return _fused_moe_decode_mixed(
            hidden_states, weight, topk_weight, topk_ids, output, limit
        )
    return _moe_shared_fold(
        hidden_states, weight, topk_weight, topk_ids, output, limit, tiles
    )


def _many_rows(M, H, I, topk, E):
    """Whether the mean rows per expert (M * topk / E) exceed the streaming kernels' limit (``_stream_max_rows``)."""
    return M * topk > _stream_max_rows(H, I, M * topk) * E


def _decode_range(M, H, I, topk):
    """Whether the decode tier takes M tokens of this shape (where its kernels hold it): below hidden 4096 while the
    per-route Gate/Up grid, M * topk * padded I / 16 workgroups, stays below DECODE_MAX_WGS, the DECODE_WIDE_HIDDEN
    sizes up to DECODE_WIDE_MAX_M tokens, other hidden sizes up to DECODE_MAX_M."""
    if H in DECODE_WIDE_HIDDEN:
        return M <= DECODE_WIDE_MAX_M
    if H >= 4096:
        return M <= DECODE_MAX_M
    return M * topk * (I // 16) < DECODE_MAX_WGS


def _routed_tier(M, H, I, topk, E, block_m, block_n, tile_k):
    """Tier of the routed experts of a call (M > 0): "decode", "stream" or "prefill". Chosen before the prefill tile
    defaults are filled in: any tile option selects prefill. Without one, within the decode tier's range
    (``_decode_range``) the decode kernels where they hold the shape (``_decode_fits``), else the streaming kernels
    if the decode Gate/Up takes the hidden size, else prefill; beyond the range the streaming kernels while the rows
    per expert stay within their limit (``_many_rows``) or where the prefill Down cannot stage the padded I (block_m
    64: I_pad <= 1024), prefill beyond. The streaming kernels take any M and padded I.
    """
    if block_m is None and block_n is None and tile_k is None:
        if _decode_range(M, H, I, topk):
            if _decode_fits(H, I, topk, False) or _decode_fits(H, I, topk, True):
                return "decode"
            if _decode_gateup_fits(H):
                return "stream"
        elif not _many_rows(M, H, I, topk, E) or 64 * I > layout.LDS_BYTES:
            return "stream"
    return "prefill"


def _check_prefill_down(M, weight, topk, block_m, block_n, tile_k):
    """ValueError before any launch when the call's prefill Down cannot hold its block_m x padded-I FP8 activation
    tile in LDS (``gemm2_a8w4``: block_m * I_padded <= 64 KiB, so I_padded <= 1024 with the default block_m 64); only
    the prefill tier builds a prefill Down, and without a tile option the dispatch streams the shapes it cannot hold
    (``_routed_tier``), so only calls with a tile option raise; the decode and streaming Downs take any padded I their
    dispatch accepts."""
    I = weight.inter_padded
    bm = 64 if block_m is None else block_m
    # The call's prefill Down, if any, fits: no tier needed.
    if bm * I <= layout.LDS_BYTES:
        return
    tiles = (block_m, block_n, tile_k)
    if not weight.has_shared:
        tier = _routed_tier(M, weight.hidden, I, topk, weight.experts, *tiles)
    elif _mixed_decode(M, weight, topk, tiles):
        return
    else:
        tier = _fold_tier(M, weight, topk, tiles)
    if tier == "prefill":
        raise ValueError(
            f"padded I {I} is too large for the prefill Down this call takes (M={M}, block_m {bm}): it stages a "
            f"block_m x I_padded FP8 tile in {layout.LDS_BYTES} bytes of LDS, so I_padded <= "
            f"{layout.LDS_BYTES // bm}; the decode and streaming tiers take larger I"
        )


def _check_routing(topk_ids, experts, shared):
    """``DEBUG_CHECKS`` (reads the ids on the host): routed ids in [0, E) or -1 (a masked route), no expert id repeated
    within a token's routed columns (-1 aside) and, with a shared expert, its id E in the last column of every row.
    """
    routed = topk_ids[:, :-1] if shared else topk_ids
    if shared and not bool((topk_ids[:, -1] == experts).all()):
        raise ValueError(
            f"with a shared expert the last column of topk_ids holds its id {experts} (E) on every row"
        )
    if not bool((((routed >= 0) & (routed < experts)) | (routed == -1)).all()):
        raise ValueError(
            f"routed expert ids must lie in [0, {experts}) or be -1 (a masked route)"
        )
    ids = routed.sort(dim=1).values
    if bool(((ids[:, 1:] == ids[:, :-1]) & (ids[:, 1:] >= 0)).any()):
        raise ValueError("a token's routed expert ids must be distinct (-1 aside)")


def _moe_routed(
    hidden_states,
    weight,
    topk_weight,
    topk_ids,
    output,
    limit,
    block_m,
    block_n,
    tile_k,
    shared=None,
    tier=None,
):
    """The routed experts of ``flydsl_moe_a8w4_ptpc`` (M > 0) into ``output``: the tier (``tier``, else
    ``_routed_tier``'s) and its kernels. With a folded shared expert (``shared``, ``_SharedFold``) the routing tensors
    keep its column, the last: the routed experts take the others, and each tier starts the shared expert's chain on
    its FP8 input and joins it before its last reduction."""
    M, H = hidden_states.shape
    cols = topk_ids.shape[1]
    topk = cols - (shared is not None)
    E, I = weight.experts, weight.inter_padded
    device = hidden_states.device
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = _routing_input(topk_ids, torch.int32, "topk_ids")
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = _routing_input(topk_weight, torch.float32, "topk_weight")
    if tier is None:
        tier = _routed_tier(M, H, I, topk, E, block_m, block_n, tile_k)
    if tier == "decode":
        return _fused_moe_decode(
            hidden_states, weight, topk_weight, topk_ids, output, limit, shared
        )
    if tier == "stream":
        return _fused_moe_stream(
            hidden_states, weight, topk_weight, topk_ids, output, limit, shared
        )
    if block_m is None and block_n is None and tile_k is None:
        tile_k = 128 if _many_rows(M, H, I, topk, E) else 256
    block_m = 64 if block_m is None else block_m
    block_n = 128 if block_n is None else block_n
    tile_k = (128 if M > 2048 else 256) if tile_k is None else tile_k

    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs = (
        _stream_prep(
            hidden_states,
            topk_weight,
            topk_ids,
            E,
            block_m,
            output,
            zero=False,
            routed=topk,
        )
    )
    grid = sorted_experts.numel()
    routing = (sorted_ids, sorted_weights, sorted_experts, num_valid)
    rows = grid * block_m
    routes = torch.empty(
        (rows + (0 if shared is None else M), H), dtype=torch.bfloat16, device=device
    )
    if shared is not None:
        shared.start(xq, xs, routes[rows:], loc_ids, rows)

    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    xcd_chunk, wgs_per_cu = _gateup_grid(H, E, M * topk, (block_m, block_n, tile_k))
    gateup = _gateup_kernel(
        2 * I,
        H,
        topk,
        block_m,
        block_n,
        tile_k,
        E,
        limit,
        xcd_chunk,
        device.index,
        _trace_constants(),
        **({"wgs_per_cu": wgs_per_cu} if wgs_per_cu else {}),
    )
    _launch(
        f"a8w4_gateup_prefill_bm{block_m}_bn{block_n}_tk{tile_k}",
        gateup,
        xq,
        weight.w1,
        weight.w1_delta,
        mid,
        *routing,
        weight.w1_scale,
        xs,
        M,
        grid,
    )
    if _recording:
        # The routes' sorted ids (4 bytes each) and num_valid (8 bytes).
        _cost(
            _gateup_cost(
                weight, M * topk, _nbytes(xq, xs, mid) + 4 * M * topk + 8, blocks=True
            )
        )
    if shared is not None:
        shared.fork()
    mq, ms = _quantize(mid, M * topk, I, "f32")
    down = _prefill_down(
        M,
        topk,
        E,
        H,
        I,
        block_m,
        device.index,
        loc_topk=None if shared is None else cols,
    )
    _launch(
        f"a8w4_down_prefill_bm{block_m}",
        down,
        mq,
        weight.w2,
        weight.w2_delta,
        routes,
        *routing,
        weight.w2_scale,
        ms,
        loc_ids,
        M,
        grid,
    )
    if _recording:
        _cost(
            _down_cost(
                weight,
                M * topk,
                _nbytes(mq, ms) + 12 * M * topk + 8 + 2 * M * topk * H,
                blocks=True,
            )
        )
    if shared is not None:
        shared.join()
    _launch(
        "a8w4_route_sum",
        _stream_sum_kernel(cols, H, device.index),
        loc_ids,
        routes,
        output,
        M,
        routes.shape[0],
    )
    if _recording:
        # loc_ids and the rows it points to (the routes and the folded shared expert's), the output.
        _cost(_Cost(fixed=_nbytes(loc_ids, output) + 2 * M * cols * H))
    return output


# fused_moe parameters that _fused_moe_route reads; every other one must keep its fused_moe default, except these
# hints that do not change the result: moe_sorting_dispatch_policy only schedules aiter's own moe_sorting (this op
# sorts with its own prep kernel), splitk only moves the order of the K sums (fused_moe_
# lets a provider ignore it), herd_routed_scaling_factor is unused while herd_topk is 0 (which is required).
_ROUTE_ARGS = frozenset(
    (
        "hidden_states",
        "w1",
        "w2",
        "topk_weight",
        "topk_ids",
        "activation",
        "quant_type",
        "quant_type_a",
        "swiglu_limit",
        "block_size_M",
        "dtype",
        "output",
        "quant_dtype_a",
        "quant_dtype_a2",
        "shared_expert_id",
    )
)
_ROUTE_IGNORED = frozenset(
    ("moe_sorting_dispatch_policy", "splitk", "herd_routed_scaling_factor")
)


@functools.cache
def _fused_moe_defaults():
    from aiter.fused_moe import fused_moe

    return {
        name: p.default for name, p in inspect.signature(fused_moe).parameters.items()
    }


def _is_default(value, default):
    return value is default or (
        default is not None and not isinstance(value, torch.Tensor) and value == default
    )


def _fused_moe_route(args):
    """``aiter.fused_moe`` on the tagged ``w1`` / ``w2`` of ``prepare_moe_a8w4_ptpc_weight`` (``args``: every
    ``fused_moe`` argument by name): one ``flydsl_moe_a8w4_ptpc`` call on the weight rebuilt from the tags, so the
    kernels and results of a direct call. Raises ``RuntimeError`` off gfx942. ``fused_moe`` calls it before its custom
    op (which drops tensor attributes), its tuned-config lookup and ``moe_sorting``. The parameters:

    - ``hidden_states``, ``topk_weight``, ``topk_ids``: passed on (BF16 [M, H]; a non-contiguous ``hidden_states`` is
      copied contiguous first, as ``fused_moe`` does for layouts its kernels cannot read);
    - ``w1``, ``w2``: the ``w1`` and ``w2`` of one ``prepare_moe_a8w4_ptpc_weight`` result (``ValueError``);
    - ``activation``, ``quant_type``, ``quant_type_a``: ``ActivationType.Silu``, ``QuantType.per_1x32`` (MXFP4
      weights) and ``QuantType.per_Token`` (FP8 activations, one scale per token) only (``ValueError``);
    - ``swiglu_limit``: ``normalize_swiglu_limit`` as on aiter's other SiLU paths (None, 0 and inf turn it off; NaN
      and negative values raise ``ValueError``), then passed on;
    - ``block_size_M``: only None, 0 or -1 (this op chooses its path and tiles); another value raises
      ``NotImplementedError`` (``flydsl_moe_a8w4_ptpc(..., block_m=...)`` pins the prefill tile);
    - ``output``: ``fused_moe``'s checks (a contiguous [M, H] tensor of the output dtype on the input's device, not
      overlapping ``hidden_states``; ``RuntimeError``), then passed on and returned;
    - ``dtype``: None or ``torch.bfloat16``; ``quant_dtype_a``, ``quant_dtype_a2``: None or ``dtypes.fp8`` (the
      FP8 E4M3FNUZ of this op); other values raise ``NotImplementedError``;
    - ``shared_expert_id``: -1 (the default, not given) or, for a weight prepared with a shared expert, its id E (the
      number of routed experts, aiter's FHMoE convention ``E - 1`` of its E + 1 rows); anything else raises
      ``ValueError``. A shared expert in the weight runs whether or not it is given: ``topk_ids`` / ``topk_weight``
      are then [M, topk + 1] with E in the last column (``flydsl_moe_a8w4_ptpc``);
    - ignored, hints that do not change the result: ``moe_sorting_dispatch_policy``, ``splitk``,
      ``herd_routed_scaling_factor`` (see ``_ROUTE_IGNORED``);
    - only at their ``fused_moe`` defaults (``NotImplementedError`` naming them): ``expert_mask``,
      ``doweight_stage1``, ``w1_scale`` / ``w2_scale`` (the scales travel with the tagged weights), ``a1_scale`` /
      ``a2_scale`` (this op quantizes its activations), ``num_local_tokens``, ``hidden_pad`` / ``intermediate_pad``
      (the weight carries its sizes), ``bias1`` / ``bias2``, ``beta`` / ``linear_beta``, ``gate_mode``, the raw
      shared-expert tensors ``shared_w1``, ``shared_w2``, ``shared_w1_scale``, ``shared_w2_scale`` (a shared expert
      goes into ``prepare_moe_a8w4_ptpc_weight(..., shared_w1=...)`` and travels with the tags; a fused MXFP4 shared
      expert is an ordinary expert id), ``stage2_scatter``, ``clamp_shared``, ``herd_topk``, ``deterministic`` (this
      op picks its kernels by token count and some paths add with atomics), ``w1_global_scale`` /
      ``w2_global_scale``, and any parameter ``fused_moe`` gains later.
    """
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import (
        _validate_output_buffer_metadata,
        _validate_output_buffer_no_overlap,
    )
    from aiter.ops.moe_limits import normalize_swiglu_limit

    _require_supported_gfx()
    weight = _tagged_weight(args["w1"], args["w2"])
    activation, quant_type, quant_type_a = (
        args["activation"],
        args["quant_type"],
        args["quant_type_a"],
    )
    if (
        ActivationType(activation) != ActivationType.Silu
        or QuantType(quant_type) != QuantType.per_1x32
        or quant_type_a is None
        or QuantType(quant_type_a) != QuantType.per_Token
    ):
        raise ValueError(
            "the w1 / w2 of prepare_moe_a8w4_ptpc_weight run with activation=ActivationType.Silu, "
            "quant_type=QuantType.per_1x32 and quant_type_a=QuantType.per_Token, got activation="
            f"{activation}, quant_type={quant_type}, quant_type_a={quant_type_a}"
        )
    defaults = _fused_moe_defaults()
    unsupported = [
        name
        for name, value in args.items()
        if name not in _ROUTE_ARGS
        and name not in _ROUTE_IGNORED
        and not _is_default(value, defaults[name])
    ]
    if args["dtype"] not in (None, torch.bfloat16):
        unsupported.append(f"dtype={args['dtype']}")
    for name in ("quant_dtype_a", "quant_dtype_a2"):
        if args[name] not in (None, dtypes.fp8):
            unsupported.append(f"{name}={args[name]}")
    hints = []
    if {"w1_scale", "w2_scale"} & set(unsupported):
        hints.append("the scales travel with the tagged w1 / w2")
    if set(_SHARED_NAMES) & set(unsupported):
        hints.append(
            "a shared expert goes into prepare_moe_a8w4_ptpc_weight(..., shared_w1=, shared_w1_scale=, shared_w2=, "
            "shared_w2_scale=) first and is routed as expert E in the last topk column"
        )
    if args["block_size_M"] not in (None, 0, -1):
        unsupported.append(f"block_size_M={args['block_size_M']}")
        hints.append(
            "this op chooses its tiles; flydsl_moe_a8w4_ptpc(..., block_m=...) pins the prefill tile"
        )
    if unsupported:
        raise NotImplementedError(
            "fused_moe on the w1 / w2 of prepare_moe_a8w4_ptpc_weight does not support "
            + ", ".join(unsupported)
            + "".join(f" ({hint})" for hint in hints)
        )
    shared_id = args["shared_expert_id"]
    if not _is_default(shared_id, defaults["shared_expert_id"]) and (
        not weight.has_shared
        or isinstance(shared_id, bool)
        or not isinstance(shared_id, numbers.Integral)
        or shared_id != weight.experts
    ):
        raise ValueError(
            f"shared_expert_id is -1 (not given) or the shared expert's id {weight.experts} (E, the last topk "
            f"column), got {shared_id!r}"
            + (
                ""
                if weight.has_shared
                else "; the weight has no shared expert (prepare_moe_a8w4_ptpc_weight(..., "
                "shared_w1=...))"
            )
        )
    limit = normalize_swiglu_limit(args["swiglu_limit"])
    hidden_states, topk_ids, output = (
        args["hidden_states"],
        args["topk_ids"],
        args["output"],
    )
    _validate_output_buffer_metadata(
        output,
        (topk_ids.shape[0], weight.hidden),
        hidden_states.dtype if args["dtype"] is None else args["dtype"],
        hidden_states.device,
    )
    _validate_output_buffer_no_overlap(output, hidden_states)
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    return flydsl_moe_a8w4_ptpc(
        hidden_states,
        weight,
        args["topk_weight"],
        topk_ids,
        output=output,
        swiglu_limit=0.0 if limit is None else limit,
    )
