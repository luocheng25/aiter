# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 A8W4 MoE (FlyDSL): FP8 per-token activations x MXFP4 weights, BF16 in and out.

``prepare_moe_a8w4_ptpc_weight`` converts standard MXFP4 MoE weights once (at load time) into the private layout
of ``kernels/moe_gemm_2stage/gfx9_fp8/common_a8w4.py``. ``flydsl_moe_a8w4_ptpc`` runs the whole MoE. Token counts
up to ``DECODE_MAX_M`` skip the sorting and use the decode kernels of ``gfx9_fp8/decode_a8w4.py`` when they support
the shape (Gate/Up per route, Down per token summing its routes in the workgroup and writing BF16; activations
quantized inside the kernels or by a row kernel). From ``DECODE_DEDUP_MIN_H`` hidden on (padded I a multiple of
128) and for the ``DECODE_WIDE_HIDDEN`` sizes the decode path reads each expert's weights once per group of its
routes instead (``gfx9_fp8/decode_dedup_a8w4.py``; the Down then adds the routes into the zeroed output with BF16x2
atomics), and runs up to ``DECODE_DEDUP_MAX_M`` tokens while at most 1.5 routes fall on an expert. Larger token
counts with at most ``STREAM_MAX_ROWS`` rows per expert (M * topk / E; ``STREAM_MAX_ROWS_WIDE`` for hidden >= 4096,
``STREAM_MAX_ROWS_H5120`` for hidden >= 5120 with many routes, ``STREAM_MAX_ROWS_FUSED_QUANT`` when the Gate/Up of
blocks of 64 quantizes the intermediate) use the weight-streaming kernels of ``gfx9_fp8/stream_a8w4.py``: one prep
launch sorts the routes into small blocks, quantizes the input and (for the atomic Down) zeroes the output, the
Gate/Up writes SiLU(gate) * up (with larger blocks quantized to FP8 in the same kernel), and the Down either adds
each route into the output with BF16x2 atomics or writes the routes for ``a8w4_route_sum`` (blocks of 64 without
the atomic Down take the prefill Down from hidden 4096 on). The rest run the prefill kernels: the sorting (the
streaming prep up to ``PREFILL_PREP_MAX_ROUTES`` routes from hidden 2560 on, else Opus ``moe_sorting`` and a
per-row FP8 quantization of the input that sets ``loc_ids`` to -1), the A8W4 Gate/Up kernel (SiLU(gate) * up ->
f32, optionally with the DeepSeek-V4.1 SwiGLU limit), per-row FP8 quantization of the intermediate, the A8W4 Down
kernel (routing weight applied; it also records each route's sorted row in ``loc_ids``) and ``sorted_sum``. The
calls issue no host synchronization. A routing id of -1 masks its route (ATOM's padding rows): no tier reads its
weights or adds it (the decode kernels load through zero-byte views and leave it out of the deduplication plan, the
sortings drop it).

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
holds E. Up to ``DECODE_SHARED_MIXED_MAX_M`` tokens, when the routed experts take the per-route decode kernels (and,
where the fold below is possible too, up to ``DECODE_SHARED_MIXED_MAX_WORK``), the shared expert runs as one more route
of the same two decode launches (``decode_a8w4``'s mixed Gate/Up and Down: its FP8 weights feed the MFMA directly,
``_fused_moe_decode_mixed``). Otherwise the routed experts run as above on the other
columns (the tier decided on them alone). Where the routed path allows it (``_shared_fold_tier``: the slot-pair decode
Down, streaming, prefill from ``SHARED_FOLD_PREFILL_MIN_M``) the shared expert is folded into the routed path's last
reduction (``_SharedFold``): two FlyDSL FP8 GEMMs (aiter's preshuffle GEMM, ``_shared_gemm``) on the routed path's FP8
input with ``gfx9_fp8/shared_a8w4.py`` between them, on a side stream up to ``SHARED_FOLD_DUAL_STREAM_MAX_M`` tokens,
write its weighted output where ``a8w4_route_sum`` / ``sorted_sum`` add it as one more route, or into the output that
the routed Down then adds onto. Elsewhere it runs ATOM's split (aiter's per-token FP8 quantization,
``gemm_a8w8_bpreshuffle``, ``silu_and_mul`` and ``gemm_a8w8_bpreshuffle`` again, ``_shared_split``), on a side stream up
to ``SHARED_DUAL_STREAM_MAX_M`` tokens, and is added with its routing weight.
"""

import functools
import inspect
import math
import numbers
import os
from dataclasses import dataclass, field

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
from .kernels.moe_gemm_2stage.gfx9_fp8.moe_reduce import sorted_sum
from .kernels.moe_gemm_2stage.gfx9_fp8.quant_a8w4 import compile_row_quant_fp8
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

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
# Token counts up to this use the decode kernels (no sorting, one workgroup per route / token).
DECODE_MAX_M = 32
# The decode path quantizes activations in separate row kernels instead of inside every workgroup:
# Down's intermediate rows from DECODE_PREQUANT_MIN_M tokens on, Gate/Up's token rows from
# DECODE_GATEUP_PREQUANT_MIN_WGS Gate/Up workgroups (M * topk * I / 16) on.
DECODE_PREQUANT_MIN_M = 4
DECODE_GATEUP_PREQUANT_MIN_WGS = 2560
DECODE_DOWN_PQ_WAVES = 3
# Hidden sizes above 4096 that take the decode kernels (the Gate/Up decode then stages the FP8 token row in two
# rounds of 16-byte pieces); measured: DeepSeek-V4.1 TP4 (H5120). Other hidden sizes above 4096 take prefill.
DECODE_WIDE_HIDDEN = (5120,)
# From DECODE_WIDE_DEDUP_MIN_M tokens on, those hidden sizes deduplicate the decode Gate/Up by expert
# (``decode_dedup_a8w4``, DECODE_WIDE_DEDUP_WAVES waves) and keep the per-token slot-pair Down (padded I = 576 is no
# multiple of 128 for the dedup Down). Measured at H5120 E384 topk 6 against the per-route kernels: M4 / 8 / 16 / 32
# 1.03x / 0.985x / 0.967x / 0.95x the time (8 waves 7-11% slower).
DECODE_WIDE_DEDUP_MIN_M = 8
DECODE_WIDE_DEDUP_WAVES = 4
# Above DECODE_MAX_M tokens those hidden sizes still take the decode path up to DECODE_WIDE_MAX_M tokens (at most
# DECODE_DEDUP_MAX_M, the plan's limit) while at most 1.5 routes fall on an expert. Measured in one process at H5120
# E384 topk 6 against the streaming kernels with the hidden-5120 rules below (sorting blocks of 16, ACT_STEPS,
# non-temporal Down weights): M33 / 40 / 48 / 56 / 64 1.04x / 1.03x / 1.07x / 1.11x / 1.15x the time, so none (32).
DECODE_WIDE_MAX_M = 32
# From DECODE_DEDUP_MIN_M tokens on, for hidden sizes from DECODE_DEDUP_MIN_H on and padded I a multiple of 128,
# the decode Down reads each expert's weights once per DECODE_DEDUP_GROUP of its routes and adds into the
# zeroed output with BF16x2 atomics (``decode_dedup_a8w4``; DECODE_DEDUP_DOWN_BLOCKS 16-channel blocks per
# wave). While tokens times padded I stay below DECODE_DEDUP_GATEUP_MIN_MI the per-route Gate/Up writes the plan
# and zeroes the output and Down quantizes its intermediate rows itself (two kernels); from it on a kernel that
# quantizes x writes the plan, Gate/Up is deduplicated too and a row kernel quantizes the intermediate (the
# repeated quantization in Down and the per-route Gate/Up cost more than that kernel with more work; MiMo TP8
# I=256 from M12, TP4 I=512 from M6).
DECODE_DEDUP_MIN_M = 2
DECODE_DEDUP_MIN_H = 4096
# Routes per group: 2 up to DECODE_DEDUP_PAIR_MAX_M tokens (fewer staged rows; an expert rarely has three
# routes), DECODE_DEDUP_GROUP beyond.
DECODE_DEDUP_GROUP = 3
DECODE_DEDUP_PAIR_MAX_M = 16
DECODE_DEDUP_DOWN_BLOCKS = 2
DECODE_DEDUP_GATEUP_MIN_MI = 3072
# The deduplicated Gate/Up splits K over 4 waves per workgroup, over 8 up to DECODE_DEDUP_GATEUP_WIDE_MAX_M
# tokens (MiMo TP8 M12 / M16 3% faster; with more tokens 3-5% slower).
DECODE_DEDUP_GATEUP_WIDE_MAX_M = 16
# From this topk * padded I on the deduplicated Gate/Up issues only its first decode_a8w4.GATEUP_PREFETCH_PAIRS block
# pairs' loads before staging the token rows (each further pair as the MFMA loop consumes one), below it all of them.
# Measured in one process against all pairs first: DeepSeek-V4.1 layers under CUDA graph replay (the folded shared
# expert's chain beside them), main TP4 (6 * 576) M 8 / 16 0.96x / 0.95x the time, but MTP TP8 (3 * 320) M 8 / 16
# 1.03x / 1.05x, main TP8 (6 * 320) M 8 / 16 1.03x / 0.98x and MTP TP4 (3 * 576) 1.10x / 0.97x; routed alone (eager)
# main TP4 M 8 / 16 / 32 0.98x, MiMo (8 * 256, 8 * 512) within 1%.
DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I = 3072
# Above DECODE_MAX_M tokens the dedup decode path still runs, up to DECODE_DEDUP_MAX_M tokens, while the mean
# routes per expert (M * topk / E) stay at or below 1.5 (MiMo E256 topk 8: M <= 48; streaming beyond).
DECODE_DEDUP_MAX_M = 64
# Above DECODE_MAX_M tokens the weight-streaming kernels run while the mean rows per expert
# (M * topk / E) stay at or below this, the prefill kernels (TILE_K 128) beyond: measured crossover
# between 22.0 and 24.7 rows per expert for TP2 and TP4 alike (E513, topk 11: M = 1024 / 1152); with blocks of 64
# Qwen TP2 (I=320) streams 27% slower at 33 and 44 rows.
STREAM_MAX_ROWS = 22
# The same with hidden >= 4096, with the atomic streaming Down (MiMo TP8) and without (MiMo TP4): measured in one
# process against the prefill kernels, 48 rows (M1536) 0.94x / 1.00x the time, 64 rows (M2048) 1.04x / 1.03x.
STREAM_MAX_ROWS_WIDE = 48
# The same for hidden 2560 to 4095 (measured at 2560) when the Gate/Up of blocks of 64 quantizes the intermediate
# itself (``_stream_fused_quant``, Qwen TP4 I=192): 33 / 44 / 55 rows (M1536 / 2048 / 2560) 0.96x / 0.92x / 0.98x.
STREAM_MAX_ROWS_FUSED_QUANT = 56
# Most mean rows per expert sorted in blocks of 32 (more: blocks of 64).
STREAM_BM32_ROWS = 28
# Whether the streaming Down adds its routes into the output with BF16x2 atomics (``_stream_atomic``): 1 while padded
# I < 512, 0 never, 2 always (diagnostics).
STREAM_ATOMIC = 1
# 16-row weight blocks the streaming Down keeps in flight.
STREAM_DOWN_PREFETCH = 4
# The shared streaming Down of 32-row blocks reads the activations from LDS at every 64-K step instead of holding
# them in VGPRs when K % 256 == 0 (no loader scratch slot: LDS and VGPRs then allow 4 workgroups per CU, MiMo TP8
# M512 94 -> 90 us; with K = 320 or blocks of 64 LDS still bounds the occupancy and the extra reads cost 6-8%).
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
# Sorting blocks of 64 rows without the atomic Down, hidden >= 4096: the prefill Down (gemm2_a8w4) writes the routes
# instead of the streaming Down (MiMo TP4 M1024: Down 303 -> 289 us; with the atomic Down it is slower, MiMo TP8
# 166 -> 187 + 29).
STREAM_PREFILL_DOWN = 1
# The prefill Down splits the hidden dimension over 2 or 4 workgroups per sorted block while the
# estimated blocks (E * ceil(M * topk / E / block_m), uniform routing) times the split stay below this:
# few blocks (M ~2048) otherwise leave the last round of workgroups mostly idle (MiMo M2048 Down -10%).
# Padded I below 256 (short Down steps) gains nothing from it and stays unsplit.
DOWN_MIN_WGS = 1000
# The prefill kernels take the streaming path's prep (one launch: sorting, input quantization, loc_ids) instead of
# Opus moe_sorting and the row kernel while M * topk stays at or below this, for hidden 2560 and above with
# H % 512 == 0 (measured at 2560 and 4096; smaller hidden sizes keep Opus). Each sorting workgroup histograms every
# route, so the prep grows with the square of the routes: measured in one process, M2048 MiMo TP8 (16384 routes)
# 0.988x, Qwen TP2 (22528) 0.979x, MiMo TP4 1.00x the time; MiMo TP8 M4096 / 8192 / 16384 1.01x / 1.03x / 1.06x,
# Qwen TP2 M8192 1.02x.
PREFILL_PREP_MAX_ROUTES = 24576
# DeepSeek-V4.1-Flash TP4 (hidden 5120, E384, topk 6, I576, swiglu_limit 10), measured in one process against the rules
# above; they apply from hidden 5120 on:
# - the streaming kernels run up to STREAM_MAX_ROWS_H5120 mean rows per expert once M * topk reaches
#   STREAM_H5120_MIN_ROUTES (prefill with the two-pass Down of gemm2_a8w4 vs streaming: 12 / 16 / 20 / 24 rows,
#   M768 / 1024 / 1280 / 1536, 1.07x / 0.995x / 0.97x / 0.92x the time); fewer routes (not measured) keep
#   STREAM_MAX_ROWS_WIDE;
# - the streaming path sorts into blocks of 16 at any route count up to 12 rows per expert (``_stream_block_m``);
# - the prefill kernels take the streaming prep up to PREFILL_PREP_MAX_ROUTES_H5120 routes (Opus and the row kernel vs
#   the prep at 7680 / 12288 / 24576 routes, M1280 / 2048 / 4096: 1.003x / 0.998x / 0.988x the time);
# - the streaming Down's weight loads are non-temporal (M48-1536 2-4% faster).
STREAM_MAX_ROWS_H5120 = 16
STREAM_H5120_MIN_ROUTES = 4096
PREFILL_PREP_MAX_ROUTES_H5120 = 12288
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
# The prefill Down stages a large activation tile in two passes (gemm2_a8w4 TWO_PASS) only for grids of more
# workgroups (estimated blocks times the split) than one per CU.
DOWN_TWO_PASS_MIN_WGS = 80
# Token counts up to this run the shared expert (``_shared_split``) on a side stream of the device, concurrently with
# the routed experts (ATOM's dual-stream MoE, ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD = 1024); more tokens run it on the
# current stream before the routed experts.
SHARED_DUAL_STREAM_MAX_M = 1024
# The same for the folded shared expert (``_SharedFold``; ATOM's threshold too). Under graph replay the side stream's
# kernels run beside the routed ones only where these leave the CUs room, and a fork / join costs about 8 us: measured
# in one process at the DeepSeek-V4.1 layers (PTL on) against the chain on the current stream, M4-1024 0.97-1.03x the
# time, prefill M2048 / 4096 1.00-1.02x.
SHARED_FOLD_DUAL_STREAM_MAX_M = 1024
# With the side stream: 1 issues its chain right after the routed path's next kernel (the Gate/Up) instead of before
# it (0), with the same dependencies (it still waits for the kernel that quantizes the input only), so that the
# routed Gate/Up does not start behind the fork under graph replay (measured in one process at the DeepSeek-V4.1
# layers, PTL on: decode M4-32 0.89-0.98x the time, streaming M64 / 256 0.97-1.00x; MTP TP8 M4, where the chain is
# longer than the routed Gate/Up, 1.06x).
SHARED_FOLD_FORK_AFTER = 1
# Decode tier without the deduplicated Gate/Up (few tokens) with a folded shared expert: 0 quantizes x once on the
# current stream for the pre-quantized routed Gate/Up and the shared chain; 1 leaves the routed Gate/Up quantizing
# its own rows and the shared chain quantizes x first (measured at the DeepSeek-V4.1 layers' M4: 1.01-1.03x the time).
SHARED_FOLD_DECODE_SIDE_QUANT = 0
# Whether a call with a shared expert folds it into the routed experts' last reduction where the routed path allows it
# (``_shared_fold_tier``; 0: ATOM's split at every token count, for diagnostics): its dense chain takes the routed
# path's FP8 input and writes its output, routing weight applied, where that reduction adds it (rows of the routes
# buffer that a8w4_route_sum / sorted_sum sum as one more column, or the output itself, onto which the slot-pair decode
# Down or the streaming Down's atomics add the routes), so the split's input quantization, final add and copies of
# the routed columns go away (the decode kernels read the routing tensors with the shared expert's column as they are).
SHARED_FOLD = 1
# The prefill tier folds the shared expert from this many tokens on (where the DeepSeek-V4.1 layers' prefill tier
# starts); fewer tokens in the prefill tier (few experts, tile options) keep ATOM's split.
SHARED_FOLD_PREFILL_MIN_M = 2048
# Token counts up to this run the shared expert as one more route of the per-route decode kernels (``decode_a8w4``'s
# mixed Gate/Up and Down: its FP8 weights feed the MFMA directly; two launches, no side stream, no final add) when the
# routed experts take the per-route decode path (``_mixed_decode``); more tokens run the fold or the split. Measured at
# M 4 on the DeepSeek-V4.1 main TP4 / TP8 and MTP TP4 / TP8 layers (CUDA graph, PTL on, same-GPU ABBA):
# 0.96 / 0.77 / 0.64 / 0.48x the time of ATOM's split, although its Down (one wave per routing column of a token) takes
# 1.2-1.7x the routed experts' slot-pair Down there; from M 8 on, H 5120 deduplicates the routed Gate/Up
# (DECODE_WIDE_DEDUP_MIN_M).
DECODE_SHARED_MIXED_MAX_M = 4
# Where the fold is possible as well (``_shared_fold_tier``: the slot-pair decode Down, from DECODE_PREQUANT_MIN_M tokens
# on), the mixed decode yields to it once M * routed topk * padded I exceeds this: the mixed kernels' per-token waves
# grow with the routed columns and I, the fold's shared chain runs beside the routed kernels. Measured at M 4 (CUDA
# graph, PTL on, one process, the decode kernels of this file's rules): DeepSeek-V4.1 main TP4 (4 * 6 * 576 = 13824)
# folded 0.94x the mixed time; main TP8 (7680), MTP TP4 (6912) and MTP TP8 (3840) mixed 0.82x / 0.76x / 0.55x the
# fold's.
DECODE_SHARED_MIXED_MAX_WORK = 10240
# Checks that read routing tensors on the host (a synchronization, so off by default; AITER_MOE_A8W4_DEBUG=1): routed
# ids in [0, E) or -1 (masked), no id repeated within a token apart from -1 and, with a shared expert, its id E in the
# last column of every row (``_check_routing``).
DEBUG_CHECKS = os.environ.get("AITER_MOE_A8W4_DEBUG", "0") == "1"

# Kernels launched by the most recent ``flydsl_moe_a8w4_ptpc`` call, in order (host-side record; tests and the
# benchmark read it). With a shared expert it also holds steps named ``shared*``: the split's (and, around them, the fork
# and join of the side streams), or ``shared_mixed`` after the two kernels (``*_mixed``) of the mixed decode, which runs
# the shared expert inside them.
last_path = []


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
    N, K, topk, block_m, block_n, tile_k, experts, limit, xcd_chunk, device, constants
):
    """Prefill Gate/Up launcher. ``device`` (a compiled handle runs on one GPU) and ``constants``
    (``_trace_constants()``) only key the cache."""
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
def _row_quant_kernel(cols, dtype, fill, device):
    """Row quantization launcher (``compile_row_quant_fp8``). ``device`` only keys the cache."""
    return compile_row_quant_fp8(cols, dtype, fill)


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


def _decode_waves(topk, paired):
    """Waves of the Down decode kernel: few tokens (latency bound) one per routing slot; pre-quantized
    rows (``paired``) DECODE_DOWN_PQ_WAVES waves with slot pairs in rounds."""
    return (
        min((topk + 1) // 2, DECODE_DOWN_PQ_WAVES)
        if paired
        else (topk if topk <= 16 else 4)
    )


def _decode_fits(H, I, topk, paired):
    """Whether the decode kernels support this shape (Gate/Up and the single-slot or slot-pair Down) and it was
    measured to take them (hidden <= 4096 or in DECODE_WIDE_HIDDEN)."""
    return (
        (H <= 4096 or H in DECODE_WIDE_HIDDEN)
        and decode_a8w4.gateup_decode_fits(H)
        and decode_a8w4.down_decode_fits(I, topk, _decode_waves(topk, paired), paired)
    )


@functools.cache
def _decode_kernels(
    N1, H, I, topk, gateup_pq, down_pq, limit, waves, device, *, shared=False
):
    """Decode Gate/Up and Down launchers. ``waves`` (``_decode_waves(topk, down_pq)``, it reads
    DECODE_DOWN_PQ_WAVES) and ``device`` (a compiled handle runs on one GPU) key the cache; with ``shared`` both
    read routing tensors of topk + 1 columns (the folded shared expert's last, unread) and the slot-pair Down adds the
    shared expert's output it finds in ``out`` (``_SharedFold``).
    """
    extra = {"ID_COLS": topk + 1} if shared else {}
    gateup = decode_a8w4.compile_a8w4_gateup_decode(
        N=N1, K=H, TOPK=topk, PREQUANT=gateup_pq, SWIGLU_LIMIT=limit, **extra
    )
    if down_pq:
        # More tokens: 16-byte loads of slot pairs.
        return gateup, decode_a8w4.compile_a8w4_down_decode_pq(
            N=H, K=I, TOPK=topk, WAVES=waves, SHARED=shared, **extra
        )
    assert not shared
    return gateup, decode_a8w4.compile_a8w4_down_decode(
        N=H, K=I, TOPK=topk, WAVES=waves
    )


@functools.cache
def _decode_mixed_kernels(N1, H, I, KP, topk, experts, limit, device):
    """Mixed decode Gate/Up and Down launchers (a shared FP8 expert, id ``experts``, in the last of ``topk`` columns;
    its Down weight's K padded to ``KP``). ``device`` only keys the cache."""
    gateup = decode_a8w4.compile_a8w4_gateup_decode_mixed(
        N=N1, K=H, TOPK=topk, E=experts, SWIGLU_LIMIT=limit
    )
    down = decode_a8w4.compile_a8w4_down_decode_mixed(
        N=H, K=I, KP=KP, TOPK=topk, E=experts
    )
    return gateup, down


@functools.cache
def _gateup_plan_kernel(N1, H, topk, gateup_pq, group, limit, max_m, device):
    """Per-route decode Gate/Up that also writes the dedup plan and zeroes the output (``PLAN``). ``max_m``
    (DECODE_MAX_M, the plan's token limit) and ``device`` key the cache."""
    return decode_a8w4.compile_a8w4_gateup_decode(
        N=N1,
        K=H,
        TOPK=topk,
        PREQUANT=gateup_pq,
        PLAN=(group, max_m, H),
        SWIGLU_LIMIT=limit,
    )


@functools.cache
def _gateup_dd_kernel(N1, H, topk, group, waves, limit, device, *, shared=False):
    """Deduplicated decode Gate/Up launcher. ``device`` only keys the cache; ``shared`` as in ``_decode_kernels``.
    From DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I (topk times padded I) on it issues only its first
    decode_a8w4.GATEUP_PREFETCH_PAIRS block pairs' loads up front."""
    prefetch = (
        decode_a8w4.GATEUP_PREFETCH_PAIRS
        if topk * (N1 // 2) >= DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I
        else None
    )
    return decode_dedup_a8w4.compile_a8w4_gateup_decode_dd(
        N=N1,
        K=H,
        TOPK=topk,
        GROUP=group,
        WAVES=waves,
        SWIGLU_LIMIT=limit,
        PREFETCH=prefetch,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


@functools.cache
def _row_quant_plan(cols, topk, group, max_m, zero, device, *, shared=False):
    """Row quantization of x that also writes the dedup plan (and with ``zero`` zeroes the output). ``device`` only
    keys the cache; ``shared`` as in ``_decode_kernels``."""
    return decode_dedup_a8w4.compile_a8w4_row_quant_plan(
        COLS=cols,
        TOPK=topk,
        MAX_M=max_m,
        GROUP=group,
        ZERO_COLS=cols if zero else 0,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


@functools.cache
def _down_pq_kernel(H, I, topk, waves, device, *, shared=False):
    """Slot-pair decode Down of the deduplicated Gate/Up of the DECODE_WIDE_HIDDEN sizes. ``waves``
    (``_decode_waves(topk, True)``, it reads DECODE_DOWN_PQ_WAVES) and ``device`` key the cache; ``shared`` as in
    ``_decode_kernels``.
    """
    return decode_a8w4.compile_a8w4_down_decode_pq(
        N=H,
        K=I,
        TOPK=topk,
        WAVES=waves,
        SHARED=shared,
        **({"ID_COLS": topk + 1} if shared else {}),
    )


@functools.cache
def _down_dd_kernel(H, I, topk, group, cb, quant, device):
    """Deduplicated decode Down launcher. ``device`` only keys the cache."""
    return decode_dedup_a8w4.compile_a8w4_down_decode_dd(
        N=H, K=I, TOPK=topk, GROUP=group, CB=cb, QUANT=quant
    )


def _down_dd_blocks(H, I):
    """16-channel blocks per wave of the dedup Down, or 0 if it does not support the shape."""
    cb = DECODE_DEDUP_DOWN_BLOCKS
    return cb if I % 128 == 0 and H % (64 * cb) == 0 else 0


def _dedup_beyond_decode(M, H, I, topk, experts):
    """Whether more than DECODE_MAX_M tokens still take the dedup decode path (see DECODE_DEDUP_MAX_M and, for the
    DECODE_WIDE_HIDDEN sizes, DECODE_WIDE_MAX_M)."""
    if H in DECODE_WIDE_HIDDEN:
        return (
            M <= min(DECODE_WIDE_MAX_M, DECODE_DEDUP_MAX_M)
            and 2 * M * topk <= 3 * experts
        )
    return (
        M <= DECODE_DEDUP_MAX_M
        and 2 * M * topk <= 3 * experts
        and H >= DECODE_DEDUP_MIN_H
        and _down_dd_blocks(H, I) > 0
    )


def _overlaps(a, b):
    a0, b0 = a.data_ptr(), b.data_ptr()
    return (
        a0 < b0 + b.numel() * b.element_size()
        and b0 < a0 + a.numel() * a.element_size()
    )


def _fused_moe_decode_dd(
    hidden_states, weight, topk_weight, topk_ids, output, down_cb, limit
):
    """Decode with expert deduplication (``decode_dedup_a8w4``): Down reads each expert's weights once per group
    of its routes and adds into the zeroed output. While M * I stays below DECODE_DEDUP_GATEUP_MIN_MI the
    per-route Gate/Up also writes the plan and zeroes the output (two kernels; not when the output overlaps the
    input, which that Gate/Up still reads); from it on one kernel quantizes x, writes the plan and zeroes the
    output, and Gate/Up is deduplicated as well."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1]
    I = weight.inter_padded
    group = 2 if M <= DECODE_DEDUP_PAIR_MAX_M else DECODE_DEDUP_GROUP
    device = hidden_states.device
    plan = torch.empty((M * topk, 4), dtype=torch.int32, device=device)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    if (
        M * I < DECODE_DEDUP_GATEUP_MIN_MI
        and I % 256 == 0
        and not _overlaps(output, hidden_states)
    ):
        gateup_pq = M * topk * (I // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS
        if gateup_pq:
            xq, xs = _quantize_decode(hidden_states, M, H, True)
        else:
            xq, xs = hidden_states, topk_weight
        _launch(
            "a8w4_gateup_decode_plan" + ("_pq" if gateup_pq else ""),
            _gateup_plan_kernel(
                2 * I, H, topk, gateup_pq, group, limit, DECODE_MAX_M, device.index
            ),
            xq,
            xs,
            topk_ids,
            weight.w1,
            weight.w1_delta,
            weight.w1_scale,
            mid,
            plan,
            output,
            M,
        )
        quant = True
    else:
        xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
        xs = torch.empty((M,), dtype=torch.float32, device=device)
        _launch(
            "a8w4_row_quant_plan_decode",
            _row_quant_plan(
                H,
                topk,
                group,
                DECODE_MAX_M if M <= DECODE_MAX_M else DECODE_DEDUP_MAX_M,
                True,
                device.index,
            ),
            hidden_states,
            xq,
            xs,
            topk_ids,
            plan,
            output,
            M,
        )
        waves = 8 if M <= DECODE_DEDUP_GATEUP_WIDE_MAX_M and H % 1024 == 0 else 4
        _launch(
            "a8w4_gateup_decode_dd",
            _gateup_dd_kernel(2 * I, H, topk, group, waves, limit, device.index),
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
        quant = False
    mq, ms = (mid, topk_weight) if quant else _quantize_decode(mid, M * topk, I, False)
    _launch(
        "a8w4_down_decode_dd",
        _down_dd_kernel(H, I, topk, group, down_cb, quant, device.index),
        mq,
        ms,
        topk_ids,
        topk_weight,
        plan,
        weight.w2,
        weight.w2_delta,
        weight.w2_scale,
        output,
        M,
    )
    return output


def _fused_moe_decode_wide_dd(
    hidden_states, weight, topk_weight, topk_ids, output, limit, shared=None
):
    """Decode of the DECODE_WIDE_HIDDEN sizes from DECODE_WIDE_DEDUP_MIN_M tokens: one kernel quantizes x and writes
    the plan, Gate/Up is deduplicated by expert (``decode_dedup_a8w4``), the row kernel quantizes the intermediate and
    the slot-pair Down sums each token's routes (padded I not a multiple of 128: no dedup Down). A folded shared
    expert (``shared``, ``_SharedFold``) starts on the plan kernel's FP8 input, writes ``output`` and the Down adds the
    routes onto it; the kernels read the routing tensors with its column (the last) as they are.
    """
    M, H = hidden_states.shape
    topk = topk_ids.shape[1] - (shared is not None)
    I = weight.inter_padded
    on = {} if shared is None else {"shared": True}
    group = 2 if M <= DECODE_DEDUP_PAIR_MAX_M else DECODE_DEDUP_GROUP
    device = hidden_states.device
    plan = torch.empty((M * topk, 4), dtype=torch.int32, device=device)
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    xq = torch.empty((M, H), dtype=torch.float8_e4m3fnuz, device=device)
    xs = torch.empty((M,), dtype=torch.float32, device=device)
    _launch(
        "a8w4_row_quant_plan_decode",
        _row_quant_plan(
            H,
            topk,
            group,
            DECODE_MAX_M if M <= DECODE_MAX_M else DECODE_DEDUP_MAX_M,
            False,
            device.index,
            **on,
        ),
        hidden_states,
        xq,
        xs,
        topk_ids,
        plan,
        output,
        M,
    )
    if shared is not None:
        shared.start(xq, xs, output)
    _launch(
        "a8w4_gateup_decode_dd",
        _gateup_dd_kernel(
            2 * I, H, topk, group, DECODE_WIDE_DEDUP_WAVES, limit, device.index, **on
        ),
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
    if shared is not None:
        shared.fork()
    mq, ms = _quantize_decode(mid, M * topk, I, False)
    if shared is not None:
        shared.join()
    _launch(
        "a8w4_down_decode_pq" + ("" if shared is None else "_shared"),
        _down_pq_kernel(H, I, topk, _decode_waves(topk, True), device.index, **on),
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
    return output


def _slot_pair_offsets_fit(weight):
    """Whether the slot-pair Down (``compile_a8w4_down_decode_pq``) can address all experts' w2 with 32-bit byte
    offsets: its range check ends one expert's bytes below 2**32 (where a masked route's offsets fall).
    """
    w2_bytes = weight.w2.numel() * weight.w2.element_size()
    return w2_bytes + layout.weight_bytes(weight.hidden, weight.inter_padded) <= 2**32


def _fused_moe_decode(
    hidden_states, weight, topk_weight, topk_ids, output, limit, shared=None
):
    """The decode path: Gate/Up per route into the f32 intermediate, Down per token summing its routes into
    ``output`` (or the expert-deduplicated variants); ``topk_ids`` / ``topk_weight`` are contiguous int32 /
    float32. A folded shared expert (``shared``, ``_SharedFold``; only where ``_decode_slot_pair_down`` holds) keeps
    its column in them, the last, which the decode kernels skip; its chain takes the FP8 input of the routed Gate/Up
    (from the plan kernel of the deduplicated Gate/Up, else from the row kernel, the per-route Gate/Up then
    pre-quantized at any M) or, with SHARED_FOLD_DECODE_SIDE_QUANT, quantizes x itself first, writes ``output`` and the
    slot-pair Down adds the routes onto it."""
    M, H = hidden_states.shape
    topk = topk_ids.shape[1] - (shared is not None)
    I = weight.inter_padded
    down_cb = _down_dd_blocks(H, I)
    if M >= DECODE_DEDUP_MIN_M and H >= DECODE_DEDUP_MIN_H and down_cb:
        assert shared is None
        return _fused_moe_decode_dd(
            hidden_states, weight, topk_weight, topk_ids, output, down_cb, limit
        )
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
    # Few tokens: each workgroup quantizes its own activation rows; more tokens: quantize once (a folded shared
    # expert's chain takes those rows, unless it quantizes x itself).
    gateup_pq = M * topk * (I // 16) >= DECODE_GATEUP_PREQUANT_MIN_WGS or (
        shared is not None and not SHARED_FOLD_DECODE_SIDE_QUANT
    )
    # The pre-quantized Down addresses all experts' weights with 32-bit byte offsets; larger weights or
    # shapes the slot-pair kernel does not support take the single-slot Down.
    down_pq = (
        M >= DECODE_PREQUANT_MIN_M
        and _slot_pair_offsets_fit(weight)
        and _decode_fits(H, I, topk, True)
    )
    assert shared is None or down_pq
    gateup, down = _decode_kernels(
        2 * I,
        H,
        I,
        topk,
        gateup_pq,
        down_pq,
        limit,
        _decode_waves(topk, down_pq),
        hidden_states.device.index,
        **({} if shared is None else {"shared": True}),
    )
    mid = torch.empty((M, topk, I), dtype=torch.float32, device=hidden_states.device)
    if gateup_pq:
        xq, xs = _quantize_decode(hidden_states, M, H, True)
    else:
        xq, xs = hidden_states, topk_weight
    if shared is not None:
        # The per-route Gate/Up without pre-quantized rows reads x: an output in place waits for it before the
        # shared expert writes there.
        shared.start(
            xq if gateup_pq else None,
            xs if gateup_pq else None,
            output,
            hold=not gateup_pq and _overlaps(output, hidden_states),
        )
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
        mid,
        mid,
        M,
    )
    if shared is not None:
        shared.release()
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
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
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
        "a8w4_gateup_decode_mixed",
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
    _launch(
        "a8w4_down_decode_mixed",
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
    # The shared expert ran inside the two kernels: a marker as the split's steps are.
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
    if _stream_fused_quant(N // 2, block_m, K):
        G = N // 2 // 64
        tile = dict(G=G, PREFETCH=(4, 3), QUANT=True, SHARE=G <= 4)  # noqa: C408
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
            ACT_STEPS=steps,
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
def _stream_prep_kernel(
    hidden, topk, experts, block_m, zero, qw, device, *, shared=False
):
    """Streaming prep launcher (``compile_stream_prep``). ``device`` only keys the cache; ``shared``: the last of the
    ``topk`` columns is a folded shared expert's, never sorted."""
    return stream_a8w4.compile_stream_prep(
        H=hidden, TOPK=topk, E=experts, BM=block_m, ZERO=zero, QW=qw, SHARED=shared
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
    ``zero`` the same launch zeroes ``output`` for the atomic Down. ``topk_ids`` / ``topk_weight`` are the
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
    # Four waves per input row while the row's 512-column chunks split evenly or rows are few (measured: H 4096
    # M512 13.2 vs 14.5 us; H 2560 M1024 one wave per row 15.4 vs 17.7 us).
    qw = 4 if H % 2048 == 0 or M <= 256 else 1
    _launch(
        "a8w4_stream_prep",
        _stream_prep_kernel(
            H,
            topk,
            experts,
            block_m,
            zero,
            qw,
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
    return sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs


@functools.cache
def _stream_sum_kernel(topk, cols, device):
    """``a8w4_route_sum`` launcher (``compile_route_sum``). ``device`` only keys the cache."""
    return stream_a8w4.compile_route_sum(topk, cols)


def _stream_down_shared(N, K, block_m):
    """Whether the streaming Down runs four waves of 8 row blocks that share the block's activation rows
    through LDS (``down_stream_lds``)."""
    # Blocks of 32 and 64 rows: four waves of 8 row blocks each share the block's activation rows
    # through LDS when they fit (with a single sub-block the sharing costs more than it saves, except for
    # a wide hidden dimension over a short K: MiMo TP8 M256 103 -> 91 us).
    wide = N >= 4096 and K <= 256
    return (
        (block_m >= 32 or wide)
        and N % 512 == 0
        and stream_a8w4.down_stream_lds(K, block_m, W=4) <= layout.LDS_BYTES
    )


def _stream_down_constants():
    """Module constants ``_stream_down_kernel`` reads while building its launcher (the STREAM_DOWN_* tuning and,
    through ``_stream_down_shared``, ``common_a8w4.LDS_BYTES``); part of its cache key.
    """
    return (
        STREAM_DOWN_PREFETCH,
        STREAM_DOWN_W_AUX,
        STREAM_DOWN_W_AUX_H5120,
        STREAM_DOWN_LDS_ACTS,
        layout.LDS_BYTES,
    )


@functools.cache
def _stream_down_kernel(N, K, topk, block_m, experts, atomic, device, constants):
    """Streaming Down launcher. ``device`` (a compiled handle runs on one GPU) and ``constants``
    (``_stream_down_constants()``) only key the cache."""
    tile = dict(R=8, W=4) if _stream_down_shared(N, K, block_m) else {}  # noqa: C408
    w_aux = STREAM_DOWN_W_AUX_H5120 if N >= 5120 else STREAM_DOWN_W_AUX
    return stream_a8w4.compile_a8w4_down_stream(
        N=N,
        K=K,
        TOPK=topk,
        BLOCK_M=block_m,
        E=experts,
        ATOMIC=atomic,
        PREFETCH=STREAM_DOWN_PREFETCH,
        W_AUX=w_aux,
        LDS_ACTS=bool(STREAM_DOWN_LDS_ACTS) and block_m == 32 and K % 256 == 0,
        **tile,
    )


def _stream_atomic(hidden, inter, block_m):
    """Whether the streaming Down adds its routes into the output with BF16 atomics (no routes buffer and
    route_sum): measured faster up to I = 320 (Qwen TP2 M512 355 -> 336 us, MiMo TP8 M512 272 -> 263 us), slower
    at I = 512 (MiMo TP4 M128 / M512 +4 us: the atomic epilogue's extra registers on top of K = 512 activations).
    ``STREAM_ATOMIC``: 0 never, 2 always (diagnostics)."""
    return STREAM_ATOMIC == 2 or (STREAM_ATOMIC == 1 and inter < 512)


def _stream_prefill_down(hidden, inter, block_m):
    """Whether the streaming path writes its routes with the prefill Down (gemm2_a8w4) instead of the streaming Down
    (see ``STREAM_PREFILL_DOWN``)."""
    return (
        bool(STREAM_PREFILL_DOWN)
        and block_m == 64
        and hidden >= 4096
        and not _stream_atomic(hidden, inter, block_m)
    )


def _stream_max_rows(hidden, inter, routes=0):
    """Most mean rows per expert (M * topk / E) for the streaming kernels (``routes`` = M * topk)."""
    if hidden >= 5120 and hidden % 512 == 0 and routes >= STREAM_H5120_MIN_ROUTES:
        return STREAM_MAX_ROWS_H5120
    if hidden >= 4096 and hidden % 512 == 0:
        return STREAM_MAX_ROWS_WIDE
    # Measured at hidden 2560 only; smaller hidden sizes keep STREAM_MAX_ROWS.
    if hidden >= 2560 and _stream_fused_quant(inter, 64, hidden):
        return STREAM_MAX_ROWS_FUSED_QUANT
    return STREAM_MAX_ROWS


def _stream_block_m(tokens, topk, experts, hidden=0):
    """Sorting block of the streaming kernels from the mean rows per expert (sub-blocks of 16).

    Block 16 only once 3 * routes >= 4 * experts (chosen when the streaming path still sorted with aiter's Opus
    moe_sorting, which below that switches block 16 to a separate fused MXFP4 sort); from hidden 5120 on block 16 at
    any route count (DeepSeek TP4 M48 / M64: 0.89x the time of blocks of 32).
    """
    rows = tokens * topk / experts
    if rows <= 12:
        own_sort = hidden >= 5120 and hidden % 512 == 0
        return 16 if 3 * tokens * topk >= 4 * experts or own_sort else 32
    return 32 if rows <= STREAM_BM32_ROWS else 64


def _quantize(x, rows, cols, dtype, fill=None):
    """Per-row FP8 quantization of ``x`` [rows, cols]; the same kernel sets int32 ``fill`` [rows, k] to -1."""
    q = torch.empty((rows, cols), dtype=torch.float8_e4m3fnuz, device=x.device)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    kernel = _row_quant_kernel(
        cols, dtype, 0 if fill is None else fill.shape[1], x.device.index
    )
    _launch(
        "a8w4_row_quant_fp8", kernel, x, q, scale, scale if fill is None else fill, rows
    )
    return q, scale


def _arg(t):
    # Pointers typed by dtype as the gfx9_fp8 launchers pass them (uint8 and FP8 as Uint8).
    return ptr_arg(t, _TORCH_TO_FX.get(t.dtype))


def _launch(name, kernel, *args):
    """Launch ``kernel`` on the current stream of the first tensor argument's device; record ``name``."""
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    _run_compiled(
        kernel,
        *(_arg(a) if isinstance(a, torch.Tensor) else a for a in args),
        current_stream_arg(device),
    )
    last_path.append(name)


def _fused_moe_stream(
    hidden_states, weight, topk_weight, topk_ids, output, limit, shared=None
):
    """Streaming path (``stream_a8w4``): the prep sorts the routes into blocks of ``_stream_block_m`` rows,
    quantizes the input and, for the atomic Down, zeroes ``output``; the streaming Gate/Up writes SiLU(gate) * up
    (quantized to FP8 in its epilogue when ``_stream_fused_quant``, else by the row kernel); the Down either adds
    every route into ``output`` with BF16x2 atomics (``_stream_atomic``) or writes the routes by sorted row for
    ``a8w4_route_sum`` (blocks of 64 from hidden 4096 on: the prefill Down, ``STREAM_PREFILL_DOWN``).

    With a folded shared expert (``shared``, ``_SharedFold``) the routing tensors keep its column (the last; the prep
    reads them as they are and drops it), its chain starts on the prep's FP8 input and writes either M more rows of
    the routes buffer, which route_sum adds as one more column of loc_ids, or (atomic Down) the whole output, which the
    prep then leaves alone and the Down adds the routes onto.
    """
    M, H = hidden_states.shape
    cols = topk_ids.shape[1]
    topk = cols - (shared is not None)
    E, I = weight.experts, weight.inter_padded
    block_m = _stream_block_m(M, topk, E, H)
    device = hidden_states.device
    atomic = _stream_atomic(H, I, block_m)
    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, xq, xs = (
        _stream_prep(
            hidden_states,
            topk_weight,
            topk_ids,
            E,
            block_m,
            output,
            atomic and shared is None,
            routed=topk,
        )
    )
    task_num = sorted_experts.numel()
    rows = task_num * block_m
    routes = None
    if shared is not None:
        if atomic:
            shared.start(xq, xs, output)
        else:
            routes = torch.empty((rows + M, H), dtype=torch.bfloat16, device=device)
            shared.start(xq, xs, routes[rows:], loc_ids, rows)
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
    if atomic:
        # The prep zeroed output (or the shared expert wrote it): Down adds every route into its token's row with
        # BF16 atomics.
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
        return output
    if routes is None:
        routes = torch.empty((rows, H), dtype=torch.bfloat16, device=device)
    if _stream_prefill_down(H, I, block_m):
        # The prefill Down also stores each route's sorted row into loc_ids, the same values the prep wrote.
        _launch(
            f"a8w4_down_prefill_bm{block_m}",
            _prefill_down(
                M,
                topk,
                E,
                H,
                I,
                block_m,
                device.index,
                loc_topk=None if shared is None else cols,
            ),
            mq,
            weight.w2,
            weight.w2_delta,
            routes,
            *down_args[:-2],
            loc_ids,
            M,
            task_num,
        )
    else:
        _launch(
            f"a8w4_down_stream_bm{block_m}",
            _stream_down_kernel(
                H, I, topk, block_m, E, False, device.index, _stream_down_constants()
            ),
            mq,
            weight.w2,
            weight.w2_delta,
            routes,
            *down_args,
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
    so a token whose ids are all -1 gets 0. Ids are distinct within a token apart from -1 (with a repeated id the
    Opus-sorted prefill keeps one route, the other paths every route); other ids are invalid. Only ``DEBUG_CHECKS``
    (``AITER_MOE_A8W4_DEBUG=1``, a host synchronization) reads the ids to check both rules (``ValueError``). int64
    ids, other floating weights and non-contiguous routing tensors are converted first.
    The input and the f32 intermediate SiLU(gate) * up are quantized per row to FP8
    E4M3FNUZ with ``scale = amax / 240`` and ``q = RNE(x / scale)``. SiLU only, no expert mask /
    bias; ``M * topk * I_padded * 4`` and ``M * H * 2`` below ``2**31``; a call that takes the prefill Down (the
    prefill tier, or streaming blocks of 64 from hidden 4096 on) needs ``block_m * I_padded <= 64 KiB`` (I_padded <=
    1024 by default). Both raise ``ValueError`` before any kernel runs. ``output`` may be a caller's contiguous
    BF16 [M, H] tensor, ``hidden_states`` itself included. Issues no host synchronization; compile a shape
    outside a CUDA graph capture before capturing it. Raises ``RuntimeError`` off gfx942.

    Unless a tile option is given, M <= ``DECODE_MAX_M`` runs the decode kernels (no sorting) when they support the
    shape (H <= 4096 or in ``DECODE_WIDE_HIDDEN``, the intermediate rows fit in LDS; with expert deduplication up to
    ``DECODE_DEDUP_MAX_M`` tokens while at most 1.5 routes fall on an expert), larger M with at most
    ``STREAM_MAX_ROWS`` rows per expert (``STREAM_MAX_ROWS_WIDE`` for hidden >= 4096, ``STREAM_MAX_ROWS_H5120`` for
    hidden >= 5120 with many routes, ``STREAM_MAX_ROWS_FUSED_QUANT`` when the Gate/Up of blocks of 64 quantizes the
    intermediate) the streaming kernels and the rest the prefill kernels (TILE_K 128 when chosen for having more
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
    ``DECODE_SHARED_MIXED_MAX_M`` tokens, when the routed experts take the per-route decode kernels (no tile option,
    decided on the routed columns as the tier is; where the fold below is possible as well, while tokens times routed
    topk times padded I stay within ``DECODE_SHARED_MIXED_MAX_WORK``), it runs as the last route of the mixed decode
    kernels (two launches, every column summed in f32 and rounded to BF16 once). Otherwise, where the routed path
    allows it (``_shared_fold_tier``: the decode tier's slot-pair Down, the streaming tier, the prefill tier from
    ``SHARED_FOLD_PREFILL_MIN_M`` tokens; an ``output`` 16-byte aligned where the shared expert writes it), it is
    folded into the routed experts' last reduction (``_SharedFold``): two FP8 GEMMs on the routed path's quantized input
    with SiLU(gate) * up, the limit, a per-row FP8 quantization and the routing weight between them, on a side stream up
    to ``SHARED_FOLD_DUAL_STREAM_MAX_M`` tokens, their BF16 output summed with the routes in f32 and rounded to BF16
    once (onto which the streaming tier's atomic Down adds the routes in BF16). Elsewhere it runs ATOM's split
    (``_shared_split``: aiter's per-token FP8 quantization of the input, ``gemm_a8w8_bpreshuffle`` for gate / up,
    ``silu_and_mul`` with ``swiglu_limit``, per-token FP8 quantization, ``gemm_a8w8_bpreshuffle`` for Down) on a side
    stream of the device up to ``SHARED_DUAL_STREAM_MAX_M`` tokens (its routing weight on another; unless ``output``
    overlaps ``hidden_states``), else before the routed experts on the current stream, and is added to their output
    with that weight per token, rounded to BF16 (ATOM's ``addcmul_``). The streams can be captured in a CUDA graph
    (compile outside the capture first, as above). Tile options apply to the routed experts.

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
    if hidden_states.shape[0] == 0:
        return output
    routed_topk = topk_ids.shape[1] - (1 if weight.has_shared else 0)
    _check_prefill_down(
        hidden_states.shape[0], weight, routed_topk, block_m, block_n, tile_k
    )
    if DEBUG_CHECKS:
        _check_routing(topk_ids, weight.experts, weight.has_shared)
    if weight.has_shared:
        return _moe_shared(
            hidden_states,
            weight,
            topk_weight,
            topk_ids,
            output,
            limit,
            (block_m, block_n, tile_k),
        )
    return _moe_routed(
        hidden_states,
        weight,
        topk_weight,
        topk_ids,
        output,
        limit,
        block_m,
        block_n,
        tile_k,
    )


def _side_streams(device):
    """The two side streams of ``device`` (created on first use): the shared expert's split and its routing weight
    (a chain of small kernels that would otherwise wait behind the routed or the shared kernels).
    """
    streams = _SIDE_STREAMS.get(device.index)
    if streams is None:
        streams = _SIDE_STREAMS[device.index] = tuple(
            torch.cuda.Stream(device=device) for _ in range(2)
        )
    return streams


_SIDE_STREAMS = {}


def _shared_split(hidden_states, weight, limit):
    """ATOM's split shared expert on the current stream with aiter's dense ops: per-token FP8 quantization of the
    input, the gate / up GEMM (``gemm_a8w8_bpreshuffle``, per-token x per-row scales), SiLU(gate) * up with the
    SwiGLU limit (``silu_and_mul``, BF16), per-token FP8 quantization of that, the Down GEMM (aiter zero-pads it to
    the weight's K, a multiple of 128). Returns BF16 [M, H]."""
    import aiter
    from aiter import dtypes

    xq, xs = aiter.per_token_quant_hip(hidden_states, quant_dtype=dtypes.fp8)
    last_path.append("shared_quant_x")
    gate_up = aiter.gemm_a8w8_bpreshuffle(
        xq, weight.shared_w1, xs, weight.shared_w1_scale, None, torch.bfloat16
    )
    last_path.append("shared_gemm_gateup")
    mid = torch.empty(
        (hidden_states.shape[0], weight.inter_padded),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    aiter.silu_and_mul(mid, gate_up, limit)
    last_path.append("shared_silu_and_mul")
    mq, ms = aiter.per_token_quant_hip(mid, quant_dtype=dtypes.fp8)
    last_path.append("shared_quant_mid")
    out = aiter.gemm_a8w8_bpreshuffle(
        mq, weight.shared_w2, ms, weight.shared_w2_scale, None, torch.bfloat16
    )
    last_path.append("shared_gemm_down")
    return out


def _shared_gate(topk_weight, topk_ids, experts):
    """The shared expert's weight per token, BF16 [M, 1] (ATOM's rule and rounding): ``topk_weight[:, -1]`` where
    ``topk_ids[:, 0] >= 0`` and ``topk_ids[:, -1] == experts``, else 0."""
    live = (topk_ids[:, 0] >= 0) & (topk_ids[:, -1] == experts)
    gate = torch.where(live, topk_weight[:, -1], torch.zeros_like(topk_weight[:, -1]))
    return gate.to(torch.bfloat16).unsqueeze(1)


@functools.cache
def _shared_mid_kernel(inter, kp, cols, experts, limit, loc, device):
    """``shared_a8w4.compile_shared_mid`` launcher. ``device`` only keys the cache."""
    return shared_a8w4.compile_shared_mid(
        I=inter, KP=kp, COLS=cols, E=experts, SWIGLU_LIMIT=limit, LOC=loc
    )


def _shared_gemm_tiles(M, N, K):
    """``(tile_m, tile_n, tile_k)`` of aiter's FlyDSL preshuffle FP8 GEMM (``flydsl_preshuffle_gemm_a8``) for the folded
    shared expert's Gate/Up (K = hidden) and Down (K = its weight's padded I) GEMMs of M rows and N columns.

    gfx942 has no tuned FP8 rows for these shapes, where ``gemm_a8w8_bpreshuffle`` runs CK's default kernels. Measured
    at the DeepSeek-V4.1 shapes (H5120, shared I 576 / 320; graph replay, PTL on), us, these tiles against CK:
    Gate/Up N1152: M4 5.5 vs 6.8, M64 7.9 vs 7.3, M128 9.2 vs 11.7, M256 13.0 vs 22.2, M512 17.7 vs 27.0, M4096 118 vs
    151 (N640 M96 / 128: tile_m 16 5.7 vs 32 8.1); Down K640: M4 3.3 vs 3.4, M64 4.2 vs 6.9, M256 8.4 vs 10.2, M512
    13.4 vs 21.8, M4096 80 vs 123 (K384 M4 2.9 vs 5.9; up to M48 now two waves of tile_n 32, as fast, see
    ``_shared_gemm``). The split's Down pads its input to K and runs CKTile: M4 22 us, M512 33 (PTL off).
    """
    if K >= 2048:
        if M <= 64 or (M <= 128 and N <= 640):
            return 16, 64, 512
        if M <= 256:
            return 32, 64, 512
        return (64, 128, 512) if M <= 512 else (64, 128, 256)
    if M <= 48:
        # Two waves (tile_n 32): a 16 x 128 FP8 A tile is 2 KiB, one 16-byte load per thread of 128 (see _shared_gemm).
        # Measured against (32, 64, 128) at the layers' M4-48: 0.98-1.00x the layer time.
        return 16, 32, 128
    return (32, 64, 128) if M <= 192 else (64, 128, 128)


def _shared_gemm(xq, wq, x_scale, w_scale, out):
    """``out`` = BF16 (xq @ wq.T) * x_scale * w_scale.T with aiter's FlyDSL preshuffle FP8 GEMM (the operands of
    ``gemm_a8w8_bpreshuffle``: wq in ``shuffle_weight(layout=(16, 16))``, scales [M, 1] / [N, 1]; tiles of
    ``_shared_gemm_tiles``, no split-K) on the current stream."""
    from .gemm_kernels import flydsl_preshuffle_gemm_a8

    M, K = xq.shape
    tile_m, tile_n, tile_k = _shared_gemm_tiles(M, wq.shape[0], K)
    # The GEMM stages each A tile in LDS with one tiled copy of 16-byte loads, one per thread (128 threads with
    # tile_n 32, else 256) and row: a tile of fewer bytes makes the copy cover rows past tile_m, which land in the
    # other LDS stage while other waves may still read it (kernels/preshuffle_gemm.py, four waves, tile_m 16 with
    # tile_k 128: wrong 8 x 16 output blocks now and then when other kernels share the CUs).
    assert tile_m * tile_k >= (128 if tile_n == 32 else 256) * 16
    flydsl_preshuffle_gemm_a8(xq, wq, x_scale, w_scale, out, tile_m, tile_n, tile_k)


class _SharedFold:
    """The shared expert of a call that folds it into the routed experts' last reduction (``_moe_shared_fold``).

    Its chain: the Gate/Up GEMM (``_shared_gemm``) on the routed path's FP8 input and per-row scale (the stream prep's,
    the row kernel's of prefill and of the decode tier, the plan kernel's of the deduplicated decode Gate/Up, or with
    ``start(None, None, ...)`` its own row quantization of x), then
    ``shared_a8w4.compile_shared_mid`` (SiLU(gate) * up with the SwiGLU limit, per-row FP8 for Down with the shared
    expert's routing weight folded into the row scale, optionally its loc_ids column) and the Down GEMM
    (``_shared_gemm``) into ``out``: M rows of the routes buffer that ``a8w4_route_sum`` /
    ``sorted_sum`` add as the last column of loc_ids, or the output itself, onto which the slot-pair decode Down or the
    streaming Down's BF16x2 atomics add the routes. A token whose shared expert is dropped (ATOM's rule, as
    ``_shared_gate``) gets +0 there. With a side stream (``side``) the side stream waits for the current stream where
    the routed path calls ``start`` (after the kernel that quantizes the input) and its kernels are issued by ``fork``
    (SHARED_FOLD_FORK_AFTER: right after the routed path's next launch) and joined by ``join`` (before the kernel that
    reads ``out``); without one the chain runs on the current stream at ``start``. ``hold`` keeps the Down GEMM back
    until ``release`` (an output in place that a routed kernel still reads).
    """

    def __init__(self, hidden_states, weight, topk_weight, topk_ids, limit, side):
        self.hidden_states = hidden_states
        self.weight = weight
        self.topk_weight = topk_weight
        self.topk_ids = topk_ids
        self.limit = limit
        self.side = side
        self.current = torch.cuda.current_stream(hidden_states.device)
        self.held = None
        self.pending = None

    def _front(self, xq, xs, loc, base):
        w = self.weight
        if xq is None:
            x = self.hidden_states
            xq = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
            xs = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
            _launch(
                "shared_fold_quant_x",
                _row_quant_decode(x.shape[1], True, x.device.index),
                x,
                xq,
                xs,
                x.shape[0],
            )
        M = xq.shape[0]
        device = xq.device
        gate_up = torch.empty(
            (M, w.shared_w1.shape[0]), dtype=torch.bfloat16, device=device
        )
        _shared_gemm(xq, w.shared_w1, xs.view(M, 1), w.shared_w1_scale, gate_up)
        last_path.append("shared_fold_gemm_gateup")
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
        return mq, ms

    def _down(self, mq, ms, out):
        w = self.weight
        _shared_gemm(mq, w.shared_w2, ms, w.shared_w2_scale, out)
        last_path.append("shared_fold_gemm_down")

    def start(self, xq, xs, out, loc=None, base=0, hold=False):
        """Run (or fork) the chain: ``xq`` / ``xs`` FP8 [M, H] / f32 [M] (the routed path's quantized input; None: the
        chain quantizes x first), ``out`` BF16 [M, H] (contiguous, 16-byte aligned), ``loc`` int32 [M, topk + 1] whose
        last column gets ``base + t`` (or -1). With a side stream and SHARED_FOLD_FORK_AFTER the side stream waits for the current stream's kernels
        so far but its kernels are issued by ``fork``, which the routed path calls right after its next launch.
        """
        if self.side is None:
            mq, ms = self._front(xq, xs, loc, base)
            if hold:
                self.held = (mq, ms, out)
            else:
                self._down(mq, ms, out)
            return
        ready = torch.cuda.Event()
        ready.record(self.current)
        self.pending = (ready, xq, xs, out, loc, base, hold)
        if not SHARED_FOLD_FORK_AFTER:
            self.fork()

    def fork(self):
        """Issue the side stream's chain held by ``start`` (no-op without one)."""
        if self.pending is None:
            return
        ready, xq, xs, out, loc, base, hold = self.pending
        self.pending = None
        self.side.wait_event(ready)
        last_path.append("shared_fork")
        with torch.cuda.stream(self.side):
            mq, ms = self._front(xq, xs, loc, base)
            if hold:
                self.held = (mq, ms, out)
            else:
                self._down(mq, ms, out)
        # Tensors of the current stream that the side stream uses (as _moe_shared records them).
        x = self.hidden_states if xq is None else None
        for t in (x, xq, xs, self.topk_weight, self.topk_ids, out, loc):
            if t is not None:
                t.record_stream(self.side)

    def release(self):
        """Run a held Down GEMM once the current stream's kernels so far are done."""
        self.fork()
        if self.held is None:
            return
        if self.side is None:
            self._down(*self.held)
        else:
            self.side.wait_stream(self.current)
            with torch.cuda.stream(self.side):
                self._down(*self.held)
        self.held = None

    def join(self):
        """The current stream waits for the chain."""
        self.release()
        if self.side is not None:
            self.current.wait_stream(self.side)
            last_path.append("shared_join")


def _decode_slot_pair_down(M, weight, topk):
    """Whether the decode tier's Down of M tokens is the slot-pair Down (``compile_a8w4_down_decode_pq``: the per-route
    path from DECODE_PREQUANT_MIN_M tokens, the DECODE_WIDE_HIDDEN sizes' deduplicated Gate/Up), as
    ``_fused_moe_decode`` chooses it, and not the single-slot or the deduplicated Down.
    """
    H, I = weight.hidden, weight.inter_padded
    if M >= DECODE_DEDUP_MIN_M and H >= DECODE_DEDUP_MIN_H and _down_dd_blocks(H, I):
        return False
    pairs = _slot_pair_offsets_fit(weight) and _decode_fits(H, I, topk, True)
    if H in DECODE_WIDE_HIDDEN and M >= DECODE_WIDE_DEDUP_MIN_M and pairs:
        return True
    return M >= DECODE_PREQUANT_MIN_M and pairs


def _shared_fold_tier(M, weight, topk, output, tiles):
    """The routed tier into whose last reduction a call (``topk`` routed columns, M > 0) folds its shared expert
    (``_SharedFold``), or None for ATOM's split: decode while the Down is the slot-pair Down
    (``_decode_slot_pair_down``), streaming and prefill from SHARED_FOLD_PREFILL_MIN_M tokens on; never without
    SHARED_FOLD. Where the shared Down GEMM writes the output itself (decode, the streaming atomic Down) the output
    must be 16-byte aligned (its 16-byte stores). The routed columns decide, as they decide the tier.
    """
    if not SHARED_FOLD:
        return None
    H, I, E = weight.hidden, weight.inter_padded, weight.experts
    tier = _routed_tier(M, H, I, topk, E, *tiles)
    aligned = output.data_ptr() % 16 == 0
    if tier == "decode":
        return tier if aligned and _decode_slot_pair_down(M, weight, topk) else None
    if tier == "stream":
        atomic = _stream_atomic(H, I, _stream_block_m(M, topk, E, H))
        return tier if aligned or not atomic else None
    return tier if M >= SHARED_FOLD_PREFILL_MIN_M else None


def _moe_shared_fold(
    hidden_states, weight, topk_weight, topk_ids, output, limit, tiles
):
    """``flydsl_moe_a8w4_ptpc`` with a shared expert folded into the routed experts' last reduction (``_SharedFold``):
    the routed tier runs on the full routing tensors and the chain on a side stream up to SHARED_FOLD_DUAL_STREAM_MAX_M
    tokens (an output in place included: the chain writes the output only after the routed kernels that read x),
    else on the current stream."""
    M = hidden_states.shape[0]
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
    side = (
        _side_streams(hidden_states.device)[0]
        if M <= SHARED_FOLD_DUAL_STREAM_MAX_M
        else None
    )
    fold = _SharedFold(hidden_states, weight, topk_weight, topk_ids, limit, side)
    return _moe_routed(
        hidden_states, weight, topk_weight, topk_ids, output, limit, *tiles, fold
    )


def _mixed_decode(M, weight, topk, tiles, fold=None):
    """Whether a call with a shared expert (``topk`` routed columns, M > 0) runs the mixed decode kernels
    (``_fused_moe_decode_mixed``): up to DECODE_SHARED_MIXED_MAX_M tokens, no tile option, the routed experts on the
    per-route decode path (decode tier, not the expert-deduplicated or the wide-hidden deduplicated Gate/Up, x quantized
    in the Gate/Up kernel; the routed columns decide it, as they decide the tier) and the mixed Down within its limits;
    where the call could fold the shared expert instead (``fold``: ``_shared_fold_tier``'s answer) only up to
    DECODE_SHARED_MIXED_MAX_WORK.
    """
    H, I, E = weight.hidden, weight.inter_padded, weight.experts
    return (
        M <= DECODE_SHARED_MIXED_MAX_M
        and not (fold and M * topk * I > DECODE_SHARED_MIXED_MAX_WORK)
        and _routed_tier(M, H, I, topk, E, *tiles) == "decode"
        and not (
            M >= DECODE_DEDUP_MIN_M
            and H >= DECODE_DEDUP_MIN_H
            and _down_dd_blocks(H, I)
        )
        and not (H in DECODE_WIDE_HIDDEN and M >= DECODE_WIDE_DEDUP_MIN_M)
        and M * topk * (I // 16) < DECODE_GATEUP_PREQUANT_MIN_WGS
        and decode_a8w4.down_decode_mixed_fits(I, topk + 1)
    )


def _moe_shared(hidden_states, weight, topk_weight, topk_ids, output, limit, tiles):
    """``flydsl_moe_a8w4_ptpc`` with a shared expert: the mixed decode kernels when ``_mixed_decode`` holds, else folded
    into the routed experts' last reduction where ``_shared_fold_tier`` allows it (``_moe_shared_fold``), else the
    routed experts on the first topk columns (``_moe_routed``), the shared expert (``_shared_split``) and its routing
    weight (``_shared_gate``) on side streams up to SHARED_DUAL_STREAM_MAX_M tokens, else first on the current stream,
    then ``output += shared * gate`` (BF16 operands, one rounding: ATOM's addcmul_)."""
    M = hidden_states.shape[0]
    fold = _shared_fold_tier(M, weight, topk_ids.shape[1] - 1, output, tiles)
    if _mixed_decode(M, weight, topk_ids.shape[1] - 1, tiles, fold):
        return _fused_moe_decode_mixed(
            hidden_states, weight, topk_weight, topk_ids, output, limit
        )
    if fold:
        return _moe_shared_fold(
            hidden_states, weight, topk_weight, topk_ids, output, limit, tiles
        )
    E = weight.experts
    routed_weight, routed_ids = topk_weight[:, :-1], topk_ids[:, :-1]
    if M <= SHARED_DUAL_STREAM_MAX_M and not _overlaps(output, hidden_states):
        # Fork the side streams from the current one and join them back before the add (also inside a graph
        # capture); tensors crossing streams are recorded on the streams that use them.
        current = torch.cuda.current_stream(hidden_states.device)
        side, aux = _side_streams(hidden_states.device)
        side.wait_stream(current)
        aux.wait_stream(current)
        last_path.append("shared_fork")
        with torch.cuda.stream(side):
            shared = _shared_split(hidden_states, weight, limit)
        with torch.cuda.stream(aux):
            gate = _shared_gate(topk_weight, topk_ids, E)
        hidden_states.record_stream(side)
        topk_weight.record_stream(aux)
        topk_ids.record_stream(aux)
        _moe_routed(
            hidden_states, weight, routed_weight, routed_ids, output, limit, *tiles
        )
        current.wait_stream(side)
        current.wait_stream(aux)
        shared.record_stream(current)
        gate.record_stream(current)
        last_path.append("shared_join")
    else:
        # The shared expert first: output may be hidden_states itself.
        shared = _shared_split(hidden_states, weight, limit)
        _moe_routed(
            hidden_states, weight, routed_weight, routed_ids, output, limit, *tiles
        )
        gate = _shared_gate(topk_weight, topk_ids, E)
    output.addcmul_(shared, gate)
    last_path.append("shared_add")
    return output


def _many_rows(M, H, I, topk, E):
    """Whether the mean rows per expert (M * topk / E) exceed the streaming kernels' limit (``_stream_max_rows``)."""
    return M * topk > _stream_max_rows(H, I, M * topk) * E


def _routed_tier(M, H, I, topk, E, block_m, block_n, tile_k):
    """Tier of the routed experts of a call (M > 0): "decode", "stream" or "prefill". Chosen before the prefill tile
    defaults are filled in: any tile option selects prefill."""
    if block_m is None and block_n is None and tile_k is None:
        if (
            M <= DECODE_MAX_M or _dedup_beyond_decode(M, H, I, topk, E)
        ) and _decode_fits(H, I, topk, False):
            return "decode"
        if M > DECODE_MAX_M and not _many_rows(M, H, I, topk, E):
            return "stream"
    return "prefill"


def _prefill_down_block_m(M, H, I, topk, E, block_m, block_n, tile_k):
    """Sorting block of the prefill Down (gemm2_a8w4) the routed experts of a call build (the prefill tier, or the
    streaming tier's blocks of 64, ``_stream_prefill_down``), 0 if none."""
    tier = _routed_tier(M, H, I, topk, E, block_m, block_n, tile_k)
    if tier == "decode":
        return 0
    if tier == "stream":
        stream_bm = _stream_block_m(M, topk, E, H)
        return stream_bm if _stream_prefill_down(H, I, stream_bm) else 0
    return 64 if block_m is None else block_m


def _check_prefill_down(M, weight, topk, block_m, block_n, tile_k):
    """ValueError before any launch when the call's prefill Down cannot hold its block_m x padded-I FP8 activation
    tile in LDS (``gemm2_a8w4``: block_m * I_padded <= 64 KiB, so I_padded <= 1024 with the default block_m 64); the
    decode and streaming Downs take any padded I their dispatch accepts."""
    I = weight.inter_padded
    # Every prefill Down of the call fits (the streaming tier's takes blocks of 64): no tier needed.
    if I * (64 if block_m is None else block_m) <= layout.LDS_BYTES:
        return
    bm = _prefill_down_block_m(
        M, weight.hidden, I, topk, weight.experts, block_m, block_n, tile_k
    )
    if bm and bm * I > layout.LDS_BYTES:
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
        raise ValueError(
            "a token's routed expert ids must be distinct (-1 aside): with a repeated id the Opus-sorted prefill "
            "keeps one route, the other paths every route"
        )


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
):
    """The routed experts of ``flydsl_moe_a8w4_ptpc`` (M > 0) into ``output``: the tier and its kernels. With a folded
    shared expert (``shared``, ``_SharedFold``) the routing tensors keep its column, the last: the routed experts take
    the others, and each tier starts the shared expert's chain on its FP8 input and joins it before its last
    reduction."""
    from aiter.fused_moe import moe_sorting

    M, H = hidden_states.shape
    cols = topk_ids.shape[1]
    topk = cols - (shared is not None)
    E, I = weight.experts, weight.inter_padded
    device = hidden_states.device
    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weight.dtype != torch.float32 or not topk_weight.is_contiguous():
        topk_weight = topk_weight.to(torch.float32).contiguous()
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

    prep_routes = (
        PREFILL_PREP_MAX_ROUTES_H5120 if H >= 5120 else PREFILL_PREP_MAX_ROUTES
    )
    if H >= 2560 and H % 512 == 0 and M * topk <= prep_routes:
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
    else:
        sort_weight, sort_ids = topk_weight, topk_ids
        if shared is not None:
            sort_weight = topk_weight[:, :topk].contiguous()
            sort_ids = topk_ids[:, :topk].contiguous()
        sorted_ids, sorted_weights, sorted_experts, num_valid = moe_sorting(
            sort_ids, sort_weight, E, H, hidden_states.dtype, block_m, accumulate=False
        )[:4]
        last_path.append("moe_sorting_opus")
        # The sort's inverse output (output_aux) costs more than Down storing it: the input quantization
        # sets loc to -1 (a repeated expert id of a token keeps one route), Down stores each route's sorted row.
        loc_ids = torch.empty((M, cols), dtype=torch.int32, device=device)
        xq, xs = _quantize(hidden_states, M, H, "bf16", fill=loc_ids)
    grid = sorted_experts.numel()
    routing = (sorted_ids, sorted_weights, sorted_experts, num_valid)
    rows = grid * block_m
    routes = torch.empty(
        (rows + (0 if shared is None else M), H), dtype=torch.bfloat16, device=device
    )
    if shared is not None:
        shared.start(xq, xs, routes[rows:], loc_ids, rows)

    mid = torch.empty((M, topk, I), dtype=torch.float32, device=device)
    gateup = _gateup_kernel(
        2 * I,
        H,
        topk,
        block_m,
        block_n,
        tile_k,
        E,
        limit,
        _xcd_chunk(GATEUP_XCD_CHUNK_H5120, H, M * topk),
        device.index,
        _trace_constants(),
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
    if shared is not None:
        shared.join()
    sorted_sum(cols, H)(loc_ids, routes, output, M)
    last_path.append("sorted_sum")
    return output


# fused_moe parameters that _fused_moe_route reads; every other one must keep its fused_moe default, except these
# hints that do not change the result: moe_sorting_dispatch_policy only schedules aiter's own moe_sorting (this op
# sorts with its own prep kernel or moe_sorting's default), splitk only moves the order of the K sums (fused_moe_
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
