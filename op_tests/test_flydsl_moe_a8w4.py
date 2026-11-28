# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark of the gfx942 FlyDSL A8W4 MoE (``flydsl_moe_a8w4_ptpc``, also reached through
``aiter.fused_moe`` with the tagged weights of ``prepare_moe_a8w4_ptpc_weight``): BF16 activations quantized
per token to FP8 E4M3FNUZ, MXFP4 weights (E2M1 + one E8M0 per 32 K), SiLU, BF16 output; optionally a shared expert
in FP8 with per-row scales (DeepSeek-V4.1-Flash), expert E of the weight.

One file, four regions:
  0. shared: the independent fp32 reference (a port of pyhip's ``testing/moe_a8w4.py``: weights and inputs drawn
     with the same generators and seeds as pyhip's benchmark, exact MXFP4 dequantization, per-row FP8
     activations; the shared expert's FP8 weights in both checkpoint formats and the reference of a whole layer) and
     the grading (``calc_diff <= 1e-3`` and every element within 4.5% of ``max|ref|``);
  1. basic functionality (``@pytest.mark.basic``): one case per target shape and token-count tier, per routed tier of
     the layers with the shared expert, and one for each ``aiter.fused_moe`` call form, the first thing to run after
     every change: ``pytest op_tests/test_flydsl_moe_a8w4.py -m basic``;
  2. other coverage: what regions 1 and 3 cannot cover (weight layout and delta flush, bit-exact row
     quantization, dedup plan, streaming prep and fused quantization, decode dispatch boundaries (per route and
     expert-deduplicated), streaming boundaries and variants (the atomic Down's epilogue branches and LDS budget
     among them), the prefill sorting front end, prefill tile options and the half-block / XCD / few-waves / two-pass
     / Down-split variants, CUDA graph, input errors and the gfx942 gate, compile-cache
     keys, repeated expert ids, in-place output, routing input dtypes, M = 0, the SwiGLU limit, the
     ``aiter.fused_moe`` entry: its tiers, arguments and errors, untagged weights, CUDA graph, the weight tag, and the
     shared expert: prepare's bit-exact conversions, the real checkpoints, errors, graph, in-place output, the mixed
     decode: its boundaries with the fold, the FP8 bytes of its fast row quantization, masks, reads and other shapes,
     the fold: every tier with masks, in place and under graph replay, its fallback to the streaming tier, GEMMs in
     row chunks and unaligned outputs, and the launch records behind the benchmark's per-kernel roofline: kernels per
     stream against the profiler, FLOPs, bytes, graph capture, and the benchmark's matching of the profiled events
     onto them): ``pytest op_tests/test_flydsl_moe_a8w4.py -m "not basic"``;
  3. benchmark (``main``, not collected by pytest): Qwen3.8 TP2 / TP4 (against aiter's FP8 per-token
     ``fused_moe``), MiMo-V2.6-Flash TP8 / TP4 and DeepSeek-V4.1-Flash TP4 x M = 1, 2, 4, ..., 32768 and 48 (85
     cases), then the DeepSeek-V4.1-Flash main and MTP layers with the shared expert, TP4 / TP8, at the same token
     counts and M = 6, the MTP TP4 layer also at M = 1280, and 22 cases at the boundaries of the dispatch (95 cases,
     against aiter's FP8 per-token ``fused_moe`` and ATOM's split), with the per-kernel roofline of each case in INFO
     lines: ``python op_tests/test_flydsl_moe_a8w4.py`` (``--quick``: the layers at 6 token counts).
``MF_SEED`` seeds the benchmark's weights and inputs.
"""

import argparse
import contextlib
import copy
import functools
import gc
import inspect
import itertools
import json
import math
import os
import shutil
import tempfile
import weakref
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import (
    MoeA8W4PtpcWeight,
    flydsl_moe_a8w4_ptpc,
    moe_a8w4,
    prepare_moe_a8w4_ptpc_weight,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import (
    common_a8w4,
    gemm2_a8w4,
    stream_a8w4,
)
from aiter.test_common import benchmark, checkAllclose

# =================================================================================================
# Region 0: shared -- independent fp32 reference and grading
# Port of pyhip src/pyhip/testing/moe_a8w4.py (tag moe-ptpc-a8w4-r2-M17): weights and inputs are drawn exactly
# as there (same generators, seeds and draw order), so the benchmark's inputs equal pyhip's. The reference
# uses torch only and never calls the kernels under test: exact MXFP4 dequantization, the input and the
# intermediate quantized per row to FP8 E4M3FNUZ (scale = amax / 240, the quantization points of the op).
# =================================================================================================

SUPPORTED_GFX = ["gfx942"]
SEED = int(os.environ.get("MF_SEED", "0"))
FP8_MAX = 240.0
CALC_DIFF_MAX = 1e-3
# The same level as tingqli's refactor/a8w4 (its error against an FP8-activation reference is about 4.2%).
MAX_ERR_RATIO = 0.045
_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _on_supported_gfx():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() in SUPPORTED_GFX
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _on_supported_gfx(), reason="the A8W4 MoE targets gfx942"
)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)


def make_weights(experts, hidden, inter, *, seed=0, fp8_inter=None, device="cuda"):
    """Random BF16 MoE weights (drawn expert by expert) quantized to standard MXFP4; optionally an FP8
    per-channel version of the same BF16 weights.

    Returns a dict: ``w1`` float4_e2m1fn_x2 [E, 2I, H/2] (rows [0, I) gate, [I, 2I) up),
    ``w1_scale`` float8_e8m0fnu [E, 2I, H/32], ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32].
    With ``fp8_inter`` (>= inter) also the same BF16 weights with I zero-padded to ``fp8_inter`` and
    quantized per output channel to FP8 E4M3FNUZ (not shuffled): ``w1_fp8`` [E, 2*fp8_inter, H], ``w1_fp8_scale``
    [E, 2*fp8_inter, 1], ``w2_fp8`` [E, H, fp8_inter], ``w2_fp8_scale`` [E, H, 1] (float32).
    Standard deviations: w1 1/sqrt(H), w2 1/sqrt(I).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    mx = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    h, i = hidden, inter
    out = {
        "w1": torch.empty((experts, 2 * i, h // 2), dtype=torch.uint8, device=device),
        "w1_scale": torch.empty(
            (experts, 2 * i, h // 32), dtype=torch.uint8, device=device
        ),
        "w2": torch.empty((experts, h, i // 2), dtype=torch.uint8, device=device),
        "w2_scale": torch.empty(
            (experts, h, i // 32), dtype=torch.uint8, device=device
        ),
    }
    fp8 = torch.float8_e4m3fnuz
    if fp8_inter is not None:
        assert fp8_inter >= inter
        p = fp8_inter
        pt = aiter.get_torch_quant(aiter.QuantType.per_Token)
        out.update(
            {
                "w1_fp8": torch.empty((experts, 2 * p, h), dtype=fp8, device=device),
                "w1_fp8_scale": torch.empty(
                    (experts, 2 * p, 1), dtype=torch.float32, device=device
                ),
                "w2_fp8": torch.empty((experts, h, p), dtype=fp8, device=device),
                "w2_fp8_scale": torch.empty(
                    (experts, h, 1), dtype=torch.float32, device=device
                ),
            }
        )
    for e in range(experts):
        v1 = torch.randn((2 * i, h), generator=gen, device=device) * h**-0.5
        v2 = torch.randn((h, i), generator=gen, device=device) * i**-0.5
        v1, v2 = v1.to(torch.bfloat16), v2.to(torch.bfloat16)
        for name, value in (("w1", v1), ("w2", v2)):
            q, s = mx(value, quant_dtype=torch.float4_e2m1fn_x2)
            out[name][e].copy_(q.view(torch.uint8))
            out[f"{name}_scale"][e].copy_(
                s.view(torch.uint8).reshape(out[f"{name}_scale"][e].shape)
            )
        if fp8_inter is not None:
            z = v1.new_zeros((p - i, h))
            v1p = torch.cat([v1[:i], z, v1[i:], z])
            v2p = torch.cat([v2, v2.new_zeros((h, p - i))], dim=1)
            for name, value in (("w1_fp8", v1p), ("w2_fp8", v2p)):
                q, s = pt(value, quant_dtype=fp8)
                out[name][e].copy_(q)
                out[f"{name}_scale"][e].copy_(s.view(-1, 1))
    for name in ("w1", "w2"):
        out[name] = out[name].view(torch.float4_e2m1fn_x2)
        out[f"{name}_scale"] = out[f"{name}_scale"].view(torch.float8_e8m0fnu)
    return out


def make_inputs(tokens, hidden, experts, topk, *, seed=0, shared=True, device="cuda"):
    """Returns ``(hidden_states, topk_weight, topk_ids)``.

    hidden_states BF16 [M, H] ~ N(0, 1). ``shared`` (Qwen3.8): the last expert (E-1) is the fused shared expert,
    selected by every token's last slot; the other topk-1 are uniformly random and distinct in [0, E-1).
    topk_weight float32: the routed ones the softmax of topk-1 random logits, the shared one sigmoid(randn).
    Otherwise (MiMo) topk experts uniformly random and distinct in [0, E), weights the softmax of topk random
    logits. ids int32 and weights float32, both contiguous.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=gen, device=device).to(torch.bfloat16)
    if not shared:
        ids = (
            torch.rand((tokens, experts), generator=gen, device=device)
            .topk(topk, dim=-1)
            .indices
        )
        logits = torch.randn((tokens, topk), generator=gen, device=device)
        return (
            x,
            torch.softmax(logits, dim=-1).float().contiguous(),
            ids.to(torch.int32).contiguous(),
        )
    routed = (
        torch.rand((tokens, experts - 1), generator=gen, device=device)
        .topk(topk - 1, dim=-1)
        .indices
    )
    shared = torch.full((tokens, 1), experts - 1, dtype=routed.dtype, device=device)
    ids = torch.cat([routed, shared], dim=1).to(torch.int32).contiguous()
    logits = torch.randn((tokens, topk - 1), generator=gen, device=device)
    shared_w = torch.sigmoid(torch.randn((tokens, 1), generator=gen, device=device))
    weights = (
        torch.cat([torch.softmax(logits, dim=-1), shared_w], dim=1).float().contiguous()
    )
    return x, weights, ids


def mask_routes(
    topk_weight, topk_ids, *, seed, share=0.3, full=(), first=(), routed=None
):
    """Masked copies ``(topk_weight, topk_ids)`` of a routing: a random ``share`` of the routed ids (the first
    ``routed`` columns, all by default), every routed id of the rows in ``full`` (``"all"``: every row) and the first id
    of the rows in ``first`` set to -1, with weight NaN (a masked route must add nothing, whatever its weight).
    """
    gen = torch.Generator(device=topk_ids.device).manual_seed(seed)
    weights, ids = topk_weight.clone(), topk_ids.clone()
    cols = ids.shape[1] if routed is None else routed
    drop = torch.rand((ids.shape[0], cols), generator=gen, device=ids.device) < share
    drop[slice(None) if full == "all" else list(full)] = True
    drop[list(first), 0] = True
    ids[:, :cols] = torch.where(drop, -1, ids[:, :cols])
    weights[:, :cols] = torch.where(drop, float("nan"), weights[:, :cols])
    return weights, ids


def poisoned(w, prepared):
    """Experts 1 .. E-1 of a make_weights dict and of its prepared weight, as views (E - 1 experts): the bytes just
    before every routed tensor of the view are expert 0's, real weights, where a kernel addressing a masked route's
    expert (-1) would read. Masked cases run on such views, so that such a read shows in the result (and stays inside
    the allocation)."""
    import dataclasses

    view = dataclasses.replace(
        prepared,
        **{
            name: getattr(prepared, name)[1:]
            for name in ("w1", "w1_delta", "w1_scale", "w2", "w2_delta", "w2_scale")
        },
        experts=prepared.experts - 1,
    )
    return {name: t[1:] for name, t in w.items()}, view


def dequant_mxfp4(q, scale):
    """Standard MXFP4 [..., N, K/2] + E8M0 [..., N, K/32] -> the exact float32 [..., N, K]."""
    b = q.view(torch.uint8)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).long()
    value = torch.tensor(_E2M1, dtype=torch.float32, device=b.device)[nib]
    code = scale.view(torch.uint8).to(torch.float64)
    factor = torch.exp2(code - 127).float()
    return (value.unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)).flatten(-2)


def quant_rows_fp8(t):
    """Quantize each row (last dimension) to FP8 E4M3FNUZ and dequantize it.

    scale = amax / 240, q = RNE(x / scale), both IEEE f32 divisions (an all-zero row takes scale 1). torch's
    ``tensor / python scalar`` becomes a multiplication by the reciprocal (up to 1 ulp off), so this divides by
    a tensor of the same shape.
    """
    t = t.float()
    amax = t.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (t / scale).to(torch.float8_e4m3fnuz).float() * scale


def reference(
    x, w1, w1_scale, w2, w2_scale, topk_weight, topk_ids, *, swiglu_limit=0.0
):
    """Independent fp32 MoE reference, float32 [M, H].

    Weights in standard MXFP4 (``make_weights``' format), SiLU(gate) * up, the routing weight applied after Down
    and summed per token. The input and the intermediate are quantized to FP8 with one scale per token row and
    per (token, slot) row. ``swiglu_limit`` = L > 0 applies DeepSeek-V4.1's SwiGLU limit (``Expert`` in its
    inference/model.py, config ``swiglu_limit``, 10 for V4.1-Flash) to the f32 gate / up (the product of the
    quantized input and the dequantized weights): gate = min(gate, L), up = clamp(up, -L, L), then
    SiLU(gate) * up; 0 (default) means no limit and gives the same bits as not passing it. An id of -1 masks its
    route: it adds nothing, whatever its weight.
    """
    M, H = x.shape
    E, n1 = w1.shape[0], w1.shape[1]
    I, topk = n1 // 2, topk_ids.shape[1]
    xa = quant_rows_fp8(x)
    flat = topk_ids.reshape(-1).long()
    # Masked routes sort last, as expert E, which the groups leave out.
    flat = torch.where(flat < 0, E, flat)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E + 1).tolist()[:E]
    tokens, slots = order // topk, order % topk
    groups, start = [], 0
    for e, n in enumerate(counts):
        if n:
            groups.append((e, tokens[start : start + n], slots[start : start + n]))
        start += n
    mid = torch.zeros((M, topk, I), dtype=torch.float32, device=x.device)
    for e, t, s in groups:
        h = xa[t] @ dequant_mxfp4(w1[e], w1_scale[e]).T
        gate, up = h[:, :I], h[:, I:]
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        mid[t, s] = torch.nn.functional.silu(gate) * up
    mid = quant_rows_fp8(mid)
    out = torch.zeros((M, H), dtype=torch.float32, device=x.device)
    for e, t, s in groups:
        y = mid[t, s] @ dequant_mxfp4(w2[e], w2_scale[e]).T
        out.index_add_(0, t, y * topk_weight[t, s].float().unsqueeze(-1))
    return out


def calc_diff(x, y):
    """1 - 2·Σxy / Σ(x² + y²) in double precision; 0 when both are all zero."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def check(out, ref, msg=""):
    """Grading against the fp32 reference: no NaN/Inf, ``calc_diff <= CALC_DIFF_MAX`` and every element within
    ``MAX_ERR_RATIO * max|ref|`` (``checkAllclose`` with rtol 0; its mismatch fraction is ``err``). Returns
    the details."""
    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return {"ok": False, "reason": "output contains NaN/Inf"}
    top = r.abs().max().item()
    err = checkAllclose(r, o, rtol=0, atol=MAX_ERR_RATIO * top, msg=msg, printLog=False)
    diff = calc_diff(o, r)
    worst = (o - r).abs().max().item()
    ratio = worst / top if top else (0.0 if worst == 0 else math.inf)
    ok = err == 0 and diff <= CALC_DIFF_MAX
    return {
        "ok": ok,
        "err": err,
        "calc_diff": diff,
        "max_err": worst,
        "max_ref": top,
        "err_ratio": ratio,
    }


def assert_close(out, ref):
    result = check(out, ref)
    assert result["ok"], f"A8W4 output out of tolerance: {result}"
    return result


# -- the DeepSeek-V4.1-Flash layer with its shared expert ------------------------------------------------------------
# The shared expert is FP8 E4M3 with per-row float32 scales (FP4PTPC checkpoints) or E8M0 scales of 32x32 blocks (the
# official checkpoint). prepare takes it as expert E, routed in the last of topk + 1 columns (ATOM's E385 / topk-7
# convention). The reference dequantizes its weights exactly and quantizes the input (the same per-row FP8 as the
# routed reference) and the intermediate per row; the clamp of the SwiGLU limit as in ``reference``.

# name: (routed experts, routed topk, I); H 5120, SwiGLU limit 10, routed weights normalized and times 1.5
# (routed_scaling_factor), the shared expert's weight 1. TP8 pads the routed I 288 to 320 (prepare), the shared too.
LAYER_SHAPES = {
    "ds_layer_tp4": (384, 6, 576),
    "ds_layer_tp8": (384, 6, 288),
    "mtp_layer_tp4": (128, 3, 576),
    "mtp_layer_tp8": (128, 3, 288),
}
LAYER_HIDDEN = 5120
LAYER_SWIGLU_LIMIT = 10.0
LAYER_ROUTED_SCALING = 1.5
FP8_OCP_MAX = 448.0


def _pow2_ceil_exponent(t):
    """k = ceil(log2(t)) for positive float64 ``t`` from its exact binary exponent (t = m * 2**e, 0.5 <= m < 1)."""
    m, e = torch.frexp(t)
    return e - (m == 0.5).to(e.dtype)


def _fp8_rows(v):
    """float32 [N, K] -> E4M3FN [N, K] and float32 [N, 1] scales amax / 448 per row (1 for a zero row), RNE: the
    per-row format of FP4PTPC checkpoints."""
    amax = v.abs().amax(dim=1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_OCP_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (v / scale).to(torch.float8_e4m3fn), scale


def _fp8_blocks(v):
    """float32 [N, K] -> E4M3FN [N, K] and E8M0 [N/32, K/32] scales of 32x32 blocks, 2**ceil(log2(amax / 448)) (1 for
    a zero block), RNE: the official checkpoint's format."""
    n, k = v.shape
    amax = v.view(n // 32, 32, k // 32, 32).abs().amax(dim=(1, 3)).double()
    exp = _pow2_ceil_exponent(
        torch.where(amax > 0, amax, torch.full_like(amax, FP8_OCP_MAX)) / FP8_OCP_MAX
    )
    scale = torch.exp2(exp.double()).float()
    full = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    codes = (exp + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    return (v / full).to(torch.float8_e4m3fn), codes


def make_shared_weights(hidden, inter, *, seed=0, device="cuda"):
    """Random BF16 shared-expert weights in both FP8 checkpoint formats.

    Returns ``{"row": (w1, w1_scale, w2, w2_scale), "block": (...)}``, the arguments ``shared_w1``,
    ``shared_w1_scale``, ``shared_w2``, ``shared_w2_scale`` of prepare: ``w1`` float8_e4m3fn [2I, H] (rows [0, I) gate,
    [I, 2I) up), ``w2`` [H, I]; "row" with float32 [N, 1] scales (``_fp8_rows``), "block" with float8_e8m0fnu scales of
    32x32 blocks (``_fp8_blocks``; None unless H and I are multiples of 32), both of the same BF16 weights. Standard
    deviations: w1 1/sqrt(H), w2 1/sqrt(I).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    v1 = torch.randn((2 * inter, hidden), generator=gen, device=device) * hidden**-0.5
    v2 = torch.randn((hidden, inter), generator=gen, device=device) * inter**-0.5
    v1, v2 = v1.to(torch.bfloat16).float(), v2.to(torch.bfloat16).float()
    blocks = hidden % 32 == 0 and inter % 32 == 0
    return {
        "row": (*_fp8_rows(v1), *_fp8_rows(v2)),
        "block": (*_fp8_blocks(v1), *_fp8_blocks(v2)) if blocks else None,
    }


def dequant_fp8(w, scale):
    """FP8 [N, K] (E4M3FN or E4M3FNUZ) with float32 per-row scales ([N] / [N, 1]) or E8M0 scales of 32x32 blocks ->
    the exact float32 [N, K]."""
    v = w.float()
    if scale.dtype == torch.float32:
        return v * scale.reshape(-1, 1)
    factor = torch.exp2(scale.view(torch.uint8).double() - 127).float()
    return v * factor.repeat_interleave(32, 0).repeat_interleave(32, 1)


def reference_shared(x, w1, w1_scale, w2, w2_scale, *, swiglu_limit=0.0):
    """Independent fp32 reference of the shared expert, float32 [M, H] before its routing weight: the input quantized
    per row to FP8 (``quant_rows_fp8``, the routed reference's), the exactly dequantized FP8 weights, the SwiGLU limit
    as in ``reference``, SiLU(gate) * up quantized per row, Down."""
    inter = w1.shape[0] // 2
    h = quant_rows_fp8(x) @ dequant_fp8(w1, w1_scale).T
    gate, up = h[:, :inter], h[:, inter:]
    if swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    mid = quant_rows_fp8(torch.nn.functional.silu(gate) * up)
    return mid @ dequant_fp8(w2, w2_scale).T


def shared_route_weight(topk_weight, topk_ids, experts):
    """The shared expert's weight per token (ATOM's rule): the last column's weight where the first id is >= 0 and the
    last id is the shared expert's (E), else 0."""
    live = (topk_ids[:, 0] >= 0) & (topk_ids[:, -1] == experts)
    return torch.where(live, topk_weight[:, -1].float(), 0.0)


def make_layer_inputs(
    tokens, hidden, experts, topk, *, seed=0, shared_weight=1.0, device="cuda"
):
    """Inputs of a layer with a shared expert: ``(hidden_states, topk_weight, topk_ids)``, ids int32 and weights
    float32 [M, topk + 1], contiguous.

    hidden_states BF16 [M, H] ~ N(0, 1); the first topk columns uniformly random distinct routed experts in [0, E)
    with the softmax of topk random logits times LAYER_ROUTED_SCALING; the last column the shared expert E with weight
    ``shared_weight``, or with None a random weight per token in [0.5, 1.5) (so that a weight applied to the wrong
    token shows).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=gen, device=device).to(torch.bfloat16)
    ids = (
        torch.rand((tokens, experts), generator=gen, device=device)
        .topk(topk, dim=-1)
        .indices
    )
    logits = torch.randn((tokens, topk), generator=gen, device=device)
    routed = torch.softmax(logits, dim=-1) * LAYER_ROUTED_SCALING
    if shared_weight is None:
        shared = torch.rand((tokens, 1), generator=gen, device=device) + 0.5
    else:
        shared = torch.full((tokens, 1), float(shared_weight), device=device)
    ids = torch.cat([ids, torch.full_like(ids[:, :1], experts)], dim=1)
    return (
        x,
        torch.cat([routed, shared], dim=1).float().contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def reference_layer(x, w, shared, topk_weight, topk_ids, *, swiglu_limit=0.0):
    """Independent fp32 reference of a layer with a shared expert, float32 [M, H]: ``reference`` of the routed
    experts (MXFP4 ``w`` in make_weights' format, the first topk columns) plus ``reference_shared`` of ``shared``
    (prepare's four shared arguments) times ``shared_route_weight``."""
    experts = w["w1"].shape[0]
    routed = reference(
        x,
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        topk_weight[:, :-1],
        topk_ids[:, :-1],
        swiglu_limit=swiglu_limit,
    )
    weight = shared_route_weight(topk_weight, topk_ids, experts)
    return routed + reference_shared(
        x, *shared, swiglu_limit=swiglu_limit
    ) * weight.unsqueeze(-1)


def aiter_shared_split(x, w1, w1_scale, w2, w2_scale, swiglu_limit):
    """ATOM's split shared expert (its ptpc_split / a8w4 arms) with aiter's dense ops, BF16 [M, H] before the routing
    weight: per-token FP8 quantization of x, ``gemm_a8w8_bpreshuffle`` on gate / up (``w1`` [2I, H] in
    ``shuffle_weight(layout=(16, 16))``'s layout, ``w1_scale`` [2I, 1]), ``silu_and_mul`` with the SwiGLU limit,
    per-token FP8 quantization, ``gemm_a8w8_bpreshuffle`` on Down (``w2`` [H, K] shuffled with ``pad_k_to=128``; aiter
    zero-pads the activations to K). The comparison of the benchmark (a8w4_split) and of the accuracy cases.
    """
    q, s = aiter.per_token_quant_hip(x, quant_dtype=aiter.dtypes.fp8)
    gate_up = aiter.gemm_a8w8_bpreshuffle(q, w1, s, w1_scale, None, torch.bfloat16)
    mid = torch.empty(
        (x.shape[0], w1.shape[0] // 2), dtype=torch.bfloat16, device=x.device
    )
    aiter.silu_and_mul(mid, gate_up, swiglu_limit)
    q, s = aiter.per_token_quant_hip(mid, quant_dtype=aiter.dtypes.fp8)
    return aiter.gemm_a8w8_bpreshuffle(q, w2, s, w2_scale, None, torch.bfloat16)


def _prefill_path(tile_k, block_m=64, block_n=128):
    """The Gate/Up and Down kernels of the prefill path (the names carry the sorting block, Gate/Up's N tile and K
    tile), as pyhip M17's tests name them."""
    return [
        f"a8w4_gateup_prefill_bm{block_m}_bn{block_n}_tk{tile_k}",
        f"a8w4_down_prefill_bm{block_m}",
    ]


def _stream_path(block_m, fused):
    """The Gate/Up and Down kernels of the streaming path (the names carry the sorting block), as pyhip M17's tests
    name them: with ``fused`` the Gate/Up quantizes the intermediate in its epilogue."""
    gateup = (
        f"a8w4_gateup_quant_stream_bm{block_m}"
        if fused
        else f"a8w4_gateup_stream_bm{block_m}"
    )
    return [gateup, f"a8w4_down_stream_bm{block_m}"]


# The step that sums the routes the prefill Down wrote into the output (a8w4_route_sum, behind the prefill Down only:
# the streaming Down adds every route into the output with BF16x2 atomics, the decode Downs write it themselves).
_ROUTE_SUMS = ("a8w4_route_sum",)


# The Gate/Up and Down kernels of the per-route decode path's three dispatches (as pyhip M17's tests name them): x and
# mid quantized inside the kernels (nn), mid quantized by the row kernel first (np), both quantized first (pp).
_DECODE_PATHS = {
    "nn": ["a8w4_gateup_decode", "a8w4_down_decode"],
    "np": ["a8w4_gateup_decode", "a8w4_down_decode_pq"],
    "pp": ["a8w4_gateup_decode_pq", "a8w4_down_decode_pq"],
}
# The kernels of the expert-deduplicated decode (H 5120 from DECODE_WIDE_DEDUP_MIN_M tokens on, Gate/Up only): the row
# kernel quantizing x and writing the plan, the deduplicated Gate/Up, the row quantization of the intermediate and the
# per-token slot-pair Down writing the output (wide).
_DEDUP_PATHS = {
    "wide": [
        "a8w4_row_quant_plan_decode",
        "a8w4_gateup_decode_dd",
        "a8w4_row_quant_decode",
        "a8w4_down_decode_pq",
    ],
}


def _assert_path(expected):
    """The kernels of ``expected`` (exact names) occur in this order in ``moe_a8w4.last_path``, other kernels may lie
    between them (pyhip M17's check): the sorting front end, the row quantizations and the final sum are not
    asserted one by one, every case grades them end to end against the reference."""
    rest = iter(moe_a8w4.last_path)
    assert all(name in rest for name in expected), (expected, moe_a8w4.last_path)


def _assert_kernels(tier=None):
    """Tier-independent check of the last call: exactly one A8W4 Gate/Up kernel and then one A8W4 Down kernel ran
    (true of every path, pyhip M17's included); with ``tier`` some kernel name also contains that fragment, or, a
    list of kernel names, those kernels ran in this order (_assert_path).
    """
    path = moe_a8w4.last_path
    gateup = [i for i, name in enumerate(path) if name.startswith("a8w4_gateup")]
    down = [i for i, name in enumerate(path) if name.startswith("a8w4_down")]
    assert len(gateup) == 1 and len(down) == 1 and gateup[0] < down[0], path
    if isinstance(tier, list):
        _assert_path(tier)
    else:
        assert tier is None or any(tier in name for name in path), (tier, path)


def _assert_h5120(tokens, path):
    """H 5120 (DeepSeek): the kernels of ``path`` while ``tokens`` is within the H 5120 decode limit
    ``DECODE_WIDE_MAX_M``, past it the streaming kernels: the prep and sorting blocks of 16 (up to 12 rows per expert,
    M <= 48 at E 24), the Down adding the routes atomically at I 576 too (no a8w4_route_sum). The cases follow the
    constant (10 by default; the cases about the decode kernels pin pyhip M17's 32) instead of assuming its value.
    """
    if tokens <= moe_a8w4.DECODE_WIDE_MAX_M:
        assert path is not None, tokens
        _assert_path(path)
    else:
        _assert_path(["a8w4_stream_prep"] + _stream_path(16, False))
        assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path


def _a8w4_prepare(w):
    return prepare_moe_a8w4_ptpc_weight(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])


@functools.cache
def _weights(experts, hidden, inter, seed):
    """MXFP4 weights and their prepared layout, shared by the cases of one shape and seed."""
    w = make_weights(experts, hidden, inter, seed=seed)
    return w, _a8w4_prepare(w)


def _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options):
    """One call into a NaN-filled output buffer, and the reference of the same inputs."""
    out = torch.full_like(x, float("nan"))
    assert (
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out, **options)
        is out
    )
    ref = reference(
        x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids
    )
    return out, ref


def _assert_shared():
    """The call ran the shared expert: some step of ``moe_a8w4.last_path`` starts with "shared".

    Deliberately independent of how the op runs it, so that later milestones can change that without these cases:
    today the mixed decode kernels where moe_a8w4._mixed_decode holds (marker shared_mixed), else the fold into the
    routed experts' last reduction (shared_fold_*, with a side stream shared_fork / shared_join); other ways record a
    "shared..." marker of their own. What a layer case checks is its output against reference_layer.
    """
    path = moe_a8w4.last_path
    assert any(name.startswith("shared") for name in path), path


def _assert_layer_path(tier):
    """Path check of a layer call (routed experts plus the shared expert), as independent of the shared expert's
    implementation as _assert_shared: the routed experts took ``tier`` ("decode", "dedup", "stream" or "prefill",
    _path_tier: told by the first A8W4 Gate/Up kernel) and the shared expert ran."""
    assert _path_tier(moe_a8w4.last_path) == tier, (tier, moe_a8w4.last_path)
    _assert_shared()


@functools.cache
def _layer_weights(experts, inter, seed, fmt="row"):
    """MXFP4 routed weights (H 5120), the shared expert's four prepare arguments in format ``fmt`` ("row" or "block",
    make_shared_weights) and the prepared layer, cached per E, I, seed and format."""
    w = make_weights(experts, LAYER_HIDDEN, inter, seed=seed)
    shared = make_shared_weights(LAYER_HIDDEN, inter, seed=seed + 1)[fmt]
    prepared = prepare_moe_a8w4_ptpc_weight(
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        shared_w1=shared[0],
        shared_w1_scale=shared[1],
        shared_w2=shared[2],
        shared_w2_scale=shared[3],
    )
    return w, shared, prepared


def _shared_over(x, shared, limit):
    """[share of gate > L, share of |up| > L] of the shared expert's Gate/Up output at the reference's quantization
    point (x quantized per row)."""
    inter = shared[0].shape[0] // 2
    h = quant_rows_fp8(x) @ dequant_fp8(shared[0], shared[1]).T
    return [
        (h[:, :inter] > limit).float().mean().item(),
        (h[:, inter:].abs() > limit).float().mean().item(),
    ]


def _layer_check(
    experts,
    topk,
    inter,
    tokens,
    seed,
    *,
    fmt="row",
    amplify=1.0,
    mask=None,
    **options,
):
    """A whole layer call (routed experts plus the shared expert, ``LAYER_SWIGLU_LIMIT``) into a NaN-filled output,
    graded against ``reference_layer``: weights of _layer_weights(seed), inputs of make_layer_inputs(seed + 7) with
    random shared weights per token, scaled by ``amplify`` (with more than 1, more than 10% of the shared expert's gate
    and up values must exceed the limit first). With ``mask`` (``mask_routes`` arguments) the routed experts are the
    ``poisoned`` view of experts 1 .. E-1 and routed ids are masked to -1. Returns ``(prepared, x, topk_weight,
    topk_ids, out)``.
    """
    w, shared, prepared = _layer_weights(experts, inter, seed, fmt)
    if mask is not None:
        w, prepared = poisoned(w, prepared)
        experts -= 1
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, LAYER_HIDDEN, experts, topk, seed=seed + 7, shared_weight=None
    )
    if mask is not None:
        topk_weight, topk_ids = mask_routes(topk_weight, topk_ids, routed=topk, **mask)
    if amplify != 1.0:
        x = (x.float() * amplify).to(torch.bfloat16)
        over = _shared_over(x, shared, LAYER_SWIGLU_LIMIT)
        assert min(over) > 0.1, over
    out = torch.full_like(x, float("nan"))
    result = flydsl_moe_a8w4_ptpc(
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=LAYER_SWIGLU_LIMIT,
        **options,
    )
    assert result is out
    ref = reference_layer(
        x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert_close(out, ref)
    return prepared, x, topk_weight, topk_ids, out


# =================================================================================================
# Region 1: basic functionality (@pytest.mark.basic)
# prepare_moe_a8w4_ptpc_weight + flydsl_moe_a8w4_ptpc on whole calls, graded against the region-0 reference, with
# the launched kernels asserted. Qwen3.8 shapes: H 2560, topk 11, the last expert a fused shared expert that every
# token selects (E cut to 33 to keep the reference fast), TP2 I 320 (M 3 / 64 / 1031) and TP4 I 160, padded to 192
# by prepare (M 257). MiMo-V2.6-Flash: H 4096, E 256, topk 8, no shared expert, uniform routing, TP8 I 256
# (M 5 / 300) and TP4 I 512 (M 24 / 1600). DeepSeek-V4.1-Flash TP4: H 5120, topk 6, I 576 (E cut to 24), with
# swiglu_limit=10 and the inputs scaled by 10 so that more than 10% of the gate and of the up values exceed the
# limit (M 8 / 64). The token counts are one per tier of the pyhip original (decode, streaming, prefill). The
# prefill cases include multi-block experts and partially filled tail blocks.
# Each call runs one A8W4 Gate/Up and one Down kernel (_assert_kernels), and ``tier`` (a kernel-name fragment as in
# pyhip M17's tests, or the kernels of a path) pins pyhip M17's tier: the per-route decode ("_decode") for Qwen M 3,
# streaming ("_stream_") for Qwen M 64 (sorting blocks of 32, the Gate/Up quantizing the intermediate), MiMo TP8
# M 300 (blocks of 16) and DeepSeek M 64 (blocks of 32), prefill (sorted by the streaming prep) for Qwen M 1031,
# TP4 M 257 and MiMo TP4 M 1600, the per-route decode with the Down that quantizes its rows for MiMo TP8 M 5 (H 4096
# decodes up to DECODE_MAX_M = 7 tokens), streaming for MiMo TP4 M 24, and for DeepSeek M 8 the decode that deduplicates Gate/Up only
# while H 5120 decodes M 8, else the streaming kernels (_assert_h5120 follows the H 5120 decode limit
# DECODE_WIDE_MAX_M).
# The aiter.fused_moe entry (test_fused_moe_entry): fused_moe on the tagged w1 / w2 of a prepared weight with
# activation=Silu, quant_type=per_1x32 and quant_type_a=per_Token runs flydsl_moe_a8w4_ptpc (one Qwen decode case).
# DeepSeek-V4.1-Flash layers with the shared expert (test_layer, test_fused_moe_shared_entry; LAYER_SHAPES with their
# full E, the bench's shapes): prepare with the shared expert, routing [M, topk + 1] with the shared expert E in the
# last column and a random weight per token, swiglu_limit 10, inputs scaled by 10 (more than 10% of the shared
# expert's gate and up values exceed the limit), graded against reference_layer, with the routed tier asserted and the
# shared expert's run checked independent of how the op runs it (_assert_layer_path: today the mixed decode kernels
# while M * routed topk * I_pad stays within DECODE_SHARED_MIXED_MAX_WORK (main TP4 up to M 3, TP8 up to M 5), else the
# fold into the routed experts' last reduction; later milestones may change that). One case per routed tier of the
# main layer TP4 (decode M 1, streaming M 64, prefill M 2048), the TP8 I 288 -> 320 padding (decode M 4, and streaming
# M 1024: the folded shared expert's own Down kernel and its mid kernel with 8 lanes per row, today's way from
# SHARED_DOWN_MIN_M_SHORT_K / SHARED_MID_WAVE_ROW_MAX_M rows at I 320) and the MTP
# layer (E 128, topk 3, the official 32x32-block format folded by prepare; the H 5120 decode deduplicating Gate/Up,
# M 8), and fused_moe on the E385 / topk-7 call (M 8, with shared_expert_id=E).
# =================================================================================================

HIDDEN, EXPERTS, TOPK = 2560, 33, 11
MIMO_HIDDEN, MIMO_EXPERTS, MIMO_TOPK = 4096, 256, 8
DS_HIDDEN, DS_EXPERTS, DS_TOPK, DS_INTER = 5120, 24, 6, 576
# The scaled inputs give gate / up a standard deviation of about 10.
SWIGLU_LIMIT, SWIGLU_AMPLIFY = 10.0, 10.0


def _a8w4_check(
    inter,
    tokens,
    seed,
    tier=None,
    hidden=HIDDEN,
    experts=EXPERTS,
    topk=TOPK,
    shared=True,
):
    w, prepared = _weights(experts, hidden, inter, seed)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=seed + 1, shared=shared
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_kernels(tier)
    assert_close(out, ref)
    return prepared


def _ds_weights():
    """DeepSeek-shape weights (E=24), shared by this region, the H5120 cases and the SwiGLU cases of region 2."""
    return _weights(DS_EXPERTS, DS_HIDDEN, DS_INTER, 400)


def _swiglu_over(x, w, topk_ids, limit):
    """[share of gate > L, share of |up| > L] over every route's Gate/Up output at the reference's
    quantization point (x quantized per row)."""
    xa = quant_rows_fp8(x)
    inter = w["w1"].shape[1] // 2
    over = torch.zeros(2, dtype=torch.int64)
    for e in topk_ids.unique().tolist():
        h = (
            xa[(topk_ids == e).any(dim=1)]
            @ dequant_mxfp4(w["w1"][e], w["w1_scale"][e]).T
        )
        over += torch.stack(
            [(h[:, :inter] > limit).sum(), (h[:, inter:].abs() > limit).sum()]
        )
    return (over / (topk_ids.numel() * inter)).tolist()


def _a8w4_swiglu_check(
    w, prepared, tokens, seed, hidden, experts, topk, shared, **options
):
    """A whole call with ``swiglu_limit=SWIGLU_LIMIT`` against the reference with the limit; the inputs are scaled
    by SWIGLU_AMPLIFY and more than 10% of the gate and of the up values must exceed the limit first (the clamp is
    hit often, and without it the error is far beyond the tolerance)."""
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=seed, shared=shared
    )
    x = (x.float() * SWIGLU_AMPLIFY).to(torch.bfloat16)
    over = _swiglu_over(x, w, topk_ids, SWIGLU_LIMIT)
    assert min(over) > 0.1, over
    out = torch.full_like(x, float("nan"))
    result = flydsl_moe_a8w4_ptpc(
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=SWIGLU_LIMIT,
        **options,
    )
    assert result is out
    ref = reference(
        x,
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        topk_weight,
        topk_ids,
        swiglu_limit=SWIGLU_LIMIT,
    )
    assert_close(out, ref)


@pytest.mark.basic
@pytest.mark.parametrize(
    "tokens, tier",
    [(3, "_decode"), (64, "_stream_"), (1031, "_prefill_")],
    ids=["m3", "m64", "m1031"],
)
def test_tp2(tokens, tier):
    _a8w4_check(320, tokens, seed=11, tier=tier)


@pytest.mark.basic
def test_tp4_padded():
    prepared = _a8w4_check(160, 257, seed=21, tier="_prefill_")
    assert (prepared.inter, prepared.inter_padded) == (160, 192)


@pytest.mark.basic
@pytest.mark.parametrize(
    "inter, tokens, tier",
    [
        (256, 5, _DECODE_PATHS["nn"]),
        (256, 300, "_stream_"),
        (512, 24, ["a8w4_stream_prep"] + _stream_path(16, False)),
        (512, 1600, "_gateup_prefill_"),
    ],
    ids=["tp8_m5", "tp8_m300", "tp4_m24", "tp4_m1600"],
)
def test_mimo(inter, tokens, tier):
    _a8w4_check(
        inter,
        tokens,
        seed=41,
        tier=tier,
        hidden=MIMO_HIDDEN,
        experts=MIMO_EXPERTS,
        topk=MIMO_TOPK,
        shared=False,
    )


@pytest.mark.basic
@pytest.mark.parametrize("tokens", [8, 64], ids=["m8", "m64"])
def test_deepseek_swiglu(tokens):
    _a8w4_swiglu_check(
        *_ds_weights(),
        tokens,
        seed=401,
        hidden=DS_HIDDEN,
        experts=DS_EXPERTS,
        topk=DS_TOPK,
        shared=False,
    )
    _assert_kernels("_stream_" if tokens == 64 else None)
    if tokens == 8:
        _assert_h5120(tokens, _DEDUP_PATHS["wide"])


def _fused_moe(x, w1, w2, topk_weight, topk_ids, **options):
    """``aiter.fused_moe`` with the A8W4 combination of its entry: ``activation=Silu``, ``quant_type=per_1x32`` (MXFP4
    weights) and ``quant_type_a=per_Token`` (FP8 activations, one scale per token); ``options`` add or override.
    """
    from aiter.fused_moe import fused_moe

    a8w4 = {
        "activation": aiter.ActivationType.Silu,
        "quant_type": aiter.QuantType.per_1x32,
        "quant_type_a": aiter.QuantType.per_Token,
    }
    return fused_moe(x, w1, w2, topk_weight, topk_ids, **{**a8w4, **options})


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.basic
def test_fused_moe_entry():
    """``aiter.fused_moe`` on the tagged w1 / w2 of a prepared weight runs the kernels of flydsl_moe_a8w4_ptpc with its
    bits, writes the given output and returns it, within the tolerance of the reference (Qwen TP2 M 3, the per-route
    decode, the weights and inputs of test_tp2[m3])."""
    w, prepared = _weights(EXPERTS, HIDDEN, 320, 11)
    x, topk_weight, topk_ids = make_inputs(3, HIDDEN, EXPERTS, TOPK, seed=12)
    want, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    path = list(moe_a8w4.last_path)
    out = torch.full_like(x, float("nan"))
    assert (
        _fused_moe(x, prepared.w1, prepared.w2, topk_weight, topk_ids, output=out)
        is out
    )
    assert moe_a8w4.last_path == path
    _assert_kernels("_decode")
    assert torch.equal(_bits(out), _bits(want))
    assert_close(out, ref)


# (shape, tokens, routed tier, shared-expert format): see the region header; the routed tier of the full-E shape
# (_path_tier, asserted by _assert_layer_path); the MTP case would take the streaming tier past the H 5120 decode
# limit DECODE_WIDE_MAX_M (as _assert_h5120). Only the tier is asserted, not the kernels: how the shared expert runs
# (and so which kernels a layer call launches) is left to later milestones.
_LAYER_CASES = [
    ("ds_layer_tp4", 1, "decode", "row"),
    ("ds_layer_tp4", 64, "stream", "row"),
    ("ds_layer_tp4", 2048, "prefill", "row"),
    ("ds_layer_tp8", 4, "decode", "row"),
    ("ds_layer_tp8", 1024, "stream", "row"),
    ("mtp_layer_tp4", 8, "dedup", "block"),
]


@pytest.mark.basic
@pytest.mark.parametrize(
    "shape, tokens, tier, fmt",
    _LAYER_CASES,
    ids=[f"{shape}_m{m}" for shape, m, _, _ in _LAYER_CASES],
)
def test_layer(shape, tokens, tier, fmt):
    experts, topk, inter = LAYER_SHAPES[shape]
    prepared = _layer_check(
        experts, topk, inter, tokens, seed=700, fmt=fmt, amplify=10.0
    )[0]
    assert (prepared.experts, prepared.inter) == (experts, inter)
    if shape.startswith("mtp") and tokens > moe_a8w4.DECODE_WIDE_MAX_M:
        tier = "stream"  # past the H 5120 decode limit, as in _assert_h5120
    _assert_layer_path(tier)


@pytest.mark.basic
def test_fused_moe_shared_entry():
    """``aiter.fused_moe`` on the tagged w1 / w2 of a weight prepared with the shared expert, called as ATOM calls the
    FP8 layer (E385 / topk 7: the shared expert E in the last column, swiglu_limit 10, here with shared_expert_id=E),
    runs flydsl_moe_a8w4_ptpc with its bits and kernels, within the tolerance of the reference (main layer TP4 M 8).
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    prepared, x, topk_weight, topk_ids, want = _layer_check(
        experts, topk, inter, 8, seed=700, amplify=10.0
    )
    path = list(moe_a8w4.last_path)
    out = torch.full_like(x, float("nan"))
    result = _fused_moe(
        x,
        prepared.w1,
        prepared.w2,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=LAYER_SWIGLU_LIMIT,
        shared_expert_id=experts,
    )
    assert result is out
    assert moe_a8w4.last_path == path
    _assert_shared()
    assert torch.equal(_bits(out), _bits(want))


def _zero_rows(out, rows):
    """Rows ``rows`` of ``out`` hold +0.0 in every element, bit for bit."""
    return bool((_bits(out[list(rows)]) == 0).all())


@pytest.mark.basic
def test_layer_masked_routes():
    """Masked routes (id -1, ATOM's padding) in a layer call, main layer TP4 M 4 (the routed experts on the decode
    tier, the shared expert however the op runs it: _assert_layer_path): a random 30% of the routed ids, every routed id
    of row 1 and the first id of row 2 are -1, with weight NaN, on the ``poisoned`` view (expert -1 is real data).
    Graded against reference_layer: a masked route adds nothing, and a masked first id also drops the row's shared
    expert (ATOM's rule), so row 1 is bitwise 0.
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    *_, topk_ids, out = _layer_check(
        experts,
        topk,
        inter,
        4,
        seed=700,
        amplify=10.0,
        mask={"seed": 711, "full": (1,), "first": (2,)},
    )
    assert int((topk_ids[:, :-1] < 0).sum()) > topk + 1
    assert _zero_rows(out, [1])
    _assert_layer_path("decode")


# =================================================================================================
# Region 2: other coverage (not basic)
# Only what region 1 and the benchmark (region 3) cannot cover; each docstring says why ("the basic tests" are
# region 1). Whole calls are graded as in region 1; bit-exact cases require zero difference. Partially filled
# tail blocks, multi-block experts and the I 160 -> 192 padding are covered by region 1, the E 513 / 256 / 384
# shapes at every token count by the benchmark.
# Cases about the prefill kernels pass a tile option where the default dispatch takes another tier for their M
# (decode for few tokens, streaming while the routes per expert are few) or another default K tile (128 once the
# routes per expert are many); a tile option selects the prefill kernels with the K tile 256 up to M 2048, else 128.
# They assert the Gate/Up and Down kernels (_assert_path); the sorting front end (the streaming prep at every H and M)
# is asserted by the streaming and front-end cases of 2.4.
# Cases about the per-route decode kernels use shapes and token counts of the decode tier (below hidden 4096 while the
# Gate/Up grid stays below DECODE_MAX_WGS workgroups, H 4096 up to DECODE_MAX_M = 7, DeepSeek H 5120 up to M 7) and
# assert the decode kernels by name; the MiMo shapes cut E to 12. The expert-deduplicated Gate/Up (DeepSeek H 5120
# from M 8 on) pins its kernels within the H 5120 decode limit DECODE_WIDE_MAX_M and the streaming kernels past it
# (_assert_h5120).
# The streaming tier (past the decode tier, with at most STREAM_MAX_ROWS* rows per expert) follows
# pyhip M17's cases too: the basic tests stream blocks of 16 and 32, the benchmark 16 / 32 / 64 without asserting
# kernels; here its boundaries with decode and prefill, the bit-exact prep and fused quantization, ACT_STEPS, the
# Gate/Up and Down tiles (bit-exact), the atomic Down's two epilogue branches, rows' table and LDS budget, the prefill
# Down behind the streaming Gate/Up, grid shrinking, the sorting block, CUDA graph, compile-cache keys, repeated ids,
# in-place output and the SwiGLU limit.
# The aiter.fused_moe entry (2.9; its gfx942 gate in 2.6) is covered by the basic test with one decode case and by no
# benchmark case (the benchmark calls flydsl_moe_a8w4_ptpc directly).
# The shared expert (2.10) is covered by the basic layer tests (one case per routed tier, TP8, the MTP layer in the
# 32x32-block format, fused_moe's E385 / topk-7 call) and the benchmark's 95 layer cases (graded); here what they
# cannot show: the bit-exact conversions of prepare (E4M3FN -> FNUZ, -0, the block folding, the I padding), the
# folding of the real checkpoints (byte-identical to FP4PTPC@97cc9b32; needs AITER_MOE_A8W4_DSV41_DIR, else skipped)
# and the accuracy on real weights, input errors and the debug checks, CUDA graph replays, in-place output, and
# fused_moe's shared_expert_id and raw shared tensors. The layer cases check that the shared expert ran and grade the
# output, but not how it ran (_assert_shared / _assert_layer_path): later milestones change where the op runs it. The
# mixed decode (2.12) and the fold (2.13) have their own sections, and so have the launch records behind the
# benchmark's per-kernel roofline lines (2.14).
# =================================================================================================


def _small_stream(block_m=16):
    """The streaming kernels past the decode tier (the prep sorting in blocks of ``block_m``, the intermediate
    quantized by the row kernel)."""
    return ["a8w4_stream_prep"] + _stream_path(block_m, False)


def _mimo_decode_path(tokens, block_m=16):
    """The kernels of a MiMo-shape call (H 4096, topk 8, no shared expert): per route with the Down that quantizes its
    rows up to DECODE_MAX_M = 7 tokens (at every M: the slot-pair Down only below hidden 4096 and at H 5120), past it
    streaming in blocks of ``block_m`` (16 at E 256, 32 for the E 12 cases at M 24)."""
    return (
        _DECODE_PATHS["nn"]
        if tokens <= moe_a8w4.DECODE_MAX_M
        else _small_stream(block_m)
    )


def _small_weights(hidden, inter):
    """E=5 weights of the small-shape cases (cached per H and I)."""
    return _weights(5, hidden, inter, hidden + inter)


# -- 2.1 weight layout and delta ------------------------------------------------------------------


def _pack(nibbles):
    return (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def test_delta_flush_boundary():
    """One 32-K group per row lies 11 (still exact) or 12 (stored as zero by convention) E8M0 codes below the row's
    largest, and the input only covers that group's K.

    The codes of one row of random weights differ by 0-3, so the basic tests and the benchmark never reach the
    delta 11 / 12 boundary; here the E2M1 codes cover all 16 values (-0 included). The output must match the
    reference that zeroes groups with delta > 11 and clearly miss the exact reference (the flush happened).
    """
    E, H, I, topk = 9, 1024, 64, 3
    gen = torch.Generator(device="cuda").manual_seed(41)
    w = make_weights(E, H, I, seed=41)
    rows, groups, top, probe = 2 * I, H // 32, 3, 17
    delta = torch.where(torch.arange(rows) % 2 == 0, 11, 12)
    codes = torch.full((E, rows, groups), 125 - 20, dtype=torch.uint8)
    codes[:, :, top] = 125
    codes[:, :, probe] = (125 - delta).to(torch.uint8)
    nibbles = torch.zeros((E, rows, H), dtype=torch.uint8)
    for g in (top, probe):
        nibbles[:, :, g * 32 : (g + 1) * 32] = torch.randint(
            0, 16, (E, rows, 32), dtype=torch.uint8, generator=gen
        )
    flushed = nibbles.clone()
    flushed[:, delta > 11, probe * 32 : (probe + 1) * 32] = 0
    codes = codes.view(torch.float8_e8m0fnu)
    x, topk_weight, topk_ids = make_inputs(64, H, E, topk, seed=42)
    probe_x = torch.zeros_like(x)
    probe_x[:, probe * 32 : (probe + 1) * 32] = x[:, probe * 32 : (probe + 1) * 32]
    exact = dict(w, w1=_pack(nibbles), w1_scale=codes)
    out, ref_exact = _a8w4_run(
        exact, _a8w4_prepare(exact), probe_x, topk_weight, topk_ids
    )
    ref_flushed = reference(
        probe_x, _pack(flushed), codes, w["w2"], w["w2_scale"], topk_weight, topk_ids
    )
    assert_close(out, ref_flushed)
    assert not check(out, ref_exact)["ok"]


def _decode_layout(q, d, s, N, K):
    """(q, d, s) decoded by the byte layout of the ``common_a8w4`` docstring -> float64 [E, N, K] and the deltas."""
    E, KB = s.shape[0], K // 64
    KQ = (KB + 3) // 4
    b = (
        q.view(E, N // 16, KB, 4, 16, 2, 4)
        .permute(0, 1, 4, 2, 3, 5, 6)
        .reshape(E, N, KB, 4, 2, 4)
    )
    nib = torch.cat([b & 0xF, b >> 4], dim=-1).reshape(E, N, K).long()
    table = torch.tensor(_E2M1, dtype=torch.float64)
    dl = (
        d.view(E, N // 16, KQ, 2, 16, 4)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(E, N, KQ * 4, 2)[:, :, :KB]
    )
    dl = dl.reshape(E, N, KB * 2).long()
    assert bool((dl % 8 == 0).all())
    factor = torch.exp2(5 - (dl // 8).double()).repeat_interleave(32, dim=-1)
    return table[nib] * factor * s.view(E, N, 1).double(), dl // 8


def test_prepare_layout_bitexact():
    """The prepared weights, decoded by the byte layout of the ``common_a8w4`` docstring, equal the exact MXFP4
    values element by element (delta 0-15, -0, the I 160 -> 192 padding, the channel factor).

    The basic tests only grade end-to-end errors: a layout that departs from the documented one but is read the
    same way by the kernels would still pass, and later kernels (and the dense A8W4 GEMM) rely on that layout.
    """
    E, H, I = 2, 512, 160
    gen = torch.Generator(device="cuda").manual_seed(71)

    def weight(n, k):
        codes = (120 - torch.randint(0, 16, (E, n, k // 32), generator=gen)).to(
            torch.uint8
        )
        codes[:, :, 0] = 120
        nibbles = torch.randint(0, 16, (E, n, k), dtype=torch.uint8, generator=gen)
        return _pack(nibbles), codes.view(torch.float8_e8m0fnu)

    w1, s1 = weight(2 * I, H)
    w2, s2 = weight(H, I)
    p = prepare_moe_a8w4_ptpc_weight(w1, s1, w2, s2)
    assert (p.inter, p.inter_padded) == (I, 192)

    def expected(q, s):
        codes = s.view(torch.uint8).long()
        value = dequant_mxfp4(q, s).double()
        diff = codes.amax(-1, keepdim=True) - codes
        value = value.masked_fill((diff > 11).repeat_interleave(32, dim=-1), 0.0)
        scale = torch.exp2(codes.amax(-1).double() - 132).float()
        return value, diff.clamp(max=11), scale

    v1, d1, c1 = expected(w1, s1)
    z1 = torch.zeros((E, 192 - I, H), dtype=torch.float64)
    got, delta = _decode_layout(p.w1, p.w1_delta, p.w1_scale, 2 * 192, H)
    assert torch.equal(got, torch.cat([v1[:, :I], z1, v1[:, I:], z1], dim=1))
    assert torch.equal(delta[:, :I], d1[:, :I]) and torch.equal(
        delta[:, 192 : 192 + I], d1[:, I:]
    )
    assert torch.equal(p.w1_scale.view(E, -1)[:, :I], c1[:, :I])

    v2, d2, c2 = expected(w2, s2)
    got, delta = _decode_layout(p.w2, p.w2_delta, p.w2_scale, H, 192)
    assert torch.equal(got, torch.nn.functional.pad(v2, (0, 192 - I)))
    assert torch.equal(delta[:, :, : I // 32], d2)
    assert torch.equal(p.w2_scale.view(E, -1), c2)


# -- 2.2 row quantization (prefill, decode, dedup plan, streaming prep and Gate/Up epilogue), bit-exact ---------


@pytest.mark.parametrize("dtype, cols", [("bf16", 2560), ("f32", 320)])
def test_row_quant_bitexact(dtype, cols):
    """The activation row quantization (scale = amax / 240, q = RNE(x / scale)) equals the reference bit for bit,
    zero rows and the exact FP8 midpoints common in BF16 included; the kernel computes the quotient as a reciprocal
    product with two FMA corrections and divides per element when scale < 2**-64, plus two rows whose FP8 values
    a plain reciprocal product would change.

    The basic tests only grade end-to-end errors: a slightly different quantization (a plain reciprocal product)
    flips a few FP8 values, which the tolerance may not notice at small M, yet it eats into the 4.5% margin at
    large M; random data hits few such elements, hence the two constructed rows.
    """
    gen = torch.Generator(device="cuda").manual_seed(81)
    x = torch.randn((997, cols), generator=gen) * 3
    x[5].zero_()
    # Row 6: scale = 52.5 / 240 = 7/32 (not a power of two), the other quotients exactly FP8 rounding midpoints
    # (mantissa 23/16 ... 31/16): a reciprocal product is one f32 ulp larger, past the midpoint, and rounds the
    # other way. Row 7 is row 6 times 2**-70: scale < 2**-64 takes the per-element division.
    tie = (
        torch.arange(23, 32, 2).float().view(-1, 1)
        / 16
        * torch.exp2(torch.arange(-6, 7).float())
    )
    tie = torch.cat([tie.flatten(), -tie.flatten()]) * (7 / 32)
    x[6] = tie.repeat(cols // tie.numel() + 1)[:cols]
    x[6, 0] = 52.5
    x[7] = x[6] * 2.0**-70
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q, scale = moe_a8w4._quantize(x, x.shape[0], cols, dtype)
    ref = quant_rows_fp8(x)
    amax = x.float().abs().amax(-1, keepdim=True)
    ref_scale = torch.where(
        amax == 0, torch.ones_like(amax), amax / torch.full_like(amax, 240.0)
    )
    assert torch.equal(scale.view(-1, 1), ref_scale)
    assert torch.equal(q.float() * scale.view(-1, 1), ref)
    assert int((q.view(torch.uint8) == 0x80).sum()) == 0
    exact = (x.float() / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    naive = (
        (x.float() * (torch.ones_like(ref_scale) / ref_scale))
        .to(torch.float8_e4m3fnuz)
        .view(torch.uint8)
    )
    assert bool((exact[6] != naive[6]).any()) and bool((exact[7] != naive[7]).any())
    assert float(scale[7]) < 2.0**-64 <= float(scale[6])


@pytest.mark.parametrize(
    "dtype, cols",
    [
        ("f32", 192),
        ("f32", 320),
        ("f32", 640),
        ("bf16", 1024),
        ("bf16", 2560),
        ("bf16", 4096),
    ],
)
def test_decode_row_quant_bitexact(dtype, cols):
    """The decode path's row quantization kernel (one wave per row, 16-byte pieces) equals the prefill path's
    (``_quantize``) and the reference bit for bit (q bytes and scale bits), and writes no row past ``rows``.

    Special rows: all zero, amax / 240 underflowing to 0 (scale 1; only f32 holds such an input), a subnormal scale,
    -0 and negatives rounding to -0 (stored as +0). f32 192 / 320 / 640 columns have 48 / 80 / 160 pieces per row
    (the last round of 64 lanes not full), bf16 1024 / 2560 / 4096 columns 128 / 320 / 512; 37 rows (the last
    workgroup has one row, the 3 rows after it must stay unwritten). End to end the tolerance does not show a few
    flipped FP8 values, and the basic tests and the benchmark only quantize random rows (f32 192 / 256 / 320 / 512 /
    576 and bf16 2560 / 4096 / 5120 columns).
    """
    rows, guard = 37, 3
    gen = torch.Generator(device="cuda").manual_seed(83)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[2].zero_()
    x[2, 7] = 1e-45
    x[3] *= 1e-38
    x[4].zero_()
    x[4, 0] = 200.0
    x[4, 1:9] = -1e-6
    x[4, 9] = -0.0
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    moe_a8w4._launch(
        "a8w4_row_quant_decode",
        moe_a8w4._row_quant_decode(cols, dtype == "bf16", x.device.index),
        x,
        q.view(torch.float8_e4m3fnuz),
        scale,
        rows,
    )
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, dtype)
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert torch.equal(scale[:rows].view(-1, 1), ref_scale)
    assert torch.equal(
        q[:rows], (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    )


@pytest.mark.parametrize(
    "rows, max_m, masked",
    [(29, 32, False), (29, 32, True), (None, None, False)],
    ids=["h5120_segments", "h5120_masked", "h5120_dispatch_max_m"],
)
def test_decode_plan_quant_bitexact(rows, max_m, masked):
    """The plan kernel of the H 5120 deduplicated Gate/Up (a8w4_row_quant_plan_decode: four waves quantize a row of x
    in 16-byte pieces, the amax merged in LDS, and the same launch writes the dedup plan; its launcher takes no
    output): xq / xs equal ``_quantize`` and the reference bit for bit, nothing past the rows is written, and the plan
    entries equal the definition (an expert's routes in route order, groups of 2: the leader holds the count and the
    member, every other route the count 0; words past the count are not read and not checked).

    Special rows as in test_decode_row_quant_bitexact (zero, a subnormal scale, -0 and negatives rounding to -0); a
    row has 640 16-byte pieces (256 threads: the third round only the first 128). 29 rows x topk 6 = 174 routes on 40
    experts planned for M <= 32 (3 ballot segments of 64 routes), so members across segments and partial tail groups
    both occur; the dispatch's own build (MAX_M = DECODE_WIDE_MAX_M, a single segment) at that many rows. End to end
    the tolerance does not show a few flipped FP8 values; a wrong plan mostly fails end to end, but random routing may
    miss the cross-segment and partial-tail combinations.

    With masked routes (id -1, a random 30% and rows 2 and 7 whole) a masked route gets the entry [0, route] and the
    groups list valid routes only: its ballots also match the -1 that pads the plan past the routes, so it must
    lead no group. End to end a masked leader would show only through the result of its group (test_masked_routes
    runs the H 5120 dedup path on poisoned weights); this pins every entry.
    """
    if max_m is None:
        rows = max_m = moe_a8w4.DECODE_WIDE_MAX_M
    cols, topk, guard, group = DS_HIDDEN, DS_TOPK, 3, 2
    gen = torch.Generator(device="cuda").manual_seed(87)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[3] *= 1e-38
    x[4].zero_()
    x[4, 0] = 200.0
    x[4, 1:9] = -1e-6
    x[4, 9] = -0.0
    x = x.to(torch.bfloat16)
    _, _, topk_ids = make_inputs(rows, 512, 40, topk, seed=88, shared=False)
    if masked:
        _, topk_ids = mask_routes(
            torch.zeros(topk_ids.shape), topk_ids, seed=89, full=(2, 7)
        )
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    plan = torch.full((rows * topk, 4), -1, dtype=torch.int32)
    moe_a8w4._launch(
        "a8w4_row_quant_plan_decode",
        moe_a8w4._row_quant_plan(cols, topk, max_m, x.device.index),
        x,
        q.view(torch.float8_e4m3fnuz),
        scale,
        topk_ids,
        plan,
        rows,
    )
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[3]) < 2.0**-126
    assert torch.equal(
        q[:rows], (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    )
    ids, entries = topk_ids.flatten().tolist(), plan.tolist()
    assert masked == (min(ids) < 0)
    full = partial = 0
    for r, e in enumerate(ids):
        if e < 0:
            assert entries[r][:2] == [0, r], (r, entries[r])
            continue
        same = [s for s, f in enumerate(ids) if f == e]
        rank = same.index(r)
        count = min(group, len(same) - rank) if rank % group == 0 else 0
        entry = entries[r]
        assert entry[:2] == [count, r], (r, entry)
        assert entry[1 : 1 + count] == same[rank : rank + count], (r, entry)
        full += count == group
        partial += 0 < count < group
    assert full and partial


def _fast_quant_rows(cols):
    """Rows for the fast row quantization of the decode kernels (decode_a8w4._quant_fast), float32 [12, cols], each
    with its exponent k (the products of a row times 2**k are about those of a random row; 0 for the zero row,
    |k| <= 120). The abs-max of a row is set exactly (and stays so in BF16): random rows with abs-max 40; 2**-56
    (fast), 2**-60 (the smallest fast one), the BF16 value just below it, 2**-68 and 2**-130 (subnormal values; all
    three divide); all zero (fast); the BF16 value just below 2**127 (fast) and 2**127 (divides). Tie rows (9 to 11):
    s times every positive E4M3FNUZ value and every midpoint between neighbours (and 0), both signs, for s = 7,
    7 * 2**-70 (abs-max 2**-59.3) and 7 * 2**115 (2**125.7), exact in BF16: the abs-max 240 s gives the scale s
    exactly, so every quotient is an FP8 value or a tie of round to nearest even, and the bare product with RN(1 / 7)
    rounds 62 of the 508 values the other way.
    """
    gen = torch.Generator(device="cuda").manual_seed(798)
    unit = torch.randn((8, cols), generator=gen)
    unit = (unit / unit.abs().amax(dim=1, keepdim=True)).to(torch.bfloat16).float()
    below = 1.0 - 2.0**-8
    grid = torch.arange(1, 128, dtype=torch.uint8).view(torch.float8_e4m3fnuz).float()
    ties = torch.cat([grid, (torch.cat([grid.new_zeros(1), grid[:-1]]) + grid) / 2])
    ties = torch.cat([ties, -ties]).repeat(cols // 508 + 1)[:cols]
    rows = torch.stack(
        [
            unit[0] * 40.0,
            unit[1] * 2.0**-56,
            unit[2] * 2.0**-60,
            unit[3] * (below * 2.0**-60),
            unit[4] * 2.0**-68,
            unit[5] * 2.0**-130,
            torch.zeros(cols),
            unit[6] * (below * 2.0**127),
            unit[7] * 2.0**127,
            ties * 7.0,
            ties * (7.0 * 2.0**-70),
            ties * (7.0 * 2.0**115),
        ]
    )
    amax = rows.abs().amax(dim=1).tolist()
    exps = [
        0 if a == 0 else max(-120, min(120, round(math.log2(40.0 / a)))) for a in amax
    ]
    return rows, exps


def _ieee_quant(x):
    """The reference row quantization by IEEE division: (q FP8 E4M3FNUZ, scale f32 [rows, 1]), scale = amax / 240
    (1 where that is 0), q = RNE(x / scale)."""
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    scale = amax / torch.full_like(amax, 240.0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (xf / scale).to(torch.float8_e4m3fnuz), scale


@pytest.mark.parametrize(
    "kernel", ["decode_bf16", "decode_f32", "plan_bf16", "prep_bf16"]
)
def test_fast_row_quant_bitexact(kernel):
    """The row-quantization kernels that divide once per row (the reciprocal) and correct each product with FMAs
    (decode_a8w4._quant_fast: one correction for BF16 rows, two for f32; rows outside _fast_div_ok divide per value):
    their FP8 bytes and scale bits equal the IEEE division's (_ieee_quant) on _fast_quant_rows, i.e. on both sides of
    both ends of the fast range (2**-60, 2**127), on zero and subnormal rows and on rounding ties, where a bare
    reciprocal product rounds values of every tie row the other way. a8w4_row_quant_decode on BF16 x (H 5120) and on
    f32 intermediate rows (I 576), a8w4_row_quant_plan_decode on BF16 x (H 5120, topk 6; its plan is
    test_decode_plan_quant_bitexact's) and the streaming prep (a8w4_stream_prep, one row per wave) on BF16 x (H 5120;
    its sort is test_stream_row_quant_bitexact's).

    test_decode_row_quant_bitexact, test_decode_plan_quant_bitexact and test_stream_row_quant_bitexact use random
    rows with zero, subnormal and -0 rows, none at the ends of the fast range; end to end a few flipped FP8 values
    stay within the tolerance.
    """
    cols = DS_INTER if kernel == "decode_f32" else DS_HIDDEN
    rows, _ = _fast_quant_rows(cols)
    x = rows if kernel == "decode_f32" else rows.to(torch.bfloat16)
    n = x.shape[0]
    q = torch.full((n, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((n,), -1.0)
    if kernel == "prep_bf16":
        _, topk_weight, topk_ids = make_inputs(
            n, 512, 40, DS_TOPK, seed=786, shared=False
        )
        *_, q, scale = _prep_launch(x, topk_weight, topk_ids, 40, 16, True)
    elif kernel == "plan_bf16":
        _, _, topk_ids = make_inputs(n, 512, 40, DS_TOPK, seed=785, shared=False)
        moe_a8w4._launch(
            "a8w4_row_quant_plan_decode",
            moe_a8w4._row_quant_plan(cols, DS_TOPK, 32, x.device.index),
            x,
            q.view(torch.float8_e4m3fnuz),
            scale,
            topk_ids,
            torch.full((n * DS_TOPK, 4), -1, dtype=torch.int32),
            n,
        )
    else:
        moe_a8w4._launch(
            "a8w4_row_quant_decode",
            moe_a8w4._row_quant_decode(cols, kernel == "decode_bf16", x.device.index),
            x,
            q.view(torch.float8_e4m3fnuz),
            scale,
            n,
        )
    want, want_scale = _ieee_quant(x)
    assert torch.equal(scale.view(torch.int32), want_scale.view(-1).view(torch.int32))
    assert torch.equal(q, want.view(torch.uint8))
    naive = (x.float() * (1.0 / want_scale)).to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert bool((naive[9:] != q[9:]).any(dim=1).all())


def _prep_launch(x, topk_weight, topk_ids, experts, block_m, zero, shared=False):
    """One a8w4_stream_prep launch on BF16 ``x`` [M, H] (``_stream_prep_kernel`` with the sizes of
    ``stream_prep_sizes``, as ``moe_a8w4._stream_prep`` passes them; ``shared``: the last column is a folded shared
    expert's) into sentinel-filled buffers with guard elements past their ends, then a synchronization. Asserts that
    nothing past the outputs is written and that out is zeroed (``zero``) or not written at all. Returns
    ``(sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, q, scale)``, q as bytes [M, H].
    """
    M, H = x.shape
    topk = topk_ids.shape[1]
    n_sort, rows, blocks = stream_a8w4.stream_prep_sizes(M, topk, experts, block_m)
    if shared:
        _, rows, blocks = stream_a8w4.stream_prep_sizes(M, topk - 1, experts, block_m)
    guard = 64
    sorted_ids = torch.full((rows + guard,), -7, dtype=torch.int32)
    sorted_weights = torch.full((rows + guard,), -7.0)
    sorted_experts = torch.full((blocks + guard,), -7, dtype=torch.int32)
    num_valid = torch.full((2 + guard,), -7, dtype=torch.int32)
    loc_ids = torch.full((M * topk + guard,), -7, dtype=torch.int32)
    q = torch.full((M + 2, H), 0x5A, dtype=torch.uint8)
    scale = torch.full((M + 2,), -1.0)
    out = torch.full((M + 2, H), float("nan"), dtype=torch.bfloat16)
    kernel = moe_a8w4._stream_prep_kernel(
        H,
        topk,
        experts,
        block_m,
        zero,
        x.device.index,
        **({"shared": True} if shared else {}),
    )
    outputs = (
        sorted_ids[:rows],
        sorted_weights[:rows],
        sorted_experts[:blocks],
        num_valid[:2],
        loc_ids[: M * topk].view(M, topk),
        q[:M],
        scale[:M],
    )
    moe_a8w4._launch(
        "a8w4_stream_prep",
        kernel,
        topk_ids,
        topk_weight,
        x,
        *outputs[:5],
        q[:M].view(torch.float8_e4m3fnuz),
        scale[:M],
        out[:M],
        M,
        n_sort,
        rows,
    )
    torch.cuda.synchronize()
    for buf, end, sentinel in (
        (sorted_ids, rows, -7),
        (sorted_weights, rows, -7.0),
        (sorted_experts, blocks, -7),
        (num_valid, 2, -7),
        (loc_ids, M * topk, -7),
        (q, M, 0x5A),
        (scale, M, -1.0),
    ):
        assert bool((buf[end:] == sentinel).all()), sentinel
    assert bool(out[M:].isnan().all())
    if zero:
        assert bool((out[:M] == 0).all())
    else:
        assert bool(out[:M].isnan().all())
    return outputs


def _assert_prep_sort(sort, topk_weight, topk_ids, experts, block_m, shared=False):
    """The prep's ``sort`` (sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids) in aiter moe_sorting's
    format: the routes with an id in [0, E) (with ``shared`` not the last column, whatever its id) grouped by expert
    in ascending order, each expert padded to whole blocks with rows of token M and weight 0; the blocks' experts;
    num_valid [padded rows, M]; each such route's loc_ids its own sorted row (unique, within its expert's rows,
    holding ``token | slot << 24`` and the bits of its weight), every other route's -1.
    """
    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids = sort
    M, topk = topk_ids.shape
    live = (topk_ids >= 0) & (topk_ids < experts)
    if shared:
        live[:, -1] = False
    ids = topk_ids[live].long()
    counts = torch.bincount(ids, minlength=experts)
    padded = (counts + block_m - 1) // block_m * block_m
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum())
    assert num_valid.tolist() == [total, M]
    assert torch.equal(
        sorted_experts[: total // block_m],
        torch.repeat_interleave(
            torch.arange(experts, dtype=torch.int32), padded // block_m
        ),
    )
    assert bool((loc_ids[~live] == -1).all())
    loc = loc_ids[live].long()
    assert torch.unique(loc).numel() == loc.numel()
    assert bool(((loc >= offsets[ids]) & (loc < offsets[ids] + counts[ids])).all())
    route = torch.arange(M * topk, dtype=torch.int32).view(M, topk)[live]
    assert torch.equal(sorted_ids[loc], (route // topk) | ((route % topk) << 24))
    assert torch.equal(
        sorted_weights[loc].view(torch.int32), topk_weight[live].view(torch.int32)
    )
    filler = torch.ones(total, dtype=torch.bool)
    filler[loc] = False
    assert bool((sorted_ids[:total][filler] == M).all())
    assert bool((sorted_weights[:total][filler] == 0).all())


@pytest.mark.parametrize("cols", [512, 2560, 4096], ids=["h512", "h2560", "h4096"])
def test_stream_row_quant_bitexact(cols):
    """The streaming prep (a8w4_stream_prep: one launch sorts the routes, quantizes x and zeroes out): q and scale
    equal ``_quantize`` and the reference bit for bit, the first ``rows`` rows of out are zero and nothing past the
    outputs is written (_prep_launch); the sort has aiter moe_sorting's format (_assert_prep_sort).

    The quantization takes a row per wave, each lane 16-byte pieces of every 512-column chunk (H 512: 1 chunk, H 2560:
    5, H 4096: 8), one reciprocal per row and one FMA correction per value, the whole wave dividing per value for a
    row outside the fast range (whose ends test_fast_row_quant_bitexact covers). Special rows: zero, a subnormal
    scale, -0 and negatives rounding to -0 (stored as +0), quotients exactly at rounding midpoints with the scale
    2**-2, and midpoints that a plain reciprocal product rounds the wrong way with the scale 7/32; 37 rows (the last
    quantizing workgroup holds one row). E 513 and topk 11: 2 sorting workgroups, 3 experts per thread.

    End to end the tolerance does not show a few flipped FP8 values, and the basic tests and the benchmark only
    quantize random rows and never look at the sort format.
    """
    rows, experts, block_m = 37, 513, 16
    gen = torch.Generator(device="cuda").manual_seed(85)
    x = torch.randn((rows, cols), generator=gen) * 3
    x[1].zero_()
    x[2] *= 1e-37
    x[3].zero_()
    x[3, 0] = 200.0
    x[3, 1:9] = -1e-6
    x[3, 9] = -0.0
    # Row 4: amax = 240 * 2**-2, so the scale is exactly 2**-2 and the other quotients (k + 0.5) * 2**e lie exactly
    # halfway between two FP8 values.
    exponent = torch.randint(-10, 1, (cols,), generator=gen).float()
    tie = (torch.randint(8, 16, (cols,), generator=gen).float() + 0.5) * torch.exp2(
        exponent
    )
    x[4] = torch.where(torch.rand((cols,), generator=gen) < 0.5, -tie, tie) * 0.25
    x[4, 0] = 60.0
    # Row 5: scale = 52.5 / 240 = 7/32 and the other quotients exactly at rounding midpoints, which a plain
    # reciprocal product rounds the wrong way (as in test_row_quant_bitexact).
    tie7 = (
        torch.arange(23, 32, 2).float().view(-1, 1)
        / 16
        * torch.exp2(torch.arange(-6, 7).float())
    )
    tie7 = torch.cat([tie7.flatten(), -tie7.flatten()]) * (7 / 32)
    x[5] = tie7.repeat(cols // tie7.numel() + 1)[:cols]
    x[5, 0] = 52.5
    x = x.to(torch.bfloat16)
    _, topk_weight, topk_ids = make_inputs(rows, 512, experts, TOPK, seed=86)
    *sort, q, scale = _prep_launch(x, topk_weight, topk_ids, experts, block_m, True)
    _assert_prep_sort(sort, topk_weight, topk_ids, experts, block_m)
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q, q0.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[2]) < 2.0**-126
    assert float(ref_scale[4]) == 0.25 and float(ref_scale[5]) == 7 / 32
    assert torch.equal(scale.view(-1, 1), ref_scale)
    exact = (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    naive = (
        (xf * (torch.ones_like(ref_scale) / ref_scale))
        .to(torch.float8_e4m3fnuz)
        .view(torch.uint8)
    )
    assert torch.equal(q, exact) and bool((exact[5] != naive[5]).any())


@pytest.mark.parametrize(
    "inter, block_m", [(320, 32), (160, 64)], ids=["tp2_bm32", "tp4_bm64"]
)
def test_stream_fused_quant_bitexact(inter, block_m):
    """The streaming Gate/Up that quantizes the intermediate in its epilogue (a reciprocal product checked at
    x(1 -+ 2**-20); an element near a rounding boundary or a scale that is no finite normal number makes the whole
    wave fall back to IEEE division) gives the same q bytes and scale bits as the same kernel writing f32 and then
    ``_quantize``, on one sort. TP2 with blocks of 32 (the dispatch's tile: 5 gate blocks per wave, activations not
    shared); TP4 (I_pad 192) with blocks of 64 (3 gate blocks, activations shared in LDS, up to 4 sub-blocks; the
    dispatch's tile for TP4 past 28 rows per expert, E 513 M >= 1306).

    The random rows of M 1031 hold quotients within 2**-20 of a rounding boundary (asserted below: the fallback
    runs); 6 tokens scaled by 1e-17 ... 1e-23 or zeroed give intermediate rows with a small normal, a subnormal and
    a zero scale (taken as 1). End to end the tolerance does not show a few flipped FP8 values.
    """
    from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting

    _, prepared = _decode_weights(inter)
    E, I, M = prepared.experts, prepared.inter_padded, 1031
    assert moe_a8w4._stream_fused_quant(I, block_m)
    x, topk_weight, topk_ids = make_inputs(M, HIDDEN, E, TOPK, seed=120)
    for row, factor in enumerate((1e-17, 1e-18, 3e-19, 1e-20, 1e-23, 0.0)):
        x[row] = (x[row].float() * factor).to(torch.bfloat16)
    sorted_ids, _, sorted_experts, num_valid = moe_sorting(
        topk_ids,
        topk_weight,
        E,
        HIDDEN,
        x.dtype,
        block_m,
        accumulate=False,
        output_aux=AUX_SORT_OPUS,
    )[:4]
    routing, task_num = (sorted_ids, sorted_experts, num_valid), sorted_experts.numel()
    xq, xs = moe_a8w4._quantize(x, M, HIDDEN, "bf16")
    mq = torch.empty((M * TOPK, I), dtype=torch.float8_e4m3fnuz)
    ms = torch.empty((M * TOPK,))
    fused = moe_a8w4._stream_gateup_kernel(
        2 * I,
        HIDDEN,
        TOPK,
        block_m,
        E,
        0.0,
        x.device.index,
        moe_a8w4._stream_gateup_constants(),
    )
    moe_a8w4._launch(
        "fused",
        fused,
        xq,
        prepared.w1,
        prepared.w1_delta,
        mq,
        *routing,
        prepared.w1_scale,
        xs,
        ms,
        M,
        task_num,
    )
    plain = stream_a8w4.compile_a8w4_gateup_stream(
        N=2 * I, K=HIDDEN, TOPK=TOPK, BLOCK_M=block_m, E=E, G=4, W=1, PREFETCH=(6, 4)
    )
    mid = torch.empty((M, TOPK, I))
    moe_a8w4._launch(
        "plain",
        plain,
        xq,
        prepared.w1,
        prepared.w1_delta,
        mid,
        *routing,
        prepared.w1_scale,
        xs,
        xs,
        M,
        task_num,
    )
    q0, s0 = moe_a8w4._quantize(mid, M * TOPK, I, "f32")
    assert torch.equal(mq.view(torch.uint8), q0.view(torch.uint8))
    assert torch.equal(ms.view(torch.int32), s0.view(torch.int32))
    assert bool(((s0 > 0) & (s0 < 2.0**-126)).any())
    quotient = mid.view(M * TOPK, I) / s0.view(-1, 1)
    ends = [
        (quotient * (1.0 + d))
        .clamp(-240.0, 240.0)
        .to(torch.float8_e4m3fnuz)
        .view(torch.uint8)
        for d in (-(2.0**-20), 2.0**-20)
    ]
    assert bool((ends[0] != ends[1]).any())


# -- 2.3 decode dispatch: tier boundaries, topk and routing variants, the w2 >= 4 GiB fallback, deduplication ----


def _decode_weights(inter):
    """E 33 Qwen-shape weights (H 2560, topk 11) of the decode and small streaming cases, cached per I (I 320 shared
    with the graph and in-place cases)."""
    return _weights(EXPERTS, HIDDEN, inter, 51)


@pytest.mark.parametrize(
    "inter, tokens, path",
    [
        (320, 4, "np"),
        (320, 11, "np"),
        (320, 12, "stream"),
        (160, 19, "np"),
        (160, 20, "stream"),
    ],
    ids=["tp2_m4", "tp2_m11", "tp2_m12", "tp4_m19", "tp4_m20"],
)
def test_decode_dispatch_boundaries(inter, tokens, path):
    """Both sides of each decode dispatch boundary on Qwen shapes (E 33), end to end with the Gate/Up and Down
    kernels: M = 4 (Down takes the slot-pair kernel on rows quantized by the row kernel; M = 3 is a basic case), and
    the decode limit below hidden 4096, the per-route Gate/Up grid M * topk * I_pad / 16 < DECODE_MAX_WGS = 2560: TP2
    M = 11 / 12 (2420 / 2640 workgroups) and TP4 M = 19 / 20 (2508 / 2640) decode / stream (the prep sorts in blocks of
    16 at 4-7 rows per expert with E 33).

    The basic tests decode Qwen M = 3 and MiMo M = 5, on none of these boundaries; the benchmark runs powers of two
    and 48 (no 11 / 12 or 19 / 20) and asserts no kernels: M = 4 runs the benchmark's path and stays as one side of its
    boundary.
    """
    i_pad = -(-inter // 64) * 64
    assert (tokens * TOPK * i_pad // 16 < moe_a8w4.DECODE_MAX_WGS) == (path != "stream")
    w, prepared = _decode_weights(inter)
    x, topk_weight, topk_ids = make_inputs(
        tokens, HIDDEN, EXPERTS, TOPK, seed=90 + tokens
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "stream":
        _assert_path(_small_stream())
        assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path
    else:
        _assert_path(_DECODE_PATHS[path])
    assert_close(out, ref)


def test_decode_wgs_limit_small_hidden():
    """The decode limit below hidden 4096 counts Gate/Up workgroups, not tokens: a small shape (H 512, I 64, E 5,
    topk 2: 8 workgroups per token) decodes up to M 319 (2552 < DECODE_MAX_WGS = 2560) and takes the prefill kernels
    at M 320 (128 rows per expert, past the streaming limit), end to end with the Gate/Up and Down kernels.

    The basic tests and the benchmark use the target shapes, whose limit lies at M 11 / 19 (Qwen TP2 / TP4,
    test_decode_dispatch_boundaries); only a small topk * I reaches the decode tier at hundreds of tokens.
    """
    w, prepared = _small_weights(512, 64)
    for tokens in (319, 320):
        x, topk_weight, topk_ids = make_inputs(
            tokens, 512, 5, 2, seed=92 + tokens, shared=False
        )
        out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
        if tokens * 2 * (64 // 16) < moe_a8w4.DECODE_MAX_WGS:
            _assert_path(_DECODE_PATHS["np"])
        else:
            _assert_kernels("_prefill_")
        assert_close(out, ref)


@pytest.mark.parametrize(
    "topk, tokens, path",
    [(8, 15, "np"), (17, 2, "np"), (1, 2, "nn"), (16, 3, "nn")],
    ids=["topk8_m15", "topk17_m2", "topk1_m2", "topk16_m3"],
)
def test_decode_topk_variants(topk, tokens, path):
    """topk other than 11: with topk = 8 (even) the slot-pair Down's 3 waves take two rounds and both slots of the
    second round's last two waves are invalid (M = 15, 2400 Gate/Up workgroups: the last token count below
    DECODE_MAX_WGS); topk = 17 is past the 16 columns of the Down that quantizes its rows (one wave per column), so
    the slot-pair Down takes it below DECODE_PREQUANT_MIN_M too (M 2: 9 slot pairs in 3 rounds of 3 waves, the last
    pair's upper slot invalid); the Down that quantizes its rows at both ends of its column count: topk 1 (one wave,
    nothing to add across waves; one random expert per token) and topk 16 (16 waves, 1024 threads).

    The Qwen cases of the basic tests and the benchmark have topk = 11 (only the last slot pair lacks its upper
    half), the others 6 or 8; the benchmark's MiMo cases (topk 8) decode up to M 7 with the Down that quantizes its
    rows, never the slot-pair Down; no case of either has topk 1 or topk > 11 (test_mixed_decode_shapes runs that Down
    with the shared expert at 2 and 16 columns).
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(
        tokens, HIDDEN, EXPERTS, topk, seed=94 + topk, shared=topk > 1
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(_DECODE_PATHS[path])
    assert_close(out, ref)


def test_decode_duplicate_and_int64_ids():
    """Several slots of a token select the same expert (both halves of a slot pair included), with int64,
    non-contiguous topk_ids and a non-contiguous topk_weight (M = 5): the entry converts them to contiguous int32 /
    float32, and the decode path computes every slot (Gate/Up per route, Down summing the token's slots), as the
    reference does.

    The generated routes never repeat an expert within a token and are contiguous int32 / float32, so the basic tests
    and the benchmark never get here; test_routing_input_dtypes converts on the prefill path.
    """
    w, prepared = _decode_weights(320)
    tokens = 5
    x, topk_weight, topk_ids = make_inputs(tokens, HIDDEN, EXPERTS, TOPK, seed=96)
    topk_ids[:, 1] = topk_ids[:, 0]
    topk_ids[:, 4] = topk_ids[:, 0]
    ids = torch.empty((tokens, 2 * TOPK), dtype=torch.int64)[:, ::2]
    ids.copy_(topk_ids)
    weights = torch.empty((tokens, 2 * TOPK))[:, 1::2]
    weights.copy_(topk_weight)
    assert not ids.is_contiguous() and not weights.is_contiguous()
    out, ref = _a8w4_run(w, prepared, x, weights, ids)
    _assert_path(_DECODE_PATHS["np"])
    assert_close(out, ref)


@pytest.mark.parametrize(
    "over, down",
    [(1, "a8w4_down_decode"), (0, "a8w4_down_decode_pq")],
    ids=["over", "fits"],
)
def test_decode_down_fallback_w2_4gib(over, down):
    """Down takes the slot-pair kernel on pre-quantized rows only while w2 plus one expert's bytes fit in 4 GiB (it
    addresses all experts' weights with 32-bit byte offsets, its range check ending one expert below 2**32, where a
    masked route's offsets fall); larger w2 takes the Down that quantizes its rows (M = 8, past
    DECODE_PREQUANT_MIN_M). Both sides of that boundary: w2 of 2**32 bytes minus one expert's (``fits``) and one byte
    more (``over``, still below 4 GiB).

    Real 4 GiB weights are too large: a stride-0 view makes w2 report that many bytes with the same data pointer (the
    kernels read by pointer and expert offset, the routes only real experts); the basic tests, the benchmark and the
    other cases have w2 far below 4 GiB.
    """
    import dataclasses

    w, prepared = _decode_weights(320)
    size = (1 << 32) - prepared.w2[0].numel() * prepared.w2.element_size() + over
    big = dataclasses.replace(prepared, w2=prepared.w2.view(-1)[:1].expand(size))
    # M 8 (1760 Gate/Up workgroups) lies within the decode limit DECODE_MAX_WGS.
    x, topk_weight, topk_ids = make_inputs(8, HIDDEN, EXPERTS, TOPK, seed=99)
    out, ref = _a8w4_run(w, big, x, topk_weight, topk_ids)
    _assert_path(["a8w4_gateup_decode", down])
    assert_close(out, ref)


# case: (E, H, I, topk, M, w2 over 4 GiB): token counts of the decode tier whose shape neither decode Down holds.
_DECODE_UNFIT_CASES = {
    "topk32_i1792_m2": (40, 512, 1792, 32, 2, False),
    "topk8_i8000_m1": (10, 512, 8000, 8, 1, False),
    "topk17_w2_over_4gib_m2": (EXPERTS, HIDDEN, 320, 17, 2, True),
}


@pytest.mark.parametrize("case", list(_DECODE_UNFIT_CASES))
def test_decode_down_unfit_streams(case):
    """A call on the decode tier's token counts whose shape neither decode Down holds streams instead (no error,
    graded against the reference): topk 32 with I 1792 (past the 16 columns of the Down that quantizes its rows, and
    the slot-pair Down's staged rows past the 64 KiB of LDS) and topk 8 with I 8000 (the 8 rows with their partial sums
    and LUT copies just past 64 KiB, the slot-pair rows far past it), H 512; topk 17 (only the slot-pair Down holds it)
    with w2 one byte past the slot-pair Down's 32-bit offsets (the stride-0 view of test_decode_down_fallback_w2_4gib),
    Qwen E 33. The streaming kernels take any M and padded I.

    The basic tests, the benchmark and the other decode cases have topk <= 11 or I <= 576 and w2 far below 4 GiB,
    where some decode Down holds the shape; test_decode_topk_variants runs topk 17 on the slot-pair Down.
    """
    import dataclasses

    experts, hidden, inter, topk, tokens, over = _DECODE_UNFIT_CASES[case]
    w, prepared = _weights(experts, hidden, inter, 900 + topk)
    if over:
        size = (1 << 32) - prepared.w2[0].numel() * prepared.w2.element_size() + 1
        prepared = dataclasses.replace(
            prepared, w2=prepared.w2.view(-1)[:1].expand(size)
        )
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=901, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    assert _path_tier(moe_a8w4.last_path) == "stream", moe_a8w4.last_path
    assert_close(out, ref)


def _mimo_small_weights(inter):
    """MiMo-shape weights (H 4096, topk 8) with E cut to 12: each expert gets 2M/3 routes on average (cached per I)."""
    return _weights(12, MIMO_HIDDEN, inter, 200 + inter)


@pytest.mark.parametrize(
    "tokens, path",
    [(4, "nn"), (7, "nn"), (8, "stream")],
    ids=["m4_nn", "m7_nn", "m8_stream"],
)
def test_decode_hidden4096_limit(tokens, path):
    """MiMo TP8 shape (H 4096, topk 8, I 256; E 12): from hidden 4096 on (DECODE_WIDE_HIDDEN aside) the decode tier
    ends at DECODE_MAX_M = 7 tokens and takes the Down that quantizes its rows at every M, also from
    DECODE_PREQUANT_MIN_M = 4 on (M 4 / 7: no row kernel, no slot-pair Down); M 8 streams (the prep sorts in blocks of
    16, the Down adds atomically, no a8w4_route_sum).

    The basic tests decode MiMo at M 5 and stream M 24 / 300; the benchmark runs M 4 and 8 without asserting kernels
    and no M 7.
    """
    assert (tokens <= moe_a8w4.DECODE_MAX_M) == (path == "nn")
    w, prepared = _mimo_small_weights(256)
    x, topk_weight, topk_ids = make_inputs(
        tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=230 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "nn":
        _assert_path(_DECODE_PATHS["nn"])
        assert "a8w4_row_quant_decode" not in moe_a8w4.last_path, moe_a8w4.last_path
    else:
        _assert_path(_small_stream())
        assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path
    assert_close(out, ref)


@pytest.mark.parametrize(
    "tokens, case",
    [
        (7, "np"),
        (12, "pp"),
        (8, "w2_4gib"),
        (32, "wide"),
        (33, "past_limit"),
        (33, "e256_past_limit"),
    ],
    ids=[
        "m7_np",
        "m12_pp_two_rounds",
        "m8_w2_4gib",
        "m32_wide",
        "m33_past_limit",
        "e256_m33_stream_bm16",
    ],
)
def test_decode_wide_hidden(monkeypatch, tokens, case):
    """The H 5120 decode (DeepSeek shape, E 24) and its fallbacks: M 7 still runs per route (Down on pre-quantized
    slot pairs; from M 8 on, DECODE_WIDE_DEDUP_MIN_M, Gate/Up is deduplicated by expert, see the basic tests); with
    that threshold raised to 33, M 12 gives 12 * 6 * 36 = 2592 >= 2560 Gate/Up workgroups and Gate/Up reads
    pre-quantized x: the FP8 row of K 5120 enters LDS in two rounds of 16-byte pieces (the second only 64 threads);
    with w2 >= 4 GiB (stride-0 view, as test_decode_down_fallback_w2_4gib) M 8 is not deduplicated and Down takes
    the Down that quantizes its rows; the H 5120 decode ends at DECODE_WIDE_MAX_M = 32: M 32 still
    deduplicates Gate/Up (groups of 2), M 33 streams. With E 256, M 33 gives 0.77 routes per expert and still
    streams, and with fewer than 4E/3 routes it still sorts in blocks of 16 (H 5120 sorts with the prep, any route
    count).

    By default H 5120 deduplicates Gate/Up from M 8 on, so the per-route pre-quantized Gate/Up (two rounds) is only
    reached with the threshold raised; the benchmark's ds_tp4 cases (E 384) run powers of two and 48 without
    asserting kernels (no M 7, 33); the basic tests and the benchmark have w2 far below 4 GiB. The cases pin the H 5120
    decode limit DECODE_WIDE_MAX_M to pyhip M17's 32 tokens (they test the kernels and the boundary logic, not where
    the default limit lies), their decode kernels within it and the streaming kernels past it (_assert_h5120), and the
    w2 fallback, moot past the limit, is skipped there.
    """
    import dataclasses

    monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_MAX_M", 32)
    if case == "e256_past_limit":
        w = make_weights(256, DS_HIDDEN, DS_INTER, seed=280)
        prepared = _a8w4_prepare(w)
    else:
        w, prepared = _ds_weights()
    experts = prepared.experts
    if case == "pp":
        monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_DEDUP_MIN_M", 33)
    elif case == "w2_4gib":
        if tokens > moe_a8w4.DECODE_WIDE_MAX_M:
            pytest.skip("H 5120 does not decode M 8: no dedup to fall back from")
        prepared = dataclasses.replace(
            prepared, w2=prepared.w2.view(-1)[:1].expand(1 << 32)
        )
    elif case == "e256_past_limit":
        assert 3 * tokens * DS_TOPK < 4 * experts
    x, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, experts, DS_TOPK, seed=270 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    path = {
        "np": _DECODE_PATHS["np"],
        "pp": _DECODE_PATHS["pp"],
        "w2_4gib": _DECODE_PATHS["nn"],
        "wide": _DEDUP_PATHS["wide"],
        "past_limit": None,
        "e256_past_limit": None,
    }[case]
    _assert_h5120(tokens, path)
    assert_close(out, ref)


@pytest.mark.parametrize(
    "tokens",
    [moe_a8w4.DECODE_WIDE_MAX_M, moe_a8w4.DECODE_WIDE_MAX_M + 1],
    ids=["limit_dedup", "past_limit_stream"],
)
def test_h5120_decode_limit(tokens):
    """Both sides of the H 5120 decode limit at its default (``DECODE_WIDE_MAX_M``, 10 tokens): DeepSeek TP4 (E 24)
    M 10 still decodes, Gate/Up deduplicated by expert (from DECODE_WIDE_DEDUP_MIN_M on), M 11 streams (the prep,
    blocks of 16, the atomic Down: _assert_h5120). The limit applies to the DECODE_WIDE_HIDDEN sizes only: each hidden
    size's decode tier is bounded on its own (``_decode_range``), so at M 10 / 11 the Qwen TP2 shape (H 2560) still
    decodes (its Gate/Up grid, 2420 workgroups at M 11, stays below DECODE_MAX_WGS) and the MiMo shape (H 4096, up to
    DECODE_MAX_M = 7 tokens) streams.

    The other H 5120 decode cases pin the limit to pyhip M17's 32 (they test the kernels, not where the default lies);
    the benchmark runs ds_tp4 at M 8 and 16 and the layers at M 10 / 11 without asserting kernels; the basic tests
    decode H 5120 at M 8 only.
    """
    limit = moe_a8w4.DECODE_WIDE_MAX_M
    assert moe_a8w4.DECODE_WIDE_DEDUP_MIN_M <= limit
    qwen = (HIDDEN, 320, TOPK, EXPERTS, None, None, None)
    mimo = (MIMO_HIDDEN, 512, MIMO_TOPK, 256, None, None, None)
    assert moe_a8w4._routed_tier(tokens, *qwen) == "decode"
    assert moe_a8w4._routed_tier(tokens, *mimo) == "stream"
    w, prepared = _ds_weights()
    x, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=460 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_h5120(tokens, _DEDUP_PATHS["wide"])
    assert_close(out, ref)


# -- 2.4 streaming and prefill: tier boundaries, grid shrinking, front end, tiles, H 5120 thresholds, variants ----


def test_grid_shrink_boundary_e513():
    """Both sides of the streaming grid shrinking at the full E 513 (the prep's block capacity, and with it the grid
    of the streaming Gate/Up and Down, is M * topk once that is <= E): M 46 (506 routes) and M 47 (517).

    The basic tests (E 33 / 256) are not on this boundary; the benchmark has only M 48 between 33 and 63 (528 routes
    > E, not shrunk, the M 47 side). Both M sort in the dispatch's block (_stream_block_m, as the benchmark's M 48):
    the case asserts the kernels of that block and the block capacity on both sides (506 = M * topk; 513 = E for
    every block size), not which block the dispatch picks.
    """
    w = make_weights(513, HIDDEN, 320, seed=31)
    prepared = _a8w4_prepare(w)
    for tokens, capacity in ((46, 46 * TOPK), (47, 513)):
        block_m = moe_a8w4._stream_block_m(tokens, TOPK, 513, HIDDEN)
        assert stream_a8w4.stream_prep_sizes(tokens, TOPK, 513, block_m)[2] == capacity
        x, topk_weight, topk_ids = make_inputs(
            tokens, HIDDEN, 513, TOPK, seed=32 + tokens
        )
        assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        fused = moe_a8w4._stream_fused_quant(prepared.inter_padded, block_m, HIDDEN)
        _assert_path(["a8w4_stream_prep"] + _stream_path(block_m, fused))


def test_stream_block_m_any_route_count():
    """The streaming sort block follows the mean rows per expert alone (``_stream_block_m``): blocks of 16 up to 12
    rows at any route count and hidden size, also below 4E/3 routes (about one route per expert), where blocks of 32
    ran while the streaming path still sorted with aiter's Opus moe_sorting: Qwen (E 513, topk 11) M 33 / 48 / 62,
    MiMo (E 256) M 33, DeepSeek (E 384) M 48; the hidden size passed or not.

    test_grid_shrink_boundary_e513 runs E 513 M 46 / 47 end to end in whatever block the dispatch picks, the
    benchmark's tp2_m48 / tp4_m48 without asserting kernels: neither checks the block.
    """
    for tokens, topk, experts, hidden in (
        (33, TOPK, 513, HIDDEN),
        (48, TOPK, 513, HIDDEN),
        (62, TOPK, 513, HIDDEN),
        (33, MIMO_TOPK, MIMO_EXPERTS, MIMO_HIDDEN),
        (48, 6, 384, DS_HIDDEN),
    ):
        assert 3 * tokens * topk < 4 * experts
        for h in (hidden, 0):
            assert moe_a8w4._stream_block_m(tokens, topk, experts, h) == 16, tokens


@pytest.mark.parametrize(
    "tokens, path",
    [(132, _stream_path(64, False)), (133, _prefill_path(128))],
    ids=["m132", "m133"],
)
def test_stream_dispatch_boundary(monkeypatch, tokens, path):
    """The boundary between streaming and prefill (streaming while M * topk <= STREAM_MAX_ROWS * E at H 2560 with
    I 320), with the limit set here to the module's 44 rows per expert (the case tests the rule, not the tuned value):
    with E 33 and topk 11, M 132 (44 rows) still streams (blocks of 64, the row kernel quantizing the intermediate)
    and M 133 takes prefill with the default K tile 128; both sort with the prep.

    The basic tests (streaming M 64 / 300, prefill M 257 / 1031 / 1600) and the benchmark are not on this boundary.
    """
    monkeypatch.setattr(moe_a8w4, "STREAM_MAX_ROWS", 44)
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(tokens, HIDDEN, EXPERTS, TOPK, seed=tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(["a8w4_stream_prep"] + path)
    assert_close(out, ref)


@pytest.mark.parametrize(
    "hidden, inter, tokens, path, reuse",
    [
        (2560, 160, 140, _stream_path(64, True), False),
        (2560, 160, 141, _prefill_path(128), None),
        (4096, 256, 62, _stream_path(32, False), False),
        (4096, 256, 159, _stream_path(64, False), False),
        (4096, 256, 160, _stream_path(64, False), True),
        (4096, 256, 640, _stream_path(64, False), True),
        (4096, 256, 641, _prefill_path(128), None),
        (4096, 512, 110, _stream_path(64, False), False),
        (4096, 512, 111, _prefill_path(128), None),
        (512, 1024, 70, _stream_path(32, False), False),
        (512, 1024, 71, _prefill_path(128), None),
        (5120, 320, 71, _stream_path(64, False), False),
    ],
    ids=[
        "h2560_i192_m140",
        "h2560_i192_m141",
        "h4096_i256_m62_bm32",
        "h4096_i256_m159",
        "h4096_i256_m160_reuse",
        "h4096_i256_m640_reuse",
        "h4096_i256_m641",
        "h4096_i512_m110",
        "h4096_i512_m111",
        "h512_i1024_m70_bm32",
        "h512_i1024_m71",
        "h5120_i320_m71_bm64",
    ],
)
def test_stream_rows_limits(monkeypatch, hidden, inter, tokens, path, reuse):
    """The streaming limit by shape, both sides each, with the limits set here to the module's values (the cases test
    the rules, not the tuned values): 256 rows per expert from H 4096 on (STREAM_MAX_ROWS_WIDE) and 56 at H 2560 when
    the Gate/Up of blocks of 64 quantizes the intermediate (I_pad 192, STREAM_MAX_ROWS_FUSED_QUANT); with E 5 and
    topk 2, M 140 / 141 give 56 / 56.4 rows and M 640 / 641 256 / 256.4. From padded I 512 on the cap is
    STREAM_MAX_ROWS (44) where the Down's tile of blocks of 64 fits in LDS (I 512 with the eight-wave Down: M 110 / 111,
    44 / 44.4 rows) and STREAM_BM32_ROWS (28) where it does not (I 1024 at H 512, the last of blocks of 32: M 70 / 71).
    H 4096 with I 256 streams with the atomic Down, which reads the activations from LDS at blocks of 32 (M 62,
    K % 256 == 0); below I 512 the H 5120 shapes with fewer than STREAM_H5120_MIN_ROUTES routes still stream past 28
    rows in blocks of 64 (I 320, M 71). Every streaming Down adds the routes atomically (no a8w4_route_sum), the
    prefill Down writes them for a8w4_route_sum. The streaming Gate/Up is built to load its weights cached (``reuse``)
    exactly from one block of rows per expert on (M * topk >= block_m * E): M 160 (64 rows) and M 640, not M 159
    (63.6 rows) nor the other streaming cases.

    The basic tests and the benchmark are not on these boundaries (the STREAM_MAX_ROWS one is
    test_stream_dispatch_boundary); the benchmark's MiMo TP8 M1024-8192 run both sides of the cached loads and MiMo TP4
    M1024 the eight-wave Down, without asserting kernels or build options. E 5 and topk 2 keep this fast.
    """
    for name, rows in (
        ("STREAM_MAX_ROWS", 44),
        ("STREAM_MAX_ROWS_WIDE", 256),
        ("STREAM_MAX_ROWS_FUSED_QUANT", 56),
        ("STREAM_BM32_ROWS", 28),
    ):
        monkeypatch.setattr(moe_a8w4, name, rows)
    built, gateup = [], moe_a8w4._stream_gateup_kernel

    def spy(*args, **options):
        built.append(options.get("reuse", False))
        return gateup(*args, **options)

    monkeypatch.setattr(moe_a8w4, "_stream_gateup_kernel", spy)
    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, 5, 2, seed=300 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
    sums = "a8w4_route_sum" in moe_a8w4.last_path
    assert sums == (_path_tier(moe_a8w4.last_path) == "prefill"), moe_a8w4.last_path
    assert built == ([] if reuse is None else [reuse]), built
    assert_close(out, ref)


@pytest.mark.parametrize(
    "inter, options", [(256, {"block_m": 128, "block_n": 256})], ids=["bm128_bn256"]
)
def test_tile_variants(inter, options):
    """Given tile options select the prefill tiles: BLOCK_N 256 (two 16-row blocks per wave) and BLOCK_M 128; the
    K tile not given is 256 for M <= 2048 (M=300).

    The basic tests and the benchmark only run the default tiles (BLOCK_M 64, BLOCK_N 128).
    """
    w = make_weights(33, HIDDEN, inter, seed=36)
    x, topk_weight, topk_ids = make_inputs(300, HIDDEN, 33, TOPK, seed=37)
    assert_close(*_a8w4_run(w, _a8w4_prepare(w), x, topk_weight, topk_ids, **options))
    _assert_path(_prefill_path(256, block_m=128, block_n=256))


@pytest.mark.parametrize(
    "filler, options, block_m, tile_k",
    [
        (0, {"block_m": 64}, 64, 256),
        (1854, {"block_m": 128}, 128, 256),
        (1855, {"block_m": 64}, 64, 128),
    ],
    ids=["bm64_m194", "bm128_m2048", "bm64_m2049"],
)
def test_prefill_half_block_boundary(filler, options, block_m, tile_k):
    """Down computes only the first BM/2 rows of a sorted block whose row BM/2 is padding, and Gate/Up (BM 64) only
    the valid rows rounded up to 16 (valid rows lead the block): the routes are built so that the experts have
    exactly 32, 33, 64, 65, 96, 97 and 1 rows, so row BM/2 of a tail block is padding or valid on either side
    (32 / 33 and 96 / 97 for BM 64, 64 / 65 for BM 128), and with K = 512 every delta byte 0-3 of each 128-K
    Gate/Up step is used. Also checks the K tile left to the op: 256 up to M = 2048 (M 194 and 2048), 128 from
    M = 2049.

    Random routing gives random tail blocks, so the basic tests and the benchmark do not reliably hit these
    boundaries, and they pass no tile options; H 512, I 64 and topk 2 keep this fast. Every case passes block_m:
    by default pyhip M17 runs M 194 with the K tile 128 (many routes per expert).
    """
    w, prepared = _weights(9, 512, 64, 131)
    counts = torch.tensor([32, 33, 64, 65, 96, 97, 1, filler, filler])
    flat = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    tokens = flat.numel() // 2
    # The routes sorted by expert, folded into two columns: no expert has more than ``tokens`` rows, so the two
    # experts of a token differ.
    topk_ids = (
        torch.stack([flat[:tokens], flat[tokens:]], dim=1).to(torch.int32).contiguous()
    )
    x, topk_weight, _ = make_inputs(tokens, 512, 9, 2, seed=132)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(_prefill_path(tile_k, block_m=block_m))
    assert_close(out, ref)


@pytest.mark.parametrize(
    "kernel, nsplit, two_pass, xcd, wgs",
    [
        ("gateup", None, None, moe_a8w4.GATEUP_XCD_CHUNK_H5120, 0),
        (
            "gateup",
            None,
            None,
            moe_a8w4.GATEUP_FEW_WAVES_XCD_CHUNK,
            moe_a8w4.GATEUP_FEW_WAVES,
        ),
        ("down", 4, True, 0, 0),
        ("down", 4, True, moe_a8w4.DOWN_XCD_CHUNK_H5120, 0),
        ("down", 1, True, moe_a8w4.DOWN_XCD_CHUNK_H5120, 0),
    ],
    ids=[
        "gateup_xcd",
        "gateup_xcd_few_waves",
        "down_two_pass",
        "down_two_pass_xcd_split4",
        "down_two_pass_xcd_split1",
    ],
)
def test_prefill_xcd_two_pass_bitexact(kernel, nsplit, two_pass, xcd, wgs):
    """The H5120 prefill grid variants equal the plain kernels bit for bit on one sort and one input: the
    XCD-chunked 1-D grid (XCD_CHUNK > 0: Gate/Up chunks of one sorted block's 9 N tiles, Down chunks of 4
    neighbouring sorted blocks with their hidden parts, dealt to the 4 XCDs in turn) for mid, routes and loc_ids,
    the Gate/Up kept to GATEUP_FEW_WAVES workgroups per CU (WGS_PER_CU: LDS padded, the activation tile staged two
    tiles ahead at K 5120) with chunks of GATEUP_FEW_WAVES_XCD_CHUNK blocks, and Down's activation tile staged in two
    passes through LDS (TWO_PASS: K = 576 as 5 + 4 blocks of 64). The baseline is the 2-D grid with one pass, as many
    workgroups per CU as fit and the hidden dimension split in 4.

    The routes are built per expert (E 24 with 32, 33, 64, 65, 96, 97, 1, 20 and 16 x 12 rows, M = 100): both
    sides of Down's half-block branch and every valid-row count of Gate/Up occur; the Opus sort's 34 tasks are no
    multiple of 4, 8 or 16, so workgroups the chunked grid maps past the task count must exit; Down runs split in 4
    and unsplit. The basic tests and the benchmark only grade errors, so a chunk computed twice, skipped or
    missing its loc_ids may go unnoticed (XCD chunking only starts at 16384 routes, the benchmark's ds_tp4
    M >= 4096, the few-waves Gate/Up at 98304, M >= 16384); the one-pass K = 576 tile only runs below 81 estimated
    workgroups.
    """
    from aiter.fused_moe import moe_sorting

    _, prepared = _ds_weights()
    E, H, inter, topk = DS_EXPERTS, DS_HIDDEN, prepared.inter_padded, DS_TOPK
    counts = torch.tensor([32, 33, 64, 65, 96, 97, 1, 20] + [12] * 16)
    flat = torch.repeat_interleave(torch.arange(E), counts)
    M = flat.numel() // topk
    # The routes sorted by expert, folded into topk columns: no expert has more than M rows, so the topk experts of
    # a token differ.
    topk_ids = flat.view(topk, M).T.to(torch.int32).contiguous()
    x, topk_weight, _ = make_inputs(M, H, E, topk, seed=136, shared=False)
    sorted_ids, sorted_weights, sorted_experts, num_valid = moe_sorting(
        topk_ids, topk_weight, E, H, x.dtype, 64, accumulate=False
    )[:4]
    routing, task_num = (
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
    ), sorted_experts.numel()
    assert task_num % 4 != 0, task_num
    xq, xs = moe_a8w4._quantize(x, M, H, "bf16")
    device, constants = x.device.index, moe_a8w4._trace_constants()

    def gateup(chunk, wgs=0):
        mid = torch.full((M, topk, inter), float("nan"))
        kernel = moe_a8w4._gateup_kernel(
            2 * inter,
            H,
            topk,
            64,
            128,
            128,
            E,
            0.0,
            chunk,
            device,
            constants,
            **({"wgs_per_cu": wgs} if wgs else {}),
        )
        moe_a8w4._launch(
            "gateup",
            kernel,
            xq,
            prepared.w1,
            prepared.w1_delta,
            mid,
            *routing,
            prepared.w1_scale,
            xs,
            M,
            task_num,
        )
        return mid

    def down(split, two, chunk):
        routes = torch.full((task_num * 64, H), float("nan"), dtype=torch.bfloat16)
        loc = torch.full((M, topk), -1, dtype=torch.int32)
        kernel = moe_a8w4._down_kernel(
            H, inter, topk, 64, E, split, two, chunk, device, constants
        )
        moe_a8w4._launch(
            "down",
            kernel,
            mq,
            prepared.w2,
            prepared.w2_delta,
            routes,
            *routing,
            prepared.w2_scale,
            ms,
            loc,
            M,
            task_num,
        )
        return routes.view(torch.int16), loc

    mid = gateup(0)
    assert not bool(mid.isnan().any())
    if kernel == "gateup":
        assert torch.equal(gateup(xcd, wgs).view(torch.int32), mid.view(torch.int32))
        return
    mq, ms = moe_a8w4._quantize(mid, M * topk, inter, "f32")
    routes, loc = down(4, False, 0)
    assert bool((loc >= 0).all())
    other_routes, other_loc = down(nsplit, two_pass, xcd)
    assert torch.equal(other_routes, routes) and torch.equal(other_loc, loc)


def _spy(monkeypatch, name, record):
    """Wrap ``moe_a8w4.<name>`` (a launcher factory) so that each call's positional arguments go to ``record``
    (keyword-only build options pass through unrecorded)."""
    original = getattr(moe_a8w4, name)

    def spy(*args, **kwargs):
        record.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(moe_a8w4, name, spy)


@pytest.mark.parametrize(
    "min_wgs, nsplit", [(5, 1), (8, 2), (1000, 4)], ids=["split1", "split2", "split4"]
)
def test_prefill_down_split(monkeypatch, min_wgs, nsplit):
    """The prefill Down splits the hidden dimension over 1, 2 or 4 workgroups per sorted block by the estimated blocks
    times the split against ``DOWN_MIN_WGS`` (E 5, M 100, topk 2: 5 blocks; DOWN_MIN_WGS 5 / 8 / default 1000); each
    part writes its columns of the routes and only part 0 the loc_ids. H 4096, I 512, 40 rows per expert; block_m=64
    selects the prefill tier (sorted by the prep) whatever the streaming limit; a8w4_route_sum adds the routes.

    The basic tests and the benchmark run whichever split their shape gets and assert neither the split nor its
    result per split (the basic MiMo TP4 M 1600 is prefill with split 4); here all three run on one shape against the
    same reference.
    """
    monkeypatch.setattr(moe_a8w4, "DOWN_MIN_WGS", min_wgs)
    assert moe_a8w4._down_split(100, 2, 5, 4096, 512, 64) == nsplit
    built = []
    _spy(monkeypatch, "_down_kernel", built)
    w, prepared = _small_weights(4096, 512)
    x, topk_weight, topk_ids = make_inputs(100, 4096, 5, 2, seed=310, shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, block_m=64)
    assert [args[5] for args in built] == [nsplit], built
    path = moe_a8w4.last_path
    _assert_path(["a8w4_stream_prep"] + _prefill_path(256))
    assert any(name in _ROUTE_SUMS for name in path), path
    assert_close(out, ref)


def _route_sum(loc_ids, routes):
    """One ``a8w4_route_sum`` launch (the prefill tier's sum, ``_stream_sum_kernel``) of ``routes`` [rows, H] by
    ``loc_ids`` [M, columns] into a NaN-filled BF16 [M, H]."""
    out = torch.full(
        (loc_ids.shape[0], routes.shape[1]), float("nan"), dtype=torch.bfloat16
    )
    kernel = moe_a8w4._stream_sum_kernel(
        loc_ids.shape[1], routes.shape[1], routes.device.index
    )
    moe_a8w4._launch(
        "a8w4_route_sum",
        kernel,
        loc_ids,
        routes,
        out,
        loc_ids.shape[0],
        routes.shape[0],
    )
    torch.cuda.synchronize()
    return out


def _route_sum_reference(loc_ids, routes):
    """a8w4_route_sum's arithmetic in torch: per token its routes' BF16 rows summed in f32 in slot order, the first slot
    starting the sum, a slot whose loc is negative or ``routes.shape[0]`` and more adding +0; rounded to BF16 (nearest
    even)."""
    rows, total = routes.shape[0], None
    for k in range(loc_ids.shape[1]):
        loc = loc_ids[:, k].long()
        valid = ((loc >= 0) & (loc < rows)).unsqueeze(1)
        values = torch.where(valid, routes[loc.clamp(0, rows - 1)].float(), 0.0)
        total = values if total is None else total + values
    return total.to(torch.bfloat16)


@pytest.mark.parametrize("hidden", [2560, 4096, 5120], ids=["h2560", "h4096", "h5120"])
def test_route_sum_bitexact(hidden):
    """``a8w4_route_sum`` (the prefill tier's sum of the routes, replacing the base's sorted_sum) gives the bits of its
    arithmetic in torch (_route_sum_reference): f32 in slot order, round to nearest even. A wave sums 2 KB of an output
    row while the row's 1 KB pieces are even in number (H 4096 / 5120: 8 / 10, all 2 * topk loads of a lane in flight)
    and 1 KB while odd (H 2560: 5); 37 tokens leave the last workgroup's waves partly past the tokens. loc_ids of 7
    columns (the layers' 6 routed and the folded shared expert) with -1, rows past the routes (``rows``, ``rows`` + 3,
    2**31 - 1: no route, nothing read), a token without any route, the first slot missing and a row taken twice.

    End to end route_sum only runs behind the prefill Down, where the basic tests, the benchmark and the prefill cases
    grade it within the tolerance and never pass a loc at or past the routes' rows.
    """
    tokens, columns, rows = 37, 7, 300
    gen = torch.Generator(device="cuda").manual_seed(990 + hidden)
    routes = (torch.randn((rows, hidden), generator=gen) * 4).to(torch.bfloat16)
    loc_ids = torch.randint(
        0, rows, (tokens, columns), generator=gen, dtype=torch.int32
    )
    loc_ids[(torch.rand((tokens, columns), generator=gen) < 0.2)] = -1
    loc_ids[1] = -1
    loc_ids[2, 0] = -1
    loc_ids[3, :3] = torch.tensor([rows, rows + 3, 2**31 - 1], dtype=torch.int32)
    loc_ids[4, 1] = loc_ids[4, 0] = rows - 1
    out = _route_sum(loc_ids, routes)
    assert torch.equal(_bits(out), _bits(_route_sum_reference(loc_ids, routes)))
    assert bool((out[1] == 0).all())


@pytest.mark.parametrize("hidden", [2560, 5120], ids=["h2560", "h5120"])
def test_route_sum_routes_past_4gib(hidden):
    """``a8w4_route_sum`` reads every route's row through a descriptor whose base is the row's own 64-bit address, so
    routes past 4 GiB (the prefill tier's at ds_layer M98304: 7 GB) are summed right: a routes buffer just over
    4 GiB (only the rows the case reads written) with rows below, across (the row holding byte 2**32) and past the
    4 GiB boundary, the last row, -1 and the row count, against _route_sum_reference; the 1 KB (H 2560) and the 2 KB
    (H 5120) wave.

    The benchmark reaches routes past 4 GiB only at its M98304 layer cases, graded within the tolerance; the basic
    tests and the other cases stay far below.
    """
    row_bytes = 2 * hidden
    across = (1 << 32) // row_bytes
    rows = across + 8
    assert across * row_bytes < 1 << 32 < (across + 1) * row_bytes
    routes = torch.empty((rows, hidden), dtype=torch.bfloat16)
    picks = torch.tensor([0, 1, across - 1, across, across + 1, rows - 1])
    gen = torch.Generator(device="cuda").manual_seed(995 + hidden)
    routes[picks.cuda()] = torch.randn((picks.numel(), hidden), generator=gen).to(
        torch.bfloat16
    )
    loc_ids = torch.tensor(
        [
            [across, 0, across + 1, rows - 1],
            [across - 1, -1, across, 1],
            [rows, across + 1, rows - 1, -1],
        ],
        dtype=torch.int32,
    )
    out = _route_sum(loc_ids, routes)
    want = _route_sum_reference(loc_ids, routes)
    del routes
    torch.cuda.empty_cache()
    assert torch.equal(_bits(out), _bits(want))


# case: (M, topk, E, H, sorting block, zero the output, the last column a folded shared expert's). The M * topk routes
# split into pieces of stream_a8w4._PREP_PIECE (256); past _PREP_SORT_WGS (64) pieces each sorting workgroup takes
# ``span`` of them. Every case leaves the last piece and the last workgroup's range partly filled.
_PREP_LARGE_CASES = {
    # 16484 routes: 65 pieces, span 2, 33 workgroups, the last one 100 routes.
    "r16484_span2": (4121, 4, 33, 512, 64, True, False),
    # 200000 routes: 782 pieces, span 13, 61 workgroups, the last one 1.25 pieces.
    "r200000_e256": (25000, 8, 256, 512, 64, False, False),
    # 210003 routes (140002 routed): 821 pieces, span 13, 64 workgroups; E 8, so thousands of routes per LDS counter.
    "r210003_e8_shared": (70001, 3, 8, 512, 64, False, True),
}


@pytest.mark.parametrize("case", list(_PREP_LARGE_CASES))
def test_stream_prep_large_routes_bitexact(case):
    """The prep at many routes (the prefill tier sorts with it at every M): past _PREP_SORT_WGS pieces each sorting
    workgroup places ``span`` pieces (span derived in the kernel from the workgroup count of the launch, the next
    piece's ids and weights in flight), histograms all routes in rounds with the next round in flight and skips the
    before-count past its own range start. The sort has moe_sorting's format (_assert_prep_sort, with 10% of the ids
    masked, token 1 all masked, one id out of range, and in the shared case repeated ids and a shared column with
    in-range ids), xq / xs equal ``_quantize`` bit for bit and nothing past the outputs is written (_prep_launch).

    The basic tests sort at most 14336 routes (one piece per workgroup); the benchmark's prefill cases reach 688128
    routes but only grade the output within the tolerance and never look at the sort format.
    """
    tokens, topk, experts, hidden, block_m, zero, shared = _PREP_LARGE_CASES[case]
    routes = tokens * topk
    n_sort = stream_a8w4.stream_prep_sizes(tokens, topk, experts, block_m)[0]
    pieces = -(-routes // stream_a8w4._PREP_PIECE)
    span = -(-pieces // n_sort)
    assert n_sort <= stream_a8w4._PREP_SORT_WGS < pieces and span > 1
    assert routes % stream_a8w4._PREP_PIECE and pieces < n_sort * span
    gen = torch.Generator(device="cuda").manual_seed(890 + topk)
    routed = topk - shared
    ids = torch.rand((tokens, experts), generator=gen).argsort(dim=1)[:, :routed]
    ids = ids.to(torch.int32)
    ids[torch.rand((tokens, routed), generator=gen) < 0.1] = -1
    ids[1] = -1
    ids[2, 0] = experts
    if shared:
        ids[::2, 1] = ids[::2, 0]
        last = torch.randint(0, experts + 1, (tokens, 1), generator=gen)
        ids = torch.cat([ids, last.to(torch.int32)], dim=1)
    topk_weight = torch.rand((tokens, topk), generator=gen)
    x = (torch.randn((tokens, hidden), generator=gen) * 3).to(torch.bfloat16)
    *sort, q, scale = _prep_launch(
        x, topk_weight, ids.contiguous(), experts, block_m, zero, shared
    )
    _assert_prep_sort(sort, topk_weight, ids, experts, block_m, shared)
    q0, scale0 = moe_a8w4._quantize(x, tokens, hidden, "bf16")
    assert torch.equal(q, q0.view(torch.uint8))
    assert torch.equal(scale.view(torch.int32), scale0.view(torch.int32))


# case: (H, I, E, topk, M, sorting block, ACT_STEPS and waves of the dispatch's tile). DeepSeek TP4 (E 24, I 576)
# M 40 / 64 sort in blocks of 16 / 32. Blocks of 64 (K % 128 == 0): H 5120 at the TP8 I 320 (E 24, topk 6, 4 waves:
# 32 bytes per lane and group, two 16-byte loads; skewed routing, so that the blocks hold 1-4 live sub-blocks) and
# H 4096 with I 256 (E 5, topk 2, 8 waves: one 16-byte load per lane and group).
_ACT_STEPS_CASES = {
    "m40_bm16": (DS_HIDDEN, DS_INTER, DS_EXPERTS, DS_TOPK, 40, 16, 4, 4),
    "m64_bm32": (DS_HIDDEN, DS_INTER, DS_EXPERTS, DS_TOPK, 64, 32, 2, 4),
    "i320_m240_bm64": (DS_HIDDEN, 320, DS_EXPERTS, DS_TOPK, 240, 64, 2, 4),
    "h4096_m120_bm64": (4096, 256, 5, 2, 120, 64, 2, 8),
}


@pytest.mark.parametrize("case", list(_ACT_STEPS_CASES))
def test_stream_act_steps_bitexact(monkeypatch, case):
    """The streaming Gate/Up's light waves (one gate and one up block each, sharing the activations through an LDS
    ring) load the block's activations ACT_STEPS 64-K steps at a time (issued two groups ahead, written to LDS at the
    end of the previous group): the dispatch's tile has 4 / 2 steps at H 5120 with blocks of 16 / 32 (16 bytes per
    lane), and 2 with blocks of 64 at any H with K % 128 == 0 (a row's 128 bytes of a group are whole cache lines;
    32 bytes per lane with 4 waves, two 16-byte loads), where it also computes the LUT with VALU instead of reading
    the LDS table (LUT_LDS False). mid equals bit for bit that of the kernel with ACT_STEPS 1 (4 / 8 / 16 bytes per
    lane and step, a barrier every 2 steps) on the same sort and input, and with blocks of 64 also that of the tile
    with ACT_STEPS 1 and the LDS table. _ACT_STEPS_CASES lists the shapes; the H 5120 blocks of 32 / 64 hold every
    count of live sub-blocks (asserted below).

    The basic DeepSeek M 64 and the benchmark's ds_tp4 M48-1024 (blocks of 16 / 32), mtp_layer_tp8 M1195-1365 and
    mimo M1024 (blocks of 64) run these tiles but only grade end-to-end errors, which do not show a few differing
    elements; no dispatch runs a tile with the other ACT_STEPS or LUT.
    """
    hidden, inter, experts, topk, tokens, block_m, steps, waves = _ACT_STEPS_CASES[case]
    compile_stream, seen = stream_a8w4.compile_a8w4_gateup_stream, {}

    def spy(**options):
        seen.update(options)
        return compile_stream(**options)

    monkeypatch.setattr(stream_a8w4, "compile_a8w4_gateup_stream", spy)
    if inter == DS_INTER:
        _, prepared = _ds_weights()
    elif hidden == DS_HIDDEN:
        _, prepared = _weights(experts, hidden, inter, 401)
    else:
        _, prepared = _small_weights(hidden, inter)
    E, inter = prepared.experts, prepared.inter_padded
    assert E == experts
    assert moe_a8w4._stream_block_m(tokens, topk, E, hidden) == block_m
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_gateup_constants()
    args = (2 * inter, hidden, topk, block_m, E, 0.0, device, constants)
    # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
    moe_a8w4._stream_gateup_kernel.__wrapped__(*args)
    assert (seen["ACT_STEPS"], seen["W"], seen["G"]) == (steps, waves, 1), seen
    assert not seen.get("QUANT"), seen
    assert seen.get("LUT_LDS", True) == (block_m != 64), seen
    variants = [{**seen, "ACT_STEPS": 1}]
    if block_m == 64:
        variants.append({**seen, "ACT_STEPS": 1, "LUT_LDS": True})
    kernels = [moe_a8w4._stream_gateup_kernel(*args)] + [
        compile_stream(**options) for options in variants
    ]
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, E, topk, seed=450 + tokens, shared=False
    )
    if block_m == 64 and hidden == DS_HIDDEN:
        # Skewed routing (expert e drawn with weight 0.1 + e / 12): the experts' tails cover 1-4 sub-blocks.
        gen = torch.Generator(device="cuda").manual_seed(455)
        probs = torch.linspace(0.1, 2.0, E, device="cuda").expand(tokens, E)
        topk_ids = torch.multinomial(probs, topk, generator=gen).to(torch.int32)
    counts = torch.bincount(topk_ids.flatten().long(), minlength=E)
    tail = counts % block_m
    if hidden == DS_HIDDEN and block_m > 16:
        live = {(int(t) + 15) // 16 for t in tail.tolist() if t}
        assert live == set(range(1, block_m // 16 + 1)), counts
        assert block_m == 32 or bool((counts >= block_m).any()), counts
    sorted_ids, _, sorted_experts, num_valid, _, xq, xs = moe_a8w4._stream_prep(
        x, topk_weight, topk_ids, E, block_m, torch.empty_like(x), zero=False
    )
    mids = []
    for kernel in kernels:
        mids.append(torch.full((tokens, topk, inter), float("nan")))
        moe_a8w4._launch(
            "gateup",
            kernel,
            xq,
            prepared.w1,
            prepared.w1_delta,
            mids[-1],
            sorted_ids,
            sorted_experts,
            num_valid,
            prepared.w1_scale,
            xs,
            xs,
            tokens,
            sorted_experts.numel(),
        )
    assert not bool(mids[0].isnan().any())
    for mid in mids[1:]:
        assert torch.equal(mids[0].view(torch.int32), mid.view(torch.int32))


def _spy_build(monkeypatch, name):
    """Wrap ``stream_a8w4.<name>`` (a kernel builder taking keyword options) so that the options of its last build land
    in the returned dict; returns ``(original builder, dict)``."""
    original, seen = getattr(stream_a8w4, name), {}

    def spy(**options):
        seen.clear()
        seen.update(options)
        return original(**options)

    monkeypatch.setattr(stream_a8w4, name, spy)
    return original, seen


@pytest.mark.parametrize(
    "tokens, block_m, tile",
    [(64, 32, (8, 4)), (40, 16, (16, 1))],
    ids=["m64_bm32", "m40_bm16"],
)
def test_stream_down_tiles_bitexact(monkeypatch, tokens, block_m, tile):
    """The streaming Down without atomics writes the same route bits with every tile on one sort and input: a wave takes
    its 16-channel weight blocks in pairs (one 16-byte load per lane and 64-K step covers two adjacent rows of either
    block of a pair, four MFMA chains per sub-block), so the pairs per wave (R 16 x W 1, or R 8 x W 4 sharing the
    block's activation rows through LDS) and the pairs in flight (PREFETCH 2: one pair, STREAM_DOWN_PREFETCH_SMALL for
    blocks of up to 32 rows; 4: two pairs) change only the order of the loads. DeepSeek H 5120 (E 24, I 576) with
    M 64 / 40, which the dispatch sorts in blocks of 32 / 16 with the tiles R8W4 / R16W1 and STREAM_DOWN_PREFETCH_SMALL
    (asserted; the launcher built without atomics, whatever the dispatch picks for the shape: with atomics the order
    of the adds, and with it the bits, changes from run to run), compared with the other tile and the other prefetch
    (2 or 4); with blocks of 32 both one and two live sub-blocks occur (asserted). An odd R fails when the launcher is
    built (the loads cover block pairs).

    The basic DeepSeek M 64 and the benchmark run one tile per shape and grade end to end, which does not show a few
    differing elements; no dispatch runs the other tile or prefetch of a shape.
    """
    compile_down, seen = _spy_build(monkeypatch, "compile_a8w4_down_stream")
    _, prepared = _ds_weights()
    E, inter = prepared.experts, prepared.inter_padded
    assert moe_a8w4._stream_block_m(tokens, DS_TOPK, E, DS_HIDDEN) == block_m
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_down_constants()
    args = (DS_HIDDEN, inter, DS_TOPK, block_m, E, False, device, constants)
    # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
    moe_a8w4._stream_down_kernel.__wrapped__(*args)
    base = dict(seen)
    assert (base.get("R", 16), base.get("W", 1), base["PREFETCH"]) == (
        *tile,
        moe_a8w4.STREAM_DOWN_PREFETCH_SMALL,
    ), base
    other = {"R": 16, "W": 1} if tile == (8, 4) else {"R": 8, "W": 4}
    kernels = [
        moe_a8w4._stream_down_kernel(*args),
        compile_down(**{**base, "PREFETCH": 4 if base["PREFETCH"] != 4 else 2}),
        compile_down(**{**base, **other}),
    ]
    with pytest.raises(AssertionError, match="pairs of 16-row blocks"):
        compile_down(**{**base, "R": 5, "W": 1})
    x, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, E, DS_TOPK, seed=470 + tokens, shared=False
    )
    tail = torch.bincount(topk_ids.flatten().long(), minlength=E) % block_m
    assert block_m == 16 or (
        bool(((tail > 0) & (tail <= 16)).any()) and bool((tail > 16).any())
    ), tail
    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, _, _ = (
        moe_a8w4._stream_prep(
            x, topk_weight, topk_ids, E, block_m, torch.empty_like(x), zero=False
        )
    )
    gen = torch.Generator(device="cuda").manual_seed(480 + tokens)
    mid = torch.randn((tokens * DS_TOPK, inter), generator=gen)
    mq, ms = moe_a8w4._quantize(mid, tokens * DS_TOPK, inter, "f32")
    task_num = sorted_experts.numel()
    routes = []
    for kernel in kernels:
        routes.append(
            torch.full(
                (task_num * block_m, DS_HIDDEN), float("nan"), dtype=torch.bfloat16
            )
        )
        moe_a8w4._launch(
            "down",
            kernel,
            mq,
            prepared.w2,
            prepared.w2_delta,
            routes[-1],
            sorted_ids,
            sorted_weights,
            sorted_experts,
            num_valid,
            prepared.w2_scale,
            ms,
            tokens,
            task_num,
        )
    live = routes[0][loc_ids.flatten().long()]
    assert not bool(live.isnan().any()) and bool((live != 0).any())
    for other_routes in routes[1:]:
        assert torch.equal(_bits(routes[0]), _bits(other_routes))


# Rows per expert of the atomic streaming Down cases (topk 2, E 8, M 134): 16 / 32 / 48 / 64 rows fill every live
# sub-block of their blocks, 8 / 17 / 33 / 50 leave the last one partly padding, so that with sorting blocks of 16, 32
# and 64 alike there are blocks of every live sub-block count without padding and with it (asserted in the case).
_ATOMIC_DOWN_COUNTS = (16, 32, 48, 64, 8, 17, 33, 50)


def _atomic_down_case(w, prepared, seed):
    """The routes of ``_ATOMIC_DOWN_COUNTS`` (sorted by expert and folded into two columns: no expert has more than M
    rows, so the two experts of a token differ), an input and routing weights, a random FP8 intermediate and the f32
    reference of the Down alone: per token the sum over its routes of weight x (mq x ms) . w2[e]^T, with w2
    dequantized exactly."""
    E, H, inter = prepared.experts, prepared.hidden, prepared.inter_padded
    flat = torch.repeat_interleave(torch.arange(E), torch.tensor(_ATOMIC_DOWN_COUNTS))
    M = flat.numel() // 2
    topk_ids = flat.view(2, M).T.to(torch.int32).contiguous()
    x, topk_weight, _ = make_inputs(M, H, E, 2, seed=seed, shared=False)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    mid = torch.randn((2 * M, inter), generator=gen)
    mq, ms = moe_a8w4._quantize(mid, 2 * M, inter, "f32")
    act, ids = mq.float() * ms.unsqueeze(1), topk_ids.flatten().long()
    ref = torch.zeros((M, H))
    for e in range(E):
        rows = (ids == e).nonzero().flatten()
        y = act[rows] @ dequant_mxfp4(w["w2"][e], w["w2_scale"][e]).T
        ref.index_add_(0, rows // 2, y * topk_weight.flatten()[rows].unsqueeze(1))
    return x, topk_weight, topk_ids, mq, ms, ref


@pytest.mark.parametrize(
    "hidden, inter", [(5120, 320), (4096, 256)], ids=["h5120_i320", "h4096_i256"]
)
def test_stream_atomic_down_epilogue(monkeypatch, hidden, inter):
    """The atomic streaming Down adds a sub-block's rows without predicates when its last row, and so every row, is a
    real one, else with them; blocks of 64 at K 256 keep the rows' output offsets and factors in an LDS table
    (ROWS_LDS) and read the activations from LDS. The routes of _ATOMIC_DOWN_COUNTS give, for blocks of 16, 32 and
    64, blocks of every live sub-block count with and without padding in the last one; with topk 2 a token's BF16x2
    atomics are two adds onto zero, whose sum does not depend on their order, so the dispatch's atomic launchers of
    the three block sizes (H 5120 with I 320: the DeepSeek and MTP TP8 layers' routed shape; H 4096 with I 256: MiMo
    TP8's; E 8) and, at K 256, blocks of 64 without the table and also without the LDS activations give the same
    bits, and the first is graded against the f32 reference of the Down.

    Random routing (the basic tests, test_stream_rows_limits, the benchmark) rarely gives an expert of exactly 16 k
    rows, runs one block size and tile per shape, and with topk > 2 can only be graded; no other case builds the Down
    without the table where the dispatch uses it.
    """
    for block_m in (16, 32, 64):
        kinds = set()
        for count in _ATOMIC_DOWN_COUNTS:
            for start in range(0, count, block_m):
                rows = min(block_m, count - start)
                kinds.add((-(-rows // 16), rows % 16 == 0))
        subs = range(1, block_m // 16 + 1)
        assert kinds == set(itertools.product(subs, (True, False))), (block_m, kinds)
    compile_down, seen = _spy_build(monkeypatch, "compile_a8w4_down_stream")
    w, prepared = _weights(len(_ATOMIC_DOWN_COUNTS), hidden, inter, 490)
    assert prepared.inter_padded == inter
    x, topk_weight, topk_ids, mq, ms, ref = _atomic_down_case(w, prepared, seed=495)
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_down_constants()
    kernels = []
    for block_m in (16, 32, 64):
        args = (hidden, inter, 2, block_m, prepared.experts, True, device, constants)
        # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
        moe_a8w4._stream_down_kernel.__wrapped__(*args)
        base = dict(seen)
        table = base.get("ROWS_LDS", False)
        assert base["ATOMIC"] and table == (block_m == 64 and inter == 256), base
        kernels.append((block_m, moe_a8w4._stream_down_kernel(*args)))
        if table:
            for other in ({"ROWS_LDS": False}, {"ROWS_LDS": False, "LDS_ACTS": False}):
                kernels.append((block_m, compile_down(**{**base, **other})))
    outs = []
    for block_m, kernel in kernels:
        out = torch.full_like(x, float("nan"))
        sorted_ids, sorted_weights, sorted_experts, num_valid, _, _, _ = (
            moe_a8w4._stream_prep(
                x, topk_weight, topk_ids, prepared.experts, block_m, out, zero=True
            )
        )
        moe_a8w4._launch(
            "down",
            kernel,
            mq,
            prepared.w2,
            prepared.w2_delta,
            out,
            sorted_ids,
            sorted_weights,
            sorted_experts,
            num_valid,
            prepared.w2_scale,
            ms,
            x.shape[0],
            sorted_experts.numel(),
        )
        outs.append(out)
    assert_close(outs[0], ref)
    for out in outs[1:]:
        assert torch.equal(_bits(outs[0]), _bits(out))


def test_stream_down_lds_budget(monkeypatch):
    """``stream_a8w4.down_stream_lds``, the LDS estimate by which the dispatch shares a block's activation rows through
    LDS (``_stream_down_shared``), equals what ``compile_a8w4_down_stream`` lays out and asserts against the 64 KiB of
    a CU, for the three tiles (R 8 x W 4 and R 4 x W 8 sharing, R 16 x W 1), with and without atomics and the rows'
    table (ROWS_LDS): a launcher builds where the estimate fits and fails with exactly that byte count where it does
    not. The eight-wave tile lays out what the dispatch estimates for the four-wave one (W * R channel blocks the
    same). At I 1024 (H 512) the dispatch shares the rows of the atomic Down with blocks of 32 but not 16 or 64 and
    builds all three; blocks of 64 at I 256 also fit the table, and at I 960, the largest padded I whose block of 64
    rows fits, take the eight-wave Down: so from padded I 512 on streaming goes up to STREAM_MAX_ROWS rows per expert
    at I 960 and STREAM_BM32_ROWS at I 1024 (``_stream_max_rows``, both limits set here). Building a launcher does not
    trace it.

    The basic tests and the benchmark build only the dispatch's tile of their shapes, within the budget, and have no
    padded I between 576 and 1024; no case builds one past it, where an estimate below the kernel's layout would fail
    only when the launcher is built (test_shape_fallbacks checks the estimate of the default tile without atomics
    against the path its call takes).
    """
    compile_down, seen = _spy_build(monkeypatch, "compile_a8w4_down_stream")
    for (K, block_m), (R, W), atomic, table in itertools.product(
        ((1024, 16), (1024, 32), (1024, 64), (960, 64), (256, 64)),
        ((8, 4), (4, 8), (16, 1)),
        (False, True),
        (False, True),
    ):
        lds = stream_a8w4.down_stream_lds(
            K, block_m, W=W, R=R, ATOMIC=atomic, ROWS_LDS=table
        )
        if (R, W) == (4, 8):
            assert lds == stream_a8w4.down_stream_lds(
                K, block_m, W=4, R=8, ATOMIC=atomic, ROWS_LDS=table
            ), (K, block_m)
        options = {"N": 512, "K": K, "TOPK": 2, "BLOCK_M": block_m, "E": 5}
        options.update(R=R, W=W, ATOMIC=atomic, ROWS_LDS=table)
        if lds <= common_a8w4.LDS_BYTES:
            compile_down(**options)
        else:
            with pytest.raises(AssertionError, match=f"Down needs {lds} B of LDS"):
                compile_down(**options)
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_down_constants()
    for K, block_m, waves in (
        (1024, 16, 1),
        (1024, 32, 4),
        (1024, 64, 1),
        (960, 64, 8),
    ):
        assert moe_a8w4._stream_down_shared(512, K, block_m) == (waves > 1), block_m
        # Builds the dispatch's atomic launcher uncached (nothing is launched with it).
        moe_a8w4._stream_down_kernel.__wrapped__(
            512, K, 2, block_m, 5, True, device, constants
        )
        assert seen["ATOMIC"] and seen.get("W", 1) == waves, seen
    monkeypatch.setattr(moe_a8w4, "STREAM_MAX_ROWS", 44)
    monkeypatch.setattr(moe_a8w4, "STREAM_BM32_ROWS", 28)
    assert moe_a8w4._stream_max_rows(512, 960) == 44
    assert moe_a8w4._stream_max_rows(512, 1024) == 28


@pytest.mark.parametrize(
    "hidden, inter", [(4096, 512), (5120, 576)], ids=["h4096_i512", "h5120_i576"]
)
def test_stream_down_eight_waves_bitexact(monkeypatch, hidden, inter):
    """Blocks of 64 from K 512 on, where the block's activation rows fit in LDS (``_stream_down_shared``), take the
    eight-wave Down: R 4 x W 8 reading the rows from LDS at every 64-K step (LDS_ACTS) instead of R 8 x W 4 holding
    them in VGPRs; a workgroup covers the same 32 channel blocks with the same loads and MFMA chains per block, so only
    the wave that computes a block and where its activations come from change. On one sort of the routes of
    _ATOMIC_DOWN_COUNTS (E 8, topk 2: blocks of 64 with every live sub-block count, with and without padding) the
    dispatch's launchers (asserted: R4W8 with LDS_ACTS, with and without atomics) give the bits of R8W4 with the rows
    in VGPRs, both writing the routes by sorted row and adding them atomically (two adds onto zero per token, whose
    sum does not depend on their order); the atomic output is graded against the f32 reference of the Down. H 4096
    with I 512 and H 5120 with I 576: the routed shapes of MiMo TP4 and of the DeepSeek / MTP TP4 layers.

    The benchmark's MiMo TP4 M1024 and MTP TP4 layer M1280 / 1365 run the eight-wave Down graded end to end only,
    which does not show a few differing elements, and no dispatch runs the four-wave tile there.
    """
    compile_down, seen = _spy_build(monkeypatch, "compile_a8w4_down_stream")
    w, prepared = _weights(len(_ATOMIC_DOWN_COUNTS), hidden, inter, 500)
    assert prepared.inter_padded == inter
    x, topk_weight, topk_ids, mq, ms, ref = _atomic_down_case(w, prepared, seed=505)
    E, tokens = prepared.experts, x.shape[0]
    sorted_ids, sorted_weights, sorted_experts, num_valid, loc_ids, _, _ = (
        moe_a8w4._stream_prep(
            x, topk_weight, topk_ids, E, 64, torch.empty_like(x), zero=False
        )
    )
    task_num = sorted_experts.numel()
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_down_constants()
    for atomic in (False, True):
        args = (hidden, inter, 2, 64, E, atomic, device, constants)
        # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
        moe_a8w4._stream_down_kernel.__wrapped__(*args)
        base = dict(seen)
        assert (base["R"], base["W"], base["LDS_ACTS"]) == (4, 8, True), base
        four = {**base, "R": 8, "W": 4, "LDS_ACTS": False}
        outs = []
        for kernel in (moe_a8w4._stream_down_kernel(*args), compile_down(**four)):
            shape = x.shape if atomic else (task_num * 64, hidden)
            outs.append(
                torch.zeros(shape, dtype=torch.bfloat16)
                if atomic
                else torch.full(shape, float("nan"), dtype=torch.bfloat16)
            )
            moe_a8w4._launch(
                "down",
                kernel,
                mq,
                prepared.w2,
                prepared.w2_delta,
                outs[-1],
                sorted_ids,
                sorted_weights,
                sorted_experts,
                num_valid,
                prepared.w2_scale,
                ms,
                tokens,
                task_num,
            )
        if atomic:
            assert_close(outs[0], ref)
        else:
            live = outs[0][loc_ids.flatten().long()]
            assert not bool(live.isnan().any()) and bool((live != 0).any())
        assert torch.equal(_bits(outs[0]), _bits(outs[1])), atomic


@pytest.mark.parametrize(
    "tokens, block_m", [(62, 32), (120, 64)], ids=["m62_bm32", "m120_bm64"]
)
def test_stream_gateup_tiles_bitexact(monkeypatch, tokens, block_m):
    """The streaming Gate/Up's light waves (one gate and one up block each, W waves sharing the block's activations
    through an LDS ring) give the intermediate bits of the unshared tile (4 gate blocks per wave, W 1) on one sort and
    input: in every tile a lane loads two adjacent rows of a gate block (lanes 0-7 of 16) or of its up block (lanes
    8-15) with one 16-byte load, MFMA chain p takes row p of the pair, and the epilogue swaps the pairs back between
    lanes l and l ^ 32 (ds_bpermute) into 4 channels of gate and up per lane. H 4096 with I 256 (E 5, topk 2): M 62 /
    120 sort in blocks of 32 / 64 and take the dispatch's 8 waves (I % 128 == 0, asserted), compared with 4 waves and
    the unshared tile; up to 2 / 4 live sub-blocks. With blocks of 64 also the launcher the dispatch builds from one
    block of rows per expert on (``reuse``: the weights loaded cached, W_AUX 0 instead of STREAM_GATEUP_W_AUX, all
    other options the same, asserted): the load policy does not change the bits.

    The 4-wave tile of H 5120 is compared with ACT_STEPS 1 by test_stream_act_steps_bitexact, the tail-quantizing tile
    of 5 gate blocks per wave (TP2, activations not shared) with the unshared tile by test_stream_fused_quant_bitexact;
    the basic tests and the benchmark (MiMo TP8 M512 / 1024 take 8 waves, M2048-8192 the cached loads) only grade end
    to end, which does not show a few differing elements.
    """
    compile_gateup, seen = _spy_build(monkeypatch, "compile_a8w4_gateup_stream")
    _, prepared = _small_weights(4096, 256)
    E, inter = prepared.experts, prepared.inter_padded
    assert moe_a8w4._stream_block_m(tokens, 2, E, 4096) == block_m
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_gateup_constants()
    args = (2 * inter, 4096, 2, block_m, E, 0.0, device, constants)
    # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
    moe_a8w4._stream_gateup_kernel.__wrapped__(*args)
    base = dict(seen)
    assert (base["G"], base["W"], base.get("QUANT", False)) == (1, 8, False), base
    assert base["W_AUX"] == moe_a8w4.STREAM_GATEUP_W_AUX, base
    kernels = [
        moe_a8w4._stream_gateup_kernel(*args),
        compile_gateup(**{**base, "W": 4}),
        compile_gateup(**{**base, "G": 4, "W": 1}),
    ]
    if block_m == 64:
        moe_a8w4._stream_gateup_kernel.__wrapped__(*args, reuse=True)
        assert seen == {**base, "W_AUX": 0}, seen
        kernels.append(moe_a8w4._stream_gateup_kernel(*args, reuse=True))
    x, topk_weight, topk_ids = make_inputs(
        tokens, 4096, E, 2, seed=490 + tokens, shared=False
    )
    sorted_ids, _, sorted_experts, num_valid, _, xq, xs = moe_a8w4._stream_prep(
        x, topk_weight, topk_ids, E, block_m, torch.empty_like(x), zero=False
    )
    mids = []
    for kernel in kernels:
        mids.append(torch.full((tokens, 2, inter), float("nan")))
        moe_a8w4._launch(
            "gateup",
            kernel,
            xq,
            prepared.w1,
            prepared.w1_delta,
            mids[-1],
            sorted_ids,
            sorted_experts,
            num_valid,
            prepared.w1_scale,
            xs,
            xs,
            tokens,
            sorted_experts.numel(),
        )
    assert not bool(mids[0].isnan().any())
    for mid in mids[1:]:
        assert torch.equal(mids[0].view(torch.int32), mid.view(torch.int32))


# case -> (M, tile options, set thresholds, kernels in order, the prefill kernels' (Gate/Up XCD_CHUNK, Down
# TWO_PASS, Down XCD_CHUNK) or None for no prefill kernel; a TWO_PASS of None: the dispatch's, both forced compared);
# DeepSeek H5120, E24, topk 6, I576, each threshold set to exactly one side of its boundary (M=96: 576 routes, exactly
# the default STREAM_MAX_ROWS_H5120 = 24 rows per expert; M=97: 582 routes, 24.25 rows; M=3: 18 routes and 18 x 4 = 72
# estimated Down workgroups; M=4: 24 routes and 96). The streaming cases follow the H 5120 row limit (M = limit * E /
# topk; blocks of 32 from 13 to 28 rows per expert), their Down adding the routes atomically.
_H5120_ROWS_M = moe_a8w4.STREAM_MAX_ROWS_H5120 * DS_EXPERTS // DS_TOPK
_H5120_PREFILL_LIMITS = {"XCD_MIN_ROUTES_H5120": 24}
_H5120_STREAM_BM32 = ["a8w4_stream_prep"] + _stream_path(32, False)
_H5120_LIMIT_CASES = {
    "rows_limit_stream": (
        _H5120_ROWS_M,
        {},
        {"STREAM_H5120_MIN_ROUTES": _H5120_ROWS_M * DS_TOPK},
        _H5120_STREAM_BM32,
        None,
    ),
    "rows_over_limit_prefill": (
        _H5120_ROWS_M + 1,
        {},
        {"STREAM_H5120_MIN_ROUTES": (_H5120_ROWS_M + 1) * DS_TOPK},
        ["a8w4_stream_prep"] + _prefill_path(128),
        (0, True, 0),
    ),
    "routes_below_stream": (
        _H5120_ROWS_M + 1,
        {},
        {"STREAM_H5120_MIN_ROUTES": (_H5120_ROWS_M + 1) * DS_TOPK + 1},
        _H5120_STREAM_BM32,
        None,
    ),
    "prep_one_and_two_pass": (
        3,
        {"block_m": 64, "tile_k": 128},
        _H5120_PREFILL_LIMITS,
        ["a8w4_stream_prep"] + _prefill_path(128),
        (0, None, 0),
    ),
    "xcd_two_pass": (
        4,
        {"block_m": 64, "tile_k": 128},
        _H5120_PREFILL_LIMITS,
        ["a8w4_stream_prep"] + _prefill_path(128),
        (moe_a8w4.GATEUP_XCD_CHUNK_H5120, True, moe_a8w4.DOWN_XCD_CHUNK_H5120),
    ),
}


@pytest.mark.parametrize("case", list(_H5120_LIMIT_CASES))
def test_h5120_dispatch_limits(monkeypatch, case):
    """The dispatch thresholds that only apply from hidden 5120 on, on both sides of each boundary, with the kernels,
    the build parameters of the prefill kernels and the end-to-end result: once M * topk reaches
    ``STREAM_H5120_MIN_ROUTES`` the streaming limit drops from ``STREAM_MAX_ROWS_WIDE`` (48) to
    ``STREAM_MAX_ROWS_H5120`` rows per expert (24 by default: M 96 with exactly 24 rows still streams, M 97 with
    24.25 takes prefill, and with the threshold one route above M 97 still streams, the limit 48 again); prefill
    Gate/Up and Down chunk their grids by XCD from ``XCD_MIN_ROUTES_H5120`` routes on. Down
    stages its activation tile in two passes with many estimated workgroups (M 4: 96; M 97: 96): M 3
    (72) takes the Down's pass count as the dispatch decides, and the same call with one pass and with two passes
    forced gives the same bits (prep sort, routes and a8w4_route_sum unchanged).

    The real thresholds (4096 and 16384 routes) need M >= 683 on E 24, where both sides are prefill, so they
    are moved onto the M 3-97 boundaries to keep this fast; the benchmark's ds_tp4 runs E 384 at powers of two and 48
    without asserting kernels or build parameters (M 1024: 16 rows, M 2048: 32); the basic tests are not on these
    boundaries.
    """
    tokens, options, limits, path, built = _H5120_LIMIT_CASES[case]
    for name, value in limits.items():
        monkeypatch.setattr(moe_a8w4, name, value)
    gateup, down = [], []
    _spy(monkeypatch, "_gateup_kernel", gateup)
    _spy(monkeypatch, "_down_kernel", down)
    w, prepared = _ds_weights()
    x, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=440 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(path)
    if built is None:
        assert not gateup and not down, (gateup, down)
    else:
        got = (gateup[-1][8], down[-1][6], down[-1][7])
        assert all(b is None or g == b for g, b in zip(got, built)), (gateup, down)
    assert_close(out, ref)
    if built is not None and built[1] is None:
        build, outs = moe_a8w4._down_kernel, []
        for two_pass in (False, True):
            with monkeypatch.context() as patch:
                patch.setattr(
                    moe_a8w4,
                    "_down_kernel",
                    lambda *a, two=two_pass, **k: build(*a[:6], two, *a[7:], **k),
                )
                outs.append(_a8w4_run(w, prepared, x, topk_weight, topk_ids, **options))
            _assert_path(path)
            assert down[-1][6] is two_pass, down
            assert_close(*outs[-1])
        assert torch.equal(_bits(outs[0][0]), _bits(outs[1][0]))


def test_gateup_few_waves_grid(monkeypatch):
    """The prefill Gate/Up keeps GATEUP_FEW_WAVES workgroups per CU (gemm1_a8w4 WGS_PER_CU) and deals XCD chunks of
    GATEUP_FEW_WAVES_XCD_CHUNK sorted blocks (``_gateup_grid``) only from hidden 5120 on with at least
    GATEUP_FEW_WAVES_MIN_E experts and GATEUP_FEW_WAVES_MIN_ROUTES routes and the default tile (64, 128, 128), each on
    both sides: the DeepSeek-V4.1 main layer (E 384, topk 6) from M 16384 on, not the MTP layer (E 128); elsewhere
    the XCD chunk of ``_xcd_chunk`` and no limit. A call passes both to the Gate/Up's launcher: with the thresholds
    moved onto DeepSeek E 24 (24 routes, the tile given), M 4 builds the limited Gate/Up and M 3 (18 routes) the plain
    one, both graded.

    The benchmark runs the limited Gate/Up only from ds_tp4 / ds_layer M 16384 on and checks neither the boundary nor
    the build parameters; the basic tests and the other cases never reach 98304 routes.
    """
    few = (moe_a8w4.GATEUP_FEW_WAVES_XCD_CHUNK, moe_a8w4.GATEUP_FEW_WAVES)
    tile, grid = (64, 128, 128), moe_a8w4._gateup_grid
    experts = moe_a8w4.GATEUP_FEW_WAVES_MIN_E
    routes = moe_a8w4.GATEUP_FEW_WAVES_MIN_ROUTES
    plain = (moe_a8w4._xcd_chunk(moe_a8w4.GATEUP_XCD_CHUNK_H5120, 5120, routes), 0)
    assert grid(5120, experts, routes, tile) == few
    assert grid(5120, experts - 1, routes, tile) == plain
    assert grid(5120, experts, routes - 1, tile) == plain
    for other in ((64, 128, 256), (128, 128, 128), (64, 256, 128)):
        assert grid(5120, experts, routes, other) == plain, other
    assert grid(4096, experts, routes, tile) == (0, 0)
    assert grid(5120, 384, 16384 * 6, tile) == few
    assert grid(5120, 384, 16383 * 6, tile) == plain
    assert grid(5120, 128, 65536 * 3, tile) == plain
    monkeypatch.setattr(moe_a8w4, "GATEUP_FEW_WAVES_MIN_E", DS_EXPERTS)
    monkeypatch.setattr(moe_a8w4, "GATEUP_FEW_WAVES_MIN_ROUTES", 4 * DS_TOPK)
    original, built = moe_a8w4._gateup_kernel, []

    def spy(*args, **options):
        built.append((args[8], options.get("wgs_per_cu", 0)))
        return original(*args, **options)

    monkeypatch.setattr(moe_a8w4, "_gateup_kernel", spy)
    w, prepared = _ds_weights()
    for tokens, expected in ((4, few), (3, (0, 0))):
        x, topk_weight, topk_ids = make_inputs(
            tokens, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=460 + tokens, shared=False
        )
        out, ref = _a8w4_run(
            w, prepared, x, topk_weight, topk_ids, block_m=64, tile_k=128
        )
        _assert_path(_prefill_path(128))
        assert built == [expected], (tokens, built)
        built.clear()
        assert_close(out, ref)


@pytest.mark.parametrize(
    "hidden, inter, tokens, options, path, shared, max_rows",
    [
        # H 4096 decodes up to DECODE_MAX_M = 7 tokens, with the Down that quantizes its rows at every M.
        (4096, 64, 7, {}, _DECODE_PATHS["nn"], None, None),
        (4608, 64, 7, {}, _prefill_path(256), None, None),
        (512, 1024, 4, {}, _DECODE_PATHS["np"], None, None),
        (512, 1088, 4, {}, _DECODE_PATHS["nn"], None, None),
        (512, 1024, 300, {"block_m": 64}, _prefill_path(256), None, None),
        (512, 1024, 48, {}, _stream_path(32, False), True, None),
        (512, 1024, 300, {}, _prefill_path(128), False, 120),
    ],
    ids=[
        "h4096_decode",
        "h4608",
        "i1024_pair_down",
        "i1088_single_down",
        "i1024",
        "i1024_bm32_shared",
        "i1024_bm64_unshared",
    ],
)
def test_shape_fallbacks(
    monkeypatch, hidden, inter, tokens, options, path, shared, max_rows
):
    """Shapes outside the target set still run and are right, on the path the kernels' limits and the measured
    ranges give: the decode kernels take hidden <= 4096 (H 4096 with I 64, I_pad % 128 != 0: the per-route decode,
    M 7) and the measured H 5120 only, so hidden 4608 = 9 x 512 (an odd count of 512-column Gate/Up steps; the target
    shapes have 5, 8 and 10) falls back to prefill (M 7: within the decode tier's token counts); the slot-pair Down needs a row's I_pad / 16 16-byte pieces in its
    64 * WAVES lanes (topk 2: one wave), so I 1088 takes the Down that quantizes its rows where I 1024 still pairs; I 1024 at
    M 300 (block_m=64 selects the prefill kernels, which pyhip M17 runs with the K tile 128 by default) fills Down's
    activation tile with exactly 64 KiB of LDS in one pass, the hidden dimension split in 4. The streaming Down lets
    4 waves share the block's activation rows only when they fit in the 64 KiB of LDS (``shared``): I 1024 fits with
    blocks of 32 (M 48) but not of 64, which the dispatch does not stream at I 1024: M 300 with STREAM_MAX_ROWS
    raised to 120 rows per expert still takes prefill past STREAM_BM32_ROWS (graded; the tile estimate of blocks of
    64 at I 1024 against the kernel's layout is test_stream_down_lds_budget's).

    The basic tests and the benchmark only use the target shapes (H 2560 / 4096 / 5120, I_pad 192-576; H 4096 only
    with I_pad % 128 == 0), out of reach of these limits; E 5 and topk 2 keep this fast.
    """
    if max_rows is not None:
        monkeypatch.setattr(moe_a8w4, "STREAM_MAX_ROWS", max_rows)
    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = make_inputs(tokens, hidden, 5, 2, seed=tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(path)
    assert_close(out, ref)
    if shared is not None:
        block_m = moe_a8w4._stream_block_m(tokens, 2, 5)
        lds = stream_a8w4.down_stream_lds(inter, block_m, W=4)
        assert (lds <= common_a8w4.LDS_BYTES) == shared, lds


def test_decode_down_needs_hidden_multiple_of_32():
    """The Down decode that quantizes its rows, routed-only and mixed, owns 32 output channels per workgroup (row-block pairs:
    one 16-byte load per lane covers two 16-row blocks) and refuse a hidden size H % 32 != 0 when built
    (AssertionError, before compiling); the decode tier never asks for one, as it needs gateup_decode_fits
    (H % 512 == 0).

    Every H of the basic tests, the benchmark and the other cases is a multiple of 512.
    """
    decode = moe_a8w4.decode_a8w4
    single = functools.partial(decode.compile_a8w4_down_decode_mixed, SHARED=False)
    for hidden in (DS_HIDDEN + 16, MIMO_HIDDEN + 48):
        assert not decode.gateup_decode_fits(hidden)
        with pytest.raises(AssertionError, match="unsupported Down decode shape"):
            single(N=hidden, K=DS_INTER, TOPK=DS_TOPK)
        with pytest.raises(AssertionError, match="unsupported mixed Down decode"):
            decode.compile_a8w4_down_decode_mixed(
                N=hidden, K=DS_INTER, KP=640, TOPK=DS_TOPK + 1, E=8
            )
    assert all(
        h % 32 == 0 for h in range(16, 16385, 16) if decode.gateup_decode_fits(h)
    )


# -- 2.5 CUDA graph -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, path",
    [
        ({}, ["a8w4_stream_prep"] + _stream_path(32, True)),
        ({"block_m": 64}, ["a8w4_stream_prep"] + _prefill_path(256)),
    ],
    ids=["stream", "prefill"],
)
def test_graph_replay(options, path):
    """The whole call can be captured in a CUDA graph (no host synchronization) and replayed, and the replay reads
    newly written inputs (the workspaces allocated in the call belong to the graph's pool). The shape is compiled
    by an eager call outside the capture first. Qwen M 64 streams by default (the prep, blocks of 32, the Gate/Up
    quantizing the intermediate, the atomic Down); block_m=64 selects the prefill kernels (K tile 256) behind the
    same prep (pyhip M17 has both).

    The basic tests and the benchmark call eagerly and never capture a graph.
    """
    w, prepared = _weights(33, HIDDEN, 320, 51)
    x, topk_weight, topk_ids = make_inputs(64, HIDDEN, 33, TOPK, seed=52)
    out = torch.empty_like(x)
    flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out, **options)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out, **options)
    _assert_path(path)
    x2, topk_weight2, topk_ids2 = make_inputs(64, HIDDEN, 33, TOPK, seed=53)
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = reference(
        x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2
    )
    assert_close(out, ref)


@pytest.mark.parametrize(
    "tokens, path",
    [(1, "nn"), (5, "np")],
    ids=["m1_nn", "m5_np"],
)
def test_decode_graph_replay(tokens, path):
    """Both dispatches of the Qwen-shape decode path can be captured in a CUDA graph and replayed, and the replay
    reads newly written inputs (mid / mq allocated in the call belong to the graph's pool). The shape is compiled by an
    eager call outside the capture first.

    The basic tests and the benchmark call eagerly; test_graph_replay captures the streaming and prefill paths, the
    layer graph cases the decode with x pre-quantized (the fold).
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(
        tokens, HIDDEN, EXPERTS, TOPK, seed=97 + tokens
    )
    out = torch.empty_like(x)
    flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out)
    _assert_path(_DECODE_PATHS[path])
    x2, topk_weight2, topk_ids2 = make_inputs(
        tokens, HIDDEN, EXPERTS, TOPK, seed=98 + tokens
    )
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = reference(
        x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2
    )
    assert_close(out, ref)


def test_decode_dedup_graph_replay():
    """The H 5120 deduplicated Gate/Up (DeepSeek TP4, E 24, M 8) can be captured in a CUDA graph and replayed: the plan
    kernel, the deduplicated Gate/Up and the slot-pair Down writing the output are all in captured kernels, so the
    replay plans the newly written routes again (plan / mid / xq allocated in the call belong to the graph's pool). The
    shape is compiled by an eager call outside the capture first; the case pins its kernels within the H 5120 decode
    limit (_assert_h5120).

    The basic tests and the benchmark call eagerly; test_decode_graph_replay captures the per-route decode only.
    """
    tokens = 8
    (w, prepared), hidden, experts, topk = (
        _ds_weights(),
        DS_HIDDEN,
        DS_EXPERTS,
        DS_TOPK,
    )
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=260 + tokens, shared=False
    )
    out = torch.empty_like(x)
    flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, output=out)
    _assert_h5120(tokens, _DEDUP_PATHS["wide"])
    x2, topk_weight2, topk_ids2 = make_inputs(
        tokens, hidden, experts, topk, seed=261 + tokens, shared=False
    )
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    ref = reference(
        x2, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight2, topk_ids2
    )
    assert_close(out, ref)


# -- 2.6 input errors, the gfx942 gate and compile-cache keys --------------------------------------


def test_rejects_bad_inputs(monkeypatch):
    """Unsupported inputs raise: weight / scale dtypes and shapes (w1 not 3-D, scales of another size), E8M0 NaN
    codes, H or I misaligned, weights on several devices; a weight not from prepare, non-BF16 or non-contiguous
    activations, a hidden size that does not match, routing tensors of another shape or dtype (ids not int32 / int64,
    weights not floating), an output of the wrong dtype or layout, tensors on other devices, token counts past the
    32-bit byte offsets; a Down BLOCK_M * I_pad that is no multiple of 4096 (BM 32 with I 320 / 192 fails when the
    kernel is built instead of computing wrong values). swiglu_limit: negative, NaN or +-inf raise ValueError, a
    non-real TypeError, and 0 / -0.0 turn the limit off (bit-identical to not passing it).

    The basic tests and the benchmark only pass valid inputs, the default block_m 64 and swiglu_limit 0 or 10.
    """
    w = make_weights(3, 512, 64, seed=61)
    args = [w["w1"], w["w1_scale"], w["w2"], w["w2_scale"]]

    def prepare_with(index, value):
        bad = list(args)
        bad[index] = value
        return prepare_moe_a8w4_ptpc_weight(*bad)

    with pytest.raises(TypeError):
        prepare_with(0, w["w1"].view(torch.uint8).view(torch.int8))
    with pytest.raises(TypeError):
        prepare_with(1, w["w1_scale"].view(torch.uint8).float())
    with pytest.raises(ValueError, match="w1 must be"):
        prepare_with(0, w["w1"][0])
    with pytest.raises(ValueError, match="expected MXFP4 scales"):
        prepare_with(1, w["w1_scale"][:, :, :8])
    with pytest.raises(ValueError):
        prepare_with(2, w["w2"][:, :, :16])
    with pytest.raises(ValueError):
        prepare_with(0, w["w1"][:, :96])  # I = 48 is no multiple of 32
    nan_scale = w["w2_scale"].view(torch.uint8).clone()
    nan_scale[0, 0, 0] = 255
    with pytest.raises(ValueError):
        prepare_with(3, nan_scale)
    with pytest.raises(ValueError, match="one GPU"):
        prepare_with(3, w["w2_scale"].cpu())
    small = make_weights(3, 256, 64, seed=62)
    with pytest.raises(ValueError):
        prepare_moe_a8w4_ptpc_weight(
            small["w1"], small["w1_scale"], small["w2"], small["w2_scale"]
        )
    prepared = prepare_moe_a8w4_ptpc_weight(*args)
    x, topk_weight, topk_ids = make_inputs(4, 512, 3, 2, seed=63)
    run = functools.partial(
        flydsl_moe_a8w4_ptpc, topk_weight=topk_weight, topk_ids=topk_ids
    )
    with pytest.raises(TypeError):
        run(x, w)
    with pytest.raises(TypeError):
        run(x.half(), prepared)
    with pytest.raises(ValueError):
        run(x[:, :256].contiguous(), prepared)
    with pytest.raises(ValueError, match="contiguous"):
        run(torch.empty((4, 1024), dtype=torch.bfloat16)[:, :512], prepared)
    with pytest.raises(ValueError):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight[:2], topk_ids[:2])
    with pytest.raises(TypeError, match="int32 / int64"):
        run(x, prepared, topk_ids=topk_ids.to(torch.int16))
    with pytest.raises(TypeError, match="int32 / int64"):
        run(x, prepared, topk_weight=topk_weight.to(torch.int32))
    with pytest.raises(TypeError):
        run(x, prepared, output=torch.empty((4, 512)))
    with pytest.raises(ValueError, match="contiguous"):
        run(x, prepared, output=torch.empty((512, 4), dtype=torch.bfloat16).T)
    with pytest.raises(ValueError, match="one GPU"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight.cpu(), topk_ids.cpu())
    monkeypatch.setattr(moe_a8w4, "_MAX_BYTES", 4 * 2 * 64 * 4)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        run(x, prepared)
    monkeypatch.undo()
    for limit in (-1.0, -1e-30, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            run(x, prepared, swiglu_limit=limit)
    for limit in ("10", None, 10j, torch.tensor(10.0)):
        with pytest.raises(TypeError):
            run(x, prepared, swiglu_limit=limit)
    off = run(x, prepared)
    for limit in (0, -0.0):
        assert torch.equal(run(x, prepared, swiglu_limit=limit), off)
    for inter in (320, 192):
        with pytest.raises(AssertionError, match="4096"):
            gemm2_a8w4.compile_a8w4_gemm2(
                N=HIDDEN, K=inter, TOPK=TOPK, BLOCK_TILE_SIZE_M=32
            )


def test_requires_gfx942(monkeypatch):
    """Off gfx942 the public entries raise RuntimeError before any kernel runs (FP8 E4M3FNUZ, the MFMA tiles and
    the MI308X tuning are gfx942-only): prepare, flydsl_moe_a8w4_ptpc and aiter.fused_moe on the tagged weights.

    The basic tests and the benchmark run on gfx942 only.
    """
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(4, 512, 5, 2, seed=64, shared=False)
    monkeypatch.setattr(moe_a8w4, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942"):
        prepare_moe_a8w4_ptpc_weight(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])
    with pytest.raises(RuntimeError, match="gfx942"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
    with pytest.raises(RuntimeError, match="gfx942"):
        _fused_moe(x, prepared.w1, prepared.w2, topk_weight, topk_ids)


def test_compile_cache_keys(monkeypatch):
    """The launcher caches are keyed by the device (a compiled FlyDSL handle runs on one GPU only) and, for the GEMM
    kernels, by the kernel-module constants read while building and tracing (``common_a8w4.XCDS``,
    ``gemm2_a8w4.A_TWO_PASS_MIN_K``): another device or another constant gets its own launcher, and a call passes
    its inputs' device and the current constants. The Gate/Up limited to GATEUP_FEW_WAVES workgroups per CU
    (``wgs_per_cu``, ``_gateup_grid``) has a launcher of its own too. Building a launcher does not trace it, so the
    other device need not exist. The call passes block_m=64 to run the prefill launchers (pyhip M17 decodes M 16 by
    default), a8w4_route_sum's (``_stream_sum_kernel``, keyed by the device too: test_stream_compile_cache_keys)
    included.

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well; the benchmark builds the limited Gate/Up only for its M >= 16384 DeepSeek cases, never the
    unlimited one of the same shape.
    """
    constants = moe_a8w4._trace_constants()
    gateup = (512, 512, 2, 64, 128, 256, 5, 0.0, 0)
    down = (512, 64, 2, 64, 5, 1, False, 0)
    first = moe_a8w4._gateup_kernel(*gateup, 0, constants)
    assert moe_a8w4._gateup_kernel(*gateup, 0, constants) is first
    assert moe_a8w4._gateup_kernel(*gateup, 1, constants) is not first
    # The K tile 128: three workgroups' LDS fit a CU.
    few_args = (*gateup[:5], 128, *gateup[6:], 0, constants)
    few = {"wgs_per_cu": moe_a8w4.GATEUP_FEW_WAVES}
    limited = moe_a8w4._gateup_kernel(*few_args, **few)
    assert moe_a8w4._gateup_kernel(*few_args, **few) is limited
    assert moe_a8w4._gateup_kernel(*few_args) is not limited
    assert moe_a8w4._down_kernel(*down, 1, constants) is not moe_a8w4._down_kernel(
        *down, 0, constants
    )
    assert moe_a8w4._row_quant_kernel(512, "bf16", 1) is not moe_a8w4._row_quant_kernel(
        512, "bf16", 0
    )
    for module, name, value in (
        (common_a8w4, "XCDS", 8),
        (gemm2_a8w4, "A_TWO_PASS_MIN_K", 256),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(module, name, value)
            assert moe_a8w4._trace_constants() != constants
            assert (
                moe_a8w4._gateup_kernel(*gateup, 0, moe_a8w4._trace_constants())
                is not first
            )
    calls = {
        name: []
        for name in (
            "_gateup_kernel",
            "_down_kernel",
            "_row_quant_kernel",
            "_stream_sum_kernel",
        )
    }
    for name, record in calls.items():
        _spy(monkeypatch, name, record)
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(16, 512, 5, 2, seed=65, shared=False)
    assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids, block_m=64))
    device = x.device.index
    assert [args[-2:] for args in calls["_gateup_kernel"] + calls["_down_kernel"]] == [
        (device, constants)
    ] * 2
    # x quantized by the prep, mid by the row kernel.
    assert [args[-1] for args in calls["_row_quant_kernel"]] == [device]
    assert calls["_stream_sum_kernel"] == [(2, 512, device)]


def _routed_decode_kernels(N1, H, I, topk, gateup_pq, down_pq, limit, device):
    """``moe_a8w4._decode_kernels`` (the routed experts' per-route decode Gate/Up and Down launchers) as the dispatch
    builds them."""
    return moe_a8w4._decode_kernels(N1, H, I, topk, gateup_pq, down_pq, limit, device)


def test_decode_compile_cache_keys(monkeypatch):
    """The decode launcher caches are keyed by the device (``_decode_kernels``, ``_row_quant_decode``; the decode
    kernels read no tunable module constant): another device gets its own launcher, and a call passes its inputs'
    device. Building a launcher does not trace it, so the other device need not exist.

    The basic tests and the benchmark run one GPU, where a cache keyed by shape alone passes as well;
    test_compile_cache_keys covers the prefill launchers.
    """
    decode = (128, 512, 64, 4, False, True, 0.0)
    first = moe_a8w4._decode_kernels(*decode, 0)
    assert moe_a8w4._decode_kernels(*decode, 0) is first
    assert moe_a8w4._decode_kernels(*decode, 1) is not first
    assert moe_a8w4._row_quant_decode(64, False, 1) is not moe_a8w4._row_quant_decode(
        64, False, 0
    )
    built, quant = [], []
    _spy(monkeypatch, "_decode_kernels", built)
    _spy(monkeypatch, "_row_quant_decode", quant)
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(4, 512, 5, 4, seed=67, shared=False)
    assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
    _assert_path(_DECODE_PATHS["np"])
    device = x.device.index
    assert [args[-1] for args in built] == [device], built
    assert [args[-1] for args in quant] == [device]


def test_decode_dedup_compile_cache_keys(monkeypatch):
    """The launcher caches of the H 5120 deduplicated Gate/Up are keyed by the device (``_row_quant_plan``,
    ``_gateup_dd_kernel`` and the slot-pair ``_down_pq_kernel``), the plan kernel also by its token limit, which a call
    takes from DECODE_WIDE_MAX_M; their other build parameters are arguments already (groups of 2 and 4 waves are
    fixed). Another device or limit gets its own launcher, and a call passes its inputs' device and the current limit
    (DeepSeek M 8 while H 5120 decodes M 8). Building a launcher does not trace it, so the other device need not exist.

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well; test_decode_compile_cache_keys covers the per-route decode launchers.
    """
    for name, args in (
        ("_row_quant_plan", (5120, 6, 32)),
        ("_gateup_dd_kernel", (1152, 5120, 6, 0.0)),
        ("_down_pq_kernel", (5120, 576, 6)),
    ):
        build = getattr(moe_a8w4, name)
        assert build(*args, 0) is build(*args, 0), name
        assert build(*args, 1) is not build(*args, 0), name
    build = moe_a8w4._row_quant_plan
    assert build(5120, 6, 16, 0) is not build(5120, 6, 32, 0)
    names = ("_row_quant_plan", "_gateup_dd_kernel", "_down_pq_kernel")
    calls = {name: [] for name in names}
    for name, record in calls.items():
        _spy(monkeypatch, name, record)
    w, prepared = _ds_weights()
    x, topk_weight, topk_ids = make_inputs(
        8, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=228, shared=False
    )
    assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
    _assert_h5120(8, _DEDUP_PATHS["wide"])
    device = x.device.index
    if 8 <= moe_a8w4.DECODE_WIDE_MAX_M:
        assert [args[-2:] for args in calls["_row_quant_plan"]] == [
            (moe_a8w4.DECODE_WIDE_MAX_M, device)
        ]
        assert [
            args[-1] for args in calls["_gateup_dd_kernel"] + calls["_down_pq_kernel"]
        ] == [device] * 2


def test_stream_compile_cache_keys(monkeypatch):
    """The streaming launcher caches are keyed by the device (``_stream_gateup_kernel``, ``_stream_down_kernel``,
    ``_stream_prep_kernel``, ``_stream_sum_kernel``) and, for Gate/Up and Down, by the module constants they read
    while building (``_stream_gateup_constants()``: STREAM_GATEUP_ACT_BYTES_H5120 / PREFETCH / SYNC / W_AUX;
    ``_stream_down_constants()``: STREAM_DOWN_PREFETCH / PREFETCH_SMALL / W_AUX / W_AUX_H5120 / LDS_ACTS and
    common_a8w4.LDS_BYTES, which decides the shared tile): another device or constant gets its own launcher, and a call
    passes its inputs' device and the current constants (H 512, I 1024, M 48: the prep, the light-wave Gate/Up and
    the shared atomic Down; no route_sum, which only the prefill tier runs, test_compile_cache_keys). Building a
    launcher does not trace it, so the other device need not exist.

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well; test_compile_cache_keys covers the prefill launchers.
    """
    gateup = (2 * 192, 2560, 11, 32, 33, 0.0)
    down = (2560, 192, 11, 32, 33, True)
    g0 = moe_a8w4._stream_gateup_constants()
    d0 = moe_a8w4._stream_down_constants()
    first = moe_a8w4._stream_gateup_kernel(*gateup, 0, g0)
    assert moe_a8w4._stream_gateup_kernel(*gateup, 0, g0) is first
    assert moe_a8w4._stream_gateup_kernel(*gateup, 1, g0) is not first
    assert moe_a8w4._stream_down_kernel(
        *down, 1, d0
    ) is not moe_a8w4._stream_down_kernel(*down, 0, d0)
    assert moe_a8w4._stream_prep_kernel(
        2560, 11, 33, 32, True, 1
    ) is not moe_a8w4._stream_prep_kernel(2560, 11, 33, 32, True, 0)
    assert moe_a8w4._stream_sum_kernel(11, 2560, 1) is not moe_a8w4._stream_sum_kernel(
        11, 2560, 0
    )
    for name, value in (
        ("STREAM_GATEUP_ACT_BYTES_H5120", 8),
        ("STREAM_GATEUP_PREFETCH", 4),
        ("STREAM_GATEUP_SYNC", 1),
        ("STREAM_GATEUP_W_AUX", 0),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(moe_a8w4, name, value)
            constants = moe_a8w4._stream_gateup_constants()
            assert constants != g0, name
            assert (
                moe_a8w4._stream_gateup_kernel(*gateup, 0, constants) is not first
            ), name
    for module, name, value in (
        (moe_a8w4, "STREAM_DOWN_PREFETCH", 2),
        (moe_a8w4, "STREAM_DOWN_PREFETCH_SMALL", 4),
        (moe_a8w4, "STREAM_DOWN_W_AUX", 2),
        (moe_a8w4, "STREAM_DOWN_W_AUX_H5120", 0),
        (moe_a8w4, "STREAM_DOWN_LDS_ACTS", 0),
        (common_a8w4, "LDS_BYTES", 32 * 1024),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(module, name, value)
            assert moe_a8w4._stream_down_constants() != d0, name
    names = (
        "_stream_gateup_kernel",
        "_stream_down_kernel",
        "_stream_prep_kernel",
        "_stream_sum_kernel",
    )
    calls = {name: [] for name in names}
    for name, record in calls.items():
        _spy(monkeypatch, name, record)
    w, prepared = _small_weights(512, 1024)
    x, topk_weight, topk_ids = make_inputs(48, 512, 5, 2, seed=48)
    assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
    _assert_path(["a8w4_stream_prep"] + _stream_path(32, False))
    assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path
    device = x.device.index
    assert [args[-2:] for args in calls["_stream_gateup_kernel"]] == [(device, g0)]
    assert [args[-2:] for args in calls["_stream_down_kernel"]] == [(device, d0)]
    assert [args[-1] for args in calls["_stream_prep_kernel"]] == [device]
    assert calls["_stream_sum_kernel"] == []


# -- 2.7 routing and output variants: repeated expert ids, in-place output, input dtypes, M = 0 -------


@pytest.mark.parametrize(
    "case", ["mimo_m5_nn", "wide_m24", "stream_atomic", "stream_i1024"]
)
def test_duplicate_ids_keep_every_route(monkeypatch, case):
    """Several slots of a token select the same expert on the H 4096 per-route decode (MiMo shape, E 12, M 5: the
    Down that quantizes its rows, one wave per slot), on the H 5120 deduplicated Gate/Up (the routes of one token may
    land in one group, and the two slots of one slot pair of the Down may be one expert) and on the streaming path
    (the prep gives every route its own sorted row; Qwen M 64 with I 320 and H 512 with I 1024, padded I >= 512, both
    add them atomically): every slot is computed once and the slots are summed, as the reference does.

    The generated routes never repeat an expert within a token, so the basic tests and the benchmark never get here;
    test_decode_duplicate_and_int64_ids covers the slot-pair decode Down, test_prefill_duplicate_ids_keep_every_route
    the prefill. The H 5120 case pins the H 5120 decode limit to pyhip M17's 32 tokens (so that M 24 stays on the
    dedup decode whatever the default limit), its kernels within it and the streaming kernels past it
    (_assert_h5120).
    """
    shared = False
    if case == "mimo_m5_nn":
        (w, prepared), hidden, experts, topk = (
            _mimo_small_weights(256),
            MIMO_HIDDEN,
            12,
            MIMO_TOPK,
        )
        tokens, path = 5, _DECODE_PATHS["nn"]
    elif case == "wide_m24":
        monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_MAX_M", 32)
        (w, prepared), hidden, experts, topk = (
            _ds_weights(),
            DS_HIDDEN,
            DS_EXPERTS,
            DS_TOPK,
        )
        tokens, path = 24, "wide"
    elif case == "stream_atomic":
        (w, prepared), hidden, experts, topk = (
            _decode_weights(320),
            HIDDEN,
            EXPERTS,
            TOPK,
        )
        tokens, path, shared = 64, ["a8w4_stream_prep"] + _stream_path(32, True), True
    else:
        (w, prepared), hidden, experts, topk = _small_weights(512, 1024), 512, 5, 2
        tokens, shared = 48, True
        path = ["a8w4_stream_prep"] + _stream_path(32, False)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=240 + tokens, shared=shared
    )
    topk_ids[::2, 1] = topk_ids[::2, 0]
    if topk > 2:
        topk_ids[::3, topk - 2] = topk_ids[::3, 0]
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "wide":
        _assert_h5120(tokens, _DEDUP_PATHS[path])
    else:
        _assert_path(path)
        assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path
    assert_close(out, ref)


def test_prefill_duplicate_ids_keep_every_route():
    """Two slots of a token select the same expert on the prefill tier (H 512, below the hidden sizes of the target
    shapes, which the prep sorts like every other): the prep gives each slot its own sorted row and loc_ids, Down
    writes both routes and a8w4_route_sum adds them; the result equals the reference that sums every slot.
    block_m=64 selects the prefill kernels (by default M 200 takes the K tile 128, as in pyhip M17).

    The generated routes never repeat an expert within a token, so the basic tests and the benchmark never get
    here; neither has H 512.
    """
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(200, 512, 5, 2, seed=320, shared=False)
    topk_ids[::2, 1] = topk_ids[::2, 0]
    assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids, block_m=64))
    _assert_path(["a8w4_stream_prep"] + _prefill_path(256))
    assert "a8w4_route_sum" in moe_a8w4.last_path, moe_a8w4.last_path


@pytest.mark.parametrize("case", ["per_route", "wide", "stream", "prefill"])
def test_inplace_output(case):
    """``output`` is ``hidden_states`` itself: the op reads all of x before it writes the output; the result equals
    the reference of the original input. The per-route decode (MiMo M 5) writes the output in its Down, after the
    Gate/Up has read x; H 5120 (DeepSeek M 8) only quantizes x in the plan kernel and the slot-pair Down writes the
    output last (pinned within the H 5120 decode limit, past it the streaming kernels, _assert_h5120); the streaming
    prep (Qwen M 64) zeroes each piece of an output row in the lane that has read that piece of x, before the atomic
    Down adds into it; on the prefill path (block_m=64) the front end reads x first and a8w4_route_sum writes last.

    The basic tests and the benchmark pass separate output buffers.
    """
    options = {}
    if case == "per_route":
        (w, prepared), hidden, experts, topk = (
            _mimo_small_weights(256),
            MIMO_HIDDEN,
            12,
            MIMO_TOPK,
        )
        x, topk_weight, topk_ids = make_inputs(
            5, hidden, experts, topk, seed=255, shared=False
        )
    elif case == "wide":
        (w, prepared), hidden, experts, topk = (
            _ds_weights(),
            DS_HIDDEN,
            DS_EXPERTS,
            DS_TOPK,
        )
        x, topk_weight, topk_ids = make_inputs(
            8, hidden, experts, topk, seed=258, shared=False
        )
    else:
        w, prepared = _weights(33, HIDDEN, 320, 51)
        x, topk_weight, topk_ids = make_inputs(64, HIDDEN, 33, TOPK, seed=250)
        if case == "prefill":
            options = {"block_m": 64}
    ref = reference(
        x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids
    )
    result = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, output=x, **options
    )
    assert result is x
    _assert_kernels()
    if case == "per_route":
        _assert_path(_DECODE_PATHS["nn"])
    elif case == "wide":
        _assert_h5120(x.shape[0], _DEDUP_PATHS["wide"])
    elif case == "stream":
        _assert_path(["a8w4_stream_prep"] + _stream_path(32, True))
    else:
        _assert_path(["a8w4_stream_prep"] + _prefill_path(256))
    assert_close(x, ref)


def _odd_output(rows, cols):
    """A BF16 [rows, cols] view 2 bytes past a 4-byte boundary (at an odd element) of a NaN-filled buffer, with
    _ODD_PAD sentinel elements before and after it: ``(buffer, view)``."""
    buf = torch.full((rows * cols + 2 * _ODD_PAD,), float("nan"), dtype=torch.bfloat16)
    view = buf[_ODD_PAD : _ODD_PAD + rows * cols].view(rows, cols)
    assert view.data_ptr() % 4 == 2, view.data_ptr()
    return buf, view


def _copies_odd_outputs():
    """Whether the op computes a call with an output off a 4-byte boundary into an aligned buffer and copies it
    (``output_copy``, since M6): on a tree without it the BF16x2 atomics would write past such an output, so the
    cases that pass one are skipped there instead of run."""
    return "output_copy" in inspect.getsource(moe_a8w4)


_ODD_PAD = 9
# case: (shape, M, options) of test_unaligned_output, one per way the op writes its output. Routed: "h512" (E 5, H 512,
# I 64, topk 2) per-route decode with the Down that quantizes its rows (M 1) and the slot-pair Down (M 8), prefill with
# a tile option (M 40); "qwen" (E 33, H 2560, I 320, topk 11) streaming with the atomic Down (M 64), prefill sorted by
# the prep (M 300); "mimo" (E 12, H 4096, I 256, topk 8) the deduplicated decode in two (M 5) and four (M 24) kernels.
# With the shared expert (_fold_weights): "small" the mixed decode (M 1), the fold on the decode tier (M 16) and on
# prefill with a tile option (M 2048); the MTP TP8 layer the fold on the streaming tier (M 64).
_UNALIGNED_CASES = {
    "h512_m1": ("h512", 1, {}),
    "h512_m8": ("h512", 8, {}),
    "h512_tile_m40": ("h512", 40, {"block_m": 64}),
    "qwen_m64": ("qwen", 64, {}),
    "qwen_m300": ("qwen", 300, {}),
    "mimo_m5": ("mimo", 5, {}),
    "mimo_m24": ("mimo", 24, {}),
    "layer_small_m1": ("small", 1, {}),
    "layer_small_m16": ("small", 16, {}),
    "layer_small_tile_m2048": ("small", 2048, {"block_m": 64}),
    "layer_mtp_tp8_m64": ("mtp_layer_tp8", 64, {}),
}


@pytest.mark.parametrize("case", list(_UNALIGNED_CASES))
def test_unaligned_output(case):
    """An output 2 bytes past a 4-byte boundary (a view at an odd BF16 element of a larger buffer, NaN sentinels on
    both sides), separate and as ``hidden_states`` itself (``output=hidden_states``), on every way the op writes its
    output (_UNALIGNED_CASES): the BF16x2 atomics of the streaming and the deduplicated decode Down add element pairs
    at 4-byte boundaries, so such a call computes into an aligned buffer and copies it into the output (one more step,
    ``output_copy``, the last; an aligned output takes none). The sentinels stay NaN, the output is graded against the
    reference and, where no Down adds atomically (_path_atomic), has the bits of the call into an aligned output.

    The basic tests and the benchmark pass outputs from torch.empty; test_inplace_output passes aligned inputs and
    test_layer_fold_unaligned_output an output 8 bytes past a 16-byte boundary (4-byte aligned).
    """
    if not _copies_odd_outputs():
        pytest.skip("this op writes such an output directly (before M6)")
    shape, tokens, options = _UNALIGNED_CASES[case]
    if shape in ("h512", "qwen", "mimo"):
        (w, prepared), hidden, experts, topk = {
            "h512": (_small_weights(512, 64), 512, 5, 2),
            "qwen": (_decode_weights(320), HIDDEN, EXPERTS, TOPK),
            "mimo": (_mimo_small_weights(256), MIMO_HIDDEN, 12, MIMO_TOPK),
        }[shape]
        x, topk_weight, topk_ids = make_inputs(
            tokens, hidden, experts, topk, seed=510 + tokens, shared=False
        )
        limit = 0.0
        ref = reference(
            x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], topk_weight, topk_ids
        )
    else:
        w, shared, prepared, topk, hidden = _fold_weights(shape)
        x, topk_weight, topk_ids = make_layer_inputs(
            tokens,
            hidden,
            prepared.experts,
            topk,
            seed=520 + tokens,
            shared_weight=None,
        )
        limit = LAYER_SWIGLU_LIMIT
        ref = reference_layer(x, w, shared, topk_weight, topk_ids, swiglu_limit=limit)

    def call(inp, out):
        got = flydsl_moe_a8w4_ptpc(
            inp,
            prepared,
            topk_weight,
            topk_ids,
            output=out,
            swiglu_limit=limit,
            **options,
        )
        assert got is out
        return list(moe_a8w4.last_path)

    want = torch.full_like(x, float("nan"))
    path = call(x, want)
    assert "output_copy" not in path, path
    for inplace in (False, True):
        buf, out = _odd_output(*x.shape)
        if inplace:
            out.copy_(x)
        got = call(out if inplace else x, out)
        assert got[-1] == "output_copy" and got.count("output_copy") == 1, got
        assert bool(buf[:_ODD_PAD].isnan().all()), (inplace, buf[:_ODD_PAD])
        assert bool(buf[_ODD_PAD + x.numel() :].isnan().all()), inplace
        assert_close(out, ref)
        if not _path_atomic(got):
            assert torch.equal(_bits(out), _bits(want)), inplace


def test_routing_input_dtypes(monkeypatch):
    """int64 and non-contiguous ``topk_ids`` and a non-contiguous BF16 ``topk_weight`` (as a router may produce
    them) are converted to the contiguous int32 / float32 the sort and the kernels read; without the output
    argument the op allocates the output. The entry converts once for every path; M 40 (H 512, E 5) streams, so the
    streaming prep sorts the converted routing (the decode limit DECODE_MAX_WGS set to 0: by default this shape
    decodes up to M 319).

    The basic tests and the benchmark pass the contiguous int32 / float32 routing of ``make_inputs`` and an output
    buffer; test_decode_duplicate_and_int64_ids converts on the decode path.
    """
    monkeypatch.setattr(moe_a8w4, "DECODE_MAX_WGS", 0)
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(40, 512, 5, 2, seed=66, shared=False)
    ids = torch.zeros((40, 4), dtype=torch.int64)[:, ::2]
    ids.copy_(topk_ids)
    weights = torch.zeros((40, 4), dtype=torch.bfloat16)[:, ::2]
    weights.copy_(topk_weight)
    assert not ids.is_contiguous() and not weights.is_contiguous()
    out = flydsl_moe_a8w4_ptpc(x, prepared, weights, ids)
    _assert_path(["a8w4_stream_prep"] + _stream_path(32, True))
    ref = reference(
        x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], weights.float(), topk_ids
    )
    assert_close(out, ref)


def test_m_zero():
    """An empty batch (M = 0) returns an empty output and launches nothing.

    The basic tests and the benchmark always have tokens.
    """
    _, prepared = _small_weights(512, 64)
    x = torch.empty((0, 512), dtype=torch.bfloat16)
    out = flydsl_moe_a8w4_ptpc(
        x, prepared, torch.empty((0, 2)), torch.empty((0, 2), dtype=torch.int32)
    )
    assert out.shape == (0, 512) and out.dtype == torch.bfloat16
    assert moe_a8w4.last_path == []


# -- 2.8 SwiGLU limit -----------------------------------------------------------------------------


def test_swiglu_limit_tile_k128():
    """``swiglu_limit=10`` in the Gate/Up kernel with the 128-K tile (its own pipeline and occupancy), inputs scaled
    by 10 so that more than 10% of the gate and of the up values exceed the limit; M 256 (E 24) sorts with the
    streaming prep, and the default dispatch picks the same kernels (many rows per expert: K tile 128).

    The basic tests clamp in the decode and streaming Gate/Up only; the benchmark's DeepSeek cases reach the 128-K
    tile from M 2048 on, but their N(0, 1) inputs almost never exceed the limit, so they measure its cost without
    checking it.
    """
    _a8w4_swiglu_check(
        *_ds_weights(),
        256,
        410 + 256,
        DS_HIDDEN,
        DS_EXPERTS,
        DS_TOPK,
        False,
        tile_k=128,
    )
    _assert_path(["a8w4_stream_prep"] + _prefill_path(128))


# case -> (shape, M, tile options, the kernels in order); qwen = H 2560, E 33, topk 11, I 320; mimo = H 4096, E 12,
# topk 8, I 256; ds = DeepSeek H 5120, E 24, topk 6, I 576. Behind the streaming Down the routes are summed by
# route_sum or by the Down's atomics (_ROUTE_SUMS).
_SWIGLU_CASES = {
    "decode_nn": ("qwen", 3, {}, _DECODE_PATHS["nn"]),
    "decode_pq": ("qwen", 8, {}, _DECODE_PATHS["np"]),
    "stream_fused_quant": (
        "qwen",
        64,
        {},
        ["a8w4_stream_prep"] + _stream_path(32, True),
    ),
    "stream_bm16": ("ds", 40, {}, ["a8w4_stream_prep"] + _stream_path(16, False)),
    "stream_bm64": ("mimo", 60, {}, ["a8w4_stream_prep"] + _stream_path(64, False)),
    "prefill_tk256": (
        "ds",
        64,
        {"block_m": 64},
        ["a8w4_stream_prep"] + _prefill_path(256),
    ),
    "h5120_nn": ("ds", 3, {}, _DECODE_PATHS["nn"]),
}


@pytest.mark.parametrize("case", list(_SWIGLU_CASES))
def test_swiglu_limit_kernels(case):
    """``swiglu_limit=10`` in each class of Gate/Up kernel (a compile-time constant wired into each kernel's own
    epilogue): the decode quantizing x in the kernel in front of either Down (M 3: the Down that quantizes its rows;
    M 8: the slot-pair Down on rows quantized by the row kernel), the streaming Gate/Up quantizing the intermediate in
    its epilogue (Qwen M 64, blocks of 32), the light-wave streaming
    Gate/Up with blocks of 16 (DeepSeek M 40, ACT_STEPS 4) and 64 (MiMo M 60, I 256: blocks of 64 with the atomic
    Down), the prefill Gate/Up with the K tile 256
    (block_m=64), and the H 5120 decode quantizing in the kernel (M = 3: 10 bf16 pairs per thread over K = 5120);
    inputs scaled by 10 so that more than 10% of the gate and of the up values exceed the limit, graded against the
    reference with the limit.

    The benchmark's DeepSeek cases have N(0, 1) inputs that almost never exceed 10 (they measure the cost of the
    limit without checking it), the Qwen / MiMo cases pass no limit. The basic DeepSeek cases clamp in the streaming
    Gate/Up with blocks of 32 (M 64) and in the deduplicated Gate/Up with groups of 2 while H 5120 decodes M 8;
    test_swiglu_limit_tile_k128 covers the 128-K prefill tile; the decode Gate/Up on pre-quantized x clamps in the
    layer fold cases (M 4-10, swiglu_limit 10).
    """
    shape, tokens, options, path = _SWIGLU_CASES[case]
    if shape == "qwen":
        (w, prepared), dims = _decode_weights(320), (HIDDEN, EXPERTS, TOPK, True)
    elif shape == "mimo":
        (w, prepared), dims = _mimo_small_weights(256), (
            MIMO_HIDDEN,
            12,
            MIMO_TOPK,
            False,
        )
    else:
        (w, prepared), dims = _ds_weights(), (DS_HIDDEN, DS_EXPERTS, DS_TOPK, False)
    _a8w4_swiglu_check(w, prepared, tokens, 410 + tokens, *dims, **options)
    _assert_path(path)


@functools.cache
def _ds_negative_gate_weights():
    """DeepSeek-shape weights (E=24) whose gate half is E2M1 -0.5 with E8M0 scale 1 (the up half unchanged): with x
    all 2**-7 every gate is exactly -0.5 * 5120 * 2**-7 = -20."""
    w = make_weights(DS_EXPERTS, DS_HIDDEN, DS_INTER, seed=420)
    q, s = w["w1"].view(torch.uint8).clone(), w["w1_scale"].view(torch.uint8).clone()
    q[:, :DS_INTER], s[:, :DS_INTER] = 0x99, 127
    w = dict(
        w, w1=q.view(torch.float4_e2m1fn_x2), w1_scale=s.view(torch.float8_e8m0fnu)
    )
    return w, _a8w4_prepare(w)


@pytest.mark.parametrize(
    "tokens, options",
    [(40, {"block_m": 64}), (40, {}), (8, {})],
    ids=["prefill_m40", "stream_m40", "dedup_m8"],
)
def test_swiglu_gate_upper_only(tokens, options):
    """The limit only caps the gate from above (DeepSeek-V4.1's ``Expert``: gate = min(gate, L)): with every gate
    -20 < -L the op computes SiLU(-20) * up; an implementation that also clamps the gate to -L would compute
    SiLU(-10) * up, about 1.1e4 times the output, far outside the tolerance. M 40 with block_m=64 runs the prefill
    Gate/Up kernel, M 40 by default the streaming Gate/Up (blocks of 16), M 8 the deduplicated decode Gate/Up of
    H 5120 (pinned within the H 5120 decode limit, past it the streaming kernels, _assert_h5120).

    In the basic tests, test_swiglu_limit_tile_k128 and the benchmark, SiLU(gate) and SiLU(-L) for gate < -L are
    both below 5e-4, far smaller than what large gates contribute, so clamping the lower side or not does not show
    within the tolerance.
    """
    w, prepared = _ds_negative_gate_weights()
    _, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=420 + tokens, shared=False
    )
    x = torch.full((tokens, DS_HIDDEN), 2.0**-7, dtype=torch.bfloat16)
    gate = (
        quant_rows_fp8(x[:1])
        @ dequant_mxfp4(w["w1"][0, :DS_INTER], w["w1_scale"][0, :DS_INTER]).T
    )
    assert torch.allclose(gate, torch.full_like(gate, -20.0)), gate.aminmax()
    out = torch.full_like(x, float("nan"))
    flydsl_moe_a8w4_ptpc(
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=SWIGLU_LIMIT,
        **options,
    )
    if options:
        _assert_path(_prefill_path(256))
    elif tokens == 40:
        _assert_path(["a8w4_stream_prep"] + _stream_path(16, False))
    else:
        _assert_h5120(tokens, _DEDUP_PATHS["wide"])
    ref = reference(
        x,
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        topk_weight,
        topk_ids,
        swiglu_limit=SWIGLU_LIMIT,
    )
    assert_close(out, ref)


# -- 2.9 the aiter.fused_moe entry: tiers, arguments, errors, untagged weights, CUDA graph, the weight tag --------
# fused_moe hands a call on tagged weights to moe_a8w4._fused_moe_route, which checks the arguments and makes one
# flydsl_moe_a8w4_ptpc call; the dispatch inside that call is covered by the cases above. Here: one case per tier and
# output form through fused_moe, what its arguments become, what it refuses, the weights it does not route, and the
# tag. Weights are those of the basic tests (cached by _weights).

# shape: (experts, hidden, inter, topk, fused shared expert, weight seed, swiglu_limit); DeepSeek with its limit and
# the amplified inputs of test_deepseek_swiglu.
_FUSED_SHAPES = {
    "tp2": (EXPERTS, HIDDEN, 320, TOPK, True, 11, 0.0),
    "tp4": (EXPERTS, HIDDEN, 160, TOPK, True, 21, 0.0),
    "mimo_tp8": (MIMO_EXPERTS, MIMO_HIDDEN, 256, MIMO_TOPK, False, 41, 0.0),
    "ds": (DS_EXPERTS, DS_HIDDEN, DS_INTER, DS_TOPK, False, 400, SWIGLU_LIMIT),
}


def _fused_inputs(shape, tokens, seed):
    """Weights and inputs of a _FUSED_SHAPES shape; DeepSeek's inputs scaled by SWIGLU_AMPLIFY (its limit is hit)."""
    experts, hidden, inter, topk, shared, weight_seed, limit = _FUSED_SHAPES[shape]
    w, prepared = _weights(experts, hidden, inter, weight_seed)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=seed, shared=shared
    )
    if limit:
        x = (x.float() * SWIGLU_AMPLIFY).to(torch.bfloat16)
    return w, prepared, x, topk_weight, topk_ids, limit


def _path_tier(path):
    """The tier of a kernel path, told by its Gate/Up kernel: per-route or deduplicated decode, streaming, prefill."""
    gateup = next(name for name in path if name.startswith("a8w4_gateup"))
    if "_decode" in gateup:
        dedup = any("_dd" in name or "plan" in name for name in path)
        return "dedup" if dedup else "decode"
    return "stream" if "_stream" in gateup else "prefill"


def _path_atomic(path):
    """Whether a path adds the routes into the output with BF16x2 atomics (an order that changes from run to run, also
    between two direct calls): the streaming Down without a8w4_route_sum.
    """
    return (
        any(name.startswith("a8w4_down_stream") for name in path)
        and "a8w4_route_sum" not in path
    )


# (shape, tokens, tier): one case per tier and output form, at token counts that select it with the cut E (asserted):
# the per-route decode (TP4 M 8, with the I 160 -> 192 padding that the tag must carry), the deduplicated Gate/Up
# with H 5120's slot-pair Down (DeepSeek M 8), streaming with the atomic Down (TP2 M 48 and DeepSeek M 64, I 576),
# prefill with a8w4_route_sum (TP2 M 300).
_FUSED_CASES = [
    ("tp4", 8, "decode"),
    ("ds", 8, "dedup"),
    ("tp2", 48, "stream"),
    ("ds", 64, "stream"),
    ("tp2", 300, "prefill"),
]


@pytest.mark.parametrize(
    "shape, tokens, tier", _FUSED_CASES, ids=[f"{s}_m{m}" for s, m, _ in _FUSED_CASES]
)
def test_fused_moe_matches_api(shape, tokens, tier):
    """fused_moe runs the kernels of flydsl_moe_a8w4_ptpc on every tier; where the sum order is fixed its output has
    the same bits, where routes are added with BF16x2 atomics it is graded against the reference. DeepSeek passes
    swiglu_limit=10 as fused_moe's swiglu_limit and the limit changes the result of its inputs (a route that dropped
    it would fail); the other shapes pass fused_moe's default None. One case per tier and output form: the route
    passes every call on unchanged, so the variants inside a tier (decode nn / np / pp, sorting blocks) and the
    boundaries between tiers are left to the flydsl_moe_a8w4_ptpc cases above.

    The basic test checks one decode case; the benchmark calls flydsl_moe_a8w4_ptpc directly.
    """
    if shape == "ds" and tokens > moe_a8w4.DECODE_WIDE_MAX_M:
        tier = "stream"  # past the H 5120 decode limit, as in _assert_h5120
    w, prepared, x, topk_weight, topk_ids, limit = _fused_inputs(
        shape, tokens, seed=600 + tokens
    )
    want = torch.full_like(x, float("nan"))
    flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, output=want, swiglu_limit=limit
    )
    path = list(moe_a8w4.last_path)
    assert _path_tier(path) == tier, path
    got = torch.full_like(x, float("nan"))
    result = _fused_moe(
        x,
        prepared.w1,
        prepared.w2,
        topk_weight,
        topk_ids,
        output=got,
        swiglu_limit=limit or None,
    )
    assert result is got
    assert moe_a8w4.last_path == path
    if _path_atomic(path):
        ref = reference(
            x,
            w["w1"],
            w["w1_scale"],
            w["w2"],
            w["w2_scale"],
            topk_weight,
            topk_ids,
            swiglu_limit=limit,
        )
        assert_close(got, ref)
    else:
        assert torch.equal(_bits(got), _bits(want))
    if limit:
        off = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
        assert not torch.equal(_bits(off), _bits(want))


def test_fused_moe_arguments(monkeypatch):
    """What the route hands flydsl_moe_a8w4_ptpc: the weight rebuilt from the tags (the prepared tensors themselves),
    hidden_states, the routing tensors and the output as given; swiglu_limit normalized as on aiter's other SiLU paths
    (None, 0 and inf off, passed as 0.0; NaN and negative values raise ValueError); never a tile option. block_size_M
    None, 0 and -1 are the automatic choice (other values raise, test_fused_moe_rejects); the hints
    moe_sorting_dispatch_policy, splitk and herd_routed_scaling_factor change nothing; dtype=bf16 and quant_dtype_a /
    quant_dtype_a2=fp8 are accepted; a non-contiguous hidden_states is passed as a contiguous copy. With the kernels,
    block_size_M None, 0 and -1 run the default dispatch of flydsl_moe_a8w4_ptpc with its bits (Qwen TP2 M 8, the
    per-route decode, where a pinned tile would select the prefill kernels).

    The basic test and test_fused_moe_matches_api pass only fused_moe's default tuning arguments.
    """
    _, prepared, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 40, seed=500)
    calls = []
    monkeypatch.setattr(
        moe_a8w4,
        "flydsl_moe_a8w4_ptpc",
        lambda *a, **k: calls.append((a, k)) or "ran",
    )

    def passed(hidden_states=x, **options):
        calls.clear()
        assert (
            _fused_moe(
                hidden_states,
                prepared.w1,
                prepared.w2,
                topk_weight,
                topk_ids,
                **options,
            )
            == "ran"
        )
        ((args, kwargs),) = calls
        return args, kwargs

    out = torch.empty_like(x)
    args, kwargs = passed(output=out)
    assert args[0] is x and args[2] is topk_weight and args[3] is topk_ids
    assert kwargs["output"] is out and set(kwargs) == {"output", "swiglu_limit"}
    for name in ("w1", "w1_delta", "w1_scale", "w2", "w2_delta", "w2_scale"):
        assert getattr(args[1], name) is getattr(prepared, name), name
    sizes = ("experts", "hidden", "inter", "inter_padded")
    assert [getattr(args[1], n) for n in sizes] == [getattr(prepared, n) for n in sizes]
    for value, limit in (
        (None, 0.0),
        (0, 0.0),
        (-0.0, 0.0),
        (math.inf, 0.0),
        (10, 10.0),
        (7.5, 7.5),
    ):
        _, kwargs = passed(swiglu_limit=value)
        assert kwargs["swiglu_limit"] == limit, (value, kwargs)
        assert type(kwargs["swiglu_limit"]) is float, (value, kwargs)
    for value in (math.nan, -1.0, -math.inf):
        with pytest.raises(ValueError):
            passed(swiglu_limit=value)
    for value in (None, 0, -1):
        assert passed(block_size_M=value)[1] == {"output": None, "swiglu_limit": 0.0}
    args, kwargs = passed(
        moe_sorting_dispatch_policy=1,
        splitk=4,
        herd_routed_scaling_factor=2.5,
        dtype=torch.bfloat16,
        quant_dtype_a=aiter.dtypes.fp8,
        quant_dtype_a2=aiter.dtypes.fp8,
    )
    assert args[0] is x and kwargs == {"output": None, "swiglu_limit": 0.0}
    wide = torch.zeros((x.shape[0], 2 * x.shape[1]), dtype=x.dtype)[:, : x.shape[1]]
    wide.copy_(x)
    args, _ = passed(hidden_states=wide)
    assert args[0] is not wide and args[0].is_contiguous() and torch.equal(args[0], x)
    monkeypatch.undo()
    _, prepared, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 8, seed=501)
    want = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
    path = list(moe_a8w4.last_path)
    assert _path_tier(path) == "decode", path
    for value in (None, 0, -1):
        got = _fused_moe(
            x, prepared.w1, prepared.w2, topk_weight, topk_ids, block_size_M=value
        )
        assert moe_a8w4.last_path == path, value
        assert torch.equal(_bits(got), _bits(want)), value


def test_fused_moe_rejects():
    """The route raises before any kernel runs: for every fused_moe parameter it cannot honour, naming it
    (NotImplementedError; w1_scale / w2_scale too, the scales travel with the tagged weights; the raw shared-expert
    tensors, pointing at prepare_moe_a8w4_ptpc_weight(..., shared_w1=...); block_size_M other than None / 0 / -1,
    pointing at flydsl_moe_a8w4_ptpc(block_m=...) to pin a tile); for another activation / quant_type / quant_type_a
    (ValueError); for w1 / w2 that are not the pair of one prepare call (another call's w2, swapped, one of them an
    untagged copy) or do not match their tag (ValueError); for an output that breaks fused_moe's contract (shape,
    dtype, layout, overlapping hidden_states: RuntimeError). Off gfx942: test_requires_gfx942. shared_expert_id:
    test_fused_moe_shared_expert_id.

    The basic test and the other cases pass only supported arguments.
    """
    w, prepared, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 4, seed=510)
    E, H, I = prepared.experts, prepared.hidden, prepared.inter

    def run(w1=prepared.w1, w2=prepared.w2, **options):
        return _fused_moe(x, w1, w2, topk_weight, topk_ids, **options)

    unsupported = {
        "expert_mask": torch.ones(E, dtype=torch.int32),
        "doweight_stage1": True,
        "w1_scale": w["w1_scale"],
        "w2_scale": w["w2_scale"],
        "a1_scale": torch.ones(1),
        "a2_scale": torch.ones(1),
        "num_local_tokens": torch.tensor([4], dtype=torch.int32),
        "hidden_pad": 64,
        "intermediate_pad": 64,
        "bias1": torch.zeros((E, 2 * I)),
        "bias2": torch.zeros((E, H)),
        "beta": 1.0,
        "linear_beta": 1.0,
        "gate_mode": "interleave",
        "shared_w1": prepared.w1,
        "shared_w2": prepared.w2,
        "shared_w1_scale": w["w1_scale"],
        "shared_w2_scale": w["w2_scale"],
        "stage2_scatter": object(),
        "clamp_shared": False,
        "herd_topk": 1,
        "deterministic": True,
        "w1_global_scale": torch.ones((E, 2)),
        "w2_global_scale": torch.ones(E),
        "dtype": torch.float16,
        "quant_dtype_a": aiter.dtypes.fp4x2,
        "quant_dtype_a2": torch.bfloat16,
    }
    for name, value in unsupported.items():
        with pytest.raises(NotImplementedError, match=name):
            run(**{name: value})
    with pytest.raises(NotImplementedError, match="scales travel"):
        run(w1_scale=w["w1_scale"], w2_scale=w["w2_scale"])
    for name in ("shared_w1", "shared_w2", "shared_w1_scale", "shared_w2_scale"):
        with pytest.raises(
            NotImplementedError,
            match=rf"{name}.*prepare_moe_a8w4_ptpc_weight\(\.\.\., shared_w1=",
        ):
            run(**{name: unsupported[name]})
    for value in (64, 32):
        with pytest.raises(
            NotImplementedError,
            match=rf"block_size_M={value} .*flydsl_moe_a8w4_ptpc\(.*block_m=",
        ):
            run(block_size_M=value)
    for options in (
        {"activation": aiter.ActivationType.Swiglu},
        {"activation": aiter.ActivationType.Gelu},
        {"quant_type": aiter.QuantType.per_Token},
        {"quant_type": aiter.QuantType.No},
        {"quant_type_a": None},
        {"quant_type_a": aiter.QuantType.per_1x32},
    ):
        with pytest.raises(ValueError, match="quant_type_a=QuantType.per_Token"):
            run(**options)
    other = _a8w4_prepare(w)
    for w1, w2 in (
        (prepared.w1, other.w2),
        (prepared.w2, prepared.w1),
        (prepared.w1, prepared.w2.clone()),
        (prepared.w1.clone(), prepared.w2),
    ):
        with pytest.raises(ValueError, match="one prepare_moe_a8w4_ptpc_weight"):
            run(w1, w2)
    short = prepared.w1[:, :-16].contiguous()
    setattr(short, moe_a8w4.WEIGHT_TAG, getattr(prepared.w1, moe_a8w4.WEIGHT_TAG))
    with pytest.raises(ValueError, match="do not match their A8W4 tag"):
        run(short)
    for out in (
        torch.empty((5, H), dtype=torch.bfloat16),
        torch.empty((4, H), dtype=torch.float16),
        torch.empty((H, 4), dtype=torch.bfloat16).T,
        x,
    ):
        with pytest.raises(RuntimeError):
            run(output=out)


def test_untagged_mxfp4_points_to_prepare(monkeypatch):
    """MXFP4 weights without the tag (a checkpoint's fp4x2 or uint8 weights, or copies of prepared tensors, which do
    not carry it) with quant_type=per_1x32 and quant_type_a=per_Token never reach the route and still raise aiter's
    NotImplementedError for mixed granularity, now pointing at prepare_moe_a8w4_ptpc_weight; other mixed pairs keep
    the message without it.

    The basic test and the other cases pass tagged weights.
    """
    w, prepared, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 4, seed=520)
    monkeypatch.setattr(
        moe_a8w4,
        "_fused_moe_route",
        lambda args: pytest.fail("untagged weights were routed"),
    )
    scales = {"w1_scale": w["w1_scale"], "w2_scale": w["w2_scale"]}
    for w1, w2 in (
        (w["w1"], w["w2"]),
        (w["w1"].view(torch.uint8), w["w2"].view(torch.uint8)),
        (prepared.w1.clone(), prepared.w2.clone()),
    ):
        with pytest.raises(NotImplementedError, match="prepare_moe_a8w4_ptpc_weight"):
            _fused_moe(x, w1, w2, topk_weight, topk_ids, **scales)
    for quant_type, quant_type_a in (
        (aiter.QuantType.per_1x32, aiter.QuantType.per_1x128),
        (aiter.QuantType.per_Token, aiter.QuantType.per_1x32),
    ):
        with pytest.raises(NotImplementedError, match="mixed") as raised:
            _fused_moe(
                x,
                w["w1"],
                w["w2"],
                topk_weight,
                topk_ids,
                quant_type=quant_type,
                quant_type_a=quant_type_a,
                **scales,
            )
        assert "prepare_moe_a8w4_ptpc_weight" not in str(raised.value)


def test_untagged_weights_keep_their_dispatch(monkeypatch):
    """Weights without the tag (here the FP8 per-token combination of the benchmark's aiter reference) never reach the
    route: fused_moe hands the layer beneath (fused_moe_) the arguments it handed it before. That the layer still
    runs aiter's tuned kernels is asserted by the benchmark (the tuned table row of every Qwen case).

    The basic test and the other cases pass tagged weights.
    """
    import aiter.fused_moe as aiter_fused_moe

    _, _, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 4, seed=530)
    calls = []
    monkeypatch.setattr(
        moe_a8w4,
        "_fused_moe_route",
        lambda args: pytest.fail("untagged weights were routed"),
    )
    monkeypatch.setattr(
        aiter_fused_moe,
        "fused_moe_",
        lambda **kwargs: calls.append(kwargs) or "native",
    )
    w1 = torch.empty((EXPERTS, 640, HIDDEN), dtype=aiter.dtypes.fp8)
    w2 = torch.empty((EXPERTS, HIDDEN, 320), dtype=aiter.dtypes.fp8)
    w1.is_shuffled = w2.is_shuffled = True
    s1 = torch.ones((EXPERTS, 640, 1))
    s2 = torch.ones((EXPERTS, HIDDEN, 1))
    result = aiter_fused_moe.fused_moe(
        x,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_Token,
        w1_scale=s1,
        w2_scale=s2,
    )
    assert result == "native"
    (kwargs,) = calls
    assert kwargs["w1"] is w1 and kwargs["w2"] is w2
    assert kwargs["w1_scale"] is s1 and kwargs["w2_scale"] is s2
    assert kwargs["hidden_states"] is x and kwargs["topk_ids"] is topk_ids
    assert kwargs["topk_weight"] is topk_weight
    assert kwargs["quant_type"] == aiter.QuantType.per_Token.value
    assert kwargs["activation"] == aiter.ActivationType.Silu.value
    assert kwargs["quant_type_a"] is None and kwargs["block_size_M"] == -1


def test_fused_moe_graph_replay():
    """fused_moe on tagged weights can be captured in a CUDA graph (the route adds host work only) and replayed: the
    replay reads newly written inputs and gives the bits of a direct call on them. Compiled by an eager call outside
    the capture first (Qwen TP2 M 8, the per-route decode).

    The basic test and the other cases call fused_moe eagerly; the graph cases above call flydsl_moe_a8w4_ptpc.
    """
    _, prepared, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 8, seed=540)
    out = torch.empty_like(x)
    _fused_moe(x, prepared.w1, prepared.w2, topk_weight, topk_ids, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _fused_moe(x, prepared.w1, prepared.w2, topk_weight, topk_ids, output=out)
    _, _, x2, topk_weight2, topk_ids2, _ = _fused_inputs("tp2", 8, seed=541)
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    want = flydsl_moe_a8w4_ptpc(x2, prepared, topk_weight2, topk_ids2)
    assert _path_tier(moe_a8w4.last_path) == "decode"
    assert torch.equal(_bits(out), _bits(want))


def test_weight_tag_lifetime():
    """The tags keep a prepared weight's other tensors but hold no reference back to w1 / w2: dropping the weight frees
    every tensor by reference counting alone (garbage collection off), and w1 / w2 kept on their own keep the rest
    alive through the tags. The tag can be copied onto an nn.Parameter wrapper of the same storage (wrappers, views and
    copies do not carry Python attributes, as with is_shuffled), and a deep copy of the pair in one call (as
    run_perftest rotates its arguments) is a pair of its own; fused_moe gives the same bits on both. The tag changes
    neither values nor shapes (uint8 [E, I_pad * H] and [E, H * I_pad / 2]).

    The basic test and the other cases keep the prepared weight alive and pass its own w1 / w2.
    """
    w, _, x, topk_weight, topk_ids, _ = _fused_inputs("tp2", 3, seed=550)
    names = ("w1", "w1_delta", "w1_scale", "w2", "w2_delta", "w2_scale")
    gc.collect()
    gc.disable()
    try:
        prepared = _a8w4_prepare(w)
        refs = {name: weakref.ref(getattr(prepared, name)) for name in names}
        del prepared
        assert all(ref() is None for ref in refs.values()), [
            name for name, ref in refs.items() if ref() is not None
        ]
        prepared = _a8w4_prepare(w)
        refs = {name: weakref.ref(getattr(prepared, name)) for name in names}
        w1, w2 = prepared.w1, prepared.w2
        del prepared
        assert all(ref() is not None for ref in refs.values())
        del w1, w2
        assert all(ref() is None for ref in refs.values())
    finally:
        gc.enable()
    prepared = _a8w4_prepare(w)
    E, H, ip = prepared.experts, prepared.hidden, prepared.inter_padded
    assert (prepared.w1.dtype, tuple(prepared.w1.shape)) == (torch.uint8, (E, ip * H))
    assert (prepared.w2.dtype, tuple(prepared.w2.shape)) == (
        torch.uint8,
        (E, H * ip // 2),
    )
    want = _fused_moe(x, prepared.w1, prepared.w2, topk_weight, topk_ids)
    p1 = torch.nn.Parameter(prepared.w1, requires_grad=False)
    p2 = torch.nn.Parameter(prepared.w2, requires_grad=False)
    assert not hasattr(p1, moe_a8w4.WEIGHT_TAG)
    for p, t in ((p1, prepared.w1), (p2, prepared.w2)):
        setattr(p, moe_a8w4.WEIGHT_TAG, getattr(t, moe_a8w4.WEIGHT_TAG))
    got = _fused_moe(x, p1, p2, topk_weight, topk_ids)
    assert torch.equal(_bits(got), _bits(want))
    c1, c2 = copy.deepcopy((prepared.w1, prepared.w2))
    assert c1.data_ptr() != prepared.w1.data_ptr()
    got = _fused_moe(x, c1, c2, topk_weight, topk_ids)
    assert torch.equal(_bits(got), _bits(want))


# -- 2.10 shared expert: prepare's conversions, real checkpoints, errors, graph, in-place, fused_moe -------------------
# Real-checkpoint cases read AITER_MOE_A8W4_DSV41_DIR (the shared experts of DeepSeek-V4.1-Flash-1017 layers 2, 37 and
# mtp.0 from the official checkpoint and FP4PTPC @97cc9b32 / @main: manifest.json and <source>_<layer>.pt of raw
# bytes, written by the task tool fetch_dsv41_shared.py) and skip without it.

_DSV41_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "F32": torch.float32,
}


def _dsv41(key):
    """The shared expert of one checkpoint layer (``<source>_<layer>``) as {"w1.weight", "w1.scale" or
    "w1.weight_scale", ... "w3.*", "w2.*"} on the GPU in the manifest's dtypes and shapes; skips without the data.
    """
    root = os.environ.get("AITER_MOE_A8W4_DSV41_DIR")
    if not root:
        pytest.skip(
            "AITER_MOE_A8W4_DSV41_DIR is not set (DeepSeek-V4.1-Flash shared experts)"
        )
    path = Path(root) / f"{key}.pt"
    if not path.exists():
        pytest.skip(f"{path} not found")
    tensors = json.loads((Path(root) / "manifest.json").read_text())[key]["tensors"]
    raw = torch.load(path)
    return {
        name.split("shared_experts.")[1]: raw[name]
        .cuda()
        .view(_DSV41_DTYPES[meta["dtype"]])
        .reshape(meta["shape"])
        for name, meta in tensors.items()
    }


def _rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _routed_only(prepared):
    """A weight prepared with a shared expert without it (the same routed tensors)."""
    p = prepared
    return MoeA8W4PtpcWeight(
        p.w1,
        p.w1_delta,
        p.w1_scale,
        p.w2,
        p.w2_delta,
        p.w2_scale,
        p.experts,
        p.hidden,
        p.inter,
        p.inter_padded,
    )


def _prepare_shared(w, shared):
    return prepare_moe_a8w4_ptpc_weight(
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        shared_w1=shared[0],
        shared_w1_scale=shared[1],
        shared_w2=shared[2],
        shared_w2_scale=shared[3],
    )


def _shared_fields(prepared):
    return (
        prepared.shared_w1,
        prepared.shared_w1_scale,
        prepared.shared_w2,
        prepared.shared_w2_scale,
    )


def _fold_reference(w, codes):
    """Independent 32x32-block folding: the exact values (dequant_fp8), one scale per row 2**ceil(log2(amax / 448))
    with the exponent from frexp (_pow2_ceil_exponent; 1 for a zero row), elements RNE to E4M3FN.
    """
    v = dequant_fp8(w, codes)
    amax = v.abs().amax(dim=1).double()
    exp = _pow2_ceil_exponent(
        torch.where(amax > 0, amax, torch.full_like(amax, FP8_OCP_MAX)) / FP8_OCP_MAX
    )
    scale = torch.exp2(exp.double()).float()
    return (v / scale[:, None]).to(torch.float8_e4m3fn), scale


def test_shared_prepare_bitexact():
    """prepare's shared-expert fields bit for bit. Per-row E4M3FN input keeps its bits except -0 (0x80, NaN in FNUZ),
    which becomes +0, with twice the scale; the same weights already in E4M3FNUZ (bits and scales converted) give the
    same fields; 32x32-block input is folded to one power-of-two scale per row exactly as an independent folding computes
    it (_fold_reference; scale codes spread over 16 values, so that elements of low blocks become E4M3FN subnormals and
    round; an all-zero row takes the scale 1); I 288 is zero-padded to 320 (gate and up rows with scale 1, Down
    columns) and the Down K to 384, in shuffle_weight's (16, 16) layout. [N] and [N, 1] per-row scales and uint8 and
    float8_e8m0fnu codes are accepted.

    The basic tests and the benchmark grade whole layers within the reference's tolerance, which a bit error of the
    conversion (a lost -0, one row's scale off by a power of two, a rounding of the folding) could pass.
    """
    from aiter.ops.shuffle import shuffle_weight

    E, H, I, ip = 3, 512, 288, 320
    w = make_weights(E, H, I, seed=710)
    gen = torch.Generator(device="cuda").manual_seed(711)

    def fp8(rows, cols):
        bits = torch.randint(
            0, 256, (rows, cols), generator=gen, device="cuda", dtype=torch.uint8
        )
        bits = torch.where((bits & 0x7F) == 0x7F, bits ^ 0x01, bits)  # no NaN codes
        bits[:, :8] = 0x80
        return bits.view(torch.float8_e4m3fn)

    def expect(q1, t1, q2, t2):
        """prepare's fields for E4M3FN q1 / q2 with float32 per-row scales t1 / t2."""
        b1, b2 = (
            q.view(torch.uint8).masked_fill(q.view(torch.uint8) == 0x80, 0)
            for q in (q1, q2)
        )
        z = torch.zeros((ip - I, H), dtype=torch.uint8)
        b1 = torch.cat([b1[:I], z, b1[I:], z])
        b2 = torch.cat([b2, torch.zeros((H, 384 - I), dtype=torch.uint8)], dim=1)
        t1 = t1.reshape(-1) * 2
        one = torch.ones(ip - I)
        return (
            shuffle_weight(b1.view(torch.float8_e4m3fnuz), layout=(16, 16)),
            torch.cat([t1[:I], one, t1[I:], one]).view(-1, 1),
            shuffle_weight(b2.view(torch.float8_e4m3fnuz), layout=(16, 16)),
            (t2.reshape(-1) * 2).view(-1, 1),
        )

    def same(got, want):
        for g, x in zip(got, want):
            assert g.shape == x.shape and g.dtype == x.dtype, (g.shape, x.shape)
            assert torch.equal(g.view(torch.uint8), x.view(torch.uint8))

    w1, w2 = fp8(2 * I, H), fp8(H, I)
    s1 = torch.rand((2 * I, 1), generator=gen, device="cuda") + 0.5
    s2 = torch.rand((H,), generator=gen, device="cuda") + 0.5
    got = _shared_fields(_prepare_shared(w, (w1, s1, w2, s2)))
    same(got, expect(w1, s1, w2, s2))
    fnuz = tuple(
        q.view(torch.uint8)
        .masked_fill(q.view(torch.uint8) == 0x80, 0)
        .view(torch.float8_e4m3fnuz)
        for q in (w1, w2)
    )
    same(_shared_fields(_prepare_shared(w, (fnuz[0], s1 * 2, fnuz[1], s2 * 2))), got)
    c1 = torch.randint(
        110, 126, (2 * I // 32, H // 32), generator=gen, device="cuda"
    ).to(torch.uint8)
    c2 = torch.randint(110, 126, (H // 32, I // 32), generator=gen, device="cuda").to(
        torch.uint8
    )
    zero = w1.view(torch.uint8).clone()
    zero[5] = 0  # an all-zero row folds to the scale 1
    zero = zero.view(torch.float8_e4m3fn)
    blocks = (zero, c1.view(torch.float8_e8m0fnu), w2, c2)
    want = expect(*_fold_reference(zero, c1), *_fold_reference(w2, c2))
    got = _shared_fields(_prepare_shared(w, blocks))
    same(got, want)
    assert got[1][5, 0].item() == 2.0  # 1, doubled for E4M3FNUZ


@pytest.mark.parametrize("layer", ["layers.2", "layers.37"])
def test_shared_fold_matches_fp4ptpc(layer):
    """prepare's folding of the official checkpoint's 32x32-block shared expert (w1, w3 and w2 whole: a row's scale
    needs the whole row) is byte-identical to FP4PTPC@97cc9b32's per-row weights and scales of the layer and within
    relative L2 1e-6 of the block dequantization; the TP4 rank-0 slice (gate / up rows [0, 576), Down columns
    [0, 576)) prepared from either checkpoint stores the same gate / up (rows fold alone; the Down slice folds over its
    own columns, within 1e-6 of the blocks too).

    The basic tests and the benchmark use synthetic weights.
    """
    official, ptpc = _dsv41(f"official_{layer}"), _dsv41(f"ptpc97_{layer}")
    for m in ("w1", "w3", "w2"):
        q, s = moe_a8w4._fold_blocks(
            official[f"{m}.weight"], official[f"{m}.scale"].view(torch.uint8)
        )
        assert torch.equal(q.view(torch.uint8), ptpc[f"{m}.weight"].view(torch.uint8))
        assert torch.equal(s, ptpc[f"{m}.weight_scale"].reshape(-1)), m
        exact = dequant_fp8(official[f"{m}.weight"], official[f"{m}.scale"])
        error = _rel_l2(dequant_fp8(q, s), exact)
        print(f"{layer} {m}: relative L2 of the folding {error:.3e}")
        assert error <= 1e-6, (m, error)
    I = 576
    w = make_weights(2, LAYER_HIDDEN, I, seed=720)
    block = (
        torch.cat([official["w1.weight"][:I], official["w3.weight"][:I]]),
        torch.cat([official["w1.scale"][: I // 32], official["w3.scale"][: I // 32]]),
        official["w2.weight"][:, :I].contiguous(),
        official["w2.scale"][:, : I // 32].contiguous(),
    )
    row = (
        torch.cat([ptpc["w1.weight"][:I], ptpc["w3.weight"][:I]]),
        torch.cat([ptpc["w1.weight_scale"][:I], ptpc["w3.weight_scale"][:I]]),
        ptpc["w2.weight"][:, :I].contiguous(),
        ptpc["w2.weight_scale"],
    )
    a = _shared_fields(_prepare_shared(w, block))
    b = _shared_fields(_prepare_shared(w, row))
    assert torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
    assert torch.equal(a[1], b[1])
    q, s = moe_a8w4._fold_blocks(block[2], block[3].view(torch.uint8))
    assert _rel_l2(dequant_fp8(q, s), dequant_fp8(block[2], block[3])) <= 1e-6


def test_shared_fold_mtp_blocks():
    """The MTP layer's shared expert (mtp.0) is in 32x32 blocks in the official checkpoint and FP4PTPC@97cc9b32 alike
    (the same bytes); prepare's folding keeps every element of at least 1% of its row's largest magnitude exactly and the
    whole within relative L2 1e-6 of the block dequantization. FP4PTPC@main stores it per row, re-quantized with other
    scales: its distance to the blocks (2.5-2.8% relative L2) is printed, not asserted.

    The basic tests and the benchmark use synthetic weights.
    """
    official, ptpc, main = (
        _dsv41(f"{source}_mtp.0") for source in ("official", "ptpc97", "ptpcmain")
    )
    for m in ("w1", "w3", "w2"):
        for part in ("weight", "scale"):
            key = f"{m}.{part}"
            assert torch.equal(
                official[key].view(torch.uint8), ptpc[key].view(torch.uint8)
            ), key
        exact = dequant_fp8(ptpc[f"{m}.weight"], ptpc[f"{m}.scale"])
        q, s = moe_a8w4._fold_blocks(
            ptpc[f"{m}.weight"], ptpc[f"{m}.scale"].view(torch.uint8)
        )
        folded = dequant_fp8(q, s)
        large = exact.abs() >= 0.01 * exact.abs().amax(dim=1, keepdim=True)
        assert torch.equal(folded[large], exact[large]), m
        error = _rel_l2(folded, exact)
        gap = _rel_l2(
            dequant_fp8(main[f"{m}.weight"], main[f"{m}.weight_scale"]), exact
        )
        print(f"mtp.0 {m}: folding {error:.3e}, FP4PTPC@main per-row weights {gap:.3%}")
        assert error <= 1e-6, (m, error)


def _quant_1x32(t):
    """The model's activation quantization: FP8 E4M3FN with one UE8M0 scale per 32 elements, 2**ceil(log2(amax /
    448)), RNE; dequantized, float32."""
    rows, cols = t.shape
    b = t.float().view(rows, cols // 32, 32)
    amax = b.abs().amax(dim=-1, keepdim=True).double()
    exp = _pow2_ceil_exponent(
        torch.where(amax > 0, amax, torch.full_like(amax, FP8_OCP_MAX)) / FP8_OCP_MAX
    )
    scale = torch.exp2(exp.double()).float()
    return ((b / scale).to(torch.float8_e4m3fn).float() * scale).view(rows, cols)


def _shared_fp32(x, w1, w2, limit, quant=None):
    """The shared expert in float32 on dequantized weights; ``quant`` quantizes x and the intermediate (None: not)."""
    inter = w1.shape[0] // 2
    xa = x.float() if quant is None else quant(x)
    h = xa @ w1.T
    gate, up = h[:, :inter].clamp(max=limit), h[:, inter:].clamp(-limit, limit)
    mid = torch.nn.functional.silu(gate) * up
    return (mid if quant is None else quant(mid)) @ w2.T


@pytest.mark.parametrize("layer", ["layers.2", "layers.37"])
def test_shared_real_weights_accuracy(layer):
    """The shared expert on the official weights of a layer (TP4 rank 0: w1 / w3 rows [0, 576), w2 columns [0, 576);
    32x32 blocks folded by prepare), inputs N(0, 1) and N(0, 1) with 4 outlier channels times 50, seeds 0-2, M 64,
    swiglu_limit 10. Its relative L2 error against the fp32 expert (the same block weights; x and the intermediate not
    quantized) is at most 1.5 times that of the model's own quantization (x and the intermediate in FP8 with one UE8M0
    scale per 32, _quant_1x32, as DeepSeek-V4.1 runs it). The routed experts get weight 0, so the call's output is the
    shared expert's; the error of ATOM's split on the same prepared weights is printed too.

    The basic tests and the benchmark grade synthetic weights against a reference with this op's own quantization.
    """
    I, limit = 576, LAYER_SWIGLU_LIMIT
    official = _dsv41(f"official_{layer}")
    shared = (
        torch.cat([official["w1.weight"][:I], official["w3.weight"][:I]]),
        torch.cat([official["w1.scale"][: I // 32], official["w3.scale"][: I // 32]]),
        official["w2.weight"][:, :I].contiguous(),
        official["w2.scale"][:, : I // 32].contiguous(),
    )
    w1, w2 = dequant_fp8(*shared[:2]), dequant_fp8(*shared[2:])
    experts, topk, tokens = 8, 2, 64
    prepared = _prepare_shared(make_weights(experts, LAYER_HIDDEN, I, seed=730), shared)
    ids = torch.arange(topk + 1, dtype=torch.int32).repeat(tokens, 1)
    ids[:, -1] = experts
    weight = torch.zeros((tokens, topk + 1))
    weight[:, -1] = 1.0
    for kind, seed in itertools.product(("gauss", "outliers"), range(3)):
        gen = torch.Generator(device="cuda").manual_seed(731 + seed)
        x = torch.randn((tokens, LAYER_HIDDEN), generator=gen, device="cuda")
        if kind == "outliers":
            channels = torch.randperm(LAYER_HIDDEN, generator=gen, device="cuda")[:4]
            x[:, channels] *= 50
        x = x.to(torch.bfloat16)
        exact = _shared_fp32(x, w1, w2, limit)
        model = _rel_l2(_shared_fp32(x, w1, w2, limit, _quant_1x32), exact)
        ours = _rel_l2(
            flydsl_moe_a8w4_ptpc(x, prepared, weight, ids, swiglu_limit=limit), exact
        )
        split = _rel_l2(
            aiter_shared_split(x, *_shared_fields(prepared), limit).float(), exact
        )
        print(
            f"{layer} {kind} seed {seed}: relative L2 ours {ours:.4e}, split {split:.4e}, model {model:.4e}, "
            f"ours / model {ours / model:.3f}"
        )
        assert ours <= 1.5 * model, (kind, seed, ours, model)


def test_shared_rejects(monkeypatch):
    """Shared-expert inputs prepare cannot take raise ValueError before any conversion: a partial set, weights that are
    not 2-D FP8 E4M3FN / E4M3FNUZ, mixed FP8 or scale formats, scales of another shape or dtype, 32x32 blocks over an I
    that is no multiple of 32, shapes other than the routed experts' [2I, H] / [H, I], NaN elements (E4M3FN and
    E4M3FNUZ input) / E8M0 codes / scales, another device; an argument that is no tensor raises TypeError, a
    MoeA8W4PtpcWeight with part of the shared fields ValueError. A call with a shared expert needs at least 2 routing
    columns (ValueError). DEBUG_CHECKS reads the routing: a last column other than E, routed ids outside [0, E) other
    than -1 (E, -2) or a routed id repeated in a token raise ValueError, masked ids (-1, repeated) pass; without it a row
    whose last id is not E gets no shared expert (ATOM's rule, as reference_layer). M = 0
    returns the output untouched with no kernel; int64 ids and bfloat16 weights give the bits of int32 / float32 ones.
    The 32-bit offset limit on M counts the routed columns, not the shared one.

    The basic tests and the benchmark pass valid inputs with E in the last column, far below the offset limit.
    """
    import dataclasses

    E, H, I = 4, 512, 64
    w = make_weights(E, H, I, seed=740)
    shared = make_shared_weights(H, I, seed=741)
    rows, blocks = shared["row"], shared["block"]

    def prepare(*args):
        return _prepare_shared(w, args)

    def bad(index, value, base=rows):
        args = list(base)
        args[index] = value
        with pytest.raises(ValueError):
            prepare(*args)

    for missing in range(4):
        args = list(rows)
        args[missing] = None
        with pytest.raises(ValueError, match="needs all of"):
            prepare(*args)
    bad(0, rows[0].float().to(torch.bfloat16))
    bad(0, rows[0][None])
    bad(0, rows[0].view(torch.uint8).view(torch.float8_e4m3fnuz))
    bad(1, rows[1].view(1, -1))
    bad(1, rows[1].half())
    bad(1, blocks[1])
    bad(3, blocks[3][:, :1].contiguous(), base=blocks)
    for hidden, inter in ((H, 32), (2 * H, I)):
        with pytest.raises(ValueError, match="must match the routed experts"):
            prepare(*make_shared_weights(hidden, inter, seed=742)["row"])
    small = make_shared_weights(H, 48, seed=742)["row"]
    with pytest.raises(ValueError, match="multiples of 32"):
        prepare(
            small[0],
            torch.zeros((96 // 32, H // 32), dtype=torch.uint8),
            small[2],
            torch.zeros((H // 32, 1), dtype=torch.uint8),
        )
    nan = rows[0].view(torch.uint8).clone()
    nan[3, 5] = 0x7F
    bad(0, nan.view(torch.float8_e4m3fn))
    bad(1, torch.where(torch.arange(2 * I)[:, None] == 2, math.inf, rows[1]))
    codes = blocks[1].view(torch.uint8).clone()
    codes[0, 0] = 255
    bad(1, codes, base=blocks)
    bad(2, rows[2].cpu())
    with pytest.raises(TypeError, match="shared_w1_scale must be a tensor"):
        prepare(rows[0], 1.0, rows[2], rows[3])
    fnuz = [
        q.view(torch.uint8).masked_fill(q.view(torch.uint8) == 0x80, 0)
        for q in (rows[0], rows[2])
    ]
    fnuz[0][3, 5] = 0x80  # NaN in E4M3FNUZ
    with pytest.raises(ValueError, match="NaN"):
        prepare(
            fnuz[0].view(torch.float8_e4m3fnuz),
            rows[1] * 2,
            fnuz[1].view(torch.float8_e4m3fnuz),
            rows[3] * 2,
        )
    prepared = prepare(*rows)
    with pytest.raises(ValueError, match="all given or all None"):
        dataclasses.replace(_routed_only(prepared), shared_w1=prepared.shared_w1)
    x, topk_weight, topk_ids = make_layer_inputs(6, H, E, 2, seed=743)
    with pytest.raises(ValueError, match="topk \\+ 1"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight[:, -1:], topk_ids[:, -1:])
    want = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
    half = topk_weight.to(torch.bfloat16)
    got = flydsl_moe_a8w4_ptpc(x, prepared, half, topk_ids.long())
    assert torch.equal(
        _bits(got), _bits(flydsl_moe_a8w4_ptpc(x, prepared, half.float(), topk_ids))
    )
    other = topk_ids.clone()
    other[::2, -1] = E - 1 - other[::2, 0] % 2
    out = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, other)
    ref = reference_layer(x, w, rows, topk_weight, other)
    assert_close(out, ref)
    assert shared_route_weight(topk_weight, other, E)[::2].eq(0).all()
    monkeypatch.setattr(moe_a8w4, "DEBUG_CHECKS", True)
    assert torch.equal(
        _bits(flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)), _bits(want)
    )
    with pytest.raises(ValueError, match="last column"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, other)
    for value in (E, -2):
        routed = topk_ids.clone()
        routed[1, 0] = value
        with pytest.raises(ValueError, match="routed expert ids"):
            flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, routed)
    routed = topk_ids.clone()
    routed[1, 1] = routed[1, 0]
    with pytest.raises(ValueError, match="distinct"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, routed)
    routed[1, :2] = -1
    flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, routed)
    monkeypatch.undo()
    empty = torch.empty((0, H), dtype=torch.bfloat16)
    out = flydsl_moe_a8w4_ptpc(
        empty, prepared, topk_weight[:0], topk_ids[:0], output=empty
    )
    assert out is empty and moe_a8w4.last_path == []
    # The 32-bit offset limit counts the routed columns only: E 8, topk 6, I 576, M 4, where M * topk * I_pad * 4
    # (55296) exceeds M * H * 2 and the shared column would add 9216.
    _, _, wide = _layer_weights(8, 576, 780)
    x, topk_weight, topk_ids = make_layer_inputs(4, LAYER_HIDDEN, 8, 6, seed=744)
    limit = 4 * 6 * wide.inter_padded * 4
    monkeypatch.setattr(moe_a8w4, "_MAX_BYTES", limit)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        flydsl_moe_a8w4_ptpc(x, wide, topk_weight, topk_ids)
    monkeypatch.setattr(moe_a8w4, "_MAX_BYTES", limit + 1)
    flydsl_moe_a8w4_ptpc(x, wide, topk_weight, topk_ids)
    _assert_shared()


@pytest.mark.parametrize("tokens", [1, 4, 2048], ids=["m1", "m4", "m2048"])
def test_layer_graph_replay(tokens):
    """A layer call with the shared expert can be captured in a CUDA graph and replayed: the replay reads newly written
    inputs and gives the bits of an eager call on them (main layer TP4: decode M 1 and 4, prefill M 2048, all without
    atomics). Compiled by an eager call outside the capture first. Independent of how the op runs the shared expert
    (_assert_shared): today the mixed decode kernels (M 1), the fold into the slot-pair decode Down (M 4, past
    DECODE_SHARED_MIXED_MAX_WORK) or into the prefill tier's final sum (M 2048), on the capturing stream; later
    milestones may run it otherwise at some of these token counts.

    The basic tests call eagerly; the benchmark times graph replays and grades their output within the reference's
    tolerance, not their bits against an eager call.
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    _, _, prepared = _layer_weights(experts, inter, 700)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, LAYER_HIDDEN, experts, topk, seed=750, shared_weight=None
    )
    out = torch.empty_like(x)
    run = functools.partial(
        flydsl_moe_a8w4_ptpc,
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=LAYER_SWIGLU_LIMIT,
    )
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    _assert_shared()
    x2, topk_weight2, topk_ids2 = make_layer_inputs(
        tokens, LAYER_HIDDEN, experts, topk, seed=751, shared_weight=None
    )
    x.copy_(x2)
    topk_weight.copy_(topk_weight2)
    topk_ids.copy_(topk_ids2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    want = flydsl_moe_a8w4_ptpc(
        x2, prepared, topk_weight2, topk_ids2, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert torch.equal(_bits(out), _bits(want))


@pytest.mark.parametrize("tokens", [1, 4], ids=["m1", "m4"])
def test_layer_inplace_output(tokens):
    """output = hidden_states with the shared expert gives the bits of a separate output (main layer TP4 decode M 1
    and 4, no atomics): whatever runs the shared expert reads the input before anything overwrites it. Independent of
    how the op runs the shared expert (_assert_shared): today the mixed decode kernels at M 1, whose Gate/Up reads the
    input before their Down writes the output, and the fold at M 4 (past DECODE_SHARED_MIXED_MAX_WORK), whose GEMMs
    read the routed path's quantized input; later milestones may run it otherwise at some of these token counts.

    The basic tests pass a separate output; test_layer_fold_tiers and test_layer_fold_fallback run every folded tier
    in place.
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    _, _, prepared = _layer_weights(experts, inter, 700)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, LAYER_HIDDEN, experts, topk, seed=760, shared_weight=None
    )
    want = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    result = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, output=x, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert result is x
    _assert_shared()
    assert torch.equal(_bits(x), _bits(want))


def test_fused_moe_shared_expert_id():
    """fused_moe's shared_expert_id on tagged weights: with a shared expert, -1 (the default) and E give the bits of
    the direct call (the shared expert runs either way: it is in the weight) and other values (E - 1, E + 1, None, True,
    float E) raise ValueError; without one, any value but -1 raises ValueError, E included.

    The basic test passes E.
    """
    experts, topk, inter = 8, 6, 576
    _, _, prepared = _layer_weights(experts, inter, 780)
    x, topk_weight, topk_ids = make_layer_inputs(
        4, LAYER_HIDDEN, experts, topk, seed=781, shared_weight=None
    )
    want = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
    for value in (-1, experts):
        got = _fused_moe(
            x, prepared.w1, prepared.w2, topk_weight, topk_ids, shared_expert_id=value
        )
        assert torch.equal(_bits(got), _bits(want)), value
    for value in (experts - 1, experts + 1, None, True, float(experts)):
        with pytest.raises(ValueError, match="shared_expert_id"):
            _fused_moe(
                x,
                prepared.w1,
                prepared.w2,
                topk_weight,
                topk_ids,
                shared_expert_id=value,
            )
    _, plain = _weights(EXPERTS, HIDDEN, 320, 11)
    x, topk_weight, topk_ids = make_inputs(3, HIDDEN, EXPERTS, TOPK, seed=782)
    for value in (EXPERTS, EXPERTS - 1, 0):
        with pytest.raises(ValueError, match="no shared expert"):
            _fused_moe(
                x, plain.w1, plain.w2, topk_weight, topk_ids, shared_expert_id=value
            )


# -- 2.11 masked routes (id -1) and entry checks ------------------------------------------------------------------
# A routing id of -1 masks its route (ATOM's padding rows): no tier reads its weights or adds it, whatever its weight.
# The basic test covers the main layer's per-route decode with the shared expert and the benchmark never masks; here
# the other tiers, whole rows and whole calls masked (the sortings left without a route, the dedup plan without a
# leader), the shared expert of masked rows on the streaming and prefill tiers, CUDA graph replays that move the masks,
# the reads of the per-route Gate/Up, and the entry checks. Masked cases run on ``poisoned`` weights (the
# bytes before every routed tensor are real weights, where a masked route's expert -1 would read) with NaN weights on
# the masked routes, graded against the reference (masked routes add nothing); a row without a live route is bitwise 0.


@pytest.mark.parametrize(
    "weights, tokens, topk, fused, options, path, full",
    [
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            3,
            DS_TOPK,
            False,
            {},
            _DECODE_PATHS["nn"],
            (1,),
            id="nn_ds_m3",
        ),
        pytest.param(
            (EXPERTS, HIDDEN, 320, 51),
            16,
            TOPK,
            True,
            {},
            _small_stream(),
            (3, 11),
            id="stream_bm16_qwen_m16",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            24,
            DS_TOPK,
            False,
            {},
            _DEDUP_PATHS["wide"],
            (0, 13),
            id="wide_ds_m24",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            64,
            DS_TOPK,
            False,
            {},
            ["a8w4_stream_prep"] + _stream_path(32, False),
            (7, 40),
            id="stream_ds_m64",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            64,
            DS_TOPK,
            False,
            {},
            ["a8w4_stream_prep"] + _stream_path(32, False),
            "all",
            id="stream_all",
        ),
        pytest.param(
            (MIMO_EXPERTS, MIMO_HIDDEN, 512, 41),
            1600,
            MIMO_TOPK,
            False,
            {},
            ["a8w4_stream_prep", "a8w4_down_prefill_bm64"],
            (9, 600),
            id="prefill_prep_mimo_m1600",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            200,
            DS_TOPK,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256),
            (4, 150),
            id="prefill_prep_ds_m200",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            200,
            DS_TOPK,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256),
            "all",
            id="prefill_prep_all",
        ),
        pytest.param(
            (5, 512, 64, 576),
            300,
            2,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256),
            (8, 299),
            id="prefill_prep_h512",
        ),
        pytest.param(
            (5, 512, 64, 576),
            300,
            2,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256),
            "all",
            id="prefill_prep_h512_all",
        ),
    ],
)
def test_masked_routes(monkeypatch, weights, tokens, topk, fused, options, path, full):
    """Masked routes on every tier but the basic test's (routed experts only): a random 30% of the ids and the case's
    rows whole, on poisoned weights with NaN weights on the masked routes; graded, the rows without a live route
    bitwise 0, the kernels asserted. The per-route decode Gate/Up and Down (nn); the H 5120 dedup plan (wide, groups
    of 2) with the slot-pair Down; streaming (Qwen M 16, blocks of 16; DeepSeek I 576, blocks of 32: its Down adds
    only the sorted routes atomically, no a8w4_route_sum); prefill sorted by the prep (DeepSeek M 200 with block_m;
    MiMo TP4 M 1600, I 512 at 50 rows per expert, the basic test's prefill case; H 512, E 5, topk 2 M 300 with
    block_m): its Down writes loc_ids only for sorted routes, the prep sets the others to -1, which a8w4_route_sum
    skips. Behind the prefill Down a8w4_route_sum is asserted (_ROUTE_SUMS), behind the streaming Down its absence.
    The ``_all`` cases mask every route: the dedup plan has no leader, the prep no row (num_valid 0), and the output
    is all zero. The H 5120 decode limit is pinned to pyhip M17's 32 tokens, so that the wide case (M 24) masks the
    H 5120 dedup decode whatever the default limit (the other H 5120 cases run M 3 or more than 32 tokens).

    The basic test masks the per-route decode (np) only; the benchmark never masks.
    """
    monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_MAX_M", 32)
    experts, hidden, inter, seed = weights
    w, prepared = poisoned(*_weights(experts, hidden, inter, seed))
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, prepared.experts, topk, seed=seed + 3, shared=fused
    )
    topk_weight, topk_ids = mask_routes(topk_weight, topk_ids, seed=seed + 4, full=full)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(path)
    path = moe_a8w4.last_path
    prefill = any(name.startswith("a8w4_down_prefill") for name in path)
    assert any(name in _ROUTE_SUMS for name in path) == prefill, path
    assert_close(out, ref)
    dead = (topk_ids < 0).all(dim=1).nonzero().flatten().tolist()
    assert set(range(tokens) if full == "all" else full) <= set(dead)
    assert _zero_rows(out, dead)


@pytest.mark.parametrize(
    "shape, tokens, tier",
    [("ds_layer_tp8", 64, "stream"), ("ds_layer_tp4", 2048, "prefill")],
    ids=["tp8_m64_stream_atomic", "tp4_m2048_prefill"],
)
def test_layer_masked_routes_tiers(shape, tokens, tier):
    """Masked routes in layer calls beyond the basic test's decode: TP8 streaming (I 320: the routed experts' atomic
    Down adds only the sorted routes into the output) and TP4 prefill, with the shared expert however the op runs it
    (_assert_layer_path; today folded: at M 64 its Down writes the output the atomic Down adds onto, at M 2048 it is
    a8w4_route_sum's last column). A random 30% of the routed ids, rows 1 and 5 whole and row 2's first id masked:
    ATOM's rule drops the shared expert of every row whose first id is -1, so rows 1 and 5 are bitwise 0. Poisoned
    weights, graded against reference_layer.

    The basic test masks the per-route decode only; the benchmark never masks.
    """
    experts, topk, inter = LAYER_SHAPES[shape]
    *_, topk_ids, out = _layer_check(
        experts,
        topk,
        inter,
        tokens,
        seed=700,
        mask={"seed": 712, "full": (1, 5), "first": (2,)},
    )
    assert int((topk_ids[:, 0] < 0).sum()) > 3
    assert _zero_rows(out, [1, 5])
    _assert_layer_path(tier)


@pytest.mark.parametrize("tokens", [8, 64], ids=["m8_wide", "m64_stream"])
def test_masked_graph_replay(tokens):
    """One CUDA graph of a layer call replays with the masks moved (main layer TP4: decode wide-dd M 8, the plan in the
    graph; streaming M 64, the prep in the graph; the shared expert however the op runs it, _assert_shared): the
    routing tensors are rewritten with another random mask plus whole rows, with no mask and with every routed id
    masked, and each replay gives the bits of an eager call on the same inputs (graded against reference_layer
    instead where the routes are added atomically, _path_atomic: their order changes from run to run), all zero when
    every row is masked. Poisoned weights.

    The other graph cases replay unmasked routing; the masked eager cases capture nothing.
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    w, shared, prepared = _layer_weights(experts, inter, 700)
    w, prepared = poisoned(w, prepared)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, LAYER_HIDDEN, prepared.experts, topk, seed=760, shared_weight=None
    )
    original = (topk_weight.clone(), topk_ids.clone())
    masked_weight, masked_ids = mask_routes(*original, seed=761, routed=topk)
    topk_weight.copy_(masked_weight)
    topk_ids.copy_(masked_ids)
    out = torch.empty_like(x)
    run = functools.partial(
        flydsl_moe_a8w4_ptpc,
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=LAYER_SWIGLU_LIMIT,
    )
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    _assert_shared()
    for seed, full in ((762, (0, 3)), (None, ()), (763, "all")):
        weights, ids = (
            original
            if seed is None
            else mask_routes(*original, seed=seed, full=full, routed=topk)
        )
        topk_weight.copy_(weights)
        topk_ids.copy_(ids)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        want = flydsl_moe_a8w4_ptpc(
            x, prepared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        )
        if _path_atomic(moe_a8w4.last_path):
            ref = reference_layer(
                x, w, shared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT
            )
            assert_close(out, ref)
        else:
            assert torch.equal(_bits(out), _bits(want)), (seed, full)
        dead = (ids[:, :-1] < 0).all(dim=1).nonzero().flatten().tolist()
        assert _zero_rows(out, dead)
        assert set(range(tokens) if full == "all" else full) <= set(dead)
        assert full or not dead


def test_masked_gateup_reads_no_weights():
    """The per-route decode Gate/Up reads no weights of a masked route: launched on poisoned weights (expert -1, where
    a masked route would read, is real data), every masked route's f32 row is exactly 0 and every other row has the bits
    of the launch without masks. DeepSeek M 3 (x quantized in the kernel, the SwiGLU limit).

    End to end a read would not show (the Down kernels drop masked routes by a select).
    """
    (w, prepared), hidden, topk, tokens, limit = (
        _ds_weights(),
        DS_HIDDEN,
        DS_TOPK,
        3,
        SWIGLU_LIMIT,
    )
    _, view = poisoned(w, prepared)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, view.experts, topk, seed=850, shared=False
    )
    _, masked = mask_routes(topk_weight, topk_ids, seed=851, full=(1,))
    I, device = view.inter_padded, x.device.index
    kernel = _routed_decode_kernels(
        2 * I, hidden, I, topk, False, False, limit, device
    )[0]

    def gateup(ids):
        mid = torch.full((tokens, topk, I), float("nan"))
        moe_a8w4._launch(
            "a8w4_gateup_decode",
            kernel,
            x,
            topk_weight,
            ids,
            view.w1,
            view.w1_delta,
            view.w1_scale,
            mid,
            tokens,
        )
        return mid

    mid0 = gateup(topk_ids)
    mid1 = gateup(masked)
    live = masked >= 0
    assert 0 < int((~live).sum()) < live.numel()
    assert bool((mid1[~live] == 0).all())
    assert torch.equal(mid1[live].view(torch.int32), mid0[live].view(torch.int32))


@pytest.mark.parametrize(
    "hidden, tokens", [(512, 300), (4096, 100)], ids=["h512_m300", "h4096_m100"]
)
def test_large_inter_streams_without_tiles(hidden, tokens):
    """Without a tile option, a shape whose padded I the prefill Down cannot stage (block_m 64 * I_pad > 64 KiB: I 1088)
    streams past the decode tier even with many rows per expert (E 5, topk 2: 120 / 40 rows, past the streaming
    limit), in blocks of 64 with the atomic Down, graded; with a tile option the same calls raise (test_entry_checks).
    H 4096 I 1088 at M 100 lies past the H 4096 decode limit (DECODE_MAX_M).

    The basic tests and the benchmark have I_pad <= 576, where the prefill Down holds every shape.
    """
    w, prepared = _small_weights(hidden, 1088)
    x, topk_weight, topk_ids = make_inputs(tokens, hidden, 5, 2, seed=860)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(["a8w4_stream_prep"] + _stream_path(64, False))
    assert "a8w4_route_sum" not in moe_a8w4.last_path, moe_a8w4.last_path
    assert_close(out, ref)


def test_entry_checks(monkeypatch):
    """Errors raised before any kernel runs (ValueError; last_path empty, the output untouched):
    - padded I above what the prefill Down stages in LDS (block_m * I_pad <= 64 KiB) when a tile option selects the
      prefill tier: I 1088 at H 512 with block_m 64 (M 40) and with a shared expert (H 5120, block_m 64: no
      shared-expert kernel either); block_m 128 with I 576 (DeepSeek). Without a tile option such shapes stream
      (test_large_inter_streams_without_tiles); streaming (H 512, M 48, blocks of 32) runs I 1088 (graded; decode in
      test_shape_fallbacks).
    - with DEBUG_CHECKS (AITER_MOE_A8W4_DEBUG=1, a host synchronization): a token's routed id repeated, an id of E or
      -2; -1, repeated or not, passes. Without it a repeated id runs (test_duplicate_ids_keep_every_route).
    The 32-bit offset limit M * topk * I_pad * 4 < 2**31 is checked by test_rejects_bad_inputs and, with a shared
    expert (its column not counted), test_shared_rejects.

    The basic tests and the benchmark pass I_pad <= 576, the default block_m and valid routing.
    """

    def rejects(prepared, x, topk_weight, topk_ids, match, **options):
        out = torch.full_like(x, float("nan"))
        with pytest.raises(ValueError, match=match):
            flydsl_moe_a8w4_ptpc(
                x, prepared, topk_weight, topk_ids, output=out, **options
            )
        assert moe_a8w4.last_path == [] and bool(out.isnan().all())

    w, prepared = _small_weights(512, 1088)
    x, topk_weight, topk_ids = make_inputs(40, 512, 5, 2, seed=860)
    rejects(prepared, x, topk_weight, topk_ids, "padded I 1088", block_m=64)
    w, prepared = _small_weights(512, 1088)
    x, topk_weight, topk_ids = make_inputs(48, 512, 5, 2, seed=861)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(_stream_path(32, False))
    assert_close(out, ref)
    _, _, layer = _layer_weights(4, 1088, 862)
    x, topk_weight, topk_ids = make_layer_inputs(40, LAYER_HIDDEN, 4, 2, seed=863)
    rejects(layer, x, topk_weight, topk_ids, "padded I 1088", block_m=64)
    w, prepared = _ds_weights()
    x, topk_weight, topk_ids = make_inputs(40, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=864)
    rejects(prepared, x, topk_weight, topk_ids, "block_m 128", block_m=128)

    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(3, HIDDEN, EXPERTS, TOPK, seed=865)
    repeated = topk_ids.clone()
    repeated[1, 3] = repeated[1, 0]
    masked = topk_ids.clone()
    masked[1, :4] = -1
    monkeypatch.setattr(moe_a8w4, "DEBUG_CHECKS", True)
    rejects(prepared, x, topk_weight, repeated, "distinct")
    for value in (EXPERTS, -2):
        bad = topk_ids.clone()
        bad[2, 5] = value
        rejects(prepared, x, topk_weight, bad, "routed expert ids")
    out, ref = _a8w4_run(w, prepared, x, topk_weight, masked)
    assert_close(out, ref)
    monkeypatch.undo()
    flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, repeated)


# -- 2.12 mixed decode: the shared expert as one more route of the decode kernels -----------------------------------
# A layer call runs the shared expert inside the two decode kernels (decode_a8w4's mixed Gate/Up and Down;
# moe_a8w4.last_path then ends with "shared_mixed") while the routed experts take the decode tier, M stays within
# DECODE_SHARED_MIXED_MAX_M and M * routed topk * I_pad within DECODE_SHARED_MIXED_MAX_WORK (moe_a8w4._mixed_decode);
# every other call folds it (2.13). Like the per-route decode kernels quantizing in the kernel, they take one IEEE
# reciprocal per row and correct each product with FMAs (_quant_fast), which must give the FP8 bytes of the IEEE
# division; rows whose abs-max is outside _fast_div_ok divide. The basic tests run the mixed decode at main TP4 M 1
# and TP8 M 4, the benchmark at every layer shape from M 1 to its last mixed M (graded; shared weight 1, the clamp
# seldom hit), test_layer_graph_replay and test_layer_inplace_output at main TP4 M 1; none of them asserts which
# kernels run the shared expert. Here: both sides of each layer shape's boundary with the fold and of the token
# limit, the work limit read at the call, the FP8 bytes of the fast quantization at the edges of its range and at
# rounding ties, a shared expert dropped by its last id, the reads of the mixed Gate/Up, and shapes beyond the layers'.


def _assert_mixed():
    """The last call ran the shared expert inside the two decode kernels: its A8W4 kernels are one decode Gate/Up and
    one decode Down (a8w4_gateup_decode / a8w4_down_decode, named as the routed ones), moe_a8w4.last_path holds the
    "shared_mixed" marker, and no other step ran the shared expert (no folded GEMM, no side stream).
    """
    path = moe_a8w4.last_path
    kernels = [name for name in path if name.startswith(("a8w4_gateup", "a8w4_down"))]
    assert kernels == ["a8w4_gateup_decode", "a8w4_down_decode"], path
    assert "shared_mixed" in path, path
    assert all("_mixed" in name for name in path if name.startswith("shared")), path


# case: (shape, M, mixed). Shape: a LAYER_SHAPES name (inputs x 10; the MTP layers in the official block format), or
# "small" (_fold_small_weights: E 8, H 512, I 64, routed topk 2), where the token limit decides. Work = M * routed topk
# * I_pad against DECODE_SHARED_MIXED_MAX_WORK (10368): main TP4 M 3 / 4 10368 / 13824, main TP8 M 5 / 6 9600 / 11520,
# MTP TP4 M 6 / 7 10368 / 12096, MTP TP8 M 8 / 10 7680 / 9600 (from M 11 on it streams); the small shape's M 10 / 11
# (1280 / 1408) lie on both sides of DECODE_SHARED_MIXED_MAX_M (10).
_MIXED_BOUNDARY_CASES = {
    "ds_tp4_m3_mixed": ("ds_layer_tp4", 3, True),
    "ds_tp4_m4_fold": ("ds_layer_tp4", 4, False),
    "ds_tp8_m5_mixed": ("ds_layer_tp8", 5, True),
    "ds_tp8_m6_fold": ("ds_layer_tp8", 6, False),
    "mtp_tp4_m6_mixed": ("mtp_layer_tp4", 6, True),
    "mtp_tp4_m7_fold": ("mtp_layer_tp4", 7, False),
    "mtp_tp8_m8_mixed": ("mtp_layer_tp8", 8, True),
    "mtp_tp8_m10_mixed": ("mtp_layer_tp8", 10, True),
    "small_m10_mixed": ("small", 10, True),
    "small_m11_fold": ("small", 11, False),
}


@pytest.mark.parametrize("case", list(_MIXED_BOUNDARY_CASES))
def test_layer_mixed_decode_boundary(case):
    """Both sides of the mixed decode's boundary with the fold (_MIXED_BOUNDARY_CASES): the work M * routed topk *
    I_pad decides it at the layers (each layer shape's last mixed M and, where the decode tier goes on, its first folded
    M; MTP TP8 M 8, past the M where H 5120 deduplicates the routed Gate/Up, mixes as well), DECODE_SHARED_MIXED_MAX_M
    on the small shape. Graded against reference_layer with a random shared weight per token; the mixed kernels
    asserted (_assert_mixed) or the fold on the routed decode tier (_assert_folded).

    The basic tests and the benchmark (its main TP4 M 3 / 4, main TP8 M 5 / 6, MTP TP4 M 6 / 8 and MTP TP8 M 8 / 10)
    accept any path for the shared expert, run the layers with the clamp seldom hit and never the small shape's limit;
    the benchmark has no MTP TP4 M 7.
    """
    shape, tokens, mixed = _MIXED_BOUNDARY_CASES[case]
    if shape == "small":
        w, shared, prepared = _fold_small_weights()
        x, topk_weight, topk_ids = make_layer_inputs(
            tokens, 512, 8, 2, seed=940 + tokens, shared_weight=None
        )
        out = flydsl_moe_a8w4_ptpc(
            x, prepared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        )
        assert_close(
            out,
            reference_layer(
                x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
            ),
        )
    else:
        experts, topk, inter = LAYER_SHAPES[shape]
        fmt = "block" if shape.startswith("mtp") else "row"
        _layer_check(experts, topk, inter, tokens, seed=700, fmt=fmt, amplify=10.0)
    if mixed:
        _assert_mixed()
    else:
        _assert_folded("decode")


# case: (DECODE_SHARED_MIXED_MAX_WORK, mixed) at main TP4 M 4, whose work 4 * 6 * 576 = 13824 folds by default.
_MIXED_WORK_CASES = {
    "ds_tp4_m4_at_limit": (4 * 6 * 576, True),
    "ds_tp4_m4_over_limit": (4 * 6 * 576 - 1, False),
}


@pytest.mark.parametrize("case", list(_MIXED_WORK_CASES))
def test_layer_mixed_decode_work_limit(monkeypatch, case):
    """DECODE_SHARED_MIXED_MAX_WORK bounds M * routed topk * I_pad inclusively and is read at the call: set to main
    TP4 M 4's work the call mixes, one below it folds. Graded against reference_layer (inputs x 10, the clamp hit), the
    mixed kernels asserted (_assert_mixed) or the fold (_assert_folded). Skipped once the op no longer has the
    constant.

    test_layer_mixed_decode_boundary runs the default limit only; the basic tests and the benchmark assert no path.
    """
    limit, mixed = _MIXED_WORK_CASES[case]
    if not hasattr(moe_a8w4, "DECODE_SHARED_MIXED_MAX_WORK"):
        pytest.skip("DECODE_SHARED_MIXED_MAX_WORK removed")
    monkeypatch.setattr(moe_a8w4, "DECODE_SHARED_MIXED_MAX_WORK", limit)
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    _layer_check(experts, topk, inter, 4, seed=700, amplify=10.0)
    if mixed:
        _assert_mixed()
    else:
        _assert_folded("decode")


def test_layer_mixed_decode_masked():
    """Masks on the mixed decode (the basic masked layer case, main TP4 M 4, folds: DECODE_SHARED_MIXED_MAX_WORK):
    MTP layer TP8 (4 Down waves, I 288 padded to 320) at M 4,
    a random 30% of the routed ids, every routed id of row 1 and the first id of row 2 masked, row 3's last column
    another expert than E; ATOM's rule drops the shared expert of rows 1 to 3, whose shared weights are NaN (as the
    masked routes' are), and keeps row 0's. Poisoned routed weights, inputs x 10; graded against reference_layer, row 1
    bitwise 0. Then every routed id masked: the output is bitwise 0.

    The mixed Down decides on its own whether a token's shared expert is live: had it taken a dropped one, it would have
    read the zero row the Gate/Up wrote for it, so only the NaN weight can show it. The benchmark never masks.
    """
    experts, topk, inter = LAYER_SHAPES["mtp_layer_tp8"]
    w, shared, prepared = _layer_weights(experts, inter, 700)
    w, prepared = poisoned(w, prepared)
    E = prepared.experts
    x, topk_weight, topk_ids = make_layer_inputs(
        4, LAYER_HIDDEN, E, topk, seed=790, shared_weight=None
    )
    x = (x.float() * 10.0).to(torch.bfloat16)
    weights, ids = mask_routes(
        topk_weight, topk_ids, seed=791, full=(1,), first=(2,), routed=topk
    )
    for row in (0, 3):
        weights[row, 0], ids[row, 0] = topk_weight[row, 0], topk_ids[row, 0]
    ids[3, -1] = E - 1
    weights[1:, -1] = float("nan")
    live = shared_route_weight(weights, ids, E)
    assert bool(live[0] > 0) and bool((live[1:] == 0).all()), live
    out = torch.full_like(x, float("nan"))
    flydsl_moe_a8w4_ptpc(
        x, prepared, weights, ids, output=out, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    _assert_mixed()
    assert_close(
        out,
        reference_layer(x, w, shared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT),
    )
    assert _zero_rows(out, [1])
    weights, ids = mask_routes(weights, ids, seed=792, full="all", routed=topk)
    out = flydsl_moe_a8w4_ptpc(
        x, prepared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    _assert_mixed()
    assert _zero_rows(out, range(4))


def test_mixed_decode_quant_bitexact():
    """The decode kernels that quantize their rows themselves (decode_a8w4._quant_fast: one reciprocal per row, FMA
    corrections per value) give the FP8 bytes of the IEEE division (_ieee_quant) on _fast_quant_rows (abs-max on both
    sides of both ends of the fast range, zero and subnormal rows, rounding ties), shown on the outputs, which equal
    those of launches on the reference's bytes: the per-route Gate/Up quantizing x (one correction) and the mixed
    Gate/Up's routed rows have the bits of the per-route Gate/Up reading x pre-quantized by _ieee_quant, one launch per
    x row with the routed channel factors times 2**k (so that tiny or huge rows give normal f32 results); the
    per-route Down and the mixed Down quantizing the intermediate (two corrections) have the bits of the per-route
    Down on the reference's FP8 values (row abs-max 240: its row scale 1 and its quantization exact; the reference's
    scale moves into the routing weight, the same product in the same order), one token per row, each with one live
    route (expert t, a different column each; 2**k split between its channel factors, 2**(k // 2) for k > 0, and its
    weight, so that every product stays normal), its other routes masked, its shared expert dropped and their
    intermediate rows NaN.
    E 12, topk 6, H 5120, I 576 (K padded to 640 for the shared Down), SwiGLU limit 10.

    These kernels keep their bytes in LDS (test_fast_row_quant_bitexact reads those of the row kernels), and the Down's
    BF16 output may hide a flipped byte of a small quotient. End to end a few flipped FP8 values stay within the
    tolerance; the basic tests and the benchmark quantize random rows of ordinary size only.
    """
    experts, topk, inter = 12, 6, 576
    _, _, prepared = _layer_weights(experts, inter, 780)
    H, I, limit = prepared.hidden, prepared.inter_padded, LAYER_SWIGLU_LIMIT
    device = torch.cuda.current_device()
    mixed_gateup, mixed_down = moe_a8w4._decode_mixed_kernels(
        2 * I, H, I, prepared.shared_w2.shape[1], topk + 1, experts, limit, device
    )
    gateup, down = _routed_decode_kernels(
        2 * I, H, I, topk, False, False, limit, device
    )
    gateup_pq = _routed_decode_kernels(2 * I, H, I, topk, True, False, limit, device)[0]
    _, topk_weight, topk_ids = make_layer_inputs(1, H, experts, topk, seed=799)
    routed_weight, routed_ids = (
        topk_weight[:, :topk].contiguous(),
        topk_ids[:, :topk].contiguous(),
    )
    rows, exps = _fast_quant_rows(H)
    for r, k in enumerate(exps):
        x = rows[r : r + 1].to(torch.bfloat16)
        xq, xs = _ieee_quant(x)
        factor = prepared.w1_scale * 2.0**k
        w = (prepared.w1, prepared.w1_delta, factor)
        mid = torch.full((1, topk + 1, I), float("nan"))
        got = torch.full((1, topk, I), float("nan"))
        want = torch.full_like(got, float("nan"))
        moe_a8w4._launch(
            "a8w4_gateup_decode_mixed",
            mixed_gateup,
            x,
            topk_ids,
            *w,
            prepared.shared_w1,
            prepared.shared_w1_scale,
            mid,
            1,
        )
        moe_a8w4._launch(
            "a8w4_gateup_decode",
            gateup,
            x,
            routed_weight,
            routed_ids,
            *w,
            got,
            1,
        )
        moe_a8w4._launch(
            "a8w4_gateup_decode_pq",
            gateup_pq,
            xq,
            xs.view(-1),
            routed_ids,
            *w,
            want,
            1,
        )
        assert bool(want.isfinite().all()) and bool((want != 0).any()) == (r != 6), r
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), r
        assert torch.equal(mid[:, :topk].view(torch.int32), want.view(torch.int32)), r

    rows, exps = _fast_quant_rows(I)
    tokens = rows.shape[0]
    assert tokens <= experts
    q, scale = _ieee_quant(rows)
    mid = torch.full((tokens, topk + 1, I), float("nan"))
    mid_q = torch.full_like(mid, float("nan"))
    ids = torch.full((tokens, topk + 1), -1, dtype=torch.int32)
    weights = torch.full((tokens, topk + 1), float("nan"))
    weights_q = torch.full_like(weights, float("nan"))
    factor = prepared.w2_scale.clone()
    for t, k in enumerate(exps):
        c = t % topk
        mid[t, c], mid_q[t, c], ids[t, c] = rows[t], q[t].float(), t
        j = max(k, 0) // 2
        factor[t] *= 2.0**j
        weights[t, c], weights_q[t, c] = 2.0 ** (k - j), scale[t, 0] * 2.0 ** (k - j)
    assert q.float().abs().amax(dim=1).tolist() == [
        240.0 * (t != 6) for t in range(tokens)
    ]
    w = (prepared.w2, prepared.w2_delta, factor)
    outs = []
    for rows_in, w_in, mixed in (
        (mid, weights, True),
        (mid, weights, False),
        (mid_q, weights_q, False),
    ):
        out = torch.full((tokens, H), float("nan"), dtype=torch.bfloat16)
        if mixed:
            moe_a8w4._launch(
                "a8w4_down_decode_mixed",
                mixed_down,
                rows_in,
                ids,
                w_in,
                *w,
                prepared.shared_w2,
                prepared.shared_w2_scale,
                out,
                tokens,
            )
        else:
            routed = w_in[:, :topk].contiguous()
            moe_a8w4._launch(
                "a8w4_down_decode",
                down,
                rows_in[:, :topk].contiguous(),
                routed,
                ids[:, :topk].contiguous(),
                routed,
                *w,
                out,
                tokens,
            )
        outs.append(out)
    want = outs[2]
    assert bool(want.isfinite().all())
    assert (want != 0).any(dim=1).tolist() == [t != 6 for t in range(tokens)]
    assert torch.equal(_bits(outs[1]), _bits(want))
    assert torch.equal(_bits(outs[0]), _bits(want))


def test_mixed_gateup_reads_no_weights():
    """The mixed decode Gate/Up reads no weights of a masked route nor of a dropped shared expert: launched on poisoned
    routed weights and a shared Gate/Up weight of NaN bytes (0x80 in E4M3FNUZ), with a random 30% of the routed ids, all
    of row 1's and row 2's first one masked and row 3's last id another expert than E, every masked route's f32 row and
    the shared rows of rows 1 to 3 are exactly 0, the other routed rows have the bits of the launch without masks, and
    row 0's shared row is NaN (the poison shows where it is read). E 8 (7 in the poisoned view), topk 6, H 5120,
    I 576, M 4, no SwiGLU limit.

    End to end a read would not show: the Down drops masked routes and dropped shared experts by a select.
    """
    w, _, prepared = _layer_weights(8, 576, 780)
    _, view = poisoned(w, prepared)
    topk, E, H, I = 6, view.experts, view.hidden, view.inter_padded
    gateup = moe_a8w4._decode_mixed_kernels(
        2 * I,
        H,
        I,
        view.shared_w2.shape[1],
        topk + 1,
        E,
        0.0,
        torch.cuda.current_device(),
    )[0]
    nan_w1 = torch.full_like(view.shared_w1.view(torch.uint8), 0x80)
    x, topk_weight, topk_ids = make_layer_inputs(4, H, E, topk, seed=796)
    _, masked = mask_routes(
        topk_weight, topk_ids, seed=797, full=(1,), first=(2,), routed=topk
    )
    masked[0, 0], masked[3, 0] = topk_ids[0, 0], topk_ids[3, 0]
    masked[3, -1] = E - 1

    def gateup_mid(ids):
        mid = torch.full((4, topk + 1, I), float("nan"))
        moe_a8w4._launch(
            "a8w4_gateup_decode_mixed",
            gateup,
            x,
            ids,
            view.w1,
            view.w1_delta,
            view.w1_scale,
            nan_w1.view(view.shared_w1.dtype),
            view.shared_w1_scale,
            mid,
            4,
        )
        return mid

    mid0, mid1 = gateup_mid(topk_ids), gateup_mid(masked)
    live = masked[:, :topk] >= 0
    assert 0 < int((~live).sum()) < live.numel()
    routed0, routed1 = mid0[:, :topk], mid1[:, :topk]
    assert bool((routed1[~live] == 0).all())
    assert torch.equal(routed1[live].view(torch.int32), routed0[live].view(torch.int32))
    assert bool((mid1[1:, topk] == 0).all())
    assert bool(mid1[0, topk].isnan().all()) and bool(mid0[:, topk].isnan().all())


@pytest.mark.parametrize(
    "experts, topk, hidden, inter, tokens, mixed",
    [
        (16, 15, 2560, 192, 3, True),
        (8, 1, 512, 64, 3, True),
        (20, 16, 2560, 192, 3, False),
        (12, 8, 4096, 256, 2, True),
        # M 7 (work 14336: folds) is the last token count of the H 4096 decode tier (DECODE_MAX_M = 7).
        (12, 8, 4096, 256, 7, False),
    ],
    ids=[
        "topk15_h2560",
        "topk1_h512",
        "topk16_not_mixed",
        "mimo_h4096_m2_mixed",
        "mimo_h4096_m7_fold",
    ],
)
def test_mixed_decode_shapes(experts, topk, hidden, inter, tokens, mixed):
    """The mixed decode on shapes beyond the layers' (a random shared weight per token, inputs x 10, SwiGLU limit 10):
    topk 15 + the shared expert (16 Down waves, the most it takes) at H 2560 (5 x pairs per Gate/Up thread, the last
    one packed alone) and I 192 (the shared Down's K padded to 256), M 3; topk 1 + the shared expert (2 waves) at H 512
    (one x pair per thread) and I 64, M 3; the MiMo shape (H 4096, topk 8, I 256, E 12) at M 2. Past it the fold, on
    the decode tier's slot-pair Down at any M: topk 16 + the shared expert (17 columns) does not fit the mixed Down and
    folds at M 3 (where routed-only calls take the Down without it); the MiMo shape at M 7 (work 14336) folds with the
    per-route Gate/Up on pre-quantized rows (routed-only calls of that shape take the Down that quantizes its rows).
    Graded against reference_layer, the mixed kernels or the fold's decode kernels asserted.

    The basic tests and the benchmark run the mixed decode and the fold at H 5120, topk 3 or 6 + 1 and I 320 or 576
    only, and assert no kernel of either.
    """
    w = make_weights(experts, hidden, inter, seed=800 + topk)
    shared = make_shared_weights(hidden, inter, seed=801 + topk)["row"]
    prepared = _prepare_shared(w, shared)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, hidden, experts, topk, seed=802 + topk, shared_weight=None
    )
    x = (x.float() * 10.0).to(torch.bfloat16)
    out = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert_close(
        out,
        reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        ),
    )
    if mixed:
        _assert_mixed()
    else:
        _assert_folded("decode")
        _assert_path(["a8w4_gateup_decode_pq", "a8w4_down_decode_pq_shared"])


# -- 2.13 shared fold: the shared expert in the routed experts' last reduction --------------------------------------
# Every layer call the mixed decode does not take folds the shared expert into the routed experts' last reduction
# (moe_a8w4._SharedFold on the routed tier of moe_a8w4._fold_tier): two FP8 GEMMs (aiter's FlyDSL preshuffle GEMM, in
# chunks of at most 65536 rows) on the routed path's quantized input with shared_a8w4's mid kernel between them write
# the output (the decode tier's slot-pair Down, at any M, then adds the routes in f32, the streaming atomic Down in
# BF16; an output off a 16-byte boundary gets the GEMM's output through a copy) or M more rows of the prefill tier's
# routes buffer that a8w4_route_sum adds as one more column; on a side stream beside the decode and streaming tiers,
# on the current stream behind the prefill tier. Where the slot-pair Down cannot run (w2 past 4 GiB, or a shape past
# its limits) the decode tier's calls stream instead. The basic layer tests (streaming M 64, prefill M 2048, the wide
# decode M 8) and the
# benchmark's layer cases (every folded tier of the layers; unmasked routing, the last column E with weight 1) grade
# it without asserting how the shared expert runs. Here: every folded tier with masks and with rows whose last column
# is a routed expert instead of E, in place and under graph replay; the GEMM tiles' LDS race; an output that is not
# 16-byte aligned; the fallback to streaming; small M with a tile option and both sides of the GEMMs' row limit; the
# compile-cache keys; shared_a8w4's own Down kernel against the preshuffle GEMM and its mid kernel's lanes per row
# against one row per wave, bit for bit. The cases pin the routed tier, the side stream by tier, a8w4_route_sum
# exactly behind the prefill Down, no mixed kernels, the copy exactly where the shared Down GEMM writes an unaligned
# output, the streaming Down of the fallback; bits are compared where no Down adds atomically (_path_atomic), else
# graded.

# case: (shape, tokens, options, routed tier as _path_tier names it, or the tiers it may take). Shape: a LAYER_SHAPES
# name, or "small" (_fold_small_weights: E 8, H 512, I 64, routed topk 2). The tiers: the pre-quantized per-route Gate/Up with the slot-pair Down (main TP4 M 6), the wide decode
# (deduplicated Gate/Up, M 16), streaming with the atomic Down at I 576 (blocks of 16) and at I 320 (blocks of 16, and
# MTP TP8 M 1365: 32 rows per expert in blocks of 64, below 4096 routes, past 1024 tokens still on the side stream),
# prefill sorted by the prep (MTP TP4 / TP8 M 1366, I 576 / 320: 4098 routes, 32 rows per expert past the 24 that H 5120
# streams from 4096 routes on; the small shape at M 2048 with a tile option), prefill with a tile option at M 40.
_FOLD_CASES = {
    "decode_pq_ds_tp4_m6": ("ds_layer_tp4", 6, {}, "decode"),
    "wide_dd_mtp_tp8_m16": ("mtp_layer_tp8", 16, {}, "dedup"),
    "stream_i576_mtp_tp4_m128": ("mtp_layer_tp4", 128, {}, "stream"),
    "prefill_prep_mtp_tp4_m1366": ("mtp_layer_tp4", 1366, {}, "prefill"),
    "stream_atomic_mtp_tp8_m64": ("mtp_layer_tp8", 64, {}, "stream"),
    "stream_bm64_mtp_tp8_m1365": ("mtp_layer_tp8", 1365, {}, "stream"),
    "prefill_prep_mtp_tp8_m1366": ("mtp_layer_tp8", 1366, {}, "prefill"),
    "prefill_prep_small_m2048": ("small", 2048, {"block_m": 64}, "prefill"),
    "prefill_tile_small_m40": ("small", 40, {"block_m": 64}, "prefill"),
}


@functools.cache
def _fold_small_weights():
    """E 8, H 512, I 64 MXFP4 routed experts with a shared expert (per-row format): ``(w, shared, prepared)``."""
    w = make_weights(8, 512, 64, seed=870)
    shared = make_shared_weights(512, 64, seed=871)["row"]
    return w, shared, _prepare_shared(w, shared)


def _fold_weights(shape):
    """``(w, shared, prepared, topk, hidden)`` of a _FOLD_CASES shape (layer weights of _layer_weights(seed 700))."""
    if shape == "small":
        return (*_fold_small_weights(), 2, 512)
    experts, topk, inter = LAYER_SHAPES[shape]
    return (*_layer_weights(experts, inter, 700), topk, LAYER_HIDDEN)


def _fold_routing(tokens, hidden, experts, topk, seed):
    """Layer inputs (make_layer_inputs, a random shared weight per token) with a random 30% of the routed ids, every
    routed id of row 1 and the first id of row 2 masked (mask_routes), and in every fourth row from row 3 the last
    column a routed expert the row does not select instead of E: ATOM's rule drops the shared expert of rows 1, 2 and
    those, and a kernel or sorting that took the last column for a routed one would add that expert with its weight.
    """
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, hidden, experts, topk, seed=seed, shared_weight=None
    )
    topk_weight, topk_ids = mask_routes(
        topk_weight, topk_ids, seed=seed + 1, full=(1,), first=(2,), routed=topk
    )
    ids = topk_ids.cpu()
    for row in range(3, tokens, 4):
        taken = set(ids[row, :-1].tolist())
        ids[row, -1] = next(e for e in range(experts) if e not in taken)
    return x, topk_weight, ids.to(topk_ids.device).contiguous()


def _assert_folded(tier):
    """The last call ran the routed experts on ``tier`` (_path_tier; a tuple: one of those tiers) and the shared
    expert outside the mixed decode kernels and without a final add: what ran it, the fold, wrote where the routed
    experts' last reduction adds it (a8w4_route_sum exactly behind the prefill Down), on a side stream (shared_fork /
    shared_join) beside the decode and streaming tiers and on the current stream behind the prefill tier.
    """
    path = moe_a8w4.last_path
    tiers = (tier,) if isinstance(tier, str) else tier
    assert _path_tier(path) in tiers, (tier, path)
    _assert_shared()
    assert "shared_add" not in path, path
    assert not any("_mixed" in name for name in path), path
    prefill = _path_tier(path) == "prefill"
    assert ("shared_fork" in path) == ("shared_join" in path) == (not prefill), path
    assert any(name in _ROUTE_SUMS for name in path) == prefill, path


@pytest.mark.parametrize("case", list(_FOLD_CASES))
def test_layer_fold_tiers(monkeypatch, case):
    """Every folded tier (_FOLD_CASES) on poisoned routed weights with _fold_routing's masks and dropped shared
    experts: graded against reference_layer, the rows without a live routed id bitwise 0 (row 1: the shared expert's
    +0 plus nothing), no final add and the side stream by tier (_assert_folded; MTP TP8 M 1365 also asserts the blocks
    of 64). The same call with output = hidden_states gives the bits of the separate output (the
    chain writes the
    output, or the routes buffer, only after the kernel that quantizes the input), and a CUDA graph of the call replays
    other inputs with the masks moved with the bits of an eager call on them; where a Down adds the routes atomically
    (_path_atomic), whose order changes from run to run, both are graded instead. The H 5120 decode limit is pinned to
    pyhip M17's 32 tokens, so that the wide case (M 16) folds into the H 5120 dedup decode whatever the default limit
    (the other cases run 6 or more than 32 tokens).

    The basic tests and the benchmark grade the folded tiers on unmasked routing with E in the last column, eagerly
    (the benchmark replays graphs but grades one replay within the tolerance) into a separate output; the basic tests
    reach neither the pre-quantized decode nor the streaming blocks of 64, the benchmark no tile option, and neither
    asserts how the shared expert runs.
    """
    monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_MAX_M", 32)
    shape, tokens, options, tier = _FOLD_CASES[case]
    w, shared, prepared, topk, hidden = _fold_weights(shape)
    w, prepared = poisoned(w, prepared)
    E = prepared.experts

    def call(x, topk_weight, topk_ids, out):
        return flydsl_moe_a8w4_ptpc(
            x,
            prepared,
            topk_weight,
            topk_ids,
            output=out,
            swiglu_limit=LAYER_SWIGLU_LIMIT,
            **options,
        )

    def ref(x, topk_weight, topk_ids):
        return reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        )

    x, topk_weight, topk_ids = _fold_routing(tokens, hidden, E, topk, seed=880)
    dropped = shared_route_weight(topk_weight, topk_ids, E) == 0
    assert int(dropped.sum()) >= 2 + len(range(3, tokens, 4))
    out = torch.full_like(x, float("nan"))
    assert call(x, topk_weight, topk_ids, out) is out
    _assert_folded(tier)
    exact = not _path_atomic(moe_a8w4.last_path)
    if case.startswith("stream_bm64"):
        assert "a8w4_down_stream_bm64" in moe_a8w4.last_path, moe_a8w4.last_path
    assert_close(out, ref(x, topk_weight, topk_ids))
    dead = (topk_ids[:, :-1] < 0).all(dim=1).nonzero().flatten().tolist()
    assert 1 in dead and _zero_rows(out, dead)

    inplace = x.clone()
    assert call(inplace, topk_weight, topk_ids, inplace) is inplace
    _assert_folded(tier)
    if exact:
        assert torch.equal(_bits(inplace), _bits(out))
    else:
        assert_close(inplace, ref(x, topk_weight, topk_ids))

    static = [t.clone() for t in (x, topk_weight, topk_ids)]
    graph_out = torch.empty_like(x)
    call(*static, graph_out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(*static, graph_out)
    _assert_folded(tier)
    other = _fold_routing(tokens, hidden, E, topk, seed=890)
    for t, value in zip(static, other):
        t.copy_(value)
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    if exact:
        want = call(*other, None)
        assert torch.equal(_bits(graph_out), _bits(want))
    else:
        assert_close(graph_out, ref(*other))


def test_layer_fold_gemm_race(monkeypatch):
    """The folded shared expert's GEMMs never take a tile that races in LDS, and graph replays of a call with the
    shared GEMMs beside the routed kernels keep the bits of an eager call.

    aiter's preshuffle GEMM copies each A tile into LDS with one 16-byte load per thread (128 threads with tile_n 32,
    else 256): a tile of fewer bytes makes the copy cover rows past tile_m, which land in the other LDS stage while
    other waves may still read it (four waves with tile_m 16 and tile_k 128: wrong 8 x 16 output blocks now and then
    when other kernels share the CUs). moe_a8w4._shared_gemm asserts the tile's bytes; here its tile choice is run for
    every M up to the GEMM's 65536 rows at the layers' Gate/Up (K 5120) and Down (K 640 / 384) shapes with the GEMM
    stubbed (no assertion fails; K is a multiple of tile_k, which the GEMM requires), then MTP layer TP8 M 32 (wide
    decode: the shared Down GEMM of 32 rows on the side stream beside the routed Gate/Up; the H 5120 decode limit
    pinned to pyhip M17's 32 tokens, whatever the default) replays 1000 times with the bits of the eager call, which
    is graded (four waves with tile (16, 64, 128) there: 14-20 of 1000 replays wrong).

    The basic tests and the benchmark run the GEMMs at a few token counts, eagerly or graded within the tolerance on
    one replay; an intermittent block error passes them.
    """
    import types

    from aiter.ops.flydsl import gemm_kernels

    seen = set()

    def stub(xq, wq, x_scale, w_scale, out, tile_m, tile_n, tile_k):
        seen.add((xq.shape[1], tile_m, tile_n, tile_k))

    with monkeypatch.context() as patch:
        patch.setattr(gemm_kernels, "flydsl_preshuffle_gemm_a8", stub)
        for n, k in ((1152, 5120), (640, 5120), (5120, 640), (5120, 384)):
            wq = types.SimpleNamespace(shape=(n, k))
            for m in range(1, 65537):
                xq = types.SimpleNamespace(shape=(m, k))
                moe_a8w4._shared_gemm(xq, wq, None, None, None)
    assert all(k % tile_k == 0 for k, _, _, tile_k in seen), seen

    monkeypatch.setattr(moe_a8w4, "DECODE_WIDE_MAX_M", 32)
    experts, topk, inter = LAYER_SHAPES["mtp_layer_tp8"]
    w, shared, prepared = _layer_weights(experts, inter, 700)
    x, topk_weight, topk_ids = make_layer_inputs(
        32, LAYER_HIDDEN, experts, topk, seed=895, shared_weight=None
    )
    out = torch.empty_like(x)
    run = functools.partial(
        flydsl_moe_a8w4_ptpc,
        x,
        prepared,
        topk_weight,
        topk_ids,
        output=out,
        swiglu_limit=LAYER_SWIGLU_LIMIT,
    )
    run()
    torch.cuda.synchronize()
    _assert_folded("dedup")
    want = out.clone()
    assert_close(
        want,
        reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        ),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    wrong = 0
    for _ in range(1000):
        out.fill_(float("nan"))
        graph.replay()
        wrong += int(not torch.equal(_bits(out), _bits(want)))
    assert wrong == 0, f"{wrong} of 1000 replays differ from the eager call"


@pytest.mark.parametrize(
    "case",
    [
        "decode_pq_ds_tp4_m6",
        "stream_atomic_mtp_tp8_m64",
        "stream_i576_mtp_tp4_m128",
        "stream_bm64_mtp_tp8_m1365",
    ],
)
def test_layer_fold_unaligned_output(case):
    """An output 8 bytes past a 16-byte boundary (a view into a larger buffer) folds the shared expert as an aligned
    one does: where its Down GEMM, which stores 16 bytes per thread, writes the output (the slot-pair decode Down and
    the streaming atomic Down then add the routes onto it), the GEMM writes an aligned buffer that one more kernel
    copies into the output (shared_fold_copy); where it writes rows of the prefill routes buffer for a8w4_route_sum
    (_ROUTE_SUMS), nothing is copied. Graded against reference_layer; the buffer before the view stays untouched.
    The Down is the preshuffle GEMM at M 6-128 and shared_a8w4's own Down kernel at MTP TP8 M 1365 (I 320, from
    SHARED_DOWN_MIN_M_SHORT_K rows on), which needs a 16-byte aligned output too.

    The basic tests and the benchmark pass outputs from torch.empty, always aligned.
    """
    shape, tokens, _, tier = _FOLD_CASES[case]
    w, shared, prepared, topk, hidden = _fold_weights(shape)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, hidden, prepared.experts, topk, seed=900, shared_weight=None
    )
    buf = torch.full(
        (tokens * hidden + 4,), float("nan"), dtype=torch.bfloat16, device=x.device
    )
    out = buf[4:].view(tokens, hidden)
    assert out.data_ptr() % 16 == 8
    result = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, output=out, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert result is out
    _assert_folded(tier)
    path = moe_a8w4.last_path
    sums = any(name in _ROUTE_SUMS for name in path)
    assert ("shared_fold_copy" in path) != sums, path
    assert_close(
        out,
        reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        ),
    )
    assert bool(buf[:4].isnan().all())


@pytest.mark.parametrize("over", [0, 1], ids=["m65536_fold", "m65537"])
def test_layer_fold_max_m(over):
    """Both sides of 65536 tokens, the row limit of aiter's FlyDSL preshuffle GEMM that the fold's two GEMMs use
    (gemm_kernels.PRESHUFFLE_M_MAX; past it one GEMM call raises): at the limit the call folds the shared expert into
    the prefill with one launch per GEMM, one token more it folds too, each GEMM in chunks of at most 65536 rows (row
    views of its input, scales and output; test_launch_records[fold_prefill_chunks] counts the chunks' launches and
    bytes); both without a final add, graded against reference_layer. The small shape (E 8, H 512, I 64, routed
    topk 2) in the per-row format: prepare folds the 32x32-block format into per-row scales, so the runtime path and
    the limit do not depend on the format.

    The benchmark runs M 32768 and 98304 without asserting the path, the basic tests at most M 2048: neither is at the
    limit.
    """
    from aiter.ops.flydsl import gemm_kernels

    tokens = gemm_kernels.PRESHUFFLE_M_MAX + over
    w, shared, prepared = _fold_small_weights()
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, 512, 8, 2, seed=910, shared_weight=None
    )
    out = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)
    path = moe_a8w4.last_path
    _assert_folded("prefill")
    assert any(name.startswith("shared_fold") for name in path), path
    assert_close(out, reference_layer(x, w, shared, topk_weight, topk_ids))


@pytest.mark.parametrize("over", [0, 1], ids=["m1024", "m1025"])
def test_layer_fold_prefill_tile(over):
    """With a tile option (block_m 64: the prefill kernels at any M) the prefill tier folds the shared expert at any M
    (shared_fold_*, no final add) on the small shape (E 8, H 512, I 64, routed topk 2), on the current stream at any M
    (no shared_fork / shared_join, _assert_folded): M 1024 and M 1025, both graded against reference_layer.
    test_layer_fold_tiers runs M 40 with a tile option in place and under graph replay.

    The default dispatch of the layer shapes first reaches prefill above M 1024 (MTP M 1366, DeepSeek M 1537), so the
    benchmark (M 1024-1536 stream, M 2048 prefill) and the basic tests (prefill M 2048) never run the prefill tier at
    or below M 1024, and neither passes a tile option.
    """
    tokens = 1024 + over
    w, shared, prepared = _fold_small_weights()
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, 512, 8, 2, seed=920, shared_weight=None
    )
    out = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, block_m=64)
    _assert_folded("prefill")
    assert any(name.startswith("shared_fold") for name in moe_a8w4.last_path)
    assert_close(out, reference_layer(x, w, shared, topk_weight, topk_ids))


@functools.cache
def _fold_fallback_weights(experts, hidden, inter, seed):
    """MXFP4 routed experts with a shared expert (per-row format) of a shape outside the layers':
    ``(w, shared, prepared)``."""
    w = make_weights(experts, hidden, inter, seed=seed)
    shared = make_shared_weights(hidden, inter, seed=seed + 1)["row"]
    return w, shared, _prepare_shared(w, shared)


# case: (weights, M, the streaming Down). The decode tier's calls whose slot-pair Down cannot run: "i1088" (E 5, H 512,
# I 1088: a row's 68 16-byte pieces exceed the slot-pair Down's lanes at topk 2; blocks of 16), "w2_4gib" (the small
# shape, E 8, H 512, I 64, with w2 one byte past what the slot-pair Down's 32-bit offsets reach, through a stride-0
# view, as test_decode_down_fallback_w2_4gib), "h4096_i1088" (E 2, H 4096, I 1088: 30 rows per expert sort in blocks
# of 64, past the 28 rows the dispatch streams where the Down's tile of blocks of 64 does not fit in LDS; the prefill
# Down holds at most I 1024).
_FOLD_FALLBACK_CASES = {
    "i1088_m16": ("i1088", 16, "a8w4_down_stream_bm16"),
    "w2_4gib_m16": ("w2_4gib", 16, "a8w4_down_stream_bm16"),
    "h4096_i1088_m30_bm64": ("h4096_i1088", 30, "a8w4_down_stream_bm64"),
}


@pytest.mark.parametrize("case", list(_FOLD_FALLBACK_CASES))
def test_layer_fold_fallback(case):
    """A call with the shared expert whose routed experts take the decode tier (H 512, M 16) where the slot-pair Down
    cannot run, or lie past the decode tier where the prefill Down cannot stage the padded I (H 4096 I 1088 M 30, 30
    rows per expert; _FOLD_FALLBACK_CASES), streams instead, folding the shared expert into the streaming tier with the
    streaming Down, never the prefill Down (so such a shape is not refused): the routed
    tier and that Down asserted (_assert_folded), graded against reference_layer, into a separate output, in place
    (the bits of the separate output, graded where the Down adds atomically, _path_atomic) and into an output 8 bytes
    past a 16-byte boundary (graded, copied by shared_fold_copy exactly where the shared Down GEMM writes the output,
    the buffer before it untouched).

    The basic tests and the benchmark run the layers' shapes only, far from these limits; routed-only calls of these
    shapes decode with the Down that quantizes its rows (test_shape_fallbacks, test_decode_down_fallback_w2_4gib).
    """
    import dataclasses

    kind, tokens, down = _FOLD_FALLBACK_CASES[case]
    if kind == "i1088":
        w, shared, prepared = _fold_fallback_weights(5, 512, 1088, 960)
    elif kind == "h4096_i1088":
        w, shared, prepared = _fold_fallback_weights(2, 4096, 1088, 962)
    else:
        w, shared, prepared = _fold_small_weights()
        size = (1 << 32) - prepared.w2[0].numel() * prepared.w2.element_size() + 1
        prepared = dataclasses.replace(
            prepared, w2=prepared.w2.view(-1)[:1].expand(size)
        )
    hidden, topk = prepared.hidden, 2
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, hidden, prepared.experts, topk, seed=964 + tokens, shared_weight=None
    )
    x = (x.float() * 4.0).to(torch.bfloat16)
    ref = reference_layer(
        x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )

    def call(inputs, out):
        result = flydsl_moe_a8w4_ptpc(
            inputs,
            prepared,
            topk_weight,
            topk_ids,
            output=out,
            swiglu_limit=LAYER_SWIGLU_LIMIT,
        )
        assert result is out
        path = moe_a8w4.last_path
        _assert_folded("stream")
        assert down in path and not any("down_prefill" in n for n in path), path
        return out

    out = call(x, torch.full_like(x, float("nan")))
    assert_close(out, ref)
    atomic = _path_atomic(moe_a8w4.last_path)
    inplace = x.clone()
    call(inplace, inplace)
    if atomic:
        assert_close(inplace, ref)
    else:
        assert torch.equal(_bits(inplace), _bits(out))
    buf = torch.full((tokens * hidden + 4,), float("nan"), dtype=torch.bfloat16)
    unaligned = buf[4:].view(tokens, hidden)
    assert unaligned.data_ptr() % 16 == 8
    call(x, unaligned)
    path = moe_a8w4.last_path
    sums = any(name in _ROUTE_SUMS for name in path)
    assert ("shared_fold_copy" in path) != sums, path
    assert_close(unaligned, ref)
    assert bool(buf[:4].isnan().all())


def test_layer_fold_compile_cache_keys(monkeypatch):
    """The fold's launcher caches: ``_shared_mid_kernel`` is keyed by the device, the routing columns, E, the SwiGLU
    limit and whether it writes loc_ids; the routed launchers' fold variants (routing tensors of topk + 1 columns, the
    slot-pair Down adding the output it finds, the prep never sorting the last column, the prefill Down's loc_ids of
    topk + 1 columns) are cached apart from the plain ones; and a call passes its inputs' device and whether it writes
    loc_ids (the small shape's folded decode M 16, past DECODE_SHARED_MIXED_MAX_M: no; its prefill M 2048: yes).
    Building a launcher does not trace it, so the other device need not exist.

    The basic tests and the benchmark run one GPU, and a cache keyed by shape alone could hand a call with the shared
    expert the launcher of one without (or the reverse) only on a shape that both run.
    """
    mid = (576, 640, 7, 384, LAYER_SWIGLU_LIMIT)
    first = moe_a8w4._shared_mid_kernel(*mid, False, 0)
    assert moe_a8w4._shared_mid_kernel(*mid, False, 0) is first
    for args in (
        (*mid, True, 0),
        (*mid, False, 1),
        (576, 640, 4, 128, LAYER_SWIGLU_LIMIT, False, 0),
        (576, 640, 7, 384, 0.0, False, 0),
    ):
        assert moe_a8w4._shared_mid_kernel(*args) is not first, args
    variants = (
        (
            "_decode_kernels",
            (1152, 5120, 576, 6, True, True, 10.0, 0),
            "shared",
            True,
        ),
        ("_gateup_dd_kernel", (1152, 5120, 6, 10.0, 0), "shared", True),
        ("_row_quant_plan", (5120, 6, 32, 0), "shared", True),
        ("_down_pq_kernel", (5120, 576, 6, 0), "shared", True),
        (
            "_stream_prep_kernel",
            (5120, 7, 128, 16, False, 0),
            "shared",
            True,
        ),
        (
            "_down_kernel",
            (5120, 576, 3, 64, 128, 1, True, 0, 0, moe_a8w4._trace_constants()),
            "loc_topk",
            4,
        ),
    )
    for name, args, key, value in variants:
        build = getattr(moe_a8w4, name)
        assert build(*args, **{key: value}) is build(*args, **{key: value}), name
        assert build(*args, **{key: value}) is not build(*args), name
    built = []
    original = moe_a8w4._shared_mid_kernel

    def spy(*args, **kwargs):
        built.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(moe_a8w4, "_shared_mid_kernel", spy)
    w, shared, prepared = _fold_small_weights()
    for tokens, options in ((16, {}), (2048, {"block_m": 64})):
        x, topk_weight, topk_ids = make_layer_inputs(
            tokens, 512, 8, 2, seed=905, shared_weight=None
        )
        out = flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids, **options)
        assert_close(out, reference_layer(x, w, shared, topk_weight, topk_ids))
    device = x.device.index
    assert [(args[-2], args[-1]) for args in built] == [
        (False, device),
        (True, device),
    ], built


@pytest.mark.parametrize("inter", [320, 576], ids=["i320", "i576"])
def test_shared_mid_lanes_bitexact(inter):
    """shared_fold_mid with several rows per wave (what moe_a8w4 runs past SHARED_MID_WAVE_ROW_MAX_M rows: 8 / 16 lanes
    per row, shared_a8w4.mid_row_lanes; the row maximum by DPP, the quotients through the reciprocal with two
    corrections, the whole wave on the IEEE division where a row's abs-max is outside decode_a8w4._fast_div_ok)
    writes the FP8 bytes, scales and loc_ids column of one row per wave (LANES 64: the per-value IEEE division, what
    runs for fewer rows). At the layers' padded shared I 320 / 576 (KP 384 / 640: 40 / 72 pieces of 8 values, so 16
    lanes leave some lanes without a last piece) with LANES 8, 16 and the default, without and with the SwiGLU limit,
    on 1003 rows (a partial last workgroup): N(0, 10) rows, a zero row, a row whose abs-max is below 2^-60 and (without
    the limit) one at 2^127, each in a wave of its own, and rows whose shared expert is dropped (first id -1, or the
    last id not E: q 0, scale +0, loc -1); the pad columns [I, KP) are 0, a live row's loc base + t.

    The basic tests and the benchmark run the default lanes at a few M and grade end to end, where a flipped FP8 byte
    stays within the tolerance, and their rows never reach the IEEE fallback.
    """
    kp, cols, experts, rows, base = inter + 64, 4, 128, 1003, 77
    tiny, huge, zero, dropped = 9, 42, 20, (3, 50)
    device = torch.cuda.current_device()
    gen = torch.Generator(device="cuda").manual_seed(985 + inter)
    gate_up = torch.randn((rows, 2 * inter), generator=gen, device="cuda") * 10.0
    gate_up[zero] = 0.0
    gate_up[tiny] *= 2.0**-40
    gate_up[huge, :inter] = 2.0**64
    gate_up[huge, inter:] = (gate_up[huge, inter:] / 40.0).clamp(-1.0, 1.0) * 2.0**63
    gate_up[huge, inter] = 2.0**63
    gate_up = gate_up.to(torch.bfloat16)
    ids = torch.randint(
        0, experts, (rows, cols), generator=gen, device="cuda", dtype=torch.int32
    )
    ids[:, -1] = experts
    ids[dropped[0], 0] = -1
    ids[dropped[1], -1] = 5
    topk_weight = torch.rand((rows, cols), generator=gen, device="cuda") + 0.5
    for limit in (0.0, LAYER_SWIGLU_LIMIT):
        results = {}
        for lanes in (64, 8, 16, 0):
            q = torch.full((rows, kp), 0x7F, dtype=torch.uint8)
            scale = torch.full((rows, 1), float("nan"))
            loc = torch.full((rows, cols), -7, dtype=torch.int32)
            kernel = moe_a8w4._shared_mid_kernel(
                inter, kp, cols, experts, limit, True, device, lanes=lanes
            )
            moe_a8w4._launch(
                "shared_fold_mid",
                kernel,
                gate_up,
                ids,
                topk_weight,
                q.view(torch.float8_e4m3fnuz),
                scale,
                loc,
                rows,
                base,
            )
            torch.cuda.synchronize()
            results[lanes] = (q, scale.view(torch.int32), loc)
        for lanes in (8, 16, 0):
            for got, want in zip(results[lanes], results[64]):
                assert torch.equal(got, want), (limit, lanes)
        q, scale, loc = results[64]
        assert bool((q[:, inter:] == 0).all())
        assert bool((q[list(dropped)] == 0).all())
        assert bool((scale[list(dropped)] == 0).all())
        assert bool((q[tiny, :inter] != 0).any())
        assert loc[:, -1].tolist() == [
            -1 if t in dropped else base + t for t in range(rows)
        ]
        assert bool((loc[:, :-1] == -7).all())


# case: (shared I (prepare pads TP8's 288 to 320), H, M (a number, or a moe_a8w4 constant and an offset), whether every
# PAIRS / OUT_AUX of the kernel is forced instead of _shared_down_config's, launches of shared_a8w4's Down kernel). The
# dispatch on both sides of its thresholds (I 320: SHARED_DOWN_MIN_M_SHORT_K rows, I 576: SHARED_DOWN_MIN_M) and past
# the preshuffle GEMM's 65536 rows (H 512, I 64: chunks of 65536 and SHARED_DOWN_MIN_M_SHORT_K rows, both on the
# kernel); the forced tiles PAIRS 4 / 10 / 20 (with the non-temporal stores on 4 and 20) at M 1000, a tail of 40 rows.
_SHARED_DOWN_CASES = {
    "i320_below": (288, LAYER_HIDDEN, ("SHARED_DOWN_MIN_M_SHORT_K", -1), False, 0),
    "i320_at": (288, LAYER_HIDDEN, ("SHARED_DOWN_MIN_M_SHORT_K", 0), False, 1),
    "i576_below": (576, LAYER_HIDDEN, ("SHARED_DOWN_MIN_M", -1), False, 0),
    "i576_at": (576, LAYER_HIDDEN, ("SHARED_DOWN_MIN_M", 0), False, 1),
    "small_chunks": (64, 512, ("SHARED_DOWN_MIN_M_SHORT_K", 65536), False, 2),
    "i320_tiles": (288, LAYER_HIDDEN, 1000, True, 3),
    "i576_tiles": (576, LAYER_HIDDEN, 1000, True, 3),
}


@pytest.mark.parametrize("case", list(_SHARED_DOWN_CASES))
def test_shared_down_bitexact(monkeypatch, case):
    """The folded shared expert's Down (moe_a8w4._shared_down: shared_a8w4's own kernel where _shared_down_config
    picks it, else aiter's preshuffle GEMM) writes the preshuffle GEMM's output bit for bit (_SHARED_DOWN_CASES: both
    sides of the thresholds, every tile, row chunks), reading only the useful K (the columns past the padded shared I
    are 0, as shared_fold_mid writes them) and never writing rows past M (64 NaN rows after the output stay NaN); rows
    of a dropped shared expert (q 0, scale 0) give +0. The kernel's launches are counted; shapes it is not built for
    (K past 576 or not a multiple of 64, N not a multiple of 512) take the preshuffle GEMM.

    The basic tests and the benchmark run the dispatch's tile at a few M and grade end to end, where a few differing
    elements stay within the tolerance; neither writes next to the output nor runs the other tiles.
    """
    from aiter.ops.flydsl import gemm_kernels

    inter, hidden, tokens, tiles, launches = _SHARED_DOWN_CASES[case]
    if isinstance(tokens, tuple):
        tokens = getattr(moe_a8w4, tokens[0]) + tokens[1]
    if hidden == 512:
        prepared = _fold_small_weights()[2]
    else:
        prepared = _fold_fallback_weights(2, hidden, inter, 980)[2]
    wq, ws = prepared.shared_w2, prepared.shared_w2_scale
    k = prepared.shared_w1.shape[0] // 2
    N, kp = wq.shape
    assert N == hidden and k == -(-inter // 64) * 64, (wq.shape, k)
    big = gemm_kernels.PRESHUFFLE_M_MAX
    assert moe_a8w4._shared_down_config(big, N, k) is not None
    for shape in ((big, N, 640), (big, N, k + 32), (big, N - 256, k)):
        assert moe_a8w4._shared_down_config(*shape) is None, shape
    gen = torch.Generator(device="cuda").manual_seed(990 + inter)
    v = torch.randn((tokens, kp), generator=gen, device="cuda") * 60.0
    v[:, k:] = 0.0
    dropped = list(range(1, tokens, 97))
    v[dropped] = 0.0
    mq = v.clamp(-240.0, 240.0).to(torch.float8_e4m3fnuz)
    ms = torch.rand((tokens, 1), generator=gen, device="cuda") * 1e-3 + 1e-4
    ms[dropped] = 0.0
    want = torch.empty((tokens, N), dtype=torch.bfloat16)
    moe_a8w4._shared_gemm(mq, wq, ms, ws, want)
    torch.cuda.synchronize()
    assert _zero_rows(want, dropped)
    built = []
    original = moe_a8w4._shared_down_kernel

    def spy(*args):
        built.append(args)
        return original(*args)

    monkeypatch.setattr(moe_a8w4, "_shared_down_kernel", spy)
    configs = [(4, 2), (10, 0), (20, 2)] if tiles else [None]
    chunks = [min(big, tokens - r) for r in range(0, tokens, big)]
    for config in configs:
        if config is not None:
            monkeypatch.setattr(
                moe_a8w4, "_shared_down_config", lambda m, n, kk, c=config: c
            )
        buf = torch.full((tokens + 64, N), float("nan"), dtype=torch.bfloat16)
        assert moe_a8w4._shared_down(mq, wq, ms, ws, buf[:tokens], k) == chunks
        torch.cuda.synchronize()
        assert torch.equal(_bits(buf[:tokens]), _bits(want)), config
        assert bool(buf[tokens:].isnan().all()), config
    assert len(built) == launches, built
    if tiles:
        assert [args[3:5] for args in built] == configs, built


# -- 2.14 launch records: the bytes and FLOPs of each kernel behind the benchmark's roofline lines -------------------
# Inside moe_a8w4.recording_launches() (off by default) moe_a8w4.last_launches records every device kernel of a call
# (its stream, minimal bytes, useful FLOPs); the benchmark prints them per kernel next to the profiler's device times
# of its unrecorded timed calls (INFO lines, not graded), matched by launch order per stream, so a kernel recorded
# twice or not at all would shift every line after it, and wrong totals would skew the roofline. Neither the basic
# tests nor the benchmark assert the records. Here, on one call of each kind: the records against the profiler's
# kernels of unrecorded calls (count per stream), the closed forms of the call's FLOPs and of the op's minimal bytes,
# an empty record after a call outside the recording, and the same records under CUDA graph capture (recording
# synchronizes nothing). Kernel names are not asserted (later milestones rename and merge kernels). Then the
# benchmark's own matching of the profiler's events onto the records (_kernel_us), on synthetic events: real profiles
# seldom show the cases it has to resolve, and a wrong match prints another kernel's time without any error.

# case: (weights, tokens, options, module constants, routing). Weights: "layer:<shape>" (_layer_weights of a
# LAYER_SHAPES shape, with the shared expert), "routed:<shape>" (its routed experts alone), "small" (_fold_small_weights:
# E 8, H 512, I 64, routed topk 2, with the shared expert), "mimo"
# (_mimo_small_weights(256): E 12, H 4096, topk 8). Routing "converted": int64 ids and strided weights (the copies
# into contiguous int32 / float32 are kernels too); "unaligned": an output 8 bytes past a 16-byte boundary; "odd": 2
# bytes past a 4-byte boundary.
_LAUNCH_CASES = {
    # Per-route decode: Gate/Up reading x and the Down that quantizes its rows.
    "decode_route": ("routed:mtp_layer_tp8", 2, {}, {}, "converted"),
    # Expert-deduplicated decode: the plan Gate/Up and the atomic Down; then the plan kernel, the deduplicated Gate/Up,
    # the row quantization and the atomic Down.
    "decode_dedup_two": ("mimo", 5, {}, {}, None),
    "decode_dedup_four": ("mimo", 24, {}, {}, None),
    # The shared expert inside the two mixed decode kernels.
    "mixed": ("layer:mtp_layer_tp8", 1, {}, {}, None),
    # Folded shared expert on the side stream: pre-quantized per-route decode, wide deduplicated decode (main TP4,
    # which keeps the fold at these M while the mixed decode widens by shape), streaming with the atomic Down at I 320
    # and at I 576, and streaming whose Gate/Up quantizes the intermediate itself (blocks of 32 below H 4096: 15 rows
    # per expert).
    "fold_decode_pq": ("layer:ds_layer_tp4", 6, {}, {}, None),
    "fold_decode_wide": ("layer:ds_layer_tp4", 8, {}, {}, None),
    "fold_stream_atomic": ("layer:mtp_layer_tp8", 64, {}, {}, None),
    "fold_stream_i576": ("layer:mtp_layer_tp4", 128, {}, {}, "converted"),
    "fold_stream_fused_quant": ("small", 60, {}, {}, None),
    # Folded into an unaligned output: the shared Down GEMM writes an aligned buffer, one more kernel copies it.
    "fold_decode_unaligned": ("layer:ds_layer_tp4", 6, {}, {}, "unaligned"),
    # Into an output 2 bytes past a 4-byte boundary: the call writes an aligned buffer, then output_copy.
    "decode_dedup_two_odd": ("mimo", 5, {}, {}, "odd"),
    # Folded, serial (the prefill tier, sorted by the prep): I 576 and I 320 from 4096 routes (32 rows per expert), and
    # past the GEMM's 65536 rows each shared GEMM in two chunks (one launch each, the weight's bytes counted on the
    # first).
    "fold_prefill_prep_i576": ("layer:mtp_layer_tp4", 1366, {}, {}, None),
    "fold_prefill_prep": ("layer:mtp_layer_tp8", 1366, {}, {}, None),
    "fold_prefill_chunks": ("small", 65537, {}, {}, None),
    # Folded, serial: prefill with a tile option at few tokens.
    "tile_prefill": ("small", 40, {"block_m": 64}, {}, None),
}


def _launch_call(monkeypatch, case):
    """``(call, weight, topk_ids)`` of a _LAUNCH_CASES case: ``call()`` runs flydsl_moe_a8w4_ptpc once into one output."""
    import dataclasses

    source, tokens, options, constants, routing = _LAUNCH_CASES[case]
    for name, value in constants.items():
        monkeypatch.setattr(moe_a8w4, name, value)
    kind, _, shape = source.partition(":")
    if kind == "mimo":
        weight, hidden, topk = _mimo_small_weights(256)[1], MIMO_HIDDEN, 8
        x, topk_weight, topk_ids = make_inputs(
            tokens, hidden, weight.experts, topk, seed=930, shared=False
        )
        limit = 0.0
    else:
        if kind == "small":
            weight, hidden, topk = _fold_small_weights()[2], 512, 2
        else:
            experts, topk, inter = LAYER_SHAPES[shape]
            weight, hidden = _layer_weights(experts, inter, 700)[2], LAYER_HIDDEN
            if kind == "routed":
                weight = dataclasses.replace(
                    weight,
                    shared_w1=None,
                    shared_w1_scale=None,
                    shared_w2=None,
                    shared_w2_scale=None,
                )
        x, topk_weight, topk_ids = make_layer_inputs(
            tokens, hidden, weight.experts, topk, seed=931
        )
        if kind == "routed":
            topk_weight = topk_weight[:, :-1].contiguous()
            topk_ids = topk_ids[:, :-1].contiguous()
        limit = LAYER_SWIGLU_LIMIT
    if routing == "converted":
        topk_ids = topk_ids.long()
        topk_weight = topk_weight.t().contiguous().t()
        assert not topk_weight.is_contiguous()
    out = torch.empty_like(x)
    if routing == "unaligned":
        out = torch.empty(x.numel() + 4, dtype=x.dtype)[4:].view_as(x)
        assert out.data_ptr() % 16 == 8
    if routing == "odd":
        if not _copies_odd_outputs():
            pytest.skip("this op writes such an output directly (before M6)")
        out = torch.empty(x.numel() + 1, dtype=x.dtype)[1:].view_as(x)
        assert out.data_ptr() % 4 == 2

    def call():
        return flydsl_moe_a8w4_ptpc(
            x,
            weight,
            topk_weight,
            topk_ids,
            output=out,
            swiglu_limit=limit,
            **options,
        )

    return call, weight, topk_ids


def _profiled_streams(call, expected, calls=3, attempts=5):
    """Device kernels per call of each stream that ``calls`` calls of ``call`` run under the profiler, as a sorted
    list. Profiled again (up to ``attempts`` times) while it differs from ``expected``: the ROCm profiler may record no
    kernel for ~0.1 s while another process starts on the GPU, or drop the last kernels' records.
    """
    import time

    for attempt in range(attempts):
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(calls):
                call()
            torch.cuda.synchronize()
        streams = {}
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                streams[e.device_resource_id] = streams.get(e.device_resource_id, 0) + 1
        found = sorted(n / calls for n in streams.values())
        if found == expected:
            break
        time.sleep(0.1 * (attempt + 1))
    return found


@pytest.mark.parametrize("case", list(_LAUNCH_CASES))
def test_launch_records(monkeypatch, case):
    """``moe_a8w4.last_launches`` of one call of each kind (_LAUNCH_CASES: every tier, both streaming Gate/Up kernels
    and the three streaming Downs, the mixed and the folded shared expert on and off the side stream, the fold's GEMMs
    in row chunks and the copies into unaligned outputs, prefill with a tile option, routing tensors converted
    first) inside ``moe_a8w4.recording_launches()`` holds exactly the device kernels the profiler sees per call on
    each stream when calls run outside it (the main / side labels onto the profiler's streams); its FLOPs sum to
    6 * M * H * (topk * I_pad + I_shared); every kernel moves bytes, the shared expert's weights count once and the
    kernels' minimal bytes cover the op's (the distinct experts' weights, the shared expert's, x read and the output
    written); a call outside the recording leaves the record empty; a CUDA graph capture of the call inside it records
    the same launches (no host synchronization; no GPU data read).

    The benchmark prints these records per kernel without asserting them (an INFO line per kernel, its mismatch line
    only as a log), and the basic tests do not look at them.
    """
    call, weight, topk_ids = _launch_call(monkeypatch, case)
    with moe_a8w4.recording_launches() as recorded:
        call()
    assert recorded is moe_a8w4.last_launches
    launches = list(recorded)
    assert launches
    # Off by default: a call outside the recording clears the record and adds nothing.
    call()
    assert moe_a8w4.last_launches == []
    labels = {}
    for launch in launches:
        assert launch.stream in ("main", "side"), launch
        labels[launch.stream] = labels.get(launch.stream, 0) + 1
    expected = sorted(float(n) for n in labels.values())
    assert _profiled_streams(call, expected) == expected, (case, launches)

    M, H = topk_ids.shape[0], weight.hidden
    topk = topk_ids.shape[1] - weight.has_shared
    shared_inter = weight.shared_w1.shape[0] // 2 if weight.has_shared else 0
    assert sum(launch.flops for launch in launches) == 6 * M * H * (
        topk * weight.inter_padded + shared_inter
    )
    routed = topk_ids[:, :topk]
    experts = int(routed[routed >= 0].unique().numel())
    per_expert = sum(
        t.numel() // t.shape[0] * t.element_size()
        for t in (
            weight.w1,
            weight.w1_delta,
            weight.w1_scale,
            weight.w2,
            weight.w2_delta,
            weight.w2_scale,
        )
    )
    shared_bytes = sum(
        t.numel() * t.element_size()
        for t in (
            weight.shared_w1,
            weight.shared_w1_scale,
            weight.shared_w2,
            weight.shared_w2_scale,
        )
        if t is not None
    )
    assert all(launch.bytes(experts) > 0 for launch in launches), launches
    assert sum(launch.expert_bytes for launch in launches) >= per_expert
    assert sum(launch.shared_bytes for launch in launches) == shared_bytes
    io = 2 * 2 * M * H
    assert sum(launch.bytes(experts) for launch in launches) >= (
        experts * per_expert + shared_bytes + io
    )

    graph = torch.cuda.CUDAGraph()
    with moe_a8w4.recording_launches(), torch.cuda.graph(graph):
        call()
    assert moe_a8w4.last_launches == launches
    del graph


def _mapping_events(kernels, calls, us=lambda c, k: 1.0 + k + c / 100, order=None):
    """Synthetic device events ``(stream, start, end, us, name)``, as _keep_device_events keeps them, of ``calls``
    calls that each run ``kernels`` (``(stream, name)`` pairs in launch order) one after another: ``us(call, kernel)``
    a kernel's time, ``order(call)`` the order in which the call runs its kernels (default: as launched).
    """
    events, t = [], 0.0
    for c in range(calls):
        for k in order(c) if order else range(len(kernels)):
            stream, name = kernels[k]
            events.append((stream, t, t + us(c, k), us(c, k), name))
            t += 10.0
    return events


_MAPPING_ABC = [(1, "a"), (1, "b"), (1, "c")]
# case: (stream label of each recorded kernel, calls, events, the expected (times, calls used, mismatch)). The times
# are the means over the calls used without the first (1 + kernel + call / 100 unless the case sets them).
_MAPPING_CASES = {
    # Three streams told apart by their kernels per call.
    "streams": (
        ["main", "side", "main", "side", "aux", "main"],
        4,
        _mapping_events(
            [(3, "a"), (7, "s1"), (3, "b"), (7, "s2"), (5, "x"), (3, "c")], 4
        ),
        ([1.02, 2.02, 3.02, 4.02, 5.02, 6.02], 4, None),
    ),
    # Two streams of one equal kernel per call, the side stream's listed first: main is the stream that ends last.
    "main_ends_last": (
        ["main", "side"],
        4,
        _mapping_events([(7, "k"), (3, "k")], 4, us=lambda c, k: 1.0 + k),
        ([2.0, 1.0], 4, None),
    ),
    # The profiler lost the records of the last calls, the last kept one cut short: the 7 whole calls.
    "dropped_tail": (
        ["main"] * 3,
        10,
        _mapping_events(_MAPPING_ABC, 10)[:23],
        ([1.035, 2.035, 3.035], 7, None),
    ),
    # One call ran its kernels in another order: left out.
    "misplaced_call": (
        ["main"] * 3,
        6,
        _mapping_events(
            _MAPPING_ABC, 6, order=lambda c: (0, 2, 1) if c == 2 else (0, 1, 2)
        ),
        ([1.0325, 2.0325, 3.0325], 5, None),
    ),
    # The profile lists the events out of order: matched by their start times.
    "start_order": (
        ["main"] * 3,
        6,
        sorted(_mapping_events(_MAPPING_ABC, 6), key=lambda e: (e[4], e[1])),
        ([1.03, 2.03, 3.03], 6, None),
    ),
    # More than 30 calls: the call whose total is an IQR outlier is left out of the means.
    "outliers": (
        ["main"] * 2,
        40,
        _mapping_events(
            _MAPPING_ABC[:2], 40, us=lambda c, k: (500.0 if c == 10 else 5.0) + 2 * k
        ),
        ([5.0, 7.0], 40, None),
    ),
    # A kernel per call that the record lacks: a mismatch (recorded, profiled per call, per label | per stream).
    "extra_kernel": (
        ["main"] * 2,
        5,
        _mapping_events(_MAPPING_ABC, 5),
        (None, 0, (2, 3.0, "main:2|3")),
    ),
    # Most calls run their kernels in another order: fewer than half usable, a mismatch.
    "scattered_calls": (
        ["main"] * 3,
        6,
        _mapping_events(
            _MAPPING_ABC, 6, order=lambda c: [(0, 2, 1), (1, 0, 2), (0, 1, 2)][c // 2]
        ),
        (None, 0, (3, 3.0, "main:3|3")),
    ),
    "no_events": (["main"] * 3, 5, [], (None, 0, None)),
}


@pytest.mark.parametrize("case", list(_MAPPING_CASES))
def test_kernel_us_mapping(case):
    """The benchmark's matching of the profiled device events onto a call's launch records (``_kernel_us``, region 3:
    the time of each ``INFO kernel`` line), on synthetic events (_MAPPING_CASES): streams assigned to the main / side
    / aux labels by their kernels per call, and between equal streams main as the one that ends last; the calls whose
    records the profiler lost at the end or put out of place left out, as long as half of them remain; events out of
    the profile's order matched by start time; IQR outlier calls left out of the means; a mismatch (no times) when the
    record lacks a kernel or most calls are out of place; nothing without events.

    The benchmark only prints the result (an ``INFO kernel_map`` line when calls are dropped or nothing fits), and its
    real profiles seldom lose or misplace records, never in a chosen way; the basic tests do not profile.
    """
    labels, calls, events, (times, used, mismatch) = _MAPPING_CASES[case]
    launches = [
        moe_a8w4.KernelLaunch(f"k{i}", label, 0, 0, 0, 0)
        for i, label in enumerate(labels)
    ]
    got = _kernel_us(launches, events, calls)
    assert got[1:] == (used, mismatch), got
    if times is None:
        assert got[0] is None, got
    else:
        assert got[0] == pytest.approx(times), got


# =================================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# 180 cases, named <shape>_m<M> with M = 1, 2, 4, ..., 32768 and 48 (the layer cases also M 6, 1280 for mtp_layer_tp4,
# see BENCH_LAYER_EXTRA_CASES, and the 22 dispatch boundaries of BENCH_LAYER_BOUNDARY_CASES); SiLU, BF16 in and out.
# First the 85 cases of moe-a8w4 (unchanged: names, seeds, timing, output), 5 shapes:
#   tp2 / tp4: Qwen3.8-Flash-Next, H 2560, E 513 (512 routed + a fused shared expert that every token's last slot
#     selects), topk 11, I 320 / 160 (padded to 192 by prepare; aiter gets the same BF16 weights padded to 192);
#     against aiter's FP8 per-token fused_moe with the shipped tuned table.
#   mimo_tp8 / mimo_tp4: MiMo-V2.6-Flash, H 4096, E 256, topk 8, uniform routing, I 256 / 512 (no aiter table).
#   ds_tp4: DeepSeek-V4.1-Flash TP4, H 5120, E 384, topk 6, uniform routing, I 576, swiglu_limit 10 for the op and
#     the reference (N(0, 1) inputs rarely reach it: the case measures its cost); no aiter table.
# Weights and inputs come from the region-0 generators with pyhip's seeds (weights MF_SEED * 1000 + shape index,
# inputs MF_SEED * 1000 + 100 + 4 * case index), so they are pyhip's benchmark inputs (its input group 0).
# Every candidate is timed by bench_time_us (run_perftest with BENCH_PERFTEST_KW, see there); the timed A8W4
# output is graded against the fp32 reference, aiter's must be finite. Per case: MF_METRIC (A8W4) and, for Qwen,
# MF_REF (aiter) with flops = 6 * M * topk * I_pad * H and each implementation's minimal bytes (the selected
# experts' weights and scales, the input read and the output write), and an INFO line with the launched kernels.
# Then 95 whole-layer cases of DeepSeek-V4.1-Flash with its shared expert (bench_moe_a8w4_layer): LAYER_SHAPES (main
# layer E 384 + shared, topk 6 + 1; MTP layer E 128 + shared, topk 3 + 1; TP4 I 576, TP8 I 288 padded to 320), H 5120,
# swiglu_limit 10, synthetic weights (routed MXFP4, shared FP8 per row), uniform routing with routed weights normalized
# and times 1.5, the shared expert E in the last column with weight 1 (ATOM's routing); seeds MF_SEED * 1000 + 500 +
# shape index (routed weights), + 550 + shape index (shared), inputs + 600 + 4 * layer case index. Four implementations,
# each timed by bench_graph_us (CUDA graph replays, wall time, so that side streams overlap as in serving):
#   a8w4 (MF_METRIC): flydsl_moe_a8w4_ptpc on the weight with the shared expert, graded against reference_layer;
#   aiter_fp8_ptpc (MF_REF, printed first, the name of the Qwen reference): aiter's fused_moe per_Token FP8 on E + 1
#     experts and topk + 1 columns as ATOM's original arm calls it, the routed weights the exact MXFP4 values
#     re-quantized per channel to FP8 E4M3FNUZ, the shared expert the last one, I padded to 320 for TP8 (aiter has no
#     kernel for I 288 with the limit); finite;
#   a8w4_split (MF_REF, in the summary table): ATOM's split arm on this worktree: flydsl_moe_a8w4_ptpc on the routed
#     experts and aiter_shared_split, on a side stream up to BENCH_SPLIT_DUAL_STREAM_MAX_M tokens, without the final
#     add (ATOM folds it into its all-reduce); finite;
#   routed (INFO): flydsl_moe_a8w4_ptpc on the routed experts alone, for the shared expert's increment.
# flops = 6 * M * H * (topk * I_pad + I) (the shared expert at its own I); bytes per implementation: the weights and
# scales of the experts the call reads in its own format, the input read and the output write. INFO: routed time, the
# shared increment (a8w4 - routed) with its TB/s and TFLOPS, the routed TB/s and TFLOPS, a8w4's kernel time summed by
# bench_time_us, a8w4's kernels, the kernels aiter resolved and a8w4's error. At the end one markdown table of all cases.
# Per-kernel roofline of the a8w4 call (INFO only: not aggregated, no MF_METRIC / MF_REF of its own; _kernel_info): one
# "INFO kernel" line per device kernel that moe_a8w4.last_launches records (stream, minimal bytes at the case's distinct
# routed experts, useful FLOPs) with its device time from the run_perftest profile that bench_time_us already runs (the
# timed run of the first 85 cases, the kernel_sum run of the layer cases; no extra timed run), matched by launch order
# per stream ("INFO kernel_map ... mismatch" when they do not fit, "... dropped_calls" when the profiler lost or
# misplaced some calls' kernels); the record comes from one more untimed a8w4 call inside moe_a8w4.recording_launches()
# after the timed runs (_recorded_launches: the timed calls record nothing); one "INFO gaps" line (the MF_METRIC time
# against the kernels' sum on the main stream); after the summary table a roofline table, one row per case.
# main: -s / -m select shapes and token counts; --quick runs the layer shapes at BENCH_QUICK_TOKENS only.
# =================================================================================================

# shape: (H, E, topk, I, padded I, fused shared expert, aiter reference)
BENCH_SHAPES = {
    "tp2": (2560, 513, 11, 320, 320, True, True),
    "tp4": (2560, 513, 11, 160, 192, True, True),
    "mimo_tp8": (4096, 256, 8, 256, 256, False, False),
    "mimo_tp4": (4096, 256, 8, 512, 512, False, False),
    "ds_tp4": (5120, 384, 6, 576, 576, False, False),
}
# swiglu_limit passed to the op and the reference (DeepSeek-V4.1's SwiGLU limit); other shapes pass none.
BENCH_SWIGLU_LIMIT = {"ds_tp4": 10.0}
BENCH_TOKENS = tuple(1 << k for k in range(16))  # 1, 2, 4, ..., 32768
# M = 48: pyhip's small-M streaming tier at E513 (M 33-62), which the powers of two skip.
BENCH_EXTRA_TOKENS = (48,)
# The order gives each case's input seed.
BENCH_CASES = (
    [f"{tp}_m{m}" for tp in ("tp2", "tp4") for m in BENCH_TOKENS]
    + [f"{tp}_m{m}" for tp in ("tp2", "tp4") for m in BENCH_EXTRA_TOKENS]
    + [
        f"{tp}_m{m}"
        for tp in ("mimo_tp8", "mimo_tp4")
        for m in BENCH_TOKENS + BENCH_EXTRA_TOKENS
    ]
    + [f"ds_tp4_m{m}" for m in BENCH_TOKENS + BENCH_EXTRA_TOKENS]
)
BENCH_TABLE = "qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv"
# run_perftest arguments of every timed candidate (num_warmup is set by bench_time_us): 101 calls under the
# profiler, the device time of all kernels of a call summed (gaps between kernels excluded), the first call
# and IQR outliers dropped; num_rotate_args 0 rotates deep copies of all tensor arguments, the weights included,
# through 64 * 128 L2 sizes (32 GiB on MI308X), so each timed call starts with cold L2 and MALL; no CUDA graph.
BENCH_PERFTEST_KW = {
    "num_iters": 101,
    "num_rotate_args": 0,
    "testGraph": False,
    "use_cuda_event": False,
    "profile_warm_iters": 1,
}
# Warm-up before every timed loop, the same for every candidate. Host work between timed loops (inputs, the
# reference, grading, deep copies) leaves the GPU idle; its memory clock drops and needs ~5 ms of load to come
# back (measured on MI308X for the dense A8W4 GEMM). run_perftest runs its warm-up calls right before the timed
# loop (after the deep copies), so each run gets calls worth BENCH_WARMUP_MS as timed on the cold GPU.
BENCH_WARMUP_MS = 18.0
# The ROCm profiler records no GPU event for ~0.1 s while another process initializes HIP on the same GPU: an
# empty record is retried after these pauses (seconds).
BENCH_RETRY_S = (0.05, 0.1, 0.2, 0.4)
# Layer cases of the folded tiers that the token counts above skip, appended so that the earlier cases keep their
# seeds: M 6 (the pre-quantized per-route decode of M 5-7) at every layer shape, and the MTP TP4 layer at M 1280
# (streaming in blocks of 64 into the prefill Down: M 1195-1365 there; no other layer shape reaches that tier).
BENCH_LAYER_EXTRA_CASES = [f"{shape}_m6" for shape in LAYER_SHAPES] + [
    "mtp_layer_tp4_m1280"
]
# Layer cases at the boundaries of the dispatch that the cases above skip (the last M of a tier or the first of the
# next; moe-a8w4-roofline SPEC 7.1), appended after them so that every earlier case keeps its seed.
BENCH_LAYER_BOUNDARY_CASES = [
    # The main TP4 layer's last mixed decode M (its M 4 folds: M * topk * I_pad 13824 > DECODE_SHARED_MIXED_MAX_WORK).
    "ds_layer_tp4_m3",
    # The first folded decode M of the other layers (the mixed decode ends at DECODE_SHARED_MIXED_MAX_M 4): the
    # pre-quantized per-route Gate/Up and the slot-pair Down, the shared chain on the side stream.
    "ds_layer_tp8_m5",
    "mtp_layer_tp4_m5",
    "mtp_layer_tp8_m5",
    # The last decode M at H 5120 (DECODE_WIDE_MAX_M 10: the deduplicated Gate/Up) and the first streaming M (blocks
    # of 16), each layer shape.
    *[f"{shape}_m{m}" for shape in LAYER_SHAPES for m in (10, 11)],
    # Main layers: the last M of streaming blocks of 16 (12 rows per expert), and the last streaming M (24 rows per
    # expert, STREAM_MAX_ROWS_H5120; blocks of 32 past SHARED_FOLD_DUAL_STREAM_MAX_M, the shared chain serial).
    "ds_layer_tp4_m768",
    "ds_layer_tp8_m768",
    "ds_layer_tp4_m1536",
    "ds_layer_tp8_m1536",
    # MTP layers: blocks of 32 with the shared chain serial (M 1025-1194), and the last streaming M (blocks of 64:
    # TP4 into the prefill Down, TP8 the atomic Down; M 1195-1365).
    "mtp_layer_tp4_m1152",
    "mtp_layer_tp8_m1152",
    "mtp_layer_tp4_m1365",
    "mtp_layer_tp8_m1365",
    # Past SHARED_FOLD_MAX_M 65536 (the fold's GEMMs' row limit): the prefill tier with ATOM's split of the shared
    # expert, serial (65536 + 32768 rows: one whole block of 65536 rows and a half one).
    "ds_layer_tp4_m98304",
    "mtp_layer_tp8_m98304",
]
# The layer cases, after the 85 above; the order gives each case's input seed.
BENCH_LAYER_CASES = (
    [
        f"{shape}_m{m}"
        for shape in LAYER_SHAPES
        for m in BENCH_TOKENS + BENCH_EXTRA_TOKENS
    ]
    + BENCH_LAYER_EXTRA_CASES
    + BENCH_LAYER_BOUNDARY_CASES
)
# --quick (bench_quick): the layer shapes at these token counts only.
BENCH_QUICK_TOKENS = (1, 4, 16, 64, 512, 4096)
# bench_graph_us: at most this many argument sets (run_perftest's num_iters) and at least this many timed replays.
BENCH_GRAPH_MAX_SETS = 101
BENCH_GRAPH_MIN_REPLAYS = 5
# a8w4_split runs the shared expert on a side stream up to this many tokens (ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD).
BENCH_SPLIT_DUAL_STREAM_MAX_M = 1024


def bench_time_us(fn, args, events=None):
    """Device time of one ``fn(*args)`` call in microseconds, the protocol of every timed candidate.

    The warm-up count is the fewest calls taking longer than BENCH_WARMUP_MS (at least 2), each call timed with
    CUDA events as the fastest of 3 bursts of 8 back-to-back calls, each burst behind one leading call (it keeps
    the launch latency after the synchronization out; the first also compiles). Then
    ``run_perftest(fn, *args, num_warmup=..., **BENCH_PERFTEST_KW)``, repeated after BENCH_RETRY_S pauses while
    the profiler returns no time. Uses only the standard library, torch, ``run_perftest`` and the BENCH_*
    constants, so another process can time another implementation with the same protocol. With ``events`` (a list)
    the device events of the profile behind the returned time are left in it (``_keep_device_events``: read once the
    profiler has stopped; the protocol and the time are the same).
    """
    import math
    import time

    import torch

    from aiter.test_common import run_perftest

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(3):
        fn(*args)
        start.record()
        for _ in range(8):
            fn(*args)
        end.record()
        end.synchronize()
        ms.append(start.elapsed_time(end) / 8)
    num_warmup = max(2, int(BENCH_WARMUP_MS / min(ms)) + 1)
    for pause in BENCH_RETRY_S + (None,):
        torch.cuda.synchronize()
        with (
            contextlib.nullcontext() if events is None else _keep_device_events(events)
        ):
            _, us = run_perftest(fn, *args, num_warmup=num_warmup, **BENCH_PERFTEST_KW)
        if math.isfinite(us) and us > 0:
            return float(us)
        if pause is None:
            raise RuntimeError(
                f"the profiler recorded no kernel time in {len(BENCH_RETRY_S) + 1} runs"
            )
        time.sleep(pause)


def bench_graph_us(fn, args):
    """Wall time of one ``fn(*args)`` call in microseconds from CUDA graph replays, the protocol of the layer cases.

    n independent sets of the arguments (``args`` and n - 1 deep copies: weights, inputs and outputs), n =
    min(ceil(L2 size * 64 * 128 / bytes of a set), BENCH_GRAPH_MAX_SETS) as run_perftest rotates its copies (32 GiB on
    MI308X; at most 90% of the free memory), so that each call starts with cold L2 and MALL. One eager call compiles;
    the n calls, one per set, are captured into one graph (side streams a call forks and joins are captured with it,
    so their overlap counts); right before timing, eager calls worth BENCH_WARMUP_MS as bench_time_us counts them
    (the fastest of 3 bursts of 8 calls) and one replay warm the GPU up; then R = max(BENCH_GRAPH_MIN_REPLAYS,
    ceil(BENCH_GRAPH_MAX_SETS / n)) replays, each timed with CUDA events. Returns the median of replay time / n and
    the results of the eager call on ``args`` (outputs written by the replays are those of the first set).
    """
    import copy
    import statistics

    tensors = {
        a.data_ptr(): a.numel() * a.element_size()
        for a in args
        if isinstance(a, torch.Tensor)
    }
    set_bytes = max(sum(tensors.values()), 1)
    device = torch.cuda.current_device()
    l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 4 << 20)
    budget = min(l2 * 64 * 128, torch.cuda.mem_get_info(device)[0] * 0.9)
    n = max(1, min(math.ceil(budget / set_bytes), BENCH_GRAPH_MAX_SETS))
    sets = [args] + [copy.deepcopy(args) for _ in range(n - 1)]
    result = fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for one in sets:
            fn(*one)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    bursts = []
    for _ in range(3):
        fn(*args)
        start.record()
        for k in range(8):
            fn(*sets[k % n])
        end.record()
        end.synchronize()
        bursts.append(start.elapsed_time(end) / 8)
    for k in range(max(2, int(BENCH_WARMUP_MS / min(bursts)) + 1)):
        fn(*sets[k % n])
    graph.replay()
    times = []
    for _ in range(max(BENCH_GRAPH_MIN_REPLAYS, math.ceil(BENCH_GRAPH_MAX_SETS / n))):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / n)
    del graph, sets
    torch.cuda.empty_cache()
    return statistics.median(times), result


# Per-kernel roofline (_kernel_info): the 100% lines of moe-a8w4-roofline (its SPEC Q1), 4 TB/s and 450 TFLOPS; a
# kernel of fewer than 450 / 4 = 112.5 FLOPs per byte is memory bound.
BENCH_ROOF_TBPS = 4.0
BENCH_ROOF_TFLOPS = 450.0
# Share of a profile's calls that _kernel_us needs intact: the ROCm profiler may drop the records of the last kernels
# when it stops (seen on MI308X: a few calls' kernels), and a call whose kernel names differ from the other calls' (a
# record out of place) is dropped too.
BENCH_PROFILE_MIN_CALLS = 0.5
# The roofline table: one row per case run (_kernel_info), printed at the end of main.
_ROOFLINE_ROWS = []


@contextlib.contextmanager
def _keep_device_events(sink):
    """While active, each profile that ``run_perftest`` evaluates also leaves its device events in ``sink`` (replacing
    its contents), read from the finished profile where run_perftest's ``get_trace_perf`` reads it: (stream, start us,
    end us, device us) per kernel, the stream being the profiler's ``device_resource_id``.
    """
    from aiter import test_common

    evaluate = test_common.get_trace_perf

    def keep(prof, *args, **kwargs):
        sink[:] = [
            (
                e.device_resource_id,
                e.time_range.start,
                e.time_range.end,
                e.self_device_time_total,
                e.name,
            )
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        return evaluate(prof, *args, **kwargs)

    test_common.get_trace_perf = keep
    try:
        yield
    finally:
        test_common.get_trace_perf = evaluate


def _recorded_launches(fn, args):
    """``moe_a8w4.last_launches`` of one more ``fn(*args)`` call made inside ``moe_a8w4.recording_launches()``, after
    the timed ones: those record nothing (recording is off by default; on, it costs host time per kernel). A call's
    kernels depend on its shapes, options and the module's constants, not on the tensors' values, so this call's
    record describes the timed calls' kernels."""
    with moe_a8w4.recording_launches() as launches:
        fn(*args)
    return list(launches)


def _kernel_us(launches, events, calls):
    """Device time (us) of each kernel of ``launches`` (``moe_a8w4.last_launches`` of a call) from the device
    ``events`` of ``calls`` such calls under the profiler (``_keep_device_events``). Returns ``(times, calls used,
    None)``, ``(None, 0, (registered, profiled, detail))`` when the profile does not fit the record (kernels per call,
    per stream in ``detail``), or ``(None, 0, None)`` without events.

    Each stream's events, in the profile's order (else by start time), are its kernels of the calls in turn. The
    recorded kernels of each stream label (main / side / aux), in launch order, take the events of one stream: the
    assignment under which every stream holds as many kernels per call as its label, at most ``calls`` calls, with the
    same sequence of kernel names in at least BENCH_PROFILE_MIN_CALLS of the calls (the calls used: the others lost at
    the end of the profile or holding a record out of place); of several, the one whose "main" stream ends last (a call
    joins its side streams before its last kernel). A kernel's time is its mean over the calls used that run_perftest
    keeps: the first (warm) call dropped, then the calls whose total is an IQR outlier (k = 1.5, more than 30 calls),
    so that the kernels of one stream sum to run_perftest's time.
    """
    if not events:
        return None, 0, None
    labels = {}
    for i, launch in enumerate(launches):
        labels.setdefault(launch.stream, []).append(i)
    order = sorted(labels, key=lambda name: labels[name][0])

    def calls_used(found, k):
        # Calls whose k kernels on the stream have the most common sequence of names.
        if len(found) > calls * k:
            return set()
        names = [
            tuple(e[4] for e in found[c * k : (c + 1) * k])
            for c in range(len(found) // k)
        ]
        if not names:
            return set()
        common = max(set(names), key=names.count)
        return {c for c, seq in enumerate(names) if seq == common}

    fit = None
    for ordered in (events, sorted(events, key=lambda e: (e[1], e[2]))):
        streams = {}
        for e in ordered:
            streams.setdefault(e[0], []).append(e)
        if len(streams) != len(labels) or "main" not in labels:
            break
        fits = []
        for chosen in itertools.permutations(streams):
            assign = dict(zip(order, chosen))
            used = set.intersection(
                *(
                    calls_used(streams[assign[label]], len(labels[label]))
                    for label in order
                )
            )
            if len(used) > 1 and len(used) >= calls * BENCH_PROFILE_MIN_CALLS:
                fits.append((assign, used))
        if fits:
            last = max(streams, key=lambda s: max(e[2] for e in streams[s]))
            fit = next((f for f in fits if f[0]["main"] == last), fits[0])
            break
    if fit is None:
        detail = (
            ",".join(f"{label}:{len(index)}" for label, index in labels.items())
            + "|"
            + ",".join(f"{len(found) / calls:g}" for found in streams.values())
        )
        return None, 0, (len(launches), len(events) / calls, detail)
    assign, used = fit
    per_call = {}
    for c in used:
        durations = [0.0] * len(launches)
        for label, index in labels.items():
            found = streams[assign[label]]
            for j, i in enumerate(index):
                durations[i] = found[c * len(index) + j][3]
        per_call[c] = durations
    kept = sorted(c for c in used if c > 0) or sorted(used)
    if len(kept) > 30:
        totals = pd.Series([sum(per_call[c]) for c in kept])
        q1, q3 = totals.quantile(0.25), totals.quantile(0.75)
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        kept = [c for c, total in zip(kept, totals) if low <= total <= high]
    times = [
        sum(per_call[c][i] for c in kept) / len(kept) for i in range(len(launches))
    ]
    return times, len(used), None


def _distinct_experts(ids):
    """Distinct routed experts of routing ids (ids >= 0; a host synchronization)."""
    return int(ids[ids >= 0].unique().numel())


def _kernel_info(case, launches, events, experts, wall_us, op_bytes, op_flops):
    """Print the per-kernel roofline of a case's a8w4 call (``INFO kernel`` lines, ``INFO gaps``; ``INFO kernel_map
    ... mismatch`` when the profile and ``launches`` disagree) and add its row to the roofline table (_ROOFLINE_ROWS).

    ``launches``: ``moe_a8w4.last_launches`` of the profiled calls; ``events``: their device events
    (``bench_time_us(..., events)``, BENCH_PERFTEST_KW's num_iters calls); ``experts``: the distinct routed experts of
    the case's routing (each kernel's bytes ``KernelLaunch.bytes``); ``wall_us``: the case's MF_METRIC time;
    ``op_bytes`` / ``op_flops``: its MF_METRIC bytes and flops, the op's minimal traffic (weights, input, output).
    The table: kernels per stream (side "s", aux "a"), the times, the kernels' bytes summed against the op's, "roof us"
    the kernels' roofline times max(bytes / 4 TB/s, FLOPs / 450 TFLOPS) summed over every stream (as if serial) and
    "op roof us" the op's, each against the wall time, and the longest kernel with its share of its bound line.
    """
    calls = BENCH_PERFTEST_KW["num_iters"]
    times, used_calls, mismatch = _kernel_us(launches, events, calls)
    if mismatch is not None:
        registered, profiled, detail = mismatch
        print(
            f"INFO kernel_map case={case} mismatch registered={registered} profiled={profiled:g} "
            f"streams={detail}",
            flush=True,
        )
    elif times is not None and used_calls < calls:
        # Calls whose records the profiler dropped or put out of place: the times are means over the other calls.
        print(
            f"INFO kernel_map case={case} dropped_calls={calls - used_calls}",
            flush=True,
        )
    ridge = BENCH_ROOF_TFLOPS / BENCH_ROOF_TBPS
    total_bytes, roof_us, top = 0, 0.0, (None, -1.0, math.nan)
    for i, launch in enumerate(launches):
        nbytes = launch.bytes(experts)
        total_bytes += nbytes
        roof = max(nbytes / BENCH_ROOF_TBPS, launch.flops / BENCH_ROOF_TFLOPS) / 1e6
        roof_us += roof
        ai = launch.flops / nbytes if nbytes else math.inf
        bound = "mem" if ai < ridge else "compute"
        us = times[i] if times is not None else None
        if us is not None and us > 0:
            tbps, tflops = nbytes / us / 1e6, launch.flops / us / 1e6
            rates = (
                f"tbps={tbps:.3f}({tbps / BENCH_ROOF_TBPS:.1%}) "
                f"tflops={tflops:.2f}({tflops / BENCH_ROOF_TFLOPS:.1%})"
            )
            share = (
                tbps / BENCH_ROOF_TBPS if bound == "mem" else tflops / BENCH_ROOF_TFLOPS
            )
        else:
            rates, share = "tbps=unavailable tflops=unavailable", math.nan
        rank = us if us is not None else roof
        if rank > top[1]:
            top = (launch.name, rank, share)
        print(
            f"INFO kernel case={case} idx={i} name={launch.name} stream={launch.stream} "
            f"us={'unavailable' if us is None else f'{us:.2f}'} bytes={nbytes} flops={launch.flops} {rates} "
            f"ai={ai:.2f} bound={bound}",
            flush=True,
        )
    if times is None:
        kernel_sum = main_sum = gap = math.nan
        sums = "kernel_sum_us=unavailable main_sum_us=unavailable gap_us=unavailable"
    else:
        kernel_sum = sum(times)
        main_sum = sum(
            us for us, launch in zip(times, launches) if launch.stream == "main"
        )
        # Rounded to the printed 0.01 us (+ 0.0: no "-0.00" where the times agree).
        gap = round(wall_us - main_sum, 2) + 0.0
        sums = f"kernel_sum_us={kernel_sum:.2f} main_sum_us={main_sum:.2f} gap_us={gap:.2f}"
    print(f"INFO gaps case={case} wall_us={wall_us:.2f} {sums}", flush=True)
    counts = {}
    for launch in launches:
        counts[launch.stream] = counts.get(launch.stream, 0) + 1
    op_roof = max(op_bytes / BENCH_ROOF_TBPS, op_flops / BENCH_ROOF_TFLOPS) / 1e6
    _ROOFLINE_ROWS.append(
        {
            "case": case,
            "kernels": "+".join(
                f"{n}{'' if stream == 'main' else stream[0]}"
                for stream, n in counts.items()
            ),
            "wall us": wall_us,
            "kernel sum us": kernel_sum,
            "main sum us": main_sum,
            "gap us": gap,
            "MB": total_bytes / 1e6,
            "op MB": op_bytes / 1e6,
            "roof us": roof_us,
            "op roof us": op_roof,
            "roof / wall": roof_us / wall_us,
            "op roof / wall": op_roof / wall_us,
            "longest kernel": top[0],
            "its us": top[1] if times is not None else math.nan,
            "its % of line": top[2] * 100,
        }
    )


@functools.lru_cache(maxsize=1)
def _bench_weights(shape):
    """(MXFP4 weights, A8W4 weights, aiter's shuffled FP8 per-channel weights or None) of a shape."""
    from aiter.ops.shuffle import shuffle_weight

    hidden, experts, _, inter, inter_pad, _, compare = BENCH_SHAPES[shape]
    w = make_weights(
        experts,
        hidden,
        inter,
        seed=SEED * 1000 + list(BENCH_SHAPES).index(shape),
        fp8_inter=inter_pad if compare else None,
    )
    prepared = prepare_moe_a8w4_ptpc_weight(
        w["w1"], w["w1_scale"], w["w2"], w["w2_scale"]
    )
    fp8_w = None
    if compare:
        fp8_w = {
            "w1": shuffle_weight(w.pop("w1_fp8"), layout=(16, 16)),
            "w1_scale": w.pop("w1_fp8_scale"),
            "w2": shuffle_weight(w.pop("w2_fp8"), layout=(16, 16)),
            "w2_scale": w.pop("w2_fp8_scale"),
        }
    return w, prepared, fp8_w


def _aiter_row(fm, tokens, hidden, inter, experts, topk):
    """The tuned-table row aiter's fused_moe looks up for this call (per_Token FP8, SiLU, BF16)."""
    from aiter import ActivationType, QuantType, dtypes

    key = (
        fm.get_gfx_runtime(),
        fm.get_cu_num(),
        fm.get_padded_M(tokens),
        hidden,
        inter,
        experts,
        topk,
        str(ActivationType.Silu),
        str(dtypes.bf16),
        str(dtypes.fp8),
        str(dtypes.fp8),
        str(QuantType.per_Token),
        1,
        0,
    )
    fm.get_2stage_cfgs(
        fm.get_padded_M(tokens),
        hidden,
        inter,
        experts,
        topk,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp8,
        QuantType.per_Token,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
    )
    row = fm.cfg_2stages[0].get(key)
    if row is None or not 0 < float(row["us"]) < math.inf:
        raise RuntimeError(f"{BENCH_TABLE} has no valid row for {key}")
    return row


def _row_kernels(row):
    name = str(row["kernelName1"])
    if int(row["run_1stage"]):
        return f"asm_1stage:{name}"
    return name if name.startswith("impl__") else f"{name}+{row['kernelName2']}"


def _uses_row(meta, row):
    """Whether the metadata aiter's dispatcher finally used is this table row (the 1-stage asm, ``impl__`` and CK
    2-stage rows of the table)."""
    name = str(row["kernelName1"])
    if int(meta.block_m) != int(row["block_m"]) or bool(meta.run_1stage) != bool(
        int(row["run_1stage"])
    ):
        return False
    if meta.run_1stage:
        return meta.stage1.keywords.get("kernelName") == name
    if meta.full_impl is not None and name.startswith("impl__"):
        configs = [
            c.cell_contents
            for c in meta.full_impl.__closure__ or ()
            if isinstance(c.cell_contents, str)
        ]
        return configs == [name.split("__", 2)[2]]
    if meta.full_impl is None and name.startswith("moe_ck2stages_"):
        return meta.stage1.keywords.get(
            "kernelName"
        ) == name and meta.stage2.keywords.get("kernelName") == str(row["kernelName2"])
    return False


def _traffic(ids, per_expert, hidden):
    """Minimal bytes of a call: every selected expert's weights and scales, the BF16 input read and output write."""
    return ids.unique().numel() * per_expert + ids.shape[0] * hidden * 2 * 2


def _bytes_per_expert(weights, experts):
    values = weights.values() if isinstance(weights, dict) else vars(weights).values()
    return (
        sum(t.numel() * t.element_size() for t in values if isinstance(t, torch.Tensor))
        // experts
    )


@benchmark()
def bench_moe_a8w4(shape, m):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    case = f"{shape}_m{m}"
    hidden, experts, topk, inter, inter_pad, shared, compare = BENCH_SHAPES[shape]
    w, prepared, fp8_w = _bench_weights(shape)
    x, topk_weight, topk_ids = make_inputs(
        m,
        hidden,
        experts,
        topk,
        seed=SEED * 1000 + 100 + BENCH_CASES.index(case) * 4,
        shared=shared,
    )
    limit_kw = (
        {"swiglu_limit": BENCH_SWIGLU_LIMIT[shape]}
        if shape in BENCH_SWIGLU_LIMIT
        else {}
    )
    ref = reference(
        x,
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        topk_weight,
        topk_ids,
        **limit_kw,
    )
    flops = 6 * m * topk * inter_pad * hidden
    out = torch.full_like(x, float("nan"))

    # The weight tensors are arguments of their own, so that run_perftest's rotation copies them too.
    def a8w4(
        x, w1, w1_delta, w1_scale, w2, w2_delta, w2_scale, topk_weight, topk_ids, out
    ):
        weight = MoeA8W4PtpcWeight(
            w1,
            w1_delta,
            w1_scale,
            w2,
            w2_delta,
            w2_scale,
            experts,
            hidden,
            inter,
            inter_pad,
        )
        return flydsl_moe_a8w4_ptpc(
            x, weight, topk_weight, topk_ids, output=out, **limit_kw
        )

    def aiter_fp8_ptpc(x, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        return fm.fused_moe(
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )

    p = prepared
    candidates = {
        "a8w4": (
            a8w4,
            (
                x,
                p.w1,
                p.w1_delta,
                p.w1_scale,
                p.w2,
                p.w2_delta,
                p.w2_scale,
                topk_weight,
                topk_ids,
                out,
            ),
            _traffic(topk_ids, _bytes_per_expert(p, experts), hidden),
        )
    }
    aiter_kernels = None
    if compare:
        args = (
            x,
            fp8_w["w1"],
            fp8_w["w2"],
            topk_weight,
            topk_ids,
            fp8_w["w1_scale"],
            fp8_w["w2_scale"],
        )
        candidates["aiter_fp8_ptpc"] = (
            aiter_fp8_ptpc,
            args,
            _traffic(topk_ids, _bytes_per_expert(fp8_w, experts), hidden),
        )
        # aiter must run exactly the tuned row of its table for this M.
        row = _aiter_row(fm, m, hidden, inter_pad, experts, topk)
        aiter_kernels = _row_kernels(row)
        seen, resolve = [], fm.get_2stage_cfgs

        def spy(*a, **k):
            meta = resolve(*a, **k)
            seen.append(meta)
            return meta

        fm.get_2stage_cfgs = spy
        try:
            aiter_fp8_ptpc(*args)
        finally:
            fm.get_2stage_cfgs = resolve
        if not seen or not _uses_row(seen[-1], row):
            raise RuntimeError(
                f"{case}: aiter did not run the tuned row {aiter_kernels} (block_m={row['block_m']}); "
                f"got {seen[-1] if seen else None}"
            )

    ret = {"gfx": get_gfx_runtime()}
    lines = []
    events = []
    for name, (fn, args, nbytes) in candidates.items():
        us = bench_time_us(fn, args, events if name == "a8w4" else None)
        if name == "a8w4":
            path = list(moe_a8w4.last_path)
            result = out  # written by the timed calls of the original arguments (the last rotation slot)
        else:
            result = fn(*args)
        graded = check(result, ref, msg=f"{case} {name}: ")
        assert "reason" not in graded, f"{case} {name}: {graded['reason']}"
        if name == "a8w4":
            assert graded["ok"], f"{case}: A8W4 output out of tolerance: {graded}"
            metric = graded
            lines.append(
                f"MF_METRIC case={case} value={us:.2f} unit=us flops={flops} bytes={nbytes}"
            )
        else:
            # aiter runs per-channel FP8 weights of the same BF16 weights, not the MXFP4 weights the reference
            # dequantizes, so its distance to this reference includes the difference of the two weight
            # quantizations: only its finiteness is asserted, and its err is recorded.
            lines.append(
                f"MF_REF case={case} impl={name} value={us:.2f} unit=us flops={flops} bytes={nbytes}"
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = graded["err"]
    for line in lines:
        print(line, flush=True)
    swiglu = f" swiglu_limit={limit_kw['swiglu_limit']:g}" if limit_kw else ""
    print(
        f"INFO case={case} I={inter}->{inter_pad}{swiglu} a8w4_path={'>'.join(path)} "
        f"a8w4_check=calc_diff:{metric['calc_diff']:.2e},err_ratio:{metric['err_ratio'] * 100:.2f}%"
        + (f" aiter_kernels={aiter_kernels}" if aiter_kernels else ""),
        flush=True,
    )
    _kernel_info(
        case,
        _recorded_launches(*candidates["a8w4"][:2]),
        events,
        _distinct_experts(topk_ids),
        ret["a8w4 us"],
        candidates["a8w4"][2],
        flops,
    )
    return ret


@functools.lru_cache(maxsize=1)
def _bench_layer_weights(shape):
    """(MXFP4 routed weights, the shared expert's four prepare arguments, the A8W4 weight with the shared expert,
    aiter's FP8 per-channel weights of E + 1 experts) of a layer shape.

    aiter's: the routed experts' exact MXFP4 values (dequant_mxfp4) and the shared expert's FP8 values with I
    zero-padded to the A8W4 padded I (320 for TP8: aiter has no kernel for I 288 with the SwiGLU limit; the padding is
    exact), quantized per output channel to FP8 E4M3FNUZ (the routed ones; the shared one keeps its FP8 bits in
    E4M3FNUZ with twice its scale), the shared expert last, then shuffle_weight(layout=(16, 16)).
    """
    from aiter.ops.shuffle import shuffle_weight

    experts, _, inter = LAYER_SHAPES[shape]
    index = list(LAYER_SHAPES).index(shape)
    w = make_weights(experts, LAYER_HIDDEN, inter, seed=SEED * 1000 + 500 + index)
    shared = make_shared_weights(LAYER_HIDDEN, inter, seed=SEED * 1000 + 550 + index)[
        "row"
    ]
    prepared = prepare_moe_a8w4_ptpc_weight(
        w["w1"],
        w["w1_scale"],
        w["w2"],
        w["w2_scale"],
        shared_w1=shared[0],
        shared_w1_scale=shared[1],
        shared_w2=shared[2],
        shared_w2_scale=shared[3],
    )
    pad = prepared.inter_padded - inter
    fp8, quant = aiter.dtypes.fp8, aiter.get_torch_quant(aiter.QuantType.per_Token)
    n1, h, device = 2 * prepared.inter_padded, LAYER_HIDDEN, w["w1"].device
    w1 = torch.empty((experts + 1, n1, h), dtype=fp8, device=device)
    s1 = torch.empty((experts + 1, n1, 1), dtype=torch.float32, device=device)
    w2 = torch.empty((experts + 1, h, prepared.inter_padded), dtype=fp8, device=device)
    s2 = torch.empty((experts + 1, h, 1), dtype=torch.float32, device=device)

    def padded(v1, v2):
        z = v1.new_zeros((pad, h))
        return torch.cat([v1[:inter], z, v1[inter:], z]), torch.nn.functional.pad(
            v2, (0, pad)
        )

    for e in range(experts):
        v1, v2 = padded(
            dequant_mxfp4(w["w1"][e], w["w1_scale"][e]),
            dequant_mxfp4(w["w2"][e], w["w2_scale"][e]),
        )
        for out, scale, value in ((w1, s1, v1), (w2, s2, v2)):
            q, sc = quant(value, quant_dtype=fp8)
            out[e].copy_(q)
            scale[e].copy_(sc.view(-1, 1))
    fnuz = [
        q.view(torch.uint8).masked_fill(q.view(torch.uint8) == 0x80, 0).view(fp8)
        for q in (shared[0], shared[2])
    ]
    q1, q2 = padded(fnuz[0].view(torch.uint8), fnuz[1].view(torch.uint8))
    w1[experts].copy_(q1.view(fp8))
    w2[experts].copy_(q2.view(fp8))
    one = torch.ones((pad, 1), device=device)
    sc1 = shared[1].reshape(-1, 1) * 2
    s1[experts].copy_(torch.cat([sc1[:inter], one, sc1[inter:], one]))
    s2[experts].copy_(shared[3].reshape(-1, 1) * 2)
    fp8_w = {
        "w1": shuffle_weight(w1, layout=(16, 16)),
        "w1_scale": s1,
        "w2": shuffle_weight(w2, layout=(16, 16)),
        "w2_scale": s2,
    }
    return w, shared, prepared, fp8_w


def _aiter_kernels(meta):
    """What aiter's fused_moe resolved (get_2stage_cfgs' metadata): ``impl__<implementation>__<config>`` of a
    whole-graph implementation (the FlyDSL gfx942 MoE picks its batch-1, split-K decode or prefill kernels from the
    config and the token count), the 1-stage asm kernel or the CK 2-stage pair."""
    from aiter.fused_moe_registry import _IMPLEMENTATIONS

    if meta.full_impl is not None:
        cells = [c.cell_contents for c in meta.full_impl.__closure__ or ()]
        config = next((c for c in cells if isinstance(c, str)), "?")
        impl = next((c for c in cells if callable(c)), None)
        names = [
            name
            for name, value in _IMPLEMENTATIONS.items()
            if value is impl
            or value
            == f"{getattr(impl, '__module__', '')}:{getattr(impl, '__name__', '')}"
        ]
        return f"impl__{names[0] if names else getattr(impl, '__qualname__', impl)}__{config}"
    kernel = getattr(meta.stage1, "keywords", {}).get("kernelName")
    if meta.run_1stage:
        return f"asm_1stage:{kernel}"
    return f"{kernel}+{getattr(meta.stage2, 'keywords', {}).get('kernelName')}"


_BENCH_SIDE_STREAMS = {}


def _bench_side_stream():
    device = torch.cuda.current_device()
    if device not in _BENCH_SIDE_STREAMS:
        _BENCH_SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return _BENCH_SIDE_STREAMS[device]


@benchmark()
def bench_moe_a8w4_layer(shape, m):
    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType

    case = f"{shape}_m{m}"
    experts, topk, inter = LAYER_SHAPES[shape]
    hidden, limit = LAYER_HIDDEN, LAYER_SWIGLU_LIMIT
    w, shared, prepared, fp8_w = _bench_layer_weights(shape)
    inter_pad = prepared.inter_padded
    x, topk_weight, topk_ids = make_layer_inputs(
        m,
        hidden,
        experts,
        topk,
        seed=SEED * 1000 + 600 + BENCH_LAYER_CASES.index(case) * 4,
    )
    routed_weight = topk_weight[:, :-1].contiguous()
    routed_ids = topk_ids[:, :-1].contiguous()
    ref = reference_layer(x, w, shared, topk_weight, topk_ids, swiglu_limit=limit)
    flops_routed = 6 * m * topk * inter_pad * hidden
    flops_shared = 6 * m * inter * hidden
    flops = flops_routed + flops_shared
    p = prepared
    routed_args = (p.w1, p.w1_delta, p.w1_scale, p.w2, p.w2_delta, p.w2_scale)
    shared_args = (p.shared_w1, p.shared_w1_scale, p.shared_w2, p.shared_w2_scale)
    # Bytes of the weights and scales a call reads, each implementation in its own format (io added below).
    shared_bytes = sum(t.numel() * t.element_size() for t in shared_args)
    routed_bytes = routed_ids.unique().numel() * _bytes_per_expert(
        MoeA8W4PtpcWeight(*routed_args, experts, hidden, inter, inter_pad), experts
    )
    aiter_bytes = topk_ids.unique().numel() * _bytes_per_expert(fp8_w, experts + 1)

    # Weight tensors are arguments of their own, so that bench_graph_us copies them for every set.
    def a8w4(
        x, w1, w1d, w1s, w2, w2d, w2s, sw1, sw1s, sw2, sw2s, topk_weight, ids, out
    ):
        weight = MoeA8W4PtpcWeight(
            w1, w1d, w1s, w2, w2d, w2s, experts, hidden, inter, inter_pad,
            sw1, sw1s, sw2, sw2s,
        )  # fmt: skip
        return flydsl_moe_a8w4_ptpc(
            x, weight, topk_weight, ids, output=out, swiglu_limit=limit
        )

    def routed(x, w1, w1d, w1s, w2, w2d, w2s, topk_weight, ids, out):
        weight = MoeA8W4PtpcWeight(
            w1, w1d, w1s, w2, w2d, w2s, experts, hidden, inter, inter_pad
        )
        return flydsl_moe_a8w4_ptpc(
            x, weight, topk_weight, ids, output=out, swiglu_limit=limit
        )

    def a8w4_split(
        x, w1, w1d, w1s, w2, w2d, w2s, sw1, sw1s, sw2, sw2s, topk_weight, ids, out
    ):
        # ATOM's split arm: the routed A8W4 call and the shared expert's aiter chain, on a side stream (forked from and
        # joined to the current one) up to BENCH_SPLIT_DUAL_STREAM_MAX_M tokens, else first; no final add.
        if m <= BENCH_SPLIT_DUAL_STREAM_MAX_M:
            current, side = torch.cuda.current_stream(), _bench_side_stream()
            side.wait_stream(current)
            with torch.cuda.stream(side):
                shared_out = aiter_shared_split(x, sw1, sw1s, sw2, sw2s, limit)
            x.record_stream(side)
            routed(x, w1, w1d, w1s, w2, w2d, w2s, topk_weight, ids, out)
            current.wait_stream(side)
            shared_out.record_stream(current)
        else:
            shared_out = aiter_shared_split(x, sw1, sw1s, sw2, sw2s, limit)
            routed(x, w1, w1d, w1s, w2, w2d, w2s, topk_weight, ids, out)
        return out, shared_out

    def aiter_fp8_ptpc(x, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale):
        return fm.fused_moe(
            x,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            swiglu_limit=limit,
        )

    io = m * hidden * 2 * 2
    out = torch.full_like(x, float("nan"))
    candidates = {
        "a8w4": (
            a8w4,
            (x, *routed_args, *shared_args, topk_weight, topk_ids, out),
            routed_bytes + shared_bytes,
        ),
        "aiter_fp8_ptpc": (
            aiter_fp8_ptpc,
            (
                x,
                fp8_w["w1"],
                fp8_w["w2"],
                topk_weight,
                topk_ids,
                fp8_w["w1_scale"],
                fp8_w["w2_scale"],
            ),
            aiter_bytes,
        ),
        "a8w4_split": (
            a8w4_split,
            (
                x,
                *routed_args,
                *shared_args,
                routed_weight,
                routed_ids,
                torch.empty_like(x),
            ),
            routed_bytes + shared_bytes,
        ),
        "routed": (
            routed,
            (x, *routed_args, routed_weight, routed_ids, torch.empty_like(x)),
            routed_bytes,
        ),
    }
    # The kernels aiter resolves for this call (one eager call with get_2stage_cfgs watched).
    seen, resolve = [], fm.get_2stage_cfgs

    def spy(*a, **k):
        meta = resolve(*a, **k)
        seen.append(meta)
        return meta

    fm.get_2stage_cfgs = spy
    try:
        aiter_fp8_ptpc(*candidates["aiter_fp8_ptpc"][1])
    finally:
        fm.get_2stage_cfgs = resolve
    aiter_kernels = _aiter_kernels(seen[-1]) if seen else "?"

    ret = {"gfx": get_gfx_runtime()}
    times, results = {}, {}
    for name, (fn, args, nbytes) in candidates.items():
        times[name], results[name] = bench_graph_us(fn, args)
        if name == "a8w4":
            path = list(moe_a8w4.last_path)
            results[name] = out  # written by the last replay (the first argument set)
        elif name == "a8w4_split":
            results[name] = results[name][0].float() + results[name][1].float()
        us = times[name]
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = (flops if name != "routed" else flops_routed) / us / 1e6
        ret[f"{name} TB/s"] = (nbytes + io) / us / 1e6
        if name == "routed":
            # Only the routed experts: finite, not graded against the layer's reference.
            assert bool(torch.isfinite(results[name]).all()), f"{case} {name}: NaN/Inf"
            continue
        # aiter's routed experts are per-channel FP8 re-quantizations of the MXFP4 weights the reference dequantizes:
        # aiter's and the split's outputs are only required to be finite; their err is recorded.
        graded = check(results[name], ref, msg=f"{case} {name}: ")
        assert "reason" not in graded, f"{case} {name}: {graded['reason']}"
        if name == "a8w4":
            assert graded["ok"], f"{case}: A8W4 output out of tolerance: {graded}"
            metric = graded
        ret[f"{name} err"] = graded["err"]
    events = []
    kernel_sum = bench_time_us(a8w4, candidates["a8w4"][1], events)
    launches = _recorded_launches(a8w4, candidates["a8w4"][1])
    ret["a8w4 kernel_sum us"] = kernel_sum
    for name, kind in (
        ("a8w4", "MF_METRIC"),
        ("aiter_fp8_ptpc", "MF_REF"),
        ("a8w4_split", "MF_REF"),
    ):
        impl = "" if kind == "MF_METRIC" else f" impl={name}"
        print(
            f"{kind} case={case}{impl} value={times[name]:.2f} unit=us flops={flops} "
            f"bytes={candidates[name][2] + io}",
            flush=True,
        )
    routed_us = times["routed"]
    increment = times["a8w4"] - routed_us

    def rate(amount, us):
        return f"{amount / us / 1e6:.3f}" if us > 0 else "nan"

    print(
        f"INFO case={case} I={inter}->{inter_pad} swiglu_limit={limit:g} routed_us={routed_us:.2f} "
        f"shared_inc_us={increment:.2f} shared_tbps={rate(shared_bytes, increment)} "
        f"shared_tflops={rate(flops_shared, increment)} routed_tbps={rate(routed_bytes + io, routed_us)} "
        f"routed_tflops={rate(flops_routed, routed_us)} kernel_sum_us={kernel_sum:.2f} "
        f"a8w4_path={'>'.join(path)} aiter_kernels={aiter_kernels} "
        f"err=calc_diff:{metric['calc_diff']:.2e},err_ratio:{metric['err_ratio'] * 100:.2f}%",
        flush=True,
    )
    _kernel_info(
        case,
        launches,
        events,
        _distinct_experts(routed_ids),
        times["a8w4"],
        candidates["a8w4"][2] + io,
        flops,
    )
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_moe_a8w4_ptpc is unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL A8W4 MoE against aiter's FP8 per-token fused_moe.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        nargs="*",
        choices=list(BENCH_SHAPES) + list(LAYER_SHAPES),
        default=None,
        help="shapes (default all; the moe-a8w4 shapes run first):\n"
        + "\n".join(
            f"  {name}: (H, E, topk, I, padded I, fused shared expert, aiter reference) {dims}"
            for name, dims in BENCH_SHAPES.items()
        )
        + "\n"
        + "\n".join(
            f"  {name}: layer with the shared expert, (routed E, routed topk, I) {dims}, H {LAYER_HIDDEN}"
            for name, dims in LAYER_SHAPES.items()
        ),
    )
    tokens = sorted(
        set(BENCH_TOKENS + BENCH_EXTRA_TOKENS)
        | {int(c.rsplit("_m", 1)[1]) for c in BENCH_LAYER_CASES}
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        choices=tokens,
        default=None,
        help="token counts M (default 1, 2, 4, ..., 32768, 48; the layer shapes also 6, mtp_layer_tp4 also 1280, and "
        "the boundaries of BENCH_LAYER_BOUNDARY_CASES: 3, 5, 10, 11, 768, 1152, 1365, 1536, 98304; a shape runs only "
        "the counts of its cases)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"the layer shapes only, at M = {', '.join(map(str, BENCH_QUICK_TOKENS))} (unless -s / -m say "
        "otherwise; the lanes' bench_quick)",
    )
    args = parser.parse_args()
    if args.shape is None:
        args.shape = (
            list(LAYER_SHAPES)
            if args.quick
            else list(BENCH_SHAPES) + list(LAYER_SHAPES)
        )
    if args.m is None:
        args.m = sorted(BENCH_QUICK_TOKENS) if args.quick else tokens
    import aiter.fused_moe as fm
    from aiter.jit.core import AITER_CONFIGS

    os.environ["AITER_ONLINE_TUNE"] = "0"
    table_src = Path(aiter.__file__).parent / "configs/model_configs" / BENCH_TABLE
    with tempfile.TemporaryDirectory(prefix="bench_moe_a8w4_") as tmp:
        # A copy: aiter's duplicate check may rewrite the table file.
        table = Path(tmp) / BENCH_TABLE
        shutil.copyfile(table_src, table)
        os.environ["AITER_CONFIG_FMOE"] = str(table)
        AITER_CONFIGS.get_config_file.cache_clear()
        fm.cfg_2stages = None
        fm.get_2stage_cfgs.cache_clear()
        print(
            f"INFO seed={SEED} aiter_table={table_src} perftest={BENCH_PERFTEST_KW} warmup_ms={BENCH_WARMUP_MS}",
            flush=True,
        )
        rows = [
            bench_moe_a8w4(shape, m)
            for shape, m in itertools.product(
                [s for s in args.shape if s in BENCH_SHAPES], args.m
            )
            if f"{shape}_m{m}" in BENCH_CASES
        ]
        layers = [s for s in args.shape if s in LAYER_SHAPES]
        if layers:
            print(
                f"INFO layer_cases timing=bench_graph_us max_sets={BENCH_GRAPH_MAX_SETS} "
                f"min_replays={BENCH_GRAPH_MIN_REPLAYS} split_dual_stream_max_m={BENCH_SPLIT_DUAL_STREAM_MAX_M}",
                flush=True,
            )
        rows += [
            bench_moe_a8w4_layer(shape, m)
            for shape, m in itertools.product(layers, args.m)
            if f"{shape}_m{m}" in BENCH_LAYER_CASES
        ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl_moe_a8w4_ptpc summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if _ROOFLINE_ROWS:
        aiter.logger.info(
            "flydsl_moe_a8w4_ptpc roofline per case (markdown; 100%% lines %g TB/s and %g TFLOPS; every kernel in the "
            "INFO kernel lines):\n%s",
            BENCH_ROOF_TBPS,
            BENCH_ROOF_TFLOPS,
            pd.DataFrame(_ROOFLINE_ROWS).to_markdown(index=False, floatfmt=".4g"),
        )


if __name__ == "__main__":
    main()
