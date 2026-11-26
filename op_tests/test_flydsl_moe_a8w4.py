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
     expert-deduplicated), streaming boundaries and variants, the prefill sorting front end, prefill tile options and
     the half-block / XCD / two-pass / Down-split variants, CUDA graph, input errors and the gfx942 gate, compile-cache
     keys, repeated expert ids, in-place output, routing input dtypes, M = 0, the SwiGLU limit, the
     ``aiter.fused_moe`` entry: its tiers, arguments and errors, untagged weights, CUDA graph, the weight tag, and the
     shared expert: prepare's bit-exact conversions, the real checkpoints, errors, the stream-mode threshold, graph,
     in-place output, TP8 padding, and the mixed decode: its threshold, the FP8 bytes of its fast row quantization,
     masks, reads and other shapes): ``pytest op_tests/test_flydsl_moe_a8w4.py -m "not basic"``;
  3. benchmark (``main``, not collected by pytest): Qwen3.8 TP2 / TP4 (against aiter's FP8 per-token
     ``fused_moe``), MiMo-V2.6-Flash TP8 / TP4 and DeepSeek-V4.1-Flash TP4 x M = 1, 2, 4, ..., 32768 and 48 (85
     cases), then the DeepSeek-V4.1-Flash main and MTP layers with the shared expert, TP4 / TP8, at the same token
     counts and M = 6, the MTP TP4 layer also at M = 1280 (73 cases, against aiter's FP8 per-token ``fused_moe`` and
     ATOM's split): ``python op_tests/test_flydsl_moe_a8w4.py`` (``--quick``: the layers at 6 token counts).
``MF_SEED`` seeds the benchmark's weights and inputs.
"""

import argparse
import copy
import functools
import gc
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


# The Gate/Up and Down kernels of the per-route decode path's three dispatches (as pyhip M17's tests name them): x and
# mid quantized inside the kernels (nn), mid quantized by the row kernel first (np), both quantized first (pp).
_DECODE_PATHS = {
    "nn": ["a8w4_gateup_decode", "a8w4_down_decode"],
    "np": ["a8w4_gateup_decode", "a8w4_down_decode_pq"],
    "pp": ["a8w4_gateup_decode_pq", "a8w4_down_decode_pq"],
}
# The kernels of the expert-deduplicated decode paths (as pyhip M17's tests name them): the per-route Gate/Up also
# writing the plan and zeroing the output, then the deduplicated Down quantizing its intermediate rows (dd2); the row
# kernel quantizing x, writing the plan and zeroing the output, the deduplicated Gate/Up, the row quantization of the
# intermediate and the deduplicated Down (dd4); for H 5120 Gate/Up only (wide: the plan kernel leaves the output
# alone and the per-token slot-pair Down writes it).
_DEDUP_PATHS = {
    "dd2": ["a8w4_gateup_decode_plan", "a8w4_down_decode_dd"],
    "dd4": [
        "a8w4_row_quant_plan_decode",
        "a8w4_gateup_decode_dd",
        "a8w4_row_quant_decode",
        "a8w4_down_decode_dd",
    ],
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
    ``DECODE_WIDE_MAX_M``, past it the streaming kernels: the prep, sorting blocks of 16 (H 5120 up to 12 rows per
    expert, M <= 48 at E 24) and route_sum (I 576 >= 512: no atomic Down). The cases follow the constant (32, as in
    pyhip M17) instead of assuming its value.
    """
    if tokens <= moe_a8w4.DECODE_WIDE_MAX_M:
        assert path is not None, tokens
        _assert_path(path)
    else:
        _assert_path(
            ["a8w4_stream_prep"] + _stream_path(16, False) + ["a8w4_route_sum"]
        )


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


# The shared expert's steps in moe_a8w4.last_path when the op runs ATOM's split (aiter's dense ops; _assert_split).
_SHARED_SPLIT = [
    "shared_quant_x",
    "shared_gemm_gateup",
    "shared_silu_and_mul",
    "shared_quant_mid",
    "shared_gemm_down",
]


def _assert_shared():
    """The call ran the shared expert: some step of ``moe_a8w4.last_path`` starts with "shared".

    Deliberately independent of how the op runs it, so that later milestones can change that without these cases:
    today the mixed decode kernels up to DECODE_SHARED_MIXED_MAX_M tokens (where the fold is possible too, up to
    DECODE_SHARED_MIXED_MAX_WORK; marker shared_mixed), the fold into the
    routed experts' last reduction where moe_a8w4._shared_fold_tier allows it (shared_fold_*, with a side stream
    shared_fork / shared_join), else ATOM's split (shared_fork, shared_gemm_*, ..., shared_add; _assert_split); other
    ways record a "shared..." marker of their own. What a layer case checks is its output against reference_layer.
    """
    path = moe_a8w4.last_path
    assert any(name.startswith("shared") for name in path), path


def _assert_layer_path(tier):
    """Path check of a layer call (routed experts plus the shared expert), as independent of the shared expert's
    implementation as _assert_shared: the routed experts took ``tier`` ("decode", "dedup", "stream" or "prefill",
    _path_tier: told by the first A8W4 Gate/Up kernel) and the shared expert ran."""
    assert _path_tier(moe_a8w4.last_path) == tier, (tier, moe_a8w4.last_path)
    _assert_shared()


def _assert_split(tokens, inplace=False):
    """The shared expert ran ATOM's split and was added last: up to SHARED_DUAL_STREAM_MAX_M tokens with a separate
    output on the side stream (fork, the split, the routed kernels on the current stream, join), beyond it or in place
    (``inplace``: the output is the input) on the current stream before the routed kernels. Only for the cases about
    the split itself, which pin it with SHARED_FOLD = 0.
    """
    path = moe_a8w4.last_path
    assert path[-1] == "shared_add", path
    if tokens <= moe_a8w4.SHARED_DUAL_STREAM_MAX_M and not inplace:
        _assert_path(["shared_fork"] + _SHARED_SPLIT + ["shared_join", "shared_add"])
        fork, join = path.index("shared_fork"), path.index("shared_join")
        assert any(name.startswith("a8w4_gateup") for name in path[fork:join]), path
    else:
        assert "shared_fork" not in path, path
        _assert_path(_SHARED_SPLIT + ["shared_add"])
        first = min(i for i, name in enumerate(path) if name.startswith("a8w4_"))
        assert path.index("shared_gemm_down") < first, path


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
# TP4 M 257 and MiMo TP4 M 1600, the expert-deduplicated decode for MiMo TP8 M 5 (two kernels) and TP4 M 24 (four
# kernels), and for DeepSeek M 8 the decode that deduplicates Gate/Up only while H 5120 decodes M 8, else the
# streaming kernels (_assert_h5120 follows the H 5120 decode limit DECODE_WIDE_MAX_M).
# The aiter.fused_moe entry (test_fused_moe_entry): fused_moe on the tagged w1 / w2 of a prepared weight with
# activation=Silu, quant_type=per_1x32 and quant_type_a=per_Token runs flydsl_moe_a8w4_ptpc (one Qwen decode case).
# DeepSeek-V4.1-Flash layers with the shared expert (test_layer, test_fused_moe_shared_entry; LAYER_SHAPES with their
# full E, the bench's shapes): prepare with the shared expert, routing [M, topk + 1] with the shared expert E in the
# last column and a random weight per token, swiglu_limit 10, inputs scaled by 10 (more than 10% of the shared
# expert's gate and up values exceed the limit), graded against reference_layer, with the routed tier asserted and the
# shared expert's run checked independent of how the op runs it (_assert_layer_path: today the mixed decode kernels
# up to M 4 (main TP4 M 4 folds), else the fold into the routed experts' last reduction, ATOM's split where the routed
# path does not allow it; later milestones may change that). One case per routed tier of the main layer TP4
# (decode M 1, streaming M 64, prefill M 2048), the TP8 I 288 -> 320 padding (decode M 4) and the MTP layer (E 128,
# topk 3, the official 32x32-block format folded by prepare; decode M 16), and fused_moe on the E385 / topk-7 call
# (M 8, with shared_expert_id=E).
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
        (256, 5, _DEDUP_PATHS["dd2"]),
        (256, 300, "_stream_"),
        (512, 24, _DEDUP_PATHS["dd4"]),
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
# (_path_tier, asserted by _assert_layer_path); the MTP M 16 case would take the streaming tier past the H 5120 decode
# limit DECODE_WIDE_MAX_M (as _assert_h5120). Only the tier is asserted, not the kernels: how the shared expert runs
# (and so which kernels a layer call launches) is left to later milestones.
_LAYER_CASES = [
    ("ds_layer_tp4", 1, "decode", "row"),
    ("ds_layer_tp4", 64, "stream", "row"),
    ("ds_layer_tp4", 2048, "prefill", "row"),
    ("ds_layer_tp8", 4, "decode", "row"),
    ("mtp_layer_tp4", 16, "dedup", "block"),
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
# They assert the Gate/Up and Down kernels (_assert_path); the sorting front end (the streaming prep from H 2560 on,
# H % 512 == 0, while M * topk <= PREFILL_PREP_MAX_ROUTES, 12288 from H 5120 on; else Opus) is asserted by the
# streaming and front-end cases of 2.4.
# Cases about the per-route decode kernels use shapes and token counts that pyhip M17 decodes per route too (H 2560 up
# to M 32, H 4096 with I_pad % 128 != 0, H 512, DeepSeek H 5120 up to M 7) and assert the decode kernels by name. The
# expert-deduplicated decode (H 4096 with I_pad % 128 == 0 from M 2 on, the MiMo shapes; DeepSeek H 5120 from M 8 on,
# Gate/Up only) follows pyhip M17's cases: MiMo shapes with E cut to 12, so that groups of two and three routes form,
# and the full E 256 only where the boundary depends on E. Its H 5120 cases pin their kernels within the H 5120
# decode limit DECODE_WIDE_MAX_M and the streaming kernels past it (_assert_h5120).
# The streaming tier (M > 32 with at most STREAM_MAX_ROWS* rows per expert) follows pyhip M17's cases too: the basic
# tests stream blocks of 16 and 32, the benchmark 16 / 32 / 64 without asserting kernels; here its boundaries with
# decode and prefill, the bit-exact prep and fused quantization, ACT_STEPS, the prefill Down behind the streaming
# Gate/Up, grid shrinking, CUDA graph, compile-cache keys, repeated ids, in-place output and the SwiGLU limit.
# The aiter.fused_moe entry (2.9; its gfx942 gate in 2.6) is covered by the basic test with one decode case and by no
# benchmark case (the benchmark calls flydsl_moe_a8w4_ptpc directly).
# The shared expert (2.10) is covered by the basic layer tests (one case per routed tier, TP8, the MTP layer in the
# 32x32-block format, fused_moe's E385 / topk-7 call) and the benchmark's 73 layer cases (graded); here what they
# cannot show: the bit-exact conversions of prepare (E4M3FN -> FNUZ, -0, the block folding, the I padding), the
# folding of the real checkpoints (byte-identical to FP4PTPC@97cc9b32; needs AITER_MOE_A8W4_DSV41_DIR, else skipped)
# and the accuracy on real weights, input errors and the debug checks, CUDA graph replays, in-place output, the
# exactness of the TP8 padding through the split's chain, ATOM's split's own stream-mode threshold, add and in-place
# order, and fused_moe's shared_expert_id and raw shared tensors. The layer cases check that the shared expert ran and
# grade the output, but not how it ran (_assert_shared / _assert_layer_path): later milestones change where the op
# folds it. The cases about the split itself (_assert_split, moe_a8w4._shared_split) pin it with SHARED_FOLD = 0; the
# mixed decode (2.12) and the fold (2.13) have their own sections.
# =================================================================================================


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
    "dtype, cols, fill",
    [("bf16", 2560, 11), ("f32", 512, 40)],
    ids=["bf16_fill11", "f32_fill40"],
)
def test_row_quant_fill(dtype, cols, fill):
    """The input quantization also sets loc_ids [rows, fill] to -1 (Down then writes only the sorted rows of valid
    routes): q and scale are bit-identical to the call without ``fill``, and the -1s cover exactly ``rows`` rows,
    nothing past them; with fill = 40 against 32 lanes per f32 row of 512 columns each lane writes two rounds.

    End to end these -1s only show with a repeated expert id of a token
    (test_prefill_duplicate_ids_keep_last_route), and an out-of-range write does not show at all.
    """
    gen = torch.Generator(device="cuda").manual_seed(91)
    rows = 997
    x = torch.randn((rows, cols), generator=gen)
    x = x.to(torch.bfloat16) if dtype == "bf16" else x
    q0, s0 = moe_a8w4._quantize(x, rows, cols, dtype)
    aux = torch.full((rows + 1, fill), 7, dtype=torch.int32)
    q, s = moe_a8w4._quantize(x, rows, cols, dtype, fill=aux[:rows])
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8)) and torch.equal(s, s0)
    assert bool((aux[:rows] == -1).all()) and bool((aux[rows] == 7).all())


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
    "cols, topk, rows, max_m, zero, masked",
    [
        (MIMO_HIDDEN, MIMO_TOPK, 37, 64, True, False),
        (DS_HIDDEN, DS_TOPK, 29, 32, False, False),
        (DS_HIDDEN, DS_TOPK, 29, 32, False, True),
    ],
    ids=["h4096_zero_out", "h5120_keep_out", "h5120_masked"],
)
def test_decode_plan_quant_bitexact(cols, topk, rows, max_m, zero, masked):
    """The plan kernel of the expert-deduplicated decode (a8w4_row_quant_plan: four waves quantize a row of x in
    16-byte pieces, the amax merged in LDS, and the same launch writes the dedup plan; for H 4096 it also zeroes the
    output rows that the atomic Down adds into): xq / xs equal ``_quantize`` and the reference bit for bit; with
    zeroing the first ``rows`` rows of out are 0 and the rows past them unwritten, for H 5120 (whose slot-pair Down
    writes out itself) no row of out is written; the plan entries equal the definition (an expert's routes in route
    order, groups of 3: the leader holds the count and the members, every other route the count 0; words past the
    count are not read and not checked).

    Special rows as in test_decode_row_quant_bitexact (zero, a subnormal scale, -0 and negatives rounding to -0); a
    row has 512 16-byte pieces for H 4096 (256 threads, two rounds) and 640 for H 5120 (the third round only the
    first 128 threads); 37 rows x topk 8 = 296 and 29 x 6 = 174 routes on 40 experts, planned for M <= 64 / <= 32 (8 /
    3 ballot segments of 64 routes), so members across segments and partial tail groups both occur. End to end the
    tolerance does not show a few flipped FP8 values, nor H 5120 zeroing out by mistake (the slot-pair Down
    overwrites whole rows); a wrong plan mostly fails end to end, but random routing may miss the cross-segment and
    partial-tail combinations.

    With masked routes (id -1, a random 30% and rows 2 and 7 whole) a masked route gets the entry [0, route] and the
    groups list valid routes only: its ballots also match the -1 that pads the plan past the routes, so it must
    lead no group. End to end a masked leader would show only through the result of its group (test_masked_routes
    runs the dedup paths on poisoned weights); this pins every entry.
    """
    guard, group = 3, 3
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
    out = torch.full((rows + guard, cols), float("nan"), dtype=torch.bfloat16)
    moe_a8w4._launch(
        "a8w4_row_quant_plan_decode",
        moe_a8w4._row_quant_plan(cols, topk, group, max_m, zero, x.device.index),
        x,
        q.view(torch.float8_e4m3fnuz),
        scale,
        topk_ids,
        plan,
        out,
        rows,
    )
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    assert bool((out[:rows] == 0).all() if zero else out[:rows].isnan().all())
    assert bool(out[rows:].isnan().all())
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


@pytest.mark.parametrize("kernel", ["decode_bf16", "decode_f32", "plan_bf16"])
def test_fast_row_quant_bitexact(kernel):
    """The decode row-quantization kernels divide once per row (the reciprocal) and correct each product with FMAs
    (decode_a8w4._quant_fast: one correction for BF16 rows, two for f32; rows outside _fast_div_ok divide per value):
    their FP8 bytes and scale bits equal the IEEE division's (_ieee_quant) on _fast_quant_rows, i.e. on both sides of
    both ends of the fast range (2**-60, 2**127), on zero and subnormal rows and on rounding ties, where a bare
    reciprocal product rounds values of every tie row the other way. a8w4_row_quant_decode on BF16 x (H 5120) and on
    f32 intermediate rows (I 576), a8w4_row_quant_plan_decode on BF16 x (H 5120, topk 6; its plan is
    test_decode_plan_quant_bitexact's).

    test_decode_row_quant_bitexact and test_decode_plan_quant_bitexact use random rows with zero, subnormal and -0
    rows, none at the ends of the fast range or at ties; end to end a few flipped FP8 values stay within the tolerance.
    """
    cols = DS_INTER if kernel == "decode_f32" else DS_HIDDEN
    rows, _ = _fast_quant_rows(cols)
    x = rows if kernel == "decode_f32" else rows.to(torch.bfloat16)
    n = x.shape[0]
    q = torch.full((n, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((n,), -1.0)
    if kernel == "plan_bf16":
        _, _, topk_ids = make_inputs(n, 512, 40, DS_TOPK, seed=785, shared=False)
        moe_a8w4._launch(
            "a8w4_row_quant_plan_decode",
            moe_a8w4._row_quant_plan(cols, DS_TOPK, 3, 32, False, x.device.index),
            x,
            q.view(torch.float8_e4m3fnuz),
            scale,
            topk_ids,
            torch.full((n * DS_TOPK, 4), -1, dtype=torch.int32),
            torch.empty((n, cols), dtype=torch.bfloat16),
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


@pytest.mark.parametrize(
    "cols, qw",
    [(512, 4), (2560, 4), (2560, 1), (4096, 4)],
    ids=["h512_qw4", "h2560_qw4", "h2560_qw1", "h4096_qw4"],
)
def test_stream_row_quant_bitexact(cols, qw):
    """The streaming prep (a8w4_stream_prep: one launch sorts the routes, quantizes x and zeroes out): q and scale
    equal ``_quantize`` and the reference bit for bit, the first ``rows`` rows of out are zero and no row past
    ``rows`` is written; the sort has aiter moe_sorting's format (each route's loc_ids holds its own sorted row,
    experts ascending, each padded to whole blocks, padding rows token M and weight 0, num_valid [padded rows, M]).

    Both quantization layouts: QW 4 takes a row per workgroup, its 4 waves splitting the 512-column chunks (H 512:
    wave 0 only; H 2560: 5 chunks, uneven), the amax merged in LDS, a reciprocal product checked at x(1 -+ 2**-20)
    that falls back to IEEE division near a rounding boundary or for a subnormal scale; QW 1 (the dispatch's choice
    for H 2560 past M 256) takes a row per wave and divides per element. Special rows: zero, a subnormal scale, -0
    and negatives rounding to -0 (stored as +0), quotients exactly at rounding midpoints with the scale 2**-2, and
    midpoints that a plain reciprocal product rounds the wrong way with the scale 7/32; 37 rows (QW 1: the last
    workgroup holds one row). E 513 and topk 11: 2 sorting workgroups, 3 experts per thread.

    End to end the tolerance does not show a few flipped FP8 values, and the basic tests and the benchmark only
    quantize random rows and never look at the sort format.
    """
    rows, guard, experts, block_m = 37, 3, 513, 16
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
    n_sort, capacity, blocks = stream_a8w4.stream_prep_sizes(
        rows, TOPK, experts, block_m
    )
    sorted_ids = torch.full((capacity,), -1, dtype=torch.int32)
    sorted_weights = torch.full((capacity,), -1.0)
    sorted_experts = torch.full((blocks,), -1, dtype=torch.int32)
    num_valid = torch.full((2,), -1, dtype=torch.int32)
    loc_ids = torch.full((rows, TOPK), -2, dtype=torch.int32)
    q = torch.full((rows + guard, cols), 0x5A, dtype=torch.uint8)
    scale = torch.full((rows + guard,), -1.0)
    out = torch.full((rows + guard, cols), float("nan"), dtype=torch.bfloat16)
    moe_a8w4._launch(
        "a8w4_stream_prep",
        moe_a8w4._stream_prep_kernel(
            cols, TOPK, experts, block_m, True, qw, x.device.index
        ),
        topk_ids,
        topk_weight,
        x,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        num_valid,
        loc_ids,
        q.view(torch.float8_e4m3fnuz),
        scale,
        out,
        rows,
        n_sort,
        capacity,
    )
    assert bool((q[rows:] == 0x5A).all()) and bool((scale[rows:] == -1.0).all())
    assert bool((out[:rows] == 0).all()) and bool(out[rows:].isnan().all())
    q0, scale0 = moe_a8w4._quantize(x, rows, cols, "bf16")
    assert torch.equal(q[:rows], q0.view(torch.uint8))
    assert torch.equal(scale[:rows].view(torch.int32), scale0.view(torch.int32))
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    ref_scale = amax / torch.full_like(amax, 240.0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)
    assert float(ref_scale[2]) < 2.0**-126
    assert float(ref_scale[4]) == 0.25 and float(ref_scale[5]) == 7 / 32
    assert torch.equal(scale[:rows].view(-1, 1), ref_scale)
    exact = (xf / ref_scale).to(torch.float8_e4m3fnuz).view(torch.uint8)
    naive = (
        (xf * (torch.ones_like(ref_scale) / ref_scale))
        .to(torch.float8_e4m3fnuz)
        .view(torch.uint8)
    )
    assert torch.equal(q[:rows], exact) and bool((exact[5] != naive[5]).any())
    counts = torch.bincount(topk_ids.flatten().long(), minlength=experts)
    padded = (counts + block_m - 1) // block_m * block_m
    total = int(padded.sum())
    assert num_valid.tolist() == [total, rows]
    loc = loc_ids.flatten().long()
    route = torch.arange(rows * TOPK, dtype=torch.int32)
    assert torch.equal(sorted_ids[loc], (route // TOPK) | ((route % TOPK) << 24))
    assert torch.equal(sorted_weights[loc], topk_weight.flatten())
    assert torch.equal(sorted_experts[loc // block_m], topk_ids.flatten())
    assert torch.equal(
        sorted_experts[: total // block_m],
        torch.repeat_interleave(
            torch.arange(experts, dtype=torch.int32), padded // block_m
        ),
    )
    filler = torch.ones(total, dtype=torch.bool)
    filler[loc] = False
    assert bool((sorted_ids[:total][filler] == rows).all())
    assert bool((sorted_weights[:total][filler] == 0).all())


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
        (320, 12, "pp"),
        (320, 32, "pp"),
        (320, 33, "stream"),
        (160, 19, "np"),
        (160, 20, "pp"),
    ],
    ids=["tp2_m4", "tp2_m11", "tp2_m12", "tp2_m32", "tp2_m33", "tp4_m19", "tp4_m20"],
)
def test_decode_dispatch_boundaries(inter, tokens, path):
    """Both sides of each decode dispatch boundary on Qwen shapes (E 33), end to end with the Gate/Up and Down
    kernels: M = 4 (Down takes the slot-pair kernel on rows quantized by the row kernel; M = 3 is a basic case),
    TP2 M = 11 / 12 and TP4 M = 19 / 20 (the Gate/Up workgroups M * topk * I_pad / 16 reach
    DECODE_GATEUP_PREQUANT_MIN_WGS = 2560 and Gate/Up reads x quantized by the row kernel), M = 32 / 33
    (DECODE_MAX_M: M = 33 takes the sorted streaming path, blocks of 16 at 11 rows per expert with E 33).

    The basic tests decode Qwen M = 3 and MiMo M = 5, on none of these boundaries; the benchmark runs powers of two
    and 48 (no 11 / 12, 19 / 20 or 33) and asserts no kernels: M = 4 and 32 run the benchmark's paths and stay as
    one side of their boundaries.
    """
    w, prepared = _decode_weights(inter)
    x, topk_weight, topk_ids = make_inputs(
        tokens, HIDDEN, EXPERTS, TOPK, seed=90 + tokens
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "stream":
        _assert_path(["a8w4_stream_prep"] + _stream_path(16, False))
    else:
        _assert_path(_DECODE_PATHS[path])
    assert_close(out, ref)


@pytest.mark.parametrize(
    "topk, tokens, path",
    [(8, 16, "pp"), (17, 2, "nn")],
    ids=["topk8_m16", "topk17_m2"],
)
def test_decode_topk_variants(topk, tokens, path):
    """topk other than 11: with topk = 8 (even) the slot-pair Down's 3 waves take two rounds and both slots of the
    second round's last two waves are invalid, and M = 16 gives exactly 2560 Gate/Up workgroups (>=: x pre-quantized);
    with topk = 17 (above 16) the Down quantizing in the kernel runs 4 waves taking the slots in turn, 3 slots of its
    last round invalid.

    The Qwen cases of the basic tests and the benchmark have topk = 11 (only the last slot pair lacks its upper
    half); the benchmark's MiMo cases (topk 8) take the expert-deduplicated decode from M 2 on (M 1 the single-slot
    Down), never the slot-pair Down; no case of either has topk > 16.
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(
        tokens, HIDDEN, EXPERTS, topk, seed=94 + topk
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
    masked route's offsets fall); larger w2 takes the single-slot kernel quantizing in the kernel (M = 16). Both sides
    of that boundary: w2 of 2**32 bytes minus one expert's (``fits``) and one byte more (``over``, still below 4 GiB).

    Real 4 GiB weights are too large: a stride-0 view makes w2 report that many bytes with the same data pointer (the
    kernels read by pointer and expert offset, the routes only real experts); the basic tests, the benchmark and the
    other cases have w2 far below 4 GiB.
    """
    import dataclasses

    w, prepared = _decode_weights(320)
    size = (1 << 32) - prepared.w2[0].numel() * prepared.w2.element_size() + over
    big = dataclasses.replace(prepared, w2=prepared.w2.view(-1)[:1].expand(size))
    x, topk_weight, topk_ids = make_inputs(16, HIDDEN, EXPERTS, TOPK, seed=99)
    out, ref = _a8w4_run(w, big, x, topk_weight, topk_ids)
    _assert_path(["a8w4_gateup_decode_pq", down])
    assert_close(out, ref)


def _mimo_small_weights(inter):
    """MiMo-shape weights (H 4096, topk 8) of the expert-deduplicated decode cases with E cut to 12: each expert gets
    2M/3 routes on average (cached per I)."""
    return _weights(12, MIMO_HIDDEN, inter, 200 + inter)


def _spy_dedup(monkeypatch):
    """Record build arguments of the dedup launchers (routes per group, waves of the deduplicated Gate/Up, whether
    the deduplicated Down quantizes in the workgroup), by their positions in pyhip M17's signatures.
    """
    seen = {}

    def wrap(name, fields):
        original = getattr(moe_a8w4, name)

        def spy(*args):
            seen.update({key: args[i] for key, i in fields.items()})
            return original(*args)

        monkeypatch.setattr(moe_a8w4, name, spy)

    wrap("_gateup_plan_kernel", {"group": 4})
    wrap("_row_quant_plan", {"group": 2})
    wrap("_gateup_dd_kernel", {"group": 3, "waves": 4})
    wrap("_down_dd_kernel", {"down_group": 3, "quant": 5})
    return seen


@pytest.mark.parametrize(
    "inter, tokens, path, group, waves",
    [
        (256, 2, "dd2", 2, None),
        (256, 11, "dd2", 2, None),
        (256, 12, "dd4", 2, 8),
        (256, 16, "dd4", 2, 8),
        (256, 17, "dd4", 3, 4),
        (512, 5, "dd2", 2, None),
        (512, 6, "dd4", 2, 8),
        (384, 2, "dd4", 2, 8),
    ],
    ids=[
        "i256_m2",
        "i256_m11",
        "i256_m12",
        "i256_m16",
        "i256_m17",
        "i512_m5",
        "i512_m6",
        "i384_m2",
    ],
)
def test_decode_dedup_dispatch(monkeypatch, inter, tokens, path, group, waves):
    """Both sides of each boundary of the expert-deduplicated decode, end to end with its kernels and build
    parameters: two kernels become four once M * I_pad reaches DECODE_DEDUP_GATEUP_MIN_MI = 3072 (I 256 M 11 / 12,
    I 512 M 5 / 6); from M 17 on (DECODE_DEDUP_PAIR_MAX_M = 16) groups hold 3 routes instead of 2 and the
    deduplicated Gate/Up runs 4 waves per workgroup instead of 8; I 384 (I_pad % 256 != 0: Down cannot quantize in
    the workgroup) takes four kernels already at M 2. The M 1 side (the per-route decode) is the benchmark's
    mimo_*_m1.

    E 12 (topk 8) gives each expert 2M/3 routes on average, so full groups of 2 / 3, partial tail groups and routes of
    one token in different groups adding atomically all occur; MiMo's E 256 gives groups of one route at small M. The
    benchmark's MiMo cases run powers of two and 48 and assert no kernels; the basic MiMo decode cases run M 5 / 24.
    """
    seen = _spy_dedup(monkeypatch)
    w, prepared = _mimo_small_weights(inter)
    x, topk_weight, topk_ids = make_inputs(
        tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=210 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(_DEDUP_PATHS[path])
    assert_close(out, ref)
    assert (seen["group"], seen.get("waves"), seen["down_group"], seen["quant"]) == (
        group,
        waves,
        group,
        path == "dd2",
    ), seen


def test_decode_dedup_plan_prequant():
    """The two-kernel form of the expert-deduplicated decode with x quantized by the row kernel first: the per-route
    Gate/Up that writes the plan and zeroes the output reads pre-quantized x (a8w4_gateup_decode_plan_pq) once its
    workgroups M * topk * I_pad / 16 reach DECODE_GATEUP_PREQUANT_MIN_WGS = 2560 while M * I_pad stays below 3072
    (H 4096, I 256, topk 16, M 10: 2560 workgroups, M * I = 2560); its plan holds up to DECODE_MAX_M * 16 = 512
    routes (8 ballot segments).

    That needs topk >= 14 at H >= 4096: the target shapes (MiMo topk 8) never get there, so neither the basic tests,
    the benchmark nor the other dedup cases run this kernel variant.
    """
    w, prepared = _weights(20, MIMO_HIDDEN, 256, 225)
    x, topk_weight, topk_ids = make_inputs(
        10, MIMO_HIDDEN, 20, 16, seed=226, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(
        [
            "a8w4_row_quant_decode",
            "a8w4_gateup_decode_plan_pq",
            "a8w4_down_decode_dd",
        ]
    )
    assert_close(out, ref)


@pytest.mark.parametrize("tokens", [48, 49], ids=["m48", "m49"])
def test_decode_dedup_stream_boundary(tokens):
    """Above DECODE_MAX_M tokens the expert-deduplicated decode continues only while an expert gets at most 1.5 routes
    on average (2 M topk <= 3 E, up to DECODE_DEDUP_MAX_M = 64): MiMo TP8 (E 256, topk 8, I 256) M 48 still takes the
    four dedup kernels, M 49 streams (the prep sorts in blocks of 16; I 256 < 512: the Down adds atomically, no
    route_sum).

    The boundary depends on E, so only the full E 256 reaches it; the benchmark runs M 48 and 64 and asserts no
    kernels.
    """
    w, prepared = _weights(MIMO_EXPERTS, MIMO_HIDDEN, 256, 230)
    x, topk_weight, topk_ids = make_inputs(
        tokens, MIMO_HIDDEN, MIMO_EXPERTS, MIMO_TOPK, seed=230 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if tokens == 48:
        _assert_path(_DEDUP_PATHS["dd4"])
    else:
        _assert_path(["a8w4_stream_prep"] + _stream_path(16, False))
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
    the single-slot kernel quantizing in the kernel; the H 5120 decode ends at DECODE_WIDE_MAX_M = 32: M 32 still
    deduplicates Gate/Up (groups of 3), M 33 streams. With E 256, M 33 gives 0.77 routes per expert (the dedup
    decode's <= 1.5 holds) and still streams, and with fewer than 4E/3 routes it still sorts in blocks of 16 (H 5120
    sorts with the prep, any route count).

    By default H 5120 deduplicates Gate/Up from M 8 on, so the per-route pre-quantized Gate/Up (two rounds) is only
    reached with the threshold raised; the benchmark's ds_tp4 cases (E 384) run powers of two and 48 without
    asserting kernels (no M 7, 33); the basic tests and the benchmark have w2 far below 4 GiB. The cases pin their
    decode kernels within the H 5120 decode limit and the streaming kernels past it (_assert_h5120), and the w2
    fallback, moot past the limit, is skipped there.
    """
    import dataclasses

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
        assert 2 * tokens * DS_TOPK <= 3 * experts
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


# -- 2.4 streaming and prefill: tier boundaries, grid shrinking, front end, tiles, H 5120 thresholds, variants ----


def test_grid_shrink_boundary_e513():
    """Both sides of the streaming grid shrinking at the full E 513 (the prep's block capacity, and with it the grid
    of the streaming Gate/Up and Down, is M * topk once that is <= E): M 46 (506 routes) and M 47 (517).

    The basic tests (E 33 / 256) are not on this boundary; the benchmark has only M 48 between 33 and 63 (528 routes
    > E, not shrunk, the M 47 side). Both M have fewer than 4E/3 routes and sort in blocks of 32, the path of the
    benchmark's M 48.
    """
    w = make_weights(513, HIDDEN, 320, seed=31)
    prepared = _a8w4_prepare(w)
    for tokens in (46, 47):
        x, topk_weight, topk_ids = make_inputs(
            tokens, HIDDEN, 513, TOPK, seed=32 + tokens
        )
        assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        _assert_path(["a8w4_stream_prep"] + _stream_path(32, True))


@pytest.mark.parametrize(
    "tokens, path",
    [(66, _stream_path(32, True)), (67, _prefill_path(128))],
    ids=["m66", "m67"],
)
def test_stream_dispatch_boundary(tokens, path):
    """The boundary between streaming and prefill (streaming while M * topk <= STREAM_MAX_ROWS * E, 22 rows per
    expert at H 2560 with I 320): with E 33 and topk 11, M 66 (22 rows) still streams (blocks of 32, the Gate/Up
    quantizing the intermediate) and M 67 takes prefill with the default K tile 128; both sort with the prep.

    The basic tests (streaming M 64 / 300, prefill M 257 / 1031 / 1600) and the benchmark are not on this boundary.
    """
    w, prepared = _decode_weights(320)
    x, topk_weight, topk_ids = make_inputs(tokens, HIDDEN, EXPERTS, TOPK, seed=tokens)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(["a8w4_stream_prep"] + path)
    assert_close(out, ref)


@pytest.mark.parametrize(
    "hidden, inter, tokens, path",
    [
        (2560, 160, 140, _stream_path(64, True)),
        (2560, 160, 141, _prefill_path(128)),
        (4096, 256, 62, _stream_path(32, False)),
        (4096, 256, 120, _stream_path(64, False)),
        (4096, 256, 121, _prefill_path(128)),
    ],
    ids=[
        "h2560_i192_m140",
        "h2560_i192_m141",
        "h4096_i256_m62_bm32",
        "h4096_i256_m120",
        "h4096_i256_m121",
    ],
)
def test_stream_rows_limits(hidden, inter, tokens, path):
    """The streaming limit by shape, both sides each: 48 rows per expert from H 4096 on (STREAM_MAX_ROWS_WIDE) and 56
    at H 2560 when the Gate/Up of blocks of 64 quantizes the intermediate (I_pad 192, STREAM_MAX_ROWS_FUSED_QUANT);
    with E 5 and topk 2, M 140 / 141 give 56 / 56.4 rows and M 120 / 121 48 / 48.4. H 4096 with I 256 streams with
    the atomic Down, which reads the activations from LDS at blocks of 32 (M 62, K % 256 == 0).

    The basic tests and the benchmark are not on these two boundaries (the 22-row one is
    test_stream_dispatch_boundary); the benchmark's MiMo TP8 M512 / 1024 run both atomic Downs without asserting
    kernels. E 5 and topk 2 keep this fast.
    """
    w, prepared = _small_weights(hidden, inter)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, 5, 2, seed=300 + tokens, shared=False
    )
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path(path)
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
    "kernel, nsplit, two_pass, xcd",
    [
        ("gateup", None, None, moe_a8w4.GATEUP_XCD_CHUNK_H5120),
        ("down", 4, True, 0),
        ("down", 4, True, moe_a8w4.DOWN_XCD_CHUNK_H5120),
        ("down", 1, True, moe_a8w4.DOWN_XCD_CHUNK_H5120),
    ],
    ids=[
        "gateup_xcd",
        "down_two_pass",
        "down_two_pass_xcd_split4",
        "down_two_pass_xcd_split1",
    ],
)
def test_prefill_xcd_two_pass_bitexact(kernel, nsplit, two_pass, xcd):
    """The two H5120 prefill grid variants equal the plain kernels bit for bit on one sort and one input: the
    XCD-chunked 1-D grid (XCD_CHUNK > 0: Gate/Up chunks of one sorted block's 9 N tiles, Down chunks of 4
    neighbouring sorted blocks with their hidden parts, dealt to the 4 XCDs in turn) for mid, routes and loc_ids,
    and Down's activation tile staged in two passes through LDS (TWO_PASS: K = 576 as 5 + 4 blocks of 64). The
    baseline is the 2-D grid with one pass and the hidden dimension split in 4.

    The routes are built per expert (E 24 with 32, 33, 64, 65, 96, 97, 1, 20 and 16 x 12 rows, M = 100): both
    sides of Down's half-block branch and every valid-row count of Gate/Up occur; the Opus sort's 34 tasks are no
    multiple of 4 or 16, so workgroups the chunked grid maps past the task count must exit; Down runs split in 4
    and unsplit. The basic tests and the benchmark only grade errors, so a chunk computed twice, skipped or
    missing its loc_ids may go unnoticed (XCD chunking only starts at 16384 routes, the benchmark's ds_tp4
    M >= 4096); the one-pass K = 576 tile only runs below 81 estimated workgroups.
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
    xq, xs = moe_a8w4._quantize(
        x, M, H, "bf16", fill=torch.empty((M, topk), dtype=torch.int32)
    )
    device, constants = x.device.index, moe_a8w4._trace_constants()

    def gateup(chunk):
        mid = torch.full((M, topk, inter), float("nan"))
        kernel = moe_a8w4._gateup_kernel(
            2 * inter, H, topk, 64, 128, 128, E, 0.0, chunk, device, constants
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
        assert torch.equal(gateup(xcd).view(torch.int32), mid.view(torch.int32))
        return
    mq, ms = moe_a8w4._quantize(mid, M * topk, inter, "f32")
    routes, loc = down(4, False, 0)
    assert bool((loc >= 0).all())
    other_routes, other_loc = down(nsplit, two_pass, xcd)
    assert torch.equal(other_routes, routes) and torch.equal(other_loc, loc)


def _spy(monkeypatch, name, record):
    """Wrap ``moe_a8w4.<name>`` (a launcher factory) so that each call's arguments go to ``record``."""
    original = getattr(moe_a8w4, name)

    def spy(*args):
        record.append(args)
        return original(*args)

    monkeypatch.setattr(moe_a8w4, name, spy)


@pytest.mark.parametrize(
    "min_wgs, nsplit", [(5, 1), (8, 2), (1000, 4)], ids=["split1", "split2", "split4"]
)
def test_stream_prefill_down_split(monkeypatch, min_wgs, nsplit):
    """Sorting blocks of 64 without the atomic Down (I >= 512) from hidden 4096 on: behind the streaming Gate/Up the
    prefill Down writes the routes (and loc_ids) for route_sum (STREAM_PREFILL_DOWN), splitting the hidden dimension
    over 1, 2 or 4 workgroups per sorted block by the estimated blocks times the split against ``DOWN_MIN_WGS`` (E 5,
    M 100, topk 2: 5 blocks; DOWN_MIN_WGS 5 / 8 / default 1000); each part writes its columns of the routes and only
    part 0 the loc_ids.

    The basic tests and the benchmark run whichever split their shape gets and assert neither the split nor its
    result per split (the basic MiMo TP4 M 1600 is prefill with split 4; the benchmark streams into the prefill Down
    only at mimo_tp4_m1024); here all three run on one shape against the same reference.
    """
    monkeypatch.setattr(moe_a8w4, "DOWN_MIN_WGS", min_wgs)
    assert moe_a8w4._down_split(100, 2, 5, 4096, 512, 64) == nsplit
    built = []
    _spy(monkeypatch, "_down_kernel", built)
    w, prepared = _small_weights(4096, 512)
    x, topk_weight, topk_ids = make_inputs(100, 4096, 5, 2, seed=310, shared=False)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    assert [args[5] for args in built] == [nsplit], built
    _assert_path(
        [
            "a8w4_stream_prep",
            "a8w4_gateup_stream_bm64",
            "a8w4_down_prefill_bm64",
            "a8w4_route_sum",
        ]
    )
    assert_close(out, ref)


@pytest.mark.parametrize(
    "tokens, first, duplicate",
    [
        (200, "a8w4_stream_prep", False),
        (201, "moe_sorting_opus", False),
        (200, "a8w4_stream_prep", True),
    ],
    ids=["m200_prep", "m201_opus", "m200_prep_duplicate_ids"],
)
def test_prefill_prep_routes_limit(monkeypatch, tokens, first, duplicate):
    """From hidden 2560 on the prefill path sorts with the streaming prep (one launch: sorting, the input
    quantization, loc_ids) instead of Opus moe_sorting and the row kernel while M * topk <= PREFILL_PREP_MAX_ROUTES,
    with Opus past it: the limit lowered to 400, H 2560, E 5 and topk 2 put M 200 / 201 (400 / 402 routes) on both
    sides; the prep gives each slot of a token that repeats an expert its own row, so the result equals the reference
    that sums every slot (the Opus front end keeps the later slot only, test_prefill_duplicate_ids_keep_last_route).

    The prefill cases of the basic tests and the benchmark are not on this boundary, assert no sorting kernel and
    never repeat an expert within a token; the real limit 24576 needs M 12288 at topk 2.
    """
    monkeypatch.setattr(moe_a8w4, "PREFILL_PREP_MAX_ROUTES", 400)
    w, prepared = _small_weights(2560, 64)
    x, topk_weight, topk_ids = make_inputs(
        tokens, 2560, 5, 2, seed=330 + tokens, shared=False
    )
    if duplicate:
        topk_ids[::2, 1] = topk_ids[::2, 0]
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    _assert_path([first] + _prefill_path(128))
    other = "moe_sorting_opus" if first == "a8w4_stream_prep" else "a8w4_stream_prep"
    assert other not in moe_a8w4.last_path, moe_a8w4.last_path
    assert_close(out, ref)


@pytest.mark.parametrize(
    "tokens, block_m, steps", [(40, 16, 4), (64, 32, 2)], ids=["m40_bm16", "m64_bm32"]
)
def test_stream_act_steps_bitexact(monkeypatch, tokens, block_m, steps):
    """The streaming Gate/Up of H 5120 (light waves: one gate and one up block each, 4 waves sharing the activations
    through an LDS ring) loads 16 bytes per lane covering ACT_STEPS 64-K steps (4 / 2 with blocks of 16 / 32, issued
    two groups ahead and written to LDS at the end of the previous group): the dispatch's tile for K 5120 has 4 / 2
    steps, and mid equals bit for bit that of the kernel with ACT_STEPS 1 (4 / 8 bytes per lane and step, a barrier
    every 2 steps) on the same sort and input. E 24 with M 40 / 64 are the M that the dispatch sorts in blocks of
    16 / 32; with blocks of 32 both one and two live sub-blocks occur (asserted below).

    The basic DeepSeek M 64 and the benchmark's ds_tp4 M48-1024 run this kernel but only grade end-to-end errors,
    which do not show a few differing elements.
    """
    compile_stream, seen = stream_a8w4.compile_a8w4_gateup_stream, {}

    def spy(**options):
        seen.update(options)
        return compile_stream(**options)

    monkeypatch.setattr(stream_a8w4, "compile_a8w4_gateup_stream", spy)
    _, prepared = _ds_weights()
    E, inter = prepared.experts, prepared.inter_padded
    assert moe_a8w4._stream_block_m(tokens, DS_TOPK, E, DS_HIDDEN) == block_m
    device, constants = torch.cuda.current_device(), moe_a8w4._stream_gateup_constants()
    args = (2 * inter, DS_HIDDEN, DS_TOPK, block_m, E, 0.0, device, constants)
    # Builds the dispatch's launcher uncached, only to read its tile (nothing is launched with it).
    moe_a8w4._stream_gateup_kernel.__wrapped__(*args)
    assert (seen["ACT_STEPS"], seen["W"], seen["G"]) == (steps, 4, 1), seen
    assert not seen.get("QUANT"), seen
    kernels = [
        moe_a8w4._stream_gateup_kernel(*args),
        compile_stream(**{**seen, "ACT_STEPS": 1}),
    ]
    x, topk_weight, topk_ids = make_inputs(
        tokens, DS_HIDDEN, E, DS_TOPK, seed=450 + tokens, shared=False
    )
    tail = torch.bincount(topk_ids.flatten().long(), minlength=E) % block_m
    assert block_m == 16 or (
        bool(((tail > 0) & (tail <= 16)).any()) and bool((tail > 16).any())
    ), tail
    sorted_ids, _, sorted_experts, num_valid, _, xq, xs = moe_a8w4._stream_prep(
        x, topk_weight, topk_ids, E, block_m, torch.empty_like(x), zero=False
    )
    mids = []
    for kernel in kernels:
        mids.append(torch.full((tokens, DS_TOPK, inter), float("nan")))
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
    assert torch.equal(mids[0].view(torch.int32), mids[1].view(torch.int32))


# case -> (M, tile options, set thresholds, kernels in order, the prefill kernels' (Gate/Up XCD_CHUNK, Down
# TWO_PASS, Down XCD_CHUNK) or None for no prefill kernel); DeepSeek H5120, E24, topk 6, I576, each threshold set to
# exactly one side of its boundary (M=64: 384 routes, 16 rows per expert; M=65: 390 routes, 16.25 rows; M=3: 18
# routes and 18 x 4 = 72 estimated Down workgroups; M=4: 24 routes and 96). The streaming cases also pin the
# H 5120 row limit itself to 16, so that they test the boundary logic and not the constant's current value.
_H5120_PREFILL_LIMITS = {
    "PREFILL_PREP_MAX_ROUTES_H5120": 18,
    "XCD_MIN_ROUTES_H5120": 24,
    "DOWN_TWO_PASS_MIN_WGS": 72,
}
_H5120_STREAM_BM32 = ["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"]
_H5120_LIMIT_CASES = {
    "rows16_stream": (
        64,
        {},
        {"STREAM_H5120_MIN_ROUTES": 384, "STREAM_MAX_ROWS_H5120": 16},
        _H5120_STREAM_BM32,
        None,
    ),
    "rows_over16_prefill": (
        65,
        {},
        {"STREAM_H5120_MIN_ROUTES": 390, "STREAM_MAX_ROWS_H5120": 16},
        ["a8w4_stream_prep"] + _prefill_path(128),
        (0, True, 0),
    ),
    "routes_below_stream": (
        65,
        {},
        {"STREAM_H5120_MIN_ROUTES": 391, "STREAM_MAX_ROWS_H5120": 16},
        _H5120_STREAM_BM32,
        None,
    ),
    "prep_one_pass": (
        3,
        {"block_m": 64, "tile_k": 128},
        _H5120_PREFILL_LIMITS,
        ["a8w4_stream_prep"] + _prefill_path(128),
        (0, False, 0),
    ),
    "opus_xcd_two_pass": (
        4,
        {"block_m": 64, "tile_k": 128},
        _H5120_PREFILL_LIMITS,
        ["moe_sorting_opus"] + _prefill_path(128),
        (moe_a8w4.GATEUP_XCD_CHUNK_H5120, True, moe_a8w4.DOWN_XCD_CHUNK_H5120),
    ),
}


@pytest.mark.parametrize("case", list(_H5120_LIMIT_CASES))
def test_h5120_dispatch_limits(monkeypatch, case):
    """The dispatch thresholds that only apply from hidden 5120 on, on both sides of each boundary, with the kernels,
    the build parameters of the prefill kernels and the end-to-end result: once M * topk reaches
    ``STREAM_H5120_MIN_ROUTES`` the streaming limit drops from ``STREAM_MAX_ROWS_WIDE`` (48) to
    ``STREAM_MAX_ROWS_H5120`` rows per expert, set to 16 here whatever its default (M 64 with exactly 16 rows still
    streams, M 65 with 16.25 takes prefill, and with the threshold one route above M 65 still streams, the limit
    48 again); prefill sorts with the streaming prep up to ``PREFILL_PREP_MAX_ROUTES_H5120`` routes and
    with Opus past it, Gate/Up and Down chunk their grids by XCD from ``XCD_MIN_ROUTES_H5120`` routes on, and Down
    stages its activation tile in two passes when the estimated workgroups (blocks x hidden split) exceed
    ``DOWN_TWO_PASS_MIN_WGS`` (M 3: 72, equal to it, still one pass; M 4: 96).

    The real thresholds (4096, 12288 and 16384 routes) need M >= 683 on E 24, where both sides are prefill, and the
    80 workgroups fall between M 3 and M 4, so they are moved onto the M 3-65 boundaries to keep this fast; the
    benchmark's ds_tp4 runs E 384 at powers of two and 48 without asserting kernels or build parameters; the basic
    tests are not on these boundaries.
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
        assert (gateup[-1][8], down[-1][6], down[-1][7]) == built, (gateup, down)
    assert_close(out, ref)


@pytest.mark.parametrize(
    "hidden, inter, tokens, options, path, shared, max_rows",
    [
        (4096, 64, 8, {}, _DECODE_PATHS["np"], None, None),
        (4608, 64, 8, {}, _prefill_path(256), None, None),
        (512, 1024, 4, {}, _DECODE_PATHS["np"], None, None),
        (512, 1088, 4, {}, _DECODE_PATHS["nn"], None, None),
        (512, 1024, 300, {"block_m": 64}, _prefill_path(256), None, None),
        (512, 1024, 48, {}, _stream_path(32, False), True, None),
        (512, 1024, 300, {}, _stream_path(64, False), False, 120),
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
    ranges give: the decode kernels take hidden <= 4096 (H 4096 with I 64, I_pad % 128 != 0: the per-route decode)
    and the measured H 5120 only, so hidden 4608 = 9 x 512 (an odd count of 512-column Gate/Up steps; the target
    shapes have 5, 8 and 10) falls back to prefill; the slot-pair Down needs a row's I_pad / 16 16-byte pieces in its
    64 * WAVES lanes (topk 2: one wave), so I 1088 takes the single-slot Down where I 1024 still pairs; I 1024 at
    M 300 (block_m=64 selects the prefill kernels, which pyhip M17 runs with the K tile 128 by default) fills Down's
    activation tile with exactly 64 KiB of LDS in one pass, the hidden dimension split in 4. The streaming Down lets
    4 waves share the block's activation rows only when they fit in the 64 KiB of LDS (``shared``): I 1024 fits with
    blocks of 32 (M 48) but not of 64 (M 300 with STREAM_MAX_ROWS raised to 120 rows per expert, as the default
    dispatch never streams blocks of 64 at H 512), where a wrongly shared tile fails when the kernel is built.

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
    """The single-slot and the mixed Down decode own 32 output channels per workgroup (decode_a8w4's row-block pairs:
    one 16-byte load per lane covers two 16-row blocks) and refuse a hidden size H % 32 != 0 when built
    (AssertionError, before compiling); the decode tier never asks for one, as it needs gateup_decode_fits
    (H % 512 == 0).

    Every H of the basic tests, the benchmark and the other cases is a multiple of 512.
    """
    decode = moe_a8w4.decode_a8w4
    for hidden in (DS_HIDDEN + 16, MIMO_HIDDEN + 48):
        assert not decode.gateup_decode_fits(hidden)
        with pytest.raises(AssertionError, match="unsupported Down decode shape"):
            decode.compile_a8w4_down_decode(N=hidden, K=DS_INTER, TOPK=DS_TOPK)
        with pytest.raises(AssertionError, match="unsupported mixed Down decode"):
            decode.compile_a8w4_down_decode_mixed(
                N=hidden, K=DS_INTER, KP=640, TOPK=DS_TOPK + 1, E=8
            )
    assert all(
        h % 32 == 0 for h in range(16, 16385, 16) if decode.gateup_decode_fits(h)
    )


# -- 2.5 CUDA graph -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, limit, path",
    [
        ({}, None, ["a8w4_stream_prep"] + _stream_path(32, True)),
        ({"block_m": 64}, None, ["a8w4_stream_prep"] + _prefill_path(256)),
        ({"block_m": 64}, 0, ["moe_sorting_opus"] + _prefill_path(256)),
    ],
    ids=["stream", "prefill", "prefill_opus"],
)
def test_graph_replay(monkeypatch, options, limit, path):
    """The whole call can be captured in a CUDA graph (no host synchronization) and replayed, and the replay reads
    newly written inputs (the workspaces allocated in the call belong to the graph's pool). The shape is compiled
    by an eager call outside the capture first. Qwen M 64 streams by default (the prep, blocks of 32, the Gate/Up
    quantizing the intermediate, the atomic Down); block_m=64 selects the prefill kernels (K tile 256) behind the
    streaming prep, or, with PREFILL_PREP_MAX_ROUTES lowered to ``limit`` = 0, behind Opus moe_sorting and the row
    kernel (pyhip M17 has the first two).

    The basic tests and the benchmark call eagerly and never capture a graph.
    """
    if limit is not None:
        monkeypatch.setattr(moe_a8w4, "PREFILL_PREP_MAX_ROUTES", limit)
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
    [(1, "nn"), (5, "np"), (16, "pp")],
    ids=["m1_nn", "m5_np", "m16_pp"],
)
def test_decode_graph_replay(tokens, path):
    """Each of the decode path's three dispatches can be captured in a CUDA graph and replayed, and the replay reads
    newly written inputs (xq / mid / mq allocated in the call belong to the graph's pool). The shape is compiled by an
    eager call outside the capture first.

    The basic tests and the benchmark call eagerly; test_graph_replay captures the streaming and prefill paths.
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


@pytest.mark.parametrize(
    "tokens, path",
    [(5, "dd2"), (24, "dd4"), (8, "wide")],
    ids=["m5_dd2", "m24_dd4", "ds_m8_wide"],
)
def test_decode_dedup_graph_replay(tokens, path):
    """Each form of the expert-deduplicated decode can be captured in a CUDA graph and replayed: the plan, the zeroing
    of the output and the atomic adds (H 5120: Gate/Up deduplicated only, the slot-pair Down writing the output) are
    all in captured kernels, so the replay plans the newly written routes again (plan / mid / xq allocated in the
    call belong to the graph's pool). The shape is compiled by an eager call outside the capture first.

    The basic tests and the benchmark call eagerly; test_decode_graph_replay captures the per-route decode only. The
    H 5120 case pins its kernels within the H 5120 decode limit (_assert_h5120).
    """
    if path == "wide":
        (w, prepared), hidden, experts, topk = (
            _ds_weights(),
            DS_HIDDEN,
            DS_EXPERTS,
            DS_TOPK,
        )
    else:
        (w, prepared), hidden, experts, topk = (
            _mimo_small_weights(256),
            MIMO_HIDDEN,
            12,
            MIMO_TOPK,
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
    if path == "wide":
        _assert_h5120(tokens, _DEDUP_PATHS[path])
    else:
        _assert_path(_DEDUP_PATHS[path])
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
    its inputs' device and the current constants. Building a launcher does not trace it, so the other device need
    not exist. The call passes block_m=64 to run the prefill launchers (pyhip M17 decodes M 16 by default).

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well.
    """
    constants = moe_a8w4._trace_constants()
    gateup = (512, 512, 2, 64, 128, 256, 5, 0.0, 0)
    down = (512, 64, 2, 64, 5, 1, False, 0)
    first = moe_a8w4._gateup_kernel(*gateup, 0, constants)
    assert moe_a8w4._gateup_kernel(*gateup, 0, constants) is first
    assert moe_a8w4._gateup_kernel(*gateup, 1, constants) is not first
    assert moe_a8w4._down_kernel(*down, 1, constants) is not moe_a8w4._down_kernel(
        *down, 0, constants
    )
    assert moe_a8w4._row_quant_kernel(
        512, "bf16", 2, 1
    ) is not moe_a8w4._row_quant_kernel(512, "bf16", 2, 0)
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
        name: [] for name in ("_gateup_kernel", "_down_kernel", "_row_quant_kernel")
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
    assert [args[-1] for args in calls["_row_quant_kernel"]] == [device] * 2


def test_decode_compile_cache_keys(monkeypatch):
    """The decode launcher caches are keyed by the device (``_decode_kernels``, ``_row_quant_decode``) and, for Down,
    by its wave count, which a call derives from DECODE_DOWN_PQ_WAVES (``_decode_waves``; the decode kernels read no
    other tunable module constant): another device or wave count gets its own launcher, and a call passes its inputs'
    device and the current wave count (topk 4: 2 waves by default, 1 with DECODE_DOWN_PQ_WAVES = 1, same result).
    Building a launcher does not trace it, so the other device need not exist.

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well; test_compile_cache_keys covers the prefill launchers.
    """
    decode = (128, 512, 64, 4, False, True, 0.0)
    first = moe_a8w4._decode_kernels(*decode, 2, 0)
    assert moe_a8w4._decode_kernels(*decode, 2, 0) is first
    assert moe_a8w4._decode_kernels(*decode, 2, 1) is not first
    assert moe_a8w4._decode_kernels(*decode, 1, 0) is not first
    assert moe_a8w4._row_quant_decode(64, False, 1) is not moe_a8w4._row_quant_decode(
        64, False, 0
    )
    built, quant = [], []
    _spy(monkeypatch, "_decode_kernels", built)
    _spy(monkeypatch, "_row_quant_decode", quant)
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(4, 512, 5, 4, seed=67, shared=False)
    for waves in (moe_a8w4.DECODE_DOWN_PQ_WAVES, 1):
        monkeypatch.setattr(moe_a8w4, "DECODE_DOWN_PQ_WAVES", waves)
        assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        _assert_path(_DECODE_PATHS["np"])
    device = x.device.index
    assert [args[-2:] for args in built] == [(2, device), (1, device)], built
    assert [args[-1] for args in quant] == [device] * 2


def test_decode_dedup_compile_cache_keys(monkeypatch):
    """The launcher caches of the expert-deduplicated decode are keyed by the device (``_gateup_plan_kernel``,
    ``_row_quant_plan``, ``_gateup_dd_kernel``, ``_down_dd_kernel`` and the H 5120 slot-pair ``_down_pq_kernel``),
    the plan Gate/Up also by its token limit, which a call takes from DECODE_MAX_M, and the slot-pair Down by its wave
    count, which a call derives from DECODE_DOWN_PQ_WAVES (as ``_decode_kernels``); their other build parameters are
    arguments already. Another device, limit or wave count gets its own launcher, and a call passes its inputs'
    device and the current constants (MiMo M 5 and M 12: the two- and the four-kernel form; DeepSeek M 8 with 3 and
    1 Down waves, same result, while H 5120 decodes M 8). Building a launcher does not trace it, so the other device
    need not exist.

    The basic tests and the benchmark run one GPU with the default constants, where a cache keyed by shape alone
    passes as well; test_decode_compile_cache_keys covers the per-route decode launchers.
    """
    plan = (512, 4096, 8, False, 2, 0.0)
    first = moe_a8w4._gateup_plan_kernel(*plan, 32, 0)
    assert moe_a8w4._gateup_plan_kernel(*plan, 32, 0) is first
    assert moe_a8w4._gateup_plan_kernel(*plan, 32, 1) is not first
    assert moe_a8w4._gateup_plan_kernel(*plan, 16, 0) is not first
    for name, args in (
        ("_row_quant_plan", (4096, 8, 2, 32, True)),
        ("_gateup_dd_kernel", (512, 4096, 8, 2, 8, 0.0)),
        ("_down_dd_kernel", (4096, 256, 8, 2, 2, True)),
        ("_down_pq_kernel", (5120, 576, 6, 3)),
    ):
        build = getattr(moe_a8w4, name)
        assert build(*args, 0) is build(*args, 0), name
        assert build(*args, 1) is not build(*args, 0), name
    assert moe_a8w4._down_pq_kernel(5120, 576, 6, 1, 0) is not moe_a8w4._down_pq_kernel(
        5120, 576, 6, 3, 0
    )
    names = (
        "_gateup_plan_kernel",
        "_row_quant_plan",
        "_gateup_dd_kernel",
        "_down_dd_kernel",
        "_down_pq_kernel",
    )
    calls = {name: [] for name in names}
    for name, record in calls.items():
        _spy(monkeypatch, name, record)
    w, prepared = _mimo_small_weights(256)
    for tokens, path in ((5, "dd2"), (12, "dd4")):
        x, topk_weight, topk_ids = make_inputs(
            tokens, MIMO_HIDDEN, 12, MIMO_TOPK, seed=215 + tokens, shared=False
        )
        assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        _assert_path(_DEDUP_PATHS[path])
    device = x.device.index
    assert [args[-2:] for args in calls["_gateup_plan_kernel"]] == [
        (moe_a8w4.DECODE_MAX_M, device)
    ]
    assert [
        args[-1]
        for args in calls["_row_quant_plan"]
        + calls["_gateup_dd_kernel"]
        + calls["_down_dd_kernel"]
    ] == [device] * 4
    w, prepared = _ds_weights()
    x, topk_weight, topk_ids = make_inputs(
        8, DS_HIDDEN, DS_EXPERTS, DS_TOPK, seed=228, shared=False
    )
    for waves in (moe_a8w4.DECODE_DOWN_PQ_WAVES, 1):
        monkeypatch.setattr(moe_a8w4, "DECODE_DOWN_PQ_WAVES", waves)
        assert_close(*_a8w4_run(w, prepared, x, topk_weight, topk_ids))
        _assert_h5120(8, _DEDUP_PATHS["wide"])
    if 8 <= moe_a8w4.DECODE_WIDE_MAX_M:
        built = [args[-2:] for args in calls["_down_pq_kernel"]]
        assert built == [(3, device), (1, device)], built


def test_stream_compile_cache_keys(monkeypatch):
    """The streaming launcher caches are keyed by the device (``_stream_gateup_kernel``, ``_stream_down_kernel``,
    ``_stream_prep_kernel``, ``_stream_sum_kernel``) and, for Gate/Up and Down, by the module constants they read
    while building (``_stream_gateup_constants()``: STREAM_GATEUP_ACT_BYTES_H5120 / PREFETCH / SYNC / W_AUX;
    ``_stream_down_constants()``: STREAM_DOWN_PREFETCH / W_AUX / W_AUX_H5120 / LDS_ACTS and common_a8w4.LDS_BYTES,
    which decides the shared tile): another device or constant gets its own launcher, and a call passes its inputs'
    device and the current constants (H 512, I 1024, M 48: the prep, the light-wave Gate/Up, the shared Down writing
    the routes and route_sum). Building a launcher does not trace it, so the other device need not exist.

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
        2560, 11, 33, 32, True, 4, 1
    ) is not moe_a8w4._stream_prep_kernel(2560, 11, 33, 32, True, 4, 0)
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
    _assert_path(["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"])
    device = x.device.index
    assert [args[-2:] for args in calls["_stream_gateup_kernel"]] == [(device, g0)]
    assert [args[-2:] for args in calls["_stream_down_kernel"]] == [(device, d0)]
    assert [
        args[-1] for args in calls["_stream_prep_kernel"] + calls["_stream_sum_kernel"]
    ] == [device] * 2


# -- 2.7 routing and output variants: repeated expert ids, in-place output, input dtypes, M = 0 -------


@pytest.mark.parametrize(
    "case", ["dedup_m5", "dedup_m24", "wide_m24", "stream_atomic", "stream_route_sum"]
)
def test_duplicate_ids_keep_every_route(case):
    """Several slots of a token select the same expert on the expert-deduplicated decode (two and four kernels: the
    routes of one token may land in one group; H 5120 deduplicates Gate/Up only, and the two slots of one slot pair
    of the Down may be one expert) and on the streaming path (the prep gives every route its own sorted row; Qwen
    M 64 with I 320 adds them atomically, H 512 with I 1024 sums them in route_sum): every slot is computed once and
    the slots are summed, as the reference does.

    The generated routes never repeat an expert within a token, so the basic tests and the benchmark never get here;
    test_decode_duplicate_and_int64_ids covers the per-route decode, test_prefill_prep_routes_limit the prefill
    sorted by the prep (every slot summed too), test_prefill_duplicate_ids_keep_last_route the Opus-sorted prefill.
    The H 5120 case pins its kernels within the H 5120 decode limit and the streaming kernels past it
    (_assert_h5120).
    """
    shared = False
    if case.startswith("dedup"):
        (w, prepared), hidden, experts, topk = (
            _mimo_small_weights(256),
            MIMO_HIDDEN,
            12,
            MIMO_TOPK,
        )
        tokens, path = (5, "dd2") if case == "dedup_m5" else (24, "dd4")
    elif case == "wide_m24":
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
        path = ["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"]
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, experts, topk, seed=240 + tokens, shared=shared
    )
    topk_ids[::2, 1] = topk_ids[::2, 0]
    if topk > 2:
        topk_ids[::3, topk - 2] = topk_ids[::3, 0]
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids)
    if path == "wide":
        _assert_h5120(tokens, _DEDUP_PATHS[path])
    elif isinstance(path, list):
        _assert_path(path)
    else:
        _assert_path(_DEDUP_PATHS[path])
    assert_close(out, ref)


def test_prefill_duplicate_ids_keep_last_route():
    """Two slots of a token select the same expert: the Opus sort keeps only the later slot, the earlier slot's loc
    keeps the -1 of the input quantization and sorted_sum skips it (no uninitialized row read, nothing added
    twice); the result equals the reference with the earlier slot's routing weight set to 0. block_m=64 selects
    the prefill kernels (by default M 200 takes the K tile 128, as in pyhip M17); H 512 keeps the Opus sort (from
    H 2560 on the streaming prep sorts and keeps every slot, test_prefill_prep_routes_limit).

    The generated routes never repeat an expert within a token, so the basic tests and the benchmark never get
    here.
    """
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(200, 512, 5, 2, seed=320, shared=False)
    topk_ids[::2, 1] = topk_ids[::2, 0]
    out, _ = _a8w4_run(w, prepared, x, topk_weight, topk_ids, block_m=64)
    _assert_path(["moe_sorting_opus"] + _prefill_path(256))
    kept = topk_weight.clone()
    kept[::2, 0] = 0
    ref = reference(x, w["w1"], w["w1_scale"], w["w2"], w["w2_scale"], kept, topk_ids)
    assert_close(out, ref)


@pytest.mark.parametrize("case", ["dedup", "wide", "stream", "prefill"])
def test_inplace_output(case):
    """``output`` is ``hidden_states`` itself: the op reads all of x before it writes the output; the result equals
    the reference of the original input. The expert-deduplicated decode (MiMo M 5) would take two kernels, whose
    per-route Gate/Up zeroes the output while other workgroups still read x, so it takes the four kernels, whose plan
    kernel reads each row of x before zeroing it; H 5120 (DeepSeek M 8) only quantizes x in the plan kernel and the
    slot-pair Down writes the output last (pinned within the H 5120 decode limit, past it the streaming kernels,
    _assert_h5120); the streaming prep (Qwen M 64) zeroes each piece of an output row in the lane that has read
    that piece of x, before the atomic Down adds into it; on the prefill path (block_m=64) the front end reads x
    first and sorted_sum writes last.

    The basic tests and the benchmark pass separate output buffers.
    """
    options = {}
    if case == "dedup":
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
    if case == "dedup":
        _assert_path(_DEDUP_PATHS["dd4"])
    elif case == "wide":
        _assert_h5120(x.shape[0], _DEDUP_PATHS["wide"])
    elif case == "stream":
        _assert_path(["a8w4_stream_prep"] + _stream_path(32, True))
    else:
        _assert_path(["a8w4_stream_prep"] + _prefill_path(256))
    assert_close(x, ref)


def test_routing_input_dtypes():
    """int64 and non-contiguous ``topk_ids`` and a non-contiguous BF16 ``topk_weight`` (as a router may produce
    them) are converted to the contiguous int32 / float32 the sort and the kernels read; without the output
    argument the op allocates the output. The entry converts once for every path; M 40 (H 512, E 5) streams, so the
    streaming prep sorts the converted routing.

    The basic tests and the benchmark pass the contiguous int32 / float32 routing of ``make_inputs`` and an output
    buffer; test_decode_duplicate_and_int64_ids converts on the decode path.
    """
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
# topk 8, I 256; ds = DeepSeek H 5120, E 24, topk 6, I 576.
_SWIGLU_CASES = {
    "decode_nn": ("qwen", 3, {}, _DECODE_PATHS["nn"]),
    "decode_pq": ("qwen", 12, {}, _DECODE_PATHS["pp"]),
    "decode_plan": ("mimo", 5, {}, _DEDUP_PATHS["dd2"]),
    "decode_dd": ("mimo", 24, {}, _DEDUP_PATHS["dd4"]),
    "stream_fused_quant": (
        "qwen",
        64,
        {},
        ["a8w4_stream_prep"] + _stream_path(32, True),
    ),
    "stream_bm16": (
        "ds",
        40,
        {},
        ["a8w4_stream_prep"] + _stream_path(16, False) + ["a8w4_route_sum"],
    ),
    "stream_bm64": (
        "ds",
        160,
        {},
        [
            "a8w4_stream_prep",
            "a8w4_gateup_stream_bm64",
            "a8w4_down_prefill_bm64",
            "a8w4_route_sum",
        ],
    ),
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
    epilogue): the decode quantizing x in the kernel (nn) or reading it pre-quantized (pq), the per-route Gate/Up
    that also writes the dedup plan (MiMo M 5) and the deduplicated Gate/Up (MiMo M 24: groups of 3, 4 waves), the
    streaming Gate/Up quantizing the intermediate in its epilogue (Qwen M 64, blocks of 32), the light-wave streaming
    Gate/Up with blocks of 16 (DeepSeek M 40, ACT_STEPS 4) and 64 (M 160), the prefill Gate/Up with the K tile 256
    (block_m=64), and the H 5120 decode quantizing in the kernel (M = 3: 10 bf16 pairs per thread over K = 5120);
    inputs scaled by 10 so that more than 10% of the gate and of the up values exceed the limit, graded against the
    reference with the limit.

    The benchmark's DeepSeek cases have N(0, 1) inputs that almost never exceed 10 (they measure the cost of the
    limit without checking it), the Qwen / MiMo cases pass no limit. The basic DeepSeek cases clamp in the streaming
    Gate/Up with blocks of 32 (M 64) and in the deduplicated Gate/Up with groups of 2 while H 5120 decodes M 8;
    test_swiglu_limit_tile_k128 covers the 128-K prefill tile.
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
        _assert_path(
            ["a8w4_stream_prep"] + _stream_path(16, False) + ["a8w4_route_sum"]
        )
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
    between two direct calls): the deduplicated decode Down, or the streaming Down without a8w4_route_sum.
    """
    return "a8w4_down_decode_dd" in path or (
        any(name.startswith("a8w4_down_stream") for name in path)
        and "a8w4_route_sum" not in path
    )


# (shape, tokens, tier): one case per tier and output form, at token counts that select it with the cut E (asserted):
# the per-route decode (TP4 M 8, with the I 160 -> 192 padding that the tag must carry), the deduplicated decode with
# its atomic Down (MiMo TP8 M 8) and with H 5120's slot-pair Down (DeepSeek M 8), streaming with the atomic Down
# (TP2 M 48) and with a8w4_route_sum (DeepSeek M 64), prefill (TP2 M 300).
_FUSED_CASES = [
    ("tp4", 8, "decode"),
    ("mimo_tp8", 8, "dedup"),
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
    passes every call on unchanged, so the variants inside a tier (decode nn / np / pp, dedup two / four kernels,
    sorting blocks) and the boundaries between tiers are left to the flydsl_moe_a8w4_ptpc cases above.

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


# -- 2.10 shared expert: prepare's conversions, real checkpoints, errors, stream mode, graph, in-place, TP8, fused_moe
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


@pytest.mark.parametrize(
    "tokens",
    [moe_a8w4.SHARED_DUAL_STREAM_MAX_M, moe_a8w4.SHARED_DUAL_STREAM_MAX_M + 1],
    ids=["side_stream_last", "serial_first"],
)
def test_layer_stream_mode_boundary(monkeypatch, tokens):
    """ATOM's split switches its stream mode exactly past SHARED_DUAL_STREAM_MAX_M tokens (ATOM's dual-stream
    threshold): at it the split runs on the side stream, one token more on the current stream before the routed
    experts; both graded against reference_layer (E 8, topk 6, I 576: the routed experts take the prefill tier on
    both sides, so only the stream mode differs). In both modes the output has the bits of the routed experts alone
    plus the split's output added as ATOM adds it (addcmul_ with the weight in BF16, one rounding). A case about the
    split itself (_assert_split): SHARED_FOLD = 0 (the op's switch: the split wherever the mixed decode does not run)
    pins it, so the case holds whatever token counts the op folds the shared expert at; the other layer cases do not
    depend on how the shared expert runs.

    The basic and graph cases are far from the threshold and do not assert the mode; neither does the benchmark. The
    layer cases grade the sum within the reference's tolerance, which another rounding of the add could pass.
    """
    monkeypatch.setattr(moe_a8w4, "SHARED_FOLD", 0)
    prepared, x, topk_weight, topk_ids, out = _layer_check(8, 6, 576, tokens, seed=780)
    _assert_kernels("_prefill")
    _assert_split(tokens)
    routed = flydsl_moe_a8w4_ptpc(
        x,
        _routed_only(prepared),
        topk_weight[:, :-1],
        topk_ids[:, :-1],
        swiglu_limit=LAYER_SWIGLU_LIMIT,
    )
    shared = moe_a8w4._shared_split(x, prepared, LAYER_SWIGLU_LIMIT)
    gate = shared_route_weight(topk_weight, topk_ids, prepared.experts)
    routed.addcmul_(shared, gate.to(torch.bfloat16)[:, None])
    assert torch.equal(_bits(out), _bits(routed))


def test_layer_split_inplace_serial(monkeypatch):
    """With output = hidden_states ATOM's split runs first on the current stream even at SHARED_DUAL_STREAM_MAX_M
    tokens, where a separate output puts it on the side stream: it reads the input before the routed experts overwrite
    it (no fork, its Down GEMM before the first A8W4 kernel), and the output has the bits of the separate one (E 8,
    topk 6, I 576, M 1024 as in test_layer_stream_mode_boundary, which grades it: the routed experts take the prefill
    tier, no atomics). A case about the split itself (_assert_split), pinned by SHARED_FOLD = 0 as there.

    test_layer_inplace_output and test_layer_fold_tiers compare bits only, independent of how the op runs the shared
    expert; a side stream reading the input while the routed kernels overwrite it would seldom change them. The
    benchmark passes a separate output.
    """
    monkeypatch.setattr(moe_a8w4, "SHARED_FOLD", 0)
    tokens = moe_a8w4.SHARED_DUAL_STREAM_MAX_M
    _, _, prepared = _layer_weights(8, 576, 780)
    x, topk_weight, topk_ids = make_layer_inputs(
        tokens, LAYER_HIDDEN, 8, 6, seed=787, shared_weight=None
    )
    want = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    _assert_split(tokens)
    result = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, output=x, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert result is x
    _assert_split(tokens, inplace=True)
    assert torch.equal(_bits(x), _bits(want))


@pytest.mark.parametrize("tokens", [1, 4, 2048], ids=["m1", "m4", "m2048"])
def test_layer_graph_replay(tokens):
    """A layer call with the shared expert can be captured in a CUDA graph and replayed: the replay reads newly written
    inputs and gives the bits of an eager call on them (main layer TP4: decode M 1 and 4, prefill M 2048, all without
    atomics). Compiled by an eager call outside the capture first. Independent of how the op runs the shared expert
    (_assert_shared): today the mixed decode kernels (M 1), the fold into the slot-pair decode Down (M 4, past
    DECODE_SHARED_MIXED_MAX_WORK) or into sorted_sum (M 2048), on the capturing stream; later milestones may run it
    otherwise at some of these token counts.

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

    The basic tests pass a separate output; test_layer_split_inplace_serial asserts the split's order in place.
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


def test_layer_tp8_padding_exact():
    """TP8's shared I 288, zero-padded to the routed experts' 320 by prepare, changes no bit of ATOM's split: the op's
    split (moe_a8w4._shared_split) on prepare's padded shared fields gives the bits of aiter's dense chain on the
    unpadded shared expert (gate / up [576, 5120], Down [5120, 288] with K padded to 384 by shuffle_weight), M 4. The
    split's own function is called, not the op, so that the case does not depend on the token counts at which the op
    runs the split (later milestones run the shared expert otherwise at some of them; the layer cases grade that).

    The basic tests and the benchmark grade TP8 within the reference's tolerance, which a small padding error could
    pass.
    """
    from aiter.ops.shuffle import shuffle_weight

    experts, topk, inter = LAYER_SHAPES["ds_layer_tp8"]
    _, shared, prepared = _layer_weights(experts, inter, 700)
    assert prepared.shared_w1.shape[0] == 2 * prepared.inter_padded == 640
    x, _, _ = make_layer_inputs(
        4, LAYER_HIDDEN, experts, topk, seed=770, shared_weight=None
    )
    got = moe_a8w4._shared_split(x, prepared, LAYER_SWIGLU_LIMIT)
    w1, s1, w2, s2 = shared
    fnuz = [
        q.view(torch.uint8)
        .masked_fill(q.view(torch.uint8) == 0x80, 0)
        .view(torch.float8_e4m3fnuz)
        for q in (w1, w2)
    ]
    unpadded = aiter_shared_split(
        x,
        shuffle_weight(fnuz[0], layout=(16, 16)),
        s1 * 2,
        shuffle_weight(fnuz[1], layout=(16, 16), pad_k_to=128),
        s2 * 2,
        LAYER_SWIGLU_LIMIT,
    )
    assert torch.equal(_bits(got), _bits(unpadded))


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
            _DECODE_PATHS["pp"],
            (3, 11),
            id="pp_qwen_m16",
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
            (12, MIMO_HIDDEN, 256, 456),
            5,
            MIMO_TOPK,
            False,
            {},
            _DEDUP_PATHS["dd2"],
            (2,),
            id="dd2_mimo_m5",
        ),
        pytest.param(
            (12, MIMO_HIDDEN, 512, 712),
            24,
            MIMO_TOPK,
            False,
            {},
            _DEDUP_PATHS["dd4"],
            (5, 6),
            id="dd4_mimo_m24",
        ),
        pytest.param(
            (12, MIMO_HIDDEN, 512, 712),
            24,
            MIMO_TOPK,
            False,
            {},
            _DEDUP_PATHS["dd4"],
            "all",
            id="dd4_all",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            64,
            DS_TOPK,
            False,
            {},
            ["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"],
            (7, 40),
            id="stream_ds_m64",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            64,
            DS_TOPK,
            False,
            {},
            ["a8w4_stream_prep"] + _stream_path(32, False) + ["a8w4_route_sum"],
            "all",
            id="stream_all",
        ),
        pytest.param(
            (MIMO_EXPERTS, MIMO_HIDDEN, 512, 41),
            1200,
            MIMO_TOPK,
            False,
            {},
            ["a8w4_stream_prep", "a8w4_gateup_stream_bm64"]
            + ["a8w4_down_prefill_bm64", "a8w4_route_sum"],
            (9, 600),
            id="stream_prefill_down_mimo_m1200",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            200,
            DS_TOPK,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256) + ["sorted_sum"],
            (4, 150),
            id="prefill_prep_ds_m200",
        ),
        pytest.param(
            (DS_EXPERTS, DS_HIDDEN, DS_INTER, 400),
            200,
            DS_TOPK,
            False,
            {"block_m": 64},
            ["a8w4_stream_prep"] + _prefill_path(256) + ["sorted_sum"],
            "all",
            id="prefill_prep_all",
        ),
        pytest.param(
            (5, 512, 64, 576),
            300,
            2,
            False,
            {"block_m": 64},
            ["moe_sorting_opus"] + _prefill_path(256) + ["sorted_sum"],
            (8, 299),
            id="prefill_opus_h512",
        ),
        pytest.param(
            (5, 512, 64, 576),
            300,
            2,
            False,
            {"block_m": 64},
            ["moe_sorting_opus"] + _prefill_path(256) + ["sorted_sum"],
            "all",
            id="prefill_opus_all",
        ),
    ],
)
def test_masked_routes(weights, tokens, topk, fused, options, path, full):
    """Masked routes on every tier but the basic test's (routed experts only): a random 30% of the ids and the case's
    rows whole, on poisoned weights with NaN weights on the masked routes; graded, the rows without a live route
    bitwise 0, the kernels asserted. The per-route decode Gate/Up and Down (nn) and the pre-quantized Gate/Up (pp) with
    the slot-pair Down; the dedup plan and its forms (H 5120 wide, the MiMo two- and four-kernel ones; groups of 2 and
    3); streaming with route_sum behind the streaming Down and behind the prefill Down (blocks of 64, H 4096: it writes
    loc_ids only for sorted routes); prefill sorted by the prep and by Opus (moe_sorting, which skips ids outside [0, E);
    the input quantization sets loc_ids to -1, which sorted_sum skips). The ``_all`` cases mask every route: the dedup
    plan has no leader, the sortings no row (num_valid 0), and the output is all zero.

    The basic test masks the per-route decode (np) only; the benchmark never masks.
    """
    experts, hidden, inter, seed = weights
    w, prepared = poisoned(*_weights(experts, hidden, inter, seed))
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, prepared.experts, topk, seed=seed + 3, shared=fused
    )
    topk_weight, topk_ids = mask_routes(topk_weight, topk_ids, seed=seed + 4, full=full)
    out, ref = _a8w4_run(w, prepared, x, topk_weight, topk_ids, **options)
    _assert_path(path)
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
    sorted_sum's last column). A random 30% of the routed ids, rows 1 and 5 whole and row 2's first id masked: ATOM's
    rule drops the shared expert of every row whose first id is -1, so rows 1 and 5 are bitwise 0. Poisoned weights,
    graded against reference_layer.

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
    masked, and each replay gives the bits of an eager call on the same inputs (all zero when every row is masked).
    Poisoned weights.

    The other graph cases replay unmasked routing; the masked eager cases capture nothing.
    """
    experts, topk, inter = LAYER_SHAPES["ds_layer_tp4"]
    w, _, prepared = _layer_weights(experts, inter, 700)
    _, prepared = poisoned(w, prepared)
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
        assert torch.equal(_bits(out), _bits(want)), (seed, full)
        dead = (ids[:, :-1] < 0).all(dim=1).nonzero().flatten().tolist()
        assert _zero_rows(out, dead)
        assert set(range(tokens) if full == "all" else full) <= set(dead)
        assert full or not dead


@pytest.mark.parametrize("case", ["ds_m3", "mimo_m5_plan"])
def test_masked_gateup_reads_no_weights(case):
    """The per-route decode Gate/Up reads no weights of a masked route: launched on poisoned weights (expert -1, where
    a masked route would read, is real data), every masked route's f32 row is exactly 0 and every other row has the bits
    of the launch without masks; the form that writes the dedup plan (dd2) also gives each masked route the entry
    [0, route] and still zeroes the output. DeepSeek M 3 (x quantized in the kernel, the SwiGLU limit), MiMo M 5 (PLAN).

    End to end a read would not show (the Down kernels drop masked routes by a select, the dedup Down never sees them).
    """
    if case == "ds_m3":
        (w, prepared), hidden, topk, tokens, limit = (
            _ds_weights(),
            DS_HIDDEN,
            DS_TOPK,
            3,
            SWIGLU_LIMIT,
        )
    else:
        (w, prepared), hidden, topk, tokens, limit = (
            _mimo_small_weights(256),
            MIMO_HIDDEN,
            MIMO_TOPK,
            5,
            0.0,
        )
    _, view = poisoned(w, prepared)
    x, topk_weight, topk_ids = make_inputs(
        tokens, hidden, view.experts, topk, seed=850, shared=False
    )
    _, masked = mask_routes(topk_weight, topk_ids, seed=851, full=(1,))
    I, device = view.inter_padded, x.device.index
    if case == "ds_m3":
        waves = moe_a8w4._decode_waves(topk, False)
        kernel = moe_a8w4._decode_kernels(
            2 * I, hidden, I, topk, False, False, limit, waves, device
        )[0]
    else:
        kernel = moe_a8w4._gateup_plan_kernel(
            2 * I, hidden, topk, False, 2, limit, moe_a8w4.DECODE_MAX_M, device
        )

    def gateup(ids):
        mid = torch.full((tokens, topk, I), float("nan"))
        plan = torch.full((tokens * topk, 4), -7, dtype=torch.int32)
        y = torch.full((tokens, hidden), float("nan"), dtype=torch.bfloat16)
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
            plan,
            y,
            tokens,
        )
        return mid, plan, y

    mid0 = gateup(topk_ids)[0]
    mid1, plan, y = gateup(masked)
    live = masked >= 0
    assert 0 < int((~live).sum()) < live.numel()
    assert bool((mid1[~live] == 0).all())
    assert torch.equal(mid1[live].view(torch.int32), mid0[live].view(torch.int32))
    if case == "mimo_m5_plan":
        entries = plan.tolist()
        for r in (~live).flatten().nonzero().flatten().tolist():
            assert entries[r][:2] == [0, r], (r, entries[r])
        assert bool((y == 0).all())


def test_entry_checks(monkeypatch):
    """Errors raised before any kernel runs (ValueError; last_path empty, the output untouched):
    - padded I above what the prefill Down stages in LDS (block_m * I_pad <= 64 KiB): I 1088 at H 512 for the prefill
      tier (M 300, many rows per expert) and with block_m 64 (M 40), at H 4096 for streaming blocks of 64 (M 100, 40
      rows per expert), which take the prefill Down, and with a shared expert (H 5120, block_m 64: no shared-expert
      kernel either); block_m 128 with I 576 (DeepSeek). Streaming with the streaming Down (H 512, M 48, blocks of 32)
      still runs I 1088 (graded; decode in test_shape_fallbacks).
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

    for hidden, tokens, options in (
        (512, 300, {}),
        (512, 40, {"block_m": 64}),
        (4096, 100, {}),
    ):
        _, prepared = _small_weights(hidden, 1088)
        x, topk_weight, topk_ids = make_inputs(tokens, hidden, 5, 2, seed=860)
        rejects(prepared, x, topk_weight, topk_ids, "padded I 1088", **options)
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
# Up to DECODE_SHARED_MIXED_MAX_M tokens, when the routed experts take the per-route decode (and, where the fold is
# possible as well, from DECODE_PREQUANT_MIN_M tokens on, while M * routed topk * I_pad stays within
# DECODE_SHARED_MIXED_MAX_WORK: main TP4 M 4 folds), a layer call runs the shared expert inside the two decode kernels
# (decode_a8w4's mixed Gate/Up and Down; moe_a8w4.last_path then ends with "shared_mixed"). Like the per-route decode
# kernels quantizing in the kernel, they take one IEEE reciprocal per row and correct each product with FMAs
# (_quant_fast), which must give the FP8 bytes of the IEEE division; rows whose abs-max is outside _fast_div_ok
# divide. The basic tests run the mixed decode at main TP4 M 1 and TP8 M 4 (main TP4 M 4, masked, now folds), the
# benchmark at every layer shape at M 1 and 2 and all but main TP4 at M 4 (graded; shared weight 1, the clamp seldom
# hit), test_layer_graph_replay and test_layer_inplace_output at main TP4 M 1; none of them asserts which kernels run
# the shared expert. Here: the thresholds (tokens, work), the FP8 bytes of the fast quantization at the edges of its
# range and at rounding ties, a shared expert dropped by its last id, the reads of the mixed Gate/Up, and shapes
# beyond the layers'.

_MIXED_PATH = ["a8w4_gateup_decode_mixed", "a8w4_down_decode_mixed", "shared_mixed"]


@pytest.mark.parametrize(
    "tokens",
    [moe_a8w4.DECODE_SHARED_MIXED_MAX_M, moe_a8w4.DECODE_SHARED_MIXED_MAX_M + 1],
    ids=["mixed_last", "past_mixed"],
)
def test_layer_mixed_decode_boundary(tokens):
    """The mixed decode ends exactly past DECODE_SHARED_MIXED_MAX_M tokens: at it a layer call runs exactly the mixed
    Gate/Up and Down, one token more runs the shared expert another way (the fold today) and no mixed kernel. MTP
    layer TP4 (the official block format), inputs x 10 (the shared expert's clamp hit), a random shared weight per
    token; both sides graded against reference_layer. Only the threshold is pinned, not what takes over past it.

    The basic tests and the benchmark accept any path for the shared expert; the benchmark has no M 5 and runs the MTP
    layers with shared weight 1 and the clamp seldom hit.
    """
    experts, topk, inter = LAYER_SHAPES["mtp_layer_tp4"]
    _layer_check(experts, topk, inter, tokens, seed=700, fmt="block", amplify=10.0)
    path = moe_a8w4.last_path
    if tokens <= moe_a8w4.DECODE_SHARED_MIXED_MAX_M:
        assert path == _MIXED_PATH, path
    else:
        assert not any("_mixed" in name for name in path), path
        _assert_shared()


# case: (layer shape, M, DECODE_SHARED_MIXED_MAX_WORK or None for the default, mixed). Work = M * routed topk * I_pad:
# main TP4 M 4 13824, main TP8 M 4 7680; main TP4 M 3 10368, where the fold is not possible (single-slot Down).
_MIXED_WORK_CASES = {
    "ds_tp4_m4_fold": ("ds_layer_tp4", 4, None, False),
    "ds_tp4_m4_at_limit": ("ds_layer_tp4", 4, 4 * 6 * 576, True),
    "ds_tp4_m4_over_limit": ("ds_layer_tp4", 4, 4 * 6 * 576 - 1, False),
    "ds_tp4_m3_no_fold": ("ds_layer_tp4", 3, None, True),
    "ds_tp8_m4_mixed": ("ds_layer_tp8", 4, None, True),
}


@pytest.mark.parametrize("case", list(_MIXED_WORK_CASES))
def test_layer_mixed_decode_work_limit(monkeypatch, case):
    """Where the fold is possible as well (moe_a8w4._shared_fold_tier: the slot-pair decode Down, from
    DECODE_PREQUANT_MIN_M tokens on) the mixed decode yields to it once M * routed topk * I_pad exceeds
    DECODE_SHARED_MIXED_MAX_WORK: with the default, main TP4 M 4 (13824) folds while main TP8 M 4 (7680) mixes, and
    main TP4 M 3 (10368, past the limit) mixes, having no fold; with the limit set to main TP4 M 4's work it mixes, one
    below it folds. Graded against reference_layer (inputs x 10, the clamp hit), the mixed kernels asserted or, for the
    fold, the routed decode tier without a final add (_assert_folded). The MTP layers at M 4 (6912 / 3840) mix in
    test_layer_mixed_decode_boundary and test_layer_mixed_decode_masked.

    The basic tests and the benchmark accept any path for the shared expert; the benchmark has no M 3.
    """
    shape, tokens, limit, mixed = _MIXED_WORK_CASES[case]
    if limit is not None:
        monkeypatch.setattr(moe_a8w4, "DECODE_SHARED_MIXED_MAX_WORK", limit)
    experts, topk, inter = LAYER_SHAPES[shape]
    _layer_check(experts, topk, inter, tokens, seed=700, amplify=10.0)
    if mixed:
        assert moe_a8w4.last_path == _MIXED_PATH, moe_a8w4.last_path
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
    assert moe_a8w4.last_path == _MIXED_PATH, moe_a8w4.last_path
    assert_close(
        out,
        reference_layer(x, w, shared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT),
    )
    assert _zero_rows(out, [1])
    weights, ids = mask_routes(weights, ids, seed=792, full="all", routed=topk)
    out = flydsl_moe_a8w4_ptpc(
        x, prepared, weights, ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    assert moe_a8w4.last_path == _MIXED_PATH, moe_a8w4.last_path
    assert _zero_rows(out, range(4))


def test_mixed_decode_quant_bitexact():
    """The decode kernels that quantize their rows themselves (decode_a8w4._quant_fast: one reciprocal per row, FMA
    corrections per value) give the FP8 bytes of the IEEE division (_ieee_quant) on _fast_quant_rows (abs-max on both
    sides of both ends of the fast range, zero and subnormal rows, rounding ties), shown on the outputs, which equal
    those of launches on the reference's bytes: the per-route Gate/Up quantizing x (one correction) and the mixed
    Gate/Up's routed rows have the bits of the per-route Gate/Up reading x pre-quantized by _ieee_quant, one launch per
    x row with the routed channel factors times 2**k (so that tiny or huge rows give normal f32 results); the
    single-slot Down and the mixed Down quantizing the intermediate (two corrections) have the bits of the single-slot
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
    waves = moe_a8w4._decode_waves(topk, False)
    gateup, down = moe_a8w4._decode_kernels(
        2 * I, H, I, topk, False, False, limit, waves, device
    )
    gateup_pq = moe_a8w4._decode_kernels(
        2 * I, H, I, topk, True, False, limit, waves, device
    )[0]
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
            got,
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
            want,
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
    "experts, topk, hidden, inter, mixed",
    [(16, 15, 2560, 192, True), (8, 1, 512, 64, True), (20, 16, 2560, 192, False)],
    ids=["topk15_h2560", "topk1_h512", "topk16_not_mixed"],
)
def test_mixed_decode_shapes(experts, topk, hidden, inter, mixed):
    """The mixed decode on shapes beyond the layers' (M 3, a random shared weight per token, inputs x 10, SwiGLU limit
    10): topk 15 + the shared expert (16 Down waves, the most it takes) at H 2560 (5 x pairs per Gate/Up thread, the
    last one packed alone) and I 192 (the shared Down's K padded to 256); topk 1 + the shared expert (2 waves) at H 512
    (one x pair per thread) and I 64; topk 16 + the shared expert (17 columns) does not fit the mixed Down and takes
    another path. Graded against reference_layer, the mixed kernels asserted or ruled out.

    The basic tests and the benchmark run the mixed decode at H 5120, topk 3 or 6 + 1 and I 320 or 576 only.
    """
    w = make_weights(experts, hidden, inter, seed=800 + topk)
    shared = make_shared_weights(hidden, inter, seed=801 + topk)["row"]
    prepared = _prepare_shared(w, shared)
    x, topk_weight, topk_ids = make_layer_inputs(
        3, hidden, experts, topk, seed=802 + topk, shared_weight=None
    )
    x = (x.float() * 10.0).to(torch.bfloat16)
    out = flydsl_moe_a8w4_ptpc(
        x, prepared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
    )
    path = list(moe_a8w4.last_path)
    assert_close(
        out,
        reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        ),
    )
    if mixed:
        assert path == _MIXED_PATH, path
    else:
        assert not any("_mixed" in name for name in path), path
        assert any(name.startswith("shared") for name in path), path


# -- 2.13 shared fold: the shared expert in the routed experts' last reduction --------------------------------------
# Past the mixed decode, where the routed tier allows it (moe_a8w4._shared_fold_tier: the decode tier's slot-pair Down,
# streaming, prefill from SHARED_FOLD_PREFILL_MIN_M tokens; an output 16-byte aligned where the shared expert writes
# it), a layer call folds the shared expert into the routed experts' last reduction: two FP8 GEMMs (aiter's FlyDSL
# preshuffle GEMM) on the routed path's quantized input with shared_a8w4's mid kernel between them write the output
# (the decode slot-pair Down then adds the routes in f32, the streaming atomic Down in BF16) or M more rows of the
# routes buffer that route_sum / sorted_sum add as one more column; on a side stream up to SHARED_FOLD_DUAL_STREAM_MAX_M
# tokens. The basic layer tests (streaming M 64, prefill M 2048, wide decode M 8 and 16) and the benchmark's layer
# cases (every folded tier: M 6 the pre-quantized decode, mtp_layer_tp4 M 1280 the streaming blocks of 64; unmasked
# routing, the last column E with weight 1) grade it without asserting how the shared expert runs. Here: every folded
# tier with masks and with rows whose last column is a routed expert instead of E, in place and under graph replay; the
# GEMM tiles' LDS race; the fallback of an output that is not 16-byte aligned; the compile-cache keys.

# case: (shape, tokens, options, routed tier as _path_tier names it, bit-deterministic). Shape: a LAYER_SHAPES name, or
# "small" (_fold_small_weights: E 8, H 512, I 64, routed topk 2; prefill with a tile option sorts with Opus below
# H 2560). The tiers reached today: the pre-quantized per-route Gate/Up with the slot-pair Down (M 6), the wide decode
# (deduplicated Gate/Up, M 16), streaming with route_sum behind the streaming Down (blocks of 16) and behind the
# prefill Down (blocks of 64, which also writes loc_ids), streaming with the atomic Down (I 320), prefill sorted by
# the prep and by Opus. The atomic Down's order of adds changes from run to run: that case is graded, not compared.
_FOLD_CASES = {
    "decode_pq_ds_tp8_m6": ("ds_layer_tp8", 6, {}, "decode", True),
    "wide_dd_mtp_tp8_m16": ("mtp_layer_tp8", 16, {}, "dedup", True),
    "stream_sum_mtp_tp4_m128": ("mtp_layer_tp4", 128, {}, "stream", True),
    "stream_prefill_down_mtp_tp4_m1280": ("mtp_layer_tp4", 1280, {}, "stream", True),
    "stream_atomic_mtp_tp8_m64": ("mtp_layer_tp8", 64, {}, "stream", False),
    "prefill_prep_mtp_tp8_m2048": ("mtp_layer_tp8", 2048, {}, "prefill", True),
    "prefill_opus_small_m2048": ("small", 2048, {"block_m": 64}, "prefill", True),
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
    """The last call ran the routed experts on ``tier`` (_path_tier) and the shared expert without a final add: what
    ran it, the fold today, wrote where the routed experts' last reduction adds it."""
    path = moe_a8w4.last_path
    assert _path_tier(path) == tier, (tier, path)
    _assert_shared()
    assert "shared_add" not in path, path


@pytest.mark.parametrize("case", list(_FOLD_CASES))
def test_layer_fold_tiers(case):
    """Every folded tier (_FOLD_CASES) on poisoned routed weights with _fold_routing's masks and dropped shared
    experts: graded against reference_layer, the rows without a live routed id bitwise 0 (row 1: the shared expert's
    +0 plus nothing), no final add (_assert_folded; the streaming case of blocks of 64 also asserts the prefill Down,
    the Opus case Opus). The same call with output = hidden_states gives the bits of the separate output (the chain
    writes the output, or the routes buffer, only after the kernel that quantizes the input), and a CUDA graph of the
    call replays other inputs with the masks moved with the bits of an eager call on them (the atomic case: graded).

    The basic tests and the benchmark grade the folded tiers on unmasked routing with E in the last column, eagerly
    (the benchmark replays graphs but grades one replay within the tolerance) into a separate output; neither reaches
    the pre-quantized decode of M 5-7 or the streaming blocks of 64, nor asserts how the shared expert runs.
    """
    shape, tokens, options, tier, exact = _FOLD_CASES[case]
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
    if case.startswith("stream_prefill_down"):
        assert any(n.startswith("a8w4_down_prefill") for n in moe_a8w4.last_path)
    if case.startswith("prefill_opus"):
        assert "moe_sorting_opus" in moe_a8w4.last_path, moe_a8w4.last_path
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
    decode: the shared Down GEMM of 32 rows on the side stream beside the routed Gate/Up) replays 1000 times with the
    bits of the eager call, which is graded (four waves with tile (16, 64, 128) there: 14-20 of 1000 replays wrong).

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
    ["decode_pq_ds_tp8_m6", "stream_atomic_mtp_tp8_m64", "stream_sum_mtp_tp4_m128"],
)
def test_layer_fold_unaligned_output(case):
    """An output 8 bytes past a 16-byte boundary (a view into a larger buffer): where the folded shared expert would
    write the output with its Down GEMM's 16-byte stores (the slot-pair decode, the streaming atomic Down) the call
    takes ATOM's split (final add); with route_sum the shared expert writes the routes buffer, the output is aligned
    or not. Graded against reference_layer; the buffer before the view stays untouched.

    The basic tests and the benchmark pass outputs from torch.empty, always aligned.
    """
    shape, tokens, options, tier, _ = _FOLD_CASES[case]
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
    path = moe_a8w4.last_path
    assert _path_tier(path) == tier, path
    _assert_shared()
    if not case.startswith("stream_sum"):
        assert path[-1] == "shared_add", path
    assert_close(
        out,
        reference_layer(
            x, w, shared, topk_weight, topk_ids, swiglu_limit=LAYER_SWIGLU_LIMIT
        ),
    )
    assert bool(buf[:4].isnan().all())


def test_layer_fold_compile_cache_keys(monkeypatch):
    """The fold's launcher caches: ``_shared_mid_kernel`` is keyed by the device, the routing columns, E, the SwiGLU
    limit and whether it writes loc_ids; the routed launchers' fold variants (routing tensors of topk + 1 columns, the
    slot-pair Down adding the output it finds, the prep never sorting the last column, the prefill Down's loc_ids of
    topk + 1 columns) are cached apart from the plain ones; and a call passes its inputs' device and whether it writes
    loc_ids (the small shape's pre-quantized decode M 6: no; its Opus prefill M 2048: yes). Building a launcher does not
    trace it, so the other device need not exist.

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
            (1152, 5120, 576, 6, True, True, 10.0, 3, 0),
            "shared",
            True,
        ),
        ("_gateup_dd_kernel", (1152, 5120, 6, 2, 4, 10.0, 0), "shared", True),
        ("_row_quant_plan", (5120, 6, 2, 32, False, 0), "shared", True),
        ("_down_pq_kernel", (5120, 576, 6, 3, 0), "shared", True),
        ("_stream_prep_kernel", (5120, 7, 128, 16, False, 4, 0), "shared", True),
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
    for tokens, options in ((6, {}), (2048, {"block_m": 64})):
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


# =================================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# 158 cases, named <shape>_m<M> with M = 1, 2, 4, ..., 32768 and 48 (the layer cases also M 6, and 1280 for
# mtp_layer_tp4; see BENCH_LAYER_EXTRA_CASES); SiLU, BF16 in and out. First the 85 cases of
# moe-a8w4 (unchanged: names, seeds, timing, output), 5 shapes:
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
# Then 73 whole-layer cases of DeepSeek-V4.1-Flash with its shared expert (bench_moe_a8w4_layer): LAYER_SHAPES (main
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
# The layer cases, after the 85 above; the order gives each case's input seed.
BENCH_LAYER_CASES = [
    f"{shape}_m{m}" for shape in LAYER_SHAPES for m in BENCH_TOKENS + BENCH_EXTRA_TOKENS
] + BENCH_LAYER_EXTRA_CASES
# --quick (bench_quick): the layer shapes at these token counts only.
BENCH_QUICK_TOKENS = (1, 4, 16, 64, 512, 4096)
# bench_graph_us: at most this many argument sets (run_perftest's num_iters) and at least this many timed replays.
BENCH_GRAPH_MAX_SETS = 101
BENCH_GRAPH_MIN_REPLAYS = 5
# a8w4_split runs the shared expert on a side stream up to this many tokens (ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD).
BENCH_SPLIT_DUAL_STREAM_MAX_M = 1024


def bench_time_us(fn, args):
    """Device time of one ``fn(*args)`` call in microseconds, the protocol of every timed candidate.

    The warm-up count is the fewest calls taking longer than BENCH_WARMUP_MS (at least 2), each call timed with
    CUDA events as the fastest of 3 bursts of 8 back-to-back calls, each burst behind one leading call (it keeps
    the launch latency after the synchronization out; the first also compiles). Then
    ``run_perftest(fn, *args, num_warmup=..., **BENCH_PERFTEST_KW)``, repeated after BENCH_RETRY_S pauses while
    the profiler returns no time. Uses only the standard library, torch, ``run_perftest`` and the BENCH_*
    constants, so another process can time another implementation with the same protocol.
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
    for name, (fn, args, nbytes) in candidates.items():
        us = bench_time_us(fn, args)
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
    kernel_sum = bench_time_us(a8w4, candidates["a8w4"][1])
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
        | {int(c.rsplit("_m", 1)[1]) for c in BENCH_LAYER_EXTRA_CASES}
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        choices=tokens,
        default=None,
        help="token counts M (default 1, 2, 4, ..., 32768, 48; the layer shapes also 6, mtp_layer_tp4 also 1280; "
        "a shape runs only the counts of its cases)",
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


if __name__ == "__main__":
    main()
