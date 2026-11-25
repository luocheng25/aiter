# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark of the gfx942 FlyDSL A8W4 MoE (``flydsl_moe_a8w4_ptpc``): BF16 activations quantized
per token to FP8 E4M3FNUZ, MXFP4 weights (E2M1 + one E8M0 per 32 K), SiLU, BF16 output.

One file, four regions:
  0. shared: the independent fp32 reference (a port of pyhip's ``testing/moe_a8w4.py``: weights and inputs drawn
     with the same generators and seeds as pyhip's benchmark, exact MXFP4 dequantization, per-row FP8
     activations) and the grading (``calc_diff <= 1e-3`` and every element within 4.5% of ``max|ref|``);
  1. basic functionality (``@pytest.mark.basic``): one case per target shape and token-count tier, the first
     thing to run after every change: ``pytest op_tests/test_flydsl_moe_a8w4.py -m basic``;
  2. other coverage: what regions 1 and 3 cannot cover (weight layout and delta flush, bit-exact row
     quantization, dedup plan, streaming prep and fused quantization, decode dispatch boundaries (per route and
     expert-deduplicated), streaming boundaries and variants, the prefill sorting front end, prefill tile options and
     the half-block / XCD / two-pass / Down-split variants, CUDA graph, input errors and the gfx942 gate, compile-cache
     keys, repeated expert ids, in-place output, routing input dtypes, M = 0, the SwiGLU limit):
     ``pytest op_tests/test_flydsl_moe_a8w4.py -m "not basic"``;
  3. benchmark (``main``, not collected by pytest): Qwen3.8 TP2 / TP4 (against aiter's FP8 per-token
     ``fused_moe``), MiMo-V2.6-Flash TP8 / TP4 and DeepSeek-V4.1-Flash TP4 x M = 1, 2, 4, ..., 32768 and 48,
     85 cases: ``python op_tests/test_flydsl_moe_a8w4.py``.
``MF_SEED`` seeds the benchmark's weights and inputs.
"""

import argparse
import functools
import itertools
import math
import os
import shutil
import tempfile
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
    SiLU(gate) * up; 0 (default) means no limit and gives the same bits as not passing it.
    """
    M, H = x.shape
    E, n1 = w1.shape[0], w1.shape[1]
    I, topk = n1 // 2, topk_ids.shape[1]
    xa = quant_rows_fp8(x)
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E).tolist()
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
    "cols, topk, rows, max_m, zero",
    [(MIMO_HIDDEN, MIMO_TOPK, 37, 64, True), (DS_HIDDEN, DS_TOPK, 29, 32, False)],
    ids=["h4096_zero_out", "h5120_keep_out"],
)
def test_decode_plan_quant_bitexact(cols, topk, rows, max_m, zero):
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
    full = partial = 0
    for r, e in enumerate(ids):
        same = [s for s, f in enumerate(ids) if f == e]
        rank = same.index(r)
        count = min(group, len(same) - rank) if rank % group == 0 else 0
        entry = entries[r]
        assert entry[:2] == [count, r], (r, entry)
        assert entry[1 : 1 + count] == same[rank : rank + count], (r, entry)
        full += count == group
        partial += 0 < count < group
    assert full and partial


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


def test_decode_down_fallback_w2_4gib():
    """With w2 of 4 GiB or more Down does not take the slot-pair kernel on pre-quantized rows (it addresses all
    experts' weights with 32-bit byte offsets) but the single-slot kernel quantizing in the kernel (M = 16).

    Real 4 GiB weights are too large: a stride-0 view makes w2 report 2**32 bytes with the same data pointer (the
    kernels read by pointer and expert offset); the basic tests, the benchmark and the other cases have w2 far below
    4 GiB.
    """
    import dataclasses

    w, prepared = _decode_weights(320)
    big = dataclasses.replace(prepared, w2=prepared.w2.view(-1)[:1].expand(1 << 32))
    x, topk_weight, topk_ids = make_inputs(16, HIDDEN, EXPERTS, TOPK, seed=99)
    out, ref = _a8w4_run(w, big, x, topk_weight, topk_ids)
    _assert_path(["a8w4_gateup_decode_pq", "a8w4_down_decode"])
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


# case -> (M, tile options, lowered thresholds, kernels in order, the prefill kernels' (Gate/Up XCD_CHUNK, Down
# TWO_PASS, Down XCD_CHUNK) or None for no prefill kernel); DeepSeek H5120, E24, topk 6, I576, each threshold set to
# exactly one side of its boundary (M=64: 384 routes, 16 rows per expert; M=65: 390 routes; M=3: 18 routes and
# 18 x 4 = 72 estimated Down workgroups; M=4: 24 routes and 96).
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
        {"STREAM_H5120_MIN_ROUTES": 384},
        _H5120_STREAM_BM32,
        None,
    ),
    "rows_over16_prefill": (
        65,
        {},
        {"STREAM_H5120_MIN_ROUTES": 390},
        ["a8w4_stream_prep"] + _prefill_path(128),
        (0, True, 0),
    ),
    "routes_below_stream": (
        65,
        {},
        {"STREAM_H5120_MIN_ROUTES": 391},
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
    ``STREAM_H5120_MIN_ROUTES`` the streaming limit drops from 48 to ``STREAM_MAX_ROWS_H5120`` = 16 rows per expert
    (M 64 with exactly 16 rows still streams, M 65 with 16.25 takes prefill, and with the threshold one route above
    M 65 still streams); prefill sorts with the streaming prep up to ``PREFILL_PREP_MAX_ROUTES_H5120`` routes and
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
    """Off gfx942 both public entries raise RuntimeError before any kernel runs (FP8 E4M3FNUZ, the MFMA tiles and
    the MI308X tuning are gfx942-only).

    The basic tests and the benchmark run on gfx942 only.
    """
    w, prepared = _small_weights(512, 64)
    x, topk_weight, topk_ids = make_inputs(4, 512, 5, 2, seed=64, shared=False)
    monkeypatch.setattr(moe_a8w4, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942"):
        prepare_moe_a8w4_ptpc_weight(w["w1"], w["w1_scale"], w["w2"], w["w2_scale"])
    with pytest.raises(RuntimeError, match="gfx942"):
        flydsl_moe_a8w4_ptpc(x, prepared, topk_weight, topk_ids)


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


# =================================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# 85 cases = 5 shapes x M = 1, 2, 4, ..., 32768 and M = 48, named <shape>_m<M>; SiLU, BF16 in and out:
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
# experts' weights and scales, the input read and the output write), and an INFO line with the launched kernels;
# at the end the markdown table.
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
        choices=list(BENCH_SHAPES),
        default=list(BENCH_SHAPES),
        help="shapes (name -> H, E, topk, I, padded I, fused shared expert, aiter reference):\n"
        + "\n".join(f"  {name}: {dims}" for name, dims in BENCH_SHAPES.items()),
    )
    tokens = sorted(BENCH_TOKENS + BENCH_EXTRA_TOKENS)
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        choices=tokens,
        default=tokens,
        help="token counts M (default 1, 2, 4, ..., 32768, 48)",
    )
    args = parser.parse_args()
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
            for shape, m in itertools.product(args.shape, args.m)
        ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl_moe_a8w4_ptpc summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
