# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests and benchmark of ``MegaMoeTPGfx942`` (aiter/ops/flydsl/mega_moe_tp_gfx942.py), gfx942, ``comm_mode="ar"``
and, in the DP section, ``comm_mode="ag_rs"``.

One file, three regions (the DP mode's tests in their own section after region 2, its 15 benchmark cases after the
51 AR cases):

1. basic tests (``pytest <file> -m basic``): one case per dispatch tier and shape; run first after every change.
2. other coverage (``pytest <file> -m "not basic"``): what the basic tests and the benchmark cannot cover; every
   test says why.
3. benchmark (``python <file> [--cases ...]``, after the last test): the SPEC cases {tp2, tp4, ds_layer_tp4} x M in
   {1, 2, 4, ..., 32768} + {48}; cold-cache protocol, slowest rank; prints ``MF_METRIC`` / ``MF_REF``.

Shapes: tp2 / tp4 are Qwen3.8-Flash-Next (513 MXFP4 experts, the last one its fused shared expert in the last of 11
slots); ds_layer_tp4 is a DeepSeek-V4.1-Flash layer (384 MXFP4 experts, topk 6, and the FP8 shared expert as expert
E in a 7th routing column, SwiGLU limit 10); mtp_layer_tp4 is its MTP layer (128 experts, topk 3 + the shared
expert; other coverage only); ds_dp4 is the ds_layer_tp4 layer in ``comm_mode="ag_rs"`` (DP4 attention + TP4 MoE:
every rank passes its own M_r tokens and gets their output; see the DP section).

Every rank of a TP group gets the same input rows and routing (generated on rank 0 and broadcast; asserted bitwise
identical) and its own weight slice (independent random weights per rank). The reference is independent FP32 torch:
per rank the exact MXFP4 / FP8 dequantization, FP8 per-token activations and FP8 per-(token, slot) intermediate
activations (scale = amax / 240), the shared expert weighted by ATOM's rule, summed over the ranks of the group.
Acceptance: finite, calc_diff <= 1e-3 and max|y - ref| <= 4.5% max|ref|; the outputs of all ranks are bitwise
identical.

Multi-GPU runs go through a pool of 4 spawned worker processes (rank r on GPU r, world group + TP2 sub-group),
started once per session; a test sends a job (``_job_<name>``) to every rank and checks the returned results.
"""

import argparse
import importlib
import importlib.abc
import importlib.util
import itertools
import json
import multiprocessing
import os
import queue
import re
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

SEED = int(os.environ.get("MF_SEED", "0"))
SHAPES = {
    # world, hidden, routed experts, routed topk, real per-rank inter_dim, routing ("qwen": make_inputs, "layer":
    # make_layer_inputs), FP8 shared expert (expert E in a last routing column), swiglu_limit
    "tp2": {
        "world": 2,
        "H": 2560,
        "E": 513,
        "topk": 11,
        "I": 320,
        "routing": "qwen",
        "shared": False,
        "limit": None,
    },  # Qwen3.8-Flash-Next TP2
    "tp4": {
        "world": 4,
        "H": 2560,
        "E": 513,
        "topk": 11,
        "I": 160,
        "routing": "qwen",
        "shared": False,
        "limit": None,
    },  # Qwen3.8-Flash-Next TP4
    "ds_layer_tp4": {
        "world": 4,
        "H": 5120,
        "E": 384,
        "topk": 6,
        "I": 576,
        "routing": "layer",
        "shared": True,
        "limit": 10.0,
    },  # DeepSeek-V4.1-Flash TP4 (E385 / topk 7 with the shared expert)
    "mtp_layer_tp4": {
        "world": 4,
        "H": 5120,
        "E": 128,
        "topk": 3,
        "I": 576,
        "routing": "layer",
        "shared": True,
        "limit": 10.0,
    },  # DeepSeek-V4.1-Flash MTP layer TP4 (other coverage only)
    "ds_dp4": {
        "world": 4,
        "H": 5120,
        "E": 384,
        "topk": 6,
        "I": 576,
        "routing": "layer",
        "shared": True,
        "limit": 10.0,
        "mode": "ag_rs",
    },  # DeepSeek-V4.1-Flash DP4 + TP4: comm_mode ag_rs, M_r tokens per rank (max_local_tokens: M_r)
}
SHAPE_INDEX = {name: i for i, name in enumerate(SHAPES)}
SHARED_ARGS = ("shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale")
MAX_TOKENS = 1 << 15
POOL_WORLD = 4

FP8_MAX = 240.0
FP8_OCP_MAX = 448.0
CALC_DIFF_MAX = 1e-3
MAX_ERR_RATIO = 0.045
LAYER_ROUTED_SCALING = 1.5
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
CA, RCCL = "custom_all_reduce", "rccl_all_reduce"

START_TIMEOUT, JOB_TIMEOUT, GRACE = 300, 600, 30


def parse_case(name):
    shape, m = name.rsplit("_m", 1)
    if shape not in SHAPES:
        raise ValueError(f"unknown shape in case {name!r}")
    return shape, int(m)


def weight_seed(shape, rank, variant=0):
    return SEED * 1000 + 100 * SHAPE_INDEX[shape] + rank + 50 * variant


def input_seed(shape, M, g):
    return SEED * 1000 + 500 + 37 * SHAPE_INDEX[shape] + 4 * M + g


# Path checks. A forward's path (layer.last_path joined by ">") holds the single-GPU op's kernels (moe_a8w4.last_path:
# with a shared expert also its "shared_*" steps or "*_shared" kernels) and then the communication. The tests pin the
# dispatch tier -- the first Gate/Up kernel (a8w4_gateup*) --, whether a shared expert ran, and exactly one
# communication segment: one all-reduce at the end (CA / RCCL), or fused communication kernels and no all-reduce (the
# fused cross-rank sums of later milestones: the word "tp" in their "_"-separated names, e.g. a8w4_down_decode_tp,
# a8w4_tp_reduce, sorted_sum_tp). Not pinned: the kernels after Gate/Up and which all-reduce an M takes (fusing the
# communication changes them; 2.2 tests the all-reduce rule on the communicator itself).
GATEUP_PREFIX, TP_WORD, SHARED_WORD, FUSED_COMM = "a8w4_gateup", "tp", "shared", "fused"
# Every Gate/Up kernel name of moe_a8w4: decode (plan / pre-quantized / deduplicated / mixed with the shared expert),
# streaming (block_m; quant: the intermediate quantized inside) and prefill (block_m x block_n x tile_k).
GATEUP_KERNEL = re.compile(
    r"a8w4_gateup_(decode(_plan)?(_pq|_dd|_mixed)?|(quant_)?stream_bm\d+|prefill_bm\d+_bn\d+_tk\d+)"
)


def gateup_tier(path):
    """The dispatch tier of a path: its first Gate/Up kernel (None without one)."""
    return next((k for k in path.split(">") if k.startswith(GATEUP_PREFIX)), None)


def comm_kind(path):
    """The communication segment of a path: CA / RCCL (one all-reduce, last, and no fused communication kernel),
    FUSED_COMM (fused communication kernels and no all-reduce), None otherwise (none, both, several,
    misplaced)."""
    kernels = path.split(">")
    ars = [k for k in kernels if k in (CA, RCCL)]
    if any(TP_WORD in k.split("_") for k in kernels):
        return None if ars else FUSED_COMM
    return ars[0] if ars == kernels[-1:] else None


def has_shared(path):
    """Whether a shared expert ran: a step or kernel with the word "shared" in its "_"-separated name."""
    return any(SHARED_WORD in k.split("_") for k in path.split(">"))


def _known_path(path):
    """A path of this implementation: a known Gate/Up kernel (tier) and exactly one communication segment."""
    tier = gateup_tier(path)
    return (
        tier is not None
        and GATEUP_KERNEL.fullmatch(tier) is not None
        and comm_kind(path) is not None
    )


# =====================================================================================================================
# Weights, inputs and the independent FP32 reference (torch only, no kernel under test)
# =====================================================================================================================
def make_weights(experts, hidden, inter, *, seed, device="cuda"):
    """Random BF16 MoE weights (generated expert by expert), quantized to standard MXFP4.

    Returns ``w1`` float4_e2m1fn_x2 [E, 2I, H/2] (rows [0, I) gate, [I, 2I) up), ``w1_scale`` float8_e8m0fnu
    [E, 2I, H/32], ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32]. Standard deviations: w1 1/sqrt(H), w2 1/sqrt(I).
    """
    import aiter

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
    for e in range(experts):
        v1 = torch.randn((2 * i, h), generator=gen, device=device) * h**-0.5
        v2 = torch.randn((h, i), generator=gen, device=device) * i**-0.5
        for name, value in (("w1", v1), ("w2", v2)):
            q, s = mx(value.to(torch.bfloat16), quant_dtype=torch.float4_e2m1fn_x2)
            out[name][e].copy_(q.view(torch.uint8))
            out[f"{name}_scale"][e].copy_(
                s.view(torch.uint8).reshape(out[f"{name}_scale"][e].shape)
            )
    for name in ("w1", "w2"):
        out[name] = out[name].view(torch.float4_e2m1fn_x2)
        out[f"{name}_scale"] = out[f"{name}_scale"].view(torch.float8_e8m0fnu)
    return out


def to_megamoe_layout(w):
    """Standard MXFP4 weights -> the ``MegaMoeTP`` layout: ``shuffle_weight(16, 16)`` and ``e8m0_shuffle`` of the
    [E * rows, groups] scales."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    out = {}
    for name in ("w1", "w2"):
        s = w[f"{name}_scale"]
        out[name] = shuffle_weight(w[name], layout=(16, 16))
        out[f"{name}_scale"] = fp4_utils.e8m0_shuffle(
            s.reshape(s.shape[0] * s.shape[1], s.shape[2])
        )
    return out


def _fp8_rows(v):
    """float32 [N, K] -> E4M3FN [N, K] and float32 [N, 1] scales amax / 448 per row (1 for a zero row), RNE: the
    per-row format of FP4PTPC checkpoints."""
    amax = v.abs().amax(dim=1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_OCP_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (v / scale).to(torch.float8_e4m3fn), scale


def make_shared_weights(hidden, inter, *, seed, device="cuda"):
    """One rank's slice of a random shared expert: ``(shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)``,
    float8_e4m3fn [2I, H] (rows [0, I) gate, [I, 2I) up) and [H, I] with float32 [N, 1] per-row scales (``_fp8_rows``)
    of BF16 weights with standard deviations 1/sqrt(H) (w1) and 1/sqrt(I) (w2)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    v1 = torch.randn((2 * inter, hidden), generator=gen, device=device) * hidden**-0.5
    v2 = torch.randn((hidden, inter), generator=gen, device=device) * inter**-0.5
    v1, v2 = v1.to(torch.bfloat16).float(), v2.to(torch.bfloat16).float()
    return (*_fp8_rows(v1), *_fp8_rows(v2))


def shared_kwargs(shared):
    """The ``shared_*`` keyword arguments of a shared expert tuple (none for None)."""
    return {} if shared is None else dict(zip(SHARED_ARGS, shared))


def make_inputs(tokens, hidden, experts, topk, *, seed, device="cuda"):
    """Qwen3.8 routing: ``(x, topk_weight, topk_ids)``, x BF16 [M, H] ~ N(0, 1), routing [M, topk].

    The last expert (E-1) is the fused shared expert, always in a token's last slot; the other topk-1 are uniform
    without repetition in [0, E-1), weighted by the softmax of topk-1 random logits, the shared one by sigmoid(randn).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, hidden), generator=gen, device=device).to(torch.bfloat16)
    routed = (
        torch.rand((tokens, experts - 1), generator=gen, device=device)
        .topk(topk - 1, dim=-1)
        .indices
    )
    last = torch.full((tokens, 1), experts - 1, dtype=routed.dtype, device=device)
    ids = torch.cat([routed, last], dim=1).to(torch.int32).contiguous()
    logits = torch.randn((tokens, topk - 1), generator=gen, device=device)
    shared_w = torch.sigmoid(torch.randn((tokens, 1), generator=gen, device=device))
    weights = (
        torch.cat([torch.softmax(logits, dim=-1), shared_w], dim=1).float().contiguous()
    )
    return x, weights, ids


def make_layer_inputs(tokens, hidden, experts, topk, *, seed, device="cuda"):
    """DeepSeek-V4.1 routing with a shared expert: ``(x, topk_weight, topk_ids)``, x BF16 [M, H] ~ N(0, 1), routing
    [M, topk + 1] (ids int32, weights float32): topk distinct uniform routed experts in [0, E) weighted by the softmax
    of topk random logits times LAYER_ROUTED_SCALING, then the shared expert E with weight 1 (ATOM's convention).
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
    ids = torch.cat([ids, torch.full_like(ids[:, :1], experts)], dim=1)
    weights = torch.cat([routed, torch.ones_like(routed[:, :1])], dim=1)
    return x, weights.float().contiguous(), ids.to(torch.int32).contiguous()


def dequant_mxfp4(q, scale):
    """Standard MXFP4 [..., N, K/2] + E8M0 [..., N, K/32] -> exact float32 [..., N, K]."""
    b = q.view(torch.uint8)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).long()
    value = torch.tensor(_E2M1, dtype=torch.float32, device=b.device)[nib]
    factor = torch.exp2(scale.view(torch.uint8).to(torch.float64) - 127).float()
    return (value.unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)).flatten(-2)


def dequant_fp8(w, scale):
    """FP8 [N, K] (E4M3FN or E4M3FNUZ) with float32 per-row scales ([N] / [N, 1]) or E8M0 scales of 32x32 blocks ->
    the exact float32 [N, K]."""
    v = w.float()
    if scale.dtype == torch.float32:
        return v * scale.reshape(-1, 1)
    factor = torch.exp2(scale.view(torch.uint8).double() - 127).float()
    return v * factor.repeat_interleave(32, 0).repeat_interleave(32, 1)


def quant_rows_fp8(t):
    """Quantize each row (last dim) to FP8 E4M3FNUZ and back: scale = amax / 240, q = RNE(x / scale), both IEEE
    f32 divisions (an all-zero row takes scale 1). Divides by a same-shape tensor: ``tensor / python scalar``
    becomes a multiplication by the reciprocal (up to 1 ulp off)."""
    t = t.float()
    amax = t.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (t / scale).to(torch.float8_e4m3fnuz).float() * scale


def reference(
    x, w1, w1_scale, w2, w2_scale, topk_weight, topk_ids, *, swiglu_limit=0.0
):
    """Independent FP32 MoE of one rank's routed experts -> float32 [M, H].

    Standard MXFP4 weights (``make_weights``), FP8 per-token input and FP8 per-(token, slot) intermediate,
    SiLU(gate) * up, routing weights applied after Down and summed per token. ``swiglu_limit`` L > 0 is the
    DeepSeek-V4.1 SwiGLU clamp on the f32 gate / up: gate = min(gate, L), up = clamp(up, -L, L). An id of -1 masks
    its route.
    """
    M, H = x.shape
    E, n1 = w1.shape[0], w1.shape[1]
    I, topk = n1 // 2, topk_ids.shape[1]  # noqa: E741,RUF100
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


def reference_shared(x, w1, w1_scale, w2, w2_scale, *, swiglu_limit=0.0):
    """Independent FP32 shared expert of one rank's slice -> float32 [M, H] before its routing weight: the input
    quantized per row to FP8 (``quant_rows_fp8``), the exactly dequantized FP8 weights, the SwiGLU limit as in
    ``reference``, SiLU(gate) * up quantized per row, Down."""
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


def reference_rank(shape, raw, shared, x, topk_weight, topk_ids):
    """This rank's FP32 share of the layer's output (summed over the group by the caller): ``reference`` of its routed
    experts on the first topk routing columns, plus ``reference_shared`` of its shared expert slice times
    ``shared_route_weight``."""
    s = SHAPES[shape]
    k, limit = s["topk"], s["limit"] or 0.0
    out = reference(
        x,
        raw["w1"],
        raw["w1_scale"],
        raw["w2"],
        raw["w2_scale"],
        topk_weight[:, :k],
        topk_ids[:, :k],
        swiglu_limit=limit,
    )
    if shared is not None:
        weight = shared_route_weight(topk_weight, topk_ids, s["E"])
        out += reference_shared(x, *shared, swiglu_limit=limit) * weight.unsqueeze(-1)
    return out


def calc_diff(x, y):
    """1 - 2 sum(xy) / sum(x^2 + y^2) in float64; 0 when both are all zero."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def check(out, ref):
    """Acceptance: finite, calc_diff <= 1e-3 and max|out - ref| <= 4.5% max|ref|."""
    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return {
            "ok": False,
            "reason": "output contains NaN/Inf",
            "calc_diff": float("inf"),
            "err_ratio": float("inf"),
        }
    from aiter.test_common import checkAllclose

    diff = calc_diff(o, r)
    err = (o - r).abs().max().item() if o.numel() else 0.0
    top = r.abs().max().item() if r.numel() else 0.0
    ratio = err / top if top else (0.0 if err == 0 else float("inf"))
    # |out - ref| <= 4.5% max|ref| per element, i.e. the err_ratio bound, through Aiter's checker (mismatch fraction)
    mismatch = (
        checkAllclose(o / top, r / top, rtol=0.0, atol=MAX_ERR_RATIO, printLog=False)
        if top
        else 0.0
    )
    return {
        "ok": diff <= CALC_DIFF_MAX and ratio <= MAX_ERR_RATIO and mismatch == 0,
        "calc_diff": diff,
        "max_err": err,
        "max_ref": top,
        "err_ratio": ratio,
    }


def config_args(shape, rank, **kw):
    """The ``MegaMoeTPConfig`` fields of a shape (``kw`` overrides them)."""
    s = SHAPES[shape]
    args = {
        "rank": rank,
        "world_size": s["world"],
        "model_dim": s["H"],
        "inter_dim": s["I"],
        "experts": s["E"],
        "topk": s["topk"],
        "max_local_tokens": MAX_TOKENS // s["world"],
        "comm_mode": s.get("mode", "ar"),
        "swiglu_limit": s["limit"],
    }
    args.update(kw)
    return args


def make_config(shape, rank, **kw):
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPConfig

    return MegaMoeTPConfig(**config_args(shape, rank, **kw))


# =====================================================================================================================
# Worker pool: POOL_WORLD processes, rank r on GPU r, kept for the session
# =====================================================================================================================
def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Pool:
    """``run(job, **kw)`` runs ``_job_<job>(state, **kw)`` on every rank and returns the per-rank results. A failed
    job (exception, dead or silent rank) kills the pool and raises with every rank's traceback; the next ``run``
    starts a new pool."""

    def __init__(self, world=POOL_WORLD):
        self.world, self.procs, self.gfx, self.seq = world, [], None, 0

    def start(self):
        if self.procs:
            return self.gfx
        ctx = multiprocessing.get_context("spawn")
        self.tasks = [ctx.Queue() for _ in range(self.world)]
        self.results = ctx.Queue()
        port = _free_port()
        self.procs = [
            ctx.Process(
                target=_worker_main,
                name=f"mega_moe_tp_gfx942_rank{r}",
                args=(r, self.world, port, self.tasks[r], self.results, os.getpid()),
            )
            for r in range(self.world)
        ]
        for p in self.procs:
            p.start()
        self.gfx = self._collect("ready", START_TIMEOUT)[0]
        return self.gfx

    def run(self, job, timeout=JOB_TIMEOUT, **kw):
        self.start()
        self.seq += 1
        for q in self.tasks:
            q.put((self.seq, job, kw))
        return self._collect(self.seq, timeout)

    def _collect(self, seq, timeout):
        got, errors = {}, {}
        deadline = time.monotonic() + timeout
        while len(got) + len(errors) < self.world and time.monotonic() < deadline:
            try:
                s, rank, status, payload = self.results.get(timeout=1.0)
            except queue.Empty:
                if any(not p.is_alive() for p in self.procs):
                    deadline = min(deadline, time.monotonic() + GRACE)
                continue
            if s != seq:
                continue
            (got if status == "ok" else errors)[rank] = payload
            if errors:  # the others may wait on the failed rank in a collective
                deadline = min(deadline, time.monotonic() + GRACE)
        if errors or len(got) < self.world:
            dead = [r for r, p in enumerate(self.procs) if not p.is_alive()]
            silent = sorted(set(range(self.world)) - set(got) - set(errors))
            self.kill()
            lines = [
                f"pool job {seq!r} failed: dead ranks {dead}, no answer from {silent}"
            ]
            lines += [f"--- rank {r} ---\n{tb}" for r, tb in sorted(errors.items())]
            raise RuntimeError("\n".join(lines))
        return [got[r] for r in range(self.world)]

    def kill(self):
        for p in self.procs:
            if p.is_alive():
                p.kill()
        for p in self.procs:
            p.join(10)
        self.procs = []

    def close(self):
        if not self.procs:
            return
        for q in self.tasks:
            q.put(None)
        for p in self.procs:
            p.join(120)
        self.kill()


def _worker_main(rank, world, port, tasks, results, ppid):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        cache = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
        if cache:  # one cache per rank: the ranks compile the same kernels concurrently
            os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = os.path.join(cache, f"rank{rank}")
            os.makedirs(os.environ["FLYDSL_RUNTIME_CACHE_DIR"], exist_ok=True)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        # No device_id: with an eagerly initialized default group, torch makes the non-members of every new
        # sub-group (gloo ones too) enter an NCCL split that the members of a gloo group never join (hang).
        dist.init_process_group("nccl", rank=rank, world_size=world)
        state = _Rank(rank, world, device)
    except BaseException:  # noqa: BLE001 (reported to the parent)
        results.put(("ready", rank, "error", traceback.format_exc()))
        return
    results.put(("ready", rank, "ok", state.gfx))
    while True:
        try:
            item = tasks.get(timeout=10)
        except queue.Empty:
            if os.getppid() != ppid:  # orphaned
                return
            continue
        if item is None:
            break
        seq, job, kw = item
        try:
            results.put((seq, rank, "ok", globals()[f"_job_{job}"](state, **kw)))
        except BaseException:  # noqa: BLE001 (reported to the parent)
            results.put((seq, rank, "error", traceback.format_exc()))
    state.close()
    dist.destroy_process_group()


class _Rank:
    """Per-process state: TP groups, one layer (with its reference weights) per shape, benchmark state."""

    def __init__(self, rank, world, device):
        import aiter  # noqa: F401
        from aiter.jit.utils.chip_info import get_gfx_runtime

        self.rank, self.world, self.device = rank, world, device
        self.gfx = get_gfx_runtime()
        # size -> (RCCL group, gloo group) of ranks [0, size); created by every rank in the same order (collective)
        self.tp = {}
        for size in sorted({2, world}):
            ranks = list(range(size))
            self.tp[size] = (
                None if size == world else dist.new_group(ranks),
                dist.new_group(ranks, backend="gloo"),
            )
        self.layers = {}
        self.bench = None

    def member(self, shape):
        return self.rank < SHAPES[shape]["world"]

    def weights(self, shape, variant=0):
        """This rank's slice of the shape's weights: (routed standard MXFP4 dict, shared expert tuple or None)."""
        s = SHAPES[shape]
        seed = weight_seed(shape, self.rank, variant)
        raw = make_weights(s["E"], s["H"], s["I"], seed=seed, device=self.device)
        shared = None
        if s["shared"]:
            shared = make_shared_weights(
                s["H"], s["I"], seed=seed + 7, device=self.device
            )
        return raw, shared

    def build(self, shape, variant=0):
        """(layer, routed weights, shared weights or None) of this rank (collective over the shape's group)."""
        from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPGfx942

        raw, shared = self.weights(shape, variant)
        group, cpu = self.tp[SHAPES[shape]["world"]]
        layer = MegaMoeTPGfx942(
            make_config(shape, self.rank),
            **to_megamoe_layout(raw),
            **shared_kwargs(shared),
            group=group,
            device=self.device,
            cpu_group=cpu,
        )
        return layer, raw, shared

    def get(self, shape):
        assert self.member(shape), f"rank {self.rank} is not in {shape}"
        if shape not in self.layers:
            self.layers[shape] = self.build(shape)
        return self.layers[shape]

    def inputs(self, shape, M, g, share=True):
        """Input group g of case (shape, M). ``share``: every rank of the group takes rank 0's copy (collective):
        ranks may break topk ties of the random routing differently, and the layer replicates the input.
        """
        s = SHAPES[shape]
        make = make_layer_inputs if s["routing"] == "layer" else make_inputs
        x, tw, ids = make(
            M,
            s["H"],
            s["E"],
            s["topk"],
            seed=input_seed(shape, M, g),
            device=self.device,
        )
        if share:
            for t in (x, tw, ids):
                dist.broadcast(t, src=0, group=self.tp[s["world"]][0])
        return x, tw, ids

    def release(self):
        """Collective over each group this rank uses: free the benchmark state and every layer."""
        if self.bench is not None:
            bench, self.bench = self.bench, None
            if bench["m7"] is not None:
                bench["m7"].close()
            bench["ar"].close()
        for shape in sorted(self.layers):
            self.layers.pop(shape)[0].close()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def close(self):
        self.release()


def _identical(t, group):
    """Whether t is bitwise the same on every rank of the group (collective)."""
    t0 = t.contiguous().clone()
    dist.broadcast(t0, src=0, group=group)
    same = torch.tensor([int(torch.equal(t0, t))], dtype=torch.int32, device=t.device)
    dist.all_reduce(same, op=dist.ReduceOp.MIN, group=group)
    return bool(same.item())


def _group_reference(state, shape, raw, shared, inputs):
    """The FP32 reference of the layer: ``reference_rank`` summed over the shape's group (collective)."""
    ref = reference_rank(shape, raw, shared, *inputs)
    dist.all_reduce(ref, group=state.tp[SHAPES[shape]["world"]][0])
    return ref


def _verify(state, shape, raw, shared, inputs, y):
    """Check this rank's output y against the FP32 reference summed over the group; bitwise identity of y and of
    the inputs over the group (collective)."""
    group = state.tp[SHAPES[shape]["world"]][0]
    res = check(y, _group_reference(state, shape, raw, shared, inputs))
    res["identical"] = _identical(y, group)
    res["inputs_identical"] = all(_identical(t, group) for t in inputs)
    return res


def _job_check(state, shape, tokens):
    """forward + verify of input group 0 for each M in tokens."""
    if not state.member(shape):
        return None
    layer, raw, shared = state.get(shape)
    rows = []
    for M in tokens:
        inputs = state.inputs(shape, M, 0)
        y = layer.forward(*inputs)
        path = ">".join(layer.last_path)
        rows.append(
            dict(M=M, path=path, **_verify(state, shape, raw, shared, inputs, y))
        )
    return rows


def _member_rows(results, shape):
    rows = []
    for rank, part in enumerate(results):
        if rank < SHAPES[shape]["world"]:
            assert part is not None, f"rank {rank} of {shape} returned nothing"
            rows += [dict(r, rank=rank) for r in part]
        else:
            assert part is None, f"rank {rank} is not in {shape}"
    return rows


def _assert_verified(rows):
    bad = [
        r for r in rows if not (r["ok"] and r["identical"] and r["inputs_identical"])
    ]
    assert not bad, "wrong or rank-dependent outputs:\n" + "\n".join(map(str, bad))


@pytest.fixture(scope="session")
def pool():
    if torch.cuda.device_count() < POOL_WORLD:
        pytest.skip(f"needs {POOL_WORLD} GPUs, found {torch.cuda.device_count()}")
    p = _Pool()
    try:
        gfx = p.start()
        if gfx != "gfx942":
            pytest.skip(f"MegaMoeTPGfx942 runs on gfx942 only (found {gfx})")
        yield p
    finally:
        p.close()


# =====================================================================================================================
# Region 1: basic tests (-m basic): one case per dispatch tier of the single-GPU op x shape -- decode (Down on
# pre-quantized rows from 4 rows), decode on pre-quantized rows, H5120's decode deduplicated by expert, the mixed
# routed + shared decode, the decode with the shared expert on a side stream, streaming with block 16 / 32 / 64
# (quant: Gate/Up quantizes the intermediate itself), prefill sorted by the streaming prep or by Opus. Each checks the
# output against the FP32 reference and pins the tier (gateup_tier), whether the shared expert ran (has_shared) and
# exactly one communication segment (comm_kind), not which one.
# =====================================================================================================================
DEC, PQ, DD, MIX = (
    "a8w4_gateup_decode",
    "a8w4_gateup_decode_pq",
    "a8w4_gateup_decode_dd",
    "a8w4_gateup_decode_mixed",
)
ST16, ST32, ST64 = (f"a8w4_gateup_stream_bm{b}" for b in (16, 32, 64))
ST32Q, ST64Q = (f"a8w4_gateup_quant_stream_bm{b}" for b in (32, 64))
PF = "a8w4_gateup_prefill_bm64_bn128_tk128"

BASIC = [
    # decode: one workgroup per route / token
    ("tp2", 1, DEC),
    ("tp4", 8, DEC),  # Down on pre-quantized rows
    ("tp2", 32, PQ),  # both GEMMs on pre-quantized rows
    ("ds_layer_tp4", 1, MIX),  # routed and shared experts in the same two kernels
    ("ds_layer_tp4", 4, PQ),  # the shared expert on a side stream, joined into Down
    ("ds_layer_tp4", 8, DD),  # H5120: Gate/Up deduplicated by expert
    # streaming: weights streamed once per expert block
    ("tp2", 256, ST16),
    ("tp4", 1024, ST32Q),
    ("tp4", 2048, ST64Q),
    ("ds_layer_tp4", 64, ST16),
    ("ds_layer_tp4", 1024, ST32),
    # prefill: sorted 64-row blocks
    ("tp2", 2048, PF),  # sorting by the streaming prep
    ("tp4", 4096, PF),  # Opus moe_sorting
    ("ds_layer_tp4", 2048, PF),  # the shared expert's GEMMs ahead of the routed ones
]


@pytest.mark.basic
@pytest.mark.parametrize("shape,M,tier", BASIC, ids=[f"{s}_m{m}" for s, m, _ in BASIC])
def test_basic(pool, shape, M, tier):
    rows = _member_rows(pool.run("check", shape=shape, tokens=[M]), shape)
    _assert_verified(rows)
    shared = SHAPES[shape]["shared"]
    for r in rows:
        p = r["path"]
        assert (
            gateup_tier(p) == tier
            and has_shared(p) == shared
            and comm_kind(p) is not None
        ), (
            f"rank {r['rank']}: expected Gate/Up {tier}, shared expert {shared} and one communication segment, "
            f"path {p}"
        )


# =====================================================================================================================
# Region 2: other coverage (-m "not basic"). Not here: full-precision checks of the 51 SPEC cases -- the benchmark
# verifies every timed output against the reference (and the cross-rank identity) and fails otherwise --, and the
# dispatch boundaries of the single-GPU op (op_tests/test_flydsl_moe_a8w4.py tests both sides of each; this layer
# adds no dispatch of its own but the all-reduce rule, 2.2).
# =====================================================================================================================
# --- 2.1 DeepSeek-V4.1's MTP layer ----------------------------------------------------------------------------------
# Why: the benchmark and the basic tests take the SPEC shapes only; the MTP layer (128 routed experts, topk 3 + the
# shared expert) reaches shared-expert tiers at other M than ds_layer_tp4: the mixed routed + shared decode still at
# 4 rows (ds_layer_tp4: up to 2), and at 1280 rows the block-64 streaming Gate/Up after the shared expert's GEMMs,
# with the prefill Down (ds_layer_tp4 streams with blocks 16 / 32 only).
MTP_CASES = ((4, MIX), (1280, ST64))


def test_mtp_layer(pool):
    shape = "mtp_layer_tp4"
    tiers = dict(MTP_CASES)
    rows = _member_rows(pool.run("check", shape=shape, tokens=list(tiers)), shape)
    _assert_verified(rows)
    for r in rows:
        p = r["path"]
        assert (
            gateup_tier(p) == tiers[r["M"]]
            and has_shared(p)
            and comm_kind(p) is not None
        ), f"M={r['M']} rank {r['rank']}: expected Gate/Up {tiers[r['M']]} with the shared expert, path {p}"


# --- 2.2 the all-reduce rule ------------------------------------------------------------------------------------
# Why: which all-reduce a forward takes is pinned nowhere else (fusing the communication changes it). Here the
# layer's communicator itself sums BF16 [rows, 2560] tensors on both sides of AR_RCCL_MIN_BYTES (TP2 and TP4): the
# custom all-reduce below, RCCL from there on, both exact (sums of small integers).
def _ar_rows(world, hidden):
    """(rows, expected all-reduce) on both sides of the RCCL threshold of a TP size, BF16 rows of ``hidden``."""
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import AR_RCCL_MIN_BYTES

    # the first row count at or above the threshold
    first = -(-AR_RCCL_MIN_BYTES[world] // (2 * hidden))
    return ((first - 1, CA), (first, RCCL))


def _job_ar_rule(state, shape):
    if not state.member(shape):
        return None
    s, dev = SHAPES[shape], state.device
    comm, w = state.get(shape)[0]._comm, s["world"]
    rows = []
    for M, kind in _ar_rows(w, s["H"]):
        gen = torch.Generator(device=dev).manual_seed(SEED + M)
        base = torch.randint(-8, 9, (M, s["H"]), generator=gen, device=dev)
        y = (base * (state.rank + 1)).to(torch.bfloat16)
        uses_custom = comm.uses_custom(y)
        out = comm.all_reduce(y)
        torch.cuda.synchronize(dev)
        exact = torch.equal(out, (base * (w * (w + 1) // 2)).to(torch.bfloat16))
        rows.append(
            {
                "M": M,
                "want": kind,
                "last": comm.last,
                "uses_custom": uses_custom,
                "bytes": y.numel() * 2,
                "exact": exact,
            }
        )
    return rows


@pytest.mark.parametrize("shape", ["tp2", "tp4"])
def test_all_reduce_rule(pool, shape):
    for rank, rows in enumerate(pool.run("ar_rule", shape=shape)):
        if rank >= SHAPES[shape]["world"]:
            assert rows is None
            continue
        for r in rows:
            assert (
                r["last"] == r["want"]
                and r["uses_custom"] == (r["want"] == CA)
                and r["exact"]
            ), f"rank {rank}: {r}"


# --- 2.3 out=, M = 0 --------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests never pass ``out`` and never send M = 0. ``out`` (NaN-filled) must hold the
# output whichever all-reduce runs (the custom one returns a new tensor, copied back; RCCL sums in place): TP4 8 and
# 4096 rows, DeepSeek-V4.1 TP4 64 rows (shared expert). M = 0 returns an empty output and launches nothing.
OUT_ARG_CASES = (("tp4", 8), ("tp4", 4096), ("ds_layer_tp4", 64))


def _job_out_arg(state):
    rows, empty = [], []
    for shape, M in OUT_ARG_CASES:
        layer, raw, shared = state.get(shape)
        inputs = state.inputs(shape, M, 1)
        out = torch.full(
            (M, SHAPES[shape]["H"]),
            float("nan"),
            dtype=torch.bfloat16,
            device=state.device,
        )
        y = layer.forward(*inputs, out=out)
        rows.append(
            dict(
                shape=shape,
                M=M,
                path=">".join(layer.last_path),
                returned_out=y is out,
                **_verify(state, shape, raw, shared, inputs, out),
            )
        )
    for shape in ("tp4", "ds_layer_tp4"):
        s, layer = SHAPES[shape], state.get(shape)[0]
        width = s["topk"] + int(s["shared"])
        y = layer.forward(
            torch.empty((0, s["H"]), dtype=torch.bfloat16, device=state.device),
            torch.empty((0, width), dtype=torch.float32, device=state.device),
            torch.empty((0, width), dtype=torch.int32, device=state.device),
        )
        empty.append((shape, tuple(y.shape), tuple(layer.last_path)))
    return {"rows": rows, "empty": empty}


def test_out_argument_and_empty_batch(pool):
    for rank, res in enumerate(pool.run("out_arg")):
        _assert_verified(res["rows"])
        assert [(r["shape"], r["M"]) for r in res["rows"]] == list(OUT_ARG_CASES)
        for r in res["rows"]:
            assert r["returned_out"] and _known_path(
                r["path"]
            ), f"rank {rank}: forward(out=...) did not return out or ran an unknown path: {r}"
        assert res["empty"] == [
            (shape, (0, SHAPES[shape]["H"]), ()) for shape in ("tp4", "ds_layer_tp4")
        ], res["empty"]


# --- 2.4 repeated calls across tiers, reset, error flags --------------------------------------------------------
# Why: the benchmark repeats one case at a time and the basic tests call each case once; here one instance jumps
# between decode / streaming / prefill (cached plans and buffers of the op; with the shared expert also its side
# stream), calls reset() in between, and the watchdog API (check_errors / poll_errors / error_flag) is read
# afterwards.
REPEAT = {
    "tp2": (1, 2048, 48, 4096, 16, 256, 1),
    "ds_layer_tp4": (1, 2048, 8, 64, 4),
}


def _job_repeat(state):
    res = {}
    for shape, tokens in REPEAT.items():
        if not state.member(shape):
            continue
        layer, raw, shared = state.get(shape)
        rows = []
        for M in tokens:
            for g in (0, 1):
                inputs = state.inputs(shape, M, g)
                y = layer.forward(*inputs)
                rows.append(
                    dict(
                        M=M,
                        g=g,
                        path=">".join(layer.last_path),
                        **_verify(state, shape, raw, shared, inputs, y),
                    )
                )
            layer.reset()
        layer.check_errors()
        res[shape] = {
            "rows": rows,
            "poll": layer.poll_errors(),
            "flag": int(layer.error_flag().abs().max().item()),
        }
    return res


def test_repeated_calls_and_reset(pool):
    for rank, res in enumerate(pool.run("repeat")):
        assert sorted(res) == sorted(s for s in REPEAT if rank < SHAPES[s]["world"])
        for shape, r in res.items():
            _assert_verified(r["rows"])
            assert all(_known_path(x["path"]) for x in r["rows"]), r["rows"]
            assert (
                r["poll"] == 0 and r["flag"] == 0
            ), f"rank {rank} {shape}: error flags {r}"


# --- 2.5 TP2 / TP4 groups interleaved ---------------------------------------------------------------------------
# Why: the benchmark and the basic tests run one shape after another; here ranks 0 and 1 interleave forwards of a
# TP2 layer and of two TP4 layers (which share one communicator) while ranks 2 and 3 run only the TP4 ones.
ALTERNATE = (
    ("tp2", 8),
    ("tp4", 64),
    ("ds_layer_tp4", 16),
    ("tp2", 4096),
    ("tp4", 1),
    ("ds_layer_tp4", 48),
    ("tp2", 1),
)


def _job_alternate(state):
    rows = []
    for shape, M in ALTERNATE:
        if not state.member(shape):
            continue
        layer, raw, shared = state.get(shape)
        inputs = state.inputs(shape, M, 3)
        y = layer.forward(*inputs)
        rows.append(
            dict(
                shape=shape,
                M=M,
                path=">".join(layer.last_path),
                **_verify(state, shape, raw, shared, inputs, y),
            )
        )
    return rows


def test_tp2_tp4_interleaved(pool):
    for rank, rows in enumerate(pool.run("alternate")):
        expect = [(s, m) for s, m in ALTERNATE if rank < SHAPES[s]["world"]]
        assert [(r["shape"], r["M"]) for r in rows] == expect
        _assert_verified(rows)
        assert all(_known_path(r["path"]) for r in rows), rows


# --- 2.6 set_weights --------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests construct each layer once with one set of weights; set_weights converts
# another layer's weights in place (routed experts, and the shared expert's) and must switch every later forward to
# them (and back).
SET_WEIGHTS = {"tp4": (16, 256), "ds_layer_tp4": (4, 64)}


def _job_set_weights(state):
    res = {}
    for shape, tokens in SET_WEIGHTS.items():
        layer, raw, shared = state.get(shape)
        raw2, shared2 = state.weights(shape, variant=1)
        rows = []
        try:
            layer.set_weights(**to_megamoe_layout(raw2), **shared_kwargs(shared2))
            for M in tokens:
                inputs = state.inputs(shape, M, 2)
                y = layer.forward(*inputs)
                rows.append(
                    dict(
                        weights=2,
                        M=M,
                        path=">".join(layer.last_path),
                        **_verify(state, shape, raw2, shared2, inputs, y),
                    )
                )
        finally:
            layer.set_weights(**to_megamoe_layout(raw), **shared_kwargs(shared))
        inputs = state.inputs(shape, tokens[0], 2)
        y = layer.forward(*inputs)
        rows.append(
            dict(
                weights=1,
                M=tokens[0],
                path=">".join(layer.last_path),
                **_verify(state, shape, raw, shared, inputs, y),
            )
        )
        res[shape] = rows
    return res


def test_set_weights(pool):
    for res in pool.run("set_weights"):
        assert sorted(res) == sorted(SET_WEIGHTS)
        for shape, rows in res.items():
            _assert_verified(rows)
            assert [r["weights"] for r in rows] == [2] * len(SET_WEIGHTS[shape]) + [1]
            assert all(has_shared(r["path"]) == SHAPES[shape]["shared"] for r in rows)


# --- 2.7 CUDA graph capture and replay --------------------------------------------------------------------------
# Why: the benchmark and the basic tests run eagerly; serving captures the decode batches in CUDA graphs. After
# prepare(), forward is captured once (static input / output buffers) and replayed on two other input groups: decode
# (TP2 8; DeepSeek-V4.1 TP4 1, the mixed decode, and 4, the shared expert on its side stream), streaming (TP4 2048,
# DeepSeek-V4.1 TP4 64 with the side stream), prefill (DeepSeek-V4.1 TP4 2048) and both all-reduces (TP2 4096 and
# DeepSeek-V4.1 TP4 2048 take RCCL, the others the custom all-reduce).
GRAPH_CASES = [
    ("tp2", 8),
    ("tp4", 2048),
    ("tp2", 4096),
    ("ds_layer_tp4", 1),
    ("ds_layer_tp4", 4),
    ("ds_layer_tp4", 64),
    ("ds_layer_tp4", 2048),
]


def _job_graph(state, shape, M):
    if not state.member(shape):
        return None
    layer, raw, shared = state.get(shape)
    dev = state.device
    layer.prepare([-(-M // SHAPES[shape]["world"])])
    xs, tws, idss = (t.clone() for t in state.inputs(shape, M, 0))
    ys = torch.empty_like(xs)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        layer.forward(xs, tws, idss, out=ys)
    torch.cuda.current_stream(dev).wait_stream(side)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.forward(xs, tws, idss, out=ys)
    path = ">".join(layer.last_path)
    rows = []
    for g in (1, 2):
        inputs = state.inputs(shape, M, g)
        for static, t in zip((xs, tws, idss), inputs):
            static.copy_(t)
        graph.replay()
        torch.cuda.synchronize(dev)
        rows.append(
            dict(M=M, g=g, path=path, **_verify(state, shape, raw, shared, inputs, ys))
        )
    graph.reset()
    return rows


@pytest.mark.parametrize(
    "shape,M", GRAPH_CASES, ids=[f"{s}_m{m}" for s, m in GRAPH_CASES]
)
def test_cuda_graph(pool, shape, M):
    rows = _member_rows(pool.run("graph", shape=shape, M=M), shape)
    _assert_verified(rows)
    assert all(
        _known_path(r["path"]) and has_shared(r["path"]) == SHAPES[shape]["shared"]
        for r in rows
    ), rows


# --- 2.8 default communicators ----------------------------------------------------------------------------------
# Why: the benchmark and the other tests pass explicit TP / gloo groups; with group=None and cpu_group=None the
# layer uses the default group and creates its gloo group itself (collective), sums over them (8 and 48 rows) and
# close() releases both.
DEFAULT_COMM_TOKENS = (8, 48)


def _job_default_comm(state):
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPGfx942

    shape = "tp4"
    _, raw, _ = state.get(shape)
    layer = MegaMoeTPGfx942(
        make_config(shape, state.rank), **to_megamoe_layout(raw), device=state.device
    )
    try:
        rows = []
        for M in DEFAULT_COMM_TOKENS:
            inputs = state.inputs(shape, M, 0)
            y = layer.forward(*inputs)
            rows.append(
                dict(
                    M=M,
                    path=">".join(layer.last_path),
                    **_verify(state, shape, raw, None, inputs, y),
                )
            )
        return rows
    finally:
        layer.close()


def test_default_communicators(pool):
    for rows in pool.run("default_comm"):
        _assert_verified(rows)
        assert [r["M"] for r in rows] == list(DEFAULT_COMM_TOKENS), rows
        assert all(_known_path(r["path"]) for r in rows), rows


# --- 2.9 rejected configurations and arguments ------------------------------------------------------------------
# Why: the benchmark and the basic tests pass valid arguments only. Every rank constructs the TP4 and DeepSeek-V4.1
# layers (collective), then rank 0 alone: every check raises before any collective (constructor checks before the
# communicator; set_weights checks before converting; forward checks before launching). With a shared expert: a
# partial set, an argument that is no tensor, no FP8 / two FP8 formats, two scale formats, scales of another shape
# or dtype, other shapes than [2I, H] / [H, I], another device, NaN elements / E8M0 codes / scales; set_weights that
# adds or drops the shared expert; a routing of the routed width only.
def _job_errors(state):
    import aiter.ops.flydsl.mega_moe_tp_gfx942 as mod
    from aiter import ActivationType

    shape, ds = "tp4", "ds_layer_tp4"
    layer, raw, _ = state.get(shape)
    ds_layer, ds_raw, ds_shared = state.get(ds)
    if state.rank != 0:
        return None
    s, dev = SHAPES[shape], state.device
    w, wd = to_megamoe_layout(raw), to_megamoe_layout(ds_raw)
    group, cpu = state.tp[s["world"]]
    x, tw, ids = state.inputs(shape, 8, 0, share=False)
    xd, twd, idsd = state.inputs(ds, 8, 0, share=False)
    rows = MAX_TOKENS + 1
    big = (
        torch.zeros((rows, s["H"]), dtype=torch.bfloat16, device=dev),
        tw[:1].expand(rows, -1).contiguous(),
        ids[:1].expand(rows, -1).contiguous(),
    )

    def build(cfg=None, **weights):
        return mod.MegaMoeTPGfx942(
            make_config(shape, **{"rank": 0, **(cfg or {})}),
            **dict(w, **weights),
            group=group,
            device=dev,
            cpu_group=cpu,
        )

    def build_ds(**shared):
        return mod.MegaMoeTPGfx942(
            make_config(ds, 0),
            **wd,
            **dict(shared_kwargs(ds_shared), **shared),
            group=group,
            device=dev,
            cpu_group=cpu,
        )

    def on_gfx950():
        real = mod.get_gfx_runtime
        mod.get_gfx_runtime = lambda: "gfx950"
        try:
            return build()
        finally:
            mod.get_gfx_runtime = real

    w1s, s1s, w2s, s2s = ds_shared
    H, inter = SHAPES[ds]["H"], SHAPES[ds]["I"]

    def blocks(rows_, cols, code=127):
        return torch.full(
            (rows_ // 32, cols // 32), code, dtype=torch.uint8, device=dev
        )

    nan_w1 = w1s.clone()
    nan_w1.view(torch.uint8)[0, 0] = 0x7F
    inf_s1 = s1s.clone()
    inf_s1[0] = float("inf")
    cases = [
        ("not gfx942", on_gfx950, ValueError, "gfx942"),
        (
            "comm_mode unknown",
            lambda: build({"comm_mode": "a2a"}),
            ValueError,
            "comm_mode",
        ),
        (
            "activation Gelu",
            lambda: build({"activation": ActivationType.Gelu}),
            ValueError,
            "activation",
        ),
        (
            "gfx950 act_dtype",
            lambda: build({"act_dtype": "fp8"}),
            ValueError,
            "gfx950",
        ),
        ("gfx950 lb_min", lambda: build({"lb_min": 1}), ValueError, "gfx950"),
        (
            "comm_dtype unknown",
            lambda: build({"comm_dtype": "int8"}),
            ValueError,
            "comm_dtype",
        ),
        (
            "ar_gather fp8",
            lambda: build({"ar_gather": "fp8"}),
            ValueError,
            "ar_gather",
        ),
        (
            "swiglu_limit < 0",
            lambda: build({"swiglu_limit": -1.0}),
            ValueError,
            "swiglu_limit",
        ),
        ("rank mismatch", lambda: build({"rank": 1}), ValueError, "rank"),
        ("w1 shape", lambda: build(w1=w["w1"][:, :-16]), ValueError, "w1"),
        ("w2 dtype", lambda: build(w2=w["w2"].view(torch.int8)), TypeError, "w2"),
        (
            "w1_scale shape",
            lambda: build(w1_scale=w["w1_scale"][:-256]),
            ValueError,
            "scale",
        ),
        (
            "w1_scale E8M0 NaN code",
            lambda: build(
                w1_scale=w["w1_scale"]
                .view(torch.uint8)
                .clone()
                .fill_(255)
                .view(w["w1_scale"].dtype)
            ),
            ValueError,
            "255",
        ),
        (
            "model_dim not a multiple of 512",
            lambda: build({"model_dim": s["H"] + 256}),
            ValueError,
            "model_dim",
        ),
        (
            "max_local_tokens < 1",
            lambda: build({"max_local_tokens": 0}),
            ValueError,
            "max_local_tokens",
        ),
        (
            "shared: a partial set",
            lambda: build_ds(shared_w2_scale=None),
            ValueError,
            "needs all of",
        ),
        (
            "shared: no tensor",
            lambda: build_ds(shared_w1=[0.0]),
            TypeError,
            "must be a tensor",
        ),
        (
            "shared: uint8 weight",
            lambda: build_ds(shared_w1=w1s.view(torch.uint8)),
            ValueError,
            "float8_e4m3fn",
        ),
        (
            "shared: E4M3FN and E4M3FNUZ",
            lambda: build_ds(shared_w2=w2s.view(torch.float8_e4m3fnuz)),
            ValueError,
            "one FP8 format",
        ),
        (
            "shared: per-row and 32x32 block scales",
            lambda: build_ds(shared_w2_scale=blocks(H, inter)),
            ValueError,
            "one format",
        ),
        (
            "shared: per-row scale shape",
            lambda: build_ds(shared_w1_scale=s1s[:-1]),
            ValueError,
            "per-row float32 scales",
        ),
        (
            "shared: block scale shape",
            lambda: build_ds(
                shared_w1_scale=blocks(2 * inter, H)[:-1],
                shared_w2_scale=blocks(H, inter),
            ),
            ValueError,
            "32x32 blocks",
        ),
        (
            "shared: bfloat16 scales",
            lambda: build_ds(
                shared_w1_scale=s1s.bfloat16(), shared_w2_scale=s2s.bfloat16()
            ),
            ValueError,
            "must be float32 per-row scales",
        ),
        (
            "shared: w1 rows",
            lambda: build_ds(shared_w1=w1s[:-32], shared_w1_scale=s1s[:-32]),
            ValueError,
            "shapes must match",
        ),
        (
            "shared: on the CPU",
            lambda: build_ds(shared_w2_scale=s2s.cpu()),
            ValueError,
            "is on cpu",
        ),
        (
            "shared: NaN element",
            lambda: build_ds(shared_w1=nan_w1),
            ValueError,
            "NaN elements",
        ),
        (
            "shared: non-finite per-row scale",
            lambda: build_ds(shared_w1_scale=inf_s1),
            ValueError,
            "non-finite",
        ),
        (
            "shared: E8M0 NaN code",
            lambda: build_ds(
                shared_w1_scale=blocks(2 * inter, H, 255),
                shared_w2_scale=blocks(H, inter),
            ),
            ValueError,
            "255",
        ),
        (
            "set_weights adds a shared expert",
            lambda: layer.set_weights(
                **w,
                **shared_kwargs(
                    make_shared_weights(s["H"], s["I"], seed=SEED + 9, device=dev)
                ),
            ),
            ValueError,
            "keep that",
        ),
        (
            "set_weights drops the shared expert",
            lambda: ds_layer.set_weights(**wd),
            ValueError,
            "keep that",
        ),
        ("x dtype", lambda: layer.forward(x.float(), tw, ids), ValueError, "x must be"),
        (
            "x hidden",
            lambda: layer.forward(x[:, :-512], tw, ids),
            ValueError,
            "x must be",
        ),
        (
            "topk shape",
            lambda: layer.forward(x, tw[:, :-1], ids[:, :-1]),
            ValueError,
            "topk",
        ),
        (
            "shared: routing without the shared column",
            lambda: ds_layer.forward(xd, twd[:, :-1], idsd[:, :-1]),
            ValueError,
            "topk",
        ),
        (
            "rows > max_local_tokens x world",
            lambda: layer.forward(*big),
            ValueError,
            "max_local_tokens",
        ),
        (
            "out dtype",
            lambda: layer.forward(x, tw, ids, out=x.float()),
            ValueError,
            "out",
        ),
        (
            "forward tail",
            lambda: layer.forward(x, tw, ids, tail=object()),
            ValueError,
            "tail",
        ),
        ("prepare tail", lambda: layer.prepare(1, tail=True), ValueError, "tail"),
    ]
    failures = []
    for name, fn, exc, text in cases:
        try:
            made = fn()
        except exc as e:
            if text not in str(e):
                failures.append(f"{name}: message {str(e)!r} does not mention {text!r}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: raised {type(e).__name__}: {e}")
        else:
            if isinstance(made, mod.MegaMoeTPGfx942):
                made.close()
            failures.append(f"{name}: no error")
    return failures


def test_rejected_arguments(pool):
    failures = pool.run("errors")[0]
    assert not failures, "\n".join(failures)
    # the rejected set_weights calls left the layers' weights in place
    for shape in ("tp4", "ds_layer_tp4"):
        rows = _member_rows(pool.run("check", shape=shape, tokens=[8]), shape)
        _assert_verified(rows)


# --- 2.10 the path checks themselves (CPU only) -----------------------------------------------------------------
# Why: the benchmark does not check paths, and the tests above only see the paths this code takes (a known Gate/Up
# kernel, one all-reduce at the end, the shared expert's steps); checks that accepted anything would pass them all.
# Their verdicts on fused reductions (later milestones), on the shared expert's steps and on broken paths (no /
# several / misplaced all-reduces, no / an unknown Gate/Up) are pinned here.
def test_path_checks():
    qwen_decode = "a8w4_gateup_decode>a8w4_down_decode"
    qwen_prefill = f"moe_sorting_opus>a8w4_row_quant_fp8>{PF}>a8w4_row_quant_fp8>a8w4_down_prefill_bm64>sorted_sum"
    shared_gemms = "shared_fold_gemm_gateup>shared_fold_mid>shared_fold_gemm_down"
    ds_mixed = f"{MIX}>a8w4_down_decode_mixed>shared_mixed"
    ds_side = (
        f"a8w4_row_quant_decode>{PQ}>shared_fork>{shared_gemms}>a8w4_row_quant_decode>shared_join>"
        "a8w4_down_decode_pq_shared"
    )
    ds_prefill = f"a8w4_stream_prep>{shared_gemms}>{PF}>a8w4_row_quant_fp8>a8w4_down_prefill_bm64>sorted_sum"
    fused = "a8w4_gateup_decode>a8w4_down_decode_tp"
    stream_fused = f"a8w4_stream_prep>{ST16}>a8w4_row_quant_fp8>a8w4_down_stream_bm16>a8w4_tp_reduce"
    cases = [  # path, comm_kind, _known_path, has_shared
        (f"{qwen_decode}>{CA}", CA, True, False),
        (f"{qwen_prefill}>{RCCL}", RCCL, True, False),
        (f"{ds_mixed}>{CA}", CA, True, True),
        (f"{ds_side}>{CA}", CA, True, True),
        (f"{ds_prefill}>{RCCL}", RCCL, True, True),
        (fused, FUSED_COMM, True, False),
        (stream_fused, FUSED_COMM, True, False),
        (f"{ds_prefill}_tp", FUSED_COMM, True, True),  # sorted_sum_tp
        (qwen_decode, None, False, False),  # no communication
        (f"{fused}>{CA}", None, False, False),  # fused and an all-reduce
        (f"{qwen_decode}>{RCCL}>{RCCL}", None, False, False),
        (f"a8w4_gateup_decode>{CA}>a8w4_down_decode", None, False, False),  # not last
        ("a8w4_gateup_decode>a8w4_down_decode_stp", None, False, False),  # no word "tp"
        (f"a8w4_down_decode>{CA}", CA, False, False),  # no Gate/Up kernel
        (f"a8w4_gateup_decode_v2>a8w4_down_decode>{CA}", CA, False, False),  # unknown
        (f"{qwen_decode}_unshared>{CA}", CA, True, False),  # no word "shared"
    ]
    wrong = [
        (p, comm_kind(p), _known_path(p), has_shared(p))
        for p, *want in cases
        if [comm_kind(p), _known_path(p), has_shared(p)] != want
    ]
    assert not wrong, wrong
    # the first Gate/Up kernel, also behind sorting / quantization / the shared expert's GEMMs
    assert gateup_tier(qwen_prefill) == gateup_tier(ds_prefill) == PF
    assert gateup_tier(ds_side) == PQ and gateup_tier(fused) == DEC
    tiers = (DEC, PQ, DD, MIX, ST16, ST32, ST64, ST32Q, ST64Q, PF)
    assert all(GATEUP_KERNEL.fullmatch(t) for t in tiers), tiers


# --- 2.11 real shared experts: DeepSeek-V4.1-Flash layers 2 and 37, TP4 ------------------------------------------
# Why: the basic tests and the benchmark grade random per-row FP8 weights against a reference with this op's own
# quantization. Here the official FP8 shared experts with E8M0 scales of 32x32 blocks, sliced for TP4 along I (rank r:
# w1 / w3 rows and w2 columns [576 r, 576 r + 576)), run through the whole layer -- every rank's slice, then the
# all-reduce -- on inputs N(0, 1) and N(0, 1) with 4 outlier channels times 50 (M 64, seeds 731-733, SwiGLU limit 10;
# 8 routed experts with routing weight 0, so the output is the shared expert's). The relative L2 error against the FP32
# expert on the same block weights (x and the intermediate not quantized; the ranks' terms summed) is at most 1.5
# times that of the model's own quantization (x and the intermediate in FP8 with one UE8M0 scale per 32, as
# DeepSeek-V4.1 runs it), as op_tests/test_flydsl_moe_a8w4.py grades rank 0's slice alone on one GPU. Reads the
# checkpoint at AITER_MEGA_MOE_TP_DSV41_DIR (default /models/DeepSeek-V4.1-Flash-1017-partial); skipped without it.
DSV41_DIR = os.environ.get(
    "AITER_MEGA_MOE_TP_DSV41_DIR", "/models/DeepSeek-V4.1-Flash-1017-partial"
)
DSV41_LAYERS = ("layers.2", "layers.37")
DSV41_TOKENS, DSV41_EXPERTS, DSV41_TOPK = 64, 8, 2


def _dsv41_files(layer):
    """{tensor name: safetensors file} of a layer's shared expert in DSV41_DIR (pytest.skip without them)."""
    index = Path(DSV41_DIR) / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"{index} not found (DeepSeek-V4.1-Flash checkpoint)")
    weight_map = json.loads(index.read_text())["weight_map"]
    names = [
        f"{layer}.ffn.shared_experts.{m}.{k}"
        for m in ("w1", "w3", "w2")
        for k in ("weight", "scale")
    ]
    files = {n: weight_map.get(n) for n in names}
    missing = [
        n for n, f in files.items() if f is None or not (Path(DSV41_DIR) / f).exists()
    ]
    if missing:
        pytest.skip(f"{DSV41_DIR}: missing {missing}")
    return {n: str(Path(DSV41_DIR) / f) for n, f in files.items()}


def _dsv41_slice(files, layer, rank, world, inter):
    """This rank's TP slice of a layer's official shared expert on the CPU: (shared_w1 [2I, H] = w1 rows then w3 rows,
    its E8M0 [2I/32, H/32] block scales, shared_w2 [H, I], [H/32, I/32])."""
    from safetensors import safe_open

    out = {}
    for name, path in files.items():
        block = 32 if name.endswith(".scale") else 1  # a scale covers 32 rows / columns
        down = ".w2." in name  # w2 [H, I]: sliced along its columns
        with safe_open(path, "pt", device="cpu") as f:
            t = f.get_slice(name)
            assert t.get_shape()[int(down)] * block == world * inter, (
                name,
                t.get_shape(),
            )
            part = slice(rank * inter // block, (rank + 1) * inter // block)
            out[name] = (t[:, part] if down else t[part]).contiguous()
    p = f"{layer}.ffn.shared_experts."
    return (
        torch.cat([out[p + "w1.weight"], out[p + "w3.weight"]]),
        torch.cat([out[p + "w1.scale"], out[p + "w3.scale"]]),
        out[p + "w2.weight"],
        out[p + "w2.scale"],
    )


def _rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _quant_1x32(t):
    """The model's activation quantization: FP8 E4M3FN with one UE8M0 scale per 32 elements, 2**ceil(log2(amax /
    448)) (exact from the binary exponent), RNE; dequantized, float32."""
    rows, cols = t.shape
    b = t.float().view(rows, cols // 32, 32)
    amax = b.abs().amax(dim=-1, keepdim=True).double()
    m, e = torch.frexp(
        torch.where(amax > 0, amax, torch.full_like(amax, FP8_OCP_MAX)) / FP8_OCP_MAX
    )
    scale = torch.exp2((e - (m == 0.5).to(e.dtype)).double()).float()
    return ((b / scale).to(torch.float8_e4m3fn).float() * scale).view(rows, cols)


def _shared_fp32(x, w1, w2, limit, quant=None):
    """A shared expert slice in float32 on dequantized weights; ``quant`` quantizes x and the intermediate (None:
    not)."""
    inter = w1.shape[0] // 2
    xa = x.float() if quant is None else quant(x)
    h = xa @ w1.T
    gate, up = h[:, :inter].clamp(max=limit), h[:, inter:].clamp(-limit, limit)
    mid = torch.nn.functional.silu(gate) * up
    return (mid if quant is None else quant(mid)) @ w2.T


def _job_real_shared(state, layer, files):
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPGfx942

    shape = "ds_layer_tp4"
    s, dev, rank = SHAPES[shape], state.device, state.rank
    group, cpu = state.tp[s["world"]]
    E, topk, M, limit = DSV41_EXPERTS, DSV41_TOPK, DSV41_TOKENS, s["limit"]
    shared = tuple(
        t.to(dev) for t in _dsv41_slice(files, layer, rank, s["world"], s["I"])
    )
    w1, w2 = dequant_fp8(*shared[:2]), dequant_fp8(*shared[2:])
    raw = make_weights(E, s["H"], s["I"], seed=730 + rank, device=dev)
    tp = MegaMoeTPGfx942(
        make_config(shape, rank, experts=E, topk=topk),
        **to_megamoe_layout(raw),
        **shared_kwargs(shared),
        group=group,
        device=dev,
        cpu_group=cpu,
    )
    ids = torch.arange(topk + 1, dtype=torch.int32, device=dev).repeat(M, 1)
    ids[:, -1] = E
    weight = torch.zeros((M, topk + 1), dtype=torch.float32, device=dev)
    weight[:, -1] = 1.0
    rows = []
    try:
        for kind, seed in itertools.product(("gauss", "outliers"), range(3)):
            gen = torch.Generator(device=dev).manual_seed(731 + seed)
            x = torch.randn((M, s["H"]), generator=gen, device=dev)
            if kind == "outliers":
                channels = torch.randperm(s["H"], generator=gen, device=dev)[:4]
                x[:, channels] *= 50
            x = x.to(torch.bfloat16)
            dist.broadcast(x, src=0, group=group)
            y = tp.forward(x, weight, ids)
            path = ">".join(tp.last_path)
            exact = _shared_fp32(x, w1, w2, limit)
            model = _shared_fp32(x, w1, w2, limit, _quant_1x32)
            for t in (exact, model):
                dist.all_reduce(t, group=group)
            rows.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "path": path,
                    "ours": _rel_l2(y, exact),
                    "model": _rel_l2(model, exact),
                    "identical": _identical(y, group),
                }
            )
    finally:
        tp.close()
    return rows


@pytest.mark.parametrize("layer", DSV41_LAYERS)
def test_real_shared_experts_tp4(pool, layer):
    pytest.importorskip("safetensors")
    files = _dsv41_files(layer)
    results = pool.run("real_shared", layer=layer, files=files)
    for rank, rows in enumerate(results):
        assert len(rows) == 6, rows
        for r in rows:
            if rank == 0:
                print(
                    f"{layer} {r['kind']} seed {r['seed']}: relative L2 ours {r['ours']:.4e}, model "
                    f"{r['model']:.4e}, ours / model {r['ours'] / r['model']:.3f}"
                )
            assert r["identical"] and has_shared(r["path"]), (rank, r)
            assert comm_kind(r["path"]) is not None, (rank, r)
            assert r["ours"] <= 1.5 * r["model"], (rank, r)


# =====================================================================================================================
# DP section: comm_mode="ag_rs" (DP4 attention + TP4 MoE), shape ds_dp4. Every rank passes its own M_r tokens (the
# same M_r on every rank); the layer gathers all world * M_r rows (rank r's at [r * M_r, (r + 1) * M_r)), runs this
# rank's slice of the layer on them and reduce-scatters the partial sums back to the tokens' ranks. Inputs: the whole
# batch generated on rank 0 and broadcast (``_Rank.inputs`` of world * M_r rows), every rank passes its rows. Each
# rank checks its output against its rows of the FP32 reference summed over the group (the acceptance of region 1);
# the outputs differ across ranks, so instead of their cross-rank identity a second call on the same inputs must give
# bitwise the same output. D.1 holds the basic tests (-m basic), the rest is other coverage. Pinned: a Gate/Up kernel
# ran, the shared expert ran, and the communication (dp_comm): one all-gather before the first Gate/Up kernel and one
# reduce-scatter after it, either replaceable by fused communication kernels (the word "tp" in their names, as in
# region 1), no all-reduce. Not pinned: the tier (the single-GPU op's, tested in region 1 and by
# op_tests/test_flydsl_moe_a8w4.py) and which collectives an M_r takes (D.2 tests the rule on the communicator).
# =====================================================================================================================
DP_SHAPE = "ds_dp4"
AG_CA, AG_RCCL = "custom_all_gather", "rccl_all_gather"
RS_CA, RS_RCCL = "custom_reduce_scatter", "rccl_reduce_scatter"


def dp_comm(path):
    """The communication of an ag_rs path: (entry, exit), None when broken. entry: the one all-gather (AG_CA /
    AG_RCCL), before the first Gate/Up kernel, or FUSED_COMM for no all-gather and a fused communication kernel (the
    word "tp") up to the first Gate/Up kernel; exit: the one reduce-scatter (RS_CA / RS_RCCL), after the first Gate/Up
    kernel, or FUSED_COMM for none and a fused communication kernel after it. None too with an all-reduce or without
    a Gate/Up kernel."""
    kernels = path.split(">")
    first = next(
        (i for i, k in enumerate(kernels) if k.startswith(GATEUP_PREFIX)), None
    )
    if first is None or any(k in (CA, RCCL) for k in kernels):
        return None
    fused = [TP_WORD in k.split("_") for k in kernels]
    ags = [i for i, k in enumerate(kernels) if k in (AG_CA, AG_RCCL)]
    rss = [i for i, k in enumerate(kernels) if k in (RS_CA, RS_RCCL)]
    if ags:
        entry = kernels[ags[0]] if ags == ags[:1] and ags[0] < first else None
    else:
        entry = FUSED_COMM if any(fused[: first + 1]) else None
    if rss:
        exit_ = kernels[rss[0]] if rss == rss[:1] and rss[0] > first else None
    else:
        exit_ = FUSED_COMM if any(fused[first + 1 :]) else None
    return None if entry is None or exit_ is None else (entry, exit_)


def _dp_known_path(path):
    """A path of the ag_rs mode: a Gate/Up kernel, the shared expert's steps and its communication (dp_comm)."""
    return (
        gateup_tier(path) is not None and has_shared(path) and dp_comm(path) is not None
    )


def _dp_full(state, m, g):
    """Input group g of the DP case M_r = m: the whole batch, world * m rows (generated on rank 0 and broadcast)."""
    return state.inputs(DP_SHAPE, SHAPES[DP_SHAPE]["world"] * m, g)


def _dp_mine(state, full):
    """This rank's rows [rank * M_r, (rank + 1) * M_r) of a whole batch (contiguous views)."""
    m = full[0].shape[0] // SHAPES[DP_SHAPE]["world"]
    return tuple(t[state.rank * m : (state.rank + 1) * m] for t in full)


def _dp_reference(state, raw, shared, full):
    """This rank's rows of the FP32 reference of the whole batch (``reference_rank`` summed over the group;
    collective)."""
    m = full[0].shape[0] // SHAPES[DP_SHAPE]["world"]
    ref = _group_reference(state, DP_SHAPE, raw, shared, full)
    return ref[state.rank * m : (state.rank + 1) * m]


def _dp_verify(state, raw, shared, full, y):
    """Check this rank's output y against its rows of the reference; whether the whole batch is bitwise the same on
    every rank (collective)."""
    group = state.tp[SHAPES[DP_SHAPE]["world"]][0]
    res = check(y, _dp_reference(state, raw, shared, full))
    res["inputs_identical"] = all(_identical(t, group) for t in full)
    return res


def _rank_rows(results):
    return [dict(r, rank=rank) for rank, part in enumerate(results) for r in part]


def _assert_dp_verified(rows):
    bad = [
        r
        for r in rows
        if not (r["ok"] and r["inputs_identical"] and r.get("repeatable", True))
    ]
    assert not bad, "wrong or unrepeatable outputs:\n" + "\n".join(map(str, bad))


# --- D.1 basic tests (-m basic): 1 token per rank (decode; the custom all-gather / reduce-scatter) and 2048 (prefill
# on 8192 rows; RCCL), each called twice.
DP_BASIC = (1, 2048)


def _job_dp_check(state, tokens):
    layer, raw, shared = state.get(DP_SHAPE)
    rows = []
    for m in tokens:
        full = _dp_full(state, m, 0)
        y = layer.forward(*_dp_mine(state, full)).clone()
        path = ">".join(layer.last_path)
        again = layer.forward(*_dp_mine(state, full))
        rows.append(
            dict(
                M=m,
                path=path,
                repeatable=torch.equal(y, again),
                **_dp_verify(state, raw, shared, full, y),
            )
        )
    return rows


@pytest.mark.basic
@pytest.mark.parametrize("m", DP_BASIC, ids=[f"{DP_SHAPE}_m{m}" for m in DP_BASIC])
def test_dp_basic(pool, m):
    rows = _rank_rows(pool.run("dp_check", tokens=[m]))
    _assert_dp_verified(rows)
    for r in rows:
        assert _dp_known_path(
            r["path"]
        ), f"rank {r['rank']}: expected an ag_rs path with the shared expert, got {r['path']}"


# --- D.2 the all-gather / reduce-scatter rule ---------------------------------------------------------------------
# Why: which collectives a forward takes is pinned nowhere else (fusing the communication changes them; the basic
# tests take one M_r per side). The layer's communicator gathers BF16 x [M_r, 5120] with int32 / float32 routing and
# reduce-scatters BF16 partials [4 M_r, 5120] on both sides of AG_RCCL_MIN_BYTES and RS_RCCL_MIN_BYTES (custom below,
# RCCL from there on), and with the custom communicator off (as where aiter disables it: RCCL, the routing packed
# into one gather): the gather exact, the reduce-scatter of small integers exact.
def _dp_rule_cases(world, hidden):
    """(M_r, all-gather, reduce-scatter) on both sides of each threshold of a TP size, BF16 rows of ``hidden``."""
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import (
        AG_RCCL_MIN_BYTES,
        RS_RCCL_MIN_BYTES,
    )

    # the first M_r whose x / whose partials (world * M_r rows) reach the threshold
    ag = -(-AG_RCCL_MIN_BYTES[world] // (2 * hidden))
    rs = -(-RS_RCCL_MIN_BYTES[world] // (2 * hidden * world))
    return [
        (m, AG_CA if m < ag else AG_RCCL, RS_CA if m < rs else RS_RCCL)
        for m in sorted({ag - 1, ag, rs - 1, rs})
    ]


def _job_dp_comm_rule(state):
    s, dev, rank = SHAPES[DP_SHAPE], state.device, state.rank
    comm, world, H = state.get(DP_SHAPE)[0]._comm, s["world"], s["H"]
    width = s["topk"] + 1

    def rows_of(r, m):
        gen = torch.Generator(device=dev).manual_seed(SEED + 7919 * r + m)
        x = torch.randint(-8, 9, (m, H), generator=gen, device=dev)
        ids = torch.randint(
            -1, s["E"] + 1, (m, width), generator=gen, device=dev, dtype=torch.int32
        )
        tw = torch.randn((m, width), generator=gen, device=dev)
        return x.to(torch.bfloat16), ids, tw

    cases = [(*c, True) for c in _dp_rule_cases(world, H)] + [
        (8, AG_RCCL, RS_RCCL, False)
    ]
    rows = []
    for m, ag, rs, custom in cases:
        saved = comm.ca
        if not custom:
            comm.ca = None
        try:
            got = comm.all_gather(*rows_of(rank, m))
            want = [torch.cat(t) for t in zip(*(rows_of(r, m) for r in range(world)))]
            gen = torch.Generator(device=dev).manual_seed(SEED + m)
            base = torch.randint(-8, 9, (world * m, H), generator=gen, device=dev)
            out = torch.empty((m, H), dtype=torch.bfloat16, device=dev)
            comm.reduce_scatter((base * (rank + 1)).to(torch.bfloat16), out)
            torch.cuda.synchronize(dev)
        finally:
            comm.ca = saved
        total = base[rank * m : (rank + 1) * m] * (world * (world + 1) // 2)
        rows.append(
            {
                "M_r": m,
                "custom": custom,
                "want": (ag, rs),
                "got": (comm.last_ag, comm.last_rs),
                "gather_exact": all(torch.equal(a, b) for a, b in zip(got, want)),
                "rs_exact": torch.equal(out, total.to(torch.bfloat16)),
            }
        )
    return rows


def test_dp_all_gather_reduce_scatter_rule(pool):
    for rank, rows in enumerate(pool.run("dp_comm_rule")):
        assert len(rows) >= 3, rows
        for r in rows:
            assert (
                r["got"] == r["want"] and r["gather_exact"] and r["rs_exact"]
            ), f"rank {rank}: {r}"


# --- D.3 padding rows -------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests send real tokens only. Serving pads every rank to the same M_r with rows
# routed to -1 in every column (output 0, no shared expert) and may mask a token's first route (its shared expert
# dropped as well, its other routes kept). Rank r pads its last 0, 1, M_r / 2, M_r rows (the last rank only padding,
# random x in every padding row) and masks the first route of its first real row; M_r 2 (decode), 300 (streaming),
# 1024 (prefill, RCCL).
DP_PAD_TOKENS = (2, 300, 1024)


def _dp_pad(full, m):
    """Pad a whole batch in place as above; returns the padding rows' mask."""
    x, _, ids = full
    pad = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for r, n in enumerate((0, 1, m // 2, m)):
        pad[r * m + m - n : (r + 1) * m] = True
        if n < m:
            ids[r * m, 0] = -1
    ids[pad] = -1
    return pad


def _job_dp_padding(state):
    layer, raw, shared = state.get(DP_SHAPE)
    rows = []
    for m in DP_PAD_TOKENS:
        full = _dp_full(state, m, 1)
        pad = _dp_mine(state, (_dp_pad(full, m),))[0]
        y = layer.forward(*_dp_mine(state, full))
        rows.append(
            dict(
                M=m,
                path=">".join(layer.last_path),
                pads=int(pad.sum()),
                pad_zero=bool((y[pad] == 0).all()),
                **_dp_verify(state, raw, shared, full, y),
            )
        )
    return rows


def test_dp_padding_rows(pool):
    rows = _rank_rows(pool.run("dp_padding"))
    _assert_dp_verified(rows)
    for r in rows:
        pads = (0, 1, r["M"] // 2, r["M"])[r["rank"]]
        assert r["pads"] == pads and r["pad_zero"] and _dp_known_path(r["path"]), r


# --- D.4 AR and DP layers interleaved ---------------------------------------------------------------------------
# Why: the benchmark and the other tests run one layer at a time; a server may keep layers of both modes in one
# process (e.g. an MTP layer in AR next to DP main layers). Here a ds_layer_tp4 (ar) and a ds_dp4 (ag_rs) instance
# alternate on the same group, sharing one communicator: the all-reduce and the all-gather / reduce-scatter on both
# sides of their thresholds.
DP_ALTERNATE = (
    ("ds_layer_tp4", 16),
    (DP_SHAPE, 4),
    ("ds_layer_tp4", 2048),
    (DP_SHAPE, 1024),
    (DP_SHAPE, 1),
    ("ds_layer_tp4", 1),
)


def _job_dp_alternate(state):
    rows = []
    for shape, m in DP_ALTERNATE:
        layer, raw, shared = state.get(shape)
        if shape == DP_SHAPE:
            full = _dp_full(state, m, 3)
            y = layer.forward(*_dp_mine(state, full))
            res = _dp_verify(state, raw, shared, full, y)
        else:
            inputs = state.inputs(shape, m, 3)
            y = layer.forward(*inputs)
            res = _verify(state, shape, raw, shared, inputs, y)
        rows.append(dict(shape=shape, M=m, path=">".join(layer.last_path), **res))
    return rows


def test_dp_ar_interleaved(pool):
    for rank, rows in enumerate(pool.run("dp_alternate")):
        assert [(r["shape"], r["M"]) for r in rows] == list(DP_ALTERNATE)
        dp = [r for r in rows if r["shape"] == DP_SHAPE]
        ar = [r for r in rows if r["shape"] != DP_SHAPE]
        _assert_dp_verified(dp)
        _assert_verified(ar)
        assert all(_dp_known_path(r["path"]) for r in dp), (rank, dp)
        assert all(_known_path(r["path"]) for r in ar), (rank, ar)


# --- D.5 CUDA graph capture and replay ----------------------------------------------------------------------------
# Why: the benchmark and the basic tests run eagerly; serving captures the decode batches in CUDA graphs. After
# prepare([M_r]) forward is captured once (static buffers) and replayed on two other input groups: M_r 1 and 16 (the
# custom all-gather / reduce-scatter) and 1024 (RCCL).
DP_GRAPH_TOKENS = (1, 16, 1024)


def _job_dp_graph(state, m):
    layer, raw, shared = state.get(DP_SHAPE)
    dev = state.device
    layer.prepare([m])
    xs, tws, idss = (t.clone() for t in _dp_mine(state, _dp_full(state, m, 0)))
    ys = torch.empty_like(xs)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        layer.forward(xs, tws, idss, out=ys)
    torch.cuda.current_stream(dev).wait_stream(side)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.forward(xs, tws, idss, out=ys)
    path = ">".join(layer.last_path)
    rows = []
    for g in (1, 2):
        full = _dp_full(state, m, g)
        for static, t in zip((xs, tws, idss), _dp_mine(state, full)):
            static.copy_(t)
        graph.replay()
        torch.cuda.synchronize(dev)
        rows.append(
            dict(M=m, g=g, path=path, **_dp_verify(state, raw, shared, full, ys))
        )
    graph.reset()
    return rows


@pytest.mark.parametrize("m", DP_GRAPH_TOKENS, ids=[f"m{m}" for m in DP_GRAPH_TOKENS])
def test_dp_cuda_graph(pool, m):
    rows = _rank_rows(pool.run("dp_graph", m=m))
    _assert_dp_verified(rows)
    assert all(_dp_known_path(r["path"]) for r in rows), rows


# --- D.6 out=, M_r = 0 ------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests never pass ``out`` and never send M_r = 0. ``out`` (NaN-filled) must hold
# the output and be returned for either reduce-scatter (M_r 8: custom, 1024: RCCL); M_r = 0 on every rank returns an
# empty output, launches nothing and communicates nothing.
DP_OUT_TOKENS = (8, 1024)


def _job_dp_out_arg(state):
    layer, raw, shared = state.get(DP_SHAPE)
    s, dev = SHAPES[DP_SHAPE], state.device
    rows = []
    for m in DP_OUT_TOKENS:
        full = _dp_full(state, m, 1)
        out = torch.full((m, s["H"]), float("nan"), dtype=torch.bfloat16, device=dev)
        y = layer.forward(*_dp_mine(state, full), out=out)
        rows.append(
            dict(
                M=m,
                path=">".join(layer.last_path),
                returned_out=y is out,
                **_dp_verify(state, raw, shared, full, out),
            )
        )
    width = s["topk"] + 1
    y = layer.forward(
        torch.empty((0, s["H"]), dtype=torch.bfloat16, device=dev),
        torch.empty((0, width), dtype=torch.float32, device=dev),
        torch.empty((0, width), dtype=torch.int32, device=dev),
    )
    return {"rows": rows, "empty": (tuple(y.shape), tuple(layer.last_path))}


def test_dp_out_argument_and_empty_batch(pool):
    for rank, res in enumerate(pool.run("dp_out_arg")):
        _assert_dp_verified(res["rows"])
        assert [r["M"] for r in res["rows"]] == list(DP_OUT_TOKENS)
        for r in res["rows"]:
            assert r["returned_out"] and _dp_known_path(r["path"]), (rank, r)
        assert res["empty"] == ((0, SHAPES[DP_SHAPE]["H"]), ()), (rank, res["empty"])


# --- D.7 rejected arguments, ranks disagreeing on M_r -----------------------------------------------------------
# Why: the benchmark and the basic tests pass valid arguments only. Rank 0 alone calls forward with each wrong
# argument: every check must raise before the first collective (a rank waiting in one would hang the job). Then with
# moe_a8w4.DEBUG_CHECKS every rank calls forward and the last one passes one token more: every rank raises (the
# counts are compared over the gloo group); afterwards the layer still runs.
def _job_dp_errors(state):
    layer = state.get(DP_SHAPE)[0]
    if state.rank != 0:
        return None
    s, dev = SHAPES[DP_SHAPE], state.device
    m, H = 8, s["H"]
    x, tw, ids = state.inputs(DP_SHAPE, m, 0, share=False)
    big_m = MAX_TOKENS // s["world"] + 1
    big = (
        torch.zeros((big_m, H), dtype=torch.bfloat16, device=dev),
        tw[:1].expand(big_m, -1).contiguous(),
        ids[:1].expand(big_m, -1).contiguous(),
    )
    cases = [
        ("x dtype", lambda: layer.forward(x.float(), tw, ids), ValueError, "x must be"),
        (
            "x hidden",
            lambda: layer.forward(x[:, :-512].contiguous(), tw, ids),
            ValueError,
            "x must be",
        ),
        (
            "x not contiguous",
            lambda: layer.forward(x.t().contiguous().t(), tw, ids),
            ValueError,
            "x must be",
        ),
        (
            "routing without the shared column",
            lambda: layer.forward(x, tw[:, :-1], ids[:, :-1]),
            ValueError,
            "topk",
        ),
        (
            "routing rows",
            lambda: layer.forward(x, tw[:-1], ids[:-1]),
            ValueError,
            "topk",
        ),
        (
            "ids float",
            lambda: layer.forward(x, tw, ids.float()),
            TypeError,
            "topk_ids",
        ),
        (
            "weights int",
            lambda: layer.forward(x, tw.to(torch.int32), ids),
            TypeError,
            "topk_weights",
        ),
        (
            "x on the CPU",
            lambda: layer.forward(x.cpu(), tw, ids),
            ValueError,
            "is on cpu",
        ),
        (
            "ids on the CPU",
            lambda: layer.forward(x, tw, ids.cpu()),
            ValueError,
            "is on cpu",
        ),
        (
            "M_r > max_local_tokens",
            lambda: layer.forward(*big),
            ValueError,
            "max_local_tokens",
        ),
        (
            "out of the whole batch",
            lambda: layer.forward(
                x,
                tw,
                ids,
                out=torch.empty((s["world"] * m, H), dtype=torch.bfloat16, device=dev),
            ),
            ValueError,
            "out",
        ),
        (
            "out dtype",
            lambda: layer.forward(x, tw, ids, out=x.float()),
            ValueError,
            "out",
        ),
        (
            "out not contiguous",
            lambda: layer.forward(
                x,
                tw,
                ids,
                out=torch.empty((H, m), dtype=torch.bfloat16, device=dev).t(),
            ),
            ValueError,
            "out",
        ),
        (
            "out on the CPU",
            lambda: layer.forward(
                x, tw, ids, out=torch.empty((m, H), dtype=torch.bfloat16)
            ),
            ValueError,
            "out",
        ),
        (
            "forward tail",
            lambda: layer.forward(x, tw, ids, tail=object()),
            ValueError,
            "tail",
        ),
    ]
    failures = []
    for name, fn, exc, text in cases:
        try:
            fn()
        except exc as e:
            if text not in str(e):
                failures.append(f"{name}: message {str(e)!r} does not mention {text!r}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: raised {type(e).__name__}: {e}")
        else:
            failures.append(f"{name}: no error")
    return failures


def _job_dp_mismatch(state):
    from aiter.ops.flydsl import moe_a8w4

    layer = state.get(DP_SHAPE)[0]
    m = 4 + int(state.rank == SHAPES[DP_SHAPE]["world"] - 1)
    x, tw, ids = state.inputs(DP_SHAPE, m, 0, share=False)
    saved, moe_a8w4.DEBUG_CHECKS = moe_a8w4.DEBUG_CHECKS, True
    try:
        layer.forward(x, tw, ids)
    except ValueError as e:
        return str(e)
    finally:
        moe_a8w4.DEBUG_CHECKS = saved
    return None


def test_dp_rejected_arguments(pool):
    failures = pool.run("dp_errors")[0]
    assert not failures, "\n".join(failures)
    for rank, msg in enumerate(pool.run("dp_mismatch")):
        assert msg is not None and "same M_r" in msg, (rank, msg)
    rows = _rank_rows(pool.run("dp_check", tokens=[4]))
    _assert_dp_verified(rows)


# --- D.8 the DP path checks themselves (CPU only) -----------------------------------------------------------------
# Why: as 2.10: the tests above only see the paths this code takes, so checks that accepted anything would pass them
# all. dp_comm's verdicts on the fused communication of later milestones and on broken paths (no / several /
# misplaced all-gathers or reduce-scatters, an all-reduce, no Gate/Up) are pinned here.
def test_dp_path_checks():
    mixed = f"{MIX}>a8w4_down_decode_mixed>shared_mixed"
    side = (
        f"a8w4_row_quant_decode>{PQ}>shared_fork>shared_split>a8w4_row_quant_decode>shared_join>"
        "a8w4_down_decode_pq_shared"
    )
    prefill = (
        "a8w4_stream_prep>shared_fold_gemm_gateup>shared_fold_mid>shared_fold_gemm_down>"
        f"{PF}>a8w4_row_quant_fp8>a8w4_down_prefill_bm64>sorted_sum"
    )
    cases = [  # path, dp_comm, _dp_known_path
        (f"{AG_CA}>{mixed}>{RS_CA}", (AG_CA, RS_CA), True),
        (f"{AG_RCCL}>{prefill}>{RS_RCCL}", (AG_RCCL, RS_RCCL), True),
        (f"{AG_RCCL}>{side}>{RS_CA}", (AG_RCCL, RS_CA), True),
        (f"a8w4_row_quant_fp8>{AG_CA}>{side}>{RS_CA}", (AG_CA, RS_CA), True),
        (f"{AG_CA}>{prefill}_tp", (AG_CA, FUSED_COMM), True),  # sorted_sum_tp
        (f"a8w4_quant_ag_tp>{side}_tp", (FUSED_COMM, FUSED_COMM), True),
        (f"{AG_CA}>{ST16}>a8w4_down_stream_bm16>{RS_CA}", (AG_CA, RS_CA), False),
        (f"{AG_CA}>{mixed}", None, False),  # no reduce-scatter
        (f"{mixed}>{RS_CA}", None, False),  # no all-gather
        (f"{AG_CA}>{mixed}>{CA}", None, False),  # an all-reduce
        (f"{AG_CA}>{mixed}>{RS_CA}>{RCCL}", None, False),
        (f"{AG_CA}>{AG_RCCL}>{mixed}>{RS_CA}", None, False),
        (f"{AG_CA}>{mixed}>{RS_CA}>{RS_RCCL}", None, False),
        (f"{mixed}>{AG_CA}>{RS_CA}", None, False),  # all-gather after Gate/Up
        (f"{AG_CA}>{RS_CA}>{mixed}", None, False),  # reduce-scatter before it
        (f"{AG_CA}>a8w4_down_decode>{RS_CA}", None, False),  # no Gate/Up
        (f"a8w4_quant_stp>{mixed}>{RS_CA}", None, False),  # no word "tp"
        (f"{mixed}>{CA}", None, False),  # the ar mode's path
    ]
    wrong = [
        (p, dp_comm(p), _dp_known_path(p))
        for p, *want in cases
        if [dp_comm(p), _dp_known_path(p)] != want
    ]
    assert not wrong, wrong
    assert comm_kind(f"{AG_CA}>{mixed}>{RS_CA}") is None


# --- D.9 routing dtypes and strides -------------------------------------------------------------------------------
# Why: the benchmark and the tests above pass contiguous int32 ids and float32 weights only. Callers may pass int64
# ids, BF16 weights or column views of a wider buffer; the ag_rs mode converts the routing to contiguous int32 /
# float32 before its all-gather (the ar mode hands it to the single-GPU op unchanged, which
# op_tests/test_flydsl_moe_a8w4.py covers). Each variant must give bitwise the output of the contiguous int32 /
# float32 routing of the same values (M_r 8).
DP_DTYPE_TOKENS = 8


def _job_dp_routing_dtypes(state):
    layer = state.get(DP_SHAPE)[0]
    x, tw, ids = _dp_mine(state, _dp_full(state, DP_DTYPE_TOKENS, 2))
    tw16, width = tw.to(torch.bfloat16), ids.shape[1]
    want = layer.forward(x, tw16.float(), ids)
    wide_tw = torch.cat([tw16.float(), tw16.float()], dim=1)
    wide_ids = torch.cat([ids, ids], dim=1)
    variants = {
        "int64 ids, bf16 weights": (tw16, ids.long()),
        "column views": (wide_tw[:, :width], wide_ids[:, :width]),
    }
    return {
        name: torch.equal(layer.forward(x, w, i), want)
        for name, (w, i) in variants.items()
    }


def test_dp_routing_dtypes(pool):
    for rank, res in enumerate(pool.run("dp_routing_dtypes")):
        assert len(res) == 2 and all(res.values()), (rank, res)


# =====================================================================================================================
# Region 3: benchmark (python <this file> [--cases ...]) -- after the last test
#
# The 51 SPEC cases {tp2, tp4, ds_layer_tp4} x M in {1, 2, 4, ..., 32768} + {48}. Cold-cache protocol: before every
# timed call a 512 MiB flush, an idle gap longer than any implementation's host issue time and a 128-byte custom
# all-reduce aligning the ranks; one call = forward + its cross-rank sum; per call the slowest rank; median of ITERS
# calls over GROUPS input groups. The implementations alternate call by call in one process:
#   mega_moe_tp_gfx942 (MF_METRIC): this implementation;
#   mega_tp_m7 (MF_REF, tp2 / tp4, the first MF_REF of those cases): M7's MegaMoeTPGfx942 (fused communication),
#       loaded from $MEGA_TP_M7_SRC (an aiter tree at tag mega-tp-a8w4-M7) by _M7Loader;
#   a8w4_shared_ar (MF_REF, every case): the tree's flydsl_moe_a8w4_ptpc and an all-reduce with the rule frozen in
#       REF_AR_RCCL_MIN_BYTES (the faster of aiter's custom all-reduce and RCCL; not imported from the
#       implementation).
# Every timed output passes the acceptance check against the FP32 reference and is bitwise identical over the ranks.
# flops per rank and call: tp2 / tp4 6 M topk H I, ds_layer_tp4 6 M H (topk I + I) (I the real per-rank inter_dim);
# bytes per rank and call: the selected experts' weights (ds_layer_tp4: and the shared expert's) in each
# implementation's own layout, x read and y written.
# =====================================================================================================================
OURS, M7, REF = "mega_moe_tp_gfx942", "mega_tp_m7", "a8w4_shared_ar"
BENCH_SHAPES = ("tp2", "tp4", "ds_layer_tp4")
M7_SHAPES = ("tp2", "tp4")
TOKENS = sorted({1 << i for i in range(16)} | {48})
CASES = [f"{shape}_m{M}" for shape in BENCH_SHAPES for M in TOKENS]
WARMUP, ITERS, GROUPS = 3, 30, 4
FLUSH_BYTES, IDLE_US = 512 << 20, 200
HBM_BYTES_PER_S, FP8_FLOPS = 5.3e12, 606e12  # MI308X
# a8w4_shared_ar's all-reduce: custom below these bytes (where should_custom_ar accepts), RCCL from there on.
REF_AR_RCCL_MIN_BYTES = {2: 6 << 20, 4: 13 << 20}
M7_ENV, M7_PACKAGE = "MEGA_TP_M7_SRC", "_mega_tp_m7"
# M7's own modules (aiter.ops.flydsl.<name>, and everything under the prefixes): always loaded from M7's files.
M7_OWN = (
    "mega_moe_tp_gfx942",
    "moe_a8w4",
    "mega_moe_tp",
    "communication_ops_host_utils",
)
M7_OWN_PREFIXES = ("kernels.moe_a8w4.", "kernels.mega_moe_tp_gfx942.")


class _M7Loader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Imports ``M7_PACKAGE.<name>`` from ``<src>/aiter/ops/flydsl/<name>``: M7_OWN modules, packages and modules
    whose file differs from the tree's are loaded from M7's file with ``aiter.ops.flydsl.`` in their source pointing
    to M7_PACKAGE (packages are always M7's: an alias would attach M7's submodules to the tree's package; the root
    package is left empty); every other module is an alias of the tree's byte-identical module (gfx9_fp8 common /
    moe_reduce, tensor_shim, communication utils, ...), sharing its state. Imports from outside aiter.ops.flydsl
    (aiter.dist's CustomAllreduce, aiter.fused_moe's moe_sorting, ...) are the tree's.
    """

    def __init__(self, src, tree):
        self.src, self.tree, self.aliases, self.specs = Path(src), Path(tree), {}, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != M7_PACKAGE and not fullname.startswith(M7_PACKAGE + "."):
            return None
        rel = fullname[len(M7_PACKAGE) + 1 :]
        base = self.src.joinpath(*rel.split(".")) if rel else self.src
        if (base / "__init__.py").is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                base / "__init__.py",
                loader=self,
                submodule_search_locations=[str(base)],
            )
        file = base.with_suffix(".py")
        if not file.is_file():
            return None
        tree = self.tree.joinpath(*rel.split(".")).with_suffix(".py")
        own = rel in M7_OWN or rel.startswith(M7_OWN_PREFIXES)
        if not own and tree.is_file() and tree.read_bytes() == file.read_bytes():
            self.aliases[fullname] = "aiter.ops.flydsl." + rel
        return importlib.util.spec_from_file_location(fullname, file, loader=self)

    def create_module(self, spec):
        if spec.name not in self.aliases:
            return None
        module = importlib.import_module(self.aliases[spec.name])
        self.specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        name = module.__spec__.name
        if name in self.aliases:  # the import system replaced the tree module's spec
            module.__spec__ = self.specs[name]
            return
        if name == M7_PACKAGE:
            return
        file = module.__spec__.origin
        src = re.sub(
            r"\baiter\.ops\.flydsl\.",
            M7_PACKAGE + ".",
            Path(file).read_text(encoding="utf-8"),
        )
        exec(compile(src, file, "exec"), module.__dict__)  # noqa: S102 (M7's source)


_M7_MODULE = None


def _load_m7():
    """M7's ``mega_moe_tp_gfx942`` module (once per process), from ``$MEGA_TP_M7_SRC`` (see _M7Loader)."""
    global _M7_MODULE
    if _M7_MODULE is None:
        src = os.environ.get(M7_ENV)
        if not src:
            raise RuntimeError(
                f"{M7_ENV} is not set (an aiter tree at mega-tp-a8w4-M7)"
            )
        import aiter.ops.flydsl as tree

        loader = _M7Loader(
            Path(src) / "aiter" / "ops" / "flydsl", Path(tree.__file__).parent
        )
        sys.meta_path.insert(0, loader)
        mod = importlib.import_module(f"{M7_PACKAGE}.mega_moe_tp_gfx942")
        for name in (f"{M7_PACKAGE}.mega_moe_tp_gfx942", f"{M7_PACKAGE}.moe_a8w4"):
            origin = Path(sys.modules[name].__file__).resolve()
            assert origin.is_relative_to(loader.src.resolve()), (name, origin)
        _M7_MODULE = mod
    return _M7_MODULE


class _RefAR:
    """a8w4_shared_ar's all-reduce: aiter's CustomAllreduce below REF_AR_RCCL_MIN_BYTES[world] bytes where its
    should_custom_ar accepts the tensor, RCCL otherwise. Its 128-byte custom all-reduce also aligns the ranks before
    every timed call."""

    def __init__(self, group, cpu_group, device, world):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce,
            IPCBufferPool,
        )

        self.group, self.cpu_group, self.device = group, cpu_group, device
        self.min_rccl = REF_AR_RCCL_MIN_BYTES[world]
        # The pool's store keys use a per-process counter; align it, as the ranks made different pools before.
        seq = torch.tensor([IPCBufferPool._pool_seq], dtype=torch.int64)
        dist.all_reduce(seq, op=dist.ReduceOp.MAX, group=cpu_group)
        IPCBufferPool._pool_seq = int(seq.item())
        self.ca = CustomAllreduce(group=cpu_group, device=device)
        assert not self.ca.disabled, "custom all-reduce disabled"
        self.tiny = torch.ones(64, dtype=torch.bfloat16, device=device)
        self.last = None

    def __call__(self, y):
        if y.numel() * y.element_size() < self.min_rccl and self.ca.should_custom_ar(y):
            self.last = CA
            return self.ca.custom_all_reduce(y)
        self.last = RCCL
        dist.all_reduce(y, group=self.group)
        return y

    def barrier(self):
        self.ca.custom_all_reduce(self.tiny)

    def close(self):
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.ca.close()


def _weight_bytes(weights):
    """(bytes per routed expert, bytes of the shared expert) of a prepared weight object, from its tensor fields
    (``shared_*``: the shared expert)."""
    routed = shared = 0
    for name, t in vars(weights).items():
        if isinstance(t, torch.Tensor):
            if name.startswith("shared_"):
                shared += t.numel() * t.element_size()
            else:
                routed += t.numel() * t.element_size()
    return routed / weights.experts, shared


def _cycles_per_us():
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda._sleep(1000)
    start.record()
    torch.cuda._sleep(1_000_000)
    end.record()
    torch.cuda.synchronize()
    return 1_000_000 / (start.elapsed_time(end) * 1e3)


def _host_us(fn, cycles_per_us):
    """Median host time to issue one call, with the GPU idle long enough that the queue never pushes back."""
    times = []
    for i in range(5):
        torch.cuda._sleep(int(20_000 * cycles_per_us))
        began = time.perf_counter()
        fn(i % GROUPS)
        times.append((time.perf_counter() - began) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(times)


def _q(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def _job_bench_setup(state, shape):
    state.release()
    if not state.member(shape):
        return None
    from aiter.ops.flydsl import moe_a8w4

    began = time.time()
    s = SHAPES[shape]
    state.layers[shape] = state.build(shape)
    _, raw, shared = state.layers[shape]
    group, cpu = state.tp[s["world"]]
    bench = state.bench = {
        "shape": shape,
        "m7": None,
        "ar": _RefAR(group, cpu, state.device, s["world"]),
        "ref": moe_a8w4.prepare_moe_a8w4_ptpc_weight(
            raw["w1"],
            raw["w1_scale"],
            raw["w2"],
            raw["w2_scale"],
            **shared_kwargs(shared),
        ),
        "cycles": _cycles_per_us(),
        "flush": torch.empty(
            FLUSH_BYTES // 4, dtype=torch.float32, device=state.device
        ),
    }
    m7_file = ""
    if shape in M7_SHAPES:
        m7 = _load_m7()
        bench["m7"] = m7.MegaMoeTPGfx942(
            m7.MegaMoeTPConfig(**config_args(shape, state.rank)),
            **to_megamoe_layout(raw),
            group=group,
            device=state.device,
            cpu_group=cpu,
        )
        m7_file = m7.__file__
    torch.cuda.synchronize(state.device)
    return {
        "took": round(time.time() - began, 1),
        "m7": m7_file,
        "cycles_per_us": round(bench["cycles"], 1),
    }


def _job_bench(state, case, warmup, iters):
    shape, M = parse_case(case)
    if not state.member(shape):
        return None
    bench = state.bench
    if bench is None or bench["shape"] != shape:
        raise RuntimeError(f"bench_setup({shape!r}) has not run")
    from aiter.ops.flydsl import moe_a8w4

    s, dev = SHAPES[shape], state.device
    layer, raw, shared = state.layers[shape]
    group, cpu = state.tp[s["world"]]
    ar, m7, ref_w, H, k = bench["ar"], bench["m7"], bench["ref"], s["H"], s["topk"]
    groups = [state.inputs(shape, M, g) for g in range(GROUPS)]
    bufs = [
        torch.empty((M, H), dtype=torch.bfloat16, device=dev) for _ in range(GROUPS)
    ]

    def ref(g):
        x, tw, ids = groups[g]
        moe_a8w4.flydsl_moe_a8w4_ptpc(
            x, ref_w, tw, ids, output=bufs[g], swiglu_limit=s["limit"] or 0.0
        )
        return ar(bufs[g])

    impls, weights = {OURS: lambda g: layer.forward(*groups[g])}, {OURS: layer._weights}
    if m7 is not None:
        impls[M7], weights[M7] = (lambda g: m7.forward(*groups[g])), m7._weights
    impls[REF], weights[REF] = ref, ref_w

    def path_of(impl):
        if impl == OURS:
            return ">".join(layer.last_path)
        if impl == M7:
            return ">".join(m7.last_path)
        return ">".join([*moe_a8w4.last_path, ar.last])

    last, paths = {n: [None] * GROUPS for n in impls}, {}
    dist.barrier(group=cpu)
    for i in range(warmup):
        for impl, fn in impls.items():
            ar.barrier()
            last[impl][i % GROUPS] = fn(i % GROUPS)
            paths.setdefault(impl, path_of(impl))
        torch.cuda.synchronize(dev)
        dist.barrier(group=cpu)
    host = {
        impl: _host_us(
            lambda g, fn=fn, impl=impl: (
                ar.barrier(),
                last[impl].__setitem__(g, fn(g)),
            ),
            bench["cycles"],
        )
        for impl, fn in impls.items()
    }
    idle = torch.tensor([max(IDLE_US, 2 * max(host.values()))], dtype=torch.float64)
    dist.all_reduce(idle, op=dist.ReduceOp.MAX, group=cpu)
    idle_us = float(idle.item())
    sleep_cycles = int(idle_us * bench["cycles"])
    torch.cuda.synchronize(dev)
    dist.barrier(group=cpu)
    events = {impl: [] for impl in impls}
    for i in range(iters):
        g = i % GROUPS
        for impl, fn in impls.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            bench["flush"].zero_()
            torch.cuda._sleep(sleep_cycles)
            ar.barrier()
            start.record()
            last[impl][g] = fn(g)
            end.record()
            events[impl].append((start, end))
    torch.cuda.synchronize(dev)

    res = {}
    for impl, pairs in events.items():
        mine = torch.tensor(
            [a.elapsed_time(b) * 1e3 for a, b in pairs], dtype=torch.float64, device=dev
        )
        every = [torch.empty_like(mine) for _ in range(s["world"])]
        dist.all_gather(every, mine, group=group)
        worst = torch.stack(every).max(0).values.tolist()
        per_expert, shared_bytes = _weight_bytes(weights[impl])
        res[impl] = {
            "us": statistics.median(worst),
            "p10": _q(worst, 0.1),
            "p90": _q(worst, 0.9),
            "rank0_us": statistics.median(every[0].tolist()),
            "path": paths[impl],
            "host_us": host[impl],
            "ok": True,
            "identical": True,
            "inputs_identical": True,
            "calc_diff": 0.0,
            "err_ratio": 0.0,
            "error_flag": 0,
            "bytes": statistics.mean(
                ids[:, :k].unique().numel() * per_expert + shared_bytes + M * H * 2 * 2
                for _, _, ids in groups
            ),
        }
    for impl, obj in ((OURS, layer), (M7, m7)):
        if impl in res:
            res[impl]["error_flag"] = max(
                int(obj.error_flag().abs().max().item()), int(obj.poll_errors())
            )
    # every timed output: the FP32 reference summed over the group, bitwise identity over the ranks
    for g, inputs in enumerate(groups):
        ref_y = _group_reference(state, shape, raw, shared, inputs)
        same_inputs = all(_identical(t, group) for t in inputs)
        for impl in impls:
            st, r = res[impl], check(last[impl][g], ref_y)
            st["ok"] = st["ok"] and bool(r["ok"])
            st["calc_diff"] = max(st["calc_diff"], r["calc_diff"])
            st["err_ratio"] = max(st["err_ratio"], r["err_ratio"])
            st["identical"] = st["identical"] and _identical(last[impl][g], group)
            st["inputs_identical"] = st["inputs_identical"] and same_inputs
        del ref_y
    inter = (k + 1) * s["I"] if s["shared"] else k * s["I"]
    return {
        "case": case,
        "flops": 6 * M * H * inter,
        "idle_us": idle_us,
        "impls": res,
    }


def _report(case, r, failures):
    shape, M = parse_case(case)
    s, flops, impls = SHAPES[shape], r["flops"], r["impls"]
    ours = impls[OURS]

    def good(st):
        return (
            st["ok"]
            and st["identical"]
            and st["inputs_identical"]
            and not st["error_flag"]
        )

    def util(st):
        t = st["us"] * 1e-6
        bw, fl = st["bytes"] / t, flops / t
        return (
            f"{bw / 1e12:.2f}TB/s({bw / HBM_BYTES_PER_S * 100:.1f}%),"
            f"{fl / 1e12:.1f}TFLOPS({fl / FP8_FLOPS * 100:.1f}%)"
        )

    if good(ours):
        print(
            f"MF_METRIC case={case} value={ours['us']:.2f} unit=us flops={flops} bytes={ours['bytes']:.0f}",
            flush=True,
        )
    else:
        failures.append(case)
        print(f"FAIL case={case} impl={OURS} {ours}", flush=True)
    for impl in (M7, REF):  # mf keeps the first MF_REF impl of a case: M7 first
        if impl not in impls:
            continue
        st, extra = impls[impl], ""
        if not good(st):
            extra = " ok=0"
            failures.append(f"{case}:{impl}")
            print(f"FAIL case={case} impl={impl} {st}", flush=True)
        print(
            f"MF_REF case={case} impl={impl} value={st['us']:.2f} unit=us flops={flops} bytes={st['bytes']:.0f} "
            f"ratio={ours['us'] / st['us']:.3f}{extra}",
            flush=True,
        )
    w = s["world"]
    # bytes a rank sends (and receives) in a ring-equivalent all-reduce
    xgmi = 2 * (w - 1) / w * M * s["H"] * 2
    times = " ".join(
        f"{impl}={st['us']:.2f}us(p10={st['p10']:.2f},p90={st['p90']:.2f},rank0={st['rank0_us']:.2f},"
        f"{util(st)})"
        for impl, st in impls.items()
    )
    ratios = " ".join(
        f"ratio_{impl}={ours['us'] / st['us']:.3f}"
        for impl, st in impls.items()
        if impl != OURS
    )
    refs = " ".join(
        f"path_{impl}={st['path']}" for impl, st in impls.items() if impl != OURS
    )
    host = ",".join(f"{impl}:{st['host_us']:.0f}" for impl, st in impls.items())
    print(
        f"INFO case={case} I={s['I']} path={ours['path']} {times} {ratios} "
        f"check=calc_diff:{ours['calc_diff']:.2e},err_ratio:{ours['err_ratio'] * 100:.2f}%,"
        f"identical:{int(ours['identical'])},inputs_identical:{int(ours['inputs_identical'])},"
        f"error_flag:{ours['error_flag']} xgmi_bytes={xgmi:.0f} xgmi_GBps={xgmi / ours['us'] / 1e3:.1f} "
        f"host_us={host} idle_us={r['idle_us']:.0f} {refs}",
        flush=True,
    )


# --- DP cases: ds_dp4 (comm_mode ag_rs) x M_r in {1, 2, 4, ..., 8192} + {12}, after the 51 AR cases -------------------
# The same protocol; one call = forward of this rank's M_r tokens (all-gather, the layer on all world * M_r rows,
# reduce-scatter), per call the slowest rank. Inputs: the whole batch (world * M_r real tokens) generated on rank 0 and
# broadcast, rank r passes its rows [r * M_r, (r + 1) * M_r). The implementations alternate call by call:
#   mega_moe_tp_gfx942 (MF_METRIC): this implementation, comm_mode ag_rs;
#   a8w4_shared_agrs (MF_REF, the only one): the tree's flydsl_moe_a8w4_ptpc on the gathered rows between an all-gather
#       and a reduce-scatter with the rules frozen in REF_AG_RCCL_MIN_BYTES / REF_RS_RCCL_MIN_BYTES (the faster of
#       aiter's custom collectives and RCCL; not imported from the implementation).
# Every timed output passes the acceptance check against this rank's rows of the FP32 reference (the group's sum over
# the whole batch), and a second call on each input group gives bitwise the same output. flops per rank and call:
# 6 (world M_r) H (topk I + I); bytes per rank and call: the weights of the experts the whole batch selects and the
# shared expert's in each implementation's own layout, x_local read and y written.
DP_REF = "a8w4_shared_agrs"
DP_TOKENS = sorted({1 << i for i in range(14)} | {12})
DP_CASES = [f"{DP_SHAPE}_m{m}" for m in DP_TOKENS]
# a8w4_shared_agrs's all-gather: one custom_all_gatherv of x and the routing below these bytes of x per rank (where it
# takes them), else RCCL for x and the custom gather (RCCL where it declines) for the routing; its reduce-scatter:
# custom below these bytes of partials (where should_custom_rs accepts), RCCL from there on.
REF_AG_RCCL_MIN_BYTES = {4: 6 << 20}
REF_RS_RCCL_MIN_BYTES = {4: 26 << 20}


class _RefAGRS(_RefAR):
    """a8w4_shared_agrs's collectives on ``_RefAR``'s custom communicator (whose 128-byte all-reduce also aligns the
    ranks before every timed call)."""

    def __init__(self, group, cpu_group, device, world):
        super().__init__(group, cpu_group, device, world)
        self.world = world
        self.min_rccl_ag = REF_AG_RCCL_MIN_BYTES[world]
        self.min_rccl_rs = REF_RS_RCCL_MIN_BYTES[world]
        self.last_ag = self.last_rs = None

    def all_gather(self, x, ids, tw):
        m, world = x.shape[0], self.world
        if x.numel() * x.element_size() < self.min_rccl_ag:
            got = self.ca.custom_all_gatherv([x, ids, tw])
            if got is not None:
                self.last_ag = AG_CA
                return tuple(got)
        xs = x.new_empty((world * m, x.shape[1]))
        dist.all_gather_into_tensor(xs, x, group=self.group)
        got = self.ca.custom_all_gatherv([ids, tw])
        if got is None:
            k = ids.shape[1]
            every = ids.new_empty((world * m, 2 * k))
            packed = torch.cat([ids, tw.view(torch.int32)], dim=1)
            dist.all_gather_into_tensor(every, packed, group=self.group)
            got = (
                every[:, :k].contiguous(),
                every[:, k:].contiguous().view(torch.float32),
            )
        self.last_ag = AG_RCCL
        return (xs, *got)

    def reduce_scatter(self, part, out):
        nbytes = part.numel() * part.element_size()
        if nbytes < self.min_rccl_rs and self.ca.should_custom_rs(part, 0):
            self.ca.custom_reduce_scatter(part, out, 0)
            self.last_rs = RS_CA
        else:
            dist.reduce_scatter_tensor(out, part, group=self.group)
            self.last_rs = RS_RCCL
        return out


def _job_bench_dp_setup(state):
    state.release()
    if not state.member(DP_SHAPE):
        return None
    from aiter.ops.flydsl import moe_a8w4

    began = time.time()
    s = SHAPES[DP_SHAPE]
    state.layers[DP_SHAPE] = state.build(DP_SHAPE)
    _, raw, shared = state.layers[DP_SHAPE]
    group, cpu = state.tp[s["world"]]
    bench = state.bench = {
        "shape": DP_SHAPE,
        "m7": None,
        "ar": _RefAGRS(group, cpu, state.device, s["world"]),
        "ref": moe_a8w4.prepare_moe_a8w4_ptpc_weight(
            raw["w1"],
            raw["w1_scale"],
            raw["w2"],
            raw["w2_scale"],
            **shared_kwargs(shared),
        ),
        "cycles": _cycles_per_us(),
        "flush": torch.empty(
            FLUSH_BYTES // 4, dtype=torch.float32, device=state.device
        ),
    }
    torch.cuda.synchronize(state.device)
    return {
        "took": round(time.time() - began, 1),
        "cycles_per_us": round(bench["cycles"], 1),
    }


def _job_bench_dp(state, case, warmup, iters):
    shape, m = parse_case(case)
    if not state.member(shape):
        return None
    bench = state.bench
    if shape != DP_SHAPE or bench is None or bench["shape"] != shape:
        raise RuntimeError(f"bench_dp_setup has not run for {case!r}")
    from aiter.ops.flydsl import moe_a8w4

    s, dev = SHAPES[shape], state.device
    layer, raw, shared = state.layers[shape]
    group, cpu = state.tp[s["world"]]
    comm, ref_w, H, k, world = bench["ar"], bench["ref"], s["H"], s["topk"], s["world"]
    fulls = [_dp_full(state, m, g) for g in range(GROUPS)]
    groups = [_dp_mine(state, full) for full in fulls]
    parts = [
        torch.empty((world * m, H), dtype=torch.bfloat16, device=dev)
        for _ in range(GROUPS)
    ]
    outs = [
        torch.empty((m, H), dtype=torch.bfloat16, device=dev) for _ in range(GROUPS)
    ]

    def ref(g):
        x, tw, ids = groups[g]
        xs, ids_all, tw_all = comm.all_gather(x, ids, tw)
        moe_a8w4.flydsl_moe_a8w4_ptpc(
            xs,
            ref_w,
            tw_all,
            ids_all,
            output=parts[g],
            swiglu_limit=s["limit"] or 0.0,
        )
        return comm.reduce_scatter(parts[g], outs[g])

    impls = {OURS: lambda g: layer.forward(*groups[g]), DP_REF: ref}
    weights = {OURS: layer._weights, DP_REF: ref_w}

    def path_of(impl):
        if impl == OURS:
            return ">".join(layer.last_path)
        return ">".join([comm.last_ag, *moe_a8w4.last_path, comm.last_rs])

    last, paths = {n: [None] * GROUPS for n in impls}, {}
    dist.barrier(group=cpu)
    for i in range(warmup):
        for impl, fn in impls.items():
            comm.barrier()
            last[impl][i % GROUPS] = fn(i % GROUPS)
            paths.setdefault(impl, path_of(impl))
        torch.cuda.synchronize(dev)
        dist.barrier(group=cpu)
    host = {
        impl: _host_us(
            lambda g, fn=fn, impl=impl: (
                comm.barrier(),
                last[impl].__setitem__(g, fn(g)),
            ),
            bench["cycles"],
        )
        for impl, fn in impls.items()
    }
    idle = torch.tensor([max(IDLE_US, 2 * max(host.values()))], dtype=torch.float64)
    dist.all_reduce(idle, op=dist.ReduceOp.MAX, group=cpu)
    idle_us = float(idle.item())
    sleep_cycles = int(idle_us * bench["cycles"])
    torch.cuda.synchronize(dev)
    dist.barrier(group=cpu)
    events = {impl: [] for impl in impls}
    for i in range(iters):
        g = i % GROUPS
        for impl, fn in impls.items():
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            bench["flush"].zero_()
            torch.cuda._sleep(sleep_cycles)
            comm.barrier()
            start.record()
            last[impl][g] = fn(g)
            end.record()
            events[impl].append((start, end))
    torch.cuda.synchronize(dev)

    res = {}
    for impl, pairs in events.items():
        mine = torch.tensor(
            [a.elapsed_time(b) * 1e3 for a, b in pairs], dtype=torch.float64, device=dev
        )
        every = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(every, mine, group=group)
        worst = torch.stack(every).max(0).values.tolist()
        per_expert, shared_bytes = _weight_bytes(weights[impl])
        res[impl] = {
            "us": statistics.median(worst),
            "p10": _q(worst, 0.1),
            "p90": _q(worst, 0.9),
            "rank0_us": statistics.median(every[0].tolist()),
            "path": paths[impl],
            "host_us": host[impl],
            "ok": True,
            "repeatable": True,
            "inputs_identical": True,
            "calc_diff": 0.0,
            "err_ratio": 0.0,
            "error_flag": 0,
            "bytes": statistics.mean(
                ids[:, :k].unique().numel() * per_expert + shared_bytes + m * H * 2 * 2
                for _, _, ids in fulls
            ),
        }
    res[OURS]["error_flag"] = max(
        int(layer.error_flag().abs().max().item()), int(layer.poll_errors())
    )
    # every timed output: this rank's rows of the FP32 reference; then a second call on the same inputs
    for g, full in enumerate(fulls):
        ref_y = _dp_reference(state, raw, shared, full)
        same_inputs = all(_identical(t, group) for t in full)
        for impl, fn in impls.items():
            st, y = res[impl], last[impl][g].clone()
            r = check(y, ref_y)
            st["ok"] = st["ok"] and bool(r["ok"])
            st["calc_diff"] = max(st["calc_diff"], r["calc_diff"])
            st["err_ratio"] = max(st["err_ratio"], r["err_ratio"])
            st["inputs_identical"] = st["inputs_identical"] and same_inputs
            st["repeatable"] = st["repeatable"] and torch.equal(fn(g), y)
        del ref_y
    return {
        "case": case,
        "flops": 6 * world * m * H * (k + 1) * s["I"],
        "idle_us": idle_us,
        "impls": res,
    }


def _report_dp(case, r, failures):
    shape, m = parse_case(case)
    s, flops, impls = SHAPES[shape], r["flops"], r["impls"]
    ours, ref = impls[OURS], impls[DP_REF]

    def good(st):
        return (
            st["ok"]
            and st["repeatable"]
            and st["inputs_identical"]
            and not st["error_flag"]
        )

    def util(st):
        t = st["us"] * 1e-6
        bw, fl = st["bytes"] / t, flops / t
        return (
            f"{bw / 1e12:.2f}TB/s({bw / HBM_BYTES_PER_S * 100:.1f}%),"
            f"{fl / 1e12:.1f}TFLOPS({fl / FP8_FLOPS * 100:.1f}%)"
        )

    if good(ours):
        print(
            f"MF_METRIC case={case} value={ours['us']:.2f} unit=us flops={flops} bytes={ours['bytes']:.0f}",
            flush=True,
        )
    else:
        failures.append(case)
        print(f"FAIL case={case} impl={OURS} {ours}", flush=True)
    extra = ""
    if not good(ref):
        extra = " ok=0"
        failures.append(f"{case}:{DP_REF}")
        print(f"FAIL case={case} impl={DP_REF} {ref}", flush=True)
    print(
        f"MF_REF case={case} impl={DP_REF} value={ref['us']:.2f} unit=us flops={flops} bytes={ref['bytes']:.0f} "
        f"ratio={ours['us'] / ref['us']:.3f}{extra}",
        flush=True,
    )
    w = s["world"]
    # bytes a rank sends (and receives): the all-gather of x and the reduce-scatter of the partials, BF16
    xgmi = 2 * (w - 1) * m * s["H"] * 2
    times = " ".join(
        f"{impl}={st['us']:.2f}us(p10={st['p10']:.2f},p90={st['p90']:.2f},rank0={st['rank0_us']:.2f},"
        f"{util(st)})"
        for impl, st in impls.items()
    )
    host = ",".join(f"{impl}:{st['host_us']:.0f}" for impl, st in impls.items())
    print(
        f"INFO case={case} I={s['I']} path={ours['path']} {times} ratio_{DP_REF}={ours['us'] / ref['us']:.3f} "
        f"check=calc_diff:{ours['calc_diff']:.2e},err_ratio:{ours['err_ratio'] * 100:.2f}%,"
        f"repeatable:{int(ours['repeatable'])},inputs_identical:{int(ours['inputs_identical'])},"
        f"error_flag:{ours['error_flag']} xgmi_bytes={xgmi:.0f} xgmi_GBps={xgmi / ours['us'] / 1e3:.1f} "
        f"host_us={host} idle_us={r['idle_us']:.0f} path_{DP_REF}={ref['path']}",
        flush=True,
    )


def _bench_main(argv=None):
    parser = argparse.ArgumentParser(
        description="MegaMoeTPGfx942 benchmark: SPEC cases, cold-cache protocol, slowest rank; references "
        f"{M7} (tp2 / tp4, from ${M7_ENV}), {REF} and {DP_REF} ({DP_SHAPE}, comm_mode ag_rs)"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES + DP_CASES,
        default=CASES + DP_CASES,
        metavar="CASE",
    )
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be >= 1")
    if not os.environ.get(M7_ENV) and any(
        parse_case(c)[0] in M7_SHAPES for c in args.cases
    ):
        parser.error(f"{M7_ENV} must name M7's aiter tree for the {M7} reference")
    began = time.time()
    print(
        f"INFO seed={SEED} warmup={args.warmup} iters={args.iters} groups={GROUPS} flush_bytes={FLUSH_BYTES} "
        f"m7_src={os.environ.get(M7_ENV, '')} ref_ar_rccl_min_bytes={REF_AR_RCCL_MIN_BYTES} "
        f"cases={len(args.cases)}",
        flush=True,
    )
    failures = []
    p = _Pool()
    try:
        gfx = p.start()
        if gfx != "gfx942":
            raise SystemExit(f"MegaMoeTPGfx942 runs on gfx942 only (found {gfx})")
        for shape in BENCH_SHAPES:
            names = [c for c in args.cases if parse_case(c)[0] == shape]
            if not names:
                continue
            setup = p.run("bench_setup", shape=shape)[0]
            print(f"INFO setup shape={shape} {setup}", flush=True)
            for case in names:
                t0 = time.time()
                _report(
                    case,
                    p.run("bench", case=case, warmup=args.warmup, iters=args.iters)[0],
                    failures,
                )
                print(f"INFO case={case} took={time.time() - t0:.0f}s", flush=True)
        dp_names = [c for c in args.cases if c in DP_CASES]
        if dp_names:
            print(
                f"INFO dp ref={DP_REF} ref_ag_rccl_min_bytes={REF_AG_RCCL_MIN_BYTES} "
                f"ref_rs_rccl_min_bytes={REF_RS_RCCL_MIN_BYTES} cases={len(dp_names)}",
                flush=True,
            )
            setup = p.run("bench_dp_setup")[0]
            print(f"INFO setup shape={DP_SHAPE} {setup}", flush=True)
            for case in dp_names:
                t0 = time.time()
                _report_dp(
                    case,
                    p.run("bench_dp", case=case, warmup=args.warmup, iters=args.iters)[
                        0
                    ],
                    failures,
                )
                print(f"INFO case={case} took={time.time() - t0:.0f}s", flush=True)
        p.close()
    finally:
        p.kill()
    print(f"INFO done in {time.time() - began:.0f}s failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _bench_main()
