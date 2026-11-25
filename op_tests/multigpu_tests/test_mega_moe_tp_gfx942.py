# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests and benchmark of ``MegaMoeTPGfx942`` (aiter/ops/flydsl/mega_moe_tp_gfx942.py), gfx942, ``comm_mode="ar"``.

One file, three regions:

1. basic tests (``pytest <file> -m basic``): one case per dispatch tier and shape; run first after every change.
2. other coverage (``pytest <file> -m "not basic"``): what the basic tests and the benchmark cannot cover; every
   test says why.
3. benchmark (``python <file> [--cases ...]``, after the last test): the SPEC cases {tp2, tp4, ds_tp4} x M in
   {1, 2, 4, ..., 32768} + {48}; cold-cache protocol, slowest rank; prints ``MF_METRIC`` / ``MF_REF``.

Every rank of a TP group gets the same input rows and routing (generated on rank 0 and broadcast; asserted bitwise
identical) and its own MXFP4 weight slice (independent random weights per rank). The reference is independent FP32
torch: per rank the exact MXFP4 dequantization, FP8 per-token activations and FP8 per-(token, slot) intermediate
activations (scale = amax / 240), summed over the ranks of the group. Acceptance: finite, calc_diff <= 1e-3 and
max|y - ref| <= 4.5% max|ref|; the outputs of all ranks are bitwise identical.

Multi-GPU runs go through a pool of 4 spawned worker processes (rank r on GPU r, world group + TP2 sub-group),
started once per session; a test sends a job (``_job_<name>``) to every rank and checks the returned results.
"""

import argparse
import multiprocessing
import os
import queue
import shutil
import socket
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

SEED = int(os.environ.get("MF_SEED", "0"))
SHAPES = {
    # world, hidden, experts, topk, real per-rank inter_dim, fused shared expert (last slot), swiglu_limit
    "tp2": {
        "world": 2,
        "H": 2560,
        "E": 513,
        "topk": 11,
        "I": 320,
        "shared": True,
        "limit": None,
    },  # Qwen3.8-Flash-Next TP2
    "tp4": {
        "world": 4,
        "H": 2560,
        "E": 513,
        "topk": 11,
        "I": 160,
        "shared": True,
        "limit": None,
    },  # Qwen3.8-Flash-Next TP4
    "ds_tp4": {
        "world": 4,
        "H": 5120,
        "E": 384,
        "topk": 6,
        "I": 576,
        "shared": False,
        "limit": 10.0,
    },  # DeepSeek-V4.1-Flash TP4
}
SHAPE_INDEX = {name: i for i, name in enumerate(SHAPES)}
TOKENS = tuple(sorted({1 << k for k in range(16)} | {48}))
CASES = [f"{shape}_m{m}" for shape in SHAPES for m in TOKENS]
MAX_TOKENS = 1 << 15
POOL_WORLD = 4

FP8_MAX = 240.0
CALC_DIFF_MAX = 1e-3
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

# Kernel sequences of moe_a8w4.fused_moe_a8w4 (layer.last_path without the all-reduce) on the unfused paths.
DEC = "a8w4_gateup_decode>a8w4_down_decode"
DPQ = "a8w4_gateup_decode>a8w4_row_quant_decode>a8w4_down_decode_pq"
PQ = "a8w4_row_quant_decode>a8w4_gateup_decode_pq>a8w4_row_quant_decode>a8w4_down_decode_pq"
DD = "a8w4_row_quant_plan_decode>a8w4_gateup_decode_dd>a8w4_row_quant_decode>a8w4_down_decode_pq"
ST16 = (
    "a8w4_stream_prep>a8w4_gateup_stream_bm16>a8w4_row_quant_fp8>a8w4_down_stream_bm16"
)
ST16_RS = ST16 + ">a8w4_route_sum"
ST32Q = "a8w4_stream_prep>a8w4_gateup_quant_stream_bm32>a8w4_down_stream_bm32"
ST64Q = "a8w4_stream_prep>a8w4_gateup_quant_stream_bm64>a8w4_down_stream_bm64"
ST32_RS = "a8w4_stream_prep>a8w4_gateup_stream_bm32>a8w4_row_quant_fp8>a8w4_down_stream_bm32>a8w4_route_sum"
PF_PREP = (
    "a8w4_stream_prep>a8w4_gateup_prefill_bm64_bn128_tk128>a8w4_row_quant_fp8>a8w4_down_prefill_bm64"
    ">sorted_sum"
)
PF_SORT = (
    "moe_sorting_opus>a8w4_row_quant_fp8>a8w4_gateup_prefill_bm64_bn128_tk128>a8w4_row_quant_fp8"
    ">a8w4_down_prefill_bm64>sorted_sum"
)
COMPUTE_PATHS = (
    DEC,
    DPQ,
    PQ,
    DD,
    ST16,
    ST16_RS,
    ST32Q,
    ST64Q,
    ST32_RS,
    PF_PREP,
    PF_SORT,
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


# Path checks. The tests pin a forward's dispatch tier -- its Gate/Up kernel, the first one named a8w4_gateup* -- and
# require exactly one communication segment: one all-reduce at the end (CA / RCCL: unfused), or at least one fused
# communication kernel and no all-reduce (fused). A fused communication kernel has the word "tp" in its
# "_"-separated name (a8w4_down_decode_tp, a8w4_tp_reduce, sorted_sum_tp). The kernels after Gate/Up are not
# pinned: fusing the TP reduction changes them.
GATEUP_PREFIX, TP_WORD, FUSED_COMM = "a8w4_gateup", "tp", "fused"


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


GATEUP_KERNELS = frozenset(map(gateup_tier, COMPUTE_PATHS))


def _known_path(path):
    """A path of this implementation: a known Gate/Up kernel (tier) and exactly one communication segment."""
    return gateup_tier(path) in GATEUP_KERNELS and comm_kind(path) is not None


# =====================================================================================================================
# Weights, inputs and the independent FP32 reference (torch only, no kernel under test)
# =====================================================================================================================
def make_weights(experts, hidden, inter, *, seed, fp8_inter=None, device="cuda"):
    """Random BF16 MoE weights (generated expert by expert), quantized to standard MXFP4.

    Returns ``w1`` float4_e2m1fn_x2 [E, 2I, H/2] (rows [0, I) gate, [I, 2I) up), ``w1_scale`` float8_e8m0fnu
    [E, 2I, H/32], ``w2`` [E, H, I/2], ``w2_scale`` [E, H, I/32]. With ``fp8_inter`` (>= inter) also the same BF16
    weights with I zero-padded to ``fp8_inter``, quantized per output channel to FP8 E4M3FNUZ (not shuffled):
    ``w1_fp8`` [E, 2P, H], ``w1_fp8_scale`` [E, 2P, 1], ``w2_fp8`` [E, H, P], ``w2_fp8_scale`` [E, H, 1] (float32).
    Standard deviations: w1 1/sqrt(H), w2 1/sqrt(I).
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
    fp8 = torch.float8_e4m3fnuz
    if fp8_inter is not None:
        assert fp8_inter >= inter
        p = fp8_inter
        pt = aiter.get_torch_quant(aiter.QuantType.per_Token)
        out.update(
            w1_fp8=torch.empty((experts, 2 * p, h), dtype=fp8, device=device),
            w1_fp8_scale=torch.empty(
                (experts, 2 * p, 1), dtype=torch.float32, device=device
            ),
            w2_fp8=torch.empty((experts, h, p), dtype=fp8, device=device),
            w2_fp8_scale=torch.empty(
                (experts, h, 1), dtype=torch.float32, device=device
            ),
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


def make_inputs(tokens, hidden, experts, topk, *, seed, shared, device="cuda"):
    """``(x, topk_weight, topk_ids)``: x BF16 [M, H] ~ N(0, 1).

    ``shared`` (Qwen3.8): the last expert (E-1) is the fused shared expert, always in a token's last slot; the
    other topk-1 are uniform without repetition in [0, E-1), weighted by the softmax of topk-1 random logits, the
    shared one by sigmoid(randn). Otherwise topk distinct uniform experts in [0, E), softmax weights.
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
    last = torch.full((tokens, 1), experts - 1, dtype=routed.dtype, device=device)
    ids = torch.cat([routed, last], dim=1).to(torch.int32).contiguous()
    logits = torch.randn((tokens, topk - 1), generator=gen, device=device)
    shared_w = torch.sigmoid(torch.randn((tokens, 1), generator=gen, device=device))
    weights = (
        torch.cat([torch.softmax(logits, dim=-1), shared_w], dim=1).float().contiguous()
    )
    return x, weights, ids


def dequant_mxfp4(q, scale):
    """Standard MXFP4 [..., N, K/2] + E8M0 [..., N, K/32] -> exact float32 [..., N, K]."""
    b = q.view(torch.uint8)
    nib = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).long()
    value = torch.tensor(_E2M1, dtype=torch.float32, device=b.device)[nib]
    factor = torch.exp2(scale.view(torch.uint8).to(torch.float64) - 127).float()
    return (value.unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)).flatten(-2)


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
    """Independent FP32 MoE of one rank's weights -> float32 [M, H].

    Standard MXFP4 weights (``make_weights``), FP8 per-token input and FP8 per-(token, slot) intermediate,
    SiLU(gate) * up, routing weights applied after Down and summed per token. ``swiglu_limit`` L > 0 is the
    DeepSeek-V4.1 SwiGLU clamp on the f32 gate / up: gate = min(gate, L), up = clamp(up, -L, L).
    """
    M, _ = x.shape
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
    out = torch.zeros((M, x.shape[1]), dtype=torch.float32, device=x.device)
    for e, t, s in groups:
        y = mid[t, s] @ dequant_mxfp4(w2[e], w2_scale[e]).T
        out.index_add_(0, t, y * topk_weight[t, s].float().unsqueeze(-1))
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


def make_config(shape, rank, **kw):
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPConfig

    s = SHAPES[shape]
    args = {
        "rank": rank,
        "world_size": s["world"],
        "model_dim": s["H"],
        "inter_dim": s["I"],
        "experts": s["E"],
        "topk": s["topk"],
        "max_local_tokens": MAX_TOKENS // s["world"],
        "comm_mode": "ar",
        "swiglu_limit": s["limit"],
    }
    args.update(kw)
    return MegaMoeTPConfig(**args)


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
    """Per-process state: TP groups, one layer (and its reference weights) per shape, benchmark state."""

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

    def build(self, shape, variant=0, fp8_inter=None):
        """(layer, standard MXFP4 weights, FP8 weights or None) of this rank (collective over the shape's group)."""
        from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPGfx942

        s = SHAPES[shape]
        raw = make_weights(
            s["E"],
            s["H"],
            s["I"],
            seed=weight_seed(shape, self.rank, variant),
            fp8_inter=fp8_inter,
            device=self.device,
        )
        fp8 = {k: raw.pop(k) for k in [k for k in raw if "_fp8" in k]} or None
        group, cpu = self.tp[s["world"]]
        layer = MegaMoeTPGfx942(
            make_config(shape, self.rank),
            **to_megamoe_layout(raw),
            group=group,
            device=self.device,
            cpu_group=cpu,
        )
        return layer, raw, fp8

    def get(self, shape):
        assert self.member(shape), f"rank {self.rank} is not in {shape}"
        if shape not in self.layers:
            self.layers[shape] = self.build(shape)[:2]
        return self.layers[shape]

    def inputs(self, shape, M, g, share=True):
        """Input group g of case (shape, M). ``share``: every rank of the group takes rank 0's copy (collective):
        ranks may break topk ties of the random routing differently, and ar replicates the input.
        """
        s = SHAPES[shape]
        x, tw, ids = make_inputs(
            M,
            s["H"],
            s["E"],
            s["topk"],
            seed=input_seed(shape, M, g),
            shared=s["shared"],
            device=self.device,
        )
        if share:
            for t in (x, tw, ids):
                dist.broadcast(t, src=0, group=self.tp[s["world"]][0])
        return x, tw, ids

    def release(self):
        """Collective over each group this rank uses: free the benchmark state and every layer."""
        if self.bench is not None:
            self.bench["ar"].close()
            self.bench = None
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


def _verify(state, shape, raw, inputs, y):
    """Check this rank's output y against the FP32 reference summed over the group; bitwise identity of y and of
    the inputs over the group (collective)."""
    s = SHAPES[shape]
    group = state.tp[s["world"]][0]
    x, tw, ids = inputs
    ref = reference(
        x,
        raw["w1"],
        raw["w1_scale"],
        raw["w2"],
        raw["w2_scale"],
        tw,
        ids,
        swiglu_limit=s["limit"] or 0.0,
    )
    dist.all_reduce(ref, group=group)
    res = check(y, ref)
    res["identical"] = _identical(y, group)
    res["inputs_identical"] = all(_identical(t, group) for t in inputs)
    return res


def _job_check(state, shape, tokens):
    """forward + verify of input group 0 for each M in tokens."""
    if not state.member(shape):
        return None
    layer, raw = state.get(shape)
    rows = []
    for M in tokens:
        inputs = state.inputs(shape, M, 0)
        y = layer.forward(*inputs)
        path = ">".join(layer.last_path)
        rows.append(dict(M=M, path=path, **_verify(state, shape, raw, inputs, y)))
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
# Region 1: basic tests (-m basic): one case per dispatch tier x shape. Each pins the tier -- the Gate/Up kernel of
# the unfused kernel sequence listed -- and the communication segment: FUSED_COMM (the cross-rank sum fused into the
# Down decode kernel up to 16 rows, into tp_reduce from 2 MiB (TP2) / 1 MiB (TP4) of BF16 output) or CA (the custom
# all-reduce in between). Not pinned: the kernels after Gate/Up.
# =====================================================================================================================
BASIC = [
    # decode: one workgroup per route / token
    ("tp2", 1, DEC, FUSED_COMM),
    ("ds_tp4", 2, DEC, FUSED_COMM),
    ("tp4", 8, DPQ, FUSED_COMM),  # Down on pre-quantized rows
    ("tp2", 32, PQ, CA),  # both GEMMs on pre-quantized rows
    ("ds_tp4", 16, DD, FUSED_COMM),  # H5120: Gate/Up deduplicated by expert
    # streaming (weights streamed once per expert block)
    ("tp2", 256, ST16, CA),
    ("ds_tp4", 512, ST16_RS, FUSED_COMM),  # Down writes routes, tp_reduce sums them
    ("tp4", 48, ST32Q, CA),  # Gate/Up quantizes the intermediate itself
    ("ds_tp4", 1024, ST32_RS, FUSED_COMM),
    ("tp4", 2048, ST64Q, FUSED_COMM),  # atomic Down, tp_reduce sums the ranks' outputs
    # prefill: sorted 64-row blocks
    ("tp2", 2048, PF_PREP, FUSED_COMM),  # sorting by the streaming prep
    ("tp4", 4096, PF_SORT, FUSED_COMM),  # Opus moe_sorting
    ("ds_tp4", 4096, PF_SORT, FUSED_COMM),
]


@pytest.mark.basic
@pytest.mark.parametrize(
    "shape,M,path,comm", BASIC, ids=[f"{s}_m{m}" for s, m, _, _ in BASIC]
)
def test_basic(pool, shape, M, path, comm):
    tier = gateup_tier(path)
    rows = _member_rows(pool.run("check", shape=shape, tokens=[M]), shape)
    _assert_verified(rows)
    for r in rows:
        assert (
            gateup_tier(r["path"]) == tier and comm_kind(r["path"]) == comm
        ), f"rank {r['rank']}: expected Gate/Up {tier} and communication {comm}, path {r['path']}"


# =====================================================================================================================
# Region 2: other coverage (-m "not basic"). Not here: full-precision checks of the 51 SPEC cases -- the benchmark
# verifies every timed output against the reference (and the cross-rank identity) and fails otherwise.
# =====================================================================================================================
# --- 2.1 both sides of every dispatch boundary --------------------------------------------------------------------
# Why: the benchmark and the basic tests only take powers of two (+48), on one side of each M threshold of the
# dispatch (decode pre-quantization, decode / streaming / prefill, block 16 / 32 / 64, prep / Opus sorting,
# DeepSeek's XCD chunking, the fused communication below); these are the M on the other side, and odd M (partial
# blocks / tiles). The benchmark does not check paths and the basic tests do not pin the fused kernel, so the
# communication kernel of every M is pinned here: decode_tp single-slot at 1 row / slot pairs at 2-16 rows, the
# custom all-reduce from 17 rows, tp_reduce from the byte threshold (1023 / 1024: its 1- / 2-token tiles at H2560).
BOUNDARY_TOKENS = {
    "tp2": (1, 2, 3, 11, 12, 16, 17, 33, 62, 63, 409, 410, 559, 560, 1023, 1024)
    + (1026, 1027, 2234, 2235),
    "tp4": (1, 2, 3, 16, 17, 19, 20, 33, 62, 63, 204, 205, 559, 560, 1023, 1024)
    + (1305, 1306, 2611, 2612),
    "ds_tp4": (1, 2, 3, 7, 16, 17, 33, 102, 103, 768, 769, 1025, 2049, 2730, 2731),
}
# First M whose cross-rank sum is fused into tp_reduce: FUSED_MIN_BYTES of BF16 output (TP2 2 MiB, TP4 1 MiB).
FUSED_FROM_M = {"tp2": 410, "tp4": 205, "ds_tp4": 103}


def expected_comm_kernel(shape, M):
    """The last kernel of an M-row forward of shape: its fused communication kernel or the custom all-reduce."""
    if M == 1:
        return "a8w4_down_decode_tp"
    if M <= 16:
        return "a8w4_down_decode_pq_tp"
    return "a8w4_tp_reduce" if M >= FUSED_FROM_M[shape] else CA


@pytest.mark.parametrize("shape", list(BOUNDARY_TOKENS))
def test_dispatch_boundaries(pool, shape):
    rows = _member_rows(
        pool.run("check", shape=shape, tokens=list(BOUNDARY_TOKENS[shape])), shape
    )
    _assert_verified(rows)
    for r in rows:
        last = expected_comm_kernel(shape, r["M"])
        assert (
            _known_path(r["path"]) and r["path"].split(">")[-1] == last
        ), f"M={r['M']} rank {r['rank']}: expected a known path ending in {last}, path {r['path']}"


# --- 2.2 out=, M = 0 ----------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests never pass ``out`` and never send M = 0. ``out`` (NaN-filled) is written
# by the fused decode Down (8 rows), copied back from the custom all-reduce (48) and written by tp_reduce after the
# atomic streaming Down accumulated into it (2048).
OUT_ARG_TOKENS = (8, 48, 2048)


def _job_out_arg(state):
    shape = "tp4"
    layer, raw = state.get(shape)
    s = SHAPES[shape]
    rows = []
    for M in OUT_ARG_TOKENS:
        inputs = state.inputs(shape, M, 1)
        out = torch.full(
            (M, s["H"]), float("nan"), dtype=torch.bfloat16, device=state.device
        )
        y = layer.forward(*inputs, out=out)
        rows.append(
            dict(
                M=M,
                path=">".join(layer.last_path),
                returned_out=y is out,
                **_verify(state, shape, raw, inputs, out),
            )
        )
    empty = layer.forward(
        torch.empty((0, s["H"]), dtype=torch.bfloat16, device=state.device),
        torch.empty((0, s["topk"]), dtype=torch.float32, device=state.device),
        torch.empty((0, s["topk"]), dtype=torch.int32, device=state.device),
    )
    return {
        "rows": rows,
        "empty_shape": tuple(empty.shape),
        "empty_path": tuple(layer.last_path),
    }


def test_out_argument_and_empty_batch(pool):
    results = pool.run("out_arg")
    for rank, res in enumerate(results):
        _assert_verified(res["rows"])
        assert [r["M"] for r in res["rows"]] == list(OUT_ARG_TOKENS), res["rows"]
        for r in res["rows"]:
            assert r[
                "returned_out"
            ], f"rank {rank} M={r['M']}: forward(out=...) did not return out"
            last = expected_comm_kernel("tp4", r["M"])
            assert (
                r["path"].split(">")[-1] == last
            ), f"rank {rank} M={r['M']}: expected {last} last, path {r['path']}"
        assert (
            res["empty_shape"] == (0, SHAPES["tp4"]["H"]) and res["empty_path"] == ()
        ), res


# --- 2.3 repeated calls across tiers, reset, error flags --------------------------------------------------------
# Why: the benchmark repeats one case at a time and the basic tests call each case once; here one instance jumps
# between decode / streaming / prefill (shared workspaces, cached handles), calls reset() in between, and the
# watchdog API (check_errors / poll_errors / error_flag) is read afterwards.
def _job_repeat(state):
    shape = "tp2"
    if not state.member(shape):
        return None
    layer, raw = state.get(shape)
    rows = []
    for M in (1, 2048, 48, 4096, 16, 256, 1):
        for g in (0, 1):
            inputs = state.inputs(shape, M, g)
            y = layer.forward(*inputs)
            rows.append(
                dict(
                    M=M,
                    g=g,
                    path=">".join(layer.last_path),
                    **_verify(state, shape, raw, inputs, y),
                )
            )
        layer.reset()
    layer.check_errors()
    return {
        "rows": rows,
        "poll": layer.poll_errors(),
        "flag": int(layer.error_flag().abs().max().item()),
    }


def test_repeated_calls_and_reset(pool):
    results = pool.run("repeat")
    for rank, res in enumerate(results[: SHAPES["tp2"]["world"]]):
        _assert_verified(res["rows"])
        assert all(_known_path(r["path"]) for r in res["rows"]), res["rows"]
        assert res["poll"] == 0 and res["flag"] == 0, f"rank {rank}: error flags {res}"
    assert results[2] is None and results[3] is None


# --- 2.4 TP2 / TP4 groups interleaved ---------------------------------------------------------------------------
# Why: the benchmark and the basic tests run one shape after another; here ranks 0 and 1 interleave forwards of a
# TP2 layer and of two TP4 layers (which share one communicator) while ranks 2 and 3 run only the TP4 ones.
ALTERNATE = (
    ("tp2", 8),
    ("tp4", 64),
    ("ds_tp4", 16),
    ("tp2", 4096),
    ("tp4", 1),
    ("ds_tp4", 48),
    ("tp2", 1),
)


def _job_alternate(state):
    rows = []
    for shape, M in ALTERNATE:
        if not state.member(shape):
            continue
        layer, raw = state.get(shape)
        inputs = state.inputs(shape, M, 3)
        y = layer.forward(*inputs)
        rows.append(
            dict(
                shape=shape,
                M=M,
                path=">".join(layer.last_path),
                **_verify(state, shape, raw, inputs, y),
            )
        )
    return rows


def test_tp2_tp4_interleaved(pool):
    results = pool.run("alternate")
    for rank, rows in enumerate(results):
        expect = [(s, m) for s, m in ALTERNATE if rank < SHAPES[s]["world"]]
        assert [(r["shape"], r["M"]) for r in rows] == expect
        _assert_verified(rows)
        assert all(_known_path(r["path"]) for r in rows), rows


# --- 2.5 set_weights --------------------------------------------------------------------------------------------
# Why: the benchmark and the basic tests construct each layer once with one set of weights; set_weights converts
# another layer's weights in place and must switch every later forward to them (and back).
def _job_set_weights(state):
    shape = "tp4"
    layer, raw = state.get(shape)
    s = SHAPES[shape]
    raw2 = make_weights(
        s["E"],
        s["H"],
        s["I"],
        seed=weight_seed(shape, state.rank, variant=1),
        device=state.device,
    )
    rows = []
    try:
        layer.set_weights(**to_megamoe_layout(raw2))
        for M in (16, 256):
            inputs = state.inputs(shape, M, 2)
            y = layer.forward(*inputs)
            rows.append(
                dict(
                    weights=2,
                    M=M,
                    path=">".join(layer.last_path),
                    **_verify(state, shape, raw2, inputs, y),
                )
            )
    finally:
        layer.set_weights(**to_megamoe_layout(raw))
    inputs = state.inputs(shape, 16, 2)
    y = layer.forward(*inputs)
    rows.append(
        dict(
            weights=1,
            M=16,
            path=">".join(layer.last_path),
            **_verify(state, shape, raw, inputs, y),
        )
    )
    return rows


def test_set_weights(pool):
    for rows in pool.run("set_weights"):
        _assert_verified(rows)
        assert [r["weights"] for r in rows] == [2, 2, 1]


# --- 2.6 CUDA graph capture and replay --------------------------------------------------------------------------
# Why: the benchmark and the basic tests run eagerly; serving captures the decode batches in CUDA graphs. After
# prepare(), forward is captured once (static input / output buffers) and replayed on two other input groups.
GRAPH_CASES = [
    ("tp2", 8),
    ("ds_tp4", 1),
    ("ds_tp4", 16),
    ("ds_tp4", 48),
    ("ds_tp4", 8192),
]


def _job_graph(state, shape, M):
    if not state.member(shape):
        return None
    layer, raw = state.get(shape)
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
        rows.append(dict(M=M, g=g, path=path, **_verify(state, shape, raw, inputs, ys)))
    graph.reset()
    return rows


@pytest.mark.parametrize(
    "shape,M", GRAPH_CASES, ids=[f"{s}_m{m}" for s, m in GRAPH_CASES]
)
def test_cuda_graph(pool, shape, M):
    rows = _member_rows(pool.run("graph", shape=shape, M=M), shape)
    _assert_verified(rows)
    assert all(_known_path(r["path"]) for r in rows), rows


# --- 2.7 default communicators ----------------------------------------------------------------------------------
# Why: the benchmark and the other tests pass explicit TP / gloo groups; with group=None and cpu_group=None the
# layer uses the default group and creates its gloo group itself (collective) -- the fused sum's workspace (8 rows)
# and the custom all-reduce (48 rows) run over it -- and close() releases both.
DEFAULT_COMM_TOKENS = (8, 48)


def _job_default_comm(state):
    from aiter.ops.flydsl.mega_moe_tp_gfx942 import MegaMoeTPGfx942

    shape = "tp4"
    _, raw = state.get(shape)
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
                    **_verify(state, shape, raw, inputs, y),
                )
            )
        return rows
    finally:
        layer.close()


def test_default_communicators(pool):
    for rows in pool.run("default_comm"):
        _assert_verified(rows)
        assert [r["M"] for r in rows] == list(DEFAULT_COMM_TOKENS), rows
        # the fused sum over the layer's own groups, and the custom all-reduce (not the RCCL fallback)
        for r in rows:
            last = expected_comm_kernel("tp4", r["M"])
            assert _known_path(r["path"]) and r["path"].split(">")[-1] == last, rows


# --- 2.8 rejected configurations and arguments ------------------------------------------------------------------
# Why: the benchmark and the basic tests pass valid arguments only. Rank 0 alone: every check raises before any
# collective.
def _job_errors(state):
    import aiter.ops.flydsl.mega_moe_tp_gfx942 as mod
    from aiter import ActivationType

    shape = "tp4"
    layer, raw = state.get(shape)
    if state.rank != 0:
        return None
    s, dev = SHAPES[shape], state.device
    w = to_megamoe_layout(raw)
    group, cpu = state.tp[s["world"]]
    x, tw, ids = state.inputs(shape, 8, 0, share=False)
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

    def on_gfx950():
        real = mod.get_gfx_runtime
        mod.get_gfx_runtime = lambda: "gfx950"
        try:
            return build()
        finally:
            mod.get_gfx_runtime = real

    cases = [
        ("not gfx942", on_gfx950, ValueError, "gfx942"),
        (
            "comm_mode ag_rs",
            lambda: build({"comm_mode": "ag_rs"}),
            NotImplementedError,
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
            "0xFF",
        ),
        (
            "model_dim not a multiple of 512",
            lambda: build({"model_dim": s["H"] + 256}),
            ValueError,
            "model_dim",
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


# --- 2.9 the path checks themselves (CPU only) ------------------------------------------------------------------
# Why: the benchmark does not check paths, and the tests above only see the paths this code takes (a known Gate/Up
# kernel, one all-reduce at the end); a check that accepted anything would pass them all. Its verdicts on fused
# reductions and on broken paths (no / several / misplaced all-reduces, no / an unknown Gate/Up) are pinned here.
def test_path_checks():
    fused = "a8w4_gateup_decode>a8w4_down_decode_tp"
    cases = [  # path, comm_kind, _known_path
        (f"{DPQ}>{CA}", CA, True),
        (f"{PF_SORT}>{RCCL}", RCCL, True),
        (fused, FUSED_COMM, True),
        (f"{ST16}>a8w4_tp_reduce", FUSED_COMM, True),
        (f"{PF_PREP}_tp", FUSED_COMM, True),  # sorted_sum_tp
        (DEC, None, False),  # no communication
        (f"{fused}>{CA}", None, False),  # fused and an all-reduce
        (f"{DEC}>{RCCL}>{RCCL}", None, False),
        (f"a8w4_gateup_decode>{CA}>a8w4_down_decode", None, False),  # not last
        ("a8w4_gateup_decode>a8w4_down_decode_stp", None, False),  # no word "tp"
        (f"a8w4_down_decode>{CA}", CA, False),  # no Gate/Up kernel
        (f"a8w4_gateup_decode_v2>a8w4_down_decode>{CA}", CA, False),  # unknown one
    ]
    wrong = [
        (p, comm_kind(p), _known_path(p))
        for p, kind, known in cases
        if (comm_kind(p), _known_path(p)) != (kind, known)
    ]
    assert not wrong, wrong
    # the first Gate/Up kernel, also behind sorting / quantization; every unfused sequence has one
    assert gateup_tier(PF_SORT) == "a8w4_gateup_prefill_bm64_bn128_tk128"
    assert gateup_tier(fused) == gateup_tier(DEC) == "a8w4_gateup_decode"
    assert None not in GATEUP_KERNELS


# --- 2.10 waits on peers: a late rank, a rank that skips a call, reset(), prepare()'s retry ----------------------
# Why: the benchmark and the basic tests start every rank's fused kernel together and never time out. Here the fused
# decode Down (8 rows) and tp_reduce (256 rows) wait for a rank that starts 0.5 s late (right sums, no error bits);
# with the wait bound cut to 50 ms, for a rank that skips the call: the others give up within the bound with their
# kernel's error bit, check_errors() raises once and clears it, the skipping rank's call after that is right or has
# its error bit too, and reset() restores right sums. prepare() resets every rank and warms up again when one rank's
# watchdog reports an error, and raises on every rank when it reports one again.
PEER_WAIT_CASES = (("tp4", 8), ("tp4", 256))
# error_flag() bit of a wait that gave up, by fused kernel (tp_reduce.ERR_TIMEOUT, decode_tp.ERR_TIMEOUT)
TIMEOUT_BIT = {
    "a8w4_tp_reduce": 1,
    "a8w4_down_decode_tp": 2,
    "a8w4_down_decode_pq_tp": 2,
}


def _job_peer_wait(state, shape, M):
    from aiter.ops.flydsl import mega_moe_tp_gfx942 as mod

    if not state.member(shape):
        return None
    layer, raw = state.get(shape)
    s = SHAPES[shape]
    W, dev = s["world"], state.device
    group, cpu = state.tp[W]
    laggard = state.rank == W - 1

    def verified(step, g):
        inputs = state.inputs(shape, M, g)
        y = layer.forward(*inputs)
        path = ">".join(layer.last_path)
        return dict(step=step, path=path, **_verify(state, shape, raw, inputs, y))

    def raised(fn):
        try:
            fn()
        except RuntimeError as e:
            return str(e)
        return None

    rows = [verified("warm", 0)]
    res = {}
    # a rank 0.5 s late, within the wait bound
    inputs = state.inputs(shape, M, 1)
    torch.cuda.synchronize(dev)
    dist.barrier(group=cpu)
    if laggard:
        time.sleep(0.5)
    y = layer.forward(*inputs)
    path = ">".join(layer.last_path)
    rows.append(dict(step="late", path=path, **_verify(state, shape, raw, inputs, y)))
    res["late_bits"] = layer.poll_errors()
    # a rank that skips the call (wait bound 50 ms), then calls after the others gave up
    x, tw, ids = inputs = state.inputs(shape, M, 2)
    ref = reference(
        x,
        raw["w1"],
        raw["w1_scale"],
        raw["w2"],
        raw["w2_scale"],
        tw,
        ids,
        swiglu_limit=s["limit"] or 0.0,
    )
    dist.all_reduce(ref, group=group)
    torch.cuda.synchronize(dev)
    dist.barrier(group=cpu)
    saved = mod.WAIT_S, mod.FIRST_WAIT_S
    mod.WAIT_S = mod.FIRST_WAIT_S = 0.05
    try:
        began = time.perf_counter()
        if not laggard:
            layer.forward(*inputs)
        torch.cuda.synchronize(dev)
        res["took"] = time.perf_counter() - began
        res["bits"] = layer.poll_errors()
        res["raised"] = raised(layer.check_errors)
        res["cleared"] = layer.poll_errors()
        dist.barrier(group=cpu)
        if laggard:
            y = layer.forward(*inputs)
            res["skip_ok"] = check(y, ref)["ok"]
            res["skip_bits"] = layer.poll_errors()
            res["skip_raised"] = raised(layer.check_errors)
    finally:
        mod.WAIT_S, mod.FIRST_WAIT_S = saved
    layer.reset()
    rows += [verified("reset", g) for g in (0, 3)]
    res["reset_bits"] = layer.poll_errors()
    # prepare(): one rank's watchdog reports an error (the warmup itself runs clean: a real sticky bit would make that
    # rank's waits give up early) once -> every rank resets and warms up again; twice -> RuntimeError on every rank
    local = -(-M // W)

    def failing(times):
        left = [times]

        def poll():
            left[0] -= 1
            return 1 if left[0] >= 0 else type(layer).poll_errors(layer)

        return poll

    for name, times in (("retry", 1), ("stuck", 2)):
        if laggard:
            layer.poll_errors = failing(times)
        try:
            res[name] = raised(lambda: layer.prepare([local]))
        finally:
            if laggard:
                del layer.poll_errors
        res[f"{name}_bits"] = layer.poll_errors()
    layer.reset()
    rows.append(verified("after prepare", 1))
    res["final_bits"] = layer.poll_errors()
    return dict(res, rows=rows)


@pytest.mark.parametrize(
    "shape,M", PEER_WAIT_CASES, ids=[f"{s}_m{m}" for s, m in PEER_WAIT_CASES]
)
def test_peer_waits_and_timeouts(pool, shape, M):
    last = expected_comm_kernel(shape, M)
    W = SHAPES[shape]["world"]
    for rank, res in enumerate(pool.run("peer_wait", shape=shape, M=M)):
        if rank >= W:
            assert res is None
            continue
        _assert_verified(res["rows"])
        assert all(r["path"].split(">")[-1] == last for r in res["rows"]), res["rows"]
        assert res["late_bits"] == 0, f"rank {rank}: error bits after a late peer {res}"
        if rank == W - 1:  # skipped; its late call is right or flagged
            assert res["bits"] == 0 and res["raised"] is None, res
            assert res["skip_ok"] or res["skip_bits"], res
            assert (res["skip_raised"] is not None) == bool(res["skip_bits"]), res
        else:
            assert res["bits"] == TIMEOUT_BIT[last] and res["took"] < 5.0, res
            assert res["raised"] and "gave up" in res["raised"], res
        assert res["cleared"] == res["reset_bits"] == 0, res
        assert res["retry"] is None and res["retry_bits"] == res["stuck_bits"] == 0, res
        assert res["stuck"] and "timed out" in res["stuck"], res
        assert res["final_bits"] == 0, res


# --- 2.11 CUDA graph replays between eager calls, two instances on one workspace --------------------------------
# Why: the benchmark and the basic tests run one instance eagerly, and 2.6 replays one graph by itself; serving
# replays decode graphs between eager calls of other sizes, and the instances of one communicator, hidden size and
# batch bound (layers of a model) share the fused sums' buffers and their epochs / parities (advanced by the kernels).
# Two TP4 instances with different weights run in graphs of 8 / 256 / 1024 rows (decode_tp, tp_reduce with 1- / 2-
# token tiles) replayed between eager calls of both (fused sizes and the custom all-reduce).
GRAPH_EAGER_TOKENS = (8, 256, 1024)
GRAPH_EAGER_STEPS = (
    ("replay", 256),
    ("eager", 4096),
    ("replay", 8),
    ("eager", 256),
    ("replay", 1024),
    ("eager", 8),
    ("replay", 256),
    ("eager", 1023),
    ("eager", 1),
    ("replay", 8),
    ("eager", 48),
    ("replay", 1024),
    ("eager", 2),
    ("replay", 8),
)


def _job_graph_shared(state):
    shape = "tp4"
    dev = state.device
    layer, raw = state.get(shape)
    other, raw2, _ = state.build(shape, variant=1)
    pairs = ((layer, raw), (other, raw2))
    graphs = {}
    try:
        for M in GRAPH_EAGER_TOKENS:
            statics = [t.clone() for t in state.inputs(shape, M, 0)]
            outs = [torch.empty_like(statics[0]) for _ in pairs]
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for (lay, _), y in zip(pairs, outs):
                    lay.forward(*statics, out=y)
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for (lay, _), y in zip(pairs, outs):
                    lay.forward(*statics, out=y)
            paths = [">".join(lay.last_path) for lay, _ in pairs]
            graphs[M] = (graph, statics, outs, paths)
        rows = []
        for i, (kind, M) in enumerate(GRAPH_EAGER_STEPS):
            inputs = state.inputs(shape, M, i % 4)
            if kind == "replay":
                graph, statics, ys, paths = graphs[M]
                for static, t in zip(statics, inputs):
                    static.copy_(t)
                graph.replay()
            else:
                ys, paths = [], []
                for lay, _ in pairs:
                    ys.append(lay.forward(*inputs))
                    paths.append(">".join(lay.last_path))
            for j, ((lay, w), y, path) in enumerate(zip(pairs, ys, paths)):
                res = _verify(state, shape, w, inputs, y)
                res.update(step=(i, kind), M=M, layer=j, path=path)
                rows.append(dict(res, bits=lay.poll_errors()))
        return {"shared": other._ws is layer._ws, "rows": rows}
    finally:
        for graph, *_ in graphs.values():
            graph.reset()
        other.close()


def test_graph_eager_shared_workspace(pool):
    for rank, res in enumerate(pool.run("graph_shared")):
        assert res["shared"], f"rank {rank}: the instances do not share the workspace"
        rows = res["rows"]
        _assert_verified(rows)
        assert len(rows) == 2 * len(GRAPH_EAGER_STEPS), rows
        for r in rows:
            last = expected_comm_kernel("tp4", r["M"])
            assert r["path"].split(">")[-1] == last and r["bits"] == 0, (rank, r)


# --- 2.12 the single-slot fused decode Down above 1 row -----------------------------------------------------------
# Why: every weight of the benchmark and the tests fits the slot-pair Down (moe_a8w4._down_pq_fits), so their fused
# decode sums 2-16 rows with the slot-pair kernel only; a w2 of 4 GiB or more (32-bit offsets) does not fit, too
# large to allocate here: with the check patched, the single-slot decode_tp kernel sums the unquantized rows.
SINGLE_SLOT_CASES = (("tp4", 2), ("tp4", 16), ("ds_tp4", 2), ("ds_tp4", 16))


def _job_single_slot(state):
    from aiter.ops.flydsl import moe_a8w4

    real = moe_a8w4._down_pq_fits
    moe_a8w4._down_pq_fits = lambda *args: False
    try:
        rows = []
        for shape, M in SINGLE_SLOT_CASES:
            layer, raw = state.get(shape)
            inputs = state.inputs(shape, M, 1)
            y = layer.forward(*inputs)
            res = _verify(state, shape, raw, inputs, y)
            rows.append(dict(res, shape=shape, M=M, path=">".join(layer.last_path)))
        return rows
    finally:
        moe_a8w4._down_pq_fits = real


def test_single_slot_decode_fallback(pool):
    for rows in pool.run("single_slot"):
        _assert_verified(rows)
        assert [(r["shape"], r["M"]) for r in rows] == list(SINGLE_SLOT_CASES), rows
        for r in rows:
            kernels = r["path"].split(">")
            # straight from Gate/Up: no row quantization in between
            assert (
                _known_path(r["path"])
                and kernels[-1] == "a8w4_down_decode_tp"
                and kernels[-2].startswith(GATEUP_PREFIX)
            ), r


# =====================================================================================================================
# Region 3: benchmark (python <this file> [--cases ...]) -- after the last test
# =====================================================================================================================
OURS, REF2, REF1 = "mega_moe_tp_gfx942", "a8w4_2stage_ar", "aiter_fp8_ptpc_ar"
FP8_INTER = {
    "tp2": 320,
    "tp4": 192,
}  # Qwen shapes: Aiter FP8 ptpc reference, I zero-padded as in its tuned table
BENCH_TABLE = "qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv"
WARMUP, ITERS, GROUPS = 3, 30, 4
FLUSH_BYTES, IDLE_US = 512 << 20, 200
HBM_BYTES_PER_S, FP8_FLOPS = 5.3e12, 606e12  # MI308X


class _RefAR:
    """The all-reduce of the references, as SGLang on AMD: aiter CustomAllreduce when should_custom_ar accepts the
    tensor, RCCL otherwise. Its 128-byte custom all-reduce also aligns the ranks before every timed call.
    """

    def __init__(self, group, cpu_group, device):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce,
            IPCBufferPool,
        )

        self.group, self.cpu_group, self.device = group, cpu_group, device
        # The pool's store keys use a per-process counter; align it, as the TP2 ranks already made pools.
        seq = torch.tensor([IPCBufferPool._pool_seq], dtype=torch.int64)
        dist.all_reduce(seq, op=dist.ReduceOp.MAX, group=cpu_group)
        IPCBufferPool._pool_seq = int(seq.item())
        self.ca = CustomAllreduce(group=cpu_group, device=device)
        assert not self.ca.disabled, "custom all-reduce disabled"
        self.tiny = torch.ones(64, dtype=torch.bfloat16, device=device)
        self.last = None

    def __call__(self, y):
        if self.ca.should_custom_ar(y):
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


def _aiter_row(fm, tokens, hidden, inter, experts, topk):
    """The row of the tuned table the Aiter dispatcher should take for this case."""
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
    if row is None or not 0 < float(row["us"]) < float("inf"):
        raise RuntimeError(f"{BENCH_TABLE} has no valid row for {key}")
    return row


def _row_kernels(row):
    name = str(row["kernelName1"])
    if int(row["run_1stage"]):
        return f"asm_1stage:{name}"
    return name if name.startswith("impl__") else f"{name}+{row['kernelName2']}"


def _uses_row(meta, row):
    """Whether the dispatcher's metadata is this table row (1-stage asm, impl__ and CK 2-stage rows)."""
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


def _a8w4_bytes_per_expert(hidden, inter):
    """Bytes of one expert in the A8W4 layout (``moe_a8w4.prepare_a8w4_weights``): FP4 codes, E8M0 deltas (two
    bytes per 64-K block, padded to groups of four blocks) and an FP32 scale per row; I zero-padded to 64.
    """
    ip = -(-inter // 64) * 64

    def matrix(n, k):
        return n * k // 2 + n * (-(-(k // 64) // 4) * 8) + n * 4

    return matrix(2 * ip, hidden) + matrix(hidden, ip)


def _bytes_per_expert(weights, experts):
    values = weights.values() if isinstance(weights, dict) else vars(weights).values()
    return (
        sum(t.numel() * t.element_size() for t in values if isinstance(t, torch.Tensor))
        // experts
    )


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
    began = time.time()
    s = SHAPES[shape]
    layer, raw, fp8 = state.build(shape, fp8_inter=FP8_INTER.get(shape))
    state.layers[shape] = (layer, raw)
    group, cpu = state.tp[s["world"]]
    bench = state.bench = {
        "shape": shape,
        "ar": _RefAR(group, cpu, state.device),
        "fp8": None,
        "pyhip": None,
        "cycles": _cycles_per_us(),
        "flush": torch.empty(
            FLUSH_BYTES // 4, dtype=torch.float32, device=state.device
        ),
    }
    if fp8 is not None:
        from aiter.ops.shuffle import shuffle_weight

        bench["fp8"] = {
            "w1": shuffle_weight(fp8["w1_fp8"], layout=(16, 16)),
            "w1_scale": fp8["w1_fp8_scale"],
            "w2": shuffle_weight(fp8["w2_fp8"], layout=(16, 16)),
            "w2_scale": fp8["w2_fp8_scale"],
        }
    src = os.environ.get("PYHIP_SRC")
    if src:  # reference 2: the pyhip M17 source this implementation was ported from
        if src not in sys.path:
            sys.path.insert(0, src)
        from pyhip.ops.moe.flydsl import moe_a8w4 as m17

        bench["pyhip"] = (
            m17,
            m17.prepare_a8w4_weights(
                raw["w1"], raw["w1_scale"], raw["w2"], raw["w2_scale"]
            ),
        )
    torch.cuda.synchronize(state.device)
    return {
        "took": time.time() - began,
        "pyhip": src or "",
        "aiter": fp8 is not None,
        "cycles_per_us": bench["cycles"],
    }


def _job_bench(state, case, warmup, iters):
    shape, M = parse_case(case)
    if not state.member(shape):
        return None
    bench = state.bench
    if bench is None or bench["shape"] != shape:
        raise RuntimeError(f"bench_setup({shape!r}) has not run")
    s, dev = SHAPES[shape], state.device
    layer, raw = state.layers[shape]
    group, cpu = state.tp[s["world"]]
    ar, H, E = bench["ar"], s["H"], s["E"]
    groups = [state.inputs(shape, M, g) for g in range(GROUPS)]
    impls, paths, kernels = {OURS: lambda g: layer.forward(*groups[g])}, {}, None
    per_expert = {OURS: _a8w4_bytes_per_expert(H, s["I"])}
    if bench["pyhip"] is not None:
        m17, prepared = bench["pyhip"]
        bufs = [
            torch.empty((M, H), dtype=torch.bfloat16, device=dev) for _ in range(GROUPS)
        ]
        limit_kw = {"swiglu_limit": s["limit"]} if s["limit"] else {}

        def ref2(g):
            x, tw, ids = groups[g]
            m17.fused_moe_a8w4(x, prepared, tw, ids, output=bufs[g], **limit_kw)
            return ar(bufs[g])

        impls[REF2] = ref2
        per_expert[REF2] = _bytes_per_expert(prepared, E)
    if bench["fp8"] is not None:
        import aiter.fused_moe as fm
        from aiter import ActivationType, QuantType

        fw = bench["fp8"]

        def ref1(g):
            x, tw, ids = groups[g]
            return ar(
                fm.fused_moe(
                    x,
                    fw["w1"],
                    fw["w2"],
                    tw,
                    ids,
                    activation=ActivationType.Silu,
                    quant_type=QuantType.per_Token,
                    w1_scale=fw["w1_scale"],
                    w2_scale=fw["w2_scale"],
                )
            )

        row = _aiter_row(fm, M, H, FP8_INTER[shape], E, s["topk"])
        seen, resolve = [], fm.get_2stage_cfgs

        def spy(*args, **kwargs):
            meta = resolve(*args, **kwargs)
            seen.append(meta)
            return meta

        fm.get_2stage_cfgs = spy
        try:
            ref1(0)
        finally:
            fm.get_2stage_cfgs = resolve
        kernels = _row_kernels(row)
        if not seen or not _uses_row(seen[-1], row):
            raise RuntimeError(
                f"{case}: Aiter did not run the tuned row {kernels} (block_m={row['block_m']}); "
                f"got {seen[-1] if seen else None}"
            )
        impls[REF1] = ref1
        per_expert[REF1] = _bytes_per_expert(fw, E)

    def path_of(impl):
        if impl == OURS:
            return ">".join(layer.last_path)
        return (
            ">".join(list(m17.last_path) if impl == REF2 else [f"aiter:{kernels}"])
            + f">{ar.last}"
        )

    last = {k: [None] * GROUPS for k in impls}
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
    # The A8W4 implementations alternate with each other; the Aiter FP8 reference runs on its own afterwards: an
    # A8W4 call right after an Aiter FP8 call measured up to 5% faster (decode / streaming), even after the flush
    # and the sleep, which would favour whichever implementation follows it.
    for batch in ([n for n in impls if n != REF1], [n for n in impls if n == REF1]):
        for i in range(iters):
            g = i % GROUPS
            for impl in batch:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                bench["flush"].zero_()
                torch.cuda._sleep(sleep_cycles)
                ar.barrier()
                start.record()
                last[impl][g] = impls[impl](g)
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
                ids.unique().numel() * per_expert[impl] + M * H * 2 * 2
                for _, _, ids in groups
            ),
        }
    res[OURS]["error_flag"] = max(
        int(layer.error_flag().abs().max().item()), int(layer.poll_errors())
    )
    # every timed output: FP32 reference (A8W4 implementations; Aiter: finite only), cross-rank identity
    for g, inputs in enumerate(groups):
        x, tw, ids = inputs
        ref = reference(
            x,
            raw["w1"],
            raw["w1_scale"],
            raw["w2"],
            raw["w2_scale"],
            tw,
            ids,
            swiglu_limit=s["limit"] or 0.0,
        )
        dist.all_reduce(ref, group=group)
        same_inputs = all(_identical(t, group) for t in inputs)
        for impl in impls:
            st, y = res[impl], last[impl][g]
            if impl == REF1:
                st["ok"] = st["ok"] and bool(torch.isfinite(y).all())
            else:
                r = check(y, ref)
                st["ok"] = st["ok"] and bool(r["ok"])
                st["calc_diff"] = max(st["calc_diff"], r["calc_diff"])
                st["err_ratio"] = max(st["err_ratio"], r["err_ratio"])
            st["identical"] = st["identical"] and _identical(y, group)
            st["inputs_identical"] = st["inputs_identical"] and same_inputs
        del ref
    flops = 6 * M * s["topk"] * H * s["I"]
    return {
        "case": case,
        "flops": flops,
        "idle_us": idle_us,
        "kernels": kernels,
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
    for impl in (
        REF2,
        REF1,
    ):  # mf keeps the first MF_REF impl of a case: reference 2 first
        if impl not in impls:
            continue
        st = impls[impl]
        extra = f" kernels={r['kernels']}" if impl == REF1 else ""
        if not good(st):
            extra += " ok=0"
            failures.append(f"{case}:{impl}")
            print(f"FAIL case={case} impl={impl} {st}", flush=True)
        print(
            f"MF_REF case={case} impl={impl} value={st['us']:.2f} unit=us flops={flops} bytes={st['bytes']:.0f} "
            f"ratio={ours['us'] / st['us']:.3f}{extra}",
            flush=True,
        )
    w = s["world"]
    xgmi = (
        2 * (w - 1) / w * M * s["H"] * 2
    )  # bytes a rank sends (and receives) in a ring-equivalent all-reduce
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


def _bench_main(argv=None):
    parser = argparse.ArgumentParser(
        description="MegaMoeTPGfx942 benchmark: SPEC cases, cold-cache protocol, "
        "slowest rank; references a8w4_2stage_ar and aiter_fp8_ptpc_ar"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=CASES, metavar="CASE"
    )
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iters < 1:
        parser.error("--warmup and --iters must be >= 1")
    began = time.time()
    # Aiter reference: only the Qwen3.8 FP8 ptpc tuned table (a copy: Aiter's duplicate check may rewrite it); the
    # workers inherit the environment.
    table_src = (
        Path(__file__).resolve().parents[2]
        / "aiter/configs/model_configs"
        / BENCH_TABLE
    )
    tmp = Path(tempfile.mkdtemp(prefix="bench_mega_moe_tp_gfx942_"))
    shutil.copyfile(table_src, tmp / BENCH_TABLE)
    os.environ["AITER_CONFIG_FMOE"] = str(tmp / BENCH_TABLE)
    os.environ["AITER_ONLINE_TUNE"] = "0"
    print(
        f"INFO seed={SEED} warmup={args.warmup} iters={args.iters} groups={GROUPS} flush_bytes={FLUSH_BYTES} "
        f"aiter_table={table_src} pyhip_src={os.environ.get('PYHIP_SRC', '')} cases={len(args.cases)}",
        flush=True,
    )
    failures = []
    p = _Pool()
    try:
        gfx = p.start()
        if gfx != "gfx942":
            raise SystemExit(f"MegaMoeTPGfx942 runs on gfx942 only (found {gfx})")
        for shape in SHAPES:
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
        p.close()
    finally:
        p.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"INFO done in {time.time() - began:.0f}s failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    _bench_main()
