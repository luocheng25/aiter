# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark of the gfx942 FlyDSL A8W4 GEMM: Y = X W^T, BF16 X quantized per
token to FP8 E4M3FNUZ, MXFP4 W (E2M1 + one E8M0 per 32 K), BF16 Y.

One file, three regions:
  1. basic functionality (``@pytest.mark.basic``): every target shape and dispatch tier, the
     first thing to run after every change: ``pytest op_tests/test_gemm_a8w4_ptpc.py -m basic``;
  2. other coverage: what regions 1 and 3 cannot cover (tile boundaries, bit-exact
     quantization with its edge rows and weight layout, non-contiguous weight slices, input
     errors, preallocated output, CUDA graph, compile cache, M = 0, M past 32-bit byte
     offsets, the large-M dispatch boundary and persistent-kernel edge cases, the small-M
     dispatch boundaries, partial token tiles and bit-exact in-kernel quantization):
     ``pytest op_tests/test_gemm_a8w4_ptpc.py -m "not basic"``;
  3. benchmark (``main``, not collected by pytest): 6 Qwen3.8-Flash-Next shapes x
     M = 1, 2, 4, ..., 32768 against aiter's FP8 per-token GEMM, every timed output
     checked: ``python op_tests/test_gemm_a8w4_ptpc.py``.
Every output is graded against the independent fp32 torch reference ``run_torch``;
``MF_SEED`` seeds all inputs.
"""

import argparse
import contextlib
import functools
import itertools
import os
import zlib

import pandas as pd
import pytest
import torch
from torch.autograd import DeviceType

import aiter
import aiter.ops.flydsl.gemm_a8w4_ptpc as a8w4_module
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import (
    GemmA8W4PtpcWeight,
    flydsl_gemm_a8w4_ptpc,
    gemm_a8w4_ptpc_kernel_names,
    gemm_a8w4_ptpc_shape_supported,
    prepare_gemm_a8w4_ptpc_weight,
)
from aiter.ops.gemm_op_a8w8 import get_GEMM_config_with_quant_type
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

# =============================================================================================
# Shared: shapes, inputs, fp32 reference, grading
# =============================================================================================

SUPPORTED_GFX = ["gfx942"]

# Qwen3.8-Flash-Next dense linear layers after SGLang's TP split -> (N, K) of Y = X W^T.
# Hidden 2560; GDN: 16 K heads x 128, 48 V heads x 128; full attention: 24 heads x 256 with a
# sigmoid output gate, 2 KV heads (replicated when TP exceeds them).
#   tp2_qkvz: GDN in_proj_qkvz, column split: (qkv 10240 + z 6144) / 2 = 8192, K = 2560.
#   tp2_qkv:  attention q + gate 12 heads x 512, k and v 1 head x 256 each = 6656, K = 2560.
#   tp2_o:    GDN out_proj and attention o_proj (same shape), row split: N = 2560, K = 6144 / 2.
#   tp4_*:    the same layers at TP4 (q + gate 6 heads x 512, k and v replicated).
SHAPES = {
    "tp2_qkvz": (8192, 2560),
    "tp2_qkv": (6656, 2560),
    "tp2_o": (2560, 3072),
    "tp4_qkvz": (4096, 2560),
    "tp4_qkv": (3584, 2560),
    "tp4_o": (2560, 1536),
}
FP8_MAX = 240.0
# prepare_gemm_a8w4_ptpc_weight stores 32-K groups this many E8M0 codes (or more) below the
# row's largest code as zero; the reference applies the same rule.
FLUSH_DELTA = 12
CALC_DIFF_MAX = 1e-5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1 = E2M1 + tuple(-v for v in E2M1)
SEED = int(os.environ.get("MF_SEED", "0"))
DEVICE = "cuda"


def _on_supported_gfx():
    try:
        return torch.cuda.is_available() and get_gfx_runtime() in SUPPORTED_GFX
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _on_supported_gfx(), reason="the A8W4 GEMM targets gfx942"
)


def case_seed(*parts):
    """Deterministic per-case seed, shifted by ``MF_SEED``."""
    return SEED * 1_000_003 + zlib.crc32(repr(parts).encode())


def _generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def pack_fp4(nibbles):
    """E2M1 codes [n, k] (uint8 0..15) -> checkpoint fp4x2 [n, k/2], low nibble = even k."""
    return (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def make_weight(n, k, seed):
    """Checkpoint MXFP4 weight [n, k]: fp4x2 [n, k/2] and E8M0 [n, k/32].

    All 16 E2M1 codes (-0 included). Each row has its own largest E8M0 code (120..123) and
    each 32-K group lies 0..3 codes below it, so a scale read for the wrong row or group
    shows in the output.
    """
    gen = _generator(seed)
    nibbles = torch.randint(0, 16, (n, k), generator=gen, device=DEVICE).to(torch.uint8)
    top = torch.randint(120, 124, (n, 1), generator=gen, device=DEVICE)
    delta = torch.randint(0, 4, (n, k // 32), generator=gen, device=DEVICE)
    codes = (top - delta).to(torch.uint8).view(torch.float8_e8m0fnu)
    return pack_fp4(nibbles), codes


def make_x(m, k, seed):
    """BF16 activation [m, k]; row magnitudes span 2**-4 .. 2**4 so every row has its own scale."""
    gen = _generator(seed)
    row_scale = torch.exp2(torch.rand((m, 1), generator=gen, device=DEVICE) * 8 - 4)
    x = torch.randn((m, k), generator=gen, device=DEVICE) * row_scale
    return x.to(torch.bfloat16)


def quant_per_token(x):
    """Per-token FP8 E4M3FNUZ: scale = amax / 240 (1 for an all-zero row), q = RNE(x / scale).

    Both are IEEE f32 divisions: torch turns ``tensor / python scalar`` into a multiplication
    by the reciprocal, so the divisor is a tensor.
    """
    t = x.float()
    amax = t.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, FP8_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (t / scale).to(torch.float8_e4m3fnuz), scale


def dequant_mxfp4(w, w_scale, flush=True):
    """Exact float32 value of MXFP4 [n, k]: E2M1 x 2**(E8M0 - 127); ``flush`` applies the
    prepare rule (groups FLUSH_DELTA or more codes below the row maximum are zero)."""
    b = w.view(torch.uint8)
    nibbles = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).long()
    value = torch.tensor(E2M1, dtype=torch.float32, device=b.device)[nibbles]
    codes = w_scale.view(torch.uint8).to(torch.int32)
    factor = torch.exp2(codes.double() - 127).float()
    if flush:
        below = codes.amax(dim=-1, keepdim=True) - codes >= FLUSH_DELTA
        factor = torch.where(below, torch.zeros_like(factor), factor)
    return (value.unflatten(-1, (-1, 32)) * factor.unsqueeze(-1)).flatten(-2)


@contextlib.contextmanager
def _ieee_fp32_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def fp32_matmul(a, b_t):
    """a [m, k] @ b_t[n, k]^T in IEEE fp32 (TF32 / XF32 off)."""
    with _ieee_fp32_matmul():
        return a.float() @ b_t.float().T


def run_torch(x, w, w_scale, flush=True):
    """Independent fp32 reference of flydsl_gemm_a8w4_ptpc: per-token FP8 X x exact MXFP4 W."""
    xq, x_scale = quant_per_token(x)
    return fp32_matmul(xq.float() * x_scale, dequant_mxfp4(w, w_scale, flush))


def calc_diff(x, y):
    """1 - 2 sum(xy) / sum(x^2 + y^2) in float64; 0 when both are all zero."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def check_output(out, ref, msg=""):
    """Per element |out - ref| <= 1% |ref| + 1% max|ref| of the row, calc_diff <= 1e-5, finite.

    Both sides differ only by f32 accumulation order and the BF16 rounding of ``out``.
    Returns the fraction of elements outside the tolerance (asserted to be 0).
    """
    assert tuple(out.shape) == tuple(ref.shape), f"{msg}: {out.shape} vs {ref.shape}"
    o, r = out.float(), ref.float()
    assert bool(torch.isfinite(o).all()), f"{msg}: output has NaN/Inf"
    row_max = r.abs().amax(dim=-1, keepdim=True)
    row_max = row_max.clamp_min(torch.finfo(torch.float32).tiny)
    err = checkAllclose(o / row_max, r / row_max, rtol=0.01, atol=0.01, msg=msg)
    assert err == 0, f"{msg}: {err:.3%} of elements outside 1% |ref| + 1% row max"
    diff = calc_diff(o, r)
    assert diff <= CALC_DIFF_MAX, f"{msg}: calc_diff {diff:.3g} > {CALC_DIFF_MAX}"
    return err


def launched_kernels(fn, attempts=3):
    """Run ``fn`` under the profiler: (its result, names of the GPU kernels it launched).

    The profiler now and then records no GPU event at all for a call that did launch kernels
    (seen on shared MI308X GPUs, about one call in a hundred). A call with no recorded kernel
    is therefore profiled again, up to ``attempts`` times in all: a call that launches nothing
    (M = 0) still returns [], and whatever is recorded still has to match exactly.
    """
    for _ in range(attempts):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            result = fn()
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
        if names:
            break
    return result, [name.removesuffix(".kd") for name in names]


# Kernel name prefix -> dispatch tier. The tests name the tier of a kernel rather than its full
# name, so that a kernel name that gains configuration fields (a tile picked per shape and M)
# does not break them; what ran is still checked against gemm_a8w4_ptpc_kernel_names.
_KERNEL_TIERS = (
    ("gemm_a8w4_ptpc_quant_", "quant"),
    ("gemm_a8w4_ptpc_decode_", "decode"),
    ("gemm_a8w4_ptpc_tile_", "tile"),
    ("gemm_a8w4_ptpc_bm", "general"),
    ("gemm_a8w4_ptpc_pp_", "large"),
)


def kernel_tiers(names):
    """Tier of each kernel name: ``quant``, the one-launch small-M kernels ``decode`` and
    ``tile``, ``general`` (the general GEMM) or ``large`` (the persistent large-M kernel); an
    unknown name is returned as it is, so it matches no tier."""
    return tuple(
        next((tier for prefix, tier in _KERNEL_TIERS if name.startswith(prefix)), name)
        for name in names
    )


@functools.lru_cache(maxsize=4)
def cached_weight(n, k, seed):
    """(w, w_scale, prepared weight) for tests that share one weight."""
    w, w_scale = make_weight(n, k, seed)
    return w, w_scale, prepare_gemm_a8w4_ptpc_weight(w, w_scale)


# =============================================================================================
# Region 1: basic functionality (@pytest.mark.basic)
# prepare_gemm_a8w4_ptpc_weight + flydsl_gemm_a8w4_ptpc on each target shape with one
# representative M, one case per dispatch tier: M = 1 (decode kernel), M = 20 on tp4_o (tile
# kernel, two token tiles), M = 33..2048 with non-powers of two (per-token quant, then the
# general GEMM) and M = 4096 (quant, then the persistent large-M kernel); asserts that the
# launched FlyDSL kernels are the ones the dispatch predicts.
# =============================================================================================

BASIC_CASES = [
    ("tp2_qkvz", 1),
    # tp4_o, not tp4_qkv: the per-shape mid-M dispatch moves M = 17..32 of the target shapes
    # to the quant and the general GEMM (tp4_o from M = 21), so tp4_o M = 20 runs the tile
    # kernel before and after that change, where tp4_qkv M = 20 would silently switch tiers.
    # Two K windows are covered in region 2.9 (N = 4160).
    ("tp4_o", 20),
    ("tp2_qkv", 33),
    ("tp2_o", 130),
    ("tp4_qkvz", 256),
    ("tp4_qkv", 1000),
    ("tp4_o", 4096),
]


@pytest.mark.basic
@pytest.mark.parametrize(
    "shape, m", BASIC_CASES, ids=[f"{s}_m{m}" for s, m in BASIC_CASES]
)
def test_gemm(shape, m):
    n, k = SHAPES[shape]
    assert gemm_a8w4_ptpc_shape_supported(n, k)
    w, w_scale = make_weight(n, k, case_seed("w", shape))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    assert isinstance(weight, GemmA8W4PtpcWeight) and (weight.n, weight.k) == (n, k)
    x = make_x(m, k, case_seed("x", shape, m))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
    out, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert kernels == list(gemm_a8w4_ptpc_kernel_names(m, n, k)), kernels
    assert out.dtype == torch.bfloat16 and out.device == x.device
    check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{m}")


# =============================================================================================
# Region 2: other coverage (not basic)
# Only what the basic cases and the benchmark cannot show; each test says why.
# =============================================================================================

# -- 2.1 tile boundaries -----------------------------------------------------------------------

BOUNDARY_CASES = [(63, 192, 256), (65, 64, 768), (129, 320, 512)]


@pytest.mark.parametrize(
    "m, n, k", BOUNDARY_CASES, ids=[f"m{m}_n{n}_k{k}" for m, n, k in BOUNDARY_CASES]
)
def test_tile_boundaries(m, n, k):
    """M one below / above the 64-row tile, N with a half-empty 128-channel tile
    (N % 128 == 64, masked stores and zero-filled weight loads), and K = 256 / 512 / 768
    (zero, one and two pipelined loop iterations).

    The target shapes have N % 128 == 0 and K >= 1536; the benchmark uses powers of two
    for M and the basic cases one M per shape.
    """
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    x = make_x(m, k, case_seed("x", m, n, k))
    out = flydsl_gemm_a8w4_ptpc(x, prepare_gemm_a8w4_ptpc_weight(w, w_scale))
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")


# -- 2.2 per-token quantization, bit-exact -----------------------------------------------------


@pytest.mark.parametrize("k", [2560, 256])
def test_quant_bitexact(k):
    """The per-token quantization equals the reference bit for bit: scale = amax / 240 and
    q = RNE(x / scale) with the IEEE quotient (the kernel computes it as a reciprocal
    multiply plus two FMA corrections, and divides per element when scale < 2**-64).
    Includes an all-zero row (scale 1), a row of exact FP8 rounding midpoints where a plain
    reciprocal multiply rounds the other way, the same row times 2**-70, and a row of -0 and
    negatives that round to zero (no byte may be 0x80, NaN in FNUZ); K = 256 runs the
    32-lanes-per-row variant.

    End-to-end grading cannot see a few flipped FP8 codes: they move the GEMM output by far
    less than its tolerance.
    """
    gen = _generator(case_seed("quant", k))
    x = torch.randn((997, k), generator=gen, device=DEVICE) * 3
    x[5].zero_()
    # Row 6: scale = 52.5 / 240 = 7/32 (not a power of two); every quotient is an FP8
    # rounding midpoint (mantissa 23/16 .. 31/16). Row 7: row 6 x 2**-70 (scale < 2**-64).
    exponents = torch.exp2(torch.arange(-6, 7, device=DEVICE).float())
    tie = torch.arange(23, 32, 2, device=DEVICE).float().view(-1, 1) / 16 * exponents
    tie = torch.cat([tie.flatten(), -tie.flatten()]) * (7 / 32)
    x[6] = tie.repeat(k // tie.numel() + 1)[:k]
    x[6, 0] = 52.5
    x[7] = x[6] * 2.0**-70
    # Row 8: -0 and negatives that round to zero (FNUZ has no -0; 0x80 reads back as NaN).
    x[8, 1::2] = -0.0
    x[8, 2::4] = -1e-6
    x = x.to(torch.bfloat16)

    q, scale = a8w4_module._quant_per_token(x)
    ref_q, ref_scale = quant_per_token(x)
    assert torch.equal(scale.view(-1, 1), ref_scale)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert int((q.view(torch.uint8) == 0x80).sum()) == 0
    assert float(scale[5]) == 1.0
    assert float(scale[7]) < 2.0**-64 <= float(scale[6])
    naive = x.float() * (torch.ones_like(ref_scale) / ref_scale)
    naive = naive.to(torch.float8_e4m3fnuz).view(torch.uint8)
    exact = ref_q.view(torch.uint8)
    assert bool((exact[6] != naive[6]).any()) and bool((exact[7] != naive[7]).any())


@pytest.mark.parametrize("k", [2560, 256])
def test_quant_max_quotient_and_nonfinite(k):
    """Rows the bit-pattern amax and the clamp-free FP8 conversion must get right: a row
    whose max |x| / scale rounds to the float above 240 in the reciprocal path (negative
    maximum, stored as -240) and one in the per-element path (subnormal scale, quotient
    240.0004) match the reference bit for bit; rows with +NaN, -NaN, or Inf and NaN get
    scale NaN and only NaN codes (0x80), like the reference; +-Inf rows get scale Inf.
    Every other row, including those sharing a wave with these (K = 256: 32 lanes per row),
    matches the reference bit for bit.

    The codes of a +-Inf row are not compared: the reciprocal path turns 0 * Inf into NaN
    for every element where torch gives 0 for the finite ones; the GEMM row is NaN either
    way. test_quant_bitexact's random rows reach a quotient above 240 only by chance, and no
    other test has a non-finite or subnormal-scale input.
    """
    gen = _generator(case_seed("quant_edge", k))
    x = (torch.randn((16, k), generator=gen, device=DEVICE) * 0.5).clamp(-1, 1)
    x[1, 3] = -1.09375
    x[2] *= 1e-39
    x[2, 5] = 2.0**-126 * 1.0625  # BF16 0x0088; scale = it / 240 is subnormal
    x[3, 7] = float("nan")
    x[4, 9] = -float("nan")
    x[5, 1], x[5, 2] = float("inf"), float("nan")
    x[6, 11] = float("inf")
    x[7, 13] = -float("inf")
    x = x.to(torch.bfloat16)

    q, scale = a8w4_module._quant_per_token(x)
    ref_q, ref_scale = quant_per_token(x)
    q, ref_q = q.view(torch.uint8), ref_q.view(torch.uint8)
    scale, ref_scale = scale.view(-1), ref_scale.view(-1)
    quotient = x.float().abs().amax(dim=-1) / ref_scale
    assert float(quotient[1]) > FP8_MAX and float(quotient[2]) > FP8_MAX
    assert 0 < float(ref_scale[2]) < torch.finfo(torch.float32).tiny
    assert int(ref_q[1, 3]) == 0xFF and int(ref_q[2, 5]) == 0x7F  # -240, 240

    nan_rows, inf_rows = [3, 4, 5], [6, 7]
    assert bool(scale[nan_rows].isnan().all())
    assert bool(ref_scale[nan_rows].isnan().all())
    assert torch.equal(q[nan_rows], ref_q[nan_rows])
    assert bool((q[nan_rows] == 0x80).all())
    assert torch.equal(scale[inf_rows], ref_scale[inf_rows])
    assert bool((scale[inf_rows] == float("inf")).all())
    rest = [r for r in range(x.shape[0]) if r not in nan_rows + inf_rows]
    assert torch.equal(scale[rest], ref_scale[rest])
    assert torch.equal(q[rest], ref_q[rest])


# -- 2.3 weight layout, the delta flush rule and weight slices ---------------------------------


def _decode_fd2(weight):
    """Decode a prepared weight following kernels/gemm_a8w4_ptpc/layout.py -> (float64 [n, k],
    delta per 32-K group [n, k/32])."""
    n, k = weight.n, weight.k
    kb = k // 64
    kq = (kb + 3) // 4
    b = weight.q.view(n // 16, kb, 4, 16, 2, 4).permute(0, 3, 1, 2, 4, 5)
    b = b.reshape(n, kb, 4, 2, 4)
    nibbles = torch.cat([b & 0xF, b >> 4], dim=-1).reshape(n, k).long()
    table = torch.tensor(E2M1, dtype=torch.float64, device=weight.q.device)
    d = weight.d.view(n // 16, kq, 2, 16, 4).permute(0, 3, 1, 4, 2)
    d = d.reshape(n, kq * 4, 2)[:, :kb].reshape(n, kb * 2).long()
    assert bool((d % 8 == 0).all())
    factor = torch.exp2(5 - (d // 8).double()).repeat_interleave(32, dim=-1)
    return table[nibbles] * factor * weight.s.view(n, 1).double(), d // 8


def test_prepare_layout_bitexact():
    """Decoding the prepared bytes as layout.py documents them gives the exact MXFP4 values:
    all 16 E2M1 codes (-0 stored as +0), group deltas 0..15 (11 exact, 12 and up zero), the
    per-row factor 2**(cmax - 132); also accepts uint8 inputs.

    The basic cases and the benchmark only see GEMM results: a layout that differs from the
    documented one but that the kernel happens to read the same way still passes them, and
    later kernels depend on the documented layout.
    """
    n, k = 192, 768
    gen = _generator(case_seed("layout"))
    nibbles = torch.randint(0, 16, (n, k), generator=gen, device=DEVICE).to(torch.uint8)
    codes = 120 - torch.randint(0, 16, (n, k // 32), generator=gen, device=DEVICE)
    codes[:, 3] = 120
    codes = codes.to(torch.uint8)
    w = pack_fp4(nibbles)
    weight = prepare_gemm_a8w4_ptpc_weight(w.view(torch.uint8), codes)
    got, delta = _decode_fd2(weight)
    expected = dequant_mxfp4(w, codes).double()
    assert torch.equal(got, expected)
    diff = codes.long().amax(-1, keepdim=True) - codes.long()
    assert torch.equal(delta, diff.clamp(max=FLUSH_DELTA - 1))
    assert torch.equal(weight.s, torch.full((n,), 2.0 ** (120 - 132), device=DEVICE))
    assert bool((diff >= FLUSH_DELTA).any()) and bool((diff == FLUSH_DELTA - 1).any())


def test_delta_flush_boundary():
    """A 32-K group 11 E8M0 codes below its row's largest is exact, 12 below is zero: every
    channel has one group at the row maximum and one probe group at delta 11 (even
    channels) or 12 (odd channels); X is non-zero only on the probe group's K. The output
    matches the flushed reference, is exactly zero on the delta-12 channels and matches the
    exact (unflushed) reference on the delta-11 ones.

    The random weights of the other tests keep every group within 3 codes of its row
    maximum, so they never reach the flush boundary.
    """
    m, n, k = 64, 256, 1024
    top_group, probe = 3, 17
    gen = _generator(case_seed("flush"))
    nibbles = torch.zeros((n, k), dtype=torch.uint8, device=DEVICE)
    for g in (top_group, probe):
        nibbles[:, 32 * g : 32 * (g + 1)] = torch.randint(
            0, 16, (n, 32), generator=gen, device=DEVICE
        ).to(torch.uint8)
    delta = torch.where(torch.arange(n, device=DEVICE) % 2 == 0, 11, 12)
    codes = torch.full((n, k // 32), 100, dtype=torch.int64, device=DEVICE)
    codes[:, top_group] = 125
    codes[:, probe] = 125 - delta
    w, w_scale = pack_fp4(nibbles), codes.to(torch.uint8).view(torch.float8_e8m0fnu)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=DEVICE)
    x[:, 32 * probe : 32 * (probe + 1)] = make_x(m, 32, case_seed("flush_x"))
    out = flydsl_gemm_a8w4_ptpc(x, prepare_gemm_a8w4_ptpc_weight(w, w_scale))
    check_output(out, run_torch(x, w, w_scale), msg="flushed reference")
    exact = run_torch(x, w, w_scale, flush=False)
    assert bool((out[:, 1::2] == 0).all())
    assert float(exact[:, 1::2].abs().amax()) > 0
    check_output(out[:, 0::2], exact[:, 0::2], msg="delta 11 vs exact reference")


def test_prepare_noncontiguous_slices():
    """Non-contiguous weight views, such as the K slice a row-parallel (TP) layer takes of a
    wider checkpoint weight, in the torch FP4 / E8M0 dtypes and as uint8, prepare to the same
    bytes as their contiguous copies.

    The basic cases and the benchmark always pass whole, contiguous weights.
    """
    n, k = 128, 512
    w, w_scale = make_weight(n, 2 * k, case_seed("slices"))
    w_part, s_part = w[:, : k // 2], w_scale[:, : k // 32]
    assert not w_part.is_contiguous() and not s_part.is_contiguous()
    expected = prepare_gemm_a8w4_ptpc_weight(w_part.contiguous(), s_part.contiguous())
    for got in (
        prepare_gemm_a8w4_ptpc_weight(w_part, s_part),
        prepare_gemm_a8w4_ptpc_weight(
            w_part.view(torch.uint8), s_part.view(torch.uint8)
        ),
    ):
        assert (got.n, got.k) == (n, k)
        for field in ("q", "d", "s"):
            assert torch.equal(getattr(got, field), getattr(expected, field)), field


# -- 2.4 input errors and the architecture gate ------------------------------------------------


def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=DEVICE)


def test_rejects_bad_inputs(monkeypatch):
    """Unsupported inputs raise instead of computing garbage: weight / scale / activation
    dtypes, N % 64, K % 256, mismatched scale or activation shapes, E8M0 code 255 (NaN),
    non-contiguous activation or output, a foreign weight object, and any GPU other than
    gfx942 (simulated).

    The basic cases and the benchmark only pass valid inputs on gfx942.
    """
    n, k = 128, 512
    w, w_scale = make_weight(n, k, case_seed("errors"))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(4, k, case_seed("errors_x"))

    for args, error in [
        ((w.view(torch.int8), w_scale), TypeError),
        ((w, w_scale.view(torch.uint8).float()), TypeError),
        ((w[:96], w_scale[:96]), ValueError),  # N % 64
        ((w[:, :64], w_scale[:, :4]), ValueError),  # K = 128
        ((w, w_scale[:, :-1]), ValueError),
        ((w.view(-1), w_scale), ValueError),
        ((w.cpu(), w_scale.cpu()), ValueError),
    ]:
        with pytest.raises(error):
            prepare_gemm_a8w4_ptpc_weight(*args)
    nan_scale = w_scale.view(torch.uint8).clone()
    nan_scale[7, 3] = 255
    with pytest.raises(ValueError, match="255"):
        prepare_gemm_a8w4_ptpc_weight(w, nan_scale)

    for args, kwargs, error in [
        ((x.half(), weight), {}, TypeError),
        ((x, (w, w_scale)), {}, TypeError),
        ((x.view(-1), weight), {}, ValueError),
        ((x[:, :256].contiguous(), weight), {}, ValueError),
        ((torch.cat([x, x], dim=1)[:, :k], weight), {}, ValueError),  # strided rows
        ((x, weight), {"out": _empty((4, n + 64), torch.bfloat16)}, ValueError),
        ((x, weight), {"out": _empty((4, n), torch.float16)}, ValueError),
        ((x, weight), {"out": _empty((n, 4), torch.bfloat16).t()}, ValueError),
        ((x, weight), {"out": _empty((4, n), torch.bfloat16).cpu()}, ValueError),
    ]:
        with pytest.raises(error):
            flydsl_gemm_a8w4_ptpc(*args, **kwargs)

    assert all(gemm_a8w4_ptpc_shape_supported(n_, k_) for n_, k_ in SHAPES.values())
    # The last one has 2**31 weight bytes, past the kernel's 32-bit offsets.
    for n_, k_ in [
        (96, 2560),
        (2560, 128),
        (0, 256),
        (256, 0),
        (2560, 2624),
        (1 << 20, 4096),
    ]:
        assert not gemm_a8w4_ptpc_shape_supported(n_, k_)
        with pytest.raises(ValueError):
            gemm_a8w4_ptpc_kernel_names(1, n_, k_)

    monkeypatch.setattr(a8w4_module, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942"):
        prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    with pytest.raises(RuntimeError, match="gfx942"):
        flydsl_gemm_a8w4_ptpc(x, weight)


# -- 2.5 preallocated output, M = 0, CUDA graph ------------------------------------------------


def test_out_preallocated():
    """``out=`` (filled with NaN) is written completely and returned, and nothing past it
    is touched (it is the head of a larger NaN buffer); a non-power-of-two M leaves a
    partial last tile.

    The basic cases and the benchmark let the op allocate its output and never look past it.
    """
    n, k = SHAPES["tp4_o"]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", "tp4_o"))
    m = 77
    x = make_x(m, k, case_seed("out", m))
    buffer = torch.full((m + 64, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    out = buffer[:m]
    result = flydsl_gemm_a8w4_ptpc(x, weight, out=out)
    assert result is out
    check_output(out, run_torch(x, w, w_scale), msg="out=")
    assert bool(buffer[m:].isnan().all())


def test_m_zero():
    """M = 0 returns an empty BF16 [0, N] tensor (or the given ``out``) and launches nothing.

    The basic cases and the benchmark have M >= 1.
    """
    n, k = SHAPES["tp4_o"]
    _, _, weight = cached_weight(n, k, case_seed("w", "tp4_o"))
    x = torch.empty((0, k), dtype=torch.bfloat16, device=DEVICE)
    y, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert (
        tuple(y.shape) == (0, n) and y.dtype == torch.bfloat16 and y.device == x.device
    )
    assert kernels == [] and gemm_a8w4_ptpc_kernel_names(0, n, k) == ()
    out = torch.empty((0, n), dtype=torch.bfloat16, device=DEVICE)
    assert flydsl_gemm_a8w4_ptpc(x, weight, out=out) is out


@pytest.mark.parametrize(
    "m", [5, 20, 48, 2049], ids=["m5_decode", "m20_tile", "m48", "m2049_large"]
)
def test_cuda_graph_replay(m):
    """The whole call (its activation buffers and output allocated inside the capture) is
    captured in a CUDA graph and replays on new input: no host synchronization. M = 5 runs
    the decode kernel and M = 20 the tile kernel (both quantize X themselves), M = 48 the
    general GEMM, M = 2049 the persistent large-M kernel.

    The basic cases and the benchmark call the op eagerly.
    """
    # tp4_o, not tp2_o: the per-shape mid-M dispatch moves tp2_o M = 17..32 to the quant and
    # the general GEMM but keeps tp4_o M = 20 on the tile kernel. Each M asserts the tier it is
    # here for (by kernel name tier, not by an internal kernel name constant).
    n, k = SHAPES["tp4_o"]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", "tp4_o"))
    tier = {5: "decode", 20: "tile", 48: "general", 2049: "large"}[m]
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers[-1] == tier, tiers
    x = make_x(m, k, case_seed("graph", 0))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = flydsl_gemm_a8w4_ptpc(x, weight)
    x_new = make_x(m, k, case_seed("graph", 1))
    x.copy_(x_new)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check_output(y, run_torch(x_new, w, w_scale), msg="graph replay")


# -- 2.6 compile cache -------------------------------------------------------------------------


def test_compile_cache():
    """Calls alternate between shapes and return to earlier ones (same N with another K,
    same K with another N, a new M on a compiled shape): every call uses its own shape's
    compiled kernels. The decode and tile kernels of tp2_o and tp4_o (N = 2560, K = 3072 /
    1536) at M = 2 and 12 even carry the same kernel name.

    The benchmark visits each shape once, in order, so a cache that only kept the most
    recent compiled kernel would still pass it.
    """
    for shape, m in [
        ("tp2_o", 7),
        ("tp4_o", 129),
        ("tp2_o", 20),
        ("tp2_o", 300),
        ("tp4_qkvz", 1),
        ("tp2_qkvz", 65),
        ("tp4_o", 2),
        ("tp4_o", 12),
        ("tp2_o", 2),
        ("tp2_o", 12),
    ]:
        n, k = SHAPES[shape]
        w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
        x = make_x(m, k, case_seed("cache", shape, m))
        out = flydsl_gemm_a8w4_ptpc(x, weight)
        check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{m}")


# -- 2.7 M past 32-bit byte offsets ------------------------------------------------------------


@pytest.mark.parametrize("n", [320, 256], ids=["n320", "n256_large"])
def test_m_past_32bit_offsets(n):
    """M = 4194401 with K = 512: BF16 X spans more than 2**32 bytes, FP8 X and Y more than
    2**31. N = 320 runs the general GEMM with 65538 row tiles (grid.y > 65535, last tile 33
    rows); N = 256 the persistent large-M kernel over 32769 tiles of 128 rows (last tile 97
    rows, starting exactly at byte 2**31 of Y and of FP8 X). Every kernel rebases its buffers
    at the first row of its CTA or tile. X is zero except the first tile, the tiles around
    each 2**31 / 2**32 byte boundary and the end; Y starts as NaN. The data rows match the
    reference and every other row is exactly zero, so a lost or misplaced store shows.

    The benchmark stops at M = 32768 (Y at most 512 MiB, X 192 MiB), far from 2**31 bytes.
    Needs about 9 GB of GPU memory.
    """
    m, k = 4194401, 512
    # By kernel name tier rather than the large kernel's name constant.
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == ("quant", "large" if n % 256 == 0 else "general"), tiers
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    # First rows at or past 2**31 bytes of BF16 X, of Y and of FP8 X (= 2**32 of BF16 X).
    boundaries = [(1 << 31) // (2 * k), -(-(1 << 31) // (2 * n)), (1 << 31) // k]
    rows = torch.cat(
        [torch.arange(0, 64), torch.arange(m // 64 * 64 - 64, m)]
        + [torch.arange(r // 64 * 64 - 64, r // 64 * 64 + 128) for r in boundaries]
    ).unique()
    rows = rows[rows < m].to(DEVICE)
    x = torch.zeros((m, k), dtype=torch.bfloat16, device=DEVICE)
    x[rows] = make_x(rows.numel(), k, case_seed("m_past_32bit", m))
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    flydsl_gemm_a8w4_ptpc(x, weight, out=out)
    check_output(out[rows], run_torch(x[rows], w, w_scale), msg=f"m{m} data rows")
    nonzero = torch.cat([(chunk != 0).any(dim=1) for chunk in out.split(1 << 18)])
    data = torch.zeros(m, dtype=torch.bool, device=DEVICE)
    data[rows] = True
    assert torch.equal(nonzero, data), "rows written to the wrong place or not at all"
    del x, out
    torch.cuda.empty_cache()


# -- 2.8 large-M dispatch and the persistent kernel --------------------------------------------

LARGE_M_DISPATCH = [
    (2048, 256, 512, False),
    (2049, 256, 512, True),
    (2049, 320, 512, False),
    (2049, 256, 256, False),
]


@pytest.mark.parametrize(
    "m, n, k, large",
    LARGE_M_DISPATCH,
    ids=[f"m{m}_n{n}_k{k}" for m, n, k, _ in LARGE_M_DISPATCH],
)
def test_large_m_dispatch(m, n, k, large):
    """These N, K are not target shapes, whose dispatch may switch at an M of their own:
    M > 2048 with N % 256 == 0 and K >= 512 runs the persistent large-M kernel; M = 2048,
    N % 256 != 0 or K = 256 run the general GEMM. The predicted and the launched kernels are
    that one and the output matches the reference. M = 2049, N = 256, K = 512 is the large
    kernel's smallest case: 17 tiles (fewer than CUs, one per CTA), a 1-row last tile, and
    K = 512 leaves no K pair loop iteration between the peeled first and last pairs.

    The basic cases and the benchmark run target shapes (N % 256 == 0, K >= 1536) at
    M = 4096 or powers of two: never at M = 2049, never ruled out by N or K, and their
    large-M tiles are full.
    """
    # By kernel name tier, not by the kernel name constants: the general GEMM's kernel name
    # may carry the tile the entry picks per shape and M (no single general kernel name).
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    assert kernel_tiers(names) == ("quant", "large" if large else "general"), names
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(m, k, case_seed("large_dispatch", m, n, k))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
    out, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert kernels == list(names), kernels
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")


def test_large_m_tiles():
    """The persistent kernel with more tiles than CUs and a partial last M block, writing
    through ``out=`` (the head of a NaN buffer): N = 1024 (4 N blocks) and one 128-row M block
    more than the CUs take, the last with 40 rows (80 CUs: M = 2600, 84 tiles), so the first
    CTAs run a full tile and then a tail tile; K = 768 runs the K pair loop once. Every row
    is written and nothing past it.

    The benchmark's large-M cases have M a multiple of 128 and K >= 1536, and the op
    allocates their output.
    """
    n, k = 1024, 768
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    m = max(num_cus // (n // 256), 16) * 128 + 40
    # By kernel name tier rather than the large kernel's name constant.
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == ("quant", "large"), tiers
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    x = make_x(m, k, case_seed("large_tiles", m))
    buffer = torch.full((m + 128, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    out = buffer[:m]
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    assert flydsl_gemm_a8w4_ptpc(x, weight, out=out) is out
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")
    assert bool(buffer[m:].isnan().all())


# -- 2.9 small M (M <= 32): the one-launch decode and tile kernels -----------------------------

# (m, n, k, kernel kinds the dispatch may run: "decode", "tile" or None for the per-token quant
# and the general GEMM; token slots of the decode kernel or 16-token tiles of the tile kernel)
SMALL_M_DISPATCH = [
    (8, 2560, 1536, ("decode",), 8),
    (9, 2560, 1536, ("tile",), 1),
    (16, 2560, 1536, ("tile",), 1),
    (17, 2560, 1536, ("tile",), 2),
    # Not pinned (was the tile kernel): the per-shape mid-M dispatch moves a target shape's
    # M = 17..32 (tp4_o's M = 21..32) to the quant and the general GEMM, so tp4_o M = 32 may
    # run either, as long as it is the predicted one and the tile kernel fits. Pin it again
    # once that dispatch has landed.
    (32, 2560, 1536, ("tile", None), 2),
    # The tile kernel's 32-token limit, moved off the target shapes (this pair was tp4_o
    # M = 32 / 33): N = 4160 is under no per-shape dispatch; two K windows.
    (32, 4160, 2560, ("tile",), 2),
    (33, 4160, 2560, (None,), 0),
    (5, 128, 768, (None,), 0),
    (20, 5184, 12288, (None,), 0),
]


@pytest.mark.parametrize(
    "m, n, k, kinds, tokens",
    SMALL_M_DISPATCH,
    ids=[f"m{m}_n{n}_k{k}" for m, n, k, _, _ in SMALL_M_DISPATCH],
)
def test_small_m_dispatch(m, n, k, kinds, tokens):
    """M <= 8 runs the decode kernel alone, M = 9..16 the tile kernel with one 16-token tile,
    M = 17..32 with two, up to the tile kernel's 32-token limit; M = 33, K % 512 != 0
    (K = 768) and a tile kernel whose token rows and deltas do not fit in the LDS (N = 5184,
    K = 12288) run the per-token quant and the general GEMM. A target shape may hand
    M = 17..32 to the quant and the general GEMM instead: tp4_o M = 32 accepts either. The
    predicted and the launched kernels are those and the output matches the reference.

    The benchmark runs M = 8, 16, 32 and 64 (one side of each boundary), all its shapes have
    K % 512 == 0 and fit, and it does not check which kernels ran.
    """
    config = a8w4_module.small_m_config(m, n, k)
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    small = [kind for kind in kinds if kind is not None]
    if small:
        assert config is not None and config[0] in small and config[3] == tokens, config
    else:
        assert config is None, config
    # The quant + general GEMM pair by kernel name tier, not by the kernel name constants: the
    # general GEMM's kernel name may carry the tile the entry picks per shape and M.
    if kernel_tiers(names) == ("quant", "general"):
        assert None in kinds, names
    else:
        assert config is not None, names
        assert names == (a8w4_module.small_m_kernel_name_of(config),), names
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(m, k, case_seed("small_dispatch", m, n, k))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
    out, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert kernels == list(names), kernels
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")


SMALL_M_PARTIAL = [
    (3, 8192, 2560),
    (7, 6656, 2560),
    (10, 3584, 2560),
    # N = 4160, not tp4_qkvz's 4096: the per-shape mid-M dispatch moves tp4_qkvz M = 17..32 to
    # the general GEMM; N = 4160 is under no per-shape dispatch and gets the same tile kernel
    # configuration (4 row blocks x 4 K parts, two token tiles, two K windows).
    (23, 4160, 2560),
    (20, 5184, 1024),
    (20, 2560, 8192),
]


@pytest.mark.parametrize(
    "m, n, k", SMALL_M_PARTIAL, ids=[f"m{m}_n{n}_k{k}" for m, n, k in SMALL_M_PARTIAL]
)
def test_small_m_partial_tiles(m, n, k):
    """The small-M kernels with M below their token capacity, written through ``out=`` (the
    head of a NaN buffer): token rows >= M are not quantized and their outputs are not stored
    (the decode kernel's stores past M fall outside the buffer bound), so every row matches
    the reference and nothing past ``out`` changes. M = 3 and 7: decode kernel with 4 and 8
    token slots (M = 7: 80 CTAs over 416 row blocks, 5 or 6 each); M = 10: tile kernel, one
    16-token tile; M = 23: two token tiles over two K windows, the K-part sums reusing the
    token-row LDS; N = 5184: 41 tile CTAs over 324 row blocks (7 or 8 each), so some wave slots
    compute a neighbouring block and store nothing; K = 8192: six K windows.

    The benchmark's M are the token capacities (1, 2, 4, 8, 16, 32), its shapes give every
    tile CTA the same number of row blocks and at most two K windows, and the op allocates
    its output.
    """
    config = a8w4_module.small_m_config(m, n, k)
    assert config is not None, config
    assert m < (config[3] if config[0] == "decode" else 16 * config[3]), config
    # The dispatch runs that kernel: a per-shape dispatch may run the general GEMM where a
    # small-M kernel fits, which would silently skip what this test is for.
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == (config[0],), tiers
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    x = make_x(m, k, case_seed("small_partial", m, n, k))
    buffer = torch.full((m + 32, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    out = buffer[:m]
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    assert flydsl_gemm_a8w4_ptpc(x, weight, out=out) is out
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")
    assert bool(buffer[m:].isnan().all())


def _small_m_quant_rows(k):
    """11 BF16 rows for the in-kernel quantization: 0..4 random, 5 all zero, 6 FP8 rounding
    midpoints (as in test_quant_bitexact), 7 row 6 x 2**-70 and 9 a random row x 1e-30
    (scale < 2**-64), 8 -0 and negatives that round to zero, 10 a maximum whose quotient is
    the float above 240 (as in test_quant_max_quotient_and_nonfinite: stored as -240)."""
    gen = _generator(case_seed("small_quant", k))
    x = torch.randn((11, k), generator=gen, device=DEVICE) * 3
    x[5].zero_()
    exponents = torch.exp2(torch.arange(-6, 7, device=DEVICE).float())
    tie = torch.arange(23, 32, 2, device=DEVICE).float().view(-1, 1) / 16 * exponents
    tie = torch.cat([tie.flatten(), -tie.flatten()]) * (7 / 32)
    x[6] = tie.repeat(k // tie.numel() + 1)[:k]
    x[6, 0] = 52.5
    x[7] = x[6] * 2.0**-70
    x[8, 1::2] = -0.0
    x[8, 2::4] = -1e-6
    x[9] *= 1e-30
    x[10] = (x[10] / 6).clamp(-1, 1)
    x[10, 3] = -1.09375
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("m", [2, 5, 10, 23], ids=["m2", "m5", "m10", "m23"])
def test_small_m_quant_bitexact(m):
    """The small-M kernels quantize X inside each CTA exactly like the reference. With a
    one-hot weight (channel c reads K position c % K with weight 1.0; N >= K) the output is
    Y[t, c] = bf16(xq[t, c % K] * x_scale[t]) bit for bit, which exposes every FP8 code and
    every scale. Rows: random, all zero (scale 1), FP8 rounding midpoints (a plain reciprocal
    multiply rounds some the other way), two rows with scale < 2**-64 (per-element division),
    -0 with negatives that round to zero (a NaN code 0x80 would turn the whole output row
    NaN) and a quotient just above 240 (no clamp: it must round to -240, not NaN). M = 2 and
    5: decode kernel (2 and 8 token slots); M = 10: tile kernel; M = 23: two token tiles,
    both holding special rows, over two separately quantized K windows.

    test_quant_bitexact covers the separate quantization kernel of the other paths; with
    random weights a few flipped FP8 codes move the output far less than its tolerance.
    """
    # N = 4160, not tp4_qkvz's 4096, so that M = 23 runs the tile kernel: the per-shape mid-M
    # dispatch moves tp4_qkvz M = 17..32 to the general GEMM; N = 4160 is under no per-shape
    # dispatch and gets the same small-M kernel configurations.
    n, k = 4160, 2560
    probe = _small_m_quant_rows(k)
    probe_q, probe_scale = quant_per_token(probe)
    assert float(probe_scale[7]) < 2.0**-64 <= float(probe_scale[6])
    assert float(probe_scale[9]) < 2.0**-64
    assert int(probe_q.view(torch.uint8)[10, 3]) == 0xFF
    order = [6, 10, 7, 5, 8, 9, 0, 1, 2, 3, 4]
    rows = ((order + order[::-1]) * 2)[:m]
    x = probe[rows].contiguous()
    columns = torch.arange(n, device=DEVICE) % k
    nibbles = torch.zeros((n, k), dtype=torch.uint8, device=DEVICE)
    nibbles[torch.arange(n, device=DEVICE), columns] = 2  # E2M1 1.0
    codes = torch.full((n, k // 32), 127, dtype=torch.uint8, device=DEVICE)  # 2**0
    weight = prepare_gemm_a8w4_ptpc_weight(pack_fp4(nibbles), codes)
    config = a8w4_module.small_m_config(m, n, k)
    assert config is not None and config[0] == ("decode" if m <= 8 else "tile"), config
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == (config[0],), tiers  # the dispatch runs that kernel
    out = flydsl_gemm_a8w4_ptpc(x, weight)
    xq, x_scale = quant_per_token(x)
    expected = (xq.float()[:, columns] * x_scale).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m", [5, 20], ids=["m5_decode", "m20_tile"])
def test_small_m_nonfinite_rows(m):
    """A token row holding NaN or +-Inf gets scale NaN or Inf inside the small-M kernels
    and turns only its own output row NaN, as in the reference; every other token of the
    CTA, in the same and the other token tile, still matches the reference.

    The basic cases and the benchmark only pass finite activations;
    test_quant_max_quotient_and_nonfinite covers the separate quantization kernel.
    """
    # tp4_o, not tp2_o: the per-shape mid-M dispatch moves tp2_o M = 17..32 to the quant and
    # the general GEMM but keeps tp4_o M = 20 on the tile kernel; the tier is asserted.
    n, k = SHAPES["tp4_o"]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", "tp4_o"))
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == ("decode" if m <= 8 else "tile",), tiers
    x = make_x(m, k, case_seed("small_nonfinite", m))
    bad = [1, 3] if m <= 8 else [1, 3, 17]
    x[1, 7] = float("nan")
    x[3, 100] = float("inf")
    if m > 8:
        x[17, 5] = -float("inf")
    good = [t for t in range(m) if t not in bad]
    out = flydsl_gemm_a8w4_ptpc(x, weight)
    ref = run_torch(x, w, w_scale)
    assert bool(out[bad].isnan().all()) and bool(ref[bad].isnan().all())
    check_output(out[good], ref[good], msg=f"m{m} finite rows")


# =============================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# Each case times the full call from BF16 X to BF16 Y (per-token quantization included) for
# this op and for aiter's FP8 per-token GEMM, as SGLang's gfx942 apply_fp8_ptpc_linear runs it:
# per_token_quant_hip + gemm_a8w8_bpreshuffle on the same MXFP4 weight dequantized (flush
# rule included) and quantized per channel to FP8. Both use run_perftest (inputs deep-copied
# and rotated, >= ~12 ms of warm-up calls, all kernels of a call summed) and grade the timed
# output with checkAllclose against an fp32 reference of their own quantized operands.
# =============================================================================================

BENCH_M = [1 << i for i in range(16)]

# Warm-up before every timed loop, the same for both implementations. The host work between
# timed loops (inputs, references, grading, deep copies; ~0.1 s) leaves the GPU idle, its
# memory clock (UCLK / fabric) drops, and it takes ~5 ms of load to come back up (measured on
# MI308X). With run_perftest's default 2 warm-up calls, the timed loops of the middle M range
# (~6 ms) start inside that ramp: a contiguous slow head (GEMM +43% at M = 256) that the IQR
# filter keeps once it spans more than ~25-32% of the iterations, so 5-20% of those runs read
# 10-30% high, always the op timed first; the short loops of small M can run slow in part or
# throughout. run_perftest runs its warm-up calls back to back right before the timed loop
# (after the deep copies), so each run gets enough of them for >= ~12 ms of load: WARMUP_MS
# of calls as timed on the cold GPU, which run faster once the clock is up.
WARMUP_MS = 18.0


def _warmup_calls(fn, args, calls=8, bursts=3):
    """run_perftest num_warmup: max(2, the fewest calls taking longer than WARMUP_MS).

    The time per call is the fastest of ``bursts`` runs of ``calls`` back-to-back calls timed
    with CUDA events, as the warm-up runs them. Each run starts behind a leading call, which
    keeps the launch latency after the synchronization out of it (otherwise up to a third of
    a small-M call); the first leading call also compiles or loads the kernels.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ms = []
    for _ in range(bursts):
        fn(*args)
        start.record()
        for _ in range(calls):
            fn(*args)
        end.record()
        end.synchronize()
        ms.append(start.elapsed_time(end) / calls)
    return max(2, int(WARMUP_MS / min(ms)) + 1)


@functools.lru_cache(maxsize=1)
def _bench_weight(shape):
    n, k = SHAPES[shape]
    w, w_scale = make_weight(n, k, case_seed("bench_w", shape))
    w_exact = dequant_mxfp4(w, w_scale)
    # aiter FP8 per-token reference: the same weight, per-channel FP8 (exact in BF16 first).
    wq, w_fp8_scale = aiter.per_token_quant_hip(
        w_exact.to(torch.bfloat16), quant_dtype=dtypes.fp8
    )
    w_fp8_exact = wq.float() * w_fp8_scale
    return (
        prepare_gemm_a8w4_ptpc_weight(w, w_scale),
        w_exact,
        shuffle_weight(wq, layout=(16, 16)),
        w_fp8_scale,
        w_fp8_exact,
    )


def aiter_fp8_ptpc(x, wq_shuffled, w_scale):
    xq, x_scale = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    return aiter.gemm_a8w8_bpreshuffle(
        xq, wq_shuffled, x_scale, w_scale, dtype=torch.bfloat16
    )


def _aiter_kernel(m, n, k):
    config = get_GEMM_config_with_quant_type(
        m, n, k, dtypes.fp8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )
    return (
        "default" if config is None else f"{config['libtype']}:{config['kernelName']}"
    )


@benchmark()
def bench_gemm_a8w4_ptpc(shape, m):
    n, k = SHAPES[shape]
    case = f"{shape}_m{m}"
    weight, w_exact, wq_shuffled, w_fp8_scale, w_fp8_exact = _bench_weight(shape)
    x = make_x(m, k, case_seed("bench_x", shape, m))
    xq, x_scale = quant_per_token(x)
    xq_aiter, x_scale_aiter = aiter.per_token_quant_hip(x, quant_dtype=dtypes.fp8)
    flops = 2 * m * n * k
    candidates = {
        "a8w4": (
            flydsl_gemm_a8w4_ptpc,
            (x, weight),
            lambda: fp32_matmul(xq.float() * x_scale, w_exact),
            m * k * 2 + n * k // 2 + n * k // 32 + 4 * n + m * n * 2,
        ),
        "aiter_fp8_ptpc": (
            aiter_fp8_ptpc,
            (x, wq_shuffled, w_fp8_scale),
            lambda: fp32_matmul(xq_aiter.float() * x_scale_aiter, w_fp8_exact),
            m * k * 2 + n * k + 4 * n + m * n * 2,
        ),
    }
    ret = {"gfx": get_gfx_runtime(), "N": n, "K": k}
    for name, (fn, args, reference, nbytes) in candidates.items():
        out, us = run_perftest(fn, *args, num_warmup=_warmup_calls(fn, args))
        err = check_output(out, reference(), msg=f"{case} {name}")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        if name == "a8w4":
            print(
                f"MF_METRIC case={case} value={us:.3f} unit=us flops={flops} bytes={nbytes}",
                flush=True,
            )
        else:
            print(
                f"MF_REF case={case} impl={name} value={us:.3f} unit=us flops={flops} "
                f"bytes={nbytes}",
                flush=True,
            )
    ret["a8w4 kernel"] = gemm_a8w4_ptpc_kernel_names(m, n, k)[-1]
    ret["aiter kernel"] = _aiter_kernel(m, n, k)
    print(f"aiter_fp8_ptpc case={case} kernel={ret['aiter kernel']}", flush=True)
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a8w4_ptpc is unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL A8W4 GEMM against aiter's FP8 per-token GEMM.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        nargs="*",
        choices=list(SHAPES),
        default=list(SHAPES),
        help="weight shapes (name -> N, K):\n"
        + "\n".join(f"  {name}: {nk}" for name, nk in SHAPES.items()),
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=BENCH_M,
        help="token counts M (default 1, 2, 4, ..., 32768)",
    )
    args = parser.parse_args()
    rows = [
        bench_gemm_a8w4_ptpc(shape, m)
        for shape, m in itertools.product(args.shape, args.m)
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "gemm_a8w4_ptpc summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
