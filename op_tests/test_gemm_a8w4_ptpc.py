# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark of the gfx942 FlyDSL A8W4 GEMM: Y = X W^T, BF16 X quantized per
token to FP8 E4M3FNUZ, MXFP4 W (E2M1 + one E8M0 per 32 K), BF16 Y.

One file, three regions:
  1. basic functionality (``@pytest.mark.basic``): one case per target shape, the first
     thing to run after every change: ``pytest op_tests/test_gemm_a8w4_ptpc.py -m basic``;
  2. other coverage: what regions 1 and 3 cannot cover (tile boundaries, bit-exact
     quantization and weight layout, non-contiguous weight slices, input errors,
     preallocated output, CUDA graph, compile cache, M = 0, M past 32-bit byte offsets):
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


def launched_kernels(fn):
    """Run ``fn`` under the profiler: (its result, names of the GPU kernels it launched)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        result = fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    return result, [name.removesuffix(".kd") for name in names]


@functools.lru_cache(maxsize=4)
def cached_weight(n, k, seed):
    """(w, w_scale, prepared weight) for tests that share one weight."""
    w, w_scale = make_weight(n, k, seed)
    return w, w_scale, prepare_gemm_a8w4_ptpc_weight(w, w_scale)


# =============================================================================================
# Region 1: basic functionality (@pytest.mark.basic)
# prepare_gemm_a8w4_ptpc_weight + flydsl_gemm_a8w4_ptpc on each target shape with one
# representative M (M = 1, non-powers of two crossing the 64-row tile, up to 4096); asserts
# the launched FlyDSL kernels (per-token quant, then the one GEMM kernel that serves every M).
# =============================================================================================

BASIC_CASES = [
    ("tp2_qkvz", 1),
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


def test_cuda_graph_replay():
    """The whole call (its activation buffers and output allocated inside the capture) is
    captured in a CUDA graph and replays on new input: no host synchronization.

    The basic cases and the benchmark call the op eagerly.
    """
    n, k = SHAPES["tp2_o"]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", "tp2_o"))
    m = 48
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
    compiled kernels.

    The benchmark visits each shape once, in order, so a cache that only kept the most
    recent compiled kernel would still pass it.
    """
    for shape, m in [
        ("tp2_o", 7),
        ("tp4_o", 129),
        ("tp2_o", 300),
        ("tp4_qkvz", 1),
        ("tp2_qkvz", 65),
        ("tp4_o", 2),
    ]:
        n, k = SHAPES[shape]
        w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
        x = make_x(m, k, case_seed("cache", shape, m))
        out = flydsl_gemm_a8w4_ptpc(x, weight)
        check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{m}")


# -- 2.7 M past 32-bit byte offsets ------------------------------------------------------------


def test_m_past_32bit_offsets():
    """M = 4194401 with N = 320, K = 512: BF16 X spans more than 2**32 bytes, FP8 X and Y more
    than 2**31, and the GEMM grid has 65538 row tiles (grid.y > 65535, last tile 33 rows); both
    kernels rebase their buffers at each CTA's first row. X is zero except the first tile, the
    tiles around each 2**31 / 2**32 byte boundary and the end; Y starts as NaN. The data rows
    match the reference and every other row is exactly zero, so a lost or misplaced store shows.

    The benchmark stops at M = 32768 (Y at most 512 MiB, X 192 MiB), far from 2**31 bytes.
    Needs about 9 GB of GPU memory.
    """
    m, n, k = 4194401, 320, 512
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


# =============================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# Each case times the full call from BF16 X to BF16 Y (per-token quantization included) for
# this op and for aiter's FP8 per-token GEMM, as SGLang's gfx942 apply_fp8_ptpc_linear runs it:
# per_token_quant_hip + gemm_a8w8_bpreshuffle on the same MXFP4 weight dequantized (flush
# rule included) and quantized per channel to FP8. Both use run_perftest (inputs deep-copied
# and rotated, all kernels of a call summed) and grade the timed output with checkAllclose
# against an fp32 reference of their own quantized operands.
# =============================================================================================

BENCH_M = [1 << i for i in range(16)]


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
        out, us = run_perftest(fn, *args)
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
