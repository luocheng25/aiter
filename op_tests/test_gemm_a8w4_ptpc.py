# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests and benchmark of the gfx942 FlyDSL A8W4 GEMM: Y = X W^T, BF16 X quantized to FP8
E4M3FNUZ per token or 1x32 (one E8M0 per token and 32 K, also taken pre-quantized by
flydsl_gemm_a8w4_mxfp8), MXFP4 W (E2M1 + one E8M0 per 32 K), BF16 Y.

One file, three regions:
  1. basic functionality (``@pytest.mark.basic``): every target shape and dispatch tier and
     both 1x32 entries, the first thing to run after every change:
     ``pytest op_tests/test_gemm_a8w4_ptpc.py -m basic``;
  2. other coverage: what regions 1 and 3 cannot cover (tile boundaries, bit-exact
     quantization with its edge rows and weight layout, non-contiguous weight slices, input
     errors, preallocated output, CUDA graph, compile cache, M = 0, M past 32-bit byte
     offsets, the large-M dispatch boundary and persistent-kernel edge cases with its tile
     orders (per-token and 1x32), the small-M dispatch boundaries and LDS fallbacks, partial
     token tiles and bit-exact in-kernel quantization, the M where each target shape's
     dispatch switches kernels or tiles; for 1x32: bit-exact quantization against the
     DeepSeek-V4.1 rule and aiter's and inside the small-M 1x32 kernel, its LDS limits,
     extreme E8M0 scales, every M where the 1x32 dispatch switches paths or tiles, a ragged N
     off the table, input errors, M = 0, CUDA graph, compile cache):
     ``pytest op_tests/test_gemm_a8w4_ptpc.py -m "not basic"``;
  3. benchmark (``main``, not collected by pytest): 6 Qwen3.8-Flash-Next shapes x
     M = 1, 2, 4, ..., 32768 against aiter's FP8 per-token GEMM, per-token quantization and
     then 1x32, every timed output checked: ``python op_tests/test_gemm_a8w4_ptpc.py``.
Every output is graded against an independent fp32 torch reference (``run_torch``,
``run_torch_1x32``); ``MF_SEED`` seeds all inputs.
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


def launched_kernels(fn, attempts=5):
    """Run ``fn`` under the profiler: (its result, names of the GPU kernels it launched).

    The GPU is synchronized first, so the profiled window holds ``fn``'s kernels only. The
    profiler records no GPU event at all for the calls made while another process starts HIP
    on the same GPU (shared MI308X GPUs: a burst of about 0.1 s in which every call comes back
    empty, so back-to-back retries all fail; no empty call in 5000 on a GPU of its own). A call
    with no recorded kernel is therefore profiled again after growing pauses (0.05, 0.1, 0.2
    and 0.4 s), up to ``attempts`` times in all: a call that launches nothing (M = 0) still
    returns [], and whatever is recorded still has to match exactly.
    """
    import time

    for attempt in range(attempts):
        if attempt:
            time.sleep(0.05 * 2 ** (attempt - 1))
        torch.cuda.synchronize()
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
# name: the general GEMM's name carries the tile the entry picks per shape and M, and the
# small-M kernels' names their configuration; what ran is still checked against
# gemm_a8w4_ptpc_kernel_names.
_KERNEL_TIERS = (
    ("gemm_a8w4_ptpc_quant_", "quant"),
    ("gemm_a8w4_ptpc_decode_", "decode"),
    ("gemm_a8w4_ptpc_dec_", "dec2"),
    ("gemm_a8w4_ptpc_tile_", "tile"),
    ("gemm_a8w4_ptpc_bm", "general"),
    ("gemm_a8w4_ptpc_pp_", "large"),
)


def kernel_tiers(names):
    """Tier of each kernel name: ``quant``, the one-launch small-M kernels ``decode``, ``dec2``
    (the role-split decode kernel) and ``tile``, ``general`` (the GEMM with a configurable tile)
    or ``large`` (the persistent large-M kernel); an unknown name is returned as it is, so it
    matches no tier."""
    return tuple(
        next((tier for prefix, tier in _KERNEL_TIERS if name.startswith(prefix)), name)
        for name in names
    )


@functools.lru_cache(maxsize=4)
def cached_weight(n, k, seed):
    """(w, w_scale, prepared weight) for tests that share one weight."""
    w, w_scale = make_weight(n, k, seed)
    return w, w_scale, prepare_gemm_a8w4_ptpc_weight(w, w_scale)


# -- 1x32 activations: one E8M0 scale per token and 32 K -------------------------------------

# The pre-quantized entry and its kernel names, from aiter.ops.flydsl's public (lazy) exports.
flydsl_gemm_a8w4_mxfp8 = aiter.ops.flydsl.flydsl_gemm_a8w4_mxfp8
gemm_a8w4_mxfp8_kernel_names = aiter.ops.flydsl.gemm_a8w4_mxfp8_kernel_names
X_BLOCK = 32
# DeepSeek-V4.1's act_quant: E4M3FN values (largest 448), block amax at least 1e-4.
FP8_FN_MAX = 448.0
X_AMAX_FLOOR = 1e-4


def make_x_1x32(m, k, seed):
    """make_x with every 32-K block times 2**r (r = -6..6): each block of a row has its own
    scale, so a scale read for the wrong block or token shows in the output."""
    x = make_x(m, k, seed).float().view(m, k // X_BLOCK, X_BLOCK)
    gen = _generator(seed + 1)
    r = torch.randint(-6, 7, (m, k // X_BLOCK, 1), generator=gen, device=DEVICE)
    return (x * torch.exp2(r.float())).view(m, k).to(torch.bfloat16)


def quant_1x32_ref(x):
    """1x32 FP8 E4M3FNUZ [m, k] and E8M0 bytes [m, k / 32] of BF16 ``x``: DeepSeek-V4.1's
    reference ``act_quant(x, block_size=32)`` with UE8M0 scales (inference/kernel.py), mapped
    to FNUZ.

    amax = max |x| of the block with NaN ignored (CUDA's fmaxf), at least f32 1e-4;
    e = ceil(log2(amax * f32(1 / 448))) from the f32 bits (the exponent, plus one for a
    nonzero mantissa); q_fn = RNE_e4m3fn(clamp(x / 2**e, -448, 448)). FNUZ doubles the scale:
    byte e + 128 with the same FP8 codes, -0 stored as 0x00 and NaN as 0x80. A block holding
    +-Inf (e = 128, an E8M0 NaN scale in the reference) gets byte 254, as the kernel
    documents; its finite values quantize to 0 and +-Inf to NaN.
    """
    m, k = x.shape
    xf = x.float().view(m, k // X_BLOCK, X_BLOCK)
    a = xf.abs()
    amax = torch.where(a.isnan(), torch.zeros_like(a), a).amax(dim=-1)
    amax = torch.maximum(amax, torch.full_like(amax, X_AMAX_FLOOR))
    inv_max = torch.ones_like(amax) / torch.full_like(amax, FP8_FN_MAX)
    bits = (amax * inv_max).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale_fn = ((e + 127) << 23).view(torch.float32).unsqueeze(-1)
    q_fn = (xf / scale_fn).clamp(-FP8_FN_MAX, FP8_FN_MAX).to(torch.float8_e4m3fn)
    q_fn = q_fn.view(torch.uint8)
    q = torch.where(q_fn == 0x80, torch.zeros_like(q_fn), q_fn)
    q = torch.where((q_fn & 0x7F) == 0x7F, torch.full_like(q_fn, 0x80), q)
    scale = (e + 128).clamp(max=254).to(torch.uint8)
    return q.view(m, k).view(torch.float8_e4m3fnuz), scale


def dequant_1x32(xq, x_scale, dtype=torch.float32):
    """Exact value of a 1x32 activation [m, k]: xq x 2**(x_scale - 127) per 32-K block."""
    m, k = xq.shape
    scale = torch.exp2(x_scale.view(torch.uint8).double() - 127).to(dtype)
    blocks = xq.float().to(dtype).view(m, k // X_BLOCK, X_BLOCK) * scale.unsqueeze(-1)
    return blocks.view(m, k)


def run_torch_1x32(xq, x_scale, w, w_scale):
    """Independent fp32 reference of the 1x32 GEMM: the sum over 32-K blocks of
    2**(e - 127) x sum(q x w), W exact with the prepare flush rule."""
    return fp32_matmul(dequant_1x32(xq, x_scale), dequant_mxfp4(w, w_scale))


def gemm_1x32_path(name):
    """Path of a 1x32 kernel name: ``q4`` or ``h2`` (the one-launch small-M 1x32 kernel's two
    token mappings, from BF16 X or pre-quantized), ``large`` (the persistent large-M kernel's
    1x32 variant), ``mma32`` (the general GEMM on MFMA 32x32x16, which gemm.py takes for tiles
    of whole 32 x 128 MMA rows: block_m % 32 == 0 and block_n % 128 == 0) or ``mma16`` (the
    general GEMM on MFMA 16x16x32, one MMA per 32-K block)."""
    import re

    small = re.match(r"gemm_a8w4_ptpc_tile_(q4|h2)_.*_(1x32|mxfp8)$", name)
    if small:
        return small.group(1)
    if kernel_tiers([name]) == ("large",):
        return "large"
    block_m, block_n = map(int, re.search(r"_bm(\d+)_bn(\d+)_", name).groups())
    return "mma32" if block_m % 32 == 0 and block_n % 128 == 0 else "mma16"


# =============================================================================================
# Region 1: basic functionality (@pytest.mark.basic)
# prepare_gemm_a8w4_ptpc_weight + flydsl_gemm_a8w4_ptpc on each target shape with one
# representative M, one case per dispatch tier: M = 1 on tp2_qkvz (dec2, the role-split decode
# kernel), M = 3 on tp4_qkv (decode kernel), M = 20 on tp4_o (tile kernel, two token tiles),
# M = 33..1000 with non-powers of two (per-token quant, then the general GEMM, each with the
# tile its shape's dispatch table picks) and M = 4096 (quant, then the persistent large-M
# kernel); asserts that the launched FlyDSL kernels are the ones the dispatch predicts. The
# 1x32 activations: both entries at a small M (the one-launch small-M 1x32 kernel), at two
# general-GEMM M (16x16x32 and 32x32x16 MMAs) and at M > 2048 (test_gemm_1x32).
# =============================================================================================

BASIC_CASES = [
    ("tp2_qkvz", 1),
    ("tp4_qkv", 3),
    # tp4_o: the only target shape whose dispatch table keeps the small-M kernels past M = 16
    # (up to 20). Two K windows are covered in region 2.9 (N = 4160).
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


# 1x32 activations, one case per entry at four M. tp2_qkv, M = 3: one launch of the small-M
# 1x32 kernel (mapping q4, 3 of its 4 token columns; 80 CTAs of 5 or 6 row blocks), which
# quantizes BF16 X itself or takes the pre-quantized X as it is. Otherwise x_quant="1x32" runs
# the 1x32 quantization and then the 1x32 GEMM, which scales every 32-K block;
# flydsl_gemm_a8w4_mxfp8 runs that GEMM on an activation quantized beforehand. The general
# GEMM: tp2_qkv, M = 100 (its MMA path left to the 1x32 dispatch table; 32x32x16 when written)
# and tp4_qkvz, M = 200 (a 64 x 128 tile on MFMA 32x32x16, asserted: the main path of the
# middle range); tp4_o, M = 2049 the persistent large-M kernel's 1x32 variant (17 M blocks,
# the last with 1 row). Where the 1x32 dispatch switches paths is pinned in
# test_1x32_dispatch.
BASIC_1X32 = [
    ("tp2_qkv", 3, "q4"),
    ("tp2_qkv", 100, None),
    ("tp4_qkvz", 200, "mma32"),
    ("tp4_o", 2049, "large"),
]


@pytest.mark.basic
@pytest.mark.parametrize("entry", ["x_quant_1x32", "mxfp8"])
@pytest.mark.parametrize(
    "shape, m, path", BASIC_1X32, ids=[f"{s}_m{m}" for s, m, _ in BASIC_1X32]
)
def test_gemm_1x32(shape, m, path, entry):
    """``flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")`` launches the kernels that
    ``gemm_a8w4_ptpc_kernel_names(..., x_quant="1x32")`` predicts (the small-M 1x32 kernel, or
    the 1x32 quantization and the 1x32 GEMM) and matches the fp32 reference on the reference
    quantization; ``flydsl_gemm_a8w4_mxfp8`` on that quantization (uint8 E8M0) launches the
    small-M kernel's pre-quantized variant or the GEMM alone and gives the same output bit for
    bit.
    """
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    x = make_x_1x32(m, k, case_seed("x_1x32", shape, m))
    xq, x_scale = quant_1x32_ref(x)
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant="1x32")
    if path in ("q4", "h2"):
        assert len(names) == 1 and gemm_1x32_path(names[0]) == path, names
        mxfp8_names = (names[0].removesuffix("_1x32") + "_mxfp8",)
    else:
        assert names[0].startswith("gemm_a8w4_ptpc_quant1x32_"), names
        assert names[1].endswith("_1x32") and len(names) == 2, names
        assert path is None or gemm_1x32_path(names[1]) == path, names
        mxfp8_names = names[1:]
    if entry == "mxfp8":
        call = functools.partial(flydsl_gemm_a8w4_mxfp8, xq, x_scale, weight)
        names = gemm_a8w4_mxfp8_kernel_names(m, n, k)
        assert names == mxfp8_names, names
    else:
        call = functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32")
    call()  # compile outside the profiler
    out, kernels = launched_kernels(call)
    assert kernels == list(names), kernels
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (m, n)
    ref = run_torch_1x32(xq, x_scale, w, w_scale)
    check_output(out, ref, msg=f"{shape}_m{m}_{entry}")
    if entry == "mxfp8":
        y = flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")
        assert torch.equal(out.view(torch.int16), y.view(torch.int16))


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
    """Off the dispatch table the general GEMM takes 32 x 64 tiles up to M = 128 and 64-row
    tiles above, 64 channels wide where N % 128 == 64 (N = 320), both with 256 K per stage:
    the last row block one row short (M = 63) or holding one row (M = 65 of 32-row blocks,
    M = 129 of 64-row blocks), K = 256 / 512 / 768 (1, 2, 3 K tiles: fewer than or about the
    prefetch depth of 2, all stages peeled, no runtime loop iteration), and M = 129 with 3 row
    blocks in super tiles of 4 (one CTA of each super tile has no tile and exits).

    The target shapes are on the table, with N % 128 == 0 and K >= 1536; the benchmark uses
    powers of two for M and the basic cases one M per shape.
    """
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    x = make_x(m, k, case_seed("x", m, n, k))
    out = flydsl_gemm_a8w4_ptpc(x, prepare_gemm_a8w4_ptpc_weight(w, w_scale))
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")


def test_lds_epilogue_ragged_n():
    """The general GEMM's epilogue through LDS (every tile with 256 K per stage) on a partly
    filled last N block: 128-channel tiles on N = 320 leave the third block half empty, whose
    16-byte row pieces past N are not stored, and M = 65 leaves 1 row in the last 64-row block;
    written through ``out=`` (the head of a NaN buffer).

    The dispatch never runs this: N % 128 == 64 gets 64-channel tiles, and the default tile
    past M = 2048 (128 K per stage) stores directly. The kernel is compiled and launched here.
    """
    from aiter.ops.flydsl.kernels.gemm_a8w4_ptpc import gemm as gemm_kernels

    m, n, k = 65, 320, 512
    cfg = gemm_kernels.GemmConfig(64, 128, 2, 2, 1, 1, 256)
    assert gemm_kernels.gemm_config_supported(n, k, cfg), cfg
    assert gemm_kernels._shuffled_store(cfg), cfg  # the epilogue through LDS
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(m, k, case_seed("lds_epilogue", m, n, k))
    xq, x_scale = a8w4_module._quant_per_token(x)
    buffer = torch.full((m + 64, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
    out = buffer[:m]
    ptr = a8w4_module.ptr_arg
    a8w4_module._run_compiled(
        gemm_kernels.compile_gemm_a8w4_ptpc(n, k, x.device.index, cfg),
        ptr(xq),
        ptr(x_scale),
        ptr(weight.q),
        ptr(weight.d),
        ptr(weight.s),
        ptr(out),
        m,
        a8w4_module.current_stream_arg(x.device),
    )
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")
    assert bool(buffer[m:].isnan().all())


# -- 2.2 per-token quantization, bit-exact -----------------------------------------------------


# (rows, K, lanes per row): up to 128 rows with K % 512 == 0 one CTA per row over K / 8 lanes
# (K = 1536: 3 waves, 3072: 6, 8192: 16, the most a CTA holds; combining their maxima through
# LDS), more rows one wave per row (64 lanes), K % 512 != 0 32 lanes (two rows per wave).
QUANT_BITEXACT = [
    (997, 2560, 64),
    (997, 256, 32),
    (128, 1536, 192),
    (129, 1536, 64),
    (128, 3072, 384),
    (128, 8192, 1024),
]


@pytest.mark.parametrize(
    "m, k, lanes", QUANT_BITEXACT, ids=[f"m{m}_k{k}" for m, k, _ in QUANT_BITEXACT]
)
def test_quant_bitexact(m, k, lanes):
    """The per-token quantization equals the reference bit for bit: scale = amax / 240 and
    q = RNE(x / scale) with the IEEE quotient (the kernel computes it as a reciprocal
    multiply plus one FMA correction, and divides per element when scale < 2**-64).
    Includes an all-zero row (scale 1), a row of exact FP8 rounding midpoints where a plain
    reciprocal multiply rounds the other way, the same row times 2**-70, and a row of -0 and
    negatives that round to zero (no byte may be 0x80, NaN in FNUZ), in each configuration
    of the kernel: 128 rows run one row per CTA over 3, 6 or 16 waves, 129 rows (the switch)
    and 997 one wave per row, K = 256 two rows per wave.

    End-to-end grading cannot see a few flipped FP8 codes: they move the GEMM output by far
    less than its tolerance.
    """
    assert a8w4_module.quant_config(m, k).lanes == lanes  # the configuration under test
    gen = _generator(case_seed("quant", k))
    x = torch.randn((m, k), generator=gen, device=DEVICE) * 3
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


GRAPH_CASES = [
    ("tp4_o", 5, "decode"),
    ("tp2_qkv", 4, "dec2"),
    ("tp4_o", 20, "tile"),
    ("tp4_o", 48, "general"),
    ("tp4_o", 2049, "large"),
]


@pytest.mark.parametrize(
    "shape, m, tier",
    GRAPH_CASES,
    ids=["m5_decode", "m4_dec2", "m20_tile", "m48", "m2049_large"],
)
def test_cuda_graph_replay(shape, m, tier):
    """The whole call (its activation buffers and output allocated inside the capture) is
    captured in a CUDA graph and replays on new input: no host synchronization. M = 5 runs
    the decode kernel, M = 4 on tp2_qkv the role-split decode kernel (its weight waves spin on
    an LDS counter that the CTA zeroes first) and M = 20 the tile kernel (all three quantize X
    themselves), M = 48 the general GEMM, M = 2049 the persistent large-M kernel.

    The basic cases and the benchmark call the op eagerly.
    """
    # Each M asserts the tier it is here for: tp4_o's table keeps the tile kernel up to M = 20.
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
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
    2**31. N = 320 runs the general GEMM (past M = 2048 off the large kernel: 64 x 128 tiles,
    the last 128-channel block half empty and its stores masked) over 65538 row blocks, last
    block 33 rows, in a 1-D grid of 4 CTAs per row block (the fourth without a tile); N = 256
    the persistent large-M kernel over 32769 tiles of 128 rows (last tile 97 rows, starting
    exactly at byte 2**31 of Y and of FP8 X). Every kernel rebases its buffers at the first
    row of its CTA or tile. X is zero except the first tile, the tiles around each 2**31 /
    2**32 byte boundary and the end; Y starts as NaN. The data rows match the reference and
    every other row is exactly zero, so a lost or misplaced store shows.

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
    """Off the dispatch table (each target shape switches at the M its table gives, see
    test_target_shape_dispatch), M > 2048 with N % 256 == 0 and K >= 512 runs the persistent
    large-M kernel; M = 2048, N % 256 != 0 or K = 256 run the general GEMM (M = 2049, N = 320:
    64 x 128 tiles, the last 128-channel block half empty). The predicted and the launched
    kernels are that one and the output matches the reference. M = 2049, N = 256, K = 512 is
    the large kernel's smallest case: 17 tiles (fewer than CUs, one per CTA), a 1-row last
    tile, and K = 512 leaves no K pair loop iteration between the peeled first and last pairs.

    The basic cases and the benchmark run target shapes (N % 256 == 0, K >= 1536) at
    M = 4096 or powers of two: never at M = 2049, never ruled out by N or K, and their
    large-M tiles are full.
    """
    # By kernel name tier: the general GEMM's kernel name carries the tile the entry picks.
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    assert kernel_tiers(names) == ("quant", "large" if large else "general"), names
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(m, k, case_seed("large_dispatch", m, n, k))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
    out, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert kernels == list(names), kernels
    check_output(out, run_torch(x, w, w_scale), msg=f"m{m}_n{n}_k{k}")


@pytest.mark.parametrize("x_quant", ["per_token", "1x32"])
def test_large_m_tiles(x_quant):
    """The persistent kernel with more tiles than CUs and a partial last M block, writing
    through ``out=`` (the head of a NaN buffer): N = 1024 (4 N blocks) and one 128-row M block
    more than the CUs take, the last with 40 rows (80 CUs: M = 2600, 84 tiles), so some CTAs
    run a second tile, one of them a full tile and then a tail tile; K = 768 runs the K pair
    loop once. Every row is written and nothing past it. With 1x32 X both entries run the
    kernel's 1x32 variant, flydsl_gemm_a8w4_mxfp8 with the scales as torch.float8_e8m0fnu,
    and their outputs are equal bit for bit.

    The benchmark's large-M cases have M a multiple of 128 and K >= 1536, the op allocates
    their output, and its 1x32 cases take BF16 X (x_quant="1x32") only.
    """
    n, k = 1024, 768
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    m = max(num_cus // (n // 256), 16) * 128 + 40
    # By kernel name tier rather than the large kernel's name constant.
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant=x_quant)
    assert len(names) == 2 and names[0].startswith("gemm_a8w4_ptpc_quant"), names
    assert kernel_tiers(names)[1] == "large", names
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    if x_quant == "1x32":
        x = make_x_1x32(m, k, case_seed("large_tiles_1x32", m))
        xq, x_scale = quant_1x32_ref(x)
        ref = run_torch_1x32(xq, x_scale, w, w_scale)
        e8m0 = x_scale.view(torch.float8_e8m0fnu)
        calls = [
            functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32"),
            functools.partial(flydsl_gemm_a8w4_mxfp8, xq, e8m0, weight),
        ]
    else:
        x = make_x(m, k, case_seed("large_tiles", m))
        ref = run_torch(x, w, w_scale)
        calls = [functools.partial(flydsl_gemm_a8w4_ptpc, x, weight)]
    outs = []
    for call in calls:
        buffer = torch.full(
            (m + 128, n), float("nan"), dtype=torch.bfloat16, device=DEVICE
        )
        out = buffer[:m]
        assert call(out=out) is out
        check_output(out, ref, msg=f"m{m}_n{n}_k{k}_{x_quant}")
        assert bool(buffer[m:].isnan().all())
        outs.append(out.view(torch.int16))
    assert all(torch.equal(outs[0], o) for o in outs[1:])


# (tile order of the persistent kernel: 0 = plain, 1 / 2 / 4 = N panels; CTAs short of a
# multiple of 4)
LARGE_M_ORDERS = [(0, 0), (1, 0), (2, 0), (4, 0), (4, 2)]


@pytest.mark.parametrize("x_quant", ["per_token", "1x32"])
@pytest.mark.parametrize(
    "panels, short",
    LARGE_M_ORDERS,
    ids=["plain", "panels1", "panels2", "panels4", "panels4_ctas_not_x4"],
)
def test_large_m_tile_orders(panels, short, x_quant):
    """The persistent kernel computes every tile once in each tile order, whatever M (the entry
    picks the order per call, see test_large_m_panel_choice; the kernel takes the plain order
    when a grid of fewer CTAs than CUs is not a multiple of 4, which a panel order cannot share
    evenly among the XCDs). N = 6656 (26 N blocks: two panels of 13, four of 7 / 6 / 7 / 6),
    K = 512, a grid of the CU count rounded down to a multiple of 4, or 2 fewer (not a multiple
    of 4: the plain order at compile time). M = 1: 26 tiles, fewer than CTAs and not a multiple
    of 4 (the plain order at run time); M = 129: 52 tiles, one per CTA; M = 3001 and 4097: 624
    and 858 tiles over several rounds (4097: XCD parts of 214 and 215 tiles, a 1-row last M
    block). Written into NaN-filled outputs. With 1x32 X the same for the kernel's 1x32 variant
    (x_block_scale), which keeps the tile orders, on the reference 1x32 quantization.

    The benchmark and the other tests run each target shape in the one order the entry picks
    for it at their M; a tile mapped twice or not at all in another order would go unseen.
    """
    n, k = 6656, 512
    device = torch.cuda.current_device()
    num_cus = torch.cuda.get_device_properties(device).multi_processor_count
    ctas = num_cus // 4 * 4 - short
    launch = a8w4_module.compile_gemm_a8w4_ptpc_large(
        n, k, ctas, device, panels, x_block_scale=x_quant == "1x32"
    )
    w, w_scale, weight = cached_weight(n, k, case_seed("w", n, k))
    ptr = a8w4_module.ptr_arg
    for m in [1, 129, 3001, 4097]:
        if x_quant == "1x32":
            x = make_x_1x32(m, k, case_seed("tile_orders_1x32", m))
            xq, x_scale = quant_1x32_ref(x)
            ref = run_torch_1x32(xq, x_scale, w, w_scale)
        else:
            x = make_x(m, k, case_seed("tile_orders", m))
            xq, x_scale = a8w4_module._quant_per_token(x)
            ref = run_torch(x, w, w_scale)
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=DEVICE)
        a8w4_module._run_compiled(
            launch,
            ptr(xq),
            ptr(x_scale),
            ptr(weight.q),
            ptr(weight.d),
            ptr(weight.s),
            ptr(out),
            m,
            a8w4_module.current_stream_arg(x.device),
        )
        check_output(out, ref, msg=f"panels={panels} m{m} {x_quant}")


# (shape, M, tile order the entry compiles the large-M kernel with). From the first large M
# up to M = 1152 (K = 2560: X of 9 M blocks fits an XCD's 3 MiB) one panel, then the plain
# order where the N blocks are a multiple of 4 and else two or four panels; tp2_o (K = 3072)
# two panels from its first large M on; tp4_o one (its weights fit 3 MiB).
LARGE_M_PANELS = [
    ("tp2_qkvz", 1152, 1),
    ("tp2_qkvz", 1153, 0),
    ("tp2_qkv", 1152, 1),
    ("tp2_qkv", 1153, 4),
    ("tp4_qkvz", 1152, 1),
    ("tp4_qkvz", 1153, 0),
    ("tp4_qkv", 1152, 1),
    ("tp4_qkv", 1153, 2),
    ("tp2_o", 1025, 2),
    ("tp4_o", 1025, 1),
]


@pytest.mark.parametrize(
    "shape, m, panels", LARGE_M_PANELS, ids=[f"{s}_m{m}" for s, m, _ in LARGE_M_PANELS]
)
def test_large_m_panel_choice(monkeypatch, shape, m, panels):
    """The tile order the entry gives the persistent kernel on the target shapes (xcd_panels
    on 80 CUs: these M are where it changes, above them it no longer depends on M; none on a
    CU count that is not a multiple of 4), the order the call compiles the kernel with on
    this GPU, and its output against the reference.

    Every order has the same kernel name, so the profiler cannot tell which one ran, and the
    benchmark does not check it.
    """
    n, k = SHAPES[shape]
    assert a8w4_module.xcd_panels(m, n, k, 80) == panels
    assert a8w4_module.xcd_panels(m, n, k, 78) == 0
    compile_large = a8w4_module.compile_gemm_a8w4_ptpc_large
    compiled = []

    def spy(n, k, num_cus, device_index, panels=0):
        compiled.append(panels)
        return compile_large(n, k, num_cus, device_index, panels)

    monkeypatch.setattr(a8w4_module, "compile_gemm_a8w4_ptpc_large", spy)
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    x = make_x(m, k, case_seed("panel_choice", shape, m))
    out = flydsl_gemm_a8w4_ptpc(x, weight)
    num_cus = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    assert compiled == [a8w4_module.xcd_panels(m, n, k, num_cus)], compiled
    check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{m}")


# -- 2.9 small M (M <= 32): the one-launch decode, dec2 and tile kernels ----------------------

# (m, n, k, small-M kernel kind or None for the per-token quant and the general GEMM, token slots
# of the decode / dec2 kernel or 16-token tiles of the tile kernel)
SMALL_M_DISPATCH = [
    (8, 2560, 1536, "decode", 8),
    (9, 2560, 1536, "tile", 1),
    (16, 2560, 1536, "tile", 1),
    (17, 2560, 1536, "tile", 2),
    # The tile kernel's 32-token limit only shows off the dispatch table (the target shapes'
    # tables end the small-M kernels at M = 16 or 20): N = 4160, two K windows.
    (32, 4160, 2560, "tile", 2),
    (33, 4160, 2560, None, 0),
    (5, 128, 768, None, 0),
    (20, 5184, 12288, None, 0),
    # LDS: the next kernel that fits runs where one would exceed 64 KiB.
    (8, 64, 7168, "tile", 1),  # decode too large: tile kernel, two K windows
    (2, 8192, 12288, "decode", 2),  # dec2 too large: decode kernel
    (3, 5632, 14336, None, 0),  # dec2, decode and tile too large
    (4, 64, 12288, "decode", 4),  # decode kernel just fits (64592 B)
    (2, 5632, 13824, "dec2", 2),  # dec2 just fits (64832 B)
]


@pytest.mark.parametrize(
    "m, n, k, kind, tokens",
    SMALL_M_DISPATCH,
    ids=[f"m{m}_n{n}_k{k}" for m, n, k, _, _ in SMALL_M_DISPATCH],
)
def test_small_m_dispatch(m, n, k, kind, tokens):
    """Where the dispatch leaves M to the small-M kernels (off the table; up to M = 16 or 20 on
    the target shapes, tp4_o here): M <= 8 runs the decode kernel alone (dec2 on the widest N,
    see test_target_shape_dispatch), M = 9..16 the tile kernel with one 16-token tile,
    M = 17..32 with two, up to the tile kernel's 32-token limit; M = 33 and K % 512 != 0
    (K = 768) run the per-token quant and the general GEMM. decode and dec2 hold the FP8 token
    rows of all of K in LDS and the tile kernel those of a K window: a kernel that would exceed
    64 KiB gives way to the next one that fits (dec2, decode, tile, then the quant and the
    general GEMM), and one just under the limit still runs (its LDS estimate is not below what
    it allocates). The predicted and the launched kernels are those and the output matches the
    reference.

    The benchmark runs M = 8 and 16 (one side of each boundary), all its shapes have
    K % 512 == 0 and K <= 3072, and it does not check which kernels ran.
    """
    config = a8w4_module.small_m_config(m, n, k)
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    if kind is None:
        assert config is None, config
        # By kernel name tier: the general GEMM's kernel name carries the tile the entry picks.
        assert kernel_tiers(names) == ("quant", "general"), names
    else:
        assert config is not None and config[0] == kind and config[3] == tokens, config
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
    (3, 6656, 2560),
    (7, 6656, 2560),
    # tp2_qkv: the tile kernel on 6 row blocks per CTA, so that its 416 row blocks fill 80 CTAs.
    (9, 6656, 2560),
    (10, 3584, 2560),
    # N = 4160, off the dispatch table (its tile kernel configuration is tp4_qkvz's: 4 row
    # blocks x 4 K parts, two token tiles, two K windows), which runs M = 17..32 of the target
    # shapes on the general GEMM.
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
    (the decode kernels' stores past M fall outside the buffer bound), so every row matches
    the reference and nothing past ``out`` changes. M = 3 and 7: decode kernel with 4 and 8
    token slots (M = 7: 80 CTAs over 416 row blocks, 5 or 6 each); M = 3 on N = 6656: dec2 with
    4 token slots, one X wave per slot, the last with no row to quantize; M = 9 on N = 6656 and
    M = 10: tile kernel, one 16-token tile (M = 9: 80 CTAs over 416 row blocks, 5 or 6 each in
    6 slots); M = 23: two token tiles over two K windows, the K-part sums reusing the token-row
    LDS; N = 5184: 41 tile CTAs over 324 row blocks (7 or 8 each), so some wave slots compute a
    neighbouring block and store nothing; K = 8192: six K windows.

    The benchmark's small-M cases are the token capacities (M = 1, 2, 4, 8, 16), its shapes give
    the tile kernel one K window, and the op allocates its output.
    """
    # The configuration the dispatch launches (small_m_config's, tp2_qkv's tile kernel spread
    # over more CTAs).
    config = a8w4_module._small_config(m, n, k)
    assert config is not None, config
    slots = config[3] if config[0] in ("decode", "dec2") else 16 * config[3]
    assert m < slots, config
    # The dispatch runs that kernel: a dispatch table may run the general GEMM where a small-M
    # kernel fits, which would silently skip what this test is for.
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    assert names == (a8w4_module.small_m_kernel_name_of(config),), names
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
    the float above 240 (as in test_quant_max_quotient_and_nonfinite: stored as -240).
    """
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


# (m, n, small-M kernel the dispatch runs); K = 2560 and N >= K for the one-hot weight.
SMALL_M_QUANT = [
    (2, 4160, "decode"),
    (4, 6656, "dec2"),
    (5, 4160, "decode"),
    (10, 4160, "tile"),
    (23, 4160, "tile"),
]


@pytest.mark.parametrize(
    "m, n, kind",
    SMALL_M_QUANT,
    ids=["m2", "m4_dec2", "m5", "m10", "m23"],
)
def test_small_m_quant_bitexact(m, n, kind):
    """The small-M kernels quantize X inside each CTA exactly like the reference. With a
    one-hot weight (channel c reads K position c % K with weight 1.0; N >= K) the output is
    Y[t, c] = bf16(xq[t, c % K] * x_scale[t]) bit for bit, which exposes every FP8 code and
    every scale. Rows: random, all zero (scale 1), FP8 rounding midpoints (a plain reciprocal
    multiply rounds some the other way), two rows with scale < 2**-64 (per-element division),
    -0 with negatives that round to zero (a NaN code 0x80 would turn the whole output row
    NaN) and a quotient just above 240 (no clamp: it must round to -240, not NaN). M = 2 and
    5: decode kernel (2 and 8 token slots); M = 4 on N = 6656: dec2, whose X waves quantize
    one row each (midpoints, quotient above 240, scale < 2**-64, all zero); M = 10: tile
    kernel; M = 23: two token tiles, both holding special rows, over two separately quantized
    K windows.

    test_quant_bitexact covers the separate quantization kernel of the other paths; with
    random weights a few flipped FP8 codes move the output far less than its tolerance.
    """
    # N = 4160 is off the dispatch table, so that M = 23 runs the tile kernel (the target
    # shapes run M = 17..32 on the general GEMM); N = 6656 (tp2_qkv) runs dec2 up to M = 4.
    k = 2560
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
    assert config is not None and config[0] == kind, config
    tiers = kernel_tiers(gemm_a8w4_ptpc_kernel_names(m, n, k))
    assert tiers == (kind,), tiers  # the dispatch runs that kernel
    out = flydsl_gemm_a8w4_ptpc(x, weight)
    xq, x_scale = quant_per_token(x)
    expected = (xq.float()[:, columns] * x_scale).to(torch.bfloat16)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


SMALL_M_NONFINITE = [
    ("tp4_o", 5, "decode"),
    ("tp2_qkv", 4, "dec2"),
    ("tp4_o", 20, "tile"),
    # x_quant="1x32": the small-M 1x32 kernel's two mappings (h2 with three token sets).
    ("tp4_o", 4, "q4"),
    ("tp4_o", 20, "h2"),
]


@pytest.mark.parametrize(
    "shape, m, kind",
    SMALL_M_NONFINITE,
    ids=["m5_decode", "m4_dec2", "m20_tile", "m4_1x32_q4", "m20_1x32_h2"],
)
def test_small_m_nonfinite_rows(shape, m, kind):
    """A token row holding NaN or +-Inf gets scale NaN or Inf inside the small-M kernels
    and turns only its own output row NaN, as in the reference; every other token of the
    CTA, in the same and the other token tile, still matches the reference (dec2: rows
    quantized by other X waves of the CTA). With x_quant="1x32" (q4, h2) NaN stays out of
    its block's maximum and a block holding +-Inf gets scale byte 254; both still turn only
    their own token's output row NaN, the other token columns and sets matching.

    The basic cases and the benchmark only pass finite activations;
    test_quant_max_quotient_and_nonfinite covers the separate quantization kernel.
    """
    # tp4_o's table keeps the small-M kernels up to M = 20; the tier or mapping is asserted.
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    x_quant = "1x32" if kind in ("q4", "h2") else "per_token"
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant=x_quant)
    if x_quant == "1x32":
        assert len(names) == 1 and gemm_1x32_path(names[0]) == kind, names
    else:
        assert kernel_tiers(names) == (kind,), names
    x = make_x(m, k, case_seed("small_nonfinite", m))
    bad = [1, 3] if m <= 8 else [1, 3, 17]
    x[1, 7] = float("nan")
    x[3, 100] = float("inf")
    if m > 8:
        x[17, 5] = -float("inf")
    good = [t for t in range(m) if t not in bad]
    out = flydsl_gemm_a8w4_ptpc(x, weight, x_quant=x_quant)
    if x_quant == "1x32":
        ref = run_torch_1x32(*quant_1x32_ref(x), w, w_scale)
    else:
        ref = run_torch(x, w, w_scale)
    assert bool(out[bad].isnan().all()) and bool(ref[bad].isnan().all())
    check_output(out[good], ref[good], msg=f"m{m} finite rows")


# -- 2.10 the target shapes' dispatch: where each switches kernels -----------------------------

# (shape, the last M of a tier, its tier, the tier from M + 1). The small-M kernel is picked by
# its limits (dec2 on the two widest shapes up to M = 2 / 4); the dispatch table of each shape
# then hands M to the per-token quant and the general GEMM, and to the large-M kernel.
TARGET_DISPATCH = [
    ("tp2_qkvz", 2, "dec2", "decode"),
    ("tp2_qkv", 4, "dec2", "decode"),
    ("tp2_qkvz", 16, "tile", "general"),
    ("tp2_qkv", 16, "tile", "general"),
    ("tp2_o", 16, "tile", "general"),
    ("tp4_qkvz", 16, "tile", "general"),
    ("tp4_qkv", 16, "tile", "general"),
    ("tp4_o", 20, "tile", "general"),
    ("tp2_qkvz", 128, "general", "large"),
    ("tp2_qkv", 512, "general", "large"),
    ("tp2_o", 1024, "general", "large"),
    ("tp4_qkvz", 1024, "general", "large"),
    ("tp4_qkv", 1024, "general", "large"),
    ("tp4_o", 1024, "general", "large"),
]


@pytest.mark.parametrize(
    "shape, m, below, above",
    TARGET_DISPATCH,
    ids=[f"{s}_m{m}_{m + 1}" for s, m, _, _ in TARGET_DISPATCH],
)
def test_target_shape_dispatch(shape, m, below, above):
    """Each target shape switches kernels at its own M: dec2 to the decode kernel after M = 2
    (tp2_qkvz) or 4 (tp2_qkv); the small-M kernels to the per-token quant and the general GEMM
    after M = 16 (tp4_o: 20); the general GEMM to the persistent large-M kernel after M = 128
    (tp2_qkvz), 512 (tp2_qkv) or 1024 (tp2_o, tp4_qkvz, tp4_qkv, tp4_o). On both sides the
    predicted kernels are of the expected tier, the launched ones are the predicted ones and
    the output matches the reference. The first M of the general GEMM leaves its last row
    block partly filled (17 of 32 rows; 1 of 16 on tp2_o, 5 on tp4_o: fewer 16-byte row pieces
    than threads in the epilogue through LDS), that of the large-M kernel a 1-row last tile.

    The benchmark runs one side of each switch at most (powers of two: neither M = 3, 5, 17,
    20, 21, 129, 513 nor 1025) and does not check which kernels ran.
    """
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    for mm, tier in [(m, below), (m + 1, above)]:
        names = gemm_a8w4_ptpc_kernel_names(mm, n, k)
        expected = (tier,) if tier in ("decode", "dec2", "tile") else ("quant", tier)
        assert kernel_tiers(names) == expected, names
        x = make_x(mm, k, case_seed("target_dispatch", shape, mm))
        flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
        out, kernels = launched_kernels(lambda x=x: flydsl_gemm_a8w4_ptpc(x, weight))
        assert kernels == list(names), kernels
        check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{mm}")


# (shape, first M of a general GEMM tile): tile switches inside the general GEMM's range. The
# others (M = 33; tp2_o's at 65 and 129, tp4_qkv's and tp4_o's at 65) are checked only by the
# benchmark at the power of two below them; a partly filled last row block works the same in
# every tile (test_target_shape_dispatch, test_tile_boundaries).
TILE_SWITCHES = [
    ("tp2_qkvz", 65),
    ("tp2_qkv", 65),
    ("tp2_qkv", 129),
    ("tp4_qkvz", 65),
    ("tp4_qkvz", 129),
    ("tp4_qkv", 129),
    ("tp4_qkv", 257),
    ("tp4_o", 129),
    ("tp4_o", 257),
]


@pytest.mark.parametrize(
    "shape, m", TILE_SWITCHES, ids=[f"{s}_m{m}" for s, m in TILE_SWITCHES]
)
def test_target_shape_tile_switches(shape, m):
    """Inside the general GEMM's range a target shape's dispatch table switches the tile (its
    size, pipeline depths, XCD super tile or K groups) from one M bucket to the next: the
    predicted general GEMM at M differs from that at M - 1, the launched kernels are the
    predicted ones and the output matches the reference with the new tile's last row block
    holding 1 row (17 of 48 on tp2_qkv at M = 65). At M = 129 the quantization also switches
    from one CTA to one wave per row.

    The benchmark runs M - 1 (powers of two), with full row blocks, and does not check which
    kernels ran.
    """
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    below = gemm_a8w4_ptpc_kernel_names(m - 1, n, k)
    names = gemm_a8w4_ptpc_kernel_names(m, n, k)
    assert kernel_tiers(below) == kernel_tiers(names) == ("quant", "general"), names
    assert below[1] != names[1], names
    x = make_x(m, k, case_seed("tile_switch", shape, m))
    flydsl_gemm_a8w4_ptpc(x, weight)  # compile outside the profiler
    out, kernels = launched_kernels(lambda: flydsl_gemm_a8w4_ptpc(x, weight))
    assert kernels == list(names), kernels
    check_output(out, run_torch(x, w, w_scale), msg=f"{shape}_m{m}")


# -- 2.11 1x32 activations: x_quant="1x32" and flydsl_gemm_a8w4_mxfp8 -------------------------
# The benchmark times x_quant="1x32" on Gaussian BF16 X at powers of two of M only; the basic
# cases run each 1x32 entry at four M (the small-M 1x32 kernel, the general GEMM's two MMA paths
# and the large-M kernel). The small-M 1x32 kernel's NaN / +-Inf rows are in 2.9.


def _bf16_block(head, fill=(0x3F80, 0x3F00, 0xBF40, 0x3E80)):
    """32 BF16 bit patterns: ``head``, then ``fill`` repeated (default 1, 0.5, -0.75, 0.25)."""
    return list(head) + [fill[i % len(fill)] for i in range(X_BLOCK - len(head))]


_TINY = (0x3800, 0xB7A0, 0x3700, 0x0001)  # 3.1e-5, -1.9e-5, 7.6e-6, a subnormal
_SUBNORMALS = (0x0001, 0x0010, 0x0040, 0x007F, 0x8001, 0x8040)

# 32-K blocks the 1x32 quantization must get right: name -> (BF16 bit patterns, scale byte
# derived by hand from the DeepSeek rule: log2 of amax / 448 rounded up, + 128).
QUANT_1X32_BLOCKS = {
    "zero": ([0x0000] * X_BLOCK, 106),
    "plus_minus_zero": ([0x0000, 0x8000] * (X_BLOCK // 2), 106),
    # amax with the mantissa 1.75: 1.75 / 448 = 2**-8 exactly, kept (aiter rounds it up).
    "amax_1.75": (_bf16_block([0x3FE0], [0x3F00]), 120),
    "amax_-1.75": (_bf16_block([0xBFE0, 0x3FDF], [0xBF00]), 120),
    "amax_1.7578": (_bf16_block([0x3FE1], [0x3F00]), 121),
    "amax_1.75*2^100": (_bf16_block([0x71E0], [0x7100, 0x3F80]), 220),
    # amax around the 1e-4 floor (BF16 0x38D1 < 1e-4 < 0x38D2), the largest amax that keeps
    # the floor's scale (1.75 x 2**-14) and the next one, and far below the floor.
    "amax_below_1e-4": (_bf16_block([0x38D1], _TINY), 106),
    "amax_above_1e-4": (_bf16_block([0xB8D2], _TINY), 106),
    "amax_1.75*2^-14": (_bf16_block([0x38E0], _TINY), 106),
    "amax_1.7578*2^-14": (_bf16_block([0x38E1], _TINY), 107),
    "amax_1e-30": (_bf16_block([0x0DA2], [0x0D00]), 106),
    "subnormals": (_bf16_block(_SUBNORMALS, [0x0001]), 106),
    "2^-120_and_subnormals": (_bf16_block((0x0380,) + _SUBNORMALS, [0x0001]), 106),
    # BF16's largest magnitudes, and tiny values next to a huge one.
    "bf16_max": (_bf16_block([0x7F7F, 0x3F80, 0x0001, 0x7F00]), 248),
    "bf16_-max": (_bf16_block([0xFF7F, 0x3F80, 0x7F00]), 248),
    "amax_1.75*2^127": (_bf16_block([0x7F60, 0x3F80, 0x7E00]), 247),
    "huge_and_tiny": (_bf16_block([0x7F00, 0x0001, 0x0080, 0x3F80]), 247),
    # Scale 2**-7: 1.1875, 1.0625, 0.53125 and 0.59375 x 128 are FP8 rounding ties.
    "ties": (_bf16_block([0x3F98, 0x3F88, 0xBF88, 0x3F08, 0x3F18, 0xBF18]), 120),
    # NaN is left out of the maximum and stored as 0x80; +-Inf gives byte 254.
    "nan": (_bf16_block([0x7FC0, 0x4000]), 121),
    "-nan": (_bf16_block([0xFFC0, 0x4000]), 121),
    "signaling_nan": (_bf16_block([0x7F81, 0x4000]), 121),
    "all_nan": ([0x7FC0] * X_BLOCK, 106),
    "nan_and_zero": ([0x7FC0] + [0x0000] * (X_BLOCK - 1), 106),
    "inf": (_bf16_block([0x7F80, 0x4000]), 254),
    "-inf": (_bf16_block([0xFF80, 0x4000]), 254),
    "inf_and_nan": (_bf16_block([0x7F80, 0x7FC0, 0x4000]), 254),
    "all_inf": ([0x7F80] * X_BLOCK, 254),
}


def _bf16_rows(bit_rows):
    """BF16 tensor of bit patterns (Python ints 0..0xFFFF)."""
    bits = torch.tensor(bit_rows, dtype=torch.int32).to(torch.int16)
    return bits.view(torch.bfloat16).to(DEVICE)


def _random_blocks_bf16(m, k, gen):
    """BF16 [m, k]: Gaussian blocks of magnitude 2**-40 .. 2**40 (a third below the 1e-4
    floor)."""
    x = torch.randn((m, k // X_BLOCK, X_BLOCK), generator=gen, device=DEVICE)
    r = torch.randint(-40, 41, (m, k // X_BLOCK, 1), generator=gen, device=DEVICE)
    return (x * torch.exp2(r.float())).view(m, k).to(torch.bfloat16)


# (rows, K, lanes per row of the quantization): up to 128 rows with K % 512 == 0 one CTA per row
# over K / 8 lanes (one load each), more rows one wave per row (64 lanes, K / 512 loads each),
# K % 512 != 0 two rows per wave (32 lanes).
QUANT_1X32_CONFIGS = [(100, 2560, 320), (130, 2560, 64), (130, 768, 32)]


@pytest.mark.parametrize(
    "m, k, lanes",
    QUANT_1X32_CONFIGS,
    ids=[f"m{m}_k{k}" for m, k, _ in QUANT_1X32_CONFIGS],
)
def test_1x32_quant_bitexact(m, k, lanes):
    """The 1x32 quantization equals DeepSeek-V4.1's rule mapped to FNUZ (quant_1x32_ref) bit
    for bit in each configuration of the kernel: the blocks of QUANT_1X32_BLOCKS (all zero,
    +-0, amax with the mantissa 1.75 and one step above it, around the 1e-4 floor and far
    below it, subnormals, BF16's largest values, rounding ties, NaN and +-Inf), each with its
    first value in each of the four lanes that share a block; random blocks of magnitude
    2**-40 .. 2**40; and random bit patterns (any NaN, Inf or subnormal). The scale bytes of
    the named blocks also equal the ones derived by hand.

    End-to-end grading cannot see a few flipped FP8 codes or a scale one step off with codes
    to match: they move the GEMM output by less than its tolerance.
    """
    assert a8w4_module.quant_config(m, k).lanes == lanes  # the configuration under test
    per_row = k // X_BLOCK
    blocks, first = [], {}
    for name, (bits, _) in QUANT_1X32_BLOCKS.items():
        first[name] = len(blocks)
        for lane in range(4):
            # The block's first value lands in lane `lane` of the four (8 values each).
            shift = X_BLOCK - 9 * lane
            blocks.append(bits[shift:] + bits[:shift])
    special_rows = -(-len(blocks) // per_row)
    blocks += [[0x3F80] * X_BLOCK] * (special_rows * per_row - len(blocks))
    gen = _generator(case_seed("quant_1x32", m, k))
    x = _random_blocks_bf16(m, k, gen)
    x[:special_rows] = _bf16_rows(blocks).view(special_rows, k)
    bits = torch.randint(-(1 << 15), 1 << 15, (m // 3, k), generator=gen, device=DEVICE)
    x[m - m // 3 :] = bits.to(torch.int16).view(torch.bfloat16)

    q, scale = a8w4_module._quant_1x32(x)
    ref_q, ref_scale = quant_1x32_ref(x)
    assert torch.equal(scale, ref_scale)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    flat = scale.view(-1)
    for name, (_, byte) in QUANT_1X32_BLOCKS.items():
        got = flat[first[name] : first[name] + 4].tolist()
        assert got == [byte] * 4, (name, got)


def test_1x32_quant_vs_aiter():
    """Against aiter's ``dynamic_mxfp8_quant(x, quant_dtype=torch.float8_e4m3fnuz)`` (the
    MXFP8 activation of gfx950's aiter GEMM), the 1x32 quantization differs only in the blocks
    quant_1x32.py documents, and in each kind of them: amax with the mantissa 1.75 exactly
    (aiter rounds the scale up), amax < 1e-4 (aiter has no floor) and amax >= BF16 0x7F61
    (aiter saturates the scale byte). Finite inputs: with NaN aiter's maximum depends on its
    reduction order.

    The basic cases and the benchmark do not compare with aiter's quantization.
    """
    from aiter.ops.triton.quant import dynamic_mxfp8_quant

    m, k = 512, 1536
    per_row = k // X_BLOCK
    x = _random_blocks_bf16(m, k, _generator(case_seed("quant_1x32_aiter")))
    finite = [
        bits
        for name, (bits, _) in QUANT_1X32_BLOCKS.items()
        if "nan" not in name and "inf" not in name
    ]
    x.view(m, per_row, X_BLOCK)[0, : len(finite)] = _bf16_rows(finite)
    q, scale = a8w4_module._quant_1x32(x)
    aiter_q, aiter_scale = dynamic_mxfp8_quant(x, quant_dtype=torch.float8_e4m3fnuz)
    differs = (q.view(torch.uint8) != aiter_q.view(torch.uint8)).view(m, per_row, -1)
    differs = differs.any(dim=-1) | (scale != aiter_scale)
    amax16 = (x.view(torch.int16).int() & 0x7FFF).view(m, per_row, -1).amax(dim=-1)
    amax = (amax16 << 16).view(torch.float32)
    kinds = {
        "mantissa 1.75": ((amax16 & 0x7F) == 0x60) & (amax >= X_AMAX_FLOOR),
        "amax < 1e-4": amax < X_AMAX_FLOOR,
        "amax >= 0x7F61": amax16 >= 0x7F61,
    }
    documented = functools.reduce(torch.logical_or, kinds.values())
    assert not bool((differs & ~documented).any()), (differs & ~documented).nonzero()
    for name, mask in kinds.items():
        assert bool((differs & mask).any()), name


# (M, mapping of the small-M 1x32 kernel) on tp4_qkvz (N >= K for the one-hot weight).
SMALL_M_1X32_QUANT = [(4, "q4"), (13, "h2")]


@pytest.mark.parametrize(
    "m, mode",
    SMALL_M_1X32_QUANT,
    ids=[f"m{m}_{mode}" for m, mode in SMALL_M_1X32_QUANT],
)
def test_small_m_1x32_quant_bitexact(m, mode):
    """The small-M 1x32 kernel quantizes BF16 X inside each CTA (one lane per 32-K block)
    exactly like quant_1x32_ref. With a one-hot weight (channel c reads K position c % K with
    weight 1.0; N >= K) the output is Y[t, c] = bf16(xq[t, c % K] x 2**(e - 127)) of its
    block bit for bit, which exposes every FP8 code and scale. Rows: the finite blocks of
    QUANT_1X32_BLOCKS (all zero, +-0, amax with the mantissa 1.75 and one step above it,
    around the 1e-4 floor and far below it, subnormals, BF16's largest values, rounding ties),
    each with its first value at four positions; random bit patterns below 2 in magnitude
    (zeros, subnormals and values under the floor included); Gaussian blocks of magnitude
    2**-40 .. 2**40. M = 4 fills the q4 mapping's 4 token columns; M = 13 runs h2 with two
    token sets, the named blocks in both. Outputs of blocks with a scale byte above 226 are
    not compared: the f32 accumulator overflows there (BF16 values from about 2**107;
    documented in the kernel).

    Grading against the reference cannot see a few flipped FP8 codes or a scale one step
    off with codes to match; test_1x32_quant_bitexact covers the separate quantization
    kernel. NaN and +-Inf would turn every output of their MFMA NaN here (0 x NaN):
    test_small_m_nonfinite_rows.
    """
    n, k = SHAPES["tp4_qkvz"]
    per_row = k // X_BLOCK
    blocks = []
    for name, (bits, _) in QUANT_1X32_BLOCKS.items():
        if "nan" not in name and "inf" not in name:
            for lane in range(4):
                shift = X_BLOCK - 9 * lane
                blocks.append(bits[shift:] + bits[:shift])
    special_rows = -(-len(blocks) // per_row)
    blocks += [[0x3F80] * X_BLOCK] * (special_rows * per_row - len(blocks))
    special = _bf16_rows(blocks).view(special_rows, k)
    gen = _generator(case_seed("small_quant_1x32", m))
    x = _random_blocks_bf16(m, k, gen)
    x[:special_rows] = special
    if m > 8:
        x[8 : 8 + special_rows] = special  # the second token set
    bits = torch.randint(-(1 << 15), 1 << 15, (k,), generator=gen, device=DEVICE)
    x[special_rows] = (bits & ~0x4000).to(torch.int16).view(torch.bfloat16)  # |x| < 2
    columns = torch.arange(n, device=DEVICE) % k
    nibbles = torch.zeros((n, k), dtype=torch.uint8, device=DEVICE)
    nibbles[torch.arange(n, device=DEVICE), columns] = 2  # E2M1 1.0
    codes = torch.full((n, k // 32), 127, dtype=torch.uint8, device=DEVICE)  # 2**0
    weight = prepare_gemm_a8w4_ptpc_weight(pack_fp4(nibbles), codes)
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant="1x32")
    assert len(names) == 1 and gemm_1x32_path(names[0]) == mode, names
    out = flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")
    xq, x_scale = quant_1x32_ref(x)
    expected = dequant_1x32(xq, x_scale)[:, columns].to(torch.bfloat16)
    compared = x_scale[:, columns // X_BLOCK] <= 226
    assert float(compared.float().mean()) > 0.9
    got = out.view(torch.int16)[compared]
    assert torch.equal(got, expected.view(torch.int16)[compared])


# (M, N, K, mapping of the small-M 1x32 kernel or None for the 1x32 quantization and the 1x32
# GEMM), off the dispatch table: the per-token dispatch runs a small-M kernel at each of them.
SMALL_M_1X32_LIMITS = [
    (8, 3584, 5632, "h2"),  # one token set: its LDS just fits (65408 bytes)
    (9, 3584, 5632, None),  # two token sets do not fit
    # Per token two 16-token tiles; 1x32: three token sets do not fit.
    (17, 4160, 2560, None),
    # Four token sets (the 32-token limit), K-part sums reusing the token rows' LDS, 20 CTAs.
    (32, 320, 1536, "h2"),
]


@pytest.mark.parametrize(
    "m, n, k, mode",
    SMALL_M_1X32_LIMITS,
    ids=[f"m{m}_n{n}_k{k}" for m, n, k, _ in SMALL_M_1X32_LIMITS],
)
def test_small_m_1x32_limits(m, n, k, mode):
    """The small-M 1x32 kernel holds the FP8 rows of all of K for its token sets, their f32
    scales and the CTA's deltas in LDS: where the per-token dispatch runs its small-M kernel
    but these exceed 64 KiB, the 1x32 calls run the 1x32 quantization and the 1x32 GEMM
    instead, and a kernel just under the limit still runs (its LDS estimate is not below what
    it allocates). M = 32 on N = 320 runs four token sets, the K-part sums in the token rows'
    LDS behind one more barrier, on 20 CTAs (fewer than CUs). Both entries write through
    ``out=`` (the head of a NaN buffer), flydsl_gemm_a8w4_mxfp8 with the scales as uint8 and
    as torch.float8_e8m0fnu: the launched kernels are the predicted ones, every row matches
    the reference, nothing past ``out`` changes and the three outputs are equal bit for bit.

    The target shapes' 1x32 small-M kernels all fit (one or two token sets, tp4_o three);
    the benchmark and the basic cases let the op allocate the output.
    """
    assert len(gemm_a8w4_ptpc_kernel_names(m, n, k)) == 1  # per token: small-M kernel
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant="1x32")
    if mode is None:
        assert a8w4_module.small_m_1x32_config(m, n, k) is None
        assert len(names) == 2 and names[0].startswith("gemm_a8w4_ptpc_quant1x32_")
        assert gemm_1x32_path(names[1]) in ("mma16", "mma32"), names
        mxfp8_names = names[1:]
    else:
        assert len(names) == 1 and gemm_1x32_path(names[0]) == mode, names
        mxfp8_names = (names[0].removesuffix("_1x32") + "_mxfp8",)
    assert gemm_a8w4_mxfp8_kernel_names(m, n, k) == mxfp8_names
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x_1x32(m, k, case_seed("small_limits_1x32", m, n, k))
    xq, x_scale = quant_1x32_ref(x)
    ref = run_torch_1x32(xq, x_scale, w, w_scale)
    outs = []
    for call, predicted in [
        (functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32"), names),
        (functools.partial(flydsl_gemm_a8w4_mxfp8, xq, x_scale, weight), mxfp8_names),
        (
            functools.partial(
                flydsl_gemm_a8w4_mxfp8, xq, x_scale.view(torch.float8_e8m0fnu), weight
            ),
            mxfp8_names,
        ),
    ]:
        buffer = torch.full(
            (m + 32, n), float("nan"), dtype=torch.bfloat16, device=DEVICE
        )
        out = buffer[:m]
        call(out=out)  # compile outside the profiler
        out.fill_(float("nan"))
        got, kernels = launched_kernels(lambda call=call, out=out: call(out=out))
        assert got is out and kernels == list(predicted), kernels
        check_output(out, ref, msg=f"m{m}_n{n}_k{k}")
        assert bool(buffer[m:].isnan().all())
        outs.append(out.view(torch.int16))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


# (E8M0 bytes of X's blocks, E8M0 bytes of W's 32-K groups) for flydsl_gemm_a8w4_mxfp8.
EXTREME_1X32 = [((0, 1), (247, 250)), ((200, 226), (40, 45))]
# (M, path) of the extreme-scale cases on tp4_o: the small-M 1x32 kernel's pre-quantized variant
# (q4; h2 with three token sets: E8M0 bytes turned into f32 scales in LDS), the general GEMM on
# MFMA 16x16x32 (a 32 x 64 tile) and 32x32x16 (64 x 128, the scale of a 32-token tile per
# lane) and, past M = 2048, the persistent large-M kernel's 1x32 variant (block scales expanded
# to f32 in LDS).
EXTREME_1X32_M = [
    (4, "q4"),
    (20, "h2"),
    (37, "mma16"),
    (300, "mma32"),
    (2049, "large"),
]


@pytest.mark.parametrize(
    "m, path", EXTREME_1X32_M, ids=[f"m{m}" for m, _ in EXTREME_1X32_M]
)
@pytest.mark.parametrize(
    "x_codes, w_codes", EXTREME_1X32, ids=["x_e8m0_0_1", "x_e8m0_200_226"]
)
def test_1x32_extreme_scales(x_codes, w_codes, m, path):
    """The 1x32 GEMM scales every block exactly at both ends of the E8M0 range it takes, with
    random FP8 codes (all but NaN) against a float64 reference: X bytes 0 and 1 with large W
    scales (byte 0 is the f32 subnormal 2**-127, which the kernels keep; the byte-0 blocks
    make up a large part of the output, checked) and X bytes up to 226 (a block amax just
    under 2**107, the bound gemm.py documents for its f32 accumulator) with tiny W scales.
    M = 4 and 20 run the small-M 1x32 kernel (q4; h2 with three token sets), which turns the
    bytes into f32 scales once (byte 0 by a floor on the f32 bits); M = 37 and 300 the general
    GEMM's two MMA paths (each builds the scale from the E8M0 byte with its own byte permute
    and shift-add), M = 2049 the persistent large-M kernel's 1x32 variant; the path is
    asserted.

    The quantization of x_quant="1x32" writes bytes 106..248 only (254 for a block holding
    +-Inf), and the basic cases and the benchmark use moderate scales.
    """
    shape = "tp4_o"
    n, k = SHAPES[shape]
    names = gemm_a8w4_mxfp8_kernel_names(m, n, k)
    assert gemm_1x32_path(names[0]) == path, names
    gen = _generator(case_seed("extreme_1x32", x_codes))
    codes = torch.randint(0, 256, (m, k), generator=gen, device=DEVICE).to(torch.uint8)
    codes[codes == 0x80] = 0  # 0x80 is NaN in FNUZ
    xq = codes.view(torch.float8_e4m3fnuz)
    lo, hi = x_codes
    x_scale = torch.randint(lo, hi + 1, (m, k // X_BLOCK), generator=gen, device=DEVICE)
    x_scale = x_scale.to(torch.uint8)
    nibbles = torch.randint(0, 16, (n, k), generator=gen, device=DEVICE).to(torch.uint8)
    lo, hi = w_codes
    group_codes = torch.randint(lo, hi + 1, (n, k // 32), generator=gen, device=DEVICE)
    w, w_scale = pack_fp4(nibbles), group_codes.to(torch.uint8)
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    w_exact = dequant_mxfp4(w, w_scale).double()
    x_exact = dequant_1x32(xq, x_scale, torch.float64)
    ref = x_exact @ w_exact.T
    assert bool(torch.isfinite(ref).all())
    out = flydsl_gemm_a8w4_mxfp8(xq, x_scale, weight)
    check_output(out, ref.float(), msg=f"X E8M0 {x_codes}, W E8M0 {w_codes}")
    if x_codes[0] == 0:
        # Flushing the subnormal 2**-127 to zero would drop the byte-0 blocks.
        x_exact.view(m, -1, X_BLOCK)[x_scale == 0] = 0
        dropped = float((x_exact @ w_exact.T - ref).abs().amax())
        assert dropped > 0.05 * float(ref.abs().amax()), dropped


# (shape, first M of a 1x32 GEMM kernel). Up to its large-M range the 1x32 GEMM takes the tile
# of the shape's 1x32 dispatch table (_MID_TABLE_1X32) at M: these are the M where that tile
# changes inside the general GEMM's range, every general-GEMM kernel of the table from M = 33
# included (tp2_qkv's 32 x 128 tile with an XCD super tile of 1 at M = 65..112). The first one
# (M = 17, tp4_o 21) and the moves to and from the small-M and the large-M kernels are pinned
# in test_1x32_dispatch.
SWITCHES_1X32 = [
    ("tp2_qkvz", 33),
    ("tp2_qkvz", 65),
    ("tp2_qkvz", 129),
    ("tp2_qkvz", 193),
    ("tp2_qkv", 33),
    ("tp2_qkv", 65),
    ("tp2_qkv", 113),
    ("tp2_qkv", 129),
    ("tp2_qkv", 513),
    ("tp2_qkv", 577),
    ("tp2_o", 33),
    ("tp2_o", 65),
    ("tp2_o", 129),
    ("tp2_o", 1025),
    ("tp4_qkvz", 33),
    ("tp4_qkvz", 65),
    ("tp4_qkvz", 129),
    ("tp4_qkvz", 1025),
    ("tp4_qkv", 33),
    ("tp4_qkv", 65),
    ("tp4_qkv", 129),
    ("tp4_qkv", 257),
    ("tp4_qkv", 1025),
    ("tp4_o", 33),
    ("tp4_o", 65),
    ("tp4_o", 129),
    ("tp4_o", 257),
    ("tp4_o", 1025),
]


@pytest.mark.parametrize(
    "shape, m", SWITCHES_1X32, ids=[f"{s}_m{m}" for s, m in SWITCHES_1X32]
)
def test_1x32_tile_switches(shape, m):
    """x_quant="1x32" at each M where the 1x32 GEMM kernel changes inside the general GEMM's
    range on a target shape (its predicted kernel at M differs from the one at M - 1: the
    tile size, MMA path, pipeline depths, XCD super tile, K groups or K per stage): the
    launched kernels are the predicted ones and the output matches the reference, the new
    tile's last row block partly filled. With test_1x32_dispatch every general-GEMM kernel
    the six shapes use from M = 17 up to their large-M kernel runs once, and both
    configurations of the quantization (one CTA per row up to M = 128, one wave per row
    above).

    The basic cases run four M (3, 100, 200 and 2049) and the benchmark powers of two only.
    """
    n, k = SHAPES[shape]
    names = gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant="1x32")
    below = gemm_a8w4_ptpc_kernel_names(m - 1, n, k, x_quant="1x32")
    assert len(below) == len(names) == 2, (below, names)
    assert gemm_1x32_path(below[1]) in ("mma16", "mma32"), below
    assert gemm_1x32_path(names[1]) in ("mma16", "mma32"), names
    assert names[-1] != below[-1], (below, names)
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    x = make_x_1x32(m, k, case_seed("tile_switch_1x32", shape, m))
    call = functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32")
    call()  # compile outside the profiler
    out, kernels = launched_kernels(call)
    assert kernels == list(names), kernels
    xq, x_scale = quant_1x32_ref(x)
    check_output(out, run_torch_1x32(xq, x_scale, w, w_scale), msg=f"{shape}_m{m}")


# (shape, the last M of a 1x32 path, its path, the path from M + 1): every move of the 1x32
# dispatch on the target shapes. The small-M 1x32 kernel runs where the per-token dispatch takes
# its small-M kernel (M <= 16, tp4_o 20), with the q4 mapping up to M = 4 (every shape; one
# here) and h2 above; then the 1x32 quantization and the general GEMM, and the large-M kernel
# where _MID_TABLE_1X32 marks a bucket "large" (tp2_qkvz from 1025, tp2_o and tp4_qkv from
# 1921, tp4_qkvz 513..640) and past M = 2048 (tp2_qkv, tp4_qkvz, tp4_o).
DISPATCH_1X32 = [
    ("tp2_qkvz", 4, "q4", "h2"),
    ("tp2_qkvz", 16, "h2", "general"),
    ("tp2_qkv", 16, "h2", "general"),
    ("tp2_o", 16, "h2", "general"),
    ("tp4_qkvz", 16, "h2", "general"),
    ("tp4_qkv", 16, "h2", "general"),
    ("tp4_o", 20, "h2", "general"),
    ("tp2_qkvz", 1024, "general", "large"),
    ("tp2_o", 1920, "general", "large"),
    ("tp4_qkv", 1920, "general", "large"),
    ("tp4_qkvz", 512, "general", "large"),
    ("tp4_qkvz", 640, "large", "general"),
    ("tp2_qkv", 2048, "general", "large"),
    ("tp4_qkvz", 2048, "general", "large"),
    ("tp4_o", 2048, "general", "large"),
]


@pytest.mark.parametrize(
    "shape, m, below, above",
    DISPATCH_1X32,
    ids=[f"{s}_m{m}_{m + 1}" for s, m, _, _ in DISPATCH_1X32],
)
def test_1x32_dispatch(shape, m, below, above):
    """Each target shape's 1x32 calls switch paths at their own M: the small-M 1x32 kernel's
    q4 mapping to h2 after M = 4; the small-M 1x32 kernel to the 1x32 quantization and the
    general GEMM after M = 16 (tp4_o: 20); the general GEMM to the large-M kernel's 1x32
    variant after M = 1024 (tp2_qkvz), 1920 (tp2_o, tp4_qkv), 512 (tp4_qkvz, back to the
    general GEMM after 640) or 2048. On both sides the predicted kernels of both entries take
    the expected path, the launched ones are the predicted ones, the output matches the
    reference and the two entries agree bit for bit. The first M of a path leaves its last
    token set, row block or tile partly filled.

    The benchmark runs powers of two only (one side of each switch at most, never M = 5, 17,
    21, 513, 641, 1025, 1921 or 2049) and does not check which kernels ran; the basic cases
    run M = 3, 100, 200 and 2049.
    """
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    for rows, path in [(m, below), (m + 1, above)]:
        names = gemm_a8w4_ptpc_kernel_names(rows, n, k, x_quant="1x32")
        mxfp8_names = gemm_a8w4_mxfp8_kernel_names(rows, n, k)
        if path in ("q4", "h2"):
            assert len(names) == 1 and gemm_1x32_path(names[0]) == path, names
            assert mxfp8_names == (names[0].removesuffix("_1x32") + "_mxfp8",)
        else:
            assert names[0].startswith("gemm_a8w4_ptpc_quant1x32_"), names
            gemm_path = gemm_1x32_path(names[1])
            expected = ("large",) if path == "large" else ("mma16", "mma32")
            assert len(names) == 2 and gemm_path in expected, names
            assert mxfp8_names == names[1:], mxfp8_names
        x = make_x_1x32(rows, k, case_seed("dispatch_1x32", shape, rows))
        xq, x_scale = quant_1x32_ref(x)
        ref = run_torch_1x32(xq, x_scale, w, w_scale)
        ptpc = functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32")
        mxfp8 = functools.partial(flydsl_gemm_a8w4_mxfp8, xq, x_scale, weight)
        outs = []
        for call, predicted in [(ptpc, names), (mxfp8, mxfp8_names)]:
            call()  # compile outside the profiler
            out, kernels = launched_kernels(call)
            assert kernels == list(predicted), (rows, kernels)
            check_output(out, ref, msg=f"{shape}_m{rows}")
            outs.append(out.view(torch.int16))
        assert torch.equal(outs[0], outs[1]), rows


OFF_TABLE_1X32 = [(65, 320, 512), (1, 192, 256)]


@pytest.mark.parametrize(
    "m, n, k", OFF_TABLE_1X32, ids=[f"m{m}_n{n}_k{k}" for m, n, k in OFF_TABLE_1X32]
)
def test_1x32_off_table_ragged_n(m, n, k):
    """Off the dispatch table the 1x32 GEMM takes the default 64 x 128 tile (128 K per stage,
    MFMA 32x32x16) and stores directly: with N % 128 == 64 the last channel block is half
    empty, and its stores past N go to a unit past the output buffer, which drops them,
    instead of branching.
    M = 65 leaves 1 row in the last row block; K = 256 is two K tiles. Both entries write
    through ``out=`` (the head of a NaN buffer), flydsl_gemm_a8w4_mxfp8 with the scales as
    uint8 and as torch.float8_e8m0fnu: every row matches the reference, nothing past ``out``
    changes and the three outputs are equal bit for bit.

    The target shapes are on the table with N % 128 == 0, the basic cases pass uint8 scales
    and let the op allocate its output, and the benchmark runs the target shapes only.
    """
    from aiter.ops.flydsl.kernels.gemm_a8w4_ptpc import gemm as gemm_kernels

    config = a8w4_module._config_1x32(m, n, k)
    assert config == gemm_kernels.GemmConfig() and n % config.block_n != 0, config
    assert gemm_1x32_path(gemm_a8w4_mxfp8_kernel_names(m, n, k)[0]) == "mma32"
    assert not gemm_kernels._shuffled_store(config), config  # direct stores
    w, w_scale = make_weight(n, k, case_seed("w", n, k))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x_1x32(m, k, case_seed("off_table_1x32", m, n, k))
    xq, x_scale = quant_1x32_ref(x)
    ref = run_torch_1x32(xq, x_scale, w, w_scale)
    outs = []
    for call in [
        functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32"),
        functools.partial(flydsl_gemm_a8w4_mxfp8, xq, x_scale, weight),
        functools.partial(
            flydsl_gemm_a8w4_mxfp8, xq, x_scale.view(torch.float8_e8m0fnu), weight
        ),
    ]:
        buffer = torch.full(
            (m + 64, n), float("nan"), dtype=torch.bfloat16, device=DEVICE
        )
        out = buffer[:m]
        assert call(out=out) is out
        check_output(out, ref, msg=f"m{m}_n{n}_k{k}")
        assert bool(buffer[m:].isnan().all())
        outs.append(out.view(torch.int16))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_1x32_rejects_bad_inputs(monkeypatch):
    """Unsupported inputs raise instead of computing garbage: an ``x_quant`` other than
    "per_token" / "1x32" (entry and kernel_names) and a bad BF16 X with x_quant="1x32"; for
    flydsl_gemm_a8w4_mxfp8 a foreign weight, xq that is not FP8 E4M3FNUZ (BF16, uint8,
    E4M3FN), x_scale that is not uint8 / E8M0 (int32, f32), wrong ranks or shapes (xq's K,
    x_scale's K and M), CPU tensors, non-contiguous xq or x_scale, xq not 16-byte or x_scale
    not 4-byte aligned, a bad ``out`` (shape, dtype, layout, device), unsupported weight
    shapes in gemm_a8w4_mxfp8_kernel_names, and any GPU other than gfx942 (simulated).

    The basic cases and the benchmark only pass valid inputs on gfx942.
    """
    n, k, m = 128, 512, 4
    w, w_scale = make_weight(n, k, case_seed("errors"))
    weight = prepare_gemm_a8w4_ptpc_weight(w, w_scale)
    x = make_x(m, k, case_seed("errors_x"))
    xq, x_scale = quant_1x32_ref(x)
    fnuz = torch.float8_e4m3fnuz

    for x_quant in ["1x16", "per_tensor", None]:
        with pytest.raises(ValueError, match="x_quant"):
            flydsl_gemm_a8w4_ptpc(x, weight, x_quant=x_quant)
        with pytest.raises(ValueError, match="x_quant"):
            gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant=x_quant)
    with pytest.raises(TypeError):
        flydsl_gemm_a8w4_ptpc(x.half(), weight, x_quant="1x32")
    with pytest.raises(ValueError):
        flydsl_gemm_a8w4_ptpc(torch.cat([x, x], dim=1)[:, :k], weight, x_quant="1x32")

    # xq 8 bytes past a 16-byte boundary, x_scale 2 bytes past a 4-byte one; strided rows.
    xq_buf = torch.empty(m * k + 16, dtype=torch.uint8, device=DEVICE)
    xq_off = xq_buf[8 : 8 + m * k].view(m, k).view(fnuz)
    s_buf = torch.empty(m * k // X_BLOCK + 4, dtype=torch.uint8, device=DEVICE)
    s_off = s_buf[2 : 2 + m * k // X_BLOCK].view(m, k // X_BLOCK)
    xq_strided = torch.cat([xq.view(torch.uint8)] * 2, dim=1)[:, :k].view(fnuz)
    s_strided = torch.cat([x_scale, x_scale], dim=1)[:, : k // X_BLOCK]
    half_k = (xq[:, : k // 2].contiguous(), x_scale[:, : k // 64].contiguous())
    good = (xq, x_scale, weight)
    for args, kwargs, error in [
        ((xq, x_scale, (w, w_scale)), {}, TypeError),
        ((x, x_scale, weight), {}, TypeError),
        ((xq.view(torch.uint8), x_scale, weight), {}, TypeError),
        ((xq.view(torch.float8_e4m3fn), x_scale, weight), {}, TypeError),
        ((xq, x_scale.int(), weight), {}, TypeError),
        ((xq, x_scale.float(), weight), {}, TypeError),
        ((xq.view(-1), x_scale, weight), {}, ValueError),
        ((*half_k, weight), {}, ValueError),
        ((xq, x_scale[:, :-1].contiguous(), weight), {}, ValueError),
        ((xq, x_scale[:-1].contiguous(), weight), {}, ValueError),
        ((xq, x_scale.view(-1), weight), {}, ValueError),
        ((xq.cpu(), x_scale, weight), {}, ValueError),
        ((xq, x_scale.cpu(), weight), {}, ValueError),
        ((xq_strided, x_scale, weight), {}, ValueError),
        ((xq, s_strided, weight), {}, ValueError),
        ((xq_off, x_scale, weight), {}, ValueError),
        ((xq, s_off, weight), {}, ValueError),
        (good, {"out": _empty((m, n + 64), torch.bfloat16)}, ValueError),
        (good, {"out": _empty((m, n), torch.float16)}, ValueError),
        (good, {"out": _empty((n, m), torch.bfloat16).t()}, ValueError),
        (good, {"out": _empty((m, n), torch.bfloat16).cpu()}, ValueError),
    ]:
        with pytest.raises(error):
            flydsl_gemm_a8w4_mxfp8(*args, **kwargs)
    for n_, k_ in [(96, 2560), (2560, 128), (0, 256)]:
        with pytest.raises(ValueError):
            gemm_a8w4_mxfp8_kernel_names(1, n_, k_)

    monkeypatch.setattr(a8w4_module, "get_gfx_runtime", lambda: "gfx950")
    with pytest.raises(RuntimeError, match="gfx942"):
        flydsl_gemm_a8w4_mxfp8(xq, x_scale, weight)
    with pytest.raises(RuntimeError, match="gfx942"):
        flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")


def test_1x32_m_zero():
    """M = 0: both 1x32 entries return an empty BF16 [0, N] tensor (or the given ``out``) and
    launch nothing; both kernel_names functions return ().

    The basic cases and the benchmark have M >= 1.
    """
    n, k = SHAPES["tp4_o"]
    _, _, weight = cached_weight(n, k, case_seed("w", "tp4_o"))
    x = torch.empty((0, k), dtype=torch.bfloat16, device=DEVICE)
    xq = torch.empty((0, k), dtype=torch.float8_e4m3fnuz, device=DEVICE)
    x_scale = torch.empty((0, k // X_BLOCK), dtype=torch.uint8, device=DEVICE)
    for call in [
        functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant="1x32"),
        functools.partial(flydsl_gemm_a8w4_mxfp8, xq, x_scale, weight),
    ]:
        y, kernels = launched_kernels(call)
        assert tuple(y.shape) == (0, n) and y.dtype == torch.bfloat16, y
        assert y.device == weight.q.device and kernels == [], kernels
        out = torch.empty((0, n), dtype=torch.bfloat16, device=DEVICE)
        assert call(out=out) is out
    assert gemm_a8w4_ptpc_kernel_names(0, n, k, x_quant="1x32") == ()
    assert gemm_a8w4_mxfp8_kernel_names(0, n, k) == ()


# (shape, M, path) of test_1x32_cuda_graph_replay: the small-M 1x32 kernel (h2, two token
# sets), the general GEMM with two K groups on MFMA 16x16x32 (32 x 64, 256 K per stage) and
# 32x32x16 (32 x 128, 128 K per stage), and the persistent large-M kernel's 1x32 variant.
GRAPH_1X32 = [
    ("tp2_qkvz", 12, "h2"),
    ("tp2_o", 48, "mma16"),
    ("tp2_qkvz", 20, "mma32"),
    ("tp4_o", 2049, "large"),
]


@pytest.mark.parametrize(
    "shape, m, path", GRAPH_1X32, ids=[f"{s}_m{m}" for s, m, _ in GRAPH_1X32]
)
def test_1x32_cuda_graph_replay(shape, m, path):
    """Both 1x32 entries (x_quant="1x32" allocating its FP8 and E8M0 buffers and its output
    inside the capture, and flydsl_gemm_a8w4_mxfp8) are captured in one CUDA graph and replay
    on new input: no host synchronization. Both replayed outputs equal an eager call on the
    new input bit for bit and match the reference. M = 12 runs the small-M 1x32 kernel (one
    launch per entry); the general GEMM runs two K groups (their sums added through LDS
    before the epilogue) on each MMA path; M = 2049 the persistent large-M kernel's 1x32
    variant (its tile order chosen on the host at capture).

    The basic cases and the benchmark call the op eagerly.
    """
    n, k = SHAPES[shape]
    w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
    names = gemm_a8w4_mxfp8_kernel_names(m, n, k)
    assert gemm_1x32_path(names[0]) == path, names
    if path in ("mma16", "mma32"):
        assert a8w4_module._config_1x32(m, n, k).k_groups == 2
    x = make_x_1x32(m, k, case_seed("graph_1x32", shape, m, 0))
    xq, x_scale = quant_1x32_ref(x)
    flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")  # compile outside the capture
    flydsl_gemm_a8w4_mxfp8(xq, x_scale, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32")
        y_mxfp8 = flydsl_gemm_a8w4_mxfp8(xq, x_scale, weight)
    x_new = make_x_1x32(m, k, case_seed("graph_1x32", shape, m, 1))
    xq_new, x_scale_new = quant_1x32_ref(x_new)
    x.copy_(x_new)
    xq.copy_(xq_new)
    x_scale.copy_(x_scale_new)
    y.fill_(float("nan"))
    y_mxfp8.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    check_output(y, run_torch_1x32(xq_new, x_scale_new, w, w_scale), msg="graph replay")
    eager = flydsl_gemm_a8w4_ptpc(x_new, weight, x_quant="1x32").view(torch.int16)
    assert torch.equal(y.view(torch.int16), eager)
    assert torch.equal(y_mxfp8.view(torch.int16), eager)


# (shape, M) of test_1x32_compile_cache: the general GEMM (where the per-token and 1x32 tables
# give the same tile, the two variants differ only in x_block_scale) and, past M = 2048, the
# persistent large-M kernel (in both modes; its two variants differ only in x_block_scale).
COMPILE_CACHE_1X32 = [("tp2_qkvz", 100), ("tp4_o", 2049)]


def test_1x32_compile_cache():
    """Per-token and 1x32 calls alternate on two shapes and M (each mode compiled, then
    called again): every call launches the kernels kernel_names predicts for its mode and
    matches its own reference, so the two compile-time variants (the 1x32 one's name ends in
    _1x32) are compiled and cached apart. Only M > 2048 asserts that both modes run the same
    kernel (the large-M kernel); the general GEMM's tiles come from the dispatch tables.

    The benchmark times every per-token case before the 1x32 ones, so it never returns to a
    per-token kernel after its 1x32 variant is compiled; every other test runs one mode per
    call site.
    """
    for shape, m in COMPILE_CACHE_1X32:
        n, k = SHAPES[shape]
        w, w_scale, weight = cached_weight(n, k, case_seed("w", shape))
        names = {
            x_quant: gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant=x_quant)
            for x_quant in ["per_token", "1x32"]
        }
        if m > 2048:
            assert names["1x32"][1] == names["per_token"][1] + "_1x32", names
        x = make_x_1x32(m, k, case_seed("cache_1x32", shape, m))
        xq, x_scale = quant_1x32_ref(x)
        refs = {
            "per_token": run_torch(x, w, w_scale),
            "1x32": run_torch_1x32(xq, x_scale, w, w_scale),
        }
        for x_quant in ["per_token", "1x32"]:  # compile outside the profiler
            out = flydsl_gemm_a8w4_ptpc(x, weight, x_quant=x_quant)
            check_output(out, refs[x_quant], msg=f"{shape}_m{m}_{x_quant}")
        for x_quant in ["per_token", "1x32"]:
            call = functools.partial(flydsl_gemm_a8w4_ptpc, x, weight, x_quant=x_quant)
            out, kernels = launched_kernels(call)
            assert kernels == list(names[x_quant]), (x_quant, kernels)
            check_output(out, refs[x_quant], msg=f"{shape}_m{m}_{x_quant} again")


# =============================================================================================
# Region 3: benchmark (script entry, not collected by pytest)
# Each case times the full call from BF16 X to BF16 Y (per-token quantization included) for
# this op and for aiter's FP8 per-token GEMM, as SGLang's gfx942 apply_fp8_ptpc_linear runs it:
# per_token_quant_hip + gemm_a8w8_bpreshuffle on the same MXFP4 weight dequantized (flush
# rule included) and quantized per channel to FP8. Both use run_perftest (inputs deep-copied
# and rotated, >= ~12 ms of warm-up calls, all kernels of a call summed) and grade the timed
# output with checkAllclose against an fp32 reference of their own quantized operands.
# After all of them, the 1x32 cases (``<shape>_m<M>_1x32``, the same shapes and M) time this
# op's full call with x_quant="1x32" (1x32 quantization included) on the per-token case's X
# and weight the same way, graded against the fp32 reference on the 1x32 quantization of X.
# Their MF_REF is aiter's time from the per-token case of the same (shape, M), not timed again
# (that would double the bench's time); running them last leaves the per-token cases timed as
# before.
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


# -- x_quant="1x32" cases ----------------------------------------------------------------------
# The bench keeps its own 1x32 reference: it uses none of the test regions' 1x32 helpers, so
# those can change without changing what the bench runs.


def _bench_quant_1x32(x):
    """1x32 FP8 E4M3FNUZ [m, k] and E8M0 bytes [m, k / 32] of finite BF16 ``x`` as
    x_quant="1x32" quantizes it: DeepSeek-V4.1's act_quant(x, block_size=32) with UE8M0
    scales, mapped to FNUZ (quant_1x32_ref of the tests without its NaN / Inf rules).

    e = ceil(log2(max(amax, 1e-4) * f32(1 / 448))) from the f32 bits, q = RNE_e4m3fn(x / 2**e);
    an FNUZ code is worth half its E4M3FN value, so the byte is e + 128 with the same codes,
    -0 (0x80, FNUZ's NaN) stored as 0x00.
    """
    m, k = x.shape
    xf = x.float().view(m, k // 32, 32)
    amax = xf.abs().amax(dim=-1)
    amax = torch.maximum(amax, torch.full_like(amax, 1e-4))
    inv_max = torch.ones_like(amax) / torch.full_like(amax, 448.0)
    bits = (amax * inv_max).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale_fn = ((e + 127) << 23).view(torch.float32).unsqueeze(-1)
    q = (xf / scale_fn).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    q = torch.where(q == 0x80, torch.zeros_like(q), q)
    return q.view(m, k).view(torch.float8_e4m3fnuz), (e + 128).to(torch.uint8)


def _bench_ref_1x32(x, w_exact):
    """fp32 reference of flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32"), ``w_exact`` the
    exact weight (prepare flush rule included): the sum over 32-K blocks of 2**(e - 127) x
    sum(q x w) on the 1x32 quantization of ``x``; each q x 2**(e - 127) is exact in f32.
    """
    xq, e = _bench_quant_1x32(x)
    m, k = xq.shape
    scale = torch.exp2(e.double() - 127).float().unsqueeze(-1)
    x_exact = (xq.float().view(m, k // 32, 32) * scale).view(m, k)
    return fp32_matmul(x_exact, w_exact)


@benchmark()
def bench_gemm_a8w4_1x32(shape, m, per_token_us, aiter_us):
    """Case ``<shape>_m<M>_1x32``: flydsl_gemm_a8w4_ptpc(x, weight, x_quant="1x32") on the X
    and weight of the per-token case ``<shape>_m<M>``, timed and graded as that case does the
    per-token call. ``per_token_us``, ``aiter_us``: that case's times of this op and of aiter's
    FP8 per-token GEMM; aiter is not timed again, so MF_REF repeats its time."""
    n, k = SHAPES[shape]
    case = f"{shape}_m{m}_1x32"
    weight, w_exact = _bench_weight(shape)[:2]
    x = make_x(m, k, case_seed("bench_x", shape, m))
    fn = functools.partial(flydsl_gemm_a8w4_ptpc, x_quant="1x32")
    args = (x, weight)
    out, us = run_perftest(fn, *args, num_warmup=_warmup_calls(fn, args))
    err = check_output(out, _bench_ref_1x32(x, w_exact), msg=f"{case} a8w4")
    flops = 2 * m * n * k
    # The per-token case's bytes (BF16 X and the prepared weight read, BF16 Y written): the
    # 1x32 FP8 X and its M * K / 32 E8M0 scales are not counted, as the per-token FP8 X and
    # scales are not: the call writes and reads them back itself.
    nbytes = m * k * 2 + n * k // 2 + n * k // 32 + 4 * n + m * n * 2
    aiter_bytes = m * k * 2 + n * k + 4 * n + m * n * 2
    print(
        f"MF_METRIC case={case} value={us:.3f} unit=us flops={flops} bytes={nbytes}",
        flush=True,
    )
    print(
        f"MF_REF case={case} impl=aiter_fp8_ptpc value={aiter_us:.3f} unit=us "
        f"flops={flops} bytes={aiter_bytes}",
        flush=True,
    )
    return {
        "gfx": get_gfx_runtime(),
        "N": n,
        "K": k,
        "a8w4_1x32 us": us,
        "a8w4_1x32 TFLOPS": flops / us / 1e6,
        "a8w4_1x32 TB/s": nbytes / us / 1e6,
        "a8w4_1x32 err": err,
        "1x32 / per_token": us / per_token_us,
        "1x32 / aiter": us / aiter_us,
        "a8w4_1x32 kernel": gemm_a8w4_ptpc_kernel_names(m, n, k, x_quant="1x32")[-1],
    }


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a8w4_ptpc is unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the FlyDSL A8W4 GEMM (per-token, then 1x32 activations) "
        "against aiter's FP8 per-token GEMM.",
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
    # The 1x32 cases after every per-token case, in the same order, each given the per-token
    # case's times of this op and aiter.
    rows_1x32 = [
        bench_gemm_a8w4_1x32(
            row["shape"], row["m"], row["a8w4 us"], row["aiter_fp8_ptpc us"]
        )
        for row in rows
    ]
    aiter.logger.info(
        "gemm_a8w4_ptpc x_quant=1x32 summary (markdown):\n%s",
        pd.DataFrame(rows_1x32).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
