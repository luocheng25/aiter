# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL paged decode correctness and tuning-policy tests.

Run all tests: python -m pytest -q op_tests/test_flydsl_pa_decode.py
CPU policy/cache tests: python -m pytest -q op_tests/test_flydsl_pa_decode.py::TestPAConfig
"""

import ast
import importlib.util
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx942", "gfx950", "gfx1250"):
        pytest.skip(f"pa_decode is unsupported on {get_gfx_runtime()}")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


@pytest.mark.parametrize(
    "batch,max_parts,budget,window,query_length",
    [
        (1, 1, 1, 0, 1),
        (6, 7, 160, 257, 4),
        (63, 7, 160, 0, 1),
        (65, 7, 512, 1, 4),
        (257, None, 4096, 8193, 4),
        (4096, None, 8192, 0, 1),
    ],
)
def test_pa_decode_plan(device, batch, max_parts, budget, window, query_length):
    """Packed tasks cover every visible tile and remain valid after graph refresh."""
    from aiter.ops.flydsl.kernels.pa_decode_plan import plan_pa_decode

    def check(plan, lengths):
        info = plan.reduce_info.cpu().tolist()
        work = plan.work_info.cpu().tolist()
        cursor = 0
        for seq, (ctx, (start, count)) in enumerate(zip(lengths, info)):
            assert start == cursor
            first = (
                max(max(ctx, 0) - query_length + 1 - window, 0) // 256 if window else 0
            )
            last = (max(ctx, 0) + 255) // 256
            assert 0 <= count <= min(last - first, plan.max_partitions)
            assert (count > 0) == (last > first)
            tile = first
            for record in work[start : start + count]:
                row, begin, end, original_length = record
                assert row == seq and original_length == ctx
                assert begin == tile and begin < end <= last
                tile = end
            assert tile == last
            cursor += count
        assert cursor <= plan.capacity
        assert all(record == [0, 0, 0, 0] for record in work[cursor:])

    pattern = (-1, 0, 1, 255, 256, 257, 65537, 2**31 - 1)
    lengths = [pattern[i % len(pattern)] for i in range(batch)]
    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    plan = plan_pa_decode(
        context,
        2,
        max_partitions=max_parts,
        workgroup_budget=budget,
        sliding_window=window,
        query_length=query_length,
    )
    check(plan, lengths)
    pointers = (plan.work_info.data_ptr(), plan.reduce_info.data_ptr())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        refreshed = plan_pa_decode(
            context, 2, sliding_window=window, query_length=query_length, plan=plan
        )
    assert refreshed is plan
    # All-empty replay must erase old active records, then mixed lengths must
    # rebuild them using the same buffers without stale padding or readback.
    for values in ([0] * batch, list(reversed(lengths))):
        context.copy_(torch.tensor(values, dtype=torch.int32, device=device))
        graph.replay()
        check(plan, values)
        assert (plan.work_info.data_ptr(), plan.reduce_info.data_ptr()) == pointers


def _reference(query, key, value, table, lengths, query_length, window, sinks):
    """FP32 attention over logical KV, independent of kernel Q/P rounding."""
    batch = lengths.numel()
    heads, dim = query.shape[1:]
    kv_heads, page_size = key.shape[1:3]
    group = heads // kv_heads
    queries = query.float().reshape(batch, query_length, kv_heads, group, dim)
    value_dim = value.shape[-1]
    output = torch.zeros(
        (*queries.shape[:-1], value_dim), dtype=torch.float32, device=query.device
    )
    positions = torch.arange(query_length, device=query.device)

    for seq, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        tokens = torch.arange(length, device=query.device)
        pages = table[seq, tokens // page_size].long()
        offsets = tokens % page_size
        keys, values = key[pages, :, offsets], value[pages, :, offsets]
        scores = torch.einsum("qhgd,khd->qhgk", queries[seq], keys) * dim**-0.5
        visible = length - query_length + 1 + positions
        masked = tokens[None, :] >= visible[:, None]
        if window > 0:
            masked |= tokens[None, :] < (visible - window)[:, None]
        scores.masked_fill_(masked[:, None, None, :], float("-inf"))
        denominator = torch.logsumexp(scores, dim=-1, keepdim=True)
        if sinks is not None:
            denominator = torch.logaddexp(
                denominator, sinks.float().reshape(1, kv_heads, group, 1)
            )
        probs = torch.exp(scores - denominator)
        # Empty query rows have no attention mass, including with -inf sinks.
        probs.masked_fill_(visible[:, None, None, None] <= 0, 0)
        output[seq] = torch.einsum("qhgk,khd->qhgd", probs, values)
    return output.reshape(batch * query_length, heads, value_dim)


def _make_inputs(
    query_length,
    kv_heads,
    group_size,
    head_dim,
    value_dim,
    page_size,
    dtype,
    pattern,
    device,
    lengths,
    table_width=None,
):
    if lengths is None:
        lengths = (0, 1, 255, 256, 257, 1027) if pattern == "random" else (1024,)
    batch = len(lengths)
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    num_pages = batch * pages_per_seq
    table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        batch, pages_per_seq
    )
    if table_width is not None:
        # Block-table padding beyond the live pages must not reach the kernel.
        table = torch.nn.functional.pad(table, (0, table_width - pages_per_seq))
    query = torch.empty(
        (batch * query_length, kv_heads * group_size, head_dim),
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        (num_pages, kv_heads, page_size, head_dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        (num_pages, kv_heads, page_size, value_dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)

    if pattern == "constant":
        query.zero_()
        key.zero_()
        value.fill_(0.5)
    elif pattern == "small_query":
        # Tiny Q values must survive FP8 normalization or native BF16 loads.
        query.fill_(2**-12)
        key.fill_(14)
        key[:, :, 1::2] = -14
        value.fill_(3.5)
        value[:, :, 1::2] = 0
    elif pattern == "probability_tail":
        # Individually tiny probabilities carry substantial mass together.
        # Keep the maximum first and use one partition to retain all 1023 tails.
        # These Q/K values give the same scores in E4M3FN and E4M3FNUZ,
        # with tail probabilities near 2**-12. Scaling P preserves the tails
        # in both formats; converting unscaled P to FP8 rounds them to zero.
        query.zero_()
        query[..., 0] = 13.4375
        key.zero_()
        key[..., 0] = -7
        key[..., 2] = -14
        value.fill_(3.5)
        first_page = table[0, 0].item()
        key[first_page, :, 0] = 0
        value[first_page, :, 0] = 0
        # Give both quantization modes the same V absmax; the other channels
        # still isolate the probability-tail mass.
        value[first_page, :, 0, 0] = 3.5
    elif pattern == "varied":
        # Uniform draws quantize to nearly one per-token scale, so a shifted
        # scale still matches the oracle. Token magnitudes spanning about a
        # decade make a wrong scale index fail it.
        key.mul_(torch.randn_like(key[..., :1]).mul_(0.4).exp_())
        value.mul_(torch.randn_like(value[..., :1]).mul_(0.4).exp_())

    context = torch.tensor(lengths, dtype=torch.int32, device=device)
    return query, key, value, table, context


_COMMON_CASES = [
    ("decode", {"block_size": 16, "max_partitions": 1}),
    (
        "window",
        {
            "query_length": 4,
            "group_size": 16,
            "block_size": 128,
            "max_partitions": 7,
            "window": 257,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "head1024",
        {
            "group_size": 4,
            "head_dim": 1024,
            "block_size": 128,
            "sink_dtype": torch.bfloat16,
        },
    ),
    *[
        (
            f"{pattern}-page{page}-g{group}",
            {"group_size": group, "block_size": page, "pattern": pattern},
        )
        for page, group in ((16, 8), (128, 16))
        for pattern in ("small_query", "probability_tail")
    ],
]
_ASYMMETRIC_BASE = {
    "query_length": 8,
    "group_size": 16,
    "head_dim": 192,
    "value_dim": 128,
    "block_size": 64,
    "max_partitions": 8,
    "sink_dtype": torch.float32,
    "capture": True,
    # Empty rows, short-context residuals, and page/tile boundaries.
    "lengths": (0, 1, 7, 8, 16, 17, 32, 33, 63, 64, 65, 255, 256, 257, 1027),
}
_ASYMMETRIC_WINDOW_CASES = [
    (
        f"asymmetric-window-{window}",
        dict(
            _ASYMMETRIC_BASE,
            window=window,
            sink_dtype=None if window in (0, 2) else torch.float32,
            capture=window in (0, 2, 33, 1025),
            lengths=(*_ASYMMETRIC_BASE["lengths"], 8192),
        ),
    )
    # Full attention, minimum windows, FP8 residual cutoff, and wider masks.
    for window in (0, 1, 2, 32, 33, 128, 1025, 2**31 + 1)
]
_ASYMMETRIC_MULTIHEAD_CASES = [
    (
        f"asymmetric-kv{heads}-gqa{group}-w{window}",
        dict(
            _ASYMMETRIC_BASE,
            kv_heads=heads,
            group_size=group,
            window=window,
            pattern="varied",
            max_partitions=4,
        ),
    )
    # Per-token scales need both residual and ordinary-window paths.
    for heads, group, window in ((2, 16, 4), (4, 8, 128))
]
_ASYMMETRIC_GENERAL_CASES = [
    (
        name,
        dict(
            _ASYMMETRIC_BASE,
            kv_heads=2,
            pattern="varied",
            max_partitions=4,
            query_length=qlen,
            group_size=group,
            block_size=page,
            window=window,
        ),
    )
    for name, qlen, group, page, window in (
        ("asymmetric-qlen1-page16", 1, 8, 16, 0),
        ("asymmetric-qlen3-page128", 3, 8, 128, 33),
        ("asymmetric-qlen12-gqa32", 12, 32, 64, 2048),
        ("asymmetric-qlen64", 64, 8, 64, 0),
    )
]
# The common FP8 cases cover the dtype/scale/layout product. Rotate those
# variants across the new widths instead of repeating the product for each.
_FP8_HEAD_CASES = [
    (
        f"head{dim}-mtp",
        {
            "head_dim": dim,
            "kv_dtype": "fp8",
            "query_dtype": dtype,
            "scale_mode": scale_mode,
            "trans_v": trans_v,
            "query_length": 3,
            "group_size": 8,
            "kv_heads": 2,
            "block_size": 64,
            "max_partitions": 4,
            "window": 128,
            "sink_dtype": torch.float32,
            "capture": dim in (192, 960),
        },
    )
    for dim, dtype, scale_mode, trans_v in (
        (192, torch.bfloat16, "per-tensor", True),
        (320, torch.bfloat16, "per-token", True),
        (448, torch.float16, "per-tensor", False),
        (576, torch.bfloat16, "per-tensor", False),
        (704, torch.float16, "per-token", True),
        (832, torch.bfloat16, "per-token", False),
        (960, torch.float16, "per-token", False),
    )
]
_FP8_CASES = _COMMON_CASES + [
    *_ASYMMETRIC_GENERAL_CASES,
    *_ASYMMETRIC_MULTIHEAD_CASES,
    *_ASYMMETRIC_WINDOW_CASES,
    (
        "asymmetric-large-window-grid",
        dict(
            _ASYMMETRIC_BASE,
            window=1024,
            lengths=(0, 1, 8, 16, 17, 257, 1027, 8192) * 16,
        ),
    ),
    (
        "asymmetric-full-long",
        dict(
            _ASYMMETRIC_BASE,
            window=0,
            sink_dtype=None,
            capture=False,
            lengths=(16384,) * 8,
        ),
    ),
    (
        "packed-mtp4",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 8,
            "head_dim": 256,
            "block_size": 128,
            "max_partitions": 7,
            "sink_dtype": torch.float32,
        },
    ),
    (
        "mtp",
        {
            "query_length": 3,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {
            "query_length": 2,
            "kv_heads": 2,
            "head_dim": 256,
            "block_size": 16,
            "max_partitions": 7,
        },
    ),
    (
        "default-cap",
        {"group_size": 16, "block_size": 128, "max_partitions": None},
    ),
]
_BF16_CASES = _COMMON_CASES + [
    *[
        (
            f"swa-general-q{qlen}-d{dim}-v{vdim}",
            {
                "query_length": qlen,
                "head_dim": dim,
                "value_dim": vdim,
                "group_size": group,
                "kv_heads": heads,
                "block_size": page,
                "window": window,
                "max_partitions": parts,
                "lengths": (0, 1, 7, 127, 257, 2047, 4097, 8192) * (batch // 8),
                "sink_dtype": torch.float32,
                "capture": True,
            },
        )
        for qlen, dim, vdim, group, heads, page, window, parts, batch in (
            (1, 64, 64, 8, 2, 16, 127, 1, 8),
            (2, 192, 192, 16, 1, 64, 512, 4, 8),
            (3, 128, 128, 16, 1, 64, 512, 8, 48),
            (4, 256, 256, 8, 2, 128, 1023, 4, 8),
            (5, 128, 128, 16, 1, 64, 4096, 1, 8),
            (6, 512, 512, 16, 1, 64, 2048, 1, 8),
            (7, 192, 128, 8, 2, 16, 1025, 4, 8),
            (8, 1024, 1024, 16, 1, 128, 2048, 4, 8),
        )
    ],
    *_ASYMMETRIC_GENERAL_CASES,
    *_ASYMMETRIC_MULTIHEAD_CASES,
    # BF16 has no FP8 residual cutoff; retain the shared mask/length coverage.
    *[
        case
        for case in _ASYMMETRIC_WINDOW_CASES
        if case[1]["window"] not in (2, 32, 33)
    ],
    *[
        (
            f"qlen8-d128-w{window}-np{parts}",
            {
                "query_length": 8,
                "group_size": 16,
                "head_dim": 128,
                "block_size": 64,
                "max_partitions": parts,
                "window": window,
                "sink_dtype": torch.float32,
                "capture": window in (1023, 2048),
                "lengths": (
                    (0, 1, 127, 2047, 2048, 2049, 4097, 8192)
                    if window == 2048
                    else (0, 1, 127, 128, 129, 257, 515, 1027)
                ),
            },
        )
        for window, parts in (
            (512, 4),
            (1023, 8),
            (1025, 8),
            (1024, 1),
            (1024, 4),
            (1023, 1),
            (1025, 1),
            (2048, 1),
            (2048, 4),
            (2048, 8),
            (2048, 16),
        )
    ],
    *[
        (
            f"qlen8-d128-w2048-b32-np{parts}",
            {
                "query_length": 8,
                "group_size": 16,
                "head_dim": 128,
                "block_size": 64,
                "max_partitions": parts,
                "window": 2048,
                "sink_dtype": torch.float32,
                "capture": True,
                "lengths": (0, 1, 127, 2047, 2048, 2049, 4097, 8192) * 4,
            },
        )
        for parts in (4, 8)
    ],
    (
        "qlen8-d128-w2048-context-bound",
        {
            "query_length": 8,
            "group_size": 16,
            "head_dim": 128,
            "block_size": 64,
            "max_partitions": 8,
            "window": 2048,
            "sink_dtype": torch.float32,
            "capture": True,
            "lengths": (0, 1, 127, 128, 129, 257, 511, 512),
        },
    ),
    (
        "mtp",
        {
            "query_length": 4,
            "kv_heads": 2,
            "group_size": 4,
            "head_dim": 64,
            "max_partitions": 3,
            "sink_dtype": torch.float16,
        },
    ),
    (
        "head256",
        {"kv_heads": 2, "head_dim": 256, "block_size": 16, "max_partitions": 7},
    ),
    ("head192-decode", {"group_size": 16, "head_dim": 192, "max_partitions": 4}),
    (
        "head192-mtp",
        {"query_length": 4, "group_size": 16, "head_dim": 192, "max_partitions": 4},
    ),
    # Large BF16 Q/P operands must split MTP rows to fit the LDS budget.
    *[
        (
            f"head{dim}-mtp-split{splits}",
            {
                "query_length": length,
                "group_size": 16,
                "head_dim": dim,
                "max_partitions": 4,
            },
        )
        for length, dim, splits in ((4, 512, 2), (4, 1024, 4), (3, 1024, 3))
    ],
    ("graph-workspace", {"query_length": 4, "max_partitions": 7, "capture": True}),
    (
        "loose-bound",
        {
            "block_size": 16,
            "lengths": (0, 1, 257, 300, 512),
            "max_partitions": None,
            "table_width": 768,
            "max_context": None,
        },
    ),
    (
        "head192-np80",
        {
            "group_size": 16,
            "head_dim": 192,
            "lengths": (80 * 256,),
            "max_partitions": 80,
        },
    ),
    *[
        (
            f"head{dim}-mixed-np96",
            {
                "query_length": 4,
                "kv_heads": 2,
                "group_size": 16,
                "head_dim": dim,
                "lengths": (97 * 256,) + (0, 1, 257, 1027) * 10,
                "max_partitions": 96,
                "budget": 4096,
            },
        )
        for dim in (256, 128)
    ],
]
# FP8 head 256, page 64, plain V, uniform query length with at most 32 query
# rows: the gfx950 pipe, for per-tensor and per-token scales. A plan-independent
# staging bound selects its specialization;
# other architectures run the same cases on the generic path.
_PIPE_BASE = {
    "kv_dtype": "fp8",
    "scale_mode": "per-tensor",
    "trans_v": False,
    "head_dim": 256,
    "block_size": 64,
    "group_size": 8,
    "pipe": True,
}
_PIPE_CASES = [
    # At GQA 8, query length q gives 8 * q rows: one M-tile at 1 and 2 (1 pads
    # it), two at 3 and 4 (3 pads the second). Lengths put some row's causal
    # edge on a page or tile boundary, and lengths below q leave rows empty.
    # P is FP8, so random rows see 0, 1 or at least 6 tokens; the token
    # one-hot token pattern covers short contexts exactly.
    #
    # A task needing more than 4096 page ids cannot stage the whole list.
    (
        "d256-unstaged-q2",
        {
            **_PIPE_BASE,
            "query_length": 2,
            "lengths": (0, 1, 65, 257, 8192),
            "max_partitions": 4,
            "budget": 1,
            "table_width": 20000,
            "max_context": None,
            "staged": False,
        },
    ),
    # max_context_length below the real context used to index past the staged
    # page ids and fault. The tail must still match a full-context reference.
    (
        "d256-page64-short-bound",
        {
            **_PIPE_BASE,
            "lengths": (128 * 1024,),
            "max_partitions": 1,
            "max_context": 8192,
            "budget": 1,
        },
    ),
    # 81 and 130 partitions leave a short tail, and sequences with few or no
    # tasks must not change a channel.
    *(
        (
            f"d256-page64-q4-np{partitions}",
            {
                **_PIPE_BASE,
                "query_length": 4,
                "lengths": (0, 1, 257, 66532),
                "max_partitions": partitions,
                "budget": partitions,
                "pattern": "one_hot_v",
            },
        )
        for partitions in (81, 130)
    ),
    (
        "d256-page64-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "lengths": (1, 65, 257, 1027),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
    (
        "d256-page64-q4-per-token",
        {
            **_PIPE_BASE,
            "scale_mode": "per-token",
            "query_length": 4,
            "lengths": (0, 1, 65, 257, 2049),
            "max_partitions": 4,
            "pattern": "varied",
        },
    ),
]
_INVALID_CASES = [
    ("fp16-query", NotImplementedError, "BF16 KV requires bfloat16 queries"),
    ("plain-v", ValueError, "BF16 KV requires the vectorized 5D value_cache layout"),
    (
        "scaled-kv",
        ValueError,
        "BF16 KV is unscaled; key_scale and value_scale must be None",
    ),
    ("vector16-k", ValueError, "key_cache shape must be"),
    ("vector16-v", ValueError, "transposed value_cache shape must be"),
]


def _check_output(output, reference, tolerance=5e-3):
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.float(), reference, atol=tolerance, rtol=tolerance
    )


def _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch):
    query, key, value, *rest = arguments
    invalid = case["invalid_input"]
    if invalid == "fp16-query":
        query = query.to(torch.float16)
    elif invalid == "plain-v":
        value = torch.empty((1, 1, 128, 64), dtype=query.dtype, device=query.device)
    elif invalid == "scaled-kv":
        options.update(
            key_scale=torch.ones(1, dtype=torch.float32, device=query.device),
            value_scale=torch.ones(1, dtype=torch.float32, device=query.device),
        )
    elif invalid == "vector16-k":
        key = torch.empty((1, 1, 8, 64, 16), dtype=query.dtype, device=query.device)
    elif invalid == "vector16-v":
        value = torch.empty((1, 1, 4, 128, 16), dtype=query.dtype, device=query.device)

    def unexpected_compile(**_):
        pytest.fail("invalid inputs reached kernel compilation")

    monkeypatch.setattr(pa, "compile_pa_decode_tile", unexpected_compile)
    value_dim = query.shape[-1] if case["value_dim"] is None else case["value_dim"]
    with pytest.raises(case["error"], match=case["match"]):
        pa.pa_decode(
            torch.empty(
                (*query.shape[:-1], value_dim), dtype=query.dtype, device=query.device
            ),
            query,
            key,
            value,
            *rest,
            work_plan=plan,
            **options,
        )


def _one_hot_kv(key, value, table, context, compute_type, tokens_per_channel=256):
    """Zero K and one-hot V in channel (token // tokens_per_channel) % dim.

    With 256 tokens per channel each tile owns one output channel, so a task
    that drops or repeats a tile changes that channel's mean. With one, a row's
    output marks exactly the tokens it sees, and zero K keeps FP8 P exact.
    Scales are explicit because a zero K must not reach the quantizer.
    """
    dim, page = value.shape[-1], value.shape[2]
    dense = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
    for seq, length in enumerate(context.tolist()):
        tokens = torch.arange(length, device=value.device)
        pages = table[seq, tokens // page].long()
        dense[pages, :, tokens % page, (tokens // tokens_per_channel) % dim] = 224
    scale = torch.ones(1, dtype=torch.float32, device=value.device)
    return (
        torch.zeros_like(key, dtype=compute_type),
        scale,
        dense.to(compute_type),
        scale * 0.25,
    )


def _check_pipe_policy(pa, case, arch, compile_kwargs):
    """Pin the selected specialization so a case cannot drift off its path."""
    from aiter.ops.flydsl.kernels.pa_decode.traits import PaDecodeSchedule

    assert compile_kwargs[-1]["wide_kv_addressing"] == case["wide"]
    if case["pipe"]:
        # The cache key ends with (D256_PIPE, staged page ids).
        schedule_kwargs = dict(compile_kwargs[-1])
        schedule_kwargs.pop("architecture", None)
        schedule_kwargs.pop("direct_output", None)
        pipe, staged = PaDecodeSchedule.select(**schedule_kwargs).cache_key[-2:]
        assert pipe == (arch == "gfx950")
        if pipe:
            assert (staged > 0) == (case["staged"] is not False)


@pytest.mark.parametrize(
    "batch,parts,splits,full_budget",
    [
        (1, 2, 8, True),
        (7, 4, 4, True),
        (8, 8, 2, True),
        (9, 16, 4, True),
        (12, 64, 2, True),
        (17, 128, 4, True),
        (33, 8, 1, True),
        (9, 16, 4, False),
    ],
)
def test_mimo_query_grid_dynamic_graph(device, monkeypatch, batch, parts, splits, full_budget):
    """Interleaved Q groups preserve packed ownership and empty-row outputs."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx942":
        pytest.skip("MiMo query-grid scheduling is specific to gfx942")
    from aiter.ops.flydsl.kernels.pa_decode.traits import (
        PaDecodeSchedule,
        PaDecodeTraits,
    )

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(814)
    pattern = (0, 1, 7, 8, 127, 255, 256, 257, 511, 1025, 4097)
    lengths = tuple(pattern[i % len(pattern)] for i in range(batch))
    # Keep a nonempty allocation even for the all-empty B1 entry.
    initial = tuple(max(length, 4097) for length in lengths)
    query, key, value, table, context = _make_inputs(
        8, 1, 16, 192, 128, 64, torch.bfloat16, "random", device,
        initial, table_width=16384,
    )
    # Reserve block zero for padded table reads, and poison every unowned token.
    key = torch.cat([torch.zeros_like(key[:1]), key])
    value = torch.cat([torch.zeros_like(value[:1]), value])
    used_pages = (4097 + 63) // 64
    table[:, :used_pages].add_(1)
    pages = key.shape[0]
    kc = torch.empty((pages, 1, 24, 64, 8), dtype=key.dtype, device=device)
    vc = torch.empty((pages, 1, 8, 128, 8), dtype=value.dtype, device=device)

    def refresh_cache():
        owned = torch.zeros((pages, 64), dtype=torch.bool, device=device)
        for seq, length in enumerate(context.tolist()):
            tokens = torch.arange(length, device=device)
            owned[table[seq, tokens // 64].long(), tokens % 64] = True
        poisoned_key = key.masked_fill(~owned[:, None, :, None], float("nan"))
        poisoned_value = value.masked_fill(~owned[:, None, :, None], float("nan"))
        kc.copy_(poisoned_key.view(pages, 1, 64, 24, 8).permute(0, 1, 3, 2, 4))
        vc.copy_(poisoned_value.view(pages, 1, 8, 8, 128).permute(0, 1, 2, 4, 3))

    context.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
    refresh_cache()
    plan = pa.plan_pa_decode(
        context, 1, max_partitions=parts,
        workgroup_budget=batch * (parts if full_budget else 3), query_length=8,
    )
    shape = (1, plan.capacity, 128)
    pmax = torch.empty(shape, dtype=torch.float32, device=device)
    psum = torch.empty_like(pmax)
    pout = torch.empty((*shape, 128), dtype=torch.bfloat16, device=device)
    guarded = torch.full((batch * 8 + 2, 16, 128), 19.0, dtype=query.dtype, device=device)
    output = guarded[1:-1]
    compiled = []
    original_compile = pa.compile_pa_decode_tile

    def compile_spy(**kwargs):
        selected = {k: v for k, v in kwargs.items() if k not in ("architecture", "direct_output")}
        traits = PaDecodeTraits.create(PaDecodeSchedule.select(**selected))
        compiled.append(traits.query_first_plan_grid)
        return original_compile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", compile_spy)

    def launch():
        pa.plan_pa_decode(context, 1, plan=plan, query_length=8)
        pa.pa_decode(
            output, query, kc, vc, context, table, 192**-0.5, 8,
            work_plan=plan, query_splits=splits, max_context_length=1048576,
            max_logits=pmax, exp_sums=psum, temporary_output=pout,
        )

    def check():
        reference = _reference(query, key.float(), value.float(), table, context, 8, 0, None)
        _check_output(output, reference, 5e-3)
        assert bool((guarded[0] == 19).all() and (guarded[-1] == 19).all())

    launch()
    check()
    assert compiled and all(flag == full_budget for flag in compiled)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch()
    for state in ("all-empty", "mixed", "remapped"):
        if state == "all-empty":
            context.zero_()
        else:
            new_lengths = tuple(pattern[(batch - i) % len(pattern)] for i in range(batch))
            context.copy_(torch.tensor(new_lengths, dtype=torch.int32, device=device))
        if state == "remapped":
            table.copy_(table.roll(1, dims=0))
            query.neg_()
        refresh_cache()
        output.fill_(float("nan"))
        graph.replay()
        check()


_PA_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "aiter/ops/flydsl"
_pa_config_spec = importlib.util.spec_from_file_location("pa_config_under_test", _PA_CONFIG_ROOT / "pa_decode_config.py")
_pa_config_policy = importlib.util.module_from_spec(_pa_config_spec)
_pa_config_spec.loader.exec_module(_pa_config_policy)


class _PAConfigStub:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _pa_tuning_function(name, **namespace):
    source = _PA_CONFIG_ROOT / "pa_decode_tuning.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.unparse(fn), str(source), "exec"), namespace)
    return namespace[name]


class TestPAConfig(unittest.TestCase):
    """CPU-only selection/cache contracts; independent of the GPU fixture."""

    def geometry(self, batch=12, **changes):
        values = dict(arch="gfx942", num_seqs=batch, num_kv_heads=1,
            query_group_size=16, head_dim=192, value_dim=128, block_size=64,
            query_length=8, query_dtype="torch.bfloat16", kv_dtype="torch.bfloat16",
            per_token_kv=False, trans_v=True, sink_dtype=None, sliding_window=0,
            max_partitions=304, num_cu=304, requested_query_splits=None)
        values.update(changes)
        return values

    def test_selected_serving_defaults(self):
        for batch, expected in [(12,(24,1)),(24,(64,2)),(40,(64,2)),(96,(3,1)),
                                (152,(2,1)),(208,(8,2)),(240,(4,2)),(256,(4,2))]:
            with self.subTest(batch=batch):
                result = _pa_config_policy.calibrated_pa_config(**self.geometry(batch))
                self.assertEqual((result['partition_count'],result['query_splits']),expected)
                self.assertEqual(result['workgroup_budget'],batch*expected[0])

    def test_untuned_geometry_and_explicit_cap(self):
        for changes in [dict(arch="gfx950"),dict(num_cu=256),
                        dict(query_length=4),dict(sliding_window=128),dict(sink_dtype="torch.float32"),
                        dict(kv_dtype="torch.float8_e4m3fnuz"),dict(num_kv_heads=2),
                        dict(trans_v=False)]:
            with self.subTest(changes=changes):
                self.assertIsNone(_pa_config_policy.calibrated_pa_config(**self.geometry(**changes)))
        for changes, parts in [(dict(num_seqs=13),16),(dict(max_partitions=16),16),
                                (dict(max_partitions=8),8)]:
            values=self.geometry(**changes)
            result=_pa_config_policy.calibrated_pa_config(**values)
            self.assertEqual(result,dict(partition_count=parts,query_splits=None,
                                        workgroup_budget=values['num_seqs']*parts))

    def test_explicit_split(self):
        result = _pa_config_policy.calibrated_pa_config(**self.geometry(requested_query_splits=2))
        self.assertEqual(result['query_splits'],2)

    def test_old_budget_cache_compatibility(self):
        self.assertEqual(_pa_config_policy.validate_pa_config({'workgroup_budget':608},
            max_partitions=304,query_length=8),(608,304,None))

    def test_invalid_cache_values(self):
        bad = [{}, {'workgroup_budget':True}, {'workgroup_budget':0},
            {'workgroup_budget':608,'partition_count':305},
            {'workgroup_budget':608,'partition_count':False},
            {'workgroup_budget':608,'query_splits':3},
            {'workgroup_budget':608,'query_splits':True},
            {'workgroup_budget':608,'unknown':1}]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _pa_config_policy.validate_pa_config(values,max_partitions=304,query_length=8)
        with self.assertRaises(ValueError):
            _pa_config_policy.validate_pa_config({'workgroup_budget':288,'query_splits':2},
                max_partitions=304,query_length=8,requested_query_splits=1)

    def test_joint_candidates_are_not_deduplicated_by_capacity(self):
        prepare_calls=[]
        resources=SimpleNamespace(args=(object(),object(),SimpleNamespace(numel=lambda:12),None,None,8),
            options={'query_splits':None},max_partitions=128,prepare=lambda *a:prepare_calls.append(a))
        fn=_pa_tuning_function('_pa_decode_configs',Config=_PAConfigStub,_cache_geometry=lambda key:(1,64))
        results=[x.kwargs for x in fn(resources=resources,num_cu=304)]
        self.assertIn(dict(workgroup_budget=288,partition_count=24,query_splits=1),results)
        self.assertIn(dict(workgroup_budget=288,partition_count=24,query_splits=2),results)
        self.assertTrue(prepare_calls)
        resources.options['query_splits']=1
        self.assertTrue(all(x.kwargs['query_splits']==1 for x in fn(resources=resources,num_cu=304)))

    def test_cache_only_does_not_search(self):
        fn=_pa_tuning_function('_resolve_pa_decode_config')
        cached=_PAConfigStub(workgroup_budget=288,partition_count=24,query_splits=1)
        calls=[]
        tuner=SimpleNamespace(_make_key=lambda *a:'key',cache={'key':cached},
            _artifact_ref=lambda *a,**kw:None,_load_artifact=lambda *a:None,
            default=lambda **kw:calls.append('default') or cached,
            resolve_config=lambda **kw:(_ for _ in ()).throw(AssertionError('Search called')))
        self.assertIs(fn(tuner,cache_only=True),cached)
        self.assertEqual(calls,[])
        tuner.cache.clear()
        self.assertIs(fn(tuner,cache_only=True),cached)
        self.assertEqual(calls,['default'])


@pytest.mark.parametrize("batch,parts,splits", [(12,24,1),(24,64,2),(40,64,2),(96,3,1),
    (152,2,1),(208,8,2),(240,4,2),(256,4,2),(7,16,None)])
def test_mimo_auto_joint_plan(device, monkeypatch, tmp_path, batch, parts, splits):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx942" or torch.cuda.get_device_properties(device).multi_processor_count != 304:
        pytest.skip("Calibrated MiMo defaults require gfx942")
    from aiter.ops.flydsl import pa_decode_tuning as tuning
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(813)
    q,k,v,table,lengths = _make_inputs(8,1,16,192,128,64,torch.bfloat16,
        "random",device,(256,)*batch,table_width=64)
    lengths.copy_(torch.tensor([128+(i%3)*57 for i in range(batch)],device=device,dtype=torch.int32))
    kc=k.view(k.shape[0],1,64,24,8).permute(0,1,3,2,4).contiguous()
    vc=v.view(v.shape[0],1,8,8,128).permute(0,1,2,4,3).contiguous()
    monkeypatch.setenv("FLYDSL_AUTOTUNE","1")
    def no_search(*args,**kwargs):
        raise AssertionError("Serving cache-only preparation launched tuning candidates")
    monkeypatch.setattr(tuning._PADecodeAutotuneResources,"prepare",no_search)
    plan=tuning.prepare_pa_decode_plan(q,kc,vc,lengths,table,192**-.5,8,
        max_batch_size=batch,max_context_length=4096,cache_only=True,config_dir=tmp_path)
    assert (plan.max_partitions,plan.query_splits)==(parts,splits)
    shape=(1,plan.capacity,128)
    pmax=torch.empty(shape,device=device,dtype=torch.float32)
    psum=torch.empty_like(pmax)
    pout=torch.empty((*shape,128),device=device,dtype=torch.bfloat16)
    output=torch.empty((batch*8,16,128),device=device,dtype=torch.bfloat16)
    pointers=(plan.work_info.data_ptr(),plan.reduce_info.data_ptr())
    def run():
        assert pa.plan_pa_decode(lengths,1,plan=plan,query_length=8) is plan
        pa.pa_decode(output,q,kc,vc,lengths,table,192**-.5,8,work_plan=plan,
            max_context_length=4096,max_logits=pmax,exp_sums=psum,temporary_output=pout)
    def check():
        reference=_reference(q,k.float(),v.float(),table,lengths,8,0,None)
        _check_output(output,reference,5e-3)
    run();check()
    graph=torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):run()
    for state in ("mixed_empty","remapped"):
        if state=="mixed_empty":lengths[::2]=0
        else:
            lengths.fill_(185)
            table.copy_(table.roll(1,dims=0))
            q.neg_()
        output.fill_(float("nan"))
        graph.replay();check()
    assert pointers==(plan.work_info.data_ptr(),plan.reduce_info.data_ptr())
    assert plan.query_splits==splits


def test_mimo_joint_tuning_cache_and_resources(device, monkeypatch, tmp_path):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx942" or torch.cuda.get_device_properties(device).multi_processor_count != 304:
        pytest.skip("Calibrated MiMo defaults require gfx942")
    from aiter.ops.flydsl import pa_decode_tuning as tuning
    from flydsl.autotune import Config
    q,k,v,table,lengths=_make_inputs(8,1,16,192,128,64,torch.bfloat16,
        "random",device,(256,)*12,table_width=64)
    kc=k.view(k.shape[0],1,64,24,8).permute(0,1,3,2,4).contiguous()
    vc=v.view(v.shape[0],1,8,8,128).permute(0,1,2,4,3).contiguous()
    arguments=(q,kc,vc,lengths,table,192**-.5,8)
    captured={}
    original=tuning._resolve_pa_decode_config
    def observe(tuner,**kwargs):
        captured.update(tuner=tuner,args={k:v for k,v in kwargs.items() if k!='cache_only'})
        return original(tuner,**kwargs)
    monkeypatch.setattr(tuning,'_resolve_pa_decode_config',observe)
    plan=tuning.prepare_pa_decode_plan(*arguments,max_batch_size=12,
        max_context_length=4096,cache_only=True,config_dir=tmp_path)
    assert (plan.max_partitions,plan.query_splits)==(24,1)
    tuner=captured['tuner'];key=tuner._make_key((),captured['args'])
    tuner.cache[key]=Config(workgroup_budget=96,partition_count=8,query_splits=2)
    tuner._save_disk_cache();tuner.cache.clear();tuner._load_disk_cache()
    cached=tuning.prepare_pa_decode_plan(*arguments,max_batch_size=12,
        max_context_length=4096,cache_only=True,config_dir=tmp_path)
    assert (cached.max_partitions,cached.query_splits,cached.capacity)==(8,2,96)
    options=dict(compute_type=None,key_scale=None,value_scale=None,sinks=None,
                 sliding_window=0,max_context_length=4096,query_splits=None)
    resources=tuning._PADecodeAutotuneResources(q,arguments[1:],options,304,12)
    for budget,parts,splits in [(608,304,None),(288,24,1),(288,24,2)]:
        resources.prepare(budget,parts,splits)
    assert len(resources.graphs)==3
    for key,graph in resources.graphs.items():
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(resources.outputs[key],resources.reference,atol=5e-3,rtol=5e-3)


@pytest.mark.parametrize(
    "case",
    [
        *[
            pytest.param(
                dict(
                    geometry,
                    kv_dtype="fp8",
                    query_dtype=dtype,
                    scale_mode=scale_mode,
                    trans_v=trans_v,
                ),
                id=f"fp8-{dtype}-{scale_mode}-{'transposed' if trans_v else 'plain'}-v-{name}",
            )
            for name, geometry in _FP8_CASES
            for dtype, scale_mode, trans_v in product(
                (torch.bfloat16, torch.float16),
                ("per-tensor", "per-token"),
                (False, True),
            )
            if geometry.get("value_dim") is None
            or (
                dtype == torch.bfloat16
                and trans_v
                and (scale_mode == "per-tensor" or geometry.get("kv_heads", 1) > 1)
            )
        ],
        *[
            pytest.param(geometry, id=f"fp8-{name}")
            for name, geometry in _FP8_HEAD_CASES
        ],
        *[pytest.param(geometry, id=f"bf16-{name}") for name, geometry in _BF16_CASES],
        pytest.param(
            dict(
                _ASYMMETRIC_GENERAL_CASES[1][1],
                kv_dtype="fp8",
                query_dtype=torch.float16,
                scale_mode="per-token",
                trans_v=False,
            ),
            id="fp8-asymmetric-fp16-plain-v",
        ),
        *[pytest.param(geometry, id=name) for name, geometry in _PIPE_CASES],
        *[
            pytest.param(
                {
                    "invalid_input": invalid,
                    "error": error,
                    "match": match,
                    "lengths": (1,),
                },
                id=f"invalid-bf16-{invalid}",
            )
            for invalid, error, match in _INVALID_CASES
        ],
    ],
)
def test_pa_decode(case, device, monkeypatch):
    """FP8/BF16 decode shares input, reference, planning and replay checks."""
    from aiter import per_tensor_quant, pertoken_quant
    from aiter.jit.utils.chip_info import get_gfx_runtime

    case = dict(
        {
            "query_length": 1,
            "kv_heads": 1,
            "group_size": 8,
            "head_dim": 128,
            "value_dim": None,
            "block_size": 64,
            "max_partitions": 1,
            "window": 0,
            "sink_dtype": None,
            "pattern": "random",
            "kv_dtype": "bf16",
            "query_dtype": torch.bfloat16,
            "scale_mode": "none",
            "trans_v": True,
            "capture": False,
            "lengths": None,
            "budget": None,
            "table_width": None,
            "max_context": "lengths",
            "wide": False,
            "pipe": False,
            "staged": None,
        },
        **case,
    )
    arch = get_gfx_runtime()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    if case["max_partitions"] is not None and case["max_partitions"] > num_cu:
        pytest.skip(f"plan_pa_decode caps max_partitions at {num_cu} CUs")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if get_gfx_runtime() == "gfx1250":
        if case.get("invalid_input") in ("fp16-query", "plain-v"):
            pytest.skip("gfx1250 supports FP16 queries and plain BF16 V caches")
        if case["head_dim"] % 128 and case["head_dim"] != 64:
            pytest.skip("gfx1250 supports head_dim 64 or multiples of 128")
    query_length, kv_heads, group_size, head_dim, page, window = (
        case[name]
        for name in (
            "query_length",
            "kv_heads",
            "group_size",
            "head_dim",
            "block_size",
            "window",
        )
    )
    value_dim = head_dim if case["value_dim"] is None else case["value_dim"]
    if value_dim != head_dim and arch not in ("gfx942", "gfx950"):
        pytest.skip("asymmetric Q/K and V widths require gfx942/gfx950")
    torch.manual_seed(0)
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        value_dim,
        page,
        case["query_dtype"],
        case["pattern"],
        device,
        case["lengths"],
        case["table_width"],
    )
    compute_type = torch.bfloat16
    key_scale = value_scale = None
    vector = 8
    if case["kv_dtype"] == "fp8":
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() in ("gfx950", "gfx1250")
            else torch.float8_e4m3fnuz
        )
        vector = 16
        mode = case["scale_mode"]
        if case["pattern"] in ("one_hot_v", "one_hot_token"):
            tokens_per_channel = 1 if case["pattern"] == "one_hot_token" else 256
            key, key_scale, value, value_scale = _one_hot_kv(
                key, value, table, context, compute_type, tokens_per_channel
            )
        elif mode in ("per-tensor", "per-token"):
            quantize = pertoken_quant if mode == "per-token" else per_tensor_quant
            key, key_scale = quantize(key, quant_dtype=compute_type)
            value, value_scale = quantize(value, quant_dtype=compute_type)
        else:
            key, value = key.to(compute_type), value.to(compute_type)
            if mode == "python":
                key_scale, value_scale = 0.25, 2.0
    logical_key = key.float() * (1.0 if key_scale is None else key_scale)
    logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    pages = key.shape[0]
    key_cache = (
        key.reshape(pages, kv_heads, page, head_dim // vector, vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = (
        value.reshape(pages, kv_heads, page // vector, vector, value_dim)
        .permute(0, 1, 2, 4, 3)
        .contiguous()
        if case["trans_v"]
        else value.permute(0, 1, 3, 2).contiguous()
    )
    sinks = None
    if case["sink_dtype"] is not None:
        sinks = torch.linspace(
            -2, 2, query.shape[1], dtype=case["sink_dtype"], device=device
        )
        sinks[0], sinks[-1] = float("-inf"), float("inf")
    options = {
        "compute_type": compute_type,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "sinks": sinks,
        "sliding_window": window,
        "max_context_length": (
            None
            if case["max_context"] is None
            else (
                context.max().item()
                if case["max_context"] == "lengths"
                else case["max_context"]
            )
        ),
    }
    arguments = (
        query,
        key_cache,
        value_cache,
        context,
        table,
        head_dim**-0.5,
        query_length,
    )
    plan = pa.plan_pa_decode(
        context,
        kv_heads,
        max_partitions=case["max_partitions"],
        workgroup_budget=case["budget"],
        query_length=query_length,
        sliding_window=window,
    )
    if "invalid_input" in case:
        _check_unsupported_input(pa, arguments, options, plan, case, monkeypatch)
        return

    compile_kwargs = []
    compile_tile = pa.compile_pa_decode_tile

    def spy_tile(**kwargs):
        compile_kwargs.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy_tile)

    output = torch.full(
        (*query.shape[:-1], value_dim),
        float("nan"),
        dtype=query.dtype,
        device=device,
    )
    workspace = {}
    if case["capture"]:
        scalar_shape = (kv_heads, plan.capacity, query_length * group_size)
        workspace = {
            "exp_sums": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "max_logits": torch.empty(scalar_shape, dtype=torch.float32, device=device),
            "temporary_output": torch.empty(
                (*scalar_shape, value_dim), dtype=query.dtype, device=device
            ),
        }

    def launch():
        pa.pa_decode(output, *arguments, work_plan=plan, **options, **workspace)

    def check():
        reference = _reference(
            query,
            logical_key,
            logical_value,
            table,
            context,
            query_length,
            window,
            sinks,
        )
        tolerance = (
            1e-2 if case["kv_dtype"] == "fp8" and value_dim != head_dim else 5e-3
        )
        _check_output(output, reference, tolerance)

    launch()
    check()
    _check_pipe_policy(pa, case, arch, compile_kwargs)
    if case["capture"]:
        # Compile/allocate before capture; replay must observe changed query data.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            launch()
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            launch()
        for _ in range(2):
            query.neg_()
            output.fill_(float("nan"))
            graph.replay()
            check()


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_asymmetric_multihead_large_address(kv_dtype, device):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip("asymmetric Q/K and V widths require gfx942/gfx950")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    fp8 = (
        torch.float8_e4m3fnuz if get_gfx_runtime() == "gfx942" else torch.float8_e4m3fn
    )
    dtype = torch.bfloat16 if kv_dtype == "bf16" else fp8
    vector = 8 if kv_dtype == "bf16" else 16
    element_bytes = 2 if kv_dtype == "bf16" else 1
    # Both referenced K and V pages lie above a 2 GiB byte offset.
    page = 2**31 // (2 * 128 * 64 * element_bytes) + 1
    key = torch.empty(
        (page + 1, 2, 192 // vector, 64, vector), dtype=dtype, device=device
    )
    value = torch.empty(
        (page + 1, 2, 64 // vector, 128, vector), dtype=dtype, device=device
    )
    key[0].zero_()
    value[0].zero_()
    key[page].zero_()
    value[page, 0].fill_(2)
    value[page, 1].fill_(4)
    query = torch.zeros((8, 16, 192), dtype=torch.bfloat16, device=device)
    output = torch.empty((8, 16, 128), dtype=query.dtype, device=device)
    context = torch.tensor([64], dtype=torch.int32, device=device)
    table = torch.tensor([[page]], dtype=torch.int32, device=device)
    plan = pa.plan_pa_decode(context, 2, query_length=8, max_partitions=4)
    pa.pa_decode(
        output, query, key, value, context, table, 192**-0.5, 8, work_plan=plan
    )
    expected = torch.empty_like(output)
    expected[:, :8].fill_(2)
    expected[:, 8:].fill_(4)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def _poison_unowned(cache, scale, owned, fill):
    """Overwrite K/V rows no sequence owns (and their per-token scales)."""
    rows = ~owned[:, None, :].expand(-1, cache.shape[1], -1)
    if cache.dtype == torch.bfloat16:
        bits = {"zero": 0, "nan": 0x7FC0, "inf": 0x7F80}[fill]
        cache.view(torch.int16)[rows] = bits
    else:
        # E4M3 has no infinity; its largest finite magnitude stands in.
        nan = 0x80 if cache.dtype == torch.float8_e4m3fnuz else 0x7F
        bits = {"zero": 0, "nan": nan, "inf": 0x7E}[fill]
        cache.view(torch.uint8)[rows] = bits
    if scale is not None and scale.numel() > 1:
        scale.view(scale.shape[:3])[rows] = {"zero": 0.0, "nan": float("nan")}.get(
            fill, float("inf")
        )


@pytest.mark.parametrize("fill", ["nan", "inf"])
@pytest.mark.parametrize(
    "kv,block_size,query_length,kv_heads,group_size,head_dim",
    [
        # Generic multi-query, page16/page64 D256 pipe, and page128 prefetch.
        ("fp8-per-tensor", 16, 4, 1, 16, 128),
        ("fp8-per-tensor", 16, 4, 2, 8, 256),
        ("fp8-per-tensor", 64, 4, 2, 8, 256),
        ("fp8-per-tensor", 128, 1, 1, 8, 128),
        # Plain V addressing across all page sizes and single/multi-query paths.
        ("fp8-per-tensor-plain-v", 16, 4, 2, 8, 256),
        ("fp8-per-tensor-plain-v", 64, 1, 1, 8, 128),
        ("fp8-per-tensor-plain-v", 128, 4, 1, 16, 128),
        # Scalar/vector scale staging, single-query and gfx950 query splitting.
        ("fp8-per-token", 16, 4, 2, 8, 256),
        ("fp8-per-token", 64, 4, 1, 16, 128),
        ("fp8-per-token", 128, 1, 1, 8, 128),
        ("fp8-per-token", 128, 4, 1, 16, 128),
        # Both BF16 tail-mask implementations, with multiple heads/M-tiles.
        ("bf16", 16, 1, 1, 8, 128),
        ("bf16", 64, 4, 2, 8, 256),
        ("bf16", 128, 4, 1, 16, 128),
    ],
)
def test_pa_decode_unowned_kv_rows(
    device, kv, block_size, query_length, kv_heads, group_size, head_dim, fill
):
    """Rows no sequence owns contribute exactly zero, even when NaN or Inf.

    Padded block-table lanes read block 0 and the last page's tail past the
    context is read as is. Both may hold NaN (a recycled page, a checkpoint
    image); P is zero there but 0 * NaN is NaN in PV.
    ``mask_unowned_kv=False`` skips the masking and matches it on clean input.
    """
    from aiter import per_tensor_quant, pertoken_quant
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() == "gfx1250":
        pytest.skip("gfx1250 pipelines still require finite unowned V rows")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.manual_seed(0)
    lengths = tuple(max(n, query_length) for n in (1, 17, 255, 257, 1000))
    query, key, value, table, context = _make_inputs(
        query_length,
        kv_heads,
        group_size,
        head_dim,
        head_dim,
        block_size,
        torch.bfloat16,
        "random",
        device,
        lengths,
    )
    # Page 0 is owned by no sequence; padded block-table lanes point at it.
    spare = torch.zeros_like(key[:1])
    key, value = torch.cat([spare, key]), torch.cat([spare, value])
    table = torch.nn.functional.pad(table + 1, (0, 2))
    owned = torch.zeros(key.shape[0], block_size, dtype=torch.bool, device=device)
    for seq, length in enumerate(lengths):
        tokens = torch.arange(length, device=device)
        owned[table[seq, tokens // block_size].long(), tokens % block_size] = True

    key_scale = value_scale = None
    vector = 8
    if kv.startswith("fp8"):
        compute_type = (
            torch.float8_e4m3fn
            if get_gfx_runtime() == "gfx950"
            else torch.float8_e4m3fnuz
        )
        vector = 16
        quantize = pertoken_quant if kv == "fp8-per-token" else per_tensor_quant
        key, key_scale = quantize(key, quant_dtype=compute_type)
        value, value_scale = quantize(value, quant_dtype=compute_type)
    logical_key = key.float() * (1.0 if key_scale is None else key_scale)
    logical_value = value.float() * (1.0 if value_scale is None else value_scale)
    reference = _reference(
        query, logical_key, logical_value, table, context, query_length, 0, None
    )
    plan = pa.plan_pa_decode(context, kv_heads, query_length=query_length)

    def run(fill, mask_unowned_kv=True):
        k, v = key.clone(), value.clone()
        ks = None if key_scale is None else key_scale.clone()
        vs = None if value_scale is None else value_scale.clone()
        _poison_unowned(k, ks, owned, fill)
        _poison_unowned(v, vs, owned, fill)
        pages = k.shape[0]
        key_cache = (
            k.reshape(pages, kv_heads, block_size, head_dim // vector, vector)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        if kv == "fp8-per-tensor-plain-v":
            value_cache = v.permute(0, 1, 3, 2).contiguous()
        else:
            value_cache = (
                v.reshape(pages, kv_heads, block_size // vector, vector, head_dim)
                .permute(0, 1, 2, 4, 3)
                .contiguous()
            )
        output = torch.full_like(query, float("nan"))
        pa.pa_decode(
            output,
            query,
            key_cache,
            value_cache,
            context,
            table,
            head_dim**-0.5,
            query_length,
            compute_type=key_cache.dtype,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            max_context_length=max(lengths),
            mask_unowned_kv=mask_unowned_kv,
        )
        return output

    clean, poisoned = run("zero"), run(fill)
    assert torch.isfinite(poisoned).all()
    # Short FP8 contexts round P coarsely; exactness is the bitwise check.
    tol = 5e-3 if kv == "bf16" else 2.5e-2
    torch.testing.assert_close(poisoned.float(), reference, atol=tol, rtol=tol)
    assert torch.equal(poisoned, clean)
    if fill == "nan":
        assert torch.equal(run("zero", mask_unowned_kv=False), clean)
