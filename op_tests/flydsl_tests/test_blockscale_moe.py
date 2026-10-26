# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed-config gfx950 blockscale kernels, graphs and public dispatch."""

import importlib
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import GateMode, moe_sorting
from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import fused_moe_gfx942 as backend
from aiter.ops.flydsl.kernels.moe_gemm_2stage.moe_8wave_down import flydsl_moe_gemm_8wave_down
from aiter.ops.flydsl.kernels.moe_gemm_2stage.a8w8_blockscale_gemm1 import fp8_blockscale_stage1
from aiter.ops.flydsl.kernels.moe_gemm_2stage.a8w8_blockscale_gemm2 import (
    M128_QUEUE_SIZE,
    make_m128_down,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.moe_packed_reduce import make_moe_sum
from op_tests.test_flydsl_moe_blockscale import (
    check_result,
    prepare_case,
    run_torch,
    torch_stages,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950", reason="requires gfx950"
)
_PATHS = ("block_m128", "block_m128_persistent", "block_8wave")


def config(path, splits=1, n=64):
    return backend.Config(
        256,
        256,
        128,
        True,
        down_path=path,
        down_block_n=n if path == "block_8wave" else 128,
        num_oc_splits=splits,
    )


def run(request, cfg):
    return backend.run_flydsl_moe_gfx942_impl(request, cfg.to_string())


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("tokens", [1, 127, 128, 129, 255, 256, 257])
def test_whole_chain(path, tokens):
    request, raw = prepare_case(tokens, 512, 256, routing="skewed")
    check_result(run_torch(request, raw), run(request, config(path)), path)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("k_rotate", [False, True])
def test_whole_chain_k_rotate(path, k_rotate):
    # K=1024 gives 8 K tiles, so rotation reorders the GateUp K loop per expert.
    request, raw = prepare_case(257, 1024, 256, experts=5, topk=3)
    cfg = replace(config(path), k_rotate=k_rotate)
    assert cfg.to_string().endswith(f":{int(k_rotate)}")
    check_result(run_torch(request, raw), run(request, cfg), path)


@pytest.mark.parametrize(
    "path,inter,n,splits",
    [
        ("block_m128", 512, 128, 4),
        ("block_m128_persistent", 512, 128, 4),
        ("block_m128", 768, 128, 2),
        ("block_m128_persistent", 1024, 128, 8),
        ("block_8wave", 128, 64, 4),
        ("block_8wave", 384, 64, 4),
        ("block_8wave", 640, 32, 8),
        ("block_8wave", 1152, 32, 2),
        ("block_8wave", 256, 128, 2),
    ],
)
def test_k_tiles_and_oc_splits(path, inter, n, splits):
    request, raw = prepare_case(33, 1024, inter, experts=5, topk=3)
    check_result(run_torch(request, raw), run(request, config(path, splits, n)), path)


@pytest.mark.parametrize("path", _PATHS)
def test_whole_chain_signed_slots(path):
    request, raw = prepare_case(1, 256, 256, experts=130, topk=130)
    check_result(run_torch(request, raw), run(request, config(path, n=32)), path)


@pytest.mark.parametrize("topk,experts", [(2, 4), (130, 130)])
def test_stage1_scale_layout_and_signed_slots(topk, experts):
    request, raw = prepare_case(1 if topk > 127 else 17, 256, 128, experts, topk)
    aq, sa, reference, *_ = torch_stages(request, raw)
    si, _, se, valid, _ = moe_sorting(
        request.topk_ids, request.topk_weight, experts, 256, torch.bfloat16, 256
    )
    output = torch.full_like(reference, float("nan"))
    fp8_blockscale_stage1(
        aq, request.w1, sa.t().contiguous(), request.w1_scale, si, se, valid, output
    )
    check_result(reference, output, "GateUp")
    saved = valid.clone()
    valid.zero_()
    output.fill_(13)
    fp8_blockscale_stage1(
        aq, request.w1, sa.t().contiguous(), request.w1_scale, si, se, valid, output
    )
    assert torch.all(output == 13)
    valid.copy_(saved)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("sort_m", [128, 256])
def test_down_counter_and_layout(path, sort_m):
    if path == "block_8wave" and sort_m == 128:
        pytest.skip("routed 8-wave Down requires native sort256")
    request, raw = prepare_case(129, 512, 256, routing="skewed")
    _, _, _, dq, sd, reference = torch_stages(request, raw)
    si, sw, se, valid, _ = moe_sorting(
        request.topk_ids, request.topk_weight, 4, 512, torch.bfloat16, sort_m
    )
    params = {"n": 512, "k": 256, "topk": 2, "num_experts": 4, "num_oc_splits": 1}
    if path == "block_8wave":
        down = flydsl_moe_gemm_8wave_down(**params, block_n=64)
        routes = torch.empty(129, 2, 512, dtype=torch.bfloat16, device=dq.device)
        counter = torch.full((1,), 12345, dtype=torch.int32, device=dq.device)
    else:
        persistent = path == "block_m128_persistent"
        down = make_m128_down(
            **params, persistent=persistent, is_sorted_bm_256=sort_m == 256
        )
        routes = torch.empty(
            se.numel() * sort_m, 512, dtype=torch.bfloat16, device=dq.device
        )
        counter = torch.zeros(
            M128_QUEUE_SIZE if persistent else 1, dtype=torch.int32, device=dq.device
        )
    scale = sd.t().contiguous()
    saved = valid.clone()
    for empty in (False, True, False):
        valid.zero_() if empty else valid.copy_(saved)
        routes.fill_(float("nan"))
        down(
            routes, dq, request.w2, scale, request.w2_scale, si, sw, se, valid, counter
        )
        if path == "block_m128_persistent":
            assert torch.count_nonzero(counter) == 0
        elif path == "block_m128":
            assert counter.item() == 0
        if empty:
            assert torch.isnan(routes).all()
            continue
        if path == "block_8wave":
            result = routes.sum(1)
        else:
            inverse = torch.full((129, 2), -1, dtype=torch.int32, device=dq.device)
            backend.invert_sorted_ids(2)(si, inverse, valid, si.numel(), 129)
            assert torch.all(inverse >= 0)
            result = torch.empty(129, 512, dtype=torch.bfloat16, device=dq.device)
            make_moe_sum(n=512, topk=2, sort_block_m=128)(result, routes, inverse)
        check_result(reference, result, path)


@pytest.mark.parametrize(
    "path,inter",
    [
        ("block_m128", 256),
        ("block_m128", 512),
        ("block_m128_persistent", 256),
        ("block_m128_persistent", 512),
        ("block_8wave", 384),
        ("block_8wave", 640),
    ],
)
def test_stream_graph_replay(path, inter):
    request, raw = prepare_case(257, 1024, inter, experts=5, topk=3)
    cfg = config(path, splits=2, n=32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(request, cfg)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            first = run(request, cfg)
            second = run(request, cfg)
    stream.synchronize()
    original = request.hidden_states.clone()
    for factor in (0.0, 1.0, -0.5):
        request.hidden_states.copy_(original * factor)
        request.topk_ids.copy_((request.topk_ids + 1) % 5)
        request.topk_weight.mul_(-0.5)
        request.w1_scale.mul_(0.75)
        request.w2_scale.mul_(1.25)
        expected = run_torch(request, raw)
        first.fill_(float("nan"))
        second.fill_(float("nan"))
        graph.replay()
        check_result(expected, first, path)
        check_result(expected, second, path)


@pytest.mark.parametrize("path", ["block_m128_persistent", "block_8wave"])
def test_concurrent_graphs_have_private_counters(path):
    captured = []
    for seed in (11, 12):
        request, raw = prepare_case(129, 512, 256, seed=seed)
        expected = run_torch(request, raw)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run(request, config(path))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run(request, config(path))
                output = run(request, config(path))
        stream.synchronize()
        captured.append((request, expected, stream, graph, output))
    for _ in range(3):
        for _, _, stream, graph, output in captured:
            with torch.cuda.stream(stream):
                output.fill_(float("nan"))
                graph.replay()
        for _, expected, stream, _, output in captured:
            stream.synchronize()
            check_result(expected, output, path)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("alias", [QuantType.per_1x128, QuantType.per_128x128])
def test_public_dispatch_and_output_buffer(monkeypatch, path, alias):
    fm = importlib.import_module("aiter.fused_moe")
    request, raw = prepare_case(17, 512, 256)
    cfg = config(path)
    key = (
        fm.get_gfx_runtime(),
        fm.get_cu_num(),
        fm.get_padded_M(17),
        512,
        256,
        4,
        2,
        str(ActivationType.Silu),
        str(torch.bfloat16),
        str(request.w1.dtype),
        str(request.w1.dtype),
        str(QuantType.per_1x128),
        True,
        False,
    )
    row = {
        "kernelName1": make_fused_moe_impl_kernel_name(
            "flydsl_gfx950", cfg.to_string()
        ),
        "kernelName2": "",
        "block_m": 256,
        "ksplit": 0,
        "run_1stage": 0,
    }
    monkeypatch.setattr(fm, "cfg_2stages", ({key: row}, {}))
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "0")
    fm.get_2stage_cfgs.cache_clear()
    output = torch.full_like(request.hidden_states, float("nan"))

    def public_call():
        return fm.fused_moe(
            request.hidden_states,
            request.w1,
            request.w2,
            request.topk_weight,
            request.topk_ids,
            activation=ActivationType.Silu,
            quant_type=alias,
            w1_scale=request.w1_scale,
            w2_scale=request.w2_scale,
            output=output,
        )

    try:
        with patch.object(
            backend, "_run_blockscale_prefill", wraps=backend._run_blockscale_prefill
        ) as called:
            result = public_call()
        assert (
            called.call_count == 1
        ), "public dispatch must execute the ported pipeline"
        assert result is output
        check_result(run_torch(request, raw), result, "public blockscale")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            public_call()
            captured = public_call()
        for value in (0.0, 0.05):
            request.hidden_states.fill_(value)
            request.topk_ids.copy_((request.topk_ids + 1) % 4)
            expected = run_torch(request, raw)
            output.fill_(float("nan"))
            graph.replay()
            assert captured is output
            check_result(expected, output, "public blockscale graph")
    finally:
        fm.get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize(
    "change",
    [
        {"activation": ActivationType.Swiglu},
        {"swiglu_limit": 7.0},
        {"doweight_stage1": True},
        {"gate_mode": GateMode.INTERLEAVE},
        {"q_dtype_a": torch.bfloat16},
        {"block_size_m": 128},
        {"ksplit": 2},
    ],
)
def test_unsupported_requests_fail_before_kernel(change):
    request, _ = prepare_case(1, 256, 256)
    with patch.object(backend, "_run_blockscale_prefill") as launch:
        with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
            run(replace(request, **change), config("block_m128"))
        launch.assert_not_called()


def test_gfx942_rejects_new_path(monkeypatch):
    request, _ = prepare_case(1, 256, 256)
    monkeypatch.setattr(backend, "get_gfx", lambda: "gfx942")
    with pytest.raises(NotImplementedError, match="requires gfx950"):
        run(request, config("block_m128"))
