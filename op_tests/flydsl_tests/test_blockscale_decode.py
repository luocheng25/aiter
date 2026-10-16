# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed FlyDSL A16W8 decode candidates; no ASM or external kernel reference."""

import importlib
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import moe_sorting
from aiter.fused_moe_registry import make_fused_moe_impl_kernel_name
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import fused_moe_gfx942 as backend
from aiter.ops.shuffle import shuffle_weight
from op_tests.test_flydsl_moe_blockscale import (
    check_result,
    prepare_case,
    run_torch,
    torch_stages_a16w8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="requires gfx942 or gfx950",
)
_FP8_DTYPE = torch.float8_e4m3fn if get_gfx() == "gfx950" else torch.float8_e4m3fnuz


def _config(tokens, block_n=64, direct=False):
    if tokens == 1 or direct:
        return backend.Config(
            16, 16, 16, False, down_path="block_a16w8", use_batch1_algorithm=tokens > 1
        )
    return backend.Config(16, block_n, 64, False, down_path="block_a16w8")


def _case(tokens, hidden=512, inter=256, experts=5, topk=3, routing="random"):
    request, raw = prepare_case(
        tokens, hidden, inter, experts, topk, routing=routing, quant_dtype=_FP8_DTYPE
    )
    # Vary every N128/K128 block and distinguish Gate from Up, not all-one scales.
    for scale in (request.w1_scale, request.w2_scale):
        scale.mul_(
            torch.linspace(0.25, 2.0, scale.numel(), device=scale.device).view_as(scale)
        )
    return request, raw


def _run(request, cfg):
    return backend.run_flydsl_moe_gfx942_impl(request, cfg.to_string())


@pytest.mark.parametrize("block_n", [64, 128])
@pytest.mark.parametrize("tokens", [2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256])
def test_sorted_decode(tokens, block_n):
    request, raw = _case(tokens, routing="skewed" if tokens % 2 else "random")
    check_result(
        torch_stages_a16w8(request, raw)[1],
        _run(request, _config(tokens, block_n)),
        "A16W8 sorted",
    )


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 7, 8])
def test_direct_decode(tokens):
    request, raw = _case(tokens)
    check_result(
        torch_stages_a16w8(request, raw)[1],
        _run(request, _config(tokens, direct=True)),
        "A16W8 direct",
    )


def test_last_supported_slot_and_padding():
    request, raw = _case(2, inter=128, experts=127, topk=127)
    # Isolate the highest encoded slot, not the separate error from summing
    # 127 nonzero routes with BF16 atomics instead of the FP32 Torch reduce.
    request.topk_weight.zero_()
    request.topk_weight[:, -1] = 1.0
    check_result(
        torch_stages_a16w8(request, raw)[1],
        _run(request, _config(2)),
        "A16W8 slot126/padding127",
    )


@pytest.mark.parametrize(
    "hidden,inter",
    [(512, 128), (1024, 256), (1536, 384), (1024, 640), (4096, 256), (6144, 256)],
)
@pytest.mark.parametrize("block_n,direct", [(64, False), (128, False), (64, True)])
def test_k_tiles_and_topk9(hidden, inter, block_n, direct):
    tokens = 4 if direct else 17
    request, raw = _case(tokens, hidden, inter, experts=11, topk=9)
    check_result(
        torch_stages_a16w8(request, raw)[1],
        _run(request, _config(tokens, block_n, direct)),
        "A16W8 K tiles",
    )


@pytest.mark.parametrize("block_n", [64, 128])
@pytest.mark.parametrize("hidden,inter", [(512, 128), (1024, 256), (1536, 384)])
def test_stage_outputs_and_empty_sort(block_n, hidden, inter):
    request, raw = _case(17, hidden, inter)
    ref_mid, reference = torch_stages_a16w8(request, raw)
    si, sw, se, valid, output = moe_sorting(
        request.topk_ids, request.topk_weight, 5, hidden, torch.bfloat16, 16
    )
    mid = torch.full_like(ref_mid, float("nan"))
    kernels = [
        backend._get_compiled_kernel(
            N=n,
            K=k,
            weight_dtype_str="fp8",
            quant_type_str="block",
            TOPK=3,
            BLOCK_TILE_SIZE_M=16,
            BLOCK_TILE_SIZE_N=block_n,
            stage=stage,
            alg="splitk",
            E=5,
        )
        for n, k, stage in ((2 * inter, hidden, "gateup"), (hidden, inter, "down"))
    ]
    args1 = (
        request.hidden_states,
        request.w1,
        mid,
        si,
        sw,
        se,
        valid,
        request.w1_scale,
        17,
        se.numel(),
    )
    args2 = (
        ref_mid,
        request.w2,
        output,
        si,
        sw,
        se,
        valid,
        request.w2_scale,
        17,
        se.numel(),
    )
    backend._launch(kernels[0], *args1)
    backend._launch(kernels[1], *args2)
    check_result(ref_mid, mid, "A16W8 Gate/Up")
    check_result(reference, output, "A16W8 Down with reference input")
    saved = valid.clone()
    valid.zero_()
    mid.fill_(13)
    output.fill_(13)
    backend._launch(kernels[0], *args1)
    backend._launch(kernels[1], *args2)
    assert torch.all(mid == 13) and torch.all(output == 13)
    valid.copy_(saved)
    output.zero_()
    backend._launch(kernels[0], *args1)
    backend._launch(kernels[1], *args2)
    check_result(ref_mid, mid, "A16W8 restored Gate/Up")
    check_result(reference, output, "A16W8 restored Down")


@pytest.mark.parametrize("tokens,direct", [(1, True), (4, True), (17, False)])
def test_weight_only_is_not_a8w8(tokens, direct):
    request, _ = prepare_case(tokens, 512, 256, quant_dtype=_FP8_DTYPE)
    x = torch.zeros_like(request.hidden_states)
    x[:, 0], x[:, 1] = 1.0, 1.0078125
    w1 = torch.zeros(4, 512, 512, dtype=torch.bfloat16, device=x.device)
    w1[:, :256, 0], w1[:, :256, 1], w1[:, 256:, 0] = 1.0, -1.0, 1.0
    w2 = torch.ones(4, 512, 256, dtype=torch.bfloat16, device=x.device)
    raw = (w1.to(_FP8_DTYPE), w2.to(_FP8_DTYPE))
    request = replace(
        request,
        hidden_states=x,
        w1=shuffle_weight(raw[0], (16, 16)),
        w2=shuffle_weight(raw[1], (16, 16)),
        topk_weight=torch.ones_like(request.topk_weight),
        w1_scale=torch.ones_like(request.w1_scale),
        w2_scale=torch.ones_like(request.w2_scale),
    )
    reference = torch_stages_a16w8(request, raw)[1]
    quantized_reference = run_torch(request, raw)
    assert torch.count_nonzero(quantized_reference) == 0
    assert reference.abs().min() > 1
    result = _run(request, _config(tokens, direct=direct))
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,direct", [(4, True), (17, False), (129, False)])
def test_stream_graph_replay(tokens, direct):
    request, raw = _case(tokens)
    cfg = _config(tokens, direct=direct)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(request, cfg)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            first = _run(request, cfg)
            second = _run(request, cfg)
    stream.synchronize()
    original = request.hidden_states.clone()
    for factor in (0.0, 1.0, -0.5):
        request.hidden_states.copy_(original * factor)
        request.topk_ids.copy_((request.topk_ids + 1) % 5)
        request.topk_weight.mul_(-0.5)
        request.w1_scale.mul_(0.75)
        request.w2_scale.mul_(1.25)
        reference = torch_stages_a16w8(request, raw)[1]
        first.fill_(float("nan"))
        second.fill_(float("nan"))
        graph.replay()
        check_result(reference, first, "A16W8 graph first")
        check_result(reference, second, "A16W8 graph second")


@pytest.mark.parametrize("tokens,direct", [(1, True), (4, True), (17, False)])
@pytest.mark.parametrize("alias", [QuantType.per_1x128, QuantType.per_128x128])
@pytest.mark.parametrize("activation_dtype", [torch.bfloat16, _FP8_DTYPE])
def test_public_dispatch_and_output_buffer(
    monkeypatch, tokens, direct, alias, activation_dtype
):
    fm = importlib.import_module("aiter.fused_moe")
    request, raw = _case(tokens)
    cfg = _config(tokens, direct=direct)
    key = (
        fm.get_gfx_runtime(),
        fm.get_cu_num(),
        fm.get_padded_M(tokens),
        512,
        256,
        5,
        3,
        str(ActivationType.Silu),
        str(torch.bfloat16),
        str(activation_dtype),
        str(request.w1.dtype),
        str(QuantType.per_1x128),
        True,
        False,
    )
    row = {
        "kernelName1": make_fused_moe_impl_kernel_name(
            f"flydsl_{get_gfx()}", cfg.to_string()
        ),
        "kernelName2": "",
        "block_m": 16,
        "ksplit": 0,
        "run_1stage": 0,
    }
    monkeypatch.setattr(fm, "cfg_2stages", ({key: row}, {}))
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "0")
    fm.get_2stage_cfgs.cache_clear()
    output = torch.full_like(request.hidden_states, float("nan"))
    try:
        result = fm.fused_moe(
            request.hidden_states,
            request.w1,
            request.w2,
            request.topk_weight,
            request.topk_ids,
            activation=ActivationType.Silu,
            quant_type=alias,
            w1_scale=request.w1_scale,
            w2_scale=request.w2_scale,
            quant_dtype_a=activation_dtype,
            output=output,
        )
        assert result is output
        check_result(torch_stages_a16w8(request, raw)[1], result, "public A16W8")
    finally:
        fm.get_2stage_cfgs.cache_clear()


@pytest.mark.parametrize(
    "change",
    [
        {"activation": ActivationType.Swiglu},
        {"swiglu_limit": 7.0},
        {"doweight_stage1": True},
        {"q_dtype_a": torch.float32},
        {"block_size_m": 32},
        {"ksplit": 2},
        {"quant_type": QuantType.per_Token},
    ],
)
def test_unsupported_requests_fail_before_kernel(change):
    request, _ = _case(17)
    with patch.object(backend, "_run_decode") as launch:
        with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
            _run(replace(request, **change), _config(17))
        launch.assert_not_called()


def test_unsupported_arch_rejects_decode(monkeypatch):
    request, _ = _case(17)
    monkeypatch.setattr(backend, "get_gfx", lambda: "gfx1250")
    with pytest.raises(
        NotImplementedError, match="Unsupported FlyDSL MoE architecture"
    ):
        _run(replace(request, q_dtype_a=None), _config(17))
