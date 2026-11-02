"""Flash-specific native MXFP4 eager/graph checks; run only after prefill.

Run in separate processes with AITER_CONFIG_FMOE selecting the pure A16W4,
pure A8W4, and mixed Flash tables. This verifies the selected kernel names
and compares against a BF16 reference using the same quantized weights.
"""

import csv
import functools
import json
import os

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


def describe(stage):
    if isinstance(stage, functools.partial):
        return {"function": stage.func.__module__ + "." + stage.func.__name__,
                "kwargs": {k: str(v) for k, v in stage.keywords.items()}}
    return {"function": stage.__module__ + "." + stage.__name__}


def cosine_error(reference, actual):
    x, y = reference.double(), actual.double()
    return float((1 - 2 * (x * y).sum() / (x.square() + y.square()).sum()).item())


@pytest.mark.parametrize("inter_dim", [256, 512])
@pytest.mark.parametrize("tokens", [1, 128, 384, 512, 672, 1024, 2880])
def test_flash_native_eager_and_graph(inter_dim, tokens):
    assert get_gfx() == "gfx942"
    assert torch.cuda.get_device_properties(0).multi_processor_count == 304
    model_dim, experts, topk = 4096, 256, 8
    torch.manual_seed(12345)
    dtype = dtypes.bf16
    inp = torch.randn(tokens, model_dim, dtype=dtype, device="cuda")
    w1 = torch.randn(experts, 2 * inter_dim, model_dim, dtype=dtype, device="cuda")
    w2 = torch.randn(experts, model_dim, inter_dim, dtype=dtype, device="cuda")
    scores = torch.randn(tokens, experts, dtype=dtype, device="cuda")
    weights, ids = fused_topk(inp, scores, topk, True)
    quantize = aiter.get_torch_quant(QuantType.per_1x32)
    w1q, s1 = quantize(w1, quant_dtype=dtypes.fp4x2)
    w2q, s2 = quantize(w2, quant_dtype=dtypes.fp4x2)
    w1q = w1q.view(dtypes.fp4x2).view(experts, 2 * inter_dim, model_dim // 2)
    w2q = w2q.view(dtypes.fp4x2).view(experts, model_dim, inter_dim // 2)
    middle = torch_moe_stage1(
        inp, w1q, w2q, weights, ids, dtype=dtype,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        a1_scale=None, w1_scale=s1.view(experts, 2 * inter_dim, model_dim // 32), doweight=False)
    reference = torch_moe_stage2(
        middle.view(tokens, topk, inter_dim), w1q, w2q, weights, ids,
        dtype=dtype, quant_type=QuantType.per_1x32,
        w2_scale=s2.view(experts, model_dim, inter_dim // 32), a2_scale=None, doweight=True)
    packed1 = shuffle_weight_a16w4(w1q, 16, False)
    packed2 = shuffle_weight_a16w4(w2q, 16, False)
    scale1 = shuffle_scale_a16w4(s1, experts, False)
    scale2 = shuffle_scale_a16w4(s2, experts, False)

    def forward():
        return fused_moe(inp, packed1, packed2, weights, ids, w1_scale=scale1,
                         w2_scale=scale2, quant_type=QuantType.per_1x32,
                         activation=ActivationType.Silu, doweight_stage1=False,
                         gate_mode=GateMode.SEPARATED.value)

    output = forward()
    torch.cuda.synchronize()
    metadata = get_2stage_cfgs(
        get_padded_M(tokens), model_dim, inter_dim, experts, topk,
        dtype, dtype, dtypes.fp4x2, QuantType.per_1x32, True,
        ActivationType.Silu, False, 0, 0, is_shuffled=True,
        gate_mode=GateMode.SEPARATED.value)
    description = {"stage1": describe(metadata.stage1), "stage2": describe(metadata.stage2)}
    with open(os.environ["AITER_CONFIG_FMOE"]) as stream:
        candidates = [r for r in csv.DictReader(stream)
                      if int(r["model_dim"]) == model_dim and int(r["inter_dim"]) == inter_dim
                      and int(r["expert"]) == experts and int(r["token"]) == get_padded_M(tokens)]
    assert len(candidates) == 1
    text = json.dumps(description)
    assert candidates[0]["kernelName1"] in text
    assert candidates[0]["kernelName2"] in text
    assert torch.isfinite(output).all().item()
    eager_error = cosine_error(reference, output)
    assert eager_error < 1e-2, (tokens, inter_dim, eager_error, description)
    # Capture after warmup; two replays also check atomic-output initialization.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = forward()
    errors = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(graph_output).all().item()
        error = cosine_error(reference, graph_output)
        assert error < 1e-2, (tokens, inter_dim, error, description)
        errors.append(error)
    print(json.dumps({"tokens": tokens, "inter_dim": inter_dim,
                      "route": "A8W4" if "afp8r" in candidates[0]["kernelName1"] else "A16W4",
                      "eager_error": eager_error, "graph_errors": errors, **description}))
