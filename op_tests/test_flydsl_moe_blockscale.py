# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 W128x128/A1x128 MoE: fixed whole-graph candidates and Torch reference."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import torch_moe_stage1, torch_moe_stage2
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.fused_moe_gfx942 import (
    Config,
    _Problem,
    run_flydsl_moe_gfx942_impl,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest


def prepare_case(
    tokens,
    hidden,
    inter,
    experts=4,
    topk=2,
    seed=23,
    routing="random",
    quant_dtype=torch.float8_e4m3fn,
):
    torch.manual_seed(seed)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") / 10

    def quant_weight(n, k):
        w = torch.randn(experts, n, k, dtype=torch.bfloat16, device=x.device) / 10
        blocks = w.view(experts, n // 128, 128, k // 128, 128).permute(0, 1, 3, 2, 4)
        q, scale = aiter.pertoken_quant(
            blocks.reshape(-1, 128 * 128), quant_dtype=quant_dtype
        )
        q = q.view(experts, n // 128, k // 128, 128, 128).permute(0, 1, 3, 2, 4)
        return q.contiguous().view_as(w), scale.view(experts, n // 128, k // 128)

    w1, s1 = quant_weight(2 * inter, hidden)
    w2, s2 = quant_weight(hidden, inter)
    if routing == "skewed":
        ids = (
            torch.arange(topk, device=x.device).expand(tokens, topk).contiguous().int()
        )
    else:
        ids = (
            torch.rand(tokens, experts, device=x.device).topk(topk, dim=1).indices.int()
        )
    weights = torch.rand(tokens, topk, dtype=torch.float32, device=x.device)
    request = FusedMoeRequest(
        x,
        shuffle_weight(w1, (16, 16)),
        shuffle_weight(w2, (16, 16)),
        weights,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        w1_scale=s1,
        w2_scale=s2,
        q_dtype_a=w1.dtype,
        q_dtype_w=w1.dtype,
    )
    return request, (w1, w2)


def torch_stages(request, raw_weights):
    """Both activation quantizations and the BF16 Gate/Up intermediate are explicit."""
    w1, w2 = raw_weights
    quantize = aiter.get_torch_quant(QuantType.per_1x128)
    a, sa = quantize(request.hidden_states, quant_dtype=w1.dtype)
    mid = torch_moe_stage1(
        a,
        w1,
        w2,
        request.topk_weight,
        request.topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        a1_scale=sa,
        w1_scale=request.w1_scale,
    )
    down, sd = quantize(mid.view(-1, w2.shape[-1]), quant_dtype=w2.dtype)
    down = down.view_as(mid)
    result = torch_moe_stage2(
        down,
        w1,
        w2,
        request.topk_weight,
        request.topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x128,
        a2_scale=sd,
        w2_scale=request.w2_scale,
    )
    return a, sa, mid, down, sd, result


def run_torch(request, raw_weights):
    return torch_stages(request, raw_weights)[-1]


def torch_stages_a16w8(request, raw_weights):
    """Weight-only reference: unit A scales, no input/intermediate quantization."""
    w1, w2 = raw_weights
    tokens, hidden = request.hidden_states.shape
    topk, inter = request.topk_ids.shape[1], w2.shape[-1]
    mid = torch_moe_stage1(
        request.hidden_states,
        w1,
        w2,
        request.topk_weight,
        request.topk_ids,
        dtype=torch.bfloat16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x128,
        w1_scale=request.w1_scale,
        a1_scale=torch.ones(tokens, hidden // 128, device=request.hidden_states.device),
    )
    result = torch_moe_stage2(
        mid,
        w1,
        w2,
        request.topk_weight,
        request.topk_ids,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x128,
        w2_scale=request.w2_scale,
        a2_scale=torch.ones(tokens, topk, inter // 128, device=mid.device),
    )
    return mid, result


def check_result(reference, result, label):
    assert result.shape == reference.shape and result.dtype == reference.dtype
    assert torch.isfinite(result).all(), f"{label}: nonfinite output"
    ref, out = reference.float(), result.float()
    err = checkAllclose(ref, out, msg=label)
    assert err <= 0.05, f"{label}: err_ratio={err}"
    denominator = (ref.square().sum() + out.square().sum()).clamp_min(1e-30)
    diff = ((ref - out).square().sum() / denominator).item()
    assert diff < 0.01, f"{label}: normalized squared diff={diff}"
    return err


def block_configs(inter, hidden, tokens, experts, topk):
    problem = _Problem(tokens, experts, 2 * inter, hidden, hidden, inter, topk, "block")
    configs = [
        Config(256, 256, 128, True, down_path=path, num_oc_splits=splits)
        for path in ("block_m128", "block_m128_persistent")
        for splits in (1, 2, 4, 8)
    ]
    configs += [
        Config(
            256,
            256,
            128,
            True,
            down_path="block_8wave",
            down_block_n=n,
            num_oc_splits=splits,
        )
        for n in (32, 64, 128)
        for splits in (1, 2, 4, 8)
    ]
    return [config for config in configs if config.unsupported_reason(problem) is None]


@benchmark()
def test_blockscale_perf(tokens, hidden, inter, experts, topk):
    request, raw = prepare_case(tokens, hidden, inter, experts, topk)
    reference = run_torch(request, raw)
    candidates = {
        f"{c.down_path}_n{c.down_block_n}_oc{c.num_oc_splits}": (
            lambda config=c: run_flydsl_moe_gfx942_impl(request, config.to_string())
        )
        for c in block_configs(inter, hidden, tokens, experts, topk)
    }
    # Full pipeline: sort, two quantizations, GEMMs, counter reset and reduce.
    flops = 6 * tokens * topk * hidden * inter
    nbytes = (
        sum(
            t.nbytes
            for t in (
                request.hidden_states,
                request.w1,
                request.w2,
                request.w1_scale,
                request.w2_scale,
                request.topk_ids,
                request.topk_weight,
            )
        )
        + tokens * hidden * 2
    )
    ret = {"gfx": get_gfx()}
    for name, run in candidates.items():
        check_result(reference, run(), name)
        out, us = run_perftest(run, num_warmup=2, num_iters=10)
        assert us > 0, f"{name}: invalid timing {us}"
        err = check_result(reference, out, name)
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
            }
        )
    return ret


# This is a CLI performance sweep, not a fixture-driven pytest case.
test_blockscale_perf.__test__ = False


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning("FP8 blockscale FlyDSL requires gfx950; skipping")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--batch", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096])
    parser.add_argument("--inter", type=int, nargs="+", default=[256, 384, 512, 640])
    parser.add_argument("--experts", type=int, nargs="+", default=[16])
    parser.add_argument("--topk", type=int, nargs="+", default=[4])
    args = parser.parse_args()
    rows = [
        test_blockscale_perf(*shape)
        for shape in itertools.product(
            args.batch, args.hidden, args.inter, args.experts, args.topk
        )
    ]
    aiter.logger.info(
        "FP8 blockscale MoE (full pipeline):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
