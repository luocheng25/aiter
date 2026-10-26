# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the gfx942 FlyDSL whole-MoE backend.

Its tuned rows use cu_num=80 (MI308X), so test_moe_2stage.py skips them on
runners with another CU count. This test runs every shipped config through the
registered backend on any gfx942 device, checks the compact Down task table
for both MI308X and MI300X CU counts, and checks the padded sorted reduction.
"""

import argparse
import functools
import itertools
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    AUX_SORT_OPUS,
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.fused_moe_registry import (
    FusedMoeRequest,
    make_fused_moe_impl_kernel_name,
    resolve_fused_moe_impl,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.fused_moe_gfx942 import Config, get_tune_space
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import (
    gemm2_8x1_compact as compact,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import sorted_sum
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

IMPL = "flydsl_gfx942"
QUANTS = {
    "no": QuantType.No,
    "per_tensor": QuantType.per_Tensor,
    "ptpc": QuantType.per_Token,
}
ACTIVATIONS = {"silu": ActivationType.Silu, "swiglu": ActivationType.Swiglu}
# MI300X then MI308X, the order in which a cached launcher meets a smaller device.
CU_COUNTS = (304, 80)
GUARD, CANARY = 64, -123
# FP8 rounding moves an element by at most 0.28 token RMS from the torch reference.
ELEMENT_ERR_LIMIT = 1.0
# FP8 rounding reaches 0.076 per token; one misrouted route gives about 0.67.
TOKEN_ERR_LIMIT = 0.2
LOGITS_DIFF_LIMIT = 1e-3
# Prefill paths of one case share Gate/Up and quantization and agree to BF16 rounding.
PATH_DIFF_LIMIT = 0.1


def guarded_task_buffers(sorted_expert_ids, experts, cu_count):
    """allocate_task_buffers with canaries before and after every buffer."""
    full, tail = compact.task_capacities(sorted_expert_ids.numel(), experts, cu_count)
    storages, buffers = [], []
    for shape in ((full, 2), (tail, 2), (2,)):
        storage = torch.full(
            (torch.Size(shape).numel() + 2 * GUARD,),
            CANARY,
            dtype=torch.int32,
            device=sorted_expert_ids.device,
        )
        storages.append(storage)
        buffers.append(storage[GUARD:-GUARD].view(shape))
    return storages, buffers


def check_guards(storages):
    for storage in storages:
        assert bool((storage[:GUARD] == CANARY).all()), "write before a buffer"
        assert bool((storage[-GUARD:] == CANARY).all()), "write after a buffer"


def check_tasks(blocks, experts, cu_count, full_tasks, tail_tasks, counts):
    """Check the split policy and that each block is covered once by its expert."""
    sizes = [0] * experts
    for expert in blocks:
        sizes[expert] += 1
    full = sum(size // 4 for size in sizes)
    tail = sum(size % 4 for size in sizes)
    remainder = full % cu_count
    split = 0
    if remainder and remainder * compact._SPLIT_DEN < cu_count * compact._SPLIT_NUM:
        split = remainder
    # The kernel caps the split at the free tail capacity.
    split = min(split, (tail_tasks.shape[0] - tail) // 4)
    expected = [(full - split) * 256, (tail + 4 * split) * 64]
    assert counts.tolist() == expected, f"counts {counts.tolist()} != {expected}"
    coverage = [0] * len(blocks)
    for tasks, rows, count in (
        (full_tasks, 256, full - split),
        (tail_tasks, 64, tail + 4 * split),
    ):
        assert count <= tasks.shape[0], "task count exceeds its buffer"
        for start, expert in tasks[:count].tolist():
            assert start % 64 == 0 and 0 <= start and start + rows <= 64 * len(blocks)
            for block in range(start // 64, (start + rows) // 64):
                assert blocks[block] == expert, "task routed to another expert"
                coverage[block] += 1
    assert coverage == [1] * len(blocks), "blocks not covered exactly once"
    return full - split, tail + 4 * split


def task_scenarios(cu_count):
    """Map a scenario to (experts, [(expert, M64 blocks)]) in metadata order."""
    # Largest remainder of the final full-task CU round that is still split.
    split = (cu_count * compact._SPLIT_NUM - 1) // compact._SPLIT_DEN
    return {
        "empty": (4, []),
        "tails_only": (8, [(1, 1), (3, 2), (6, 3)]),
        "single_expert": (1, [(0, 8)]),
        "full_round": (2, [(1, 4 * cu_count)]),
        "split_limit": (2, [(0, 4 * (cu_count + split)), (1, 3)]),
        "no_split": (2, [(0, 4 * (cu_count + split + 1)), (1, 3)]),
        # Every expert has one full task and three tails, filling the tail capacity.
        "tail_capacity": (split, [(expert, 7) for expert in range(split)]),
        # 60 full tasks split into tails with 304 CUs but not with 80.
        "cu_dependent": (1, [(0, 240)]),
        "experts_2048": (2048, [(expert, 1 + expert % 7) for expert in range(2048)]),
    }


# One launcher per expert count serves every CU count, like the cached builder.
build_task_table = functools.cache(compact.build_task_table)


@benchmark()
def test_task_table(scenario, cu_count, buffer_cu_count):
    experts, runs = task_scenarios(cu_count)[scenario]
    blocks = [expert for expert, size in runs for _ in range(size)]
    # Padding names a real expert, so reading past the valid blocks breaks a run.
    metadata = torch.tensor(blocks + [0] * 4, dtype=torch.int32)
    valid_rows = torch.tensor([64 * len(blocks)], dtype=torch.int32)
    storages, (full_tasks, tail_tasks, counts) = guarded_task_buffers(
        metadata, experts, buffer_cu_count
    )
    launcher = build_task_table(experts)
    pointers = [_ptr(t) for t in (metadata, valid_rows, full_tasks, tail_tasks, counts)]

    def build():
        _run_compiled(
            launcher,
            *pointers,
            cu_count,
            tail_tasks.shape[0],
            torch.cuda.current_stream(),
        )
        return counts

    _, us = run_perftest(build)
    full, tail = check_tasks(blocks, experts, cu_count, full_tasks, tail_tasks, counts)
    check_guards(storages)
    if scenario == "tail_capacity" and buffer_cu_count == cu_count:
        assert tail == tail_tasks.shape[0], "tail capacity is not tight"
    nbytes = metadata.nbytes + valid_rows.nbytes + 8 * (full + tail) + counts.nbytes
    return {
        "blocks": len(blocks),
        "full tasks": full,
        "full capacity": full_tasks.shape[0],
        "tail tasks": tail,
        "tail capacity": tail_tasks.shape[0],
        "us": us,
        "TB/s": nbytes / us / 1e6,
    }


@benchmark()
def test_sorted_sum(token, model_dim, topk, padding):
    dtype = dtypes.bf16
    experts = max(4, topk)
    ids = (torch.arange(token * topk).reshape(token, topk) % experts).to(torch.int32)
    weights = torch.ones((token, topk), dtype=torch.float32)
    storage = torch.full((token * model_dim + 2 * GUARD,), CANARY, dtype=dtype)
    _, _, sorted_experts, _, output, _, reverse_sorted = moe_sorting(
        ids,
        weights,
        experts,
        model_dim,
        dtype,
        64,
        output_aux=AUX_SORT_OPUS,
        output=storage[GUARD:-GUARD].view(token, model_dim),
    )
    loc_ids = reverse_sorted.view_as(ids)
    # Specialized Down rows carry padding that holds canaries the sum must skip.
    down_output = torch.full(
        (sorted_experts.numel() * 64, model_dim + padding // 2), CANARY, dtype=dtype
    )
    routes = torch.randn((token, topk, model_dim)).to(dtype)
    down_output[loc_ids.long(), :model_dim] = routes
    loc_ids[::3, -1] = -1
    loc_ids[::7, :] = -1
    routes = torch.where((loc_ids >= 0).unsqueeze(-1), routes, 0)
    reference = routes[:, 0].float()
    for route in range(1, topk):
        reference = reference + routes[:, route].float()
    reduce = sorted_sum(topk, model_dim, padding)

    def run():
        reduce(loc_ids, down_output, output, token)
        return output

    actual, us = run_perftest(run)
    err = checkAllclose(
        reference.to(dtype).float(),
        actual.float(),
        rtol=0,
        atol=0,
        msg=f"sorted_sum padding={padding}",
    )
    assert err == 0, "padded sorted reduction is not exact"
    check_guards([storage])
    flops = token * model_dim * (topk - 1)
    nbytes = (int((loc_ids >= 0).sum()) + token) * model_dim * output.element_size()
    nbytes += loc_ids.nbytes
    return {
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def test_config_strings():
    for encoded in get_tune_space():
        assert Config.from_string(encoded).to_string() == encoded
    # Rows tuned before the Down suffix existed keep loading as the default Down.
    for legacy in ("16_16_16_False", "64_128_128_True", "64_256_128_True"):
        assert Config.from_string(legacy) == Config.from_string(f"{legacy}:default:0")
    # Tiles without the direct-decoding flag keep their meaning.
    assert Config.from_string("16_16_16_False_False:default:0") == Config.from_string(
        "16_16_16_False:default:0"
    )
    for invalid in (
        "64_256_128_True:default",
        "64_256_128_True:default:0:0",
        "64_256_128_Maybe",
        "64_256_True",
        "64_256_128_True:default:128",
        "64_256_128_True:8x2:128",
        "16_16_16_False:8x1_compact:128",
        "16_16_16_False_Maybe:default:0",
        "16_16_16_False_True_False:default:0",
        "16_16_16_True_True:default:0",
        "16_16_16_False_True:1x4_64x256:128",
    ):
        try:
            Config.from_string(invalid)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {invalid!r}")


def route(hidden, experts, topk, routing):
    token = hidden.shape[0]
    if routing == "random":
        return fused_topk(
            hidden, torch.randn((token, experts), dtype=hidden.dtype), topk, True
        )
    # "hot": every token picks the same highest experts and the others stay empty.
    ids = torch.arange(experts - topk, experts, dtype=torch.int32)
    weights = torch.rand((token, topk), dtype=torch.float32)
    return (
        weights / weights.sum(dim=-1, keepdim=True),
        ids.expand(token, topk).contiguous(),
    )


@functools.lru_cache(maxsize=1)
def make_case(
    token,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    activation,
    routing,
    quantize_activations,
):
    """Inputs and torch reference as test_moe_2stage.py builds them."""
    torch.manual_seed(0)
    q_type, act_type = QUANTS[quant], ACTIVATIONS[activation]
    q_dtype = None if q_type == QuantType.No else dtypes.fp8
    torch_quant = aiter.get_torch_quant(q_type)

    def quantize_activation(x):
        # Decode and batch-1 kernels read BF16 activations even with FP8 weights.
        if quantize_activations:
            return torch_quant(x, quant_dtype=q_dtype)
        return x, torch.ones(1, dtype=torch.float32)

    hidden = torch.randn((token, model_dim), dtype=dtypes.bf16)
    w1 = torch.randn((experts, 2 * inter_dim, model_dim), dtype=dtypes.bf16)
    w2 = torch.randn((experts, model_dim, inter_dim), dtype=dtypes.bf16)
    topk_weights, topk_ids = route(hidden, experts, topk, routing)
    if q_type == QuantType.per_Tensor:
        w1_q, w1_scale = aiter.pertoken_quant(w1.view(experts, -1), quant_dtype=q_dtype)
        w2_q, w2_scale = aiter.pertoken_quant(w2.view(experts, -1), quant_dtype=q_dtype)
        w1_q, w2_q = w1_q.view(w1.shape), w2_q.view(w2.shape)
    else:
        w1_q, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
        w2_q, w2_scale = torch_quant(w2, quant_dtype=q_dtype)
    a1, a1_scale = quantize_activation(hidden)
    stage1 = torch_moe_stage1(
        a1,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        activation=act_type,
        quant_type=q_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
    )
    a2, a2_scale = quantize_activation(stage1)
    reference = torch_moe_stage2(
        a2.view(token, topk, -1),
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=q_type,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
    )
    request = FusedMoeRequest(
        hidden_states=hidden,
        w1=shuffle_weight(w1_q, layout=(16, 16)),
        w2=shuffle_weight(w2_q, layout=(16, 16)),
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=act_type,
        quant_type=q_type,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    # Holds the first FlyDSL output of the case for the cross-path check.
    return request, reference, {}


class GuardedTaskBuffers:
    """Replaces allocate_task_buffers and keeps the last allocation for checks."""

    def __call__(self, sorted_expert_ids, experts, cu_count):
        self.metadata, self.cu_count = sorted_expert_ids, cu_count
        self.storages, self.buffers = guarded_task_buffers(
            sorted_expert_ids, experts, cu_count
        )
        return self.buffers


def token_rms(x):
    return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def token_error(expected, actual):
    """Largest per-token relative L2 error."""
    x, y = expected.double(), actual.double()
    return float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max())


def check_output(reference, output, msg):
    """Assert every accuracy check; returns the checkAllclose mismatch ratio."""
    assert bool(torch.isfinite(output).all()), f"{msg}: non-finite output"
    # Errors in units of each token's reference RMS, so one tolerance fits all shapes.
    rms = token_rms(reference)
    err = checkAllclose(
        reference.float() / rms,
        output.float() / rms,
        rtol=0,
        atol=ELEMENT_ERR_LIMIT,
        msg=msg,
    )
    x, y = reference.double(), output.double()
    logits_diff = float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())
    token_err = token_error(reference, output)
    assert err == 0, f"{msg}: {err:.3%} of elements beyond {ELEMENT_ERR_LIMIT} RMS"
    assert logits_diff < LOGITS_DIFF_LIMIT, f"{msg}: logits_diff={logits_diff}"
    assert token_err < TOKEN_ERR_LIMIT, f"{msg}: worst token error {token_err}"
    return err


def run_compact(impl, request, experts, cu_count):
    """Run with guarded task buffers and check the task table built for cu_count."""
    buffers = GuardedTaskBuffers()
    with patch.object(compact, "allocate_task_buffers", buffers):
        output = impl(request)
    assert buffers.cu_count == cu_count
    sizes = torch.bincount(request.topk_ids.flatten().long(), minlength=experts)
    blocks = buffers.metadata[: int(((sizes + 63) // 64).sum())].tolist()
    full, tail = check_tasks(blocks, experts, cu_count, *buffers.buffers)
    check_guards(buffers.storages)
    return output, f"{full}/{tail}"


def rms_error(expected, actual, rms):
    """Largest elementwise difference in units of the token's reference RMS."""
    return float(((actual.float() - expected.float()).abs() / rms).max())


def check_moe(
    token, model_dim, inter_dim, experts, topk, quant, activation, routing, config
):
    """Run one config through the registered backend and assert its accuracy."""
    parsed = Config.from_string(config)
    request, reference, first_output = make_case(
        token,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        activation,
        routing,
        parsed.use_prefill,
    )
    impl = resolve_fused_moe_impl(make_fused_moe_impl_kernel_name(IMPL, config))
    rms = token_rms(reference)
    tasks = "-"
    if parsed.down_path == "8x1_compact":
        device_cu = get_cu_num()
        output, tasks = run_compact(impl, request, experts, device_cu)
        # The cached launcher must stay in bounds and correct for the other CU count.
        for cu_count in CU_COUNTS:
            if cu_count != device_cu:
                with patch(
                    "aiter.ops.flydsl.fused_moe_gfx942.get_cu_num",
                    return_value=cu_count,
                ):
                    other, _ = run_compact(impl, request, experts, cu_count)
                diff = rms_error(output, other, rms)
                assert diff < PATH_DIFF_LIMIT, f"{config} at {cu_count} CUs: {diff=}"
    else:
        output = impl(request)
    err = check_output(reference, output, config)
    # Decode kernels accumulate routes with atomics, so only prefill paths are compared.
    if parsed.use_prefill:
        first = first_output.setdefault("output", output.clone())
        path_diff = rms_error(first, output, rms)
        assert (
            path_diff < PATH_DIFF_LIMIT
        ), f"{config}: {path_diff=} from the first path"
    return impl, request, tasks, err


@benchmark()
def test_packed_sum(token, model_dim, topk):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.gemm2_4x1 import (
        make_packed_sum,
    )

    experts = max(4, topk)
    ids = (torch.arange(token * topk).reshape(token, topk) % experts).to(torch.int32)
    # Power-of-two route factors keep the FP32 sums exact.
    weights = 2.0 ** torch.randint(-2, 3, (token, topk)).float()
    scales = 2.0 ** torch.randint(-3, 4, (token * topk,)).float()
    _, sorted_weights, sorted_experts, _, _, _, reverse_sorted = moe_sorting(
        ids, weights, experts, model_dim, dtypes.bf16, 256, output_aux=AUX_SORT_OPUS
    )
    locations = reverse_sorted.view_as(ids)
    source_rows = sorted_experts.numel() * 256
    packed = torch.full(
        (source_rows // 128, model_dim // 64, 128, 64), CANARY, dtype=dtypes.bf16
    )
    routes = torch.randn((token, topk, model_dim)).to(dtypes.bf16)
    positions = locations.flatten().long()
    packed[positions // 128, :, positions % 128, :] = routes.reshape(
        -1, model_dim // 64, 64
    )
    locations[::3, -1] = -1
    locations[::5, 0] = source_rows
    valid = (locations >= 0) & (locations < source_rows)
    routes = torch.where(valid.unsqueeze(-1), routes, 0)
    factors = weights * scales.view(token, topk)
    reference = routes[:, 0].float() * factors[:, :1]
    for route in range(1, topk):
        reference = reference + routes[:, route].float() * factors[:, route : route + 1]
    reference = reference.to(dtypes.bf16).float()
    storage = torch.full((token * model_dim + 2 * GUARD,), CANARY, dtype=dtypes.bf16)
    output = storage[GUARD:-GUARD].view(token, model_dim)
    launcher = make_packed_sum(N=model_dim, TOPK=topk)
    pointers = [
        _ptr(tensor) for tensor in (output, packed, locations, sorted_weights, scales)
    ]

    def run():
        _run_compiled(
            launcher, *pointers, token, source_rows, torch.cuda.current_stream()
        )
        return output

    candidates = {"packed": run}
    flops = token * model_dim * (topk - 1)
    nbytes = int(valid.sum()) * model_dim * routes.element_size()
    nbytes += output.nbytes + locations.nbytes
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        actual, us = run_perftest(candidate, num_iters=10, num_warmup=2)
        err = checkAllclose(reference, actual.float(), rtol=0, atol=0, msg=name)
        assert err == 0, "packed reduction is not exact"
        check_guards([storage])
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_sort_inverse(token, experts, topk, routing, block_size):
    ids = (torch.arange(token * topk).reshape(token, topk) % experts).to(torch.int32)
    if routing == "duplicate":
        ids[:, -1] = ids[:, 0]
    weights = torch.rand((token, topk), dtype=torch.float32)
    expert_mask = None
    num_local_tokens = None
    if routing == "masked":
        expert_mask = (torch.arange(experts) % 2 == 0).to(torch.int32)
        num_local_tokens = torch.tensor([max(token - 3, 0)], dtype=torch.int32)
    elif routing == "empty":
        expert_mask = torch.zeros(experts, dtype=torch.int32)
    candidates = {
        "sort_inverse": lambda: moe_sorting(
            ids, weights, experts, 2560, dtypes.bf16, block_size,
            expert_mask, num_local_tokens,
            accumulate=False, output_aux=AUX_SORT_OPUS,
        )
    }
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        result, us = run_perftest(candidate, num_iters=10, num_warmup=2)
        sorted_ids, _, _, valid_rows, _, _, inverse = result
        positions = torch.arange(int(valid_rows[0]), dtype=torch.int64)
        encoded = sorted_ids[:positions.numel()].to(torch.int64)
        tokens = encoded & 0xFFFFFF
        slots = encoded >> 24
        valid = (tokens < token) & (slots < topk)
        reference = torch.full((token * topk,), -1, dtype=torch.int32)
        reference[tokens[valid] * topk + slots[valid]] = positions[valid].to(torch.int32)
        err = checkAllclose(reference.float(), inverse.float(), rtol=0, atol=0, msg=name)
        assert torch.equal(reference, inverse), "sorting inverse differs from sorted positions"
        nbytes = ids.nbytes + weights.nbytes + sum(tensor.nbytes for tensor in result)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 0.0
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_moe(token, model_dim, inter_dim, experts, topk, quant, activation, config):
    impl, request, tasks, err = check_moe(
        token, model_dim, inter_dim, experts, topk, quant, activation, "random", config
    )
    _, us = run_perftest(impl, request, num_iters=10, num_warmup=2)
    flops = 6 * token * topk * model_dim * inter_dim
    # Input and output activations plus both weight tensors.
    nbytes = 2 * request.hidden_states.nbytes + request.w1.nbytes + request.w2.nbytes
    return {
        "full/tail tasks": tasks,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def shipped_rows():
    """Every shipped FlyDSL row as test_moe arguments, grouped by shape."""
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    df = df[df["kernelName1"].astype(str).str.startswith(f"impl__{IMPL}__")]
    if "_tag" in df:
        df = df[df["_tag"].fillna("").astype(str).str.strip() == ""]
    quants = {"No": "no", "per_Tensor": "per_tensor", "per_Token": "ptpc"}
    rows = set()
    for _, row in df.iterrows():
        shape = (
            int(row["model_dim"]),
            int(row["inter_dim"]),
            int(row["expert"]),
            int(row["topk"]),
            quants[str(row["q_type"]).split(".")[-1]],
            str(row["act_type"]).split(".")[-1].lower(),
        )
        rows.add((shape, int(row["token"]), str(row["kernelName1"]).split("__", 2)[2]))
    return [(token, *shape, config) for shape, token, config in sorted(rows)]


def cross_config_cases(rows):
    """All shipped prefill configs of a shape on the same inputs, so outputs compare."""
    configs = {}
    for _, *shape, config in rows:
        if Config.from_string(config).use_prefill:
            configs.setdefault(tuple(shape), set()).add(config)
    return [
        (16384, *shape, "random", config)
        for shape, shape_configs in sorted(configs.items())
        if len(shape_configs) > 1
        for config in sorted(shape_configs)
    ]


# Functional checks only, not timed:
# (token, model_dim, inter_dim, experts, topk, quant, activation, routing, config)
EDGE_CASES = [
    # The 2048-expert limit of the compact task table, with all or 8 experts active.
    *(
        (
            4096,
            512,
            256,
            2048,
            8,
            "ptpc",
            "silu",
            routing,
            f"64_256_128_True:{path}:128",
        )
        for routing in ("random", "hot")
        for path in ("8x1_compact", "1x4_64x256")
    ),
    # One expert with 64 full tasks: kept on 80 CUs, split into tails on 304.
    *(
        (16384, 1024, 384, 1, 1, "per_tensor", "silu", "hot", config)
        for config in (
            "64_256_128_True:8x1_compact:128",
            "64_256_128_True:1x4_64x256:128",
            "64_256_128_True:default:0",
        )
    ),
    # Every row padding of the specialized Down paths, reduced with the same stride.
    *(
        (
            16384,
            2048,
            512,
            257,
            9,
            "ptpc",
            "silu",
            "random",
            f"64_256_128_True:{path}:{padding}",
        )
        for path in ("8x1_compact", "1x4_64x256")
        for padding in (0, 32, 64)
    ),
]


# gfx950 four-wave 256x256 Gate/Up and M128 Down: tiny batches leave persistent
# CTAs idle, inter_dim 192/320 has a partial last N tile, hot routing puts every
# row on one expert, and the persistent M128 runs twice to check its queue reset.
FOUR_WAVE_CASES = [
    *(
        (token, 2048, 128, 257, 9, "ptpc", "silu", "random", f"256_256_128_True:{path}:0")
        for token in (1, 7, 300, 16384)
        for path in ("m128", "m128p", "m128p")
    ),
    (4096, 6144, 256, 384, 8, "ptpc", "silu", "hot", "256_256_128_True:m128p:0"),
    # M128 Down above K256 streams K128 sub-packets: merged halves and OC splits.
    (300, 6144, 384, 128, 4, "ptpc", "swiglu", "random", "256_256_128_True:m128:0"),
    (16384, 6144, 384, 129, 5, "ptpc", "swiglu", "random", "256_256_128_True:m128p:0"),
    (2048, 2048, 512, 257, 9, "ptpc", "silu", "random", "256_256_128_True:m128:0"),
    (4097, 2048, 512, 257, 9, "ptpc", "silu", "random", "256_256_128_True:m128p:0"),
    (7, 2560, 640, 513, 11, "ptpc", "silu", "random", "256_256_128_True:m128:0"),
    (8192, 2560, 640, 513, 11, "ptpc", "silu", "random", "256_256_128_True:m128p:0"),
    # inter_dim % 128 == 64 pads the M128 Down's last K128 step with zero A.
    (4096, 2560, 192, 513, 11, "ptpc", "silu", "random", "256_256_128_True:m128:0"),
    (300, 2560, 320, 513, 11, "ptpc", "silu", "random", "256_256_128_True:m128p:0"),
    (4096, 6144, 384, 128, 4, "ptpc", "swiglu", "random", "256_256_128_True:8x1_compact:128"),
    (4096, 2560, 192, 513, 11, "ptpc", "silu", "random", "256_256_128_True:8x1_compact:128"),
    (4096, 2560, 320, 513, 11, "ptpc", "silu", "random", "256_256_128_True:1x4_64x256:128"),
    (4096, 4096, 192, 192, 8, "per_tensor", "silu", "random", "256_256_128_True:default:0"),
    (2048, 4096, 640, 1, 1, "ptpc", "silu", "hot", "256_256_128_True:1x4_64x256:128"),
    # Half-height tiles: bottom halves with routes (hot, MiniMax 4k) run before the
    # top-half schedule; the rest checks tiny batches, N tails and M64 Down re-sorts.
    (7, 2048, 128, 257, 9, "ptpc", "silu", "random", "128_256_128_True:m128:0"),
    (4096, 6144, 256, 384, 8, "ptpc", "silu", "hot", "128_256_128_True:m128p:0"),
    (4096, 6144, 384, 128, 4, "ptpc", "swiglu", "random", "128_256_128_True:m128:0"),
    (2048, 2560, 192, 513, 11, "ptpc", "silu", "random", "128_256_128_True:m128p:0"),
    (300, 2560, 320, 513, 11, "ptpc", "silu", "random", "128_256_128_True:1x4_64x256:128"),
    (4096, 4096, 192, 192, 8, "per_tensor", "silu", "random", "128_256_128_True:default:0"),
]

# gfx950 M64 Gate/Up walking the M128 Down's 256-row sort blocks: small batches
# leave most 64-row tiles empty, hot routing fills every tile, and the token
# counts at or below E / topk take the launcher's route-count bound.
M64_M128_CASES = [
    (1, 2048, 128, 257, 9, "ptpc", "silu", "random", "64_128_128_True:m128:0"),
    (7, 2048, 128, 257, 9, "ptpc", "silu", "random", "64_256_128_True:m128p:0"),
    (4096, 2048, 128, 257, 9, "ptpc", "silu", "random", "64_128_128_True:m128p:0"),
    (16384, 2048, 128, 257, 9, "ptpc", "silu", "random", "64_256_128_True:m128p:0"),
    (300, 6144, 256, 384, 8, "ptpc", "silu", "random", "64_128_256_True:m128:0"),
    (4096, 6144, 256, 384, 8, "ptpc", "silu", "hot", "64_256_128_True:m128p:0"),
    (1, 6144, 384, 128, 4, "ptpc", "swiglu", "random", "64_256_128_True:m128:0"),
    (300, 6144, 384, 128, 4, "ptpc", "swiglu", "random", "64_128_128_True:m128:0"),
    (2048, 2048, 512, 257, 9, "ptpc", "silu", "random", "64_256_128_True:m128p:0"),
    (7, 2560, 640, 513, 11, "ptpc", "silu", "random", "64_128_256_True:m128:0"),
]


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() == "gfx950":
        # Only the gfx950 Gate/Up and Down paths; shipped rows are gfx942 tunes.
        test_config_strings()
        summarize(
            "Packed reduction",
            [
                test_packed_sum(token, model_dim, topk)
                for token, (model_dim, topk) in itertools.product(
                    (65,), ((512, 1), (2048, 9), (2560, 11), (4608, 4), (6144, 4))
                )
            ],
        )
        summarize(
            "Sorting inverse",
            [
                test_sort_inverse(token, 7, 3, routing, block_size)
                for token, routing, block_size in itertools.product(
                    (1, 65), ("random", "duplicate", "masked", "empty"), (64, 256)
                )
            ],
        )
        summarize(
            "Masked sorted reduction",
            [
                test_sorted_sum(token, model_dim, topk, padding)
                for token, (model_dim, topk), padding in itertools.product(
                    (1, 65), ((256, 1), (2560, 11), (6144, 4)), (0, 128)
                )
            ],
        )
        for case in FOUR_WAVE_CASES + M64_M128_CASES:
            check_moe(*case)
        aiter.logger.info(
            "gfx950 cases passed: %d four-wave, %d M64 Gate/Up + M128 Down",
            len(FOUR_WAVE_CASES),
            len(M64_M128_CASES),
        )
        return
    if get_gfx() != "gfx942":
        aiter.logger.warning(
            "gfx942 FlyDSL whole-MoE backend unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=None,
        help="Only time the shipped rows with these token counts (default: every row).",
    )
    args = parser.parse_args()

    test_config_strings()
    summarize(
        "Compact task table",
        [
            test_task_table(scenario, cu_count, cu_count)
            for cu_count in CU_COUNTS
            for scenario in task_scenarios(cu_count)
        ]
        # Buffers sized for 80 CUs under a 304-CU split: the device guard caps it.
        + [test_task_table("cu_dependent", *CU_COUNTS)],
    )
    summarize(
        "Padded sorted reduction",
        [
            test_sorted_sum(token, model_dim, topk, padding)
            for token, (model_dim, topk), padding in itertools.product(
                (65, 2048),
                ((2048, 9), (2560, 11), (4096, 8), (6144, 4)),
                (0, 32, 64, 128),
            )
        ],
    )
    rows = shipped_rows()
    for case in cross_config_cases(rows) + EDGE_CASES:
        check_moe(*case)
    timed = [row for row in rows if not args.batch or row[0] in args.batch]
    summarize("Whole MoE", [test_moe(*row) for row in timed])


if __name__ == "__main__":
    main()
