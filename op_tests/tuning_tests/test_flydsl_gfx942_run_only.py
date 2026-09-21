# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cold-process, run-only coverage for whole-graph MoE AOT helpers."""

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch


def _precompile_in_fresh_process(config, inter_dim, weight_dtype, quant_type, batch):
    from aiter.ops.flydsl import fused_moe_gfx942 as backend

    with patch.object(torch.cuda, "is_available", return_value=False):
        backend.precompile_flydsl_moe(
            config_string=config,
            batch=batch,
            model_dim=512,
            inter_dim=inter_dim,
            experts=2,
            topk=2,
            weight_dtype=weight_dtype,
            quant_type=quant_type,
            activation="silu",
        )


def _run_config_in_fresh_process(config_file, expected_device, batch):
    import aiter.test_common as common
    from aiter.ops.flydsl import fused_moe_gfx942 as backend
    from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import gemm2_8x1_compact
    from aiter.test_common import checkAllclose
    from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import FmoeTuner

    frame = pd.read_csv(config_file)
    frame["token"] = batch
    keys = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]
    tuner = FmoeTuner("fmoeTuner", keys, [], "Whole-graph run-only regression")
    tuner.untunedf = frame
    options = tuner.parser.parse_args(["--mp", "1", "--warmup", "1", "--iters", "2"])
    original_compile = backend._get_compiled_kernel
    original_allocate = gemm2_8x1_compact.allocate_task_buffers
    launched_devices = []
    errors = []
    task_counts = []

    def allocate(*args, **kwargs):
        buffers = original_allocate(*args, **kwargs)
        task_counts.append(buffers[2])
        return buffers

    def compiled(*args, **kwargs):
        launched_devices.append(torch.cuda.current_device())
        assert kwargs["device"].index == expected_device
        return original_compile(*args, **kwargs)

    def measured(fn, *args, num_iters=2, num_warmup=1, **kwargs):
        # 在输入GPU分配完成后切到另一当前GPU，再验证完整生产入口。
        previous = torch.cuda.current_device()
        stream = torch.cuda.Stream(device=expected_device)
        stream.wait_stream(torch.cuda.current_stream(expected_device))
        with torch.cuda.stream(stream):
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(0 if expected_device != 0 else 1)
            ambient = torch.cuda.current_device()
            try:
                result = fn(*args, **kwargs)
                assert torch.cuda.current_device() == ambient
                assert result.device.index == expected_device
                stream.synchronize()
                return result, 1.0
            finally:
                torch.cuda.set_device(previous)

    def checked(output, reference, **kwargs):
        a, b = output.float(), reference.float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        delta = (a - b).square().sum()
        ref2 = b.square().sum().clamp_min(1e-30)
        rel_l2 = (delta / ref2).sqrt().item()
        logits_diff = (delta / (a.square().sum() + ref2)).item()
        assert logits_diff <= 0.01
        errors.append({"rel_l2": rel_l2, "logits_diff": logits_diff})
        return checkAllclose(output, reference, **kwargs)

    torch.cuda.set_device(expected_device)
    with (
        patch.object(backend, "_get_compiled_kernel", side_effect=compiled),
        patch.object(gemm2_8x1_compact, "allocate_task_buffers", side_effect=allocate),
        patch.object(common, "run_perftest", side_effect=measured),
        patch.object(common, "checkAllclose", side_effect=checked),
    ):
        results = tuner.run_config(options)
    assert all(row["status"] == "ok" for row in results), results
    assert launched_devices and set(launched_devices) == {expected_device}
    for counts in task_counts:
        assert (
            (counts > 0).all().item()
        ), "Compact full and tail tasks must both execute"
    print("WHOLE_GRAPH_RUN_ONLY_PASS", json.dumps(errors), flush=True)


class TestFlydslGfx942RunOnly(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "A GPU is required")
    def test_whole_graph_aot_cache_runs_in_a_fresh_process(self):
        from aiter.fused_moe import get_padded_M

        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":", 1)[0] != "gfx942":
            self.skipTest("FP8 FNUZ whole-graph coverage requires gfx942")
        # 满块数至少达到0.6*CU，避免compact把全部M256任务拆成M64。
        compact_batch = ((3 * properties.multi_processor_count + 9) // 10) * 256 + 64
        cases = (
            ("no", "16_16_16_False_True", 128, 2),
            ("ptpc", "16_16_16_False", 128, 16),
            ("no", "64_128_128_True", 128, 64),
            ("ptpc", "64_128_128_True", 128, 64),
            ("per_tensor", "64_128_128_True", 192, 64),
            ("per_tensor", "64_128_128_True:1x4_64x256:64:128", 192, 64),
            ("ptpc", "256_128_128_True:8x1:64:128", 256, 64),
            ("ptpc", "64_128_128_True:8x1_compact:64:128", 320, compact_batch),
        )
        for quant_type, config, inter_dim, batch in cases:
            with self.subTest(
                config=config, quant_type=quant_type
            ), tempfile.TemporaryDirectory() as directory:
                cache = Path(directory) / "cache"
                config_file = Path(directory) / "config.csv"
                weight_dtype = "bf16" if quant_type == "no" else "fp8"
                dtype = (
                    "torch.bfloat16"
                    if weight_dtype == "bf16"
                    else "torch.float8_e4m3fnuz"
                )
                row = {
                    "gfx": "gfx942",
                    "cu_num": properties.multi_processor_count,
                    "token": get_padded_M(batch),
                    "model_dim": 512,
                    "inter_dim": inter_dim,
                    "expert": 2,
                    "topk": 2,
                    "act_type": "ActivationType.Silu",
                    "dtype": "torch.bfloat16",
                    "q_dtype_a": dtype,
                    "q_dtype_w": dtype,
                    "q_type": "QuantType."
                    + {"no": "No", "ptpc": "per_Token", "per_tensor": "per_Tensor"}[
                        quant_type
                    ],
                    "use_g1u1": 1,
                    "doweight_stage1": 0,
                    "block_m": int(config.split("_")[0]),
                    "ksplit": 0,
                    "kernelName1": "impl__flydsl_gfx942__" + config,
                    "kernelName2": "",
                    "us": 1.0,
                }
                pd.DataFrame([row]).to_csv(config_file, index=False)
                env = dict(os.environ)
                env.pop("COMPILE_ONLY", None)
                env.update(
                    FLYDSL_RUNTIME_CACHE_DIR=str(cache),
                    FLYDSL_RUNTIME_ENABLE_CACHE="1",
                    FLYDSL_RUNTIME_RUN_ONLY="1",
                    FLYDSL_GPU_ARCH="gfx942",
                    CU_NUM=str(properties.multi_processor_count),
                    AITER_CONFIG_FMOE=str(config_file),
                )
                compile_code = (
                    "from op_tests.tuning_tests.test_flydsl_gfx942_run_only import "
                    "_precompile_in_fresh_process; "
                    f"_precompile_in_fresh_process({config!r}, {inter_dim}, {weight_dtype!r}, {quant_type!r}, {batch})"
                )
                compiled = subprocess.run(
                    [sys.executable, "-c", compile_code],
                    env={**env, "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_RUN_ONLY": "0"},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    compiled.returncode, 0, compiled.stdout + compiled.stderr
                )
                self.assertTrue(list(cache.rglob("*.pkl")))
                devices = (0, 1, 0) if torch.cuda.device_count() > 1 else (device,)
                code = (
                    "from op_tests.tuning_tests.test_flydsl_gfx942_run_only import "
                    "_run_config_in_fresh_process; "
                    f"[_run_config_in_fresh_process({str(config_file)!r}, target, {batch}) for target in {devices!r}]"
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    result.stdout.count("WHOLE_GRAPH_RUN_ONLY_PASS"), len(devices)
                )
                for target_device in devices:
                    print(
                        f"AOT_RUN_ONLY_CASE_PASS {config} {quant_type} B{batch} device={target_device}",
                        flush=True,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "A GPU is required")
    def test_compact_gpu_task_distributions_match_torch_reference(self):
        from aiter import ActivationType, QuantType
        from aiter.ops.flydsl import fused_moe_gfx942 as backend
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import gemm2_8x1_compact
        from aiter.ops.shuffle import shuffle_weight

        device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        if properties.gcnArchName.split(":", 1)[0] != "gfx942":
            self.skipTest("FP8 FNUZ compact numerical coverage requires gfx942")
        cu_count = properties.multi_processor_count
        native_full_batch = math.ceil(0.6 * cu_count / 2) * 256 + 64
        cases = (
            ("native_tails", 96, 4),
            ("native_full_and_tail", native_full_batch, 2),
            ("split_underfilled_full_wave", 320, 2),
            ("skewed_with_empty_expert", 700, 4),
        )

        def quantize(value):
            scale = value.float().abs().amax(-1, keepdim=True) / 240.0
            scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
            return (value.float() / scale).to(torch.float8_e4m3fnuz), scale

        for name, batch, experts in cases:
            with self.subTest(distribution=name), patch.dict(
                os.environ, {"CU_NUM": str(cu_count)}
            ):
                torch.manual_seed(17)
                dim, inter, topk = 512, 192, 2
                hidden = (
                    torch.randn((batch, dim), dtype=torch.bfloat16, device=device) / 10
                )
                w1, scale1 = quantize(
                    torch.randn((experts, 2 * inter, dim), device=device) / 10
                )
                w2, scale2 = quantize(
                    torch.randn((experts, dim, inter), device=device) / 10
                )
                token = torch.arange(batch, dtype=torch.int32, device=device)
                if name == "skewed_with_empty_expert":
                    ids = torch.stack((torch.zeros_like(token), 1 + token % 2), dim=1)
                else:
                    ids = torch.stack((token % experts, (token + 1) % experts), dim=1)
                routing = torch.rand((batch, topk), device=device)
                routing /= routing.sum(1, keepdim=True)
                xq, xs = quantize(hidden)
                x = xq.float() * xs
                stage1 = torch.empty(
                    (batch, topk, inter), dtype=torch.bfloat16, device=device
                )
                for expert in range(experts):
                    positions = (ids == expert).nonzero(as_tuple=True)
                    if positions[0].numel():
                        gu = x[positions[0]] @ (w1[expert].float() * scale1[expert]).T
                        gate, up = gu.chunk(2, dim=-1)
                        stage1[positions] = (
                            torch.nn.functional.silu(gate) * up
                        ).bfloat16()
                middle, middle_scale = quantize(stage1)
                stage2 = torch.zeros(
                    (batch, topk, dim), dtype=torch.bfloat16, device=device
                )
                for expert in range(experts):
                    positions = (ids == expert).nonzero(as_tuple=True)
                    if positions[0].numel():
                        values = middle.float()[positions] * middle_scale[positions]
                        down = values @ (w2[expert].float() * scale2[expert]).T
                        stage2[positions] = (
                            down * routing[positions].unsqueeze(-1)
                        ).bfloat16()
                reference = stage2.float().sum(1).bfloat16()
                w1_runtime = shuffle_weight(w1, (16, 16))
                w2_runtime = shuffle_weight(w2, (16, 16))
                w1_runtime.is_shuffled = w2_runtime.is_shuffled = True
                saved = []
                allocate = gemm2_8x1_compact.allocate_task_buffers

                def guarded_allocate(
                    sorted_experts,
                    expert_count,
                    allocate=allocate,
                    saved=saved,
                    **kwargs,
                ):
                    allocated = allocate(sorted_experts, expert_count, **kwargs)
                    buffers, storage = [], []
                    for tensor in allocated:
                        backing = torch.full(
                            (tensor.numel() + 64,),
                            -123,
                            dtype=torch.int32,
                            device=device,
                        )
                        buffers.append(backing[32:-32].view(tensor.shape))
                        storage.append(backing)
                    saved.append((sorted_experts, buffers, storage))
                    return tuple(buffers)

                with patch.object(
                    gemm2_8x1_compact,
                    "allocate_task_buffers",
                    side_effect=guarded_allocate,
                ):
                    result = backend.run_flydsl_moe_gfx942(
                        hidden,
                        w1_runtime,
                        w2_runtime,
                        routing,
                        ids,
                        ActivationType.Silu,
                        QuantType.per_Token,
                        scale1,
                        scale2,
                        None,
                        None,
                        0,
                        "64_128_128_True:8x1_compact:64:128",
                    )
                torch.cuda.synchronize(device)
                self.assertTrue(torch.isfinite(result).all())
                delta = (result.float() - reference.float()).square().sum()
                ref2 = reference.float().square().sum().clamp_min(1e-30)
                logits_diff = (delta / (ref2 + result.float().square().sum())).item()
                self.assertLessEqual(logits_diff, 0.01)
                counts = (
                    torch.bincount(ids.flatten().long(), minlength=experts)
                    .cpu()
                    .tolist()
                )
                blocks = [(count + 63) // 64 for count in counts]
                full = sum(count // 4 for count in blocks)
                tail = sum(count % 4 for count in blocks)
                remainder = full % cu_count
                split = remainder if 0 < remainder * 5 < cu_count * 3 else 0
                expected = ((full - split) * 256, (tail + 4 * split) * 64)
                metadata, buffers, guards = saved[0]
                self.assertEqual(tuple(buffers[2].cpu().tolist()), expected)
                covered = []
                metadata_cpu = metadata.cpu().tolist()
                for tasks, rows, width in (
                    (buffers[0], expected[0], 256),
                    (buffers[1], expected[1], 64),
                ):
                    for start, expert in tasks[: rows // width].cpu().tolist():
                        self.assertEqual(start % 64, 0)
                        slots = list(range(start // 64, (start + width) // 64))
                        self.assertTrue(
                            all(metadata_cpu[slot] == expert for slot in slots)
                        )
                        covered.extend(slots)
                self.assertEqual(sorted(covered), list(range(sum(blocks))))
                for guard in guards:
                    self.assertTrue((guard[:32] == -123).all())
                    self.assertTrue((guard[-32:] == -123).all())
                print(
                    "COMPACT_DISTRIBUTION_PASS",
                    name,
                    expected,
                    "rel_l2",
                    (delta / ref2).sqrt().item(),
                    "logits_diff",
                    logits_diff,
                    flush=True,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "A GPU is required")
    def test_compact_cu_override_has_sufficient_tail_capacity(self):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
            gemm2_8x1_compact as compact,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        metadata = torch.arange(2, device=device, dtype=torch.int32).repeat_interleave(
            160
        )
        valid = torch.tensor([320 * 64], device=device, dtype=torch.int32)
        with patch.dict(os.environ, {"CU_NUM": "256"}):
            buffers = compact.prepare_tasks(metadata, valid, 2)
        torch.cuda.synchronize(device)
        self.assertEqual(buffers[2].cpu().tolist(), [0, 320 * 64])
        self.assertGreaterEqual(buffers[1].shape[0], 320)
        tails = buffers[1][:320].cpu().tolist()
        self.assertEqual(
            sorted(start for start, _ in tails), list(range(0, 320 * 64, 64))
        )
        self.assertEqual([expert for _, expert in tails], [0] * 160 + [1] * 160)


if __name__ == "__main__":
    unittest.main()
