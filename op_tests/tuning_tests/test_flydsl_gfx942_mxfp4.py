# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import importlib
import itertools
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe_registry import FusedMoeRequest, resolve_fused_moe_impl
from aiter.ops.flydsl import fused_moe_gfx942 as backend
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import gemm as gemm_dispatch
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import gemm2
from aiter.ops.flydsl.moe_common import GateMode


def _mxfp4_inputs(batch=4, experts=2, hidden_dim=512, inter_dim=128, topk=2):
    hidden_states = torch.empty((batch, hidden_dim), dtype=torch.bfloat16)
    w1 = torch.empty(
        (experts, 2 * inter_dim, hidden_dim // 2),
        dtype=torch.float4_e2m1fn_x2,
    )
    w2 = torch.empty(
        (experts, hidden_dim, inter_dim // 2),
        dtype=torch.float4_e2m1fn_x2,
    )
    w1.is_shuffled = True
    w2.is_shuffled = True
    gate_groups = ((hidden_dim // 32 + 7) // 8) * 8
    down_groups = ((inter_dim // 32 + 7) // 8) * 8
    w1_scale = torch.empty(
        (experts * 2 * inter_dim, gate_groups), dtype=torch.float8_e8m0fnu
    )
    w2_scale = torch.empty(
        (experts * hidden_dim, down_groups), dtype=torch.float8_e8m0fnu
    )
    topk_weight = torch.ones((batch, topk), dtype=torch.float32)
    topk_ids = torch.zeros((batch, topk), dtype=torch.int32)
    return hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale


class TestFlydslGfx942Mxfp4(unittest.TestCase):
    def test_two_stage_entry_does_not_select_whole_graph(self):
        fused = importlib.import_module("aiter.fused_moe")
        key = (
            "gfx942",
            80,
            1,
            2048,
            128,
            257,
            9,
            str(ActivationType.Silu),
            str(torch.bfloat16),
            str(torch.float8_e4m3fnuz),
            str(torch.float8_e4m3fnuz),
            str(QuantType.per_Token),
            True,
            False,
        )
        cfg = {
            "kernelName1": "impl__flydsl_gfx942__16_16_16_False",
            "block_m": 16,
            "ksplit": 0,
        }
        request = (
            1,
            2048,
            128,
            257,
            9,
            torch.bfloat16,
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fnuz,
            QuantType.per_Token,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
        )
        fused.get_2stage_cfgs.cache_clear()
        try:
            with (
                patch.object(fused, "cfg_2stages", ({key: cfg}, {})),
                patch.object(fused, "get_cu_num", return_value=80),
                patch.object(fused, "get_gfx_runtime", return_value="gfx942"),
            ):
                self.assertIsNotNone(fused.get_2stage_cfgs(*request).full_impl)
                staged = fused.get_2stage_cfgs(*request, _disable_full_impl=True)
                self.assertIsNone(staged.full_impl)
                self.assertIsNotNone(staged.stage1)
                self.assertIsNotNone(staged.stage2)
        finally:
            fused.get_2stage_cfgs.cache_clear()

    def test_compiled_kernel_cache_supports_compile_only_without_gpu(self):
        sentinel = object()
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                backend, "_get_compiled_kernel_cached", return_value=sentinel
            ) as cached,
        ):
            output = backend._get_compiled_kernel(
                N=256,
                K=512,
                weight_dtype_str="fp4",
                quant_type_str="mxfp4",
                TOPK=2,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=64,
                stage="gateup",
                alg="batch1",
                E=None,
            )

        self.assertIs(output, sentinel)
        self.assertEqual(cached.call_args.args[0], (None, None, None))

    def test_compiled_kernel_cache_key_includes_target_arch(self):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942.common import (
            get_device_cache_key,
        )

        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx942"}):
                gfx942_key = get_device_cache_key()
            with patch.dict(os.environ, {"FLYDSL_GPU_ARCH": "gfx950"}):
                gfx950_key = get_device_cache_key()

        self.assertEqual(gfx942_key, (None, "gfx942", None))
        self.assertEqual(gfx950_key, (None, "gfx950", None))
        self.assertNotEqual(gfx942_key, gfx950_key)

    def test_aot_parser_and_worker_support_whole_graph_rows(self):
        from aiter.aot.flydsl import moe as aot_moe

        columns = [
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
            "doweight_stage1",
            "kernelName1",
            "kernelName2",
        ]
        row = [
            256,
            4,
            3584,
            384,
            896,
            16,
            "ActivationType.Situv2",
            "torch.bfloat16",
            "torch.bfloat16",
            "torch.float4_e2m1fn_x2",
            "QuantType.per_1x32",
            0,
            "impl__flydsl_gfx950__16_16_16_False_True",
            "",
        ]
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerow(row)
            handle.flush()
            jobs = aot_moe.parse_csv(handle.name)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["stage"], "whole_graph")
        self.assertEqual(jobs[0]["weight_dtype"], "fp4")
        self.assertEqual(jobs[0]["quant_type"], "mxfp4")

        with patch.object(backend, "precompile_flydsl_moe") as precompile:
            result = aot_moe.compile_one_config(**jobs[0])

        self.assertIsNotNone(result["compile_time"])
        precompile.assert_called_once_with(
            config_string="16_16_16_False_True",
            batch=4,
            model_dim=3584,
            inter_dim=384,
            experts=896,
            topk=16,
            weight_dtype="fp4",
            quant_type="mxfp4",
            activation="situv2",
        )

    def test_gfx950_whole_graph_implementation_is_registered(self):
        implementation = resolve_fused_moe_impl(
            "impl__flydsl_gfx950__16_16_16_False_True"
        )
        self.assertIsNotNone(implementation)

    def test_problem_restores_packed_dimensions_and_validates_e8m0_scales(self):
        hidden_states, w1, w2, _, topk_ids, w1_scale, w2_scale = _mxfp4_inputs()
        problem = backend._Problem.from_inputs(
            hidden_states, w1, w2, topk_ids, QuantType.per_1x32
        )

        self.assertEqual(problem.hidden_dim, 512)
        self.assertEqual(problem.inter_dim, 128)
        self.assertEqual(problem.gateup_dim, 256)
        self.assertEqual(problem.quant_type, "mxfp4")
        backend._validate_mxfp4_inputs(w1, w2, w1_scale, w2_scale, problem)

        with self.assertRaisesRegex(ValueError, "E8M0"):
            backend._validate_mxfp4_inputs(w1, w2, w1_scale.float(), w2_scale, problem)
        with self.assertRaisesRegex(ValueError, "expected at least"):
            backend._validate_mxfp4_inputs(w1, w2, w1_scale[:1], w2_scale, problem)

    def test_mxfp4_config_gates_prefill_and_specialized_down(self):
        hidden_states, w1, w2, _, topk_ids, _, _ = _mxfp4_inputs()
        problem = backend._Problem.from_inputs(
            hidden_states, w1, w2, topk_ids, QuantType.per_1x32
        )

        self.assertIsNone(
            backend.Config.from_string("16_16_16_False").unsupported_reason(problem)
        )
        self.assertIsNone(
            backend.Config.from_string("16_16_16_False_True").unsupported_reason(
                problem
            )
        )
        self.assertIn(
            "prefill",
            backend.Config.from_string("64_128_128_True").unsupported_reason(problem),
        )
        specialized = backend.Config(
            64,
            128,
            128,
            True,
            down_path="1x4_64x256",
            GATEUP_BLOCK_M=64,
            down_output_padding_bytes=128,
        )
        self.assertIn("prefill", specialized.unsupported_reason(problem))
        mxfp4_space = backend.get_tune_space(4, include_prefill=False)
        self.assertIn("16_16_16_False_True", mxfp4_space)
        self.assertTrue(
            all(
                not backend.Config.from_string(item).use_prefill for item in mxfp4_space
            )
        )

    def test_extended_config_preserves_direct_flag(self):
        config = backend.Config.from_string("16_16_16_False_True:default:16:none")
        self.assertTrue(config.use_batch1_algorithm)
        self.assertEqual(config.to_string(), "16_16_16_False_True")

        problem = backend._Problem(
            batch=4,
            experts=8,
            gateup_dim=256,
            hidden_dim=512,
            model_dim=512,
            inter_dim=128,
            topk=2,
            quant_type="mxfp4",
        )
        padded = backend.Config.from_string("16_16_16_False_True:default:16:0")
        self.assertIn("extended", padded.unsupported_reason(problem))

    def test_direct_path_rejects_incomplete_k_tiles_before_launch(self):
        for quant_type in ("no", "ptpc", "per_tensor"):
            for hidden_dim, inter_dim, expected in (
                (384, 128, "gateup K"),
                (512, 96, "down K"),
            ):
                with self.subTest(quant_type=quant_type, shape=(hidden_dim, inter_dim)):
                    problem = backend._Problem(
                        batch=2,
                        experts=2,
                        gateup_dim=2 * inter_dim,
                        hidden_dim=hidden_dim,
                        model_dim=hidden_dim,
                        inter_dim=inter_dim,
                        topk=1,
                        quant_type=quant_type,
                    )
                    reason = backend.Config.from_string(
                        "16_16_16_False_True"
                    ).unsupported_reason(problem)
                    self.assertIsNotNone(reason)
                    self.assertIn(expected, reason)

        with (
            patch.object(backend, "_get_compiled_kernel") as compile_kernel,
            self.assertRaisesRegex(ValueError, "gateup K"),
        ):
            backend.precompile_flydsl_moe(
                config_string="16_16_16_False_True",
                batch=2,
                model_dim=384,
                inter_dim=128,
                experts=2,
                topk=1,
                weight_dtype="bf16",
                quant_type="no",
                activation="silu",
            )
        compile_kernel.assert_not_called()

    def test_batch1_builder_rejects_truncated_gateup_k(self):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import gemm1

        for weight_dtype, quant_type in (("bf16", "no"), ("fp8", "ptpc")):
            with (
                self.subTest(weight_dtype=weight_dtype),
                patch.object(gemm1, "get_rocm_arch", return_value="gfx942"),
                self.assertRaisesRegex(AssertionError, "K divisible by 256"),
            ):
                gemm1._build_moe_gemm1(
                    N=256,
                    K=384,
                    weight_dtype=weight_dtype,
                    weight_quant_type=quant_type,
                    TOPK=1,
                    BLOCK_TILE_SIZE_M=16,
                    BLOCK_TILE_SIZE_N=32,
                    alg="batch1",
                )

    def test_aot_rejects_unsupported_whole_graph_semantics(self):
        from aiter.aot.flydsl import moe as aot_moe

        row = {
            "token": 2,
            "model_dim": 512,
            "inter_dim": 128,
            "expert": 2,
            "topk": 1,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float8_e4m3fnuz",
            "q_dtype_w": "torch.float8_e4m3fnuz",
            "q_type": "QuantType.per_Token",
            "doweight_stage1": 0,
            "kernelName1": "impl__flydsl_gfx942__16_16_16_False_True",
            "kernelName2": "",
        }
        for unsupported in (
            {"act_type": "ActivationType.Gelu"},
            {"doweight_stage1": 1},
        ):
            with (
                self.subTest(**unsupported),
                tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as file,
            ):
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow({**row, **unsupported})
                file.flush()
                self.assertEqual(aot_moe.parse_csv(file.name), [])

    def test_situv2_scalars_are_runtime_values(self):
        self.assertEqual(
            backend._activation_scalars("situv2", 0.5, 2.0, None),
            (0.5, 2.0, 2.0, 0.5, float("inf")),
        )
        self.assertEqual(
            backend._activation_scalars("situv2", 0.5, 2.0, 6.0),
            (0.5, 2.0, 2.0, 0.5, 6.0),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            backend._activation_scalars("situv2", 0.0, 2.0, None)

    def test_public_wrapper_routes_mxfp4_direct_with_runtime_parameters(self):
        inputs = _mxfp4_inputs(batch=4)
        hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale = inputs
        sentinel = object()

        with (
            patch.object(backend, "get_gfx", return_value="gfx950"),
            patch.object(backend, "_run_batch1", return_value=sentinel) as run,
        ):
            output = backend.run_flydsl_moe_gfx942(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                ActivationType.Situv2,
                QuantType.per_1x32,
                w1_scale,
                w2_scale,
                None,
                None,
                0,
                "16_16_16_False_True",
                None,
                0.5,
                2.0,
                GateMode.INTERLEAVE,
            )

        self.assertIs(output, sentinel)
        self.assertEqual(run.call_args.args[-3:], (0.5, 2.0, True))

    def test_public_wrapper_routes_bf16_situv2_direct(self):
        hidden_states = torch.empty((2, 512), dtype=torch.bfloat16)
        w1 = torch.empty((2, 256, 512), dtype=torch.bfloat16)
        w2 = torch.empty((2, 512, 128), dtype=torch.bfloat16)
        w1.is_shuffled = True
        w2.is_shuffled = True
        topk_weight = torch.ones((2, 2), dtype=torch.float32)
        topk_ids = torch.zeros((2, 2), dtype=torch.int32)
        sentinel = object()

        with patch.object(backend, "_run_batch1", return_value=sentinel) as run:
            output = backend.run_flydsl_moe_gfx942(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                ActivationType.Situv2,
                QuantType.No,
                None,
                None,
                None,
                None,
                0,
                "16_16_16_False_True",
                None,
                0.5,
                2.0,
                GateMode.SEPARATED,
            )

        self.assertIs(output, sentinel)
        self.assertEqual(run.call_args.args[-3:], (0.5, 2.0, False))

    def test_public_wrapper_rejects_unsupported_gate_modes(self):
        hidden_states = torch.empty((1, 512), dtype=torch.bfloat16)
        w1 = torch.empty((2, 256, 512), dtype=torch.bfloat16)
        w2 = torch.empty((2, 512, 128), dtype=torch.bfloat16)
        w1.is_shuffled = True
        w2.is_shuffled = True
        topk_weight = torch.ones((1, 2), dtype=torch.float32)
        topk_ids = torch.zeros((1, 2), dtype=torch.int32)

        for gate_mode in (GateMode.GATE_ONLY, GateMode.MOCK_GATE_ONLY):
            with self.assertRaisesRegex(RuntimeError, "Unsupported gate mode"):
                backend.run_flydsl_moe_gfx942(
                    hidden_states,
                    w1,
                    w2,
                    topk_weight,
                    topk_ids,
                    ActivationType.Silu,
                    QuantType.No,
                    None,
                    None,
                    None,
                    None,
                    0,
                    "16_16_16_False",
                    gate_mode=gate_mode,
                )

        with self.assertRaisesRegex(RuntimeError, "interleaved.*only.*MXFP4"):
            backend.run_flydsl_moe_gfx942(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                ActivationType.Silu,
                QuantType.No,
                None,
                None,
                None,
                None,
                0,
                "16_16_16_False",
                gate_mode=GateMode.INTERLEAVE,
            )

    def test_mxfp4_direct_path_fuses_output_clear_and_forwards_situv2(self):
        inputs = _mxfp4_inputs(batch=4)
        hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale = inputs
        problem = backend._Problem.from_inputs(
            hidden_states, w1, w2, topk_ids, QuantType.per_1x32
        )
        compiled = []
        launched = []

        def fake_compile(**kwargs):
            compiled.append(kwargs)
            return kwargs["stage"]

        def fake_launch(kernel, *args):
            launched.append((kernel, args))

        with (
            patch.object(backend, "_get_compiled_kernel", side_effect=fake_compile),
            patch.object(backend, "_launch", side_effect=fake_launch),
        ):
            output = backend._run_batch1(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale,
                w2_scale,
                problem,
                "situv2",
                None,
                0.5,
                2.0,
                False,
            )

        self.assertEqual([entry["stage"] for entry in compiled], ["gateup", "down"])
        self.assertEqual(compiled[0]["weight_dtype_str"], "fp4")
        self.assertEqual(compiled[0]["BLOCK_TILE_SIZE_N"], 64)
        self.assertFalse(compiled[0]["mxfp4_gate_up_interleaved"])
        self.assertTrue(compiled[0]["fused_down_clear"])
        self.assertEqual(compiled[1]["BLOCK_TILE_SIZE_N"], 32)
        self.assertIs(launched[0][1][4], output)
        self.assertIs(launched[1][1][2], output)
        self.assertEqual(launched[0][1][-5:], (0.5, 2.0, 2.0, 0.5, float("inf")))
        self.assertEqual(launched[1][1][-5:], (0.5, 2.0, 2.0, 0.5, float("inf")))

    def test_batch1_clear_covers_all_weight_types_and_tokens(self):
        # 所有 batch1/direct 组合开启清零，保持原有 dtype、layout 和 tile。
        cases = (
            (torch.float8_e4m3fnuz, QuantType.per_Token, False),
            (torch.float8_e4m3fnuz, QuantType.per_Tensor, False),
            (torch.bfloat16, QuantType.No, False),
            (torch.float4_e2m1fn_x2, QuantType.per_1x32, False),
            (torch.float4_e2m1fn_x2, QuantType.per_1x32, True),
        )
        for (
            batch,
            (weight_dtype, quant_type, interleaved),
            activation,
        ) in itertools.product(range(1, 9), cases, ("silu", "swiglu", "situv2")):
            with self.subTest(
                batch=batch,
                dtype=weight_dtype,
                activation=activation,
                interleaved=interleaved,
            ):
                is_mxfp4 = weight_dtype == torch.float4_e2m1fn_x2
                if is_mxfp4:
                    hidden, w1, w2, weights, ids, scale1, scale2 = _mxfp4_inputs(
                        batch=batch
                    )
                else:
                    hidden = torch.empty((batch, 512), dtype=torch.bfloat16)
                    w1 = torch.empty((2, 256, 512), dtype=weight_dtype)
                    w2 = torch.empty((2, 512, 128), dtype=weight_dtype)
                    weights = torch.ones((batch, 2), dtype=torch.float32)
                    ids = torch.zeros((batch, 2), dtype=torch.int32)
                    scale1 = scale2 = (
                        None if weight_dtype == torch.bfloat16 else torch.ones(1)
                    )
                problem = backend._Problem.from_inputs(hidden, w1, w2, ids, quant_type)
                compiled, launched = [], []
                with (
                    patch.object(
                        backend,
                        "_get_compiled_kernel",
                        side_effect=lambda compiled=compiled, **kw: (
                            compiled.append(kw) or kw["stage"]
                        ),
                    ),
                    patch.object(
                        backend,
                        "_launch",
                        side_effect=lambda kernel, *args, launched=launched: (
                            launched.append(args)
                        ),
                    ),
                    patch.object(torch, "zeros", wraps=torch.zeros) as zeros,
                ):
                    output = backend._run_batch1(
                        hidden,
                        w1,
                        w2,
                        weights,
                        ids,
                        scale1,
                        scale2,
                        problem,
                        activation,
                        None,
                        0.5,
                        2.0,
                        interleaved,
                    )
                self.assertTrue(compiled[0]["fused_down_clear"])
                self.assertEqual(
                    compiled[0]["BLOCK_TILE_SIZE_N"],
                    64 if is_mxfp4 and batch >= 4 else 32,
                )
                self.assertEqual(
                    compiled[1]["BLOCK_TILE_SIZE_N"],
                    32 if is_mxfp4 and batch > 1 else 64,
                )
                self.assertEqual(compiled[0]["mxfp4_gate_up_interleaved"], interleaved)
                zeros.assert_not_called()
                self.assertIs(launched[0][4], output)
                self.assertIs(launched[1][2], output)
                self.assertIs(launched[1][4], weights)
                # TOPK 已在 grid.y；M=token 数，不能误传 topk。
                self.assertEqual([args[6] for args in launched], [batch, batch])
                expected = backend._activation_scalars(activation, 0.5, 2.0, None)
                self.assertEqual(launched[0][-5:], expected)
                self.assertEqual(launched[1][-5:], expected)

    def test_precompile_batch1_clear_matches_runtime_for_all_types(self):
        for batch, (weight_dtype, quant_type) in itertools.product(
            range(1, 9),
            (("bf16", "no"), ("fp8", "ptpc"), ("fp8", "per_tensor"), ("fp4", "mxfp4")),
        ):
            with self.subTest(
                batch=batch, weight_dtype=weight_dtype, quant_type=quant_type
            ):
                compiled, launched = [], []
                with (
                    patch.object(
                        backend,
                        "_get_compiled_kernel",
                        side_effect=lambda compiled=compiled, **kw: (
                            compiled.append(kw) or kw["stage"]
                        ),
                    ),
                    patch.object(backend, "_ptr", side_effect=lambda tensor: tensor),
                    patch.object(
                        backend,
                        "_run_compiled",
                        side_effect=lambda kernel, *args, launched=launched: (
                            launched.append(args)
                        ),
                    ),
                ):
                    backend.precompile_flydsl_moe(
                        config_string=(
                            "16_16_16_False" if batch == 1 else "16_16_16_False_True"
                        ),
                        batch=batch,
                        model_dim=2048,
                        inter_dim=128,
                        experts=257,
                        topk=9,
                        weight_dtype=weight_dtype,
                        quant_type=quant_type,
                        activation="silu",
                    )
                is_mxfp4 = weight_dtype == "fp4"
                gate_count = 2 if is_mxfp4 else 1
                self.assertEqual(len(compiled), gate_count + 1)
                for gate, args in zip(compiled[:gate_count], launched[:gate_count]):
                    self.assertTrue(gate["fused_down_clear"])
                    self.assertEqual(
                        gate["BLOCK_TILE_SIZE_N"], 64 if is_mxfp4 and batch >= 4 else 32
                    )
                    self.assertEqual(args[4].dtype, torch.bfloat16)
                self.assertEqual(
                    compiled[-1]["BLOCK_TILE_SIZE_N"],
                    32 if is_mxfp4 and batch > 1 else 64,
                )
                self.assertEqual(launched[-1][4].dtype, torch.float32)
                self.assertEqual(
                    [args[6] for args in launched], [batch] * (gate_count + 1)
                )

    def test_mxfp4_splitk_forwards_layout_scales_and_runtime_situv2(self):
        inputs = _mxfp4_inputs(batch=16)
        hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale = inputs
        problem = backend._Problem.from_inputs(
            hidden_states, w1, w2, topk_ids, QuantType.per_1x32
        )
        compiled = []
        launched = []
        sorted_ids = torch.zeros((16,), dtype=torch.int32)
        sorted_weights = torch.ones((16,), dtype=torch.float32)
        sorted_expert_ids = torch.zeros((1,), dtype=torch.int32)
        num_valid_ids = torch.tensor([16], dtype=torch.int32)
        expected_output = torch.zeros(
            (problem.batch, problem.model_dim), dtype=torch.bfloat16
        )

        def fake_compile(**kwargs):
            compiled.append(kwargs)
            return kwargs["stage"]

        with (
            patch.object(
                backend,
                "moe_sorting",
                return_value=(
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    expected_output,
                ),
            ),
            patch.object(backend, "_get_compiled_kernel", side_effect=fake_compile),
            patch.object(
                backend,
                "_launch",
                side_effect=lambda kernel, *args: launched.append((kernel, args)),
            ),
        ):
            output = backend._run_decode(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                w1_scale,
                w2_scale,
                None,
                None,
                0,
                backend.Config.from_string("16_16_16_False"),
                problem,
                "situv2",
                None,
                0.5,
                2.0,
                True,
            )

        self.assertIs(output, expected_output)
        self.assertEqual([entry["alg"] for entry in compiled], ["splitk", "splitk"])
        self.assertEqual(
            [entry["weight_dtype_str"] for entry in compiled], ["fp4", "fp4"]
        )
        self.assertTrue(compiled[0]["mxfp4_gate_up_interleaved"])
        self.assertIs(launched[0][1][7], w1_scale)
        self.assertIs(launched[1][1][7], w2_scale)
        self.assertEqual(launched[0][1][-5:], (0.5, 2.0, 2.0, 0.5, float("inf")))
        self.assertEqual(launched[1][1][-5:], (0.5, 2.0, 2.0, 0.5, float("inf")))

    def test_stage2_only_forwards_fp4_options_to_default_builder(self):
        common = {
            "N": 512,
            "K": 128,
            "weight_dtype": "fp4",
            "weight_quant_type": "mxfp4",
            "TOPK": 2,
            "BLOCK_TILE_SIZE_M": 16,
            "BLOCK_TILE_SIZE_N": 64,
        }

        gemm2._compile_moe_gemm2_cached.cache_clear()
        with patch.dict(gemm2._BUILDERS, {"default": lambda **kwargs: kwargs}):
            default_kwargs = gemm2._compile_moe_gemm2_cached(
                None,
                **common,
                down_path="default",
                mxfp4_gate_up_interleaved=False,
                fused_down_clear=True,
            )
        self.assertFalse(default_kwargs["mxfp4_gate_up_interleaved"])
        self.assertTrue(default_kwargs["fused_down_clear"])

        gemm2._compile_moe_gemm2_cached.cache_clear()
        with patch.dict(gemm2._BUILDERS, {"1x4_64x256": lambda **kwargs: kwargs}):
            specialized_kwargs = gemm2._compile_moe_gemm2_cached(
                None,
                **{**common, "weight_dtype": "fp8", "weight_quant_type": "ptpc"},
                down_path="1x4_64x256",
                mxfp4_gate_up_interleaved=False,
                fused_down_clear=True,
            )
        self.assertNotIn("mxfp4_gate_up_interleaved", specialized_kwargs)
        self.assertNotIn("fused_down_clear", specialized_kwargs)

    def test_force_batch1_compile_flag_overrides_algorithm(self):
        sentinel = object()
        with (
            patch.object(gemm_dispatch, "get_device_cache_key", return_value=None),
            patch.object(
                gemm_dispatch, "_compile_gemm_cached", return_value=sentinel
            ) as compile_cached,
        ):
            output = gemm_dispatch.compile_gemm(
                N=256,
                K=512,
                weight_dtype="fp4",
                weight_quant_type="mxfp4",
                TOPK=2,
                BLOCK_TILE_SIZE_M=16,
                BLOCK_TILE_SIZE_N=64,
                alg="splitk",
                force_batch1_path=True,
            )

        self.assertIs(output, sentinel)
        self.assertEqual(compile_cached.call_args.args[9], "batch1")

    def test_request_forwards_beta_linear_beta_and_gate_mode(self):
        hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale = (
            _mxfp4_inputs()
        )
        request = FusedMoeRequest(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            beta=0.5,
            linear_beta=2.0,
            gate_mode=GateMode.INTERLEAVE,
            q_dtype_a=torch.bfloat16,
            q_dtype_w=torch.float4_e2m1fn_x2,
        )
        sentinel = object()
        with patch.object(
            backend, "run_flydsl_moe_gfx942", return_value=sentinel
        ) as run:
            self.assertIs(
                backend.run_flydsl_moe_gfx942_impl(request, "16_16_16_False"), sentinel
            )
        self.assertEqual(run.call_args.args[-3:], (0.5, 2.0, GateMode.INTERLEAVE))

    def test_precompile_mxfp4_direct_covers_both_gate_layouts(self):
        compiled = []
        launched = []

        def fake_compile(**kwargs):
            compiled.append(kwargs)
            return kwargs["stage"]

        with (
            patch.object(backend, "_get_compiled_kernel", side_effect=fake_compile),
            patch.object(
                backend,
                "_run_compiled",
                side_effect=lambda kernel, *args: launched.append((kernel, args)),
            ),
        ):
            backend.precompile_flydsl_moe(
                config_string="16_16_16_False_True",
                batch=4,
                model_dim=3584,
                inter_dim=384,
                experts=896,
                topk=16,
                weight_dtype="fp4",
                quant_type="mxfp4",
                activation="situv2",
            )

        self.assertEqual(
            [item["stage"] for item in compiled], ["gateup", "gateup", "down"]
        )
        self.assertEqual(
            [item["mxfp4_gate_up_interleaved"] for item in compiled[:2]],
            [False, True],
        )
        self.assertTrue(all(item["fused_down_clear"] for item in compiled[:2]))
        self.assertEqual(compiled[2]["BLOCK_TILE_SIZE_N"], 32)
        self.assertEqual(len(launched), 3)

    def test_precompile_specialized_prefill_uses_specialized_launcher_abi(self):
        compiled = []
        launched = []

        def fake_compile(**kwargs):
            compiled.append(kwargs)
            return kwargs["stage"]

        with (
            patch.object(backend, "_get_compiled_kernel", side_effect=fake_compile),
            patch.object(
                backend,
                "_run_compiled",
                side_effect=lambda kernel, *args: launched.append((kernel, args)),
            ),
        ):
            backend.precompile_flydsl_moe(
                config_string="64_128_128_True:1x4_64x256:64:128",
                batch=1024,
                model_dim=4096,
                inter_dim=192,
                experts=192,
                topk=8,
                weight_dtype="fp8",
                quant_type="per_tensor",
                activation="silu",
            )

        self.assertEqual([item["stage"] for item in compiled], ["gateup", "down"])
        self.assertEqual(compiled[1]["down_path"], "1x4_64x256")
        self.assertFalse(compiled[1]["USE_ATOMIC_WRITE"])
        self.assertEqual(len(launched[0][1]), 17)
        self.assertEqual(len(launched[1][1]), 12)

    def test_precompile_does_not_launch_cached_executable_in_compile_only(self):
        class CachedExecutable:
            _cf = object()

        with (
            patch.object(
                backend, "_get_compiled_kernel", return_value=CachedExecutable()
            ),
            patch.object(backend, "_run_compiled") as run_compiled,
            patch.dict(os.environ, {"COMPILE_ONLY": "1"}),
        ):
            backend.precompile_flydsl_moe(
                config_string="16_16_16_False_True",
                batch=4,
                model_dim=3584,
                inter_dim=384,
                experts=896,
                topk=16,
                weight_dtype="fp4",
                quant_type="mxfp4",
                activation="situv2",
            )

        run_compiled.assert_not_called()

    def test_compact_task_capacities_and_offline_cu_count(self):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942.gemm2_8x1_compact import (
            device_cu_count,
            task_capacities,
        )

        with patch.dict(os.environ, {"CU_NUM": "80"}):
            self.assertEqual(device_cu_count(), 80)
        self.assertEqual(task_capacities(0, 1, cu_count=80), (1, 1))
        self.assertEqual(task_capacities(1000, 128, cu_count=80), (250, 572))
        self.assertEqual(
            task_capacities(
                1000,
                128,
                cu_count=80,
                min_tail_utilization=0,
            ),
            (250, 384),
        )

    def test_compact_prefill_allocates_workspace_and_uses_metadata_block(self):
        from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import (
            gemm2_8x1_compact,
        )

        batch, experts, topk = 2, 8, 4
        hidden_states = torch.empty((batch, 256), dtype=torch.bfloat16)
        w1 = torch.empty((experts, 512, 256), dtype=torch.float8_e4m3fnuz)
        w2 = torch.empty((experts, 256, 256), dtype=torch.float8_e4m3fnuz)
        topk_weight = torch.ones((batch, topk), dtype=torch.float32)
        topk_ids = torch.zeros((batch, topk), dtype=torch.int32)
        w1_scale = torch.ones((experts, 512), dtype=torch.float32)
        w2_scale = torch.ones((experts, 256), dtype=torch.float32)
        sorted_ids = torch.zeros(64, dtype=torch.int32)
        sorted_weights = torch.ones(64, dtype=torch.float32)
        sorted_expert_ids = torch.zeros(8, dtype=torch.int32)
        num_valid_ids = torch.tensor([64, batch], dtype=torch.int32)
        expected_output = torch.empty((batch, 256), dtype=torch.bfloat16)
        full_tasks = torch.empty((2, 2), dtype=torch.int32)
        tail_tasks = torch.empty((8, 2), dtype=torch.int32)
        task_counts = torch.empty(2, dtype=torch.int32)
        compiled = []
        launched = []

        def fake_quant(tensor, **_kwargs):
            return (
                torch.empty_like(tensor, dtype=torch.float8_e4m3fnuz),
                torch.ones(1, dtype=torch.float32),
            )

        with (
            patch.object(
                backend,
                "moe_sorting",
                return_value=(
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    expected_output,
                ),
            ) as sorting,
            patch.object(backend.aiter, "get_hip_quant", return_value=fake_quant),
            patch.object(
                backend,
                "_get_compiled_kernel",
                side_effect=lambda **kwargs: compiled.append(kwargs) or kwargs["stage"],
            ),
            patch.object(
                backend,
                "_launch",
                side_effect=lambda kernel, *args: launched.append((kernel, args)),
            ),
            patch.object(
                gemm2_8x1_compact,
                "allocate_task_buffers",
                return_value=(full_tasks, tail_tasks, task_counts),
            ),
            patch.object(
                backend, "invert_sorted_ids", return_value=lambda *_args: None
            ),
            patch.object(backend, "sorted_sum", return_value=lambda *_args: None),
        ):
            output = backend._run_prefill(
                hidden_states,
                w1,
                w2,
                topk_weight,
                topk_ids,
                QuantType.per_Token,
                w1_scale,
                w2_scale,
                None,
                None,
                0,
                backend.Config.from_string("64_128_128_True:8x1_compact:64:128"),
                backend._Problem(
                    batch=batch,
                    experts=experts,
                    gateup_dim=512,
                    hidden_dim=256,
                    model_dim=256,
                    inter_dim=256,
                    topk=topk,
                    quant_type="ptpc",
                ),
                "silu",
                None,
                1.0,
                1.0,
            )

        self.assertIs(output, expected_output)
        self.assertEqual(sorting.call_args.args[5], 64)
        self.assertEqual(compiled[0]["METADATA_TILE_SIZE_M"], 64)
        self.assertEqual(compiled[1]["down_path"], "8x1_compact")
        self.assertEqual(compiled[1]["BLOCK_TILE_SIZE_M"], 64)
        self.assertEqual(compiled[1]["BLOCK_TILE_SIZE_N"], 128)
        self.assertEqual(compiled[1]["METADATA_TILE_SIZE_M"], 64)
        self.assertEqual(launched[1][1][2].shape, (512, 320))
        self.assertEqual(len(launched[1][1]), 16)
        self.assertIs(launched[1][1][11], full_tasks)
        self.assertIs(launched[1][1][12], tail_tasks)
        self.assertIs(launched[1][1][13], task_counts)
        self.assertEqual(launched[1][1][-2:], (2, 8))

    def test_precompile_compact_uses_cpu_task_workspace_abi(self):
        compiled = []
        launched = []

        def fake_compile(**kwargs):
            compiled.append(kwargs)
            return kwargs["stage"]

        with (
            patch.object(backend, "_get_compiled_kernel", side_effect=fake_compile),
            patch.object(
                backend,
                "_run_compiled",
                side_effect=lambda kernel, *args: launched.append((kernel, args)),
            ),
            patch.dict(os.environ, {"CU_NUM": "80"}),
        ):
            backend.precompile_flydsl_moe(
                config_string="64_128_128_True:8x1_compact:64:128",
                batch=1024,
                model_dim=256,
                inter_dim=256,
                experts=128,
                topk=4,
                weight_dtype="fp8",
                quant_type="ptpc",
                activation="silu",
            )

        self.assertEqual([item["stage"] for item in compiled], ["gateup", "down"])
        self.assertEqual(compiled[0]["METADATA_TILE_SIZE_M"], 64)
        self.assertEqual(compiled[1]["down_path"], "8x1_compact")
        self.assertEqual(compiled[1]["METADATA_TILE_SIZE_M"], 64)
        self.assertEqual(len(launched[1][1]), 17)
        self.assertEqual(type(launched[1][1][11]).__name__, "PointerJitArg")
        self.assertEqual(type(launched[1][1][12]).__name__, "PointerJitArg")
        self.assertEqual(type(launched[1][1][13]).__name__, "PointerJitArg")
        self.assertGreaterEqual(launched[1][1][14], 1)
        self.assertGreaterEqual(launched[1][1][15], 1)
        self.assertEqual(launched[1][1][16], 0)


if __name__ == "__main__":
    unittest.main()
