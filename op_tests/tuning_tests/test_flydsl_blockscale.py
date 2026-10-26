# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for FlyDSL blockscale E2E tuner eligibility and CSV output."""

import ast
import functools
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_NAME = "aiter.ops.flydsl.fused_moe_gfx942"
_LIMIT_ENV = "AITER_MXFP4_TUNE_SWIGLU_LIMIT"
_BLOCK_QUANTS = ("QuantType.per_1x128", "QuantType.per_128x128")
_BLOCK_VARIANTS = (
    ("block_m128", 128),
    ("block_m128_persistent", 128),
    ("block_8wave", 32),
    ("block_8wave", 64),
    ("block_8wave", 128),
)
_KEYS = [
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
_RESULT_COLUMNS = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "nt",
]


def _load_cpu_code():
    # Importing the tuner probes hardware and sets the default device to CUDA.
    # Execute its real definitions instead; enums and device-dependent calls
    # are stand-ins. Config, LDS checks, eligibility and E2E control flow are real.
    code = ModuleType(__name__)
    code.__dict__.update(
        dataclass=dataclass,
        os=os,
        pd=pd,
        torch=torch,
        get_gfx=lambda: "gfx950",
        ActivationType=SimpleNamespace(Silu="ActivationType.Silu"),
        QuantType=SimpleNamespace(
            **{
                name: f"QuantType.{name}"
                for name in (
                    "No",
                    "per_Token",
                    "per_Tensor",
                    "per_1x32",
                    "per_1x128",
                    "per_128x128",
                )
            }
        ),
        run_flydsl_moe_gfx942=mock.Mock(
            side_effect=AssertionError("GPU kernels must not run in this test")
        ),
    )

    def load(relative_path, names, owner=None):
        path = _ROOT / relative_path
        body = ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body
        if owner is not None:
            body = next(
                node for node in body if getattr(node, "name", None) == owner
            ).body
        nodes = [
            node
            for node in body
            if getattr(node, "name", None) in names
            or (
                isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) in names for target in node.targets)
            )
        ]
        if len(nodes) != len(names):
            raise AssertionError(f"Missing definitions in {path}: {names}")
        exec(  # noqa: S102 - trusted checkout definitions, no module-level GPU setup
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
            code.__dict__,
        )

    load(
        "aiter/ops/flydsl/kernels/moe_gemm_2stage/a8w8_blockscale_config.py",
        {"m128_down_lds_bytes", "routed_down_lds_bytes"},
    )
    load(
        "aiter/ops/flydsl/fused_moe_gfx942.py",
        {"_BLOCKSCALE_PREFILL_PATHS", "Config", "_Problem", "get_tune_space"},
    )
    load(
        "aiter/fused_moe_registry.py",
        {"_KERNEL_NAME_PREFIX", "make_fused_moe_impl_kernel_name"},
    )
    tuner_path = "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
    load(tuner_path, {"_flydsl_e2e_row_supported"})
    load(tuner_path, {"e2e_tune"}, owner="FmoeTuner")
    return code


def _row(**overrides):
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 17,
        "model_dim": 1024,
        "inter_dim": 256,
        "expert": 4,
        "topk": 2,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float8_e4m3fn",
        "q_dtype_w": "torch.float8_e4m3fn",
        "q_type": _BLOCK_QUANTS[0],
        "use_g1u1": 1,
        "doweight_stage1": 0,
        **overrides,
    }


class TestFlydslBlockscaleTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code = _load_cpu_code()

    def setUp(self):
        for patcher in (
            mock.patch.dict(sys.modules, {_BACKEND_NAME: self.code}),
            mock.patch.dict(os.environ, {_LIMIT_ENV: ""}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _config(self, path="block_m128", **kwargs):
        return self.code.Config(256, 256, 128, True, down_path=path, **kwargs)

    def test_block_aliases_and_valid_shapes(self):
        shapes = (
            ("block_m128", 128, 256),
            ("block_m128", 128, 512),
            ("block_m128_persistent", 128, 256),
            ("block_m128_persistent", 128, 512),
            ("block_8wave", 32, 128),
            ("block_8wave", 64, 512),
            ("block_8wave", 128, 256),
        )
        for quant, (path, block_n, inter_dim) in product(_BLOCK_QUANTS, shapes):
            with self.subTest(quant=quant, path=path, block_n=block_n, k=inter_dim):
                config = self._config(path, down_block_n=block_n, num_oc_splits=2)
                row = _row(q_type=quant, inter_dim=inter_dim)
                with mock.patch.object(
                    config, "unsupported_reason", wraps=config.unsupported_reason
                ) as check:
                    self.assertTrue(self.code._flydsl_e2e_row_supported(config, row))
                self.assertEqual(check.call_args.args[0].quant_type, "block")
                self.assertEqual(len(config.to_string().split(":")), 5)
                self.assertEqual(
                    self.code.Config.from_string(config.to_string()), config
                )

    def test_block_input_restrictions(self):
        invalid = (
            {"dtype": "torch.float16"},
            {"dtype": "torch.float32"},
            {"q_dtype_a": "torch.bfloat16"},
            {"q_dtype_w": "torch.bfloat16"},
            {"q_dtype_a": "torch.float8_e4m3fnuz"},
            {"q_dtype_w": "torch.float8_e4m3fnuz"},
            {"act_type": "ActivationType.Gelu"},
            {"act_type": "ActivationType.Swiglu"},
            {"act_type": "ActivationType.Situv2"},
            {"use_g1u1": 0},
            {"doweight_stage1": 1},
        )
        for quant, (path, block_n), change in product(
            _BLOCK_QUANTS, _BLOCK_VARIANTS, invalid
        ):
            with self.subTest(quant=quant, path=path, block_n=block_n, change=change):
                config = self._config(path, down_block_n=block_n)
                self.assertFalse(
                    self.code._flydsl_e2e_row_supported(
                        config, _row(q_type=quant, **change)
                    )
                )

    def test_clamp_matches_run_config_and_runtime(self):
        for limit in (None, "", "0", "0.0", "8", "-1"):
            with self.subTest(limit=limit), mock.patch.dict(os.environ):
                if limit is None:
                    os.environ.pop(_LIMIT_ENV, None)
                else:
                    os.environ[_LIMIT_ENV] = limit
                self.assertEqual(
                    self.code._flydsl_e2e_row_supported(self._config(), _row()),
                    limit in (None, "", "0", "0.0"),
                )

    def test_n_split_and_lds_rejections_use_config(self):
        cases = (
            ("block_m128", 128, 4, {"model_dim": 768}, "num_oc_splits"),
            ("block_m128_persistent", 128, 4, {"model_dim": 768}, "num_oc_splits"),
            ("block_8wave", 128, 2, {"model_dim": 512}, "3 N tiles"),
            ("block_8wave", 128, 1, {"inter_dim": 512}, "LDS"),
            ("block_m128", 128, 1, {"model_dim": 2**21}, "LDS"),
            ("block_m128_persistent", 128, 1, {"model_dim": 2**21}, "LDS"),
            ("block_m128", 128, 1, {"inter_dim": 128}, "multiple of256"),
        )
        for quant, (path, block_n, splits, change, reason) in product(
            _BLOCK_QUANTS, cases
        ):
            with self.subTest(quant=quant, path=path, change=change, reason=reason):
                config = self._config(path, down_block_n=block_n, num_oc_splits=splits)
                with mock.patch.object(
                    config, "unsupported_reason", wraps=config.unsupported_reason
                ) as check:
                    self.assertFalse(
                        self.code._flydsl_e2e_row_supported(
                            config, _row(q_type=quant, **change)
                        )
                    )
                self.assertIn(
                    reason, config.unsupported_reason(check.call_args.args[0])
                )

    def test_quant_modes_do_not_mix(self):
        legacy = (
            self.code.Config(16, 64, 64, False),
            self.code.Config(64, 128, 128, True),
            self.code.Config(64, 128, 128, True, down_path="8x1_compact"),
        )
        for config, quant in product(legacy, _BLOCK_QUANTS):
            with self.subTest(config=config, quant=quant):
                self.assertFalse(
                    self.code._flydsl_e2e_row_supported(config, _row(q_type=quant))
                )
        for (path, block_n), quant in product(
            _BLOCK_VARIANTS,
            (
                "QuantType.No",
                "QuantType.per_Token",
                "QuantType.per_Tensor",
                "QuantType.per_1x32",
                "unknown",
            ),
        ):
            with self.subTest(path=path, block_n=block_n, quant=quant):
                self.assertFalse(
                    self.code._flydsl_e2e_row_supported(
                        self._config(path, down_block_n=block_n), _row(q_type=quant)
                    )
                )

    def test_get_tune_space_owns_architecture_gating(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            with (
                self.subTest(gfx=gfx),
                mock.patch.object(self.code, "get_gfx", return_value=gfx),
            ):
                configs = [
                    self.code.Config.from_string(text)
                    for text in self.code.get_tune_space()
                ]
                block = [config for config in configs if config.is_blockscale_prefill]
                if gfx == "gfx950":
                    self.assertEqual(
                        {(c.down_path, c.down_block_n) for c in block},
                        set(_BLOCK_VARIANTS),
                    )
                    self.assertEqual({c.num_oc_splits for c in block}, {1, 2, 4, 8})
                    # Every Down candidate is tuned with and without stage1 K rotation.
                    down = {(c.down_path, c.down_block_n, c.num_oc_splits) for c in block}
                    self.assertEqual(
                        sorted(
                            (c.down_path, c.down_block_n, c.num_oc_splits, c.k_rotate)
                            for c in block
                        ),
                        sorted((*d, k) for d in down for k in (False, True)),
                    )
                else:
                    self.assertEqual(block, [])

    def test_k_rotate_field_is_optional_and_strict(self):
        row = _row()
        for (path, block_n), k_rotate in product(_BLOCK_VARIANTS, (None, False, True)):
            with self.subTest(path=path, block_n=block_n, k_rotate=k_rotate):
                config = self._config(
                    path, down_block_n=block_n, num_oc_splits=2, k_rotate=k_rotate
                )
                fields = config.to_string().split(":")
                # None keeps the five-field spelling of existing tuned CSVs.
                self.assertEqual(len(fields), 5 if k_rotate is None else 6)
                if k_rotate is not None:
                    self.assertEqual(fields[5], str(int(k_rotate)))
                self.assertEqual(
                    self.code.Config.from_string(config.to_string()), config
                )
                self.assertEqual(
                    self.code._flydsl_e2e_row_supported(config, row),
                    self.code._flydsl_e2e_row_supported(
                        replace(config, k_rotate=None), row
                    ),
                )
        base = self._config().to_string()
        for text in (
            f"{base}:2",
            f"{base}:true",
            f"{base}:",
            "16_64_64_False_False:default:0:128:1:1",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.code.Config.from_string(text)
        for make in (
            lambda: self._config(k_rotate=1),
            lambda: self._config(k_rotate="1"),
            lambda: self.code.Config(16, 64, 64, False, k_rotate=True),
            lambda: self.code.Config(
                16, 64, 64, False, down_path="block_a16w8", k_rotate=False
            ),
        ):
            with self.assertRaises(ValueError):
                make()

    def test_decode_candidates_and_shape_bounds(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            with mock.patch.object(self.code, "get_gfx", return_value=gfx):
                configs = [
                    self.code.Config.from_string(s) for s in self.code.get_tune_space()
                ]
            decode = [c for c in configs if c.is_blockscale_decode]
            self.assertEqual(len(decode), 4 if gfx in ("gfx942", "gfx950") else 0)
            fp8_dtype = str(
                torch.float8_e4m3fn if gfx == "gfx950" else torch.float8_e4m3fnuz
            )
            for config, token, quant in product(
                decode, (1, 2, 4, 8, 9, 16, 256, 257), _BLOCK_QUANTS
            ):
                with (
                    self.subTest(gfx=gfx, config=config, token=token, quant=quant),
                    mock.patch.object(self.code, "get_gfx", return_value=gfx),
                ):
                    self.assertFalse(config.is_blockscale_prefill)
                    self.assertEqual(len(config.to_string().split(":")), 3)
                    self.assertEqual(
                        self.code.Config.from_string(config.to_string()), config
                    )
                    if config.use_batch1_algorithm:
                        eligible = 2 <= token <= 8
                    elif config.BLOCK_N == 16:
                        eligible = token == 1
                    else:
                        eligible = 2 <= token <= 256
                    self.assertEqual(
                        self.code._flydsl_e2e_row_supported(
                            config,
                            _row(
                                gfx=gfx,
                                token=token,
                                q_type=quant,
                                q_dtype_a=fp8_dtype,
                                q_dtype_w=fp8_dtype,
                            ),
                        ),
                        eligible,
                    )
        config = self.code.Config(16, 64, 64, False, down_path="block_a16w8")
        for change in (
            {"model_dim": 768},
            {"inter_dim": 192},
            {"topk": 0},
            {"topk": 128, "expert": 256},
            {"topk": 129, "expert": 256},
            {"q_type": "QuantType.per_Token"},
            {"q_dtype_a": "torch.bfloat16"},
            {"act_type": "ActivationType.Gelu"},
            {"use_g1u1": 0},
            {"dtype": "torch.float16"},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    self.code._flydsl_e2e_row_supported(config, _row(**change))
                )
        with self.assertRaises(ValueError):
            self.code.Config(256, 256, 128, True, down_path="block_a16w8")
        with self.assertRaises(ValueError):
            self.code.Config(
                16,
                64,
                64,
                False,
                down_path="block_a16w8",
                down_output_padding_bytes=128,
            )

    def test_blockscale_formats_match_architecture(self):
        decode = self.code.Config(16, 64, 64, False, down_path="block_a16w8")
        for gfx, dtype_a, dtype_w, quant in product(
            ("gfx942", "gfx950", "gfx1250"),
            (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
            _BLOCK_QUANTS,
        ):
            with (
                self.subTest(gfx=gfx, dtype_a=dtype_a, dtype_w=dtype_w, quant=quant),
                mock.patch.object(self.code, "get_gfx", return_value=gfx),
            ):
                dtype = (
                    torch.float8_e4m3fn if gfx == "gfx950" else torch.float8_e4m3fnuz
                )
                row = _row(
                    gfx=gfx,
                    q_type=quant,
                    q_dtype_a=str(dtype_a),
                    q_dtype_w=str(dtype_w),
                )
                self.assertEqual(
                    self.code._flydsl_e2e_row_supported(decode, row),
                    gfx in ("gfx942", "gfx950") and dtype_a == dtype_w == dtype,
                )
                self.assertEqual(
                    self.code._flydsl_e2e_row_supported(self._config(), row),
                    gfx == "gfx950" and dtype_a == dtype_w == torch.float8_e4m3fn,
                )

    def test_moe_2stage_blockscale_alias_preparation(self):
        # Execute the real test function, but not its module-level CLI/GPU sweep.
        path = _ROOT / "op_tests/test_moe_2stage.py"
        function = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "test_fmoe"
        )
        function.decorator_list = []
        quant_path = _ROOT / "aiter/ops/quant.py"
        quant_nodes = [
            node
            for node in ast.parse(quant_path.read_text()).body
            if getattr(node, "name", None)
            in {"get_dtype_max", "pertoken_quant", "per_block_quant_wrapper"}
        ]
        ns = {
            "torch": torch,
            "functools": functools,
            "dtypes": SimpleNamespace(
                fp32=torch.float32,
                bf16=torch.bfloat16,
                fp16=torch.float16,
                fp8=torch.float8_e4m3fn,
                i8=torch.int8,
                fp4x2=getattr(torch, "float4_e2m1fn_x2", None),
                i4x2=getattr(torch, "int4", None),
            ),
            "get_gfx": lambda: "gfx950",
            "AITER_MOE_NUM_EXPERT_ACTIVATED": 0,
            "AITER_MOE_EXPERT_BALANCE": False,
            "shuffle_weight": lambda tensor, **kwargs: tensor,
            "fp4_utils": SimpleNamespace(e8m0_shuffle=lambda scale: scale),
            "logger": mock.Mock(),
            "checkAllclose": lambda *args, **kwargs: 0,
            "fused_moe": mock.Mock(),
        }
        exec(  # noqa: S102 - trusted pure-Torch quantization definitions
            compile(
                ast.Module(body=quant_nodes, type_ignores=[]), str(quant_path), "exec"
            ),
            ns,
        )
        ns["aiter"] = SimpleNamespace(
            QuantType=self.code.QuantType,
            ActivationType=SimpleNamespace(
                Silu="silu", Swiglu="swiglu", Situv2="situv2"
            ),
            pertoken_quant=ns["pertoken_quant"],
            get_torch_quant=lambda quant: ns["per_block_quant_wrapper"]((1, 128))(
                ns["pertoken_quant"]
            ),
        )
        captured = []

        def stage1(a, w1, w2, weights, ids, **kwargs):
            self.assertEqual(tuple(kwargs["w1_scale"].shape), (2, 2, 2))
            self.assertEqual(tuple(kwargs["a1_scale"].shape), (3, 2))
            captured.extend((a, w1, w2, kwargs["a1_scale"], kwargs["w1_scale"]))
            return (
                torch.arange(3 * 2 * 128, dtype=torch.float32)
                .reshape(3, 2, 128)
                .bfloat16()
            )

        def stage2(a, w1, w2, weights, ids, **kwargs):
            self.assertEqual(tuple(kwargs["w2_scale"].shape), (2, 2, 1))
            self.assertEqual(tuple(kwargs["a2_scale"].shape), (3, 2, 1))
            captured.extend((a, kwargs["a2_scale"], kwargs["w2_scale"]))
            return torch.ones(3, 256, dtype=torch.bfloat16)

        ns.update(
            fused_topk=lambda x, score, topk, renormalize: (
                torch.ones(3, 2),
                torch.arange(2).expand(3, 2).int(),
            ),
            torch_moe_stage1=stage1,
            torch_moe_stage2=stage2,
            run_perftest=lambda *args, **kwargs: (
                torch.ones(3, 256, dtype=torch.bfloat16),
                1.0,
            ),
        )
        exec(  # noqa: S102 - real data preparation with only GPU consumers replaced
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns
        )
        for dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            with self.subTest(dtype=dtype), torch.device("cpu"):
                outputs = []
                for quant in (
                    self.code.QuantType.per_128x128,
                    self.code.QuantType.per_1x128,
                ):
                    torch.manual_seed(23)
                    captured.clear()
                    ns["test_fmoe"](
                        torch.bfloat16,
                        3,
                        256,
                        128,
                        2,
                        2,
                        "silu",
                        "separated",
                        quant,
                        dtype,
                        dtype,
                        use_g1u1=True,
                    )
                    outputs.append(
                        [
                            tensor.contiguous().view(torch.uint8).clone()
                            for tensor in captured
                        ]
                    )
                self.assertEqual(len(outputs[0]), 8)
                self.assertEqual(len(outputs[1]), 8)
                for alias, canonical in zip(*outputs):
                    torch.testing.assert_close(alias, canonical, atol=0, rtol=0)

    def test_legacy_eligibility_is_unchanged(self):
        configs = (
            "16_16_16_False_False:default:0",
            "16_16_16_False_True:default:0",
            "16_64_64_False_False:default:0",
            "16_128_64_False_False:default:0",
            "64_128_128_True_False:default:0",
            "64_128_128_True_False:8x1_compact:128",
            "64_256_128_True_False:1x4_64x256:128",
        )
        modes = {
            "QuantType.No": "no",
            "QuantType.per_Token": "ptpc",
            "QuantType.per_Tensor": "per_tensor",
            "QuantType.per_1x32": "mxfp4",
        }
        with mock.patch.dict(os.environ, {_LIMIT_ENV: "8"}):
            for text, (quant, mode), token, doweight in product(
                configs, modes.items(), (1, 4, 17, 257), (0, 1)
            ):
                with self.subTest(
                    config=text, quant=quant, token=token, doweight=doweight
                ):
                    config = self.code.Config.from_string(text)
                    self.assertEqual(config.to_string(), text)
                    # These fields were not filtered for legacy configs. Runtime
                    # validation, not this tuner refactor, still owns them.
                    row = _row(
                        q_type=quant,
                        token=token,
                        doweight_stage1=doweight,
                        dtype="torch.float16",
                        q_dtype_a="torch.bfloat16",
                        q_dtype_w="torch.int8",
                        act_type="ActivationType.Gelu",
                        use_g1u1=0,
                    )
                    problem = self.code._Problem(
                        token, 4, 512, 1024, 1024, 256, 2, mode
                    )
                    expected = (
                        not doweight and config.unsupported_reason(problem) is None
                    )
                    self.assertEqual(
                        self.code._flydsl_e2e_row_supported(config, row), expected
                    )

    def test_e2e_maps_filtered_results_and_preserves_csv_metadata(self):
        for path, block_n in (*_BLOCK_VARIANTS, ("block_a16w8", 128)):
            with (
                self.subTest(path=path, block_n=block_n),
                tempfile.TemporaryDirectory() as tmp,
            ):
                config = (
                    self.code.Config(16, block_n, 64, False, down_path=path)
                    if path == "block_a16w8"
                    else self._config(path, down_block_n=block_n, num_oc_splits=2)
                )
                text = config.to_string()
                rows = [
                    _row(token=16, dtype="torch.float16"),
                    _row(token=17, q_type=_BLOCK_QUANTS[0]),
                    _row(token=25, doweight_stage1=1),
                    _row(token=33, q_type=_BLOCK_QUANTS[1]),
                    _row(token=49, q_type="QuantType.per_Token"),
                    _row(token=33, q_type=_BLOCK_QUANTS[0]),
                ]
                original = pd.DataFrame(rows, index=[10, 70, 20, 90, 30, 60])
                columns = _KEYS + _RESULT_COLUMNS
                output = str(Path(tmp) / "tuned.csv")
                args = SimpleNamespace(tune_file=output)
                existing = {
                    **rows[0],
                    **dict.fromkeys(_RESULT_COLUMNS, 0),
                    "kernelName1": "unchanged",
                    "kernelName2": "",
                    "err1": "0%",
                    "err2": "0%",
                    "_tag": "keep",
                }
                pd.DataFrame([existing], columns=columns + ["_tag"]).to_csv(
                    output, index=False
                )
                tuner = SimpleNamespace(
                    keys=_KEYS,
                    columns=columns,
                    untunedf=original,
                    get_out_file=lambda filename: filename,
                    calculate=mock.Mock(return_value=(1.25, 2.5)),
                    _run_config_for_shapes=mock.Mock(
                        return_value=[
                            {"e2e_us": 20.0, "status": "ok", "err_ratio": 0.0}
                            for _ in rows
                        ]
                    ),
                )

                def run_candidate(
                    passed_args,
                    target_fused_moe,
                    config_string,
                    args=args,
                    text=text,
                    tuner=tuner,
                    original=original,
                ):
                    self.assertIs(passed_args, args)
                    self.assertEqual(config_string, text)
                    self.assertEqual(target_fused_moe.keywords["config_string"], text)
                    pd.testing.assert_frame_equal(
                        tuner.untunedf, original.iloc[[1, 3, 5]].reset_index(drop=True)
                    )
                    return [
                        {"e2e_us": 7.5, "status": "ok", "err_ratio": 0.02},
                        {"e2e_us": 3.25, "status": "ok", "err_ratio": 0.01},
                        {"e2e_us": 4.25, "status": "ok", "err_ratio": 0.01},
                    ]

                tuner.run_config = mock.Mock(side_effect=run_candidate)
                with (
                    mock.patch.object(self.code, "get_tune_space", return_value=[text]),
                    redirect_stdout(io.StringIO()),
                ):
                    self.code.e2e_tune(tuner, args)
                tuner.run_config.assert_called_once()
                self.assertIs(tuner.untunedf, original)
                self.code.run_flydsl_moe_gfx942.assert_not_called()
                saved = pd.read_csv(output, keep_default_na=False)
                self.assertEqual(list(saved.columns), columns + ["_tag"])
                self.assertEqual(len(saved), 3)
                self.assertEqual(saved.iloc[0].to_dict(), existing)
                for position, us, err in ((1, 7.5, "2.00%"), (3, 3.25, "1.00%")):
                    result = saved[saved["token"] == rows[position]["token"]].iloc[0]
                    expected_key = {**rows[position], "q_type": _BLOCK_QUANTS[0]}
                    self.assertEqual(result[_KEYS].to_dict(), expected_key)
                    self.assertEqual(
                        result[_RESULT_COLUMNS].tolist(),
                        [
                            config.BLOCK_M,
                            0,
                            us,
                            f"impl__flydsl_gfx950__{text}",
                            err,
                            0.0,
                            "",
                            "0%",
                            us,
                            0,
                            0,
                            0,
                            1.25,
                            2.5,
                            0,
                        ],
                    )
                    self.assertEqual(result["_tag"], "")


if __name__ == "__main__":
    unittest.main()
