# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU regressions for the whole-graph tuner baseline contract."""

import contextlib
import importlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from aiter import ActivationType, QuantType
from aiter.ops.flydsl import fused_moe_gfx942 as backend


class TestFmoeE2eBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 调优模块有历史默认设备副作用；CPU测试不让它影响其它测试。
        with patch.object(torch, "set_default_device"):
            cls.module = importlib.import_module(
                "csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune"
            )

    def run_search(
        self,
        *,
        baseline_us=10.0,
        baseline_status="ok",
        candidate_us=12.0,
        output_exists=True,
        exact=False,
        empty_output=False,
        exclude_patterns=(),
    ):
        row = {
            "gfx": "gfx942",
            "cu_num": 80,
            "token": 2,
            "model_dim": 512,
            "inter_dim": 128,
            "expert": 2,
            "topk": 1,
            "act_type": str(ActivationType.Silu),
            "dtype": str(torch.bfloat16),
            "q_dtype_a": str(torch.float8_e4m3fnuz),
            "q_dtype_w": str(torch.float8_e4m3fnuz),
            "q_type": str(QuantType.per_Token),
            "use_g1u1": 1,
            "doweight_stage1": 0,
        }
        result_columns = [
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
        ]
        tuner = object.__new__(self.module.FmoeTuner)
        tuner.keys = list(row)
        tuner.columns = tuner.keys + result_columns
        tuner.untunedf = pd.DataFrame([row], index=[7])
        tuned_row = {**row, **dict.fromkeys(result_columns, 0)}
        tuned_row.update(
            kernelName1="existing_gateup", kernelName2="existing_down", us=baseline_us
        )
        if not exact:
            tuned_row["act_type"] = str(ActivationType.Swiglu)
        observed = {}

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "tuned.csv")
            existing = pd.DataFrame([tuned_row], columns=tuner.columns)
            if empty_output:
                Path(output).touch()
                existing = pd.DataFrame(columns=tuner.columns)
                original = b""
            elif output_exists:
                existing.to_csv(output, index=False)
                original = Path(output).read_bytes()
            else:
                existing = pd.DataFrame(columns=tuner.columns)
                original = None

            def baseline(_args, shapes, config_file=None):
                observed["config_file"] = config_file
                observed["baseline_index"] = shapes.index.tolist()
                return [
                    {"e2e_us": baseline_us, "err_ratio": 0.0, "status": baseline_status}
                ]

            def candidate(_args, target_fused_moe=None, config_string=""):
                observed["candidate_index"] = tuner.untunedf.index.tolist()
                return [{"e2e_us": candidate_us, "err_ratio": 0.0, "status": "ok"}]

            with (
                patch.object(tuner, "get_out_file", return_value=output),
                patch.object(tuner, "get_tuned_gemm_list", return_value=existing),
                patch.object(tuner, "_run_config_for_shapes", side_effect=baseline),
                patch.object(tuner, "run_config", side_effect=candidate),
                patch.object(tuner, "calculate", return_value=(1.0, 1.0)),
                patch.object(self.module, "get_gfx", return_value="gfx942"),
                patch.object(
                    self.module, "_TUNE_EXCLUDE_KERNEL_PATTERNS", exclude_patterns
                ),
                patch.object(
                    backend, "get_tune_space", return_value=["16_16_16_False_True"]
                ),
                contextlib.redirect_stdout(io.StringIO()) as messages,
            ):
                tuner.e2e_tune(SimpleNamespace(tune_file=output))
            final = Path(output).read_bytes() if os.path.exists(output) else None
            frame = pd.read_csv(output) if final else None
            observed.update(
                original=original,
                final=final,
                frame=frame,
                messages=messages.getvalue(),
                output=output,
            )
        self.assertEqual(tuner.untunedf.index.tolist(), [7])
        self.assertEqual(observed["candidate_index"], [0])
        return observed

    def test_slower_candidate_does_not_replace_measured_fallback(self):
        result = self.run_search()
        self.assertEqual(result["final"], result["original"])
        self.assertIn("e2e_status=kept_baseline", result["messages"])

    def test_faster_candidate_adds_exact_row_without_dropping_other_activation(self):
        result = self.run_search(candidate_us=8.0)
        self.assertEqual(len(result["frame"]), 2)
        selected = result["frame"][
            result["frame"]["act_type"] == str(ActivationType.Silu)
        ].iloc[0]
        self.assertEqual(
            selected["kernelName1"], "impl__flydsl_gfx942__16_16_16_False_True"
        )
        self.assertEqual(selected["us"], 8.0)

    def test_invalid_baseline_allows_valid_candidate(self):
        result = self.run_search(
            baseline_status="error:nonfinite output", baseline_us=-1.0
        )
        self.assertEqual(len(result["frame"]), 2)
        self.assertIn("e2e_status=updated", result["messages"])

    def test_slower_candidate_preserves_exact_tuned_row(self):
        result = self.run_search(exact=True)
        self.assertEqual(result["final"], result["original"])
        self.assertEqual(result["config_file"], result["output"])

    def test_missing_output_uses_default_config_and_keeps_faster_baseline(self):
        result = self.run_search(output_exists=False)
        self.assertIsNone(result["config_file"])
        self.assertIsNone(result["final"])

    def test_missing_output_can_write_a_faster_candidate(self):
        result = self.run_search(output_exists=False, candidate_us=8.0)
        self.assertIsNone(result["config_file"])
        self.assertEqual(len(result["frame"]), 1)

    def test_empty_output_uses_default_config_and_keeps_faster_baseline(self):
        result = self.run_search(empty_output=True)
        self.assertIsNone(result["config_file"])
        self.assertEqual(result["final"], b"")

    def test_unknown_fallback_cannot_override_kernel_exclusions(self):
        result = self.run_search(exclude_patterns=("existing_",))
        self.assertEqual(len(result["frame"]), 2)
        self.assertIn("e2e_status=updated", result["messages"])


if __name__ == "__main__":
    unittest.main()
