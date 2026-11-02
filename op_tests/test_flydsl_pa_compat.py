# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility contracts for PA on the retained FlyDSL 0.3.2 runtime."""

import importlib
from contextlib import contextmanager

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.pa_decode import compat
from aiter.ops.flydsl.kernels.pa_decode_plan import PADecodePlan
from op_tests.test_flydsl_pa_decode import _make_inputs, _reference


@pytest.mark.parametrize("size", [0, 65536])
def test_legacy_lds_limit_accepts_supported_sizes(monkeypatch, size):
    monkeypatch.setattr(compat, "_native_check_smem_capacity", None)
    compat.check_smem_capacity(size, "gfx942")


@pytest.mark.parametrize("size", [-1, 65537, True, 1.5])
def test_legacy_lds_limit_rejects_invalid_sizes(monkeypatch, size):
    monkeypatch.setattr(compat, "_native_check_smem_capacity", None)
    with pytest.raises(ValueError, match="64 KiB"):
        compat.check_smem_capacity(size, "gfx942")


def test_legacy_lds_limit_does_not_claim_other_architectures(monkeypatch):
    monkeypatch.setattr(compat, "_native_check_smem_capacity", None)
    with pytest.raises(NotImplementedError, match="gfx942"):
        compat.check_smem_capacity(4096, "gfx950")


def test_native_lds_checker_is_preferred(monkeypatch):
    calls = []
    monkeypatch.setattr(
        compat, "_native_check_smem_capacity", lambda *args: calls.append(args)
    )
    compat.check_smem_capacity(131072, "gfx1250")
    assert calls == [(131072, "gfx1250")]


def test_missing_tdm_fails_explicitly():
    with pytest.raises(NotImplementedError, match="TDM operation"):
        compat._UnavailableTdm().make_tile_tdm_atom()


@pytest.mark.parametrize("splits", [0, -1, True, 1.5, 3, "2"])
def test_query_splits_rejects_invalid_values_before_tensor_work(splits):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    with pytest.raises(ValueError, match="query_splits"):
        pa.pa_decode(
            None, None, None, None, None, None, 1.0, 4,
            work_plan=object.__new__(PADecodePlan), query_splits=splits,
        )


def _inputs(kv_heads=2, group_size=8):
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx942":
        pytest.skip("FlyDSL 0.3.2 PA execution is qualified on gfx942")
    torch.manual_seed(5887)
    query, key, value, table, lengths = _make_inputs(
        4, kv_heads, group_size, 192, 128, 64, torch.bfloat16,
        "random", "cuda", (129, 257)
    )
    key, value = key.bfloat16(), value.bfloat16()
    pages = key.shape[0]
    packed_key = key.view(pages, kv_heads, 64, 24, 8).permute(0, 1, 3, 2, 4).contiguous()
    packed_value = value.view(pages, kv_heads, 8, 8, 128).permute(0, 1, 2, 4, 3).contiguous()
    sinks = torch.linspace(-3, 3, 16, device="cuda", dtype=torch.float32)
    return query, key, value, packed_key, packed_value, table, lengths, sinks


@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("kv_heads,window", [(1, 0), (2, 0), (2, 128)])
def test_query_splits_reaches_compiler_and_replays(
    monkeypatch, splits, kv_heads, window
):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    group_size = 16 // kv_heads
    query, key, value, packed_key, packed_value, table, lengths, sinks = _inputs(
        kv_heads, group_size
    )
    plan = pa.plan_pa_decode(lengths, kv_heads, max_partitions=1,
                            sliding_window=window, query_length=4)
    output = torch.empty(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    shape = (kv_heads, plan.capacity, 4 * group_size)
    pmax = torch.empty(shape, device="cuda", dtype=torch.float32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*shape, 128, device="cuda", dtype=torch.bfloat16)
    compile_kernel = pa.compile_pa_decode_tile
    observed = []

    def compile_with_capture(**kwargs):
        observed.append(kwargs["query_splits"])
        return compile_kernel(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", compile_with_capture)

    def run():
        pa.plan_pa_decode(lengths, kv_heads, plan=plan,
                          sliding_window=window, query_length=4)
        pa.pa_decode(
            output, query, packed_key, packed_value, lengths, table, 192**-0.5, 4,
            work_plan=plan, max_context_length=320, sinks=sinks,
            sliding_window=window, query_splits=splits,
            max_logits=pmax, exp_sums=psum, temporary_output=pout,
        )

    def check():
        expected = _reference(query, key.float(), value.float(), table,
                              lengths, 4, window, sinks)
        torch.testing.assert_close(output.float(), expected, atol=0.005, rtol=0.025)

    run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths.copy_(torch.tensor([257, 129], dtype=torch.int32, device="cuda"))
    table.copy_(table.flip(0).clone())
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        check()
    assert observed and all(value == splits for value in observed)


def test_legacy_autotune_resolves_without_default_launch_and_validates_search(
    monkeypatch, tmp_path
):
    tuning = importlib.import_module("aiter.ops.flydsl.pa_decode_tuning")
    query, _, _, key, value, table, lengths, sinks = _inputs()
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")
    prepare = tuning._PADecodeAutotuneResources.prepare
    calls = []

    def track_prepare(self, budget):
        calls.append(budget)
        return prepare(self, budget)

    monkeypatch.setattr(tuning._PADecodeAutotuneResources, "prepare", track_prepare)
    args = (query, key, value, lengths, table, 192**-0.5, 4)
    options = dict(sliding_window=128, sinks=sinks, max_partitions=1,
                   max_context_length=320, config_dir=tmp_path)
    plan = tuning.prepare_pa_decode_plan(*args, **options)
    assert plan.capacity == 2 and calls == []
    tuning.prepare_pa_decode_plan(*args, **options, tune=True)
    assert len(calls) >= 2
    assert (tmp_path / "gfx942" / "_pa_decode_autotuner.json").is_file()
    calls.clear()
    tuning.prepare_pa_decode_plan(*args, **options)
    assert calls == []


def test_legacy_autotune_does_not_accept_failed_validation(monkeypatch, tmp_path):
    tuning = importlib.import_module("aiter.ops.flydsl.pa_decode_tuning")
    if hasattr(tuning._pa_decode_autotuner, "resolve_config"):
        pytest.skip("This case exercises the older native autotuner compatibility path")
    query, _, _, key, value, table, lengths, sinks = _inputs()
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "0")

    @contextmanager
    def reject_candidate(arguments):
        yield
        raise AssertionError("injected candidate validation failure")

    monkeypatch.setattr(tuning, "_validate_pa_decode_config", reject_candidate)
    with pytest.raises(RuntimeError, match="failed validation"):
        tuning.prepare_pa_decode_plan(
            query, key, value, lengths, table, 192**-0.5, 4,
            sliding_window=128, sinks=sinks, max_partitions=1,
            max_context_length=320, config_dir=tmp_path, tune=True,
        )
    assert not (tmp_path / "gfx942" / "_pa_decode_autotuner.json").exists()
