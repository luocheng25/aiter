#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sustained clock/power ceilings for the MiMo 32K MoE on this GPU.

Replays the Gate/Up and Down launches captured from one real MoE call with random
and zero operands, next to hipBLASLt FP8 GEMMs and streaming copies. Each case runs
about a second of back-to-back work while rocm-smi samples clock and package power.

  python mytest/power_probe.py --json mytest/results/b10_power_probe.json
"""

import argparse
import importlib
import json
import statistics
import subprocess
import threading
from unittest.mock import patch

import torch

from bench_mimo import EXPERTS, HIDDEN, INTER, TOPK, make_inputs, run

TOKENS = 32768
ROUTES = TOKENS * TOPK
GATEUP_FLOPS = 4 * ROUTES * HIDDEN * INTER
DOWN_IDEAL_BYTES = (
    ROUTES * INTER + ROUTES * 4 + EXPERTS * HIDDEN * INTER + EXPERTS * HIDDEN * 4
    + ROUTES * HIDDEN * 2
)


def status():
    card = json.loads(subprocess.check_output(
        ["rocm-smi", "--showclocks", "--showpower", "--showuse", "--json"],
        text=True, stderr=subprocess.DEVNULL,
    ))["card0"]
    return (
        float(card["sclk clock speed:"].strip("()Mhz")),
        float(card["Current Socket Graphics Package Power (W)"]),
        float(card["GPU use (%)"]),
    )


def sustained(call, seconds):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        call()
    begin.record()
    for _ in range(5):
        call()
    end.record()
    end.synchronize()
    iters = max(20, int(seconds * 1e3 * 5 / begin.elapsed_time(end)))
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            samples.append(status())

    # A quarter of the run settles the power controller before the timed part.
    for _ in range(iters // 4):
        call()
    sampler = threading.Thread(target=sample)
    sampler.start()
    begin.record()
    for _ in range(iters):
        call()
    end.record()
    end.synchronize()
    stop.set()
    sampler.join()
    # The last sample may land after the work drained; keep busy samples only.
    busy = [s for s in samples[:-1] if s[2] >= 90] or samples
    return dict(
        us=begin.elapsed_time(end) * 1e3 / iters, iters=iters, samples=len(busy),
        mhz=statistics.median(s[0] for s in busy), watts=statistics.median(s[1] for s in busy),
    )


def capture_moe(inputs, config):
    backend = importlib.import_module("aiter.ops.flydsl.fused_moe_gfx942")
    launch = backend._launch
    captured = {}

    def record(kernel, *arguments):
        result = launch(kernel, *arguments)
        for name, weight in (("gateup", inputs["w1"]), ("down", inputs["w2"])):
            if arguments[1] is weight and name not in captured:
                captured[name] = (kernel, list(arguments))
        return result

    with patch.object(backend, "_launch", record):
        run(inputs, config, 0)
    torch.cuda.synchronize()

    def replay(name, zero_activation=False, zero_weight=False):
        kernel, arguments = captured[name]
        arguments = list(arguments)
        if zero_activation:
            arguments[0] = torch.zeros_like(arguments[0])
        if zero_weight:
            arguments[1] = torch.zeros_like(arguments[1])
        return lambda: launch(kernel, *arguments)

    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="64_256_128_True:8x1_compact:128")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--cases", nargs="*", default=None, help="run only these case names")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    torch.set_default_device("cuda")
    replay = capture_moe(make_inputs(TOKENS, 1), args.config)
    one = torch.tensor(1.0)
    fp8 = torch.float8_e4m3fnuz

    def gemm(m, n, k, fill):
        a = fill(m, k, dtype=torch.bfloat16).to(fp8)
        b = fill(n, k, dtype=torch.bfloat16).to(fp8)
        return lambda: torch._scaled_mm(a, b.t(), scale_a=one, scale_b=one,
                                        out_dtype=torch.bfloat16)

    fill_out = torch.empty(ROUTES, HIDDEN, dtype=torch.bfloat16)
    copy_in = torch.randn(ROUTES // 2, HIDDEN, dtype=torch.bfloat16)
    copy_out = torch.empty_like(copy_in)
    cases = {
        "gateup_random": (replay("gateup"), "gateup"),
        "gateup_zero_weight": (replay("gateup", zero_weight=True), "gateup"),
        "gateup_zero_both": (replay("gateup", True, True), "gateup"),
        "down_random": (replay("down"), "down"),
        "down_zero_both": (replay("down", True, True), "down"),
        "hipblaslt_routes_x512x6144_random": (gemm(ROUTES, 2 * INTER, HIDDEN, torch.randn), "gateup"),
        "hipblaslt_8192_random": (gemm(8192, 8192, 8192, torch.randn), 2 * 8192**3),
        "hipblaslt_8192_zero": (gemm(8192, 8192, 8192, torch.zeros), 2 * 8192**3),
        "fill_routes_x6144_bf16": (lambda: fill_out.fill_(1.0), fill_out.numel() * 2),
        "copy_half_routes_x6144_bf16": (lambda: copy_out.copy_(copy_in), copy_in.numel() * 4),
    }
    if args.cases:
        cases = {name: cases[name] for name in args.cases}
    results = {}
    for round_index in range(args.rounds):
        names = list(cases) if round_index % 2 == 0 else list(cases)[::-1]
        for name in names:
            call, metric = cases[name]
            record = sustained(call, args.seconds)
            if metric == "gateup":
                record["useful_TFLOPS"] = GATEUP_FLOPS / record["us"] / 1e6
            elif metric == "down":
                record["ideal_TBps"] = DOWN_IDEAL_BYTES / record["us"] / 1e6
            elif name.startswith("hipblaslt"):
                record["TFLOPS"] = metric / record["us"] / 1e6
            else:
                record["TBps"] = metric / record["us"] / 1e6
            results.setdefault(name, []).append(record)
            print(round_index, name, json.dumps(record), flush=True)
    if args.json:
        with open(args.json, "w") as output:
            json.dump(dict(config=args.config, results=results), output, indent=1)


if __name__ == "__main__":
    main()
