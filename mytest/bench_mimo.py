#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MiMo-V2.5-Pro MoE (6144/256/E384/top8, FP8 PTPC) on the gfx942 FlyDSL backend.

Per-kernel GPU time from torch.profiler, L2<->fabric traffic from rocprofv3
FetchSize/WriteSize (separate passes), and the accuracy checks of
op_tests/test_flydsl_moe_gfx942.py against the torch reference.

  python mytest/bench_mimo.py --token 32768 --configs 64_256_128_True:8x1_compact:128
  python mytest/bench_mimo.py --token 32768 --configs A B --rounds 3 --pmc
"""

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942
from aiter.ops.shuffle import shuffle_weight

HIDDEN, INTER, EXPERTS, TOPK = 6144, 256, 384, 8
ELEMENT_ERR_LIMIT, TOKEN_ERR_LIMIT, LOGITS_DIFF_LIMIT = 1.0, 0.2, 1e-3

# Ordered (role, substring) rules; the first match names a kernel.
ROLES = (
    ("sort", "moe_sorting"),
    ("quant", "per_token"),
    ("quant", "quant"),
    ("gateup", "gateup"),
    ("compact", "compact_m64_tasks"),
    ("down_full", "down_prefill_8x1"),
    ("down", "down_prefill_1x4_64x256"),
    ("down", "down_m128"),
    ("down", "moe_down"),
    ("down", "down_prefill"),
    ("inverse", "invert_sorted_ids"),
    ("reduce", "sorted_sum"),
)


def role_of(name, index, roles_seen):
    for role, key in ROLES:
        if key in name:
            if role == "quant":
                role = "quant_input" if "gateup" not in roles_seen else "quant_inter"
            if role == "down" and "down_full" in roles_seen:
                role = "down_tail"
            return role
    return name[:40]


def make_inputs(token, copies, seed=0):
    torch.manual_seed(seed)
    quant = aiter.get_torch_quant(QuantType.per_Token)
    w1 = torch.randn((EXPERTS, 2 * INTER, HIDDEN), dtype=dtypes.bf16)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp8)
    del w1
    w2 = torch.randn((EXPERTS, HIDDEN, INTER), dtype=dtypes.bf16)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp8)
    del w2
    sets = []
    for _ in range(copies):
        hidden = torch.randn((token, HIDDEN), dtype=dtypes.bf16)
        gating = torch.randn((token, EXPERTS), dtype=dtypes.bf16)
        weights, ids = fused_topk(hidden, gating, TOPK, True)
        sets.append((hidden, weights.contiguous(), ids.contiguous()))
    w1_s = shuffle_weight(w1_q, layout=(16, 16))
    w2_s = shuffle_weight(w2_q, layout=(16, 16))
    return dict(w1_q=w1_q, w1_scale=w1_scale, w2_q=w2_q, w2_scale=w2_scale,
                w1=w1_s, w2=w2_s, sets=sets)


def reference(inputs):
    hidden, weights, ids = inputs["sets"][0]
    quant = aiter.get_torch_quant(QuantType.per_Token)
    a1, a1_scale = quant(hidden, quant_dtype=dtypes.fp8)
    stage1 = torch_moe_stage1(a1, inputs["w1_q"], inputs["w2_q"], weights, ids,
                              dtype=dtypes.bf16, activation=ActivationType.Silu,
                              quant_type=QuantType.per_Token, a1_scale=a1_scale,
                              w1_scale=inputs["w1_scale"])
    a2, a2_scale = quant(stage1, quant_dtype=dtypes.fp8)
    out = torch_moe_stage2(a2.view(hidden.shape[0], TOPK, -1), inputs["w1_q"],
                           inputs["w2_q"], weights, ids, dtype=dtypes.bf16,
                           quant_type=QuantType.per_Token, w2_scale=inputs["w2_scale"],
                           a2_scale=a2_scale)
    torch.cuda.empty_cache()
    return out


def run(inputs, config, index):
    hidden, weights, ids = inputs["sets"][index % len(inputs["sets"])]
    return run_flydsl_moe_gfx942(hidden, inputs["w1"], inputs["w2"], weights, ids,
                                 ActivationType.Silu, QuantType.per_Token,
                                 inputs["w1_scale"], inputs["w2_scale"], None, None, 0,
                                 config)


def check(ref, out):
    """The element, logits and token checks of test_flydsl_moe_gfx942.check_output."""
    assert bool(torch.isfinite(out).all()), "non-finite output"
    rms = ref.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    element = float(((out.float() - ref.float()).abs() / rms).max())
    bad = float(((out.float() - ref.float()).abs() / rms > ELEMENT_ERR_LIMIT).float().mean())
    x, y = ref.double(), out.double()
    logits = float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())
    token = float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max())
    ok = bad == 0 and logits < LOGITS_DIFF_LIMIT and token < TOKEN_ERR_LIMIT
    return dict(ok=ok, max_rms_err=element, bad_ratio=bad, logits_diff=logits,
                token_err=token)


def profile_config(inputs, config, iters, warmup, gap_cycles=0):
    from torch.profiler import ProfilerActivity, profile

    for i in range(warmup):
        run(inputs, config, i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            run(inputs, config, i)
            if gap_cycles:
                torch.cuda._sleep(gap_cycles)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name in ("CUDA", "HIP")
               and e.time_range.elapsed_us() >= 0 and "spin_kernel" not in e.name]
    kernels.sort(key=lambda e: e.time_range.start)
    per_iter, rem = divmod(len(kernels), iters)
    assert per_iter and rem == 0, f"{len(kernels)} kernels for {iters} iterations"
    names = [kernels[i].name for i in range(per_iter)]
    times = [[kernels[it * per_iter + i].time_range.elapsed_us() for it in range(iters)]
             for i in range(per_iter)]
    totals = [sum(times[i][it] for i in range(per_iter)) for it in range(iters)]
    rows, seen = [], []
    for i, name in enumerate(names):
        role = role_of(name, i, seen)
        seen.append(role)
        rows.append(dict(role=role, name=name, us=statistics.median(times[i])))
    return dict(total_us=statistics.median(totals), kernels=rows)


def pmc_child(args):
    inputs = make_inputs(args.token, 1)
    for i in range(3):
        run(inputs, args.configs[0], 0)
    torch.cuda.synchronize()
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    run(inputs, args.configs[0], 0)
    torch.cuda.synchronize()


def read_counters(directory):
    files = glob.glob(os.path.join(directory, "**", "*counter_collection.csv"), recursive=True)
    assert files, f"no counter csv under {directory}"
    dispatches = {}
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            key = int(row["Dispatch_Id"])
            entry = dispatches.setdefault(key, dict(name=row["Kernel_Name"], counters={}))
            entry["counters"][row["Counter_Name"]] = (
                entry["counters"].get(row["Counter_Name"], 0.0) + float(row["Counter_Value"]))
    ordered = [dispatches[k] for k in sorted(dispatches)]
    last_marker = max(i for i, d in enumerate(ordered) if "spin_kernel" in d["name"])
    return ordered[last_marker + 1:]


def pmc_config(args, config, counters=("FetchSize", "WriteSize")):
    result = {}
    for counter in counters:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["rocprofv3", "--pmc", counter, "--output-format", "csv", "-d", tmp,
                   "-o", "pmc", "--", sys.executable, os.path.abspath(__file__),
                   "--pmc-child", "--token", str(args.token), "--configs", config]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for i, d in enumerate(read_counters(tmp)):
                entry = result.setdefault(i, dict(name=d["name"]))
                scale = 1024 if counter in ("FetchSize", "WriteSize") else 1
                entry[counter] = d["counters"].get(counter, 0.0) * scale
    return [result[i] for i in sorted(result)]


def report(token, config, prof, traffic, accuracy):
    print(f"\n### token={token} config={config}")
    print(f"accuracy: {json.dumps(accuracy)}")
    print(f"GPU total us (median): {prof['total_us']:.3f}")
    rows = prof["kernels"]
    if traffic is not None and len(traffic) != len(rows):
        print(f"warning: {len(traffic)} PMC dispatches vs {len(rows)} profiled kernels")
        traffic = None
    print("| role | us | read GB | write GB | TB/s | kernel |")
    print("|---|---:|---:|---:|---:|---|")
    for i, row in enumerate(rows):
        if traffic is not None:
            rd, wr = traffic[i]["FetchSize"] / 1e9, traffic[i]["WriteSize"] / 1e9
            bw = (rd + wr) * 1e9 / row["us"] / 1e6
            extra = f"{rd:.4f} | {wr:.4f} | {bw:.3f}"
        else:
            extra = "- | - | -"
        print(f"| {row['role']} | {row['us']:.3f} | {extra} | {row['name'][:70]} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", type=int, default=32768)
    parser.add_argument("--configs", nargs="+", default=["64_256_128_True:8x1_compact:128"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--gap-cycles", type=int, default=0,
                        help="torch.cuda._sleep cycles between iterations (excluded from times)")
    parser.add_argument("--pmc", action="store_true")
    parser.add_argument("--pmc-counters", nargs="*", default=None,
                        help="extra counters (one rocprofv3 pass each), printed for GEMM kernels")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--pmc-child", action="store_true")
    args = parser.parse_args()
    torch.set_default_device("cuda")
    if args.pmc_child:
        return pmc_child(args)

    inputs = make_inputs(args.token, args.copies)
    ref = None if args.no_check else reference(inputs)
    results = {c: dict(profiles=[]) for c in args.configs}
    for c in args.configs:
        out = run(inputs, c, 0)
        torch.cuda.synchronize()
        results[c]["accuracy"] = None if ref is None else check(ref, out)
    for r in range(args.rounds):
        order = args.configs if r % 2 == 0 else args.configs[::-1]
        for c in order:
            results[c]["profiles"].append(
                profile_config(inputs, c, args.iters, args.warmup, args.gap_cycles))
    for c in args.configs:
        profiles = results[c]["profiles"]
        best = sorted(profiles, key=lambda p: p["total_us"])[len(profiles) // 2]
        results[c]["median_profile"] = best
        results[c]["round_totals"] = [p["total_us"] for p in profiles]
        if args.pmc:
            results[c]["traffic"] = pmc_config(args, c)
        report(args.token, c, best, results[c].get("traffic"), results[c]["accuracy"])
        print(f"round totals: {[round(t, 3) for t in results[c]['round_totals']]}")
        if args.pmc_counters:
            extra = pmc_config(args, c, args.pmc_counters)
            results[c]["counters"] = extra
            for row, entry in zip(best["kernels"], extra):
                if row["role"].startswith(("gateup", "down")):
                    values = ", ".join(f"{k}={entry.get(k, 0):.4g}" for k in args.pmc_counters)
                    print(f"  {row['role']}: {values}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(token=args.token, results=results), f, indent=1)


if __name__ == "__main__":
    main()
