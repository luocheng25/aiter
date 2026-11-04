#!/usr/bin/env python3
"""MiMo Flash DFlash8 paged-attention performance benchmark for MI300X/gfx942.

Run with the same AITER/FlyDSL/Torch environment as the server. Example:
  python3 op_tests/op_benchmarks/flydsl/bench_mimo_pa_decode.py \
      --attn-type full swa draft --output /tmp/pa-run

Omit --output to write results into the current directory. Existing directories
are accepted, and result files from previous runs are overwritten.
An aligned Markdown summary is printed at the end and saved in README.md.
Uses the current environment's AITER FlyDSL PA API. Full attention selects
cached/calibrated P/QS with prepare_pa_decode_plan before warmup and capture.
There is no online tuning search; explicit partition overrides use a fixed plan.

Measures one layer on one GPU, with local attention BS (not DP-gathered MoE M).
Defaults use the ten ISL/BS pairs, BF16 packed page64 caches, physical
pool extents and table bounds used by the PR36 TP8/DP2 decode campaign.
Synthetic inputs exclude QKV projection/cache writes, table construction,
communication and scheduling. Two timings separate PA+reduction from
plan-refresh+PA+reduction. Neither is whole-model decode latency.
Accuracy and dynamic replay tests live in op_tests/test_flydsl_pa_decode.py.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import statistics
import time


# ISL, local BS, full/draft pool token capacity, target SWA/full ratio.
CASES = [
    (4096, 256, 10513664, 0.03),
    (8192, 256, 10513664, 0.03),
    (16384, 240, 10513664, 0.03),
    (32768, 208, 10513664, 0.03),
    (65536, 152, 10513664, 0.03),
    (131072, 96, 12863872, 0.012),
    (262144, 40, 12863872, 0.012),
    (524288, 24, 13211264, 0.003),
    (786432, 12, 10513664, 0.03),
    (1040000, 12, 13211264, 0.003),
]
PAGE, QLEN, HQ, DV, CONTEXT_BOUND = 64, 8, 16, 128, 1048576


def geometry(case, family, args):
    isl, bs, full_pool, ratio = case
    if family == "full":
        hkv, dk, window, length = 1, 192, 0, isl + QLEN
        pool, width = full_pool, CONTEXT_BOUND // PAGE
    elif family == "swa":
        # The grouped adapter retains the final four pages. At aligned ISL,
        # three complete pages plus eight new tokens give compact length200.
        hkv, dk, window = 2, 192, 128
        length = 3 * PAGE + (isl + QLEN - 1) % PAGE + 1
        pool, width = int(full_pool * ratio) // PAGE * PAGE, 4
    else:
        # Aligned draft entry: retained prefix1024 followed by eight queries.
        hkv, dk, window = 2, 128, 1023
        length = 1024 + QLEN
        pool, width = full_pool, CONTEXT_BOUND // PAGE
    pages_per_seq = math.ceil(length / PAGE)
    if pages_per_seq > width or bs * pages_per_seq + 2 > pool // PAGE:
        raise ValueError("Batch size exceeds the case's KV pool or page table")
    override = args.case_overrides.get(family, {}).get(str(isl), {})
    return dict(family=family, isl=isl, bs_per_dp=bs, global_c=2 * bs,
                local_query_tokens=bs * QLEN, qlen=QLEN, hq=HQ, hkv=hkv,
                dk=dk, dv=DV, page=PAGE, length=length, window=window,
                pool_tokens=pool, table_width=width,
                partitions=override.get("partitions", getattr(args, family + "_partitions")),
                query_splits=override.get("query_splits", args.query_splits))


def make_inputs(torch, spec, args):
    b, h, d, length = (spec[k] for k in ("bs_per_dp", "hkv", "dk", "length"))
    pages = spec["pool_tokens"] // PAGE
    used = math.ceil(length / PAGE)
    q = torch.randn(b * QLEN, HQ, d, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((pages, h, d // 8, PAGE, 8), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((pages, h, PAGE // 8, DV, 8), device="cuda", dtype=torch.bfloat16)
    # Spread live pages across the real allocation, exercising high addresses.
    ids = torch.arange(b * used, device="cuda")
    ids = 2 + (ids * (pages - 2) // (b * used) if args.page_layout == "spread" else ids)
    if args.kv_pattern == "random":
        for begin in range(0, ids.numel(), 4096):
            selected = ids[begin:begin + 4096]
            k[selected] = torch.randn(selected.numel(), h, d // 8, PAGE, 8, device="cuda", dtype=k.dtype)
            v[selected] = torch.randn(selected.numel(), h, PAGE // 8, DV, 8, device="cuda", dtype=v.dtype)
    tables = torch.zeros((b, spec["table_width"]), device="cuda", dtype=torch.int32)
    tables[:, :used] = ids.reshape(b, used).int()
    if length % PAGE:
        last = ids.reshape(b, used)[:, -1]
        tail = torch.arange(length % PAGE, PAGE, device="cuda")
        k[last[:, None], :, :, tail[None, :], :] = 0
        v[last[:, None], :, tail[None, :] // 8, :, tail[None, :] % 8] = 0
    lengths = torch.full((b,), length, device="cuda", dtype=torch.int32)
    sinks = torch.randn(HQ, device="cuda", dtype=torch.float32) if spec["window"] else None
    out = torch.empty(b * QLEN, HQ, DV, device="cuda", dtype=torch.bfloat16)
    return dict(q=q, k=k, v=v, lengths=lengths, tables=tables, sinks=sinks, out=out)


def measure(torch, pa, spec, args):
    data = make_inputs(torch, spec, args)
    requested_partitions, requested_splits = spec["partitions"], spec["query_splits"]
    if requested_partitions is None:
        from aiter.ops.flydsl.pa_decode_tuning import prepare_pa_decode_plan

        plan = prepare_pa_decode_plan(
            data["q"], data["k"], data["v"], data["lengths"], data["tables"],
            1 / math.sqrt(spec["dk"]), QLEN, sinks=data["sinks"],
            sliding_window=spec["window"], max_batch_size=spec["bs_per_dp"],
            max_context_length=spec["table_width"] * PAGE,
            query_splits=requested_splits, cache_only=True,
        )
    else:
        plan = pa.plan_pa_decode(data["lengths"], spec["hkv"], max_partitions=requested_partitions,
                                max_batch_size=spec["bs_per_dp"],
                                workgroup_budget=spec["bs_per_dp"] * spec["hkv"] * requested_partitions,
                                sliding_window=spec["window"], query_length=QLEN)
    spec = dict(spec, partitions=plan.max_partitions,
                query_splits=requested_splits if requested_splits is not None else plan.query_splits)
    shape = (spec["hkv"], plan.capacity, QLEN * HQ // spec["hkv"])
    pmax = torch.empty(shape, device="cuda", dtype=torch.float32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*shape, DV, device="cuda", dtype=torch.bfloat16)

    def decode():
        pa.pa_decode(data["out"], data["q"], data["k"], data["v"], data["lengths"], data["tables"],
                     1 / math.sqrt(spec["dk"]), QLEN, work_plan=plan,
                     sliding_window=spec["window"], sinks=data["sinks"],
                     max_context_length=spec["table_width"] * PAGE,
                     exp_sums=psum, max_logits=pmax, temporary_output=pout,
                     query_splits=spec["query_splits"], mask_unowned_kv=True)

    def refresh_decode():
        pa.plan_pa_decode(data["lengths"], spec["hkv"], plan=plan, sliding_window=spec["window"], query_length=QLEN)
        decode()

    for _ in range(10):
        refresh_decode()
    torch.cuda.synchronize()
    graphs = {}
    # Capture on a side stream after eager compilation/warmup.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name, fn in (("decode_reduce", decode), ("plan_decode_reduce", refresh_decode)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graph.replay()
            stream.synchronize()
            graphs[name] = graph
    torch.cuda.current_stream().wait_stream(stream)
    warmup = {}
    for name, graph in graphs.items():
        start, count = time.perf_counter(), 0
        while count < 200 or time.perf_counter() - start < args.warmup_seconds:
            for _ in range(50):
                graph.replay()
            torch.cuda.synchronize()
            count += 50
        warmup[name] = dict(replays=count, seconds=time.perf_counter() - start)
    rounds = []
    for i in range(args.rounds):
        order = list(graphs) if i % 2 == 0 else list(reversed(graphs))
        row = {}
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.repeat):
                graphs[name].replay()
            end.record()
            end.synchronize()
            row[name] = 1000 * begin.elapsed_time(end) / args.repeat
        rounds.append(row)
    values = {name: [r[name] for r in rounds] for name in graphs}
    return dict(**spec, requested_partitions=requested_partitions, requested_query_splits=requested_splits,
                plan_selection="cache/default" if requested_partitions is None else "fixed",
                plan_capacity=plan.capacity, warmup=warmup, rounds_us=rounds,
                median_us={n: statistics.median(v) for n, v in values.items()},
                cv_percent={n: 100 * statistics.pstdev(v) / statistics.mean(v) for n, v in values.items()})


def summary_markdown(results):
    headers = ["Attn-type", "ISL", "BS/DP", "BS×8", "max_partitions", "query split",
               "PA + reduce µs", "Plan + PA + reduce µs"]
    rows = []
    for r in results:
        splits = r["query_splits"]
        rows.append([r["family"], f"{r['isl']:,}", str(r["bs_per_dp"]),
                     f"{r['local_query_tokens']:,}", str(r["partitions"]),
                     str(splits) if splits is not None else "auto",
                     f"{r['median_us']['decode_reduce']:.2f}",
                     f"{r['median_us']['plan_decode_reduce']:.2f}"])
    widths = [max(4, *(len(row[i]) for row in [headers, *rows])) for i in range(len(headers))]

    def line(row):
        return "| " + " | ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                                  for i, value in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * width if i == 0 else "-" * (width - 1) + ":"
                                    for i, width in enumerate(widths)) + " |"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


def write_report(report, directory):
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["family", "isl", "bs_per_dp", "global_c", "local_query_tokens", "length", "pool_tokens", "table_width", "partitions", "query_splits", "decode_reduce_us", "plan_decode_reduce_us"]
    with (directory / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in report["results"]:
            writer.writerow({**{k: r[k] for k in fields[:-2]}, **{k + "_us": v for k, v in r["median_us"].items()}})
    lines = ["# MiMo Flash DFlash8 PA decode performance", "", f"Status: **{report['status']}**.", "",
             "One layer on one MI300X: BF16 Q/K/V, page64, Q8. Unused KV and page tails are zero-filled; masking stays enabled. Setup and compilation are outside timing. Each mode receives at least 200 warmup replays and the configured warmup duration. Values are medians of alternating graph timing rounds.", "",
             summary_markdown(report["results"]), "",
             "max_partitions is the selected per-sequence partition limit. query split comes from the selected plan or explicit override; query split=auto delegates to the compiler. Automatic selection reads cached/calibrated settings before warmup and capture, with no online search. Plan + PA + reduce includes only the GPU plan-refresh kernel, PA and reduction. These are component timings, excluding projection, KV writes, MoE, communication and scheduling.", "",
             "Geometry, warmup counts and all timing intervals are in [results.json](results.json). Accuracy and dynamic replay coverage are in op_tests/test_flydsl_pa_decode.py."]
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attn-type", nargs="+", choices=("full", "swa", "draft"), default=["full"])
    parser.add_argument("--isls", nargs="+", type=int, choices=[c[0] for c in CASES])
    parser.add_argument("--batch-sizes", nargs="+", type=int, help="Override local BS for a fixed-ISL scaling sweep")
    parser.add_argument("--case-config", type=Path, help="JSON family/ISL mapping to partitions and query_splits")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--page-layout", choices=("spread", "contiguous"), default="spread")
    parser.add_argument("--kv-pattern", choices=("random", "zero"), default="random")
    parser.add_argument("--full-partitions", type=int,
                        help="Fixed full-attention partition cap; default uses AITER cached/calibrated P/QS")
    parser.add_argument("--swa-partitions", type=int, default=1)
    parser.add_argument("--draft-partitions", type=int, default=8)
    parser.add_argument("--query-splits", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--list", action="store_true", help="Print geometry without importing Torch or using GPUs")
    parser.add_argument("--output", type=Path, default=Path("."),
                        help="Result directory (default: current directory); overwrites existing result files")
    args = parser.parse_args()
    args.case_overrides = json.loads(args.case_config.read_text()) if args.case_config else {}
    for family, cases in args.case_overrides.items():
        if family not in ("full", "swa", "draft"):
            parser.error(f"Unknown case-config family: {family}")
        for isl, config in cases.items():
            if int(isl) not in {c[0] for c in CASES} or set(config) - {"partitions", "query_splits"}:
                parser.error(f"Invalid case-config entry: {family}/{isl}")
            if "partitions" in config and (type(config["partitions"]) is not int or config["partitions"] < 1):
                parser.error("case-config partitions must be positive integers")
            if "query_splits" in config:
                value = config["query_splits"]
                if value is not None and (type(value) is not int or value not in (1, 2, 4, 8)):
                    parser.error("case-config query_splits must be1,2,4,8 or null")
    positive = (args.full_partitions, args.swa_partitions, args.draft_partitions, args.rounds, args.repeat)
    if any(value is not None and value < 1 for value in positive) or args.warmup_seconds < 0:
        parser.error("Partitions/rounds/repeat must be positive; warmup must be nonnegative")
    if args.batch_sizes and min(args.batch_sizes) < 1:
        parser.error("Batch sizes must be positive")
    cases = [c for c in CASES if args.isls is None or c[0] in args.isls]
    if args.batch_sizes:
        cases = [(c[0], bs, c[2], c[3]) for c in cases for bs in args.batch_sizes]
    specs = [geometry(c, f, args) for c in cases for f in args.attn_type]
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    args.output = args.output.resolve()
    if args.output.exists() and not args.output.is_dir():
        parser.error(f"--output must be a directory: {args.output}")
    import torch
    import flydsl
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(args.gpu)
    if properties.gcnArchName.split(":")[0] != "gfx942":
        raise RuntimeError("This benchmark requires MI300X/gfx942")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                  torch=torch.__version__, flydsl=getattr(flydsl, "__version__", "unknown"), gpu=properties.name,
                  aiter_module=pa.__file__, arch=properties.gcnArchName,
                  config={**vars(args), "output": str(args.output),
                          "case_config": str(args.case_config) if args.case_config else None},
                  results=[])
    write_report(report, args.output)
    try:
        for spec in specs:
            print(f"START {spec['family']} ISL={spec['isl']} BS={spec['bs_per_dp']}", flush=True)
            result = measure(torch, pa, spec, args)
            report["results"].append(result)
            write_report(report, args.output)
            print(json.dumps({k: result[k] for k in ("family", "isl", "median_us", "cv_percent")}), flush=True)
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(report, args.output)
        print(f"\nmimo_pa_decode summary (markdown): status={report['status']}, "
              f"recorded {len(report['results'])}/{len(specs)} cases\n"
              f"{summary_markdown(report['results'])}", flush=True)


if __name__ == "__main__":
    main()
