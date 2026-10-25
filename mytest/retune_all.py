"""Retune MI300X fused-MoE configs with the PR #5932 recipe used for B0 in opt.md.

Per model: staged search from an empty CSV, a copy searched with whole-MoE FlyDSL
(--e2e_tune), three alternating --run_config rounds, then optional promotion into
aiter/configs/model_configs. Completed stages are resumed from .done.json receipts.
"""

import argparse
import csv
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import os
import runpy
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "mytest/results/b11_retune"
CACHE_ROOT = Path("/tmp/b11_retune")
MODELS = (
    "hunyuan3_fp8_per_tensor",
    "minimax_m3_fp8_ptpc",
    "qwen3_5_35b_fp8_ptpc",
    "qwen3_5_397b_fp8_ptpc",
    "qwen38_flash_next_bf16",
    "qwen38_flash_next_fp8_ptpc",
    "mimo_v2_5_pro",
)
TUNER = ROOT / "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
FLYDSL_PATH = "/tmp/aiter_mimo_flydsl040"
# The prebuilt aiter/jit/module_aiter_core.so predates this source; these were rebuilt from it.
FRESH_JIT = Path("/tmp/aiter_mimo_resume_jit")
COMMON_ENV = {
    "TUNE_ONLY": "flydslv2,cktile,flydsl,asm",
    "AITER_FMOE_TUNE_EXCLUDE_KERNELS": "fmoe_stage1",
    "AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED": "1",
    "GPU_ARCHS": "gfx942",
    "PYTHONHASHSEED": "0",
    "PYTHONUNBUFFERED": "1",
}
ARMS = ("staged", "whole_moe")
ROUND_ORDERS = (ARMS, ARMS[::-1], ARMS)
WHOLE_MOE_QUANT = ("QuantType.No", "QuantType.per_Token", "QuantType.per_Tensor")
DISPATCH_COLUMNS = (
    "block_m", "ksplit", "kernelName1", "kernelName2", "run_1stage", "xbf16", "flat", "nt",
)
WHOLE_MOE_MARK = "impl__flydsl_gfx942__"


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def shape_label(row):
    return (
        f"({row['token']}, {row['model_dim']}, {row['inter_dim']}, "
        f"E={row['expert']}, topk={row['topk']}, {row['act_type']}, "
        f"{row['dtype']}, {row['q_dtype_a']}, {row['q_dtype_w']}, "
        f"{row['q_type']}, g1u1={bool(int(row['use_g1u1']))}, "
        f"dw_s1={bool(int(row['doweight_stage1']))})"
    )


def prepare(args):
    output = ARTIFACTS / ("smoke" if args.smoke else "full")
    output.mkdir(parents=True, exist_ok=True)
    plans = []
    for position, model in enumerate(args.models):
        source = ROOT / f"aiter/configs/model_configs/{model}_untuned_fmoe.csv"
        columns, rows = read_csv(source)
        skipped = [row for row in rows if row["q_type"] == "QuantType.per_1x32"]
        rows = [row for row in rows if row["q_type"] != "QuantType.per_1x32"]
        for row in rows:
            if row["q_type"] == "QuantType.per_128x128":
                row["q_type"] = "QuantType.per_1x128"
        assert rows, f"{model}: no gfx942 shapes"
        assert len({tuple(row[c] for c in columns) for row in rows}) == len(rows), model
        if args.smoke:
            rows = [row for row in rows if int(row["token"]) in args.smoke_tokens] or rows[:1]
        directory = output / model
        directory.mkdir(exist_ok=True)
        input_file = directory / f"{model}_untuned_fmoe.csv"
        write_csv(input_file, columns, rows)
        common = [
            sys.executable, str(TUNER), "-i", str(input_file),
            "--mp", "1", "--warmup", "10", "--iters", "51", "--all",
        ]
        whole_rows = [row for row in rows if row["q_type"] in WHOLE_MOE_QUANT]
        whole_command = common
        if len(whole_rows) != len(rows):
            whole_input = directory / f"{model}_whole_moe_untuned_fmoe.csv"
            write_csv(whole_input, columns, whole_rows)
            whole_command = common.copy()
            whole_command[whole_command.index("-i") + 1] = str(whole_input)
        staged = directory / f"staged_{model}_tuned_fmoe.csv"
        whole = directory / f"whole_moe_{model}_tuned_fmoe.csv"
        plans.append({
            "model": model,
            "gpu": args.gpus[position % len(args.gpus)],
            "shape_count": len(rows),
            "whole_moe_eligible_shapes": len(whole_rows),
            "skipped_shapes": skipped,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "input": str(input_file),
            "staged": str(staged),
            "whole_moe": str(whole),
            "staged_command": common + ["-o", str(staged)],
            "whole_moe_command": whole_command + ["-o", str(whole), "--e2e_tune"],
            "benchmark_commands": {
                "staged": common + ["-o", str(staged), "--run_config", str(staged)],
                "whole_moe": common + ["-o", str(whole), "--run_config", str(whole)],
            },
        })
    manifest = {
        "head": git("rev-parse", "HEAD"),
        "methodology_pr": "https://github.com/ROCm/aiter/pull/5932",
        "comparison": "staged search versus the same staged winners plus whole-MoE FlyDSL e2e search",
        "common_env": COMMON_ENV,
        "flydsl_path": FLYDSL_PATH,
        "round_orders": ROUND_ORDERS,
        "plans": plans,
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, (
            "Existing experiment differs; use a new output directory."
        )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "git_status.txt").write_text(git("status", "--porcelain", "--untracked-files=no") + "\n")
    print(f"Prepared {len(plans)} models, {sum(p['shape_count'] for p in plans)} shapes; head={manifest['head']}")
    for plan in plans:
        print(f"{plan['model']}: {plan['shape_count']} shapes "
              f"({plan['whole_moe_eligible_shapes']} whole-MoE eligible, "
              f"{len(plan['skipped_shapes'])} skipped), GPU {plan['gpu']}")
    return output, plans


def capture_benchmark():
    sys.path.insert(0, str(ROOT))
    import torch
    from aiter.utility.base_tuner import TunerCommon

    torch.manual_seed(0)
    original = TunerCommon._print_benchmark_results
    original_config = TunerCommon._set_config_env_for_run_config
    destination = Path(os.environ["MOE_COMPARISON_RESULTS"])

    def order_shapes(self, args, config_file=None):
        self.untunedf = self.untunedf.sort_values(self.keys, kind="stable").reset_index(drop=True)
        return original_config(self, args, config_file=config_file)

    def record(self, title, results, *rest, **kwargs):
        destination.write_text(json.dumps(results, indent=2) + "\n")
        original(self, title, results, *rest, **kwargs)
        invalid = [r for r in results if r["status"] != "ok" or r["e2e_us"] <= 0]
        if not results or invalid:
            raise RuntimeError(f"Invalid benchmark results: {invalid}")

    TunerCommon._set_config_env_for_run_config = order_shapes
    TunerCommon._print_benchmark_results = record
    sys.argv = [str(TUNER)] + sys.argv[2:]
    runpy.run_path(str(TUNER), run_name="__main__")


def execute(command, log_file, environment, capture=False):
    marker = log_file.with_suffix(".done.json")
    if marker.exists():
        print(f"Already completed: {log_file.parent.name}/{log_file.name}", flush=True)
        return
    if capture:
        command = [sys.executable, str(Path(__file__).resolve()), "--capture"] + command[2:]
        environment = dict(environment, MOE_COMPARISON_RESULTS=str(log_file.with_suffix(".json")))
    print(f"{time.strftime('%H:%M:%S')} START {log_file.parent.name}/{log_file.name}", flush=True)
    started = time.monotonic()
    with log_file.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    print(f"{time.strftime('%H:%M:%S')} END {log_file.parent.name}/{log_file.name}: "
          f"exit={result.returncode}, {elapsed:.1f}s", flush=True)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log_file}")
    marker.write_text(json.dumps(
        {"command": command, "returncode": result.returncode, "elapsed_seconds": elapsed}, indent=2) + "\n")


def validate_config(plan, path, staged=False):
    rows = read_csv(path)[1]
    assert len(rows) == plan["shape_count"], (path, len(rows), plan["shape_count"])
    assert all(row["gfx"] == "gfx942" and int(row["cu_num"]) == 304 for row in rows), path
    assert all(float(row["us"]) > 0 for row in rows), path
    if staged:
        assert not any(WHOLE_MOE_MARK in row["kernelName1"] for row in rows), path
    return rows


def seed_jit(directory):
    marker = directory / ".seeded"
    if marker.exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "aiter/jit", FRESH_JIT):
        for binary in source.glob("*.so"):
            shutil.copy2(binary, directory / binary.name)
    marker.write_text(f"{ROOT / 'aiter/jit'} then {FRESH_JIT}\n")


def run_model(plan, output):
    directory = Path(plan["input"]).parent
    gpu = plan["gpu"]
    environment = dict(os.environ, **COMMON_ENV)
    for key in (
        "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "OPUS_ONLY", "OPUS_SKIP_CKTILE",
        "TUNE_MOE_KERNEL_REGEX", "AITER_CONFIG_FMOE", "AITER_REBUILD", "FLYDSL_RUNTIME_ENABLE_CACHE",
    ):
        environment.pop(key, None)
    jit_directory = CACHE_ROOT / output.name / "jit" / f"gpu{gpu}"
    cache_directory = CACHE_ROOT / output.name / "flydsl_cache" / f"gpu{gpu}"
    environment.update(
        ROCR_VISIBLE_DEVICES=str(gpu),
        MAX_JOBS="8",
        OMP_NUM_THREADS="4",
        PYTHONPATH=os.pathsep.join([str(ROOT), FLYDSL_PATH]),
        AITER_JIT_DIR=str(jit_directory),
        FLYDSL_RUNTIME_CACHE_DIR=str(cache_directory),
    )
    with (output / f"gpu{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seed_jit(jit_directory)
        cache_directory.mkdir(parents=True, exist_ok=True)
        runtime = {
            key: value for key, value in environment.items()
            if key.startswith(("AITER_", "TUNE_", "ROCR_", "GPU_ARCHS", "MAX_JOBS", "OMP_", "PYTHON", "FLYDSL_"))
        }
        (directory / "runtime_environment.json").write_text(json.dumps(runtime, indent=2) + "\n")
        execute(plan["staged_command"], directory / "staged_tune.log", environment)
        validate_config(plan, plan["staged"], staged=True)
        if not (directory / "whole_moe_tune.done.json").exists():
            shutil.copyfile(plan["staged"], plan["whole_moe"])
        execute(plan["whole_moe_command"], directory / "whole_moe_tune.log", environment)
        validate_config(plan, plan["whole_moe"])
        for round_index, order in enumerate(ROUND_ORDERS, 1):
            for arm in order:
                execute(
                    plan["benchmark_commands"][arm],
                    directory / f"round{round_index}_{arm}.log",
                    environment,
                    capture=True,
                )
    return plan["model"]


def load_rounds(plan):
    directory = Path(plan["input"]).parent
    rounds = {}
    for arm in ARMS:
        rounds[arm] = []
        for round_index in range(1, len(ROUND_ORDERS) + 1):
            entries = json.loads((directory / f"round{round_index}_{arm}.json").read_text())
            assert len(entries) == plan["shape_count"], (plan["model"], arm, round_index)
            assert all(e["status"] == "ok" and e["e2e_us"] > 0 for e in entries)
            rounds[arm].append({entry["shape"]: entry for entry in entries})
    return rounds


def summarize(output, plans):
    comparisons, summaries = [], []
    for plan in plans:
        rounds = load_rounds(plan)
        staged_rows = validate_config(plan, plan["staged"], staged=True)
        whole_rows = validate_config(plan, plan["whole_moe"])
        input_columns = read_csv(plan["input"])[0]
        whole_by_key = {tuple(row[c] for c in input_columns): row for row in whole_rows}
        for row in staged_rows:
            whole_row = whole_by_key[tuple(row[c] for c in input_columns)]
            shape = shape_label(row)
            times = {arm: [r[shape]["e2e_us"] for r in rounds[arm]] for arm in ARMS}
            staged_us = statistics.median(times["staged"])
            whole_us = statistics.median(times["whole_moe"])
            comparisons.append({
                "model": plan["model"],
                **{c: row[c] for c in input_columns},
                "gpu": plan["gpu"],
                "staged_us": staged_us,
                "whole_moe_us": whole_us,
                "speedup": staged_us / whole_us,
                "latency_change_pct": (whole_us / staged_us - 1) * 100,
                "paired_wins": sum(s > w for s, w in zip(times["staged"], times["whole_moe"])),
                "selected_whole_moe": WHOLE_MOE_MARK in whole_row["kernelName1"],
                "staged_rounds": times["staged"],
                "whole_moe_rounds": times["whole_moe"],
                "staged_kernel1": row["kernelName1"],
                "whole_moe_kernel1": whole_row["kernelName1"],
                "staged_kernel2": row["kernelName2"],
                "whole_moe_kernel2": whole_row["kernelName2"],
            })
    subsets = {
        "all": lambda row: True,
        "whole_moe_selected": lambda row: row["selected_whole_moe"],
        "decode_1_8": lambda row: int(row["token"]) <= 8,
        "token_ge_2048": lambda row: int(row["token"]) >= 2048,
        "whole_moe_eligible": lambda row: row["q_type"] in WHOLE_MOE_QUANT,
    }
    for model in [plan["model"] for plan in plans] + ["ALL"]:
        rows = [row for row in comparisons if model == "ALL" or row["model"] == model]
        for subset, keep in subsets.items():
            selected = [row for row in rows if keep(row)]
            if not selected:
                continue
            speedup = statistics.geometric_mean(row["speedup"] for row in selected)
            summaries.append({
                "model": model,
                "subset": subset,
                "shapes": len(selected),
                "whole_moe_selected": sum(row["selected_whole_moe"] for row in selected),
                "staged_geomean_us": statistics.geometric_mean(row["staged_us"] for row in selected),
                "whole_moe_geomean_us": statistics.geometric_mean(row["whole_moe_us"] for row in selected),
                "speedup": speedup,
                "latency_change_pct": (1 / speedup - 1) * 100,
                "wins_over_3pct": sum(row["speedup"] > 1.03 for row in selected),
                "regressions_over_3pct": sum(row["speedup"] < 1 / 1.03 for row in selected),
            })
    for filename, rows in (("per_shape.csv", comparisons), ("summary.csv", summaries)):
        write_csv(output / filename, list(rows[0]), rows)
    for row in summaries:
        if row["subset"] in ("all", "whole_moe_selected", "token_ge_2048"):
            print(f"{row['model']:<28} {row['subset']:<20} n={row['shapes']:<3} "
                  f"whole={row['whole_moe_selected']:<3} {row['staged_geomean_us']:10.3f} -> "
                  f"{row['whole_moe_geomean_us']:10.3f} us  {row['speedup']:.3f}x", flush=True)


def promote(output, dry_run=False):
    manifest = json.loads((output / "manifest.json").read_text())
    assert git("rev-parse", "HEAD") == manifest["head"], "Source revision changed since measurement."
    shape_columns = read_csv(ROOT / "aiter/configs/untuned_fmoe.csv")[0]
    key_columns = list(dict.fromkeys(shape_columns + ["cu_num", "gfx", "_tag"]))

    def identity(row):
        return tuple(row.get(column) or "" for column in key_columns)

    decisions, pending = [], []
    for plan in manifest["plans"]:
        assert sha256(ROOT / plan["source"]) == plan["source_sha256"], plan["source"]
        rounds = load_rounds(plan)
        staged_rows = validate_config(plan, plan["staged"], staged=True)
        whole_rows = {identity(row): row for row in validate_config(plan, plan["whole_moe"])}
        destination = ROOT / f"aiter/configs/model_configs/{plan['model']}_tuned_fmoe.csv"
        columns, existing = read_csv(destination)
        existing_by_key = {identity(row): row for row in existing}
        assert len(existing_by_key) == len(existing), destination
        new_rows = []
        for staged_row in staged_rows:
            whole_row = whole_rows[identity(staged_row)]
            shape = shape_label(staged_row)
            staged_times = [r[shape]["e2e_us"] for r in rounds["staged"]]
            whole_times = [r[shape]["e2e_us"] for r in rounds["whole_moe"]]
            speedup = statistics.median(staged_times) / statistics.median(whole_times)
            is_whole = WHOLE_MOE_MARK in whole_row["kernelName1"]
            keep_whole = is_whole and speedup > 1.03 and all(
                s > w for s, w in zip(staged_times, whole_times))
            selected = whole_row if keep_whole else staged_row
            previous = existing_by_key.get(identity(selected))
            if previous is None:
                extra = {c for c in selected if c not in columns}
                assert extra <= {"nt"} and selected.get("nt", "0") in ("0", "0.0", ""), (extra, selected)
                new_rows.append({c: selected.get(c, "") for c in columns})
                action = "added"
            elif all((previous.get(c) or "0") == (selected.get(c) or "0") for c in DISPATCH_COLUMNS):
                action = "existing_identical"
            else:
                action = "existing_kept_differs"
            decisions.append({
                "model": plan["model"],
                **{c: staged_row.get(c, "") for c in key_columns},
                "selected": "whole_moe" if keep_whole else "staged",
                "reason": "three_paired_wins_and_over_3pct" if keep_whole else (
                    "whole_moe_not_selected_by_tuner" if not is_whole else "gain_not_confirmed"),
                "action": action,
                "staged_median_us": statistics.median(staged_times),
                "whole_moe_median_us": statistics.median(whole_times),
                "measured_speedup": speedup,
                "kernelName1": selected["kernelName1"],
                "kernelName2": selected["kernelName2"],
            })
        if new_rows:
            pending.append((destination, columns, new_rows))
    for destination, columns, rows in pending:
        if dry_run:
            print(f"[dry-run] would add {len(rows)} MI300 rows to {destination.relative_to(ROOT)}")
            continue
        content = destination.read_text()
        with destination.open("a", newline="") as stream:
            if content and not content.endswith("\n"):
                stream.write("\n")
            csv.DictWriter(stream, fieldnames=columns, lineterminator="\n").writerows(rows)
        print(f"Added {len(rows)} MI300 rows to {destination.relative_to(ROOT)}")
    write_csv(output / "promotion.csv", list(decisions[0]), decisions)
    for action in ("added", "existing_identical", "existing_kept_differs"):
        print(f"{action}: {sum(d['action'] == action for d in decisions)}")
    print("whole-MoE promoted:", sum(d["selected"] == "whole_moe" and d["action"] == "added" for d in decisions))


def main():
    global ARTIFACTS
    if len(sys.argv) > 1 and sys.argv[1] == "--capture":
        capture_benchmark()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(len(MODELS))))
    parser.add_argument("--output-root", type=Path, default=ARTIFACTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-tokens", nargs="+", type=int, default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="with --promote, do not edit official CSVs")
    args = parser.parse_args()
    ARTIFACTS = args.output_root.resolve()
    if args.promote:
        promote(ARTIFACTS / ("smoke" if args.smoke else "full"), dry_run=args.dry_run)
        return
    output, plans = prepare(args)
    if args.prepare_only:
        return
    if not args.summarize_only:
        packages = {name: importlib.metadata.version(name) for name in ("torch", "flydsl", "pandas", "numpy")}
        assert packages["flydsl"] == "0.4.0", packages
        (output / "environment.json").write_text(json.dumps(
            {"python": sys.executable, "packages": packages, "environment": COMMON_ENV}, indent=2) + "\n")
        with ThreadPoolExecutor(max_workers=len(plans)) as executor:
            completed = list(executor.map(functools.partial(run_model, output=output), plans))
        print("Completed:", completed, flush=True)
    summarize(output, plans)


if __name__ == "__main__":
    main()
