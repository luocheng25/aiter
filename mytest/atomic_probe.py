"""Cost of a fused MoE Down epilogue: per-route BF16 stores vs FP32 atomic accumulation.

Uses the real MiMo routing/sorting at a given token count. Kernels only generate a
per-row constant, so the timings isolate the output path (no GEMM).
"""

import argparse
import json
import statistics

import torch
import triton
import triton.language as tl

from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting

HIDDEN, EXPERTS, TOPK = 6144, 384, 8
MODES = {
    "store_bf16_routes": 0,
    "atomic_f32_agent": 1,
    "atomic_f32_workgroup_scope": 2,
    "atomic_f32_agent_chunk8k": 3,
    "atomic_bf16_agent": 4,
}


@triton.jit
def _epilogue(sorted_ids, dst, M, N: tl.constexpr, TOPK: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MODE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ids = tl.load(sorted_ids + rows)
    token = ids & 0xFFFFFF
    slot = ids >> 24
    mask = (token < M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    value = (1.0 + (rows % 7).to(tl.float32) * 0.125)[:, None] + tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    if MODE == 0:
        ptr = dst + (token * TOPK + slot).to(tl.int64)[:, None] * N + cols[None, :]
        tl.store(ptr, value.to(tl.bfloat16), mask=mask)
    elif MODE == 1:
        ptr = dst + token.to(tl.int64)[:, None] * N + cols[None, :]
        tl.atomic_add(ptr, value, mask=mask, sem="relaxed", scope="gpu")
    elif MODE == 2:
        ptr = dst + token.to(tl.int64)[:, None] * N + cols[None, :]
        tl.atomic_add(ptr, value, mask=mask, sem="relaxed", scope="cta")
    elif MODE == 3:
        ptr = dst + (token % 8192).to(tl.int64)[:, None] * N + cols[None, :]
        tl.atomic_add(ptr, value, mask=mask, sem="relaxed", scope="gpu")
    else:
        ptr = dst + token.to(tl.int64)[:, None] * N + cols[None, :]
        tl.atomic_add(ptr, value.to(tl.bfloat16), mask=mask, sem="relaxed", scope="gpu")


def time_us(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", type=int, default=32768)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    M, N = args.token, HIDDEN
    hidden = torch.randn((M, N), dtype=dtypes.bf16)
    gating = torch.randn((M, EXPERTS), dtype=dtypes.bf16)
    weights, ids = fused_topk(hidden, gating, TOPK, True)
    sorted_ids, _, _, num_valid, _ = moe_sorting(ids, weights, EXPERTS, N, dtypes.bf16, args.block_m)
    rows = int(num_valid[0].item())
    rows -= rows % args.block_m
    grid = (N // args.block_n, rows // args.block_m)
    routes = M * TOPK
    results = {"token": M, "sorted_rows": rows, "routes": routes, "block_m": args.block_m,
               "block_n": args.block_n}
    inter = torch.empty((routes, N), dtype=dtypes.bf16)
    acc = torch.zeros((M, N), dtype=torch.float32)
    acc_bf16 = torch.zeros((M, N), dtype=dtypes.bf16)
    for name in args.modes:
        mode = MODES[name]
        dst = inter if mode == 0 else acc_bf16 if mode == 4 else acc
        try:
            us = time_us(lambda: _epilogue[grid](sorted_ids, dst, M, N, TOPK, args.block_m, args.block_n, mode,
                                                 num_warps=4), args.iters)
        except Exception as error:  # noqa: BLE001
            results[name] = f"failed: {type(error).__name__}: {str(error)[:200]}"
            print(name, results[name], flush=True)
            continue
        payload = routes * N * (2 if mode in (0, 4) else 4)
        results[name] = {"us": us, "payload_GB": payload / 1e9, "TB_s": payload / us / 1e6}
        print(f"{name:28s} {us:10.1f} us  payload {payload / 1e9:6.3f} GB  {payload / us / 1e6:6.3f} TB/s", flush=True)
    out = torch.empty((M, N), dtype=dtypes.bf16)
    results["zero_f32_acc_us"] = time_us(lambda: acc.zero_(), args.iters)
    results["convert_f32_to_bf16_us"] = time_us(lambda: out.copy_(acc), args.iters)
    print(f"zero f32 acc {results['zero_f32_acc_us']:.1f} us, convert f32->bf16 {results['convert_f32_to_bf16_us']:.1f} us")
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
