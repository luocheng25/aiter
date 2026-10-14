# FlyDSL FP8 blockscale MoE on gfx950

The existing `flydsl_gfx950` whole-graph backend supports W128x128/A1x128
FP8 prefill and a separate **A16W8 weight-only decode** candidate through
`run_flydsl_moe_gfx942_impl`. Prefill requires **gfx950**; decode now has
compile-time gfx942/gfx950 specializations. gfx942 has compile-only validation;
its hardware correctness/performance remain to be tested. Decode's implementation,
numerical distinction, validation and measured gfx950 results are in the
[decode report](flydsl_mimo_v26_decode.md). The sections below describe prefill.
The subsequent [full-model E2E retune](flydsl_mimo_v26_e2e_retune.md) updates the
published Pro/Flash CSVs after independent before/after validation.

## Implementation and configuration

- [Backend/config](../aiter/ops/flydsl/fused_moe_gfx942.py): shared sort256,
  HIP A1x128 quantization, Gate/Up, a second HIP quantization, Down and reduce.
- [Gate/Up](../aiter/ops/flydsl/kernels/moe_fp8_blockscale_stage1.py): separated
  SiLU, BF16 routed intermediate, no clamp or routing-weight application.
- [M128 Down](../aiter/ops/flydsl/kernels/moe_fp8_blockscale_stage2.py): independent
  or persistent scheduling; retains the K256 specialized pipeline and streams
  larger K in K256 chunks.
- [8-wave Down](../aiter/ops/flydsl/kernels/moe_8wave_down.py): whole-K storage,
  native sort256, routed BF16 output.
- [Packed reduce](../aiter/ops/flydsl/kernels/moe_packed_reduce.py) and
  [shape/LDS checks](../aiter/ops/flydsl/kernels/moe_blockscale_config.py) are
  copied alongside the GEMMs. The small
  [common adapter](../aiter/ops/flydsl/kernels/moe_blockscale_common.py) reuses
  Aiter's pointer, buffer/LDS and explicit FMA primitives.

No production or test module imports PyHIP. Kernel source licenses are retained.
The copied device/launcher ASTs match the source after excluding import changes;
formatting and DSL-specific lint comments do not change their arithmetic.

The new `Config.down_path` values are `block_m128`, `block_m128_persistent`, and
`block_8wave`. `BLOCK_M/BLOCK_N/BLOCK_K` are fixed at `256/256/128`; `BLOCK_M`
describes shared sorting, not the M128 Down tile. `down_block_n` is 128 for M128
or 32/64/128 for 8-wave. `num_oc_splits` partitions **N**, never K.

Old config strings are unchanged. New strings append Down N and OC splits, e.g.
`256_256_128_True_False:block_m128:0:128:4` and
`256_256_128_True_False:block_8wave:0:32:8`.

The [E2E tuner](../csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py) proposes
these configurations on gfx950 and filters them using the same shape/LDS checks.
Both `QuantType.per_1x128` and `QuantType.per_128x128` are accepted; new tuned rows
use canonical `per_1x128`, matching public dispatch. Duplicate alias candidates
produce only one winning row. Model tuning results are linked below; the existing
default backend selection is not changed without a tuned row.

The public path is enabled by loading an E2E-tuned CSV. Tune and use this family
only for the supported **unclamped SiLU** calls; do not reuse its rows for other
activations or incompatible dtype overrides. The tuner filters these candidates,
and backend validation remains in place. There is no blockscale-specific guard
in the main dispatcher to discard an incompatible row and retry legacy heuristics.

## Input and workspace contracts

- BF16 input/output; both preshuffled weights use `torch.float8_e4m3fn` and
  `shuffle_weight(16,16)`. FP32 weight scales are `[E,N/128,K/128]`.
- Gate/Up requires positive `H % 256 == 0` and `I % 128 == 0`. M128 Down requires
  `I % 256 == 0`. 8-wave accepts positive I multiples of128 subject to whole-K
  LDS limits and at least three N tiles per OC split.
- Activation scales from `transpose_scale=True` are already physically
  `[K/128,rows]`. A reshape/view must not transpose these bytes again.
- M128 output is packed M128/N64, even with sort256. It must use inverse routing
  plus the packed reducer with stride128, not the row-major `sorted_sum`.
- 8-wave writes `[M,topk,N]` and uses a Torch TOPK sum. Both Down kernels already
  apply routing weights before writing BF16 routes.
- Persistent M128 receives a fresh zero-initialized 256-word queue; its eight
  heads self-reset. 8-wave receives a private one-word counter that its launcher
  clears each time. Cached code owns no tensors, counters or streams.
- Native unmasked sorting must contain every real `(token,slot)` once. No EP,
  masked/dropped routes, prequantized input, bias, padding, split-K, interleaved
  Gate/Up, `doweight_stage1`, non-SiLU activation or active clamp is supported.
  Callers must preserve the tuned SiLU/dtype contract, including leaving
  `quant_dtype_a2` unset or FP8 E4M3FN. That override is not forwarded through the
  existing whole-graph request; no main-dispatcher check is added for it.

## Validation (2026-10-13)

Hardware: MI350X/gfx950; Torch 2.9.1 ROCm7.2. Tested both local FlyDSL0.3.2 and
Aiter's pinned FlyDSL0.3.4.1, loading this Aiter checkout in both environments.

- Each environment: **86 passed, 1 skipped, 407 unittest subtests passed** across
  the [GPU regressions](../op_tests/flydsl_tests/test_blockscale_moe.py),
  [CPU tuner regressions](../op_tests/tuning_tests/test_flydsl_blockscale.py),
  [benchmark module collection](../op_tests/test_flydsl_moe_blockscale.py) and
  [existing whole-graph registration tests](../op_tests/tuning_tests/test_online_tune.py).
  The skip is the unsupported 8-wave/native-sort128 combination.
  These results were rerun after removing the main-dispatcher blockscale guard
  and its six automatic-fallback tests; tuner and backend checks are unchanged.
- Covered K256 specialization, K512/768/1024 streaming, 8-wave K128/384/640/1152,
  OC splits, rows around128/256 boundaries, TOPK130 signed IDs, empty/restore
  valid prefixes, queue reset, nondefault streams, double-call graph capture,
  replay with changed inputs/scales/routes, independent concurrent graphs,
  public output buffers and backend rejection of unsupported requests.
- Source-versus-port Gate/Up, both M128 schedulers, 8-wave and packed reduce were
  bit-exact. **Nine device ELF pairs were byte-identical** on FlyDSL0.3.4.1:
  Gate/Up for I256/I512; independent/persistent M128 for K256/K512; 8-wave for
  K256/K512; and the K-independent packed reducer. This is not an equivalence
  claim for untested specializations or host timing.
- Sixteen legacy gfx950 BF16/PTPC/per-tensor/MXFP4 decode cases (M1, M4 direct,
  M4 sorted, M17 sorted) were bit-exact against the unmodified backend.
- A fresh process forbidding all PyHIP imports ran four real `FmoeTuner.run_config`
  cases successfully. Official `--e2e_tune` enumerated20 candidates for one
  shape and wrote a temporary CSV; a new official `--run_config` process loaded
  the canonical row, selected `impl__flydsl_gfx950__...block_m128...`, and passed.
- Ruff, Black and syntax checks pass for changed Python files. Editor-only
  unresolved Torch/Pandas imports persist in the remote language service;
  actual imports and tests pass. No gfx942 hardware validation was performed.

### Full-pipeline timing smoke

The [standalone benchmark](../op_tests/test_flydsl_moe_blockscale.py) uses Aiter
`run_perftest`, two warmups and ten iterations, and checks both initial and timed
outputs against a Torch reference containing both quantizations. H=4096, E=16,
TOPK=4. All96 legal shape/config combinations passed across the two batches.

| Tokens | I | Legal configs | Fastest in this run | us |
|---:|---:|---:|---|---:|
| 1024 | 256 | 20 | M128, OC4 | 100.811 |
| 1024 | 384 | 8 | 8-wave N64, OC8 | 110.277 |
| 1024 | 512 | 16 | 8-wave N32, OC8 | 113.695 |
| 1024 | 640 | 4 | 8-wave N32, OC8 | 158.238 |
| 8192 | 256 | 20 | M128, OC4 | 313.661 |
| 8192 | 384 | 8 | 8-wave N64, OC4 | 402.332 |
| 8192 | 512 | 16 | 8-wave N32, OC4 | 489.981 |
| 8192 | 640 | 4 | 8-wave N32, OC8 | 742.684 |

These are ordinary full-pipeline measurements, including sorting, both
quantizations, counter reset and inverse/reduce. They are not clock-locked,
PTL-certified or paired before/after measurements, and are not a speedup claim.
The official tuner's separate smoke chose OC8 at106.335us and replayed at107.28us
with its own input distribution; do not compare that directly with this table.

### Tuned MiMo-shape comparison

See the [native-tuned versus E2E-tuned report](flydsl_fp8_blockscale_mimo_perf.md)
for H6144, I-per-TP256, E384, TOPK8 and batches1024 through32768. It includes
the full native candidate search, public-call revalidation, ABBA measurements,
raw round aggregates and separate baseline/winner CSVs. The new paths reduce
GPU time by3.71%,17.27%,18.02%,15.30% at batches4096,8192,16384,32768;
batches1024 and2048 retain the same native CK implementation. These remain
ordinary unlocked GPU-kernel timings, not CPU wall-time or PTL-certified results.

For the later user-supplied **TOPK9** Pro/Flash CSVs (E385/E257), see the
[32-shape native/E2E tuning report](flydsl_mimo_v26_tk9_tuning.md). It includes
separate native baselines, published gfx950 configs, all ABBA round aggregates
and default-merged public-dispatch validation. It does not replace the earlier
TOPK8 measurements above.