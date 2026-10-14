# MiMo V2.6：FlyDSL blockscale A16W8 decode

日期：2026-10-13。新增 `block_a16w8` 是 **BF16 activation + FP8 blockscale
weight** 的独立 decode 候选；不是 W128x128/A1x128 prefill 的等价替换。
代码只使用 Aiter/FlyDSL，不导入 PyHIP，不调用或新增 ASM kernel。

后续已增加 gfx942 编译期特化，见下节；本文后面的首轮 gfx950 decode 性能表
保留原始结果，不作为 gfx942 实测。
后续全16档 E2E 搜索及正式模型 CSV 更新见
[全量再调优报告](flydsl_mimo_v26_e2e_retune.md)。

## gfx942 编译期特化（同日追加）

- 同一个 `make_gemm_helpers` 在构建 kernel 时按目标 gfx 选择指令，没有 GPU
  运行时架构分支。gfx950 保留 packed FP8→BF16 与 K32 BF16 MFMA；gfx942 使用
  已有 FP8→FP32→BF16 conversion 和 K16 BF16 MFMA。两者都保持每 wave K64
  partial、FP32 weight-scale FMA、Gate 四-wave reduce 和 Down BF16 atomics。
  不是将 scale 乘入 BF16 weights 后再 GEMM；没有改变现有 gfx950 数值流程。
- gfx942 权重格式为 `torch.float8_e4m3fnuz`，gfx950 为 `torch.float8_e4m3fn`。
  backend 输入检查和 E2E CSV 资格筛选同步使用对应格式。四个 decode 配置现在
  两个架构均可搜索；三个 blockscale prefill 路径仍只允许 gfx950。
- gfx942 离线编译24个 Gate/Down 特化：H/I 为512/128、1536/384、4096/256、
  6144/256，各覆盖 direct、sorted BN64、sorted BN128。IR/ELF/ISA 均生成，
  ISA 使用 K16 BF16 MFMA，不含 gfx950-only conversion/K32 BF16 指令；
  所查 metadata 的 scratch/private segment 和 SGPR/VGPR spill count 均为0。
- 编译使用 FlyDSL0.3.4.1 的 `COMPILE_ONLY=1`、`ARCH=gfx942` 与
  `FLYDSL_GPU_ARCH=gfx942`，断言未建立 ExecutionEngine、FlyDSL device runtime
  或 Torch CUDA context。为避免 Aiter 顶层无关的 Triton GPU 探测，离线检查直接
  从实际文件加载 Gate/Down/helper；没有替换 kernel 源码或执行 gfx942 binary。
- gfx950 两个 MiMo 模型的12个 Gate/Down ELF 与特化前逐字节一致。
  定向回归为 **169 passed、1 expected skip、570 subtests passed**。
  Black/Ruff 通过。以上不代表 gfx942 硬件正确性或性能已经验证。
- 到 gfx942 机器后运行同一个
  [decode pytest](../op_tests/flydsl_tests/test_blockscale_decode.py) 即可：它会自动
  选择 FNUZ 和 `flydsl_gfx942`，覆盖 direct/sorted、非均匀 scales、stage 输出、
  相消、stream/graph 以及 public dispatch。精度门槛未更改。
  在 gfx942 重新 E2E tune 时，输入 CSV 的 `q_dtype_a/q_dtype_w` 需要使用 FNUZ；
  本仓库当前两个 MiMo FN 输入 CSV 与实测 tuned 行仍用于 gfx950，不应直接套用。

离线编译与回归产物位于 `/tmp/aiter-blockscale-gfx942-20261013`。

## 接入与数值流程

- [共享 GEMM helper](../aiter/ops/flydsl/kernels/moe_gemm_2stage/common.py)：
  gfx950 packed FP8→BF16 conversion 使用 scale=1；每个 wave 的 K64 分段由
  两个 K32 BF16 MFMA 得到独立 FP32 partial，再用对应 W128x128 scale 做 FP32 FMA。
  任意 FP32 weight scale 不放入指数 conversion，也不移到整个 GEMM 末尾。
- [Gate/Up](../aiter/ops/flydsl/kernels/moe_gemm_2stage/gemm1.py) 保持四-wave
  split-K、LDS reduce、SiLU 和 BF16 intermediate；
  [Down](../aiter/ops/flydsl/kernels/moe_gemm_2stage/gemm2.py) 保持单 wave、
  routing-weight 乘法及 BF16 packed atomics。沿用现有 BF16 rounding helper。
- [Backend](../aiter/ops/flydsl/fused_moe_gfx942.py) 复用 direct/sorted host 路径；
  M1 走 batch1，M2–8 同时搜索 direct 与 sorted，M9–256 搜索 sorted。
  sorted BM16、BN64/128、BK64；direct 实际 Gate BN32、Down BN64。
- 首版只支持 gfx950、BF16 输入/输出、FP8 E4M3FN shuffled weights、FP32
  `[E,N/128,K/128]` scales、separated/unclamped SiLU、H%512=0、I%128=0。
  routing slot 限制为 1～min(E,127)：旧 decode 的 signed slot mask 不支持
  TOPK128 所产生的负数 padding slot。原 decode/prefill 的范围不变。
- 缓存仅保存 code；当前 stream、输出清零与工作张量生命周期沿用原 host 路径。
  最后一对 K tile 单独收尾，不预取越过 K 的 A/B/scale。
  新分支的 expert weight/scale offset 在乘法前升为64位。
- Config 使用原三字段语法，例如 `16_64_64_False_False:block_a16w8:0`。
  `is_blockscale_decode` 与 `is_blockscale_prefill` 互斥；原 config 拼写不变。
  `ksplit=0` 保持不变，四-wave split-K 是 CTA 内部算法，不是公共 grid split-K。

未修改 [主 dispatcher](../aiter/fused_moe.py)，没有默认启用或新增 fallback guard。
仍须由 E2E tune 写入 `impl__flydsl_gfx950__...:block_a16w8:0` 后显式选中。
本轮没有加入 Down LDS coalescing 或扩大 BM，不能把这些作为已实现的优化。

## 验证

环境：MI350X/gfx950、256 CUs，HIP logical0 / PCI 0000:75:00.0。
Python3.10.12、FlyDSL0.3.4.1、Torch2.9.1+rocm7.2.0.git7e1940d4，
HIP7.2.26015-fc0010cf6a；Aiter4848005c 加当前工作区改动。
没有修改频率、功率或 PTL；未验证 gfx942 或其他 FlyDSL 版本。

- 最终定向回归：**168 passed、1 expected skip、482 subtests passed**。
  包含原 prefill、[新 decode](../op_tests/flydsl_tests/test_blockscale_decode.py)、
  [CPU tuner](../op_tests/tuning_tests/test_flydsl_blockscale.py)、原注册测试和
  [共享参考/benchmark](../op_tests/test_flydsl_moe_blockscale.py) 的收集。
- 覆盖 direct M1/2/3/4/7/8、sorted 边界 M15/16/17…255/256、BN64/128、
  H512/1024/1536/4096/6144、I128/256/384/640、TOPK9、N/K 非均匀 scales、
  Gate/Down 独立输出、empty→restore、非默认 stream、双调用 graph replay、
  public CSV dispatch/output、block quant 两种 alias 和 dtype override。
- 原 BF16/PTPC/per-tensor/MXFP4 的 M1/M4-direct/M17/M64 共16例，
  修改前后输出逐位一致。旧量化分支不使用新增 block pipeline。
- 两份归档 CSV 的6个新 decode 行在独立进程通过真实 public API/output buffer
  验证，进程禁止任何 PyHIP 导入；默认 config family 碰撞检查20项通过。
  Black/Ruff/磁盘语法检查通过；远程编辑器仍有既有 Torch/FlyDSL import 解析问题，
  实际选择的解释器可以正常导入并运行测试。
- 相消反例保留：x0=1、x1=1.0078125，构造 Gate=x0−x1、Up=x0，
  A8W8 reference 为零，而新 A16W8 路径与 weight-only reference 逐位一致且非零。
  这明确证明两种流程不等价。
- TOPK127 的全非零随机 routes 曾有5.1758%元素超出共享1%逐元素容差，
  与 BF16 atomics 相对 FP32 TOPK sum 的累计误差相关；未放宽容差。
  最终 slot 边界用例只启用第126号 slot，专门验证最高有效 slot 与 padding127。
  因此不宣称127路随机 reduce 满足该逐元素门槛。
- Pro/Flash × direct/BN64/BN128 × Gate/Down 共12个真实模型特化均编译运行通过。
  ISA 包含 native packed conversion、BF16 K32 MFMA 和 packed FP32 FMA；
  全部 private segment、SGPR/VGPR spill count 为0。

| 特化 | Gate VGPR metadata | Down VGPR metadata | Gate / Down LDS bytes |
|---|---:|---:|---:|
| direct | 92 | 116 | 16384 / 0 |
| sorted BN64 | 156 | 118 | 16384 / 1024 |
| sorted BN128 | 268 Flash / 272 Pro | 210 | 16384 / 1024 |

VGPR metadata 包含编译器的寄存器资源核算，不直接等同 occupancy。
本轮检查生成代码，不声称与参考 ASM 的 ELF/调度逐条相同。

## 调优和测量范围

沿用用户当前配置：Pro H6144/I256/E385/TOPK9；Flash H4096/I256/E257/TOPK9。
两者 M1～256、2倍递增，共18个 shape；随机 routing，没有强制第9路 shared expert。

1. 基线复用[上一轮完整 native tune + 整链排名](flydsl_mimo_v26_tk9_tuning.md)
   的赢家。本轮没有重跑全部 native 候选搜索；先核对9个 decode shape 的原生
   kernel/M/NT/FLAT 配置与上一轮一致，再重新测量基线，不拿历史耗时直接计算收益。
2. 调用未改 reference/timing/筛选逻辑的 `FmoeTuner.e2e_tune`。
   每模型检查200项：180个原 prefill + 20个新增 decode shape/config 组合，
   全部通过原有 whole-call 精度门槛。最终两模型都只更新 M1/2/4。
3. 再以实际 public `fused_moe` 做独立 ABBA，逐轮核对 CSV 分派和真实计时输出。
   每轮原 `run_perftest`：5 warmups、101 iterations、2组轮换参数，保留原首轮
   剔除/IQR 聚合。每实现两个返回值取中位数；下面不是搜索过程最低值。
4. 计时为整链 GPU kernel 时间之和，含排序（若需要）、Gate/Up、输出清零、Down；
   不含权重量化/shuffle、topk生成、JIT、参考、CPU提交间隙或TP通信。
   ABBA共用逻辑输入，但每轮 deepcopy 的轮换 allocation 不保证同址。
   这是普通未锁频测量，不是 CPU/graph 墙钟或 PTL 认证结果。
5. 有效 FLOPs 为 $6M\times TOPK\times H\times I$；
   $TFLOPS=F/(t_{us}\times10^6)$。

## 独立 ABBA 结果

| Batch | Pro before / after us | Pro 耗时降低 | Flash before / after us | Flash 耗时降低 |
|---:|---:|---:|---:|---:|
| 1 | 32.828 / 18.456 | **43.78%** | 24.198 / 14.276 | **41.00%** |
| 2 | 41.485 / 25.955 | **37.43%** | 24.572 / 18.016 | **26.68%** |
| 4 | 60.800 / 42.511 | **30.08%** | 35.937 / 27.329 | **23.95%** |
| 8 | 76.044 / 76.215 | -0.22%* | 49.645 / 51.419 | -3.57%* |
| 16 | 132.469 / 132.553 | -0.06%* | 87.008 / 86.867 | 0.16%* |
| 32 | 201.670 / 201.524 | 0.07%* | 130.676 / 130.720 | -0.03%* |
| 64 | 292.528 / 292.088 | 0.15%* | 157.842 / 155.371 | 1.57%* |
| 128 | 343.051 / 343.617 | -0.16%* | 175.471 / 172.709 | 1.57%* |
| 256 | 366.297 / 365.768 | 0.14%* | 197.516 / 194.507 | 1.52%* |

\* 配置相同，重复测量波动，不计为新路径收益。
M512～32768没有重测，保留上一轮7行配置，不声称本次 decode 带来 prefill 改善。

| Batch | Pro before / after TFLOPS | Flash before / after TFLOPS | 新配置 |
|---:|---:|---:|---|
| 1 | 2.587 / 4.602 | 2.340 / 3.966 | `16_16_16_False_False:block_a16w8:0` |
| 2 | 4.095 / 6.545 | 4.609 / 6.286 | `16_16_16_False_True:block_a16w8:0` |
| 4 | 5.588 / 7.992 | 6.302 / 8.288 | `16_16_16_False_True:block_a16w8:0` |

### 精度：通过原门槛，不是逐元素等价

E2E reference 仍包含两次 A1x128 activation quant。原 gate 要求有限、非零，且
`err_ratio <= 0.1` **或** `logits_diff <= 0.01`；没有换 reference 或放宽容差。
新 decode 在这组输入上通过第二个条件，不通过第一个条件。

| 模型 / Batch | 新路径逐元素 err_ratio | 新路径 logits_diff |
|---|---:|---:|
| Pro / 1 | 77.913%～77.962% | 0.001052439～0.001053751 |
| Pro / 2 | 77.596%～77.873% | 0.001003861～0.001006126 |
| Pro / 4 | 77.445%～77.539% | 0.000983298～0.000984192 |
| Flash / 1 | 72.412%～72.583% | 0.000947714～0.000947773 |
| Flash / 2 | 71.570%～71.790% | 0.000958800～0.000960946 |
| Flash / 4 | 71.289%～71.448% | 0.000932574～0.000932574 |

其中 `logits_diff = 1 - 2*sum(x*y)/(sum(x*x)+sum(y*y))`，不是相对 L2。
若模型必须保留 A8W8 数值流程，不应使用该 A16W8 候选。

## 全部轮次聚合值

单位 us；A1/B1/B2/A2 为实际执行顺序，包含所有计时器返回值。

| 模型 | Batch | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|---:|
| Pro | 1 | 32.817429 | 17.873840 | 19.037711 | 32.837813 |
| Pro | 2 | 41.675105 | 26.092232 | 25.818289 | 41.294938 |
| Pro | 4 | 60.820919 | 42.566755 | 42.454458 | 60.780052 |
| Pro | 8 | 75.873240 | 75.887240 | 76.542080 | 76.215265 |
| Pro | 16 | 134.548670 | 134.941286 | 130.164629 | 130.390196 |
| Pro | 32 | 204.967918 | 205.042939 | 198.005306 | 198.371253 |
| Pro | 64 | 297.843300 | 297.209280 | 286.967160 | 287.213160 |
| Pro | 128 | 348.532800 | 348.865220 | 338.367939 | 337.569600 |
| Pro | 256 | 371.128082 | 371.440327 | 360.095052 | 361.465380 |
| Flash | 1 | 24.244200 | 14.261273 | 14.290313 | 24.150814 |
| Flash | 2 | 24.642640 | 17.928280 | 18.103440 | 24.501939 |
| Flash | 4 | 35.902269 | 27.239696 | 27.417891 | 35.972648 |
| Flash | 8 | 51.110591 | 51.547592 | 51.289505 | 48.179094 |
| Flash | 16 | 88.119072 | 88.080103 | 85.653939 | 85.897299 |
| Flash | 32 | 132.586657 | 133.150653 | 128.288959 | 128.766030 |
| Flash | 64 | 159.743312 | 155.485778 | 155.257137 | 155.940600 |
| Flash | 128 | 178.279794 | 172.947429 | 172.470265 | 172.661245 |
| Flash | 256 | 200.812360 | 194.983080 | 194.030163 | 194.220525 |

## 配置与产物

- [Pro 完整 E2E 配置](results/mimo_v26_decode_20261013/pro_tuned_fmoe.csv)
- [Flash 完整 E2E 配置](results/mimo_v26_decode_20261013/flash_tuned_fmoe.csv)

两份配置直接保留官方 E2E 输出的 us/err 字段；与上面的独立 ABBA 数值不是同一轮。
只更新各模型 M1/2/4 的3行，其余13行保持原配置。归档放在文档目录，**不参加
默认 config merge，也没有覆盖用户已经暂存的两份模型配置**。需要启用时可显式
选择该完整 CSV；不要与原模型 CSV 作为两个同 shape 来源一起合并。

本机完整产物保存在 `/tmp/aiter-blockscale-decode-20261013`：两模型的
`tuning.json`（各200候选）、`paired.json`（各36轮）、调优/配对日志、最终 JUnit、
旧 decode 前后输出与日志、12个特化的 IR/ISA。首版编译 API/layout 失败和
TOPK127 随机累加误差记录也保留。没有改测试容差、安装依赖或提交/暂存代码。
