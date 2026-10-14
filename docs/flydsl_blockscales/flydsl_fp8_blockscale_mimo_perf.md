# MiMo FP8 blockscale MoE：原生调优与移植后性能

日期：2026-10-13。对应 [FP8 blockscale 移植](flydsl_fp8_blockscale_moe.md)，
Aiter 基线 commit `4848005c` 加本次未提交的移植改动；性能测试期间未再修改 kernel、
正式调优配置或依赖。

## 结果

固定 `HIDDEN_SIZE=6144`、`INTER_SIZE(per TP)=256`、`Experts=384`、`TOPK=8`。
I 已经是 TP 切分后的值，不再次除以 TP。BF16 输入/输出，FP8 E4M3FN
W128x128/A1x128，separated SiLU，无 clamp，routing weight 在 Down 应用。

下表是**调优完成后独立 ABBA 复测**的整链 GPU 耗时，不是搜索过程的最低值。

| Batch | 最优原生 us | E2E tune 后 us | 加速比 | 耗时降低 | E2E 选中路径 | 原生 / E2E 有效 TFLOPS |
|---:|---:|---:|---:|---:|---|---:|
| 1024 | 447.838 | 444.390 | 1.008× | 0.77%* | 保留 CK 两阶段 M32、NT | 172.63 / 173.97 |
| 2048 | 540.782 | 541.194 | 0.999× | -0.08%* | 保留 CK 两阶段 M64、NT | 285.92 / 285.70 |
| 4096 | 759.105 | 730.956 | 1.039× | 3.71% | M128 persistent，N128，OC8 | 407.37 / 423.06 |
| 8192 | 1205.838 | 997.637 | 1.209× | 17.27% | M128 independent，N128，OC4 | 512.90 / 619.94 |
| 16384 | 2143.363 | 1757.196 | 1.220× | 18.02% | M128 persistent，N128，OC4 | 577.11 / 703.93 |
| 32768 | 4146.455 | 3512.172 | 1.181× | 15.30% | 8-wave，N64，OC1 | 596.63 / 704.38 |

\* 1024、2048 前后是**相同原生配置**，差异是重复测量波动，不是本次改动收益。
这两个 batch 的最快新增候选在搜索中为 536.618、593.706 us，未超过原生基线，
所以官方 E2E tuner 没有替换它们。

有效 GEMM 工作量为 $F=6M\times8\times6144\times256$；
$TFLOPS=F/(t_{us}\times10^6)$。不把 sort、量化或激活额外操作算入有效 FLOPs。
加速比为原生耗时/E2E 耗时；耗时降低为 $1-t_{E2E}/t_{native}$。

## 原生基线不是默认配置

1. 官方原生 tuner 使用 `TUNE_ONLY=''`、`OPUS_ONLY=0`、`OPUS_SKIP_CKTILE=0`，
   无 kernel regex/exclusion 限制，`--all --mp 1`，六个 shape 全部调优。
   每个 shape 53 项：18 ASM Stage1、20 CK stage 项（包含 NT）、15 ASM 单阶段项
   （9 FP8 输入、6 BF16 输入，包含 FLAT）。其他原生组对此格式没有适用项。
2. 官方 stage 调优共 318 项，其中 172 项失败，**不是全部通过**。Stage1 的独立参考
   存在二次量化问题；仅按 stage 结果筛选会丢掉可用的 CK 整链。原生 FLAT 在
   M4096 失败，在 M8192 的两个变体发生 GPU memory fault/worker timeout。
3. 因此额外通过真实 public `fused_moe` 和独立整链参考，复核每个 shape 的 85 个
   Stage1/Stage2 匹配组合及单阶段配置：共 510 条记录，330 通过，104 数值不通过，
   72 被原生 N512/I256 支持检查拒绝，4 个已知失败的 FLAT 组合不再运行。
   每个通过项测两轮，取较小排序值对应的配置；每项排序值是两轮耗时的中位数。
4. 最终原生基线：M1024 为 CK M32/NT，M2048 为 CK M64/NT；其余四个 shape 为
   FP8-input ASM 单阶段 M64/N256。这是**当前原生 tuner 实际生成搜索空间内，
   经整链检查的最快配置**，不是所有理论参数的全局最优保证。
5. 以该基线为初始 CSV，官方 `--all --e2e_tune --mp 1 --warmup 5 --iters 101`
   搜索 20 个新增配置 × 6 个 shape：112 项通过，8 项 M32768 M128 容量拒绝。
   最终替换 4 个 shape，保留 2 个原生配置。

未修改原生参考、kernel 或容差来让候选通过。原生 stage 失败、FLAT fault 和所有
整链失败均保留在原始日志中，不把失败候选记为有效速度。

## 测量范围与环境

- 同一块 MI350X/gfx950、256 CUs，PCI `0000:75:00.0`。本机 HIP logical 0 对应
  rocm-smi card3，不能把这两个编号混用。所有 GPU 测量串行执行。
- Python 3.10.12，Torch `2.9.1+rocm7.2.0.git7e1940d4`，HIP `7.2.26015-fc0010cf6a`，
  **FlyDSL 0.3.4.1**，实际导入当前 Aiter checkout；没有 PyHIP 调用依赖。
- 使用官方 `FmoeTuner.generate_data` 的随机输入/权重及非强制均衡 routing。
  每个 batch 的 ABBA 共用同一份原始张量及参考，真实 CSV 决定 public dispatch；
  对原生检查 kernel 名，对新增整链检查绑定的 config，防止 heuristic fallback。
- 每轮调用未修改的 [run_perftest](../aiter/test_common.py#L242-L260)：5 warmups、
  101 iterations、2 组轮换参数。顺序为原生/调优后/调优后/原生，各自两轮取中位数。
  轮换副本由计时器每次重新 deepcopy，**不是所有地址都固定不变的同址对照**。
- 计时器累加完整 `fused_moe` 的 GPU kernel 时间；保留其现有
  [首轮剔除和 IQR 处理](../aiter/test_common.py#L294-L364)，没有替换 timer 或事后挑轮次。
  这是 MoE 整链 GPU 口径，不是 CPU wall time、CUDA Graph wall time 或整模型时延。
  包括 sort、两次量化、Gate/Up、Down、counter reset、inverse/reduce 等实际执行的
  GPU 工作；不含预先生成的 gating/topk、权重量化/shuffle、JIT、参考/校验、CPU
  提交间隙或 TP 通信。
- 测量前后该卡均为 0% GPU use、约 298 MB VRAM。未改频率、功率或 PTL，
  **未锁频、未做 PTL 验证**，不能据此作受控硬件环境或跨平台性能保证。

## 正确性

所有最终赢家的 warm output 和实际计时返回的 output 都通过原生整链标准：有限值、
非空结果，且 `err_ratio <= 0.1` 或 `logits_diff <= 0.01`，未放宽阈值。
`err_ratio` 使用原 `atol=rtol=0.01`；`logits_diff` 是
$1-2\sum xy/(\sum x^2+\sum y^2)$，不能与 strict allclose 或 rel_l2 混称。

- 原生 CK M1024/M2048：元素超差比例约 1.38%/1.28%，`logits_diff` 约 $10^{-5}$。
- 原生单阶段 M4096～32768：元素超差比例约 31.2%，通过的是原有
  `logits_diff` 分支，值约 $7.2\text{--}7.3\times10^{-5}$；**不声称 strict allclose**。
- 新增路径 M4096～32768：元素超差比例 0.43%～0.49%，`logits_diff`
  为 $4.29\text{--}4.41\times10^{-6}$。

M32768 的 sort256 保守容量为 360448 行；M128 packed output 本体占
4,429,185,024 bytes（4.125 GiB），已超过现有 32-bit buffer 限制，尚未计 OOB margin。
8-wave routed output 为 3 GiB，合法。测试保留了 M128 的保护断言，没有修改边界。

## 原始配对轮次与配置

以下保留计时器返回的全部 24 个轮次聚合值，单位 us；它们不是 101 次内部迭代的
未过滤 trace。表头顺序就是执行顺序 A/B/B/A。

| Batch | A1 原生 | B1 调优后 | B2 调优后 | A2 原生 | 相邻配对比值中位数 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 453.375306 | 443.939495 | 444.840540 | 442.300480 | 1.007772 |
| 2048 | 545.449212 | 544.714619 | 537.673475 | 536.115753 | 0.999226 |
| 4096 | 762.412580 | 730.249000 | 731.662720 | 755.796940 | 1.038515 |
| 8192 | 1218.806040 | 996.401957 | 998.871400 | 1192.869020 | 1.208712 |
| 16384 | 2133.728395 | 1756.770278 | 1757.620878 | 2152.997859 | 1.219762 |
| 32768 | 4163.324580 | 3507.877408 | 3516.466082 | 4129.586000 | 1.180603 |

相邻配对比值中位数是 `median(A1/B1, A2/B2)`；结果主表使用
`median(A1,A2)/median(B1,B2)`，两者在这里接近，但定义不同。

归档的 [原生赢家 CSV](results/flydsl_blockscale_mimo_20261013/native_best.csv) 和
[E2E 赢家 CSV](results/flydsl_blockscale_mimo_20261013/e2e_tuned.csv)
可分别通过 `AITER_CONFIG_FMOE` 使用。它们在文档目录，不被默认配置扫描；
**只选一个，不要把两份相同 shape 的配置合并**。CSV 的 `us` 保留搜索阶段结果，
不是本报告最终配对耗时。原生归档中的 `us1/us2/err1/err2/tflops/bw` 零值是整链
排名脚本的占位字段，不代表单 stage 耗时或零误差；实际正确性见上文。

完整本机 run 目录：/tmp/aiter-blockscale-mimo-20261013。保留六个输入 shape、
318 项官方调优日志和明细、510 条整链排名 JSON、120 项 E2E 搜索日志、
配对 JSON/日志（含每轮正确性及实际 dispatch）以及临时运行脚本。
没有更新正式配置、暂存文件、创建 commit 或推送；PyHIP 工作区保持不变。