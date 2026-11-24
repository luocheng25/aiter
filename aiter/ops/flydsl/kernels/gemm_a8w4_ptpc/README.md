# gemm_a8w4_ptpc（gfx942 dense GEMM：按 token fp8 激活 × mxfp4 权重）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-24 06:27）· 目标：aiter FlyDSL dense A8W4 GEMM on MI308X (gfx942): Y[M,N] = X[M,K] W^T, bf16 out; X bf16 quantized per token to FP8 E4M3FNUZ, W MXFP4 (E2M1 + one E8M0 per 32 K) converted once at load; Qwen3.8-Flash-Next dense linear layers at TP2/TP4 (6 (N,K)) x M = 1..32768: correct everywhere, small M near the HBM bound (clearly faster than aiter FP8 per-token GEMM), large M near the FP8 MFMA bound（1.05× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M1 Small-M path (M <= 32): single-launch decode-style kernel with in-kernel per-token quant and K split across waves
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
在 MI308X（gfx942）上做 dense GEMM Y = X·Wᵀ：bf16 X 按 token 动态量化成 fp8 e4m3fnuz，W 为 mxfp4，fp32 累加，输出 bf16。
- 输入：x bf16 [M, K]（任意 M、连续）；权重为 checkpoint 的 mxfp4（fp4x2 [N, K/2] + E8M0 [N, K/32]），加载时转换一次。
- 形状：Qwen3.8-Flash-Next dense 线性层 TP2 / TP4 的 6 个 (N, K)（N 2560–8192，K 1536–3072）× M = 1…32768。
- 约束：K % 256 == 0、N % 64 == 0、权重 < 2 GiB，其他形状与非 gfx942 明确报错；无 host 同步，可被 CUDA graph 捕获。
- 验收：每 case 耗时 ≤ 1.05× 同一 aiter 的 fp8 ptpc GEMM，M ≤ 32 时 ≤ 0.7×，M > 2048 达到 compute bound（第 4 节）。
- 精度：对照测试文件内的独立 fp32 参考，逐元素按行 1% 容差（第 4 节）。
- 不做：bias、fp16 输出、已量化输入的接口、gfx950 / gfx1250、W4A16。

## 2. 总体设计
权重在寄存器里按每组 LUT 解成 fp8、走 fp8 MFMA；现在一档到底：先量化 X，再用一个 GEMM kernel 覆盖所有 M。
```text
加载时 prepare_gemm_a8w4_ptpc_weight：mxfp4 → fd2 布局（按 MFMA K 序排的 fp4 码、每 32-K 组 delta、每通道因子）
flydsl_gemm_a8w4_ptpc(x, weight, out=None)（无 host 同步）：
  quant_x：x bf16 [M, K] → xq fp8 [M, K] + xs f32 [M]（scale = amax/240，商逐位等于 IEEE 除法）
  gemm：xq、xs、fd2 权重 → y bf16 [M, N] = acc · 通道因子 · xs
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| 所有 M（M = 0 不启动 kernel） | quant_x → gemm（BM64·BN128） | 先求正确；分档留给后续里程碑 |

核心 kernel 骨架（完整的任务划分与逐拍流水线见 [kernels.md](kernels.md)）：
```text
quant_x：每行 L 个 lane（K % 512 == 0 时 64，否则 32），CTA 256 线程管 256/L 行；16 B 读 → 行内 max → scale → 量化 → 8 B 写
gemm：grid (⌈N/128⌉, ⌈M/64⌉)，4 wave 管 64 token × 128 通道（每 wave 两个 16 通道块，全部 64 token）
  xq tile 经 LDS 双缓冲，fd2 权重直读 VGPR，两者都提前 1 个 K tile（128 K）预取
  每 128-K 步每 lane 建一个 LUT → fp4 解码成 fp8 → MFMA 16x16x32（解码与 LDS 读排进 MFMA 之间）
  收尾：acc × 通道因子 × xs → bf16；≥ M 的行由 buffer 边界丢弃，≥ N 的通道跳过
```

## 3. 关键决策
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 解成 fp8，走 fp8 MFMA | 解成 bf16 走 bf16 MFMA | 无 fp4 MFMA；fp8 算力为 bf16 的 2 倍 | 激活须量化成 fp8 |
| fd2：组 delta 进 LUT，通道因子收尾乘 | 每组 E8M0 现场乘 | 每 8 个码解码只要 2 次 v_perm | delta ≥ 12 的组置 0 |
| 先量化 X 再 GEMM（两个 kernel） | GEMM 内量化 | 量化可逐位校验；GEMM 沿用移植的 kernel | 多一次启动与 xq 读写 |
| 一个 tile 配置覆盖所有 M | 按 M 分档 | 先求正确，作冻结基线 | 小 M 只有 N/128 个 CTA |
| 每个 CTA 把 buffer 基址移到本 tile 首行 | 全局 32 位偏移 | 任意 M 不溢出；越界行读 0、写丢弃 | 无 |

## 4. 性能与精度
现状：96 个 case 输出都正确，但各档都慢于对照，小 M 最慢；冻结基线与逐 case 数据由 mf 冻结后填入下方生成块。
| M（6 形状） | 本实现 ÷ 对照（几何均值，范围） | 本实现（几何均值） | 对照（几何均值） | 结论 |
|---|---|---|---|---|
| ≤ 32 | 2.43×（1.89–3.60） | 0.22 TB/s（峰值 4%） | 0.99 TB/s（19%） | CTA 太少，远离访存上限 |
| 64–2048 | 1.32×（0.97–2.60） | 136 TFLOPS（22%） | 179 TFLOPS（30%） | M ≤ 128 仍受 CTA 数限制 |
| ≥ 4096 | 1.12×（1.05–1.17） | 338 TFLOPS（56%） | 379 TFLOPS（63%） | tile 与流水线未调 |
- 上表是 bench 自测（每 case 一次 `run_perftest`，未经 mf 配对复核）；TB/s 按各自最小数据量，峰值 5.3 TB/s、606 TFLOPS。
- 对照：同一 worktree（aiter 上游 main `162173e89`）的 fp8 ptpc，即 `per_token_quant_hip(x, fp8)` + `gemm_a8w8_bpreshuffle`，
  与 SGLang gfx942 `apply_fp8_ptpc_linear` 的调用一致；权重 = 同一份 mxfp4 精确反量化（含置 0 规则）后按通道量化成 fp8。
- 对照选中的 kernel：TP2 三个形状命中调优表的 CK kernel（tp2_o 的 M32768 用默认），TP4 全部用默认 CK；bench 逐 case 打印。
- 计时：两边同进程、逐 case 交替、同协议（aiter `run_perftest`，输入深拷贝轮换，一次调用的全部 kernel 时间相加）。
- 精度参考：测试文件内独立 fp32 参考（X 同一量化点，W 精确反量化并施加同一置 0 规则，fp32 matmul，关 TF32）。
- 判据：逐元素 |Y−ref| ≤ 1%·|ref| + 1%·该行 max|ref|，calc_diff ≤ 1e-5，无 NaN/Inf；对照用它自己的量化操作数按同一判据校验。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类测试都在 [op_tests/test_gemm_a8w4_ptpc.py](../../../../../op_tests/test_gemm_a8w4_ptpc.py)，每个输出都对照独立 fp32 参考。

**基本功能测试**（`pytest op_tests/test_gemm_a8w4_ptpc.py -m basic`，6 例，约 12 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| 6 个形状各一个 M（1–4096，含非 2 的幂） | `test_gemm` ×6 | 通过 |
| profiler 看到的 kernel = 分派结果 | 同上 | 通过 |

**性能测试**（`python op_tests/test_gemm_a8w4_ptpc.py`，96 个 case，被计时的输出逐 case 校验）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| 唯一一档：6 形状 × M = 1…32768（2 的幂） | `<形状>_m<M>` | 96 例校验通过；÷ 对照见第 4 节 |

**其他覆盖测试**（`pytest op_tests/test_gemm_a8w4_ptpc.py -m "not basic"`，14 例，约 14 秒，峰值约 9 GB 显存）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| M%64=63/1、N%128=64、K 循环 0–2 次 | 目标形状没有；bench 的 M 是 2 的幂 | `test_tile_boundaries` ×3 | 通过 |
| 大 M：字节偏移过 2³¹ / 2³²、grid.y > 65535 | bench 最大 M = 32768，偏移远小于 2³¹ | `test_m_past_32bit_offsets` | 通过 |
| 量化逐位：全零行、舍入中点、极小 scale、-0 | 少数 fp8 码翻转在 GEMM 容差内 | `test_quant_bitexact` ×2 | 通过 |
| prepare 逐字节布局：16 个码、delta 0–15 | 只看 GEMM 结果时布局错也可能读对 | `test_prepare_layout_bitexact` | 通过 |
| prepare 接受非连续切片（TP 行切的 K 段） | 前两类都传整块连续权重 | `test_prepare_noncontiguous_slices` | 通过 |
| delta 11 精确、delta 12 置 0 | 随机权重的组 delta 只有 0–3 | `test_delta_flush_boundary` | 通过 |
| 报错：dtype、形状、E8M0=255、非 gfx942 | 前两类只给合法输入 | `test_rejects_bad_inputs` | 通过 |
| `out=` 填 NaN 哨兵，且不写越界 | 前两类由算子分配输出 | `test_out_preallocated` | 通过 |
| M = 0 | 前两类 M ≥ 1 | `test_m_zero` | 通过 |
| CUDA graph 捕获与重放 | 前两类都是 eager 调用 | `test_cuda_graph_replay` | 通过 |
| 编译缓存：形状交替、回到旧形状 | bench 每个形状只顺序访问一次 | `test_compile_cache` | 通过 |

未覆盖 / 跳过：
- 非 gfx942 只用替身模拟架构判断（本机只有 gfx942）。
- 多 GPU 时每卡各一份编译句柄、x 与权重不在同一卡：测试只用一张卡，没有用例。
- 权重接近 2 GiB 上限时的正确性：目标权重最大 10.5 MB，这样的 prepare 要十几 GB 显存；只测了拒绝一侧。

## 6. 风险与待定
- 小 M（≤ 32）为对照的 1.9–3.6 倍（目标 ≤ 0.7×）：只有 N/128 个 CTA、无 split-K、量化单独启动 → 小 M 里程碑做融合量化与 split-K。
- 中间档（64–2048）为对照的 0.97–2.6 倍，M ≤ 128 最慢 → 中间档里程碑按 M 分档选 tile。
- 大 M（≥ 4096）为对照的 1.05–1.17 倍、峰值的 48–60%，未达 compute bound → 大 M 里程碑加大 tile、解码与 MFMA 重叠。
- compute bound 的阈值未定：冻结后用同卡纯 MFMA 微基准测出持续上限，随第一次计划给出。
- delta ≥ 12 的组置 0：组 scale 不到通道最大的 2⁻¹¹，真实权重少见；出现时有精度损失（参考施加同一规则）。

## 7. 进一步阅读
- 代码入口：[gemm_a8w4_ptpc.py](../../gemm_a8w4_ptpc.py)（接口与报错见 docstring）；权重布局：[layout.py](layout.py)
- kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)；GEMM 的 region 表与不变量：[gemm.py](gemm.py) 模块 docstring
- 里程碑（每个提交写了改了什么、为什么）：见下方生成块

<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · Functional A8W4 GEMM: fd2 MXFP4 weight …
- M1 ⏳ 承诺 · 性能 · Small-M path (M <= 32): single-launch d… · 预期 1.5×

下一步（本提交的承诺）：M1 Small-M path (M <= 32): single-launch decode-style kernel with in-kernel per-token quant and K split across waves
<!-- /mf:roadmap -->
