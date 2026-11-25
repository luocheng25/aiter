# mega_moe_tp_gfx942（gfx942 TP MoE 层：fp8 激活（按 token）× mxfp4 权重，all-reduce 模式）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-25 13:07）· 目标：让 aiter 的 MegaMoE TP 在 gfx942（MI308X）上跑 A8W4：激活按 token 量化成 FP8 × MXFP4 权重，experts 复制、inter_dim 按 TP 切，MoE 计算与 TP 通信融合；Qwen3.8-Flash-Next TP2/TP4 与 DeepSeek-V4.1-Flash TP4，M=1…32768 的 2 的幂及 48，一次完整 MoE 层（含通信）的耗时逐 case 不慢于两个对照（aiter fp8 ptpc fused_moe + 单独 all-reduce；pyhip 2stage A8W4 + 单独 all-reduce）中较快的；主流程框架（MegaMoE TP / 2stage+融合通信 / #8530 改 TP）由实验决定（1.08× 相对冻结基线）· 已完成 2 个里程碑（含本提交 M1a） · 最佳累计 — · 下一步：M1 融合通信第一版：M≥48 把 RS+AG 并进最后一个 kernel（L1：部分和写到 owner 卡、owner 按 rank 顺序求和后写回各卡），M≤16 用 decode TP 尾部（f32 部分和写进各卡收件箱、按 rank 顺序求和），M=32 保留 AR
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

## 1. 目标与范围
在 MI308X（gfx942）上提供与 `MegaMoeTP` 同接口的 TP MoE 层 `MegaMoeTPGfx942`：本卡 A8W4 两阶段 MoE + 一次 all-reduce。
- 输入：每卡相同的 x bf16 [M, H] 与路由（topk_weights f32、topk_ids）；输出：每卡 y bf16 [M, H] 为整层结果，各卡逐位相同。
- 权重：experts 复制、inter_dim 按 TP 切；收 `MegaMoeTP` 的 mxfp4 布局（`shuffle_weight(16,16)` + `e8m0_shuffle`），加载时转一次。
- 计算：x 按 token 量化成 fp8（amax/240）→ Gate/Up（mxfp4 在寄存器里解成 fp8 走 fp8 MFMA）→ SiLU / SwiGLU → 中间激活按行量化 → Down。
- 形状：Qwen3.8-Flash-Next TP2 / TP4（H2560、E513 含 1 个融合 shared、topk11、每卡 I 320 / 160）。
- 形状：DeepSeek-V4.1-Flash TP4（H5120、E384、topk6、I576、swiglu_limit 10）；M = 1…32768 的 2 的幂及 48，共 51 case。
- 验收：逐 case 耗时不慢于对照 1、2（第 4 节）中较快者，目标相对冻结基线几何均值 1.08×；精度过 fp32 参考判据，各卡输出逐位相同。
- 只做 `comm_mode="ar"`；不做 ag_rs、fused tail、TP8、gfx950 / gfx1250、已量化输入、跨次运行逐位确定，这些与其他架构都明确报错。

## 2. 总体设计
本卡先算部分和（pyhip M17 的 A8W4 两阶段 kernel 原样移植，按 M 分 decode / 流式 / prefill 三档），再单独 all-reduce。
```text
构造 / set_weights：MegaMoeTP 布局 → 标准 mxfp4 → prepare_a8w4_weights：私有布局（按 MFMA K 序排的 fp4 码、每 32-K 组一个
  delta 字节、每通道因子），I 补零到 64 的倍数（Qwen TP4 160 → 192）
forward(x, topk_weights, topk_ids)（无 host 同步，可被 CUDA graph 捕获）：
  part = fused_moe_a8w4(x, 本卡切片)：bf16 [M, H] 本卡部分和（档位见下表）
  y = aiter CustomAllreduce(part)（should_custom_ar 为真：≤ 64 MiB，aiter 默认上限）；否则 RCCL all_reduce(part)（原地）
  CA 与 RCCL 对每个元素在各卡按同一顺序求和（或一卡求和后广播），所以各卡 y 逐位相同
```
| 51 case 的 M | 路径（kernel 顺序；之后都接 AR） | 为什么 |
|---|---|---|
| 1–2（三个形状） | decode：gateup_decode（WG 内量化 x）→ down_decode（WG 内量化 mid） | 不排序，两次 launch |
| tp2 4–8、tp4 4–16、ds 4 | 同上，但 mid 先 row_quant_decode，Down 换槽对版 down_decode_pq | 路由多时每个 WG 重复量化整行不划算 |
| tp2 16–32、tp4 32 | x 也先 row_quant_decode，Gate/Up 换 gateup_decode_pq | Gate/Up WG 数 ≥ 2560 时单独量化 x 更快 |
| ds 8–32 | row_quant_plan_decode（量化 x、去重计划）→ gateup_decode_dd → row_quant_decode → down_decode_pq | 同 expert 共读权重 |
| 48–1024（tp4 到 2048） | 流式：stream_prep → gateup_stream →［row_quant_fp8］→ down_stream →［route_sum］ | 每 expert 行少：权重只读一次 |
| tp2 2048、ds 2048 | prefill：stream_prep → gateup_prefill → row_quant_fp8 → down_prefill → sorted_sum | 路由少时 prep 更快 |
| ≥ 4096 | prefill：Opus 排序 → row_quant_fp8 → gateup_prefill → row_quant_fp8 → down_prefill → sorted_sum | 免 prep 的平方开销 |
- 流式细节：I < 512（Qwen）的 Down 用 bf16x2 原子加进 prep 清零的 out，省掉 route_sum；I576（DeepSeek）写 routes 再 route_sum。
- 流式细节：排序块 > 16、H < 4096 且累加器 8·(I/64)·(块/16) ≤ 96 个 VGPR 时 Gate/Up 一个 WG 管全部 I、自己量化 mid。
- AR：H2560 在 M ≥ 16384、H5120 在 M ≥ 8192 时超过 CA 的 64 MiB 上限，走 RCCL；其余走 CA。
- 核心 kernel 骨架（每个 kernel 的 Gluon 风格分阶段写法、任务划分与流水线见 [kernels.md](kernels.md)）：
```text
decode Gate/Up：grid (I/16, M·topk)，4 wave = gate / up × K 前后半；每 lane 一条 16 B 覆盖两个 64-K 块，部分和经 LDS 相加
decode Down：grid (H/16, M)，wave 按槽取（槽对版一条 16 B 覆盖两个槽）；各 wave 部分和经 LDS 按固定顺序相加 → bf16 写 out
去重 Gate/Up：同 grid，只有组首 WG 计算（同 expert 的 2–3 条路由的 x 行进 LDS，共用一次权重读）
流式 prep：前 ⌈M·topk/256⌉ 个 WG 各统计全部路由（LDS 原子）后放置自己的 256 条，其余 WG 量化 x 行（并清零 out）
流式 Gate/Up：grid (I/(16·G·W), 排序块)，权重（non-temporal）与激活直读 VGPR、提前 P 个 64-K 步；按有效 16 行子块数特化
流式 Down：grid (H/(16·R·W), 排序块)，本块激活驻留 VGPR；逐通道块解码 → MFMA → bf16 写 routes 或两块凑一次原子加进 out
prefill Gate/Up：grid (2I/128, 排序块)，4 wave 管 64 行 × 64 gate 与同号 64 up；权重进 VGPR 提前 2 tile，x 经 LDS 双缓冲
prefill Down：64 行 mq 经 LDS（I576 分两段）装入后驻留 VGPR；逐列块解码 → MFMA → bf16 写 routes，并写每条路由的排序行号
```

## 3. 关键决策
本卡计算原样复用已调好的 pyhip M17 kernel（最终 ISA 逐条相同），通信先用 SGLang 在 AMD 上的默认 all-reduce，融合留给后续。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| 框架 B：两阶段 A8W4 + 通信 | A：MegaMoeTP（gfx950）；C：#8530 改 TP | A 用 gfx950 专有指令；C 的形状断言挡掉 51 case、小中 M 慢 1.25–2.2× | 通信要自己融合 |
| kernel 原样移植，只换 import、删不用的分支 | 按 aiter 新 API 重写 | 13 个代表配置 17 个 kernel 的 ISA 与源相同 | raw rocdl / llvm / asm 保留（注明原因） |
| 不融合 AR：CA（≤ 64 MiB）否则 RCCL | RS / AG 并进末个 kernel | 先证明移植不走样；融合预计 decode 快 5–12%、M > 32 快 0–12% | AR 占耗时最高 37% |
| 另建类，接口同 `MegaMoeTP` | 在 `MegaMoeTP` 里按架构分派 | gfx950 代码零改动；两边权重格式与可调字段不同（#8530 同例） | 调用方按架构选类 |
| 收 `MegaMoeTP` 的 mxfp4 布局，加载时转私有布局 | 直接收私有布局 | 调用方不改权重加载 | 每层转换一次（host 侧，非热路径） |
| kernel 放 `kernels/moe_a8w4/`，同源的 common 与 sorted_sum 直接 import gfx9_fp8 | 放在本目录 | 以后单卡 A8W4 MoE 直接复用同一份 | — |

## 4. 性能与精度
现状：51 case 全过精度判据且各卡逐位相同；与对照 2 持平（逐 case 0.988–1.008，几何均值 0.999），Qwen 34 case 全部快于对照 1（0.66–0.94）。
| 档 | ÷ 对照 2 | ÷ 对照 1 | 本实现 TB/s | 本实现 TFLOPS | 对照 1 TB/s | 对照 1 TFLOPS | 受限于 |
|---|---|---|---|---|---|---|---|
| decode | 0.988–1.008 | 0.66–0.87 | 0.46–2.22（9–42%） | 1.4–10.6（≤ 2%） | 0.73–2.30（14–43%） | 1.2–6.7（≤ 1%） | 权重读 |
| 流式 | 0.990–1.006 | 0.67–0.93 | 0.72–2.51（14–47%） | 8.0–107（1–18%） | 1.13–2.87（21–54%） | 6.1–90（1–15%） | 权重读 |
| prefill | 0.992–1.002 | 0.86–0.94 | 0.10–1.29（2–24%） | 109–223（18–37%） | 0.15–1.33（3–25%） | 93–167（15–28%） | MFMA |
- 对照 2 `a8w4_2stage_ar`：移植源 pyhip M17（`811e03a` + FlyDSL 0.5.3 补丁 `45ff9f5`）的 `fused_moe_a8w4` + 同一 AR。
- 对照 2 与本实现是同一套 kernel 与分档，TB/s、TFLOPS 只差耗时比（≤ 1.2%），表中不再单列。
- 对照 1 `aiter_fp8_ptpc_ar`（只 Qwen）：本分支 aiter `fused_moe`（fp8 per-token，Qwen3.8 fp8 ptpc 调优表）+ 同一 AR。
- 对照 1 选中：M ≤ 16 多为 FlyDSL 16×16×16，32–1024 多为 asm 一阶段 32×320 / 32×192，≥ 2048 多为 FlyDSL 64×128×256。
- 计时：三者同一进程、同一冷协议（每次调用前各卡清 512 MiB 缓存、GPU 空转盖住下发、128 B CA 对齐），取最慢卡、30 轮中位。
- 顺序：本实现与对照 2 逐轮交替；对照 1 单独一轮，因为紧跟 aiter fp8 调用的 A8W4 调用快 4–5%，交替会偏向排在它后面的一方。
- 吞吐：flops = 6·M·topk·H·I（本卡真实 I）；bytes = 用到的 expert 的本卡权重（各自格式）+ x 读 y 写；峰值 5.3 TB/s、606 TFLOPS。
- 精度：独立 fp32 参考（各卡部分和之和），判据 calc_diff ≤ 1e-3 且 max|y − ref| ≤ 4.5%·max|ref|；实测最大 8.3e-6 与 1.11%。
- 目标按较快的对照 2 定（bench 的第一个 MF_REF）；冻结基线（mf 在 GPU 空闲时测）与逐 case 数据见下方生成块。

<!-- mf:perf -->
基线来源（mf 记录）：11-25 12:47 冻结于提交 `e19c1fc3a6`（里程碑 M0，tag `mega-tp-a8w4-M0`），bench 跑 3 次取中位数，GPU0,1,2,3，噪声 ±0.18%；bench 命令 `/root/.myflow-r2/tasks/mega-tp-a8w4/venv/bin/python op_tests/multigpu_tests/test_mega_moe_tp_gfx942.py`。对照实现与精度参考的来源见本节手写部分。

（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类用例都在 [test_mega_moe_tp_gfx942.py](../../../../../op_tests/multigpu_tests/test_mega_moe_tp_gfx942.py)，全部通过；多卡用例共用一个 4 进程常驻池，有输出的用例都按 fp32 参考判据校验并查各卡逐位相同。
- 路径断言只固定档位与通信段：档位 = 首个 GateUp kernel；通信段恰好一个（末尾一个 CA / RCCL，或名字含 `tp` 词的融合 kernel 且无 AR）。
- GateUp 之后的 kernel 不固定（通信融合会改它们）；AR 种类只在不融合时由 `out=` 与默认通信组两例检查。

**基本功能测试**（每次改动先跑：`commands.correctness` = `pytest <文件> -m basic`，13 例，约 72 秒；每例另断言档位与通信段，防止静默换档）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| decode 两次 launch（WG 内量化） | tp2_m1、ds_tp4_m2 | 过 |
| decode 预量化 mid / 预量化 x 与 mid | tp4_m8、tp2_m32 | 过 |
| decode 去重 Gate/Up（H5120） | ds_tp4_m16 | 过 |
| 流式 Qwen 原子加：块 16、块 32 / 64 自量化 | tp2_m256、tp4_m48、tp4_m2048 | 过 |
| 流式 H5120：块 16 / 32 + route_sum | ds_tp4_m512、ds_tp4_m1024 | 过 |
| prefill：prep 排序 / Opus 排序 | tp2_m2048、tp4_m4096、ds_tp4_m4096 | 过 |

**性能测试**（`commands.bench` = `python <文件>`，51 个 case：3 形状 × M = 1…32768 的 2 的幂及 48）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| decode × 三形状（CA） | M 1–32，18 个 | ÷对照 2 0.988–1.008，≤ 2.22 TB/s |
| 流式 × 三形状（CA） | M 48–1024（tp4 到 2048），19 个 | ÷对照 2 0.990–1.006，≤ 2.51 TB/s |
| prefill × 三形状（CA 与 RCCL） | M ≥ 2048（tp4 ≥ 4096），14 个 | ÷对照 2 0.992–1.002，≤ 223 TFLOPS |
- 被计时的输出都查各卡逐位相同，A8W4 两者另按 fp32 参考校验（对照 1 只查有限值）；不过则记失败、退出码非 0。

**其他覆盖测试**（只测前两类覆盖不了的；`commands.verify_extra` = `pytest <文件> -m "not basic"`，15 例，约 100 秒）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| 每个分档门槛两侧与奇数 M | 前两类只取 2 的幂与 48，只在门槛一侧 | test_dispatch_boundaries（3 例） | 过 |
| `out=` 写回（CA 拷回、RCCL 原地）、M = 0 | 前两类不传 `out`、没有空批 | test_out_argument_and_empty_batch | 过 |
| 同一层跨档反复调用、`reset`、错误旗标 API | 前两类每个输入只算一次 | test_repeated_calls_and_reset | 过 |
| TP2 与 TP4 层在同一进程交替调用 | 前两类按形状分批，不交替 | test_tp2_tp4_interleaved | 过 |
| `set_weights` 换权重后用新权重 | 前两类不换权重 | test_set_weights | 过 |
| CUDA graph 捕获与重放（CA 与 RCCL） | 前两类都是 eager 调用 | test_cuda_graph（5 例） | 过 |
| 不传 group / cpu_group 时自建通信组 | 前两类都显式传组 | test_default_communicators | 过 |
| 非法参数 / 形状、非 gfx942、NaN scale | 前两类只走合法输入 | test_rejected_arguments（20 种） | 过 |
| 路径判定本身：融合、不融合、两者都有 / 都没有 | 前两类只见不融合路径；判定放行一切也照过 | test_path_checks（纯 CPU） | 过 |

未覆盖 / 跳过：
- 少于 4 张 gfx942 时多卡用例跳过（常驻池拉不起），只剩 test_path_checks。
- 同一 GateUp 档内的子路径（decode 是否预量化 mid、是否 route_sum、prefill 的排序方式）不固定：换了不报，只查结果。
- 3 个目标形状以外的形状：没有用例（接口按 H % 512、I % 32 放行，见第 6 节）。
- 跨次运行逐位相同：不要求（第 1 节），不测；Qwen 流式的 bf16 原子加顺序不定。

## 6. 风险与待定
- 通信不融合（AR 占耗时 3–37%）：逐 case 只能与对照 2 持平（18 case 慢 ≤ 0.8%），验收未达成 → 下一步 decode 尾部融合、RS + AG 并进末个 kernel。
- aiter CA 的 `IPCBufferPool` 序号按进程计数：同进程先后建不同组的 CA 会让各卡 store key 不一致而挂死 → 建 CA 前在组内取最大值对齐。
- 别的形状没有用例：接口按 H % 512、I % 32 放行，分档门槛只为这 3 个形状实测 → 用新形状前先补用例与 bench case。
- raw rocdl / llvm / inline asm 未换成 FlyDSL 0.5.3 的 typed API（换了可能改 ISA）→ 后续单独迁移并再做 ISA 比对。
- GPU 0–3 常有外部租户：有租户时耗时涨 2–3 倍、同次比值抖动 ±10%，比较前先看 rocm-smi 记录（第 4 节数据在空闲时测得）。

## 7. 进一步阅读
- 代码入口：[mega_moe_tp_gfx942.py](../../mega_moe_tp_gfx942.py)（类与通信）、[moe_a8w4.py](../../moe_a8w4.py)（权重转换与分档）。
- 计算 kernel：[kernels/moe_a8w4/](../moe_a8w4/)；kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)。
- 里程碑（每个提交写了改了什么、为什么）：本分支提交历史。

<!-- mf:roadmap -->
- M0 ✅ 已推送 · 功能 · 功能版（框架 B）：pyhip 2stage A8W4 移植到 aiter（k… · [提交](https://github.com/luocheng25/aiter/commit/e19c1fc3a6675ba05ee9035ca31a1ff660d930a5)
- M1 ⏳ 承诺 · 性能 · 融合通信第一版：M≥48 把 RS+AG 并进最后一个 kernel（L1：部… · 预期 1.046×
- M1a ✅ 本提交 · 维护 · 用例：kernel 路径断言放开通信段（为 M1 融合通信让路）

下一步（本提交的承诺）：M1 融合通信第一版：M≥48 把 RS+AG 并进最后一个 kernel（L1：部分和写到 owner 卡、owner 按 rank 顺序求和后写回各卡），M≤16 用 decode TP 尾部（f32 部分和写进各卡收件箱、按 rank 顺序求和），M=32 保留 AR
<!-- /mf:roadmap -->
