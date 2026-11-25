# gfx9_fp8 的 A8W4 MoE（FP8 按 token 激活 × MXFP4 权重）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-25 07:04）· 目标：把 pyhip moe-ptpc-a8w4-r2-M17（811e03a47e）的 MoE A8W4 移植到 aiter（FlyDSL 0.5.3）并接入 aiter.fused_moe 分派：bench 85 例逐 case 耗时 ≤ M17 ×1.02（同卡、run_perftest 同协议、分进程交替测），同输入输出与 M17 逐位一致；另用一个里程碑采纳 int9-A2（1.25× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M1 decode 档：移植 M17 decode kernel（逐路由 Gate/Up、Down / Down pq、decode 行量化），M≤32 不排序
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->

本文只讲 A8W4（本目录的 `*_a8w4.py` 与入口 `aiter/ops/flydsl/moe_a8w4.py`）；同目录的 FP8 / BF16 两阶段底座不在范围内。

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
在 MI308X（gfx942）上提供 MXFP4 权重、FP8 按 token 激活的两阶段 FlyDSL MoE，逐位移植自 pyhip `moe-ptpc-a8w4-r2-M17`（811e03a47e）。
- 入口：`prepare_moe_a8w4_ptpc_weight`（加载时把标准 MXFP4 权重转成私有布局）与 `flydsl_moe_a8w4_ptpc`（bf16 进出，SiLU，可选 `swiglu_limit`）。
- 目标形状：Qwen3.8-Flash-Next TP2 / TP4（H2560 E513 topk11 I320 / 160→192）、MiMo-V2.6-Flash TP8 / TP4（H4096 E256 topk8
  I256 / 512）、DeepSeek-V4.1-Flash TP4（H5120 E384 topk6 I576，swiglu_limit=10）。
- 当前只有 prefill 一档：每个 M 都走排序 + 两个 prefill GEMM；decode、去重 decode、流式三档尚未移植（路线见第 6 节）。
- 验收：bench 85 例逐 case 耗时 ≤ M17 × 1.02（同卡、同一计时协议）；同输入时确定性路径的输出与 M17 逐位一致。
- 精度：对独立 fp32 参考 calc_diff ≤ 1e-3、最大误差 ≤ 4.5%·max|ref|；aiter fp8 per-token 只作性能参照（第 4 节）。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、gfx950；接入 `aiter.fused_moe` 分派在路线上。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。 -->
权重在寄存器里按每 32-K 组的 LUT 解成 fp8 走 fp8 MFMA；激活按行量化成 fp8，与参考的量化点逐位一致。
```text
加载时 prepare_moe_a8w4_ptpc_weight：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
flydsl_moe_a8w4_ptpc（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  Opus moe_sorting（BM64）→ row_quant_fp8(x，顺带 loc_ids 置 −1) → Gate/Up prefill（gather xq 行）→ mid f32
  → row_quant_fp8(mid) → Down prefill（× 通道因子 × ms × 路由权重，写 loc_ids）→ routes bf16 → sorted_sum（底座）→ out
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| 任意 M（当前唯一一档） | prefill：BM64·BN128·TK256 | decode / 流式档尚未移植 |
| M > 2048 | Gate/Up 改用 TK128 | LDS 与寄存器减半，大 M 占用更高 |
| I_pad ≥ 256 且排序块少 | Down 把 H 切给 2 / 4 个 WG（`DOWN_MIN_WGS`） | 块少时末轮 WG 大半空闲（MiMo M2048 Down 快 10%） |
| I > 512 且估计 WG > 80 | Down 的激活分两段经 LDS | I576 的 LDS 36 → 20 KiB，每 CU 1 → 2 个 WG |
| H ≥ 5120 且路由 ≥ 16384 | 两个 GEMM 用按 XCD 分块的 1-D grid | 同块与同 expert 相邻块共用 L2，快 1–4% |
| 给了 tile 参数 | 按参数选 BM / BN / TK | 调试与 tile 对照 |
核心 kernel 骨架（完整的 Gluon 风格写法、任务划分与逐拍流水线见 [kernels.md](kernels.md)）：
```text
Gate/Up prefill：grid (2I/128, 排序块)，4 wave 管 TILE_M 行 × 64 gate 与同号 64 up；权重进 VGPR 提前 2 tile，x 经 LDS 双缓冲
  每 lane 16 B 读相邻两行 → 每 128-K 步建 LUT → 解码 → MFMA；epilogue 跨 lane 换回 → SiLU(gate)·up → f32 写 mid
Down prefill：grid (切分 1/2/4, 排序块)，bm 行 mq 经 LDS（I>512 可分两段）装入后驻留 VGPR；逐 64 列块解码 → MFMA
  → × 通道因子 × 行 scale → bf16 经 LDS 写 routes；最后写有效路由的 loc_ids
行量化：每行 L 个 lane，整行进 VGPR；amax → scale = amax/240 → 倒数乘加两次 FMA 修正（正确舍入的商）→ cvt_pk_fp8
```
截断（`swiglu_limit` L>0）：Gate/Up 收尾在乘完通道因子与 xs 后、SiLU 前令 gate ≤ L、|up| ≤ L；L 是编译期常量（kernel 名加 `_swiglu`）。

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
主线是逐位移植 M17、只改 aiter 规则要求的部分：kernel 与 M17 只差 import 与 FlyDSL 0.5.3 的两处修正，布局 helper 与 gemm-a8w4 文字一致。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 经 LUT 解成 fp8 走 fp8 MFMA | 解成 bf16（A16W4） | gfx942 无 fp4 MFMA；bf16 的 MFMA 与解码 VALU 都翻倍 | delta > 11 的组按 0 |
| 私有布局按 MFMA K 序预排，加载时转一次 | kernel 内重排标准布局 | 每 lane 每 64-K 读 8 B 即 2 条 MFMA 操作数 | 需 prepare；同 dense GEMM 的 fd2 |
| 行量化的商逐位等于 IEEE 除法 | aiter HIP 量化（乘倒数） | 量化不翻转，与 M17 / 参考逐位一致 | 倒数 + 两次 FMA；scale < 2^-64 逐元素除 |
| 保留 M17 的 raw rocdl / buffer op，逐处注明缺的稳定 API | 改写成 typed API | 改写会变 ISA，危及逐位一致与性能 | 依赖 FlyDSL 0.5.3 的 raw 方言接口 |
| 编译句柄缓存按 device 与 kernel 读的模块常量作键 | 只按形状（M17） | 句柄只能在编译它的 GPU 上发射（AGENTS.md） | 形状不变时无额外编译 |
| 所有 M 先走 Opus 排序 + 行量化 | M17 的流式 prep（排序与量化一次 launch） | prep 属流式档、尚未移植；路由不重复时输出逐位相同 | 小 M 多一次 launch |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面。 -->
M ≥ 128 时 Qwen 为 aiter fp8 per-token 的 0.79–1.00 倍；M ≤ 8 还没有 decode / 流式档，是其 1.2–4.9 倍；输出与 M17 逐位一致。
| 档（M） | Qwen TP2 / TP4 ÷ aiter | 本实现 TB/s | aiter TB/s | 本实现 TFLOPS | aiter TFLOPS | 结论 |
|---|---|---|---|---|---|---|
| 1–8 | 1.24–3.58 / 1.70–4.85 | 0.14–1.17（≤22%） | 1.23–2.69（≤51%） | 0.5–4.8 | 2.4–6.0 | 缺 decode 档，排序与填充开销为主 |
| 16–64 | 0.89–1.03 / 1.05–1.14 | 1.08–1.75（≤33%） | 2.29–3.08（≤58%） | 5.0–12.5 | 5.7–11.9 | 缺流式档 |
| 128–1024 | 0.89–0.94 / 0.85–1.00 | 1.38–1.74（≤33%） | 2.16–2.90（≤55%） | 17–115 | 17–109 | 读权重为主 |
| ≥ 2048 | 0.87–0.94 / 0.79–0.94 | 0.15–1.08 | 0.21–1.71 | 143–266（≤44%） | 135–251（≤41%） | 算力与功耗墙 |
- MiMo / DeepSeek 只测本实现（aiter 无调优表）：M ≥ 2048 时 179–321 / 164–291 TFLOPS；冻结基线与逐 case 数据见下方生成块。
- 性能对照：同进程的 `aiter.fused_moe` per_Token（基线提交 ab0223ec4e），同一 BF16 权重按通道量化成 fp8（TP4 补零到 192）。
- 对照所选：只 Qwen 34 例，断言命中自带的 qwen38 调优表；M ≤ 8、M ≥ 2048 多为 FlyDSL gfx942，M128–1024 为 asm 1 阶段。
- 计时：两者同一 `bench_time_us`（`run_perftest` 101 次、profiler 只计 kernel 时间，权重在内轮换 32 GiB，预热 ≥18 ms）。
- 验收对照：pyhip M17（FlyDSL 0.5.0）由任务工具同卡分进程交替测，跑同一 `bench_time_us`，逐 case ≤ 1.02×，M3 起 85 例全量。
- 精度参考：测试文件的独立 fp32 参考（MXFP4 精确反量化，输入与中间按行 fp8）；判据见第 1 节，bench 最大误差 1.83%。
- 与 M17 逐位：85 个 bench 输入同 tile 选项下 85/85 相同；M17 的原子加路径自身不可逐位复现，那几档只按误差判据。
- 真实权重精度沿用 [pyhip 设计文档](https://github.com/luocheng25/pyhip/blob/moe-ptpc-a8w4-r2-M17/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)第 4 节：DeepSeek V4.1 真实权重下误差为模型参考（每 32 元素 UE8M0）的 0.48–0.98 倍。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类用例都在 [op_tests/test_flydsl_moe_a8w4.py](../../../../../../op_tests/test_flydsl_moe_a8w4.py)（参考、基本、其他、bench 四个注释区），只断言后续各档移植后仍成立的分派。

**基本功能测试**（每次改动先跑：`pytest op_tests/test_flydsl_moe_a8w4.py -m basic`，10 例，约 30 秒）
| 覆盖什么（每例断言误差与恰一个 Gate/Up、一个 Down kernel） | 用例 | 结果 |
|---|---|---|
| Qwen TP2 / TP4（E33 含融合 shared，I160 补零到 192）：M3 / 64 / 1031、257（后两例钉 prefill） | `test_tp2`、`test_tp4_padded` | 通过 |
| MiMo TP8 / TP4（E256）：M5 / 300 / 24 / 1600（M1600 钉 prefill） | `test_mimo[4 例]` | 通过 |
| DeepSeek TP4 + swiglu_limit=10（截断大量触发）：M8 / 64 | `test_deepseek_swiglu[2 例]` | 通过 |

**性能测试**（`python op_tests/test_flydsl_moe_a8w4.py`，85 个 case，约 2 分钟）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| prefill 档：5 个形状 × M = 1, 2, 4 … 32768 与 48；Qwen 另测 aiter | `<shape>_m<M>` | Qwen ÷ aiter 0.79–4.85；≤266 TFLOPS；全过判据 |

**其他覆盖测试**（只测前两类覆盖不了的；`-m "not basic"`，31 例，约 1 分钟；测 prefill 的用例必要时显式给 tile 参数）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| delta 11 / 12 清零边界、布局逐字节、行量化逐位（中点、小 scale）、loc 置 −1 | 随机数据碰不到边界；少量位错在判据内看不出 | `test_delta_flush_boundary` 等 4 个 | 通过 |
| tile 参数、半块交界、XCD / 两段 / 切分逐位、H5120 门槛 | 默认 tile、随机尾块、真实门槛碰不到交界 | `test_tile_variants` 等 5 个 | 通过 |
| CUDA graph；非目标形状（H4608、I1024） | 只做 eager；只有目标形状 | `test_graph_replay`、`test_shape_fallbacks` | 通过 |
| 报错（含路由 dtype、scale 形状）、gfx942 门禁、缓存键（device / 常量） | 只给合法输入、单卡、默认常量 | `test_rejects_bad_inputs` 等 3 个 | 通过 |
| 重复 expert、原地输出、路由 dtype 转换、M=0、TK128 截断、只截 gate 上侧 | 生成的输入碰不到 | `test_m_zero` 等 6 个 | 通过 |

未覆盖 / 跳过：与 M17 逐位（需 FlyDSL 0.5.0 与 pyhip，由任务工具核对）；I_pad > 1024（Down 激活超出 LDS，构建时断言）；多卡；gfx950（用例跳过）。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
主要差距是小 M 缺档；路线：M1 decode → M2 去重 decode（含 H5120 wide-dd）→ M3 流式 → M4 int9-A2（H5120 M6–32 改流式）→ M5 接入分派。
- 小 M 只有 prefill 档 → Qwen M ≤ 8 是 aiter 的 1.2–4.9 倍 → M1–M3 补齐三档，M3 起 85 例对 M17 全量验收。
- Opus 排序的 P0 阶段跨进程 8–13 µs → M ≤ 8 的 bench 跨进程波动约 ±7% → decode 档（M ≤ 32 不排序）移植后消失。
- 底座 `sorted_sum` 的编译缓存不按 device → 同进程多卡调用可能在错的 GPU 上发射 → 底座不在本任务范围，报给上游。
- 依赖 FlyDSL 0.5.3 的 raw 接口，门槛只在 MI308X（4 个 XCD）上调过 → 升级或换卡可能变 ISA、不再最优 → 重跑逐位对照与 bench。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../../../moe_a8w4.py)（两个入口的 docstring 写明接口、报错与数值约定）；kernel：[gemm1_a8w4.py](gemm1_a8w4.py)、
  [gemm2_a8w4.py](gemm2_a8w4.py)、[quant_a8w4.py](quant_a8w4.py)、[common_a8w4.py](common_a8w4.py)（私有布局的字节级定义）。
- kernel 设计（任务划分与流水线）：[kernels.md](kernels.md)；移植源的设计文档：[pyhip M17 README](https://github.com/luocheng25/pyhip/blob/moe-ptpc-a8w4-r2-M17/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)。
- 里程碑（每个提交写了改了什么、为什么）：分支 `luocheng/moe-a8w4` 的 `moe-a8w4-M<k>` tag。

<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · prefill 一档到底：A8W4 prefill 两段 kernel、权重准…
- M1 ⏳ 承诺 · 性能 · decode 档：移植 M17 decode kernel（逐路由 Gate/… · 预期 1.2×

下一步（本提交的承诺）：M1 decode 档：移植 M17 decode kernel（逐路由 Gate/Up、Down / Down pq、decode 行量化），M≤32 不排序
<!-- /mf:roadmap -->
