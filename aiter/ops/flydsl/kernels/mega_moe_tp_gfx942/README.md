# mega_moe_tp_gfx942（gfx942 TP MoE 层：FP8 激活（按 token）× MXFP4 routed + FP8 shared expert，all-reduce 模式）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-27 06:17）· 目标：让 aiter 的 MegaMoeTPGfx942（gfx942 / MI308X，A8W4：FP8 按 token × MXFP4）在带 shared expert 的整层上更快：以 mega-tp-a8w4-M7 为起点叠加 moe-a8w4-shared-M6（只留一套 kernel）。AR 模式：Qwen3.8-Flash-Next TP2/TP4 与 DeepSeek-V4.1-Flash 主干整层 TP4（E384+1 FP8 shared），M=1…32768 与 48，对冻结基线（每卡 shared-M6 + 单独 AR）几何平均 ≥1.10×；DP4+TP4 模式（comm_mode=ag_rs，只 DeepSeek 主干，每卡 M_r=1…8192 与 12）对冻结基线（AG + 每卡 shared-M6 + RS）几何平均 ≥1.20×；两者分别考核，逐 case 不慢于基线，routed-only 形状不慢于 mega-tp-a8w4-M7（1.122× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M0b DP 模式功能版（冻结前）：MegaMoeTPGfx942 实现 comm_mode="ag_rs"（x_local / out 为本卡 token，各卡 M_r 相同，padding 路由 −1）= 入口 AG（bf16）+ 每卡 flydsl_moe_a8w4_ptpc 算全部 4·M_r token + 出口 RS（RCCL / aiter 取快）；bench 加 15 个 ds_dp4_m<M_r> case（M_r 1…8192 与 12）与 a8w4_shared_agrs 对照，用例分区；之后一次冻结 AR 51 + DP 15 例
<!-- /mf:summary -->

<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->
## 1. 目标与范围
在 MI308X（gfx942）上提供与 `MegaMoeTP` 同接口的 TP MoE 层 `MegaMoeTPGfx942`：每卡跑单卡 A8W4 入口（可带 FP8 shared），再一次 all-reduce。
- 输入：各卡相同的 x bf16 [M, H] 与路由；有 shared 时路由为 [M, topk + 1]、末列为 E（ATOM 约定）。
- 输出：每卡 y bf16 [M, H] 为整层结果，各卡逐位相同。
- 权重：routed 复制、I 按 TP 切，收 `MegaMoeTP` 的 MXFP4 布局；shared 收本卡切片（FP8，逐行 f32 或 32×32 E8M0 块 scale）。
- 形状：Qwen3.8-Flash-Next TP2 / TP4（H2560、E513 含 1 个融合 MXFP4 shared、topk 11、每卡 I 320 / 160 补零到 192）。
- 形状：DeepSeek-V4.1-Flash 主干 TP4（H5120、E384 + 1 个 FP8 shared、topk 6 + 1、每卡 I576、SwiGLU limit 10）；M = 1…32768 的 2 的幂及 48。
- 验收：本版即基线；任务目标对它几何均值 ≥ 1.10×、逐 case 不慢，Qwen 34 case 不慢于 mega-tp-a8w4-M7；精度过 FP32 参考、各卡逐位相同。
- 不做：`comm_mode="ag_rs"`（第 6 节）、fused tail、gfx950 字段、非 bf16 的 x（都明确报错）；TP8 能跑但未测；不保证跨次运行逐位相同。
## 2. 总体设计
TP 层只做权重转换、校验与一次 all-reduce：本卡计算整段交给单卡入口 `flydsl_moe_a8w4_ptpc`（分档与 kernel 都在 gfx9_fp8）。
```text
构造 / set_weights：MegaMoeTP 布局 → 反 shuffle 成标准 MXFP4 → prepare_moe_a8w4_ptpc_weight（routed + 可选 shared，转一次）
构造（集合操作）：同一 (TP 组, gloo 组, 设备) 的实例共用一个通信器 = aiter CustomAllreduce + RCCL 组（引用计数）
forward(x, topk_weights, topk_ids, out=None)（可被 CUDA graph 捕获；M = 0 不发 kernel）：
  part = flydsl_moe_a8w4_ptpc(x, weight, topk_weights, topk_ids, output=part, swiglu_limit)    # 本卡部分和 bf16
  y = 字节 < AR_RCCL_MIN_BYTES[卡数] 且 CA 接受 ? CustomAllreduce(part)（新张量）: RCCL all_reduce(part)（原地）
  last_path = 单卡入口的 last_path + ("custom_all_reduce" | "rccl_all_reduce")
  各卡 y 逐位相同：两种 AR 下每个元素在各卡的求和顺序相同（用例逐位检查）
```
| M（51 case） | 本卡路径（单卡入口的档，kernel 依次） | 单卡入口为什么选它 |
|---|---|---|
| Qwen 1–2 | decode：gateup_decode → down_decode | 不排序：排序块几乎全是填充 |
| Qwen tp2 4–8、tp4 4–16 | decode，先量化 mid：gateup_decode → row_quant_decode → down_decode_pq | Down 槽对版：16 B 覆盖两槽 |
| Qwen tp2 16–32、tp4 32 | decode，x 也先量化：row_quant_decode → gateup_decode_pq → … → down_decode_pq | 免得每个 WG 各量化整行 |
| ds 1–2 | 混合 decode：shared 作末路，进同两个 kernel（gateup / down_decode_mixed） | 省 shared 的 split 链与相加 |
| ds 4–8 | decode 槽对 Down（M8 去重 Gate/Up）；shared 在侧流算 GEMM，折进 down_decode_pq_shared | 折叠省量化与相加 |
| Qwen 48–1024、tp4 2048、ds 16–1024 | 流式：stream_prep → gateup_stream（块 16 / 32 / 64）→ down_stream | 每 expert 行少，读权重为主 |
| tp2 2048、ds 2048 | prefill：stream_prep 排序 →（ds：shared GEMM）→ gateup / down_prefill → sorted_sum | prep 一次 launch 更快 |
| ≥ 4096 | prefill：Opus moe_sorting 排序、x 单独量化，其余同上 | 路由多时 prep 慢 |
- ds 的流式档 shared 也在侧流、Down 后 route_sum 合并；Qwen 的流式 Down 原子加进 prep 清零的输出（I < 512）。
- AR 按部分和字节分：Qwen TP2 M ≤ 1228、Qwen TP4 M ≤ 2662、ds M ≤ 1331 走 CA，更大的走 RCCL（阈值见第 3 节）。
- 本层没有自己的 kernel（[kernels.md](kernels.md)）；各档 kernel 的骨架、任务划分与流水线见 gfx9_fp8 的
  [README](../moe_gemm_2stage/gfx9_fp8/README.md) 与 [kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md)。
## 3. 关键决策
本卡计算原样复用 shared-M6 的单卡入口（只留一套 kernel）；跨卡求和先用一次单独的 AR，按字节取 CA 与 RCCL 中较快者。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| 只留 gfx9_fp8 一套 kernel，M7 的融合通信先删 | 两套 kernel 并存 | 两套是同一批 kernel 的重复移植，M7 那套不认 shared | Qwen 大 M 慢于 M7（第 4 节） |
| 本卡整段调单卡入口 | TP 层自己分派 kernel | 单卡分档已在 TP 切片形状上调过；TP 层零 kernel | 部分和多一次 bf16 写读 |
| AR 取快：TP2 ≥ 6 MiB、TP4 ≥ 13 MiB 走 RCCL | 总用 CA | 冷测 CA / RCCL µs：TP2 7.5 MiB 202 / 200，TP4 13.75 MiB 201 / 198 | 只测本机 |
| shared 收本卡切片，格式与校验同单卡入口 | TP 层自己切 shared | ATOM 已按 TP 持有切片；报错与单卡一致 | 切片由调用方负责 |
| routed 收 `MegaMoeTP` 布局，构造时反 shuffle | 只收标准 MXFP4 | 调用方按架构换类即可，不改权重加载 | 构造 / set_weights 多一次转换 |
## 4. 性能与精度
本版与同算法对照 ref 持平；对 M7 几何均值慢 4%：Qwen 流式档（48–512）更快，prefill 档慢 13–35%（缺融合通信）。
| 档（M） | 本÷M7 | 本÷ref | 本版最高 TB/s / TFLOPS（占峰值） | 对照最高 TB/s / TFLOPS | 结论 |
|---|---|---|---|---|---|
| Qwen decode 1–32 | 0.95–1.10 | 0.99–1.01 | 1.77（33%）/ 9.6（2%） | M7 1.77 / 9.5 | M7 把 AR 融进 Down，M1 快 4–10% |
| Qwen 流式 48–512 | 0.78–1.00 | 0.98–1.00 | 2.34（44%）/ 71.5（12%） | M7 2.13 / 71.5 | 块 16 + CA 时快 7–10%；M48 M7 选块 32 |
| Qwen 自量化 1024、tp4 2048 | 1.05–1.08 | 1.00–1.01 | 1.28（24%）/ 102（17%） | M7 1.38 / 110 | M7 用 tp_reduce 省掉单独 AR |
| Qwen prefill ≥2048（tp4 ≥4096） | 1.13–1.35 | 0.99–1.00 | 0.81（15%）/ 177（29%） | M7 1.00 / 239 | M7 归约融进 Down、SDMA 重叠 |
| ds decode 1–8 | — | 0.99–1.01 | 1.64（31%）/ 7.3（1%） | ref 1.66 / 7.4 | M7 不支持 shared；与 ref 持平 |
| ds 流式 16–1024 | — | 0.98–1.00 | 2.71（51%）/ 126（21%） | ref 2.71 / 125 | 同上 |
| ds prefill ≥ 2048 | — | 1.00 | 1.22（23%）/ 231（38%） | ref 1.21 / 231 | 同上 |
- 峰值按每卡 5.3 TB/s、606 TFLOPS；字节与 FLOPs 按本卡的 I 切片计，耗时取最慢的卡。
- 对照 M7（mega_tp_m7）：从 `MEGA_TP_M7_SRC` 私有加载 M7 源码，路径同 M7（decode_tp / Down_tp / SDMA 分块）。
- 对照 ref（a8w4_shared_ar）：树内单卡入口 + bench 自己冻结的 AR 规则；三者同进程、同协议（第 5 节）逐次交替计时。
- 精度参考：FP32 参考，calc_diff ≤ 1e-3、max 误差 ≤ 4.5%·max|ref|；51 case 最大 9.1e-5 / 1.97%，各卡逐位相同。
- ds_layer_tp4_m1 在整轮 bench 中受瞬时干扰（本版 492 µs、ref 333 µs），同卡复测 32.6 / 32.7 µs，表中用复测值。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类用例都在 [test_mega_moe_tp_gfx942.py](../../../../../op_tests/multigpu_tests/test_mega_moe_tp_gfx942.py)，全部通过；多卡用例共用一个 4 进程常驻池。
单卡入口的分档交界（门槛两侧、奇数 M）由 [test_flydsl_moe_a8w4.py](../../../../../op_tests/test_flydsl_moe_a8w4.py) 覆盖，下面两个用例命令都带上它。
有输出的用例都按 FP32 参考判据（calc_diff ≤ 1e-3、max 误差 ≤ 4.5%·max|ref|）校验，并查各卡输出逐位相同。

**基本功能测试**（`commands.correctness` = `pytest <文件> -m basic`，14 例约 80 秒；另跑单卡文件 `-m basic` 18 例约 60 秒）
| 覆盖什么 | 用例 | 结果 |
|---|---|---|
| Qwen decode：单槽 Down、槽对 Down（预量化 mid）、预量化 x 与 mid | tp2_m1、tp4_m8、tp2_m32 | 过 |
| shared decode：混合、侧流并进 Down、H5120 去重 Gate/Up | ds_layer_tp4_m1、_m4、_m8 | 过 |
| 流式：块 16、块 32 / 64（Gate/Up 自量化），ds 带侧流 shared | tp2_m256、tp4_m1024、tp4_m2048、ds_layer_tp4_m64、_m1024 | 过 |
| prefill：prep 排序、Opus 排序，ds 的 shared GEMM 在前 | tp2_m2048、tp4_m4096、ds_layer_tp4_m2048 | 过 |
- 每例另断言档位：首个 Gate/Up kernel、有无 shared、恰好一个通信段。

**性能测试**（`commands.bench` = `python <文件>`，51 个 case：3 形状 × M = 1…32768 的 2 的幂及 48；计时的输出都过精度与逐位检查）
| 覆盖什么 | case | 本÷对照 | 最高 TB/s / TFLOPS |
|---|---|---|---|
| Qwen TP2 / TP4 的全部档（decode、流式、prefill；CA 与 RCCL 两侧） | tp2_m*、tp4_m*（34） | ÷M7 0.78–1.35，÷ref 0.98–1.01 | 2.34 / 177 |
| DeepSeek 主干 + shared 的全部档（混合 decode、侧流折叠、流式、prefill） | ds_layer_tp4_m*（17） | ÷ref 0.98–1.01 | 2.71 / 231 |
- 协议：每次调用前写 512 MiB 冲掉 L2 / MALL、空闲一段并对齐各卡，4 组输入轮换，warmup 3；采样 30 次取中位数。

**其他覆盖测试**（只测前两类覆盖不了的；`commands.verify_extra` = `pytest <文件> -m "not basic"`，19 例约 60 秒；另跑单卡文件全量）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| MTP 层（E128、topk 3 + 1）：M4 仍走混合 decode，M1280 流式块 64 接 prefill Down | 前两类只取 SPEC 形状，ds 主干到不了这两档 | test_mtp_layer | 过 |
| AR 取快规则：阈值两侧（TP2 / TP4）的种类与结果 | 前两类不钉 CA / RCCL（以后融合会变） | test_all_reduce_rule（2 例） | 过 |
| `out=`（CA 拷回、RCCL 原地、带 shared）；M = 0 | 前两类不传 `out`、没有空批 | test_out_argument_and_empty_batch | 过 |
| 同一层跨档反复调用（含 shared 侧流）、`reset`、错误旗标 API | 前两类每个输入只算一次 | test_repeated_calls_and_reset | 过 |
| TP2 与 TP4 层（共用通信器）在同一进程交替调用 | 前两类按形状分批，不交替 | test_tp2_tp4_interleaved | 过 |
| `set_weights` 换 routed 与 shared 权重后用新权重 | 前两类不换权重 | test_set_weights | 过 |
| CUDA graph 捕获与重放：decode、侧流 shared、流式、prefill、CA 与 RCCL | 前两类都是 eager 调用 | test_cuda_graph（7 例） | 过 |
| 不传 group / cpu_group 时自建 gloo 组，`close` 释放 | 前两类都显式传组 | test_default_communicators | 过 |
| 非法配置与参数，含 shared 的缺项、dtype、格式、形状、NaN、E8M0 255 | 前两类只走合法输入 | test_rejected_arguments | 过 |
| 路径判定本身：融合通信、shared 步骤、坏路径 | 判定放行一切时前两类也照过 | test_path_checks（纯 CPU） | 过 |
| 真实 shared（DeepSeek-V4.1 layers 2 / 37，TP4 切片）：误差 ≤ 模型量化的 1.5 倍 | 前两类用随机逐行权重 | test_real_shared_experts_tp4（2 例） | 过 |

未覆盖 / 跳过（一行一条，写原因）：
- 少于 4 张 gfx942 时多卡用例跳过（常驻池拉不起），只剩 test_path_checks；没有真实权重时 test_real_shared_experts_tp4 跳过。
- 单卡文件全量 230 过、3 跳过：本机没有 FP4PTPC 与 mtp.0 的权重数据。
- 错误旗标置位后的报错：本版没有 kernel 置位，触发不了；只测旗标 API 与 `reset` 清零。
- TP8 与 3 个目标形状以外的形状：没有用例（接口按 H % 512、I % 32 放行）。
- 跨次运行逐位相同：不要求（第 1 节），不测；多 stream 并发 forward 不测。
## 6. 风险与待定
- Qwen 大 M 慢于 M7（M7 把通信融进 Down 与末个 kernel）→ M1 起把 decode_tp / tp_reduce / Down_tp / SDMA 分块移植到本 kernel（带 shared）。
- DP4+TP4（`comm_mode="ag_rs"`）未实现、构造时报错 → 下一步在本层加入口 AG + 每卡单卡入口 + 出口 RS，与 AR 模式一起冻结基线。
- AR 阈值只在本机（MI308X × 4、当前 ROCm / RCCL）冷测 → 换机器或版本时按 test_all_reduce_rule 的尺寸复测。
- bench 的 M7 对照与移植后的 SDMA 各占一份用户态 SDMA 队列，同一引擎能否共存未验证 → 移植 SDMA 时先测。
- 本机 GPU 常有外部租户：受干扰时单个 case 可慢数倍 → 比较只信同次运行、逐轮交替的比值，离群值复测。
## 7. 进一步阅读
- 代码入口：[mega_moe_tp_gfx942.py](../../mega_moe_tp_gfx942.py)（类、AR 规则、接口约定与报错清单在 docstring）。
- 单卡入口：[moe_a8w4.py](../../moe_a8w4.py)（`prepare_moe_a8w4_ptpc_weight` / `flydsl_moe_a8w4_ptpc` 的 docstring 是接口契约）。
- kernel 设计：[kernels.md](kernels.md)；计算 kernel 见 [gfx9_fp8/kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md)。
- 里程碑（每个提交写了改了什么、为什么）：本分支提交历史。

<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · 新基线（功能版）：在 M7 上带进 moe-a8w4-shared-M6，只留…
- M0b ⏳ 承诺 · 功能 · DP 模式功能版（冻结前）：MegaMoeTPGfx942 实现 comm_m…

下一步（本提交的承诺）：M0b DP 模式功能版（冻结前）：MegaMoeTPGfx942 实现 comm_mode="ag_rs"（x_local / out 为本卡 token，各卡 M_r 相同，padding 路由 −1）= 入口 AG（bf16）+ 每卡 flydsl_moe_a8w4_ptpc 算全部 4·M_r token + 出口 RS（RCCL / aiter 取快）；bench 加 15 个 ds_dp4_m<M_r> case（M_r 1…8192 与 12）与 a8w4_shared_agrs 对照，用例分区；之后一次冻结 AR 51 + DP 15 例
<!-- /mf:roadmap -->
