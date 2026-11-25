# gfx9_fp8 的 A8W4 MoE（FP8 按 token 激活 × MXFP4 routed + FP8 shared expert）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-25 18:29）· 目标：DeepSeek-V4.1-Flash 主干与 MTP 层（TP4 / TP8）整层（routed MXFP4 + shared FP8 逐通道，含量化、排序、规约）每个 M ≤ aiter 融合 FP8 PTPC ×1.02 且不慢于 A8W4 split；moe-a8w4 原 85 例逐例 ≤ 起点 moe-a8w4-M4 ×1.02；完成 pyhip M17 遗留项 1–8（性能项与端到端精度目标随计划确认）（1.1× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M1 功能：入口校验与屏蔽路由（topk_ids=-1 各档不读权重、整行屏蔽含 shared、graph 重放）
<!-- /mf:summary -->
<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->
本文只讲 A8W4（本目录的 `*_a8w4.py` 与入口 `aiter/ops/flydsl/moe_a8w4.py`）；同目录的 FP8 / BF16 两阶段底座不在范围内。

## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
在 MI308X（gfx942）上提供 MXFP4 routed（可带 FP8 逐行 shared expert）的两阶段 FlyDSL MoE，主目标是 DeepSeek-V4.1-Flash 整层。
- 入口：`prepare_moe_a8w4_ptpc_weight`（加载时转私有布局，可带 shared）与 `flydsl_moe_a8w4_ptpc`（bf16 进出）；`aiter.fused_moe` 认标记转调。
- 整层形状：主干 / MTP（E384 / 128 + shared，topk 6 / 3 + 1）× TP4 / TP8（I576 / 288→320），H5120，swiglu_limit 10。
- routed-only 形状（原 85 例）：Qwen3.8-Flash-Next TP2 / TP4、MiMo-V2.6-Flash TP8 / TP4、DeepSeek-V4.1-Flash TP4。
- 性能验收：整层每个 M ≤ aiter 融合 FP8 PTPC × 1.02 且不慢于 A8W4 split；原 85 例逐例 ≤ 起点 moe-a8w4-M4 × 1.02。
- 精度验收：对独立 FP32 参考 err ≤ 4.5%·max|ref|、calc_diff ≤ 1e-3；shared 真实权重的误差 ≤ 模型参考的 1.5 倍。
- shared 权重：收 FP4PTPC 的每行 F32 scale 或官方 32×32 E8M0 块（prepare 折叠，整矩阵与 FP4PTPC@97cc9b32 逐字节相同）。
- 不做：EP / expert_mask、bias、SiLU 以外的激活、预量化输入、gfx950、非 MXFP4 的 Qwen shared；经 fused_moe 时这些与指定 tile 报错。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。 -->
权重在寄存器里按每 32-K 组的 LUT 解成 fp8 走 fp8 MFMA；激活按行量化成 fp8（量化点与参考逐位一致）；按 M 与每 expert 行数分四档；shared 走 aiter 稠密链。
```text
加载时 prepare_moe_a8w4_ptpc_weight：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
  [shared FP8：块 scale 先折叠成每行 2 的幂 → e4m3fn 转 fnuz（scale ×2）→ I 补零 → shuffle(16,16)，作 expert E]
aiter.fused_moe(x, w.w1, w.w2, …, Silu, per_1x32, per_Token)：首行见 w1 / w2 的标记 → _fused_moe_route（门禁、配对、查参数）→ ↓
flydsl_moe_a8w4_ptpc（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  decode：[行量化(x)] → Gate/Up decode（WG 管一条路由）→ mid f32 → [行量化(mid)] → Down decode（WG 管一个 token）→ out
  去重 decode：计划 + 行量化(x) + 清零 out → Gate/Up 去重（组首 WG 算同 expert 的 2–3 条路由）→ 行量化(mid) → Down 去重
    → bf16x2 原子加进 out；M·I 小时只用两个 kernel；H5120 只去重 Gate/Up，逐 token 的槽对 Down 直接写 out
  流式：prep（排序成 16 / 32 / 64 行的块 + 行量化(x) [+ 清零 out]）→ Gate/Up 流式 → [行量化(mid)] → Down 流式
    → bf16x2 原子加进 out，或 bf16 routes → route_sum → out
  prefill：[prep | Opus 排序 BM64 → 行量化(x，loc_ids 置 −1)] → Gate/Up prefill（gather xq 行）→ mid f32 → 行量化(mid)
    → Down prefill（× 通道因子 × ms × 路由权重，写 loc_ids）→ routes bf16 → sorted_sum（底座）→ out
  带 shared（expert E，最后一列）：routed 走前 topk 列；shared 走 ATOM split（aiter 量化 → GEMM → SiLU(L) → 量化 → GEMM）→ out += 权重·shared
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| M ≤ 32，H ≤ 4096 或 5120，mid 行放得进 LDS | decode：不排序、不规约 | 排序块几乎全是填充；M=1 快 2.6–5.8 倍 |
| （decode）M·topk·I/16 ≥ 2560；M ≥ 4 且 w2 < 4 GiB | 先量化 x；先量化 mid，Down 用槽对版 | 免得每个 WG 各量化整行；16 B 覆盖两槽 |
| （decode）H4096、I%128=0、M ≥ 2；路由 ≤ 1.5E 时到 M64 | 按 expert 去重；M·I < 3072 时两个 kernel | 同 expert 共读权重，比逐路由快 6–24% |
| （decode）H5120、I%128≠0、M8–32 | 只去重 Gate/Up，Down 用槽对版 | I576 非 128 倍数；对逐路由 0.99–1.11× |
| M > 32 且每 expert 平均 ≤ 22 行（H ≥ 4096 为 48） | 流式：排序块 16（≤12 行且路由 ≥ 4E/3）/ 32（≤28 行）/ 64 | 读权重为主；上限取实测交叉 |
| （流式）BM ≥ 32、H < 4096 且累加器 ≤ 96 VGPR（H2560 BM64 到 56 行） | Gate/Up 一个 WG 管全部 I、尾部量化 | 省一次行量化；H4096 时轻 wave 更快 |
| （流式 Down）I < 512；BM64 且 H ≥ 4096 | 原子加进 prep 清零的 out；换 prefill Down | 免 route_sum（I512 反慢）；MiMo TP4 M1024 快 5% |
| （H5120）流式路由 ≥ 4096；prefill 路由 ≥ 16384 | 流式上限 16 行，BM16 不看路由；按 XCD 分块 | 20 行起 prefill 快 3–8%，BM16 快 11%；XCD 快 1–4% |
| 其余 M；给了 tile 参数 | prefill BM64·BN128；行多（给参数时 M > 2048）TK128 | 多行共用权重；TK128 的 LDS 与寄存器减半 |
| （prefill）H ≥ 2560 且 M·topk ≤ 24576（H5120 为 12288） | 排序与 x 量化改用 prep（一次 launch） | TP2 M2048 快 2.1%；路由多则 prep 慢 |
| （prefill）I ≥ 256 且排序块少；I > 512 且 WG > 80 | Down 的 H 切给 2 / 4 个 WG；激活分两段 | 末轮少空闲（M2048 快 10%）；每 CU 2 WG |
| 带 shared：M ≤ 1024 且 out 不与 x 重叠；其余 | shared 链与其路由权重各占一条 side stream；当前流先算 shared | ATOM 的双流阈值；原地须先读 x |
- fused_moe 入口：标记带同一次 prepare 的其余张量（含 shared）、不引用 w1 / w2；`shared_expert_id` 只收 −1 或 E；全表见 [`_fused_moe_route`](../../../moe_a8w4.py#L1895)。
- 核心 kernel 骨架（完整的 Gluon 风格写法、任务划分与逐拍流水线见 [kernels.md](kernels.md)）：
```text
Gate/Up prefill：grid (2I/128, 排序块)，4 wave 管 TILE_M 行 × 64 gate 与同号 64 up；权重进 VGPR 提前 2 tile，x 经 LDS 双缓冲
  每 lane 16 B 读相邻两行 → 每 128-K 步建 LUT → 解码 → MFMA；epilogue 跨 lane 换回 → SiLU(gate)·up → f32 写 mid
Down prefill：grid (切分 1/2/4, 排序块)，bm 行 mq 经 LDS（I>512 可分两段）装入后驻留 VGPR；逐 64 列块解码 → MFMA
  → × 通道因子 × 行 scale → bf16 经 LDS 写 routes；最后写有效路由的 loc_ids
Gate/Up decode：grid (I/16, M·topk)，4 wave = gate / up × K 前后半，每 lane 16 B 覆盖两个 64-K 块；x 行经 LDS 广播，部分和经 LDS 相加
Down decode：grid (H/16, M)，wave 按槽取 mid 行（槽对版 16 B 覆盖两槽）→ 解码 → MFMA；各 wave 部分和经 LDS 树形相加 → bf16 写 out
去重 Gate/Up：grid 同逐路由版，只有组首 WG 计算：组员的 x 行进 LDS（A 的行 m、8+m 取组员 m），本 expert 的权重只读一次
去重 Down：grid (H/128, M·topk)，组首 WG 读 8 个 16 通道块，各组员 × 通道因子 × 行 scale × 路由权重 → bf16x2 原子加进 out
流式 Gate/Up：grid (I/(16·G·W), 排序块)，wave 管 G 个 gate 块与同号 up 块；权重（non-temporal）与激活直读 VGPR，提前 P 个 64-K 步
  按有效 16 行子块数特化；W > 1 时激活经 LDS 环共享（H5120 每 lane 一次 16 B 跨 4 / 2 步）；尾部量化时 4 wave 管全部 I
流式 Down：grid (H/(16·R·W), 排序块)，wave 管 R 个 16 通道块 × 全部 I；本块激活驻留 VGPR（4 wave 时经 LDS 装入）
  权重提前 4 块 → 解码 → 两条 MFMA 链 → × 通道因子 × 行 scale → bf16 写 routes，或两块凑一次 bf16x2 原子加进 out
流式 prep：前 ⌈M·topk/256⌉ 个 WG 各统计全部路由（LDS 原子加）后放置自己的 256 条，其余 WG 量化 x 行（并清零 out）
行量化：整行进 VGPR，amax → scale = amax/240 → 商逐位等于 IEEE 除法 → cvt_pk_fp8；decode 版与计划 kernel 每行一个 wave / WG
```

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
routed 逐位移植 M17（保留其 raw rocdl / buffer op，ISA 与 M17 逐行相同）；shared 先按 ATOM split 接上，作为融合前的冻结基线。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 按 MFMA K 序预排、经 LUT 解成 fp8 走 fp8 MFMA | 解成 bf16；运行时重排 | 无 fp4 MFMA；每 64-K 读 8 B 即 2 条操作数 | 需 prepare；delta>11 按 0 |
| 行量化的商逐位等于 IEEE 除法 | aiter HIP 量化（乘倒数） | 量化不翻转，与 M17 / 参考逐位一致 | 倒数 + 两次 FMA；scale < 2^-64 逐元素除 |
| M ≤ 32 不排序、每 lane 16 B；H4096 去重；行少时 16–64 行小块流式 | 全走 BM64；逐路由读 | M128 时 BM64 块内约 95% 是填充 | 四套 kernel；原子加不逐位确定 |
| fused_moe 按 w1 / w2 的标记转调，tile 自动 | 内部转换缓存；按 block_size_M 钉 tile | 内部转换要双份权重；钉 tile 时 DS M=1 慢 3.5 倍 | 标记是 Python 属性 |
| shared 进 prepare 作 expert E（E385/K7 约定） | 调用时另传 shared、topk 只放 routed | ATOM 原版与 FHMoE 同约定，ATOM 只换权重 | shared 须在最后一列 |
| shared 先按 ATOM split 用 aiter 稠密算子 | 直接融合进 A8W4 kernel | 先有可测、可冻结的整层基线 | 比 split 对照多一次相加与列拷贝 |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面。 -->
整层还没有可下结论的计时（PTL 自整机 reset 起 Disabled，基线等它恢复再冻结）；routed 85 例代码同起点，下表沿用起点的复核数字：
| 档（M） | Qwen TP2 / TP4 ÷ aiter | 本实现 TB/s | aiter TB/s | 本实现 TFLOPS | aiter TFLOPS | 结论 |
|---|---|---|---|---|---|---|
| 1–2 逐路由 decode | 0.84–0.87 / 0.85–0.96 | 0.78–1.19（≤23%） | 1.23–1.85（≤35%） | 2.9–4.8 | 2.4–4.1 | launch 与固定开销为主 |
| 4–32 逐路由 decode | 0.68–0.94 / 0.68–0.78 | 1.37–2.09（≤40%） | 1.98–3.05（≤58%） | 5.5–10.5 | 4.3–9.4 | 访存指令条数受限 |
| 2–48 去重 decode | —（只 MiMo / DS） | 1.16–2.43（≤46%） | — | 4.9–15.0 | — | 比全走 prefill 快 1.10–2.67 倍 |
| 48–2048 流式 | 0.73–0.91 / 0.68–0.87 | 0.98–2.26（≤43%） | 1.55–3.03（≤57%） | 12–151（≤25%） | 9.5–131（≤22%） | 解码 VALU 受限 |
| ≥ 2048 prefill | 0.84–0.94 / 0.80–0.90 | 0.15–1.11 | 0.21–1.70 | 174–263（≤43%） | 141–248（≤41%） | 算力与功耗墙 |
- 冻结基线：将是 M0（shared 按 ATOM split），见生成块第一行；原 85 例对起点 moe-a8w4-M4 只查回退（逐例 ≤ 1.02 倍）。
- aiter_fp8_ptpc 对照：同 worktree（ab0223ec4e）的 `aiter.fused_moe` per_Token，主干 E385/K7、MTP E129/K4，TP8 的 I 补零到 320。
- 对照所选：无 H5120 调优行，都是 `impl__flydsl_gfx942__` 默认：M ≤ 256 为 BF16 激活的 16_16_16 / 16_128_64，≥ 512 为 64_128_128。
- a8w4_split 对照：routed-only A8W4 + aiter CK `gemm_a8w8_bpreshuffle` 链，M ≤ 1024 双流，不计最后相加（ATOM 并进 all-reduce）。
- 计时：整层 68 例的四个实现同进程、CUDA graph 重放墙钟（计入双流并发）；原 85 例 `run_perftest`；都过 PTL 门禁。
- 精度参考：独立 FP32 参考（MXFP4 / FP8 精确反量化，x 与 mid 逐 token FP8）；整层 68 例最大误差 3.4%·max|ref|、calc_diff ≤ 1.6e-4。
- shared 真实权重（官方 layer 2 / 37 的 TP4 切片）：误差是模型参考（1×32 UE8M0 激活 + 32×32 块权重）的 0.70–1.00 倍。

<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类用例都在 [op_tests/test_flydsl_moe_a8w4.py](../../../../../../op_tests/test_flydsl_moe_a8w4.py)（参考、基本、其他、bench 四个注释区），分派断言与 M17 的最终分派一致。

**基本功能测试**（每次改动先跑：`pytest op_tests/test_flydsl_moe_a8w4.py -m basic`，17 例，约 30–50 秒）
| 覆盖什么（每例断言误差与恰一个 Gate/Up、一个 Down kernel） | 用例 | 结果 |
|---|---|---|
| Qwen TP2 / TP4（E33，I160 补零到 192）：M3 逐路由 decode、M64 流式、M1031 / 257 prefill | `test_tp2` ×3、`test_tp4_padded` | 通过 |
| MiMo TP8 / TP4（E256）：M5 去重两个 kernel、M24 去重四个 kernel、M300 流式、M1600 prefill | `test_mimo` ×4 | 通过 |
| DeepSeek TP4 + swiglu_limit=10（截断大量触发）：M8 只去重 Gate/Up、M64 流式 | `test_deepseek_swiglu` ×2 | 通过 |
| `aiter.fused_moe`：Qwen M3、整层 E385/K7（`shared_expert_id`=E，M8）与直接调用同 kernel、逐位相同 | `test_fused_moe_entry` 等 2 例 | 通过 |
| 整层 + shared（全量 E，截断触发）：主干 TP4 M1 / 64 / 2048（串行）、TP8 M4（补零）、MTP 块格式 M16 | `test_layer` ×5 | 通过 |

**性能测试**（`python op_tests/test_flydsl_moe_a8w4.py`，原 85 例 + 整层 68 例 = 153 例；`--quick` 只跑整层 24 例；每例也校验输出）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| 逐路由 decode：Qwen M1–32、MiMo M1、DeepSeek M1–4（17 例） | `tp2_m1` 等 | Qwen ÷ aiter 0.68–0.96；≤2.09 TB/s |
| 去重 decode：MiMo M2–48、DeepSeek M8–32（15 例） | `mimo_tp8_m2` 等 | 无对照；1.16–2.43 TB/s（≤46%） |
| 流式：Qwen M48–1024（TP4 到 2048）、MiMo M64–1024、DeepSeek M48–1024（29 例） | `tp2_m48` 等 | Qwen ÷ aiter 0.68–0.91；≤2.38 TB/s |
| prefill：M2048 由 prep 排序（4 例）、M ≥ 4096 由 Opus 排序（20 例） | `tp2_m2048` 等 | Qwen ÷ aiter 0.80–0.94；≤319 TFLOPS |
| 整层 + shared（4 形状 × M1…32768、48）：decode M1–4、去重 M8–32、流式 M48–1024、prefill ≥ 2048 | `ds_layer_tp4_m1` 等 68 例 | 待冻结 |

**其他覆盖测试**（只测前两类覆盖不了的；`-m "not basic"`，150 例，约 4 分钟）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| 逐路由 decode 交界：M4、11/12、19/20、32/33（M33 流式）；topk 8/17 | 不在交界、不查 kernel | `test_decode_dispatch_boundaries` 等 2 个 | 通过 |
| decode 重复 expert 与 int64 路由、w2≥4 GiB 回退、graph、缓存键 | 输入规整、w2 小、eager、单卡 | `test_decode_graph_replay` 等 4 个 | 通过 |
| 去重交界：M·I=3072、M16/17、I384、topk16 预量化、E256 M48/49 | bench 无这些 M，不查 kernel | `test_decode_dedup_dispatch` 等 3 个 | 通过 |
| H5120：M7 逐路由、M32/33 上限、M12 两轮装载 x、w2 回退、E256 M33 仍 BM16 | bench 无 M7/12/33，w2 小 | `test_decode_wide_hidden` ×6 | 通过 |
| 去重计划逐位；重复 expert（去重、流式）；去重 graph、缓存键 | 位错在判据内；输入规整、eager、单卡 | `test_decode_plan_quant_bitexact` 等 4 个 | 通过 |
| 流式 22/48/56 行上限、E513 grid 收缩；H5120 流式、prep、XCD、两段门槛 | 不在交界、不查 kernel | `test_stream_dispatch_boundary` 等 4 个 | 通过 |
| prep / Opus 排序的路由数上限、重复 expert；流式接 prefill Down 切 1/2/4 | 不在交界；只一种切分 | `test_prefill_prep_routes_limit` 等 2 个 | 通过 |
| 逐位：布局、delta、各版行量化、prep 排序、尾部量化、ACT_STEPS、loc 置 −1、XCD / 两段 | 随机数据碰不到；位错在判据内 | `test_row_quant_bitexact` 等 9 个 | 通过 |
| prefill 的 tile 参数、半块交界；非目标形状 7 种（含流式 Down 共享 tile 的 LDS 上限） | 只跑默认 tile 与目标形状 | `test_shape_fallbacks` 等 3 个 | 通过 |
| graph（流式、prep / Opus + prefill）、报错、三个入口的 gfx942 门禁、prefill 与流式缓存键 | 只 eager、合法输入、单卡 | `test_graph_replay` 等 5 个 | 通过 |
| Opus 排序重复 expert 留后一槽、原地输出、路由 dtype、M=0；各类 Gate/Up 的截断 | 输入规整、另给 out；bench 几乎不截断 | `test_inplace_output` 等 7 个 | 通过 |
| fused_moe 各档（确定性逐位、原子加对参考、DS 截断）；graph 重放 | 基本只测一个 decode；bench 直接调用 | `test_fused_moe_matches_api` ×6 等 7 个 | 通过 |
| fused_moe 参数透传与各类报错；未标记权重原样下传、fp4x2 提示 prepare；标记生命周期 | 基本只传默认参数、带标记权重 | `test_fused_moe_rejects` 等 5 个 | 通过 |
| shared 折叠 / fnuz / 补零逐位（含全零行）；真实权重与 FP4PTPC@97cc9b32 逐字节、精度 ≤1.5× 模型参考 | 合成权重、容差内 | `test_shared_fold_*` 等 4 个 | 通过 |
| shared 报错（非张量、fnuz NaN、2^31 只计 routed 列）与调试开关；双流阈值 M1024 / 1025 | 合法输入；M 远离阈值 | `test_shared_rejects` 等 2 个 | 通过 |
| shared 的 graph（双流 / 串行）、原地输出、TP8 补零逐位、`shared_expert_id` | eager、另给 out、容差内 | `test_layer_graph_replay` 等 4 个 | 通过 |

未覆盖 / 跳过：屏蔽路由（首列 −1，尚未支持）；端到端（暂缓）；真实权重用例缺数据目录时跳过；H4096 去重到 M64（要 E ≥ 342）；去重 Down 的
WG 内量化只靠端到端；I_pad > 1024 的 prefill；`STREAM_*` 诊断开关；多卡与 gfx950；fused_moe 的 torch.compile；aiter 其余 fused_moe 用例。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
shared 已按 ATOM split 接上、整层功能与精度达标；性能基线要等 PTL 恢复后冻结；下一步调研 shared 的融合方向并定计划。
- PTL 自整机 reset 起一直 Disabled（计时慢 30–40%）→ M0 的数字只作功能参考、不能冻结 → PTL 恢复后再冻结与下结论。
- M0 比 split 对照多一次相加与 routed 列拷贝 → 小 M 难做到“不慢于 split” → 融合时把 shared 并进 routed 的规约。
- FP4PTPC@main 的 MTP shared 已用非 2 的幂 scale 重量化（对块反量化差 2.5–2.8%）→ prepare 照收但更不准 → MTP 用块格式权重。
- 屏蔽路由（首列 −1）未支持，routed id 须在 [0, E) → padding 行会越界读权重 → 按计划补上，且不读被屏蔽路由的权重。
- fused_moe 按 Python 属性认权重：view / 拷贝 / 新 `nn.Parameter` 丢标记，torch.compile 追进 FlyDSL → 现只支持 eager 与 graph。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../../../moe_a8w4.py)（各入口的 docstring 写明接口、报错与数值约定）、[fused_moe.py](../../../../../fused_moe.py)（首行按标记转调）。
- kernel（任务划分与流水线见 [kernels.md](kernels.md)）：[gemm1_a8w4.py](gemm1_a8w4.py)、[gemm2_a8w4.py](gemm2_a8w4.py)、[decode_a8w4.py](decode_a8w4.py)、
  [decode_dedup_a8w4.py](decode_dedup_a8w4.py)（去重）、[stream_a8w4.py](stream_a8w4.py)（流式）、[quant_a8w4.py](quant_a8w4.py)、[common_a8w4.py](common_a8w4.py)（私有布局）；移植源：[pyhip M17 README](https://github.com/luocheng25/pyhip/blob/moe-ptpc-a8w4-r2-M17/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)。

<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · 功能版：shared expert（E385/K7 约定）接口、prepare… · 预期 1×
- M1 ⏳ 承诺 · 功能 · 功能：入口校验与屏蔽路由（topk_ids=-1 各档不读权重、整行屏蔽含 s… · 预期 1×

下一步（本提交的承诺）：M1 功能：入口校验与屏蔽路由（topk_ids=-1 各档不读权重、整行屏蔽含 shared、graph 重放）
<!-- /mf:roadmap -->
