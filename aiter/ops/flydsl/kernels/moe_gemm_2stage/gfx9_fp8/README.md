# gfx9_fp8 的 A8W4 MoE（FP8 按 token 激活 × MXFP4 routed + FP8 shared expert）：设计说明

<!-- mf:summary -->
> 由 mf 生成（11-27 17:39）· 目标：在 moe-a8w4-shared-M6（0ccb573229）上：①MoE 链路逐 kernel（含 kernel 间空隙）的 roofline 与差距分解，各项合计对上实测（误差≤10%），给出每个 kernel 的可达上限；②按结论优化：可改的 kernel 达到 4 TB/s 或 450 TFLOPS，做不到的达到可达上限；③显著减少 bench 的 kernel 序列数与 kernel 家族数，删去诊断开关与死代码；始终保持整层 73 例逐例 ≤ min(1.02×aiter 融合 FP8 PTPC, A8W4 split)、158 例逐例 ≤ 起点 M6 ×1.02（具体阈值与目标数随第一次计划确认）（1.05× 相对冻结基线）· 已完成 1 个里程碑（含本提交 M0） · 最佳累计 — · 下一步：M1 目标一：MoE 链路逐 kernel roofline 与差距分解（含 kernel 间空隙与侧流 fork/join），给出每个 kernel 的可达上限；盘点分派路径（survey.md）
<!-- /mf:summary -->
<!-- 设计文档标准（skill myflow-review）：读者是 tech lead，10 分钟读完，只靠本文能回答
     做什么 · 整体怎么做 · 为什么这样做 · 做到什么程度 · 风险与下一步。
     正文 ≤150 行；一行只说一件事（≤120 字，链接地址不计）；每节第一句是结论；只写当前设计（过程写在提交信息里）；
     接口细节、报错清单、数据格式留在代码 docstring，这里只给链接。这些注释读者看不到，也不计入篇幅。 -->
## 1. 目标与范围
<!-- ≤8 行：做什么；输入 / 输出、目标硬件与形状；验收（性能对照谁、精度参考谁）；不做什么。 -->
在 MI308X（gfx942）上提供 MXFP4 routed（可带 FP8 逐行 shared expert）的两阶段 FlyDSL MoE，主目标是 DeepSeek-V4.1-Flash 整层。
- 入口：`prepare_moe_a8w4_ptpc_weight`（加载时转私有布局，可带 shared）与 `flydsl_moe_a8w4_ptpc`（bf16 进出）；`aiter.fused_moe` 认标记转调。
- 整层形状：主干 / MTP（E384 / 128 + shared，topk 6 / 3 + 1）× TP4 / TP8（I576 / 288→320），H5120，swiglu_limit 10。
- routed-only 形状（原 85 例）：Qwen3.8-Flash-Next TP2 / TP4、MiMo-V2.6-Flash TP8 / TP4、DeepSeek-V4.1-Flash TP4。
- 性能验收：整层 73 例逐例 ≤ min(1.02× 融合 FP8 PTPC, A8W4 split)，原 158 例逐例 ≤ 起点 moe-a8w4-shared-M6 × 1.02；新增 22 例只查回退。
- 精度验收：对独立 FP32 参考 err ≤ 4.5%·max|ref|、calc_diff ≤ 1e-3；shared 真实权重的误差 ≤ 模型参考的 1.5 倍。
- shared 权重：收 FP4PTPC 的每行 F32 scale 或官方 32×32 E8M0 块（prepare 折叠，与 FP4PTPC@97cc9b32 逐字节同）；@main 的 MTP 有损，MTP 用块格式。
- 不做：同目录的 FP8 / BF16 底座；EP / expert_mask、bias、SiLU 以外的激活、预量化输入、gfx950、非 MXFP4 的 Qwen shared；经 fused_moe 时这些与指定 tile 报错。

## 2. 总体设计
<!-- ≤35 行：数据流图（≤12 行）；分档表；每个核心 kernel ≤8 行伪代码（tile 划分与流水线的骨架，不写寄存器与偏移）。 -->
权重经 LUT 解成 fp8 走 fp8 MFMA；激活按行量化（与参考逐位一致）；按 M 与每 expert 行数分四档；shared 在 M ≤ 4 并进 decode（主干 TP4 M4 除外），其余折进 routed 末次规约。
```text
加载时 prepare_moe_a8w4_ptpc_weight：mxfp4 → 私有布局（按 MFMA K 序排的 fp4、每 32-K 组 delta、每通道因子），I 补零到 64 的倍数
  [shared FP8：块 scale 先折叠成每行 2 的幂 → e4m3fn 转 fnuz（scale ×2）→ I 补零 → shuffle(16,16)，作 expert E]
aiter.fused_moe(x, w.w1, w.w2, …, Silu, per_1x32, per_Token)：首行见 w1 / w2 的标记 → _fused_moe_route（门禁、配对、查参数）→ ↓
flydsl_moe_a8w4_ptpc（bf16 x[M,H] → bf16 out[M,H]；无 host 同步，可被 CUDA graph 捕获）：
  decode：[行量化(x)] → Gate/Up decode（WG 管一条路由）→ mid f32 → [行量化(mid)] → Down decode（WG 管一个 token）→ out
    混合（带 shared、M ≤ 4，主干 TP4 M4 除外）：shared 作同两个 kernel 的末路（FP8 直读；Gate/Up 每通道块 2 个 WG，Down 末 wave），不相加
  去重 decode：计划 + 行量化(x) → Gate/Up 去重（组首 WG 算同 expert 的 2–3 条路由）→ 行量化(mid) → Down 去重原子加 / H5120 槽对 Down
  流式：prep（排序成 16 / 32 / 64 行的块 + 行量化(x) [+ 清零 out]）→ Gate/Up 流式 → [行量化(mid)] → Down 流式
    → bf16x2 原子加进 out，或 bf16 routes → route_sum → out
  prefill：[prep | Opus 排序 BM64 → 行量化(x，loc_ids 置 −1)] → Gate/Up prefill（gather xq 行）→ mid f32 → 行量化(mid)
    → Down prefill（× 通道因子 × ms × 路由权重，写 loc_ids）→ routes bf16 → sorted_sum（底座）→ out
  折叠（带 shared、M ≥ 5 与主干 TP4 M4，见分档表）：routed 跳过末列；shared GEMM(xq) → shared_mid（SiLU(L)·up、行量化、× 权重）→ GEMM
    → 写 out（槽对 / 原子 Down 加在其上）或 routes 末尾 M 行（route_sum / sorted_sum 多加一列）；M ≤ 1024 在侧流
```
| 输入范围 | 路径 | 为什么 |
|---|---|---|
| M ≤ 32（H5120 只到 M10），H ≤ 4096 或 5120，mid 行放得进 LDS | decode：不排序、不规约 | 排序块几乎全是填充；M=1 快 2.6–5.8 倍 |
| （decode）M·topk·I/16 ≥ 2560；M ≥ 4 且 w2 加一个 expert ≤ 4 GiB | 先量化 x；先量化 mid，Down 用槽对版 | 免得每个 WG 各量化整行；16 B 覆盖两槽 |
| （decode）H4096、I%128=0、M ≥ 2；路由 ≤ 1.5E 时到 M64 | 按 expert 去重；M·I < 3072 时两个 kernel | 同 expert 共读权重，比逐路由快 6–24% |
| （decode）H5120、I%128≠0、M8–10 | 只去重 Gate/Up，Down 用槽对版 | I576 非 128 倍数；M8 比逐路由快 1.5% |
| M > 32（H5120 > 10），每 expert 平均 ≤ 22 行（H ≥ 4096 为 48） | 流式：排序块 16（≤12 行）/ 32（≤28 行）/ 64 | 读权重为主；H5120 M11 起快 1–7% |
| （流式）BM ≥ 32、H < 4096 且累加器 ≤ 96 VGPR（H2560 BM64 到 56 行） | Gate/Up 一个 WG 管全部 I、尾部量化 | 省一次行量化；H4096 时轻 wave 更快 |
| （流式 Down）I < 512；BM64 且 H ≥ 4096 | 原子加进 prep 清零的 out；换 prefill Down | 免 route_sum（I512 反慢）；MiMo TP4 M1024 快 5% |
| （H5120）流式路由 ≥ 4096；prefill 路由 ≥ 16384 | 流式上限 24 行；按 XCD 分块 | 17–24 行流式快 3–8%，28 行起 prefill 快；XCD 快 1–4% |
| 其余 M；给了 tile 参数 | prefill BM64·BN128；行多（给参数时 M > 2048）TK128；BM·I_pad > 64 KiB 入口报错 | 多行共用权重；TK128 的 LDS 与寄存器减半 |
| （prefill）H ≥ 2560 且 M·topk ≤ 24576（H5120 为 12288） | 排序与 x 量化改用 prep（一次 launch） | TP2 M2048 快 2.1%；路由多则 prep 慢 |
| （prefill）I ≥ 256 且排序块少；I > 512 且 WG > 80 | Down 的 H 切给 2 / 4 个 WG；激活分两段 | 末轮少空闲（M2048 快 10%）；每 CU 2 WG |
| 带 shared：M ≤ 4、逐路由 decode 核内量化、≤ 16 列；可折叠时 M·topk·I ≤ 10240 | 混合：shared 作末路 | 省 split 链与相加；M1 为 split 的 0.27–0.39 |
| 带 shared、M ≥ 5 或 M·topk·I > 10240：decode 槽对 Down、流式、prefill 1025–65536 | 折叠进末次规约；否则 split | 省量化与相加；主干 TP4 M4 快 6% |
- 屏蔽（id −1，ATOM 的 padding 行）：decode 经越界 buffer 视图读 0（不访存）、Down 用 select 丢弃，去重计划与排序不收；首列 −1 或末列非 E 时 shared 也丢。
- 核心 kernel 骨架（完整的 Gluon 风格写法、任务划分与逐拍流水线见 [kernels.md](kernels.md)）：
```text
Gate/Up prefill：grid (2I/128, 排序块)，4 wave 管 TILE_M 行 × 64 gate + 64 up；权重每 lane 16 B 读两行、提前 2 tile，x 经 LDS 双缓冲
Down prefill：grid (切分 1/2/4, 排序块)，mq 经 LDS（I>512 分两段）装入后驻留 VGPR；逐 64 列块解码 → MFMA → bf16 写 routes 与 loc_ids
Gate/Up decode：grid (I/16, M·topk)，4 wave = gate / up × K 前后半，每 lane 16 B 读两个 64-K 块、先发 6 对；x 经 LDS 广播
Down decode：grid (H/32, M)，wave 按槽取 mid 行，每 lane 16 B 读两个 16 行块（槽对版 H/16、读两槽）→ MFMA → LDS 树形相加 → out
混合 Gate/Up / Down：shared 作末路，Gate/Up 拆 2 个 WG（各 8 个 gate 与同号 up 行）、Down 用末 wave；FP8 块直进 MFMA，行量化每行一次倒数
去重 Gate/Up：grid 同逐路由版，只有组首 WG 计算：组员的 x 行进 LDS（A 的行 m、8+m 取组员 m），本 expert 的权重只读一次
去重 Down：grid (H/128, M·topk)，组首 WG 读 8 个 16 通道块，各组员 × 通道因子 × 行 scale × 路由权重 → bf16x2 原子加进 out
流式 Gate/Up：grid (I/(16·G·W), 排序块)，wave 管 G 个 gate 块与同号 up 块；权重直读 VGPR：lane 0–7 / 8–15 各 16 B 读 gate / up 块相邻两行
  提前 P 个 64-K 步；链 p 取行对第 p 行，收尾 lane l↔l^32 换回；W > 1 时激活经 LDS 环共享（H5120 一次 16 B 跨 4 / 2 步）；尾部量化时 4 wave 管全部 I
流式 Down：grid (H/(16·R·W), 排序块)，wave 管 R/2 个 16 通道块对 × 全部 I；激活驻留 VGPR，每 lane 16 B 读块对的相邻两行
  权重提前 1–2 个块对（BM ≤ 32 时 1 对）→ 每子块 4 条 MFMA 链 → bf16 routes，或每 lane 一对通道 bf16x2 原子加
```

## 3. 关键决策
<!-- 3–6 条；“为什么”给一个关键数字。 -->
routed 输出与 M17 逐位相同（只改读取宽度、解码、量化算法与预取）；shared 在 M ≤ 4 多并进 decode，其余复用 xq、折进末次规约。
| 决策 | 备选 | 为什么 | 代价 / 限制 |
|---|---|---|---|
| fp4 按 MFMA K 序预排；每 lane 读 16 B | 解成 bf16；8 B 读 | 流式读指令 226→126；decode VGPR 153→87 | prefill Down 仍 8 B；delta>11 按 0 |
| 流式解码查 wave 自写的 LDS LUT，64 位移位拼行 | v_perm 现算 | 每行每步 VALU 20→15 | prefill Down 受 MFMA 限，用了反慢（LDS 超 32 KiB） |
| 行量化商逐位等于 IEEE 除法，每行一次倒数 + FMA 修正 | 乘倒数不修正；逐值除法 | 不翻转量化；逐值除法约占 decode wave 1/3 VALU | 极端行逐值除；去重 Down 仍逐值除 |
| M ≤ 32（H5120 ≤ 10）不排序；H4096 去重；行少时 16–64 行小块流式 | 全走 BM64；逐路由读 | M128 时 BM64 块内约 95% 是填充 | 四套 kernel；原子加不逐位确定 |
| fused_moe 按 w1 / w2 的标记转调，tile 自动 | 内部转换缓存；按 block_size_M 钉 tile | 内部转换要双份权重；钉 tile 时 DS M=1 慢 3.5 倍 | 标记是 Python 属性 |
| shared 作 expert E：M ≤ 4 进 decode，其余用 FlyDSL GEMM 折进规约 | split 后相加 | 省量化与相加；GEMM 快 26–40% | 末列；16 B 对齐；M ≤ 65536 |

## 4. 性能与精度
<!-- ≤15 行，写在生成块前面。 -->
整层原 73 例 ≤ 0.98× 融合 FP8 且 ≤ split（起点的复核数据）；新增交界 22 例（本代码 4 次中位数）中 MTP TP8 M5 与两个 M98304 未达标（见第 6 节）。
| 档（整层 4 形状） | ÷ 融合 FP8 | ÷ split | 本实现 TB/s（占 4 TB/s） | FP8 TB/s | 本实现 / FP8 TFLOPS（占 450） | 结论 |
|---|---|---|---|---|---|---|
| M1–4 混合（11 例） | 0.64–0.84 | 0.27–0.64 | 1.14–2.01（≤50%） | 1.39–2.47（≤62%） | 3.3–8.0 / 2.7–6.0 | 达标 |
| 主干 TP4 M4、M6 逐路由 decode | 0.66–0.98 | 0.79–0.86 | 1.15–2.16（≤54%） | 1.96–2.59（≤65%） | 5.2–9.1 / 5.1–6.3 | 达标 |
| M8 去重 decode（档为 M8–10） | 0.66–0.82 | 0.88–0.95 | 1.45–2.28（≤57%） | 2.14–2.74（≤69%） | 6.4–9.7 / 5.3–6.6 | 达标 |
| M16–1280 流式（含块 64） | 0.50–0.80 | 0.91–1.00 | 1.22–2.68（≤67%） | 1.68–2.97（≤74%） | 9.9–172 / 6.4–138 | 达标，M48–256 贴线 |
| M ≥ 2048 prefill | 0.77–0.94 | 0.90–0.97 | 0.18–1.35（≤34%） | 0.23–1.97（≤49%） | 164–318 / 126–288（≤71%） | 达标 |
| 交界 22 例 | 0.55–0.90，3 例 >1 | 0.74–0.99，2 例 >1 | 0.08–2.33（≤58%） | 0.12–2.81（≤70%） | 4.5–267 / 5.0–277（≤59%） | 3 例未达标 |
- aiter_fp8_ptpc 对照：同 worktree（aiter 基线 ab0223ec4e）的 `aiter.fused_moe` per_Token，主干 E385/K7、MTP E129/K4，TP8 的 I 补零到 320；
  无 H5120 调优行，都是 `impl__flydsl_gfx942__` 默认：M ≤ 256 为 BF16 激活的 16_16_16 / 16_128_64，≥ 512 为 64_128_128。
- a8w4_split 对照（ATOM 的 split 臂）：同 worktree 的 routed-only A8W4 + aiter CK `gemm_a8w8_bpreshuffle` 链，M ≤ 1024 双流，不计最后相加。
- 计时：整层例的四个实现同进程、CUDA graph 重放墙钟（计入双流并发）；原 85 例 `run_perftest`；都过 PTL 门禁。
- 精度参考：独立 FP32 参考（MXFP4 / FP8 精确反量化，x 与 mid 逐 token FP8）；整层最大误差 ≤ 3.9%·max|ref|（混合 M1–4 ≤ 0.3%）、calc_diff ≤ 1.1e-4。
- shared 真实权重（官方 layer 2 / 37 的 TP4 切片）：误差是模型参考（1×32 UE8M0 激活 + 32×32 块权重）的 0.69–0.99 倍。
<!-- mf:perf -->
（尚无复核数据）
<!-- /mf:perf -->

## 5. 测试覆盖
三类用例都在 [op_tests/test_flydsl_moe_a8w4.py](../../../../../../op_tests/test_flydsl_moe_a8w4.py)（参考、基本、其他、bench 四个注释区），分派断言跟随 moe_a8w4.py 的分派常量。

**基本功能测试**（每次改动先跑：`pytest op_tests/test_flydsl_moe_a8w4.py -m basic`，18 例，约 30–60 秒）
| 覆盖什么（每例断言误差与所走的档 / kernel；整层只断言 routed 档与 shared 已跑） | 用例 | 结果 |
|---|---|---|
| Qwen TP2 / TP4（E33，I160 补零到 192）：M3 逐路由 decode、M64 流式、M1031 / 257 prefill | `test_tp2` ×3、`test_tp4_padded` | 通过 |
| MiMo TP8 / TP4（E256）：M5 去重两个 kernel、M24 去重四个 kernel、M300 流式、M1600 prefill | `test_mimo` ×4 | 通过 |
| DeepSeek TP4 + swiglu_limit=10（截断大量触发）：M8 只去重 Gate/Up、M64 流式 | `test_deepseek_swiglu` ×2 | 通过 |
| `aiter.fused_moe`：Qwen M3、整层 E385/K7（`shared_expert_id`=E，M8）与直接调用同 kernel、逐位相同 | `test_fused_moe_entry` 等 2 例 | 通过 |
| 整层 + shared（全 E、截断）：主干 TP4 M1 / 64 / 2048、TP8 M4（补零）、MTP 块格式 M8；TP4 M4 屏蔽（折叠） | `test_layer` ×5 等 6 例 | 通过 |

**性能测试**（`python op_tests/test_flydsl_moe_a8w4.py`，原 85 + 整层 95 = 180 例；`--quick` 只跑整层 24 例；每例校验输出并打印[逐 kernel INFO](kernels.md)）
| 覆盖什么（分档 / 形状） | case | 结果 |
|---|---|---|
| decode：逐路由 Qwen M1–32、MiMo M1、DS M1–4；去重 MiMo M2–48、DS M8（30 例） | `tp2_m1` 等 | Qwen ÷ aiter 0.66–0.90；≤2.57 TB/s |
| 流式：Qwen M48–1024（TP4 到 2048）、MiMo M64–1024、DeepSeek M16–1024（31 例） | `tp2_m48` 等 | Qwen ÷ aiter 0.58–0.87；≤2.74 TB/s |
| prefill：M2048 由 prep 排序（4 例）、M ≥ 4096 由 Opus 排序（20 例） | `tp2_m2048` 等 | Qwen ÷ aiter 0.80–0.94；≤326 TFLOPS |
| 整层 + shared：4 形状 × 17 个 M、M6 与 MTP TP4 M1280（混合、折叠各档） | `ds_layer_tp4_m1` 等 73 例 | ÷FP8 ≤ 0.98、≤ split；≤2.68 TB/s |
| 分派交界（权重 0）：混合↔折叠 M3/5、decode↔流式 M10/11、流式末段、M98304 | `ds_layer_tp4_m3` 等 22 例 | ÷FP8 0.55–0.90，3 例 >1；≤2.33 TB/s |

**其他覆盖测试**（只测前两类覆盖不了的；`-m "not basic"`，240 例，约 6–7 分钟）
| 覆盖什么 | 为什么前两类覆盖不了 | 用例 | 结果 |
|---|---|---|---|
| 逐路由 decode 交界：M4、11/12、19/20、32/33（M33 流式）；topk 8/17 | 不在交界、不查 kernel | `test_decode_dispatch_boundaries` 等 2 个 | 通过 |
| decode 重复 expert 与 int64 路由、槽对 Down 的 w2 门槛两侧、graph、缓存键 | 输入规整、w2 小、eager、单卡 | `test_decode_graph_replay` 等 4 个 | 通过 |
| 去重交界：M·I=3072、M16/17、I384、topk16 预量化、E256 M48/49 | bench 无这些 M，不查 kernel | `test_decode_dedup_dispatch` 等 3 个 | 通过 |
| H5120 decode：上限 M10/11；钉 32 测 M7、M12 两轮 x、M32 去重、w2 回退 | bench 不查 kernel、w2 小 | `test_decode_wide_hidden` 等 8 例 | 通过 |
| 去重计划逐位（含 −1）；重复 expert（去重、流式）；去重 graph、缓存键 | 位错在判据内；输入规整、eager、单卡 | `test_decode_plan_quant_bitexact` 等 4 个 | 通过 |
| 流式 22/48/56 行上限、E513 grid 收缩、块 16；H5120 流式 24 行、prep、XCD、两段 | 多不在交界、不查 kernel | `test_stream_rows_limits` 等 5 个 | 通过 |
| prep / Opus 排序的路由数上限、重复 expert；流式接 prefill Down 切 1/2/4 | 不在交界；只一种切分 | `test_prefill_prep_routes_limit` 等 2 个 | 通过 |
| 逐位：布局、delta、行量化（含快速）、prep 排序、尾部量化、ACT_STEPS、流式 tile、loc −1、XCD | 碰不到；位错在判据内 | `test_row_quant_bitexact` 等 13 个 | 通过 |
| prefill 的 tile 参数、半块交界；非目标形状 7 种（含 LDS 上限）、Down 的 H%32 断言 | 只跑默认 tile 与目标形状 | `test_shape_fallbacks` 等 4 个 | 通过 |
| 流式与 prefill 的 graph、报错与入口校验（I_pad 1088、BM128、调试开关）、gfx942 门禁、缓存键 | 只 eager、合法输入、单卡 | `test_graph_replay` 等 6 个 | 通过 |
| Opus 排序重复 expert 留后一槽、原地输出、路由 dtype、M=0；各类 Gate/Up 的截断 | 输入规整、另给 out；bench 几乎不截断 | `test_inplace_output` 等 7 个 | 通过 |
| fused_moe 各档（逐位 / 对参考）、graph、参数透传与报错、未标记权重下传、标记生命周期 | 基本只测 decode、带标记的默认调用 | `test_fused_moe_matches_api` 等 8 个 | 通过 |
| shared 的 prepare（块 scale、fnuz、补零）逐位与报错；真实权重同 FP4PTPC 逐字节、误差 ≤1.5× 模型 | 合成权重、容差内 | `test_shared_fold_*` 等 5 个 | 通过 |
| split 自身（`SHARED_FOLD=0` 钉住）：双流阈值两侧、相加逐位、原地串行、TP8 补零逐位 | 整层例不查实现；容差内 | `test_layer_stream_mode_boundary` 等 3 个 | 通过 |
| 屏蔽各档（随机 / 整行 / 全部）：逐路由两种 Down、去重三种、流式两种 Down、prefill 两种排序 | 基本只测一种 decode；bench 不屏蔽 | `test_masked_routes` ×13 | 通过 |
| 整层：graph 重放（含换屏蔽）、原地与 eager 逐位同；屏蔽 TP8 流式原子加、TP4 prefill | 只 eager、另给 out、不屏蔽 | `test_layer_graph_replay` 等 4 个 | 通过 |
| 白盒：逐路由与混合 Gate/Up 不读屏蔽路由与丢弃 shared 的权重、计划不收它 | Down 丢弃它，端到端看不出 | `test_masked_gateup_reads_no_weights` 等 3 例 | 通过 |
| 混合 decode：M5 与 M·topk·I 交界、末列非 E、整批屏蔽、topk 1/15/16 + 1 | 不查谁算 shared | `test_mixed_decode_shapes` 等 4 个 | 通过 |
| 折叠 7 档（含 M5–7、流式接 prefill Down、prefill M1366）：屏蔽、末列非 E、原地、graph | 不屏蔽、末列恒 E、另给 out | `test_layer_fold_tiers` ×7 | 通过 |
| 折叠 GEMM：各 M 的 tile 不越 LDS stage；graph 重放 1000 次逐位同 eager | 竞争偶发，块错在容差内 | `test_layer_fold_gemm_race` | 通过 |
| 折叠交界：prefill 起点 M1025、M65536；out 未对齐退回 split；变体缓存 | bench 不在交界、out 对齐、单卡 | `test_layer_fold_max_m` 等 4 个 | 通过 |
| 逐 kernel 登记（各档与 shared 各跑法的条数对 profiler、运算量、字节、graph）与 bench 的事件对齐 | bench 只打印 | `test_launch_records` 等 25 例 | 通过 |

未覆盖 / 跳过：端到端精度（本任务不做）；真实权重缺数据目录时跳过；H4096 去重到 M64（要 E ≥ 342）；去重 Down 的 WG 内量化只靠端到端；快速除法只抽样；CK 排序的 −1；逐 kernel 字节只查合计下界；
`STREAM_*` / `SHARED_FOLD_*` 开关；多卡、gfx950；fused_moe 的 torch.compile 与其余用例；bench 无 H5120 M7、M9、prefill 折叠 M<2048。

## 6. 风险与待定
<!-- ≤5 条：风险 → 影响 → 打算。 -->
新增交界例中 3 例未达整层约束（权重 0、只查回退），修复放进第一次计划；原 73 例无阻塞，但单次 bench 有贴线。
- MTP TP8 M5 ÷FP8 1.11–1.15：混合 decode 止于 M4，M5 折叠多 x / mid 两次行量化，侧流 fork / join 留约 13 µs 空隙 → 第一次计划。
- M98304 两例 ÷FP8 1.04 / 1.16、÷split 1.04 / 1.11：超过折叠 GEMM 的 65536 行，shared 走 split 并最后相加 → 拟按 65536 行分块折叠。
- 贴线：ds_layer_tp4 M16 / 48 / 64、ds_layer_tp8 M64 / 512 / 1024 单次 ÷split 达 1.00–1.02，MTP TP8 M6 ÷FP8 0.98–1.02 → 按多次中位数判。
- 外部 kernel：preshuffle GEMM 偶发 LDS 块错（已限 tile 并断言）→ 达不到上限就重写；登记按 aiter 当前派发建模 → 派发变了由 mismatch 行与用例暴露。
- 测量与接口：reset 后 PTL 变 Disabled、别的容器占卡 → 门禁拒绝；fused_moe 认 Python 属性 → 只支持 eager 与 graph；登记开关非线程安全。

## 7. 进一步阅读
- 代码入口：[moe_a8w4.py](../../../moe_a8w4.py)（各入口的 docstring 写明接口、报错与数值约定，`recording_launches` 为逐 kernel 登记）、[fused_moe.py](../../../../../fused_moe.py)（首行按标记转调）。
- kernel（任务划分见 [kernels.md](kernels.md)）：[gemm1_a8w4.py](gemm1_a8w4.py)、[gemm2_a8w4.py](gemm2_a8w4.py)、[decode_a8w4.py](decode_a8w4.py)（含混合 decode）、[decode_dedup_a8w4.py](decode_dedup_a8w4.py)（去重）、
  [stream_a8w4.py](stream_a8w4.py)（流式）、[shared_a8w4.py](shared_a8w4.py)（折叠的 shared_mid）、[quant_a8w4.py](quant_a8w4.py)、[common_a8w4.py](common_a8w4.py)（私有布局）；移植源：[pyhip M17 README](https://github.com/luocheng25/pyhip/blob/moe-ptpc-a8w4-r2-M17/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md)。
<!-- mf:roadmap -->
- M0 ✅ 本提交 · 功能 · bench 扩展：整层 22 个交界 / M>65536 case 与逐 ke… · 预期 1×
- M1 ⏳ 承诺 · 分析 · 目标一：MoE 链路逐 kernel roofline 与差距分解（含 ker… · 预期 1×

下一步（本提交的承诺）：M1 目标一：MoE 链路逐 kernel roofline 与差距分解（含 kernel 间空隙与侧流 fork/join），给出每个 kernel 的可达上限；盘点分派路径（survey.md）
<!-- /mf:roadmap -->
