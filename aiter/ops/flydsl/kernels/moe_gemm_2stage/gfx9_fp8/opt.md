# gfx950 FP8 MoE（FlyDSL `fused_moe_gfx942` 后端）优化记录

本文件记录 `gfx9_fp8/` 下 MoE kernel 在 gfx950（MI350X）上的优化方法、当前状态和失败尝试。
**后续每次优化都在文末“变更记录”追加一条**，写明日期、改动、方法、A/B 数据和结论；失败的尝试也要记录。

## 1. 测试方法

- **硬件与环境**
  - 8× MI350X（gfx950，256 CU，8 XCD），ROCm 7.2，torch 2.9.1，FlyDSL 0.4.0（`/opt/venv`）。
  - 每个进程使用独立的 `AITER_JIT_DIR`，并设置 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=1`（该变量现在只影响 gfx942；gfx950 的 BF16 输出固定用硬件 RNE 转换）。
- **模型**：全部为 PTPC FP8（per-token 激活 × per-channel 权重）。

  | 名称 | D | I | E | topk | 激活 |
  |---|---:|---:|---:|---:|---|
  | MiniMax-e128 | 6144 | 384 | 128 | 4 | Swiglu |
  | MiniMax-e129 | 6144 | 384 | 129 | 5 | Swiglu |
  | MiMo | 6144 | 256 | 384 | 8 | SiLU |
  | Qwen3.5-i128 | 2048 | 128 | 257 | 9 | SiLU |
  | Qwen3.5-i512 | 2048 | 512 | 257 | 9 | SiLU |
  | Qwen397 | 4096 | 128 | 513 | 11 | SiLU |
  | Qwen3.8-i640 | 2560 | 640 | 513 | 11 | SiLU |

- **batch**：tokens 取 1024–32768。
- **计时**
  - 用 torch profiler 统计每次迭代所有 kernel 的时间之和（sort、输入 quant、Gate/Up、中间 quant、Down、reduce），取中位数。
  - 每轮 20 次迭代，3 轮交替执行。
- **两种口径**
  - b2b：kernel 连续发射。
  - sleep：每个 kernel 前先跑 1M cycle 的 `torch.cuda._sleep`，与 pyhip cudaPerf 的方法一致。
- **上限参考**
  - 带宽：读 5.2 TB/s，写（fill）4.65 TB/s，copy 4.57 TB/s。
  - FP8 dense：真实数据下约 2.0–2.5 PF。
  - Down 的理想字节数 = R·I + R·4 + E_act·D·I + E_act·D·4 + R·D·2，其中 R = tokens·topk。
- **注意事项**
  1. **持续负载会降频。** 例如 4w Gate/Up 在 MiMo 4k：GPU 刚休息过约 265–275 µs，稳态约 325–330 µs，需要约 150 µs 空闲才能恢复时钟。比较不同实现时迭代次数必须一致：pyhip 测试默认只测 sync 之后的 3 次迭代，相当于“刚休息过”，需设 `PYHIP_ITERS=20`。
  2. **FlyDSL 磁盘缓存（`~/.flydsl/cache`）可能返回旧二进制。** 怀疑性能回退时，先用 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 重新编译，对比输出是否逐位一致以及耗时。2026-10-23 的旧缓存已移到 `~/.flydsl/cache_stale_20261023`。
  3. benchmark 运行期间不要修改 kernel 源码（FlyDSL 会延迟编译，并用 getsource 生成缓存键）。
  4. 不同 GPU 并行跑时相差可达几个百分点；小于 5% 的效果要在同一进程内交替 A/B。
  5. **gfx950 上 rocprofv3 的 `FetchSize` 偏小**：它把 128B 读请求按 64B 计。读流量要用 `TCC_EA0_RDREQ_sum`、`TCC_EA0_RDREQ_128B_sum`、`TCC_EA0_RDREQ_32B_sum` 自己算（128·n128 + 64·(n − n128 − n32) + 32·n32）。

## 2. Pipeline 与配置

配置串格式为 `BLOCK_M_BLOCK_N_BLOCK_K_prefill:down_path:pad`。

| 阶段 | 实现 | 文件 | 配置 |
|---|---|---|---|
| sort | aiter `moe_sorting`（opus） | — | block：Down 为 M128 时取 256，否则等于 BLOCK_M（64 或 256） |
| 输入 quant | aiter per-token | — | — |
| Gate/Up M64 | prefill 1x4 | `gemm1.py` | `64_{256,128}_{128,256}_True`；使用 sort64 元数据，搭配 M128 Down 时读取 sort256 元数据 |
| Gate/Up 4 波 | 2×2 波，256×256 tile，persistent，stream-K | `gemm1_2x2.py` | `256_256_128_True`，使用 sort256 元数据 |
| 中间 quant | FlyDSL 短行 per-token（128 < I ≤ 1024）；其余用 aiter | `quant.py` | — |
| Down M64 1x4 | 64×256 tile，按 N 切成 n_splits 份 | `gemm2_1x4.py` | `:1x4_64x256:128` |
| Down 8x1 compact | M256 full + M64 tail 任务 | `gemm2_8x1.py`, `gemm2_8x1_compact.py` | `:8x1_compact:128` |
| Down M128 | 4×1 波，128 行块，packed 输出 | `gemm2_4x1.py` | `:m128:0`（one-shot），`:m128p:0`（persistent）；Gate/Up 可以是 M64 或 4 波 |
| topk reduce | `invert_sorted_ids` + `sorted_sum`（M64 Down）；`make_packed_sum`（M128 Down） | `moe_reduce.py`, `gemm2_4x1.py` | — |

- 4w Gate/Up 搭配 M64 Down 时，要用 block 64 重新排序一次，额外开销约 25–65 µs。
- M128 Down 需要 sort256 元数据；M64 和 4 波 Gate/Up 都能直接读取，不需要重新排序。

## 3. 已完成的优化

1. **gfx950 适配**
   - `common.py` 新增 `fp8_dtype()`，`get_down_device_config()` 支持 gfx950（xcd=8）。
2. **M64 Gate/Up（`gemm1.py`）**
   - task 按 XCD 连续映射（1D grid）。
   - 使用 `MFMA_Scale(16,16,128)` 的 K128 atom（不带 scale 的 f8f6f4）；仅 TILE_K=128 时使用 K128 交织调度。
   - `moe_sorting(accumulate=False)`。
   - MiniMax 32k 时 Gate/Up 从 1352 µs 降到 837 µs。
   - 8x1 Down 同样改为 XCD 连续映射。
3. **4 波 Gate/Up（`gemm1_2x2.py`）**
   - 移植自 pyhip `fp8_gemm_950` 的 4 波 FP8 GEMM。
   - 256 行 × (128 gate + 128 up) 的 tile。2×2 波，每个波负责 4 个象限各 64×64，累加器占 256 个 AGPR。
   - persistent：grid = CU 数，带 XCD slot 映射；A/B 通过 LDS DMA 做 ping-pong。
   - stream-K 只在余数 ≤ C/8 时启用：partial 用 SC1 读写，flag 用 agent 范围的 monotonic 原子。
   - 32k 时 Gate/Up 达到 1450–2054 TF（原来 1307–1520）。
4. **中间 quant（`quant.py` `flydsl_quant_per_token`）**
   - 结果与 aiter per-token kernel 逐位一致。
   - 约 5 TB/s；aiter 每个 256 线程的 block 只处理 1 行，只有 1.3–2.4 TB/s。
5. **M64 Down 按 N 切分（`gemm2_1x4.py`）**
   - 一个任务 = 64 行块 × N/n 列。
   - host 取 D/256 的最小因子 n，使每个 CU 至少有 6 个任务，用于缓解 wave quantization。
   - MiniMax-e128 4k 时 Down 从 182 µs 降到 145 µs。
6. **M128 Down（`gemm2_4x1.py`）**
   - 移植自 pyhip `moe_multistage_down_m128`，改为 PTPC。
   - 4 个波 × 32 行；B 通过 LDS DMA ring 预取（4 个槽，提前 3 个 packet），A 常驻寄存器。
   - 上一个 packet 的 epilogue（scale、BF16 转换、permlane 打包）穿插在当前 packet 的 MFMA 之间。
   - 输出为 packed 布局 `[block128, N64, row128, col64]`，再由 `make_packed_sum` 做 topk 求和。
   - `merge_halves`：每个 expert 平均不足 128 条路由时，一个任务处理 sort256 块的两半，后半段只在包含路由时才算。
   - `m128p`：512 个 CTA，8 个分片原子队列，最后一个领取者把计数器清零。
   - I ≤ 256 时约 4.0–4.57 TB/s。
7. **M128 Down 支持 I ≤ 640（2026-10-23）**
   - K>256 时，每个 N64 packet 拆成 N64×K128 子包流过 ring：K=384/512/640 分别 9/8/5 个 8 KiB 槽（≤72 KiB，每 CU 仍能放 2 个 CTA）。
   - B 固定 32 个 VGPR，A 占 K/8 个；VGPR 总数分别为 194/201/238，无溢出。
   - OC 切分在 host 端决定（`fused_moe_gfx942._m128_oc_splits`）：每个切片宽度 ≥ K（因为每个切片都要重读 A），在此前提下取能保证每个 worker ≥ 8 个任务的最少份数。
   - Down 结果（µs；括号内为理想 TB/s）：

     | 模型 | 32k b2b：原最优 M64 → M128 | 32k sleep：原最优 M64 → M128 |
     |---|---|---|
     | MiniMax-e128 | 706 → 565（2.79 → 3.48） | 676 → 501（2.91 → 3.93） |
     | MiniMax-e129 | 880 → 669（2.71 → 3.57） | 856 → 597（2.78 → 4.00） |
     | Qwen3.5-i512 | 618 → 501（2.64 → 3.26） | 581 → 449（2.81 → 3.63） |
     | Qwen3.8-i640 | 1111 → 904（2.63 → 3.23） | 1011 → 864（2.89 → 3.38） |

   - 整体 MoE：32k 时 b2b 快 1.08–1.11×、sleep 快 1.10–1.19×；16k 时 1.04–1.11× / 1.05–1.16×；8k 及以下基本持平。
8. **M64 Gate/Up 读取 sort256 元数据，小 batch 也能用 M128 Down（2026-10-23）**
   - 新配置 `64_{256,128}_{128,256}_True:m128:0` / `:m128p:0`：sort 用 256 行块，M64 Gate/Up 把每个块当作 4 个 64 行 tile，expert 取 `sorted_expert_ids[tile / 4]`。sort 把 expert 的路由排在 padding 前面，所以 tile 的第一个 sorted id 是 padding 时整个 tile 为空，直接退出。
   - **派发顺序决定性能。** 按 expert 顺序把一个块的 4 个 tile 相邻派发时，小 batch 下大部分 tile 为空，空 workgroup 夹在有效 workgroup 之间派发，Gate/Up 慢 17%（MiMo 1k：252 → 294 µs）。让空 workgroup 不读 sorted_ids 也一样慢，开销来自派发本身。
   - **改为 lead tile 顺序**：每个 XCD 负责连续的一段块，先派发每个块的前 `lead` 个 tile（同一块的 tile 仍然相邻，共享 L2 中的权重），再派发其余 tile（小 batch 时基本为空，放到最后）。
   - `lead` 在 kernel 内计算：均值 μ = 路由数 / min(块数, E)，超过 57 / 118 / 180 时 lead 分别取 2 / 3 / 4（即 Poisson 计数的 80% 分位达到 k 个 tile 时的均值）。除以 min(块数, E)，是因为路由数刚超过 256 的 expert 会多出一个几乎为空的块，把均值拉低。
   - `lead` 偏小时，有效 tile 落到后段，要从 HBM 再读一遍权重（MiMo 4k：lead=1 为 548 µs，lead=2 为 321 µs）；偏大时，空 tile 又夹在有效 tile 之间。
   - Gate/Up 与 sort64 相比（b2b，MiniMax-e128、MiMo、Qwen397）：1k–4k 在 −5%～+5% 之间，8k 慢 3–9%。收益来自 Down + reduce（例如 MiMo 1k：1x4 Down 204 µs + sorted_sum 29 µs → M128 174 µs + packed_sum 24 µs）。
   - 整体 MoE（同一次运行里，加入新配置前后各自的最优配置；7 个模型）：

     | 口径 | 1k | 2k | 4k | 8k | 提升 >1% 的点 |
     |---|---:|---:|---:|---:|---:|
     | b2b（几何平均） | 1.037× | 1.042× | 1.044× | 1.007× | 19/28 |
     | sleep（几何平均） | 1.037× | 1.033× | 1.024× | 1.003× | 12/28 |

   - 提升最多的是 I 小、expert 多的模型：Qwen3.5-i128 在 1k–4k 快 1.07–1.14×，Qwen397 在 1k–8k 快 1.02–1.12×，MiMo b2b 在 1k–4k 快 1.04–1.06×。Qwen3.8-i640（1k–2k）和 Qwen3.5-i512（1k）的 M128 Down 不比 1x4 Down 快，最优配置不变。
9. **M128 Down one-shot 的任务顺序（2026-10-23）**
   - CTA w 固定跑在 XCD w % 8。任务表按宽度 W 转置后，每个 XCD 只走 1/W 段，一个 128 行块的各个 OC 切片分到 8/W 个 XCD 上，每个 XCD 都要读一遍这个块的 A（128×K）。
   - one-shot 原来 W=2：一个块的 A 被 4 个 XCD 各读一次。改为 N ≤ 2560 且 OC ≥ 4 时 W=8（一个块的所有切片都在同一个 XCD，A 只读一次），其余 W=4。persistent 仍是 W=4。
   - 判据：块的 A 至少占该 expert 权重（N×K）的 5% 时，A 的重复读取才比权重局部性更重要。
   - 例：Qwen3.5-i512 4k 的 Down 读流量 0.57 → 0.30 GB（理想约 0.29 GB）。
   - Down（µs，同进程 A/B，输出逐位一致）：

     | 模型 | 变化 |
     |---|---|
     | Qwen3.5-i512 | 2k 96.8 → 93.8；4k 136.1 → 120.3；8k 222.1 → 178.6；16k 321.5 → 287.5 |
     | Qwen3.8-i640 | 16k 597.9 → 521.3；32k 880.9 → 860.0 |
     | Qwen3.5-i128 | 4k 56.0 → 53.5；8k 109.2 → 94.2；16k 191.7 → 178.1 |
     | Qwen397 | 8k 239.0 → 228.7 |
     | MiniMax-e128 | 16k 338.2 → 320.7；8k 199.4 → 200.9（最大回退 +0.7%） |

   - 其余 one-shot 生产配置在 ±1% 内。

## 4. 当前状态（2026-10-23）

- **对比 pyhip（双方都是 20 次迭代；1k–8k 为加入 M64 Gate/Up + M128 Down 后的结果，16k/32k 沿用之前的结果）**

  | 口径 | 1k | 2k | 4k | 8k | 16k | 32k | 胜出点数 | 范围 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | b2b（几何平均） | 1.41× | 1.29× | 1.20× | 1.19× | 1.20× | 1.24× | 42/42 | 1.14–1.55× |
  | sleep（几何平均） | 1.20× | 1.15× | 1.14× | 1.19× | 1.24× | 1.33× | 42/42 | 1.06–1.44× |

- **对比 aiter 自带的 PTPC MoE kernel**（ASM 1-stage、ASM stage1 + CK stage2、CK 2-stage；在 I>256 的 M128 Down 加入之前测得）
  - 两种口径都是 39/42 胜出；几何平均 b2b 1.46×，sleep 1.53×。
  - 输掉的 3 个点都在 1k，对手都是 ASM 1-stage：MiMo 0.90×、Qwen397 0.90×、Qwen3.5-i128 0.95×。
  - 用这些 aiter 数据估算加入 M64 Gate/Up + M128 Down 后的 1k：b2b MiMo 0.95×、Qwen397 0.97×、Qwen3.5-i128 1.03×；sleep MiMo 0.98×、Qwen397 1.00×、Qwen3.5-i128 1.06×。
- **剩余开销**
  - 32k 时 topk reduce 占总时间 22–33%。
  - I>256 时 Down 在 b2b 下仍只有写带宽上限的 69–77%。
  - 1k–2k 时，M64 Gate/Up 读取 sort256 元数据仍比 sort64 慢 0–5%，空 tile 仍然要派发。
- **gfx950 调优配置**（2026-10-23）：6 个 model_configs CSV 各补上 gfx950（cu_num 256）行，共 202 行，覆盖 MiniMax-M3、Qwen3.5-35B、Qwen3.5-397B、Qwen3.8 FP8/BF16 和 Hunyuan3 per-tensor。
  - 选中的 kernel：本后端 156 行，ASM 1-stage 17 行，其余 29 行是 aiter 的 CK 等 2-stage kernel（Qwen3.8 BF16 27 行、Qwen3.5-35B 2 行）。
  - 端到端对比 aiter 2-stage 调优的最优配置（只统计被替换的行，同一种 e2e 测法）：MiniMax 1.98×（36/36 行）、Qwen397 1.89×（17/18）、Qwen3.5-35B 1.61×（25/34）、Qwen3.8 FP8 1.53×（39/48）、Qwen3.8 BF16 1.30×（21/48）。Hunyuan3 per-tensor 在 gfx950 上的 staged 调优对所有 shape 都报 no valid candidate（每批 60 个 CK 任务里 30 个 crash 或报错），18 行全用本后端。
  - 全部 202 行都用生产 `fused_moe` 跑过精度检查（`test_moe_2stage.test_fmoe`）。logits_diff 最大 1.2e-3；MiniMax 的 decode 行最大 8.8e-3，原因是 decode kernel 直接读 BF16 激活，而参考结果对激活做了 FP8 量化。
  - 改了 M128 Down one-shot 的任务顺序后，有 5 行改用 one-shot M128（见变更记录）。

## 5. 失败或放弃的尝试（不要重复）

- **1x4_64x256 Down 改用 K128 MFMA**：VGPR 从 242 增加到 264，occupancy 从 2 降到 1，变慢。
- **TILE_K=256 时设 `waves_per_eu=3`**：寄存器溢出。
- **4w Gate/Up**
  - 让各 CTA 错开 K 的起点：更慢，因为同步共享数据的 CTA 原本可以合并 miss。
  - half tile 配静态调度：更慢。
  - 调整 group_m：没有影响。
  - stream-K 用于大余数：余数 189 时慢 9–17%，因为 slot 区间漂移后失去了同步共享。
  - 通过 scf.if 传递 256 个累加器：386 个 VGPR 溢出。
  - 使用 agent 范围的 fence（buffer_wbl2/inv）：多出约 36 µs。
  - 跳过空下半块的 MFMA：结果逐位一致，但产生 116–322 个溢出；b2b 只快 2–5%，sleep 反而慢 10–13%。
- **M128 Down**
  - 两半都有路由时仍合并处理：权重切片被逐出，需要从 HBM 重读。
  - pyhip 的 PF4 pair-refill：没有收益。
  - one-shot 循环在 barrier 之后重读 LDS 里的 task slot：GPU 挂死，rank 必须保存在寄存器里。
  - K>256 时仍用 OC=8：A 被重读 8 次；Qwen3.5-i512 32k 时 OC=8 为 613 µs，OC=2 为 501 µs。
- **M128 Down 的任务顺序与权重复用（2026-10-23）**
  - 去掉 packet 起点轮转（`rotation = worker // 8`）：读流量降低约 20%，但 32k 慢 3–14%。
  - OC 切得更细（切片 128–384 列），让每个 XCD 在途的权重切片能放进 L2：慢 3–29%。
  - XCD 内按切片优先遍历（同一切片连续派发不同块）：MiniMax 4k 慢 25%。
  - 按 sort 块轮流分给各 XCD（同一 expert 的块分散到多个 XCD）：16k–32k 慢 15–32%。
  - 每个 XCD 取连续一段块（等价于 W=8）对 MiniMax（N=6144）慢 2–9%，对 32k（OC=2）慢 4–9%；所以 W=8 只用在 N ≤ 2560 且 OC ≥ 4。
  - persistent 用 W=8：与 W=4 持平（±1.5%），保持 W=4。
  - 读流量与耗时不成正比：例如 Qwen3.8 32k 时 W=8 的读流量比 W=2 多 13%，而 Qwen3.5-i512 4k 少 47%。
- **M64 Gate/Up 读取 sort256 元数据**
  - 按 expert 顺序把每块的 4 个 tile 相邻派发：小 batch 时空 tile 夹在有效 tile 之间，Gate/Up 慢 17–18%（MiMo 1k 252 → 294 µs，2k 266 → 314 µs）。空 workgroup 跳过 sorted_ids 读取也一样慢。
  - 所有块的 tile 0 先发、其余全部放后段（相当于 lead=1）：4k 时每块有 2 个有效 tile，第二个 tile 要从 HBM 重读权重，MiMo 4k Gate/Up 从 321 µs 涨到 548 µs。
  - lead 按 Poisson 95% 分位选：均值 171–176（MiMo / Qwen397 8k）时取 4，比取 3 慢 5–11%（479 vs 455 µs，263 vs 237 µs）。改用 80% 分位。
  - 均值只用 路由数 / 块数：MiniMax-e128 8k 每个 expert 约 256 条路由，一半 expert 多出一个几乎为空的块，均值降到约 171，lead 取 3，满块的第 4 个 tile 落到后段（强制 lead=3 实测 Gate/Up 390 µs，lead=4 为 306 µs）。改为除以 min(块数, E)。

## 6. 待做（按预期收益排序）

1. **把 topk reduce 融合进 Down**（原子累加，或按 token 输出的 epilogue）。32k 时占 22–33%，也是 1k 落后于 ASM 1-stage 的主要原因。
2. **Gate/Up 大余数尾波**（MiniMax-e128 余数约 198，e129 余数 37）：尾块改用更小的 tile（128×512 / 128×256），在单独的同步阶段执行。预计 4–6%。
3. **K>256 的 M128 Down 减少权重重读**：16k–32k 时 Down 读流量是理想值的 2–3.5 倍（Qwen3.8 32k：3.6 GB 对 1.07 GB），多数 128 行块要从 MALL 重读自己的权重切片，因为 L2 放不下每个 XCD 在途的切片。调整任务顺序和 OC 切分都没有效果（见 §5），需要让一个 CTA 连续处理同一 expert 的多个块并复用 LDS 里的 B。
4. **M64 Gate/Up 的空 tile**：小 batch（lead=1）时仍要在最后派发 3×块数×N 块数 个空 workgroup，Gate/Up 仍比 sort64 慢 0–5%。可以让每个块的尾部 tile 由一个 workgroup 循环检查，减少空 workgroup。
5. **I=192/320 的 N 尾部浪费；K>640 的 M128 Down**（A 会超过 80 个 VGPR，需要换一种切分方式）。

## 7. 工具（仓库外，`/tmp/e2e_ab_gfx950`）

| 工具 | 用途 |
|---|---|
| `bench_moe.py` | 单配置整体 MoE 的分 kernel profile 和精度检查；`SLEEP_BEFORE_KERNEL=1` 切换为 sleep 口径 |
| `run_e2e3.sh` + `summarize_e2e.py` | 对比 pyhip；`bench_pyhip.py` 设 `PYHIP_ITERS=20` |
| `run_m128k.sh`、`summarize_m128k.py`、`summarize_down_m128k.py`、`bench_m128.py` | Down 对比和 OC 扫描 |
| `run_m64m128.sh` + `summarize_m64m128.py` | 1k–8k 时加入 M64 Gate/Up + M128 Down 前后的最优配置对比；`test_m64m128.py` 只跑这组正确性用例 |
| `tune950.sh`、`val950.sh` + `validate950.py`、`summarize950.py` | gfx950 配置调优（先跑 aiter kernel 的 staged 调优，再跑 `--e2e_tune`）和逐行的生产路径精度检查；输入在 `csv/`，输出在 `out950/` |
| `ptpc/` | 对比 aiter PTPC kernel：`tune.sh`、`tune_family.sh`、`make_fam.py`、`bench_vs_aiter.py`、`run_vs.sh`、`summarize_vs.py` |
| `gap_gu4w.py`、`loop_gu4w.py` | 降频测量 |
| `compile_var.py` | 缓存二进制与重新编译结果的对比 |
| `ab_bf16_e2e.py` | 同进程 A/B：替换各 kernel 模块里的 `_f32_to_bf16`，清掉 `_get_compiled_kernel`、`_get_gateup_4w_kernel`、`build_moe_gateup_4w` 的缓存后交替测量；`AB_ROUNDS` 设轮数 |
| `ab_m128_final.py` + `gemm2_4x1_old.py` | M128 Down 同进程 A/B：HEAD 版本（改成绝对 import 的副本）对比工作区版本，并检查输出逐位一致 |
| `pmc_m128.py` | 单个 arm 的 M128 Down 读请求计数（按 128B/64B/32B 算读流量） |
| `balance_m128.py` | 用真实 sort 结果离线统计各 XCD 的有效任务数 |
| `retune950.sh` | 改 kernel 后只重跑 `--e2e_tune`，输出在 `out950r/` |

gfx950 正确性测试：`op_tests/test_flydsl_moe_gfx942.py`（`FOUR_WAVE_CASES` 24 个 + `M64_M128_CASES` 10 个）。

## 变更记录（只追加）

- 2026-10-22：gfx950 适配；M64 Gate/Up 和 8x1 Down 改为 XCD 连续映射，Gate/Up 改用 K128 MFMA atom。
- 2026-10-23：
  - 4w Gate/Up（含 stream-K）。
  - M128 Down（I ≤ 256，`m128`/`m128p`，merge_halves）。
  - FlyDSL 短行 per-token quant；1x4 Down 按 N 切分。
  - 对比 aiter PTPC kernel：39/42 胜出。
- 2026-10-23：
  - M128 Down 支持 I ≤ 640（K128 子包），加入 OC 切分策略；op test 增加到 24 个用例。
  - 修正 pyhip 的 b2b 测量口径（20 次迭代）。重测结果：b2b 42/42（几何平均 1.22×），sleep 42/42（几何平均 1.18×）。
  - 4w Gate/Up 跳过空半块的尝试已回退。
  - 文件改名：`gemm1_4w.py` → `gemm1_2x2.py`，`gemm2_m128.py` → `gemm2_4x1.py`。
- 2026-10-23：
  - M64 Gate/Up 读取 sort256 元数据，新增 `64_*_True:m128:0` / `:m128p:0` 配置（tune space 增加 6 个）；Gate/Up 按 lead tile 顺序派发。
  - 1k–8k 整体 MoE：b2b 几何平均快 0.7–4.4%，sleep 快 0.3–3.7%；最多 Qwen3.5-i128 2k 快 13.5%。对比 pyhip 的几何平均：b2b 1.223× → 1.253×，sleep 1.184× → 1.207×。
  - op test 增加 `M64_M128_CASES`（10 个），共 34 个用例通过。
- 2026-10-23：
  - 6 个 model_configs CSV 补上 gfx950 调优行（202 行），流程同 PR #5932：`TUNE_ONLY=flydslv2,cktile,flydsl,asm`、`AITER_FMOE_TUNE_EXCLUDE_KERNELS=fmoe_stage1`、`AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=1`。
  - 逐行生产路径精度检查全部通过；按 merged 配置跑 `test_moe_2stage.py --no-legacy --csv-filter impl__flydsl_gfx942` 156/156 通过；shape 冲突检查（`test_config_shape_collision`）通过。
- 2026-10-23：
  - gfx950 上 `_f32_to_bf16` 直接用 `.to(fx.BFloat16)`，生成硬件指令 `v_cvt_pk_bf16_f32`（RNE）。`AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED`、`AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED` 和 RTE 内联汇编只用于 gfx942。
  - ISA：所有调用该函数的 kernel 都改用 `v_cvt_pk_bf16_f32`。4w Gate/Up 的 spill 从 4 个增加到 6 个，新增的 reload 在 K 循环外（每个 tile 一次）。
  - 同进程 A/B（硬件 RNE 对比原 RTA，22 个生产配置，覆盖 decode、M64、4w、1x4、8x1 compact、BF16 default）：Gate/Up 和 Down 都在 ±1.5% 以内，与未改动的 M128 Down 的波动相当；MiniMax-e128 16k 的 4w Gate/Up 快 1.1%（8 轮）。结论：性能持平。
  - 与 RTA 的结果只在舍入平局处相差 1 ulp。202 个调优行重新逐行检查全部通过；op test 34 个用例通过。
- 2026-10-23：
  - M128 Down one-shot 的任务转置宽度从 2 改为：N ≤ 2560 且 OC ≥ 4 时取 8，其余取 4；persistent 不变（见 §3.9）。输出逐位不变。
  - Down：Qwen3.5-i512 2k–16k 快 3–20%，Qwen3.8-i640 16k 快 13%，Qwen3.5-i128 4k–16k 快 4–14%，Qwen397 8k 快 4%，MiniMax-e128 16k 快 5%；其余 one-shot 生产配置在 ±1% 内。
  - 试过但放弃：去掉 packet 轮转、更细的 OC 切分、切片优先顺序、按 sort 块轮流分配 XCD、persistent 用 W=8（见 §5）。K>256 的权重重读仍未解决，§6 第 3 项改写为新的方向。
  - 重跑 4 个 FP8 PTPC 模型的 `--e2e_tune`，只替换“改用 one-shot M128 且同一次运行内快 ≥ 1.5%”的 5 行（原地替换，不增加 shape）：Qwen3.5-i512 4k/8k/16k（m128p → m128，快 1.8–3.4%）、Qwen3.8-i640 16k（m128p → m128，快 1.9%）、MiniMax-e128 4k（`256_256_128_True:m128p:0` → `64_256_128_True:m128:0`；bench_moe 5 轮交替 458 → 432 µs）。
  - 5 行都用生产 `fused_moe` 做了精度检查（logits_diff ≤ 4.5e-6）；shape 冲突检查通过；op test 34 个用例通过。
- 2026-10-24：按 reduce、Down 权重复用、Gate/Up 尾部、空 tile、融合 quant 的顺序做最小实验；保留 reduce 分块，其余原型已回退，未改调优 CSV。
  - 测法：GPU 1 空闲时同进程交替 A/B，4 轮，每轮 warmup 5 次、profile 20 次，3 份输入；以下均为 b2b 的中位数，单位 µs。未重跑全量调优、sleep 口径或 pyhip 对比。
  - **保留 packed reduce 分块**：`N % 2048 == 512` 时采用 64 线程、512 列/CTA，其余维持 256 线程、2048 列。先试过 128 线程、1024 列（Qwen3.8-i640 32k reduce 452.95 → 420.39），最终采用无尾列浪费的 512 列版本：32k reduce 452.89 → 419.11，整体 3009.03 → 2978.15；1k reduce 15.17 → 13.72，整体 654.72 → 653.98。输出逐位一致，求和顺序未变。
  - **Down 双行组共享 B packet**：8 波 CTA 同时处理两个 M128 行组，只由前 4 波加载 B，保持原 packed 输出布局。K384 32k 输出逐位一致，但 MiniMax-e128 persistent Down 587.91 → 583.50、e129 persistent 710.07 → 703.85，均不足 1%；e128 one-shot 576.04 → 589.61。没有稳定净收益，回退；不能据此断言所有跨块 B 复用方案都无效。
  - **Gate/Up N64 尾列独立发射**：I192 32k，完整列仍用 4w，尾列复用现有 M64/M128/M256 kernel，精度检查通过。M64 尾列：Gate/Up 501.10 → 488.24，整体 1823.82 → 1822.25；M128 尾列：500.92 → 481.63，整体 1827.76 → 1806.91；M256 尾列：493.51，整体 1815.42。净收益最多约 1.1%，本轮不接入。
  - **Gate/Up 大余数独立 M128 尾波**：保留短余数 stream-K，大余数时主核只跑完整轮次，剩余 tile 拆成两个 M128。MiniMax-e128 32k 精度通过，但 Gate/Up 686.74 → 723.08，整体 1899.73 → 1933.96，回退。
  - **M64 在一个 CTA 内顺序检查 sort256 的 4 个 tile**：MiMo 1k 输出逐位一致，Gate/Up 259.48 → 258.31，整体 492.03 → 489.70；约 0.5% 收益不足以引入新派发路径，回退。
  - **I128 Gate/Up 融合输出 quant**：先保留 BF16 舍入，再跨波归约 absmax、写 FP8 与 scale；实验双方关闭 stream-K。Qwen3.5-i128 8k 输出逐位一致，省去 quant 约 7.38，但 Gate/Up 84.50 → 94.88，整体 302.58 → 304.88，回退。跨波同步和 epilogue 成本仍需降低。
  - **FP32 标量原子融合 reduce**：Down 将每条 route 舍入到 BF16 后转回 FP32，按 token 做 `llvm.atomicrmw fadd`；包含 FP32 输出清零与 BF16 copy，跳过 inverse/packed sum。Qwen3.8-i640 32k 本次输出逐位一致，但 Down 877.58 → 29215.79，整体 2990.86 → 30739.29，回退。这只说明当前标量原子实现不可用；未验证原生 buffer/packed atomic 的可行性，也不保证原子顺序在其他输入上逐位一致。
  - **验证**：现有 34 个 gfx950 MoE 用例通过；现有测试文件新增 5 个 packed reduce 用例，覆盖 N=512/2048/2560/4608/6144、TOPK=1/4/9/11、负数与越界 inverse、输出 canary，均精确通过。原型回退后再次检查 K640 Down 与 packed reduce。
  - **修正分析口径**：Qwen3.5-i512 4k 的 A 仅 0.0189 GB，四个 XCD 各读一次变为一次最多解释 0.0566 GB，而此前实测总读量下降约 0.2681 GB，不能全部归因于 A 重读。TCC 请求量不能单独确定 MALL/HBM 命中来源；理想读写字节数除以时间，也不能直接与纯写带宽上限相比作为利用率。
- 2026-10-24：借鉴仓库已有实现，依次验证连续输出、原生 packed atomic、重排 epilogue 融合量化、排序直接生成 inverse；新增保留最后一项，前三项原型回退。
  - **连续 `[token, topk, D]` 输出**：Down 改为 64-bit 地址的向量 scatter store，复用 `kernels/moe_reduce.py` 的 BF16 连续归约，跳过 inverse。Qwen3.8-i640 32k 输出逐位一致，整体 2996.85 → 3112.61 µs；将连续归约也改为 64 线程/512 列后，2999.74 → 3121.42 µs。Down 写出和连续归约都未胜过当前 packed 路径，回退。
  - **gfx950 原生 BF16 packed atomic**：使用 `BufferAtomicPkAdd(BFloat16)`，每次累加两个 BF16，计入输出清零且无末尾 copy。Qwen3.8-i640 32k 整体 2992.57 → 8148.15 µs，Down 约 885.55 → 6762.84 µs；1k 整体 653.29 → 673.12 µs。32k 与基线相对 RMS 差 0.00452，88 个元素不满足 atol=rtol=0.02；1k 也有 5 个超限元素。虽比标量 FP32 atomic 原型快，仍无净收益且舍入语义不同，回退。
  - **LDS 行重排后融合 I128 quant**：每次将 64 行 BF16 结果放入行距 136 元素的 LDS 暂存，16-lane 子组处理整行 I128，避免逐段跨波 absmax。双方关闭 stream-K。Qwen3.5-i128 8k 输出逐位一致，整体 304.47 → 303.71 µs，基本持平；32k 严格比较有 1/67108864 个输出相差 0.00390625（一个 BF16 ULP），未继续计时或放宽门槛，回退。
  - **保留 OPUS sorting inverse 复用**：M128 的 `moe_sorting` 调用使用已有 `output_aux="opus"`，将返回的 `reverse_sorted` 直接作为 `make_packed_sum` 的 loc_ids，不再发射 `invert_sorted_ids`。没有新增 C++ 接口；目前也会生成未消费的 m_indices。OPUS 辅助输出保留调用者的 dispatch_policy，不再将显式单阶段/多阶段重置为自动。无效 route 在排序中设为 -1，packed sum 按零处理。
  - inverse 对比采用原始顺序的基线（独立 inverse 在 Down 后执行），GPU 1 同进程 6 轮交替，每轮 warmup 5、profile 20 次、3 份输入，以下均为 µs：

    | 模型 / tokens | sort+inverse 旧 → 新 | 整体 MoE 旧 → 新 |
    |---|---|---|
    | Qwen3.8-i640 / 1024 | 23.33 → 18.49 | 652.00 → 649.19 |
    | Qwen3.8-i640 / 32768 | 83.25 → 80.26 | 2988.96 → 2986.64 |
    | MiniMax-e128 / 1024 | 17.77 → 13.63 | 267.12 → 264.52 |
    | MiniMax-e128 / 32768 | 69.59 → 66.94 | 1899.20 → 1898.21 |
    | Qwen3.5-i128 / 8192 | 35.35 → 31.11 | 301.41 → 299.43 |

  - 五个 shape 的整体输出逐位一致；收益是稳定省掉约 2.6–4.8 µs 的排序/inverse 开销，端到端不足 1%，不视为 GEMM 加速。没有测试 sleep 口径或重新调优 CSV。
  - **验证**：34 个既有 gfx950 MoE + 5 个 packed reduce + 24 个 sorting inverse 用例通过；inverse 用例覆盖 token=1/65、重复 expert、expert mask、num_local_tokens 尾部、全空路由、dispatch_policy=0/1/2。随后为显式策略增加 mock 参数断言，三种策略的 masked 用例再次通过。
- 2026-10-24：按用户要求撤回 `aiter/fused_moe.py` 的公共 dispatch_policy 修改，改为在本后端统一复用 OPUS inverse。
  - 所有 prefill reduce 路径使用对应排序返回的 `reverse_sorted`，后端不再调用独立 `invert_sorted_ids`。需要 4w Gate/Up → M64 Down 二次排序时，首次 sort256 不生成辅助数组，使用第二次 sort64 的 inverse；decode/batch1 不额外生成 inverse。使用已有 `AUX_SORT_OPUS` 常量和具名解包，移除 `sort_result[6]`。
  - 普通 `sorted_sum` 改用有界 buffer copy：负索引使用安全基址和 0 字节可读范围，加载结果为零；有效行只允许读取 N 列，保留原输入行 padding。无需新增零行、清零 kernel 或改变 FP32 求和顺序。旧 `invert_sorted_ids` helper 仍保留供外部兼容，但本后端和本测试文件均不再调用。
  - 公共排序分发语义保持原状：`output_aux=AUX_SORT_OPUS` 沿用已有自动分发规则；删除前一轮新增的显式策略保持断言。inverse 测试改为覆盖 sort64/sort256，加入普通归约的负索引和全空 route 测试。
  - GPU 1，4 轮同进程交替、每轮 5 次 warmup + 20 次 profile、3 份输入，基线为独立 inverse + 原裸指针 sorted_sum。整体 MoE（µs）：Qwen3.8 FP8 I192 4k / 1x4：400.52 → 397.25；I192 32k / 4w+8x1 compact（二次排序）：1822.35 → 1817.08；Qwen3.8 BF16 I320 8k / default：1315.47 → 1311.98。三者输出逐位一致。
  - **验证**：gfx950 测试目标共 67 个用例通过（34 MoE、5 packed reduce、16 sort inverse、12 masked sorted sum）。本机没有 gfx942，普通 sorted_sum 在 gfx942 上的执行与性能仍需该平台回归；未修改 CSV、未重跑全量调优。
- 2026-10-24：按用户要求，普通 `sorted_sum` 的输入读取改回 `UniversalCopy`，输出仍用原有 `BufferCopy`。移除输入 buffer descriptor；每个 token 先判断是否包含负 inverse，全有效时直接向量加载，仅含负索引时使用零初始化 fragment + 有效性分支跳过读取，保持 FP32 累加顺序。
  - 逐次判断的初版在全有效输入下约慢 8%，改为 token 级快路径后，同卡 4 轮 A/B（4096 tokens、N=2560、TOPK=11、padding=128）原无保护 UniversalCopy 59.65 µs，新版本 59.91 µs，输出逐位一致。负索引、全空 route、padding、canary 检查通过；该数据是归约单核微测，不是端到端收益。
