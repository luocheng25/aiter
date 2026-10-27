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
  | Qwen3.8-i320 | 2560 | 320 | 513 | 11 | SiLU |
  | Qwen3.8-i192 | 2560 | 192 | 513 | 11 | SiLU |

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
| sort | aiter `moe_sorting`（opus） | — | block：Down 为 M128 或 Gate/Up 为 4 波（含半高）时取 256，否则取 64 |
| 输入 quant | aiter per-token | — | — |
| Gate/Up M64 | prefill 1x4 | `gemm1.py` | `64_{256,128}_{128,256}_True`；使用 sort64 元数据，搭配 M128 Down 时读取 sort256 元数据 |
| Gate/Up 4 波 | 2×2 波，256×256 tile，persistent，stream-K（余数 ≤ C/8 时均分，否则把尾部 tile 按 K 拆成 ≤ 4 段） | `gemm1_2x2.py` | `256_256_128_True`，使用 sort256 元数据 |
| Gate/Up 4 波半高 | 128×256 tile，每个 K tile 2 个象限；sort256 块的上半块按 4 波的调度（含 stream-K），下半块只在有路由时计算 | `gemm1_2x2.py`（`rows=128`） | `128_256_128_True`，使用 sort256 元数据 |
| 中间 quant | FlyDSL 短行 per-token（128 < I ≤ 1024）；其余用 aiter | `quant.py` | — |
| Down M64 1x4 | 64×256 tile，按 N 切成 n_splits 份 | `gemm2_1x4.py` | `:1x4_64x256:128` |
| Down 8x1 compact | M256 full + M64 tail 任务 | `gemm2_8x1.py`, `gemm2_8x1_compact.py` | `:8x1_compact:128` |
| Down M128 | 4×1 波，128 行块，packed 输出（不乘 activation scale 和 topk weight） | `gemm2_4x1.py` | `:m128:0`（one-shot），`:m128p:0`（persistent）；Gate/Up 可以是 M64 或 4 波；K % 64 == 0 且 128 ≤ K ≤ 640 |
| topk reduce | sort 返回的 inverse + `sorted_sum`（M64 Down）；`make_packed_sum`（M128 Down，每条路由乘 activation scale × topk weight 后求和） | `moe_reduce.py`, `gemm2_4x1.py` | — |

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
10. **4w Gate/Up stream-K 尾部拆分（2026-10-25）**
    - 余数 rest ≤ C/8 时仍用原来的均分 stream-K。
    - 其余情况：最后一轮剩下的 rest 个 tile 各按 K 拆成 f = min(C // rest, 4) 段（f ≥ 2 才拆）。第 c 段处理 K pair [c·P/f, (c+1)·P/f)。
    - 非首段把 partial 用 SC1 写进自己 slot 的 workspace，再对首段的 flag 做原子加。首段在主循环之后等 flag 达到 f−1，在 epilogue 里按静态展开、动态掩码的方式合并这些 partial，最后清零 flag。
    - 真实输入每份的余数不同，例如 MiniMax-e128 8k 三份输入的余数分别是 37、76、58。
    - MiniMax-e128 8k 的 Gate/Up：b2b 324 → 299 µs，sleep 257 → 235 µs。其余 shape 在 ±1% 内。
11. **4w Gate/Up 热循环交织（2026-10-25）**
    - 改动前 ISA 里 2 个 MFMA 后面跟着成簇的 8 个 ds_read 和 4 个 DMA，每个 512 cycle 的象限约有 140 cycle MFMA 空闲（ATT：MFMA 58%，LDS DMA 12%，ds_read 11%，wait_vmcnt < 2%，属于发射受限）。
    - `_schedule_compute` 改为 DMA 在前：MFMA×1，(VMEM, MFMA)×vmem，(DSRD, MFMA)×dsrd，再接剩余的 MFMA。每个象限的程序顺序也改为 wait → DMA → ds_read → MFMA。编译器认为 LDS DMA 写可能与 LDS 读重叠，所以 DMA 必须在 ds_read 之前，否则交织会被打乱。
    - b2b 下 Gate/Up 快 3–12%（多数 8–11%），每个象限的 cycle 减少约 15%。
    - sleep 下 MiMo 16k/32k、MiniMax-e128/e129 16k、Qwen3.8 16k、Qwen397 16k/32k 慢 6–15%：MFMA 更密集后触发功耗降频（例如 MiMo 32k 从 1.39 GHz 降到 1.25 GHz）。b2b 收益更大，所以保留。
12. **M128 Down 的 row factor 移到 `make_packed_sum`（2026-10-25）**
    - Down 的 epilogue 不再乘 activation scale × topk weight，去掉了 LDS 里的 row factor 区和每个 MFMA 对应的 `v_mul_f32`。
    - `make_packed_sum` 新增 `p_sorted_weights`、`p_route_scales` 两个参数，每条路由乘 `sorted_weights[loc] * route_scales[token*TOPK + route]` 后再求和。kernel ABI 里 Down 仍保留这两个指针，但不再使用。
    - Down：b2b 快 1–9%，sleep 最多快 15%（Qwen3.5-i128 32k）；reduce 慢约 1%。
    - `packet_index` 简化：`rotation` 只算一次，取模改为条件回绕。性能持平（±1%），Qwen3.8 32k sleep 快 2.6%。
13. **M128 Down 支持 K = 192 / 320（2026-10-25）**
    - 条件从 `K % 128 == 0` 放宽为 `K % 64 == 0`。KB = ceil(K/128)，超出 K 的 A chunk 置零，因此 B 多读的部分乘的是 0。
    - Qwen3.8 i192 / i320 的 Down 在 2k–32k 从 2.82–4.09 TB/s 提高到 3.66–4.30 TB/s；整体 MoE 快 3–11%（i320 2k 持平）。
14. **host 开销（2026-10-25）**
    - `common.launch(exe, *args)`：直接调用编译产物的 `_cf`，张量传 `data_ptr()` 整数，stream 用 `torch._C._cuda_getCurrentRawStream`（约 0.2 µs；`torch.cuda.current_stream()` 约 2.7 µs），不再为每个参数构造 PointerJitArg。没有 `_cf` 时退回 `_run_compiled`。Gate/Up、Down、quant、reduce 的发射都改用它；`fused_moe_gfx942._launch` 保留，测试工具会替换它。
    - 缓存配置串解析（`_parse_config`）和支持性检查（`_unsupported_reason`）；`aiter/fused_moe.py` 缓存 OPUS sort 的 workspace 大小（`_opus_sort_workspace_size`）。
    - 直接调用本后端时，每次 MoE 的 CPU 时间从 153–164 µs 降到 106–121 µs。Qwen3.5-i128 1k 每次调用的 stream 时间从 157 µs 降到 114 µs（kernel 合计 88 µs，仍是 host 受限）。
    - 分项（Qwen3.5-i128 1k）：3 次 FlyDSL 发射 52.5 → 20 µs；`moe_sorting` 47 → 37 µs（其中 aiter op 包装约 20 µs、`torch.empty` 约 9 µs）；aiter per-token quant 每次约 17 µs，没变。
    - 剩下的主要是 aiter op 的 torch custom op 和 pybind 转换（每个 op 15–20 µs，属于 aiter 公共部分）。经生产 `fused_moe` 入口调用还要多约 40 µs 的分发（`fused_moe_` custom op、`_fused_moe_impl`），每次约 157 µs。
15. **4w Gate/Up 半高 tile（`128_256_128_True`，2026-10-25）**
    - 仍用 sort256 元数据，每个块分成上下两个 128 行的半 tile。上半块按原来 4 波的 persistent 调度（含 stream-K 尾部拆分）。下半块只在有路由时计算：每个 CTA 先按轮转（逻辑 tile jb·C + slot）做完自己那份有路由的下半块（不拆 K），再做上半块。CTA 开始时先数出自己的下半块任务数，下一个候选用动态 while 循环查找。
    - 每个 K tile 只有左右 2 个象限：A 的寄存器双缓冲，每个 K tile 3 组 DMA（B 左、B 右、A），in_flight 从 5 降到 3。`_schedule_dense` 把 8 个 ds_read + 8 个 DMA、16 个 ds_read + 4 个 DMA 分别铺到两个象限的 MFMA 之间。
    - VGPR 408 + AGPR 188，没有溢出（256 行版本有 4 个）。256 行版本的代码路径不变。Gate/Up 输出与 256 行版本逐位一致。
    - 对比 256 行版本（同进程交替，b2b，Down 都是 m128；整体耗时比 半高 / 256 行，括号内为 Gate/Up）：

      | tokens | 结果 |
      |---|---|
      | 2k | Qwen3.5-i128 0.77（0.60），Qwen3.8-i320 0.85，i192 0.86，Qwen3.5-i512 0.87，Qwen397 0.90，Qwen3.8-i640 0.90，MiMo 0.91，MiniMax-e129 0.92，e128 0.94（0.90） |
      | 4k | Qwen3.8-i320 0.89，i192 0.89，Qwen3.8-i640 0.92，Qwen397 0.92，MiMo 0.93（0.83）；MiniMax-e128 1.12（1.32） |
      | 8k–16k | Qwen3.5-i128 8k 1.07（1.21），MiMo 16k 1.13（1.37），MiniMax-e129 8k 1.16（1.40） |

    - 超过 128 行的块要算两个半 tile，比一个 256 行 tile 慢，所以只在几乎所有块都不超过 128 行时有收益。和各点原来的最优配置（很多是 M64 Gate/Up）比：4k 时 MiMo 快 6%、Qwen397 快 5%、Qwen3.8-i640 快 8%、i320 快 3%、i192 持平；2k 互有胜负（MiniMax 快 5–6%，Qwen3.8 i192 / i320 慢 5–9%）。
    - tune space 增加 5 个半高配置（Down 为 default、1x4_64x256、8x1_compact、m128、m128p）。重调后最终替换 4 行（见变更记录）。

## 4. 当前状态（2026-10-25）

- **目标口径（2026-10-25 起采用）**
  - Gate/Up：每个 expert 平均路由数 rows/E ≥ 150 时 1200 TFLOP/s（b2b）；rows/E < 150 时取 min(1200, 0.9 × 权重流式 roofline)。roofline = 4·R·D·I / ((tokens·D + E_act·2I·D) / 5.2 TB/s + R·I·2 / 4.65 TB/s)。
  - Down（理想字节数 / 时间）：I ≤ 384 时 4.5 TB/s；I ≥ 512 时，tokens ≥ 16k 为 4.0 TB/s，更小的 batch 仍为 4.5 TB/s。
  - b2b 和 sleep 两种口径都要报告；FLOP 只按有效路由计（不计 padding）。
- **对照目标（§3.10–3.13 之后，9 个模型配置 × 1k–32k 共 54 个点，每个点取 b2b 最优配置，同一配置再测 sleep；`gap/after_table.md`）**

  | 指标 | 改动前 | 改动后 |
  |---|---:|---:|
  | Gate/Up 达标（b2b） | 17/54 | 17/54 |
  | Gate/Up 达标（sleep） | 23/54 | 24/54 |
  | Down 达标（b2b） | 2/54 | 6/54 |
  | Down 达标（sleep） | 10/54 | 14/54 |
  | 整体耗时几何平均（改动后 / 改动前） | — | b2b 0.961，sleep 0.984 |

  - Gate/Up 16k–32k 基本达标（b2b 1380–2045 TF）。没达标的集中在 8k（b2b 835–1162 TF）和 4k 及以下（离目标差 20–35%）。8k 的主要原因是 sort256 的 padding：rows/E 为 171–320 时，按 256 行补齐后计算量是有效量的 1.43–1.76 倍。
  - Down b2b 多数在 3.4–4.4 TB/s，sleep 3.4–4.9 TB/s。差距主要在 b2b 降频（b2b 1.25–1.6 GHz，休息后 1.85–2.1 GHz）。
  - MiniMax-e129 4k 的 b2b 最优配置从 4w 换成了 M64（b2b 只差 1%），这个配置在 sleep 下的 sort 计时异常偏高，所以表中 sleep 总时间是 +16.9%；仍用 4w 配置时 sleep 为 433 → 425 µs。
- **调优 CSV**：2026-10-25 重调后原地替换了 17 行，主要是 Qwen3.8 i192 / i320 改用 M128 Down，以及几个 2k–16k 的点改用 4w Gate/Up（见变更记录）。配置名没变的行（例如 4w Gate/Up 的行）直接用上了新 kernel，CSV 里的 `us` 仍是旧值。同日加入半高 tile 后再次重调，又替换 4 行（MiniMax-e128 / e129 2k、Qwen3.8-i640 4k、Qwen397 4k）。
- 对照目标的 54 个点是 §3.14–3.15 之前测的，之后没有重跑。

以下为 2026-10-23 的对比，之后没有重测：

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
- **4w Gate/Up 的 stream-K 与热循环（2026-10-25）**
  - 只放宽均分 stream-K 的阈值（MiniMax-e128 8k，余数 37）：基本没有收益。stream-K 部分每单位工作慢 50–80%，因为各 slot 的 K 区间漂移后失去了 L2 上的同步共享。
  - 尾部 tile 拆成 6 段：多个 partial 的合并成本超过收益，比不拆还慢；上限取 4。
  - ds_read 在 DMA 之前的交织顺序：在 Qwen3.8-i192/i320 32k 上比 DMA 在前慢 6–7%，其余持平。
- **M128 Down 热循环（2026-10-25）**
  - 去掉任务末尾的 store drain：±1%，回退。
  - 没做：m128p 的 A 预取（第二个 CTA 已经掩盖了这部分延迟）；B 的寄存器双缓冲和减少 barrier（lgkm 等待约占每个 MFMA 周期的 9%，已被另一个波掩盖；K640 已用 238 个 VGPR，没有余量；K>256 的 32k 受限于 fabric 带宽）。
  - **权重重读不是 K>256 Down 的瓶颈**。FORCE_E0 上界实验（所有任务都读 expert 0 的权重）只快 9–20%，大致等于把权重读一遍的时间（例如 MiniMax-e128 32k 约 66 µs ≈ 0.30 GB / 4.6 TB/s）。说明重读都命中 MALL，HBM 上的权重流量已接近理想值。让一个 CTA 连续处理同一 expert 的两个块来复用 B，收益空间不超过这个上界中的一小部分，与 2026-10-24 的 8 波实验（< 1%）一致；没有实现。
- **评估后没有实现（2026-10-25）**
  - **4w Gate/Up 的 128 行变体（sort128）**。理想情况下只对 MiniMax-e129 8k（padding 1.61 → 1.21）、Qwen3.5 8k（1.74 → 1.32）、MiniMax-e128 8k（1.43 → 1.22）有用；MiMo、Qwen397、Qwen3.8 的 8k 用 sort128 和 sort256 的 padding 相同。代价是每个 K tile 的 L2 → LDS 需求从 32 升到 47 B/clk/CU（按满 MFMA 速率），而现在 256 行版本实测只有约 20 B/clk/CU、MFMA 利用率约 61%；每个 K tile 只剩 2 个象限，A 的寄存器要双缓冲，预取距离缩短一半。估计每行效率下降 20–30%，基本抵消 padding 收益。此前“跳过空下半块”实测 b2b 也只快 2–5%。
    - 补充（同日）：上面只看了 8k 和整体改用 sort128。若仍用 sort256，只把 ≤ 128 行的块换成半高 tile，4k 时 MiMo、Qwen397、Qwen3.8 的全部块、16k 时这几个模型约一半的块都符合条件，收益比上面估的大，转入 §6 第 1 项。
  - **M64 Gate/Up 在一个块的多个 tile 之间复用 B**。要在一个 K tile 内用同一份 B 算完块内所有有效的 64 行 tile，A 的 LDS 和累加器都要按 256 行分配（每个波 256 个累加器），这就是一个带 64 行动态跳过的 M256 kernel。4w 的 2×2 布局里按 64 行跳过只会让部分波空闲，并不省时间；只有按 128 行跳过才省时间，而这正是已失败的“跳过空下半块”（溢出 116–322 个寄存器）。A 在 LDS 里 41% 的 bank conflict 也没有修：这个比例是在 4k–8k 测到的，那些点受限于 fabric 读带宽（4.6–4.9 TB/s），不受限于 LDS。调优 CSV 中 8k–16k 仍有 5 行使用 M64 Gate/Up（Qwen3.5-i512 8k、Qwen3.5-i128 16k、Qwen397 8k、Qwen3.8-i192 8k/16k），这些点没有单独评估。
- **2026-10-25（第三轮）**
  - **M128 Down 的 cache policy**（现在输出用 SC1|NT，aux=18；权重用 SC1，aux=16）。同进程交替 A/B，7 个模型配置 × 1k–32k，b2b 和 sleep：
    - 输出只用 NT（aux=2）：整体 0.989–1.018×，持平。
    - 输出用默认策略（aux=0）：32k 整体慢 1.2–6.1%，reduce 慢 3–9%，sleep 下 Down 最多慢 17%（MiniMax-e128 32k 471.7 → 549.9 µs）。≤ 8k 时 Down 有时快 4–10%，但 reduce 变慢，整体 0.978–1.020×。
    - 权重用默认策略（aux=0）：±1.2%。
    - 结论：保持现状。
  - **I=128 的中间 quant 改用 FlyDSL 短行 kernel**：随机数据逐位一致，但 Qwen397 4k 的真实数据里 45056 行中有 1 行的 FP8 值不同（aiter 对 128 列走 per-group kernel，用乘倒数而不是除法）。回退，`per_token_quant_supported` 仍是 128 < cols ≤ 1024。
  - **长行（2048–6144 列）用 FlyDSL quant**：与 aiter 不逐位一致，GPU 时间也没有收益。
  - **半高 tile 用在满块较多的点**：约一半块超过 128 行时（MiniMax-e128 4k、e129 8k、MiMo 16k、Qwen3.5-i128 8k），Gate/Up 慢 21–40%，整体慢 7–16%。半高版本关掉 stream-K 后整体仍慢 5–13%。

## 6. 待做（按预期收益排序；2026-10-25 更新）

当前各阶段占整体时间的比例（54 个点的 b2b 最优配置，按 tokens 取平均）：

| tokens | sort | 输入 quant | Gate/Up | 中间 quant | Down | reduce |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 6.3% | 2.1% | 50.0% | 2.0% | 32.8% | 5.7% |
| 4k | 6.6% | 2.9% | 44.0% | 2.0% | 31.1% | 12.7% |
| 32k | 4.2% | 4.6% | 32.9% | 2.7% | 31.9% | 23.4% |

1. **4w Gate/Up 按块混合 256 行和半高 tile；半宽 tile**（半高 tile 已完成，见 §3.15）。
   - 现在一个点只能整体选 256 行或半高。半高只在 2k–4k 的部分点胜出；8k–16k 约一半块超过 128 行，这些块拆成两个半 tile 会慢 21–40%。
   - 要做的是按块选择：超过 128 行的块用 256 行 tile，其余用半高 tile。两条路径各自包含 K 循环和 epilogue，避免通过 scf.if 传递累加器（之前因此溢出）；256 行路径本身已有 4 个溢出，寄存器余量是主要风险。
   - 按 bench 路由估算（半 tile 耗时按整 tile 的 0.75 计），混合后 Gate/Up 的 tile 工作量 8k 少 8–12%（MiniMax、Qwen3.5），16k 少 12%（MiMo、Qwen397、Qwen3.8）。
   - 半宽：I = 192 / 320 时最后一个 N tile 只有 128 列，还没做。按同样的估算，与混合半高合计 16k 少 19–23%；32k 只有半宽起作用，少 8–13%。
2. **host 开销的剩余部分**（FlyDSL 发射和后端自身的开销已处理，见 §3.14）。
   - 后端每次仍要 106–121 µs CPU 时间，大部分在 aiter op 包装（`moe_sorting`、per-token quant 各 15–20 µs）和 `torch.empty`。生产 `fused_moe` 入口另有约 40 µs。Qwen3.5-i128 ≤ 1k 时仍是 host 受限。
   - 方向：确认整套流程能被 CUDA graph 捕获；或在 aiter 公共层减少 custom op 包装。
3. **sort**。OPUS multi-phase 在 1k / 4k / 32k 分别要 10–13 / 18–21 / 45–65 µs，占整体 4–8%（Qwen3.5-i128 为 8–17%）。
   - 单 kernel 策略慢 20–400 倍；仓库里的 FlyDSL sort 速度相同。
   - 数据只有几 MB，工作量与路由数成正比的计数排序应能把 32k 压到约 10–15 µs。这是共享组件，其他后端也受益。
4. **输入 quant**。D = 6144 的 32k 约 150 µs（约 4.0 TB/s，占 5–8%）。
   - 可以在集成层接受上游（RMSNorm + quant 融合）给出的 FP8 激活和 scale，省掉整个 kernel。
   - 或者写一个长行版的 FlyDSL quant（短行版约 5 TB/s）。
5. **K>256 的 M128 Down**。K640 每个子包约有 336 个串行的非 MFMA cycle；I ≥ 512 时 16k–32k 的 b2b 只有 3.2–3.4 TB/s（目标 4.0）。
   - 方向：更大的子包，或把 A 放到 LDS 腾出 VGPR 给 B 双缓冲。
   - Down 的按路由输出已经是 SC1|NT；换其他 cache policy 没有收益（§5）。
6. **topk reduce**。平均占比从 1k 的 6% 升到 32k 的 23%（Qwen397 32k 为 32%）。
   - packed sum 已接近带宽上限（MiMo 32k：3.6 GB / 767 µs ≈ 4.7 TB/s），只有减少字节才有用。
   - 原子累加受限于原子操作速率（2026-10-24 实测慢 7–30 倍）。
   - 剩下的办法在集成层：与下游的 TP all-reduce 或 residual add 融合。
7. **中间 quant**（占 2–5%）。Gate/Up 写 BF16，并用原子 max 给出每行的 absmax；Down 加载 A 时自己量化。这样省掉 quant kernel 的一次读和一次写，结果可以逐位不变。
8. **4w Gate/Up 交织后的 sleep 回退**（7 个 16k–32k 点慢 6–15%）。**大余数尾波**：rest > C/2 时不拆，例如 MiniMax-e128 32k 最后一轮约有 23% 的 CU 空闲。
9. **M64 Gate/Up 的空 tile**（小 batch 时慢 0–5%）；**K>640 的 M128 Down**。

非性能项：

- stream-K 的 workspace 和 m128p 的队列计数器只按设备缓存。两个 stream 并发跑 MoE 会互相覆盖，应按 stream 区分（参考 `batched_gemm_a8w8_gfx950._split_counters(device, stream)`）。
- 普通 `sorted_sum` 的改动还没有在 gfx942 上回归。
- 调度参数（lead、OC 切分、stream-K 阈值）只用均匀路由测过，应该用真实的、不均匀的路由再验证。

原“K>256 的 M128 Down 减少权重重读”已撤下：FORCE_E0 实验表明重读命中 MALL，HBM 上的权重流量接近理想值（见 §5）。

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
| `retune950s.sh` + `apply_retune_s.py` | 2026-10-25 的重调：以仓库当前 CSV（`out950b/`）为基准，输出在 `out950s/`；只替换涉及 4w Gate/Up 或 M128 Down、同一次运行内快 ≥ 1.5% 的行 |
| `asm_ab/` | 用生产 `fused_moe`（`validate950.py`）交替计时两份只含几行的 CSV，复核换掉 ASM 1-stage 的行 |
| `gap/sweep2.py`、`gap/summarize_after.py` | 54 个点（9 个模型配置 × 1k–32k）的 b2b / sleep 扫描与对照目标的汇总（`after_table.md`） |
| `gap/ab_knob.py` + `gap/run_ab.sh`、`gap/run_ab_down.sh` | 同进程交替 A/B：用 `patch_gu4w(**kw)` / `patch_m128(**kw)` / `patch_down_module(name)` 切换 builder 参数或 Down 模块副本，并比较 Gate/Up 输出（`gu_diff`；最终输出经过 FP8 量化，对 Gate/Up 的差异不敏感）。arm `half` / `halfnosk` 是半高 tile，`o2` / `o0` / `w0` 是 M128 Down 的 cache policy（副本 `gap/gemm2_4x1_aux_*.py`） |
| `gap/host_overhead.py`、`gap/host_phases.py`、`gap/launch_cost.py` | 每次 MoE 调用的 CPU 时间与 stream 时间、分阶段的 CPU 时间、单次 FlyDSL 发射的 CPU 开销 |
| `gap/sort_policy.py`、`gap/quant_long_rows.py` | M128 路径 `moe_sorting` 各 dispatch policy 的耗时；FlyDSL 短行 quant 用在其他行长时与 aiter 的逐位比较和耗时 |
| `retune950t.sh` + `apply_retune_t.py` | 加入半高 tile 后的重调：基准 `out950c/`，输出 `out950t/`；只统计改用 `128_256_128_True` 且快 ≥ 1.5% 的行，`--apply` 写回仓库 CSV |
| `half_ab/` | 同 `asm_ab/`，复核改用半高 tile 的行 |
| `gap/pmc_kern.py`、`gap/att_kern.py`、`gap/isa_order.py` | 单 kernel 的计数器、ATT 指令级 trace、ISA 中 MFMA / DMA / ds_read 的排列 |
| `refactor/dump_isa.py` + `refactor/cmp_isa.sh` | 只编译、不运行：导出 25 个特化（12 个 4w Gate/Up、10 个 M128 Down、3 个 packed sum）的最终 ISA（`<目录>/<名字>/<kernel>_0/22_final_isa.s`），再与基线目录逐个比较；纯重构用它确认 ISA 逐字节一致。基线在 `refactor/isa_base/` |

gfx950 正确性测试：`op_tests/test_flydsl_moe_gfx942.py`（`FOUR_WAVE_CASES` 32 个，其中 6 个半高；`M64_M128_CASES` 10 个；另有 packed reduce、sort inverse 和 masked sorted sum 用例）。

## 8. 分支总结（`luocheng/moe_gemm_308_opt3-350`，2026-10-26）

本分支在 `luocheng/moe_gemm_308_opt3`（合并点 `e59153bd61`）之上有 4 个提交，都是 gfx950（MI350X）上 FlyDSL `fused_moe_gfx942` 后端的 MoE 优化。以 FP8 PTPC 为主；调优 CSV 也覆盖 Hunyuan3 FP8 per-tensor 和 Qwen3.8 BF16。改动集中在 `gfx9_fp8/`、`fused_moe_gfx942.py`、`op_tests/test_flydsl_moe_gfx942.py` 和 6 个 model_configs CSV。`luocheng/moe_gemm_308_opt3` 之后新增的 `c8e81586f4` 没有合入。

| 提交 | 主要内容 |
|---|---|
| `ab90f2bbf2` | gfx950 适配；M64 Gate/Up 的 XCD 映射和 K128 atom；4w Gate/Up；M128 Down（I ≤ 640）；M64 Gate/Up 读 sort256；FlyDSL 中间 quant；1x4 Down 按 N 切分；BF16 改用硬件 RNE；6 个 CSV 补 202 行 gfx950 配置 |
| `05e2f778a3` | M128 Down one-shot 按 XCD 排任务；重调 5 行 |
| `f1c4df747d` | 复用 OPUS sort 的 inverse；packed reduce 在 N % 2048 == 512 时用 512 列分块；`sorted_sum` 跳过负索引 |
| `4e2be2278e` | 4w Gate/Up 的 stream-K 尾部拆分、热循环交织和半高 tile；M128 Down 的 row factor 移到 reduce、支持 K % 64；降低 host 开销；重调 20 行 |

### 8.1 各阶段的改动和效果

| 阶段 | 改动 | 效果（未注明的为 b2b） | 见 |
|---|---|---|---|
| Gate/Up M64（`gemm1.py`） | XCD 连续映射、K128 MFMA atom；读 sort256 元数据，按 lead tile 顺序派发 | MiniMax 32k 1352 → 837 µs；小 batch 能搭配 M128 Down，1k–4k 整体快 2.4–4.4%（几何平均），最多 13.5% | §3.2、§3.8 |
| Gate/Up 4w（`gemm1_2x2.py`） | 移植 pyhip 的 4 波 FP8 GEMM（256×256 tile、persistent、LDS DMA ping-pong）；stream-K 和尾部 tile 按 K 拆分；DMA 在前的热循环交织；半高 tile | 32k 1450–2054 TF（M64 为 1307–1520 TF）；交织后快 3–12%；半高 tile 在 2k–4k 多数点整体快 6–23% | §3.3、§3.10、§3.11、§3.15 |
| 中间 quant（`quant.py`） | FlyDSL 短行 per-token（128 < I ≤ 1024），与 aiter 逐位一致 | 约 5 TB/s（aiter 为 1.3–2.4 TB/s） | §3.4 |
| Down 1x4 / 8x1 | 1x4 按 N 切分，每个 CU 至少 6 个任务；8x1 改为 XCD 连续映射 | MiniMax-e128 4k 的 1x4 Down 182 → 145 µs | §3.2、§3.5 |
| Down M128（`gemm2_4x1.py`） | 移植 pyhip M128 Down 并改为 PTPC：packed 输出、merge_halves、persistent `m128p`；K ≤ 640 的 K128 子包和 OC 切分；one-shot 按 XCD 排任务；row factor 移到 reduce；支持 K % 64 | I ≤ 256 约 4.0–4.6 TB/s；32k 比最优 M64 Down 快 19–24%（I = 384–640）；按 XCD 排任务后 Qwen3.5-i512 2k–16k 快 3–20%；Qwen3.8 i192 / i320 从 2.8–4.1 提高到 3.7–4.3 TB/s | §3.6、§3.7、§3.9、§3.12、§3.13 |
| reduce（`make_packed_sum`、`moe_reduce.py`） | packed sum 按 N 选分块宽度；复用 OPUS inverse，不再单独求逆；`sorted_sum` 跳过负索引 | packed sum 约 4.7 TB/s；Qwen3.8-i640 32k reduce 453 → 419 µs；省掉求逆的 2.6–4.8 µs | 变更记录 2026-10-24 |
| BF16 输出 | gfx950 用硬件 RNE 指令 `v_cvt_pk_bf16_f32` | 性能持平；与原 RTA 只在舍入平局处差 1 ulp | 变更记录 2026-10-23 |
| host | `common.launch` 直接调用编译产物；缓存配置解析、支持性检查和 sort workspace 大小 | 每次调用的 CPU 时间 153–164 → 106–121 µs | §3.14 |

### 8.2 整体结果

- **对比 pyhip**（2026-10-23，7 个模型 × 1k–32k，双方 20 次迭代）：b2b 42/42 胜出，几何平均 1.25×（1.14–1.55×）；sleep 42/42 胜出，几何平均 1.21×（1.06–1.44×）。之后的改动没有重新对比。
- **对比 aiter 自带的 PTPC kernel**（ASM 1-stage、ASM stage1 + CK stage2、CK 2-stage；在 I > 256 的 M128 Down 加入之前测得）：两种口径都是 39/42 胜出，几何平均 b2b 1.46×、sleep 1.53×。输掉的 3 个点都在 1k，对手是 ASM 1-stage。
- **调优 CSV**：6 个 CSV 共 202 行 gfx950 配置。
  - 本后端 156 行（≥ 512 tokens 的 73 行，更小 batch 的 83 行）；ASM 1-stage 17 行；aiter CK 等 2-stage kernel 29 行（Qwen3.8 BF16 27 行、Qwen3.5-35B 2 行）。
  - ≥ 512 tokens 的 73 行里，Gate/Up 用 4w 的 32 行、半高 4 行、M64 37 行；Down 用 M128 的 56 行（one-shot 42、persistent 14）。
  - 与 aiter 2-stage 调优的最优配置相比（只统计被替换的行，2026-10-23）：MiniMax 1.98×、Qwen397 1.89×、Qwen3.5-35B 1.61×、Qwen3.8 FP8 1.53×、Qwen3.8 BF16 1.30×。
- **对照目标**（2026-10-25，54 个点，§4）：
  - Gate/Up 达标 b2b 17/54、sleep 24/54。16k–32k 基本达标（b2b 1380–2045 TF）；没达标的集中在 8k（主要因为 sort256 的 padding）和 4k 及以下（差 20–35%）。
  - Down 达标 b2b 6/54、sleep 14/54。多数点 3.4–4.4 TB/s，主要差在 b2b 降频。
  - §3.14–3.15 之后没有重测。

### 8.3 经验

1. **测量口径**：持续负载会降频（b2b 1.25–1.6 GHz，休息后 1.85–2.1 GHz）。对比时迭代次数必须一致，b2b 和 sleep 都要报告，小于 5% 的差异要在同一进程内交替 A/B。MFMA 更密集的改动可能 b2b 变快、sleep 变慢（§3.11）。
2. **派发顺序和数据局部性常常比 kernel 内部更关键**。
   - M64 读 sort256 时，空 tile 夹在有效 tile 之间会慢 17%。
   - M128 one-shot 按 XCD 排任务后，Qwen3.5-i512 4k 的 Down 读流量从 0.57 GB 降到 0.30 GB。
   - 均分 stream-K 在余数大时会打乱各 CTA 在 L2 上的同步共享，所以大余数改为把尾部 tile 按 K 拆段。
3. **4w Gate/Up 受寄存器限制**：通过 scf.if 传递累加器（386 个 VGPR 溢出）和跳过空下半块（116–322 个溢出）都失败了；半高 tile 用独立的 2 象限代码路径才没有溢出。
4. **调优结果要复核**：`--e2e_tune` 选出的切换里，3 个从 ASM 1-stage 换到本后端的、1 个换成半高 tile 的，用生产 `fused_moe` 交替计时都没有得到确认，没有采用。
5. **数值**：quant 类改动要用真实数据逐位比较。I=128 的 FlyDSL quant 在随机数据上逐位一致，在真实数据上 45056 行中有 1 行不同。
6. **工具**：FlyDSL 磁盘缓存可能返回旧的二进制；测量期间不能改 kernel 源码；gfx950 上 rocprofv3 的 `FetchSize` 把 128B 读请求按 64B 计。

### 8.4 未完成

- **性能**（详见 §6，按预期收益排序）：4w Gate/Up 按块混合 256 行和半高 tile，以及半宽 tile；host 剩余开销；sort；输入 quant；K > 256 的 M128 Down；reduce 与下游融合；中间 quant 融合进 Gate/Up；交织后的 sleep 回退和大余数尾波。
- **非性能**：stream-K 的 workspace 和 m128p 的队列计数器只按设备缓存，多个 stream 并发跑 MoE 不安全；普通 `sorted_sum` 等改动没有在 gfx942 上回归；调度参数只用均匀路由验证过。
- **代码风格**：本分支改动或新增的 13 个 Python 文件中，`gemm1_2x2.py`、`gemm2_4x1.py`、`common.py` 在 2026-10-26 重构后通过 black 26.10.1 和 ruff 0.16.0；仍有 6 个不符合 black（`__init__.py`、`quant.py`、`gemm2_1x4.py`、`gemm2_8x1.py`、`fused_moe_gfx942.py`、`test_flydsl_moe_gfx942.py`），ruff 剩 `quant.py` 的 1 个 SIM102。CI 用这两个工具检查，提 PR 前要处理。
- **测试**：`op_tests/test_flydsl_moe_gfx942.py` 在 gfx950 上有 75 个用例（42 个整体 MoE、5 个 packed reduce、16 个 sort inverse、12 个 masked sorted sum），最近一次全部通过；gfx942 部分没有在本机跑。

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
- 2026-10-25：采用新的目标口径（§4），按 8 条改进建议依次处理：第 1、2、3、5 条完成，第 4 条只做了一部分，第 6–8 条评估后没有实现。
  - **测法**：同进程交替 A/B（`gap/ab_knob.py`），8 张卡各跑一组 shape；每轮 warmup 后 profile 20 次，3 份真实路由的输入（各自的余数不同），b2b 和 sleep 都测。Gate/Up 的改动用 Gate/Up 输出比较（`gu_diff`）确认逐位一致，因为最终输出经过 FP8 量化，对 Gate/Up 的差异不敏感。
  - **1. stream-K 尾部拆分**（§3.10）：MiniMax-e128 8k 的 Gate/Up b2b 324 → 299 µs，sleep 257 → 235 µs；其余 shape 在 ±1% 内。另用一次负测试确认 head 的合并路径确实执行。只放宽阈值和拆成 6 段的结果见 §5。
  - **2. 热循环交织**（§3.11）：b2b 下 Gate/Up（µs）MiniMax-e128 16k 429.5 → 379.8，MiniMax-e129 32k 838.0 → 745.5，MiMo 32k 921.0 → 824.6，Qwen3.8-i640 32k 1406.3 → 1284.3，Qwen3.8-i320 32k 757.6 → 683.5。sleep 下 7 个 16k–32k 点慢 6–15%（例如 MiniMax-e129 16k 391.2 → 436.0），原因是功耗降频。
  - **3. row factor 移到 `make_packed_sum`**（§3.12）：Down b2b 快 1–9%，sleep 最多快 15%；reduce 慢约 1%。packed sum 的测试改用 2 的幂次的 weight 和 scale，结果仍精确相等。
  - **4. K>256 Down 的热循环**：只保留 `packet_index` 简化（持平，Qwen3.8 32k sleep 快 2.6%）；去掉 store drain 已回退，B 双缓冲和减少 barrier 没做（§5）。
  - **5. M128 Down 支持 K=192/320**（§3.13）：op test 增加 2 个用例（tokens=4096、I=192，`256_256_128_True:m128:0`；tokens=300、I=320，`256_256_128_True:m128p:0`；都是 D=2560、E=513、topk=11）。
  - **6. K>256 Down 的权重复用**：FORCE_E0 上界实验表明重读命中 MALL，没有实现（§5）；§6 撤下原第 3 项。
  - **7. sort128 4w Gate/Up**、**8. M64 Gate/Up 的 B 复用和 LDS bank conflict**：评估后没有实现（§5）。
  - **第 1–5 条合计**（54 个点，§4）：整体耗时几何平均 b2b 0.961、sleep 0.984。Down 达标点 b2b 2 → 6、sleep 10 → 14；Gate/Up 达标点 b2b 17 → 17、sleep 23 → 24。
  - **重调 CSV**：5 个 FP8 模型以仓库当前 CSV 为基准重跑 `--e2e_tune`（`retune950s.sh`）。只考虑涉及 4w Gate/Up 或 M128 Down、同一次运行内快 ≥ 1.5% 的切换，共 21 行；Hunyuan3 per-tensor 没有这类切换（它的 6 个切换只在未改动的部分之间变化：M64 的 tile，或 1x4 / 8x1 Down）。其中 4 行没有得到复核确认，保留原配置：
    - Qwen3.5-i128 512 / 1024 和 Qwen3.8-i320 1024 原来用 ASM 1-stage。用生产 `fused_moe` 交替计时 3 轮（`asm_ab/`），本后端分别只快 1.0%、0.7%，以及慢 2.3%。
    - MiniMax-e128 2k 从 m128 改为 m128p，调优只快 2.0%，而 54 点扫描里 m128 更快。
  - **最终原地替换 17 行**（不增加 shape）：
    - Qwen3.8 i192 / i320 的 2k–32k 共 10 行改用 M128 Down，快 3.7–11.8%。
    - Qwen3.8-i640 8k：m128p → m128，快 2.2%。
    - MiniMax-e129 2k、e128 4k、e129 8k 改用 4w Gate/Up，快 3.6–4.8%。
    - Qwen3.5-i512 2k（m128p → m128，快 4.3%）和 16k（改用 4w，快 4.8%）。
    - Qwen397 32k：m128p → m128，快 3.1%。
  - **验证**：5 个模型共 154 个 gfx950 行都用生产 `fused_moe`（`test_moe_2stage.test_fmoe`）跑过，全部通过；prefill 行（token ≥ 512，共 73 行，其中 33 行用 4w Gate/Up、55 行用 M128 Down）的 logits_diff ≤ 4.4e-4。shape 冲突检查（`test_config_shape_collision`）通过；op test（26 个 4w 用例 + 10 个 M64/M128 用例 + packed reduce 等）通过。
- 2026-10-25（第三轮）：按 §6 的建议依次做 host 开销（第 2 项）、Down 输出的 cache policy 实验（第 5 项）、4w Gate/Up 半高 tile（第 1 项的一部分）。
  - **host 开销**（§3.14）：新增 `common.launch`，并缓存配置解析、支持性检查和 OPUS sort 的 workspace 大小。直接调用本后端时每次 CPU 时间 153–164 → 106–121 µs；Qwen3.5-i128 1k 每次调用的 stream 时间 157 → 114 µs。I=128 的中间 quant 改用 FlyDSL 的尝试因为真实数据上不逐位一致而回退（§5）。
  - **cache policy**：输出只用 NT、输出用默认策略、权重用默认策略三种都没有收益，保持 SC1|NT / SC1（§5）。
  - **半高 tile**（§3.15）：`gemm1_2x2.py` 的 builder 增加 `rows`（128 / 256），`fused_moe_gfx942` 增加 `128_256_128_True` 配置（sort 仍用 256 行块），tune space 增加 5 个配置。op test 增加 6 个用例（tokens 7 / 300 / 2048 / 4096，覆盖 m128、m128p、1x4 Down、default Down、per-tensor 和 Swiglu）。
  - 2k–4k 多数点比 256 行版本快 6–23%，约一半块超过 128 行的点（MiniMax-e128 4k、8k–16k）慢 7–16%。
  - **重调 CSV**：5 个 FP8 模型以仓库当前 CSV 为基准重跑 `--e2e_tune`（`retune950t.sh`）。只考虑改用半高 tile、同一次运行内快 ≥ 1.5% 的切换，共 5 行；Qwen3.5-35B 和 Hunyuan3 没有（Qwen3.5 在 2k–4k 仍是 M64 Gate/Up 更快，例如 i128 4k 半高 209.6 µs，M64 172.7 µs）。用生产 `fused_moe` 交替计时 3 轮复核（`half_ab/`，GPU 6）：

    | 行 | 原配置 | 原 → 新（µs） | 调优时 | 复核 |
    |---|---|---|---:|---:|
    | MiniMax-e128 2k | `64_128_256_True:m128:0` | 320.6 → 306.3 | 1.077× | 1.047× |
    | MiniMax-e129 2k | `256_256_128_True:m128:0` | 344.6 → 332.7 | 1.053× | 1.036× |
    | Qwen3.8-i640 4k | `64_256_128_True:m128p:0` | 804.8 → 750.6 | 1.067× | 1.072× |
    | Qwen397 4k | `64_256_128_True:m128:0` | 427.3 → 418.1 | 1.016× | 1.022× |
    | Qwen3.8-i320 4k | `64_128_128_True:m128p:0` | 486.5 → 481.2 | 1.051× | 1.011×（不采用） |

  - 原地替换前 4 行，新配置都是 `128_256_128_True:m128:0`；Qwen3.8-i320 4k 保留原配置。
  - **验证**：op test（32 个 4w 用例 + 10 个 M64/M128 用例 + packed reduce 等）通过；替换的 4 行用生产 `fused_moe` 检查，logits_diff ≤ 4.4e-6；shape 冲突检查通过。没有重跑 54 点扫描、sleep 口径和 pyhip 对比。
- 2026-10-26：参照 gfx950-gluon-tutorials 的 GEMM（gfx942 的 intra_wave / inter_wave），重构 `gemm1_2x2.py`（4w Gate/Up）和 `gemm2_4x1.py`（M128 Down），让 kernel 主干只写流水线，细节放到 builder 内的工厂函数。
  - **Gate/Up 主干**：tile 调度 → prologue（首个 job 的 K tile 0/1 → LDS，再读 B_l(0)、A_t(0)）→ 每个 job 的 K pair 循环 → 最后一个 K pair 与 epilogue 重叠并预取下一个 job → stream-K head 等待并合并 partial。
  - **`k_pair` 按 region 书写**，对应 intra_wave：每个 region 是一个 `with _region(group, in_flight, dma, read)`，里面依次是 DMA fill、LDS read、MFMA。进入时等待 DMA 组，退出时按 `dma` / `read` 的个数把它们交织进 16 条 MFMA。region 表写在模块 docstring 里。
  - **Down 主干**：任务队列 → 定位任务 → 排序 ID 和 weight scale 写入 LDS → B ring prologue 和 A 行进 VGPR → packet 循环（静态 prologue、动态稳态循环、静态尾部）→ 存最后两个 packet。packet 的 memory / compute 两个 stage 对应 inter_wave 的 `warp_pipeline_stage`，在 docstring 里列成表。
  - **工厂函数**：
    - Gate/Up：`kernel_context`、`tile_schedule`、`global_io`、`mma_operands`、`stream_k_exchange`、`epilogue`、`prefetch`、`k_pair`。
    - Down：`kernel_context`、`task_queue`、`locate`、`slice_scales`、`b_ring`、`lane_rows`、`packet_pipeline`、`next_half_live`。
    - 工厂函数返回 `common.Bundle`。
  - **FlyDSL 的约束**：
    - 动态 `if` / `for` / `while` 只能写在 kernel 函数体里（AST rewriter 不改写 builder 层的函数）。
    - rewriter 会把在动态区域里被调用方法或下标赋值的变量当作该区域的携带值。装着函数的 `SimpleNamespace` 会报 `Cannot extract IR values`。`Bundle` 实现了 DslType 协议、不含 IR 值，可以穿过动态区域；它不能继承 `SimpleNamespace`，因为 `_promote` 会重建 `SimpleNamespace`。
    - 编译缓存的 key 包含同目录模块级函数和闭包函数的源码，不包含类方法和属性访问。所以 helper 都写成函数，`_region` 用 `contextlib.ExitStack` 回调模块级的 `_interleave`。
    - IR 的生成顺序会影响最终 ISA：把 output rsrc 移到 operand 之后，ISA 有 20–392 行不同。所以工厂函数保持原来的生成顺序；拆长表达式时也保持操作数的求值顺序。
  - **代码风格**：这两个文件和 `common.py` 通过 black 26.10.1 和 ruff 0.16.0；原有的 21 个 B023 随闭包改写消失。`gemm2_4x1.py` 里三处嵌套 `if` 不能合并（两处外层是编译期 guard，一处要先确定线程 0 再读 LDS），都加了带原因的 `# noqa: SIM102`。
  - **验证**：用 `refactor/dump_isa.py` 只编译 25 个特化，最终 ISA 与重构前逐字节一致（`cmp_isa.sh`：identical=25），所以性能和数值都不变，没有重新计时。25 个特化是：12 个 Gate/Up（rows 128 / 256、SiLU / SwiGLU、per-tensor、关闭 stream-K、I = 192 / 512 / 640、Qwen397）、10 个 Down（one-shot / persistent、merge_halves、OC 切分、K = 128–640、sort128）、3 个 packed sum。另外用 `FLYDSL_RUNTIME_ENABLE_CACHE=0` 重新编译后跑了 op test（32 个 4w 用例 + 10 个 M64/M128 用例等），全部通过。
- 2026-10-26（续）：逐行对照参考的 `matmul_kernel.py`（intra_wave / inter_wave）复查，又做了一轮重构。
  - **复查结论**：参考的主循环在 kernel 里逐个 region 写出，每个 region 一个标题、每条访存一行，并在注释里标出 K 下标（`# LR A_bot(k)`、`# GR B_left(k+2)`）。我们的 region 体原来缺这些标注，`_region(4 * buf + 0, in_flight, dma=1, read=1)` 的参数也太多。Down 把上一个 packet 的 epilogue 编码成元组操作队列，再用 `run_op` 分派，读起来不直观；最后一个 packet 的 `pack` 又把同一个 epilogue 写了一遍。
  - **Gate/Up**：
    - `k_pair` 按参考主循环的写法改写：每个 region 一行 `with region(...):  # C_tl += B_l A_t`，三条操作分别注明 `# DMA B_l(k+2)`、`# LR A_b(k)`，与 docstring 的 region 表一一对应。
    - builder 内的 `region(group, dma=1, read=1)` 绑定 `in_flight`。
    - 主干分成 Prologue、本 job 与下一个 job、Main loop、Last K pair 四段。K 循环写成 `acc0` / `acc` / `yield`。新增 `op.source(tile, a_reg)`，本 job 和下一个 job 的 DMA 来源由 5 行缩为 2 行。
  - **Down**：
    - epilogue 改为生成器 `finish_row` / `finish_steps`：按顺序直写，每步之后 `yield`。compute stage 每条 MFMA 后调用 `per_step` 次 `next(steps, None)`，代替原来的操作队列和游标。
    - 最后一个 packet 也复用 `finish_row`，删除重复的 `pack`。
    - `packet` 里标出两个 stage 的优先级和内容；两个连续的 `if task.live:` 合成一个；任务循环加了分段标题。
  - **仍与参考不同的地方**：
    - FlyDSL 的动态 `if/for/while` 只能写在 kernel 体里，所以半高 tile 的 bottom 候选搜索、stream-K 自旋等待和任务认领仍留在主干。
    - 主循环体 `k_pair` 是函数，因为每个 job 的最后一个 K pair 要复用它预取下一个 job。
    - 循环携带值需要显式的 `init` / `yield`。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25），重新编译后 op test 全部通过；black / ruff 通过。
- 2026-10-27：按参考 `a16w16_intra_wave_gfx942` 的标准再次重构：主流程直接写出流水线，枝节放到 helper 里。
  - **Gate/Up 主流程**（`moe_gateup_4w`）：
    - Prologue 逐条写出 job 0 的 K tile 0、1 的 DMA（每个 K tile 有 B_l、A_t、A_b、B_r 各一组，每组每线程 4 × 16 B；半高 tile 没有 A_b），以及 LR B_l(0)、A_t(0)。
    - 主循环按 region 逐条写出 MFMA、LR、DMA，并标出 K 下标（如 `# LR A_b(k)`、`# DMA B_l(k+2)`）；全高和半高两种 region 序列都直接写出。
    - 每个 job 的最后一个 K pair 像参考的 peel 一样再写一遍，注明 DMA 的 k+2 是下一个 job 的 K tile 0、1；epilogue 的 VMEM 读放在它前面。
    - 最后是 epilogue 的 stores 和 stream-K head 的合并。
  - **Down 主流程**（`moe_down_m128`）：
    - Prologue 写出 LDS staging、B ring 的 DMA 和 A 行的 VMEM 读。
    - packet 的静态 prologue、动态主循环和静态尾部都直接写出 `begin`、每个 sub-packet 的 `memory` / `compute` 两个 stage、`end`；两个 stage 的内容（LR、DMA、MFMA，以及 q-1 的 epilogue、q-2 的 stores）写在注释表里。
  - **隐藏的枝节**：
    - Gate/Up：半高 tile 的 bottom 候选计数和查找、job 的 A offset 寄存器（`tile_schedule`、`job_cursor`）；stream-K 的发布和 head 合并（`epi.finish` / `epi.merge_head`）。
    - Down：任务认领和广播（`task_queue` 的 `first_step` / `claim_next` / `next_step` / `drain`）、LDS staging（`stage_task`）。
    - 这些 helper 含运行时 `if` / `while` / `for`，用 `@flyc.jit` 修饰：kernel trace 期间调用时，FlyDSL 会直接内联执行经 AST 改写的函数（`ir.Context.current` 非空时走 `self.func(...)`），所以动态控制流不必留在 kernel 体内。
  - **代价**：Gate/Up 主循环与最后一个 K pair 的 region 代码重复一份（两种 tile 各一份），以换取主流程可以直接读出整条流水线。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25），重新编译后 op test 全部通过（76 项 passed）；black / ruff 通过。
- 2026-10-27（续）：函数之间的衔接改成参考那样以 tensor（数据）为中心；Down 按同样的流水线风格重构。
  - **Gate/Up**：
    - 操作数 `A_t` / `A_b` / `B_l` / `B_r` 只提供三个操作：`fill_g2s` / `refill_g2s`（DMA，global → LDS）和 `read_s2r(buf)`（LR，LDS → VGPR，返回 fragment）。
    - `mma(acc, b, a)` 直接接收 fragment，例如 `mma(C.tl, b_l, a_t)`。
    - 主流程里的数据流是显式的：prologue 读出 `b_l`、`a_t`；每个 region 读出后面 region 要用的 fragment（`a_b`、`b_r`、`b_l`、`a_t`）；半高 tile 用 `a_t1` 暂存下一个 K tile 的 A_t。
  - **Down**：
    - `b_ring(...).fill_g2s(s)` 发出 B sub-packet 的 DMA；`rows_g2r(...)` 返回 A 行 `a`（VGPR）和输出地址 `out`。
    - `packet_pipeline` 改为数据进、数据出：
      - `zero_acc()` → `c`；`side_work(q, c1, ws1, out2)` → `side`（q-1 的 epilogue、q-2 的 stores）。
      - `read_s2r(q, sub)` → `(b, ws)`：memory stage。
      - `mma(c, a, b, sub, side)`：compute stage。
      - `end(c, side, ws)` → `(c1, out2, ws1)`；`epilogue(c1, ws1)` → 最后一个 packet 的打包结果；`store_r2g(packed, q)`。
    - 原来不透明的 `state` / `pk` 拆成显式的 `c1`（C(q-1)）、`out2`（q-2 的打包结果）和 `ws1`（WS(q-1)）；主循环的 `init` / `yield` 也直接传这三者。
    - 最后两个 packet 的 stores 在主流程直接写出。
  - **为什么 Down 的每个 stage 仍是一次调用**：
    - memory stage 每 4 条 LR 之间插一个 DMA turn；compute stage 每条 MFMA 之后插 epilogue 步骤和 stores。
    - 这些交错靠 IR 的发射顺序实现。把 DMA、LR、MFMA 拆成主流程里各自的调用会改变发射顺序，从而改变 ISA。
    - 所以 stage 内部的交错留在 helper 里，主流程用数据（`b`、`ws`、`side`）把两个 stage 连起来，各 stage 的内容写在注释表里。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25），重新编译后 op test 全部通过（76 项 passed）；black / ruff 通过。
- 2026-10-27（再续）：Down 的 stage 内交错改为在主流程里逐 slot 写出，helper 只保留叶子操作。
  - **原因**：Down 的调度就是程序顺序。编译器不会把 ds_read 提到可能别名的 LDS DMA 写之前，epilogue 步骤又被 `sched_barrier` 固定。所以不能像 2x2 那样分组写、再靠 `sched_group_barrier` 交错；要让主流程看到交错，就得按发射顺序写出。
  - **主流程**：kernel 内的 `packet(q, cycle, c1, out2, ws1)` 写出一个 packet，静态头部、主循环和静态尾部各调用一次。
    - memory stage：`B.wait(cur)`；`nxt = B.ahead(cur)` 是 B[s + lookahead]，DMA 进 B[s-1] 的 slot；每个 turn 先 `B.refill_g2s(nxt, t)`，再 `B.read_s2r(cur, t, b)` 读 4 条（K128 t // 2，N32 t % 2）；最后一个 sub-packet 读 `WS.read_s2r(q)`；`B.land(b, ws)`。
    - compute stage：对 packet 的第 i 条 MFMA，若 `i in store_at` 先 `O.store_r2g(out2, q - 2, store_at[i])`，再 `mma(c[mi * 4 + ni], b[...], a[...])`，然后 `epi.step()`（out(q-1) 的 per_step 个 VALU 步骤）。
    - packet 返回 `(C(q), out(q-1), WS(q))`；`c1, out2, ws1 = None, [], None` 表示还没有 q-1、q-2。
  - **helper**：`b_ring` 提供 `at` / `ahead` / `wait` / `refill_g2s` / `fill_g2s`；`packet_operands` 提供 `B.frag` / `B.read_s2r` / `B.land`、`WS.read_s2r`、`O.store_r2g`、`mma`（建在 prologue 之后，固定的 LDS 读地址要排在 A 行加载之后）；`epilogue_steps` 提供 `step` / `pack`。`_epilogue_row`、`_zero_acc`、`_pin_acc`、`_flat` / `_unflat` 移到模块级。删除 `packet_pipeline` 及其 `side_work` / `end` / `start`。
  - **ISA 不变的两处移动**：sub-packet 的 `asyncmark` 从最后 4 条 LR 之后移到最后一个 DMA turn 之后（组内只有 DMA）；`words` view 改为每次读时现建。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25），重新编译后 op test 全部通过（76 项 passed）；black / ruff 通过。
- 2026-10-27（命名）：主流程里的叶子操作改为按含义命名，数据流显式。
  - DMA 统一叫 `load_g2s`：Down 的 `refill_g2s` 和 Gate/Up 的 `refill_g2s` 都改名；prologue 的 `fill_g2s` 保留。
  - Down：`ahead` → `prefetch_of`（返回预取的 B[s + lookahead]，变量 `pf`）；`wait` → `wait_g2s`；`at` → `subpacket`。
  - Down 的 LR 改为有返回值：`b += B.read_s2r(cur, t)` 收集在途的 ds_read 结果；`b, ws = B.wait_s2r(b, ws)` 等 lgkmcnt(0)、把它们固定在 VGPR 里并组成 MFMA fragment（取代 `frag` + 原地写 + `land`）。
  - epilogue 合成一个对象：`epi = epilogue(c1, ws1)`，`epi.run(per_step)` 发出下 per_step 个 VALU 步骤（前后各一道 sched_barrier），`epi.finish()` 跑完剩余步骤并返回 4 个打包记录；最后一个 packet 用 `epilogue(c1, ws1).finish()`，逐行排空后合并，发射顺序与原 `_epilogue` 相同。`_pin_acc` → `_acc_values`。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25），重新编译后 op test 全部通过（76 项 passed）；black / ruff 通过。
- 2026-10-27（命名 2）：Down 的下标按 M / N / K 方向统一命名。
  - packet 序号 `q` → `n`（第 n 个处理的 N64 packet；它在 slice 中的位置是 `packet_index(n)`，起点按 worker 旋转）；sub-packet 序号 `sub` → `k`（packet 内沿 K 的切块）；K64 半块 `part` → `kh`；`piece` → `n16_bytes`；ring 的 `phase` → `slot`（`phases_per_base` → `slots_per_base`）；epilogue 里表示 ni 奇偶的局部变量 `half` → `odd`（`half` 只留给 M 方向的 128 行半块）。
  - pyhip 的 "turn" 改为 LDS DMA 的说法：`_TURN_BYTES` → `_DMA_BYTES`（每线程 16 B，CTA 共 4 KiB），`sub_turns` → `sub_dmas`，循环变量 `t` → `j`；`dma_turn` 并入 `load_g2s(sp, j)`；`ws_turns` → `ws_per_thread`。
  - `prefetch_of` → `prefetch_target`（只返回预取目标，不发指令）；`_Sub(q, sub, ...)` → `_SubPacket(n, k, ...)`。
  - 保留：`oc` / `oc_splits`（对外接口，host 端在用），`lr` / `lk`（MFMA lane 坐标，A 的 M 行和 B 的 N 列都用它），`half` / `halves`（M 方向）。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25）；black / ruff 通过。
- 2026-10-27（命名 3）：Gate/Up 按同样的规则改名（M / N / K 方向用 m / n / k，`n_` 不再表示个数）。
  - tile 与 job：`_Tile(block, n_tile, first, end, half)` → `_Tile(m_block, n_tile, kp0, kp1, half)`，`_Job` 的 K-pair 区间同样改为 `kp0` / `kp1`；`num_k_tiles` → `k_tiles`，`num_pairs` → `k_pairs`。
  - 调度：`n_blocks` → `m_blocks`；`n_dp` → `dp_jobs`；`n_bottom` → `bottom_jobs`；stream-K 的 `g0` / `g1` → `w0` / `w1`（拼接后 K-pair 工作量里的区间）；`u0` / `u` → `sk_tile0` / `sk_tile`；`f` / `max_f` → `ranges` / `max_ranges`（tail tile 的 K 区间数）；`chunk` → `k_range`；bottom 候选 `jb` / `jb_n` / `cur_b` / `is_b` → `bottom` / `next_bottom` / `cur_bottom` / `is_bottom`。
  - 片段下标：epilogue 和 stream-K 里的 `qm` / `rr` / `cr` / `cp` → `mh` / `mi` / `ni` / `ni0`，象限 `q` → `quad`；partial 循环变量 `k` → `j`，`n_parts` → `parts`；`head_block` / `head_n` → `head_m_block` / `head_n_tile`。
  - 操作数：`b_slot` → `b_voffsets`；组内 DMA 序号 `c` → `j`；`first_row` → `n_first`；`a_offsets` 里的 `h` / `c` → `mh` / `j`；`operand(src, ...)` 的 LDS 源改叫 `srcs`，`fill_g2s(src, ...)` 的 `src` 指 DMA 源。
  - 主流程：prologue 的 `b0` → `buf0`（和循环体内的 `buf` 不同名，避免动态 job 循环把它当 loop-carried 值）；job 序号 `it` → `i`；`kk` → `k_load`；`nb`（另一个 buffer）→ `next_buf`。
  - **验证**：25 个特化的最终 ISA 逐字节一致（identical=25）；black / ruff 通过。
- 2026-10-27（命名规则）：两个 kernel 的模块 docstring 末尾各加一节 `Naming:`，写明 M / N / K 的含义、m / n / k 表示方向、`n_` 前缀只表示 N，以及各自的操作数 / fragment 名、下标字母、数据搬运动词（`*_g2s` / `*_s2r` / `*_r2g`）和注释缩写（DMA、LR）。只改 docstring，不影响生成代码；black / ruff 通过。
- 2026-10-27（编排规则）：两个模块 docstring 各加一节 `Pipeline rules`，列出改主流程时必须保持的约束。
  - Gate/Up：region 内先 DMA、再 LR、再 MFMA；DMA group 的发出顺序和 `in_flight` 的计数（5 / 3，prologue 6 / 4 = 所读 group 之后已发出的 group 数）；LDS 写后读（DMA 至少在该 buffer 最后一次读的下一个 region，靠 region 开头的 lgkmcnt(0) + s_barrier）；VGPR 写后读（region r 读的 fragment 从 r + 1 起才用，读只能覆盖已用完的 fragment，所以半高 tile 用 a_t1）；预取距离 2 和跨 job 预取；epilogue 的 VMEM 读在最后一个 K pair 之前、store 在之后；结束时的 merge 和 drain。
  - Down：程序顺序即调度；ring 的 slot 复用和 barrier；每个 sub-packet 一个 async group 及等待计数；memory / compute stage 内的顺序与优先级；per_step 的条件；三级 packet 流水和携带量；动态循环的起止条件；pin 的位置；task 结束的 drain 与 LDS 交接。原先单独的 "Program order is the schedule" 段并入第 1 条。
  - 只改 docstring；black / ruff 通过。
- 2026-10-27（易读性整理）：按易读性审查的第 1–12 条整理两个 kernel；第 13 条（拆分 Gate/Up 的 `epilogue.store` 和 `mma_operands`）按要求未做。
  - 公共原语收进 common.py：`scalar`、`swap_pack`（不 pin 的版本；Down 在外面套 `_pin_packet`），以及路由解码 `route_token` / `split_route`（返回 token、topk_slot、是否为路由）/ `is_route`。两个 kernel 里原来手写的 `& 0xFFFFFF`、`>> 24` 和有效性判断都改为调用它们。保留两个判断函数是为了让 IR 发射顺序与原来完全一致。
  - Gate/Up 结构：
    - 主循环和最后一个 K pair 共用 kernel 内的 `k_pair(src, k_load, b_l, a_t)`，region 序列（全高 / 半高）只写一次。
    - `in_flight = 2 * groups - 3`（`groups = 2 + halves`）、prologue 等待 `2 * groups - 2`，推导写在定义旁边。
    - `tile_schedule` 的 docstring 列出一个 CTA 的 job 序列；stream-K 的规划收进 `stream_k_plan(count)`，返回 `plan`。
    - `epi.prepare` 返回具名的 `_EpilogueIn`。
    - 主循环加注释：C 在 AGPR 里，K 循环携带的是它的值。
  - Down 结构：
    - ring 几何收进纯 Python 的 `_ring_plan(K, packets)`，返回 `_RingPlan`，并把编排规则写成断言：一个 DMA 配四个 LR；ring 不超过 9 个 8 KiB slot；`per_step * 8 * KB >= 44`；主循环从 ring cycle 边界开始，且循环内每个 sub-packet 都还在预取。
    - packet 间的携带量改为 `_Carry(c1, ws1, out2)`，带 `has_out2` / `flat()` / `unflat()`。原来散在各处的哨兵值和 8/4/4 拆分现在都收在这个类里。
    - B 只构造一次，A 行加载之后显式调用 `B.pin_read_bases()`。
    - `packet()` 的默认参数加注释，说明它们绑定的是当前 task 的操作数。
    - `next_half_live` 移进使用它的 `task_queue`。
    - 魔数起名：`_N16_K16_BYTES`、`_N16_K64_BYTES`、`_N16_K128_BYTES`、`_EPILOGUE_STEPS`、`_OUT_PACKET_BYTES`。
  - 命名：
    - 两个 kernel 的 `kc` → `ctx`。
    - Gate/Up：`job` → `cursor`，`op` → `operands`，`rows` → `epi_in`，`slot_id` → `topk_slot`。
    - Down：`t` → `task`，LDS 广播字 → `claim_word`，路由 slot → `topk_slot`。
    - Gate/Up 的 `tid` / `tid_i` 加注释说明各自的用途。
  - docstring 统一为：概述 → 布局 / 输出 → 调度 → 流水线表 → `Naming` → `Pipeline rules`。表格只写"做什么"，规则只写"必须保持什么"。
    - Gate/Up：K step 段落不再重复规则 1–2；`in_flight` 的推导移到定义旁边。
    - Down：Output 移到 Task 之后，新增 Schedule 段；Stages 表写出 prio、WS(n) 和 store_at；规则 4–6 只保留不变量（DMA 与 LR 的配对、epilogue 在 packet n 内结束、`_Carry` 携带）；规则 8 改为"A 行加载之后 pin"。
  - 可选的调度单测（host 端模型，检查每个 (tile, K pair) 恰好被覆盖一次）未加。
  - **验证**：
    - 25 个特化的最终 ISA 逐字节一致（identical=25）。
    - 重新编译后 op test 全部通过（rc=0，76 项 passed，"32 four-wave, 10 M64 Gate/Up + M128 Down"）。
    - black / ruff 通过。
