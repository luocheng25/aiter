# gfx9_fp8 的 A8W4 MoE：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。
     本目录的其他 kernel（gemm1.py、gemm2*.py、quant.py、moe_reduce.py）是 FP8 / BF16 底座，不在本文范围。 -->

A8W4 的 GEMM kernel（prefill、decode 与流式的 Gate/Up、Down）共用 [common_a8w4.py](common_a8w4.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；布局上 lane `16b + r` 对应第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32
  的操作数——这是布局单位，不是读取宽度（decode、流式与 prefill Gate/Up 每 lane 一次读 16 B，见下）。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](common_a8w4.py#L64) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](common_a8w4.py#L79) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼。流式 kernel 字节相同、指令更少：
  表从各 wave 写进 LDS 的预建表按 delta 字节取（每行每步 1 条 VALU + 1 条 ds_read_b64，代替 4 条 VALU），
  一行两个 dword 的半字节移位合成一条 64 位移位（每行省 2 条）。
- decode kernel 的 routed 权重每 lane 一条 16 B（dwordx4）加载：Gate/Up（逐路由、混合与 [decode_dedup_a8w4.py](decode_dedup_a8w4.py)
  的去重版）与去重 Down 读两个相邻 64-K 块（块对），槽对 Down 读两个槽的同一块（槽对），单槽 Down 与混合 Down 的 routed 列读同一
  64-K 块的相邻两个 16 行块（行块对：每个 WG 32 个输出通道，每个通道的 MFMA 乘积与求和顺序和 8 B 单块读法相同，输出逐位不变）；
  LUT 从 LDS 里预建的表按 delta 字节取；lane 映射见 decode_a8w4.py 的 docstring（Block pairs、Slot pairs、Row-block pairs）。
- 流式 kernel 的 routed 权重也是每 lane 一条 16 B，读两个 16 行块各两行（成对行）：lane `16b + i` 读布局 lane `16b + 2(i&7)` 与 `+1`
  （i < 8 读第一块、≥ 8 读第二块，每个 quad 读连续 64 B），MFMA 链 p 取各 lane 行对的第 p 行；见两节流式 kernel。
  prefill 的 Gate/Up 每 lane 16 B 读相邻两行；prefill Down 仍每 lane 8 B（每 64 列块 13 条读对 72 条 MFMA，不是访存瓶颈）。
- 解出的值是 |e2m1| × 2^(5−delta)，delta ≤ 11 时精确；通道因子 s = 2^(cmax−132)（f32）与激活 scale 在 epilogue 一起乘。
- 这套布局与 dense A8W4 GEMM（gemm-a8w4 分支 `gemm_a8w4_ptpc/layout.py`）的 fd2 布局相同，每个 expert 一份；两边的辅助函数逐字相同。

## 先读：Gluon 风格的分阶段写法
每个 kernel 一节，先给一段 Gluon 风格的函数，一个 program = 一个 CTA。不用纯 Triton 的原因：Triton 的 `tl.load` / `tl.dot`
只描述块级张量算什么，数据经不经 LDS、留在哪些寄存器、提前几步加载、在哪里 barrier、wave 之间怎么分工，都由编译器决定，
源码里没有地方写，而这些正是这几个 kernel 的设计要点。所以每段先用注释写**放置**与**预取距离**，再按序言 / 稳态 / 收尾写出各阶段；
Gluon 没有的操作写成伪操作。指令级的细节（sched_group 排布、s_waitcnt 计数、lane 取址）在后面的逐拍流水线。
只求看懂、不求能运行：张量下标代替 offsets 运算，省略 mask、布局对象与类型转换，参数用 host 侧的张量名。共用的记号：
```python
# —— 张量 ——
# M：token 数；H：hidden；I：中间维（补零后）；E：expert 数；TOPK：每个 token 的路由数。路由 route = token·TOPK + slot。
# x / out：bf16 [M, H] 的输入与输出。xq / xs：x 按行量化后的 fp8 [M, H] 与 f32 scale [M]。
# mid：f32 [M, TOPK, I]（= swiglu(gate, up)）；mq / ms：mid 按行量化后的 fp8 [M·TOPK, I] 与 f32 [M·TOPK]。
# 排序输出（流式与路由不多的 prefill 用 stream_prep，其余 prefill 用 aiter Opus moe_sorting，格式相同；int32，权重 f32）：
#   排序块 e = 同一 expert（sorted_expert_ids[e]）的 BM 个排序行；sorted_ids[排序行] = token | slot << 24，token ≥ M 的是
#   填充行（只在每个 expert 的末块尾部）；sorted_weights[排序行] = 路由权重；num_valid = 排序行总数（含填充行）：grid 按
#   排序块数的上界发射，e·BM ≥ num_valid 的 CTA 直接退出；loc_ids[token, slot] = 该路由排序后的行号（< 0 表示没有这条
#   路由，求和时跳过；同一 token 重复选同一 expert 时 Opus 排序只保留最后一个槽、其余为 −1，stream_prep 每条路由都保留）。
#   topk_ids 里的 −1 是被屏蔽的路由（ATOM 的 padding 行）：两种排序都不放它（loc = −1），decode 各 kernel 见各节。
# w1 / w2：fp4 权重码，w1_delta / w2_delta：delta 字节；w1_scale / w2_scale [E, 通道]：通道因子 2^(cmax − 132)，收尾才乘。
# plan [M·TOPK, 4]（decode 按 expert 去重的计划）：同一 expert 的路由按路由号每 GROUP 条一组（M ≤ 16 时 2，否则 3）；
#   plan[r] = [条数, r, 同 expert 的后续路由…]，条数只在组首非 0；组首的 CTA 算全组，只读前「条数」个组员。
# —— Gluon 的真实操作（gl = triton.experimental.gluon.language，cdna3 = gl.amd.cdna3）——
# r = cdna3.buffer_load(t[下标])   global → VGPR；结果到第一次使用才等待，循环里先发后面几步的加载再算这一步就是预取
#                                  （decode_a8w4.py 读激活行用普通 global load，不查越界：越过行尾的 lane 钳到最后一片或不发；
#                                  decode_dedup_a8w4.py 与 stream_a8w4.py 用 raw buffer，越界偏移（_DROP）读到 0、写被丢弃）
# cdna3.buffer_store(v, t[下标])   VGPR → global；偏移越界（如填充行）的写被硬件丢弃
# L = gl.allocate_shared_memory(类型, [槽数, …], 布局)；L.index(i).store(r) / .load(布局)：VGPR ↔ LDS
# gl.barrier()                     CTA 内同步：之前写进 LDS 的数据对所有 wave 可见，之前读完的槽此后可以覆盖
# acc = cdna3.mfma(a, b, acc)      fp8 MFMA 16x16x32、f32 累加：acc[行, 列] += a[行, K]·b[K, 列]
# —— 伪操作 ——
# wave_id() / lane_id()：本 wave / lane 的编号；uniform(v)：readfirstlane，把 wave 内相同的值变成标量，供整 wave 分支
# wait_vmem()：等此前发出的全部 global 加载到齐（s_waitcnt vmcnt(0)）
# decode(wr, …)：寄存器里的 fp4 码 + delta 字节 → fp8 操作数（v_perm 查表）；rcp(v)：硬件近似倒数
# to_fp8(v)：RNE 转 fp8，-0 存成 +0；xcd_work(L)：XCD 分块 1-D grid 的工作编号（见 Gate/Up 一节）
# wave_max(v)：wave 内求 max（4 步 DPP + 4 次 readlane）；tree_sum(v, axis=0)：顺序固定的树形求和（前 ⌈n/2⌉ 项与其余各自递归相加）
# atomic_add(t[下标], v)：global 原子加（agent 范围、不取回；bf16 时每 lane 两个相邻元素一条 bf16x2 指令），和的顺序不定；
#   LDS 上的 atomic_add 返回旧值（stream_prep 的计数与放置）
# thread_id()：CTA 内的线程号；exclusive_cumsum(v)：不含自身的前缀和（stream_prep 的扫描）
# valid_subblocks(ids, e, num_valid, M)：排序块 e 里首行是真实行（token < M）的 16 行子块数（有效行在块首，所以是前缀）
# ballot(p)：本 wave 各 lane 的谓词拼成 64 位掩码；popcount / 最低置位用来数同 expert 的路由（去重计划）
# —— 语义 ——
# silu(g) = g · rcp(1 + exp2(−g·log2e))；swiglu(g, u) = silu(g)·u，g、u 已乘完通道因子与激活 scale；
#   SwiGLU 截断 L > 0（编译期常量，kernel 名加 _swiglu）时先 g = min(g, L)、u = clamp(u, −L, L)；L = 0 时不发指令
# 屏蔽路由（id −1）：任何 kernel 都不读它的权重、不把它计入 out（路由权重是什么都一样），整行屏蔽的 token 得 bf16 +0
# row_scale(amax) = amax / 240（IEEE 除法），结果为 0 时取 1；quant_row(v) → (q, s)：s = row_scale(max|v|)，q = to_fp8(v / s)
#   行量化都按 quant_row 逐位一致：row_quant_fp8 是通用版（bf16 或 f32 输入），row_quant_decode 是 decode 档用的版本，
#   去重的 row_quant_plan 与流式的 stream_prep 在同一次 launch 里顺带量化 x，流式 Gate/Up 可在收尾直接量化 mid
# quant_fast(v, s, k)：r = RN(1/s) 每行一次，q = RN(v·r) 后做 k 次 FMA 修正 q += (v − q·s)·r，再 clamp、to_fp8；decode 档的行量化
#   都用（bf16 的 x 一次修正：逐路由与混合 Gate/Up 的核内量化、row_quant_decode(x)、row_quant_plan；f32 的 mid 两次：单槽与混合
#   Down、row_quant_decode(mid)；去重 Down 的 QUANT 仍逐值除），fast_ok(amax)（amax ∈ [2^-60, 2^127) 或为 0）时与 quant_row 的
#   字节逐位相同，否则退回逐值除法
# sorted_sum（底座 moe_reduce.py）/ route_sum（stream_a8w4.py）：out[t, :] = Σ_slot routes[loc_ids[t, slot], :]
#   （f32 按槽顺序相加后转 bf16，不读 loc < 0 的槽）
# 调用顺序（host 见 ../../../moe_a8w4.py；给了 tile 参数时一律 prefill）：
#   decode（M ≤ 32，H ≤ 4096 或 5120，decode kernel 放得下；不排序，也没有 sorted_sum）：
#     逐路由（H < 4096、I%128 ≠ 0 或 M = 1；H5120 且 I%128 ≠ 0 时只到 M = 7）：
#       [row_quant_decode(x)：M·TOPK·I/16 ≥ 2560] → gateup_decode → [row_quant_decode(mid)：M ≥ 4、w2 加一个 expert ≤ 4 GiB、
#       槽对版放得下]
#       → down_decode(_pq) → out
#     按 expert 去重（H ≥ 4096、I%128 = 0、M ≥ 2；H4096 时 M ≤ 64 且 2·M·TOPK ≤ 3E 的 M > 32 也用；Down 原子加进清零的 out）：
#       M·I < 3072、I%256 = 0 且 out 与 x 不重叠：[row_quant_decode(x)] → gateup_decode(PLAN：写计划、清零 out)
#         → down_decode_dd(WG 内量化 mid)
#       其余：row_quant_plan(x；写计划、清零 out) → gateup_decode_dd → row_quant_decode(mid) → down_decode_dd
#     只去重 Gate/Up（H5120、I%128 ≠ 0、8 ≤ M ≤ 32、w2 加一个 expert ≤ 4 GiB；Down 不去重，逐 token 直接写 out）：
#       row_quant_plan(x；写计划、不清零 out) → gateup_decode_dd(4 wave) → row_quant_decode(mid) → down_decode_pq
#   流式（M > 32 且 M·TOPK ≤ R·E；R = 22，H ≥ 4096 为 48，H5120 且 M·TOPK ≥ 4096 为 16，H2560–4095 且排序块 64 尾部量化为 56；
#     排序块：每 expert 平均 ≤ 12 行为 16（H < 5120 还要 3·M·TOPK ≥ 4E，否则 32），≤ 28 行为 32，其余 64）：
#     stream_prep(排序 + 量化 x [+ 清零 out]) → gateup_stream（尾部量化，或另跑 row_quant_fp8(mid)）→ down_stream（I < 512：
#     原子加进 out；否则写 routes → route_sum；排序块 64 且 H ≥ 4096 时换 down_prefill）→ out
#   prefill（其余，含 decode 放不下的 M ≤ 32）：H ≥ 2560 且 M·TOPK ≤ 24576（H5120 为 12288）时 stream_prep(排序 + 量化 x、
#     不清零 out)，否则 moe_sorting → row_quant_fp8(x；顺带把 loc_ids 置 −1)；之后都是 gateup_prefill → row_quant_fp8(mid)
#     → down_prefill（写有效路由的 loc_ids）→ sorted_sum → out；H5120 且 M·TOPK ≥ 16384 时两个 GEMM 按 XCD 分块
#     （XCD_CHUNK）；I > 512 且估计 WG > 80 时 down_prefill 的激活分两段（TWO_PASS）
#   入口：会建 down_prefill（prefill 档，或流式 BM64 且 H ≥ 4096）且 BM·I > 64 KiB（BM64 即 I > 1024）时，任何 kernel 启动前报错
#   带 shared expert（expert E 在路由最后一列，见 moe_a8w4.py 的 _moe_shared）：M ≤ 4、routed 列走核内量化的逐路由 decode
#     且 TOPK ≤ 16（含 shared 列）时 gateup_decode_mixed → down_decode_mixed（shared 作末路，不另 launch、不相加；能折叠时
#     还要 M·routed topk·I ≤ DECODE_SHARED_MIXED_MAX_WORK，主干 TP4 的 M4 因此折叠）；
#     其余 M 在 decode 槽对 Down、流式、prefill M ≥ 2048 处折叠（末节；shared 写 out 的档还要 out 16 B 对齐）：上面各档的
#     kernel 以 TOPK+1 列的路由编译、跳过末列；x 量化之后 preshuffle GEMM(gate / up) → shared_mid → preshuffle GEMM(Down)
#     写 out 或 routes 末尾 M 行（M ≤ 1024 在 side stream 上、routed Gate/Up 之后发出），最后一次规约前 join；
#     再其余（单槽 / 去重 Down、prefill M < 2048、out 未对齐）按 ATOM split：per_token_quant_hip(x) → gemm_a8w8_bpreshuffle
#     (gate / up) → silu_and_mul(L) → per_token_quant_hip(mid) → gemm_a8w8_bpreshuffle(Down)；M ≤ 1024 且 out 与 x 不重叠时
#     在 side stream 上与 routed 并发（路由权重另一条 side stream），join 后 out.addcmul_(shared, bf16(权重))；否则当前流上先跑
```

## A8W4 Gate/Up prefill（[gemm1_a8w4.py](gemm1_a8w4.py#L629)）
按排序块算 mid[token, slot, :] = swiglu(x·w1_gate, x·w1_up)，f32 写出；结构同底座 `gemm1.py` 的 prefill_1x4，权重换成 mxfp4。

**Gluon 风格**
```python
# grid = (2·I/BN, 排序块数；M·TOPK ≤ E 时取 M·TOPK)，256 线程；默认 BM=64、BN=128，TILE_K = 128（M > 2048）或 256
#   （BN128·TK128 时固定每 SIMD 4 wave）；XCD_CHUNK > 0 时改用 1-D grid
# 放置：x 按 token gather，global → VGPR 暂存（A_AHEAD 份）→ LDS 双缓冲 A[2]，4 个 wave 共读；权重 global → 本 wave
#       的 VGPR 两槽，不经 LDS；累加器一直在 VGPR。预取距离：权重 2 个 tile，x 在 H ≤ 2560 时 2 个 tile（A_AHEAD = 2），否则 1 个
@gluon.jit
def a8w4_gateup_prefill(xq, xs, w1, w1_delta, w1_scale, sorted_ids, sorted_expert_ids, num_valid, mid):
    n_blk, e = gl.program_id(0), gl.program_id(1)    # n_blk：BN/2 个 gate 通道及同号 up 通道；e：排序块
    if XCD_CHUNK > 0: w = xcd_work(gl.program_id(0)); n_blk, e = w % (2 * I // BN), w // (2 * I // BN)
    if e * BM >= num_valid: return
    ids = sorted_ids[e * BM + gl.arange(0, BM)]      # 经 LDS 分给各线程；取完后 barrier，这块 LDS 让给 A[0]
    TILE_M = 有效行数向上取到 16 的倍数             # BM64；其他 BM 为 BM/2 或 BM（有效行在块首，各一份特化）
    token, slot, expert = ids[:TILE_M] & 0xFFFFFF, ids[:TILE_M] >> 24, sorted_expert_ids[e]
    ch = n_blk * (BN // 2) + 16 * wave_id() + gl.arange(0, 16)   # 本 wave 的 16 个 gate 通道；up 用同号的 I + ch
    A = gl.allocate_shared_memory(fp8, [2, BM, TILE_K], …)
    NT = H // TILE_K
    xr = [cdna3.buffer_load(xq[token, tile(0)]), None]   # gather：每线程 TILE_M·TILE_K/4096 条 16 B
    wr = [cdna3.buffer_load(w1[expert, wrow, tile(0)]), None]   # 每 lane 16 B = 相邻两行各 8 B，连同 delta
    wait_vmem()                                      # x(0)、W(0) 到齐；之后发的 W(1)、x(1) 不等
    wr[1] = cdna3.buffer_load(w1[expert, wrow, tile(1)])
    A.index(0).store(xr[0])
    if A_AHEAD == 2: xr[1] = cdna3.buffer_load(xq[token, tile(1)])
    wd = decode(wr[0], 块 0); gl.barrier()
    ce, co = gl.zeros([16, TILE_M], f32), gl.zeros([16, TILE_M], f32)       # 偶数行 / 奇数行（通道 2R、2R+1）
    for t in range(NT):
        if t + A_AHEAD < NT: xr[t % A_AHEAD] = cdna3.buffer_load(xq[token, tile(t + A_AHEAD)])
        for ki in gl.static_range(TILE_K // 64):
            b = A.index(t % 2).load(B 操作数布局)[64 * ki:64 * ki + 64, :]   # 按 K 置换视图读
            if ki > 0: wd = decode(wr[t % 2], 块 ki)
            ce = cdna3.mfma(wd.偶数行, b, ce); co = cdna3.mfma(wd.奇数行, b, co)
        if t + 2 < NT: wr[t % 2] = cdna3.buffer_load(w1[expert, wrow, tile(t + 2)])   # 本 tile 的槽已解码完
        if t + 1 < NT:
            wd = decode(wr[(t + 1) % 2], 块 0)
            A.index((t + 1) % 2).store(xr[(t + 1) % A_AHEAD])
        gl.barrier()                                 # 新槽写完才可读；本槽读完才会被 tile t+2 覆盖
    g, u = exchange(ce, co)                          # lane l 与 l^32 交换（ds_bpermute），每 lane 得 4 个相邻通道
    sx = xs[token][None, :]
    v = swiglu(g * w1_scale[expert, ch][:, None] * sx, u * w1_scale[expert, I + ch][:, None] * sx)
    cdna3.buffer_store(v.T, mid[token, slot, ch])    # f32，每 lane 4 个通道一次 16 B；填充行偏移越界被丢弃
```

**任务划分**（默认 BM64、BN128；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (2I/128, 排序块数) | x：64 gate + 同号 64 up 通道；y：1 个排序块 |
| CTA（256 线程） | 4 个 wave | 本块 TILE_M 个排序行（同一 expert）× 128 通道 × 全部 H |
| wave | 1 个 16 通道块 | gate、up 各 16 通道 × TILE_M 行 |
| lane | g = l/16、i = l%16 | gate（i<8）或 up 块的相邻两行 2(i%8)、+1 在每个 128-K 步里的第 g 个 32-K 组；token i 的累加 |
| K 步 | 128-K 步（两个 64-K 半块） | 每个半块每 wave 4·TILE_M/16 条 MFMA 16x16x32 |

**XCD 分块**（`XCD_CHUNK` = C > 0，[Gate/Up](gemm1_a8w4.py#L646)、[Down](gemm2_a8w4.py#L138)）：MI308X 把一次 launch 的 WG L 发到 XCD L % 4
（`common_a8w4.XCDS`）；连续 C 个排序块的全部 WG 落在同一 XCD，同块各 N tile 共用激活行、同 expert 的相邻块共用权重。
host 在 H ≥ 5120 且路由 ≥ 16384 时开（Gate/Up C = 1、Down C = 4）；只改哪个 WG 算哪块，结果逐位不变。
K 置换：A 的 LDS 读视图把步内 MFMA 侧的 K 64h + 16g + 8c + v 映射到存储的 32g + 16h + 8c + v，权重按同一置换取址，
所以 lane 组 g 在步内两个半块用同一个 32-K 组：每步只建一次 LUT。

**流水线**（region 表与 4 条流水线规则见 [gemm1_a8w4.py 文件头](gemm1_a8w4.py#L29)，这里只列骨架）
```text
序言  tid < BM：sorted_ids → LDS；barrier；定 TILE_M；发 x(0)、W(0)；等全部到齐；发 W(1)；x(0) → buf 0；发 x(1)（A_AHEAD = 2）；解码 W(0) 第 0 块；barrier
tile k  region 0：发 x(k+A_AHEAD)；读 buf 的第 0、1 块；解码 W(k) 第 1 块；MFMA（偶 / 奇行各一套）
        region i：读第 i+1 块；解码 W(k) 第 i+1 块；MFMA
        last：MFMA；发 W(k+2) 进 tile k 的槽；解码 W(k+1) 第 0 块；暂存的 x(k+1) → buf^1；barrier
收尾  最后 A_AHEAD 个 tile 不发 x，最后一个不写 LDS，倒数第二个不发 W(k+2)
epilogue  ds_bpermute 交换偶 / 奇行累加器 → × 通道因子 × xs → [截断] → SiLU(gate)·up → 16 B buffer store 到 mid
```
关键代码：[排序行号进 LDS](gemm1_a8w4.py#L680)、[半块分支](gemm1_a8w4.py#L727)、[K 置换读视图](gemm1_a8w4.py#L284)、
[lane 取址](gemm1_a8w4.py#L343)、[权重加载](gemm1_a8w4.py#L396)、[解码与 LUT 复用](gemm1_a8w4.py#L423)、
[指令排布](gemm1_a8w4.py#L463)、[一拍](gemm1_a8w4.py#L490)、[序言](gemm1_a8w4.py#L540)、[主循环](gemm1_a8w4.py#L552)、
[反量化与截断](gemm1_a8w4.py#L566)、[SiLU 与写出](gemm1_a8w4.py#L128)、[1-D grid](gemm1_a8w4.py#L773)。

**资源**：LDS 2·BM·TILE_K（BM64 时 TILE_K128 为 16 KiB、TILE_K256 为 32 KiB，排序行号的 1 KiB 与之复用）。

## A8W4 Down prefill（[gemm2_a8w4.py](gemm2_a8w4.py#L118)）
按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同底座 `gemm2.py` 的默认 prefill Down。
loc 的列数是 LOC_TOPK（默认 TOPK；折叠 shared expert 时多一列，见末节，本 kernel 不写那一列）。

**Gluon 风格**
```python
# grid = (NSPLIT, 排序块数)，256 线程；NSPLIT = 1 / 2 / 4 个 WG 分 H（各自重新 gather 本块激活）；I ≥ 256 时关 packed FP32
# 放置：mq 的 BM 行按路由 gather，global → VGPR → LDS，再整块（bm × I）读回 VGPR 常驻到结束，之后同一块 LDS 改作输出双缓冲
#       C[2]。TWO_PASS（I > 512 且 BM·I > 32 KiB）分两段 gather 同一块 LDS（I576：5 + 4 块）。权重按 64 列块
#       global → VGPR（两套轮换），预取距离 1 块；结果经 C 转成按行连续，再 16 B 合并写出
@gluon.jit
def a8w4_down_prefill(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                      routes, loc):
    part, e = gl.program_id(0), gl.program_id(1)    # 一个排序块 × 第 part 段 H/NSPLIT 列
    if XCD_CHUNK > 0: w = xcd_work(gl.program_id(0)); part, e = w % NSPLIT, w // NSPLIT
    if e * BM >= num_valid: return
    ids = sorted_ids[e * BM + gl.arange(0, BM)]
    route = (ids & 0xFFFFFF) * TOPK + (ids >> 24)
    L.store(cdna3.buffer_load(mq[route, 第一段])); gl.barrier()        # 全 CTA 一起 gather，每次 16 B
    bm = BM // 2 if uniform(ids[BM // 2] & 0xFFFFFF) >= M else BM   # 后一半全是填充：只算前一半（两份特化）
    b = [L.slice(0, bm).load(B 操作数布局)]          # TWO_PASS：barrier → gather 第二段 → barrier → 再读
    gl.barrier()                                     # L 读完才能改作 C
    row_scale = ms[route[:bm]] * sorted_weights[e * BM + gl.arange(0, bm)]
    for n in range(H // 64 // NSPLIT):               # 稳态：算第 n 块时，第 n+1 块的权重在路上，上一块从 C 写出
        acc = 全部 K 的 MFMA(decode(w2[expert, 本 wave 16 列, :]), b)
        C.index(n % 2).store((acc * w2_scale[expert, 列] * row_scale).to(bf16).T); gl.barrier()
        cdna3.buffer_store(C.index(n % 2).load(按行连续), routes[e * BM + 行, 列])
    if part == 0:                                    # 主循环后才写（写在开头整体慢约 30%）
        cdna3.buffer_store(e * BM + gl.arange(0, BM), loc[route], mask=(ids & 0xFFFFFF) < M)
```

**任务划分**（默认 BM64；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (NSPLIT, 排序块数) | 1 个排序块 × H/NSPLIT 列 |
| CTA（256 线程） | 4 个 wave | bm 行 mq 整块驻留；逐个 64 列块走完本段 |
| wave | 每个 64 列块中的 16 列 | 列 n·64 + 16w … +15 × bm 行 |
| lane | l = 16b + r | 权重列 r 的 K 16b…16b+15；B 操作数的 1/64 |
| K 步 | I/64 个 64-K 块 × 2 个 K32 | 每个列块一次算完全部 K |

NSPLIT 由 host 选（`_down_split`）：I_pad ≥ 256 时，排序块数·NSPLIT < 1000 就加倍，至多 4。TWO_PASS 只改激活进 VGPR 的方式，
MFMA 的次序与操作数都不变，结果与一段逐位相同。通道因子按 (N, 16) 的 stride-0 视图取（FlyDSL ≥ 0.5.1 要求片段跨整个 N tile）。

**流水线**（权重 VGPR 两槽，预取距离 1 个列块；输出 LDS 双缓冲 C[0] / C[1]）
```text
序言  gather mq 的 BM 行 → LDS；barrier；readfirstlane 判半块定 bm；前 bm 行 → VGPR（TWO_PASS 再 gather 第二段）；barrier
      行 scale = ms × 路由权重；load_block(0)；compute(槽 0)；load_block(1)；× 行 scale → bf16 → C[0]；barrier
主循环（每次两拍）  拍 0：C[0] → routes 第 n 块；load_block(n+2)；compute(槽 1) → C[1]；barrier
                    拍 1：C[1] → routes 第 n+1 块；load_block(n+3)；compute(槽 0) → C[0]；barrier
收尾  C[0] → 倒数第二块；compute(槽 1) → C[1]；barrier；C[1] → 最后一块；part 0 写有效路由的 loc
compute  每个 64-K 块：delta 字节 → LUT → 16 个 fp8；MFMA × bm/8；× 通道因子；bf16 = RNE 位运算 + 一条 v_perm 拼两值
```
关键代码：[gather](gemm2_a8w4.py#L194)、[半块分支](gemm2_a8w4.py#L437)、[通道因子视图](gemm2_a8w4.py#L223)、
[激活进 VGPR 与第二段](gemm2_a8w4.py#L262)、[行 scale](gemm2_a8w4.py#L295)、[load_block](gemm2_a8w4.py#L333)、
[compute](gemm2_a8w4.py#L363)、[bf16 打包](gemm2_a8w4.py#L108)、[写 LDS](gemm2_a8w4.py#L390)、[写 routes](gemm2_a8w4.py#L403)、
[主循环](gemm2_a8w4.py#L414)、[两段的划分](gemm2_a8w4.py#L79)、[1-D grid](gemm2_a8w4.py#L487)。

**资源**：LDS = max(64·I B 激活, 2 × 64 × 64 × 2 B 输出)；TWO_PASS 时激活只占较大的一段（I576：20 KiB，不分段 36 KiB）。
64·I 须 ≤ 64 KiB 且每段 64·列数是 4096 的倍数，所以 BM32 配 I320 / 192 不可用，I_pad > 1024 时 prefill 不可用。

## 行量化 prefill（[quant_a8w4.py](quant_a8w4.py#L76)）
把 bf16 输入 [M, H] 与 f32 中间 [M·topk, I] 按行量化成 fp8 与每行 scale，结果与参考（IEEE 除法）逐位一致。

**Gluon 风格**
```python
# grid = ⌈rows / (256/L)⌉，256 线程：每行 L 个相邻 lane；不用 LDS，整行一次 global → VGPR，行内归约走 lane 间 shuffle
@gluon.jit
def row_quant_fp8(x, q, scale, loc, rows):          # x：bf16 [M, H]（输入）或 f32 [M·TOPK, I]（mid）
    r = gl.program_id(0) * (256 // L) + gl.arange(0, 256 // L)   # 行号 ≥ rows 的读到 0、不写
    if FILL: cdna3.buffer_store(-1, loc[r, 0:FILL])  # 输入量化顺带把 loc_ids 置 −1
    v = cdna3.buffer_load(x[r, :])                   # 16 B 片按 lane 交错：lane l 取片 l、l+L、…
    s = row_scale(gl.max(gl.abs(v), axis=1))         # maxnum（输入须有限），再 log2(L) 次 shuffle_xor
    if s >= 2 ** -64: y = 1 / s; qv = v * y; qv += (v - qv * s) * y; qv += (v - qv * s) * y   # 正确舍入的商
    else: qv = v / s                                 # 残差可能下溢：逐元素 IEEE 除法
    cdna3.buffer_store(to_fp8(qv), q[r, :]); cdna3.buffer_store(s, scale[r])
```
lane 数 L 从 64 起减半，直到整除每行的 16 B 片数且每 lane 至少 4 片（H2560 bf16：L64、每 lane 5 片；I320 / I192 f32：L16 / L8）。
关键代码：[lane 数](quant_a8w4.py#L36)、[16 B 读](quant_a8w4.py#L121)、[amax 与 scale](quant_a8w4.py#L136)、
[倒数与修正](quant_a8w4.py#L162)、[打包与 -0](quant_a8w4.py#L65)、[写回](quant_a8w4.py#L146)。

**资源**：不用 LDS。

## A8W4 Gate/Up decode（[decode_a8w4.py](decode_a8w4.py#L290)）
逐路由 decode（M ≤ 32、形状受支持：H%512 = 0，见 [gateup_decode_fits](decode_a8w4.py#L96)；host 只放行 H ≤ 4096 与 H5120）不排序：
每个 WG 算一条路由 (token, slot) 的 16 个 gate 与同号 16 个 up 通道，写 mid[token, slot, ch]（f32）。
去重的两 kernel 路径也用它（编译期 `PLAN`：顺带写去重计划并清零 out，给后面的 down_decode_dd）；不带 PLAN 时 plan / y 两个参数位不读。

**Gluon 风格**
```python
# grid = (I/16, M·TOPK)，256 线程（4 个 wave）；不排序
# 放置：x 的一行（H 个 fp8）放 LDS，4 个 wave 共读（A 操作数是广播读）；LUT 表（delta 0–11 各 8 B）放 LDS；权重 global →
#       本 wave 的 VGPR：序言先发前 P 个块对（核内量化版 P = min(块对数, GATEUP_PREFETCH_PAIRS = 6)，预量化版 P = 全部），
#       主循环每消耗一个块对补发一个（预取距离 P − 1），不分 tile、无双缓冲，第 p 个块对只等自己那几条加载；
#       4 个 wave 的部分和经 LDS 相加
@gluon.jit
def a8w4_gateup_decode(x, xq, xs, topk_ids, w1, w1_delta, w1_scale, mid, plan, out):
    j, route = gl.program_id(0), gl.program_id(1)    # j：16 个 gate 通道及同号 up 通道；route：一条路由
    token, slot, expert = route // TOPK, route % TOPK, topk_ids[route]
    w = wave_id()                                    # w%2：gate / up；w//2：K 前半 / 后半
    ch = 16 * j + gl.arange(0, 16)                   # 本 CTA 的 16 个 gate 通道；up 用同号的 I + ch
    wrows = (w % 2) * I + ch                         # 本 wave 的 16 个权重行
    kb0 = (w // 2) * (H // 128)                      # 本 wave 那半段 K 的第一个 64-K 块
    lut = gl.allocate_shared_memory(u8, [96], …); lut.store(LUT 表)
    xl = gl.allocate_shared_memory(fp8, [H], …)
    # 序言：x 行不依赖 expert，最先发（预量化版线程 t 读 16 B 片 t + 256u，H5120 分两轮）；再发前 P 个块对
    xr = cdna3.buffer_load(xq[token, :] if PREQUANT else x[token, :])
    wr = [cdna3.buffer_load(w1[expert, wrows, 64 * (kb0 + 2 * p):64 * (kb0 + 2 * p + 2)])   # 块对 p：每 lane 16 B；
          for p in range(P)]                 # p 为偶数时先发一条 w1_delta 的 8 B（两行 × 4 个 64-K 块，供块对 p、p+1）
    cs = w1_scale[expert, ch], w1_scale[expert, I + ch]   # 收尾的通道因子也在序言读（tid < 16），收尾不再等访存
    # 屏蔽路由（expert = −1）：w1 / w1_delta 的 buffer 视图记录数取 0（越界检查连 SGPR 偏移一起算 → 读到 0、不访存），
    #   cs 不读（0），于是 mid 行为 0；id 零扩展后才乘字节数（符号扩展会让编译器在 id 的 s_load 后立刻等它，挡住 x 行加载）
    if PREQUANT:                                     # 编译期开关：M·TOPK·I/16（CTA 数）≥ 2560 时 host 先用 row_quant_decode 量化 x
        xl.store(xr); s = xs[token]
    else:                                            # CTA 少时每个 CTA 自己量化 bf16 的 x 行，省一次 launch
        red = gl.allocate_shared_memory(f32, [4], …)
        red.index(w).store(wave_max(gl.abs(xr))); gl.barrier()   # 每个 wave 只持有 x 行的一部分；barrier 保护 4 个 wave 的 max
        amax = gl.max(red.load(…)); s = row_scale(amax)
        xl.store(quant_fast(xr, s, 1) if fast_ok(amax) else to_fp8(gl.clamp(xr / s, -240, 240)))
    gl.barrier()                                     # x 行写完 LDS 才可读
    # 主循环：本 wave 那半段 K，共 H/256 个块对
    acc = gl.zeros([2, 16, 16], f32)                 # h = 0/1：lane 持有的两个权重行
    for p in range(H // 256):
        if 0 < p and p + P - 1 < H // 256:           # 补发：块对 p−1 的寄存器刚用完
            wr.append(cdna3.buffer_load(w1[expert, wrows, 块对 p + P − 1]))
        a = xl.load(A 操作数布局)[64 * (kb0 + 2 * p):64 * (kb0 + 2 * p + 2)]   # A 的 16 行都取自这个 token 的 x，
                                                     #   但行 0–7 与行 8–15 取的 K 组不同（见收尾）
        for h in gl.static_range(2):
            acc[h] = cdna3.mfma(a, decode(wr[p], h, lut), acc[h])   # LUT 按 delta 字节从 LDS 取
    # 收尾：输出 (行 m, 列 n) 只在 m//8 == n//8 时 A 与权重的 K 组对齐，所以取行 0 的列 0–7 与行 8 的列 8–15，
    #   各含一半 K 组；列 m 与 m+8 相加得通道 2m+h 的全 K（见代码 docstring “Block pairs”）。4 个 wave 的部分和经 LDS 相加
    part = gl.allocate_shared_memory(f32, [4, 2, 16], …)
    part.index(w).store(acc 的行 0 列 0–7 与行 8 列 8–15); gl.barrier()   # barrier：part 写完才读
    g, u = part[0] + part[2], part[1] + part[3]      # gate = wave 0 + 2，up = wave 1 + 3（各自先合并列 m 与 m+8）
    mid[token, slot, ch] = swiglu(g * cs[0] * s, u * cs[1] * s)   # f32；tid < 16 写
    if PLAN:                                         # 编译期开关（去重的两 kernel 路径）：每个 CTA 清零 out 里本 token 行的一段，
        out[token, 本 CTA 的一段] = 0                 #   通道块 0 的 wave 0 写本路由的计划条目（算法同 row_quant_plan）；
        if j == 0 and w == 0: plan[route] = plan_route(route)   #   out 与 x 重叠时 host 不走这条路（本 kernel 还在读 x）
```

**任务划分**（数字按 H2560、I320）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·topk) | x：16 个 gate 通道块 j 与同号 up 块；y：一条路由 |
| CTA（256 线程） | 4 个 wave | 该路由 expert 的 32 行权重 × 全部 H |
| wave | w%2：gate / up；w/2：K 前半 / 后半 | 16 行 × H/2（10 个块对） |
| lane | l | 块对中第 l/32 块的行 2(l%8)、2(l%8)+1，K 组 (l%32)/8，共 16 B |
| K 步 | 块对（相邻两个 64-K 块） | 每 wave 4 条 MFMA 16x16x32（2 行 × 2 个 K32） |

**流水线**（不分 tile：序言发出前 P 个块对，主循环按序消费、每步补发一个；无双缓冲）
```text
序言
  tid ≤ 11：LUT 表 → LDS lut（delta d 在字节 8d：|e2m1| 0–3、4–7 的 fp8 各一个 dword）
  expert = topk_ids[route]；本 wave 的 16 行块 rb = (w%2)·I/16 + j 与 K 起点 kb0
  发 x 行加载：预量化版线程 t 读片 t + 256u（片号 < H/16；H5120 两轮，第二轮只有 t < 64）各 16 B fp8；核内量化版每线程读 H/512 个 bf16 对
  for p < P（sched_barrier 固定顺序）：p 偶时 delta dwordx2（两行 × 4 块）；权重 dwordx4
  tid < 16：通道因子 ws[gate][ch]、ws[up][ch]
x 行 → LDS xq
  预量化版：各片 16 B → LDS；barrier；scale = x_scale[token]
  核内量化版：lane 内 max|x| → DPP 4 步 + readlane 4 行 → LDS red[wave]；barrier
              scale = max(red) / 240（IEEE，0 取 1）；fast_ok 时 q = quant_fast(x, scale, 1)，否则 RNE(clamp(x/scale))；
              两个 bf16 对共用一个 cvt_pk_fp8 dword、-0 修正一次管 4 字节 → 每对 2 B → LDS；barrier
主循环 p = 0 … H/256 − 1（第 p 个块对只等到自己的加载）
  p ≥ 1 且 p + P − 1 < H/256：先补发块对 p + P − 1（偶数时连同 delta）
  xop = LDS xq 第 kb0+2p+l/32 块、K 组 (l%32)/8 的 16 B（2 个 K32 的 A 操作数）
  每行 h：LUT 对 = LDS lut[delta 字节]；fp4 经 64 位移位 + v_perm 查表 → 2 个 fp8 i64
  MFMA 16x16x32 × 4：acc[h] += xop[c] · w[h][c]（c = 0, 1）
收尾
  lane 0–7（行 0 列 0–7）、lane 40–47（行 8 列 8–15）：acc[h][0] → LDS part[wave, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8；gate = wave 0 + wave 2，up = wave 1 + wave 3
  × 通道因子 × scale → [截断] → gate · rcp(1 + exp2(−gate·log2e)) · up → f32 写 mid[route, j·16 + tid]
  [PLAN] tid < ZPW：16 B 零写 out[token] 的第 (slot·I/16 + j)·ZPW + tid 片；通道块 0 的 wave 0：plan_route(route) → plan[route]
```
关键代码：[LUT 表](decode_a8w4.py#L220)、[x 行加载](decode_a8w4.py#L376)、[权重加载](decode_a8w4.py#L404)、[片写 LDS](decode_a8w4.py#L450)、
[核内量化](decode_a8w4.py#L459)、[主循环](decode_a8w4.py#L484)、[块解码](decode_a8w4.py#L256)、[收尾](decode_a8w4.py#L502)、
[计划与清零](decode_a8w4.py#L535)。

**命名**：rb 本 wave 的 16 行权重块号；kb0 本 wave 那半段 K 的首个 64-K 块；块对 p = 64-K 块 kb0+2p、kb0+2p+1；raws[p] 块对 p 的
16 B，deltas[p/2] 块对 p、p+1 共用的 delta 8 B；accs[h] 行 2m+h（h = 0/1）的累加器；ZPW 每个 CTA 清零的 16 B 片数
（一个 token 行由它的 TOPK·I/16 个 CTA 分着清零）。

**流水线不变量**
1. x 行加载先于全部权重加载发出（它不依赖 expert），每条 delta 加载紧挨它覆盖的两个块对之前（sched_barrier 固定），第 p 个块对只等自己那几条；
   核内量化版序言只发前 P 个块对，其余在主循环里按序补发（同一时刻在途的块对 ≤ P，寄存器随之减少、每 SIMD 能多驻留 wave）。
2. 通道因子在序言由 tid < 16 读入（屏蔽路由不读），收尾不再访存。
3. LDS xq 写完后有一次 barrier 才进主循环；核内量化版另有一次 barrier 保护 4 个 wave 的 max（red）。
4. 只有 lane 0–7、40–47 的 acc[h][0] 有效（A 与 B 的 K 组对齐的输出），写 part 后 barrier，tid < 16 才合并。
5. PLAN 版清零 out 时同 token 的其他 CTA 可能还在读 x：out 与 x 重叠时 host 改走去重的四个 kernel。
6. 屏蔽路由不读权重、delta 与通道因子（0 字节视图、跳过），mid 行为 0；PLAN 版照样清零 out 并写它的计划条目（条数 0）。

**资源**：LDS 约 3.2 KiB（xq H B、part 512 B、red 16 B、LUT 96 B；H5120 多 2.5 KiB）；同时在途的权重：核内量化版至多 6 个块对
（每 lane 6 条 16 B；H5120 87 VGPR、每 SIMD 5 个 wave，全部先发时是 153 VGPR、3 个 wave），预量化版全部（H2560 10 条、H5120 20 条）。

## A8W4 Down decode（核内量化，[decode_a8w4.py](decode_a8w4.py#L621)）
M ≤ 3、w2 加一个 expert 超过 4 GiB 或槽对版放不下时用：每个 WG 算一个 token 的 32 个输出通道（相邻两个 16 行块，行块对），
在 LDS 里把 topk 个 expert 相加后直接写 bf16 out。它的 LDS 放不下时（[down_decode_fits](decode_a8w4.py#L102)）decode 整体不用，M ≤ 32 走 prefill。

**Gluon 风格**
```python
# grid = (H/32, M)，64·W 线程：W = TOPK 个 wave（TOPK > 16 时 W = 4），R = ⌈TOPK/W⌉ 轮
# 放置：wave w 管槽 w, w+W, …：mid 行（f32）global → VGPR（lane 交错读列 l + 64u），wave 内量化后写 LDS mq[w, r]，
#       再按 A 操作数布局读回（经 LDS 是为了换布局，只有本 wave 读）；权重、通道因子与路由权重 global → VGPR：
#       各轮的 64-K 块展平后序言先发前 DOWN_PREFETCH_BLOCKS = 3 块，主循环每消耗一块补发一块；LUT 在 LDS。
#       W 个 wave 的部分和经 LDS 相加后直接写 bf16 out
@gluon.jit
def a8w4_down_decode(mid, topk_ids, topk_weight, w2, w2_delta, w2_scale, out):
    j, token = gl.program_id(0), gl.program_id(1)    # j：16 行块 2j、2j+1（32 个输出通道）
    w, l = wave_id(), lane_id()
    blk, rp = (l % 16) // 8, l % 8                   # lane 16g + n：行块 2j + n//8 的行 2(n%8)、2(n%8)+1，K 组 g = l//16
    slot = [min(w + W * r, TOPK - 1) for r in range(R)]    # 越界槽钳到 TOPK−1，贡献置零
    expert = [topk_ids[token, q] for q in slot]
    tw = [topk_weight[token, q] for q in slot]       # 路由权重，与 expert 一起最先读
    lut = gl.allocate_shared_memory(u8, [96], …); lut.store(LUT 表)
    mq_l = gl.allocate_shared_memory(fp8, [W, R, I], …)
    # 序言：mid 行先发（不依赖 expert），再发前 3 个块（每 4 个 64-K 块一条 delta，排在其前）与通道因子
    mr = [cdna3.buffer_load(mid[token, q, :]) for q in slot]   # lane 读 I/64 个 f32（列 l + 64u）
    wr = [cdna3.buffer_load(w2[expert[r], 行块 2j+blk 的行 2rp、2rp+1, 块 kb 的 K 组 g])   # 每 lane 16 B = 两行各 8 B；
          for (r, kb) in 展平后的前 3 块]          # delta 每 4 块一条 8 B（两行各一个 dword）
    cs = [cdna3.buffer_load(w2_scale[expert[r], 行块 2j+blk 的行 2rp、2rp+1]) for r in range(R)]   # 两行的通道因子，8 B
    # expert = −1（屏蔽）：三者都经记录数为 0 的 buffer 视图读到 0、不访存（同 Gate/Up decode）
    sa = []
    for r in range(R):                               # wave 内量化本槽的 mid 行
        amax = wave_max(gl.abs(mr[r])); sa.append(row_scale(amax))
        mq_l.index(w).index(r).store(quant_fast(mr[r], sa[r], 2) if fast_ok(amax) else to_fp8(gl.clamp(mr[r] / sa[r], -240, 240)))
    gl.barrier()                                     # mq 写完才读
    total = gl.zeros([2], f32)                       # h = 0/1：lane 持有的两行
    for r in range(R):
        acc = gl.zeros([2, 16, 16], f32)             # A 的 16 行都是本槽的 mq（广播），结果取行 0
        for kb in gl.static_range(I // 64):          # 每个 64-K 块 4 条 MFMA（2 行 × 2 个 K32）
            补发展平序列里的下一块（若还有）
            a = mq_l.index(w).index(r).load(A 操作数布局)[:, 64 * kb:64 * kb + 64]
            for h in gl.static_range(2):
                acc[h] = cdna3.mfma(a, decode(wr[r][kb], h, lut), acc[h])   # 列 n 是行块 2j+n//8 的行 2(n%8)+h
        for h in gl.static_range(2):
            total[h] += gl.where(w + W * r < TOPK and expert[r] >= 0, acc[h][0] * cs[r][h] * sa[r] * tw[r], 0)   # select，不乘 0
    part = gl.allocate_shared_memory(f32, [W, 32], …)
    part.index(w).store(total 按通道 16·blk + 2rp + h 排开); gl.barrier()   # barrier：W 个 wave 的部分和写完才合并
    out[token, 32 * j + gl.arange(0, 32)] = tree_sum(part.load(…), axis=0).to(bf16)   # RTE；直接写 out，不需清零，也没有 sorted_sum
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/32, M) | x：两个 16 行通道块 2j、2j+1；y：一个 token |
| CTA（64·W 线程） | W = topk 个 wave（topk > 16 时 W = 4） | 该 token 的全部 topk 个槽 |
| wave | 槽 w, w+W, …（越界槽钳到 topk−1、贡献置零） | 该槽 expert 的 32 行 × 全部 I |
| lane | l = 16g + n | 行块 2j + n/8 的行 2(n%8)、2(n%8)+1，K 组 g，每块 16 B；mid 列 l + 64u |
| K 步 | 64-K 块 | 4 条 MFMA 16x16x32（I320 为 5 块） |

行块对的乘积与 8 B 单块读法相同：lane 16g + n 在 MFMA 里是 lane 组 g、列 n，B 是行 2(n%8)+h 的 K 组 g，A 是同一 K 组的广播激活，
所以 MFMA (kb, c) 的列 n 恰是单块读法里该行那一列的 32 个乘积；各行的累加链（kb 递增、c = 0, 1）也相同，输出逐位不变。

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；每轮 r：slot = w + W·r、expert、路由权重
  发 mid 行加载：lane 读 I/64 个 f32（列 l + 64u）
  发权重加载：各轮的块展平后前 3 块，每块 1 条 dwordx4（两个行块各两行），每 4 块 1 条 delta dwordx2（排在其前）；每轮两行的通道因子 dwordx2
wave 内量化本槽 mid 行
  amax：lane 内 → DPP + readlane；scale = amax / 240（IEEE，0 取 1）
  fast_ok 时 q = quant_fast(mid, scale, 2)，否则 RNE(clamp(mid/scale)) → 逐字节写 LDS mq[w, r]；barrier
主循环 r、kb = 0 … I/64 − 1
  先补发展平序列里的下一块；xop = LDS mq[w, r] 第 kb 块、K 组 l/16 的 16 B
  每行 h：LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64；MFMA 16x16x32 × 4：acc[h] += xop[c] · w[h][c]
  total[h] += 有效 ? acc[h] 行 0 × 通道因子[h] × scale × 路由权重 : 0（lane < 16 的列 n 即通道 16·(n/8) + 2(n%8) + h）
收尾
  lane < 16：total[h] → LDS part[w, 16·(l/8) + 2(l%8) + h]；barrier
  tid < 32：W 个 wave 的部分和树形相加 → bf16（RTE）→ out[token, 32j + tid]（不需清零与 sorted_sum）
```
关键代码：[行块对偏移](decode_a8w4.py#L585)、[加载](decode_a8w4.py#L692)、[wave 内量化](decode_a8w4.py#L734)、
[主循环](decode_a8w4.py#L746)、[收尾](decode_a8w4.py#L764)。

**命名与不变量**：slots[r] / valid[r] 第 r 轮的槽（钳过）与是否有效；mids[r] 本 lane 的 I/64 个 mid 值；raws[r][i] 第 i 个 64-K 块的 16 B；
PREFETCH = min(R·I/64, DOWN_PREFETCH_BLOCKS)。
1. mid 行加载先于权重加载发出；每条 delta 加载紧挨它覆盖的 4 个块之前；在途的块 ≤ PREFETCH，其余按展平顺序在主循环里补发。
2. 越界槽钳到 topk−1（读的都是合法地址），贡献由 valid 置零；屏蔽槽（id −1）的权重读到 0，贡献同样由 select 置零
   （它的 mid 行可能是任意值、路由权重可能是 NaN）；每个 wave 只读自己写的 mq 行，barrier 后才读。
3. 部分和按固定顺序 tree_sum：同输入的输出逐位确定，且与 8 B 单块读法逐位相同（每通道的乘积、累加链与 select 都不变）。

**资源**：LDS = W·⌈topk/W⌉·I B（mq）+ 128·W B（part）+ 96 B，TP2 topk11 约 4.9 KiB；VGPR：DS TP4 60（单块读法 56）、TP2 58。

## A8W4 Down decode 槽对（预量化，[decode_a8w4.py](decode_a8w4.py#L815)）
M ≥ 4、w2 加一个 expert ≤ 4 GiB 且放得下（每行 I/16 个 16 B 片 ≤ 64·W 个线程，全零行由每线程写一片；LDS 够）时用（H5120 只去重 Gate/Up 的
路径也用它）：读预量化的 mq fp8 与 ms；一条 16 B 加载覆盖两个路由槽的同一 64-K 块，其余同核内量化版。

**Gluon 风格**
```python
# grid = (H/16, M)，64·W 线程：W = min(⌈TOPK/2⌉, 3) 个 wave，R = ⌈⌈TOPK/2⌉/W⌉ 轮，每轮一个槽对
# 放置：mq 片 global → VGPR → LDS，LUT 在 LDS，mq 已由前一个 row_quant_decode 量化好；权重直进 VGPR：各轮的 64-K 块展平后
#       序言先发前 DOWN_PQ_PREFETCH_BLOCKS = 6 块，主循环每消耗一块补发一块，通道因子在序言发出；一个 wave 同时算一个槽对：
#       lane 0–31 读低槽、32–63 读高槽 expert 的同一 64-K 块，一条 16 B 加载、一组 MFMA 推进两个槽
@gluon.jit
def a8w4_down_decode_pq(mq, ms, topk_ids, topk_weight, w2, w2_delta, w2_scale, out):
    j, token = gl.program_id(0), gl.program_id(1)
    w, half, n = wave_id(), lane_id() // 32, 16 * j + gl.arange(0, 16)
    slot = [[min(2 * (w + W * r) + s, TOPK - 1) for s in (0, 1)] for r in range(R)]   # 越界槽钳到 TOPK−1、贡献置零
    expert = [[uniform(topk_ids[token, q]) for q in pr] for pr in slot]
    tw = [[topk_weight[token, q] for q in pr] for pr in slot]                   # 两槽各自的路由权重
    lut = gl.allocate_shared_memory(u8, [96], …); lut.store(LUT 表)
    xl = gl.allocate_shared_memory(fp8, [W, R, 2, I], …)
    zero = gl.allocate_shared_memory(fp8, [I], …); zero.store(0)   # 被屏蔽 lane 的 A 操作数
    # 序言：mq 片先发（不依赖 expert），再发权重；权重按整张量寻址，lane 偏移 = 本半槽 expert × 每 expert 字节数（32 位）
    xr = [cdna3.buffer_load(mq[token * TOPK + slot[r][half], :]) for r in range(R)]   # lane l 读本半槽那一行的
    #   第 l%32 + 32u 个 16 B 片（u = 0 … ⌈I/512⌉−1；越界片钳到最后一片，只写有效片）
    sm = [[ms[token * TOPK + q] for q in pr] for pr in slot]                     # 两槽各自的行 scale
    wr = [cdna3.buffer_load(w2[expert[r][half], n, 块 kb]) for (r, kb) in 展平后的前 6 块]   # 每块一条 16 B；每 4 块一条 delta 8 B
    cs = [[cdna3.buffer_load(w2_scale[e, n]) for e in pr] for pr in expert]    # 两槽各自的通道因子
    for r in range(R): xl.index(w).index(r).index(half).store(xr[r])
    gl.barrier()                                     # mq 片与零行写完才读
    # A 的行 0–7 与行 8–15 取的 K 组不同（同 Gate/Up decode 的块对），输出只取行 0/1 的列 0–7 与行 8/9 的列 8–15；
    #   列 m 与 m+8 各含一半 K 组，合成输出通道 16j + 2m + h（细节见 docstring “Slot pairs”）
    vals = gl.zeros([2, 16], f32)                    # vals[h][列]：两槽合并后的部分和
    for r in range(R):
        acc = gl.zeros([2, 16, 16], f32)             # h = 0/1：lane 持有的两个权重行
        for kb in gl.static_range(I // 64):
            keep = lane_id() % 2 == half             # 其余 lane 的 A 取零行：输出行 0/8 只含低槽、行 1/9 只含高槽
            src = xl.index(w).index(r).index(half) if keep else zero   # 按 lane 选地址，只读一次
            a = src.load(A 操作数布局)[:, 64 * kb:64 * kb + 64]
            for h in gl.static_range(2):
                acc[h] = cdna3.mfma(a, decode(wr[r], kb, h, lut), acc[h])
        for h in gl.static_range(2):
            for s in gl.static_range(2):             # acc[h] 的行 s 与 8+s 属于槽 s；两槽的因子不同，分开乘
                ok = 2 * (w + W * r) + s < TOPK
                ok = ok and expert[r][s] >= 0           # 屏蔽槽：select 置零
                vals[h] += gl.where(ok, acc[h][行 s 与 8+s] * cs[r][s][本列的通道] * sm[r][s] * tw[r][s], 0)
    part = gl.allocate_shared_memory(f32, [W, 2, 16], …)
    part.index(w).store(vals 的行 0/1 列 0–7 与行 8/9 列 8–15); gl.barrier()
    y = tree_sum(part[:, h, m] + part[:, h, m + 8])
    if SHARED: y += sh.to(f32)    # sh = out[token, n]：序言里与权重一起发出的读，out 先由折叠的 shared 写好（见末节）
    out[token, 16 * j + 2 * m + h] = y.to(bf16)   # RTE，同 a8w4_down_decode
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/16, M) | 同核内量化版 |
| CTA（64·W 线程） | W = min(⌈topk/2⌉, 3)；topk11 为 3 wave × 2 轮 | 该 token 的全部槽对 |
| wave | 槽对 p = w + W·r：低槽 2p、高槽 2p+1 | 越界槽钳到 topk−1、贡献置零 |
| lane | l：0–31 低槽、32–63 高槽 | 本槽块的行 2(l%8)、2(l%8)+1，K 组 (l%32)/8，共 16 B |
| K 步 | 64-K 块 | 4 条 MFMA 16x16x32（2 行 × 2 个 K32） |

A 操作数：l%2 == l/32 的 lane 取本槽本 K 组的激活，其余取 LDS 的全零行；
同一 MFMA 的输出行 0/8 只含低槽、行 1/9 只含高槽（两槽的通道因子不同，须分开乘；见 docstring “Slot pairs”）。

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；tid < I/16：LDS 末尾写一行全零（被屏蔽 lane 的 A 操作数）
  每轮 r、半 s：slot、expert（readfirstlane 成标量）、路由权重、ms[token, slot]
  发 mq 行加载：lane 读本半槽的第 l%32 + 32u 个 16 B 片（越界片钳到最后一片）
  发权重加载：整张量 buffer，每 lane 偏移 = 本半槽 expert × 每 expert 字节数（32 位）
    各轮的块展平后前 6 块：每块 1 条 dwordx4；每 4 块 1 条 delta dwordx2；两槽的通道因子各 1 条 dwordx2
mq 片 → LDS 行 [w, r, 半]（只写有效片）；barrier
主循环 r、kb = 0 … I/64 − 1
  先补发展平序列里的下一块；xop = 保留的 lane ? LDS 本槽行第 kb 块、K 组 (l%32)/8 的 16 B : 零行
  每行 h：LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64；MFMA × 4：acc[h] += xop[c] · w[h][c]
  vals[h] += Σ 有效槽 s：acc[h][s] × 通道因子[s][h] × ms[s] × 路由权重[s]（[0] 低槽、[1] 高槽）
  SHARED：再发 out[token, j·16 + tid%16] 的 bf16 读（折叠的 shared expert 的值，已乘路由权重）
收尾
  lane 0–7、40–47：vals[h] → LDS part[w, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8 → 树形相加（SHARED：再加读到的值）→ bf16（RTE）→ out[token, j·16 + tid]
```
关键代码：[零行](decode_a8w4.py#L875)、[槽对](decode_a8w4.py#L883)、[mq 片](decode_a8w4.py#L899)、
[整张量寻址](decode_a8w4.py#L920)、[屏蔽 A 操作数](decode_a8w4.py#L989)、[主循环](decode_a8w4.py#L995)、[收尾](decode_a8w4.py#L1027)。

**命名与不变量**：ROWS = 2·W·R 个暂存行，之后是零行；upper = l ≥ 32（高槽半）；keep = (l%2 == l/32)；chan_scale[q][h] 槽 q 的行 2(l%8)+h。
1. 零行（tid < I/16 各写一片）与 mq 片都在同一次 barrier 之前写完；越界片读最后一片、不写。
2. expert 号 readfirstlane 成标量，供通道因子的标量偏移；权重偏移是 32 位，整张量视图的记录数是 2^32 减一个 expert 的字节：
   expert −1 的偏移（含 SGPR 偏移）正落在其上，读到 0、不访存；host 只在 w2 加一个 expert ≤ 4 GiB 时选本 kernel。
3. 被屏蔽 lane 读零行：行 s / 8+s 只含槽 s，两槽各乘自己的通道因子、ms 与路由权重后才相加。
4. 部分和按固定顺序 tree_sum：同输入的输出逐位确定。
5. 屏蔽槽（id −1）的权重读到 0；H5120 只去重 Gate/Up 的路径里没人写它的 mid 行，mq / ms 由它量化而来（可为任意值，含 NaN）：
   按不变量 3 只进它自己的输出行，再由 select 丢弃，不会混进同一 MFMA 的另一槽。

**资源**：LDS = (2·W·轮数 + 1)·I B + 128·W B + 96 B，TP2 topk11 约 4.5 KiB；
每 lane 偏移是 32 位，w2 总字节数加一个 expert 须 ≤ 4 GiB（否则 host 改用核内量化版）。

## 行量化 decode（[decode_a8w4.py](decode_a8w4.py#L1082)）
槽对 Down 与去重的四个 kernel 路径量化 mid（f32 [M·topk, I]），M·topk·I/16 ≥ 2560 时量化 x（bf16 [M, H]）；每行一次倒数加 FMA
修正（quant_fast：bf16 一次、f32 两次，fast_ok 之外逐元素 IEEE 除法），字节与 [quant_a8w4.py](quant_a8w4.py) 逐位一致。

**Gluon 风格**
```python
# grid = ⌈rows/4⌉，256 线程：每个 wave 一行
# 放置：不用 LDS；lane 读 16 B 片 l + 64m 进 VGPR，行内 max 用 DPP + readlane
@gluon.jit
def a8w4_row_quant_decode(x, q, scale, rows):        # x：bf16 [M, H] 或 f32 mid [M·TOPK, I]
    r = 4 * gl.program_id(0) + wave_id()             # 行号 ≥ rows 的 wave 不读不写
    v = cdna3.buffer_load(x[r, :])                   # 越过行尾的 lane 重读最后一片（不改 amax）
    amax = wave_max(gl.abs(v))
    s = row_scale(amax)
    q_r = quant_fast(v, s, 1 if bf16 else 2) if fast_ok(amax) else to_fp8(gl.clamp(v / s, -240, 240))
    cdna3.buffer_store(q_r, q[r, :])                 # 与 a8w4_row_quant_fp8 逐位一致
    cdna3.buffer_store(s, scale[r])                  # lane 0 写
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows/4⌉ | 4 行 |
| CTA（256 线程） | 4 个 wave | 每 wave 1 行（rows 以外的 wave 不读不写） |
| lane | 16 B 片 l + 64m | bf16 8 个、f32 4 个元素 |

**流水线**
```text
lane 读 16 B 片 l + 64m（越过行尾的 lane 重读最后一片，不改 amax）
lane 内 max|x| → DPP 4 步 + readlane 4 行得整行 amax；scale = amax / 240（IEEE，0 取 1）
fast_ok：r = 1/scale（一次 IEEE 除法），q = RN(x·r) + bf16 一次、f32 两次 FMA 修正；否则 q = x/scale（IEEE）
q clamp ±240 → cvt_pk_fp8（RNE；两次转换写同一 dword）→ -0 改 +0（一次管 4 字节）→ 每 4 个打包成 32 位字
有效片写回：bf16 每片 8 B、f32 每片 4 B；lane 0 写 scale
```
关键代码：[读片](decode_a8w4.py#L1107)、[scale](decode_a8w4.py#L1130)、[写回](decode_a8w4.py#L1132)、[-0 处理](decode_a8w4.py#L193)。

**资源**：不用 LDS。

## 混合 Gate/Up decode：shared FP8 作末路（[decode_a8w4.py](decode_a8w4.py#L1359)）
带 shared 的调用在 M ≤ `DECODE_SHARED_MIXED_MAX_M`（4）、routed 列走核内量化的逐路由 decode、TOPK ≤ 16 时用；能折叠时（M ≥ 4 的
槽对 Down）只到 M·routed topk·I ≤ `DECODE_SHARED_MIXED_MAX_WORK`（10240：主干 TP4 M4 = 13824 改走折叠，graph 下 0.94× 混合；
TP8 与 MTP 的 M4 仍混合，是折叠的 0.55–0.82×）
（[`_mixed_decode`](../../../moe_a8w4.py#L2264)，档位只按 routed 列判断）：shared（topk_ids 的末列，id = E）和 routed 列在同一个 launch
里算 Gate/Up，接着混合 Down（下一节）在同一个 launch 里把每列加进 out，没有 side stream、没有最后相加。
shared 权重直接用 prepare 存的 `shuffle_weight(16,16)`：K%64 = 0 时它就是 `[N/16][K/64][4][16 行][16 B]`（routed 块的两倍字节、
K 自然序），lane 16g+n 一条 16 B 加载拿到行 n 的 K 组 g，字节直接作 MFMA 的 B（不读 delta、不查 LUT），每行 scale 代替通道因子。
shared 生效 = 末列为 E 且 ids[t,0] ≥ 0（ATOM 规则）；不生效时 buffer 记录数取 0，不读权重，mid 行为 0。

**Gluon 风格**
```python
# grid = (I/16, M·(TOPK+1))，256 线程（4 个 wave）；TOPK 列 = R 个 routed 列 + 末列 shared；token t 的 grid 行 t·(TOPK+1)+s：
#   s < R 为 routed 路由（同逐路由 Gate/Up），s = R、R+1 是 shared 的两半 h（各 8 个 gate 行与同号 8 个 up 行 = 一条 MFMA 的
#   16 列），这样每个 WG 读的字节数与 routed 的相同
# 放置：同逐路由 Gate/Up：x 行（H 个 fp8）在 LDS，4 个 wave 共读；权重 global → VGPR，序言先发前 GATEUP_PREFETCH_PAIRS = 6 条
#       16 B（每 lane 共 H/256 条，两类 WG 是同一组加载指令，只是描述符与偏移按 WG 选），其余在主循环里每消耗一条补发一条
#       （两个分支各自补发；shared 分支不补发 delta）；LUT 在 LDS（routed 用）；4 个 wave 的部分和经 LDS 相加
@gluon.jit
def a8w4_gateup_decode_mixed(x, topk_ids, w1, w1_delta, w1_scale, shared_w1, shared_s1, mid):
    j, y = gl.program_id(0), gl.program_id(1)
    token, s, w = y // (TOPK + 1), y % (TOPK + 1), wave_id()
    h = s - R                                        # shared 的哪一半（s ≥ R 时）
    xr = cdna3.buffer_load(x[token, :])              # 先发（不依赖 id）：线程 t 读 bf16 对 t + 256m
    if s >= R:                                       # 分支：shared WG 两个 id 一起发，routed WG 只等自己的 id
        live = topk_ids[token, R] == E and topk_ids[token, 0] >= 0
        rows = 16 * j + 8 * h + gl.arange(0, 8)      # lane 16g+n：n < 8 取 gate 行 rows[n]，n ≥ 8 取 up 行 I + rows[n−8]
        Q = H // 256                                 # 本 wave 管 K 的第 w 个四分之一：Q 个 64-K 块，每块每 lane 16 B
        wr = [cdna3.buffer_load(shared_w1[gate / up 行, 64 * (w * Q + p):64 * (w * Q + p + 1)]) for p in range(Q)]
        cs = shared_s1[rows], shared_s1[I + rows]    # tid < 8 且 live 才读
    else:
        expert = topk_ids[token, s]; live = expert >= 0
        wr, deltas, cs = 逐路由 Gate/Up 的块对加载与通道因子      # w%2 管 gate / up，w//2 管 K 的前后半
    # 不 live：权重与 delta 的 buffer 记录数取 0（读到 0、不访存），cs 不读（0），于是 mid 行为 0
    red.index(w).store(wave_max(gl.abs(xr))); gl.barrier()   # barrier 保护 4 个 wave 的 max
    amax = gl.max(red.load(…)); sx = row_scale(amax)
    xl.store(quant_fast(xr, sx, 1) if fast_ok(amax) else to_fp8(gl.clamp(xr / sx, -240, 240)))
    gl.barrier()                                     # x 行写完 LDS 才可读
    if s >= R:
        acc = gl.zeros([16, 16], f32)                # A 的 16 行相同（x 的 K 组 g），每个输出行都是列 n 的全组和
        for p in range(Q):
            acc = cdna3.mfma(xl.load(A 操作数布局)[64 * (w * Q + p):…], wr[p], acc)   # 每块 2 条
        part.index(w).store(acc[0]); gl.barrier()    # barrier：part 写完才读
        g, u = tree_sum(part[:, :8], axis=0), tree_sum(part[:, 8:], axis=0)   # tid < 8：4 个 K 四分之一相加
        mid[token, R, rows] = swiglu(g * cs[0] * sx, u * cs[1] * sx)
    else:
        mid[token, s, 16 * j:16 * j + 16] = 逐路由 Gate/Up 的主循环与收尾(xl, wr, deltas, cs, sx)
```

**任务划分**（数字按 DS TP4：H5120、I576、TOPK 7）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·(TOPK+1)) = (36, 8M) | x：16 个通道块 j；y：token 的 R 条 routed 路由与 shared 两半 |
| CTA（256 线程） | 4 个 wave | routed：32 行 × 全部 H；shared：8 个 gate 与 8 个 up 行 × 全部 H |
| wave（routed） | w%2：gate / up；w/2：K 前后半 | 16 行 × H/2（20 个块对） |
| wave（shared） | w：K 的第 w 个四分之一 | 16 行 × H/4（20 个 64-K 块） |
| lane（shared） | l = 16g + n | 行 n（n < 8 gate、≥ 8 up）的 K 组 g，每块 16 B |
| K 步 | routed 块对；shared 64-K 块 | 块对 4 条、块 2 条 MFMA 16x16x32 |

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；发 x 行加载（线程 t 读 bf16 对 t + 256m，H5120 每线程 10 对）
  shared WG：读 ids[t, R]、ids[t, 0] → live；routed WG：读 ids[t, s] → live；按 WG 选权重描述符与偏移
  for p < 6（sched_barrier 固定顺序）：p 偶时 delta 8 B（shared 的记录数为 0，读到 0）；权重 16 B（其余在主循环里补发）
  tid < 16（shared 为 tid < 8）且 live：通道因子 / 行 scale 的 gate、up 值
x 行 → LDS xq
  lane 内 max|x| → DPP 4 步 + readlane → LDS red[wave]；barrier；scale = max(red) / 240（IEEE，0 取 1）
  fast_ok：r = 1/scale（一次 IEEE 除法），q = RN(x·r) + 一次 FMA 修正；否则 q = x/scale（IEEE）
  两个 bf16 对共用一个 cvt_pk_fp8 dword（第二次写高半字），-0 修正一次管 4 字节 → 每对 2 B 写 LDS；barrier
主循环
  routed：同逐路由 Gate/Up（块对、LUT 解码、每块对 4 条 MFMA）
  shared：p < H/256：xop = LDS xq 第 w·H/256 + p 块、K 组 l/16 的 16 B；MFMA × 2：acc += xop[c] · raw[p][c]
收尾
  shared：lane < 16 的 acc[0] → LDS part[w]；barrier；tid < 8：gate = Σ_w part[w][tid]、up = Σ_w part[w][8 + tid]
          × 行 scale × x scale → [截断] → silu · up → f32 写 mid[t, R, 16j + 8h + tid]
  routed：同逐路由 Gate/Up → mid[t, s, 16j + tid]
```
关键代码：[x 行加载](decode_a8w4.py#L1435)、[按 WG 选描述符](decode_a8w4.py#L1453)、[权重加载](decode_a8w4.py#L1489)、
[行 scale 与通道因子](decode_a8w4.py#L1507)、[x 行量化](decode_a8w4.py#L1517)、[主循环](decode_a8w4.py#L1538)、[收尾](decode_a8w4.py#L1575)、
[快速除法](decode_a8w4.py#L1207)、[x 行暂存](decode_a8w4.py#L1304)。

**命名与不变量**：R routed 列数（TOPK − 1）；SLOTS = TOPK + 1 每个 token 的 grid 行；half 是 shared 的哪 8 个通道；PAIRS 每 lane 的
16 B 加载数（H/256）；S_BLOCK 一个 16 行 × 64-K 的 FP8 块（1024 B）；S_UP 从 gate 行块到同号 up 行块的字节距离。
1. x 行加载先于 id 与权重加载发出；两类 WG 的权重加载是同一组指令（每 lane PAIRS 条 16 B、soffset 步长 1024 B）。
2. shared 生效 = 末列 E 且首列 ≥ 0；不生效时权重记录数为 0、行 scale 不读，mid 行为 0（屏蔽的 routed 路由同逐路由版）。
3. x 行的 FP8 字节与逐值 IEEE 除法逐位相同（一次修正；行最大值越界退回除法）；写完 LDS 后 barrier 才进主循环。
4. shared 只用 lane < 16 的 acc[0]（各输出行相同）；4 个 wave 的部分和按固定顺序 tree_sum，同输入的输出逐位确定。

**快速除法**：两个混合 kernel 在每 CU ≥ 2 个 WG 时受 VALU 限制，逐值 IEEE 除法（约 11 条）占 x 行量化的大半（routed Gate/Up
wave 约 1/3 的 VALU）。混合 kernel 每行只做一次 IEEE 除法求 r = RN(1/scale)，每值 q = RN(v·r) 再用 FMA 修正 q += (v − q·scale)·r：
Gate/Up 一次（BF16 x 对全部 BF16 行最大值穷举，FP8 字节与除法相同），Down 两次（第一次得忠实舍入、第二次得正确舍入）。
行最大值不在 [2^-60, 2^127) 且不为 0 时仍逐值除（太小的行余数会下溢）。逐路由 Gate/Up、单槽 Down 与 decode 的行量化 kernel 也用这种量化。

**资源**（H5120 各形状相同）：102 VGPR（只先发 6 条 16 B，每 SIMD 5 个 wave；全部先发时 155、3 个 wave），LDS 5744 B
（xq 5120、part 512、red 16、LUT 96）。shared 拆两半让每个 WG 的字节数相同；代价是 WG 数从 R·I/16·M 增到 (R+2)·I/16·M
（DS TP4 M1：288 个，超过每 CU 3 个 WG 的 240 个槽）。

## 混合 Down decode：shared FP8 作末 wave（[decode_a8w4.py](decode_a8w4.py#L1667)）
混合 Gate/Up 之后的 Down：每个 WG 算一个 token 的 32 个输出通道（行块对），每个路由列一个 wave（DS 7 个、MTP 4 个，与 M 无关），
最后一个 wave 读 shared 的 FP8 Down。每个 wave 独立跑完自己那一列（自己的 LUT 副本与 LDS 行，量化与 MFMA 之间只有 wave 级
fence），所以全部 wave 先发权重加载，整个 WG 只在求和前 barrier 一次。[down_decode_mixed_fits](decode_a8w4.py#L1348)：K%64 = 0、
2 ≤ TOPK ≤ 16、每个 wave 的 mid 行、部分和与 LUT 副本放得进 LDS。

**Gluon 风格**
```python
# grid = (H/32, M)，64·TOPK 线程（TOPK 个 wave）；wave w 管第 w 列（w = R = TOPK − 1 是 shared）
# 放置：每个 wave 自己的 mid 行（f32）global → VGPR，wave 内量化后写 LDS mq[w]、按 A 操作数布局读回（只有本 wave 读，
#       wave 级 fence 代替 barrier）；routed 权重每块每 lane 16 B（行块对，同单槽 Down），shared 每块每个行块 16 B，
#       global → VGPR，序言一次发出（全部块；graph 下比只先发 3 块快 0–3%）；
#       LUT 每个 wave 一份；TOPK 个 wave 的部分和经 LDS 相加后直接写 bf16 out
@gluon.jit
def a8w4_down_decode_mixed(mid, topk_ids, topk_weight, w2, w2_delta, w2_scale, shared_w2, shared_s2, out):
    j, token, w = gl.program_id(0), gl.program_id(1), wave_id()
    n = 16 * j + gl.arange(0, 16)
    expert, tw, first = topk_ids[token, w], topk_weight[token, w], topk_ids[token, 0]
    lut.index(w).store(LUT 表)                       # lane ≤ 11 写本 wave 的副本
    if w == R:                                       # 两个分支各自先发 mid 行（不依赖 id；shared 用常数行号）
        mr = cdna3.buffer_load(mid[token, R, :])
        live = expert == E and first >= 0
        wr = [[cdna3.buffer_load(shared_w2[行块 2j+b, 64 * kb:64 * kb + 64]) for kb in range(I // 64)]
              for b in (0, 1)]                       # 行块步长按补零的 KP；每块每个行块一条 16 B
        cs = shared_s2[两个行块的 n]
    else:
        mr = cdna3.buffer_load(mid[token, w, :]); live = expert >= 0
        wr, deltas, cs = 单槽 Down 的加载(expert)     # 行块对：每 64-K 块每 lane 16 B，每 4 块一条 delta 8 B
    # 不 live：权重、delta、因子都经记录数为 0 的 buffer 视图读到 0、不访存
    amax = wave_max(gl.abs(mr)); s = row_scale(amax)
    mq.index(w).store(quant_fast(mr, s, 2) if fast_ok(amax) else to_fp8(gl.clamp(mr / s, -240, 240)))
    wave_fence()                                     # 本 wave 写完 mq[w] 再读；不发指令
    acc = gl.zeros([2, 16, 16], f32)                 # shared：两个行块；routed：两行（h，同单槽 Down）
    for kb in range(I // 64):                        # 每块 4 条 MFMA；shared 的字节直接作 B
        a = mq.index(w).load(A 操作数布局)[:, 64 * kb:64 * kb + 64]
        for b in (0, 1):
            acc[b] = cdna3.mfma(a, wr[b][kb] if w == R else decode(wr[kb], b, deltas, lut.index(w)), acc[b])
    part.index(w).store(gl.where(live, acc[·][0] * cs * s * tw, 0))   # select，不乘 0；32 个通道
    gl.barrier()                                     # 唯一的 WG barrier：TOPK 个部分和写完才合并
    out[token, 32 * j:32 * j + 32] = tree_sum(part.load(…), axis=0).to(bf16)   # RTE；直接写 out
```

**任务划分**（数字按 DS TP4：H5120、I576、TOPK 7）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/32, M) = (160, M) | x：16 行通道块 2j、2j+1；y：一个 token |
| CTA（64·TOPK 线程） | TOPK 个 wave | 该 token 的全部列 |
| wave | w < R：routed 第 w 列；w = R：shared | 该列的 32 行 × 全部 I |
| lane | l = 16g + r | routed 同单槽 Down（行块对，每块 16 B）；shared 两个行块各自的行 r、K 组 g（各 16 B）；mid 列 l + 64u |
| K 步 | 64-K 块 | 4 条 MFMA 16x16x32（I576 为 9 块） |

**流水线**
```text
序言（每个 wave 各自进行）
  lane ≤ 11：本 wave 的 LUT 副本 → LDS；读本列 expert、路由权重与首列 id
  shared wave：发 mid 行（常数行号）→ 每块两个行块各 1 条 16 B 权重 → 两个行块的行 scale
  routed wave：发 mid 行 → 每块 1 条 dwordx4 权重（行块对；每 4 块前 1 条 delta dwordx2）→ 两行的通道因子（dwordx2）
wave 内量化本列 mid 行
  amax → scale；fast_ok：r = 1/scale，q = RN(v·r) + 两次 FMA 修正；否则 q = v/scale（IEEE）→ 逐字节写 LDS mq[w]；wave 级 fence
主循环 kb = 0 … I/64 − 1
  xop = LDS mq[w] 第 kb 块、K 组 l/16 的 16 B；shared：两个行块的 16 B 直接作 B；routed：LUT 解码 → 两行各 2 个 fp8 i64
  MFMA 16x16x32 × 4 → acc[0], acc[1]
收尾
  total = live ? acc[·][0] × 因子 × scale × 路由权重 : 0；lane < 16：两个 total → LDS part[w, 通道]；barrier
  tid < 32：TOPK 个部分和 tree_sum → bf16（RTE）→ out[token, 32j + tid]
```
关键代码：[LUT 副本](decode_a8w4.py#L1733)、[shared wave](decode_a8w4.py#L1745)、[routed wave 的量化](decode_a8w4.py#L1841)、
[部分和](decode_a8w4.py#L1866)、[求和](decode_a8w4.py#L1881)。

**命名与不变量**：KB / KBP 是 I 与 shared 补零后的 K（KP，128 的倍数）的 64-K 块数；R 是 shared 的 wave 号。
1. 两个分支各自先发 mid 行、shared 分支用常数行号：否则 LLVM 把两边相同的 abs-max 提到分支前，等 mid 返回才发权重加载。
2. 每个 wave 只读自己写的 mq 行与自己的 LUT 副本，量化与 MFMA 之间只有 wave 级 fence；整个 WG 只在部分和处 barrier 一次。
3. mid 行的 FP8 字节与逐值 IEEE 除法逐位相同（两次修正；行最大值越界退回除法）。
4. 屏蔽列与不生效的 shared 不读权重（记录数 0），贡献由 select 置零（它们的 mid 行与路由权重可以是任意值）；
   部分和按固定顺序 tree_sum，同输入的输出逐位确定。

**资源**：LDS = TOPK·(I + 128 + 96) B（DS TP4 5600 B、MTP TP8 2176 B）；VGPR MTP TP8 62（8 B 单块读法时 44）。主干 TP4 的 M4 改走折叠（见混合 Gate/Up 一节的
DECODE_SHARED_MIXED_MAX_WORK）；M8 起 H5120 的 routed Gate/Up 去重，不走混合。

## 去重计划与 x 行量化（[decode_dedup_a8w4.py](decode_dedup_a8w4.py#L196)）
按 expert 去重的四个 kernel 的路径与只去重 Gate/Up 的 H5120 路径的第一步：一次 launch 量化 x 的每一行（与 row_quant_decode
同算术、逐位一致）、把 out 清零（给 down_decode_dd 原子加；H5120 路径编译成 ZERO_COLS = 0、不清零，其槽对 Down 直接写），
并为每条路由写计划条目（格式见文首 plan）。两 kernel 路径由 PLAN 版 Gate/Up decode 写同样的计划。

**Gluon 风格**
```python
# grid = M·(1 + ⌈TOPK/4⌉)，256 线程：前 M 个 CTA 各量化 x 的一行，其余 CTA 的每个 wave 规划一条路由
# 放置：x 行 global → VGPR（每线程 16 B 片 t + 256m），行 amax 先在 wave 内、再经 LDS 的 4 个槽合并；规划时每个 wave
#       把全部 M·TOPK 个 expert id 按 64 条一段读进 VGPR（lane l 持有第 l + 64k 条），用 ballot 数同 expert 的路由
@gluon.jit
def a8w4_row_quant_plan(x, xq, xs, topk_ids, plan, out):
    b = gl.program_id(0)
    if b >= M:                                       # 规划：每个 token 用 ⌈TOPK/4⌉ 个 CTA，wave 管一个槽
        t, slot = (b - M) // ⌈TOPK/4⌉, ((b - M) % ⌈TOPK/4⌉) * 4 + wave_id()
        if slot < TOPK:
            r = t * TOPK + slot
            ids = topk_ids.flat[lane_id() + 64 * gl.arange(0, SEGS)]   # SEGS = ⌈MAX_M·TOPK/64⌉，越界为 −1
            e = topk_ids.flat[r]
            hit = [ballot(ids[k] == e) for k in range(SEGS)]           # 每段 64 位
            rank = Σ popcount(hit 中路由号 < r 的位); total = Σ popcount(hit)
            nxt = 依次取上一个成员之后的最低置位（从低段找起），共 GROUP−1 个；没有记 0x7FFFFFFF（其后的字无意义）
            lead = rank % GROUP == 0 and e >= 0       # 屏蔽路由（e = −1）也会配上越界的 −1，所以不让它领组
            plan[r] = [min(GROUP, total - rank) if lead else 0, r, *nxt]   # lane 0 写 16 B
    else:                                            # 量化第 b 行
        v = cdna3.buffer_load(x[b, :])               # 越过行尾的线程重读最后一片（不改 amax）
        red = gl.allocate_shared_memory(f32, [4], …)
        red.index(wave_id()).store(wave_max(gl.abs(v))); gl.barrier()   # barrier：4 个 wave 的 max 都写完才合并
        s = row_scale(gl.max(red.load(…)))
        xq[b, :] = quant_fast(v, s, 1) if fast_ok(amax) else to_fp8(gl.clamp(v / s, -240, 240)); xs[b] = s
        if ZERO_COLS: out[b, :] = 0                  # 同一 CTA 先读完 x 行（barrier 之前）再清零，所以 out 可以就是 x
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | M·(1 + ⌈topk/4⌉) | 前 M 个：x 的一行；其余：一个 token 的 4 条路由 |
| CTA（256 线程） | 4 个 wave | 一行的全部 H（H4096 每线程 2 片 16 B；H5120 640 片，第三轮只有前 128 个线程）或 4 条路由 |
| lane | 量化：16 B 片 t + 256m；规划：第 l + 64k 条路由的 id | MAX_M = 64 时 SEGS ≤ 8 |

**流水线**
```text
规划 CTA（b ≥ M）：wave 管路由 r：lane l 读第 l + 64k 条路由的 id（越界为 −1）；标量读 expert = ids[r]
  每段 ballot(id == expert) → rank、total → 依次找后 GROUP−1 个成员 → lane 0 写 plan[r]（16 B）
量化 CTA（b < M）：线程读 x 行的 16 B 片 t + 256m → lane 内 max → wave_max → LDS red[wave]；barrier
  scale = max(red)/240（0 取 1）；q = quant_fast(x, scale, 1)（fast_ok 之外 clamp(x/scale)，IEEE）→ RNE fp8 → 每片 8 B 写 xq；tid 0 写 xs；ZERO_COLS > 0 时同一行
  out 每片 16 B 写 0
```
关键代码：[规划一条路由](decode_dedup_a8w4.py#L139)、[量化分支](decode_dedup_a8w4.py#L252)、[清零 out](decode_dedup_a8w4.py#L309)。

**命名与不变量**：PW = ⌈TOPK/4⌉ 每个 token 的规划 CTA 数；SEGS 规划的 ballot 段数；_NONE = 0x7FFFFFFF 表示没有后续成员。
1. 量化 CTA 读 x 行在 barrier 之前、清零 out 在 barrier 之后：out 可以就是 x（原地输出时 host 走这条路）。
2. 规划 wave 只读 topk_ids、只由 lane 0 写 plan[r]；越界的路由号读 −1，不会与任何 expert ≥ 0 相同；屏蔽路由（−1）
   与它们相配，所以条数固定为 0：不领组、也不进任何组，去重的 Gate/Up 与 Down 都不算它。
3. SEGS 编译期固定（MAX_M 由 host 按 M ≤ 32 / ≤ 64 选）：M > MAX_M 时 ballot 段不够，计划出错。

**资源**：LDS 16 B（4 个 wave 的 amax）；不用 MFMA。

## A8W4 Gate/Up decode 去重（[decode_dedup_a8w4.py](decode_dedup_a8w4.py#L513)）
按 expert 去重的四个 kernel 的路径与 H5120 只去重 Gate/Up 的路径里代替逐路由 Gate/Up：grid 与逐路由版相同，但只有组首路由的
CTA 计算，一次读该 expert 的 16 行块，算组内 ≤ GROUP 条路由（每个 token 都选的 shared expert 只读 ⌈M/GROUP⌉ 次而不是 M 次）。

**Gluon 风格**
```python
# grid = (I/16, M·TOPK)，64·WAVES 线程：WAVES = 8（M ≤ 16，H%1024 = 0；H5120 路径除外）或 4；GROUP = 2（M ≤ 16）或 3
# 放置：组员 token 的 fp8 x 行 global → LDS [GROUP, H+16]，A 操作数按块对布局从 LDS 读（各 wave 读自己的 K 段）；
#       权重 global → VGPR：序言发出本 wave 那段 K 的全部块对，topk·I ≥ DECODE_DD_GATEUP_PREFETCH_MIN_TOPK_I（3072：主干 TP4、
#       MiMo TP4）时只先发 6 个、主循环每消耗一个补发一个；LUT 在 LDS；部分和经 LDS 跨 wave 相加
@gluon.jit
def a8w4_gateup_decode_dd(xq, xs, topk_ids, plan, w1, w1_delta, w1_scale, mid):
    j, route = gl.program_id(0), gl.program_id(1)
    count, members, expert = plan[route, 0], plan[route, 1:1 + GROUP], topk_ids.flat[route]
    # 上面三个标量加载与 kernarg 用空 asm 钉在分支前一起发出（否则 LLVM 把它们下沉进分支，变成几轮串行访存）
    if count == 0: return                            # 非组首：本路由由组首的 CTA 一起算
    w = wave_id()
    rb = (w % 2) * (I // 16) + j                     # w%2：gate / up 的 16 行块
    kb0 = (w // 2) * KB_PART                         # w//2：K 的第几段（H/64 个 64-K 块分 WAVES/2 段）
    xl = gl.allocate_shared_memory(fp8, [GROUP, H + 16], …)   # 行距 H+16
    xr = cdna3.buffer_load(xq[members // TOPK, :])   # 组员的 x 行先发（不等权重）；count 以外的行读到 0
    wr = [cdna3.buffer_load(w1[expert, rb, 64 * (kb0 + 2 * p):64 * (kb0 + 2 * p + 2)])   # 块对 p：每 lane 16 B；
          for p in range(PAIRS)]                     #   每两对先发一条 delta 8 B；PAIRS = KB_PART/2
    sx, cg, cu = xs[组员 token], w1_scale[expert, 16 * j + t], w1_scale[expert, I + 16 * j + t]   # tid < GROUP·16
    xl.store(xr); gl.barrier()                       # barrier：x 行写完才可读（LUT 表也在此前写好）
    acc = gl.zeros([2, 16, 16], f32)                 # h = 0/1：lane 持有的两个权重行
    for p in range(PAIRS):                           # 第 p 对只等自己的加载
        a = xl.load(块对 A 布局)[:, 64 * (kb0 + 2 * p):64 * (kb0 + 2 * p + 2)]   # A 行 m 与 8+m 都取组员 m % GROUP
        for h in gl.static_range(2):
            acc[h] = cdna3.mfma(a, decode(wr[p], h, lut), acc[h])
    # acc[h] 的行 m 列 0–7 与行 8+m 列 8–15（m < GROUP）是组员 m 的两半 K 组，写 LDS；其余行是重复的组员，丢弃
    part = gl.allocate_shared_memory(f32, [WAVES, 2, GROUP, 16], …)
    part.index(w).store(acc 的有效行); gl.barrier()  # barrier：各 wave 的部分和写完才相加
    # tid < GROUP·16：组员 m = t // 16、通道 16j + t%16（= 2c + h）：列 c 与 c+8 相加得全 K，偶数 wave 是 gate、奇数是 up
    g = tree_sum([part[v, h, m, c] + part[v, h, m, c + 8] for v in range(0, WAVES, 2)]) * cg * sx[m]
    u = tree_sum([part[v, h, m, c] + part[v, h, m, c + 8] for v in range(1, WAVES, 2)]) * cu * sx[m]
    mid[members[m], 16 * j + t % 16] = swiglu(g, u)   # f32；WAVES=4 时求和顺序同逐路由版（逐位相同），8 wave 时 K 分 4 段相加
```

**任务划分**（H4096；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·topk) | x：16 个 gate 通道块 j 与同号 up 块；y：一条路由（只有组首计算） |
| CTA（64·WAVES 线程） | WAVES = 8（M ≤ 16）/ 4 | 一组 ≤ GROUP 条路由 × 32 行权重 × 全部 H |
| wave | w%2：gate / up；w/2：K 的第几段 | 16 行 × H/(WAVES/2)：8 wave 为 8 个块对，4 wave 为 16 个 |
| lane | 块对中第 l/32 块的行 2(l%8)、+1，K 组 (l%32)/8，共 16 B | A 取组员 (l%8) mod GROUP 的同段 x |
| K 步 | 块对 | 4 条 MFMA 16x16x32（2 行 × 2 个 K32），组内各路由共用 |

**流水线**（不分 tile：序言一次发出本 wave 的全部加载，主循环按序消费；无双缓冲）
```text
序言
  tid ≤ 11：LUT → LDS；标量读 expert、plan[route]（条数、组员）与 kernarg，pin 在分支前（一轮标量访存）；条数 0 → 退出
  发组员 x 行：线程 t 读第 t + 64·WAVES·u 片 16 B（组员 m = 片号 / (H/16)，m ≥ 条数的读到 0）
  发本 wave 的权重：每块对 1 条 dwordx4，每两对 1 条 delta dwordx2（sched_barrier 固定顺序）
  tid < GROUP·16：组员 token 的 xs、通道因子（gate、up）
x 片 → LDS [组员, H+16]；barrier
主循环 p = 0 … PAIRS−1（第 p 对只等自己的加载）
  xop = LDS 组员 (l%8)%GROUP 行、第 kb0+2p+l/32 块、K 组 (l%32)/8 的 16 B
  每行 h：LUT[delta 字节] → 解码 → MFMA × 2（两个 K32）：acc[h] += xop · w[h]
收尾
  lane < 32 的行 m 列 0–7、lane ≥ 32 的行 8+m 列 8–15（m < GROUP）→ LDS part[wave, h, m]；barrier
  tid < GROUP·16：各 wave 的列 c + 列 c+8 → tree_sum（偶数 wave 为 gate）→ × 通道因子 × xs → [截断] → SiLU(gate)·up
  → f32 写 mid[组员路由, 16j + t%16]（条数以外的线程不写）
```
关键代码：[组首判断](decode_dedup_a8w4.py#L608)、[x 行与权重](decode_dedup_a8w4.py#L384)、[主循环](decode_dedup_a8w4.py#L453)、
[收尾](decode_dedup_a8w4.py#L469)、[pin](decode_dedup_a8w4.py#L85)、[host 分派](../../../moe_a8w4.py#L996)、
[H5120 路径](../../../moe_a8w4.py#L1092)。

**命名与不变量**：count / members / expert 组首条目与本路由的 expert；G = GROUP；XS = H+16 LDS 行距；KB_PART 本 wave 的 64-K 块数；
em = t//16、cr = t%16 收尾线程的组员与通道行 2m'+h。
1. 条数、组员、expert 与 kernarg 的标量加载由 _pin 钉在分支前一起发出；非组首 CTA 在发任何向量加载之前退出。
2. 组员 x 行先于权重加载发出（不依赖 expert）；条数以外的组员读到 0、不分支；第 p 个块对只等自己的加载。
3. x 片写 LDS 后一次 barrier 才进主循环；部分和写 part 后一次 barrier，tid < GROUP·16 才合并。
4. 4 wave 时每个组员的输出与逐路由 Gate/Up 逐位相同；8 wave 时 K 分 4 段相加，末位可能不同（之后还要按行量化）。

**资源**：LDS = GROUP·(H+16) B + 128·WAVES·GROUP B + 96 B LUT（H4096：GROUP2·8 wave 约 10.1 KiB，GROUP3·4 wave 约 13.6 KiB；
H5120 GROUP3·4 wave 约 16.6 KiB）；每 lane PAIRS 条 16 B 权重加载同时在途（4 wave 为 16 条）。

## A8W4 Down decode 去重（[decode_dedup_a8w4.py](decode_dedup_a8w4.py#L846)）
按 expert 去重的两种路径（两个 / 四个 kernel）都用它：组首的 CTA 一次读 expert 的 4·CB 个 16 通道块，算组内各路由，结果乘路由权重后用 bf16x2
原子加进已清零的 out（求和顺序不定，输出不逐位确定）。

**Gluon 风格**
```python
# grid = (H/(64·CB), M·TOPK)，256 线程（4 个 wave，每 wave CB = 2 个 16 通道块）；只有组首路由的 CTA 计算
# 放置：组员的 mid 行进 LDS [ROWS, I+16]：QUANT（两 kernel 的路径）时 wave m 把组员 m 的 f32 行读进 VGPR、wave 内量化后写
#       LDS（ROWS = 4），否则 fp8 行 global → LDS（ROWS = GROUP）；A 操作数从 LDS 读，全部块对一次读进 VGPR；本 wave 的权重与
#       通道因子在序言一次发出直进 VGPR；结果不经 LDS，lane 0–7 直接原子加
@gluon.jit
def a8w4_down_decode_dd(mid_or_mq, ms, topk_ids, topk_weight, plan, w2, w2_delta, w2_scale, out):
    jj, route = gl.program_id(0), gl.program_id(1)
    count, members, expert = plan[route, 0], plan[route, 1:1 + GROUP], topk_ids.flat[route]   # 钉在分支前，同 Gate/Up
    if count == 0: return
    tok, tw = members // TOPK, topk_weight.flat[members]      # 每个组员的 token 与路由权重（标量）
    blocks = [(4 * jj + wave_id()) * CB + cb for cb in range(CB)]
    if QUANT: v = cdna3.buffer_load(mid[members[wave_id()], :]) if wave_id() < count else 0   # lane 读 16 B 片 l + 64u
    else:     xr = cdna3.buffer_load(mq[members[:count], :]); sm = ms[members]              # 全 CTA 分片读
    wr = [[cdna3.buffer_load(w2[expert, 16 * b:16 * b + 16, 块对 p]) for p in range(I // 128)] for b in blocks]
    cs = [w2_scale[expert, 16 * b + 2 * (lane_id() % 8) + (0, 1)] for b in blocks]   # lane 0–7 用
    if QUANT:                                        # 与 row_quant_decode 同算术
        s = row_scale(wave_max(gl.abs(v))); xl[wave_id()] = to_fp8(gl.clamp(v / s, -240, 240)); sl[wave_id()] = s
    else:
        xl.store(xr)
    gl.barrier()                                     # barrier：组员行与 scale 写完才可读
    f = (sl[:GROUP] if QUANT else sm) * tw           # 每个组员：行 scale × 路由权重
    a = [xl.load(块对 A 布局)[:, 块对 p] for p in range(I // 128)]   # A 行 m 与 8+m 都取组员 m % GROUP
    for b, wb, cb in zip(blocks, wr, cs):
        acc = gl.zeros([2, 16, 16], f32)             # 行 = 组员，列 = 通道
        for p in range(I // 128):
            for h in gl.static_range(2):
                acc[h] = cdna3.mfma(a[p], decode(wb[p], h, lut), acc[h])
        for g in range(GROUP):                       # lane c < 8：行 g 列 c 加上 lane c+40 的行 8+g 列 c+8（ds_bpermute）
            pair = [(acc[h][g, c] + acc[h][8 + g, c + 8]) * cb[h] * f[g] for h in (0, 1)]   # 通道 16b + 2c + h
            if lane_id() < 8 and g < count:
                atomic_add(out[tok[g], 16 * b + 2 * c:16 * b + 2 * c + 2], to_bf16(pair))   # RTE，agent 范围
```

**任务划分**（H4096、CB2；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/128, M·topk) | x：128 个输出通道；y：一条路由（只有组首计算） |
| CTA（256 线程） | 4 个 wave | 一组 ≤ GROUP 条路由 × 128 通道 × 全部 I |
| wave | CB = 2 个 16 通道块 | 每块 16 行权重 × I；QUANT 时 wave m 还量化组员 m 的 mid 行 |
| lane | 块对中第 l/32 块的行 2(l%8)、+1，K 组 (l%32)/8 | lane 0–7 各写组员的通道 2c、2c+1（4 B 原子加） |
| K 步 | 块对（I/128 个） | 每块 4 条 MFMA 16x16x32 |

**流水线**（不分 tile：序言一次发出全部加载；无双缓冲）
```text
序言
  LUT → LDS；标量 expert、plan[route] 与 kernarg pin 在分支前；条数 0 → 退出
  每个组员（标量）：路由（条数以外取组员 0）、token、路由权重；非 QUANT 还有 ms[路由]
  发组员 mid 行：QUANT 时 wave m 读组员 m 的 f32 行（lane 读 16 B 片 l + 64u）；否则全 CTA 分片读 fp8 行
  发本 wave CB 个块的权重：每块 I/128 条 dwordx4 + 每两对 1 条 delta dwordx2；每块 1 条 dwordx2 通道因子
QUANT：wave 内 amax → scale → 逐元素 IEEE 除法 → fp8 写 LDS 行 m，scale 写 LDS 槽 m；否则 fp8 片 → LDS
barrier；QUANT 时从 LDS 取各组员的 scale；因子[g] = scale × 路由权重
A 操作数：I/128 个块对各读 16 B（组员 (l%8)%GROUP 行）
for 本 wave 的 CB 个块：
  for 块对 p：LUT → 解码 → MFMA × 4（2 行 × 2 个 K32）
  每个组员 g：lane c < 8 的行 g 列 c + ds_bpermute 取 lane c+40 的行 8+g 列 c+8 → × 通道因子[h] × 因子[g]
    → 通道 2c、2c+1 打包 bf16x2 → lane < 8 且 g < 条数：agent 范围原子加 out[token_g, 16·块 + 2c]
```
关键代码：[组首判断](decode_dedup_a8w4.py#L932)、[组员行](decode_dedup_a8w4.py#L714)、[权重](decode_dedup_a8w4.py#L747)、
[WG 内量化](decode_dedup_a8w4.py#L771)、[主循环](decode_dedup_a8w4.py#L812)、[原子加](decode_dedup_a8w4.py#L843)。

**命名与不变量**：ROWS 暂存行数（QUANT 为 4，否则 GROUP）；factors[g] 组员 g 的行 scale × 路由权重；src = (l+40)·4 是 ds_bpermute 的源地址。
1. 同 Gate/Up 去重的 1、2：标量钉在分支前，组员行先于权重发出，条数以外的组员读到 0。
2. QUANT 时 wave m 只量化组员 m 的行，写 LDS 行 m 与 scale 槽 m；一次 barrier 后各 wave 才读 A 操作数与 scale。
3. 每个组员 g 只由 lane 0–7 原子加（g < 条数），每 lane 一条 bf16x2；out 须先由计划 kernel 或 PLAN 版 Gate/Up 清零。

**资源**：LDS = ROWS·(I+16) B + 16 B scale + 96 B LUT（I256 约 1.2 KiB）；Down 要求 I%128 = 0、QUANT 时 I%256 = 0。

## A8W4 Gate/Up 流式（[stream_a8w4.py](stream_a8w4.py#L450)）
流式档（M > 32、每个 expert 平均不超过流式上限且不走去重 decode）按排序块（同一 expert 的 16 / 32 / 64 行）
算 swiglu(x·w1_gate, x·w1_up)：权重与激活直接从 global 读进 VGPR，不经 LDS 双缓冲；写 f32
mid[token, slot, :]，或在尾部量化成 fp8 mq[route, :] 与 ms[route]。

**Gluon 风格**
```python
# grid = (I/(16·G·W), 排序块数)，64·W 线程；排序块 BM = 16 / 32 / 64，G、W 见任务划分
# 放置：权重与激活都 global → VGPR 直读（权重只读一次，带 non-temporal 位），没有 prefill 那种整块 LDS 双缓冲；
#       SHARE（W > 1；尾部量化时还要 G ≤ 4）时激活由全 WG 分片只读一份，经 LDS 环（2·SYNC 槽，每槽一个 64-K 步；
#       轻 wave SYNC = 2，尾部量化 1）分给各 wave。预取距离 P 个 64-K 步，由分派给：
#       尾部量化 S = 1 为 4、否则 3（子块多时少预取，省 VGPR）；其余为 5。
#       共享时步 k+SYNC 的激活在步 k 写进 LDS（覆盖步 k−SYNC 用过的槽），每 SYNC 步一次 barrier
#       ACT_STEPS = A > 1（K ≥ 5120 的轻 wave；排序块 16 / 32 时 A = 4 / 2）：SYNC = A，每 lane 一次 16 B 加载取一行 A 个
#       64-K 步里的一片（A = 1 时每步每 lane 4 / 8 B）；激活按 A 步一组提前 2 组发，组 g 末尾写组 g+1（覆盖组 g−1 的槽）再 barrier
@gluon.jit
def a8w4_gateup_stream(xq, xs, w1, w1_delta, w1_scale, sorted_ids, sorted_expert_ids, num_valid, mid, mq, ms):
    c, e = gl.program_id(0), gl.program_id(1)        # c：W·G 个 16 通道 gate 块及同号 up 块；e：排序块
    ids = sorted_ids[e * BM + gl.arange(0, BM)]
    S = valid_subblocks(ids, e, num_valid, M)        # e·BM ≥ num_valid 时为 0。为 S = 1…BM/16 各编一份代码，
                                                     #   运行时按 S 分支进入其中一份，S = 0 直接退出
    token, slot, expert = ids[:16 * S] & 0xFFFFFF, ids[:16 * S] >> 24, sorted_expert_ids[e]
    ch = 16 * ((c * W + wave_id()) * G + gl.arange(0, G))   # 本 wave 的 G 个 gate 16 行块起点，up 块同号（I + ch）
    if SHARE: ring = gl.allocate_shared_memory(fp8, [2 * SYNC, BM, 64], 行内 16 B 片按 (行 >> 1) & 3 异或错开)   # 只用前 16·S 行
    NK = H // 64
    # 序言：先发 P 步。激活：不共享时每 lane 每子块 16 B（自己的那行）；共享时全 WG 把本步 BM 行 × 64 B 切片（子块 ≥ S 的片
    #   不读），每 lane 读 max(4, BM/W) B，lane 多于片数时多余的 wave 写 LDS 草稿槽。权重：每步 G 条 16 B（lane 0–7 读 gate
    #   块、8–15 读同号 up 块的相邻两行），每 4 步一条 8 B delta（两行各一个 dword）
    xr = {k: cdna3.buffer_load(xq[token, 64 * k:64 * k + 64]) for k in range(P)}
    wr = {k: cdna3.buffer_load(w1[expert, (ch, I + ch), 64 * k:64 * k + 64]) for k in range(P)}
    if SHARE:
        for j in range(SYNC): ring.index(j).store(xr.pop(j))
        gl.barrier()
    acc = gl.zeros([G, 2, 16, 16 * S], f32)          # [块对 j][链 p]：A 的行 R = gate（R < 8）或 up 块的通道 2(R&7)+p
    for k in range(NK):                              # 稳态
        if k + P < NK:                               # 先激活后权重：等激活时不必等后发的权重
            xr[k + P] = cdna3.buffer_load(xq[token, 64 * (k + P):64 * (k + P + 1)])   # A > 1：改为 k % A == 0 时发组 k+2A
            wr[k + P] = cdna3.buffer_load(w1[expert, (ch, I + ch), 64 * (k + P):64 * (k + P + 1)])
        if SHARE:
            if k + SYNC < NK: ring.index((k + SYNC) % (2 * SYNC)).store(xr.pop(k + SYNC))   # 写步 k−SYNC 用过的槽；
                                             # A > 1：改在组末（(k+1) % A == 0）写下一组的 A 个槽
            b = ring.index(k % (2 * SYNC)).load(B 操作数布局)   # [64, 16·S]
        else:
            b = xr.pop(k)                            # 每 lane 每子块 16 B，加载时已是 B 操作数布局
        for j in gl.static_range(G):                 # 每对：两行各从 LDS 表取 LUT、两个 dword 解码，每子块 4 条 MFMA
            for p in gl.static_range(2):
                acc[j, p] = cdna3.mfma(decode(wr[k][j], 行 p), b, acc[j, p])
    g, u = exchange(acc)                             # lane l 与 l^32 换一半（4 条 ds_bpermute）：各持 4 个通道的 gate 与 up
        if SHARE and (k + 1) % SYNC == 0 and k + 1 < NK: gl.barrier()   # 新写的槽可读；读完的槽此后可覆盖
    # 收尾（累加器不经 LDS）
    sx = xs[token][None, :]
    v = swiglu(g * w1_scale[expert, gate 通道] * sx, u * w1_scale[expert, up 通道] * sx)
    if QUANT:                                        # H < 4096、BM > 16 且累加器放得下（8·(I/64)·(BM/16) ≤ 96 个 VGPR）时分派打开：
                                                     #   G = I/64、W = 4，一个 CTA 恰好算完一行的全部 I 个通道
        amax_l = gl.allocate_shared_memory(f32, [BM, W], …)
        amax_l.store(本 wave 每行的 max|v|); gl.barrier()   # lane 内 → xor 16、32
        amax = gl.max(amax_l.load(…), axis=1); s = row_scale(amax)
        route = token * TOPK + slot
        q = to_fp8(gl.clamp(v * rcp(s), -240, 240))  # v·rcp(s)×(1∓2^-20) 两端 RNE 一致即用，否则整 wave 回退 IEEE 除法
        cdna3.buffer_store(q, mq[route, 通道])        # 与 quant_row 逐位一致；每 4 个 fp8 一次 4 B
        cdna3.buffer_store(s, ms[route])             # wave 0 的 lane < 16：每个子块各写自己那一行
    else:
        cdna3.buffer_store(v, mid[token, slot, 通道])   # f32，每 lane 4 个通道一次 16 B；host 随后用 row_quant_fp8 量化
```

**任务划分**（G、W 由分派选：尾部量化时 G = I/64、W = 4，一个 WG 管全部 I；其余 G1，排序块 ≥ 32 且 I%128 = 0 时 W8、否则 W4，
激活都经 LDS 共享；H ≥ 4096 时不尾部量化，轻 wave 加 row_quant_fp8 更快；K ≥ 5120 时 ACT_STEPS = max(16·W/BM, 1)）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/(16·G·W), 排序块数) | x：W·G 个 gate 16 行块与同号 up 块；y：1 个排序块 |
| CTA（64·W 线程） | W 个 wave | 本块 expert 的 2·G·W 个 16 行权重块 × 全部 H |
| wave | G 个 gate 块 + 同号 G 个 up 块 | 本块的 S 个有效 16 行子块（S = 1 … BM/16） |
| lane | l = 16b + i | i < 8：gate 块第 2(i&7)、+1 行，i ≥ 8：同号 up 块的这两行，K 16b…16b+15（每 64-K 16 B）；子块第 i 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每个块对：两行各 1 次 LUT（LDS 表）、2 个 dword 解码（64 位移位），每子块 4 条 MFMA 16x16x32 |

MFMA 的 A 是权重（链 p 的行 R 是 gate（R < 8）或 up 块的通道 2(R&7)+p），B 是 16 个排序行；输出 lane 16q+t 持有排序行 t
的 gate（q < 2）或 up 块 8 个通道 8(q&1) … +7，收尾 lane l 与 l^32 换一半后各持通道 8(q&1)+4(q>>1) … +3 的 gate 与 up。
每个通道的 MFMA 与原先每 lane 8 B 读一块时同序（同一链、同样的 K 分组），结果逐位相同。
M·topk ≤ E 时 grid 的 y 维取 M·topk；排序块里有效行在前，所以有效子块是前缀，kernel 按 S 生成 BM/16 份特化。

**流水线**（预取距离 P 个 64-K 步：S = 1 时 PREFETCH[0]，否则 PREFETCH[1]；W>1（尾部量化还要 G≤4）时激活经 LDS 环共享）
```text
序言（一轮访存）：num_valid、本块 expert、本 lane 各子块的 sorted id → token / slot
  S = 首行有效的子块数（越过 num_valid 为 0）；进入第 S 份特化体，S = 0 直接退出
发 kb = 0 … P−1：激活（共享时每 lane 读 max(4, BM/W) B，否则每子块 16 B）→ G 个块对各 16 B → 每 4 步 delta 1 条 8 B
各 wave 把 LUT 表（delta 0…11 各 8 B）写进 LDS（lane 写一项，内容相同，无需 barrier）
共享：激活(0 … SYNC−1) → LDS 槽 0 … SYNC−1；lgkmcnt(0) + barrier
主循环 kb = 0 … H/64 − 1：
  发 kb+P 的激活与权重（先激活后权重，等激活时不必等后发的权重）
  共享：激活(kb+SYNC) → LDS 槽 (kb+SYNC)%(2·SYNC)；从槽 kb%(2·SYNC) 读本 lane 的 16 B 作 B（行内 16 B 片按 (行号>>1)&3 异或错开）
  不共享：B = 寄存器里的激活(kb)
  for G 个块对 × 2 行：取 delta 第 kb%4 字节（1 条 VALU）→ ds_read_b64 取 LUT → 64 位移位 + 2 个 dword 解成 2 个 fp8 i64
    → 每子块 MFMA ×2（链 p）
  共享：每 SYNC 步一次 lgkmcnt(0) + barrier（读完的槽此后才会被覆盖）；sched_barrier
ACT_STEPS = A > 1：序言只发组 0、1 的激活（每 lane 一条 16 B）并写组 0；组首步发组 g+2，组末步算完后写组 g+1 再 barrier
收尾
  lane l 与 l^32 交换一半累加值（4 条 ds_bpermute / 块对 / 子块）
  acc × 通道因子（gate、up 各自）× xs[token] → [截断] → g · rcp(1 + exp2(−g·log2e)) · u
  不量化：每子块 4 个 f32 一次 16 B store → mid[token, slot, ch … ch+3]（填充行偏移越界，store 被丢弃）
  量化：行 amax（lane 内 4G 个 → xor 16、32）→ LDS [子块][行][wave]；barrier；取 W 个 wave 的最大值
        scale = amax/240（0 取 1）；q = v·rcp(scale) 在 ×(1∓2^-20) 两端 RNE 一致即用，否则整 wave 回退 IEEE 除法
        每 4 个 fp8 一次 4 B store → mq[route, ch … ch+3]；wave 0 的 lane < 16 写 ms[route]
```
关键代码：[前导](stream_a8w4.py#L417)、[按 S 特化](stream_a8w4.py#L879)、[共享激活](stream_a8w4.py#L711)、[ACT_STEPS 取址](stream_a8w4.py#L672)、
[分组发与写](stream_a8w4.py#L763)、[预取](stream_a8w4.py#L762)、[主循环](stream_a8w4.py#L777)、[SiLU 与截断](stream_a8w4.py#L523)、
[尾部量化](stream_a8w4.py#L564)、[夹逼与回退](stream_a8w4.py#L265)、[分派的 tile](../../../moe_a8w4.py#L1381)。

**命名与不变量**：S 有效子块数；P 预取步数；SYNC / RING = 2·SYNC 激活环的步数与槽数；LOADER 每 lane 每步读的激活字节，
LOADERS 读激活的 wave 数（其余 wave 写草稿槽）；A = ACT_STEPS。
1. SYNC ≤ min(PREFETCH)（构建时断言）：步 k 写进环的步 k+SYNC 激活此时已在途。
2. 环槽 (k+SYNC)%RING 在步 k 写、步 k+SYNC 读；每 SYNC 步一次 barrier，保证写前该槽已被所有 wave 读完、读前已写完。
3. ACT_STEPS > 1 只用于共享激活的轻 wave（不尾部量化），且 H/64 是 A 的倍数；此时 SYNC = A。
4. 尾部量化与 quant_row 逐位一致：近舍入边界或 scale 非有限正规数时整 wave 回退 IEEE 除法。

**资源**：LDS = 128 B LUT 表 + 共享时 2·SYNC·BM·64 B 激活环（SYNC1：BM16 2 KiB、BM64 8 KiB，轻 wave 的 SYNC2 翻倍，ACT_STEPS 时
SYNC = A，BM16 / 32 都是 8 KiB；BM/W < 4 时另加 64·4 B 草稿槽，分派用不到）+ 量化时 BM·W·4 B amax；累加器 8·G·S 个 VGPR，
分派只在它 ≤ 96 时让一个 WG 管全部 I 并尾部量化（TP2 BM32 为 80，BM64 为 160 故不量化）。

## A8W4 Down 流式（[stream_a8w4.py](stream_a8w4.py#L1351)）
流式档按排序块算 (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重：I < 512 时（`ATOMIC`）用 bf16x2 原子加直接累加进 prep 清零过的 out，
否则按排序行 bf16 写 routes 给 route_sum。

**Gluon 风格**
```python
# grid = (H/(16·R·W), 排序块数)，64·W 线程（R8·W4 或 R16·W1，见任务划分）
# 放置：本块激活（16·S 行 mq，每行 I B）整块常驻 VGPR：W4（SHARE）时全 WG 分片只读一份进 LDS 再各自读回
#       （LDS_ACTS：BM32 且 I%256 = 0 时留在 LDS、每个 64-K 步再读，省 VGPR、每 CU 多放一个 WG），
#       W1 时直接从 global 读；通道因子一条 16 B 加载后放 LDS（每 wave 1 KiB）；权重按 16 通道块对 global → VGPR（每 lane
#       每 64-K 块一条 16 B：lane 0–7 读块 2u、8–15 读块 2u+1 的相邻两行；H ≥ 5120 时带 non-temporal 位），预取 P/2 个块对
#       （块对 u 算到第 kb 个 64-K 块时发块对 u+P/2 的第 kb 个）；结果不经 LDS：写 routes 每 lane 16 B，或每行一条 bf16x2 原子加
@gluon.jit
def a8w4_down_stream(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                     routes_or_out):
    c, e = gl.program_id(0), gl.program_id(1)        # c：W·R 个 16 通道输出块；e：排序块
    ids = sorted_ids[e * BM + gl.arange(0, BM)]
    S = valid_subblocks(ids, e, num_valid, M)        # 同 a8w4_gateup_stream
    rows = e * BM + gl.arange(0, 16 * S)             # 本块的排序行号
    route = (ids[:16 * S] & 0xFFFFFF) * TOPK + (ids[:16 * S] >> 24)
    expert, w = sorted_expert_ids[e], wave_id()
    ch = lambda u: 16 * ((c * W + w) * R + 2 * u) + gl.arange(0, 32)   # 本 wave 依次算的 R/2 个块对（32 通道）
    if SHARE:                                        # 激活加载先于权重发出：等激活时不必等权重
        A = gl.allocate_shared_memory(fp8, [BM, I + 16], …)   # 行距 I+16：16 行同列的 16 B 读落在不同 bank
        xr = cdna3.buffer_load(mq[route, :])         # 切成 16 B 片分给全 WG 的 lane
    sc = gl.allocate_shared_memory(f32, [W, 256], …)
    fr = cdna3.buffer_load(w2_scale[expert, ch(0)[0]:ch(0)[0] + 16 * R])   # 本 wave 的 R·16 个通道因子
    wr = {u: cdna3.buffer_load(w2[expert, ch(u), :]) for u in range(P // 2)}   # 先发 P/2 对：每对 I/64 × 16 B + delta
    sc.index(w).store(fr)
    if SHARE:
        A.store(xr); gl.barrier()
        a = A.load(B 操作数布局)                     # 每子块 I/64 个 16 B
    else:                                            # W1：激活在前 P 块权重之后、通道因子写进 LDS（要等 fr 返回）之后才发，
        a = cdna3.buffer_load(mq[route, :])          #   算第 0 块前要等齐这些加载；每子块直接读 I/64 个 16 B
    # a[kb, h]：第 kb 个 64-K 块中第 h 条 MFMA 的 B 操作数（两条 MFMA 的 K 按文首的布局交错，不是前后两半）
    row_scale = ms[route] * sorted_weights[rows]     # 中间激活 scale × 路由权重
    if ATOMIC:                                       # 子块里 4 行一组的 token 与 row_scale 经 ds_bpermute 换到 C 布局（lane 16b+j → 行 4b…4b+3）
        tok4, f4 = token[4 * (lane_id() // 16) + gl.arange(0, 4)], row_scale[同上]
    # delta：lane 组 b 与 b^1（相隔 16 个 lane）同属一个 32-K 组，要的 delta（两行各一个 dword）完全相同。于是每两个 256-K
    #   组只发一条 8 B 加载：偶数组取第 q 组、奇数组取第 q+1 组，再 xor-16 交换，各自拿齐，delta 加载减半；
    #   块对 u+1 的交换在算块对 u 时就做，藏住交换延迟
    for u in range(R // 2):                          # 稳态
        acc = gl.zeros([2, 2, 16, 16 * S], f32)      # [行 p][MMA h]：A 的行 R 是块 2u+(R>>3) 的通道 2(R&7)+p；每个 64-K 块
        for kb in gl.static_range(I // 64):          #   的两条 MFMA 各进一条链（与一块一读时同序），末尾 h 两链相加
            if u + P // 2 < R // 2: wr[u + P // 2][kb] = cdna3.buffer_load(w2[expert, ch(u + P // 2), 64 * kb:64 * kb + 64])
            for p in gl.static_range(2):             # 每行：delta 字节 → LDS 表取 LUT → 64 位移位 + 两个 dword 解码
                for h in gl.static_range(2):
                    acc[p, h] = cdna3.mfma(decode(wr[u], kb, p, h), a[kb, h], acc[p, h])   # ATOMIC 时 A、B 互换
        if not ATOMIC:                               # lane 16q+t 持有排序行 t 的通道 8q … 8q+7（链 p 为第 p 个奇偶）
            v = (acc[:, 0] + acc[:, 1]) * sc.index(w).load(…)[32 * u + 8 * q:…+8] * row_scale[t]
            cdna3.buffer_store(v.to(bf16), routes[rows, ch(u)[8 * q:8 * q + 8]])   # RNE；每 lane 16 B；填充行丢弃；之后 route_sum
        else:                                        # lane 16b+j 持有通道 2j、2j+1（链 0、1）× 行 4b … 4b+3
            v = (acc[:, 0] + acc[:, 1]) * sc.index(w).load(…)[32 * u + 2 * j:…+2] * f4
            atomic_add(out[tok4, 通道对 2j], to_bf16(两个值))   # 每行一条（RNE）；一条指令覆盖 4 行各 64 B；token ≥ M 不发
```

**任务划分**（R、W 由分派选：排序块 ≥ 32，或 H ≥ 4096 且 I ≤ 256，且共享激活行放得进 64 KiB LDS 时 R8·W4，否则 R16·W1）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/(16·R·W), 排序块数) | x：W·R 个 16 通道输出块；y：1 个排序块 |
| CTA（64·W 线程） | W 个 wave | 共享时一起把本块 S·16 行 mq 装入 LDS |
| wave | R/2 个块对（R 个 16 通道块） | 每块全部 I；本块激活 S × I/64 × 16 B 驻留 VGPR |
| lane | l = 16b + i | 块 2u（i < 8）或 2u+1 的第 2(i&7)、+1 行，K 16b…16b+15（每 64-K 16 B）；子块第 i 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每子块 4 条 MFMA，分属 4 条累加链（行 p × 第 h 条 MMA；I320 为 5 步） |

每个通道的 MFMA 与原先每 lane 8 B 读一块时同序（链 h 与原来的两条链一一对应，末尾同样两链相加），结果逐位相同。

**流水线**（预取距离 P/2 个块对，P = min(4, R)）
```text
序言：前导同 Gate/Up；进入第 S 份特化体，S = 0 退出
共享：本块 S·16 行 mq（每行 I B）切成 16 B 片分给全 WG 的 lane 发加载（先于权重发出）
通道因子：本 wave R·16 个 f32，一条 16 B 加载（lane·16 < R·64）
各 wave 把 LUT 表写进 LDS（同 Gate/Up）
发前 P/2 个块对的权重：每对 I/64 × 16 B + delta（一条 8 B 取两组：奇数 16-lane 组取奇数组）
通道因子 → LDS 本 wave 的 1 KiB 区
共享：激活片 → LDS（行距 I+16，16 行的 16 B 读落在不同 bank）；barrier；每子块读 I/64 个 16 B → VGPR
不共享：每子块直接从 global 读 I/64 个 16 B → VGPR
行 scale = ms[route] × sorted_weights[排序行]；ATOMIC：每子块 4 行一组的 token 与行 scale 经 ds_bpermute 换到 C 布局
for u in 0 … R/2−1：
  块对 u+1 的 delta 经 xor-16 交换拼出两组（提前一对，藏住交换延迟）
  for kb：发块对 u+P/2 的第 kb 个 64-K 块；两行各：delta 字节 → LDS 表取 LUT → 2 个 dword 解码 → 每子块 2 条 MFMA
    （链 (p, 0) / (p, 1)；ATOMIC 时激活作 A、权重作 B）
  不 ATOMIC：每行两链相加 × 通道因子（LDS，2 条 16 B 读）× 行 scale → 链 0 / 1 两两 RNE 打包成 bf16 → 16 B store
    → routes[排序行, 块对 u 的通道 8q … 8q+7]（填充行丢弃）
  ATOMIC：两链相加 × 通道因子（LDS，一条 8 B 读）× 行 scale → 链 0 / 1 即一对通道 → RNE 成 bf16x2 → agent 范围原子加
    out[token, 通道对]（每子块 4 条；token ≥ M 的填充行不发）
```
关键代码：[共享激活](stream_a8w4.py#L1498)、[通道因子](stream_a8w4.py#L1528)、[权重与 delta](stream_a8w4.py#L1457)、
[delta 交换](stream_a8w4.py#L1474)、[主循环](stream_a8w4.py#L1593)、[写 routes](stream_a8w4.py#L1648)、[原子累加](stream_a8w4.py#L1623)、
[bf16x2 原子加](stream_a8w4.py#L136)、[LDS 预算](stream_a8w4.py#L1337)、[H5120 的权重缓存位](../../../moe_a8w4.py#L1528)。

**命名与不变量**：S 有效子块数；P = min(PREFETCH, R) 预取的通道块数（PP = P/2 个块对）；PAIRS = R/2；REGION 每 wave 1 KiB
通道因子区；KP = I+16 LDS 行距。
1. R 为偶数且 ≤ 16（16 B 加载读块对；一条 16 B 加载放下本 wave 的全部通道因子）。
2. 共享时激活加载先于权重发出；激活写进 LDS 后一次 barrier 才读回；LDS 总量 ≤ 64 KiB 在构建时断言。
3. 每条 8 B delta 只由一个 16-lane 组加载，xor-16 交换给另一组；块对 u+1 的交换在算块对 u 时完成。
4. ATOMIC 时 out 须由 prep（ZERO）先清零；填充行（token ≥ M）不发原子加、不写 routes。

**资源**：LDS = 128 B LUT 表 + W·1 KiB 通道因子 + 共享时 BM·(I+16) B 激活 + W·1 KiB 越界片槽（构建时断言 ≤ 64 KiB；TP2 BM32
为 18.6 KiB，I1024 BM64 要 73 KiB 所以不共享）；激活占 S·I/64·4 个 VGPR（BM64、I320 为 80）；4 条累加链占 16·S 个（一块一读时
8·S，块对多 8·S：DS BM16 R16W1 149→158 VGPR，仍 3 wave/SIMD；BM32 的 R8W4 掉一档，但更快）；ATOMIC 另占 8·S 个（行 token
与行 scale），I = 512 时加上 K512 的激活反而更慢，所以分派只在 I < 512 时用。

## 流式 prep（排序 + 输入量化 + 清零）与 route_sum（[stream_a8w4.py](stream_a8w4.py#L1066)）
流式档（以及 H ≥ 2560、M·topk ≤ 24576（H5120 为 12288）的 prefill 档）用一次 launch 的 prep 代替 aiter moe_sorting 与输入量化：
排序结果格式相同（但同一 token 重复的 expert 每条路由都保留），量化与 quant_row 逐位一致；Down 原子累加时同一次 launch 顺带把
out 清零。I ≥ 512 时 Down 写 routes，route_sum 代替底座 sorted_sum（同顺序同舍入）。折叠 shared expert 时（末节）prep 直接读
[M, topk+1] 的路由、按 SHARED 编译：末列一律不排序（不论 id），其 loc = −1，排序行与块的容量按 routed 列计。

**Gluon 风格**
```python
# prep：grid = n_sort + (QW=4 ? M : ⌈M/4⌉)，256 线程；前 n_sort = ⌈M·TOPK/256⌉ 个 CTA 排序（每个管 256 条路由），其余量化 x
# 放置：排序的计数 cnt、本区间之前的计数 bef 与游标 cur 都在 LDS（每 expert 一个字，另有每 lane 一个哑槽收无效 id）；
#       各排序 CTA 都统计全部路由，所以不用跨 CTA 通信。量化不经 LDS，QW=4 时 4 个 wave 的行 amax 经 LDS 交换
@gluon.jit
def a8w4_stream_prep(topk_ids, topk_weight, x, sorted_ids, sorted_weights, sorted_expert_ids, num_valid, loc_ids,
                     xq, xs, out):
    b, t = gl.program_id(0), thread_id()
    if b < n_sort:
        i = 256 * b + t; my_e, my_w = topk_ids.flat[i], topk_weight.flat[i]   # 本线程的路由先发
        cnt, bef, cur = LDS 清零; gl.barrier()
        for base in range(0, M * TOPK, 4096):        # 每线程 4 条 16 B 读 id
            for e, k in 本轮的 (id, 路由号): atomic_add(cnt[e], 1); atomic_add(bef[e], 1) if k < 256 * b
        gl.barrier()                                 # barrier：计数齐了才扫描
        padded = (cnt + BM - 1) // BM * BM           # 每线程 ⌈E/256⌉ 个连续 expert：线程内求和 → wave 内 6 步 ds_bpermute
        first = exclusive_cumsum(padded)             #   前缀和 → 4 个 wave 的总和经 LDS（一次 barrier）
        cur = first + bef; gl.barrier()              # barrier：游标写完才放置
        pos = atomic_add(cur[my_e], 1)               # 本区间各路由的排序行（同一 expert 内的先后不确定）
        sorted_ids[pos], sorted_weights[pos], loc_ids.flat[i] = token | slot << 24, my_w, pos   # 无效 id：loc = −1
        for e in 本线程的 expert 且 e % n_sort == b:   # 每个 expert 的块号与填充行只由一个排序 CTA 写
            sorted_expert_ids[first[e] // BM:(first[e] + padded[e]) // BM] = e
            sorted_ids[first[e] + cnt[e]:first[e] + padded[e]], sorted_weights[同上] = M, 0
        if b == 0 and t == 0: num_valid[:] = [Σ padded, M]
    else:                                            # QW=4：一个 CTA 一行，wave w 管 512 列块 w, w+4, …；QW=1：一个 wave 一行
        v = cdna3.buffer_load(x[行, 本 wave 的列块])  # 每列块 lane 读 16 B
        amax = wave_max(gl.abs(v))                   # QW=4：再经 LDS 取 4 个 wave 的最大值（barrier）
        s = row_scale(amax)
        q = to_fp8(v * rcp(s) 两端夹逼，不确定时整 wave 回退 v / s) if QW == 4 else to_fp8(v / s)   # 都与 quant_row 逐位一致
        xq[行, 本 wave 的列块], xs[行] = q, s
        if ZERO: out[行, 本 wave 的列块] = 0          # 同一 lane 先读完 x 再写 0，out 可以就是 x

@gluon.jit  # grid = ⌈M·H/512 / 4⌉，256 线程：每个 wave 管一段 = 一个 token 的 512 列，lane 16 B
def a8w4_route_sum(loc_ids, routes, out, M):
    seg = 4 * gl.program_id(0) + wave_id()
    t, cols = seg // (H // 512), (seg % (H // 512)) * 512 + gl.arange(0, 512)
    loc = loc_ids[t, :]                              # lane < TOPK 读一次，再 readlane 逐槽广播成标量
    v = [cdna3.buffer_load(routes[loc[s], cols]) if loc[s] >= 0 else 0.0   # TOPK 条加载同时在途；
         for s in range(TOPK)]                                             #   loc < 0：没有这条路由，不读
    acc = v[0]                                       # f32
    for s in range(1, TOPK):                         # 按槽顺序相加（与 sorted_sum 同序同舍入）
        acc += v[s]
    cdna3.buffer_store(acc.to(bf16), out[t, cols])   # RNE
```

**任务划分**
| kernel | grid × CTA | wave / lane |
|---|---|---|
| `a8w4_stream_prep` 排序 | ⌈M·topk/256⌉ × 256 线程 | 线程放本区间 1 条路由，扫描时管 ⌈E/256⌉ 个连续 expert |
| `a8w4_stream_prep` 量化 | QW=4：M；QW=1：⌈M/4⌉（× 256 线程） | QW=4：wave 管列块 w, w+4…；QW=1：wave 一行；lane 每块 16 B |
| `a8w4_route_sum` | ⌈M·H/512 / 4⌉ × 256 线程 | wave 管一个 token 的 512 列；lane 16 B |

QW 由 host 选：H%2048 = 0 或 M ≤ 256 时 4（列块均分或行少），否则 1（行多时 CTA 少）。排序块 BM 也由 host 选
（[_stream_block_m](../../../moe_a8w4.py#L1574)：H5120 时每 expert 平均 ≤ 12 行一律 16）。

**流水线**
```text
prep 排序 CTA：先发本线程那条路由的 id 与权重；LDS 计数清零；barrier
  直方图：每轮 4096 条路由（每线程 4 条 16 B id 读）→ LDS 原子加 cnt[e]，路由号在本区间之前的再加 bef[e]；barrier
  扫描：padded = ⌈cnt/BM⌉·BM → 线程内求和 → wave 内 6 步 ds_bpermute 前缀和 → 4 个 wave 总和经 LDS（barrier）→ first[e]
  cur[e] = first[e] + bef[e] → LDS；barrier；本线程的路由：pos = LDS 原子加 cur[e] → 写 sorted_ids / sorted_weights / loc_ids
  e % n_sort == 本 CTA 的 expert：写块 expert 号与填充行（token = M、权重 0）；CTA 0 的线程 0 写 num_valid
prep 量化 CTA：读 16 B 片 → lane 内 max → 6 次 shuffle 得 wave 的 amax →（QW=4）LDS 交换 4 个 wave 的 amax；barrier
  scale = amax/240（0 取 1）；QW=4：v·rcp(scale) 在 ×(1∓2^-20) 两端 RNE 一致即用，否则整 wave 回退 IEEE 除法；QW=1：IEEE 除法
  每片 8 B 写 xq；ZERO：同一片的 out 写 0（16 B）；行首 lane 写 xs
route_sum：lane < topk 读 loc_ids[token] → readlane 逐槽广播 → topk 条 16 B 加载同时在途（loc < 0 不读、加 +0）
  f32 按槽顺序相加 → RNE 成 bf16 → 16 B 写 out[token, 列]
```
关键代码：[容量](stream_a8w4.py#L1032)、[直方图](stream_a8w4.py#L1102)、[扫描](stream_a8w4.py#L1115)、[放置](stream_a8w4.py#L1141)、
[量化与清零](stream_a8w4.py#L1158)、[排序 CTA 主体](stream_a8w4.py#L1217)、[route_sum](stream_a8w4.py#L940)。

**命名与不变量**：n_sort 排序 CTA 数；EPT = ⌈E/256⌉ 每线程扫描的 expert 数；CNT / BEF / CUR 三段 LDS 计数（每段后跟每 lane 一个哑槽）。
1. 每个排序 CTA 独立统计全部路由与本区间之前的路由，得到各 expert 的起点与自己的基址，CTA 之间不通信。
2. 计数齐了才扫描、游标写完才放置，各一次 barrier；同一 expert 内的放置顺序由 LDS 原子加决定，不确定。
3. 每个 expert 的块号与填充行只由 e % n_sort 号排序 CTA 写，填充行 token = M、权重 0；越界 id 的路由 loc = −1。
4. ZERO 时每个 lane 先读完自己那片 x 再把同一片 out 写 0，所以 out 可以就是 x。

**资源**：prep 的 LDS 约 3·(256·⌈E/256⌉ + 64) 个字（E256 约 3.8 KiB、E513 约 9.8 KiB）；route_sum 不用 LDS。
排序容量按每个 expert 至多多出 BM−1 个填充行估上界（[stream_prep_sizes](stream_a8w4.py#L1032)），grid 按它发射，多出的块 CTA 退出。

## shared expert 折叠（[shared_a8w4.py](shared_a8w4.py)，入口 `moe_a8w4._SharedFold`）
带 FP8 shared expert（DeepSeek-V4.1，路由末列 = E）的调用，在 routed 路径允许处（`_shared_fold_tier`）不再单独相加：shared
的两次稠密 GEMM（aiter 的 FlyDSL preshuffle FP8 GEMM，`_shared_gemm` 按 M 选 tile；gfx942 上比 CK 默认配置快）吃 routed
路径已量化的 x，中间一步是本文件的 `a8w4_shared_mid`，结果（已乘 shared 的路由权重）写到 routed 路径最后一次规约会加它的地方。
档位只按 routed 列判断；routed kernel 只看前 topk 列（prep 以 SHARED 编译；decode 的 kernel 以 ID_COLS = topk+1 编译，按 `common_a8w4.id_index` 跳过末列，不再拷贝 routed 列）。

| routed 档 | x 的量化来自 | shared Down 写到 | 规约 |
|---|---|---|---|
| decode 槽对 Down（np / pp / wide-dd） | row_quant_decode（np / pp；Gate/Up 随之一律预量化）或计划 kernel（wide-dd） | out 本身 | 槽对 Down 的 SHARED 变体读回、相加后舍入一次 |
| 流式，I_pad ≥ 512 | prep | routes 末尾预留的 M 行，loc[:, topk] = 行号 | route_sum 以 topk+1 列求和 |
| 流式，原子 Down（I_pad < 512） | prep | out 本身（prep 不再清零） | routed 的 bf16x2 原子加在其上（本就不确定） |
| prefill，M ≥ 2048 | prep 或 `_quantize(fill)` | routes 末尾预留的 M 行 | sorted_sum 以 topk+1 列求和（底座 kernel 不改） |

`_moe_shared` 先判混合 decode（M ≤ `DECODE_SHARED_MIXED_MAX_M` = 4，上面两节），再判折叠，所以折叠的 decode 档从 M5 起；
混合不收的单槽 Down（M ≤ 3）、去重 Down（H ≥ 4096 且 I_pad 为 128 的倍数）、prefill M < 2048、shared 要写 out 而 out 未按 16 B
对齐（decode、原子流式）、M > `SHARED_FOLD_MAX_M` = 65536（下述 GEMM 行上限）仍走 ATOM 的 split。

shared 的两次 GEMM 要求 tile_m·tile_k ≥ 每线程 16 B（四波 256 线程、两波（tile_n 32）128 线程；`_shared_gemm` 断言）：preshuffle
GEMM 的 A 按“每线程一个 16 B 片”拷进 LDS，tile 更小时拷贝覆盖的行超出 tile_m、写进另一 LDS stage，别的 wave 还在读时就是竞争
（四波 tile_m 16·tile_k 128：与别的 kernel 共享 CU 时偶发 8 行×16 列块错）。所以 M ≤ 48 的 Down（K = I_pad 640 / 384，tile_k 只能
128）用两波 (16, 32, 128)，速度与 (16, 64, 128) 相同。preshuffle GEMM 最多 65536 行（`gemm_kernels.PRESHUFFLE_M_MAX`：kernel 以
65536 行的 layout 看 A 和 C，超出即报错），所以 `_shared_fold_tier` 在 M > 65536 时退回 split（CK 的 `gemm_a8w8_bpreshuffle` 无此上限）。

**Gluon 风格**
```python
# shared_mid：grid = ⌈M/4⌉，256 线程，一 wave 一行；lane l 读 gate / up 的第 l + 64m 个 16 B 片（8 个 bf16）
@gluon.jit
def a8w4_shared_mid(gate_up, topk_ids, topk_weight, mq, ms, loc, base):   # gate_up bf16 [M, 2I]，mq fp8 [M, KP]
    t = 4 * gl.program_id(0) + wave_id()
    live = topk_ids[t, 0] >= 0 and topk_ids[t, TOPK] == E              # ATOM 的规则（首列屏蔽连 shared 一起丢）
    w = topk_weight[t, TOPK] if live else 0
    g, u = swiglu 截断(gate_up[t, :I].to(f32), gate_up[t, I:].to(f32))
    h = silu(g) * u if live else 0                                       # 丢弃的行整行为 0：Down GEMM 得 +0
    q, s = quant_row(h)
    mq[t, :I] = q; mq[t, I:KP] = 0                                       # KP = Down 权重补零后的 K（640 / 384）
    ms[t] = s * w                                                        # 路由权重并进行 scale
    if LOC: loc[t, TOPK] = base + t if live else -1                       # 预留 routes 行
```

**次序（每档同一模式）**：x 量化的 kernel 之后记一个事件（`start`），侧流等它，但侧流的 GEMM → shared_mid → GEMM 在 routed
的下一个 kernel（Gate/Up）发出之后才发出（`fork`，`SHARED_FOLD_FORK_AFTER`：graph 重放时 fork 不再把 Gate/Up 推后，decode
M8–32 为 0.89–0.99 倍）；读 shared 结果的 kernel 之前 join。输出就地（out = x）时只有 np 档的 Gate/Up 在 fork 之后还读 x（仅
`SHARED_FOLD_DECODE_SIDE_QUANT` = 1 时，Gate/Up 不预量化）：shared 的 Down GEMM 先 hold，Gate/Up 发出后再等一次主流（`release`）。prefill 的 shared_mid 在 `_quantize(fill)` 之后写 loc 末列，gemm2 只写 routed 列。
侧流用到 `SHARED_FOLD_DUAL_STREAM_MAX_M`（1024），更多 token 在当前流上串行。graph 重放时每条跨流边（fork、join 各一条）
在 routed 关键路径上约 5–6 µs，侧流链本身藏在 routed Gate/Up 后面（只让它慢约 2 µs）：对串行，decode M8–32 快 2–10%、流式持平
到快 3%；与 bench 的 split 参考（也是一 fork 一 join、不做最后相加）在流式档持平。fork 挪到调用开头（侧流自己量化 x）没有收益。

**不变量**：丢弃 shared 的行 mq 全 0、ms = 0 → Down 输出 +0，整行屏蔽仍逐位 +0；route_sum / sorted_sum 对 shared 行的 f32
求和顺序在 routed 之后、只舍入一次；除原子 Down 外结果逐位确定（graph 重放 = eager）。
