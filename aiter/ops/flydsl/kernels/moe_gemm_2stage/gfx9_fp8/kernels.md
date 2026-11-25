# gfx9_fp8 的 A8W4 MoE：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。
     本目录的其他 kernel（gemm1.py、gemm2*.py、quant.py、moe_reduce.py）是 FP8 / BF16 底座，不在本文范围。 -->

A8W4 的 GEMM kernel（prefill 与 decode 的 Gate/Up、Down）共用 [common_a8w4.py](common_a8w4.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；lane `16b + r` 读第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32 的操作数。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](common_a8w4.py#L64) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](common_a8w4.py#L79) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼。
- decode kernel 每 lane 一条 16 B 加载读两个相邻 64-K 块（块对）或两个槽的同一块（槽对），LUT 从 LDS 里预建的表按 delta 字节取；
  lane 映射见 decode_a8w4.py 的 docstring（[块对](decode_a8w4.py#L23)、[槽对](decode_a8w4.py#L33)）。
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
# 排序输出（aiter Opus moe_sorting；int32，权重 f32）：排序块 e = 同一 expert（sorted_expert_ids[e]）的 BM 个排序行；
#   sorted_ids[排序行] = token | slot << 24，token ≥ M 的是填充行（只在每个 expert 的末块尾部）；sorted_weights[排序行] = 路由权重；
#   num_valid = 排序行总数（含填充行）：grid 按排序块数的上界发射，e·BM ≥ num_valid 的 CTA 直接退出；
#   loc_ids[token, slot] = 该路由排序后的行号（−1 表示没有这条路由，求和时跳过；同一 token 重复选同一 expert 时
#   Opus 排序只保留最后一个槽）。
# w1 / w2：fp4 权重码，w1_delta / w2_delta：delta 字节；w1_scale / w2_scale [E, 通道]：通道因子 2^(cmax − 132)，收尾才乘。
# —— Gluon 的真实操作（gl = triton.experimental.gluon.language，cdna3 = gl.amd.cdna3）——
# r = cdna3.buffer_load(t[下标])   global → VGPR；结果到第一次使用才等待，循环里先发后面几步的加载再算这一步就是预取
#                                  （decode_a8w4.py 读激活行用普通 global load，不查越界：越过行尾的 lane 钳到最后一片或不发）
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
# —— 语义 ——
# silu(g) = g · rcp(1 + exp2(−g·log2e))；swiglu(g, u) = silu(g)·u，g、u 已乘完通道因子与激活 scale；
#   SwiGLU 截断 L > 0（编译期常量，kernel 名加 _swiglu）时先 g = min(g, L)、u = clamp(u, −L, L)；L = 0 时不发指令
# row_scale(amax) = amax / 240（IEEE 除法），结果为 0 时取 1；quant_row(v) → (q, s)：s = row_scale(max|v|)，q = to_fp8(v / s)
#   行量化都按 quant_row 逐位一致：row_quant_fp8 是 prefill 用的通用版，row_quant_decode 是 decode 档用的版本
# sorted_sum（底座 moe_reduce.py）：out[t, :] = Σ_slot routes[loc_ids[t, slot], :]（f32 按槽顺序相加后转 bf16）
# 调用顺序（host 见 ../../../moe_a8w4.py；给了 tile 参数时一律 prefill）：
#   decode（M ≤ 32，H ≥ 4096 且 I%128 = 0 时 M ≤ 20；H ≤ 4096 或 5120，decode kernel 放得下；不排序，也没有 sorted_sum）：
#     [row_quant_decode(x)：M·TOPK·I/16 ≥ 2560] → gateup_decode → [row_quant_decode(mid)：M ≥ 4、w2 < 4 GiB、槽对版放得下]
#     → down_decode(_pq) → out
#   prefill（其余）：moe_sorting → row_quant_fp8(x；顺带把 loc_ids 置 −1) → gateup_prefill → row_quant_fp8(mid) → down_prefill
#     （写有效路由的 loc_ids）→ sorted_sum → out；H5120 且 M·TOPK ≥ 16384 时两个 GEMM 按 XCD 分块（XCD_CHUNK）；I > 512 且估计
#     WG > 80 时 down_prefill 的激活分两段（TWO_PASS）
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

**XCD 分块**（`XCD_CHUNK` = C > 0，[Gate/Up](gemm1_a8w4.py#L646)、[Down](gemm2_a8w4.py#L125)）：MI308X 把一次 launch 的 WG L 发到 XCD L % 4
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

## A8W4 Down prefill（[gemm2_a8w4.py](gemm2_a8w4.py#L105)）
按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同底座 `gemm2.py` 的默认 prefill Down。

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
关键代码：[gather](gemm2_a8w4.py#L181)、[半块分支](gemm2_a8w4.py#L424)、[通道因子视图](gemm2_a8w4.py#L210)、
[激活进 VGPR 与第二段](gemm2_a8w4.py#L249)、[行 scale](gemm2_a8w4.py#L282)、[load_block](gemm2_a8w4.py#L320)、
[compute](gemm2_a8w4.py#L350)、[bf16 打包](gemm2_a8w4.py#L95)、[写 LDS](gemm2_a8w4.py#L377)、[写 routes](gemm2_a8w4.py#L390)、
[主循环](gemm2_a8w4.py#L401)、[两段的划分](gemm2_a8w4.py#L66)、[1-D grid](gemm2_a8w4.py#L474)。

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

## A8W4 Gate/Up decode（[decode_a8w4.py](decode_a8w4.py#L242)）
M ≤ 32（H ≥ 4096 且 I%128 = 0 时 ≤ 20）且形状受支持（H%512 = 0，见 [gateup_decode_fits](decode_a8w4.py#L59)；host 只放行
H ≤ 4096 与 H5120）时不排序：每个 WG 算一条路由 (token, slot) 的 16 个 gate 与同号 16 个 up 通道，写 mid[token, slot, ch]（f32）。
launcher 的 plan / y 两个参数位不读（留给按 expert 去重的 decode）。

**Gluon 风格**
```python
# grid = (I/16, M·TOPK)，256 线程（4 个 wave）；不排序
# 放置：x 的一行（H 个 fp8）放 LDS，4 个 wave 共读（A 操作数是广播读）；LUT 表（delta 0–11 各 8 B）放 LDS；权重 global →
#       本 wave 的 VGPR，序言一次全部发出（H2560 每 lane 10 条 16 B），不分 tile、无双缓冲，第 p 个块对只等自己那几条加载；
#       4 个 wave 的部分和经 LDS 相加
@gluon.jit
def a8w4_gateup_decode(x, xq, xs, topk_ids, w1, w1_delta, w1_scale, mid):
    j, route = gl.program_id(0), gl.program_id(1)    # j：16 个 gate 通道及同号 up 通道；route：一条路由
    token, slot, expert = route // TOPK, route % TOPK, topk_ids[route]
    w = wave_id()                                    # w%2：gate / up；w//2：K 前半 / 后半
    ch = 16 * j + gl.arange(0, 16)                   # 本 CTA 的 16 个 gate 通道；up 用同号的 I + ch
    wrows = (w % 2) * I + ch                         # 本 wave 的 16 个权重行
    kb0 = (w // 2) * (H // 128)                      # 本 wave 那半段 K 的第一个 64-K 块
    lut = gl.allocate_shared_memory(u8, [96], …); lut.store(LUT 表)
    xl = gl.allocate_shared_memory(fp8, [H], …)
    # 序言：所有加载一次发出；x 行不依赖 expert，最先发（预量化版线程 t 读 16 B 片 t + 256u，H5120 分两轮）
    xr = cdna3.buffer_load(xq[token, :] if PREQUANT else x[token, :])
    wr = [cdna3.buffer_load(w1[expert, wrows, 64 * (kb0 + 2 * p):64 * (kb0 + 2 * p + 2)])   # 块对 p：每 lane 16 B；
          for p in range(H // 256)]          # p 为偶数时先发一条 w1_delta 的 8 B（两行 × 4 个 64-K 块，供块对 p、p+1）
    cs = w1_scale[expert, ch], w1_scale[expert, I + ch]   # 收尾的通道因子也在序言读（tid < 16），收尾不再等访存
    if PREQUANT:                                     # 编译期开关：M·TOPK·I/16（CTA 数）≥ 2560 时 host 先用 row_quant_decode 量化 x
        xl.store(xr); s = xs[token]
    else:                                            # CTA 少时每个 CTA 自己量化 bf16 的 x 行，省一次 launch
        red = gl.allocate_shared_memory(f32, [4], …)
        red.index(w).store(wave_max(gl.abs(xr))); gl.barrier()   # 每个 wave 只持有 x 行的一部分；barrier 保护 4 个 wave 的 max
        amax = gl.max(red.load(…)); s = row_scale(amax)
        xl.store(to_fp8(gl.clamp(xr / s, -240, 240)))
    gl.barrier()                                     # x 行写完 LDS 才可读
    # 主循环：本 wave 那半段 K，共 H/256 个块对
    acc = gl.zeros([2, 16, 16], f32)                 # h = 0/1：lane 持有的两个权重行
    for p in range(H // 256):
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
```

**任务划分**（数字按 H2560、I320）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·topk) | x：16 个 gate 通道块 j 与同号 up 块；y：一条路由 |
| CTA（256 线程） | 4 个 wave | 该路由 expert 的 32 行权重 × 全部 H |
| wave | w%2：gate / up；w/2：K 前半 / 后半 | 16 行 × H/2（10 个块对） |
| lane | l | 块对中第 l/32 块的行 2(l%8)、2(l%8)+1，K 组 (l%32)/8，共 16 B |
| K 步 | 块对（相邻两个 64-K 块） | 每 wave 4 条 MFMA 16x16x32（2 行 × 2 个 K32） |

**流水线**（不分 tile：序言一次发出本 wave 的全部加载，主循环按序消费；无双缓冲）
```text
序言
  tid ≤ 11：LUT 表 → LDS lut（delta d 在字节 8d：|e2m1| 0–3、4–7 的 fp8 各一个 dword）
  expert = topk_ids[route]；本 wave 的 16 行块 rb = (w%2)·I/16 + j 与 K 起点 kb0
  发 x 行加载：预量化版线程 t 读片 t + 256u（片号 < H/16；H5120 两轮，第二轮只有 t < 64）各 16 B fp8；核内量化版每线程读 H/512 个 bf16 对
  for p in 块对（sched_barrier 固定顺序）：p 偶时 delta dwordx2（两行 × 4 块）；权重 dwordx4
  tid < 16：通道因子 ws[gate][ch]、ws[up][ch]
x 行 → LDS xq
  预量化版：各片 16 B → LDS；barrier；scale = x_scale[token]
  核内量化版：lane 内 max|x| → DPP 4 步 + readlane 4 行 → LDS red[wave]；barrier
              scale = max(red) / 240（IEEE，0 取 1）；q = RNE(clamp(x/scale)) → 每线程 2 B → LDS；barrier
主循环 p = 0 … H/256 − 1（第 p 个块对只等到自己的加载）
  xop = LDS xq 第 kb0+2p+l/32 块、K 组 (l%32)/8 的 16 B（2 个 K32 的 A 操作数）
  每行 h：LUT 对 = LDS lut[delta 字节]；fp4 经 64 位移位 + v_perm 查表 → 2 个 fp8 i64
  MFMA 16x16x32 × 4：acc[h] += xop[c] · w[h][c]（c = 0, 1）
收尾
  lane 0–7（行 0 列 0–7）、lane 40–47（行 8 列 8–15）：acc[h][0] → LDS part[wave, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8；gate = wave 0 + wave 2，up = wave 1 + wave 3
  × 通道因子 × scale → [截断] → gate · rcp(1 + exp2(−gate·log2e)) · up → f32 写 mid[route, j·16 + tid]
```
关键代码：[LUT 表](decode_a8w4.py#L177)、[x 行加载](decode_a8w4.py#L305)、[权重加载](decode_a8w4.py#L333)、[片写 LDS](decode_a8w4.py#L369)、
[核内量化](decode_a8w4.py#L378)、[主循环](decode_a8w4.py#L408)、[块解码](decode_a8w4.py#L208)、[收尾](decode_a8w4.py#L424)。

**命名**：rb 本 wave 的 16 行权重块号；kb0 本 wave 那半段 K 的首个 64-K 块；块对 p = 64-K 块 kb0+2p、kb0+2p+1；raws[p] 块对 p 的
16 B，deltas[p/2] 块对 p、p+1 共用的 delta 8 B；accs[h] 行 2m+h（h = 0/1）的累加器。

**流水线不变量**
1. x 行加载先于全部权重加载发出（它不依赖 expert），每条 delta 加载紧挨它覆盖的两个块对之前（sched_barrier 固定），第 p 个块对只等自己那几条。
2. 通道因子在序言由 tid < 16 读入，收尾不再访存。
3. LDS xq 写完后有一次 barrier 才进主循环；核内量化版另有一次 barrier 保护 4 个 wave 的 max（red）。
4. 只有 lane 0–7、40–47 的 acc[h][0] 有效（A 与 B 的 K 组对齐的输出），写 part 后 barrier，tid < 16 才合并。

**资源**：LDS 约 3.2 KiB（xq H B、part 512 B、red 16 B、LUT 96 B；H5120 多 2.5 KiB）；本 wave 的权重全部同时在途
（H2560 每 lane 10 条 16 B，H5120 为 20 条）。

## A8W4 Down decode（核内量化，[decode_a8w4.py](decode_a8w4.py#L489)）
M ≤ 3、w2 ≥ 4 GiB 或槽对版放不下时用：每个 WG 算一个 token 的 16 个输出通道，在 LDS 里把 topk 个 expert 相加后直接写 bf16 out。
它的 LDS 放不下时（[down_decode_fits](decode_a8w4.py#L65)）decode 整体不用，M ≤ 32 走 prefill。

**Gluon 风格**
```python
# grid = (H/16, M)，64·W 线程：W = TOPK 个 wave（TOPK > 16 时 W = 4），R = ⌈TOPK/W⌉ 轮
# 放置：wave w 管槽 w, w+W, …：mid 行（f32）global → VGPR（lane 交错读列 l + 64u），wave 内量化后写 LDS mq[w, r]，
#       再按 A 操作数布局读回（经 LDS 是为了换布局，只有本 wave 读）；权重、通道因子与路由权重 global → VGPR；
#       全部加载在序言一次发出，没有循环预取；LUT 在 LDS。W 个 wave 的部分和经 LDS 相加后直接写 bf16 out
@gluon.jit
def a8w4_down_decode(mid, topk_ids, topk_weight, w2, w2_delta, w2_scale, out):
    j, token = gl.program_id(0), gl.program_id(1)    # j：16 个输出通道
    w, n = wave_id(), 16 * j + gl.arange(0, 16)
    slot = [min(w + W * r, TOPK - 1) for r in range(R)]    # 越界槽钳到 TOPK−1，贡献置零
    expert = [topk_ids[token, q] for q in slot]
    tw = [topk_weight[token, q] for q in slot]       # 路由权重，与 expert 一起最先读
    lut = gl.allocate_shared_memory(u8, [96], …); lut.store(LUT 表)
    mq_l = gl.allocate_shared_memory(fp8, [W, R, I], …)
    # 序言：mid 行先发（不依赖 expert），再发权重（每 4 个 64-K 块一条 delta，排在其前）与通道因子
    mr = [cdna3.buffer_load(mid[token, q, :]) for q in slot]   # lane 读 I/64 个 f32（列 l + 64u）
    wr = [cdna3.buffer_load(w2[expert[r], n, :]) for r in range(R)]   # 连同 w2_delta；每 64-K 块每 lane 8 B（同 prefill）
    cs = [cdna3.buffer_load(w2_scale[expert[r], n]) for r in range(R)]   # 通道因子
    sa = []
    for r in range(R):                               # wave 内量化本槽的 mid 行
        amax = wave_max(gl.abs(mr[r])); sa.append(row_scale(amax))
        mq_l.index(w).index(r).store(to_fp8(gl.clamp(mr[r] / sa[r], -240, 240)))
    gl.barrier()                                     # mq 写完才读
    total = gl.zeros([16], f32)
    for r in range(R):
        acc = gl.zeros([16, 16], f32)                # A 的 16 行都是本槽的 mq，结果取行 0
        for kb in gl.static_range(I // 64):          # 每个 64-K 块 2 条 MFMA
            a = mq_l.index(w).index(r).load(A 操作数布局)[:, 64 * kb:64 * kb + 64]
            acc = cdna3.mfma(a, decode(wr[r], kb, lut), acc)
        total += gl.where(w + W * r < TOPK, acc[0] * cs[r] * sa[r] * tw[r], 0)
    part = gl.allocate_shared_memory(f32, [W, 16], …)
    part.index(w).store(total); gl.barrier()         # barrier：W 个 wave 的部分和写完才合并
    out[token, n] = tree_sum(part.load(…), axis=0).to(bf16)   # RTE；直接写 out，不需清零，也没有 sorted_sum
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/16, M) | x：16 个输出通道块 j；y：一个 token |
| CTA（64·W 线程） | W = topk 个 wave（topk > 16 时 W = 4） | 该 token 的全部 topk 个槽 |
| wave | 槽 w, w+W, …（越界槽钳到 topk−1、贡献置零） | 该槽 expert 的 16 行 × 全部 I |
| lane | l = 16b + r | 行 r 的 K 组 b（每块 8 B，同 prefill）；mid 列 l + 64u |
| K 步 | 64-K 块 | 2 条 MFMA 16x16x32（I320 为 5 块） |

**流水线**
```text
序言
  tid ≤ 11：LUT → LDS；每轮 r：slot = w + W·r、expert、路由权重
  发 mid 行加载：lane 读 I/64 个 f32（列 l + 64u）
  发权重加载：每 64-K 块 1 条 dwordx2，每 4 块 1 条 delta dword（排在其前）；通道因子 ws[ch]
wave 内量化本槽 mid 行
  amax：lane 内 → DPP + readlane；scale = amax / 240（IEEE，0 取 1）
  q = RNE(clamp(mid/scale)) → 逐字节写 LDS mq[w, r]；barrier
主循环 r、kb = 0 … I/64 − 1
  xop = LDS mq[w, r] 第 kb 块、K 组 l/16 的 16 B；LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64
  MFMA 16x16x32 × 2 → acc；行 0（lane < 16 的 acc[0]）即本槽的 16 个通道
  total += 有效 ? acc[0] × 通道因子 × scale × 路由权重 : 0
收尾
  lane < 16：total → LDS part[w]；barrier
  tid < 16：W 个 wave 的部分和树形相加 → bf16（RTE）→ out[token, j·16 + tid]（不需清零与 sorted_sum）
```
关键代码：[槽与 expert](decode_a8w4.py#L543)、[加载](decode_a8w4.py#L552)、[wave 内量化](decode_a8w4.py#L588)、
[主循环](decode_a8w4.py#L614)、[收尾](decode_a8w4.py#L629)。

**命名与不变量**：slots[r] / valid[r] 第 r 轮的槽（钳过）与是否有效；mids[r] 本 lane 的 I/64 个 mid 值；raws[r][i] 第 i 个 64-K 块的 8 B。
1. mid 行加载先于权重加载发出；每条 delta dword 紧挨它覆盖的 4 个块之前。
2. 越界槽钳到 topk−1（读的都是合法地址），贡献由 valid 置零；每个 wave 只读自己写的 mq 行，barrier 后才读。
3. 部分和按固定顺序 tree_sum：同输入的输出逐位确定。

**资源**：LDS = W·⌈topk/W⌉·I B（mq）+ 64·W B（part）+ 96 B，TP2 topk11 约 4.2 KiB。

## A8W4 Down decode 槽对（预量化，[decode_a8w4.py](decode_a8w4.py#L679)）
M ≥ 4、w2 < 4 GiB 且放得下（每行 I/16 个 16 B 片 ≤ 64·W 个线程，全零行由每线程写一片；LDS 够）时用：读预量化的 mq fp8 与 ms；
一条 16 B 加载覆盖两个路由槽的同一 64-K 块，其余同核内量化版。

**Gluon 风格**
```python
# grid = (H/16, M)，64·W 线程：W = min(⌈TOPK/2⌉, 3) 个 wave，R = ⌈⌈TOPK/2⌉/W⌉ 轮，每轮一个槽对
# 放置：同核内量化版（mq 片 global → VGPR → LDS，权重与通道因子在序言一次发出直进 VGPR，LUT 在 LDS），只是 mq 已由前一个
#       row_quant_decode 量化好；一个 wave 同时算一个槽对：lane 0–31 读低槽、32–63 读高槽 expert 的同一 64-K 块，
#       一条 16 B 加载、一组 MFMA 推进两个槽
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
    wr = [cdna3.buffer_load(w2[expert[r][half], n, :]) for r in range(R)]   # 每 64-K 块一条 16 B；每 4 块一条 delta 8 B
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
                vals[h] += gl.where(ok, acc[h][行 s 与 8+s] * cs[r][s][本列的通道] * sm[r][s] * tw[r][s], 0)
    part = gl.allocate_shared_memory(f32, [W, 2, 16], …)
    part.index(w).store(vals 的行 0/1 列 0–7 与行 8/9 列 8–15); gl.barrier()
    out[token, 16 * j + 2 * m + h] = tree_sum(part[:, h, m] + part[:, h, m + 8]).to(bf16)   # RTE，同 a8w4_down_decode
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
    每 64-K 块 1 条 dwordx4；每 4 块 1 条 delta dwordx2；两槽的通道因子各 1 条 dwordx2
mq 片 → LDS 行 [w, r, 半]（只写有效片）；barrier
主循环 r、kb = 0 … I/64 − 1
  xop = 保留的 lane ? LDS 本槽行第 kb 块、K 组 (l%32)/8 的 16 B : 零行
  每行 h：LUT 对 = LDS lut[delta 字节]；解码 → 2 个 fp8 i64；MFMA × 4：acc[h] += xop[c] · w[h][c]
  vals[h] += Σ 有效槽 s：acc[h][s] × 通道因子[s][h] × ms[s] × 路由权重[s]（[0] 低槽、[1] 高槽）
收尾
  lane 0–7、40–47：vals[h] → LDS part[w, h]；barrier
  tid < 16（通道 2m+h）：各 wave 取列 m + 列 m+8 → 树形相加 → bf16（RTE）→ out[token, j·16 + tid]
```
关键代码：[零行](decode_a8w4.py#L729)、[槽对](decode_a8w4.py#L737)、[mq 片](decode_a8w4.py#L753)、
[整张量寻址](decode_a8w4.py#L774)、[屏蔽 A 操作数](decode_a8w4.py#L826)、[主循环](decode_a8w4.py#L832)、[收尾](decode_a8w4.py#L858)。

**命名与不变量**：ROWS = 2·W·R 个暂存行，之后是零行；upper = l ≥ 32（高槽半）；keep = (l%2 == l/32)；chan_scale[q][h] 槽 q 的行 2(l%8)+h。
1. 零行（tid < I/16 各写一片）与 mq 片都在同一次 barrier 之前写完；越界片读最后一片、不写。
2. expert 号 readfirstlane 成标量，供通道因子的标量偏移；权重偏移是 32 位，host 只在 w2 < 4 GiB 时选本 kernel。
3. 被屏蔽 lane 读零行：行 s / 8+s 只含槽 s，两槽各乘自己的通道因子、ms 与路由权重后才相加。
4. 部分和按固定顺序 tree_sum：同输入的输出逐位确定。

**资源**：LDS = (2·W·轮数 + 1)·I B + 128·W B + 96 B，TP2 topk11 约 4.5 KiB；
每 lane 偏移是 32 位，w2 总字节数须 < 4 GiB（否则 host 改用核内量化版）。

## 行量化 decode（[decode_a8w4.py](decode_a8w4.py#L913)）
槽对 Down 之前量化 mid（f32 [M·topk, I]），M·topk·I/16 ≥ 2560 时量化 x（bf16 [M, H]）；逐元素 IEEE 除法，
结果与 [quant_a8w4.py](quant_a8w4.py) 逐位一致。

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
    cdna3.buffer_store(to_fp8(gl.clamp(v / s, -240, 240)), q[r, :])   # 逐元素 IEEE 除法，与 a8w4_row_quant_fp8 逐位一致
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
q = clamp(x / scale, ±240)（IEEE）→ cvt_pk_fp8（RNE）→ -0 改 +0 → 每 4 个打包成 32 位字
有效片写回：bf16 每片 8 B、f32 每片 4 B；lane 0 写 scale
```
关键代码：[读片](decode_a8w4.py#L936)、[scale](decode_a8w4.py#L959)、[写回](decode_a8w4.py#L962)、[-0 处理](decode_a8w4.py#L150)。

**资源**：不用 LDS。
