# gfx9_fp8 的 A8W4 MoE：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。
     本目录的其他 kernel（gemm1.py、gemm2*.py、quant.py、moe_reduce.py）是 FP8 / BF16 底座，不在本文范围。 -->

A8W4 的 prefill 两个 GEMM（Gate/Up、Down）共用 [common_a8w4.py](common_a8w4.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；lane `16b + r` 读第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32 的操作数。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](common_a8w4.py#L64) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](common_a8w4.py#L79) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼。
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
# cdna3.buffer_store(v, t[下标])   VGPR → global；偏移越界（如填充行）的写被硬件丢弃
# L = gl.allocate_shared_memory(类型, [槽数, …], 布局)；L.index(i).store(r) / .load(布局)：VGPR ↔ LDS
# gl.barrier()                     CTA 内同步：之前写进 LDS 的数据对所有 wave 可见，之前读完的槽此后可以覆盖
# acc = cdna3.mfma(a, b, acc)      fp8 MFMA 16x16x32、f32 累加：acc[行, 列] += a[行, K]·b[K, 列]
# —— 伪操作 ——
# wave_id() / lane_id()：本 wave / lane 的编号；uniform(v)：readfirstlane，把 wave 内相同的值变成标量，供整 wave 分支
# wait_vmem()：等此前发出的全部 global 加载到齐（s_waitcnt vmcnt(0)）
# decode(wr, …)：寄存器里的 fp4 码 + delta 字节 → fp8 操作数（v_perm 查表）；rcp(v)：硬件近似倒数
# to_fp8(v)：RNE 转 fp8，-0 存成 +0；xcd_work(L)：XCD 分块 1-D grid 的工作编号（见 Gate/Up 一节）
# —— 语义 ——
# silu(g) = g · rcp(1 + exp2(−g·log2e))；swiglu(g, u) = silu(g)·u，g、u 已乘完通道因子与激活 scale；
#   SwiGLU 截断 L > 0（编译期常量，kernel 名加 _swiglu）时先 g = min(g, L)、u = clamp(u, −L, L)；L = 0 时不发指令
# row_scale(amax) = amax / 240（IEEE 除法），结果为 0 时取 1；quant_row(v) → (q, s)：s = row_scale(max|v|)，q = to_fp8(v / s)
# sorted_sum（底座 moe_reduce.py）：out[t, :] = Σ_slot routes[loc_ids[t, slot], :]（f32 按槽顺序相加后转 bf16）
# 调用顺序（host 见 ../../../moe_a8w4.py，每个 M 都走这一条）：
#   moe_sorting → row_quant_fp8(x；顺带把 loc_ids 置 −1) → gateup_prefill → row_quant_fp8(mid) → down_prefill（写有效路由的
#   loc_ids）→ sorted_sum → out；H5120 且 M·TOPK ≥ 16384 时两个 GEMM 按 XCD 分块（XCD_CHUNK）；I > 512 且估计 WG > 80 时
#   down_prefill 的激活分两段（TWO_PASS）
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
