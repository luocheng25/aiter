# mega_moe_tp_gfx942：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。 -->

`MegaMoeTPGfx942.forward` = 本卡的 A8W4 两阶段 MoE（下面各 kernel，源码在 [kernels/moe_a8w4/](../moe_a8w4/)，host 分派在
[moe_a8w4.py](../../moe_a8w4.py)）+ 一次 all-reduce（aiter `CustomAllreduce` 或 RCCL，没有自己的通信 kernel）。
流式与 prefill 用到的 `sorted_sum` 与公共工具取自 aiter 的 [gfx9_fp8](../moe_gemm_2stage/gfx9_fp8/) 两阶段 MoE。

A8W4 的 GEMM kernel（prefill 两个、decode 四个（三个逐路由 / token、一个按 expert 去重的 Gate/Up）、流式两个）共用 [common.py](../moe_a8w4/common.py) 的私有权重布局与解码（字节级定义见该文件 docstring）：
- 权重 q：每 16 行 × 64-K 一个 512 B 块；lane `16b + r` 读第 r 行 K `16b … 16b+15` 的 8 B，dword c 即第 c 条 MFMA 16x16x32 的操作数。
- delta d：每行每 32-K 组一个字节 `min(cmax − code, 11) × 8`；lane 每 4 个 64-K 块读 1 个 dword，第 kb%4 字节给第 kb 块。
- 解码：[fp8_lut](../moe_a8w4/common.py#L63) 用 1 次 v_perm 广播 delta、2 次减法得 8 项 |e2m1|→fp8 表；
  [fp4x8_to_fp8](../moe_a8w4/common.py#L74) 每个 dword 用 2 次 v_perm 查表得 8 个 fp8，符号位另拼（流式 kernel 用同样的两次 v_perm）。
- 解出的值是 |e2m1| × 2^(5−delta)，delta ≤ 11 时精确；通道因子 s = 2^(cmax−132)（f32）与激活 scale 在 epilogue 一起乘。
- Gate/Up decode、槽对 Down 与去重 Gate/Up（[decode.py](../moe_a8w4/decode.py)、[decode_dedup.py](../moe_a8w4/decode_dedup.py)）每 lane 一条
  16 B 加载，取布局里相邻两个 lane 的 8 B（行 2(l%8) 与 +1）；核内量化的 Down decode 与 prefill 一样每 lane 每 64-K 块 8 B。
  decode kernel 都把一行一个块的两个 dword 的移位合成一次 64 位移位，LUT 都预先算好放在 LDS（delta 0–11 各 8 B，用 delta 字节直接寻址）。

## 先读：Gluon 风格的分阶段写法
每个 kernel 一节，先给一段 Gluon 风格的函数，一个 program = 一个 CTA。不用纯 Triton 的原因：Triton 的 `tl.load` / `tl.dot`
只描述块级张量算什么，数据经不经 LDS、留在哪些寄存器、提前几步加载、在哪里 barrier、wave 之间怎么分工，都由编译器按
num_stages / num_warps 决定，源码里没有地方写，而这些正是这几个 kernel 的设计要点。Gluon（Triton 的底层方言）把它们交给
写 kernel 的人：自己分配 LDS、自己发 buffer_load 进寄存器、自己放 barrier。所以每段先用注释写**放置**（每个操作数的路径：
global → VGPR 直读，还是经 LDS 共享 / 双缓冲）与**预取距离**，再按序言 / 稳态 / 收尾写出各阶段；Gluon 没有的操作写成伪操作。
指令级的细节（sched_group 排布、s_waitcnt 计数、lane 取址）仍在后面的逐拍流水线。
只求看懂、不求能运行：张量下标代替 offsets 运算，省略 mask、布局对象与类型转换，参数用 host 侧的张量名。共用的记号：
```python
# —— 张量 ——
# M：token 数；H：hidden；I：中间维（补零后）；E：expert 数；TOPK：每个 token 的路由数。路由 route = token·TOPK + slot。
# x / out：bf16 [M, H] 的输入与输出。xq / xs：x 按行量化后的 fp8 [M, H] 与 f32 scale [M]。
# mid：f32 [M, TOPK, I]（= swiglu(gate, up)，见语义）；mq / ms：mid 按行量化后的 fp8 [M·TOPK, I] 与 f32 [M·TOPK]。
# 排序输出（流式与路由不多的 prefill 用自写的 stream_prep，其余 prefill 用 aiter moe_sorting，格式相同；int32，权重 f32）：排序块 e = 同一 expert
#   （sorted_expert_ids[e]）的 BM 个排序行；sorted_ids[排序行] = token | slot << 24，token ≥ M 的是填充行（只在每个
#   expert 的末块尾部）；sorted_weights[排序行] = 该路由的路由权重；num_valid = 排序行总数（含填充行）：grid 按排序块
#   数的上界发射，e·BM ≥ num_valid 的 CTA 直接退出；loc_ids[token, slot] = 该路由排序后的行号（< 0 表示没有这条路由，
#   求和时跳过；同一 token 重复选同一 expert 时，aiter 排序只保留最后一个槽、其余槽为 −1，stream_prep 每条路由都保留）。
# plan [M·TOPK, 4]（decode 按 expert 去重的计划）：同一 expert 的路由按路由号每 GROUP 条一组（M ≤ 16 时 2，否则 3）；
#   plan[r] = [条数, r, 同 expert 的后续路由…]，条数只在组首非 0；组首的 CTA 算全组，只读前「条数」个组员。
# w1 / w2：fp4 权重码，w1_delta / w2_delta：delta 字节（布局见上）；w1_scale / w2_scale [E, 通道]：通道因子 2^(cmax − 132)，收尾才乘。
# —— Gluon 的真实操作（gl = triton.experimental.gluon.language，cdna3 = gl.amd.cdna3）——
# r = cdna3.buffer_load(t[下标])   global → VGPR。结果到第一次使用才等待，所以循环里先发后面几步的加载、再算这一步，
#                                  就是预取，提前的步数就是预取距离；加载结果由循环变量带到用它的那一步
#                                  （decode.py 读激活行用普通 global load，没有越界检查：越过行尾的 lane 钳到最后一片或不发）
# cdna3.buffer_store(v, t[下标])   VGPR → global；偏移越界（如填充行）的写被硬件丢弃
# L = gl.allocate_shared_memory(类型, [槽数, …], 布局)：一块 LDS；L.index(i) 是第 i 槽
# L.index(i).store(r) / .load(布局)：VGPR → LDS / LDS → VGPR，load 的布局决定每个 lane 取到哪些元素；
#                                  按「B 操作数布局」load 直接得到 MFMA 的 b[K, 列]（LDS 里按行存，转置由布局完成）
# gl.barrier()                     CTA 内同步：之前写进 LDS 的数据对所有 wave 可见，之前读完的槽此后可以覆盖
# acc = cdna3.mfma(a, b, acc)      fp8 MFMA 16x16x32、f32 累加：acc[行, 列] += a[行, K]·b[K, 列]（形状更大即多条指令）
# —— 伪操作（Gluon 没有，或只能用 inline_asm_elementwise 写）——
# wave_id() / lane_id()：本 wave / lane 的编号（Gluon 靠布局的 warps_per_cta 分配 wave，没有显式编号）
# uniform(v)：readfirstlane，把 wave 内相同的值变成标量，供整 wave 分支
# decode(wr, …)：寄存器里的 fp4 码 + delta 字节 → fp8 操作数，|e2m1| × 2^(5−delta) 带符号（v_perm 查表，见上面的布局）
# wave_max(v)：wave 内求 max（DPP / shuffle_xor + readlane）；rcp(v)：硬件近似倒数；to_fp8(v)：RNE 转 fp8，-0 存成 +0
# tree_sum(v, axis=0)：顺序固定的树形求和（前 ⌈n/2⌉ 项与其余各自递归求和再相加），decode Down 用它合并各 wave 的部分和
# atomic_add(t[下标], v)：global 原子加（agent 范围、不取回；bf16 时每 lane 两个相邻元素一条 bf16x2 指令），和的顺序不定；
#   LDS 上的 atomic_add 返回旧值（stream_prep 的计数与放置）
# ballot(p)：本 wave 各 lane 的谓词拼成 64 位掩码；popcount / 最低置位用来数同 expert 的路由（去重计划）
# —— 语义 ——
# silu(g) = g · sigmoid(g)，实现为 g · rcp(1 + exp2(−g·log2e))
# swiglu(g, u) = silu(g)·u，g、u 已乘完通道因子与激活 scale；SwiGLU 截断 L > 0（编译期常量 SWIGLU_LIMIT，kernel 名加 _swiglu）
#   时先 g = min(g, L)、u = clamp(u, −L, L)（a8w4_common.swiglu_clamp，min / max 不传播 NaN）；L = 0 时不发指令
# row_scale(amax) = amax / 240（IEEE 除法），结果为 0（全零行或下溢）时取 1
# quant_row(v) → (q, s)：s = row_scale(max|v|)；q = to_fp8(clamp(v / s, ±240))
# 行量化都按 quant_row 把一行量化成 fp8，彼此逐位一致：row_quant_fp8 通用（输入 bf16 或 f32），row_quant_decode 是 decode
#   档用的版本；去重的 row_quant_plan 与流式的 stream_prep 在各自那次 launch 里顺带量化 x。
# sorted_sum / route_sum：out[t, :] = Σ_slot routes[loc_ids[t, slot], :]（f32 按槽顺序相加后转 bf16），不读填充行。
# 三条路径（host 见 moe_a8w4.py 的 fused_moe_a8w4）：
#   decode（M ≤ 32，H ≤ 4096 或 H5120（DECODE_WIDE_HIDDEN），decode kernel 放得下；不排序，也就没有 sorted_sum）：
#     逐路由（Gate/Up 按路由、Down 按 token 分 CTA）：
#       [row_quant_decode(x)] → gateup_decode → [row_quant_decode(mid)] → down_decode(_pq) → out
#       先量化 x：Gate/Up 的 CTA 数 M·TOPK·I/16 ≥ 2560；先量化 mid、Down 换槽对版：M ≥ 4、槽对版放得下且 w2 < 4 GiB
#     只去重 Gate/Up（H5120、8 ≤ M ≤ 32、槽对版放得下且 w2 < 4 GiB；同 expert 的路由每组 M ≤ 16 时 2 条，否则 3 条）：
#       row_quant_plan(x；写计划) → gateup_decode_dd(4 wave) → row_quant_decode(mid) → down_decode_pq → out
#   流式（M > 32 且 M·TOPK ≤ R·E；R = 22，H ≥ 4096 为 48，H5120 且 M·TOPK ≥ 4096 为 16，H2560 且排序块 64 能尾部量化时 56；
#     排序块：每 expert 平均 ≤ 12 行为 16（还要 3·M·TOPK ≥ 4E 或 H ≥ 5120，否则 32），≤ 28 行为 32，其余 64）：
#     stream_prep(排序 + 量化 x [+ 清零 out]) → gateup_stream（带尾部量化，或另跑 row_quant_fp8）→ down_stream（I < 512：
#     原子累加进 prep 清零的 out；否则写 routes → route_sum）→ out
#   prefill（其余，含 decode 放不下的 M ≤ 32）：H ≥ 2560 且 M·TOPK ≤ 24576（PREFILL_PREP_MAX_ROUTES；H5120 为 12288）时
#     stream_prep(排序 + 量化 x，不清零 out)，否则 moe_sorting → row_quant_fp8(x；顺带把 loc_ids 置 −1)；之后都是
#     gateup_prefill（BM64·BN128·TK128）→ row_quant_fp8(mid) → down_prefill（并写有效路由的 loc_ids）→ sorted_sum → out；
#     H5120 且 M·TOPK ≥ 16384 时两个 GEMM 按 XCD 分块排布（XCD_CHUNK）；I > 512 且 BM·I > 32 KiB 时 down_prefill 的激活分两段
```

## A8W4 Gate/Up prefill（[gemm1.py](../moe_a8w4/gemm1.py#L624)）
prefill 档（M > 32 且每个 expert 平均超过流式上限、给了 tile 参数，或 decode 放不下的 M ≤ 32）按排序块算
mid[token, slot, :] = swiglu(x·w1_gate, x·w1_up)，f32 写出；结构同 aiter gfx9_fp8 `gemm1.py` 的 prefill_1x4，权重换成 mxfp4。

**Gluon 风格**
```python
# grid = (2·I/BN, 排序块数；M·TOPK ≤ E 时取 M·TOPK)，256 线程；默认 BM=64、BN=128、TILE_K=128（此时固定每 SIMD 4 wave）
#   XCD_CHUNK > 0（H5120 且路由 ≥ 16384 时为 1）时改用 1-D grid，编号见下文「XCD 分块」
# BN = 本 CTA 的 gate + up 通道总数，输出只有 BN/2 列（每列由同号的一个 gate、一个 up 通道得到）
# 放置：x 按 token gather，global → VGPR 暂存（A_AHEAD 份）→ LDS 双缓冲 A[2]，4 个 wave 共读；权重 global → 本 wave
#       的 VGPR 两槽，不经 LDS；累加器一直在 VGPR。预取距离：权重 2 个 tile（tile t 的槽在其最后一块解码后装 t+2），
#       x 在 H ≤ 2560 时 2 个 tile（A_AHEAD = 2），否则 1 个
@gluon.jit
def a8w4_gateup_prefill(xq, xs, w1, w1_delta, w1_scale, sorted_ids, sorted_expert_ids, num_valid, mid):
    n_blk, e = gl.program_id(0), gl.program_id(1)    # n_blk：BN/2 个 gate 通道及同号 up 通道；e：排序块
    if XCD_CHUNK > 0: w = xcd_work(gl.program_id(0)); n_blk, e = w % (2 * I // BN), w // (2 * I // BN)
    if e * BM >= num_valid: return                   # 多发的 CTA 直接退出
    ids = sorted_ids[e * BM + gl.arange(0, BM)]      # 经 LDS 分给各线程；取完后 barrier，这块 LDS 让给 A[0]
    TILE_M = 有效行数向上取到 16 的倍数             # BM64；其他 BM 为 BM/2 或 BM（有效行在块首，按取值各一份特化）
    token, slot, expert = ids[:TILE_M] & 0xFFFFFF, ids[:TILE_M] >> 24, sorted_expert_ids[e]
    ch = n_blk * (BN // 2) + 16 * wave_id() + gl.arange(0, 16)   # 本 wave 的 16 个 gate 通道；up 用同号的 I + ch
    wrow = 每 lane 取本 wave gate（lane%16 < 8）或 up 块的相邻两行 2·(lane%8)、+1   # 一条 16 B = 两行各 8 B
    A = gl.allocate_shared_memory(fp8, [2, BM, TILE_K], …)    # 各特化都按 BM 行分配，只用前 TILE_M 行
    NT = H // TILE_K
    # 序言：x 的 tile 0…A_AHEAD−1，权重 tile 0、1
    xr = [cdna3.buffer_load(xq[token, tile(j)]) for j in range(A_AHEAD)]   # gather：每线程 TILE_M·TILE_K/4096 条 16 B
    wr = [cdna3.buffer_load(w1[expert, wrow, tile(j)]) for j in (0, 1)]    # 连同 delta（TK128 时相邻两 tile 共用一条）
    A.index(0).store(xr[0]); wd = decode(wr[0], 块 0); gl.barrier()
    ce = gl.zeros([16, TILE_M], f32)                 # A 操作数 = 各 lane 的偶数行：行 R < 8 是 gate 通道 2R，R ≥ 8 是 up
    co = gl.zeros([16, TILE_M], f32)                 # 奇数行（通道 2R+1）；B = TILE_M 个 token 的 x
    # 稳态：算 tile t；x 提前 A_AHEAD 个 tile，权重提前 2 个 tile
    for t in range(NT):
        if t + A_AHEAD < NT:                         # 先发，之后才用
            xr[t % A_AHEAD] = cdna3.buffer_load(xq[token, tile(t + A_AHEAD)])
        for ki in gl.static_range(TILE_K // 64):
            b = A.index(t % 2).load(B 操作数布局)[64 * ki:64 * ki + 64, :]   # LDS → VGPR，[K, token]；按 K 置换视图读
            if ki > 0: wd = decode(wr[t % 2], 块 ki) # K 置换（见下文）让一个 128-K 步的两个 64-K 块用同一个 delta 字节
            ce = cdna3.mfma(wd.偶数行, b, ce)
            co = cdna3.mfma(wd.奇数行, b, co)
        if t + 2 < NT: wr[t % 2] = cdna3.buffer_load(w1[expert, wrow, tile(t + 2)])   # 本 tile 的槽已解码完
        if t + 1 < NT:
            wd = decode(wr[(t + 1) % 2], 块 0)       # 下一 tile 的第 0 块：其权重已在路上一整个 tile
            A.index((t + 1) % 2).store(xr[(t + 1) % A_AHEAD])   # 写另一槽，不碰本 tile 正在读的槽
        gl.barrier()                                 # 每 tile 一次：新槽写完才可读；本槽读完才会被 tile t+2 覆盖
    # 收尾：lane l 与 l^32 交换累加器（ds_bpermute），每 lane 得到 4 个相邻通道的 gate 与 up；不经 LDS
    g, u = exchange(ce, co)
    sx = xs[token][None, :]                          # 每个 token 的激活 scale
    v = swiglu(g * w1_scale[expert, ch][:, None] * sx, u * w1_scale[expert, I + ch][:, None] * sx)
    cdna3.buffer_store(v.T, mid[token, slot, ch])    # f32；每 lane 4 个通道一次 16 B；填充行偏移越界被丢弃
```

**任务划分**（默认 BM64、BN128、TILE_K128；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (2I/128, 排序块数) | x：64 gate + 同号 64 up 通道；y：1 个排序块 |
| CTA（256 线程） | 4 个 wave | 本块 TILE_M 个排序行（同一 expert）× 128 通道 × 全部 H |
| wave | 1 个 16 通道块 | gate、up 各 16 通道 × TILE_M 行（TILE_M/16 个 token 块） |
| lane | g = l/16、i = l%16 | gate（i<8）或 up 块的相邻两行 2(i%8)、+1 在每个 128-K 步里的第 g 个 32-K 组；token i 的累加 |
| K 步 | 128-K 步（两个 64-K 半块） | 每个半块每 wave 4·TILE_M/16 条 MFMA 16x16x32 |

TILE_M：BM64 时为有效行数向上取到 16 的倍数（有效行在块首），其他 BM 在第 BM/2 行是填充时取 BM/2、否则 BM；各取值一份特化，
由 readfirstlane 后的标量分支选；x 仍按整趟拷贝装载（TILE_K128 时一趟 32 行）。
`e·BM ≥ num_valid` 的 CTA 直接退出；M·topk ≤ E 时 grid 的 y 维取 M·topk（每条路由至多占一个排序块）。
**XCD 分块**（`XCD_CHUNK` = C > 0，[Gate/Up](../moe_a8w4/gemm1.py#L638)、[Down](../moe_a8w4/gemm2.py#L120)）：MI308X 把一次 launch 的 WG L 发到 XCD L % 4
（`a8w4_common.XCDS`）。1-D grid 共 ⌈task/(4C)⌉·4·C·T 个 WG（T = 每个排序块的 WG 数：Gate/Up 为 N tile 数 2I/BN，Down 为 NSPLIT），
WG L 做第 w = xcd_work(L) = (⌊⌊L/4⌋ / CT⌋·4 + L % 4)·CT + ⌊L/4⌋ % CT 个工作（排序块 w / T 的第 w % T 个 N tile 或 hidden 段）：
连续 C 个排序块的全部 WG 落在同一 XCD，同块各 N tile 共用激活行、同 expert 的相邻块共用权重，都留在该 XCD 的 L2；编号越过 task 数的
WG 与越过 num_valid 的一样直接退出。host 在 H ≥ 5120 且路由 ≥ 16384 时开（Gate/Up C = 1、Down C = 4）；只改哪个 WG 算哪块，结果逐位不变。
K 置换：A 的 LDS 读视图把步内 MFMA 侧的 K 64h + 16g + 8c + v 映射到存储的 32g + 16h + 8c + v，权重按同一置换取址，
所以 lane 组 g 在步内两个半块都用同一个 32-K 组：每步只用一个 delta 字节（第 g/2 + 2·(步号%2) 个）建一次 LUT。

**流水线**（每个 tile 的 region 表——MFMA / LDS → VGPR / global → VGPR / 解码 / VGPR → LDS——与 4 条流水线规则见
[gemm1.py 文件头](../moe_a8w4/gemm1.py#L29)，这里只列骨架；x 的 LDS 双缓冲 buf = k%2，权重 VGPR 两槽 k%2）
```text
序言
  tid < BM：sorted_ids[e·BM + tid] → LDS；barrier；各线程取 epilogue 的 (token, slot) 与 gather 行号；定 TILE_M
  发 x(0 … A_AHEAD−1)、W(0)；等待；x(0) → buf 0；解码 W(0) 第 0 块；barrier；发 W(1)
tile k（TILE_K/64 个 region）
  region 0：发 x(k+A_AHEAD)；读 buf 的第 0、1 块；解码 W(k) 第 1 块；MFMA（偶 / 奇行各一套）
  region i：读第 i+1 块；解码 W(k) 第 i+1 块；MFMA
  last：MFMA；发 W(k+2) 进 tile k 的槽；解码 W(k+1) 第 0 块；暂存的 x(k+1) → buf^1；barrier
  vmcnt 只等本拍要用的加载，更晚发的仍在路上；TK128 时偶数 tile 取两 tile 共用的 delta，奇数 tile 从槽 0 复制
收尾：最后 A_AHEAD 个 tile 不发 x，最后一个不写 LDS，倒数第二个不发 W(k+2)
epilogue（累加器留在 VGPR，不过 LDS）
  ds_bpermute 在 lane l 与 l^32 间交换偶 / 奇行累加器 → 每 lane 持 4 个相邻通道的 gate 与 up
  acc ×= 通道因子 s[ch]（gate、up 各自）× xs[token]；[截断]；SiLU(gate)·up = gate · rcp(1 + exp2(−gate·log2e)) · up
  每个 token 块 4 个 f32 一次 16 B buffer store → mid[token, slot, ch … ch+3]；填充行的偏移改成越界值被丢弃
```
关键代码：[排序行号进 LDS](../moe_a8w4/gemm1.py#L672)、[半块分支](../moe_a8w4/gemm1.py#L724)、[K 置换读视图](../moe_a8w4/gemm1.py#L275)、
[lane 取址](../moe_a8w4/gemm1.py#L330)、[权重加载](../moe_a8w4/gemm1.py#L390)、[解码与 LUT 复用](../moe_a8w4/gemm1.py#L417)、
[指令排布](../moe_a8w4/gemm1.py#L457)、[一拍](../moe_a8w4/gemm1.py#L484)、[序言](../moe_a8w4/gemm1.py#L534)、[主循环](../moe_a8w4/gemm1.py#L546)、
[反量化与截断](../moe_a8w4/gemm1.py#L560)、[SiLU 与写出](../moe_a8w4/gemm1.py#L127)、[1-D grid](../moe_a8w4/gemm1.py#L764)。

**资源**：LDS 2·BM·TILE_K（A_ping / A_pong，排序行号的 1 KiB 与之复用）：host 固定 BM64、TILE_K128，为 16 KiB；
kernel 也支持 TILE_K256（32 KiB、权重寄存器翻倍、占用率更低）。

## A8W4 Down prefill（[gemm2.py](../moe_a8w4/gemm2.py#L105)）
prefill 档按排序块算 routes[排序行, :] = (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重，bf16 写出给 `sorted_sum`；结构同 aiter gfx9_fp8 `gemm2.py` 的默认 prefill Down。

**Gluon 风格**
```python
# grid = (NSPLIT, 排序块数)，256 线程；默认 BM=64；NSPLIT = 1 / 2 / 4 个 WG 分 H（各自重新 gather 本块激活）；I ≥ 256 时关 packed FP32；
#   XCD_CHUNK > 0（H5120 且路由 ≥ 16384 时为 4）时改用 1-D grid，编号同 Gate/Up 一节的「XCD 分块」（T = NSPLIT）
# 放置：mq 的 BM 行按路由 gather，global → VGPR → LDS，再整块（bm × I）读回 VGPR 常驻到结束（每个 wave 各存一份，
#       BM64、I320 时每 lane 80 个 VGPR），之后同一块 LDS 改作输出双缓冲 C[2]。I > 512 且 BM·I > 32 KiB 时
#       （one_pass 为假）分两段：前 ⌈KB/2⌉ 个 64 列块 gather 进 LDS、读进 VGPR 后，再 gather 其余列块覆盖同一块 LDS
#       （I576：5 + 4 块，LDS 36 → 20 KiB，每 CU 1 → 2 个 WG）。权重按 64 列块 global → VGPR（两套轮换），预取距离 1 块。
#       结果经 C 转成按行连续，再 16 B 合并写出
@gluon.jit
def a8w4_down_prefill(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                      routes, loc):
    part, e = gl.program_id(0), gl.program_id(1)    # 一个排序块 × 第 part 段 H/NSPLIT 列
    if XCD_CHUNK > 0: w = xcd_work(gl.program_id(0)); part, e = w % NSPLIT, w // NSPLIT
    if e * BM >= num_valid: return
    ids = sorted_ids[e * BM + gl.arange(0, BM)]
    route = (ids & 0xFFFFFF) * TOPK + (ids >> 24)
    segs = [(0, I)] if one_pass else [(0, 64 * ceil(KB / 2)), (64 * ceil(KB / 2), 64 * (KB // 2))]   # (起列, 列数)
    L = gl.allocate_shared_memory(fp8, [BM, max(段的列数)], …)
    L.store(cdna3.buffer_load(mq[route, 第一段])); gl.barrier()   # 全 CTA 一起 gather，每次 16 B
    bm = BM // 2 if uniform(ids[BM // 2] & 0xFFFFFF) >= M else BM   # 后一半全是填充：只算前一半（两份特化）
    rows, expert = e * BM + gl.arange(0, bm), sorted_expert_ids[e]  # 本块的排序行号
    b = [L.slice(0, bm).load(B 操作数布局)]          # [段的列数, bm]
    for k0, n in segs[1:]:                           # 两段时的第二段
        gl.barrier()                                 # 各 wave 都读完上一段，才能覆盖
        L.store(cdna3.buffer_load(mq[route, k0:k0 + n])); gl.barrier()
        b.append(L.slice(0, bm).load(B 操作数布局))
    gl.barrier()                                     # 激活整块已进 VGPR；L 读完才能改作 C
    C = L 改作 bf16 [2, bm, 64]
    row_scale = ms[route[:bm]] * sorted_weights[rows]     # 中间激活 scale × 路由权重
    NB = H // 64 // NSPLIT
    cols = lambda n: 64 * (part * NB + n) + gl.arange(0, 64)    # 本段第 n 个 64 列块
    ch = lambda n: cols(n)[16 * wave_id():16 * wave_id() + 16]  # 其中本 wave 的 16 列
    wr = cdna3.buffer_load(w2[expert, ch(0), :])          # 连同 delta 字节与这 16 列的通道因子
    for n in range(NB):                              # 稳态：算第 n 块时，第 n+1 块的权重在路上
        if n > 0:
            cdna3.buffer_store(C.index((n - 1) % 2).load(按行连续), routes[rows, cols(n - 1)])   # 上一块写出，
            if n + 1 < NB: wn = cdna3.buffer_load(w2[expert, ch(n + 1), :])                               #   与本块计算重叠
        acc = gl.zeros([16, bm], f32)                # A = 权重 16 列，B = bm 行激活
        for kb in gl.static_range(I // 64):          # 一个列块一次算完全部 K
            acc = cdna3.mfma(decode(wr, kb), cat(b)[64 * kb:64 * kb + 64, :], acc)   # cat：各段按 K 拼接，只是编号
        if n == 0: wn = cdna3.buffer_load(w2[expert, ch(1), :])   # 第 0 块例外：算完才发第 1 块
        c = (acc * w2_scale[expert, ch(n)][:, None] * row_scale[None, :]).to(bf16)   # RNE（位运算，输入须有限）
        C.index(n % 2).store(c.T)
        gl.barrier()                                 # 每块一次：C[n%2] 写完才可读；C[(n+1)%2] 已写出，此后可覆盖
        if n + 1 < NB: wr = wn
    cdna3.buffer_store(C.index((NB - 1) % 2).load(按行连续), routes[rows, cols(NB - 1)])
    # 填充行也写（值无意义），之后 sorted_sum 只按 loc_ids 读有效行
    if part == 0:                                    # 主循环后才写（写在开头会多占 VGPR、整体慢约 30%）
        valid = (ids & 0xFFFFFF) < M                 # 排序丢掉的槽（重复 expert）保持 x 行量化写的 −1
        cdna3.buffer_store(e * BM + gl.arange(0, BM), loc[route], mask=valid)   # 每条有效路由的排序行
```

**任务划分**（默认 BM64；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (NSPLIT, 排序块数) | 1 个排序块 × H/NSPLIT 列 |
| CTA（256 线程） | 4 个 wave | bm 行 mq 整块驻留；逐个 64 列块走完本段 |
| wave | 每个 64 列块中的 16 列 | 列 n·64 + 16w … +15 × bm 行 |
| lane | l = 16b + r | 权重列 r 的 K 16b…16b+15；B 操作数的 1/64 |
| K 步 | I/64 个 64-K 块 × 2 个 K32 | 每个列块一次算完全部 K（I320 为 10 步） |

bm = BM；排序块第 BM/2 行是填充时取 BM/2（两份特化各用一份构建缓存）。退出与 grid 收缩同 Gate/Up。H2560 共 40 个列块。
分两段只改激活进 VGPR 的方式（第二段在 bm 定下之后、特化体里 gather），MFMA 的次序与操作数都不变，结果与一段逐位相同。
NSPLIT 由 host 选（`_down_split`）：I_pad ≥ 256 时，排序块数·NSPLIT < 1000 就加倍，至多 4；块少时末轮 WG 不再大半空闲。

**流水线**（逐拍；权重 VGPR 两槽 0 / 1，预取距离 1 个列块；输出 LDS 双缓冲 C[0] / C[1]）
```text
序言
  全 CTA 按 sorted id gather mq 的 BM 行（每次 16 B，共 BM·I/4096 轮；两段时只取前 ⌈KB/2⌉ 个 64 列块）→ LDS A；barrier
  sorted_ids[BM/2]（readfirstlane）是填充 → 以下 bm = BM/2，否则 bm = BM
  LDS A 的前 bm 行 → VGPR（B 操作数）；两段时：barrier → gather 其余 ⌊KB/2⌋ 块覆盖 LDS A → barrier → 前 bm 行 → VGPR
  barrier（bm × I 整块已在 VGPR，之后 LDS 改作输出缓冲 C）
  行 scale = ms[token, slot] × 路由权重，按累加器布局放在 VGPR
  load_block(0) → 槽 0：本 wave 16 行权重 I/64 × dwordx2 + ⌈I/256⌉ × dword，及 64 个通道因子
  compute(槽 0) = 第 0 块；load_block(1) → 槽 1；第 0 块 × 行 scale → bf16 → C[0]；barrier
主循环 n = 0, 2, …, H/(64·NSPLIT) − 4（每次两拍；H2560、NSPLIT 1 为 19 次；块号都是本段内的）
  拍 0：C[0] → routes 第 n 块；load_block(n+2) → 槽 0
        compute(槽 1) = 第 n+1 块 → × 行 scale → bf16 → C[1]；barrier
  拍 1：C[1] → routes 第 n+1 块；load_block(n+3) → 槽 1
        compute(槽 0) = 第 n+2 块 → × 行 scale → bf16 → C[0]；barrier
收尾：C[0] → 本段倒数第二块；compute(槽 1) = 最后一块 → C[1]；barrier；C[1] → 最后一块
  part 0 的线程 tid < BM：排序行 tid 是有效路由时 loc[token, slot] = e·BM + tid（填充行写到越界偏移被丢弃）
compute(槽)
  for kb in 0 … I/64 − 1：delta dword[kb/4] 第 kb%4 字节 → LUT；2 个权重 dword → 16 个 fp8
  for kb：MFMA 16x16x32 fp8 × bm/8（bm/16 个 token 块 × 2 个 K32）；结果 × 通道因子
→ bf16：f32 位加 0x7FFF 与第 16 位（RNE，输入须有限），一条 v_perm 取两个值的高半字拼成一个字
C → routes：bf16 结果经 LDS 转成按行连续，再 16 B buffer store 合并写出 bm 行
```
关键代码：[gather](../moe_a8w4/gemm2.py#L178)、[半块分支](../moe_a8w4/gemm2.py#L416)、[激活进 VGPR 与第二段](../moe_a8w4/gemm2.py#L243)、
[行 scale](../moe_a8w4/gemm2.py#L278)、[load_block](../moe_a8w4/gemm2.py#L314)、[compute](../moe_a8w4/gemm2.py#L344)、[bf16 打包](../moe_a8w4/gemm2.py#L90)、
[写 LDS](../moe_a8w4/gemm2.py#L371)、[写 routes](../moe_a8w4/gemm2.py#L384)、[主循环](../moe_a8w4/gemm2.py#L395)、[两段的划分](../moe_a8w4/gemm2.py#L65)、[1-D grid](../moe_a8w4/gemm2.py#L463)。

**资源**：LDS = max(64·I B 激活, 2 × 64 × 64 × 2 B 输出) = 20 KiB（I320）/ 16 KiB（I192）；两段时激活只占较大的一段
（I576：64·320 B = 20 KiB；不分段要 36 KiB，每 CU 只放得下 1 个 WG）。64·I 须 ≤ 64 KiB，每段 64·列数须是 4096 的倍数（gather 每轮
256 线程 × 16 B），所以 BM32 配 I320 / 192 不可用，I > 1024 时整个 prefill 不可用。

## 行量化 prefill（[quant.py](../moe_a8w4/quant.py#L75)）
prefill 档把 bf16 输入 [M, H] 与 f32 中间 [M·topk, I]、流式档把不在 Gate/Up 尾部量化的 f32 中间按行量化成 fp8 与每行 scale，
结果与参考（IEEE 除法）逐位一致。

**Gluon 风格**
```python
# grid = ⌈rows / (256/L)⌉，256 线程：每行 L 个相邻 lane（L 见任务划分）
# 放置：不用 LDS；整行一次 global → VGPR（16 B 片按 lane 交错：lane l 取片 l、l+L、…），行内归约走 lane 间 shuffle
@gluon.jit
def a8w4_row_quant_fp8(x, q, scale, loc, rows):      # x：bf16 [M, H]（输入）或 f32 [M·TOPK, I]（mid）
    r = gl.program_id(0) * (256 // L) + gl.arange(0, 256 // L)   # 行号 ≥ rows 的读到 0、不写
    if FILL: cdna3.buffer_store(-1, loc[r, 0:FILL])  # prefill 的输入量化顺带把 loc_ids 置 −1（lane 写列 l、l+L、…）
    v = cdna3.buffer_load(x[r, :])
    amax = gl.max(gl.abs(v), axis=1)                 # maxnum（输入须有限）：lane 内求 max，再 log2(L) 次 shuffle_xor
    s = row_scale(amax)                              # 每行一次 IEEE 除法
    if s >= 2 ** -64:                                # 按行选：倒数乘再两次 FMA 修正，得正确舍入的 v / s
        y = 1 / s
        qv = v * y; qv += (v - qv * s) * y; qv += (v - qv * s) * y
    else:                                            # 残差可能下溢：逐元素 IEEE 除法
        qv = v / s
    cdna3.buffer_store(to_fp8(qv), q[r, :])          # |qv| ≤ 240·(1+2^-23) 舍入到 240，不必夹紧；与 IEEE 的 x / scale 逐位一致
    cdna3.buffer_store(s, scale[r])                  # 每行首个 lane 写
```

**任务划分**（L 从 64 起减半，直到整除每行的 16 B 片数且每 lane 至少 4 片，最小 4）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈rows / (256/L)⌉ | 256/L 行 |
| CTA（256 线程） | 4 个 wave | 每行 L 个相邻 lane |
| lane | 本行 16 B 片 l、l+L、… | H2560 bf16：L64、每 lane 5 片；I320 / I192 f32：L16 / L8、各 5 / 6 片 |

**流水线**
```text
FILL > 0：loc[行, 0:FILL] 置 −1（lane 写列 l、l+L、…，列号 ≥ FILL 的写到越界偏移被丢弃）
lane 读片 l、l+L、…（每次 16 B buffer load，一条指令覆盖连续内存；rows 以外的行读到 0，写被丢弃）
lane 内 max|x|（v_max3）→ log2(L) 次 shuffle_xor 得整行 amax；scale = amax / 240（IEEE；为 0 时取 1）；每行首个 lane 写 scale
scale ≥ 2^-64：y = 1/scale（IEEE，每行一次）；q = x·y，再两次 q += (x − q·scale)·y（FMA），得正确舍入的商
scale < 2^-64（残差可能下溢）：q = x / scale，逐元素 IEEE 除法
每 4 个 cvt_pk_fp8（RNE，不夹紧）成一个字 → 0x80（-0）改 0 → 每片一次写（bf16 片 8 B、f32 片 4 B）
```
关键代码：[lane 数](../moe_a8w4/quant.py#L38)、[16 B 读](../moe_a8w4/quant.py#L115)、[amax 与 scale](../moe_a8w4/quant.py#L133)、
[倒数与修正](../moe_a8w4/quant.py#L156)、[打包与 -0](../moe_a8w4/quant.py#L65)、[写回](../moe_a8w4/quant.py#L143)。

**资源**：不用 LDS。

## A8W4 Gate/Up decode（[decode.py](../moe_a8w4/decode.py#L233)）
M ≤ 32 且形状受支持（H%512 = 0，见 [gateup_decode_fits](../moe_a8w4/decode.py#L56)；host 只放行 H ≤ 4096 与 H5120，后者 M ≥ 8 改走去重）时不排序：
每个 WG 算一条路由 (token, slot) 的 16 个 gate 与同号 16 个 up 通道，写 mid[token, slot, ch]（f32）。

**Gluon 风格**
```python
# grid = (I/16, M·TOPK)，256 线程（4 个 wave）；不排序
# 放置：x 的一行（H 个 fp8）放 LDS，4 个 wave 共读（A 操作数是广播读）；LUT 表（delta 0–11 各 8 B）放 LDS；权重 global →
#       本 wave 的 VGPR，序言一次全部发出（H2560 每 lane 10 条 16 B），不分 tile、无双缓冲，第 p 个块对只等自己那几条加载；
#       4 个 wave 的部分和经 LDS 相加
@gluon.jit
def a8w4_gateup_decode(x, xq, xs, topk_ids, w1, w1_delta, w1_scale, mid, pad0, pad1):   # pad0/1 不用，保持参数布局与 ISA
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
        red.index(w).store(wave_max(gl.abs(xr))); gl.barrier()   # 每个 wave 只持有 x 行的一部分
        amax = gl.max(red.load(…)); s = row_scale(amax)
        xl.store(to_fp8(gl.clamp(xr / s, -240, 240)))
    gl.barrier()
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
    part.index(w).store(acc 的行 0 列 0–7 与行 8 列 8–15); gl.barrier()
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

A 操作数的 16 行都是同一 token 的 x；输出取行 0 的列 0–7 与行 8 的列 8–15（各含一半 K 组），
列 m 与 m+8 相加得行 2m+h 的全 K（映射见代码 docstring 的 “Block pairs”）。

**流水线**（不分 tile：序言一次发出本 wave 的全部加载，主循环按序消费；无双缓冲）
```text
序言
  tid ≤ 11：LUT 表 → LDS lut（delta d 在字节 8d：|e2m1| 0–3、4–7 的 fp8 各一个 dword）
  expert = topk_ids[route]；本 wave 的 16 行块 rb 与 K 起点 kb0
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
关键代码：[LUT 表](../moe_a8w4/decode.py#L173)、[x 行加载](../moe_a8w4/decode.py#L297)、[权重加载](../moe_a8w4/decode.py#L328)、[片写 LDS](../moe_a8w4/decode.py#L361)、
[核内量化](../moe_a8w4/decode.py#L368)、[主循环](../moe_a8w4/decode.py#L401)、[块解码](../moe_a8w4/decode.py#L204)、[收尾](../moe_a8w4/decode.py#L418)。

**资源**：LDS 约 3.2 KiB（xq H B、part 512 B、red 16 B、LUT 96 B；H5120 多 2.5 KiB）；本 wave 的权重全部同时在途
（H2560 每 lane 10 条 16 B，H5120 为 20 条）。

## A8W4 Down decode（核内量化，[decode.py](../moe_a8w4/decode.py#L477)）
M ≤ 3、w2 ≥ 4 GiB 或槽对版放不下时用：每个 WG 算一个 token 的 16 个输出通道，在 LDS 里把 topk 个 expert 相加后直接写 bf16 out。
它的 LDS 放不下时（[down_decode_fits](../moe_a8w4/decode.py#L62)）decode 整体不用，M ≤ 32 走 prefill。

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
    gl.barrier()
    total = gl.zeros([16], f32)
    for r in range(R):
        acc = gl.zeros([16, 16], f32)                # A 的 16 行都是本槽的 mq，结果取行 0
        for kb in gl.static_range(I // 64):          # 每个 64-K 块 2 条 MFMA
            a = mq_l.index(w).index(r).load(A 操作数布局)[:, 64 * kb:64 * kb + 64]
            acc = cdna3.mfma(a, decode(wr[r], kb, lut), acc)
        total += gl.where(w + W * r < TOPK, acc[0] * cs[r] * sa[r] * tw[r], 0)
    part = gl.allocate_shared_memory(f32, [W, 16], …)
    part.index(w).store(total); gl.barrier()
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
关键代码：[槽与 expert](../moe_a8w4/decode.py#L532)、[加载](../moe_a8w4/decode.py#L542)、[wave 内量化](../moe_a8w4/decode.py#L578)、
[主循环](../moe_a8w4/decode.py#L603)、[收尾](../moe_a8w4/decode.py#L617)。

**资源**：LDS = W·⌈topk/W⌉·I B（mq）+ 64·W B（part）+ 96 B，TP2 topk11 约 4.2 KiB。

## A8W4 Down decode 槽对（预量化，[decode.py](../moe_a8w4/decode.py#L667)）
M ≥ 4、w2 < 4 GiB 且放得下（每行 I/16 个 16 B 片 ≤ 64·W 个线程，全零行由每线程写一片；LDS 够）时用（H5120 只去重 Gate/Up 的
路径也用它）：读预量化的 mq fp8 与 ms；
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
    gl.barrier()
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
    part.index(w).store(vals); gl.barrier()          # 列 0–7 取自行 0/1（lane 0–7），列 8–15 取自行 8/9（lane 40–47）
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
关键代码：[零行](../moe_a8w4/decode.py#L717)、[槽对](../moe_a8w4/decode.py#L728)、[mq 片](../moe_a8w4/decode.py#L741)、
[整张量寻址](../moe_a8w4/decode.py#L761)、[屏蔽 A 操作数](../moe_a8w4/decode.py#L816)、[主循环](../moe_a8w4/decode.py#L818)、[收尾](../moe_a8w4/decode.py#L845)。

**资源**：LDS = (2·W·轮数 + 1)·I B + 128·W B + 96 B，TP2 topk11 约 4.5 KiB；
每 lane 偏移是 32 位，w2 总字节数须 < 4 GiB（否则 host 改用核内量化版）。

## 行量化 decode（[decode.py](../moe_a8w4/decode.py#L899)）
槽对 Down 之前量化 mid（f32 [M·topk, I]），M·topk·I/16 ≥ 2560 时量化 x（bf16 [M, H]）；逐元素 IEEE 除法，
结果与 [quant.py](../moe_a8w4/quant.py) 逐位一致。

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
关键代码：[读片](../moe_a8w4/decode.py#L921)、[scale](../moe_a8w4/decode.py#L945)、[写回](../moe_a8w4/decode.py#L949)、[-0 处理](../moe_a8w4/decode.py#L148)。

**资源**：不用 LDS。

## 去重计划与 x 行量化（[decode_dedup.py](../moe_a8w4/decode_dedup.py#L147)）
只去重 Gate/Up 的 decode 路径（H5120、M 8–32）的第一步：一次 launch 量化 x 的每一行（与 row_quant_decode 同算术、逐位一致），
并为每条路由写计划条目（格式见文首 plan）。

**Gluon 风格**
```python
# grid = M·(1 + ⌈TOPK/4⌉)，256 线程：前 M 个 CTA 各量化 x 的一行，其余 CTA 的每个 wave 规划一条路由
# 放置：x 行 global → VGPR（每线程 16 B 片 t + 256m），行 amax 先在 wave 内、再经 LDS 的 4 个槽合并；规划时每个 wave
#       把全部 M·TOPK 个 expert id 按 64 条一段读进 VGPR（lane l 持有第 l + 64k 条），用 ballot 数同 expert 的路由
@gluon.jit
def a8w4_row_quant_plan(x, xq, xs, topk_ids, plan, pad):   # pad 不用，保持参数布局与 ISA
    b = gl.program_id(0)
    if b < M:                                        # 量化第 b 行
        v = cdna3.buffer_load(x[b, :])               # 越过行尾的线程重读最后一片（不改 amax）
        red = gl.allocate_shared_memory(f32, [4], …)
        red.index(wave_id()).store(wave_max(gl.abs(v))); gl.barrier()   # barrier：4 个 wave 的 max 都写完才合并
        s = row_scale(gl.max(red.load(…)))
        xq[b, :] = to_fp8(gl.clamp(v / s, -240, 240)); xs[b] = s       # 逐元素 IEEE 除法
    else:                                            # 规划：每个 token 用 ⌈TOPK/4⌉ 个 CTA，wave 管一个槽
        t, slot = (b - M) // ⌈TOPK/4⌉, ((b - M) % ⌈TOPK/4⌉) * 4 + wave_id()
        if slot < TOPK:
            r = t * TOPK + slot
            ids = topk_ids.flat[lane_id() + 64 * gl.arange(0, SEGS)]   # SEGS = ⌈MAX_M·TOPK/64⌉（MAX_M = 32），越界为 −1
            e = topk_ids.flat[r]
            hit = [ballot(ids[k] == e) for k in range(SEGS)]           # 每段 64 位
            rank = Σ popcount(hit 中路由号 < r 的位); total = Σ popcount(hit)
            nxt = 依次取上一个成员之后的最低置位（从低段找起），共 GROUP−1 个；没有记 0x7FFFFFFF（其后的字无意义）
            plan[r] = [min(GROUP, total - rank) if rank % GROUP == 0 else 0, r, *nxt]   # lane 0 写 16 B
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | M·(1 + ⌈topk/4⌉) | 前 M 个：x 的一行；其余：一个 token 的 4 条路由 |
| CTA（256 线程） | 4 个 wave | 一行的全部 H（H5120 共 640 片 16 B，第三轮只有前 128 个线程）或 4 条路由 |
| lane | 量化：16 B 片 t + 256m；规划：第 l + 64k 条路由的 id | M ≤ 32、topk6 时 SEGS = 3 |

**流水线**
```text
量化 CTA（b < M）：线程读 x 行的 16 B 片 t + 256m → lane 内 max → wave_max → LDS red[wave]；barrier
  scale = max(red)/240（0 取 1）；q = clamp(x/scale)（IEEE）→ RNE fp8 → 每片 8 B 写 xq；tid 0 写 xs
规划 CTA：wave 管路由 r：lane l 读第 l + 64k 条路由的 id（越界为 −1）；标量读 expert = ids[r]
  每段 ballot(id == expert) → rank、total → 依次找后 GROUP−1 个成员 → lane 0 写 plan[r]（16 B）
```

关键代码：[规划一条路由](../moe_a8w4/decode_dedup.py#L103)、[量化分支](../moe_a8w4/decode_dedup.py#L195)。
**资源**：LDS 64 B（4 个 wave 的 amax）。计划只支持 M ≤ MAX_M（= 32），超出时 ballot 段不够。

## A8W4 Gate/Up decode 去重（[decode_dedup.py](../moe_a8w4/decode_dedup.py#L421)）
只去重 Gate/Up 的 decode 路径（H5120、M 8–32）里代替逐路由 Gate/Up：grid 与逐路由版相同，但只有组首路由的 CTA 计算，一次读该 expert 的
16 行块，算组内 ≤ GROUP 条路由（每个 token 都选的 shared expert 只读 ⌈M/GROUP⌉ 次而不是 M 次）。

**Gluon 风格**
```python
# grid = (I/16, M·TOPK)，64·WAVES 线程：host 用 WAVES = 4（kernel 也支持 8）；GROUP = 2（M ≤ 16）或 3
# 放置：组员 token 的 fp8 x 行 global → LDS [GROUP, H+16]，A 操作数按块对布局从 LDS 读（4 个 wave 各读自己的 K 段）；
#       权重 global → VGPR，序言一次发出本 wave 那段 K 的全部块对（无循环预取）；LUT 在 LDS；部分和经 LDS 跨 wave 相加
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
    mid[members[m], 16 * j + t % 16] = swiglu(g, u)   # f32；WAVES=4 时求和顺序同逐路由版
```

**任务划分**（H5120、4 wave；I 指补零后的值）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (I/16, M·topk) | x：16 个 gate 通道块 j 与同号 up 块；y：一条路由（只有组首计算） |
| CTA（256 线程） | 4 个 wave | 一组 ≤ GROUP 条路由 × 32 行权重 × 全部 H |
| wave | w%2：gate / up；w/2：K 前半 / 后半 | 16 行 × H/2（20 个块对） |
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

关键代码：[组首判断](../moe_a8w4/decode_dedup.py#L495)、[x 行与权重](../moe_a8w4/decode_dedup.py#L269)、[主循环](../moe_a8w4/decode_dedup.py#L366)、
[收尾](../moe_a8w4/decode_dedup.py#L380)、[pin](../moe_a8w4/decode_dedup.py#L61)、[host 分派](../../moe_a8w4.py#L363)。
**资源**：LDS = GROUP·(H+16) B + 128·WAVES·GROUP B + 96 B LUT（H5120、4 wave：GROUP2 约 11.1 KiB，GROUP3 约 16.6 KiB）；
每 lane PAIRS 条 16 B 权重加载同时在途（H5120 为 20 条）。

## A8W4 Gate/Up 流式（[stream.py](../moe_a8w4/stream.py#L361)）
流式档（M > 32、每个 expert 平均不超过流式上限）按排序块（同一 expert 的 16 / 32 / 64 行）
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
    S = valid_subblocks(ids, e, num_valid, M)        # 首行是真实行（token < M）的 16 行子块数；有效行在块首，所以是前缀；
                                                     #   e·BM ≥ num_valid 时为 0。为 S = 1…BM/16 各编一份代码，运行时
                                                     #   按 S 分支进入其中一份，S = 0 直接退出
    token, slot, expert = ids[:16 * S] & 0xFFFFFF, ids[:16 * S] >> 24, sorted_expert_ids[e]
    ch = 16 * ((c * W + wave_id()) * G + gl.arange(0, G))   # 本 wave 的 G 个 gate 16 行块起点，up 块同号（I + ch）
    if SHARE: ring = gl.allocate_shared_memory(fp8, [2 * SYNC, BM, 64], 行内 16 B 片按 (行 >> 1) & 3 异或错开)   # 只用前 16·S 行
    NK = H // 64
    # 序言：先发 P 步。激活：不共享时每 lane 每子块 16 B（自己的那行）；共享时全 WG 把本步 BM 行 × 64 B 切片（子块 ≥ S 的片
    #   不读），每 lane 读 max(4, BM/W) B，lane 多于片数时多余的 wave 写 LDS 草稿槽。权重：每步 2G 块 × 8 B，每 4 步一条 delta
    xr = {k: cdna3.buffer_load(xq[token, 64 * k:64 * k + 64]) for k in range(P)}
    wr = {k: cdna3.buffer_load(w1[expert, (ch, I + ch), 64 * k:64 * k + 64]) for k in range(P)}
    if SHARE:
        for j in range(SYNC): ring.index(j).store(xr.pop(j))
        gl.barrier()
    acc = gl.zeros([2 * G, 16, 16 * S], f32)         # A = 权重 16 通道，B = 16·S 个排序行
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
        for i in gl.static_range(2 * G):             # 每块：建一次 LUT、2 个 dword 解码，每子块 2 条 MFMA
            acc[i] = cdna3.mfma(decode(wr[k], i), b, acc[i])
        if SHARE and (k + 1) % SYNC == 0 and k + 1 < NK: gl.barrier()   # 新写的槽可读；读完的槽此后可覆盖
    # 收尾（累加器不经 LDS）
    sx = xs[token][None, :]
    v = swiglu(acc[:G] * w1_scale[expert, gate 通道] * sx, acc[G:] * w1_scale[expert, up 通道] * sx)
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
| lane | l = 16b + t | 权重行 t 的 K 16b…16b+15（每 64-K 8 B）；子块第 t 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每个权重块：1 次 LUT、2 个 dword 解码，每子块 2 条 MFMA 16x16x32 |

MFMA 的 A 是权重（16 个通道行），B 是 16 个排序行；输出 lane l 持有通道 4(l/16) … +3、排序行 l%16。
M·topk ≤ E 时 grid 的 y 维取 M·topk；排序块里有效行在前，所以有效子块是前缀，kernel 按 S 生成 BM/16 份特化。

**流水线**（预取距离 P 个 64-K 步：S = 1 时 PREFETCH[0]，否则 PREFETCH[1]；W>1（尾部量化还要 G≤4）时激活经 LDS 环共享）
```text
序言（一轮访存）：num_valid、本块 expert、本 lane 各子块的 sorted id → token / slot
  S = 首行有效的子块数（越过 num_valid 为 0）；进入第 S 份特化体，S = 0 直接退出
发 kb = 0 … P−1：激活（共享时每 lane 读 max(4, BM/W) B，否则每子块 16 B）→ 2G 个权重块各 8 B → 每 4 步 delta 1 dword
共享：激活(0 … SYNC−1) → LDS 槽 0 … SYNC−1；lgkmcnt(0) + barrier
主循环 kb = 0 … H/64 − 1：
  发 kb+P 的激活与权重（先激活后权重，等激活时不必等后发的权重）
  共享：激活(kb+SYNC) → LDS 槽 (kb+SYNC)%(2·SYNC)；从槽 kb%(2·SYNC) 读本 lane 的 16 B 作 B（行内 16 B 片按 (行号>>1)&3 异或错开）
  不共享：B = 寄存器里的激活(kb)
  for 2G 个权重块：LUT（delta 第 kb%4 字节）→ 2 个 dword 解成 2 个 fp8 i64 → 每子块 MFMA ×2
  共享：每 SYNC 步一次 lgkmcnt(0) + barrier（读完的槽此后才会被覆盖）；sched_barrier
ACT_STEPS = A > 1：序言只发组 0、1 的激活（每 lane 一条 16 B）并写组 0；组首步发组 g+2，组末步算完后写组 g+1 再 barrier
收尾
  acc × 通道因子（gate、up 各自）× xs[token] → [截断] → g · rcp(1 + exp2(−g·log2e)) · u
  不量化：每子块 4 个 f32 一次 16 B store → mid[token, slot, ch … ch+3]（填充行偏移越界，store 被丢弃）
  量化：行 amax（lane 内 4G 个 → xor 16、32）→ LDS [子块][行][wave]；barrier；取 W 个 wave 的最大值
        scale = amax/240（0 取 1）；q = v·rcp(scale) 在 ×(1∓2^-20) 两端 RNE 一致即用，否则整 wave 回退 IEEE 除法
        每 4 个 fp8 一次 4 B store → mq[route, ch … ch+3]；wave 0 的 lane < 16 写 ms[route]
```
关键代码：[前导](../moe_a8w4/stream.py#L330)、[按 S 特化](../moe_a8w4/stream.py#L742)、[共享激活](../moe_a8w4/stream.py#L605)、[ACT_STEPS 取址](../moe_a8w4/stream.py#L568)、[分组发与写](../moe_a8w4/stream.py#L656)、
[预取](../moe_a8w4/stream.py#L655)、[主循环](../moe_a8w4/stream.py#L670)、[SiLU 与截断](../moe_a8w4/stream.py#L434)、[尾部量化](../moe_a8w4/stream.py#L471)、
[夹逼与回退](../moe_a8w4/stream.py#L215)、[分派的 tile](../../moe_a8w4.py#L490)。

**资源**：LDS = 共享时 2·SYNC·BM·64 B 激活环（SYNC1：BM16 2 KiB、BM64 8 KiB，轻 wave 的 SYNC2 翻倍，ACT_STEPS 时 SYNC = A，BM16 / 32 都是 8 KiB；BM/W < 4 时另加 64·4 B 草稿槽，分派用不到）+ 量化时
BM·W·4 B amax；累加器 8·G·S 个 VGPR，分派只在它 ≤ 96 时让一个 WG 管全部 I 并尾部量化（TP2 BM32 为 80，BM64 为 160 故不量化）。

## A8W4 Down 流式（[stream.py](../moe_a8w4/stream.py#L1205)）
流式档按排序块算 (mq 行 · w2ᵀ) × 通道因子 × ms × 路由权重：I < 512 时（`ATOMIC`）用 bf16x2 原子加直接累加进 prep 清零过的 out，
否则按排序行 bf16 写 routes 给 route_sum。

**Gluon 风格**
```python
# grid = (H/(16·R·W), 排序块数)，64·W 线程（R8·W4 或 R16·W1，见任务划分）
# 放置：本块激活（16·S 行 mq，每行 I B）整块常驻 VGPR：W4（SHARE）时全 WG 分片只读一份进 LDS 再各自读回
#       W1 时直接从 global 读；通道因子一条 16 B 加载后放 LDS（每 wave 1 KiB）；权重按 16 通道块 global → VGPR（H ≥ 5120
#       时带 non-temporal 位），预取距离 P = min(4, R) 块；结果不经 LDS：写 routes 每 lane 8 B，或两块凑一次 bf16x2 原子加进 out
@gluon.jit
def a8w4_down_stream(mq, ms, w2, w2_delta, w2_scale, sorted_ids, sorted_weights, sorted_expert_ids, num_valid,
                     routes_or_out):
    c, e = gl.program_id(0), gl.program_id(1)        # c：W·R 个 16 通道输出块；e：排序块
    ids = sorted_ids[e * BM + gl.arange(0, BM)]
    S = valid_subblocks(ids, e, num_valid, M)        # 同 a8w4_gateup_stream
    rows = e * BM + gl.arange(0, 16 * S)             # 本块的排序行号
    route = (ids[:16 * S] & 0xFFFFFF) * TOPK + (ids[:16 * S] >> 24)
    expert, w = sorted_expert_ids[e], wave_id()
    ch = lambda r: 16 * ((c * W + w) * R + r) + gl.arange(0, 16)   # 本 wave 依次算的 R 个 16 通道块
    if SHARE:                                        # 激活加载先于权重发出：等激活时不必等权重
        A = gl.allocate_shared_memory(fp8, [BM, I + 16], …)   # 行距 I+16：16 行同列的 16 B 读落在不同 bank
        xr = cdna3.buffer_load(mq[route, :])         # 切成 16 B 片分给全 WG 的 lane
    sc = gl.allocate_shared_memory(f32, [W, 256], …)
    fr = cdna3.buffer_load(w2_scale[expert, ch(0)[0]:ch(0)[0] + 16 * R])   # 本 wave 的 R·16 个通道因子
    wr = {r: cdna3.buffer_load(w2[expert, ch(r), :]) for r in range(P)}   # 先发 P 块：每块 I/64 × 8 B + delta
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
    # delta：lane 组 b 与 b^1（相隔 16 个 lane）同属一个 32-K 组，要的 delta dword 完全相同。于是每两个 dword（各管 4 个
    #   64-K 块）只发一条加载：偶数组取第 q 个、奇数组取第 q+1 个，再 xor-16 交换，各自拿齐两个，delta 加载减半；
    #   第 r+1 块的交换在算第 r 块时就做（即 decode(wr[r+1]) 前的 shuffle_xor 提前一块），藏住交换延迟
    for r in range(R):                               # 稳态
        if r + P < R: wr[r + P] = cdna3.buffer_load(w2[expert, ch(r + P), :])
        acc = gl.zeros([2, 16, 16 * S], f32)         # 两条累加链：每个 64-K 块的两条 MFMA 各进一条，互不等待
        for kb in gl.static_range(I // 64):
            for h in gl.static_range(2):
                acc[h] = cdna3.mfma(decode(wr[r], kb, h), a[kb, h], acc[h])   # ATOMIC 时 A、B 互换：acc 为 [行, 通道]
        if not ATOMIC:
            v = (acc[0] + acc[1]) * sc.index(w).load(…)[16 * r:16 * r + 16, None] * row_scale[None, :]
            cdna3.buffer_store(v.T.to(bf16), routes[rows, ch(r)])   # RNE；每 lane 4 个通道 8 B；填充行被丢弃；之后 route_sum
        else:                                        # lane 16b+j 持有块 r 的通道 j、行 4b … 4b+3
            v = (acc[0] + acc[1]) * sc.index(w).load(…)[16 * r + j] * f4
            if r % 2 == 1:                           # 与块 r−1 凑对：shuffle_xor 1 后偶数 lane 持有块 r−1、奇数 lane 持有块 r 的
                atomic_add(out[tok4, 通道对], to_bf16(两个值))   #   同一对通道（RNE）；一条指令覆盖 4 行各 64 B；token ≥ M 不发
```

**任务划分**（R、W 由分派选：排序块 ≥ 32，或 H ≥ 4096 且 I ≤ 256，且共享激活行放得进 64 KiB LDS 时 R8·W4，否则 R16·W1）
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/(16·R·W), 排序块数) | x：W·R 个 16 通道输出块；y：1 个排序块 |
| CTA（64·W 线程） | W 个 wave | 共享时一起把本块 S·16 行 mq 装入 LDS |
| wave | R 个 16 通道块 | 每块全部 I；本块激活 S × I/64 × 16 B 驻留 VGPR |
| lane | l = 16b + t | 权重行 t 的 K 16b…16b+15；子块第 t 个排序行的同段激活 16 B |
| K 步 | 64-K 块 | 每子块 2 条 MFMA，分属两条累加链（I320 为 5 步） |

**流水线**（预取距离 P = min(4, R) 个通道块）
```text
序言：前导同 Gate/Up；进入第 S 份特化体，S = 0 退出
共享：本块 S·16 行 mq（每行 I B）切成 16 B 片分给全 WG 的 lane 发加载（先于权重发出）
通道因子：本 wave R·16 个 f32，一条 16 B 加载（lane·16 < R·64）
发前 P 个通道块的权重：每块 I/64 × 8 B + delta（一条 dword 取两组：奇数 16-lane 组取奇数组）
通道因子 → LDS 本 wave 的 1 KiB 区
共享：激活片 → LDS（行距 I+16，16 行的 16 B 读落在不同 bank）；barrier；每子块读 I/64 个 16 B → VGPR
不共享：每子块直接从 global 读 I/64 个 16 B → VGPR
行 scale = ms[route] × sorted_weights[排序行]；ATOMIC：每子块 4 行一组的 token 与行 scale 经 ds_bpermute 换到 C 布局
for r in 0 … R−1：
  发 r+P 块的权重；r+1 块的 delta 经 xor-16 交换拼出两组（提前一块，藏住交换延迟）
  for kb：LUT → 2 个 dword 解码 → 每子块 2 条 MFMA，分别累加到链 0 / 链 1（ATOMIC 时激活作 A、权重作 B）
  不 ATOMIC：两链相加 × 通道因子（LDS）× 行 scale → 两两 RNE 打包成 bf16 → 8 B store → routes[排序行, ch … ch+3]（填充行丢弃）
  ATOMIC：两链相加 × 通道因子 × 行 scale；奇数块与上一块凑对：shuffle_xor 1 → 偶数 lane 持块 r−1、奇数 lane 持块 r 的通道对
    → RNE 成 bf16x2 → agent 范围原子加 out[token, 通道对]（每子块 4 条；token ≥ M 的填充行不发）
```
关键代码：[共享激活](../moe_a8w4/stream.py#L1331)、[通道因子](../moe_a8w4/stream.py#L1361)、[权重与 delta](../moe_a8w4/stream.py#L1294)、
[delta 交换](../moe_a8w4/stream.py#L1312)、[主循环](../moe_a8w4/stream.py#L1424)、[写 routes](../moe_a8w4/stream.py#L1496)、[原子累加](../moe_a8w4/stream.py#L1445)、
[bf16x2 原子加](../moe_a8w4/stream.py#L107)、[LDS 预算](../moe_a8w4/stream.py#L1192)、[H5120 的权重缓存位](../../moe_a8w4.py#L588)。

**资源**：LDS = W·1 KiB 通道因子 + 共享时 BM·(I+16) B 激活 + W·1 KiB 越界片槽（构建时断言 ≤ 64 KiB；TP2 BM32 为 18.5 KiB，
I1024 BM64 要 73 KiB 所以不共享）；激活占 S·I/64·4 个 VGPR（BM64、I320 为 80）；ATOMIC 另占 8·S 个（行 token 与行 scale），
I = 512 时加上 K512 的激活反而更慢，所以分派只在 I < 512 时用。

## 流式 prep（排序 + 输入量化 + 清零）与 route_sum（[stream.py](../moe_a8w4/stream.py#L927)）
流式档（以及 H ≥ 2560、M·topk ≤ 24576（H5120 为 12288）的 prefill 档）用一次 launch 的 prep 代替 aiter moe_sorting 与输入量化：排序结果格式相同
（但同一 token 重复的 expert 每条路由都保留），
量化与 quant_row 逐位一致；Down 原子累加时同一次 launch 顺带把 out 清零。I ≥ 512 时 Down 写 routes，route_sum 代替 aiter gfx9_fp8 的
sorted_sum（同顺序同舍入）。

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

QW 由 host 选：H%2048 = 0 或 M ≤ 256 时 4（列块均分或行少），否则 1（行多时 CTA 少）。排序块 BM 也由 host 选（[_stream_block_m](../../moe_a8w4.py#L621)：H5120 时每 expert 平均 ≤ 12 行一律 16）。

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
关键代码：[容量](../moe_a8w4/stream.py#L893)、[直方图](../moe_a8w4/stream.py#L962)、[扫描](../moe_a8w4/stream.py#L973)、[放置](../moe_a8w4/stream.py#L998)、
[量化与清零](../moe_a8w4/stream.py#L1013)、[排序 CTA 主体](../moe_a8w4/stream.py#L1093)、[route_sum](../moe_a8w4/stream.py#L802)。

**资源**：prep 的 LDS 约 3·(256·⌈E/256⌉ + 64) 个字（E384 约 6.8 KiB、E513 约 9.8 KiB）；route_sum 不用 LDS。
排序容量按每个 expert 至多多出 BM−1 个填充行估上界（[stream_prep_sizes](../moe_a8w4/stream.py#L893)），grid 按它发射，多出的块 CTA 退出。
