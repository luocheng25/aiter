# gemm_a8w4_ptpc：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程；与代码对不上就改这里。三个 GEMM 模块的 region 表 / 阶段表、命名与流水线不变量写在
     gemm.py、gemm_large.py、small_m.py 的模块 docstring，这里链接过去，只补 Gluon 风格写法、任务划分与资源。 -->

## 先读：Gluon 风格的分阶段写法
下面每个 kernel 先用 Gluon 风格写一遍：Triton 风格只有块级张量，写不出操作数放在哪（直读 VGPR 还是经 LDS）、
预取几步、barrier 保护什么、wave 怎么分工；Gluon 能写出这些。一个 program = 一个 CTA；下标代替偏移运算，
省略 mask、布局对象与类型转换，只求看懂。
```python
# —— 张量 ——
# M：token 数（任意，M=0 不 launch）；N：输出通道（N % 64 == 0）；K：规约维（K % 256 == 0）。
# x：bf16 [M, K]；y：bf16 [M, N]。xq / xs：x 按 token 量化后的 fp8 e4m3fnuz [M, K] 与 f32 scale [M]。
# wq / wd / ws：prepare 出的 fd2 布局（layout.py）：wq = 按 MFMA K 序排的 fp4 码；wd = 每 32-K 组的
#   delta 字节（行内最大 E8M0 码 − 本组码，存 8·min(delta, 11)）；ws = f32 [N] 通道因子 2^(cmax − 132)。
# —— Gluon 的真实操作（gl = triton.experimental.gluon.language，cdna3 = gl.amd.cdna3）——
# r = cdna3.buffer_load(t[下标])   global → VGPR；越过 buffer 记录数（有效行之后）的读返回 0
# cdna3.buffer_store(v, t[下标])   VGPR → global；越界的写被硬件丢弃
# L = gl.allocate_shared_memory(fp8, [2, 64, 128], 布局)：两槽 LDS；L.index(i).store(r) / .load(布局)
# gl.barrier()                     CTA 内同步：之前的 LDS 写对所有 wave 可见，之前读完的槽此后可覆盖
# acc = cdna3.mfma(a, b, acc)      fp8 MFMA 16x16x32，f32 累加：acc[i, j] += a[i, K]·b[K, j]（通常 a = 权重 16 通道、
#                                  b = 16 个 token；decode 核反过来：a = token，b = 权重）
# —— 伪操作 ——
# wave_id() / lane_id()：wave / lane 编号；lane 组 g = lane // 16，行 r = lane % 16；tid() = 64·wave_id() + lane_id()
# lut(d, j)：delta 字节 d[j] → 8 项 fp8 表 |e2m1| × 2^(5 − delta)（v_perm 广播 + 两次减法）
# decode(wr, lut)：一个 fp4 dword（8 个码）→ 8 个 fp8（两次 v_perm 查表 + 符号位）
# wave_max(v)：同一行的 lanes 之间求 max（shuffle_xor）；to_fp8(v)：RNE 转 fp8（v_cvt_pk_fp8_f32）
# shuffle_xor(v, m)：取 lane ^ m 的 v（wave 内跨 lane 交换）
# max_u16(v)：bf16 位模式按 16 位无符号两两取 max（v_pk_max_u16）；bits(v) / as_f32(u)：按位重解释
# setprio(p)：本 wave 的发射优先级（s_setprio）；计算相设 1，MFMA 先于同 SIMD 另一个 wave 的访存发射
# —— 语义 ——
# row_scale(amax) = amax / 240（IEEE 除法），为 0（全零行）时取 1；amax 为 NaN 时 scale 为 NaN
# quant(v, s) = to_fp8(v / s)，v / s 为 IEEE 商（倒数乘加两次 FMA 修正；s < 2^-64 时逐元素除）；
#   |v / s| 至多略超 240（< 248），RNE 舍到 ±240，所以不必 clamp；NaN 仍是 NaN（0x80）
# —— 调用顺序（按 M 分三档）——
# flydsl_gemm_a8w4_ptpc(x, w)：
#   M ≤ 32 且 K % 512 == 0（tile 核还要 LDS 放得下）：一次 launch，M ≤ 8 decode(x, wq, wd, ws) → y，否则 tile(…) → y
#   其余：quant_x(x) → xq, xs；然后 M > 2048 且 N % 256 == 0、K ≥ 512：gemm_large(xq, xs, wq, wd, ws) → y；
#     否则 gemm(xq, xs, wq, wd, ws) → y
```

## gemm_a8w4_ptpc_quant_k{K}（[代码](quant.py)）
把 bf16 x 按 token 量化成 fp8 与 f32 scale，GEMM 之前一次 launch；结果与 torch 逐位一致（±Inf 行除外：整行写 NaN 码）。

**Gluon 风格**
```python
# grid = (ceil(M / R),)，256 线程；每行 L 个 lane（K % 512 == 0 时 L = 64，如 6 个目标形状；否则 32，如 K = 256、768），
#   R = 256 / L 行
# 放置：x 行段 global → VGPR 直读，lane sub 的第 v 次 16 B 读行内第 sub + v·L 个 16 B 单元（同一条指令的 L 个 lane
#   读连续的 16·L B）；不用 LDS；xq 同样交错直写（第 v 次 8 B 写第 sub + v·L 个 8 B 单元），xs 由 sub == 0 的 lane 写
def quant_x(x, xq, xs, M):
    row = gl.program_id(0) * R + tid() // L           # 本 lane 的行；≥ M 的行读 0、写丢弃
    sub = tid() % L                                   # 行内第 sub 个 lane
    for v in range(K // (8 * L)):                     # 展开，先发全部读
        u = sub + v * L                               # 行内第 u 个 16 B 单元 = 元素 8u … 8u+7
        r[v] = cdna3.buffer_load(x[row, 8 * u : 8 * u + 8])
    amax = wave_max(max_u16(bits(r) & 0x7FFF))        # 去符号的 bf16 位模式与 |x| 同序、NaN 最大；行内 L 个 lane 归约
    s = row_scale(as_f32(amax << 16))
    if sub == 0:
        cdna3.buffer_store(s, xs[row])
    for v in range(K // (8 * L)):
        u = sub + v * L
        cdna3.buffer_store(quant(r[v], s), xq[row, 8 * u : 8 * u + 8])   # 每次 8 B
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ceil(M / R) | 每 CTA R 行（R = 4 或 8） |
| CTA（256 线程） | R 行 × L lane | 互不相关的 R 行 |
| lane | K / L 个元素 | 一行里交错的 16 B 单元：读、量化、写 |

x、xq、xs 的 buffer 基址在每个 CTA 移到它的首行（64 位），任意 M 都不溢出 32 位偏移。

**流水线**：无循环。每 lane 发全部 16 B 读 → 段内 bf16 位模式两两取 max → 行内 shuffle 归约 → 求 scale → 量化 → 8 B 写。

**资源**：无 LDS；36 VGPR、19 SGPR（K = 1536），58 VGPR（K = 2560）。

## gemm_a8w4_ptpc_bm64_bn128_tk128（[代码](gemm.py)）
算 y = (xq·xs) · (fd2 解码后的 W · ws)ᵀ：权重在寄存器里按每组 LUT 解成 fp8，走 fp8 MFMA，收尾乘两个 scale 写 bf16。
M ≤ 2048 以及大 M 核不接的形状（N % 256 ≠ 0 或 K = 256）都走它。

**Gluon 风格**
```python
# grid = (ceil(N / 128), ceil(M / 64))，N 块最快；256 线程 = 4 wave；一个 CTA 算 y[64 token, 128 通道]
# 放置：xq tile 经 LDS 双缓冲（2 槽 × 64 × 128 B，XOR swizzle），提前 1 个 K tile（128 K）预取；
#   权重 global → VGPR 直读（fd2 字节，2 槽按 K tile 奇偶），同样提前 1 个 K tile；
#   解码出的 fp8 A 操作数 wf 比用它的 MFMA 早一个 64-K 半步；acc 在 VGPR；收尾直写 y（每 lane 8 B）
def gemm(xq, xs, wq, wd, ws, y, M):
    n0, m0 = gl.program_id(0) * 128, gl.program_id(1) * 64
    w = wave_id()                                     # 本 wave 的通道块：ch(h) = n0 + 64 h + 16 w，h = 0, 1
    L = gl.allocate_shared_memory(fp8, [2, 64, 128])
    acc = zeros([2, 4, 16, 16], f32)                  # [半区 h][token 列 n][通道][token]
    T = K // 128
    # 序言：tile 0
    xr = cdna3.buffer_load(xq[m0 : m0 + 64, tile 0])           # 每 lane 16 B，两遍各 32 行
    wr[0], dr[0] = cdna3.buffer_load(wq[ch(h), tile 0]), cdna3.buffer_load(wd[ch(h), tile 0])
    L.index(0).store(xr)
    wf0 = decode(wr[0].half0, lut(dr[0], g // 2))              # tile 0 的第一个 64-K 半步
    gl.barrier()                                               # L[0] 写完，各 wave 才能读
    for k in range(T):                                         # 运行时循环每次两个 tile（ping、pong），最后两个剥出
        s, t = k % 2, (k + 1) % 2
        if k + 1 < T:                                          # 预取下一 tile（距离 1），与本 tile 头几条 MFMA 交错
            xr = cdna3.buffer_load(xq[m0 : m0 + 64, tile k + 1])
            wr[t], dr[t] = cdna3.buffer_load(wq[ch(h), tile k + 1]), cdna3.buffer_load(wd[ch(h), tile k + 1])
        xb0 = L.index(s).load(B 布局, 半步 0)                  # K 置换：MMA 侧 64h+16g+8c+v ← 存储 32g+16h+8c+v
        acc = cdna3.mfma(wf0, xb0, acc)                        # 2 半区 × 4 列 × 2 条 = 16 条
        xb1 = L.index(s).load(B 布局, 半步 1)
        wf1 = decode(wr[s].half1, 同一个 LUT)                  # 第二个半步复用本 128-K 步的 LUT
        acc = cdna3.mfma(wf1, xb1, acc)
        if k + 1 < T:
            wf0 = decode(wr[t].half0, lut(dr[t], 2 * t + g // 2))   # 下一 tile 的第一个半步，建新 LUT
            L.index(t).store(xr)
        gl.barrier()                                           # 保护：下一阶段覆盖 L[s]（WAR）、读 L[t]（RAW）
    # 收尾：每 lane 4 个连续通道 × 1 个 token × 8 个 tile
    out = (acc * ws[通道] * xs[token]).to(bf16)
    cdna3.buffer_store(out, y[m0 : m0 + 64, n0 : n0 + 128])   # ≥ M 的行越界丢弃；N % 128 == 64 时跳过 ≥ N 的通道
```
与代码顺序的差别：代码里一个阶段的 VMEM 发射、LDS 读、解码与 MFMA 由 `hot_loop_scheduler` 交错排布（见下）；
解码的 LUT 字节选择 `2·(step % 2) + g // 2` 因为一个 delta dword 覆盖两个 128-K 步。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (⌈N/128⌉, ⌈M/64⌉)，N 块最快 | 一个 CTA 一块 y[64, 128] |
| CTA（256 线程） | 4 wave | xq tile 经 LDS 共享；128 通道权重 |
| wave | 2 × 16 通道 × 64 token | 两个 16 行权重块直读；64 行 x 从 LDS 读 |
| lane | lane 组 g、行 r | 每步每半区读 16 B 码 + 4 B delta；出 4 通道×8 tile |
| K 步 | 128 K = 两个 64-K 半步 | 每半步 16 条 MFMA；每步一个 LUT |

lane 组 g 在每个 128-K 步只读第 g 个 32-K 组（两个半步各 16 个码），所以每 lane 每步只要一个 delta、建一个 LUT。
xq、xs、y 的 buffer 基址在每个 CTA 移到本 tile 首行（64 位），块内偏移不超过 64 行，任意 M 都不溢出 32 位偏移；
权重（< 2 GiB）用全局偏移。

**流水线**（一个 K tile = 一个阶段；region 表与编号的不变量见 [gemm.py 模块 docstring](gemm.py#L4-L58)）
```text
序言：buffer_load x(0)、w(0)、d(0) → s_waitcnt vmcnt(0) → x(0) 写 L[0] → 解码 w(0) 前半 → barrier
阶段 k（读 L[k%2]，写 L[(k+1)%2]）：
  sched：先排本半步 0 的 4 条 LDS 读
  半步 0：16 条 MFMA；前 8 条各带 1 条 VMEM（x(k+1) 2 条 + w/d(k+1) 6 条）；MFMA 之间插解码 w(k) 后半的 VALU
          与半步 1 的 4 条 LDS 读
  半步 1：16 条 MFMA；之间插解码 w(k+1) 前半（含建 LUT）的 VALU；最后 2 条 MFMA 各带 1 条 LDS 写 x(k+1)
  sched_barrier → barrier
主循环：每次迭代 = 阶段 2i（ping）+ 阶段 2i+1（pong），累加器作为循环携带值
收尾：阶段 T−2（预取最后一个 tile）、阶段 T−1（不预取）→ 乘 ws·xs → bf16 → 8 B 写 y
```
权重槽 wr[s] 的前半在上一阶段已解码、后半在本阶段解码，所以本阶段的预取只能写 wr[t]（代码的 read_i / write_i，docstring 规则 2）。

**命名**与**流水线不变量**：见 [gemm.py 模块 docstring](gemm.py#L4-L58)（Naming、Pipeline rules 1–5）。

**资源**：LDS 16 KB（2 × 64 × 128 B）；122 VGPR、42 SGPR、无 spill；每 CU 4 个 CTA（LDS 64 KB 与 VGPR 同时到顶）。

## gemm_a8w4_ptpc_pp_bm128_bn256_tk128（[代码](gemm_large.py)）
M > 2048 且 N % 256 == 0、K ≥ 512 时的 GEMM：运算与上一个 kernel 相同（同一 fd2 解码、同一 K 序），换成持久化的
128 token × 256 通道 tile、两组 wave ping-pong，收尾写在 tile 最后一个 K 步里。

**Gluon 风格**
```python
# grid = (G,)，G = min(tiles, num_cus)，tiles = ⌈M/128⌉ · NB，NB = N / 256；512 线程 = 8 wave，每 CU 一个 CTA
# tile t = (M 块 t // NB, N 块 t % NB)；CTA c 依次做 t = c, c + G, …；一个 tile 算 y[128 token, 256 通道]
# 放置：xq tile 经 LDS 双缓冲（2 槽 × 128 × 128 B，XOR swizzle，槽 = K 步奇偶）；每线程的 32 B 先进 VGPR（x_stage），
#   VMEM 提前 2 个 K 步、写 LDS 提前 1 步；权重 global → VGPR 直读，两槽（K 步奇偶）× 两个 64-K 半步：
#   前半与 delta 提前 2 步、后半提前 1 步；解码出的 fp8 A 操作数 wf 比用它的 MFMA 早半步；acc 64 f32 / lane 常驻 VGPR
def gemm_large(xq, xs, wq, wd, ws, y, M):
    w, grp = wave_id(), wave_id() // 4                 # wave w：通道 32w … 32w+31 × 128 token；组 0 = wave 0–3
    L = gl.allocate_shared_memory(fp8, [2, 128, 128])
    T = K // 128                                       # 偶数且 ≥ 4
    # 序言（第一个 tile）：X(0) → L[0]，X(1) → x_stage，W0(0)、W1(0)、W0(1) 与 D(0)、D(1) 进槽，建 LUT、解码 W0(0)
    gl.barrier()
    if grp == 1:
        gl.barrier()                                   # 组 1 晚一相；循环后组 0 补一次，各 wave 的 barrier 数相同
    mem_phase(0)
    for tile in my_tiles():                            # 运行时循环；tile 内 K 步成对展开，首尾两对剥出
        for j in range(T):
            compute_phase(j, store=(j == T - 1))       # C(j)
            mem_phase(j + 1)                           # M(j+1)；j = T−1 时是下一 tile 的 M(0)

def mem_phase(j):                                      # M(j)：同 SIMD 另一组的 wave 正在发 MFMA
    xb = L.index(j % 2).load(B 布局)                   # X(j) 两个半步的 B 操作数
    L.index((j + 1) % 2).store(x_stage)                # X(j+1) 里本组那一半行
    x_stage = cdna3.buffer_load(xq[tile 行, K 步 j+2])  # j+2 ≥ T 时取下一 tile 的；≥ M 的行读 0
    wr[(j+1) % 2].half1 = cdna3.buffer_load(wq[本 wave 32 通道, K 步 j+1 后半])   # 每 lane 16 B
    wr[j % 2].half0 = cdna3.buffer_load(wq[本 wave 32 通道, K 步 j+2 前半])        # 每 lane 16 B
    dr[j % 2] = cdna3.buffer_load(wd[本 wave 32 通道, K 步 j+2])                   # 每 lane 8 B
    if j == T - 2:
        xs_r, ws_r = cdna3.buffer_load(xs[tile 行]), cdna3.buffer_load(ws[本 lane 8 个通道])
    gl.barrier()                                       # 保护：X(j+1) 写完才在 M(j+1) 读；X(j−1) 读完它的槽才被覆盖

def compute_phase(j, store):                           # C(j)，前后 setprio(1) / setprio(0)
    acc = cdna3.mfma(wf0, xb0, 0 if j == 0 else acc)   # 2 行 tile × 8 列 tile × 2 = 32 条；新 tile 从 0 累加
    wf1 = decode(wr[j % 2].half1, lut_r)               # W1(j)，VALU 交织在上面 32 条 MFMA 之间
    acc = cdna3.mfma(wf1, xb1, acc)                    # 32 条；store 时按 token 块两两发
    lut_r = lut(dr[(j+1) % 2], 2 * ((j+1) % 2) + g // 2)
    wf0 = decode(wr[(j+1) % 2].half0, lut_r)           # W0(j+1)，交织在后半的 MFMA 之间
    if store:                                          # 与后半 MFMA 穿插：上一对 token 块缩放、bf16 RNE、写出
        cdna3.buffer_store((acc * ws_r * xs_r).to(bf16), y[tile 行, tile 通道])   # 每 lane 每 16-token 块 16 B
    gl.barrier()
```
与代码顺序的差别：代码里 tile 内的 K 步按对展开（槽与 LDS 缓冲按奇偶静态确定），首尾两对剥出，中间成对的运行时循环
（K = 512 时 0 次）；计算相的解码 VALU 由 `sched_group_barrier` 均匀插进 MFMA 之间；store 那一相的后半按 token 块两两发
MFMA（同一累加器的两条隔 4 条），第一对期间解码下一 tile 的 W0(0)，之后每对期间写上一对，最后一对在末条 MFMA 后写。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | G = min(tile 数, CU 数)，持久化 | CTA c 依次做 tile c, c+G, …（N 块最快） |
| CTA（512 线程） | 8 wave = 2 组 × 4，每 SIMD 各一个 | y[128 token, 256 通道]；X tile 经 LDS 共享 |
| wave 组 | 组 0 = wave 0–3，组 1 = wave 4–7 | 各搬 X(j+1) 的一半行；相位错开一相 |
| wave | 32 通道 × 128 token | 2 个 MFMA 行 tile × 8 个 16-token 列 tile |
| lane | lane 组 g、行 r | 行 tile m 的行 r = 通道 32w+2r+m；出 8 个通道 |
| K 步 | 128 K = 两个 64-K 半步 | 每个计算相 64 条 MFMA；每步一个 LUT |

通道映射 32w+2r+m 让一个 lane 两个行 tile 的码在 fd2 里相邻：每半步一条 16 B 加载读满 8 条 128 B 行；输出时 lane 组 g
持有同一 token 的通道 8g…8g+7，每个 16-token 块一条 16 B 写。xq、xs、y 的 buffer 基址在每个 tile 移到首行（64 位），
≥ M 的行读 0、写丢弃；权重（< 2 GiB）用全局偏移。同时运行的 CTA 按 N 块最快取 tile，共享少数几个 X 的 M 块。

**流水线**（相位表与 tile 交界见 [gemm_large.py 模块 docstring](gemm_large.py#L26-L53)）
```text
序言（每 CTA 一次）：VMEM X(0)、W0(0)+D(0)、W1(0)、W0(1)+D(1) → vmcnt(0) → X(0) 写 L[0]、VMEM X(1) → LUT、解码 W0(0) → barrier
组 1 再等一个 barrier；之后每个 wave 依次 M(0) C(0) M(1) C(1) … C(T−1) M(0′) …，每相以一次 CTA barrier 结束
格 = 两次 barrier 之间：格 2j 组 0 做 M(j)、组 1 做 C(j−1)；格 2j+1 组 0 做 C(j)、组 1 做 M(j)
预取距离：X 的 VMEM 提前 2 步、写 LDS 提前 1 步；W 前半与 delta 提前 2 步、后半提前 1 步；解码提前半步
tile 交界：M(T−2) 起换读下一 tile 的 X(0)、W0(0)、D(0)，并读本 tile 的 xs、ws；M(T−1) 读下一 tile 的 X(1)、W1(0)、W0(1)；
  C(T−1) 写完本 tile，接着是下一 tile 的 M(0)；CTA 的最后一个 tile 把自己当“下一 tile”多读一遍（不写 y）
```
| 相 | MFMA | LDS → VGPR | VMEM（每 lane） | VGPR → LDS | VALU |
|---|---|---|---|---|---|
| M(j) | — | X(j) 两个半步 | 读 X(j+2) 2×16 B、W1(j+1) 16 B、W0(j+2) 16 B、D(j+2) 8 B | X(j+1) 本组行 | — |
| C(j) 前半 | 32：W0(j)·X0(j) | — | — | — | 解码 W1(j) |
| C(j) 后半 | 32：W1(j)·X1(j) | — | — | — | 建 LUT(j+1)、解码 W0(j+1) |
| C(T−1) 后半 | 4 对 token 块 × 8 条 | — | 写 y：上一对块各 16 B | — | 第一对建 LUT、解码 W0(0′)；其余缩放、bf16 RNE |

**命名**与**流水线不变量**：见 [gemm_large.py 模块 docstring](gemm_large.py#L54-L76)（Naming、Pipeline rules 1–7）。

**资源**：LDS 32 KB（2 × 128 × 128 B）；256 VGPR（无 AGPR）、58 SGPR、无 spill；`waves_per_eu = 2`，每 CU 1 个 CTA。

## gemm_a8w4_ptpc_decode_rb{rb}_ks{ks}_m{mcap}_g{G}（[代码](small_m.py#L251-L539)）
M ≤ 8 时的整个调用：一次 launch 里把 x 按 token 量化进 LDS 并算完 y；块对映射让 ≤ 8 个 token 也不空转 MFMA 的半边。

**Gluon 风格**
```python
# grid = (G,)，G、rb、ks 按 (M, N, K) 查表（small_m_config，目标形状 G = 64–128）；线程 = 64·rb·ks（8 或 16 wave）
# CTA b 管 16 行块 [b·NB/G, (b+1)·NB/G)（NB = N/16，至多 rb 个）× 全部 token（M 补到 mcap = 1、2、4、8）× 全部 K
# 放置：x 行 global → VGPR（lane l 读行内第 l + 64u 个 16 B 单元）→ CTA 内量化成 fp8 写 LDS（mcap 行 × (K+16) B），
#   xs 进 LDS；本 CTA 行块的 delta global → LDS；权重 global → VGPR 直读，本 wave 的全部块对一次发完（每块对每 lane
#   一条 16 B）；不预取、不流水（K 由 wave 分段并行）；acc 在 VGPR；K 段部分和经 LDS 相加；收尾直写 y
def decode(x, wq, wd, ws, y, M):
    w, l = wave_id(), lane_id()
    slot, kpart = w % rb, w // rb                    # 行块槽、K 段：块对 [kpart·P/ks, (kpart+1)·P/ks)，P = K/128
    blk = b·NB/G + slot; valid = slot < 本 CTA 的行块数  # 越过的槽照算相邻块、不写
    X = gl.allocate_shared_memory(fp8, [mcap, K + 16])  # 另有 D（delta）、XS（xs）、Pt（K 段部分和）
    # 序言：先发全部 VMEM，按使用顺序（sched_barrier 隔开：vmcnt 按发射顺序计数）
    xr[j] = cdna3.buffer_load(x[w + rb·ks·j, 单元 l + 64u])   # 本 wave 要量化的行；≥ M 的行读 0
    dr = cdna3.buffer_load(wd[本 CTA 行块])                   # 全 CTA 线程分摊
    wr[s] = cdna3.buffer_load(wq[blk, 块对 s, 第 l 个 16 B])  # 块 l//32、lane 组 (l%32)//8 的 2 行 × 16 个码
    wsr = cdna3.buffer_load(ws[16·blk + 2·(l%8) + h] for h in (0, 1))
    for j:                                            # CTA 内量化：wave w 管 token 行 w, w + rb·ks, …（< M 的）
        sj = row_scale(as_f32(wave_max(max_u16(bits(xr[j]) & 0x7FFF)) << 16))
        XS[行] = sj（lane 0）; X[行, 单元 l + 64u] = quant(xr[j], sj)     # 每单元 8 B
    D.store(dr)
    gl.barrier()                                      # 保护：X、XS、D 写完才读
    xa[s] = X.load(token l%8, 块对 s 中块 l//32、lane 组 (l%32)//8 的 16 B)   # 先读全部块对的 x 与 delta
    dd[s] = D.load(blk, 半区 (l%32)//16, 通道 2·(l%8) 与 2·(l%8)+1, 块对 s)
    for s in 本 wave 的块对:                            # 展开；每块对的链只剩 等权重 → 解码 → MFMA
        for h in (0, 1):                              # lane 的两行 = 通道 2·(l%8)+h
            lr = lut(dd[s][h], 2·(s%2) + l//32)
            for c in (0, 1):                          # 16 个码 = 两条 MFMA 的 K
                acc[h] = cdna3.mfma(xa[s].k(c), decode(wr[s][h][c], lr), acc[h])
                # a 行 t 与 8+t 都是 token t（K 分属偶 / 奇 lane 组），b 列 j 与 8+j 都是通道 2·(j%8)+h
    v = acc + shuffle_xor(acc, 40)                    # 列 j 的偶组半段 + 列 8+j 的奇组半段 = 本 K 段的完整和
    if ks > 1:                                        # 有用 lane：l < 32 且 l%16 < 8，持 token 4·(l//16)+v
        if kpart > 0 and valid and 有用 lane: Pt[slot, kpart] = v
        gl.barrier()                                  # 保护：各 K 段部分和写完才读
        v = v + Pt[slot, 1] + … + Pt[slot, ks−1]      # 按 K 段号顺序：结果与 wave 时序无关
    if kpart == 0 and valid and 有用 lane:
        cdna3.buffer_store((v · wsr · XS[token]).to(bf16), y[token, 16·blk + 2·(l%8) + h])  # ≥ M 的 token 越界丢弃
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | G（查表，目标形状 64–128） | CTA b：连续 ≤ rb 个 16 行块 × 全部 token × 全部 K |
| CTA（64·rb·ks 线程） | rb·ks = 8 或 16 个 wave | 量化全部 M 行进 LDS；本 CTA 的 delta 进 LDS |
| wave | 1 个 16 行块 × K/ks | 本 K 段的全部块对；量化 token 行 w, w+rb·ks, … |
| lane | 块对内第 l 个 16 B | 2 个通道 × 16 个码；A 为 token l%8 的同段 x（16 B） |
| 块对（K 步） | 128 K = 两个 64-K 块 | 每 wave 4 条 MFMA、2 个 LUT |

CTA 数不整除行块数时（如 tp2_qkv：416 个行块分给 80 个 CTA）各 CTA 管 5 或 6 块，多出的槽照算相邻块、不写。
y 的 buffer 记录数是 M 行：≥ M 的 token（含 MFMA 8 行里多出的）照算照写，写被硬件丢弃。

**流水线**（不流水；阶段表见 [small_m.py 模块 docstring](small_m.py#L30-L40)）
```text
序言：VMEM x 行（每 lane K/512 条 16 B × ⌈mcap/波数⌉ 行）→ delta（16 B，全 CTA 分摊）→ 权重（每块对 1 条 16 B）→ ws
量化：等 x → lane 内 max_u16 + 6 次 shuffle_xor 得 amax → scale → 倒数与两次 FMA（packed f32）→ fp8 → 8 B 写 LDS
delta 写 LDS → barrier → 全部块对的 x（16 B）与 delta（8 B）读进 VGPR
块对 s（展开 P/ks 个）：等 wr[s] → 2 个 LUT → 解码 4 个 dword → 4 条 MFMA
收尾：lane^40 相加 → K 段 1..ks−1 写部分和（每 lane 8 个 f32）→ barrier → K 段 0 按段号相加 → × ws × xs → bf16 → 4 B 写
```

**命名**与**流水线不变量**：见 [small_m.py 模块 docstring](small_m.py#L42-L53)（Naming、Rules 1–4）。

**资源**：LDS = mcap·(K+16) + rb·⌈K/256⌉·128 + 部分和 rb·(ks−1)·512 B（目标形状 6–34 KB）；68–124 VGPR、无 spill。
LDS 随 mcap·K 增长，分派没有检查：K 大时（如 M 5–8 且 K ≥ 7168）超过 64 KB、编译失败（见入口文档第 6 节）。

## gemm_a8w4_ptpc_tile_rb{rb}_ks{ks}_t{tq}_kp{kp}_g{G}（[代码](small_m.py#L542-L810)）
9 ≤ M ≤ 32 时的整个调用：与 decode 一样在 CTA 内量化，换标准映射（权重作 A、16-token 片作 B），K 分窗口让 32 行 fp8 装进 LDS。

**Gluon 风格**
```python
# grid = (G,)，G = ⌈NB/rb⌉（rb、ks、kp 查表，目标形状 G = 52–80）；线程 = 64·rb·ks（8 或 16 wave）；每 CU 1 个 CTA
# CTA b 管 16 行块 [b·NB/G, (b+1)·NB/G)（至多 rb 个）× tq 个 16-token 片（M ≤ 16 时 1 个，否则 2 个）× 全部 K
# 放置：x 行 global → VGPR（同 decode，全部 K 一次读完），按 K 窗口量化成 fp8 写 LDS（16·tq 行 × 窗口宽 + 16 B，
#   kp 个窗口轮流用同一块）；xs、delta 进 LDS；权重 global → VGPR 直读，本 wave 在各窗口的全部 K 步一次发完（每步
#   每 lane 2 × 8 B）；不预取、不流水；acc 在 VGPR；K 段部分和经 LDS（与 X 合计放不下时复用 X 区）；收尾直写 y
def tile(x, wq, wd, ws, y, M):
    w, r, g = wave_id(), lane_id() % 16, lane_id() // 16   # r：a 的通道行、b 的 token 列；g：MFMA 的 K 块
    slot, kpart = w % rb, w // rb; blk = b·NB/G + slot; valid = slot < 本 CTA 的行块数
    X = gl.allocate_shared_memory(fp8, [16·tq, 最宽窗口 K + 16])   # 另有 D、XS、Pt
    # 序言：先发全部 VMEM（按使用顺序，sched_barrier 隔开）
    xr[j] = cdna3.buffer_load(x[w + rb·ks·j, 单元 l + 64u])     # 16·tq 行轮流分给各 wave；≥ M 的行读 0
    dr = cdna3.buffer_load(wd[本 CTA 行块])
    wr[p][s][h] = cdna3.buffer_load(wq[blk, 通道 r, 块 2s + g//2, lane 组 2·(g%2) + h, 8 B])  # 窗口 p 中本 K 段的步
    wsr = cdna3.buffer_load(ws[16·blk + 4g … 4g+3])
    sc[j] = row_scale(…)（整行，各窗口共用）; XS[行] = sc[j]（行 < M，lane 0）; D.store(dr)
    for p in range(kp):                               # K 窗口：整 512-K 单元，大的在前
        if p > 0: gl.barrier()                        # 保护：上一窗口的 X 读完才覆盖（WAR）
        X[行, 窗口内单元] = quant(xr[j][窗口 p 的单元], sc[j]) for 行 < M   # 每单元 8 B
        gl.barrier()                                  # 保护：本窗口的 X（首个窗口还有 XS、D）写完才读
        for s in 本 K 段在窗口 p 的 spw 个连续步:         # 展开
            lr = lut(D[blk, 半区 g%2, 通道 r, 步 s], 2·(s%2) + g//2)   # 每步每 lane 一个 LUT（32-K 组 4s+g）
            xb[t] = X.load(token 16t + r, K 128s + 32g 起 32 B) for t in range(tq)
            for h in (0, 1): for c in (0, 1):
                acc[t] = cdna3.mfma(decode(wr[p][s][h][c], lr), xb[t].k(h, c), acc[t])   # a = 16 通道，b = 16 token
    if ks > 1:
        if Pt 复用 X: gl.barrier()                     # 保护：X 全部读完才写部分和
        if kpart > 0 and valid: Pt[slot, kpart, t] = acc[t]   # 每 lane 每片 16 B
        gl.barrier()                                  # 保护：部分和写完才读
        acc[t] = acc[t] + Pt[slot, 1, t] + … + Pt[slot, ks−1, t]   # 按 K 段号顺序
    if kpart == 0 and valid and 16t + r < M:
        cdna3.buffer_store((acc[t] · wsr · XS[16t + r]).to(bf16), y[16t + r, 16·blk + 4g … 4g+3])   # 8 B
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | G = ⌈NB/rb⌉（目标形状 52–80） | CTA b：连续 ≤ rb 个 16 行块 × 全部 token × 全部 K |
| CTA（64·rb·ks 线程） | rb·ks = 8 或 16 个 wave | 量化 < M 的 token 行进 LDS；delta 进 LDS |
| wave | 1 个 16 行块 × 每个窗口的 1/ks | tq 块 y[16 token, 16 通道]；量化 token 行 w, w+rb·ks, … |
| lane | 通道 r = l%16、K 块 g = l//16 | 每步每半 8 B 码；每片 32 B x；每片出 4 通道 × 1 token |
| K 步 | 128 K | 每片 4 条 MFMA；1 个 LUT |
| K 窗口 | kp 个，整 512-K 单元 | 本窗口 16·tq 行 fp8 在 LDS（tq = 2 且 K = 2560、3072 时 kp = 2） |

lane 组 g 每步只读第 4s+g 个 32-K 组，所以每 lane 每步一个 delta、一个 LUT；部分和与 X 合计超过 64 KB 时
（目标形状里 N ≥ 3584 的四个在 M 17–32 时）复用 X 区，前面多一次 barrier。

**流水线**（不流水；阶段表见 [small_m.py 模块 docstring](small_m.py#L30-L40)）
```text
序言：VMEM x 行（每 lane K/512 条 16 B × ⌈16tq/波数⌉ 行）→ delta → 各窗口权重（每步每半 1 条 8 B）→ ws
求每行 scale（整行，各窗口共用）→ lane 0 写 xs；delta 写 LDS
窗口 p：[p > 0：barrier] → 量化窗口 p 的单元 → 8 B 写 LDS → barrier
  步 s（展开 spw 个）：LDS 读 delta 4 B、每片 x 32 B → 建 LUT → 每半解码 2 个 dword → 每片 2 条 MFMA
收尾：[部分和复用 X：barrier] → K 段 1..ks−1 写部分和（每 lane 每片 16 B）→ barrier → 按段号相加 → × ws × xs → bf16 → 8 B 写
```

**命名**与**流水线不变量**：见 [small_m.py 模块 docstring](small_m.py#L42-L53)（Naming、Rules 1–4）。

**资源**：LDS = 16·tq·(最宽窗口 K + 16) + rb·⌈K/256⌉·128 + 部分和 rb·(ks−1)·tq·1 KB（可复用 X）+ xs；目标形状 57–64 KB，
每 CU 1 个 CTA；125–169 VGPR、无 spill。x 行与全部权重常驻 VGPR，寄存器随 K 增长：K = 8192 时 256 VGPR 且有 spill。
