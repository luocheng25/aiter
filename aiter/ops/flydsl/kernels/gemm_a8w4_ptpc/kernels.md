# gemm_a8w4_ptpc：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程；与代码对不上就改这里。GEMM 的 region 表、命名与流水线不变量写在 gemm.py 的模块 docstring，
     这里链接过去，只补 Gluon 风格写法、任务划分与资源。 -->

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
# acc = cdna3.mfma(a, b, acc)      fp8 MFMA 16x16x32，f32 累加：acc[通道, token] += a[通道, K]·b[K, token]
# —— 伪操作 ——
# wave_id() / lane_id()：wave / lane 编号；lane 组 g = lane // 16，行 r = lane % 16；tid() = 64·wave_id() + lane_id()
# lut(d, j)：delta 字节 d[j] → 8 项 fp8 表 |e2m1| × 2^(5 − delta)（v_perm 广播 + 两次减法）
# decode(wr, lut)：一个 fp4 dword（8 个码）→ 8 个 fp8（两次 v_perm 查表 + 符号位）
# wave_max(v)：同一行的 lanes 之间求 max（shuffle_xor）；to_fp8(v)：RNE 转 fp8（v_cvt_pk_fp8_f32）
# —— 语义 ——
# row_scale(amax) = amax / 240（IEEE 除法），为 0（全零行）时取 1
# quant(v, s) = to_fp8(clamp(v / s, ±240))，v / s 为 IEEE 商（倒数乘加两次 FMA 修正；s < 2^-64 时逐元素除）
# —— 调用顺序（一档到底）——
# flydsl_gemm_a8w4_ptpc(x, w)：quant_x(x) → xq, xs；gemm(xq, xs, wq, wd, ws) → y（两次 launch）
```

## gemm_a8w4_ptpc_quant_k{K}（[代码](quant.py)）
把 bf16 x 按 token 量化成 fp8 与 f32 scale，GEMM 之前一次 launch；结果与 torch 逐位一致。

**Gluon 风格**
```python
# grid = (ceil(M / R),)，256 线程；每行 L 个 lane（K % 512 == 0 时 L = 64，如 6 个目标形状；否则 32，如 K = 256、768），
#   R = 256 / L 行
# 放置：x 行段 global → VGPR 直读（每 lane 连续 K / L 个 bf16，16 B 一次）；不用 LDS；xq、xs 直写
def quant_x(x, xq, xs, M):
    row = gl.program_id(0) * R + tid() // L           # 本 lane 的行；≥ M 的行读 0、写丢弃
    sub = tid() % L                                   # 行内第 sub 段
    v = cdna3.buffer_load(x[row, sub * K // L : (sub + 1) * K // L])
    s = row_scale(wave_max(abs(v).max()))             # 行内 L 个 lane 归约
    if sub == 0:
        cdna3.buffer_store(s, xs[row])
    cdna3.buffer_store(quant(v, s), xq[row, sub * K // L : (sub + 1) * K // L])   # 每次 8 B
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ceil(M / R) | 每 CTA R 行（R = 4 或 8） |
| CTA（256 线程） | R 行 × L lane | 互不相关的 R 行 |
| lane | K / L 个元素 | 一行的一段：读、量化、写 |

x、xq、xs 的 buffer 基址在每个 CTA 移到它的首行（64 位），任意 M 都不溢出 32 位偏移。

**流水线**：无循环。每 lane 发全部 16 B 读 → 段内 max → 行内 shuffle 归约 → 求 scale → 量化 → 8 B 写。

**资源**：无 LDS；36 VGPR、19 SGPR（K = 1536）。

## gemm_a8w4_ptpc_bm64_bn128_tk128（[代码](gemm.py)）
算 y = (xq·xs) · (fd2 解码后的 W · ws)ᵀ：权重在寄存器里按每组 LUT 解成 fp8，走 fp8 MFMA，收尾乘两个 scale 写 bf16。

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
