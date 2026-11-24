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
# wave_max(v)：同一行的 lanes 之间求 max（小 M 核用 shuffle_xor；quant_x 用 DPP 换位 + readlane，不经 LDS）
# to_fp8(v)：RNE 转 fp8（v_cvt_pk_fp8_f32）
# shuffle_xor(v, m)：取 lane ^ m 的 v（wave 内跨 lane 交换）
# max_u16(v)：bf16 位模式按 16 位无符号两两取 max（v_pk_max_u16）；bits(v) / as_f32(u)：按位重解释
# setprio(p)：本 wave 的发射优先级（s_setprio）；计算相设 1，MFMA 先于同 SIMD 另一个 wave 的访存发射
# atomic_add(R, v) / R.load(acquire) / s_sleep(n)：LDS 计数器的 release 原子加、acquire 读；让出发射若干周期
# —— 语义 ——
# row_scale(amax) = amax / 240（IEEE 除法），为 0（全零行）时取 1；amax 为 NaN 时 scale 为 NaN
# quant(v, s) = to_fp8(v / s)，v / s 为 IEEE 商（每行一次倒数，乘后 FMA 修正：quant_x 一步、小 M 核两步；
#   s < 2^-64 时逐元素除）；|v / s| 至多略超 240（< 248），RNE 舍到 ±240，所以不必 clamp；NaN 仍是 NaN（0x80）
# —— 调用顺序（入口 _gemm_choice 按 (N, K) × M 档分派；各目标形状的 M 范围见入口文档第 2 节）——
# flydsl_gemm_a8w4_ptpc(x, w)：
#   小 M（目标形状 M ≤ 16，tp4_o ≤ 20；表外 M ≤ 32）且 K % 512 == 0：一次 launch，small_m_config 依次试
#     dec2（M ≤ 4、NB > 320、xw + rb·ks ≤ 16）→ decode（M ≤ 8）→ tile，取第一个 LDS ≤ 64 KiB 的 → y；都不行走下面
#     （tp2_qkv 的 tile 核由入口改成 6 个行块 × 80 个 CTA，让 80 个 CU 都有活）
#   其余：quant_x(x)（配置按 M、K 选）→ xq, xs；然后按 M 档：gemm(cfg)(xq, xs, wq, wd, ws) → y
#     或 gemm_large(panels = xcd_panels(M, N, K, CU 数))(xq, xs, wq, wd, ws) → y，两次 launch 的参数先建好再连发
#     表外：M > 2048 且 N % 256 == 0、K ≥ 512 走 gemm_large；否则 gemm：M ≤ 128 用 32×64，到 2048 用 64×128
#     （N % 128 == 64 时 64×64），更大用默认 64×128·TK128
```

## gemm_a8w4_ptpc_quant_k{K}_l{L}_r{R}（[代码](quant.py)）
把 bf16 x 按 token 量化成 fp8 与 f32 scale，GEMM 之前一次 launch；结果与 torch 逐位一致（±Inf 行除外：整行写 NaN 码）。
两种配置（`quant_config`）：≤ 128 行且 K % 512 == 0、K ≤ 8192 时**宽配置**，每 CTA 一行、L = K/8 个 lane（R = 1）；
否则**窄配置**，每 CTA 一个 wave，L = 64（K % 512 ≠ 0 时 32），R = 64/L 行。

**Gluon 风格**
```python
# grid = (ceil(M / R),)，L·R 线程；宽配置一行跨 W = L/64 个 wave（目标 K：3、5、6），窄配置 W = 1
# 放置：x 行段 global → VGPR 直读，lane sub 的第 v 次 16 B 读行内第 sub + v·L 个 16 B 单元（同一条指令的 L 个 lane
#   读连续的 16·L B；宽配置每 lane 只读 1 次）；xq 同样交错直写（第 v 次 8 B 写第 sub + v·L 个 8 B 单元），xs 由
#   sub == 0 的 lane 写；只有宽配置用 LDS（每 wave 一个 u32 的 max）；窄配置的 xq 写穿 L2（sc0|sc1）
def quant_x(x, xq, xs, M):
    row = gl.program_id(0) * R + tid() // L           # 本 lane 的行；≥ M 的行读 0、写丢弃
    sub = tid() % L                                   # 行内第 sub 个 lane
    for v in range(K // (8 * L)):                     # 展开，先发全部读（宽配置 1 次，窄配置 K/512 或 K/256 次）
        u = sub + v * L                               # 行内第 u 个 16 B 单元 = 元素 8u … 8u+7
        r[v] = cdna3.buffer_load(x[row, 8 * u : 8 * u + 8])
    amax = wave_max(max_u16(bits(r) & 0x7FFF))        # 去符号的 bf16 位模式与 |x| 同序、NaN 最大；DPP + readlane
    if W > 1:                                         # 宽配置：合并一行 W 个 wave 的 max
        if lane_id() == 0:
            Lmax[wave_id()] = amax
        gl.barrier()                                  # 保护：W 个 wave 的 max 都写进 LDS 才读
        amax = max(Lmax[0 : W])
    s = row_scale(as_f32(amax << 16))
    if sub == 0:
        cdna3.buffer_store(s, xs[row])
    for v in range(K // (8 * L)):
        u = sub + v * L
        cdna3.buffer_store(quant(r[v], s), xq[row, 8 * u : 8 * u + 8])   # 每次 8 B；窄配置写穿 L2
```

**任务划分**
| 层级 | 宽配置（≤ 128 行，K % 512 == 0） | 窄配置（其余） |
|---|---|---|
| grid | M 个 CTA | ceil(M / R) 个 CTA，R = 64/L（1 或 2） |
| CTA | K/8 线程 = W 个 wave，管 1 行 | 1 个 wave，管 R 行 |
| lane | 1 个 16 B 单元（8 个元素） | K/L 个元素：交错的 K/(8L) 个 16 B 单元 |

x、xq、xs 的 buffer 基址在每个 CTA 移到它的首行（64 位），任意 M 都不溢出 32 位偏移。宽配置每 lane 只有一条加载，
从读到写的依赖链最短（适合小 M）；窄配置每 lane 多条读同时在途、无 barrier，适合大 M。窄配置的 xq 写穿 L2：上一次
GEMM 把 xq 行留在多个 XCD 的 L2 里，带缓存的写会慢（tp4_qkvz M 2048 时 10.4 → 7.1 µs）；宽配置带缓存写快 0.05–0.1 µs。

**流水线**：无循环。每 lane 发全部 16 B 读 → 段内 bf16 位模式两两取 max → wave 内 DPP（quad 置换、行镜像、行广播）
+ readlane 归约 →（宽配置 W > 1：LDS 写 → barrier → 读 W 个 max）→ 求 scale → 每行一次 IEEE 倒数 → 打包 f32 的
一步 Markstein 修正（s < 2^-64 的行逐元素除）→ fp8 → 8 B 写。

**资源**：宽配置 26 VGPR、16 SGPR、LDS 4·max(W, 4) B；窄配置无 LDS，52 VGPR（K = 1536）、78 VGPR（K = 2560）。

## gemm_a8w4_ptpc_bm{BM}_bn{BN}_tk{TK}_p{P}_x{XS}_g{xcd_m}_kg{kg}（[代码](gemm.py#L184-L744)）
算 y = (xq·xs) · (fd2 解码后的 W · ws)ᵀ：权重在寄存器里按每组 LUT 解成 fp8，走 fp8 MFMA，收尾乘两个 scale 写 bf16。
tile 由 `GemmConfig`（BM, BN, P, XS, xcd_m, kg, TK）给出，入口按 (N, K) × M 档查表选；表外与大 M 核不接的形状用通用回退。

**Gluon 风格**
```python
# grid = (⌈⌈M/BM⌉/xcd_m⌉ · ⌈NB/xcd_n⌉ · 4,)，NB = ⌈N/BN⌉，xcd_n = 4/xcd_m；256·kg 线程 = kg 个 4-wave 组
# 一个 CTA 算 y[BM token, BN 通道]；组 grp 只算 K tile [grp·T', (grp+1)·T')，T' = K/TK/kg
# 放置：本组 xq tile 经 LDS 双缓冲（2 槽 × BM × TK B，XOR swizzle），global 读先进 VGPR 暂存（XS 组），提前 XS 个 tile；
#   权重 global → VGPR 直读，P+1 槽，提前 P 个 tile；解码出的 fp8 A 操作数 wf 比用它的 MFMA 早一个 64-K 子步；
#   acc（BN/64 × BM/16 个 16×16 tile）在 VGPR；K 组部分和经 LDS（复用 X 区）；收尾由组 0 做：输出 tile 放得下时
#   （全部 TK = 256 的配置）先写进 LDS 的 O（复用 X 区，行距 2·BN+16 B），再每 lane 16 B 整行写 y；否则每 lane 8 B 直写
def gemm(xq, xs, wq, wd, ws, y, M):
    c = gl.program_id(0); xcd, sup = c % 4, c // 4            # CTA 按线性号轮流派到 4 个 XCD
    mb = sup // super_n * xcd_m + xcd // xcd_n                 # 同一 XCD 总拿 M 块 ≡ xcd // xcd_n（mod xcd_m）
    nb = sup % super_n * xcd_n + xcd % xcd_n                   # super_n = ⌈NB/xcd_n⌉
    if nb >= NB or mb * BM >= M: return                        # 边缘超块里没有 tile 的 CTA
    grp, w = wave_id() // 4, wave_id() % 4                     # wave w 管每个 64 通道组的第 w 个 16 通道块 ch(m)
    L = gl.allocate_shared_memory(fp8, [kg, 2, BM, TK])
    # 序言：X(0) → L[grp][0]；X(1 … XS−1) 进暂存、W(0 … P−1) 进槽；解码 W(0) 子步 0；barrier
    for k in range(T'):                                        # 运行时循环每次 U = lcm(P+1, 2, XS) 个阶段，最后 U…2U−1 个
                                                               #   剥出（T' < U 时全部剥出）
        if k + XS < T': xr[k % XS] = cdna3.buffer_load(xq[mb 行, tile k+XS])            # 预取距离 XS
        if k + P < T': wr[(k+P) % (P+1)], dr[…] = cdna3.buffer_load(wq / wd[ch(m), tile k+P])  # 距离 P
        for ki in range(TK // 64):                             # 64-K 子步；VMEM 均匀插在本阶段全部 MFMA 之间（P = 1：头几条）
            xb = L[grp][k % 2].load(B 布局, 子步 ki)           # K 置换：MMA 侧 64h+16g+8c+v ← 存储 32g+16h+8c+v
            if ki > 0: wf = decode(wr[k % (P+1)][ki], lut)      # 偶数子步建 LUT（每 128 K 一个），奇数子步复用
            acc = cdna3.mfma(wf, xb, acc)                       # 每子步 2 · BN/64 · BM/16 条
        if k + 1 < T':
            wf = decode(wr[(k+1) % (P+1)][0], lut(dr, …))       # 下一 tile 的子步 0
            L[grp][(k+1) % 2].store(xr[(k+1) % XS])             # X(k+1) 暂存 → LDS
        gl.barrier()                                           # 保护：下一阶段覆盖 L[k%2]（WAR）、读 L[(k+1)%2]（RAW）
    for h in (kg // 2, kg // 4, …, 1):                        # K 组树形相加，顺序固定（确定性）
        if h <= grp < 2 * h: P_lds[grp − h] = acc               # 复用 X 区
        gl.barrier()
        acc = acc + P_lds[grp % h]                              # 每个 wave 都加（acc 不进分支）；组 ≥ h 的和此后不再用
        if h > 1: gl.barrier()
    if kg > 1: gl.barrier()                                   # 保护：部分和都读完，O 才覆盖 X 区
    if grp == 0: O[token, 通道] = (acc * ws[通道] * xs[token]).to(bf16)   # 每 lane 每 tile 8 B 写 LDS
    gl.barrier()                                               # 保护：O 写完才读
    if grp == 0:                                               # 每 lane 16 B = 一行的 8 个通道；单元数 < 线程数时多余 lane 不写
        cdna3.buffer_store(O[行, 8 个通道], y[mb 行, nb 通道])   # ≥ M 的行越界丢弃；N % BN ≠ 0 时跳过 ≥ N 的通道
```
与代码顺序的差别：一个阶段的 VMEM、LDS 读、解码与 MFMA 由 `hot_loop_scheduler` 交错排布（见下）；循环体静态展开 U 个阶段，
所以槽、LDS 缓冲与暂存组都是常量下标；TK = 128 且 K 组从奇数 tile 开始时，LUT 的 delta 字节奇偶在运行时翻转。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈⌈M/BM⌉/xcd_m⌉ · ⌈NB/xcd_n⌉ · 4，1-D | CTA c 到 XCD c % 4；超块 = xcd_m 个 M 块 × xcd_n 个 N 块 |
| CTA（256·kg 线程） | kg 个 4-wave 组 | y[BM, BN]；组 g 管 K 的第 g 段，xq tile 经本组 LDS 共享 |
| wave | BN/64 个 16 通道块 × BM token | 权重块直读；BM 行 x 从 LDS 读 |
| lane | lane 组 g、行 r | 每 128 K 读第 g 个 32-K 组的码与 delta；出 4 通道 × 1 token / tile |
| K 步（阶段） | TK = 128 / 256 = 2 / 4 个 64-K 子步 | 每子步 2·(BN/64)·(BM/16) 条 MFMA；每 128 K 一个 LUT |

xcd_m = 4 时一个 X 块只经过一个 XCD 的 L2：X 行散在多个 L2 时，下一次 quant_x 重写同一 xq 会慢至多约 15 µs。
xq、xs、y 的 buffer 基址在每个 CTA 移到本 tile 首行（64 位），任意 M 都不溢出 32 位偏移；权重（< 2 GiB）用全局偏移。

**流水线**（K 步 region 表、编号的不变量见 [gemm.py 模块 docstring](gemm.py#L39-L92)）
```text
序言：VMEM X(0)、W(0)，X(1…XS−1)、W(1…P−1) → X(0) 写 L[0] → 解码 W(0) 子步 0 → barrier
阶段 k（读 L[k%2]、W 槽 k%(P+1)、写 L[(k+1)%2]）：
  sched：先排子步 0 的 LDS 读；本阶段的 VMEM（X(k+XS) 进暂存 k%XS，W/D(k+P) 进槽 (k+P)%(P+1)）均匀分到全部 MFMA
    之间（P = 1 时各跟头几条 MFMA：W(k+1) 在本阶段末解码）；连发会塞满 CU 的向量访存队列，卡住各 wave 的 MFMA 发射
  子步 ki：MFMA 之间插子步 ki+1 的 LDS 读与解码 W(k) 子步 ki+1（末子步解码 W(k+1) 子步 0）的 VALU
  末子步最后几条 MFMA 各带一条 LDS 写 X(k+1) → sched_barrier → barrier
主循环：每次 U 个阶段（槽、缓冲、暂存组为常量），累加器是循环携带值；剥出的阶段到组的末 tile 为止停发 VMEM
K 组（kg > 1）：组 [h, 2h) 写部分和 → barrier → 全部 wave 加 → barrier，h = kg/2 … 1
收尾：[kg > 1：barrier] → 组 0 把 acc × ws × xs 的 bf16 写 LDS（每 lane 每 tile 8 B）→ barrier → 组 0 每 lane 16 B 整行写 y
  （输出 tile 放不下，即默认 64×128·TK128：组 0 每 lane 每 tile 8 B 直写 y）；ws、xs 在收尾才读，不挤 K 循环的首批加载
```

**命名**与**流水线不变量**：见 [gemm.py 模块 docstring](gemm.py#L65-L92)（Naming、Pipeline rules 1–7）。

**资源**：LDS = max(kg·2·BM·TK, kg/2·BM·BN·4) B（X 区与 K 组部分和共用）：目标形状的配置 16–32 KB，默认 64×128·TK128 为 16 KB；
收尾的输出 tile（BM·(2·BN+16) B）放得下时复用这块，不另占 LDS。

## gemm_a8w4_ptpc_pp_bm128_bn256_tk128（[代码](gemm_large.py)）
表交给它的 M（目标形状从 129 / 513 / 1025 起）与表外 M > 2048 的 GEMM，要 N % 256 == 0、K ≥ 512：运算与 gemm 相同
（同一 fd2 解码、同一 K 序），换成持久化的 128 token × 256 通道 tile、两组 wave ping-pong，收尾写在 tile 最后一个 K 步里。

**Gluon 风格**
```python
# grid = (G,)，G = min(tiles, num_cus)，tiles = ⌈M/128⌉ · NB，NB = N / 256；512 线程 = 8 wave，每 CU 一个 CTA
# 一个 tile 算 y[128 token, 256 通道]；tile 次序 panels 是编译期常量，入口每次调用按 M 用 xcd_panels 选（核名不变）：
#   0（原次序）：tile t = (M 块 t // NB, N 块 t % NB)，CTA c 依次做 t = c, c + G, …
#   1 / 2 / 4：N 块切成宽度差 ≤ 1 的面板，全部 tile 按面板、面板内 M 主序排成一列，切成 4 段；XCD x（c % 4 == x）的
#   CTA 依次取第 x 段的第 c // 4, c // 4 + G/4, … 个；tile 数 < CU 数且不是 4 的倍数时运行期退回原次序
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
    for tile in my_tiles():                            # 按上面的次序；运行时循环，tile 内 K 步成对展开，首尾两对剥出
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
        cdna3.buffer_store((acc * ws_r * xs_r).to(bf16), y[tile 行, tile 通道])   # 每 lane 每 16-token 块 16 B；
                                                       # panels ≥ 2 时 nt 写，不把 L2 里的 W 挤出去
    gl.barrier()
```
与代码顺序的差别：代码里 tile 内的 K 步按对展开（槽与 LDS 缓冲按奇偶静态确定），首尾两对剥出，中间成对的运行时循环
（K = 512 时 0 次）；计算相的解码 VALU 由 `sched_group_barrier` 均匀插进 MFMA 之间；store 那一相的后半按 token 块两两发
MFMA（同一累加器的两条隔 4 条），第一对期间解码下一 tile 的 W0(0)，之后每对期间写上一对，最后一对在末条 MFMA 后写。

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | G = min(tile 数, CU 数)，持久化 | CTA 依次做次序里属于它的 tile（原次序：c, c+G, …，N 块最快） |
| CTA（512 线程） | 8 wave = 2 组 × 4，每 SIMD 各一个 | y[128 token, 256 通道]；X tile 经 LDS 共享 |
| wave 组 | 组 0 = wave 0–3，组 1 = wave 4–7 | 各搬 X(j+1) 的一半行；相位错开一相 |
| wave | 32 通道 × 128 token | 2 个 MFMA 行 tile × 8 个 16-token 列 tile |
| lane | lane 组 g、行 r | 行 tile m 的行 r = 通道 32w+2r+m；出 8 个通道 |
| K 步 | 128 K = 两个 64-K 半步 | 每个计算相 64 条 MFMA；每步一个 LUT |

通道映射 32w+2r+m 让一个 lane 两个行 tile 的码在 fd2 里相邻：每半步一条 16 B 加载读满 8 条 128 B 行；输出时 lane 组 g
持有同一 token 的通道 8g…8g+7，每个 16-token 块一条 16 B 写。xq、xs、y 的 buffer 基址在每个 tile 移到首行（64 位），
≥ M 的行读 0、写丢弃；权重（< 2 GiB）用全局偏移。

tile 次序（CTA 按线性号轮流派到 4 个 XCD，各有 4 MiB L2）：原次序下同时运行的 CTA 共享少数几个 X 的 M 块，但 XCD x
要轮到 N 块 ≡ x（mod gcd(NB, 4)）的全部权重。`xcd_panels` 按 3 MiB 余量选：整份 W 或本次 X ≤ 3 MiB 时 1 面板（每个 XCD
拿整的 M 块，下一次 quant_x 重写 xq 时只碰一个 L2：tp4_o M 2048 端到端快约 15%）；否则 NB % 4 ≠ 0 时，原次序每 XCD 的
W 超 3 MiB 用 4 面板（tp2_qkv），W 加一轮 X 超 3 MiB 且 gcd(NB, 4) = 2 用 2 面板（tp4_qkv、tp2_o），每个 XCD 约留一个
面板的 W；其余（tp2_qkvz、tp4_qkvz 的大 M）原次序。CU 数不是 4 的倍数时一律原次序。

**流水线**（相位表与 tile 交界见 [gemm_large.py 模块 docstring](gemm_large.py#L35-L61)）
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

**命名**与**流水线不变量**：见 [gemm_large.py 模块 docstring](gemm_large.py#L63-L85)（Naming、Pipeline rules 1–7）。

**资源**：LDS 32 KB（2 × 128 × 128 B）；256 VGPR（无 AGPR）、58 SGPR、无 spill；`waves_per_eu = 2`，每 CU 1 个 CTA。

## gemm_a8w4_ptpc_decode_rb{rb}_ks{ks}_m{mcap}_g{G}（[代码](small_m.py#L304-L631)）
M ≤ 8 时的整个调用（最宽两形状的 M ≤ 2 / 4 换 dec2）：一次 launch 里把 x 按 token 量化进 LDS 并算完 y；块对映射让 ≤ 8 个 token 也不空转 MFMA 的半边。

**Gluon 风格**
```python
# grid = (G,)，G、rb、ks 按 (M, N) 选（small_m_config，目标形状 G = 64–128）；线程 = 64·rb·ks（8 或 16 wave）
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
| grid | G（按 N 选，目标形状 64–128） | CTA b：连续 ≤ rb 个 16 行块 × 全部 token × 全部 K |
| CTA（64·rb·ks 线程） | rb·ks = 8 或 16 个 wave | 量化全部 M 行进 LDS；本 CTA 的 delta 进 LDS |
| wave | 1 个 16 行块 × K/ks | 本 K 段的全部块对；量化 token 行 w, w+rb·ks, … |
| lane | 块对内第 l 个 16 B | 2 个通道 × 16 个码；A 为 token l%8 的同段 x（16 B） |
| 块对（K 步） | 128 K = 两个 64-K 块 | 每 wave 4 条 MFMA、2 个 LUT |

CTA 数不整除行块数时（如 tp2_qkv：416 个行块分给 80 个 CTA）各 CTA 管 5 或 6 块，多出的槽照算相邻块、不写。
y 的 buffer 记录数是 M 行：≥ M 的 token（含 MFMA 8 行里多出的）照算照写，写被硬件丢弃。

**流水线**（不流水；阶段表见 [small_m.py 模块 docstring](small_m.py#L39-L50)）
```text
序言：VMEM x 行（每 lane K/512 条 16 B × ⌈mcap/波数⌉ 行）→ delta（16 B，全 CTA 分摊）→ 权重（每块对 1 条 16 B）→ ws
量化：等 x → lane 内 max_u16 + 6 次 shuffle_xor 得 amax → scale → 倒数与两次 FMA（packed f32）→ fp8 → 8 B 写 LDS
delta 写 LDS → barrier → 全部块对的 x（16 B）与 delta（8 B）读进 VGPR
块对 s（展开 P/ks 个）：等 wr[s] → 2 个 LUT → 解码 4 个 dword → 4 条 MFMA
收尾：lane^40 相加 → K 段 1..ks−1 写部分和（每 lane 8 个 f32）→ barrier → K 段 0 按段号相加 → × ws × xs → bf16 → 4 B 写
```

**命名**与**流水线不变量**：见 [small_m.py 模块 docstring](small_m.py#L52-L63)（Naming、Rules 1–4）。

**资源**：LDS = mcap·(K+16) + rb·⌈K/256⌉·128 + 部分和 rb·(ks−1)·512 B（目标形状 6–34 KB）；68–124 VGPR、无 spill。
LDS 随 mcap·K 增长：分派按 `_decode_lds`（与 SharedStorage 逐段一致）检查，超过 64 KiB 时退到 tile 或通用路径，compile 报错。

## gemm_a8w4_ptpc_dec_rb{rb}_ks{ks}_x{xw}_m{mcap}_g{G}（[代码](small_m.py#L634-L1004)）
最宽两个形状（tp2_qkvz M ≤ 2、tp2_qkv M ≤ 4）的 decode：映射与运算同上，wave 分角色，权重 wave 不等 CTA barrier。

**Gluon 风格**
```python
# grid = (G,)，G = 80（每 CU 一个 CTA）；线程 = 64·(xw + rb·ks)：xw = mcap 个 X wave + rb·2 个权重 wave，≤ 16 wave
# CTA b 管 16 行块 [b·NB/G, (b+1)·NB/G)（至多 rb 个）× 全部 token（M 补到 mcap = 1、2、4）× 全部 K
# 放置：同 decode（x 量化进 LDS、delta 进 LDS、权重直读 VGPR、K 段部分和经 LDS）；LDS 计数器 R 代替第一次 barrier
def dec2(x, wq, wd, ws, y, M):
    if tid() == 0: R.store(0)                         # LDS 初值不定：先清零
    gl.barrier()
    if wave_id() < xw:                                # X wave：只量化 x、搬 delta
        xr[j] = cdna3.buffer_load(x[wave_id() + xw·j, 单元 l + 64u])   # 先发全部 VMEM；≥ M 的行读 0
        dr[i] = cdna3.buffer_load(wd[本 CTA 行块, 第 tid() + 64·xw·i 个 16 B])
        for j: if 行 < M: XS[行] = s = row_scale(…); X[行, 单元 l + 64u] = quant(xr[j], s)   # 同 decode
        D.store(dr)
        if lane_id() == 0: atomic_add(R, 1)           # release：本 wave 的 LDS 写对读到计数的 wave 可见
        gl.barrier()                                  # 与权重 wave 收尾的 barrier 配对
    else:                                             # 权重 wave：行块槽 (w − xw) % rb、K 段 (w − xw) // rb
        wr[s] = cdna3.buffer_load(wq[blk, 块对 s, 第 l 个 16 B])   # 本 K 段全部块对一次发完；越过的槽记录数为 0、不读
        wsr = cdna3.buffer_load(ws[16·blk + 2·(l%8) + h] for h in (0, 1))
        while R.load(acquire) < xw: s_sleep(1)        # 等全部 X wave 发布；不进 CTA barrier
        xa, dd = X.load(…), D.load(…)                 # 块对循环、lane^40 相加同 decode
        if kpart > 0 and valid and 有用 lane: Pt[slot] = v
        gl.barrier()                                  # 保护：部分和写完才读
        v = v + Pt[slot]
        if kpart == 0 and valid and 有用 lane: cdna3.buffer_store((v · wsr · XS[token]).to(bf16), y[token, 通道])
```

**任务划分**：同 decode，只是 grid 固定 80、rb = ⌈NB/80⌉（tp2_qkvz 7，tp2_qkv 6）、ks = 2，另加 xw 个 X wave；
X wave j 量化 token 行 j, j + xw, …（xw = mcap 时每 wave 一行），并分摊本 CTA 的 delta。

**流水线**（不流水，两种角色并行）：
```text
清零 R → barrier
X wave：VMEM x 行、delta → 量化 → 写 LDS（x、xs、delta）→ atomic_add(R) → barrier
权重 wave：VMEM 全部块对权重、ws → 自旋等 R = xw → LDS 读 x、delta → 块对 LUT / 解码 / MFMA
  → lane^40 相加 → K 段 1 写部分和 → barrier（与 X wave 的那次配对）→ K 段 0 相加 → × ws × xs → bf16 → 4 B 写
```
为什么分角色：加载按 wave 先后被接受，decode 核的 CTA barrier 要等排在最后的 wave 发完权重加载；dec2 的权重 wave 不进这次 barrier。

**资源**：LDS = decode 的各段 + 16 B 计数器（每段 16 B 对齐，`_decode_lds`）；超过 64 KiB 时分派退到 decode，compile 报错。

## gemm_a8w4_ptpc_tile_rb{rb}_ks{ks}_t{tq}_kp{kp}_g{G}（[代码](small_m.py#L1007-L1308)）
9 ≤ M ≤ 16（tp4_o ≤ 20，表外 ≤ 32）时的整个调用：与 decode 一样在 CTA 内量化，换标准映射（权重作 A、16-token 片作 B），K 分窗口让 32 行 fp8 装进 LDS。

**Gluon 风格**
```python
# grid = (G,)，G = ⌈NB/rb⌉（rb、ks 按 N 选，kp 取 LDS 放得下的最小值）；tp2_qkv 由入口改成 rb = 6、G = 80（416 个行块，
#   每 CTA 5 或 6 个），让 80 个 CU 都有 CTA；目标形状 G = 56–80；线程 = 64·rb·ks（8 或 16 wave）；每 CU 1 个 CTA
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
| grid | G = ⌈NB/rb⌉，tp2_qkv 为 80（目标形状 56–80） | CTA b：连续 ≤ rb 个 16 行块 × 全部 token × 全部 K |
| CTA（64·rb·ks 线程） | rb·ks = 8 或 16 个 wave | 量化 < M 的 token 行进 LDS；delta 进 LDS |
| wave | 1 个 16 行块 × 每个窗口的 1/ks | tq 块 y[16 token, 16 通道]；量化 token 行 w, w+rb·ks, … |
| lane | 通道 r = l%16、K 块 g = l//16 | 每步每半 8 B 码；每片 32 B x；每片出 4 通道 × 1 token |
| K 步 | 128 K | 每片 4 条 MFMA；1 个 LUT |
| K 窗口 | kp 个，整 512-K 单元 | 本窗口 16·tq 行 fp8 在 LDS（目标形状 kp = 1；表外如 N = 4160、M 17–32 时 2） |

lane 组 g 每步只读第 4s+g 个 32-K 组，所以每 lane 每步一个 delta、一个 LUT；部分和与 X 合计超过 64 KB 时
（目标形状的小 M 档不会，表外如 N = 4160 的 M 17–32）复用 X 区，前面多一次 barrier。

**流水线**（不流水；阶段表见 [small_m.py 模块 docstring](small_m.py#L39-L50)）
```text
序言：VMEM x 行（每 lane K/512 条 16 B × ⌈16tq/波数⌉ 行）→ delta → 各窗口权重（每步每半 1 条 8 B）→ ws
求每行 scale（整行，各窗口共用）→ lane 0 写 xs；delta 写 LDS
窗口 p：[p > 0：barrier] → 量化窗口 p 的单元 → 8 B 写 LDS → barrier
  步 s（展开 spw 个）：LDS 读 delta 4 B、每片 x 32 B → 建 LUT → 每半解码 2 个 dword → 每片 2 条 MFMA
收尾：[部分和复用 X：barrier] → K 段 1..ks−1 写部分和（每 lane 每片 16 B）→ barrier → 按段号相加 → × ws × xs → bf16 → 8 B 写
```

**命名**与**流水线不变量**：见 [small_m.py 模块 docstring](small_m.py#L52-L63)（Naming、Rules 1–4）。

**资源**：LDS = 16·tq·(最宽窗口 K + 16) + rb·⌈K/256⌉·128 + 部分和 rb·(ks−1)·tq·1 KB（可复用 X）+ xs；目标形状 32–64 KB，
每 CU 1 个 CTA；125–169 VGPR、无 spill。x 行与全部权重常驻 VGPR，寄存器随 K 增长：K = 8192 时 256 VGPR 且有 spill。
