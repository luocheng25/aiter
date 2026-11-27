# mega_moe_tp_gfx942：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。
     源码 docstring 已写全的规则（Rules）与命名（Naming）只链接过去，不在这里重复。 -->

本目录四个 kernel 各把单卡 op 的最后一步与其后的跨卡求和（all-reduce；`comm_mode="ag_rs"` 时为 reduce-scatter）合进同一个
kernel（down_tp 之后另有一个 gather kernel）：
- [decode_tp.py](decode_tp.py)：decode 档的 Down decode（单槽 / 槽对 / 混合）+ 跨卡求和；
- [sum_tp.py](sum_tp.py)：中段（TP4、输出小于 `FUSED_MIN_BYTES`）的 route_sum + 跨卡求和；
- [tp_reduce.py](tp_reduce.py)：融合档的 route_sum / sorted_sum + 跨卡求和（原子流式 Down 之后为拷贝模式）；
- [down_tp.py](down_tp.py)：融合档的 prefill / 流式 Down + 跨卡求和，其后接 gather（launch 名 `a8w4_tp_reduce`）。

计算部分就是 gfx9_fp8 的 kernel 主体（decode_tp 直接调用其 partials 函数；down_tp 用其带编译期 TP 钩子的 Down，钩子关闭时
单卡 kernel 不变），任务划分与流水线见其
[kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md)；host 分派见 [mega_moe_tp_gfx942.py](../../mega_moe_tp_gfx942.py) 与
[moe_a8w4.py](../../moe_a8w4.py#L2089) 的 `_TPHooks`。

## 先读：Gluon 风格的分阶段写法
每个 kernel 一节，先给一段 Gluon 风格的函数，一个 program = 一个 CTA。写法与计算部分的记号（M、H、TOPK、route、loc_ids、
routes、tree_sum、route_sum、wave_id、uniform、wait_vmem、ballot、atomic_add 等）沿用 gfx9_fp8
[kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md) 的「先读」，这里只补跨卡的部分。不用纯 Triton，是因为这几个 kernel 的
要点在块级张量写法里无处可写：partial 留在 VGPR 还是经 LDS、写到哪张卡的哪块缓冲、用哪种可见性（system scope）、谁等谁、
等多久、哪个 wave 负责交换、barrier 保护什么。所以每段先用注释写**放置**（含跨卡缓冲）与**预取**，再按序言 / 稳态 / 收尾写
各阶段，Gluon 没有的操作写成伪操作；只求看懂、不求能运行（张量下标代替地址运算，省略 mask 与类型转换）。
```python
# —— 卡、token 与 partial ——
# rank：本卡号；WORLD：TP 卡数（decode_tp / sum_tp：2、4；tp_reduce / down_tp：2、4、8）；peer：其余各卡。
#   各卡以同样的 M 做同样的调用：parity / epoch 的推进不靠 host 状态，CUDA graph 重放也照样推进。
# AR（comm_mode="ar"）：各卡输入同样的 M 个 token，各卡都要完整的 out bf16 [M, H]。
# DP（comm_mode="ag_rs"）：rank r 持 M_r 个 token，先 all_gather 成 M = WORLD·M_r 行；token t 的 home(t) = t // M_r。
#   每卡只要自己 token 的和：out bf16 [M_r, H]（decode_tp 例外：写 part [M, H] 里 home 的行，host 再取本卡的 M_r 行）。
# partial：本卡（中间维的 1/WORLD）对一个输出值的贡献，即单卡 kernel 本该写出的值；结果 = rank_sum(各卡的 partial)。
# X@s：rank s 上的对称缓冲 X（各卡同址同布局，地址查 tab）；X@rank 是本卡的。
# —— 划分单位 ——
# blk（decode_tp）：一个 token 的 16 个输出通道，NB = H/16，blk = token·NB + jj；交换 wave 的 lane l = 16s + c（rank s、通道 c）。
# unit（sum_tp、down_tp）/ 项（item，tp_reduce）：一个 token 的 512 列（down_tp：列块里的 512 列），CB = H/512，
#   sum_tp 的 u = t·CB + c；一个 wave 一个 unit / 项，一个 lane 一个 16 B 包（8 列）。
# tile（tp_reduce、down_tp）：TT 个相邻 token，tile i 从 token i·TT 起（末 tile 截到 M）。
# owner：收齐各卡 partial、求和、写出（AR 时再推给各 peer）的 rank。OWN_TILE（AR）：tile i 归 i % WORLD；
#   OWN_HOME（DP）：tile r + WORLD·i 是 rank r 的 token r·M_r + i·TT 起的 TT 个（截在 (r+1)·M_r，每卡 tpr = ⌈M_r/TT⌉ 个），
#   归 rank r（仍是 tile % WORLD）：owner 交错，同时在途的 tile 用上所有链路，不会都挤向一张卡。
#   sum_tp："ar2" 的 owner = u % WORLD，"dp" 的 owner = home(t)。decode_tp 无 owner（"ar" 各卡都求和，"dp" 只 home 求和）。
# 列块 q（down_tp）：列 [q·CWD, (q+1)·CWD)，CWD = H/S（S = NSPLIT ≤ NSPLIT_MAX = 16，CWD 为 512 的倍数）；
#   flag / epoch 槽 v = q·tpq + tile，tpq = WORLD·⌈⌈m_max/WORLD⌉/4⌉（最大批在最小 TT = 4 下的 tile 数），按 NSPLIT_MAX·tpq 布局。
# 角色（down_tp，同一个 1-D grid 里交错）：C(s) 计算 WG（列块 s 的 Down，写完 route 行后 ctr[s] += 1）；X(q) reduce WG
#   （等 ctr，按 route 求和成 partial 发给 owner，置 RS）；Y(q) owner WG（等 RS、rank_sum、写 out 与 landing、置 AG）；
#   gather：第二个 kernel（launch 名 a8w4_tp_reduce），等 AG 后把 landing 里别卡求和的行拷进 out。
# COPY（down_tp）：计算 WG 用 agent 范围 bf16 原子加把各 route 加进本卡 routes 的 token 行（不写 loc），partial 即该行；
#   TOPK = 0（tp_reduce 拷贝模式）：partial 即 src（原子流式 Down 的输出，可以就是 out）的行。
# AG = 0（只配 OWN_HOME）：owner 只写自己的 out、不写 landing；AG flag 照发（告诉 peer 它的 partial 已读走），gather 不拷贝。
# shared 列：折叠的 shared expert 算一路 route（loc 的末列、routes 里 routed 行之后），TOPK = loc 的列数（含它）。
# —— Down_tp 的划分与开关（DownTP 字段，编译期）——
# task_pad：每列块的 C 数，task_num（流式 task_num·GPB）向上取整到 XCD 整轮；GPB：流式每列块的列组数 = CWD/512。
# UPW：一个 tile 每 wave 的 unit 数；UX：X 一步的 unit 数（0 即 UPW），STEPS = UPW/UX。
# NX / NY（末块 NXT / NYT）：每列块的 X / Y 数；AX / AY：X / Y 段插在下一列块 C 的 AX/8、AY/8 处；
#   GX / GY：X / Y 一组的 tile 数（一组一次 wait_vmem + barrier + flag）。
# out0：AG = 0 时本卡 out 第 0 行对应的 token（rank·M_r），否则 0。
# ROUTE_AUX：C 写 routes（prefill 另有 loc）的 cache 位，默认 SYS（写穿）；SYS_NT = SYS | 2，另加 nt。
# PRO / SB / LAS / WPE：prefill C 的调度开关（结果逐位不变）；DRY / TRACE：只用于计时分析（见 DownTP docstring）。
# —— 跨卡缓冲（对称、不缓存）——
# inbox：收 peer 的 partial。decode_tp int32 [2][WORLD][MAX_M·NB][16]；sum_tp int32 [2][WORLD][UNITS][256]；
#   tp_reduce bf16 [(WORLD−1)·rps, H]，区 (rank − owner − 1) % WORLD；down_tp bf16 [WORLD·rps, H]，owner 自己的在区 WORLD−1。
#   rps：一个源区的行数，OWN_TILE ⌈tiles/WORLD⌉·TT、OWN_HOME tpr·TT；tp_reduce 与 down_tp 共用一块 inbox（取大者）。
# landing：owner 推回的和。sum_tp（"ar2"）int32 [2][UNITS][256]；tp_reduce / down_tp bf16 [M_max, H]。
# flags（tp_reduce / down_tp）int32：RS[i][s]（rank s 的 partial 已到 owner）在 i·WORLD + s；AG[i]（和已推出）在 slots·WORLD + i；
#   flags@q.RS[i][s] 即 rank q 上的。down_tp 用自己的 flags2 / epoch2，以槽 v 代替 tile i。
# tab int64（按 int32 读）：各卡缓冲的地址。decode_tp [WORLD]（inbox）；sum_tp [2, WORLD]（inbox、landing）；
#   tp_reduce / down_tp [3, WORLD]（landing、inbox、flags）。
# 哨兵：decode_tp 的字 0xFFFFFFFF、sum_tp 的半字 0xFFFF，都是 kernel 不会写出的 NaN：数据就是 flag，不另设 flag、不要 fence。
# —— 本卡状态（普通显存，建立时与 reset() 时初始化）——
# parity[blk / u]（decode_tp / sum_tp）int32：本次用的 buffer b ∈ {0, 1}；每卡用完都翻 b ^= 1，复位与再写错开一次调用。
# epoch[i / v]（tp_reduce / down_tp）int32：本次 epoch = epoch + 1，flag 里存它（比低 30 位）；POISON = 1 << 30：owner 等 RS 超时。
# ctr[q]（down_tp）int32：列块 q 已完成的计算 WG 数（agent 范围原子加），= task_pad 即完成；gather 的 WG 0 清零。
# err int32 [1]：等待超时置位，tp_reduce / down_tp 位 1、decode_tp 位 2、sum_tp 位 4；limit：等待上限，单位
#   LIMIT_TICKS = 1024 个 100 MHz 时钟；发现 err 已置位的等待只读一次就放弃。
# —— 伪操作（计算部分的见 gfx9_fp8 先读）——
# s_load(t[i])：标量（SMEM）读，wave 内统一；keep(v…)：空的有副作用 asm，让这批标量读在此一起发出（否则各沉到首次使用处）
# sys_load(X@s[i]) / sys_store(X@s[i], v)：system scope 读写，写穿到目标卡显存、读绕过缓存；buffer 指令用
#   aux = SYS = sc0|sc1 = 17，decode_tp 用 one-as monotonic 的 global 读写。跨卡数据与同一 launch 内跨 XCD 的数据都用它
# shuffle(v, src)：取本 wave lane src 的 v（ds_bpermute）；readlane(v, k)：取 lane k 的 v 成标量
# ballot_any(p)：本 wave 有 lane 的 p 为真（ballot ≠ 0），wave 内统一，可直接做循环条件
# late, v = wait_until(load, done, limit)：反复 v = load() 直到 done(v)，每轮 s_sleep 2；memrealtime（100 MHz）计时，
#   满 limit·LIMIT_TICKS 就放弃（late = 真）；简写 wait_until(读什么, 到齐=条件)：同上，limit 省略
# atomic_or(err, bit)：global 原子 OR；wait_vmem()：s_waitcnt vmcnt(0)；gl.barrier()：s_barrier；tid：CTA 内线程号
# _DROP：越界的 buffer 偏移 0x7FFFFF00，读到 0、写被丢弃：让部分 lane「不读 / 不写」而不分支
# —— 语义 ——
# rank_sum(p_0 … p_{WORLD−1})：各卡 partial 转 f32，按 rank 0 … WORLD−1 的顺序相加，只舍入一次到 bf16（RTE）
# route_sum：partial = Σ_k routes[loc[t, k]]，f32 按槽顺序（loc < 0 加 +0），舍入一次（同单卡 a8w4_route_sum / sorted_sum）
# canonical(v)：恰为哨兵的值改成另一个 NaN（字 0x7FFFFFFF、半字 0x7FFF）：发出去的数据不会被当成「未到」
# —— 各路径的条件与 kernel 调用顺序 ——
# host：forward（AR）/ _forward_ag_rs（DP）经 moe_a8w4._TPHooks 替换 op 的最后一步；op 的分档见 gfx9_fp8 先读。
# 对称缓冲 ws 只在 WORLD ∈ FUSED_WORLDS = (2, 4) 时建，否则都不融合。
# decode 档上限：DECODE_TIER_MAX_M[(WORLD, H)]（只有 (4, 2560)，为 10），其余用 op 自己的（H5120 为 10，其余 32）。
# AR：
#   decode（M ≤ DECODE_TP_MAX_M = 32 且在 decode 档）：op 的 decode 前段 → decode_tp：
#     M < 3 单槽 a8w4_down_decode_tp；M ≥ DECODE_TP_PREQUANT_MIN_M = 3 且槽对放得下时槽对 a8w4_down_decode_pq_tp
#     （op 给的是单槽输入时层先 row_quant_decode(mid)）；折叠 shared（op 只在槽对时折叠）为 …_pq_shared_tp；
#     混合 decode 为 a8w4_down_decode_mixed_tp。按 expert 去重的 Down 不走 hook，其后 all_reduce
#   融合（_fuses：M ≥ FUSED_MIN_M = 33 且 2·M·H ≥ FUSED_MIN_BYTES = {2: 2 MiB, 4: 1 MiB}）：
#     prefill Down（含流式档里的 BM64 prefill Down）且 (WORLD, H) ∈ STREAM_SPEC：down_tp → gather
#     流式 Down 且 STREAM_TIER_SPEC 对该 M 有值：down_tp（流式；原子 Down 时 COPY）→ gather
#     其余：op 的 Down → tp_reduce（a8w4_tp_reduce；原子流式 Down 之后为拷贝模式）
#   中段（未融合，_mid_fuses：WORLD ∈ MID_WORLDS = (4,)、H % 512 = 0、M ≤ ws.mid_m（输出 < FUSED_MIN_BYTES）、
#     M·CB ≥ WORLD；op 走流式 / prefill 档）：
#     非原子 Down → sum_tp "ar2"（a8w4_route_sum_tp）；原子流式 Down 之后不接 kernel → all_reduce
#   其余：op 原样 → all_reduce（custom all-reduce；输出 ≥ AR_RCCL_MIN_BYTES 时 RCCL）
# DP：all_gather(x, ids, w) → 同一个 op（M = WORLD·M_r，条件同上），融合 kernel 换成只写 home 的版本（名字加 _rs）：
#   decode：decode_tp "dp"（…_tp_rs）；中段：sum_tp "dp"（a8w4_route_sum_tp_rs；原子流式 Down 之后同「其余」）
#   融合：prefill Down 且 STREAM_SPEC：down_tp（OWN_HOME、AG = 0，…_tp_rs）→ gather（只等 AG）；
#     流式 Down 不融合（STREAM_TIER_SPEC_DP 为空）→ tp_reduce（OWN_HOME、AG = 0，a8w4_tp_reduce_rs）
#   其余：reduce_scatter_ordered（custom reduce-scatter，或 all_to_all 后本卡按 rank 序 f32 相加）
```

## A8W4 Down decode 跨卡版 decode_tp（[decode_tp.py](decode_tp.py#L302)）
decode 档的 Down decode：主体与单卡 kernel 相同（gfx9_fp8 [kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md) 的「A8W4 Down
decode（核内量化）」「槽对」「混合 Down decode」三节），只把最后一步「W 个 wave 的部分和经 LDS 相加后写 bf16 out」换成：
一个 wave 与各卡交换 16 通道块的 f32 partial，按 rank 序相加后写出。一次 launch 替换 Down 与其后的 all-reduce（"dp"：
reduce-scatter）。变体：单槽 `a8w4_down_decode_tp`、槽对 `a8w4_down_decode_pq_tp`、槽对加折叠 shared
`a8w4_down_decode_pq_shared_tp`（[compile_down_decode_tp](decode_tp.py#L302)）与混合 `a8w4_down_decode_mixed_tp`
（[compile_down_decode_mixed_tp](decode_tp.py#L484)），共用交换 [_exchange](decode_tp.py#L186)；host 侧见
[_decode_down](../../mega_moe_tp_gfx942.py#L1174)。

**Gluon 风格**
```python
# grid = (G, M)，64·W 线程。单槽 / 混合：G = H/32，WG 管 32 通道 = 块 2j、2j+1；槽对：G = H/16，WG 管块 j
#   W：单槽 TOPK（TOPK > 16 时 4）；槽对 min(⌈TOPK/2⌉, 3)；混合 TOPK（末列 shared expert 也占一个 wave）
# 放置：主体同单卡：mq / LUT 在 LDS，权重直进 VGPR；序言先发的 64-K 块：单槽 3、槽对 min(6, 全部)、混合全部（K/64）；
#       各 wave 的部分和写 LDS part。
#       交换不用 LDS：本卡 partial 在交换 wave 的 VGPR；peer 的 partial 由 peer sys_store 进本卡 inbox，本 wave 轮询读回。
#       交换的标量状态（parity[blk]、err、tab）在 barrier 之后与 part 的 LDS 读一批发出
@gluon.jit
def a8w4_down_decode_tp(mid, mid_scale, topk_ids, topk_weight, w2, w2_delta, w2_scale, out, M,
                        tab, parity, err, rank, limit):   # DEST、WORLD、MAX_M = 32、变体都是编译期常量
    j, token = gl.program_id(0), gl.program_id(1)
    # 序言 + 稳态：单卡 kernel 到「part 写进 LDS」为止（SHARED：序言另读 out[token, 16j + c]，本 rank 的 shared 输出）
    part = down_decode_partials(…)    # 槽对：down_decode_pq_partials；混合：down_decode_mixed_partials
    gl.barrier()                      # 保护：W 个 wave 的 part 都写完，交换 wave 才读（每个通道要读全部 W 行）
    # 收尾，wave 分工：单槽 / 混合 W ≥ 2 时 wave 0、1 各换一块（并行），W = 1 时 wave 0 依次换两块；
    #   槽对只 wave 0 换块 j。其余 wave 到此结束
    for jj in my_blocks(wave_id()):
        exchange(jj, token, part)

def exchange(jj, token, part):        # 一个 wave；lane l = 16s + c：rank s 的通道 c（s ≥ WORLD 的 lane 闲置）
    blk = token * NB + jj
    b, e, base = s_load(parity[blk]), s_load(err), s_load(tab[:])     # 先发标量读……
    y = tree_sum(part[:, 本块的通道 c])  # ……再发 part 的 LDS 读；keep(b, e, base)：两批一起在途，只等一次
    #   槽对：每个 wave 两列相加再 tree_sum，SHARED 再 + shared 值；y 即单卡写 out 前的 f32 值
    peer = s < WORLD and s != rank
    if DEST == "ar":                  # 各卡都收、都写
        send, recv, writes = peer, peer, l < 16
    else:                             # "dp"：只有 token 的 home 收、写；其余卡只发给 home
        h = home(token)
        send, recv, writes = peer and s == h, peer and rank == h, l < 16 and rank == h
    if send: sys_store(inbox@s[b][rank][blk][c], canonical(y))   # 每个收方 peer 一次 64 B 写穿；先发后等
    own = inbox@rank[b][s][blk][c]
    late, v = wait_until(lambda: sys_load(own), lambda v: not ballot_any(recv and v == SENTINEL),
                         limit if e == 0 else 0)
    if recv: sys_store(own, SENTINEL) # 复位：两次调用之后 buffer b 再用
    total = rank_sum([y if r == rank else shuffle(v, 16 * r + c) for r in range(WORLD)])
    if writes: out[token, 16 * jj + c] = total.to(bf16)
    if l == 0: parity[blk] = b ^ 1    # 每个 rank 都翻（"dp" 的非 home 也翻），普通 store
    if late and l == 0: atomic_or(err, 2)
```

**任务划分**

| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | (H/32, M)；槽对 (H/16, M) | 一个 token 的 32（16）通道 |
| CTA | 64·W 线程 | 1–2 个 16 通道块 |
| wave（主体） | W 个 | 同单卡：槽 / 槽对 / 列 |
| wave（交换） | wave 0、1；槽对 wave 0 | 一个块（W = 1：两块） |
| lane（交换） | l = 16s + c，s < WORLD | rank s 的通道 c |
| K 步 | I/64 个 64-K 块 | 同单卡 |

**流水线**
```text
主体（同单卡，见 gfx9_fp8 三节的逐拍流水线）
  序言：expert / 路由权重 → mid 行（单槽、混合）或 mq 片（槽对）→ 权重块（单槽前 3 块、槽对前 min(6, 全部) 块、
        混合全部 K/64 块）与通道因子；SHARED 另读 out[token, 16j + c]
  稳态：每个 64-K 块一组 MFMA，单槽 / 槽对每消耗一块补发一块；收尾乘通道因子 · 行 scale · 路由权重 → part 写 LDS
  barrier（单槽 L401、槽对 L371、混合 L532）
交换（一个 wave，_exchange L186）
  1. s_load：parity[blk]、err 各一条，tab 每 4 字一条（2·WORLD 字）——先发
  2. part 的 LDS 读：单槽 / 混合 W 个字，槽对 2W 个字（列 col、col + 8）——后发；keep（L210）让两批一起在途
  3. y = tree_sum（槽对再 + bf16 的 shared 值）；bits = y 的位，0xFFFFFFFF 改 0x7FFFFFFF（L233）
  4. send：收方 peer s 的 lane 16s + c 把 bits 写进 inbox@s[b][rank][blk][c]（system scope monotonic store）
  5. wait：lane 读 inbox@rank[b][s][blk][c]（system scope load）；ballot 看收方 lane 有无哨兵，有则 s_sleep 2 再读；
     memrealtime 计时，满 limit·1024 tick 放弃（err ≠ 0 时 limit = 0：读一次）
  6. recv 的 lane 把读到的字写回 0xFFFFFFFF（system scope，L248）
  7. total：WORLD 次 shuffle 取 lane 16s + c 的字（本 rank 用 y），f32 按 rank 序相加
  8. writes 的 16 个 lane 写 out[token, 16jj + c] = bf16(total)；lane 0 普通 store parity[blk] = b ^ 1；
     超时则 lane 0 atomic_or(err, 2)（L267–269）
W = 1（单槽 TOPK = 1）：wave 0 对块 2j、2j+1 依次做 1–8；W ≥ 2：wave 0、1 各做一块，同时进行
```
行号：[_exchange_state](decode_tp.py#L116)、[_keep](decode_tp.py#L125)、[_spin_value](decode_tp.py#L146)、
[_exchange](decode_tp.py#L186)、[_exchange_pair](decode_tp.py#L272)、单槽 / 槽对 barrier [L401](decode_tp.py#L401) /
[L371](decode_tp.py#L371)、混合 [L532](decode_tp.py#L532)；主体 [down_decode_partials](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L646) /
[down_decode_pq_partials](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L896) /
[down_decode_mixed_partials](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L1805)，求和
[down_decode_sum](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L793) /
[down_decode_pq_sum](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L1100) /
[down_decode_mixed_sum](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L1989)。

**命名**
- kernel 名 [kernel_name](decode_tp.py#L105)：base + `_tp`，DEST "dp" 再加 `_rs`；base 为 `a8w4_down_decode`、
  `a8w4_down_decode_pq`、`a8w4_down_decode_pq_shared`、`a8w4_down_decode_mixed`。
- launch 参数：单卡 Down decode 的参数（到 `out`、`M`）之后接 `tab, parity, err, rank, limit`，含义见 docstring 的 Naming
  （[L50](decode_tp.py#L50)）；host 传 `ws.dtab`、`ws.parity` 与层的 `_err`。inbox 字数 [inbox_words](decode_tp.py#L100)。
- 代码里：`jj` 块在 token 内的序号，`sub` 交换 wave 号，`buf` = b，`own` 本卡 inbox 的字地址，`seen` 读到的字，
  `send` / `recv` / `writes` 三个 lane 谓词，`slot_rank` 闲置 lane 用的 rank（取本卡）。

**流水线不变量**（docstring Rules 1–4 见 [L53](decode_tp.py#L53)：无死锁、两个 buffer、等待有界、逐位一致；下面补实现层面的）
1. part 在 LDS：单槽 / 混合 part[w·32 + ch]，ch = 16·(n//8) + 2·(n%8) + h（lane n < 16 写两个，
   [_pair_channel](../moe_gemm_2stage/gfx9_fp8/decode_a8w4.py#L626)）；槽对 part[(2w + h)·16 + n%16]（lane 0–7、40–47 写），
   通道 c 取列 (c&1)·16 + c//2 与它 + 8。交换 wave 读全部 W 个 wave 的一列，所以 barrier 在交换之前，之后不再写 part。
2. 交换 wave：单槽 / 混合 W ≥ 2 时 wave 0 换块 2j、wave 1 换块 2j+1，W = 1 时 wave 0 先 2j 后 2j+1；槽对只 wave 0。
   各卡同一块由同号 WG、同号 wave 交换，W = 1 时块的先后也相同（规则 1 的前提）。
3. 先发后等：send 在 wait 之前，wait 只等本块的 peer（规则 1）。
4. 数据即 flag：发出的位先 canonical；recv 的 lane 读到后马上写回哨兵，所以两次调用之后 buffer b 再用时 inbox 全是哨兵。
5. parity[blk] 每次调用由每个 rank 的 lane 0 翻一次（"dp" 的非 home 也翻，不论收不收）。
6. "dp"：home 不发，非 home 不收、不写；home 写 out（host 传的 part [M, H]）的 token 行，其余行不动。要求每卡每次都收
   （M ≥ WORLD，规则 2）：DP 的 M = WORLD·M_r 自然满足。
7. 超时：lane 0 atomic_or(err, 2) 后照常求和、写出（结果错、不挂起）；之后的调用要 `reset()`（规则 3）。
8. 本卡那份 partial 用寄存器里的 y（canonical 之前），peer 收到 canonical(y)：只有 y 的位恰为 0xFFFFFFFF（NaN）时两者不同。

**资源**
- LDS：同单卡版（mq + part f32 [W, 32] + LUT，见 gfx9_fp8 三节）；交换不占 LDS。
- 跨卡缓冲：inbox int32 2·WORLD·32·NB·16 个 = 2·WORLD·32·H·4 B（H2560：TP2 1.25 MiB、TP4 2.5 MiB；H5120 TP4 5 MiB），
  建立时与 `reset()` 时填 0xFF；parity int32 [32·NB]（H2560 为 20 KiB）。
- SGPR：交换多 2 + 2·WORLD 个标量字；VGPR / 占用：未测。

## 路由求和 + 跨卡求和 sum_tp（[sum_tp.py](sum_tp.py#L176)）
中段（TP4，输出小于 `FUSED_MIN_BYTES`，非原子流式 / prefill Down 之后）一次 launch 做 route_sum 与跨卡求和：每个 wave 先把本卡
一个 unit 的 route 行求和成 partial，再按 custom all-reduce 的协议交换（只推送、数据即 flag、无 fence、无 barrier）。
DEST "ar2"（AR）：two-shot，unit 的 owner `u % WORLD` 收齐、rank_sum、写自己的 out 并推到各 peer 的 landing，peer 在自己的
landing 等到后拷进 out。"dp"（DP）：owner = home(t)，只写 home 的 out [M_r, H]，其余卡只发。host 侧见
[_mid_reducer](../../mega_moe_tp_gfx942.py#L1134)。

**Gluon 风格**
```python
# grid = ⌈M·CB/4⌉，256 线程 = 4 wave；wave 管 unit u = 4·pid + wave_id()（u ≥ M·CB 的 wave 直接结束），lane 管 8 列
# 放置：loc 一个 lane 一列（lane < TOPK）读进 VGPR，readlane 成 TOPK 个标量；route 行 global → VGPR，TOPK 条 16 B 读
#       同时在途（每行一个 64 位基址的描述符）；partial 与和都在 VGPR。peer 的 partial 由 peer sys_store 进本卡 inbox，
#       owner 的和进本卡 landing，本 wave 轮询读回；不用 LDS、不用 barrier；状态的标量读与 route 读一起发出
@gluon.jit
def a8w4_route_sum_tp(loc, routes, out, tab, parity, err, M, rank, limit):   # DEST、WORLD、H、TOPK、UNITS 编译期
    u, l = uniform(4 * gl.program_id(0) + wave_id()), lane_id()
    if u >= M * CB: return
    t, cols = u // CB, 512 * (u % CB) + 8 * l + gl.arange(0, 8)
    b, e, base = s_load(parity[u]), s_load(err), s_load(tab[:])   # tab："ar2" inbox + landing，"dp" 只 inbox
    lk = [readlane(loc[t, l] if l < TOPK else 0, k) for k in range(TOPK)]   # 一条 4 B 读 + TOPK 次 readlane
    rows = [cdna3.buffer_load(routes[lk[k], cols]) if lk[k] >= 0 else 0 for k in range(TOPK)]   # 全部在途
    keep(b, e, base)                  # 标量读钉在 route 读之下，一起等
    sent = canonical(route_sum(rows)) # 发出的、也是本卡求和用的 partial
    lim = limit if e == 0 else 0
    owner = u % WORLD if DEST == "ar2" else home(t)
    if rank == owner:                 # missing(p)：p 的 4 个字里有半字 = 0xFFFF
        late, p = wait_until(lambda: [sys_load(inbox@rank[b][s][u][l]) for s != rank],
                             lambda p: not ballot_any(missing(p)), lim)
        for s != rank: sys_store(inbox@rank[b][s][u][l], 0xFFFFFFFF ×4)     # 复位
        if late and l == 0: atomic_or(err, 4)
        total = rank_sum([sent if s == rank else p[s] for s in range(WORLD)])
        if DEST == "ar2":
            total = canonical(total)
            out[t, cols] = total      # 普通 store
            for q != rank: sys_store(landing@q[b][u][l], total)
        else:
            out[t - rank * M_r, cols] = total   # home 的 out [M_r, H]
    else:
        sys_store(inbox@owner[b][rank][u][l], sent)   # 64 lane × 16 B = 1 KiB 写穿；"dp" 到此为止
        if DEST == "ar2":
            late, v = wait_until(lambda: sys_load(landing@rank[b][u][l]), lambda v: not ballot_any(missing(v)), lim)
            sys_store(landing@rank[b][u][l], 0xFFFFFFFF ×4)                  # 复位
            out[t, cols] = v
            if late and l == 0: atomic_or(err, 4)
    if l == 0: parity[u] = b ^ 1      # 每个 rank 都翻，普通 store
```

**任务划分**

| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | ⌈M·CB/4⌉ 个 WG（1-D） | 4 个相邻 unit |
| CTA | 256 线程 = 4 个 wave | 各 wave 互不通信 |
| wave | 1 个 unit | token t 的 512 列 |
| lane | 1 个 16 B 包 | 8 列（4 个字） |
| K 步 | 无；TOPK 条 route 行 | 一次全部在途 |

**流水线**
```text
一个 wave 一个 unit（body L194）
1. s_load：parity[u]、err 各一条，tab 每 4 字一条（"ar2" 4·WORLD 字，"dp" 2·WORLD 字）
2. lane < TOPK 一条 4 B buffer 读 loc[t, l]（其余 lane _DROP）；等它，TOPK 次 readlane 得各槽的 loc
3. TOPK 条 16 B buffer 读 route 行（每行一个描述符，基址 64 位；loc < 0 的槽 _DROP 读 0），全部在途；
   keep（L228）让 1 的标量读与它们一起等
4. f32 按槽相加、舍入一次（_sum_bf16），canonical → sent（4 个字）
5. 非 owner：1 条 16 B system scope 写到 inbox@owner[b][rank][u]（"dp" 的非 owner 接着做 9）
6. owner 等：每轮 WORLD 条 16 B system scope 读本卡 inbox[b][s][u]（本卡那条 _DROP），逐半字查 0xFFFF，ballot；
   未齐则 s_sleep 2 再读；memrealtime 计时
7. owner：WORLD 条 system scope 写把这些包复位成全 1（本卡那条 _DROP）；rank_sum，本卡那份用 sent
8. "ar2" owner：canonical(total) → 普通 store out[t, cols]，再 WORLD 条 system scope 写到各 peer 的 landing（本卡 _DROP）；
   "dp" owner：普通 store out[t − rank·M_r, cols]；
   "ar2" peer：在 landing@rank[b][u] 轮询（同 6，每轮 1 条 16 B 读），复位，普通 store out[t, cols]
9. 超时的 wave 由 lane 0 atomic_or(err, 4)；lane 0 普通 store parity[u] = b ^ 1（其余 lane 的偏移 _DROP，L336）
```
行号：[body](sum_tp.py#L194)、[receive](sum_tp.py#L250)、"ar2" [owner](sum_tp.py#L284) / [as_owner](sum_tp.py#L290) /
[as_peer](sum_tp.py#L301)、"dp" [owner](sum_tp.py#L318)、[_spin_words](sum_tp.py#L142)、[_sum_bf16](tp_reduce.py#L198)、
[_home](decode_tp.py#L177)。

**命名**
- kernel 名 [kernel_name](sum_tp.py#L113)：`a8w4_route_sum_tp`，"dp" 加 `_rs`。
- launch 参数 `(loc, src, out, tab, parity, err, M, rank, limit, nwg)`：`src` = routes（按排序行的 bf16），`loc` = loc_ids；
  `tab`、`parity`、`err`、`limit` 见 docstring 的 Naming（[L43](sum_tp.py#L43)）；host 传 `ws.mtab`、`ws.mparity`。
- 缓冲大小：[inbox_words](sum_tp.py#L103) = 2·WORLD·UNITS·256，[landing_words](sum_tp.py#L108) = 2·UNITS·256；
  UNITS = `ws.munits` = mid_m·CB。
- 代码里：`pack` 本 lane 的包在 unit 缓冲里的字节偏移，`col2` 本 lane 在 out / routes 行内的字节偏移，`slot(s)` 包
  [b][s][u] 在 inbox 里的偏移，`at` 包 [b][u] 在 landing 里的偏移，`sent` 发出的 partial。

**流水线不变量**（docstring Rules 1–4 见 [L47](sum_tp.py#L47)：无死锁、两个 buffer、等待有界、逐位一致；下面补实现层面的）
1. unit 之间完全独立：一个 wave 一个 unit，不用 LDS、不用 barrier，WG 只是 4 个 wave 的打包；u ≥ M·CB 的 wave 什么也不做
   （也不翻 parity）。
2. 先发后等：非 owner 先发 partial 再等 landing；"ar2" 的 owner 收齐后才推 landing（规则 1）。
3. 一个 16 B 包可能分几次落地，所以逐半字查哨兵（[_missing](sum_tp.py#L118)）；发出的 partial 与 "ar2" 推到 landing 的和
   都先 canonical（[_canonical](sum_tp.py#L128)）；"dp" 的和只写 out，不 canonical。
4. 本卡在求和里用 sent（canonical 之后），与 peer 收到的位相同；"ar2" 只有 owner 求和，各卡存同一份位（规则 4）。
5. parity[u] 每次调用由每个 rank 的 lane 0 翻一次；每卡每次都要收到每个 peer 的数据，两次调用后再写同一 buffer 时对方才已
   复位（规则 2）："ar2" 要 M·CB ≥ WORLD（[_mid_fuses](../../mega_moe_tp_gfx942.py#L1123) 检查），"dp" 要 M ≥ WORLD（DP 自然满足）。
6. 超时：等到超时的 wave 由 lane 0 atomic_or(err, 4)，照常写出（结果错、不挂起）；之后要 `reset()`（规则 3）。
   "ar2" 的 owner 超时后仍把缺一份（哨兵 NaN）的和推到各卡 landing：晚到的卡在这些 unit 上不必等、读到即用，结果错而
   本卡 err 不置位（报错靠其他卡的 err；decode_tp / tp_reduce / down_tp 晚到的卡要么结果对、要么本卡置位）。

**资源**
- LDS：0；barrier：0。VGPR / 占用：未测。
- 跨卡缓冲：inbox 2·WORLD·UNITS KiB，landing 2·UNITS KiB（UNITS = mid_m·CB；m_max 够大时 TP4 H2560 mid_m = 204、
  H5120 = 102，UNITS 都是 1020：inbox 约 8 MiB、landing 约 2 MiB），建立时与 `reset()` 时填 0xFF；parity int32 [UNITS]。

## 跨卡求和 a8w4_tp_reduce（[tp_reduce.py](tp_reduce.py#L208)）
融合批里不走 Down_tp 的那些用它代替本卡的末步规约（`a8w4_route_sum` / `sorted_sum`；原子流式 Down 之后没有末步）与
all-reduce：token tile 归 owner，非 owner 把 tile 的部分和写进 owner 的 inbox（RS），owner 按 rank 序求和后写本卡 out
与各同伴的 landing（AG），各卡再把 landing 拷进 out；各卡结果逐位相同。部分和两种来源：SUM（`TOPK` > 0）按 loc 把
routes 按槽相加（与 route_sum / sorted_sum 同序同舍入），COPY（`TOPK` = 0）直接读原子 Down 写好的行。层里由 `reduce`
钩子 [`_reducer`](../../mega_moe_tp_gfx942.py#L1461) 发起，用在 Down_tp 没接管的融合档：tp4 M205–384
`a8w4_down_stream_bm16>a8w4_tp_reduce`（原子流式 Down 之后，COPY）；ds M103–512 `a8w4_down_stream_bm16>shared_join>a8w4_tp_reduce`
（SUM，`TOPK` = topk + 1；`shared_join` 不是 kernel，折叠的 shared expert 此前已写好 routes 行与 loc 末列）；DP M_r 26–384
的末个 kernel `a8w4_tp_reduce_rs`（`OWNER=OWN_HOME, AG=0`）。命名与规则 1–6 见 [docstring](tp_reduce.py#L4)，这里不重复。

**Gluon 风格**
```python
# grid = nwg 个 CTA（host 取 tiles：每 CTA 一个 tile；kernel 也支持轮取），256 线程（4 wave）；不用 LDS
# 放置：全部 global → VGPR 直读直写。部分和：一项的 TOPK 行（每行 16 B，各用一个 64 位基址的描述符：routes 可超
#       2 GiB）同时在途，项之间 sched_barrier，不跨项 / 跨 tile 预取；一个 tile 的 IPW 份部分和驻留 VGPR 直到推给
#       owner 或参与求和。inbox / landing / 标志 / err 是 sys_load / sys_store；out、epoch、loc、routes 普通读写。
#       写出：out 直写；AG 时同一份和再 sys_store 到每个同伴的 landing
@gluon.jit
def a8w4_tp_reduce(loc, src, out, tab, epoch, err, M, rank, slots, limit, nwg):
    for i in range(gl.program_id(0), tiles, nwg):    # 第一遍：部分和 → RS → 求和 → AG
        owner, t0, end = place(i)                    # OWN_TILE：i % WORLD、i·TT、M；OWN_HOME：见 docstring Owners
        ep, base = epoch[i] + 1, (i // WORLD) * TT - t0  # inbox 行 = 区·rps + base + t
        items = [(t, cols) for 本 wave 的项 idx = wave_id() + 4j]   # t = t0 + idx // CH，cols = 512 列块 idx % CH
        part = [(Σ_k routes[loc[t, k], cols]).to(bf16) if TOPK else src[t, cols] for t, cols in items]
        #       ↑ lane k < TOPK 读 loc[t, k]，readlane 逐槽广播；f32 按槽相加（loc < 0 加 +0），舍入一次
        if owner == rank:
            late, _ = wait_until(sys_load(flags@rank.RS[i][s]) for s != rank, 到齐=全等于 ep)   # wave 0 轮询
            gl.barrier()                             # 获取：见到 RS 后全 CTA 才读 inbox
            for (t, cols), p in zip(items, part):
                got = [sys_load(inbox@rank[(s - rank - 1) % WORLD, base + t, cols])   # s = rank 一项丢弃
                       for s in range(WORLD)]
                total = sum(p if s == rank else got[s] for s in range(WORLD)).to(bf16)   # f32，rank 序，舍入一次
                out[t if AG else t - rank * M_r, cols] = total
                if AG: for q != rank: sys_store(landing@q[t, cols], total)
            wait_vmem(); gl.barrier()         # 发布：全部 wave 写进各卡 landing 的数据都已完成，才发 AG
            if tid == 0:
                for q != rank: sys_store(flags@q.AG[i], (ep & 低 30 位) | (POISON if late else 0))
                epoch[i] = ep
        else:
            for (t, cols), p in zip(items, part):
                sys_store(inbox@owner[(rank - owner - 1) % WORLD, base + t, cols], p)
            wait_vmem(); gl.barrier()         # 发布：全部 wave 写进 owner inbox 的数据都已完成，才发 RS
            if tid == 0: sys_store(flags@owner.RS[i][rank], ep)   # 第一遍不写 epoch
    for i in range(gl.program_id(0), tiles, nwg):    # 第二遍：别的卡 own 的 tile，等 AG 后拷贝
        if i % WORLD == rank: continue
        ep = epoch[i] + 1                            # 与第一遍同一个 ep
        late, ag = wait_until(sys_load(flags@rank.AG[i]), 到齐=低 30 位等于 ep)   # ag 带 POISON：err |= 1
        gl.barrier()                                 # 获取：见到 AG 后全 CTA 才读 landing
        rows = [sys_load(landing@rank[t, cols]) for t, cols in items] if AG else []   # 全部先发
        for (t, cols), r in zip(items, rows): out[t, cols] = r
        if tid == 0: epoch[i] = ep
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | nwg（host：= tiles） | CTA w：tile w、w + nwg、…，两遍同序 |
| CTA（256 线程） | 4 wave | 一个 tile：TT 个 token × H 列 |
| wave | IPW = ⌈TT·H/512 / 4⌉ 项 | 项 wave + 4j |
| lane | 8 列（16 B） | 每项一段；SUM 时 lane k < TOPK 读槽 k |
| 项 | (token, 512 列) | TOPK 行（COPY：1 行）的部分和 |

TT 由 host 定（[`_tile_tokens`](../../mega_moe_tp_gfx942.py#L495)：M < 1024 取 1，否则 H ≤ 1280 取 4、≤ 2560 取 2、更大取 1，
一个 tile 约 10 KB bf16；当前形状 IPW ≤ 3）。

**流水线**（每个 tile；没有跨 tile 的预取）
```text
第一遍：
  epoch[i]（普通读，uniform）→ ep = +1
  每项：SUM：lane < TOPK 读 loc（普通）→ readlane 取行号 → TOPK 条 16 B 路由行（普通）→ f32 槽序相加 → bf16
        COPY：src 一条 16 B；项之间 sched_barrier（下一项的加载不越过本项的相加）
  owner：
    wave 0：sys_load err → lane s 轮询 RS[i][s]（sys_load，s ≠ rank；s_sleep 2；超时 lane 0 原子 err |= 1）→ barrier
    每项：WORLD 条 inbox 读（sys_load；s = rank 那条偏移 _DROP 不访存，改用本卡部分和）→ f32 rank 序相加 → bf16
          → out 写（普通）→ AG：WORLD 条 landing 写（sys_store，q = rank 那条 _DROP）→ sched_barrier
    s_waitcnt vmcnt(0) → barrier → tid 0：WORLD 条 AG[i] 写（sys_store，q = rank 那条 _DROP）→ epoch[i] = ep（普通）
  非 owner：
    每项：一条 inbox 写（sys_store；owner 的 inbox 地址按 owner 从 tab 的 WORLD 个地址里 select 出来）
    s_waitcnt vmcnt(0) → barrier → tid 0：RS[i][rank] 写到 owner（sys_store）
第二遍（owner ≠ rank 的 tile）：
  epoch[i] → ep；wave 0：sys_load err → 64 lane 同址轮询本卡 AG[i]（sys_load，比低 30 位；见 ep | POISON：err |= 1）
  → barrier
  AG：IPW 条 landing 读（sys_load）全部先发 → IPW 条 out 写（普通）；AG = 0：不读不写
  tid 0：epoch[i] = ep
```
关键代码：[几何](tp_reduce.py#L236)、[项](tp_reduce.py#L276)、[等待](tp_reduce.py#L287)、[限时轮询](tp_reduce.py#L163)、
[部分和](tp_reduce.py#L318)、[相加](tp_reduce.py#L198)、[owner](tp_reduce.py#L355)、[发布 AG](tp_reduce.py#L408)、
[非 owner](tp_reduce.py#L422)、[拷贝](tp_reduce.py#L451)、[两遍](tp_reduce.py#L487)、[host 调用](../../mega_moe_tp_gfx942.py#L1476)。

**命名**：项、`ITEMS = TT·H/512`、`IPW`、`CH = H/512`；`base`：tile 的 token 在源区里的行偏移（(i // WORLD)·TT − t0）；
源区：owner 的 inbox 分 WORLD − 1 区，rank s 的部分和在区 (s − owner − 1) mod WORLD；`tab` / `landing` / `inbox` / `flags` /
`epoch` / `err` 与标志布局见 [docstring Naming](tp_reduce.py#L41)；`SYS` = 17（sc0 sc1，[L95](tp_reduce.py#L95)）。

**流水线不变量**
1. 设备上的 epoch、发布 / 获取、跨调用复用、防死锁、限时与毒值、逐位相同：[docstring 规则 1–6](tp_reduce.py#L46)。
2. 两遍走同一组 tile、同一次序（同一 `_loop(first, tiles, nwg)`），第一遍不等 AG：第二遍的等待只依赖各卡同号 CTA 的
   第一遍（规则 4）。
3. 非 owner 第一遍不写 epoch，第二遍重读得到同一个 ep；owner 第一遍写回、第二遍跳过：每卡每 tile 每次调用 epoch 恰好 +1。
4. 一个 tile 的 out 行只由取到它的 CTA 读写，先读（COPY 的部分和）后写（和或 landing 拷贝），且写依赖读到的值：
   所以 COPY 时 src 可以就是 out（原子 Down 之后正是如此）。
5. AG = 0：out 是 [M_r, H]，owner（就是本卡）写行 t − rank·M_r；第二遍只等 AG（owner 已读完本卡的 inbox 区，
   规则 3 复用的前提），不拷贝，照常推进 epoch。
6. 每项的 inbox 读 / landing 写都发 WORLD 条，自己那条偏移 `_DROP`：指令照发、不访存，部分和槽用 select 换成本卡的。
7. 路由行按 64 位基址各建描述符（[L342](tp_reduce.py#L342)）：routes 可超 2 GiB，不能用 32 位 voffset。
8. 所有等待限时（`limit`），err 已置位时只轮询一次；超时后照常走完并留下错误位，不挂死（规则 5）。

**资源**：不用 LDS；VGPR / SGPR 未测（每 wave 驻留 IPW 份 16 B 部分和，SUM 时另有 TOPK 行在途）。
workspace（[`_Workspace`](../../mega_moe_tp_gfx942.py#L510)）：`landing` [M_max, H] bf16、`inbox`（与 Down_tp 共用，
[max((WORLD − 1)·rps, WORLD·rps₂), H]）、`flags` [slots·(WORLD + 1)] int32，都是 hipIpc uncached 对称内存；
`tab`、`epoch` [slots] 在本卡显存。

## 流式跨卡求和：Down_tp 与 gather（[down_tp.py](down_tp.py#L218)）
把跨卡求和藏到 Down 下面：一次 launch 的 1-D grid 里，计算 CTA（C）按列块依次算 Down，插在其间的 reduce CTA（X）把已完成
列块的路由行按 token 求和推给 owner（RS），owner CTA（Y）按 rank 序求和后推给同伴（AG）；第二个 kernel gather（launch 名
一律 `a8w4_tp_reduce`）把别人求的行从 landing 拷进 out。算术、buffer 与标志协议同 tp_reduce：非 COPY 时结果与 tp_reduce
逐位相同，COPY 时同「原子 Down + tp_reduce 拷贝模式」（原子加的次序本身不定）。C 就是 gfx9_fp8 的 Down（prefill
`gemm2_a8w4` 或流式 `stream_a8w4`），经编译期 `TP=` 钩子加上角色分派与收尾；`TP=None` 时单卡 kernel 不变。层里由
[`_prefill_down`](../../mega_moe_tp_gfx942.py#L1302) / [`_stream_down`](../../mega_moe_tp_gfx942.py#L1373) 挂上，launch 名
`a8w4_down_{prefill,stream}_bm{bm}_tp[_rs]`，之后 [`_gather`](../../mega_moe_tp_gfx942.py#L1266)：例如 tp2 M410–559
`a8w4_down_stream_bm16_tp>a8w4_tp_reduce`（原子 Down，COPY）；ds M ≥ 1537 `a8w4_down_prefill_bm64_tp>a8w4_tp_reduce`；
DP M_r ≥ 385 `a8w4_down_prefill_bm64_tp_rs>a8w4_tp_reduce`（`OWNER=OWN_HOME, AG=0`）。各形状的档见入口文档第 2 节；
规则 7–10 见 [docstring](down_tp.py#L4)。

**Gluon 风格**（Down_tp）
```python
# grid = S·task_pad + (S − 1)·(NX + NY) + NXT + NYT 个 CTA（1-D，按 id 顺序起），256 线程（4 wave；流式 Down 也是 4 wave）
# 分段：C(0) | C(1)[:a] X(0) C(1)[a:b] Y(0) C(1)[b:] | … | C(S−1)[:a] X(S−2) C(S−1)[a:b] Y(S−2) C(S−1)[b:] | X(S−1) Y(S−1)
#       a / b = task_pad 的 AX/8、AY/8，向下取整到 4 的倍数
# 放置：C 同单卡 Down（gfx9_fp8 kernels.md），只把 routes（prefill 另有 loc）的写改成 ROUTE_AUX（默认 SYS，写穿）；
#       X / Y 全部 global → VGPR 直读直写、不用 LDS。X：下一步的路由行在本步写之前发（距离 1 步；跨 tile 时
#       连带再下一个 tile 的 loc）；Y：组内下一 tile 的 WORLD 份部分和在本 tile 写之前发。同卡跨 CTA 的读写也用 sys_*（规则 7）
@gluon.jit
def a8w4_down_{prefill,stream}_tp(<Down 的参数>, loc, out, ctr, tab, epoch, err, rank, tpq, limit):   # TPArgs
    role, j, q = roles(gl.program_id(0), task_pad)   # 角色、角色内序号、列块
    if role == C:                                    # Down 主体：一个排序块 × 列块 q（流式：q 的 GPB 个列组之一）
        down(q, j)                                   # routes 写带 ROUTE_AUX（ATOMIC：原子加进 out）；prefill 列块 0 再写 loc
        wait_vmem(); gl.barrier()             # 发布：本 CTA 的 routes / loc 都已写穿
        if tid == 0: atomic_add(ctr[q], 1)           # 空块、填充块也计数：ctr[q] == task_pad ⇔ 列块 q 完成
    elif role == X:                                  # tile j、j + NX、…（末块 NXT）
        wait_until(sys_load(ctr[0]), sys_load(ctr[q]), 到齐=都等于 task_pad); gl.barrier()   # lane 0 / 1 各读一个
        sl = sys_load(loc[tile 0 的 token, 槽])       # lane l < UPW·TOPK：unit l // TOPK 的槽 l % TOPK
        nxt = sys_load(routes[sl 的行, cols]) for tile 0 第 0 步; sl1 = sys_load(loc[tile 1]); wait_vmem()
        for n_i, i in enumerate(本 CTA 的 tile):
            for s in range(STEPS):                   # 一步 UX 个 unit，每 unit TOPK 行
                part = (Σ_k nxt[u][k]).to(bf16) for u   # f32 槽序；COPY：nxt 就是 out 的一行
                nxt = 第 s + 1 步的行，末步换成下一 tile 第 0 步的行 + sys_load(再下一 tile 的 loc)   # 预取
                sys_store(inbox@owner[(rank - owner - 1) % WORLD, (i // WORLD)·TT + t − t0, cols], part)
            if n_i % GX == GX − 1 or 最后一个:        # 一组：lane < GX 读 epoch[v] → vmcnt(0) → barrier
                for 组内 tile: sys_store(flags@owner.RS[v][rank], epoch[v] + 1)   # X 不写 epoch
    elif role == Y:                                  # 自有 tile rank + WORLD·i，组 i ∈ GY·(j + NY·m) + [0, GY)
        want = epoch[v_l] + 1                        # lane l < GY·WORLD：组内第 l // WORLD 个 tile、来源 l % WORLD（含本卡）
        late, _ = wait_until(sys_load(flags@rank.RS[v_l][l % WORLD]), 到齐=等于 want); gl.barrier()   # 一次轮询整组
        parts = [sys_load(inbox@rank[(s - rank - 1) % WORLD, …]) for s in range(WORLD)]   # 组首 tile；本卡的在区 WORLD − 1
        for k in range(GY):
            total = sum(parts).to(bf16)              # f32，rank 序，舍入一次
            if k + 1 < GY: parts = 下一 tile 的 WORLD 份   # 预取
            out[t − out0, cols] = total              # AG：for q != rank: sys_store(landing@q[t, cols], total)
        ep = epoch[v] + 1 (tid < GY); wait_vmem(); gl.barrier()   # 发布：全部 wave 的和都已写穿
        for q != rank: sys_store(flags@q.AG[v], (ep & 低 30 位) | (POISON if late else 0)); epoch[v] = ep
```

**Gluon 风格**（gather）
```python
# grid = ⌈S·tiles / G⌉ 个 CTA，256 线程；G = 4（S·tiles ≥ 2048，GATHER_ITEMS / GATHER_FROM）否则 1
# 放置：landing → VGPR → out，不用 LDS；一组 G 项的全部 landing 读先发、再全部写，组间无预取
@gluon.jit
def a8w4_tp_reduce(out, tab, epoch, ctr, err, M, rank, tpq, limit, nwg):   # compile_tp_gather
    if gl.program_id(0) == 0 and tid < S: sys_store(ctr[tid], 0)   # Down 已结束：下次调用从 0 计数
    for f0 in range(G·gl.program_id(0), S·tiles, G·nwg):           # 项 f = q·tiles + tile（列块优先）
        live = [f0 + g 的 tile 不归本卡 for g in range(G)]          # lane g < G 管一项
        ep = epoch[v] + 1
        late, ag = wait_until(sys_load(flags@rank.AG[v]) for live, 到齐=低 30 位等于 ep); gl.barrier()   # POISON：err |= 1
        rows = [sys_load(landing@rank[t, cols]) for live 项的 unit] if AG else []   # 全部先发
        for (t, cols), r in zip(…, rows): out[t, cols] = r
        epoch[v] = ep                                # wave 0 的 live lane（G = 1：tid 0）
```

**任务划分**
| 层级 | 形状 / 数量 | 负责的数据 |
|---|---|---|
| grid | [DownTP.grid](down_tp.py#L333) | 分段：C / X / Y |
| C，prefill（256 线程） | 每列块 task_pad | 一个排序块 × CWD 列 |
| C，流式（4 wave） | 每列块 task_pad | 一个排序块 × 512 列（一个列组） |
| X（256 线程） | 每列块 NX（末块 NXT） | tile j + NX·i；每 GX 个一组发 RS |
| Y（256 线程） | 每列块 NY（末块 NYT） | 自有 tile，每 GY 个一组 |
| wave（X / Y / gather） | 每 tile UPW unit | unit wave + 4u |
| lane | 8 列（16 B） | X：持一个 loc 槽；Y：轮询一个 RS |
| X 一步 | UX unit × TOPK 行 | 相加后推 owner |
| gather（256 线程） | ⌈S·tiles / G⌉ | G 个相邻项 |

task_pad：prefill 取 task_num 向上到 4·max(XCD_CHUNK, 1) 的倍数（[gemm2_a8w4.py](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L651)，
列块号与 XCD_CHUNK 置换见 [L165](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L165)）；流式取 task_num·GPB 向上到 4 的倍数
（[stream_a8w4.py](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1899)，排序块 = j // GPB，列组 = q·GPB + j % GPB）。

**当前分档**（[STREAM_SPEC / DOWN_TUNE](../../mega_moe_tp_gfx942.py#L409)、[STREAM_TIER_SPEC](../../mega_moe_tp_gfx942.py#L436)、
[`_stream_spec`](../../mega_moe_tp_gfx942.py#L461)）。全部 S = 5、TT = 8、AX / AY = 2 / 6、GX / GY = 4 / 2：

| (WORLD, H) | Down | NX / NY | NXT / NYT | 其他字段 |
|---|---|---|---|---|
| (2, 2560) | prefill | 32 / 16 | 256 / 128 | UX 1；PRO、LAS、WPE 3 |
| (4, 2560) | prefill | 64 / 32 | 256 / 128 | UX 1；PRO、SB、WPE 4 |
| (4, 5120) | prefill | 32 / 16 | 256 / 128 | SB、LAS |
| (2, 2560) | 流式 | 16 / 8 | 128 / 64 | UX 1 |
| (4, 2560)，M > 384 | 流式 | 16 / 8 | 128 / 64 | UX 1 |
| (4, 5120)，M > 512 | 流式 | 16 / 8 | 128 / 64 | — |

(4, 2560) prefill 在 M > 8192 时另加 `ROUTE_AUX = SYS_NT`；流式档 `COPY` = 是否原子 Down。DP：prefill 同表再加
`OWNER=OWN_HOME, AG=0`；流式表为空（`STREAM_TIER_SPEC_DP = {}`），走 tp_reduce。流式 C 取 4 wave、R = 8，SHARE / PREFETCH
（bm64 为 4，否则 2）/ LDS_ACTS（bm32 且 K % 256 = 0）/ W_AUX（H ≥ 5120 时 nt）同单卡流式 Down
（[调用](../../mega_moe_tp_gfx942.py#L1409)）；prefill C 的 TWO_PASS 取 blocks·S > 80。

**流水线**
```text
C(s)（单卡 Down 的流水线见 gfx9_fp8 kernels.md「A8W4 Down prefill / 流式」；钩子只加三处）：
  序言：roles → 非 C 的 CTA 不进主体（prefill active = 0、流式 live = 0；流式的 X / Y 仍写 LUT 表、读 block head）
  主循环照旧；routes 写带 ROUTE_AUX（SYS 写穿或 SYS_NT）；流式 ATOMIC：bf16x2 原子加进 out（agent 范围）
  prefill：主循环之后列块 0 的 CTA 写 loc（ROUTE_AUX）
  finish：s_waitcnt vmcnt(0) → barrier → tid 0：ctr[s] += 1（agent 原子）
X(q)：
  wave 0：sys_load err → lane 0 / 1 轮询 ctr[0] / ctr[q]（sys_load）到 task_pad → barrier
  序言：loc(tile 0)（sys_load）→ tile 0 第 0 步的 UX·TOPK 条路由行 + loc(tile 1)（sys_load）→ s_waitcnt vmcnt(0)
  每 tile、每步（循环只携带下一步的行与两份 loc）：
    本步 UX 个 unit：f32 槽序相加 → bf16（COPY：原样）
    发下一步的路由行（sys_load）；末步发下一 tile 第 0 步 + 再下一 tile 的 loc          ← 预取 1 步，先于本步写
    UX 条 inbox 写（sys_store）
  每 GX 个 tile（或最后一个）：lane < GX 读 epoch → s_waitcnt vmcnt(0) → barrier → RS[v][rank]（sys_store）
    （这次 vmcnt(0) 也等掉已发的下一步加载）
Y(q)，每组 GY 个自有 tile：
  lane < GY·WORLD：读 epoch（普通）→ wave 0 轮询 RS[v][s]（sys_load，含本卡）→ barrier
  组首 tile：UPW·WORLD 条 inbox 读（sys_load）
  每 tile：f32 rank 序相加 → bf16；发组内下一 tile 的 inbox 读（预取，先于本 tile 写）
           → out 写（普通，行减 out0）；AG：WORLD 条 landing 写（sys_store，q = rank 那条 _DROP）
  tid < GY 读 epoch → s_waitcnt vmcnt(0) → barrier → AG[v] = ep | POISON?（sys_store，各同伴）→ epoch[v] = ep
gather，每组 G 项：
  CTA 0：ctr[0..S) = 0（sys_store）
  读 epoch（G > 1：lane g < G 各读一项）→ wave 0 轮询 AG[v]（sys_load，比低 30 位；见 POISON：err |= 1）→ barrier
  AG：G 项全部 landing 读（sys_load）先发 → 全部 out 写（普通）；AG = 0：都不发
  epoch[v] = ep（普通）
```
关键代码：[segments](down_tp.py#L314)、[roles](down_tp.py#L339)、[finish](down_tp.py#L369)、[计数](down_tp.py#L381)、
[X 等 ctr](down_tp.py#L502)、[loc 槽](down_tp.py#L513)、[路由行](down_tp.py#L535)、[发 RS](down_tp.py#L564)、
[X 一步](down_tp.py#L596)、[X 序言](down_tp.py#L633)、[Y 轮询](down_tp.py#L681)、[Y 预取](down_tp.py#L708)、
[发 AG](down_tp.py#L726)、[gather](down_tp.py#L835)。prefill 钩子：[roles](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L165)、
[active](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L184)、[routes 写](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L472)、
[loc 写](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L550)、[finish](../moe_gemm_2stage/gfx9_fp8/gemm2_a8w4.py#L557)。
流式钩子：[断言](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1421)、[roles](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1816)、
[live](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1827)、[routes 写](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1689)、
[finish](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1849)。

**命名**：C / X / Y（`ROLE_*`）、q、`CWD`、unit、`UPW` / `UX` / `STEPS`、槽 v、`tpq`、`ctr`、`task_pad`、`out0` 见文首记号；
buffer 与标志布局见 [docstring Naming](down_tp.py#L69)，X / Y 每步的加载与写见 [docstring Steps](down_tp.py#L62)；
`i // WORLD`：tile 在其 owner 的 tile 里的序号；`n`：本 CTA 的 tile 数（X）或组数（Y）。

**流水线不变量**
1. tp_reduce 的规则 1–6 加本文件的 [规则 7–10](down_tp.py#L76)（同卡跨 XCD 一致、防死锁、跨调用复用、X / Y 循环的在途访存）。
2. C 从不等待：钩子只在 Down 末尾加 `finish`，单卡主体与 `TP=None` 时相同；非 C 的 CTA 在主体前被屏蔽。
3. `ctr[s] == task_pad` ⇔ 列块 s 的 routes（与 loc）全部写穿：每个 C（含空块、填充块）恰好加一次，且在
   `s_waitcnt vmcnt(0)` + barrier 之后（[`_signal`](down_tp.py#L381)）。
4. X 等 C(0) 与 C(q)：loc 由 prefill 的列块 0 写；流式 Down 不写 loc（launch 前已写好），等 ctr[0] 无害。
5. X 不写 epoch；本卡自有槽由 Y 在组末写回，别人的由 gather 写回：每卡每槽每次调用恰好 +1。X 读 epoch 在它发 RS 之前，
   Y 写 epoch 在见到该 RS 之后，所以 X 读到的是本次调用的旧值。
6. owner 自己的部分和也走 inbox（区 WORLD − 1），Y 一律从 inbox 读 WORLD 份。COPY 时 routes 就是 out：Y(q) 写某 tile 的 out 行前
   已等到本卡 X(q) 对它的 RS（X 对这些行的读已完成）；C(s > q) 只原子加列块 s 的列，与列块 q 不相交。
7. 槽按 `NSPLIT_MAX·tpq` 布局，不随本次 S、M 变：同一 (q, tile) 在任何调用里落同一槽，标志只存最近 epoch，无需清零（规则 1）。
8. gather 与 Down 同一 stream、在其后：CTA 0 清零 ctr 时本卡所有 X 已读完 ctr，下次调用的 C 从 0 计数。
9. landing / inbox 与 tp_reduce 共用，`flags2` / `epoch2` / `ctr` 是 Down_tp 自己的：跨路径复用同样由规则 9 保证。
10. `AX ≤ AY`、NX / NY 为 4 的倍数、`UPW·TOPK ≤ 64`（X 一条 loc 加载放下一个 tile 的槽）、`GY·WORLD ≤ 64`（Y 一次轮询放下
    一组）在 [`__post_init__`](down_tp.py#L271) 断言；流式另断言 `COPY == ATOMIC`、`H == N`、`TOPK ∈ {topk, topk + 1}`、
    不开 PRO / SB / LAS（[stream_a8w4.py](../moe_gemm_2stage/gfx9_fp8/stream_a8w4.py#L1421)）。

**资源**：C 的 LDS / VGPR 同单卡 Down（gfx9_fp8 kernels.md 两节的资源）；SB / LAS 抬高 VGPR，配 WPE（`waves_per_eu`
编译提示）保占用（[DownTP docstring](down_tp.py#L242)）。X / Y / gather 不用 LDS（流式 kernel 的 LDS 按 Down 分配，
X / Y 只写 LUT 表）；整个 launch 的 VGPR 取各角色的最大值，X / Y 的用量未测（X 每 wave 持下一步 UX·TOPK 份 16 B，
Y 持 UPW·WORLD 份）。workspace（[`_Workspace`](../../mega_moe_tp_gfx942.py#L510)）：`inbox` 至少 WORLD·rps₂ 行、
`flags2` [slots₂·(WORLD + 1) + 512] int32、`epoch2` [slots₂]、`ctr` 64 字（+ TRACE），slots₂ = 16·tpq。
