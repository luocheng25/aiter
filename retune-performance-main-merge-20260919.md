# main 合并、调优与性能复测（2026-09-19）

本文按执行顺序追加记录；保留已有性能文档不变。

## 1. 合并基线与约束

- 当前分支：`luocheng/moe_gemm_308-opt`。
- 合并前：`b64921fd526a27709b15346cedfdc7d06be02640`。
- 本次冻结的最新 `origin/main`：`c15adc7a6d1ad4422202424983aed97cab93db77`。
- PR #3987 的 main squash：`7438d1e16`。
- 回退分支：`backup/moe-gemm-308-opt-before-main-20260919`。
- 用户要求：调优时显式设置 `AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED=1`；两边复测保持相同设置。
- 初始已跟踪文件干净；现有未跟踪文件保留，不纳入合并提交。

### review 修复的合并处理

- 保留 main 的 `output` buffer 返回约定、缺失 FlyDSL 的安全回退、输入连续性/scale 校验，以及 bias、EP、预量化、局部 token、scatter 和 padding 的拒绝或回退。
- 保留分支拆分内核、BF16、gfx950 MXFP4/SiTUv2、batch 2–8、1x4/8x1/compact 路径。
- 将 main 的显式 RTA/RTE 转换开关合入实际被 Python 导入的拆分包；默认仍为保留 NaN 的 RTE。简化 RTA 不保证所有 NaN 编码的保留，因此实际测试额外检查有限值。
- 显式两阶段入口不再选择 `full_impl`；避免整图配置的空 stage1/stage2 被误用。
- e2e tuner 按完整 key 回写，从实际输出配置读取 baseline，保留无提升的已有行，拒绝非有限输出/时延。
- 修正旧整图 CSV 的 `block_m=16`：以配置编码中的 metadata `BLOCK_M` 为准。
- 跟随 main 的 Kimi a16w4 配置改名，避免新旧表重复 shape；gfx950 不计入 MI308 性能结果。

## 2. 环境与测量方案

- GPU：MI308X，gfx942，80 CU，192 GiB；正式 A/B 在同一张空闲卡上顺序运行。
- ROCm 7.2；Torch `2.9.1+rocm7.2.0.git7e1940d4`；Triton `3.7.0`；FlyDSL `0.3.2`。
- 解释器：系统 Python 3.10，显式载入现有 FlyDSL 环境；每个进程断言导入的 aiter 来自目标 worktree。
- 初始 PTL：所有卡 `enabled / VECTOR,F8`，无需修改；GPU2 初始 performance level 为 `auto`。
- 调优遵循 PR：`TUNE_ONLY=flydslv2,cktile,flydsl`，先 staged，再 `--e2e_tune`；Qwen397B 保留已知 ASM 排除项。最终以每个预期 shape 的 `OK` 判断，而非仅检查退出码。
- 正式性能独立于 tuner 的 profiler 均值：10 个输入 buffer 轮换、warmup 10、每轮 51 个 `cudaPerf` 样本取中位数；两分支交替次序、多轮复测。
- 延迟包含生产 `fused_moe` 整图，输入生成、权重转换和编译不计时；各 buffer 单独 CUDA graph 捕获。
- 有效 GEMM 工作量：`FLOPs = 6 × tokens × topk × model_dim × inter_dim`；`有效 TFLOPS = FLOPs / us / 10^6`。不包含排序/量化/激活算术计数；不是 ATT 模型 TFLOPS。
- 逐 shape 报告模型、权重/激活量化类型、batch、D/I/E/topk；不同 TP 形状不合并成单个原始数据点。

## 3. 合并后的初步验证

- 配置 family 重复检查：17/17 通过。
- 整图/缓存/AOT/两阶段回退及在线调优单元测试：39/39 通过。
- 相关文件 Ruff 检查通过。
- Qwen35B FP8、Qwen125B BF16、Hunyuan3 FP8 的 decode/prefill 代表 shape 通过；main 的 Qwen35B 代表 shape 通过。
- 验证驱动另外实际检查预分配输出 buffer 的返回地址及数值。
- 本节不以旧 CSV 中的 `us` 宣称性能；正式全量结果将在后续追加。