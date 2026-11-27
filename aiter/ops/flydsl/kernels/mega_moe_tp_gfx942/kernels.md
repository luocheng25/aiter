# mega_moe_tp_gfx942：kernel 设计（任务划分与流水线）

<!-- 入口文档第 2 节只放骨架；这里给每个核心 kernel 的完整任务划分与逐拍流水线，写给要读 / 改 kernel 的人。
     只写当前实现，不写过程（哪个里程碑、谁定的、试过什么）；关键行链接到代码；与代码对不上就改这里。 -->

本层目前没有自己的 kernel：`MegaMoeTPGfx942.forward` = 本卡调单卡入口 `flydsl_moe_a8w4_ptpc` + 一次 all-reduce。
- 计算 kernel（decode / 去重 / 混合 decode、流式、prefill、shared 的折叠 GEMM、排序与行量化）：任务划分与流水线见
  gfx9_fp8 的 [kernels.md](../moe_gemm_2stage/gfx9_fp8/kernels.md)，各档条件见其 [README](../moe_gemm_2stage/gfx9_fp8/README.md)
  第 2 节，host 分派在 [moe_a8w4.py](../../moe_a8w4.py)。
- 跨卡求和：aiter `CustomAllreduce`（[custom_all_reduce.py](../../../../dist/device_communicators/custom_all_reduce.py)）
  或 RCCL `all_reduce`，按 [mega_moe_tp_gfx942.py](../../mega_moe_tp_gfx942.py) 的 `AR_RCCL_MIN_BYTES` 选，都不是本目录的 kernel。
- 把跨卡求和融进 kernel 的版本（Down decode 跨卡版、`a8w4_tp_reduce`、Down_tp 与 gather、SDMA token 分块；M1 起移植，
  源码可从 tag `mega-tp-a8w4-M7` 取回）加进本目录时，按模板逐个补节。
