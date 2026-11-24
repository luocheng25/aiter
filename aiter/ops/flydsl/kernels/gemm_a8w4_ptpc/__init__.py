# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 dense A8W4 GEMM kernels (per-token FP8 activations x MXFP4 weights).

``layout`` documents the fd2 weight layout and holds the FP4 -> FP8 decode helpers,
``quant`` the per-token activation quantization kernel and ``gemm`` the GEMM kernel.
The public entry points live in ``aiter.ops.flydsl.gemm_a8w4_ptpc``.
"""
