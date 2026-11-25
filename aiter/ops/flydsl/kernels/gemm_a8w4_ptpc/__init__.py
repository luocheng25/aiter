# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 dense A8W4 GEMM kernels (FP8 activations, per token or 1x32, x MXFP4 weights).

``layout`` documents the fd2 weight layout and holds the FP4 -> FP8 decode helpers,
``quant`` the per-token activation quantization kernel, ``quant_1x32`` the 1x32 (MXFP8)
one and ``gemm`` the GEMM kernel.
The public entry points live in ``aiter.ops.flydsl.gemm_a8w4_ptpc``.
"""
