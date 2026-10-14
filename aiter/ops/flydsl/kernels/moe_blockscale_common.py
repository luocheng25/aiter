# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small shared boundary for the gfx950 FP8 blockscale kernels.

Reuse Aiter's pointer, tensor and explicit FMA primitives. Buffer/LDS offsets
remain precomputed byte offsets; no tensor or stream is retained in code caches.
"""

import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import rocdl

from .moe_gemm_2stage.common import _ptr, view_as_torch_tensor
from .moe_gemm_2stage.gemm2_common import BufferTensor, LdsTensor, eltwise_op


def get_device_cache_key():
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return (
        device,
        properties.name,
        properties.gcnArchName,
        properties.multi_processor_count,
    )


def rocdl_aux(value=0):
    """The pinned FlyDSL raw buffer operations take an i32 attribute."""
    return ir.IntegerAttr.get(fx.Int32.ir_type, value)


def s_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    # Preserve the ported gfx950 pipeline's exact wait-counter encoding.
    rocdl.s_waitcnt(
        (vmcnt & 15) | (expcnt << 4) | (lgkmcnt << 8) | ((vmcnt >> 4) << 14)
    )


__all__ = [
    "BufferTensor",
    "LdsTensor",
    "_ptr",
    "eltwise_op",
    "get_device_cache_key",
    "rocdl_aux",
    "s_waitcnt",
    "view_as_torch_tensor",
]
