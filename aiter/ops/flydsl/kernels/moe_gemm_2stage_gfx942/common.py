# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.float64: fx.Float64,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


def down_device_config_from_properties(gcn_arch_name, cu_count):
    is_gfx942_80cu = gcn_arch_name.split(":", 1)[0] == "gfx942" and cu_count == 80
    return is_gfx942_80cu, 4 if is_gfx942_80cu else 8


def get_down_device_config():
    if not torch.cuda.is_available():
        return False, 8
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return down_device_config_from_properties(
        properties.gcnArchName,
        properties.multi_processor_count,
    )


@functools.cache
def _get_device_cache_key(device):
    properties = torch.cuda.get_device_properties(device)
    return (
        device,
        properties.name,
        properties.gcnArchName,
        properties.multi_processor_count,
    )


def get_device_cache_key():
    if not torch.cuda.is_available():
        return None
    return _get_device_cache_key(torch.cuda.current_device())


def torch_tensor_to_pointer(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())
