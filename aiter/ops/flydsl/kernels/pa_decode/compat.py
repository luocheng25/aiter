# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional PA compiler helpers for the supported FlyDSL runtimes."""

try:
    from flydsl.runtime.device import check_smem_capacity as _native_check_smem_capacity
except ImportError:
    _native_check_smem_capacity = None


def check_smem_capacity(size_bytes: int, architecture: str) -> None:
    """Use the native check when available, or the gfx942 64 KiB bound."""
    if _native_check_smem_capacity is not None:
        return _native_check_smem_capacity(size_bytes, architecture)
    if architecture != "gfx942":
        raise NotImplementedError(
            "FlyDSL without check_smem_capacity is supported only for gfx942 PA"
        )
    if type(size_bytes) is not int or not 0 <= size_bytes <= 65536:
        raise ValueError(
            f"gfx942 PA shared-memory request must fit 64 KiB, got {size_bytes!r}"
        )


class _UnavailableTdm:
    def __getattr__(self, name):
        raise NotImplementedError(
            f"Installed FlyDSL has no TDM operation {name!r}; "
            "this PA path requires a runtime with TDM support"
        )


try:
    from flydsl.expr.rocdl import tdm_ops
except ImportError:
    tdm_ops = _UnavailableTdm()
