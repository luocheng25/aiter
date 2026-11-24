# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 dense A8W4 GEMM (FlyDSL): per-token FP8 activations x MXFP4 weights -> BF16.

``prepare_gemm_a8w4_ptpc_weight`` converts a checkpoint MXFP4 weight once, at load time,
into the fd2 layout of ``kernels/gemm_a8w4_ptpc/layout.py``. ``flydsl_gemm_a8w4_ptpc``
computes ``Y = X W^T`` from BF16 ``X``: one kernel quantizes ``X`` per token to FP8
E4M3FNUZ (``scale = amax / 240``, ``q = RNE(x / scale)``), the GEMM kernel decodes the
weight to FP8 in registers, accumulates in f32 and writes BF16. The call issues no host
synchronization and can be captured in a CUDA graph.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gemm_a8w4_ptpc import layout
from .kernels.gemm_a8w4_ptpc.gemm import (
    K_ALIGN,
    N_ALIGN,
    GemmConfig,
    compile_gemm_a8w4_ptpc,
    gemm_config_supported,
    gemm_kernel_name,
)
from .kernels.gemm_a8w4_ptpc.gemm_large import (
    KERNEL_NAME as LARGE_GEMM_KERNEL_NAME,
    compile_gemm_a8w4_ptpc_large,
    large_m_supported,
    xcd_panels,
)
from .kernels.gemm_a8w4_ptpc.quant import (
    compile_quant_x,
    quant_config,
    quant_kernel_name,
)
from .kernels.gemm_a8w4_ptpc.small_m import (
    compile_small_m,
    small_m_config,
    small_m_kernel_name_of,
)
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

__all__ = [
    "GemmA8W4PtpcWeight",
    "flydsl_gemm_a8w4_ptpc",
    "gemm_a8w4_ptpc_kernel_names",
    "gemm_a8w4_ptpc_shape_supported",
    "prepare_gemm_a8w4_ptpc_weight",
]

SUPPORTED_GFX = ("gfx942",)
# Byte offsets into the prepared weight are 32-bit.
_MAX_WEIGHT_BYTES = 1 << 31
_E8M0_NAN = 255
_FP4_DTYPES = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
_E8M0_DTYPES = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
# Token counts above this use the persistent large-M kernel when the shape allows it.
_LARGE_M = 2048
# Per weight shape (N, K) and token bucket (M up to the bucket): "small" for the single
# small-M kernel, the general kernel's GemmConfig fields (block_m, block_n, prefetch,
# x_stages, xcd_m, k_groups, tile_k) or "large" for the persistent kernel (both after the
# quantization); measured on MI308X with the end-to-end benchmark protocol.
_SMALL = "small"
_LARGE = "large"
_TK256 = 256
_MID_TABLE = {
    (8192, 2560): {
        16: _SMALL,
        32: (32, 64, 2, 2, 1, 1, _TK256),
        64: (64, 128, 2, 2, 1, 1, _TK256),
        128: (64, 128, 2, 2, 2, 1, _TK256),
        256: _LARGE,
    },
    (6656, 2560): {
        16: _SMALL,
        64: (32, 64, 2, 2, 1, 1, _TK256),
        128: (48, 128, 2, 2, 1, 1, _TK256),
        512: (64, 128, 2, 2, 2, 1, _TK256),
        1024: _LARGE,
    },
    (2560, 3072): {
        16: _SMALL,
        32: (16, 64, 2, 2, 2, 2, _TK256),
        64: (32, 64, 2, 2, 2, 2, _TK256),
        128: (32, 64, 2, 2, 4, 1, _TK256),
        1024: (64, 128, 2, 2, 4, 1, _TK256),
        2048: _LARGE,
    },
    (4096, 2560): {
        16: _SMALL,
        32: (32, 64, 2, 2, 1, 2, _TK256),
        64: (32, 64, 2, 2, 1, 1, _TK256),
        128: (64, 128, 2, 2, 2, 1, _TK256),
        1024: (64, 128, 2, 2, 4, 1, _TK256),
        2048: _LARGE,
    },
    (3584, 2560): {
        16: _SMALL,
        32: (32, 64, 2, 2, 1, 2, _TK256),
        64: (32, 64, 2, 2, 1, 1, _TK256),
        128: (32, 64, 2, 1, 2, 1, _TK256),
        256: (64, 128, 2, 2, 2, 1, _TK256),
        1024: (64, 128, 2, 2, 4, 1, _TK256),
        2048: _LARGE,
    },
    (2560, 1536): {
        20: _SMALL,
        32: (16, 64, 2, 2, 2, 2, _TK256),
        64: (32, 64, 2, 2, 2, 2, _TK256),
        128: (32, 64, 2, 2, 2, 1, _TK256),
        256: (64, 64, 2, 2, 4, 1, _TK256),
        1024: (64, 128, 2, 2, 4, 1, _TK256),
        2048: _LARGE,
    },
}

# Small-M tile kernels (9 <= M <= 16) on fewer row blocks per CTA than small_m_config picks,
# so all CUs get one: (N, K) -> (row blocks, CTAs); its LDS use only shrinks.
_SMALL_TILE_SPREAD = {(6656, 2560): (6, 80)}


def _small_config(m: int, n: int, k: int):
    """small_m_config's kernel, with the row blocks per CTA of _SMALL_TILE_SPREAD."""
    config = small_m_config(m, n, k)
    spread = _SMALL_TILE_SPREAD.get((n, k))
    if config is not None and spread is not None and config[0] == "tile" and m <= 16:
        kind, rb, ks, tq, kp, _ = config
        rb_new, num_wg = spread
        if rb_new <= rb and -(-n // 16 // num_wg) <= rb_new:
            config = (kind, rb_new, ks, tq, kp, num_wg)
    return config


@dataclass(frozen=True)
class GemmA8W4PtpcWeight:
    """MXFP4 weight ``[n, k]`` in the fd2 layout; build it with ``prepare_gemm_a8w4_ptpc_weight``.

    ``q`` uint8 (FP4 codes in MFMA K order), ``d`` uint8 (per 32-K group delta to the row's
    largest E8M0 code), ``s`` float32 ``[n]`` (per-channel factor); see
    ``kernels/gemm_a8w4_ptpc/layout.py``.
    """

    q: torch.Tensor
    d: torch.Tensor
    s: torch.Tensor
    n: int
    k: int


def _require_supported_gfx() -> None:
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_GFX:
        raise RuntimeError(
            f"FlyDSL A8W4 GEMM supports {', '.join(SUPPORTED_GFX)} only, got {arch!r}"
        )


def gemm_a8w4_ptpc_shape_supported(n: int, k: int) -> bool:
    """Whether a weight of ``n`` output channels and reduction ``k`` can run the kernel."""
    return (
        n > 0
        and k > 0
        and n % N_ALIGN == 0
        and k % K_ALIGN == 0
        and layout.weight_bytes(n, k) < _MAX_WEIGHT_BYTES
    )


def gemm_a8w4_ptpc_kernel_names(m: int, n: int, k: int) -> tuple[str, ...]:
    """Kernels ``flydsl_gemm_a8w4_ptpc`` launches, in order, for ``x`` [m, k] and weight [n, k]."""
    if not gemm_a8w4_ptpc_shape_supported(n, k):
        raise ValueError(_shape_error(n, k))
    if m == 0:
        return ()
    choice = _gemm_choice(m, n, k)
    if choice == _SMALL:
        return (small_m_kernel_name_of(_small_config(m, n, k)),)
    gemm_name = (
        LARGE_GEMM_KERNEL_NAME if choice == _LARGE else gemm_kernel_name(choice)
    )
    return (quant_kernel_name(k, quant_config(m, k)), gemm_name)


def _gemm_choice(m: int, n: int, k: int):
    """Path for x [m, k] x weight [n, k]: ``_SMALL`` (the single small-M kernel), or after the
    quantization ``_LARGE`` or a GemmConfig."""
    large_ok = large_m_supported(n, k)
    small_ok = small_m_config(m, n, k) is not None
    if m > _LARGE_M:
        if large_ok:
            return _LARGE
        return GemmConfig()
    table = _MID_TABLE.get((n, k))
    if table is None:
        if small_ok:
            return _SMALL
    else:
        # The smallest bucket that holds M; past the last one, the last one.
        entry = table[next((b for b in sorted(table) if m <= b), max(table))]
        if entry == _SMALL and small_ok:
            return _SMALL
        if entry == _LARGE and large_ok:
            return _LARGE
        if entry not in (_SMALL, _LARGE):
            cfg = GemmConfig(*entry)
            if gemm_config_supported(n, k, cfg):
                return cfg
    if m <= 128:
        cfg = GemmConfig(32, 64, 2, 2, 1, 1, _TK256)
    else:
        cfg = GemmConfig(64, 128 if n % 128 == 0 else 64, 2, 2, 4, 1, _TK256)
    return cfg if gemm_config_supported(n, k, cfg) else GemmConfig()


@functools.cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _shape_error(n: int, k: int) -> str:
    return (
        f"unsupported A8W4 weight shape N={n}, K={k}: needs N % {N_ALIGN} == 0, "
        f"K % {K_ALIGN} == 0 and N * K / 2 < 2**31"
    )


def _to_fd2_layout(q: torch.Tensor, codes: torch.Tensor):
    """FP4 bytes [n, k/2] + E8M0 codes [n, k/32] -> (q, d, s) of ``layout.py``."""
    n, half_k = q.shape
    k = half_k * 2
    kb = k // 64
    kq = layout.delta_dwords(k)
    codes = codes.to(torch.int32)
    cmax = codes.amax(dim=-1)
    diff = cmax[:, None] - codes
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).view(n, k // 32, 32)
    # -0 is stored as +0; groups more than MAX_DELTA codes below the row maximum flush to 0.
    nib = nib.masked_fill(nib == 8, 0).masked_fill(
        (diff > layout.MAX_DELTA)[..., None], 0
    )
    nib = nib.view(n, k // 8, 8)
    out_q = (nib[..., :4] | (nib[..., 4:] << 4)).view(n // 16, 16, kb, 4, 8)
    out_q = out_q.permute(0, 2, 3, 1, 4).reshape(-1)
    delta = (diff.clamp(max=layout.MAX_DELTA) * 8).to(torch.uint8).view(n, kb, 2)
    delta = F.pad(delta, (0, 0, 0, kq * 4 - kb))
    out_d = delta.view(n // 16, 16, kq, 4, 2).permute(0, 2, 4, 1, 3).reshape(-1)
    scale = torch.ldexp(torch.ones_like(cmax, dtype=torch.float64), cmax - 132)
    return out_q.contiguous(), out_d.contiguous(), scale.float().contiguous()


def prepare_gemm_a8w4_ptpc_weight(
    w: torch.Tensor, w_scale: torch.Tensor
) -> GemmA8W4PtpcWeight:
    """Convert a checkpoint MXFP4 weight to the A8W4 GEMM layout (run once, at load time).

    ``w`` [N, K/2] FP4x2 (``torch.float4_e2m1fn_x2`` or uint8; the low nibble is the even
    K), ``w_scale`` [N, K/32] E8M0 (``torch.float8_e8m0fnu`` or uint8), on the same GPU.
    Requires ``N % 64 == 0`` and ``K % 256 == 0``; an E8M0 code 255 (NaN) is rejected.
    Each 32-K group is decoded to FP8 relative to its row's largest E8M0 code; groups more
    than 11 codes (2**11) below it are treated as zero. Synchronizes with the host once.
    """
    _require_supported_gfx()
    if w.dtype not in _FP4_DTYPES:
        raise TypeError(f"w must be float4_e2m1fn_x2 or uint8, got {w.dtype}")
    if w_scale.dtype not in _E8M0_DTYPES:
        raise TypeError(f"w_scale must be float8_e8m0fnu or uint8, got {w_scale.dtype}")
    if w.dim() != 2 or w_scale.dim() != 2:
        raise ValueError(
            f"w and w_scale must be 2-D, got {tuple(w.shape)} and {tuple(w_scale.shape)}"
        )
    if w.device.type != "cuda" or w.device != w_scale.device:
        raise ValueError(
            f"w and w_scale must be on one GPU, got {w.device} and {w_scale.device}"
        )
    n, k = w.shape[0], w.shape[1] * 2
    if not gemm_a8w4_ptpc_shape_supported(n, k):
        raise ValueError(_shape_error(n, k))
    if tuple(w_scale.shape) != (n, k // 32):
        raise ValueError(
            f"w_scale must have shape {(n, k // 32)}, got {tuple(w_scale.shape)}"
        )
    codes = w_scale.contiguous().view(torch.uint8)
    if bool((codes == _E8M0_NAN).any()):
        raise ValueError("E8M0 scale code 255 (NaN) is not supported")
    q, d, s = _to_fd2_layout(w.contiguous().view(torch.uint8), codes)
    return GemmA8W4PtpcWeight(q=q, d=d, s=s, n=n, k=k)


def _quant_per_token(x: torch.Tensor):
    """Contiguous BF16 ``x`` [M, K] (M > 0, K % 64 == 0) -> FP8 E4M3FNUZ [M, K], f32 scale [M]."""
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fnuz, device=x.device)
    x_scale = torch.empty((m,), dtype=torch.float32, device=x.device)
    _run_compiled(
        compile_quant_x(k, x.device.index, quant_config(m, k)),
        ptr_arg(x),
        ptr_arg(xq),
        ptr_arg(x_scale),
        m,
        current_stream_arg(x.device),
    )
    return xq, x_scale


def flydsl_gemm_a8w4_ptpc(
    x: torch.Tensor,
    weight: GemmA8W4PtpcWeight,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``Y = X W^T``: BF16 ``x`` [M, K] (contiguous) x prepared weight [N, K] -> BF16 [M, N].

    ``x`` is quantized per token to FP8 E4M3FNUZ (``scale = amax / 240``, an all-zero row
    gets scale 1); accumulation is f32. ``out`` may be a preallocated contiguous BF16
    [M, N] tensor on ``x.device``. Any M (M = 0 returns the empty output).
    """
    _require_supported_gfx()
    if not isinstance(weight, GemmA8W4PtpcWeight):
        raise TypeError("weight must come from prepare_gemm_a8w4_ptpc_weight()")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D [M, K], got shape {tuple(x.shape)}")
    m, k = x.shape
    n = weight.n
    if k != weight.k:
        raise ValueError(f"x has K={k} but the weight has K={weight.k}")
    if x.device != weight.q.device:
        raise ValueError(f"x is on {x.device} but the weight is on {weight.q.device}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    else:
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n):
            raise ValueError(
                f"out must be bfloat16 {(m, n)}, got {out.dtype} {tuple(out.shape)}"
            )
        if out.device != x.device or not out.is_contiguous():
            raise ValueError(f"out must be contiguous on {x.device}")
    if m == 0:
        return out

    choice = _gemm_choice(m, n, k)
    if choice == _SMALL:
        # Small M: one kernel that quantizes x per token itself.
        _run_compiled(
            compile_small_m(_small_config(m, n, k), n, k, x.device.index),
            ptr_arg(x),
            ptr_arg(weight.q),
            ptr_arg(weight.d),
            ptr_arg(weight.s),
            ptr_arg(out),
            m,
            current_stream_arg(x.device),
        )
        return out
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fnuz, device=x.device)
    x_scale = torch.empty((m,), dtype=torch.float32, device=x.device)
    quant = compile_quant_x(k, x.device.index, quant_config(m, k))
    if choice == _LARGE:
        num_cus = _num_cus(x.device.index)
        gemm = compile_gemm_a8w4_ptpc_large(
            n, k, num_cus, x.device.index, xcd_panels(m, n, k, num_cus)
        )
    else:
        gemm = compile_gemm_a8w4_ptpc(n, k, x.device.index, choice)
    stream = current_stream_arg(x.device)
    quant_args = (ptr_arg(x), ptr_arg(xq), ptr_arg(x_scale), m, stream)
    gemm_args = (
        ptr_arg(xq),
        ptr_arg(x_scale),
        ptr_arg(weight.q),
        ptr_arg(weight.d),
        ptr_arg(weight.s),
        ptr_arg(out),
        m,
        stream,
    )
    # Both launches' arguments are built first, so the GPU waits less between the kernels.
    _run_compiled(quant, *quant_args)
    _run_compiled(gemm, *gemm_args)
    return out
