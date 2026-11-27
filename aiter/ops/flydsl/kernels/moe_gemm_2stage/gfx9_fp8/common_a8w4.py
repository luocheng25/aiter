# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W4 MoE (FP8 activation x MXFP4 weight, gfx942): private weight layout, FP4 -> FP8 decode, device helpers.

``aiter.ops.flydsl.moe_a8w4.prepare_moe_a8w4_ptpc_weight`` writes the layout once at load time; per expert,
for a weight with ``N`` output rows and reduction ``K`` (``N % 16 == 0``, ``K % 64 == 0``):

* ``q`` uint8 ``[N/16, K/64, 4, 16, 8]``: 16-row block, 64-K block ``kb``, lane group ``b``
  (K ``16b .. 16b+15`` of the block), row ``r``, 8 bytes = 16 FP4. Dword ``c`` holds K
  ``8c .. 8c+7`` as byte ``i = k_i | k_(i+4) << 4`` (E2M1 nibbles, -0 stored as +0). Lane
  ``16b + r`` of an MFMA 16x16x32 wave reads bytes ``8l .. 8l+7`` of the 512-byte block and
  dword ``c`` becomes the operand of MMA ``c`` (the K permutation of the FP8 prefill kernels).
* ``d`` uint8 ``[N/16, ceil(K/256), 2, 16, 4]``: ``min(cmax - code, 11) * 8`` of 32-K group
  ``2*kb + h`` of row ``r`` at ``[kb // 4, h, r, kb % 4]``; ``cmax`` is the row's largest E8M0
  code. Lane ``16b + r`` loads dword ``[kb // 4, b // 2, r]``.
* ``s`` float32 ``[N]``: ``2**(cmax - 132)``, applied with the activation scale in the epilogue.

An E2M1 magnitude ``m`` decodes to the FP8 E4M3FNUZ value ``m * 2**(5 - delta)``, exact for
``delta <= 11``, so ``decoded * s`` equals the MXFP4 value. Groups with ``cmax - code > 11``
are stored as zero (their weights are below ``2**-11`` of the row's largest group scale).

This is the fd2 layout of the dense A8W4 GEMM (``kernels/gemm_a8w4_ptpc/layout.py`` on the gemm-a8w4
branch), one weight per expert. The helpers from ``W_BLOCK_BYTES`` to ``fp4x8_to_fp8`` are written exactly as
there, so whichever of the two lands second can import the other's.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from . import common as fxh
from .common import BufferTensor

# Bytes of one 16-row x 64-K block of ``q`` and of one 16-row x 256-K block of ``d``.
W_BLOCK_BYTES = 512
D_BLOCK_BYTES = 128
# Largest delta decoded exactly; groups further below the row maximum are stored as zero.
MAX_DELTA = 11


def delta_dwords(k):
    """Delta dwords per row and K half: one per four 64-K blocks (``k`` is a multiple of 64)."""
    return (k // 64 + 3) // 4


def weight_bytes(n, k):
    return n * k // 2


def delta_bytes(n, k):
    return n // 16 * delta_dwords(k) * D_BLOCK_BYTES


def _perm(hi, lo, sel):
    # FlyDSL 0.5.3 has no typed byte permute. One v_perm_b32 picks four bytes of the
    # 8-byte table (hi:lo) and is the whole per-nibble LUT step of the decode; a
    # shift / mask / select form costs several VALU ops per byte.
    return fx.Uint32(fx.rocdl.perm_b32(fx.Uint32(hi), fx.Uint32(lo), fx.Uint32(sel)))


def fp8_lut(deltas, selector):
    """|E2M1| -> FP8 byte table for the delta in byte ``j`` of ``deltas``.

    ``selector = j * 0x01010101`` broadcasts that byte.

    Returns ``(lo, hi)`` holding the FP8 bytes of magnitudes 0..3 and 4..7; entry 0 is zero.
    At delta 0 the table is ``m * 2**5``: 0x60 (0.5) ... 0x7C (6); each delta step subtracts
    one from every exponent field.
    """
    d = _perm(deltas, deltas, selector)
    lo = (fx.Uint32(0x6C686080) - d) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - d
    return lo, hi


def fp4x8_to_fp8(raw, lut):
    """One ``q`` dword (8 FP4, K order 0..7 after decode) -> 8 FP8 E4M3FNUZ values."""
    lo, hi = lut
    raw = fx.Uint32(raw)
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    even = ((raw << 4) & sign) | _perm(hi, lo, raw & mag)
    odd = (raw & sign) | _perm(hi, lo, (raw >> 4) & mag)
    return Vec.from_elements([even, odd], fx.Uint32).bitcast(fx.Float8E4M3FNUZ)


# XCDs of an MI308X: workgroup L of a launch runs on XCD L % XCDS (the XCD-chunked prefill grids).
XCDS = 4
# LDS of one workgroup on gfx942 (the shape checks of the decode kernels).
LDS_BYTES = 64 * 1024


def expert_bytes(p, expert, nbytes, masked=False):
    """Buffer view of one expert's ``nbytes`` bytes of a byte array. With ``masked`` a negative ``expert`` (a masked
    route, id -1) gets a view of 0 bytes: the range check returns 0 for every load through it, which reads no memory.
    """
    base = fx.recast_iter(fx.Uint8, fxh._as_ptr(p))
    if not masked:
        return fx.rocdl.make_buffer_tensor(
            fx.make_view(base + fx.Int64(expert) * nbytes, fx.make_layout(nbytes, 1)),
            max_size=False,
            num_records_bytes=nbytes,
        )
    # The id zero-extended (a masked one gives some offset that the 0-byte view never reads): sign-extending it makes
    # LLVM build its high word right after its load, and the wait for that load then precedes the loads issued
    # ahead of this view.
    offset = fx.Int64(fx.Uint32(expert)) * nbytes
    records = (fx.Int32(expert) >= 0).select(fx.Int64(nbytes), fx.Int64(0))
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(base + offset, fx.make_layout(nbytes, 1)),
        max_size=False,
        num_records_bytes=records,
    )


def load_dwordx2(buffer, voffset_bytes, soffset_bytes):
    """Raw 8-byte buffer load at (VGPR, SGPR) byte offsets -> 2 x Uint32."""
    packet = BufferTensor(fx.make_view(fx.get_iter(buffer), fx.make_layout(8, 1)))
    return packet.load(
        voffset_bytes=voffset_bytes, soffset_bytes=soffset_bytes
    ).bitcast(fx.Uint32)


def load_dword(buffer, voffset_bytes, soffset_bytes):
    """Raw 4-byte buffer load -> 1 x Uint32 (scalar i32 result; LLVM cannot scalarize v1i32)."""
    # Raw buffer load: fx.copy with a BufferCopy32b atom addresses through a view layout and
    # fills a 1-element vector fragment; the weight loops need a per-lane VGPR plus a
    # wave-uniform SGPR byte offset (the block base stays scalar) and a scalar i32 result.
    resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(buffer))
    value = rocdl_dialect.RawPtrBufferLoadOp(
        fx.Int32.ir_type,
        resource,
        voffset_bytes.ir_value(),
        soffset_bytes.ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    ).result
    return Vec.from_elements([fx.Uint32(value)], fx.Uint32)


def store_vector(buffer, vector, voffset_bytes, soffset_bytes):
    """Raw buffer store of a 2/4-element 32-bit vector; out-of-range bytes are dropped."""
    # Raw buffer store: the typed fx.copy store takes its offset from a view layout; the epilogue
    # computes a per-lane byte offset and sends padding rows past the buffer end, where the
    # hardware range check drops them (no branch).
    resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(buffer))
    rocdl_dialect.RawPtrBufferStoreOp(
        _raw(vector),
        resource,
        voffset_bytes.ir_value(),
        soffset_bytes.ir_value(),
        aux=ir.IntegerAttr.get(fx.Int32.ir_type, 0),
    )


def swiglu_clamp(gate, up, limit):
    """SwiGLU limit of DeepSeek-V4.1 on the dequantized f32 gate / up values: ``min(gate, limit)`` and
    ``up`` clamped to ``[-limit, limit]`` (NaN-ignoring min / max). ``limit`` is a compile-time constant;
    0 returns the values unchanged and emits no instructions."""
    if limit <= 0:
        return gate, up
    top = fx.Float32(limit)
    return fx.minnumf(gate, top), fx.minnumf(fx.maxnumf(up, fx.Float32(-limit)), top)


def swiglu_kernel_name(base, limit):
    """Kernel symbol of a Gate/Up kernel: FlyDSL's default (``<base>_0``) without a SwiGLU limit, ``<base>_swiglu``
    with one."""
    return f"{base}_swiglu" if limit > 0 else None


def id_index(route, topk, id_cols):
    """Element of route ``route`` (= token * topk + slot) in routing tensors of ``id_cols`` columns per token
    (``id_cols`` > topk: a fused shared expert's column follows the topk routed ones, ``moe_a8w4._SharedFold``, and
    the decode kernels leave it unread); trace-time identity for ``id_cols == topk``."""
    if id_cols == topk:
        return route
    return (route // topk) * id_cols + route % topk


def lane_offsets(tid, K):
    """Loop-invariant per-lane byte offsets ``(weight, delta)`` of a wave's 16-row block.

    Wave ``w`` owns 16-row block ``w`` of each 64-row group; callers add the group's
    16-row block index through the scalar offset.
    """
    lane = tid % 64
    wave = tid // 64
    KB = K // 64
    KQ = delta_dwords(K)
    w_off = lane * 8 + wave * (KB * W_BLOCK_BYTES)
    d_off = (lane // 32) * 64 + (lane % 16) * 4 + wave * (KQ * D_BLOCK_BYTES)
    return fx.Int32(w_off), fx.Int32(d_off)
