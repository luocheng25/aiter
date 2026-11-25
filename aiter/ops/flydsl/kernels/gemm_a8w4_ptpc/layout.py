# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fd2 layout of a dense MXFP4 weight for the gfx942 A8W4 GEMM and its FP4 -> FP8 decode.

``aiter.ops.flydsl.gemm_a8w4_ptpc.prepare_gemm_a8w4_ptpc_weight`` writes it once at load
time from a checkpoint MXFP4 weight with ``N`` output channels and reduction ``K``
(``N % 16 == 0``, ``K % 64 == 0``):

* ``q`` uint8 ``[N/16, K/64, 4, 16, 8]``: 16-row block, 64-K block ``kb``, lane group ``b``
  (K ``16b .. 16b+15`` of the block), row ``r``, 8 bytes = 16 FP4. Dword ``c`` holds K
  ``8c .. 8c+7`` of the lane group as byte ``i = k_i | k_(i+4) << 4`` (E2M1 nibbles, -0
  stored as +0), so one dword decodes to the eight FP8 operand bytes of an MFMA 16x16x32 lane.
* ``d`` uint8 ``[N/16, ceil(K/256), 2, 16, 4]``: ``min(cmax - code, 11) * 8`` of 32-K group
  ``2*kb + h`` of row ``r`` at ``[kb // 4, h, r, kb % 4]``; ``cmax`` is the row's largest
  E8M0 code. One dword holds the deltas of four 64-K blocks of one row and K half ``h``.
* ``s`` float32 ``[N]``: ``2**(cmax - 132)``, applied with the activation scale in the epilogue.

An E2M1 magnitude ``m`` decodes to the FP8 E4M3FNUZ value ``m * 2**(5 - delta)``, exact for
``delta <= 11``, so ``decoded * s`` equals the MXFP4 value. Groups with ``cmax - code > 11``
are stored as zero (their weights are below ``2**-11`` of the row's largest group scale).
"""

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

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


def fp4x4_to_fp8(raw, lut, odd):
    """Half of ``fp4x8_to_fp8``: the even (``odd=False``) or odd nibbles of a ``q`` dword ->
    one dword of 4 FP8 E4M3FNUZ bytes (the first / second dword of ``fp4x8_to_fp8``)."""
    lo, hi = lut
    raw = fx.Uint32(raw)
    sign = fx.Uint32(0x80808080)
    mag = fx.Uint32(0x07070707)
    if odd:
        return (raw & sign) | _perm(hi, lo, (raw >> 4) & mag)
    return ((raw << 4) & sign) | _perm(hi, lo, raw & mag)
