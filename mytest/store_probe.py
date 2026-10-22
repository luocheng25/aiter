#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write bandwidth of 16 B/lane vs 8 B/lane buffer stores for several lane->row mappings.

Each wave writes 32 rows x 64 BF16 columns (4 KB) per step into a [rows, 6208] BF16
buffer, mirroring the M128 Down epilogue. Patterns:
  row128:  8 consecutive lanes write one 128 B row segment (16 B each).
  row64:   4 consecutive lanes write one 64 B row segment (16 B each).
  mfma32:  MFMA C layout, lane (q, r) writes 8 B of row r (4 lanes x 8 B = 32 B rows).
"""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import gpu, range_constexpr

from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

STRIDE = 6208 * 2
COLS = 6144


def build(pattern, aux):
    words = 2 if pattern == "mfma32" else 4

    @flyc.kernel(known_block_size=[256, 1, 1])
    def store_probe(out: fx.Pointer, rows: fx.Int32):
        tid = fx.Int32(gpu.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        block = fx.Int32(gpu.block_idx.x)
        rsrc = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(out, num_records_bytes=fx.Int64(rows) * STRIDE)
        )
        row_base = block * 128 + wave * 32
        value = fx.Vector.filled(words, 0, fx.Int32) + lane
        for packet in range(fx.Int32(0), fx.Int32(COLS // 64), fx.Int32(1)):
            column = fx.Int32(packet) * 128
            for s in range_constexpr(4 if words == 4 else 8):
                if fx.const_expr(pattern == "row128"):
                    row, byte = s * 8 + lane // 8, (lane % 8) * 16
                elif fx.const_expr(pattern == "row64"):
                    row, byte = (s % 2) * 16 + lane // 4, (s // 2) * 64 + (lane % 4) * 16
                else:
                    row = (s % 2) * 16 + lane % 16
                    byte = (s // 2) * 32 + (lane // 16) * 8
                offset = (row_base + row) * STRIDE + byte
                rocdl_dialect.RawPtrBufferStoreOp(
                    value.ir_value(), rsrc, offset.ir_value(), column.ir_value(),
                    aux=ir.IntegerAttr.get(fx.Int32.ir_type, aux))

    @flyc.jit
    def launch(out: fx.Pointer, rows: fx.Int32, stream: fx.Stream):
        store_probe(out, rows).launch(grid=(rows // 128, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def main():
    torch.set_default_device("cuda")
    rows = 262144
    out = torch.empty((rows, STRIDE // 2), dtype=torch.bfloat16)
    nbytes = rows * COLS * 2
    for pattern in ("row128", "row64", "mfma32"):
        for aux in (0, 2):
            fn = build(pattern, aux)
            args = (_ptr(out), rows, torch.cuda.current_stream())
            for _ in range(3):
                _run_compiled(fn, *args)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(10):
                _run_compiled(fn, *args)
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) / 10 * 1e3
            print(f"{pattern:7s} aux={aux}: {us:8.1f} us  {nbytes / us / 1e6:.3f} TB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
