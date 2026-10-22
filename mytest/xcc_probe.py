#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record HW_REG_XCC_ID per workgroup to check the block % 8 == XCD assumption."""

import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu

from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def build(lds_bytes, spin):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def xcc_probe(out: fx.Pointer):
        tid = fx.Int32(gpu.thread_idx.x)
        storage = fx.SharedAllocator().allocate(fx.Array[fx.Int32, lds_bytes // 4, 16])
        lds = storage.peek().view(fx.make_layout(lds_bytes // 4, 1))
        lds[tid] = tid
        gpu.barrier()
        xcc = fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
                                       "=s", has_side_effects=True))
        for _ in range(fx.Int32(0), fx.Int32(spin), fx.Int32(1)):
            lds[tid] = lds[(tid + 1) % 256] + 1
        if tid == 0:
            out[fx.Int32(gpu.block_idx.x)] = (xcc & 0xF) + lds[0] * 0

    @flyc.jit
    def launch(out: fx.Pointer, n: fx.Int32, stream: fx.Stream):
        xcc_probe(out).launch(grid=(n, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def main():
    torch.set_default_device("cuda")
    for lds_bytes, spin, n in ((1024, 0, 4096), (16384, 64, 100000), (49152, 256, 20000)):
        out = torch.full((n,), -1, dtype=torch.int32)
        _run_compiled(build(lds_bytes, spin), _ptr(out), n, torch.cuda.current_stream())
        torch.cuda.synchronize()
        ids = torch.arange(n, dtype=torch.int32) % 8
        match = float((out == ids).float().mean())
        counts = torch.bincount(out.long(), minlength=8).tolist()
        print(f"lds={lds_bytes} spin={spin} grid={n}: match(block%8)={match:.6f} per-xcc={counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
