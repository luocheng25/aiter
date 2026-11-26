# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sorted-block lists of the token chunks of the chunked A8W4 prefill (``fused_moe_a8w4``'s ``prefill_down.chunks``).

A chunk descriptor row (int32, ``DESC_WORDS + cap`` words, one row per chunk) is passed to the ``CHUNKED`` Gate/Up
and Down in place of ``num_valid_ids``: ``[0]`` the valid count, ``[1]`` / ``[2]`` the chunk's tokens ``[t_lo,
t_hi)``, ``[3]`` its index, ``[4]`` the number of its sorted blocks, ``[DESC_WORDS + i]`` its i-th sorted block (in
sorted order). The host fills words 1-3; ``compile_chunk_blocks`` the rest after the sort. A CHUNKED kernel launches
an upper bound of the chunk's blocks (``chunk_bound``) and maps position ``p < desc[4]`` to block ``desc[DESC_WORDS +
p]`` (``common.chunk_block``): the grid holds few idle workgroups, against all the sorted blocks of the call when every
chunk ran the full grid.

A block belongs to the chunk holding the token of its first route. Within an expert the sort keeps the token order,
so after the chunks up to c every route of the tokens below chunk c's end is computed.

``compile_chunk_blocks``: one workgroup of 16 waves; wave w takes sorted blocks ``[64 K w, 64 K (w + 1))``, 64 at a
time (``K`` per lane). Per chunk, a ballot over 64 blocks ranks a block among its wave's blocks of the chunk; the
waves' counts (LDS) give each wave's offset in the list.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

from ..moe_gemm_2stage.gfx9_fp8 import common as fxh
from .stream import _DROP, _address, _load, _rsrc, _store

DESC_WORDS = 8
WAVES = 16
# Room past a chunk's block bound: positions of the XCD remap of the launched grid (multiples of 8 * XCD_CHUNK) and
# the reduce / owner workgroups' positions in the Down stay inside the row.
SLACK = 1024


def desc_row_words(grid):
    """Words of a chunk descriptor row for a call of ``grid`` sorted blocks."""
    return DESC_WORDS + grid + SLACK


def chunk_bound(tokens, topk, block_m, experts, grid):
    """Upper bound of the sorted blocks of a chunk of ``tokens`` tokens: an expert with ``k`` of the chunk's routes
    starts at most ``ceil(k / block_m)`` blocks in it."""
    return min(grid, tokens * topk // block_m + min(experts, tokens * topk) + 1)


def _ballot(pred):
    return fx.Int64(rocdl.ballot(T.i64, _raw(pred)))


def _popcount(mask):
    halves = Vec.from_elements([mask], fx.Int64).bitcast(fx.Int32)
    return fx.Int32(fx.math.ctpop(_raw(fx.Int32(halves[0])))) + fx.Int32(
        fx.math.ctpop(_raw(fx.Int32(halves[1])))
    )


@functools.cache
def compile_chunk_blocks(*, CHUNKS, K, BM, ROW):
    """Launcher ``(sorted_ids, num_valid_ids, desc, stream)``: the block lists of ``CHUNKS`` chunks (module docstring)
    for up to ``64 * WAVES * K`` sorted blocks of ``BM`` rows; ``desc`` int32 [CHUNKS, ROW] (``desc_row_words``)."""
    assert 1 <= CHUNKS <= 4 and K >= 1, (CHUNKS, K)

    @fx.struct
    class Storage:
        cnt: fx.Array[fx.Int32, WAVES * CHUNKS, 16]

    @flyc.kernel(name="a8w4_chunk_blocks", known_block_size=[64 * WAVES, 1, 1])
    def chunk_blocks(p_sorted_ids: fx.Pointer, p_num_valid: fx.Pointer, p_desc: fx.Pointer):
        tid = fx.Int32(gpu.thread_idx.x)
        lane = tid & 63
        wave = fx.Int32(rocdl.readfirstlane(T.i32, _raw(tid >> 6)))
        counts = fx.SharedAllocator().allocate(Storage).peek().cnt.ptr
        valid = fx.Int32(fx.recast_iter(fx.Int32, fxh._as_ptr(p_num_valid))[0])
        blocks = (valid + (BM - 1)) // BM
        desc = fx.recast_iter(fx.Int32, fxh._as_ptr(p_desc))
        bounds = [(fx.Int32(desc[c * ROW + 1]), fx.Int32(desc[c * ROW + 2])) for c in range_constexpr(CHUNKS)]
        ids_rsrc = _rsrc(_address(p_sorted_ids), fx.Int64(valid) * 4)
        desc_rsrc = _rsrc(_address(p_desc), CHUNKS * ROW * 4)
        block = [(wave * K + j) * 64 + lane for j in range_constexpr(K)]
        token = [
            fx.Int32(_load(ids_rsrc, (block[j] < blocks).select(block[j] * (BM * 4), fx.Int32(_DROP)), 0, 1)[0])
            & fx.Int32(0xFFFFFF)
            for j in range_constexpr(K)
        ]

        def member(j, c):
            lo, hi = bounds[c]
            return (block[j] < blocks) & (token[j] >= lo) & (token[j] < hi)

        for c in range_constexpr(CHUNKS):
            n = fx.Int32(0)
            for j in range_constexpr(K):
                n = n + _popcount(_ballot(member(j, c)))
            if lane == 0:
                counts[wave * CHUNKS + fx.Int32(c)] = n
        gpu.barrier()
        below = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        for c in range_constexpr(CHUNKS):
            at = fx.Int32(0)
            total = fx.Int32(0)
            for w in range_constexpr(WAVES):
                n = fx.Int32(counts[fx.Int32(w * CHUNKS + c)])
                total = total + n
                at = at + (fx.Int32(w) < wave).select(n, fx.Int32(0))
            for j in range_constexpr(K):
                inside = member(j, c)
                mask = _ballot(inside)
                pos = at + _popcount(mask & below)
                _store(
                    desc_rsrc,
                    [block[j]],
                    inside.select((c * ROW + DESC_WORDS + pos) * 4, fx.Int32(_DROP)),
                )
                at = at + _popcount(mask)
            head = (wave == 0) & (lane < 2)
            _store(
                desc_rsrc,
                [(lane == 0).select(valid, total)],
                head.select((c * ROW + lane * 4) * 4, fx.Int32(_DROP)),
            )

    @flyc.jit
    def launch(p_sorted_ids: fx.Pointer, p_num_valid: fx.Pointer, p_desc: fx.Pointer, stream: fx.Stream):
        CompilationContext.get_current()
        chunk_blocks(p_sorted_ids, p_num_valid, p_desc).launch(
            grid=(1, 1, 1), block=(64 * WAVES, 1, 1), stream=stream
        )

    return launch
