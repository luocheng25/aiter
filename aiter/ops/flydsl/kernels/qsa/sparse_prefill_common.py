# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Settings and gfx942 BF16 D256 helpers shared by the QSA sparse attention kernels.

The first part holds the per-platform calibration and the private resource
policy. The platform entry is picked by gfx model and CU count (nearest
calibrated CU count of the same model) and read once per process;
AITER_QSA_PLATFORM overrides fields of the picked entry, e.g.
``AITER_QSA_PLATFORM="union_step=3.0,max_packed_kv_bytes=0"``.

The second part holds raw-buffer operations and pipeline barriers. The third
part holds the QK and PV MFMA loops and the lazy softmax rescale, each defined
once for the union and both direct kernels. The fourth part holds the
BM128/BN64 MFMA, softmax and LDS pieces of the eight-wave staggered D256
pipeline that the union kernel runs. Everything below FlyDSL's stable API is in
sparse_prefill_primitives (prim).
"""

import dataclasses
import functools
import logging
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from . import sparse_prefill_primitives as prim

# ---- Platform calibration and private resource policy ----

# Resource policy; private, not configurable from outside.
WORKSPACE_CACHE = 8  # layouts kept per process; captured workspaces are never evicted
ARENA_GROWTH_PERCENT = 125  # a growing arena takes at least this share of its size
SCRATCH_ALIGN = 256  # bytes between the scratch views of an arena
UNION_SCRATCH_LIMIT = 4 << 30  # union scratch bytes; past it a layout runs direct only
MAX_ROW_PACKED_KV_BYTES = 1 << 30  # PK+PV bytes of the pack_tokens_per_row rule


@dataclasses.dataclass(frozen=True)
class Platform:
    """Calibrated per GPU; the route-model terms are microseconds."""

    union_fixed: float  # fixed cost of the union phase
    union_step: float  # per N64 step of the slowest union CTA
    row_step: float  # per BN32 step of the longest direct row
    route_margin: float  # keeping every union tile must be this share slower to lose
    direct_fixed: float  # fixed cost of the direct phase
    pack_block: float  # per packed KV block and KV head
    direct_cu_step: float  # per BN32 direct step and KV head, times the CU count
    union_ratio: float  # union while padded union steps <= ratio x direct steps
    union_min_rows: int  # below max(rows, rows_per_head x H/HK) direct-only wins
    union_min_rows_per_head: int
    wide_tiles_per_cu: float  # full H6/H3 prefill: BQ21/BQ42 from this many tiles/CU
    max_packed_kv_bytes: int  # direct packs KV while PK+PV fits, else reads the raw KV
    # Past that budget, direct still packs a context of at most this many tokens per query row
    # (packing copies the whole context each call).
    pack_tokens_per_row: int


PLATFORMS = {
    # MI308X phase timings.
    ("gfx942", 80): Platform(
        union_fixed=10.0,
        union_step=3.2,
        row_step=0.7,
        route_margin=0.05,
        direct_fixed=41.0,
        pack_block=0.0041,
        direct_cu_step=0.304,
        union_ratio=1.7,
        union_min_rows=384,
        union_min_rows_per_head=64,
        wide_tiles_per_cu=1.0,
        max_packed_kv_bytes=64 * 1024 * 1024,
        pack_tokens_per_row=32,
    ),
}
OVERRIDE = "AITER_QSA_PLATFORM"


@functools.cache
def platform() -> Platform:
    gfx, cus = get_gfx_runtime(), get_cu_num()
    keys = [key for key in PLATFORMS if key[0] == gfx]
    if not keys:
        raise ValueError(f"QSA sparse attention has no calibration for {gfx}")
    key = min(keys, key=lambda key: abs(key[1] - cus))
    if key[1] != cus:
        logging.getLogger("aiter").warning(
            "QSA sparse attention is tuned only for MI308X (gfx942, 80 CUs); this %s has "
            "%d CUs and reuses the %d-CU MI308X settings: results are unaffected, "
            "performance may not be optimal",
            gfx,
            cus,
            key[1],
        )
    entry = PLATFORMS[key]
    text = os.environ.get(OVERRIDE, "").strip()
    if not text:
        return entry
    names = {field.name for field in dataclasses.fields(Platform)}
    values = {}
    for item in text.split(","):
        name, _, value = (part.strip() for part in item.partition("="))
        if name not in names or not value:
            raise ValueError(
                f"{OVERRIDE}: expected name=value with name in {sorted(names)}"
            )
        values[name] = type(getattr(entry, name))(value)
    entry = dataclasses.replace(entry, **values)
    values = dataclasses.astuple(entry)
    if not all(math.isfinite(v) and v >= 0 for v in values) or entry.route_margin >= 1:
        raise ValueError(f"{OVERRIDE}: values must be finite, >= 0, route_margin < 1")
    return entry


# ---- Raw buffers, pipeline barriers ----


def _min(a, b):
    return (a < b).select(a, b)


def _join(a, b):
    a, b = fx.Vector(a), fx.Vector(b)
    return fx.Vector.from_elements(
        [a[i] for i in range(a.numel)] + [b[i] for i in range(b.numel)], fx.Float32
    )


def _exp(value):
    """2**value by one bare v_exp_f32."""
    # afn lowers to llvm.amdgcn.exp2; plain math.exp2 adds denormal range reduction.
    return fx.math.exp2(fx.Float32(value), fastmath=fx.FastMathFlags.afn)


def _mfma(a, b, c):
    """One 16x16x16 BF16 MFMA on register vectors: a/b hold 4 BF16 bit patterns, c 4 FP32."""
    frag_a = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_b = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    frag_a.store(fx.Vector(a).bitcast(fx.BFloat16))
    frag_b.store(fx.Vector(b).bitcast(fx.BFloat16))
    frag_c.store(fx.Vector(c))
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _maximum(a, b):
    # nnan drops the canonicalizing maxes that quiet signaling NaNs.
    return fx.Float32(fx.maxnumf(fx.Float32(a), fx.Float32(b), fastmath="nnan"))


# Lowest finite float32: floors an empty row's -inf max without clipping any finite logit.
_MAX_FLOOR = -3.4028234663852886e38


def _pack_bf16(values):
    # gfx942 has no native packed FP32 -> BF16 instruction; software RNE.
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    # One v_perm_b32 joins both high halves; element i lands in the low half.
    words = [
        fx.Uint32(prim.perm_b32(rounded[i + 1], rounded[i], 0x07060302))
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    prim.sched_barrier(0)
    prim.s_barrier()
    prim.sched_barrier(0)


def _schedule(pairs, count, group, exp=False):
    for _ in range(pairs):
        prim.sched_group_barrier(prim.MFMA, 1, group)
        prim.sched_group_barrier(prim.TRANS if exp else prim.VALU, count, group)


def _schedule_exps(count, group):
    # Per value: subtraction, MFMA, exp, MFMA (the fastest order measured).
    for _ in range(count):
        for mask in (prim.VALU, prim.MFMA, prim.TRANS, prim.MFMA):
            prim.sched_group_barrier(mask, 1, group)


def _buffer(tensor, size_bytes):
    """Byte-addressed buffer pointer over tensor; loads at or past size_bytes return zero."""
    return fx.rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), num_records_bytes=size_bytes
    )


def _buffer_at(address, size_bytes):
    """Byte-addressed buffer pointer at a 16-byte aligned 64-bit global address; loads at or
    past size_bytes return zero."""
    pointer = fx.inttoptr(
        fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global, 16),
        fx.Int64(address),
    )
    return fx.rocdl.make_buffer_ptr(pointer, num_records_bytes=size_bytes)


def _buffer_words(buffer, offset, words=4):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(32 * words), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset):
    words = fx.Vector(words)
    data = fx.make_rmem_tensor(4 * words.numel, fx.Int8)
    data.store(words.bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(32 * words.numel), fx.Int8),
        data,
        fx.make_view(buffer + offset, fx.make_layout(4 * words.numel, 1)),
    )


def _write_address(address, words):
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16)
    lds = fx.make_view(fx.inttoptr(ptr_type, fx.Int32(address)), fx.make_layout(4, 1))
    data = fx.make_rmem_tensor(4, fx.Int32)
    data.store(fx.Vector(words).bitcast(fx.Int32))
    fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), data, fx.Tensor(lds))


def _lds_pointer(address, dtype):
    ptr_type = fx.PointerType.get(
        dtype.ir_type, fx.AddressSpace.Shared, dtype.width // 8
    )
    return fx.inttoptr(ptr_type, fx.Int32(address))


def _lds_load(address, dtype=fx.Int32):
    """One value at an LDS byte address."""
    return fx.generic_load(_lds_pointer(address, dtype))


def _lds_store(address, value, dtype=fx.Int32):
    fx.generic_store(_lds_pointer(address, dtype), dtype(value))


# ---- QK, PV and the lazy rescale shared by the union and direct kernels ----


@flyc.jit
def _rescale(values, ballot, factor):
    """Lazy online-softmax rescale: multiply values (O accumulators, a row sum) by
    factor() when any lane's ballot bit is set; factor is only evaluated there."""
    values = tuple(values)
    if ballot != fx.Int64(0):
        correction = factor()
        values = tuple(value * correction for value in values)
    return values


def _read_q_g2r(resource, offset, lane):
    """The lane's 32 Q words: the B operands of a 16x16x16 MFMA over D = 256, four BF16
    per 16-dim step, step s in words 2s and 2s + 1."""
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _qk(q, k0, k1):
    """Scores S = K Q^T of 32 keys over D = 256: one 16x16x16 BF16 MFMA per 16-key tile
    and 16-dim step, steps in order (the same dim order in every kernel).

    q holds the B operands (see _read_q_g2r); k0/k1 the A operands of keys 0-15 and
    16-31, four BF16 per step, step s in words 2s and 2s + 1. Returns the 8 FP32 scores:
    keys 0-15's accumulator, then keys 16-31's. The tiled MMA's eight-wave layout only
    shapes the per-lane fragments, which come from registers; it is the form the union
    pipeline's MFMA/VALU interleave was tuned with (per-MFMA calls schedule differently).
    """
    mma = fx.make_tiled_mma(
        fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
        fx.make_layout((1, 8, 1), (1, 1, 0)),
    )
    b = mma.make_fragment_B(
        fx.make_rmem_tensor(fx.make_layout((128, 256), (1, 128)), fx.BFloat16)
    )
    b.store(fx.Vector(q).bitcast(fx.BFloat16))
    k0, k1 = fx.Vector(k0), fx.Vector(k1)
    a = fx.make_rmem_tensor(fx.make_layout((4, 16, 2), (1, 4, 64)), fx.BFloat16)
    a.store(
        fx.Vector.from_elements(
            [k0[i] for i in range(32)] + [k1[i] for i in range(32)], fx.Int32
        ).bitcast(fx.BFloat16)
    )
    a = fx.make_view(fx.get_iter(a), fx.make_layout((4, 2, 16), (1, 64, 4)))
    acc = mma.make_fragment_C(
        fx.make_rmem_tensor(fx.make_layout((32, 128), (1, 32)), fx.Float32)
    )
    acc.fill(0.0)
    fx.gemm(mma, acc, a, b, acc, traversal_order="kmn")
    return acc.load()


def _pv(p, operand, acc):
    """O += P V over one 16-key step: one 16x16x16 BF16 MFMA per 16-dim tile of O.

    p holds the step's 4 BF16 probabilities (as Int16), operand(n) the A operand of
    tile n (4 BF16 of V as Int16), acc the tiles' 4 FP32 accumulators, updated in order.
    """
    return [_mfma(operand(n), p, acc[n]) for n in range(len(acc))]


def _v_pair(values, row, n):
    """The A operand words of O tile n: V dim n of rows row .. row + 3 as loaded (four
    words, 8 BF16 dims, per row and lane), two rows per word by v_perm_b32."""
    selector = 0x03020706 if n % 2 else 0x01000504
    return [
        fx.Int32(
            prim.perm_b32(
                values[(row + r) * 4 + n // 2],
                values[(row + r + 1) * 4 + n // 2],
                selector,
            )
        )
        for r in (0, 2)
    ]


def _tiles(values):
    """FP32 values as MFMA accumulator tiles of 4."""
    values = fx.Vector(values)
    return [
        fx.Vector.from_elements([values[n * 4 + i] for i in range(4)], fx.Float32)
        for n in range(values.numel // 4)
    ]


def _untile(tiles):
    return fx.Vector.from_elements(
        [tile[i] for tile in tiles for i in range(4)], fx.Float32
    )


# ---- D256 BM128/BN64 eight-wave pipeline: MFMA fragments, softmax, LDS ----


def _sum(values):
    partials = [values[i] + values[i + 1] for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        partials = [partials[2 * i] + partials[2 * i + 1] for i in range(width)]
    return partials[0]


def _max(values):
    # Keep MFMA -> max dependencies visible to LLVM: an opaque max tree failed
    # repeated bit-exact checks, and maxnum avoids maximumf's NaN expansion.
    parts = [fx.maxnumf(values[i], values[i + 1]) for i in range(0, 16, 2)]
    for width in (4, 2, 1):
        parts = [fx.maxnumf(parts[2 * i], parts[2 * i + 1]) for i in range(width)]
    return parts[0]


def _cross(value, addresses, queue):
    bits = fx.Float32(value).bitcast(fx.Int32)
    result = tuple(
        queue.bpermute(address, bits).bitcast(fx.Float32) for address in addresses
    )
    # LLVM would sink the ds_bpermutes below the LDS reads that the queue counts after them.
    prim.sched_barrier(0)
    return result


def _exps(values, maximum):
    # values are the rounded S * scale the max came from: an FMA overshoots by ulp / 2.
    return fx.Vector.from_elements(
        [_exp(values[i] - maximum) for i in range(16)], fx.Float32
    )


def _k_phase(token):
    return (token & 3) | ((token >> 3) & 1) * 4


def _k_lds_address(token, dimension):
    return token * 512 + ((dimension // 2) ^ (_k_phase(token) * 4)) * 4


def _v_lds_address(token, dimension):
    return token * 512 + dimension * 2


def _copy_coordinates(wave, packet, page=0):
    token = (wave & 3) * 16 + packet if page == 4 else (wave & 3) + packet * 4
    return token, (wave >> 2) * 128


def _k_reads(addresses, half, lower_first=False):
    """(slot, address, immediate) of one K half's 16 LDS reads, in issue order."""
    order = (
        [(n, k + high * 4) for high in range(2) for n in range(2) for k in range(4)]
        if lower_first
        else [(n, k) for n in range(2) for k in range(8)]
    )
    return [
        (n * 8 + k, addresses[n * 2 + k % 2], half * 16384 + (k // 2) * 128)
        for n, k in order
    ]


def _v_reads(address, half):
    """(slot, address, immediate) of one V half's 16 LDS reads, in issue order."""
    return [
        (block * 8 + row, address, half * 256 + block * 16384 + row * 512)
        for block in range(2)
        for row in range(8)
    ]


def _lds_words(parts):
    return fx.Vector.from_elements(
        [part[i] for part in parts for i in range(4)], fx.Int32
    )


def _read_s2r(plan, queue):
    """Issue a (storage, reads) plan; ready(n) waits until its first n reads have landed."""
    storage, reads = plan
    first = queue.issued
    parts = [None] * len(reads)
    for slot, address, immediate in reads:
        parts[slot] = queue.load(storage, address, immediate)
    return _lds_words(parts), queue.ready(first, queue.issued - first)


def _k_operands(words):
    """The _qk A operands of keys 0-15 and 16-31 from one K half's 64 LDS words."""
    # MFMA-only uses leave the loads AV-class; the scheduler then reverts the QK interleave.
    prim.keep_vgpr(words)
    prim.sched_barrier(0)
    words = fx.Vector(words)
    return tuple(
        fx.Vector.from_elements([words[n * 32 + i] for i in range(32)], fx.Int32)
        for n in range(2)
    )


def _v_operands(values, step, ready=None):
    raw = fx.Vector(values)
    row = step * 4
    if ready is not None:
        ready(row + 4)
        prim.keep_vgpr(
            fx.Vector.from_elements(
                [raw[i] for i in range(row * 4, row * 4 + 16)], fx.Int32
            )
        )
        prim.sched_barrier(0)
    words = []
    for n in range(8):
        words.extend(_v_pair(raw, row, n))
    return fx.Vector.from_elements(words, fx.Int32)


def _pv_half(
    probabilities,
    values,
    output,
    ready=None,
    prepared=None,
    center=None,
    sum_max=None,
    steps=4,
):
    """O += P V over one V half of a key tile (its 4 key steps, 8 O tiles), by _pv per
    step; center and sum_max fold softmax VALU work into its MFMAs.

    sum_max = (S, lane sum, other sums, l) also returns l + row sum and the lane max of S;
    center = (S, lane max, other maxima, scale, m) returns S * scale, m' and the ballot.
    """
    p = fx.Vector(probabilities)
    acc = _tiles(output)
    prepared_steps = 0 if prepared is None else prepared.numel // 16
    operands = (
        _v_operands(values, 0, ready)
        if prepared is None
        else fx.Vector.from_elements([prepared[i] for i in range(16)], fx.Int32)
    )
    scaled = []
    prim.sched_barrier(0)
    for step in range(steps):
        if step + 1 < prepared_steps:
            following = fx.Vector.from_elements(
                [prepared[(step + 1) * 16 + i] for i in range(16)], fx.Int32
            )
        else:
            following = (
                _v_operands(values, step + 1, ready) if step + 1 < steps else operands
            )
        b = fx.Vector.from_elements([p[step * 4 + i] for i in range(4)], fx.BFloat16)
        acc = _pv(
            b.bitcast(fx.Int16),
            lambda n, words=operands: fx.Vector.from_elements(
                [words[n * 2 + i] for i in range(2)], fx.Int32
            ).bitcast(fx.Int16),
            acc,
        )
        if center is not None:
            current, candidate, maxima, scale, maximum = center
            if step == 0:
                candidate = (
                    _maximum(
                        _maximum(candidate, maxima[0]), _maximum(maxima[1], maxima[2])
                    )
                    * scale
                )
                predicate = candidate > maximum + 7.0
                ballot = prim.ballot(predicate)
                new_max = predicate.select(candidate + 1.0, maximum)
            for i in range(step * 4, step * 4 + 4):
                scaled.append(fx.Float32(current[i]) * scale)
        if sum_max is not None and step == 3:
            current, total, sums, row_sum = sum_max
            row_sum = row_sum + ((total + sums[0]) + (sums[1] + sums[2]))
            candidate = _max(current)
        if step + 1 < steps and step + 1 >= prepared_steps:
            _schedule(8, 3 if center is not None else 2, 20 + step)
        elif sum_max is not None:
            _schedule(8, 3, 23)
        prim.sched_barrier(0)
        operands = following
    result = _untile(acc)
    if center is not None:
        # Pinned: unpinned, LLVM sinks the scaling into the next memory stage.
        scaled = prim.pin(fx.Vector.from_elements(scaled, fx.Float32))
        return result, scaled, new_max, ballot
    if sum_max is not None:
        return result, row_sum, candidate
    return result
