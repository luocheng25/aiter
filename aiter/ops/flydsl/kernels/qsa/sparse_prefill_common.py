# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Settings and gfx942 D256 helpers shared by the QSA sparse attention kernels.

The first part holds the per-platform calibration and the private resource
policy. The platform entry is picked by gfx model and CU count (nearest
calibrated CU count of the same model) and read once per process; FP8 K/V
replace its route terms (FP8_ROUTES); AITER_QSA_PLATFORM overrides fields of the
picked entry, e.g. ``AITER_QSA_PLATFORM="union_step=3.0,max_packed_kv_bytes=0"``.

The second part holds raw-buffer operations and pipeline barriers. The third
part holds the lazy softmax rescale (every BF16 kernel) and the 16x16x16 BF16 QK
and PV MFMA steps of the BF16 direct kernels (the BF16 union pipeline runs its own
32x32x8 steps). The fourth part holds what both FP8 K/V kernels share: the FP8
MFMA, the lazy max of their softmax, the BF16 output rounding and the FP8
quantization of Q. The fifth part holds the persistent direct launches' pacing and
work queue. Everything below FlyDSL's stable API is in sparse_prefill_primitives
(prim).
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
# Arena views start on 4 KiB boundaries: a packed K/V block (1 KiB FP8, 2 KiB BF16) of a view
# that starts mid-page straddles two, and the direct kernels' gathers of such blocks run slower
# (random selections over a 128k-token context, H6: direct +0.3% behind BQ21 union scratch that
# is no multiple of 4 KiB).
SCRATCH_ALIGN = 4096
UNION_SCRATCH_LIMIT = 4 << 30  # union scratch bytes; past it a layout runs direct only
MAX_ROW_PACKED_KV_BYTES = 1 << 30  # PK+PV bytes of the pack_tokens_per_row rule


@dataclasses.dataclass(frozen=True)
class Platform:
    """Calibrated per GPU; the route-model terms are microseconds."""

    union_fixed: float  # fixed cost of the union phase
    union_step: float  # per N64 step of the slowest union CTA
    row_step: float  # per BN32 step of the longest direct row
    route_margin: float  # each route step to fewer union tiles must save this share
    direct_fixed: float  # fixed cost of the direct phase
    pack_block: float  # per packed KV block and KV head
    direct_cu_step: float  # per BN32 direct step and KV head, times the CU count
    union_ratio: float  # union while padded union steps <= ratio x direct steps
    union_min_rows: int  # below max(rows, rows_per_head x H/HK) direct-only wins
    union_min_rows_per_head: int
    wide_tiles_per_cu: float  # H6/H3 union rows: BQ21/BQ42 from this many tiles/CU
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
# FP8 K/V (sparse_prefill_*_fp8, by K/V scale layout): route terms fitted on the FP8 kernels'
# phase timings over synthetic and SGLang-captured selections: union_step on the union
# kernel's times of 23 plans per scale layout (union_fixed kept; median error of the
# throughput-bound ones -0.4% / +0.4%, split plans up to 57% high and plans of two or three
# tasks per CTA 19% low), the direct terms (direct_cu_step, row_step) on the direct kernel's
# times of 176 plans (median error 0.4%); the per-tile ratio is the sweep's best with
# them. Union and minimum-row thresholds otherwise follow the BF16 entry.
FP8_ROUTES = {
    ("gfx942", 80): {
        "tensor": {
            "union_fixed": 5.7,
            "union_step": 1.571,
            "row_step": 1.05,
            "direct_fixed": 11.9,
            "pack_block": 0.00184,
            "direct_cu_step": 0.145,
            "union_ratio": 1.7,
        },
        "token": {
            "union_fixed": 5.14,
            "union_step": 1.771,
            "row_step": 1.12,
            "direct_fixed": 11.87,
            "pack_block": 0.00213,
            "direct_cu_step": 0.152,
            "union_ratio": 1.7,
        },
    },
}
OVERRIDE = "AITER_QSA_PLATFORM"


@functools.cache
def _calibrated() -> tuple:
    """The PLATFORMS key of this GPU (nearest calibrated CU count of its gfx model); warns
    once per process for a CU count without its own entry."""
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
    return key


@functools.cache
def platform(fp8=None) -> Platform:
    """The calibration of this GPU: BF16 K/V, or FP8 K/V with "tensor" / "token" scales."""
    key = _calibrated()
    entry = PLATFORMS[key]
    if fp8 is not None:
        entry = dataclasses.replace(entry, **FP8_ROUTES[key][fp8])
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


def _buffer_words(buffer, offset, words=4, cache=0):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(32 * words, cache), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset, cache=0):
    words = fx.Vector(words)
    data = fx.make_rmem_tensor(4 * words.numel, fx.Int8)
    data.store(words.bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy(32 * words.numel, cache), fx.Int8),
        data,
        fx.make_view(buffer + offset, fx.make_layout(4 * words.numel, 1)),
    )


# gfx942 buffer cache policy (aux bits: sc0 = 1, nt = 2, sc1 = 16). sc1 alone is device scope:
# loads read past the CU's L1. sc0 | sc1 is system scope, so stores write through and loads read
# past the CU's L1 and this XCD's L2 (coherent between CTAs of one launch on any XCD).
SC_DEVICE = 16
SC_SYSTEM = 17


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


# ---- The lazy rescale; the BF16 direct kernels' QK and PV ----


@flyc.jit
def _lazy(ballot, values, update):
    """values replaced by update(*values) when any lane's ballot bit is set: the rare branch
    of a lazy online-softmax rescale, so update is only evaluated there."""
    values = tuple(values)
    if ballot != fx.Int64(0):
        values = tuple(update(*values))
    return values


def _rescale(values, ballot, factor):
    """Lazy online-softmax rescale: multiply values (O accumulators, a row sum) by
    factor() when any lane's ballot bit is set; factor is only evaluated there."""

    def update(*values):
        correction = factor()
        return tuple(value * correction for value in values)

    return _lazy(ballot, values, update)


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
    shapes the per-lane fragments, which come from registers (per-MFMA calls schedule
    differently).
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


# ---- FP8 (E4M3FNUZ) K/V: MFMA, lazy max, output rounding and Q quantization ----

FP8_MAX = 240.0  # largest finite E4M3FNUZ
# Lazy max of the FP8 kernels: m moves only when some S'' exceeds LAZY (P' <= 2**7.5 fits
# E4M3FNUZ), and then puts the row max at S'' = P_BIAS: P' of a key down to 2**-(11 + P_BIAS)
# of the max stays nonzero (with P' <= 1, attention sinks over long flat tails lost the tail,
# a quarter of the output).
LAZY = 7.5
P_BIAS = 4.0
# A V scale enters as log2 and reciprocal; the clamp keeps both finite (NaN maps to the floor).
SCALE_MIN, SCALE_MAX = 2.0**-60, 2.0**60
# Q quantization: two rows per 16-lane group (16 dims per lane and row, every row load in
# flight at once), so QROWS rows per 256-thread CTA.
QR = 2
QROWS = 16 * QR


def _as_f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _as_i32(value):
    return fx.Float32(value).bitcast(fx.Int32)


def _fma(a, b, c):
    return fx.Float32(fx.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def _mfma_fp8(a, b, acc):
    """One E4M3FNUZ MFMA on register words: a and b hold 8 FP8 values (two words) per lane,
    acc 16 FP32 (32x32x16) or 4 (16x16x32)."""
    acc = fx.Vector(acc)
    m = 32 if acc.numel == 16 else 16
    frag_a = fx.make_rmem_tensor(8, fx.Float8E4M3FNUZ)
    frag_b = fx.make_rmem_tensor(8, fx.Float8E4M3FNUZ)
    frag_c = fx.make_rmem_tensor(acc.numel, fx.Float32)
    frag_a.store(fx.Vector.from_elements(list(a), fx.Int32).bitcast(fx.Float8E4M3FNUZ))
    frag_b.store(fx.Vector.from_elements(list(b), fx.Int32).bitcast(fx.Float8E4M3FNUZ))
    frag_c.store(acc)
    mma = fx.make_mma_atom(fx.rocdl.MFMA(m, m, 512 // m, fx.Float8E4M3FNUZ))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _fp8_word(values):
    """Four FP32 -> one word of E4M3FNUZ (byte k = value k), round to nearest even."""
    return prim.cvt_pk_fp8(
        values[2], values[3], prim.cvt_pk_fp8(values[0], values[1]), True
    )


def _v_scale_terms(sv):
    """(log2(v_scale), 1 / v_scale) of a clamped V scale."""
    sv = fx.minnumf(
        fx.maxnumf(fx.Float32(sv), fx.Float32(SCALE_MIN)), fx.Float32(SCALE_MAX)
    )
    return prim.log2(sv), prim.rcp(sv)


def _lazy_max(o, row_sum, scores, m_run, ballot, tile_max):
    """The FP8 kernels' lazy max (S'' = S * c - m in log2 units): when any lane's ballot bit
    is set, m moves by max(tile_max() - P_BIAS, 0), and a fresh row (m = FLOOR, centered at
    0) takes tile_max() - P_BIAS (FLOOR while it has no key: its S'' stay -inf); O (a list
    of accumulator vectors) and the row sum l scale by 2**-delta and S'' shifts by -delta.
    Returns (o, row_sum, scores, m_run)."""

    def update(*values):
        *o, row_sum, scores, m_run = values
        tile = tile_max()
        fresh = m_run == fx.Float32(_MAX_FLOOR)
        delta = _maximum(
            fx.Float32(tile) - P_BIAS,
            fresh.select(fx.Float32(_MAX_FLOOR), fx.Float32(0.0)),
        )
        corr = fresh.select(fx.Float32(0.0), _exp(fx.Float32(0.0) - delta))
        return (
            *(fx.Vector(x) * corr for x in o),
            fx.Float32(row_sum) * corr,
            fx.Vector(scores) - delta,
            fresh.select(fx.Float32(0.0), m_run) + delta,
        )

    *o, row_sum, scores, m_run = _lazy(ballot, (*o, row_sum, scores, m_run), update)
    return o, row_sum, scores, m_run


def _scaled_bf16(values, inv):
    """Two BF16 words of four FP32 values times inv: FP32 multiplies, then round half away
    from zero (+0x8000) and the high halves joined by v_perm."""
    words = []
    for i in (0, 2):
        lo = _as_i32(fx.Float32(values[i]) * inv) + fx.Int32(0x8000)
        hi = _as_i32(fx.Float32(values[i + 1]) * inv) + fx.Int32(0x8000)
        words.append(fx.Int32(prim.perm_b32(hi, lo, 0x07060302)))
    return words


def _quantize_part(raw):
    """16 BF16 dims of a row (16 lanes per row): FP8 words and the row scale amax / 240."""
    values = []
    for vec in raw:
        for j in range(4):
            word = fx.Int32(vec[j])
            values += [_as_f32(word << 16), _as_f32(word & fx.Int32(-65536))]
    level = [fx.absf(v) for v in values]
    while len(level) > 1:
        level = [_maximum(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    amax = level[0]
    for shift in (1, 2, 4, 8):
        amax = _maximum(amax, fx.gpu.shuffle_xor(amax, shift, 64))
    inverse = (amax > fx.Float32(0.0)).select(prim.rcp(amax) * FP8_MAX, fx.Float32(0.0))
    words = [
        _fp8_word([values[4 * k + r] * inverse for r in range(4)]) for k in range(4)
    ]
    return words, amax * (1.0 / FP8_MAX)


@flyc.jit
def quantize_rows(Q, Q8, QS, ROWS, cta):
    """Rows cta * QROWS .. + QROWS - 1 of Q (rows of 256 BF16) for a 256-thread CTA: row
    r * 16 + (thread >> 4) of the CTA's rows per step r; FP8 rows (E4M3FNUZ, RNE) and their
    scales amax / 240 (0 for a zero row). Run by the FP8 pack kernel's CTAs past its blocks
    and by the FP8 union module's quantize_q."""
    tid = fx.Int32(fx.thread_idx.x)
    group, part = tid >> 4, tid & 15
    extent = fx.Uint32(ROWS) * 512
    gq = _buffer(Q, extent)
    rows = [cta * QROWS + r * 16 + group for r in range(QR)]
    raws = []
    for row in rows:
        offset = (row < ROWS).select(row * 512 + part * 32, fx.Int32(extent))
        raws.append([_buffer_words(gq, offset + 16 * i, 4) for i in range(2)])
    g8 = _buffer(Q8, fx.Uint32(ROWS) * 256)
    scales = []
    for row, raw in zip(rows, raws):
        words, scale = _quantize_part(raw)
        scales.append(scale)
        _buffer_store_words(
            fx.Vector.from_elements(words, fx.Int32),
            g8,
            (row < ROWS).select(row * 256 + part * 16, fx.Int32(ROWS * 256)),
        )
    # Lane `part` of a group stores the scale of its row step `part`.
    scale = scales[QR - 1]
    for r in fx.range_constexpr(QR - 1):
        scale = (part == r).select(scales[r], scale)
    row = cta * QROWS + part * 16 + group
    gs = _buffer(QS, fx.Uint32(ROWS) * 4)
    _buffer_store_words(
        fx.Vector.from_elements([_as_i32(scale)], fx.Int32),
        gs,
        ((part < QR) & (row < ROWS)).select(row * 4, fx.Int32(ROWS * 4)),
    )


# ---- Persistent launches: pacing and the work queue ----
#
# A persistent direct launch runs one WG of one-row waves per CU. Waves w and w ^ 4 share a
# SIMD and the older one issues first; each key step publishes the wave's step count in LDS
# (_lds_relaxed_store) and a wave behind its sibling raises its priority (_behind_prio).
# Work items come from the launch's queue: word 0 counts the items taken, word 64 the waves
# out (the last one zeroes both, so graph replays start from zero); words 1-63 are free for
# the kernel. Lane 0 updates them with buffer atomics whose other lanes are out of range:
# one request, no divergent branch.
QUEUE_WORDS = 128
_QUEUE_OUT = 256  # byte offset of the waves-out word


def _lds_relaxed_load(address):
    """An LDS word another wave stores (a relaxed atomic load: not hoisted out of loops)."""
    return fx.generic_load(
        _lds_pointer(address, fx.Int32),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="workgroup",
    )


def _lds_relaxed_store(address, value):
    fx.generic_store(
        _lds_pointer(address, fx.Int32),
        fx.Int32(value),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="workgroup",
    )


@flyc.jit
def _behind_prio(count, sibling, margin, prio):
    """s_setprio(prio) while the wave is more than `margin` steps behind its sibling, else
    0 (a wave-uniform condition)."""
    if count + margin < sibling:
        prim.s_setprio(prio)
    else:
        prim.s_setprio(0)


def _queue_take(queue, lane):
    """The queue's next item index (word 0), per lane: read it with prim.readlane(..., 0).
    queue: the buffer of the launch's QUEUE_WORDS words."""
    beyond = fx.Int32(QUEUE_WORDS * 4)
    return prim.buffer_atomic(
        "add", queue, (lane == 0).select(fx.Int32(0), beyond), fx.Int32(1)
    )


def _queue_exit(queue, lane, waves):
    """Count the wave out; the last of `waves` zeroes the taken and out words (every take
    of the launch is done once all its waves are out)."""
    beyond = fx.Int32(QUEUE_WORDS * 4)
    out = prim.buffer_atomic(
        "add", queue, (lane == 0).select(fx.Int32(_QUEUE_OUT), beyond), fx.Int32(1)
    )
    last = (prim.readlane(out, 0) == waves - 1) & (lane == 0)
    prim.buffer_atomic("swap", queue, last.select(fx.Int32(0), beyond), fx.Int32(0))
    prim.buffer_atomic(
        "swap", queue, last.select(fx.Int32(_QUEUE_OUT), beyond), fx.Int32(0)
    )
