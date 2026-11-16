# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA union attention over FP8 (E4M3FNUZ) K/V with FP8 MFMAs.

Same plans and tasks as sparse_prefill_union: a task is 128 (query, head) rows of a union tile
over its compact block list, with exact per-query membership. Four waves of 32 rows (one wave
per SIMD) run 32x32x16 FP8 MFMAs, S^T = K Q^T and O^T += V^T P; the 128 O values of a lane
stay in registers (where LLVM allocates them).

Numerics. sparse_prefill_common.quantize_rows (in the FP8 pack launch, or quantize_q) stores Q
per (query, head) row as E4M3FNUZ with scale = row amax / 240. K/V scales are per token and
KV head (FP32 [N, HK]; with a slot table [P, HK]) or one per tensor ([1]). Softmax runs in
log2 units with a lazy max m (sparse_prefill_common._lazy_max): S'' = S * c - m with
c = q_scale * SCALE_LOG2 (times k_scale per tensor, per key otherwise), SCALE_LOG2 the
runtime FP32 softmax_scale * log2(e). Per token, log2(v_scale) joins S'', so the FP8
P' = exp2(S'') carries the V scale and stays below 2**LAZY while l accumulates P' / v_scale;
per tensor the output takes v_scale. m moves only when some row's S'' exceeds LAZY or a row
sees its first key; then the row max lands at S'' = P_BIAS (P' = 2**P_BIAS, so small weights
stay above the FP8 underflow) and O, l and S'' follow.

Phase t (64-key tile t; K and V^T double-buffered in 64 KiB LDS):
  Part 1  QK(t) [K slot t&1]; after QK MFMA FIX_AT tile t-1 is finished under the issued
          MFMAs (lazy max, rescale), then P'(t-1) = exp2(S''(t-1)) and its row sums;
          K(t+1) -> K slot (t+1)&1; V(t), ids(t+2) and mask(t) loads; V block ids of t+1
          by v_readlane, K and scale ids by ds_bpermute. After QK MFMA BAR_AT: lgkmcnt(0)
          + barrier, then the first V reads and the FP8 P' of PV(t-1) under the last QK
          MFMAs.
  Part 2  PV(t-1) [V slot (t-1)&1] || P' packs, S''(t) (scales, centering); V(t) -> V
          slot t&1 interleaved with S''(t)'s membership masks (every tile, branch free:
          whole tiles keep every bit) and its masked row max; K(t+2) and scale(t+1) loads;
          the first K reads of QK(t+1).

V^T comes from the FP8 pack of the direct plan (sparse_prefill_direct_fp8: four-token
blocks, V^T per block), so it needs no transpose: a wave loads its four blocks of a tile
(16 bytes per lane: four dims of a block's four tokens) and writes one 16-byte chunk per dim.

The grid is persistent: a CTA prefetches its next task's ids, masks, Q, K(0..2), V(0) and
scales under the current task's epilogue and output, and reads the task order with scalar
loads, which do not wait behind the output stores.

Split plans (fewer tasks than CUs, sparse_prefill_prepare.SPLIT_PARTS): K3 gives a long task two
order entries, its first ceil(steps / 2) and remaining N64 steps. Each part stores its O^T
fragments, m and l (system-scope stores, any XCD may merge) and counts on the task's slot; the
part counting last merges both records, part 0 first (O = sum O_p * 2^(m_p - m), l likewise),
and writes the rows like an unsplit task.

Core/residual plans (core_ratio > 0, RESIDUAL): every row's log-sum-exp m + log2(l) of S * c
(log2 units) goes to LSE, which the FP8 direct kernel merges its residual blocks with.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from . import sparse_prefill_primitives as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    P_BIAS,
    QROWS,
    SC_SYSTEM,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fma,
    _lazy_max,
    _maximum,
    _mfma_fp8,
    _min,
    _scaled_bf16,
    _stage_end,
    _v_scale_terms,
    quantize_rows,
)
from .sparse_prefill_prepare import PARTIAL_FLOATS, SPLIT_PARTS, SPLIT_SLOTS

__all__ = ["launcher", "quantize_q", "run"]

BM, BN, D, THREADS = 128, 64, 256, 256
SLOT = BN * D  # one FP8 K or V^T tile
PACK_BLOCK = 4 * D  # one four-token block of one KV head in the FP8 pack (PV)
LDS_BYTES = 4 * SLOT  # K slots 0/1, V^T slots 0/1
V0 = 2 * SLOT
NEG_INF = -8388608  # FP32 -inf bits
# Pipeline placement (MI308X): K/V LDS reads issued KPRE/VPRE MFMAs ahead, the Part 1 barrier
# after QK MFMA BAR_AT, VALU steps per Part 1 / Part 2 MFMA (P1 / P2, TAIL after the barrier),
# MFMAs per K(t+1) write and steps per V(t) write.
KPRE, VPRE, BAR_AT, P1, TAIL, P2, KW_GAP, SPREAD = 4, 4, 27, 1, 3, 4, 3, 4
FIX_AT = 1  # Part 1 finishes tile t-1 after this QK MFMA
# A split part's record (sparse_prefill_prepare.PARTIAL_FLOATS): O^T fragments in MFMA order
# (wave, block mb, register quad k: 16 bytes per lane), then m and the lane's l per wave.
PARTIAL_BYTES = PARTIAL_FLOATS * 4
PARTIAL_ML = 128 * D * 4
K_ORDER = tuple((b, c) for c in range(8) for b in range(2))


class _Env:
    """Trace-time bundle of one task's per-lane values; the helpers below are plain Python."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _drain(gen):
    for _ in gen:
        pass


def _pin():
    prim.sched_barrier(0)


def _step(gen):
    next(gen, None)
    _pin()


def _bperm(value, lane_index):
    return prim.bpermute(fx.Int32(lane_index * 4), fx.Int32(value))


def _write_lds(storage, offset, words):
    """ds_write_b64 / b128 of 2 or 4 words at a byte offset of storage."""
    count = len(words)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4 * count),
        fx.get_iter(storage) + offset,
    )
    data = fx.make_rmem_tensor(count, fx.Int32)
    data.store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32))
    atom = fx.UniversalCopy64b() if count == 2 else fx.UniversalCopy128b()
    fx.copy(
        fx.make_copy_atom(atom, fx.Int32),
        data,
        fx.make_view(ptr, fx.make_layout(count, 1)),
    )


# ---- LDS layouts ----
# K tile: key-major rows of 256 bytes, 16-byte chunks XOR-swizzled by the key.
# V^T tile: dim-major rows of 64 key bytes; rows pair-swapped and chunks swizzled so the b128
# writes (lanes over dims) and the b128 operand reads (lanes over dims) are conflict free.
# Chunk 2h + q of a dim holds the tokens of blocks 8q + h + 0, 2, 4, 6 (PV K-steps 2q, 2q + 1).


def _swz_k(key):
    return (key & 3) | ((key >> 2) & 4)


def _k_offset(key, chunk):
    return key * 256 + ((chunk ^ _swz_k(key)) << 4)


def _vt_chunk(d, chunk):
    row = d ^ ((d >> 3) & 1)
    sig = (((d >> 1) ^ (d >> 4)) & 1) | ((((d >> 2) ^ (d >> 6)) & 1) << 1)
    return row * 64 + ((chunk ^ sig) << 4)


def _k_read(env, slot, b, c):
    """K operand words of keys 32b + lane, dims 32c + 16h .. + 15."""
    return prim.load_lds(env.lds, env.k_addr[c], slot * SLOT + 8192 * b)


def _v_read(env, slot, mb, q):
    """V^T operand of dims 32mb + lane: words of PV K-steps 2q, 2q + 1 in order."""
    if isinstance(slot, int):
        return prim.load_lds(
            env.lds, env.v_addr[(mb >> 1) & 1][q], V0 + slot * SLOT + 2048 * mb
        )
    return prim.load_lds(
        env.lds, env.v_addr[(mb >> 1) & 1][q] + slot * SLOT, V0 + 2048 * mb
    )


# ---- Global loads ----


def _load_idl(env, t):
    """Block of slot (lane & 15) of key tile t (clamped to the last); slots past count read 0."""
    return _buffer_words(
        env.gblocks, (_min(t, env.last) * 16 + (env.lane & 15)) * 4, 1
    )[0]


def _fix_ids(env, idl, t):
    """Slots past the compact list take a block past the K/V extent, so their K, V and scale
    loads read 0 without per-load checks."""
    slot = _min(t, env.last) * 16 + (env.lane & 15)
    return (slot < env.count).select(fx.Int32(idl), env.id_oob)


def _load_mask(env, t):
    """Membership words of the lane's query over key tile t (quarters 0-1, 2-3)."""
    w = _buffer_words(env.gmask, env.mask_lane + _min(t, env.last) * env.mask_step, 2)
    return w[0], w[1]


def _mask_words(env, words):
    """The lane's key-half bits: value (b, i) is bit 16 * ((i >> 2) & 1) + 8b + (i & 3) of
    word i >> 3. Rows outside the task keep every key."""
    shift = env.h * 4
    return [env.valid.select(fx.Int32(w) >> shift, fx.Int32(-1)) for w in words]


def _keep(words, b, i):
    """0 / -1 membership of value (b, i) (signed bit extraction)."""
    position = 16 * ((i >> 2) & 1) + 8 * b + (i & 3)
    return (words[i >> 3] << (31 - position)) >> 31


def _masked(value, keep):
    # Bit select: the score bits or -inf, without a compare and VCC per value.
    bits = _as_i32(value)
    return _as_f32((bits & keep) | ((~keep) & fx.Int32(NEG_INF)))


def _load_lidl(env, t):
    """Request-local block of slot (lane & 15) of key tile t (with a slot table; otherwise
    _load_idl already reads request-local blocks)."""
    return _buffer_words(env.gvids, (_min(t, env.last) * 16 + (env.lane & 15)) * 4, 1)[
        0
    ]


def _vids(env, lidl):
    """The wave's V blocks of a key tile, slots vslot + 0, 2, 4, 6 of its request-local ids
    (lane s holds slot s): v_readlane, so no scalar load shares the LDS counter. Slots past
    the list read block 0, whose keys every row masks."""
    return tuple(
        prim.uniform(prim.readlane(fx.Int32(lidl), env.vslot + 2 * j)) for j in range(4)
    )


def _gen_v_loads(env, vids, out):
    """The wave's four V blocks from the FP8 pack (V^T per block, 16 bytes per lane): dims
    4 * lane .. + 3 of each block's four tokens (tokens past a request's end are 0)."""
    for vid in vids:
        offset = (env.pstart + vid) * env.pv_step + env.v_lane
        out.append(_buffer_words(env.gpv, offset, 4))
        yield


def _gen_k_loads(env, kid, out):
    """The thread's token of block kid: four 16-byte chunks."""
    row = (kid * 4 + env.ktok) * env.tok_bytes
    if env.slotted:
        partial = (kid == env.final) & (env.ktok >= env.tail_len)
        row = partial.select(env.kv_oob, row)
    for i in range(4):
        part = _buffer_words(env.gk, row + env.k_goff[i], 4)
        out.extend(part[j] for j in range(4))
        yield


def _gen_scales(env, sids, out):
    """(sk0, sk1, sv0, sv1) of the lane's scale keys; per tensor: zeros."""
    if not env.per_token:
        out.extend([fx.Float32(0.0)] * 4)
        return
    offsets = []
    for b in range(2):
        offset = (sids[b] * 4 + env.stok) * env.scale_step
        if env.slotted:
            partial = (sids[b] == env.final) & (env.stok >= env.tail_len)
            offset = partial.select(env.scale_oob, offset)
        offsets.append(offset)
    scales = []
    for buffer in (env.gks, env.gvs):
        for b in range(2):
            scales.append(_as_f32(_buffer_words(buffer, offsets[b], 1)[0]))
            yield
    out.extend(scales)


# ---- LDS writes ----


def _gen_k_writes(env, kdata, slot, gap=1):
    for i in range(4):
        _write_lds(
            env.lds, env.k_wbase[i] + slot * SLOT, [kdata[4 * i + j] for j in range(4)]
        )
        for _ in range(gap):
            yield


def _gen_v_writes(env, vdata, slot, spread):
    """V^T rows of the lane's four dims: per dim one 16-byte chunk, the four tokens of each of
    the wave's blocks (word j of the pack loads holds dim 4 * lane + j)."""
    for j in range(4):
        words = [fx.Vector(vdata[b])[j] for b in range(4)]
        _write_lds(env.lds, env.v_wbase[j] + V0 + slot * SLOT, words)
        for _ in range(spread):
            yield


# ---- Softmax pieces ----


def _gen_part1(env, s_prev, rsv, res):
    """P'(t-1) = exp2(S''(t-1)) and the row sum of P' (per token P' / v_scale) in the issue
    slack of the exps, a value's term trailing its exp by two steps (a DPP instruction
    reading the previous VALU's result needs two wait states)."""
    p, sums = [None] * 32, [None, None]
    res["p"] = p
    for i in range(34):
        if i < 32:
            p[i] = _exp(s_prev[i])
        if i > 1:
            k = i - 2
            term = (
                p[k] * prim.row_broadcast(rsv[k // 16], k % 16)
                if env.per_token
                else p[k]
            )
            sums[k & 1] = term if sums[k & 1] is None else sums[k & 1] + term
        yield
    prim.hold(*sums)
    res["sum"] = sums[0] + sums[1]


def _gen_pack(p, s, pw):
    """FP8 words of PV K-step s: values 8s .. 8s + 7 of P'."""
    words = []
    for i in (0, 4):
        values = p[8 * s + i : 8 * s + i + 4]
        lo = prim.cvt_pk_fp8(values[0], values[1])
        yield
        words.append(prim.cvt_pk_fp8(values[2], values[3], lo, True))
        yield
    pw[s] = words


def _interleave(*gens):
    """One step of each generator in turn until all are done."""
    live = list(gens)
    while live:
        for gen in list(live):
            try:
                next(gen)
            except StopIteration:
                live.remove(gen)
                continue
            yield


def _gen_part2(env, ctx):
    """P' K-steps 1-3; S''(t) = S * c - m (speculative, m of t-1) with the scales, its
    membership masks and masked row max; V(t) -> LDS."""
    p, pw = ctx["p"], ctx["pw"]
    c_scale, neg = env.c_scale, ctx["neg"]
    terms = [None, None, None, None]

    def gen_terms():
        if env.per_token:
            for b in range(2):
                terms[b], terms[2 + b] = _v_scale_terms(ctx["sv"][b])
                yield

    def gen_center(lo, hi, vals, part):
        if env.per_token:
            # A DPP instruction reading a VGPR the previous VALU wrote needs two wait states
            # (any VGPR operand): the log2(v_scale) add of a value trails its FMA by one value.
            xs = {}
            for k in range(lo, hi + 1):
                if k < hi:
                    b, i = divmod(k, 16)
                    v = ctx["s_acc"][b][i] * prim.row_broadcast(ctx["sk"][b], i)
                    xs[k] = _fma(v, c_scale, neg)
                    yield
                if k > lo:
                    kk = k - 1
                    b, i = divmod(kk, 16)
                    vals[kk] = xs.pop(kk) + prim.row_broadcast(terms[b], i)
                    yield
            return
        for k in range(lo, hi):
            b, i = divmod(k, 16)
            vals[k] = _fma(ctx["s_acc"][b][i], c_scale, neg)
            yield

    def gen_mask_max(part):
        # Membership masks of every tile (whole tiles keep all bits), then the masked row
        # max: no masked branch at the next phase's fix-up.
        ws = _mask_words(env, ctx["mask"])
        yield
        for k in range(32):
            b, i = divmod(k, 16)
            vals[k] = _masked(vals[k], _keep(ws, b, i))
            yield
            if k % 2 == 1:
                a, c = vals[k - 1], vals[k]
                part.append(
                    _maximum(a, c) if not part else _maximum(_maximum(part.pop(), a), c)
                )
                yield

    vals, part = [None] * 32, []
    yield from gen_terms()
    if env.per_token:
        prim.hold(*terms[2:])
    yield from _gen_pack(p, 1, pw)
    yield from gen_center(0, 16, vals, part)
    yield from _gen_pack(p, 2, pw)
    yield from gen_center(16, 32, vals, part)
    yield from _gen_pack(p, 3, pw)
    yield from _interleave(
        _gen_v_writes(env, ctx["vdata"], ctx["cur"], SPREAD), gen_mask_max(part)
    )
    local = part[0]
    other = fx.gpu.shuffle_xor(local, 32, 64)
    yield
    ctx.update(s_cur=vals, local=local, other=other, rsv=terms[2:])


def _lazy_ballot(local, other, fresh):
    """(row max of S'', the lazy-max ballot): some row's S'' above LAZY, or a fresh row
    (no key yet, m = FLOOR, centered at 0) seeing its first key."""
    tile_max = _maximum(fx.Float32(local), fx.Float32(other))
    first = fresh & (tile_max > fx.Float32(-math.inf))
    trigger = (tile_max > LAZY) | first
    return tile_max, prim.ballot(trigger)


def _qk_simple(env, slot):
    s_acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
    for b in range(2):
        for c in range(8):
            kw, q = _k_read(env, slot, b, c), env.q_words[c]
            for e in range(2):
                s_acc[b] = _mfma_fp8(
                    (kw[2 * e], kw[2 * e + 1]), (q[2 * e], q[2 * e + 1]), s_acc[b]
                )
    return s_acc


def _first_max(env, s_acc, scales, words):
    """Tile 0: m = row max - P_BIAS (FLOOR without a visible key) and S''(0); (log2 sv, 1/sv)."""
    terms = [None] * 4
    if env.per_token:
        for b in range(2):
            terms[b], terms[2 + b] = _v_scale_terms(scales[2 + b])
    ws = _mask_words(env, words)
    vals = []
    for k in range(32):
        b, i = divmod(k, 16)
        v = s_acc[b][i]
        if env.per_token:
            x = (
                v * prim.row_broadcast(scales[b], i)
            ) * env.c_scale + prim.row_broadcast(terms[b], i)
        else:
            x = v * env.c_scale
        vals.append(_masked(x, _keep(ws, b, i)))
    local = vals[0]
    for v in vals[1:]:
        local = _maximum(local, v)
    row_max = _maximum(
        _maximum(local, fx.gpu.shuffle_xor(local, 32, 64)), fx.Float32(_MAX_FLOOR)
    )
    m_run = (row_max == fx.Float32(_MAX_FLOOR)).select(row_max, row_max - P_BIAS)
    neg = fx.Float32(0.0) - m_run
    return (
        fx.Vector.from_elements([v + neg for v in vals], fx.Float32),
        m_run,
        terms[2:],
    )


# ---- Phase parts ----


def _part1(env, st, t, cur, prev, fixup):
    """QK(t) and P'(t-1); the barrier; the first V reads and P' K-step 0 of PV(t-1).

    After QK MFMA FIX_AT, fixup() finishes tile t-1 (masks, lazy max, rescale) under the
    issued MFMAs; the exponentials of P'(t-1) follow it."""
    mem = {"vdata": []}

    def gen_mem():
        vgen = _gen_v_loads(env, st["vids"], mem["vdata"])
        kgen = _gen_k_writes(env, st["kdata"], 1 - cur, KW_GAP)
        # Interleave the K(t+1) writes (one per KW_GAP MFMAs) with the V(t) loads.
        for _ in zip(kgen, vgen):
            yield
        yield from kgen
        yield from vgen
        mem["idl2"] = _load_idl(env, t + 2)
        mem["lidl2"] = _load_lidl(env, t + 2) if env.slotted else fx.Int32(0)
        yield
        mem["mask"] = _load_mask(env, t)
        yield
        mem["vids"] = _vids(env, st["lidl1"] if env.slotted else st["idl1"])
        idl1 = _fix_ids(env, st["idl1"], t + 1)
        yield
        mem["sids"] = (_bperm(idl1, env.sj),)
        yield
        mem["sids"] += (_bperm(idl1, env.sj + 8),)
        yield

    res, pw, vreads = {}, [None] * 4, {}
    valu, fixed = iter(()), None
    memg = gen_mem()
    tail = None
    reads = list(st["kpre"]) + [None] * (16 - len(st["kpre"]))
    s_acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
    for i in range(16):
        b, c = K_ORDER[i]
        if i + KPRE < 16:
            reads[i + KPRE] = _k_read(env, cur, *K_ORDER[i + KPRE])
            _pin()
        kw, q = reads[i], env.q_words[c]
        for e in range(2):
            j = 2 * i + e
            _pin()
            s_acc[b] = _mfma_fp8(
                (kw[2 * e], kw[2 * e + 1]), (q[2 * e], q[2 * e + 1]), s_acc[b]
            )
            _pin()
            for _ in range(P1):
                _step(valu)
            if tail is not None:
                for _ in range(TAIL):
                    _step(tail)
            else:
                _step(memg)
            if j == FIX_AT:
                fixed = fixup()
                _pin()
                valu = _gen_part1(env, fixed["s_prev"], st["rsv_prev"], res)
            if j == BAR_AT:
                # Publishes K(t+1) and V(t-1); retires every wave's QK(t-1) and PV(t-2) reads.
                _drain(memg)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _stage_end()
                for r in range(VPRE):
                    q_, mb_ = divmod(r, 8)
                    vreads[(q_, mb_)] = _v_read(env, prev, mb_, q_)
                _pin()
                tail = _gen_pack(res["p"], 0, pw)
    _drain(valu)
    _drain(memg)
    if tail is not None:
        _drain(tail)
    return s_acc, res, mem, pw, vreads, fixed


def _part2(env, st, res, s_acc, mem, pw, vreads, t, cur, prev, first=False):
    """PV(t-1) and S''(t); K(t+2) and scale(t+1) loads; the first K reads of QK(t+1). In a
    task's first phase (first) the K-step 0 MFMAs start O."""
    o = list(st["o"]) if not first else [None] * 8
    m_run = st["m_run"]
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    ctx = {
        "p": res["p"],
        "pw": pw,
        "s_acc": s_acc,
        "sk": st["sk"],
        "sv": st["sv"],
        "vdata": mem["vdata"],
        "mask": mem["mask"],
        "fresh": fresh,
        "neg": fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run),
        "cur": cur,
    }
    valu = _gen_part2(env, ctx)
    out = {}

    def gen_mem():
        kid = _bperm(_fix_ids(env, mem["idl2"], t + 2), env.kblk)
        yield
        kdata = []
        yield from _gen_k_loads(env, kid, kdata)
        out["kdata"] = kdata
        scales = []
        yield from _gen_scales(env, mem["sids"], scales)
        out["scales"] = scales
        yield

    memg = gen_mem()
    order = [(q, e, mb) for q in range(2) for e in range(2) for mb in range(8)]
    issue_at = {}
    for r in range(VPRE, 16):
        q, mb = divmod(r, 8)
        issue_at.setdefault(max(0, 16 * q + mb - VPRE), []).append((q, mb))
    kpre = []
    for j, (q, e, mb) in enumerate(order):
        for rq, rmb in issue_at.get(j, []):
            vreads[(rq, rmb)] = _v_read(env, prev, rmb, rq)
            _pin()
        if j >= 32 - KPRE:
            kpre.append(_k_read(env, 1 - cur, *K_ORDER[len(kpre)]))
            _pin()
        if pw[2 * q + e] is None:
            while pw[2 * q + e] is None:
                next(valu)
        vw = vreads[(q, mb)]
        _pin()
        # A task's first K-step starts O (no accumulator to load).
        acc = fx.Vector.filled(16, 0.0, fx.Float32) if o[mb] is None else o[mb]
        o[mb] = _mfma_fp8((vw[2 * e], vw[2 * e + 1]), pw[2 * q + e], acc)
        _pin()
        for _ in range(P2):
            _step(valu)
        if j % 3 == 1:
            _step(memg)
    _drain(valu)
    _drain(memg)
    return o, ctx, out, kpre


def _store_v_now(env, vdata, slot):
    _drain(_gen_v_writes(env, vdata, slot, 0))


def _pv_last(env, pw, o):
    o = list(o)
    for q in range(2):
        for mb in range(8):
            vw = _v_read(env, env.last & 1, mb, q)
            for e in range(2):
                o[mb] = _mfma_fp8((vw[2 * e], vw[2 * e + 1]), pw[2 * q + e], o[mb])
    return o


def _epilogue(env, s_last, l_run, o, rsv):
    """P'(last), its row sums and PV(last); returns O and the lane row sum."""
    l_run, o = fx.Float32(l_run), [fx.Vector(x) for x in o]
    res = {}
    _drain(_gen_part1(env, fx.Vector(s_last), rsv, res))
    l_run = l_run + res["sum"]
    pw = [None] * 4
    for s in range(4):
        _drain(_gen_pack(res["p"], s, pw))
    # The V(last) writes landed; the next task's id and Q loads stay in flight.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o = _pv_last(env, pw, o)
    # Every wave finished its V reads before the output shuffle reuses LDS.
    _stage_end()
    return o, l_run


def _store_out(env, o, inv):
    """O * inv -> BF16 rows through the wave's 16 KiB of LDS, then 16-byte row stores."""
    base = env.wave * 16384
    n, h = env.lane & 31, env.lane >> 5
    for mb in range(8):
        for k in range(4):
            words = _scaled_bf16([o[mb][4 * k + r] for r in range(4)], inv)
            chunk = 4 * mb + k
            offset = (
                base + n * 512 + ((chunk ^ (n & 7)) << 4) + ((h ^ ((n >> 3) & 1)) << 3)
            )
            _write_lds(env.lds, offset, (words[0], words[1]))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    for i in range(16):
        r, c = 2 * i + (env.lane >> 5), env.lane & 31
        w = prim.load_lds(env.lds, base + r * 512 + ((c ^ (r & 7)) << 4))
        # Bit 3 of r = 2i + (lane >> 5) is bit 2 of i: the half swap is static.
        swap = (i >> 2) & 1
        words = fx.Vector.from_elements([w[j ^ (2 * swap)] for j in range(4)], fx.Int32)
        row = env.row_offset + env.wave * 32 + r
        query, head = row // env.gp, row % env.gp
        valid = (query < env.qvalid) & (query < env.bq) & (head < env.group)
        offset = (query * env.heads + head) * (D * 2) + c * 16
        _buffer_store_words(words, env.gout, valid.select(offset, env.q_oob))


def _store_whole(env, o, l_run, v_scale, m_run=None):
    """O / l (times the V scale) as BF16; with m_run also the rows' log-sum-exp
    m + log2(l) of S * c (log2 units), which the residual direct merges with."""
    total = l_run + fx.gpu.shuffle_xor(l_run, 32, 64)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    if v_scale is not None:
        inv = inv * v_scale
    if m_run is not None:
        lse = fx.Float32(m_run) + prim.log2(total)
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(lse)], fx.Int32), env.glse, env.lse_lane
        )
    _store_out(env, o, inv)


def _partial_offset(env, part):
    return (env.slot * SPLIT_PARTS + part) * PARTIAL_BYTES


def _store_partial(env, o, m_run, l_run):
    """This part's O^T fragments, m and the lane's l (device-scope stores: the merging CTA may
    run on another XCD)."""
    base = _partial_offset(env, env.part)
    lane_o = base + env.wave * (64 * 128 * 4) + env.lane * 16
    for mb in range(8):
        for k in range(4):
            words = [_as_i32(o[mb][4 * k + r]) for r in range(4)]
            _buffer_store_words(
                fx.Vector.from_elements(words, fx.Int32),
                env.gpart,
                lane_o + (4 * mb + k) * 1024,
                SC_SYSTEM,
            )
    lane_ml = base + PARTIAL_ML + env.wave * 512 + env.lane * 4
    for i, value in enumerate((m_run, l_run)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            env.gpart,
            lane_ml + 256 * i,
            SC_SYSTEM,
        )


def _merge_out(env, o, m_run, l_run, v_scale):
    """The part counting last: its O, m and l in registers, the other part's record loaded
    (all loads in flight at once). O = O0 * 2^(m0 - m) + O1 * 2^(m1 - m) with m the larger
    (part 0 first, so either part computes the same output), l likewise; then the output of
    an unsplit task."""
    other = SPLIT_PARTS - 1 - env.part
    base = _partial_offset(env, other)
    lane_ml = base + PARTIAL_ML + env.wave * 512 + env.lane * 4
    lane_o = base + env.wave * (64 * 128 * 4) + env.lane * 16
    loaded = [
        _buffer_words(env.gpart, lane_o + (4 * mb + k) * 1024, 4, SC_SYSTEM)
        for mb in range(8)
        for k in range(4)
    ]
    m_other = _as_f32(_buffer_words(env.gpart, lane_ml, 1, SC_SYSTEM)[0])
    l_other = _as_f32(_buffer_words(env.gpart, lane_ml + 256, 1, SC_SYSTEM)[0])
    first = env.part == 0
    m0, m1 = first.select(m_run, m_other), first.select(m_other, m_run)
    top = _maximum(m0, m1)
    # A part without a key of the row has m = FLOOR: weight 0 (both: 1, l = 0, output 0).
    w0, w1 = _exp(m0 - top), _exp(m1 - top)
    l0, l1 = first.select(l_run, l_other), first.select(l_other, l_run)
    l_sum = _fma(l1, w1, l0 * w0)
    total = l_sum + fx.gpu.shuffle_xor(l_sum, 32, 64)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    if v_scale is not None:
        inv = inv * v_scale
    merged = []
    for mb in range(8):
        values = []
        for i in range(16):
            mine = fx.Float32(fx.Vector(o[mb])[i])
            theirs = _as_f32(loaded[4 * mb + i // 4][i % 4])
            a, b = first.select(mine, theirs), first.select(theirs, mine)
            values.append(_fma(b, w1, a * w0))
        merged.append(fx.Vector.from_elements(values, fx.Float32))
    _store_out(env, merged, inv)


def _lane_env(lds, per_token, slotted, hk):
    """Per-lane constants every task shares."""
    tid = fx.Int32(fx.thread_idx.x)
    wave, lane = prim.uniform(tid >> 6), tid & 63
    n, h = lane & 31, lane >> 5
    # V^T writes: wave (q, h) = (wave >> 1, wave & 1) fills chunk 2h + q of every dim.
    vq, vh = wave >> 1, wave & 1
    key, kpart, flip = tid >> 2, tid & 3, (tid >> 4) & 1
    k_chunks = [((fx.Int32(i) ^ flip) << 2) + kpart for i in range(4)]
    return {
        "lds": lds,
        "lane": lane,
        "wave": wave,
        "n": n,
        "h": h,
        "per_token": per_token,
        "slotted": slotted,
        "tok_bytes": hk * D,
        "scale_step": hk * 4,
        "stok": lane & 3,
        # Membership bits of the other key half (h): value (b, i) uses bits 8b + 4h + 0..3.
        "foreign_bits": (h != 0).select(fx.Int32(0x0F0F0F0F), fx.Int32(-0x0F0F0F10)),
        "k_addr": [_k_offset(n, 2 * c + h) for c in range(8)],
        "v_addr": [
            [_vt_chunk(n + 64 * hi, 2 * h + q) - 4096 * hi for q in range(2)]
            for hi in range(2)
        ],
        "v_wbase": [_vt_chunk(4 * lane + j, 2 * vh + vq) for j in range(4)],
        "vslot": 8 * vq + vh,
        "v_lane": (lane & 3) * 256 + (lane >> 2) * 16,
        "k_wbase": [key * 256 + ((k_chunks[i] ^ _swz_k(key)) << 4) for i in range(4)],
        "k_goff": [k_chunks[i] * 16 for i in range(4)],
        "ktok": key & 3,
        "kblk": tid >> 4,
        "sj": 2 * ((lane & 15) >> 2) + h,
    }


def _sload(tensor, index):
    """One int32 of a tensor at a wave-uniform index by a scalar load (SMEM, lgkmcnt): it does
    not wait behind the vector memory counter, which also counts the output stores."""
    # readfirstlane marks the result uniform for LLVM's divergence analysis (no-op on an SGPR):
    # descriptors and the persistent loop built from it stay scalar.
    return prim.uniform(fx.rocdl.s_buffer_load(_buffer(tensor, 0x7FFFFFFF), index))


def _decode(ORDER, ACTIVE, work, limit, per_tile, split=1):
    """Order entry of a persistent slot and whether it runs; past the order or routed direct:
    idle. K3 orders each CTA's slots by decreasing cost, so its runnable tasks come first.
    With split > 1 an entry is (task << 8) | (slot << 2) | (part << 1) | split."""
    inside = work < limit
    mapped = inside.select(_sload(ORDER, _min(work, limit - 1)), fx.Int32(-1))
    task = mapped >> 8 if split > 1 else mapped
    tile = (mapped >= 0).select(task // per_tile, fx.Int32(0))
    return mapped, (mapped >= 0).select(_sload(ACTIVE, tile), fx.Int32(0))


def _task_scalars(T, mapped, enabled, HK, BQ, GP, slotted, split=1):
    """A task's tile, KV head, row slice and plan entries (scalar loads); an idle slot (enabled
    0) gets no rows and no keys, so its loads read nothing. A part of a split task keeps its
    key range: N64 steps [first, end) of the tile's list, part 0 the first ceil(steps / 2).
    """
    slices = (BQ * GP + 127) // 128
    raw = (mapped >= 0).select(mapped, fx.Int32(0))
    task = raw >> 8 if split > 1 else raw
    on = enabled != 0
    tile_id = task // (HK * slices)
    sc = {
        "tile_id": tile_id,
        "hkv": task % HK,
        "row_offset": ((task // HK) % slices) * 128,
        "q0": _sload(T.META, tile_id * 5),
        "qvalid": on.select(_sload(T.META, tile_id * 5 + 1), fx.Int32(0)),
        "k0": _sload(T.META, tile_id * 5 + 2),
        "kv_len": on.select(_sload(T.META, tile_id * 5 + 3), fx.Int32(0)),
        "count": on.select(_sload(T.COUNTS, tile_id), fx.Int32(0)),
        "tail": _sload(T.TAILS, tile_id) if slotted else fx.Int32(-1),
        "pstart": _sload(T.PSTARTS, tile_id),
        "first": fx.Int32(0),
    }
    if split > 1:
        parted = (raw & 1) != 0
        part = (raw >> 1) & 1
        steps = (sc["count"] + 15) >> 4
        half = (steps + 1) >> 1
        first = (parted & (part == 1)).select(half, fx.Int32(0))
        end = (parted & (part == 0)).select(half, steps)
        sc.update(
            count=_min(sc["count"], end * 16) - first * 16,
            first=first,
            split=parted.select(fx.Int32(1), fx.Int32(0)),
            slot=(raw >> 2) & (SPLIT_SLOTS - 1),
            part=part,
        )
    return sc


def _task_env(lane, T, sc, H, HK, NQ, CAP, POOL, PACKED, BQ, GP):
    """One task's rows, key list and buffers from its scalars."""
    group = H // HK
    tile_id, hkv, row_offset = sc["tile_id"], sc["hkv"], sc["row_offset"]
    q0, qvalid, k0, kv_len, count = (
        sc["q0"],
        sc["qvalid"],
        sc["k0"],
        sc["kv_len"],
        sc["count"],
    )
    n, h = lane["n"], lane["h"]
    row = row_offset + lane["wave"] * 32 + n
    query, head = row // GP, row % GP
    valid = (query < qvalid) & (query < BQ) & (head < group)
    # Byte offsets are unsigned 32-bit: Q and each request's (or the pool's) K/V stay below 4 GiB.
    q_rows = fx.Uint32(NQ - q0) * H - hkv * group
    q_base = fx.Int64(q0) * H + hkv * group
    q8_extent = q_rows * D
    q8_oob = fx.Int32(q8_extent)
    pv_extent = fx.Uint32(PACKED) * HK * PACK_BLOCK - hkv * PACK_BLOCK
    tok_bytes = lane["tok_bytes"]
    if fx.const_expr(lane["slotted"]):
        kv_extent = fx.Uint32(POOL) * HK * D - hkv * D
        kv_offset = fx.Int64(hkv * D)
        scale_extent = (fx.Uint32(POOL) * HK - hkv) * 4
        scale_offset = fx.Int64(hkv)
        final = (sc["tail"] >= 0).select(sc["tail"] >> 2, fx.Int32(-1))
    else:
        kv_extent = fx.Uint32(kv_len) * HK * D - hkv * D
        kv_offset = fx.Int64(k0) * (HK * D) + hkv * D
        scale_extent = (fx.Uint32(kv_len) * HK - hkv) * 4
        scale_offset = fx.Int64(k0) * HK + hkv
        final = fx.Int32(-1)
    env = _Env(**lane)
    env.__dict__.update(
        valid=valid,
        count=count,
        last=(count + 15) // 16 - 1,
        final=final,
        tail_len=kv_len & 3,
        gblocks=_buffer(
            fx.make_view(
                fx.get_iter(T.BLOCKS) + tile_id * CAP + sc["first"] * 16,
                fx.make_layout(CAP, 1),
            ),
            count * 4,
        ),
        # Without a slot table the block list is already request-local.
        gvids=(
            _buffer(
                fx.make_view(
                    fx.get_iter(T.LBLOCKS) + tile_id * CAP + sc["first"] * 16,
                    fx.make_layout(CAP, 1),
                ),
                count * 4,
            )
            if lane["slotted"]
            else None
        ),
        gpv=_buffer(
            fx.make_view(
                fx.get_iter(T.PV) + hkv * PACK_BLOCK, fx.make_layout(pv_extent, 1)
            ),
            pv_extent,
        ),
        pstart=sc["pstart"],
        pv_step=HK * PACK_BLOCK,
        gmask=_buffer(
            fx.make_view(
                fx.get_iter(T.MEMBERS) + (tile_id * (CAP // 16) + sc["first"]) * BQ * 4,
                fx.make_layout((CAP // 16) * BQ * 4, 1),
            ),
            (CAP // 16 - sc["first"]) * BQ * 8,
        ),
        mask_lane=_min(query, fx.Int32(BQ - 1)) * 8,
        mask_step=BQ * 8,
        gq8=_buffer(
            fx.make_view(fx.get_iter(T.Q8) + q_base * D, fx.make_layout(q8_extent, 1)),
            q8_extent,
        ),
        q8_lane=valid.select((query * H + head) * D + h * 16, q8_oob),
        gqs=_buffer(
            fx.make_view(fx.get_iter(T.QS) + q_base, fx.make_layout(q_rows, 1)),
            q_rows * 4,
        ),
        qs_lane=valid.select((query * H + head) * 4, fx.Int32(q_rows * 4)),
        # Row log-sum-exp for the residual direct merge (QS layout), from the first half.
        glse=_buffer(
            fx.make_view(fx.get_iter(T.LSE) + q_base, fx.make_layout(q_rows, 1)),
            q_rows * 4,
        ),
        lse_lane=(valid & (h == 0)).select(
            (query * H + head) * 4, fx.Int32(q_rows * 4)
        ),
        gout=_buffer(
            fx.make_view(fx.get_iter(T.OUT) + q_base * D, fx.make_layout(q8_extent, 1)),
            q8_extent * 2,
        ),
        q_oob=fx.Int32(q8_extent * 2),
        gk=_buffer(
            fx.make_view(fx.get_iter(T.K) + kv_offset, fx.make_layout(kv_extent, 1)),
            kv_extent,
        ),
        gv=_buffer(
            fx.make_view(fx.get_iter(T.V) + kv_offset, fx.make_layout(kv_extent, 1)),
            kv_extent,
        ),
        kv_oob=fx.Int32(kv_extent),
        # First block wholly past the K/V extent (and so past the scales' extent).
        id_oob=fx.Int32((kv_extent + (4 * tok_bytes - 1)) // (4 * tok_bytes)),
        gks=_buffer(
            fx.make_view(
                fx.get_iter(T.KS) + scale_offset, fx.make_layout(scale_extent, 1)
            ),
            scale_extent,
        ),
        gvs=_buffer(
            fx.make_view(
                fx.get_iter(T.VS) + scale_offset, fx.make_layout(scale_extent, 1)
            ),
            scale_extent,
        ),
        scale_oob=fx.Int32(scale_extent),
        row_offset=row_offset,
        qvalid=qvalid,
        bq=BQ,
        gp=GP,
        group=group,
        heads=H,
    )
    if env.gvids is None:
        env.gvids = env.gblocks
    if "split" in sc:
        env.split, env.slot, env.part = sc["split"], sc["slot"], sc["part"]
        env.gpart = _buffer(T.PARTIALS, SPLIT_SLOTS * SPLIT_PARTS * PARTIAL_BYTES)
    return env


def _prefetch_ids(env):
    """Ids and masks of tiles 0-2, FP8 Q words and the q scale of a task."""
    idl = [_load_idl(env, fx.Int32(i)) for i in range(3)]
    return {
        "idl": idl,
        "lidl": (
            [_load_lidl(env, fx.Int32(i)) for i in range(3)] if env.slotted else idl
        ),
        "w0": _load_mask(env, fx.Int32(0)),
        "q8": [_buffer_words(env.gq8, env.q8_lane + 32 * c, 4) for c in range(8)],
        "qs": _as_f32(_buffer_words(env.gqs, env.qs_lane, 1)[0]),
    }


def _prefetch_kv(env, a):
    """K(0..2), V(0) and scales(0..1) of a task once its ids landed."""
    fixed = [_fix_ids(env, a["idl"][i], fx.Int32(i)) for i in range(3)]
    kids = [_bperm(fixed[i], env.kblk) for i in range(3)]
    vids0 = _vids(env, a["lidl"][0])
    vids1 = _vids(env, a["lidl"][1])
    sids = [(_bperm(fixed[i], env.sj), _bperm(fixed[i], env.sj + 8)) for i in range(2)]
    vdata, k0w, k1w, k2w, sc0, sc1 = [], [], [], [], [], []
    _drain(_gen_v_loads(env, vids0, vdata))
    _drain(_gen_k_loads(env, kids[0], k0w))
    _drain(_gen_k_loads(env, kids[1], k1w))
    _drain(_gen_scales(env, sids[0], sc0))
    _drain(_gen_scales(env, sids[1], sc1))
    _drain(_gen_k_loads(env, kids[2], k2w))
    return {
        "vdata": vdata,
        "k0": k0w,
        "k1": k1w,
        "k2": k2w,
        "sc0": sc0,
        "sc1": sc1,
        "vids1": vids1,
    }


def _pack_pre(a, b):
    """The loop-carried prefetch of the next task as six vectors."""
    misc = [_as_i32(a["qs"]), *a["w0"], a["idl"][2]]
    misc += [_as_i32(x) for x in b["sc0"] + b["sc1"]]
    misc += [*b["vids1"], a["lidl"][2]]
    words = [w[j] for w in a["q8"] for j in range(4)]
    return (
        fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32),
        fx.Vector.from_elements([fx.Int32(w) for w in misc], fx.Int32),
        fx.Vector.from_elements(b["k0"], fx.Int32),
        fx.Vector.from_elements(b["k1"], fx.Int32),
        fx.Vector.from_elements(b["k2"], fx.Int32),
        fx.Vector.from_elements([w[j] for w in b["vdata"] for j in range(4)], fx.Int32),
    )


def _unpack_pre(q8, misc, k0, k1, k2, v0):
    q8, misc, v0 = fx.Vector(q8), fx.Vector(misc), fx.Vector(v0)
    return {
        "q_words": [
            fx.Vector.from_elements([q8[4 * c + j] for j in range(4)], fx.Int32)
            for c in range(8)
        ],
        "qs": _as_f32(misc[0]),
        "w0": (misc[1], misc[2]),
        "idl2": misc[3],
        "sc0": [_as_f32(misc[4 + i]) for i in range(4)],
        "sc1": [_as_f32(misc[8 + i]) for i in range(4)],
        "vids1": tuple(prim.uniform(misc[12 + j]) for j in range(4)),
        "lidl2": misc[16],
        "k0": [fx.Vector(k0)[i] for i in range(16)],
        "k1": [fx.Vector(k1)[i] for i in range(16)],
        "k2": fx.Vector(k2),
        "vdata": [
            fx.Vector.from_elements([v0[4 * i + j] for j in range(4)], fx.Int32)
            for i in range(4)
        ],
    }


@flyc.jit
def _task(
    Q8,
    QS,
    K,
    V,
    KS,
    VS,
    OUT,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    TAILS,
    ORDER,
    ACTIVE,
    PV,
    LBLOCKS,
    PSTARTS,
    PARTIALS,
    COUNTERS,
    LSE,
    lds,
    mapped,
    work,
    limit,
    grid,
    pre_q8,
    pre_misc,
    pre_k0,
    pre_k1,
    pre_k2,
    pre_v0,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    PACKED: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
):
    """One task (128 (query, head) rows of a union tile over its compact key list, or one key
    range of it) from its prefetched first tiles; prefetches the next slot's under its
    epilogue and output."""
    T = _Env(
        Q8=Q8,
        QS=QS,
        K=K,
        V=V,
        KS=KS,
        VS=VS,
        OUT=OUT,
        META=META,
        BLOCKS=BLOCKS,
        MEMBERS=MEMBERS,
        COUNTS=COUNTS,
        TAILS=TAILS,
        PV=PV,
        LBLOCKS=LBLOCKS,
        PSTARTS=PSTARTS,
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    lane = _lane_env(lds, PER_TOKEN, SLOTTED, HK)
    env = _task_env(
        lane,
        T,
        _task_scalars(T, mapped, fx.Int32(1), HK, BQ, GP, SLOTTED, SPLIT),
        H,
        HK,
        NQ,
        CAP,
        POOL,
        PACKED,
        BQ,
        GP,
    )
    pre = _unpack_pre(pre_q8, pre_misc, pre_k0, pre_k1, pre_k2, pre_v0)

    # ---- Prologue: K(0), K(1), V(0) to LDS; QK(0) and the exact max ----
    c_scale = pre["qs"] * SCALE_LOG2
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(KS[0])
    env.q_words, env.c_scale = pre["q_words"], c_scale
    _store_v_now(env, pre["vdata"], 0)
    _drain(_gen_k_writes(env, pre["k0"], 0))
    _drain(_gen_k_writes(env, pre["k1"], 1))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    s_acc0 = _qk_simple(env, 0)
    # The next slot's task (scalar loads) resolves under QK(0) and the first max.
    per_tile = HK * ((BQ * GP + 127) // 128)
    next_mapped, next_enabled = _decode(
        ORDER, ACTIVE, work + grid, limit, per_tile, SPLIT
    )
    s_cur, row_max, rsv0 = _first_max(env, s_acc0, pre["sc0"], pre["w0"])
    # Every wave read K(0) before phase 1 writes K(2) over it.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    kpre = [_k_read(env, 1, *K_ORDER[i]) for i in range(KPRE)]
    zeros = tuple(fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(8))
    if fx.const_expr(not PER_TOKEN):
        rsv0 = [fx.Float32(0.0), fx.Float32(0.0)]
    sc1 = pre["sc1"]
    # Tile 0 is final (exact max, masked): phase 1's fix-up passes it through.
    ninf = fx.Float32(-math.inf)
    state = (
        (
            s_cur,
            ninf,
            ninf,
            row_max,
            fx.Float32(0.0),
        )
        + zeros
        + (rsv0[0], rsv0[1], sc1[0], sc1[1], sc1[2], sc1[3])
        + (pre["k2"], *pre["vids1"], pre["idl2"], pre["lidl2"])
        + tuple(kpre)
        + tuple(fx.Vector.filled(4, 0, fx.Int32) for _ in range(4 - KPRE))
    )

    def finish(s_prev, local, other, m_run, l_run, o):
        """Tile t-1's lazy max and rescale of O, l and its S'' (Part 2 masked it)."""
        m_run = fx.Float32(m_run)
        tile_max, ballot = _lazy_ballot(local, other, m_run == fx.Float32(_MAX_FLOOR))
        o, l_run, s_prev, m_run = _lazy_max(
            o, l_run, fx.Vector(s_prev), m_run, ballot, lambda: tile_max
        )
        return {"o": o, "l_run": l_run, "s_prev": s_prev, "m_run": m_run}

    @flyc.jit
    def phase(
        s_prev,
        local_prev,
        other_prev,
        m_run,
        l_run,
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        rsvp0,
        rsvp1,
        skc0,
        skc1,
        svc0,
        svc1,
        kflat,
        vid0,
        vid1,
        vid2,
        vid3,
        idl1,
        lidl1,
        kp0,
        kp1,
        kp2,
        kp3,
        t,
        CUR: fx.Constexpr[int],
        PREV: fx.Constexpr[int],
        FIRST: fx.Constexpr[bool],
    ):
        kflat = fx.Vector(kflat)
        o_in = [fx.Vector(x) for x in (o0, o1, o2, o3, o4, o5, o6, o7)]
        st = {
            "rsv_prev": [fx.Float32(rsvp0), fx.Float32(rsvp1)],
            "sk": [fx.Float32(skc0), fx.Float32(skc1)],
            "sv": [fx.Float32(svc0), fx.Float32(svc1)],
            "kdata": [kflat[i] for i in range(16)],
            "vids": tuple(prim.uniform(fx.Int32(v)) for v in (vid0, vid1, vid2, vid3)),
            "idl1": fx.Int32(idl1),
            "lidl1": fx.Int32(lidl1),
            "kpre": [fx.Vector(x) for x in (kp0, kp1, kp2, kp3)[:KPRE]],
        }
        t = fx.Int32(t)

        def fixup():
            if fx.const_expr(FIRST):
                # Tile 0 is final (exact max, masked): no ballot, no rescale, no O yet.
                return {
                    "o": None,
                    "l_run": fx.Float32(l_run),
                    "s_prev": fx.Vector(s_prev),
                    "m_run": fx.Float32(m_run),
                }
            return finish(
                s_prev,
                local_prev,
                other_prev,
                m_run,
                l_run,
                o_in,
            )

        s_acc, res, mem, pw, vreads, fixed = _part1(env, st, t, CUR, PREV, fixup)
        st.update(o=fixed["o"], m_run=fixed["m_run"])
        o, ctx, out, kpre = _part2(
            env, st, res, s_acc, mem, pw, vreads, t, CUR, PREV, FIRST
        )
        row_sum = fixed["l_run"] + res["sum"]
        rsv = ctx["rsv"] if PER_TOKEN else [fx.Float32(0.0), fx.Float32(0.0)]
        sc = out["scales"]
        pads = [fx.Vector.filled(4, 0, fx.Int32)] * (4 - KPRE)
        return (
            fx.Vector.from_elements(ctx["s_cur"], fx.Float32),
            ctx["local"],
            ctx["other"],
            fixed["m_run"],
            row_sum,
            *o,
            rsv[0],
            rsv[1],
            sc[0],
            sc[1],
            sc[2],
            sc[3],
            fx.Vector.from_elements(out["kdata"], fx.Int32),
            *mem["vids"],
            mem["idl2"],
            mem["lidl2"],
            *kpre,
            *pads,
        )

    key_tiles = env.last + 1
    if key_tiles > 1:
        state = phase(*state, fx.Int32(1), 1, 0, True)
    for t in range(fx.Int32(2), key_tiles - 1, fx.Int32(2)):
        state = phase(*state, t, 0, 1, False)
        state = phase(*state, fx.Int32(t) + 1, 1, 0, False)
    if (key_tiles > 2) & ((key_tiles & 1) == 1):
        state = phase(*state, env.last, 0, 1, False)

    # ---- Epilogue and output; the next slot's loads issue under them ----
    next_scalars = _task_scalars(
        T, next_mapped, next_enabled, HK, BQ, GP, SLOTTED, SPLIT
    )
    nenv = _task_env(lane, T, next_scalars, H, HK, NQ, CAP, POOL, PACKED, BQ, GP)
    a = _prefetch_ids(nenv)
    last = finish(*state[:5], state[5:13])
    o, l_run = _epilogue(
        env,
        last["s_prev"],
        last["l_run"],
        last["o"],
        [fx.Float32(state[13]), fx.Float32(state[14])],
    )
    b = _prefetch_kv(nenv, a)
    v_scale = None if PER_TOKEN else fx.Float32(VS[0])
    if fx.const_expr(SPLIT > 1):
        if env.split != 0:
            # Both parts store their records; the one counting last merges them.
            _store_partial(env, o, last["m_run"], l_run)
            fx.rocdl.s_waitcnt(vmcnt=0)
            _stage_end()
            if fx.Int32(fx.thread_idx.x) == 0:
                done = fx.atomic_add(
                    fx.get_iter(COUNTERS) + env.slot, fx.Int32(1), syncscope="agent"
                )
                _write_lds(env.lds, 0, (done, done))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _stage_end()
            done = prim.uniform(fx.Int32(prim.load_lds(env.lds, 0)[0]))
            # Every wave read the count before the output shuffle reuses LDS.
            _stage_end()
            if done == SPLIT_PARTS - 1:
                _merge_out(env, o, last["m_run"], l_run, v_scale)
        else:
            _store_whole(env, o, l_run, v_scale)
    else:
        _store_whole(env, o, l_run, v_scale, last["m_run"] if RESIDUAL else None)
    # The next task's K/V writes reuse the LDS of every wave's output shuffle.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    return (next_mapped, next_enabled) + _pack_pre(a, b)


@flyc.kernel(name="sparse_prefill_union_fp8_d256", known_block_size=[THREADS, 1, 1])
def _kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    PV: fx.Tensor,
    LBLOCKS: fx.Tensor,
    PSTARTS: fx.Tensor,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    PACKED: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
):
    allocator = fx.SharedAllocator()
    lds = (
        allocator.allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    T = _Env(
        Q8=Q8,
        QS=QS,
        K=K,
        V=V,
        KS=KS,
        VS=VS,
        OUT=OUT,
        META=META,
        BLOCKS=BLOCKS,
        MEMBERS=MEMBERS,
        COUNTS=COUNTS,
        TAILS=TAILS,
        PV=PV,
        LBLOCKS=LBLOCKS,
        PSTARTS=PSTARTS,
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    per_tile = HK * ((BQ * GP + 127) // 128)
    limit = ((TASKS + GRID - 1) // GRID) * GRID
    work = fx.Int32(fx.block_idx.x)
    mapped, enabled = _decode(ORDER, ACTIVE, work, limit, per_tile, SPLIT)
    first = _task_env(
        _lane_env(lds, PER_TOKEN, SLOTTED, HK),
        T,
        _task_scalars(T, mapped, enabled, HK, BQ, GP, SLOTTED, SPLIT),
        H,
        HK,
        NQ,
        CAP,
        POOL,
        PACKED,
        BQ,
        GP,
    )
    ids = _prefetch_ids(first)
    pre = _pack_pre(ids, _prefetch_kv(first, ids))
    while enabled != 0:
        result = _task(
            Q8,
            QS,
            K,
            V,
            KS,
            VS,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            TAILS,
            ORDER,
            ACTIVE,
            PV,
            LBLOCKS,
            PSTARTS,
            PARTIALS,
            COUNTERS,
            LSE,
            lds,
            mapped,
            work,
            limit,
            GRID,
            *pre,
            H,
            HK,
            NQ,
            CAP,
            POOL,
            PACKED,
            BQ,
            GP,
            SCALE_LOG2,
            PER_TOKEN,
            SLOTTED,
            SPLIT,
            RESIDUAL,
        )
        work = work + GRID
        mapped, enabled, pre = result[0], result[1], result[2:]


@flyc.jit
def _launch(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    PV: fx.Tensor,
    LBLOCKS: fx.Tensor,
    PSTARTS: fx.Tensor,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    PACKED: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if GRID > 0:
        _kernel(
            Q8,
            QS,
            K,
            V,
            KS,
            VS,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            TAILS,
            ACTIVE,
            ORDER,
            PV,
            LBLOCKS,
            PSTARTS,
            PARTIALS,
            COUNTERS,
            LSE,
            H,
            HK,
            NQ,
            CAP,
            POOL,
            PACKED,
            BQ,
            GP,
            TASKS,
            GRID,
            SCALE_LOG2,
            PER_TOKEN,
            SLOTTED,
            SPLIT,
            RESIDUAL,
            value_attrs=prim.kernel_attrs(),
        ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


# ---- Q quantization alone (the pack launch quantizes Q for attention calls) ----


@flyc.kernel(name="qsa_quantize_q_fp8", known_block_size=[THREADS, 1, 1])
def _quantize_kernel(Q: fx.Tensor, Q8: fx.Tensor, QS: fx.Tensor, ROWS: fx.Int32):
    quantize_rows(Q, Q8, QS, ROWS, fx.Int32(fx.block_idx.x))


@flyc.jit
def _quantize_launch(
    Q: fx.Tensor, Q8: fx.Tensor, QS: fx.Tensor, ROWS: fx.Int32, stream: fx.Stream
):
    _quantize_kernel(Q, Q8, QS, ROWS, value_attrs=prim.kernel_attrs()).launch(
        grid=((ROWS + QROWS - 1) // QROWS, 1, 1), block=(THREADS, 1, 1), stream=stream
    )


_COMPILED = {}
_Q_COMPILED = {}


def quantize_q(q, q8=None, qs=None):
    """Per-(query, head) E4M3FNUZ Q (uint8 bytes) and FP32 scales amax / 240 of BF16
    Q[M, H, 256], as compute="fp8" quantizes Q (for references of the FP8 kernels)."""
    if q8 is None:
        q8 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
    if qs is None:
        qs = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    launch = quantize_launcher(q)
    if launch is not None:
        with torch.cuda.device(q.device):
            launch(q.view(-1), q8.view(-1), qs.view(-1))
    return q8, qs


def quantize_launcher(q):
    """launch(q, q8, qs) with flat tensors for Q of q's shape; None for an empty Q."""
    rows = q.shape[0] * q.shape[1]
    if rows == 0:
        return None
    stream = torch.cuda.current_stream(q.device)
    compiled = _Q_COMPILED.get(q.device)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        q8 = torch.empty(q.shape, dtype=torch.uint8, device=q.device)
        qs = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        with torch.cuda.device(q.device), torch.cuda.stream(stream):
            compiled = _Q_COMPILED[q.device] = flyc.compile(
                _quantize_launch, q.view(-1), q8.view(-1), qs.view(-1), rows, stream
            )
    return lambda q, q8, qs: compiled(q, q8, qs, rows, stream)


def _args(inputs, plan, packed, out, stream):
    # Slot-table plans read K through K3's pool block table, addressed from the pool's base;
    # V^T comes from the FP8 pack of the direct plan (packed), by request-local block ids.
    k_scale, v_scale = inputs.k_scale, inputs.v_scale
    return (
        inputs.q8.view(-1),
        inputs.qs.view(-1),
        inputs.k,
        inputs.v,
        k_scale.view(-1),
        v_scale.view(-1),
        out.view(-1),
        inputs.scale_log2,
        plan.metadata.view(-1),
        (plan.pool_blocks if plan.slotted else plan.blocks).view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.tails if plan.slotted else plan.counts,
        plan.active,
        plan.task_order,
        packed.packed_value,
        plan.blocks.view(-1),
        plan.pack_starts,
        plan.partials.view(-1) if plan.split > 1 else plan.counts,
        plan.split_counters if plan.split > 1 else plan.counts,
        plan.lse.view(-1) if plan.core_ratio > 0 else plan.counts,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        plan.block_capacity,
        inputs.k.shape[0],
        packed.packed_blocks,
        plan.query_tile,
        plan.group_padded,
        # Order entries (tasks times key-range parts).
        plan.num_tiles
        * inputs.k.shape[1]
        * ((plan.query_tile * plan.group_padded + 127) // 128)
        * plan.split,
        plan.grid,
        k_scale.numel() > 1,
        plan.slotted,
        plan.split,
        plan.core_ratio > 0,
        stream,
    )


def _compiled(inputs, args, stream):
    # H, HK, BQ, GP, PER_TOKEN, SLOTTED, SPLIT, RESIDUAL; the softmax scale is a kernel
    # argument.
    key = (inputs.q.device, *args[21:23], *args[27:29], *args[31:35])
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            # Four waves of 512 registers (VGPRs and AGPRs) and 64 KiB LDS per CTA: one wave
            # per SIMD. Compiled with an empty grid, so it launches nothing.
            compiled = _COMPILED[key] = flyc.compile[{"waves_per_eu": 1}](
                _launch, *args[:30], 0, *args[31:]
            )
    return compiled


def run(*, inputs, plan, packed, out):
    """FP8 union attention after the FP8 pack of the direct plan packed (its V^T). inputs adds
    k_scale / v_scale (FP32 [N, HK] per token or [1]), the FP8 Q of the pack launch
    (inputs.q8 / inputs.qs) and the FP32 softmax_scale * log2(e) (inputs.scale_log2)."""
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, packed, out, stream)
    with torch.cuda.device(inputs.q.device):
        _compiled(inputs, args, stream)(*args)


def launcher(*, inputs, plan, packed, out):
    """launch(q8, qs, k, v, out, k_scale, v_scale, scale_log2) with flat Q8/QS/out/scales and
    unflattened K/V; other arguments prebuilt. Valid while the plan's scratch bindings and
    stream stay.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, packed, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[8:]
    return lambda q8, qs, k, v, o, ks, vs, scale_log2: compiled(
        q8, qs, k, v, ks, vs, o, scale_log2, *tail
    )
