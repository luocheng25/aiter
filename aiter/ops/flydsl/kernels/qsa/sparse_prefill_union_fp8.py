# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA union attention over FP8 (E4M3FNUZ) K/V with FP8 MFMAs.

Same plans and tasks as sparse_prefill_union: a task is 128 (query, head) rows of a union tile
over its compact block list, with exact per-query membership. Eight waves (two per SIMD) run
32x32x16 FP8 MFMAs, S^T = K Q^T and O^T += V^T P.

Waves w and w + 4 share a SIMD and the same 32 rows. The leader L (w < 4) scores keys 32..63 of
every 64-key tile (S block 1) and accumulates O over dims 0..127; the follower F (w >= 4)
scores keys 0..31 and accumulates dims 128..255. Each scores half of a row's keys and needs the
other half's FP8 P' for its PV, so P', the lane maxima and (per token) F's row-sum chains go
through LDS.
CTA barriers delimit slots; L's M-stages (MFMAs) run in even slots and F's in odd ones, so a
SIMD's MFMA pipe runs one wave's M-stage while the other runs its V-stage (softmax, K/V copies),
and an M-stage holds issue priority (s_setprio).

Numerics. sparse_prefill_common.quantize_rows (in the FP8 pack launch) stores Q per (query,
head) row as E4M3FNUZ with scale = row amax / 240. K/V scales are per token and KV head (FP32
[N, HK]; with a slot table [P, HK]) or one per tensor ([1]). Softmax runs in
log2 units with a lazy max m (sparse_prefill_common._lazy_max): S'' = S * c - m with
c = q_scale * SCALE_LOG2 (times k_scale per tensor, per key otherwise), SCALE_LOG2 the
runtime FP32 softmax_scale * log2(e). Per token, log2(v_scale) joins S'', so the FP8
P' = exp2(S'') carries the V scale and stays below 2**LAZY while l accumulates P' / v_scale;
per tensor the output takes v_scale. m moves only when some row's S'' exceeds LAZY or a row
sees its first key (a task starts with every row fresh, m = FLOOR); then the row max lands at
S'' = P_BIAS (P' = 2**P_BIAS, so small weights stay above the FP8 underflow) and O, l and S''
follow. Both halves of a row see the same maxima and so take the same m. Per token the row
sums run as two chains over a row's 32 lane values, F's 16 first, then L's, and L holds l;
per tensor the wave holding l (F, with a slot table L) sums the FP8 P' words its PV
multiplies with two sparse 16x16x64 MFMAs per M-stage, so l and O add the same values.

Tile u of a task (64 keys, 16 four-token blocks; L's step u is slots 2u (M) and 2u + 1 (V), F's
is slots 2u (V) and 2u + 1 (M)):
  L  M     QK(u) [K half, double buffered]; exp2 of S''(u-2), (per token) the row sums
           continuing F's chains and P'(u-2) to LDS under the QK MFMAs; PV(u-2) [V^T half]
           with F's P'(u-2) (per tensor with a slot table, then the row sums of both halves'
           P'(u-2)).
     V     decide u-1 with F's lane max (rare rescale of O, S''(u-1), l if held); S''(u)
           (centring, membership mask) and its lane max to LDS; K(u+2) and V(u-1) to LDS;
           loads of K(u+3), V(u), masks, the next id window and scales; the first K reads of
           QK(u+1).
  F  V     pending O (and l if held) rescale of decision u-2; S''(u-1), its lane max to LDS;
           decide u-1 with L's lane max (rare rescale of S''(u-1), O and l deferred until
           PV(u-2) is in); K(u+1) and V(u-2) to LDS, loads; the first K reads of QK(u).
     M     QK(u); exp2 of S''(u-1), (per token) its row-sum chains and P'(u-1) to LDS; PV(u-2)
           with its own P'(u-2) (per tensor without split tasks kept in VGPRs since its
           previous M-stage, otherwise read back from LDS) and L's (per tensor without a slot
           table, then the row sums of both halves' P'(u-2)).
A task's first and last two steps leave out the work of tiles outside 0 .. last.

Membership is per key, but a four-key block's bits are all set or all clear except at a
causal or length edge: the centring folds the mask into the block's addend (-m or -inf), and
a wave whose tile holds a mixed block (ballot) masks each value instead. Both waves of a pair
hold the same rows; per tensor L loads the membership words of both key halves and publishes
F's nibbles through LDS (one vector load less in F's V-stage). The block ids of four tiles sit
in one VGPR (lane l: tile t + (l >> 4), slot l & 15), reloaded every step; from its second step
F takes L's window of the same tiles through LDS (written in L's M-stage, read in F's next
one) instead of loading it.

Registers: two waves per SIMD leave 256 VGPRs and no AGPRs, and a VGPR spill would take
scratch. Lane ids are re-derived per task (and for the output, after the key loop) from mbcnt
over a mask LLVM cannot prove all ones, so no thread-id register and no prologue or output
constant lives through the key loops.

K and V come from the raw K/V (packed, or a pool through K3's pool blocks), one buffer per
four-token block from its own 64-bit address, so K/V (a pool) may exceed 4 GiB: its records end
the final partial block at the request's last token and are 0 for slots outside the compact
list (steps whose loads read whole tiles only, before a task's last tiles, skip the checks). A
K load reads one block whole (lane: token lane >> 4, 16-byte chunk lane & 15); a wave loads the
two blocks of its group's key half that it writes. A V load reads a dword (four dims) of one
token per lane: the lanes 2i and 2i + 1 load the same four dims of the group's half, of tokens
0 and 2 and of tokens 1 and 3 of each of the wave's four blocks, and trade half of them (one DPP
move per block), so each lane holds two dims of the four tokens, one V^T word per dim and block,
and writes one 16-byte chunk per dim. Per token the kernel reads the raw K/V scales of the
lane's scale key and takes log2(v_scale) and 1 / v_scale of the clamped V scale
(sparse_prefill_common._v_scale_terms). The transpose and the scale terms run in the M-stage
before the V-stage that uses them, in the MFMAs' issue slack: the V-stages set the slots.

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

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    SC_SYSTEM,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_at,
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
)
from .sparse_prefill_prepare import PARTIAL_FLOATS, SPLIT_PARTS, SPLIT_SLOTS

__all__ = ["launcher", "run"]

D, THREADS = 256, 512
HALF = 32 * D  # one group's half of a K tile (32 keys) or of a V^T tile (128 dims)
# V^T transpose of a lane pair (_v_words): v_perm selectors of the even lane, the odd lane's
# differ by XOR with (lane & 1) times SEL_FLIP (the kept and traded halves) and 2 * SEL_FLIP
# (the operands of the joined words).
SEL_TRADE, SEL_KEEP, SEL_DIM0, SEL_DIM1, SEL_FLIP = (
    0x07060302,
    0x05040100,
    0x06020400,
    0x07030501,
    0x02020202,
)
DPP_SWAP = 0xB1  # quad_perm [1, 0, 3, 2]: lanes 2i and 2i + 1 swap
# LDS: K halves of group g, buffer j at K_LDS + (2g + j) * HALF; V^T halves at V_LDS + g * HALF;
# per pair p: P' words of group g (16 B per lane), lane maxima of group g (4 B per lane), F's
# row-sum chains (16 B per lane, per token), the output scale 1 / l (4 B per lane), F's
# membership nibbles (4 B per lane, per tensor) and L's id window for F (4 B per lane, in the
# chains' area per tensor, in the nibbles' per token). A
# drained task's output rows reuse all of it (8 KiB per wave); a task's prologue stages its 128
# Q rows in the upper half (Q_LDS), which the waves read before their first V^T and exchange
# writes.
K_LDS, V_LDS = 0, 4 * HALF
Q_LDS = V_LDS
P_LDS = V_LDS + 2 * HALF
MX_LDS = P_LDS + 2 * 4096
CH_LDS = MX_LDS + 2 * 1024
INV_LDS = CH_LDS + 4096
MK_LDS = INV_LDS + 1024
LDS_COUNT = 0  # a split task's part count, read after every wave's last K read
LDS_BYTES = 65536
NEG_INF = -8388608  # FP32 -inf bits
# K operand reads a V-stage issues for the next M-stage (KPRE) and V operand reads issued ahead of
# their PV MFMA (VLEAD); exp-work steps per MFMA in QK (EXP_QK) and PV (EXP_PV). The V-stage's K
# reads go out at its end, when the M-stage's operand reads are done; all four SIMDs start QK
# together after the barrier, so the K reads taken out of QK thin the LDS burst at its start
# (six measured slower than four: the V-stage then waits longer for them before the barrier).
KPRE, VLEAD, EXP_QK, EXP_PV = 4, 4, 1, 2
# Per token V operand reads go out VLEAD_TOKEN MFMAs ahead: its M-stage's register peak is at the
# end of QK, where each read in flight holds four VGPRs (per tensor fewer would cost slot time).
VLEAD_TOKEN = 2
# Per token or with split tasks one K read fewer ahead of the M-stage: their VGPRs (the scale
# terms, the split's records) are scarcer.
KPRE_TIGHT = 1
# Steps of the V-stage's VALU work moved into an M-stage (the V^T transpose of the raw V loads,
# per token the V scale terms) per MFMA of its second half, in the MFMAs' issue slack.
EXTRA = 2
# Issue priority of a wave in its M-stage and in its V-stage.
M_PRIO, V_PRIO = 1, 0
# The order of a V-stage's LDS writes (_step_l, _step_f): they share the LDS with the M-stages'
# operand reads of all four SIMDs, which start together after the barrier: K reads at the start
# of QK, V^T reads at its end and in PV. Per tensor without split tasks the V^T write goes out at
# the V-stage's start, the K write (and the K loads that reuse its registers) after the leader's
# decision or the follower's other loads; both writes at the start, both after the decision, or
# either just ahead of the V-stage's last K reads cost 6-17% of the kernel time (a 12000-token
# prefill at H12, H6 and H3). Per token and with split tasks the K write stays ahead of the V^T
# write at the V-stage's start, where the later one did not measure faster (a 12000-token
# prefill at H12 per token; a 2048-token prefill of every causal block at H3, split tasks).
# Per tensor (_gen_exp_late): 12 exps on the QK MFMAs, the other 4 on PV MFMAs
# 0-3; the FP8 cvts of words 0-1 on the exp-free QK MFMAs, of words 2-3 on PV 4-7 (P' words
# published before PV 8). A V-stage issues its K reads for the next M-stage right after the
# centring, ahead of the lane max (and of F's decision).
# Per tensor l is the sum of the FP8 P' words PV multiplies (consistent with O), taken by the
# wave holding l with two sparse 16x16x64 MFMAs per M-stage, each over two K-steps of both
# groups' words (A: a constant pattern selecting each lane's row), into a four-word
# accumulator whose word 0 holds c0 times the sum over half of a row's keys in lane n and c1
# times the other half in lane n + 32; at the end of the task l = word 0 / c (lane n: c0, lane
# n + 32: c1), which the output's lane-32 exchange adds. No wave keeps row-sum chains.
# The sparse MFMA's B is four P' words (two K-steps; lane l: column l & 15, K-block l >> 4:
# 16 keys, groups of four = one word). Its A sits in the lanes of A rows 0, 8 (K-blocks 0, 2:
# P' rows 0-15) and 4, 12 (K-blocks 1, 3: rows 16-31; lane l: A row l & 15, K-block l >> 4),
# a lane's eight values all c0 (rows 0, 4) or c1 (rows 8, 12); D row r lands in word r & 3 of
# lane 16 (r >> 2) + column. The A word is also the sparse index (its low 16 bits: two 2-bit
# key positions per group of four, group g in bits 4g .. 4g + 3), so no VGPR holds the index:
# c0 = 2.0 (E4M3FNUZ 0x48) picks keys 0, 2 of groups 0, 2 (nibble 8) and 0, 1 of groups 1, 3
# (nibble 4); c1 = -52.0 (0xED) the others, 1, 3 (nibble D) and 2, 3 (nibble E).
RSUM_A = (0x48484848, 0xEDEDEDED - (1 << 32))  # c0 = 2.0, c1 = -52.0 (int32 words)
RSUM_C = (2.0, -52.0)
# Without a slot table the follower takes the row sums (with its PV of the same P' words, both
# groups' at hand from its M-stage's start) and holds l (env.l_g): the leader's M-stage (the
# slot's last arrival with the sums) and its VGPRs (the kernel's peak) shed them. With a slot
# table the follower holding l lengthens every task boundary by about 400 cycles, more than
# its M-stage gains, so the leader keeps them.
# The PV MFMAs that the two row-sum MFMAs follow.
RS_AT = (8, 10)
# A split part's record (sparse_prefill_prepare.PARTIAL_FLOATS): per wave its 64 O^T values per
# lane in MFMA order (block mb, register quad k: 16 bytes per lane), then m and l per pair (by
# the group holding l).
PARTIAL_BYTES = PARTIAL_FLOATS * 4
PARTIAL_ML = 128 * D * 4


class _Env:
    """Trace-time bundle of one task's per-lane values; the helpers below are plain Python."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _pin():
    prim.sched_barrier(0)


def _bperm(value, lane_index):
    return prim.bpermute(fx.Int32(lane_index * 4), fx.Int32(value))


def _lds_ptr(storage, offset, width):
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4 * width),
        fx.get_iter(storage) + offset,
    )


def _write_lds(storage, offset, words):
    """ds_write_b32 / b64 / b128 of 1, 2 or 4 words at a byte offset of storage."""
    count = len(words)
    data = fx.make_rmem_tensor(count, fx.Int32)
    data.store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32))
    atom = {1: fx.UniversalCopy32b, 2: fx.UniversalCopy64b, 4: fx.UniversalCopy128b}[
        count
    ]
    fx.copy(
        fx.make_copy_atom(atom(), fx.Int32),
        data,
        fx.make_view(_lds_ptr(storage, offset, count), fx.make_layout(count, 1)),
    )


def _barrier():
    """Every LDS operation of this wave retired, then the CTA rendezvous."""
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()


def _barrier_reads(issue):
    """issue()'s LDS reads after the lane max's write, then the CTA rendezvous with the reads in
    flight: a ledger from the write counts them, and its wait for the write (LDS operations
    retire in issue order) covers every earlier LDS operation and none of the reads."""
    ledger = prim.LgkmLedger(1)
    for _ in issue():
        ledger.issue()
    ledger.wait(0)
    _stage_end()


# ---- LDS layouts ----
# K half tile: 32 key rows of 256 bytes, 16-byte chunks XOR-swizzled by the key.
# V^T half tile: 128 dim rows of 64 key bytes; dim bits 0 and 1 swapped in the row and the
# 16-byte chunk XOR-swizzled, so the b128 writes (eight lanes over dims 16a + 4p + 2b + x) and the
# b128 operand reads (eight lanes over consecutive dims) both hit eight distinct bank groups.
# Chunk 2h + q of a dim holds the tokens of blocks 8q + h + 0, 2, 4, 6 (PV K-steps 2q, 2q + 1).


def _swz_k(key):
    return (key & 3) | ((key >> 2) & 4)


def _k_offset(key, chunk):
    return key * 256 + ((chunk ^ _swz_k(key)) << 4)


def _vt_offset(d, chunk):
    row = (d & ~3) | ((d & 1) << 1) | ((d >> 1) & 1)
    sig = ((d ^ (d >> 4)) & 1) | ((((d >> 2) ^ (d >> 5)) & 1) << 1)
    return row * 64 + ((chunk ^ sig) << 4)


def _k_read(env, buf, c):
    """K operand words of the group's keys (lane n), dims 32c + 16h .. + 15, of K buffer buf."""
    # Chunks 2c + h of c >= 4 sit 128 bytes past those of c - 4 (the swizzle keeps bit 3).
    return prim._load_lds(
        env.lds, env.k_addr[c & 3], env.k_lds + buf * HALF + 128 * (c >> 2)
    )


def _v_read(env, mb, q):
    """V^T operand of the group's dims 32mb + lane: words of PV K-steps 2q, 2q + 1."""
    return prim._load_lds(env.lds, env.v_addr[mb & 1][q], env.v_lds + 2048 * mb)


# ---- Global loads ----


def _load_win(env, t):
    """Block ids of key tiles t .. t + 3 in one VGPR: lane l holds slot l & 15 of tile
    t + (l >> 4) (tiles clamped to the last; slots past count read 0; with a slot table pool
    blocks). t is wave-uniform; taken through readfirstlane, LLVM keeps t + (l >> 4) a
    per-step add instead of hoisting a VGPR of (l >> 4) + k per offset k out of the key loop.
    """
    t = prim._uniform(t)
    tile = _min(t + (env.lane >> 4), env.last)
    return _buffer_words(env.gblocks, (tile * 16 + (env.lane & 15)) * 4, 1)[0]


def _fix_win(env, win, t):
    """Slots past the compact list take a block past the scales' extent, so their scale loads
    read 0 without per-load checks (t through readfirstlane as in _load_win)."""
    slot = _min(prim._uniform(t) + (env.lane >> 4), env.last) * 16 + (env.lane & 15)
    return (slot < env.count).select(fx.Int32(win), env.id_oob)


def _load_mask(env, t):
    """Membership words of the lane's query over key tile t (quarters 0-1, 2-3)."""
    w = _buffer_words(env.gmask, env.mask_lane + _min(t, env.last) * env.mask_step, 2)
    return [w[0], w[1]]


def _kv_block(env, base, win, t, d, slot):
    """The buffer of block slot of key tile t + d (window win holds tiles t .. t + 3) from the
    block's own 64-bit address (K/V may exceed 4 GiB): the final partial block ends at the
    request's end, and slots outside the compact list (tiles past the last or before the
    first) read 0. In a step of whole tiles (env.full) every block is whole."""
    block = prim._uniform(prim.readlane(fx.Int32(win), slot + 16 * d))
    address = base + fx.Int64(fx.Uint32(block)) * (4 * env.tok_bytes)
    whole = fx.Int32(4 * env.tok_bytes)
    if env.full:
        return _buffer_at(address, whole)
    index = (prim._uniform(t) + d) * 16 + slot
    records = (block == env.final).select(env.partial, whole)
    # One unsigned compare: a negative index (tile -1) is past the count too.
    inside = fx.Uint32(index) < fx.Uint32(env.count)
    return _buffer_at(address, inside.select(records, fx.Int32(0)))


def _k_loads(env, win, t, d):
    """K of the wave's two blocks of the group's key half of tile t + d (window win from
    tile t), each read whole: 16 bytes per lane (token lane >> 4, chunk lane & 15)."""
    return [
        _buffer_words(
            _kv_block(env, env.k_base, win, t, d, env.kslot + j), env.k_lane, 4
        )
        for j in range(2)
    ]


def _v_loads(env, win, t):
    """The wave's four V blocks of key tile t (slots vslot + 0, 2, 4, 6; window win from tile
    t), the group's dims: per lane and block one dword (four dims) of tokens lane & 1 and
    2 + (lane & 1); _gen_v_words transposes them."""
    out = []
    for j in range(4):
        buffer = _kv_block(env, env.v_base, win, t, 0, env.vslot + 2 * j)
        words = [
            _buffer_words(buffer, env.v_lane + 2 * r * env.tok_bytes, 1)[0]
            for r in range(2)
        ]
        out.append(fx.Vector.from_elements(words, fx.Int32))
    return out


def _gen_v_words(env, vraw, out):
    """V^T words of the lane's two dims (16dd + 4pp + 2(lane & 1) + 0, 1) per block (to out)
    from the raw loads of a lane pair: the even lane holds tokens 0 and 2 of four dims, the
    odd one tokens 1 and 3; each keeps the two tokens of its dims, trades the others (one DPP
    move) and joins them (four tokens per word). A step per two VALU (MFMA issue slack).
    """
    flip = ((fx.Int32(env.lane) << 31) >> 31) & SEL_FLIP
    trade, keep = flip ^ SEL_TRADE, flip ^ SEL_KEEP
    yield
    dim0 = (flip << 1) ^ SEL_DIM0
    dim1 = dim0 + 0x01010101
    yield
    sent, kept = [], []
    for raw in vraw:
        raw = fx.Vector(raw)
        sent.append(fx.Int32(prim.perm_b32(raw[1], raw[0], trade)))
        kept.append(fx.Int32(prim.perm_b32(raw[1], raw[0], keep)))
        yield
    # The DPP moves trail the VALU writing their sources by more than two wait states.
    got = []
    for i in range(0, 4, 2):
        got += [prim.dpp_move(s, DPP_SWAP) for s in sent[i : i + 2]]
        yield
    for g, k in zip(got, kept):
        words = [fx.Int32(prim.perm_b32(g, k, sel)) for sel in (dim0, dim1)]
        out.append(fx.Vector.from_elements(words, fx.Int32))
        yield


def _scale_loads(env, fixed, d):
    """Raw (k_scale, v_scale) of the lane's scale key (per token): token lane & 3 of block sj of
    the group's half of the key tile at window position d (fixed: the window after _fix_win;
    slots past the list read 0, and with a slot table so do the tokens past the request in
    its final block)."""
    sid = _bperm(fixed, env.sj + 16 * d)
    offset = (sid * 4 + env.stok) * env.scale_step
    if env.slotted:
        partial = (sid == env.final) & (env.stok >= env.tail_len)
        offset = partial.select(env.scale_oob, offset)
    return [_as_f32(_buffer_words(buf, offset, 1)[0]) for buf in (env.gks, env.gvs)]


def _gen_terms(env, raw, out):
    """(k_scale, log2(v_scale), 1 / v_scale) of a lane's raw scales (to out; per tensor
    zeros)."""
    if not env.per_token:
        out.extend([fx.Float32(0.0)] * 3)
        return
    out.extend([fx.Float32(raw[0]), *_v_scale_terms(raw[1])])
    yield


# ---- LDS writes ----


def _k_write(env, kdata, buf):
    # The second block sits 1024 bytes past the first (the swizzle keeps key bits 2-3).
    for j in range(2):
        _write_lds(
            env.lds,
            env.k_wbase + 1024 * j + env.k_lds + buf * HALF,
            [fx.Vector(kdata[j])[x] for x in range(4)],
        )


def _v_write(env, vdata):
    """V^T rows of the lane's two dims: per dim one 16-byte chunk, the four tokens of each of
    the wave's blocks."""
    for x in range(2):
        words = [fx.Vector(vdata[j])[x] for j in range(4)]
        _write_lds(env.lds, env.v_wbase[x] + env.v_lds, words)


# ---- Softmax pieces ----


def _mask_nibbles(env, ws, b):
    """The lane's membership nibbles of key half b: byte a holds keys 8a + 4h .. + 3 (bit j:
    key 8a + 4h + j, one four-key block per byte); rows outside the task keep every
    key."""
    select = b | ((2 + b) << 8) | ((4 + b) << 16) | ((6 + b) << 24)
    picked = fx.Int32(prim.perm_b32(fx.Int32(ws[1]), fx.Int32(ws[0]), select))
    nibbles = (picked >> (env.h * 4)) & fx.Int32(0x0F0F0F0F)
    return env.valid.select(nibbles, fx.Int32(0x0F0F0F0F))


def _bit(word, position):
    """0 / -1 from bit position of word (sign-extended, v_bfe_i32)."""
    return (fx.Int32(word) << (31 - position)) >> 31


def _masked(value, keep):
    # Bit select: the value's bits or -inf, without a compare and VCC per value.
    bits = _as_i32(value)
    return _as_f32((bits & keep) | ((~keep) & fx.Int32(NEG_INF)))


def _centre_values(s, c_scale, neg, nibbles, sk, lsv, per_value, per_token):
    """S'' of the 16 values. per_value: each value masked by its bit; otherwise each four-key
    block's addend is -m or -inf (its bits all set or all clear), so the FMA masks it.
    """
    if per_value:
        addend = [neg] * 4
    else:
        addend = [_masked(neg, _bit(nibbles, 8 * a + 3)) for a in range(4)]
    vals = [None] * 16
    if per_token:
        # A DPP instruction reading a VGPR the previous VALU wrote needs two wait states (any
        # VGPR operand): the log2(v_scale) add of a value trails its FMA by one value.
        xs = {}
        for k in range(17):
            if k < 16:
                xs[k] = _fma(s[k] * prim.row_broadcast(sk, k), c_scale, addend[k >> 2])
            if k > 0:
                kk = k - 1
                x = xs.pop(kk) + prim.row_broadcast(lsv, kk)
                vals[kk] = (
                    _masked(x, _bit(nibbles, 8 * (kk >> 2) + (kk & 3)))
                    if per_value
                    else x
                )
    else:
        for k in range(16):
            x = _fma(s[k], c_scale, addend[k >> 2])
            vals[k] = (
                _masked(x, _bit(nibbles, 8 * (k >> 2) + (k & 3))) if per_value else x
            )
    return fx.Vector.from_elements(vals, fx.Float32)


@flyc.jit
def _centre_tile(s_acc, c_scale, neg, nibbles, sk, lsv, PER_TOKEN: fx.Constexpr[bool]):
    """S'' of the 16 values, block addends unless some lane has a mixed block (a causal or
    length edge) in the tile."""
    s = fx.Vector(s_acc)
    edge = (nibbles ^ (nibbles >> 1)) & fx.Int32(0x07070707)
    vals = s
    if prim.ballot(edge != fx.Int32(0)) != fx.Int64(0):
        vals = _centre_values(s, c_scale, neg, nibbles, sk, lsv, True, PER_TOKEN)
    else:
        vals = _centre_values(s, c_scale, neg, nibbles, sk, lsv, False, PER_TOKEN)
    return vals


def _centre(env, s_acc, m_run, nibbles, sc, early):
    """S'' = S * c - m of the group's 16 values (per token S * sk * c - m + log2 sv, sc the
    lane's scale entry), masked (nibbles: _mask_nibbles), and their lane max, written for the
    pair's other wave. Returns (S'', lane max). early(): issued between the centring and the
    lane max."""
    m_run = fx.Float32(m_run)
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    neg = fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run)
    vals = fx.Vector(
        _centre_tile(
            s_acc,
            env.c_scale,
            neg,
            nibbles,
            fx.Float32(sc[0]),
            fx.Float32(sc[1]),
            env.per_token,
        )
    )
    early()
    part = None
    for k in range(0, 16, 2):
        part = (
            _maximum(vals[k], vals[k + 1])
            if part is None
            else _maximum(_maximum(part, vals[k]), vals[k + 1])
        )
    _write_lds(env.lds, env.mx_wr, [_as_i32(part)])
    return vals, part


def _decide(mine, theirs, fresh):
    """(lane max of S'' over the row's keys in both halves, the lazy-max ballot): some lane
    (so some row) above LAZY, or a fresh row (no key yet, m = FLOOR, centred at 0) seeing its
    first key. Both waves of a pair compute the same ballot."""
    comb = _maximum(fx.Float32(mine), fx.Float32(theirs))
    # One compare against the row's bar: a fresh row's is -inf (its first key), above LAZY
    # implies above -inf.
    bar = fresh.select(fx.Float32(-math.inf), fx.Float32(LAZY))
    return comb, prim.ballot(comb > bar)


def _row_max(comb):
    return lambda: _maximum(comb, fx.gpu.shuffle_xor(comb, 32, 64))


def _decision(mine, theirs, scores, m_run):
    """The lazy-max decision of a tile from both waves' lane maxima: (S'' shifted, m, the
    factor of O and l, the ballot); O and l take the factor later (when the ballot is set).
    """
    m_run = fx.Float32(m_run)
    comb, ballot = _decide(mine, theirs, m_run == fx.Float32(_MAX_FLOOR))
    _, corr, scores, m_run = _lazy_max(
        [], fx.Float32(1.0), fx.Vector(scores), m_run, ballot, _row_max(comb)
    )
    return scores, m_run, corr, ballot


def _gen_exp(env, s, rsv, chains, res):
    """P' = exp2(S'') of the group's 16 values, then the FP8 words of the group's two PV
    K-steps; per tensor as _gen_exp_late plans them. Per token the exps come first, then a step
    per FP8 half word, each adding two row-sum terms P' / v_scale (two VALU: the V scale's DPP
    broadcast and the add) to the two chains (continued if given), one per chain."""
    p = [None] * 16
    if not env.per_token:
        yield from _gen_exp_late(s, p, res)
        return
    c = [None, None] if chains is None else list(chains)

    def add(k):
        term = p[k] * prim.row_broadcast(rsv, k)
        c[k & 1] = term if c[k & 1] is None else c[k & 1] + term

    for i in range(16):
        p[i] = _exp(s[i])
        yield
    words = []
    for st in range(2):
        for half in (0, 4):
            v = p[8 * st + half : 8 * st + half + 4]
            lo = prim.cvt_pk_fp8(v[0], v[1])
            add(8 * st + half)
            add(8 * st + half + 1)
            yield
            words.append(prim.cvt_pk_fp8(v[2], v[3], lo, True))
            add(8 * st + half + 2)
            add(8 * st + half + 3)
            yield
    res["chains"] = c
    res["words"] = words


def _gen_exp_late(s, p, res):
    """Per tensor, one step per MFMA: 12 exps on the QK MFMAs; on every fourth (which carries
    the next K read, no exp) one FP8 cvt of words 0 and 1 (QK 3, 7: word 0; 11, 15: word 1),
    so their FP32 values die within QK; exps 12-15 on PV MFMAs 0-3, then one cvt of words 2
    and 3 per PV MFMA 4-7 (the last step yields ("words", words) for the P' publication). The
    row sums come from the FP8 words (sparse MFMAs, _m_stage)."""
    lo, words = {}, []
    for j in range(16):
        if j % 4 != 3:
            i = j - (j >> 2)
            p[i] = _exp(s[i])
        else:
            w = j >> 3
            if j % 8 == 3:
                lo[w] = prim.cvt_pk_fp8(p[4 * w], p[4 * w + 1])
            else:
                words.append(prim.cvt_pk_fp8(p[4 * w + 2], p[4 * w + 3], lo[w], True))
        yield
    for i in range(12, 16):
        p[i] = _exp(s[i])
        yield
    for w in (2, 3):
        lo[w] = prim.cvt_pk_fp8(p[4 * w], p[4 * w + 1])
        yield
        words.append(prim.cvt_pk_fp8(p[4 * w + 2], p[4 * w + 3], lo[w], True))
        if w == 3:
            res["words"] = words
            yield ("words", words)
        else:
            yield


# ---- Stages ----


def _m_stage(
    env,
    buf,
    kpre,
    o,
    valu,
    pw_early,
    publish,
    qk,
    pv,
    publish_words=None,
    sums=None,
    extra=None,
):
    """QK over the group's 32 keys (K buffer buf; kpre: its first KPRE operand reads), then PV
    over the group's 128 dims (per block mb, K-steps 0..3 in order); valu steps after every
    MFMA, and publish() runs once it is drained. PV K-steps 0-1 take the P words pw_early[0:4]
    and K-steps 2-3 pw_early[4:8], or publish()'s words (or publish_words()'s, called when a
    step yields ("words", words)). sums: {"acc": the row-sum accumulator}; the row-sum MFMAs of
    K-steps 0-1 and 2-3 follow PV MFMAs RS_AT. extra: VALU work of the V-stage moved into the
    MFMAs' issue slack, EXTRA steps after each MFMA of the stage's second half."""
    seq = [("qk", j) for j in range(16)] if qk else []
    seq += [("pv", j) for j in range(16)] if pv else []
    npre = len(kpre)
    reads = list(kpre) + [None] * (8 - npre) if qk else None
    pw = list(pw_early) + [None] * (8 - len(pw_early))
    base = 16 if qk else 0
    due = sorted(
        ((base + 8 * q + mb - env.vlead, (mb, q)) for q in range(2) for mb in range(4)),
        key=lambda x: x[0],
    )
    vreads = {}
    s_acc = fx.Vector.filled(16, 0.0, fx.Float32)
    o = list(o)
    state = {"published": publish is None}
    extra = iter(()) if extra is None else extra

    def step():
        x = next(valu, StopIteration)
        if x is StopIteration:
            if not state["published"]:
                state["published"] = True
                words = publish()
                if words is not None:
                    pw[4:8] = words
            return False
        if isinstance(x, tuple) and publish_words is not None:
            words = publish_words(x[1])
            if words is not None:
                pw[4:8] = words
        return True

    def advance(count):
        for _ in range(count):
            step()
            _pin()

    def drain():
        while step():
            _pin()
        _pin()

    def advance_extra(count):
        for _ in range(count):
            if next(extra, StopIteration) is StopIteration:
                return
            _pin()

    for idx, (kind, j) in enumerate(seq):
        if pv:
            while (
                due
                and due[0][0] <= idx
                and not (kind == "qk" and j % 2 == 0 and j < 2 * (8 - npre))
            ):
                key = due.pop(0)[1]
                vreads[key] = _v_read(env, *key)
                _pin()
        if kind == "qk":
            c, e = j >> 1, j & 1
            if e == 0 and c + npre < 8:
                reads[c + npre] = _k_read(env, buf, c + npre)
                _pin()
            kw, q = reads[c], env.q_words[c]
            s_acc = _mfma_fp8(
                (kw[2 * e], kw[2 * e + 1]), (q[2 * e], q[2 * e + 1]), s_acc
            )
        else:
            # PV j is K-step st of block mb: it waits for its own V^T read (mb, st >> 1).
            st, mb = j >> 2, j & 3
            need = (mb, st >> 1)
            while need not in vreads and due:
                key = due.pop(0)[1]
                vreads[key] = _v_read(env, *key)
                _pin()
            if pw[2 * st] is None:
                drain()
            vw = vreads[need]
            e = st & 1
            o[mb] = _mfma_fp8(
                (vw[2 * e], vw[2 * e + 1]), (pw[2 * st], pw[2 * st + 1]), o[mb]
            )
            if sums is not None and j in RS_AT:
                # Row sums of K-steps 0, 1 (2, 3) (F has all words at hand from the M-stage's
                # start, L from PV 8 on).
                _pin()
                k = 2 * RS_AT.index(j)
                sums["acc"] = prim.smfmac_fp8(
                    (env.rs_a, env.rs_a), pw[2 * k : 2 * k + 4], sums["acc"], env.rs_a
                )
        _pin()
        advance(env.exp_qk if kind == "qk" else env.exp_pv)
        if 2 * idx >= len(seq):
            advance_extra(EXTRA)
    drain()
    advance_extra(64)
    return s_acc, o


def _row_env(env):
    """Per-pair LDS addresses of the exchanges."""
    lane, pair = env.lane, env.pair
    own, other = env.g, 1 - env.g
    env.p_wr = P_LDS + own * 4096 + pair * 1024 + lane * 16
    env.p_rd = P_LDS + other * 4096 + pair * 1024 + lane * 16
    env.mx_wr = MX_LDS + own * 1024 + pair * 256 + lane * 4
    env.mx_rd = MX_LDS + other * 1024 + pair * 256 + lane * 4
    env.ch = CH_LDS + pair * 1024 + lane * 16
    env.mk = MK_LDS + pair * 256 + lane * 4
    # The leader's id window for the follower: in the chains' area per tensor, in the nibbles'
    # per token.
    env.wn = (MK_LDS if env.per_token else CH_LDS) + pair * 256 + lane * 4


def _step_l(env, st, u, buf, qk, ex, dec, cen, kmem, vw, kn):
    """Leader step u. M-stage: QK(u) (qk); exp2 and P' of tile u-2 and PV(u-2) (ex; per token,
    or per tensor with a slot table, with the row sums).
    V-stage: decide u-1 (dec), S''(u) and its lane max (cen); K(u+2) / V(u-1) to LDS (kmem /
    vw), the loads of the next steps and the first K reads of QK(u+1) (kn; per tensor false
    in the steps no QK follows, where the reads are dead)."""
    lds = env.lds
    res = {}
    valu, publish, pw_early, publish_words = iter(()), None, [], None
    sums = {"acc": st["l"]} if env.rsum and ex and env.l_g == 0 else None
    if ex and env.rsum:
        pf = prim._load_lds(lds, env.p_rd)
        valu = _gen_exp(env, st["sp2"], st["rsv2"], None, res)
        pw_early = [pf[i] for i in range(4)]

        def publish_words(words):
            _write_lds(lds, env.p_wr, words)
            return list(words)

    elif ex:
        # Per token: F's two chain words continue into this wave's chains.
        ch = prim._load_lds(lds, env.ch)
        pf = prim._load_lds(lds, env.p_rd)
        valu = _gen_exp(
            env, st["sp2"], st["rsv2"], (_as_f32(ch[0]), _as_f32(ch[1])), res
        )
        pw_early = [pf[i] for i in range(4)]

        def publish():
            res["l"] = fx.Float32(st["l"]) + (res["chains"][0] + res["chains"][1])
            _write_lds(lds, env.p_wr, res["words"])
            return list(res["words"])

    # The V-stage's VALU work in the MFMAs' slack: V(u-1)^T from its raw loads and (per token)
    # the scale terms of tile u.
    vt, terms = [], []

    def extra():
        if vw:
            yield from _gen_v_words(env, st["vd"], vt)
        if cen:
            yield from _gen_terms(env, st["sc"], terms)
        if dec:
            # The id window of tiles u .. u + 3 for the follower's next V-stage (from step 1:
            # in step 0 the Q rows staged over this area may still be read).
            _write_lds(lds, env.wn, [st["win"]])
            yield

    prim.s_setprio(M_PRIO)
    s_new, o = _m_stage(
        env,
        buf,
        st["kpre"],
        st["o"],
        valu,
        pw_early,
        publish,
        qk,
        ex,
        publish_words,
        sums,
        extra(),
    )
    _barrier()
    prim.s_setprio(V_PRIO)
    if env.l_g != 0:
        l_run = fx.Float32(0.0)
    elif env.rsum:
        l_run = fx.Vector(sums["acc"] if sums else st["l"])
    else:
        l_run = res.get("l", fx.Float32(st["l"]))
    # ---- V-stage ----
    pm = _as_f32(prim._load_lds32(lds, env.mx_rd)) if dec else None
    # The window holds tiles u .. u + 3.
    win = st["win"]
    kd = st["kd"]
    if kmem and not env.k_mid:
        _k_write(env, kd, buf)
        kd = _k_loads(env, win, u, 3)
    if vw:
        _v_write(env, vt)
    vd = _v_loads(env, win, u)
    ws = _load_mask(env, u + 1)
    sc = st["sc"]
    if env.per_token:
        sc = _scale_loads(env, win if env.full else _fix_win(env, win, u), 1)
    nwin = _load_win(env, u + 1)
    _pin()
    m_run, sp1 = fx.Float32(st["m"]), st["sp1"]
    if dec:
        comb, ballot = _decide(st["mx1"], pm, m_run == fx.Float32(_MAX_FLOOR))
        if env.rsum and env.l_g == 0:
            # The row-sum accumulator scales like an O block.
            scaled, _, sp1, m_run = _lazy_max(
                list(o) + [l_run],
                fx.Float32(0.0),
                fx.Vector(sp1),
                m_run,
                ballot,
                _row_max(comb),
            )
            o, l_run = scaled[:4], scaled[4]
        else:
            o, l_run, sp1, m_run = _lazy_max(
                o, l_run, fx.Vector(sp1), m_run, ballot, _row_max(comb)
            )
    if kmem and env.k_mid:
        # K(u+2) to LDS between the M-stages' K and V^T reads (the V-stage's LDS write order is
        # described next to M_PRIO).
        _k_write(env, kd, buf)
        kd = _k_loads(env, win, u, 3)
        _pin()
    vals, local, rsv = st["sp1"], st["mx1"], st["rsv1"]
    kpre = []

    def issue_kpre():
        if kn:
            kpre.extend(_k_read(env, 1 - buf, c) for c in range(env.kpre_n))
            _pin()
        return kpre

    cross = env.kpre_cross and cen and kn
    if cen:
        early = (lambda: None) if cross else issue_kpre
        if not env.per_token:
            # The follower's nibbles of tile u (it centres S''(u) in the next slot).
            _write_lds(lds, env.mk, [_mask_nibbles(env, st["ws"], 0)])
        nibbles = _mask_nibbles(env, st["ws"], env.b)
        vals, local = _centre(env, s_new, m_run, nibbles, terms, early)
        rsv = terms[2]
    if cross:
        _barrier_reads(issue_kpre)
    else:
        if not kpre:
            issue_kpre()
        _barrier()
    return {
        "o": o,
        "sp2": sp1,
        "sp1": vals,
        "mx1": local,
        "l": l_run,
        "m": m_run,
        # Without K reads (no QK follows) the state keeps this step's in their place.
        "kpre": kpre if kn else st["kpre"],
        "kd": kd,
        "vd": vd,
        "ws": ws,
        "win": nwin,
        "sc": sc,
        "rsv2": st["rsv1"],
        "rsv1": rsv,
    }


def _step_f(env, st, u, buf, qk, ex, pv, dec, pend, kmem, vw):
    """Follower step u. V-stage: the pending O rescale of decision u-2 (pend); S''(u-1), its
    lane max, decide u-1 (dec); K(u+1) / V(u-2) to LDS (kmem / vw), loads. M-stage: QK(u) (qk);
    exp2 and P' of tile u-1 (ex); PV(u-2) and, holding l, its row sums (pv)."""
    lds = env.lds
    o = list(st["o"])
    acc = st["l"]
    if pend:
        if env.l_g != 0:
            # The row-sum accumulator takes O's pending factor (same P' words, same PVs).
            o = list(_scale_pending(o + [acc], st["corr"], st["pend"]))
            o, acc = o[:4], o[4]
        else:
            o = list(_scale_pending(o, st["corr"], st["pend"]))
    pm = _as_f32(prim._load_lds32(lds, env.mx_rd)) if dec else None
    # Per tensor the membership nibbles of tile u - 1 come from the leader (written in the
    # previous slot).
    nibbles = None
    if dec and not env.per_token:
        nibbles = prim._load_lds32(lds, env.mk)
    # The window holds tiles u - 1 .. u + 2.
    win = st["win"]
    kd = st["kd"]
    if kmem and not env.k_mid:
        _k_write(env, kd, 1 - buf)
        kd = _k_loads(env, win, u - 1, 3)
    if vw:
        _v_write(env, st["vd"])
    vd = _v_loads(env, win, u - 1)
    ws = _load_mask(env, u) if env.per_token else st["ws"]
    sc = [fx.Float32(0.0)] * 2
    if env.per_token:
        sc = _scale_loads(env, win if env.full else _fix_win(env, win, u - 1), 1)
    # Steps that decide get the next window from the leader (_step_l), the others load it.
    nwin = [] if dec else [_load_win(env, u)]
    _pin()
    if kmem and env.k_mid:
        # K(u+1) to LDS between the M-stages' K and V^T reads (the V-stage's LDS write order is
        # described next to M_PRIO).
        _k_write(env, kd, 1 - buf)
        kd = _k_loads(env, win, u - 1, 3)
        _pin()
    m_run, corr, ballot = fx.Float32(st["m"]), fx.Float32(1.0), fx.Int64(0)
    sp, rsv = None, None
    kpre = []

    def issue_kpre():
        if qk:
            kpre.extend(_k_read(env, buf, c) for c in range(env.kpre_n))
            _pin()
        return kpre

    cross = env.kpre_cross and dec and qk
    if dec:
        early = (lambda: None) if cross else issue_kpre
        if nibbles is None:
            nibbles = _mask_nibbles(env, st["ws"], env.b)
        vals, local = _centre(env, st["sacc"], m_run, nibbles, st["sc"], early)
        rsv = st["sc"][2]
        sp, m_run, corr, ballot = _decision(local, pm, vals, m_run)
    if cross:
        _barrier_reads(issue_kpre)
    else:
        if not kpre:
            issue_kpre()
        _barrier()
    # ---- M-stage ----
    prim.s_setprio(M_PRIO)
    res = {}
    pw_early = []
    if pv and env.keep_pw:
        # Its own P' words of tile u - 2 (kept from the previous M-stage), then the leader's.
        pl = prim._load_lds(lds, env.p_rd)
        pw_early = list(st["pwo"]) + [pl[i] for i in range(4)]
    elif pv:
        # Its own P' words of tile u - 2 (published in the previous M-stage), then the leader's.
        own = prim._load_lds(lds, env.p_wr)
        pl = prim._load_lds(lds, env.p_rd)
        pw_early = [own[i] for i in range(4)] + [pl[i] for i in range(4)]
    valu, publish, publish_words = iter(()), None, None
    if ex:
        valu = _gen_exp(env, fx.Vector(sp), rsv, None, res)
        if not env.per_token:

            def publish_words(words):
                _write_lds(lds, env.p_wr, words)

        else:

            def publish():
                c0, c1 = res["chains"]
                _write_lds(
                    lds, env.ch, [_as_i32(c0), _as_i32(c1), _as_i32(c0), _as_i32(c1)]
                )
                _write_lds(lds, env.p_wr, res["words"])

    sums = {"acc": acc} if env.l_g != 0 and pv else None
    # The next V-stage's VALU work in the MFMAs' slack: V^T of this step's V loads and (per
    # token) the scale terms of this step's scale loads.
    vt, terms = [], []

    def extra():
        yield from _gen_v_words(env, vd, vt)
        yield from _gen_terms(env, sc, terms)
        if dec:
            # The next V-stage's id window (tiles u .. u + 3), written by the leader's M-stage
            # in the previous slot: one vector load less in the follower's V-stage.
            nwin.append(prim._load_lds32(lds, env.wn))
            yield

    s_new, o = _m_stage(
        env, buf, kpre, o, valu, pw_early, publish, qk, pv, publish_words, sums, extra()
    )
    _barrier()
    prim.s_setprio(V_PRIO)
    return {
        "o": o,
        "l": sums["acc"] if sums else acc,
        "pwo": list(res["words"]) if ex and env.keep_pw else st["pwo"],
        "sacc": s_new if qk else st["sacc"],
        "m": m_run,
        "corr": corr,
        # A decision before the follower's first PV leaves no O rescale pending (O is still 0).
        "pend": (
            fx.Int32(0)
            if dec and not pv
            else (ballot != fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
        ),
        "kd": kd,
        "vd": vt,
        "ws": ws,
        "win": nwin[0],
        "sc": terms,
    }


@flyc.jit
def _scale_pending(o, corr, pend):
    """O times the factor of F's previous lazy-max move (rare)."""
    o = tuple(o)
    if pend != fx.Int32(0):
        o = tuple(fx.Vector(x) * corr for x in o)
    return o


# ---- Output ----


def _store_rows(env, o, inv, early=False):
    """O * inv -> BF16 rows of the group's 128 dims through the wave's 8 KiB of LDS, then
    16-byte row stores; with early all eight LDS row reads go out before the first store.
    """
    base = env.wave * 8192
    n, h = env.n, env.h
    for mb in range(4):
        for k in range(4):
            words = _scaled_bf16([fx.Vector(o[mb])[4 * k + r] for r in range(4)], inv)
            chunk = 4 * mb + k
            offset = (
                base + n * 256 + ((chunk ^ (n & 7)) << 4) + ((h ^ ((n >> 3) & 1)) << 3)
            )
            _write_lds(env.lds, offset, (words[0], words[1]))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    if early:
        lane_r, lane_c = env.lane >> 4, env.lane & 15
        reads = []
        for i in range(8):
            r = 4 * i + lane_r
            reads.append(
                prim._load_lds(env.lds, base + r * 256 + ((lane_c ^ (r & 7)) << 4))
            )
    for i in range(8):
        r, c = 4 * i + (env.lane >> 4), env.lane & 15
        if early:
            w = reads[i]
        else:
            w = prim._load_lds(env.lds, base + r * 256 + ((c ^ (r & 7)) << 4))
        # Bit 3 of r = 4i + (lane >> 4) is bit 1 of i: the half swap is static.
        swap = (i >> 1) & 1
        words = fx.Vector.from_elements([w[j ^ (2 * swap)] for j in range(4)], fx.Int32)
        row = env.row_offset + env.pair * 32 + r
        query, head = row // env.gp, row % env.gp
        valid = (query < env.qvalid) & (query < env.bq) & (head < env.group)
        offset = (query * env.heads + head) * (D * 2) + env.g * 256 + c * 16
        _buffer_store_words(words, env.gout, valid.select(offset, env.q_oob))


def _inverse(total, v_scale):
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    return inv if v_scale is None else inv * v_scale


def _store_whole(env, o, l_run, m_run, v_scale, residual):
    """O / l (times the V scale) as BF16; the group holding l (env.l_g) passes 1 / l to the
    other through LDS. With residual also the rows' log-sum-exp m + log2(l) of S * c (log2
    units), which the residual direct merges with. The lane addresses are derived here, not
    kept live through the key loop."""
    inv_at = INV_LDS + env.pair * 256 + env.lane * 4
    if env.g == env.l_g:
        total = l_run + fx.gpu.shuffle_xor(l_run, 32, 64)
        inv = _inverse(total, v_scale)
        _write_lds(env.lds, inv_at, [_as_i32(inv)])
        if residual:
            row = env.row_offset + env.pair * 32 + env.n
            query, head = row // env.gp, row % env.gp
            valid = (query < env.qvalid) & (query < env.bq) & (head < env.group)
            lse = fx.Float32(m_run) + prim.log2(total)
            _buffer_store_words(
                fx.Vector.from_elements([_as_i32(lse)], fx.Int32),
                env.glse,
                (valid & (env.h == 0)).select(
                    (query * env.heads + head) * 4, env.lse_end
                ),
            )
    _barrier()
    if env.g != env.l_g:
        inv = _as_f32(prim._load_lds32(env.lds, inv_at))
    # Every wave read 1 / l before the output rows reuse LDS.
    _barrier()
    # Per tensor with a slot table, no split tasks and no residual merge the row reads go out
    # before the stores (the volatile LDS reads otherwise keep their order around the stores,
    # one read's latency per store). Without a slot table that measured slower (a 12000-token
    # prefill per tensor: +0.6%), with a residual merge mixed (the same prefill through a slot
    # table at H6: -0.3% and +0.8% in two runs).
    _store_rows(env, o, inv, env.k_mid and env.slotted and not residual)


def _partial_offset(env, part):
    return (env.slot * SPLIT_PARTS + part) * PARTIAL_BYTES


def _ml_offset(env, part):
    return _partial_offset(env, part) + PARTIAL_ML + env.pair * 512 + env.lane * 4


def _store_partial(env, o, m_run, l_run):
    """This part's O^T fragments of the wave, and m and lane l of the group holding l
    (device-scope stores: the merging CTA may run on another XCD)."""
    lane_o = _partial_offset(env, env.part) + env.wave * 16384 + env.lane * 16
    for mb in range(4):
        for k in range(4):
            words = [_as_i32(fx.Vector(o[mb])[4 * k + r]) for r in range(4)]
            _buffer_store_words(
                fx.Vector.from_elements(words, fx.Int32),
                env.gpart,
                lane_o + (4 * mb + k) * 1024,
                SC_SYSTEM,
            )
    if env.g == env.l_g:
        for i, value in enumerate((m_run, l_run)):
            _buffer_store_words(
                fx.Vector.from_elements([_as_i32(value)], fx.Int32),
                env.gpart,
                _ml_offset(env, env.part) + 256 * i,
                SC_SYSTEM,
            )


def _merge_out(env, o, m_run, l_run, v_scale):
    """The part counting last: its O and m in registers, the other part's record loaded (all
    loads in flight at once; the group not holding l also loads its own part's l, which the
    other stored). O = O0 * 2^(m0 - m) + O1 * 2^(m1 - m) with m the larger (part 0 first, so
    either part computes the same output), l likewise; then the output of an unsplit task.
    """
    other = SPLIT_PARTS - 1 - env.part
    lane_o = _partial_offset(env, other) + env.wave * 16384 + env.lane * 16
    loaded = [
        _buffer_words(env.gpart, lane_o + (4 * mb + k) * 1024, 4, SC_SYSTEM)
        for mb in range(4)
        for k in range(4)
    ]
    m_other = _as_f32(_buffer_words(env.gpart, _ml_offset(env, other), 1, SC_SYSTEM)[0])
    l_other = _as_f32(
        _buffer_words(env.gpart, _ml_offset(env, other) + 256, 1, SC_SYSTEM)[0]
    )
    if env.g != env.l_g:
        l_run = _as_f32(
            _buffer_words(env.gpart, _ml_offset(env, env.part) + 256, 1, SC_SYSTEM)[0]
        )
    first = env.part == 0
    m0, m1 = first.select(m_run, m_other), first.select(m_other, m_run)
    top = _maximum(m0, m1)
    # A part without a key of the row has m = FLOOR: weight 0 (both: 1, l = 0, output 0).
    w0, w1 = _exp(m0 - top), _exp(m1 - top)
    l0, l1 = first.select(l_run, l_other), first.select(l_other, l_run)
    l_sum = _fma(l1, w1, l0 * w0)
    inv = _inverse(l_sum + fx.gpu.shuffle_xor(l_sum, 32, 64), v_scale)
    merged = []
    for mb in range(4):
        values = []
        for i in range(16):
            mine = fx.Float32(fx.Vector(o[mb])[i])
            theirs = _as_f32(loaded[4 * mb + i // 4][i % 4])
            a, b = first.select(mine, theirs), first.select(theirs, mine)
            values.append(_fma(b, w1, a * w0))
        merged.append(fx.Vector.from_elements(values, fx.Float32))
    _store_rows(env, merged, inv)


# ---- Lanes, tasks ----


def _lane_env(lds, per_token, slotted, split, hk, g, wave, zero):
    """Per-lane constants every task shares, for group g (0: leader, 1: follower). zero is a
    task scalar LLVM cannot fold (0 at run time): it keeps the constants of the task's
    prologue and output inside the task instead of hoisted out of the persistent loop, live
    (spilled) across every key loop."""
    # The lane id by mbcnt over a mask LLVM cannot fold: no thread-id VGPR lives through the
    # persistent loop.
    lane = prim.lane_id(fx.Int32(-1) + zero)
    pair = wave & 3
    n, h = lane & 31, lane >> 5
    b = 1 - g  # S block (key half) the group scores
    # K half writes: the wave's two blocks 2 pair + j of the half, token lane >> 4, 16-byte chunk
    # lane & 15.
    key = pair * 8 + (lane >> 4)
    chunk = lane & 15
    # V^T half writes: wave (vq, vh) = (pair >> 1, pair & 1) fills chunk 2vh + vq of the group's
    # dims; a lane carries dims 16dd + 4pp + 2hf + 0, 1 of the wave's four blocks.
    vq, vh = pair >> 1, pair & 1
    pp, dd, hf = lane >> 4, (lane >> 1) & 7, lane & 1
    tok_bytes = hk * D
    return {
        "lds": lds,
        "zero": zero,
        "lane": lane,
        "wave": wave,
        "pair": pair,
        "n": n,
        "h": h,
        "g": g,
        "b": b,
        "per_token": per_token,
        # MFMA row sums (per tensor); the group holding l: F for them without a slot table,
        # otherwise L (lsum_l: L's MFMA row sums).
        "rsum": not per_token,
        "l_g": 0 if per_token or slotted else 1,
        "lsum_l": slotted and not per_token,
        "rs_a": (((lane & 23) == 0) | ((lane & 23) == 20)).select(
            ((lane & 8) == 0).select(fx.Int32(RSUM_A[0]), fx.Int32(RSUM_A[1])),
            fx.Int32(0),
        ),
        "kpre_n": KPRE_TIGHT if per_token or split > 1 else KPRE,
        # Per tensor without split tasks the K write goes to the V-stage's middle (the V-stage's
        # LDS write order is described next to M_PRIO).
        "k_mid": not per_token and split == 1,
        # Per tensor without split tasks the follower keeps its own P' words in VGPRs from the
        # M-stage that converts them to the one whose PV multiplies them; otherwise it reads
        # them back from LDS (keeping them did not measure faster per token or with split
        # tasks: a 12000-token prefill at H12 per token, a 2048-token prefill of every causal
        # block at H3).
        "keep_pw": not per_token and split == 1,
        # Per tensor with a slot table a V-stage's K reads follow the lane max's write and
        # stay in flight across the barrier (_barrier_reads), so the V-stage need not wait for
        # them; the buffer they read is rewritten two barriers later at the earliest. Without
        # a slot table the next M-stage, which then waits for them, sets the slot (about +1%).
        "kpre_cross": not per_token and split == 1 and slotted,
        # Split tasks per token finish the exp work within QK (SGPR spills need a VGPR).
        "exp_qk": EXP_QK + 1 if per_token and split > 1 else EXP_QK,
        # Per tensor the generator plans one step per PV MFMA.
        "exp_pv": EXP_PV if per_token else 1,
        "vlead": VLEAD_TOKEN if per_token else VLEAD,
        "slotted": slotted,
        # A step whose K/V loads read whole tiles only (constant buffer records).
        "full": False,
        "tok_bytes": tok_bytes,
        "scale_step": hk * 4,
        "stok": lane & 3,
        "k_lds": K_LDS + 2 * g * HALF,
        "v_lds": V_LDS + g * HALF,
        "k_addr": [_k_offset(n, 2 * c + h) for c in range(4)],
        "v_addr": [
            [_vt_offset(n + 32 * odd, 2 * h + q) - 2048 * odd for q in range(2)]
            for odd in range(2)
        ],
        "k_wbase": _k_offset(key, chunk),
        "k_lane": (lane >> 4) * tok_bytes + chunk * 16,
        "kslot": 8 * b + 2 * pair,
        "v_wbase": [
            _vt_offset(16 * dd + 4 * pp + 2 * hf + x, 2 * vh + vq) for x in range(2)
        ],
        "vslot": 8 * vq + vh,
        "v_lane": hf * tok_bytes + g * 128 + dd * 16 + pp * 4,
        "sj": 2 * ((lane & 15) >> 2) + h + 8 * b,
    }


def _relane(env, win):
    """The lane id for the steps after the key loop, re-derived from the loop's id window (a
    block id is below 2**31: 0 at run time, opaque to LLVM): the window offsets of those
    steps' loads are computed after the loop instead of kept (spilled) across it."""
    env.lane = prim.lane_id(fx.Int32(-1) + (prim._uniform(fx.Int32(win)) >> 31))


def _out_env(env, zero):
    """env with the output's buffers and its lane ids re-derived from zero (0 at run time,
    opaque to LLVM): their SGPRs and lane addresses start after the key loop instead of living
    through it."""
    out = _Env(**env.__dict__)
    T, q_rows, q_base = env.T, env.q_rows, env.q_base
    q8_extent = q_rows * D
    # Row log-sum-exp for the residual direct merge (QS layout), from the first half.
    out.glse = _buffer(
        fx.make_view(fx.get_iter(T.LSE) + q_base, fx.make_layout(q_rows, 1)), q_rows * 4
    )
    out.lse_end = fx.Int32(q_rows * 4)
    out.gout = _buffer(
        fx.make_view(fx.get_iter(T.OUT) + q_base * D, fx.make_layout(q8_extent, 1)),
        q8_extent * 2,
    )
    out.q_oob = fx.Int32(q8_extent * 2)
    out.gpart = _buffer(T.PARTIALS, SPLIT_SLOTS * SPLIT_PARTS * PARTIAL_BYTES)
    out.lane = prim.lane_id(fx.Int32(-1) + zero)
    out.n, out.h = out.lane & 31, out.lane >> 5
    return out


def _env_for(T, lds, wave, mapped, enabled, dims, flags):
    """Lane constants and env of order entry mapped (enabled 0: an idle slot); dims = (H, HK,
    NQ, CAP, POOL, BQ, GP), flags = (per_token, slotted, split, g)."""
    H, HK, NQ, CAP, POOL, BQ, GP = dims
    per_token, slotted, split, g = flags
    sc = _task_scalars(T, mapped, enabled, HK, BQ, GP, slotted, split)
    # A block count is below 2**31.
    zero = fx.Int32(sc["count"]) >> 31
    lane = _lane_env(lds, per_token, slotted, split, HK, g, wave, zero)
    return _task_env(lane, T, sc, H, HK, NQ, CAP, POOL, BQ, GP)


def _sload(tensor, index):
    """One int32 of a tensor at a wave-uniform index by a scalar load (SMEM, lgkmcnt): it does
    not wait behind the vector memory counter, which also counts the output stores."""
    # readfirstlane marks the result uniform for LLVM's divergence analysis (no-op on an SGPR):
    # descriptors and the persistent loop built from it stay scalar.
    return prim._uniform(fx.rocdl.s_buffer_load(_buffer(tensor, 0x7FFFFFFF), index))


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


def _task_env(lane, T, sc, H, HK, NQ, CAP, POOL, BQ, GP):
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
    n = lane["n"]
    row = row_offset + lane["pair"] * 32 + n
    query, head = row // GP, row % GP
    valid = (query < qvalid) & (query < BQ) & (head < group)
    # Byte offsets are unsigned 32-bit: Q and the scales (each request's or the pool's) stay below
    # 4 GiB; K/V are read per block (_kv_block).
    q_rows = fx.Uint32(NQ - q0) * H - hkv * group
    q_base = fx.Int64(q0) * H + hkv * group
    q8_extent = q_rows * D
    q8_oob = fx.Int32(q8_extent)
    tok_bytes = lane["tok_bytes"]
    if fx.const_expr(lane["slotted"]):
        kv_rows = POOL
        kv_offset = fx.Int64(hkv * D)
        scale_extent = (fx.Uint32(POOL) * HK - hkv) * 4
        scale_offset = fx.Int64(hkv)
        final = (sc["tail"] >= 0).select(sc["tail"] >> 2, fx.Int32(-1))
    else:
        kv_rows = kv_len
        kv_offset = fx.Int64(k0) * (HK * D) + hkv * D
        scale_extent = (fx.Uint32(kv_len) * HK - hkv) * 4
        scale_offset = fx.Int64(k0) * HK + hkv
        # Request-local blocks: a final partial block is kv_len >> 2.
        final = kv_len >> 2
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
        q8_oob=q8_oob,
        gqs=_buffer(
            fx.make_view(fx.get_iter(T.QS) + q_base, fx.make_layout(q_rows, 1)),
            q_rows * 4,
        ),
        qs_lane=valid.select((query * H + head) * 4, fx.Int32(q_rows * 4)),
        # The output's buffers are made after the key loop (_out_env), from these.
        T=T,
        q_rows=q_rows,
        q_base=q_base,
        k_base=fx.Int64(fx.ptrtoint(fx.get_iter(T.K))) + kv_offset,
        v_base=fx.Int64(fx.ptrtoint(fx.get_iter(T.V))) + kv_offset,
        partial=(kv_len & 3) * tok_bytes,
        # First block wholly past the scales' extent. The out-of-range offsets take the task's
        # opaque zero: their VGPR copies (a select's operand) stay inside the task instead of
        # living (spilled) across the key loop.
        id_oob=((kv_rows + 3) >> 2) + lane["zero"],
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
        scale_oob=fx.Int32(scale_extent) + lane["zero"],
        row_offset=row_offset,
        qvalid=qvalid,
        bq=BQ,
        gp=GP,
        group=group,
        heads=H,
    )
    if "split" in sc:
        env.split, env.slot, env.part = sc["split"], sc["slot"], sc["part"]
    return env


def _q_rows(env, i):
    """The Q row of the wave's coalesced load i: 16 rows per wave, four 256-byte rows per load
    (lane: row 4i + (lane >> 4), 16-byte chunk lane & 15)."""
    return 16 * env.wave + 4 * i + (env.lane >> 4)


def _q_st(r, j):
    """Q_LDS address of row r's 16-byte chunk j (chunks XOR-swizzled by the row)."""
    return Q_LDS + r * 256 + ((j ^ (r & 15)) << 4)


def _prefetch_q(env, g):
    """A task's first loads without a block id: the id window (the leader's holds tiles 0 .. 3,
    the follower's -1 .. 2, tile -1 reading 0: a V load nothing writes), the leader's mask(0),
    Q and its scales. The CTA loads its Q rows once and coalesced (each wave 16 rows, 1 KiB
    per load); the prologue stages them in LDS for every wave's operand reads (one load per row
    chunk instead of a 16-byte piece of 32 rows per load, twice)."""
    zero = fx.Int32(0)
    win = _load_win(env, fx.Int32(-g))
    q8 = []
    for i in range(4):
        row = env.row_offset + _q_rows(env, i)
        query, head = row // env.gp, row % env.gp
        valid = (query < env.qvalid) & (query < env.bq) & (head < env.group)
        offset = (query * env.heads + head) * D + (env.lane & 15) * 16
        q8.append(_buffer_words(env.gq8, valid.select(offset, env.q8_oob), 4))
    return {
        "q8": q8,
        "qs": _as_f32(_buffer_words(env.gqs, env.qs_lane, 1)[0]),
        "win": win,
        "ws": _load_mask(env, zero) if g == 0 else [fx.Int32(-1), fx.Int32(-1)],
    }


def _prefetch_k(env, pre, g):
    """The loads by block id: K(0), K(1) (the leader also K(2)) and the leader's scales(0)."""
    first = fx.Int32(-g)
    sc = [fx.Float32(0.0)] * 2
    if env.per_token and g == 0:
        sc = _scale_loads(env, _fix_win(env, pre["win"], first), 0)
    blank = [fx.Vector.filled(4, 0, fx.Int32)] * 2
    return {
        **pre,
        "sc": sc,
        "k0": _k_loads(env, pre["win"], first, g),
        "k1": _k_loads(env, pre["win"], first, g + 1),
        "k2": _k_loads(env, pre["win"], first, 2) if g == 0 else blank,
    }


def _flat_pre(pre):
    return (
        *(fx.Vector(x) for x in pre["q8"]),
        fx.Float32(pre["qs"]),
        fx.Int32(pre["win"]),
        fx.Vector.from_elements([fx.Int32(w) for w in pre["ws"]], fx.Int32),
        *(fx.Float32(x) for x in pre["sc"]),
        *(fx.Vector(x) for k in ("k0", "k1", "k2") for x in pre[k]),
    )


def _unflat_pre(a):
    return {
        "q8": [fx.Vector(x) for x in a[0:4]],
        "qs": fx.Float32(a[4]),
        "win": fx.Int32(a[5]),
        "ws": [fx.Vector(a[6])[i] for i in range(2)],
        "sc": [fx.Float32(x) for x in a[7:9]],
        "k0": [fx.Vector(x) for x in a[9:11]],
        "k1": [fx.Vector(x) for x in a[11:13]],
        "k2": [fx.Vector(x) for x in a[13:15]],
    }


def _prologue(env, pre, g):
    """A task's start from its prefetched loads: K(0) (the leader also K(1)) and the wave's Q
    rows to LDS."""
    _k_write(env, pre["k0"], 0)
    if g == 0:
        _k_write(env, pre["k1"], 1)
    for i, words in enumerate(pre["q8"]):
        _write_lds(
            env.lds,
            _q_st(_q_rows(env, i), env.lane & 15),
            [fx.Vector(words)[j] for j in range(4)],
        )
    return {
        "qs": pre["qs"],
        "kd": pre["k2"] if g == 0 else pre["k1"],
        "vd": [fx.Vector.filled(2, 0, fx.Int32) for _ in range(4)],
        **{k: pre[k] for k in ("ws", "win", "sc")},
    }


def _after_full(env):
    """The first step pair after the whole-tile loop (range(2, last - 4, 2)): the first even
    u >= max(2, last - 4)."""
    return 2 + 2 * ((_maximum_i(env.last - 6, fx.Int32(0)) + 1) >> 1)


def _maximum_i(a, b):
    return (a > b).select(a, b)


def _row_sums(env, l):
    """The task's lane l: with MFMA row sums word 0 of the accumulator, c0 (lane n) and c1
    (lane n + 32) times the halves of row n's keys, which the output's lane-32 exchange
    adds."""
    if not env.rsum:
        return l
    return fx.Vector(l)[0] * ((env.lane & 32) == 0).select(
        fx.Float32(1.0 / RSUM_C[0]), fx.Float32(1.0 / RSUM_C[1])
    )


def _flat_l(st):
    return (
        *st["o"],
        fx.Vector(st["sp2"]),
        fx.Vector(st["sp1"]),
        fx.Float32(st["mx1"]),
        st["l"] if isinstance(st["l"], fx.Vector) else fx.Float32(st["l"]),
        fx.Float32(st["m"]),
        fx.Vector.from_elements(
            [fx.Vector(r)[j] for r in st["kpre"] for j in range(4)], fx.Int32
        ),
        *(fx.Vector(x) for x in st["kd"]),
        *(fx.Vector(x) for x in st["vd"]),
        fx.Vector.from_elements([fx.Int32(w) for w in st["ws"]], fx.Int32),
        fx.Int32(st["win"]),
        *(fx.Float32(x) for x in st["sc"]),
        fx.Float32(st["rsv2"]),
        fx.Float32(st["rsv1"]),
    )


def _unflat_l(a, rsum=False):
    kp = fx.Vector(a[9])
    return {
        "o": [fx.Vector(x) for x in a[0:4]],
        "sp2": fx.Vector(a[4]),
        "sp1": fx.Vector(a[5]),
        "mx1": fx.Float32(a[6]),
        "l": fx.Vector(a[7]) if rsum else fx.Float32(a[7]),
        "m": fx.Float32(a[8]),
        "kpre": [
            fx.Vector.from_elements([kp[4 * r + j] for j in range(4)], fx.Int32)
            for r in range(kp.numel // 4)
        ],
        "kd": [fx.Vector(x) for x in a[10:12]],
        "vd": [fx.Vector(x) for x in a[12:16]],
        "ws": [fx.Vector(a[16])[i] for i in range(2)],
        "win": fx.Int32(a[17]),
        "sc": [fx.Float32(x) for x in a[18:20]],
        "rsv2": fx.Float32(a[20]),
        "rsv1": fx.Float32(a[21]),
    }


def _flat_f(st):
    return (
        *st["o"],
        fx.Vector(st["sacc"]),
        fx.Float32(st["m"]),
        fx.Float32(st["corr"]),
        fx.Int32(st["pend"]),
        *(fx.Vector(x) for x in st["kd"]),
        *(fx.Vector(x) for x in st["vd"]),
        fx.Vector.from_elements([fx.Int32(w) for w in st["ws"]], fx.Int32),
        fx.Int32(st["win"]),
        *(fx.Float32(x) for x in st["sc"]),
        fx.Vector(st["l"]) if isinstance(st["l"], fx.Vector) else fx.Float32(st["l"]),
        fx.Vector.from_elements([fx.Int32(w) for w in st["pwo"]], fx.Int32),
    )


def _unflat_f(a, rsum=False):
    return {
        "o": [fx.Vector(x) for x in a[0:4]],
        "sacc": fx.Vector(a[4]),
        "m": fx.Float32(a[5]),
        "corr": fx.Float32(a[6]),
        "pend": fx.Int32(a[7]),
        "kd": [fx.Vector(x) for x in a[8:10]],
        "vd": [fx.Vector(x) for x in a[10:14]],
        "ws": [fx.Vector(a[14])[i] for i in range(2)],
        "win": fx.Int32(a[15]),
        "sc": [fx.Float32(x) for x in a[16:19]],
        "l": fx.Vector(a[19]) if rsum else fx.Float32(a[19]),
        "pwo": [fx.Vector(a[20])[i] for i in range(4)],
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
    PARTIALS,
    COUNTERS,
    LSE,
    lds,
    wave,
    mapped,
    next_mapped,
    next_enabled,
    pre,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
    G: fx.Constexpr[int],
):
    """One task (128 (query, head) rows of a union tile over its compact key list, or one key
    range of it) for group G of the CTA; both groups take the same barriers. Without split
    the task starts from its prefetched first loads (pre) and returns the next order entry's,
    which its drain issues; a split CTA mostly runs one task, so it loads its own."""
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
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    dims, flags = (H, HK, NQ, CAP, POOL, BQ, GP), (PER_TOKEN, SLOTTED, SPLIT, G)
    env = _env_for(T, lds, wave, mapped, fx.Int32(1), dims, flags)
    _row_env(env)
    if fx.const_expr(SPLIT > 1):
        pro = _prologue(env, _prefetch_k(env, _prefetch_q(env, G), G), G)
    else:
        pro = _prologue(env, _unflat_pre(pre), G)
        # The next entry's scalars load under this task (its first loads issue in the drain).
        nxt = _env_for(T, lds, wave, next_mapped, next_enabled, dims, flags)
    c_scale = pro["qs"] * SCALE_LOG2
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(KS[0])
    env.c_scale = c_scale
    _barrier()
    # The QK B operands from the staged rows (row 32 pair + n, chunk 2c + h); every wave's reads
    # retire at its next barrier, ahead of the first V^T and exchange writes.
    row = env.pair * 32 + env.n
    env.q_words = [prim._load_lds(lds, _q_st(row, 2 * c + env.h)) for c in range(8)]
    zeros = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(4)]
    floor, zero = fx.Float32(_MAX_FLOOR), fx.Float32(0.0)
    key_tiles = env.last + 1
    v_scale = None if PER_TOKEN else fx.Float32(VS[0])

    if fx.const_expr(G == 0):
        state = _flat_l(
            {
                "o": zeros,
                "sp2": fx.Vector.filled(16, 0.0, fx.Float32),
                "sp1": fx.Vector.filled(16, 0.0, fx.Float32),
                "mx1": zero,
                "l": (
                    fx.Vector.filled(4, 0.0, fx.Float32)
                    if env.rsum and env.l_g == 0
                    else zero
                ),
                "m": floor,
                "kpre": [_k_read(env, 0, c) for c in range(env.kpre_n)],
                "rsv2": zero,
                "rsv1": zero,
                **{k: pro[k] for k in ("kd", "vd", "ws", "win", "sc")},
            }
        )

        @flyc.jit
        def step(
            o0,
            o1,
            o2,
            o3,
            sp2,
            sp1,
            mx1,
            l_run,
            m_run,
            kpre,
            kd0,
            kd1,
            vd0,
            vd1,
            vd2,
            vd3,
            ws,
            win,
            sk,
            sv,
            rsv2,
            rsv1,
            u,
            BUF: fx.Constexpr[int],
            QK: fx.Constexpr[bool],
            EX: fx.Constexpr[bool],
            DEC: fx.Constexpr[bool],
            CEN: fx.Constexpr[bool],
            KMEM: fx.Constexpr[bool],
            VW: fx.Constexpr[bool],
            KN: fx.Constexpr[bool],
            FULL: fx.Constexpr[bool] = False,
        ):
            env.full = FULL
            st = _unflat_l(
                (
                    o0,
                    o1,
                    o2,
                    o3,
                    sp2,
                    sp1,
                    mx1,
                    l_run,
                    m_run,
                    kpre,
                    kd0,
                    kd1,
                    vd0,
                    vd1,
                    vd2,
                )
                + (vd3, ws, win, sk, sv, rsv2, rsv1),
                env.lsum_l,
            )
            out = _flat_l(
                _step_l(env, st, fx.Int32(u), BUF, QK, EX, DEC, CEN, KMEM, VW, KN)
            )
            env.full = False
            return out

        state = step(
            *state, fx.Int32(0), 0, True, False, False, True, True, False, True
        )
        if key_tiles > 1:
            state = step(
                *state, fx.Int32(1), 1, True, False, True, True, True, True, True
            )
        # Steps u, u + 1 of the first loop load K and V of whole tiles only (u + 4 < last).
        for u in range(fx.Int32(2), env.last - 4, fx.Int32(2)):
            state = step(*state, u, 0, True, True, True, True, True, True, True, True)
            state = step(
                *state,
                fx.Int32(u) + 1,
                1,
                *(True,) * 8,
            )
        for u in range(_after_full(env), key_tiles - 1, fx.Int32(2)):
            state = step(*state, u, 0, True, True, True, True, True, True, True)
            state = step(
                *state, fx.Int32(u) + 1, 1, True, True, True, True, True, True, True
            )
        _relane(env, _unflat_l(state, env.lsum_l)["win"])
        # No QK follows the steps from here on: per tensor they issue no K reads (KN). Per
        # token dropping them measured -0.8% to +0.6% depending on the shape, so they stay.
        kn = env.per_token
        if (key_tiles > 2) & ((key_tiles & 1) == 1):
            state = step(*state, env.last, 0, True, True, True, True, True, True, kn)
        if key_tiles > 1:
            state = step(
                *state, key_tiles, 0, False, True, True, False, False, True, kn
            )
        else:
            state = step(
                *state, key_tiles, 0, False, False, True, False, False, True, kn
            )
        win_out = _unflat_l(state, env.lsum_l)["win"]
        if fx.const_expr(SPLIT == 1):
            nxt_pre = _prefetch_q(nxt, G)
        state = step(
            *state, key_tiles + 1, 0, False, True, False, False, False, False, kn
        )
        st = _unflat_l(state, env.lsum_l)
        l_run = _row_sums(env, st["l"]) if env.l_g == 0 else None
    else:
        state = _flat_f(
            {
                "o": zeros,
                "sacc": fx.Vector.filled(16, 0.0, fx.Float32),
                "m": floor,
                "corr": fx.Float32(1.0),
                "pend": fx.Int32(0),
                **{k: pro[k] for k in ("kd", "vd", "ws", "win")},
                # Scale terms of the next centring (from the previous M-stage).
                "sc": [zero] * 3,
                "l": fx.Vector.filled(4, 0.0, fx.Float32) if env.l_g != 0 else zero,
                "pwo": [fx.Int32(0)] * 4,
            }
        )

        @flyc.jit
        def step(
            o0,
            o1,
            o2,
            o3,
            sacc,
            m_run,
            corr,
            pend,
            kd0,
            kd1,
            vd0,
            vd1,
            vd2,
            vd3,
            ws,
            win,
            sk,
            lsv,
            rsv,
            l_acc,
            pwo,
            u,
            BUF: fx.Constexpr[int],
            QK: fx.Constexpr[bool],
            EX: fx.Constexpr[bool],
            PVS: fx.Constexpr[bool],
            DEC: fx.Constexpr[bool],
            PEND: fx.Constexpr[bool],
            KMEM: fx.Constexpr[bool],
            VW: fx.Constexpr[bool],
            FULL: fx.Constexpr[bool] = False,
        ):
            env.full = FULL
            st = _unflat_f(
                (
                    o0,
                    o1,
                    o2,
                    o3,
                    sacc,
                    m_run,
                    corr,
                    pend,
                    kd0,
                    kd1,
                    vd0,
                    vd1,
                    vd2,
                    vd3,
                    ws,
                )
                + (win, sk, lsv, rsv, l_acc, pwo),
                env.l_g != 0,
            )
            out = _flat_f(
                _step_f(env, st, fx.Int32(u), BUF, QK, EX, PVS, DEC, PEND, KMEM, VW)
            )
            env.full = False
            return out

        state = step(
            *state, fx.Int32(0), 0, True, False, False, False, False, True, False
        )
        if key_tiles > 1:
            state = step(
                *state, fx.Int32(1), 1, True, True, False, True, False, True, False
            )
        for u in range(fx.Int32(2), env.last - 4, fx.Int32(2)):
            state = step(*state, u, 0, True, True, True, True, True, True, True, True)
            state = step(
                *state,
                fx.Int32(u) + 1,
                1,
                *(True,) * 8,
            )
        for u in range(_after_full(env), key_tiles - 1, fx.Int32(2)):
            state = step(*state, u, 0, True, True, True, True, True, True, True)
            state = step(
                *state, fx.Int32(u) + 1, 1, True, True, True, True, True, True, True
            )
        _relane(env, _unflat_f(state, env.l_g != 0)["win"])
        if (key_tiles > 2) & ((key_tiles & 1) == 1):
            state = step(*state, env.last, 0, True, True, True, True, True, True, True)
        if key_tiles > 1:
            state = step(
                *state, key_tiles, 0, False, True, True, True, True, False, True
            )
        else:
            state = step(
                *state, key_tiles, 0, False, True, False, True, False, False, False
            )
        win_out = _unflat_f(state, env.l_g != 0)["win"]
        if fx.const_expr(SPLIT == 1):
            nxt_pre = _prefetch_q(nxt, G)
        state = step(
            *state, key_tiles + 1, 0, False, False, True, False, True, False, True
        )
        st = _unflat_f(state, env.l_g != 0)
        l_run = _row_sums(env, st["l"]) if env.l_g != 0 else None
    if fx.const_expr(SPLIT == 1):
        # The next entry's loads by block id, under this task's output.
        nxt_pre = _prefetch_k(nxt, nxt_pre, G)

    # ---- Output of the drained task (or split record and merge); every wave takes the same
    # barriers ----
    o, m_run = st["o"], st["m"]
    # A block id is below 2**31; the window from before the last drain step arrived ahead of the
    # next entry's loads (F's through LDS), so reading it waits for no more than itself.
    env = _out_env(env, prim._uniform(fx.Int32(win_out)) >> 31)
    if fx.const_expr(SPLIT > 1):
        if env.split != 0:
            # Both parts store their records; the one counting last merges them.
            _store_partial(env, o, m_run, l_run)
            fx.rocdl.s_waitcnt(vmcnt=0)
            _stage_end()
            if (env.wave == 0) & (env.lane == 0):
                done = fx.atomic_add(
                    fx.get_iter(COUNTERS) + env.slot, fx.Int32(1), syncscope="agent"
                )
                _write_lds(env.lds, LDS_COUNT, (done, done))
            _barrier()
            done = prim._uniform(fx.Int32(prim._load_lds(env.lds, LDS_COUNT)[0]))
            # Every wave read the count before the output rows reuse LDS.
            _barrier()
            if done == SPLIT_PARTS - 1:
                _merge_out(env, o, m_run, l_run, v_scale)
        else:
            _store_whole(env, o, l_run, m_run, v_scale, False)
    else:
        _store_whole(env, o, l_run, m_run, v_scale, RESIDUAL)
    # The next task's K writes reuse the LDS of every wave's output rows.
    _barrier()
    if fx.const_expr(SPLIT == 1):
        return _flat_pre(nxt_pre)


@flyc.jit
def _persist(
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
    PARTIALS,
    COUNTERS,
    LSE,
    lds,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
    wave,
    G: fx.Constexpr[int],
):
    """The CTA's persistent slots for group G (wave: the wave's index in the CTA); without split
    a slot's first loads issue while the previous one drains."""
    per_tile = HK * ((BQ * GP + 127) // 128)
    limit = ((TASKS + GRID - 1) // GRID) * GRID
    work = fx.Int32(fx.block_idx.x)
    mapped, enabled = _decode(ORDER, ACTIVE, work, limit, per_tile, SPLIT)
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
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    if fx.const_expr(SPLIT > 1):
        while enabled != 0:
            _task(
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
                PARTIALS,
                COUNTERS,
                LSE,
                lds,
                wave,
                mapped,
                mapped,
                enabled,
                (),
                H,
                HK,
                NQ,
                CAP,
                POOL,
                BQ,
                GP,
                SCALE_LOG2,
                PER_TOKEN,
                SLOTTED,
                SPLIT,
                RESIDUAL,
                G,
            )
            work = work + GRID
            mapped, enabled = _decode(ORDER, ACTIVE, work, limit, per_tile, SPLIT)
    else:
        dims = (H, HK, NQ, CAP, POOL, BQ, GP)
        first = _env_for(
            T, lds, wave, mapped, enabled, dims, (PER_TOKEN, SLOTTED, SPLIT, G)
        )
        pre = _flat_pre(_prefetch_k(first, _prefetch_q(first, G), G))
        while enabled != 0:
            next_mapped, next_enabled = _decode(
                ORDER, ACTIVE, work + GRID, limit, per_tile, SPLIT
            )
            pre = _task(
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
                PARTIALS,
                COUNTERS,
                LSE,
                lds,
                wave,
                mapped,
                next_mapped,
                next_enabled,
                pre,
                H,
                HK,
                NQ,
                CAP,
                POOL,
                BQ,
                GP,
                SCALE_LOG2,
                PER_TOKEN,
                SLOTTED,
                SPLIT,
                RESIDUAL,
                G,
            )
            work = work + GRID
            mapped, enabled = next_mapped, next_enabled


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
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
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
    args = (
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
        PARTIALS,
        COUNTERS,
        LSE,
        lds,
        H,
        HK,
        NQ,
        CAP,
        POOL,
        BQ,
        GP,
        TASKS,
        GRID,
        SCALE_LOG2,
        PER_TOKEN,
        SLOTTED,
        SPLIT,
        RESIDUAL,
    )
    # Waves w and w + 4 share a SIMD: the leaders (0-3) and the followers (4-7) alternate
    # M-stages on it.
    wave = prim._uniform(fx.Int32(fx.thread_idx.x) >> 6)
    if wave < 4:
        _persist(*args, wave, 0)
    else:
        _persist(*args, wave, 1)


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
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
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
            PARTIALS,
            COUNTERS,
            LSE,
            H,
            HK,
            NQ,
            CAP,
            POOL,
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


_COMPILED = {}


def _args(inputs, plan, out, stream):
    # Slot-table plans read K/V through K3's pool block table, per block from the pool's base.
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
        plan.partials.view(-1) if plan.split > 1 else plan.counts,
        plan.split_counters if plan.split > 1 else plan.counts,
        plan.lse.view(-1) if plan.core_ratio > 0 else plan.counts,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        plan.block_capacity,
        inputs.k.shape[0],
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
    key = (inputs.q.device, *args[18:20], *args[23:25], *args[27:31])
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        # Uniform branches left unstructurized (no flow blocks of exec-mask SALU in every step,
        # fewer SGPR spills) run faster in every variant. Set every time: compile hints merge
        # into the function.
        hints = {
            "waves_per_eu": 2,
            "llvm_options": prim.uniform_branch_options(True),
        }
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            # Eight waves of 256 registers (VGPRs and AGPRs) and 64 KiB LDS per CTA: two waves
            # per SIMD. Compiled with an empty grid, so it launches nothing.
            compiled = _COMPILED[key] = flyc.compile[hints](
                _launch, *args[:26], 0, *args[27:]
            )
    return compiled


def run(*, inputs, plan, out):
    """FP8 union attention after the FP8 pack launch (which quantizes Q). inputs adds
    k_scale / v_scale (FP32 [N, HK] per token or [1]), the FP8 Q of the pack launch
    (inputs.q8 / inputs.qs) and the FP32 softmax_scale * log2(e) (inputs.scale_log2)."""
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    with torch.cuda.device(inputs.q.device):
        _compiled(inputs, args, stream)(*args)


def launcher(*, inputs, plan, out):
    """launch(q8, qs, k, v, out, k_scale, v_scale, scale_log2) with flat Q8/QS/out/scales and
    unflattened K/V; other arguments prebuilt. Valid while the plan's scratch bindings and
    stream stay.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[8:]
    return lambda q8, qs, k, v, o, ks, vs, scale_log2: compiled(
        q8, qs, k, v, ks, vs, o, scale_log2, *tail
    )
