# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx942 MiMo block-8 verify PA with probabilities kept in registers.

Shape: BF16 Q/K/V, one KV head per 16 query heads, Q/K 192 and V 128, page 64,
eight MTP queries (128 query rows), SHUFFLE-5D caches, no sliding window.

Four waves are two token groups x two row groups. Each wave owns 64 query rows
(four MTP positions x 16 heads) and two of the four 64-token pages of every
256-token tile. QK computes S^T = K Q^T, whose accumulator layout is PV's B
operand (O^T = V^T P^T), so probabilities never leave the wave: the tile loop
has no LDS traffic and no barrier. Q stays in registers for the whole task.
Each wave keeps its own online softmax with a lazily moved reference; the two
token groups of a row group merge through LDS once per task and write the
shared reducer's packed partials (natural-log max, sum, normalized BF16 O).
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .buffer_ops import buffer_load, create_buffer_resource_from_addr
from .tensor_shim import buf_base_i64
from .utils import rcp_f32

D, DV, PAGE, GQA, QLEN = 192, 128, 64, 16, 8
ROWS = QLEN * GQA
WAVE = 64
THREADS = 256
LOG2E = math.log2(math.e)
K_PAGE_BYTES = D * PAGE * 2
V_PAGE_BYTES = DV * PAGE * 2
K_PACKS = D // 16
DV_TILES = DV // 16
# O^T accumulators: [dv tile j][row tile n] x 4 values, flattened.
O_ELEMS = DV_TILES * 4 * 4
# A row's reference max moves only once its max passes it by this much, so P <= 2^8.
MAX_SLACK = 8.0
# Per row group: one fp32 O half (64 rows x 64 columns), then 64 maxima and 64 sums.
LDS_O_WORDS = 64 * 64
LDS_ROWGROUP_WORDS = LDS_O_WORDS + 2 * 64
LDS_WORDS = 2 * LDS_ROWGROUP_WORDS

_CACHE = {}


def supports_mimo_verify(*, arch, kv_dtype, query_dtype, head_dim, value_dim, block_size,
                         query_group_size, query_length, num_kv_heads, trans_v,
                         per_token_kv, sliding_window, query_splits):
    return (
        arch == "gfx942"
        and kv_dtype == "bf16"
        and query_dtype == "bf16"
        and (head_dim, value_dim, block_size) == (D, DV, PAGE)
        and (query_group_size, query_length, num_kv_heads) == (GQA, QLEN, 1)
        and trans_v
        and not per_token_kv
        and sliding_window == 0
        and query_splits in (None, 1)
    )


def _flat_loader(ptr, elem_ty, extent):
    """16-byte global loads (8 BF16) at element offsets of a flat view, as two i64."""
    atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_ty)
    reg = fx.make_rmem_tensor(fx.make_layout(8, 1), elem_ty)
    flat = fx.Tensor(fx.make_view(fx.recast_iter(elem_ty, ptr), fx.make_layout(extent, 1)))
    tiled = fx.logical_divide(flat, fx.make_layout(1, 1))

    def load(elem_idx):
        fx.copy(atom, fx.slice(tiled, (None, elem_idx)), reg)
        return fx.Vector(fx.memref_load_vec(reg)).bitcast(fx.Int64)

    return load


def _mfma(a, b, acc):
    a4 = fx.Vector.from_elements([a], dtype=fx.Int64).bitcast(fx.Int16)
    b4 = fx.Vector.from_elements([b], dtype=fx.Int64).bitcast(fx.Int16)
    return fx.Vector(fx.rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [a4, b4, acc, 0, 0, 0]))


def _max3(a, b, c):
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [a.ir_value(), b.ir_value(), c.ir_value()],
                                      "v_max3_f32 $0, $1, $2, $3", "=v,v,v,v", has_side_effects=False))


def _max(a, b):
    # Inline asm: llvm.maxnum would first canonicalize both operands (two extra VALU ops).
    return fx.Float32(llvm.inline_asm(fx.Float32.ir_type, [a.ir_value(), b.ir_value()],
                                      "v_max_f32 $0, $1, $2", "=v,v,v", has_side_effects=False))


def _exp2(value):
    return fx.Float32(llvm.call_intrinsic(fx.Float32.ir_type, "llvm.amdgcn.exp2.f32",
                                          [fx.Float32(value).ir_value()], [], []))


def _pack_bf16(values):
    """Four finite FP32 to one i64 of BF16 (RNE, element i in the low half)."""
    bits = fx.Vector(values).bitcast(fx.Uint32)
    rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
    selector = fx.Int32(0x07060302)
    words = [
        fx.Int32(llvm.inline_asm(fx.Int32.ir_type,
                                 [rounded[i + 1].ir_value(), rounded[i].ir_value(), selector.ir_value()],
                                 "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True))
        for i in range(0, 4, 2)
    ]
    return fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int64)[0]


def _uniform(value):
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value()))


def _agpr_read(word):
    """A dword read through an AGPR: its only VALU use then does not pull an MFMA operand into the arch VGPRs."""
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [fx.Int32(word).ir_value()], "v_accvgpr_read_b32 $0, $1",
                                    "=v,a", has_side_effects=False))


def _loop_vgpr(value, step):
    """value in a VGPR that depends on step, so LICM keeps constant offsets added to it next to their loads.

    Each load group takes its own step: CSE would otherwise share one copy, and its offsets, across blocks.
    """
    return fx.Int32(llvm.inline_asm(fx.Int32.ir_type, [fx.Int32(value).ir_value(), fx.Int32(step).ir_value()],
                                    "", "=v,0,s", has_side_effects=False))


def _load16(rsrc, voffset, soffset):
    """16 bytes at byte offsets voffset + soffset of a buffer; a constant in voffset becomes the instruction offset."""
    words = rocdl.raw_ptr_buffer_load(ir.VectorType.get([4], fx.Int32.ir_type), rsrc, fx.Int32(voffset).ir_value(),
                                      fx.Int32(soffset).ir_value(), aux=0)
    return fx.Vector(words).bitcast(fx.Int64)


def _quad(flat, index):
    return fx.Vector.from_elements([flat[index * 4 + i] for i in range(4)], dtype=fx.Float32)


def _flatten(quads):
    return fx.Vector.from_elements([q[i] for q in quads for i in range(4)], dtype=fx.Float32)


def _tok_first(c, tg):
    """First token of step c >= 0 for token group tg: pages 2 * tg and 2 * tg + 1 of each 256-token tile."""
    return ((c >> 2) << 8) + tg * 128 + ((c & 3) << 5)


def _p_with_qk(flat_s, scale_f, neg_m, l_sum, k_ops, q_ops):
    """P = exp2(S * scale - m) of a step as BF16 PV operands and its row sums, with the next step's QK.

    Each QK MFMA is followed by one exponential or up to three scalar VALU, the most a 16x16x16 MFMA hides
    (gfx942: 3 VALU fully, an exponential all but 4 cycles), fenced so the scheduler keeps the order. Group g
    (row tile g // 2, half g % 2) issues its FMAs one group ahead and its rounding, packing and sums one
    group behind its exponentials. Same arithmetic as the unscheduled form.
    """
    acc = [[fx.full(4, 0.0, fx.Float32) for _ in range(4)] for _ in range(2)]
    mfmas = [(half, n, p) for p in range(K_PACKS) for half in range(2) for n in range(4)]
    x, e, t, b, w, pair = {}, {}, {}, {}, {}, {}
    sums = [l_sum[n] for n in range(4)]
    p_ops = [[None] * 4 for _ in range(2)]
    stream = []

    def fma(g, j):
        n, half = g // 2, g % 2
        x[g, j] = fx.fma(_quad(flat_s, half * 4 + n)[j], scale_f, neg_m[n])

    def exp(g, j):
        e[g, j] = _exp2(x[g, j])

    def bfe(g, j):
        bits = fx.Vector.from_elements([e[g, j]], dtype=fx.Float32).bitcast(fx.Uint32)[0]
        t[g, j] = (bits, (bits >> 16) & fx.Uint32(1))

    def add3(g, j):
        bits, lsb = t[g, j]
        b[g, j] = bits + fx.Uint32(0x7FFF) + lsb

    def perm(g, k):
        w[g, k] = fx.Int32(llvm.inline_asm(
            fx.Int32.ir_type, [b[g, 2 * k + 1].ir_value(), b[g, 2 * k].ir_value(),
                               fx.Int32(0x07060302).ir_value()],
            "v_perm_b32 $0, $1, $2, $3", "=v,v,v,s", has_side_effects=True))
        if k == 1:
            p_ops[g % 2][g // 2] = fx.Vector.from_elements([w[g, 0], w[g, 1]], fx.Int32).bitcast(fx.Int64)[0]

    def pair_sum(g, k):
        pair[g, k] = e[g, 2 * k] + e[g, 2 * k + 1]

    def acc_sum(g, k):
        sums[g // 2] = sums[g // 2] + pair[g, k]

    def valu(fn, *args):
        stream.append(("valu", lambda: fn(*args)))

    for j in range(4):
        valu(fma, 0, j)
    for g in range(9):
        if g < 8:
            for j in range(4):
                stream.append(("exp", lambda g=g, j=j: exp(g, j)))
        if g + 1 < 8:
            for j in range(4):
                valu(fma, g + 1, j)
        if g >= 1:
            for j in range(4):
                valu(bfe, g - 1, j)
                valu(add3, g - 1, j)
            for k in range(2):
                valu(perm, g - 1, k)
            for k in range(2):
                valu(pair_sum, g - 1, k)
            for k in range(2):
                valu(acc_sum, g - 1, k)
    slots = []
    for kind, fn in stream:
        if kind == "exp" or not slots or slots[-1][0] != "valu" or len(slots[-1][1]) == 3:
            slots.append((kind, [fn]))
        else:
            slots[-1][1].append(fn)
    for i in range(max(len(mfmas), len(slots))):
        if i < len(mfmas):
            half, n, p = mfmas[i]
            acc[half][n] = _mfma(k_ops[half * K_PACKS + p], q_ops[n * K_PACKS + p], acc[half][n])
        if i < len(slots):
            for fn in slots[i][1]:
                fn()
        rocdl.sched_barrier(0)
    s_next = _flatten([acc[half][n] for half in range(2) for n in range(4)])
    p_vec = fx.Vector.from_elements([p_ops[half][n] for half in range(2) for n in range(4)], dtype=fx.Int64)
    return p_vec, fx.Vector.from_elements(sums, dtype=fx.Float32), s_next


def compile_mimo_verify_tile(*, softmax_scale):
    """Kernel with pa_decode's tile launch signature for the MiMo verify shape."""
    key = float(softmax_scale)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    scale = float(softmax_scale) * LOG2E
    _cache_tag = key

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, LDS_WORDS, 16]

    @flyc.jit
    def _qk(k_ops, q_ops):
        """S^T of one step, flattened [half][row tile n][j]: lane holds tokens 8 * grp + 4 * half + j of row l16."""
        s = [[None] * 4 for _ in range(2)]
        for half in range_constexpr(2):
            for n in range_constexpr(4):
                acc = fx.full(4, 0.0, fx.Float32)
                for p in range_constexpr(K_PACKS):
                    acc = _mfma(k_ops[half * K_PACKS + p], q_ops[n * K_PACKS + p], acc)
                s[half][n] = acc
        return _flatten([s[half][n] for half in range(2) for n in range(4)])

    @flyc.jit
    def _mask_s(flat_s, c, need, context, rg, tg, grp):
        """Causal/context mask of step c, if `need`: some row sees fewer than all 32 tokens of the step."""
        neg_inf = fx.Float32(float("-inf"))
        if need:
            tok_first = _tok_first(c, tg)
            masked = []
            for half in range_constexpr(2):
                for n in range_constexpr(4):
                    limit = context - (QLEN - 1) + rg * 4 + n - tok_first - grp * 8
                    for j in range_constexpr(4):
                        masked.append((fx.Int32(half * 4 + j) < limit).select(
                            flat_s[(half * 4 + n) * 4 + j], neg_inf))
            flat_s = fx.Vector.from_elements(masked, dtype=fx.Float32)
        return flat_s

    @flyc.jit
    def _soft_qk(flat_s, k_next, o_acc, m_ref, l_sum, neg_m, thr, q_ops, next_qk: fx.Constexpr[bool]):
        """Online softmax of one step from its scores flat_s; with next_qk, also QK of the next step.

        With next_qk, P and the next QK are interleaved by _p_with_qk.
        m_ref is the reference max in scaled log2 units; neg_m = -m_ref (0 while a row has none) and
        thr = (m_ref + MAX_SLACK) / scale change only when the reference moves.
        """
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        scale_f = fx.Float32(scale)
        # Lane-local row maxima over this lane's 8 scores. The 4 lane groups of a row are reduced only when
        # some lane passes the row's reference by MAX_SLACK, so a reference move stays row-uniform.
        local_max = []
        for n in range_constexpr(4):
            a = _quad(flat_s, n)
            b = _quad(flat_s, 4 + n)
            local_max.append(_max3(_max3(a[0], a[1], a[2]), _max3(a[3], b[0], b[1]), _max3(b[2], b[3], b[3])))
        lane_moves = [local_max[n] > thr[n] for n in range_constexpr(4)]
        any_move = lane_moves[0] | lane_moves[1] | lane_moves[2] | lane_moves[3]
        ballot = fx.Int64(rocdl.ballot(fx.Int64.ir_type, any_move.ir_value()))
        m_new = m_ref
        l_new = l_sum
        o_new = o_acc
        neg_m_new = neg_m
        thr_new = thr
        if ballot != fx.Int64(0):
            alphas = []
            m_list = []
            neg_m_list = []
            thr_list = []
            for n in range_constexpr(4):
                v = local_max[n]
                v = _max(v, v.shuffle_xor(16, WAVE))
                v = _max(v, v.shuffle_xor(32, WAVE))
                move = v > thr[n]
                m_n = move.select(v * scale_f, m_ref[n])
                alphas.append(move.select(_exp2(m_ref[n] - m_n), fx.Float32(1.0)))
                m_list.append(m_n)
                # An empty row keeps reference 0 for P to avoid -inf - (-inf).
                neg_m_list.append(-(m_n > neg_inf).select(m_n, zero))
                thr_list.append((m_n + fx.Float32(MAX_SLACK)) * fx.Float32(1.0 / scale))
            m_new = fx.Vector.from_elements(m_list, dtype=fx.Float32)
            neg_m_new = fx.Vector.from_elements(neg_m_list, dtype=fx.Float32)
            thr_new = fx.Vector.from_elements(thr_list, dtype=fx.Float32)
            l_new = fx.Vector.from_elements([l_sum[n] * alphas[n] for n in range_constexpr(4)], dtype=fx.Float32)
            o_new = fx.Vector.from_elements(
                [o_acc[(j * 4 + n) * 4 + i] * alphas[n]
                 for j in range_constexpr(DV_TILES) for n in range_constexpr(4) for i in range_constexpr(4)],
                dtype=fx.Float32)
        if const_expr(next_qk):
            p_vec, l_new, s_next = _p_with_qk(flat_s, scale_f, neg_m_new, l_new, k_next, q_ops)
        else:
            # P = exp2(S * scale - m).
            p_ops = [[None] * 4 for _ in range(2)]
            l_list = []
            for n in range_constexpr(4):
                acc_l = l_new[n]
                for half in range_constexpr(2):
                    sv = _quad(flat_s, half * 4 + n)
                    p = fx.Vector.from_elements([_exp2(fx.fma(sv[j], scale_f, neg_m_new[n]))
                                                 for j in range_constexpr(4)], dtype=fx.Float32)
                    acc_l = acc_l + (p[0] + p[1]) + (p[2] + p[3])
                    p_ops[half][n] = _pack_bf16(p)
                l_list.append(acc_l)
            l_new = fx.Vector.from_elements(l_list, dtype=fx.Float32)
            s_next = flat_s
            p_vec = fx.Vector.from_elements([p_ops[half][n] for half in range(2) for n in range(4)], dtype=fx.Int64)
        return p_vec, o_new, m_new, l_new, neg_m_new, thr_new, s_next

    @flyc.jit
    def _pv(c, need, v_ops, p_vec, o_acc, context, tg, grp):
        """O^T[dv tile j][row tile n] += V^T P^T of step c over both M-tiles (k-steps)."""
        # Zero V rows at or past the context (`need`: the step has some): 0 * NaN would still be NaN.
        v_words = v_ops
        if need:
            remaining = context - _tok_first(c, tg) - grp * 8
            masked_v = []
            for j in range_constexpr(DV_TILES):
                for half in range_constexpr(2):
                    words = fx.Vector.from_elements([v_ops[j * 2 + half]], dtype=fx.Int64).bitcast(fx.Int32)
                    kept = []
                    for w in range_constexpr(2):
                        token = half * 4 + w * 2
                        lo = (fx.Int32(token) < remaining).select(fx.Int32(0xFFFF), fx.Int32(0))
                        hi = (fx.Int32(token + 1) < remaining).select(fx.Int32(-65536), fx.Int32(0))
                        kept.append(_agpr_read(words[w]) & (lo | hi))
                    masked_v.append(fx.Vector.from_elements(kept, dtype=fx.Int32).bitcast(fx.Int64)[0])
            v_words = fx.Vector.from_elements(masked_v, dtype=fx.Int64)
        # O^T[dv tile j][row tile n] += V^T P^T over both M-tiles (k-steps).
        quads = []
        for j in range_constexpr(DV_TILES):
            for n in range_constexpr(4):
                acc = _quad(o_acc, j * 4 + n)
                for half in range_constexpr(2):
                    acc = _mfma(v_words[j * 2 + half], p_vec[half * 4 + n], acc)
                quads.append(acc)
        return _flatten(quads)

    @flyc.jit
    def _merge_half(half, tg, region, ml_base, base, o_fin, merged, m_fin, l_tot, m_out, l_out, l16, grp):
        """Token group 1 hands one 64-column O half to token group 0, which merges it."""
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        if tg == 1:
            for jj in range_constexpr(4):
                j = half * 4 + jj
                for n in range_constexpr(4):
                    row = n * 16 + l16
                    fx.ptr_store(_quad(o_fin, j * 4 + n), fx.add_offset(base, region + row * 64 + jj * 16 + grp * 4))
        gpu.barrier()
        if tg == 0:
            values = [merged[i] for i in range(O_ELEMS)]
            m_list, l_list = [], []
            for n in range_constexpr(4):
                row = n * 16 + l16
                m1 = fx.ptr_load(fx.add_offset(base, ml_base + row), result_type=fx.Vector.make_type(1, fx.Float32))[0]
                l1 = fx.ptr_load(fx.add_offset(base, ml_base + 64 + row),
                                 result_type=fx.Vector.make_type(1, fx.Float32))[0]
                m0 = m_fin[n]
                big = fx.maxnumf(m0, m1)
                a0 = (m0 > neg_inf).select(_exp2(m0 - big), zero)
                a1 = (m1 > neg_inf).select(_exp2(m1 - big), zero)
                m_list.append(big)
                l_list.append(l_tot[n] * a0 + l1 * a1)
                for jj in range_constexpr(4):
                    j = half * 4 + jj
                    o1 = fx.Vector(fx.ptr_load(fx.add_offset(base, region + row * 64 + jj * 16 + grp * 4),
                                               result_type=fx.Vector.make_type(4, fx.Float32)))
                    for i in range_constexpr(4):
                        k = (j * 4 + n) * 4 + i
                        values[k] = values[k] * a0 + o1[i] * a1
            merged = fx.Vector.from_elements(values, dtype=fx.Float32)
            m_out = fx.Vector.from_elements(m_list, dtype=fx.Float32)
            l_out = fx.Vector.from_elements(l_list, dtype=fx.Float32)
        gpu.barrier()
        return merged, m_out, l_out

    @flyc.jit
    def _task(pmax_ptr, psum_ptr, pout_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
              max_blocks_per_seq, stride_q_row, stride_q_head, num_sequences, slot, seq, start, end, context):
        _ = _cache_tag
        tid = fx.Int32(gpu.thread_id("x"))
        wave = _uniform(tid // WAVE)
        lane = tid % WAVE
        l16 = lane % 16
        grp = lane // 16
        tg = wave & 1
        rg = wave >> 1

        load_q = _flat_loader(query_ptr, fx.BFloat16, 1 << 40)
        k_base = buf_base_i64(key_cache_ptr)
        v_base = buf_base_i64(value_cache_ptr)
        # Scalar page-table reads wait on lgkmcnt, so a lookup does not drain the K/V loads in flight.
        table = create_buffer_resource_from_addr(
            buf_base_i64(block_tables_ptr).ir_value(),
            num_records_bytes=fx.Int64(num_sequences) * fx.Int64(max_blocks_per_seq) * 4)
        num_pages = (context + (PAGE - 1)) // PAGE
        table_row = seq * max_blocks_per_seq

        # 16-byte d-chunk of k-pack pair (qkhe, qkr) for this lane group; K and Q share it.
        def d_chunk(qkhe, qkr):
            return (qkhe * 4 + grp) * 2 + qkr

        # Q^T as QK's B operand: row tile n is MTP position 4 * rg + n, head l16.
        q_list = []
        for n in range_constexpr(4):
            row_base = (seq * QLEN + rg * 4 + n) * stride_q_row + l16 * stride_q_head
            for qkhe in range_constexpr(3):
                for qkr in range_constexpr(2):
                    w = load_q(row_base + d_chunk(qkhe, qkr) * 8)
                    q_list.extend([w[0], w[1]])
        q_ops = fx.Vector.from_elements(q_list, dtype=fx.Int64)

        # K token of this lane: 8 * (l16 // 4) + l16 % 4 (+ 4 in the second M-tile), so S^T rows
        # 4 * grp + j are tokens 8 * grp + j (+ 4): one 16-byte V load per lane and dv tile.
        k_token = (l16 // 4) * 8 + l16 % 4
        # Byte offsets in a page: K (d_chunk * 64 + token) * 16, V (token group * 128 + dv) * 16. The lane parts
        # stay in one VGPR each; per-load constants go to the instruction offset and soffset.
        lane_k = grp * 2048 + k_token * 16
        lane_v = grp * 2048 + l16 * 16
        last_page = (num_pages > 0).select(num_pages - 1, fx.Int32(0))

        def page_of(e):
            """Page id of the even step e (and e + 1); past the sequence, the last page (fully masked)."""
            index = (e & -4) + tg * 2 + ((e >> 1) & 1)
            index = (index < last_page).select(index, last_page)
            return fx.Int32(buffer_load(table, table_row + index, vec_width=1, is_scalar=True))

        def k_rsrc(page):
            return create_buffer_resource_from_addr((k_base + fx.Int64(page) * fx.Int64(K_PAGE_BYTES)).ir_value(),
                                                    num_records_bytes=K_PAGE_BYTES)

        def v_addr(page):
            return v_base + fx.Int64(page) * fx.Int64(V_PAGE_BYTES)

        def load_k(rsrc, step, par):
            lane = _loop_vgpr(lane_k, step)
            k_list = []
            for half in range_constexpr(2):
                for qkhe in range_constexpr(3):
                    for qkr in range_constexpr(2):
                        w = _load16(rsrc, lane + (qkr * 1024 + half * 64 + par * 512), qkhe * 8192)
                        k_list.extend([w[0], w[1]])
            return fx.Vector.from_elements(k_list, dtype=fx.Int64)

        def load_v(addr, step, par):
            rsrc = create_buffer_resource_from_addr(addr.ir_value(), num_records_bytes=V_PAGE_BYTES)
            lane = _loop_vgpr(lane_v, step)
            v_list = []
            for j in range_constexpr(DV_TILES):
                w = _load16(rsrc, lane + j * 256, par * 8192)
                v_list.extend([w[0], w[1]])
            return fx.Vector.from_elements(v_list, dtype=fx.Int64)

        def first_step_past(limit):
            """First step whose first token is past `limit` (the first token grows with the step)."""
            y = limit - tg * 128
            r = ((y & 255) >> 5) + 1
            x = (y >> 8) * 4 + (r < 4).select(r, fx.Int32(4))
            return (y < 0).select(fx.Int32(0), x)

        # Software pipeline: step c runs the softmax of c with the QK of c + 1, then PV(c), which shares a block with
        # the row maxima of c + 1 (the mask of S(c + 1) comes first). K(c + 2) is issued at the start of step c and
        # V(c + 1) once PV(c) has consumed V(c), so each load has about a step to land.
        # Steps come in multiples of 4 (one 256-token tile): two per iteration (c even, so steps c and c + 1 share
        # a page), the last two peeled. Each iteration looks up the page of the next one.
        first = start * 4
        last = end * 4
        # From these steps on, S needs the causal/context mask and V rows past the context are zeroed.
        s_from = first_step_past(context - 32 - (QLEN - 1))
        v_from = first_step_past(context - 32)
        page = page_of(first)
        k0 = load_k(k_rsrc(page), first, 0)
        k_cur = load_k(k_rsrc(page), first + 1, 1)
        v_addr_c = v_addr(page)
        v_cur = load_v(v_addr_c, first, 0)
        page_n = page_of(first + 2)
        s_cur = _mask_s(_qk(k0, q_ops), first, first >= s_from, context, rg, tg, grp)
        o_acc = fx.full(O_ELEMS, 0.0, fx.Float32)
        m_ref = fx.full(4, float("-inf"), fx.Float32)
        l_sum = fx.full(4, 0.0, fx.Float32)
        neg_m = fx.full(4, 0.0, fx.Float32)
        thr = fx.full(4, float("-inf"), fx.Float32)
        for c in range(first, last - 2, fx.Int32(2)):
            rsrc_k = k_rsrc(page_n)
            v_addr_n = v_addr(page_n)
            page_nn = page_of(c + 4)
            k2 = load_k(rsrc_k, c + 2, 0)
            p_c, o_c, m_c, l_c, nm_c, th_c, s1 = _soft_qk(s_cur, k_cur, o_acc, m_ref, l_sum, neg_m, thr, q_ops, True)
            s1 = _mask_s(s1, c + 1, c + 1 >= s_from, context, rg, tg, grp)
            o_c = _pv(c, c >= v_from, v_cur, p_c, o_c, context, tg, grp)
            v1 = load_v(v_addr_c, c + 1, 1)
            k_cur = load_k(rsrc_k, c + 3, 1)
            p_c, o_c, m_c, l_c, nm_c, th_c, s_cur = _soft_qk(s1, k2, o_c, m_c, l_c, nm_c, th_c, q_ops, True)
            s_cur = _mask_s(s_cur, c + 2, c + 2 >= s_from, context, rg, tg, grp)
            o_acc = _pv(c + 1, c + 1 >= v_from, v1, p_c, o_c, context, tg, grp)
            v_cur = load_v(v_addr_n, c + 2, 0)
            m_ref, l_sum, neg_m, thr = m_c, l_c, nm_c, th_c
            page_n, v_addr_c = page_nn, v_addr_n
        p_c, o_c, m_c, l_c, nm_c, th_c, s_last = _soft_qk(s_cur, k_cur, o_acc, m_ref, l_sum, neg_m, thr, q_ops, True)
        s_last = _mask_s(s_last, last - 1, last - 1 >= s_from, context, rg, tg, grp)
        o_c = _pv(last - 2, last - 2 >= v_from, v_cur, p_c, o_c, context, tg, grp)
        v_last = load_v(v_addr_c, last - 1, 1)
        p_c, o_c, m_fin, l_fin, _, _, _ = _soft_qk(s_last, k_cur, o_c, m_c, l_c, nm_c, th_c, q_ops, False)
        o_fin = _pv(last - 1, last - 1 >= v_from, v_last, p_c, o_c, context, tg, grp)

        # Row sums over the 4 lane groups.
        l_list = []
        for n in range_constexpr(4):
            v = l_fin[n]
            v = v + v.shuffle_xor(16, WAVE)
            v = v + v.shuffle_xor(32, WAVE)
            l_list.append(v)
        l_tot = fx.Vector.from_elements(l_list, dtype=fx.Float32)

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        base = fx.recast_iter(fx.Float32, smem.buf.ptr)
        region = rg * LDS_ROWGROUP_WORDS
        ml_base = region + LDS_O_WORDS
        if tg == 1:
            if grp == 0:
                for n in range_constexpr(4):
                    row = n * 16 + l16
                    fx.ptr_store(fx.Vector.from_elements([m_fin[n]], dtype=fx.Float32),
                                 fx.add_offset(base, ml_base + row))
                    fx.ptr_store(fx.Vector.from_elements([l_tot[n]], dtype=fx.Float32),
                                 fx.add_offset(base, ml_base + 64 + row))
        merged, m_out, l_out = o_fin, m_fin, l_tot
        for half in range_constexpr(2):
            merged, m_out, l_out = _merge_half(half, tg, region, ml_base, base, o_fin, merged, m_fin, l_tot,
                                               m_out, l_out, l16, grp)

        if tg == 0:
            zero = fx.Float32(0.0)
            # Widen before multiplying: packed partials can exceed 2**31 elements.
            pout = fx.add_offset(fx.recast_iter(fx.BFloat16, pout_ptr), fx.Int64(slot) * fx.Int64(ROWS * DV))
            pmax = fx.recast_iter(fx.Float32, pmax_ptr)
            psum = fx.recast_iter(fx.Float32, psum_ptr)
            for n in range_constexpr(4):
                row = (rg * 4 + n) * 16 + l16
                l_row = l_out[n]
                inv_l = fx.Float32(rcp_f32((l_row > zero).select(l_row, fx.Float32(1.0))))
                for j in range_constexpr(DV_TILES):
                    o = _quad(merged, j * 4 + n) * fx.full(4, inv_l, fx.Float32)
                    word = fx.Vector.from_elements([_pack_bf16(o)], dtype=fx.Int64).bitcast(fx.BFloat16)
                    fx.ptr_store(word, fx.add_offset(pout, row * DV + j * 16 + grp * 4))
                if grp == 0:
                    offset = slot * ROWS + row
                    fx.ptr_store(fx.Vector.from_elements([m_out[n] * fx.Float32(1.0 / LOG2E)], dtype=fx.Float32),
                                 fx.add_offset(pmax, offset))
                    fx.ptr_store(fx.Vector.from_elements([l_row], dtype=fx.Float32), fx.add_offset(psum, offset))

    @flyc.kernel(name="pa_decode_mimo_verify_regp_bf16_d192_v128", known_block_size=(THREADS, 1, 1))
    def kernel(pmax_ptr: fx.Pointer, psum_ptr: fx.Pointer, pout_ptr: fx.Pointer, query_ptr: fx.Pointer,
               key_cache_ptr: fx.Pointer, value_cache_ptr: fx.Pointer, block_tables_ptr: fx.Pointer,
               max_blocks_per_seq: fx.Int32, stride_q_row: fx.Int32, stride_q_head: fx.Int32,
               work_info_ptr: fx.Pointer, num_sequences: fx.Int32):
        slot = fx.Int32(fx.Uint32(gpu.block_id("x")) * fx.Uint32(gpu.grid_dim.z) + fx.Uint32(gpu.block_id("z")))
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        task = fx.ptr_load(fx.add_offset(work, slot * 4), result_type=fx.Vector.make_type(4, fx.Int32))
        start = _uniform(task[1])
        end = _uniform(task[2])
        if start < end:
            _task(pmax_ptr, psum_ptr, pout_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
                  max_blocks_per_seq, stride_q_row, stride_q_head, num_sequences, slot,
                  _uniform(task[0]), start, end, _uniform(task[3]))

    @flyc.jit
    def launch(pmax: fx.Pointer, psum: fx.Pointer, pout: fx.Pointer, query: fx.Pointer, key_cache: fx.Pointer,
               value_cache: fx.Pointer, block_tables: fx.Pointer, key_scale: fx.Pointer, value_scale: fx.Pointer,
               max_blocks_per_seq: fx.Int32, num_seqs: fx.Int32, num_kv_heads: fx.Int32,
               stride_ks_block: fx.Int32, stride_ks_head: fx.Int32, stride_q_row: fx.Int32,
               stride_q_head: fx.Int32, work_info: fx.Pointer, work_capacity: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        _ = _cache_tag
        # Batch-first (first partitions of all sequences dispatch first) when the plan capacity is a whole
        # number of partitions per sequence, else flat; slot = x * grid_z + z covers both.
        even = work_capacity % num_seqs == 0
        grid_x = even.select(num_seqs, work_capacity)
        grid_z = even.select(work_capacity // num_seqs, fx.Int32(1))
        # No packed FP32: a v_pk_* op never co-issues with an MFMA (+12 cycles after one, gfx942).
        kernel(pmax, psum, pout, query, key_cache, value_cache, block_tables, max_blocks_per_seq,
               stride_q_row, stride_q_head, work_info, num_seqs,
               value_attrs={"llvm.target_features": ir.Attribute.parse(
                   '#llvm.target_features<["-packed-fp32-ops"]>')}).launch(
            grid=(grid_x, 1, grid_z), block=(THREADS, 1, 1), stream=stream)

    compiled = {"launch": launch, "kernel": kernel}
    return _CACHE.setdefault(key, compiled)
