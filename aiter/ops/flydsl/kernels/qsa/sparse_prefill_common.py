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
work queue. Everything below FlyDSL's stable API (flydsl.expr, flydsl.expr.rocdl
__all__, flydsl.compiler) is in the low-level section, each item with the reason it
stays: empty inline asm register pins (no public asm entry or placement hint), the
target-features attribute and an internal LLVM option, ROCDL operations FlyDSL wraps
without exporting, an LLVM buffer atomic intrinsic, volatile LDS reads and the
lgkmcnt ledger of the union pipelines' partial waits and cross-wave LDS reuse.
"""

import dataclasses
import functools
import logging
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

# ---- Low-level primitives: everything below FlyDSL's stable API ----

# ---- Empty inline asm ----
#
# FlyDSL 0.5.0 has no public asm entry, register-placement hint or compiler-only memory
# barrier. A constraint LLVM cannot satisfy is only printed ("error: could not allocate
# output register for constraint 'v'") and the kernel computes garbage.


def _pin_i32(value):
    """value materialized in a VGPR at this point (tied empty asm)."""
    # No fx equivalent; LLVM would sink or rematerialize the value into its uses.
    return fx.Int32(
        llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(value).ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


def _pin(values, chunk=8):
    """Vector values materialized in VGPRs at this point, chunk words per tied empty asm."""
    # A one-element vector constraint fails register allocation (only printed by LLVM).
    values = fx.Vector(values)
    dtype = values.dtype
    words = values.bitcast(fx.Int32) if dtype.width < 32 else values
    if words.numel % chunk == 1:
        raise ValueError("_pin: a one-word chunk cannot be tied; use _pin_i32")
    result = []
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, min(start + chunk, words.numel))],
            words.dtype,
        )
        tied = fx.Vector(
            llvm.inline_asm(
                part.ir_value().type,
                [part.ir_value()],
                "",
                "=v,0",
                has_side_effects=True,
            )
        )
        result.extend(tied[i] for i in range(tied.numel))
    return fx.Vector.from_elements(result, words.dtype).bitcast(dtype)


def _keep_vgpr(words, chunk=4):
    """An empty asm use that keeps words in VGPRs (not AV-class operands of MFMA only)."""
    # A tied pin's def would cost an s_nop; only a use is needed here.
    words = fx.Vector(words)
    for start in range(0, words.numel, chunk):
        part = fx.Vector.from_elements(
            [words[i] for i in range(start, start + chunk)], words.dtype
        )
        llvm.inline_asm(None, [part.ir_value()], "", "v", has_side_effects=True)


def hold(*values):
    """Empty asm uses: the FP32 values are computed at this point."""
    # MachineSink ignores sched_barrier: results used only after a later branch would sink
    # past it and run after the last MFMA.
    for value in values:
        llvm.inline_asm(
            None, [fx.Float32(value).ir_value()], "", "v", has_side_effects=True
        )


def compiler_fence():
    """Empty asm with a memory clobber: a compiler memory barrier, no instruction."""
    # fx.memory_fence emits a real fence; this only keeps MachineSink from moving loads
    # issued before it into a later block.
    llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)


# ---- Kernel attributes and LLVM options ----


def kernel_attrs():
    """value_attrs of the sparse prefill kernels.

    With packed FP32 VALU (v_pk_*_f32) allowed, LLVM packs FP32 math in the kernels' loops
    and they run slower (MI308X, a 12000-token prefill at groups 12, 6 and 3, kernel
    times): the BF16 union by 10-11%, the FP8 union by 12-14% with per tensor scales and
    53-61% with per token scales, the BF16 packed direct by 17-18%; the memory-bound direct
    launches change by at most 1%. Outputs are the same. FlyDSL has no public
    target-feature option (no compile hint carries one): llvm.passthrough forwards the
    LLVM function attribute.
    """
    return {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}


def uniform_branch_options(unstructured):
    """FlyDSL compile hints (llvm_options) that leave a kernel's wave-uniform branches
    unstructurized when unstructured (for the compile they are passed to only; FlyDSL sets
    the LLVM option around that compile and restores it).

    AMDGPU's StructurizeCFG also structurizes branches LLVM proves wave-uniform: an if/else
    becomes the if branch, a flow block that tests a saved SGPR mask, then the else branch,
    whose live-ins stay live through the if branch. In the FP8 union's key loops that costs
    exec-mask SALU in every step and SGPR spills (per token with a slot table at two KV
    heads: 56 -> 42 s_cbranch, 93 -> 77 SGPR spills without and with the option).
    structurizecfg-skip-uniform-regions is an internal LLVM cl::opt, not an API: this is
    the one place that names it. If LLVM drops the option, setting it fails the compile; if
    it stays but stops taking effect, the branches are structurized again (slower, same
    results).
    """
    return {"structurizecfg-skip-uniform-regions": bool(unstructured)}


# ---- ROCDL operations without a stable FlyDSL wrapper ----
#
# No stable equivalent, or the stable one adds instructions or rounds differently.


def _uniform(value):
    """Lane 0's value for the whole wave (v_readfirstlane_b32): an SGPR."""
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value)))


def readlane(value, lane):
    """Lane ``lane``'s value for the whole wave (v_readlane_b32)."""
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, value, lane))


def lane_id(mask):
    """The lane's index in its wave by mbcnt over mask, which must be all ones at run time: a
    mask LLVM cannot prove all ones keeps the count from folding into the thread id register
    (which would then stay live wherever the lane id is re-derived)."""
    mask = fx.Int32(mask).ir_value()
    low = rocdl.mbcnt_lo(fx.Int32.ir_type, mask, fx.Int32(0).ir_value())
    return fx.Int32(rocdl.mbcnt_hi(fx.Int32.ir_type, mask, low))


def ballot(predicate):
    """The wave's predicate bits."""
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, predicate))


def bpermute(address, value):
    """The value of the lane at byte ``address`` (ds_bpermute_b32)."""
    # fx.gpu.shuffle_idx scales the lane and checks its range on every call.
    return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, address, value))


def perm_b32(high, low, selector):
    """v_perm_b32: bytes of (high, low) picked by selector."""
    return rocdl.perm_b32(high, low, selector)


def sched_barrier(mask=0):
    """Compiler scheduling fence: only instruction classes in mask may cross it."""
    rocdl.sched_barrier(mask)


def s_barrier():
    """Workgroup rendezvous without memory fences."""
    # fx.gpu.barrier adds workgroup fences, i.e. waits for every memory counter.
    rocdl.s_barrier()


def s_setprio(level):
    """Wave issue priority."""
    rocdl.s_setprio(level)


def buffer_load_lds(resource, lds, size, voffset, soffset, offset):
    """LDS DMA of size bytes per lane (buffer_load ... lds) without a wait."""
    # rocdl.BufferCopyLDS* copy atoms add their own vmcnt wait.
    rocdl.raw_ptr_buffer_load_lds(resource, lds, size, voffset, soffset, offset)


def buffer_load_x3(resource, voffset, soffset):
    """Three 32-bit words by one buffer load (buffer_load_dwordx3) with a wave-uniform SGPR
    soffset: rocdl.BufferCopy has no 96-bit width (its i96 load does not legalize)."""
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(
            fx.Vector.make_type(3, fx.Int32), resource, voffset, soffset, 0
        )
    )


def buffer_atomic(op, resource, offset, value):
    """Buffer atomic ``op`` ("add", "swap", ...) of an Int32 at byte ``offset`` of the
    buffer, returning the old value; a lane at or past the buffer's extent does nothing and
    reads 0, so one lane updates a global counter without a divergent branch."""
    # rocdl.BufferAtomic* copy atoms return no old value; fx.atomic_add on a pointer needs a
    # branch around one lane (or puts every lane's request on one address).
    return fx.Int32(
        llvm.call_intrinsic(
            fx.Int32.ir_type,
            f"llvm.amdgcn.raw.ptr.buffer.atomic.{op}.i32",
            [
                fx.Int32(value).ir_value(),
                rocdl.get_buffer_rsrc(resource),
                fx.Int32(offset).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(0).ir_value(),
            ],
            [],
            [],
        )
    )


def dpp_move(value, control):
    """The 32-bit value moved across lanes by one DPP control (row-local, no LDS crossbar)."""
    raw = value.ir_value()
    return fx.Int32(
        rocdl.update_dpp(fx.Int32.ir_type, raw, raw, control, 0xF, 0xF, True)
    )


def dpp_banks(old, value, control, bank_mask):
    """FP32 value moved across lanes by one DPP control in the 4-lane banks of each 16-lane
    row that bank_mask enables, old in the others (one v_mov_b32_dpp)."""
    # dpp_move writes every bank (its old operand is the value itself).
    return fx.Float32(
        rocdl.update_dpp(
            fx.Float32.ir_type,
            fx.Float32(old).ir_value(),
            fx.Float32(value).ir_value(),
            control,
            0xF,
            bank_mask,
            False,
        )
    )


def lanes_below(mask):
    """Set bits of the 64-bit mask below this lane (v_mbcnt_lo/hi)."""
    low = rocdl.mbcnt_lo(
        fx.Int32.ir_type, fx.Int32(mask).ir_value(), fx.Int32(0).ir_value()
    )
    return fx.Int32(
        rocdl.mbcnt_hi(fx.Int32.ir_type, fx.Int32(mask >> 32).ir_value(), low)
    )


def row_broadcast(value, lane):
    """Lane ``lane`` of each 16-lane row (DPP row_newbcast); LLVM folds it into the VALU
    instruction that uses it."""
    # fx.gpu shuffles go through ds_bpermute or ds_swizzle and cannot fold into a VALU.
    return fx.Float32(
        rocdl.update_dpp(
            fx.Float32.ir_type,
            llvm.mlir_poison(fx.Float32.ir_type),
            fx.Float32(value).ir_value(),
            0x150 + lane,
            0xF,
            0xF,
            True,
        )
    )


def cvt_pk_fp8(a, b, word=None, high=False):
    """FP32 a, b as two E4M3FNUZ bytes (round to nearest even) in the low (high) half of
    word, its other half kept (v_cvt_pk_fp8_f32); no word starts a new one."""
    # Float32.to(Float8E4M3FNUZ) converts one value per instruction.
    old = llvm.mlir_poison(fx.Int32.ir_type) if word is None else fx.Int32(word)
    return fx.Int32(rocdl.cvt_pk_fp8_f32(fx.Int32.ir_type, a, b, old, high))


def fp8_pair(word, high):
    """The low (high) two E4M3FNUZ bytes of word as two FP32 values (one
    v_cvt_pk_f32_fp8): cvt_pk_fp8's inverse."""
    pair = fx.Vector(
        rocdl.cvt_pk_f32_fp8(fx.Vector.make_type(2, fx.Float32), fx.Int32(word), high)
    )
    return [fx.Float32(pair[0]), fx.Float32(pair[1])]


def smfmac_fp8(a, b, acc, index):
    """acc + A x B by one v_smfmac_f32_16x16x64_fp8_fp8 (E4M3FNUZ, 2:4 sparse A): a holds a
    lane's eight A values (two words), index's low 16 bits their K positions (two bits per
    value, two values per group of four keys), b sixteen B values (four words), acc four FP32.
    rocdl's MFMA atoms are dense; it exports no sparse MFMA."""
    return fx.Vector(
        rocdl.smfmac_f32_16x16x64_fp8_fp8(
            fx.Vector.make_type(4, fx.Float32),
            fx.Vector.from_elements([fx.Int32(w) for w in a], fx.Int32).ir_value(),
            fx.Vector.from_elements([fx.Int32(w) for w in b], fx.Int32).ir_value(),
            fx.Vector(acc).ir_value(),
            fx.Int32(index).ir_value(),
            0,
            0,
        ).result
    )


def log2(value):
    """log2 by one bare v_log_f32."""
    # fx.math.log2 adds denormal range reduction.
    return fx.Float32(rocdl.log(fx.Float32.ir_type, fx.Float32(value).ir_value()))


def rcp(value):
    """1 / value by one bare v_rcp_f32."""
    # Division lowers to a correctly rounded sequence of several instructions.
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, fx.Float32(value).ir_value()))


# ---- Volatile LDS reads and the lgkmcnt ledger ----


def _load_lds(storage, offset, immediate=0):
    """16 bytes of LDS at storage + offset + immediate (one ds_read_b128).

    Volatile keeps the hand-written order: LLVM neither merges nor reorders these
    loads, which the lgkmcnt ledger counts. The LDS variable's GEP gives it an alias scope.
    """
    ptr = fx.get_iter(storage) + (fx.Int32(offset) + immediate)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16), ptr
    )
    return fx.Vector(fx.generic_load(ptr, count=4, volatile=True))


def _load_lds32(storage, offset):
    """One LDS word (ds_read_b32) at storage + offset, volatile like _load_lds: it keeps its
    place among the LDS reads that a ledger counts."""
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4),
        fx.get_iter(storage) + offset,
    )
    return fx.Int32(fx.generic_load(ptr, count=1, volatile=True))


def store_lds(storage, offset, words, immediate=0):
    """4 words to LDS at storage + offset + immediate (one ds_write_b128): _load_lds's
    counterpart, for a wave that reads back what its other lanes stored."""
    data = fx.make_rmem_tensor(4, fx.Int32)
    data.store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32))
    pointer = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16),
        fx.get_iter(storage) + (fx.Int32(offset) + immediate),
    )
    fx.copy(
        fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
        data,
        fx.make_view(pointer, fx.make_layout(4, 1)),
    )


LGKMCNT_MAX = 15  # gfx942 s_waitcnt lgkmcnt field


class LgkmLedger:
    """Trace-time lgkmcnt ledger of a wave's LGKM operations (LDS reads and writes,
    ds_bpermute), which retire in issue order: operation i has landed at
    lgkmcnt(issued - i - 1). A kernel waits for the operations it uses instead of LLVM's
    wait before each consumer, and before a barrier lets other waves overwrite LDS it still
    reads (which LLVM cannot see).

    The counts are derived: every counted operation must be one instruction (_load_lds:
    volatile loads are neither merged nor reordered), issue()d in program order. `issued`
    operations so far (a ledger may start with some issued before it), the first `done`
    known complete. No SMEM load (also counted, out of order) may be issued while a
    partial wait is outstanding.
    """

    def __init__(self, issued=0):
        self.issued, self.done = issued, 0

    def issue(self):
        """Count one operation; returns its index."""
        self.issued += 1
        return self.issued - 1

    def settle(self, index):
        """The lgkmcnt a wait for operation index needs (None: known complete), with the
        ledger updated; emits nothing. Raises ValueError for an operation not issued."""
        if not 0 <= index < self.issued:
            raise ValueError(f"wait for operation {index} of {self.issued} issued")
        if index < self.done:
            return None
        # LLVM waits for at most LGKMCNT_MAX - 1 (it would strengthen a larger count).
        count = min(self.issued - index - 1, LGKMCNT_MAX - 1)
        self.done = self.issued - count
        return count

    def wait(self, index):
        """s_waitcnt until operation index has landed (nothing if known complete)."""
        count = self.settle(index)
        if count is not None:
            rocdl.s_waitcnt(lgkmcnt=count)
            # The consumers stay after the wait (else LLVM adds its own partial waits).
            sched_barrier(0)

    def all_done(self):
        """After an lgkmcnt(0) emitted outside the ledger."""
        self.done = self.issued


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
# phase timings over synthetic, dense and SGLang-captured selections (12k prompts at TP2/4/8,
# 16k-token chunks after 16k-1M prefixes) planned with per-tile ratios 0 / 1.2 / 2.0 / 3.0:
# union_step on the union kernel's times of unsplit plans (union_fixed kept; mean error 7%),
# direct_cu_step and row_step on the direct kernel's times of plans up to 128k-token contexts
# (direct_fixed kept; mean error 6%), pack_block on the pack's (apart from its Q quantization).
# The per-tile ratio is the sweep's best with them (from 1.7 down, short chunks after long
# prefixes send whole tiles direct and pack the context: +35%). Past the packed-KV budget, FP8
# direct packs up to 320 tokens per query row (its raw K/V reads cost more than BF16's). Union
# and minimum-row thresholds otherwise follow the BF16 entry.
FP8_ROUTES = {
    ("gfx942", 80): {
        "tensor": {
            "union_fixed": 5.7,
            "union_step": 1.61,
            "row_step": 0.54,
            "direct_fixed": 11.9,
            "pack_block": 0.0016,
            "direct_cu_step": 0.152,
            "union_ratio": 2.0,
            "pack_tokens_per_row": 320,
        },
        "token": {
            "union_fixed": 5.14,
            "union_step": 1.94,
            "row_step": 0.58,
            "direct_fixed": 11.87,
            "pack_block": 0.0018,
            "direct_cu_step": 0.158,
            "union_ratio": 2.0,
            "pack_tokens_per_row": 320,
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
            "%d CUs and reuses the %d-CU MI308X settings: plans and routes follow the CU "
            "count, performance may not be optimal",
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
    mma = fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16))
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
        fx.Uint32(perm_b32(rounded[i + 1], rounded[i], 0x07060302))
        for i in range(0, values.numel, 2)
    ]
    return fx.Vector.from_elements(words, fx.Uint32)


def _stage_end():
    # Compiler fences and CTA rendezvous, not a memory-counter wait.
    sched_barrier(0)
    s_barrier()
    sched_barrier(0)


def _buffer(tensor, size_bytes):
    """Byte-addressed buffer pointer over tensor; loads at or past size_bytes return zero."""
    return rocdl.make_buffer_ptr(
        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), num_records_bytes=size_bytes
    )


def _buffer_at(address, size_bytes):
    """Byte-addressed buffer pointer at a 16-byte aligned 64-bit global address; loads at or
    past size_bytes return zero."""
    pointer = fx.inttoptr(
        fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Global, 16),
        fx.Int64(address),
    )
    return rocdl.make_buffer_ptr(pointer, num_records_bytes=size_bytes)


def _buffer_words(buffer, offset, words=4, cache=0):
    data = fx.make_rmem_tensor(4 * words, fx.Int8)
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words, cache), fx.Int8),
        fx.make_view(buffer + offset, fx.make_layout(4 * words, 1)),
        data,
    )
    return fx.Vector(data.load()).bitcast(fx.Int32)


def _buffer_store_words(words, buffer, offset, cache=0):
    words = fx.Vector(words)
    data = fx.make_rmem_tensor(4 * words.numel, fx.Int8)
    data.store(words.bitcast(fx.Int8))
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopy(32 * words.numel, cache), fx.Int8),
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
        fx.make_mma_atom(rocdl.MFMA(16, 16, 16, fx.BFloat16)),
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


def _pv_step(p, operand, acc):
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
            perm_b32(
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
# A V scale enters as log2 and reciprocal; the clamp keeps both finite (NaN maps to the floor),
# and its 2**120 spread keeps a row's P' / v_scale, its row sum terms, within FP32's range.
SCALE_MIN, SCALE_MAX = 2.0**-60, 2.0**60
# Q rows of amax below Q_MIN quantize to 0 like zero rows: 240 / amax overflows FP32 below
# about 2**-120, and their scores are 0 to FP32 precision anyway.
Q_MIN = 2.0**-100
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
    mma = fx.make_mma_atom(rocdl.MFMA(m, m, 512 // m, fx.Float8E4M3FNUZ))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Vector(frag_c.load())


def _fp8_word(values):
    """Four FP32 -> one word of E4M3FNUZ (byte k = value k), round to nearest even."""
    return cvt_pk_fp8(values[2], values[3], cvt_pk_fp8(values[0], values[1]), True)


def _v_scale_terms(sv):
    """(log2(v_scale), 1 / v_scale) of a clamped V scale."""
    sv = fx.minnumf(
        fx.maxnumf(fx.Float32(sv), fx.Float32(SCALE_MIN)), fx.Float32(SCALE_MAX)
    )
    return log2(sv), rcp(sv)


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
    """Two BF16 words of four FP32 values times inv: FP32 multiplies, then round to nearest
    even like _pack_bf16 and the high halves joined by v_perm."""

    def rne(value):
        bits = _as_i32(fx.Float32(value) * inv)
        return bits + fx.Int32(0x7FFF) + ((bits >> 16) & fx.Int32(1))

    words = []
    for i in (0, 2):
        lo, hi = rne(values[i]), rne(values[i + 1])
        words.append(fx.Int32(perm_b32(hi, lo, 0x07060302)))
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
    inverse = (amax >= fx.Float32(Q_MIN)).select(rcp(amax) * FP8_MAX, fx.Float32(0.0))
    words = [
        _fp8_word([values[4 * k + r] * inverse for r in range(4)]) for k in range(4)
    ]
    return words, amax * (1.0 / FP8_MAX)


@flyc.jit
def quantize_rows(Q, Q8, QS, ROWS, cta):
    """Rows cta * QROWS .. + QROWS - 1 of Q (rows of 256 BF16) for a 256-thread CTA: row
    r * 16 + (thread >> 4) of the CTA's rows per step r; FP8 rows (E4M3FNUZ, RNE; 0 for
    rows of amax below Q_MIN) and their scales amax / 240 (0 for a zero row). Run by the
    FP8 pack kernel's CTAs past its blocks.
    """
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
        s_setprio(prio)
    else:
        s_setprio(0)


def _queue_take(queue, lane):
    """The queue's next item index (word 0), per lane: read it with readlane(..., 0).
    queue: the buffer of the launch's QUEUE_WORDS words."""
    beyond = fx.Int32(QUEUE_WORDS * 4)
    return buffer_atomic(
        "add", queue, (lane == 0).select(fx.Int32(0), beyond), fx.Int32(1)
    )


def _queue_exit(queue, lane, waves):
    """Count the wave out; the last of `waves` zeroes the taken and out words (every take
    of the launch is done once all its waves are out)."""
    beyond = fx.Int32(QUEUE_WORDS * 4)
    out = buffer_atomic(
        "add", queue, (lane == 0).select(fx.Int32(_QUEUE_OUT), beyond), fx.Int32(1)
    )
    last = (readlane(out, 0) == waves - 1) & (lane == 0)
    buffer_atomic("swap", queue, last.select(fx.Int32(0), beyond), fx.Int32(0))
    buffer_atomic("swap", queue, last.select(fx.Int32(_QUEUE_OUT), beyond), fx.Int32(0))
