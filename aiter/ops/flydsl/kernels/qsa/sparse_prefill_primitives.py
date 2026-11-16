# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Low-level primitives of the QSA sparse prefill kernels.

The sparse_prefill_* kernels use FlyDSL's stable API (``flydsl.expr`` and
``flydsl.expr.rocdl`` ``__all__``, ``flydsl.compiler``; FlyDSL docs/api_stability.md)
except for what this module gathers, each with the reason it stays. No other module
of these kernels imports ``flydsl._mlir`` or calls a ROCDL operation that ``fx.rocdl``
does not export.

- Empty inline asm (pin_i32, pin, keep_vgpr, hold, compiler_fence): FlyDSL 0.5.0 has no
  public asm entry, register-placement hint or compiler-only memory barrier. A constraint
  LLVM cannot satisfy is only printed ("error: could not allocate output register for
  constraint 'v'") and the kernel computes garbage, so the guard tests check compiler
  output for errors.
- kernel_attrs: the target-features function attribute (LLVM semantics).
- ROCDL operations FlyDSL wraps without exporting (lane broadcasts, ballot, ds_bpermute,
  v_perm_b32, scheduling barriers, raw buffer loads, DPP, mbcnt, s_setprio, s_barrier,
  FP8 conversion, bare log2 and reciprocal, the sparse FP8 MFMA): no stable equivalent, or
  the stable one adds instructions or rounds differently.
- load_lds and LgkmLedger: volatile LDS reads and the trace-time lgkmcnt ledger of the
  BF16 and FP8 unions' partial waits, cross-wave LDS reuse and reads across barriers.
- uniform_branch_options: the name of an internal LLVM option (a compile hint) that
  leaves wave-uniform branches unstructurized.
- buffer_atomic: an LLVM intrinsic call (llvm.amdgcn.raw.ptr.buffer.atomic.*): FlyDSL's
  buffer atomics are copy atoms without the old value, and a pointer atomic needs a
  divergent branch for a one-lane update.
"""

import contextlib

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl

# ---- Empty inline asm ----


def pin_i32(value):
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


def pin(values, chunk=8):
    """Vector values materialized in VGPRs at this point, chunk words per tied empty asm."""
    # A one-element vector constraint fails register allocation (only printed by LLVM).
    values = fx.Vector(values)
    dtype = values.dtype
    words = values.bitcast(fx.Int32) if dtype.width < 32 else values
    if words.numel % chunk == 1:
        raise ValueError("pin: a one-word chunk cannot be tied; use pin_i32")
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


def keep_vgpr(words, chunk=4):
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


# ---- Kernel attributes ----


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


# ---- ROCDL operations without a stable FlyDSL wrapper ----


def uniform(value):
    """Lane 0's value for the whole wave (v_readfirstlane_b32): an SGPR."""
    return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value)))


def readlane(value, lane):
    """Lane ``lane``'s value for the whole wave (v_readlane_b32)."""
    return fx.Int32(rocdl.readlane(fx.Int32.ir_type, value, lane))


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
    # fx.rocdl.BufferCopyLDS* copy atoms add their own vmcnt wait.
    rocdl.raw_ptr_buffer_load_lds(resource, lds, size, voffset, soffset, offset)


def dpp_move(value, control):
    """The 32-bit value moved across lanes by one DPP control (row-local, no LDS crossbar)."""
    raw = value.ir_value()
    return fx.Int32(
        rocdl.update_dpp(fx.Int32.ir_type, raw, raw, control, 0xF, 0xF, True)
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


def log2(value):
    """log2 by one bare v_log_f32."""
    # fx.math.log2 adds denormal range reduction.
    return fx.Float32(rocdl.log(fx.Float32.ir_type, fx.Float32(value).ir_value()))


def rcp(value):
    """1 / value by one bare v_rcp_f32."""
    # Division lowers to a correctly rounded sequence of several instructions.
    return fx.Float32(rocdl.rcp(fx.Float32.ir_type, fx.Float32(value).ir_value()))


# ---- Volatile LDS reads and the lgkmcnt ledger ----


def load_lds(storage, offset, immediate=0):
    """16 bytes of LDS at storage + offset + immediate (one ds_read_b128).

    Volatile keeps the hand-written order: LLVM neither merges nor reorders these
    loads, which the lgkmcnt ledger counts. The LDS variable's GEP gives it an alias scope.
    """
    ptr = fx.get_iter(storage) + (fx.Int32(offset) + immediate)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16), ptr
    )
    return fx.Vector(fx.generic_load(ptr, count=4, volatile=True))


LGKMCNT_MAX = 15  # gfx942 s_waitcnt lgkmcnt field
_recorded = None


@contextlib.contextmanager
def record_lds_waits():
    """Collect (LGKM operations issued since the last full wait, lgkmcnt) of every partial
    LgkmLedger wait traced in the block: the pairs an ISA scan finds at the s_waitcnt
    instructions."""
    global _recorded
    saved, _recorded = _recorded, []
    try:
        yield _recorded
    finally:
        _recorded = saved


class LgkmLedger:
    """Trace-time lgkmcnt ledger of a wave's LGKM operations (LDS reads and writes,
    ds_bpermute), which retire in issue order: operation i has landed at
    lgkmcnt(issued - i - 1). A kernel waits for the operations it uses instead of LLVM's
    wait before each consumer, and before a barrier lets other waves overwrite LDS it still
    reads (which LLVM cannot see).

    The counts are derived: every counted operation must be one instruction (load_lds:
    volatile loads are neither merged nor reordered), issue()d in program order. `issued`
    operations so far (a ledger may start with some issued before it), the first `done`
    known complete. No SMEM load (also counted, out of order) may be issued while a
    partial wait is outstanding.
    """

    def __init__(self, issued=0):
        # retired: issued at the last full wait (the guards' ISA scan counts from there)
        self.issued, self.done, self.retired = issued, 0, 0

    def issue(self):
        """Count one operation; returns its index."""
        self.issued += 1
        return self.issued - 1

    def settle(self, index):
        """The lgkmcnt a wait for operation index needs (None: known complete), with the
        ledger updated and a partial wait recorded (see record_lds_waits); emits nothing.
        Raises ValueError for an operation not issued."""
        if not 0 <= index < self.issued:
            raise ValueError(f"wait for operation {index} of {self.issued} issued")
        if index < self.done:
            return None
        # LLVM waits for at most LGKMCNT_MAX - 1 (it would strengthen a larger count).
        count = min(self.issued - index - 1, LGKMCNT_MAX - 1)
        if _recorded is not None and count > 0:
            _recorded.append((self.issued - self.retired, count))
        self.done = self.issued - count
        if count == 0:
            self.retired = self.issued
        return count

    def wait(self, index):
        """s_waitcnt until operation index has landed (nothing if known complete)."""
        count = self.settle(index)
        if count is not None:
            fx.rocdl.s_waitcnt(lgkmcnt=count)
            # The consumers stay after the wait (else LLVM adds its own partial waits).
            sched_barrier(0)

    def all_done(self):
        """After an lgkmcnt(0) emitted outside the ledger."""
        self.done = self.retired = self.issued


def lane_id(mask):
    """The lane's index in its wave by mbcnt over mask, which must be all ones at run time: a
    mask LLVM cannot prove all ones keeps the count from folding into the thread id register
    (which would then stay live wherever the lane id is re-derived)."""
    mask = fx.Int32(mask).ir_value()
    low = rocdl.mbcnt_lo(fx.Int32.ir_type, mask, fx.Int32(0).ir_value())
    return fx.Int32(rocdl.mbcnt_hi(fx.Int32.ir_type, mask, low))


def buffer_load_x3(resource, voffset, soffset):
    """Three 32-bit words by one buffer load (buffer_load_dwordx3) with a wave-uniform SGPR
    soffset: fx.rocdl.BufferCopy has no 96-bit width (its i96 load does not legalize).
    """
    return fx.Vector(
        rocdl.raw_ptr_buffer_load(
            fx.Vector.make_type(3, fx.Int32), resource, voffset, soffset, 0
        )
    )


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
    results), which the guard tests catch by compiling a layout with and without it.
    """
    return {"structurizecfg-skip-uniform-regions": bool(unstructured)}


def buffer_atomic(op, resource, offset, value):
    """Buffer atomic ``op`` ("add", "swap", ...) of an Int32 at byte ``offset`` of the
    buffer, returning the old value; a lane at or past the buffer's extent does nothing and
    reads 0, so one lane updates a global counter without a divergent branch."""
    # fx.rocdl.BufferAtomic* copy atoms return no old value; fx.atomic_add on a pointer
    # needs a branch around one lane (or puts every lane's request on one address).
    return fx.Int32(
        llvm.call_intrinsic(
            fx.Int32.ir_type,
            f"llvm.amdgcn.raw.ptr.buffer.atomic.{op}.i32",
            [
                fx.Int32(value).ir_value(),
                fx.rocdl.get_buffer_rsrc(resource),
                fx.Int32(offset).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(0).ir_value(),
            ],
            [],
            [],
        )
    )


def smfmac_fp8(a, b, acc, index):
    """acc + A x B by one v_smfmac_f32_16x16x64_fp8_fp8 (E4M3FNUZ, 2:4 sparse A): a holds a
    lane's eight A values (two words), index's low 16 bits their K positions (two bits per
    value, two values per group of four keys), b sixteen B values (four words), acc four FP32.
    fx.rocdl's MFMA atoms are dense; it exports no sparse MFMA."""
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


def store_lds(storage, offset, words, immediate=0):
    """4 words to LDS at storage + offset + immediate (one ds_write_b128): load_lds's
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


def fp8_pair(word, high):
    """The low (high) two E4M3FNUZ bytes of word as two FP32 values (one
    v_cvt_pk_f32_fp8): cvt_pk_fp8's inverse."""
    pair = fx.Vector(
        rocdl.cvt_pk_f32_fp8(fx.Vector.make_type(2, fx.Float32), fx.Int32(word), high)
    )
    return [fx.Float32(pair[0]), fx.Float32(pair[1])]
