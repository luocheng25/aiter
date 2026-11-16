# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Low-level primitives of the QSA sparse prefill kernels.

The sparse_prefill_* kernels use FlyDSL's stable API (``flydsl.expr`` and
``flydsl.expr.rocdl`` ``__all__``, ``flydsl.compiler``; FlyDSL docs/api_stability.md)
except for what this module gathers, each with the reason it stays. No other module
of these kernels imports ``flydsl._mlir`` or calls a ROCDL operation that ``fx.rocdl``
does not export.

- Empty inline asm (pin_i32, pin, keep_vgpr): FlyDSL 0.5.0 has no public asm entry or
  register-placement hint. A constraint LLVM cannot satisfy is only printed ("error:
  could not allocate output register for constraint 'v'") and the kernel computes
  garbage, so the guard tests check compiler output for errors.
- kernel_attrs: the target-features function attribute (LLVM semantics).
- ROCDL operations FlyDSL wraps without exporting (lane broadcasts, ballot, ds_bpermute,
  v_perm_b32, scheduling barriers, raw buffer loads, DPP, mbcnt, s_setprio, s_barrier):
  no stable equivalent, or the stable one adds instructions.
- load_lds and LdsQueue: volatile LDS reads and the lgkmcnt ledger of the union
  pipeline's cross-wave LDS reuse.
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


# ---- Kernel attributes ----


def kernel_attrs():
    """value_attrs of the sparse prefill kernels.

    Packed FP32 VALU (v_pk_*_f32) makes the union phase loop about 8% slower, and FlyDSL
    has no public target-feature option (no compile hint carries one): llvm.passthrough
    forwards the LLVM function attribute.
    """
    return {"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]}


# ---- ROCDL operations without a stable FlyDSL wrapper ----

# sched_group_barrier masks; fx.rocdl names only the MFMA, VMEM and DS ones.
VALU, MFMA, TRANS = 0x002, 0x008, 0x400


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


def sched_group_barrier(mask, size, group):
    """Schedule size instructions of class mask into pipeline group ``group``."""
    # fx.rocdl.sched_mfma etc. fix group 0 and name no VALU/TRANS class.
    rocdl.sched_group_barrier(mask, size, group)


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


def buffer_load_i16(resource, voffset, soffset):
    """One 16-bit buffer load with a wave-uniform SGPR soffset."""
    return rocdl.raw_ptr_buffer_load(fx.Int16.ir_type, resource, voffset, soffset, 0)


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
    """Collect (LDS ops issued since the last full wait, lgkmcnt) of every LdsQueue wait
    traced in the block: the pairs an ISA scan finds at the s_waitcnt instructions."""
    global _recorded
    saved, _recorded = _recorded, []
    try:
        yield _recorded
    finally:
        _recorded = saved


class LdsQueue:
    """lgkmcnt ledger of a wave's LDS reads and ds_bpermute, which retire in issue order.

    "The first n of them have landed" is lgkmcnt(issued - n). The union pipeline waits so
    before a barrier lets other waves overwrite LDS it still reads, which LLVM cannot see.
    The counts are derived: counted operations are emitted only through load() and
    bpermute(), one instruction each (volatile LDS loads are neither merged nor
    reordered). A ledger starts after every LDS op retired (lgkmcnt(0)), and no SMEM load
    (also counted, out of order) may be issued while it waits partially.
    """

    def __init__(self):
        self.issued = 0
        self.retired = 0  # issued at the last full wait

    def load(self, storage, offset, immediate=0):
        """load_lds, counted."""
        self.issued += 1
        return load_lds(storage, offset, immediate)

    def bpermute(self, address, value):
        self.issued += 1
        return bpermute(address, value)

    def ready(self, first, count):
        """ready(n) waits until the n first of the count ops issued from first landed."""

        def wait(done):
            if not 0 <= done <= count:
                raise ValueError(f"ready({done}) of {count} operations")
            self.wait(first + done)

        return wait

    def wait(self, upto=None, vmcnt=None):
        """Wait until the ops issued before index upto (default: all) have landed."""
        upto = self.issued if upto is None else upto
        if not 0 <= upto <= self.issued:
            raise ValueError(f"wait for {upto} of {self.issued} issued LDS operations")
        lgkmcnt = self.issued - upto
        if lgkmcnt > LGKMCNT_MAX:
            raise ValueError(f"lgkmcnt({lgkmcnt}) exceeds the counter")
        if _recorded is not None:
            _recorded.append((self.issued - self.retired, lgkmcnt))
        if lgkmcnt == 0:
            self.retired = self.issued
        fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
