# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Guards of the QSA sparse prefill kernels' low-level code (compile only, untimed).

The kernels keep a few low-level primitives, all in sparse_prefill_primitives.py: empty
inline asm pinning values in VGPRs or holding them at a point, the "-packed-fp32-ops"
target feature, ROCDL operations FlyDSL does not export (DPP, FP8 conversion, bare log2
and reciprocal, ...), volatile LDS reads counted by an lgkmcnt ledger (prim.LgkmLedger,
kept while a union is traced), and the direct key loop's hand-counted vmcnt waits.
When FlyDSL, MLIR or LLVM stop honouring them, nothing fails: LLVM only prints an asm
constraint it cannot meet and the kernel computes garbage, a dropped attribute or a split
load only costs speed or makes a counted wait wrong. These checks make such changes fail.

One process compiles eleven layouts of test_flydsl_qsa_sparse_prefill.py with FlyDSL's IR
dumps (FLYDSL_DUMP_IR; dumping bypasses FlyDSL's disk cache) of the kernels it checks; K1-K3,
which it does not check, compile without them, so a warm disk cache supplies those. BF16,
at group 12:
prefill_12000 (union, raw direct and packed direct), kv_slots_ragged_4 (slotted union;
at group 6 too, kv_slots_ragged_4_g6, where the union splits into two key ranges),
kv_slots_pool_4g_ragged_4 (the union and raw direct over a pool past 4 GiB: the union
variant with the most registers, 512 VGPRs and AGPRs),
kv_slots_pool_4g_extend_64_ctx_70065 (raw direct over a pool past 4 GiB, the WIDE
variant) and, at group 3, topk256_chunk_1024_ctx_31024_g3 (every row selects topk blocks
after a 30000-token prefix: the packed direct launch runs persistent, beside a split
union). FP8 (FP8_CASES entries, at their own group): fp8_ragged_4_token (union with
core/residual tiles, direct with the residual merge, pack, per token scales),
fp8_kv_slots_ragged_4_tensor (the same through a slot table, per tensor),
fp8_kv_slots_topk1024_mixed_ctx_132096_token (union split into two key ranges, T =
1024, direct without a merge) and, GUARD_FP8_CASES entries the op test does not run,
fp8_kv_slots_hk2_mixed_t768_token (hk2_mixed_t768 through a slot table at group 12, per
token, core/residual tiles, with its direct kernel: the union variant with the most SGPR
spills) and
fp8_ragged_4_tensor (fp8_ragged_4_token per tensor: without a slot table the union's
follower sums the rows and holds l). It then compiles the FP8 union of
fp8_ragged_4_token once more with its uniform-branch LLVM option off (UNIFORM_LABEL).
Their final ISA and LLVM IR of the union, raw direct and packed direct kernels, and of
the FP8 union, direct and pack kernels, must show:

  compile output   no "error:" line while compiling, and finite outputs
  resources        union: no scratch or VGPR spill, SGPR spills at most
                   UNION_SGPR_SPILLS', one wave per SIMD, the split variant exactly
                   for UNION_SPLIT; direct kernels: no scratch and no spill
  FP8 resources    no scratch; FP8 union: two waves per SIMD (leader and follower
                   alternate M-stages), at most UNION_FP8_SPILLS VGPR and SGPR spills;
                   FP8 direct and pack: no VGPR spill, two and eight waves per SIMD; pack:
                   no SGPR spill; direct: SGPR spills to VGPR lanes, but only outside its
                   chunk loop (FP8 direct chunk loop)
  target features  "target-features"="-packed-fp32-ops" on every kernel and the
                   "amdgpu-waves-per-eu" of WAVES_PER_EU in LLVM IR; no v_pk_*_f32
  empty asm        the baseline number of ;;#ASMSTART blocks (BASELINE, FP8_BASELINE;
                   a BF16 split adds UNION_SPLIT_ASM), all empty, and of each LLVM IR asm
                   constraint (FP8: prim.hold's "v", prim.compiler_fence's "~{memory}")
  lgkmcnt ledger   the (LDS operations since the last lgkmcnt(0), N) pairs of union's
                   partial lgkmcnt(N) waits, recorded by its trace-time ledger
                   (prim.LgkmLedger, collected by record_lds_waits), equal those a scan
                   of its ISA finds, with no SMEM load counted at any of them. FP8
                   union, whose ISA also holds LLVM's partial waits for its volatile LDS
                   reads (each before a use), so that the LDS operations since the last
                   lgkmcnt(0) depend on the path: as many barriers LGKM operations of
                   their basic block may be in flight across (by every wait in the block,
                   LLVM's too) as the ledger recorded waits, only LDS reads in flight
                   across them, a partial wait last before one a count the ledger
                   recorded; it records some exactly for UNION_FP8_GATED
  vmcnt counts     one vmcnt(8) and one vmcnt(16) in each direct kernel: on every path
                   into vmcnt(16) 24 loads since the loop's vmcnt(0) (V segment 1, then
                   the 16 K loads it keeps in flight), fewer only past another drain;
                   at vmcnt(8) 8 or 9 loads since vmcnt(16) (V segment 0 and a block-id
                   prefetch). LLVM adds any wait a register needs (it turned a vmcnt(9)
                   there into vmcnt(8)), so a stale count costs speed, not results: it
                   waits for loads the loop means to keep in flight
  persistent direct
                   the persistent packed direct (PERSISTENT: 512-thread WGs of one-row
                   waves) narrows exec nowhere: its lane-0 queue and count updates stay
                   PERSIST_ATOMICS buffer atomics whose other lanes are out of range (no
                   divergent branch, no atomicrmw); and it keeps its 2 s_setprio (the
                   pace with its SIMD sibling)
  exp2             every exp2 lowered to llvm.amdgcn.exp2.f32 (no llvm.exp2.f32 or
                   __ocml_exp2), the baseline number of them (a BF16 split adds
                   UNION_SPLIT_EXP2)
  FP8 MFMA         FP8 union: v_mfma_f32_32x32x16_fp8_fp8 and, per tensor, the
                   v_smfmac_f32_16x16x64_fp8_fp8 of its row sums (ROW_SUM_MFMA); direct:
                   v_mfma_f32_16x16x32_fp8_fp8 only; the baseline numbers; none writes
                   AGPRs (no asm keeps O there any more)
  FP8 log2 / rcp   prim.log2 / prim.rcp as llvm.amdgcn.log.f32 / llvm.amdgcn.rcp.f32 (no
                   llvm.log2, __ocml_log or __ocml_rcp), the baseline number of each
  FP8 conversion   prim.cvt_pk_fp8 and prim.fp8_pair: the baseline numbers of
                   v_cvt_pk_fp8_f32 (encode) and v_cvt_pk_f32_fp8 (decode, its _e32 and
                   _sdwa forms), and no other FP8 or BF8 conversion
  FP8 direct chunk loop
                   the FP8 direct kernel's chunk loops (the innermost loops of its final
                   ISA, by back edges, that hold its MFMAs) have no v_readlane or
                   v_writelane: SGPR spills, if any, reload outside them (once per row of
                   the persistent plain launch), whatever their number
  uniform branches every FP8 union variant is compiled with
                   prim.uniform_branch_options' LLVM option (an internal cl::opt that
                   leaves wave-uniform branches unstructurized): UNIFORM_LABEL's union
                   has fewer s_cbranch with it than compiled again without it, and no
                   more SGPR spills. If LLVM drops the option, setting it fails the
                   compile ("compile output"); if it stays but stops taking effect, the
                   branches are structurized again, which only costs speed

and, compiling no kernel: the lgkmcnt ledger's own assertions, prim.pin rejecting a one-word chunk
while tracing, a source audit that the kernel modules (every sparse_prefill_*.py but
sparse_prefill_primitives.py) use FlyDSL's stable API only (FlyDSL
docs/api_stability.md): no flydsl._mlir or bare rocdl import, no inline_asm,
Attribute.parse or llvm.passthrough, every fx./flyc. path in its module's __all__ and
none of the deprecated ones (the deprecated Numeric/Vector methods, shuffle_xor, exp2,
..., are reported on any receiver except the fx modules that export them as functions,
fx.gpu.shuffle_xor and fx.math.exp2), and that the FP8 modules wait for zero
outstanding operations (s_waitcnt arguments all 0) or, in sparse_prefill_union_fp8.py
only, by a prim.LgkmLedger (no hand-counted wait), that no kernel module names an LLVM
option prim.uniform_branch_options sets (only sparse_prefill_primitives.py does), and that
every kernel launch passes value_attrs=prim.kernel_attrs() but KERNEL_ATTRS_EXEMPT's (K1-K3
included, which "target features" does not see).
The FP8 direct kernel's unsigned record and LSE offsets past 2 GiB are checked by the op
test (fp8_limit_packed_token, fp8_hk16_prefill_24576_token), not here.

Usage:
    python op_tests/test_flydsl_qsa_sparse_prefill_guards.py [--keep DIR]
"""

import argparse
import ast
import collections
import contextlib
import ctypes
import importlib
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels import qsa as qsa_kernels
from aiter.ops.flydsl.kernels.qsa import sparse_prefill_primitives as prim

SUPPORTED_GFX = ["gfx942"]
OP_TEST = Path(__file__).with_name("test_flydsl_qsa_sparse_prefill.py")
UNION = "sparse_prefill_union_bf16_d256"
DIRECT = "sparse_prefill_direct_bf16_d256"
PACKED = "sparse_prefill_direct_packed_bf16_d256"
UNION_FP8 = "sparse_prefill_union_fp8_d256"
DIRECT_FP8 = "sparse_prefill_direct_fp8_d256"
PACK_FP8 = "sparse_prefill_pack_kv_fp8_d256"
SHORT = {
    UNION: "union",
    DIRECT: "direct",
    PACKED: "packed",
    UNION_FP8: "union_fp8",
    DIRECT_FP8: "direct_fp8",
    PACK_FP8: "pack_fp8",
}
TOPK1024 = "fp8_kv_slots_topk1024_mixed_ctx_132096_token"
SLOTS_HK2_TOKEN = "fp8_kv_slots_hk2_mixed_t768_token"
PACKED_TENSOR = "fp8_ragged_4_tensor"
# FP8 layouts as FP8_CASES entries that only the guards compile (PACKED_TENSOR: per
# tensor without a slot table, so its union's follower sums the rows and holds l)
GUARD_FP8_CASES = {
    SLOTS_HK2_TOKEN: ("hk2_mixed_t768", 12, "token", "fp8", (None, 1, False)),
    PACKED_TENSOR: ("ragged_4", 12, "tensor", "fp8", None),
}
# label -> (a layout or SLOT_CASES entry of the op test, Q heads per KV head); FP8
# labels are FP8_CASES or GUARD_FP8_CASES entries, at their own group (None).
LAYOUTS = {
    "prefill_12000": ("prefill_12000", 12),
    "kv_slots_ragged_4": ("kv_slots_ragged_4", 12),
    "kv_slots_ragged_4_g6": ("kv_slots_ragged_4", 6),
    "kv_slots_pool_4g_ragged_4": ("kv_slots_pool_4g_ragged_4", 12),
    "kv_slots_pool_4g_extend_64_ctx_70065": (
        "kv_slots_pool_4g_extend_64_ctx_70065",
        12,
    ),
    "topk256_chunk_1024_ctx_31024_g3": ("topk256_chunk_1024_ctx_31024", 3),
    "fp8_ragged_4_token": ("fp8_ragged_4_token", None),
    "fp8_kv_slots_ragged_4_tensor": ("fp8_kv_slots_ragged_4_tensor", None),
    TOPK1024: (TOPK1024, None),
    SLOTS_HK2_TOKEN: (SLOTS_HK2_TOKEN, None),
    PACKED_TENSOR: (PACKED_TENSOR, None),
}
# label -> the union, direct and pack kernels its call must compile (the packed direct
# kernel and the pack of kv_slots_pool_4g_ragged_4 are kv_slots_ragged_4's, and the
# direct of PACKED_TENSOR is fp8_kv_slots_ragged_4_tensor's, compiled already); by name,
# so that adding a layout anywhere in LAYOUTS leaves the others' entries alone
EXPECTED = {
    "prefill_12000": (UNION, DIRECT, PACKED),
    "kv_slots_ragged_4": (UNION, DIRECT, PACKED),
    "kv_slots_ragged_4_g6": (UNION, DIRECT, PACKED),
    "kv_slots_pool_4g_ragged_4": (UNION, DIRECT),
    "kv_slots_pool_4g_extend_64_ctx_70065": (DIRECT, PACKED),
    # the persistent packed direct (PERSISTENT) and a split union at group 3
    "topk256_chunk_1024_ctx_31024_g3": (UNION, PACKED),
    "fp8_ragged_4_token": (UNION_FP8, DIRECT_FP8, PACK_FP8),
    "fp8_kv_slots_ragged_4_tensor": (UNION_FP8, DIRECT_FP8, PACK_FP8),
    TOPK1024: (UNION_FP8, DIRECT_FP8, PACK_FP8),
    SLOTS_HK2_TOKEN: (UNION_FP8, DIRECT_FP8, PACK_FP8),
    PACKED_TENSOR: (UNION_FP8, PACK_FP8),
}
# kernel -> (;;#ASMSTART blocks, LLVM IR asm constraints, llvm.amdgcn.exp2.f32 calls).
# Union: prim.hold of S (32 values) in the prologue's and the key loop's softmax and
# prim.keep_vgpr of the V loads (8 per V write stream, loop and epilogue) are the 80 "v";
# 18 prim.pin_i32 ("=v,0") of task and lane values and of the next work item's index and
# ORDER entry across the key loop; 32 exp2 of P in the loop and in the epilogue, one of
# the lazy rescale. Direct: prim.pin of the zero O registers (8 chunks); 8 exp2 of P and
# one of the rescale, and packed adds the 4 part weights of a split slot's merge.
BASELINE = {
    UNION: (98, {"=v,0": 18, "v": 80}, 65),
    DIRECT: (8, {"=v,0": 8}, 9),
    PACKED: (8, {"=v,0": 8}, 13),
}
# label -> most SGPR spills of its union kernel: none without a split; with one, at most 8
# over the union's 57 reachable variants, and the 7- and 8-spill variants reload them
# outside the key loop.
UNION_SGPR_SPILLS = {
    "prefill_12000": 0,
    "kv_slots_ragged_4": 0,
    "kv_slots_ragged_4_g6": 8,
    "kv_slots_pool_4g_ragged_4": 0,
    "topk256_chunk_1024_ctx_31024_g3": 8,
}
# Labels whose BF16 union plan splits long tasks into two key ranges (K3 below twice the
# CUs' tasks): their union counts its parts with a buffer atomic, its merge (a branch per
# merging part index: two at two parts) adds 4 exp2, and prim.pin_i32 keeps the partial
# records' and counters' address words in VGPRs across the key loop (4) and pins the lane
# in each of the three output paths (3: an unsplit task's and each merge branch's), while
# the next work item stays in SGPRs (-2): 5 more "=v,0" asm in LLVM IR and blocks in the
# ISA. Their SGPR spills: 2 slotted at group 6, none at group 3 (at most 8 over the 57
# reachable union variants).
UNION_SPLIT = {"kv_slots_ragged_4_g6", "topk256_chunk_1024_ctx_31024_g3"}
UNION_SPLIT_EXP2 = 4
UNION_SPLIT_ASM = (5, 5)
# Labels whose packed direct launch runs persistent (direct.prepare: packed, rows x HK >= CUs
# x PERSIST_WAVES, every first query sees 4 x topk keys): 512-thread WGs, and one more
# prim.pin_i32 (the lane, per row) than the one-wave kernel: 9 "=v,0" asm. Its buffer
# atomics: a split row's part count, the queue take, the exit count and the two resets
# (lane 0's offsets, the other lanes' out of range); its s_setprio: the pace (behind its
# SIMD sibling or not).
PERSISTENT = {"topk256_chunk_1024_ctx_31024_g3"}
PERSISTENT_ASM = (1, 1)
PERSIST_WG = 512
PERSIST_ATOMICS = 5
PERSIST_SETPRIO = 2
# (FP8 label, kernel) -> (;;#ASMSTART blocks, MFMA instructions, llvm.amdgcn.exp2.f32,
# llvm.amdgcn.log.f32 and llvm.amdgcn.rcp.f32 calls, FP8 encodes, FP8 decodes). Union counts per
# instance of its leader and follower steps (11 centrings, 11 exp blocks of 16 exp2 and 8
# conversions, 320 MFMAs): per token the union reads log2 and 1 / v_scale from the pack's
# scale records (no empty asm, log2 or rcp of its own); a split adds 4 exp2 (two per group's
# merge); the residual merge adds 2 exp2, a log2 and an rcp to direct and a log2 storing the
# log-sum-exp to union; pack adds log2 / rcp per token scale.
# Direct runs four copies of a chunk (two per loop iteration, an odd one, the last chunk
# apart), each 32 MFMAs, 9 exp2, 4 encodes of P' and 4 decodes of them (its row sums add the
# FP8 P' that PV multiplies), and fences the load groups of the three that issue loads.
FP8_BASELINE = {
    ("fp8_ragged_4_token", UNION_FP8): (0, 320, 185, 1, 0, 88, 0),
    ("fp8_ragged_4_token", DIRECT_FP8): (3, 128, 38, 1, 1, 16, 16),
    ("fp8_ragged_4_token", PACK_FP8): (0, 0, 0, 1, 3, 16, 0),
    ("fp8_kv_slots_ragged_4_tensor", UNION_FP8): (0, 320, 185, 1, 0, 88, 0),
    ("fp8_kv_slots_ragged_4_tensor", DIRECT_FP8): (3, 128, 38, 1, 1, 16, 16),
    ("fp8_kv_slots_ragged_4_tensor", PACK_FP8): (0, 0, 0, 0, 2, 16, 0),
    (TOPK1024, UNION_FP8): (0, 320, 189, 0, 0, 88, 0),
    (TOPK1024, DIRECT_FP8): (3, 128, 36, 0, 0, 16, 16),
    (TOPK1024, PACK_FP8): (0, 0, 0, 1, 3, 16, 0),
    (SLOTS_HK2_TOKEN, UNION_FP8): (0, 320, 185, 1, 0, 88, 0),
    (SLOTS_HK2_TOKEN, DIRECT_FP8): (3, 128, 38, 1, 1, 16, 16),
    (SLOTS_HK2_TOKEN, PACK_FP8): (0, 0, 0, 1, 3, 16, 0),
    (PACKED_TENSOR, UNION_FP8): (0, 320, 185, 1, 0, 88, 0),
    (PACKED_TENSOR, PACK_FP8): (0, 0, 0, 0, 2, 16, 0),
}
# FP8 conversions by opcode (suffix _e32 / _e64 / _sdwa / _dpp dropped): prim.cvt_pk_fp8's
# encode and prim.fp8_pair's decode; any other FP8 or BF8 conversion fails "FP8 conversion".
FP8_ENCODE, FP8_DECODE = "v_cvt_pk_fp8_f32", "v_cvt_pk_f32_fp8"
FP8_CONVERSION = re.compile(r"^(v_cvt_\w*?[fb]p?8\w*?)(?:_e32|_e64|_sdwa|_dpp)?$")
# FP8 kernel -> the constraint of its empty asm, its MFMA instruction
FP8_ASM = {UNION_FP8: "v", DIRECT_FP8: "~{memory}"}
FP8_MFMA = {
    UNION_FP8: "v_mfma_f32_32x32x16_fp8_fp8",
    DIRECT_FP8: "v_mfma_f32_16x16x32_fp8_fp8",
}
# Per tensor the FP8 union sums the FP8 P' words of a tile with two sparse 16x16x64
# MFMAs per M-stage with PV (its l; the leader with a slot table, the follower without:
# PACKED_TENSOR): (FP8 label, kernel) -> their number.
ROW_SUM_MFMA = "v_smfmac_f32_16x16x64_fp8_fp8"
FP8_ROW_SUMS = {
    ("fp8_kv_slots_ragged_4_tensor", UNION_FP8): 10,
    (PACKED_TENSOR, UNION_FP8): 10,
}
# The FP8 module that may count lgkmcnt waits, by a prim.LgkmLedger only, and the FP8
# labels whose union does: it keeps LDS reads in flight across a barrier, waiting for the
# LDS operations before them (lgkmcnt ledger). The FP8 direct and pack wait for zero.
FP8_LEDGER_MODULE = "sparse_prefill_union_fp8.py"
UNION_FP8_GATED = {"fp8_kv_slots_ragged_4_tensor"}
# (module, @flyc.kernel function) launched without prim.kernel_attrs(), whose
# "-packed-fp32-ops" "target features" checks on the dumped kernels only: compact (K2)
# measures slower with it, and the BF16 K/V packs have no FP32 arithmetic.
KERNEL_ATTRS_EXEMPT = {
    ("sparse_prefill_prepare.py", "_compact_kernel"),
    ("sparse_prefill_prepare.py", "_compact_kernel_large"),
    ("sparse_prefill_direct_packed.py", "_pack"),
    ("sparse_prefill_direct_packed.py", "_pack_gated"),
}
# The layout whose FP8 union is compiled a second time with the uniform-branch option off
# (every FP8 union variant takes the option): 42 / 56 s_cbranch and 20 / 38 SGPR spills
# with / without it; fp8_kv_slots_hk2_mixed_t768_token's union: 76 SGPR spills with it.
UNIFORM_LABEL = "fp8_ragged_4_token"
# kernel -> waves per SIMD (union kernels and FP8 direct set it by waves_per_eu)
WAVES = {UNION: 1, UNION_FP8: 2, DIRECT_FP8: 2, PACK_FP8: 8}
WAVES_PER_EU = {UNION: "1", UNION_FP8: "2", DIRECT_FP8: "2"}
# Most VGPR and SGPR spills of the FP8 union. Two waves per SIMD leave no AGPRs, so any VGPR
# spill would take scratch: none. SGPR spills (kernel arguments and the task's buffers, live
# through the key loops) go to VGPR lanes outside them: at most 77 over the union's 88
# reachable variants (group 6, two KV heads, per token, slots, no split; with the 24
# core/residual ones, which dense layouts no longer reach: their selections are fixed), 76 in
# fp8_kv_slots_hk2_mixed_t768_token's union (group 12, core/residual tiles), so the bound holds
# for every variant. PyHIP's version used up to 60 B of scratch and 14 VGPR spills.
UNION_FP8_SPILLS = (0, 88)
# The FP8 direct's persistent plain launch (TOPK1024's rows all select topk blocks) keeps its
# kernel arguments and row-loop counts live through the row loop: they spill to VGPR lanes
# (FP8 resources prints how many) and reload once per row. Their number is not bounded; that
# none is in a chunk loop is (FP8 direct chunk loop).
ISA_LABEL = re.compile(r"^(\.L\w+):")
ISA_BRANCH = re.compile(r"^s_(?:cbranch_\w+|branch)\s+(\.L\w+)")
LANE_SPILL_OPS = ("v_readlane_b32", "v_writelane_b32")
LGKMCNT = re.compile(r"lgkmcnt\((\d+)\)")
VMEM_LOAD = re.compile(r"^(buffer|global)_load")
SMEM_LOAD = re.compile(r"^s_(buffer_)?load")


# ---- Compiling with dumps ----


def load_op_test():
    spec = importlib.util.spec_from_file_location("qsa_sparse_prefill_op_test", OP_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def captured(path):
    """Everything written to file descriptors 1 and 2 (LLVM's diagnostics included)."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(path, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            libc.fflush(None)
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


def compile_layout(op, label, root):
    """One qsa_sparse_prefill call of a layout (inputs and seed as the op test makes
    them; FP8 as test_qsa_fp8) with its kernels dumped to root/label. Returns (outputs
    finite, the (LDS operations, lgkmcnt) pairs the ledgers recorded while tracing)."""
    case, group = LAYOUTS[label]
    if case in op.FP8_CASES or case in GUARD_FP8_CASES:
        layout, group, scales, compute, pool = {**op.FP8_CASES, **GUARD_FP8_CASES}[case]
        query_lens, prefix_lens, kv_heads = op.CASES[layout][:3]
        heads = group * kv_heads
        generator = torch.Generator(device="cuda").manual_seed(op.fp8_seed(case, heads))
        q, k, k_scale, v, v_scale, indices = op.make_fp8_inputs(
            layout, heads, scales == "token", generator
        )
        kwargs = {"k_scale": k_scale, "v_scale": v_scale, "compute": compute}
        if pool is not None:
            pool_rows, first_page, last_page = pool
            k, v, slots = op.make_pool(
                k,
                v,
                query_lens,
                prefix_lens,
                pool_rows,
                first_page,
                generator,
                last_page,
            )
            kwargs["kv_slots"] = slots
            kwargs["k_scale"] = op.pool_scale(k_scale, slots, k.shape[0])
            kwargs["v_scale"] = op.pool_scale(v_scale, slots, k.shape[0])
    else:
        layout, pool = case, None
        if case in op.SLOT_CASES:
            layout, *pool = op.SLOT_CASES[case]
        query_lens, prefix_lens, kv_heads = op.CASES[layout][:3]
        heads = group * kv_heads
        generator = torch.Generator(device="cuda").manual_seed(
            list(op.CASES).index(layout) * 16 + heads
        )
        q, k, v, indices = op.make_inputs(layout, heads, generator)
        kwargs = {}
        if pool is not None:
            k, v, kwargs["kv_slots"] = op.make_pool(
                k, v, query_lens, prefix_lens, *pool, generator
            )
    kwargs.update(query_lens=query_lens, prefix_lens=prefix_lens)
    os.environ["FLYDSL_DUMP_DIR"] = str(root / label)
    with captured(root / f"{label}.log"), prim.record_lds_waits() as waits:
        out = op.qsa_sparse_prefill(q, k, v, indices, softmax_scale=op.SCALE, **kwargs)
        torch.cuda.synchronize()
    return bool(torch.isfinite(out).all()), list(waits)


@contextlib.contextmanager
def plan_kernels_undumped():
    """K1-K3 (sparse_prefill_prepare._compiled) compile without FlyDSL's IR dumps for the
    duration: no check reads their dumps, writing them (some 23 IR stages of each) took
    about a sixth of the guards' time, and without dumping FlyDSL may take them from its
    disk cache (dumping bypasses it)."""
    prepare = importlib.import_module(
        "aiter.ops.flydsl.kernels.qsa.sparse_prefill_prepare"
    )
    compiled = prepare._compiled

    def undumped(*args, **kwargs):
        dump = os.environ.pop("FLYDSL_DUMP_IR", None)
        try:
            return compiled(*args, **kwargs)
        finally:
            if dump is not None:
                os.environ["FLYDSL_DUMP_IR"] = dump

    prepare._compiled = undumped
    try:
        yield
    finally:
        prepare._compiled = compiled


@contextlib.contextmanager
def uniform_branches_off():
    """FP8 union compiles with prim.uniform_branch_options' option off: the hint is
    replaced for the duration and the union's compile cache cleared before and after (the
    hints are part of FlyDSL's compile key, so its kernel compiles again)."""
    union = importlib.import_module(
        "aiter.ops.flydsl.kernels.qsa.sparse_prefill_union_fp8"
    )
    options = prim.uniform_branch_options
    prim.uniform_branch_options = lambda unstructured: options(False)
    union._COMPILED.clear()
    try:
        yield
    finally:
        prim.uniform_branch_options = options
        union._COMPILED.clear()


# ---- Reading the dumps ----


def latest(directory, suffix):
    files = sorted(
        directory.glob(f"*_{suffix}"), key=lambda path: int(path.name.split("_")[0])
    )
    return files[-1]


def metadata(text):
    """{kernel: {key: value}} of the .amdgpu_metadata block of a final ISA."""
    block = re.search(
        r"\.amdgpu_metadata\n(.*?)\n\s*\.end_amdgpu_metadata", text, re.DOTALL
    )
    kernels, current = [], None
    for line in block.group(1).splitlines():
        match = re.match(r"^  (- |  )\.(\w+):\s*(\S*)", line)
        if match is None:
            continue
        if match.group(1) == "- ":
            current = {}
            kernels.append(current)
        value = match.group(3)
        current[match.group(2)] = int(value) if value.isdigit() else value
    return {kernel["name"]: kernel for kernel in kernels}


def isa_kernels(path):
    """{kernel: (instruction and label lines, metadata)} of a final ISA dump."""
    text = path.read_text()
    symbols = set(re.findall(r"^\s*\.globl\s+(\S+)", text, re.MULTILINE))
    bodies, name = {}, None
    for line in text.splitlines():
        label = re.match(r"^(\S+):", line)
        if label and label.group(1) in symbols:
            name = label.group(1)
            bodies[name] = []
        elif name and re.match(r"^\.Lfunc_end\d+:", line):
            name = None
        elif name:
            bodies[name].append(line.strip())
    meta = metadata(text)
    return {name: (body, meta[name]) for name, body in bodies.items()}


def ir_functions(path):
    """{kernel: (function text, its attribute group text)} of an LLVM IR dump."""
    text = path.read_text()
    groups = dict(re.findall(r"^attributes #(\d+) = \{(.*)\}$", text, re.MULTILINE))
    functions = {}
    for match in re.finditer(
        r"^define [^@]*@([\w.]+)\(.*?\) #(\d+)[^\n]*\{\n(.*?)^\}$",
        text,
        re.MULTILINE | re.DOTALL,
    ):
        functions[match.group(1)] = (match.group(3), groups[match.group(2)])
    return functions


def read_dumps(root, labels=None):
    """{(label, kernel): {"isa", "meta", "ir", "attributes"}} of the dumped union and
    direct kernels (of LAYOUTS, or the given labels)."""
    kernels = {}
    for label in labels or LAYOUTS:
        for directory in sorted(p for p in (root / label).iterdir() if p.is_dir()):
            functions = ir_functions(latest(directory, "llvm_ir.ll"))
            for name, (body, meta) in isa_kernels(
                latest(directory, "final_isa.s")
            ).items():
                if name not in SHORT:
                    continue
                ir, attributes = functions[name]
                kernels[(label, name)] = {
                    "isa": body,
                    "meta": meta,
                    "ir": ir,
                    "attributes": attributes,
                }
    return kernels


def instructions(body):
    for line in body:
        if line and not line.startswith((";", ".", "//")):
            yield line, line.split()[0]


# ---- Checks of the compiled kernels ----


def waves_per_simd(meta):
    """gfx942: 512 VGPRs per lane (granule 8), 800 SGPRs (granule 16), 64 KiB LDS per CU
    shared by 4 SIMDs, at most 8 waves per SIMD."""
    by_vgpr = 512 // (-(-meta["vgpr_count"] // 8) * 8)
    by_sgpr = 800 // (-(-meta["sgpr_count"] // 16) * 16)
    waves = -(-meta["max_flat_workgroup_size"] // 64)
    lds = meta["group_segment_fixed_size"]
    by_lds = 65536 // lds * waves // 4 if lds else 8
    return min(8, by_vgpr, by_sgpr, by_lds)


def check_resources(kernels):
    found = []
    for (label, name), kernel in kernels.items():
        if name not in BASELINE:
            continue
        meta = kernel["meta"]
        spills = (meta["vgpr_spill_count"], meta["sgpr_spill_count"])
        scratch = meta["private_segment_fixed_size"]
        limit = UNION_SGPR_SPILLS[label] if name == UNION else 0
        assert scratch == 0 and spills[0] == 0 and spills[1] <= limit, (
            f"{label} {name}: scratch {scratch} B, VGPR/SGPR spills {spills} "
            f"(at most 0 B, 0 and {limit})"
        )
        waves = waves_per_simd(meta)
        if name == UNION:
            assert (
                waves == WAVES[UNION]
            ), f"{label} {name}: {waves} waves per SIMD, not {WAVES[UNION]}"
            split = "llvm.amdgcn.raw.ptr.buffer.atomic.add" in kernel["ir"]
            assert split == (
                label in UNION_SPLIT
            ), f"{label} {name}: split variant {split}, expected {label in UNION_SPLIT}"
        found.append(f"{label} {SHORT[name]} {meta['vgpr_count']}v/{waves}w")
    return "; ".join(found)


def check_target_features(kernels):
    for (label, name), kernel in kernels.items():
        features = re.search(r'"target-features"="([^"]*)"', kernel["attributes"])
        assert features and "-packed-fp32-ops" in features.group(1).split(","), (
            f"{label} {name}: LLVM IR function attributes lack -packed-fp32-ops: "
            f"{kernel['attributes']}"
        )
        if name in WAVES_PER_EU:
            waves = f'"amdgpu-waves-per-eu"="{WAVES_PER_EU[name]}"'
            assert (
                waves in kernel["attributes"]
            ), f"{label} {name}: {waves} missing: {kernel['attributes']}"
        packed = [
            op for _, op in instructions(kernel["isa"]) if re.match(r"v_pk_\w*_f32", op)
        ]
        assert not packed, f"{label} {name}: packed FP32 VALU {sorted(set(packed))}"
    return "present; no v_pk_*_f32"


def asm_baseline(label, name):
    """(;;#ASMSTART blocks, LLVM IR asm constraints) a kernel must have."""
    if name == UNION and label in UNION_SPLIT:
        blocks, constraints = BASELINE[name][:2]
        constraints = {**constraints, "=v,0": constraints["=v,0"] + UNION_SPLIT_ASM[1]}
        return blocks + UNION_SPLIT_ASM[0], constraints
    if name == PACKED and label in PERSISTENT:
        blocks, constraints = BASELINE[name][:2]
        constraints = {**constraints, "=v,0": constraints["=v,0"] + PERSISTENT_ASM[1]}
        return blocks + PERSISTENT_ASM[0], constraints
    if name in BASELINE:
        return BASELINE[name][:2]
    blocks = FP8_BASELINE[(label, name)][0]
    return blocks, {FP8_ASM[name]: blocks} if blocks else {}


def check_empty_asm(kernels):
    found = []
    for (label, name), kernel in kernels.items():
        expected_blocks, expected_constraints = asm_baseline(label, name)
        blocks, inside, nonempty = 0, False, 0
        for line in kernel["isa"]:
            if line.startswith(";;#ASMSTART"):
                blocks, inside = blocks + 1, True
            elif line.startswith(";;#ASMEND"):
                inside = False
            elif inside and line:
                nonempty += 1
        constraints = collections.Counter(
            re.findall(r'asm sideeffect "", "([^"]*)"', kernel["ir"])
        )
        nonempty_ir = re.findall(r'asm sideeffect "([^"]+)"', kernel["ir"])
        assert (blocks, nonempty, dict(constraints), nonempty_ir) == (
            expected_blocks,
            0,
            expected_constraints,
            [],
        ), (
            f"{label} {name}: {blocks} asm blocks ({nonempty} non-empty lines), IR "
            f"constraints {dict(constraints)}, non-empty IR asm {nonempty_ir}; "
            f"expected {expected_blocks} empty blocks and {expected_constraints}"
        )
        found.append(f"{label} {SHORT[name]} {blocks}")
    return "; ".join(found)


def lgkmcnt_scan(body):
    """(LDS operations, SMEM loads) issued since the last lgkmcnt(0), and N, of every
    partial s_waitcnt lgkmcnt(N) in text order."""
    lds = smem = 0
    found = []
    for line, op in instructions(body):
        if op.startswith("ds_"):
            lds += 1
        elif SMEM_LOAD.match(op):
            smem += 1
        elif op == "s_waitcnt" and (match := LGKMCNT.search(line)):
            count = int(match.group(1))
            if count == 0:
                lds = smem = 0
            else:
                found.append((lds, smem, count))
    return found


def lgkm_crossings(body):
    """Every s_barrier LGKM operations may be in flight across by its basic block: issued
    in the block since its last lgkmcnt(0), or kept by a partial lgkmcnt(N) wait last
    before the barrier (no LGKM operation between). Per barrier: (that N, else None; the
    opcodes possibly in flight, oldest first, "s_load" for an SMEM load and "?" for any
    the block inherits). Each wait in the block counts, LLVM's too (lgkmcnt(N) keeps the
    last N issued)."""
    found = []
    for index, block in enumerate(blocks_of(body)[0]):
        flight, issued, count = () if index == 0 else ("?",), False, None
        for line in block:
            op = line.split()[0]
            if op.startswith("ds_") or SMEM_LOAD.match(op):
                flight += ("s_load" if SMEM_LOAD.match(op) else op,)
                issued, count = True, None
            elif op == "s_waitcnt" and (match := LGKMCNT.search(line)):
                count = int(match.group(1))
                flight = flight[len(flight) - count :] if count else ()
                if count == 0:
                    issued, count = False, None
            elif op == "s_barrier":
                if flight and (issued or count):
                    found.append((count, flight))
                count = None
    return found


def check_lgkmcnt_ledger(kernels, recorded):
    found = []
    for label, waits in recorded.items():
        ledger = collections.Counter(pair for pair in waits if pair[1] > 0)
        if (label, UNION) in kernels:
            scan = lgkmcnt_scan(kernels[(label, UNION)]["isa"])
            smem = [wait for wait in scan if wait[1]]
            isa = collections.Counter((lds, count) for lds, _, count in scan)
            assert ledger and ledger == isa and not smem, (
                f"{label} {UNION}: ledger {dict(ledger)} vs ISA {dict(isa)}, "
                f"waits with SMEM in flight {smem}"
            )
            found.append(f"{label} {dict(sorted(ledger.items()))}")
        elif (label, UNION_FP8) in kernels:
            # LLVM's own partial waits, each before a use of what it retires, make the LDS
            # operations since the last lgkmcnt(0) path dependent: compared instead, the
            # barriers the ledger's waits let reads cross.
            crossings = lgkm_crossings(kernels[(label, UNION_FP8)]["isa"])
            counts = collections.Counter(count for _, count in ledger.elements())
            isa = collections.Counter(count for count, _ in crossings if count)
            unsafe = [
                (count, ops)
                for count, ops in crossings
                if not all(op.startswith("ds_read") for op in ops)
            ]
            gated = label in UNION_FP8_GATED
            assert (
                bool(ledger) == gated
                and len(crossings) == counts.total()
                and not isa - counts
                and not unsafe
            ), (
                f"{label} {UNION_FP8}: ledger waits {dict(counts)} (expected some: "
                f"{gated}), {len(crossings)} barriers crossed in the ISA, at lgkmcnt "
                f"{dict(isa)}; crossed by another LGKM operation than an LDS read: {unsafe}"
            )
            found.append(
                f"{label} {len(crossings)} barriers crossed by LDS reads, ledger waits "
                f"{dict(counts)}"
            )
        else:
            assert not ledger, f"{label}: ledger waits {ledger} without a union kernel"
    return "; ".join(found)


def blocks_of(body):
    """Basic blocks [(instructions, successor indices)] and predecessor lists."""
    blocks, labels = [[]], {}
    for line in body:
        label = re.match(r"^(\.LBB\d+_\d+):", line)
        if label:
            labels[label.group(1)] = len(blocks)
            blocks.append([])
            continue
        if not line or line.startswith((";", ".", "//")):
            continue
        blocks[-1].append(line)
        if line.startswith(("s_branch", "s_cbranch", "s_endpgm")):
            blocks.append([])
    successors = []
    for index, block in enumerate(blocks):
        last = block[-1].split() if block else [""]
        following = [index + 1] if index + 1 < len(blocks) else []
        if last[0] == "s_branch":
            successors.append([labels[last[1]]])
        elif last[0].startswith("s_cbranch"):
            successors.append([labels[last[1]], *following])
        elif last[0] == "s_endpgm":
            successors.append([])
        else:
            successors.append(following)
    predecessors = [[] for _ in blocks]
    for index, targets in enumerate(successors):
        for target in targets:
            predecessors[target].append(index)
    return blocks, predecessors


def is_drain(line):
    return line.startswith("s_waitcnt") and "vmcnt(0)" in line


def loads_back(blocks, predecessors, block, index, until):
    """Over every path into instruction index of block, walked backwards, the VMEM loads
    issued since the last instruction until() accepts; a path through a vmcnt(0) first,
    or reaching the kernel entry, gives none."""
    counts, seen, stack = set(), set(), [(block, index, 0)]
    while stack:
        block, index, count = stack.pop()
        if (block, index, count) in seen or count > 64:
            continue
        seen.add((block, index, count))
        for line in reversed(blocks[block][:index]):
            if until(line):
                counts.add(count)
                break
            if VMEM_LOAD.match(line):
                count += 1
            elif is_drain(line):
                break
        else:
            stack.extend((p, len(blocks[p]), count) for p in predecessors[block])
    return counts


def check_vmcnt(kernels):
    """The direct loop's waits, per iteration: drain (vmcnt(0)), V segment 1 (two
    values(), 4 loads each), K (two load_key(), 8 each), vmcnt(16) keeps K in flight and
    retires V; V segment 0 (8) and maybe a block-id prefetch, vmcnt(8) retires K. Every
    path into vmcnt(16) must carry at most 24 loads since a drain, 24 without one inside
    the loop body (the K loads always follow); at vmcnt(8), 8 or 9 loads since vmcnt(16)
    on every path not drained in between."""
    found = []
    for (label, name), kernel in kernels.items():
        if name not in (DIRECT, PACKED):
            continue
        blocks, predecessors = blocks_of(kernel["isa"])
        waits = {
            count: [
                (b, i)
                for b, block in enumerate(blocks)
                for i, line in enumerate(block)
                if line.startswith("s_waitcnt") and f"vmcnt({count})" in line
            ]
            for count in (8, 16)
        }
        assert all(len(places) == 1 for places in waits.values()), (
            f"{label} {name}: vmcnt(8) at {waits[8]}, vmcnt(16) at {waits[16]}; "
            "expected one of each"
        )
        (b16, i16), (b8, i8) = waits[16][0], waits[8][0]
        since_drain = loads_back(blocks, predecessors, b16, i16, is_drain)
        since_16 = loads_back(
            blocks, predecessors, b8, i8, lambda line: "vmcnt(16)" in line
        )
        assert (
            since_drain
            and max(since_drain) == 24
            and min(since_drain) >= 16
            and since_16
            and min(since_16) == 8
            and max(since_16) <= 9
        ), (
            f"{label} {name}: loads since a drain at vmcnt(16) {sorted(since_drain)} "
            f"(expected at most 24, 24 on the main path, at least the 16 K loads); "
            f"since vmcnt(16) at vmcnt(8) {sorted(since_16)} (expected 8, or 9 with "
            "a block-id prefetch)"
        )
        found.append(f"{label} {SHORT[name]} {sorted(since_drain)}/{sorted(since_16)}")
    return "; ".join(found)


def check_persistent(kernels):
    found = []
    for label in sorted(PERSISTENT):
        kernel = kernels.get((label, PACKED))
        assert kernel, f"{label}: no {PACKED} dump"
        meta = kernel["meta"]
        assert meta["max_flat_workgroup_size"] == PERSIST_WG, (
            f"{label} {PACKED}: {meta['max_flat_workgroup_size']}-thread WGs, not the "
            f"persistent launch's {PERSIST_WG}"
        )
        ops = collections.Counter(op for _, op in instructions(kernel["isa"]))
        narrowing = sorted(op for op in ops if "exec" in op or op.startswith("v_cmpx"))
        exec_writes = [
            line for line in kernel["isa"] if re.match(r"^s_\w+\s+exec(_lo)?\s*,", line)
        ]
        assert not narrowing and not exec_writes, (
            f"{label} {PACKED}: exec narrowed by {narrowing} {exec_writes[:3]} "
            "(a divergent branch)"
        )
        atomicrmw = kernel["ir"].count("atomicrmw")
        buffer = len(
            re.findall(r"@llvm\.amdgcn\.raw\.ptr\.buffer\.atomic\.\w+", kernel["ir"])
        )
        assert not atomicrmw and buffer == PERSIST_ATOMICS, (
            f"{label} {PACKED}: {atomicrmw} atomicrmw, {buffer} buffer atomics in LLVM IR "
            f"(expected 0 and {PERSIST_ATOMICS})"
        )
        assert ops["s_setprio"] == PERSIST_SETPRIO, (
            f"{label} {PACKED}: {ops['s_setprio']} s_setprio (expected "
            f"{PERSIST_SETPRIO})"
        )
        found.append(
            f"{label} {meta['vgpr_count']}v/{meta['max_flat_workgroup_size']} threads, "
            f"no exec narrowing, {buffer} buffer atomics, {ops['s_setprio']} s_setprio"
        )
    return "; ".join(found)


def check_exp2(kernels):
    found = []
    for (label, name), kernel in kernels.items():
        amdgcn = len(re.findall(r"call float @llvm\.amdgcn\.exp2\.f32\(", kernel["ir"]))
        other = re.findall(r"@llvm\.exp2\.\w+|__ocml_exp2\w*", kernel["ir"])
        expected = (
            BASELINE[name][2] if name in BASELINE else FP8_BASELINE[(label, name)][2]
        )
        if name == UNION and label in UNION_SPLIT:
            expected += UNION_SPLIT_EXP2
        assert amdgcn == expected and not other, (
            f"{label} {name}: {amdgcn} llvm.amdgcn.exp2.f32 (expected {expected}), "
            f"other exp2 {sorted(set(other))}"
        )
        found.append(f"{label} {SHORT[name]} {amdgcn}")
    return "; ".join(found)


def fp8_only(kernels):
    return [
        (label, name, kernel)
        for (label, name), kernel in kernels.items()
        if name in (UNION_FP8, DIRECT_FP8, PACK_FP8)
    ]


def check_fp8_resources(kernels):
    found = []
    for label, name, kernel in fp8_only(kernels):
        meta = kernel["meta"]
        spills = (meta["vgpr_spill_count"], meta["sgpr_spill_count"])
        scratch = meta["private_segment_fixed_size"]
        limit = UNION_FP8_SPILLS if name == UNION_FP8 else (0, 0)
        if name == DIRECT_FP8:
            # Its SGPR spills: FP8 direct chunk loop.
            limit = (0, spills[1])
        waves = waves_per_simd(meta)
        assert (
            scratch == 0
            and spills[0] <= limit[0]
            and spills[1] <= limit[1]
            and waves == WAVES[name]
        ), (
            f"{label} {name}: scratch {scratch} B, VGPR/SGPR spills {spills} (at most "
            f"0 B and {limit}), {waves} waves per SIMD (expected {WAVES[name]})"
        )
        found.append(
            f"{label} {SHORT[name]} {meta['vgpr_count']}v/{spills[1]}s/{waves}w"
        )
    return "; ".join(found)


def innermost_loops(body, opcode):
    """(first, last) line ranges of the innermost loops of an ISA body that hold opcode: a
    loop runs from a label to the last branch back to it."""
    at = {m.group(1): i for i, line in enumerate(body) if (m := ISA_LABEL.match(line))}
    ends = {}
    for i, line in enumerate(body):
        m = ISA_BRANCH.match(line)
        if m and m.group(1) in at and at[m.group(1)] <= i:
            ends[at[m.group(1)]] = max(ends.get(at[m.group(1)], i), i)
    holding = [
        (first, last)
        for first, last in ends.items()
        if any(
            line.split(maxsplit=1)[0] == opcode
            for line in body[first : last + 1]
            if line
        )
    ]
    return sorted(
        (first, last)
        for first, last in holding
        if not any(
            (a, b) != (first, last) and first <= a and b <= last for a, b in holding
        )
    )


def check_fp8_direct_loops(kernels):
    found = []
    for label, name, kernel in fp8_only(kernels):
        if name != DIRECT_FP8:
            continue
        body = kernel["isa"]
        loops = innermost_loops(body, FP8_MFMA[DIRECT_FP8])
        assert loops, f"{label} {name}: no loop holds {FP8_MFMA[DIRECT_FP8]}"
        for first, last in loops:
            lane_ops = [
                line.split(maxsplit=1)[0]
                for line in body[first : last + 1]
                if line and line.split(maxsplit=1)[0] in LANE_SPILL_OPS
            ]
            assert not lane_ops, (
                f"{label} {name}: {len(lane_ops)} v_readlane / v_writelane in the chunk "
                f"loop of lines {first}-{last} ({kernel['meta']['sgpr_spill_count']} "
                "SGPR spills)"
            )
        mfmas = [
            sum(
                1
                for line in body[first : last + 1]
                if line and line.split(maxsplit=1)[0] == FP8_MFMA[DIRECT_FP8]
            )
            for first, last in loops
        ]
        found.append(
            f"{label} {'/'.join(map(str, mfmas))} MFMAs, "
            f"{kernel['meta']['sgpr_spill_count']} SGPR spills outside"
        )
    return "; ".join(found)


def check_fp8_mfma(kernels):
    found = []
    for label, name, kernel in fp8_only(kernels):
        ops = collections.Counter(
            op
            for _, op in instructions(kernel["isa"])
            if op.startswith(("v_mfma", "v_smfmac"))
        )
        count = FP8_BASELINE[(label, name)][1]
        expected = {FP8_MFMA[name]: count} if count else {}
        if (label, name) in FP8_ROW_SUMS:
            expected[ROW_SUM_MFMA] = FP8_ROW_SUMS[(label, name)]
        agpr = [line for line in kernel["isa"] if re.match(r"v_s?mfma\S*\s+a\[", line)]
        assert dict(ops) == expected and not agpr, (
            f"{label} {name}: MFMA {dict(ops)} (expected {expected}), AGPR "
            f"destinations {agpr[:3]}"
        )
        if count:
            found.append(f"{label} {SHORT[name]} {count}")
    return "; ".join(found)


def check_fp8_log_rcp(kernels):
    found = []
    for label, name, kernel in fp8_only(kernels):
        ir = kernel["ir"]
        counts = (
            len(re.findall(r"call float @llvm\.amdgcn\.log\.f32\(", ir)),
            len(re.findall(r"call float @llvm\.amdgcn\.rcp\.f32\(", ir)),
        )
        other = sorted(set(re.findall(r"@llvm\.log2?\.\w+|__ocml_(?:log|rcp)\w*", ir)))
        expected = FP8_BASELINE[(label, name)][3:5]
        assert counts == expected and not other, (
            f"{label} {name}: llvm.amdgcn.log/rcp {counts} (expected {expected}), "
            f"other log2/rcp {other}"
        )
        found.append(f"{label} {SHORT[name]} {counts[0]}/{counts[1]}")
    return "; ".join(found)


def check_fp8_conversion(kernels):
    found = []
    for label, name, kernel in fp8_only(kernels):
        ops = [
            m.group(1)
            for _, op in instructions(kernel["isa"])
            if (m := FP8_CONVERSION.match(op))
        ]
        counts = (ops.count(FP8_ENCODE), ops.count(FP8_DECODE))
        other = sorted(set(ops) - {FP8_ENCODE, FP8_DECODE})
        expected = FP8_BASELINE[(label, name)][5:7]
        assert counts == expected and not other, (
            f"{label} {name}: {FP8_ENCODE} / {FP8_DECODE} {counts} (expected "
            f"{expected}), other FP8 conversions {other}"
        )
        found.append(f"{label} {SHORT[name]} {counts[0]}/{counts[1]}")
    return "; ".join(found)


# ---- Checks without compiling a kernel ----


def check_ledger_assertions():
    """prim.LgkmLedger refuses a wait for an operation not issued before it emits s_waitcnt,
    and clamps a count past the lgkmcnt field to LGKMCNT_MAX - 1 (recorded as such)."""
    cases = {
        "wait(3) of 3 issued": (3, 3),
        "wait(4) of 3 issued": (3, 4),
        "wait(-1)": (3, -1),
    }
    accepted = []
    for label, (issued, index) in cases.items():
        try:
            prim.LgkmLedger(issued).wait(index)
            accepted.append(label)
        except ValueError:
            pass
        except Exception as error:  # noqa: BLE001
            # Past its checks: emitting s_waitcnt outside a kernel trace fails.
            accepted.append(f"{label} ({type(error).__name__})")
    assert not accepted, f"LgkmLedger accepted {accepted}"
    clamp = prim.LGKMCNT_MAX - 1
    ledger = prim.LgkmLedger(16)
    with prim.record_lds_waits() as waits:
        count = ledger.settle(0)
    assert (
        count == clamp and waits == [(16, clamp)] and ledger.done == 16 - clamp
    ), f"wait(0) of 16 issued: lgkmcnt({count}), recorded {waits}, done {ledger.done}"
    return f"{len(cases)} rejected, wait(0) of 16 issued clamped to lgkmcnt({clamp})"


@flyc.kernel(name="qsa_guard_pin_probe", known_block_size=[64, 1, 1])
def _pin_probe(OUT: fx.Tensor, WORDS: fx.Constexpr[int]):
    prim.pin(fx.Vector.filled(WORDS, 0.0, fx.Float32))


@flyc.jit
def _pin_launch(OUT: fx.Tensor, WORDS: fx.Constexpr[int], stream: fx.Stream):
    _pin_probe(OUT, WORDS).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def check_pin_rejects_one_word():
    """prim.pin raises while tracing when a chunk would be a single word, which LLVM
    cannot allocate (it prints an error and the kernel computes garbage)."""
    accepted = []
    for words in (1, 9):
        try:
            flyc.compile(
                _pin_launch, torch.empty(1), words, torch.cuda.current_stream()
            )
            accepted.append(words)
        except ValueError as error:
            frames = [
                (frame.f_code.co_name, frame.f_code.co_filename)
                for frame, _ in traceback.walk_tb(error.__traceback__)
            ]
            assert (
                "pin",
                prim.__file__,
            ) in frames, f"not prim.pin's ValueError: {error}"
    assert not accepted, f"prim.pin accepted one-word chunks of {accepted} words"
    return "1 and 9 words rejected"


def stable_names():
    """FlyDSL's stable flydsl.expr names (docs/api_stability.md section 2.1): the
    __all__ of the direct-child modules expr/__init__.py star-imports, and the backend
    entry points of _BACKEND_MODULES."""
    init = Path(fx.__file__)
    tree = ast.parse(init.read_text())
    children = [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and [alias.name for alias in node.names] == ["*"]
    ]
    names = {}
    for child in children:
        module = importlib.import_module(f"flydsl.expr.{child}")
        for name in module.__all__:
            names.setdefault(name, child)
    backends = {name for name in fx._BACKEND_MODULES if name != "tdm_ops"}
    return children, names, backends


DEPRECATED_PATHS = {
    ("get",),
    ("index_cast",),
    ("constant_vector",),
    ("tdm_ops",),
    ("rocdl", "BufferCopyLDS64b"),
}
DEPRECATED_METHODS = {"maximumf", "minimumf", "shrui", "addf", "exp2", "shuffle_xor"}


def audit_source(path, stable):
    """(line, use) of every unstable or deprecated FlyDSL use in one kernel module."""
    children, names, backends = stable
    compiler_all = set(flyc.__all__)
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    modules, aliases, problems = set(), {}, []
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.asname or alias.name.split(".")[0])
                if alias.name in ("flydsl.expr", "flydsl.compiler"):
                    aliases[alias.asname or alias.name] = alias.name
                elif alias.name.split(".")[0] == "flydsl":
                    problems.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            modules.update(alias.asname or alias.name for alias in node.names)
            if (node.module or "").split(".")[0] == "flydsl":
                problems.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            if not isinstance(node.value, str):
                continue
            if "llvm.passthrough" in node.value or "target-features" in node.value:
                problems.append((node.lineno, f"string {node.value!r}"))
        elif isinstance(node, ast.Name) and node.id in ("inline_asm", "_mlir"):
            problems.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and (
            node.attr in ("inline_asm", "_mlir")
            or (node.attr == "parse" and getattr(node.value, "attr", "") == "Attribute")
        ):
            problems.append((node.lineno, node.attr))

    def members(rest, lineno, text):
        # Members of a stable object are stable unless underscore-prefixed.
        if any(name.startswith("_") and not name.startswith("__") for name in rest):
            problems.append((lineno, f"{text}: underscore member"))

    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in seen:
            continue
        parts, current = [], node
        while isinstance(current, ast.Attribute):
            seen.add(id(current))
            parts.append(current.attr)
            current = current.value
        parts.reverse()
        root = current.id if isinstance(current, ast.Name) else None
        if root not in aliases:
            # A deprecated Numeric/Vector method on a value (not a module function).
            if root not in modules and DEPRECATED_METHODS.intersection(parts):
                problems.append((node.lineno, f"deprecated method in {parts}"))
            continue
        text = f"{root}." + ".".join(parts)
        if aliases[root] == "flydsl.compiler":
            if parts[0] not in compiler_all:
                problems.append((node.lineno, f"{text}: not in compiler.__all__"))
            continue
        if tuple(parts[:1]) in DEPRECATED_PATHS or tuple(parts[:2]) in DEPRECATED_PATHS:
            problems.append((node.lineno, f"{text}: deprecated"))
        elif parts[0] in backends:
            # Every namespace down to the name must be in its parent's __all__.
            owner, rest = importlib.import_module(f"flydsl.expr.{parts[0]}"), parts[1:]
            while rest and isinstance(owner, types.ModuleType):
                if rest[0] == "experimental" or rest[0] not in getattr(
                    owner, "__all__", ()
                ):
                    problems.append((node.lineno, f"{text}: {rest[0]} not exported"))
                    break
                owner, rest = getattr(owner, rest[0]), rest[1:]
            else:
                members(rest, node.lineno, text)
        elif parts[0] in children and len(parts) > 1:
            module = importlib.import_module(f"flydsl.expr.{parts[0]}")
            if parts[1] not in module.__all__:
                problems.append((node.lineno, f"{text}: not in its __all__"))
            else:
                members(parts[2:], node.lineno, text)
        elif parts[0] in names:
            members(parts[1:], node.lineno, text)
        else:
            problems.append((node.lineno, f"{text}: not a stable flydsl.expr name"))
    return problems


def kernel_sources():
    directory = Path(qsa_kernels.__path__[0])
    return sorted(
        path
        for path in directory.glob("sparse_prefill_*.py")
        if path.name != "sparse_prefill_primitives.py"
    )


def check_sources(paths=None):
    stable = stable_names()
    problems = {
        path.name: found
        for path in (paths or kernel_sources())
        if (found := audit_source(path, stable))
    }
    assert not problems, f"unstable or deprecated FlyDSL uses: {problems}"
    return f"{len(paths or kernel_sources())} modules"


def check_fp8_waits(paths=None):
    """The FP8 modules wait for zero outstanding operations, every s_waitcnt argument 0,
    or by a prim.LgkmLedger in the FP8 union only (its partial waits: lgkmcnt ledger), so
    they keep no hand-counted wait."""
    paths = paths or [path for path in kernel_sources() if "fp8" in path.name]
    counted = {}
    for path in paths:
        for node in ast.walk(ast.parse(Path(path).read_text(), filename=str(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "s_waitcnt"
            ):
                values = [*node.args, *(keyword.value for keyword in node.keywords)]
                if not values or not all(
                    isinstance(value, ast.Constant) and value.value == 0
                    for value in values
                ):
                    counted.setdefault(path.name, []).append(node.lineno)
            elif path.name != FP8_LEDGER_MODULE and "LgkmLedger" in (
                getattr(node, "id", None),
                getattr(node, "attr", None),
            ):
                counted.setdefault(path.name, []).append(node.lineno)
    assert paths and not counted, f"counted waits in FP8 modules {paths}: {counted}"
    return f"{len(paths)} modules, a ledger in {FP8_LEDGER_MODULE} only"


def check_llvm_options(paths=None):
    """No kernel module names an LLVM option prim.uniform_branch_options sets: the
    internal option names stay in sparse_prefill_primitives.py."""
    names = set(prim.uniform_branch_options(True))
    named = {
        path.name: found
        for path in (paths or kernel_sources())
        if (found := sorted(name for name in names if name in path.read_text()))
    }
    assert names and not named, f"LLVM options named outside the primitives: {named}"
    return f"{', '.join(sorted(names))}: in sparse_prefill_primitives.py only"


def check_kernel_attrs(paths=None):
    """Every launch of a @flyc.kernel function of the kernel modules, K1-K3 included,
    passes value_attrs=prim.kernel_attrs(), but those of KERNEL_ATTRS_EXEMPT."""
    kernels, missing = set(), {}
    for path in map(Path, paths or kernel_sources()):
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "kernel"
                for decorator in node.decorator_list
            )
        }
        launched = collections.Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in names:
                    continue
                launched[node.func.id] += 1
                attrs = [kw.value for kw in node.keywords if kw.arg == "value_attrs"]
                if (path.name, node.func.id) not in KERNEL_ATTRS_EXEMPT and not (
                    len(attrs) == 1
                    and isinstance(attrs[0], ast.Call)
                    and ast.unparse(attrs[0].func) == "prim.kernel_attrs"
                ):
                    missing.setdefault(path.name, []).append(node.lineno)
        unlaunched = names - set(launched)
        assert (
            not unlaunched
        ), f"{path.name}: kernels never launched {sorted(unlaunched)}"
        kernels |= {(path.name, name) for name in names}
    stale = KERNEL_ATTRS_EXEMPT - kernels
    assert kernels and not missing and not stale, (
        f"launches without prim.kernel_attrs(): {missing}; "
        f"exempt kernels not found: {sorted(stale)}"
    )
    return (
        f"{len(kernels) - len(KERNEL_ATTRS_EXEMPT)} kernels with it, "
        f"{len(KERNEL_ATTRS_EXEMPT)} exempt"
    )


def branches_and_spills(kernel):
    branches = sum(op.startswith("s_cbranch") for _, op in instructions(kernel["isa"]))
    return branches, kernel["meta"]["sgpr_spill_count"]


def check_uniform_branches(on, off, log, finite):
    """UNIFORM_LABEL's FP8 union compiled with its uniform-branch option (on) and without
    (off, uniform_branches_off): fewer s_cbranch with it (the flow blocks StructurizeCFG
    adds around wave-uniform branches) and no more SGPR spills. Relative, so that other
    changes to the kernel move both."""
    errors = [
        line
        for line in log.read_text(errors="replace").splitlines()
        if re.search(r"\berror:", line)
    ]
    assert on and off and finite and not errors, (
        f"{UNIFORM_LABEL} union without the option: compiled again {off is not None} (not "
        "when the hints are the same with and without it), finite "
        f"{finite}, compiler errors {errors}"
    )
    (branches_on, spills_on), (branches_off, spills_off) = (
        branches_and_spills(on),
        branches_and_spills(off),
    )
    assert branches_on < branches_off and spills_on <= spills_off, (
        f"{UNIFORM_LABEL} union: {branches_on} s_cbranch and {spills_on} SGPR spills with "
        f"prim.uniform_branch_options' option, {branches_off} and {spills_off} without: it "
        "no longer leaves the wave-uniform branches unstructurized"
    )
    return (
        f"{UNIFORM_LABEL} union {branches_on} / {branches_off} s_cbranch, {spills_on} / "
        f"{spills_off} SGPR spills (option on / off)"
    )


# ---- Driver ----


def check_compile_output(root, finite, seconds):
    errors = {}
    for label in LAYOUTS:
        log = (root / f"{label}.log").read_text(errors="replace").splitlines()
        if found := [line for line in log if re.search(r"\berror:", line)]:
            errors[label] = found
    assert not errors and all(
        finite.values()
    ), f"compiler errors {errors}; outputs finite {finite}"
    return ", ".join(f"{label} {seconds[label]:.1f} s" for label in LAYOUTS)


def check_dumped(kernels):
    missing = [
        (label, name)
        for label, names in EXPECTED.items()
        for name in names
        if (label, name) not in kernels
    ]
    assert not missing, f"no dump of {missing}"
    return f"{len(kernels)} kernels"


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "QSA sparse prefill guards: %s unsupported; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--keep", help="dump into this new directory and keep it")
    args = parser.parse_args()
    start = time.time()
    torch.set_default_device("cuda")
    results = []

    def guard(name, check, *check_args):
        try:
            results.append((name, "pass", check(*check_args)))
        except AssertionError as error:
            results.append((name, "FAIL", str(error)))

    guard("source uses the stable FlyDSL API", check_sources)
    guard("FP8 sources count no wait by hand", check_fp8_waits)
    guard("lgkmcnt ledger assertions", check_ledger_assertions)
    guard("pin rejects one-word chunks", check_pin_rejects_one_word)
    guard("LLVM options in the primitives only", check_llvm_options)
    guard("kernel attributes at every launch", check_kernel_attrs)

    root = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="qsa_guards_"))
    )
    root.mkdir(parents=True, exist_ok=not args.keep)
    op = load_op_test()
    finite, recorded, seconds = {}, {}, {}
    os.environ["FLYDSL_DUMP_IR"] = "1"
    off_root = root / "uniform_branches_off"
    off_root.mkdir()
    try:
        with plan_kernels_undumped():
            for label in LAYOUTS:
                began = time.time()
                finite[label], recorded[label] = compile_layout(op, label, root)
                seconds[label] = time.time() - began
            with uniform_branches_off():
                off_finite, _ = compile_layout(op, UNIFORM_LABEL, off_root)
    finally:
        os.environ.pop("FLYDSL_DUMP_IR")
        os.environ.pop("FLYDSL_DUMP_DIR", None)
    kernels = read_dumps(root)
    # No dump when the union did not compile again: hints equal with and without the option.
    off = (
        read_dumps(off_root, [UNIFORM_LABEL]).get((UNIFORM_LABEL, UNION_FP8))
        if (off_root / UNIFORM_LABEL).is_dir()
        else None
    )
    guard("compile output", check_compile_output, root, finite, seconds)
    guard("kernels dumped", check_dumped, kernels)
    guard("resources", check_resources, kernels)
    guard("FP8 resources", check_fp8_resources, kernels)
    guard("FP8 direct chunk loop", check_fp8_direct_loops, kernels)
    guard("target features", check_target_features, kernels)
    guard("empty asm", check_empty_asm, kernels)
    guard("lgkmcnt ledger", check_lgkmcnt_ledger, kernels, recorded)
    guard("vmcnt counts", check_vmcnt, kernels)
    guard("persistent direct", check_persistent, kernels)
    guard("exp2 lowering", check_exp2, kernels)
    guard("FP8 MFMA", check_fp8_mfma, kernels)
    guard("FP8 log2 / rcp", check_fp8_log_rcp, kernels)
    guard("FP8 conversion", check_fp8_conversion, kernels)
    guard(
        "uniform branches",
        check_uniform_branches,
        kernels.get((UNIFORM_LABEL, UNION_FP8)),
        off,
        off_root / f"{UNIFORM_LABEL}.log",
        off_finite,
    )
    if not args.keep:
        shutil.rmtree(root)
    aiter.logger.info(
        "qsa_sparse_prefill guards on %s, %.1f s (markdown):\n%s",
        get_gfx(),
        time.time() - start,
        pd.DataFrame(results, columns=["guard", "result", "details"]).to_markdown(
            index=False
        ),
    )
    failed = [name for name, result, _ in results if result != "pass"]
    assert not failed, f"QSA sparse prefill guards failed: {failed}"


if __name__ == "__main__":
    main()
