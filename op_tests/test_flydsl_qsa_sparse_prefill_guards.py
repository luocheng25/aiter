# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Guards of the QSA sparse prefill kernels' low-level code (compile only, untimed).

The kernels keep a few low-level primitives, all in sparse_prefill_primitives.py: empty
inline asm pinning values in VGPRs, the "-packed-fp32-ops" target feature, ROCDL
operations FlyDSL does not export, volatile LDS reads counted by an lgkmcnt ledger
(LdsQueue), and the direct key loop's hand-counted vmcnt waits. When FlyDSL, MLIR or
LLVM stop honouring them, nothing fails: LLVM only prints an asm constraint it cannot
meet and the kernel computes garbage, a dropped attribute or a split load only costs
speed or makes a counted wait wrong. These checks make such changes fail.

One process compiles three layouts of test_flydsl_qsa_sparse_prefill.py with FlyDSL's
IR dumps (FLYDSL_DUMP_IR; dumping bypasses FlyDSL's disk cache): prefill_12000 at
group 12 (union, raw direct and packed direct), kv_slots_ragged_4 at group 12 (slotted
union) and kv_slots_pool_4g_extend_64_ctx_70065 at group 12 (raw direct over a pool past
4 GiB, the WIDE variant). Its final ISA and LLVM IR of the union, raw direct and packed
direct kernels must show:

  compile output   no "error:" line while compiling, and finite outputs
  resources        union: no scratch or VGPR spill, SGPR spills at most
                   UNION_SGPR_SPILLS', two waves per SIMD; direct kernels: no scratch
                   and no spill
  target features  "target-features"="-packed-fp32-ops" on the three kernels and
                   "amdgpu-waves-per-eu"="2" on union in LLVM IR; no v_pk_*_f32
  empty asm        BASELINE's number of ;;#ASMSTART blocks, all empty, and of each
                   LLVM IR asm constraint
  lgkmcnt ledger   the (LDS operations since the last lgkmcnt(0), N) pairs of union's
                   partial lgkmcnt(N) waits, recorded by LdsQueue while tracing
                   (record_lds_waits), equal those a scan of its ISA finds, with no
                   SMEM load counted at any of them
  vmcnt counts     one vmcnt(8) and one vmcnt(16) in each direct kernel: on every path
                   into vmcnt(16) 24 loads since the loop's vmcnt(0) (V segment 1, then
                   the 16 K loads it keeps in flight), fewer only past another drain;
                   at vmcnt(8) 8 or 9 loads since vmcnt(16) (V segment 0 and a block-id
                   prefetch). LLVM adds any wait a register needs (it turned a vmcnt(9)
                   there into vmcnt(8)), so a stale count costs speed, not results: it
                   waits for loads the loop means to keep in flight
  exp2             every exp2 lowered to llvm.amdgcn.exp2.f32 (no llvm.exp2.f32 or
                   __ocml_exp2), BASELINE's number of them

and, compiling no kernel: LdsQueue's own assertions, prim.pin rejecting a one-word chunk
while tracing, and a source audit that the kernel modules (every sparse_prefill_*.py but
sparse_prefill_primitives.py) use FlyDSL's stable API only (FlyDSL
docs/api_stability.md): no flydsl._mlir or bare rocdl import, no inline_asm,
Attribute.parse or llvm.passthrough, every fx./flyc. path in its module's __all__ and
none of the deprecated ones. The deprecated Numeric/Vector methods (shuffle_xor, exp2,
...) are reported on any receiver except the fx modules that export them as functions
(fx.gpu.shuffle_xor, fx.math.exp2).

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
SHORT = {UNION: "union", DIRECT: "direct", PACKED: "packed"}
# label -> (a layout or SLOT_CASES entry of the op test, Q heads per KV head)
LAYOUTS = {
    "prefill_12000": ("prefill_12000", 12),
    "kv_slots_ragged_4": ("kv_slots_ragged_4", 12),
    "kv_slots_pool_4g_extend_64_ctx_70065": (
        "kv_slots_pool_4g_extend_64_ctx_70065",
        12,
    ),
}
# label -> the union and direct kernels its call must compile
EXPECTED = {
    "prefill_12000": (UNION, DIRECT, PACKED),
    "kv_slots_ragged_4": (UNION, DIRECT, PACKED),
    "kv_slots_pool_4g_extend_64_ctx_70065": (DIRECT, PACKED),
}
# kernel -> (;;#ASMSTART blocks, LLVM IR asm constraints, llvm.amdgcn.exp2.f32 calls)
BASELINE = {
    UNION: (479, {"=v,0": 32, "v": 448}, 134),
    DIRECT: (8, {"=v,0": 8}, 9),
    PACKED: (8, {"=v,0": 8}, 9),
}
# label -> most SGPR spills of its union kernel: 6 with a slot table (pool block ids).
UNION_SGPR_SPILLS = {"prefill_12000": 0, "kv_slots_ragged_4": 6}
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
    them) with its kernels dumped to root/label. Returns (outputs finite, the
    (LDS operations, lgkmcnt) pairs LdsQueue recorded while tracing)."""
    case, group = LAYOUTS[label]
    layout, pool = case, None
    if case in op.SLOT_CASES:
        layout, *pool = op.SLOT_CASES[case]
    query_lens, prefix_lens, kv_heads = op.CASES[layout][:3]
    heads = group * kv_heads
    generator = torch.Generator(device="cuda").manual_seed(
        list(op.CASES).index(layout) * 16 + heads
    )
    q, k, v, indices = op.make_inputs(layout, heads, generator)
    kwargs = {"query_lens": query_lens, "prefix_lens": prefix_lens}
    if pool is not None:
        k, v, kwargs["kv_slots"] = op.make_pool(
            k, v, query_lens, prefix_lens, *pool, generator
        )
    os.environ["FLYDSL_DUMP_DIR"] = str(root / label)
    with captured(root / f"{label}.log"), prim.record_lds_waits() as waits:
        out = op.qsa_sparse_prefill(q, k, v, indices, softmax_scale=op.SCALE, **kwargs)
        torch.cuda.synchronize()
    return bool(torch.isfinite(out).all()), list(waits)


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


def read_dumps(root):
    """{(label, kernel): {"isa", "meta", "ir", "attributes"}} of the dumped union and
    direct kernels."""
    kernels = {}
    for label in LAYOUTS:
        for directory in sorted(p for p in (root / label).iterdir() if p.is_dir()):
            functions = ir_functions(latest(directory, "llvm_ir.ll"))
            for name, (body, meta) in isa_kernels(
                latest(directory, "final_isa.s")
            ).items():
                if name not in BASELINE:
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
            assert waves == 2, f"{label} {name}: {waves} waves per SIMD, not 2"
        found.append(f"{label} {SHORT[name]} {meta['vgpr_count']}v/{waves}w")
    return "; ".join(found)


def check_target_features(kernels):
    for (label, name), kernel in kernels.items():
        features = re.search(r'"target-features"="([^"]*)"', kernel["attributes"])
        assert features and "-packed-fp32-ops" in features.group(1).split(","), (
            f"{label} {name}: LLVM IR function attributes lack -packed-fp32-ops: "
            f"{kernel['attributes']}"
        )
        if name == UNION:
            assert (
                '"amdgpu-waves-per-eu"="2"' in kernel["attributes"]
            ), f"{label} {name}: waves_per_eu 2 missing: {kernel['attributes']}"
        packed = [
            op for _, op in instructions(kernel["isa"]) if re.match(r"v_pk_\w*_f32", op)
        ]
        assert not packed, f"{label} {name}: packed FP32 VALU {sorted(set(packed))}"
    return "present; no v_pk_*_f32"


def check_empty_asm(kernels):
    found = []
    for (label, name), kernel in kernels.items():
        expected_blocks, expected_constraints, _ = BASELINE[name]
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


def check_lgkmcnt_ledger(kernels, recorded):
    found = []
    for label, waits in recorded.items():
        ledger = collections.Counter(pair for pair in waits if pair[1] > 0)
        if (label, UNION) not in kernels:
            assert not ledger, f"{label}: ledger waits {ledger} without a union kernel"
            continue
        scan = lgkmcnt_scan(kernels[(label, UNION)]["isa"])
        smem = [wait for wait in scan if wait[1]]
        isa = collections.Counter((lds, count) for lds, _, count in scan)
        assert ledger and ledger == isa and not smem, (
            f"{label} {UNION}: ledger {dict(ledger)} vs ISA {dict(isa)}, "
            f"waits with SMEM in flight {smem}"
        )
        found.append(f"{label} {dict(sorted(ledger.items()))}")
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
        if name == UNION:
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


def check_exp2(kernels):
    found = []
    for (label, name), kernel in kernels.items():
        amdgcn = len(re.findall(r"call float @llvm\.amdgcn\.exp2\.f32\(", kernel["ir"]))
        other = re.findall(r"@llvm\.exp2\.\w+|__ocml_exp2\w*", kernel["ir"])
        expected = BASELINE[name][2]
        assert amdgcn == expected and not other, (
            f"{label} {name}: {amdgcn} llvm.amdgcn.exp2.f32 (expected {expected}), "
            f"other exp2 {sorted(set(other))}"
        )
        found.append(f"{label} {SHORT[name]} {amdgcn}")
    return "; ".join(found)


# ---- Checks without compiling a kernel ----


def check_ledger_assertions():
    """LdsQueue refuses impossible waits before it emits s_waitcnt."""
    cases = {
        "wait(4) of 3 issued": (3, lambda queue: queue.wait(4)),
        "wait(-1)": (3, lambda queue: queue.wait(-1)),
        "ready(0, 3)(4)": (3, lambda queue: queue.ready(0, 3)(4)),
        "lgkmcnt(16)": (16, lambda queue: queue.wait(0)),
    }
    accepted = []
    for label, (issued, call) in cases.items():
        queue = prim.LdsQueue()
        queue.issued = issued
        try:
            call(queue)
            accepted.append(label)
        except ValueError:
            pass
        except Exception as error:  # noqa: BLE001
            # Past its checks: emitting s_waitcnt outside a kernel trace fails.
            accepted.append(f"{label} ({type(error).__name__})")
    assert not accepted, f"LdsQueue accepted {accepted}"
    return f"{len(cases)} rejected"


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
    guard("LdsQueue assertions", check_ledger_assertions)
    guard("pin rejects one-word chunks", check_pin_rejects_one_word)

    root = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="qsa_guards_"))
    )
    root.mkdir(parents=True, exist_ok=not args.keep)
    op = load_op_test()
    finite, recorded, seconds = {}, {}, {}
    os.environ["FLYDSL_DUMP_IR"] = "1"
    try:
        for label in LAYOUTS:
            began = time.time()
            finite[label], recorded[label] = compile_layout(op, label, root)
            seconds[label] = time.time() - began
    finally:
        os.environ.pop("FLYDSL_DUMP_IR")
        os.environ.pop("FLYDSL_DUMP_DIR", None)
    kernels = read_dumps(root)
    guard("compile output", check_compile_output, root, finite, seconds)
    guard("kernels dumped", check_dumped, kernels)
    guard("resources", check_resources, kernels)
    guard("target features", check_target_features, kernels)
    guard("empty asm", check_empty_asm, kernels)
    guard("lgkmcnt ledger", check_lgkmcnt_ledger, kernels, recorded)
    guard("vmcnt counts", check_vmcnt, kernels)
    guard("exp2 lowering", check_exp2, kernels)
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
