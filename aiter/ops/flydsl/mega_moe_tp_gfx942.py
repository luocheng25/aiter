# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer for gfx942 with the interface of ``MegaMoeTP`` (``comm_mode="ar"`` only).

Experts are replicated over the TP group and ``inter_dim`` is sharded. Every rank gets the same input rows and
routing; ``forward`` runs this rank's slice with the A8W4 two-stage kernels (``moe_a8w4.fused_moe_a8w4``: FP8
per-token activations, MXFP4 weights) and sums the partial outputs over the ranks:

* TP2 / TP4 batches of at most ``DECODE_TP_MAX_M`` rows (the decode path): the sum is fused into the Down decode
  kernel (``kernels/mega_moe_tp_gfx942/decode_tp.py``, ``a8w4_down_decode[_pq]_tp``: each workgroup sends its f32
  partials of 16 channels to the other ranks' inboxes over hipIpc symmetric memory and adds the ranks' partials in
  rank order; the slot-pair ``_pq`` kernel from ``DECODE_TP_PREQUANT_MIN_M`` rows on);
* TP2 / TP4 batches of more than ``FUSED_MIN_M - 1`` rows (the streaming and prefill paths) whose BF16 output
  has at least ``FUSED_MIN_BYTES[world_size]`` bytes: the sum is fused into the last kernel
  (``kernels/mega_moe_tp_gfx942/tp_reduce.py``, ``a8w4_tp_reduce``: route sum, reduce-scatter of token tiles to
  their owner rank over hipIpc symmetric memory, f32 sum in rank order, all-gather of the BF16 rows); on the
  prefill path of the shapes in ``STREAM_SPEC`` (``STREAM_DOWN``) the same sum streams out of the Down kernel
  instead (``kernels/mega_moe_tp_gfx942/down_tp.py``, ``a8w4_down_prefill_bm64_tp``: workgroups of the Down launch
  reduce, scatter and gather each column block of the output while the next ones compute), and ``a8w4_tp_reduce``
  only copies the rows the other ranks summed into the output; TP2 batches of the shapes in ``CHUNK_SPEC``
  (``AITER_MEGA_MOE_TP_CHUNK``) run in token chunks instead, each one's Down storing this rank's partials that an
  SDMA queue pushes to the peer while the next chunk computes, and ``a8w4_tp_reduce`` adds the two partials;
* otherwise: the BF16 partials are all-reduced: aiter ``CustomAllreduce`` when its ``should_custom_ar`` accepts
  the tensor, RCCL (``torch.distributed.all_reduce`` on ``group``) otherwise.

The buffers of the fused sums (``_Workspace``) are allocated once per (communicator, ``model_dim``,
``max_local_tokens``) and shared by those instances. The outputs of all ranks are bitwise identical.

Weights come in the ``MegaMoeTP`` layout (MXFP4, ``shuffle_weight(16, 16)`` + ``e8m0_shuffle``) and are converted
once (constructor / ``set_weights``) to the A8W4 layout; ``inter_dim`` is zero-padded to a multiple of 64.

Contract (as ``MegaMoeTP``):

* ``forward`` is a collective: every rank calls it the same number of times, in the same order, with the same
  input and routing.
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles and runs every kernel these batches launch,
  then checks every rank's watchdog (``reset()`` and retry once on a timeout, else raise).
* A wait of a fused kernel on a peer gives up after ``WAIT_S`` seconds (``FIRST_WAIT_S`` on an instance's first
  forward of a batch size, which may still be compiling on a peer), sets a sticky bit and goes on: that launch's
  output is invalid, never a hang. ``error_flag()`` is that bit as a device tensor (no sync);
  ``check_errors()`` synchronizes and raises; ``reset()`` (collective) restarts the cross-rank state (device-side
  epochs, flags, inboxes and parities shared by the instances of the workspace), which a timed-out launch leaves
  inconsistent.
* ``activation`` must be ``Silu``; a finite ``swiglu_limit`` L > 0 computes
  ``SiLU(gate.clamp(max=L)) * up.clamp(-L, L)`` (DeepSeek-V4.1). The all-reduce is BF16 whatever
  ``comm_dtype`` says; the gfx950 fields (``schedule``, ``act_dtype``, ``lb_*``, ``bigm_min_tokens``, Situ betas,
  ``ar_gather="fp8"``) and the fused tail are rejected.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import warnings

import torch
import torch.distributed as dist

from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx_runtime

from . import moe_a8w4
from .kernels.mega_moe_tp_gfx942 import decode_tp, down_tp, tp_reduce
from .kernels.moe_a8w4 import chunks as a8w4_chunks
from .mega_moe_tp import MegaMoeTPConfig

__all__ = ["MegaMoeTPConfig", "MegaMoeTPGfx942", "mega_moe_tp_gfx942_supported"]


def mega_moe_tp_gfx942_supported(gfx: str | None = None) -> bool:
    """Whether this device runs ``MegaMoeTPGfx942`` (gfx942)."""
    return (get_gfx_runtime() if gfx is None else gfx) == "gfx942"


# ---------------------------------------------------------------------------------------------------------------
# MegaMoeTP weight layout -> standard MXFP4
# ---------------------------------------------------------------------------------------------------------------
def _unshuffle_weight(w, rows, cols_bytes):
    """Inverse of ``shuffle_weight(layout=(16, 16))`` for packed FP4: [E, rows, cols_bytes] uint8."""
    E = w.shape[0]
    t = w.view(torch.uint8).reshape(E, rows // 16, cols_bytes // 16, 16, 16)
    return t.permute(0, 1, 3, 2, 4).reshape(E, rows, cols_bytes)


def _unshuffle_scale(s, experts, rows, groups):
    """Inverse of ``e8m0_shuffle`` (``shuffle_scale``) of an [experts * rows, groups] E8M0 tensor (padded to
    [ceil256, ceil8]) -> [experts, rows, groups] uint8."""
    m = experts * rows
    sm, sn = -(-m // 256) * 256, -(-groups // 8) * 8
    if s.dim() != 2 or tuple(s.shape) != (sm, sn):
        raise ValueError(
            f"expected an e8m0_shuffle'd scale of shape ({sm}, {sn}), got {tuple(s.shape)}"
        )
    t = (
        s.view(torch.uint8)
        .reshape(sm // 32, sn // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(sm, sn)
    )
    return t[:m, :groups].reshape(experts, rows, groups)


def _check_weight(name, w, shape):
    if w.dtype not in (torch.float4_e2m1fn_x2, torch.uint8):
        raise TypeError(
            f"MegaMoeTPGfx942: {name} must be float4_e2m1fn_x2 or uint8, got {w.dtype}"
        )
    if tuple(w.shape) != shape:
        raise ValueError(
            f"MegaMoeTPGfx942: {name} must have shape {shape}, got {tuple(w.shape)}"
        )


# ---------------------------------------------------------------------------------------------------------------
# communicators: one gloo group + CustomAllreduce per (TP group, device), shared by the instances
# ---------------------------------------------------------------------------------------------------------------
def _align_ipc_pool_seq(cpu_group):
    """Make the per-process ``IPCBufferPool`` sequence number equal on every rank of ``cpu_group`` (collective).

    The custom all-reduce keys its store exchange on that counter, which only advances in the processes that
    create a pool. A rank that already joined another group's pool (TP2 and TP4 layers in one process) would
    otherwise wait on other keys than its peers and hang.
    """
    from aiter.dist.device_communicators.custom_all_reduce import IPCBufferPool

    seq = torch.tensor([IPCBufferPool._pool_seq], dtype=torch.int64)
    dist.all_reduce(seq, op=dist.ReduceOp.MAX, group=cpu_group)
    IPCBufferPool._pool_seq = int(seq.item())


class _Comm:
    def __init__(self, group, cpu_group, device):
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        if cpu_group is None:
            ranks = (
                dist.get_process_group_ranks(group)
                if group is not None
                else list(range(dist.get_world_size()))
            )
            # Collective over the default group, as torch.distributed.new_group (every rank of the job enters).
            cpu_group = self._own_cpu_group = dist.new_group(
                ranks=ranks, backend="gloo"
            )
        else:
            self._own_cpu_group = None
        self.group, self.cpu_group, self.device, self.refs = group, cpu_group, device, 0
        _align_ipc_pool_seq(cpu_group)
        ca = CustomAllreduce(group=cpu_group, device=device)
        self.ca = None if ca.disabled else ca
        self.last = None

    def all_reduce(self, y):
        """Sum of ``y`` over the group: a new tensor (custom all-reduce) or ``y`` itself (RCCL, in place).
        ``last`` names the one used."""
        if self.ca is not None and self.ca.should_custom_ar(y):
            self.last = "custom_all_reduce"
            return self.ca.custom_all_reduce(y)
        self.last = "rccl_all_reduce"
        dist.all_reduce(y, group=self.group)
        return y

    def close(self):
        """Collective: no peer may still read this rank's buffers when they are freed."""
        if self.ca is not None:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.cpu_group)
            self.ca.close()
            self.ca = None
        if self._own_cpu_group is not None:
            dist.destroy_process_group(self._own_cpu_group)
            self._own_cpu_group = None


_COMMS: dict = {}


def _acquire_comm(group, cpu_group, device):
    key = (id(group), id(cpu_group), device.index)
    comm = _COMMS.get(key)
    if comm is None:
        comm = _COMMS[key] = _Comm(group, cpu_group, device)
    comm.refs += 1
    return key, comm


def _release_comm(key):
    comm = _COMMS.get(key)
    if comm is None:
        return
    comm.refs -= 1
    if comm.refs <= 0:
        comm.close()
        del _COMMS[key]


# ---------------------------------------------------------------------------------------------------------------
# fused cross-rank sum: symmetric buffers shared by the instances of one communicator, hidden size and batch bound
# ---------------------------------------------------------------------------------------------------------------
# TP sizes whose batches of at least FUSED_MIN_M rows and FUSED_MIN_BYTES[world] bytes of BF16 output fuse the
# cross-rank sum into the last kernel (tp_reduce). Smaller outputs keep the all-reduce: the reduce-scatter and the
# all-gather are two dependent cross-GPU trips, the custom all-reduce one (paired timing on MI308X: TP2 fused from
# 2.6 MB, all-reduce up to 1.3 MB; TP4 fused from 1.3 MB, all-reduce up to 0.66 MB).
FUSED_WORLDS = (2, 4)
FUSED_MIN_M = moe_a8w4.DECODE_MAX_M + 1
FUSED_MIN_BYTES = {2: 2 << 20, 4: 1 << 20}
# Decode batches of at most DECODE_TP_MAX_M rows (of the same TP sizes) sum across the ranks in the Down decode
# kernel (decode_tp); the batches up to moe_a8w4.DECODE_MAX_M above keep the all-reduce (S2 prototype: fused
# 2-4 % slower at 32 rows).
DECODE_TP_MAX_M = 16
# From DECODE_TP_PREQUANT_MIN_M rows on (below moe_a8w4.DECODE_PREQUANT_MIN_M) the fused Down is the slot-pair kernel
# on pre-quantized rows (paired timing on MI308X, slot-pair / single-slot fused at 2 / 3 rows: TP2 0.94 / 0.88,
# TP4 0.96 / 0.87, DeepSeek-V4.1 TP4 0.97 / 0.98; 1 row: 1.06-1.14).
DECODE_TP_PREQUANT_MIN_M = 2
# Fused batches on the prefill path stream the sum out of the Down kernel (down_tp); 0 in the environment variable
# keeps the Down kernel and tp_reduce of the other fused batches.
STREAM_DOWN = os.environ.get("AITER_MEGA_MOE_TP_STREAM_DOWN", "1") != "0"
# The same for the fused batches on the stream path (the streaming Down, four waves of STREAM_TP_R row blocks).
STREAM_TIER = True
STREAM_TP_R = 8


# The streamed Down of a fused prefill batch by (world size, hidden size): (column blocks, reduce workgroups, owner
# workgroups per block, tokens per tile[, DownTP.DRY[, DownTP.ROUTE_AUX]][, {other DownTP fields}]), or pairs
# (largest batch, that tuple) in increasing order, the first one the batch fits (None: any). Shapes without an
# entry keep the Down kernel and tp_reduce. Paired timing on MI308X against those (20261126-084336-p1-run-ce23):
# TP2 0.87-0.90 (M 2048-32768), TP4 0.96-0.99 (M 4096-32768), DeepSeek-V4.1 TP4 0.93-0.95 (M 2048-32768; 10
# column blocks 0.97 at M 2048). DeepSeek-V4.1 TP4 with the compute schedule of DOWN_TUNE: 5 column blocks with
# 32 / 16 reduce / owner workgroups against the former 10 column blocks 0.96-0.97 (M 4096-32768) and against
# 64 / 32 0.98 (M 2048); there 64 / 32 or 96 / 48 0.99-1.04 and 16 / 8, 24 / 12, 32 / 8 or 16 / 16 0.96-1.00
# against 10 column blocks, 2 column blocks 1.16-1.20 (20261126-222126-dtax-run-94b4,
# 20261126-222417-dtax-run-843e, 20261126-222419-dtax-run-2025). TP2 / TP4 with fewer reduce / owner
# workgroups 1.00-1.02.
STREAM_SPEC = {
    (2, 2560): (5, 32, 16, 8, {"UX": 1}),
    (4, 2560): (5, 64, 32, 8, {"UX": 1}),
    (4, 5120): (5, 32, 16, 8),
}
# Schedule of the compute workgroups of the streamed prefill Down (STREAM_SPEC; down_tp.DownTP PRO / SB / LAS with
# WPE, and ROUTE_AUX: the same bits) by (world, hidden size), or pairs (largest batch, fields) as there. Paired
# timing on MI308X against none (20261126-214137-dtax-run-303d, 20261126-214139-dtax-run-84f7): TP4 0.97-0.98 (M
# 8192-32768), TP2 0.98-0.99 (M 2048-32768), DeepSeek-V4.1 TP4 0.98-0.99 (M 2048-32768; at 5 column blocks again,
# 20261126-222707-dtax-run-89ab). down_tp.SYS_NT only
# from TP4 M 16384 on: 0.98-1.01 at TP4 M 4096-8192, TP2 and DeepSeek-V4.1 TP4 (the reduce workgroups' reads of
# the rows then miss more).
_DOWN_TUNE_TP4 = {"PRO": 1, "SB": 1, "WPE": 4}
DOWN_TUNE = {
    (2, 2560): {"PRO": 1, "LAS": 1, "WPE": 3},
    (4, 2560): ((8192, _DOWN_TUNE_TP4), (None, {**_DOWN_TUNE_TP4, "ROUTE_AUX": down_tp.SYS_NT})),
    (4, 5120): {"SB": 1, "LAS": 1},
}
# The same for the fused batches on the stream path (the streaming Down's own column blocks, DownTP.COPY set by the
# caller). Paired timing on MI308X against the Down kernel and tp_reduce (20261126-100503-p1-run-aaf7): TP2 0.94 /
# 0.89 (M 512 / 1024), TP4 0.97 / 0.93 / 0.92 (M 512 / 1024 / 2048) but 1.05 at M 256; DeepSeek-V4.1 TP4 1.01-1.04
# (M 128-512, 20261126-092445-p1-run-d2d3) and 0.98 (M 1024, 20261126-102139-p1-run-d3d7).
_STREAM_TIER_TP = (5, 16, 8, 8, {"UX": 1, "NXT": 128, "NYT": 64})
STREAM_TIER_SPEC = {
    (2, 2560): _STREAM_TIER_TP,
    (4, 2560): ((384, None), (None, _STREAM_TIER_TP)),
    (4, 5120): ((512, None), (None, (5, 16, 8, 8, {"NXT": 128, "NYT": 64}))),
}
# Items (tiles of a column block) per workgroup of the gather kernel (down_tp.compile_tp_gather, G) of a call
# with at least GATHER_FROM items, else 1. Paired timing on MI308X (20261126-112217-p1-run-090b), 4 against 1
# item: 0.5-1.0 % faster at TP2 / TP4 M 8192 and 32768 and DeepSeek-V4.1 TP4 M 4096 and 16384, unchanged at
# 640-1280 items.
GATHER_ITEMS = 4
GATHER_FROM = 2048
# Token chunks of the streamed prefill Down with the one-shot cross-rank sum (down_tp rules 11-12) per (world,
# hidden size): (from M rows, split points as fractions of M[, {DownTP fields of the one-shot Down}]) -- the batches
# of at least that many rows run in
# token chunks (the split points rounded to whole tiles of every rank; chunk c's sort, Gate/Up, quantization and
# Down after chunk c - 1's Down), and every rank's SDMA queue (on KFD's recommended engine, sdma.push_order) pushes
# each chunk's partials into the peer's landing block by block while the Downs go on; the gather sums both ranks'.
# Only with SDMA queues on every rank (else the batch takes the path without chunks). Paired timing on MI308X
# against the path without chunks (20261126-180318-tcs-run-a818, 20261126-180107-tcs-run-0a5b,
# 20261126-184959-tcs-run-c2a7): TP2 0.95 / 0.92 / 0.90 (M 8192 / 16384 / 32768) but 1.01-1.04 at M 4096 and 1.17
# at M 2048; one chunk 0.99-1.02. TP4 with an SDMA all-gather of the owners' sums instead: 0.99 at M 32768,
# 1.01-1.06 at M 8192 - 16384; DeepSeek-V4.1 TP4 1.01-1.04 (M 16384 / 32768): none. The one-shot Down has no owner
# workgroups (NY = 0), so it takes more reduce workgroups: NX 96 instead of 32, paired 0.98-0.99 at TP2 M 8192 -
# 32768 (20261126-220800-dtax-run-7e84; NX 64 / 128 alike, more for the last block nothing).
# AITER_MEGA_MOE_TP_CHUNK=0: no token chunks.
CHUNK_SPEC = {
    (2, 2560): (8192, (0.78,), {"NX": 96}),
}
CHUNK_DOWN = os.environ.get("AITER_MEGA_MOE_TP_CHUNK", "1") != "0"
# Timing analysis of the one-shot path (DownTP.STAMP; needs TRACE_WGS >= down_tp.STAMP_BYTES / 16).
OS_STAMP = 0
# Timing analysis (DownTP.TRACE; DownTP.STAMP needs down_tp.STAMP_BYTES): room for this many workgroups'
# timestamps after the counters.
TRACE_WGS = 0


def _stream_spec(world, H, topk, M, table=None, **over):
    """The ``down_tp.DownTP`` of an M-row batch of a shape (``table``, default ``STREAM_SPEC`` with ``DOWN_TUNE``;
    ``over``: more fields), None if it does not stream."""

    def pick(spec):
        if isinstance(spec, tuple) and spec and isinstance(spec[0], tuple):
            return next((t for up_to, t in spec if up_to is None or M <= up_to), None)
        return spec

    spec = pick((STREAM_SPEC if table is None else table).get((world, H)))
    if spec is None:
        return None
    nsplit, nx, ny, tt, *extra = spec
    fields = extra.pop() if extra and isinstance(extra[-1], dict) else {}
    tune = (pick(DOWN_TUNE.get((world, H))) or {}) if table is None else {}
    return down_tp.DownTP(
        WORLD=world,
        H=H,
        TOPK=topk,
        TT=tt,
        NSPLIT=nsplit,
        NX=nx,
        NY=ny,
        **{**tune, **dict(zip(("DRY", "ROUTE_AUX"), extra)), **fields, **over},
    )


# Wait bounds of the fused kernels in seconds: every forward, and an instance's first forward of a batch size.
WAIT_S = 10.0
FIRST_WAIT_S = 300.0
# s_memrealtime frequency on gfx942.
_TIMER_HZ = 100e6


def _tile_tokens(m, H):
    """Tokens per tile of tp_reduce for an m-row batch of hidden size H: one below 1024 rows, else about 10 KB
    of BF16 per tile (one tile per workgroup: larger tiles hold more VGPRs, fewer workgroups in flight; paired
    timing on MI308X at 4096-32768 rows, against 4-token tiles: TP4 1.1-1.9 %, DeepSeek-V4.1 TP4 1.5-1.9 % faster,
    TP2 within 0.3 %)."""
    if m < 1024:
        return 1
    return 4 if H <= 1280 else 2 if H <= 2560 else 1


def _chunk_starts(M, unit, fracs):
    """First tokens of the token chunks of an M-row batch (``CHUNK_SPEC``): 0, then each split point rounded to
    a multiple of ``unit`` (empty chunks dropped)."""
    starts = [0]
    for f in fracs:
        first = round(M * f / unit) * unit
        if starts[-1] < first < M:
            starts.append(first)
    return tuple(starts)


def _wait_limit(seconds):
    """The ``limit`` launch argument of tp_reduce for a wait bound in seconds."""
    return max(1, math.ceil(seconds * _TIMER_HZ / tp_reduce.LIMIT_TICKS))


class _Workspace:
    """Buffers of the fused sums (their module docstrings) of one communicator, hidden size and batch bound
    ``m_max``. tp_reduce: symmetric ``landing`` / ``inbox`` / ``flags`` (hipIpc, uncached, zeroed), their address
    table ``tab`` and this rank's ``epoch``; down_tp: ``landing`` / ``inbox`` and its own symmetric ``flags2``, their
    table ``tab2``, this rank's ``epoch2`` and block counters ``ctr`` (room for the token chunks in ``flags2`` and
    ``ctr``); decode_tp: the symmetric ``dinbox`` (uncached, 0xFF), its address table ``dtab`` and this rank's
    ``parity``. ``sdma_table`` (the token chunks' SDMA pushes, down_tp rule 12) acquires this device's SDMA queues
    on first use. Creation, ``close`` and the first ``sdma_table`` are collective over the communicator's gloo group.
    """

    def __init__(self, comm, rank, world, H, m_max):
        from .communication_ops_host_utils import IpcSymmetricHeap

        # The largest batch of each tile size bounds the tiles and the inbox rows of that size.
        tiers = [(m, _tile_tokens(m, H)) for m in {min(m_max, 1023), m_max}]
        self.slots = max(-(-m // tt) for m, tt in tiers)
        # Rows per owner rank: an owner's share of the tiles (tile i -> rank i % world), in tokens.
        rps = max((-(-m // tt) + world - 1) // world * tt for m, tt in tiers)
        # down_tp: tiles of its smallest tile size bound the slots; an owner keeps its own partials in its inbox.
        self.tpq = -(-m_max // min(down_tp.TILE_TOKENS))
        self.slots2 = down_tp.NSPLIT_MAX * self.tpq
        rps2 = max(
            (-(-m_max // tt) + world - 1) // world * tt for tt in down_tp.TILE_TOKENS
        )
        self.comm, self.refs = comm, 0
        self.heap = IpcSymmetricHeap(
            rank=rank,
            world_size=world,
            device=comm.device,
            group=comm.cpu_group,
            memory="uncached",
        )
        error = None
        try:
            self.landing = self.heap.create_tensor((m_max, H), torch.bfloat16)
            self.inbox = self.heap.create_tensor(
                (max((world - 1) * rps, world * rps2), H), torch.bfloat16
            )
            self.flags = self.heap.create_tensor(
                (self.slots * (world + 1),), torch.int32
            )
            self.flags2 = self.heap.create_tensor(
                (down_tp.flag_words(self.slots2, world, True),), torch.int32
            )
            self.dinbox = self.heap.create_tensor(
                (decode_tp.inbox_words(H, world, DECODE_TP_MAX_M),), torch.int32
            )
            # Filled before the handles are published: from then on a peer may write into it.
            self.dinbox.fill_(decode_tp.SENTINEL)
            torch.cuda.synchronize(comm.device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        # Raises on every rank if one failed (and closes the heap collectively).
        self.heap.publish(error)
        self.tab = torch.stack(
            [self.heap.p2p_table(t) for t in (self.landing, self.inbox, self.flags)]
        ).view(torch.int32)
        self.epoch = torch.zeros(self.slots, dtype=torch.int32, device=comm.device)
        self._tab2 = torch.stack(
            [self.heap.p2p_table(t) for t in (self.landing, self.inbox, self.flags2)]
        )
        self.tab2 = self._tab2.view(torch.int32)
        self.epoch2 = torch.zeros(self.slots2, dtype=torch.int32, device=comm.device)
        self.ctr = torch.zeros(
            down_tp.ctr_words(True) + 4 * TRACE_WGS,
            dtype=torch.int32,
            device=comm.device,
        )
        self.rank, self.world, self.sdma, self.tab3, self._sdma_tried = rank, world, None, None, False
        self.sdma_order = self._sdma_desc = None
        self.dtab = self.heap.p2p_table(self.dinbox).view(torch.int32)
        self.parity = torch.zeros(
            DECODE_TP_MAX_M * (H // 16), dtype=torch.int32, device=comm.device
        )

    def sdma_table(self):
        """``tab2`` with a fourth row (``down_tp`` rule 12: the address of this device's
        ``sdma.SdmaQueues.desc`` in the order ``sdma_order``, queue k the one of peer ``(rank + 1 + k) % world``) if
        every rank has its SDMA queues, else None (with a warning). The first call acquires them (collective, not
        while capturing a CUDA graph: None then)."""
        if not self._sdma_tried and not torch.cuda.is_current_stream_capturing():
            self._sdma_tried = True
            got = _acquire_sdma(self.comm, self.rank, self.world)
            if got is not None:
                self.sdma, self.sdma_order = got
                self._sdma_desc = self.sdma.ordered(self.sdma_order)
                row = torch.zeros(self.world, dtype=torch.int64, device=self.comm.device)
                row[0] = self._sdma_desc.data_ptr()
                self.tab3 = torch.cat([self._tab2, row[None]]).view(torch.int32)
        return self.tab3

    def reset(self):
        """Collective: zero the flags, epochs and parities, refill the decode inbox and recreate the SDMA queues
        once no kernel of any rank runs, before any rank launches again."""
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)
        if self.sdma is not None:
            self.sdma.recreate()
        self.flags.zero_()
        self.epoch.zero_()
        self.flags2.zero_()
        self.epoch2.zero_()
        self.ctr.zero_()
        self.dinbox.fill_(decode_tp.SENTINEL)
        self.parity.zero_()
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)

    def close(self):
        """Collective (``IpcSymmetricHeap.close``)."""
        if self.sdma is not None:
            from .kernels.mega_moe_tp_gfx942 import sdma

            torch.cuda.synchronize(self.comm.device)
            sdma.release(self.sdma)
            self.sdma = None
        self.heap.close()
        self.landing = self.inbox = self.flags = self.tab = self.epoch = None
        self.flags2 = self._tab2 = self.tab2 = self.tab3 = self.epoch2 = self.ctr = None
        self.dinbox = self.dtab = self.parity = self._sdma_desc = None


def _acquire_sdma(comm, rank, world):
    """Collective: (this device's SDMA queues, the queue order of its pushes: ``sdma.push_order`` for the peers
    ``(rank + 1 + k) % world``, the identity if ``sdma.ENGINES_ENV`` is set) if every rank got its own queues, else
    None (with a warning)."""
    from .kernels.mega_moe_tp_gfx942 import sdma

    queues, error = None, None
    try:
        queues = sdma.acquire(comm.device)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * world
    dist.all_gather_object(errors, (error, queues.gpu_id if queues is not None else 0), group=comm.cpu_group)
    failed = [e for e, _ in errors if e is not None]
    if not failed:
        order = tuple(range(len(queues.engines)))
        if not os.environ.get(sdma.ENGINES_ENV):
            peers = [errors[(rank + 1 + k) % world][1] for k in range(world - 1)]
            order = sdma.push_order(queues.engines, [sdma.recommended(queues.gpu_id, p) for p in peers])
        return queues, order
    if queues is not None:
        sdma.release(queues)
    warnings.warn(
        f"MegaMoeTPGfx942: no SDMA queues ({failed[0]}): the batches take the path without token chunks"
    )
    return None


_WORKSPACES: dict = {}


def _acquire_workspace(comm_key, comm, rank, world, H, m_max):
    key = (comm_key, H, m_max)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _WORKSPACES[key] = _Workspace(comm, rank, world, H, m_max)
    ws.refs += 1
    return key, ws


def _release_workspace(key):
    ws = _WORKSPACES.get(key)
    if ws is None:
        return
    ws.refs -= 1
    if ws.refs <= 0:
        del _WORKSPACES[key]
        ws.close()


# ---------------------------------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------------------------------
def _check_config(cfg: MegaMoeTPConfig) -> float:
    """Reject what this implementation does not run; returns the SwiGLU limit (0.0: off)."""
    if cfg.comm_mode != "ar":
        raise NotImplementedError(
            f"MegaMoeTPGfx942: comm_mode {cfg.comm_mode!r} (only 'ar')"
        )
    if cfg.activation != ActivationType.Silu:
        raise ValueError(
            f"MegaMoeTPGfx942: unsupported activation {cfg.activation} (only Silu)"
        )
    gfx950 = {
        "schedule": (cfg.schedule, "dynamic"),
        "act_dtype": (cfg.act_dtype, "fp4"),
        "lb_min": (cfg.lb_min, None),
        "lb_mt": (cfg.lb_mt, None),
        "lb_mt_small": (cfg.lb_mt_small, None),
        "lb_small_max": (cfg.lb_small_max, None),
        "lb_npp": (cfg.lb_npp, None),
        "lb_q": (cfg.lb_q, None),
        "bigm_min_tokens": (cfg.bigm_min_tokens, None),
        "beta": (cfg.beta, None),
        "linear_beta": (cfg.linear_beta, None),
    }
    for field, (value, default) in gfx950.items():
        if value != default:
            raise ValueError(
                f"MegaMoeTPGfx942: {field}={value!r} is a gfx950 MegaMoeTP option (leave {default!r})"
            )
    if cfg.comm_dtype not in ("fp8", "bf16"):
        raise ValueError(f"MegaMoeTPGfx942: unknown comm_dtype {cfg.comm_dtype!r}")
    if cfg.ar_gather not in ("auto", "bf16"):
        raise ValueError(
            f"MegaMoeTPGfx942: ar_gather={cfg.ar_gather!r} (the all-reduce is BF16: 'auto' or 'bf16')"
        )
    if (
        cfg.model_dim % 512
        or cfg.inter_dim % 32
        or cfg.experts < 1
        or not 0 < cfg.topk <= cfg.experts
    ):
        raise ValueError(
            f"MegaMoeTPGfx942: unsupported shape model_dim={cfg.model_dim} (needs % 512), "
            f"inter_dim={cfg.inter_dim} (needs % 32), experts={cfg.experts}, topk={cfg.topk}"
        )
    if cfg.max_local_tokens < 1:
        raise ValueError(
            f"MegaMoeTPGfx942: max_local_tokens {cfg.max_local_tokens} < 1"
        )
    limit = cfg.swiglu_limit
    if limit is None or limit == math.inf:
        return 0.0
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(
            f"MegaMoeTPGfx942: swiglu_limit must be > 0 and finite, or None / inf, got {limit!r}"
        )
    return float(limit)


class MegaMoeTPGfx942:
    """One TP MoE layer on gfx942 (see the module docstring)."""

    def __init__(
        self,
        cfg: MegaMoeTPConfig,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        group=None,
        device: torch.device | None = None,
        cpu_group=None,
    ):
        """``group``: the TP process group (RCCL; None: the default group). ``cpu_group``: a gloo group of the same
        ranks for the custom all-reduce; None: created with ``torch.distributed.new_group``, so every rank of the
        job must construct its layer at the same point (pass it when only the TP group's ranks construct). The
        instances of one (group, cpu_group, device) share one communicator, and those of one communicator,
        ``model_dim`` and ``max_local_tokens`` the fused sum's buffers (collective when first allocated).
        """
        if not mega_moe_tp_gfx942_supported():
            raise ValueError("MegaMoeTPGfx942: needs gfx942")
        self._limit = _check_config(cfg)
        rank, world = dist.get_rank(group=group), dist.get_world_size(group=group)
        if (cfg.rank, cfg.world_size) != (rank, world):
            raise ValueError(
                f"MegaMoeTPGfx942: cfg rank / world_size {cfg.rank} / {cfg.world_size} do not match the "
                f"group ({rank} / {world})"
            )
        device = torch.device(device if device is not None else "cuda")
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.cfg, self.device = cfg, device
        self._group = group
        self._weights = None
        # Kernels of the most recent forward in launch order (fused: ending with "a8w4_tp_reduce" or
        # "a8w4_down_decode[_pq]_tp"), else then its all-reduce ("custom_all_reduce" / "rccl_all_reduce"); tests
        # check the dispatch with it.
        self.last_path: tuple[str, ...] = ()
        self.set_weights(w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale)
        self._comm_key, self._comm = _acquire_comm(group, cpu_group, device)
        self._ws_key, self._ws = None, None
        if world in FUSED_WORLDS:
            try:
                self._ws_key, self._ws = _acquire_workspace(
                    self._comm_key,
                    self._comm,
                    rank,
                    world,
                    cfg.model_dim,
                    cfg.max_local_tokens * world,
                )
            except Exception:
                _release_comm(self._comm_key)
                self._comm = None
                raise
        # Sticky watchdog bits of this instance's fused launches (tp_reduce.ERR_TIMEOUT, decode_tp.ERR_TIMEOUT).
        self._err = torch.zeros(1, dtype=torch.int32, device=device)
        # Batch sizes this instance ran fused (their next forwards wait WAIT_S, not FIRST_WAIT_S; token chunks:
        # (M, starts)).
        self._seen: set = set()
        self._chunk_descs: dict = {}

    # -- weights ------------------------------------------------------------------------------------------------
    def set_weights(
        self,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Run the next forwards on another layer's weights of the same shape (converted here, once)."""
        E, H, I = self.cfg.experts, self.cfg.model_dim, self.cfg.inter_dim
        _check_weight("w1", w1, (E, 2 * I, H // 2))
        _check_weight("w2", w2, (E, H, I // 2))
        for name, t in (
            ("w1", w1),
            ("w1_scale", w1_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ):
            if t.device != self.device:
                raise ValueError(
                    f"MegaMoeTPGfx942: {name} is on {t.device}, expected {self.device}"
                )
        with torch.no_grad():
            q1 = _unshuffle_weight(w1, 2 * I, H // 2)
            c1 = _unshuffle_scale(w1_scale, E, 2 * I, H // 32)
            q2 = _unshuffle_weight(w2, H, I // 2)
            c2 = _unshuffle_scale(w2_scale, E, H, I // 32)
            self._weights = None
            self._weights = moe_a8w4.prepare_a8w4_weights(q1, c1, q2, c2)

    # -- forward ------------------------------------------------------------------------------------------------
    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        tail=None,
        bf16: bool = False,
    ) -> torch.Tensor:
        """x [M, H] BF16, replicated on every rank, with its routing (topk_weights [M, topk], topk_ids
        [M, topk] in [0, experts)) -> y [M, H] BF16, the whole layer's output, identical on every rank
        (``out`` if given, else a new tensor)."""
        if tail is not None:
            raise ValueError("MegaMoeTPGfx942: no fused tail (gfx950 ag_rs only)")
        cfg = self.cfg
        if (
            x_local.dim() != 2
            or x_local.shape[1] != cfg.model_dim
            or x_local.dtype != torch.bfloat16
        ):
            raise ValueError(
                f"MegaMoeTPGfx942: x must be bfloat16 [M, {cfg.model_dim}], got {x_local.dtype} "
                f"{tuple(x_local.shape)}"
            )
        M = x_local.shape[0]
        if tuple(topk_ids.shape) != (M, cfg.topk) or tuple(topk_weights.shape) != (
            M,
            cfg.topk,
        ):
            raise ValueError(
                f"MegaMoeTPGfx942: topk_ids / topk_weights must be [{M}, {cfg.topk}], got "
                f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
            )
        if -(-M // cfg.world_size) > cfg.max_local_tokens:
            raise ValueError(
                f"MegaMoeTPGfx942: {M} rows exceed max_local_tokens {cfg.max_local_tokens} x "
                f"world_size {cfg.world_size}"
            )
        if out is not None and (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (M, cfg.model_dim)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"MegaMoeTPGfx942: out must be a contiguous bfloat16 [{M}, {cfg.model_dim}] tensor"
            )
        part = (
            out
            if out is not None
            else torch.empty(
                (M, cfg.model_dim), dtype=torch.bfloat16, device=self.device
            )
        )
        if M == 0:
            self.last_path = ()
            return part
        fused = []
        moe_a8w4.fused_moe_a8w4(
            x_local,
            self._weights,
            topk_weights,
            topk_ids,
            output=part,
            swiglu_limit=self._limit,
            reduce=self._reducer(M, fused) if self._fuses(M) else None,
            decode_down=self._decode_down(M, fused) if self._decode_fuses(M) else None,
            prefill_down=self._prefill_down(M, fused) if self._fuses(M) else None,
            stream_down=self._stream_down(M, fused) if self._fuses(M) else None,
        )
        if fused:
            self._seen.add(M)
            self.last_path = tuple(moe_a8w4.last_path)
            return part
        y = self._comm.all_reduce(part)
        self.last_path = (*moe_a8w4.last_path, self._comm.last)
        if out is not None and y is not out:
            out.copy_(y)
            return out
        return y

    __call__ = forward

    def _fuses(self, M):
        """Whether an M-row batch fuses the cross-rank sum into its last kernel."""
        return (
            self._ws is not None
            and M >= FUSED_MIN_M
            and M * self.cfg.model_dim * 2 >= FUSED_MIN_BYTES[self.cfg.world_size]
        )

    def _decode_fuses(self, M):
        """Whether an M-row batch on the decode path sums across the ranks in its Down kernel."""
        return self._ws is not None and M <= DECODE_TP_MAX_M

    def _decode_down(self, M, fused):
        """The ``decode_down`` hook of ``fused_moe_a8w4``: decode_tp in place of the Down decode kernel and the
        all-reduce (appends to ``fused``); from DECODE_TP_PREQUANT_MIN_M rows on the slot-pair kernel, on rows it
        quantizes itself if ``fused_moe_a8w4`` passes the single-slot kernel's inputs.
        """
        cfg, ws = self.cfg, self._ws
        limit = _wait_limit(WAIT_S if M in self._seen else FIRST_WAIT_S)

        def down(paired, mid, mid_scale, topk_ids, topk_weight, weights, output):
            topk = topk_ids.shape[1]
            if (
                not paired
                and M >= DECODE_TP_PREQUANT_MIN_M
                and moe_a8w4._down_pq_fits(cfg.model_dim, weights, topk)
            ):
                mid, mid_scale = moe_a8w4._quantize_decode(
                    mid, M * topk, weights.inter_padded, False
                )
                paired = True
            launch = decode_tp.compile_down_decode_tp(
                N=cfg.model_dim,
                K=weights.inter_padded,
                TOPK=topk,
                WAVES=moe_a8w4._decode_waves(topk, paired),
                WORLD=cfg.world_size,
                MAX_M=DECODE_TP_MAX_M,
                PAIRED=paired,
            )
            moe_a8w4._launch(
                "a8w4_down_decode_pq_tp" if paired else "a8w4_down_decode_tp",
                launch,
                mid,
                mid_scale,
                topk_ids,
                topk_weight,
                weights.w2,
                weights.w2_delta,
                weights.w2_scale,
                output,
                M,
                ws.dtab,
                ws.parity,
                self._err,
                cfg.rank,
                limit,
            )
            fused.append(M)

        return down

    def _prefill_down(self, M, fused):
        """The ``prefill_down`` hook of ``fused_moe_a8w4``: the Down kernel with the cross-rank sum streamed out of
        it (down_tp), then ``a8w4_tp_reduce`` copying the other ranks' rows (appends to ``fused``); None if the
        batch does not stream (``STREAM_DOWN``, ``STREAM_SPEC``). In token chunks (``CHUNK_SPEC``, attribute
        ``chunks``): one one-shot Down per chunk (``DownTP.OS``), then ``a8w4_tp_reduce`` summing both ranks'
        partials (``compile_tp_gather(OS=1)``)."""
        cfg, ws = self.cfg, self._ws
        spec = _stream_spec(cfg.world_size, cfg.model_dim, cfg.topk, M)
        if not STREAM_DOWN or spec is None:
            return None
        starts, tab = (), ws.tab2
        entry = CHUNK_SPEC.get((cfg.world_size, cfg.model_dim)) if CHUNK_DOWN else None
        if entry is not None and M >= entry[0] and cfg.world_size == 2 and ws.sdma_table() is not None:
            starts, tab = _chunk_starts(M, cfg.world_size * spec.TT, entry[1]), ws.tab3
            spec = dataclasses.replace(
                spec, NY=0, NYT=0, OS=1, STAMP=OS_STAMP, **(entry[2] if len(entry) > 2 else {})
            )
        key = (M, starts) if starts else M
        limit = _wait_limit(WAIT_S if key in self._seen else FIRST_WAIT_S)

        def down(mid, mid_scale, routing, loc_ids, routes, output, weights, grid, xcd, c=0):
            launch = down_tp.compile_down_tp(
                N=cfg.model_dim,
                K=weights.inter_padded,
                TOPK=loc_ids.shape[1],
                BLOCK_TILE_SIZE_M=moe_a8w4.PREFILL_BLOCK_M,
                E=weights.experts,
                XCD_CHUNK=xcd,
                tp=spec,
            )
            moe_a8w4._launch(
                f"a8w4_down_prefill_bm{moe_a8w4.PREFILL_BLOCK_M}_tp",
                launch,
                mid,
                weights.w2,
                weights.w2_delta,
                routes,
                *routing,
                weights.w2_scale,
                mid_scale,
                loc_ids,
                output,
                ws.ctr,
                tab,
                ws.epoch2,
                self._err,
                M,
                cfg.rank,
                ws.tpq,
                limit,
                grid,
            )
            if c < len(starts) - 1:
                return
            tiles = -(-M // spec.TT)
            items = spec.NSPLIT * tiles
            per_wg = GATHER_ITEMS if items >= GATHER_FROM else 1
            one_shot, extra = {}, ()
            if starts:
                # Every item: this rank's partial plus the peer's pushed one (down_tp rule 12).
                firsts = [s // spec.TT for s in starts[1:]]
                one_shot = dict(OS=1, STAMP=spec.STAMP)
                extra = (*firsts, *[tiles] * (down_tp.MAX_CHUNKS - 1 - len(firsts)), len(starts))
            gather = down_tp.compile_tp_gather(
                WORLD=cfg.world_size,
                H=cfg.model_dim,
                TT=spec.TT,
                NSPLIT=spec.NSPLIT,
                DRY=spec.DRY,
                G=per_wg,
                **one_shot,
            )
            moe_a8w4._launch(
                "a8w4_tp_reduce",
                gather,
                output,
                tab,
                ws.epoch2,
                ws.ctr,
                self._err,
                M,
                cfg.rank,
                ws.tpq,
                limit,
                -(-items // per_wg),
                *extra,
            )
            fused.append(M)
            if starts:
                self._seen.add(key)

        if starts:
            down.chunks = (starts, functools.partial(self._chunk_desc, M, starts))
        return down

    def _chunk_desc(self, M, starts, grid):
        """The chunk descriptors of ``fused_moe_a8w4``'s ``prefill_down.chunks`` for a call of ``grid`` sorted blocks
        (int32 [chunks, ``moe_a8w4.chunks.desc_row_words(grid)``], ``moe_a8w4.chunks``; cached per batch size,
        split and grid; the host sets the tokens and indices, the sort's block lists the rest)."""
        desc = self._chunk_descs.get((M, starts, grid))
        if desc is None:
            words = a8w4_chunks.desc_row_words(grid)
            desc = torch.zeros((len(starts), words), dtype=torch.int32, device=self.device)
            for c, (lo, hi) in enumerate(zip(starts, (*starts[1:], M))):
                desc[c, 1].fill_(lo)
                desc[c, 2].fill_(hi)
                desc[c, 3].fill_(c)
            self._chunk_descs[(M, starts, grid)] = desc
        return desc

    def _stream_down(self, M, fused):
        """The ``stream_down`` hook of ``fused_moe_a8w4``: the streaming Down kernel with the cross-rank sum streamed
        out of it (down_tp, four waves of ``STREAM_TP_R`` row blocks), then ``a8w4_tp_reduce`` copying the other
        ranks' rows (appends to ``fused``); None if the batch does not stream (``STREAM_DOWN``, ``STREAM_TIER``,
        ``STREAM_TIER_SPEC``)."""
        cfg, ws = self.cfg, self._ws
        atomic = moe_a8w4._stream_atomic(self._weights.inter_padded)
        tp = _stream_spec(cfg.world_size, cfg.model_dim, cfg.topk, M, STREAM_TIER_SPEC, COPY=int(atomic))
        if not (STREAM_DOWN and STREAM_TIER) or tp is None:
            return None
        limit = _wait_limit(WAIT_S if M in self._seen else FIRST_WAIT_S)

        def down(mid, mid_scale, routing, loc_ids, routes, output, weights, block_m, task_num):
            H, K = cfg.model_dim, weights.inter_padded
            assert (routes is None) == atomic, (routes is None, atomic)
            launch = down_tp.compile_down_stream_tp(
                N=H,
                K=K,
                TOPK=loc_ids.shape[1],
                BLOCK_M=block_m,
                E=weights.experts,
                ATOMIC=atomic,
                R=STREAM_TP_R,
                SHARE=moe_a8w4._stream_down_shared(H, K, block_m),
                PREFETCH=moe_a8w4.STREAM_DOWN_PREFETCH,
                W_AUX=(
                    moe_a8w4.STREAM_DOWN_W_AUX_H5120
                    if H >= 5120
                    else moe_a8w4.STREAM_DOWN_W_AUX
                ),
                tp=tp,
            )
            moe_a8w4._launch(
                f"a8w4_down_stream_bm{block_m}_tp",
                launch,
                mid,
                weights.w2,
                weights.w2_delta,
                output if atomic else routes,
                *routing,
                weights.w2_scale,
                mid_scale,
                M,
                task_num,
                loc_ids,
                output,
                ws.ctr,
                ws.tab2,
                ws.epoch2,
                self._err,
                cfg.rank,
                ws.tpq,
                limit,
            )
            items = tp.NSPLIT * -(-M // tp.TT)
            per_wg = GATHER_ITEMS if items >= GATHER_FROM else 1
            gather = down_tp.compile_tp_gather(
                WORLD=cfg.world_size,
                H=H,
                TT=tp.TT,
                NSPLIT=tp.NSPLIT,
                DRY=tp.DRY,
                G=per_wg,
            )
            moe_a8w4._launch(
                "a8w4_tp_reduce",
                gather,
                output,
                ws.tab2,
                ws.epoch2,
                ws.ctr,
                self._err,
                M,
                cfg.rank,
                ws.tpq,
                limit,
                -(-items // per_wg),
            )
            fused.append(M)

        return down

    def _reducer(self, M, fused):
        """The ``reduce`` hook of ``fused_moe_a8w4``: tp_reduce in place of its last step (appends to ``fused``)."""
        cfg, ws = self.cfg, self._ws
        tt = _tile_tokens(M, cfg.model_dim)
        limit = _wait_limit(WAIT_S if M in self._seen else FIRST_WAIT_S)

        def reduce(output, loc_ids, routes):
            launch = tp_reduce.compile_tp_reduce(
                WORLD=cfg.world_size,
                H=cfg.model_dim,
                TOPK=0 if loc_ids is None else loc_ids.shape[1],
                TT=tt,
            )
            moe_a8w4._launch(
                "a8w4_tp_reduce",
                launch,
                ws.epoch if loc_ids is None else loc_ids,
                output if routes is None else routes,
                output,
                ws.tab,
                ws.epoch,
                self._err,
                M,
                cfg.rank,
                ws.slots,
                limit,
                -(-M // tt),
            )
            fused.append(M)

        return reduce

    def prepare(
        self,
        local_tokens,
        tail: bool = False,
        tail_bf16: bool = False,
        warmup: bool = True,
    ) -> None:
        """Collective: compile and run once every kernel of the batches of these local token counts (ar: every
        replicated batch M with ceil(M / world_size) in ``local_tokens``), so that serving and CUDA graph capture
        compile nothing; then check every rank's watchdog: on a timeout ``reset()`` and retry once, else raise.
        """
        if tail or tail_bf16:
            raise ValueError("MegaMoeTPGfx942: no fused tail (gfx950 ag_rs only)")
        if not warmup:
            return
        cfg = self.cfg
        ts = [local_tokens] if isinstance(local_tokens, int) else list(local_tokens)
        tp = cfg.world_size
        rows = sorted(
            {m for t in ts for m in range((t - 1) * tp + 1, t * tp + 1) if m > 0}
        )
        for _ in range(2):
            for m in rows:
                x = torch.zeros(
                    (m, cfg.model_dim), dtype=torch.bfloat16, device=self.device
                )
                # balanced routing: token i's slot j to expert (i * topk + j) % experts (distinct within a token)
                ids = (
                    torch.arange(m, device=self.device)[:, None] * cfg.topk
                    + torch.arange(cfg.topk, device=self.device)
                ) % cfg.experts
                tw = torch.full(
                    (m, cfg.topk),
                    1.0 / cfg.topk,
                    dtype=torch.float32,
                    device=self.device,
                )
                self.forward(x, tw, ids.to(torch.int32))
            torch.cuda.synchronize(self.device)
            if self._ws is None:
                return
            errs = [None] * tp
            dist.all_gather_object(errs, self.poll_errors(), group=self._comm.cpu_group)
            if not any(errs):
                return
            self.reset()
        raise RuntimeError(
            f"MegaMoeTPGfx942.prepare: warmup launches timed out (error bits per rank: {errs})"
        )

    @property
    def max_local_tokens(self) -> int:
        return self.cfg.max_local_tokens

    def tail_ok(self, local_tokens: int) -> bool:
        """forward(tail=...) runs for this local token count: never (no fused tail)."""
        return False

    # -- errors / lifetime --------------------------------------------------------------------------------------
    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bits (nonzero: a wait of a fused launch gave up; ``tp_reduce.ERR_TIMEOUT`` = 1,
        ``decode_tp.ERR_TIMEOUT`` = 2) as a device tensor (no sync)."""
        return self._err

    def poll_errors(self) -> int:
        """Nonzero if a wait inside a fused launch gave up (a peer never arrived); synchronizes."""
        return int(self._err.item())

    def check_errors(self) -> None:
        """Raise RuntimeError (and clear) if a wait gave up since the last check."""
        bits = self.poll_errors()
        if bits:
            self.clear_errors()
            raise RuntimeError(
                f"MegaMoeTPGfx942: a wait on a peer gave up (error bits {bits:#x}): outputs since the last check "
                "are invalid; reset() (collective) before the next forward"
            )

    def clear_errors(self) -> None:
        self._err.zero_()

    def reset(self) -> None:
        """Collective: restart the cross-rank state (after a timeout / desync) and clear this instance's errors."""
        if self._ws is not None:
            self._ws.reset()
        self.clear_errors()

    def close(self) -> None:
        """Collective: release the weights and this instance's share of the fused sum's buffers and of the
        communicator (unusable afterwards)."""
        self._weights = None
        if self._ws is not None:
            _release_workspace(self._ws_key)
            self._ws = None
        if self._comm is not None:
            _release_comm(self._comm_key)
            self._comm = None
