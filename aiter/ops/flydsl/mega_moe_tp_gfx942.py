# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer for gfx942 with the interface of ``MegaMoeTP`` (``comm_mode="ar"`` and ``"ag_rs"``).

Experts are replicated over the TP group and ``inter_dim`` is sharded. ``comm_mode="ar"``: every rank gets the same
input rows and routing; ``forward`` runs this rank's slice with the single-GPU A8W4 op
(``moe_a8w4.flydsl_moe_a8w4_ptpc``: FP8 per-token activations, MXFP4 routed experts, an optional FP8 shared expert)
and sums the partial outputs over the ranks:

* TP2 / TP4 batches of at most ``DECODE_TP_MAX_M`` rows on the per-route decode kernels (the shared expert's mixed
  decode and its fold into the slot-pair Down included; the op's decode tier ends at ``DECODE_TIER_MAX_M`` rows in
  the shapes listed there): the sum is fused into the Down decode kernel
  (``kernels/mega_moe_tp_gfx942/decode_tp.py``, ``a8w4_down_decode[_pq[_shared]|_mixed]_tp``: each workgroup sends
  its f32 partials of 16 channels to the other ranks' inboxes over hipIpc symmetric memory and adds the ranks'
  partials in rank order; the slot-pair ``_pq`` kernel from ``DECODE_TP_PREQUANT_MIN_M`` rows on);
* TP4 batches past the decode path whose BF16 output has fewer than ``FUSED_MIN_BYTES[world_size]`` bytes, on the
  op's streaming and prefill tiers that end in a route sum (not after the atomic streaming Down): the route sum and
  the sum are one kernel (``kernels/mega_moe_tp_gfx942/sum_tp.py``, ``a8w4_route_sum_tp``: the custom all-reduce's
  two-shot push protocol over hipIpc symmetric memory, ``MID_WORLDS``);
* TP2 / TP4 batches of more than ``FUSED_MIN_M - 1`` rows whose BF16 output has at least
  ``FUSED_MIN_BYTES[world_size]`` bytes, on the op's streaming and prefill tiers: the sum is fused into the last
  kernels (the op's private entry ``moe_a8w4._flydsl_moe_a8w4_ptpc_tp`` with ``moe_a8w4._TPHooks``). On the shapes
  of ``STREAM_SPEC`` / ``STREAM_TIER_SPEC`` (``STREAM_DOWN``) the sum streams out of the Down kernel
  (``kernels/mega_moe_tp_gfx942/down_tp.py``, ``a8w4_down_prefill_bm*_tp`` / ``a8w4_down_stream_bm*_tp``: workgroups
  of the Down launch reduce, scatter and gather each column block of the output while the next ones compute), and
  ``a8w4_tp_reduce`` only copies the rows the other ranks summed into the output; otherwise the op's last reduction
  is ``kernels/mega_moe_tp_gfx942/tp_reduce.py`` (``a8w4_tp_reduce``: route sum, reduce-scatter of token tiles to
  their owner rank over hipIpc symmetric memory, f32 sum in rank order, all-gather of the BF16 rows). A shared expert
  folded into the routed experts' last reduction (``moe_a8w4._SharedFold``) is one more route of that sum.
* otherwise (the other batches, and the tiers the hooks do not reach: the expert-deduplicated decode and a shared
  expert added by ATOM's split; other TP sizes): the BF16 partials are all-reduced: aiter's custom all-reduce
  (``CustomAllreduce``) below ``AR_RCCL_MIN_BYTES[world_size]`` bytes where its ``should_custom_ar`` accepts the
  tensor, else RCCL (``torch.distributed.all_reduce`` on ``group``).

The outputs of all ranks are bitwise identical.

``comm_mode="ag_rs"`` (data-parallel attention in front, sequence-parallel in and out): every rank passes its own
M_r tokens (the same M_r on every rank) and gets their output. ``forward`` gathers every rank's rows and routing
(all-gather; rank r's rows at [r * M_r, (r + 1) * M_r), the tokens r is the home rank of), runs this rank's slice
of the layer on all M = world_size * M_r rows with the same op and sends each rank the sum of the partials of its
rows (reduce-scatter). The all-gather is one ``custom_all_gatherv`` of x and the routing below
``AG_RCCL_MIN_BYTES[world_size]`` bytes of x where it takes them, else RCCL for x (``all_gather_into_tensor``) and
the custom gather (RCCL where it declines) for the routing. At every M_r the home rank of a token sums the ranks'
partials of it in f32 in rank order 0, 1, ... and rounds to BF16 once:

* the batches whose sum the ar mode fuses fuse it the same way, the home rank of each token its owner and nothing
  gathered back: ``decode_tp`` with ``DEST="dp"`` (``a8w4_down_decode[_pq[_shared]|_mixed]_tp_rs``); ``down_tp``
  with ``OWNER=OWN_HOME`` and ``AG=0`` (``a8w4_down_prefill_bm*_tp_rs`` / ``a8w4_down_stream_bm*_tp_rs``, then
  ``a8w4_tp_reduce`` only waits) on the prefill Down of ``STREAM_SPEC`` and the streaming Down of
  ``STREAM_TIER_SPEC_DP``; else ``tp_reduce`` with ``OWNER=OWN_HOME`` and ``AG=0`` (``a8w4_tp_reduce_rs``). Tile
  ``r + world_size * i`` holds rank r's tokens ``i * TT ..``: the tiles in flight together go to every owner;
  ``sum_tp`` with ``DEST="dp"`` (``a8w4_route_sum_tp_rs``) below the fused Down;
* otherwise (``_Comm.reduce_scatter_ordered``) ``custom_reduce_scatter`` where its ``should_custom_rs`` accepts the
  partials (it sums in that order), else an RCCL all-to-all of the ranks' rows (``all_to_all_single``) and the sum
  on this rank. ``_Comm.reduce_scatter`` (RCCL's ``reduce_scatter_tensor`` from ``RS_RCCL_MIN_BYTES[world_size]``
  bytes of partials on: its ring adds in BF16 hop by hop) is no longer on the path.

The buffers of the fused sums (``_Workspace``) are allocated once per (communicator, ``model_dim``,
``max_local_tokens``) and shared by those instances. Both modes are deterministic: the same inputs give bitwise the
same outputs, but for the streaming Down that adds its routes with BF16 atomics (``moe_a8w4._stream_atomic``), as
in the single-GPU op.

Weights: the routed experts come in the ``MegaMoeTP`` layout (MXFP4, ``shuffle_weight(16, 16)`` + ``e8m0_shuffle``)
and are unshuffled and converted once (constructor / ``set_weights``) by ``moe_a8w4.prepare_moe_a8w4_ptpc_weight``;
``inter_dim`` is zero-padded to a multiple of 64. The optional shared expert (DeepSeek-V4.1-Flash; all four of
``shared_w1``, ``shared_w1_scale``, ``shared_w2``, ``shared_w2_scale`` or none) is this rank's TP slice of it in the
formats ``prepare_moe_a8w4_ptpc_weight`` takes: FP8 (``float8_e4m3fn`` or ``float8_e4m3fnuz``) ``shared_w1``
[2 * inter_dim, model_dim] (this rank's gate rows, then its up rows) and ``shared_w2`` [model_dim, inter_dim], with
float32 per-row scales ([N] or [N, 1]) or E8M0 scales of 32x32 blocks ([N / 32, K / 32]). Whether a layer has a
shared expert is fixed at construction.

Routing (ATOM's convention, as the single-GPU op): ``cfg.experts`` / ``cfg.topk`` are the routed experts' E / topk.
Without a shared expert ``topk_ids`` / ``topk_weights`` are [M, topk]; with one they are [M, topk + 1], the last
column holding the shared expert's id E (DeepSeek-V4.1-Flash: E385 / topk 7 around E = 384 routed experts). Routed
ids lie in [0, E) or are -1 (a masked route); a row's shared expert counts with its last-column weight when its first
id is >= 0 and its last id is E, else with weight 0.

Contract (as ``MegaMoeTP``):

* ``forward`` is a collective: every rank calls it the same number of times, in the same order, with the same
  input and routing (ar) or the same number of local tokens M_r (ag_rs; pad a shorter batch with rows whose
  ``topk_ids`` are all -1: such a row gets 0, and a first id of -1 also drops the row's shared expert).
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles and runs every kernel these batches launch,
  then checks every rank's watchdog (``reset()`` and retry once on a timeout, else raise).
* A wait of a fused kernel on a peer gives up after ``WAIT_S`` seconds (``FIRST_WAIT_S`` on an instance's first
  forward of a batch size, which may still be compiling on a peer), sets a sticky bit and goes on: that launch's
  output is invalid, never a hang. ``error_flag()`` is that bit as a device tensor (no sync);
  ``check_errors()`` synchronizes and raises; ``reset()`` (collective) restarts the cross-rank state (device-side
  epochs, flags, counters, decode and sum_tp inboxes and parities shared by the instances of the workspace), which a
  timed-out launch leaves inconsistent.
* ``activation`` must be ``Silu``; a finite ``swiglu_limit`` L > 0 computes
  ``SiLU(gate.clamp(max=L)) * up.clamp(-L, L)`` (DeepSeek-V4.1), the shared expert's too. The collectives are BF16
  (the ag_rs routing int32 / float32) whatever ``comm_dtype`` says; the gfx950 fields (``schedule``, ``act_dtype``,
  ``lb_*``, ``bigm_min_tokens``, Situ betas, ``ar_gather="fp8"``) and the fused tail are rejected.

Errors: the constructor raises ValueError off gfx942, for an unknown ``comm_mode``, for the rejected config fields
above and for a cfg rank / world_size other than the group's. The constructor and
``set_weights`` raise TypeError for routed weights of another dtype, ValueError for other shapes or another device
than the layer's, pass the shared expert's errors of ``prepare_moe_a8w4_ptpc_weight`` on (ValueError for a partial
set, other dtypes, shapes or devices, mixed scale formats, NaN elements or scales and the E8M0 code 255; TypeError
for an argument that is no tensor), and ``set_weights`` raises ValueError when it would add or drop the shared
expert. ``forward`` raises ValueError for a tail, an x that is no bfloat16 [M, model_dim], routing of another shape
than [M, topk] ([M, topk + 1] with a shared expert), more than ``max_local_tokens * world_size`` rows and an ``out``
that is no contiguous bfloat16 [M, model_dim]; the single-GPU op's checks follow (TypeError for routing of other
dtypes, ValueError for a non-contiguous x). In the ag_rs mode it checks everything before the first collective:
ValueError for a tail, an x that is no contiguous bfloat16 [M_r, model_dim], routing of another shape than
[M_r, topk] ([M_r, topk + 1]), inputs on another device than the layer's, M_r > ``max_local_tokens`` and an ``out``
that is no contiguous bfloat16 [M_r, model_dim]; TypeError for ids other than int32 / int64 or weights that are not
floating; with ``moe_a8w4.DEBUG_CHECKS`` (``AITER_MOE_A8W4_DEBUG=1``; outside a CUDA graph capture) ValueError on
every rank when the ranks' M_r differ (compared over the gloo group). ``prepare`` raises RuntimeError when the
warmup launches time out twice.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist

from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx_runtime

from . import moe_a8w4
from .kernels.mega_moe_tp_gfx942 import decode_tp, down_tp, sum_tp, tp_reduce
from .mega_moe_tp import MegaMoeTPConfig

__all__ = ["MegaMoeTPConfig", "MegaMoeTPGfx942", "mega_moe_tp_gfx942_supported"]

# The all-reduce of the BF16 partials: aiter's custom all-reduce below AR_RCCL_MIN_BYTES[world_size] bytes (where its
# should_custom_ar accepts the tensor), RCCL from there on; other TP sizes take the custom all-reduce wherever it
# accepts. Cold paired timing on MI308X (CA / RCCL us): TP2 5 MiB 136 / 143, 6.25 MiB 173 / 173, 7.5 MiB 202 / 200;
# TP4 11.25 MiB 168 / 168, 12.5 MiB 187 / 188, 13.75 MiB 201 / 198.
AR_RCCL_MIN_BYTES = {2: 6 << 20, 4: 13 << 20}
# ag_rs: the all-gather takes one custom_all_gatherv of x and the routing below AG_RCCL_MIN_BYTES[world_size] bytes of
# x per rank, RCCL for x from there on (the routing still through the custom gather); the reduce-scatter takes the
# custom one below RS_RCCL_MIN_BYTES[world_size] bytes of partials (world_size * M_r rows), RCCL from there on. Other
# TP sizes take the custom collectives wherever they accept. Cold paired timing on idle MI308X, TP4 H5120 (custom /
# RCCL us): all-gather at M_r 576 / 614 / 640 171 / 173, 184 / 183, 192 / 188 (DeepSeek x: M_r <= 614 custom);
# reduce-scatter at M_r 640 / 704 / 768 196 / 196, 214 / 213, 233 / 226 (M_r <= 665 custom).
AG_RCCL_MIN_BYTES = {4: 6 << 20}
RS_RCCL_MIN_BYTES = {4: 26 << 20}
COMM_MODES = ("ar", "ag_rs")


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
        self.world = dist.get_world_size(group=group)
        _align_ipc_pool_seq(cpu_group)
        ca = CustomAllreduce(group=cpu_group, device=device)
        self.ca = None if ca.disabled else ca
        self.last = None
        self.last_ag = self.last_rs = None

    def uses_custom(self, y):
        """Whether ``all_reduce(y)`` takes the custom all-reduce (see AR_RCCL_MIN_BYTES), else RCCL."""
        nbytes = y.numel() * y.element_size()
        return (
            self.ca is not None
            and nbytes < AR_RCCL_MIN_BYTES.get(self.world, math.inf)
            and self.ca.should_custom_ar(y)
        )

    def all_reduce(self, y):
        """Sum of ``y`` over the group: a new tensor (custom all-reduce) or ``y`` itself (RCCL, in place), whichever
        ``uses_custom`` picks. ``last`` names the one used."""
        if self.uses_custom(y):
            self.last = "custom_all_reduce"
            return self.ca.custom_all_reduce(y)
        self.last = "rccl_all_reduce"
        dist.all_reduce(y, group=self.group)
        return y

    def all_gather(self, x, topk_ids, topk_weights):
        """Every rank's rows of x [m, H] BF16, topk_ids [m, K] int32 and topk_weights [m, K] float32 (contiguous, the
        same m on every rank) -> three new tensors [world * m, ...], rank r's rows at r * m (see AG_RCCL_MIN_BYTES).
        ``last_ag`` names the collective that gathered x ("custom_all_gather" / "rccl_all_gather").
        """
        world, m = self.world, x.shape[0]
        if self.ca is not None and x.numel() * x.element_size() < AG_RCCL_MIN_BYTES.get(
            world, math.inf
        ):
            # None (on every rank alike) where one launch cannot take them
            got = self.ca.custom_all_gatherv([x, topk_ids, topk_weights])
            if got is not None:
                self.last_ag = "custom_all_gather"
                return tuple(got)
        xs = x.new_empty((world * m, x.shape[1]))
        dist.all_gather_into_tensor(xs, x, group=self.group)
        got = (
            self.ca.custom_all_gatherv([topk_ids, topk_weights])
            if self.ca is not None
            else None
        )
        if got is None:
            k = topk_ids.shape[1]
            packed = torch.cat([topk_ids, topk_weights.view(torch.int32)], dim=1)
            every = packed.new_empty((world * m, 2 * k))
            dist.all_gather_into_tensor(every, packed, group=self.group)
            got = (
                every[:, :k].contiguous(),
                every[:, k:].contiguous().view(torch.float32),
            )
        self.last_ag = "rccl_all_gather"
        return (xs, *got)

    def reduce_scatter(self, part, out):
        """out [m, H] = the sum over the ranks of their rows [rank * m, (rank + 1) * m) of part [world * m, H] (BF16,
        contiguous; see RS_RCCL_MIN_BYTES) -> out. ``last_rs`` names the collective ("custom_reduce_scatter": each
        element summed in f32 in rank order, rounded once / "rccl_reduce_scatter": the ring, BF16 per hop).
        """
        nbytes = part.numel() * part.element_size()
        if (
            self.ca is not None
            and nbytes < RS_RCCL_MIN_BYTES.get(self.world, math.inf)
            and self.ca.should_custom_rs(part, 0)
        ):
            self.ca.custom_reduce_scatter(part, out, 0)
            self.last_rs = "custom_reduce_scatter"
        else:
            dist.reduce_scatter_tensor(out, part, group=self.group)
            self.last_rs = "rccl_reduce_scatter"
        return out

    def reduce_scatter_ordered(self, part, out):
        """``reduce_scatter`` that sums each element in f32 in rank order and rounds once at every size (the ag_rs
        layer's batches whose sum no kernel fuses): the custom reduce-scatter wherever its ``should_custom_rs``
        accepts ``part``, else an RCCL all-to-all of the ranks' rows ("rccl_all_to_all") and the sum on this rank.
        """
        if self.ca is not None and self.ca.should_custom_rs(part, 0):
            self.ca.custom_reduce_scatter(part, out, 0)
            self.last_rs = "custom_reduce_scatter"
            return out
        m = out.shape[0]
        rows = torch.empty_like(part)
        dist.all_to_all_single(rows, part, group=self.group)
        acc = rows[:m].float()
        for s in range(1, self.world):
            acc += rows[s * m : (s + 1) * m].float()
        out.copy_(acc)
        self.last_rs = "rccl_all_to_all"
        return out

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
# kernel (decode_tp): every batch of the decode path (the S2 prototype's MXFP4 kernels were 2-4 % slower fused at 32
# rows; the gfx9_fp8 slot-pair kernels are not: MI308X, bench protocol, paired with the all-reduce, 32 rows TP2
# 0.972x, TP4 0.997x the time).
DECODE_TP_MAX_M = moe_a8w4.DECODE_MAX_M
# From DECODE_TP_PREQUANT_MIN_M rows on (below moe_a8w4.DECODE_PREQUANT_MIN_M) the fused Down is the slot-pair kernel
# on pre-quantized rows (MI308X, bench protocol, gfx9_fp8 kernels, single-slot / slot-pair fused: 2 rows TP2
# 25.7 / 28.6 us, TP4 21.2 / 23.4 us; 3 rows TP2 35.7 / 35.0 us, TP4 27.2 / 27.2 us; the MXFP4-only version of M7
# took the slot-pair kernel from 2 rows on).
DECODE_TP_PREQUANT_MIN_M = 3
# The batches between (more than DECODE_TP_MAX_M rows, or past the op's decode path, BF16 output below
# FUSED_MIN_BYTES) of these TP sizes sum across the ranks in the op's route sum (sum_tp: the route sum, then the
# custom all-reduce's two-shot protocol, one launch for both; MI308X, bench protocol, ds_layer_tp4 16 / 32 / 48 / 64
# rows 0.969 / 0.981 / 0.985 / 0.979x the time with the all-reduce). After the atomic streaming Down (no route sum)
# they keep the custom all-reduce: tp2 / tp4 48-256 rows took 0.5-1 % longer with the sum alone in sum_tp's protocol
# (one-shot at TP2), and so did the non-atomic Down with sum_tp at TP4.
MID_WORLDS = (4,)
# The op's decode tier stops at DECODE_TIER_MAX_M[(world size, hidden size)] rows in these layers (other shapes: the
# op's limit; ``moe_a8w4._TPHooks.decode_max_m``, both modes): with the inter dim sharded the streaming tier and the
# custom all-reduce overtake the decode kernels with the fused sum (decode_tp) well below the op's DECODE_MAX_M. MI308X,
# bench protocol, same-GPU ABBA, streaming / decode, Qwen3.8 TP4 M 8 / 10 / 12 / 14 / 16 / 24 / 32: 1.04 / 1.01 /
# 0.955 / 0.91 / 0.90 / 0.85 / 0.83x (20261127-101618-L2-run-d709, 20261127-102239-L2-run-05ed). TP2 crosses at 6
# rows (M 5 / 6 / 7 / 8 / 16 / 24 / 32: 1.10 / 1.01 / 0.97 / 0.92 / 0.925 / 0.89 / 0.84x) but keeps the op's limit
# for now: the basic tests pin its pre-quantized decode at 32 rows.
DECODE_TIER_MAX_M = {(4, 2560): 10}
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
    (4, 2560): (
        (8192, _DOWN_TUNE_TP4),
        (None, {**_DOWN_TUNE_TP4, "ROUTE_AUX": down_tp.SYS_NT}),
    ),
    (4, 5120): {"SB": 1, "LAS": 1},
}
# The same for the fused batches on the stream path (the streaming Down's own column blocks, DownTP.COPY set by the
# caller). Paired timing on MI308X against the Down kernel and tp_reduce (20261126-100503-p1-run-aaf7): TP2 0.94 /
# 0.89 (M 512 / 1024), TP4 0.97 / 0.93 / 0.92 (M 512 / 1024 / 2048) but 1.05 at M 256; DeepSeek-V4.1 TP4 1.01-1.04
# (M 128-512, 20261126-092445-p1-run-d2d3) and 0.98 (M 1024, 20261126-102139-p1-run-d3d7).
_STREAM_TIER_TP = (5, 16, 8, 8, {"UX": 1, "NXT": 128, "NYT": 64})
_STREAM_TIER_H5120 = (5, 16, 8, 8, {"NXT": 128, "NYT": 64})
STREAM_TIER_SPEC = {
    (2, 2560): _STREAM_TIER_TP,
    (4, 2560): ((384, None), (None, _STREAM_TIER_TP)),
    (4, 5120): ((512, None), (None, _STREAM_TIER_H5120)),
}
# ag_rs (OWNER = OWN_HOME, AG = 0: the owners keep the sums of their own tokens, nothing is pushed back): the
# fused batches of the stream path whose streaming Down streams the sum out (as STREAM_TIER_SPEC); the others end
# in tp_reduce (a8w4_tp_reduce_rs). Paired timing on MI308X, DeepSeek-V4.1 DP4 against the ag_rs reduce-scatter
# (20261127-084710-L1-run-577e): tp_reduce 0.98-1.00, the streaming Down 1.01-1.06 (M_r 32-256). TP2 / TP4 (not
# timed in ag_rs) the same.
STREAM_TIER_SPEC_DP = {}
# The DownTP fields of the ag_rs layer (module docstring of down_tp, Owners). The prefill Down of STREAM_SPEC
# streams the sum out in ag_rs too: 0.92-0.93 against the ag_rs reduce-scatter, tp_reduce 0.95-0.96 (DeepSeek-V4.1
# DP4 M_r 512-8192, the same run).
_HOME_OWNERS = {"OWNER": down_tp.OWN_HOME, "AG": 0}
# Items (tiles of a column block) per workgroup of the gather kernel (down_tp.compile_tp_gather, G) of a call
# with at least GATHER_FROM items, else 1. Paired timing on MI308X (20261126-112217-p1-run-090b), 4 against 1
# item: 0.5-1.0 % faster at TP2 / TP4 M 8192 and 32768 and DeepSeek-V4.1 TP4 M 4096 and 16384, unchanged at
# 640-1280 items.
GATHER_ITEMS = 4
GATHER_FROM = 2048
# Timing analysis (DownTP.TRACE): room for this many workgroups' timestamps after the counters.
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


def _wait_limit(seconds):
    """The ``limit`` launch argument of tp_reduce for a wait bound in seconds."""
    return max(1, math.ceil(seconds * _TIMER_HZ / tp_reduce.LIMIT_TICKS))


class _Workspace:
    """Buffers of the fused sums (their module docstrings) of one communicator, hidden size and batch bound
    ``m_max``. tp_reduce: symmetric ``landing`` / ``inbox`` / ``flags`` (hipIpc, uncached, zeroed), their address
    table ``tab`` and this rank's ``epoch``; down_tp: ``landing`` / ``inbox`` and its own symmetric ``flags2``, their
    table ``tab2``, this rank's ``epoch2`` and block counters ``ctr`` (room for the token chunks in ``flags2`` and
    ``ctr``); decode_tp: the symmetric ``dinbox`` (uncached, 0xFF), its address table ``dtab`` and this rank's
    ``parity``; sum_tp (batches of at most ``mid_m`` rows): the symmetric ``minbox`` / ``mland`` (uncached, 0xFF) of
    ``munits`` units, their address table ``mtab`` and this rank's ``mparity``. Creation and ``close`` are collective
    over the communicator's gloo group.
    """

    def __init__(self, comm, rank, world, H, m_max):
        from .communication_ops_host_utils import IpcSymmetricHeap

        # The largest batch of each tile size bounds the tiles and the inbox rows of that size, with the home-rank
        # owner map too (ag_rs, tp_reduce.OWN_HOME: tiles of each rank's ceil(m / world) tokens).
        tiers = [(m, _tile_tokens(m, H)) for m in {min(m_max, 1023), m_max}]
        self.slots = max(world * -(-m // (world * tt)) for m, tt in tiers)
        # Rows per owner rank: an owner's share of the tiles (tile i -> rank i % world), in tokens; the same bound
        # for OWN_HOME (ceil(ceil(m / tt) / world) == ceil(ceil(m / world) / tt)).
        rps = max((-(-m // tt) + world - 1) // world * tt for m, tt in tiers)
        # down_tp: tiles of its smallest tile size bound the slots, with the home-rank owner map too (ag_rs,
        # down_tp.OWN_HOME: tiles of each rank's ceil(m_max / world) tokens); an owner keeps its own partials in its
        # inbox.
        local = -(-m_max // world)
        self.tpq = world * -(-local // min(down_tp.TILE_TOKENS))
        self.slots2 = down_tp.NSPLIT_MAX * self.tpq
        rps2 = max(
            (-(-m_max // tt) + world - 1) // world * tt for tt in down_tp.TILE_TOKENS
        )
        # sum_tp: every batch below FUSED_MIN_BYTES of BF16 output.
        self.mid_m = (
            min(m_max, -(-FUSED_MIN_BYTES[world] // (2 * H)) - 1)
            if H % 512 == 0 and world in MID_WORLDS
            else 0
        )
        self.munits = max(1, self.mid_m * (H // 512))
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
            self.minbox = self.heap.create_tensor(
                (sum_tp.inbox_words(self.munits, world),), torch.int32
            )
            self.mland = self.heap.create_tensor(
                (sum_tp.landing_words(self.munits),), torch.int32
            )
            # Filled before the handles are published: from then on a peer may write into them.
            self.dinbox.fill_(decode_tp.SENTINEL)
            self.minbox.fill_(sum_tp.SENTINEL)
            self.mland.fill_(sum_tp.SENTINEL)
            torch.cuda.synchronize(comm.device)
        except Exception as exc:  # noqa: BLE001
            error = exc
        # Raises on every rank if one failed (and closes the heap collectively).
        self.heap.publish(error)
        self.tab = torch.stack(
            [self.heap.p2p_table(t) for t in (self.landing, self.inbox, self.flags)]
        ).view(torch.int32)
        self.epoch = torch.zeros(self.slots, dtype=torch.int32, device=comm.device)
        self.tab2 = torch.stack(
            [self.heap.p2p_table(t) for t in (self.landing, self.inbox, self.flags2)]
        ).view(torch.int32)
        self.epoch2 = torch.zeros(self.slots2, dtype=torch.int32, device=comm.device)
        self.ctr = torch.zeros(
            down_tp.ctr_words(True) + 4 * TRACE_WGS,
            dtype=torch.int32,
            device=comm.device,
        )
        self.rank, self.world = rank, world
        self.dtab = self.heap.p2p_table(self.dinbox).view(torch.int32)
        self.parity = torch.zeros(
            DECODE_TP_MAX_M * (H // 16), dtype=torch.int32, device=comm.device
        )
        self.mtab = torch.stack(
            [self.heap.p2p_table(t) for t in (self.minbox, self.mland)]
        ).view(torch.int32)
        self.mparity = torch.zeros(self.munits, dtype=torch.int32, device=comm.device)

    def reset(self):
        """Collective: zero the flags, epochs, counters and parities and refill the decode and sum_tp inboxes once no
        kernel of any rank runs, before any rank launches again."""
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)
        self.flags.zero_()
        self.epoch.zero_()
        self.flags2.zero_()
        self.epoch2.zero_()
        self.ctr.zero_()
        self.dinbox.fill_(decode_tp.SENTINEL)
        self.parity.zero_()
        self.minbox.fill_(sum_tp.SENTINEL)
        self.mland.fill_(sum_tp.SENTINEL)
        self.mparity.zero_()
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)

    def close(self):
        """Collective (``IpcSymmetricHeap.close``)."""
        self.heap.close()
        self.landing = self.inbox = self.flags = self.tab = self.epoch = None
        self.flags2 = self.tab2 = self.epoch2 = self.ctr = None
        self.dinbox = self.dtab = self.parity = None
        self.minbox = self.mland = self.mtab = self.mparity = None


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
    if cfg.comm_mode not in COMM_MODES:
        raise ValueError(
            f"MegaMoeTPGfx942: unknown comm_mode {cfg.comm_mode!r} (one of {COMM_MODES})"
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


_SHARED_NAMES = ("shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale")


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
        shared_w1: torch.Tensor | None = None,
        shared_w1_scale: torch.Tensor | None = None,
        shared_w2: torch.Tensor | None = None,
        shared_w2_scale: torch.Tensor | None = None,
        group=None,
        device: torch.device | None = None,
        cpu_group=None,
    ):
        """``shared_*``: this rank's slice of an optional shared expert (all four or none; see the module docstring),
        which then takes the last routing column. ``group``: the TP process group (RCCL; None: the default group).
        ``cpu_group``: a gloo group of the same ranks for the custom all-reduce; None: created with
        ``torch.distributed.new_group``, so every rank of the job must construct its layer at the same point (pass it
        when only the TP group's ranks construct). The instances of one (group, cpu_group, device) share one
        communicator, and those of one communicator, ``model_dim`` and ``max_local_tokens`` the fused sum's buffers
        (collective when first created / allocated).
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
        # Whether the layer has a shared expert (fixed by the constructor's weights).
        self._shared = None
        # Kernels of the most recent forward in launch order (moe_a8w4.last_path; fused: ending with
        # "a8w4_tp_reduce", "a8w4_route_sum_tp" or "a8w4_down_decode[_pq[_shared]|_mixed]_tp"), else then its
        # all-reduce ("custom_all_reduce" / "rccl_all_reduce"); ag_rs: its all-gather ("custom_all_gather" /
        # "rccl_all_gather"), the kernels (fused: ending with "a8w4_tp_reduce", "a8w4_route_sum_tp_rs" or
        # "a8w4_down_decode[_pq[_shared]|_mixed]_tp_rs"), else then its reduce-scatter ("custom_reduce_scatter" /
        # "rccl_all_to_all"). Tests check the dispatch with it.
        self.last_path: tuple[str, ...] = ()
        self.set_weights(
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            shared_w1=shared_w1,
            shared_w1_scale=shared_w1_scale,
            shared_w2=shared_w2,
            shared_w2_scale=shared_w2_scale,
        )
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
        # Batch sizes this instance ran fused (their next forwards wait WAIT_S, not FIRST_WAIT_S).
        self._seen: set = set()

    # -- weights ------------------------------------------------------------------------------------------------
    def set_weights(
        self,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        shared_w1: torch.Tensor | None = None,
        shared_w1_scale: torch.Tensor | None = None,
        shared_w2: torch.Tensor | None = None,
        shared_w2_scale: torch.Tensor | None = None,
    ) -> None:
        """Run the next forwards on another layer's weights of the same shape (converted here, once), with a shared
        expert exactly when the layer was constructed with one. The previous weights are released first: after an
        error the layer holds none until the next successful call."""
        E, H, inter = self.cfg.experts, self.cfg.model_dim, self.cfg.inter_dim
        _check_weight("w1", w1, (E, 2 * inter, H // 2))
        _check_weight("w2", w2, (E, H, inter // 2))
        shared = (shared_w1, shared_w1_scale, shared_w2, shared_w2_scale)
        has_shared = any(t is not None for t in shared)
        if self._shared is not None and has_shared != self._shared:
            raise ValueError(
                f"MegaMoeTPGfx942: the layer was constructed {'with' if self._shared else 'without'} a shared "
                f"expert (routing width topk{' + 1' if self._shared else ''}); set_weights must keep that"
            )
        named = [
            ("w1", w1),
            ("w1_scale", w1_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ] + [
            (n, t) for n, t in zip(_SHARED_NAMES, shared) if isinstance(t, torch.Tensor)
        ]
        for name, t in named:
            if t.device != self.device:
                raise ValueError(
                    f"MegaMoeTPGfx942: {name} is on {t.device}, expected {self.device}"
                )
        with torch.no_grad():
            q1 = _unshuffle_weight(w1, 2 * inter, H // 2)
            c1 = _unshuffle_scale(w1_scale, E, 2 * inter, H // 32)
            q2 = _unshuffle_weight(w2, H, inter // 2)
            c2 = _unshuffle_scale(w2_scale, E, H, inter // 32)
            self._weights = None
            self._weights = moe_a8w4.prepare_moe_a8w4_ptpc_weight(
                q1,
                c1,
                q2,
                c2,
                shared_w1=shared_w1,
                shared_w1_scale=shared_w1_scale,
                shared_w2=shared_w2,
                shared_w2_scale=shared_w2_scale,
            )
        self._shared = has_shared

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
        """x [M, H] BF16, replicated on every rank, with its routing (topk_weights / topk_ids [M, topk], with a shared
        expert [M, topk + 1] ending in its id E; see the module docstring) -> y [M, H] BF16, the whole layer's
        output, identical on every rank (``out`` if given, else a new tensor).

        ag_rs: x_local [M_r, H] BF16 holds this rank's own tokens (the same M_r on every rank, padding rows routed to
        -1) with their routing [M_r, topk] ([M_r, topk + 1]) -> their output [M_r, H] BF16 (``out`` if given).
        """
        if tail is not None:
            raise ValueError("MegaMoeTPGfx942: no fused tail (gfx950 ag_rs only)")
        if self.cfg.comm_mode == "ag_rs":
            return self._forward_ag_rs(x_local, topk_weights, topk_ids, out)
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
        width = cfg.topk + 1 if self._shared else cfg.topk
        if tuple(topk_ids.shape) != (M, width) or tuple(topk_weights.shape) != (
            M,
            width,
        ):
            what = (
                " (topk routed columns, then the shared expert's)"
                if self._shared
                else ""
            )
            raise ValueError(
                f"MegaMoeTPGfx942: topk_ids / topk_weights must be [{M}, {width}]{what}, got "
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
        fuses = self._fuses(M)
        moe_a8w4._flydsl_moe_a8w4_ptpc_tp(
            x_local,
            self._weights,
            topk_weights,
            topk_ids,
            output=part,
            swiglu_limit=self._limit,
            hooks=moe_a8w4._TPHooks(
                reduce=(
                    self._reducer(M, fused)
                    if fuses
                    else self._mid_reducer(M, fused) if self._mid_fuses(M) else None
                ),
                prefill_down=self._prefill_down(M, fused) if fuses else None,
                stream_down=self._stream_down(M, fused) if fuses else None,
                decode_down=(
                    self._decode_down(M, fused) if self._decode_fuses(M) else None
                ),
                decode_max_m=self._decode_tier_max_m(),
            ),
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

    def _forward_ag_rs(self, x_local, topk_weights, topk_ids, out):
        """``forward`` of comm_mode ag_rs: all-gather, the op on every rank's rows, reduce-scatter (fused into the
        Down where the ar mode fuses the sum, module docstring). It checks everything before the first collective, so
        that a rank that raises leaves no peer waiting in one.
        """
        cfg, dev = self.cfg, self.device
        if (
            x_local.dim() != 2
            or x_local.shape[1] != cfg.model_dim
            or x_local.dtype != torch.bfloat16
            or not x_local.is_contiguous()
        ):
            raise ValueError(
                f"MegaMoeTPGfx942: x must be a contiguous bfloat16 [M_r, {cfg.model_dim}] tensor (ag_rs), got "
                f"{x_local.dtype} {tuple(x_local.shape)}"
            )
        m = x_local.shape[0]
        width = cfg.topk + 1 if self._shared else cfg.topk
        if tuple(topk_ids.shape) != (m, width) or tuple(topk_weights.shape) != (
            m,
            width,
        ):
            what = (
                " (topk routed columns, then the shared expert's)"
                if self._shared
                else ""
            )
            raise ValueError(
                f"MegaMoeTPGfx942: topk_ids / topk_weights must be [{m}, {width}]{what}, got "
                f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
            )
        if (
            topk_ids.dtype not in (torch.int32, torch.int64)
            or not topk_weights.is_floating_point()
        ):
            raise TypeError(
                f"MegaMoeTPGfx942: topk_ids must be int32 / int64 and topk_weights floating, got "
                f"{topk_ids.dtype} / {topk_weights.dtype}"
            )
        for name, t in (
            ("x", x_local),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
        ):
            if t.device != dev:
                raise ValueError(
                    f"MegaMoeTPGfx942: {name} is on {t.device}, the layer on {dev}"
                )
        if m > cfg.max_local_tokens:
            raise ValueError(
                f"MegaMoeTPGfx942: M_r {m} exceeds max_local_tokens {cfg.max_local_tokens}"
            )
        if out is not None and (
            out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, cfg.model_dim)
            or not out.is_contiguous()
            or out.device != dev
        ):
            raise ValueError(
                f"MegaMoeTPGfx942: out must be a contiguous bfloat16 [{m}, {cfg.model_dim}] tensor on {dev}"
            )
        comm = self._comm
        if moe_a8w4.DEBUG_CHECKS and not torch.cuda.is_current_stream_capturing():
            counts = [None] * cfg.world_size
            dist.all_gather_object(counts, m, group=comm.cpu_group)
            if len(set(counts)) != 1:
                raise ValueError(
                    f"MegaMoeTPGfx942: ag_rs needs the same M_r on every rank, got {counts}"
                )
        if out is None:
            out = torch.empty((m, cfg.model_dim), dtype=torch.bfloat16, device=dev)
            given = False
        else:
            given = True
        if m == 0:
            self.last_path = ()
            return out
        xs, ids, tw = comm.all_gather(
            x_local,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
        )
        M = cfg.world_size * m
        part = torch.empty_like(xs)
        # Fused: the Down kernels store this rank's tokens' sums into out, the decode Down into part's rows of them.
        fused, decoded = [], []
        fuses = self._fuses(M)
        moe_a8w4._flydsl_moe_a8w4_ptpc_tp(
            xs,
            self._weights,
            tw,
            ids,
            output=part,
            swiglu_limit=self._limit,
            hooks=moe_a8w4._TPHooks(
                reduce=(
                    self._reducer(M, fused, out)
                    if fuses
                    else (
                        self._mid_reducer(M, fused, out) if self._mid_fuses(M) else None
                    )
                ),
                prefill_down=self._prefill_down(M, fused, out) if fuses else None,
                stream_down=self._stream_down(M, fused, out) if fuses else None,
                decode_down=(
                    self._decode_down(M, decoded, "dp")
                    if self._decode_fuses(M)
                    else None
                ),
                decode_max_m=self._decode_tier_max_m(),
            ),
        )
        if fused or decoded:
            self._seen.add(M)
            self.last_path = (comm.last_ag, *moe_a8w4.last_path)
            if not decoded:
                return out
            mine = part[cfg.rank * m : (cfg.rank + 1) * m]
            if not given:
                return mine
            out.copy_(mine)
            return out
        comm.reduce_scatter_ordered(part, out)
        self.last_path = (comm.last_ag, *moe_a8w4.last_path, comm.last_rs)
        return out

    def _balanced_batch(self, start, m):
        """Rows [start, start + m) of the warm-up batch: x 0, token i's slot j routed to expert
        (i * topk + j) % experts (distinct within a token), the shared expert E last with weight 1.
        """
        cfg, dev = self.cfg, self.device
        x = torch.zeros((m, cfg.model_dim), dtype=torch.bfloat16, device=dev)
        ids = (
            torch.arange(start, start + m, device=dev)[:, None] * cfg.topk
            + torch.arange(cfg.topk, device=dev)
        ) % cfg.experts
        tw = torch.full((m, cfg.topk), 1.0 / cfg.topk, dtype=torch.float32, device=dev)
        if self._shared:
            ids = torch.cat([ids, torch.full_like(ids[:, :1], cfg.experts)], dim=1)
            tw = torch.cat([tw, torch.ones_like(tw[:, :1])], dim=1)
        return x, tw, ids.to(torch.int32)

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

    def _decode_tier_max_m(self):
        """The op's decode tier limit in this layer (``DECODE_TIER_MAX_M``), None for the op's own."""
        return DECODE_TIER_MAX_M.get((self.cfg.world_size, self.cfg.model_dim))

    def _mid_fuses(self, M):
        """Whether an M-row batch below the fused Down (``_fuses``) sums across the ranks in its last reduction
        (sum_tp) when the op takes the streaming or prefill path: every rank must own a unit (sum_tp, rule 2).
        """
        ws = self._ws
        return (
            ws is not None
            and M <= ws.mid_m
            and M * (self.cfg.model_dim // 512) >= self.cfg.world_size
        )

    def _mid_reducer(self, M, fused, home_out=None):
        """The ``reduce`` hook (``moe_a8w4._TPHooks``) of the batches of ``_mid_fuses``: sum_tp in place of the op's
        route sum and the all-reduce (appends to ``fused``); after the atomic streaming Down (no route sum, ``loc_ids``
        None) nothing: the output is this rank's partial, all-reduced by ``forward`` (``MID_WORLDS``). ``home_out``
        (ag_rs, [M / world_size, H]): the tokens' home ranks own their sums and store them there only (sum_tp's
        ``DEST`` "dp", ``a8w4_route_sum_tp_rs``), the reduce-scatter.
        """
        cfg, ws = self.cfg, self._ws
        limit = self._limit_of(M)
        nwg = -(-M * (cfg.model_dim // 512) // 4)
        dest = "ar2" if home_out is None else "dp"

        def reduce(output, loc_ids, routes):
            if loc_ids is None:
                return
            launch = sum_tp.compile_sum_tp(
                WORLD=cfg.world_size,
                H=cfg.model_dim,
                TOPK=loc_ids.shape[1],
                UNITS=ws.munits,
                DEST=dest,
            )
            moe_a8w4._launch(
                sum_tp.kernel_name(dest),
                launch,
                loc_ids,
                routes,
                output if home_out is None else home_out,
                ws.mtab,
                ws.mparity,
                self._err,
                M,
                cfg.rank,
                limit,
                nwg,
            )
            fused.append(M)

        return reduce

    def _decode_down(self, M, fused, dest="ar"):
        """The ``decode_down`` hook (``moe_a8w4._TPHooks``): decode_tp in place of the Down decode kernel and the
        all-reduce (appends to ``fused``); from DECODE_TP_PREQUANT_MIN_M rows on the slot-pair kernel, on rows it
        quantizes itself if the op passes the single-slot kernel's inputs. A folded shared expert takes the slot-pair
        kernel that adds it (``_shared``), the mixed decode its mixed Down (``_mixed``). ``dest``: decode_tp's
        ``DEST`` ("dp", ag_rs: the kernel writes only this rank's tokens' rows of the op's output, the reduce-scatter).
        """
        cfg, ws = self.cfg, self._ws
        limit = self._limit_of(M)
        H = cfg.model_dim

        def down(
            paired, mid, mid_scale, topk_ids, topk_weight, weights, output, *, shared
        ):
            cols = topk_ids.shape[1]
            tail = (output, M, ws.dtab, ws.parity, self._err, cfg.rank, limit)
            if shared == "mixed":
                launch = decode_tp.compile_down_decode_mixed_tp(
                    N=H,
                    K=weights.inter_padded,
                    KP=weights.shared_w2.shape[1],
                    TOPK=cols,
                    E=weights.experts,
                    WORLD=cfg.world_size,
                    MAX_M=DECODE_TP_MAX_M,
                    DEST=dest,
                )
                moe_a8w4._launch(
                    decode_tp.kernel_name("a8w4_down_decode_mixed", dest),
                    launch,
                    mid,
                    topk_ids,
                    topk_weight,
                    weights.w2,
                    weights.w2_delta,
                    weights.w2_scale,
                    weights.shared_w2,
                    weights.shared_w2_scale,
                    *tail,
                )
                fused.append(M)
                return
            fold = shared == "fold"
            topk, I = cols - fold, weights.inter_padded
            if (
                not paired
                and M >= DECODE_TP_PREQUANT_MIN_M
                and moe_a8w4._slot_pair_offsets_fit(weights)
                and moe_a8w4._decode_fits(H, I, topk, True)
            ):
                mid, mid_scale = moe_a8w4._quantize_decode(mid, M * topk, I, False)
                paired = True
            launch = decode_tp.compile_down_decode_tp(
                N=H,
                K=I,
                TOPK=topk,
                WAVES=moe_a8w4._decode_waves(topk, paired),
                WORLD=cfg.world_size,
                MAX_M=DECODE_TP_MAX_M,
                PAIRED=paired,
                SHARED=fold,
                ID_COLS=cols,
                DEST=dest,
            )
            moe_a8w4._launch(
                decode_tp.kernel_name(
                    "a8w4_down_decode"
                    + ("_pq" if paired else "")
                    + ("_shared" if fold else ""),
                    dest,
                ),
                launch,
                mid,
                mid_scale,
                topk_ids,
                topk_weight,
                weights.w2,
                weights.w2_delta,
                weights.w2_scale,
                *tail,
            )
            fused.append(M)

        return down

    def _width(self):
        """Columns of the routing (and of ``loc_ids``): the routed experts, then a shared expert's."""
        return self.cfg.topk + (1 if self._shared else 0)

    def _limit_of(self, M):
        return _wait_limit(WAIT_S if M in self._seen else FIRST_WAIT_S)

    def _gather(self, M, spec, output, limit):
        """``a8w4_tp_reduce`` after a streamed Down (``down_tp.compile_tp_gather``): copy the rows the other ranks
        summed into ``output`` (``spec.AG`` = 0: only wait until the owners have read this rank's partials).
        """
        cfg, ws = self.cfg, self._ws
        if spec.OWNER == down_tp.OWN_HOME:
            tiles = cfg.world_size * -(-(M // cfg.world_size) // spec.TT)
        else:
            tiles = -(-M // spec.TT)
        items = spec.NSPLIT * tiles
        per_wg = GATHER_ITEMS if items >= GATHER_FROM else 1
        gather = down_tp.compile_tp_gather(
            WORLD=cfg.world_size,
            H=cfg.model_dim,
            TT=spec.TT,
            NSPLIT=spec.NSPLIT,
            DRY=spec.DRY,
            G=per_wg,
            OWNER=spec.OWNER,
            AG=spec.AG,
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

    def _prefill_down(self, M, fused, home_out=None):
        """The ``prefill_down`` hook (``moe_a8w4._TPHooks``): the prefill Down with the cross-rank sum streamed out
        of it (down_tp), then ``a8w4_tp_reduce`` copying the other ranks' rows (appends to ``fused``); None if the
        batch does not stream (``STREAM_DOWN``, ``STREAM_SPEC``). ``home_out`` (ag_rs, [M / world_size, H]): the
        owners are the tokens' home ranks and store the sums of their own tokens there only (``down_tp.OWN_HOME``,
        ``AG`` = 0), the reduce-scatter."""
        cfg, ws = self.cfg, self._ws
        over = {} if home_out is None else _HOME_OWNERS
        spec = _stream_spec(cfg.world_size, cfg.model_dim, self._width(), M, **over)
        if not STREAM_DOWN or spec is None:
            return None
        limit = self._limit_of(M)
        name = "_tp" if home_out is None else "_tp_rs"

        def down(
            mid,
            mid_scale,
            routing,
            loc_ids,
            routes,
            output,
            weights,
            grid,
            *,
            topk,
            block_m,
            xcd_chunk,
        ):
            # The single-GPU Down's activation staging (moe_a8w4._prefill_down) at this hidden split.
            blocks = min(
                M * topk, weights.experts * -(-M * topk // (weights.experts * block_m))
            )
            launch = down_tp.compile_down_tp(
                N=cfg.model_dim,
                K=weights.inter_padded,
                TOPK=topk,
                LOC_TOPK=loc_ids.shape[1],
                BLOCK_TILE_SIZE_M=block_m,
                E=weights.experts,
                TWO_PASS=blocks * spec.NSPLIT > moe_a8w4.DOWN_TWO_PASS_MIN_WGS,
                XCD_CHUNK=xcd_chunk,
                tp=spec,
            )
            dst = output if home_out is None else home_out
            moe_a8w4._launch(
                f"a8w4_down_prefill_bm{block_m}{name}",
                launch,
                mid,
                weights.w2,
                weights.w2_delta,
                routes,
                *routing,
                weights.w2_scale,
                mid_scale,
                loc_ids,
                dst,
                ws.ctr,
                ws.tab2,
                ws.epoch2,
                self._err,
                M,
                cfg.rank,
                ws.tpq,
                limit,
                grid,
            )
            self._gather(M, spec, dst, limit)
            fused.append(M)

        return down

    def _stream_down(self, M, fused, home_out=None):
        """The ``stream_down`` hook (``moe_a8w4._TPHooks``): the streaming Down kernel with the cross-rank sum
        streamed out of it (down_tp, four waves of ``STREAM_TP_R`` row blocks), then ``a8w4_tp_reduce`` copying the
        other ranks' rows (appends to ``fused``); None if the batch does not stream (``STREAM_DOWN``,
        ``STREAM_TIER``, ``STREAM_TIER_SPEC``). ``home_out``: as ``_prefill_down``'s (``STREAM_TIER_SPEC_DP``).
        """
        cfg, ws = self.cfg, self._ws
        H, width = cfg.model_dim, self._width()
        table, over, name = STREAM_TIER_SPEC, {}, "_tp"
        if home_out is not None:
            table, over, name = STREAM_TIER_SPEC_DP, _HOME_OWNERS, "_tp_rs"
        if not (STREAM_DOWN and STREAM_TIER) or (
            _stream_spec(cfg.world_size, H, width, M, table, **over) is None
        ):
            return None
        limit = self._limit_of(M)

        def down(
            mid,
            mid_scale,
            routing,
            loc_ids,
            routes,
            output,
            weights,
            block_m,
            task_num,
            *,
            topk,
        ):
            K = weights.inter_padded
            # The atomic Down (no routes buffer) adds the routes onto output: the owners read its rows (COPY).
            atomic = routes is None
            tp = _stream_spec(
                cfg.world_size, H, width, M, table, COPY=int(atomic), **over
            )
            launch = down_tp.compile_down_stream_tp(
                N=H,
                K=K,
                TOPK=topk,
                BLOCK_M=block_m,
                E=weights.experts,
                ATOMIC=atomic,
                R=STREAM_TP_R,
                SHARE=moe_a8w4._stream_down_shared(H, K, block_m),
                PREFETCH=(
                    moe_a8w4.STREAM_DOWN_PREFETCH
                    if block_m == 64
                    else moe_a8w4.STREAM_DOWN_PREFETCH_SMALL
                ),
                LDS_ACTS=bool(moe_a8w4.STREAM_DOWN_LDS_ACTS)
                and block_m == 32
                and K % 256 == 0,
                W_AUX=(
                    moe_a8w4.STREAM_DOWN_W_AUX_H5120
                    if H >= 5120
                    else moe_a8w4.STREAM_DOWN_W_AUX
                ),
                tp=tp,
            )
            dst = output if home_out is None else home_out
            moe_a8w4._launch(
                f"a8w4_down_stream_bm{block_m}{name}",
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
                dst,
                ws.ctr,
                ws.tab2,
                ws.epoch2,
                self._err,
                cfg.rank,
                ws.tpq,
                limit,
            )
            self._gather(M, tp, dst, limit)
            fused.append(M)

        return down

    def _reducer(self, M, fused, home_out=None):
        """The ``reduce`` hook (``moe_a8w4._TPHooks``): tp_reduce in place of the op's last reduction (appends to
        ``fused``). ``home_out``: as ``_prefill_down``'s (``tp_reduce.OWN_HOME``, ``AG`` = 0, ``a8w4_tp_reduce_rs``).
        """
        cfg, ws = self.cfg, self._ws
        world = cfg.world_size
        tt = _tile_tokens(M, cfg.model_dim)
        limit = self._limit_of(M)
        if home_out is None:
            over, name, tiles = {}, "a8w4_tp_reduce", -(-M // tt)
        else:
            over = {"OWNER": tp_reduce.OWN_HOME, "AG": 0}
            name, tiles = "a8w4_tp_reduce_rs", world * -(-(M // world) // tt)

        def reduce(output, loc_ids, routes):
            launch = tp_reduce.compile_tp_reduce(
                WORLD=world,
                H=cfg.model_dim,
                TOPK=0 if loc_ids is None else loc_ids.shape[1],
                TT=tt,
                **over,
            )
            moe_a8w4._launch(
                name,
                launch,
                ws.epoch if loc_ids is None else loc_ids,
                output if routes is None else routes,
                output if home_out is None else home_out,
                ws.tab,
                ws.epoch,
                self._err,
                M,
                cfg.rank,
                ws.slots,
                limit,
                tiles,
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
        replicated batch M with ceil(M / world_size) in ``local_tokens``; ag_rs: every M_r in ``local_tokens``), so
        that serving and CUDA graph capture compile nothing; then check every rank's watchdog: on a timeout
        ``reset()`` and retry once, else raise.
        """
        if tail or tail_bf16:
            raise ValueError("MegaMoeTPGfx942: no fused tail (gfx950 ag_rs only)")
        if not warmup:
            return
        cfg = self.cfg
        ts = [local_tokens] if isinstance(local_tokens, int) else list(local_tokens)
        tp = cfg.world_size
        if cfg.comm_mode == "ag_rs":
            # this rank's rows of the warm-up batch of tp * t rows
            batches = [(cfg.rank * t, t) for t in sorted({t for t in ts if t > 0})]
        else:
            batches = [
                (0, m)
                for m in sorted(
                    {
                        m
                        for t in ts
                        for m in range((t - 1) * tp + 1, t * tp + 1)
                        if m > 0
                    }
                )
            ]
        for _ in range(2):
            for start, m in batches:
                self.forward(*self._balanced_batch(start, m))
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
        ``decode_tp.ERR_TIMEOUT`` = 2, ``sum_tp.ERR_TIMEOUT`` = 4) as a device tensor (no sync).
        """
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
