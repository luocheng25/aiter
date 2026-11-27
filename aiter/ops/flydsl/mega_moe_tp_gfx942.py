# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer for gfx942 with the interface of ``MegaMoeTP`` (``comm_mode="ar"`` and ``"ag_rs"``).

Experts are replicated over the TP group and ``inter_dim`` is sharded. ``comm_mode="ar"``: every rank gets the same
input rows and routing; ``forward`` runs this rank's slice with the single-GPU A8W4 op
(``moe_a8w4.flydsl_moe_a8w4_ptpc``: FP8 per-token activations, MXFP4 routed experts, an optional FP8 shared expert)
into a BF16 partial output and sums the partials over the ranks with one all-reduce: aiter's custom all-reduce
(``CustomAllreduce``) below ``AR_RCCL_MIN_BYTES[world_size]`` bytes where its ``should_custom_ar`` accepts the
tensor, else RCCL (``torch.distributed.all_reduce`` on ``group``). The outputs of all ranks are bitwise identical.

``comm_mode="ag_rs"`` (data-parallel attention in front, sequence-parallel in and out): every rank passes its own
M_r tokens (the same M_r on every rank) and gets their output. ``forward`` gathers every rank's rows and routing
(all-gather; rank r's rows at [r * M_r, (r + 1) * M_r)), runs this rank's slice of the layer on all world_size * M_r
rows with the same op and sends each rank the sum of the partials of its rows (reduce-scatter). The all-gather is
one ``custom_all_gatherv`` of x and the routing below ``AG_RCCL_MIN_BYTES[world_size]`` bytes of x where it takes
them, else RCCL for x (``all_gather_into_tensor``) and the custom gather (RCCL where it declines) for the routing;
the reduce-scatter is ``custom_reduce_scatter`` below ``RS_RCCL_MIN_BYTES[world_size]`` bytes of partials where its
``should_custom_rs`` accepts them, else RCCL (``reduce_scatter_tensor``). The custom reduce-scatter sums each element
in f32 in rank order 0, 1, ... and rounds to BF16 once; RCCL's ring adds in BF16 hop by hop in an order fixed by
the size and the ranks. Both modes are deterministic: the same inputs give bitwise the same outputs.

This version fuses no communication into the kernels in either mode: the fused cross-rank sums of the MXFP4-only
version (Down kernels that sum across the ranks, the token-tile reduce-scatter, SDMA chunks) return on top of the
same op later. What only existed for them is trivial here: ``prepare`` only compiles and warms up, the watchdog bits
stay 0 and ``reset`` only clears them.

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
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles and runs every kernel these batches launch.
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
every rank when the ranks' M_r differ (compared over the gloo group).
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx_runtime

from . import moe_a8w4
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
        communicator (collective when first created).
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
        # Kernels of the most recent forward in launch order (moe_a8w4.last_path), then its all-reduce
        # ("custom_all_reduce" / "rccl_all_reduce"); ag_rs: its all-gather ("custom_all_gather" / "rccl_all_gather"),
        # the kernels, its reduce-scatter ("custom_reduce_scatter" / "rccl_reduce_scatter"). Tests check the dispatch
        # with it.
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
        # Watchdog bits of the MegaMoeTP interface: no kernel of this version waits on a peer, so they stay 0.
        self._err = torch.zeros(1, dtype=torch.int32, device=device)

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
        moe_a8w4.flydsl_moe_a8w4_ptpc(
            x_local,
            self._weights,
            topk_weights,
            topk_ids,
            output=part,
            swiglu_limit=self._limit,
        )
        y = self._comm.all_reduce(part)
        self.last_path = (*moe_a8w4.last_path, self._comm.last)
        if out is not None and y is not out:
            out.copy_(y)
            return out
        return y

    __call__ = forward

    def _forward_ag_rs(self, x_local, topk_weights, topk_ids, out):
        """``forward`` of comm_mode ag_rs: all-gather, the op on every rank's rows, reduce-scatter. It checks
        everything before the first collective, so that a rank that raises leaves no peer waiting in one.
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
        if m == 0:
            self.last_path = ()
            return out
        xs, ids, tw = comm.all_gather(
            x_local,
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
        )
        part = torch.empty_like(xs)
        moe_a8w4.flydsl_moe_a8w4_ptpc(
            xs, self._weights, tw, ids, output=part, swiglu_limit=self._limit
        )
        comm.reduce_scatter(part, out)
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

    def prepare(
        self,
        local_tokens,
        tail: bool = False,
        tail_bf16: bool = False,
        warmup: bool = True,
    ) -> None:
        """Collective: compile and run once every kernel of the batches of these local token counts (ar: every
        replicated batch M with ceil(M / world_size) in ``local_tokens``; ag_rs: every M_r in ``local_tokens``), so
        that serving and CUDA graph capture compile nothing. (No watchdog to check in this version.)
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
            for t in sorted({t for t in ts if t > 0}):
                self.forward(*self._balanced_batch(cfg.rank * t, t))
            torch.cuda.synchronize(self.device)
            return
        rows = sorted(
            {m for t in ts for m in range((t - 1) * tp + 1, t * tp + 1) if m > 0}
        )
        for m in rows:
            self.forward(*self._balanced_batch(0, m))
        torch.cuda.synchronize(self.device)

    @property
    def max_local_tokens(self) -> int:
        return self.cfg.max_local_tokens

    def tail_ok(self, local_tokens: int) -> bool:
        """forward(tail=...) runs for this local token count: never (no fused tail)."""
        return False

    # -- errors / lifetime --------------------------------------------------------------------------------------
    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bits of the ``MegaMoeTP`` interface as a device tensor (no sync); always 0 here: no
        kernel of this version waits on a peer (the fused sums of later versions set them).
        """
        return self._err

    def poll_errors(self) -> int:
        """Nonzero if a wait inside a launch gave up (a peer never arrived); synchronizes. Always 0 here."""
        return int(self._err.item())

    def check_errors(self) -> None:
        """Raise RuntimeError (and clear) if a wait gave up since the last check (never here)."""
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
        """Collective by contract (as ``MegaMoeTP``): restart the cross-rank state and clear this instance's errors.
        Without fused kernels there is no cross-rank state: it only clears the errors.
        """
        self.clear_errors()

    def close(self) -> None:
        """Collective: release the weights and this instance's share of the communicator (unusable afterwards)."""
        self._weights = None
        if self._comm is not None:
            _release_comm(self._comm_key)
            self._comm = None
