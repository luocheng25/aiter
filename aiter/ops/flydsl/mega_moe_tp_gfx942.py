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
  their owner rank over hipIpc symmetric memory, f32 sum in rank order, all-gather of the BF16 rows);
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

import math

import torch
import torch.distributed as dist

from aiter import ActivationType
from aiter.jit.utils.chip_info import get_gfx_runtime

from . import moe_a8w4
from .kernels.mega_moe_tp_gfx942 import decode_tp, tp_reduce
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
    table ``tab`` and this rank's ``epoch``; decode_tp: the symmetric ``dinbox`` (uncached, 0xFF), its address table
    ``dtab`` and this rank's ``parity``. Creation and ``close`` are collective over the communicator's gloo group.
    """

    def __init__(self, comm, rank, world, H, m_max):
        from .communication_ops_host_utils import IpcSymmetricHeap

        # The largest batch of each tile size bounds the tiles and the inbox rows of that size.
        tiers = [(m, _tile_tokens(m, H)) for m in {min(m_max, 1023), m_max}]
        self.slots = max(-(-m // tt) for m, tt in tiers)
        # Rows per owner rank: an owner's share of the tiles (tile i -> rank i % world), in tokens.
        rps = max((-(-m // tt) + world - 1) // world * tt for m, tt in tiers)
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
            self.inbox = self.heap.create_tensor(((world - 1) * rps, H), torch.bfloat16)
            self.flags = self.heap.create_tensor(
                (self.slots * (world + 1),), torch.int32
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
        self.dtab = self.heap.p2p_table(self.dinbox).view(torch.int32)
        self.parity = torch.zeros(
            DECODE_TP_MAX_M * (H // 16), dtype=torch.int32, device=comm.device
        )

    def reset(self):
        """Collective: zero the flags, epochs and parities and refill the decode inbox once no kernel of any rank
        runs, before any rank launches again."""
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)
        self.flags.zero_()
        self.epoch.zero_()
        self.dinbox.fill_(decode_tp.SENTINEL)
        self.parity.zero_()
        torch.cuda.synchronize(self.comm.device)
        dist.barrier(group=self.comm.cpu_group)

    def close(self):
        """Collective (``IpcSymmetricHeap.close``)."""
        self.heap.close()
        self.landing = self.inbox = self.flags = self.tab = self.epoch = None
        self.dinbox = self.dtab = self.parity = None


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
        # Batch sizes this instance ran fused (their next forwards wait WAIT_S, not FIRST_WAIT_S).
        self._seen: set[int] = set()

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
