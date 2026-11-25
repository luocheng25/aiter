# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer for gfx942 with the interface of ``MegaMoeTP`` (``comm_mode="ar"`` only).

Experts are replicated over the TP group and ``inter_dim`` is sharded. Every rank gets the same input rows and
routing; ``forward`` runs this rank's slice with the A8W4 two-stage kernels (``moe_a8w4.fused_moe_a8w4``: FP8
per-token activations, MXFP4 weights) and all-reduces the BF16 partial sums: aiter ``CustomAllreduce`` when its
``should_custom_ar`` accepts the tensor, RCCL (``torch.distributed.all_reduce`` on ``group``) otherwise. The
outputs of all ranks are bitwise identical.

Weights come in the ``MegaMoeTP`` layout (MXFP4, ``shuffle_weight(16, 16)`` + ``e8m0_shuffle``) and are converted
once (constructor / ``set_weights``) to the A8W4 layout; ``inter_dim`` is zero-padded to a multiple of 64.

Contract (as ``MegaMoeTP``):

* ``forward`` is a collective: every rank calls it the same number of times, in the same order, with the same
  input and routing.
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles and runs every kernel these batches launch.
* No kernel waits on a peer: ``error_flag()`` is always zero, ``reset()`` has no cross-rank state to restart.
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
        instances of one (group, cpu_group, device) share one communicator."""
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
        # Kernels of the most recent forward in launch order, then its all-reduce ("custom_all_reduce" /
        # "rccl_all_reduce"); tests check the dispatch with it.
        self.last_path: tuple[str, ...] = ()
        self.set_weights(w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale)
        self._comm_key, self._comm = _acquire_comm(group, cpu_group, device)
        self._err = torch.zeros(1, dtype=torch.int32, device=device)

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
        moe_a8w4.fused_moe_a8w4(
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

    def prepare(
        self,
        local_tokens,
        tail: bool = False,
        tail_bf16: bool = False,
        warmup: bool = True,
    ) -> None:
        """Collective: compile and run once every kernel of the batches of these local token counts (ar: every
        replicated batch M with ceil(M / world_size) in ``local_tokens``), so that serving and CUDA graph capture
        compile nothing."""
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
                (m, cfg.topk), 1.0 / cfg.topk, dtype=torch.float32, device=self.device
            )
            self.forward(x, tw, ids.to(torch.int32))
        torch.cuda.synchronize(self.device)

    @property
    def max_local_tokens(self) -> int:
        return self.cfg.max_local_tokens

    def tail_ok(self, local_tokens: int) -> bool:
        """forward(tail=...) runs for this local token count: never (no fused tail)."""
        return False

    # -- errors / lifetime --------------------------------------------------------------------------------------
    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bits as a device tensor: always zero (no kernel waits on a peer)."""
        return self._err

    def poll_errors(self) -> int:
        return 0

    def check_errors(self) -> None:
        """Raise RuntimeError if a wait gave up: never."""

    def clear_errors(self) -> None:
        pass

    def reset(self) -> None:
        """Collective in ``MegaMoeTP``; nothing to restart here (no cross-rank state)."""

    def close(self) -> None:
        """Collective: release the weights and this instance's share of the communicator (unusable afterwards)."""
        self._weights = None
        if self._comm is not None:
            _release_comm(self._comm_key)
            self._comm = None
