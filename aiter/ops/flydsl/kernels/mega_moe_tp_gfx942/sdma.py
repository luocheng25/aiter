# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-driven SDMA copies on gfx942: user-mode SDMA queues whose packets a kernel writes and submits.

Host (``acquire(device)``: one ``SdmaQueues`` per device, shared and refcounted): user-mode SDMA queues created
with the KFD ioctl ``AMDKFD_IOC_CREATE_QUEUE`` (type ``SDMA_BY_ENG_ID``, one queue on each engine of
``engines()``). Their ring and 4 KiB pointer page (write pointer at ``WPTR``, read pointer at ``RPTR``, the
completion word at ``DONE``) come from ROCr's coarse-grained VRAM pool of the device; their doorbell is a word of
the doorbell page ROCr's thunk already mapped from ``/dev/kfd`` for its own queues (so the device must have run a
kernel first). The doorbell address works from the GPU as well. ``desc`` (int64 [queues, 4] on the device: ring,
page, doorbell, 0) or its rows in another order (``ordered``) is what kernels read; ``recreate`` rewrites them in
place (CUDA graphs stay valid). No library beyond the HIP / HSA runtime torch already loaded.

Device (FlyDSL helpers, one wave per queue, wave-uniform arguments): ``Ring(desc, q, lane)`` loads queue ``q``;
``place`` gives the ring offset of a submission (NOP padding up to the ring end when it would wrap, ``Ring.pad``);
the wave stores the packets (``poll``, ``copy2d``, ``fence``, NOPs) with ``Ring.store``; ``Ring.commit`` publishes
the write pointer and rings the doorbell. A copy gated by ``poll`` on a flag runs when another workgroup or rank
stores the flag value: "copy these (src, dst, bytes) segments once that flag is set". The last packet of a
submission is ``fence(ring.done_address, low 32 bits of the new write pointer)``: the queue is idle (``Ring.done``
equals the write pointer) once the copies before it are complete in memory (a later kernel's loads see the data).

Rules:

1. One writer at a time: kernels using the queues of a device must not overlap in time (one stream: the workspaces
   of ``MegaMoeTPGfx942`` share them). A sequence of submissions (one kernel's, or consecutive kernels' as the token
   chunks of down_tp rule 12) starts on an idle queue, and its last kernel waits for the last fence before it
   ends. The write pointer is a byte count that only grows; the ring offset is its low bits. A submission is a
   multiple of 64 bytes, and a sequence's submissions together are at most ``RING_BYTES // 2`` (with the padding
   they never fill the ring, whose full and empty states would look alike).
2. Release to the engine: packet stores (system scope), ``s_waitcnt vmcnt(0)``, the write pointer store (system
   scope), ``s_waitcnt vmcnt(0)``, the doorbell store (64-bit write pointer).
3. The engine reads and writes through this process's GPU virtual memory: any address a kernel of this device may
   access (local memory, peers' memory opened with hipIpc) is a valid copy source / destination / flag.
"""

import ctypes
import fcntl
import os
import struct

import torch

import flydsl.expr as fx
from flydsl.expr import rocdl

from ..moe_a8w4.stream import _DROP, _load, _pack64, _rsrc, _uniform
from .tp_reduce import _loop
from .tp_reduce import _store as _raw_store

# Ring of each queue in bytes (power of two) and the pointer page's layout (bytes).
RING_BYTES = 1 << 16
PAGE_BYTES = 4096
WPTR, RPTR, DONE = 0, 64, 128
# Cache policy of the stores the engine reads (ring, write pointer, doorbell): system scope (sc0 | sc1).
SYS = 17
NOP = 0
# Engines of the queues (xGMI SDMA engines are 2 .. 7; even ones only: on MI308X the odd ids name fused-off engines
# and a doorbell on them hangs, MORI anvil.cpp). ENGINES_ENV ("4,2,6": queue q on the q-th engine) reorders them: with
# every GPU pulling from all its peers at once, the receive bandwidth of the slowest GPU depends on the order and the
# best order on the set of GPUs (tools/sdma_probe.py phase 6: 88 .. 134 GB/s; KFD's recommended engines do not help).
# Pushes (down_tp rule 12) put each peer on a queue of KFD's recommended engine (push_order) unless
# ENGINES_ENV is set.
XGMI_ENGINES = (2, 4, 6)
ENGINES_ENV = "AITER_MEGA_TP_SDMA_ENGINES"
# Bytes of a queue's entry in ``desc``.
DESC_BYTES = 32


# ---------------------------------------------------------------------------------------------------------------
# Packets (SDMA 4.4, dwords). Addresses are fx.Int64 or ints; values fx.Int32 / ints.
# ---------------------------------------------------------------------------------------------------------------
def _i32(v):
    """A Python int as a signed 32-bit int."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


def _lo(a):
    return _i32(a) if isinstance(a, int) else fx.Int32(fx.Int64(a))


def _hi(a):
    return _i32(a >> 32) if isinstance(a, int) else fx.Int32(fx.Int64(a) >> 32)


def poll(addr, value, mask=0xFFFFFFFF, func=3, interval=4):
    """POLL_REGMEM on memory: wait until ``(*addr & mask) func value`` (3: ==, 5: >=), polling every ``interval``
    (16-clock units), no retry limit. 6 dwords."""
    header = 8 | (func << 28) | (1 << 31)
    return [_i32(header), _lo(addr), _hi(addr), value, _i32(mask), interval | (0xFFF << 16)]


def copy2d(src, dst, pitch, width, rows):
    """COPY_LINEAR_SUBWIN_LARGE: ``rows`` (>= 1) rows of ``width`` bytes, ``pitch`` bytes apart in both ``src`` and
    ``dst``. 20 dwords."""
    pm1 = fx.Int32(pitch) - 1
    return [1 | (36 << 8), _lo(src), _hi(src), 0, 0, 0, pm1, 0, 0, _lo(dst), _hi(dst), 0, 0, 0, pm1, 0, 0,
            fx.Int32(width) - 1, fx.Int32(rows) - 1, 0]


def fence(addr, value):
    """FENCE: store ``value`` at ``addr`` once the previous packets are complete. 4 dwords."""
    return [5, _lo(addr), _hi(addr), value]


def stamp(addr):
    """TIMESTAMP (GET_GLOBAL_TIMESTAMP): stores the 64-bit global timestamp (the ``s_memrealtime`` clock) at ``addr``
    (64-byte aligned) once the previous packets are processed (timing analysis). 3 dwords."""
    return [13 | (2 << 8), _lo(addr), _hi(addr)]


def select(cond, words, other):
    """Per-dword ``cond ? words : other`` (lists of equal length)."""
    assert len(words) == len(other)
    return [cond.select(fx.Int32(a), fx.Int32(b)) for a, b in zip(words, other)]


def nops(n):
    return [NOP] * n


def place(wptr, nbytes):
    """(ring offset, padding bytes, write pointer after) of a submission of ``nbytes`` (a multiple of 64, at most
    ``RING_BYTES // 2``) after write pointer ``wptr``: it starts at the ring start, after NOPs up to the ring end,
    when it would wrap."""
    start = fx.Int32(wptr) & (RING_BYTES - 1)
    wraps = (start + nbytes) > RING_BYTES
    pad = wraps.select(RING_BYTES - start, fx.Int32(0))
    return wraps.select(fx.Int32(0), start), pad, wptr + fx.Int64(pad + nbytes)


# ---------------------------------------------------------------------------------------------------------------
# Device-side queue access
# ---------------------------------------------------------------------------------------------------------------
def _store(rsrc, words, voffset):
    _raw_store(rsrc, [fx.Uint32(fx.Int32(w)) for w in words], voffset, SYS)


def _w64(v):
    v = fx.Int64(v)
    return [fx.Int32(v), fx.Int32(v >> 32)]


def _ld64(rsrc, offset, aux=0):
    w = _load(rsrc, offset, 0, 2, aux=aux)
    return _pack64(fx.Uint32(_uniform(w[0])), fx.Uint32(_uniform(w[1])))


class Ring:
    """Queue ``q`` (wave-uniform) of the descriptor table at ``desc`` (int64 address of ``SdmaQueues.desc``) seen
    by one wave; ``lane`` is the wave's lane id."""

    def __init__(self, desc, q, lane):
        table = _rsrc(desc, len(XGMI_ENGINES) * DESC_BYTES)
        self.ring_address = _ld64(table, q * DESC_BYTES)
        self.page_address = _ld64(table, q * DESC_BYTES + 8)
        self.ring = _rsrc(self.ring_address, RING_BYTES)
        self.page = _rsrc(self.page_address, PAGE_BYTES)
        self.bell = _rsrc(_ld64(table, q * DESC_BYTES + 16), 8)
        self.done_address = self.page_address + DONE
        self.lane = lane

    def wptr(self):
        """The queue's write pointer (bytes, int64, wave-uniform)."""
        return _ld64(self.page, WPTR, SYS)

    def done(self):
        """The value of the last completed fence at ``done_address`` (int32, wave-uniform)."""
        return fx.Int32(_uniform(_load(self.page, DONE, 0, 1, aux=SYS)[0]))

    def pad(self, wptr, pad):
        """NOPs over the ``pad`` bytes after ``wptr`` (``place``)."""
        start = fx.Int32(wptr) & (RING_BYTES - 1)
        _loop(self.lane, pad >> 4, 64, lambda i: _store(self.ring, nops(4), start + i * 16))

    def store(self, offset, words, live=None):
        """Store packet dwords (a multiple of 4) at ring byte ``offset`` (16-byte aligned) on the lanes where
        ``live`` (all if None)."""
        assert len(words) % 4 == 0, len(words)
        for i in range(0, len(words), 4):
            at = offset + 4 * i
            _store(self.ring, words[i : i + 4], at if live is None else live.select(at, fx.Int32(_DROP)))

    def commit(self, new_wptr):
        """Publish the packets stored so far up to ``new_wptr`` and ring the doorbell (rule 2)."""
        first = self.lane == 0
        rocdl.s_waitcnt(vmcnt=0)
        _store(self.page, _w64(new_wptr), first.select(fx.Int32(WPTR), fx.Int32(_DROP)))
        rocdl.s_waitcnt(vmcnt=0)
        _store(self.bell, _w64(new_wptr), first.select(fx.Int32(0), fx.Int32(_DROP)))
        rocdl.s_waitcnt(vmcnt=0)


# ---------------------------------------------------------------------------------------------------------------
# Host: queue creation
# ---------------------------------------------------------------------------------------------------------------
_KFD_CREATE_QUEUE = (3 << 30) | (96 << 16) | (0x4B << 8) | 0x02
_KFD_DESTROY_QUEUE = (3 << 30) | (8 << 16) | (0x4B << 8) | 0x03
_CQ_FMT = "<QQQQIIIIIIQQQIIII"
_QUEUE_TYPE_SDMA_BY_ENG_ID = 4
_TOPOLOGY = "/sys/class/kfd/kfd/topology/nodes"
_POOL_EXECUTABLE, _POOL_UNCACHED = 1 << 2, 1 << 3
_AGENT_INFO_DEVICE, _AGENT_INFO_BDFID, _AGENT_INFO_DOMAIN = 17, 0xA006, 0xA00F

_AGENT_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p)
_POOL_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p)
_LIBS = {}


def _runtime(name, sonames):
    """The HIP / HSA runtime torch already loaded (a second copy would be another runtime)."""
    lib = _LIBS.get(name)
    if lib is None:
        for soname in sonames:
            try:
                lib = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError(f"SDMA queues: {name} runtime not loaded in this process")
        _LIBS[name] = lib
    return lib


def _hip():
    return _runtime("hip", ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"))


def _hsa():
    hsa = _runtime("hsa", ("libhsa-runtime64.so.1", "libhsa-runtime64.so"))
    if getattr(hsa, "_sdma_typed", False):
        return hsa
    hsa._sdma_typed = True
    hsa.hsa_iterate_agents.argtypes = [_AGENT_CB, ctypes.c_void_p]
    hsa.hsa_agent_get_info.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
    hsa.hsa_amd_agent_iterate_memory_pools.argtypes = [ctypes.c_uint64, _POOL_CB, ctypes.c_void_p]
    hsa.hsa_amd_memory_pool_get_info.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p]
    hsa.hsa_amd_memory_pool_allocate.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint32,
                                                 ctypes.POINTER(ctypes.c_void_p)]
    hsa.hsa_amd_memory_pool_free.argtypes = [ctypes.c_void_p]
    hsa.hsa_amd_memory_fill.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_size_t]
    return hsa


def _check_hsa(what, status):
    if status != 0:
        raise RuntimeError(f"SDMA queues: {what} failed (hsa_status {status:#x})")


def _kfd_fd():
    for name in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{name}") == "/dev/kfd":
                return int(name)
        except OSError:
            continue
    raise RuntimeError("SDMA queues: no /dev/kfd descriptor in this process")


def _bdf(index):
    """(PCI domain, bus << 8 | device << 3 | function) of HIP device ``index``."""
    buf = ctypes.create_string_buffer(64)
    hip = _hip()
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    if hip.hipDeviceGetPCIBusId(buf, 64, index) != 0:
        raise RuntimeError(f"SDMA queues: hipDeviceGetPCIBusId({index}) failed")
    dom, bus, rest = buf.value.decode().split(":")
    dev, fn = rest.split(".")
    return int(dom, 16), (int(bus, 16) << 8) | (int(dev, 16) << 3) | int(fn, 16)


def _kfd_node(key):
    """(gpu_id, xGMI SDMA engines) of the KFD topology node of PCI location ``key``."""
    for n in os.listdir(_TOPOLOGY):
        try:
            with open(f"{_TOPOLOGY}/{n}/gpu_id") as f:
                gpu_id = int(f.read().strip() or 0)
            props = {}
            with open(f"{_TOPOLOGY}/{n}/properties") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        props[parts[0]] = int(parts[1])
        except (OSError, ValueError):
            continue
        if gpu_id and (props.get("domain", 0), props.get("location_id")) == key:
            return gpu_id, props.get("num_sdma_xgmi_engines", 0)
    raise RuntimeError(f"SDMA queues: no KFD node at PCI location {key}")


def _props(path):
    props = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                props[parts[0]] = int(parts[1])
    return props


def recommended(gpu_id, peer_gpu_id):
    """KFD's recommended engines for copies from KFD GPU ``gpu_id`` into ``peer_gpu_id``'s memory (the
    ``recommended_sdma_engine_id_mask`` of the topology io_link between them), odd ids folded to the even ones (on
    MI308X the odd ids name fused-off engines); empty if KFD names none."""
    nodes = {}
    for n in os.listdir(_TOPOLOGY):
        try:
            with open(f"{_TOPOLOGY}/{n}/gpu_id") as f:
                nodes[int(f.read().strip() or 0)] = n
        except (OSError, ValueError):
            continue
    src, dst = nodes.get(gpu_id), nodes.get(peer_gpu_id)
    if not gpu_id or src is None or dst is None:
        return frozenset()
    links = f"{_TOPOLOGY}/{src}/io_links"
    for link in sorted(os.listdir(links)):
        try:
            props = _props(f"{links}/{link}/properties")
        except (OSError, ValueError):
            continue
        if str(props.get("node_to")) == dst:
            mask = props.get("recommended_sdma_engine_id_mask", 0)
            return frozenset(e & ~1 for e in range(2, 8) if mask >> e & 1)
    return frozenset()


def push_order(engines_, recs):
    """Queue order of a rank's pushes: ``order[k]`` is the queue (index into ``engines_``) at position k, whose
    copies go to the peer with recommended engines ``recs[k]`` (``recommended``; positions beyond ``recs`` take the
    remaining queues): the permutation with the most positions on a recommended engine, ties to the earliest in
    ``engines_`` order (no recommendation: the identity). Pushing on another engine can slow this GPU's kernels
    meanwhile at the same copy bandwidth (tools/tcs_engbw.py: 51 GB/s on every engine; the one-shot push of down_tp
    rule 12 on engine 2 of the second of MI308X GPUs 0 .. 3, where KFD recommends engine 5, folded to 4, makes the
    whole TP2 layer 5 % slower at M 32768)."""
    import itertools

    best, score = tuple(range(len(engines_))), 0
    for perm in itertools.permutations(range(len(engines_))):
        hits = sum(engines_[perm[k]] in recs[k] for k in range(min(len(recs), len(perm))))
        if hits > score:
            best, score = perm, hits
    return best


def _agent_pool(key):
    """(HSA agent, its coarse-grained VRAM pool) of PCI location ``key``."""
    hsa, agents, pools = _hsa(), [], []

    def visit_agent(agent, _data):
        kind, bdf, dom = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
        if hsa.hsa_agent_get_info(agent, _AGENT_INFO_DEVICE, ctypes.byref(kind)) == 0 and kind.value == 1:
            hsa.hsa_agent_get_info(agent, _AGENT_INFO_BDFID, ctypes.byref(bdf))
            hsa.hsa_agent_get_info(agent, _AGENT_INFO_DOMAIN, ctypes.byref(dom))
            if (dom.value, bdf.value) == key:
                agents.append(agent)
        return 0

    def visit_pool(pool, _data):
        segment, flags, alloc = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_bool()
        hsa.hsa_amd_memory_pool_get_info(pool, 0, ctypes.byref(segment))
        hsa.hsa_amd_memory_pool_get_info(pool, 1, ctypes.byref(flags))
        hsa.hsa_amd_memory_pool_get_info(pool, 5, ctypes.byref(alloc))
        # Global segment, coarse-grained, allocatable.
        if segment.value == 0 and flags.value & 4 and alloc.value:
            pools.append(pool)
        return 0

    agent_cb, pool_cb = _AGENT_CB(visit_agent), _POOL_CB(visit_pool)
    _check_hsa("hsa_iterate_agents", hsa.hsa_iterate_agents(agent_cb, None))
    if not agents:
        raise RuntimeError(f"SDMA queues: no HSA agent at PCI location {key}")
    _check_hsa("hsa_amd_agent_iterate_memory_pools",
               hsa.hsa_amd_agent_iterate_memory_pools(agents[0], pool_cb, None))
    if not pools:
        raise RuntimeError("SDMA queues: no coarse-grained VRAM pool")
    return agents[0], pools[0]


def _mapping_at(offset):
    """Start address of this process's /dev/kfd mapping at file offset ``offset`` (0 if none)."""
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[5] == "/dev/kfd" and int(parts[2], 16) == offset:
                return int(parts[0].split("-")[0], 16)
    return 0


class _Queue:
    def __init__(self, kfd, gpu_id, pool, engine):
        hsa = _hsa()
        self.kfd, self.engine, self.ring, self.page, self.qid = kfd, engine, None, None, None
        try:
            self.ring = self._alloc(hsa, pool, RING_BYTES)
            self.page = self._alloc(hsa, pool, PAGE_BYTES)
            args = bytearray(struct.pack(_CQ_FMT, self.ring, self.page + WPTR, self.page + RPTR, 0, RING_BYTES,
                                         gpu_id, _QUEUE_TYPE_SDMA_BY_ENG_ID, 100, 15, 0, 0, 0, 0, 0, 0, engine, 0))
            fcntl.ioctl(kfd, _KFD_CREATE_QUEUE, args, True)
            fields = struct.unpack(_CQ_FMT, args)
            self.qid = fields[9]
            offset = fields[3]
            frame = _mapping_at(offset & ~0x1FFF)
            if not frame:
                raise RuntimeError(
                    f"SDMA queues: doorbell page {offset & ~0x1FFF:#x} not mapped (no ROCr queue on the device yet)"
                )
            self.doorbell = frame + (offset & 0x1FFF)
        except Exception:
            self.destroy()
            raise

    @staticmethod
    def _alloc(hsa, pool, nbytes):
        ptr = ctypes.c_void_p()
        _check_hsa("hsa_amd_memory_pool_allocate",
                   hsa.hsa_amd_memory_pool_allocate(pool, nbytes, _POOL_EXECUTABLE | _POOL_UNCACHED,
                                                    ctypes.byref(ptr)))
        _check_hsa("hsa_amd_memory_fill", hsa.hsa_amd_memory_fill(ptr, 0, nbytes // 4))
        return ptr.value

    def destroy(self):
        hsa = _hsa()
        if self.qid is not None:
            fcntl.ioctl(self.kfd, _KFD_DESTROY_QUEUE, bytearray(struct.pack("<II", self.qid, 0)), True)
            self.qid = None
        for name in ("ring", "page"):
            ptr = getattr(self, name)
            if ptr:
                hsa.hsa_amd_memory_pool_free(ctypes.c_void_p(ptr))
                setattr(self, name, None)


class SdmaQueues:
    """User-mode SDMA queues of the CUDA device ``device`` (torch.device with an index), one per engine of
    ``engines``; ``desc``: int64 [len(engines), 4] on the device, per queue (ring, pointer page, doorbell, 0).
    Not thread-safe; ``recreate`` / ``close`` only while no kernel uses the queues."""

    def __init__(self, device, engines=XGMI_ENGINES):
        self.device, self.engines = device, tuple(engines)
        key = _bdf(device.index)
        self.gpu_id, xgmi = _kfd_node(key)
        bad = [e for e in self.engines if not 2 <= e < 2 + xgmi]
        if bad:
            raise RuntimeError(f"SDMA queues: engines {bad} are not xGMI engines (2 .. {1 + xgmi})")
        _, self.pool = _agent_pool(key)
        self.kfd = _kfd_fd()
        self.queues, self._ordered = [], {}
        self.desc = torch.zeros(len(self.engines), DESC_BYTES // 8, dtype=torch.int64, device=device)
        self._create()

    def _create(self):
        try:
            for engine in self.engines:
                self.queues.append(_Queue(self.kfd, self.gpu_id, self.pool, engine))
        except Exception:
            self.close()
            raise
        rows = [[q.ring, q.page, q.doorbell, 0] for q in self.queues]
        self.desc.copy_(torch.tensor(rows, dtype=torch.int64))
        for order, rows in self._ordered.items():
            rows.copy_(self.desc[list(order)])
        torch.cuda.synchronize(self.device)

    def ordered(self, order):
        """``desc`` with its rows in the queue order ``order`` (queue indices), kept in step with ``desc`` by
        ``recreate`` (one tensor per order, shared by the callers)."""
        order = tuple(order)
        if order == tuple(range(len(self.engines))):
            return self.desc
        rows = self._ordered.get(order)
        if rows is None:
            rows = self._ordered[order] = self.desc[list(order)].clone()
        return rows

    def recreate(self):
        """Fresh queues (empty rings, write pointers 0) in ``desc``, e.g. after a launch that gave up waiting left
        packets behind."""
        self.close()
        self._create()

    def close(self):
        queues, self.queues = self.queues, []
        for q in queues:
            q.destroy()


_SHARED: dict = {}


def engines():
    """The engines of the queues, in queue order: ``XGMI_ENGINES`` or the permutation of them in ``ENGINES_ENV``."""
    text = os.environ.get(ENGINES_ENV, "")
    if not text:
        return XGMI_ENGINES
    order = tuple(int(x) for x in text.split(","))
    if sorted(order) != sorted(XGMI_ENGINES):
        raise ValueError(f"{ENGINES_ENV}={text!r}: a permutation of {XGMI_ENGINES} expected")
    return order


def acquire(device):
    """The ``SdmaQueues`` of CUDA device ``device`` on ``engines()``, shared by the callers (``release``)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    entry = _SHARED.get(index)
    if entry is None:
        entry = _SHARED[index] = [SdmaQueues(torch.device("cuda", index), engines()), 0]
    entry[1] += 1
    return entry[0]


def release(queues):
    """Drop one reference of ``acquire``; the last one destroys the queues (no kernel may use them any more)."""
    for index, entry in list(_SHARED.items()):
        if entry[0] is queues:
            entry[1] -= 1
            if entry[1] == 0:
                del _SHARED[index]
                queues.close()
            return
