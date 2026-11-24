# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A grouped (MoE) plan's variant-pack calls, replayed from an earlier call with the same buffer signature.

A variant-pack call of a grouped plan (``_CompiledMoeBase._call_variant_pack``) resolves the pack, checks every buffer
against the plan, builds the kernel's argument views and the problem tuple, fills the outputs that need a seed and
launches. Everything it decides is a function of the plan and of the bound buffers' :func:`signature`: each buffer's
type, shape, strides, dtype, device and the residue of its address modulo :data:`PTR_RESIDUE`, and which bound buffers
share an address. The checks read an address only through alignment tests of at most ``MAX_MEM_ACCESS_BYTES`` bytes,
which the residue settles, and through whether two bound buffers share it.

The first call with a signature takes that full path. The second takes it while a :class:`LaunchRecord` records what
it hands the driver -- the workspace carve, each output fill and the launch arguments -- and :func:`derive` turns the
record into a :class:`LaunchEntry`: per launch argument, the bound buffer it views and the permute / view steps that
give it. A later call with the same signature runs only :func:`arguments` and :func:`fill` on its own buffers before the
launch. A call whose signature has no entry is a call the full path has not checked, so it takes the full path. Calls
whose signatures never repeat pay for reading and looking up their signature, and displace no entry of calls that do.

An entry holds no buffer: slot indices, shapes, permutations, fill plans and the problem tuple. The workspace, the
launch epoch and the stream come from each call, so a replayed call captured into a CUDA graph is the launch the full
path would have captured."""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
from typing import NamedTuple

from cudnn.frost import buffers

from .dtypes import MAX_MEM_ACCESS_BYTES

_LOG = logging.getLogger(__name__)

# A multiple of every alignment a launch-path check tests (at most MAX_MEM_ACCESS_BYTES bytes): two addresses with the
# same residue pass and fail the same checks.
PTR_RESIDUE = 256
if PTR_RESIDUE % MAX_MEM_ACCESS_BYTES:
    raise ImportError(f"launch_cache.PTR_RESIDUE ({PTR_RESIDUE}) must be a multiple of MAX_MEM_ACCESS_BYTES ({MAX_MEM_ACCESS_BYTES})")

# Entries a plan keeps, and signatures it remembers having seen once; past either bound the oldest is dropped.
MAX_ENTRIES = 64

# Stored for a signature whose calls cannot be replayed from a record: they keep taking the full path.
UNCACHEABLE = object()

_LOCK = threading.Lock()
_STATE = threading.local()


class LaunchRecord:
    """What one full-path call handed the driver, in order: workspace carves ``(descriptor slots, carve)``, output fills
    ``(address, plan, kind, value)`` and launches ``(problem without the epoch, K-padding kernel?, arguments)``."""

    __slots__ = ("carves", "fills", "launches")

    def __init__(self):
        self.carves = []
        self.fills = []
        self.launches = []


def active_record() -> "LaunchRecord | None":
    """The record this thread's call is filling, or None."""
    return getattr(_STATE, "record", None)


@contextlib.contextmanager
def recording():
    """Record the driver work of the full-path call made inside the block, on this thread."""
    previous = getattr(_STATE, "record", None)
    record = _STATE.record = LaunchRecord()
    try:
        yield record
    finally:
        _STATE.record = previous


class LaunchEntry(NamedTuple):
    """A recorded call, by bound-buffer slot (the order of ``binding.bound_tensors()``)."""

    workspace_slots: int  # the descriptor slots the call carves from its workspace
    # (slot, plan, kind, value): kind "zero", a memset (value: element bytes); "kernel", the plan's programmatic fill of
    # its COMBINE output (``combine_zero``) over the one region of ``plan`` (value: element bytes); or "word" (value: the
    # 32-bit word)
    fills: tuple
    problem: "tuple | None"  # the problem tuple without its launch epoch; None: the call launches nothing
    k_pad: bool  # the launch takes the plan's K-padding kernel
    args: tuple  # per launch argument: None for the workspace carve, else (slot, steps) (``view``)


def signature(slots) -> tuple:
    """The facts a grouped call's checks and views read off its bound buffers: per buffer its type, shape, strides,
    dtype, ``device`` where its type has one (a framework tensor's; a variant-pack operand has none: it is on its pack's
    device) and address residue; then, when two of them share an address, the first slot at each one's address."""
    ptrs = [b.data_ptr() for b in slots]
    facts = tuple([(type(b), tuple(b.shape), tuple(b.stride()), b.dtype, getattr(b, "device", None), p % PTR_RESIDUE) for b, p in zip(slots, ptrs)])
    return facts, (None if len(set(ptrs)) == len(ptrs) else tuple(ptrs.index(p) for p in ptrs))


def fill_count(chain) -> int:
    """The output fills a grouped call of ``chain`` issues: one per reduction output, one for a COMBINE output."""
    return sum(1 for spec in chain.outputs if spec.is_reduction) + (1 if chain.moe.mode == "combine" else 0)


def view(buf, steps):
    """``buf`` through ``steps``: each a ``(permutation, None)`` or a ``(None, shape)``, which views ``buf`` at that
    shape. A view never copies: a reshape the launch path made without a copy is the view of the same shape."""
    for perm, shape in steps:
        buf = buf.permute(*perm) if perm is not None else buf.view(shape)
    return buf


def _facts(buf) -> tuple:
    return type(buf), buf.dtype, tuple(buf.shape), tuple(buf.stride()), int(buf.data_ptr())


def _permutation(src_shape, src_stride, shape, stride) -> "tuple | None":
    """The axis order that moves each (extent, stride) of the source onto the target's, or None."""
    if len(src_shape) != len(shape):
        return None
    free = list(range(len(src_shape)))
    perm = []
    for extent, step in zip(shape, stride):
        for axis in free:
            if src_shape[axis] == extent and src_stride[axis] == step:
                free.remove(axis)
                perm.append(axis)
                break
        else:
            return None
    return tuple(perm)


def _gives(src, steps, want) -> bool:
    try:
        return _facts(view(src, steps)) == want
    except Exception:  # noqa: BLE001 -- a step this buffer type does not take, or a shape it cannot view
        return False


def view_steps(src, arg) -> "tuple | None":
    """The steps (:func:`view`) that give ``arg`` from ``src`` -- the same type, dtype, address, shape and strides -- or
    None. Tried: none or one permute (the one that moves src's extents and strides onto arg's), one view, a permute
    then a view, which covers the views the grouped launch paths build. Each candidate is checked by building it: steps
    that give ``arg`` from ``src`` give the same view of any buffer with ``src``'s type, dtype, shape and strides."""
    want = _facts(arg)
    shape, stride = want[2], want[3]
    rank = len(src.shape)
    perm = _permutation(tuple(src.shape), tuple(src.stride()), shape, stride)
    if perm is not None:
        steps = () if perm == tuple(range(rank)) else ((perm, None),)
        if _gives(src, steps, want):
            return steps
    if _gives(src, ((None, shape),), want):
        return ((None, shape),)
    for turn in itertools.permutations(range(rank)):
        steps = ((turn, None), (None, shape))
        if turn != tuple(range(rank)) and _gives(src, steps, want):
            return steps
    return None


class _NotReplayable(Exception):
    """This signature's calls do something a :class:`LaunchEntry` does not describe."""


def _at(ptrs, ptr) -> list:
    """The slots of the bound buffers at ``ptr``."""
    found = [i for i, p in enumerate(ptrs) if p == ptr]
    if not found:
        raise _NotReplayable(f"address 0x{ptr:x} is no bound buffer's")
    return found


def derive(record: LaunchRecord, slots, fills: int):
    """The :class:`LaunchEntry` that replays the call ``record`` recorded, on the bound buffers ``slots``, or
    :data:`UNCACHEABLE` when this signature's calls cannot be replayed.

    A launch argument views the bound buffer at its address; where several share it (the signature says which), any of
    them whose steps give the argument gives it in every call of the signature, as they share their address there too."""
    try:
        if len(record.carves) != 1 or len(record.launches) > 1:
            raise _NotReplayable(f"{len(record.carves)} workspace carves, {len(record.launches)} launches")
        if len(record.fills) != fills:
            raise _NotReplayable(f"{len(record.fills)} output fills where the plan issues {fills}")
        ptrs = [int(buf.data_ptr()) for buf in slots]
        workspace_slots, carve = record.carves[0]
        recorded_fills = tuple((_at(ptrs, int(ptr))[0], plan, kind, value) for ptr, plan, kind, value in record.fills)
        if not record.launches:
            return LaunchEntry(int(workspace_slots), recorded_fills, None, False, ())
        problem, k_pad, launch_args = record.launches[0]
        if sum(1 for arg in launch_args if arg is carve) != 1:
            raise _NotReplayable("the launch does not take the workspace carve exactly once")
        args = []
        for at, arg in enumerate(launch_args):
            if arg is carve:
                args.append(None)
                continue
            found = next(((slot, steps) for slot in _at(ptrs, int(arg.data_ptr())) for steps in [view_steps(slots[slot], arg)] if steps is not None), None)
            if found is None:
                raise _NotReplayable(f"no permute / view of a bound buffer gives launch argument {at}")
            args.append(found)
        return LaunchEntry(int(workspace_slots), recorded_fills, tuple(problem), bool(k_pad), tuple(args))
    except Exception as exc:  # noqa: BLE001 -- the call already ran; a record this module cannot read only costs replay
        _LOG.debug("cudnn.frost gemm: grouped calls of this signature take the full path: %s", exc)
        return UNCACHEABLE


def arguments(entry: LaunchEntry, slots, carve) -> list:
    """The launch arguments of ``entry`` on this call's bound buffers and workspace carve."""
    args = []
    for arg in entry.args:
        if arg is None:
            args.append(carve)
            continue
        buf = slots[arg[0]]
        for perm, shape in arg[1]:
            buf = buf.permute(*perm) if perm is not None else buf.view(shape)
        args.append(buf)
    return args


def fill(entry: LaunchEntry, slots, stream, zero_kernel=None, as_custream=None) -> None:
    """Issue ``entry``'s output fills on this call's bound buffers, as the full path issues them: a "kernel" fill through
    the plan's programmatic COMBINE fill ``zero_kernel``, on ``stream`` as ``as_custream`` gives it (the family's
    ``_initialize_combine_output`` calls it the same way)."""
    for slot, plan, kind, value in entry.fills:
        ptr = slots[slot].data_ptr()
        if kind == "zero":
            buffers.apply_zero_fill_plan(ptr, plan, value, stream)
        elif kind == "kernel":
            offset, pitch, width, height = plan[0]
            zero_kernel(ptr + offset * value, pitch * value, width * value, height, as_custream(stream))
        else:
            buffers.apply_fill_plan(ptr, plan, value, stream)


def _bounded_set(table: dict, key, value) -> None:
    """``table[key] = value``, dropping the oldest key when that makes more than MAX_ENTRIES (the caller holds _LOCK)."""
    if key not in table and len(table) >= MAX_ENTRIES:
        del table[next(iter(table))]
    table[key] = value


class LaunchCache:
    """One grouped plan's entries by signature, and the signatures seen once, for the binding they were derived for."""

    __slots__ = ("binding", "bound", "index", "fills", "entries", "seen")

    def __init__(self, binding, fills: int):
        self.binding = binding
        self.bound = tuple(binding.bound_tensors())
        self.index = {id(t): i for i, t in enumerate(self.bound)}
        self.fills = fills
        self.entries: dict = {}
        self.seen: dict = {}

    def slots(self, variant_pack: dict, resolve) -> "list | None":
        """The pack's buffer for each bound tensor, in slot order; None when a bound tensor has none (the full path names
        it). A key that is not a bound tensor object -- a uid or a name -- is left to ``resolve``
        (``graph_analyzer.resolve_variant_pack``), which also refuses a key that matches no bound tensor."""
        index = self.index
        slots = [None] * len(self.bound)
        for key, buf in variant_pack.items():
            at = index.get(id(key))
            if at is None:
                resolved = resolve(variant_pack, self.binding)
                slots = [resolved.get(id(t)) for t in self.bound]
                break
            slots[at] = buf
        for buf in slots:
            if buf is None:
                return None
        return slots

    def first_sighting(self, key: tuple) -> bool:
        """Whether no call with signature ``key`` came before (among the last MAX_ENTRIES signatures seen once); notes
        this one."""
        with _LOCK:
            seen = self.seen
            if key in seen:
                return False
            if len(seen) >= MAX_ENTRIES:
                del seen[next(iter(seen))]
            seen[key] = None
            return True

    def learn(self, key: tuple, record: LaunchRecord, slots) -> None:
        """Keep the entry ``record`` gives for signature ``key``."""
        entry = derive(record, slots, self.fills)
        with _LOCK:
            self.seen.pop(key, None)
            _bounded_set(self.entries, key, entry)
