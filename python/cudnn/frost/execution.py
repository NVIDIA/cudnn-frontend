# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared, same-stream composition of executors over caller-owned storage.

No graph operations or kernel DSL objects live here. Engines declare fixed
buffer contracts and ordered steps. A step prepares a zero-argument enqueue
call without writing device storage; every step is prepared before any enqueue.
Temporary tensors have disjoint workspace regions. Step-local scratch shares
one region because all enqueues use the same stream. Dynamic geometry, storage
reuse between tensor lifetimes, and implicit aliasing are deliberately absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from operator import index
from typing import Callable, Literal

from cudnn import _pybind_module

from . import buffers
from .device import device_context
from .workspace import Workspace, WorkspaceLayout, carve_plan


class VariantPackPlan:
    """Normalize a prepared executor's call convention without changing its ABI.

    Legacy executors may order workspace and stream differently. At a shared
    graph-plan boundary, pass them by keyword and preserve executor metadata.
    """

    def __init__(self, executor):
        self._executor = executor

    def __getattr__(self, name):
        return getattr(self._executor, name)

    def __call__(self, variant_pack, stream=None, workspace=None):
        return self._executor(variant_pack, stream=stream, workspace=workspace)


@dataclass(frozen=True)
class BufferBinding:
    """Fixed physical storage for a logical value, not an operation or a view."""

    key: object
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    role: Literal["input", "output", "temporary", "inout"] = "input"
    label: str = "buffer"
    alignment: int = field(default=1, kw_only=True)
    span: int = field(init=False)
    nbytes: int = field(init=False)

    def __post_init__(self):
        hash(self.key)
        alignment = index(self.alignment)
        if alignment < 1 or alignment & (alignment - 1) or alignment > (1 << 63) - 1:
            raise ValueError("buffer binding alignment must be a positive Int64 power of two")
        object.__setattr__(self, "alignment", alignment)
        shape, strides = tuple(map(index, self.shape)), tuple(map(index, self.strides))
        if len(shape) != len(strides) or any(d < 0 for d in shape) or any(s < 0 for s in strides):
            raise ValueError("buffer binding requires nonnegative extents and matching nonnegative strides")
        if self.dtype not in buffers.DTYPE_ITEMSIZE or self.role not in ("input", "output", "temporary", "inout"):
            raise ValueError("buffer binding requires a byte-addressable dtype and a known storage role")
        span = 0 if prod(shape) == 0 else 1 + sum((d - 1) * s for d, s in zip(shape, strides))
        nbytes = span * buffers.DTYPE_ITEMSIZE[self.dtype]
        if max(prod(shape), span, nbytes, *shape, *strides) > (1 << 63) - 1:
            raise ValueError("buffer binding exceeds signed Int64 geometry")
        if self.role != "input" and not buffers.is_nonoverlapping(shape, strides):
            raise ValueError("writable buffer binding requires non-overlapping storage")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", strides)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "nbytes", nbytes)


@dataclass(frozen=True)
class ExecutionStep:
    """Prepare bindings and guards, then return an enqueue closure.

    ``prepare(values, scratch, stream)`` must not launch, synchronize, allocate
    device storage, or retain per-call buffers in persistent plan state. Its
    returned closure may enqueue work only on the supplied stream. ``scratch``
    is a uint8 view of exactly workspace_bytes bytes, or None when zero.
    """

    name: str
    reads: tuple[object, ...]
    writes: tuple[object, ...]
    prepare: Callable
    workspace_bytes: int = 0

    def __post_init__(self):
        size = index(self.workspace_bytes)
        if not self.name or not callable(self.prepare) or not 0 <= size <= (1 << 63) - 1:
            raise ValueError("execution step requires a name, preparation callback and nonnegative Int64 scratch size")
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        object.__setattr__(self, "workspace_bytes", size)


class SequentialPlan:
    """Framework-neutral ordered execution with full preflight before writes.

    Distinct writable logical values must have disjoint storage. An explicitly
    declared inout value can be read and written under the same key. The plan
    owns declarations and enqueue callbacks, never device buffers or streams.
    """

    def __init__(self, bindings, steps, *, device: int, owner="sequential plan"):
        self.device, self.owner = index(device), owner
        self.steps = tuple(steps)
        if self.device < 0 or not self.steps:
            raise ValueError("sequential plan requires a device and at least one execution step")
        bindings = tuple(bindings)
        self._bindings = {binding.key: binding for binding in bindings}
        if len(self._bindings) != len(bindings):
            raise ValueError("sequential plan has duplicate buffer keys")
        ready = {b.key for b in bindings if b.role in ("input", "inout")}
        for step in self.steps:
            if any(key not in self._bindings for key in (*step.reads, *step.writes)):
                raise ValueError(f"{step.name}: undeclared buffer dependency")
            if any(key not in ready for key in step.reads):
                raise ValueError(f"{step.name}: reads a value before its producer")
            if any(self._bindings[key].role == "input" for key in step.writes):
                raise ValueError(f"{step.name}: writes a read-only input")
            ready.update(step.writes)
        if any(b.role in ("output", "temporary") and b.key not in ready for b in bindings):
            raise ValueError("sequential plan has an output without a producer")
        self.bound = tuple(b.key for b in bindings if b.role != "temporary")
        self._writable = tuple(b.key for b in bindings if b.role != "input")
        layout = WorkspaceLayout()
        regions = []
        self._temporaries = tuple(b for b in bindings if b.role == "temporary" and b.nbytes)
        self._empty_temporaries = tuple(b for b in bindings if b.role == "temporary" and not b.nbytes)
        for binding in self._temporaries:
            regions.append((layout.add(binding.nbytes, align=binding.alignment), binding.dtype, (binding.span,)))
        scratch_size = max(step.workspace_bytes for step in self.steps)
        if scratch_size:
            regions.append((layout.add(scratch_size), "uint8", (scratch_size,)))
        self.workspace_bytes, self._alignment = layout.size, layout.base_align
        if self.workspace_bytes > (1 << 63) - 1:
            raise ValueError("sequential plan workspace exceeds signed Int64 geometry")
        self._carve = carve_plan(owner, regions) if regions else None

    def __call__(self, variant_pack, stream=None, workspace=None):
        with device_context(self.device):
            self._execute(variant_pack, workspace, stream)

    def _execute(self, variant_pack, workspace, stream):
        if any(key not in variant_pack for key in self.bound):
            raise ValueError(f"{self.owner}: variant pack is missing a prepared operand")
        # Unknown pack entries are not this plan's dependencies or alias checks.
        values = {key: variant_pack[key] for key in self.bound}
        workspace_buffer = None
        if self.workspace_bytes:
            if not isinstance(workspace, Workspace):
                workspace = Workspace(workspace, self.workspace_bytes, self.owner, align=self._alignment, device=self.device)
            workspace_buffer = workspace.view(0, "uint8", (self.workspace_bytes,))
            if workspace_buffer.__dlpack_device__() != (2, self.device) or buffers.data_ptr(workspace_buffer) % self._alignment:
                raise ValueError(f"{self.owner}: workspace has the wrong device or alignment")
        if self._carve is not None:
            carved = workspace.carve(self._carve)
            for binding, flat in zip(self._temporaries, carved):
                code, bits = buffers.DTYPES[binding.dtype]
                values[binding.key] = _pybind_module.make_operand_buffer(flat.data_ptr(), list(binding.shape), code, bits, self.device, list(binding.strides))
        else:
            carved = ()
        for binding in self._empty_temporaries:
            code, bits = buffers.DTYPES[binding.dtype]
            values[binding.key] = _pybind_module.make_operand_buffer(0, list(binding.shape), code, bits, self.device, list(binding.strides))
        ranges = {}
        for key, value in values.items():
            binding = self._bindings[key]
            ptr, shape, strides, dtype, device = buffers.probe(value)
            stride_method = getattr(value, "stride", None)
            if callable(stride_method):
                strides = tuple(stride_method())
            elif strides is None:
                strides = tuple(prod(max(d, 1) for d in shape[i + 1 :]) for i in range(len(shape)))
            if (tuple(shape), tuple(strides), dtype) != (binding.shape, binding.strides, binding.dtype):
                raise ValueError(f"{self.owner}: {binding.label} differs from its prepared shape/stride/dtype")
            dlpack_device = getattr(value, "__dlpack_device__", None)
            if dlpack_device is not None:
                device_type, device = dlpack_device()
                if int(device_type) != 2:
                    raise ValueError(f"{self.owner}: bindings must be CUDA buffers on the compiled device")
            else:
                device_attr = getattr(value, "device", None)
                if getattr(device_attr, "type", None) is not None and getattr(device_attr, "index", None) is not None:
                    if device_attr.type != "cuda":
                        raise ValueError(f"{self.owner}: bindings must be CUDA buffers on the compiled device")
                    device = device_attr.index
                elif binding.nbytes:
                    # CAI alone has no device ordinal. As for Workspace, query
                    # pointer metadata rather than trusting probe's default 0.
                    from cuda.bindings import runtime

                    error, attributes = runtime.cudaPointerGetAttributes(ptr)
                    if error != runtime.cudaError_t.cudaSuccess:
                        raise ValueError(f"{self.owner}: cannot identify the CUDA buffer device: {error}")
                    device = attributes.device
                else:
                    raise ValueError(f"{self.owner}: an empty binding requires explicit CUDA device metadata")
            if int(device) != self.device:
                raise ValueError(f"{self.owner}: bindings must be CUDA buffers on the compiled device")
            if ptr < 0 or ptr + binding.nbytes > 1 << 64 or (binding.nbytes and not ptr):
                raise ValueError(f"{self.owner}: buffer address range is invalid")
            if binding.nbytes and ptr % binding.alignment:
                raise ValueError(f"{self.owner}: {binding.label} requires {binding.alignment}-byte pointer alignment")
            ranges[key] = (ptr, ptr + binding.nbytes)
        if workspace_buffer is not None:
            base = buffers.data_ptr(workspace_buffer)
            for key in self.bound:
                lo, hi = ranges[key]
                if lo < hi and base < hi and lo < base + self.workspace_bytes:
                    raise ValueError(f"{self.owner}: workspace must not overlap caller operands")
        for key in self._writable:
            lo, hi = ranges[key]
            for other, (other_lo, other_hi) in ranges.items():
                if other != key and lo < hi and other_lo < other_hi and lo < other_hi and other_lo < hi:
                    raise ValueError(f"{self.owner}: outputs/workspace must not overlap other operands")
        calls = []
        for step in self.steps:
            scratch = None
            if step.workspace_bytes:
                code, bits = buffers.DTYPES["uint8"]
                scratch = _pybind_module.make_operand_buffer(carved[-1].data_ptr(), [step.workspace_bytes], code, bits, self.device)
            call = step.prepare(values, scratch, stream)
            if not callable(call):
                raise TypeError(f"{step.name}: preparation did not return an enqueue callable")
            calls.append(call)
        for call in calls:
            call()
