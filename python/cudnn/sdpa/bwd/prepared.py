# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable backward launch metadata and per-call pointer binding."""

from dataclasses import dataclass
import math

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.prepared import facts_of_roles

ROLES = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv", "seq_q", "seq_kv", "sink", "dsink", "bias", "dbias")


@dataclass(frozen=True)
class Operand:
    dtype: str
    shape: tuple
    strides: tuple
    span: int
    alignment: int
    itemsize: int
    allowed_numels: tuple = ()


@dataclass(frozen=True)
class BwdLaunchSpec:
    artifact: object
    fn: object
    operands: tuple
    workspace_bytes: int
    device_index: int
    scale: float
    name: str = "sdpa_bwd_sm120"
    length_form: bool = False


def build_sm120_spec(api):
    operands = []
    for role in ROLES:
        if role in ("seq_q", "seq_kv"):
            enabled = getattr(api, role + "_lens_present")
            op = Operand("int32", (api.batch_size,), (1,), api.batch_size, 4, 4) if enabled else None
        else:
            desc = getattr(api, role + "_desc")
            if desc is None:
                op = None
            else:
                shape, strides = tuple(desc.shape), tuple(desc.stride)
                span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
                alignment = 16 if role in ROLES[:5] + ROLES[6:9] else desc.dtype.itemsize
                op = Operand(str(desc.dtype).split(".")[-1], shape, strides, span, alignment, desc.dtype.itemsize)
        operands.append(op)
    owner = api._compiled_kernel
    if owner.workspace_bytes != api.scratch_workspace_bytes():
        raise RuntimeError("SM120 backward compiled workspace layout disagrees with its advertised size")
    fn = positional_entry(owner.entry)
    if fn is None:
        raise NotImplementedError("SM120 backward requires a positional tvm-ffi entry")
    return BwdLaunchSpec(owner, fn, tuple(operands), owner.workspace_bytes, int(api.q_desc.device.index or 0), api.scale_softmax)


def _same_geometry(actual, expected):
    # Singleton strides do not address any second element. In particular Stats
    # may add or remove its trailing singleton dimension without changing layout.
    return tuple((n, st) for n, st in zip(*actual) if n != 1) == tuple((n, st) for n, st in zip(*expected) if n != 1)


def execute(spec, facts, workspace_ptr, stream_int, *, scale=None, geometry=None, raw_storage=False):
    """Validate every operand before launching any stage, including bias initialization."""
    if not workspace_ptr or workspace_ptr % 16:
        raise ValueError(f"{spec.name} needs an aligned caller workspace")
    frame = []
    for i, (name, op) in enumerate(zip(ROLES, spec.operands)):
        f = facts.get(name)
        label = name + "_lens" if name in ("seq_q", "seq_kv") else name
        if op is None:
            if f is not None:
                raise ValueError(f"{spec.name}: {name} was not compiled into this specialization")
            frame.append(None)
            continue
        if f is None:
            raise ValueError(f"{spec.name}: {name} is required by this specialization")
        if f.device not in ((2, spec.device_index), (-1, -1)):
            raise ValueError(f"{spec.name}: {label} must be on CUDA device {spec.device_index}")
        if f.dtype and f.dtype != op.dtype:
            raise ValueError(f"{spec.name}: {name} must be {op.dtype}; got {f.dtype}")
        if not f.ptr or f.ptr % op.alignment:
            raise ValueError(f"{spec.name}: {name} base address must be {op.alignment}-byte aligned")
        if f.span >= 0 and f.span < op.span:
            raise ValueError(f"{spec.name}: {name} backing storage is too small for the declared strides")
        if (
            not raw_storage
            and name in ("seq_q", "seq_kv", "sink", "dsink", "bias", "dbias")
            and f.shape
            and (not f.contiguous or f.numel not in (op.allowed_numels or (math.prod(op.shape),)))
        ):
            raise ValueError(f"{spec.name}: {label} must be contiguous with {math.prod(op.shape)} elements")
        if geometry is not None and geometry[i] is not None and f.shape and not _same_geometry((f.shape, f.strides), geometry[i]):
            raise ValueError(f"{spec.name}: {name} runtime geometry must match this fixed backward plan")
        span = f.numel if op.allowed_numels and f.shape and not raw_storage else op.span
        if workspace_ptr < f.ptr + span * op.itemsize and f.ptr < workspace_ptr + spec.workspace_bytes:
            raise ValueError(f"{spec.name}: caller workspace overlaps {name}")
        frame.append(f.ptr)
    scale = spec.scale if scale is None or scale == 0 else float(scale)
    frame.extend((workspace_ptr, scale * math.log2(math.e), scale))
    if spec.length_form:
        # Graph THD declarations carry B lengths. Standalone also accepts B+1
        # prefixes; their form is host metadata, never a device read.
        form = 0
        if not raw_storage:
            for bit, name in enumerate(("seq_q", "seq_kv")):
                f = facts.get(name)
                if f is not None and f.numel == spec.operands[9 + bit].shape[0] + 1:
                    form |= 1 << bit
        frame.append(form)
    frame.append(stream_int)
    spec.fn(*frame)


class PreparedBwdLaunch:
    def __init__(self, spec, binding):
        self.spec = spec
        attributes = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv", "seq_len_q", "seq_len_kv", "sink_token", "dsink", "bias", "dbias")
        tensors = [getattr(binding, name) for name in attributes]
        self._roles = [name for name, tensor in zip(ROLES, tensors) if tensor is not None]
        self._uids = [tensor.get_uid() for tensor in tensors if tensor is not None]
        self._geometry = tuple((tuple(t.get_dim()), tuple(t.get_stride())) if t is not None else None for t in tensors)
        self._indices = None

    def execute(self, pack, workspace_ptr, stream, stream_int):
        if self._indices is None:
            self._indices = [pack.index_of(uid) for uid in self._uids]
        facts = dict(zip(self._roles, facts_of_roles(pack, self._indices)))
        # Graph bindings are raw storage under the declared layout, including
        # strided producer views. Explicit overrides are different: they change
        # the requested operation, which this fixed backward artifact cannot do.
        geometry = None
        if pack.overridden:
            overridden = {role for role, index in zip(self._roles, self._indices) if index in pack.overridden}
            geometry = tuple(g if role in overridden else None for role, g in zip(ROLES, self._geometry))
        execute(self.spec, facts, workspace_ptr, stream_int, geometry=geometry, raw_storage=True)
