# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable backward launch metadata and per-call pointer binding.

Half graph, standalone and staged plans share a native fixed-contract binder
on SM80, SM100/SM103, SM107 and SM120. Each half host declares that ownership
when building its spec. Quantized contracts retain the Python implementation;
graph raw storage and standalone carrier rules meet in the same half binder.
"""

from dataclasses import dataclass, field
import math

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.prepared import _native_pack_from_facts, facts_of_roles

ROLES = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv", "seq_q", "seq_kv", "sink", "dsink", "bias", "dbias")
ATTRIBUTES = ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv", "seq_len_q", "seq_len_kv", "sink_token", "dsink", "bias", "dbias")


@dataclass(frozen=True)
class Operand:
    dtype: str
    shape: tuple
    strides: tuple
    span: int
    alignment: int
    itemsize: int
    allowed_numels: tuple = ()
    opaque_bytes: bool = False
    # A PACKED per-tile byte blob (appended; 0 = fixed): the MXFP8 THD scale-factor tensors, laid out per (head, 128-token tile)
    # in cu_seqlens order at ``packed_tile_bytes`` per tile row.  Their LIVE byte count is a per-call fact of the bound buffer
    # (the forward's convention, ``fwd/prepared._bind_mxfp8_scales``): ``bind()`` requires whole tile rows, derives
    # ``count = nbytes // packed_tile_bytes`` and refuses a count above the plan's capacity (``span`` = the capacity in bytes: the
    # larger of ``ceil(T_cap / 128) + B`` tiles per head and the declared scale-factor sample's own count);
    # the counts reach the artifact as appended Int32 frame entries (``BwdLaunchSpec.packed_tile_groups``).
    packed_tile_bytes: int = 0


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
    roles: tuple = ROLES
    attributes: tuple = ATTRIBUTES
    scale_log2: bool = True
    # Roles only the standalone ``execute`` binds (the sm107 half row's caller-provided ``delta``): no graph declares them, so
    # ``PreparedBwdLaunch`` frames them as absent instead of reading a ``SdpaBinding`` attribute that does not exist.  Every
    # other attribute is read strictly -- a misspelled role in a spec still fails at plan build, never as a silent None.
    standalone_only_roles: tuple = ()
    # Groups of PACKED per-tile roles (appended; empty on every dense plan): each entry names the roles that must share ONE
    # packed tile count -- derived per call from their bound byte sizes (``Operand.packed_tile_bytes``) -- and contributes ONE
    # appended Int32 frame entry (the group's count, in this order) right after ``lens_form``.  The MXFP8 THD plan declares two:
    # the q side (``sf_q``, ``sf_q_T``, ``sf_do``, ``sf_do_T``) and the kv side (``sf_k``, ``sf_k_T``, ``sf_v``); a count of 0 (no live
    # tile on that side) is framed as 1 -- a tensor map needs a positive extent, and the kernels' clamped maps never read it.
    packed_tile_groups: tuple = ()
    native_binding: bool = False
    native: object = field(init=False, default=None, repr=False, compare=False)
    native_roles: tuple = field(init=False, default=(), repr=False, compare=False)
    native_indices: tuple = field(init=False, default=(), repr=False, compare=False)

    def __post_init__(self):
        if self.native_binding:
            from cudnn import _pybind_module

            object.__setattr__(self, "native_roles", self.roles[: len(self.operands)])
            object.__setattr__(self, "native_indices", tuple(range(len(self.operands))))
            object.__setattr__(self, "native", _pybind_module._SdpaBwdBinder(self, (None,) * len(self.operands)))


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
    return BwdLaunchSpec(owner, fn, tuple(operands), owner.workspace_bytes, int(api.q_desc.device.index or 0), api.scale_softmax, native_binding=True)


def _same_geometry(actual, expected):
    # Singleton strides do not address any second element. In particular Stats
    # may add or remove its trailing singleton dimension without changing layout.
    return tuple((n, st) for n, st in zip(*actual) if n != 1) == tuple((n, st) for n, st in zip(*expected) if n != 1)


def bind(spec, facts, workspace_ptr, stream_int, *, scale=None, geometry=None, raw_storage=False):
    """Validate every operand before launching any stage, including bias initialization."""
    if spec.native is not None:
        pack = _native_pack_from_facts(facts, spec.native_roles)
        return list(spec.native.bind(pack, spec.native_indices, workspace_ptr, stream_int, (), scale, raw_storage, geometry))
    return _bind_python(spec, facts, workspace_ptr, stream_int, scale=scale, geometry=geometry, raw_storage=raw_storage)


def _bind_python(spec, facts, workspace_ptr, stream_int, *, scale=None, geometry=None, raw_storage=False):
    """Quantized backward contracts and the migration's explicit test reference."""
    if not workspace_ptr or workspace_ptr % 16:
        raise ValueError(f"{spec.name} needs an aligned caller workspace")
    frame = []
    packed_tiles = {}
    for i, (name, op) in enumerate(zip(spec.roles, spec.operands)):
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
        if f.dtype and f.dtype != op.dtype and not op.opaque_bytes:
            raise ValueError(f"{spec.name}: {name} must be {op.dtype}; got {f.dtype}")
        if not f.ptr or f.ptr % op.alignment:
            raise ValueError(f"{spec.name}: {name} base address must be {op.alignment}-byte aligned")
        observed_span = f.span
        if op.opaque_bytes:
            from cudnn.frost.buffers import DTYPE_ITEMSIZE
            from cudnn.sdpa.fwd.prepared import _sf_byte_count

            width = DTYPE_ITEMSIZE.get(f.dtype, 1)
            observed_span *= width
            if op.packed_tile_bytes:
                # A packed per-tile blob: the LIVE byte count is the bound buffer's (whole tile rows, at most the plan's
                # capacity ``op.span``); its tile count is framed for the artifact below (``packed_tile_groups``).
                nbytes = _sf_byte_count(f.shape, f.strides, f.dtype) if f.shape else observed_span
                if nbytes < 0:
                    raise ValueError(f"{spec.name}: {name} must expose its storage extent (a packed scale-factor tensor's tile count is derived from it)")
                if nbytes % op.packed_tile_bytes:
                    raise ValueError(f"{spec.name}: {name} must hold whole packed SF tile rows of {op.packed_tile_bytes} bytes; got {nbytes} bytes")
                if nbytes > op.span:
                    raise ValueError(
                        f"{spec.name}: {name} holds {nbytes // op.packed_tile_bytes} packed SF tiles per head, above the plan's capacity of "
                        f"{op.span // op.packed_tile_bytes} (the larger of ceil(max_total_seq_len / 128) + B and the declared scale-factor sample's tile count)"
                    )
                packed_tiles[name] = nbytes // op.packed_tile_bytes
            elif f.shape and _sf_byte_count(f.shape, f.strides, f.dtype) != op.span:
                raise ValueError(f"{spec.name}: {name} must contain {op.span} dense storage bytes")
        if not op.packed_tile_bytes and observed_span >= 0 and observed_span < op.span:
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
        # the operand's extent for the overlap test: a packed per-tile blob by its LIVE bytes -- its ``span`` is the plan's
        # capacity, which may run past the bound buffer into a caller workspace placed right after it
        if op.packed_tile_bytes and name in packed_tiles:
            extent = packed_tiles[name] * op.packed_tile_bytes
        else:
            extent = (f.numel if op.allowed_numels and f.shape and not raw_storage else op.span) * op.itemsize
        if workspace_ptr < f.ptr + extent and f.ptr < workspace_ptr + spec.workspace_bytes:
            raise ValueError(f"{spec.name}: caller workspace overlaps {name}")
        frame.append(f.ptr)
    scale = spec.scale if scale is None else float(scale)
    frame.append(workspace_ptr)
    if spec.scale_log2:
        frame.append(scale * math.log2(math.e))
    frame.append(scale)
    if spec.length_form:
        # Graph THD declarations carry B lengths. Standalone also accepts B+1
        # prefixes; their form is host metadata, never a device read.  The two
        # length operands are found by ROLE NAME: a slot index would be a hidden
        # coupling between every THD spec's operand order and this binder (a
        # differently ordered sibling spec would read a scalar's shape as a
        # length form, silently).
        form = 0
        if not raw_storage:
            for bit, name in enumerate(("seq_q", "seq_kv")):
                f = facts.get(name)
                if f is None or name not in spec.roles:
                    continue
                op = spec.operands[spec.roles.index(name)]
                if op is not None and f.numel == op.shape[0] + 1:
                    form |= 1 << bit
        frame.append(form)
    for group in spec.packed_tile_groups:
        # One packed SF tile count per group, derived above from the bound buffers; every role of the group must agree (the
        # forward's "sf_k and sf_v must have the same packed tile count").  0 live tiles is framed as 1 (a positive descriptor
        # extent the kernels' clamped maps never read).
        counts = {name: packed_tiles[name] for name in group if name in packed_tiles}
        if len(set(counts.values())) > 1:
            raise ValueError(f"{spec.name}: {' / '.join(group)} must share one packed SF tile count; got {counts}")
        frame.append(max(1, next(iter(counts.values()), 0)))
    frame.append(stream_int)
    return frame


def execute(spec, facts, workspace_ptr, stream_int, *, scale=None, geometry=None, raw_storage=False):
    frame = bind(spec, facts, workspace_ptr, stream_int, scale=scale, geometry=geometry, raw_storage=raw_storage)
    spec.fn(*frame)


class PreparedBwdLaunch:
    def __init__(self, spec, binding):
        self.spec = spec
        # A role listed in ``BwdLaunchSpec.standalone_only_roles`` is ABSENT on the graph path (framed as None by ``bind``):
        # the sm107 half row's ``delta`` slot (``prepared_sm107.EXTERNAL_DELTA_ROLE``), which no graph declares.  Every
        # other role is a ``SdpaBinding`` attribute, read strictly.
        tensors = [None if name in spec.standalone_only_roles else getattr(binding, name) for name in spec.attributes]
        self._roles = [name for name, tensor in zip(spec.roles, tensors) if tensor is not None]
        self._uids = [tensor.get_uid() for tensor in tensors if tensor is not None]
        self._geometry = tuple((tuple(t.get_dim()), tuple(t.get_stride())) if t is not None else None for t in tensors)
        self._indices = None
        self._native_indices = None
        self._native = None
        if spec.native_binding:
            from cudnn import _pybind_module

            self._native = _pybind_module._SdpaBwdBinder(spec, self._geometry)

    def execute(self, pack, workspace_ptr, stream, stream_int):
        if self._indices is None:
            self._indices = [pack.index_of(uid) for uid in self._uids]
        if self._native is not None:
            if self._native_indices is None:
                indices = dict(zip(self._roles, self._indices))
                self._native_indices = tuple(indices.get(role, -1) for role in self.spec.roles[: len(self.spec.operands)])
            self._native.execute(pack.native, self._native_indices, workspace_ptr, stream_int, tuple(pack.overridden))
            return
        facts = dict(zip(self._roles, facts_of_roles(pack, self._indices)))
        # Graph bindings are raw storage under the declared layout, including
        # strided producer views. Explicit overrides are different: they change
        # the requested operation, which this fixed backward artifact cannot do.
        geometry = None
        if pack.overridden:
            overridden = {role for role, index in zip(self._roles, self._indices) if index in pack.overridden}
            geometry = tuple(g if role in overridden else None for role, g in zip(self.spec.roles, self._geometry))
        execute(self.spec, facts, workspace_ptr, stream_int, geometry=geometry, raw_storage=True)
