# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed SM80 dense declarations and per-call pointer binding.

The graph and standalone adapter share this binder. No runtime tensor views,
layout copies, allocation, compilation or stream state belong to the spec.
"""

from dataclasses import dataclass
import math

from cudnn.frost.compiled_cache import positional_entry
from cudnn.sdpa.fwd.prepared import facts_of_roles

ROLES = ("q", "k", "v", "o", "stats", "seq_kv", "seq_q", "sink", "bias", "rope")


@dataclass(frozen=True)
class Operand:
    shape: tuple
    strides: tuple
    dtype: str
    span: int
    alignment: int
    contiguous: bool


@dataclass(frozen=True)
class LaunchSpec:
    artifact: object
    fn: object
    operands: tuple
    device_index: int
    scale: float


def native_layouts(api):
    """Admit declarations the vector loads/stores can serve without staging."""
    return (
        not api._rope_max_s
        and api.head_dim_v == api.flavor_d_v
        and api.head_dim_qk % 8 == 0
        and all(all(n == 1 or st % 8 == 0 for n, st in zip(desc.shape[:-1], desc.stride[:-1])) for desc in (api.q_desc, api.k_desc, api.v_desc, api.o_desc))
    )


def build_spec(api, *, compiler=None):
    from cudnn.frost.buffers import is_contiguous
    from cudnn.frost.compiled_cache import template_key
    from cudnn.sdpa.fwd.kernels.sm80.prepared_host import compile_host

    operands, geometry = [], []

    def add(shape, strides, dtype, alignment, *, bshd=False):
        shape, strides = tuple(shape), tuple(strides)
        span = 1 + sum((int(n) - 1) * int(st) for n, st in zip(shape, strides))
        operands.append(Operand(shape, strides, str(dtype).split(".")[-1], span, alignment, is_contiguous(shape, strides)))
        order = (0, 2, 1, 3) if bshd else tuple(range(len(shape)))
        geometry.append((tuple(shape[i] for i in order), tuple(strides[i] for i in order)))

    for desc in (api.q_desc, api.k_desc, api.v_desc, api.o_desc):
        add(desc.shape, desc.stride, desc.dtype, 16, bshd=True)
    if api.lse_desc is not None:
        add(api.lse_desc.shape, api.lse_desc.stride, "float32", 4)
    else:
        operands.append(None)
        geometry.append(None)
    for enabled, count, dtype in (
        (api.seq_kv_lens_present, api.batch_size, "int32"),
        (api.seq_q_lens_present, api.batch_size, "int32"),
        (api.has_sink, api.h_q, "float32"),
    ):
        if enabled:
            add((count,), (1,), dtype, 4)
        else:
            operands.append(None)
            geometry.append(None)
    if api._bias_present:
        shape = (1, api.h_q, api.s_q_max, api.s_k_max)
        strides = (api.h_q * api.s_q_max * api.s_k_max, api.s_q_max * api.s_k_max, api.s_k_max, 1)
        add(shape, strides, "float32" if api._bias_fp32 else api.dtype, 4 if api._bias_fp32 else 2)
    else:
        operands.append(None)
        geometry.append(None)
    if api._rope_max_s:
        add((api._rope_max_s, api.flavor_d_qk // 2, 2), (api.flavor_d_qk, 2, 1), "float32", 4)
    geometry = tuple(geometry)
    key = template_key(
        vars(api._k_mod),
        dict(geometry=geometry, swa_window=api.swa_window_runtime, right_bound=api.right_bound_runtime),
        "prepared_dense",
    )
    artifact = (compile_host if compiler is None else compiler)(api._k_mod, api._params, geometry, api.swa_window_runtime, api.right_bound_runtime, key)
    fn = positional_entry(artifact)
    if fn is None:
        raise NotImplementedError("SM80 prepared forward requires a positional tvm-ffi entry")
    return LaunchSpec(artifact, fn, tuple(operands), int(api.q_desc.device.index or 0), api.scale_softmax)


def _same_geometry(actual, expected):
    # Graph Stats and vector operands may carry extra singleton dimensions.
    return tuple((n, st) for n, st in zip(*actual) if n != 1) == tuple((n, st) for n, st in zip(*expected) if n != 1)


def bind(spec, facts, stream_int, *, scale=None, overridden=None, raw_storage=False):
    frame = []
    for name, op in zip(ROLES, spec.operands):
        f = facts.get(name)
        if op is None:
            if f is not None:
                raise ValueError(f"sdpa_fwd_sm80: {name} was not compiled into this specialization")
            frame.append(None)
            continue
        if f is None:
            raise ValueError(f"sdpa_fwd_sm80: {name} is required by this specialization")
        if f.device not in ((2, spec.device_index), (-1, -1)):
            raise ValueError(f"sdpa_fwd_sm80: {name} must be on CUDA device {spec.device_index}")
        if f.dtype and f.dtype != op.dtype:
            raise ValueError(f"sdpa_fwd_sm80: {name} must be {op.dtype}; got {f.dtype}")
        if not f.ptr or f.ptr % op.alignment:
            raise ValueError(f"sdpa_fwd_sm80: {name} base address must be {op.alignment}-byte aligned")
        if f.span >= 0 and f.span < op.span:
            raise ValueError(f"sdpa_fwd_sm80: {name} backing storage is too small for the declared strides")
        if not raw_storage and name in ("seq_q", "seq_kv", "sink") and f.shape:
            if not f.contiguous or f.numel != math.prod(op.shape):
                raise ValueError(f"sdpa_fwd_sm80: {name} must be contiguous with {math.prod(op.shape)} elements")
        elif not raw_storage and name == "stats" and f.shape:
            # Preserve the standalone Stats storage contract: compact plans may
            # bind a flat contiguous buffer; strided plans use declared strides.
            if f.numel != math.prod(op.shape) or (op.contiguous and not f.contiguous):
                raise ValueError("sdpa_fwd_sm80: Stats must match the declared element count and storage layout")
        elif not raw_storage and name == "bias" and f.shape:
            # The standalone ABI broadcasts the first contiguous [H,SQ,SKV]
            # bias plane; retain that view semantics without constructing it.
            if len(f.shape) != 4 or f.shape[0] < 1 or not _same_geometry((f.shape[1:], f.strides[1:]), (op.shape[1:], op.strides[1:])):
                raise ValueError("sdpa_fwd_sm80: bias must have a contiguous [H,SQ,SKV] first plane")
        elif (not raw_storage or (overridden is not None and name in overridden)) and f.shape:
            if not _same_geometry((f.shape, f.strides), (op.shape, op.strides)):
                raise ValueError(f"sdpa_fwd_sm80: {name} runtime geometry must match this fixed forward plan")
        frame.append(f.ptr)
    scale = spec.scale if scale is None or scale == 0 else float(scale)
    frame.extend((scale * math.log2(math.e), 1.0 / scale, stream_int))
    return frame


def execute(spec, facts, stream_int, *, scale=None, overridden=None, raw_storage=False):
    spec.fn(*bind(spec, facts, stream_int, scale=scale, overridden=overridden, raw_storage=raw_storage))


class PreparedSm80Launch:
    def __init__(self, spec, binding):
        self.spec = spec
        attributes = ("q", "k", "v", "o", "stats", "seq_len_kv", "seq_len_q", "sink_token", "bias")
        tensors = [getattr(binding, name) for name in attributes]
        self._roles = [name for name, t in zip(ROLES, tensors) if t is not None]
        self._uids = [t.get_uid() for t in tensors if t is not None]
        self._indices = None

    def execute(self, pack, workspace_ptr, stream, stream_int):
        if self._indices is None:
            self._indices = [pack.index_of(uid) for uid in self._uids]
        facts = dict(zip(self._roles, facts_of_roles(pack, self._indices)))
        overridden = {role for role, index in zip(self._roles, self._indices) if index in pack.overridden}
        execute(self.spec, facts, stream_int, overridden=overridden, raw_storage=True)
