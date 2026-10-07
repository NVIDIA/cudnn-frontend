# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed SM80 dense declarations and per-call pointer binding.

The graph and standalone adapter share this binder. No runtime tensor views,
layout copies, allocation, compilation or stream state belong to the spec.
"""

from dataclasses import dataclass, field

from cudnn.frost.compiled_cache import positional_entry

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
    native: object = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        from cudnn import _pybind_module

        object.__setattr__(self, "native", _pybind_module._SdpaSm80FwdBinder(self))


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


def bind(spec, facts, stream_int, *, scale=None, overridden=None, raw_storage=False):
    """Bind staged metadata through the template's sole native contract."""
    from .prepared import _native_pack_from_facts

    roles = ROLES[: len(spec.operands)]
    pack = _native_pack_from_facts(facts, roles)
    overridden_indices = tuple(i for i, role in enumerate(roles) if overridden is not None and role in overridden)
    return list(spec.native.bind(pack, tuple(range(len(roles))), stream_int, scale, overridden_indices, raw_storage))


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
        self._native_indices = None

    def execute(self, pack, workspace_ptr, stream, stream_int):
        if self._indices is None:
            self._indices = [pack.index_of(uid) for uid in self._uids]
        if self._native_indices is None:
            roles = dict(zip(self._roles, self._indices))
            self._native_indices = tuple(roles.get(role, -1) for role in ROLES[: len(self.spec.operands)])
        self.spec.native.execute(pack.native, self._native_indices, stream_int, None, tuple(pack.overridden), True)


def execute_tensors(spec, buffers, stream_int, *, scale=None):
    """Observe current standalone carriers once; keep their stricter contract."""
    from cudnn import _pybind_module
    from cudnn.sdpa.fwd.prepared import _set_native_fact, facts_of_tensor

    pack, unread = _pybind_module._read_buffer_sequence(buffers)
    for index in unread:
        _set_native_fact(pack, index, facts_of_tensor(buffers[index]))
    spec.native.execute(pack, tuple(range(len(spec.operands))), stream_int, scale, (), False)
