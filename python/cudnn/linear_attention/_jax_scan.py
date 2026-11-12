# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared JAX execution and differentiation for native FROST scans."""

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache, partial
import math
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import cudnn
from cudnn.jax import call, zeros_init


class _Operation(Enum):
    GDN = "gdn"
    GDN2 = "gdn2"
    GDP = "gdp"
    MAMBA2 = "mamba2"


@dataclass(frozen=True)
class _DeltaConfig:
    operation: _Operation
    output_final_state: bool = False
    scale: float | None = None
    use_qk_l2norm: bool = False
    use_beta_sigmoid: bool = False
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    gate_domain: str = "log"
    batch_invariant: bool = False
    checkpoint_every_n_tokens: int = 0
    num_householder: int = 1
    beta_guard: bool = False
    gate_lower_bound: float | None = None


@dataclass(frozen=True)
class _MambaConfig:
    output_final_state: bool = False
    checkpoint_every_n_tokens: int = 0
    dt_softplus: bool = True
    dt_min: float = 0.0
    dt_max: float | None = None
    batch_invariant: bool = False

    @property
    def operation(self):
        return _Operation.MAMBA2


@partial(jax.tree_util.register_dataclass, data_fields=("primals", "checkpoints", "pre_gate"), meta_fields=("config",))
@dataclass(frozen=True)
class ScanResidual:
    primals: tuple
    checkpoints: jax.Array | None
    pre_gate: jax.Array | None
    config: _DeltaConfig | _MambaConfig


class DeltaGradients(NamedTuple):
    dq: jax.Array
    dk: jax.Array
    dv: jax.Array
    dg: jax.Array
    dbeta: jax.Array
    dw: jax.Array | None
    d_initial_state: jax.Array | None
    d_a_log: jax.Array | None
    d_dt_bias: jax.Array | None


class Mamba2Gradients(NamedTuple):
    dx: jax.Array
    ddt: jax.Array
    da: jax.Array
    db: jax.Array
    dc: jax.Array
    d_initial_state: jax.Array | None
    dd: jax.Array | None
    d_dt_bias: jax.Array | None
    dz: jax.Array | None


def _ports(primals, config):
    if config.operation == _Operation.MAMBA2:
        names = ("v", "g", "a", "k", "q", "cu_seqlens", "initial_state", "d", "dt_bias", "z")
    else:
        names = ("q", "k", "v", "g", "beta", "w", "cu_seqlens", "initial_state", "a_log", "dt_bias")
    return dict(zip(names, primals, strict=True))


def _validate(primals, config):
    values = _ports(primals, config)
    required = ("q", "k", "v", "g", "cu_seqlens", "a") if config.operation == _Operation.MAMBA2 else ("q", "k", "v", "g", "beta", "cu_seqlens")
    if config.operation == _Operation.GDN2:
        required += ("w",)
    for name in required:
        if values[name] is None:
            raise TypeError(f"{name} must be a JAX array")
    for name, value in values.items():
        if value is not None and not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX array")
    q, k, v, g, cu = (values[name] for name in ("q", "k", "v", "g", "cu_seqlens"))
    if any(value.ndim != 3 for value in (q, k, v)) or cu.ndim != 1 or cu.shape[0] < 2:
        raise ValueError("linear attention requires THD q/k/v and rank-one cu_seqlens with at least two entries")
    total, hq, dk = q.shape
    hv, dv = v.shape[1:]
    ho, batch = max(hq, hv), cu.shape[0] - 1
    if min(total, hq, k.shape[1], hv, dk, dv) <= 0 or total >= 2**31 or batch * ho >= 2**31:
        raise ValueError("positive token/head dimensions and signed-int32 token/scheduler counts are required")
    if any(ho % heads or ho // heads not in (1, 2, 4, 8) for heads in (hq, k.shape[1], hv)):
        raise NotImplementedError("JAX linear attention supports head ratios 1/2/4/8")
    expanded = config.num_householder if config.operation == _Operation.GDP else 1
    if not isinstance(expanded, Integral) or isinstance(expanded, bool) or expanded <= 0 or total * expanded >= 2**31:
        raise ValueError("num_householder must be positive and expanded token counts must fit signed int32")
    expected = {
        "k": (total * expanded, k.shape[1], dk),
        "v": (total * expanded, hv, dv),
        "g": (total, ho, dk) if config.operation == _Operation.GDN2 else (total, ho),
        "beta": (total, ho, dk) if config.operation == _Operation.GDN2 else (total * expanded, ho),
        "w": (total, ho, dv),
        "initial_state": (batch, ho, dv, dk),
        "a_log": (ho,),
        "dt_bias": (ho, dk) if config.operation == _Operation.GDN2 else (ho,),
        "a": (ho,),
        "z": (total, ho, dv),
    }
    for name, shape in expected.items():
        value = values.get(name)
        if value is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    if values.get("d") is not None and values["d"].shape not in ((ho,), (ho, dv)):
        raise ValueError("D must have shape [heads] or [heads, value_dim]")
    if cu.dtype not in (jnp.int32, jnp.int64):
        raise ValueError("cu_seqlens must be int32 or int64")
    cadence = config.checkpoint_every_n_tokens
    if not isinstance(cadence, Integral) or isinstance(cadence, bool) or not 0 <= cadence < 2**31:
        raise ValueError("checkpoint_every_n_tokens must be a nonnegative signed-int32 integer")
    if isinstance(config, _DeltaConfig):
        if config.scale is not None and not math.isfinite(config.scale):
            raise ValueError("scale must be finite")
        if config.gate_domain not in ("log", "linear") or (config.safe_gate and config.gate_domain != "log"):
            raise ValueError("gate_domain must be log or linear; safe_gate requires log")
        if config.operation == _Operation.GDN2 and config.gate_lower_bound is not None and not -5 <= config.gate_lower_bound < 0:
            raise ValueError("gate_lower_bound must be in [-5, 0)")
    elif not math.isfinite(config.dt_min) or config.dt_min < 0 or (config.dt_max is not None and (math.isnan(config.dt_max) or config.dt_max < config.dt_min)):
        raise ValueError("dt bounds must satisfy 0 <= dt_min <= dt_max")


def _device(values):
    placed = {device for value in values if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer) for device in value.devices()}
    if len(placed) > 1:
        raise ValueError("linear-attention arrays must be on the same GPU")
    target = next(iter(placed)) if placed else (jax.config.jax_default_device or jax.local_devices()[0])
    if target.platform != "gpu":
        raise NotImplementedError(f"JAX linear attention requires a CUDA GPU; got {target.platform}")
    from cudnn.frost.device import compute_capability

    device = target.local_hardware_id
    capability = compute_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError(f"JAX linear attention requires SM100 or SM103; got {capability}")
    return device, f"sm_{capability[0]}{capability[1]}a"


def _attributes(config, backward):
    attrs = asdict(config)
    attrs.pop("operation", None)
    if isinstance(config, _DeltaConfig):
        if config.operation != _Operation.GDP:
            attrs.pop("num_householder")
        if config.operation != _Operation.GDN2:
            attrs.pop("beta_guard")
            attrs.pop("gate_lower_bound")
    if backward:
        attrs.pop("output_final_state")
        attrs["checkpoint_every_n_tokens"] = config.checkpoint_every_n_tokens or None
    return attrs


@lru_cache(maxsize=128)
def _build(metadata, config, device, gpu_arch):
    from cudnn.frost.device import build_device

    operation = config.operation
    if operation in (_Operation.GDN, _Operation.GDP):
        from .frost.gdn_engine import GdnFrostEngine, build_gdn
        from .frost.gdp_engine import GdpFrostEngine
        from .frost.gdn_jax import make_launcher

        engine, builder = (GdpFrostEngine if operation == _Operation.GDP else GdnFrostEngine), build_gdn
    elif operation == _Operation.GDN2:
        from .frost.gdn2_engine import Gdn2FrostEngine, build_gdn2
        from .frost.gdn2_jax import make_launcher

        engine, builder = Gdn2FrostEngine, build_gdn2
    else:
        from .frost.mamba2_engine import Mamba2FrostEngine, build_mamba2
        from .frost.mamba2_jax import make_launcher

        engine, builder = Mamba2FrostEngine, build_mamba2
    data_types = dict(
        float16=cudnn.data_type.HALF, bfloat16=cudnn.data_type.BFLOAT16, float32=cudnn.data_type.FLOAT, int32=cudnn.data_type.INT32, int64=cudnn.data_type.INT64
    )
    graph = cudnn.pygraph()
    tensors = {}
    for name, shape, dtype in metadata:
        if dtype not in data_types:
            raise ValueError(f"unsupported linear-attention dtype: {dtype}")
        tensors[name] = graph.tensor(shape, data_type=data_types[dtype], name=name)
    backward = "dO" in tensors
    attrs = _attributes(config, backward)
    if operation == _Operation.MAMBA2 and not backward:
        attrs["output_pre_gate"] = "z" in tensors
    getattr(graph, operation.value + ("_bwd" if backward else ""))(**tensors, **attrs)
    node = graph.nodes[0]
    sources = dict(
        dQ="q", dK="k", dV="v", dG="g", dBeta="beta", dW="w", d_initial_state="initial_state", d_a_log="a_log", d_dt_bias="dt_bias", dA="a", dD="d", dZ="z"
    )
    sources.update(O="q", O_pre="q", final_state="initial_state", state_checkpoints="q")
    for name, tensor in node.outputs.items():
        source = tensors.get(sources.get(name))
        tensor.set_output(True).set_data_type(source.get_data_type() if source is not None else cudnn.data_type.FLOAT)
    with build_device(device):
        graph.validate()
        engine().check_support(graph)
        plan = builder(graph)
        launcher = make_launcher(plan, tuple(node.inputs) + tuple(node.outputs))
    reverse_dtype = {value: key for key, value in data_types.items()}
    shapes = tuple(jax.ShapeDtypeStruct(tuple(t.dim), reverse_dtype[t.get_data_type()]) for t in node.outputs.values())
    workspace = jax.ShapeDtypeStruct((plan.workspace_bytes(),), jnp.uint8)
    initialized = {i: zeros_init for i, name in enumerate(node.outputs) if name == "state_checkpoints"}
    invoke = call(
        launcher, output_shape_dtype=shapes + (workspace,), initialized_outputs=initialized, use_static_tensors=True, compile_options=f"--gpu-arch={gpu_arch}"
    )
    return invoke, tuple(node.inputs), tuple(node.outputs)


def _execute(primals, config, **backward):
    values = _ports(primals, config)
    values.update(backward)
    metadata = tuple((name, tuple(value.shape), np.dtype(value.dtype).name) for name, value in values.items() if value is not None)
    invoke, inputs, outputs = _build(metadata, config, *_device(values.values()))
    return dict(zip(outputs, invoke(*(values[name] for name in inputs))[:-1], strict=True))


def _forward(primals, config):
    _validate(primals, config)
    results = _execute(primals, config)
    residual = ScanResidual(primals, results.get("state_checkpoints"), results.get("O_pre"), config)
    return results["O"], results.get("final_state"), residual


def _backward(residual, doutput, d_final_state):
    config = residual.config
    values = _ports(residual.primals, config)
    q, v, cu = (values[name] for name in ("q", "v", "cu_seqlens"))
    shape = (q.shape[0], max(q.shape[1], v.shape[1]), v.shape[-1])
    if doutput.shape != shape or doutput.dtype != q.dtype:
        raise ValueError(f"doutput must have shape {shape} and dtype {q.dtype}")
    if d_final_state is not None:
        state = values["initial_state"]
        expected = (cu.shape[0] - 1, shape[1], v.shape[-1], q.shape[-1])
        if not config.output_final_state or d_final_state.shape != expected or d_final_state.dtype != (jnp.float32 if state is None else state.dtype):
            raise ValueError("d_final_state must match a requested forward final state")
    results = _execute(
        residual.primals,
        config,
        dO=doutput,
        d_final_state=d_final_state,
        state_checkpoints=residual.checkpoints,
        **({"O_pre": residual.pre_gate} if residual.pre_gate is not None else {}),
    )
    if config.operation == _Operation.MAMBA2:
        names = ("dV", "dG", "dA", "dK", "dQ", "d_initial_state", "dD", "d_dt_bias", "dZ")
        return Mamba2Gradients(*(results.get(name) for name in names))
    names = ("dQ", "dK", "dV", "dG", "dBeta", "dW", "d_initial_state", "d_a_log", "d_dt_bias")
    return DeltaGradients(*(results.get(name) for name in names))


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _differentiable(primals, config):
    return _forward(primals, config)[:2]


def _vjp_forward(primals, config):
    output, final, residual = _forward(primals, config)
    return (output, final), residual


def _vjp_backward(config, residual, cotangents):
    grads = _backward(residual, cotangents[0], cotangents[1])
    if config.operation == _Operation.MAMBA2:
        return ((*grads[:5], None, grads.d_initial_state, grads.dd, grads.d_dt_bias, grads.dz),)
    return ((*grads[:6], None, *grads[6:]),)


_differentiable.defvjp(_vjp_forward, _vjp_backward)
