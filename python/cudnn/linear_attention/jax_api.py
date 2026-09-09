# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-order JAX KDA using Frost's existing forward and backward kernels."""

from dataclasses import asdict, dataclass
from functools import lru_cache, partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import cudnn
from cudnn.jax.call import call, zeros_init

PRIMAL_NAMES = ("q", "k", "v", "g", "beta", "cu_seqlens", "initial_state", "a_log", "dt_bias")
GRADIENT_NAMES = ("dQ", "dK", "dV", "dG", "dBeta", "d_initial_state", "d_a_log", "d_dt_bias")


@dataclass(frozen=True)
class KdaConfig:
    output_final_state: bool = False
    scale: float | None = None
    use_qk_l2norm_in_kernel: bool = False
    use_beta_sigmoid_in_kernel: bool = False
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    gate_lower_bound: float | None = None
    batch_invariant: bool = False
    checkpoint_every_n_tokens: int = 0


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("primals", "checkpoints"),
    meta_fields=("config",),
)
@dataclass(frozen=True)
class KdaResidual:
    primals: tuple
    checkpoints: jax.Array | None
    config: KdaConfig


class KdaGradients(NamedTuple):
    dq: jax.Array
    dk: jax.Array
    dv: jax.Array
    dg: jax.Array
    dbeta: jax.Array
    d_initial_state: jax.Array | None
    d_a_log: jax.Array | None
    d_dt_bias: jax.Array | None


def validate_inputs(primals, config):
    q, k, v, g, beta, cu, state, a, dt = primals
    if any(x.ndim != 3 for x in (q, k, v)) or cu.ndim != 1:
        raise ValueError("KDA requires THD q/k/v and rank-one cu_seqlens")
    total, hq, dk = q.shape
    hv, dv = v.shape[1:]
    ho, batch = max(hq, hv), cu.shape[0] - 1
    if min(total, hq, hv, k.shape[1], batch) <= 0:
        raise ValueError("KDA requires positive token, head, and sequence counts")
    if total >= 2**31 or batch * ho >= 2**31:
        raise ValueError("KDA token and scheduler counts must fit signed int32")
    if dk not in (64, 128) or dv not in (64, 128):
        raise NotImplementedError("JAX KDA supports head dimensions 64 and 128")
    if ho % min(hq, hv) or ho // min(hq, hv) not in (1, 2, 4, 8) or k.shape[1] not in (hq, hv):
        raise NotImplementedError("JAX KDA supports head ratios 1/2/4/8 with HK=HQ or HV")
    expected = (
        (k, (total, k.shape[1], dk), "k"),
        (v, (total, hv, dv), "v"),
        (g, (total, ho, dk), "g"),
        (beta, (total, ho), "beta"),
        (state, (batch, ho, dv, dk), "initial_state"),
        (a, (ho,), "a_log"),
        (dt, (ho, dk), "dt_bias"),
    )
    for value, shape, name in expected:
        if value is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    if q.dtype not in (jnp.float16, jnp.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have matching float16 or bfloat16 dtype")
    if cu.dtype != jnp.int32:
        raise ValueError("cu_seqlens must be int32")
    if config.checkpoint_every_n_tokens not in (0, 16):
        raise NotImplementedError("JAX KDA supports checkpoint cadence 0 or 16")
    if config.allow_neg_eigval and not config.use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if not config.safe_gate and (a is not None or dt is not None or config.gate_lower_bound is not None):
        raise ValueError("gate parameters require safe_gate=True")
    if config.gate_lower_bound is not None and not -5 <= config.gate_lower_bound < 0:
        raise ValueError("gate_lower_bound must be in [-5, 0)")
    if config.scale is not None and not math.isfinite(config.scale):
        raise ValueError("scale must be finite")


def target_device():
    devices = jax.local_devices(backend="gpu")
    if len(devices) != 1:
        raise NotImplementedError("Draft JAX KDA requires one visible GPU; set CUDA_VISIBLE_DEVICES")
    device = devices[0].local_hardware_id
    from cudnn.frost.device import compute_capability

    major, minor = compute_capability(device)
    sm = major * 10 + minor
    if sm not in (100, 103):
        raise NotImplementedError(f"JAX KDA requires SM100 or SM103; found {sm}")
    return device


@lru_cache(maxsize=128)
def build_call(metadata, config, device):
    from cudnn.frost.device import build_device
    from .frost.kda_engine import KdaFrostEngine, build_kda
    from cutlass.jax import TensorSpec

    data_types = dict(float16=cudnn.data_type.HALF, bfloat16=cudnn.data_type.BFLOAT16, float32=cudnn.data_type.FLOAT, int32=cudnn.data_type.INT32)
    graph = cudnn.pygraph()
    tensors = {}
    for name, shape, dtype in metadata:
        if dtype not in data_types:
            raise ValueError(f"unsupported KDA dtype: {dtype}")
        tensors[name] = graph.tensor(shape, data_type=data_types[dtype], name=name)
    attributes = asdict(config)
    attributes["use_qk_l2norm"] = attributes.pop("use_qk_l2norm_in_kernel")
    attributes["use_beta_sigmoid"] = attributes.pop("use_beta_sigmoid_in_kernel")
    if "dO" in tensors:
        attributes.pop("output_final_state")
        attributes["checkpoint_every_n_tokens"] = config.checkpoint_every_n_tokens or None
        graph.kda_bwd(**tensors, **attributes)
        output_sources = dict(zip(GRADIENT_NAMES, PRIMAL_NAMES[:5] + PRIMAL_NAMES[6:]))
    else:
        graph.kda(**tensors, **attributes)
        output_sources = dict(O="q", final_state="initial_state", state_checkpoints="q")
    node = graph.nodes[0]
    for name, tensor in node.outputs.items():
        source = tensors.get(output_sources[name])
        dtype = source.get_data_type() if source is not None else cudnn.data_type.FLOAT
        tensor.set_output(True).set_data_type(dtype)
    with build_device(device):
        graph.validate()
        KdaFrostEngine().check_support(graph)
        plan = build_kda(graph)
    from .frost.kda_launch import make_launcher

    input_names, output_names = tuple(node.inputs), tuple(node.outputs)
    reverse_dtype = {v: k for k, v in data_types.items()}
    shapes = tuple(jax.ShapeDtypeStruct(tuple(t.dim), reverse_dtype[t.get_data_type()]) for t in node.outputs.values())
    workspace = jax.ShapeDtypeStruct((plan.workspace_bytes(),), jnp.uint8)
    initialized = {i: zeros_init for i, name in enumerate(output_names) if name == "state_checkpoints"}

    def specs(shapes):
        return tuple(TensorSpec(layout=tuple(reversed(range(len(s))))) for s in shapes)

    invoke = call(
        make_launcher(plan, input_names, output_names),
        output_shape_dtype=shapes + (workspace,),
        input_spec=specs([t.dim for t in node.inputs.values()]),
        output_spec=specs([s.shape for s in shapes + (workspace,)]),
        initialized_outputs=initialized,
        use_static_tensors=True,
    )
    return invoke, input_names, output_names


def execute(primals, config, **backward_inputs):
    values = dict(zip(PRIMAL_NAMES, primals))
    values.update(backward_inputs)
    metadata = tuple((name, tuple(value.shape), np.dtype(value.dtype).name) for name, value in values.items() if value is not None)
    invoke, inputs, outputs = build_call(metadata, config, target_device())
    return dict(zip(outputs, invoke(*(values[name] for name in inputs))[:-1]))


def forward(primals, config):
    validate_inputs(primals, config)
    results = execute(primals, config)
    return results["O"], results.get("final_state"), KdaResidual(primals, results.get("state_checkpoints"), config)


def kimi_delta_attention_fwd(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
):
    """Return ``(output, final_state_or_None, residual)``; see the JAX KDA guide."""
    config = KdaConfig(
        output_final_state,
        scale,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        batch_invariant,
        checkpoint_every_n_tokens,
    )
    return forward((q, k, v, g, beta, cu_seqlens, initial_state, a_log, dt_bias), config)


def kimi_delta_attention_bwd(residual, doutput, *, d_final_state=None):
    """Explicit first-order backward. State and gate parameter gradients are included."""
    primals, config = residual.primals, residual.config
    q, k, v, g, beta, cu, state, a, dt = primals
    expected = (q.shape[0], g.shape[1], v.shape[2])
    if doutput.shape != expected or doutput.dtype != q.dtype:
        raise ValueError(f"doutput must have shape {expected} and dtype {q.dtype}")
    if d_final_state is not None:
        if not config.output_final_state:
            raise ValueError("d_final_state requires output_final_state=True in forward")
        shape = (cu.shape[0] - 1, g.shape[1], v.shape[2], q.shape[2])
        if d_final_state.shape != shape or d_final_state.dtype != (state.dtype if state is not None else jnp.float32):
            raise ValueError("d_final_state must match the forward final state shape and dtype")
    results = execute(primals, config, dO=doutput, d_final_state=d_final_state, state_checkpoints=residual.checkpoints)
    return KdaGradients(*(results.get(name) for name in GRADIENT_NAMES))


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def differentiable(primals, config):
    return forward(primals, config)[:2]


def vjp_forward(primals, config):
    o, fs, residual = forward(primals, config)
    return (o, fs), residual


def vjp_backward(config, residual, cotangents):
    grads = kimi_delta_attention_bwd(residual, cotangents[0], d_final_state=cotangents[1])
    return ((*grads[:5], None, *grads[5:]),)


differentiable.defvjp(vjp_forward, vjp_backward)


def kimi_delta_attention(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    *,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    output_final_state=False,
    scale=None,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    batch_invariant=False,
    checkpoint_every_n_tokens=0,
):
    """KDA on packed THD arrays; return ``(output, final_state_or_None)``.

    Supports jit and first-order reverse-mode AD. cu_seqlens must start at zero,
    end at T, and be nondecreasing; these device values are not read on the host.
    Options are static under jit. Forward AD, higher derivatives, vmap and
    distributed execution are not supported by this draft.
    """
    config = KdaConfig(
        output_final_state,
        scale,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        batch_invariant,
        checkpoint_every_n_tokens,
    )
    return differentiable((q, k, v, g, beta, cu_seqlens, initial_state, a_log, dt_bias), config)
