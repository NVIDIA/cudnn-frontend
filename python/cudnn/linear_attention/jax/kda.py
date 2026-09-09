# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-order JAX KDA using Frost's existing forward and backward kernels."""

from dataclasses import dataclass
from functools import lru_cache, partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import cudnn
from cudnn.jax.call import call, zeros_init
from ..kda_graph import build_fprop_graph, build_bprop_graph


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


def tensor_metadata(tensor):
    return None if tensor is None else (tuple(tensor.shape), np.dtype(tensor.dtype).name)


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
def build_call(
    metadata,
    config,
    device,
    backward=False,
    dfs_meta=None,
    checkpoints_meta=None,
):
    from cudnn.frost.device import build_device
    from ..frost.kda_engine import KdaFrostEngine, build_kda

    q, k, v, g, beta, cu, state, a, dt = metadata
    total, hq, dk = q[0]
    hv, dv = v[0][1:]
    dtype_map = dict(
        float16=cudnn.data_type.HALF,
        bfloat16=cudnn.data_type.BFLOAT16,
        float32=cudnn.data_type.FLOAT,
        int32=cudnn.data_type.INT32,
    )

    def dtype(meta):
        if meta is None:
            return None
        if meta[1] not in dtype_map:
            raise ValueError(f"unsupported KDA dtype: {meta[1]}")
        return dtype_map[meta[1]]

    kwargs = dict(
        total=total,
        N=cu[0][0] - 1,
        H=hq,
        HK=k[0][1],
        HV=hv,
        K=dk,
        V=dv,
        io_dtype=dtype(q),
        g_dtype=dtype(g),
        beta_dtype=dtype(beta),
        state_dtype=dtype(state),
        cu_dtype=dtype(cu),
        scale=config.scale,
        use_qk_l2norm=config.use_qk_l2norm_in_kernel,
        batch_invariant=config.batch_invariant,
        use_beta_sigmoid=config.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=config.allow_neg_eigval,
        safe_gate=config.safe_gate,
        gate_lower_bound=config.gate_lower_bound,
        a_log_dtype=dtype(a),
        dt_bias_dtype=dtype(dt),
    )
    if backward:
        graph, _ = build_bprop_graph(
            **kwargs,
            dstate_in_dtype=dtype(dfs_meta),
            checkpoint_rows=checkpoints_meta[0][0] if checkpoints_meta else None,
            checkpoint_every_n_tokens=config.checkpoint_every_n_tokens,
        )
    else:
        graph, _ = build_fprop_graph(
            **kwargs,
            output_final_state=config.output_final_state,
            checkpoint=config.checkpoint_every_n_tokens,
        )
    node = graph.nodes[0]
    output_dtypes = dict(
        O=dtype(q),
        final_state=dtype(state) or cudnn.data_type.FLOAT,
        state_checkpoints=dtype(q),
        dQ=dtype(q),
        dK=dtype(k),
        dV=dtype(v),
        dG=dtype(g),
        dBeta=dtype(beta),
        d_initial_state=dtype(state),
        d_a_log=dtype(a),
        d_dt_bias=dtype(dt),
    )
    for name, tensor in node.outputs.items():
        tensor.set_output(True).set_data_type(output_dtypes[name])
    with build_device(device):
        graph.validate()
        KdaFrostEngine().check_support(graph)
        plan = build_kda(graph)
    from ..frost.kda_launch import make_launcher

    input_names, output_names = tuple(node.inputs), tuple(node.outputs)
    reverse_dtype = {v: k for k, v in dtype_map.items()}
    shapes = tuple(jax.ShapeDtypeStruct(tuple(t.dim), reverse_dtype[t.get_data_type()]) for t in node.outputs.values())
    workspace = jax.ShapeDtypeStruct((plan.workspace_bytes(),), jnp.uint8)
    initialized = {i: zeros_init for i, name in enumerate(output_names) if name == "state_checkpoints"}
    from cutlass.jax import TensorSpec

    in_shapes = [tuple(t.dim) for t in node.inputs.values()]

    def specs(shapes):
        return tuple(TensorSpec(layout=tuple(reversed(range(len(s))))) for s in shapes)

    invoke = call(
        make_launcher(plan, input_names, output_names),
        output_shape_dtype=shapes + (workspace,),
        input_spec=specs(in_shapes),
        output_spec=specs([s.shape for s in shapes + (workspace,)]),
        initialized_outputs=initialized,
        use_static_tensors=True,
    )
    return invoke, input_names, output_names


def forward(primals, config):
    validate_inputs(primals, config)
    invoke, inputs, outputs = build_call(tuple(map(tensor_metadata, primals)), config, target_device())
    values = dict(
        zip(
            (
                "q",
                "k",
                "v",
                "g",
                "beta",
                "cu_seqlens",
                "initial_state",
                "a_log",
                "dt_bias",
            ),
            primals,
        )
    )
    results = dict(zip(outputs, invoke(*(values[n] for n in inputs))[:-1]))
    residual = KdaResidual(primals, results.get("state_checkpoints"), config)
    return results["O"], results.get("final_state"), residual


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
    invoke, inputs, outputs = build_call(
        tuple(map(tensor_metadata, primals)),
        config,
        target_device(),
        backward=True,
        dfs_meta=tensor_metadata(d_final_state),
        checkpoints_meta=tensor_metadata(residual.checkpoints),
    )
    values = dict(
        zip(
            (
                "q",
                "k",
                "v",
                "g",
                "beta",
                "cu_seqlens",
                "initial_state",
                "a_log",
                "dt_bias",
            ),
            primals,
        )
    )
    values.update(dO=doutput, d_final_state=d_final_state, state_checkpoints=residual.checkpoints)
    results = dict(zip(outputs, invoke(*(values[n] for n in inputs))[:-1]))
    return KdaGradients(
        *(
            results.get(n)
            for n in (
                "dQ",
                "dK",
                "dV",
                "dG",
                "dBeta",
                "d_initial_state",
                "d_a_log",
                "d_dt_bias",
            )
        )
    )


@partial(jax.custom_vjp, nondiff_argnums=(9,))
def differentiable(q, k, v, g, beta, cu, state, a, dt, config):
    return forward((q, k, v, g, beta, cu, state, a, dt), config)[:2]


def vjp_forward(q, k, v, g, beta, cu, state, a, dt, config):
    o, fs, residual = forward((q, k, v, g, beta, cu, state, a, dt), config)
    return (o, fs), residual


def vjp_backward(config, residual, cotangents):
    grads = kimi_delta_attention_bwd(residual, cotangents[0], d_final_state=cotangents[1])
    return (*grads[:5], None, *grads[5:])


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
    return differentiable(q, k, v, g, beta, cu_seqlens, initial_state, a_log, dt_bias, config)
