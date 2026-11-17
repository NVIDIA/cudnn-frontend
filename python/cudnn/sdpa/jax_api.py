# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX SDPA using Python graph construction and native cuDNN execution."""

from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as np

import cudnn
from .jax_runtime import make_call


def _spec(x):
    return None if x is None else (tuple(x.shape), str(x.dtype))


@lru_cache(maxsize=128)
def _plan(specs, scale, causal, window, deterministic, backward, has_dbias, device):
    from cudnn.frost.device import build_device

    with build_device(device):
        return _build_plan(specs, scale, causal, window, deterministic, backward, has_dbias)


def _build_plan(specs, scale, causal, window, deterministic, backward, has_dbias):
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16 if specs[0][1] == "bfloat16" else cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    inputs = []

    def tensor(spec, name):
        if spec is None:
            return None
        shape, dtype = spec
        types = {"bfloat16": cudnn.data_type.BFLOAT16, "float16": cudnn.data_type.HALF, "float32": cudnn.data_type.FLOAT, "int32": cudnn.data_type.INT32}
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        t = graph.tensor(name=name, dim=list(shape), stride=list(stride), data_type=types[dtype])
        inputs.append((t, jax.ShapeDtypeStruct(shape, np.dtype(dtype))))
        return t

    q, k, v, bias, lq, lk = [tensor(s, n) for s, n in zip(specs[:6], ("q", "k", "v", "bias", "length_q", "length_k"))]
    options = dict(
        attn_scale=scale,
        bias=bias,
        use_padding_mask=lq is not None,
        seq_len_q=lq,
        seq_len_kv=lk,
        diagonal_band_left_bound=None if window is None else window[0] + 1,
        diagonal_band_right_bound=0 if causal else (None if window is None else window[1]),
    )
    outputs = []

    def output(t, spec):
        shape, dtype = spec
        t.set_output(True).set_dim(list(shape)).set_stride([math.prod(shape[i + 1 :]) for i in range(len(shape))])
        t.set_data_type({"bfloat16": cudnn.data_type.BFLOAT16, "float16": cudnn.data_type.HALF, "float32": cudnn.data_type.FLOAT}[dtype])
        outputs.append((t, jax.ShapeDtypeStruct(shape, np.dtype(dtype))))

    if backward:
        o, stats, do = [tensor(s, n) for s, n in zip(specs[6:], ("o", "stats", "do"))]
        dbias = None
        if has_dbias:
            dbias = graph.tensor_like(bias, name="dbias")
        dq, dk, dv = graph.sdpa_backward(q=q, k=k, v=v, o=o, stats=stats, dO=do, dBias=dbias, use_deterministic_algorithm=deterministic, **options)
        for t, s in zip((dq, dk, dv), specs[:3]):
            output(t, s)
        if dbias is not None:
            output(dbias, specs[3])
    else:
        o, stats = graph.sdpa(q=q, k=k, v=v, generate_stats=True, **options)
        output(o, specs[0])
        output(stats, (specs[0][0][:-1] + (1,), "float32"))
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    # Only backend plans currently have a serialized native JAX executor.
    graph.deselect_engines([engine.name for engine in graph._candidate_engines()])
    graph.check_support()
    graph.build_plans()
    return make_call(graph, inputs, outputs)


@lru_cache(maxsize=128)
def _operation(scale, causal, window, deterministic, has_dbias, device):
    def forward(q, k, v, bias, lq, lk):
        args = (q, k, v, bias, lq, lk)
        call = _plan(tuple(map(_spec, args)), scale, causal, window, deterministic, False, has_dbias, device)
        o, stats = call(*(x for x in args if x is not None))
        return (o, stats), (*args, o, stats)

    @jax.custom_vjp
    def attention(q, k, v, bias, lq, lk):
        return forward(q, k, v, bias, lq, lk)[0]

    def backward(saved, grads):
        args = (*saved, grads[0])
        call = _plan(tuple(map(_spec, args)), scale, causal, window, deterministic, True, has_dbias, device)
        result = call(*(x for x in args if x is not None))
        return (*result[:3], result[3] if has_dbias else None, None, None)

    attention.defvjp(forward, backward)
    return attention


def sdpa(
    query,
    key,
    value,
    bias=None,
    mask=None,
    *,
    scale=None,
    is_causal=False,
    query_seq_lengths=None,
    key_value_seq_lengths=None,
    local_window_size=None,
    return_residual=False,
    deterministic=False,
):
    """Dense FP16/BF16 attention in JAX BTNH layout, with reverse-mode AD.

    Requires a CUDA toolkit and C++ compiler for the native bridge's first build.
    Masks and lengths are device inputs. Flags, scale and shapes are static.
    Backend support checks apply to each forward/backward configuration.
    """
    placed = {d for x in (query, key, value) if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1:
        raise NotImplementedError("SDPA JAX currently requires one CUDA device")
    target_device = next(iter(placed)) if placed else (jax.config.jax_default_device or jax.local_devices(backend="gpu")[0])
    if target_device.platform != "gpu":
        raise NotImplementedError("SDPA JAX requires a CUDA device")
    device = target_device.local_hardware_id
    q, k, v = map(jnp.asarray, (query, key, value))
    unbatched = q.ndim == 3
    q, k, v = (x[None] if x.ndim == 3 else x for x in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected BTNH query and matching BSKH key/value")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError("incompatible query/key shapes")
    if q.dtype not in (jnp.float16, jnp.bfloat16) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("query, key and value must share float16 or bfloat16 dtype")
    window = (local_window_size, local_window_size) if isinstance(local_window_size, int) else local_window_size
    if window is not None and (len(window) != 2 or min(window) < 0):
        raise ValueError("local_window_size must contain two nonnegative bounds")
    scale = float(scale) if scale is not None else q.shape[-1] ** -0.5
    target = (q.shape[0], q.shape[2], q.shape[1], k.shape[1])
    has_dbias = bias is not None
    if bias is not None:
        bias = jnp.asarray(bias)
        bias = jnp.reshape(bias, (1,) * (4 - bias.ndim) + bias.shape)
        if np.broadcast_shapes(bias.shape, target) != target:
            raise ValueError("bias is not broadcastable to BNTS")
    if mask is not None:
        mask = jnp.asarray(mask)
        if mask.dtype != jnp.bool_:
            raise ValueError("mask must be boolean")
        if np.broadcast_shapes(mask.shape, target) != target:
            raise ValueError("mask is not broadcastable to BNTS")
        negative = jnp.asarray((-2 << 40) if q.dtype == jnp.bfloat16 else (-2 << 14), q.dtype)
        masked = jnp.where(mask, jnp.asarray(0, q.dtype), negative)
        bias = masked if bias is None else bias + masked
        bias = jnp.reshape(bias, (1,) * (4 - bias.ndim) + bias.shape)
    lengths = (query_seq_lengths, key_value_seq_lengths)
    if any(x is not None for x in lengths):
        lengths = tuple(jnp.full((q.shape[0],), size, jnp.int32) if x is None else jnp.asarray(x) for x, size in zip(lengths, (q.shape[1], k.shape[1])))
        if any(x.dtype != jnp.int32 or x.shape != (q.shape[0],) for x in lengths):
            raise ValueError("sequence lengths must be int32 arrays of shape (B,)")
        lengths = tuple(x.reshape((-1, 1, 1, 1)) for x in lengths)
    q, k, v = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
    o, stats = _operation(scale, is_causal, window, deterministic, has_dbias, device)(q, k, v, bias, *lengths)
    o = o.transpose(0, 2, 1, 3)
    stats = jax.lax.stop_gradient(stats[..., 0].transpose(0, 2, 1).astype(o.dtype))
    if unbatched:
        o, stats = o[0], stats[0]
    return (o, stats) if return_residual else o
