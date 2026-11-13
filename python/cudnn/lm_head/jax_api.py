# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cake LM-head log-probabilities with JIT and first-order reverse-mode AD."""

from functools import lru_cache

import jax
import jax.numpy as jnp

from cudnn._cake.jax import TensorRef, make_call
from .cake import LmHeadLogprobPlan, _View


def _device(arrays):
    placed = {d for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1 or jax.process_count() != 1:
        raise NotImplementedError("Cake LM head requires one GPU in one process")
    device = placed.pop() if placed else (jax.config.jax_default_device or jax.local_devices()[0])
    if device.platform != "gpu":
        raise NotImplementedError("Cake LM head requires a GPU")
    return device.local_hardware_id


def _validate(x, weight, labels):
    if x.ndim != 2 or weight.ndim != 2 or labels.shape != (x.shape[0],) or x.shape[1] != weight.shape[1]:
        raise ValueError("LM head requires x[T,H], weight[V,H], and labels[T]")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError("LM head requires BF16 x and weight")
    if labels.dtype != jnp.int64:
        raise ValueError("LM head requires int64 labels; enable jax_enable_x64 before creating labels")
    LmHeadLogprobPlan.check_support(x.shape[0], x.shape[1], weight.shape[0])


@lru_cache(maxsize=128)
def _call(tokens, hidden, vocab, device, backward):
    plan = LmHeadLogprobPlan(tokens, hidden, vocab, device=device)
    inputs = ("X", "W", "labels", "lse", "d_in") if backward else ("X", "W", "labels")
    outputs = ("dx_acc", "dw_acc") if backward else ("logp", "lse")
    references = {name: TensorRef(index, *plan.spec[name]) for index, name in enumerate(inputs + outputs)}
    workspace_index = len(inputs) + len(outputs)
    references.update({name: TensorRef(workspace_index, shape, dtype, offset) for name, (offset, shape, dtype) in plan.regions.items()})
    stages = []
    for key in plan.backward_order if backward else plan.forward_order:
        fn, template = plan.launches[key]
        arguments = []
        for value in template:
            if isinstance(value, _View):
                ref = references[value.name]
                value = TensorRef(ref.index, value.shape, value.dtype, ref.offset + value.offset)
            arguments.append(value)
        stages.append((fn, tuple(arguments)))
    shapes = tuple(jax.ShapeDtypeStruct(*plan.spec[name]) for name in outputs) + (jax.ShapeDtypeStruct((plan.workspace_size,), jnp.uint8),)
    return make_call(stages, len(inputs), shapes, device)


def lm_head_logprob_forward_jax(x, weight, labels):
    """Return FP32 ``(logp, lse)``; labels are vocabulary indices or -100."""
    _validate(x, weight, labels)
    invoke = _call(x.shape[0], x.shape[1], weight.shape[0], _device((x, weight, labels)), False)
    return tuple(invoke(x, weight, labels)[:2])


def lm_head_logprob_backward_jax(x, weight, labels, lse, dlogp):
    """Return FP32 ``(dx, dweight)`` using the forward LSE and FP32 dlogp."""
    _validate(x, weight, labels)
    if any(value.shape != (x.shape[0],) or value.dtype != jnp.float32 for value in (lse, dlogp)):
        raise ValueError("lse and dlogp must be FP32 [tokens]")
    invoke = _call(x.shape[0], x.shape[1], weight.shape[0], _device((x, weight, labels, lse, dlogp)), True)
    return tuple(invoke(x, weight, labels, lse, dlogp)[:2])


@jax.custom_vjp
def lm_head_logprob_jax(x, weight, labels):
    """Return per-token log-probabilities; supports JIT and first-order gradients.

    Inputs are BF16 x[T,H]/weight[V,H] and int64 labels[T]. Labels must be
    in [0,V) or -100 (ignored). AD returns BF16 gradients matching inputs;
    the explicit backward returns FP32 gradients. Higher derivatives are unsupported.
    """
    return lm_head_logprob_forward_jax(x, weight, labels)[0]


def _forward(x, weight, labels):
    logp, lse = lm_head_logprob_forward_jax(x, weight, labels)
    return logp, (x, weight, labels, lse)


def _backward(residual, dlogp):
    x, weight, labels, lse = residual
    dx, dw = lm_head_logprob_backward_jax(x, weight, labels, lse, dlogp)
    return dx.astype(x.dtype), dw.astype(weight.dtype), None


lm_head_logprob_jax.defvjp(_forward, _backward)
