# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional LayerNorm and RMSNorm with native CuTeDSL reverse-mode derivatives."""

from dataclasses import dataclass
from functools import lru_cache, partial
import math
from numbers import Integral, Real

from cudnn.jax import call, zeros_init

import jax
import jax.numpy as jnp

from ._lazy import dsl_unavailable

_requirement_error = dsl_unavailable("JAX normalization")
if _requirement_error:
    raise ImportError(_requirement_error)

from .config_sm100 import NormVariant, RowwiseSpec, rowwise_spec
from .jax_kernels import make_backward, make_forward


@dataclass(frozen=True)
class _Config:
    spec: RowwiseSpec
    dtype: str
    has_beta: bool
    epsilon: float
    device: int


def _require_array(value, name):
    if not isinstance(value, (jax.Array, jax.core.Tracer)):
        raise TypeError(f"{name} must be a JAX array")


def _target_device(*arrays):
    from cudnn.frost.device import compute_capability

    if jax.process_count() != 1:
        raise NotImplementedError("JAX normalization requires a single process")
    placed = {device for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for device in x.devices()}
    if len(placed) > 1:
        raise ValueError("JAX normalization arrays must be on the same GPU")
    # Tracers have no concrete placement. The default device supplies tuning
    # properties; XLA places execution, and compilation explicitly targets SM100.
    target = next(iter(placed)) if placed else (jax.config.jax_default_device or jax.local_devices()[0])
    if target.platform != "gpu":
        raise NotImplementedError(f"JAX normalization requires a CUDA GPU; got {target.platform}")
    device = target.local_hardware_id
    capability = compute_capability(device)
    if capability != (10, 0):
        raise NotImplementedError(f"JAX normalization requires SM100 (compute capability 10.0); got {capability}")
    return device


def _configuration(x, weight, bias, normalized_shape, eps, variant):
    _require_array(x, "x")
    if not x.shape or any(not isinstance(dim, Integral) or dim <= 0 for dim in x.shape):
        raise ValueError("x must have a positive static shape with at least one dimension")
    dtypes = {jnp.dtype(jnp.bfloat16): "bf16", jnp.dtype(jnp.float16): "fp16", jnp.dtype(jnp.float32): "fp32"}
    if x.dtype not in dtypes:
        raise TypeError(f"x must have dtype bfloat16, float16, or float32; got {x.dtype}")
    if normalized_shape is None:
        shape = x.shape[-1:]
    elif isinstance(normalized_shape, Integral) and not isinstance(normalized_shape, bool):
        shape = (int(normalized_shape),)
    elif isinstance(normalized_shape, (tuple, list)):
        shape = tuple(normalized_shape)
    else:
        raise TypeError("normalized_shape must be a static integer or sequence of integers")
    if not shape or any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise ValueError("normalized_shape must contain positive static dimensions")
    if len(shape) > x.ndim or tuple(x.shape[-len(shape) :]) != shape:
        raise ValueError(f"normalized_shape {shape} must match the trailing dimensions of x.shape {x.shape}")
    for name, value in (("weight", weight), ("bias", bias)):
        if value is None:
            continue
        _require_array(value, name)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
        if value.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match x dtype {x.dtype}; got {value.dtype}")
    if not isinstance(eps, Real) or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive static scalar")
    if not float(jnp.finfo(jnp.float32).smallest_subnormal) <= eps <= float(jnp.finfo(jnp.float32).max):
        raise ValueError("eps must be within the positive finite float32 range")
    spec = rowwise_spec(variant, x.shape, normalized_shape=shape)
    if spec.R * spec.M >= 2**31:
        raise ValueError("JAX normalization requires the flattened tensor size to fit signed int32")
    return _Config(spec, dtypes[x.dtype], bias is not None, float(eps), _target_device(x, weight, bias))


@lru_cache(maxsize=128)
def _forward_call(config):
    from cudnn.frost.device import build_device

    spec = config.spec
    with build_device(config.device):
        launch = make_forward(
            spec.R,
            spec.M,
            config.dtype,
            has_mean=spec.has_mean,
            has_beta=config.has_beta,
            epsilon=config.epsilon,
        )
    dtype = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[config.dtype]
    stats = jax.ShapeDtypeStruct((spec.R,), jnp.float32)
    outputs = (jax.ShapeDtypeStruct((spec.R, spec.M), dtype), stats if spec.has_mean else None, stats)
    return call(launch, output_shape_dtype=outputs, use_static_tensors=True, compile_options="--gpu-arch=sm_100a")


@lru_cache(maxsize=128)
def _backward_call(config):
    from cudnn.frost.device import build_device, multiprocessor_count

    spec = config.spec
    with build_device(config.device):
        launch, partial_shape = make_backward(
            spec.R,
            spec.M,
            config.dtype,
            has_mean=spec.has_mean,
            has_beta=config.has_beta,
            sm_count=multiprocessor_count(config.device),
        )
    dtype = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[config.dtype]
    parameter = jax.ShapeDtypeStruct((spec.M,), jnp.float32)
    partials = jax.ShapeDtypeStruct(partial_shape, jnp.float32) if partial_shape is not None else None
    outputs = (
        jax.ShapeDtypeStruct((spec.R, spec.M), dtype),
        parameter,
        parameter if config.has_beta else None,
        partials,
        partials if config.has_beta else None,
    )
    initialized = {1: zeros_init}
    if config.has_beta:
        initialized[2] = zeros_init
    return call(launch, output_shape_dtype=outputs, initialized_outputs=initialized, use_static_tensors=True, compile_options="--gpu-arch=sm_100a")


def _forward(config, x, weight, bias):
    gamma = jnp.ones((config.spec.M,), dtype=x.dtype) if weight is None else weight
    beta = gamma if bias is None else bias
    y, mean, rstd = _forward_call(config)(x, gamma, beta)
    return y.reshape(x.shape), (x, weight, bias, gamma, mean, rstd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _normalize(config, x, weight, bias):
    return _forward(config, x, weight, bias)[0]


def _backward(config, residual, dy):
    x, weight, bias, gamma, mean, rstd = residual
    dx, dgamma, dbeta, _, _ = _backward_call(config)(dy, x, gamma, mean, rstd)
    dweight = None if weight is None else dgamma.reshape(weight.shape).astype(weight.dtype)
    dbias = None if bias is None else dbeta.reshape(bias.shape).astype(bias.dtype)
    return dx.reshape(x.shape), dweight, dbias


_normalize.defvjp(_forward, _backward)


def layer_norm_forward_jax(x, weight=None, bias=None, *, normalized_shape=None, eps=1e-5):
    """Normalize trailing dimensions on SM100, with first-order x/weight/bias gradients.

    Inputs have matching BF16, FP16, or FP32 dtype. Affine parameters, when present,
    have exactly ``normalized_shape``; the default normalizes the last dimension.
    ``normalized_shape`` and positive ``eps`` are static under ``jax.jit``.
    Weight and bias gradients use FP32 atomic accumulation and need not be
    bitwise repeatable across calls with identical inputs.
    Returns ``(y,)``; ``y`` has the same shape and dtype as ``x``.
    """
    config = _configuration(x, weight, bias, normalized_shape, eps, NormVariant.LAYER_NORM)
    return (_normalize(config, x, weight, bias),)


def rms_norm_forward_jax(x, weight=None, *, normalized_shape=None, eps=1e-5):
    """RMS-normalize trailing dimensions on SM100, with first-order x/weight gradients.

    The dtype, shape, and static-argument contract matches :func:`layer_norm_forward_jax`.
    No bias is applied; omitted weight means one.
    Weight gradients use FP32 atomic accumulation and need not be bitwise repeatable.
    Returns ``(y,)``; ``y`` has the same shape and dtype as ``x``.
    """
    config = _configuration(x, weight, None, normalized_shape, eps, NormVariant.RMS_NORM)
    return (_normalize(config, x, weight, None),)
