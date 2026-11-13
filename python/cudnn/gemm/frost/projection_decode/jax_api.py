# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX binding for native FP8 projection at decode row counts."""

from functools import lru_cache
from numbers import Integral

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

from .fp8_projection_decode import cc_supported, dsl_error
from .kernel.fp8_projection_decode_config import Geometry, check_numerics, part_readings, pick, resolve_form, token_tile_for, validate_cfg


def _target_device(arrays):
    from cudnn.frost.device import compute_capability

    placed = {device for value in arrays if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer) for device in value.devices()}
    if len(placed) > 1:
        raise ValueError("JAX FP8 projection decode operands must be on the same GPU")
    target = next(iter(placed)) if placed else (jax.config.jax_default_device or jax.local_devices()[0])
    if target.platform != "gpu":
        raise NotImplementedError("JAX FP8 projection decode requires a CUDA GPU")
    device = target.local_hardware_id
    capability = compute_capability(device)
    if not cc_supported(capability):
        raise NotImplementedError(f"JAX FP8 projection decode requires an SM10x GPU; got compute capability {capability}")
    return device, capability


@lru_cache(maxsize=128)
def _configuration(rows, numerics, device, capability):
    from cudnn.frost.device import multiprocessor_count, shared_memory_per_block_optin, shared_memory_per_multiprocessor

    readings = part_readings(shared_memory_per_multiprocessor(device), shared_memory_per_block_optin(device), multiprocessor_count(device))
    form = pick(numerics, rows, readings["SM_COUNT"], readings["SMEM_PER_SM_BYTES"], readings["SMEM_OPTIN_BYTES"])
    cfg = resolve_form(numerics, rows, form, readings)
    try:
        validate_cfg(cfg)
    except ValueError as error:
        raise NotImplementedError(f"JAX FP8 projection decode cannot run on device {device}: {error}") from error
    requirement_error = dsl_error(capability)
    if requirement_error:
        raise NotImplementedError(requirement_error)
    return cfg


@lru_cache(maxsize=128)
def _call(cfg, device, gpu_arch):
    import cutlass.cute as cute

    from cudnn.jax import TensorSpec, call

    from .kernel.fp8_projection_decode_e4m3 import Fp8ProjectionDecodeKernel, launch as launch_projection

    kernel = Fp8ProjectionDecodeKernel(cfg)

    @cute.experimental.jit
    def launch(stream, weight, activation, alpha, out):
        # A static singleton row folds the TMA descriptor to the wrong rank.
        activation = cute.make_tensor(activation.iterator, cute.make_layout((activation.shape[0], cfg.K), stride=(cfg.K, 1)))
        alpha = cute.make_tensor(alpha.iterator, cute.make_layout(1))
        launch_projection(kernel, weight, activation, out, alpha, stream)

    return call(
        launch,
        output_shape_dtype=(jax.ShapeDtypeStruct((cfg.ROWS, cfg.N), jnp.bfloat16),),
        input_spec=(TensorSpec(static=True), TensorSpec(static=False), TensorSpec(static=True)),
        output_spec=(TensorSpec(static=True),),
        use_static_tensors=True,
        compile_options=f"--gpu-arch={gpu_arch}",
    )


def fp8_projection_decode_forward_jax(weight, activation, alpha, *, split_k=None):
    """Return ``(out,)`` for E4M3 projection with FP32 scale and BF16 output.

    Weight is ``[N,K]``, activation ``[T,K]``, and alpha scalar or ``[1]``.
    T is 1..32. Supported (N,K,split_k) are (4608,8192,4), (4608,8192,2),
    and (8192,2048,2). The default split depends only on geometry, never T.
    Requires CuTeDSL >= 4.9.0 for experimental-launcher support in cutlass.jax.
    The function supports eager calls and JIT; differentiation is unsupported.
    """
    requirement_error = cutedsl_requirement_error("JAX FP8 projection decode", floor=(4, 9, 0))
    if requirement_error:
        raise NotImplementedError(requirement_error)
    arrays = (weight, activation, alpha)
    for name, value in zip(("weight", "activation", "alpha"), arrays):
        if not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX array")
        if any(not isinstance(size, Integral) or size <= 0 for size in value.shape):
            raise ValueError(f"{name} must have a positive static shape; got {value.shape}")
    if weight.dtype != jnp.float8_e4m3fn or activation.dtype != jnp.float8_e4m3fn:
        raise TypeError("weight and activation must have dtype float8_e4m3fn")
    if alpha.dtype != jnp.float32:
        raise TypeError("alpha must have dtype float32")
    if weight.ndim != 2 or activation.ndim != 2 or weight.shape[1] != activation.shape[1]:
        raise ValueError("FP8 projection decode requires weight[N,K] and activation[T,K]")
    if alpha.shape not in ((), (1,)):
        raise ValueError("alpha must be a scalar or a one-element vector")
    rows = activation.shape[0]
    token_tile_for(rows)
    numerics = check_numerics(Geometry(*weight.shape), split_k)
    device, capability = _target_device(arrays)
    cfg = _configuration(rows, numerics, device, capability)
    gpu_arch = f"sm_{capability[0]}{capability[1]}a"
    return _call(cfg, device, gpu_arch)(*arrays)
