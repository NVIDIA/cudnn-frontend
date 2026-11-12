# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX binding for native FP8 projection at decode row counts."""

from functools import lru_cache
from numbers import Integral

from cudnn.frost.buffers import cutedsl_requirement_error

_requirement_error = cutedsl_requirement_error("JAX FP8 projection decode", floor=(4, 8, 0))
if _requirement_error:
    raise ImportError(_requirement_error)

import jax
import jax.numpy as jnp

from cudnn._cake.jax import TensorRef, make_call
from cudnn._device import ensure_current_context

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
    import cutlass
    import cutlass.cute as cute

    from .kernel.fp8_projection_decode_e4m3 import Fp8ProjectionDecodeKernel, launch as launch_projection

    ensure_current_context(device=device)
    fake = cute.runtime.make_fake_compact_tensor
    compiled = cute.experimental.compile(
        launch_projection,
        Fp8ProjectionDecodeKernel(cfg),
        fake(cutlass.Uint8, (cfg.N, cfg.K), stride_order=(1, 0), assumed_align=16),
        # Preserve the one-row activation's rank in its TMA descriptor.
        fake(cutlass.Uint8, (cute.sym_int32(), cfg.K), stride_order=(1, 0), assumed_align=16),
        fake(cutlass.BFloat16, (cfg.ROWS, cfg.N), stride_order=(1, 0), assumed_align=16),
        fake(cutlass.Float32, (1,), assumed_align=16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --gpu-arch={gpu_arch}",
    )
    arguments = (
        TensorRef(0, (cfg.N, cfg.K), "uint8"),
        TensorRef(1, (cfg.ROWS, cfg.K), "uint8"),
        TensorRef(3, (cfg.ROWS, cfg.N), "bfloat16"),
        TensorRef(2, (1,), "float32"),
    )
    outputs = (jax.ShapeDtypeStruct((cfg.ROWS, cfg.N), jnp.bfloat16),)
    return make_call(((compiled, arguments),), 3, outputs, device)


def fp8_projection_decode_forward_jax(weight, activation, alpha, *, split_k=None):
    """Return ``(out,)`` for E4M3 projection with FP32 scale and BF16 output.

    Weight is ``[N,K]``, activation ``[T,K]``, and alpha scalar or ``[1]``.
    T is 1..32. Supported (N,K,split_k) are (4608,8192,4), (4608,8192,2),
    and (8192,2048,2). The default split depends only on geometry, never T.
    The function supports eager calls and JIT; differentiation is unsupported.
    """
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
