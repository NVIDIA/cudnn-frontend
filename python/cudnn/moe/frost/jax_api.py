# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX binding for the native shared-expert decode kernel pair."""

from functools import lru_cache
from numbers import Integral

from .shared_expert_decode import _dsl_version_error, cc_supported
from .kernel.shared_expert_decode_config import SCRATCH_ROWS, SERVED_GEOMETRIES, Geometry, default_cfg, form_for, token_tiles_for

_requirement_error = _dsl_version_error()
if _requirement_error:
    raise ImportError(_requirement_error)

import jax
import jax.numpy as jnp

from cudnn.jax import call


def _target_device(arrays):
    from cudnn.frost.device import compute_capability

    placed = {device for value in arrays if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer) for device in value.devices()}
    if len(placed) > 1:
        raise ValueError("JAX shared-expert decode operands must be on the same GPU")
    target = next(iter(placed)) if placed else (jax.config.jax_default_device or jax.local_devices()[0])
    if target.platform != "gpu":
        raise NotImplementedError("JAX shared-expert decode requires a CUDA GPU")
    device = target.local_hardware_id
    capability = compute_capability(device)
    if not cc_supported(capability):
        raise NotImplementedError(f"JAX shared-expert decode requires an SM10x GPU; got compute capability {capability}")
    return device, f"sm_{capability[0]}{capability[1]}a"


@lru_cache(maxsize=128)
def _configuration(tokens, geometry, device):
    from cudnn.frost.device import multiprocessor_count, shared_memory_per_block_optin, shared_memory_per_multiprocessor

    try:
        cfg = default_cfg(geometry, shared_memory_per_multiprocessor(device), shared_memory_per_block_optin(device), multiprocessor_count(device))
        return form_for(cfg, tokens)
    except ValueError as error:
        raise NotImplementedError(f"JAX shared-expert decode cannot run on device {device}: {error}") from error


@lru_cache(maxsize=128)
def _call(cfg, tokens, device, gpu_arch):
    import cutlass
    import cutlass.cute as cute

    from .kernel.shared_expert_decode_f16 import launch as launch_pair

    @cute.jit
    def launch(stream, x, w_gate_up, w_down, w_gate, out, act, gate_part):
        x = cute.make_tensor(x.iterator, cute.make_layout(tokens * cfg.H))
        w_gate_up = cute.make_tensor(w_gate_up.iterator, cute.make_layout(2 * cfg.I * cfg.H))
        w_down = cute.make_tensor(w_down.iterator, cute.make_layout(cfg.H * cfg.I))
        w_gate = cute.make_tensor(w_gate.iterator, cute.make_layout(cfg.H))
        out = cute.make_tensor(out.iterator, cute.make_layout(tokens * cfg.H))
        act = cute.make_tensor(act.iterator, cute.make_layout(SCRATCH_ROWS * cfg.I))
        gate_part = cute.make_tensor(gate_part.iterator, cute.make_layout(SCRATCH_ROWS * cfg.GATE_SLOTS))
        launch_pair(cfg, x, w_gate_up, w_down, w_gate, act, gate_part, out, cutlass.Int32(tokens), cfg.H, cfg.H, stream)

    outputs = (
        jax.ShapeDtypeStruct((tokens, cfg.H), jnp.bfloat16),
        jax.ShapeDtypeStruct((SCRATCH_ROWS, cfg.I), jnp.bfloat16),
        jax.ShapeDtypeStruct((SCRATCH_ROWS, cfg.GATE_SLOTS), jnp.float32),
    )
    return call(launch, output_shape_dtype=outputs, use_static_tensors=True, compile_options=f"--gpu-arch={gpu_arch}")


def shared_expert_decode_forward_jax(x, w_gate_up, w_down, w_gate):
    """Return ``(y,)`` from native BF16 shared-expert decode, eagerly or under JIT.

    Inputs are ``x[T,H]``, ``w_gate_up[2*I,H]``, ``w_down[H,I]``, and ``w_gate[H]``
    or ``w_gate[1,H]``. Supported geometries are H=8192, I=128/256 and T=1..24.
    Outputs and scratch belong to XLA. Automatic differentiation is unsupported.
    """
    arrays = (x, w_gate_up, w_down, w_gate)
    for name, value in zip(("x", "w_gate_up", "w_down", "w_gate"), arrays):
        if not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX array")
        if value.dtype != jnp.bfloat16:
            raise TypeError(f"{name} must have dtype bfloat16; got {value.dtype}")
        if any(not isinstance(size, Integral) or size <= 0 for size in value.shape):
            raise ValueError(f"{name} must have a positive static shape; got {value.shape}")
    if x.ndim != 2 or w_gate_up.ndim != 2:
        raise ValueError("shared-expert decode requires x[T,H] and w_gate_up[2*I,H]")
    tokens, hidden = x.shape
    token_tiles_for(tokens)
    if w_gate_up.shape[0] % 2:
        raise ValueError("w_gate_up must contain equally sized gate and up projections")
    intermediate = w_gate_up.shape[0] // 2
    if w_gate_up.shape[1] != hidden or w_down.shape != (hidden, intermediate) or w_gate.shape not in ((hidden,), (1, hidden)):
        raise ValueError("shared-expert decode requires w_gate_up[2*I,H], w_down[H,I], and w_gate[H] or [1,H]")
    geometry = Geometry(hidden, intermediate)
    if geometry not in SERVED_GEOMETRIES:
        raise NotImplementedError(f"JAX shared-expert decode does not support {geometry.describe()}")
    device, gpu_arch = _target_device(arrays)
    cfg = _configuration(tokens, geometry, device)
    out, _, _ = _call(cfg, tokens, device, gpu_arch)(*arrays)
    return (out,)
