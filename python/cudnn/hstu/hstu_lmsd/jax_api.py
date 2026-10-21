# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental, single-device SM10x JAX HSTU LayerNorm-Multiply-SiLU (no dropout)."""

from dataclasses import dataclass
from functools import lru_cache
import math

import jax
import jax.numpy as jnp

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX HSTU LMSD")
if requirement_error:
    raise ImportError(requirement_error)

import cutlass
import cutlass.cute as cute

from cudnn.jax import call
from ._kernels._common import MAX_FLAT_ELEMENTS, is_supported_hidden_size
from ._kernels._config import HSTULMSDBwdConfig, HSTULMSDFwdConfig
from ._kernels.hstu_lmsd_bwd import HSTULMSDBackward, HSTULMSDGradReduce
from ._kernels.hstu_lmsd_fwd import HSTULMSDForward


@dataclass(frozen=True)
class Config:
    rows: int
    hidden: int
    dtype: str
    eps: float
    apply_u_silu: bool
    concat_u: bool
    concat_x: bool
    compute_dweight: bool
    sm_count: int
    arch: str

    @property
    def segments(self):
        return 1 + int(self.concat_u) + int(self.concat_x)


def require_array(x, name, shape, dtype):
    if not isinstance(x, (jax.Array, jax.core.Tracer)):
        raise TypeError(f"{name} must be a JAX array")
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(x.shape)}")
    if x.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {x.dtype}")


def target_device(arrays):
    placed = {d for x in arrays if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer) for d in x.devices()}
    if len(placed) > 1:
        raise ValueError("all HSTU LMSD arrays must be on the same GPU")
    # Traced arrays carry no device: under jax.jit the call runs on JAX's default device.
    device = next(iter(placed)) if placed else jax.devices()[0]
    capability = str(getattr(device, "compute_capability", ""))
    if jax.process_count() != 1 or device.platform != "gpu" or not capability.startswith("10."):
        raise ValueError("JAX HSTU LMSD requires a single process and arrays on one SM10x GPU")
    return device, "sm_" + capability.replace(".", "") + "a"


def configuration(x, u, weight, bias, eps, dropout_ratio, apply_u_silu, concat_u, concat_x, compute_dweight):
    if not isinstance(x, (jax.Array, jax.core.Tracer)) or x.ndim != 2:
        raise ValueError("x_tensor must be a rank-2 JAX array")
    if x.dtype != jnp.bfloat16:
        raise TypeError(f"x_tensor must be bfloat16, got {x.dtype}")
    n, d = x.shape
    require_array(u, "u_tensor", (n, d), x.dtype)
    require_array(weight, "weight_tensor", (d,), x.dtype)
    require_array(bias, "bias_tensor", (d,), x.dtype)
    if not is_supported_hidden_size(d):
        raise ValueError(f"HSTU LMSD supports D divisible by 8 in [8, 1024), got D={d}")
    if not 1 <= n <= MAX_FLAT_ELEMENTS // d:
        raise ValueError(f"x_tensor row count must be in [1, {MAX_FLAT_ELEMENTS // d}], got {n}")
    if dropout_ratio != 0.0:
        raise NotImplementedError("JAX HSTU LMSD does not support dropout yet; pass dropout_ratio=0.0")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive, finite static float")
    device, arch = target_device((x, u, weight, bias))
    return Config(n, d, str(x.dtype), float(eps), bool(apply_u_silu), bool(concat_u), bool(concat_x), bool(compute_dweight), device.core_count, arch)


def output_segments(y, c):
    """Split an (N, S, D) buffer into the kernel's (silu, x, lmsd) (N, D) segment views."""
    lmsd = y[None, c.segments - 1, None]
    return (y[None, 0, None] if c.concat_u else lmsd), (y[None, int(c.concat_u), None] if c.concat_x else lmsd), lmsd


# The kernels keep per-trace state on the instance, and the JAX bridge traces an
# adapter more than once, so each trace builds a fresh kernel object.
class Forward:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, x, u, weight, bias, y, mean, rstd):
        c = self.c
        config = HSTULMSDFwdConfig.from_hidden_size(c.hidden)
        kernel = HSTULMSDForward(c.hidden, apply_u_silu=c.apply_u_silu, concat_u=c.concat_u, concat_x=c.concat_x, has_dropout=False, config=config)
        row_blocks = (c.rows + config.rows_per_cta - 1) // config.rows_per_cta
        grid = config.launch_grid(c.rows, c.sm_count)
        silu, x_out, lmsd = output_segments(y, c)
        kernel(
            x,
            u,
            weight,
            bias,
            silu,
            x_out,
            lmsd,
            x,
            mean,
            rstd,
            cutlass.Int64(0),
            cutlass.Int32(c.rows),
            cutlass.Int32(c.hidden),
            cutlass.Float32(c.eps),
            cutlass.Float32(0.0),
            cutlass.Uint32(0),
            cutlass.Int32(row_blocks),
            cutlass.Int32((row_blocks + grid - 1) // grid),
            cutlass.Int32(grid),
            stream,
        )


class Backward:
    def __init__(self, c):
        self.c = c

    @cute.jit
    def __call__(self, stream, dy, x, u, weight, bias, mean, rstd, dx, du, dweight, dbias, dweight_workspace, dbias_workspace):
        c = self.c
        config = HSTULMSDBwdConfig.from_hidden_size(c.hidden)
        main = HSTULMSDBackward(
            c.hidden,
            apply_u_silu=c.apply_u_silu,
            concat_u=c.concat_u,
            concat_x=c.concat_x,
            has_dropout=False,
            compute_dweight=c.compute_dweight,
            config=config,
        )
        grid = config.launch_grid(c.rows, c.sm_count)
        dy_silu, dy_x, dy_lmsd = output_segments(dy, c)
        main(
            dy_silu,
            dy_x,
            dy_lmsd,
            x,
            u,
            weight,
            bias,
            x,
            dx,
            du,
            mean,
            rstd,
            dweight_workspace,
            dbias_workspace,
            cutlass.Float32(0.0),
            cutlass.Int32(c.hidden),
            cutlass.Int32(c.rows),
            cutlass.Int32(grid),
            stream,
        )
        HSTULMSDGradReduce(c.compute_dweight)(dweight_workspace, dbias_workspace, dweight, dbias, cutlass.Int32(grid), cutlass.Int32(c.hidden), stream)


@lru_cache(maxsize=128)
def forward_call(c):
    desc = jax.ShapeDtypeStruct
    stats = desc((c.rows,), jnp.float32)
    return call(
        Forward(c),
        output_shape_dtype=(desc((c.rows, c.segments, c.hidden), c.dtype), stats, stats),
        compile_options=f"--gpu-arch={c.arch}",
        use_static_tensors=True,
    )


@lru_cache(maxsize=128)
def backward_call(c):
    desc = jax.ShapeDtypeStruct
    grad = desc((c.rows, c.hidden), c.dtype)
    vector = desc((c.hidden,), c.dtype)
    workspace = desc((HSTULMSDBwdConfig.from_hidden_size(c.hidden).workspace_rows(c.sm_count), c.hidden), jnp.float32)
    return call(
        Backward(c),
        output_shape_dtype=(grad, grad, vector, vector, workspace, workspace),
        compile_options=f"--gpu-arch={c.arch}",
        use_static_tensors=True,
    )


def hstu_lmsd_forward(
    x_tensor,
    u_tensor,
    weight_tensor,
    bias_tensor,
    eps=1e-6,
    dropout_ratio=0.0,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
):
    """Return ``{"y_tensor", "mean_tensor", "rstd_tensor", "mask_tensor"}``; ``mask_tensor`` is None (no dropout).

    ``y`` concatenates the optional activated-u and x segments before the LMSD segment.
    """
    c = configuration(x_tensor, u_tensor, weight_tensor, bias_tensor, eps, dropout_ratio, apply_u_silu, concat_u, concat_x, True)
    y, mean, rstd = forward_call(c)(x_tensor, u_tensor, weight_tensor, bias_tensor)
    return {"y_tensor": y.reshape(c.rows, c.segments * c.hidden), "mean_tensor": mean, "rstd_tensor": rstd, "mask_tensor": None}


def hstu_lmsd_backward(
    dy_tensor,
    x_tensor,
    u_tensor,
    weight_tensor,
    bias_tensor,
    mean_tensor,
    rstd_tensor,
    mask_tensor=None,
    dropout_ratio=0.0,
    apply_u_silu=True,
    concat_u=True,
    concat_x=True,
    compute_dweight=True,
):
    """Return ``{"dx_tensor", "du_tensor", "dweight_tensor", "dbias_tensor"}``; ``dweight_tensor`` is None without ``compute_dweight``.

    Pass the forward's inputs, ``mean``/``rstd`` and options.
    """
    if mask_tensor is not None:
        raise NotImplementedError("JAX HSTU LMSD does not support dropout yet; mask_tensor must be None")
    c = configuration(x_tensor, u_tensor, weight_tensor, bias_tensor, 1e-6, dropout_ratio, apply_u_silu, concat_u, concat_x, compute_dweight)
    require_array(dy_tensor, "dy_tensor", (c.rows, c.segments * c.hidden), x_tensor.dtype)
    require_array(mean_tensor, "mean_tensor", (c.rows,), jnp.float32)
    require_array(rstd_tensor, "rstd_tensor", (c.rows,), jnp.float32)
    dy = dy_tensor.reshape(c.rows, c.segments, c.hidden)
    dx, du, dweight, dbias, *_ = backward_call(c)(dy, x_tensor, u_tensor, weight_tensor, bias_tensor, mean_tensor, rstd_tensor)
    return {"dx_tensor": dx, "du_tensor": du, "dweight_tensor": dweight if c.compute_dweight else None, "dbias_tensor": dbias}
