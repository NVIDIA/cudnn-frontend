# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for fused RMSNorm + RHT + per-CTA amax."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple, Optional

from .._rmsnorm_rht_amax_config import resolve_launch_config

from ._static_values import optional_static_int, require_concrete_dims, require_static_float
from .cutedsl import BufferSpec, TensorLayout, call_cutedsl


class RmsNormRhtAmaxResult(NamedTuple):
    """Functional JAX outputs for RMSNorm + RHT + amax."""

    output: Any
    amax: Any


@lru_cache(maxsize=None)
def _make_launcher(
    n: int,
    num_threads: int,
    rows_per_cta: int,
    eps: float,
):
    # Keep optional CuTe DSL imports off the cudnn.jax import path.
    from cutlass import Float32
    from ..rmsnorm_rht_amax.kernel import RMSNormRHTAmaxKernel

    kernel = RMSNormRHTAmaxKernel(
        n=n,
        num_threads=num_threads,
        eps=eps,
        rows_per_cta=rows_per_cta,
    )

    def launch(stream, x, weight, output, amax):
        kernel(x, weight, output, amax, Float32(eps), stream)

    return launch


def rmsnorm_rht_amax_sm100(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> RmsNormRhtAmaxResult:
    """Apply fused RMSNorm, 16-wide RHT, and per-CTA amax from JAX.

    This functional API is intended for use inside ``jax.jit``. ``x`` must be
    a row-major ``bfloat16`` array of shape ``(M, N)`` and ``weight`` a
    ``bfloat16`` array of shape ``(N,)``. ``M`` and ``N`` are concrete in this
    proof of concept; shape-polymorphic export is a follow-up.

    ``eps``, ``num_threads``, and ``rows_per_cta`` are compile-time
    configuration. Close them over a jitted function or list them in
    ``jax.jit(static_argnames=...)``.

    Returns ``(output, amax)`` with shapes ``(M, N)`` and
    ``(M // rows_per_cta,)`` respectively.
    """

    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("rmsnorm_rht_amax_sm100 requires JAX; install the 'jax' " "optional dependencies") from exc

    if x.ndim != 2:
        raise ValueError(f"x must have rank 2, got shape {x.shape}")
    if weight.ndim != 1:
        raise ValueError(f"weight must have rank 1, got shape {weight.shape}")

    m, n = require_concrete_dims(x.shape, "M", "N")
    if tuple(weight.shape) != (n,):
        raise ValueError(f"weight shape must match the hidden dimension ({n},), " f"got {weight.shape}")
    if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError("x and weight must both have dtype bfloat16, " f"got {x.dtype} and {weight.dtype}")

    num_threads = optional_static_int(num_threads, name="num_threads")
    rows_per_cta = optional_static_int(rows_per_cta, name="rows_per_cta")
    eps = require_static_float(eps, name="eps")
    resolved_num_threads, resolved_rows_per_cta = resolve_launch_config(
        m,
        n,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )
    # JAX/XLA owns physical buffers. These specs constrain ordinary row-major
    # storage while providing the divisibility facts used by the CuTe kernel.
    x_layout = TensorLayout(divisibility=(resolved_rows_per_cta, 16))
    weight_layout = TensorLayout(divisibility=(16,))
    amax_layout = TensorLayout()

    output, amax = call_cutedsl(
        _make_launcher(n, resolved_num_threads, resolved_rows_per_cta, eps),
        (x, weight),
        outputs=(
            BufferSpec(
                "output",
                (m, n),
                jnp.bfloat16,
                tensor_layout=x_layout,
            ),
            BufferSpec(
                "amax",
                (m // resolved_rows_per_cta,),
                jnp.float32,
                tensor_layout=amax_layout,
            ),
        ),
        input_layouts=(x_layout, weight_layout),
        use_static_tensors=True,
    )
    return RmsNormRhtAmaxResult(output=output, amax=amax)


__all__ = ["RmsNormRhtAmaxResult", "rmsnorm_rht_amax_sm100"]
