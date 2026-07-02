# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for fused RMSNorm + RHT + per-CTA amax."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from .._jax.api_base import ApiBaseJax, BufferSpec, call_cutedsl
from .config import RmsNormRhtAmaxPlan, validate_rmsnorm_rht_amax


class RmsNormRhtAmaxResult(NamedTuple):
    """Functional JAX outputs for RMSNorm + RHT + amax."""

    output: Any
    amax: Any


@dataclass(frozen=True)
class _RmsNormRhtAmaxJaxConfig:
    eps: float
    num_threads: Optional[int]
    rows_per_cta: Optional[int]


@lru_cache(maxsize=None)
def _make_launcher(
    n: int,
    num_threads: int,
    rows_per_cta: int,
    eps: float,
):
    # Load the configuration-specific kernel only when tracing the operation.
    from cutlass import Float32
    from .kernel import RMSNormRHTAmaxKernel

    kernel = RMSNormRHTAmaxKernel(
        n=n,
        num_threads=num_threads,
        eps=eps,
        rows_per_cta=rows_per_cta,
    )

    def launch(stream, x, weight, output, amax):
        kernel(x, weight, output, amax, Float32(eps), stream)

    return launch


class RmsNormRhtAmaxSm100(ApiBaseJax):
    """JAX callable specialized from sample shape and dtype metadata.

    Sample values are converted to descriptors during construction and are not
    retained. Actual arrays are passed to :meth:`__call__`, so this object can
    be used directly with ``jax.jit`` without capturing array constants.
    ``check_support()`` validates the abstract signature and static kernel
    configuration; final device capability is determined during lowering.
    """

    def __init__(
        self,
        sample_x: Any,
        sample_w: Any,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.x_desc = self.make_tensor_desc(sample_x, name="sample_x")
        self.w_desc = self.make_tensor_desc(sample_w, name="sample_w")
        self._config = _RmsNormRhtAmaxJaxConfig(
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )
        self._plan: Optional[RmsNormRhtAmaxPlan] = None
        self.num_threads: Optional[int] = None
        self.rows_per_cta: Optional[int] = None
        self.n: Optional[int] = None

    def _check_support(self) -> bool:
        config = self._config
        self._plan = validate_rmsnorm_rht_amax(
            self.x_desc,
            self.w_desc,
            num_threads=config.num_threads,
            rows_per_cta=config.rows_per_cta,
        )
        self.num_threads = self._plan.num_threads
        self.rows_per_cta = self._plan.rows_per_cta
        self.n = self._plan.n
        return True

    def __call__(self, x: Any, weight: Any) -> RmsNormRhtAmaxResult:
        """Run with arrays matching the validated sample signature."""

        return super().__call__(x, weight)

    def _call_impl(self, x: Any, weight: Any) -> RmsNormRhtAmaxResult:
        plan = self._plan
        if plan is None:
            raise RuntimeError("check_support() did not produce a launch plan")

        self.check_tensor_signature(x, self.x_desc, name="X")
        self.check_tensor_signature(weight, self.w_desc, name="W")

        # JAX/XLA owns physical buffers. These specs constrain ordinary
        # row-major storage while providing the divisibility facts used by the
        # CuTe kernel.
        x_spec = TensorSpec(divisibility=(plan.rows_per_cta, 16))
        weight_spec = TensorSpec(divisibility=(16,))

        output, amax = call_cutedsl(
            _make_launcher(plan.n, plan.num_threads, plan.rows_per_cta, self._config.eps),
            (x, weight),
            outputs=(
                BufferSpec(
                    "output",
                    plan.output_shape,
                    jnp.bfloat16,
                    tensor_spec=x_spec,
                ),
                BufferSpec(
                    "amax",
                    plan.amax_shape,
                    jnp.float32,
                ),
            ),
            input_specs=(x_spec, weight_spec),
            use_static_tensors=True,
        )
        return RmsNormRhtAmaxResult(output=output, amax=amax)


def rmsnorm_rht_amax_sm100(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> RmsNormRhtAmaxResult:
    """Apply fused RMSNorm, 16-wide RHT, and per-CTA amax from JAX.

    ``x`` must be a row-major ``bfloat16`` array of shape ``(M, N)`` and
    ``weight`` a ``bfloat16`` array of shape ``(N,)``. ``M`` and ``N`` must be
    concrete; shape-polymorphic export is unsupported. The underlying callable
    is not wrapped in ``jax.jit``; callers own JIT and sharding policy.

    ``eps``, ``num_threads``, and ``rows_per_cta`` are compile-time
    configuration. Returns ``(output, amax)`` with shapes ``(M, N)`` and
    ``(M // rows_per_cta,)`` respectively.
    """

    return RmsNormRhtAmaxSm100(
        x,
        weight,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )(x, weight)


__all__ = ["RmsNormRhtAmaxResult", "RmsNormRhtAmaxSm100", "rmsnorm_rht_amax_sm100"]
