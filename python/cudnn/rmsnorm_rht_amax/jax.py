# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for fused RMSNorm + RHT + per-CTA amax."""

from __future__ import annotations

from typing import Any, Optional

import jax

from .._jax import JaxApiBase
from .kernel import RMSNormRHTAmaxKernel


class RmsNormRhtAmaxSm100(JaxApiBase):
    """JAX callable specialized from sample shape and dtype metadata."""

    def __init__(
        self,
        sample_x: Any,
        sample_w: Any,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ) -> None:
        self.x_desc = self._to_tensor_desc(sample_x, "sample_x")
        self.w_desc = self._to_tensor_desc(sample_w, "sample_w")
        self.kernel = RMSNormRHTAmaxKernel(
            x=self.x_desc,
            weight=self.w_desc,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )

    def check_support(self) -> bool:
        return self.kernel.check_support()

    def __call__(self, x: Any, weight: Any) -> tuple[Any, Any]:
        self.check_support()
        self._check_tensor_signature(x, self.x_desc)
        self._check_tensor_signature(weight, self.w_desc)

        x_spec = self._to_tensor_spec(
            self.x_desc,
            divisibility=(self.kernel.rows_per_cta, 16),
        )
        weight_spec = self._to_tensor_spec(self.w_desc, divisibility=(16,))
        return self._call_kernel(
            (x, weight),
            input_spec=(x_spec, weight_spec),
            output_spec=(x_spec, None),
        )


@jax.jit(static_argnames=("eps", "num_threads", "rows_per_cta"))
def rmsnorm_rht_amax_sm100(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-5,
    num_threads: Optional[int] = None,
    rows_per_cta: Optional[int] = None,
) -> tuple[Any, Any]:
    """Apply fused RMSNorm, 16-wide RHT, and per-CTA amax from JAX."""

    return RmsNormRhtAmaxSm100(
        x,
        weight,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )(x, weight)


__all__ = [
    "RmsNormRhtAmaxSm100",
    "rmsnorm_rht_amax_sm100",
]
