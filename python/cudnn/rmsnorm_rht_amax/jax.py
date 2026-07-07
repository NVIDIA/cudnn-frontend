# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for fused RMSNorm + RHT + per-CTA amax."""

from __future__ import annotations

from typing import Any, Optional

import jax

from .. import data_type
from .._jax import JaxApiBase, JaxTensorDesc
from .kernel import RMSNormRHTAmaxKernel
from .op import RmsNormRhtAmaxSm100Op, pick_rows_per_cta


class RmsNormRhtAmaxSm100(JaxApiBase):
    """JAX callable specialized from sample shape and dtype metadata."""

    def __init__(
        self,
        sample_x: Any,
        sample_w: Any,
        *,
        sample_o: Any | None = None,
        sample_amax: Any | None = None,
        eps: float = 1e-5,
        num_threads: Optional[int] = None,
        rows_per_cta: Optional[int] = None,
    ) -> None:
        self.x_desc = self._to_tensor_desc(sample_x, "sample_x")
        self.w_desc = self._to_tensor_desc(sample_w, "sample_w")
        if (sample_o is None) != (sample_amax is None):
            raise ValueError("sample_o and sample_amax must be provided together")

        if sample_o is None:
            self.o_desc, self.amax_desc = self._default_output_descs(rows_per_cta)
        else:
            self.o_desc = self._to_tensor_desc(sample_o, "sample_o")
            self.amax_desc = self._to_tensor_desc(sample_amax, "sample_amax")

        self._op = RmsNormRhtAmaxSm100Op(
            x=self.x_desc,
            weight=self.w_desc,
            output=self.o_desc,
            amax=self.amax_desc,
            eps=eps,
            num_threads=num_threads,
            rows_per_cta=rows_per_cta,
        )

    def _default_output_descs(self, rows_per_cta: Optional[int]) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        if self.x_desc.ndim != 2:
            raise ValueError(f"X must have rank 2, got shape {self.x_desc.shape}")
        m, n = self.x_desc.shape
        resolved_rows_per_cta = pick_rows_per_cta(m) if rows_per_cta is None else rows_per_cta
        if resolved_rows_per_cta <= 0:
            raise ValueError(f"rows_per_cta must be positive, got {resolved_rows_per_cta}")

        return (
            self.x_desc.compact_like(
                cudnn_dtype=data_type.BFLOAT16,
                shape=(m, n),
                name="sample_o",
            ),
            self.x_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(m // resolved_rows_per_cta,),
                name="sample_amax",
            ),
        )

    def check_support(self) -> bool:
        self._op.check_support()
        self._check_device_compatibility(
            minimum_compute_capability=100,
            operation_name="RmsNormRhtAmaxSm100",
        )
        return True

    def __call__(self, x: Any, weight: Any) -> tuple[Any, Any]:
        self.check_support()
        self._check_tensor_signature(x, self.x_desc)
        self._check_tensor_signature(weight, self.w_desc)

        x_spec = self._to_tensor_spec(
            self.x_desc,
            divisibility=(self._op.rows_per_cta, 16),
        )
        weight_spec = self._to_tensor_spec(self.w_desc, divisibility=(16,))
        output_spec = self._to_tensor_spec(
            self.o_desc,
            divisibility=(self._op.rows_per_cta, 16),
        )
        return self._call_kernel(
            (x, weight),
            output_descs=(self.o_desc, self.amax_desc),
            input_spec=(x_spec, weight_spec),
            output_spec=(output_spec, None),
        )

    def _launch(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        x, weight = inputs
        output, amax = outputs
        kernel = RMSNormRHTAmaxKernel(
            n=self._op.n,
            num_threads=self._op.num_threads,
            eps=self._op.eps,
            rows_per_cta=self._op.rows_per_cta,
        )
        kernel(x, weight, output, amax, stream)


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

    sample_x = jax.ShapeDtypeStruct(x.shape, x.dtype)
    sample_weight = jax.ShapeDtypeStruct(weight.shape, weight.dtype)
    return RmsNormRhtAmaxSm100(
        sample_x,
        sample_weight,
        eps=eps,
        num_threads=num_threads,
        rows_per_cta=rows_per_cta,
    )(x, weight)


__all__ = [
    "RmsNormRhtAmaxSm100",
    "rmsnorm_rht_amax_sm100",
]
