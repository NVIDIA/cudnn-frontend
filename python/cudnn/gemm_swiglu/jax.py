# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the standard dense GEMM + SwiGLU kernel."""

from __future__ import annotations

from functools import partial
import os
from typing import Any

import jax
import jax.numpy as jnp

from .. import data_type
from .._cute_compiler import compile_options_for_target
from .._dense_gemm import require_gemm_inputs
from .._jax import JaxApiBase, JaxTensorDesc, TupleDict
from .._jax.datatypes import jax_to_cudnn_dtype
from .._jax.gemm import gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import to_public_axes
from .op import GemmSwigluSm100Op

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)


def _normalize_dtype(value: Any | None, default: Any, name: str) -> Any:
    try:
        return jnp.dtype(default if value is None else value)
    except TypeError as error:
        raise TypeError(f"{name} must be a JAX dtype, got {value!r}") from error


class GemmSwigluSm100(JaxApiBase):
    """JAX callable specialized from dense GEMM input metadata.

    ``a_layout``, ``b_layout``, and ``c_layout`` describe the public JAX axis
    order. Descriptors stored by the operation remain in the kernel's
    canonical ``MKL``, ``NKL``, and ``MNL`` orders.

    This first JAX surface intentionally selects only the standard dense
    kernel. Block-scaled inputs and quantized outputs are added separately so
    their packed scale-factor ABI does not complicate the base contract.
    """

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        *,
        sample_ab12: Any | None = None,
        sample_c: Any | None = None,
        alpha: float = 1.0,
        c_layout: str = "LMN",
        ab12_dtype: Any | None = None,
        c_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] | None = None,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
        target_compute_capability: int | None = None,
    ) -> None:
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.c_layout = c_layout
        self.a_mode = gemm_a_mode(a_layout)
        self.b_mode = gemm_b_mode(b_layout)
        self.output_mode = gemm_output_mode(c_layout)

        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "GemmSwigluSm100",
        )
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=self.a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=self.b_mode)
        self.acc_dtype = _normalize_dtype(acc_dtype, jnp.float32, "acc_dtype")

        if (sample_ab12 is None) != (sample_c is None):
            raise ValueError("sample_ab12 and sample_c must be provided together")
        if sample_ab12 is None:
            resolved_ab12_dtype = _normalize_dtype(ab12_dtype, jnp.float32, "ab12_dtype")
            resolved_c_dtype = _normalize_dtype(c_dtype, jnp.float16, "c_dtype")
            self.ab12_desc, self.c_desc = self._default_output_descs(
                resolved_ab12_dtype,
                resolved_c_dtype,
            )
        else:
            self.ab12_desc = self._to_tensor_desc(
                sample_ab12,
                "sample_ab12",
                mode=self.output_mode,
            )
            self.c_desc = self._to_tensor_desc(
                sample_c,
                "sample_c",
                mode=self.output_mode,
            )
            self._check_requested_output_dtype(ab12_dtype, self.ab12_desc, "ab12_dtype")
            self._check_requested_output_dtype(c_dtype, self.c_desc, "c_dtype")

        acc_cudnn_dtype = jax_to_cudnn_dtype(self.acc_dtype)
        if acc_cudnn_dtype == data_type.NOT_SET:
            raise ValueError(f"Unsupported JAX accumulator dtype {self.acc_dtype}")

        self._op = GemmSwigluSm100Op(
            a=self.a_desc,
            b=self.b_desc,
            ab12=self.ab12_desc,
            c=self.c_desc,
            alpha=alpha,
            acc_dtype=acc_cudnn_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _default_output_descs(
        self,
        ab12_dtype: Any,
        c_dtype: Any,
    ) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        m, n, _, batch = require_gemm_inputs(self.a_desc, self.b_desc)
        if n % 2:
            raise ValueError(f"SwiGLU requires an even N dimension, got {n}")

        return (
            self._to_tensor_desc(
                jax.ShapeDtypeStruct(
                    to_public_axes((m, n, batch), self.output_mode),
                    ab12_dtype,
                ),
                "sample_ab12",
                mode=self.output_mode,
            ),
            self._to_tensor_desc(
                jax.ShapeDtypeStruct(
                    to_public_axes((m, n // 2, batch), self.output_mode),
                    c_dtype,
                ),
                "sample_c",
                mode=self.output_mode,
            ),
        )

    @staticmethod
    def _check_requested_output_dtype(
        requested: Any | None,
        desc: JaxTensorDesc,
        name: str,
    ) -> None:
        if requested is None:
            return
        requested_dtype = _normalize_dtype(requested, requested, name)
        actual_dtype = jnp.dtype(desc.dtype)
        if requested_dtype != actual_dtype:
            raise ValueError(f"{name}={requested_dtype} does not match the explicit sample dtype {actual_dtype}")

    def check_support(self) -> bool:
        return self._op.check_support()

    def __call__(self, a_tensor: Any, b_tensor: Any) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(a_tensor, self.a_desc, mode=self.a_mode)
        self._check_tensor_signature(b_tensor, self.b_desc, mode=self.b_mode)

        ab12_tensor, c_tensor = self._call_kernel(
            (a_tensor, b_tensor),
            output_descs=(self.ab12_desc, self.c_desc),
            input_spec=(
                self._to_tensor_spec(self.a_desc, mode=self.a_mode),
                self._to_tensor_spec(self.b_desc, mode=self.b_mode),
            ),
            output_spec=(
                self._to_tensor_spec(self.ab12_desc, mode=self.output_mode),
                self._to_tensor_spec(self.c_desc, mode=self.output_mode),
            ),
            compile_options=compile_options_for_target(self.target_compute_capability),
        )
        return TupleDict(
            ab12_tensor=ab12_tensor,
            c_tensor=c_tensor,
            sfc_tensor=None,
            amax_tensor=None,
        )

    def _launch(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        if workspaces:
            raise RuntimeError(f"GemmSwigluSm100 received unexpected workspaces: {len(workspaces)}")
        a, b = inputs
        ab12, c = outputs

        import cutlass
        from cutlass.jax import jax_to_cutlass_dtype

        from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

        kernel = PersistentDenseGemmKernel(
            acc_dtype=jax_to_cutlass_dtype(self.acc_dtype),
            use_2cta_instrs=self._op.mma_tiler_mn[0] == 256,
            mma_tiler_mn=self._op.mma_tiler_mn,
            cluster_shape_mn=self._op.cluster_shape_mn,
        )
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(self._op.cluster_shape_mn[0] * self._op.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be positive after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        kernel(
            a,
            b,
            ab12,
            c,
            cutlass.Float32(self._op.alpha),
            max_active_clusters,
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "alpha",
        "c_layout",
        "ab12_dtype",
        "c_dtype",
        "acc_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "a_layout",
        "b_layout",
        "target_compute_capability",
    ),
)
def gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    c_layout: str = "LMN",
    ab12_dtype: Any | None = None,
    c_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] | None = None,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute the standard dense batched GEMM and fused SwiGLU projection."""

    return GemmSwigluSm100(
        jax.ShapeDtypeStruct(a_tensor.shape, a_tensor.dtype),
        jax.ShapeDtypeStruct(b_tensor.shape, b_tensor.dtype),
        alpha=alpha,
        c_layout=c_layout,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        a_layout=a_layout,
        b_layout=b_layout,
        target_compute_capability=target_compute_capability,
    )(a_tensor, b_tensor)


__all__ = [
    "GemmSwigluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gemm_swiglu_wrapper_sm100",
]
