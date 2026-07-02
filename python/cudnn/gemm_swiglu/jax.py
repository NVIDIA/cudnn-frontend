# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense GEMM + SwiGLU on SM100."""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp

from .._jax.api_base import ApiBaseJax, BufferSpec, TupleDict, call_cutedsl
from .._jax.gemm import (
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    require_16_byte_extent,
    require_gemm_inputs,
)
from ..gemm_validation import (
    require_full_mma_rows,
    resolve_max_active_clusters,
)


def _launch(
    stream,
    a,
    b,
    ab12,
    c,
    *,
    alpha: float,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    cluster_overlap_margin: int,
):
    # These operations happen during CUDA lowering, not abstract evaluation.
    import cutlass
    from cutlass.jax import jax_to_cutlass_dtype

    from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

    kernel = PersistentDenseGemmKernel(
        acc_dtype=jax_to_cutlass_dtype(acc_dtype),
        use_2cta_instrs=mma_tiler_mn[0] == PersistentDenseGemmKernel.TWO_CTA_MMA_TILER_M,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
    )
    max_active_clusters = resolve_max_active_clusters(
        cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
        cluster_overlap_margin,
    )
    kernel(
        a,
        b,
        ab12,
        c,
        cutlass.Float32(alpha),
        max_active_clusters,
        stream,
    )


class GemmSwigluSm100(ApiBaseJax):
    """JAX GEMM + SwiGLU callable specialized from sample metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        alpha: float = 1.0,
        c_major: str = "n",
        ab12_dtype: Any = None,
        c_dtype: Any = None,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sample_sfa: Optional[Any] = None,
        sample_sfb: Optional[Any] = None,
        sample_norm_const: Optional[Any] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        ab12_stages: int = 4,
        *,
        a_major: str = "k",
        b_major: str = "k",
    ) -> None:
        super().__init__()
        self.a_major = a_major
        self.b_major = b_major
        self.c_major = c_major
        self.a_spec = gemm_a_tensor_spec(a_major)
        self.b_spec = gemm_b_tensor_spec(b_major)
        self.c_spec = gemm_c_tensor_spec(c_major)
        self.a_desc = self.make_tensor_desc(sample_a, layout=self.a_spec.layout, name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, layout=self.b_spec.layout, name="sample_b")
        self.sfa_desc = self.make_optional_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self.make_optional_tensor_desc(sample_sfb, name="sample_sfb")
        self.norm_const_desc = self.make_optional_tensor_desc(sample_norm_const, name="sample_norm_const")

        self.alpha = alpha
        self._ab12_dtype = self.as_optional_dtype(ab12_dtype)
        self._c_dtype = self.as_optional_dtype(c_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = None if cluster_shape_mn is None else tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.ab12_stages = ab12_stages
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _check_support(self) -> bool:
        if any(desc is not None for desc in (self.sfa_desc, self.sfb_desc, self.norm_const_desc)):
            raise NotImplementedError(
                "The JAX GEMM + SwiGLU API currently supports only the unquantized path; " "sample_sfa, sample_sfb, and sample_norm_const must be None"
            )

        from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

        kernel = PersistentDenseGemmKernel
        self.m, self.n, self.k, self.batch, a_dtype = require_gemm_inputs(self.a_desc, self.b_desc)
        self.output_n = kernel.get_output_n(self.n)
        self.a_dtype = self.require_dtype(
            "sample_a.dtype",
            a_dtype,
            (
                jnp.float16,
                jnp.bfloat16,
                jnp.float32,
                jnp.float8_e4m3fn,
                jnp.float8_e5m2,
            ),
        )
        self.acc_dtype = self.require_dtype(
            "acc_dtype",
            self._acc_dtype,
            (jnp.float32, jnp.float16),
            default=jnp.float32,
        )
        if self.acc_dtype == jnp.dtype(jnp.float32):
            supported_ab12 = (jnp.float32, jnp.float16, jnp.bfloat16)
        else:
            supported_ab12 = (jnp.float16, jnp.bfloat16)
            if self.a_dtype not in {
                jnp.dtype(jnp.float16),
                jnp.dtype(jnp.float8_e4m3fn),
                jnp.dtype(jnp.float8_e5m2),
            }:
                raise ValueError(f"float16 accumulation does not support input dtype {self.a_dtype}")
        self.ab12_dtype = self.require_dtype("ab12_dtype", self._ab12_dtype, supported_ab12, default=jnp.float32)
        self.c_dtype = self.require_dtype("c_dtype", self._c_dtype, (jnp.float16, jnp.bfloat16), default=jnp.float16)

        self.mma_tiler_mn = kernel.require_mma_tiler(self.mma_tiler_mn)
        if self.mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M:
            require_full_mma_rows(
                self.m,
                self.mma_tiler_mn[0],
                reason="2-CTA MMA requires a complete CTA pair",
            )
        if self.cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 2) if self.mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
        self.cluster_shape_mn = kernel.require_cluster_shape(
            self.cluster_shape_mn,
            mma_tiler_mn=self.mma_tiler_mn,
        )

        require_16_byte_extent("sample_a", self.m if self.a_major == "m" else self.k, self.a_dtype)
        require_16_byte_extent("sample_b", self.n if self.b_major == "n" else self.k, self.a_dtype)
        require_16_byte_extent("ab12_tensor", self.m if self.c_major == "m" else self.n, self.ab12_dtype)
        require_16_byte_extent("c_tensor", self.m if self.c_major == "m" else self.output_n, self.c_dtype)
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Optional[Any] = None,
        sfb_tensor: Optional[Any] = None,
        norm_const_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(a_tensor, b_tensor, sfa_tensor, sfb_tensor, norm_const_tensor)

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Optional[Any],
        sfb_tensor: Optional[Any],
        norm_const_tensor: Optional[Any],
    ) -> TupleDict:
        self.check_tensor_signature(a_tensor, self.a_desc, name="A")
        self.check_tensor_signature(b_tensor, self.b_desc, name="B")
        self.check_optional_tensor_signature(sfa_tensor, self.sfa_desc, name="SFA")
        self.check_optional_tensor_signature(sfb_tensor, self.sfb_desc, name="SFB")
        self.check_optional_tensor_signature(norm_const_tensor, self.norm_const_desc, name="norm_const")

        ab12_tensor, c_tensor = call_cutedsl(
            _launch,
            (a_tensor, b_tensor),
            outputs=(
                BufferSpec(
                    "ab12_tensor",
                    (self.m, self.n, self.batch),
                    self.ab12_dtype,
                    tensor_spec=self.c_spec,
                ),
                BufferSpec(
                    "c_tensor",
                    (self.m, self.output_n, self.batch),
                    self.c_dtype,
                    tensor_spec=self.c_spec,
                ),
            ),
            input_specs=(self.a_spec, self.b_spec),
            static_args={
                "alpha": float(self.alpha),
                "acc_dtype": self.acc_dtype,
                "mma_tiler_mn": self.mma_tiler_mn,
                "cluster_shape_mn": self.cluster_shape_mn,
                "cluster_overlap_margin": self.num_cluster_overlap_margin,
            },
            use_static_tensors=True,
        )
        return TupleDict(
            ab12_tensor=ab12_tensor,
            c_tensor=c_tensor,
            sfc_tensor=None,
            amax_tensor=None,
        )


def gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    c_major: str = "n",
    ab12_dtype: Any = None,
    c_dtype: Any = None,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sfa_tensor: Optional[Any] = None,
    sfb_tensor: Optional[Any] = None,
    norm_const_tensor: Optional[Any] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
    *,
    a_major: str = "k",
    b_major: str = "k",
) -> TupleDict:
    """Compute a dense batched GEMM and its fused SwiGLU projection."""

    return GemmSwigluSm100(
        a_tensor,
        b_tensor,
        alpha=alpha,
        c_major=c_major,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sample_sfa=sfa_tensor,
        sample_sfb=sfb_tensor,
        sample_norm_const=norm_const_tensor,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        ab12_stages=ab12_stages,
        a_major=a_major,
        b_major=b_major,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, norm_const_tensor)


__all__ = ["GemmSwigluSm100", "gemm_swiglu_wrapper_sm100"]
