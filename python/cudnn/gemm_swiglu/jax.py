# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense GEMM + SwiGLU on SM100."""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp

from .._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    JaxTensorDesc,
    TupleDict,
    call_cutedsl,
)
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
        c_layout: str = "LMN",
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
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        self.a_desc = self.make_tensor_desc(sample_a, tensor_spec=gemm_a_tensor_spec(a_layout), name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, tensor_spec=gemm_b_tensor_spec(b_layout), name="sample_b")
        self.sfa_desc = self.make_optional_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self.make_optional_tensor_desc(sample_sfb, name="sample_sfb")
        self.norm_const_desc = self.make_optional_tensor_desc(sample_norm_const, name="sample_norm_const")
        self._c_layout = c_layout

        self.alpha = alpha
        self._ab12_dtype = self.as_optional_dtype(ab12_dtype)
        self._c_dtype = self.as_optional_dtype(c_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = None if cluster_shape_mn is None else tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = bool(vector_f32)
        self.ab12_stages = ab12_stages
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _check_support(self) -> None:
        if any(desc is not None for desc in (self.sfa_desc, self.sfb_desc, self.norm_const_desc)):
            raise NotImplementedError(
                "The JAX GEMM + SwiGLU API currently supports only the unquantized path; " "sample_sfa, sample_sfb, and sample_norm_const must be None"
            )
        if self.sf_vec_size != 16:
            raise NotImplementedError(f"The unquantized JAX GEMM + SwiGLU path supports only sf_vec_size=16, got {self.sf_vec_size}")
        if self.vector_f32:
            raise NotImplementedError(f"The unquantized JAX GEMM + SwiGLU path supports only vector_f32=False, got {self.vector_f32}")
        if self.ab12_stages != 4:
            raise NotImplementedError(f"The unquantized JAX GEMM + SwiGLU path supports only ab12_stages=4, got {self.ab12_stages}")

        from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

        kernel = PersistentDenseGemmKernel
        self.m, self.n, self.k, self.batch, a_dtype = require_gemm_inputs(self.a_desc, self.b_desc)
        self.output_n = kernel.get_output_n(self.n)
        self.a_dtype = self.require_dtype(
            a_dtype,
            (
                jnp.float16,
                jnp.bfloat16,
                jnp.float32,
                jnp.float8_e4m3fn,
                jnp.float8_e5m2,
            ),
            name="sample_a.dtype",
        )
        self.acc_dtype = self.require_dtype(
            self._acc_dtype,
            (jnp.float32, jnp.float16),
            name="acc_dtype",
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
        self.ab12_dtype = self.require_dtype(
            self._ab12_dtype,
            supported_ab12,
            name="ab12_dtype",
            default=jnp.float32,
        )
        self.c_dtype = self.require_dtype(
            self._c_dtype,
            (jnp.float16, jnp.bfloat16),
            name="c_dtype",
            default=jnp.float16,
        )

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

        require_16_byte_extent("sample_a", self.a_desc.shape[self.a_desc.stride_order[0]], self.a_dtype)
        require_16_byte_extent("sample_b", self.b_desc.shape[self.b_desc.stride_order[0]], self.a_dtype)

        output_spec = gemm_c_tensor_spec(self._c_layout)
        self.ab12_desc = JaxTensorDesc(
            dtype=self.ab12_dtype,
            shape=(self.m, self.n, self.batch),
            tensor_spec=output_spec,
            name="ab12_tensor",
        )
        self.c_desc = JaxTensorDesc(
            dtype=self.c_dtype,
            shape=(self.m, self.output_n, self.batch),
            tensor_spec=output_spec,
            name="c_tensor",
        )
        require_16_byte_extent(
            "ab12_tensor",
            self.ab12_desc.shape[self.ab12_desc.stride_order[0]],
            self.ab12_dtype,
        )
        require_16_byte_extent("c_tensor", self.c_desc.shape[self.c_desc.stride_order[0]], self.c_dtype)

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
                    self.ab12_desc.array_shape,
                    self.ab12_desc.dtype,
                    tensor_spec=self.ab12_desc.tensor_spec,
                ),
                BufferSpec(
                    "c_tensor",
                    self.c_desc.array_shape,
                    self.c_desc.dtype,
                    tensor_spec=self.c_desc.tensor_spec,
                ),
            ),
            input_specs=(self.a_desc.tensor_spec, self.b_desc.tensor_spec),
            static_args={
                "alpha": float(self.alpha),
                "acc_dtype": self.acc_dtype,
                "mma_tiler_mn": self.mma_tiler_mn,
                "cluster_shape_mn": self.cluster_shape_mn,
                "cluster_overlap_margin": self.num_cluster_overlap_margin,
            },
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
    c_layout: str = "LMN",
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
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute a dense batched GEMM and its fused SwiGLU projection."""

    return GemmSwigluSm100(
        a_tensor,
        b_tensor,
        alpha=alpha,
        c_layout=c_layout,
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
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, norm_const_tensor)


__all__ = ["GemmSwigluSm100", "gemm_swiglu_wrapper_sm100"]
