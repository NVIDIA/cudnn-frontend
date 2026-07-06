# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + squared ReLU on SM100."""

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
    require_array,
)
from .._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_16_byte_extent,
    require_fp8_block_scales,
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
    sfa,
    sfb,
    prob,
    c,
    d,
    *,
    alpha: float,
    sf_vec_size: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    vector_f32: bool,
    cluster_overlap_margin: int,
):
    # These operations happen during CUDA lowering, not abstract evaluation.
    import cutlass
    import cutlass.cute as cute

    from .dense_blockscaled_gemm_persistent_srelu_quant import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    kernel = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vector_f32=vector_f32,
    )
    max_active_clusters = resolve_max_active_clusters(
        cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
        cluster_overlap_margin,
    )

    def squared_relu(x):
        return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2

    kernel(
        a,
        b,
        sfa,
        sfb,
        c,
        d,
        prob,
        None,
        None,
        None,
        cutlass.Float32(alpha),
        max_active_clusters,
        stream,
        epilogue_op=squared_relu,
    )


class GemmSreluSm100(ApiBaseJax):
    """JAX GEMM + squared-ReLU callable specialized from sample metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_prob: Any,
        alpha: float = 1.0,
        c_layout: str = "LMN",
        c_dtype: Any = None,
        d_dtype: Any = None,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sample_norm_const: Optional[Any] = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        *,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        self.a_desc = self.make_tensor_desc(sample_a, tensor_spec=gemm_a_tensor_spec(a_layout), name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, tensor_spec=gemm_b_tensor_spec(b_layout), name="sample_b")
        self.sfa_desc = self.make_tensor_desc(sample_sfa, tensor_spec=block_scale_tensor_spec(), name="sample_sfa")
        self.sfb_desc = self.make_tensor_desc(sample_sfb, tensor_spec=block_scale_tensor_spec(), name="sample_sfb")
        self.prob_desc = self.make_tensor_desc(sample_prob, tensor_spec=probability_tensor_spec(), name="sample_prob")
        self.norm_const_desc = self.make_optional_tensor_desc(sample_norm_const, name="sample_norm_const")
        self._c_layout = c_layout

        self.alpha = alpha
        self._c_dtype = self.as_optional_dtype(c_dtype)
        self._d_dtype = self.as_optional_dtype(d_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = None if cluster_shape_mn is None else tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _check_support(self) -> None:
        if self.norm_const_desc is not None:
            raise NotImplementedError("sample_norm_const is used by the FP8 output path, which is not available in the JAX GEMM + squared-ReLU API")

        from .dense_blockscaled_gemm_persistent_srelu_quant import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )

        kernel = Sm100BlockScaledPersistentDenseGemmKernel
        self.m, self.n, self.k, self.batch, self.ab_dtype = require_gemm_inputs(self.a_desc, self.b_desc)
        supported_inputs = {jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)}
        if self.ab_dtype not in supported_inputs:
            raise NotImplementedError("The JAX GEMM + squared-ReLU API supports float8_e4m3fn and " f"float8_e5m2 inputs, got {self.ab_dtype}")
        require_fp8_block_scales(
            self.sfa_desc,
            self.sfb_desc,
            m=self.m,
            n=self.n,
            k=self.k,
            batch=self.batch,
            sf_vec_size=self.sf_vec_size,
        )
        require_array(
            self.prob_desc,
            shape=(self.m, 1, self.batch),
            dtype=jnp.float32,
        )

        supported_outputs = (jnp.float16, jnp.bfloat16, jnp.float32)
        self.c_dtype = self.require_dtype(
            self._c_dtype,
            supported_outputs,
            name="c_dtype",
            default=jnp.bfloat16,
        )
        self.d_dtype = self.require_dtype(
            self._d_dtype,
            supported_outputs,
            name="d_dtype",
            default=jnp.bfloat16,
        )
        self.require_dtype(
            self._acc_dtype,
            (jnp.float32,),
            name="acc_dtype",
            default=jnp.float32,
        )

        self.mma_tiler_mn = kernel.require_mma_tiler(self.mma_tiler_mn)
        require_full_mma_rows(
            self.m,
            self.mma_tiler_mn[0],
            cta_group_size=(2 if self.mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else 1),
            reason="the probability load is not predicated",
        )
        if self.cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 1) if self.mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
        self.cluster_shape_mn = kernel.require_cluster_shape(
            self.cluster_shape_mn,
            mma_tiler_mn=self.mma_tiler_mn,
        )

        require_16_byte_extent("sample_a", self.a_desc.shape[self.a_desc.stride_order[0]], self.ab_dtype)
        require_16_byte_extent("sample_b", self.b_desc.shape[self.b_desc.stride_order[0]], self.ab_dtype)

        output_spec = gemm_c_tensor_spec(self._c_layout)
        output_shape = (self.m, self.n, self.batch)
        self.c_desc = JaxTensorDesc(
            dtype=self.c_dtype,
            shape=output_shape,
            tensor_spec=output_spec,
            name="c_tensor",
        )
        self.d_desc = JaxTensorDesc(
            dtype=self.d_dtype,
            shape=output_shape,
            tensor_spec=output_spec,
            name="d_tensor",
        )
        require_16_byte_extent("c_tensor", self.c_desc.shape[self.c_desc.stride_order[0]], self.c_dtype)
        require_16_byte_extent("d_tensor", self.d_desc.shape[self.d_desc.stride_order[0]], self.d_dtype)

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor, norm_const_tensor)

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Optional[Any],
    ) -> TupleDict:
        self.check_tensor_signature(a_tensor, self.a_desc, name="A")
        self.check_tensor_signature(b_tensor, self.b_desc, name="B")
        self.check_tensor_signature(sfa_tensor, self.sfa_desc, name="SFA")
        self.check_tensor_signature(sfb_tensor, self.sfb_desc, name="SFB")
        self.check_tensor_signature(prob_tensor, self.prob_desc, name="prob")
        self.check_optional_tensor_signature(norm_const_tensor, self.norm_const_desc, name="norm_const")

        c_tensor, d_tensor = call_cutedsl(
            _launch,
            (a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor),
            outputs=(
                BufferSpec(
                    "c_tensor",
                    self.c_desc.array_shape,
                    self.c_desc.dtype,
                    tensor_spec=self.c_desc.tensor_spec,
                ),
                BufferSpec(
                    "d_tensor",
                    self.d_desc.array_shape,
                    self.d_desc.dtype,
                    tensor_spec=self.d_desc.tensor_spec,
                ),
            ),
            input_specs=(
                self.a_desc.tensor_spec,
                self.b_desc.tensor_spec,
                self.sfa_desc.tensor_spec,
                self.sfb_desc.tensor_spec,
                self.prob_desc.tensor_spec,
            ),
            static_args={
                "alpha": float(self.alpha),
                "sf_vec_size": self.sf_vec_size,
                "mma_tiler_mn": self.mma_tiler_mn,
                "cluster_shape_mn": self.cluster_shape_mn,
                "vector_f32": bool(self.vector_f32),
                "cluster_overlap_margin": self.num_cluster_overlap_margin,
            },
        )
        return TupleDict(
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            amax_tensor=None,
            sfd_tensor=None,
        )


def gemm_srelu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    c_layout: str = "LMN",
    c_dtype: Any = None,
    d_dtype: Any = None,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    norm_const_tensor: Optional[Any] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute MXFP8 GEMM and ``D = relu(C)**2 * prob``."""

    return GemmSreluSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        prob_tensor,
        alpha=alpha,
        c_layout=c_layout,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sample_norm_const=norm_const_tensor,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor, norm_const_tensor)


__all__ = ["GemmSreluSm100", "gemm_srelu_wrapper_sm100"]
