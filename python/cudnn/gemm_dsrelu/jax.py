# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + squared-ReLU backward on SM100."""

from __future__ import annotations

import os
from typing import Any, Optional

import jax.numpy as jnp

from .._jax.api_base import ApiBaseJax, BufferSpec, TupleDict, call_cutedsl
from .._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_16_byte_extent,
    require_array,
    require_fp8_block_scales,
    require_gemm_inputs,
)
from ..gemm_validation import (
    require_full_mma_rows,
    require_shape,
    resolve_max_active_clusters,
)


def _launch(
    stream,
    a,
    b,
    c,
    sfa,
    sfb,
    prob,
    d,
    dprob,
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

    from .dense_blockscaled_gemm_persistent_dsrelu_quant import (
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

    def squared_relu_backward(x, upstream):
        return cute.where(x > 0, x, cute.full_like(x, 0)) * 2 * upstream

    kernel(
        a,
        b,
        sfa,
        sfb,
        c,
        d,
        prob,
        dprob,
        None,
        None,
        None,
        cutlass.Float32(alpha),
        max_active_clusters,
        stream,
        epilogue_op=squared_relu_backward,
    )


class GemmDsreluSm100(ApiBaseJax):
    """JAX squared-ReLU backward callable specialized from sample metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_c: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_prob: Any,
        alpha: float = 1.0,
        d_major: str = "n",
        d_dtype: Any = None,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[tuple[int, int]] = None,
        sample_norm_const: Optional[Any] = None,
        sf_vec_size: int = 32,
        vector_f32: bool = False,
        *,
        a_major: str = "k",
        b_major: str = "k",
    ) -> None:
        super().__init__()
        self.a_major = a_major
        self.b_major = b_major
        self.d_major = d_major
        self.a_spec = gemm_a_tensor_spec(a_major)
        self.b_spec = gemm_b_tensor_spec(b_major)
        self.output_spec = gemm_c_tensor_spec(d_major)
        self.scale_spec = block_scale_tensor_spec()
        self.prob_spec = probability_tensor_spec()

        self.a_desc = self.make_tensor_desc(sample_a, layout=self.a_spec.layout, name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, layout=self.b_spec.layout, name="sample_b")
        self.c_desc = self.make_tensor_desc(sample_c, layout=self.output_spec.layout, name="sample_c")
        self.sfa_desc = self.make_tensor_desc(sample_sfa, layout=self.scale_spec.layout, name="sample_sfa")
        self.sfb_desc = self.make_tensor_desc(sample_sfb, layout=self.scale_spec.layout, name="sample_sfb")
        self.prob_desc = self.make_tensor_desc(sample_prob, layout=self.prob_spec.layout, name="sample_prob")
        self.norm_const_desc = self.make_optional_tensor_desc(sample_norm_const, name="sample_norm_const")

        self.alpha = alpha
        self._d_dtype = self.as_optional_dtype(d_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = None if cluster_shape_mn is None else tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _check_support(self) -> bool:
        if self.norm_const_desc is not None:
            raise NotImplementedError("sample_norm_const is used by the FP8 output path, which is not available in the JAX squared-ReLU backward API")

        from .dense_blockscaled_gemm_persistent_dsrelu_quant import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )

        kernel = Sm100BlockScaledPersistentDenseGemmKernel
        self.m, self.n, self.k, self.batch, self.ab_dtype = require_gemm_inputs(self.a_desc, self.b_desc)
        supported_inputs = {jnp.dtype(jnp.float8_e4m3fn), jnp.dtype(jnp.float8_e5m2)}
        if self.ab_dtype not in supported_inputs:
            raise NotImplementedError("The JAX squared-ReLU backward API supports float8_e4m3fn and " f"float8_e5m2 inputs, got {self.ab_dtype}")
        require_fp8_block_scales(
            self.sfa_desc,
            self.sfb_desc,
            m=self.m,
            n=self.n,
            k=self.k,
            batch=self.batch,
            sf_vec_size=self.sf_vec_size,
        )

        c_shape = require_array("sample_c", self.c_desc, 3)
        require_shape("sample_c", c_shape, (self.m, self.n, self.batch))
        prob_shape = require_array("sample_prob", self.prob_desc, 3)
        require_shape("sample_prob", prob_shape, (self.m, 1, self.batch))
        self.require_dtype("sample_prob.dtype", self.prob_desc, (jnp.float32,))

        supported_outputs = (jnp.float16, jnp.bfloat16, jnp.float32)
        self.c_dtype = self.require_dtype("sample_c.dtype", self.c_desc, supported_outputs)
        self.d_dtype = self.require_dtype("d_dtype", self._d_dtype, supported_outputs, default=jnp.bfloat16)
        self.acc_dtype = self.require_dtype("acc_dtype", self._acc_dtype, (jnp.float32,), default=jnp.float32)

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

        require_16_byte_extent("sample_a", self.m if self.a_major == "m" else self.k, self.ab_dtype)
        require_16_byte_extent("sample_b", self.n if self.b_major == "n" else self.k, self.ab_dtype)
        require_16_byte_extent("sample_c", self.m if self.d_major == "m" else self.n, self.c_dtype)
        require_16_byte_extent("d_tensor", self.m if self.d_major == "m" else self.n, self.d_dtype)
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            prob_tensor,
            norm_const_tensor,
        )

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Optional[Any],
    ) -> TupleDict:
        self.check_tensor_signature(a_tensor, self.a_desc, name="A")
        self.check_tensor_signature(b_tensor, self.b_desc, name="B")
        self.check_tensor_signature(c_tensor, self.c_desc, name="C")
        self.check_tensor_signature(sfa_tensor, self.sfa_desc, name="SFA")
        self.check_tensor_signature(sfb_tensor, self.sfb_desc, name="SFB")
        self.check_tensor_signature(prob_tensor, self.prob_desc, name="prob")
        self.check_optional_tensor_signature(norm_const_tensor, self.norm_const_desc, name="norm_const")

        d_tensor, dprob_tensor = call_cutedsl(
            _launch,
            (a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor),
            outputs=(
                BufferSpec(
                    "d_tensor",
                    (self.m, self.n, self.batch),
                    self.d_dtype,
                    tensor_spec=self.output_spec,
                ),
                BufferSpec(
                    "dprob_tensor",
                    (self.m, 1, self.batch),
                    jnp.float32,
                    fill_value=0.0,
                ),
            ),
            input_specs=(
                self.a_spec,
                self.b_spec,
                self.output_spec,
                self.scale_spec,
                self.scale_spec,
                self.prob_spec,
            ),
            static_args={
                "alpha": float(self.alpha),
                "sf_vec_size": self.sf_vec_size,
                "mma_tiler_mn": self.mma_tiler_mn,
                "cluster_shape_mn": self.cluster_shape_mn,
                "vector_f32": bool(self.vector_f32),
                "cluster_overlap_margin": self.num_cluster_overlap_margin,
            },
            use_static_tensors=True,
        )
        return TupleDict(
            d_tensor=d_tensor,
            dprob_tensor=dprob_tensor,
            amax_tensor=None,
            sfd_tensor=None,
        )


def gemm_dsrelu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    d_major: str = "n",
    d_dtype: Any = None,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    norm_const_tensor: Optional[Any] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    *,
    a_major: str = "k",
    b_major: str = "k",
) -> TupleDict:
    """Compute the MXFP8 squared-ReLU backward fusion."""

    return GemmDsreluSm100(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        prob_tensor,
        alpha=alpha,
        d_major=d_major,
        d_dtype=d_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sample_norm_const=norm_const_tensor,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        a_major=a_major,
        b_major=b_major,
    )(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        prob_tensor,
        norm_const_tensor,
    )


__all__ = ["GemmDsreluSm100", "gemm_dsrelu_wrapper_sm100"]
