# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for block-scaled dense GEMM + squared-ReLU backward."""

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
from .._jax.datatypes import jax_to_cudnn_dtype, normalize_jax_dtype
from .._jax.gemm import BLOCK_SCALE_MODE, PROBABILITY_MODE, gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import to_public_axes
from .op import GemmDsreluSm100Op

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
_JAX_INPUT_DTYPES = frozenset({data_type.FP4_E2M1, data_type.FP8_E4M3, data_type.FP8_E5M2})
_WIDE_OUTPUT_DTYPES = frozenset({data_type.HALF, data_type.BFLOAT16, data_type.FLOAT})


class GemmDsreluSm100(JaxApiBase):
    """JAX callable specialized from a block-scaled squared-ReLU backward signature."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_c: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_prob: Any,
        *,
        sample_d: Any | None = None,
        sample_dprob: Any | None = None,
        sample_norm_const: Any | None = None,
        alpha: float = 1.0,
        d_layout: str = "LMN",
        d_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        self.a_mode = gemm_a_mode(a_layout)
        self.b_mode = gemm_b_mode(b_layout)
        self.output_mode = gemm_output_mode(d_layout, name="d_layout")
        self.scale_mode = BLOCK_SCALE_MODE
        self.probability_mode = PROBABILITY_MODE

        self.compute_capability = self._resolve_compute_capability(
            None,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "GemmDsreluSm100",
        )
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=self.a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=self.b_mode)
        self.c_desc = self._to_tensor_desc(sample_c, "sample_c", mode=self.output_mode)
        self.sfa_desc = self._to_tensor_desc(sample_sfa, "sample_sfa", mode=self.scale_mode)
        self.sfb_desc = self._to_tensor_desc(sample_sfb, "sample_sfb", mode=self.scale_mode)
        self.prob_desc = self._to_tensor_desc(sample_prob, "sample_prob", mode=self.probability_mode)

        self.acc_dtype = normalize_jax_dtype(acc_dtype, jnp.float32, "acc_dtype")
        resolved_d_dtype = normalize_jax_dtype(d_dtype, jnp.bfloat16, "d_dtype")
        if (sample_d is None) != (sample_dprob is None):
            raise ValueError("sample_d and sample_dprob must be provided together")
        if sample_d is None:
            self.d_desc, self.dprob_desc = self._default_output_descs(resolved_d_dtype)
        else:
            self.d_desc = self._to_tensor_desc(sample_d, "sample_d", mode=self.output_mode)
            self.dprob_desc = self._to_tensor_desc(
                sample_dprob,
                "sample_dprob",
                mode=self.probability_mode,
                init_value=0.0,
            )
            self._check_requested_output_dtype(d_dtype, self.d_desc, "d_dtype")

        self.amax_desc = None
        if self.a_desc.cudnn_dtype == data_type.FP4_E2M1 and self.d_desc.cudnn_dtype in _WIDE_OUTPUT_DTYPES:
            self.amax_desc = self.d_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(1,),
                name="amax_tensor",
                init_value=float("-inf"),
            )
        self.norm_const_desc = None
        self.sfd_desc = None
        if self.d_desc.cudnn_dtype in {data_type.FP8_E4M3, data_type.FP8_E5M2}:
            raise NotImplementedError("FP8 D output is unavailable because the current dsReLU kernel does not implement SFD generation")
        if sample_norm_const is not None:
            raise ValueError("sample_norm_const is only used with FP8 D output, which is not implemented")

        acc_cudnn_dtype = jax_to_cudnn_dtype(self.acc_dtype)
        if acc_cudnn_dtype == data_type.NOT_SET:
            raise ValueError(f"Unsupported JAX accumulator dtype {self.acc_dtype}")
        if not isinstance(vector_f32, bool):
            raise TypeError(f"vector_f32 must be a bool, got {type(vector_f32).__name__}")

        self._op = GemmDsreluSm100Op(
            a=self.a_desc,
            b=self.b_desc,
            c=self.c_desc,
            d=self.d_desc,
            sfa=self.sfa_desc,
            sfb=self.sfb_desc,
            prob=self.prob_desc,
            dprob=self.dprob_desc,
            sfd=self.sfd_desc,
            amax=self.amax_desc,
            norm_const=self.norm_const_desc,
            alpha=alpha,
            acc_dtype=acc_cudnn_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        self.vector_f32 = vector_f32
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _default_output_descs(self, d_dtype: Any) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        m, n, _, batch = require_gemm_inputs(self.a_desc, self.b_desc)
        d_shape = to_public_axes((m, n, batch), self.output_mode)
        dprob_shape = to_public_axes((m, 1, batch), self.probability_mode)
        return (
            self._to_tensor_desc(
                jax.ShapeDtypeStruct(d_shape, d_dtype),
                "sample_d",
                mode=self.output_mode,
            ),
            self._to_tensor_desc(
                jax.ShapeDtypeStruct(dprob_shape, jnp.float32),
                "sample_dprob",
                mode=self.probability_mode,
                init_value=0.0,
            ),
        )

    @staticmethod
    def _check_requested_output_dtype(requested: Any | None, desc: JaxTensorDesc, name: str) -> None:
        if requested is None:
            return
        requested_dtype = normalize_jax_dtype(requested, requested, name)
        actual_dtype = jnp.dtype(desc.dtype)
        if requested_dtype != actual_dtype:
            raise ValueError(f"{name}={requested_dtype} does not match the explicit sample dtype {actual_dtype}")

    def check_support(self) -> bool:
        self._op.check_support()
        if self.a_desc.cudnn_dtype not in _JAX_INPUT_DTYPES:
            raise NotImplementedError("The JAX GEMM + dsReLU API supports native FP4 and FP8 inputs")
        if self.c_desc.cudnn_dtype not in _WIDE_OUTPUT_DTYPES or self.d_desc.cudnn_dtype not in _WIDE_OUTPUT_DTYPES:
            raise NotImplementedError("The JAX GEMM + dsReLU API received an unsupported output dtype")
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        c_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
        norm_const_tensor: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        for value, desc, mode in (
            (a_tensor, self.a_desc, self.a_mode),
            (b_tensor, self.b_desc, self.b_mode),
            (c_tensor, self.c_desc, self.output_mode),
            (sfa_tensor, self.sfa_desc, self.scale_mode),
            (sfb_tensor, self.sfb_desc, self.scale_mode),
            (prob_tensor, self.prob_desc, self.probability_mode),
        ):
            self._check_tensor_signature(value, desc, mode=mode)
        if norm_const_tensor is not None:
            raise ValueError("norm_const_tensor is only used with FP8 D output, which is not implemented")

        max_active_clusters = self._get_max_active_clusters(
            self._op.cluster_shape_mn[0] * self._op.cluster_shape_mn[1],
            overlap_margin=self.num_cluster_overlap_margin,
        )

        def launch(stream, a, b, c, sfa, sfb, prob, d, dprob, *optional_outputs):
            expected_optional_outputs = int(self.amax_desc is not None)
            if len(optional_outputs) != expected_optional_outputs:
                raise RuntimeError(f"GemmDsreluSm100 received {len(optional_outputs)} optional output buffers; expected {expected_optional_outputs}")
            amax = optional_outputs[0] if self.amax_desc is not None else None

            import cutlass
            import cutlass.cute as cute

            from .dense_blockscaled_gemm_persistent_dsrelu_quant import Sm100BlockScaledPersistentDenseGemmKernel

            kernel = Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=self._op.sf_vec_size,
                mma_tiler_mn=self._op.mma_tiler_mn,
                cluster_shape_mn=self._op.cluster_shape_mn,
                vector_f32=self.vector_f32,
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
                amax,
                None,
                None,
                cutlass.Float32(self._op.alpha),
                max_active_clusters,
                stream,
                epilogue_op=squared_relu_backward,
            )

        optional_output_descs = () if self.amax_desc is None else (self.amax_desc,)
        output_descs = (self.d_desc, self.dprob_desc) + optional_output_descs
        output_specs = (
            self._to_tensor_spec(self.d_desc, mode=self.output_mode),
            self._to_tensor_spec(self.dprob_desc, mode=self.probability_mode),
        ) + tuple(self._to_tensor_spec(desc) for desc in optional_output_descs)
        results = self._call_kernel(
            (a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor),
            launch=launch,
            output_descs=output_descs,
            input_spec=(
                self._to_tensor_spec(self.a_desc, mode=self.a_mode),
                self._to_tensor_spec(self.b_desc, mode=self.b_mode),
                self._to_tensor_spec(self.c_desc, mode=self.output_mode),
                self._to_tensor_spec(self.sfa_desc, mode=self.scale_mode),
                self._to_tensor_spec(self.sfb_desc, mode=self.scale_mode),
                self._to_tensor_spec(self.prob_desc, mode=self.probability_mode),
            ),
            output_spec=output_specs,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        d_tensor, dprob_tensor, *optional_results = results
        return TupleDict(
            d_tensor=d_tensor,
            dprob_tensor=dprob_tensor,
            amax_tensor=optional_results[0] if self.amax_desc is not None else None,
            sfd_tensor=None,
        )


@partial(
    jax.jit,
    static_argnames=(
        "alpha",
        "d_layout",
        "d_dtype",
        "acc_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "vector_f32",
        "a_layout",
        "b_layout",
    ),
)
def gemm_dsrelu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    d_layout: str = "LMN",
    d_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] | None = None,
    norm_const_tensor: Any | None = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute the block-scaled squared-ReLU backward fusion."""

    return GemmDsreluSm100(
        jax.ShapeDtypeStruct(a_tensor.shape, a_tensor.dtype),
        jax.ShapeDtypeStruct(b_tensor.shape, b_tensor.dtype),
        jax.ShapeDtypeStruct(c_tensor.shape, c_tensor.dtype),
        jax.ShapeDtypeStruct(sfa_tensor.shape, sfa_tensor.dtype),
        jax.ShapeDtypeStruct(sfb_tensor.shape, sfb_tensor.dtype),
        jax.ShapeDtypeStruct(prob_tensor.shape, prob_tensor.dtype),
        sample_norm_const=(None if norm_const_tensor is None else jax.ShapeDtypeStruct(norm_const_tensor.shape, norm_const_tensor.dtype)),
        alpha=alpha,
        d_layout=d_layout,
        d_dtype=d_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor, norm_const_tensor)


__all__ = [
    "GemmDsreluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gemm_dsrelu_wrapper_sm100",
]
