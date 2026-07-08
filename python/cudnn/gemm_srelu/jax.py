# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for block-scaled dense GEMM + squared ReLU."""

from __future__ import annotations

from functools import partial
import os
from typing import Any

import jax
import jax.numpy as jnp

from .. import data_type
from .._jax.compiler import compile_options_for_target
from ..gemm.helpers import require_gemm_inputs
from .._jax import JaxApiBase, JaxTensorDesc, TupleDict
from .._jax.datatypes import jax_to_cudnn_dtype, normalize_jax_dtype
from .._jax.gemm import BLOCK_SCALE_MODE, PROBABILITY_MODE, gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import to_public_axes
from .op import GemmSreluSm100Op

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
_JAX_INPUT_DTYPES = frozenset({data_type.FP4_E2M1, data_type.FP8_E4M3, data_type.FP8_E5M2})
_WIDE_OUTPUT_DTYPES = frozenset({data_type.HALF, data_type.BFLOAT16, data_type.FLOAT})


class GemmSreluSm100(JaxApiBase):
    """JAX callable specialized from a block-scaled GEMM + sReLU signature."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        sample_prob: Any,
        *,
        sample_c: Any | None = None,
        sample_d: Any | None = None,
        alpha: float = 1.0,
        c_layout: str = "LMN",
        c_dtype: Any | None = None,
        d_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (256, 256),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        a_mode = gemm_a_mode(a_layout)
        b_mode = gemm_b_mode(b_layout)
        output_mode = gemm_output_mode(c_layout)

        self.compute_capability = self._resolve_compute_capability(
            None,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "GemmSreluSm100",
        )
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=b_mode)
        self.sfa_desc = self._to_tensor_desc(sample_sfa, "sample_sfa", mode=BLOCK_SCALE_MODE)
        self.sfb_desc = self._to_tensor_desc(sample_sfb, "sample_sfb", mode=BLOCK_SCALE_MODE)
        self.prob_desc = self._to_tensor_desc(sample_prob, "sample_prob", mode=PROBABILITY_MODE)

        self.acc_dtype = normalize_jax_dtype(acc_dtype, jnp.float32, "acc_dtype")
        if (sample_c is None) != (sample_d is None):
            raise ValueError("sample_c and sample_d must be provided together")
        if sample_c is None:
            resolved_c_dtype = normalize_jax_dtype(c_dtype, jnp.bfloat16, "c_dtype")
            resolved_d_dtype = normalize_jax_dtype(d_dtype, jnp.bfloat16, "d_dtype")
            self.c_desc, self.d_desc = self._default_output_descs(
                resolved_c_dtype,
                resolved_d_dtype,
                output_mode,
            )
        else:
            if c_dtype is not None or d_dtype is not None:
                raise ValueError("c_dtype and d_dtype cannot be specified with sample_c and sample_d")
            self.c_desc = self._to_tensor_desc(sample_c, "sample_c", mode=output_mode)
            self.d_desc = self._to_tensor_desc(sample_d, "sample_d", mode=output_mode)

        self.amax_desc = None
        if self.a_desc.cudnn_dtype == data_type.FP4_E2M1 and self.d_desc.cudnn_dtype in _WIDE_OUTPUT_DTYPES:
            self.amax_desc = self.d_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(1,),
                name="amax_tensor",
                init_value=float("-inf"),
            )
        acc_cudnn_dtype = jax_to_cudnn_dtype(self.acc_dtype)
        if acc_cudnn_dtype == data_type.NOT_SET:
            raise ValueError(f"Unsupported JAX accumulator dtype {self.acc_dtype}")
        if not isinstance(vector_f32, bool):
            raise TypeError(f"vector_f32 must be a bool, got {type(vector_f32).__name__}")

        self._op = GemmSreluSm100Op(
            a=self.a_desc,
            b=self.b_desc,
            c=self.c_desc,
            d=self.d_desc,
            sfa=self.sfa_desc,
            sfb=self.sfb_desc,
            prob=self.prob_desc,
            sfd=None,
            amax=self.amax_desc,
            norm_const=None,
            alpha=alpha,
            acc_dtype=acc_cudnn_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        self.vector_f32 = vector_f32
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _default_output_descs(
        self,
        c_dtype: Any,
        d_dtype: Any,
        output_mode: tuple[int, ...],
    ) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        m, n, _, batch = require_gemm_inputs(self.a_desc, self.b_desc)
        public_shape = to_public_axes((m, n, batch), output_mode)
        return (
            JaxTensorDesc.from_shape(
                public_shape,
                c_dtype,
                name="sample_c",
                mode=output_mode,
            ),
            JaxTensorDesc.from_shape(
                public_shape,
                d_dtype,
                name="sample_d",
                mode=output_mode,
            ),
        )

    def check_support(self) -> bool:
        self._op.check_support()
        if self.a_desc.cudnn_dtype not in _JAX_INPUT_DTYPES:
            raise NotImplementedError("The JAX GEMM + sReLU API supports native FP4 and FP8 inputs")
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
        prob_tensor: Any,
    ) -> TupleDict:
        self.check_support()

        max_active_clusters = self._get_max_active_clusters(
            self._op.cluster_shape_mn[0] * self._op.cluster_shape_mn[1],
            overlap_margin=self.num_cluster_overlap_margin,
        )

        def launch(stream, a, b, sfa, sfb, prob, c, d, *optional_outputs):
            expected_optional_outputs = int(self.amax_desc is not None)
            if len(optional_outputs) != expected_optional_outputs:
                raise RuntimeError(f"GemmSreluSm100 received {len(optional_outputs)} optional output buffers; expected {expected_optional_outputs}")
            amax = optional_outputs[0] if self.amax_desc is not None else None

            import cutlass
            import cutlass.cute as cute

            from .dense_blockscaled_gemm_persistent_srelu_quant import Sm100BlockScaledPersistentDenseGemmKernel

            kernel = Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=self._op.sf_vec_size,
                mma_tiler_mn=self._op.mma_tiler_mn,
                cluster_shape_mn=self._op.cluster_shape_mn,
                vector_f32=self.vector_f32,
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
                amax,
                None,
                None,
                cutlass.Float32(self._op.alpha),
                max_active_clusters,
                stream,
                epilogue_op=squared_relu,
            )

        optional_output_descs = () if self.amax_desc is None else (self.amax_desc,)
        output_descs = (self.c_desc, self.d_desc) + optional_output_descs
        results = self._call_kernel(
            (a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor),
            launch=launch,
            input_descs=(self.a_desc, self.b_desc, self.sfa_desc, self.sfb_desc, self.prob_desc),
            output_descs=output_descs,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        c_tensor, d_tensor, *optional_results = results
        return TupleDict(
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            amax_tensor=optional_results[0] if self.amax_desc is not None else None,
            sfd_tensor=None,
        )


@partial(
    jax.jit,
    static_argnames=(
        "alpha",
        "c_layout",
        "c_dtype",
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
def gemm_srelu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    prob_tensor: Any,
    alpha: float = 1.0,
    c_layout: str = "LMN",
    c_dtype: Any | None = None,
    d_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: tuple[int, int] | None = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute block-scaled GEMM and ``D = prob * relu(C) ** 2``."""

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
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor, prob_tensor)


__all__ = [
    "GemmSreluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gemm_srelu_wrapper_sm100",
]
