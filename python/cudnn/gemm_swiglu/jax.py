# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for standard and block-scaled dense GEMM + SwiGLU."""

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
from .._jax.gemm import BLOCK_SCALE_MODE, gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import to_public_axes
from .op import BlockScaledGemmSwigluSm100Op, GemmSwigluSm100Op

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)


class GemmSwigluSm100(JaxApiBase):
    """JAX callable specialized from dense GEMM input metadata.

    ``a_layout``, ``b_layout``, and ``c_layout`` describe the public JAX axis
    order. Descriptors stored by the operation remain in the kernel's
    canonical ``MKL``, ``NKL``, and ``MNL`` orders.

    Providing SFA and SFB samples selects the block-scaled kernel. Public scale
    arrays are compact row-major ``[L, row_tiles, k_tiles, 32, 4, 4]`` arrays;
    the adapter maps them to the kernel's packed scale-factor layout.
    """

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        *,
        sample_ab12: Any | None = None,
        sample_c: Any | None = None,
        sample_sfa: Any | None = None,
        sample_sfb: Any | None = None,
        sample_amax: Any | None = None,
        alpha: float = 1.0,
        c_layout: str = "LMN",
        ab12_dtype: Any | None = None,
        c_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] | None = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        ab12_stages: int = 4,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        a_mode = gemm_a_mode(a_layout)
        b_mode = gemm_b_mode(b_layout)
        output_mode = gemm_output_mode(c_layout)

        self.compute_capability = self._resolve_compute_capability(
            None,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "GemmSwigluSm100",
        )
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=b_mode)
        self.acc_dtype = normalize_jax_dtype(acc_dtype, jnp.float32, "acc_dtype")

        self.is_block_scaled = any(
            sample is not None
            for sample in (
                sample_sfa,
                sample_sfb,
                sample_amax,
            )
        )
        if self.is_block_scaled and (sample_sfa is None or sample_sfb is None):
            raise ValueError("sample_sfa and sample_sfb are required for block-scaled GEMM + SwiGLU")
        if not self.is_block_scaled:
            if sf_vec_size != 16:
                raise ValueError("sf_vec_size only applies to block-scaled GEMM + SwiGLU")
            if vector_f32:
                raise ValueError("vector_f32 only applies to block-scaled GEMM + SwiGLU")
            if ab12_stages != 4:
                raise ValueError("ab12_stages only applies to block-scaled GEMM + SwiGLU")
        self.sfa_desc = None if sample_sfa is None else self._to_tensor_desc(sample_sfa, "sample_sfa", mode=BLOCK_SCALE_MODE)
        self.sfb_desc = None if sample_sfb is None else self._to_tensor_desc(sample_sfb, "sample_sfb", mode=BLOCK_SCALE_MODE)

        if (sample_ab12 is None) != (sample_c is None):
            raise ValueError("sample_ab12 and sample_c must be provided together")
        if sample_ab12 is None:
            resolved_ab12_dtype = normalize_jax_dtype(ab12_dtype, jnp.float32, "ab12_dtype")
            resolved_c_dtype = normalize_jax_dtype(c_dtype, jnp.float16, "c_dtype")
            self.ab12_desc, self.c_desc = self._default_output_descs(
                resolved_ab12_dtype,
                resolved_c_dtype,
                output_mode,
            )
        else:
            if ab12_dtype is not None or c_dtype is not None:
                raise ValueError("ab12_dtype and c_dtype cannot be specified with sample_ab12 and sample_c")
            self.ab12_desc = self._to_tensor_desc(
                sample_ab12,
                "sample_ab12",
                mode=output_mode,
            )
            self.c_desc = self._to_tensor_desc(
                sample_c,
                "sample_c",
                mode=output_mode,
            )

        self.amax_desc = None
        if self.is_block_scaled:
            needs_amax = self.a_desc.cudnn_dtype == data_type.FP4_E2M1 and self.c_desc.cudnn_dtype == data_type.BFLOAT16
            if sample_amax is not None:
                self.amax_desc = self._to_tensor_desc(sample_amax, "sample_amax", init_value=float("-inf"))
            elif needs_amax:
                self.amax_desc = self._default_amax_desc()

        acc_cudnn_dtype = jax_to_cudnn_dtype(self.acc_dtype)
        if acc_cudnn_dtype == data_type.NOT_SET:
            raise ValueError(f"Unsupported JAX accumulator dtype {self.acc_dtype}")

        if self.is_block_scaled:
            self._op = BlockScaledGemmSwigluSm100Op(
                a=self.a_desc,
                b=self.b_desc,
                sfa=self.sfa_desc,
                sfb=self.sfb_desc,
                ab12=self.ab12_desc,
                c=self.c_desc,
                sfc=None,
                amax=self.amax_desc,
                norm_const=None,
                alpha=alpha,
                acc_dtype=acc_cudnn_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                ab12_stages=ab12_stages,
            )
        else:
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
        output_mode: tuple[int, ...],
    ) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        m, n, _, batch = require_gemm_inputs(self.a_desc, self.b_desc)
        if n % 2:
            raise ValueError(f"SwiGLU requires an even N dimension, got {n}")

        return (
            JaxTensorDesc.from_shape(
                to_public_axes((m, n, batch), output_mode),
                ab12_dtype,
                name="sample_ab12",
                mode=output_mode,
            ),
            JaxTensorDesc.from_shape(
                to_public_axes((m, n // 2, batch), output_mode),
                c_dtype,
                name="sample_c",
                mode=output_mode,
            ),
        )

    def _default_amax_desc(self) -> JaxTensorDesc:
        return JaxTensorDesc.from_shape(
            (1,),
            jnp.float32,
            name="sample_amax",
            init_value=float("-inf"),
        )

    def check_support(self) -> bool:
        return self._op.check_support()

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any | None = None,
        sfb_tensor: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        if self.is_block_scaled:
            if sfa_tensor is None or sfb_tensor is None:
                raise ValueError("sfa_tensor and sfb_tensor are required for this block-scaled callable")
        elif sfa_tensor is not None or sfb_tensor is not None:
            raise ValueError("Scale-factor tensors are not part of this standard GEMM + SwiGLU callable")

        max_active_clusters = self._get_max_active_clusters(
            self._op.cluster_shape_mn[0] * self._op.cluster_shape_mn[1],
            overlap_margin=self.num_cluster_overlap_margin,
        )

        if not self.is_block_scaled:

            def launch(
                stream: Any,
                a: Any,
                b: Any,
                ab12: Any,
                c: Any,
            ) -> None:
                import cutlass
                from cutlass.jax import jax_to_cutlass_dtype

                from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

                kernel = PersistentDenseGemmKernel(
                    acc_dtype=jax_to_cutlass_dtype(self.acc_dtype),
                    use_2cta_instrs=self._op.mma_tiler_mn[0] == 256,
                    mma_tiler_mn=self._op.mma_tiler_mn,
                    cluster_shape_mn=self._op.cluster_shape_mn,
                )
                kernel(
                    a,
                    b,
                    ab12,
                    c,
                    cutlass.Float32(self._op.alpha),
                    max_active_clusters,
                    stream,
                )

            ab12_tensor, c_tensor = self._call_kernel(
                (a_tensor, b_tensor),
                launch=launch,
                input_descs=(self.a_desc, self.b_desc),
                output_descs=(self.ab12_desc, self.c_desc),
                compile_options=compile_options_for_target(self.compute_capability),
            )
            return TupleDict(
                ab12_tensor=ab12_tensor,
                c_tensor=c_tensor,
                sfc_tensor=None,
                amax_tensor=None,
            )

        kernel_inputs = (a_tensor, b_tensor, sfa_tensor, sfb_tensor)
        input_descs = (self.a_desc, self.b_desc, self.sfa_desc, self.sfb_desc)

        output_descs = (self.ab12_desc, self.c_desc)
        if self.amax_desc is not None:
            output_descs += (self.amax_desc,)

        def launch_block_scaled(
            stream: Any,
            a: Any,
            b: Any,
            sfa: Any,
            sfb: Any,
            ab12: Any,
            c: Any,
            *optional_outputs: Any,
        ) -> None:
            expected_optional_outputs = int(self.amax_desc is not None)
            if len(optional_outputs) != expected_optional_outputs:
                raise RuntimeError(f"GemmSwigluSm100 received {len(optional_outputs)} optional output buffers; expected {expected_optional_outputs}")
            amax = optional_outputs[0] if self.amax_desc is not None else None

            import cutlass

            from .dense_blockscaled_gemm_persistent_swiglu_interleaved_quant import Sm100BlockScaledPersistentDenseGemmKernel

            kernel = Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=self._op.sf_vec_size,
                mma_tiler_mn=self._op.mma_tiler_mn,
                cluster_shape_mn=self._op.cluster_shape_mn,
                vector_f32=self._op.vector_f32,
                ab12_stages=self._op.ab12_stages,
            )
            kernel(
                a,
                b,
                sfa,
                sfb,
                c,
                ab12,
                amax,
                None,
                None,
                cutlass.Float32(self._op.alpha),
                max_active_clusters,
                stream,
            )

        results = self._call_kernel(
            kernel_inputs,
            launch=launch_block_scaled,
            input_descs=input_descs,
            output_descs=output_descs,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        ab12_tensor, c_tensor, *optional_results = results
        amax_tensor = optional_results[0] if self.amax_desc is not None else None
        return TupleDict(
            ab12_tensor=ab12_tensor,
            c_tensor=c_tensor,
            sfc_tensor=None,
            amax_tensor=amax_tensor,
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
        "sf_vec_size",
        "vector_f32",
        "ab12_stages",
        "a_layout",
        "b_layout",
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
    sfa_tensor: Any | None = None,
    sfb_tensor: Any | None = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    ab12_stages: int = 4,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute standard or block-scaled GEMM + SwiGLU on a local SM100-family GPU."""

    return GemmSwigluSm100(
        a_tensor,
        b_tensor,
        sample_sfa=sfa_tensor,
        sample_sfb=sfb_tensor,
        alpha=alpha,
        c_layout=c_layout,
        ab12_dtype=ab12_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        vector_f32=vector_f32,
        ab12_stages=ab12_stages,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)


__all__ = [
    "GemmSwigluSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gemm_swiglu_wrapper_sm100",
]
