# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for block-scaled dense GEMM + amax on SM100."""

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
from .._jax.gemm import BLOCK_SCALE_MODE, gemm_a_mode, gemm_b_mode, gemm_output_mode
from .._jax.layout import to_public_axes
from .op import GemmAmaxSm100Op

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)


class GemmAmaxSm100(JaxApiBase):
    """JAX callable specialized from block-scaled GEMM input metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        *,
        sample_c: Any | None = None,
        sample_amax: Any | None = None,
        c_layout: str = "LMN",
        c_dtype: Any | None = None,
        acc_dtype: Any | None = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.c_layout = c_layout
        self.a_mode = gemm_a_mode(a_layout)
        self.b_mode = gemm_b_mode(b_layout)
        self.output_mode = gemm_output_mode(c_layout)
        self.scale_mode = BLOCK_SCALE_MODE

        self.compute_capability = self._resolve_compute_capability(
            None,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "GemmAmaxSm100",
        )
        self.a_desc = self._to_tensor_desc(sample_a, "sample_a", mode=self.a_mode)
        self.b_desc = self._to_tensor_desc(sample_b, "sample_b", mode=self.b_mode)
        self.sfa_desc = self._to_tensor_desc(
            sample_sfa,
            "sample_sfa",
            mode=self.scale_mode,
        )
        self.sfb_desc = self._to_tensor_desc(
            sample_sfb,
            "sample_sfb",
            mode=self.scale_mode,
        )
        self.acc_dtype = normalize_jax_dtype(acc_dtype, jnp.float32, "acc_dtype")

        if (sample_c is None) != (sample_amax is None):
            raise ValueError("sample_c and sample_amax must be provided together")
        if sample_c is None:
            resolved_c_dtype = normalize_jax_dtype(c_dtype, jnp.float32, "c_dtype")
            self.c_desc, self.amax_desc = self._default_output_descs(resolved_c_dtype)
        else:
            self.c_desc = self._to_tensor_desc(
                sample_c,
                "sample_c",
                mode=self.output_mode,
            )
            self.amax_desc = self._to_tensor_desc(
                sample_amax,
                "sample_amax",
                init_value=float("-inf"),
            )
            self._check_requested_c_dtype(c_dtype)

        acc_cudnn_dtype = jax_to_cudnn_dtype(self.acc_dtype)
        if acc_cudnn_dtype == data_type.NOT_SET:
            raise ValueError(f"Unsupported JAX accumulator dtype {self.acc_dtype}")
        self._op = GemmAmaxSm100Op(
            a=self.a_desc,
            b=self.b_desc,
            sfa=self.sfa_desc,
            sfb=self.sfb_desc,
            c=self.c_desc,
            amax=self.amax_desc,
            acc_dtype=acc_cudnn_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    def _default_output_descs(
        self,
        c_dtype: Any,
    ) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        m, n, _, batch = require_gemm_inputs(self.a_desc, self.b_desc)
        return (
            self._to_tensor_desc(
                jax.ShapeDtypeStruct(
                    to_public_axes((m, n, batch), self.output_mode),
                    c_dtype,
                ),
                "sample_c",
                mode=self.output_mode,
            ),
            self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, 1, 1), jnp.float32),
                "sample_amax",
                init_value=float("-inf"),
            ),
        )

    def _check_requested_c_dtype(self, requested: Any | None) -> None:
        if requested is None:
            return
        requested_dtype = normalize_jax_dtype(requested, requested, "c_dtype")
        actual_dtype = jnp.dtype(self.c_desc.dtype)
        if requested_dtype != actual_dtype:
            raise ValueError(f"c_dtype={requested_dtype} does not match the explicit sample dtype " f"{actual_dtype}")

    def check_support(self) -> bool:
        """Validate the JAX dtype surface and common operation contract."""

        if self.a_desc.cudnn_dtype not in {
            data_type.FP4_E2M1,
            data_type.FP8_E4M3,
            data_type.FP8_E5M2,
        }:
            raise ValueError(f"The JAX GEMM + amax API requires FP4 or FP8 A and B, got " f"{self.a_desc.dtype}")
        if self.b_desc.cudnn_dtype not in {
            data_type.FP4_E2M1,
            data_type.FP8_E4M3,
            data_type.FP8_E5M2,
        }:
            raise ValueError(f"The JAX GEMM + amax API requires FP4 or FP8 A and B, got " f"{self.b_desc.dtype}")
        if self.sfa_desc.cudnn_dtype not in {
            data_type.FP8_E8M0,
            data_type.FP8_E4M3,
        }:
            raise ValueError(f"The JAX GEMM + amax API requires E8M0 or E4M3 scale factors, " f"got {self.sfa_desc.dtype}")
        if self.sfb_desc.cudnn_dtype not in {
            data_type.FP8_E8M0,
            data_type.FP8_E4M3,
        }:
            raise ValueError(f"The JAX GEMM + amax API requires E8M0 or E4M3 scale factors, " f"got {self.sfb_desc.dtype}")
        if self.c_desc.cudnn_dtype not in {
            data_type.FLOAT,
            data_type.HALF,
            data_type.BFLOAT16,
            data_type.FP8_E4M3,
            data_type.FP8_E5M2,
            data_type.FP4_E2M1,
        }:
            raise ValueError(f"The JAX GEMM + amax API received unsupported C dtype " f"{self.c_desc.dtype}")
        return self._op.check_support()

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
    ) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(a_tensor, self.a_desc, mode=self.a_mode)
        self._check_tensor_signature(b_tensor, self.b_desc, mode=self.b_mode)
        self._check_tensor_signature(sfa_tensor, self.sfa_desc, mode=self.scale_mode)
        self._check_tensor_signature(sfb_tensor, self.sfb_desc, mode=self.scale_mode)
        max_active_clusters = self._get_max_active_clusters(
            self._op.cluster_shape_mn[0] * self._op.cluster_shape_mn[1],
            overlap_margin=self.num_cluster_overlap_margin,
        )

        def launch(stream, a, b, sfa, sfb, c, amax):
            from .dense_blockscaled_gemm_persistent_amax import (
                Sm100BlockScaledPersistentDenseGemmKernel,
            )

            kernel = Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=self._op.sf_vec_size,
                mma_tiler_mn=self._op.mma_tiler_mn,
                cluster_shape_mn=self._op.cluster_shape_mn,
            )
            kernel(
                a,
                b,
                sfa,
                sfb,
                c,
                amax,
                max_active_clusters,
                stream,
            )

        c_tensor, amax_tensor = self._call_kernel(
            (a_tensor, b_tensor, sfa_tensor, sfb_tensor),
            launch=launch,
            output_descs=(self.c_desc, self.amax_desc),
            input_spec=(
                self._to_tensor_spec(self.a_desc, mode=self.a_mode),
                self._to_tensor_spec(self.b_desc, mode=self.b_mode),
                self._to_tensor_spec(self.sfa_desc, mode=self.scale_mode),
                self._to_tensor_spec(self.sfb_desc, mode=self.scale_mode),
            ),
            output_spec=(
                self._to_tensor_spec(self.c_desc, mode=self.output_mode),
                self._to_tensor_spec(self.amax_desc),
            ),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(c_tensor=c_tensor, amax_tensor=amax_tensor)


@partial(
    jax.jit,
    static_argnames=(
        "c_layout",
        "c_dtype",
        "acc_dtype",
        "mma_tiler_mn",
        "cluster_shape_mn",
        "sf_vec_size",
        "a_layout",
        "b_layout",
    ),
)
def gemm_amax_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_layout: str = "LMN",
    c_dtype: Any | None = None,
    acc_dtype: Any | None = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute a block-scaled GEMM and global max-absolute reduction."""

    return GemmAmaxSm100(
        jax.ShapeDtypeStruct(a_tensor.shape, a_tensor.dtype),
        jax.ShapeDtypeStruct(b_tensor.shape, b_tensor.dtype),
        jax.ShapeDtypeStruct(sfa_tensor.shape, sfa_tensor.dtype),
        jax.ShapeDtypeStruct(sfb_tensor.shape, sfb_tensor.dtype),
        c_layout=c_layout,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)


__all__ = [
    "GemmAmaxSm100",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "gemm_amax_wrapper_sm100",
]
