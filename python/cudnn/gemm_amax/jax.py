# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + amax on SM100."""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp

from .._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    JaxTensorDesc,
    TupleDict,
    call_cutedsl,
)
from .._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
)
from ..gemm_validation import require_gemm_shapes, resolve_max_active_clusters
from .validation import validate_gemm_amax


def _launch(
    stream,
    a,
    b,
    sfa,
    sfb,
    c,
    amax,
    *,
    sf_vec_size: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    cluster_overlap_margin: int,
):
    # These operations happen during CUDA lowering, not abstract evaluation.
    import cutlass

    from .dense_blockscaled_gemm_persistent_amax import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    kernel = Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=sf_vec_size,
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
        sfa,
        sfb,
        c,
        amax,
        max_active_clusters,
        stream,
    )


class GemmAmaxSm100(ApiBaseJax):
    """JAX GEMM + amax callable specialized from sample metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        c_layout: str = "LMN",
        c_dtype: Any = None,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
        *,
        a_layout: str = "LMK",
        b_layout: str = "LNK",
    ) -> None:
        super().__init__()
        self.a_desc = self.make_tensor_desc(sample_a, tensor_spec=gemm_a_tensor_spec(a_layout), name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, tensor_spec=gemm_b_tensor_spec(b_layout), name="sample_b")
        self.sfa_desc = self.make_tensor_desc(sample_sfa, tensor_spec=block_scale_tensor_spec(), name="sample_sfa")
        self.sfb_desc = self.make_tensor_desc(sample_sfb, tensor_spec=block_scale_tensor_spec(), name="sample_sfb")
        self._c_layout = c_layout

        self._c_dtype = self.as_optional_dtype(c_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._plan = None

    def _check_support(self) -> None:
        supported_inputs = (jnp.float8_e4m3fn, jnp.float8_e5m2)
        self.require_dtype(self.a_desc, supported_inputs)
        self.require_dtype(self.b_desc, supported_inputs)
        if self.sf_vec_size != 32:
            raise NotImplementedError(f"The JAX MXFP8 path requires sf_vec_size=32, got {self.sf_vec_size}")
        self.require_dtype(self.sfa_desc, (jnp.float8_e8m0fnu,))
        self.require_dtype(self.sfb_desc, (jnp.float8_e8m0fnu,))
        self.c_dtype = self.require_dtype(
            self._c_dtype,
            (jnp.float32, jnp.float16, jnp.bfloat16),
            name="c_dtype",
            default=jnp.float32,
        )
        self.acc_dtype = self.require_dtype(
            self._acc_dtype,
            (jnp.float32,),
            name="acc_dtype",
            default=jnp.float32,
        )

        from .dense_blockscaled_gemm_persistent_amax import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )

        kernel = Sm100BlockScaledPersistentDenseGemmKernel
        self.mma_tiler_mn = kernel.require_mma_tiler(self.mma_tiler_mn)
        self.cluster_shape_mn = kernel.require_cluster_shape(
            self.cluster_shape_mn,
            mma_tiler_mn=self.mma_tiler_mn,
        )

        m, n, _, batch = require_gemm_shapes(self.a_desc.shape, self.b_desc.shape)
        self.c_desc = JaxTensorDesc(
            dtype=self.c_dtype,
            shape=(m, n, batch),
            tensor_spec=gemm_c_tensor_spec(self._c_layout),
            name="c_tensor",
        )
        self.amax_desc = JaxTensorDesc(
            dtype=jnp.float32,
            shape=(1, 1, 1),
            name="amax_tensor",
        )
        self._plan = validate_gemm_amax(
            self.a_desc,
            self.b_desc,
            self.sfa_desc,
            self.sfb_desc,
            self.c_desc,
            self.amax_desc,
            acc_dtype=self.acc_dtype,
            sf_vec_size=self.sf_vec_size,
            supported_sf_vec_sizes=kernel.SF_VEC_SIZES,
            mma_tiler_mn=self.mma_tiler_mn,
        )

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
    ) -> TupleDict:
        return super().__call__(a_tensor, b_tensor, sfa_tensor, sfb_tensor)

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
    ) -> TupleDict:
        if self._plan is None:
            raise RuntimeError("check_support() did not produce a launch plan")
        self.check_tensor_signature(a_tensor, self.a_desc, name="A")
        self.check_tensor_signature(b_tensor, self.b_desc, name="B")
        self.check_tensor_signature(sfa_tensor, self.sfa_desc, name="SFA")
        self.check_tensor_signature(sfb_tensor, self.sfb_desc, name="SFB")

        c_tensor, amax_tensor = call_cutedsl(
            _launch,
            (a_tensor, b_tensor, sfa_tensor, sfb_tensor),
            outputs=(
                BufferSpec(
                    "c_tensor",
                    self.c_desc.array_shape,
                    self.c_desc.dtype,
                    tensor_spec=self.c_desc.tensor_spec,
                ),
                BufferSpec(
                    "amax_tensor",
                    self.amax_desc.shape,
                    self.amax_desc.dtype,
                    fill_value=-float("inf"),
                ),
            ),
            input_specs=(
                self.a_desc.tensor_spec,
                self.b_desc.tensor_spec,
                self.sfa_desc.tensor_spec,
                self.sfb_desc.tensor_spec,
            ),
            static_args={
                "sf_vec_size": self.sf_vec_size,
                "mma_tiler_mn": self.mma_tiler_mn,
                "cluster_shape_mn": self.cluster_shape_mn,
                "cluster_overlap_margin": self.num_cluster_overlap_margin,
            },
        )
        return TupleDict(c_tensor=c_tensor, amax_tensor=amax_tensor)


def gemm_amax_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_layout: str = "LMN",
    c_dtype: Any = None,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    *,
    a_layout: str = "LMK",
    b_layout: str = "LNK",
) -> TupleDict:
    """Compute FP8 block-scaled GEMM and a global max-absolute reduction."""

    return GemmAmaxSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_layout=c_layout,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        a_layout=a_layout,
        b_layout=b_layout,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)


__all__ = ["GemmAmaxSm100", "gemm_amax_wrapper_sm100"]
