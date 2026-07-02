# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + amax on SM100."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, NamedTuple

import jax.numpy as jnp

from .._jax.api_base import ApiBaseJax, JaxTensorDesc
from .._jax.cutedsl import BufferSpec, call_cutedsl
from .._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
)
from ..gemm_validation import require_gemm_shapes, resolve_max_active_clusters
from .validation import validate_gemm_amax


class GemmAmaxResult(NamedTuple):
    """Functional outputs from block-scaled GEMM + amax."""

    c_tensor: Any
    amax_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    sf_vec_size: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    cluster_overlap_margin: int,
):
    def launch(stream, a, b, sfa, sfb, c, amax):
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

    return launch


class GemmAmaxSm100(ApiBaseJax):
    """JAX GEMM + amax callable specialized from sample metadata."""

    def __init__(
        self,
        sample_a: Any,
        sample_b: Any,
        sample_sfa: Any,
        sample_sfb: Any,
        c_major: str = "n",
        c_dtype: Any = None,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        cluster_shape_mn: tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
        *,
        a_major: str = "k",
        b_major: str = "k",
    ) -> None:
        super().__init__()
        self.a_spec = gemm_a_tensor_spec(a_major)
        self.b_spec = gemm_b_tensor_spec(b_major)
        self.c_spec = gemm_c_tensor_spec(c_major)
        self.scale_spec = block_scale_tensor_spec()

        self.a_desc = self.make_tensor_desc(sample_a, layout=self.a_spec.layout, name="sample_a")
        self.b_desc = self.make_tensor_desc(sample_b, layout=self.b_spec.layout, name="sample_b")
        self.sfa_desc = self.make_tensor_desc(sample_sfa, layout=self.scale_spec.layout, name="sample_sfa")
        self.sfb_desc = self.make_tensor_desc(sample_sfb, layout=self.scale_spec.layout, name="sample_sfb")

        self._c_dtype = self.as_optional_dtype(c_dtype)
        self._acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._plan = None

    def _check_support(self) -> bool:
        supported_inputs = (jnp.float8_e4m3fn, jnp.float8_e5m2)
        self.require_dtype("sample_a.dtype", self.a_desc, supported_inputs)
        self.require_dtype("sample_b.dtype", self.b_desc, supported_inputs)
        if self.sf_vec_size != 32:
            raise NotImplementedError(f"The JAX MXFP8 path requires sf_vec_size=32, got {self.sf_vec_size}")
        self.require_dtype("sample_sfa.dtype", self.sfa_desc, (jnp.float8_e8m0fnu,))
        self.require_dtype("sample_sfb.dtype", self.sfb_desc, (jnp.float8_e8m0fnu,))
        self.c_dtype = self.require_dtype(
            "c_dtype",
            self._c_dtype,
            (jnp.float32, jnp.float16, jnp.bfloat16),
            default=jnp.float32,
        )
        self.acc_dtype = self.require_dtype("acc_dtype", self._acc_dtype, (jnp.float32,), default=jnp.float32)

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
        c_desc = JaxTensorDesc(
            dtype=self.c_dtype,
            shape=(m, n, batch),
            stride_order=tuple(sorted(range(3), key=self.c_spec.layout.__getitem__)),
            jax_layout=self.c_spec.layout,
            name="c_tensor",
        )
        amax_desc = JaxTensorDesc(
            dtype=jnp.float32,
            shape=(1, 1, 1),
            stride_order=(2, 1, 0),
            jax_layout=(2, 1, 0),
            name="amax_tensor",
        )
        self._plan = validate_gemm_amax(
            self.a_desc,
            self.b_desc,
            self.sfa_desc,
            self.sfb_desc,
            c_desc,
            amax_desc,
            acc_dtype=self.acc_dtype,
            sf_vec_size=self.sf_vec_size,
            supported_sf_vec_sizes=kernel.SF_VEC_SIZES,
            mma_tiler_mn=self.mma_tiler_mn,
        )
        return True

    def __call__(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
    ) -> GemmAmaxResult:
        return super().__call__(a_tensor, b_tensor, sfa_tensor, sfb_tensor)

    def _call_impl(
        self,
        a_tensor: Any,
        b_tensor: Any,
        sfa_tensor: Any,
        sfb_tensor: Any,
    ) -> GemmAmaxResult:
        if self._plan is None:
            raise RuntimeError("check_support() did not produce a launch plan")
        self.check_tensor_signature(a_tensor, self.a_desc, name="A")
        self.check_tensor_signature(b_tensor, self.b_desc, name="B")
        self.check_tensor_signature(sfa_tensor, self.sfa_desc, name="SFA")
        self.check_tensor_signature(sfb_tensor, self.sfb_desc, name="SFB")

        launcher = _make_launcher(
            sf_vec_size=self.sf_vec_size,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            cluster_overlap_margin=self.num_cluster_overlap_margin,
        )
        c_tensor, amax_tensor = call_cutedsl(
            launcher,
            (a_tensor, b_tensor, sfa_tensor, sfb_tensor),
            outputs=(
                BufferSpec(
                    "c_tensor",
                    self._plan.c_shape,
                    self.c_dtype,
                    tensor_spec=self.c_spec,
                ),
                BufferSpec(
                    "amax_tensor",
                    self._plan.amax_shape,
                    jnp.float32,
                    fill_value=-float("inf"),
                ),
            ),
            input_specs=(self.a_spec, self.b_spec, self.scale_spec, self.scale_spec),
            use_static_tensors=True,
        )
        return GemmAmaxResult(c_tensor=c_tensor, amax_tensor=amax_tensor)


def gemm_amax_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    c_major: str = "n",
    c_dtype: Any = None,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    cluster_shape_mn: tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    *,
    a_major: str = "k",
    b_major: str = "k",
) -> GemmAmaxResult:
    """Compute FP8 block-scaled GEMM and a global max-absolute reduction."""

    return GemmAmaxSm100(
        a_tensor,
        b_tensor,
        sfa_tensor,
        sfb_tensor,
        c_major=c_major,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sf_vec_size=sf_vec_size,
        a_major=a_major,
        b_major=b_major,
    )(a_tensor, b_tensor, sfa_tensor, sfb_tensor)


__all__ = ["GemmAmaxResult", "GemmAmaxSm100", "gemm_amax_wrapper_sm100"]
