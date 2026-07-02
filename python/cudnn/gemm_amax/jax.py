# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + amax on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple

import jax.numpy as jnp

from .._jax.cutedsl import BufferSpec, call_cutedsl
from .._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    require_16_byte_extent,
    require_fp8_block_scales,
    require_gemm_inputs,
)
from .._jax.validation import require_dtype
from ..gemm_validation import resolve_max_active_clusters


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
    """Compute FP8 block-scaled GEMM and a global max-absolute reduction.

    ``a_tensor`` and ``b_tensor`` use logical shapes ``(M, K, L)`` and
    ``(N, K, L)``. ``sfa_tensor`` and ``sfb_tensor`` use the six-dimensional
    block-scale atom layout documented by the Torch API. The returned tensors
    have shapes ``(M, N, L)`` and ``(1, 1, 1)``.

    The JAX binding accepts FP8 inputs with E8M0 scale factors and
    ``sf_vec_size=32``. Packed FP4 storage is not exposed because JAX's scalar
    FP4 dtype does not have the Torch wrapper's packed-FP4x2 ABI. Configuration
    values must be static under ``jax.jit``. JAX owns both output buffers and
    initializes ``amax_tensor`` before the kernel's atomic reduction.
    """

    m, n, k, batch, a_dtype = require_gemm_inputs(a_tensor, b_tensor)
    require_fp8_block_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        batch=batch,
        sf_vec_size=sf_vec_size,
    )

    supported_inputs = {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
    if a_dtype not in supported_inputs:
        raise NotImplementedError("The JAX GEMM + amax API currently supports float8_e4m3fn and " f"float8_e5m2 inputs, got {a_dtype}")
    c_dtype = require_dtype(
        "c_dtype",
        c_dtype,
        (jnp.float32, jnp.float16, jnp.bfloat16),
        default=jnp.float32,
    )
    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)

    from .dense_blockscaled_gemm_persistent_amax import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    kernel = Sm100BlockScaledPersistentDenseGemmKernel
    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    cluster_shape_mn = kernel.require_cluster_shape(
        cluster_shape_mn,
        mma_tiler_mn=mma_tiler_mn,
    )

    a_spec = gemm_a_tensor_spec(a_major)
    b_spec = gemm_b_tensor_spec(b_major)
    c_spec = gemm_c_tensor_spec(c_major)
    scale_spec = block_scale_tensor_spec()
    require_16_byte_extent("a_tensor", m if a_major == "m" else k, a_dtype)
    require_16_byte_extent("b_tensor", n if b_major == "n" else k, a_dtype)
    require_16_byte_extent("c_tensor", m if c_major == "m" else n, c_dtype)

    cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    launcher = _make_launcher(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        cluster_overlap_margin=cluster_overlap_margin,
    )
    c_tensor, amax_tensor = call_cutedsl(
        launcher,
        (a_tensor, b_tensor, sfa_tensor, sfb_tensor),
        outputs=(
            BufferSpec("c_tensor", (m, n, batch), c_dtype, tensor_spec=c_spec),
            BufferSpec(
                "amax_tensor",
                (1, 1, 1),
                jnp.float32,
                fill_value=-float("inf"),
            ),
        ),
        input_specs=(a_spec, b_spec, scale_spec, scale_spec),
        use_static_tensors=True,
    )
    return GemmAmaxResult(c_tensor=c_tensor, amax_tensor=amax_tensor)


__all__ = ["GemmAmaxResult", "gemm_amax_wrapper_sm100"]
