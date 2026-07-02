# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for dense GEMM + SwiGLU on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from .._jax.cutedsl import BufferSpec, call_cutedsl
from .._jax.gemm import (
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    require_16_byte_extent,
    require_gemm_inputs,
)
from .._jax.validation import require_dtype
from ..gemm_validation import (
    require_cluster_shape,
    require_full_mma_rows,
    require_mma_tiler,
    require_swiglu_n,
    resolve_max_active_clusters,
)


class GemmSwigluResult(NamedTuple):
    """Functional outputs from dense GEMM + SwiGLU."""

    ab12_tensor: Any
    c_tensor: Any
    sfc_tensor: Any | None
    amax_tensor: Any | None


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    alpha: float,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    cluster_overlap_margin: int,
):
    def launch(stream, a, b, ab12, c):
        # These operations happen during CUDA lowering, not abstract evaluation.
        import cutlass
        from cutlass.jax import jax_to_cutlass_dtype

        from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

        kernel = PersistentDenseGemmKernel(
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=mma_tiler_mn[0] == 256,
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

    return launch


def gemm_swiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    alpha: float = 1.0,
    c_major: str = "n",
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
    a_major: str = "k",
    b_major: str = "k",
) -> GemmSwigluResult:
    """Compute a dense batched GEMM and its fused SwiGLU projection.

    ``a_tensor`` and ``b_tensor`` use logical shapes ``(M, K, L)`` and
    ``(N, K, L)``. The result contains ``ab12_tensor`` with shape ``(M, N, L)``
    and ``c_tensor`` with shape ``(M, N // 2, L)``. The physical major modes
    are compile-time choices enforced at the XLA custom-call boundary.

    This JAX binding currently supports the standard, unquantized kernel.
    Block-scaled SwiGLU arguments are reserved for the quantized binding.
    Configuration values must be static when the function is used in
    ``jax.jit``. JAX owns the output buffers and supplies the CUDA stream.
    """

    del sf_vec_size, vector_f32, ab12_stages
    if any(value is not None for value in (sfa_tensor, sfb_tensor, norm_const_tensor)):
        raise NotImplementedError(
            "The JAX GEMM + SwiGLU API currently supports only the " "unquantized path; sfa_tensor, sfb_tensor, and norm_const_tensor " "must be None"
        )

    m, n, k, batch, a_dtype = require_gemm_inputs(a_tensor, b_tensor)
    output_n = require_swiglu_n(n)

    a_dtype = require_dtype(
        "a_tensor.dtype",
        a_dtype,
        (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32, jnp.float16), default=jnp.float32)
    if acc_dtype == jnp.dtype(jnp.float32):
        supported_ab12 = (jnp.float32, jnp.float16, jnp.bfloat16)
    else:
        supported_ab12 = (jnp.float16, jnp.bfloat16)
        if a_dtype not in {
            jnp.dtype(jnp.float16),
            jnp.dtype(jnp.float8_e4m3fn),
            jnp.dtype(jnp.float8_e5m2),
        }:
            raise ValueError(f"float16 accumulation does not support input dtype {a_dtype}")
    ab12_dtype = require_dtype("ab12_dtype", ab12_dtype, supported_ab12, default=jnp.float32)
    c_dtype = require_dtype("c_dtype", c_dtype, (jnp.float16, jnp.bfloat16), default=jnp.float16)

    mma_tiler_mn = require_mma_tiler(
        mma_tiler_mn,
        allowed_m=(128, 256),
        allowed_n=(64, 128, 192, 256),
    )
    if mma_tiler_mn[0] == 256:
        require_full_mma_rows(m, mma_tiler_mn[0], reason="2-CTA MMA requires a complete CTA pair")
    if cluster_shape_mn is None:
        cluster_shape_mn = (1, 1) if mma_tiler_mn[0] == 128 else (2, 2)
    cluster_shape_mn = require_cluster_shape(cluster_shape_mn, mma_m=mma_tiler_mn[0])
    if mma_tiler_mn[0] == 128 and cluster_shape_mn != (1, 1):
        raise ValueError("cluster_shape_mn must be (1, 1) with a 128-wide M tile")

    a_spec = gemm_a_tensor_spec(a_major)
    b_spec = gemm_b_tensor_spec(b_major)
    c_spec = gemm_c_tensor_spec(c_major)
    require_16_byte_extent("a_tensor", m if a_major == "m" else k, a_dtype)
    require_16_byte_extent("b_tensor", n if b_major == "n" else k, a_dtype)
    require_16_byte_extent("ab12_tensor", m if c_major == "m" else n, ab12_dtype)
    require_16_byte_extent("c_tensor", m if c_major == "m" else output_n, c_dtype)

    cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    launcher = _make_launcher(
        alpha=float(alpha),
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        cluster_overlap_margin=cluster_overlap_margin,
    )
    ab12_tensor, c_tensor = call_cutedsl(
        launcher,
        (a_tensor, b_tensor),
        outputs=(
            BufferSpec("ab12_tensor", (m, n, batch), ab12_dtype, tensor_spec=c_spec),
            BufferSpec("c_tensor", (m, output_n, batch), c_dtype, tensor_spec=c_spec),
        ),
        input_specs=(a_spec, b_spec),
        use_static_tensors=True,
    )
    return GemmSwigluResult(
        ab12_tensor=ab12_tensor,
        c_tensor=c_tensor,
        sfc_tensor=None,
        amax_tensor=None,
    )


__all__ = ["GemmSwigluResult", "gemm_swiglu_wrapper_sm100"]
