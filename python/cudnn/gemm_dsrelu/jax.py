# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for block-scaled dense GEMM + squared-ReLU backward on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from .._jax.cutedsl import BufferSpec, call_cutedsl
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
from .._jax.validation import require_dtype
from ..gemm_validation import (
    require_full_mma_rows,
    require_shape,
    resolve_max_active_clusters,
)


class GemmDsreluResult(NamedTuple):
    """Functional outputs from block-scaled squared-ReLU backward."""

    d_tensor: Any
    dprob_tensor: Any
    amax_tensor: Any | None
    sfd_tensor: Any | None


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    alpha: float,
    sf_vec_size: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    vector_f32: bool,
    cluster_overlap_margin: int,
):
    def launch(stream, a, b, c, sfa, sfb, prob, d, dprob):
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

    return launch


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
) -> GemmDsreluResult:
    """Compute the MXFP8 squared-ReLU backward fusion.

    Let ``X`` be the block-scaled GEMM of A and B. This operation returns
    ``D = C * prob * 2 * relu(X)`` and
    ``dprob = sum_N(C * relu(X)**2)``. A and B use logical shapes
    ``(M, K, L)`` and ``(N, K, L)``; C and D use ``(M, N, L)``; probability
    tensors use ``(M, 1, L)``.

    The JAX binding supports FP8 inputs, E8M0 scales with ``sf_vec_size=32``,
    and non-quantized C/D. ``dprob_tensor`` is a fresh zero-initialized result
    for each invocation. Configuration values must be static under ``jax.jit``.
    """

    if norm_const_tensor is not None:
        raise NotImplementedError("norm_const_tensor is used by the FP8 output path, which is not " "available in the JAX squared-ReLU backward API")

    from .dense_blockscaled_gemm_persistent_dsrelu_quant import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    kernel = Sm100BlockScaledPersistentDenseGemmKernel
    m, n, k, batch, ab_dtype = require_gemm_inputs(a_tensor, b_tensor)
    supported_inputs = {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }
    if ab_dtype not in supported_inputs:
        raise NotImplementedError("The JAX squared-ReLU backward API supports float8_e4m3fn and " f"float8_e5m2 inputs, got {ab_dtype}")
    require_fp8_block_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        batch=batch,
        sf_vec_size=sf_vec_size,
    )

    c_shape = require_array("c_tensor", c_tensor, 3)
    require_shape("c_tensor", c_shape, (m, n, batch))
    prob_shape = require_array("prob_tensor", prob_tensor, 3)
    require_shape("prob_tensor", prob_shape, (m, 1, batch))
    require_dtype("prob_tensor.dtype", prob_tensor, (jnp.float32,))

    supported_outputs = (jnp.float16, jnp.bfloat16, jnp.float32)
    c_dtype = require_dtype("c_tensor.dtype", c_tensor, supported_outputs)
    d_dtype = require_dtype("d_dtype", d_dtype, supported_outputs, default=jnp.bfloat16)
    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)

    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    require_full_mma_rows(
        m,
        mma_tiler_mn[0],
        cta_group_size=2 if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else 1,
        reason="the probability load is not predicated",
    )
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(
        cluster_shape_mn,
        mma_tiler_mn=mma_tiler_mn,
    )

    a_spec = gemm_a_tensor_spec(a_major)
    b_spec = gemm_b_tensor_spec(b_major)
    output_spec = gemm_c_tensor_spec(d_major)
    scale_spec = block_scale_tensor_spec()
    prob_spec = probability_tensor_spec()
    require_16_byte_extent("a_tensor", m if a_major == "m" else k, ab_dtype)
    require_16_byte_extent("b_tensor", n if b_major == "n" else k, ab_dtype)
    require_16_byte_extent("c_tensor", m if d_major == "m" else n, c_dtype)
    require_16_byte_extent("d_tensor", m if d_major == "m" else n, d_dtype)

    launcher = _make_launcher(
        alpha=float(alpha),
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        vector_f32=bool(vector_f32),
        cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
    )
    d_tensor, dprob_tensor = call_cutedsl(
        launcher,
        (a_tensor, b_tensor, c_tensor, sfa_tensor, sfb_tensor, prob_tensor),
        outputs=(
            BufferSpec("d_tensor", (m, n, batch), d_dtype, tensor_spec=output_spec),
            BufferSpec(
                "dprob_tensor",
                (m, 1, batch),
                jnp.float32,
                fill_value=0.0,
            ),
        ),
        input_specs=(
            a_spec,
            b_spec,
            output_spec,
            scale_spec,
            scale_spec,
            prob_spec,
        ),
        use_static_tensors=True,
    )
    return GemmDsreluResult(
        d_tensor=d_tensor,
        dprob_tensor=dprob_tensor,
        amax_tensor=None,
        sfd_tensor=None,
    )


__all__ = ["GemmDsreluResult", "gemm_dsrelu_wrapper_sm100"]
