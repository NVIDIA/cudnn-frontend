# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for contiguous grouped GEMM + dSwiGLU on SM100."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.gemm import (
    block_scale_tensor_spec,
    gemm_a_tensor_spec,
    gemm_b_tensor_spec,
    gemm_c_tensor_spec,
    probability_tensor_spec,
    require_16_byte_extent,
)
from ..._jax.grouped_gemm import (
    require_grouped_fp8_scales,
    require_grouped_gemm_inputs,
    require_grouped_probability,
    require_grouped_vector,
)
from ..._jax.validation import require_dtype
from ...gemm_validation import (
    block_scale_shape,
    require_shape,
    resolve_max_active_clusters,
)


class GroupedGemmDswigluResult(NamedTuple):
    """Functional outputs from contiguous grouped GEMM + dSwiGLU."""

    d_row_tensor: Any
    d_col_tensor: Any
    dprob_tensor: Any
    amax_tensor: None
    sfd_row_tensor: Any
    sfd_col_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    acc_dtype: Any,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sf_vec_size: int,
    vector_f32: bool,
    discrete_col_sfd: bool,
    expert_cnt: int,
    epilogue_op: str,
    cluster_overlap_margin: int,
):
    def launch(
        stream,
        a,
        b,
        c,
        sfa,
        sfb,
        padded_offsets,
        alpha,
        beta,
        prob,
        norm_const,
        d_row,
        d_col,
        dprob,
        sfd_row,
        sfd_col,
    ):
        import cutlass
        import cutlass.cute as cute
        from cutlass.jax import jax_to_cutlass_dtype

        from .grouped_gemm_dswiglu_quant import BlockScaledContiguousGroupedGemmKernel

        if epilogue_op == "relu":

            def epilogue(x):
                return cute.where(x > 0, x, cute.full_like(x, 0))

        elif epilogue_op == "srelu":

            def epilogue(x):
                return cute.where(x > 0, x, cute.full_like(x, 0)) ** 2

        else:

            def epilogue(x):
                return x

        kernel = BlockScaledContiguousGroupedGemmKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=jax_to_cutlass_dtype(acc_dtype),
            use_2cta_instrs=mma_tiler_mn[0] == BlockScaledContiguousGroupedGemmKernel.TWO_CTA_MMA_TILER_M,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=vector_f32,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=expert_cnt,
            use_mono_increase_expert_idx=True,
        )
        max_active_clusters = resolve_max_active_clusters(
            cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]),
            cluster_overlap_margin,
        )
        kernel(
            a,
            b,
            c,
            d_row,
            d_col,
            sfa,
            sfb,
            sfd_row,
            sfd_col,
            None,
            norm_const,
            padded_offsets,
            alpha,
            beta,
            prob,
            dprob,
            max_active_clusters,
            stream,
            epilogue_op=epilogue,
        )

    return launch


def grouped_gemm_dswiglu_wrapper_sm100(
    a_tensor: Any,
    b_tensor: Any,
    c_tensor: Any,
    sfa_tensor: Any,
    sfb_tensor: Any,
    padded_offsets: Any,
    alpha_tensor: Any,
    beta_tensor: Optional[Any],
    prob_tensor: Any,
    norm_const_tensor: Any,
    acc_dtype: Any = None,
    d_dtype: Any = None,
    cd_major: str = "n",
    mma_tiler_mn: tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[tuple[int, int]] = None,
    sf_vec_size: int = 32,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    epilogue_op: Optional[str] = None,
) -> GroupedGemmDswigluResult:
    """Compute the MXFP8 contiguous grouped dSwiGLU fusion.

    ``dprob_tensor`` is modeled as a fresh zero-initialized JAX result instead
    of a caller-owned mutable buffer. If ``beta_tensor`` is ``None``, a vector
    of ones is supplied, matching the convenience behavior of the Torch API.
    Configuration values are static when traced by :func:`jax.jit`.
    """

    from .grouped_gemm_dswiglu_quant import BlockScaledContiguousGroupedGemmKernel

    kernel = BlockScaledContiguousGroupedGemmKernel
    m, n, k, experts, ab_dtype = require_grouped_gemm_inputs(
        a_tensor,
        b_tensor,
        padded_offsets,
        alpha_tensor,
        max_experts=kernel.MAX_EXPERTS,
    )
    if sf_vec_size != kernel.FP8_SF_VEC_SIZE:
        raise ValueError(f"FP8 grouped GEMM requires sf_vec_size={kernel.FP8_SF_VEC_SIZE}, got {sf_vec_size}")
    if m_aligned != kernel.FIX_PAD_SIZE:
        raise ValueError(f"m_aligned must be {kernel.FIX_PAD_SIZE}, got {m_aligned}")
    require_grouped_fp8_scales(
        sfa_tensor,
        sfb_tensor,
        m=m,
        n=n,
        k=k,
        experts=experts,
        sf_vec_size=sf_vec_size,
    )

    output_n = 2 * n
    c_shape = tuple(getattr(c_tensor, "shape", ()))
    require_shape("c_tensor", c_shape, (m, output_n, 1))
    c_dtype = require_dtype(
        "c_tensor.dtype",
        c_tensor,
        (jnp.float32, jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2),
    )
    if vector_f32 and c_dtype in {
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    }:
        raise ValueError("vector_f32 does not support an FP8 c_tensor")
    require_grouped_probability("prob_tensor", prob_tensor, m=m)
    require_grouped_vector("norm_const_tensor", norm_const_tensor, length=1)
    if beta_tensor is None:
        beta_tensor = jnp.ones((experts,), dtype=jnp.float32)
    else:
        require_grouped_vector("beta_tensor", beta_tensor, length=experts)

    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    d_dtype = require_dtype(
        "d_dtype",
        d_dtype,
        (jnp.float8_e4m3fn, jnp.float8_e5m2),
        default=ab_dtype,
    )
    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major!r}")
    normalized_epilogue = "identity" if epilogue_op in (None, "none", "identity") else epilogue_op
    if normalized_epilogue not in ("identity", "relu", "srelu"):
        raise ValueError(f"epilogue_op must be None, 'none', 'identity', 'relu', or 'srelu', got {epilogue_op!r}")

    mma_tiler_mn = kernel.require_mma_tiler(mma_tiler_mn)
    if cluster_shape_mn is None:
        cluster_shape_mn = (2, 1) if mma_tiler_mn[0] == kernel.TWO_CTA_MMA_TILER_M else (1, 1)
    cluster_shape_mn = kernel.require_cluster_shape(cluster_shape_mn, mma_tiler_mn=mma_tiler_mn)

    a_spec = gemm_a_tensor_spec("k")
    b_spec = gemm_b_tensor_spec("k")
    output_spec = gemm_c_tensor_spec("n")
    scale_spec = block_scale_tensor_spec()
    require_16_byte_extent("a_tensor", k, ab_dtype)
    require_16_byte_extent("b_tensor", k, ab_dtype)
    require_16_byte_extent("c_tensor", output_n, c_dtype)
    require_16_byte_extent("d_tensor", output_n, d_dtype)

    d_row_tensor, d_col_tensor, dprob_tensor, sfd_row_tensor, sfd_col_tensor = call_cutedsl(
        _make_launcher(
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=bool(vector_f32),
            discrete_col_sfd=bool(discrete_col_sfd),
            expert_cnt=experts,
            epilogue_op=normalized_epilogue,
            cluster_overlap_margin=int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0")),
        ),
        (
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            norm_const_tensor,
        ),
        outputs=(
            BufferSpec("d_row_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
            BufferSpec("d_col_tensor", (m, output_n, 1), d_dtype, tensor_spec=output_spec),
            BufferSpec("dprob_tensor", (m, 1, 1), jnp.float32, fill_value=0.0),
            BufferSpec(
                "sfd_row_tensor",
                block_scale_shape(m, output_n, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
            BufferSpec(
                "sfd_col_tensor",
                block_scale_shape(output_n, m, 1, sf_vec_size),
                jnp.float8_e8m0fnu,
                tensor_spec=scale_spec,
            ),
        ),
        input_specs=(
            a_spec,
            b_spec,
            output_spec,
            scale_spec,
            scale_spec,
            None,
            None,
            None,
            probability_tensor_spec(),
            None,
        ),
        use_static_tensors=True,
    )
    return GroupedGemmDswigluResult(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        amax_tensor=None,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )


__all__ = ["GroupedGemmDswigluResult", "grouped_gemm_dswiglu_wrapper_sm100"]
