# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous grouped MXFP8 SwiGLU with quantization through cudnn.jax.call."""

import cutlass

from ..glu.jax_api import blockscaled_glu_jax


def grouped_gemm_swiglu(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    norm_const_tensor,
    c_dtype=cutlass.BFloat16,
    d_dtype=cutlass.Float8E4M3FN,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    discrete_col_sfd=False,
):
    """Canonical MXFP8 forward, eagerly or under jax.jit.

    A (m,k), B (experts,n,k), prob (m,) fp32/bf16, explicit alpha (experts,)
    fp32, norm_const (1,) fp32, and int32 padded_offsets (experts,). Offsets
    must be nondecreasing multiples of 256 within [0,m]; m is padded to 256.
    SF buffers contain packed E8M0 MMA-tiled bytes, at any dense rank (uint8
    bit patterns also accepted). Outputs use natural 2-D shapes and physical
    6-D SF buffers. ``discrete_col_sfd=True`` packs column scales by expert.
    Rows at or past padded_offsets[-1] are unspecified, as in the torch path.
    Only FP8 A/B and FP8 D are supported. No automatic differentiation rule;
    use cudnn.jax.grouped_gemm_dswiglu for the fused backward operation.
    Alias of grouped_gemm_glu_jax_sm100's MXFP8 mode with act_func="swiglu"
    and generate_c=True.
    """
    return blockscaled_glu_jax(
        a_tensor=a_tensor,
        b_tensor=b_tensor,
        sfa_tensor=sfa_tensor,
        sfb_tensor=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        prob_tensor=prob_tensor,
        norm_const_tensor=norm_const_tensor,
        c_dtype=c_dtype,
        d_dtype=d_dtype,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        discrete_col_sfd=discrete_col_sfd,
        act_func="swiglu",
        linear_offset=None,
        geglu_alpha=1.702,
        glu_clamp_max=7.0,
        glu_clamp_min=-7.0,
        generate_c=True,
    )
