# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous grouped MXFP8 SwiGLU with quantization through cudnn.jax.call."""

import cutlass
import cutlass.cute as cute

from cudnn.jax import call, zeros_init
from ..canonical_jax import grouped_plan, output_type, row_spec, sf_array, sf_shape, sf_zeros
from .api import GroupedGemmSwigluSm100


@cute.jit
def grouped_swiglu_adapter(stream, a, b, sfa, sfb, padded_offsets, alpha, prob, norm_const, c, d, d_col, sfd_row, sfd_col, *, kernel, mac):
    kernel(
        a=a,
        b=b,
        c=c,
        d=d,
        d_col=d_col,
        sfa=sfa,
        sfb=sfb,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_swiglu_jax_sm100(
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
):
    """Canonical MXFP8 forward, eagerly or under jax.jit.

    A (m,k), B (experts,n,k), prob (m,) fp32/bf16, explicit alpha (experts,)
    fp32, norm_const (1,) fp32, and int32 padded_offsets (experts,). Offsets
    must be nondecreasing multiples of 256 within [0,m]; m is padded to 256.
    SF buffers contain packed E8M0 MMA-tiled bytes, at any dense rank (uint8
    bit patterns also accepted). Outputs use natural 2-D shapes and physical
    6-D SF buffers. Output storage is zero-initialized for untouched padding.
    Only FP8 A/B and FP8 D are supported. No automatic differentiation rule;
    use grouped_gemm_dswiglu_jax_sm100 for the fused backward operation.
    """
    m = a_tensor.shape[0]
    if b_tensor.ndim != 3:
        raise ValueError("B must have shape (experts, n, k)")
    n = b_tensor.shape[1]
    inputs = dict(
        a=a_tensor,
        b=b_tensor,
        sfa=sf_array(sfa_tensor),
        sfb=sf_array(sfb_tensor),
        padded_offsets=padded_offsets,
        alpha=alpha_tensor,
        prob=prob_tensor,
        norm_const=norm_const_tensor,
    )
    outputs = dict(
        c=output_type((m, n), c_dtype),
        d=output_type((m, n // 2), d_dtype),
        d_col=output_type((m, n // 2), d_dtype),
        sfd_row=output_type(sf_shape(m, n // 2), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(n // 2, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = grouped_plan(GroupedGemmSwigluSm100, inputs, outputs, backward=False, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    result = call(
        grouped_swiglu_adapter,
        output_shape_dtype=tuple(outputs.values()),
        input_spec=tuple(row_spec(t) for t in inputs.values()),
        output_spec=tuple(row_spec(t) for t in outputs.values()),
        initialized_outputs={0: zeros_init, 1: zeros_init, 2: zeros_init, 3: sf_zeros, 4: sf_zeros},
        kernel=kernel,
        mac=mac,
    )(*inputs.values())
    return {**{f"{name}_tensor": value for name, value in zip(outputs, result)}, "amax_tensor": None}
