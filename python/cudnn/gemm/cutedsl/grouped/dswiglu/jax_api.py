# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous grouped MXFP8 dSwiGLU with quantization through cudnn.jax.call."""

import cutlass
import cutlass.cute as cute

from cudnn.api_base import TupleDict
from ..canonical_jax import check_jax_inputs, grouped_call, grouped_plan, output_type, sf_array, sf_shape
from .api import GroupedGemmDswigluSm100


@cute.jit
def grouped_dswiglu_adapter(stream, a, b, c, sfa, sfb, padded_offsets, alpha, beta, prob, norm_const, d_row, d_col, dprob, sfd_row, sfd_col, *, kernel, mac):
    kernel(
        a=a,
        b=b,
        c=c,
        d=d_row,
        d_col=d_col,
        sfa=sfa,
        sfb=sfb,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        beta=beta,
        prob=prob,
        dprob=dprob,
        max_active_clusters=mac,
        stream=stream,
    )


def grouped_gemm_dswiglu(
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
    d_dtype=cutlass.Float8E4M3FN,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
):
    """Canonical MXFP8 backward, eagerly or under jax.jit.

    A (m,k), B (experts,n,k), saved C (m,2n), prob (m,) fp32/bf16,
    explicit alpha/beta (experts,) fp32, norm_const (1,) fp32, and int32
    padded_offsets (experts,). Offsets must be nondecreasing multiples of 256
    within [0,m]; m is padded to 256. SF buffers contain packed E8M0 MMA-tiled
    bytes at any dense rank (uint8 bit patterns also accepted). Returns
    D_row/D_col (m,2n), dprob (m,), and physical 6-D SF buffers. Output storage
    is zero-initialized for padding and dprob accumulation. Only FP8 A/B/D.
    """
    inputs = dict(
        a=a_tensor,
        b=b_tensor,
        c=c_tensor,
        sfa=sfa_tensor,
        sfb=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha=alpha_tensor,
        beta=beta_tensor,
        prob=prob_tensor,
        norm_const=norm_const_tensor,
    )
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    n = b_tensor.shape[1]
    outputs = dict(
        d_row=output_type((m, 2 * n), d_dtype),
        d_col=output_type((m, 2 * n), d_dtype),
        dprob=output_type((m,), cutlass.Float32),
        sfd_row=output_type(sf_shape(m, 2 * n), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(2 * n, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = grouped_plan(GroupedGemmDswigluSm100, inputs, outputs, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn)
    result = grouped_call(
        grouped_dswiglu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
    )(*inputs.values())
    return TupleDict(
        d_row_tensor=result[0],
        d_col_tensor=result[1],
        dprob_tensor=result[2],
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
