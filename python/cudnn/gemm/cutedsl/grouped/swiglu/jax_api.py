# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contiguous grouped MXFP8 SwiGLU with quantization through cudnn.jax.call."""

import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import get_compute_capability
from ..canonical_jax import check_grouped_shapes, check_jax_inputs, grouped_call, output_type, sf_array, sf_shape
from ..glu.moe_blockscaled_grouped_gemm_glu_bias import BlockScaledMoEGroupedGemmGluBiasKernel
from ..moe_utils import MoEWeightMode

kernel_cache = {}


@cute.jit
def grouped_swiglu_adapter(stream, a, b, sfa, sfb, padded_offsets, alpha, prob, norm_const, c, d, d_col, sfd_row, sfd_col, *, kernel, mac):
    kernel(
        a=a,
        b=b,
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=c,
        d=d,
        d_col=d_col,
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        prob=prob,
        bias=None,
        max_active_clusters=mac,
        stream=stream,
    )


def swiglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn):
    check_grouped_shapes(inputs, outputs, backward=False)
    m, k = inputs["a"].shape
    experts, n, _ = inputs["b"].shape
    rest_k = ceil_div(ceil_div(k, 32), 4)
    for name, rows, groups in (("sfa", m, 1), ("sfb", n, experts)):
        if math.prod(inputs[name].shape) != 512 * ceil_div(rows, 128) * rest_k * groups:
            raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")
    if _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must be float32 or bfloat16")
    if experts > 1024:
        raise ValueError(f"expert count must be <= 1024, got {experts}")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    if not BlockScaledMoEGroupedGemmGluBiasKernel.can_implement(
        _convert_to_cutlass_data_type(inputs["a"].dtype),
        cutlass.Float8E8M0FNU,
        32,
        cutlass.Float32,
        _convert_to_cutlass_data_type(outputs["d"].dtype),
        use_2cta_instrs,
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        m,
        n,
        k,
        experts,
        "k",
        "k",
        "n",
        BlockScaledMoEGroupedGemmGluBiasKernel.FIX_PAD_SIZE,
    ):
        raise ValueError("Unsupported grouped GEMM SwiGLU tile, cluster, alignment, or layout configuration")
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"GroupedGemmSwiglu requires SM100+ compute capability, but found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = BlockScaledMoEGroupedGemmGluBiasKernel(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=True,
            discrete_col_sfd=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func="swiglu",
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


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
):
    """Canonical MXFP8 forward, eagerly or under jax.jit.

    A (m,k), B (experts,n,k), prob (m,) fp32/bf16, explicit alpha (experts,)
    fp32, norm_const (1,) fp32, and int32 padded_offsets (experts,). Offsets
    must be nondecreasing multiples of 256 within [0,m]; m is padded to 256.
    SF buffers contain packed E8M0 MMA-tiled bytes, at any dense rank (uint8
    bit patterns also accepted). Outputs use natural 2-D shapes and physical
    6-D SF buffers. Output storage is zero-initialized for untouched padding.
    Only FP8 A/B and FP8 D are supported. No automatic differentiation rule;
    use cudnn.jax.grouped_gemm_dswiglu for the fused backward operation.
    Runs the unified grouped GEMM GLU kernel with act_func="swiglu".
    """
    inputs = dict(
        a=a_tensor,
        b=b_tensor,
        sfa=sfa_tensor,
        sfb=sfb_tensor,
        padded_offsets=padded_offsets,
        alpha=alpha_tensor,
        prob=prob_tensor,
        norm_const=norm_const_tensor,
    )
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    n = b_tensor.shape[1]
    outputs = dict(
        c=output_type((m, n), c_dtype),
        d=output_type((m, n // 2), d_dtype),
        d_col=output_type((m, n // 2), d_dtype),
        sfd_row=output_type(sf_shape(m, n // 2), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(n // 2, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = swiglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn)
    result = grouped_call(
        grouped_swiglu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
    )(*inputs.values())
    return TupleDict(
        c_tensor=result[0],
        d_tensor=result[1],
        d_col_tensor=result[2],
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
