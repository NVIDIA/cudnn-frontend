# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical dense MXFP8 JAX bindings for the shared Blackwell/Rubin dGLU kernels."""

import math
import os

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div, get_device_type
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import get_compute_capability
from ..canonical import kernel_facing_b, kernel_facing_mx, kernel_facing_prob
from ..canonical_jax import check_grouped_shapes, check_jax_inputs, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode

_kernel_cache = {}
_validated_configs = set()


def dglu_kernel_type():
    if get_device_type() == "rubin":
        from .moe_blockscaled_grouped_gemm_dglu_rubin import BlockScaledMoEGroupedGemmDgluKernel

        return BlockScaledMoEGroupedGemmDgluKernel
    from .moe_blockscaled_grouped_gemm_dglu_dbias import BlockScaledMoEGroupedGemmDgluDbiasKernel

    return BlockScaledMoEGroupedGemmDgluDbiasKernel


@cute.jit
def grouped_dglu_adapter(
    stream,
    a,
    b,
    c,
    sfa,
    sfb,
    offsets,
    alpha,
    beta,
    prob,
    norm_const,
    d_row,
    d_col,
    dprob,
    sfd_row,
    sfd_col,
    *,
    kernel,
    mac,
    rubin,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
    # Metadata-only CuTe views preserve the shared kernel ABI without data copies.
    kernel(
        a=kernel_facing_mx(a),
        b=kernel_facing_b(b),
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=kernel_facing_mx(c),
        d=kernel_facing_mx(d_row),
        d_col=kernel_facing_mx(d_col),
        sfa=sfa,
        sfd_row_tensor=sfd_row,
        sfd_col_tensor=sfd_col,
        amax_tensor=None,
        norm_const_tensor=norm_const,
        padded_offsets=offsets,
        alpha=alpha,
        beta=beta,
        prob=kernel_facing_prob(prob),
        dprob=kernel_facing_prob(dprob),
        dbias_tensor=None,
        max_active_clusters=mac,
        stream=stream,
        linear_offset=cutlass.Float32(linear_offset) if cutlass.const_expr(rubin) else linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=glu_clamp_max,
        glu_clamp_min=glu_clamp_min,
    )


def dglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func):
    check_grouped_shapes(inputs, outputs, backward=True)
    if act_func not in ("dswiglu", "dgeglu"):
        raise ValueError(f"act_func must be 'dswiglu' or 'dgeglu' for the JAX MXFP8 path, got {act_func!r}")
    m, k = inputs["a"].shape
    experts, n, _ = inputs["b"].shape
    if inputs["c"].shape != (m, 2 * n):
        raise ValueError(f"C must have shape {(m, 2 * n)}, got {inputs['c'].shape}")
    if _convert_to_cutlass_data_type(inputs["c"].dtype) not in (cutlass.BFloat16, cutlass.Float16):
        raise ValueError("C must be BF16 or FP16")
    if _convert_to_cutlass_data_type(inputs["a"].dtype) is not _convert_to_cutlass_data_type(inputs["b"].dtype):
        raise ValueError("A and B must have the same MXFP8 dtype")
    if _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must be float32 or bfloat16")
    if experts > 1024:
        raise ValueError("expert count must be <= 1024")
    use_2cta = mma_tiler_mn[0] == 256
    cluster = tuple(cluster_shape_mn or ((2, 1) if use_2cta else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    key = (get_device_type(), experts, tuple(mma_tiler_mn), cluster, margin, discrete_col_sfd, act_func)
    signature = (key, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if signature not in _validated_configs:
        for name, rows, groups in (("sfa", m, 1), ("sfb", n, experts)):
            expected = groups * 512 * ceil_div(rows, 128) * ceil_div(ceil_div(k, 32), 4)
            if math.prod(inputs[name].shape) != expected:
                raise ValueError(f"{name.upper()} must contain the complete MMA-packed scale buffer")
        kernel_type = dglu_kernel_type()
        if not kernel_type.can_implement(
            _convert_to_cutlass_data_type(inputs["a"].dtype),
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            _convert_to_cutlass_data_type(outputs["d_row"].dtype),
            use_2cta,
            tuple(mma_tiler_mn),
            cluster,
            m,
            n,
            k,
            experts,
            "k",
            "k",
            "n",
            kernel_type.FIX_PAD_SIZE,
            act_func,
        ):
            raise ValueError("Unsupported grouped GEMM dGLU tile, cluster, alignment, or layout configuration")
        _validated_configs.add(signature)
    if key not in _kernel_cache:
        major, minor = get_compute_capability()
        if major * 10 + minor < 100:
            raise RuntimeError(f"Grouped GEMM dGLU requires SM100+ compute capability, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = dglu_kernel_type()(
            sf_vec_size=32,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster,
            vectorized_f32=False,
            discrete_col_sfd=discrete_col_sfd,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            act_func=act_func,
        )
        _kernel_cache[key] = (kernel, mac)
    return _kernel_cache[key]


def blockscaled_dglu_jax(
    *,
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
    d_dtype,
    mma_tiler_mn,
    cluster_shape_mn,
    discrete_col_sfd,
    act_func,
    linear_offset,
    geglu_alpha,
    glu_clamp_max,
    glu_clamp_min,
):
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
    m, n = a_tensor.shape[0], b_tensor.shape[1]
    outputs = dict(
        d_row=output_type((m, 2 * n), d_dtype),
        d_col=output_type((m, 2 * n), d_dtype),
        dprob=output_type((m,), cutlass.Float32),
        sfd_row=output_type(sf_shape(m, 2 * n), cutlass.Float8E8M0FNU),
        sfd_col=output_type(sf_shape(2 * n, m), cutlass.Float8E8M0FNU),
    )
    kernel, mac = dglu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, act_func)
    if linear_offset is None:
        linear_offset = 1.0 if act_func == "dgeglu" else 0.0
    result = grouped_call(
        grouped_dglu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        backward=True,
        rubin=get_device_type() == "rubin",
        linear_offset=float(linear_offset),
        geglu_alpha=float(geglu_alpha),
        glu_clamp_max=float(glu_clamp_max),
        glu_clamp_min=float(glu_clamp_min),
    )(*inputs.values())
    return TupleDict(
        d_row_tensor=result[0],
        d_col_tensor=result[1],
        dprob_tensor=result[2],
        dbias_tensor=None,
        amax_tensor=None,
        sfd_row_tensor=result[3],
        sfd_col_tensor=result[4],
    )
