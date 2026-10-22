# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JAX-native (XLA custom call) entry point for grouped GEMM SReLU, built on
:func:`cudnn.jax.call`.

``grouped_gemm_srelu_jax_forward`` is the canonical block-scaled forward with dense
weights: MXFP8 or packed FP4 A ``(m, k)``, B ``(experts, n, k)`` and already
MMA-packed scale bytes. ``padded_offsets`` values cannot be host-validated under
tracing; malformed offsets are the caller's responsibility here (the eager wrapper
validates them).
"""

import os

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import OperandMajorMode

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.jax import neg_inf_init
from cudnn.tensor_adapter import get_compute_capability
from ..canonical import kernel_facing_b, kernel_facing_fp4, kernel_facing_mx, kernel_facing_prob
from ..canonical_jax import check_block_scaled_operands, check_jax_inputs, check_packed_sf_size, fp8_dtypes, grouped_call, output_type, sf_array, sf_shape
from ..moe_utils import MoEWeightMode
from .moe_blockscaled_grouped_gemm_srelu_quant import BlockScaledMoEGroupedGemmQuantKernel, EpilogueType

kernel_cache = {}
validated_configs = set()


@cute.jit
def grouped_srelu_adapter(stream, a, b, sfa, sfb, padded_offsets, alpha, prob, *rest, kernel, mac, low_precision):
    # rest: [norm_const], c, d, then d_col, sfd_row, sfd_col for FP8 output or amax otherwise
    norm_const = rest[0] if cutlass.const_expr(low_precision) else None
    outputs = rest[int(low_precision) :]
    d = kernel_facing_mx(outputs[1])
    kernel(
        a=kernel_facing_mx(kernel_facing_fp4(a)),
        b=kernel_facing_b(kernel_facing_fp4(b)),
        sfb=sfb,
        n=cutlass.Int32(0),
        k=cutlass.Int32(0),
        b_stride_size=cutlass.Int64(0),
        b_major_mode=OperandMajorMode.K,
        workspace_ptr=cute.make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=128),
        c=kernel_facing_mx(outputs[0]),
        d=d,
        # Without quantization the kernel only needs a D_col descriptor and does not write through it.
        d_col=kernel_facing_mx(outputs[2]) if cutlass.const_expr(low_precision) else d,
        sfa=sfa,
        sfd_row_tensor=outputs[3] if cutlass.const_expr(low_precision) else None,
        sfd_col_tensor=outputs[4] if cutlass.const_expr(low_precision) else None,
        amax_tensor=None if cutlass.const_expr(low_precision) else outputs[2],
        norm_const_tensor=norm_const,
        padded_offsets=padded_offsets,
        alpha=alpha,
        bias=None,
        prob=kernel_facing_prob(prob),
        max_active_clusters=mac,
        stream=stream,
    )


def srelu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size, tanh_clamp_scale):
    a, b = inputs["a"], inputs["b"]
    m, k = a.shape
    experts, n, b_k = b.shape
    if m <= 0 or experts <= 0 or k != b_k:
        raise ValueError("Expected nonempty A/B with matching K and at least one expert")
    if m % 256:
        raise ValueError("A rows must be padded to a multiple of 256")
    ab_dtype, sf_dtype = check_block_scaled_operands(a, b, inputs["sfa"], inputs["sfb"], sf_vec_size)
    if ab_dtype is cutlass.Float4E2M1FN:
        k *= 2
    for name, shape in (("alpha", (experts,)), ("norm_const", (1,))):
        if name in inputs and (tuple(inputs[name].shape) != shape or _convert_to_cutlass_data_type(inputs[name].dtype) is not cutlass.Float32):
            raise ValueError(f"{name} must have shape {shape} and dtype float32")
    if inputs["prob"].shape != (m,) or _convert_to_cutlass_data_type(inputs["prob"].dtype) not in (cutlass.Float32, cutlass.BFloat16):
        raise ValueError("prob must have shape (m,) and dtype float32 or bfloat16")
    if inputs["padded_offsets"].shape != (experts,) or _convert_to_cutlass_data_type(inputs["padded_offsets"].dtype) is not cutlass.Int32:
        raise ValueError("padded_offsets must have shape (experts,) and dtype int32")
    d_dtype = _convert_to_cutlass_data_type(outputs["d"].dtype)
    low_precision = d_dtype in fp8_dtypes
    if low_precision and ab_dtype not in fp8_dtypes:
        raise ValueError("FP4 inputs require a BF16 or FP16 d_dtype")
    use_2cta_instrs = mma_tiler_mn[0] == 256
    cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if use_2cta_instrs else (1, 1)))
    margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
    config = (experts, tuple(mma_tiler_mn), cluster_shape_mn, margin, discrete_col_sfd, low_precision, sf_vec_size, tanh_clamp_scale)
    validation_key = (config, tuple((name, tuple(t.shape), str(t.dtype)) for name, t in (*inputs.items(), *outputs.items())))
    if validation_key not in validated_configs:
        check_packed_sf_size(inputs, (("sfa", m, 1), ("sfb", n, experts)), ceil_div(ceil_div(k, sf_vec_size), 4))
        if experts > 1024:
            raise ValueError(f"expert count must be <= 1024, got {experts}")
        if not BlockScaledMoEGroupedGemmQuantKernel.can_implement(
            ab_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.Float32,
            d_dtype,
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
            BlockScaledMoEGroupedGemmQuantKernel.FIX_PAD_SIZE,
        ):
            raise ValueError("Unsupported grouped GEMM SReLU dtype, tile, cluster, alignment, or layout configuration")
        validated_configs.add(validation_key)
    if config not in kernel_cache:
        major, minor = get_compute_capability()
        if major < 10:
            raise RuntimeError(f"Grouped GEMM SReLU requires SM100+, found SM{major}{minor}")
        mac = cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1]) - margin
        if mac <= 0:
            raise ValueError("CUDNNFE_CLUSTER_OVERLAP_MARGIN leaves no active clusters")
        kernel = BlockScaledMoEGroupedGemmQuantKernel(
            sf_vec_size=sf_vec_size,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=tuple(mma_tiler_mn),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=False,
            generate_sfd=low_precision,
            discrete_col_sfd=discrete_col_sfd,
            generate_c=True,
            enable_bias=False,
            expert_cnt=experts,
            weight_mode=MoEWeightMode.DENSE,
            epilogue_type=EpilogueType.SRELU.value,
            tanh_clamp_scale=tanh_clamp_scale,
        )
        kernel_cache[config] = (kernel, mac)
    return kernel_cache[config]


def grouped_gemm_srelu_jax_forward(
    a_tensor,
    b_tensor,
    sfa_tensor,
    sfb_tensor,
    padded_offsets,
    alpha_tensor,
    prob_tensor,
    norm_const_tensor=None,
    *,
    c_dtype=cutlass.BFloat16,
    d_dtype=cutlass.BFloat16,
    mma_tiler_mn=(256, 256),
    cluster_shape_mn=None,
    discrete_col_sfd=False,
    sf_vec_size=32,
    tanh_clamp_scale=None,
):
    """Canonical block-scaled grouped GEMM + SReLU forward, eagerly or under jax.jit.

    A (m,k), dense B (experts,n,k), explicit alpha (experts,) fp32, prob (m,)
    fp32/bf16, and int32 padded_offsets (experts,). Offsets must be
    nondecreasing multiples of 256 within [0,m]; m is padded to 256. A/B are
    MXFP8, or packed FP4 as uint8 with two values per byte (K is twice the
    array extent). SF buffers contain packed MMA-tiled bytes at any dense rank:
    E8M0 with ``sf_vec_size=32`` for MXFP8 (uint8 bit patterns also accepted),
    E8M0 or E4M3 (NVFP4, ``sf_vec_size=16``) for FP4. Returns a TupleDict with
    the eager wrapper's keys: the pre-activation C (m,n) and D (m,n). With a
    BF16/FP16 ``d_dtype`` the per-expert amax (experts,1) is also produced.
    With an FP8 ``d_dtype``, which needs MXFP8 inputs, norm_const (1,) fp32 is
    required and D_col and the physical 6-D SFD buffers are also produced;
    ``discrete_col_sfd=True`` packs column scales by expert. Outputs are not
    initialized, so rows at or past padded_offsets[-1] are unspecified, as in
    the torch path. SM100+, as in Torch.
    """
    d_dtype = _convert_to_cutlass_data_type(d_dtype)
    low_precision = d_dtype in fp8_dtypes
    inputs = dict(a=a_tensor, b=b_tensor, sfa=sfa_tensor, sfb=sfb_tensor, padded_offsets=padded_offsets, alpha=alpha_tensor, prob=prob_tensor)
    if low_precision:
        inputs["norm_const"] = norm_const_tensor
    check_jax_inputs(inputs)
    inputs["sfa"] = sf_array(sfa_tensor)
    inputs["sfb"] = sf_array(sfb_tensor)
    m = a_tensor.shape[0]
    experts, n = b_tensor.shape[:2]
    outputs = dict(c=output_type((m, n), c_dtype), d=output_type((m, n), d_dtype))
    if low_precision:
        outputs.update(
            d_col=output_type((m, n), d_dtype),
            sfd_row=output_type(sf_shape(m, n, sf_vec_size), cutlass.Float8E8M0FNU),
            sfd_col=output_type(sf_shape(n, m, sf_vec_size), cutlass.Float8E8M0FNU),
        )
    else:
        outputs["amax"] = output_type((experts, 1), cutlass.Float32)
    kernel, mac = srelu_plan(inputs, outputs, mma_tiler_mn, cluster_shape_mn, discrete_col_sfd, sf_vec_size, tanh_clamp_scale)
    result = grouped_call(
        grouped_srelu_adapter,
        kernel,
        mac,
        tuple(output_type(t.shape, t.dtype) for t in inputs.values()),
        tuple(outputs.values()),
        initialized=() if low_precision else ((2, neg_inf_init),),
        low_precision=low_precision,
    )(*inputs.values())
    return TupleDict(
        c_tensor=result[0],
        d_tensor=result[1],
        d_col_tensor=result[2] if low_precision else None,
        amax_tensor=None if low_precision else result[2],
        sfd_row_tensor=result[3] if low_precision else None,
        sfd_col_tensor=result[4] if low_precision else None,
    )
