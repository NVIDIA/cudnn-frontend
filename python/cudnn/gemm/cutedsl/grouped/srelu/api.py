# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM SReLU Kernel (SM100+)

This module provides a single API class that supports both dense (contiguous)
and discrete weight modes for grouped block-scaled GEMM with output
SReLU output quantization in MoE (Mixture of Experts) workloads.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cutlass
from cuda.bindings import driver as cuda

from cudnn.api_base import TupleDict, ceil_div
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    detect_framework,
    framework_dtype,
)

from ..moe_utils import MoEWeightMode

from ._blockscaled_api import (
    GroupedGemmSreluSm100,
    _JAX_SF_LAYOUT_ERROR,
)


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmSreluSm100Objects = {}


def grouped_gemm_srelu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfa_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    padded_offsets: Optional[torch.Tensor] = None,
    alpha_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    norm_const_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    c_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    tanh_clamp_scale: Optional[float] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM SReLU operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group scaling
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        bias_tensor: Optional per-expert bias, shape ``(n, l)`` in dense mode or ``(n, num_experts)``
            in discrete mode, stride ``(1, n)``. Bias fusion requires ``mma_tiler_mn[1] == 256``.
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant. Required when using FP8
            input configurations (i.e., when a_tensor.dtype is FP8 and sfa_tensor.dtype is FP8).
            Should be None for FP4/BF16 input configurations.
        prob_tensor: Probability tensor for per-row gating (shape `(valid_m, 1, 1)`).
            This argument is required. Pass a tensor of ones when no gating is needed.
        acc_dtype: Accumulator data type
        c_dtype: Output C tensor data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Enable discrete col-major scale factor tensor
        tanh_clamp_scale: Optional soft-clamp scale ``s``. When set, computes
            ``(s * tanh(relu(x) / s)) ** 2`` instead of plain ``relu(x) ** 2``; must be
            finite and > 0. ``None`` (default) is bit-identical to today's SReLU.
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **c_tensor** (torch.Tensor): Accumulator output tensor
            - **d_tensor** (torch.Tensor): Final output tensor
            - **d_col_tensor** (torch.Tensor or None): Column-wise output tensor for low-precision D output
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (for SReLU output quantization)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D (FP8 only)
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D (FP8 only)

            Example usage::

                # Dictionary-style access
                result = grouped_gemm_srelu_wrapper_sm100(...)
                c = result["c_tensor"]
                d = result["d_tensor"]

                # Tuple unpacking
                c, d, d_col, amax, sfd_row, sfd_col = grouped_gemm_srelu_wrapper_sm100(...)

                # Integer indexing
                c = result[0]  # c_tensor
                d = result[1]  # d_tensor
    """
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    framework = detect_framework(a_tensor)
    if framework == "jax" and b_tensor is not None:
        from ..canonical_jax import check_jax_wrapper_options
        from .jax_api import grouped_gemm_srelu_jax_forward

        check_jax_wrapper_options(
            acc_dtype=acc_dtype,
            cd_major=cd_major,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            current_stream=current_stream,
            bias_tensor=bias_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            b_major=b_major,
            use_dynamic_sched=use_dynamic_sched,
            sf_vec_sizes=(16, 32),
        )
        for name, rejected in (("n", n is not None), ("b_dtype", b_dtype is not None)):
            if rejected:
                raise ValueError(f"{name} is unsupported for the JAX block-scaled path")
        return grouped_gemm_srelu_jax_forward(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            padded_offsets,
            alpha_tensor,
            prob_tensor,
            norm_const_tensor,
            c_dtype=c_dtype if c_dtype is not None else cutlass.BFloat16,
            d_dtype=d_dtype if d_dtype is not None else cutlass.BFloat16,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            discrete_col_sfd=discrete_col_sfd,
            sf_vec_size=sf_vec_size,
            tanh_clamp_scale=tanh_clamp_scale,
        )
    if framework == "jax":
        raise ValueError(f"grouped_gemm_srelu_wrapper_sm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_srelu_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    c_dtype = _convert_to_cutlass_data_type(c_dtype) if c_dtype is not None else cutlass.BFloat16
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    valid_m, k_physical, _ = a_tensor.shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_out, _, l = b_tensor.shape
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, l):
            raise ValueError(f"bias_tensor must have shape {(n_out, l)}, got {tuple(bias_tensor.shape)}")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        k_logical = k_physical * 2 if b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n, k_logical)
        n_out = n
        l = num_experts
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, num_experts):
            raise ValueError(f"bias_tensor must have shape {(n_out, num_experts)}, got {tuple(bias_tensor.shape)}")

    is_fp8_input_config = _convert_to_cutlass_data_type(a_tensor.dtype) in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
    ) and _convert_to_cutlass_data_type(sfa_tensor.dtype) in (
        cutlass.Float8E8M0FNU,
        cutlass.Float8E4M3FN,
    )
    is_low_precision_output_config = d_dtype in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
        cutlass.Float4E2M1FN,
    )

    _logger.debug("grouped_gemm_srelu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        c_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
        d_col_tensor = (
            torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
            if is_low_precision_output_config
            else None
        )
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if is_fp8_input_config and is_low_precision_output_config and norm_const_tensor is None:
        raise ValueError(
            "norm_const_tensor is required when FP8 inputs are used with FP8 output "
            "(a_tensor is FP8 and sfa_tensor is FP8 and d_dtype is FP8). "
            "Pass a tensor with shape (1,), e.g. torch.tensor([0.01], dtype=torch.float32, device=a_tensor.device)."
        )

    if not is_low_precision_output_config:
        norm_const_tensor = None

    if is_fp8_input_config and is_low_precision_output_config:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (
            1,
            ceil_div(valid_m, 128),
            ceil_div(sf_k_row, 4),
            32,
            4,
            4,
        )
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (
            1,
            ceil_div(n_out, 128),
            ceil_div(sf_k_col, 4),
            32,
            4,
            4,
        )
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in (cutlass.BFloat16, cutlass.Float16):
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if prob_tensor is None:
        raise ValueError(
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed."
        )

    if valid_m == 0:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            d_col_tensor=d_col_tensor,
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
        )

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def dynamic_tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return None, stride_order(tensor), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Tuple[int, ...], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    use_full_dynamic = is_dense and os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

    if is_dense:
        cache_key = (
            weight_mode,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            a_tensor.dtype,
            b_tensor.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            c_tensor.shape[1:] if not use_full_dynamic else None,
            stride_order(c_tensor),
            c_tensor.dtype,
            d_tensor.shape[1:] if not use_full_dynamic else None,
            stride_order(d_tensor),
            *(
                dynamic_tensor_signature(sfa_tensor)
                if use_full_dynamic
                else dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,))
            ),
            *(dynamic_tensor_signature(sfb_tensor) if use_full_dynamic else tensor_signature(sfb_tensor)),
            *(dynamic_tensor_signature(bias_tensor) if use_full_dynamic else tensor_signature(bias_tensor)),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )
    else:
        cache_key = (
            weight_mode,
            a_tensor.shape[1:],
            stride_order(a_tensor),
            a_tensor.dtype,
            b_shape,
            b_dtype,
            c_tensor.shape[1:],
            stride_order(c_tensor),
            c_tensor.dtype,
            d_tensor.shape[1:],
            stride_order(d_tensor),
            *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
            *tensor_signature(bias_tensor),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            tuple(b_ptrs.shape),
            tuple(b_ptrs.stride()),
            b_ptrs.dtype,
            tuple(sfb_ptrs.shape),
            tuple(sfb_ptrs.stride()),
            sfb_ptrs.dtype,
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            b_major,
            num_experts,
            float(tanh_clamp_scale) if tanh_clamp_scale is not None else None,
        )

    if cache_key in _cache_of_GroupedGemmSreluSm100Objects:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: Using previously cached GroupedGemmSreluSm100 object")
        grouped_gemm_srelu = _cache_of_GroupedGemmSreluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_srelu_wrapper_sm100: No previously cached object found, creating new GroupedGemmSreluSm100 object")
        if is_dense:
            grouped_gemm_srelu = GroupedGemmSreluSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_c=c_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                tanh_clamp_scale=tanh_clamp_scale,
            )
        else:
            grouped_gemm_srelu = GroupedGemmSreluSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_c=c_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                b_major=b_major,
                tanh_clamp_scale=tanh_clamp_scale,
            )

        assert grouped_gemm_srelu.check_support(), "Unsupported configuration"
        grouped_gemm_srelu.compile()
        _cache_of_GroupedGemmSreluSm100Objects[cache_key] = grouped_gemm_srelu

    if is_dense:
        grouped_gemm_srelu.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )
    else:
        grouped_gemm_srelu.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
