# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
API for Grouped GEMM SwiGLU Forward Kernel (SM100+)

This module provides the SwiGLU API class and wrapper for contiguous grouped
block-scaled GEMM in MoE (Mixture of Experts) workloads. The unified grouped
GEMM GLU implementation computes it with ``act_func="swiglu"``.
"""

from __future__ import annotations

from cuda.bindings import driver as cuda
import os
from typing import Tuple, Optional

import cutlass

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import TupleDict, ceil_div
from cudnn.tensor_adapter import detect_framework, framework_dtype
from ..canonical import is_canonical_b, is_flat_sf
from ..glu._blockscaled_api import GroupedGemmGluBlockScaledAPI

_JAX_SF_LAYOUT_ERROR = (
    "the block scale-factor tensors (sfa/sfb and the sfd outputs) are MMA-tiled "
    "(32, 4, m//128, 4, rest_k, l) strided views that are not expressible as JAX arrays "
    "(a row-major JAX array of that shape has different memory); pass torch tensors. "
    "For canonical MXFP8 JAX arrays, use cudnn.jax.grouped_gemm_swiglu"
)


class GroupedGemmSwigluSm100(GroupedGemmGluBlockScaledAPI):
    """API class for Grouped GEMM SwiGLU forward operation on SM100+ GPUs.

    This kernel performs contiguous grouped block-scaled GEMM with SwiGLU activation,
    designed for MoE (Mixture of Experts) workloads. It keeps the SwiGLU argument
    order and runs the unified block-scaled GLU kernel with ``act_func="swiglu"``.

    Key features:
    - Supports variable M per group (aligned to cta_tile_m)
    - Contiguous memory layout for A and D tensors
    - Block-scaled quantization support (MXF8, MXF4, NVF4)

    Example:
        >>> api = GroupedGemmSwigluSm100(
        ...     sample_a=a_tensor,
        ...     ...
        ... )
        >>> api.check_support()
        >>> api.compile()
        >>> api.execute(..., stream)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        # Required quantization output (column-quantized D tensor)
        sample_d_col: torch.Tensor,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
    ):
        """Initialize the GroupedGemmSwigluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_b: Sample B tensor (n, k, l) where l = num_groups
        :param sample_c: Sample C tensor for intermediate storage
        :param sample_d: Sample D output tensor (valid_m, n/2, 1) after SwiGLU
        :param sample_sfa: Sample scale factor A tensor
        :param sample_sfb: Sample scale factor B tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d_col: Column-quantized D tensor (required for quant kernel)
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization
        :param sample_norm_const: Optional normalization constant
        :param sample_prob: Optional probability tensor for gating
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor. Only applies when already output scale factor tensors are provided.
        """
        framework = detect_framework(sample_a)
        if framework == "jax":
            raise ValueError(f"GroupedGemmSwigluSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        if framework != "torch":
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmSwigluSm100; pass torch tensors")
        super().__init__(
            sample_a=sample_a,
            sample_c=sample_c,
            sample_d=sample_d,
            sample_sfa=sample_sfa,
            sample_padded_offsets=sample_padded_offsets,
            sample_alpha=sample_alpha,
            sample_d_col=sample_d_col,
            sample_b=sample_b,
            sample_sfb=sample_sfb,
            sample_sfd_row=sample_sfd_row,
            sample_sfd_col=sample_sfd_col,
            sample_amax=sample_amax,
            sample_norm_const=sample_norm_const,
            sample_prob=sample_prob,
            acc_dtype=None if acc_dtype is None else framework_dtype(acc_dtype, "torch"),
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            discrete_col_sfd=discrete_col_sfd,
            act_func="swiglu",
        )

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor
        :param b_tensor: Input B tensor (weights)
        :param c_tensor: Intermediate C tensor
        :param d_tensor: Output D tensor
        :param sfa_tensor: Scale factor A
        :param sfb_tensor: Scale factor B
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param d_col_tensor: Optional column-quantized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Optional probability tensor
        :param current_stream: CUDA stream
        """
        super().execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmSwigluSm100Objects = {}


def grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
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
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM SwiGLU forward operation.

    Canonical MXFP8 JAX arrays and tracers dispatch to the cudnn.jax API,
    including under jax.jit. Set sf_vec_size=32 and an explicit FP8 d_dtype;
    wrapper defaults stay unchanged. Unsupported JAX options raise ValueError.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Canonical layouts (additive): each input is also accepted in its natural
    row-major form and normalized internally -- A as (valid_m, k), B as (l, n, k)
    C-contiguous, SFA/SFB as dense C-contiguous buffers of any shape with the
    MMA-tiled element count (e.g. flat 1-D, or physical
    (l, mn//128, ceil(ceil(k/sf_vec_size)/4), 32, 4, 4)), and prob as (valid_m,)
    float32 or bfloat16. When A is canonical (2-D), outputs come back natural-shaped:
    c (valid_m, n), d/d_col (valid_m, n//2) row-major, and sfd_row/sfd_col as
    C-contiguous physical (1, mn//128, rest, 32, 4, 4) buffers. The pre-permuted
    kernel-facing forms below keep working unchanged.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1), or canonical (valid_m, k) row-major
        b_tensor: Weight B tensor (n, k, l) k-major, or canonical (l, n, k) row-major
        sfa_tensor: Scale factor A (MMA-tiled view, or canonical dense buffer)
        sfb_tensor: Scale factor B (MMA-tiled view, or canonical dense buffer)
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group scaling; required
        norm_const_tensor: Optional normalization constant. Required when using FP8
            input configurations (i.e., when a_tensor.dtype is FP8 and sfa_tensor.dtype is FP8).
            Should be None for FP4/BF16 input configurations.
        prob_tensor: Optional probability tensor for gating
        acc_dtype: Accumulator data type
        c_dtype: Intermediate C tensor data type (always bfloat16)
        d_dtype: Output D tensor data type (fp8 when ab is fp8, bf16 when ab is fp4)
        cd_major: CD major dimension (note: only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor. Only applies when already output scale factor tensors are provided.
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **c_tensor** (torch.Tensor): Intermediate result tensor
            - **d_tensor** (torch.Tensor): Final output tensor after SwiGLU
            - **d_col_tensor** (torch.Tensor): Column-wise output tensor
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (for quantization)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D (FP8 only)
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D (FP8 only)

            Example usage::

                # Dictionary-style access
                result = grouped_gemm_swiglu_wrapper_sm100(...)
                c = result["c_tensor"]
                d = result["d_tensor"]

                # Tuple unpacking
                c, d, d_col, amax, sfd_row, sfd_col = grouped_gemm_swiglu_wrapper_sm100(...)

                # Integer indexing
                c = result[0]  # c_tensor
    """
    framework = detect_framework(a_tensor)
    if framework == "jax":
        from cudnn.jax import grouped_gemm_swiglu
        from ..canonical_jax import check_jax_wrapper_options

        check_jax_wrapper_options(
            acc_dtype=acc_dtype,
            cd_major=cd_major,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            discrete_col_sfd=discrete_col_sfd,
            current_stream=current_stream,
        )
        return grouped_gemm_swiglu(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            norm_const_tensor=norm_const_tensor,
            c_dtype=c_dtype if c_dtype is not None else cutlass.BFloat16,
            d_dtype=d_dtype if d_dtype is not None else cutlass.BFloat16,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_swiglu_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    c_dtype = _convert_to_cutlass_data_type(c_dtype) if c_dtype is not None else cutlass.BFloat16
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    valid_m = a_tensor.shape[0]
    if is_canonical_b(b_tensor):
        l, n, _ = b_tensor.shape
    else:
        n, _, l = b_tensor.shape
    n_out = n // 2  # After SwiGLU

    # Canonical (sum_m, k) A selects natural-shaped outputs: (m, x) row-major C/D and
    # dense C-contiguous SFD buffers instead of the pre-permuted kernel-facing views.
    canonical_outputs = a_tensor.ndim == 2

    if alpha_tensor is None:
        raise ValueError("alpha_tensor is required for grouped_gemm_swiglu_wrapper_sm100")

    _logger.debug("grouped_gemm_swiglu_wrapper_sm100: Creating output tensors c_tensor, d_tensor, d_col_tensor")

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")
    if canonical_outputs:
        c_tensor = torch.empty((valid_m, n), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        d_tensor = torch.empty((valid_m, n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
        d_col_tensor = torch.empty((valid_m, n_out), dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
    else:
        # 1, m, n, permute (1, 2, 0) -> (m, n, 1)
        c_tensor = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=framework_dtype(c_dtype, "torch"), device=a_tensor.device)
        d_tensor = torch.empty_strided(
            (valid_m, n_out, 1),
            (n_out, 1, valid_m * n_out),
            dtype=framework_dtype(d_dtype, "torch"),
            device=a_tensor.device,
        )
        d_col_tensor = torch.empty_strided(
            (valid_m, n_out, 1),
            (n_out, 1, valid_m * n_out),
            dtype=framework_dtype(d_dtype, "torch"),
            device=a_tensor.device,
        )

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if _convert_to_cutlass_data_type(a_tensor.dtype) in (
        cutlass.Float8E4M3FN,
        cutlass.Float8E5M2,
    ) and _convert_to_cutlass_data_type(
        sfa_tensor.dtype
    ) in (cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN):
        _logger.debug("grouped_gemm_swiglu_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # sfd_row: l=1, mn=valid_m, k=n_out
        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (
            1,
            ceil_div(valid_m, 128),
            ceil_div(sf_k_row, 4),
            32,
            4,
            4,
        )
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device)

        # sfd_col: l=1, mn=n_out, k=valid_m
        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (
            1,
            ceil_div(n_out, 128),
            ceil_div(sf_k_col, 4),
            32,
            4,
            4,
        )
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device)
        if not canonical_outputs:
            sfd_row_tensor = sfd_row_tensor.permute(mma_permute_order)
            sfd_col_tensor = sfd_col_tensor.permute(mma_permute_order)

    if valid_m == 0:
        if d_dtype in (cutlass.BFloat16, cutlass.Float16):
            amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

        _logger.debug("grouped_gemm_swiglu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            d_col_tensor=d_col_tensor,
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
        )

    use_full_dynamic = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    cache_key = (
        use_full_dynamic,
        a_tensor.shape[1:] if not use_full_dynamic else None,
        b_tensor.shape if not use_full_dynamic else None,
        c_tensor.shape[1:] if not use_full_dynamic else None,
        a_tensor.dtype,
        b_tensor.dtype,
        c_tensor.dtype,
        stride_order(a_tensor),
        stride_order(b_tensor),
        stride_order(c_tensor),
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        padded_offsets.shape if not use_full_dynamic else None,
        padded_offsets.stride() if not use_full_dynamic else None,
        padded_offsets.dtype,
        acc_dtype,
        c_dtype,
        d_dtype,
        cd_major,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        discrete_col_sfd,
        prob_tensor is not None,
        l,
        # Canonical-vs-kernel-facing input forms compile different signatures.
        prob_tensor.dtype if prob_tensor is not None else None,
        prob_tensor.ndim if prob_tensor is not None else None,
        (is_flat_sf(sfa_tensor), sfa_tensor.ndim),
        (is_flat_sf(sfb_tensor), sfb_tensor.ndim),
        # The compiled signature binds the SF dtype (e8m0 vs e4m3).
        sfa_tensor.dtype,
        sfb_tensor.dtype,
    )

    if cache_key in _cache_of_GroupedGemmSwigluSm100Objects:
        _logger.debug("group_gemm_swiglu_wrapper_sm100: Using previously cached GroupedGemmSwigluSm100 object")
        grouped_gemm_swiglu, amax_tensor = _cache_of_GroupedGemmSwigluSm100Objects[cache_key]
        # The cuDNN graph API binds data pointers at execute time, not plan-build time.
        # During CUDA graph capture, padded_offsets is allocated in the graph pool
        # (stable address across replays), so passing it directly is graph-safe.
        grouped_gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
    else:
        _logger.debug("group_gemm_swiglu_wrapper_sm100: No previously cached GroupedGemmSwigluSm100 object found, creating new GroupedGemmSwigluSm100 object")
        # Allocate amax_tensor once here; cache-hit calls reuse this buffer so
        # the FillFunctor (torch.full) only fires during warmup, not every step.
        if d_dtype in (cutlass.BFloat16, cutlass.Float16):
            amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)
        grouped_gemm_swiglu = GroupedGemmSwigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_amax=amax_tensor,
            sample_d_col=d_col_tensor,
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
        )

        assert grouped_gemm_swiglu.check_support(), "Unsupported configuration"
        grouped_gemm_swiglu.compile()
        grouped_gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
        _cache_of_GroupedGemmSwigluSm100Objects[cache_key] = (grouped_gemm_swiglu, amax_tensor)

    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
