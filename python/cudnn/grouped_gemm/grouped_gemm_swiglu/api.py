# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
API for Grouped GEMM SwiGLU Forward Kernel (SM100+)

This module provides the API class for contiguous grouped block-scaled GEMM
with SwiGLU activation for MoE (Mixture of Experts) workloads.
"""

from .grouped_gemm_swiglu_quant import (
    BlockScaledContiguousGroupedGemmKernel,
    BlockScaledContiguousGroupedGemmKernelNoDlpack,
)
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
from packaging import version

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2


class GroupedGemmSwigluSm100(APIBase):
    """API class for Grouped GEMM SwiGLU forward operation on SM100+ GPUs.

    This kernel performs contiguous grouped block-scaled GEMM with SwiGLU activation,
    designed for MoE (Mixture of Experts) workloads.

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
        sample_tile_idx_to_expert_idx: torch.Tensor,
        sample_num_non_exiting_tiles: torch.Tensor,
        sample_alpha: torch.Tensor,
        # Required quantization output (column-quantized D tensor)
        sample_d_col: torch.Tensor,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        sample_m_split_cumsum: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: torch.dtype = torch.float32,
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
        :param sample_tile_idx_to_expert_idx: Mapping from tile index to expert/group index
        :param sample_num_non_exiting_tiles: Number of valid tiles
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d_col: Column-quantized D tensor (required for quant kernel)
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization
        :param sample_norm_const: Optional normalization constant
        :param sample_prob: Optional probability tensor for gating
        :param sample_m_split_cumsum: Optional m split cumulative sum tensor. Required when discrete_col_sfd is True.
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor. Only applies when already output scale factor tensors are provided.
        """
        super().__init__()

        self._logger.warning("GroupedGemmSwigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        # Store sample tensors
        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_c = sample_c
        self.sample_d = sample_d
        self.sample_sfa = sample_sfa
        self.sample_sfb = sample_sfb
        self.sample_tile_idx_to_expert_idx = sample_tile_idx_to_expert_idx
        self.sample_num_non_exiting_tiles = sample_num_non_exiting_tiles
        self.sample_alpha = sample_alpha

        # Optional quantization outputs
        self.sample_d_col = sample_d_col
        self.sample_sfd_row = sample_sfd_row
        self.sample_sfd_col = sample_sfd_col
        self.sample_amax = sample_amax
        self.sample_norm_const = self._unpad_tensor_to_ndim(sample_norm_const, 1, "norm_const")
        self.sample_prob = sample_prob
        self.sample_m_split_cumsum = sample_m_split_cumsum

        # Configuration
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        else:
            self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.discrete_col_sfd = discrete_col_sfd

        # Determine kernel variant based on sample tensor dtypes
        # NoDlpack kernels are required for:
        # - FP4 dtypes (any of ab_dtype, c_dtype, d_dtype)
        # - FP8 dtypes on PyTorch < 2.10.0
        ab_dtype = self.sample_a.dtype
        c_dtype = self.sample_c.dtype
        d_dtype = self.sample_d.dtype
        torch_version = version.parse(torch.__version__)
        is_ab_fp4 = self._is_fp4x2(ab_dtype)
        is_c_fp4 = self._is_fp4x2(c_dtype)
        is_d_fp4 = self._is_fp4x2(d_dtype)
        is_ab_fp8 = self._is_fp8(ab_dtype)
        is_c_fp8 = self._is_fp8(c_dtype)
        is_d_fp8 = self._is_fp8(d_dtype)
        _fp8_dlpack_supported = version.parse(torch_version.base_version) >= version.parse("2.10.0")
        use_no_dlpack_kernel = is_ab_fp4 or is_c_fp4 or is_d_fp4 or ((is_ab_fp8 or is_c_fp8 or is_d_fp8) and not _fp8_dlpack_supported)

        if use_no_dlpack_kernel:
            self._logger.debug("Using NoDlpack kernel due to FP4 dtype or FP8 dtype on incompatible torch version")
            self._kernel = BlockScaledContiguousGroupedGemmKernelNoDlpack
        else:
            self._kernel = BlockScaledContiguousGroupedGemmKernel

        self._interpret_uint8_as_fp4x2 = True
        self._logger.debug(f"__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        all_none = all(x is None for x in [self.sample_sfd_row, self.sample_sfd_col, self.sample_norm_const])
        none_none = all(x is not None for x in [self.sample_sfd_row, self.sample_sfd_col, self.sample_norm_const])
        self._value_error_if(
            not (all_none or none_none),
            "sample_sfd_row, sample_sfd_col, and norm_const must be all None or all not None",
        )
        self.generate_sfd = none_none
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, discrete_col_sfd will be ignored")
            self.discrete_col_sfd = False
        self._value_error_if(self.discrete_col_sfd and self.sample_m_split_cumsum is None, "sample_m_split_cumsum is required when discrete_col_sfd is True")

        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self._tensor_shape(self.sample_a, name="sample_a")
        n, _, l = self._tensor_shape(self.sample_b, name="sample_b")
        _, _, _one = self._tensor_shape(self.sample_c, name="sample_c")
        _, n_2, _one = self._tensor_shape(self.sample_d, name="sample_d")

        self._check_tensor_shape(self.sample_a, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.sample_b, (n, k, l), "B")
        self._check_tensor_shape(self.sample_c, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.sample_d, (tensor_m, n // 2, 1), "D")

        self._check_tensor_shape(self.sample_d_col, (tensor_m, n // 2, 1), "D_col")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfa, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        self._check_tensor_shape(self.sample_sfb, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
        rest_n2 = ceil_div(ceil_div(n // 2, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfd_row, (32, 4, ceil_div(tensor_m, 128), 4, rest_n2, 1), "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfd_col, (32, 4, ceil_div(n // 2, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.sample_alpha, (l,), "alpha")
        self._check_tensor_shape(self.sample_prob, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.sample_amax, (l, 1), "amax")
        self._check_tensor_shape(self.sample_num_non_exiting_tiles, (1,), "num_non_exiting_tiles")
        self._check_tensor_shape(self.sample_norm_const, (1,), "norm_const")
        self._check_tensor_shape(self.sample_m_split_cumsum, (l + 1,), "m_split_cumsum")

        _, self.a_stride_order = self._check_tensor_stride(self.sample_a, stride=[(k, 1, tensor_m * k)], extra_error_msg="A must have k-major layout")
        _, self.b_stride_order = self._check_tensor_stride(self.sample_b, stride=[(k, 1, n * k)], extra_error_msg="B must have k-major layout")
        _, self.c_stride_order = self._check_tensor_stride(self.sample_c, stride=[(n, 1, tensor_m * n)], extra_error_msg="C must have n-major layout")
        _, self.d_stride_order = self._check_tensor_stride(self.sample_d, stride=[(n_2, 1, tensor_m * n_2)], extra_error_msg="D must have n-major layout")
        _, self.d_col_stride_order = self._check_tensor_stride(
            self.sample_d_col, stride=[(n_2, 1, tensor_m * n_2)], extra_error_msg="D_col must have n-major layout"
        )
        self.cd_stride_order = self.c_stride_order

        self._logger.debug("Checking data types")
        self.ab_dtype = self._check_dtype(
            self.sample_a,
            dtype=[
                torch.float4_e2m1fn_x2,
                torch.uint8,
                torch.float8_e5m2,
                torch.float8_e4m3fn,
            ],
            name="A/B",
        )
        self._check_dtype(self.sample_b, dtype=self.ab_dtype, name="B", extra_error_msg="B must have the same dtype as A")

        self.sf_dtype = self._check_dtype(
            self.sample_sfa,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD_row/SFD_col",
        )
        self._check_dtype(self.sample_sfb, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must have the same dtype as SFA")
        self._check_dtype(self.sample_sfd_row, dtype=self.sf_dtype, name="SFD_row", extra_error_msg="SFD_row must have the same dtype as SFA")
        self._check_dtype(self.sample_sfd_col, dtype=self.sf_dtype, name="SFD_col", extra_error_msg="SFD_col must have the same dtype as SFA")

        self._value_error_if(self.sf_vec_size not in [16, 32], f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        self._value_error_if(
            self.sf_dtype in [torch.float8_e4m3fn] and self.sf_vec_size == 32,
            f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )
        self._value_error_if(
            self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16, f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported"
        )

        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Accumulator", extra_error_msg="Accumulator must be float32")
        self.c_dtype = self._check_dtype(
            self.sample_c,
            dtype=[torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2],
            name="C",
            extra_error_msg="C must have the same dtype as A",
        )

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.sample_d, dtype=[torch.bfloat16, torch.float32], name="D", extra_error_msg="D must be bf16 or float32 when ab_dtype is fp4"
            )
        else:
            self.d_dtype = self._check_dtype(
                self.sample_d,
                dtype=[
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.float4_e2m1fn_x2,
                ],  # torch.float32 fails non-deterministicly
                name="D",
            )
        self._check_dtype(self.sample_d_col, dtype=self.d_dtype, name="D_col", extra_error_msg="D_col must have the same dtype as D")

        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and self.sf_vec_size == 16 and self.d_dtype == torch.float32,  # Fails to compile
            f"Invalid configuration: fp4 ab_dtype, sf_vec_size 16, d_dtype float32 is not supported. Please use sf_vec_size 32 or d_dtype bf16 instead",
        )

        self._logger.debug("Checking MMA tile shape and cluster shape")
        self._value_error_if(
            not self.use_2cta_instrs and self.mma_tiler_mn[0] not in [64, 128],
            f"MMA tiler M must be 64 or 128 when use_2cta_instrs=False, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.use_2cta_instrs and self.mma_tiler_mn[0] not in [128, 256],
            f"MMA tiler M must be 128 or 256 when use_2cta_instrs=True, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(self.mma_tiler_mn[1] not in [128, 256], f"MMA tiler N must be 128 or 256, got {self.mma_tiler_mn[1]}")
        self._value_error_if(
            self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0,
            f"cluster_shape_mn[0] must be divisible by 2 when use_2cta_instrs=True, got {self.cluster_shape_mn[0]}",
        )
        self._value_error_if(
            not (
                self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
                and self.cluster_shape_mn[0] > 0
                and self.cluster_shape_mn[1] > 0
                and self.cluster_shape_mn[0] <= 4
                and self.cluster_shape_mn[1] <= 4
                and is_power_of_2(self.cluster_shape_mn[0])
                and is_power_of_2(self.cluster_shape_mn[1])
            ),
            f"Invalid cluster shape: expected values to be powers of 2 and cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn[0]},{self.cluster_shape_mn[1]}",
        )
        cluster_tiler_m = (self.cluster_shape_mn[0] // (2 if self.use_2cta_instrs else 1)) * self.mma_tiler_mn[0]
        # Skip invalid cluster tiler shape since contiguous layout can't handle oob access
        # The contiguous layout means the aligned data is stored in a contiguous manner.
        # It can't handle runtime oob when alignment is not align with the tile_M,
        # since the problem shape of TMA store can't be changed at runtime.
        self._value_error_if(cluster_tiler_m not in [128, 256], f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}")
        # Check if m_aligned is a multiple of cluster_tiler_m
        # This ensures that each group's M dimension (which is a multiple of m_aligned)
        # won't be split across tiles, preventing a single tile from loading data
        # from multiple groups (which would access wrong B matrix data)
        self._value_error_if(
            self.m_aligned % self.mma_tiler_mn[0] != 0,
            f"Invalid m_aligned: expected m_aligned to be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0",
        )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        self._value_error_if(
            not (
                check_contigous_16B_alignment(self.ab_dtype, self.a_stride_order, (tensor_m, k, l))
                and check_contigous_16B_alignment(self.ab_dtype, self.b_stride_order, (n, k, l))
                and check_contigous_16B_alignment(self.d_dtype, self.cd_stride_order, (tensor_m, n, l))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        # Disabled configurations
        self._not_implemented_error_if(
            (self._is_fp8(self.ab_dtype)) and (self.mma_tiler_mn[1] == 128) and (self._is_fp8(self.d_dtype)),
            f"Invalid configuration: fp8 ab_dtype and sf_vec_size 32 with mma_tiler_mn[1] == 128 and fp8 d_dtype is not supported"
            + f"Please use mma_tiler_mn[1] == 256 instead",
        )
        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and (self.c_dtype not in [torch.float16, torch.bfloat16]),
            f"Invalid configuration: for fp4 ab_dtype, c_dtype must be float16 or bfloat16, got {self.c_dtype}",
        )

        # Check environment
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmSwiglu requires SM100+ compute capability, " f"but found SM{compute_capability} on device {device}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        """Compile the kernel.

        :param current_stream: CUDA stream to use
        """
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        gemm_swiglu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vector_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

        if self._kernel is BlockScaledContiguousGroupedGemmKernel:
            self._logger.debug("Compiling grouped_gemm_swiglu kernel (dlpack)")
            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                a=from_dlpack(self.sample_a, assumed_align=16),
                b=from_dlpack(self.sample_b, assumed_align=16),
                c=from_dlpack(self.sample_c, assumed_align=16),
                d=from_dlpack(self.sample_d, assumed_align=16),
                d_col=from_dlpack(self.sample_d_col, assumed_align=16) if self.sample_d_col is not None else None,
                sfa=from_dlpack(self.sample_sfa, assumed_align=16),
                sfb=from_dlpack(self.sample_sfb, assumed_align=16),
                sfd_row_tensor=from_dlpack(self.sample_sfd_row, assumed_align=16) if self.sample_sfd_row is not None else None,
                sfd_col_tensor=from_dlpack(self.sample_sfd_col, assumed_align=16) if self.sample_sfd_col is not None else None,
                amax_tensor=from_dlpack(self.sample_amax, assumed_align=16) if self.sample_amax is not None else None,
                norm_const_tensor=from_dlpack(self.sample_norm_const) if self.sample_norm_const is not None else None,
                tile_idx_to_expert_idx=from_dlpack(self.sample_tile_idx_to_expert_idx, assumed_align=16),
                num_non_exiting_tiles=from_dlpack(self.sample_num_non_exiting_tiles, assumed_align=16),
                m_split_cumsum=from_dlpack(self.sample_m_split_cumsum, assumed_align=16) if self.sample_m_split_cumsum is not None else None,
                alpha=from_dlpack(self.sample_alpha, assumed_align=16),
                prob=from_dlpack(self.sample_prob, assumed_align=16) if self.sample_prob is not None else None,
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
            self._logger.debug("Compiling grouped_gemm_swiglu kernel (no_dlpack)")
            # Create cute pointers/tensors manually to avoid DLPack requirements
            a_ptr, a_shape, a_order = self._make_cute_tensor_descriptor(self.sample_a, name="A")
            b_ptr, b_shape, b_order = self._make_cute_tensor_descriptor(self.sample_b, name="B")
            c_ptr, c_shape, c_order = self._make_cute_tensor_descriptor(self.sample_c, name="C")
            d_ptr, d_shape, d_order = self._make_cute_tensor_descriptor(self.sample_d, name="D")
            d_col_ptr, d_col_shape, d_col_order = self._make_cute_tensor_descriptor(self.sample_d_col, name="D_col")
            sfa_ptr, sfa_shape, sfa_order = self._make_cute_tensor_descriptor(self.sample_sfa, name="SFA")
            sfb_ptr, sfb_shape, sfb_order = self._make_cute_tensor_descriptor(self.sample_sfb, name="SFB")
            sfd_row_ptr, sfd_row_shape, sfd_row_order = self._make_cute_tensor_descriptor(self.sample_sfd_row, name="SFD_row")
            sfd_col_ptr, sfd_col_shape, sfd_col_order = self._make_cute_tensor_descriptor(self.sample_sfd_col, name="SFD_col")
            amax_ptr, amax_shape, amax_order = self._make_cute_tensor_descriptor(self.sample_amax, name="amax")
            norm_const_ptr, norm_const_shape, norm_const_order = self._make_cute_tensor_descriptor(self.sample_norm_const, name="norm_const")
            tile_idx_ptr, tile_idx_shape, tile_idx_order = self._make_cute_tensor_descriptor(self.sample_tile_idx_to_expert_idx, name="tile_idx")
            num_tiles_ptr, num_tiles_shape, num_tiles_order = self._make_cute_tensor_descriptor(self.sample_num_non_exiting_tiles, name="num_tiles")
            m_split_cumsum_ptr, m_split_cumsum_shape, m_split_cumsum_order = self._make_cute_tensor_descriptor(
                self.sample_m_split_cumsum, name="m_split_cumsum"
            )
            alpha_ptr, alpha_shape, alpha_order = self._make_cute_tensor_descriptor(self.sample_alpha, name="alpha")
            prob_ptr, prob_shape, prob_order = self._make_cute_tensor_descriptor(self.sample_prob, name="prob")

            self._compiled_kernel = cute.compile(
                gemm_swiglu,
                a_ptr=a_ptr,
                a_shape=a_shape,
                a_order=a_order,
                b_ptr=b_ptr,
                b_shape=b_shape,
                b_order=b_order,
                c_ptr=c_ptr,
                c_shape=c_shape,
                c_order=c_order,
                d_ptr=d_ptr,
                d_shape=d_shape,
                d_order=d_order,
                d_col_ptr=d_col_ptr,
                d_col_shape=d_col_shape,
                d_col_order=d_col_order,
                sfa_ptr=sfa_ptr,
                sfa_shape=sfa_shape,
                sfa_order=sfa_order,
                sfb_ptr=sfb_ptr,
                sfb_shape=sfb_shape,
                sfb_order=sfb_order,
                sfd_row_ptr=sfd_row_ptr,
                sfd_row_shape=sfd_row_shape,
                sfd_row_order=sfd_row_order,
                sfd_col_ptr=sfd_col_ptr,
                sfd_col_shape=sfd_col_shape,
                sfd_col_order=sfd_col_order,
                amax_ptr=amax_ptr,
                amax_shape=amax_shape,
                amax_order=amax_order,
                norm_const_ptr=norm_const_ptr,
                norm_const_shape=norm_const_shape,
                norm_const_order=norm_const_order,
                tile_idx_to_expert_idx_ptr=tile_idx_ptr,
                tile_idx_to_expert_idx_shape=tile_idx_shape,
                tile_idx_to_expert_idx_order=tile_idx_order,
                num_non_exiting_tiles_ptr=num_tiles_ptr,
                num_non_exiting_tiles_shape=num_tiles_shape,
                num_non_exiting_tiles_order=num_tiles_order,
                m_split_cumsum_ptr=m_split_cumsum_ptr,
                m_split_cumsum_shape=m_split_cumsum_shape,
                m_split_cumsum_order=m_split_cumsum_order,
                alpha_ptr=alpha_ptr,
                alpha_shape=alpha_shape,
                alpha_order=alpha_order,
                prob_ptr=prob_ptr,
                prob_shape=prob_shape,
                prob_order=prob_order,
                max_active_clusters=max_active_clusters,
                stream=current_stream,
            )
        else:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        alpha_tensor: torch.Tensor,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        m_split_cumsum: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor
        :param b_tensor: Input B tensor (weights)
        :param c_tensor: Intermediate C tensor
        :param d_tensor: Output D tensor
        :param sfa_tensor: Scale factor A
        :param sfb_tensor: Scale factor B
        :param tile_idx_to_expert_idx: Tile to expert mapping
        :param num_non_exiting_tiles: Number of valid tiles
        :param alpha_tensor: Per-group scaling factors
        :param d_col_tensor: Optional column-quantized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Optional probability tensor
        :param m_split_cumsum: Optional m split cumulative sum tensor
        :param current_stream: CUDA stream
        :param skip_compile: If True, use JIT execution without prior compilation
        """
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")

        if not skip_compile:
            self._runtime_error_if(
                self._compiled_kernel is None,
                "Kernel not compiled; call compile() first or use skip_compile=True",
            )

            if self._kernel is BlockScaledContiguousGroupedGemmKernel:
                self._logger.debug("Executing grouped_gemm_swiglu kernel (dlpack)")
                self._compiled_kernel(
                    a=from_dlpack(a_tensor, assumed_align=16),
                    b=from_dlpack(b_tensor, assumed_align=16),
                    c=from_dlpack(c_tensor, assumed_align=16),
                    d=from_dlpack(d_tensor, assumed_align=16),
                    d_col=from_dlpack(d_col_tensor, assumed_align=16) if d_col_tensor is not None else None,
                    sfa=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb=from_dlpack(sfb_tensor, assumed_align=16),
                    sfd_row_tensor=from_dlpack(sfd_row_tensor, assumed_align=16) if sfd_row_tensor is not None else None,
                    sfd_col_tensor=from_dlpack(sfd_col_tensor, assumed_align=16) if sfd_col_tensor is not None else None,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16) if amax_tensor is not None else None,
                    norm_const_tensor=from_dlpack(norm_const_tensor, assumed_align=16) if norm_const_tensor is not None else None,
                    tile_idx_to_expert_idx=from_dlpack(tile_idx_to_expert_idx, assumed_align=16),
                    num_non_exiting_tiles=from_dlpack(num_non_exiting_tiles, assumed_align=16),
                    m_split_cumsum=from_dlpack(m_split_cumsum, assumed_align=16) if m_split_cumsum is not None else None,
                    alpha=from_dlpack(alpha_tensor, assumed_align=16),
                    prob=from_dlpack(prob_tensor, assumed_align=16) if prob_tensor is not None else None,
                    stream=current_stream,
                )
            elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
                self._logger.debug("Executing grouped_gemm_swiglu kernel (no_dlpack)")
                # Create cute pointers manually to avoid DLPack requirements
                a_ptr = self._make_cute_pointer(a_tensor, assumed_align=16)
                b_ptr = self._make_cute_pointer(b_tensor, assumed_align=16)
                c_ptr = self._make_cute_pointer(c_tensor, assumed_align=16)
                d_ptr = self._make_cute_pointer(d_tensor, assumed_align=16)
                d_col_ptr = self._make_cute_pointer(d_col_tensor, assumed_align=16)
                sfa_ptr = self._make_cute_pointer(sfa_tensor, assumed_align=16)
                sfb_ptr = self._make_cute_pointer(sfb_tensor, assumed_align=16)
                sfd_row_ptr = self._make_cute_pointer(sfd_row_tensor, assumed_align=16)
                sfd_col_ptr = self._make_cute_pointer(sfd_col_tensor, assumed_align=16)
                amax_ptr = self._make_cute_pointer(amax_tensor, assumed_align=16)
                norm_const_ptr = self._make_cute_pointer(norm_const_tensor, assumed_align=16)
                tile_idx_ptr = self._make_cute_pointer(tile_idx_to_expert_idx, assumed_align=16)
                num_tiles_ptr = self._make_cute_pointer(num_non_exiting_tiles, assumed_align=16)
                m_split_cumsum_ptr = self._make_cute_pointer(m_split_cumsum, assumed_align=16)
                alpha_ptr = self._make_cute_pointer(alpha_tensor, assumed_align=16)
                prob_ptr = self._make_cute_pointer(prob_tensor, assumed_align=16)

                self._compiled_kernel(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    c_ptr=c_ptr,
                    d_ptr=d_ptr,
                    d_col_ptr=d_col_ptr,
                    sfa_ptr=sfa_ptr,
                    sfb_ptr=sfb_ptr,
                    sfd_row_ptr=sfd_row_ptr,
                    sfd_col_ptr=sfd_col_ptr,
                    amax_ptr=amax_ptr,
                    norm_const_ptr=norm_const_ptr,
                    tile_idx_to_expert_idx_ptr=tile_idx_ptr,
                    num_non_exiting_tiles_ptr=num_tiles_ptr,
                    m_split_cumsum_ptr=m_split_cumsum_ptr,
                    alpha_ptr=alpha_ptr,
                    prob_ptr=prob_ptr,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            generate_sfd = sfd_row_tensor is not None and sfd_col_tensor is not None and norm_const_tensor is not None
            discrete_col_sfd = self.discrete_col_sfd and generate_sfd

            gemm_swiglu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vector_f32=self.vector_f32,
                generate_sfd=generate_sfd,
                discrete_col_sfd=discrete_col_sfd,
            )

            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

            if self._kernel is BlockScaledContiguousGroupedGemmKernel:
                self._logger.debug("JIT executing grouped_gemm_swiglu kernel (dlpack)")
                gemm_swiglu(
                    a=from_dlpack(a_tensor, assumed_align=16),
                    b=from_dlpack(b_tensor, assumed_align=16),
                    c=from_dlpack(c_tensor, assumed_align=16),
                    d=from_dlpack(d_tensor, assumed_align=16),
                    d_col=from_dlpack(d_col_tensor, assumed_align=16) if d_col_tensor is not None else None,
                    sfa=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb=from_dlpack(sfb_tensor, assumed_align=16),
                    sfd_row_tensor=from_dlpack(sfd_row_tensor, assumed_align=16) if sfd_row_tensor is not None else None,
                    sfd_col_tensor=from_dlpack(sfd_col_tensor, assumed_align=16) if sfd_col_tensor is not None else None,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16) if amax_tensor is not None else None,
                    norm_const_tensor=from_dlpack(norm_const_tensor) if norm_const_tensor is not None else None,
                    tile_idx_to_expert_idx=from_dlpack(tile_idx_to_expert_idx, assumed_align=16),
                    num_non_exiting_tiles=from_dlpack(num_non_exiting_tiles, assumed_align=16),
                    m_split_cumsum=from_dlpack(m_split_cumsum, assumed_align=16) if self.m_split_cumsum is not None else None,
                    alpha=from_dlpack(alpha_tensor, assumed_align=16),
                    prob=from_dlpack(prob_tensor, assumed_align=16) if prob_tensor is not None else None,
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
                self._logger.debug("JIT executing grouped_gemm_swiglu kernel (no_dlpack)")
                # Create cute tensor descriptors manually to avoid DLPack requirements
                a_ptr, a_shape, a_order = self._make_cute_tensor_descriptor(a_tensor, name="A")
                b_ptr, b_shape, b_order = self._make_cute_tensor_descriptor(b_tensor, name="B")
                c_ptr, c_shape, c_order = self._make_cute_tensor_descriptor(c_tensor, name="C")
                d_ptr, d_shape, d_order = self._make_cute_tensor_descriptor(d_tensor, name="D")
                d_col_ptr, d_col_shape, d_col_order = self._make_cute_tensor_descriptor(d_col_tensor, name="D_col")
                sfa_ptr, sfa_shape, sfa_order = self._make_cute_tensor_descriptor(sfa_tensor, name="SFA")
                sfb_ptr, sfb_shape, sfb_order = self._make_cute_tensor_descriptor(sfb_tensor, name="SFB")
                sfd_row_ptr, sfd_row_shape, sfd_row_order = self._make_cute_tensor_descriptor(sfd_row_tensor, name="SFD_row")
                sfd_col_ptr, sfd_col_shape, sfd_col_order = self._make_cute_tensor_descriptor(sfd_col_tensor, name="SFD_col")
                amax_ptr, amax_shape, amax_order = self._make_cute_tensor_descriptor(amax_tensor, name="amax")
                norm_const_ptr, norm_const_shape, norm_const_order = self._make_cute_tensor_descriptor(norm_const_tensor, name="norm_const")
                tile_idx_ptr, tile_idx_shape, tile_idx_order = self._make_cute_tensor_descriptor(tile_idx_to_expert_idx, name="tile_idx")
                num_tiles_ptr, num_tiles_shape, num_tiles_order = self._make_cute_tensor_descriptor(num_non_exiting_tiles, name="num_tiles")
                alpha_ptr, alpha_shape, alpha_order = self._make_cute_tensor_descriptor(alpha_tensor, name="alpha")
                prob_ptr, prob_shape, prob_order = self._make_cute_tensor_descriptor(prob_tensor, name="prob")
                m_split_cumsum_ptr, m_split_cumsum_shape, m_split_cumsum_order = self._make_cute_tensor_descriptor(m_split_cumsum, name="m_split_cumsum")

                gemm_swiglu(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_order,
                    d_ptr=d_ptr,
                    d_shape=d_shape,
                    d_order=d_order,
                    d_col_ptr=d_col_ptr,
                    d_col_shape=d_col_shape,
                    d_col_order=d_col_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_order,
                    sfd_row_ptr=sfd_row_ptr,
                    sfd_row_shape=sfd_row_shape,
                    sfd_row_order=sfd_row_order,
                    sfd_col_ptr=sfd_col_ptr,
                    sfd_col_shape=sfd_col_shape,
                    sfd_col_order=sfd_col_order,
                    amax_ptr=amax_ptr,
                    amax_shape=amax_shape,
                    amax_order=amax_order,
                    norm_const_ptr=norm_const_ptr,
                    norm_const_shape=norm_const_shape,
                    norm_const_order=norm_const_order,
                    tile_idx_to_expert_idx_ptr=tile_idx_ptr,
                    tile_idx_to_expert_idx_shape=tile_idx_shape,
                    tile_idx_to_expert_idx_order=tile_idx_order,
                    num_non_exiting_tiles_ptr=num_tiles_ptr,
                    num_non_exiting_tiles_shape=num_tiles_shape,
                    num_non_exiting_tiles_order=num_tiles_order,
                    m_split_cumsum_ptr=m_split_cumsum_ptr,
                    m_split_cumsum_shape=m_split_cumsum_shape,
                    m_split_cumsum_order=m_split_cumsum_order,
                    alpha_ptr=alpha_ptr,
                    alpha_shape=alpha_shape,
                    alpha_order=alpha_order,
                    prob_ptr=prob_ptr,
                    prob_shape=prob_shape,
                    prob_order=prob_order,
                    max_active_clusters=max_active_clusters,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmSwigluSm100Objects = {}


def grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    alpha_tensor: torch.Tensor,
    norm_const_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    m_split_cumsum: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    c_dtype: torch.dtype = torch.bfloat16,
    d_dtype: torch.dtype = torch.bfloat16,
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

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        b_tensor: Weight B tensor (n, k, l)
        sfa_tensor: Scale factor A
        sfb_tensor: Scale factor B
        tile_idx_to_expert_idx: Tile to expert mapping
        num_non_exiting_tiles: Number of valid tiles
        alpha_tensor: Per-group scaling
        norm_const_tensor: Optional normalization constant. Required when using FP8
            input configurations (i.e., when a_tensor.dtype is FP8 and sfa_tensor.dtype is FP8).
            Should be None for FP4/BF16 input configurations.
        prob_tensor: Optional probability tensor for gating
        m_split_cumsum: Optional m split cumulative sum tensor. Required when discrete_col_sfd is True.
        acc_dtype: Accumulator data type
        c_dtype: Intermediate C tensor data type (always bfloat16)
        d_dtype: Output D tensor data type (fp8 when ab is fp8, bf16 when ab is fp4)
        cd_major: CD major dimension (note: only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment
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
    valid_m, k, _ = a_tensor.shape
    n, _, l = b_tensor.shape
    n_out = n // 2  # After SwiGLU

    _logger.debug(f"grouped_gemm_swiglu_wrapper_sm100: Creating output tensors c_tensor, d_tensor, d_col_tensor")

    if cd_major == "n":
        # 1, m, n, permute (1, 2, 0) -> (m, n, 1)
        c_tensor = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=c_dtype, device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
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
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

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
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_swiglu_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        sfa_tensor.shape,
        sfb_tensor.shape,
        sfa_tensor.stride(),
        sfb_tensor.stride(),
        sfa_tensor.dtype,
        sfb_tensor.dtype,
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        m_split_cumsum.shape if m_split_cumsum is not None else None,
        m_split_cumsum.stride() if m_split_cumsum is not None else None,
        m_split_cumsum.dtype if m_split_cumsum is not None else None,
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
    )

    if cache_key in _cache_of_GroupedGemmSwigluSm100Objects:
        _logger.debug("group_gemm_swiglu_wrapper_sm100: Using previously cached GroupedGemmSwigluSm100 object")
        grouped_gemm_swiglu = _cache_of_GroupedGemmSwigluSm100Objects[cache_key]
        grouped_gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            alpha_tensor=alpha_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            m_split_cumsum=m_split_cumsum,
            current_stream=current_stream,
        )
    else:
        _logger.debug("group_gemm_swiglu_wrapper_sm100: No previously cached GroupedGemmSwigluSm100 object found, creating new GroupedGemmSwigluSm100 object")
        grouped_gemm_swiglu = GroupedGemmSwigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            sample_num_non_exiting_tiles=num_non_exiting_tiles,
            sample_alpha=alpha_tensor,
            sample_amax=amax_tensor,
            sample_d_col=d_col_tensor,
            sample_sfd_row=sfd_row_tensor,
            sample_sfd_col=sfd_col_tensor,
            sample_norm_const=norm_const_tensor,
            sample_prob=prob_tensor,
            sample_m_split_cumsum=m_split_cumsum,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            discrete_col_sfd=discrete_col_sfd,
        )

        assert grouped_gemm_swiglu.check_support(), "Unsupported configuration"
        grouped_gemm_swiglu.compile(current_stream=current_stream)
        grouped_gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            alpha_tensor=alpha_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            m_split_cumsum=m_split_cumsum,
            current_stream=current_stream,
        )
        _cache_of_GroupedGemmSwigluSm100Objects[cache_key] = grouped_gemm_swiglu

    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
