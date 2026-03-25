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
Unified API for Grouped GEMM Quant Kernel (SM100+)

This module provides a single API class that supports both dense (contiguous)
and discrete weight modes for grouped block-scaled GEMM with output
quantization in MoE (Mixture of Experts) workloads.
"""

import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, TupleDict, ceil_div, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type

from .grouped_gemm_quant import (
    BlockScaledMoEGroupedGemmQuantKernel,
)
from ..moe_utils import MoEWeightMode
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.runtime import from_dlpack


class GroupedGemmQuantSm100(APIBase):
    """Unified API for grouped GEMM quant operation on SM100+ GPUs.

    This kernel performs block-scaled grouped GEMM with output quantization
    (D = quant(alpha * A @ B)), designed for MoE workloads. It supports both
    dense (contiguous) and discrete (per-expert pointer) weight layouts
    through ``BlockScaledMoEGroupedGemmQuantKernel``.

    Weight mode is auto-detected from the constructor arguments:

    - Dense: provide ``sample_b`` and ``sample_sfb``.
    - Discrete: provide ``num_experts``, ``b_shape``, and ``b_dtype``.
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d: torch.Tensor,
        sample_d_col: torch.Tensor,
        # Dense mode (contiguous) -- provide these:
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        # Internal: C tensor placeholder (kernel compilation requires it)
        sample_c: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ):
        """Initialize the GroupedGemmQuantSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d: Sample D output tensor (valid_m, n, 1)
        :param sample_d_col: Column-quantized D tensor (required for quant kernel)
        :param sample_b: (Dense) Sample B tensor (n, k, l)
        :param sample_sfb: (Dense) Sample scale factor B tensor
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: (Discrete) Data type of B tensors
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization
        :param sample_norm_const: Optional normalization constant
        :param sample_prob: Optional probability tensor for gating
        :param sample_c: Internal C tensor placeholder (kernel requires it for dtype inference)
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Enable discrete col-major scale factor tensor
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        """
        super().__init__()

        self._logger.warning("GroupedGemmQuantSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        # ---- Weight mode auto-detection ----
        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
            if sample_sfb is None:
                raise ValueError("sample_sfb is required when sample_b is provided (dense mode)")
        elif num_experts is not None and sample_b is None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError("Provide either (sample_b, sample_sfb) for dense mode " "or (num_experts, b_shape, b_dtype) for discrete mode, but not both.")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")

        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col")
        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row")
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col")
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const"),
            1,
            "norm_const",
        )
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")

        # C tensor: required by kernel for dtype inference but never written to (generate_c=False).
        # If not provided, derive from D descriptor with bfloat16 dtype.
        if sample_c is not None:
            self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        else:
            self.c_desc = TensorDesc(
                dtype=torch.bfloat16,
                shape=self.d_desc.shape,
                stride=self.d_desc.stride,
                stride_order=self.d_desc.stride_order,
                device=self.d_desc.device,
                name="sample_c",
            )

        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
            self.expert_cnt = self.padded_offsets_desc.shape[0]
        else:
            self._value_error_if(num_experts == 0, "num_experts must be > 0")
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = b_dtype
            self.b_major = b_major
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) " f"must equal num_experts ({self.expert_cnt})",
            )

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
        self.use_dynamic_sched = use_dynamic_sched
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_major = b_major

        self._interpret_uint8_as_fp4x2 = True
        self._kernel = BlockScaledMoEGroupedGemmQuantKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        print(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")
        self._workspace = None
        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        all_none = all(x is None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        all_provided = all(x is not None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        self._value_error_if(
            not (all_none or all_provided),
            "sfd_row_desc, sfd_col_desc, and norm_const_desc must be all None or all not None",
        )
        self.generate_sfd = all_provided
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, discrete_col_sfd will be ignored")
            self.discrete_col_sfd = False

        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self._tensor_shape(self.a_desc, name="sample_a")

        if self.weight_mode == MoEWeightMode.DENSE:
            n, _, l = self._tensor_shape(self.b_desc, name="sample_b")
        else:
            if len(self.b_shape) == 2:
                n, b_k = self.b_shape
            else:
                n, b_k, _ = self.b_shape
            self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")
            l = self.expert_cnt

        _, _, _one = self._tensor_shape(self.c_desc, name="sample_c")
        _, _, _one = self._tensor_shape(self.d_desc, name="sample_d")

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n, 1), "D")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n, 1), "D_col")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
        rest_n = ceil_div(ceil_div(n, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_row_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_n, 1), "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_col_desc, (32, 4, ceil_div(n, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._value_error_if(
            self.prob_desc is None,
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed.",
        )
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.amax_desc, (self.expert_cnt, 1), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            if self._is_fp8(self.a_desc):
                _ = self._check_tensor_stride(
                    self.b_desc,
                    stride=[(k, 1, n * k), (1, n, n * k)],
                    extra_error_msg="For fp8 ab_dtype, B must have k- or n-major layout",
                )
            else:
                _ = self._check_tensor_stride(
                    self.b_desc,
                    stride=[(k, 1, n * k)],
                    extra_error_msg="For fp4 ab_dtype, B must have k-major layout",
                )
        _ = self._check_tensor_stride(
            self.c_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="C must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="D must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_col_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="D_col must have n-major layout",
        )

        self._logger.debug("Checking data types")
        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[
                torch.float4_e2m1fn_x2,
                torch.uint8,
                torch.float8_e5m2,
                torch.float8_e4m3fn,
            ],
            name="A/B",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.b_desc,
                dtype=self.ab_dtype,
                name="B",
                extra_error_msg="B must have the same dtype as A",
            )
        else:
            self._value_error_if(
                self.b_dtype != self.ab_dtype,
                f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
            )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD_row/SFD_col",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.sfb_desc,
                dtype=self.sf_dtype,
                name="SFB",
                extra_error_msg="SFB must have the same dtype as SFA",
            )
        self._check_dtype(
            self.sfd_row_desc,
            dtype=self.sf_dtype,
            name="SFD_row",
            extra_error_msg="SFD_row must have the same dtype as SFA",
        )
        self._check_dtype(
            self.sfd_col_desc,
            dtype=self.sf_dtype,
            name="SFD_col",
            extra_error_msg="SFD_col must have the same dtype as SFA",
        )

        self._value_error_if(
            self.sf_vec_size not in [16, 32],
            f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}",
        )
        self._value_error_if(
            self.sf_dtype in [torch.float8_e4m3fn] and self.sf_vec_size == 32,
            f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )
        self._value_error_if(
            self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16,
            f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )

        self._check_dtype(
            self.acc_dtype,
            dtype=torch.float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )
        self.c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float4_e2m1fn_x2,
            ],
            name="C",
        )

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_desc,
                dtype=[torch.float16, torch.bfloat16, torch.float32],
                name="D",
                extra_error_msg="D must be fp16, bf16, or float32 when ab_dtype is fp4",
            )
        else:
            self.d_dtype = self._check_dtype(
                self.d_desc,
                dtype=[
                    torch.float16,
                    torch.bfloat16,
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                    torch.float4_e2m1fn_x2,
                ],
                name="D",
            )
        self._check_dtype(
            self.d_col_desc,
            dtype=self.d_dtype,
            name="D_col",
            extra_error_msg="D_col must have the same dtype as D",
        )

        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and self.sf_vec_size == 16 and self.d_dtype == torch.float32,
            "Invalid configuration: fp4 ab_dtype, sf_vec_size 16, d_dtype float32 is not supported. Please use sf_vec_size 32 or d_dtype bf16 instead",
        )

        if self.weight_mode == MoEWeightMode.DISCRETE:
            self._value_error_if(
                self.b_major not in ["k", "n"],
                f"b_major must be 'k' or 'n', got {self.b_major}",
            )
            self._value_error_if(
                self._is_fp4x2(self.ab_dtype) and self.b_major != "k",
                "b_major must be 'k' when ab_dtype is fp4",
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
        self._value_error_if(
            self.mma_tiler_mn[1] not in [128, 256],
            f"MMA tiler N must be 128 or 256, got {self.mma_tiler_mn[1]}",
        )
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
        self._value_error_if(
            cluster_tiler_m not in [128, 256],
            f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}",
        )
        self._value_error_if(
            self.m_aligned % self.mma_tiler_mn[0] != 0,
            f"Invalid m_aligned: expected m_aligned to be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0",
        )
        self._value_error_if(
            self.m_aligned != BlockScaledMoEGroupedGemmQuantKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledMoEGroupedGemmQuantKernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
        )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        if self.weight_mode == MoEWeightMode.DENSE:
            b_stride_order_for_check = self.b_desc.stride_order
            b_shape_for_check = (n, k, l)
        else:
            b_stride_order_for_check = (0, 1, 2) if self.b_major == "n" else (1, 0, 2)
            b_shape_for_check = (n, k, 1)

        self._value_error_if(
            not (
                check_contigous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (tensor_m, k, l))
                and check_contigous_16B_alignment(self.ab_dtype, b_stride_order_for_check, b_shape_for_check)
                and check_contigous_16B_alignment(self.d_dtype, self.d_desc.stride_order, (tensor_m, n, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        self._value_error_if(
            self.expert_cnt > 1024,
            f"expert_cnt must be <= 1024, got {self.expert_cnt}",
        )

        self._not_implemented_error_if(
            (self._is_fp8(self.ab_dtype)) and (self.mma_tiler_mn[1] == 128) and (self._is_fp8(self.d_dtype)),
            "Invalid configuration: fp8 ab_dtype and sf_vec_size 32 with mma_tiler_mn[1] == 128 and fp8 d_dtype is not supported. "
            "Please use mma_tiler_mn[1] == 256 instead",
        )
        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and (self.c_dtype not in [torch.float16, torch.bfloat16]),
            f"Invalid configuration: for fp4 ab_dtype, c_dtype must be float16 or bfloat16, got {self.c_dtype}",
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmQuant requires SM100+ compute capability, but found SM{compute_capability} on device {device}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the kernel."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return
        if self.a_desc.shape[0] == 0:
            self._logger.debug("sample valid_m is zero, skipping kernel compilation")
            return

        gemm_quant = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
            generate_c=False,
            enable_bias=False,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        workspace_bytes = gemm_quant.get_workspace_bytes()
        self._workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device="cuda")

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(gemm_quant, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(gemm_quant, max_active_clusters, fake_stream)

        self._logger.debug("Kernel compiled successfully")

    def _compile_dense(self, gemm_quant, max_active_clusters, fake_stream) -> None:
        """Compile for dense (contiguous) weight mode."""
        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        self._logger.debug("Compiling grouped_gemm_quant kernel")
        _compiled_kernel = cute.compile(
            gemm_quant,
            a=self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16),
            b=self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16),
            sfb=self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16),
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=OperandMajorMode.K,
            workspace_ptr=fake_workspace_ptr,
            c=self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16),
            d=self._make_fake_cute_tensor_from_desc(self.d_desc, assumed_align=16),
            d_col=self._make_fake_cute_tensor_from_desc(self.d_col_desc, assumed_align=16),
            sfa=self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16),
            sfd_row_tensor=self._make_fake_cute_tensor_from_desc(self.sfd_row_desc, assumed_align=16),
            sfd_col_tensor=self._make_fake_cute_tensor_from_desc(self.sfd_col_desc, assumed_align=16),
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            bias=None,
            prob=self._make_fake_cute_tensor_from_desc(self.prob_desc, assumed_align=16),
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            prob_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            _compiled_kernel(
                a_tensor,
                b_tensor,
                sfb_tensor,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int64(0),
                cached_workspace_ptr,
                c_tensor,
                d_tensor,
                d_col_tensor,
                sfa_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                None,
                prob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, gemm_quant, max_active_clusters, fake_stream) -> None:
        """Compile for discrete (per-expert pointer) weight mode."""
        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape

        b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
        b_stride_size = k if self.b_major == "k" else n

        ab_cutlass_dtype = _convert_to_cutlass_data_type(self.a_desc.dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2)
        align = 32 if ab_cutlass_dtype.width == 4 else 16

        a_tensor = self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=align)
        if a_tensor is not None:
            a_tensor.mark_layout_dynamic(leading_dim=1)
        c_tensor = self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16)
        if c_tensor is not None:
            c_tensor.mark_layout_dynamic(leading_dim=1)
        d_tensor = self._make_fake_cute_tensor_from_desc(self.d_desc, assumed_align=16)
        if d_tensor is not None:
            d_tensor.mark_layout_dynamic(leading_dim=1)
        d_col_tensor = self._make_fake_cute_tensor_from_desc(self.d_col_desc, assumed_align=16)
        if d_col_tensor is not None:
            d_col_tensor.mark_layout_dynamic(leading_dim=1)
        sfa_tensor = self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16)
        if sfa_tensor is not None:
            sfa_tensor.mark_layout_dynamic(leading_dim=3)
        sfd_row_tensor = self._make_fake_cute_tensor_from_desc(self.sfd_row_desc, assumed_align=16)
        if sfd_row_tensor is not None:
            sfd_row_tensor.mark_layout_dynamic(leading_dim=3)
        sfd_col_tensor = self._make_fake_cute_tensor_from_desc(self.sfd_col_desc, assumed_align=16)
        if sfd_col_tensor is not None:
            sfd_col_tensor.mark_layout_dynamic(leading_dim=3)
        amax_tensor = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        norm_const_tensor_cute = self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16)
        padded_offsets_tensor = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        alpha_tensor = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        prob_tensor = self._make_fake_cute_tensor_from_desc(self.prob_desc, assumed_align=16)
        if prob_tensor is not None:
            prob_tensor.mark_layout_dynamic(leading_dim=1)

        b_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        sfb_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        b_ptrs_cute = from_dlpack(b_ptrs_placeholder, assumed_align=8).iterator
        sfb_ptrs_cute = from_dlpack(sfb_ptrs_placeholder, assumed_align=8).iterator
        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        self._logger.debug("Compiling discrete grouped_gemm_quant kernel")
        _compiled_kernel = cute.compile(
            gemm_quant,
            a_tensor,
            b_ptrs_cute,
            sfb_ptrs_cute,
            cutlass.Int32(n),
            cutlass.Int32(k),
            cutlass.Int64(b_stride_size),
            b_major_mode,
            workspace_ptr_cute,
            c_tensor,
            d_tensor,
            d_col_tensor,
            sfa_tensor,
            sfd_row_tensor,
            sfd_col_tensor,
            amax_tensor,
            norm_const_tensor_cute,
            padded_offsets_tensor,
            alpha_tensor,
            None,
            prob_tensor,
            max_active_clusters,
            fake_stream,
            lambda x: x,
            options="--enable-tvm-ffi",
        )

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        cached_n = cutlass.Int32(n)
        cached_k = cutlass.Int32(k)
        cached_b_stride = cutlass.Int64(b_stride_size)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_ptrs_device: torch.Tensor,
            sfb_ptrs_device: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            prob_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            b_ptrs_addr = int(b_ptrs_device.data_ptr())
            sfb_ptrs_addr = int(sfb_ptrs_device.data_ptr())
            _compiled_kernel(
                a_tensor,
                b_ptrs_addr,
                sfb_ptrs_addr,
                cached_n,
                cached_k,
                cached_b_stride,
                cached_workspace_ptr,
                c_tensor,
                d_tensor,
                d_col_tensor,
                sfa_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                None,
                prob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        c_tensor: Optional[torch.Tensor] = None,
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
        :param sfa_tensor: Scale factor A
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param d_tensor: Output D tensor
        :param b_tensor: (Dense) Input B tensor (weights)
        :param sfb_tensor: (Dense) Scale factor B
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param c_tensor: Optional C tensor placeholder (kernel requires it but never writes to it;
            a minimal dummy is created automatically if not provided)
        :param d_col_tensor: Optional column-quantized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Probability tensor for per-row gating. Required.
        :param current_stream: CUDA stream
        """
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if a_tensor.shape[0] == 0:
            self._logger.debug("execute: valid_m is zero, skipping kernel execution")
            return
        self._runtime_error_if(
            self._compiled_kernel is None,
            "Kernel not compiled; call compile() first",
        )

        if c_tensor is None:
            c_tensor = torch.empty_strided(
                self.c_desc.shape,
                self.c_desc.stride,
                dtype=self.c_desc.dtype,
                device=d_tensor.device,
            )
        self._value_error_if(
            prob_tensor is None,
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed.",
        )

        self._logger.debug("Executing grouped_gemm_quant kernel")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                c_tensor=c_tensor,
                d_tensor=d_tensor,
                d_col_tensor=d_col_tensor,
                sfa_tensor=sfa_tensor,
                sfb_tensor=sfb_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                stream=current_stream,
            )
        else:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_ptrs_device=b_ptrs,
                sfb_ptrs_device=sfb_ptrs,
                c_tensor=c_tensor,
                d_tensor=d_tensor,
                d_col_tensor=d_col_tensor,
                sfa_tensor=sfa_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                prob_tensor=prob_tensor,
                stream=current_stream,
            )

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmQuantSm100Objects = {}


def grouped_gemm_quant_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    norm_const_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
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
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM Quant operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group scaling
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
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
        c_dtype: Internal C tensor data type (not user-visible)
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Enable discrete col-major scale factor tensor
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **d_tensor** (torch.Tensor): Final output tensor
            - **d_col_tensor** (torch.Tensor): Column-wise output tensor
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (for quantization)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D (FP8 only)
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D (FP8 only)

            Example usage::

                # Dictionary-style access
                result = grouped_gemm_quant_wrapper_sm100(...)
                d = result["d_tensor"]

                # Tuple unpacking
                d, d_col, amax, sfd_row, sfd_col = grouped_gemm_quant_wrapper_sm100(...)

                # Integer indexing
                d = result[0]  # d_tensor
    """
    from cudnn.discrete_grouped_gemm.discrete_kernel_utils import _require_pointer_tensor

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
    else:
        weight_mode = MoEWeightMode.DISCRETE
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        num_experts = b_ptrs.shape[0]
        _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
        b_shape = (n, k_logical)
        n_out = n
        l = num_experts

    _logger.debug("grouped_gemm_quant_wrapper_sm100: Creating output tensors d_tensor, d_col_tensor")

    if cd_major == "n":
        c_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=c_dtype, device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    is_fp8_input_config = a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [
        torch.float8_e8m0fnu,
        torch.float8_e4m3fn,
    ]
    is_fp8_output_config = d_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    ]

    if is_fp8_input_config and is_fp8_output_config and norm_const_tensor is None:
        raise ValueError(
            "norm_const_tensor is required when FP8 inputs are used with FP8 output "
            "(a_tensor is FP8 and sfa_tensor is FP8 and d_dtype is FP8). "
            "Pass a tensor with shape (1,), e.g. torch.tensor([0.01], dtype=torch.float32, device=a_tensor.device)."
        )

    if not is_fp8_output_config:
        norm_const_tensor = None

    if is_fp8_input_config and is_fp8_output_config:
        _logger.debug("grouped_gemm_quant_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

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

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_quant_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if prob_tensor is None:
        raise ValueError(
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed."
        )

    if valid_m == 0:
        _logger.debug("grouped_gemm_quant_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
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

    if is_dense:
        cache_key = (
            weight_mode,
            tuple(a_tensor.shape),
            tuple(b_tensor.shape),
            a_tensor.dtype,
            b_tensor.dtype,
            tuple(a_tensor.stride()),
            tuple(b_tensor.stride()),
            tuple(c_tensor.shape),
            tuple(c_tensor.stride()),
            c_tensor.dtype,
            *tensor_signature(sfa_tensor),
            *tensor_signature(sfb_tensor),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *tensor_signature(prob_tensor),
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
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
            use_dynamic_sched,
        )
    else:
        cache_key = (
            weight_mode,
            tuple(a_tensor.shape),
            tuple(a_tensor.stride()),
            a_tensor.dtype,
            b_shape,
            b_dtype,
            tuple(c_tensor.shape),
            tuple(c_tensor.stride()),
            c_tensor.dtype,
            *tensor_signature(sfa_tensor),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *tensor_signature(prob_tensor),
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
            c_dtype,
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
        )

    if cache_key in _cache_of_GroupedGemmQuantSm100Objects:
        _logger.debug("grouped_gemm_quant_wrapper_sm100: Using previously cached GroupedGemmQuantSm100 object")
        grouped_gemm_quant = _cache_of_GroupedGemmQuantSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_quant_wrapper_sm100: No previously cached object found, creating new GroupedGemmQuantSm100 object")
        if is_dense:
            grouped_gemm_quant = GroupedGemmQuantSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                sample_c=c_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
            )
        else:
            grouped_gemm_quant = GroupedGemmQuantSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d=d_tensor,
                sample_d_col=d_col_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                sample_c=c_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                b_major=b_major,
            )

        assert grouped_gemm_quant.check_support(), "Unsupported configuration"
        grouped_gemm_quant.compile()
        _cache_of_GroupedGemmQuantSm100Objects[cache_key] = grouped_gemm_quant

    if is_dense:
        grouped_gemm_quant.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            d_tensor=d_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
    else:
        grouped_gemm_quant.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            d_tensor=d_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            c_tensor=c_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
