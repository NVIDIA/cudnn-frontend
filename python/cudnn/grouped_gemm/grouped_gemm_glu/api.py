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
Unified API for Grouped GEMM GLU Forward Kernel (SM100+)

This module provides a single API class that supports both contiguous (dense)
and discrete weight modes for block-scaled grouped GEMM with GLU activation
(SwiGLU / GeGLU) in MoE (Mixture of Experts) workloads.

Dense mode
    All expert weights are packed contiguously in a 3-D tensor (N, K, L).
    Callers supply ``sample_b`` and ``sample_sfb``.

Discrete mode
    Each expert has its own memory allocation.  Callers supply
    ``num_experts``, ``b_shape``, ``b_dtype``, and per-expert pointer arrays
    at execution time.
"""

from .moe_blockscaled_grouped_gemm_glu import BlockScaledMoEGroupedGemmGluKernel
from .continugous_blockscaled_grouped_gemm_glu_quant_bias_fusion import BlockScaledMoEGroupedGemmGluBiasKernel
from ..moe_utils import MoEWeightMode
from cuda.bindings import driver as cuda
import logging
import os
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2


class GroupedGemmGluSm100(APIBase):
    """Unified API for grouped GEMM GLU forward operation on SM100+ GPUs.

    This kernel performs block-scaled grouped GEMM with GLU activation
    (SwiGLU or GeGLU), designed for MoE workloads.  It supports both dense
    (contiguous) and discrete (per-expert pointer) weight layouts through
    the ``BlockScaledMoEGroupedGemmGluKernel``.

    Weight mode is auto-detected from the constructor arguments:

    - **Dense**: provide ``sample_b`` and ``sample_sfb``.
    - **Discrete**: provide ``num_experts``, ``b_shape``, and ``b_dtype``.

    Example::

        # Dense mode
        api = GroupedGemmGluSm100(
            sample_a=a, sample_b=b, sample_c=c, sample_d=d,
            sample_sfa=sfa, sample_sfb=sfb,
            sample_padded_offsets=offsets, sample_alpha=alpha,
            sample_d_col=d_col,
        )

        # Discrete mode
        api = GroupedGemmGluSm100(
            sample_a=a, num_experts=8, b_shape=(n, k), b_dtype=torch.uint8,
            sample_c=c, sample_d=d, sample_sfa=sfa,
            sample_padded_offsets=offsets, sample_alpha=alpha,
            sample_d_col=d_col,
        )

        api.check_support()
        api.compile()
        api.execute(...)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d_col: torch.Tensor,
        # Dense mode (contiguous) -- provide these. sample_bias is optional:
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
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
        # Configuration
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        act_func: str = "swiglu",
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ):
        """Initialize the GroupedGemmGluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_c: Sample C tensor for intermediate storage
        :param sample_d: Sample D output tensor (valid_m, n/2, 1) after GLU
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d_col: Column-quantized D tensor
        :param sample_b: (Dense) Sample B tensor (n, k, l)
        :param sample_sfb: (Dense) Sample scale factor B tensor
        :param sample_bias: (Dense, optional) Bias tensor with shape (n, l) and stride (1, n)
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: (Discrete) Data type of B tensors
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
        :param discrete_col_sfd: Generate discrete col-major scale factor tensor
        :param act_func: Activation function, one of "swiglu" or "geglu"
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        """
        super().__init__()

        self._logger.warning("GroupedGemmGluSm100 is an experimental API")
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

        # ---- Common tensor descriptors ----
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")

        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row")
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col")
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const"),
            1,
            "norm_const",
        )
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")

        # ---- Mode-specific state ----
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
            self.expert_cnt = self.padded_offsets_desc.shape[0]
        else:
            self._value_error_if(sample_bias is not None, "bias_tensor is only supported in dense mode")
            self._value_error_if(num_experts == 0, "num_experts must be > 0")
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = b_dtype
            self.b_major = b_major
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) " f"must equal num_experts ({self.expert_cnt})",
            )

        # ---- Configuration ----
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
        self.act_func = act_func
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_major = b_major  # stored for both modes

        self.use_dynamic_sched = use_dynamic_sched

        self._interpret_uint8_as_fp4x2 = True
        self._uses_bias_kernel = self.weight_mode == MoEWeightMode.DENSE and self.bias_desc is not None
        self._kernel = BlockScaledMoEGroupedGemmGluBiasKernel if self._uses_bias_kernel else BlockScaledMoEGroupedGemmGluKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        print(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")

        self._workspace = None

        self._logger.debug("__init__ completed")

    # --------------------------------------------------------------------- #
    #  check_support
    # --------------------------------------------------------------------- #

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        # ---- SFD group validation ----
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

        # ---- Shapes and strides ----
        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self._tensor_shape(self.a_desc, name="sample_a")

        if self.weight_mode == MoEWeightMode.DENSE:
            n, _, l = self._tensor_shape(self.b_desc, name="sample_b")
        else:
            # Discrete: extract n, k from b_shape
            if len(self.b_shape) == 2:
                n, b_k = self.b_shape
            else:
                n, b_k, _ = self.b_shape
            self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")
            l = self.expert_cnt  # for shape checks that use l

        _, n_2, _one = self._tensor_shape(self.d_desc, name="sample_d")

        self._value_error_if(
            n % 64 != 0,
            f"N must be divisible by 64 for GLU (two consecutive 32-column blocks), got N={n}",
        )

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n // 2, 1), "D")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n // 2, 1), "D_col")
        self._check_tensor_shape(self.bias_desc, (n, l), "bias")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")

        rest_n2 = ceil_div(ceil_div(n // 2, self.sf_vec_size), 4)
        self._check_tensor_shape(
            self.sfd_row_desc,
            (32, 4, ceil_div(tensor_m, 128), 4, rest_n2, 1),
            "SFD_row",
        )
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_col_desc, (32, 4, ceil_div(n // 2, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.amax_desc, (self.expert_cnt, 1), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        # Strides
        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            _ = self._check_tensor_stride(
                self.b_desc,
                stride=[(k, 1, n * k)],
                extra_error_msg="B must have k-major layout",
            )
        _ = self._check_tensor_stride(
            self.c_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="C must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_desc,
            stride=[(n_2, 1, tensor_m * n_2)],
            extra_error_msg="D must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_col_desc,
            stride=[(n_2, 1, tensor_m * n_2)],
            extra_error_msg="D_col must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.bias_desc,
            stride=[(1, n)],
        )

        # ---- Data types ----
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
            self._check_dtype(
                self.bias_desc,
                dtype=[torch.bfloat16, torch.float16, torch.float32],
                name="bias",
            )
        else:
            self._value_error_if(
                self.b_dtype != self.ab_dtype,
                f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
            )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD",
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
            self.bias_desc is None and self._is_fp4x2(self.ab_dtype) and self.sf_vec_size == 16 and self.d_dtype == torch.float32,
            "Invalid configuration: fp4 ab_dtype, sf_vec_size 16, d_dtype float32 is not supported. " "Please use sf_vec_size 32 or d_dtype bf16 instead",
        )

        # ---- Activation function validation (both modes) ----
        self._value_error_if(
            self.act_func not in ["swiglu", "geglu"],
            f"act_func must be 'swiglu' or 'geglu', got {self.act_func}",
        )

        # ---- Discrete-mode-specific validation ----
        if self.weight_mode == MoEWeightMode.DISCRETE:
            self._value_error_if(
                self.b_major not in ["k", "n"],
                f"b_major must be 'k' or 'n', got {self.b_major}",
            )
            self._value_error_if(
                self._is_fp4x2(self.ab_dtype) and self.b_major != "k",
                "b_major must be 'k' when ab_dtype is fp4",
            )

        # ---- MMA tile / cluster shape ----
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
            f"Invalid cluster shape: expected values to be powers of 2 and product <= 16, got {self.cluster_shape_mn}",
        )
        cluster_tiler_m = (self.cluster_shape_mn[0] // (2 if self.use_2cta_instrs else 1)) * self.mma_tiler_mn[0]
        self._value_error_if(
            cluster_tiler_m not in [128, 256],
            f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}",
        )
        self._value_error_if(
            self.m_aligned % self.mma_tiler_mn[0] != 0,
            f"m_aligned must be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0",
        )
        self._value_error_if(
            self.m_aligned != BlockScaledMoEGroupedGemmGluKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledMoEGroupedGemmGluKernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
        )

        # ---- Tensor alignment ----
        self._logger.debug("Checking tensor alignment")

        def check_contiguous_16B_alignment(dtype, stride_order, tensor_shape):
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
                check_contiguous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (tensor_m, k, l))
                and check_contiguous_16B_alignment(self.ab_dtype, b_stride_order_for_check, b_shape_for_check)
                and check_contiguous_16B_alignment(self.d_dtype, self.d_desc.stride_order, (tensor_m, n_2, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        # ---- Expert count limit ----
        self._value_error_if(
            self.expert_cnt > 1024,
            f"expert_cnt must be <= 1024, got {self.expert_cnt}",
        )

        # ---- Disabled configurations ----
        self._not_implemented_error_if(
            (self._is_fp8(self.ab_dtype)) and (self.mma_tiler_mn[1] == 128) and (self._is_fp8(self.d_dtype)),
            "Invalid configuration: fp8 ab_dtype with mma_tiler_mn[1] == 128 and fp8 d_dtype is not supported. " "Please use mma_tiler_mn[1] == 256 instead",
        )
        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and (self.c_dtype not in [torch.float16, torch.bfloat16]),
            f"Invalid configuration: for fp4 ab_dtype, c_dtype must be float16 or bfloat16, got {self.c_dtype}",
        )
        self._not_implemented_error_if(
            self.bias_desc is not None and self.mma_tiler_mn[1] != 256,
            "Dense bias fusion currently requires mma_tiler_mn[1] == 256",
        )

        # ---- SM100+ check ----
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmGlu requires SM100+ compute capability, " f"but found SM{compute_capability} on device {device}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    # --------------------------------------------------------------------- #
    #  compile
    # --------------------------------------------------------------------- #

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

        # ---- Instantiate the unified kernel ----
        if self._uses_bias_kernel:
            gemm_glu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vectorized_f32=self.vector_f32,
                generate_sfd=self.generate_sfd,
                discrete_col_sfd=self.discrete_col_sfd,
                expert_cnt=self.expert_cnt,
                act_func=self.act_func,
                enable_bias=True,
                use_dynamic_sched=self.use_dynamic_sched,
            )
        else:
            gemm_glu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vectorized_f32=self.vector_f32,
                generate_sfd=self.generate_sfd,
                discrete_col_sfd=self.discrete_col_sfd,
                expert_cnt=self.expert_cnt,
                weight_mode=self.weight_mode,
                act_func=self.act_func,
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

        # ---- Allocate workspace ----
        workspace_bytes = gemm_glu.get_workspace_bytes()
        self._workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device="cuda")

        if self.weight_mode == MoEWeightMode.DENSE:
            if self._uses_bias_kernel:
                self._compile_dense_bias_fused(gemm_glu, max_active_clusters, fake_stream)
            else:
                self._compile_dense(gemm_glu, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(gemm_glu, max_active_clusters, fake_stream)

        self._logger.debug("Kernel compiled successfully")

    # -- Dense compile path ------------------------------------------------- #

    def _compile_dense(self, gemm_glu, max_active_clusters, fake_stream) -> None:
        """Compile for dense (contiguous) weight mode."""
        use_full_dynamic = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL") is not None

        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        if not use_full_dynamic:
            valid_m = cute.sym_int(divisibility=256)

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, *self.a_desc.shape[1:]),
                stride_order=self.a_desc.stride_order,
            )
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, *self.c_desc.shape[1:]),
                stride_order=self.c_desc.stride_order,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, *self.d_desc.shape[1:]),
                stride_order=self.d_desc.stride_order,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, *self.d_col_desc.shape[1:]),
                stride_order=self.d_col_desc.stride_order,
            )

            tensor_m_128 = cute.sym_int()
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], 1),
                stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_tensor_m_128),
            )

            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)

            prob_cute_fake = None
            if self.prob_desc is not None:
                prob_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, 1, 1),
                    stride_order=self.prob_desc.stride_order,
                )

            sfd_row_fake = None
            sfd_col_fake = None
            if self.sfd_row_desc is not None:
                stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfd_row_desc.shape[4], 1),
                    stride=(16, 4, self.sfd_row_desc.stride[2], 1, 512, stride_sfd_m),
                )
            if self.sfd_col_desc is not None:
                rest_m = cute.sym_int(divisibility=1)
                stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
                stride_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, self.sfd_col_desc.shape[2], 4, rest_m, 1),
                    stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                )
        else:
            valid_m = cute.sym_int(divisibility=256)
            n_sym = cute.sym_int()
            n_2_sym = cute.sym_int()
            k_sym = cute.sym_int()
            l_sym = cute.sym_int()

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, k_sym, 1),
                stride_order=self.a_desc.stride_order,
                dynamic_mode=self.a_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            b_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=(n_sym, k_sym, l_sym),
                stride_order=self.b_desc.stride_order,
                dynamic_mode=self.b_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.c_desc.stride_order,
                dynamic_mode=self.c_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.c_desc.dtype) else 16,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.d_desc.stride_order,
                dynamic_mode=self.d_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_desc.dtype) else 16,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.d_col_desc.stride_order,
                dynamic_mode=self.d_col_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_col_desc.dtype) else 16,
            )

            tensor_m_128 = cute.sym_int()
            rest_k = cute.sym_int()
            stride_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, rest_k, 1),
                stride=(16, 4, stride_rest_k, 1, 512, stride_tensor_m_128),
            )

            tensor_n_128 = cute.sym_int()
            stride_sfb_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_sfb_tensor_n_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfb_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfb_desc.dtype,
                shape=(32, 4, tensor_n_128, 4, rest_k, l_sym),
                stride=(16, 4, stride_sfb_tensor_n_128, 1, 512, stride_sfb_rest_k),
            )

            prob_cute_fake = None
            if self.prob_desc is not None:
                prob_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, 1, 1),
                    stride_order=self.prob_desc.stride_order,
                )

            sfd_row_fake = None
            sfd_col_fake = None
            if self.sfd_row_desc is not None:
                rest_n2 = cute.sym_int()
                stride_sfd_rest_n2 = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_rest_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, rest_n2, 1),
                    stride=(16, 4, stride_sfd_rest_n2, 1, 512, stride_sfd_rest_tensor_m_128),
                )
            if self.sfd_col_desc is not None:
                tensor_n2_128 = cute.sym_int()
                rest_m_dyn = cute.sym_int()
                stride_sfd_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_n2 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, tensor_n2_128, 4, rest_m_dyn, 1),
                    stride=(16, 4, stride_sfd_rest_m, 1, 512, stride_sfd_n2),
                )

        # Compile with keyword args (dense mode uses the unified __call__ positional order)
        _compiled_kernel = cute.compile(
            gemm_glu,
            a=a_cute_fake,
            b=b_cute_fake,
            sfb=sfb_cute_fake,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=OperandMajorMode.K,
            workspace_ptr=fake_workspace_ptr,
            c=c_cute_fake,
            d=d_cute_fake,
            d_col=d_col_cute_fake,
            sfa=sfa_cute_fake,
            sfd_row_tensor=sfd_row_fake,
            sfd_col_tensor=sfd_col_fake,
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            prob=prob_cute_fake,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            epilogue_op=lambda x: x,
            linear_offset=cutlass.Float32(0.0),
            options="--enable-tvm-ffi",
        )

        # Cache workspace pointer for the tensor_api closure
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
                prob_tensor,
                stream,
                cutlass.Float32(0.0),  # linear_offset (epilogue_op is Constexpr, baked in)
            )

        self._compiled_kernel = tensor_api

    def _compile_dense_bias_fused(self, gemm_glu, max_active_clusters, fake_stream) -> None:
        """Compile the dense fused-bias GLU kernel."""
        use_full_dynamic = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL") is not None

        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        if not use_full_dynamic:
            valid_m = cute.sym_int(divisibility=256)

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, *self.a_desc.shape[1:]),
                stride_order=self.a_desc.stride_order,
            )
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, *self.c_desc.shape[1:]),
                stride_order=self.c_desc.stride_order,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, *self.d_desc.shape[1:]),
                stride_order=self.d_desc.stride_order,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, *self.d_col_desc.shape[1:]),
                stride_order=self.d_col_desc.stride_order,
            )

            tensor_m_128 = cute.sym_int()
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], 1),
                stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_tensor_m_128),
            )

            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
            prob_cute_fake = None
            if self.prob_desc is not None:
                prob_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, 1, 1),
                    stride_order=self.prob_desc.stride_order,
                )

            sfd_row_fake = None
            sfd_col_fake = None
            if self.sfd_row_desc is not None:
                stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfd_row_desc.shape[4], 1),
                    stride=(16, 4, self.sfd_row_desc.stride[2], 1, 512, stride_sfd_m),
                )
            if self.sfd_col_desc is not None:
                rest_m = cute.sym_int(divisibility=1)
                stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
                stride_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, self.sfd_col_desc.shape[2], 4, rest_m, 1),
                    stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                )

            bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)
        else:
            valid_m = cute.sym_int(divisibility=256)
            n_sym = cute.sym_int()
            n_2_sym = cute.sym_int()
            k_sym = cute.sym_int()
            l_sym = cute.sym_int()

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, k_sym, 1),
                stride_order=self.a_desc.stride_order,
                dynamic_mode=self.a_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            b_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=(n_sym, k_sym, l_sym),
                stride_order=self.b_desc.stride_order,
                dynamic_mode=self.b_desc.stride_order[0],
                divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
            )
            c_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.c_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.c_desc.stride_order,
                dynamic_mode=self.c_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.c_desc.dtype) else 16,
            )
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.d_desc.stride_order,
                dynamic_mode=self.d_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_desc.dtype) else 16,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, n_2_sym, 1),
                stride_order=self.d_col_desc.stride_order,
                dynamic_mode=self.d_col_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_col_desc.dtype) else 16,
            )

            tensor_m_128 = cute.sym_int()
            rest_k = cute.sym_int()
            stride_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, rest_k, 1),
                stride=(16, 4, stride_rest_k, 1, 512, stride_tensor_m_128),
            )

            tensor_n_128 = cute.sym_int()
            stride_sfb_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_sfb_tensor_n_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfb_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfb_desc.dtype,
                shape=(32, 4, tensor_n_128, 4, rest_k, l_sym),
                stride=(16, 4, stride_sfb_tensor_n_128, 1, 512, stride_sfb_rest_k),
            )

            prob_cute_fake = None
            if self.prob_desc is not None:
                prob_cute_fake = self._make_fake_cute_compact_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, 1, 1),
                    stride_order=self.prob_desc.stride_order,
                )

            sfd_row_fake = None
            sfd_col_fake = None
            if self.sfd_row_desc is not None:
                rest_n2 = cute.sym_int()
                stride_sfd_rest_n2 = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_rest_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, rest_n2, 1),
                    stride=(16, 4, stride_sfd_rest_n2, 1, 512, stride_sfd_rest_tensor_m_128),
                )
            if self.sfd_col_desc is not None:
                tensor_n2_128 = cute.sym_int()
                rest_m_dyn = cute.sym_int()
                stride_sfd_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_n2 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, tensor_n2_128, 4, rest_m_dyn, 1),
                    stride=(16, 4, stride_sfd_rest_m, 1, 512, stride_sfd_n2),
                )

            bias_cute_fake = self._make_fake_cute_tensor(
                dtype=self.bias_desc.dtype,
                shape=(n_sym, l_sym),
                stride=(1, n_sym),
            )

        _compiled_kernel = cute.compile(
            gemm_glu,
            a=a_cute_fake,
            b=b_cute_fake,
            c=c_cute_fake,
            d=d_cute_fake,
            d_col=d_col_cute_fake,
            sfa=sfa_cute_fake,
            sfb=sfb_cute_fake,
            sfd_row_tensor=sfd_row_fake,
            sfd_col_tensor=sfd_col_fake,
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            bias=bias_cute_fake,
            prob=prob_cute_fake,
            workspace_ptr=fake_workspace_ptr,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            epilogue_op=lambda x: x,
            linear_offset=cutlass.Float32(0.0),
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
            bias_tensor: torch.Tensor,
            prob_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            _compiled_kernel(
                a_tensor,
                b_tensor,
                c_tensor,
                d_tensor,
                d_col_tensor,
                sfa_tensor,
                sfb_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                bias_tensor,
                prob_tensor,
                cached_workspace_ptr,
                stream,
                cutlass.Float32(0.0),
            )

        self._compiled_kernel = tensor_api

    # -- Discrete compile path ---------------------------------------------- #

    def _compile_discrete(self, gemm_glu, max_active_clusters, fake_stream) -> None:
        """Compile for discrete (per-expert pointer) weight mode."""
        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape

        b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
        if self.b_major == "k":
            b_stride_size = k
        else:
            b_stride_size = n

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

        # Compile-time pointer placeholders
        b_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        sfb_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        b_ptrs_cute = from_dlpack(b_ptrs_placeholder, assumed_align=8).iterator
        sfb_ptrs_cute = from_dlpack(sfb_ptrs_placeholder, assumed_align=8).iterator

        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        self._logger.debug("Compiling discrete grouped GEMM GLU kernel")
        _compiled_kernel = cute.compile(
            gemm_glu,
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
            prob_tensor,
            max_active_clusters,
            fake_stream,
            lambda x: x,  # epilogue_op (Constexpr, baked in)
            cutlass.Float32(0.0),  # linear_offset
            options="--enable-tvm-ffi",
        )

        self._n = n
        self._k = k
        self._b_stride_size = b_stride_size

        # Cache constant values for execute() closure
        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        cached_n = cutlass.Int32(self._n)
        cached_k = cutlass.Int32(self._k)
        cached_b_stride = cutlass.Int64(self._b_stride_size)

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
                prob_tensor,
                stream,
                cutlass.Float32(0.0),  # linear_offset (epilogue_op is Constexpr, baked in)
            )

        self._compiled_kernel = tensor_api

    # --------------------------------------------------------------------- #
    #  execute
    # --------------------------------------------------------------------- #

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        # Optional:
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        For dense mode, supply ``b_tensor`` and ``sfb_tensor``.
        For discrete mode, supply ``b_ptrs`` and ``sfb_ptrs``.

        :param a_tensor: Input A tensor
        :param c_tensor: Intermediate C tensor
        :param d_tensor: Output D tensor
        :param sfa_tensor: Scale factor A
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param b_tensor: (Dense) Input B tensor (weights)
        :param sfb_tensor: (Dense) Scale factor B
        :param bias_tensor: (Dense, optional) Bias tensor with shape (n, l) and stride (1, n)
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param d_col_tensor: Optional column-quantized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Optional probability tensor
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

        self._logger.debug("Executing grouped GEMM GLU kernel")
        if self.weight_mode == MoEWeightMode.DENSE:
            if self._uses_bias_kernel:
                self._value_error_if(
                    bias_tensor is None,
                    "bias_tensor must be provided at execute() when the API was compiled with sample_bias",
                )
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
                    bias_tensor=bias_tensor,
                    prob_tensor=prob_tensor,
                    stream=current_stream,
                )
            else:
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


# --------------------------------------------------------------------------- #
#  Convenience wrapper with caching
# --------------------------------------------------------------------------- #

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmGluSm100Objects = {}


def grouped_gemm_glu_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    # Dense mode. bias_tensor is optional:
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    # Discrete mode:
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    # Common:
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
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM GLU forward operation.

    Auto-detects dense vs. discrete mode based on which weight arguments
    are provided.

    Dense mode: provide ``b_tensor`` and ``sfb_tensor``.
    Discrete mode: provide ``b_ptrs``, ``sfb_ptrs``, ``n``, and ``b_dtype``.

    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        sfa_tensor: Scale factor A
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group scaling
        b_tensor: (Dense) Weight B tensor (n, k, l)
        sfb_tensor: (Dense) Scale factor B
        bias_tensor: (Dense, optional) Bias tensor with shape (n, l) and stride (1, n)
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        n: (Discrete) B weight N dimension (full N before GLU split)
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension ("k" or "n")
        norm_const_tensor: Optional normalization constant
        prob_tensor: Optional probability tensor for gating
        acc_dtype: Accumulator data type
        c_dtype: Intermediate C tensor data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (only "n" supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Generate discrete col-major scale factor tensor
        act_func: Activation function ("swiglu" or "geglu")
        use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: c_tensor, d_tensor, d_col_tensor, amax_tensor,
            sfd_row_tensor, sfd_col_tensor
    """
    from cudnn.discrete_grouped_gemm.discrete_kernel_utils import _require_pointer_tensor

    # ---- Auto-detect weight mode ----
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    valid_m, k_physical, _ = a_tensor.shape

    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_full, _, l = b_tensor.shape
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_full, l):
            raise ValueError(f"bias_tensor must have shape {(n_full, l)}, got {tuple(bias_tensor.shape)}")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        if bias_tensor is not None:
            raise ValueError("bias_tensor is only supported in dense mode")
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        num_experts = b_ptrs.shape[0]
        _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_full = n
        k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
        b_shape = (n_full, k_logical)
        l = num_experts

    n_out = n_full // 2

    _logger.debug("grouped_gemm_glu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        c_tensor_out = torch.empty_strided((valid_m, n_full, 1), (n_full, 1, valid_m * n_full), dtype=c_dtype, device=a_tensor.device)
        d_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    if valid_m == 0:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            d_col_tensor=d_col_tensor,
            amax_tensor=amax_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
        )

    # ---- Build cache key ----
    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    use_full_dynamic = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL") is not None

    if is_dense:
        cache_key = (
            weight_mode,
            act_func,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape if not use_full_dynamic else None,
            c_tensor_out.shape[1:] if not use_full_dynamic else None,
            a_tensor.dtype,
            b_tensor.dtype,
            c_tensor_out.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
            stride_order(c_tensor_out),
            *tensor_signature(sfa_tensor),
            *tensor_signature(alpha_tensor),
            *tensor_signature(sfb_tensor),
            *tensor_signature(bias_tensor),
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
            use_dynamic_sched,
            prob_tensor.shape if prob_tensor is not None else None,
            prob_tensor.stride() if prob_tensor is not None else None,
            prob_tensor.dtype if prob_tensor is not None else None,
        )
    else:
        cache_key = (
            weight_mode,
            act_func,
            a_tensor.shape,
            b_shape,
            a_tensor.dtype,
            b_dtype,
            a_tensor.stride(),
            c_tensor_out.shape,
            c_tensor_out.stride(),
            c_tensor_out.dtype,
            sfa_tensor.shape,
            sfa_tensor.stride(),
            sfa_tensor.dtype,
            b_ptrs.shape,
            sfb_ptrs.shape,
            b_ptrs.stride(),
            sfb_ptrs.stride(),
            b_ptrs.dtype,
            sfb_ptrs.dtype,
            sfd_row_tensor.shape if sfd_row_tensor is not None else None,
            sfd_row_tensor.stride() if sfd_row_tensor is not None else None,
            sfd_row_tensor.dtype if sfd_row_tensor is not None else None,
            sfd_col_tensor.shape if sfd_col_tensor is not None else None,
            sfd_col_tensor.stride() if sfd_col_tensor is not None else None,
            sfd_col_tensor.dtype if sfd_col_tensor is not None else None,
            norm_const_tensor.shape if norm_const_tensor is not None else None,
            norm_const_tensor.stride() if norm_const_tensor is not None else None,
            norm_const_tensor.dtype if norm_const_tensor is not None else None,
            *tensor_signature(alpha_tensor),
            padded_offsets.shape,
            padded_offsets.stride(),
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
            prob_tensor.shape if prob_tensor is not None else None,
            prob_tensor.stride() if prob_tensor is not None else None,
            prob_tensor.dtype if prob_tensor is not None else None,
        )

    # ---- Cache lookup or create + compile ----
    if cache_key in _cache_of_GroupedGemmGluSm100Objects:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Using cached object")
        api = _cache_of_GroupedGemmGluSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_glu_wrapper_sm100: Creating new object")
        if is_dense:
            api = GroupedGemmGluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor_out,
                sample_d=d_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d_col=d_col_tensor,
                sample_bias=bias_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                use_dynamic_sched=use_dynamic_sched,
            )
        else:
            api = GroupedGemmGluSm100(
                sample_a=a_tensor,
                sample_c=c_tensor_out,
                sample_d=d_tensor,
                sample_sfa=sfa_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_d_col=d_col_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_amax=amax_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                act_func=act_func,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
            )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_GroupedGemmGluSm100Objects[cache_key] = api

    # ---- Execute ----
    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            bias_tensor=bias_tensor,
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
    else:
        api.execute(
            a_tensor=a_tensor,
            c_tensor=c_tensor_out,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        c_tensor=c_tensor_out,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
