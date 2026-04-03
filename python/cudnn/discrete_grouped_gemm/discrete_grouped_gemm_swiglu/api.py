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
API for Discrete-weight Grouped GEMM GLU Forward Kernel (SM100+)

This module provides the API class for discrete-weight block-scaled grouped GEMM
with GLU activation (SwiGLU/GeGLU) for MoE (Mixture of Experts) workloads.

Unlike the contiguous kernel (GroupedGemmSwigluSm100), which requires all experts'
weights to be packed into a single (n, k, l) tensor, this kernel accepts per-expert
weight pointers -- avoiding the need to copy/reshape weights from independent
parameter allocations.
"""

from .discrete_B_blockscaled_grouped_gemm_glu_bias import (
    BlockScaledDiscreteWeightGroupedGemmBiasKernel,
)
from cuda.bindings import driver as cuda
import logging
import os
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
from cutlass.cute.runtime import make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2
from cudnn.discrete_grouped_gemm.discrete_kernel_utils import _require_pointer_tensor


class DiscreteGroupedGemmSwigluSm100(APIBase):
    """API class for discrete-weight grouped GEMM GLU forward operation on SM100+ GPUs.

    This kernel performs discrete-weight block-scaled grouped GEMM with GLU
    activation (SwiGLU or GeGLU), designed for MoE workloads where each expert's
    weight is a separate memory allocation.

    Key features:
    - Per-expert weight pointers (no weight stacking/copy required)
    - Supports SwiGLU and GeGLU activations
    - Variable M per group (aligned to m_aligned)
    - Block-scaled quantization support (MXF8, MXF4, NVF4)

    Example:
        >>> api = DiscreteGroupedGemmSwigluSm100(
        ...     sample_a=a_tensor,
        ...     num_experts=8, b_shape=(n, k), b_dtype=torch.uint8,
        ...     ...
        ... )
        >>> api.check_support()
        >>> api.compile()
        >>> api.execute(..., stream)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        num_experts: int,
        b_shape: Tuple[int, ...],
        b_dtype: torch.dtype,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_d_col: torch.Tensor,
        sample_bias: Optional[torch.Tensor] = None,
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
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
        """Initialize the DiscreteGroupedGemmSwigluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param num_experts: Number of experts
        :param b_shape: Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: Data type of B tensors
        :param sample_c: Sample C tensor for intermediate storage
        :param sample_d: Sample D output tensor (valid_m, n/2, 1) after GLU
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d_col: Column-quantized D tensor
        :param sample_bias: Optional bias tensor with shape (n, expert_cnt), stride (1, n), and fp16/bfloat16 dtype
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

        self._logger.warning("DiscreteGroupedGemmSwigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        self._value_error_if(num_experts == 0, "num_experts must be > 0")

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

        self.expert_cnt = num_experts
        self._value_error_if(
            self.padded_offsets_desc.shape[0] != self.expert_cnt,
            f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) must equal expert_cnt ({self.expert_cnt})",
        )

        self.b_dtype = b_dtype
        self.b_shape = b_shape

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
        self.b_major = b_major
        self.use_dynamic_sched = use_dynamic_sched

        self._interpret_uint8_as_fp4x2 = True
        self._has_bias = self.bias_desc is not None
        self._kernel = BlockScaledDiscreteWeightGroupedGemmBiasKernel

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

        if len(self.b_shape) == 2:
            n, b_k = self.b_shape
        else:
            n, b_k, _ = self.b_shape
        self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")

        _, n_2, _one = self._tensor_shape(self.d_desc, name="sample_d")

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.c_desc, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n // 2, 1), "D")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n // 2, 1), "D_col")
        self._check_tensor_shape(self.bias_desc, (n, self.expert_cnt), "bias")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
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

        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
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

        self._value_error_if(
            self.b_dtype != self.ab_dtype,
            f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
        )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD",
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
        self._check_dtype(
            self.bias_desc,
            dtype=[torch.bfloat16, torch.float16],
            name="bias",
            extra_error_msg="bias must be fp16 or bfloat16",
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
            "Invalid configuration: fp4 ab_dtype, sf_vec_size 16, d_dtype float32 is not supported",
        )

        self._value_error_if(
            self.act_func not in ["swiglu", "geglu"],
            f"act_func must be 'swiglu' or 'geglu', got {self.act_func}",
        )
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
            not self.use_2cta_instrs and self.mma_tiler_mn[0] != 128,
            f"MMA tiler M must be 128 when use_2cta_instrs=False, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.use_2cta_instrs and self.mma_tiler_mn[0] != 256,
            f"MMA tiler M must be 256 when use_2cta_instrs=True, got {self.mma_tiler_mn[0]}",
        )
        self._value_error_if(
            self.mma_tiler_mn[1] != 256,
            f"MMA tiler N must be 256, got {self.mma_tiler_mn[1]}",
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
            self.m_aligned != BlockScaledDiscreteWeightGroupedGemmBiasKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledDiscreteWeightGroupedGemmBiasKernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
        )

        self._logger.debug("Checking tensor alignment")

        def check_contiguous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        b_stride_order = (0, 1, 2) if self.b_major == "n" else (1, 0, 2)
        self._value_error_if(
            not (
                check_contiguous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (tensor_m, k, 1))
                and check_contiguous_16B_alignment(self.ab_dtype, b_stride_order, (n, k, 1))
                and check_contiguous_16B_alignment(self.d_dtype, self.d_desc.stride_order, (tensor_m, n_2, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        self._value_error_if(
            self.expert_cnt > 1024,
            f"expert_cnt must be <= 1024, got {self.expert_cnt}",
        )

        self._not_implemented_error_if(
            (self._is_fp8(self.ab_dtype)) and (self.mma_tiler_mn[1] == 128) and (self._is_fp8(self.d_dtype)),
            "Invalid configuration: fp8 ab_dtype with mma_tiler_mn[1] == 128 and fp8 d_dtype is not supported",
        )
        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and (self.c_dtype not in [torch.float16, torch.bfloat16]),
            f"Invalid configuration: for fp4 ab_dtype, c_dtype must be float16 or bfloat16, got {self.c_dtype}",
        )
        self._not_implemented_error_if(
            self._has_bias and self.mma_tiler_mn[1] != 256,
            "Discrete bias fusion currently requires mma_tiler_mn[1] == 256",
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"DiscreteGroupedGemmSwiglu requires SM100+, but found SM{compute_capability}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the kernel from tensor descriptors captured in __init__."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            self._logger.debug("Kernel already compiled; skipping recompilation")
            return
        if self.a_desc.shape[0] == 0:
            self._logger.debug("sample valid_m is zero, skipping kernel compilation")
            return

        from cutlass.cute.runtime import from_dlpack

        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape

        b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
        if self.b_major == "k":
            b_stride_size = k
        else:
            b_stride_size = n

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
            enable_bias=self._has_bias,
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

        workspace_bytes = gemm_glu.get_workspace_bytes()
        self._workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")

        ab_cutlass_dtype = _convert_to_cutlass_data_type(self.a_desc.dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2)
        align = 32 if ab_cutlass_dtype.width == 4 else 16

        valid_m = cute.sym_int(divisibility=256)
        a_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, *self.a_desc.shape[1:]),
            stride_order=self.a_desc.stride_order,
            assumed_align=align,
        )
        c_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.c_desc.dtype,
            shape=(valid_m, *self.c_desc.shape[1:]),
            stride_order=self.c_desc.stride_order,
        )
        d_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.d_desc.dtype,
            shape=(valid_m, *self.d_desc.shape[1:]),
            stride_order=self.d_desc.stride_order,
        )
        d_col_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.d_col_desc.dtype,
            shape=(valid_m, *self.d_col_desc.shape[1:]),
            stride_order=self.d_col_desc.stride_order,
        )

        tensor_m_128 = cute.sym_int()
        stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
        sfa_shape = list(self.sfa_desc.shape)
        sfa_shape[2] = tensor_m_128
        sfa_stride = list(self.sfa_desc.stride)
        sfa_stride[5] = stride_tensor_m_128
        sfa_tensor = self._make_fake_cute_tensor(
            dtype=self.sfa_desc.dtype,
            shape=tuple(sfa_shape),
            stride=tuple(sfa_stride),
            assumed_align=16,
        )
        sfd_row_tensor = None
        if self.sfd_row_desc is not None:
            stride_sfd_m = cute.sym_int(divisibility=32 * 4 * 4)
            sfd_row_tensor = self._make_fake_cute_tensor(
                dtype=self.sfd_row_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfd_row_desc.shape[4], 1),
                stride=(16, 4, self.sfd_row_desc.stride[2], 1, 512, stride_sfd_m),
                assumed_align=16,
            )
        sfd_col_tensor = None
        if self.sfd_col_desc is not None:
            rest_m = cute.sym_int(divisibility=1)
            stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
            stride_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
            sfd_col_tensor = self._make_fake_cute_tensor(
                dtype=self.sfd_col_desc.dtype,
                shape=(32, 4, self.sfd_col_desc.shape[2], 4, rest_m, 1),
                stride=(16, 4, stride_rest_m, 1, 512, stride_sfd_n),
                assumed_align=16,
            )
        amax_tensor = self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16)
        norm_const_tensor_cute = self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16)
        padded_offsets_tensor = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        alpha_tensor = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        prob_tensor = None
        if self.prob_desc is not None:
            prob_tensor = self._make_fake_cute_tensor(
                dtype=self.prob_desc.dtype,
                shape=(valid_m, *self.prob_desc.shape[1:]),
                stride=self.prob_desc.stride,
                assumed_align=16,
            )
        bias_tensor = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)

        # Use internal device-resident int64 arrays to provide valid pointer-like
        # compile-time placeholders for b_ptrs/sfb_ptrs (required by kernel __call__).
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
            bias_tensor,
            max_active_clusters,
            fake_stream,
            options="--enable-tvm-ffi",
        )

        self._gemm_glu = gemm_glu
        self._b_major_mode = b_major_mode
        self._b_stride_size = b_stride_size
        self._n = n
        self._k = k

        # Cache values that are constant across execute() calls to avoid
        # per-call from_dlpack / object creation overhead.
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
            bias_tensor: Optional[torch.Tensor],
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
                bias_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_ptrs: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_ptrs: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        bias_tensor: Optional[torch.Tensor] = None,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        This method is designed to be CUDA-graph-safe: it performs no
        allocations and accepts only pre-built device tensors.

        :param a_tensor: Input A tensor
        :param b_ptrs: 1-D int64 device tensor of per-expert B data pointers.
            Build once via ``torch.tensor([b.data_ptr() for b in experts], dtype=torch.int64, device="cuda")``.
        :param c_tensor: Intermediate C tensor
        :param d_tensor: Output D tensor
        :param sfa_tensor: Scale factor A
        :param sfb_ptrs: 1-D int64 device tensor of per-expert SFB data pointers.
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param bias_tensor: Optional bias tensor with shape (n, expert_cnt) and stride (1, n).
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

        self._logger.debug("Executing discrete grouped GEMM GLU kernel")
        if self._has_bias:
            self._value_error_if(
                bias_tensor is None,
                "bias_tensor must be provided at execute() when the API was compiled with sample_bias",
            )

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
            bias_tensor=bias_tensor,
            stream=current_stream,
        )

        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_DiscreteGroupedGemmSwigluSm100Objects = {}


def discrete_grouped_gemm_swiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_ptrs: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_ptrs: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    n: int,
    b_dtype: torch.dtype,
    bias_tensor: Optional[torch.Tensor] = None,
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
    b_major: str = "k",
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for discrete-weight grouped GEMM GLU forward operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        b_ptrs: 1-D int64 device tensor of per-expert B data pointers.
            Build via ``torch.tensor([b.data_ptr() for b in experts], dtype=torch.int64, device="cuda")``.
        sfa_tensor: Scale factor A
        sfb_ptrs: 1-D int64 device tensor of per-expert SFB data pointers.
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group scaling
        n: B weight N dimension (full N before GLU split)
        b_dtype: B weight data type
        bias_tensor: Optional bias tensor with shape (n, expert_cnt) and stride (1, n)
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
        b_major: B tensor major dimension ("k" or "n")
        current_stream: CUDA stream

    Returns:
        TupleDict with keys: c_tensor, d_tensor, d_col_tensor, amax_tensor,
            sfd_row_tensor, sfd_col_tensor
    """
    valid_m, k_physical, _ = a_tensor.shape
    _require_pointer_tensor(b_ptrs, "b_ptrs")
    num_experts = b_ptrs.shape[0]
    _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
    n_out = n // 2
    k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
    b_shape = (n, k_logical)
    if bias_tensor is not None and tuple(bias_tensor.shape) != (n, num_experts):
        raise ValueError(f"bias_tensor must have shape {(n, num_experts)}, got {tuple(bias_tensor.shape)}")

    _logger.debug("discrete_grouped_gemm_swiglu_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        c_tensor = torch.empty_strided((valid_m, n, 1), (n, 1, valid_m * n), dtype=c_dtype, device=a_tensor.device)
        d_tensor = torch.empty_strided(
            (valid_m, n_out, 1),
            (n_out, 1, valid_m * n_out),
            dtype=d_dtype,
            device=a_tensor.device,
        )
        d_col_tensor = torch.empty_strided(
            (valid_m, n_out, 1),
            (n_out, 1, valid_m * n_out),
            dtype=d_dtype,
            device=a_tensor.device,
        )
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("discrete_grouped_gemm_swiglu_wrapper_sm100: Detected fp8 config, constructing sfd tensors")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("discrete_grouped_gemm_swiglu_wrapper_sm100: Constructing amax_tensor")
        amax_tensor = torch.full((num_experts, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    cache_key = (
        a_tensor.shape[1:],
        stride_order(a_tensor),
        a_tensor.dtype,
        b_shape,
        b_dtype,
        c_tensor.shape[1:],
        stride_order(c_tensor),
        c_tensor.dtype,
        *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
        *tensor_signature(alpha_tensor),
        *tensor_signature(bias_tensor),
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
        c_dtype,
        d_dtype,
        cd_major,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        discrete_col_sfd,
        act_func,
        b_major,
        use_dynamic_sched,
        num_experts,
    )

    if cache_key in _cache_of_DiscreteGroupedGemmSwigluSm100Objects:
        _logger.debug("discrete_grouped_gemm_swiglu_wrapper_sm100: Using cached object")
        api = _cache_of_DiscreteGroupedGemmSwigluSm100Objects[cache_key]
        api.execute(
            a_tensor=a_tensor,
            b_ptrs=b_ptrs,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_ptrs=sfb_ptrs,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            bias_tensor=bias_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
    else:
        _logger.debug("discrete_grouped_gemm_swiglu_wrapper_sm100: Creating new object")
        api = DiscreteGroupedGemmSwigluSm100(
            sample_a=a_tensor,
            num_experts=num_experts,
            b_shape=b_shape,
            b_dtype=b_dtype,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=sfa_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_amax=amax_tensor,
            sample_d_col=d_col_tensor,
            sample_bias=bias_tensor,
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
            act_func=act_func,
            b_major=b_major,
            use_dynamic_sched=use_dynamic_sched,
        )

        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        api.execute(
            a_tensor=a_tensor,
            b_ptrs=b_ptrs,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_ptrs=sfb_ptrs,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            bias_tensor=bias_tensor,
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            current_stream=current_stream,
        )
        _cache_of_DiscreteGroupedGemmSwigluSm100Objects[cache_key] = api

    return TupleDict(
        c_tensor=c_tensor,
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
