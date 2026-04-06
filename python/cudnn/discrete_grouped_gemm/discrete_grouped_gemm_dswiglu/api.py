# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
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
API for Discrete-weight Grouped GEMM dGLU Backward Kernel (SM100+)

This module provides the API class for discrete-weight block-scaled grouped GEMM
backward pass with dGLU (dSwiGLU/dGeGLU) activation gradient for MoE workloads.
"""

from .discrete_B_blockscaled_grouped_gemm_dglu_dbias import (
    BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel,
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


class DiscreteGroupedGemmDswigluSm100(APIBase):
    """API class for discrete-weight grouped GEMM dGLU backward operation on SM100+ GPUs.

    This kernel performs the backward pass of the discrete-weight grouped GEMM
    with dSwiGLU/dGeGLU activation gradient for MoE workloads.

    Example:
        >>> api = DiscreteGroupedGemmDswigluSm100(
        ...     sample_a=a_tensor, num_experts=8, b_shape=(n, k), b_dtype=torch.uint8, ...
        ... )
        >>> api.check_support()
        >>> api.compile()
        >>> api.execute(b_ptrs=b_ptrs_tensor, sfb_ptrs=sfb_ptrs_tensor, ...)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        num_experts: int,
        b_shape: Tuple[int, ...],
        b_dtype: torch.dtype,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_d_col: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_dprob: torch.Tensor,
        sample_dbias: Optional[torch.Tensor] = None,
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        act_func: str = "dswiglu",
        b_major: str = "k",
        epilogue_op: Optional[str] = None,
        use_dynamic_sched: bool = False,
    ):
        """Initialize the DiscreteGroupedGemmDswigluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1) -- gradient input
        :param num_experts: Number of experts
        :param b_shape: Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: Data type of B tensors
        :param sample_c: Sample C tensor (valid_m, 2n, 1) -- forward activations input
        :param sample_d_row: Sample D row output tensor (valid_m, 2n, 1)
        :param sample_d_col: Sample D column output tensor (valid_m, 2n, 1)
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_beta: Per-group beta scaling factors
        :param sample_prob: Per-row probability tensor (valid_m, 1, 1)
        :param sample_dprob: Gradient of probability tensor (valid_m, 1, 1), must be zero-initialized
        :param sample_dbias: Optional dbias output tensor (expert_cnt, 2*n, 1)
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor, shape (expert_cnt, 2, 1)
        :param sample_norm_const: Optional normalization constant
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Generate discrete col-major scale factor tensor
        :param act_func: Activation function, one of "dswiglu" or "dgeglu"
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param epilogue_op: Optional epilogue operation ("relu", "srelu", or None)
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        """
        super().__init__()

        self._logger.warning("DiscreteGroupedGemmDswigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        self._value_error_if(num_experts == 0, "num_experts must be > 0")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_row_desc = self._make_tensor_desc(sample_d_row, name="sample_d_row")
        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")
        self.beta_desc = self._make_tensor_desc(sample_beta, name="sample_beta")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob")
        self.dbias_desc = self._make_tensor_desc(sample_dbias, name="sample_dbias")

        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row")
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col")
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const"),
            1,
            "norm_const",
        )

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

        if epilogue_op in [None, "none", "identity"]:
            self.epilogue_op = lambda x: x
        elif epilogue_op == "relu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))
        elif epilogue_op == "srelu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0)) ** 2
        else:
            raise ValueError(f"Invalid epilogue operation: {epilogue_op}. Valid: None, 'relu', 'srelu'")

        self._interpret_uint8_as_fp4x2 = True
        self._kernel = BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        print(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")
        self._workspace = None

        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported."""
        self._logger.debug("Entering check_support")

        all_none = all(x is None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        all_provided = all(x is not None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        self._value_error_if(
            not (all_none or all_provided),
            "sfd_row, sfd_col, norm_const must be all None or all not None",
        )
        self._user_requested_sfd = all_provided

        tensor_m, k, _ = self._tensor_shape(self.a_desc, name="sample_a")
        if len(self.b_shape) == 2:
            n, b_k = self.b_shape
        else:
            n, b_k, _ = self.b_shape
        self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")

        n_out = 2 * n

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.c_desc, (tensor_m, n_out, 1), "C")
        self._check_tensor_shape(self.d_row_desc, (tensor_m, n_out, 1), "D_row")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n_out, 1), "D_col")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        rest_n_out = ceil_div(ceil_div(n_out, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_row_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_n_out, 1), "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_col_desc, (32, 4, ceil_div(n_out, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._check_tensor_shape(self.beta_desc, (self.expert_cnt,), "beta")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.dprob_desc, (tensor_m, 1, 1), "dprob")
        self._check_tensor_shape(self.dbias_desc, (self.expert_cnt, n_out, 1), "dbias")
        self._check_tensor_shape(self.amax_desc, (self.expert_cnt, 2, 1), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        self._logger.debug("Checking tensor strides")
        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
        )
        _ = self._check_tensor_stride(
            self.c_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="C must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_row_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="D_row must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_col_desc,
            stride=[(n_out, 1, tensor_m * n_out)],
            extra_error_msg="D_col must have n-major layout",
        )

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

        self._value_error_if(
            self.sf_vec_size not in [16, 32],
            f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}",
        )
        self._value_error_if(
            self.sf_dtype in [torch.float8_e4m3fn] and self.sf_vec_size == 32,
            f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} not supported",
        )
        self._value_error_if(
            self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16,
            f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} not supported",
        )

        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Accumulator")
        self._check_dtype(
            self.prob_desc,
            dtype=torch.float32,
            name="prob",
            extra_error_msg="prob must be float32",
        )
        self._check_dtype(
            self.dprob_desc,
            dtype=torch.float32,
            name="dprob",
            extra_error_msg="dprob must be float32",
        )
        self._check_dtype(
            self.dbias_desc,
            dtype=torch.bfloat16,
            name="Dbias",
            extra_error_msg="dbias must be bfloat16",
        )
        self.c_dtype = self._check_dtype(self.c_desc, dtype=[torch.float32, torch.float16, torch.bfloat16], name="C")

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
                dtype=[torch.float16, torch.bfloat16, torch.float32],
                name="D",
            )
        else:
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
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

        kernel_generate_sfd = self._is_fp8(self.ab_dtype) and self.sf_dtype == torch.float8_e8m0fnu and self._is_fp8(self.d_dtype)
        self._value_error_if(
            kernel_generate_sfd and not self._user_requested_sfd,
            "sfd_row, sfd_col, and norm_const are required for FP8 input/FP8 output with sf_dtype=torch.float8_e8m0fnu",
        )
        if not kernel_generate_sfd and self._user_requested_sfd:
            self._logger.warning(
                "sfd_row/sfd_col/norm_const were provided, but this configuration does not generate SFD outputs; the tensors will be ignored by the kernel",
            )
        self.generate_sfd = kernel_generate_sfd
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, will be ignored")
            self.discrete_col_sfd = False

        self._value_error_if(
            self.act_func not in ["dswiglu", "dgeglu"],
            f"act_func must be 'dswiglu' or 'dgeglu', got {self.act_func}",
        )
        self._value_error_if(
            self.b_major not in ["k", "n"],
            f"b_major must be 'k' or 'n', got {self.b_major}",
        )
        self._value_error_if(
            self._is_fp4x2(self.ab_dtype) and self.b_major != "k",
            "b_major must be 'k' when ab_dtype is fp4",
        )

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
            self.m_aligned != BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.FIX_PAD_SIZE,
            f"m_aligned must be {BlockScaledDiscreteWeightDgluDbiasGroupedGemmKernel.FIX_PAD_SIZE}, got {self.m_aligned}",
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
                and check_contiguous_16B_alignment(self.d_dtype, self.d_row_desc.stride_order, (tensor_m, n, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        self._value_error_if(
            self.expert_cnt > 1024,
            f"expert_cnt must be <= 1024, got {self.expert_cnt}",
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        if major * 10 + minor < 100:
            raise RuntimeError(f"DiscreteGroupedGemmDswiglu requires SM100+, found SM{major * 10 + minor}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the backward kernel from tensor descriptors captured in __init__."""
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
        b_stride_size = k if self.b_major == "k" else n

        gemm_dglu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
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

        workspace_bytes = gemm_dglu.get_workspace_bytes()
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
        d_row_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.d_row_desc.dtype,
            shape=(valid_m, *self.d_row_desc.shape[1:]),
            stride_order=self.d_row_desc.stride_order,
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
        beta_tensor = self._make_fake_cute_tensor_from_desc(self.beta_desc, assumed_align=16)
        prob_tensor = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, *self.prob_desc.shape[1:]),
            stride=self.prob_desc.stride,
            assumed_align=16,
        )
        dprob_tensor = self._make_fake_cute_tensor(
            dtype=self.dprob_desc.dtype,
            shape=(valid_m, *self.dprob_desc.shape[1:]),
            stride=self.dprob_desc.stride,
            assumed_align=16,
        )
        dbias_tensor = self._make_fake_cute_tensor_from_desc(self.dbias_desc, assumed_align=16)

        # Use internal device-resident int64 arrays to provide valid pointer-like
        # compile-time placeholders for b_ptrs/sfb_ptrs (required by kernel __call__).
        b_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        sfb_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
        b_ptrs_cute = from_dlpack(b_ptrs_placeholder, assumed_align=8).iterator
        sfb_ptrs_cute = from_dlpack(sfb_ptrs_placeholder, assumed_align=8).iterator
        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        self._logger.debug("Compiling discrete grouped GEMM dGLU kernel")
        _compiled_kernel = cute.compile(
            gemm_dglu,
            a_tensor,
            b_ptrs_cute,
            sfb_ptrs_cute,
            cutlass.Int32(n),
            cutlass.Int32(k),
            cutlass.Int64(b_stride_size),
            b_major_mode,
            workspace_ptr_cute,
            c_tensor,
            d_row_tensor,
            d_col_tensor,
            sfa_tensor,
            sfd_row_tensor,
            sfd_col_tensor,
            amax_tensor,
            norm_const_tensor_cute,
            padded_offsets_tensor,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            dprob_tensor,
            cutlass.Float32(0.0),
            dbias_tensor,
            max_active_clusters,
            fake_stream,
            self.epilogue_op,
            options="--enable-tvm-ffi",
        )

        self._gemm_dglu = gemm_dglu
        self._b_major_mode = b_major_mode
        self._b_stride_size = b_stride_size
        self._n = n
        self._k = k

        # Cache values that are constant across execute() calls
        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        cached_n = cutlass.Int32(self._n)
        cached_k = cutlass.Int32(self._k)
        cached_b_stride = cutlass.Int64(self._b_stride_size)
        cached_linear_offset = cutlass.Float32(0.0)

        def tensor_api(
            a_tensor,
            b_ptrs_device,
            sfb_ptrs_device,
            c_tensor,
            d_row_tensor,
            d_col_tensor,
            sfa_tensor,
            sfd_row_tensor,
            sfd_col_tensor,
            amax_tensor,
            norm_const_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            dprob_tensor,
            dbias_tensor,
            stream,
        ):
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
                d_row_tensor,
                d_col_tensor,
                sfa_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                beta_tensor,
                prob_tensor,
                dprob_tensor,
                cached_linear_offset,
                dbias_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_ptrs: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_ptrs: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        dbias_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled backward kernel (CUDA-graph-safe).

        :param a_tensor: Input A tensor (gradient input)
        :param b_ptrs: 1-D int64 device tensor of per-expert B data pointers
        :param c_tensor: Forward activations (input to backward)
        :param d_row_tensor: Output D row tensor
        :param d_col_tensor: Output D column tensor
        :param sfa_tensor: Scale factor A
        :param sfb_ptrs: 1-D int64 device tensor of per-expert SFB data pointers
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group alpha scaling
        :param beta_tensor: Per-group beta scaling
        :param prob_tensor: Per-row probability (from forward)
        :param dprob_tensor: Gradient of probability (output, must be zero-initialized)
        :param dbias_tensor: Optional dbias output tensor (expert_cnt, 2*n, 1)
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param current_stream: CUDA stream
        """
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if a_tensor.shape[0] == 0:
            self._logger.debug("execute: valid_m is zero, skipping")
            return
        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")

        self._compiled_kernel(
            a_tensor=a_tensor,
            b_ptrs_device=b_ptrs,
            sfb_ptrs_device=sfb_ptrs,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            stream=current_stream,
        )
        self._logger.debug("Execute completed")


_logger = logging.getLogger(__name__)
_cache_of_DiscreteGroupedGemmDswigluSm100Objects = {}


def discrete_grouped_gemm_dswiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_ptrs: torch.Tensor,
    c_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_ptrs: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    dprob_tensor: torch.Tensor,
    n: int,
    b_dtype: torch.dtype,
    generate_dbias: bool = False,
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    d_dtype: torch.dtype = torch.bfloat16,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    act_func: str = "dswiglu",
    b_major: str = "k",
    epilogue_op: Optional[str] = None,
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for discrete-weight grouped GEMM dGLU backward.

    Args:
        a_tensor: Input A tensor (gradient input)
        b_ptrs: 1-D int64 device tensor of per-expert B data pointers.
        c_tensor: Forward activations (input to backward)
        sfa_tensor: Scale factor A
        sfb_ptrs: 1-D int64 device tensor of per-expert SFB data pointers.
        padded_offsets: End offset per expert after padding
        alpha_tensor: Per-group alpha scaling
        beta_tensor: Per-group beta scaling
        prob_tensor: Per-row probability (from forward)
        dprob_tensor: Gradient of probability (output, must be zero-initialized)
        n: B weight N dimension
        b_dtype: B weight data type
        generate_dbias: Allocate and return dbias output

    Returns:
        TupleDict with keys: d_row_tensor, d_col_tensor, dprob_tensor,
            amax_tensor, sfd_row_tensor, sfd_col_tensor
    """
    valid_m, k_physical, _ = a_tensor.shape
    _require_pointer_tensor(b_ptrs, "b_ptrs")
    num_experts = b_ptrs.shape[0]
    _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
    k_logical = k_physical * 2 if b_dtype in (torch.float4_e2m1fn_x2, torch.uint8) else k_physical
    b_shape = (n, k_logical)

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    n_out = 2 * n
    d_row_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)
    d_col_tensor = torch.empty_strided((valid_m, n_out, 1), (n_out, 1, valid_m * n_out), dtype=d_dtype, device=a_tensor.device)

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None
    dbias_tensor = None

    if a_tensor.dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n_out, sf_vec_size)
        mma_shape_row = (1, ceil_div(valid_m, 128), ceil_div(sf_k_row, 4), 32, 4, 4)
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (1, ceil_div(n_out, 128), ceil_div(sf_k_col, 4), 32, 4, 4)
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        amax_tensor = torch.full(
            (num_experts, 2, 1),
            float("-inf"),
            dtype=torch.float32,
            device=a_tensor.device,
        )
    if generate_dbias:
        dbias_tensor = torch.zeros((num_experts, n_out, 1), dtype=torch.bfloat16, device=a_tensor.device)

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
        *tensor_signature(beta_tensor),
        *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
        *dynamic_m_tensor_signature(dprob_tensor, (1, 1)),
        *tensor_signature(dbias_tensor),
        *tensor_signature(norm_const_tensor),
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
        act_func,
        b_major,
        epilogue_op,
        use_dynamic_sched,
        num_experts,
    )

    if cache_key in _cache_of_DiscreteGroupedGemmDswigluSm100Objects:
        api = _cache_of_DiscreteGroupedGemmDswigluSm100Objects[cache_key]
    else:
        api = DiscreteGroupedGemmDswigluSm100(
            sample_a=a_tensor,
            num_experts=num_experts,
            b_shape=b_shape,
            b_dtype=b_dtype,
            sample_c=c_tensor,
            sample_d_row=d_row_tensor,
            sample_d_col=d_col_tensor,
            sample_sfa=sfa_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_beta=beta_tensor,
            sample_prob=prob_tensor,
            sample_dprob=dprob_tensor,
            sample_dbias=dbias_tensor,
            sample_amax=amax_tensor,
            sample_sfd_row=sfd_row_tensor,
            sample_sfd_col=sfd_col_tensor,
            sample_norm_const=norm_const_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            discrete_col_sfd=discrete_col_sfd,
            act_func=act_func,
            b_major=b_major,
            epilogue_op=epilogue_op,
            use_dynamic_sched=use_dynamic_sched,
        )
        if not api.check_support():
            raise RuntimeError("Unsupported configuration")
        api.compile()
        _cache_of_DiscreteGroupedGemmDswigluSm100Objects[cache_key] = api

    api.execute(
        a_tensor=a_tensor,
        b_ptrs=b_ptrs,
        c_tensor=c_tensor,
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        sfa_tensor=sfa_tensor,
        sfb_ptrs=sfb_ptrs,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        beta_tensor=beta_tensor,
        prob_tensor=prob_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
        amax_tensor=amax_tensor,
        norm_const_tensor=norm_const_tensor,
        current_stream=current_stream,
    )

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        dbias_tensor=dbias_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
