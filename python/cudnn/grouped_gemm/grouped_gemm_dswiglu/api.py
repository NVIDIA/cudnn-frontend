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
API for Grouped GEMM dSwiGLU Backward Kernel (SM100+)

This module provides the API class for contiguous grouped block-scaled GEMM
backward pass with dSwiGLU activation gradient for MoE (Mixture of Experts) workloads.
"""

from .grouped_gemm_dswiglu_quant import (
    BlockScaledContiguousGroupedGemmKernel,
)
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2


class GroupedGemmDswigluSm100(APIBase):
    """API class for Grouped GEMM dSwiGLU backward operation on SM100+ GPUs.

    This kernel performs contiguous grouped block-scaled GEMM backward pass
    with dSwiGLU activation gradient, designed for MoE (Mixture of Experts) workloads.

    Key features:
    - Supports variable M per group (aligned to m_aligned)
    - Contiguous memory layout for A and D tensors
    - Block-scaled quantization support (MXF8, MXF4, NVF4)
    - Uses padded_offsets interface for expert mapping

    Example:
        >>> api = GroupedGemmDswigluSm100(
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
        sample_d_row: torch.Tensor,
        sample_d_col: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_dprob: torch.Tensor,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        epilogue_op: Optional[str] = None,
    ):
        """Initialize the GroupedGemmDswigluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_b: Sample B tensor (n, k, l) where l = num_groups
        :param sample_c: Sample C tensor for intermediate storage (valid_m, 2n, 1)
        :param sample_d_row: Sample D row output tensor (valid_m, 2n, 1) after dSwiGLU
        :param sample_d_col: Sample D column output tensor (valid_m, 2n, 1) after dSwiGLU
        :param sample_sfa: Sample scale factor A tensor
        :param sample_sfb: Sample scale factor B tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_beta: Per-group beta scaling factors
        :param sample_prob: Per-row probability tensor (valid_m, 1, 1)
        :param sample_dprob: Gradient of probability tensor (valid_m, 1, 1). Must be zero-initialized.
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization, shape (l, 2, 1)
        :param sample_norm_const: Optional normalization constant
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor
        :param epilogue_op: Optional epilogue operation. Valid values: None, "none", "identity", "relu", "srelu"
        """
        super().__init__()

        self._logger.warning("GroupedGemmDswigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        # Store sample tensor descriptors
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_row_desc = self._make_tensor_desc(sample_d_row, name="sample_d_row")
        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")
        self.beta_desc = self._make_tensor_desc(sample_beta, name="sample_beta")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob")

        # Optional quantization outputs
        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row")
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col")
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax")
        self.norm_const_desc = self._unpad_tensor_to_ndim(self._make_tensor_desc(sample_norm_const, name="sample_norm_const"), 1, "norm_const")

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

        # expert_cnt derived from padded_offsets shape
        self.expert_cnt = self.padded_offsets_desc.shape[0]

        # Epilogue operation
        if epilogue_op in [None, "none", "identity"]:
            self.epilogue_op = lambda x: x
        elif epilogue_op == "relu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))
        elif epilogue_op == "srelu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0)) ** 2
        else:
            raise ValueError(f"Invalid epilogue operation: {epilogue_op}. Valid values: None, 'none', 'identity', 'relu', 'srelu'")

        self._interpret_uint8_as_fp4x2 = True
        self._kernel = BlockScaledContiguousGroupedGemmKernel
        self._logger.debug(f"__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        all_none = all(x is None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        none_none = all(x is not None for x in [self.sfd_row_desc, self.sfd_col_desc, self.norm_const_desc])
        if not (all_none or none_none):
            raise ValueError("sfd_row, sfd_col, and norm_const must be all None or all not None")
        self.generate_sfd = none_none
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, discrete_col_sfd will be ignored")
            self.discrete_col_sfd = False

        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self.a_desc.shape
        n, _, l = self.b_desc.shape
        _, _, _one = self.c_desc.shape
        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n * 2, 1), "C")
        self._check_tensor_shape(self.d_row_desc, (tensor_m, n * 2, 1), "D_row")
        self._check_tensor_shape(self.d_col_desc, (tensor_m, n * 2, 1), "D_col")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
        # SFD uses full n dimension since D has n columns (interleaved ab and dswiglu)
        rest_n2_full = ceil_div(ceil_div(n * 2, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_row_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_n2_full, 1), "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfd_col_desc, (32, 4, ceil_div(n * 2, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.padded_offsets_desc, (l,), "padded_offsets")
        self._check_tensor_shape(self.alpha_desc, (l,), "alpha")
        self._check_tensor_shape(self.beta_desc, (l,), "beta")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.dprob_desc, (tensor_m, 1, 1), "dprob")
        self._check_tensor_shape(self.amax_desc, (l, 2, 1), "amax")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")

        _ = self._check_tensor_stride(self.a_desc, stride=[(k, 1, tensor_m * k)], extra_error_msg="A must have k-major layout")
        if self._is_fp8(self.a_desc):
            _ = self._check_tensor_stride(
                self.b_desc, stride=[(k, 1, n * k), (1, n, n * k)], extra_error_msg="For fp8 ab_dtype, B must have k- or n-major layout"
            )
        else:
            _ = self._check_tensor_stride(self.b_desc, stride=[(k, 1, n * k)], extra_error_msg="For fp4 ab_dtype, B must have k-major layout")
        _ = self._check_tensor_stride(self.c_desc, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="C must have n-major layout")
        # D has same shape as C (n columns for interleaved ab and dswiglu)
        _ = self._check_tensor_stride(self.d_row_desc, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="D_row must have n-major layout")
        _ = self._check_tensor_stride(self.d_col_desc, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="D_col must have n-major layout")

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
        self._check_dtype(self.b_desc, dtype=self.ab_dtype, name="B", extra_error_msg="B must have the same dtype as A")

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD_row/SFD_col",
        )
        self._check_dtype(self.sfb_desc, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must have the same dtype as SFA")
        self._check_dtype(self.sfd_row_desc, dtype=self.sf_dtype, name="SFD_row", extra_error_msg="SFD_row must have the same dtype as SFA")
        self._check_dtype(self.sfd_col_desc, dtype=self.sf_dtype, name="SFD_col", extra_error_msg="SFD_col must have the same dtype as SFA")

        if self.sf_vec_size not in [16, 32]:
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        if self.sf_dtype in [torch.float8_e4m3fn] and self.sf_vec_size == 32:
            raise ValueError(f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported")
        if self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16:
            raise ValueError(f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported")

        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Accumulator", extra_error_msg="Accumulator must be float32")
        self._check_dtype(self.prob_desc, dtype=torch.float32, name="Prob", extra_error_msg="Prob must be float32")
        self._check_dtype(self.dprob_desc, dtype=torch.float32, name="Dprob", extra_error_msg="Dprob must be float32")
        self.c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2],
            name="C",
        )
        if self._is_fp8(self.c_dtype) and self.vector_f32:
            raise ValueError(
                f"Invalid configuration: fp8 ab_dtype, c_dtype, and vector_f32 is not supported. Please use vector_f32=False or c_dtype=bfloat16 instead"
            )

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
                dtype=[torch.float16, torch.bfloat16, torch.float32],
                name="D_row",
                extra_error_msg="D_row must be fp16, bf16, or float32 when ab_dtype is fp4",
            )
        elif self._is_fp8(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_row_desc,
                dtype=[
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ],
                name="D_row",
                extra_error_msg="D_row must be fp8 dtype when ab_dtype is fp8",
            )
        else:
            raise NotImplementedError(f"Invalid ab_dtype: {self.ab_dtype}, expected fp4 or fp8")
        self._check_dtype(self.d_col_desc, dtype=self.d_dtype, name="D_col", extra_error_msg="D_col must have the same dtype as D_row")

        self._logger.debug("Checking MMA tile shape and cluster shape")
        if not self.use_2cta_instrs and self.mma_tiler_mn[0] not in [64, 128]:
            raise ValueError(f"MMA tiler M must be 64 or 128 when use_2cta_instrs=False, got {self.mma_tiler_mn[0]}")
        if self.use_2cta_instrs and self.mma_tiler_mn[0] not in [128, 256]:
            raise ValueError(f"MMA tiler M must be 128 or 256 when use_2cta_instrs=True, got {self.mma_tiler_mn[0]}")
        if self.mma_tiler_mn[1] not in [128, 256]:
            raise ValueError(f"MMA tiler N must be 128 or 256, got {self.mma_tiler_mn[1]}")
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(f"cluster_shape_mn[0] must be divisible by 2 when use_2cta_instrs=True, got {self.cluster_shape_mn[0]}")
        if not (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
            and self.cluster_shape_mn[0] > 0
            and self.cluster_shape_mn[1] > 0
            and self.cluster_shape_mn[0] <= 4
            and self.cluster_shape_mn[1] <= 4
            and is_power_of_2(self.cluster_shape_mn[0])
            and is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise ValueError(
                f"Invalid cluster shape: expected values to be powers of 2 and cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn[0]},{self.cluster_shape_mn[1]}"
            )
        cluster_tiler_m = (self.cluster_shape_mn[0] // (2 if self.use_2cta_instrs else 1)) * self.mma_tiler_mn[0]
        if cluster_tiler_m not in [128, 256]:
            raise ValueError(f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}")
        if self.m_aligned % self.mma_tiler_mn[0] != 0:
            raise ValueError(f"Invalid m_aligned: expected m_aligned to be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0")
        if self.m_aligned != BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE:
            raise ValueError(
                f"Invalid m_aligned: expected m_aligned to be equal to FIX_PAD_SIZE, got {self.m_aligned} != {BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE}"
            )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        if not (
            check_contigous_16B_alignment(self.ab_dtype, self.a_desc.stride_order, (tensor_m, k, l))
            and check_contigous_16B_alignment(self.ab_dtype, self.b_desc.stride_order, (n, k, l))
            and check_contigous_16B_alignment(self.d_dtype, self.d_row_desc.stride_order, (tensor_m, n, l))  # c, d_row, and d_col have the same stride order
        ):
            raise ValueError("Invalid tensor alignment: tensors must be 16B aligned")

        # Check expert_cnt constraint
        if self.expert_cnt > 1024:
            raise ValueError(f"expert_cnt must be <= 1024, got {self.expert_cnt}")

        # Check environment
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmDswiglu requires SM100+ compute capability, " f"but found SM{compute_capability} on device {device}")
        if compute_capability == 103:
            raise RuntimeError("cuteDSL GroupedGemmDswiglu is not supported on SM103")

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

        gemm_dswiglu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
            use_mono_increase_expert_idx=True,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        self._logger.debug("Compiling grouped_gemm_dswiglu kernel")
        _compiled_kernel = cute.compile(
            gemm_dswiglu,
            a=self._make_fake_cute_tensor_from_desc(self.a_desc, assumed_align=16),
            b=self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16),
            c=self._make_fake_cute_tensor_from_desc(self.c_desc, assumed_align=16),
            d=self._make_fake_cute_tensor_from_desc(self.d_row_desc, assumed_align=16),
            d_col=self._make_fake_cute_tensor_from_desc(self.d_col_desc, assumed_align=16),
            sfa=self._make_fake_cute_tensor_from_desc(self.sfa_desc, assumed_align=16),
            sfb=self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16),
            sfd_row_tensor=self._make_fake_cute_tensor_from_desc(self.sfd_row_desc, assumed_align=16),
            sfd_col_tensor=self._make_fake_cute_tensor_from_desc(self.sfd_col_desc, assumed_align=16),
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            beta=self._make_fake_cute_tensor_from_desc(self.beta_desc, assumed_align=16),
            prob=self._make_fake_cute_tensor_from_desc(self.prob_desc, assumed_align=16),
            dprob=self._make_fake_cute_tensor_from_desc(self.dprob_desc, assumed_align=16),
            max_active_clusters=max_active_clusters,
            epilogue_op=self.epilogue_op,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_row_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            beta_tensor: torch.Tensor,
            prob_tensor: torch.Tensor,
            dprob_tensor: torch.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            _compiled_kernel(
                a_tensor,
                b_tensor,
                c_tensor,
                d_row_tensor,
                d_col_tensor,
                sfa_tensor,
                sfb_tensor,
                sfd_row_tensor,
                sfd_col_tensor,
                amax_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                beta_tensor,
                prob_tensor,
                dprob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor
        :param b_tensor: Input B tensor (weights)
        :param c_tensor: Intermediate C tensor (from forward pass)
        :param d_row_tensor: Output D row tensor
        :param d_col_tensor: Output D column tensor
        :param sfa_tensor: Scale factor A
        :param sfb_tensor: Scale factor B
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group alpha scaling factors
        :param beta_tensor: Per-group beta scaling factors
        :param prob_tensor: Per-row probability tensor
        :param dprob_tensor: Gradient of probability tensor. Must be zero-initialized.
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param current_stream: CUDA stream
        """
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if self._compiled_kernel is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        self._logger.debug("Executing grouped_gemm_dswiglu kernel")
        self._compiled_kernel(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            stream=current_stream,
        )

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDswigluSm100Objects = {}


def grouped_gemm_dswiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,  # Intermediate from forward pass (required)
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
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
    epilogue_op: Optional[str] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM dSwiGLU backward operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        b_tensor: Weight B tensor (n, k, l)
        c_tensor: Intermediate C tensor from forward pass (valid_m, 2n, 1)
        sfa_tensor: Scale factor A
        sfb_tensor: Scale factor B
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group alpha scaling
        beta_tensor: Per-group beta scaling
        prob_tensor: Per-row probability tensor
        norm_const_tensor: Optional normalization constant
        acc_dtype: Accumulator data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (note: only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment
        discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor
        epilogue_op: Optional epilogue operation. Valid values: None, "none", "identity", "relu", "srelu"
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **d_row_tensor** (torch.Tensor): Final output tensor after dSwiGLU
            - **d_col_tensor** (torch.Tensor): Column-wise output tensor
            - **dprob_tensor** (torch.Tensor): Gradient tensor for prob (shape `(valid_m, 1, 1)`)
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (shape l, 2, 1)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D
    """
    valid_m = a_tensor.shape[0]
    n, _, l = b_tensor.shape

    _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Creating output tensors d_row_tensor, d_col_tensor, dprob_tensor")

    if cd_major == "n":
        d_row_tensor = torch.empty_strided((valid_m, n * 2, 1), (n * 2, 1, valid_m * n * 2), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n * 2, 1), (n * 2, 1, valid_m * n * 2), dtype=d_dtype, device=a_tensor.device)
        dprob_tensor = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n * 2, sf_vec_size)
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
            ceil_div(n * 2, 128),
            ceil_div(sf_k_col, 4),
            32,
            4,
            4,
        )
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 2, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        c_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        c_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        c_tensor.stride(),
        sfa_tensor.shape,
        sfb_tensor.shape,
        sfa_tensor.stride(),
        sfb_tensor.stride(),
        sfa_tensor.dtype,
        sfb_tensor.dtype,
        padded_offsets.shape,
        padded_offsets.stride(),
        padded_offsets.dtype,
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        acc_dtype,
        d_dtype,
        cd_major,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        discrete_col_sfd,
        epilogue_op,
    )

    if cache_key in _cache_of_GroupedGemmDswigluSm100Objects:
        _logger.debug("group_gemm_dswiglu_wrapper_sm100: Using previously cached GroupedGemmDswigluSm100 object")
        grouped_gemm_dswiglu = _cache_of_GroupedGemmDswigluSm100Objects[cache_key]
        grouped_gemm_dswiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
    else:
        _logger.debug(
            "group_gemm_dswiglu_wrapper_sm100: No previously cached GroupedGemmDswigluSm100 object found, creating new GroupedGemmDswigluSm100 object"
        )
        grouped_gemm_dswiglu = GroupedGemmDswigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d_row=d_row_tensor,
            sample_d_col=d_col_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_beta=beta_tensor,
            sample_prob=prob_tensor,
            sample_dprob=dprob_tensor,
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
            epilogue_op=epilogue_op,
        )

        assert grouped_gemm_dswiglu.check_support(), "Unsupported configuration"
        grouped_gemm_dswiglu.compile()
        grouped_gemm_dswiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
        _cache_of_GroupedGemmDswigluSm100Objects[cache_key] = grouped_gemm_dswiglu

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
