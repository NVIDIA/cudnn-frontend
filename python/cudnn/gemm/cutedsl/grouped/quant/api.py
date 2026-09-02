# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unified API for Grouped GEMM Quant Kernel (SM100+)

This module provides a single API class that supports both dense (contiguous)
and discrete weight modes for grouped block-scaled GEMM with output
quantization in MoE (Mixture of Experts) workloads.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, TupleDict, ceil_div, get_device_type, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    allocate_byte_workspace,
    cuda_is_available,
    default_stream,
    detect_framework,
    framework_dtype,
    get_compute_capability,
    get_data_ptr,
    get_device,
)

from .grouped_gemm_quant import (
    BlockScaledMoEGroupedGemmQuantKernel,
)
from ..moe_utils import MoEWeightMode
from ..backend_utils import rubin_single_group_offsets_kwarg
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack

_JAX_SF_LAYOUT_ERROR = (
    "the block scale-factor tensors (sfa/sfb and the sfd outputs) are MMA-tiled "
    "(32, 4, m//128, 4, rest_k, l) strided views that are not expressible as JAX arrays "
    "(a row-major JAX array of that shape has different memory); pass torch tensors"
)


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_quant_rubin import (
        BlockScaledMoEGroupedGemmQuantKernel as RubinBlockScaledMoEGroupedGemmQuantKernel,
    )

    return RubinBlockScaledMoEGroupedGemmQuantKernel


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
        sample_d_col: Optional[torch.Tensor] = None,
        # Dense mode (contiguous) -- provide these:
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
        sample_row_scale: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        use_single_group_runtime_offsets: bool = False,
    ):
        """Initialize the GroupedGemmQuantSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_d: Sample D output tensor (valid_m, n, 1)
        :param sample_d_col: Optional column-quantized D tensor. Required only when SFD outputs are generated.
        :param sample_b: (Dense) Sample B tensor (n, k, l)
        :param sample_sfb: (Dense) Sample scale factor B tensor
        :param sample_bias: Optional bias tensor with shape (n, l) or (n, expert_cnt), stride (1, n).
            Dense mode supports fp16/bfloat16/float32 bias; discrete mode supports fp16/bfloat16 bias.
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: (Discrete) Data type of B tensors
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization
        :param sample_norm_const: Optional normalization constant
        :param sample_prob: Optional probability tensor for gating
        :param sample_row_scale: Optional 1-D FP32 row-scale tensor. When
            provided, the epilogue scales GEMM accumulators by
            ``alpha[expert] * row_scale[m]`` before output conversion.
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param sf_fp8_dtype_override: Reinterpret the FP8-format block scale factors
            as E5M3 instead of the E4M3 implied by their storage dtype. ``None``
            (default) leaves the format inferred, as every caller did before this
            knob existed. ``"e5m3"`` requires Rubin and the NVFP4 recipe, and the
            scale tensors are still supplied as ``torch.float8_e4m3fn`` because
            torch has no e5m3 dtype -- only the CuTe element type is overridden.
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Enable discrete col-major scale factor tensor
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        """
        framework = detect_framework(sample_a)
        if framework == "jax":
            raise ValueError(f"GroupedGemmQuantSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        if framework != "torch":
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmQuantSm100; pass torch tensors")
        if acc_dtype is None:
            acc_dtype = cutlass.Float32
        super().__init__()
        self._framework = framework

        self._warn_experimental_api()
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

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", canonical=True)
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d", canonical=True)
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha", canonical=True)

        self._has_d_col = sample_d_col is not None
        self.d_col_desc = self._make_tensor_desc(sample_d_col, name="sample_d_col", canonical=True)
        if self.d_col_desc is None:
            self.d_col_desc = TensorDesc(
                dtype=self.d_desc.dtype,
                shape=self.d_desc.shape,
                stride=self.d_desc.stride,
                stride_order=self.d_desc.stride_order,
                device=self.d_desc.device,
                name="sample_d_col",
            )
        self.sfd_row_desc = self._make_tensor_desc(sample_sfd_row, name="sample_sfd_row", canonical=True)
        self.sfd_col_desc = self._make_tensor_desc(sample_sfd_col, name="sample_sfd_col", canonical=True)
        self.amax_desc = self._make_tensor_desc(sample_amax, name="sample_amax", canonical=True)
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const", canonical=True),
            1,
            "norm_const",
        )
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.row_scale_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_row_scale, name="sample_row_scale", canonical=True),
            1,
            "row_scale",
        )
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias", canonical=True)

        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", canonical=True)
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb", canonical=True)
            self.expert_cnt = self.padded_offsets_desc.shape[0]
        else:
            self._value_error_if(num_experts == 0, "num_experts must be > 0")
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = _convert_to_cutlass_data_type(b_dtype)
            self.b_major = b_major
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) " f"must equal num_experts ({self.expert_cnt})",
            )

        self.acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        else:
            self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size
        self.sf_fp8_dtype_override = sf_fp8_dtype_override
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.discrete_col_sfd = discrete_col_sfd
        self.use_dynamic_sched = use_dynamic_sched
        self._value_error_if(
            use_single_group_runtime_offsets and self.expert_cnt != 1,
            "use_single_group_runtime_offsets requires exactly one expert",
        )
        self.use_single_group_runtime_offsets = use_single_group_runtime_offsets
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_major = b_major

        self._interpret_uint8_as_fp4x2 = True
        self._has_bias = self.bias_desc is not None
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledMoEGroupedGemmQuantKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")
        self._workspace = None
        self._use_full_dynamic_mnkl = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"
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
        self._value_error_if(
            self.generate_sfd and not self._has_d_col,
            "sample_d_col is required when SFD outputs are generated",
        )
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

        _, _, _one = self._tensor_shape(self.d_desc, name="sample_d")

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
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
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._not_implemented_error_if(
            self._is_rubin_kernel and self.row_scale_desc is not None,
            "Rubin grouped GEMM quant does not support row_scale fusion",
        )
        if not self._is_rubin_kernel:
            self._check_tensor_shape(self.row_scale_desc, (tensor_m,), "row_scale")
        self._check_tensor_shape(self.bias_desc, (n, l), "bias")
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
            self.d_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="D must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.d_col_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="D_col must have n-major layout",
        )
        _ = self._check_tensor_stride(
            self.bias_desc,
            stride=[(1, n)],
        )
        if not self._is_rubin_kernel:
            _ = self._check_tensor_stride(
                self.row_scale_desc,
                stride=[(1,)],
                extra_error_msg="row_scale must be a contiguous 1-D tensor",
            )

        self._logger.debug("Checking data types")
        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[
                cutlass.Float4E2M1FN,
                cutlass.Uint8,
                cutlass.Float8E5M2,
                cutlass.Float8E4M3FN,
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
                dtype=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32],
                name="bias",
                extra_error_msg="bias must be fp16, bfloat16, or float32",
            )
        else:
            self._value_error_if(
                self.b_dtype != self.ab_dtype,
                f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
            )
            self._check_dtype(
                self.bias_desc,
                dtype=[cutlass.BFloat16, cutlass.Float16],
                name="bias",
                extra_error_msg="bias must be fp16 or bfloat16 in discrete mode",
            )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=[cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN],
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
            self.sf_dtype in [cutlass.Float8E4M3FN] and self.sf_vec_size == 32,
            f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )
        self._value_error_if(
            self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16,
            f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported",
        )

        # torch has no e5m3 dtype and TVM-FFI cannot marshal FloatNV8E5M3FNU, so e5m3
        # scale factors arrive as e4m3 storage of the same width and the Rubin kernel
        # reinterprets them. That reinterpretation is the only real override; every
        # other format the kernel reads straight off sfa.element_type.

        # e5m3 is the only override currently supported
        self._value_error_if(
            self.sf_fp8_dtype_override not in (None, "e5m3"),
            f"sf_fp8_dtype_override must be None or 'e5m3', got {self.sf_fp8_dtype_override!r}",
        )
        if self.sf_fp8_dtype_override == "e5m3":
            # Only allow e5m3 to pretend to be e4m3fn
            self._value_error_if(
                self.sf_dtype != cutlass.Float8E4M3FN,
                f"sf_fp8_dtype_override='e5m3' requires the NVFP4 recipe -- FP4 A/B with "
                f"torch.float8_e4m3fn scale factors at sf_vec_size 16 -- but got "
                f"ab_dtype={self.ab_dtype}, sf_dtype={self.sf_dtype}, sf_vec_size={self.sf_vec_size}",
            )
            # Only allow e5m3 for rubin kernels
            self._value_error_if(
                not self._is_rubin_kernel,
                f"sf_fp8_dtype_override='e5m3' requires Rubin (SM107), got device type {self._device_type!r}",
            )

        self._check_dtype(
            self.acc_dtype,
            dtype=cutlass.Float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )
        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.d_desc,
                dtype=[cutlass.Float16, cutlass.BFloat16, cutlass.Float32],
                name="D",
                extra_error_msg="D must be fp16, bf16, or float32 when ab_dtype is fp4",
            )
        else:
            self.d_dtype = self._check_dtype(
                self.d_desc,
                dtype=[
                    cutlass.Float16,
                    cutlass.BFloat16,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E5M2,
                    cutlass.Float4E2M1FN,
                ],
                name="D",
            )
        self._check_dtype(
            self.d_col_desc,
            dtype=self.d_dtype,
            name="D_col",
            extra_error_msg="D_col must have the same dtype as D",
        )
        if not self._is_rubin_kernel:
            self._check_dtype(
                self.row_scale_desc,
                dtype=cutlass.Float32,
                name="row_scale",
                extra_error_msg="row_scale must be float32",
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
            self.m_aligned != self._kernel.FIX_PAD_SIZE,
            f"m_aligned must be {self._kernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
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

        self._not_implemented_error_if(self._has_bias and self.mma_tiler_mn[1] != 256, "Bias fusion currently requires mma_tiler_mn[1] == 256")

        self._not_implemented_error_if(
            (self._is_fp8(self.ab_dtype)) and (self.mma_tiler_mn[1] == 128) and (self._is_fp8(self.d_dtype)),
            "Invalid configuration: fp8 ab_dtype and sf_vec_size 32 with mma_tiler_mn[1] == 128 and fp8 d_dtype is not supported. "
            "Please use mma_tiler_mn[1] == 256 instead",
        )
        if not cuda_is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = get_compute_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmQuant requires SM100+ compute capability, but found SM{compute_capability}")

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

        self._use_full_dynamic_mnkl = os.environ.get("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"

        kernel_kwargs = dict(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
            enable_bias=self._has_bias,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            **rubin_single_group_offsets_kwarg(self._is_rubin_kernel, self.use_single_group_runtime_offsets),
            # Only the Rubin kernel accepts sf_fp8_dtype_override, and check_support
            # rejects "e5m3" unless _is_rubin_kernel -- the same flag that selected
            # self._kernel. The kernel maps the string to FloatNV8E5M3FNU itself, so
            # that internal-only type is never named outside the Rubin module.
            **({"sf_fp8_dtype_override": self.sf_fp8_dtype_override} if self.sf_fp8_dtype_override == "e5m3" else {}),
        )
        if self._is_rubin_kernel:
            # The Rubin quant kernel supports optional C materialization, but
            # this cuDNN FE wrapper only exposes quantized D/D_col outputs.
            kernel_kwargs["generate_c"] = False
        gemm_quant = self._kernel(**kernel_kwargs)

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        workspace_bytes = gemm_quant.get_workspace_bytes()
        # Internal scratch in the caller's framework allocator; kernels write through its
        # raw pointer and it is never surfaced as a framework array.
        self._workspace = allocate_byte_workspace(self._framework, workspace_bytes, self.a_desc.device)

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
        use_full_dynamic = self._use_full_dynamic_mnkl

        if not use_full_dynamic:
            valid_m = cute.sym_int(divisibility=256)

            a_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=(valid_m, *self.a_desc.shape[1:]),
                stride_order=self.a_desc.stride_order,
            )
            b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
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
            sfa_shape = list(self.sfa_desc.shape)
            sfa_shape[2] = tensor_m_128
            sfa_stride = list(self.sfa_desc.stride)
            sfa_stride[5] = stride_tensor_m_128
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=tuple(sfa_shape),
                stride=tuple(sfa_stride),
            )

            sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)

            prob_cute_fake = None
            if self.prob_desc is not None:
                prob_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, *self.prob_desc.shape[1:]),
                    stride=self.prob_desc.stride,
                )
            row_scale_cute_fake = None
            if self.row_scale_desc is not None:
                row_scale_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.row_scale_desc.dtype,
                    shape=(valid_m,),
                    stride=self.row_scale_desc.stride,
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
            n_sym_divisibility = 128 // _convert_to_cutlass_data_type(self.bias_desc.dtype).width if self.bias_desc is not None else 1
            n_sym = cute.sym_int(divisibility=n_sym_divisibility)
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
            d_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_desc.dtype,
                shape=(valid_m, n_sym, 1),
                stride_order=self.d_desc.stride_order,
                dynamic_mode=self.d_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_desc.dtype) else 16,
            )
            d_col_cute_fake = self._make_fake_cute_compact_tensor(
                dtype=self.d_col_desc.dtype,
                shape=(valid_m, n_sym, 1),
                stride_order=self.d_col_desc.stride_order,
                dynamic_mode=self.d_col_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.d_col_desc.dtype) else 16,
            )

            tensor_m_128 = cute.sym_int()
            rest_k = cute.sym_int()
            stride_rest_k = cute.sym_int(divisibility=32 * 4 * 4)
            stride_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfa_shape = list(self.sfa_desc.shape)
            sfa_shape[2] = tensor_m_128
            sfa_shape[4] = rest_k
            sfa_stride = list(self.sfa_desc.stride)
            sfa_stride[2] = stride_rest_k
            sfa_stride[5] = stride_tensor_m_128
            sfa_cute_fake = self._make_fake_cute_tensor(
                dtype=self.sfa_desc.dtype,
                shape=tuple(sfa_shape),
                stride=tuple(sfa_stride),
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
                prob_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.prob_desc.dtype,
                    shape=(valid_m, *self.prob_desc.shape[1:]),
                    stride=self.prob_desc.stride,
                )
            row_scale_cute_fake = None
            if self.row_scale_desc is not None:
                row_scale_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.row_scale_desc.dtype,
                    shape=(valid_m,),
                    stride=self.row_scale_desc.stride,
                )

            sfd_row_fake = None
            sfd_col_fake = None
            if self.sfd_row_desc is not None:
                rest_n = cute.sym_int()
                stride_sfd_rest_n = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_rest_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_row_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_row_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, rest_n, 1),
                    stride=(16, 4, stride_sfd_rest_n, 1, 512, stride_sfd_rest_tensor_m_128),
                )
            if self.sfd_col_desc is not None:
                tensor_n_128 = cute.sym_int()
                rest_m_dyn = cute.sym_int()
                stride_sfd_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfd_n = cute.sym_int(divisibility=32 * 4 * 4)
                sfd_col_fake = self._make_fake_cute_tensor(
                    dtype=self.sfd_col_desc.dtype,
                    shape=(32, 4, tensor_n_128, 4, rest_m_dyn, 1),
                    stride=(16, 4, stride_sfd_rest_m, 1, 512, stride_sfd_n),
                )

            bias_cute_fake = None
            if self.bias_desc is not None:
                bias_cute_fake = self._make_fake_cute_tensor(
                    dtype=self.bias_desc.dtype,
                    shape=(n_sym, l_sym),
                    stride=(1, n_sym),
                )

        compile_kwargs = dict(
            a=a_cute_fake,
            b=b_cute_fake,
            sfb=sfb_cute_fake,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=OperandMajorMode.K,
            workspace_ptr=fake_workspace_ptr,
            d=d_cute_fake,
            d_col=d_col_cute_fake,
            sfa=sfa_cute_fake,
            sfd_row_tensor=sfd_row_fake,
            sfd_col_tensor=sfd_col_fake,
            amax_tensor=self._make_fake_cute_tensor_from_desc(self.amax_desc, assumed_align=16),
            norm_const_tensor=self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16),
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            bias=bias_cute_fake,
            prob=prob_cute_fake,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            options="--enable-tvm-ffi",
        )
        if self._is_rubin_kernel:
            compile_kwargs["c"] = d_cute_fake
            compile_kwargs["epilogue_op"] = lambda x: x
        else:
            compile_kwargs["row_scale"] = row_scale_cute_fake
        _compiled_kernel = cute.compile(gemm_quant, **compile_kwargs)

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
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
            row_scale_tensor: Optional[torch.Tensor],
            prob_tensor: Optional[torch.Tensor],
            bias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            if self._is_rubin_kernel:
                _compiled_kernel(
                    a_tensor,
                    b_tensor,
                    sfb_tensor,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int64(0),
                    cached_workspace_ptr,
                    d_tensor,
                    d_tensor,
                    d_col_tensor,
                    sfa_tensor,
                    sfd_row_tensor,
                    sfd_col_tensor,
                    amax_tensor,
                    norm_const_tensor,
                    padded_offsets,
                    alpha_tensor,
                    bias_tensor,
                    prob_tensor,
                    stream,
                )
            else:
                _compiled_kernel(
                    a_tensor,
                    b_tensor,
                    sfb_tensor,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int64(0),
                    cached_workspace_ptr,
                    d_tensor,
                    d_col_tensor,
                    sfa_tensor,
                    sfd_row_tensor,
                    sfd_col_tensor,
                    amax_tensor,
                    norm_const_tensor,
                    padded_offsets,
                    alpha_tensor,
                    row_scale_tensor,
                    bias_tensor,
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

        valid_m = cute.sym_int(divisibility=256)
        a_tensor = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, *self.a_desc.shape[1:]),
            stride_order=self.a_desc.stride_order,
            assumed_align=align,
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
        row_scale_tensor = None
        if self.row_scale_desc is not None:
            row_scale_tensor = self._make_fake_cute_tensor(
                dtype=self.row_scale_desc.dtype,
                shape=(valid_m,),
                stride=self.row_scale_desc.stride,
                assumed_align=16,
            )
        bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)

        # Compile-time placeholders for the pointer-array arguments: real device bytes
        # (fake tensors have dummy iterators) allocated in the caller's framework,
        # retyped to Int64 via the element_type override.
        self._compile_b_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        b_ptrs_placeholder = from_dlpack(self._compile_b_ptrs, assumed_align=8)
        b_ptrs_placeholder.element_type = cutlass.Int64
        b_ptrs_cute = b_ptrs_placeholder.iterator
        sfb_ptrs_placeholder = from_dlpack(self._compile_sfb_ptrs, assumed_align=8)
        sfb_ptrs_placeholder.element_type = cutlass.Int64
        sfb_ptrs_cute = sfb_ptrs_placeholder.iterator
        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        self._logger.debug("Compiling discrete grouped_gemm_quant kernel")
        compile_kwargs = dict(
            a=a_tensor,
            b=b_ptrs_cute,
            sfb=sfb_ptrs_cute,
            n=cutlass.Int32(n),
            k=cutlass.Int32(k),
            b_stride_size=cutlass.Int64(b_stride_size),
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr_cute,
            d=d_tensor,
            d_col=d_col_tensor,
            sfa=sfa_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor_cute,
            padded_offsets=padded_offsets_tensor,
            alpha=alpha_tensor,
            bias=bias_cute_fake,
            prob=prob_tensor,
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            epilogue_op=lambda x: x,
            options="--enable-tvm-ffi",
        )
        if self._is_rubin_kernel:
            compile_kwargs["c"] = d_tensor
        else:
            compile_kwargs["row_scale"] = row_scale_tensor
        _compiled_kernel = cute.compile(gemm_quant, **compile_kwargs)

        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        cached_n = cutlass.Int32(n)
        cached_k = cutlass.Int32(k)
        cached_b_stride = cutlass.Int64(b_stride_size)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_ptrs_device: torch.Tensor,
            sfb_ptrs_device: torch.Tensor,
            d_tensor: torch.Tensor,
            d_col_tensor: Optional[torch.Tensor],
            sfa_tensor: torch.Tensor,
            sfd_row_tensor: Optional[torch.Tensor],
            sfd_col_tensor: Optional[torch.Tensor],
            amax_tensor: Optional[torch.Tensor],
            norm_const_tensor: Optional[torch.Tensor],
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            row_scale_tensor: Optional[torch.Tensor],
            prob_tensor: Optional[torch.Tensor],
            bias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            b_ptrs_addr = int(get_data_ptr(b_ptrs_device))
            sfb_ptrs_addr = int(get_data_ptr(sfb_ptrs_device))
            if self._is_rubin_kernel:
                _compiled_kernel(
                    a_tensor,
                    b_ptrs_addr,
                    sfb_ptrs_addr,
                    cached_n,
                    cached_k,
                    cached_b_stride,
                    cached_workspace_ptr,
                    d_tensor,
                    d_tensor,
                    d_col_tensor,
                    sfa_tensor,
                    sfd_row_tensor,
                    sfd_col_tensor,
                    amax_tensor,
                    norm_const_tensor,
                    padded_offsets,
                    alpha_tensor,
                    bias_tensor,
                    prob_tensor,
                    stream,
                )
            else:
                _compiled_kernel(
                    a_tensor,
                    b_ptrs_addr,
                    sfb_ptrs_addr,
                    cached_n,
                    cached_k,
                    cached_b_stride,
                    cached_workspace_ptr,
                    d_tensor,
                    d_col_tensor,
                    sfa_tensor,
                    sfd_row_tensor,
                    sfd_col_tensor,
                    amax_tensor,
                    norm_const_tensor,
                    padded_offsets,
                    alpha_tensor,
                    row_scale_tensor,
                    bias_tensor,
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
        bias_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        d_col_tensor: Optional[torch.Tensor] = None,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        row_scale_tensor: Optional[torch.Tensor] = None,
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
        :param bias_tensor: Optional bias tensor with shape (n, l) and stride (1, n).
            Bias fusion is specialized at compile time: if ``sample_bias`` was omitted
            at construction, ``bias_tensor`` must also be omitted at execute time.
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param d_col_tensor: Optional column-quantized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Optional probability tensor for per-row gating. When
            omitted, the kernel compiles out the probability load and multiply.
        :param row_scale_tensor: Optional contiguous FP32 tensor of shape ``(valid_m,)``.
            When provided, the epilogue multiplies accumulators by
            ``alpha_tensor[expert] * row_scale_tensor[m]`` before output
            conversion.
        :param current_stream: CUDA stream
        """
        self._logger.debug("Entering execute")
        if current_stream is None:
            # torch inputs stay ordered with the caller's current torch stream;
            # other frameworks default to the CUDA legacy default stream.
            current_stream = default_stream(detect_framework(a_tensor))

        if a_tensor.shape[0] == 0:
            self._logger.debug("execute: valid_m is zero, skipping kernel execution")
            return
        self._runtime_error_if(
            self._compiled_kernel is None,
            "Kernel not compiled; call compile() first",
        )

        if d_col_tensor is None:
            self._value_error_if(
                self.generate_sfd,
                "d_col_tensor is required when SFD outputs are generated",
            )
            d_col_tensor = d_tensor
        if self._has_bias:
            self._value_error_if(
                bias_tensor is None,
                "bias_tensor must be provided at execute() when the API was compiled with sample_bias",
            )
        else:
            self._value_error_if(
                bias_tensor is not None,
                "bias_tensor must be omitted at execute() when the API was compiled without sample_bias",
            )
        if self._is_rubin_kernel:
            self._value_error_if(
                row_scale_tensor is not None,
                "row_scale_tensor is not supported on Rubin (sm107)",
            )
        elif self.row_scale_desc is None:
            self._value_error_if(
                row_scale_tensor is not None,
                "row_scale_tensor must be omitted at execute() when the API was compiled without sample_row_scale",
            )
        else:
            self._value_error_if(
                row_scale_tensor is None,
                "row_scale_tensor must be provided at execute() when the API was compiled with sample_row_scale",
            )

        self._logger.debug("Executing grouped_gemm_quant kernel")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
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
                row_scale_tensor=row_scale_tensor,
                prob_tensor=prob_tensor,
                bias_tensor=bias_tensor,
                stream=current_stream,
            )
        else:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_ptrs_device=b_ptrs,
                sfb_ptrs_device=sfb_ptrs,
                d_tensor=d_tensor,
                d_col_tensor=d_col_tensor,
                sfa_tensor=sfa_tensor,
                sfd_row_tensor=sfd_row_tensor,
                sfd_col_tensor=sfd_col_tensor,
                amax_tensor=amax_tensor,
                norm_const_tensor=norm_const_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                row_scale_tensor=row_scale_tensor,
                prob_tensor=prob_tensor,
                bias_tensor=bias_tensor,
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
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    norm_const_tensor: Optional[torch.Tensor] = None,
    prob_tensor: Optional[torch.Tensor] = None,
    row_scale_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    d_tensor: Optional[torch.Tensor] = None,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    use_dynamic_sched: bool = False,
    use_single_group_runtime_offsets: bool = False,
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
        prob_tensor: Optional probability tensor for per-row gating (shape
            `(valid_m, 1, 1)`). When omitted, the kernel compiles out the
            probability load and multiply.
        row_scale_tensor: Optional FP32 tensor of shape `(valid_m,)`.
            When provided, the epilogue multiplies accumulators by
            `alpha_tensor[expert] * row_scale_tensor[m]` before output
            conversion.
        acc_dtype: Accumulator data type
        d_dtype: Output D tensor data type
        d_tensor: Optional preallocated output tensor to write into instead of
            allocating. Must match the internal layout: shape (valid_m, n_out, 1),
            stride (n_out, 1, valid_m * n_out), dtype d_dtype, on a_tensor.device.
        cd_major: CD major dimension (only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        sf_fp8_dtype_override: Reinterpret the FP8-format block scale factors as
            E5M3 instead of the encoding implied by ``sfa_tensor.dtype``. ``None``
            (default) infers as usual -- E4M3 for NVFP4, E8M0 for MXFP4/MXFP8.
            ``"e5m3"`` selects an unsigned 5-exponent-bit, 3-mantissa-bit format
            that trades two mantissa bits for one exponent bit to widen the scale
            range; it is Rubin-only, requires the NVFP4 recipe, and the scale
            tensors are still passed as ``torch.float8_e4m3fn`` because torch has
            no e5m3 dtype.
        vector_f32: Use vectorized f32
        m_aligned: M alignment (must be 256)
        discrete_col_sfd: Enable discrete col-major scale factor tensor
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **d_tensor** (torch.Tensor): Final output tensor
            - **d_col_tensor** (torch.Tensor or None): Column-wise output tensor for low-precision D output
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
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    framework = detect_framework(a_tensor)
    if framework == "jax":
        raise ValueError(f"grouped_gemm_quant_wrapper_sm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_quant_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
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

    _logger.debug("grouped_gemm_quant_wrapper_sm100: Creating output tensors")

    if cd_major == "n":
        expected_shape = (valid_m, n_out, 1)
        expected_stride = (n_out, 1, valid_m * n_out)
        if d_tensor is None:
            d_tensor = torch.empty_strided(expected_shape, expected_stride, dtype=framework_dtype(d_dtype, "torch"), device=a_tensor.device)
        elif (
            tuple(d_tensor.shape) != expected_shape
            or tuple(d_tensor.stride()) != expected_stride
            or _convert_to_cutlass_data_type(d_tensor.dtype) != d_dtype
            or get_device(d_tensor) != get_device(a_tensor)
        ):
            raise ValueError(
                f"d_tensor must have shape {expected_shape}, stride {expected_stride}, "
                f"dtype {d_dtype}, device {a_tensor.device}, but got shape {tuple(d_tensor.shape)}, "
                f"stride {tuple(d_tensor.stride())}, dtype {d_tensor.dtype}, device {d_tensor.device}."
            )
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

    if d_dtype in (cutlass.BFloat16, cutlass.Float16):
        _logger.debug("grouped_gemm_quant_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    device_type = get_device_type()
    if row_scale_tensor is not None:
        if device_type == "rubin":
            raise NotImplementedError("Rubin grouped GEMM quant does not support row_scale fusion")
        if _convert_to_cutlass_data_type(row_scale_tensor.dtype) is not cutlass.Float32:
            raise ValueError(f"row_scale_tensor must be float32, got {row_scale_tensor.dtype}")
        if tuple(row_scale_tensor.shape) != (valid_m,):
            raise ValueError(f"row_scale_tensor must have shape {(valid_m,)}, got {tuple(row_scale_tensor.shape)}")
        if tuple(row_scale_tensor.stride()) != (1,):
            raise ValueError(f"row_scale_tensor must be contiguous with stride (1,), got {tuple(row_scale_tensor.stride())}")

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
            device_type,
            weight_mode,
            use_full_dynamic,
            a_tensor.shape[1:] if not use_full_dynamic else None,
            b_tensor.shape[2] if use_full_dynamic else tuple(b_tensor.shape),
            a_tensor.dtype,
            b_tensor.dtype,
            stride_order(a_tensor),
            stride_order(b_tensor),
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
            *dynamic_m_tensor_signature(row_scale_tensor, ()),
            tuple(padded_offsets.shape),
            tuple(padded_offsets.stride()),
            padded_offsets.dtype,
            acc_dtype,
            d_dtype,
            cd_major,
            mma_tiler_mn,
            cluster_shape_mn,
            sf_vec_size,
            sf_fp8_dtype_override,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_single_group_runtime_offsets,
        )
    else:
        cache_key = (
            device_type,
            weight_mode,
            a_tensor.shape[1:],
            stride_order(a_tensor),
            a_tensor.dtype,
            b_shape,
            b_dtype,
            d_tensor.shape[1:],
            stride_order(d_tensor),
            *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1) if sfa_tensor is not None else None, dynamic_stride_dims=(5,)),
            *tensor_signature(bias_tensor),
            *tensor_signature(alpha_tensor),
            *tensor_signature(norm_const_tensor),
            *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
            *dynamic_m_tensor_signature(row_scale_tensor, ()),
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
            sf_fp8_dtype_override,
            vector_f32,
            m_aligned,
            discrete_col_sfd,
            use_dynamic_sched,
            use_single_group_runtime_offsets,
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
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                sample_row_scale=row_scale_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
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
                sample_bias=bias_tensor,
                sample_amax=amax_tensor,
                sample_sfd_row=sfd_row_tensor,
                sample_sfd_col=sfd_col_tensor,
                sample_norm_const=norm_const_tensor,
                sample_prob=prob_tensor,
                sample_row_scale=row_scale_tensor,
                acc_dtype=acc_dtype,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                discrete_col_sfd=discrete_col_sfd,
                use_dynamic_sched=use_dynamic_sched,
                use_single_group_runtime_offsets=use_single_group_runtime_offsets,
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
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            row_scale_tensor=row_scale_tensor,
            bias_tensor=bias_tensor,
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
            d_col_tensor=d_col_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            prob_tensor=prob_tensor,
            row_scale_tensor=row_scale_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )

    return TupleDict(
        d_tensor=d_tensor,
        d_col_tensor=d_col_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )
