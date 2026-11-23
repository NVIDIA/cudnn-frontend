# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Block-scaled implementation of the grouped GEMM SReLU API (SM100+).

Private module: callers use :mod:`cudnn.gemm.cutedsl.grouped.srelu.api`.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, ceil_div, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    allocate_byte_workspace,
    cuda_is_available,
    default_stream,
    detect_framework,
    get_compute_capability,
    has_tcgen05,
    get_data_ptr,
)

from .moe_blockscaled_grouped_gemm_srelu_quant import (
    BlockScaledMoEGroupedGemmQuantKernel,
    EpilogueType,
)
from ..moe_utils import MoEWeightMode
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack

_JAX_SF_LAYOUT_ERROR = (
    "the block scale-factor tensors (sfa/sfb and the sfd outputs) are MMA-tiled "
    "(32, 4, m//128, 4, rest_k, l) strided views that are not expressible as JAX arrays "
    "(a row-major JAX array of that shape has different memory); pass torch tensors"
)


def _reinterpret_raw_grouped_fp4_tensor(tensor: torch.Tensor) -> torch.Tensor:
    import torch

    if tensor.dtype == torch.uint8:
        cute_tensor = from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)
        cute_tensor.element_type = cutlass.Float4E2M1FN
        return cute_tensor
    return tensor


class GroupedGemmSreluBlockScaledAPI(APIBase):
    """Unified API for grouped GEMM SReLU operation on SM100+ GPUs.

    This kernel performs block-scaled grouped GEMM with output SReLU output quantization
    (D = srelu(alpha * A @ B)), designed for MoE workloads. It supports both
    dense (contiguous) and discrete (per-expert pointer) weight layouts
    through ``BlockScaledMoEGroupedGemmQuantKernel``.

    Weight mode is auto-detected from the constructor arguments:

    - Dense: provide ``sample_b`` and ``sample_sfb``.
    - Discrete: provide ``num_experts``, ``b_shape``, and ``b_dtype``.
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        # Dense mode (contiguous) -- provide these:
        sample_b: Optional[torch.Tensor] = None,
        sample_c: Optional[torch.Tensor] = None,
        sample_d: Optional[torch.Tensor] = None,
        sample_sfa: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_padded_offsets: Optional[torch.Tensor] = None,
        sample_alpha: Optional[torch.Tensor] = None,
        sample_d_col: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Optional SReLU output quantization output arguments
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
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        tanh_clamp_scale: Optional[float] = None,
    ):
        """Initialize the GroupedGemmSreluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_sfa: Sample scale factor A tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_c: Sample C output tensor (valid_m, n, 1) before SReLU
        :param sample_d: Sample D output tensor (valid_m, n, 1)
        :param sample_d_col: Optional column-sreluized D tensor. Required only when SFD outputs are generated.
        :param sample_b: (Dense) Sample B tensor (n, k, l)
        :param sample_sfb: (Dense) Sample scale factor B tensor
        :param sample_bias: Optional bias tensor with shape (n, l) or (n, expert_cnt), stride (1, n).
            Dense mode supports fp16/bfloat16/float32 bias; discrete mode supports fp16/bfloat16 bias.
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k)
        :param b_dtype: (Discrete) Data type of B tensors
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for SReLU output quantization
        :param sample_norm_const: Optional normalization constant
        :param sample_prob: Optional probability tensor for gating
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Enable discrete col-major scale factor tensor
        :param b_major: Major dimension for B tensor, one of "k" or "n"
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        :param tanh_clamp_scale: Optional soft-clamp scale ``s``. When set, computes
            ``(s * tanh(relu(x) / s)) ** 2`` instead of plain ``relu(x) ** 2``; must be
            finite and > 0. ``None`` (default) is bit-identical to today's SReLU.
        """
        framework = detect_framework(sample_a)
        if framework == "jax":
            raise ValueError(f"GroupedGemmSreluSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        if framework != "torch":
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmSreluSm100; pass torch tensors")
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

        self._sample_a_tensor = sample_a
        self._sample_b_tensor = sample_b

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", interpret_uint8_as_fp4x2=False, canonical=True)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c", canonical=True)
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
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias", canonical=True)

        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", interpret_uint8_as_fp4x2=False, canonical=True)
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
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.discrete_col_sfd = discrete_col_sfd
        self.use_dynamic_sched = use_dynamic_sched
        if tanh_clamp_scale is not None:
            self._value_error_if(
                not math.isfinite(tanh_clamp_scale) or tanh_clamp_scale <= 0.0,
                f"tanh_clamp_scale must be finite and positive, got {tanh_clamp_scale}",
            )
        self.tanh_clamp_scale = tanh_clamp_scale
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_major = b_major

        self._interpret_uint8_as_fp4x2 = True
        self._has_bias = self.bias_desc is not None
        self._kernel = BlockScaledMoEGroupedGemmQuantKernel

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
        _ = self._check_tensor_stride(
            self.bias_desc,
            stride=[(1, n)],
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

        self._check_dtype(
            self.acc_dtype,
            dtype=cutlass.Float32,
            name="Accumulator",
            extra_error_msg="Accumulator must be float32",
        )
        self.c_dtype = self._check_dtype(
            self.c_desc,
            dtype=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2],
            name="C",
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

        self._not_implemented_error_if(
            self._is_fp4x2(self.ab_dtype) and self.sf_vec_size == 16 and self.d_dtype is cutlass.Float32,
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
            "Invalid cluster shape: expected values to be powers of 2 and " f"cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn}",
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
        if not has_tcgen05(major):
            raise RuntimeError(f"GroupedGemmSrelu requires SM100+ compute capability with tcgen05 (not cc 12.x), but found SM{compute_capability}")

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

        gemm_srelu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            generate_sfd=self.generate_sfd,
            discrete_col_sfd=self.discrete_col_sfd,
            generate_c=True,
            enable_bias=self._has_bias,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            epilogue_type=EpilogueType.SRELU.value,
            tanh_clamp_scale=self.tanh_clamp_scale,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        workspace_bytes = gemm_srelu.get_workspace_bytes()
        # Internal scratch in the caller's framework allocator; kernels write through its
        # raw pointer and it is never surfaced as a framework array.
        self._workspace = allocate_byte_workspace(self._framework, workspace_bytes, self.a_desc.device)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(gemm_srelu, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(gemm_srelu, max_active_clusters, fake_stream)

        self._logger.debug("Kernel compiled successfully")

    def _compile_dense(self, gemm_srelu, max_active_clusters, fake_stream) -> None:
        """Compile for dense (contiguous) weight mode."""
        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        self._logger.debug("Compiling grouped_gemm_srelu kernel")
        use_full_dynamic = self._use_full_dynamic_mnkl

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
                shape=(valid_m, n_sym, 1),
                stride_order=self.c_desc.stride_order,
                dynamic_mode=self.c_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.c_desc.dtype) else 16,
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

        _compiled_kernel = cute.compile(
            gemm_srelu,
            a=_reinterpret_raw_grouped_fp4_tensor(self._sample_a_tensor) if self.a_desc.dtype is cutlass.Uint8 else a_cute_fake,
            b=_reinterpret_raw_grouped_fp4_tensor(self._sample_b_tensor) if self.b_desc.dtype is cutlass.Uint8 else b_cute_fake,
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
            bias=bias_cute_fake,
            prob=prob_cute_fake,
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
            bias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            _compiled_kernel(
                _reinterpret_raw_grouped_fp4_tensor(a_tensor),
                _reinterpret_raw_grouped_fp4_tensor(b_tensor),
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
                bias_tensor,
                prob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, gemm_srelu, max_active_clusters, fake_stream) -> None:
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
        prob_tensor = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, *self.prob_desc.shape[1:]),
            stride=self.prob_desc.stride,
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

        self._logger.debug("Compiling discrete grouped_gemm_srelu kernel")
        _compiled_kernel = cute.compile(
            gemm_srelu,
            a=a_tensor,
            b=b_ptrs_cute,
            sfb=sfb_ptrs_cute,
            n=cutlass.Int32(n),
            k=cutlass.Int32(k),
            b_stride_size=cutlass.Int64(b_stride_size),
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr_cute,
            c=c_tensor,
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
            bias_tensor: Optional[torch.Tensor],
            stream: cuda.CUstream,
        ) -> None:
            norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")
            b_ptrs_addr = int(get_data_ptr(b_ptrs_device))
            sfb_ptrs_addr = int(get_data_ptr(sfb_ptrs_device))
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
        c_tensor: torch.Tensor,
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
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor
        :param sfa_tensor: Scale factor A
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param c_tensor: Output C tensor before SReLU
        :param d_tensor: Output D tensor
        :param b_tensor: (Dense) Input B tensor (weights)
        :param sfb_tensor: (Dense) Scale factor B
        :param bias_tensor: Optional bias tensor with shape (n, l) and stride (1, n).
            Bias fusion is specialized at compile time: if ``sample_bias`` was omitted
            at construction, ``bias_tensor`` must also be omitted at execute time.
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param d_col_tensor: Optional column-sreluized output
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param prob_tensor: Probability tensor for per-row gating. Required.
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
        self._value_error_if(
            prob_tensor is None,
            "prob_tensor is required: the kernel unconditionally multiplies output by per-row gating probability. "
            "Pass a tensor of ones with shape (valid_m, 1, 1) if no gating is needed.",
        )
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

        self._logger.debug("Executing grouped_gemm_srelu kernel")
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
                bias_tensor=bias_tensor,
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
                bias_tensor=bias_tensor,
                stream=current_stream,
            )

        self._logger.debug("Execute completed")
