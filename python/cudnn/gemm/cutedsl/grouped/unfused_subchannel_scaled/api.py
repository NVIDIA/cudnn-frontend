# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
API for the Unfused Subchannel-Scaled Grouped GEMM Kernel (SM100+)

Second-level-scaled ("subchannel-scaled") unfused block-scaled grouped GEMM for
MoE workloads: NVFP4 A/B inputs carry two scale levels -- per-(1, 16) FP8 block
scale factors (SFA/SFB) plus FP32 second-level scales (SFA2/SFB2) at
``block2_shape = (sgm, sgn, sgk)`` granularity -- and the BF16 output is
``D = (second-level-scaled GEMM) * alpha * prob`` (plus ``prob * bias`` when a
bias is fused). There is no output quantization: this is the unfused GEMM-only
counterpart of the fused dGLU kernels.

Supports both dense (contiguous) and discrete (per-expert pointer) weight
modes through ``BlockScaledSubChannelMoEGroupedGemmKernel``.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.api_base import APIBase, TupleDict, ceil_div, get_device_type, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.tensor_adapter import (
    allocate_byte_workspace,
    canonicalize_unit_dim_strides,
    cuda_is_available,
    default_stream,
    detect_framework,
    framework_dtype,
    get_compute_capability,
    get_data_ptr,
    get_device,
)

from .grouped_gemm_unfused_subchannel_scaled import (
    BlockScaledSubChannelMoEGroupedGemmKernel,
)
from ..moe_utils import MoEWeightMode
from cutlass.cute.nvgpu import OperandMajorMode

_JAX_SF_LAYOUT_ERROR = (
    "the block scale-factor tensors (sfa/sfb) are MMA-tiled "
    "(32, 4, m//128, 4, rest_k, l) strided views that are not expressible as JAX arrays "
    "(a row-major JAX array of that shape has different memory); pass torch tensors"
)


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_unfused_subchannel_scaled_rubin import (
        BlockScaledSubChannelMoEGroupedGemmKernelSm107,
    )

    return BlockScaledSubChannelMoEGroupedGemmKernelSm107


class GroupedGemmUnfusedSubchannelScaledSm100(APIBase):
    """API for the unfused subchannel-scaled grouped GEMM operation on SM100+ GPUs.

    This kernel performs a second-level-scaled block-scaled grouped GEMM
    (``D = (sum over sgk-tiles of sfa2 * sfb2 * (A_fp4 @ B_fp4^T)) * alpha * prob``,
    optionally ``+ prob * bias``) for MoE workloads. Inputs are NVFP4 with FP8
    first-level scale factors and FP32 second-level (subchannel) scale factors;
    the output D is BF16. Both dense (contiguous) and discrete (per-expert
    pointer) weight layouts are supported through
    ``BlockScaledSubChannelMoEGroupedGemmKernel``.

    Weight mode is auto-detected from the constructor arguments:

    - Dense: provide ``sample_b``, ``sample_sfb``, and ``sample_sfb2``.
    - Discrete: provide ``num_experts``, ``b_shape``, and ``b_dtype``.
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfa2: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_d: torch.Tensor,
        # Dense mode (contiguous) -- provide these:
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_sfb2: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        block2_shape: Tuple[int, int, int] = (1, 256, 256),
        mma_tiler_mn: Tuple[int, int] = (256, 128),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        vector_f32: bool = True,
        m_aligned: int = 256,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ):
        """Initialize the GroupedGemmUnfusedSubchannelScaledSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1), FP4, k-major
        :param sample_sfa: Sample first-level scale factor A tensor
            (MMA-tiled (32, 4, valid_m//128, 4, rest_k, 1) view)
        :param sample_sfa2: Sample second-level scale factor A tensor,
            shape (ceil(valid_m / sgm), ceil(k / sgk), 1), FP32,
            stride (1, rows, rows * cols) (rows-contiguous)
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors, shape (expert_cnt,)
        :param sample_prob: Per-token probability tensor (valid_m, 1, 1), FP32.
            Required: the kernel always fuses the prob multiply. Pass ones for
            a plain GEMM.
        :param sample_d: Sample D output tensor (valid_m, n, 1), BF16, n-major
        :param sample_b: (Dense) Sample B tensor (n, k, l), FP4, k-major
        :param sample_sfb: (Dense) Sample first-level scale factor B tensor
        :param sample_sfb2: (Dense) Sample second-level scale factor B tensor,
            shape (ceil(n / sgn), ceil(k / sgk), l), FP32,
            stride (1, rows, rows * cols)
        :param sample_bias: Optional bias tensor with shape (n, l), stride (1, n), BF16.
            The epilogue computes ``d = gemm * alpha + prob * bias`` semantics
            (prob scales the bias term).
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, e.g. (n, k) with logical k
        :param b_dtype: (Discrete) Data type of B tensors
        :param acc_dtype: Accumulator data type (must be float32)
        :param block2_shape: Second-level scale granularity (sgm, sgn, sgk):
            SFA2 blocks cover (sgm rows x sgk k), SFB2 blocks (sgn cols x sgk k)
        :param mma_tiler_mn: MMA tiler shape (M, N); (256, 128) or (256, 256)
        :param cluster_shape_mn: Cluster shape (M, N); default (2, 1)
        :param sf_vec_size: First-level scale factor vector size (must be 16)
        :param sf_fp8_dtype_override: Reinterpret the FP8-format block scale factors
            as E5M3 instead of the E4M3 implied by their storage dtype. Rubin-only;
            the scale tensors are still supplied as ``torch.float8_e4m3fn`` because
            torch has no e5m3 dtype -- only the CuTe element type is overridden.
        :param vector_f32: Use vectorized (packed) f32 arithmetic. The default True
            is the byte-exact-verified configuration.
        :param m_aligned: Alignment for group M dimension (must be 256)
        :param b_major: Major dimension for B tensor (must be "k" for FP4)
        :param use_dynamic_sched: Enable dynamic tile scheduling for load balancing
        """
        framework = detect_framework(sample_a)
        if framework == "jax":
            raise ValueError(f"GroupedGemmUnfusedSubchannelScaledSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        if framework != "torch":
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmUnfusedSubchannelScaledSm100; pass torch tensors")
        if acc_dtype is None:
            acc_dtype = cutlass.Float32
        super().__init__()
        self._framework = framework

        self._warn_experimental_api()
        self._logger.debug("Entering __init__")

        # ---- Weight mode auto-detection ----
        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
            if sample_sfb is None or sample_sfb2 is None:
                raise ValueError("sample_sfb and sample_sfb2 are required when sample_b is provided (dense mode)")
        elif num_experts is not None and sample_b is None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError(
                "Provide either (sample_b, sample_sfb, sample_sfb2) for dense mode " "or (num_experts, b_shape, b_dtype) for discrete mode, but not both."
            )

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", canonical=True)
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d", canonical=True)
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa", canonical=True)
        self.sfa2_desc = self._make_tensor_desc(sample_sfa2, name="sample_sfa2", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha", canonical=True)
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias", canonical=True)

        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", canonical=True)
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb", canonical=True)
            self.sfb2_desc = self._make_tensor_desc(sample_sfb2, name="sample_sfb2", canonical=True)
            self.expert_cnt = self.padded_offsets_desc.shape[0]
        else:
            self._value_error_if(num_experts == 0, "num_experts must be > 0")
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = _convert_to_cutlass_data_type(b_dtype)
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) " f"must equal num_experts ({self.expert_cnt})",
            )

        self.acc_dtype = _convert_to_cutlass_data_type(acc_dtype)
        self._value_error_if(
            len(block2_shape) != 3,
            f"block2_shape must be (sgm, sgn, sgk), got {block2_shape}",
        )
        self.block2_shape = tuple(block2_shape)
        self.sgm, self.sgn, self.sgk = self.block2_shape
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
        self.use_dynamic_sched = use_dynamic_sched
        self.b_major = b_major

        self._interpret_uint8_as_fp4x2 = True
        self._has_bias = self.bias_desc is not None
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledSubChannelMoEGroupedGemmKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._logger.debug(f"setting num_cluster_overlap_margin: {self.num_cluster_overlap_margin}")
        self._workspace = None
        self._logger.debug("__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        self._value_error_if(
            self.prob_desc is None,
            "sample_prob is required: the kernel always fuses the per-token prob multiply " "(pass a tensor of ones for a plain GEMM)",
        )

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

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.d_desc, (tensor_m, n, 1), "D")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")

        # ---- Second-level (subchannel) scale factors ----
        self._value_error_if(
            self.sgm < 1 or self.sgn < 1 or self.sgk < 1,
            f"block2_shape components must be >= 1, got {self.block2_shape}",
        )
        self._value_error_if(
            k % self.sgk != 0,
            f"k ({k}) must be divisible by sgk ({self.sgk}): the second-level " "K loop assumes whole sgk-tiles",
        )
        self._value_error_if(
            tensor_m % self.sgm != 0,
            f"valid_m ({tensor_m}) must be divisible by sgm ({self.sgm})",
        )
        sfa2_rows = ceil_div(tensor_m, self.sgm)
        sf2_cols = ceil_div(k, self.sgk)
        self._check_tensor_shape(self.sfa2_desc, (sfa2_rows, sf2_cols, 1), "SFA2")
        # Descriptors canonicalize the strides of extent-1 dims (unobservable by
        # the kernel), so compare against the canonicalized expected stride.
        sfa2_shape = (sfa2_rows, sf2_cols, 1)
        _ = self._check_tensor_stride(
            self.sfa2_desc,
            stride=[canonicalize_unit_dim_strides(sfa2_shape, (1, sfa2_rows, sfa2_rows * sf2_cols))],
            extra_error_msg=f"SFA2 must be rows-contiguous with stride (1, {sfa2_rows}, {sfa2_rows * sf2_cols})",
        )
        sfb2_rows = ceil_div(n, self.sgn)
        if self.weight_mode == MoEWeightMode.DENSE:
            sfb2_shape = (sfb2_rows, sf2_cols, l)
            self._check_tensor_shape(self.sfb2_desc, sfb2_shape, "SFB2")
            _ = self._check_tensor_stride(
                self.sfb2_desc,
                stride=[canonicalize_unit_dim_strides(sfb2_shape, (1, sfb2_rows, sfb2_rows * sf2_cols))],
                extra_error_msg="SFB2 must be rows-contiguous per expert",
            )
        else:
            # Discrete SFB2 per-expert base pointers are declared 16B-aligned in
            # the sched extension (assumed_align=16). With back-to-back per-expert
            # f32 (ceil(n/sgn), ceil(k/sgk)) blocks, every base past the first is
            # misaligned unless the block byte size is a multiple of 16.
            sfb2_block_bytes = sfb2_rows * sf2_cols * 4
            self._value_error_if(
                self.expert_cnt > 1 and sfb2_block_bytes % 16 != 0,
                f"discrete SFB2 per-expert block ({sfb2_rows} x {sf2_cols} f32 = "
                f"{sfb2_block_bytes} bytes) must be a multiple of 16 bytes when "
                f"num_experts > 1 (per-expert bases are assumed 16B-aligned)",
            )

        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.bias_desc, (n, l), "bias")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        _ = self._check_tensor_stride(
            self.a_desc,
            stride=[(k, 1, tensor_m * k)],
            extra_error_msg="A must have k-major layout",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            _ = self._check_tensor_stride(
                self.b_desc,
                stride=[(k, 1, n * k)],
                extra_error_msg="B must have k-major layout (required for fp4 ab_dtype)",
            )
        _ = self._check_tensor_stride(
            self.d_desc,
            stride=[(n, 1, tensor_m * n)],
            extra_error_msg="D must have n-major layout",
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
            ],
            name="A/B",
            extra_error_msg="only NVFP4 (fp4) inputs are supported",
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
            self._value_error_if(
                self.b_major != "k",
                f"b_major must be 'k' for fp4 ab_dtype, got {self.b_major}",
            )
        self._check_dtype(
            self.bias_desc,
            dtype=cutlass.BFloat16,
            name="bias",
            extra_error_msg="bias must be bfloat16",
        )

        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=cutlass.Float8E4M3FN,
            name="SFA/SFB",
            extra_error_msg="first-level scale factors must be torch.float8_e4m3fn (the NVFP4 recipe)",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.sfb_desc,
                dtype=self.sf_dtype,
                name="SFB",
                extra_error_msg="SFB must have the same dtype as SFA",
            )
        self._check_dtype(
            self.sfa2_desc,
            dtype=cutlass.Float32,
            name="SFA2",
            extra_error_msg="second-level scale factors must be float32",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(
                self.sfb2_desc,
                dtype=cutlass.Float32,
                name="SFB2",
                extra_error_msg="second-level scale factors must be float32",
            )

        self._value_error_if(
            self.sf_vec_size != 16,
            f"sf_vec_size must be 16 (NVFP4), got {self.sf_vec_size}",
        )

        # e5m3 is the only override currently supported; torch has no e5m3
        # dtype, so e5m3 scale factors arrive as e4m3 storage and the Rubin
        # kernel reinterprets the CuTe element type.
        self._value_error_if(
            self.sf_fp8_dtype_override not in (None, "e5m3"),
            f"sf_fp8_dtype_override must be None or 'e5m3', got {self.sf_fp8_dtype_override!r}",
        )
        if self.sf_fp8_dtype_override == "e5m3":
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
        self.d_dtype = self._check_dtype(
            self.d_desc,
            dtype=cutlass.BFloat16,
            name="D",
            extra_error_msg="D must be bfloat16",
        )
        self._check_dtype(
            self.alpha_desc,
            dtype=cutlass.Float32,
            name="alpha",
        )
        self._check_dtype(
            self.prob_desc,
            dtype=cutlass.Float32,
            name="prob",
        )
        self._check_dtype(
            self.padded_offsets_desc,
            dtype=cutlass.Int32,
            name="padded_offsets",
        )

        self._logger.debug("Checking MMA tile shape and cluster shape")
        self._value_error_if(
            self.mma_tiler_mn not in ((256, 128), (256, 256)),
            f"mma_tiler_mn must be (256, 128) or (256, 256), got {self.mma_tiler_mn}",
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
        self._value_error_if(
            n % 64 != 0,
            f"n must be divisible by 64, got {n}",
        )
        self._value_error_if(
            tensor_m % 256 != 0,
            f"valid_m must be divisible by 256 (FIX_PAD_SIZE-aligned per-expert groups), got {tensor_m}",
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
            b_stride_order_for_check = (1, 0, 2)
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

        if not cuda_is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = get_compute_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmUnfusedSubchannelScaled requires SM100+ compute capability, but found SM{compute_capability}")

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

        kernel_kwargs = dict(
            sf_vec_size=self.sf_vec_size,
            sgm=self.sgm,
            sgn=self.sgn,
            sgk=self.sgk,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            enable_bias=self._has_bias,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            # Only the Rubin kernel accepts sf_fp8_dtype_override, and check_support
            # rejects "e5m3" unless _is_rubin_kernel -- the same flag that selected
            # self._kernel. The kernel maps the string to FloatNV8E5M3FNU itself, so
            # that internal-only type is never named outside the Rubin module.
            **({"sf_fp8_dtype_override": self.sf_fp8_dtype_override} if self.sf_fp8_dtype_override == "e5m3" else {}),
        )
        gemm = self._kernel(**kernel_kwargs)

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        workspace_bytes = gemm.get_workspace_bytes()
        # Internal scratch in the caller's framework allocator; kernels write through its
        # raw pointer and it is never surfaced as a framework array.
        self._workspace = allocate_byte_workspace(self._framework, workspace_bytes, self.a_desc.device)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(gemm, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(gemm, max_active_clusters, fake_stream)

        self._logger.debug("Kernel compiled successfully")

    def _make_dynamic_m_fakes(self, valid_m):
        """Fake tensors for the operands whose leading extent scales with valid_m."""
        a_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, *self.a_desc.shape[1:]),
            stride_order=self.a_desc.stride_order,
            assumed_align=32 if self._is_fp4x2(self.ab_dtype) else 16,
        )
        d_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.d_desc.dtype,
            shape=(valid_m, *self.d_desc.shape[1:]),
            stride_order=self.d_desc.stride_order,
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
            assumed_align=16,
        )

        # SFA2 rows scale with valid_m (rows = valid_m / sgm); the kernel reads
        # shape[0]/shape[1] and strides off the plain rank-3 tensor and rebuilds
        # its hierarchical ((sgm, rows), (sgk, cols), 1) view internally.
        sfa2_rows = cute.sym_int()
        sfa2_stride_1 = cute.sym_int()
        sfa2_stride_2 = cute.sym_int()
        sfa2_cute_fake = self._make_fake_cute_tensor(
            dtype=self.sfa2_desc.dtype,
            shape=(sfa2_rows, self.sfa2_desc.shape[1], 1),
            stride=(1, sfa2_stride_1, sfa2_stride_2),
            assumed_align=16,
        )

        prob_cute_fake = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, *self.prob_desc.shape[1:]),
            stride=self.prob_desc.stride,
            assumed_align=16,
        )
        return a_cute_fake, d_cute_fake, sfa_cute_fake, sfa2_cute_fake, prob_cute_fake

    def _compile_dense(self, gemm, max_active_clusters, fake_stream) -> None:
        """Compile for dense (contiguous) weight mode."""
        fake_workspace_ptr = cute.runtime.nullptr(
            dtype=cutlass.Uint8,
            assumed_align=128,
        )

        self._logger.debug("Compiling grouped_gemm_unfused_subchannel_scaled kernel (dense)")

        valid_m = cute.sym_int(divisibility=256)
        a_cute_fake, d_cute_fake, sfa_cute_fake, sfa2_cute_fake, prob_cute_fake = self._make_dynamic_m_fakes(valid_m)

        b_cute_fake = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
        sfb_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
        sfb2_cute_fake = self._make_fake_cute_tensor_from_desc(self.sfb2_desc, assumed_align=16)
        bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)

        _compiled_kernel = cute.compile(
            gemm,
            a=a_cute_fake,
            b=b_cute_fake,
            sfb=sfb_cute_fake,
            sfb2=sfb2_cute_fake,
            n=cutlass.Int32(0),
            k=cutlass.Int32(0),
            b_stride_size=cutlass.Int64(0),
            b_major_mode=OperandMajorMode.K,
            workspace_ptr=fake_workspace_ptr,
            d=d_cute_fake,
            sfa=sfa_cute_fake,
            sfa2=sfa2_cute_fake,
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
            sfb_tensor: torch.Tensor,
            sfb2_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfa2_tensor: torch.Tensor,
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            bias_tensor: Optional[torch.Tensor],
            prob_tensor: torch.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            _compiled_kernel(
                a_tensor,
                b_tensor,
                sfb_tensor,
                sfb2_tensor,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int64(0),
                cached_workspace_ptr,
                d_tensor,
                sfa_tensor,
                sfa2_tensor,
                padded_offsets,
                alpha_tensor,
                bias_tensor,
                prob_tensor,
                stream,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, gemm, max_active_clusters, fake_stream) -> None:
        """Compile for discrete (per-expert pointer) weight mode."""
        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape

        b_major_mode = OperandMajorMode.K
        b_stride_size = k

        self._logger.debug("Compiling grouped_gemm_unfused_subchannel_scaled kernel (discrete)")

        valid_m = cute.sym_int(divisibility=256)
        a_cute_fake, d_cute_fake, sfa_cute_fake, sfa2_cute_fake, prob_cute_fake = self._make_dynamic_m_fakes(valid_m)

        bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)

        # Compile-time placeholders for the pointer-array arguments: real device bytes
        # (fake tensors have dummy iterators) allocated in the caller's framework,
        # retyped to Int64 via the element_type override.
        self._compile_b_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb2_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        b_ptrs_placeholder = from_dlpack(self._compile_b_ptrs, assumed_align=8)
        b_ptrs_placeholder.element_type = cutlass.Int64
        b_ptrs_cute = b_ptrs_placeholder.iterator
        sfb_ptrs_placeholder = from_dlpack(self._compile_sfb_ptrs, assumed_align=8)
        sfb_ptrs_placeholder.element_type = cutlass.Int64
        sfb_ptrs_cute = sfb_ptrs_placeholder.iterator
        sfb2_ptrs_placeholder = from_dlpack(self._compile_sfb2_ptrs, assumed_align=8)
        sfb2_ptrs_placeholder.element_type = cutlass.Int64
        sfb2_ptrs_cute = sfb2_ptrs_placeholder.iterator
        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        _compiled_kernel = cute.compile(
            gemm,
            a=a_cute_fake,
            b=b_ptrs_cute,
            sfb=sfb_ptrs_cute,
            sfb2=sfb2_ptrs_cute,
            n=cutlass.Int32(n),
            k=cutlass.Int32(k),
            b_stride_size=cutlass.Int64(b_stride_size),
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr_cute,
            d=d_cute_fake,
            sfa=sfa_cute_fake,
            sfa2=sfa2_cute_fake,
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16),
            bias=bias_cute_fake,
            prob=prob_cute_fake,
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
            sfb2_ptrs_device: torch.Tensor,
            d_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfa2_tensor: torch.Tensor,
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            bias_tensor: Optional[torch.Tensor],
            prob_tensor: torch.Tensor,
            stream: cuda.CUstream,
        ) -> None:
            b_ptrs_addr = int(get_data_ptr(b_ptrs_device))
            sfb_ptrs_addr = int(get_data_ptr(sfb_ptrs_device))
            sfb2_ptrs_addr = int(get_data_ptr(sfb2_ptrs_device))
            _compiled_kernel(
                a_tensor,
                b_ptrs_addr,
                sfb_ptrs_addr,
                sfb2_ptrs_addr,
                cached_n,
                cached_k,
                cached_b_stride,
                cached_workspace_ptr,
                d_tensor,
                sfa_tensor,
                sfa2_tensor,
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
        sfa2_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        sfb2_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        sfb2_ptrs: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor (valid_m, k, 1), FP4, k-major
        :param sfa_tensor: First-level scale factor A (MMA-tiled view)
        :param sfa2_tensor: Second-level scale factor A,
            (ceil(valid_m / sgm), ceil(k / sgk), 1) FP32, stride (1, rows, rows * cols)
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group scaling factors
        :param prob_tensor: Per-token probability tensor (valid_m, 1, 1), FP32. Required.
        :param d_tensor: Output D tensor (valid_m, n, 1), BF16
        :param b_tensor: (Dense) Input B tensor (weights)
        :param sfb_tensor: (Dense) First-level scale factor B
        :param sfb2_tensor: (Dense) Second-level scale factor B,
            (ceil(n / sgn), ceil(k / sgk), l) FP32, stride (1, rows, rows * cols)
        :param bias_tensor: Optional bias tensor with shape (n, l) and stride (1, n).
            Bias fusion is specialized at compile time: if ``sample_bias`` was omitted
            at construction, ``bias_tensor`` must also be omitted at execute time.
        :param b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        :param sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        :param sfb2_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB2 data pointers
            (each base must be 16B-aligned)
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

        self._value_error_if(
            prob_tensor is None,
            "prob_tensor is required (the kernel always fuses the prob multiply)",
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

        self._logger.debug("Executing grouped_gemm_unfused_subchannel_scaled kernel")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                sfb_tensor=sfb_tensor,
                sfb2_tensor=sfb2_tensor,
                d_tensor=d_tensor,
                sfa_tensor=sfa_tensor,
                sfa2_tensor=sfa2_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                bias_tensor=bias_tensor,
                prob_tensor=prob_tensor,
                stream=current_stream,
            )
        else:
            self._compiled_kernel(
                a_tensor=a_tensor,
                b_ptrs_device=b_ptrs,
                sfb_ptrs_device=sfb_ptrs,
                sfb2_ptrs_device=sfb2_ptrs,
                d_tensor=d_tensor,
                sfa_tensor=sfa_tensor,
                sfa2_tensor=sfa2_tensor,
                padded_offsets=padded_offsets,
                alpha_tensor=alpha_tensor,
                bias_tensor=bias_tensor,
                prob_tensor=prob_tensor,
                stream=current_stream,
            )

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmUnfusedSubchannelScaledSm100Objects = {}


def grouped_gemm_unfused_subchannel_scaled_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfa2_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    sfb2_tensor: Optional[torch.Tensor] = None,
    bias_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    sfb2_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    acc_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    d_tensor: Optional[torch.Tensor] = None,
    cd_major: str = "n",
    block2_shape: Tuple[int, int, int] = (1, 256, 256),
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    vector_f32: bool = True,
    m_aligned: int = 256,
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for the unfused subchannel-scaled grouped GEMM operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1), FP4, k-major
        sfa_tensor: First-level scale factor A (MMA-tiled (32, 4, valid_m//128, 4, rest_k, 1) view)
        sfa2_tensor: Second-level scale factor A, shape (ceil(valid_m/sgm), ceil(k/sgk), 1) FP32,
            stride (1, rows, rows * cols) (rows-contiguous; e.g. built via
            ``torch.zeros((1, cols, rows)).permute(2, 1, 0)``)
        padded_offsets: End offset per expert after padding (l,), int32, cumulative
        alpha_tensor: Per-group scaling (l,), FP32
        prob_tensor: Per-token probability tensor (valid_m, 1, 1), FP32. Required --
            the kernel always fuses the prob multiply; pass ones for a plain GEMM.
        b_tensor: (Dense) Weight B tensor (n, k, l), FP4, k-major
        sfb_tensor: (Dense) First-level scale factor B (MMA-tiled view)
        sfb2_tensor: (Dense) Second-level scale factor B, shape (ceil(n/sgn), ceil(k/sgk), l) FP32,
            stride (1, rows, rows * cols)
        bias_tensor: Optional per-expert bias, shape (n, l), stride (1, n), BF16.
            The epilogue computes ``d = gemm * alpha + prob * bias``.
        b_ptrs: (Discrete) 1-D int64 device tensor of per-expert B data pointers
        sfb_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB data pointers
        sfb2_ptrs: (Discrete) 1-D int64 device tensor of per-expert SFB2 data pointers
            (each base must be 16B-aligned; per-expert block bytes must be a
            multiple of 16 when num_experts > 1)
        n: (Discrete) B weight N dimension
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B tensor major dimension (must be "k" for FP4)
        acc_dtype: Accumulator data type (must be float32)
        d_dtype: Output D tensor data type (must be bfloat16)
        d_tensor: Optional preallocated output tensor to write into instead of
            allocating. Must match the internal layout: shape (valid_m, n_out, 1),
            stride (n_out, 1, valid_m * n_out), dtype bfloat16, on a_tensor.device.
        cd_major: CD major dimension (only "n"-major layout is supported)
        block2_shape: Second-level scale granularity (sgm, sgn, sgk)
        mma_tiler_mn: MMA tiler shape; (256, 128) or (256, 256)
        cluster_shape_mn: Cluster shape; default (2, 1)
        sf_vec_size: First-level scale factor vector size (must be 16)
        sf_fp8_dtype_override: Reinterpret FP8 block scale factors as E5M3 (Rubin-only);
            scale tensors are still passed as ``torch.float8_e4m3fn``.
        vector_f32: Use vectorized (packed) f32 arithmetic (default True -- the
            byte-exact-verified configuration)
        m_aligned: M alignment (must be 256)
        use_dynamic_sched: Enable dynamic tile scheduling
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing the output tensor:
            - **d_tensor** (torch.Tensor): BF16 output tensor (valid_m, n, 1)
    """
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    framework = detect_framework(a_tensor)
    if framework == "jax":
        raise ValueError(f"grouped_gemm_unfused_subchannel_scaled_wrapper_sm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for grouped_gemm_unfused_subchannel_scaled_wrapper_sm100; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    d_dtype = _convert_to_cutlass_data_type(d_dtype) if d_dtype is not None else cutlass.BFloat16
    b_dtype = _convert_to_cutlass_data_type(b_dtype) if b_dtype is not None else None

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None

    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor, sfb2_tensor) or (b_ptrs, sfb_ptrs, sfb2_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor, sfb2_tensor) or (b_ptrs, sfb_ptrs, sfb2_ptrs)")
    if prob_tensor is None:
        raise ValueError("prob_tensor is required (the kernel always fuses the prob multiply); pass ones for a plain GEMM")

    valid_m, k_physical, _ = a_tensor.shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        if sfb_tensor is None or sfb2_tensor is None:
            raise ValueError("sfb_tensor and sfb2_tensor are required in dense mode")
        n_out, _, l = b_tensor.shape
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, l):
            raise ValueError(f"bias_tensor must have shape {(n_out, l)}, got {tuple(bias_tensor.shape)}")
        num_experts = None
        b_shape = None
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        _validate_pointer_tensor(sfb2_ptrs, "sfb2_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        k_logical = k_physical * 2 if b_dtype in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n, k_logical)
        n_out = n
        l = num_experts
        if bias_tensor is not None and tuple(bias_tensor.shape) != (n_out, num_experts):
            raise ValueError(f"bias_tensor must have shape {(n_out, num_experts)}, got {tuple(bias_tensor.shape)}")

    _logger.debug("grouped_gemm_unfused_subchannel_scaled_wrapper_sm100: Creating output tensors")

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
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    if valid_m == 0:
        _logger.debug("grouped_gemm_unfused_subchannel_scaled_wrapper_sm100: valid_m is zero, skipping kernel execution")
        return TupleDict(d_tensor=d_tensor)

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Tuple[int, ...], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    device_type = get_device_type()

    common_key_tail = (
        tuple(padded_offsets.shape),
        tuple(padded_offsets.stride()),
        padded_offsets.dtype,
        acc_dtype,
        d_dtype,
        cd_major,
        tuple(block2_shape),
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_fp8_dtype_override,
        vector_f32,
        m_aligned,
        use_dynamic_sched,
    )
    dynamic_m_key = (
        a_tensor.shape[1:],
        stride_order(a_tensor),
        a_tensor.dtype,
        d_tensor.shape[1:],
        stride_order(d_tensor),
        *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1), dynamic_stride_dims=(5,)),
        *dynamic_m_tensor_signature(sfa2_tensor, (sfa2_tensor.shape[1], 1), dynamic_stride_dims=(1, 2)),
        *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
        *tensor_signature(alpha_tensor),
    )
    if is_dense:
        cache_key = (
            device_type,
            weight_mode,
            *dynamic_m_key,
            *tensor_signature(b_tensor),
            *tensor_signature(sfb_tensor),
            *tensor_signature(sfb2_tensor),
            *tensor_signature(bias_tensor),
            *common_key_tail,
        )
    else:
        cache_key = (
            device_type,
            weight_mode,
            *dynamic_m_key,
            b_shape,
            b_dtype,
            *tensor_signature(bias_tensor),
            tuple(b_ptrs.shape),
            b_ptrs.dtype,
            tuple(sfb_ptrs.shape),
            sfb_ptrs.dtype,
            tuple(sfb2_ptrs.shape),
            sfb2_ptrs.dtype,
            *common_key_tail,
            b_major,
            num_experts,
        )

    if cache_key in _cache_of_GroupedGemmUnfusedSubchannelScaledSm100Objects:
        _logger.debug("grouped_gemm_unfused_subchannel_scaled_wrapper_sm100: Using previously cached object")
        api = _cache_of_GroupedGemmUnfusedSubchannelScaledSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_unfused_subchannel_scaled_wrapper_sm100: No previously cached object found, creating new one")
        if is_dense:
            api = GroupedGemmUnfusedSubchannelScaledSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_sfa2=sfa2_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_d=d_tensor,
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfb2=sfb2_tensor,
                sample_bias=bias_tensor,
                acc_dtype=acc_dtype,
                block2_shape=block2_shape,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                use_dynamic_sched=use_dynamic_sched,
            )
        else:
            api = GroupedGemmUnfusedSubchannelScaledSm100(
                sample_a=a_tensor,
                sample_sfa=sfa_tensor,
                sample_sfa2=sfa2_tensor,
                sample_padded_offsets=padded_offsets,
                sample_alpha=alpha_tensor,
                sample_prob=prob_tensor,
                sample_d=d_tensor,
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                sample_bias=bias_tensor,
                acc_dtype=acc_dtype,
                block2_shape=block2_shape,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sf_vec_size=sf_vec_size,
                sf_fp8_dtype_override=sf_fp8_dtype_override,
                vector_f32=vector_f32,
                m_aligned=m_aligned,
                b_major=b_major,
                use_dynamic_sched=use_dynamic_sched,
            )

        assert api.check_support(), "Unsupported configuration"
        api.compile()
        _cache_of_GroupedGemmUnfusedSubchannelScaledSm100Objects[cache_key] = api

    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            sfa2_tensor=sfa2_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            d_tensor=d_tensor,
            b_tensor=b_tensor,
            sfb_tensor=sfb_tensor,
            sfb2_tensor=sfb2_tensor,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )
    else:
        api.execute(
            a_tensor=a_tensor,
            sfa_tensor=sfa_tensor,
            sfa2_tensor=sfa2_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            d_tensor=d_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            sfb2_ptrs=sfb2_ptrs,
            bias_tensor=bias_tensor,
            current_stream=current_stream,
        )

    return TupleDict(d_tensor=d_tensor)
