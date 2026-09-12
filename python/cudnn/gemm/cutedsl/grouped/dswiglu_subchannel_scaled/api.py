# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
API for the Subchannel-Scaled dSwiGLU Grouped GEMM Kernel (SM100+)

Second-level-scaled ("subchannel-scaled") dSwiGLU-backward block-scaled grouped
GEMM for MoE workloads: NVFP4 A (upstream gradient dY) and B (FC2 weights)
carry two scale levels -- per-(1, 16) FP8 block scale factors (SFA/SFB) plus
FP32 second-level scales (SFA2/SFB2) at ``block2_shape = (sgm, sgn, sgk)``
granularity. The fused epilogue consumes the forward FC1 pre-activations ``c``
(gate/up interleaved in 32-column bands) and produces:

- ``d`` -- the dSwiGLU backward output, ``2n`` wide (gate/up), either BF16 or
  NVFP4-quantized (``d_quant`` packed e2m1 + ``d_quant_sf`` e4m3 row scales,
  ``norm_const``-scaled);
- ``dprob`` -- the routing-probability gradient (always produced,
  atomic-add accumulated);
- ``dbias`` -- optional per-expert column sums of d (atomics);
- ``sfd2`` -- second-level output descales, per-(sgm x sgn) block max of the
  deinterleaved ``dy_gate`` / ``dy_up`` halves (atomic-max accumulated).

``d_deinterleaved=True`` (default) stores the n-axis outputs (d, its row
scale factors, dbias) DEINTERLEAVED -- gate bands first ``[0, n)``, up bands
second ``[n, 2n)`` -- and lays the fused ``sfd2`` out as concatenated halves,
which is exactly a consumer GEMM's ``(rows, k/sgk)`` ``a_scales2`` grid with
``sgk = sgn``. Only the quantized-D configuration supports the deinterleaved
layout (the harness-validated combination).

FE ``n`` convention: ``n`` is the GEMM/weight width (B is ``(n, k, l)``); the
n-axis outputs cover ``2n``. On Rubin (SM107) devices the API transparently
dispatches to the Rubin-native kernel module
(``moe_blockscaled_grouped_gemm_dswiglu_subchannel_scaled_rubin.py``), which also
accepts ``sf_fp8_dtype_override="e5m3"`` for E5M3 first-level scales.
"""

from __future__ import annotations

import math
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

from .grouped_gemm_dswiglu_subchannel_scaled import (
    DEFAULT_CLUSTER_SHAPE,
    DEFAULT_CTA_SHAPE,
    BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernel,
)
from ..moe_utils import MoEWeightMode
from cutlass.cute.nvgpu import OperandMajorMode


def _get_rubin_kernel():
    """Lazy import of the Rubin (sm107) kernel module.

    It needs a cutlass-dsl build with ``cutlass.utils.rubin_helpers``, so it is
    only imported when a Rubin device is detected (APIBase._is_rubin_kernel).
    """
    from .moe_blockscaled_grouped_gemm_dswiglu_subchannel_scaled_rubin import (
        BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernelSm107,
    )

    return BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernelSm107


_JAX_SF_LAYOUT_ERROR = (
    "the block scale-factor tensors (sfa/sfb and the d_quant_sf output) are MMA-tiled "
    "(32, 4, m//128, 4, rest_k, l) strided views that are not expressible as JAX arrays "
    "(a row-major JAX array of that shape has different memory); pass torch tensors"
)


class GroupedGemmDswigluSubchannelScaledSm100(APIBase):
    """API for the subchannel-scaled dSwiGLU-backward grouped GEMM on SM100+ GPUs.

    ``D2n = dSwiGLU(alpha^2 * subchannel_scaled_gemm(A, B), beta * C, prob)``
    with fused ``dprob``, optional ``dbias``, second-level output descales
    ``sfd2``, and optional NVFP4 output quantization. Weight mode is
    auto-detected from the constructor arguments:

    - Dense: provide ``sample_b``, ``sample_sfb``, and ``sample_sfb2``.
    - Discrete: provide ``num_experts``, ``b_shape``, and ``b_dtype``.

    Zero-init contract: ``dprob``, ``dbias``, and ``sfd2`` are ATOMICALLY
    accumulated by the kernel -- the caller (or the high-level wrapper, which
    owns allocation) must pass zero-initialized tensors on every call.
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfa2: torch.Tensor,
        sample_c: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_d: torch.Tensor,
        sample_dprob: torch.Tensor,
        sample_sfd2: torch.Tensor,
        sample_dbias: Optional[torch.Tensor] = None,
        sample_d_quant_sf: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        # Dense mode (contiguous) -- provide these:
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        sample_sfb2: Optional[torch.Tensor] = None,
        # Discrete mode -- provide these instead:
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        # Configuration
        acc_dtype: Optional[torch.dtype] = None,
        block2_shape: Tuple[int, int, int] = (1, 256, 256),
        mma_tiler_mn: Tuple[int, int] = DEFAULT_CTA_SHAPE,
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = True,
        m_aligned: int = 256,
        b_major: str = "k",
        use_dynamic_sched: bool = False,
        d_deinterleaved: bool = True,
        glu_clamp_max: Optional[float] = 7.0,
        glu_clamp_min: Optional[float] = -7.0,
        dsmem_rowwise: bool = True,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
    ):
        """Initialize the GroupedGemmDswigluSubchannelScaledSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1), FP4, k-major (upstream dY)
        :param sample_sfa: Sample first-level scale factor A tensor
            (MMA-tiled (32, 4, valid_m//128, 4, rest_k, 1) view)
        :param sample_sfa2: Sample second-level scale factor A tensor,
            (ceil(valid_m/sgm), ceil(k/sgk), 1) FP32, stride (1, rows, rows*cols)
        :param sample_c: Forward FC1 pre-activations (valid_m, 2n, 1) BF16,
            n-major, gate/up interleaved in 32-column bands
        :param sample_padded_offsets: End offset per expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-expert accumulator scale (expert_cnt,) f32.
            The kernel applies alpha TWICE (alpha^2 on the GEMM accumulator).
        :param sample_beta: Per-expert C scale (expert_cnt,) f32 (applied
            inside dSwiGLU, after the clamps)
        :param sample_prob: Per-token routing probability (valid_m, 1, 1) f32
        :param sample_d: Output D. BF16 (valid_m, 2n, 1) n-major, or FP4
            (torch.float4_e2m1fn_x2, physical (valid_m, n, 1)) -- the dtype
            selects the quantized-output mode.
        :param sample_dprob: dprob output (valid_m, 1, 1) f32. MUST be
            zero-initialized (atomic-add accumulated).
        :param sample_sfd2: Fused second-level output descales
            (rows, 2*nd, 1) f32, rows-contiguous stride (1, rows, 2*nd*rows),
            where rows = ceil(valid_m/sgm), nd = ceil(n/sgn). MUST be
            zero-initialized (atomic-max accumulated). d_deinterleaved=True
            lays it out as concatenated halves [gate | up]; False interleaves
            gate/up per block column.
        :param sample_dbias: Optional dbias output (expert_cnt, 2n, 1), BF16
            or FP32 (FP32 is a testing-only plain-atomic path). MUST be
            zero-initialized.
        :param sample_d_quant_sf: Row scale factors of the quantized D
            (MMA-tiled (32, 4, valid_m//128, 4, ceil(ceil(2n/16)/4), 1) e4m3
            view). Required iff sample_d is FP4.
        :param sample_norm_const: Global D-quant encode scale, shape (1,) f32.
            Required iff sample_d is FP4.
        :param sample_b: (Dense) Sample B tensor (n, k, l), FP4, k-major
        :param sample_sfb: (Dense) Sample first-level scale factor B tensor
        :param sample_sfb2: (Dense) Sample second-level scale factor B tensor,
            (ceil(n/sgn), ceil(k/sgk), l) FP32, stride (1, rows, rows*cols)
        :param num_experts: (Discrete) Number of experts
        :param b_shape: (Discrete) Shape of a single expert B tensor, (n, k) with logical k
        :param b_dtype: (Discrete) Data type of B tensors
        :param acc_dtype: Accumulator data type (must be float32)
        :param block2_shape: Second-level scale granularity (sgm, sgn, sgk).
            sgn counts DEINTERLEAVED output columns (one sfd2 block spans sgn
            columns of dy_gate/dy_up = 2*sgn interleaved D columns) and must be
            a multiple of the MMA tile N (128).
        :param mma_tiler_mn: MMA tiler shape; must be (256, 128)
        :param cluster_shape_mn: Cluster shape; (2, 1), (2, 2), or (2, 4). cluster_n > 1 makes N-adjacent work tiles cluster peers (the rowwise-sfd2 DSMEM geometry) and requires ceil(n/128) % cluster_n == 0
        :param sf_vec_size: First-level scale factor vector size (must be 16)
        :param vector_f32: Use vectorized (packed) f32 arithmetic. The default
            True is the byte-exact-verified configuration.
        :param m_aligned: Alignment for group M dimension (must be 256)
        :param b_major: Major dimension for B tensor (must be "k" for FP4)
        :param use_dynamic_sched: Enable dynamic tile scheduling
        :param d_deinterleaved: Store the n-axis outputs (d, d_quant_sf, dbias)
            deinterleaved [gate | up] and sfd2 as concatenated halves. Only
            supported with the quantized-D output (the harness-validated
            combination).
        :param glu_clamp_max: Clamp max on the raw C pre-activations (both
            clamps must be set or both None)
        :param glu_clamp_min: Clamp min on the raw C pre-activations
        :param dsmem_rowwise: Allow the rowwise-sfd2 DSMEM reduction (cluster
            red.shared::cluster max + mbarrier handoff) instead of the gmem
            atomic + counter-spin protocol, when the geometry is eligible
            (quantized D + sgn/128 == cluster_n > 1). Byte-exact either way.
        :param sf_fp8_dtype_override: Reinterpret the FP8-format first-level
            scale factors (sfa/sfb) as E5M3 instead of the E4M3 implied by their
            storage dtype. Rubin-only; the tensors are still supplied as
            ``torch.float8_e4m3fn`` since torch has no e5m3 dtype. With e5m3
            inputs the quantized-D row scales (``d_quant_sf``) are e5m3-encoded
            too (output scale format follows the input's) and the sfd2 norm
            becomes ``6 * 61440``.
        """
        framework = detect_framework(sample_a)
        if framework == "jax":
            raise ValueError(f"GroupedGemmDswigluSubchannelScaledSm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
        if framework != "torch":
            raise ValueError(f"Unsupported tensor framework '{framework}' for GroupedGemmDswigluSubchannelScaledSm100; pass torch tensors")
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
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa", canonical=True)
        self.sfa2_desc = self._make_tensor_desc(sample_sfa2, name="sample_sfa2", canonical=True)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c", canonical=True)
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets", canonical=True)
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha", canonical=True)
        self.beta_desc = self._make_tensor_desc(sample_beta, name="sample_beta", canonical=True)
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob", canonical=True)
        self.dprob_desc = self._make_tensor_desc(sample_dprob, name="sample_dprob", canonical=True)
        self.sfd2_desc = self._make_tensor_desc(sample_sfd2, name="sample_sfd2", canonical=True)
        self.dbias_desc = self._make_tensor_desc(sample_dbias, name="sample_dbias", canonical=True)
        self.d_quant_sf_desc = self._make_tensor_desc(sample_d_quant_sf, name="sample_d_quant_sf", canonical=True)
        self.norm_const_desc = self._unpad_tensor_to_ndim(
            self._make_tensor_desc(sample_norm_const, name="sample_norm_const", canonical=True),
            1,
            "norm_const",
        )
        # sfd2 view descriptors (the kernel receives the two half views of the
        # fused buffer; their strides differ between the deint/interleaved
        # layouts, so capture them from actual view samples).
        nd_cols = sample_sfd2.shape[1] // 2
        if d_deinterleaved:
            sfd2_gate_view = sample_sfd2[:, :nd_cols, :]
            sfd2_up_view = sample_sfd2[:, nd_cols:, :]
        else:
            sfd2_gate_view = sample_sfd2[:, 0::2, :]
            sfd2_up_view = sample_sfd2[:, 1::2, :]
        self.sfd2_gate_desc = self._make_tensor_desc(sfd2_gate_view, name="sample_sfd2[gate]", canonical=True)
        self.sfd2_up_desc = self._make_tensor_desc(sfd2_up_view, name="sample_sfd2[up]", canonical=True)

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
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.use_2cta_instrs = self.mma_tiler_mn[0] == 256
        if cluster_shape_mn is None:
            self.cluster_shape_mn = DEFAULT_CLUSTER_SHAPE
        else:
            self.cluster_shape_mn = tuple(cluster_shape_mn)
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.use_dynamic_sched = use_dynamic_sched
        self.b_major = b_major
        self.d_deinterleaved = d_deinterleaved
        self.glu_clamp_max = glu_clamp_max
        self.glu_clamp_min = glu_clamp_min
        self.dsmem_rowwise = dsmem_rowwise
        self.sf_fp8_dtype_override = sf_fp8_dtype_override

        self._interpret_uint8_as_fp4x2 = True
        # D dtype-driven quantized-output mode (glu_hadamard_quant precedent).
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d", canonical=True)
        self.d_quant = self._is_fp4x2(self.d_desc.dtype)
        self._has_dbias = self.dbias_desc is not None
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledSubChannelMoEGroupedGemmDgluDbiasKernel

        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._workspace = None
        self._workspace_bytes = 0
        self._gemm = None
        self._max_active_clusters = None
        self._logger.debug("__init__ completed")

    # ---- geometry helpers ----

    def _n_from_descs(self) -> int:
        if self.weight_mode == MoEWeightMode.DENSE:
            return self._tensor_shape(self.b_desc, name="sample_b")[0]
        return self.b_shape[0]

    @property
    def sfd2_n_contrib(self) -> int:
        return max(1, self.sgn // self.mma_tiler_mn[1])

    def _sfd2_counter_cnt(self, valid_m: int, n: int) -> int:
        """One u32 per (global m CTA tile, sfd2 n-block); must match the
        kernel's _sfd2_counter_cnt formula. Zero when the arrival protocol is
        inactive (bf16 D, or sgn <= tile N)."""
        if not (self.d_quant and self.sfd2_n_contrib > 1):
            return 0
        cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
        cta_n = self.mma_tiler_mn[1]
        n_tiles_f = ceil_div(n, cta_n)
        return (valid_m // cta_m) * ceil_div(n_tiles_f, self.sfd2_n_contrib)

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

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
        n2 = 2 * n

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

        # ---- shapes ----
        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n2, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n2, 1), "D")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.dprob_desc, (tensor_m, 1, 1), "dprob")
        self._check_tensor_shape(self.dbias_desc, (l, n2, 1), "dbias")
        self._check_tensor_shape(self.alpha_desc, (self.expert_cnt,), "alpha")
        self._check_tensor_shape(self.beta_desc, (self.expert_cnt,), "beta")
        self._check_tensor_shape(self.padded_offsets_desc, (self.expert_cnt,), "padded_offsets")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
        rest_n2 = ceil_div(ceil_div(n2, self.sf_vec_size), 4)
        self._check_tensor_shape(self.d_quant_sf_desc, (32, 4, ceil_div(tensor_m, 128), 4, rest_n2, 1), "d_quant_sf")
        self._check_tensor_shape(self.norm_const_desc, (1,), "norm_const")

        # ---- second-level (subchannel) scale factors ----
        self._value_error_if(
            self.sgm < 1 or self.sgn < 1 or self.sgk < 1,
            f"block2_shape components must be >= 1, got {self.block2_shape}",
        )
        self._value_error_if(
            k % self.sgk != 0,
            f"k ({k}) must be divisible by sgk ({self.sgk})",
        )
        self._value_error_if(
            tensor_m % self.sgm != 0,
            f"valid_m ({tensor_m}) must be divisible by sgm ({self.sgm})",
        )
        self._value_error_if(
            self.sgn % self.mma_tiler_mn[1] != 0,
            f"sgn ({self.sgn}) must be a multiple of the MMA tile N "
            f"({self.mma_tiler_mn[1]}): one sfd2 block = sgn deinterleaved "
            f"columns and every N work tile must sit inside one block",
        )
        sfa2_rows = ceil_div(tensor_m, self.sgm)
        sf2_cols = ceil_div(k, self.sgk)
        sfa2_shape = (sfa2_rows, sf2_cols, 1)
        self._check_tensor_shape(self.sfa2_desc, sfa2_shape, "SFA2")
        _ = self._check_tensor_stride(
            self.sfa2_desc,
            stride=[canonicalize_unit_dim_strides(sfa2_shape, (1, sfa2_rows, sfa2_rows * sf2_cols))],
            extra_error_msg="SFA2 must be rows-contiguous",
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
            sfb2_block_bytes = sfb2_rows * sf2_cols * 4
            self._value_error_if(
                self.expert_cnt > 1 and sfb2_block_bytes % 16 != 0,
                f"discrete SFB2 per-expert block ({sfb2_rows} x {sf2_cols} f32 = "
                f"{sfb2_block_bytes} bytes) must be a multiple of 16 bytes when "
                f"num_experts > 1 (per-expert bases are assumed 16B-aligned)",
            )

        # ---- sfd2 output (fused buffer + views) ----
        sfd2_rows = ceil_div(tensor_m, self.sgm)
        nd = ceil_div(n, self.sgn)
        sfd2_shape = (sfd2_rows, 2 * nd, 1)
        self._check_tensor_shape(self.sfd2_desc, sfd2_shape, "sfd2")
        _ = self._check_tensor_stride(
            self.sfd2_desc,
            stride=[canonicalize_unit_dim_strides(sfd2_shape, (1, sfd2_rows, 2 * nd * sfd2_rows))],
            extra_error_msg="sfd2 must be rows-contiguous (fused (rows, 2*nd, 1) buffer)",
        )
        if self.d_deinterleaved:
            # up half base = rows*nd*4 bytes from the buffer base (16B-aligned
            # contract of the kernel's assumed_align on the up view).
            self._value_error_if(
                (sfd2_rows * nd * 4) % 16 != 0,
                f"sfd2 rows={sfd2_rows} nd={nd} breaks the up-half 16B alignment",
            )
        else:
            self._value_error_if(
                (sfd2_rows * 4) % 16 != 0,
                f"sfd2 rows={sfd2_rows} breaks the up-view 16B alignment",
            )

        # ---- dtypes ----
        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[cutlass.Float4E2M1FN, cutlass.Uint8],
            name="A/B",
            extra_error_msg="only NVFP4 (fp4) inputs are supported",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.b_desc, dtype=self.ab_dtype, name="B", extra_error_msg="B must have the same dtype as A")
        else:
            self._value_error_if(
                self.b_dtype != self.ab_dtype,
                f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})",
            )
            self._value_error_if(self.b_major != "k", f"b_major must be 'k' for fp4 ab_dtype, got {self.b_major}")
        self.sf_dtype = self._check_dtype(
            self.sfa_desc,
            dtype=cutlass.Float8E4M3FN,
            name="SFA/SFB",
            extra_error_msg="first-level scale factors must be torch.float8_e4m3fn (the NVFP4 recipe)",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.sfb_desc, dtype=self.sf_dtype, name="SFB")
        self._check_dtype(self.sfa2_desc, dtype=cutlass.Float32, name="SFA2")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.sfb2_desc, dtype=cutlass.Float32, name="SFB2")
        self._check_dtype(self.c_desc, dtype=cutlass.BFloat16, name="C")
        self.d_dtype = self._check_dtype(
            self.d_desc,
            dtype=[cutlass.BFloat16, cutlass.Float4E2M1FN],
            name="D",
            extra_error_msg="D must be bfloat16 (plain output) or fp4 (quantized output)",
        )
        self._check_dtype(self.acc_dtype, dtype=cutlass.Float32, name="Accumulator")
        self._check_dtype(self.alpha_desc, dtype=cutlass.Float32, name="alpha")
        self._check_dtype(self.beta_desc, dtype=cutlass.Float32, name="beta")
        self._check_dtype(self.prob_desc, dtype=cutlass.Float32, name="prob")
        self._check_dtype(self.dprob_desc, dtype=cutlass.Float32, name="dprob")
        self._check_dtype(self.sfd2_desc, dtype=cutlass.Float32, name="sfd2")
        self._check_dtype(
            self.dbias_desc,
            dtype=[cutlass.BFloat16, cutlass.Float32],
            name="dbias",
            extra_error_msg="dbias must be bfloat16 (or float32 -- a testing-only plain-atomic path)",
        )
        self._check_dtype(self.d_quant_sf_desc, dtype=cutlass.Float8E4M3FN, name="d_quant_sf")
        self._check_dtype(self.norm_const_desc, dtype=cutlass.Float32, name="norm_const")
        self._check_dtype(self.padded_offsets_desc, dtype=cutlass.Int32, name="padded_offsets")
        self._value_error_if(
            self.sf_vec_size != 16,
            f"sf_vec_size must be 16 (NVFP4), got {self.sf_vec_size}",
        )

        # ---- quant-mode coupling ----
        if self.d_quant:
            self._value_error_if(
                self.d_quant_sf_desc is None or self.norm_const_desc is None,
                "quantized D output (fp4 sample_d) requires sample_d_quant_sf and sample_norm_const",
            )
        else:
            self._value_error_if(
                self.d_quant_sf_desc is not None or self.norm_const_desc is not None,
                "sample_d_quant_sf/sample_norm_const must be omitted for the bf16 D output",
            )

        # ---- deinterleaved-layout gate ----
        self._value_error_if(
            self.d_deinterleaved and not self.d_quant,
            "d_deinterleaved=True is only supported with the quantized (fp4) D "
            "output -- the harness-validated combination; use "
            "d_deinterleaved=False for the bf16 D output",
        )

        # ---- launch config (pinned by the kernel) ----
        self._value_error_if(
            self.mma_tiler_mn != (256, 128),
            f"mma_tiler_mn must be (256, 128), got {self.mma_tiler_mn}",
        )
        self._value_error_if(
            self.cluster_shape_mn not in ((2, 1), (2, 2), (2, 4)),
            f"cluster_shape_mn must be (2, 1), (2, 2), or (2, 4), got {self.cluster_shape_mn} "
            f"(cluster_m stays 2: wider cluster-M tiles need a per-expert-M gate that cannot "
            f"be verified host-side on ragged inputs)",
        )
        if self.cluster_shape_mn[1] > 1:
            # The scheduler decomposes over CLUSTER N tiles: the N work-tile
            # count must be a cluster_n multiple or the last N cluster's peers
            # would map past the edge.
            self._value_error_if(
                ceil_div(n, self.mma_tiler_mn[1]) % self.cluster_shape_mn[1] != 0,
                f"with cluster_n={self.cluster_shape_mn[1]}, the N work-tile count " f"ceil({n}/{self.mma_tiler_mn[1]}) must be a cluster_n multiple",
            )
        self._value_error_if(
            self.m_aligned != self._kernel.FIX_PAD_SIZE,
            f"m_aligned must be {self._kernel.FIX_PAD_SIZE} (FIX_PAD_SIZE), got {self.m_aligned}",
        )
        self._value_error_if(
            (self.glu_clamp_max is None) != (self.glu_clamp_min is None),
            "glu_clamp_max and glu_clamp_min must both be set or both be None",
        )

        # ---- problem-shape gates ----
        _ = self._check_tensor_stride(self.a_desc, stride=[(k, 1, tensor_m * k)], extra_error_msg="A must have k-major layout")
        if self.weight_mode == MoEWeightMode.DENSE:
            _ = self._check_tensor_stride(self.b_desc, stride=[(k, 1, n * k)], extra_error_msg="B must have k-major layout")
        _ = self._check_tensor_stride(self.c_desc, stride=[(n2, 1, tensor_m * n2)], extra_error_msg="C must have n-major layout")
        _ = self._check_tensor_stride(self.d_desc, stride=[(n2, 1, tensor_m * n2)], extra_error_msg="D must have n-major layout")
        if self.dbias_desc is not None:
            _ = self._check_tensor_stride(
                self.dbias_desc,
                stride=[canonicalize_unit_dim_strides((l, n2, 1), (n2, 1, l * n2))],
                extra_error_msg="dbias must have contiguous columns per expert",
            )
        self._value_error_if(n2 % 256 != 0, f"output width 2n ({n2}) must be divisible by 256 (n % 128 == 0)")
        self._value_error_if(k % 64 != 0, f"k ({k}) must be divisible by 64")
        self._value_error_if(tensor_m % 256 != 0, f"valid_m must be divisible by 256, got {tensor_m}")
        self._value_error_if(self.expert_cnt > 1024, f"expert_cnt must be <= 1024, got {self.expert_cnt}")

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
                and check_contigous_16B_alignment(self.d_dtype, self.d_desc.stride_order, (tensor_m, n2, 1))
            ),
            "Invalid tensor alignment: tensors must be 16B aligned",
        )

        if not cuda_is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = get_compute_capability()
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmDswigluSubchannelScaled requires SM100+ compute capability, but found SM{compute_capability}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self) -> None:
        """Compile the kernel."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        if self.a_desc.shape[0] == 0:
            self._logger.debug("sample valid_m is zero, skipping kernel compilation")
            return

        n = self._n_from_descs()
        gemm = self._kernel(
            self.sf_vec_size,
            self.sgm,
            self.sgn,
            self.sgk,
            cutlass.Float32,
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            self.vector_f32,
            self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func="dswiglu",
            generate_sfd2=True,
            glu_alpha=None,
            glu_clamp_max=self.glu_clamp_max,
            glu_clamp_min=self.glu_clamp_min,
            d_deinterleaved=self.d_deinterleaved,
            deint_n=(2 * n) if self.d_deinterleaved else 0,
            dsmem_rowwise=self.dsmem_rowwise,
            # Only the Rubin kernel accepts sf_fp8_dtype_override, and check_support
            # rejects "e5m3" unless _is_rubin_kernel -- the same flag that selected
            # self._kernel. The kernel maps the string to FloatNV8E5M3FNU itself, so
            # that internal-only type is never named outside the Rubin module.
            **({"sf_fp8_dtype_override": self.sf_fp8_dtype_override} if self.sf_fp8_dtype_override == "e5m3" else {}),
        )
        self._gemm = gemm

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(
            max_active_clusters <= 0,
            "max_active_clusters must be > 0 after applying overlap margin; reduce CUDNNFE_CLUSTER_OVERLAP_MARGIN",
        )
        self._max_active_clusters = max_active_clusters
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        # Workspace: sized here for the sample valid_m; execute() grows it if a
        # later call's sfd2 sync-counter count needs more bytes.
        counter_cnt = self._sfd2_counter_cnt(self.a_desc.shape[0], n)
        self._workspace_bytes = gemm.get_workspace_bytes(sfd2_sync_counter_cnt=counter_cnt)
        self._workspace = allocate_byte_workspace(self._framework, self._workspace_bytes, self.a_desc.device)

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
        c_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.c_desc.dtype,
            shape=(valid_m, *self.c_desc.shape[1:]),
            stride_order=self.c_desc.stride_order,
        )
        d_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.d_desc.dtype,
            shape=(valid_m, *self.d_desc.shape[1:]),
            stride_order=self.d_desc.stride_order,
            assumed_align=32 if self.d_quant else 16,
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

        sfa2_rows = cute.sym_int()
        sfa2_cute_fake = self._make_fake_cute_tensor(
            dtype=self.sfa2_desc.dtype,
            shape=(sfa2_rows, self.sfa2_desc.shape[1], 1),
            stride=(1, cute.sym_int(), cute.sym_int()),
            assumed_align=16,
        )

        prob_cute_fake = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, *self.prob_desc.shape[1:]),
            stride=self.prob_desc.stride,
            assumed_align=16,
        )
        dprob_cute_fake = self._make_fake_cute_tensor(
            dtype=self.dprob_desc.dtype,
            shape=(valid_m, *self.dprob_desc.shape[1:]),
            stride=self.dprob_desc.stride,
            assumed_align=16,
        )

        # sfd2 half views: rows scale with valid_m; the column stride differs
        # between the deint (rows) and interleaved (2*rows) layouts, so keep
        # both strides dynamic (the layout choice is a kernel constexpr).
        def sfd2_view_fake(desc):
            return self._make_fake_cute_tensor(
                dtype=desc.dtype,
                shape=(cute.sym_int(), desc.shape[1], 1),
                stride=(1, cute.sym_int(), cute.sym_int()),
                assumed_align=16,
            )

        sfd2_gate_fake = sfd2_view_fake(self.sfd2_gate_desc)
        sfd2_up_fake = sfd2_view_fake(self.sfd2_up_desc)

        sfd_row_fake = None
        if self.d_quant:
            sfd_m_128 = cute.sym_int()
            sfd_stride_m = cute.sym_int(divisibility=32 * 4 * 4)
            sfd_shape = list(self.d_quant_sf_desc.shape)
            sfd_shape[2] = sfd_m_128
            sfd_stride = list(self.d_quant_sf_desc.stride)
            sfd_stride[5] = sfd_stride_m
            sfd_row_fake = self._make_fake_cute_tensor(
                dtype=self.d_quant_sf_desc.dtype,
                shape=tuple(sfd_shape),
                stride=tuple(sfd_stride),
                assumed_align=16,
            )

        return (
            a_cute_fake,
            c_cute_fake,
            d_cute_fake,
            sfa_cute_fake,
            sfa2_cute_fake,
            prob_cute_fake,
            dprob_cute_fake,
            sfd2_gate_fake,
            sfd2_up_fake,
            sfd_row_fake,
        )

    def _common_static_fakes(self):
        norm_const_fake = self._make_fake_cute_tensor_from_desc(self.norm_const_desc, assumed_align=16) if self.d_quant else None
        dbias_fake = self._make_fake_cute_tensor_from_desc(self.dbias_desc, assumed_align=16)
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        alpha_fake = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        beta_fake = self._make_fake_cute_tensor_from_desc(self.beta_desc, assumed_align=16)
        return norm_const_fake, dbias_fake, offsets_fake, alpha_fake, beta_fake

    def _make_tensor_api(self, _compiled_kernel, cached_n, cached_k, cached_b_stride, discrete: bool):
        def tensor_api(
            a_tensor,
            b_arg,
            sfb_arg,
            sfb2_arg,
            c_tensor,
            d_tensor,
            sfa_tensor,
            sfa2_tensor,
            sfd2_up_tensor,
            sfd2_gate_tensor,
            sfd_row_tensor,
            norm_const_tensor,
            padded_offsets,
            alpha_tensor,
            beta_tensor,
            prob_tensor,
            dprob_tensor,
            dbias_tensor,
            workspace_ptr,
            stream,
        ) -> None:
            if discrete:
                b_arg = int(get_data_ptr(b_arg))
                sfb_arg = int(get_data_ptr(sfb_arg))
                sfb2_arg = int(get_data_ptr(sfb2_arg))
            _compiled_kernel(
                a_tensor,
                b_arg,
                sfb_arg,
                sfb2_arg,
                cached_n,
                cached_k,
                cached_b_stride,
                workspace_ptr,
                c_tensor,
                d_tensor,
                sfa_tensor,
                sfa2_tensor,
                sfd2_up_tensor,
                sfd2_gate_tensor,
                sfd_row_tensor,
                norm_const_tensor,
                padded_offsets,
                alpha_tensor,
                beta_tensor,
                prob_tensor,
                dprob_tensor,
                0.0,  # linear_offset (dgeglu-only; act_func is pinned to dswiglu)
                dbias_tensor,
                stream,
            )

        return tensor_api

    def _compile_dense(self, gemm, max_active_clusters, fake_stream) -> None:
        fake_workspace_ptr = cute.runtime.nullptr(dtype=cutlass.Uint8, assumed_align=128)
        valid_m = cute.sym_int(divisibility=256)
        a_f, c_f, d_f, sfa_f, sfa2_f, prob_f, dprob_f, sfd2_gate_f, sfd2_up_f, sfd_row_f = self._make_dynamic_m_fakes(valid_m)
        norm_const_f, dbias_f, offsets_f, alpha_f, beta_f = self._common_static_fakes()
        b_f = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
        sfb_f = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
        sfb2_f = self._make_fake_cute_tensor_from_desc(self.sfb2_desc, assumed_align=16)

        _compiled_kernel = cute.compile(
            gemm,
            a_f,
            b_f,
            sfb_f,
            sfb2_f,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            OperandMajorMode.K,
            fake_workspace_ptr,
            c_f,
            d_f,
            sfa_f,
            sfa2_f,
            sfd2_up_f,
            sfd2_gate_f,
            sfd_row_f,
            norm_const_f,
            offsets_f,
            alpha_f,
            beta_f,
            prob_f,
            dprob_f,
            0.0,
            dbias_f,
            max_active_clusters,
            fake_stream,
            options="--enable-tvm-ffi",
        )
        self._compiled_kernel = self._make_tensor_api(_compiled_kernel, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int64(0), discrete=False)

    def _compile_discrete(self, gemm, max_active_clusters, fake_stream) -> None:
        if len(self.b_shape) == 2:
            n, k = self.b_shape
        else:
            n, k, _ = self.b_shape
        valid_m = cute.sym_int(divisibility=256)
        a_f, c_f, d_f, sfa_f, sfa2_f, prob_f, dprob_f, sfd2_gate_f, sfd2_up_f, sfd_row_f = self._make_dynamic_m_fakes(valid_m)
        norm_const_f, dbias_f, offsets_f, alpha_f, beta_f = self._common_static_fakes()

        # Compile-time placeholders for the pointer-array arguments (real
        # device bytes retyped to Int64 -- fake tensors have dummy iterators).
        self._compile_b_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)
        self._compile_sfb2_ptrs = allocate_byte_workspace(self._framework, 8 * self.expert_cnt, self.a_desc.device)

        def _ptr_iter(buf):
            placeholder = from_dlpack(buf, assumed_align=8)
            placeholder.element_type = cutlass.Int64
            return placeholder.iterator

        workspace_ptr_cute = from_dlpack(self._workspace, assumed_align=128).iterator

        _compiled_kernel = cute.compile(
            gemm,
            a_f,
            _ptr_iter(self._compile_b_ptrs),
            _ptr_iter(self._compile_sfb_ptrs),
            _ptr_iter(self._compile_sfb2_ptrs),
            cutlass.Int32(n),
            cutlass.Int32(k),
            cutlass.Int64(k),
            OperandMajorMode.K,
            workspace_ptr_cute,
            c_f,
            d_f,
            sfa_f,
            sfa2_f,
            sfd2_up_f,
            sfd2_gate_f,
            sfd_row_f,
            norm_const_f,
            offsets_f,
            alpha_f,
            beta_f,
            prob_f,
            dprob_f,
            0.0,
            dbias_f,
            max_active_clusters,
            fake_stream,
            options="--enable-tvm-ffi",
        )
        self._compiled_kernel = self._make_tensor_api(_compiled_kernel, cutlass.Int32(n), cutlass.Int32(k), cutlass.Int64(k), discrete=True)

    def _ensure_workspace(self, valid_m: int, n: int):
        """Grow-only workspace: the sfd2 sync-counter count scales with the
        runtime valid_m, so re-size when a call needs more bytes and derive a
        fresh iterator every call (the closure takes it as a parameter)."""
        required = self._gemm.get_workspace_bytes(sfd2_sync_counter_cnt=self._sfd2_counter_cnt(valid_m, n))
        if required > self._workspace_bytes:
            self._workspace = allocate_byte_workspace(self._framework, required, self.a_desc.device)
            self._workspace_bytes = required
        return from_dlpack(self._workspace, assumed_align=128).iterator

    def _assert_sfd2_liveness(self, valid_m: int, n: int):
        """Liveness of the sfd2 cross-tile arrival protocol (active when the
        quantized-D row-scale fold consumes an sfd2 block spanning multiple N
        work tiles). Conservative ragged-safe bound: per-expert m is unknown
        host-side, so bound the partner tile distance with the total valid_m."""
        if self._sfd2_counter_cnt(valid_m, n) == 0:
            return
        # A spinning tile waits on its block partners in OTHER clusters: one
        # block spans max(1, contrib/cluster_n) N-adjacent clusters (cluster_n
        # N-peers share a cluster and arrive together); N-adjacent cluster
        # linear distance is 1 when N varies fastest, m_cluster_tiles otherwise
        # (the scheduler's short-side-first order is over CLUSTER tiles).
        # Require the farthest partner distance < grid clusters. On the
        # DSMEM-eligible geometries (contrib == cluster_n) block_n_clusters is
        # 1 and the bound is vacuously true — all contributors are co-resident
        # cluster peers.
        cta_m = self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1)
        cta_n = self.mma_tiler_mn[1]
        n_tiles_f = ceil_div(n, cta_n)
        n_cluster_tiles = ceil_div(n_tiles_f, self.cluster_shape_mn[1])
        m_cluster_tiles_worst = ceil_div(valid_m, cta_m * self.cluster_shape_mn[0])
        base_dist_worst = min(m_cluster_tiles_worst, n_cluster_tiles)
        block_n_clusters = max(1, self.sfd2_n_contrib // self.cluster_shape_mn[1])
        max_partner_dist = (block_n_clusters - 1) * base_dist_worst
        self._runtime_error_if(
            max_partner_dist >= self._max_active_clusters,
            f"sfd2 arrival protocol could deadlock: worst-case partner cluster "
            f"distance {max_partner_dist} (contrib={self.sfd2_n_contrib}, "
            f"block_n_clusters={block_n_clusters}, "
            f"m_cluster_tiles<={m_cluster_tiles_worst}, n_cluster_tiles={n_cluster_tiles}) "
            f"must be < persistent grid clusters {self._max_active_clusters}",
        )

    def execute(
        self,
        a_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfa2_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        sfd2_gate_tensor: torch.Tensor,
        sfd2_up_tensor: torch.Tensor,
        dbias_tensor: Optional[torch.Tensor] = None,
        d_quant_sf_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        # Dense mode:
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        sfb2_tensor: Optional[torch.Tensor] = None,
        # Discrete mode:
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        sfb2_ptrs: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Execute the compiled kernel.

        ``dprob_tensor``, ``dbias_tensor``, and the sfd2 views MUST be
        zero-initialized: the kernel accumulates into them with atomics.
        ``sfd2_gate_tensor``/``sfd2_up_tensor`` are the two half views of the
        fused sfd2 buffer (concat halves when d_deinterleaved, even/odd
        column-strided otherwise).
        """
        self._logger.debug("Entering execute")
        if current_stream is None:
            current_stream = default_stream(detect_framework(a_tensor))
        if a_tensor.shape[0] == 0:
            self._logger.debug("execute: valid_m is zero, skipping kernel execution")
            return
        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")

        if self._has_dbias:
            self._value_error_if(dbias_tensor is None, "dbias_tensor must be provided at execute() when the API was compiled with sample_dbias")
        else:
            self._value_error_if(dbias_tensor is not None, "dbias_tensor must be omitted at execute() when the API was compiled without sample_dbias")
        if self.d_quant:
            self._value_error_if(
                d_quant_sf_tensor is None or norm_const_tensor is None,
                "d_quant_sf_tensor and norm_const_tensor must be provided at execute() for the quantized D output",
            )
        else:
            self._value_error_if(
                d_quant_sf_tensor is not None or norm_const_tensor is not None,
                "d_quant_sf_tensor/norm_const_tensor must be omitted for the bf16 D output",
            )

        valid_m = a_tensor.shape[0]
        n = self._n_from_descs()
        self._assert_sfd2_liveness(valid_m, n)
        workspace_ptr = self._ensure_workspace(valid_m, n)

        common = dict(
            a_tensor=a_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfa2_tensor=sfa2_tensor,
            sfd2_up_tensor=sfd2_up_tensor,
            sfd2_gate_tensor=sfd2_gate_tensor,
            sfd_row_tensor=d_quant_sf_tensor,
            norm_const_tensor=norm_const_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            dbias_tensor=dbias_tensor,
            workspace_ptr=workspace_ptr,
            stream=current_stream,
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._compiled_kernel(b_arg=b_tensor, sfb_arg=sfb_tensor, sfb2_arg=sfb2_tensor, **common)
        else:
            self._compiled_kernel(b_arg=b_ptrs, sfb_arg=sfb_ptrs, sfb2_arg=sfb2_ptrs, **common)

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDswigluSubchannelScaledSm100Objects = {}


def grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfa2_tensor: torch.Tensor,
    c_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    sfb2_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    sfb2_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    d_dtype: Optional[torch.dtype] = None,
    norm_const_tensor: Optional[torch.Tensor] = None,
    dbias: bool = False,
    dbias_dtype: Optional[torch.dtype] = None,
    acc_dtype: Optional[torch.dtype] = None,
    block2_shape: Tuple[int, int, int] = (1, 256, 256),
    mma_tiler_mn: Tuple[int, int] = DEFAULT_CTA_SHAPE,
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = True,
    m_aligned: int = 256,
    use_dynamic_sched: bool = False,
    d_deinterleaved: bool = True,
    glu_clamp_max: Optional[float] = 7.0,
    glu_clamp_min: Optional[float] = -7.0,
    dsmem_rowwise: bool = True,
    current_stream: Optional[cuda.CUstream] = None,
    sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
) -> TupleDict:
    """Convenience wrapper for the subchannel-scaled dSwiGLU grouped GEMM.

    Creates the API, compiles, and executes in one call; compiled kernels are
    cached per configuration. The wrapper owns the OUTPUT allocations and their
    zero-init contract (dprob/dbias/sfd2 are atomically accumulated).

    Args:
        a_tensor: Upstream gradient dY (valid_m, k, 1), FP4, k-major
        sfa_tensor: First-level scale factor A (MMA-tiled view)
        sfa2_tensor: Second-level scale factor A, (ceil(valid_m/sgm), ceil(k/sgk), 1) FP32,
            stride (1, rows, rows * cols)
        c_tensor: Forward FC1 pre-activations (valid_m, 2n, 1) BF16, gate/up
            interleaved in 32-column bands
        padded_offsets: End offset per expert after padding (l,), int32, cumulative
        alpha_tensor: Per-expert accumulator scale (l,), f32 (applied SQUARED)
        beta_tensor: Per-expert C scale (l,), f32
        prob_tensor: Per-token routing probability (valid_m, 1, 1), f32
        b_tensor: (Dense) FC2 weights (n, k, l), FP4, k-major
        sfb_tensor: (Dense) First-level scale factor B (MMA-tiled view)
        sfb2_tensor: (Dense) Second-level scale factor B, (ceil(n/sgn), ceil(k/sgk), l) FP32
        b_ptrs/sfb_ptrs/sfb2_ptrs: (Discrete) 1-D int64 device tensors of
            per-expert base pointers (SFB2 bases must be 16B-aligned)
        n: (Discrete) B weight N dimension (the GEMM width)
        b_dtype: (Discrete) B weight data type
        b_major: (Discrete) B major dimension (must be "k")
        d_dtype: Output D dtype -- torch.bfloat16 (default; plain output) or
            torch.float4_e2m1fn_x2 (NVFP4-quantized output)
        norm_const_tensor: (1,) f32 global D-quant encode scale; required iff
            d_dtype is fp4
        dbias: Also produce the per-expert dbias output
        dbias_dtype: torch.bfloat16 (default) or torch.float32 (testing-only)
        block2_shape: Second-level scale granularity (sgm, sgn, sgk); sgn must
            be a multiple of 128 (the MMA tile N)
        d_deinterleaved: Store d/d_quant_sf/dbias deinterleaved [gate | up] and
            sfd2 as concatenated halves (default True; requires the quantized-D
            output). In this layout the fused sfd2 output is directly a
            consumer GEMM's a_scales2 grid with sgk = sgn.
        sf_fp8_dtype_override: Reinterpret the FP8 first-level scale factors as
            E5M3 (Rubin-only); the quantized-D row scales come out e5m3-encoded
            as well. Part of the compile cache key.
        (remaining config parameters mirror the class constructor)

    Returns:
        TupleDict with keys (also the unpack order):
        - **d_tensor**: BF16 (valid_m, 2n, 1) output; None in quantized mode
        - **d_quant_tensor**: FP4 (physical (valid_m, n, 1), logical 2n wide)
          packed output; None in bf16 mode
        - **d_quant_sf_tensor**: e4m3 MMA-tiled row scale factors; None in bf16 mode
        - **dprob_tensor**: f32 (valid_m, 1, 1), always produced
        - **dbias_tensor**: (l, 2n, 1) dbias or None
        - **sfd2_tensor**: fused f32 (rows, 2*ceil(n/sgn), 1) second-level descales
        - **sfd2_gate_tensor** / **sfd2_up_tensor**: live half views of sfd2_tensor
    """
    from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor

    framework = detect_framework(a_tensor)
    if framework == "jax":
        raise ValueError(f"grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100 does not support JAX arrays: {_JAX_SF_LAYOUT_ERROR}")
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}'; pass torch tensors")
    import torch

    acc_dtype = _convert_to_cutlass_data_type(acc_dtype) if acc_dtype is not None else cutlass.Float32
    d_dtype_torch = d_dtype if d_dtype is not None else torch.bfloat16
    dbias_dtype_torch = dbias_dtype if dbias_dtype is not None else torch.bfloat16
    d_quant = d_dtype_torch != torch.bfloat16

    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor, sfb2_tensor) or (b_ptrs, sfb_ptrs, sfb2_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor, sfb2_tensor) or (b_ptrs, sfb_ptrs, sfb2_ptrs)")

    valid_m, k_physical, _ = a_tensor.shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        if sfb_tensor is None or sfb2_tensor is None:
            raise ValueError("sfb_tensor and sfb2_tensor are required in dense mode")
        n_gemm, _, l = b_tensor.shape
        num_experts = None
        b_shape = None
    else:
        weight_mode = MoEWeightMode.DISCRETE
        num_experts = _validate_pointer_tensor(b_ptrs, "b_ptrs")
        _validate_pointer_tensor(sfb_ptrs, "sfb_ptrs", num_experts)
        _validate_pointer_tensor(sfb2_ptrs, "sfb2_ptrs", num_experts)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        b_dtype_cutlass = _convert_to_cutlass_data_type(b_dtype)
        k_logical = k_physical * 2 if b_dtype_cutlass in (cutlass.Float4E2M1FN, cutlass.Uint8) else k_physical
        b_shape = (n, k_logical)
        n_gemm = n
        l = num_experts

    n2 = 2 * n_gemm
    sgm, sgn, sgk = block2_shape
    device = a_tensor.device

    if d_quant:
        if norm_const_tensor is None:
            raise ValueError("norm_const_tensor is required for the quantized (fp4) D output")
    elif norm_const_tensor is not None:
        raise ValueError("norm_const_tensor must be omitted for the bf16 D output")

    _logger.debug("grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100: Creating output tensors")

    # ---- outputs ----
    if d_quant:
        # Physical fp4x2 (valid_m, n2/2, 1); the descriptor doubles the packed
        # innermost dim to the logical (valid_m, n2, 1) the kernel expects.
        d_out = torch.zeros((1, valid_m, n2 // 2), dtype=torch.uint8, device=device).view(torch.float4_e2m1fn_x2).permute(1, 2, 0)
        d_quant_sf_out = torch.zeros(
            (1, ceil_div(valid_m, 128), ceil_div(ceil_div(n2, 16), 4), 32, 4, 4),
            dtype=torch.float8_e4m3fn,
            device=device,
        ).permute(3, 4, 1, 5, 2, 0)
        d_bf16_out = None
    else:
        d_bf16_out = torch.empty_strided((valid_m, n2, 1), (n2, 1, valid_m * n2), dtype=torch.bfloat16, device=device)
        d_quant_sf_out = None
    d_kernel_tensor = d_out if d_quant else d_bf16_out

    dprob_out = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=device)
    dbias_out = torch.zeros((l, n2, 1), dtype=dbias_dtype_torch, device=device) if dbias else None

    sfd2_rows = ceil_div(valid_m, sgm)
    nd = ceil_div(n_gemm, sgn)
    sfd2_buf = torch.zeros((1, 2 * nd, sfd2_rows), dtype=torch.float32, device=device).permute(2, 1, 0)
    if d_deinterleaved:
        sfd2_gate_view = sfd2_buf[:, :nd, :]
        sfd2_up_view = sfd2_buf[:, nd:, :]
    else:
        sfd2_gate_view = sfd2_buf[:, 0::2, :]
        sfd2_up_view = sfd2_buf[:, 1::2, :]

    if valid_m == 0:
        return TupleDict(
            d_tensor=d_bf16_out,
            d_quant_tensor=d_out if d_quant else None,
            d_quant_sf_tensor=d_quant_sf_out,
            dprob_tensor=dprob_out,
            dbias_tensor=dbias_out,
            sfd2_tensor=sfd2_buf,
            sfd2_gate_tensor=sfd2_gate_view,
            sfd2_up_tensor=sfd2_up_view,
        )

    def tensor_signature(tensor):
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def stride_order(tensor):
        return tuple(i for i, s in sorted(enumerate(tensor.stride()), key=lambda x: x[1]))

    def dynamic_m_tensor_signature(tensor, static_shape_suffix, dynamic_stride_dims=()):
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if i in dynamic_stride_dims else s for i, s in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    device_type = get_device_type()
    common_key_tail = (
        tuple(padded_offsets.shape),
        padded_offsets.dtype,
        acc_dtype,
        d_dtype_torch,
        dbias,
        dbias_dtype_torch,
        tuple(block2_shape),
        tuple(mma_tiler_mn),
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        use_dynamic_sched,
        d_deinterleaved,
        glu_clamp_max,
        glu_clamp_min,
        dsmem_rowwise,
        sf_fp8_dtype_override,
    )
    dynamic_m_key = (
        a_tensor.shape[1:],
        stride_order(a_tensor),
        a_tensor.dtype,
        c_tensor.shape[1:],
        stride_order(c_tensor),
        *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1), dynamic_stride_dims=(5,)),
        *dynamic_m_tensor_signature(sfa2_tensor, (sfa2_tensor.shape[1], 1), dynamic_stride_dims=(1, 2)),
        *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
        *tensor_signature(alpha_tensor),
        *tensor_signature(beta_tensor),
    )
    if is_dense:
        cache_key = (
            device_type,
            weight_mode,
            *dynamic_m_key,
            *tensor_signature(b_tensor),
            *tensor_signature(sfb_tensor),
            *tensor_signature(sfb2_tensor),
            *common_key_tail,
        )
    else:
        cache_key = (
            device_type,
            weight_mode,
            *dynamic_m_key,
            b_shape,
            b_dtype,
            tuple(b_ptrs.shape),
            b_ptrs.dtype,
            *common_key_tail,
            b_major,
            num_experts,
        )

    if cache_key in _cache_of_GroupedGemmDswigluSubchannelScaledSm100Objects:
        api = _cache_of_GroupedGemmDswigluSubchannelScaledSm100Objects[cache_key]
    else:
        _logger.debug("grouped_gemm_dswiglu_subchannel_scaled_wrapper_sm100: creating new API object")
        common_kwargs = dict(
            sample_a=a_tensor,
            sample_sfa=sfa_tensor,
            sample_sfa2=sfa2_tensor,
            sample_c=c_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_beta=beta_tensor,
            sample_prob=prob_tensor,
            sample_d=d_kernel_tensor,
            sample_dprob=dprob_out,
            sample_sfd2=sfd2_buf,
            sample_dbias=dbias_out,
            sample_d_quant_sf=d_quant_sf_out,
            sample_norm_const=norm_const_tensor,
            acc_dtype=acc_dtype,
            block2_shape=block2_shape,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            use_dynamic_sched=use_dynamic_sched,
            d_deinterleaved=d_deinterleaved,
            glu_clamp_max=glu_clamp_max,
            glu_clamp_min=glu_clamp_min,
            dsmem_rowwise=dsmem_rowwise,
            sf_fp8_dtype_override=sf_fp8_dtype_override,
        )
        if is_dense:
            api = GroupedGemmDswigluSubchannelScaledSm100(
                sample_b=b_tensor,
                sample_sfb=sfb_tensor,
                sample_sfb2=sfb2_tensor,
                **common_kwargs,
            )
        else:
            api = GroupedGemmDswigluSubchannelScaledSm100(
                num_experts=num_experts,
                b_shape=b_shape,
                b_dtype=b_dtype,
                b_major=b_major,
                **common_kwargs,
            )
        assert api.check_support(), "Unsupported configuration"
        api.compile()
        _cache_of_GroupedGemmDswigluSubchannelScaledSm100Objects[cache_key] = api

    exec_kwargs = dict(
        a_tensor=a_tensor,
        sfa_tensor=sfa_tensor,
        sfa2_tensor=sfa2_tensor,
        c_tensor=c_tensor,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha_tensor,
        beta_tensor=beta_tensor,
        prob_tensor=prob_tensor,
        d_tensor=d_kernel_tensor,
        dprob_tensor=dprob_out,
        sfd2_gate_tensor=sfd2_gate_view,
        sfd2_up_tensor=sfd2_up_view,
        dbias_tensor=dbias_out,
        d_quant_sf_tensor=d_quant_sf_out,
        norm_const_tensor=norm_const_tensor,
        current_stream=current_stream,
    )
    if is_dense:
        api.execute(b_tensor=b_tensor, sfb_tensor=sfb_tensor, sfb2_tensor=sfb2_tensor, **exec_kwargs)
    else:
        api.execute(b_ptrs=b_ptrs, sfb_ptrs=sfb_ptrs, sfb2_ptrs=sfb2_ptrs, **exec_kwargs)

    return TupleDict(
        d_tensor=d_bf16_out,
        d_quant_tensor=d_out if d_quant else None,
        d_quant_sf_tensor=d_quant_sf_out,
        dprob_tensor=dprob_out,
        dbias_tensor=dbias_out,
        sfd2_tensor=sfd2_buf,
        sfd2_gate_tensor=sfd2_gate_view,
        sfd2_up_tensor=sfd2_up_view,
    )
