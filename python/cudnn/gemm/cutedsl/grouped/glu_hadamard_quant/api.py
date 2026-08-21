# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FE API for grouped GEMM GLU forward fusion with fused RHT output and NVFP4 quantization.

Output modes are dtype driven, mirroring the kernel:
  - D is bf16, or NVFP4 (packed e2m1 data + e4m3/ue5m3 block scales in ``sfd``)
  - The optional RHT output is bf16, or NVFP4 (packed e2m1 data + e4m3/ue5m3
    block scales in ``sfrht``)

The RHT data is always stored at D's own (m, f) orientation — only the SCALE grid
follows the transform orientation: swizzled scale factors for logical (m, f)
when ``rht_rowwise``, and swizzled scale factors for logical (f, m) otherwise.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, Optional, Tuple

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.api_base import APIBase, TupleDict, ceil_div, get_device_type, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type

from ..moe_utils import MoEWeightMode
from .rht_utils import HADAMARD_SIZE
from .moe_blockscaled_grouped_gemm_glu_hadamard_quant import BlockScaledMoEGroupedGemmGluHadamardQuantKernel


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_glu_hadamard_quant_rubin import (
        BlockScaledMoEGroupedGemmGluHadamardQuantKernel as RubinBlockScaledMoEGroupedGemmGluHadamardQuantKernel,
    )

    return RubinBlockScaledMoEGroupedGemmGluHadamardQuantKernel


# The GLU + Hadamard + quant fusion is block-scaled only: its mandatory
# scale-factor inputs use an MMA-interleaved 6-D layout with no row-major
# equivalent, so they are not expressible as JAX arrays and the API stays
# torch-only.
_JAX_ERROR = (
    "grouped GEMM GLU hadamard quant is not supported for JAX arrays: the block-scaled "
    "scale-factor tensors (sfa/sfb) use an MMA-interleaved layout that is not expressible as JAX arrays; "
    "pass torch tensors"
)


def _require_torch_inputs(sample: Any, api_name: str) -> None:
    from cudnn.tensor_adapter import detect_framework

    framework = detect_framework(sample)
    if framework == "jax":
        raise ValueError(_JAX_ERROR)
    if framework != "torch":
        raise ValueError(f"Unsupported tensor framework '{framework}' for {api_name}; pass torch tensors")


def _sf_layout_shape(rows: int, cols: int, sf_vec_size: int) -> Tuple[int, int, int, int, int, int]:
    return (32, 4, ceil_div(rows, 128), 4, ceil_div(ceil_div(cols, sf_vec_size), 4), 1)


class GroupedGemmGluHadamardQuantSm100(APIBase):
    """Grouped GEMM GLU forward kernel with fused RHT output and NVFP4 quantization."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        sample_sfb: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        b_major: str = "k",
        sample_sfd: Optional[torch.Tensor] = None,
        sample_rht: Optional[torch.Tensor] = None,
        sample_sfrht: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        act_func: str = "swiglu",
        use_dynamic_sched: bool = False,
        rht_rowwise: bool = False,
        glu_alpha: Optional[float] = None,
        glu_limit: Optional[float] = None,
    ):
        if sample_a is not None:
            _require_torch_inputs(sample_a, "GroupedGemmGluHadamardQuantSm100")
        import torch

        if acc_dtype is None:
            acc_dtype = torch.float32
        super().__init__()

        self._warn_experimental_api()
        self._interpret_uint8_as_fp4x2 = True
        self._sample_a_tensor = sample_a
        self._sample_b_tensor = sample_b
        self._sample_d_tensor = sample_d
        self._sample_rht_tensor = sample_rht

        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
            if sample_sfb is None:
                raise ValueError("sample_sfb is required when sample_b is provided")
        elif num_experts is not None and sample_b is None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError("Provide either (sample_b, sample_sfb) or (num_experts, b_shape, b_dtype)")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a", interpret_uint8_as_fp4x2=False)
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d", interpret_uint8_as_fp4x2=False)
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")
        self.sfd_desc = self._make_tensor_desc(sample_sfd, name="sample_sfd")
        self.rht_desc = self._make_tensor_desc(sample_rht, name="sample_rht", interpret_uint8_as_fp4x2=False)
        self.sfrht_desc = self._make_tensor_desc(sample_sfrht, name="sample_sfrht")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        if self.weight_mode == MoEWeightMode.DENSE:
            self.b_desc = self._make_tensor_desc(sample_b, name="sample_b", interpret_uint8_as_fp4x2=False)
            self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
            self.expert_cnt = self.padded_offsets_desc.shape[0]
            self.b_shape = None
            self.b_dtype = None
            self.b_major = b_major
        else:
            self.b_desc = None
            self.sfb_desc = None
            self.expert_cnt = num_experts
            self.b_shape = b_shape
            self.b_dtype = b_dtype
            self.b_major = b_major
            self._value_error_if(
                self.padded_offsets_desc.shape[0] != self.expert_cnt,
                f"padded_offsets length ({self.padded_offsets_desc.shape[0]}) must equal num_experts ({self.expert_cnt})",
            )

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn if cluster_shape_mn is not None else ((2, 1) if self.use_2cta_instrs else (1, 1))
        self.sf_vec_size = sf_vec_size
        self.sf_fp8_dtype_override = sf_fp8_dtype_override
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.act_func = act_func
        self.use_dynamic_sched = use_dynamic_sched
        self.rht_rowwise = rht_rowwise
        self.glu_alpha = glu_alpha
        self.glu_limit = glu_limit
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledMoEGroupedGemmGluHadamardQuantKernel
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self._workspace = None

    def check_support(self) -> bool:
        import torch

        tensor_m, k, _ = self._tensor_shape(self.a_desc, name="sample_a")
        if self.weight_mode == MoEWeightMode.DENSE:
            n, _, l = self._tensor_shape(self.b_desc, name="sample_b")
        else:
            if len(self.b_shape) == 2:
                n, b_k = self.b_shape
            else:
                n, b_k, _ = self.b_shape
            self._value_error_if(b_k != k, f"B K dimension ({b_k}) must match A K dimension ({k})")
            l = self.expert_cnt
        _, n_c, _ = self._tensor_shape(self.c_desc, name="sample_c")
        _, n_d, _ = self._tensor_shape(self.d_desc, name="sample_d")
        n_out = n if self.act_func == "srelu" else n // 2

        self._value_error_if(l != self.expert_cnt, f"B L dimension ({l}) must match expert_cnt ({self.expert_cnt})")
        self._value_error_if(n % 64 != 0, f"N must be divisible by 64, got {n}")
        self._value_error_if(n_out % (2 * HADAMARD_SIZE) != 0, f"D N dimension must be divisible by {2 * HADAMARD_SIZE}, got {n_out}")

        # ---- Output / dump modes (dtype driven, mirroring the kernel) ----
        self.d_quant = self._is_fp4x2(self.d_desc)
        self.generate_rht = self.rht_desc is not None
        self.rht_quant = self.generate_rht and self._is_fp4x2(self.rht_desc)
        self._value_error_if(
            self.d_quant != (self.sfd_desc is not None),
            "NVFP4 sample_d and sample_sfd must be passed together",
        )
        self._value_error_if(
            self.rht_quant != (self.sfrht_desc is not None),
            "NVFP4 sample_rht and sample_sfrht must be passed together",
        )
        self._value_error_if(
            (self.d_quant or self.rht_quant) and n_out % (8 * HADAMARD_SIZE) != 0,
            f"NVFP4 quantization requires the D N dimension to be divisible by {8 * HADAMARD_SIZE}, got {n_out}",
        )
        self._value_error_if(
            (self.d_quant or self.rht_quant) and self.act_func == "srelu",
            "NVFP4 quantization is not supported with act_func 'srelu'",
        )

        self._check_tensor_shape(self.a_desc, (tensor_m, k, 1), "A")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.b_desc, (n, k, l), "B")
        self._check_tensor_shape(self.c_desc, (tensor_m, n, 1), "C")
        self._check_tensor_shape(self.d_desc, (tensor_m, n_out, 1), "D")
        self._check_tensor_shape(self.sfa_desc, (32, 4, ceil_div(tensor_m, 128), 4, ceil_div(ceil_div(k, self.sf_vec_size), 4), 1), "SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.sfb_desc, (32, 4, ceil_div(n, 128), 4, ceil_div(ceil_div(k, self.sf_vec_size), 4), l), "SFB")
        self._check_tensor_shape(self.padded_offsets_desc, (l,), "padded_offsets")
        self._check_tensor_shape(self.alpha_desc, (l,), "alpha")
        self._check_tensor_shape(self.prob_desc, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.bias_desc, (n, l), "bias")
        if self.d_quant:
            self._check_tensor_shape(self.sfd_desc, _sf_layout_shape(tensor_m, n_out, self.sf_vec_size), "SFD")
        if self.generate_rht:
            self._check_tensor_shape(self.rht_desc, (tensor_m, n_out, 1), "RHT")
        if self.rht_quant:
            if self.rht_rowwise:
                self._check_tensor_shape(self.sfrht_desc, _sf_layout_shape(tensor_m, n_out, self.sf_vec_size), "SFRHT")
            else:
                self._check_tensor_shape(self.sfrht_desc, _sf_layout_shape(n_out, tensor_m, self.sf_vec_size), "SFRHT")

        self._check_tensor_stride(self.a_desc, stride=[(k, 1, tensor_m * k)], name="A", extra_error_msg="A must have k-major layout")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_stride(self.b_desc, stride=[(k, 1, n * k)], name="B", extra_error_msg="B must have k-major layout")
        self._check_tensor_stride(self.c_desc, stride=[(n_c, 1, tensor_m * n_c)], name="C", extra_error_msg="C must have n-major layout")
        self._check_tensor_stride(self.d_desc, stride=[(n_d, 1, tensor_m * n_d)], name="D", extra_error_msg="D must have n-major layout")
        self._check_tensor_stride(self.bias_desc, stride=[(1, n)], name="bias")
        if self.generate_rht:
            self._check_tensor_stride(self.rht_desc, stride=[(n_d, 1, tensor_m * n_d)], name="RHT", extra_error_msg="RHT must have n-major layout")

        self.ab_dtype = self._check_dtype(
            self.a_desc,
            dtype=[torch.float4_e2m1fn_x2],
            name="A",
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.b_desc, dtype=self.ab_dtype, name="B", extra_error_msg="B must match A dtype")
        else:
            self._value_error_if(self.b_dtype != self.ab_dtype, f"b_dtype ({self.b_dtype}) must match A dtype ({self.ab_dtype})")
            self._value_error_if(self.b_major not in ["k", "n"], f"b_major must be 'k' or 'n', got {self.b_major}")
            self._value_error_if(self._is_fp4x2(self.ab_dtype) and self.b_major != "k", "b_major must be 'k' when ab_dtype is fp4")
        self.sf_dtype = self._check_dtype(self.sfa_desc, dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn], name="SFA")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.sfb_desc, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must match SFA dtype")
        self.c_dtype = self._check_dtype(self.c_desc, dtype=[torch.float16, torch.bfloat16], name="C")
        self.d_dtype = self._check_dtype(self.d_desc, dtype=[torch.bfloat16, torch.float4_e2m1fn_x2], name="D")
        self._check_dtype(self.alpha_desc, dtype=torch.float32, name="alpha")
        self._check_dtype(self.prob_desc, dtype=torch.float32, name="prob")
        self._check_dtype(self.bias_desc, dtype=[torch.float16, torch.bfloat16, torch.float32], name="bias")
        if self.d_quant:
            self._check_dtype(self.sfd_desc, dtype=self.sf_dtype, name="SFD", extra_error_msg="SFD must match SFA dtype")
        if self.generate_rht:
            self._check_dtype(self.rht_desc, dtype=[torch.bfloat16, torch.float4_e2m1fn_x2], name="RHT")
        if self.rht_quant:
            self._check_dtype(self.sfrht_desc, dtype=self.sf_dtype, name="SFRHT", extra_error_msg="SFRHT must match SFA dtype")
        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="acc_dtype")

        self._value_error_if(self.sf_vec_size != 16, f"sf_vec_size must be 16, got {self.sf_vec_size}")
        self._value_error_if(
            self.sf_fp8_dtype_override not in (None, "e5m3"),
            f"sf_fp8_dtype_override must be None or 'e5m3', got {self.sf_fp8_dtype_override!r}",
        )
        if self.sf_fp8_dtype_override == "e5m3":
            self._value_error_if(
                self.sf_dtype != torch.float8_e4m3fn,
                f"sf_fp8_dtype_override='e5m3' requires torch.float8_e4m3fn scale-factor storage at sf_vec_size 16, "
                f"got sf_dtype={self.sf_dtype}, sf_vec_size={self.sf_vec_size}",
            )
            self._value_error_if(
                not self._is_rubin_kernel,
                f"sf_fp8_dtype_override='e5m3' requires Rubin (SM107), got device type {self._device_type!r}",
            )
        self._value_error_if(
            self.act_func not in ["swiglu", "geglu", "srelu"],
            f"act_func must be 'swiglu', 'geglu', or 'srelu', got {self.act_func}",
        )
        self._value_error_if(
            not self.use_2cta_instrs or self.mma_tiler_mn != (256, 256), f"RHT fusion requires mma_tiler_mn=(256, 256), got {self.mma_tiler_mn}"
        )
        self._value_error_if(self.cluster_shape_mn[0] % 2 != 0, f"cluster_shape_mn[0] must be divisible by 2, got {self.cluster_shape_mn[0]}")
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
            f"Invalid cluster shape: {self.cluster_shape_mn}",
        )
        self._value_error_if(
            self.m_aligned != self._kernel.FIX_PAD_SIZE,
            f"m_aligned must be {self._kernel.FIX_PAD_SIZE}, got {self.m_aligned}",
        )
        self._value_error_if(self.expert_cnt > 1024, f"expert_cnt must be <= 1024, got {self.expert_cnt}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmGluHadamardQuantSm100 requires SM100+, found SM{compute_capability}")

        if not self._kernel.can_implement(
            _convert_to_cutlass_data_type(self.ab_dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2),
            _convert_to_cutlass_data_type(self.sf_dtype),
            self.sf_vec_size,
            _convert_to_cutlass_data_type(self.acc_dtype),
            _convert_to_cutlass_data_type(self.d_desc.dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            self.m_aligned,
            n,
            k,
            l,
            "k",
            self.b_major,
            "n",
            self.m_aligned,
            rht_quant=self.rht_quant,
            d_quant=self.d_quant,
        ):
            raise RuntimeError("Unsupported grouped GEMM GLU Hadamard Quant configuration")

        self._is_supported = True
        return True

    def compile(self) -> None:
        import torch

        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        if self.a_desc.shape[0] == 0:
            return

        kernel_kwargs = dict(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func=self.act_func,
            enable_bias=self.bias_desc is not None,
            rht_rowwise=self.rht_rowwise if self.generate_rht else False,
            glu_alpha=self.glu_alpha,
            glu_limit=self.glu_limit,
        )
        if self.sf_fp8_dtype_override == "e5m3":
            kernel_kwargs["sf_fp8_dtype_override"] = self.sf_fp8_dtype_override
        kernel = self._kernel(**kernel_kwargs)

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        max_active_clusters -= self.num_cluster_overlap_margin
        self._value_error_if(max_active_clusters <= 0, "max_active_clusters must be > 0 after overlap margin")
        self._workspace = torch.empty(max(kernel.get_workspace_bytes(), 1), dtype=torch.uint8, device="cuda")
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        fake_workspace_ptr = cute.runtime.nullptr(dtype=cutlass.Uint8, assumed_align=128)
        cached_workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator

        valid_m = cute.sym_int(divisibility=self.m_aligned)
        tensor_m_128 = cute.sym_int()
        stride_sfa_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)

        a_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=(valid_m, self.a_desc.shape[1], 1),
            stride_order=self.a_desc.stride_order,
            dynamic_mode=self.a_desc.stride_order[0],
            divisibility=32 if self._is_fp4x2(self.ab_dtype) else 16,
        )
        c_cute_fake = self._make_fake_cute_compact_tensor(
            dtype=self.c_desc.dtype,
            shape=(valid_m, self.c_desc.shape[1], 1),
            stride_order=self.c_desc.stride_order,
            dynamic_mode=self.c_desc.stride_order[0],
            divisibility=8 if self._is_f16(self.c_desc) else 16,
        )
        d_cute_arg = self._make_fake_cute_compact_tensor(
            dtype=self.d_desc.dtype,
            shape=(valid_m, self.d_desc.shape[1], 1),
            stride_order=self.d_desc.stride_order,
            dynamic_mode=self.d_desc.stride_order[0],
            divisibility=8 if self._is_f16(self.d_desc) else 32,
        )
        sfd_cute_arg = None
        if self.d_quant:
            stride_sfd_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
            sfd_cute_arg = self._make_fake_cute_tensor(
                dtype=self.sfd_desc.dtype,
                shape=(32, 4, tensor_m_128, 4, self.sfd_desc.shape[4], 1),
                stride=(16, 4, self.sfd_desc.stride[2], 1, 512, stride_sfd_tensor_m_128),
            )
        rht_cute_arg = None
        if self.generate_rht:
            rht_cute_arg = self._make_fake_cute_compact_tensor(
                dtype=self.rht_desc.dtype,
                shape=(valid_m, self.rht_desc.shape[1], 1),
                stride_order=self.rht_desc.stride_order,
                dynamic_mode=self.rht_desc.stride_order[0],
                divisibility=8 if self._is_f16(self.rht_desc) else 32,
            )
        sfrht_cute_arg = None
        if self.rht_quant:
            if self.rht_rowwise:
                stride_sfrht_tensor_m_128 = cute.sym_int(divisibility=32 * 4 * 4)
                sfrht_cute_arg = self._make_fake_cute_tensor(
                    dtype=self.sfrht_desc.dtype,
                    shape=(32, 4, tensor_m_128, 4, self.sfrht_desc.shape[4], 1),
                    stride=(16, 4, self.sfrht_desc.stride[2], 1, 512, stride_sfrht_tensor_m_128),
                )
            else:
                sfrht_rest_m = cute.sym_int()
                stride_sfrht_rest_m = cute.sym_int(divisibility=32 * 4 * 4)
                stride_sfrht_l = cute.sym_int(divisibility=32 * 4 * 4)
                sfrht_cute_arg = self._make_fake_cute_tensor(
                    dtype=self.sfrht_desc.dtype,
                    shape=(32, 4, self.sfrht_desc.shape[2], 4, sfrht_rest_m, 1),
                    stride=(16, 4, stride_sfrht_rest_m, 1, 512, stride_sfrht_l),
                )
        prob_cute_fake = self._make_fake_cute_tensor(
            dtype=self.prob_desc.dtype,
            shape=(valid_m, 1, 1),
            stride=self.prob_desc.stride,
        )
        sfa_cute_fake = self._make_fake_cute_tensor(
            dtype=self.sfa_desc.dtype,
            shape=(32, 4, tensor_m_128, 4, self.sfa_desc.shape[4], 1),
            stride=(16, 4, self.sfa_desc.stride[2], 1, 512, stride_sfa_tensor_m_128),
        )
        if self.weight_mode == MoEWeightMode.DENSE:
            b_cute_arg = self._make_fake_cute_tensor_from_desc(self.b_desc, assumed_align=16)
            sfb_cute_arg = self._make_fake_cute_tensor_from_desc(self.sfb_desc, assumed_align=16)
            n_arg = cutlass.Int32(0)
            k_arg = cutlass.Int32(0)
            b_stride_arg = cutlass.Int64(0)
            b_major_arg = OperandMajorMode.K
            workspace_arg = fake_workspace_ptr
        else:
            if len(self.b_shape) == 2:
                n_compile, k_compile = self.b_shape
            else:
                n_compile, k_compile, _ = self.b_shape
            b_major_arg = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN
            b_stride_size = k_compile if self.b_major == "k" else n_compile
            b_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
            sfb_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device="cuda")
            b_cute_arg = from_dlpack(b_ptrs_placeholder, assumed_align=8).iterator
            sfb_cute_arg = from_dlpack(sfb_ptrs_placeholder, assumed_align=8).iterator
            n_arg = cutlass.Int32(n_compile)
            k_arg = cutlass.Int32(k_compile)
            b_stride_arg = cutlass.Int64(b_stride_size)
            workspace_arg = cached_workspace_ptr
            self._n = n_compile
            self._k = k_compile
            self._b_stride_size = b_stride_size
        alpha_cute_fake = self._make_fake_cute_tensor_from_desc(self.alpha_desc, assumed_align=16)
        padded_offsets_cute_fake = self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc, assumed_align=16)
        bias_cute_fake = self._make_fake_cute_tensor_from_desc(self.bias_desc, assumed_align=16)
        cached_linear_offset = cutlass.Float32(1.0 if self.act_func == "geglu" else 0.0)

        compiled_kernel = cute.compile(
            kernel,
            a_cute_fake,
            b_cute_arg,
            sfa_cute_fake,
            sfb_cute_arg,
            n_arg,
            k_arg,
            b_stride_arg,
            b_major_arg,
            workspace_arg,
            c_cute_fake,
            d_cute_arg,
            sfd_cute_arg,
            rht_cute_arg,
            sfrht_cute_arg,
            padded_offsets_cute_fake,
            alpha_cute_fake,
            prob_cute_fake,
            bias_cute_fake,
            max_active_clusters,
            fake_stream,
            linear_offset=cached_linear_offset,
            norm_const=cutlass.Float32(1.0),
            rht_norm_const=cutlass.Float32(1.0),
            options="--enable-tvm-ffi",
        )

        if self.weight_mode == MoEWeightMode.DENSE:

            def tensor_api(
                a_tensor: torch.Tensor,
                b_tensor: torch.Tensor,
                c_tensor: torch.Tensor,
                d_tensor: torch.Tensor,
                sfa_tensor: torch.Tensor,
                sfb_tensor: torch.Tensor,
                padded_offsets: torch.Tensor,
                alpha_tensor: torch.Tensor,
                prob_tensor: torch.Tensor,
                sfd_tensor: Optional[torch.Tensor],
                rht_tensor: Optional[torch.Tensor],
                sfrht_tensor: Optional[torch.Tensor],
                bias_tensor: Optional[torch.Tensor],
                norm_const: float,
                rht_norm_const: float,
                stream: cuda.CUstream,
            ) -> None:
                compiled_kernel(
                    a_tensor,
                    b_tensor,
                    sfa_tensor,
                    sfb_tensor,
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int64(0),
                    cached_workspace_ptr,
                    c_tensor,
                    d_tensor,
                    sfd_tensor,
                    rht_tensor,
                    sfrht_tensor,
                    padded_offsets,
                    alpha_tensor,
                    prob_tensor,
                    bias_tensor,
                    stream,
                    cached_linear_offset,
                    cutlass.Float32(norm_const),
                    cutlass.Float32(rht_norm_const),
                )

            self._compiled_kernel = tensor_api
        else:
            cached_n = cutlass.Int32(self._n)
            cached_k = cutlass.Int32(self._k)
            cached_b_stride = cutlass.Int64(self._b_stride_size)

            def tensor_api(
                a_tensor: torch.Tensor,
                b_ptrs_device: torch.Tensor,
                sfb_ptrs_device: torch.Tensor,
                c_tensor: torch.Tensor,
                d_tensor: torch.Tensor,
                sfa_tensor: torch.Tensor,
                padded_offsets: torch.Tensor,
                alpha_tensor: torch.Tensor,
                prob_tensor: torch.Tensor,
                sfd_tensor: Optional[torch.Tensor],
                rht_tensor: Optional[torch.Tensor],
                sfrht_tensor: Optional[torch.Tensor],
                bias_tensor: Optional[torch.Tensor],
                norm_const: float,
                rht_norm_const: float,
                stream: cuda.CUstream,
            ) -> None:
                compiled_kernel(
                    a_tensor,
                    int(b_ptrs_device.data_ptr()),
                    sfa_tensor,
                    int(sfb_ptrs_device.data_ptr()),
                    cached_n,
                    cached_k,
                    cached_b_stride,
                    cached_workspace_ptr,
                    c_tensor,
                    d_tensor,
                    sfd_tensor,
                    rht_tensor,
                    sfrht_tensor,
                    padded_offsets,
                    alpha_tensor,
                    prob_tensor,
                    bias_tensor,
                    stream,
                    cached_linear_offset,
                    cutlass.Float32(norm_const),
                    cutlass.Float32(rht_norm_const),
                )

            self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        sfb_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        sfb_ptrs: Optional[torch.Tensor] = None,
        sfd_tensor: Optional[torch.Tensor] = None,
        rht_tensor: Optional[torch.Tensor] = None,
        sfrht_tensor: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        norm_const: float = 1.0,
        rht_norm_const: float = 1.0,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        import torch

        self._ensure_support_checked()
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel has not been compiled")
        if a_tensor.shape[0] == 0:
            return
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(a_tensor.device).cuda_stream)
        if self.d_quant and sfd_tensor is None:
            raise ValueError("sfd_tensor must be provided when D is NVFP4")
        if self.generate_rht and rht_tensor is None:
            raise ValueError("rht_tensor must be provided when the RHT output is enabled")
        if self.rht_quant and sfrht_tensor is None:
            raise ValueError("sfrht_tensor must be provided when the RHT output is NVFP4")

        if self.weight_mode == MoEWeightMode.DENSE:
            if b_tensor is None or sfb_tensor is None:
                raise ValueError("b_tensor and sfb_tensor must be provided in dense mode")
            self._compiled_kernel(
                a_tensor,
                b_tensor,
                c_tensor,
                d_tensor,
                sfa_tensor,
                sfb_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                sfd_tensor,
                rht_tensor,
                sfrht_tensor,
                bias_tensor,
                norm_const,
                rht_norm_const,
                current_stream,
            )
        else:
            if b_ptrs is None or sfb_ptrs is None:
                raise ValueError("b_ptrs and sfb_ptrs must be provided in discrete mode")
            self._compiled_kernel(
                a_tensor,
                b_ptrs,
                sfb_ptrs,
                c_tensor,
                d_tensor,
                sfa_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                sfd_tensor,
                rht_tensor,
                sfrht_tensor,
                bias_tensor,
                norm_const,
                rht_norm_const,
                current_stream,
            )


_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmGluHadamardQuantSm100Objects = {}


def grouped_gemm_glu_hadamard_quant_wrapper_sm100(
    a_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    b_tensor: Optional[torch.Tensor] = None,
    sfb_tensor: Optional[torch.Tensor] = None,
    b_ptrs: Optional[torch.Tensor] = None,
    sfb_ptrs: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
    b_dtype: Optional[torch.dtype] = None,
    b_major: str = "k",
    bias_tensor: Optional[torch.Tensor] = None,
    acc_dtype: Optional[torch.dtype] = None,
    c_dtype: Optional[torch.dtype] = None,
    d_dtype: Optional[torch.dtype] = None,
    cd_major: str = "n",
    rht_output: bool = True,
    rht_dtype: Optional[torch.dtype] = None,
    rht_rowwise: bool = False,
    glu_alpha: Optional[float] = None,
    glu_limit: Optional[float] = None,
    norm_const: float = 1.0,
    rht_norm_const: float = 1.0,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    sf_fp8_dtype_override: Optional[str] = None,
    vector_f32: bool = False,
    m_aligned: int = 256,
    act_func: str = "swiglu",
    use_dynamic_sched: bool = False,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper for grouped GEMM GLU forward fusion with fused RHT output.

    Output modes are dtype driven: ``d_dtype``/``rht_dtype`` of
    ``torch.float4_e2m1fn_x2`` emit packed NVFP4 data plus e4m3/ue5m3 block scales
    (``sfd_tensor``/``sfrht_tensor``); ``torch.bfloat16`` emits plain bf16.
    ``sf_fp8_dtype_override="e5m3"`` reinterprets ``torch.float8_e4m3fn``
    SFA/SFB storage as UE5M3 input scale factors on Rubin.
    ``norm_const``/``rht_norm_const`` are the NVFP4 global encode scales
    (2688/global_amax, or 1.0).
    """
    from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

    if a_tensor is not None:
        _require_torch_inputs(a_tensor, "grouped_gemm_glu_hadamard_quant_wrapper_sm100")
    import torch

    if acc_dtype is None:
        acc_dtype = torch.float32
    if c_dtype is None:
        c_dtype = torch.bfloat16
    if d_dtype is None:
        d_dtype = torch.bfloat16
    if rht_dtype is None:
        rht_dtype = torch.bfloat16
    if a_tensor.dtype == torch.uint8:
        raise ValueError("a_tensor dtype torch.uint8 is not supported as packed FP4 for this fusion; use torch.float4_e2m1fn_x2")
    if b_tensor is not None and b_tensor.dtype == torch.uint8:
        raise ValueError("b_tensor dtype torch.uint8 is not supported as packed FP4 for this fusion; use torch.float4_e2m1fn_x2")
    if b_dtype == torch.uint8:
        raise ValueError("b_dtype torch.uint8 is not supported as packed FP4 for this fusion; use torch.float4_e2m1fn_x2")
    if d_dtype == torch.uint8:
        raise ValueError("d_dtype torch.uint8 is not supported as packed FP4 for this fusion; use torch.float4_e2m1fn_x2")
    if rht_dtype == torch.uint8:
        raise ValueError("rht_dtype torch.uint8 is not supported as packed FP4 for this fusion; use torch.float4_e2m1fn_x2")

    valid_m = a_tensor.shape[0]
    is_dense = b_tensor is not None
    is_discrete = b_ptrs is not None
    if is_dense and is_discrete:
        raise ValueError("Provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs), not both")
    if not is_dense and not is_discrete:
        raise ValueError("Must provide either (b_tensor, sfb_tensor) or (b_ptrs, sfb_ptrs)")

    _, k_physical, _ = a_tensor.shape
    if is_dense:
        weight_mode = MoEWeightMode.DENSE
        n_full, _, l = b_tensor.shape
        if sfb_tensor is None:
            raise ValueError("sfb_tensor is required in dense mode")
    else:
        weight_mode = MoEWeightMode.DISCRETE
        _require_pointer_tensor(b_ptrs, "b_ptrs")
        l = b_ptrs.shape[0]
        _require_pointer_tensor(sfb_ptrs, "sfb_ptrs", l)
        if n is None or b_dtype is None:
            raise ValueError("n and b_dtype are required for discrete mode")
        n_full = n
        k_logical = k_physical * 2 if b_dtype == torch.float4_e2m1fn_x2 else k_physical
        b_shape = (n_full, k_logical)
    n_out = n_full if act_func == "srelu" else n_full // 2

    if cd_major != "n":
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    d_quant = d_dtype == torch.float4_e2m1fn_x2
    rht_quant = rht_output and rht_dtype == torch.float4_e2m1fn_x2
    device = a_tensor.device

    def alloc_n_major(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty_strided((rows, cols, 1), (cols, 1, rows * cols), dtype=dtype, device=device)

    def alloc_swizzled_sf(rows: int, cols: int) -> torch.Tensor:
        shape = (1, ceil_div(rows, 128), ceil_div(ceil_div(cols, sf_vec_size), 4), 32, 4, 4)
        return torch.empty(shape, dtype=sfa_tensor.dtype, device=device).permute(3, 4, 1, 5, 2, 0)

    c_tensor = alloc_n_major(valid_m, n_full, c_dtype)
    if d_quant:
        d_tensor = alloc_n_major(valid_m, n_out // 2, d_dtype)
        sfd_tensor = alloc_swizzled_sf(valid_m, n_out)
    else:
        d_tensor = alloc_n_major(valid_m, n_out, d_dtype)
        sfd_tensor = None
    rht_tensor = None
    sfrht_tensor = None
    if rht_output:
        rht_tensor = alloc_n_major(valid_m, n_out // 2 if rht_quant else n_out, rht_dtype)
        if rht_quant:
            if rht_rowwise:
                sfrht_tensor = alloc_swizzled_sf(valid_m, n_out)
            else:
                sfrht_tensor = alloc_swizzled_sf(n_out, valid_m)

    if valid_m == 0:
        return TupleDict(c_tensor=c_tensor, d_tensor=d_tensor, sfd_tensor=sfd_tensor, rht_tensor=rht_tensor, sfrht_tensor=sfrht_tensor)

    def stride_order(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(i for i, _ in sorted(enumerate(tensor.stride()), key=lambda item: item[1]))

    def tensor_signature(tensor: Optional[torch.Tensor]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype

    def dynamic_m_tensor_signature(
        tensor: Optional[torch.Tensor], static_shape_suffix: Optional[Tuple[int, ...]], dynamic_stride_dims: Tuple[int, ...] = ()
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[torch.dtype]]:
        if tensor is None:
            return None, None, None
        stride_signature = tuple(None if idx in dynamic_stride_dims else value for idx, value in enumerate(tensor.stride()))
        return static_shape_suffix, stride_signature, tensor.dtype

    device_type = get_device_type()

    cache_key = (
        device_type,
        weight_mode,
        act_func,
        a_tensor.shape[1:],
        tuple(b_tensor.shape) if is_dense else b_shape,
        c_tensor.shape[1:],
        a_tensor.dtype,
        b_tensor.dtype if is_dense else b_dtype,
        c_tensor.dtype,
        d_tensor.dtype,
        rht_output,
        rht_dtype if rht_output else None,
        rht_rowwise if rht_output else None,
        glu_alpha,
        glu_limit,
        stride_order(a_tensor),
        stride_order(b_tensor) if is_dense else b_major,
        stride_order(c_tensor),
        *dynamic_m_tensor_signature(sfa_tensor, (sfa_tensor.shape[4], 1), dynamic_stride_dims=(5,)),
        *(tensor_signature(sfb_tensor) if is_dense else (tuple(sfb_ptrs.shape), tuple(sfb_ptrs.stride()), sfb_ptrs.dtype)),
        *tensor_signature(alpha_tensor),
        *dynamic_m_tensor_signature(prob_tensor, (1, 1)),
        *tensor_signature(bias_tensor),
        *tensor_signature(padded_offsets),
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        sf_fp8_dtype_override,
        vector_f32,
        m_aligned,
        use_dynamic_sched,
        *((tuple(b_ptrs.shape), tuple(b_ptrs.stride()), b_ptrs.dtype, l) if is_discrete else ()),
    )

    if cache_key in _cache_of_GroupedGemmGluHadamardQuantSm100Objects:
        api = _cache_of_GroupedGemmGluHadamardQuantSm100Objects[cache_key]
    else:
        common_kwargs = dict(
            sample_a=a_tensor,
            sample_c=c_tensor,
            sample_d=d_tensor,
            sample_sfa=sfa_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_prob=prob_tensor,
            sample_sfd=sfd_tensor,
            sample_rht=rht_tensor,
            sample_sfrht=sfrht_tensor,
            sample_bias=bias_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            sf_fp8_dtype_override=sf_fp8_dtype_override,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            act_func=act_func,
            use_dynamic_sched=use_dynamic_sched,
            rht_rowwise=rht_rowwise,
            glu_alpha=glu_alpha,
            glu_limit=glu_limit,
        )
        if is_dense:
            api = GroupedGemmGluHadamardQuantSm100(sample_b=b_tensor, sample_sfb=sfb_tensor, **common_kwargs)
        else:
            api = GroupedGemmGluHadamardQuantSm100(
                num_experts=l,
                b_shape=b_shape,
                b_dtype=b_dtype,
                b_major=b_major,
                **common_kwargs,
            )
        api.check_support()
        api.compile()
        _cache_of_GroupedGemmGluHadamardQuantSm100Objects[cache_key] = api

    if is_dense:
        api.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            sfd_tensor=sfd_tensor,
            rht_tensor=rht_tensor,
            sfrht_tensor=sfrht_tensor,
            bias_tensor=bias_tensor,
            norm_const=norm_const,
            rht_norm_const=rht_norm_const,
            current_stream=current_stream,
        )
    else:
        api.execute(
            a_tensor=a_tensor,
            b_ptrs=b_ptrs,
            sfb_ptrs=sfb_ptrs,
            c_tensor=c_tensor,
            d_tensor=d_tensor,
            sfa_tensor=sfa_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            prob_tensor=prob_tensor,
            sfd_tensor=sfd_tensor,
            rht_tensor=rht_tensor,
            sfrht_tensor=sfrht_tensor,
            bias_tensor=bias_tensor,
            norm_const=norm_const,
            rht_norm_const=rht_norm_const,
            current_stream=current_stream,
        )
    return TupleDict(c_tensor=c_tensor, d_tensor=d_tensor, sfd_tensor=sfd_tensor, rht_tensor=rht_tensor, sfrht_tensor=sfrht_tensor)
