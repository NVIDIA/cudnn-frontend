# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified FE API for grouped GEMM wgrad on SM100+."""

from typing import Optional, Tuple, Union

import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, ceil_div, is_power_of_2
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

from .moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
from ..moe_utils import MoEWeightMode, WGradInputOrder


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_wgrad_rubin import (
        BlockScaledMoEGroupedGemmWgradRubinKernel,
    )

    return BlockScaledMoEGroupedGemmWgradRubinKernel


def _is_supported_rubin_quantization(ab_dtype: torch.dtype, sf_dtype: torch.dtype, sf_vec_size: int) -> bool:
    is_fp4 = ab_dtype in (torch.float4_e2m1fn_x2, torch.uint8)
    if is_fp4:
        return (sf_dtype == torch.float8_e4m3fn and sf_vec_size == 16) or (
            sf_dtype == torch.float8_e8m0fnu and sf_vec_size == 32
        )
    return ab_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and sf_dtype == torch.float8_e8m0fnu and sf_vec_size == 32


def _round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


class GroupedGemmWgradBlockScaledAPI(APIBase):
    """Unified grouped GEMM wgrad FE API for SM100+ GPUs."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_offsets: torch.Tensor,
        sample_wgrad: Optional[torch.Tensor] = None,
        sample_wgrad_expert: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        wgrad_shape: Optional[Tuple[int, int]] = None,
        wgrad_dtype: Optional[torch.dtype] = None,
        sample_global_scale_a: Optional[torch.Tensor] = None,
        sample_global_scale_b: Optional[torch.Tensor] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        accumulate_on_output: bool = False,
        input_order: Union[WGradInputOrder, str] = WGradInputOrder.Tensor2D,
    ):
        super().__init__()
        self._warn_experimental_api()
        self.input_order = WGradInputOrder(input_order)

        if sample_wgrad is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
        elif sample_wgrad is None and num_experts is not None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if wgrad_shape is None or wgrad_dtype is None:
                raise ValueError("wgrad_shape and wgrad_dtype are required in discrete mode")
        else:
            raise ValueError("Provide either sample_wgrad for dense mode or " "(num_experts, wgrad_shape, wgrad_dtype) for discrete mode, but not both")

        self._interpret_uint8_as_fp4x2 = True
        self.sample_a_tensor = sample_a if self._is_fp4x2(sample_a) else None
        self.sample_b_tensor = sample_b if self._is_fp4x2(sample_b) else None
        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.sfa_desc = self._make_tensor_desc(sample_sfa, name="sample_sfa")
        self.sfb_desc = self._make_tensor_desc(sample_sfb, name="sample_sfb")
        self.offsets_desc = self._make_tensor_desc(sample_offsets, name="sample_offsets")
        self.global_scale_a_desc = self._make_tensor_desc(sample_global_scale_a, name="sample_global_scale_a")
        self.global_scale_b_desc = self._make_tensor_desc(sample_global_scale_b, name="sample_global_scale_b")
        self.sf_vec_size = sf_vec_size
        tokens_sum_a = self.a_desc.shape[1]
        tokens_sum_b = self.b_desc.shape[0]
        self._value_error_if(
            tokens_sum_a != tokens_sum_b,
            f"sample_a and sample_b token dimensions must match, got {tokens_sum_a} and {tokens_sum_b}",
        )
        self._offset_values = self._validate_offsets(sample_offsets, tokens_sum_a, name="sample_offsets")
        self._scale_cols = _round_up(ceil_div(tokens_sum_a, self.sf_vec_size), 4)

        if self.weight_mode == MoEWeightMode.DENSE:
            self.wgrad_desc = self._make_tensor_desc(sample_wgrad, name="sample_wgrad")
            self.expert_cnt = self.wgrad_desc.shape[0]
            self.wgrad_shape = self.wgrad_desc.shape[1:]
            self.wgrad_dtype = self.wgrad_desc.dtype
            self.single_expert_wgrad_desc = TensorDesc(
                dtype=self.wgrad_desc.dtype,
                shape=self.wgrad_desc.shape[1:],
                stride=self.wgrad_desc.stride[1:],
                stride_order=tuple(i for i, s in sorted(enumerate(self.wgrad_desc.stride[1:]), key=lambda x: x[1])),
                device=self.wgrad_desc.device,
                name="single_expert_wgrad",
            )
        else:  # MoEWeightMode.DISCRETE
            self.expert_cnt = num_experts
            self.wgrad_shape = tuple(wgrad_shape)
            self.wgrad_dtype = wgrad_dtype
            self.wgrad_desc = None
            if sample_wgrad_expert is not None:
                self.single_expert_wgrad_desc = self._make_tensor_desc(
                    sample_wgrad_expert,
                    name="sample_wgrad_expert",
                )
            else:
                self.single_expert_wgrad_desc = TensorDesc(
                    dtype=wgrad_dtype,
                    shape=self.wgrad_shape,
                    stride=(self.wgrad_shape[1], 1),
                    stride_order=(1, 0),
                    device=self.a_desc.device,
                    name="single_expert_wgrad",
                )

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        self.cluster_shape_mn = cluster_shape_mn or ((2, 1) if self.use_2cta_instrs else (1, 1))
        self.accumulate_on_output = accumulate_on_output
        self._kernel = _get_rubin_kernel() if self._is_rubin_kernel else BlockScaledMoEGroupedGemmWgradKernel
        self._workspace = None

    def _validate_offsets(self, offsets_tensor: torch.Tensor, tokens_sum: int, name: str) -> Tuple[int, ...]:
        self._value_error_if(offsets_tensor.ndim != 1, f"{name} must be rank-1, got shape {tuple(offsets_tensor.shape)}")

        offset_values = tuple(int(offset) for offset in offsets_tensor.detach().cpu().tolist())
        prev_offset = 0
        for idx, offset in enumerate(offset_values):
            self._value_error_if(
                offset < prev_offset,
                f"{name} must be a non-decreasing cumulative sum, but index {idx} has {offset} after {prev_offset}",
            )
            prev_offset = offset

        if offset_values:
            self._value_error_if(
                offset_values[-1] > tokens_sum,
                f"{name} last value must not exceed total tokens {tokens_sum}, got {offset_values[-1]}",
            )
        else:
            self._value_error_if(tokens_sum != 0, f"{name} cannot be empty when total tokens is {tokens_sum}")

        return offset_values

    def _check_rubin_quantization_support(self) -> None:
        if not self._is_rubin_kernel:
            return

        self._value_error_if(
            self.sfa_desc.dtype != self.sfb_desc.dtype,
            "Rubin wgrad requires sample_sfa and sample_sfb to have the same dtype",
        )
        self._value_error_if(
            not _is_supported_rubin_quantization(self.a_desc.dtype, self.sfa_desc.dtype, self.sf_vec_size),
            "Rubin wgrad supports NVFP4 (E2M1/E4M3, vec16), MXFP4 "
            "(E2M1/E8M0, vec32), and MXFP8 (E4M3 or E5M2/E8M0, vec32)",
        )
        self._value_error_if(self.acc_dtype != torch.float32, "Rubin wgrad requires float32 accumulation")

        if self._is_fp4x2(self.a_desc):
            self._value_error_if(
                self.a_desc.stride[1] != 1 or self.b_desc.stride[0] != 1,
                "Four-bit Rubin wgrad requires K-major sample_a and sample_b layouts",
            )

    def check_support(self) -> bool:
        m, tokens_sum = self._tensor_shape(self.a_desc, name="sample_a")
        _, n = self._tensor_shape(self.b_desc, name="sample_b")

        _ = self._check_tensor_shape(self.a_desc, (m, tokens_sum), "sample_a")
        _ = self._check_tensor_shape(self.b_desc, (tokens_sum, n), "sample_b")
        _ = self._check_tensor_shape(self.sfa_desc, (_round_up(m, 128), self._scale_cols), "sample_sfa")
        _ = self._check_tensor_shape(self.sfb_desc, (_round_up(n, 128), self._scale_cols), "sample_sfb")
        _ = self._check_tensor_shape(self.offsets_desc, (self.expert_cnt,), "sample_offsets")

        dtype = self._check_dtype(self.a_desc, [torch.float4_e2m1fn_x2, torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn], "sample_a")
        self._check_dtype(self.b_desc, dtype, "sample_b", extra_error_msg="sample_b must have the same dtype as sample_a")
        self._check_dtype(
            self.sfa_desc,
            [torch.float8_e8m0fnu, torch.float8_e4m3fn],
            "sample_sfa",
            extra_error_msg="sample_sfa must have dtype float8_e8m0fnu or float8_e4m3fn",
        )
        self._check_dtype(
            self.sfb_desc,
            [torch.float8_e8m0fnu, torch.float8_e4m3fn],
            "sample_sfb",
            extra_error_msg="sample_sfb must have dtype float8_e8m0fnu or float8_e4m3fn",
        )
        self._check_rubin_quantization_support()
        self._check_dtype(self.offsets_desc, torch.int32, "sample_offsets", extra_error_msg="sample_offsets must be int32")
        self._check_dtype(
            self.wgrad_dtype, [torch.bfloat16, torch.float16, torch.float32], "wgrad_dtype", extra_error_msg="wgrad_dtype must be bfloat16, float16, or float32"
        )

        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_tensor_shape(self.wgrad_desc, (self.expert_cnt, m, n), "sample_wgrad")
        else:
            self._check_tensor_shape(self.wgrad_shape, (m, n), "wgrad_shape")
            self._check_tensor_shape(self.single_expert_wgrad_desc, (m, n), "single_expert_wgrad")
            self._check_dtype(
                self.single_expert_wgrad_desc,
                self.wgrad_dtype,
                "sample_wgrad_expert",
                extra_error_msg="sample_wgrad_expert must have the same dtype as wgrad_dtype",
            )

        self._value_error_if(self.mma_tiler_mn[0] not in (128, 256), f"mma_tiler_mn[0] must be 128 or 256, got {self.mma_tiler_mn[0]}")
        self._value_error_if(self.mma_tiler_mn[1] not in (128, 256), f"mma_tiler_mn[1] must be 128 or 256, got {self.mma_tiler_mn[1]}")
        self._value_error_if(
            self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0,
            f"cluster_shape_mn[0] must be divisible by 2 when use_2cta_instrs=True, got {self.cluster_shape_mn[0]}",
        )
        self._value_error_if(self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16, f"cluster shape product must be <= 16, got {self.cluster_shape_mn}")
        self._value_error_if(
            not (is_power_of_2(self.cluster_shape_mn[0]) and is_power_of_2(self.cluster_shape_mn[1])),
            f"cluster shape values must be powers of 2, got {self.cluster_shape_mn}",
        )

        has_global_scale = self.global_scale_a_desc is not None or self.global_scale_b_desc is not None
        if has_global_scale:
            self._value_error_if(
                self.global_scale_a_desc is None or self.global_scale_b_desc is None,
                "sample_global_scale_a and sample_global_scale_b must be provided together",
            )
            self._value_error_if(
                self.global_scale_a_desc.shape != (self.expert_cnt,),
                f"sample_global_scale_a must have shape {(self.expert_cnt,)}, got {self.global_scale_a_desc.shape}",
            )
            self._value_error_if(
                self.global_scale_b_desc.shape != (self.expert_cnt,),
                f"sample_global_scale_b must have shape {(self.expert_cnt,)}, got {self.global_scale_b_desc.shape}",
            )
            self._check_dtype(self.global_scale_a_desc, torch.float32, "sample_global_scale_a")
            self._check_dtype(self.global_scale_b_desc, torch.float32, "sample_global_scale_b")

        requires_global_scale = (
            self._is_fp4x2(self.a_desc) and self.sf_vec_size == 16 and self.sfa_desc.dtype == torch.float8_e4m3fn and self.sfb_desc.dtype == torch.float8_e4m3fn
        )
        self._value_error_if(requires_global_scale and not has_global_scale, "NVFP4 wgrad requires sample_global_scale_a and sample_global_scale_b")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmWgrad requires SM100+ compute capability, but found SM{compute_capability} on device {device}")

        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            accumulate_on_output=self.accumulate_on_output,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            input_order=self.input_order,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        self._workspace = torch.empty(max(kernel.get_workspace_bytes(), 1), dtype=torch.uint8, device=self.a_desc.device)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(kernel, max_active_clusters, fake_stream)
        else:
            self._compile_discrete(kernel, max_active_clusters, fake_stream)

        if self.sample_a_tensor is not None:
            del self.sample_a_tensor
        if self.sample_b_tensor is not None:
            del self.sample_b_tensor

    def _compile_dense(self, kernel, max_active_clusters, fake_stream) -> None:
        a_fake = (
            from_dlpack(self.sample_a_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=1,
                stride_order=self.sample_a_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_a_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=self.a_desc.shape,
                stride_order=self.a_desc.stride_order,
                assumed_align=16,
                dynamic_mode=1,
                divisibility=16,
            )
        )
        b_fake = (
            from_dlpack(self.sample_b_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=0,
                stride_order=self.sample_b_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_b_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=self.b_desc.shape,
                stride_order=self.b_desc.stride_order,
                assumed_align=16,
                dynamic_mode=0,
                divisibility=16,
            )
        )
        sfa_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfa_desc.dtype,
            shape=self.sfa_desc.shape,
            stride_order=self.sfa_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        sfb_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfb_desc.dtype,
            shape=self.sfb_desc.shape,
            stride_order=self.sfb_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        wgrad_fake = self._make_fake_cute_tensor_from_desc(self.wgrad_desc, assumed_align=16)
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.offsets_desc, assumed_align=4)
        workspace_fake = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        gs_a_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_a_desc, assumed_align=4)
        gs_b_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_b_desc, assumed_align=4)

        compiled = cute.compile(
            kernel,
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            wgrad_fake,
            offsets_fake,
            workspace_fake,
            max_active_clusters,
            fake_stream,
            gs_a_fake,
            gs_b_fake,
            None,
            options="--enable-tvm-ffi",
        )

        cached_workspace = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            wgrad_tensor: torch.Tensor,
            offsets_tensor: torch.Tensor,
            stream: cuda.CUstream,
            global_scale_a: Optional[torch.Tensor],
            global_scale_b: Optional[torch.Tensor],
        ) -> None:
            compiled(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                wgrad_tensor,
                offsets_tensor,
                cached_workspace,
                stream,
                global_scale_a,
                global_scale_b,
                None,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, kernel, max_active_clusters, fake_stream) -> None:
        a_fake = (
            from_dlpack(self.sample_a_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=1,
                stride_order=self.sample_a_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_a_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.a_desc.dtype,
                shape=self.a_desc.shape,
                stride_order=self.a_desc.stride_order,
                assumed_align=16,
                dynamic_mode=1,
                divisibility=16,
            )
        )
        b_fake = (
            from_dlpack(self.sample_b_tensor, assumed_align=16, enable_tvm_ffi=True).mark_compact_shape_dynamic(
                mode=0,
                stride_order=self.sample_b_tensor.dim_order(),
                divisibility=16,
            )
            if self.sample_b_tensor is not None
            else self._make_fake_cute_compact_tensor(
                dtype=self.b_desc.dtype,
                shape=self.b_desc.shape,
                stride_order=self.b_desc.stride_order,
                assumed_align=16,
                dynamic_mode=0,
                divisibility=16,
            )
        )
        sfa_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfa_desc.dtype,
            shape=self.sfa_desc.shape,
            stride_order=self.sfa_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        sfb_fake = self._make_fake_cute_compact_tensor(
            dtype=self.sfb_desc.dtype,
            shape=self.sfb_desc.shape,
            stride_order=self.sfb_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=4,
        )
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.offsets_desc, assumed_align=4)
        workspace_fake = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        gs_a_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_a_desc, assumed_align=4)
        gs_b_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_b_desc, assumed_align=4)
        wgrad_ptrs_placeholder = torch.empty((self.expert_cnt,), dtype=torch.int64, device=self.a_desc.device)
        wgrad_ptrs_fake = from_dlpack(wgrad_ptrs_placeholder, assumed_align=8, enable_tvm_ffi=True).iterator
        single_expert_fake = self._make_fake_cute_tensor(
            dtype=self.single_expert_wgrad_desc.dtype,
            shape=self.single_expert_wgrad_desc.shape,
            stride=self.single_expert_wgrad_desc.stride,
            assumed_align=16,
        )

        compiled = cute.compile(
            kernel,
            a_fake,
            b_fake,
            sfa_fake,
            sfb_fake,
            wgrad_ptrs_fake,
            offsets_fake,
            workspace_fake,
            max_active_clusters,
            fake_stream,
            gs_a_fake,
            gs_b_fake,
            single_expert_fake,
            options="--enable-tvm-ffi",
        )

        cached_workspace = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        single_expert_placeholder = torch.empty_strided(
            self.single_expert_wgrad_desc.shape,
            self.single_expert_wgrad_desc.stride,
            dtype=self.single_expert_wgrad_desc.dtype,
            device=self.single_expert_wgrad_desc.device,
        )
        cached_single_expert = from_dlpack(
            single_expert_placeholder,
            assumed_align=16,
            enable_tvm_ffi=True,
        )

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            wgrad_ptrs: torch.Tensor,
            offsets_tensor: torch.Tensor,
            stream: cuda.CUstream,
            global_scale_a: Optional[torch.Tensor],
            global_scale_b: Optional[torch.Tensor],
        ) -> None:
            compiled(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                wgrad_ptrs.data_ptr(),
                offsets_tensor,
                cached_workspace,
                stream,
                global_scale_a,
                global_scale_b,
                cached_single_expert,
            )

        self._compiled_kernel = tensor_api

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        global_scale_a: Optional[torch.Tensor] = None,
        global_scale_b: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        current_stream = self._get_default_stream(current_stream)
        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")

        if self.weight_mode == MoEWeightMode.DENSE:
            self._value_error_if(wgrad_tensor is None, "wgrad_tensor is required in dense mode")
            self._compiled_kernel(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                wgrad_tensor,
                offsets_tensor,
                current_stream,
                global_scale_a,
                global_scale_b,
            )
            return

        if wgrad_ptrs is None:
            self._value_error_if(wgrad_tensor is None, "Provide wgrad_tensor or wgrad_ptrs in discrete mode")
            self._value_error_if(wgrad_tensor.ndim != 3, f"wgrad_tensor must be rank-3, got {tuple(wgrad_tensor.shape)}")
            self._value_error_if(not wgrad_tensor.is_cuda, f"wgrad_tensor must be a CUDA tensor, got {wgrad_tensor.device}")
            if wgrad_tensor.shape[0] == 0:
                wgrad_ptrs = torch.empty((0,), dtype=torch.int64, device=wgrad_tensor.device)
            else:
                expert_stride_bytes = wgrad_tensor.stride(0) * wgrad_tensor.element_size()
                ptrs = [wgrad_tensor.data_ptr() + i * expert_stride_bytes for i in range(wgrad_tensor.shape[0])]
                wgrad_ptrs = torch.tensor(ptrs, dtype=torch.int64, device=wgrad_tensor.device)
        _require_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", self.expert_cnt)
        self._compiled_kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            wgrad_ptrs,
            offsets_tensor,
            current_stream,
            global_scale_a,
            global_scale_b,
        )


__all__ = ["GroupedGemmWgradBlockScaledAPI"]
