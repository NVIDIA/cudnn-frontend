# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified FE API for grouped GEMM wgrad on SM100+."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import torch

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_ptr

from cudnn.api_base import APIBase, TensorDesc, ceil_div, is_power_of_2
from cudnn._torch_stream import as_torch_stream
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.frost.workspace import Workspace, align_up
from cudnn.gemm.cutedsl.grouped.unfused._bf16_api import _validate_pointer_tensor
from cudnn.tensor_adapter import get_device, is_torch_tensor

from ._bf16_api import WGRAD_PTRS_REQUIRED
from .moe_blockscaled_grouped_gemm_wgrad import BlockScaledMoEGroupedGemmWgradKernel
from ..backend_utils import debug_validate_pointer_values, retain_workspace
from ..moe_utils import MoEWeightMode, WGradInputOrder


def _get_rubin_kernel():
    from .moe_blockscaled_grouped_gemm_wgrad_rubin import (
        BlockScaledMoEGroupedGemmWgradRubinKernel,
    )

    return BlockScaledMoEGroupedGemmWgradRubinKernel


def _is_supported_rubin_quantization(ab_dtype: torch.dtype, sf_dtype: torch.dtype, sf_vec_size: int) -> bool:
    import torch

    is_fp4 = ab_dtype in (torch.float4_e2m1fn_x2, torch.uint8)
    if is_fp4:
        return (sf_dtype == torch.float8_e4m3fn and sf_vec_size == 16) or (sf_dtype == torch.float8_e8m0fnu and sf_vec_size == 32)
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
        acc_dtype: Optional[torch.dtype] = None,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        sf_fp8_dtype_override: Optional[Literal["e5m3"]] = None,
        accumulate_on_output: bool = False,
        input_order: Union[WGradInputOrder, str] = WGradInputOrder.Tensor2D,
    ):
        if sample_a is not None and not is_torch_tensor(sample_a):
            raise ValueError(
                "The block-scaled wgrad backend supports torch tensors only: its B operand "
                "(and fp4-packed A/B operands) require K-major (token-innermost) layouts that "
                "are not expressible as row-major JAX arrays"
            )
        import torch

        from cudnn.tensor_adapter import framework_dtype

        # This backend is torch-internal: normalize loose dtype parameters (which the
        # type-erased wrapper/facade may pass as cutlass or numpy dtypes) to torch dtypes.
        acc_dtype = framework_dtype(acc_dtype, "torch") if acc_dtype is not None else torch.float32
        wgrad_dtype = framework_dtype(wgrad_dtype, "torch") if wgrad_dtype is not None else None
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
        self.sf_fp8_dtype_override = sf_fp8_dtype_override
        tokens_sum_a = self.a_desc.shape[1]
        tokens_sum_b = self.b_desc.shape[0]
        self._value_error_if(
            tokens_sum_a != tokens_sum_b,
            f"sample_a and sample_b token dimensions must match, got {tokens_sum_a} and {tokens_sum_b}",
        )
        self._check_offsets_rank(self.offsets_desc, name="sample_offsets")
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
        self._kernel_obj = None

    def _check_offsets_rank(self, offsets_desc: TensorDesc, name: str) -> None:
        # Metadata only: the offset VALUES are a device-data contract read in-kernel
        # (non-decreasing cumulative ends, last <= tokens_sum); no host copy at build.
        self._value_error_if(offsets_desc.ndim != 1, f"{name} must be rank-1, got shape {offsets_desc.shape}")

    def _check_rubin_quantization_support(self) -> None:
        import torch

        if not self._is_rubin_kernel:
            return

        self._value_error_if(
            self.sfa_desc.dtype != self.sfb_desc.dtype,
            "Rubin wgrad requires sample_sfa and sample_sfb to have the same dtype",
        )
        self._value_error_if(
            not _is_supported_rubin_quantization(self.a_desc.dtype, self.sfa_desc.dtype, self.sf_vec_size),
            "Rubin wgrad supports NVFP4 (E2M1/E4M3, vec16), MXFP4 " "(E2M1/E8M0, vec32), and MXFP8 (E4M3 or E5M2/E8M0, vec32)",
        )
        self._value_error_if(self.acc_dtype != torch.float32, "Rubin wgrad requires float32 accumulation")

        if self._is_fp4x2(self.a_desc):
            self._value_error_if(
                self.a_desc.stride[1] != 1 or self.b_desc.stride[0] != 1,
                "Four-bit Rubin wgrad requires K-major sample_a and sample_b layouts",
            )

    def check_support(self) -> bool:
        import torch

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
                self.sfa_desc.dtype != torch.float8_e4m3fn,
                f"sf_fp8_dtype_override='e5m3' requires the NVFP4 recipe -- FP4 A/B with "
                f"torch.float8_e4m3fn scale factors at sf_vec_size 16 -- but got "
                f"ab_dtype={self.a_desc.dtype}, sf_dtype={self.sfa_desc.dtype}, sf_vec_size={self.sf_vec_size}",
            )
            # Only allow e5m3 for rubin kernels
            self._value_error_if(
                not self._is_rubin_kernel,
                f"sf_fp8_dtype_override='e5m3' requires Rubin (SM107), got device type {self._device_type!r}",
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

    def _kernel_instance(self):
        if self._kernel_obj is None:
            self._kernel_obj = self._kernel(
                sf_vec_size=self.sf_vec_size,
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                accumulate_on_output=self.accumulate_on_output,
                expert_cnt=self.expert_cnt,
                weight_mode=self.weight_mode,
                input_order=self.input_order,
                sf_fp8_dtype_override=self.sf_fp8_dtype_override,
            )
        return self._kernel_obj

    def scratch_workspace_bytes(self) -> int:
        """Caller-provided scratch ``execute(workspace=)`` carves (recipe R2): the per-expert
        TMA-descriptor slots, 128-byte aligned, never 0."""
        self._ensure_support_checked()
        return max(align_up(self._kernel_instance().get_workspace_bytes(), 128), 128)

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return

        kernel = self._kernel_instance()

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
        # The kernel's workspace is a cute.Tensor: its fake matches the uint8 view carved at
        # execute (R11). interpret_uint8_as_fp4x2 is this API's default and must not apply.
        workspace_fake = self._make_fake_cute_tensor(cutlass.Uint8, (self.scratch_workspace_bytes(),), (1,), assumed_align=128, interpret_uint8_as_fp4x2=False)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._compile_dense(kernel, max_active_clusters, fake_stream, workspace_fake)
        else:
            self._compile_discrete(kernel, max_active_clusters, fake_stream, workspace_fake)

        if self.sample_a_tensor is not None:
            del self.sample_a_tensor
        if self.sample_b_tensor is not None:
            del self.sample_b_tensor

    def _compile_dense(self, kernel, max_active_clusters, fake_stream, workspace_fake) -> None:
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

        def tensor_api(
            a_tensor: torch.Tensor,
            b_tensor: torch.Tensor,
            sfa_tensor: torch.Tensor,
            sfb_tensor: torch.Tensor,
            wgrad_tensor: torch.Tensor,
            offsets_tensor: torch.Tensor,
            workspace,
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
                workspace,
                stream,
                global_scale_a,
                global_scale_b,
                None,
            )

        self._compiled_kernel = tensor_api

    def _compile_discrete(self, kernel, max_active_clusters, fake_stream, workspace_fake) -> None:
        import torch

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
        gs_a_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_a_desc, assumed_align=4)
        gs_b_fake = self._make_fake_cute_tensor_from_desc(self.global_scale_b_desc, assumed_align=4)
        wgrad_ptrs_fake = make_ptr(cutlass.Int64, 16, cute.AddressSpace.gmem, assumed_align=8)
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
            workspace,
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
                workspace,
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
        *,
        workspace=None,
        descriptor_workspace: Optional[torch.Tensor] = None,
    ) -> None:
        """``workspace``: caller-owned, 128-byte-aligned, contiguous device buffer of at least
        ``scratch_workspace_bytes()`` bytes; never allocated here. ``descriptor_workspace`` is
        the same buffer under its public name (a uint8 torch tensor sized with
        ``get_grouped_gemm_wgrad_workspace_size_sm100``); passing both requires one object.
        Discrete mode requires ``wgrad_ptrs`` (see
        :func:`cudnn.gemm.cutedsl.grouped.wgrad.api.wgrad_expert_ptrs`)."""
        current_stream = self._get_default_stream(current_stream)
        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")
        nbytes = self.scratch_workspace_bytes()
        workspace = self._resolve_workspace_alias(workspace, descriptor_workspace, nbytes, a_tensor)

        if self.weight_mode == MoEWeightMode.DENSE:
            self._value_error_if(wgrad_tensor is None, "wgrad_tensor is required in dense mode")
            ws_view = Workspace(workspace, nbytes, type(self).__name__).take(nbytes, "uint8")
            retain_workspace(self, workspace, current_stream)
            self._compiled_kernel(
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                wgrad_tensor,
                offsets_tensor,
                ws_view,
                current_stream,
                global_scale_a,
                global_scale_b,
            )
            return

        self._value_error_if(wgrad_ptrs is None, WGRAD_PTRS_REQUIRED)
        _validate_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", self.expert_cnt)
        if get_device(wgrad_ptrs) != self.a_desc.device:
            raise ValueError(f"wgrad_ptrs must be on {self.a_desc.device}, got {get_device(wgrad_ptrs)}")
        debug_validate_pointer_values(wgrad_ptrs, "wgrad_ptrs", stream=current_stream)
        if is_torch_tensor(wgrad_ptrs):
            wgrad_ptrs.record_stream(as_torch_stream(int(current_stream), wgrad_ptrs.device))
        ws_view = Workspace(workspace, nbytes, type(self).__name__).take(nbytes, "uint8")
        retain_workspace(self, workspace, current_stream)
        self._compiled_kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            wgrad_ptrs,
            offsets_tensor,
            ws_view,
            current_stream,
            global_scale_a,
            global_scale_b,
        )

    def _resolve_workspace_alias(self, workspace, descriptor_workspace, nbytes: int, a_tensor):
        """``descriptor_workspace`` is an alias of ``workspace``: one caller-owned buffer, never
        a plan-owned fallback. The alias keeps its torch-only validation messages."""
        if descriptor_workspace is None:
            return workspace
        if workspace is not None and workspace is not descriptor_workspace:
            raise ValueError("workspace and descriptor_workspace name the same buffer; pass one of them, or the same object to both")
        self._value_error_if(
            not is_torch_tensor(descriptor_workspace),
            f"descriptor_workspace must be a torch.uint8 tensor, got {type(descriptor_workspace).__name__}",
        )
        import torch

        self._value_error_if(
            descriptor_workspace.dtype != torch.uint8,
            f"descriptor_workspace must have dtype uint8, got {descriptor_workspace.dtype}",
        )
        self._value_error_if(
            descriptor_workspace.device != get_device(a_tensor),
            "descriptor_workspace and WGrad operands must be on the same device",
        )
        self._value_error_if(
            not descriptor_workspace.is_contiguous(),
            "descriptor_workspace must be contiguous",
        )
        self._value_error_if(
            descriptor_workspace.numel() < nbytes,
            f"descriptor_workspace requires at least {nbytes} bytes, got {descriptor_workspace.numel()}",
        )
        return descriptor_workspace


__all__ = ["GroupedGemmWgradBlockScaledAPI"]
