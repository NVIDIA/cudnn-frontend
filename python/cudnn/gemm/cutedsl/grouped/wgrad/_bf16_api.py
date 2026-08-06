# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private descriptor-first BF16 API for SM100 grouped GEMM wgrad."""

import os
import weakref
from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import from_dlpack, make_fake_stream
import torch

from cudnn.api_base import APIBase, TensorDesc
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

from ..backend_utils import _torch_stream_context
from ..moe_utils import MoEWeightMode, WGradInputOrder
from .moe_grouped_gemm_wgrad import MoEGroupedGemmWgradBF16Kernel

_OUTPUT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


class GroupedGemmWgradBf16API(APIBase):
    """Descriptor-first lifecycle API for the source BF16 wgrad kernel."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: Optional[torch.Tensor],
        sample_sfb: Optional[torch.Tensor],
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
    ) -> None:
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

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.offsets_desc = self._make_tensor_desc(sample_offsets, name="sample_offsets")
        self.wgrad_desc = self._make_tensor_desc(sample_wgrad, name="sample_wgrad")
        self.single_expert_wgrad_desc = self._make_tensor_desc(sample_wgrad_expert, name="sample_wgrad_expert")
        self.expert_cnt = self.wgrad_desc.shape[0] if self.weight_mode == MoEWeightMode.DENSE and self.wgrad_desc.ndim == 3 else int(num_experts or 0)
        self.wgrad_shape = self.wgrad_desc.shape[1:] if self.weight_mode == MoEWeightMode.DENSE and self.wgrad_desc.ndim == 3 else tuple(wgrad_shape or ())
        self.wgrad_dtype = self.wgrad_desc.dtype if self.weight_mode == MoEWeightMode.DENSE else wgrad_dtype
        if self.weight_mode == MoEWeightMode.DISCRETE and self.single_expert_wgrad_desc is None:
            self.single_expert_wgrad_desc = TensorDesc(
                dtype=self.wgrad_dtype,
                shape=self.wgrad_shape,
                stride=(self.wgrad_shape[1], 1),
                stride_order=(1, 0),
                device=self.a_desc.device,
                name="single_expert_wgrad",
            )

        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.use_2cta_instrs = self.mma_tiler_mn[0] == 256
        self.cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if self.use_2cta_instrs else (1, 1)))
        self.accumulate_on_output = accumulate_on_output
        self.sf_vec_size = sf_vec_size
        self._scale_controls = (
            sample_sfa,
            sample_sfb,
            sample_global_scale_a,
            sample_global_scale_b,
        )
        self._kernel = MoEGroupedGemmWgradBF16Kernel
        self._workspace: Optional[torch.Tensor] = None
        self._compile_wgrad_ptrs: Optional[torch.Tensor] = None
        self._single_expert_placeholder: Optional[torch.Tensor] = None
        self._validated_offsets: dict[int, tuple] = {}
        self._validated_pointer_values: dict[int, tuple] = {}
        self._sample_offset_values = self._copy_values_to_host(sample_offsets)
        self._sample_offsets_ref = weakref.ref(sample_offsets)
        self._sample_offsets_version = int(sample_offsets._version)
        self._sample_data_ptrs = {
            name: tensor.data_ptr()
            for name, tensor in (
                ("sample_a", sample_a),
                ("sample_b", sample_b),
                ("sample_offsets", sample_offsets),
                ("sample_wgrad", sample_wgrad),
                ("sample_wgrad_expert", sample_wgrad_expert),
            )
            if tensor is not None
        }
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))
        self.a_major: Optional[str] = None
        self.b_major: Optional[str] = None

    @staticmethod
    def _copy_values_to_host(tensor: torch.Tensor) -> Tuple[int, ...]:
        return tuple(int(value) for value in tensor.detach().cpu().tolist())

    @staticmethod
    def _is_validation_cached(cache: dict[int, tuple], tensor: torch.Tensor, extra) -> bool:
        cached = cache.get(id(tensor))
        return bool(cached and cached[0]() is tensor and cached[1] == int(tensor._version) and cached[2] == extra)

    @staticmethod
    def _remember_validation(cache: dict[int, tuple], tensor: torch.Tensor, extra) -> None:
        key = id(tensor)

        def discard(_reference, *, cache=cache, key=key):
            cache.pop(key, None)

        cache[key] = (weakref.ref(tensor, discard), int(tensor._version), extra)

    @staticmethod
    def _validate_offset_sequence(values: Tuple[int, ...], *, expert_cnt: int, tokens_sum: int) -> Tuple[int, ...]:
        if len(values) != expert_cnt:
            raise ValueError(f"sample_offsets length mismatch: expected {expert_cnt}, got {len(values)}")
        groups = []
        previous = 0
        for index, value in enumerate(values):
            if value < previous:
                raise ValueError("sample_offsets must be a non-decreasing cumulative sum; " f"index {index} is {value} after {previous}")
            group_k = value - previous
            if group_k % MoEGroupedGemmWgradBF16Kernel.FIX_PAD_SIZE != 0:
                raise ValueError(f"sample_offsets group {index} must be 256-aligned, got {group_k}")
            groups.append(group_k)
            previous = value
        if not values or values[-1] != tokens_sum:
            raise ValueError(f"sample_offsets last value must equal total tokens {tokens_sum}, " f"got {values[-1] if values else None}")
        return tuple(groups)

    def _validate_offsets_once(self, offsets: torch.Tensor, *, tokens_sum: int) -> None:
        extra = (self.expert_cnt, tokens_sum)
        if self._is_validation_cached(self._validated_offsets, offsets, extra):
            return
        values = self._copy_values_to_host(offsets)
        self._validate_offset_sequence(values, expert_cnt=self.expert_cnt, tokens_sum=tokens_sum)
        self._remember_validation(self._validated_offsets, offsets, extra)

    def _validate_pointer_values_once(self, pointers: torch.Tensor) -> None:
        if self._is_validation_cached(self._validated_pointer_values, pointers, self.expert_cnt):
            return
        values = self._copy_values_to_host(pointers)
        if any(value == 0 or value % 16 != 0 for value in values):
            raise ValueError("wgrad_ptrs entries must be non-null and 16-byte aligned")
        self._remember_validation(self._validated_pointer_values, pointers, self.expert_cnt)

    @staticmethod
    def _validate_pointer_array_alignment(tensor: torch.Tensor) -> None:
        if tensor.data_ptr() % 8 != 0:
            raise ValueError("wgrad_ptrs data pointer must be 8-byte aligned")

    @staticmethod
    def _validate_data_alignment(tensor: torch.Tensor, name: str, alignment: int = 16) -> None:
        if tensor.data_ptr() % alignment != 0:
            raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")

    @staticmethod
    def _record_pointer_stream(pointers: torch.Tensor, current_stream: cuda.CUstream) -> None:
        handle = int(current_stream)
        torch_current = torch.cuda.current_stream(pointers.device)
        torch_default = torch.cuda.default_stream(pointers.device)
        if handle == torch_current.cuda_stream:
            launch_stream = torch_current
        elif handle == torch_default.cuda_stream:
            launch_stream = torch_default
        else:
            launch_stream = torch.cuda.ExternalStream(handle, device=pointers.device)
        pointers.record_stream(launch_stream)

    @staticmethod
    def _infer_a_major(desc: TensorDesc) -> str:
        m, tokens = desc.shape
        if desc.stride == (tokens, 1):
            return "k"
        if desc.stride == (1, m):
            return "m"
        raise ValueError(f"A tensor must use a supported K-major or M-major layout, got stride {desc.stride}")

    @staticmethod
    def _infer_b_major(desc: TensorDesc) -> str:
        tokens, n = desc.shape
        if desc.stride == (1, tokens):
            return "k"
        if desc.stride == (n, 1):
            return "n"
        raise ValueError(f"B tensor must use a supported K-major or N-major layout, got stride {desc.stride}")

    @staticmethod
    def _expect_device(desc: TensorDesc, device: torch.device, name: str) -> None:
        if desc.device != device:
            raise ValueError(f"{name} must be on {device}, got {desc.device}")

    def check_support(self) -> bool:
        if self.a_desc.ndim != 2:
            raise ValueError(f"sample_a must be rank-2, got {self.a_desc.shape}")
        if self.b_desc.ndim != 2:
            raise ValueError(f"sample_b must be rank-2, got {self.b_desc.shape}")
        m, tokens_sum = self.a_desc.shape
        tokens_b, n = self.b_desc.shape
        if tokens_b != tokens_sum:
            raise ValueError(f"sample_a and sample_b token dimensions must match, got {tokens_sum} and {tokens_b}")
        self.a_major = self._infer_a_major(self.a_desc)
        self.b_major = self._infer_b_major(self.b_desc)
        self._check_dtype(self.a_desc, torch.bfloat16, "sample_a")
        self._check_dtype(self.b_desc, torch.bfloat16, "sample_b")
        self._check_dtype(self.offsets_desc, torch.int32, "sample_offsets")
        self._check_dtype(self.wgrad_dtype, _OUTPUT_DTYPES, "wgrad_dtype")
        if self.acc_dtype != torch.float32:
            raise ValueError(f"acc_dtype must be torch.float32, got {self.acc_dtype}")
        if any(control is not None for control in self._scale_controls):
            raise ValueError("BF16 wgrad forbids scale and global-scale tensors")
        if self.sf_vec_size != 16:
            raise ValueError(f"BF16 wgrad requires sf_vec_size=16, got {self.sf_vec_size}")
        if self.offsets_desc.shape != (self.expert_cnt,):
            raise ValueError(f"sample_offsets must have shape {(self.expert_cnt,)}, got {self.offsets_desc.shape}")
        if self.offsets_desc.stride != (1,):
            raise ValueError("sample_offsets must be contiguous")

        if self.weight_mode == MoEWeightMode.DENSE:
            if self.wgrad_desc.ndim != 3:
                raise ValueError(f"sample_wgrad must be rank-3, got {self.wgrad_desc.shape}")
            if self.wgrad_desc.shape != (self.expert_cnt, m, n):
                raise ValueError(f"sample_wgrad shape mismatch: expected {(self.expert_cnt, m, n)}, got {self.wgrad_desc.shape}")
            if self.wgrad_desc.stride != (m * n, n, 1):
                raise ValueError("sample_wgrad must be contiguous in expert/M/N order")
            output_desc = self.wgrad_desc
        else:
            if self.wgrad_shape != (m, n):
                raise ValueError(f"wgrad_shape mismatch: expected {(m, n)}, got {self.wgrad_shape}")
            output_desc = self.single_expert_wgrad_desc
            if output_desc.shape not in ((m, n), (m, n, 1)):
                raise ValueError(f"sample_wgrad_expert shape mismatch: expected {(m, n)}, got {output_desc.shape}")
            expected_stride = (n, 1) if output_desc.ndim == 2 else (n, 1, 1)
            if output_desc.stride != expected_stride:
                raise ValueError("sample_wgrad_expert must be contiguous in M/N order")
            self._check_dtype(output_desc, self.wgrad_dtype, "sample_wgrad_expert")

        device = self.a_desc.device
        for desc, name in (
            (self.b_desc, "sample_b"),
            (self.offsets_desc, "sample_offsets"),
            (output_desc, "sample_wgrad"),
        ):
            self._expect_device(desc, device, name)
        for name, pointer in self._sample_data_ptrs.items():
            alignment = 4 if name == "sample_offsets" else 16
            if pointer % alignment:
                raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")

        groups = self._validate_offset_sequence(
            self._sample_offset_values,
            expert_cnt=self.expert_cnt,
            tokens_sum=tokens_sum,
        )
        sample_offsets = self._sample_offsets_ref()
        if sample_offsets is not None and int(sample_offsets._version) == self._sample_offsets_version:
            self._remember_validation(
                self._validated_offsets,
                sample_offsets,
                (self.expert_cnt, tokens_sum),
            )
        elif sample_offsets is not None:
            self._validate_offsets_once(sample_offsets, tokens_sum=tokens_sum)

        if not self._kernel.can_implement(
            _convert_to_cutlass_data_type(torch.bfloat16),
            _convert_to_cutlass_data_type(self.wgrad_dtype),
            _convert_to_cutlass_data_type(self.acc_dtype),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            m,
            n,
            list(groups),
            self.expert_cnt,
            self.a_major,
            self.b_major,
            self.weight_mode,
            self.input_order,
        ):
            raise ValueError("Unsupported BF16 grouped GEMM wgrad configuration: check mma_tiler, cluster, and alignment")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.a_desc.device)
        capability = major * 10 + minor
        if capability < 100:
            raise RuntimeError(f"GroupedGemmWgradSm100 requires SM100+, found SM{capability} on {self.a_desc.device}")
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        kernel = self._kernel(
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
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1]) - self.num_cluster_overlap_margin
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying CUDNNFE_CLUSTER_OVERLAP_MARGIN")
        self._workspace = torch.empty(
            max(kernel.get_workspace_bytes(), 1),
            dtype=torch.uint8,
            device=self.a_desc.device,
        )
        self._validate_data_alignment(self._workspace, "workspace", 128)
        workspace_fake = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        a_fake = self._make_fake_cute_compact_tensor(
            dtype=self.a_desc.dtype,
            shape=self.a_desc.shape,
            stride_order=self.a_desc.stride_order,
            assumed_align=16,
            dynamic_mode=1,
            divisibility=256,
        )
        b_fake = self._make_fake_cute_compact_tensor(
            dtype=self.b_desc.dtype,
            shape=self.b_desc.shape,
            stride_order=self.b_desc.stride_order,
            assumed_align=16,
            dynamic_mode=0,
            divisibility=256,
        )
        offsets_fake = self._make_fake_cute_tensor_from_desc(self.offsets_desc, assumed_align=4)
        if self.weight_mode == MoEWeightMode.DENSE:
            out_fake = self._make_fake_cute_tensor_from_desc(self.wgrad_desc, assumed_align=16)
            single_expert_fake = None
        else:
            self._compile_wgrad_ptrs = torch.empty((self.expert_cnt,), dtype=torch.int64, device=self.a_desc.device)
            self._validate_pointer_array_alignment(self._compile_wgrad_ptrs)
            out_fake = from_dlpack(self._compile_wgrad_ptrs, assumed_align=8).iterator
            single_expert_fake = self._make_fake_cute_tensor_from_desc(self.single_expert_wgrad_desc, assumed_align=16)
        raw_compiled = cute.compile(
            kernel,
            a_fake,
            b_fake,
            out_fake,
            offsets_fake,
            workspace_fake,
            max_active_clusters,
            fake_stream,
            single_expert_fake,
            options="--enable-tvm-ffi",
        )
        cached_workspace = from_dlpack(self._workspace, assumed_align=128, enable_tvm_ffi=True)
        if self.weight_mode == MoEWeightMode.DISCRETE:
            self._single_expert_placeholder = torch.empty_strided(
                self.single_expert_wgrad_desc.shape,
                self.single_expert_wgrad_desc.stride,
                dtype=self.single_expert_wgrad_desc.dtype,
                device=self.single_expert_wgrad_desc.device,
            )
            self._validate_data_alignment(self._single_expert_placeholder, "single expert placeholder")
            cached_single_expert = from_dlpack(
                self._single_expert_placeholder,
                assumed_align=16,
                enable_tvm_ffi=True,
            )
        else:
            cached_single_expert = None

        def tensor_api(a_tensor, b_tensor, output, offsets, stream) -> None:
            out_arg = output if self.weight_mode == MoEWeightMode.DENSE else int(output.data_ptr())
            raw_compiled(
                a_tensor,
                b_tensor,
                out_arg,
                offsets,
                cached_workspace,
                stream,
                cached_single_expert,
            )

        self._compiled_kernel = tensor_api

    def _validate_live_input(
        self,
        tensor: torch.Tensor,
        sample: TensorDesc,
        name: str,
        *,
        token_axis: int,
    ) -> TensorDesc:
        desc = self._make_tensor_desc(tensor, name=name)
        if desc.dtype != sample.dtype:
            raise ValueError(f"{name} dtype mismatch: expected {sample.dtype}, got {desc.dtype}")
        if desc.device != sample.device:
            raise ValueError(f"{name} device mismatch: expected {sample.device}, got {desc.device}")
        for axis, (actual, expected) in enumerate(zip(desc.shape, sample.shape)):
            if axis != token_axis and actual != expected:
                raise ValueError(f"{name} shape mismatch: expected static dimension {expected} at axis {axis}, got {actual}")
        expected_major = self.a_major if name == "a_tensor" else self.b_major
        actual_major = self._infer_a_major(desc) if name == "a_tensor" else self._infer_b_major(desc)
        if actual_major != expected_major:
            raise ValueError(f"{name} layout mismatch: expected {expected_major}-major, got {actual_major}-major")
        return desc

    def _validate_live_output(self, tensor: torch.Tensor) -> None:
        desc = self._make_tensor_desc(tensor, name="wgrad_tensor")
        expected = (self.expert_cnt, *self.wgrad_shape)
        if desc.shape != expected:
            raise ValueError(f"wgrad_tensor shape mismatch: expected {expected}, got {desc.shape}")
        if desc.stride != (
            self.wgrad_shape[0] * self.wgrad_shape[1],
            self.wgrad_shape[1],
            1,
        ):
            raise ValueError("wgrad_tensor must be contiguous in expert/M/N order")
        if desc.dtype != self.wgrad_dtype:
            raise ValueError(f"wgrad_tensor dtype mismatch: expected {self.wgrad_dtype}, got {desc.dtype}")
        if desc.device != self.a_desc.device:
            raise ValueError(f"wgrad_tensor device mismatch: expected {self.a_desc.device}, got {desc.device}")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: Optional[torch.Tensor],
        sfb_tensor: Optional[torch.Tensor],
        offsets_tensor: torch.Tensor,
        wgrad_tensor: Optional[torch.Tensor] = None,
        wgrad_ptrs: Optional[torch.Tensor] = None,
        global_scale_a: Optional[torch.Tensor] = None,
        global_scale_b: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        current_stream = self._get_default_stream(current_stream)
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        forbidden = (
            ("sfa_tensor", sfa_tensor),
            ("sfb_tensor", sfb_tensor),
            ("global_scale_a", global_scale_a),
            ("global_scale_b", global_scale_b),
        )
        for name, value in forbidden:
            if value is not None:
                raise ValueError(f"BF16 forbids scale control {name}")
        a_desc = self._validate_live_input(a_tensor, self.a_desc, "a_tensor", token_axis=1)
        b_desc = self._validate_live_input(b_tensor, self.b_desc, "b_tensor", token_axis=0)
        tokens_sum = a_desc.shape[1]
        if b_desc.shape[0] != tokens_sum:
            raise ValueError("a_tensor and b_tensor token dimensions must match")
        offsets_desc = self._make_tensor_desc(offsets_tensor, name="offsets_tensor")
        if offsets_desc.shape != (self.expert_cnt,) or offsets_desc.stride != (1,) or offsets_desc.dtype != torch.int32:
            raise ValueError("offsets_tensor must be a contiguous rank-1 int32 tensor with one entry per expert")
        if offsets_desc.device != self.a_desc.device:
            raise ValueError(f"offsets_tensor device mismatch: expected {self.a_desc.device}, got {offsets_desc.device}")
        self._validate_offsets_once(offsets_tensor, tokens_sum=tokens_sum)
        self._validate_data_alignment(a_tensor, "a_tensor")
        self._validate_data_alignment(b_tensor, "b_tensor")
        self._validate_data_alignment(offsets_tensor, "offsets_tensor", 4)

        if self.weight_mode == MoEWeightMode.DENSE:
            if wgrad_tensor is None or wgrad_ptrs is not None:
                raise ValueError("Dense execution requires wgrad_tensor and forbids wgrad_ptrs")
            self._validate_live_output(wgrad_tensor)
            self._validate_data_alignment(wgrad_tensor, "wgrad_tensor")
            output = wgrad_tensor
        else:
            if wgrad_tensor is not None:
                self._validate_live_output(wgrad_tensor)
                self._validate_data_alignment(wgrad_tensor, "wgrad_tensor")
            generated_wgrad_ptrs = wgrad_ptrs is None
            if wgrad_ptrs is None:
                if wgrad_tensor is None:
                    raise ValueError("Discrete execution requires wgrad_tensor or wgrad_ptrs")
                stride_bytes = wgrad_tensor.stride(0) * wgrad_tensor.element_size()
                with _torch_stream_context(current_stream, wgrad_tensor.device):
                    wgrad_ptrs = torch.tensor(
                        [wgrad_tensor.data_ptr() + index * stride_bytes for index in range(self.expert_cnt)],
                        dtype=torch.int64,
                        device=wgrad_tensor.device,
                    )
            _require_pointer_tensor(wgrad_ptrs, "wgrad_ptrs", self.expert_cnt)
            if wgrad_ptrs.device != self.a_desc.device:
                raise ValueError(f"wgrad_ptrs must be on {self.a_desc.device}, got {wgrad_ptrs.device}")
            self._validate_pointer_array_alignment(wgrad_ptrs)
            if not generated_wgrad_ptrs:
                self._validate_pointer_values_once(wgrad_ptrs)
            self._record_pointer_stream(wgrad_ptrs, current_stream)
            output = wgrad_ptrs
        self._compiled_kernel(a_tensor, b_tensor, output, offsets_tensor, current_stream)
