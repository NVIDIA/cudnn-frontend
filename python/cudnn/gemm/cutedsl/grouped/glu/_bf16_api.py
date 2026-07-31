# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private BF16 API for the SM100 grouped GEMM GLU kernel."""

import os
import weakref
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cute.runtime import from_dlpack, make_fake_stream
import torch

from cudnn.api_base import APIBase, TensorDesc
from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.gemm.cutedsl.discrete_grouped.discrete_kernel_utils import _require_pointer_tensor

from ..moe_utils import MoEWeightMode
from .moe_grouped_gemm_glu_bias import MoEGroupedGemmGluBiasBf16Kernel

_OUTPUT_DTYPES = [torch.bfloat16, torch.float16, torch.float32]


class GroupedGemmGluBf16API(APIBase):
    """Descriptor-first lifecycle API for BF16 GLU forward."""

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_b: Optional[torch.Tensor] = None,
        sample_bias: Optional[torch.Tensor] = None,
        sample_prob: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        b_shape: Optional[Tuple[int, ...]] = None,
        b_dtype: Optional[torch.dtype] = None,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        vector_f32: bool = False,
        m_aligned: int = 256,
        generate_c: bool = False,
        act_func: str = "swiglu",
        b_major: str = "k",
        use_dynamic_sched: bool = False,
    ) -> None:
        super().__init__()
        self._warn_experimental_api()

        if sample_b is not None and num_experts is None:
            self.weight_mode = MoEWeightMode.DENSE
        elif sample_b is None and num_experts is not None:
            self.weight_mode = MoEWeightMode.DISCRETE
            if b_shape is None or b_dtype is None:
                raise ValueError("b_shape and b_dtype are required in discrete mode")
        else:
            raise ValueError("Provide sample_b for dense mode or (num_experts, b_shape, b_dtype) " "for discrete mode, but not both")

        self.a_desc = self._make_tensor_desc(sample_a, name="sample_a")
        self.c_desc = self._make_tensor_desc(sample_c, name="sample_c")
        self.d_desc = self._make_tensor_desc(sample_d, name="sample_d")
        self.padded_offsets_desc = self._make_tensor_desc(sample_padded_offsets, name="sample_padded_offsets")
        self.alpha_desc = self._make_tensor_desc(sample_alpha, name="sample_alpha")
        self.b_desc = self._make_tensor_desc(sample_b, name="sample_b")
        self.bias_desc = self._make_tensor_desc(sample_bias, name="sample_bias")
        self.prob_desc = self._make_tensor_desc(sample_prob, name="sample_prob")

        self._sample_offset_values = self._copy_values_to_host(sample_padded_offsets)
        self._sample_offsets_ref = weakref.ref(sample_padded_offsets)
        self._sample_offsets_version = int(sample_padded_offsets._version)
        self._sample_data_ptrs = {
            name: tensor.data_ptr()
            for name, tensor in (
                ("sample_a", sample_a),
                ("sample_b", sample_b),
                ("sample_c", sample_c),
                ("sample_d", sample_d),
                ("sample_padded_offsets", sample_padded_offsets),
                ("sample_alpha", sample_alpha),
                ("sample_bias", sample_bias),
                ("sample_prob", sample_prob),
            )
            if tensor is not None
        }

        self.expert_cnt = self.b_desc.shape[2] if self.weight_mode == MoEWeightMode.DENSE and self.b_desc.ndim == 3 else int(num_experts or 0)
        self.b_shape = tuple(b_shape) if b_shape is not None else None
        self.b_dtype = b_dtype if b_dtype is not None else self.b_desc.dtype
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.use_2cta_instrs = self.mma_tiler_mn[0] == 256
        self.cluster_shape_mn = tuple(cluster_shape_mn or ((2, 1) if self.use_2cta_instrs else (1, 1)))
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.generate_c = generate_c
        self.act_func = act_func
        self.b_major = b_major
        self.use_dynamic_sched = use_dynamic_sched
        self._has_bias = self.bias_desc is not None
        self._kernel = MoEGroupedGemmGluBiasBf16Kernel
        self._workspace: Optional[torch.Tensor] = None
        self._compile_b_ptrs: Optional[torch.Tensor] = None
        self._validated_offsets: dict[int, tuple] = {}
        self._validated_pointer_values: dict[int, tuple] = {}
        self.num_cluster_overlap_margin = int(os.getenv("CUDNNFE_CLUSTER_OVERLAP_MARGIN", "0"))

    @staticmethod
    def _expect_shape(desc: TensorDesc, expected: Tuple[int, ...], name: str) -> None:
        if desc.shape != expected:
            raise ValueError(f"{name} shape mismatch: expected {expected}, got {desc.shape}")

    @staticmethod
    def _expect_stride(desc: TensorDesc, expected: Tuple[int, ...], name: str) -> None:
        if desc.stride != expected:
            raise ValueError(f"{name} must use the source-compatible layout with stride " f"{expected}, got {desc.stride}")

    @staticmethod
    def _expect_device(desc: TensorDesc, device: torch.device, name: str) -> None:
        if desc.device != device:
            raise ValueError(f"{name} must be on {device}, got {desc.device}")

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
    def _validate_offset_sequence(values: Tuple[int, ...], *, expert_cnt: int, tensor_m: int) -> None:
        if len(values) != expert_cnt:
            raise ValueError(f"padded_offsets length mismatch: expected {expert_cnt}, got {len(values)}")
        previous = 0
        for index, value in enumerate(values):
            if value < previous:
                raise ValueError("padded_offsets must be a non-decreasing cumulative sum; " f"index {index} is {value} after {previous}")
            if value % MoEGroupedGemmGluBiasBf16Kernel.FIX_PAD_SIZE != 0:
                raise ValueError(f"padded_offsets[{index}] must be 256-aligned, got {value}")
            previous = value
        if not values or values[-1] <= 0 or values[-1] > tensor_m:
            raise ValueError(f"padded_offsets last value must be in [1, {tensor_m}], got " f"{values[-1] if values else None}")

    def _validate_offsets_once(self, offsets: torch.Tensor, *, tensor_m: int) -> None:
        extra = (self.expert_cnt, tensor_m)
        if self._is_validation_cached(self._validated_offsets, offsets, extra):
            return
        values = self._copy_values_to_host(offsets)
        self._validate_offset_sequence(values, expert_cnt=self.expert_cnt, tensor_m=tensor_m)
        self._remember_validation(self._validated_offsets, offsets, extra)

    def _validate_pointer_values_once(self, b_ptrs: torch.Tensor) -> None:
        if self._is_validation_cached(self._validated_pointer_values, b_ptrs, self.expert_cnt):
            return
        pointer_values = self._copy_values_to_host(b_ptrs)
        if any(value == 0 or value % 16 != 0 for value in pointer_values):
            raise ValueError("b_ptrs entries must be non-null and 16-byte aligned")
        self._remember_validation(self._validated_pointer_values, b_ptrs, self.expert_cnt)

    @staticmethod
    def _validate_data_alignment(tensor: torch.Tensor, name: str) -> None:
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} data pointer must be 16-byte aligned")

    @staticmethod
    def _validate_pointer_array_alignment(tensor: torch.Tensor) -> None:
        if tensor.data_ptr() % 8 != 0:
            raise ValueError("b_ptrs data pointer must be 8-byte aligned")

    @staticmethod
    def _record_pointer_stream(b_ptrs: torch.Tensor, current_stream: cuda.CUstream) -> None:
        handle = int(current_stream)
        torch_current = torch.cuda.current_stream(b_ptrs.device)
        torch_default = torch.cuda.default_stream(b_ptrs.device)
        if handle == torch_current.cuda_stream:
            launch_stream = torch_current
        elif handle == torch_default.cuda_stream:
            launch_stream = torch_default
        else:
            launch_stream = torch.cuda.ExternalStream(handle, device=b_ptrs.device)
        b_ptrs.record_stream(launch_stream)

    def check_support(self) -> bool:
        if self.a_desc.ndim != 3:
            raise ValueError(f"sample_a must be rank-3, got {self.a_desc.shape}")
        tensor_m, k, one = self.a_desc.shape
        if one != 1:
            raise ValueError(f"sample_a trailing dimension must be 1, got {one}")

        if self.weight_mode == MoEWeightMode.DENSE:
            if self.b_desc.ndim != 3:
                raise ValueError(f"sample_b must be rank-3, got {self.b_desc.shape}")
            n, b_k, experts = self.b_desc.shape
            if b_k != k:
                raise ValueError(f"sample_b K dimension ({b_k}) must match sample_a ({k})")
            if experts != self.expert_cnt:
                raise ValueError("sample_b expert dimension is inconsistent")
            self._expect_stride(self.b_desc, (k, 1, n * k), "sample_b")
        else:
            if len(self.b_shape) not in (2, 3):
                raise ValueError(f"b_shape must be rank-2 or rank-3, got {self.b_shape}")
            n, b_k = self.b_shape[:2]
            if len(self.b_shape) == 3 and self.b_shape[2] != 1:
                raise ValueError(f"b_shape trailing dimension must be 1, got {self.b_shape}")
            if b_k != k:
                raise ValueError(f"b_shape K dimension ({b_k}) must match sample_a ({k})")
        if n % 64 != 0:
            raise ValueError(f"N must be divisible by 64 for paired GLU blocks, got {n}")

        n_out = n // 2
        self._expect_shape(self.c_desc, (tensor_m, n, 1), "sample_c")
        self._expect_shape(self.d_desc, (tensor_m, n_out, 1), "sample_d")
        self._expect_shape(self.padded_offsets_desc, (self.expert_cnt,), "sample_padded_offsets")
        self._expect_shape(self.alpha_desc, (self.expert_cnt,), "sample_alpha")
        if self.prob_desc is None:
            raise ValueError("sample_prob is required")
        self._expect_shape(self.prob_desc, (tensor_m, 1, 1), "sample_prob")

        self._expect_stride(self.a_desc, (k, 1, tensor_m * k), "A tensor")
        self._expect_stride(self.c_desc, (n, 1, tensor_m * n), "sample_c")
        self._expect_stride(self.d_desc, (n_out, 1, tensor_m * n_out), "sample_d")
        self._expect_stride(self.padded_offsets_desc, (1,), "sample_padded_offsets")
        self._expect_stride(self.alpha_desc, (1,), "sample_alpha")
        self._expect_stride(self.prob_desc, (1, 1, 1), "sample_prob")

        self._check_dtype(self.a_desc, torch.bfloat16, "sample_a")
        if self.weight_mode == MoEWeightMode.DENSE:
            self._check_dtype(self.b_desc, torch.bfloat16, "sample_b")
        self._check_dtype(self.b_dtype, torch.bfloat16, "b_dtype")
        self._check_dtype(self.c_desc, _OUTPUT_DTYPES, "sample_c")
        self._check_dtype(self.d_desc, _OUTPUT_DTYPES, "sample_d")
        self._check_dtype(self.padded_offsets_desc, torch.int32, "sample_padded_offsets")
        self._check_dtype(self.alpha_desc, torch.float32, "sample_alpha")
        self._check_dtype(self.prob_desc, torch.float32, "sample_prob")

        device = self.a_desc.device
        for desc, name in (
            (self.c_desc, "sample_c"),
            (self.d_desc, "sample_d"),
            (self.padded_offsets_desc, "sample_padded_offsets"),
            (self.alpha_desc, "sample_alpha"),
            (self.prob_desc, "sample_prob"),
        ):
            self._expect_device(desc, device, name)
        if self.b_desc is not None:
            self._expect_device(self.b_desc, device, "sample_b")

        if self.bias_desc is not None:
            self._expect_shape(self.bias_desc, (n, self.expert_cnt), "sample_bias")
            self._expect_stride(self.bias_desc, (1, n), "sample_bias")
            self._check_dtype(self.bias_desc, _OUTPUT_DTYPES, "sample_bias")
            self._expect_device(self.bias_desc, device, "sample_bias")

        for name, data_ptr in self._sample_data_ptrs.items():
            if data_ptr % 16 != 0:
                raise ValueError(f"{name} data pointer must be 16-byte aligned")

        if self.acc_dtype != torch.float32:
            raise ValueError(f"acc_dtype must be torch.float32, got {self.acc_dtype}")
        if self.m_aligned != MoEGroupedGemmGluBiasBf16Kernel.FIX_PAD_SIZE:
            raise ValueError(f"m_aligned must be 256, got {self.m_aligned}")
        if self.act_func not in ("swiglu", "geglu"):
            raise ValueError(f"act_func must be 'swiglu' or 'geglu', got {self.act_func}")
        if self.b_major not in ("k", "n"):
            raise ValueError(f"b_major must be 'k' or 'n', got {self.b_major}")
        if self.expert_cnt <= 0 or self.expert_cnt > 1024:
            raise ValueError(f"expert count must be in [1, 1024], got {self.expert_cnt}")
        if tensor_m % 256 != 0:
            raise ValueError(f"sample_a M dimension must be 256-aligned, got {tensor_m}")

        self._validate_offset_sequence(
            self._sample_offset_values,
            expert_cnt=self.expert_cnt,
            tensor_m=tensor_m,
        )
        sample_offsets = self._sample_offsets_ref()
        if sample_offsets is not None and int(sample_offsets._version) == self._sample_offsets_version:
            self._remember_validation(
                self._validated_offsets,
                sample_offsets,
                (self.expert_cnt, tensor_m),
            )
        elif sample_offsets is not None:
            self._validate_offsets_once(sample_offsets, tensor_m=tensor_m)

        if not self._kernel.can_implement(
            _convert_to_cutlass_data_type(torch.bfloat16),
            _convert_to_cutlass_data_type(self.c_desc.dtype),
            _convert_to_cutlass_data_type(self.d_desc.dtype),
            _convert_to_cutlass_data_type(self.acc_dtype),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            tensor_m,
            n,
            k,
            self.expert_cnt,
            "k",
            self.b_major,
            "n",
            self.m_aligned,
        ):
            raise ValueError("Unsupported BF16 grouped GEMM GLU tile, cluster, alignment, " "or layout configuration")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        major, minor = torch.cuda.get_device_capability(self.a_desc.device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmGluSm100 requires SM100+, found SM{compute_capability} " f"on {self.a_desc.device}")

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
            vectorized_f32=self.vector_f32,
            expert_cnt=self.expert_cnt,
            weight_mode=self.weight_mode,
            use_dynamic_sched=self.use_dynamic_sched,
            act_func=self.act_func,
            enable_bias=self._has_bias,
            generate_c=self.generate_c,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1]) - self.num_cluster_overlap_margin
        if max_active_clusters <= 0:
            raise ValueError("max_active_clusters must be > 0 after applying " "CUDNNFE_CLUSTER_OVERLAP_MARGIN")

        workspace_bytes = kernel.get_workspace_bytes()
        self._workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device=self.a_desc.device)
        if self._workspace.data_ptr() % 128 != 0:
            raise RuntimeError("workspace allocation must be 128-byte aligned")
        workspace_ptr = from_dlpack(self._workspace, assumed_align=128).iterator
        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        valid_m = cute.sym_int(divisibility=256)
        a_fake = self._make_fake_cute_compact_tensor(
            self.a_desc.dtype,
            self.a_desc.shape,
            self.a_desc.stride_order,
            dynamic_mode=0,
            divisibility=256,
        )
        c_fake = self._make_fake_cute_compact_tensor(
            self.c_desc.dtype,
            self.c_desc.shape,
            self.c_desc.stride_order,
            dynamic_mode=0,
            divisibility=256,
        )
        d_fake = self._make_fake_cute_compact_tensor(
            self.d_desc.dtype,
            self.d_desc.shape,
            self.d_desc.stride_order,
            dynamic_mode=0,
            divisibility=256,
        )
        prob_fake = self._make_fake_cute_tensor(self.prob_desc.dtype, (valid_m, 1, 1), self.prob_desc.stride)

        if self.weight_mode == MoEWeightMode.DENSE:
            b_fake = self._make_fake_cute_tensor_from_desc(self.b_desc)
            n_value = cutlass.Int32(0)
            k_value = cutlass.Int32(0)
            b_stride = cutlass.Int64(0)
            b_major_mode = OperandMajorMode.K
        else:
            self._compile_b_ptrs = torch.empty((self.expert_cnt,), dtype=torch.int64, device=self.a_desc.device)
            self._validate_pointer_array_alignment(self._compile_b_ptrs)
            b_fake = from_dlpack(self._compile_b_ptrs, assumed_align=8).iterator
            n, k = self.b_shape[:2]
            n_value = cutlass.Int32(n)
            k_value = cutlass.Int32(k)
            b_stride = cutlass.Int64(k if self.b_major == "k" else n)
            b_major_mode = OperandMajorMode.K if self.b_major == "k" else OperandMajorMode.MN

        raw_compiled = cute.compile(
            kernel,
            a=a_fake,
            b=b_fake,
            n=n_value,
            k=k_value,
            b_stride_size=b_stride,
            b_major_mode=b_major_mode,
            workspace_ptr=workspace_ptr,
            c=c_fake,
            d=d_fake,
            padded_offsets=self._make_fake_cute_tensor_from_desc(self.padded_offsets_desc),
            alpha=self._make_fake_cute_tensor_from_desc(self.alpha_desc),
            prob=prob_fake,
            bias=self._make_fake_cute_tensor_from_desc(self.bias_desc),
            max_active_clusters=max_active_clusters,
            stream=fake_stream,
            linear_offset=cutlass.Float32(0.0),
            options="--enable-tvm-ffi",
        )

        cached_n = n_value
        cached_k = k_value
        cached_b_stride = b_stride

        def tensor_api(
            a_tensor: torch.Tensor,
            c_tensor: torch.Tensor,
            d_tensor: torch.Tensor,
            padded_offsets: torch.Tensor,
            alpha_tensor: torch.Tensor,
            b_tensor: Optional[torch.Tensor],
            b_ptrs: Optional[torch.Tensor],
            bias_tensor: Optional[torch.Tensor],
            prob_tensor: torch.Tensor,
            stream: cuda.CUstream,
            linear_offset: float,
        ) -> None:
            b_arg = b_tensor if self.weight_mode == MoEWeightMode.DENSE else int(b_ptrs.data_ptr())
            raw_compiled(
                a_tensor,
                b_arg,
                cached_n,
                cached_k,
                cached_b_stride,
                workspace_ptr,
                c_tensor,
                d_tensor,
                padded_offsets,
                alpha_tensor,
                prob_tensor,
                bias_tensor,
                stream,
                cutlass.Float32(linear_offset),
            )

        self._compiled_kernel = tensor_api

    def _validate_live_tensor(
        self,
        tensor: torch.Tensor,
        sample: TensorDesc,
        name: str,
        *,
        dynamic_m: bool = False,
    ) -> TensorDesc:
        desc = self._make_tensor_desc(tensor, name=name)
        if desc.dtype != sample.dtype:
            raise ValueError(f"{name} dtype mismatch: expected {sample.dtype}, got {desc.dtype}")
        if desc.device != sample.device:
            raise ValueError(f"{name} device mismatch: expected {sample.device}, got {desc.device}")
        if dynamic_m:
            if desc.shape[1:] != sample.shape[1:]:
                raise ValueError(f"{name} shape suffix mismatch: expected {sample.shape[1:]}, " f"got {desc.shape[1:]}")
            if desc.stride_order != sample.stride_order:
                raise ValueError(f"{name} layout mismatch: expected stride order " f"{sample.stride_order}, got {desc.stride_order}")
        elif desc.shape != sample.shape or desc.stride != sample.stride:
            raise ValueError(f"{name} descriptor mismatch: expected shape/stride " f"{sample.shape}/{sample.stride}, got {desc.shape}/{desc.stride}")
        return desc

    def execute(
        self,
        a_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        b_tensor: Optional[torch.Tensor] = None,
        b_ptrs: Optional[torch.Tensor] = None,
        bias_tensor: Optional[torch.Tensor] = None,
        prob_tensor: Optional[torch.Tensor] = None,
        linear_offset: float = 0.0,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        current_stream = self._get_default_stream(current_stream)
        if self._compiled_kernel is None:
            raise RuntimeError("Kernel not compiled; call compile() first")
        if prob_tensor is None:
            raise ValueError("prob_tensor is required")

        a_desc = self._validate_live_tensor(a_tensor, self.a_desc, "a_tensor", dynamic_m=True)
        c_desc = self._validate_live_tensor(c_tensor, self.c_desc, "c_tensor", dynamic_m=True)
        d_desc = self._validate_live_tensor(d_tensor, self.d_desc, "d_tensor", dynamic_m=True)
        self._validate_live_tensor(padded_offsets, self.padded_offsets_desc, "padded_offsets")
        self._validate_live_tensor(alpha_tensor, self.alpha_desc, "alpha_tensor")
        prob_desc = self._validate_live_tensor(prob_tensor, self.prob_desc, "prob_tensor", dynamic_m=True)

        tensor_m, k, _ = a_desc.shape
        n = c_desc.shape[1]
        n_out = n // 2
        if tensor_m % 256 != 0:
            raise ValueError(f"a_tensor M dimension must be 256-aligned, got {tensor_m}")
        self._expect_shape(c_desc, (tensor_m, n, 1), "c_tensor")
        self._expect_shape(d_desc, (tensor_m, n_out, 1), "d_tensor")
        self._expect_shape(prob_desc, (tensor_m, 1, 1), "prob_tensor")
        self._expect_stride(a_desc, (k, 1, tensor_m * k), "a_tensor")
        self._expect_stride(c_desc, (n, 1, tensor_m * n), "c_tensor")
        self._expect_stride(d_desc, (n_out, 1, tensor_m * n_out), "d_tensor")
        self._expect_stride(prob_desc, (1, 1, 1), "prob_tensor")
        self._validate_offsets_once(padded_offsets, tensor_m=tensor_m)

        for tensor, name in (
            (a_tensor, "a_tensor"),
            (c_tensor, "c_tensor"),
            (d_tensor, "d_tensor"),
            (padded_offsets, "padded_offsets"),
            (alpha_tensor, "alpha_tensor"),
            (prob_tensor, "prob_tensor"),
        ):
            self._validate_data_alignment(tensor, name)

        if self._has_bias:
            if bias_tensor is None:
                raise ValueError("bias_tensor is required because the API was compiled " "with sample_bias")
            self._validate_live_tensor(bias_tensor, self.bias_desc, "bias_tensor")
            self._validate_data_alignment(bias_tensor, "bias_tensor")
        elif bias_tensor is not None:
            raise ValueError("bias_tensor must be omitted because the API was compiled " "without sample_bias")

        if self.weight_mode == MoEWeightMode.DENSE:
            if b_tensor is None or b_ptrs is not None:
                raise ValueError("Dense execution requires b_tensor and forbids b_ptrs")
            self._validate_live_tensor(b_tensor, self.b_desc, "b_tensor")
            self._validate_data_alignment(b_tensor, "b_tensor")
        else:
            if b_tensor is not None or b_ptrs is None:
                raise ValueError("Discrete execution requires b_ptrs and forbids b_tensor")
            _require_pointer_tensor(b_ptrs, "b_ptrs", self.expert_cnt)
            if b_ptrs.device != self.a_desc.device:
                raise ValueError(f"b_ptrs must be on the same device as a_tensor " f"({self.a_desc.device}), got {b_ptrs.device}")
            if b_ptrs.data_ptr() % 8 != 0:
                raise ValueError("b_ptrs data pointer must be 8-byte aligned")
            self._validate_pointer_values_once(b_ptrs)
            self._record_pointer_stream(b_ptrs, current_stream)

        self._compiled_kernel(
            a_tensor,
            c_tensor,
            d_tensor,
            padded_offsets,
            alpha_tensor,
            b_tensor,
            b_ptrs,
            bias_tensor,
            prob_tensor,
            current_stream,
            linear_offset,
        )
