# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone grouped weight-only NVFP4 projection API for SM100/SM103."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal, Optional

import cutlass
from cutlass import cute
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.tensor_adapter import detect_framework, get_data_ptr, get_device

from . import _kernel_sm100

Epilogue = Literal["linear", "squared_relu"]


@dataclass(frozen=True)
class _Projection:
    epilogue: Epilogue
    k: int
    n: int
    scheduler_rows: int
    entry: object


_PROJECTIONS = {
    ("squared_relu", 2688, 1856): _Projection(
        epilogue="squared_relu",
        k=2688,
        n=1856,
        scheduler_rows=384,
        entry=_kernel_sm100.weight_only_nvfp4_lightning_fc1_sm100,
    ),
    ("linear", 1856, 2688): _Projection(
        epilogue="linear",
        k=1856,
        n=2688,
        scheduler_rows=128,
        entry=_kernel_sm100.weight_only_nvfp4_lightning_fc2_sm100,
    ),
}

_PLAN_CACHE: dict[tuple[int, int, Epilogue], "GroupedGemmWeightOnlyNvfp4"] = {}


def _device_index(device) -> int:
    index = getattr(device, "index", None)
    return 0 if index is None else int(index)


def _projection(epilogue: Epilogue, k: int, n: int) -> _Projection:
    projection = _PROJECTIONS.get((epilogue, k, n))
    if projection is None:
        supported = ", ".join(f"epilogue={item.epilogue!r}, K={item.k}, N={item.n}" for item in _PROJECTIONS.values())
        raise NotImplementedError(f"grouped weight-only NVFP4 supports only {supported}; got epilogue={epilogue!r}, K={k}, N={n}")
    return projection


class GroupedGemmWeightOnlyNvfp4(APIBase):
    """Precompiled standalone grouped W4A16 projection.

    The checkpoint-native inputs are packed E2M1 weights in ``uint8`` storage,
    E4M3 group-16 weight scales, BF16 routed tokens, starts-only INT32 expert
    offsets, and one FP32 output factor per expert. The kernel writes a
    preallocated BF16 output and owns no workspace.
    """

    def __init__(
        self,
        sample_routed_tokens: torch.Tensor,
        sample_packed_weight: torch.Tensor,
        sample_weight_scale: torch.Tensor,
        sample_first_token_offset: torch.Tensor,
        sample_factor: torch.Tensor,
        sample_output: torch.Tensor,
        *,
        epilogue: Epilogue,
    ) -> None:
        framework = detect_framework(sample_routed_tokens)
        if framework != "torch":
            raise ValueError(f"GroupedGemmWeightOnlyNvfp4 supports torch tensors; got {framework!r}")
        super().__init__()
        self._warn_experimental_api()
        self._framework = framework
        self.epilogue = epilogue

        self.routed_tokens_desc = self._make_tensor_desc(sample_routed_tokens, name="routed_tokens", canonical=True)
        self.packed_weight_desc = self._make_tensor_desc(sample_packed_weight, name="packed_weight", canonical=True)
        self.weight_scale_desc = self._make_tensor_desc(sample_weight_scale, name="weight_scale", canonical=True)
        self.first_token_offset_desc = self._make_tensor_desc(sample_first_token_offset, name="first_token_offset", canonical=True)
        self.factor_desc = self._make_tensor_desc(sample_factor, name="factor", canonical=True)
        self.output_desc = self._make_tensor_desc(sample_output, name="output", canonical=True)

        token_shape = self.routed_tokens_desc.shape
        weight_shape = self.packed_weight_desc.shape
        self._value_error_if(len(token_shape) != 3 or token_shape[0] != 1, f"routed_tokens must have shape [1,S,K]; got {token_shape}")
        self._value_error_if(len(weight_shape) != 3, f"packed_weight must have shape [E,N,K/2]; got {weight_shape}")
        k = int(token_shape[2])
        n = int(weight_shape[1])
        self._projection = _projection(epilogue, k, n)
        self.num_experts = int(weight_shape[0])
        self.device = self.routed_tokens_desc.device

    def _validate_metadata(
        self,
        routed_tokens,
        packed_weight,
        weight_scale,
        first_token_offset,
        factor,
        output,
    ) -> int:
        tensors = (
            ("routed_tokens", routed_tokens),
            ("packed_weight", packed_weight),
            ("weight_scale", weight_scale),
            ("first_token_offset", first_token_offset),
            ("factor", factor),
            ("output", output),
        )
        for name, tensor in tensors:
            if isinstance(tensor, TensorDesc):
                device = tensor.device
            else:
                framework = detect_framework(tensor)
                self._value_error_if(framework != "torch", f"{name} must be a torch tensor; got {framework!r}")
                device = get_device(tensor)
            self._value_error_if(device.type != "cuda", f"{name} must be a CUDA tensor; got {device}")
            self._value_error_if(device != self.device, f"{name} must be on plan device {self.device}; got {device}")

        projection = self._projection
        e, k, n = self.num_experts, projection.k, projection.n
        token_desc = routed_tokens if isinstance(routed_tokens, TensorDesc) else self._make_tensor_desc(routed_tokens, name="routed_tokens", canonical=True)
        packed_desc = packed_weight if isinstance(packed_weight, TensorDesc) else self._make_tensor_desc(packed_weight, name="packed_weight", canonical=True)
        scale_desc = weight_scale if isinstance(weight_scale, TensorDesc) else self._make_tensor_desc(weight_scale, name="weight_scale", canonical=True)
        offsets_desc = (
            first_token_offset
            if isinstance(first_token_offset, TensorDesc)
            else self._make_tensor_desc(first_token_offset, name="first_token_offset", canonical=True)
        )
        factor_desc = factor if isinstance(factor, TensorDesc) else self._make_tensor_desc(factor, name="factor", canonical=True)
        output_desc = output if isinstance(output, TensorDesc) else self._make_tensor_desc(output, name="output", canonical=True)

        self._check_dtype(token_desc, cutlass.BFloat16, "routed_tokens")
        self._check_dtype(packed_desc, cutlass.Uint8, "packed_weight")
        self._check_dtype(scale_desc, cutlass.Float8E4M3FN, "weight_scale")
        self._check_dtype(offsets_desc, cutlass.Int32, "first_token_offset")
        self._check_dtype(factor_desc, cutlass.Float32, "factor")
        self._check_dtype(output_desc, cutlass.BFloat16, "output")

        self._check_tensor_shape(packed_desc, (e, n, k // 2), "packed_weight")
        self._check_tensor_shape(scale_desc, (e, n, k // 16), "weight_scale")
        self._check_tensor_shape(offsets_desc, (e, 1, 1), "first_token_offset")
        self._check_tensor_shape(factor_desc, (e, 1, 1), "factor")
        self._value_error_if(
            len(token_desc.shape) != 3 or token_desc.shape[0] != 1 or token_desc.shape[2] != k,
            f"routed_tokens must have shape [1,S,{k}]; got {token_desc.shape}",
        )
        total_rows = int(token_desc.shape[1])
        self._value_error_if(total_rows < 1, "routed_tokens requires S > 0")
        max_rows = (65535 - e + 1) * projection.scheduler_rows
        self._value_error_if(total_rows > max_rows, f"routed_tokens S={total_rows} exceeds the schedule limit {max_rows} for E={e}")
        self._check_tensor_shape(output_desc, (1, total_rows, n), "output")

        self._check_tensor_stride(packed_desc, (n * (k // 2), k // 2, 1), name="packed_weight")
        self._check_tensor_stride(scale_desc, (n * (k // 16), k // 16, 1), name="weight_scale")
        self._check_tensor_stride(offsets_desc, (1, e, e), name="first_token_offset")
        self._check_tensor_stride(factor_desc, (1, e, e), name="factor")
        self._value_error_if(token_desc.stride[1:] != (k, 1), f"routed_tokens inner strides must be {(k, 1)}; got {token_desc.stride}")
        self._value_error_if(output_desc.stride[1:] != (n, 1), f"output inner strides must be {(n, 1)}; got {output_desc.stride}")

        for name, tensor, alignment in (
            ("routed_tokens", routed_tokens, 16),
            ("packed_weight", packed_weight, 16),
            ("weight_scale", weight_scale, 16),
            ("first_token_offset", first_token_offset, 4),
            ("factor", factor, 4),
            ("output", output, 16),
        ):
            if not isinstance(tensor, TensorDesc):
                self._value_error_if(get_data_ptr(tensor) % alignment != 0, f"{name} pointer must be {alignment}-byte aligned")
        return total_rows

    def check_support(self) -> bool:
        """Validate the representative tensor contract and target device."""
        import torch

        self._value_error_if(self.device.type != "cuda", f"routed_tokens must be a CUDA tensor; got {self.device}")
        capability = torch.cuda.get_device_capability(self.device)
        self._not_implemented_error_if(
            capability not in ((10, 0), (10, 3)), f"grouped weight-only NVFP4 requires SM100 or SM103; got sm_{capability[0]}{capability[1]}"
        )
        self._value_error_if(not (1 <= self.num_experts <= 128), f"num_experts must be in [1,128]; got {self.num_experts}")
        self._validate_metadata(
            self.routed_tokens_desc,
            self.packed_weight_desc,
            self.weight_scale_desc,
            self.first_token_offset_desc,
            self.factor_desc,
            self.output_desc,
        )
        self._is_supported = True
        return True

    def compile(self) -> None:
        """Compile one symbolic-S kernel for this expert count and epilogue."""
        import torch

        self._ensure_support_checked()
        e, k, n = self.num_experts, self._projection.k, self._projection.n
        sym_s = cute.sym_int(divisibility=1)
        packed = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (e, n, k // 2), stride_order=(2, 1, 0), assumed_align=16)
        scale = cute.runtime.make_fake_compact_tensor(cutlass.Float8E4M3FN, (e, n, k // 16), stride_order=(2, 1, 0), assumed_align=16)
        token = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (1, sym_s, k), stride_order=(2, 1, 0), assumed_align=16)
        offsets = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (e, 1, 1), stride_order=(2, 1, 0), assumed_align=4)
        output = cute.runtime.make_fake_compact_tensor(cutlass.BFloat16, (1, sym_s, n), stride_order=(2, 1, 0), assumed_align=16)
        factor = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (e, 1, 1), stride_order=(2, 1, 0), assumed_align=4)
        options = "--enable-tvm-ffi" if importlib.util.find_spec("tvm_ffi") is not None else ""
        with torch.cuda.device(self.device):
            self._compiled_kernel = cute.compile(
                self._projection.entry,
                packed,
                scale,
                token,
                offsets,
                output,
                factor,
                make_fake_stream(use_tvm_ffi_env_stream=False),
                options=options,
            )

    def execute(
        self,
        routed_tokens: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        first_token_offset: torch.Tensor,
        factor: torch.Tensor,
        output: torch.Tensor,
        *,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        """Launch into ``output`` without allocation, conversion, or fallback."""
        import torch

        self._runtime_error_if(self._compiled_kernel is None, "Kernel not compiled; call compile() first")
        self._validate_metadata(routed_tokens, packed_weight, weight_scale, first_token_offset, factor, output)
        if current_stream is None:
            current_stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        self._compiled_kernel(packed_weight, weight_scale, routed_tokens, first_token_offset, output, factor, current_stream)


def grouped_gemm_weight_only_nvfp4(
    routed_tokens: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    first_token_offset: torch.Tensor,
    factor: torch.Tensor,
    *,
    epilogue: Epilogue,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Allocate and return the BF16 grouped-projection output.

    The cache key excludes routed-token extent ``S`` because the compiled
    kernel represents it symbolically. Runtime buffers are still validated on
    every call before launch.
    """
    framework = detect_framework(routed_tokens)
    if framework != "torch":
        raise ValueError(f"grouped_gemm_weight_only_nvfp4 supports torch tensors; got {framework!r}")
    import torch

    if len(routed_tokens.shape) != 3 or len(packed_weight.shape) != 3:
        raise ValueError(f"expected routed_tokens [1,S,K] and packed_weight [E,N,K/2]; got {tuple(routed_tokens.shape)} and {tuple(packed_weight.shape)}")
    e, n = int(packed_weight.shape[0]), int(packed_weight.shape[1])
    s, k = int(routed_tokens.shape[1]), int(routed_tokens.shape[2])
    _projection(epilogue, k, n)
    output = torch.empty((1, s, n), dtype=torch.bfloat16, device=routed_tokens.device)

    device = get_device(routed_tokens)
    key = (_device_index(device), e, epilogue)
    op = _PLAN_CACHE.get(key)
    if op is None:
        op = GroupedGemmWeightOnlyNvfp4(
            routed_tokens,
            packed_weight,
            weight_scale,
            first_token_offset,
            factor,
            output,
            epilogue=epilogue,
        )
        op.check_support()
        op.compile()
        _PLAN_CACHE[key] = op
    op.execute(
        routed_tokens,
        packed_weight,
        weight_scale,
        first_token_offset,
        factor,
        output,
        current_stream=current_stream,
    )
    return TupleDict(output=output)


__all__ = [
    "GroupedGemmWeightOnlyNvfp4",
    "grouped_gemm_weight_only_nvfp4",
]
