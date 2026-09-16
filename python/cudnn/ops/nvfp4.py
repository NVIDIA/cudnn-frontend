# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit NVFP4 block-scale conversion operations.

These operations materialize the packed E2M1 data and E4M3 scale tensors.  The
result can be reused by FROST block-scale GEMM or by any other consumer that
implements the same F8_128x4 scale layout; conversion is not coupled to a GEMM
plan or hidden inside a graph-pattern matcher.
"""

from __future__ import annotations

import threading
from typing import Any

import cutlass
from cuda.bindings import driver as cuda

from cudnn.api_base import APIBase, TensorDesc, TupleDict
from cudnn.tensor_adapter import detect_framework, get_device, get_shape

from ._nvfp4_block_scale_kernels import (
    BLOCK_SIZE,
    MAX_GROUPS_PER_ROW,
    compile_nvfp4_block_scale_dequantize,
    compile_nvfp4_block_scale_quantize,
)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    return TensorDesc._compute_contiguous_stride(shape)


def _check_single_cuda_device(api: APIBase, *descs: TensorDesc) -> None:
    devices = {desc.device for desc in descs}
    api._value_error_if(
        len(devices) != 1 or next(iter(devices)).type != "cuda",
        f"all tensors must be on one CUDA device; got {sorted(str(device) for device in devices)}",
    )


def _check_sm100_or_newer(api: APIBase, device: Any) -> None:
    """Gate by capability class, not one hard-coded device minor."""

    import torch

    api._runtime_error_if(not torch.cuda.is_available(), "CUDA is not available")
    major, minor = torch.cuda.get_device_capability(device)
    api._not_implemented_error_if(
        major < 10,
        f"NVFP4 block-scale conversion requires SM100+; found SM{major}{minor}",
    )


def _check_problem_shape(api: APIBase, shape: tuple[int, ...]) -> tuple[int, int]:
    api._value_error_if(len(shape) != 3 or shape[0] != 1, f"input must have shape [1,M,K]; got {shape}")
    _, m, k = shape
    api._value_error_if(m <= 0 or m % 128, f"M must be positive and divisible by 128; got {m}")
    api._value_error_if(k <= 0 or k % 64, f"K must be positive and divisible by 64; got {k}")
    api._value_error_if(
        k // BLOCK_SIZE > MAX_GROUPS_PER_ROW,
        f"K/{BLOCK_SIZE} must be <= {MAX_GROUPS_PER_ROW}; got {k // BLOCK_SIZE}",
    )
    return m, k


def _check_desc_contiguous(api: APIBase, name: str, desc: TensorDesc) -> None:
    expected = _contiguous_stride(desc.shape)
    api._value_error_if(desc.stride != expected, f"{name} must be C-contiguous; got shape {desc.shape}, stride {desc.stride}")


def _as_custream(stream: Any, device: Any) -> cuda.CUstream:
    import torch

    if stream is None:
        return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    if isinstance(stream, cuda.CUstream):
        return stream
    if hasattr(stream, "cuda_stream"):
        stream = stream.cuda_stream
    return cuda.CUstream(int(stream))


def _packed_carrier(tensor):
    """Return a metadata-only int8 view accepted by the TVM-FFI kernel ABI."""

    import torch

    if tensor.dtype != torch.int8:
        return tensor.view(torch.int8)
    return tensor


def _runtime_tensor_meta(tensor, *, name: str):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}")
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


def _require_torch_sample(tensor, *, name: str) -> None:
    if detect_framework(tensor) != "torch":
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_runtime_tensor(
    tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtypes: tuple[Any, ...],
    device: Any,
    alignment: int,
) -> None:
    actual_shape, stride, dtype, actual_device = _runtime_tensor_meta(tensor, name=name)
    if actual_shape != shape or stride != _contiguous_stride(shape) or dtype not in dtypes or actual_device != device:
        expected_dtypes = ", ".join(str(item) for item in dtypes)
        raise ValueError(
            f"{name} must have shape {shape}, contiguous stride {_contiguous_stride(shape)}, "
            f"dtype one of ({expected_dtypes}), and device {device}; got shape {actual_shape}, "
            f"stride {stride}, dtype {dtype}, device {actual_device}"
        )
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} pointer must be {alignment}-byte aligned; got 0x{tensor.data_ptr():x}")


class Nvfp4BlockScaleQuantizer(APIBase):
    """Prepared BF16 to packed-E2M1 + E4M3/F8_128x4 conversion plan.

    ``M`` is symbolic in the compiled kernel.  One plan can therefore execute
    any positive ``M`` divisible by 128 while ``K`` remains the plan-time
    specialization.
    """

    def __init__(self, sample_input, sample_encode_scale, sample_packed, sample_scales):
        super().__init__()
        self._warn_experimental_api()
        for name, tensor in (
            ("sample_input", sample_input),
            ("sample_encode_scale", sample_encode_scale),
            ("sample_packed", sample_packed),
            ("sample_scales", sample_scales),
        ):
            _require_torch_sample(tensor, name=name)
        self.input_desc = self._make_tensor_desc(sample_input, name="input", canonical=True)
        self.encode_desc = self._make_tensor_desc(sample_encode_scale, name="encode_scale", canonical=True)
        self.packed_desc = self._make_tensor_desc(
            sample_packed,
            name="packed",
            interpret_uint8_as_fp4x2=True,
            canonical=True,
        )
        self.scales_desc = self._make_tensor_desc(sample_scales, name="scales", canonical=True)
        self.k = self.input_desc.shape[-1]

    def check_support(self) -> bool:
        self._check_dtype(self.input_desc, cutlass.BFloat16, "input")
        self._check_dtype(self.encode_desc, cutlass.Float32, "encode_scale")
        self._check_dtype(self.packed_desc, [cutlass.Float4E2M1FN, cutlass.Uint8], "packed")
        self._check_dtype(self.scales_desc, cutlass.Float8E4M3FN, "scales")
        m, k = _check_problem_shape(self, self.input_desc.shape)
        self._check_tensor_shape(self.encode_desc, (1, 1, 1), "encode_scale")
        self._check_tensor_shape(self.packed_desc, (1, m, k), "packed")
        self._check_tensor_shape(self.scales_desc, (1, m, k // BLOCK_SIZE), "scales")
        for name, desc in (
            ("input", self.input_desc),
            ("encode_scale", self.encode_desc),
            ("packed", self.packed_desc),
            ("scales", self.scales_desc),
        ):
            _check_desc_contiguous(self, name, desc)
        _check_single_cuda_device(self, self.input_desc, self.encode_desc, self.packed_desc, self.scales_desc)
        _check_sm100_or_newer(self, self.input_desc.device)
        self._is_supported = True
        return True

    def compile(self) -> None:
        import torch

        self._ensure_support_checked()
        if self._compiled_kernel is None:
            with torch.cuda.device(self.input_desc.device):
                self._compiled_kernel = compile_nvfp4_block_scale_quantize(k=self.k)

    def execute(
        self,
        input_tensor,
        encode_scale,
        packed_tensor,
        scale_tensor,
        current_stream: cuda.CUstream | None = None,
    ) -> None:
        import torch

        self._runtime_error_if(self._compiled_kernel is None, "Nvfp4BlockScaleQuantizer is not compiled")
        shape, _, _, device = _runtime_tensor_meta(input_tensor, name="input")
        m, k = _check_problem_shape(self, shape)
        self._value_error_if(k != self.k, f"runtime K must equal compiled K={self.k}; got {k}")
        self._value_error_if(device != self.input_desc.device, f"runtime device must equal compiled device {self.input_desc.device}; got {device}")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        packed_dtypes = tuple(dtype for dtype in (fp4_dtype, torch.uint8) if dtype is not None)
        _require_runtime_tensor(input_tensor, name="input", shape=(1, m, k), dtypes=(torch.bfloat16,), device=device, alignment=16)
        _require_runtime_tensor(encode_scale, name="encode_scale", shape=(1, 1, 1), dtypes=(torch.float32,), device=device, alignment=4)
        _require_runtime_tensor(packed_tensor, name="packed", shape=(1, m, k // 2), dtypes=packed_dtypes, device=device, alignment=16)
        _require_runtime_tensor(
            scale_tensor,
            name="scales",
            shape=(1, m, k // BLOCK_SIZE),
            dtypes=(torch.float8_e4m3fn,),
            device=device,
            alignment=16,
        )
        with torch.cuda.device(device):
            self._compiled_kernel(
                input_tensor,
                encode_scale,
                _packed_carrier(packed_tensor),
                scale_tensor,
                _as_custream(current_stream, device),
            )


class Nvfp4BlockScaleDequantizer(APIBase):
    """Prepared packed-E2M1 + E4M3/F8_128x4 to BF16 conversion plan."""

    def __init__(self, sample_packed, sample_scales, sample_decode_scale, sample_output):
        super().__init__()
        self._warn_experimental_api()
        for name, tensor in (
            ("sample_packed", sample_packed),
            ("sample_scales", sample_scales),
            ("sample_decode_scale", sample_decode_scale),
            ("sample_output", sample_output),
        ):
            _require_torch_sample(tensor, name=name)
        self.packed_desc = self._make_tensor_desc(
            sample_packed,
            name="packed",
            interpret_uint8_as_fp4x2=True,
            canonical=True,
        )
        self.scales_desc = self._make_tensor_desc(sample_scales, name="scales", canonical=True)
        self.decode_desc = self._make_tensor_desc(sample_decode_scale, name="decode_scale", canonical=True)
        self.output_desc = self._make_tensor_desc(sample_output, name="output", canonical=True)
        self.k = self.output_desc.shape[-1]

    def check_support(self) -> bool:
        self._check_dtype(self.packed_desc, [cutlass.Float4E2M1FN, cutlass.Uint8], "packed")
        self._check_dtype(self.scales_desc, cutlass.Float8E4M3FN, "scales")
        self._check_dtype(self.decode_desc, cutlass.Float32, "decode_scale")
        self._check_dtype(self.output_desc, cutlass.BFloat16, "output")
        m, k = _check_problem_shape(self, self.output_desc.shape)
        self._check_tensor_shape(self.packed_desc, (1, m, k), "packed")
        self._check_tensor_shape(self.scales_desc, (1, m, k // BLOCK_SIZE), "scales")
        self._check_tensor_shape(self.decode_desc, (1, 1, 1), "decode_scale")
        for name, desc in (
            ("packed", self.packed_desc),
            ("scales", self.scales_desc),
            ("decode_scale", self.decode_desc),
            ("output", self.output_desc),
        ):
            _check_desc_contiguous(self, name, desc)
        _check_single_cuda_device(self, self.packed_desc, self.scales_desc, self.decode_desc, self.output_desc)
        _check_sm100_or_newer(self, self.output_desc.device)
        self._is_supported = True
        return True

    def compile(self) -> None:
        import torch

        self._ensure_support_checked()
        if self._compiled_kernel is None:
            with torch.cuda.device(self.output_desc.device):
                self._compiled_kernel = compile_nvfp4_block_scale_dequantize(k=self.k)

    def execute(
        self,
        packed_tensor,
        scale_tensor,
        decode_scale,
        output_tensor,
        current_stream: cuda.CUstream | None = None,
    ) -> None:
        import torch

        self._runtime_error_if(self._compiled_kernel is None, "Nvfp4BlockScaleDequantizer is not compiled")
        output_shape, _, _, device = _runtime_tensor_meta(output_tensor, name="output")
        m, k = _check_problem_shape(self, output_shape)
        self._value_error_if(k != self.k, f"runtime K must equal compiled K={self.k}; got {k}")
        self._value_error_if(device != self.output_desc.device, f"runtime device must equal compiled device {self.output_desc.device}; got {device}")
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        packed_dtypes = tuple(dtype for dtype in (fp4_dtype, torch.uint8) if dtype is not None)
        _require_runtime_tensor(packed_tensor, name="packed", shape=(1, m, k // 2), dtypes=packed_dtypes, device=device, alignment=16)
        _require_runtime_tensor(
            scale_tensor,
            name="scales",
            shape=(1, m, k // BLOCK_SIZE),
            dtypes=(torch.float8_e4m3fn,),
            device=device,
            alignment=16,
        )
        _require_runtime_tensor(decode_scale, name="decode_scale", shape=(1, 1, 1), dtypes=(torch.float32,), device=device, alignment=4)
        _require_runtime_tensor(output_tensor, name="output", shape=(1, m, k), dtypes=(torch.bfloat16,), device=device, alignment=16)
        with torch.cuda.device(device):
            self._compiled_kernel(
                _packed_carrier(packed_tensor),
                scale_tensor,
                decode_scale,
                output_tensor,
                _as_custream(current_stream, device),
            )


_QUANTIZER_CACHE: dict[tuple[int, int], Nvfp4BlockScaleQuantizer] = {}
_DEQUANTIZER_CACHE: dict[tuple[int, int], Nvfp4BlockScaleDequantizer] = {}
_CACHE_LOCK = threading.Lock()


def _allocate_packed(shape, *, device):
    import torch

    carrier = torch.empty(shape, device=device, dtype=torch.uint8)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return carrier.view(fp4_dtype) if fp4_dtype is not None else carrier


def nvfp4_block_scale_quantize(input_tensor, encode_scale) -> TupleDict:
    """Quantize contiguous BF16 ``[1,M,K]`` into reusable NVFP4 tensors.

    ``encode_scale`` is a CUDA FP32 tensor of shape ``[1,1,1]``.  The returned
    ``packed_tensor`` has physical shape ``[1,M,K/2]`` and the returned E4M3
    ``scale_tensor`` has shape ``[1,M,K/16]`` in F8_128x4 physical order.
    """

    import torch

    if detect_framework(input_tensor) != "torch":
        raise TypeError("nvfp4_block_scale_quantize currently accepts torch tensors")
    shape = get_shape(input_tensor)
    if len(shape) != 3:
        raise ValueError(f"input must have shape [1,M,K]; got {shape}")
    batch, m, k = shape
    if batch != 1 or m <= 0 or m % 128 or k <= 0 or k % 64 or k // BLOCK_SIZE > MAX_GROUPS_PER_ROW:
        raise ValueError(
            "input must have shape [1,M,K] with positive M divisible by 128 and "
            f"positive K divisible by 64 with K/{BLOCK_SIZE} <= {MAX_GROUPS_PER_ROW}; got {shape}"
        )
    device = get_device(input_tensor)
    packed_tensor = _allocate_packed((1, m, k // 2), device=device)
    scale_tensor = torch.empty((1, m, k // BLOCK_SIZE), device=device, dtype=torch.float8_e4m3fn)
    key = (device.index, k)
    with torch.cuda.device(device):
        with _CACHE_LOCK:
            plan = _QUANTIZER_CACHE.get(key)
            if plan is None:
                plan = Nvfp4BlockScaleQuantizer(input_tensor, encode_scale, packed_tensor, scale_tensor)
                plan.check_support()
                plan.compile()
                _QUANTIZER_CACHE[key] = plan
        plan.execute(input_tensor, encode_scale, packed_tensor, scale_tensor)
    return TupleDict(packed_tensor=packed_tensor, scale_tensor=scale_tensor)


def nvfp4_block_scale_dequantize(packed_tensor, scale_tensor, decode_scale):
    """Dequantize reusable NVFP4 tensors to contiguous BF16 ``[1,M,K]``.

    ``decode_scale`` is multiplied after the per-block E4M3 scale.  Pass a CUDA
    FP32 ``[1,1,1]`` tensor containing one when no global rescale is required.
    """

    import torch

    if detect_framework(packed_tensor) != "torch":
        raise TypeError("nvfp4_block_scale_dequantize currently accepts torch tensors")
    packed_shape = get_shape(packed_tensor)
    if len(packed_shape) != 3:
        raise ValueError(f"packed_tensor must have physical shape [1,M,K/2]; got {packed_shape}")
    batch, m, packed_k = packed_shape
    k = packed_k * 2
    if batch != 1 or m <= 0 or m % 128 or k <= 0 or k % 64 or k // BLOCK_SIZE > MAX_GROUPS_PER_ROW:
        raise ValueError(
            "packed_tensor must have physical shape [1,M,K/2] with positive M divisible by 128 and "
            f"positive K divisible by 64 with K/{BLOCK_SIZE} <= {MAX_GROUPS_PER_ROW}; got {packed_shape}"
        )
    device = get_device(packed_tensor)
    output_tensor = torch.empty((batch, m, k), device=device, dtype=torch.bfloat16)
    key = (device.index, k)
    with torch.cuda.device(device):
        with _CACHE_LOCK:
            plan = _DEQUANTIZER_CACHE.get(key)
            if plan is None:
                plan = Nvfp4BlockScaleDequantizer(packed_tensor, scale_tensor, decode_scale, output_tensor)
                plan.check_support()
                plan.compile()
                _DEQUANTIZER_CACHE[key] = plan
        plan.execute(packed_tensor, scale_tensor, decode_scale, output_tensor)
    return output_tensor


__all__ = [
    "Nvfp4BlockScaleDequantizer",
    "Nvfp4BlockScaleQuantizer",
    "nvfp4_block_scale_dequantize",
    "nvfp4_block_scale_quantize",
]
