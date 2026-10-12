# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared in-place tail RoPE plus microscaled QDQ on SM100."""

from functools import lru_cache
import math

import torch

from cudnn.api_base import APIBase, TupleDict


@lru_cache(maxsize=1)
def _kernel():
    import triton
    from packaging.version import Version

    if Version(triton.__version__) < Version("3.7.0"):
        raise RuntimeError(f"FROST RoPE QDQ requires Triton >=3.7.0; installed {triton.__version__}")
    from .frost.kernels import cudnn_frost_rope_qdq_inplace

    return cudnn_frost_rope_qdq_inplace


@lru_cache(maxsize=1)
def _compressed_kernel():
    _kernel()  # Check the Triton version before importing either kernel module.
    from .frost.compressed_kv import cudnn_frost_compkv_rope_qdq

    return cudnn_frost_compkv_rope_qdq


@lru_cache(maxsize=64)
def _compile(quantization, heads, position_dtype, device_index, capability, group_size=32, scale_format="ue8m0"):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    compressed = group_size == 16
    kernel = _compressed_kernel() if compressed else _kernel()
    rows = 8 if quantization == "fp4" and not compressed else 2
    artifacts = {}
    # Explicit signatures avoid MockTensor's optimistic alignment/range and
    # JIT specialization of sample token counts. N_ROWS has a signed i64 ABI.
    with torch.cuda.device(device_index):
        for alignment in (16, 4, 2):
            constants = (
                dict(ROWS=rows, PACKED_IO=alignment >= 4) if compressed else dict(HEADS=heads, FP4=quantization == "fp4", ROWS=rows, PACKED_IO=alignment >= 4)
            )
            signature = dict(X="*bf16", CACHE="*fp32", POSITIONS="*i32" if position_dtype == torch.int32 else "*i64", N_ROWS="i64")
            signature.update({key: "constexpr" for key in constants})
            attrs = {
                (0,): [["tt.divisibility", alignment]],
                (1,): [["tt.divisibility", 16 if alignment == 16 else 4]],
                (2,): [["tt.divisibility", 16 if alignment == 16 else (4 if position_dtype == torch.int32 else 8)]],
            }
            compiled = triton.compile(
                ASTSource(kernel, signature, constexprs=constants, attrs=attrs),
                target=GPUTarget("cuda", capability[0] * 10 + capability[1], 32),
                options=dict(num_warps=4, enable_fp_fusion=False),
            )
            # Resolve the device module and launcher at compile time. Execute
            # can choose a new grid without compiling or loading any binary.
            compiled[(1, 1, 1)]
            artifacts[alignment] = (compiled, tuple(constants.values()))
    return rows, artifacts


def _device(device):
    result = torch.device(device.type, device.index)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


class RopeQDQInplace(APIBase):
    """Rotate the last 64 channels, then quantize/dequantize all channels.

    Group32/UE8M0 FP4 accepts BF16 [N,H,128], H in {1,4,32}.
    Group32/UE8M0 FP8 and group16/E4M3 FP4 accept BF16 [...,512].
    The output aliases x. Cache is FP32 [P,64] (32 cosines then 32 sines).
    Position IDs are contiguous int32/int64, one per token for [N,H,128]
    or per flattened row for [...,512]. There is no global scale.
    Token counts, cache length and pointer alignment may change after compile.
    """

    def __init__(self, x, cache, positions, *, quantization, backend="frost", group_size=32, scale_format="ue8m0"):
        super().__init__()
        self._warn_experimental_api()
        self._descs = {name: self._make_tensor_desc(tensor, name=name) for name, tensor in dict(x=x, cache=cache, positions=positions).items()}
        self._quantization, self._backend = quantization, backend
        self._group_size, self._scale_format = group_size, scale_format
        self._device = _device(self._descs["x"].device)
        self._heads = self._descs["x"].shape[1] if quantization == "fp4" and group_size == 32 and self._descs["x"].ndim == 3 else 1
        self._position_dtype = self._descs["positions"].dtype

    def _validate_layouts(self, tensors):
        x, cache, positions = (tensors[name] for name in ("x", "cache", "positions"))
        for name, tensor in tensors.items():
            if _device(tensor.device) != self._device or not tensor.is_contiguous():
                strides = tensor.stride() if callable(tensor.stride) else tensor.stride
                raise NotImplementedError(f"{name} must be contiguous on {self._device}; strides={strides}")
        if x.dtype != torch.bfloat16:
            raise ValueError("x must be BF16")
        if self._quantization == "fp4" and self._group_size == 32:
            if x.ndim != 3 or x.shape[1:] != (self._heads, 128) or self._heads not in (1, 4, 32):
                raise ValueError("FP4 x must be [N,H,128] with the planned H in {1,4,32}")
            tokens = x.shape[0]
        else:
            if x.ndim < 2 or x.shape[-1] != 512:
                raise ValueError("FP8 or group16 FP4 x must be [...,512] with at least two dimensions")
            tokens = math.prod(x.shape[:-1])
        rows = tokens * self._heads
        tile_rows = 8 if self._quantization == "fp4" and self._group_size == 32 else 2
        if rows <= 0 or rows > tile_rows * (2**31 - 1):
            raise ValueError("row count must be positive and fit the CUDA grid limit")
        if cache.dtype != torch.float32 or cache.ndim != 2 or cache.shape[1] != 64 or not 0 < cache.shape[0] < 2**55:
            raise ValueError("cache must be FP32 [P,64] with positive P and int64-addressable storage")
        if positions.dtype != self._position_dtype or positions.dtype not in (torch.int32, torch.int64) or positions.shape != (tokens,):
            raise ValueError("positions must match the planned int32/int64 dtype and contain one ID per token/row")
        return rows

    def check_support(self):
        if self._backend != "frost":
            raise ValueError("backend must be 'frost'")
        if self._quantization not in ("fp4", "fp8"):
            raise ValueError("quantization must be 'fp4' or 'fp8'")
        if type(self._group_size) is not int or (self._quantization, self._group_size, self._scale_format) not in (
            ("fp4", 32, "ue8m0"),
            ("fp8", 32, "ue8m0"),
            ("fp4", 16, "e4m3"),
        ):
            raise ValueError("supported (quantization, group_size, scale_format): (fp4,32,ue8m0), (fp8,32,ue8m0), (fp4,16,e4m3)")
        if self._device.type != "cuda":
            raise NotImplementedError("FROST RoPE QDQ requires CUDA SM100")
        self._validate_layouts(self._descs)
        self._capability = torch.cuda.get_device_capability(self._device)
        if self._capability != (10, 0):
            raise NotImplementedError("FROST RoPE QDQ currently supports SM100")
        _compressed_kernel() if self._group_size == 16 else _kernel()
        self._is_supported = True
        return True

    def compile(self):
        self._ensure_support_checked()
        self._compiled_kernel = _compile(
            self._quantization, self._heads, self._position_dtype, self._device.index, self._capability, self._group_size, self._scale_format
        )

    def scratch_workspace_bytes(self):
        self._ensure_support_checked()
        return 0

    def execute(self, x, cache, positions, *, current_stream=None):
        if self._compiled_kernel is None:
            raise RuntimeError("compile() must run before execute()")
        tensors = dict(x=x, cache=cache, positions=positions)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
            raise TypeError("execute arguments must be torch tensors")
        rows = self._validate_layouts(tensors)
        if any(tensor.requires_grad for tensor in tensors.values()):
            raise ValueError("in-place RoPE QDQ has no autograd/STE contract; detach inputs explicitly")
        start, end = x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()
        for name, tensor in (("cache", cache), ("positions", positions)):
            low, high = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
            if start < high and low < end:
                raise ValueError(f"x must not overlap {name}")
        alignment = 16 if all(tensor.data_ptr() % 16 == 0 for tensor in tensors.values()) else (4 if x.data_ptr() % 4 == 0 else 2)
        tile_rows, artifacts = self._compiled_kernel
        compiled, constants = artifacts[alignment]
        with torch.cuda.device(self._device):
            stream = torch.cuda.current_stream(self._device).cuda_stream if current_stream is None else int(current_stream)
            compiled[((rows + tile_rows - 1) // tile_rows, 1, 1)](x, cache, positions, rows, *constants, stream=stream)
        return TupleDict(out=x)


def rope_qdq_inplace(x, cache, positions, *, quantization, backend="frost", stream=None, group_size=32, scale_format="ue8m0"):
    """Convenience wrapper returning ``TupleDict(out=x)``; prewarm before capture."""
    op = RopeQDQInplace(x, cache, positions, quantization=quantization, backend=backend, group_size=group_size, scale_format=scale_format)
    op.compile()
    return op.execute(x, cache, positions, current_stream=stream)
