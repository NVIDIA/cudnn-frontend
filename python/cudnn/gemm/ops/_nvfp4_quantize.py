# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-private SM100 BF16-to-NVFP4 activation quantization.

This module intentionally is not re-exported from :mod:`cudnn.gemm`.  Its
packed E2M1 and F8_128x4 scale-factor layouts are an internal bridge for the
Qwen-Image low-precision benchmark, not a stable public API.
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Optional, Tuple

import torch

_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()
_ONES_CACHE = {}
_ONES_LOCK = threading.Lock()
_INT32_MAX = (1 << 31) - 1


def _scale_factor_shape(m: int, k: int) -> Tuple[int, int]:
    """Physical byte shape of a row-major logical ``[M, K/16]`` F8_128x4 tensor."""
    return ((m + 127) // 128 * 128, (k // 16 + 3) // 4 * 4)


def _load_extension():
    """Build once per process; cpp_extension also caches the binary on disk."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    with _EXTENSION_LOCK:
        if _EXTENSION is None:
            # Keep cpp_extension and nvcc completely off the import path.  In
            # particular, unsupported devices fail validation before reaching
            # this function.
            from torch.utils.cpp_extension import load

            source_dir = Path(__file__).with_name("csrc")
            _EXTENSION = load(
                name="_cudnn_fe_nvfp4_quantize_sm100_f212ec82",
                sources=[str(source_dir / "nvfp4_quantize_sm100.cu")],
                extra_include_paths=[str(source_dir)],
                extra_cuda_cflags=[
                    "-O3",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "-DFLASHINFER_ENABLE_FP8_E8M0",
                    "-DFLASHINFER_ENABLE_FP4_E2M1",
                ],
                with_cuda=True,
                verbose=os.environ.get("CUDNN_NVFP4_BUILD_VERBOSE", "0") == "1",
            )
    return _EXTENSION


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    shape: Tuple[int, ...],
    alignment: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")


def _ones_pre_quant_scale(device: torch.device, k: int, stream: torch.cuda.Stream) -> torch.Tensor:
    # Initialization and first consumption are ordered on the same stream.  A
    # stream is part of the key so another stream cannot observe an unfinished
    # asynchronous fill from the first call.
    key = (device.index, stream.cuda_stream, k)
    value = _ONES_CACHE.get(key)
    if value is not None:
        return value
    with _ONES_LOCK:
        value = _ONES_CACHE.get(key)
        if value is None:
            value = torch.ones(k, dtype=torch.bfloat16, device=device)
            _ONES_CACHE[key] = value
    return value


def nvfp4_quantize(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    pre_quant_scale: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    scale_factors: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous BF16 matrix into ModelOpt-style NVFP4 buffers.

    ``x`` has shape ``[M, K]`` and ``pre_quant_scale`` has shape ``[K]``.
    The kernel first performs the BF16 multiply ``x * pre_quant_scale`` and
    then block-quantizes each 16-value K block.  ``global_scale`` is a
    same-device contiguous one-element FP32 tensor, conventionally
    ``448 * 6 / amax``.

    Returns packed E2M1 bytes ``[M, K/2]`` and F8_128x4 scale-factor bytes
    ``[ceil(M/128)*128, ceil((K/16)/4)*4]``.  Callers may provide either output
    buffer.  Execution is asynchronous on the current PyTorch stream.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"x must have dtype {torch.bfloat16}, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"x must have rank 2, got rank {x.ndim}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    m, k = x.shape
    if m <= 0 or k <= 0:
        raise ValueError(f"x dimensions must be positive, got {(m, k)}")
    if k % 16:
        raise ValueError(f"x.shape[1] must be divisible by 16, got {k}")
    if m > _INT32_MAX or k > _INT32_MAX:
        raise ValueError(f"x dimensions must fit signed 32-bit integers, got {(m, k)}")
    if x.data_ptr() % 16:
        raise ValueError("x data pointer must be 16-byte aligned")

    device = x.device
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(f"nvfp4_quantize requires SM100, got SM{capability[0]}{capability[1]}")
    if type(enable_pdl) is not bool:
        raise TypeError(f"enable_pdl must be bool, got {type(enable_pdl).__name__}")

    if not isinstance(global_scale, torch.Tensor):
        raise TypeError(f"global_scale must be a torch.Tensor, got {type(global_scale).__name__}")
    if global_scale.device != device:
        raise ValueError(f"global_scale must be on {device}, got {global_scale.device}")
    if global_scale.dtype is not torch.float32:
        raise TypeError(f"global_scale must have dtype {torch.float32}, got {global_scale.dtype}")
    if tuple(global_scale.shape) not in ((), (1,)):
        raise ValueError("global_scale must have shape () or (1,), " f"got shape {tuple(global_scale.shape)}")
    if not global_scale.is_contiguous():
        raise ValueError("global_scale must be contiguous")
    if global_scale.data_ptr() % 4:
        raise ValueError("global_scale data pointer must be 4-byte aligned")

    output_shape = (m, k // 2)
    sf_shape = _scale_factor_shape(m, k)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        if pre_quant_scale is None:
            pre_quant_scale = _ones_pre_quant_scale(device, k, stream)
        else:
            _check_tensor(
                pre_quant_scale,
                name="pre_quant_scale",
                dtype=torch.bfloat16,
                device=device,
                shape=(k,),
                alignment=16,
            )

        if out is None:
            out = torch.empty(output_shape, dtype=torch.uint8, device=device)
        else:
            _check_tensor(
                out,
                name="out",
                dtype=torch.uint8,
                device=device,
                shape=output_shape,
                alignment=8,
            )

        if scale_factors is None:
            scale_factors = torch.empty(sf_shape, dtype=torch.uint8, device=device)
        else:
            _check_tensor(
                scale_factors,
                name="scale_factors",
                dtype=torch.uint8,
                device=device,
                shape=sf_shape,
                alignment=4,
            )

        extension = _load_extension()
        multiprocessor_count = torch.cuda.get_device_properties(device).multi_processor_count
        extension.launch(
            x.data_ptr(),
            pre_quant_scale.data_ptr(),
            global_scale.data_ptr(),
            out.data_ptr(),
            scale_factors.data_ptr(),
            m,
            k,
            multiprocessor_count,
            stream.cuda_stream,
            enable_pdl,
        )
        # The binding receives raw addresses.  Explicitly tell the caching
        # allocator that every backing allocation remains in use until work on
        # the launch stream has completed.
        for tensor in (x, pre_quant_scale, global_scale, out, scale_factors):
            tensor.record_stream(stream)
        return out, scale_factors


__all__ = ["nvfp4_quantize"]
