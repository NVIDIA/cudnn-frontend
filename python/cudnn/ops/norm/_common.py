# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime support for graph-backed PyTorch normalization ops."""

import threading
from typing import Callable, Dict, Tuple

import cudnn
import torch

TORCH_DTYPE_TO_CUDNN = {
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float32: cudnn.data_type.FLOAT,
}

_tls = threading.local()


def get_handle(device: torch.device):
    """Return this thread's cuDNN handle for ``device`` on the current stream."""
    handles = getattr(_tls, "handles", None)
    if handles is None:
        handles = _tls.handles = {}
    if device not in handles:
        with torch.cuda.device(device):
            handles[device] = cudnn.create_handle()
    cudnn.set_stream(handle=handles[device], stream=torch.cuda.current_stream(device).cuda_stream)
    return handles[device]


class GraphCache:
    """Thread-safe, bounded cache for immutable built graphs."""

    def __init__(self, max_entries: int = 128):
        self._entries: Dict[tuple, Tuple[object, int]] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get_or_build(self, key: tuple, build: Callable[[], Tuple[object, int]]) -> Tuple[object, int]:
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._entries.get(key)
            if hit is None:
                hit = self._entries[key] = build()
                while len(self._entries) > self._max_entries:
                    self._entries.pop(next(iter(self._entries)))
        return hit


def require_cuda(name: str, **tensors: torch.Tensor) -> None:
    non_cuda = {tensor_name: str(tensor.device) for tensor_name, tensor in tensors.items() if not tensor.is_cuda}
    if non_cuda:
        raise ValueError(f"{name}: all tensors must be CUDA tensors; got {non_cuda}")


def require_same_device(name: str, reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    mismatched = {tensor_name: str(tensor.device) for tensor_name, tensor in tensors.items() if tensor.device != reference.device}
    if mismatched:
        raise ValueError(f"{name}: all tensors must be on {reference.device}; got {mismatched}")


def require_dtype(name: str, tensor: torch.Tensor, expected: torch.dtype) -> None:
    if tensor.dtype != expected:
        raise TypeError(f"{name}: expected dtype {expected}, got {tensor.dtype}")


def require_canonical_4d(name: str, tensor: torch.Tensor, rows: int, hidden_size: int) -> None:
    expected_shape = (rows, hidden_size, 1, 1)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name}: expected shape {expected_shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: expected a contiguous tensor")


def epsilon_tensor(eps: float) -> torch.Tensor:
    """Create the host pass-by-value tensor consumed by cuDNN."""
    return torch.tensor(eps, dtype=torch.float32).reshape(1, 1, 1, 1)
