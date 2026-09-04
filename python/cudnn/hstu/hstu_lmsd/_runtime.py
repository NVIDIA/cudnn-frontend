# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for the HSTU LMSD APIs and wrappers."""

from __future__ import annotations

from contextlib import nullcontext

from cuda.bindings import driver as cuda
import torch


def tensor_signature(tensor: torch.Tensor, *, dynamic_rows: bool = False) -> tuple:
    """Return the plan-time tensor contract, optionally erasing runtime N."""
    shape = tuple(tensor.shape)
    if dynamic_rows:
        shape = (None, *shape[1:])
    return shape, tuple(tensor.stride()), tensor.dtype, tensor.device


def as_torch_stream(stream, device: torch.device) -> torch.cuda.Stream:
    """Return a PyTorch stream view for a torch stream or CUDA stream handle."""
    if isinstance(stream, torch.cuda.Stream):
        if stream.device != device:
            raise ValueError(f"stream must be on {device}, got {stream.device}")
        return stream
    if int(stream) == 0:
        return torch.cuda.default_stream(device)
    return torch.cuda.ExternalStream(int(stream), device=device)


def allocation_context(stream, device: torch.device):
    """Allocate on the launch stream when the caller supplies one."""
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(as_torch_stream(stream, device))


def stream_handle(
    stream: cuda.CUstream | torch.cuda.Stream | None,
    device: torch.device,
) -> cuda.CUstream:
    """Normalize an optional stream to the CUDA handle expected by CuTe DSL."""
    if stream is None:
        return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    if isinstance(stream, torch.cuda.Stream):
        as_torch_stream(stream, device)
        return cuda.CUstream(stream.cuda_stream)
    return stream


def record_streams(tensors, stream, device: torch.device) -> None:
    """Keep tensor storage alive on an explicitly selected launch stream."""
    if stream is None:
        return
    torch_stream = as_torch_stream(stream, device)
    for tensor in tensors:
        tensor.record_stream(torch_stream)
