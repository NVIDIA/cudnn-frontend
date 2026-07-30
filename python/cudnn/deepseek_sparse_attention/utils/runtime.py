# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by DSA Python wrappers."""

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, Optional

import torch
import cuda.bindings.driver as cuda


@lru_cache(maxsize=None)
def device_major() -> int:
    return torch.cuda.get_device_capability()[0]


def maybe_contiguous(
    x: torch.Tensor | None,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if x is None or x.stride(-1) == 1:
        return x
    with torch_stream_context(stream):
        return x.contiguous()


def validate_q_causal_offsets(
    q_causal_offsets: torch.Tensor | None,
    batch: int,
    device: torch.device,
    stream: Optional[cuda.CUstream] = None,
) -> torch.Tensor | None:
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != torch.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}")
    if not q_causal_offsets.is_cuda:
        raise ValueError("q_causal_offsets must be a CUDA tensor")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.is_contiguous():
        return q_causal_offsets
    with torch_stream_context(stream):
        return q_causal_offsets.contiguous()


def resolve_stream(current_stream: Optional[cuda.CUstream] = None) -> cuda.CUstream:
    if current_stream is not None:
        return current_stream
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@contextmanager
def torch_stream_context(current_stream: Optional[cuda.CUstream] = None) -> Iterator[None]:
    if current_stream is None:
        yield
        return
    with torch.cuda.stream(torch.cuda.get_stream_from_external(int(current_stream))):
        yield
