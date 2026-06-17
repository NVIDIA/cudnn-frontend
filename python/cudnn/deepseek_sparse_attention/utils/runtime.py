"""Runtime helpers shared by DSA Python wrappers."""

from contextlib import nullcontext
from functools import lru_cache
from typing import Optional

import torch
import cuda.bindings.driver as cuda


@lru_cache(maxsize=None)
def device_major() -> int:
    return torch.cuda.get_device_capability()[0]


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def validate_q_causal_offsets(q_causal_offsets, batch: int, device: torch.device):
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
    return q_causal_offsets if q_causal_offsets.is_contiguous() else q_causal_offsets.contiguous()


def resolve_stream(current_stream: Optional[cuda.CUstream] = None) -> cuda.CUstream:
    if current_stream is not None:
        return current_stream
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def torch_stream_context(current_stream: Optional[cuda.CUstream] = None):
    if current_stream is None:
        return nullcontext()
    return torch.cuda.stream(torch.cuda.ExternalStream(int(current_stream)))
