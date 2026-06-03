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


def resolve_stream(current_stream: Optional[cuda.CUstream] = None) -> cuda.CUstream:
    if current_stream is not None:
        return current_stream
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def torch_stream_context(current_stream: Optional[cuda.CUstream] = None):
    if current_stream is None:
        return nullcontext()
    return torch.cuda.stream(torch.cuda.ExternalStream(int(current_stream)))
