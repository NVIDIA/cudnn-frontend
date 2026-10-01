# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A raw CUDA stream handle as the torch stream torch work must run on (Hard Rule 5).

Every torch-facing engine receives the launch stream as a driver handle (the
execute-time handle's stream). Turning it into a torch stream has one trap: the
default-stream sentinels ``0`` (cudaStream_t null), ``1`` (cudaStreamLegacy) and
``2`` (cudaStreamPerThread) must map to ``torch.cuda.default_stream``, never to
``torch.cuda.ExternalStream`` -- torch before pytorch/pytorch#183258 (v2.13.0)
returns a fresh NON-BLOCKING pool stream for ``ExternalStream(0)``, so torch work
issued in it never orders against a kernel launched on ``CUstream(0)``. Thirteen
call sites carried that guard as local copies and the fourteenth missed it
(cudnn-frontend #1165); this module is the one copy.

torch is imported lazily: ``python/cudnn`` must import without it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, Optional

# cudaStream_t 0, cudaStreamLegacy and cudaStreamPerThread. torch's default
# stream is the legacy one and every blocking stream orders against it.
DEFAULT_STREAM_HANDLES = frozenset({0, 1, 2})

_RAW_CURRENT: Any = None  # torch._C._cuda_getCurrentRawStream, resolved once
_GET_DEVICE: Any = None  # torch._C._cuda_getDevice, resolved once
_STREAMS: dict = {}  # (handle, device index) -> torch stream; the objects are non-owning handle wrappers
_DEFAULT_RAW: dict = {}  # device index -> torch's default-stream handle


def _current_device(torch) -> int:
    """The current device via the private getter -- ``torch.cuda.current_device()``
    adds ~0.3 us of Python lazy-init checks per call."""
    global _GET_DEVICE
    if _GET_DEVICE is None:
        _GET_DEVICE = getattr(torch._C, "_cuda_getDevice", None) or torch.cuda.current_device
    return _GET_DEVICE()


def _device_index(torch, device) -> int:
    if device is None:
        return _current_device(torch)
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else _current_device(torch)


def _raw_current_stream(torch, device) -> Optional[int]:
    """torch's current stream handle on ``device`` via the private raw getter
    (~0.07 us) -- ``torch.cuda.current_stream()`` builds a Stream object (~3.4 us)."""
    global _RAW_CURRENT
    if _RAW_CURRENT is None:
        _RAW_CURRENT = getattr(torch._C, "_cuda_getCurrentRawStream", False)
    if not _RAW_CURRENT:
        return None
    return int(_RAW_CURRENT(_device_index(torch, device)))


def as_torch_stream(stream, device=None):
    """The ``torch.cuda.Stream`` for ``stream``: a driver ``CUstream``, an int handle,
    or already a torch stream.

    Sentinels and torch's own default stream -> ``torch.cuda.default_stream(device)``;
    torch's current stream -> itself; only a genuine side stream is wrapped in
    ``torch.cuda.ExternalStream(handle, device=device)``.
    """
    import torch

    if isinstance(stream, torch.cuda.Stream):
        if device is not None and stream.device != torch.device("cuda", _device_index(torch, device)):
            raise ValueError(f"stream must be on cuda:{_device_index(torch, device)}, got {stream.device}")
        return stream
    handle = int(stream)
    index = _device_index(torch, device)
    resolved = _STREAMS.get((handle, index))
    if resolved is not None:
        return resolved
    default = torch.cuda.default_stream(index)
    if handle in DEFAULT_STREAM_HANDLES or handle == default.cuda_stream:
        resolved = default
    else:
        current = torch.cuda.current_stream(index)
        resolved = current if handle == current.cuda_stream else torch.cuda.ExternalStream(handle, device=index)
    if len(_STREAMS) >= 256:
        _STREAMS.clear()
    _STREAMS[(handle, index)] = resolved
    return resolved


def launch_stream(stream, device=None):
    """The torch stream a launch on ``stream`` runs on: ``as_torch_stream(stream, device)``,
    or torch's current stream on ``device`` when ``stream`` is None."""
    import torch

    index = _device_index(torch, device)
    if stream is None:
        stream = _raw_current_stream(torch, index)
        if stream is None:
            return torch.cuda.current_stream(index)
    return as_torch_stream(stream, index)


def device_context(device):
    """``torch.cuda.device(device)``, or a no-op when ``device`` is already current."""
    import torch

    index = device if isinstance(device, int) else device.index
    if index is None or index == _current_device(torch):
        return nullcontext()
    return torch.cuda.device(index)


def stream_context(stream, device=None, *, verify_current: bool = False) -> ContextManager[None]:
    """``torch.cuda.stream(as_torch_stream(stream, device))``; a no-op when ``stream``
    is None, or when ``device`` is the current device and ``stream`` is already
    torch's current stream there.

    ``verify_current=False`` (default) takes the raw-handle fast path for the
    common case where the launch stream is the one torch is already on.
    """
    if stream is None:
        return nullcontext()
    import torch

    is_stream = isinstance(stream, torch.cuda.Stream)
    handle = stream.cuda_stream if is_stream else int(stream)
    if not verify_current:
        current = _current_device(torch)
        index = current if device is None else _device_index(torch, device)
        # Default-stream handles are equal on every device, so the no-op also needs the
        # requested device current and a Stream that lives there; otherwise the context below
        # switches the device or rejects the pair.
        raw = _raw_current_stream(torch, index) if index == current and (not is_stream or stream.device_index == index) else None
        if raw is not None:
            if handle in DEFAULT_STREAM_HANDLES:
                default_raw = _DEFAULT_RAW.get(index)
                if default_raw is None:
                    default_raw = _DEFAULT_RAW.setdefault(index, torch.cuda.default_stream(index).cuda_stream)
                handle = default_raw
            if handle == raw:
                return nullcontext()
    return torch.cuda.stream(as_torch_stream(stream, device))


def record_streams(tensors, stream, device=None) -> None:
    """Order the caching allocator's reuse of each tensor's block behind the work already
    enqueued on ``stream`` (R1): ``tensor.record_stream(as_torch_stream(stream, device))``
    for every CUDA tensor in ``tensors`` (``None`` entries skipped).

    A no-op only when ``stream`` is None (the work runs on torch's current stream under the
    caller's own context). A launch stream that happens to be torch's current stream is still
    recorded: the allocator orders reuse against the tensor's ALLOCATION stream, which may be
    another one when the caller entered a side-stream context before calling.
    """
    if stream is None:
        return
    import torch

    consumer = None
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        if consumer is None:
            consumer = as_torch_stream(stream, device if device is not None else tensor.device)
        tensor.record_stream(consumer)


def contiguous_on_stream(tensor, stream, device=None):
    """``tensor`` itself when it is None or contiguous; otherwise a contiguous copy made on
    ``stream`` after ``record_streams((tensor,), stream, device)``.

    The recording is what makes the copy safe to rebind over: a wrapper doing
    ``t = t.contiguous()`` under ``stream_context(side)`` drops the caller's reference while
    the copy kernel is still queued on ``side``; if the caller also releases ``t``, its block
    returns to the allocator's pool for the caller's stream and the next same-size allocation
    there overwrites what the copy has yet to read.
    """
    if tensor is None or tensor.is_contiguous():
        return tensor
    record_streams((tensor,), stream, device)
    with stream_context(stream, device):
        return tensor.contiguous()


def copy_into_on_stream(dst, src, stream, device=None) -> None:
    """``dst.copy_(src)`` on ``stream`` with both tensors recorded there first: the wrapper copy-back
    into a caller-owned buffer is asynchronous too, so neither a ``dst`` the caller releases right
    after the call nor a ``src`` allocated on another stream may be reused under the pending copy."""
    record_streams((src, dst), stream, device)
    with stream_context(stream, device):
        dst.copy_(src)
