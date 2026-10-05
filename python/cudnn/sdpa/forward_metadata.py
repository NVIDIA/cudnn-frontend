# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize forward lengths and sinks to the shared graph declarations."""

from contextlib import nullcontext
from functools import lru_cache

import torch

from cudnn._device import ensure_current_context
from cudnn._torch_stream import _raw_current_stream
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old

_DTYPE_NAMES = {dtype: str(dtype) for dtype in (torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32)}


@lru_cache(maxsize=16)
def _can_prepare(device_index, installed, version):
    return installed and not cutedsl_too_old(version) and not cutedsl_arch_requirement_error(torch.cuda.get_device_capability(device_index))


@lru_cache(maxsize=128)
def _plan(dtypes, device_index):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.forward_metadata import compile_metadata

    major, minor = torch.cuda.get_device_capability(device_index)
    artifact = compile_metadata(dtypes, device_index, f"sm_{major}{minor}")
    entry = positional_entry(artifact)
    # A foreign target may compile without a runnable entry in some DSL
    # versions. Preserve Torch preparation for that unsupported environment.
    return None if entry is None else (artifact, entry)


def _torch_column(tensor, dtype, shape):
    result = tensor.to(dtype).contiguous()
    if result.data_ptr() % 16:
        result = result.clone(memory_format=torch.contiguous_format)
    return result.reshape(shape)


def prepare_forward_metadata(seq_q, seq_kv, sinks, batch, heads):
    """Return compact, aligned graph operands without retaining runtime storage.

    Native compact inputs remain views. Other common CUDA inputs share one
    conversion launch; the Torch fallback preserves the same layout contract.
    """
    q_native = seq_q is None or (seq_q.dtype == torch.int32 and seq_q.is_contiguous() and seq_q.data_ptr() % 16 == 0)
    kv_native = seq_kv is None or (seq_kv.dtype == torch.int32 and seq_kv.is_contiguous() and seq_kv.data_ptr() % 16 == 0)
    sink_native = sinks is None or (sinks.dtype == torch.float32 and sinks.is_contiguous() and sinks.data_ptr() % 16 == 0)
    if q_native and kv_native and sink_native:
        # reshape checks the element count without a separate metadata query.
        return (
            seq_q.reshape(batch, 1, 1, 1) if seq_q is not None else None,
            seq_kv.reshape(batch, 1, 1, 1) if seq_kv is not None else None,
            sinks.reshape(1, heads, 1, 1) if sinks is not None else None,
        )

    # A compact half sink alone already needs just one cast. Keep that cheap
    # Torch path: a dtype-changing copy is compact and freshly aligned.
    if seq_q is None and seq_kv is None and sinks.dtype in (torch.float16, torch.bfloat16) and sinks.is_contiguous():
        return None, None, sinks.to(torch.float32).reshape(1, heads, 1, 1)

    values = (seq_q, seq_kv, sinks)
    counts = (batch, batch, heads)
    shapes = ((batch, 1, 1, 1), (batch, 1, 1, 1), (1, heads, 1, 1))
    types = (torch.int32, torch.int32, torch.float32)
    pending = (not q_native, not kv_native, not sink_native)
    for tensor, count in zip(values, counts):
        if tensor is not None and tensor.numel() != count:
            raise ValueError(f"SDPA forward metadata needs {count} elements; got {tensor.numel()}")

    device = next(tensor.device for tensor in values if tensor is not None)
    installed, version = cutedsl_state()
    allowed = ((torch.int32, torch.int64), (torch.int32, torch.int64), (torch.float16, torch.bfloat16, torch.float32))
    supported = (
        device.type == "cuda"
        and _can_prepare(device.index, installed, version)
        and all(not copy or (tensor.device == device and tensor.ndim == 1 and tensor.dtype in dtypes) for tensor, copy, dtypes in zip(values, pending, allowed))
    )
    if not supported:
        return tuple(_torch_column(tensor, dtype, shape) if tensor is not None else None for tensor, dtype, shape in zip(values, types, shapes))

    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        active_counts = (0 if q_native else batch, 0 if kv_native else batch, 0 if sink_native else heads)
        if max(active_counts) == 0:
            return tuple(_torch_column(tensor, dtype, shape) if tensor is not None else None for tensor, dtype, shape in zip(values, types, shapes))
        stream = _raw_current_stream(torch, device)
        if stream is None:
            stream = torch.cuda.current_stream(device).cuda_stream
        ensure_current_context(stream, device.index)
        dtypes = (
            None if q_native else _DTYPE_NAMES[seq_q.dtype],
            None if kv_native else _DTYPE_NAMES[seq_kv.dtype],
            None if sink_native else _DTYPE_NAMES[sinks.dtype],
        )
        plan = _plan(dtypes, device.index)
        if plan is None:
            return tuple(_torch_column(tensor, dtype, shape) if tensor is not None else None for tensor, dtype, shape in zip(values, types, shapes))
        out_q = (seq_q.reshape(shapes[0]) if seq_q is not None else None) if q_native else torch.empty(shapes[0], dtype=torch.int32, device=device)
        out_kv = (seq_kv.reshape(shapes[1]) if seq_kv is not None else None) if kv_native else torch.empty(shapes[1], dtype=torch.int32, device=device)
        out_sink = (sinks.reshape(shapes[2]) if sinks is not None else None) if sink_native else torch.empty(shapes[2], dtype=torch.float32, device=device)
        plan[1](
            None if q_native else seq_q.data_ptr(),
            None if kv_native else seq_kv.data_ptr(),
            None if sink_native else sinks.data_ptr(),
            None if q_native else out_q.data_ptr(),
            None if kv_native else out_kv.data_ptr(),
            None if sink_native else out_sink.data_ptr(),
            active_counts,
            (0 if q_native else seq_q.stride(0), 0 if kv_native else seq_kv.stride(0), 0 if sink_native else sinks.stride(0)),
            stream,
        )
        return out_q, out_kv, out_sink
