# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize forward lengths and sinks to the shared graph declarations."""

from contextlib import nullcontext
from functools import lru_cache

import torch

from cudnn._device import ensure_current_context
from cudnn._torch_stream import _raw_current_stream
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old


@lru_cache(maxsize=128)
def _plan(dtypes, device_index):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.forward_metadata import compile_metadata

    major, minor = torch.cuda.get_device_capability(device_index)
    artifact = compile_metadata(dtypes, device_index, f"sm_{major}{minor}")
    entry = positional_entry(artifact)
    if entry is None:
        raise NotImplementedError("SDPA forward metadata requires a positional tvm-ffi entry")
    return artifact, entry


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
    values = (seq_q, seq_kv, sinks)
    counts = (batch, batch, heads)
    shapes = ((batch, 1, 1, 1), (batch, 1, 1, 1), (1, heads, 1, 1))
    types = (torch.int32, torch.int32, torch.float32)
    outputs, pending = [], []
    for tensor, count, shape, dtype in zip(values, counts, shapes, types):
        if tensor is not None and tensor.numel() != count:
            raise ValueError(f"SDPA forward metadata needs {count} elements; got {tensor.numel()}")
        native = tensor is None or (tensor.dtype == dtype and tensor.is_contiguous() and tensor.data_ptr() % 16 == 0)
        outputs.append(None if tensor is None or not native else tensor.reshape(shape))
        pending.append(not native)
    if not any(pending):
        return tuple(outputs)

    device = next(tensor.device for tensor in values if tensor is not None)
    installed, version = cutedsl_state()
    allowed = ((torch.int32, torch.int64), (torch.int32, torch.int64), (torch.float16, torch.bfloat16, torch.float32))
    supported = (
        installed
        and not cutedsl_too_old(version)
        and device.type == "cuda"
        and all(not copy or (tensor.device == device and tensor.ndim == 1 and tensor.dtype in dtypes) for tensor, copy, dtypes in zip(values, pending, allowed))
    )
    if not supported or cutedsl_arch_requirement_error(torch.cuda.get_device_capability(device)):
        return tuple(
            _torch_column(tensor, dtype, shape) if copy else output for tensor, dtype, shape, copy, output in zip(values, types, shapes, pending, outputs)
        )

    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        if pending[0] or pending[1]:
            pitch = (batch + 3) // 4 * 4
            owner = torch.empty((2, pitch), dtype=torch.int32, device=device)
            columns = owner.as_strided((2, batch, 1, 1, 1), (pitch, 1, 1, 1, 1)).unbind(0)
            for index in range(2):
                if pending[index]:
                    outputs[index] = columns[index]
        if pending[2]:
            outputs[2] = torch.empty(shapes[2], dtype=torch.float32, device=device)
        active_counts = tuple(count if copy else 0 for count, copy in zip(counts, pending))
        if max(active_counts) == 0:
            return tuple(outputs)
        stream = _raw_current_stream(torch, device)
        if stream is None:
            stream = torch.cuda.current_stream(device).cuda_stream
        ensure_current_context(stream, device.index)
        dtypes = tuple(str(tensor.dtype) if copy else None for tensor, copy in zip(values, pending))
        plan = _plan(dtypes, device.index)
        plan[1](
            *(tensor.data_ptr() if copy else None for tensor, copy in zip(values, pending)),
            *(tensor.data_ptr() if copy else None for tensor, copy in zip(outputs, pending)),
            active_counts,
            tuple(tensor.stride(0) if copy else 0 for tensor, copy in zip(values, pending)),
            stream,
        )
        return tuple(outputs)
