# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-independent length and offset preparation for the torch SDPA ops."""

from contextlib import nullcontext
from functools import lru_cache

import torch

from cudnn._device import ensure_current_context
from cudnn._torch_stream import _raw_current_stream
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old


@lru_cache(maxsize=128)
def _plan(q_dtype, kv_dtype, n_q_offsets, n_kv_offsets, device_index):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.varlen_metadata import compile_metadata

    major, minor = torch.cuda.get_device_capability(device_index)
    artifact = compile_metadata(q_dtype, kv_dtype, n_q_offsets, n_kv_offsets, device_index, f"sm_{major}{minor}")
    entry = positional_entry(artifact)
    # An explicit foreign target can have no execution entry in the installed
    # DSL. This optional producer must then retain the Torch implementation.
    return None if entry is None else (artifact, entry)


def _torch_metadata(cu_q, cu_kv, q_strides, kv_strides):
    """Retain the existing torch contract when the optional fast path declines."""
    q64, kv64 = cu_q.to(torch.int64), cu_kv.to(torch.int64)
    lengths = ((cu_q[1:] - cu_q[:-1]).to(torch.int32), (cu_kv[1:] - cu_kv[:-1]).to(torch.int32))
    offsets = tuple(q64 * stride for stride in q_strides) + tuple(kv64 * stride for stride in kv_strides)
    return tuple(t.reshape(-1, 1, 1, 1) for t in (*lengths, *offsets))


def prepare_varlen_metadata(cu_q, cu_kv, q_strides, kv_strides):
    """Return two int32 length columns followed by Q/KV int64 offset columns.

    Only the allocating torch wrapper owns the temporary buffers. The compiled
    artifact owns no tensors or pointers; lengths, prefix strides, token strides
    and the launch stream are rebound for every call, for either graph provider.
    """
    installed, version = cutedsl_state()
    supported = (
        installed
        and not cutedsl_too_old(version)
        and cu_q.device.type == "cuda"
        and cu_q.device == cu_kv.device
        and cu_q.ndim == cu_kv.ndim == 1
        and cu_q.numel() == cu_kv.numel()
        and cu_q.numel() > 1
        and cu_q.dtype in (torch.int32, torch.int64)
        and cu_kv.dtype in (torch.int32, torch.int64)
    )
    if not supported or cutedsl_arch_requirement_error(torch.cuda.get_device_capability(cu_q.device)):
        return _torch_metadata(cu_q, cu_kv, q_strides, kv_strides)

    device = cu_q.device
    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        stream = _raw_current_stream(torch, device)
        if stream is None:
            stream = torch.cuda.current_stream(device).cuda_stream
        ensure_current_context(stream, device.index)
        plan = _plan(str(cu_q.dtype), str(cu_kv.dtype), len(q_strides), len(kv_strides), device.index)
        if plan is None:
            return _torch_metadata(cu_q, cu_kv, q_strides, kv_strides)
        n = cu_q.numel() - 1
        # Each column starts at a 16-byte boundary, including odd batch counts.
        offset_pitch, length_pitch = (n + 2) // 2 * 2, (n + 3) // 4 * 4
        offsets = torch.empty((len(q_strides) + len(kv_strides), offset_pitch), dtype=torch.int64, device=device)
        lengths = torch.empty((2, length_pitch), dtype=torch.int32, device=device)
        plan[1](
            cu_q.data_ptr(),
            cu_kv.data_ptr(),
            offsets.data_ptr(),
            lengths.data_ptr(),
            n,
            (cu_q.stride(0), cu_kv.stride(0)),
            (*q_strides, *kv_strides),
            offset_pitch,
            length_pitch,
            stream,
        )
        length_columns = lengths.as_strided((2, n, 1, 1, 1), (length_pitch, 1, 1, 1, 1)).unbind(0)
        offset_columns = offsets.as_strided((offsets.shape[0], n + 1, 1, 1, 1), (offset_pitch, 1, 1, 1, 1)).unbind(0)
        return length_columns + offset_columns
