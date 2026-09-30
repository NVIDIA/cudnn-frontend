# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared data copies used by the existing SM80 packed convenience wrappers."""

from contextlib import nullcontext
from functools import lru_cache

import torch

from cudnn._device import ensure_current_context
from cudnn._torch_stream import _raw_current_stream


@lru_cache(maxsize=128)
def _plan(shapes):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.sm80.staged_copy import compile_gather

    artifact = compile_gather(shapes, packed=True)
    entry = positional_entry(artifact)
    if entry is None:
        raise NotImplementedError("SM80 packed copies require a positional tvm-ffi entry")
    return artifact, entry


def _copy(plan, sources, outputs, stream, t_q, t_kv):
    plan[1](
        tuple(t.data_ptr() for t in sources),
        tuple(t.data_ptr() for t in outputs),
        tuple(tuple(t.stride()) for t in sources),
        t_q,
        t_kv,
        stream,
    )


@lru_cache(maxsize=128)
def _layout(dimensions, widths, query_roles, contiguous):
    """Memoize wrapper allocation metadata, separately from compiled artifacts.

    Capacities may change this bounded host memo but never the compiler key.
    Tensor ownership, addresses, strides, devices and streams are not retained.
    """
    if not (len(dimensions) == len(widths) == len(query_roles)):
        raise ValueError("SM80 packed copy roles and widths must match the operand count")
    t_q = t_kv = None
    for shape, query in zip(dimensions, query_roles):
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError("SM80 packed operands must be [1,T,H,D]")
        capacity = t_q if query else t_kv
        if capacity is not None and shape[1] != capacity:
            raise ValueError("SM80 packed operands must agree on Q and KV capacities")
        if query:
            t_q = shape[1]
        else:
            t_kv = shape[1]
    t_q, t_kv = t_q or 0, t_kv or 0
    selected = tuple(i for i, (shape, width) in enumerate(zip(dimensions, widths)) if shape[-1] != width or (contiguous and not contiguous[i]))
    shapes = tuple((1, -1 if query_roles[i] else -2, dimensions[i][2], min(dimensions[i][3], widths[i]), widths[i]) for i in selected)
    outputs = tuple((*dimensions[i][:-1], widths[i]) for i in selected)
    live = any(dimensions[i][1] for i in selected)
    return selected, shapes, outputs, t_q, t_kv, live


def copy_packed_half(tensors, widths, query_roles, *, compact=False):
    """Pad, trim or compact existing [1,T,H,D] operands on the current stream.

    Allocation remains a convenience-wrapper responsibility. Cached artifacts
    contain only head/width geometry; capacities, strides and pointers are bound
    per call. CPU prefixes and scalar operands retain their separate contract.
    """
    selected, shapes, output_shapes, t_q, t_kv, live = _layout(
        tuple(t.shape for t in tensors), widths, query_roles, tuple(t.is_contiguous() for t in tensors) if compact else ()
    )
    if not selected:
        return tensors
    device, dtype = tensors[0].device, tensors[0].dtype
    if dtype not in (torch.float16, torch.bfloat16) or device.type != "cuda":
        raise ValueError("SM80 packed copies require CUDA FP16/BF16 tensors")
    if any(t.dtype != dtype or t.device != device for t in tensors):
        raise ValueError("SM80 packed operands must have matching dtype and device")
    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        plan = _plan(shapes)
        outputs = list(tensors)
        for i, shape in zip(selected, output_shapes):
            outputs[i] = torch.empty(shape, dtype=dtype, device=device)
        if live:
            stream = _raw_current_stream(torch, device)
            if stream is None:
                stream = torch.cuda.current_stream(device).cuda_stream
            ensure_current_context(stream, device.index)
            _copy(plan, tuple(tensors[i] for i in selected), tuple(outputs[i] for i in selected), stream, t_q, t_kv)
    return tuple(outputs)
