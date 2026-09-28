# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared packed-to-padded Stats preparation for allocating torch wrappers."""

from contextlib import nullcontext
from functools import lru_cache

import torch

from cudnn._device import ensure_current_context
from cudnn._torch_stream import _raw_current_stream
from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old


@lru_cache(maxsize=128)
def _plan(prefix_dtype, device_index):
    from cudnn.frost.compiled_cache import positional_entry
    from .fwd.kernels.packed_lse import compile_packed_lse

    major, minor = torch.cuda.get_device_capability(device_index)
    artifact = compile_packed_lse(prefix_dtype, device_index, f"sm_{major}{minor}")
    entry = positional_entry(artifact)
    # Some DSL versions compile a foreign architecture without producing a
    # runnable entry. Keep the established Torch path for that environment.
    return None if entry is None else (artifact, entry)


def _torch_repad(lse, cu_seqlens, max_seqlen):
    """Preserve the torch implementation when the prepared path declines."""
    batch = cu_seqlens.numel() - 1
    tokens, heads = lse.shape
    cu = cu_seqlens.long()
    token = torch.arange(tokens, device=lse.device)
    sequence = torch.searchsorted(cu[1:], token, right=True)
    position = token - cu[sequence]
    padded = torch.zeros(batch, heads, max_seqlen, 1, dtype=torch.float32, device=lse.device)
    padded[sequence, :, position, 0] = lse
    return padded


def _execute(lse, cu_seqlens, max_seqlen):
    installed, version = cutedsl_state()
    supported = (
        installed
        and not cutedsl_too_old(version)
        and lse.device.type == "cuda"
        and lse.device == cu_seqlens.device
        and lse.dtype == torch.float32
        and lse.ndim == 2
        and lse.shape[1] <= 65535
        and cu_seqlens.ndim == 1
        and cu_seqlens.numel() > 0
        and cu_seqlens.numel() - 1 <= 65535
        and cu_seqlens.dtype in (torch.int32, torch.int64)
    )
    if not supported or cutedsl_arch_requirement_error(torch.cuda.get_device_capability(lse.device)):
        return _torch_repad(lse, cu_seqlens, max_seqlen)

    device = lse.device
    context = nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
    with context:
        batch, (tokens, heads) = cu_seqlens.numel() - 1, lse.shape
        if batch * heads * max_seqlen == 0:
            return torch.empty((batch, heads, max_seqlen, 1), dtype=torch.float32, device=device)
        stream = _raw_current_stream(torch, device)
        if stream is None:
            stream = torch.cuda.current_stream(device).cuda_stream
        ensure_current_context(stream, device.index)
        plan = _plan(str(cu_seqlens.dtype), device.index)
        if plan is None:
            return _torch_repad(lse, cu_seqlens, max_seqlen)
        padded = torch.empty((batch, heads, max_seqlen, 1), dtype=torch.float32, device=device)
        plan[1](lse.data_ptr(), cu_seqlens.data_ptr(), padded.data_ptr(), batch, tokens, heads, max_seqlen, lse.stride(), cu_seqlens.stride(0), stream)
        return padded


_lib = torch.library.Library("cudnn", "FRAGMENT")
_lib.define("_packed_lse_to_padded(Tensor lse, Tensor cu_seqlens, SymInt max_seqlen) -> Tensor")
_lib.impl("_packed_lse_to_padded", _execute, "CompositeExplicitAutograd")
_compiled_repad = torch.ops.cudnn._packed_lse_to_padded.default


@torch.library.register_fake("cudnn::_packed_lse_to_padded")
def _fake_repad(lse, cu_seqlens, max_seqlen):
    return torch.empty((cu_seqlens.numel() - 1, lse.shape[1], max_seqlen, 1), dtype=torch.float32, device=lse.device)


def prepare_padded_lse(lse, cu_seqlens, max_seqlen):
    """Allocate Stats without reading device prefixes or caching runtime bindings."""
    # Stats consumed by SDPA backward are non-differentiable. Keep the old
    # differentiable torch behavior for other direct callers of this helper.
    if lse.requires_grad and torch.is_grad_enabled():
        return _torch_repad(lse, cu_seqlens, max_seqlen)
    # AOT backward can trace FakeTensor/FunctionalTensor inputs while
    # torch.compiler.is_compiling() is false. Always keep the pointer launch
    # behind the custom-op boundary, including calls made from autograd.
    return _compiled_repad(lse, cu_seqlens, max_seqlen)
