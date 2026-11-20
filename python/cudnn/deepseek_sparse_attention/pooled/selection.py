# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public pooled selection from precomputed indexer logits."""

import cuda.bindings.driver as cuda
import torch

from cudnn._torch_stream import stream_context
from cudnn.api_base import TupleDict, check_buffers_disjoint
from .forward import _compile_pooled_select
from .indices import _require_dsl


def _workspace_layout(logits, max_pools):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch tensor")
    if logits.ndim != 2 or logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("logits must be a BF16 or FP32 matrix")
    rows, columns = logits.shape
    max_pools = columns if max_pools is None else max_pools
    if not isinstance(max_pools, int) or not 1 <= columns <= max_pools <= 32768:
        raise ValueError("require 1 <= logits columns <= max_pools <= 32768")
    if rows >= 2**31 or 2 * rows * columns >= 2**31:
        raise ValueError("logits exceed the selector's int32 addressing limit")
    bucket = 1 << (max_pools - 1).bit_length()
    shape = (1, 2, 4) if logits.dtype == torch.bfloat16 and 8192 <= bucket <= 32768 else (rows, 2, columns)
    return max_pools, shape, shape[0] * shape[1] * shape[2] * 4


def pooled_select_workspace_size(logits, *, max_pools=None):
    """Return required workspace bytes from tensor metadata; no device reads."""
    return _workspace_layout(logits, max_pools)[2]


def pooled_select_forward(
    logits,
    positions,
    *,
    selected_pools=512,
    pool_size=4,
    mode="decode",
    max_pools=None,
    output=None,
    workspace=None,
    current_stream=None,
):
    """Select causal pools and emit token IDs in one launch.

    Logits are BF16/FP32 [queries, pools]; positions are INT32/INT64
    zero-based token positions. Negative positions disable a query.
    Output is INT32 [output_rows, width], with width at least
    selected_pools * pool_size + pool_size - 1. Extra entries become -1.
    Tied pool IDs and output order are unspecified. No backward is provided.

    Supply output and a contiguous, 256-byte-aligned UINT8 workspace to
    avoid device allocation. Query its size with pooled_select_workspace_size.
    Warm up before capture. Buffers must remain live on current_stream.
    """
    _require_dsl()
    max_pools, scratch_shape, scratch_bytes = _workspace_layout(logits, max_pools)
    if pool_size not in (4, 16) or mode not in ("prefill", "decode"):
        raise ValueError("pool_size must be 4/16 and mode must be prefill/decode")
    if not isinstance(selected_pools, int) or not 1 <= selected_pools <= 2048 or selected_pools % 32:
        raise ValueError("selected_pools must be a multiple of 32 up to 2048")
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch tensor")
    if positions.ndim != 1 or positions.shape[0] != logits.shape[0] or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be INT32/INT64 with one entry per logits row")
    device = logits.device
    if device.type != "cuda":
        raise ValueError("logits must be on a CUDA device")
    capability = torch.cuda.get_device_capability(device)
    if capability not in ((10, 0), (10, 3)):
        raise NotImplementedError("Pooled DSA requires Blackwell SM100 or SM103")
    width = selected_pools * pool_size + pool_size - 1
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.ndim != 2
        or output.dtype != torch.int32
        or not logits.shape[0] <= output.shape[0] < 2**31
        or not width <= output.shape[1] <= 65535 * 256
    ):
        raise ValueError("output must be INT32 and cover all query rows, expanded pools, and tails")
    if workspace is not None and (
        not isinstance(workspace, torch.Tensor)
        or workspace.ndim != 1
        or workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < scratch_bytes
        or workspace.data_ptr() % 256
    ):
        raise ValueError(f"workspace must be a contiguous, 256-byte-aligned UINT8 vector with at least {scratch_bytes} bytes")
    for name, tensor in (("logits", logits), ("positions", positions), ("output", output), ("workspace", workspace)):
        if tensor is None:
            continue
        strides = tensor.stride()
        if tensor.device != device or tensor.requires_grad:
            raise ValueError(f"{name} must be on the logits device and require no autograd")
        if any(s <= 0 for s in strides) or (tensor.ndim == 2 and (strides[1] != 1 or strides[0] < tensor.shape[1])):
            raise ValueError(f"{name} requires positive strides and non-overlapping row-major axes")
        if tensor.numel() and tensor.data_ptr() % (32 if name == "logits" else tensor.element_size()):
            raise ValueError(f"{name} has an unsupported address alignment")
    check_buffers_disjoint((("output", output), ("workspace", workspace)), (("logits", logits), ("positions", positions)), what="pooled selection")
    with torch.cuda.device(device), stream_context(current_stream, device):
        if output is None:
            output = torch.empty((logits.shape[0], width), dtype=torch.int32, device=device)
        if workspace is None:
            workspace = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)
        kernel = _compile_pooled_select(device.index, capability, max_pools, selected_pools, pool_size, positions.dtype, mode, logits.dtype)
        if output.shape[0]:
            scratch = workspace[:scratch_bytes].view(torch.int32).view(scratch_shape)
            kernel(logits, scratch, positions, output, cuda.CUstream(torch.cuda.current_stream(device).cuda_stream))
    return TupleDict(indices=output)
