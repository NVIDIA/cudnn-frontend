# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compilation and caller-owned storage for the H128 D576 two-CTA backend."""

from functools import lru_cache
import math

import torch
import cutlass
import cutlass.cute as cute

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context

from ._interface_sm100 import _BLACKWELL_CAPABILITIES, _align_workspace_bytes, _workspace_shapes_sm100, flash_attn_bwd_sm100_workspace_size

BACKEND = "h128_d576_2cta_m64"


def _workspace_sizes(total_s_q: int, total_s_kv: int) -> tuple[int, int]:
    """Byte sizes of the FP32 LSE/OdO and dKV planes, shared with the generic kernel."""
    return tuple(math.prod(shape) for shape in _workspace_shapes_sm100(total_s_q, total_s_kv, 576, 128, False))


def _workspace_bytes(total_s_q: int, total_s_kv: int) -> int:
    """Caller scratch bytes for this shape; identical to the generic SM100 formula."""
    return flash_attn_bwd_sm100_workspace_size(total_s_q, total_s_kv, 576, 128, False)


def _split_count(total_s_q: int, max_topk: int, sm_count: int) -> int:
    """Spread short query batches across otherwise idle two-SM clusters."""
    max_tiles = (max_topk + 63) // 64
    return max(1, min((sm_count // 2) // max(total_s_q, 1), 1 + max_tiles // 2, 8))


@lru_cache(maxsize=None)
def _compile_d576_2cta(device_capability, max_topk: int, has_topk_length: bool, single_query: bool, split_regime: bool):
    """Compile from metadata; sequence extents and scale remain runtime inputs."""
    if device_capability not in _BLACKWELL_CAPABILITIES:
        raise ValueError(f"the H128 D576 two-CTA backend requires an SM100-class device with compute capability in {_BLACKWELL_CAPABILITIES}")

    from .dsa_bwd_sm100_h128_d576_2cta import FlashAttentionDSABackwardSm100H128D576TwoCTA

    sq, skv = cute.sym_int(), cute.sym_int()

    def tensor(dtype, shape, alignment=16):
        """Fake compact row-major tensor whose leading extent stays symbolic."""
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=alignment)

    q = tensor(cutlass.BFloat16, (sq, 128, 576))
    kv = tensor(cutlass.BFloat16, (skv, 576))
    out = tensor(cutlass.BFloat16, (sq, 128, 512))
    lse = tensor(cutlass.Float32, (sq, 128), 8)
    sink = tensor(cutlass.Float32, (128,))
    indices = tensor(cutlass.Int32, (sq, max_topk))
    lengths = tensor(cutlass.Int32, (sq,), 4) if has_topk_length else None
    stats_workspace = tensor(cutlass.Uint8, (cute.sym_int(),))
    dkv_workspace = tensor(cutlass.Uint8, (cute.sym_int(),))
    kernel = FlashAttentionDSABackwardSm100H128D576TwoCTA(
        element_dtype=cutlass.BFloat16,
        head_dim=576,
        head_dim_v=512,
        block_tile=64,
        max_topk=max_topk,
        single_query=single_query,
    )
    return cute.compile(
        kernel,
        (cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(576), (cutlass.Int32(128), cutlass.Int32(1))),
        q,
        kv,
        out,
        out,
        lse,
        sink,
        indices,
        lengths,
        q,
        kv,
        sink,
        stats_workspace,
        dkv_workspace,
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        split_regime,
        options=compile_options(),
    )


def _execute_d576_2cta(
    compiled_kernel,
    q,
    kv,
    out,
    dout,
    lse,
    attn_sink,
    topk_idxs,
    topk_length,
    dq,
    dkv,
    d_sink,
    workspace,
    softmax_scale,
    split_count,
    current_stream,
):
    """Validate preallocated arguments and launch without allocation or copies."""
    sq, skv = q.shape[0], kv.shape[0]
    specifications = (
        ("q", q, (sq, 128, 576), torch.bfloat16, 16),
        ("kv", kv, (skv, 576), torch.bfloat16, 16),
        ("out", out, (sq, 128, 512), torch.bfloat16, 16),
        ("dout", dout, (sq, 128, 512), torch.bfloat16, 16),
        ("lse", lse, (sq, 128), torch.float32, 8),
        ("attn_sink", attn_sink, (128,), torch.float32, 16),
        ("topk_idxs", topk_idxs, (sq, topk_idxs.shape[1]), torch.int32, 16),
        ("dq", dq, (sq, 128, 576), torch.bfloat16, 16),
        ("dkv", dkv, (skv, 576), torch.bfloat16, 16),
        ("d_sink", d_sink, (128,), torch.float32, 16),
    )
    if topk_length is not None:
        specifications += (("topk_length", topk_length, (sq,), torch.int32, 4),)
    for name, value, shape, dtype, alignment in specifications:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        if tuple(value.shape) != shape or value.dtype != dtype or value.device != q.device:
            raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {q.device}")
        if not value.is_contiguous() or value.data_ptr() % alignment:
            raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")

    required = _workspace_bytes(sq, skv)
    if not isinstance(workspace, torch.Tensor):
        raise ValueError(f"the H128 D576 two-CTA backend requires a {required}-byte caller workspace")
    if workspace.dtype != torch.uint8 or workspace.device != q.device or not workspace.is_contiguous():
        raise ValueError("workspace must be a contiguous uint8 tensor on Q's device")
    if workspace.numel() < required or workspace.data_ptr() % 16:
        raise ValueError(f"workspace must contain at least {required} bytes and be 16-byte aligned")
    stats_bytes, dkv_bytes = _workspace_sizes(sq, skv)
    flat = workspace.view(-1)
    dkv_begin = _align_workspace_bytes(stats_bytes)
    stats_workspace = flat[:stats_bytes]
    dkv_workspace = flat[dkv_begin : dkv_begin + dkv_bytes]
    scale = 1.0 / math.sqrt(576) if softmax_scale is None else softmax_scale
    if not isinstance(scale, (float, int)):
        raise TypeError("softmax_scale must be a Python scalar")

    with torch.cuda.device(q.device):
        current_stream = resolve_stream(current_stream)
        with torch.cuda.nvtx.range(f"flash_attn_bwd_sm100_kernel[{BACKEND}]"):
            compiled_kernel(
                (sq, skv, 576, (128, 1)),
                q,
                kv,
                out,
                dout,
                lse,
                attn_sink,
                topk_idxs,
                topk_length,
                dq,
                dkv,
                d_sink,
                stats_workspace,
                dkv_workspace,
                scale,
                split_count,
                current_stream,
            )
    return dq, dkv, d_sink


def flash_attn_bwd_sm100_h128_d576(
    q,
    kv,
    out,
    dout,
    lse,
    attn_sink,
    topk_idxs,
    softmax_scale=None,
    topk_length=None,
    dq=None,
    dkv=None,
    current_stream=None,
    workspace=None,
    *,
    d_sink=None,
):
    """Direct entry point: allocate omitted outputs and scratch, then launch.

    ``SparseAttentionBackward`` keeps compilation at plan time and forbids
    allocation in ``execute()``; this function offers ``flash_attn_bwd_sm100``
    callers the same kernel with that interface's allocate-on-demand behavior.
    """
    sq, skv = q.shape[0], kv.shape[0]
    max_topk = topk_idxs.shape[1]
    with torch.cuda.device(q.device):
        split_count = _split_count(sq, max_topk, torch.cuda.get_device_properties(q.device).multi_processor_count)
        compiled_kernel = _compile_d576_2cta(torch.cuda.get_device_capability(q.device), max_topk, topk_length is not None, sq == 1, split_count > 1)
        current_stream = resolve_stream(current_stream)
        with torch_stream_context(current_stream):
            if dq is None:
                dq = torch.empty_like(q)
            if dkv is None:
                dkv = torch.empty_like(kv)
            if d_sink is None:
                d_sink = torch.empty_like(attn_sink)
            if workspace is None:
                workspace = torch.empty(_workspace_bytes(sq, skv), dtype=torch.uint8, device=q.device)
    return _execute_d576_2cta(
        compiled_kernel,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length,
        dq,
        dkv,
        d_sink,
        workspace,
        softmax_scale,
        split_count,
        current_stream,
    )
