# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared compiler and launch adapter for SM90 DSA sparse-prefill forward."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from cudnn.api_base import check_buffers_disjoint
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import fake_cute_tensor

from ._interface_sm100 import _check_output, _contiguous_aligned, _launch_context, _record_stream
from ._support_sm90 import sm90_dsa_cutedsl_requirement_error

_TORCH_TO_CUTE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
_SUPPORTED_HEADS = (16, 32, 64, 96, 128)
_SUPPORTED_HEAD_DIMS = (512, 576)


@lru_cache(maxsize=None)
def compile_sparse_attention_forward_sm90(
    device_index: int,
    dtype: torch.dtype,
    head_dim: int,
    has_attn_sink: bool,
    has_topk_length: bool,
    indexer_topk: int,
):
    """Compile one storage-free launchable; all sequence extents remain symbolic."""
    dsl_error = sm90_dsa_cutedsl_requirement_error()
    if dsl_error is not None:
        raise RuntimeError(dsl_error)

    # Keep the kernel import behind the version gate: the module uses the
    # CuTe DSL primitive layer at import time, and an environment that bypassed
    # dependency resolution must receive an actionable version error instead
    # of an AttributeError/ImportError from inside the DSL (AGENTS.md Rule 7).
    from .dsa_fwd_sm90 import SparseAttentionForwardSm90

    capability = tuple(torch.cuda.get_device_capability(device_index))
    if capability != (9, 0):
        raise RuntimeError(f"SM90 DSA forward requires compute capability (9, 0), found {capability}")
    element_dtype = _TORCH_TO_CUTE.get(dtype)
    if element_dtype is None:
        raise ValueError(f"SM90 DSA forward supports FP16 or BF16, got {dtype}")
    kernel = SparseAttentionForwardSm90(
        head_dim,
        dtype=element_dtype,
        has_attn_sink=has_attn_sink,
        has_topk_length=has_topk_length,
        indexer_topk=indexer_topk,
    )
    q = fake_cute_tensor(element_dtype, 3)
    kv = fake_cute_tensor(element_dtype, 2)
    indices = fake_cute_tensor(cutlass.Int32, 2, assumed_align=4)
    out = fake_cute_tensor(element_dtype, 3)
    stats = fake_cute_tensor(cutlass.Float32, 2, assumed_align=16)
    lse_indexer = fake_cute_tensor(cutlass.Float32, 2, assumed_align=16) if indexer_topk else None
    attn_sink = fake_cute_tensor(cutlass.Float32, 1, assumed_align=4) if has_attn_sink else None
    topk_length = fake_cute_tensor(cutlass.Int32, 1, assumed_align=4) if has_topk_length else None
    with torch.cuda.device(device_index), torch.cuda.nvtx.range("dsa_sparse_attention_forward_sm90_compile"):
        return cute.compile(
            kernel,
            q,
            kv,
            indices,
            out,
            stats,
            stats,
            lse_indexer,
            attn_sink,
            topk_length,
            cutlass.Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options=compile_options("--opt-level 2", gpu_arch="sm_90a"),
        )


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    indexer_topk: int,
) -> Tuple[int, int, int]:
    """Validate SM90 runtime operands and return ``(S_q, H, D_qk)``."""
    if q.ndim != 3:
        raise ValueError(f"Q must be 3-D (total_S_q, H, D_qk), got {tuple(q.shape)}")
    if kv.ndim != 2:
        raise ValueError(f"KV must be 2-D (total_S_kv, D_qk), got {tuple(kv.shape)}")
    if topk_idxs.ndim != 2:
        raise ValueError(f"topk_idxs must be 2-D (total_S_q, logical_K), got {tuple(topk_idxs.shape)}")
    total_s_q, num_heads, head_dim = q.shape
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(f"SM90 DSA forward supports H in {_SUPPORTED_HEADS}, got {num_heads}")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"SM90 DSA forward supports Dqk in {_SUPPORTED_HEAD_DIMS}, got {head_dim}")
    if q.dtype not in _TORCH_TO_CUTE:
        raise ValueError(f"Q must be float16 or bfloat16, got {q.dtype}")
    if kv.dtype != q.dtype:
        raise ValueError(f"Q and KV must have the same dtype, got {q.dtype} and {kv.dtype}")
    if kv.shape[1] != head_dim:
        raise ValueError(f"KV head dimension ({kv.shape[1]}) must match Q ({head_dim})")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32, got {topk_idxs.dtype}")
    if topk_idxs.shape[0] != total_s_q:
        raise ValueError(f"topk_idxs first dimension ({topk_idxs.shape[0]}) must match Q ({total_s_q})")
    if indexer_topk not in (0, 512, 1024, 2048):
        raise ValueError(f"indexer_topk={indexer_topk} is unsupported on SM90; expected one of (0, 512, 1024, 2048)")
    if indexer_topk > topk_idxs.shape[1]:
        raise ValueError(f"indexer_topk ({indexer_topk}) must not exceed logical K ({topk_idxs.shape[1]})")
    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")
    inputs = (q, kv, topk_idxs, attn_sink, topk_length)
    if any(t is not None and (not t.is_cuda or t.device != device) for t in inputs):
        raise ValueError(f"All inputs must be CUDA tensors on {device}")
    if attn_sink is not None and (attn_sink.dtype != torch.float32 or attn_sink.shape != (num_heads,)):
        raise ValueError(f"attn_sink must be FP32 with shape {(num_heads,)}, got {attn_sink.dtype} {tuple(attn_sink.shape)}")
    if topk_length is not None and (topk_length.dtype != torch.int32 or topk_length.shape != (total_s_q,)):
        raise ValueError(f"topk_length must be INT32 with shape {(total_s_q,)}, got {topk_length.dtype} {tuple(topk_length.shape)}")
    return total_s_q, num_heads, head_dim


def sparse_attention_forward_sm90(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    out: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    lse_indexer: Optional[torch.Tensor] = None,
    current_stream=None,
    compiled_kernel=None,
):
    """Validate and launch SM90; a prepared caller passes ``compiled_kernel``."""
    total_s_q, num_heads, head_dim = _validate_inputs(q, kv, topk_idxs, attn_sink, topk_length, int(indexer_topk))
    device = q.device
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (9, 0):
        raise RuntimeError(f"SM90 DSA forward requires compute capability (9, 0), found {capability}")
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)
    _check_output(out, name="out", shape=(total_s_q, num_heads, 512), dtype=q.dtype, device=device)
    _check_output(max_logits, name="max_logits", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
    _check_output(lse, name="lse", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
    if indexer_topk == 0 and lse_indexer is not None:
        raise ValueError("lse_indexer must be None when indexer_topk == 0")
    _check_output(lse_indexer, name="lse_indexer", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)

    if any(t is None for t in (out, max_logits, lse)) or (indexer_topk and lse_indexer is None):
        raise ValueError("SM90 plan execution requires preallocated outputs; use sparse_attention_forward_wrapper for allocation")
    for name, tensor in (("max_logits", max_logits), ("lse", lse), ("lse_indexer", lse_indexer)):
        if tensor is not None and tensor.data_ptr() % 16:
            raise ValueError(f"{name} base pointer must be 16-byte aligned, got 0x{tensor.data_ptr():x}")
    for name, tensor, alignment in (
        ("Q", q, 16),
        ("KV", kv, 16),
        ("topk_idxs", topk_idxs, 4),
        ("attn_sink", attn_sink, 4),
        ("topk_length", topk_length, 4),
    ):
        if tensor is not None and (not tensor.is_contiguous() or tensor.data_ptr() % alignment):
            raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned for prepared SM90 execution")
    check_buffers_disjoint(
        (("out", out), ("max_logits", max_logits), ("lse", lse), ("lse_indexer", lse_indexer)),
        (("Q", q), ("KV", kv), ("topk_idxs", topk_idxs), ("attn_sink", attn_sink), ("topk_length", topk_length)),
        what="the SM90 DSA forward kernel",
    )

    if compiled_kernel is None:
        compiled_kernel = compile_sparse_attention_forward_sm90(
            device.index,
            q.dtype,
            head_dim,
            attn_sink is not None,
            topk_length is not None,
            int(indexer_topk),
        )
    with _launch_context(device, current_stream) as (stream, consumer):
        if total_s_q == 0:
            pass
        else:
            with torch.cuda.nvtx.range("dsa_sparse_attention_forward_sm90_kernel"):
                compiled_kernel(q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length, cutlass.Float32(scale), stream)
        _record_stream((q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length), consumer)
    return out, max_logits, lse, lse_indexer


def normalize_wrapper_inputs(q, kv, topk_idxs, attn_sink, topk_length, stream):
    """Wrapper-only staging; prepared execute remains allocation-free."""
    with _launch_context(q.device, stream) as (stream, consumer):
        inputs = (q, kv, topk_idxs, attn_sink, topk_length)
        _record_stream(inputs, consumer)
        q = _contiguous_aligned(q, 16)
        kv = _contiguous_aligned(kv, 16)
        topk_idxs = _contiguous_aligned(topk_idxs, 4)
        attn_sink = None if attn_sink is None else _contiguous_aligned(attn_sink, 4)
        topk_length = None if topk_length is None else _contiguous_aligned(topk_length, 4)
    return q, kv, topk_idxs, attn_sink, topk_length
