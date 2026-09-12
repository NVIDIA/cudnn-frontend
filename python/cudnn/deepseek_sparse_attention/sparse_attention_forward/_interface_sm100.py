# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime interface for the two SM100 DSA sparse-prefill kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

_compile_cache: dict = {}
_ARCH_FLAGS = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_100f",
}


def _gpu_arch_flag(device: torch.device) -> str:
    """Return the architecture-specific compiler target for ``device``."""
    if not torch.cuda.is_available():
        raise RuntimeError("SparseAttentionForward compilation requires CUDA")
    capability = torch.cuda.get_device_capability(device)
    arch = _ARCH_FLAGS.get(capability)
    if arch is None:
        raise RuntimeError(f"SparseAttentionForward does not map compute capability {capability} to a CuTe compiler target")
    return arch


def _compile_options(device: torch.device) -> str:
    return f"--enable-tvm-ffi --gpu-arch {_gpu_arch_flag(device)} --opt-level 2"


def _kernel_variant(num_heads: int, head_dim: int) -> str:
    if num_heads == 64 and head_dim in (512, 576):
        return "head64_regular"
    if num_heads == 128 and head_dim == 512:
        return "head128_small_topk_prefill"
    raise ValueError(f"Unsupported SparseAttentionForward variant H={num_heads}, D_qk={head_dim}")


def _make_kernel(variant: str, head_dim: int, indexer_topk: int):
    """Construct one variant behind a narrow adapter for signature changes."""
    if variant == "head64_regular":
        from .dsa_fwd_sm100_head64 import SparseAttentionForwardSm100Head64

        return SparseAttentionForwardSm100Head64(head_dim=head_dim, indexer_topk=indexer_topk)
    if variant == "head128_small_topk_prefill":
        from .dsa_fwd_sm100_head128_small_topk import SparseAttentionForwardSm100Head128SmallTopKPrefill

        return SparseAttentionForwardSm100Head128SmallTopKPrefill(d_qk=head_dim, indexer_topk=indexer_topk)
    raise AssertionError(f"Unknown kernel variant {variant}")


def _compile_kernel(
    kernel_obj,
    q,
    kv,
    indices,
    out,
    max_logits,
    lse,
    lse_indexer,
    attn_sink,
    topk_length,
    softmax_scale: float,
    stream,
    device: torch.device,
):
    """Compile both variants through their shared flat-tensor call contract."""
    return cute.compile(
        kernel_obj,
        to_cute_tensor(q, divisibility=q.shape[-1]),
        to_cute_tensor(kv, divisibility=kv.shape[-1]),
        to_cute_tensor(indices, assumed_align=4),
        to_cute_tensor(out, divisibility=out.shape[-1]),
        to_cute_tensor(max_logits, assumed_align=4),
        to_cute_tensor(lse, assumed_align=4),
        to_cute_tensor(lse_indexer, assumed_align=4) if lse_indexer is not None else None,
        to_cute_tensor(attn_sink, assumed_align=4) if attn_sink is not None else None,
        to_cute_tensor(topk_length, assumed_align=4) if topk_length is not None else None,
        cutlass.Float32(softmax_scale),
        stream,
        # CuTe DSL defaults to O3, but these large persistent kernels get a
        # worse CFG/register allocation at that level.  Keep O2 explicit: it
        # is both faster and smaller on the target SM103 (B300) compiler.
        options=_compile_options(device),
    )


def _check_output(
    tensor: Optional[torch.Tensor],
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    # ``from_dlpack(..., assumed_align=16)`` is used for 16-bit O.  A storage-
    # offset view can still be contiguous while its base pointer is only
    # 2-byte aligned, so contiguity alone is not sufficient.
    if dtype in (torch.float16, torch.bfloat16) and tensor.data_ptr() % 16:
        raise ValueError(f"{name} base pointer must be 16-byte aligned, got 0x{tensor.data_ptr():x}")


def _contiguous_aligned(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """Materialize a contiguous, aligned tensor when either property is absent."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    else:
        # PyTorch ignores singleton dimensions when deciding contiguity, so a
        # broadcast view such as shape (1, K), stride (0, 1) is contiguous but
        # has a different DLPack layout signature. Canonicalize only the
        # metadata (no copy) so structurally cached CuTe callables do not
        # depend on which singleton-stride representation compiled first.
        expected_strides = []
        running_stride = 1
        for size in reversed(tensor.shape):
            expected_strides.append(running_stride)
            running_stride *= max(int(size), 1)
        expected_strides = tuple(reversed(expected_strides))
        if tensor.stride() != expected_strides:
            tensor = tensor.as_strided(tensor.shape, expected_strides)
    # ``contiguous()`` is a no-op for a contiguous storage-offset view.  Force
    # a fresh allocator-backed storage in that case.
    if tensor.data_ptr() % alignment:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
    return tensor


def _record_stream(tensors, stream, device: torch.device) -> None:
    """Tell PyTorch's allocator that raw kernel pointers live on ``stream``."""
    consumer = torch.cuda.get_stream_from_external(int(stream), device)
    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(consumer)


def _normalize_and_validate(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    topk_length: Optional[torch.Tensor],
    indexer_topk: int,
    stream,
):
    if q.ndim != 3:
        raise ValueError(f"Q must be 3-D (total_S_q, H, D_qk), got {tuple(q.shape)}")
    if kv.ndim != 2:
        raise ValueError(f"KV must be 2-D (total_S_kv, D_qk), got {tuple(kv.shape)}")
    if topk_idxs.ndim != 2:
        raise ValueError(f"topk_idxs must be 2-D (total_S_q, logical_K), got {tuple(topk_idxs.shape)}")
    total_s_q, num_heads, head_dim = q.shape
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be float16 or bfloat16, got {q.dtype}")
    if kv.dtype != q.dtype:
        raise ValueError(f"Q and KV must have the same dtype, got {q.dtype} and {kv.dtype}")
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs must be int32, got {topk_idxs.dtype}")
    if kv.shape[1] != head_dim:
        raise ValueError(f"KV head dimension ({kv.shape[1]}) must match Q ({head_dim})")
    if topk_idxs.shape[0] != total_s_q:
        raise ValueError(f"topk_idxs first dimension ({topk_idxs.shape[0]}) must match Q ({total_s_q})")
    variant = _kernel_variant(num_heads, head_dim)
    valid_indexer = (0, 512, 1024, 2048) if num_heads == 64 else (0, 512, 1024)
    if indexer_topk not in valid_indexer:
        raise ValueError(f"indexer_topk={indexer_topk} is unsupported for H={num_heads}; expected one of {valid_indexer}")
    logical_topk = topk_idxs.shape[1]
    if indexer_topk > logical_topk:
        raise ValueError(f"indexer_topk ({indexer_topk}) must not exceed logical K ({logical_topk})")

    device = q.device
    if device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {device}")
    inputs = [q, kv, topk_idxs]
    if attn_sink is not None:
        if attn_sink.dtype != torch.float32 or attn_sink.shape != (num_heads,):
            raise ValueError(f"attn_sink must be FP32 with shape {(num_heads,)}, got {attn_sink.dtype} {tuple(attn_sink.shape)}")
        inputs.append(attn_sink)
    if topk_length is not None:
        if topk_length.dtype != torch.int32 or topk_length.shape != (total_s_q,):
            raise ValueError(f"topk_length must be INT32 with shape {(total_s_q,)}, got {topk_length.dtype} {tuple(topk_length.shape)}")
        inputs.append(topk_length)
    if any(not tensor.is_cuda or tensor.device != device for tensor in inputs):
        raise ValueError(f"All inputs must be CUDA tensors on {device}")

    # The normalization copies below are asynchronous on an explicit stream.
    # Record their original sources before replacing local references with
    # aligned/contiguous tensors so the caching allocator cannot recycle the
    # source storage while a copy is still pending.
    _record_stream(inputs, stream, device)
    with torch_stream_context(stream):
        q = _contiguous_aligned(q, 16)
        kv = _contiguous_aligned(kv, 16)
        topk_idxs = topk_idxs if topk_idxs.is_contiguous() else topk_idxs.contiguous()
        attn_sink = None if attn_sink is None else _contiguous_aligned(attn_sink, 4)
        topk_length = None if topk_length is None else _contiguous_aligned(topk_length, 4)

        padded_topk = ((logical_topk + 63) // 64) * 64
        if padded_topk != logical_topk:
            padding = torch.full((total_s_q, padded_topk - logical_topk), -1, dtype=torch.int32, device=device)
            topk_idxs = torch.cat((topk_idxs, padding), dim=1)

        # Head128 issues one 256-bit load per eight indices.  Normalize after
        # padding so both the base and every 64-INT32 row remain 32B-aligned.
        topk_idxs = _contiguous_aligned(topk_idxs, 32)

    return q, kv, topk_idxs, attn_sink, topk_length, variant, logical_topk


def sparse_attention_forward_sm100(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Normalize, compile, and launch one SM100 sparse-prefill variant."""
    if q.device.type != "cuda":
        raise ValueError(f"Q must live on CUDA, got {q.device}")
    device = q.device
    with torch.cuda.device(device):
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 10:
            raise RuntimeError(f"SparseAttentionForward requires an SM100-family GPU, found SM{capability[0]}{capability[1]}")
        # Validate the exact architecture even for a zero-size problem that
        # returns before cute.compile.
        _gpu_arch_flag(device)
        current_stream = resolve_stream(current_stream)
        stream_status, stream_device = cuda.cuStreamGetDevice(current_stream)
        if stream_status != cuda.CUresult.CUDA_SUCCESS:
            raise ValueError(f"Unable to resolve the CUDA device for stream {current_stream}: {stream_status}")
        if int(stream_device) != device.index:
            raise ValueError(f"stream belongs to cuda:{int(stream_device)}, but Q is on {device}")
        q, kv, topk_idxs, attn_sink, topk_length, variant, logical_topk = _normalize_and_validate(
            q,
            kv,
            topk_idxs,
            attn_sink,
            topk_length,
            int(indexer_topk),
            current_stream,
        )
        total_s_q, num_heads, head_dim = q.shape
        head_dim_v = 512
        scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)

        _check_output(out, name="out", shape=(total_s_q, num_heads, head_dim_v), dtype=q.dtype, device=device)
        _check_output(max_logits, name="max_logits", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
        _check_output(lse, name="lse", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)
        if indexer_topk == 0 and lse_indexer is not None:
            raise ValueError("lse_indexer must be None when indexer_topk == 0")
        _check_output(lse_indexer, name="lse_indexer", shape=(total_s_q, num_heads), dtype=torch.float32, device=device)

        with torch_stream_context(current_stream):
            if out is None:
                out = torch.empty((total_s_q, num_heads, head_dim_v), dtype=q.dtype, device=device)
            if max_logits is None:
                max_logits = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)
            if lse is None:
                lse = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)
            if indexer_topk and lse_indexer is None:
                lse_indexer = torch.empty((total_s_q, num_heads), dtype=torch.float32, device=device)

            # Zero-size problems are a stream-ordered host-side epilogue.  No
            # gather pointer is formed and no CuTe kernel is launched.
            if total_s_q == 0 or logical_topk == 0 or kv.shape[0] == 0:
                out.zero_()
                max_logits.fill_(float("-inf"))
                lse.fill_(float("inf"))
                if lse_indexer is not None:
                    lse_indexer.fill_(float("inf"))
                _record_stream(
                    (q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length),
                    current_stream,
                    device,
                )
                return out, max_logits, lse, lse_indexer

        # ``cutlass.Float32(scale)`` is a TVM-FFI runtime scalar argument, as
        # in the existing DSA indexer-forward interface; it intentionally does
        # not specialize generated code or enter this cache key.
        # ``to_cute_tensor`` makes the sequence/top-k extents and their
        # normalized layouts runtime-dynamic.  Cache only properties that
        # change the generated kernel or its optional-argument signature.
        compile_key = (
            device,
            capability,
            variant,
            int(num_heads),
            int(head_dim),
            q.dtype,
            kv.dtype,
            topk_idxs.dtype,
            attn_sink is not None,
            topk_length is not None,
            int(indexer_topk),
        )
        compiled = _compile_cache.get(compile_key)
        if compiled is None:
            kernel_obj = _make_kernel(variant, head_dim, int(indexer_topk))
            with torch.cuda.nvtx.range("dsa_sparse_attention_forward_compile"):
                compiled = _compile_kernel(
                    kernel_obj,
                    q,
                    kv,
                    topk_idxs,
                    out,
                    max_logits,
                    lse,
                    lse_indexer,
                    attn_sink,
                    topk_length,
                    scale,
                    current_stream,
                    device,
                )
            _compile_cache[compile_key] = compiled

        with torch.cuda.nvtx.range("dsa_sparse_attention_forward_kernel"):
            compiled(
                q,
                kv,
                topk_idxs,
                out,
                max_logits,
                lse,
                lse_indexer,
                attn_sink,
                topk_length,
                cutlass.Float32(scale),
                current_stream,
            )
        _record_stream(
            (q, kv, topk_idxs, out, max_logits, lse, lse_indexer, attn_sink, topk_length),
            current_stream,
            device,
        )
        return out, max_logits, lse, lse_indexer
