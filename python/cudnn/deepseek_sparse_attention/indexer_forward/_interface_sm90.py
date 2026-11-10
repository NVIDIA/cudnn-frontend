# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexer Forward Interface - SM90 direct CuTe DSL backend.

This wrapper launches ``IndexerForwardSm90``, a direct CuTeDSL port of the old
SM90 C++ forward kernel while avoiding the optimized dense-score kernel reuse
path.
"""

from __future__ import annotations

from typing import Optional

import torch

import cutlass
import cutlass.cute as cute

from .indexer_fwd_sm90 import IndexerForwardSm90
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options, gpu_arch_flag
from cudnn.deepseek_sparse_attention.utils.runtime import (
    device_capability,
    maybe_contiguous as _maybe_contiguous,
    resolve_stream,
    torch_stream_context as _torch_stream_context,
    validate_q_causal_offsets,
)
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import (
    fake_cute_tensor,
    get_broadcast_dims,
    to_cute_tensor as _to_cute_tensor,
)
from cudnn.frost.capture import is_capturing

_SUPPORTED_QHPKV = (16, 32, 64)
_SUPPORTED_FP8_QHPKV = (32, 64)
_compile_cache: dict = {}


def _make_i64_cand_buffer_compile_tensor():
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.Float32,
        shape=(cute.sym_int64(symbol="cand_buffer_numel"),),
        stride=(1,),
        assumed_align=16,
    )


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def _validate_common(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    precision: str,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    use_fp8_scales: bool,
) -> None:
    if precision == "bf16":
        assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
        assert k.dtype == torch.bfloat16, f"k must be bfloat16, got {k.dtype}"
        if q_scale is not None or k_scale is not None:
            raise ValueError("q_scale and k_scale are only valid with precision='fp8'")
    elif precision == "fp8":
        assert q.dtype == torch.float8_e4m3fn, f"q must be float8_e4m3fn, got {q.dtype}"
        assert k.dtype == torch.float8_e4m3fn, f"k must be float8_e4m3fn, got {k.dtype}"
        if q_scale is None or k_scale is None:
            raise ValueError("precision='fp8' requires q_scale and k_scale")
        assert q_scale.dtype == torch.float32, f"q_scale must be float32 descale, got {q_scale.dtype}"
        assert k_scale.dtype == torch.float32, f"k_scale must be float32 descale, got {k_scale.dtype}"
        assert q_scale.is_cuda and k_scale.is_cuda, "q_scale and k_scale must be CUDA tensors"
    else:
        raise ValueError(f"precision must be 'bf16' or 'fp8', got {precision!r}")
    expected_w_dtypes = (torch.bfloat16, torch.float32)
    assert w.dtype in expected_w_dtypes, f"w must be one of {expected_w_dtypes}, got {w.dtype}"
    if precision == "bf16" and w.dtype == torch.float32 and w.stride(-1) != 1:
        raise NotImplementedError(f"FP32 w requires unit last stride, got strides {w.stride()}")
    assert q.is_cuda and k.is_cuda and w.is_cuda, "q, k, w must be CUDA tensors"


def indexer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    *,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    current_stream=None,
    is_compressed_logits: bool = False,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    _compile_only: bool = False,
    _compiled_kernel=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Indexer QK forward pass using the direct SM90 CuTe DSL port.

    ``precision="bf16"`` accepts BF16 Q/K and BF16 or FP32 W.
    ``precision="fp8"`` is the Hopper 1x128 descale path: Q/K are
    ``torch.float8_e4m3fn`` and ``q_scale``/``k_scale`` are FP32 descale
    tensors with one value per token/head.  W may be BF16, or FP32 when it has
    already been pre-scaled by ``q_scale * sm_scale``.
    """
    precision = precision.lower()
    if _compile_only and not is_compressed_logits:
        raise ValueError("_compile_only is reserved for the prepared compressed-logits path")
    use_fp8_scales = precision == "fp8"
    _validate_common(
        q,
        k,
        w,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        use_fp8_scales=use_fp8_scales,
    )
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    compute_lse = bool(return_lse or lse_out is not None)

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    if is_varlen_q != is_varlen_k:
        raise ValueError("THD input requires both cu_seqlens_q and cu_seqlens_k")
    is_varlen = is_varlen_q
    if is_compressed_logits and not is_varlen:
        raise NotImplementedError("SM90 compressed indexer forward currently supports packed THD only")
    capability = device_capability(q.device)
    if is_compressed_logits and capability != (9, 0):
        raise NotImplementedError(f"SM90 compressed indexer forward requires compute capability (9, 0), got {capability}")
    device_index = q.device.index
    assert device_index is not None

    current_stream = resolve_stream(current_stream, q.device)
    q = _maybe_contiguous(q, current_stream)
    k = _maybe_contiguous(k, current_stream)
    w = _maybe_contiguous(w, current_stream)
    q_scale = _maybe_contiguous(q_scale, current_stream) if q_scale is not None else None
    k_scale = _maybe_contiguous(k_scale, current_stream) if k_scale is not None else None

    if is_varlen:
        assert cu_seqlens_q is not None and cu_seqlens_k is not None
        for t, name in ((cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")):
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.ndim == 1, f"{name} must be 1D"
            assert t.is_cuda, f"{name} must be on CUDA"
            assert t.stride(0) == 1, f"{name} must be contiguous"
        assert q.ndim == 3, f"THD q must be 3D, got {q.ndim}D"
        assert k.ndim == 3, f"THD k must be 3D, got {k.ndim}D"
        assert w.ndim == 2, f"THD w must be 2D, got {w.ndim}D"
        total_q, n_heads_q, head_dim = q.shape
        _, n_heads_kv, head_dim_k = k.shape
        batch_size = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_k.shape == (batch_size + 1,)
        if max_seqlen_q is None or max_seqlen_k is None:
            raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")
        seqlen_q_dim = int(max_seqlen_q)
        seqlen_k_dim = int(max_seqlen_k)
        out_shape = None if is_compressed_logits else (total_q, seqlen_k_dim)
        lse_shape = (total_q,)
        if use_fp8_scales:
            assert q_scale is not None and k_scale is not None
            assert q_scale.shape == (total_q, n_heads_q), f"THD q_scale shape must be {(total_q, n_heads_q)}, got {tuple(q_scale.shape)}"
            assert k_scale.shape == (k.shape[0], n_heads_kv), f"THD k_scale shape must be {(k.shape[0], n_heads_kv)}, got {tuple(k_scale.shape)}"
    else:
        assert q.ndim == 4, f"q must be 4D BSHD, got {q.ndim}D"
        assert k.ndim == 4, f"k must be 4D BSHD, got {k.ndim}D"
        assert w.ndim == 3, f"w must be 3D BSH, got {w.ndim}D"
        batch_size, seqlen_q_dim, n_heads_q, head_dim = q.shape
        kb, seqlen_k_dim, n_heads_kv, head_dim_k = k.shape
        assert kb == batch_size, f"q batch ({batch_size}) != k batch ({kb})"
        assert w.shape == (batch_size, seqlen_q_dim, n_heads_q), f"w shape must be {(batch_size, seqlen_q_dim, n_heads_q)}, got {tuple(w.shape)}"
        out_shape = (batch_size, seqlen_q_dim, seqlen_k_dim)
        lse_shape = (batch_size, seqlen_q_dim)
        if use_fp8_scales:
            assert q_scale is not None and k_scale is not None
            assert q_scale.shape == (batch_size, seqlen_q_dim, n_heads_q), (
                f"BSHD q_scale shape must be {(batch_size, seqlen_q_dim, n_heads_q)}, " f"got {tuple(q_scale.shape)}"
            )
            assert k_scale.shape == (batch_size, seqlen_k_dim, n_heads_kv), (
                f"BSHD k_scale shape must be {(batch_size, seqlen_k_dim, n_heads_kv)}, " f"got {tuple(k_scale.shape)}"
            )

    assert head_dim == head_dim_k, f"q head_dim ({head_dim}) != k head_dim ({head_dim_k})"
    assert head_dim == 128, f"head_dim must be 128, got {head_dim}"
    assert n_heads_kv == 1, f"SM90 direct fwd currently supports num_head_kv == 1, got {n_heads_kv}"
    assert n_heads_q % n_heads_kv == 0
    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    assert qhead_per_kv_head == n_heads_q // n_heads_kv
    supported_qhpkv = _SUPPORTED_FP8_QHPKV if use_fp8_scales else _SUPPORTED_QHPKV
    assert qhead_per_kv_head in supported_qhpkv, f"qhead_per_kv_head must be one of {supported_qhpkv}, got {qhead_per_kv_head}"
    q_causal_offsets = validate_q_causal_offsets(q_causal_offsets, int(batch_size), q.device, stream=current_stream)

    if is_compressed_logits:
        if cand_batch_offsets is None:
            raise ValueError("SM90 compressed indexer forward requires cand_batch_offsets")
        if (
            not cand_batch_offsets.is_cuda
            or cand_batch_offsets.dtype != torch.int64
            or cand_batch_offsets.ndim != 1
            or cand_batch_offsets.numel() != batch_size + 1
            or not cand_batch_offsets.is_contiguous()
            or cand_batch_offsets.device != q.device
        ):
            raise ValueError("cand_batch_offsets must be a contiguous int64 CUDA tensor of shape (batch_size + 1,) on the input device")
        if out is None and not _compile_only:
            raise ValueError("SM90 compressed indexer forward requires a caller-provided cand_buffer")
        if out is not None and (
            not out.is_cuda or out.dtype != torch.float32 or out.ndim != 1 or not out.is_contiguous() or out.device != q.device or out.data_ptr() % 16 != 0
        ):
            raise ValueError("cand_buffer must be a contiguous, 16-byte-aligned 1D float32 CUDA tensor on the input device")

    if out is None and not _compile_only:
        with _torch_stream_context(current_stream, q.device):
            out = torch.empty(out_shape, dtype=torch.float32, device=q.device)
    elif not is_compressed_logits:
        assert out.shape == out_shape, f"out must have shape {out_shape}, got {tuple(out.shape)}"
        assert out.dtype == torch.float32 and out.is_cuda
        assert out.is_contiguous(), "out must be contiguous"
    if compute_lse:
        if lse_out is None:
            if not _compile_only:
                with _torch_stream_context(current_stream, q.device):
                    lse_out = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
        else:
            assert lse_out.shape == lse_shape, f"lse_out must have shape {lse_shape}, got {tuple(lse_out.shape)}"
            assert lse_out.dtype == torch.float32 and lse_out.is_cuda
            assert lse_out.is_contiguous(), "lse_out must be contiguous"

    # The forward kernel now always writes reduced scores directly from
    # registers to global memory. FP8 adds the best validated qh64 specialization
    # on top of that direct-store path.
    q_tokens_per_tile = 128 // int(qhead_per_kv_head)
    use_unchecked_qh64 = (
        use_fp8_scales
        and qhead_per_kv_head == 64
        and not is_varlen
        and not is_compressed_logits
        and seqlen_q_dim % q_tokens_per_tile == 0
        and out.shape[-1] == seqlen_k_dim
    )
    use_unchecked_qh64_masked = use_unchecked_qh64 and seqlen_k_dim % 64 == 0

    compile_key = (
        device_index,
        capability,
        q.dtype,
        w.dtype,
        int(head_dim),
        int(qhead_per_kv_head),
        int(ratio),
        bool(is_varlen),
        bool(use_unchecked_qh64),
        bool(use_unchecked_qh64_masked),
        bool(compute_lse),
        bool(is_compressed_logits),
        get_broadcast_dims(q),
        get_broadcast_dims(k),
        get_broadcast_dims(w),
        get_broadcast_dims(q_scale) if q_scale is not None else None,
        get_broadcast_dims(k_scale) if k_scale is not None else None,
        q_causal_offsets is not None,
    )
    compiled = _compiled_kernel
    if compiled is None and compile_key not in _compile_cache:
        if is_capturing(current_stream):
            raise ValueError("SM90 indexer forward is not prepared; call plan.compile() before execute() or warm up before CUDA Graph capture")
        q_cute = _to_cute_tensor(q)
        k_cute = _to_cute_tensor(k)
        w_cute = _to_cute_tensor(w)
        q_scale_cute = _to_cute_tensor(q_scale) if q_scale is not None else None
        k_scale_cute = _to_cute_tensor(k_scale) if k_scale is not None else None
        out_cute = _make_i64_cand_buffer_compile_tensor() if is_compressed_logits else _to_cute_tensor(out)
        lse_cute = (fake_cute_tensor(cutlass.Float32, 1) if lse_out is None else _to_cute_tensor(lse_out)) if compute_lse else None
        cu_q_cute = _to_cute_tensor(cu_seqlens_q, leading_dim=0) if is_varlen else None
        cu_k_cute = _to_cute_tensor(cu_seqlens_k, leading_dim=0) if is_varlen else None
        q_offsets_cute = _to_cute_tensor(q_causal_offsets, leading_dim=0) if q_causal_offsets is not None else None
        cand_offsets_cute = _to_cute_tensor(cand_batch_offsets, leading_dim=0) if is_compressed_logits else None
        kernel_obj = IndexerForwardSm90(
            torch2cute_dtype_map[q.dtype],
            torch2cute_dtype_map[w.dtype],
            head_dim=int(head_dim),
            qhead_per_kvhead=int(qhead_per_kv_head),
            ratio=int(ratio),
            is_varlen=is_varlen,
            use_unchecked_qh64=use_unchecked_qh64,
            use_unchecked_qh64_masked=use_unchecked_qh64_masked,
            compute_lse=compute_lse,
            is_compressed_logits=is_compressed_logits,
        )
        gpu_arch = gpu_arch_flag(q.device)
        with torch.cuda.device(q.device):
            _compile_cache[compile_key] = cute.compile(
                kernel_obj,
                q_cute,
                k_cute,
                w_cute,
                q_scale_cute,
                k_scale_cute,
                out_cute,
                lse_cute,
                cutlass.Int32(int(n_heads_kv)),
                cutlass.Int32(int(seqlen_q_dim)),
                cutlass.Int32(int(seqlen_k_dim)),
                cutlass.Float32(float(sm_scale)),
                cu_q_cute,
                cu_k_cute,
                q_offsets_cute,
                cand_offsets_cute,
                current_stream,
                options=compile_options(gpu_arch=gpu_arch),
            )
    if compiled is None:
        compiled = _compile_cache[compile_key]

    if _compile_only:
        return compiled

    if not is_compressed_logits:
        with _torch_stream_context(current_stream, q.device):
            out.fill_(float("-inf"))
    with torch.cuda.device(q.device), torch.cuda.nvtx.range("indexer_fwd_kernel_sm90_direct_dsl"):
        compiled(
            q,
            k,
            w,
            q_scale if use_fp8_scales else None,
            k_scale if use_fp8_scales else None,
            out,
            lse_out if compute_lse else None,
            cutlass.Int32(int(n_heads_kv)),
            cutlass.Int32(int(seqlen_q_dim)),
            cutlass.Int32(int(seqlen_k_dim)),
            cutlass.Float32(float(sm_scale)),
            cu_seqlens_q if is_varlen else None,
            cu_seqlens_k if is_varlen else None,
            q_causal_offsets,
            cand_batch_offsets if is_compressed_logits else None,
            current_stream,
        )
    if compute_lse:
        assert lse_out is not None
        if return_lse:
            return out, lse_out
    return out


__all__ = ["indexer_fwd"]
