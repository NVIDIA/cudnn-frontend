# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper and dispatcher for indexer forward CuTe DSL score kernels.

``indexer_forward_wrapper`` produces dense indexer scores Q @ K^T with
per-head ReLU, weighted head reduction, and a ratio causal mask.
``indexer_forward_top_k_wrapper`` is the SM100 combined path that generates
compact scores, selects Top-K, and optionally computes Top-K softmax without
materializing the dense score tensor.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict

from cudnn.deepseek_sparse_attention.utils.runtime import device_major

from ._interface import indexer_fwd as indexer_fwd_sm100
from ._interface_sm90 import indexer_fwd as indexer_fwd_sm90

TMA_ALIGN_ELEMS = 4  # FP32 output => seqlen_k padded to multiples of 4 (16 B)


class IndexerForward(APIBase):
    """SM100+ BF16 APIBase shell for the shared forward interface.

    The backend interface owns lazy compilation and its kernel cache. Hopper
    dispatch uses the direct SM90 wrapper in ``_interface_sm90.py``.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k: torch.Tensor,  # (B, S_k, H_kv, D) BF16
        sample_w: torch.Tensor,  # (B, S_q, H_q) BF16
        sample_out: torch.Tensor,  # (B, S_q, S_k_padded) FP32
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
    ):
        super().__init__()

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.o_desc = self._make_tensor_desc(sample_out, name="sample_out")

        self.ratio = int(ratio)
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.q_stage = int(q_stage)
        self.kv_stage = int(kv_stage)
        self.sm_scale = float(sm_scale)
        self.qhead_per_kv_head = qhead_per_kv_head

        self.batch_size = None
        self.s_q = None
        self.s_k = None
        self.s_k_padded = None
        self.h_q = None
        self.h_kv = None
        self.head_dim = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        self._value_error_if(
            self.q_desc.ndim != 4,
            f"Q must be 4-D (B, S_q, H_q, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 4,
            f"K must be 4-D (B, S_k, H_kv, D), got {self.k_desc.shape}",
        )
        self._value_error_if(
            self.w_desc.ndim != 3,
            f"W must be 3-D (B, S_q, H_q), got {self.w_desc.shape}",
        )
        self._value_error_if(
            self.o_desc.ndim != 3,
            f"Out must be 3-D (B, S_q, S_k_padded), got {self.o_desc.shape}",
        )

        b, s_q, h_q, d = self.q_desc.shape
        b_k, s_k, h_kv, d_k = self.k_desc.shape
        b_o, s_q_out, s_k_padded_from_out = self.o_desc.shape
        self._value_error_if(b != b_k, f"Batch size mismatch Q={b} vs K={b_k}")
        self._value_error_if(b != b_o, f"Batch size mismatch Q={b} vs Out={b_o}")
        self._value_error_if(s_q != s_q_out, f"S_q mismatch Q={s_q} vs Out={s_q_out}")
        self._value_error_if(d != d_k, f"Head dim mismatch Q={d} vs K={d_k}")
        self._value_error_if(
            d != 128,
            f"IndexerForward is tuned for head_dim=128 only, got {d}",
        )

        qhpkv = self.qhead_per_kv_head if self.qhead_per_kv_head is not None else (h_q // h_kv)
        self._value_error_if(
            qhpkv * h_kv != h_q,
            f"qhead_per_kv_head * h_kv != h_q ({qhpkv} * {h_kv} != {h_q})",
        )
        self._value_error_if(
            qhpkv not in (32, 64),
            f"qhead_per_kv_head must be 32 or 64, got {qhpkv}",
        )
        self.qhead_per_kv_head = qhpkv

        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, torch.bfloat16, name="W")
        self._check_dtype(self.o_desc, torch.float32, name="Out")

        self._value_error_if(
            s_k_padded_from_out % TMA_ALIGN_ELEMS != 0,
            f"Out seqlen_k dim must be a multiple of {TMA_ALIGN_ELEMS}, got {s_k_padded_from_out}",
        )
        self._value_error_if(
            s_k_padded_from_out < s_k,
            f"Out seqlen_k dim ({s_k_padded_from_out}) must cover logical K length ({s_k})",
        )

        major = device_major()
        self._runtime_error_if(
            major < 10,
            f"IndexerForward requires SM100+ compute capability, found SM{major}",
        )

        self.batch_size = b
        self.s_q = s_q
        self.s_k = s_k
        self.s_k_padded = s_k_padded_from_out
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        # The direct interface owns its compile cache and needs real tensors
        # to preserve runtime layouts. Mark this APIBase object ready and let
        # the interface compile lazily on first execute().
        self._compiled_kernel = True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("IndexerForward kernel not compiled")

        # APIBase callers provide a TMA-aligned output allocation, while the
        # shared interface describes only the logical K columns. Passing the
        # logical view lets the interface manage any internal padded buffer
        # without exposing padded columns as scores.
        logical_out = out[..., : self.s_k]
        indexer_fwd_sm100(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=logical_out,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            sm_scale=self.sm_scale,
            precision="bf16",
            current_stream=current_stream,
        )


def indexer_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
) -> TupleDict:
    """High-level wrapper. Allocates the output buffer with TMA padding on S_k.

    Returns ``{'scores': (B, S_q, S_k) FP32}``. The ratio causal mask marks
    positions outside the valid KV range with -inf. ``q_causal_offsets`` may
    specify the global uncompressed token index for each batch/THD segment's
    local q[0].
    """
    if device_major() == 9:
        if cu_seqlens_q_scale_padded is not None or cu_seqlens_k_scale_padded is not None:
            raise NotImplementedError("MXFP8 scale padded cu_seqlens are only supported on SM100+")
        unsupported = []
        if m_block_size != 128:
            unsupported.append(f"m_block_size={m_block_size}")
        if n_block_size != 128:
            unsupported.append(f"n_block_size={n_block_size}")
        if q_stage != 2:
            unsupported.append(f"q_stage={q_stage}")
        if kv_stage != 4:
            unsupported.append(f"kv_stage={kv_stage}")
        if unsupported:
            raise ValueError(
                "SM90 indexer_forward_wrapper only supports default tuning parameters "
                "(m_block_size=128, n_block_size=128, q_stage=2, kv_stage=4); got " + ", ".join(unsupported)
            )
        # Both arches route through their own indexer_fwd wrapper (which owns
        # output allocation + TMA padding); Hopper uses the SM90 variant.
        result = indexer_fwd_sm90(
            q,
            k,
            w,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
            return_lse=return_lse,
            lse_out=lse_out,
            current_stream=stream,
        )
        if return_lse:
            scores, lse = result
            return TupleDict(scores=scores, lse=lse)
        return TupleDict(scores=result)

    if return_lse or lse_out is not None:
        raise NotImplementedError("SM100 dense indexer forward does not expose LSE; " "LSE is produced by dense_indexer_score_recompute_wrapper")

    # BSHD and THD both go through indexer_fwd (it branches on cu_seqlens
    # internally): it owns output allocation + TMA padding and uses a single
    # shape-agnostic compile cache. cu_seqlens_q/k / max_seqlen_q/k are None for
    # BSHD; seqlen is then derived from the tensor shapes at runtime.
    scores = indexer_fwd_sm100(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        out=out,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=384,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        current_stream=stream,
    )
    return TupleDict(scores=scores)


def indexer_forward_top_k_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    top_k: int,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
    return_softmax: Optional[bool] = None,
    softmax_out: Optional[torch.Tensor] = None,
    microbatch_rows: int = -1,
    topk_indices_global: bool = True,
    cand_buffer: Optional[torch.Tensor] = None,
    out_indices: Optional[torch.Tensor] = None,
    out_logits: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    deterministic: bool = False,
) -> TupleDict:
    """Combined SM100 indexer score generation and Top-K selection API.

    Returns ``{'indices', 'logits', 'softmax'}`` by default, plus ``'lse'``
    when requested. Set ``return_softmax=False`` to return only indices and
    logits. Unlike :func:`indexer_forward_wrapper`, this path never
    materializes the dense score tensor.

    BSHD outputs have shape ``(B, S_q, top_k)`` and THD outputs have shape
    ``(total_q, top_k)``. Caller-owned candidate and output buffers may be
    supplied to avoid per-call allocation. Indices are global KV ids by
    default; set ``topk_indices_global=False`` for local ids matching
    ``indexer_top_k_wrapper``, and use the same convention downstream.
    Set ``deterministic=True`` to resolve exact-value ties at the K-th boundary
    toward the smallest local KV indices. This makes the selected set
    reproducible, although output slot order remains unspecified.
    """
    if device_major() < 10:
        raise NotImplementedError("compressed indexer forward is SM100-only; standalone IndexerTopK remains SM90+")

    result = indexer_fwd_sm100(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=384,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        precision=precision,
        q_scale=q_scale,
        k_scale=k_scale,
        cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded=cu_seqlens_k_scale_padded,
        sf_vec_size=sf_vec_size,
        is_compressed_logits=True,
        topk=top_k,
        microbatch_rows=microbatch_rows,
        topk_indices_global=topk_indices_global,
        cand_buffer=cand_buffer,
        out_indices=out_indices,
        out_logits=out_logits,
        cand_batch_offsets=cand_batch_offsets,
        return_lse=return_lse,
        lse_out=lse_out,
        return_softmax=return_softmax,
        softmax_out=softmax_out,
        deterministic=deterministic,
        current_stream=stream,
    )

    want_softmax = (return_softmax is not False) or softmax_out is not None
    want_lse = return_lse or lse_out is not None
    result_iter = iter(result)
    values = {
        "indices": next(result_iter),
        "logits": next(result_iter),
    }
    if want_softmax:
        values["softmax"] = next(result_iter)
    if want_lse:
        values["lse"] = next(result_iter)
    return TupleDict(**values)
