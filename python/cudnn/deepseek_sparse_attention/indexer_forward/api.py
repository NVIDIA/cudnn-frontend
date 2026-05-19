"""APIBase wrapper for IndexerForwardSm100 (CuTe DSL score kernel).

Produces dense indexer scores Q @ K^T with per-head ReLU, weighted head
reduction, and a ratio causal mask. Does NOT fuse top-K — pair with
:class:`cudnn.deepseek_sparse_attention.indexer_top_k.IndexerTopK` for a
two-stage unfused path (score → top-K).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from cudnn.api_base import APIBase, TupleDict

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import device_major, resolve_stream

from .indexer_fwd_sm100 import IndexerForwardSm100
from ._interface import indexer_fwd

TMA_ALIGN_ELEMS = 4  # FP32 output => seqlen_k padded to multiples of 4 (16 B)


class IndexerForward(APIBase):
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
        self._kernel = IndexerForwardSm100

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
        if self._compiled_kernel is not None:
            return

        kernel_obj = self._kernel(
            head_dim=self.head_dim,
            qhead_per_kvhead=self.qhead_per_kv_head,
            ratio=self.ratio,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
        )

        fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)

        _compiled_kernel = cute.compile(
            kernel_obj,
            self._make_fake_cute_tensor_from_desc(self.q_desc, assumed_align=16),
            self._make_fake_cute_tensor_from_desc(self.k_desc, assumed_align=16),
            self._make_fake_cute_tensor_from_desc(self.w_desc, assumed_align=16),
            self._make_fake_cute_tensor_from_desc(self.o_desc, assumed_align=16),
            self.h_kv,
            cutlass.Int32(self.s_q),
            cutlass.Int32(self.s_k),
            cutlass.Float32(self.sm_scale),
            None,
            None,
            fake_stream,
            options=compile_options(),
        )

        def tensor_api(q, k, w, out, stream):
            # The kernel only writes to tiles it actually computes. Callers
            # that depend on -inf in skipped positions should ensure the output
            # was filled to -inf before invoking (the wrapper does this).
            return _compiled_kernel(
                q,
                k,
                w,
                out,
                self.h_kv,
                cutlass.Int32(self.s_q),
                cutlass.Int32(self.s_k),
                cutlass.Float32(self.sm_scale),
                None,
                None,
                stream,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = resolve_stream(current_stream)
        if self._compiled_kernel is None:
            raise ValueError("IndexerForward kernel not compiled")
        self._compiled_kernel(q, k, w, out, current_stream)


_logger = logging.getLogger(__name__)
_cache_of_IndexerForwardObjects: dict = {}


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
    stream: Optional[cuda.CUstream] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
) -> TupleDict:
    """High-level wrapper. Allocates the output buffer with TMA padding on S_k.

    Returns ``{'scores': (B, S_q, S_k) FP32}``. The ratio causal mask marks
    positions outside the valid KV range with -inf.
    """
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        scores = indexer_fwd(
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
        )
        return TupleDict(scores=scores)

    b, s_q, h_q, d = q.shape
    _, s_k, h_kv, _ = k.shape
    device = q.device

    s_k_padded = (s_k + TMA_ALIGN_ELEMS - 1) // TMA_ALIGN_ELEMS * TMA_ALIGN_ELEMS
    needs_view = s_k_padded != s_k
    if needs_view:
        out_buf = torch.empty((b, s_q, s_k_padded), dtype=torch.float32, device=device)
        out_padded = out_buf
    else:
        out_buf = torch.empty((b, s_q, s_k), dtype=torch.float32, device=device)
        out_padded = out_buf

    # -inf fill so that skipped n-blocks and masked positions are -inf.
    out_buf.fill_(float("-inf"))

    if qhead_per_kv_head is None:
        qhead_per_kv_head = h_q // h_kv

    cache_key = (
        q.dtype,
        k.dtype,
        w.dtype,
        b,
        s_q,
        s_k_padded,
        h_q,
        h_kv,
        d,
        q.stride(),
        k.stride(),
        w.stride(),
        out_buf.stride(),
        int(ratio),
        int(qhead_per_kv_head),
        int(m_block_size),
        int(n_block_size),
        int(q_stage),
        int(kv_stage),
        float(sm_scale),
    )

    obj = _cache_of_IndexerForwardObjects.get(cache_key)
    if obj is None:
        obj = IndexerForward(
            sample_q=q,
            sample_k=k,
            sample_w=w,
            sample_out=out_padded,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            kv_stage=kv_stage,
            sm_scale=sm_scale,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerForwardObjects[cache_key] = obj

    obj.execute(q, k, w, out_padded, current_stream=stream)

    scores = out_buf[:, :, :s_k] if needs_view else out_buf
    return TupleDict(scores=scores)
