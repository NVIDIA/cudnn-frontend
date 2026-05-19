"""APIBase wrappers for the four DSA score-recompute operations.

Wraps the SM100 and SM90 CuTe-DSL score kernels. Each backend provides
indexer and attention score variants, giving four public classes.

Tile / SMEM dispatch logic and compile caching live in the backend interface
modules. This module adapts those entry points to the APIBase contract.
"""

from __future__ import annotations

from typing import Optional

import torch
import cuda.bindings.driver as cuda

from cudnn.api_base import APIBase, TupleDict

from . import _interface_sm100 as _iface_sm100

# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


def _check_score_arch(api: APIBase) -> None:
    major, _ = torch.cuda.get_device_capability()
    api._runtime_error_if(
        major != 9 and major < 10,
        f"{type(api).__name__} requires SM90 or SM100+ compute capability, found SM{major}",
    )


class _ScoreRecomputeBase(APIBase):
    """Common APIBase shell for score-recompute ops.

    ``_interface.py`` owns compile caches keyed per-kernel; ``check_support``
    and ``compile`` are thin markers that simply gate invocation. ``execute``
    delegates to the relevant ``_iface`` entry point.
    """

    def __init__(self):
        super().__init__()

    def compile(self) -> None:
        self._ensure_support_checked()
        # The score backends compile from real execute-time tensors because the
        # compile key includes concrete layouts and some paths allocate temporary
        # tensors. Using fake tensors here would turn compile() into a hidden
        # kernel launch, so this API validates eagerly and lets the backend cache
        # compile on first execute().
        self._compiled_kernel = True


def _check_sparse_score_shapes(
    api: APIBase,
    q_desc,
    k_desc,
    aux_desc,
    topk_desc,
    out_desc,
    topk_length_desc,
    aux_name: str,
    qhead_per_kv_head: Optional[int],
) -> None:
    api._value_error_if(q_desc.ndim != 4, f"Q must be 4-D (B, S_q, H_q, D), got {q_desc.shape}")
    api._value_error_if(k_desc.ndim != 3, f"K must be 3-D (B, S_k, D) MQA, got {k_desc.shape}")
    api._value_error_if(aux_desc.ndim != 3, f"{aux_name} must be 3-D (B, S_q, H_q), got {aux_desc.shape}")
    api._value_error_if(topk_desc.ndim != 3, f"topk_indices must be 3-D (B, S_q, topk), got {topk_desc.shape}")
    api._value_error_if(out_desc.ndim != 3, f"out must be 3-D (B, S_q, topk), got {out_desc.shape}")

    b, s_q, h_q, d = q_desc.shape
    api._value_error_if(k_desc.shape[0] != b, f"K batch dim {k_desc.shape[0]} must match Q batch dim {b}")
    api._value_error_if(k_desc.shape[2] != d, f"K head dim {k_desc.shape[2]} must match Q head dim {d}")
    api._value_error_if(aux_desc.shape != (b, s_q, h_q), f"{aux_name} shape must be {(b, s_q, h_q)}, got {aux_desc.shape}")
    api._value_error_if(topk_desc.shape[:2] != (b, s_q), f"topk_indices leading dims must be {(b, s_q)}, got {topk_desc.shape[:2]}")
    api._value_error_if(out_desc.shape != topk_desc.shape, f"out shape must match topk_indices shape {topk_desc.shape}, got {out_desc.shape}")
    if qhead_per_kv_head is not None:
        api._value_error_if(qhead_per_kv_head != h_q, f"qhead_per_kv_head must equal H_q ({h_q}) for MQA sparse score, got {qhead_per_kv_head}")
    if topk_length_desc is not None:
        api._check_dtype(topk_length_desc, torch.int32, name="topk_length")
        api._value_error_if(topk_length_desc.shape != (b, s_q), f"topk_length must be shape {(b, s_q)}, got {topk_length_desc.shape}")


def _check_dense_score_shapes(
    api: APIBase,
    q_desc,
    k_desc,
    aux_desc,
    out_desc,
    denom_desc,
    aux_name: str,
    is_thd: bool,
    qhead_per_kv_head: Optional[int],
) -> None:
    if is_thd:
        api._value_error_if(q_desc.ndim != 3, f"THD Q must be 3-D (total_q, H_q, D), got {q_desc.shape}")
        api._value_error_if(k_desc.ndim != 3, f"THD K must be 3-D (total_k, H_kv, D), got {k_desc.shape}")
        api._value_error_if(aux_desc.ndim != 2, f"THD {aux_name} must be 2-D (total_q, H_q), got {aux_desc.shape}")
        api._value_error_if(out_desc.ndim != 2, f"THD out must be 2-D (total_q, max_seqlen_k), got {out_desc.shape}")
        api._value_error_if(denom_desc.ndim != 1, f"THD denom_out must be 1-D (total_q,), got {denom_desc.shape}")
        total_q, h_q, d = q_desc.shape
        _, h_kv, d_k = k_desc.shape
        api._value_error_if(aux_desc.shape != (total_q, h_q), f"THD {aux_name} shape must be {(total_q, h_q)}, got {aux_desc.shape}")
        api._value_error_if(out_desc.shape[0] != total_q, f"THD out first dim must be total_q ({total_q}), got {out_desc.shape[0]}")
        api._value_error_if(denom_desc.shape != (total_q,), f"THD denom_out shape must be {(total_q,)}, got {denom_desc.shape}")
    else:
        api._value_error_if(q_desc.ndim != 4, f"BSHD Q must be 4-D (B, S_q, H_q, D), got {q_desc.shape}")
        api._value_error_if(k_desc.ndim != 4, f"BSHD K must be 4-D (B, S_k, H_kv, D), got {k_desc.shape}")
        api._value_error_if(aux_desc.ndim != 3, f"BSHD {aux_name} must be 3-D (B, S_q, H_q), got {aux_desc.shape}")
        api._value_error_if(out_desc.ndim != 3, f"BSHD out must be 3-D (B, S_q, S_k), got {out_desc.shape}")
        api._value_error_if(denom_desc.ndim != 2, f"BSHD denom_out must be 2-D (B, S_q), got {denom_desc.shape}")
        b, s_q, h_q, d = q_desc.shape
        b_k, s_k, h_kv, d_k = k_desc.shape
        api._value_error_if(b_k != b, f"K batch dim {b_k} must match Q batch dim {b}")
        api._value_error_if(aux_desc.shape != (b, s_q, h_q), f"{aux_name} shape must be {(b, s_q, h_q)}, got {aux_desc.shape}")
        api._value_error_if(out_desc.shape != (b, s_q, s_k), f"out shape must be {(b, s_q, s_k)}, got {out_desc.shape}")
        api._value_error_if(denom_desc.shape != (b, s_q), f"denom_out shape must be {(b, s_q)}, got {denom_desc.shape}")
    api._value_error_if(d_k != d, f"K head dim {d_k} must match Q head dim {d}")
    api._value_error_if(h_kv <= 0 or h_q % h_kv != 0, f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")
    if qhead_per_kv_head is not None:
        api._value_error_if(h_q != h_kv * qhead_per_kv_head, f"H_q ({h_q}) must equal H_kv ({h_kv}) * qhead_per_kv_head ({qhead_per_kv_head})")


# ---------------------------------------------------------------------------
# Sparse indexer score
# ---------------------------------------------------------------------------


class SparseIndexerScoreRecompute(_ScoreRecomputeBase):
    """Sparse indexer score recompute.

    Computes per-query ``softmax( sum_h ReLU(Q_h · K_topk^T) * W_h )`` over
    the top-K KV positions given by ``topk_indices``.
    """

    def __init__(
        self,
        sample_q_indexer: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k_indexer: torch.Tensor,  # (B, S_k, D) BF16 (MQA)
        sample_weights: torch.Tensor,  # (B, S_q, H_q) BF16
        sample_topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
        sample_out: torch.Tensor,  # (B, S_q, topk) FP32
        sample_topk_length: Optional[torch.Tensor] = None,  # (B, S_q) INT32
        qhead_per_kv_head: Optional[int] = None,
        topk_indices_global: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q_indexer, name="sample_q_indexer")
        self.k_desc = self._make_tensor_desc(sample_k_indexer, name="sample_k_indexer")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.topk_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, torch.bfloat16, name="W")
        self._check_dtype(self.topk_desc, torch.int32, name="topk_indices")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        _check_sparse_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.w_desc,
            self.topk_desc,
            self.out_desc,
            self.topk_length_desc,
            "W",
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q_indexer: torch.Tensor,
        k_indexer: torch.Tensor,
        weights: torch.Tensor,
        topk_indices: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> torch.Tensor:
        major, _ = torch.cuda.get_device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.sparse_indexer_score_recompute(
                q_indexer,
                k_indexer,
                weights,
                topk_indices,
                out=out,
                topk_length=topk_length,
                topk_indices_global=self.topk_indices_global,
                current_stream=current_stream,
            )
        return _iface_sm100.sparse_indexer_score_recompute(
            q_indexer,
            k_indexer,
            weights,
            topk_indices,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_indices_global=self.topk_indices_global,
            out=out,
            topk_length=topk_length,
            current_stream=current_stream,
        )


_cache_of_SparseIndexerScoreRecomputeObjects: dict = {}


def sparse_indexer_score_recompute_wrapper(
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = False,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'predict': (B, S_q, topk) FP32}``.

    ``topk_indices`` are per-batch local KV ids by default. Set
    ``topk_indices_global=True`` when passing ids encoded as
    ``batch_idx * S_k + local_idx``.
    """
    key = (
        q_indexer.dtype,
        q_indexer.shape,
        k_indexer.shape,
        weights.shape,
        topk_indices.shape,
        q_indexer.stride(),
        k_indexer.stride(),
        weights.stride(),
        topk_indices.stride(),
        qhead_per_kv_head,
        topk_length is not None,
        bool(topk_indices_global),
    )
    obj = _cache_of_SparseIndexerScoreRecomputeObjects.get(key)
    if obj is None:
        if out is None:
            b, s_q, _ = topk_indices.shape if topk_indices.ndim == 3 else (0, 0, 0)
            topk = topk_indices.shape[-1]
            out_sample = torch.empty(
                (q_indexer.shape[0], q_indexer.shape[1], topk),
                dtype=torch.float32,
                device=q_indexer.device,
            )
        else:
            out_sample = out
        obj = SparseIndexerScoreRecompute(
            sample_q_indexer=q_indexer,
            sample_k_indexer=k_indexer,
            sample_weights=weights,
            sample_topk_indices=topk_indices,
            sample_out=out_sample,
            sample_topk_length=topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseIndexerScoreRecomputeObjects[key] = obj

    predict = obj.execute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        out=out,
        topk_length=topk_length,
        current_stream=stream,
    )
    return TupleDict(predict=predict)


# ---------------------------------------------------------------------------
# Sparse attention score
# ---------------------------------------------------------------------------


class SparseAttnScoreRecompute(_ScoreRecomputeBase):
    """Sparse attention score recompute.

    Recovers per-head softmax from ``LSE``, sums across heads, and L1-normalizes
    over the top-K KV positions:
    ``target[b,q,i] = (sum_h exp(Q_h·K_topk^T·scale - LSE_h)) / sum_i(...)``.
    """

    def __init__(
        self,
        sample_q_attn: torch.Tensor,  # (B, S_q, H_q, D) BF16
        sample_k_attn: torch.Tensor,  # (B, S_k, D) BF16
        sample_lse: torch.Tensor,  # (B, S_q, H_q) FP32
        sample_topk_indices: torch.Tensor,  # (B, S_q, topk) INT32
        sample_out: torch.Tensor,  # (B, S_q, topk) FP32
        softmax_scale: float,
        sample_topk_length: Optional[torch.Tensor] = None,
        qhead_per_kv_head: Optional[int] = None,
        topk_indices_global: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q_attn, name="sample_q_attn")
        self.k_desc = self._make_tensor_desc(sample_k_attn, name="sample_k_attn")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.topk_desc = self._make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = float(softmax_scale)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.topk_desc, torch.int32, name="topk_indices")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        _check_sparse_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            self.topk_desc,
            self.out_desc,
            self.topk_length_desc,
            "LSE",
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q_attn: torch.Tensor,
        k_attn: torch.Tensor,
        lse: torch.Tensor,
        topk_indices: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ) -> torch.Tensor:
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        major, _ = torch.cuda.get_device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.sparse_attn_score_recompute(
                q_attn,
                k_attn,
                lse,
                topk_indices,
                scale,
                out=out,
                topk_length=topk_length,
                topk_indices_global=self.topk_indices_global,
                current_stream=current_stream,
            )
        return _iface_sm100.sparse_attn_score_recompute(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_indices_global=self.topk_indices_global,
            out=out,
            topk_length=topk_length,
            current_stream=current_stream,
        )


_cache_of_SparseAttnScoreRecomputeObjects: dict = {}


def sparse_attn_score_recompute_wrapper(
    q_attn: torch.Tensor,
    k_attn: torch.Tensor,
    lse: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[torch.Tensor] = None,
    topk_indices_global: bool = False,
    out: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """High-level wrapper. Returns ``{'target': (B, S_q, topk) FP32}``.

    ``topk_indices`` are per-batch local KV ids by default. Set
    ``topk_indices_global=True`` when passing ids encoded as
    ``batch_idx * S_k + local_idx``.
    """
    key = (
        q_attn.dtype,
        q_attn.shape,
        k_attn.shape,
        lse.shape,
        topk_indices.shape,
        q_attn.stride(),
        k_attn.stride(),
        lse.stride(),
        topk_indices.stride(),
        qhead_per_kv_head,
        topk_length is not None,
        bool(topk_indices_global),
        float(softmax_scale),
    )
    obj = _cache_of_SparseAttnScoreRecomputeObjects.get(key)
    if obj is None:
        topk = topk_indices.shape[-1]
        out_sample = (
            out
            if out is not None
            else torch.empty(
                (q_attn.shape[0], q_attn.shape[1], topk),
                dtype=torch.float32,
                device=q_attn.device,
            )
        )
        obj = SparseAttnScoreRecompute(
            sample_q_attn=q_attn,
            sample_k_attn=k_attn,
            sample_lse=lse,
            sample_topk_indices=topk_indices,
            sample_out=out_sample,
            softmax_scale=softmax_scale,
            sample_topk_length=topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_SparseAttnScoreRecomputeObjects[key] = obj

    target = obj.execute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        out=out,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        current_stream=stream,
    )
    return TupleDict(target=target)


# ---------------------------------------------------------------------------
# Dense indexer score
# ---------------------------------------------------------------------------


def _uses_thd(cu_seqlens_q: Optional[torch.Tensor], cu_seqlens_k: Optional[torch.Tensor]) -> bool:
    if (cu_seqlens_q is None) != (cu_seqlens_k is None):
        raise ValueError("THD dense score requires both cu_seqlens_q and cu_seqlens_k")
    return cu_seqlens_q is not None


def _max_from_cu_seqlens(cu_seqlens: torch.Tensor, name: str) -> int:
    if cu_seqlens.ndim != 1:
        raise ValueError(f"{name} must be a 1D cumulative sequence length tensor")
    if cu_seqlens.numel() <= 1:
        return 0
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def _dense_sample_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
) -> tuple[bool, int, int, tuple[int, ...], tuple[int, ...]]:
    is_thd = _uses_thd(cu_seqlens_q, cu_seqlens_k)
    if is_thd:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("THD dense score expects q/k with shape (total, heads, dim)")
        max_q = int(max_seqlen_q) if max_seqlen_q is not None else _max_from_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
        max_k = int(max_seqlen_k) if max_seqlen_k is not None else _max_from_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
        return True, max_q, max_k, (q.shape[0], max_k), (q.shape[0],)

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("BSHD dense score expects q/k with shape (B, S, H, D)")
    return False, q.shape[1], k.shape[1], (q.shape[0], q.shape[1], k.shape[1]), (q.shape[0], q.shape[1])


class DenseIndexerScoreRecompute(_ScoreRecomputeBase):
    """Dense indexer score recompute over full KV."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_weights: torch.Tensor,
        sample_out: torch.Tensor,
        sample_denom_out: torch.Tensor,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        ratio: int = 1,
        is_thd: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_weights, name="sample_weights")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.denom_desc = self._make_tensor_desc(sample_denom_out, name="sample_denom_out")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.sm_scale = float(sm_scale)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.w_desc, torch.bfloat16, name="W")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        self._check_dtype(self.denom_desc, torch.float32, name="denom_out")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        _check_dense_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.w_desc,
            self.out_desc,
            self.denom_desc,
            "W",
            self.is_thd,
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        denom_out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        ratio: Optional[int] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ):
        scale = self.sm_scale if sm_scale is None else float(sm_scale)
        ratio_value = self.ratio if ratio is None else int(ratio)
        if cu_seqlens_q is not None:
            if max_seqlen_q is None:
                max_seqlen_q = _max_from_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
            if max_seqlen_k is None:
                max_seqlen_k = _max_from_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
        major, _ = torch.cuda.get_device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.dense_indexer_score_recompute(
                q,
                k,
                weights,
                out=out,
                denom_out=denom_out,
                sm_scale=scale,
                ratio=ratio_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                current_stream=current_stream,
            )
        return _iface_sm100.dense_indexer_score_recompute(
            q,
            k,
            weights,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            sm_scale=scale,
            ratio=ratio_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            current_stream=current_stream,
        )


_cache_of_DenseIndexerScoreRecomputeObjects: dict = {}


def dense_indexer_score_recompute_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    is_thd, max_q, max_k, out_shape, denom_shape = _dense_sample_shapes(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    key = (
        q.dtype,
        q.shape,
        k.shape,
        weights.shape,
        q.stride(),
        k.stride(),
        weights.stride(),
        qhead_per_kv_head,
        float(sm_scale),
        int(ratio),
        is_thd,
        max_q,
        max_k,
        tuple(cu_seqlens_q.shape) if cu_seqlens_q is not None else None,
        tuple(cu_seqlens_k.shape) if cu_seqlens_k is not None else None,
    )
    obj = _cache_of_DenseIndexerScoreRecomputeObjects.get(key)
    if obj is None:
        out_sample = (
            out
            if out is not None
            else torch.empty(
                out_shape,
                dtype=torch.float32,
                device=q.device,
            )
        )
        denom_sample = (
            denom_out
            if denom_out is not None
            else torch.empty(
                denom_shape,
                dtype=torch.float32,
                device=q.device,
            )
        )
        obj = DenseIndexerScoreRecompute(
            sample_q=q,
            sample_k=k,
            sample_weights=weights,
            sample_out=out_sample,
            sample_denom_out=denom_sample,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=sm_scale,
            ratio=ratio,
            is_thd=is_thd,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_DenseIndexerScoreRecomputeObjects[key] = obj

    o, d = obj.execute(
        q,
        k,
        weights,
        out=out,
        denom_out=denom_out,
        sm_scale=sm_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q if is_thd else max_seqlen_q,
        max_seqlen_k=max_k if is_thd else max_seqlen_k,
        current_stream=stream,
    )
    return TupleDict(out=o, denom=d)


# ---------------------------------------------------------------------------
# Dense attention score
# ---------------------------------------------------------------------------


class DenseAttnScoreRecompute(_ScoreRecomputeBase):
    """Dense attention score recompute over full KV."""

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_lse: torch.Tensor,
        sample_out: torch.Tensor,
        sample_denom_out: torch.Tensor,
        softmax_scale: float,
        qhead_per_kv_head: Optional[int] = None,
        ratio: int = 1,
        is_thd: bool = False,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.lse_desc = self._make_tensor_desc(sample_lse, name="sample_lse")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.denom_desc = self._make_tensor_desc(sample_denom_out, name="sample_denom_out")
        self.softmax_scale = float(softmax_scale)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)

    def check_support(self) -> bool:
        _check_score_arch(self)
        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(self.lse_desc, torch.float32, name="LSE")
        self._check_dtype(self.out_desc, torch.float32, name="out")
        self._check_dtype(self.denom_desc, torch.float32, name="denom_out")
        self._value_error_if(self.ratio < 1, f"ratio must be >= 1, got {self.ratio}")
        _check_dense_score_shapes(
            self,
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            self.out_desc,
            self.denom_desc,
            "LSE",
            self.is_thd,
            self.qhead_per_kv_head,
        )
        self._is_supported = True
        return True

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        lse: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        denom_out: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        ratio: Optional[int] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        current_stream: Optional[cuda.CUstream] = None,
    ):
        scale = self.softmax_scale if softmax_scale is None else float(softmax_scale)
        ratio_value = self.ratio if ratio is None else int(ratio)
        if cu_seqlens_q is not None:
            if max_seqlen_q is None:
                max_seqlen_q = _max_from_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
            if max_seqlen_k is None:
                max_seqlen_k = _max_from_cu_seqlens(cu_seqlens_k, "cu_seqlens_k")
        major, _ = torch.cuda.get_device_capability()
        if major == 9:
            from . import _interface_sm90 as _iface_sm90

            return _iface_sm90.dense_attn_score_recompute(
                q,
                k,
                lse,
                scale,
                out=out,
                denom_out=denom_out,
                ratio=ratio_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                current_stream=current_stream,
            )
        return _iface_sm100.dense_attn_score_recompute(
            q,
            k,
            lse,
            scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=out,
            denom_out=denom_out,
            ratio=ratio_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            current_stream=current_stream,
        )


_cache_of_DenseAttnScoreRecomputeObjects: dict = {}


def dense_attn_score_recompute_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    denom_out: Optional[torch.Tensor] = None,
    ratio: int = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    is_thd, max_q, max_k, out_shape, denom_shape = _dense_sample_shapes(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    key = (
        q.dtype,
        q.shape,
        k.shape,
        lse.shape,
        q.stride(),
        k.stride(),
        lse.stride(),
        qhead_per_kv_head,
        float(softmax_scale),
        int(ratio),
        is_thd,
        max_q,
        max_k,
        tuple(cu_seqlens_q.shape) if cu_seqlens_q is not None else None,
        tuple(cu_seqlens_k.shape) if cu_seqlens_k is not None else None,
    )
    obj = _cache_of_DenseAttnScoreRecomputeObjects.get(key)
    if obj is None:
        out_sample = (
            out
            if out is not None
            else torch.empty(
                out_shape,
                dtype=torch.float32,
                device=q.device,
            )
        )
        denom_sample = (
            denom_out
            if denom_out is not None
            else torch.empty(
                denom_shape,
                dtype=torch.float32,
                device=q.device,
            )
        )
        obj = DenseAttnScoreRecompute(
            sample_q=q,
            sample_k=k,
            sample_lse=lse,
            sample_out=out_sample,
            sample_denom_out=denom_sample,
            softmax_scale=softmax_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            ratio=ratio,
            is_thd=is_thd,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_DenseAttnScoreRecomputeObjects[key] = obj

    o, d = obj.execute(
        q,
        k,
        lse,
        out=out,
        denom_out=denom_out,
        softmax_scale=softmax_scale,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q if is_thd else max_seqlen_q,
        max_seqlen_k=max_k if is_thd else max_seqlen_k,
        current_stream=stream,
    )
    return TupleDict(out=o, denom=d)
