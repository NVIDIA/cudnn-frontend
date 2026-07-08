# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral DeepSeek sparse and dense indexer backward operations."""

from __future__ import annotations

import math
from typing import Any, Optional

from ... import data_type
from ..._op import Op
from ..._tensor_desc import TensorDesc

DEFAULT_BLOCK_I = 128
SUPPORTED_HEAD_DIM = 128
MIN_HEADS = 64


def _require_desc(
    name: str, desc: TensorDesc[Any] | None, *, optional: bool = False
) -> None:
    if desc is None and optional:
        return
    if not isinstance(desc, TensorDesc):
        expected = "a TensorDesc or None" if optional else "a TensorDesc"
        raise TypeError(f"{name} must be {expected}, got {type(desc).__name__}")


def _require_rank(desc: TensorDesc[Any], rank: int, name: str) -> None:
    if desc.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {desc.shape}")


def _require_dtype(
    desc: TensorDesc[Any], expected: data_type | tuple[data_type, ...], name: str
) -> None:
    expected_values = expected if isinstance(expected, tuple) else (expected,)
    if desc.cudnn_dtype not in expected_values:
        expected_text = " or ".join(value.name.lower() for value in expected_values)
        raise ValueError(f"{name} must have dtype {expected_text}, got {desc.dtype}")


def _require_shape(desc: TensorDesc[Any], expected: tuple[int, ...], name: str) -> None:
    if desc.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {desc.shape}")


def _require_compact(desc: TensorDesc[Any]) -> None:
    if not desc.is_compact():
        raise ValueError(
            f"{desc.name or 'tensor'} must be compact, got stride {desc.stride}"
        )


def _require_contiguous_tail(desc: TensorDesc[Any], rank: int) -> None:
    expected_stride = 1
    for axis in range(desc.ndim - 1, desc.ndim - rank - 1, -1):
        if desc.shape[axis] != 1 and desc.stride[axis] != expected_stride:
            raise ValueError(
                f"{desc.name or 'tensor'} must have its final {rank} axes contiguous, "
                f"got shape {desc.shape} and stride {desc.stride}"
            )
        expected_stride *= max(desc.shape[axis], 1)


def _resolve_scale(value: float, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar, got {value!r}") from error
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite, got {resolved}")
    return resolved


class IndexerBackwardOp(Op):
    """Complete fixed BSHD or packed THD sparse backward signature."""

    def __init__(
        self,
        *,
        index_q: TensorDesc[Any],
        weights: TensorDesc[Any],
        index_k: TensorDesc[Any],
        d_index_q: TensorDesc[Any],
        d_weights: TensorDesc[Any],
        d_index_k: TensorDesc[Any],
        attn_score: TensorDesc[Any],
        index_score: TensorDesc[Any],
        topk_indices: TensorDesc[Any],
        sm_scale: float = 1.0,
        block_i: int = DEFAULT_BLOCK_I,
        topk_indices_global: bool = False,
    ) -> None:
        descriptors = (
            ("index_q", index_q),
            ("weights", weights),
            ("index_k", index_k),
            ("d_index_q", d_index_q),
            ("d_weights", d_weights),
            ("d_index_k", d_index_k),
            ("attn_score", attn_score),
            ("index_score", index_score),
            ("topk_indices", topk_indices),
        )
        for name, desc in descriptors:
            _require_desc(name, desc)

        self.index_q = index_q
        self.weights = weights
        self.index_k = index_k
        self.d_index_q = d_index_q
        self.d_weights = d_weights
        self.d_index_k = d_index_k
        self.attn_score = attn_score
        self.index_score = index_score
        self.topk_indices = topk_indices
        self.requested_sm_scale = sm_scale
        self.block_i = int(block_i)
        self.topk_indices_global = bool(topk_indices_global)

        self.is_thd: Optional[bool] = None
        self.batch: Optional[int] = None
        self.seqlen_q: Optional[int] = None
        self.seqlen_k: Optional[int] = None
        self.heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.topk: Optional[int] = None
        self.sm_scale: Optional[float] = None

    def check_support(self) -> bool:
        self.is_thd = None
        self.batch = self.seqlen_q = self.seqlen_k = self.heads = self.head_dim = (
            self.topk
        ) = None
        self.sm_scale = None

        if self.index_q.ndim not in (3, 4):
            raise ValueError(
                f"index_q must use BSHD rank 4 or packed THD rank 3, got shape {self.index_q.shape}"
            )
        is_thd = self.index_q.ndim == 3
        q_rank = 3 if is_thd else 4
        auxiliary_rank = q_rank - 1
        for desc, rank, name in (
            (self.index_q, q_rank, "index_q"),
            (self.weights, auxiliary_rank, "weights"),
            (self.index_k, auxiliary_rank, "index_k"),
            (self.d_index_q, q_rank, "d_index_q"),
            (self.d_weights, auxiliary_rank, "d_weights"),
            (self.d_index_k, auxiliary_rank, "d_index_k"),
            (self.attn_score, auxiliary_rank, "attn_score"),
            (self.index_score, auxiliary_rank, "index_score"),
            (self.topk_indices, auxiliary_rank, "topk_indices"),
        ):
            _require_rank(desc, rank, name)

        for desc, name in (
            (self.index_q, "index_q"),
            (self.weights, "weights"),
            (self.index_k, "index_k"),
        ):
            _require_dtype(desc, data_type.BFLOAT16, name)
        _require_dtype(self.d_index_q, data_type.BFLOAT16, "d_index_q")
        _require_dtype(self.d_weights, data_type.BFLOAT16, "d_weights")
        _require_dtype(
            self.d_index_k, (data_type.BFLOAT16, data_type.FLOAT), "d_index_k"
        )
        _require_dtype(self.attn_score, data_type.FLOAT, "attn_score")
        _require_dtype(self.index_score, data_type.FLOAT, "index_score")
        _require_dtype(self.topk_indices, data_type.INT32, "topk_indices")

        if is_thd:
            seqlen_q, heads, head_dim = self.index_q.shape
            seqlen_k, head_dim_k = self.index_k.shape
            batch = batch_k = 1
            if not self.topk_indices_global:
                raise ValueError(
                    "Packed THD IndexerBackward requires topk_indices_global=True"
                )
        else:
            batch, seqlen_q, heads, head_dim = self.index_q.shape
            batch_k, seqlen_k, head_dim_k = self.index_k.shape
        topk = self.topk_indices.shape[-1]
        dimensions = (batch, seqlen_q, seqlen_k, heads, head_dim, topk)
        if any(value <= 0 for value in dimensions):
            raise ValueError(
                f"Indexer-backward dimensions must be positive, got {dimensions}"
            )
        if batch_k != batch or head_dim_k != head_dim:
            raise ValueError("index_k batch and head dimensions must match index_q")
        if heads < MIN_HEADS:
            raise ValueError(
                f"IndexerBackward requires heads >= {MIN_HEADS}, got {heads}"
            )
        if head_dim != SUPPORTED_HEAD_DIM:
            raise ValueError(
                f"IndexerBackward requires head_dim={SUPPORTED_HEAD_DIM}, got {head_dim}"
            )
        if self.block_i != DEFAULT_BLOCK_I:
            raise ValueError(f"block_i must be {DEFAULT_BLOCK_I}, got {self.block_i}")
        if topk % self.block_i != 0:
            raise ValueError(
                f"topk ({topk}) must be divisible by block_i ({self.block_i})"
            )

        weights_shape = (seqlen_q, heads) if is_thd else (batch, seqlen_q, heads)
        score_shape = (seqlen_q, topk) if is_thd else (batch, seqlen_q, topk)
        _require_shape(self.weights, weights_shape, "weights")
        _require_shape(self.d_index_q, self.index_q.shape, "d_index_q")
        _require_shape(self.d_weights, self.weights.shape, "d_weights")
        _require_shape(self.d_index_k, self.index_k.shape, "d_index_k")
        _require_shape(self.attn_score, score_shape, "attn_score")
        _require_shape(self.index_score, score_shape, "index_score")
        _require_shape(self.topk_indices, score_shape, "topk_indices")
        for desc in (
            self.index_q,
            self.weights,
            self.index_k,
            self.d_index_q,
            self.d_weights,
            self.d_index_k,
            self.attn_score,
            self.index_score,
            self.topk_indices,
        ):
            _require_compact(desc)
        for desc in (self.index_q, self.d_index_q):
            _require_contiguous_tail(desc, 2)
        for desc in (
            self.weights,
            self.index_k,
            self.d_weights,
            self.d_index_k,
            self.attn_score,
            self.index_score,
            self.topk_indices,
        ):
            _require_contiguous_tail(desc, 1)

        self.is_thd = is_thd
        self.batch = batch
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.heads = heads
        self.head_dim = head_dim
        self.topk = topk
        self.sm_scale = _resolve_scale(self.requested_sm_scale, "sm_scale")
        return True


class DenseIndexerBackwardOp(Op):
    """Complete BSHD or packed-THD dense indexer-backward signature."""

    def __init__(
        self,
        *,
        index_q: TensorDesc[Any],
        weights: TensorDesc[Any],
        index_k: TensorDesc[Any],
        d_index_q: TensorDesc[Any],
        d_weights: TensorDesc[Any],
        d_index_k: TensorDesc[Any],
        attn_score: TensorDesc[Any],
        attn_l1norm: TensorDesc[Any],
        index_score: TensorDesc[Any],
        index_lse: TensorDesc[Any],
        cu_seqlens_q: TensorDesc[Any] | None = None,
        cu_seqlens_k: TensorDesc[Any] | None = None,
        q_causal_offsets: TensorDesc[Any] | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        sm_scale: float = 1.0,
        block_i: int = DEFAULT_BLOCK_I,
        ratio: int = 1,
    ) -> None:
        descriptors = (
            ("index_q", index_q),
            ("weights", weights),
            ("index_k", index_k),
            ("d_index_q", d_index_q),
            ("d_weights", d_weights),
            ("d_index_k", d_index_k),
            ("attn_score", attn_score),
            ("attn_l1norm", attn_l1norm),
            ("index_score", index_score),
            ("index_lse", index_lse),
        )
        for name, desc in descriptors:
            _require_desc(name, desc)
        for name, desc in (
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_k", cu_seqlens_k),
            ("q_causal_offsets", q_causal_offsets),
        ):
            _require_desc(name, desc, optional=True)

        self.index_q = index_q
        self.weights = weights
        self.index_k = index_k
        self.d_index_q = d_index_q
        self.d_weights = d_weights
        self.d_index_k = d_index_k
        self.attn_score = attn_score
        self.attn_l1norm = attn_l1norm
        self.index_score = index_score
        self.index_lse = index_lse
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.q_causal_offsets = q_causal_offsets
        self.requested_max_seqlen_q = max_seqlen_q
        self.requested_max_seqlen_k = max_seqlen_k
        self.requested_sm_scale = sm_scale
        self.block_i = int(block_i)
        self.ratio = int(ratio)

        self.is_thd: Optional[bool] = None
        self.batch: Optional[int] = None
        self.normalization_tokens: Optional[int] = None
        self.total_k: Optional[int] = None
        self.heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.max_seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None
        self.sm_scale: Optional[float] = None

    def check_support(self) -> bool:
        self.is_thd = None
        self.batch = self.normalization_tokens = self.total_k = self.heads = (
            self.head_dim
        ) = None
        self.max_seqlen_q = self.max_seqlen_k = None
        self.sm_scale = None

        if (self.cu_seqlens_q is None) != (self.cu_seqlens_k is None):
            raise ValueError("cu_seqlens_q and cu_seqlens_k must be provided together")
        is_thd = self.cu_seqlens_q is not None
        if is_thd:
            batch, normalization_tokens, total_k, heads, head_dim, max_q, max_k = (
                self._check_thd()
            )
        else:
            batch, normalization_tokens, total_k, heads, head_dim, max_q, max_k = (
                self._check_bshd()
            )

        for desc, name in (
            (self.index_q, "index_q"),
            (self.weights, "weights"),
            (self.index_k, "index_k"),
        ):
            _require_dtype(desc, data_type.BFLOAT16, name)
        _require_dtype(self.d_index_q, data_type.BFLOAT16, "d_index_q")
        _require_dtype(self.d_weights, data_type.BFLOAT16, "d_weights")
        _require_dtype(
            self.d_index_k, (data_type.BFLOAT16, data_type.FLOAT), "d_index_k"
        )
        for desc, name in (
            (self.attn_score, "attn_score"),
            (self.attn_l1norm, "attn_l1norm"),
            (self.index_score, "index_score"),
            (self.index_lse, "index_lse"),
        ):
            _require_dtype(desc, data_type.FLOAT, name)
        for desc, name in (
            (self.cu_seqlens_q, "cu_seqlens_q"),
            (self.cu_seqlens_k, "cu_seqlens_k"),
            (self.q_causal_offsets, "q_causal_offsets"),
        ):
            if desc is not None:
                _require_dtype(desc, data_type.INT32, name)

        if heads < MIN_HEADS:
            raise ValueError(
                f"DenseIndexerBackward requires heads >= {MIN_HEADS}, got {heads}"
            )
        if head_dim != SUPPORTED_HEAD_DIM:
            raise ValueError(
                f"DenseIndexerBackward requires head_dim={SUPPORTED_HEAD_DIM}, got {head_dim}"
            )
        if self.block_i != DEFAULT_BLOCK_I:
            raise ValueError(f"block_i must be {DEFAULT_BLOCK_I}, got {self.block_i}")
        if self.ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {self.ratio}")

        for desc in (
            self.index_q,
            self.weights,
            self.index_k,
            self.d_index_q,
            self.d_weights,
            self.d_index_k,
            self.attn_score,
            self.attn_l1norm,
            self.index_score,
            self.index_lse,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.q_causal_offsets,
        ):
            if desc is not None:
                _require_compact(desc)
        for desc in (self.index_q, self.d_index_q):
            _require_contiguous_tail(desc, 2)
        for desc in (
            self.weights,
            self.index_k,
            self.d_weights,
            self.d_index_k,
            self.attn_score,
            self.index_score,
        ):
            _require_contiguous_tail(desc, 1)

        self.is_thd = is_thd
        self.batch = batch
        self.normalization_tokens = normalization_tokens
        self.total_k = total_k
        self.heads = heads
        self.head_dim = head_dim
        self.max_seqlen_q = max_q
        self.max_seqlen_k = max_k
        self.sm_scale = _resolve_scale(self.requested_sm_scale, "sm_scale")
        return True

    def _check_bshd(self) -> tuple[int, int, int, int, int, int, int]:
        for desc, rank, name in (
            (self.index_q, 4, "index_q"),
            (self.weights, 3, "weights"),
            (self.index_k, 3, "index_k"),
            (self.d_index_q, 4, "d_index_q"),
            (self.d_weights, 3, "d_weights"),
            (self.d_index_k, 3, "d_index_k"),
            (self.attn_score, 3, "attn_score"),
            (self.attn_l1norm, 2, "attn_l1norm"),
            (self.index_score, 3, "index_score"),
            (self.index_lse, 2, "index_lse"),
        ):
            _require_rank(desc, rank, name)

        batch, seqlen_q, heads, head_dim = self.index_q.shape
        batch_k, seqlen_k, head_dim_k = self.index_k.shape
        if min(batch, seqlen_q, seqlen_k, heads, head_dim) <= 0:
            raise ValueError("Dense indexer-backward dimensions must be positive")
        if batch_k != batch or head_dim_k != head_dim:
            raise ValueError("index_k batch and head dimensions must match index_q")
        _require_shape(self.weights, (batch, seqlen_q, heads), "weights")
        _require_shape(self.d_index_q, self.index_q.shape, "d_index_q")
        _require_shape(self.d_weights, self.weights.shape, "d_weights")
        _require_shape(self.d_index_k, self.index_k.shape, "d_index_k")
        score_shape = (batch, seqlen_q, seqlen_k)
        denom_shape = (batch, seqlen_q)
        _require_shape(self.attn_score, score_shape, "attn_score")
        _require_shape(self.index_score, score_shape, "index_score")
        _require_shape(self.attn_l1norm, denom_shape, "attn_l1norm")
        _require_shape(self.index_lse, denom_shape, "index_lse")
        if self.q_causal_offsets is not None:
            _require_rank(self.q_causal_offsets, 1, "q_causal_offsets")
            _require_shape(self.q_causal_offsets, (batch,), "q_causal_offsets")

        max_q = (
            seqlen_q
            if self.requested_max_seqlen_q is None
            else int(self.requested_max_seqlen_q)
        )
        max_k = (
            seqlen_k
            if self.requested_max_seqlen_k is None
            else int(self.requested_max_seqlen_k)
        )
        if (max_q, max_k) != (seqlen_q, seqlen_k):
            raise ValueError(
                f"BSHD max_seqlen_q/k must match tensor extents {(seqlen_q, seqlen_k)}, got {(max_q, max_k)}"
            )
        return batch, batch * seqlen_q, batch * seqlen_k, heads, head_dim, max_q, max_k

    def _check_thd(self) -> tuple[int, int, int, int, int, int, int]:
        for desc, rank, name in (
            (self.index_q, 3, "index_q"),
            (self.weights, 2, "weights"),
            (self.index_k, 2, "index_k"),
            (self.d_index_q, 3, "d_index_q"),
            (self.d_weights, 2, "d_weights"),
            (self.d_index_k, 2, "d_index_k"),
            (self.attn_score, 2, "attn_score"),
            (self.attn_l1norm, 1, "attn_l1norm"),
            (self.index_score, 2, "index_score"),
            (self.index_lse, 1, "index_lse"),
            (self.cu_seqlens_q, 1, "cu_seqlens_q"),
            (self.cu_seqlens_k, 1, "cu_seqlens_k"),
        ):
            _require_rank(desc, rank, name)

        total_q, heads, head_dim = self.index_q.shape
        total_k, head_dim_k = self.index_k.shape
        batch = self.cu_seqlens_q.shape[0] - 1
        if min(total_q, total_k, batch, heads, head_dim) <= 0:
            raise ValueError("Dense THD indexer-backward dimensions must be positive")
        if self.cu_seqlens_k.shape != (batch + 1,):
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must describe the same batch"
            )
        if head_dim_k != head_dim:
            raise ValueError("index_k head dimension must match index_q")
        if self.requested_max_seqlen_q is None or self.requested_max_seqlen_k is None:
            raise ValueError(
                "THD dense indexer backward requires max_seqlen_q and max_seqlen_k"
            )
        max_q = int(self.requested_max_seqlen_q)
        max_k = int(self.requested_max_seqlen_k)
        if max_q <= 0 or max_k <= 0:
            raise ValueError(f"max_seqlen_q/k must be positive, got {(max_q, max_k)}")

        _require_shape(self.weights, (total_q, heads), "weights")
        _require_shape(self.d_index_q, self.index_q.shape, "d_index_q")
        _require_shape(self.d_weights, self.weights.shape, "d_weights")
        _require_shape(self.d_index_k, self.index_k.shape, "d_index_k")
        score_shape = (total_q, max_k)
        _require_shape(self.attn_score, score_shape, "attn_score")
        _require_shape(self.index_score, score_shape, "index_score")
        _require_shape(self.attn_l1norm, (total_q,), "attn_l1norm")
        _require_shape(self.index_lse, (total_q,), "index_lse")
        if self.q_causal_offsets is not None:
            _require_rank(self.q_causal_offsets, 1, "q_causal_offsets")
            _require_shape(self.q_causal_offsets, (batch,), "q_causal_offsets")
        return batch, total_q, total_k, heads, head_dim, max_q, max_k


__all__ = [
    "DEFAULT_BLOCK_I",
    "DenseIndexerBackwardOp",
    "IndexerBackwardOp",
    "MIN_HEADS",
    "SUPPORTED_HEAD_DIM",
]
