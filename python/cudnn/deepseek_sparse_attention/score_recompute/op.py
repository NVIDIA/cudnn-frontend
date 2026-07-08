# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral signatures for DSA score-recompute operations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from operator import index
from typing import Any

from ... import data_type
from ...common.op import Op
from ...common.tensor_desc import TensorDesc
from .config import (
    DenseScoreKernelConfig,
    SparseScoreKernelConfig,
    resolve_dense_score_kernel_config,
    resolve_sparse_score_kernel_config,
)

SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)


@dataclass(frozen=True)
class SparseScoreSm90Config:
    tile_m: int
    tile_n: int
    kv_stage: int
    num_threads: int
    num_head_tiles: int


@dataclass(frozen=True)
class DenseScoreSm90Config:
    tile_m: int
    tile_n: int
    kv_stage: int
    num_threads: int
    num_head_tiles: int


def _require_desc(
    value: Any, name: str, *, optional: bool = False
) -> TensorDesc[Any] | None:
    if optional and value is None:
        return None
    if not isinstance(value, TensorDesc):
        raise TypeError(f"{name} must be a TensorDesc, got {type(value).__name__}")
    return value


def _require_dtype(desc: TensorDesc[Any], expected: data_type, name: str) -> None:
    if desc.cudnn_dtype != expected:
        raise ValueError(f"{name} must have dtype {expected}, got {desc.cudnn_dtype}")


def _require_rank(desc: TensorDesc[Any], rank: int, name: str) -> None:
    if desc.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {desc.shape}")


def _require_compact(desc: TensorDesc[Any], name: str) -> None:
    if not desc.is_compact():
        raise ValueError(f"{name} must be compact, got stride {desc.stride}")


def _require_contiguous_tail(desc: TensorDesc[Any], rank: int, name: str) -> None:
    expected_stride = 1
    for axis in range(desc.ndim - 1, desc.ndim - rank - 1, -1):
        if desc.shape[axis] != 1 and desc.stride[axis] != expected_stride:
            raise ValueError(
                f"{name} must have its final {rank} axes contiguous, "
                f"got shape {desc.shape} and stride {desc.stride}"
            )
        expected_stride *= max(desc.shape[axis], 1)


def _positive_dimensions(named_dimensions: dict[str, int], operation: str) -> None:
    invalid = ", ".join(
        f"{name}={value}" for name, value in named_dimensions.items() if value <= 0
    )
    if invalid:
        raise ValueError(f"{operation} dimensions must be positive, got {invalid}")


def _normalize_target(target_compute_capability: int) -> int:
    if isinstance(target_compute_capability, bool):
        raise TypeError("target_compute_capability must be an integer")
    try:
        target = index(target_compute_capability)
    except TypeError as error:
        raise TypeError("target_compute_capability must be an integer") from error
    if target not in SUPPORTED_COMPUTE_CAPABILITIES:
        supported = ", ".join(f"SM{value}" for value in SUPPORTED_COMPUTE_CAPABILITIES)
        raise ValueError(
            f"Unsupported score-recompute target SM{target}; supported targets are {supported}"
        )
    return target


def _normalize_scale(scale: float, name: str) -> float:
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError(f"{name} must be a real number, got {type(scale).__name__}")
    return float(scale)


class SparseScoreRecomputeOp(Op):
    """Complete sparse score signature and architecture-specific configuration."""

    def __init__(
        self,
        *,
        q: TensorDesc[Any],
        k: TensorDesc[Any],
        per_head: TensorDesc[Any],
        topk_indices: TensorDesc[Any],
        output: TensorDesc[Any],
        score_type: str,
        softmax_scale: float,
        target_compute_capability: int,
        topk_length: TensorDesc[Any] | None = None,
        qhead_per_kv_head: int | None = None,
        topk_indices_global: bool = False,
    ) -> None:
        self.q = _require_desc(q, "q")
        self.k = _require_desc(k, "k")
        self.per_head = _require_desc(per_head, "per_head")
        self.topk_indices = _require_desc(topk_indices, "topk_indices")
        self.output = _require_desc(output, "output")
        self.topk_length = _require_desc(topk_length, "topk_length", optional=True)
        if score_type not in ("indexer", "attention"):
            raise ValueError(
                f"score_type must be 'indexer' or 'attention', got {score_type!r}"
            )
        if not isinstance(topk_indices_global, bool):
            raise TypeError(
                f"topk_indices_global must be a bool, got {type(topk_indices_global).__name__}"
            )

        self.score_type = score_type
        self.softmax_scale = _normalize_scale(softmax_scale, "softmax_scale")
        self.target_compute_capability = _normalize_target(target_compute_capability)
        self.requested_qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = topk_indices_global

        self.qhead_per_kv_head: int | None = None
        self.config: SparseScoreKernelConfig | SparseScoreSm90Config | None = None

    def check_support(self) -> bool:
        self.qhead_per_kv_head = None
        self.config = None

        _require_rank(self.q, 4, "Q")
        _require_rank(self.k, 3, "K")
        _require_rank(
            self.per_head, 3, "weights" if self.score_type == "indexer" else "LSE"
        )
        _require_rank(self.topk_indices, 3, "topk_indices")
        _require_rank(self.output, 3, "output")
        _require_dtype(self.q, data_type.BFLOAT16, "Q")
        _require_dtype(self.k, data_type.BFLOAT16, "K")
        _require_dtype(
            self.per_head,
            data_type.BFLOAT16 if self.score_type == "indexer" else data_type.FLOAT,
            "weights" if self.score_type == "indexer" else "LSE",
        )
        _require_dtype(self.topk_indices, data_type.INT32, "topk_indices")
        _require_dtype(self.output, data_type.FLOAT, "output")
        for desc, name in (
            (self.q, "Q"),
            (self.k, "K"),
            (self.per_head, "weights" if self.score_type == "indexer" else "LSE"),
            (self.topk_indices, "topk_indices"),
            (self.output, "output"),
        ):
            _require_compact(desc, name)
        _require_contiguous_tail(self.q, 2, "Q")
        for desc, name in (
            (self.k, "K"),
            (
                self.per_head,
                "weights" if self.score_type == "indexer" else "LSE",
            ),
            (self.topk_indices, "topk_indices"),
            (self.output, "output"),
        ):
            _require_contiguous_tail(desc, 1, name)

        batch, seqlen_q, num_query_heads, head_dim = self.q.shape
        k_batch, seqlen_k, k_head_dim = self.k.shape
        topk_batch, topk_seqlen_q, topk = self.topk_indices.shape
        _positive_dimensions(
            {
                "B": batch,
                "S_q": seqlen_q,
                "S_k": seqlen_k,
                "H_q": num_query_heads,
                "D": head_dim,
                "topk": topk,
            },
            "Sparse score-recompute",
        )
        if k_batch != batch or k_head_dim != head_dim:
            raise ValueError(
                f"K shape must be {(batch, seqlen_k, head_dim)}, got {self.k.shape}"
            )
        expected_per_head = (batch, seqlen_q, num_query_heads)
        if self.per_head.shape != expected_per_head:
            raise ValueError(
                f"per-head tensor must have shape {expected_per_head}, got {self.per_head.shape}"
            )
        if (topk_batch, topk_seqlen_q) != (batch, seqlen_q):
            raise ValueError(
                f"topk_indices leading dimensions must be {(batch, seqlen_q)}, got {self.topk_indices.shape[:2]}"
            )
        if self.output.shape != self.topk_indices.shape:
            raise ValueError(
                f"output must have shape {self.topk_indices.shape}, got {self.output.shape}"
            )

        if self.topk_length is not None:
            _require_rank(self.topk_length, 2, "topk_length")
            _require_dtype(self.topk_length, data_type.INT32, "topk_length")
            _require_compact(self.topk_length, "topk_length")
            if self.topk_length.shape != (batch, seqlen_q):
                raise ValueError(
                    f"topk_length must have shape {(batch, seqlen_q)}, got {self.topk_length.shape}"
                )

        qhead_per_kv_head = (
            num_query_heads
            if self.requested_qhead_per_kv_head is None
            else self.requested_qhead_per_kv_head
        )
        if isinstance(qhead_per_kv_head, bool):
            raise TypeError("qhead_per_kv_head must be an integer")
        try:
            qhead_per_kv_head = index(qhead_per_kv_head)
        except TypeError as error:
            raise TypeError("qhead_per_kv_head must be an integer") from error
        if qhead_per_kv_head != num_query_heads:
            raise ValueError(
                f"qhead_per_kv_head must equal H_q ({num_query_heads}) for sparse MQA, got {qhead_per_kv_head}"
            )

        if self.target_compute_capability == 90:
            if topk % 128:
                raise ValueError(
                    f"SM90 sparse score recompute requires topk to be a multiple of 128, got {topk}"
                )
            if qhead_per_kv_head <= 1:
                raise ValueError(
                    "SM90 sparse score recompute requires qhead_per_kv_head > 1"
                )
            tile_m = min(qhead_per_kv_head, 64)
            if qhead_per_kv_head % tile_m:
                raise ValueError(
                    f"qhead_per_kv_head ({qhead_per_kv_head}) must be divisible by tile_m ({tile_m})"
                )
            config: SparseScoreKernelConfig | SparseScoreSm90Config = (
                SparseScoreSm90Config(
                    tile_m=tile_m,
                    tile_n=64,
                    kv_stage=2,
                    num_threads=256,
                    num_head_tiles=qhead_per_kv_head // tile_m,
                )
            )
        else:
            config = resolve_sparse_score_kernel_config(
                score_type=self.score_type,
                head_dim=head_dim,
                qhead_per_kv_head=qhead_per_kv_head,
                topk=topk,
                have_topk_length=self.topk_length is not None,
            )

        self.qhead_per_kv_head = qhead_per_kv_head
        self.config = config
        return True


class DenseScoreRecomputeOp(Op):
    """Complete dense score signature and architecture-specific configuration."""

    def __init__(
        self,
        *,
        q: TensorDesc[Any],
        k: TensorDesc[Any],
        per_head: TensorDesc[Any],
        output: TensorDesc[Any],
        denominator: TensorDesc[Any],
        score_type: str,
        scale: float,
        ratio: int,
        target_compute_capability: int,
        qhead_per_kv_head: int | None = None,
        is_thd: bool = False,
        cu_seqlens_q: TensorDesc[Any] | None = None,
        cu_seqlens_k: TensorDesc[Any] | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        q_causal_offsets: TensorDesc[Any] | None = None,
    ) -> None:
        self.q = _require_desc(q, "q")
        self.k = _require_desc(k, "k")
        self.per_head = _require_desc(per_head, "per_head")
        self.output = _require_desc(output, "output")
        self.denominator = _require_desc(denominator, "denominator")
        self.cu_seqlens_q = _require_desc(cu_seqlens_q, "cu_seqlens_q", optional=True)
        self.cu_seqlens_k = _require_desc(cu_seqlens_k, "cu_seqlens_k", optional=True)
        self.q_causal_offsets = _require_desc(
            q_causal_offsets, "q_causal_offsets", optional=True
        )
        if score_type not in ("indexer", "attention"):
            raise ValueError(
                f"score_type must be 'indexer' or 'attention', got {score_type!r}"
            )
        if not isinstance(is_thd, bool):
            raise TypeError(f"is_thd must be a bool, got {type(is_thd).__name__}")
        if isinstance(ratio, bool):
            raise TypeError("ratio must be an integer")
        try:
            ratio = index(ratio)
        except TypeError as error:
            raise TypeError("ratio must be an integer") from error

        self.score_type = score_type
        self.scale = _normalize_scale(scale, "scale")
        self.ratio = ratio
        self.target_compute_capability = _normalize_target(target_compute_capability)
        self.requested_qhead_per_kv_head = qhead_per_kv_head
        self.is_thd = is_thd
        self.requested_max_seqlen_q = max_seqlen_q
        self.requested_max_seqlen_k = max_seqlen_k

        self.qhead_per_kv_head: int | None = None
        self.max_seqlen_q: int | None = None
        self.max_seqlen_k: int | None = None
        self.config: DenseScoreKernelConfig | DenseScoreSm90Config | None = None

    @staticmethod
    def _positive_static(value: int | None, name: str) -> int:
        if value is None or isinstance(value, bool):
            raise ValueError(
                f"{name} must be provided as a positive integer for THD inputs"
            )
        try:
            value = index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be an integer") from error
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    def check_support(self) -> bool:
        self.qhead_per_kv_head = None
        self.max_seqlen_q = None
        self.max_seqlen_k = None
        self.config = None
        if self.ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {self.ratio}")

        expected_rank = 3 if self.is_thd else 4
        _require_rank(self.q, expected_rank, "Q")
        _require_rank(self.k, expected_rank, "K")
        _require_rank(
            self.per_head,
            expected_rank - 1,
            "weights" if self.score_type == "indexer" else "LSE",
        )
        _require_rank(self.output, expected_rank - 1, "output")
        _require_rank(self.denominator, expected_rank - 2, "denominator")
        _require_dtype(self.q, data_type.BFLOAT16, "Q")
        _require_dtype(self.k, data_type.BFLOAT16, "K")
        _require_dtype(
            self.per_head,
            data_type.BFLOAT16 if self.score_type == "indexer" else data_type.FLOAT,
            "weights" if self.score_type == "indexer" else "LSE",
        )
        _require_dtype(self.output, data_type.FLOAT, "output")
        _require_dtype(self.denominator, data_type.FLOAT, "denominator")
        for desc, name in (
            (self.q, "Q"),
            (self.k, "K"),
            (self.per_head, "weights" if self.score_type == "indexer" else "LSE"),
            (self.output, "output"),
            (self.denominator, "denominator"),
        ):
            _require_compact(desc, name)
        for desc, name in ((self.q, "Q"), (self.k, "K")):
            _require_contiguous_tail(desc, 2, name)
        for desc, name in (
            (
                self.per_head,
                "weights" if self.score_type == "indexer" else "LSE",
            ),
            (self.output, "output"),
        ):
            _require_contiguous_tail(desc, 1, name)

        if self.is_thd:
            total_q, num_query_heads, head_dim = self.q.shape
            total_k, num_kv_heads, k_head_dim = self.k.shape
            _positive_dimensions(
                {
                    "total_q": total_q,
                    "total_k": total_k,
                    "H_q": num_query_heads,
                    "H_kv": num_kv_heads,
                    "D": head_dim,
                },
                "Dense THD score-recompute",
            )
            if self.per_head.shape != (total_q, num_query_heads):
                raise ValueError(
                    f"THD per-head tensor must have shape {(total_q, num_query_heads)}, got {self.per_head.shape}"
                )
            if self.cu_seqlens_q is None or self.cu_seqlens_k is None:
                raise ValueError(
                    "THD dense score recompute requires both cu_seqlens_q and cu_seqlens_k"
                )
            for desc, name in (
                (self.cu_seqlens_q, "cu_seqlens_q"),
                (self.cu_seqlens_k, "cu_seqlens_k"),
            ):
                _require_rank(desc, 1, name)
                _require_dtype(desc, data_type.INT32, name)
                _require_compact(desc, name)
            if self.cu_seqlens_q.shape != self.cu_seqlens_k.shape:
                raise ValueError(
                    f"cu_seqlens_q and cu_seqlens_k shapes must match, got {self.cu_seqlens_q.shape} and {self.cu_seqlens_k.shape}"
                )
            batch = self.cu_seqlens_q.shape[0] - 1
            if batch <= 0:
                raise ValueError(
                    "cumulative sequence-length tensors must contain at least two entries"
                )
            max_seqlen_q = self._positive_static(
                self.requested_max_seqlen_q, "max_seqlen_q"
            )
            max_seqlen_k = self._positive_static(
                self.requested_max_seqlen_k, "max_seqlen_k"
            )
            expected_output = (total_q, max_seqlen_k)
            expected_denominator = (total_q,)
            if self.target_compute_capability == 90:
                raise NotImplementedError(
                    "SM90 THD score recompute uses host-side sequence-length reads and cannot be traced by JAX; use SM100+ or BSHD inputs"
                )
        else:
            if self.cu_seqlens_q is not None or self.cu_seqlens_k is not None:
                raise ValueError(
                    "BSHD dense score recompute does not accept cu_seqlens_q or cu_seqlens_k"
                )
            batch, seqlen_q, num_query_heads, head_dim = self.q.shape
            k_batch, seqlen_k, num_kv_heads, k_head_dim = self.k.shape
            _positive_dimensions(
                {
                    "B": batch,
                    "S_q": seqlen_q,
                    "S_k": seqlen_k,
                    "H_q": num_query_heads,
                    "H_kv": num_kv_heads,
                    "D": head_dim,
                },
                "Dense BSHD score-recompute",
            )
            if k_batch != batch:
                raise ValueError(f"K batch dimension must be {batch}, got {k_batch}")
            if self.per_head.shape != (batch, seqlen_q, num_query_heads):
                raise ValueError(
                    f"BSHD per-head tensor must have shape {(batch, seqlen_q, num_query_heads)}, got {self.per_head.shape}"
                )
            max_seqlen_q, max_seqlen_k = seqlen_q, seqlen_k
            expected_output = (batch, seqlen_q, seqlen_k)
            expected_denominator = (batch, seqlen_q)

        if k_head_dim != head_dim:
            raise ValueError(f"K head dimension must be {head_dim}, got {k_head_dim}")
        if num_kv_heads != 1:
            raise ValueError(
                f"Dense score recompute currently requires MQA with H_kv=1, got {num_kv_heads}"
            )
        if self.output.shape != expected_output:
            raise ValueError(
                f"output must have shape {expected_output}, got {self.output.shape}"
            )
        if self.denominator.shape != expected_denominator:
            raise ValueError(
                f"denominator must have shape {expected_denominator}, got {self.denominator.shape}"
            )

        if self.q_causal_offsets is not None:
            _require_rank(self.q_causal_offsets, 1, "q_causal_offsets")
            _require_dtype(self.q_causal_offsets, data_type.INT32, "q_causal_offsets")
            _require_compact(self.q_causal_offsets, "q_causal_offsets")
            if self.q_causal_offsets.shape != (batch,):
                raise ValueError(
                    f"q_causal_offsets must have shape {(batch,)}, got {self.q_causal_offsets.shape}"
                )

        inferred_qhead_per_kv_head = num_query_heads // num_kv_heads
        qhead_per_kv_head = (
            inferred_qhead_per_kv_head
            if self.requested_qhead_per_kv_head is None
            else self.requested_qhead_per_kv_head
        )
        if isinstance(qhead_per_kv_head, bool):
            raise TypeError("qhead_per_kv_head must be an integer")
        try:
            qhead_per_kv_head = index(qhead_per_kv_head)
        except TypeError as error:
            raise TypeError("qhead_per_kv_head must be an integer") from error
        if qhead_per_kv_head != inferred_qhead_per_kv_head:
            raise ValueError(
                f"qhead_per_kv_head must equal H_q / H_kv ({inferred_qhead_per_kv_head}), got {qhead_per_kv_head}"
            )

        if self.target_compute_capability == 90:
            if qhead_per_kv_head <= 1:
                raise ValueError(
                    "SM90 dense score recompute requires qhead_per_kv_head > 1"
                )
            tile_m = min(qhead_per_kv_head, 64)
            if qhead_per_kv_head % tile_m:
                raise ValueError(
                    f"qhead_per_kv_head ({qhead_per_kv_head}) must be divisible by tile_m ({tile_m})"
                )
            config: DenseScoreKernelConfig | DenseScoreSm90Config = (
                DenseScoreSm90Config(
                    tile_m=tile_m,
                    tile_n=64,
                    kv_stage=2,
                    num_threads=384,
                    num_head_tiles=qhead_per_kv_head // tile_m,
                )
            )
        else:
            config = resolve_dense_score_kernel_config(
                score_type=self.score_type,
                head_dim=head_dim,
                qhead_per_kv_head=qhead_per_kv_head,
            )

        self.qhead_per_kv_head = qhead_per_kv_head
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.config = config
        return True


__all__ = [
    "DenseScoreRecomputeOp",
    "DenseScoreSm90Config",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SparseScoreRecomputeOp",
    "SparseScoreSm90Config",
]
