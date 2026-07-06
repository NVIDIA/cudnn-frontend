# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the DSA sparse score-recompute kernels."""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_array,
)
from .config import (
    resolve_dense_score_kernel_config,
    resolve_sparse_score_kernel_config,
)


def _launch_sparse_with_topk_length(
    stream,
    q,
    k,
    per_head,
    topk_indices,
    topk_length,
    out,
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    m_block_size: int,
    n_block_size: int,
    k_block_size: int | None,
    kv_stage: int,
    topk_in_smem: bool,
    topk_indices_global: bool,
    softmax_scale: float,
):
    # Load the architecture-specific kernel only when tracing the operation.
    from cutlass import Float32

    from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100

    kernel = SparseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        topk=topk,
        kv_stage=kv_stage,
        score_type=score_type,
        have_topk_length=True,
        topk_in_smem=topk_in_smem,
        k_block_size=k_block_size,
        topk_indices_global=topk_indices_global,
    )

    kernel(
        q,
        k,
        per_head,
        topk_indices,
        out,
        topk_length,
        Float32(softmax_scale),
        stream,
    )


def _launch_sparse_without_topk_length(
    stream,
    q,
    k,
    per_head,
    topk_indices,
    out,
    topk_length_workspace,
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    topk: int,
    m_block_size: int,
    n_block_size: int,
    k_block_size: int | None,
    kv_stage: int,
    topk_in_smem: bool,
    topk_indices_global: bool,
    softmax_scale: float,
):
    # Load the architecture-specific kernel only when tracing the operation.
    from cutlass import Float32

    from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100

    kernel = SparseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        topk=topk,
        kv_stage=kv_stage,
        score_type=score_type,
        have_topk_length=False,
        topk_in_smem=topk_in_smem,
        k_block_size=k_block_size,
        topk_indices_global=topk_indices_global,
    )
    kernel(
        q,
        k,
        per_head,
        topk_indices,
        out,
        topk_length_workspace,
        Float32(softmax_scale),
        stream,
    )


def _launch_dense_with_q_causal_offsets(
    stream,
    q,
    k,
    per_head,
    q_causal_offsets,
    out,
    denom,
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    ratio: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
):
    from cutlass import Float32, Int32

    from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100

    config = resolve_dense_score_kernel_config(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    kernel = DenseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=config.m_block_size,
        n_block_size=config.n_block_size,
        k_block_size=config.k_block_size,
        kv_stage=config.kv_stage,
        score_type=score_type,
        ratio=ratio,
        is_varlen=False,
    )
    kernel(
        q,
        k,
        per_head,
        out,
        denom,
        Float32(scale),
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        None,
        None,
        q_causal_offsets,
        stream,
    )


def _launch_dense_without_q_causal_offsets(
    stream,
    q,
    k,
    per_head,
    out,
    denom,
    *,
    score_type: str,
    head_dim: int,
    qhead_per_kv_head: int,
    ratio: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
):
    from cutlass import Float32, Int32

    from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100

    config = resolve_dense_score_kernel_config(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
    )
    kernel = DenseScoreRecomputeSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        m_block_size=config.m_block_size,
        n_block_size=config.n_block_size,
        k_block_size=config.k_block_size,
        kv_stage=config.kv_stage,
        score_type=score_type,
        ratio=ratio,
        is_varlen=False,
    )
    kernel(
        q,
        k,
        per_head,
        out,
        denom,
        Float32(scale),
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        None,
        None,
        None,
        stream,
    )


def _dense_score_recompute(
    q: Any,
    k: Any,
    per_head: Any,
    *,
    score_type: str,
    per_head_name: str,
    per_head_dtype: Any,
    scale: float,
    qhead_per_kv_head: Optional[int],
    ratio: int,
    q_causal_offsets: Any | None,
    _validate_only: bool = False,
) -> TupleDict:
    q_shape = require_array(q, name="q", rank=4, dtype=jnp.bfloat16)
    k_shape = require_array(k, name="k", rank=4, dtype=jnp.bfloat16)

    batch, seqlen_q, num_query_heads, head_dim = q_shape
    k_batch, seqlen_k, num_kv_heads, k_head_dim = k_shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "H_q": num_query_heads,
        "H_kv": num_kv_heads,
        "head dimension": head_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Dense score-recompute dimensions must be positive, got " + ", ".join(nonpositive))
    if k_batch != batch:
        raise ValueError(f"q and k batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError("q and k head dimensions must match, got " f"{head_dim} and {k_head_dim}")
    require_array(
        per_head,
        name=per_head_name,
        shape=(batch, seqlen_q, num_query_heads),
        dtype=per_head_dtype,
    )
    if num_query_heads % num_kv_heads:
        raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv ({num_kv_heads})")

    inferred_qhead_per_kv_head = num_query_heads // num_kv_heads
    if qhead_per_kv_head is None:
        qhead_per_kv_head = inferred_qhead_per_kv_head
    if qhead_per_kv_head != inferred_qhead_per_kv_head:
        raise ValueError("qhead_per_kv_head must equal H_q / H_kv, got " f"{qhead_per_kv_head} and {num_query_heads} / {num_kv_heads}")
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")

    inputs = (q, k, per_head)
    if q_causal_offsets is not None:
        require_array(
            q_causal_offsets,
            name="q_causal_offsets",
            shape=(batch,),
            dtype=jnp.int32,
        )
        inputs += (q_causal_offsets,)

    resolved_scale = float(scale)
    if _validate_only:
        return None

    launcher = _launch_dense_with_q_causal_offsets if q_causal_offsets is not None else _launch_dense_without_q_causal_offsets
    out, denom = call_cutedsl(
        launcher,
        inputs,
        outputs=(
            BufferSpec(
                "out",
                (batch, seqlen_q, seqlen_k),
                jnp.float32,
                fill_value=float("-inf"),
            ),
            BufferSpec("denom", (batch, seqlen_q), jnp.float32),
        ),
        static_args={
            "score_type": str(score_type),
            "head_dim": int(head_dim),
            "qhead_per_kv_head": int(qhead_per_kv_head),
            "ratio": int(ratio),
            "max_seqlen_q": int(seqlen_q),
            "max_seqlen_k": int(seqlen_k),
            "scale": resolved_scale,
        },
    )
    return TupleDict(out=out, denom=denom)


def _sparse_score_recompute(
    q: Any,
    k: Any,
    per_head: Any,
    topk_indices: Any,
    *,
    score_type: str,
    output_name: str,
    per_head_name: str,
    per_head_dtype: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int],
    topk_length: Optional[Any],
    topk_indices_global: bool,
    _validate_only: bool = False,
) -> Any:
    q_shape = require_array(q, name="q", rank=4, dtype=jnp.bfloat16)
    k_shape = require_array(k, name="k", rank=3, dtype=jnp.bfloat16)
    topk_shape = require_array(
        topk_indices,
        name="topk_indices",
        rank=3,
        dtype=jnp.int32,
    )

    batch, seqlen_q, num_query_heads, head_dim = q_shape
    k_batch, seqlen_k, k_head_dim = k_shape
    topk_batch, topk_seqlen_q, topk = topk_shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "H_q": num_query_heads,
        "head dimension": head_dim,
        "topk": topk,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Sparse score-recompute dimensions must be positive, got " + ", ".join(nonpositive))
    if k_batch != batch:
        raise ValueError(f"q and k batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError(f"q and k head dimensions must match, got {head_dim} and {k_head_dim}")
    require_array(
        per_head,
        name=per_head_name,
        shape=(batch, seqlen_q, num_query_heads),
        dtype=per_head_dtype,
    )
    if (topk_batch, topk_seqlen_q) != (batch, seqlen_q):
        raise ValueError(
            "topk_indices leading dimensions must match q's batch and sequence " f"dimensions {(batch, seqlen_q)}, got {(topk_batch, topk_seqlen_q)}"
        )

    if qhead_per_kv_head is None:
        qhead_per_kv_head = num_query_heads
    if qhead_per_kv_head != num_query_heads:
        raise ValueError("qhead_per_kv_head must equal H_q for the MQA sparse score kernel, " f"got {qhead_per_kv_head} and H_q={num_query_heads}")

    if topk_length is not None:
        require_array(
            topk_length,
            name="topk_length",
            shape=(batch, seqlen_q),
            dtype=jnp.int32,
        )

    # Keep an explicit length on the compact path. The Torch-only optimization
    # that sometimes drops it assumes every tail index was already set to -1;
    # the JAX contract treats topk_length itself as authoritative.
    config = resolve_sparse_score_kernel_config(
        score_type=score_type,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        topk=topk,
        have_topk_length=topk_length is not None,
    )
    if _validate_only:
        return None

    inputs = (q, k, per_head, topk_indices)
    workspaces = ()
    if topk_length is not None:
        inputs += (topk_length,)
    else:
        # The SM100 kernel has a mandatory tensor argument even when its static
        # ``have_topk_length`` mode is false. It never reads this dummy buffer.
        workspaces = (BufferSpec("topk_length_workspace", (1, 1), jnp.int32),)

    launcher = _launch_sparse_with_topk_length if config.have_topk_length else _launch_sparse_without_topk_length
    (out,) = call_cutedsl(
        launcher,
        inputs,
        outputs=(BufferSpec(output_name, (batch, seqlen_q, topk), jnp.float32),),
        workspaces=workspaces,
        static_args={
            "score_type": str(score_type),
            "head_dim": int(head_dim),
            "qhead_per_kv_head": int(qhead_per_kv_head),
            "topk": int(topk),
            "m_block_size": int(config.m_block_size),
            "n_block_size": int(config.n_block_size),
            "k_block_size": (None if config.k_block_size is None else int(config.k_block_size)),
            "kv_stage": int(config.kv_stage),
            "topk_in_smem": bool(config.topk_in_smem),
            "topk_indices_global": bool(topk_indices_global),
            "softmax_scale": float(softmax_scale),
        },
    )
    return out


class SparseIndexerScoreRecompute(ApiBaseJax):
    """Sample-signature-bound JAX callable for sparse indexer score recompute."""

    def __init__(
        self,
        sample_q_indexer: Any,
        sample_k_indexer: Any,
        sample_weights: Any,
        sample_topk_indices: Any,
        qhead_per_kv_head: Optional[int] = None,
        sample_topk_length: Optional[Any] = None,
        topk_indices_global: bool = False,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q_indexer, name="sample_q_indexer")
        self.k_desc = self.make_tensor_desc(sample_k_indexer, name="sample_k_indexer")
        self.weights_desc = self.make_tensor_desc(sample_weights, name="sample_weights")
        self.topk_indices_desc = self.make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.topk_length_desc = self.make_optional_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = topk_indices_global

    def _check_support(self) -> None:
        _sparse_score_recompute(
            self.q_desc,
            self.k_desc,
            self.weights_desc,
            self.topk_indices_desc,
            score_type="indexer",
            output_name="predict",
            per_head_name="weights",
            per_head_dtype=jnp.bfloat16,
            softmax_scale=1.0,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=self.topk_length_desc,
            topk_indices_global=self.topk_indices_global,
            _validate_only=True,
        )

    def __call__(
        self,
        q_indexer: Any,
        k_indexer: Any,
        weights: Any,
        topk_indices: Any,
        topk_length: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(q_indexer, k_indexer, weights, topk_indices, topk_length)

    def _call_impl(
        self,
        q_indexer: Any,
        k_indexer: Any,
        weights: Any,
        topk_indices: Any,
        topk_length: Optional[Any] = None,
    ) -> TupleDict:
        for value, expected, name in (
            (q_indexer, self.q_desc, "Q"),
            (k_indexer, self.k_desc, "K"),
            (weights, self.weights_desc, "weights"),
            (topk_indices, self.topk_indices_desc, "topk_indices"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        self.check_optional_tensor_signature(topk_length, self.topk_length_desc, name="topk_length")
        predict = _sparse_score_recompute(
            q_indexer,
            k_indexer,
            weights,
            topk_indices,
            score_type="indexer",
            output_name="predict",
            per_head_name="weights",
            per_head_dtype=jnp.bfloat16,
            softmax_scale=1.0,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        return TupleDict(predict=predict)


class SparseAttnScoreRecompute(ApiBaseJax):
    """Sample-signature-bound JAX callable for sparse attention score recompute."""

    def __init__(
        self,
        sample_q_attn: Any,
        sample_k_attn: Any,
        sample_lse: Any,
        sample_topk_indices: Any,
        softmax_scale: float,
        qhead_per_kv_head: Optional[int] = None,
        sample_topk_length: Optional[Any] = None,
        topk_indices_global: bool = False,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q_attn, name="sample_q_attn")
        self.k_desc = self.make_tensor_desc(sample_k_attn, name="sample_k_attn")
        self.lse_desc = self.make_tensor_desc(sample_lse, name="sample_lse")
        self.topk_indices_desc = self.make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.topk_length_desc = self.make_optional_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = softmax_scale
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = topk_indices_global

    def _check_support(self) -> None:
        _sparse_score_recompute(
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            self.topk_indices_desc,
            score_type="attention",
            output_name="target",
            per_head_name="lse",
            per_head_dtype=jnp.float32,
            softmax_scale=self.softmax_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=self.topk_length_desc,
            topk_indices_global=self.topk_indices_global,
            _validate_only=True,
        )

    def __call__(
        self,
        q_attn: Any,
        k_attn: Any,
        lse: Any,
        topk_indices: Any,
        topk_length: Optional[Any] = None,
    ) -> TupleDict:
        return super().__call__(q_attn, k_attn, lse, topk_indices, topk_length)

    def _call_impl(
        self,
        q_attn: Any,
        k_attn: Any,
        lse: Any,
        topk_indices: Any,
        topk_length: Optional[Any] = None,
    ) -> TupleDict:
        for value, expected, name in (
            (q_attn, self.q_desc, "Q"),
            (k_attn, self.k_desc, "K"),
            (lse, self.lse_desc, "LSE"),
            (topk_indices, self.topk_indices_desc, "topk_indices"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        self.check_optional_tensor_signature(topk_length, self.topk_length_desc, name="topk_length")
        target = _sparse_score_recompute(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            score_type="attention",
            output_name="target",
            per_head_name="lse",
            per_head_dtype=jnp.float32,
            softmax_scale=self.softmax_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        return TupleDict(target=target)


class DenseIndexerScoreRecompute(ApiBaseJax):
    """Sample-signature-bound JAX callable for dense indexer score recompute."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_weights: Any,
        qhead_per_kv_head: Optional[int] = None,
        sm_scale: float = 1.0,
        ratio: int = 1,
        sample_q_causal_offsets: Any | None = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.weights_desc = self.make_tensor_desc(sample_weights, name="sample_weights")
        self.q_causal_offsets_desc = self.make_optional_tensor_desc(sample_q_causal_offsets, name="sample_q_causal_offsets")
        self.qhead_per_kv_head = qhead_per_kv_head
        self.sm_scale = sm_scale
        self.ratio = ratio

    def _check_support(self) -> None:
        _dense_score_recompute(
            self.q_desc,
            self.k_desc,
            self.weights_desc,
            score_type="indexer",
            per_head_name="weights",
            per_head_dtype=jnp.bfloat16,
            scale=self.sm_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio,
            q_causal_offsets=self.q_causal_offsets_desc,
            _validate_only=True,
        )

    def __call__(self, q: Any, k: Any, weights: Any, q_causal_offsets: Any | None = None) -> TupleDict:
        return super().__call__(q, k, weights, q_causal_offsets)

    def _call_impl(self, q: Any, k: Any, weights: Any, q_causal_offsets: Any | None = None) -> TupleDict:
        self.check_tensor_signature(q, self.q_desc, name="Q")
        self.check_tensor_signature(k, self.k_desc, name="K")
        self.check_tensor_signature(weights, self.weights_desc, name="weights")
        self.check_optional_tensor_signature(q_causal_offsets, self.q_causal_offsets_desc, name="q_causal_offsets")
        return _dense_score_recompute(
            q,
            k,
            weights,
            score_type="indexer",
            per_head_name="weights",
            per_head_dtype=jnp.bfloat16,
            scale=self.sm_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio,
            q_causal_offsets=q_causal_offsets,
        )


class DenseAttnScoreRecompute(ApiBaseJax):
    """Sample-signature-bound JAX callable for dense attention score recompute."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_lse: Any,
        softmax_scale: float,
        qhead_per_kv_head: Optional[int] = None,
        ratio: int = 1,
        sample_q_causal_offsets: Any | None = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.lse_desc = self.make_tensor_desc(sample_lse, name="sample_lse")
        self.q_causal_offsets_desc = self.make_optional_tensor_desc(sample_q_causal_offsets, name="sample_q_causal_offsets")
        self.softmax_scale = softmax_scale
        self.qhead_per_kv_head = qhead_per_kv_head
        self.ratio = ratio

    def _check_support(self) -> None:
        _dense_score_recompute(
            self.q_desc,
            self.k_desc,
            self.lse_desc,
            score_type="attention",
            per_head_name="lse",
            per_head_dtype=jnp.float32,
            scale=self.softmax_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio,
            q_causal_offsets=self.q_causal_offsets_desc,
            _validate_only=True,
        )

    def __call__(self, q: Any, k: Any, lse: Any, q_causal_offsets: Any | None = None) -> TupleDict:
        return super().__call__(q, k, lse, q_causal_offsets)

    def _call_impl(self, q: Any, k: Any, lse: Any, q_causal_offsets: Any | None = None) -> TupleDict:
        self.check_tensor_signature(q, self.q_desc, name="Q")
        self.check_tensor_signature(k, self.k_desc, name="K")
        self.check_tensor_signature(lse, self.lse_desc, name="LSE")
        self.check_optional_tensor_signature(q_causal_offsets, self.q_causal_offsets_desc, name="q_causal_offsets")
        return _dense_score_recompute(
            q,
            k,
            lse,
            score_type="attention",
            per_head_name="lse",
            per_head_dtype=jnp.float32,
            scale=self.softmax_scale,
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio,
            q_causal_offsets=q_causal_offsets,
        )


def sparse_indexer_score_recompute_wrapper(
    q_indexer: Any,
    k_indexer: Any,
    weights: Any,
    topk_indices: Any,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[Any] = None,
    topk_indices_global: bool = False,
) -> TupleDict:
    """Recompute normalized sparse indexer scores with the SM100 kernel.

    Inputs have shapes ``(B, S_q, H_q, D)``, ``(B, S_k, D)``,
    ``(B, S_q, H_q)``, and ``(B, S_q, topk)``. The first three tensors use
    ``bfloat16`` and ``topk_indices`` uses ``int32``. ``topk_length``, when
    present, is an ``int32`` tensor of shape ``(B, S_q)``. The returned
    ``predict`` tensor has shape ``(B, S_q, topk)`` and dtype ``float32``.

    This API supports fixed-shape SM100 execution. Configuration and
    ``topk_indices_global`` are compile-time values.
    """

    return SparseIndexerScoreRecompute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        qhead_per_kv_head=qhead_per_kv_head,
        sample_topk_length=topk_length,
        topk_indices_global=topk_indices_global,
    )(q_indexer, k_indexer, weights, topk_indices, topk_length)


def sparse_attn_score_recompute_wrapper(
    q_attn: Any,
    k_attn: Any,
    lse: Any,
    topk_indices: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    topk_length: Optional[Any] = None,
    topk_indices_global: bool = False,
) -> TupleDict:
    """Recompute normalized sparse attention scores with the SM100 kernel.

    ``q_attn`` and ``k_attn`` use ``bfloat16``; ``lse`` uses ``float32``;
    indices and optional lengths use ``int32``. Shapes match
    :func:`sparse_indexer_score_recompute_wrapper`, and the returned ``target``
    is ``(B, S_q, topk)`` ``float32``. ``softmax_scale`` and all configuration
    values are compile-time values.
    """

    return SparseAttnScoreRecompute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        softmax_scale=softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        sample_topk_length=topk_length,
        topk_indices_global=topk_indices_global,
    )(q_attn, k_attn, lse, topk_indices, topk_length)


def dense_indexer_score_recompute_wrapper(
    q: Any,
    k: Any,
    weights: Any,
    qhead_per_kv_head: Optional[int] = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    q_causal_offsets: Any | None = None,
) -> TupleDict:
    """Compute dense indexer scores and their log-sum-exp denominator.

    This binding currently supports fixed-shape SM100 BSHD tensors. ``q`` and
    ``k`` have shapes ``(B, S_q, H_q, D)`` and ``(B, S_k, H_kv, D)``;
    ``weights`` has shape ``(B, S_q, H_q)``. All inputs use ``bfloat16``.
    The returned ``out`` and ``denom`` arrays use ``float32`` and have shapes
    ``(B, S_q, S_k)`` and ``(B, S_q)``.
    """

    return DenseIndexerScoreRecompute(
        q,
        k,
        weights,
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
        ratio=ratio,
        sample_q_causal_offsets=q_causal_offsets,
    )(q, k, weights, q_causal_offsets)


def dense_attn_score_recompute_wrapper(
    q: Any,
    k: Any,
    lse: Any,
    softmax_scale: float,
    qhead_per_kv_head: Optional[int] = None,
    ratio: int = 1,
    q_causal_offsets: Any | None = None,
) -> TupleDict:
    """Compute dense attention scores and their L1 denominator.

    This binding currently supports fixed-shape SM100 BSHD tensors. ``q`` and
    ``k`` use ``bfloat16`` and ``lse`` uses ``float32``. The returned ``out``
    and ``denom`` arrays use ``float32`` and have shapes ``(B, S_q, S_k)`` and
    ``(B, S_q)``.
    """

    return DenseAttnScoreRecompute(
        q,
        k,
        lse,
        softmax_scale=softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        sample_q_causal_offsets=q_causal_offsets,
    )(q, k, lse, q_causal_offsets)


__all__ = [
    "DenseAttnScoreRecompute",
    "DenseIndexerScoreRecompute",
    "SparseAttnScoreRecompute",
    "SparseIndexerScoreRecompute",
    "dense_attn_score_recompute_wrapper",
    "dense_indexer_score_recompute_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "sparse_indexer_score_recompute_wrapper",
]
