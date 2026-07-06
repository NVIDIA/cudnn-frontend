# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for the DeepSeek indexer-forward score kernel."""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_array,
)

_TMA_ALIGN_ELEMENTS = 4


def _launch(
    stream,
    q,
    k,
    w,
    scores,
    *,
    head_dim: int,
    qhead_per_kv_head: int,
    ratio: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    kv_stage: int,
    num_kv_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    # Load the configuration-specific kernel only when tracing the operation.
    from cutlass import Float32, Int32

    from .indexer_fwd_sm100 import IndexerForwardSm100

    kernel = IndexerForwardSm100(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kv_head,
        ratio=ratio,
        is_varlen=False,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        q_stage=q_stage,
        kv_stage=kv_stage,
    )

    kernel(
        q,
        k,
        w,
        scores,
        Int32(num_kv_heads),
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        Float32(sm_scale),
        None,
        None,
        None,
        stream,
    )


def _require_supported_config(
    *,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    kv_stage: int,
) -> None:
    supported = {
        "m_block_size": (m_block_size, 128),
        "n_block_size": (n_block_size, 128),
        "q_stage": (q_stage, 2),
        "kv_stage": (kv_stage, 4),
    }
    unsupported = [f"{name}={actual} (expected {expected})" for name, (actual, expected) in supported.items() if actual != expected]
    if unsupported:
        raise ValueError("The JAX indexer-forward API supports only the validated SM100 " "kernel configuration: " + ", ".join(unsupported))


def _indexer_forward_impl(
    q: Any,
    k: Any,
    w: Any,
    *,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute fixed-shape BSHD indexer scores with the SM100 CuTe kernel.

    ``q`` and ``k`` must have shapes ``(B, S_q, H_q, 128)`` and
    ``(B, S_k, H_kv, 128)``. ``w`` must have shape ``(B, S_q, H_q)``.
    All three inputs use ``bfloat16`` and the returned scores use ``float32``.

    This API supports fixed-shape BSHD inputs on SM100 only.
    Variable-length THD inputs and the SM90 implementation remain available
    only through the existing PyTorch API. All configuration arguments are
    compile-time values; close them over a jitted function or mark them static
    with :func:`jax.jit`.

    The kernel intentionally skips causal and padded score positions. The
    wrapper therefore initializes its XLA-owned output storage to ``-inf``
    before launching the kernel.
    """

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
        raise ValueError("Indexer-forward dimensions must be positive, got " + ", ".join(nonpositive))

    if k_batch != batch:
        raise ValueError(f"q and k batch dimensions must match, got {batch} and {k_batch}")
    if k_head_dim != head_dim:
        raise ValueError(f"q and k head dimensions must match, got {head_dim} and {k_head_dim}")
    if head_dim != 128:
        raise ValueError(f"head dimension must be 128, got {head_dim}")
    require_array(
        w,
        name="w",
        shape=(batch, seqlen_q, num_query_heads),
        dtype=jnp.bfloat16,
    )

    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")
    if seqlen_q > seqlen_k * ratio:
        raise ValueError(f"S_q ({seqlen_q}) must be no greater than " f"S_k * ratio ({seqlen_k * ratio})")

    if qhead_per_kv_head is None:
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"H_q ({num_query_heads}) must be divisible by H_kv " f"({num_kv_heads})")
        qhead_per_kv_head = num_query_heads // num_kv_heads
    if qhead_per_kv_head * num_kv_heads != num_query_heads:
        raise ValueError("qhead_per_kv_head * H_kv must equal H_q, got " f"{qhead_per_kv_head} * {num_kv_heads} != {num_query_heads}")
    if qhead_per_kv_head not in (32, 64):
        raise ValueError("qhead_per_kv_head must be 32 or 64, " f"got {qhead_per_kv_head}")

    _require_supported_config(
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        q_stage=q_stage,
        kv_stage=kv_stage,
    )

    seqlen_k_padded = ((seqlen_k + _TMA_ALIGN_ELEMENTS - 1) // _TMA_ALIGN_ELEMENTS) * _TMA_ALIGN_ELEMENTS
    if _validate_only:
        return None

    # Constrain the initialized physical result to the compact row-major ABI
    # expected by the TMA store. CUTLASS infers equivalent defaults for inputs.
    scores_spec = TensorSpec(
        layout=(2, 1, 0),
        mode=(0, 1, 2),
        divisibility=(None, None, _TMA_ALIGN_ELEMENTS),
    )

    (scores_padded,) = call_cutedsl(
        _launch,
        (q, k, w),
        outputs=(
            BufferSpec(
                "scores",
                (batch, seqlen_q, seqlen_k_padded),
                jnp.float32,
                tensor_spec=scores_spec,
                fill_value=float("-inf"),
            ),
        ),
        static_args={
            "head_dim": int(head_dim),
            "qhead_per_kv_head": int(qhead_per_kv_head),
            "ratio": int(ratio),
            "m_block_size": int(m_block_size),
            "n_block_size": int(n_block_size),
            "q_stage": int(q_stage),
            "kv_stage": int(kv_stage),
            "num_kv_heads": int(num_kv_heads),
            "max_seqlen_q": int(seqlen_q),
            "max_seqlen_k": int(seqlen_k),
            "sm_scale": float(sm_scale),
        },
    )
    return TupleDict(scores=scores_padded[..., :seqlen_k])


class IndexerForward(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 indexer forward."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_w: Any,
        *,
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self.make_tensor_desc(sample_w, name="sample_w")
        self.ratio = ratio
        self.qhead_per_kv_head = qhead_per_kv_head
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        self.kv_stage = kv_stage
        self.sm_scale = sm_scale

    def _check_support(self) -> None:
        _indexer_forward_impl(
            self.q_desc,
            self.k_desc,
            self.w_desc,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            sm_scale=self.sm_scale,
            _validate_only=True,
        )

    def __call__(self, q: Any, k: Any, w: Any) -> TupleDict:
        return super().__call__(q, k, w)

    def _call_impl(self, q: Any, k: Any, w: Any) -> TupleDict:
        self.check_tensor_signature(q, self.q_desc, name="Q")
        self.check_tensor_signature(k, self.k_desc, name="K")
        self.check_tensor_signature(w, self.w_desc, name="W")
        return _indexer_forward_impl(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            sm_scale=self.sm_scale,
        )


def indexer_forward_wrapper(
    q: Any,
    k: Any,
    w: Any,
    *,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
) -> TupleDict:
    """Compute fixed-shape BSHD indexer scores with the SM100 CuTe kernel."""

    return IndexerForward(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
    )(q, k, w)


__all__ = ["IndexerForward", "indexer_forward_wrapper"]
