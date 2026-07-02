# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 sparse indexer backward.

The dense backward path is intentionally not exposed here. Its score-gradient
stage currently consumes ``grad_loss`` through a host scalar and overwrites the
predict-score input before launching a second kernel. A sound JAX binding needs
that kernel ABI to accept runtime loss data and a distinct gradient output.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype


class IndexerBackwardResult(NamedTuple):
    """Functional gradients from :func:`indexer_backward_wrapper`."""

    d_index_q: Any
    d_weights: Any
    d_index_k: Any


def require_array(name: str, value: Any, rank: int, dtype: Any) -> tuple[int, ...]:
    """Require array metadata and return its shape."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a JAX array with shape and dtype metadata")
    if len(value.shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}")
    require_dtype(f"{name}.dtype", value, (dtype,))
    return tuple(value.shape)


def require_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    """Require an exact static shape."""

    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def as_grad_loss_operand(grad_loss: Any) -> Any:
    """Normalize a scalar or one-element FP32 loss gradient to shape ``(1,)``."""

    if hasattr(grad_loss, "shape") and hasattr(grad_loss, "dtype"):
        require_dtype("grad_loss.dtype", grad_loss, (jnp.float32,))
        if tuple(grad_loss.shape) == (1,):
            return grad_loss
        if tuple(grad_loss.shape) == ():
            return jnp.reshape(grad_loss, (1,))
        raise ValueError(f"grad_loss must be scalar or shape (1,), got {grad_loss.shape}")
    return jnp.asarray((grad_loss,), dtype=jnp.float32)


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    heads: int,
    head_dim: int,
    topk: int,
    block_i: int,
    sm_scale: float,
    grad_scale: float,
    topk_indices_global: bool,
):
    from cutlass import Float32

    from .indexer_backward_sm100 import IndexerBackwardSm100, ScoreGradSm100

    score_grad = ScoreGradSm100(topk=topk)
    backward = IndexerBackwardSm100(
        head_dim=head_dim,
        heads=heads,
        block_I=block_i,
        topk=topk,
        topk_indices_global=topk_indices_global,
    )

    def launch(
        stream,
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        grad_loss,
        d_index_q,
        d_weights,
        d_index_k_accum,
        grad_signal,
    ):
        score_grad(
            attn_score,
            index_score,
            grad_loss,
            Float32(grad_scale),
            stream,
            grad_signal,
            None,
        )
        backward(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k_accum,
            grad_signal,
            topk_indices,
            Float32(sm_scale),
            stream,
        )

    return launch


def indexer_backward_wrapper(
    index_q: Any,
    weights: Any,
    index_k: Any,
    attn_score: Any,
    index_score: Any,
    topk_indices: Any,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    grad_loss: Any = 1.0,
    block_I: int = 128,
    topk_indices_global: bool = False,
) -> IndexerBackwardResult:
    """Compute sparse indexer gradients with fixed-shape SM100 kernels.

    Inputs use BSHD shapes ``index_q=(B,S_q,64,128)``,
    ``weights=(B,S_q,64)``, and ``index_k=(B,S_k,128)``. Target scores,
    predicted scores, and top-K indices use shape ``(B,S_q,topk)``. Floating
    model inputs use ``bfloat16``, scores and ``grad_loss`` use ``float32``,
    and indices use ``int32``.

    The score-gradient and GEMM stages execute in one custom call. Caller score
    tensors remain immutable; a hidden XLA-owned buffer carries the gradient
    signal between stages. ``grad_loss`` remains a runtime array operand while
    configuration and ``loss_coeff`` are compile-time values.

    This is a standalone backward operation, not a custom VJP. Runtime top-K
    indices are trusted to follow the selected local/global index convention.
    """

    q_shape = require_array("index_q", index_q, 4, jnp.bfloat16)
    weights_shape = require_array("weights", weights, 3, jnp.bfloat16)
    k_shape = require_array("index_k", index_k, 3, jnp.bfloat16)
    attn_shape = require_array("attn_score", attn_score, 3, jnp.float32)
    index_score_shape = require_array("index_score", index_score, 3, jnp.float32)
    topk_shape = require_array("topk_indices", topk_indices, 3, jnp.int32)

    batch, seqlen_q, heads, head_dim = q_shape
    k_batch, seqlen_k, k_head_dim = k_shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "heads": heads,
        "head_dim": head_dim,
        "topk": topk_shape[-1],
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Indexer-backward dimensions must be positive, got " + ", ".join(nonpositive))
    if heads != 64 or head_dim != 128:
        raise ValueError("The JAX SM100 sparse indexer-backward path requires " f"heads=64 and head_dim=128, got {heads} and {head_dim}")
    if (k_batch, k_head_dim) != (batch, head_dim):
        raise ValueError("index_k batch and head dimensions must match index_q, got " f"{(k_batch, k_head_dim)} and {(batch, head_dim)}")
    require_shape("weights", weights_shape, (batch, seqlen_q, heads))
    score_shape = (batch, seqlen_q, topk_shape[-1])
    require_shape("attn_score", attn_shape, score_shape)
    require_shape("index_score", index_score_shape, score_shape)
    require_shape("topk_indices", topk_shape, score_shape)

    topk = topk_shape[-1]
    if block_I != 128:
        raise ValueError(f"block_I must be 128, got {block_I}")
    if topk % block_I:
        raise ValueError(f"topk ({topk}) must be divisible by block_I ({block_I})")

    grad_loss_operand = as_grad_loss_operand(grad_loss)
    sm_scale = float(sm_scale)
    grad_scale = float(loss_coeff) / (batch * seqlen_q)
    tensor_spec = TensorSpec(divisibility=head_dim)

    d_index_q, d_weights, d_index_k_accum = call_cutedsl(
        _make_launcher(
            heads=heads,
            head_dim=head_dim,
            topk=topk,
            block_i=block_I,
            sm_scale=sm_scale,
            grad_scale=grad_scale,
            topk_indices_global=bool(topk_indices_global),
        ),
        (
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss_operand,
        ),
        outputs=(
            BufferSpec("d_index_q", q_shape, jnp.bfloat16, tensor_spec=tensor_spec),
            BufferSpec("d_weights", weights_shape, jnp.bfloat16),
            BufferSpec(
                "d_index_k_accum",
                k_shape,
                jnp.float32,
                tensor_spec=tensor_spec,
                fill_value=0.0,
            ),
        ),
        workspaces=(BufferSpec("grad_signal", score_shape, jnp.float32),),
        input_specs=(tensor_spec, None, tensor_spec, None, None, None, None),
        use_static_tensors=True,
    )
    return IndexerBackwardResult(
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k_accum.astype(jnp.bfloat16),
    )


__all__ = ["IndexerBackwardResult", "indexer_backward_wrapper"]
