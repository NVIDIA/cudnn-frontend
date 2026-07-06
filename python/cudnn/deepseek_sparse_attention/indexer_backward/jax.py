# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX APIs for fixed-shape SM100 sparse and dense indexer backward."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_array,
)


def as_grad_loss_operand(grad_loss: Any) -> Any:
    """Normalize a scalar or one-element FP32 loss gradient to shape ``(1,)``."""

    if hasattr(grad_loss, "shape") or hasattr(grad_loss, "dtype"):
        shape = require_array(grad_loss, name="grad_loss", dtype=jnp.float32)
        if shape == (1,):
            return grad_loss
        if shape == ():
            return jnp.reshape(grad_loss, (1,))
        raise ValueError(f"grad_loss must be scalar or shape (1,), got {shape}")
    return jnp.asarray((grad_loss,), dtype=jnp.float32)


def _launch(
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


def _launch_dense(
    stream,
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    grad_loss,
    d_index_q,
    d_weights,
    d_index_k_accum,
    grad_signal,
    *,
    heads: int,
    head_dim: int,
    block_i: int,
    ratio: int,
    sm_scale: float,
    grad_scale: float,
    max_seqlen_q: int,
    max_seqlen_k: int,
):
    from cutlass import Float32, Int32

    from .dense_indexer_backward_sm100 import (
        DenseIndexerBackward2QGemmSm100,
        ScoreGradDense,
    )

    score_grad = ScoreGradDense(ratio=ratio, block_I=block_i)
    backward = DenseIndexerBackward2QGemmSm100(
        head_dim=head_dim,
        heads=heads,
        block_I=block_i,
        ratio=ratio,
    )

    score_grad(
        index_score,
        grad_signal,
        attn_score,
        index_lse,
        attn_l1norm,
        None,
        None,
        None,
        grad_loss,
        Float32(grad_scale),
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        stream,
    )
    backward(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k_accum,
        grad_signal,
        None,
        None,
        None,
        Float32(sm_scale),
        Int32(max_seqlen_q),
        Int32(max_seqlen_k),
        stream,
    )


def _indexer_backward_impl(
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
    _validate_only: bool = False,
) -> TupleDict:
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

    q_shape = require_array(index_q, name="index_q", rank=4, dtype=jnp.bfloat16)
    k_shape = require_array(index_k, name="index_k", rank=3, dtype=jnp.bfloat16)
    topk_shape = require_array(
        topk_indices,
        name="topk_indices",
        rank=3,
        dtype=jnp.int32,
    )

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
    score_shape = (batch, seqlen_q, topk_shape[-1])
    weights_shape = require_array(
        weights,
        name="weights",
        shape=(batch, seqlen_q, heads),
        dtype=jnp.bfloat16,
    )
    require_array(
        attn_score,
        name="attn_score",
        shape=score_shape,
        dtype=jnp.float32,
    )
    require_array(
        index_score,
        name="index_score",
        shape=score_shape,
        dtype=jnp.float32,
    )
    if topk_shape[:2] != score_shape[:2]:
        raise ValueError("topk_indices leading dimensions must match index_q's batch and " f"sequence dimensions {score_shape[:2]}, got {topk_shape[:2]}")

    topk = topk_shape[-1]
    if block_I != 128:
        raise ValueError(f"block_I must be 128, got {block_I}")
    if topk % block_I:
        raise ValueError(f"topk ({topk}) must be divisible by block_I ({block_I})")

    grad_loss_operand = as_grad_loss_operand(grad_loss)
    sm_scale = float(sm_scale)
    grad_scale = float(loss_coeff) / (batch * seqlen_q)
    if _validate_only:
        return None

    tensor_spec = TensorSpec(divisibility=head_dim)

    d_index_q, d_weights, d_index_k_accum = call_cutedsl(
        _launch,
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
        static_args={
            "heads": int(heads),
            "head_dim": int(head_dim),
            "topk": int(topk),
            "block_i": int(block_I),
            "sm_scale": float(sm_scale),
            "grad_scale": float(grad_scale),
            "topk_indices_global": bool(topk_indices_global),
        },
    )
    return TupleDict(
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k_accum.astype(jnp.bfloat16),
    )


def _dense_indexer_backward_impl(
    index_q: Any,
    weights: Any,
    index_k: Any,
    attn_score: Any,
    attn_l1norm: Any,
    index_score: Any,
    index_lse: Any,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    grad_loss: Any = 1.0,
    block_I: int = 128,
    ratio: int = 1,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute fixed-shape dense indexer gradients on SM100.

    This binding covers compact BSHD inputs. Score tensors remain immutable;
    an XLA-owned workspace carries the dense score gradient between the two
    CuTe launches. ``grad_loss`` is a runtime FP32 operand while the remaining
    configuration is static when traced.
    """

    q_shape = require_array(index_q, name="index_q", rank=4, dtype=jnp.bfloat16)
    k_shape = require_array(index_k, name="index_k", rank=3, dtype=jnp.bfloat16)

    batch, seqlen_q, heads, head_dim = q_shape
    k_batch, seqlen_k, k_head_dim = k_shape
    dimensions = {
        "batch": batch,
        "S_q": seqlen_q,
        "S_k": seqlen_k,
        "heads": heads,
        "head_dim": head_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Dense indexer-backward dimensions must be positive, got " + ", ".join(nonpositive))
    if heads != 64 or head_dim != 128:
        raise ValueError("The JAX SM100 dense indexer-backward path requires " f"heads=64 and head_dim=128, got {heads} and {head_dim}")
    if (k_batch, k_head_dim) != (batch, head_dim):
        raise ValueError("index_k batch and head dimensions must match index_q, got " f"{(k_batch, k_head_dim)} and {(batch, head_dim)}")

    score_shape = (batch, seqlen_q, seqlen_k)
    denom_shape = (batch, seqlen_q)
    weights_shape = require_array(
        weights,
        name="weights",
        shape=(batch, seqlen_q, heads),
        dtype=jnp.bfloat16,
    )
    require_array(
        attn_score,
        name="attn_score",
        shape=score_shape,
        dtype=jnp.float32,
    )
    require_array(
        attn_l1norm,
        name="attn_l1norm",
        shape=denom_shape,
        dtype=jnp.float32,
    )
    require_array(
        index_score,
        name="index_score",
        shape=score_shape,
        dtype=jnp.float32,
    )
    require_array(
        index_lse,
        name="index_lse",
        shape=denom_shape,
        dtype=jnp.float32,
    )

    if block_I != 128:
        raise ValueError(f"block_I must be 128, got {block_I}")
    if ratio < 1:
        raise ValueError(f"ratio must be at least 1, got {ratio}")

    grad_loss_operand = as_grad_loss_operand(grad_loss)
    sm_scale = float(sm_scale)
    grad_scale = float(loss_coeff) / (batch * seqlen_q)
    if _validate_only:
        return None

    q_tensor_spec = TensorSpec(
        layout=(3, 2, 1, 0),
        divisibility=head_dim,
    )
    k_tensor_spec = TensorSpec(
        layout=(2, 1, 0),
        divisibility=head_dim,
    )
    d_index_q, d_weights, d_index_k_accum = call_cutedsl(
        _launch_dense,
        (
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            grad_loss_operand,
        ),
        outputs=(
            BufferSpec(
                "d_index_q",
                q_shape,
                jnp.bfloat16,
                tensor_spec=q_tensor_spec,
            ),
            BufferSpec("d_weights", weights_shape, jnp.bfloat16),
            BufferSpec(
                "d_index_k_accum",
                k_shape,
                jnp.float32,
                tensor_spec=k_tensor_spec,
            ),
        ),
        workspaces=(BufferSpec("grad_signal", score_shape, jnp.float32),),
        input_specs=(
            q_tensor_spec,
            None,
            k_tensor_spec,
            None,
            None,
            None,
            None,
            None,
        ),
        static_args={
            "heads": int(heads),
            "head_dim": int(head_dim),
            "block_i": int(block_I),
            "ratio": int(ratio),
            "sm_scale": sm_scale,
            "grad_scale": grad_scale,
            "max_seqlen_q": int(seqlen_q),
            "max_seqlen_k": int(seqlen_k),
        },
    )
    return TupleDict(
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k_accum.astype(jnp.bfloat16),
    )


class IndexerBackward(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 sparse indexer backward."""

    def __init__(
        self,
        sample_index_q: Any,
        sample_weights: Any,
        sample_index_k: Any,
        sample_attn_score: Any,
        sample_index_score: Any,
        sample_topk_indices: Any,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        sample_grad_loss: Any = 1.0,
        block_I: int = 128,
        topk_indices_global: bool = False,
    ) -> None:
        super().__init__()
        self.index_q_desc = self.make_tensor_desc(sample_index_q, name="sample_index_q")
        self.weights_desc = self.make_tensor_desc(sample_weights, name="sample_weights")
        self.index_k_desc = self.make_tensor_desc(sample_index_k, name="sample_index_k")
        self.attn_score_desc = self.make_tensor_desc(sample_attn_score, name="sample_attn_score")
        self.index_score_desc = self.make_tensor_desc(sample_index_score, name="sample_index_score")
        self.topk_indices_desc = self.make_tensor_desc(sample_topk_indices, name="sample_topk_indices")
        self.grad_loss_desc = self.make_tensor_desc(as_grad_loss_operand(sample_grad_loss), name="sample_grad_loss")
        self.sm_scale = sm_scale
        self.loss_coeff = loss_coeff
        self.block_I = block_I
        self.topk_indices_global = topk_indices_global

    def _check_support(self) -> None:
        _indexer_backward_impl(
            self.index_q_desc,
            self.weights_desc,
            self.index_k_desc,
            self.attn_score_desc,
            self.index_score_desc,
            self.topk_indices_desc,
            self.sm_scale,
            self.loss_coeff,
            self.grad_loss_desc,
            self.block_I,
            self.topk_indices_global,
            _validate_only=True,
        )

    def __call__(
        self,
        index_q: Any,
        weights: Any,
        index_k: Any,
        attn_score: Any,
        index_score: Any,
        topk_indices: Any,
        grad_loss: Any = 1.0,
    ) -> TupleDict:
        return super().__call__(index_q, weights, index_k, attn_score, index_score, topk_indices, grad_loss)

    def _call_impl(
        self,
        index_q: Any,
        weights: Any,
        index_k: Any,
        attn_score: Any,
        index_score: Any,
        topk_indices: Any,
        grad_loss: Any = 1.0,
    ) -> TupleDict:
        for value, expected, name in (
            (index_q, self.index_q_desc, "index_q"),
            (weights, self.weights_desc, "weights"),
            (index_k, self.index_k_desc, "index_k"),
            (attn_score, self.attn_score_desc, "attn_score"),
            (index_score, self.index_score_desc, "index_score"),
            (topk_indices, self.topk_indices_desc, "topk_indices"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        grad_loss_operand = as_grad_loss_operand(grad_loss)
        self.check_tensor_signature(grad_loss_operand, self.grad_loss_desc, name="grad_loss")
        return _indexer_backward_impl(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            self.sm_scale,
            self.loss_coeff,
            grad_loss_operand,
            self.block_I,
            self.topk_indices_global,
        )


class DenseIndexerBackward(ApiBaseJax):
    """Sample-signature-bound JAX callable for dense SM100 indexer backward."""

    def __init__(
        self,
        sample_index_q: Any,
        sample_weights: Any,
        sample_index_k: Any,
        sample_attn_score: Any,
        sample_attn_l1norm: Any,
        sample_index_score: Any,
        sample_index_lse: Any,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        sample_grad_loss: Any = 1.0,
        block_I: int = 128,
        ratio: int = 1,
    ) -> None:
        super().__init__()
        self.index_q_desc = self.make_tensor_desc(
            sample_index_q,
            name="sample_index_q",
        )
        self.weights_desc = self.make_tensor_desc(
            sample_weights,
            name="sample_weights",
        )
        self.index_k_desc = self.make_tensor_desc(
            sample_index_k,
            name="sample_index_k",
        )
        self.attn_score_desc = self.make_tensor_desc(
            sample_attn_score,
            name="sample_attn_score",
        )
        self.attn_l1norm_desc = self.make_tensor_desc(
            sample_attn_l1norm,
            name="sample_attn_l1norm",
        )
        self.index_score_desc = self.make_tensor_desc(
            sample_index_score,
            name="sample_index_score",
        )
        self.index_lse_desc = self.make_tensor_desc(
            sample_index_lse,
            name="sample_index_lse",
        )
        self.grad_loss_desc = self.make_tensor_desc(
            as_grad_loss_operand(sample_grad_loss),
            name="sample_grad_loss",
        )
        self.sm_scale = sm_scale
        self.loss_coeff = loss_coeff
        self.block_I = block_I
        self.ratio = ratio

    def _check_support(self) -> None:
        _dense_indexer_backward_impl(
            self.index_q_desc,
            self.weights_desc,
            self.index_k_desc,
            self.attn_score_desc,
            self.attn_l1norm_desc,
            self.index_score_desc,
            self.index_lse_desc,
            self.sm_scale,
            self.loss_coeff,
            self.grad_loss_desc,
            self.block_I,
            self.ratio,
            _validate_only=True,
        )

    def __call__(
        self,
        index_q: Any,
        weights: Any,
        index_k: Any,
        attn_score: Any,
        attn_l1norm: Any,
        index_score: Any,
        index_lse: Any,
        grad_loss: Any = 1.0,
    ) -> TupleDict:
        return super().__call__(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            grad_loss,
        )

    def _call_impl(
        self,
        index_q: Any,
        weights: Any,
        index_k: Any,
        attn_score: Any,
        attn_l1norm: Any,
        index_score: Any,
        index_lse: Any,
        grad_loss: Any = 1.0,
    ) -> TupleDict:
        for value, expected, name in (
            (index_q, self.index_q_desc, "index_q"),
            (weights, self.weights_desc, "weights"),
            (index_k, self.index_k_desc, "index_k"),
            (attn_score, self.attn_score_desc, "attn_score"),
            (attn_l1norm, self.attn_l1norm_desc, "attn_l1norm"),
            (index_score, self.index_score_desc, "index_score"),
            (index_lse, self.index_lse_desc, "index_lse"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        grad_loss_operand = as_grad_loss_operand(grad_loss)
        self.check_tensor_signature(
            grad_loss_operand,
            self.grad_loss_desc,
            name="grad_loss",
        )
        return _dense_indexer_backward_impl(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            self.sm_scale,
            self.loss_coeff,
            grad_loss_operand,
            self.block_I,
            self.ratio,
        )


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
) -> TupleDict:
    """Compute sparse indexer gradients with fixed-shape SM100 kernels."""

    return IndexerBackward(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        sample_grad_loss=grad_loss,
        block_I=block_I,
        topk_indices_global=topk_indices_global,
    )(index_q, weights, index_k, attn_score, index_score, topk_indices, grad_loss)


def dense_indexer_backward_wrapper(
    index_q: Any,
    weights: Any,
    index_k: Any,
    attn_score: Any,
    attn_l1norm: Any,
    index_score: Any,
    index_lse: Any,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    grad_loss: Any = 1.0,
    block_I: int = 128,
    ratio: int = 1,
) -> TupleDict:
    """Compute dense indexer gradients for compact BSHD inputs on SM100."""

    return DenseIndexerBackward(
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        sample_grad_loss=grad_loss,
        block_I=block_I,
        ratio=ratio,
    )(
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss,
    )


__all__ = [
    "DenseIndexerBackward",
    "IndexerBackward",
    "dense_indexer_backward_wrapper",
    "indexer_backward_wrapper",
]
