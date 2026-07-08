# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX APIs for sparse and dense DeepSeek indexer backward."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax import JaxApiBase, JaxTensorDesc, TupleDict
from ..utils.compiler import compile_options_for_target
from .op import DEFAULT_BLOCK_I, DenseIndexerBackwardOp, IndexerBackwardOp

_SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)
_SUPPORTED_COMPUTE_CAPABILITY_FAMILIES = (90, 100)


def _grad_loss_operand(value: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.shape == ():
        return jnp.reshape(value, (1,))
    if value.shape != (1,):
        raise ValueError(f"grad_loss must be scalar or shape (1,), got {value.shape}")
    return value


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    return result


class _IndexerJaxBase(JaxApiBase):
    target_compute_capability: int | None
    compute_capability: int | None

    def _resolve_target(self, operation_name: str) -> None:
        self.compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            _SUPPORTED_COMPUTE_CAPABILITIES,
            operation_name,
        )

    @property
    def _architecture_family(self) -> int:
        if self.compute_capability is None:
            raise RuntimeError("check_support() must resolve the compute capability before lowering")
        family = self._compute_capability_family(self.compute_capability, _SUPPORTED_COMPUTE_CAPABILITY_FAMILIES)
        if family is None:
            raise RuntimeError(f"No indexer-backward kernel for SM{self.compute_capability}")
        return family

    @staticmethod
    def _output_desc(
        sample: Any | None,
        *,
        source: JaxTensorDesc,
        cudnn_dtype: data_type,
        shape: tuple[int, ...],
        name: str,
    ) -> JaxTensorDesc:
        if sample is not None:
            return JaxApiBase._to_tensor_desc(sample, name)
        return source.compact_like(cudnn_dtype=cudnn_dtype, shape=shape, name=name)

    def _last_axis_spec(self, desc: JaxTensorDesc, alignment: int) -> Any:
        return self._to_tensor_spec(
            desc,
            divisibility=(None,) * (desc.ndim - 1) + (alignment,),
        )


class IndexerBackward(_IndexerJaxBase):
    """Sparse indexer backward for BSHD or packed THD JAX arrays.

    Packed THD uses ``Q=(T_q,H,D)``, ``W=(T_q,H)``, ``K=(T_k,D)``, and
    score/index tensors shaped ``(T_q,topk)``. Packed top-K indices must be
    global flat K indices because this API intentionally has no host-visible
    sequence metadata from which to reconstruct per-batch offsets.
    """

    def __init__(
        self,
        sample_index_q: Any,
        sample_weights: Any,
        sample_index_k: Any,
        sample_attn_score: Any,
        sample_index_score: Any,
        sample_topk_indices: Any,
        sample_d_index_q: Any | None = None,
        sample_d_weights: Any | None = None,
        sample_d_index_k: Any | None = None,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        block_I: int = DEFAULT_BLOCK_I,
        topk_indices_global: bool = False,
        target_compute_capability: int | None = None,
    ) -> None:
        self.iq_desc = self._to_tensor_desc(sample_index_q, "sample_index_q")
        self.w_desc = self._to_tensor_desc(sample_weights, "sample_weights")
        self.ik_desc = self._to_tensor_desc(sample_index_k, "sample_index_k")
        self.attn_desc = self._to_tensor_desc(sample_attn_score, "sample_attn_score")
        self.idx_score_desc = self._to_tensor_desc(sample_index_score, "sample_index_score")
        self.topk_desc = self._to_tensor_desc(sample_topk_indices, "sample_topk_indices")
        self.diq_desc = self._output_desc(
            sample_d_index_q,
            source=self.iq_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.iq_desc.shape,
            name="sample_d_index_q",
        )
        self.dw_desc = self._output_desc(
            sample_d_weights,
            source=self.w_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.w_desc.shape,
            name="sample_d_weights",
        )
        self.dik_desc = self._output_desc(
            sample_d_index_k,
            source=self.ik_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.ik_desc.shape,
            name="sample_d_index_k",
        )
        self.grad_loss_desc = self.iq_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=(1,),
            name="grad_loss",
        )
        self._op = IndexerBackwardOp(
            index_q=self.iq_desc,
            weights=self.w_desc,
            index_k=self.ik_desc,
            d_index_q=self.diq_desc,
            d_weights=self.dw_desc,
            d_index_k=self.dik_desc,
            attn_score=self.attn_desc,
            index_score=self.idx_score_desc,
            topk_indices=self.topk_desc,
            sm_scale=sm_scale,
            block_i=block_I,
            topk_indices_global=topk_indices_global,
        )
        self.loss_coeff = _finite_float(loss_coeff, "loss_coeff")
        self.target_compute_capability = target_compute_capability
        self.compute_capability = None

    def check_support(self) -> bool:
        self._op.check_support()
        self._resolve_target("IndexerBackward")
        return True

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
        self.check_support()
        for value, expected in (
            (index_q, self.iq_desc),
            (weights, self.w_desc),
            (index_k, self.ik_desc),
            (attn_score, self.attn_desc),
            (index_score, self.idx_score_desc),
            (topk_indices, self.topk_desc),
        ):
            self._check_tensor_signature(value, expected)
        grad_loss = _grad_loss_operand(grad_loss)
        self._check_tensor_signature(grad_loss, self.grad_loss_desc)

        dk_accum_desc = self.ik_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.ik_desc.shape,
            name="d_index_k_accum",
            init_value=0.0,
        )
        grad_signal_desc = self.attn_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.attn_desc.shape,
            name="grad_signal",
        )
        d_index_q, d_weights, d_index_k_accum = self._call_kernel(
            (index_q, weights, index_k, attn_score, index_score, topk_indices, grad_loss),
            launch=self._launch_kernel,
            output_descs=(self.diq_desc, self.dw_desc, dk_accum_desc),
            workspace_descs=(grad_signal_desc,),
            input_spec=(
                self._last_axis_spec(self.iq_desc, self._op.head_dim),
                None,
                self._last_axis_spec(self.ik_desc, self._op.head_dim),
                None,
                None,
                None,
                None,
            ),
            output_spec=(
                self._last_axis_spec(self.diq_desc, self._op.head_dim),
                None,
                self._last_axis_spec(dk_accum_desc, self._op.head_dim),
            ),
            compile_options=compile_options_for_target(self.compute_capability, "--opt-level 3"),
        )
        return TupleDict(
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k_accum.astype(self.dik_desc.dtype),
        )

    def _launch_kernel(
        self,
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
    ) -> None:
        import cutlass

        grad_scale = self.loss_coeff / (self._op.batch * self._op.seqlen_q)

        if self._op.is_thd:
            index_q = _prepend_unit_batch(index_q)
            weights = _prepend_unit_batch(weights)
            index_k = _prepend_unit_batch(index_k)
            attn_score = _prepend_unit_batch(attn_score)
            index_score = _prepend_unit_batch(index_score)
            topk_indices = _prepend_unit_batch(topk_indices)
            d_index_q = _prepend_unit_batch(d_index_q)
            d_weights = _prepend_unit_batch(d_weights)
            d_index_k_accum = _prepend_unit_batch(d_index_k_accum)
            grad_signal = _prepend_unit_batch(grad_signal)

        if self._architecture_family == 100:
            from .indexer_backward_sm100 import IndexerBackwardSm100, ScoreGradSm100

            ScoreGradSm100(topk=self._op.topk)(
                attn_score,
                index_score,
                grad_loss,
                cutlass.Float32(grad_scale),
                stream,
                grad_signal,
                None,
            )
            IndexerBackwardSm100(
                head_dim=self._op.head_dim,
                heads=self._op.heads,
                block_I=self._op.block_i,
                topk=self._op.topk,
                topk_indices_global=self._op.topk_indices_global,
            )(
                index_q,
                weights,
                index_k,
                d_index_q,
                d_weights,
                d_index_k_accum,
                grad_signal,
                topk_indices,
                cutlass.Float32(self._op.sm_scale),
                stream,
            )
            return

        from .indexer_backward_sm90 import IndexerBackwardSm90, ScoreGradSm90

        ScoreGradSm90(topk=self._op.topk, index_is_log=False)(
            attn_score,
            index_score,
            grad_loss,
            cutlass.Float32(grad_scale),
            stream,
            grad_signal,
        )
        IndexerBackwardSm90(
            head_dim=self._op.head_dim,
            heads=self._op.heads,
            block_I=self._op.block_i,
            topk=self._op.topk,
            topk_indices_global=self._op.topk_indices_global,
        )(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k_accum,
            grad_signal,
            topk_indices,
            cutlass.Float32(self._op.sm_scale),
            stream,
            None,
            None,
            cutlass.Int32(self._op.seqlen_q),
            cutlass.Int32(self._op.seqlen_k),
            None,
        )


def _prepend_unit_batch(tensor: Any) -> Any:
    """Expose a compact packed tensor to the sparse kernels as batch one."""

    import cutlass.cute as cute

    leading_stride = tensor.shape[0] * tensor.stride[0]
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, *tensor.shape),
            stride=(leading_stride, *tensor.stride),
        ),
    )


class DenseIndexerBackward(_IndexerJaxBase):
    """Sample-signature-bound dense BSHD or packed-THD backward callable."""

    def __init__(
        self,
        sample_index_q: Any,
        sample_weights: Any,
        sample_index_k: Any,
        sample_attn_score: Any,
        sample_attn_l1norm: Any,
        sample_index_score: Any,
        sample_index_lse: Any,
        sample_d_index_q: Any | None = None,
        sample_d_weights: Any | None = None,
        sample_d_index_k: Any | None = None,
        sample_cu_seqlens_q: Any | None = None,
        sample_cu_seqlens_k: Any | None = None,
        sample_q_causal_offsets: Any | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        sm_scale: float = 1.0,
        loss_coeff: float = 1.0,
        block_I: int = DEFAULT_BLOCK_I,
        ratio: int = 1,
        target_compute_capability: int | None = None,
    ) -> None:
        self.iq_desc = self._to_tensor_desc(sample_index_q, "sample_index_q")
        self.w_desc = self._to_tensor_desc(sample_weights, "sample_weights")
        self.ik_desc = self._to_tensor_desc(sample_index_k, "sample_index_k")
        self.attn_desc = self._to_tensor_desc(sample_attn_score, "sample_attn_score")
        self.attn_denom_desc = self._to_tensor_desc(sample_attn_l1norm, "sample_attn_l1norm")
        self.idx_score_desc = self._to_tensor_desc(sample_index_score, "sample_index_score")
        self.idx_lse_desc = self._to_tensor_desc(sample_index_lse, "sample_index_lse")
        self.diq_desc = self._output_desc(
            sample_d_index_q,
            source=self.iq_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.iq_desc.shape,
            name="sample_d_index_q",
        )
        self.dw_desc = self._output_desc(
            sample_d_weights,
            source=self.w_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.w_desc.shape,
            name="sample_d_weights",
        )
        self.dik_desc = self._output_desc(
            sample_d_index_k,
            source=self.ik_desc,
            cudnn_dtype=data_type.BFLOAT16,
            shape=self.ik_desc.shape,
            name="sample_d_index_k",
        )
        self.grad_loss_desc = self.iq_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=(1,),
            name="grad_loss",
        )
        self.cuq_desc = None if sample_cu_seqlens_q is None else self._to_tensor_desc(sample_cu_seqlens_q, "sample_cu_seqlens_q")
        self.cuk_desc = None if sample_cu_seqlens_k is None else self._to_tensor_desc(sample_cu_seqlens_k, "sample_cu_seqlens_k")
        self.q_offsets_desc = None if sample_q_causal_offsets is None else self._to_tensor_desc(sample_q_causal_offsets, "sample_q_causal_offsets")
        self._op = DenseIndexerBackwardOp(
            index_q=self.iq_desc,
            weights=self.w_desc,
            index_k=self.ik_desc,
            d_index_q=self.diq_desc,
            d_weights=self.dw_desc,
            d_index_k=self.dik_desc,
            attn_score=self.attn_desc,
            attn_l1norm=self.attn_denom_desc,
            index_score=self.idx_score_desc,
            index_lse=self.idx_lse_desc,
            cu_seqlens_q=self.cuq_desc,
            cu_seqlens_k=self.cuk_desc,
            q_causal_offsets=self.q_offsets_desc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            block_i=block_I,
            ratio=ratio,
        )
        self.loss_coeff = _finite_float(loss_coeff, "loss_coeff")
        self.target_compute_capability = target_compute_capability
        self.compute_capability = None

    def check_support(self) -> bool:
        self._op.check_support()
        self._resolve_target("DenseIndexerBackward")
        return True

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
        cu_seqlens_q: Any | None = None,
        cu_seqlens_k: Any | None = None,
        q_causal_offsets: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        for value, expected in (
            (index_q, self.iq_desc),
            (weights, self.w_desc),
            (index_k, self.ik_desc),
            (attn_score, self.attn_desc),
            (attn_l1norm, self.attn_denom_desc),
            (index_score, self.idx_score_desc),
            (index_lse, self.idx_lse_desc),
        ):
            self._check_tensor_signature(value, expected)
        for value, expected, name in (
            (cu_seqlens_q, self.cuq_desc, "cu_seqlens_q"),
            (cu_seqlens_k, self.cuk_desc, "cu_seqlens_k"),
            (q_causal_offsets, self.q_offsets_desc, "q_causal_offsets"),
        ):
            if (value is None) != (expected is None):
                raise ValueError(f"{name} presence must match its sample")
            if value is not None:
                self._check_tensor_signature(value, expected)
        grad_loss = _grad_loss_operand(grad_loss)
        self._check_tensor_signature(grad_loss, self.grad_loss_desc)

        inputs = (index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, grad_loss)
        for optional in (cu_seqlens_q, cu_seqlens_k, q_causal_offsets):
            if optional is not None:
                inputs += (optional,)
        dk_accum_desc = self.ik_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.ik_desc.shape,
            name="d_index_k_accum",
            init_value=0.0,
        )
        grad_signal_desc = self.idx_score_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.idx_score_desc.shape,
            name="grad_signal",
        )
        d_index_q, d_weights, d_index_k_accum = self._call_kernel(
            inputs,
            launch=self._launch_kernel,
            output_descs=(self.diq_desc, self.dw_desc, dk_accum_desc),
            workspace_descs=(grad_signal_desc,),
            input_spec=(
                self._last_axis_spec(self.iq_desc, self._op.head_dim),
                None,
                self._last_axis_spec(self.ik_desc, self._op.head_dim),
            )
            + (None,) * (len(inputs) - 3),
            output_spec=(
                self._last_axis_spec(self.diq_desc, self._op.head_dim),
                None,
                self._last_axis_spec(dk_accum_desc, self._op.head_dim),
            ),
            compile_options=compile_options_for_target(self.compute_capability, "--opt-level 3"),
        )
        return TupleDict(
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k_accum.astype(self.dik_desc.dtype),
        )

    def _launch_kernel(self, stream, *arguments) -> None:
        import cutlass

        *inputs, d_index_q, d_weights, d_index_k_accum, grad_signal = arguments
        index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse, grad_loss, *optional = inputs
        cursor = 0
        cu_seqlens_q = cu_seqlens_k = q_causal_offsets = None
        if self._op.is_thd:
            cu_seqlens_q, cu_seqlens_k = optional[:2]
            cursor = 2
        if self._op.q_causal_offsets is not None:
            q_causal_offsets = optional[cursor]

        grad_scale = self.loss_coeff / max(self._op.normalization_tokens, 1)

        if self._architecture_family == 100:
            from .dense_indexer_backward_sm100 import DenseIndexerBackward2QGemmSm100, ScoreGradDense

            ScoreGradDense(ratio=self._op.ratio, block_I=self._op.block_i)(
                index_score,
                attn_score,
                index_lse,
                attn_l1norm,
                cu_seqlens_q,
                cu_seqlens_k,
                q_causal_offsets,
                cutlass.Float32(grad_scale),
                cutlass.Int32(self._op.max_seqlen_q),
                cutlass.Int32(self._op.max_seqlen_k),
                stream,
                grad_signal,
                grad_loss,
            )
            DenseIndexerBackward2QGemmSm100(
                head_dim=self._op.head_dim,
                heads=self._op.heads,
                block_I=self._op.block_i,
                ratio=self._op.ratio,
            )(
                index_q,
                weights,
                index_k,
                d_index_q,
                d_weights,
                d_index_k_accum,
                grad_signal,
                cu_seqlens_q,
                cu_seqlens_k,
                q_causal_offsets,
                cutlass.Float32(self._op.sm_scale),
                cutlass.Int32(self._op.max_seqlen_q),
                cutlass.Int32(self._op.max_seqlen_k),
                stream,
            )
            return

        from .dense_indexer_backward_sm90 import ScoreGradDenseSm90
        from .indexer_backward_sm90 import IndexerBackwardSm90

        ScoreGradDenseSm90(ratio=self._op.ratio, block_I=self._op.block_i)(
            index_score,
            attn_score,
            index_lse,
            attn_l1norm,
            cu_seqlens_q,
            cu_seqlens_k,
            q_causal_offsets,
            cutlass.Float32(grad_scale),
            cutlass.Int32(self._op.max_seqlen_q),
            cutlass.Int32(self._op.max_seqlen_k),
            stream,
            grad_signal,
            grad_loss,
        )
        IndexerBackwardSm90(
            head_dim=self._op.head_dim,
            heads=self._op.heads,
            block_I=self._op.block_i,
            topk=self._op.max_seqlen_k,
            is_dense=True,
            ratio=self._op.ratio,
        )(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k_accum,
            grad_signal,
            None,
            cutlass.Float32(self._op.sm_scale),
            stream,
            cu_seqlens_q,
            cu_seqlens_k,
            cutlass.Int32(self._op.max_seqlen_q),
            cutlass.Int32(self._op.max_seqlen_k),
            q_causal_offsets,
        )


@jax.jit(static_argnames=("sm_scale", "loss_coeff", "block_I", "topk_indices_global", "target_compute_capability"))
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
    block_I: int = DEFAULT_BLOCK_I,
    topk_indices_global: bool = False,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute sparse indexer gradients for fixed BSHD or packed THD inputs.

    Packed THD callers must pass ``topk_indices_global=True``.
    """

    samples = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (index_q, weights, index_k, attn_score, index_score, topk_indices))
    return IndexerBackward(
        *samples,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        block_I=block_I,
        topk_indices_global=topk_indices_global,
        target_compute_capability=target_compute_capability,
    )(index_q, weights, index_k, attn_score, index_score, topk_indices, grad_loss)


@jax.jit(
    static_argnames=(
        "sm_scale",
        "loss_coeff",
        "block_I",
        "ratio",
        "max_seqlen_q",
        "max_seqlen_k",
        "target_compute_capability",
    )
)
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
    block_I: int = DEFAULT_BLOCK_I,
    ratio: int = 1,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    q_causal_offsets: Any | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    core = (index_q, weights, index_k, attn_score, attn_l1norm, index_score, index_lse)
    samples = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in core)
    sample_cuq = None if cu_seqlens_q is None else jax.ShapeDtypeStruct(cu_seqlens_q.shape, cu_seqlens_q.dtype)
    sample_cuk = None if cu_seqlens_k is None else jax.ShapeDtypeStruct(cu_seqlens_k.shape, cu_seqlens_k.dtype)
    sample_offsets = None if q_causal_offsets is None else jax.ShapeDtypeStruct(q_causal_offsets.shape, q_causal_offsets.dtype)
    return DenseIndexerBackward(
        *samples,
        sample_cu_seqlens_q=sample_cuq,
        sample_cu_seqlens_k=sample_cuk,
        sample_q_causal_offsets=sample_offsets,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        block_I=block_I,
        ratio=ratio,
        target_compute_capability=target_compute_capability,
    )(
        *core,
        grad_loss=grad_loss,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
    )


__all__ = [
    "DenseIndexerBackward",
    "IndexerBackward",
    "dense_indexer_backward_wrapper",
    "indexer_backward_wrapper",
]
