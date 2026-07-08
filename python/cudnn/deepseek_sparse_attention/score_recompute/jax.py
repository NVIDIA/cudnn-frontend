# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX APIs for DSA sparse and dense score recomputation."""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax import JaxApiBase, JaxTensorDesc, TupleDict
from .config import DenseScoreKernelConfig, SparseScoreKernelConfig
from .op import (
    DenseScoreRecomputeOp,
    DenseScoreSm90Config,
    SUPPORTED_COMPUTE_CAPABILITIES,
    SparseScoreRecomputeOp,
    SparseScoreSm90Config,
)


def _compile_options(target_compute_capability: int) -> str:
    from ..utils.compiler import compile_options_for_target

    return compile_options_for_target(target_compute_capability)


def _optional_desc(api: JaxApiBase, value: Any | None, name: str) -> JaxTensorDesc | None:
    return None if value is None else api._to_tensor_desc(value, name)


def _check_optional_signature(api: JaxApiBase, value: Any | None, expected: JaxTensorDesc | None, name: str) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} was not part of the sample signature")
        return
    if value is None:
        raise ValueError(f"{name} is required by the sample signature")
    api._check_tensor_signature(value, expected)


class _SparseScoreRecompute(JaxApiBase):
    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_per_head: Any,
        sample_topk_indices: Any,
        *,
        score_type: str,
        softmax_scale: float,
        sample_out: Any | None,
        sample_topk_length: Any | None,
        qhead_per_kv_head: int | None,
        topk_indices_global: bool,
        target_compute_capability: int | None,
    ) -> None:
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.per_head_desc = self._to_tensor_desc(sample_per_head, "sample_per_head")
        self.topk_indices_desc = self._to_tensor_desc(sample_topk_indices, "sample_topk_indices")
        self.topk_length_desc = _optional_desc(self, sample_topk_length, "sample_topk_length")
        self.out_desc = self._default_output_desc() if sample_out is None else self._to_tensor_desc(sample_out, "sample_out")
        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            type(self).__name__,
        )
        self._op = SparseScoreRecomputeOp(
            q=self.q_desc,
            k=self.k_desc,
            per_head=self.per_head_desc,
            topk_indices=self.topk_indices_desc,
            output=self.out_desc,
            topk_length=self.topk_length_desc,
            score_type=score_type,
            softmax_scale=softmax_scale,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
            target_compute_capability=self.target_compute_capability,
        )

    def _default_output_desc(self) -> JaxTensorDesc:
        if self.topk_indices_desc.ndim != 3:
            raise ValueError(f"topk_indices must have rank 3, got shape {self.topk_indices_desc.shape}")
        return self.topk_indices_desc.compact_like(
            cudnn_dtype=data_type.FLOAT,
            shape=self.topk_indices_desc.shape,
            name="sample_out",
        )

    def check_support(self) -> bool:
        return self._op.check_support()

    def _call(self, q: Any, k: Any, per_head: Any, topk_indices: Any, topk_length: Any | None) -> Any:
        self.check_support()
        for value, expected in (
            (q, self.q_desc),
            (k, self.k_desc),
            (per_head, self.per_head_desc),
            (topk_indices, self.topk_indices_desc),
        ):
            self._check_tensor_signature(value, expected)
        _check_optional_signature(self, topk_length, self.topk_length_desc, "topk_length")

        if self.target_compute_capability == 90:
            kernel_k = jnp.expand_dims(k, axis=2)
            kernel_per_head = jnp.transpose(per_head, (0, 2, 1))
            inputs = (q, kernel_k, kernel_per_head, topk_indices)
        else:
            inputs = (q, k, per_head, topk_indices)

        workspace_descs = ()
        if topk_length is not None:
            inputs += (topk_length,)
        elif self.target_compute_capability != 90:
            workspace_descs = (
                self.topk_indices_desc.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(1, 1),
                    name="topk_length_workspace",
                ),
            )

        (output,) = self._call_kernel(
            inputs,
            launch=self._launch_kernel,
            output_descs=(self.out_desc,),
            workspace_descs=workspace_descs,
            compile_options=_compile_options(self.target_compute_capability),
        )
        return output

    def _launch_kernel(
        self,
        stream: Any,
        *arguments: Any,
    ) -> None:
        uses_workspace = self.topk_length_desc is None and self.target_compute_capability != 90
        if uses_workspace:
            *inputs, output, topk_length_workspace = arguments
        else:
            *inputs, output = arguments
            topk_length_workspace = None
        q, k, per_head, topk_indices, *optional_inputs = inputs
        topk_length = optional_inputs[0] if optional_inputs else topk_length_workspace

        if self.target_compute_capability == 90:
            import cutlass

            from .sparse_score_recompute_sm90 import SparseScoreRecomputeSm90

            config = cast(SparseScoreSm90Config, self._op.config)
            kernel = SparseScoreRecomputeSm90(
                cutlass.BFloat16,
                head_dim=self.q_desc.shape[-1],
                qhead_per_kvhead=cast(int, self._op.qhead_per_kv_head),
                tile_m=config.tile_m,
                tile_n=config.tile_n,
                KV_stage=config.kv_stage,
                num_threads=config.num_threads,
                swap_AB=True,
                topk_max=self.topk_indices_desc.shape[-1],
                is_index_scores=self._op.score_type == "indexer",
                softmax_scale=self._op.softmax_scale,
                has_topk_length=self.topk_length_desc is not None,
                num_head_tiles=config.num_head_tiles,
                is_sparse=True,
                output_log_probs=False,
                topk_indices_global=self._op.topk_indices_global,
            )
            kernel(q, k, topk_indices, stream, output, per_head, topk_length, None)
            return

        import cutlass

        from .sparse_score_recompute_sm100 import SparseScoreRecomputeSm100

        config = cast(SparseScoreKernelConfig, self._op.config)
        kernel = SparseScoreRecomputeSm100(
            head_dim=self.q_desc.shape[-1],
            qhead_per_kvhead=cast(int, self._op.qhead_per_kv_head),
            m_block_size=config.m_block_size,
            n_block_size=config.n_block_size,
            k_block_size=config.k_block_size,
            topk=self.topk_indices_desc.shape[-1],
            kv_stage=config.kv_stage,
            score_type=self._op.score_type,
            have_topk_length=config.have_topk_length,
            topk_in_smem=config.topk_in_smem,
            topk_indices_global=self._op.topk_indices_global,
        )
        kernel(q, k, per_head, topk_indices, output, topk_length, cutlass.Float32(self._op.softmax_scale), stream)


class SparseIndexerScoreRecompute(_SparseScoreRecompute):
    """JAX callable specialized for sparse indexer score recomputation."""

    def __init__(
        self,
        sample_q_indexer: Any,
        sample_k_indexer: Any,
        sample_weights: Any,
        sample_topk_indices: Any,
        sample_out: Any | None = None,
        sample_topk_length: Any | None = None,
        qhead_per_kv_head: int | None = None,
        topk_indices_global: bool = False,
        target_compute_capability: int | None = None,
    ) -> None:
        super().__init__(
            sample_q_indexer,
            sample_k_indexer,
            sample_weights,
            sample_topk_indices,
            score_type="indexer",
            softmax_scale=1.0,
            sample_out=sample_out,
            sample_topk_length=sample_topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
            target_compute_capability=target_compute_capability,
        )

    def __call__(
        self,
        q_indexer: Any,
        k_indexer: Any,
        weights: Any,
        topk_indices: Any,
        topk_length: Any | None = None,
    ) -> TupleDict:
        return TupleDict(predict=self._call(q_indexer, k_indexer, weights, topk_indices, topk_length))


class SparseAttnScoreRecompute(_SparseScoreRecompute):
    """JAX callable specialized for sparse attention score recomputation."""

    def __init__(
        self,
        sample_q_attn: Any,
        sample_k_attn: Any,
        sample_lse: Any,
        sample_topk_indices: Any,
        softmax_scale: float,
        sample_out: Any | None = None,
        sample_topk_length: Any | None = None,
        qhead_per_kv_head: int | None = None,
        topk_indices_global: bool = False,
        target_compute_capability: int | None = None,
    ) -> None:
        super().__init__(
            sample_q_attn,
            sample_k_attn,
            sample_lse,
            sample_topk_indices,
            score_type="attention",
            softmax_scale=softmax_scale,
            sample_out=sample_out,
            sample_topk_length=sample_topk_length,
            qhead_per_kv_head=qhead_per_kv_head,
            topk_indices_global=topk_indices_global,
            target_compute_capability=target_compute_capability,
        )

    def __call__(
        self,
        q_attn: Any,
        k_attn: Any,
        lse: Any,
        topk_indices: Any,
        topk_length: Any | None = None,
    ) -> TupleDict:
        return TupleDict(target=self._call(q_attn, k_attn, lse, topk_indices, topk_length))


class _DenseScoreRecompute(JaxApiBase):
    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_per_head: Any,
        *,
        score_type: str,
        scale: float,
        sample_out: Any | None,
        sample_denom_out: Any | None,
        qhead_per_kv_head: int | None,
        ratio: int,
        sample_cu_seqlens_q: Any | None,
        sample_cu_seqlens_k: Any | None,
        max_seqlen_q: int | None,
        max_seqlen_k: int | None,
        sample_q_causal_offsets: Any | None,
        target_compute_capability: int | None,
    ) -> None:
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.per_head_desc = self._to_tensor_desc(sample_per_head, "sample_per_head")
        self.cu_seqlens_q_desc = _optional_desc(self, sample_cu_seqlens_q, "sample_cu_seqlens_q")
        self.cu_seqlens_k_desc = _optional_desc(self, sample_cu_seqlens_k, "sample_cu_seqlens_k")
        self.q_causal_offsets_desc = _optional_desc(self, sample_q_causal_offsets, "sample_q_causal_offsets")
        self.is_thd = self.q_desc.ndim == 3
        self.requested_max_seqlen_q = max_seqlen_q
        self.requested_max_seqlen_k = max_seqlen_k

        default_out, default_denom = self._default_output_descs(max_seqlen_k)
        self.out_desc = default_out if sample_out is None else self._to_tensor_desc(sample_out, "sample_out", init_value=float("-inf"))
        self.denom_desc = default_denom if sample_denom_out is None else self._to_tensor_desc(sample_denom_out, "sample_denom_out")
        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            type(self).__name__,
        )
        self._op = DenseScoreRecomputeOp(
            q=self.q_desc,
            k=self.k_desc,
            per_head=self.per_head_desc,
            output=self.out_desc,
            denominator=self.denom_desc,
            score_type=score_type,
            scale=scale,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            is_thd=self.is_thd,
            cu_seqlens_q=self.cu_seqlens_q_desc,
            cu_seqlens_k=self.cu_seqlens_k_desc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=self.q_causal_offsets_desc,
            target_compute_capability=self.target_compute_capability,
        )

    def _default_output_descs(self, max_seqlen_k: int | None) -> tuple[JaxTensorDesc, JaxTensorDesc]:
        if self.is_thd:
            if max_seqlen_k is None:
                raise ValueError("max_seqlen_k is required to infer THD dense score outputs")
            out_shape = (self.q_desc.shape[0], int(max_seqlen_k))
            denom_shape = (self.q_desc.shape[0],)
        elif self.q_desc.ndim == 4 and self.k_desc.ndim == 4:
            out_shape = (self.q_desc.shape[0], self.q_desc.shape[1], self.k_desc.shape[1])
            denom_shape = self.q_desc.shape[:2]
        else:
            raise ValueError(f"Dense score inputs must both use BSHD or THD layout, got Q={self.q_desc.shape}, K={self.k_desc.shape}")
        return (
            self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=out_shape,
                name="sample_out",
                init_value=float("-inf"),
            ),
            self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=denom_shape,
                name="sample_denom_out",
            ),
        )

    def check_support(self) -> bool:
        return self._op.check_support()

    def _call(
        self,
        q: Any,
        k: Any,
        per_head: Any,
        cu_seqlens_q: Any | None,
        cu_seqlens_k: Any | None,
        q_causal_offsets: Any | None,
    ) -> TupleDict:
        self.check_support()
        for value, expected in ((q, self.q_desc), (k, self.k_desc), (per_head, self.per_head_desc)):
            self._check_tensor_signature(value, expected)
        _check_optional_signature(self, cu_seqlens_q, self.cu_seqlens_q_desc, "cu_seqlens_q")
        _check_optional_signature(self, cu_seqlens_k, self.cu_seqlens_k_desc, "cu_seqlens_k")
        _check_optional_signature(self, q_causal_offsets, self.q_causal_offsets_desc, "q_causal_offsets")

        if self.target_compute_capability == 90:
            kernel_per_head = jnp.transpose(per_head, (0, 2, 1))
            inputs = (q, k, kernel_per_head)
        else:
            inputs = (q, k, per_head)
            if self.is_thd:
                inputs += (cu_seqlens_q, cu_seqlens_k)
        if q_causal_offsets is not None:
            inputs += (q_causal_offsets,)

        output, denominator = self._call_kernel(
            inputs,
            launch=self._launch_kernel,
            output_descs=(self.out_desc, self.denom_desc),
            compile_options=_compile_options(self.target_compute_capability),
        )
        return TupleDict(out=output, denom=denominator)

    def _launch_kernel(
        self,
        stream: Any,
        *arguments: Any,
    ) -> None:
        *inputs, output, denominator = arguments
        q, k, per_head, *optional_inputs = inputs

        if self.target_compute_capability == 90:
            import cutlass

            from .dense_score_recompute_sm90 import DenseScoreRecomputeSm90

            q_causal_offsets = optional_inputs[0] if optional_inputs else None
            config = cast(DenseScoreSm90Config, self._op.config)
            kernel = DenseScoreRecomputeSm90(
                cutlass.BFloat16,
                head_dim=self.q_desc.shape[-1],
                qhead_per_kvhead=cast(int, self._op.qhead_per_kv_head),
                tile_m=config.tile_m,
                tile_n=config.tile_n,
                KV_stage=config.kv_stage,
                num_threads=config.num_threads,
                swap_AB=True,
                topk_max=cast(int, self._op.max_seqlen_k),
                is_index_scores=self._op.score_type == "indexer",
                softmax_scale=self._op.scale,
                has_topk_length=False,
                num_head_tiles=config.num_head_tiles,
                ratio=self._op.ratio,
            )
            kernel(q, k, None, stream, output, per_head, None, denominator, q_causal_offsets)
            return

        import cutlass

        from .dense_score_recompute_sm100 import DenseScoreRecomputeSm100

        if self.is_thd:
            cu_seqlens_q, cu_seqlens_k, *optional_inputs = optional_inputs
        else:
            cu_seqlens_q = cu_seqlens_k = None
        q_causal_offsets = optional_inputs[0] if optional_inputs else None
        config = cast(DenseScoreKernelConfig, self._op.config)
        kernel = DenseScoreRecomputeSm100(
            head_dim=self.q_desc.shape[-1],
            qhead_per_kvhead=cast(int, self._op.qhead_per_kv_head),
            m_block_size=config.m_block_size,
            n_block_size=config.n_block_size,
            k_block_size=config.k_block_size,
            kv_stage=config.kv_stage,
            score_type=self._op.score_type,
            ratio=self._op.ratio,
            is_varlen=self.is_thd,
        )
        kernel(
            q,
            k,
            per_head,
            output,
            denominator,
            cutlass.Float32(self._op.scale),
            cutlass.Int32(cast(int, self._op.max_seqlen_q)),
            cutlass.Int32(cast(int, self._op.max_seqlen_k)),
            cu_seqlens_q,
            cu_seqlens_k,
            q_causal_offsets,
            stream,
        )


class DenseIndexerScoreRecompute(_DenseScoreRecompute):
    """JAX callable specialized for dense indexer score recomputation."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_weights: Any,
        sample_out: Any | None = None,
        sample_denom_out: Any | None = None,
        qhead_per_kv_head: int | None = None,
        sm_scale: float = 1.0,
        ratio: int = 1,
        sample_cu_seqlens_q: Any | None = None,
        sample_cu_seqlens_k: Any | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        sample_q_causal_offsets: Any | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        super().__init__(
            sample_q,
            sample_k,
            sample_weights,
            score_type="indexer",
            scale=sm_scale,
            sample_out=sample_out,
            sample_denom_out=sample_denom_out,
            qhead_per_kv_head=qhead_per_kv_head,
            ratio=ratio,
            sample_cu_seqlens_q=sample_cu_seqlens_q,
            sample_cu_seqlens_k=sample_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sample_q_causal_offsets=sample_q_causal_offsets,
            target_compute_capability=target_compute_capability,
        )

    def __call__(
        self,
        q: Any,
        k: Any,
        weights: Any,
        cu_seqlens_q: Any | None = None,
        cu_seqlens_k: Any | None = None,
        q_causal_offsets: Any | None = None,
    ) -> TupleDict:
        return self._call(q, k, weights, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)


class DenseAttnScoreRecompute(_DenseScoreRecompute):
    """JAX callable specialized for dense attention score recomputation."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_lse: Any,
        softmax_scale: float,
        sample_out: Any | None = None,
        sample_denom_out: Any | None = None,
        qhead_per_kv_head: int | None = None,
        ratio: int = 1,
        sample_cu_seqlens_q: Any | None = None,
        sample_cu_seqlens_k: Any | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        sample_q_causal_offsets: Any | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        super().__init__(
            sample_q,
            sample_k,
            sample_lse,
            score_type="attention",
            scale=softmax_scale,
            sample_out=sample_out,
            sample_denom_out=sample_denom_out,
            qhead_per_kv_head=qhead_per_kv_head,
            ratio=ratio,
            sample_cu_seqlens_q=sample_cu_seqlens_q,
            sample_cu_seqlens_k=sample_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sample_q_causal_offsets=sample_q_causal_offsets,
            target_compute_capability=target_compute_capability,
        )

    def __call__(
        self,
        q: Any,
        k: Any,
        lse: Any,
        cu_seqlens_q: Any | None = None,
        cu_seqlens_k: Any | None = None,
        q_causal_offsets: Any | None = None,
    ) -> TupleDict:
        return self._call(q, k, lse, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)


@partial(
    jax.jit,
    static_argnames=("qhead_per_kv_head", "topk_indices_global", "target_compute_capability"),
)
def sparse_indexer_score_recompute_wrapper(
    q_indexer: Any,
    k_indexer: Any,
    weights: Any,
    topk_indices: Any,
    qhead_per_kv_head: int | None = None,
    topk_length: Any | None = None,
    topk_indices_global: bool = False,
    target_compute_capability: int | None = None,
) -> TupleDict:
    return SparseIndexerScoreRecompute(
        jax.ShapeDtypeStruct(q_indexer.shape, q_indexer.dtype),
        jax.ShapeDtypeStruct(k_indexer.shape, k_indexer.dtype),
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        jax.ShapeDtypeStruct(topk_indices.shape, topk_indices.dtype),
        sample_topk_length=None if topk_length is None else jax.ShapeDtypeStruct(topk_length.shape, topk_length.dtype),
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=topk_indices_global,
        target_compute_capability=target_compute_capability,
    )(q_indexer, k_indexer, weights, topk_indices, topk_length)


@partial(
    jax.jit,
    static_argnames=("softmax_scale", "qhead_per_kv_head", "topk_indices_global", "target_compute_capability"),
)
def sparse_attn_score_recompute_wrapper(
    q_attn: Any,
    k_attn: Any,
    lse: Any,
    topk_indices: Any,
    softmax_scale: float,
    qhead_per_kv_head: int | None = None,
    topk_length: Any | None = None,
    topk_indices_global: bool = False,
    target_compute_capability: int | None = None,
) -> TupleDict:
    return SparseAttnScoreRecompute(
        jax.ShapeDtypeStruct(q_attn.shape, q_attn.dtype),
        jax.ShapeDtypeStruct(k_attn.shape, k_attn.dtype),
        jax.ShapeDtypeStruct(lse.shape, lse.dtype),
        jax.ShapeDtypeStruct(topk_indices.shape, topk_indices.dtype),
        softmax_scale,
        sample_topk_length=None if topk_length is None else jax.ShapeDtypeStruct(topk_length.shape, topk_length.dtype),
        qhead_per_kv_head=qhead_per_kv_head,
        topk_indices_global=topk_indices_global,
        target_compute_capability=target_compute_capability,
    )(q_attn, k_attn, lse, topk_indices, topk_length)


@partial(
    jax.jit,
    static_argnames=(
        "qhead_per_kv_head",
        "sm_scale",
        "ratio",
        "max_seqlen_q",
        "max_seqlen_k",
        "target_compute_capability",
    ),
)
def dense_indexer_score_recompute_wrapper(
    q: Any,
    k: Any,
    weights: Any,
    qhead_per_kv_head: int | None = None,
    sm_scale: float = 1.0,
    ratio: int = 1,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    q_causal_offsets: Any | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    return DenseIndexerScoreRecompute(
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        qhead_per_kv_head=qhead_per_kv_head,
        sm_scale=sm_scale,
        ratio=ratio,
        sample_cu_seqlens_q=None if cu_seqlens_q is None else jax.ShapeDtypeStruct(cu_seqlens_q.shape, cu_seqlens_q.dtype),
        sample_cu_seqlens_k=None if cu_seqlens_k is None else jax.ShapeDtypeStruct(cu_seqlens_k.shape, cu_seqlens_k.dtype),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sample_q_causal_offsets=None if q_causal_offsets is None else jax.ShapeDtypeStruct(q_causal_offsets.shape, q_causal_offsets.dtype),
        target_compute_capability=target_compute_capability,
    )(q, k, weights, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)


@partial(
    jax.jit,
    static_argnames=(
        "softmax_scale",
        "qhead_per_kv_head",
        "ratio",
        "max_seqlen_q",
        "max_seqlen_k",
        "target_compute_capability",
    ),
)
def dense_attn_score_recompute_wrapper(
    q: Any,
    k: Any,
    lse: Any,
    softmax_scale: float,
    qhead_per_kv_head: int | None = None,
    ratio: int = 1,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    q_causal_offsets: Any | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    return DenseAttnScoreRecompute(
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(lse.shape, lse.dtype),
        softmax_scale,
        qhead_per_kv_head=qhead_per_kv_head,
        ratio=ratio,
        sample_cu_seqlens_q=None if cu_seqlens_q is None else jax.ShapeDtypeStruct(cu_seqlens_q.shape, cu_seqlens_q.dtype),
        sample_cu_seqlens_k=None if cu_seqlens_k is None else jax.ShapeDtypeStruct(cu_seqlens_k.shape, cu_seqlens_k.dtype),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sample_q_causal_offsets=None if q_causal_offsets is None else jax.ShapeDtypeStruct(q_causal_offsets.shape, q_causal_offsets.dtype),
        target_compute_capability=target_compute_capability,
    )(q, k, lse, cu_seqlens_q, cu_seqlens_k, q_causal_offsets)


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
