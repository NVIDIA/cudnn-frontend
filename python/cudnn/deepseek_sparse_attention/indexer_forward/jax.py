# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Optional JAX API for DeepSeek indexer forward."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ..._jax import JaxApiBase, JaxTensorDesc, TupleDict
from ..._jax.compiler import compile_options_for_target
from ..._jax.layout import mode_from_layout, to_public_axes
from .op import IndexerForwardOp, SUPPORTED_COMPUTE_CAPABILITIES, TMA_ALIGN_ELEMENTS


def _resolve_layout(
    name: str,
    layout: str | None,
    *,
    default: str,
    kernel_axes: str,
    supported: tuple[str, ...],
) -> tuple[str, tuple[int, ...]]:
    layout = default if layout is None else layout
    mode = mode_from_layout(layout, kernel_axes=kernel_axes)
    if layout not in supported:
        choices = ", ".join(repr(value) for value in supported)
        raise ValueError(f"{name} must be one of ({choices}), got {layout!r}")
    return layout, mode


class IndexerForward(JaxApiBase):
    """JAX callable specialized from fixed BSHD or packed THD metadata.

    Fixed inputs may use batch-major ``BSHD``/``BSH`` or sequence-major
    ``SBHD``/``SBH`` public axis orders. The output may similarly use ``BSK``
    or ``SBK``. Packed inputs currently use ``THD``/``TH`` and return ``TK``.

    ``sample_out`` describes the padded physical FP32 kernel buffer. When it
    is omitted, the adapter creates that descriptor and returns a logical view
    with the unpadded K extent.
    """

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_w: Any,
        *,
        sample_out: Any | None = None,
        sample_cu_seqlens_q: Any | None = None,
        sample_cu_seqlens_k: Any | None = None,
        sample_q_causal_offsets: Any | None = None,
        ratio: int = 4,
        qhead_per_kv_head: int | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
        q_layout: str | None = None,
        k_layout: str | None = None,
        w_layout: str | None = None,
        output_layout: str | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        if (sample_cu_seqlens_q is None) != (sample_cu_seqlens_k is None):
            raise ValueError("THD input requires both sample_cu_seqlens_q and sample_cu_seqlens_k")

        self.is_varlen = sample_cu_seqlens_q is not None
        if self.is_varlen:
            self.q_layout, self.q_mode = _resolve_layout(
                "q_layout",
                q_layout,
                default="THD",
                kernel_axes="THD",
                supported=("THD",),
            )
            self.k_layout, self.k_mode = _resolve_layout(
                "k_layout",
                k_layout,
                default="THD",
                kernel_axes="THD",
                supported=("THD",),
            )
            self.w_layout, self.w_mode = _resolve_layout(
                "w_layout",
                w_layout,
                default="TH",
                kernel_axes="TH",
                supported=("TH",),
            )
            self.output_layout, self.output_mode = _resolve_layout(
                "output_layout",
                output_layout,
                default="TK",
                kernel_axes="TK",
                supported=("TK",),
            )
        else:
            self.q_layout, self.q_mode = _resolve_layout(
                "q_layout",
                q_layout,
                default="BSHD",
                kernel_axes="BSHD",
                supported=("BSHD", "SBHD"),
            )
            self.k_layout, self.k_mode = _resolve_layout(
                "k_layout",
                k_layout,
                default="BSHD",
                kernel_axes="BSHD",
                supported=("BSHD", "SBHD"),
            )
            self.w_layout, self.w_mode = _resolve_layout(
                "w_layout",
                w_layout,
                default="BSH",
                kernel_axes="BSH",
                supported=("BSH", "SBH"),
            )
            self.output_layout, self.output_mode = _resolve_layout(
                "output_layout",
                output_layout,
                default="BSK",
                kernel_axes="BSK",
                supported=("BSK", "SBK"),
            )

        self.target_compute_capability = self._resolve_compute_capability(
            target_compute_capability=target_compute_capability,
            supported_compute_capabilities=SUPPORTED_COMPUTE_CAPABILITIES,
            operation_name="IndexerForward",
        )
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q", mode=self.q_mode)
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k", mode=self.k_mode)
        self.w_desc = self._to_tensor_desc(sample_w, "sample_w", mode=self.w_mode)
        self.cu_seqlens_q_desc = None if sample_cu_seqlens_q is None else self._to_tensor_desc(sample_cu_seqlens_q, "sample_cu_seqlens_q")
        self.cu_seqlens_k_desc = None if sample_cu_seqlens_k is None else self._to_tensor_desc(sample_cu_seqlens_k, "sample_cu_seqlens_k")
        self.q_causal_offsets_desc = (
            None
            if sample_q_causal_offsets is None
            else self._to_tensor_desc(
                sample_q_causal_offsets,
                "sample_q_causal_offsets",
            )
        )
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

        if sample_out is None:
            self.o_desc = self._default_output_desc(max_seqlen_k)
        else:
            self.o_desc = self._to_tensor_desc(
                sample_out,
                "sample_out",
                mode=self.output_mode,
                init_value=float("-inf"),
            )

        self._op = IndexerForwardOp(
            q=self.q_desc,
            k=self.k_desc,
            weight=self.w_desc,
            output=self.o_desc,
            cu_seqlens_q=self.cu_seqlens_q_desc,
            cu_seqlens_k=self.cu_seqlens_k_desc,
            q_causal_offsets=self.q_causal_offsets_desc,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            kv_stage=kv_stage,
            sm_scale=sm_scale,
            target_compute_capability=self.target_compute_capability,
        )

    def _default_output_desc(self, max_seqlen_k: int | None) -> JaxTensorDesc:
        if self.is_varlen:
            if self.q_desc.ndim != 3:
                raise ValueError(f"THD Q must have rank 3, got {self.q_desc.shape}")
            if max_seqlen_k is None:
                raise ValueError("THD input requires max_seqlen_k")
            leading_shape = (self.q_desc.shape[0],)
            logical_seqlen_k = int(max_seqlen_k)
        else:
            if self.q_desc.ndim != 4 or self.k_desc.ndim != 4:
                raise ValueError("Fixed indexer forward requires rank-4 Q and K")
            leading_shape = (self.q_desc.shape[0], self.q_desc.shape[1])
            logical_seqlen_k = self.k_desc.shape[1]

        padded_seqlen_k = ((logical_seqlen_k + TMA_ALIGN_ELEMENTS - 1) // TMA_ALIGN_ELEMENTS) * TMA_ALIGN_ELEMENTS
        canonical_shape = (*leading_shape, padded_seqlen_k)
        return JaxTensorDesc.from_shape(
            to_public_axes(canonical_shape, self.output_mode),
            jnp.float32,
            name="sample_out",
            mode=self.output_mode,
            init_value=float("-inf"),
        )

    def check_support(self) -> bool:
        return self._op.check_support()

    def __call__(
        self,
        q: Any,
        k: Any,
        w: Any,
        *,
        cu_seqlens_q: Any | None = None,
        cu_seqlens_k: Any | None = None,
        q_causal_offsets: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        self._check_optional_signature(cu_seqlens_q, self.cu_seqlens_q_desc, "cu_seqlens_q")
        self._check_optional_signature(cu_seqlens_k, self.cu_seqlens_k_desc, "cu_seqlens_k")
        self._check_optional_signature(q_causal_offsets, self.q_causal_offsets_desc, "q_causal_offsets")

        inputs = [q, k, w]
        input_descs = [self.q_desc, self.k_desc, self.w_desc]
        for value, desc in (
            (cu_seqlens_q, self.cu_seqlens_q_desc),
            (cu_seqlens_k, self.cu_seqlens_k_desc),
            (q_causal_offsets, self.q_causal_offsets_desc),
        ):
            if desc is not None:
                inputs.append(value)
                input_descs.append(desc)

        output_desc = self.o_desc.with_divisibility(
            (None,) * (self.o_desc.ndim - 1) + (TMA_ALIGN_ELEMENTS,)
        )
        (scores_padded,) = self._call_kernel(
            tuple(inputs),
            launch=self._launch_kernel,
            output_descs=(output_desc,),
            input_descs=tuple(input_descs),
            compile_options=compile_options_for_target(self.target_compute_capability),
        )
        if self._op.s_k is None:
            raise RuntimeError("IndexerForward output shape was not resolved by check_support()")
        output_slice = [slice(None)] * self.o_desc.ndim
        output_slice[self.output_mode[-1]] = slice(0, self._op.s_k)
        return TupleDict(scores=scores_padded[tuple(output_slice)])

    @staticmethod
    def _check_optional_signature(value: Any | None, desc: JaxTensorDesc | None, name: str) -> None:
        if (value is None) != (desc is None):
            expected = "omitted" if desc is None else "provided"
            raise ValueError(f"{name} must be {expected} for this specialized callable")

    def _launch_kernel(
        self,
        stream: Any,
        *arguments: Any,
    ) -> None:
        from cutlass import BFloat16, Float32, Int32

        *inputs, output = arguments
        q, k, w, *optional_inputs = inputs
        cu_seqlens_q = optional_inputs.pop(0) if self.cu_seqlens_q_desc is not None else None
        cu_seqlens_k = optional_inputs.pop(0) if self.cu_seqlens_k_desc is not None else None
        q_causal_offsets = optional_inputs.pop(0) if self.q_causal_offsets_desc is not None else None
        if optional_inputs:
            raise RuntimeError("Unexpected IndexerForward kernel inputs")
        resolved = (
            self._op.head_dim,
            self._op.qhead_per_kv_head,
            self._op.h_kv,
            self._op.s_q,
            self._op.s_k,
        )
        if any(value is None for value in resolved):
            raise RuntimeError("IndexerForward launch configuration was not resolved by check_support()")
        head_dim, qhead_per_kv_head, h_kv, s_q, s_k = resolved
        if self.target_compute_capability < 100:
            from .indexer_fwd_sm90 import IndexerForwardSm90

            kernel = IndexerForwardSm90(
                BFloat16,
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                ratio=self._op.ratio,
                is_varlen=bool(self._op.is_varlen),
            )
        else:
            from .indexer_fwd_sm100 import IndexerForwardSm100

            kernel = IndexerForwardSm100(
                head_dim=head_dim,
                qhead_per_kvhead=qhead_per_kv_head,
                ratio=self._op.ratio,
                is_varlen=bool(self._op.is_varlen),
                m_block_size=self._op.m_block_size,
                n_block_size=self._op.n_block_size,
                q_stage=self._op.q_stage,
                kv_stage=self._op.kv_stage,
            )

        kernel(
            q,
            k,
            w,
            output,
            Int32(h_kv),
            Int32(s_q),
            Int32(s_k),
            Float32(self._op.sm_scale),
            cu_seqlens_q,
            cu_seqlens_k,
            q_causal_offsets,
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "ratio",
        "qhead_per_kv_head",
        "max_seqlen_q",
        "max_seqlen_k",
        "m_block_size",
        "n_block_size",
        "q_stage",
        "kv_stage",
        "sm_scale",
        "q_layout",
        "k_layout",
        "w_layout",
        "output_layout",
        "target_compute_capability",
    ),
)
def indexer_forward_wrapper(
    q: Any,
    k: Any,
    w: Any,
    *,
    ratio: int = 4,
    qhead_per_kv_head: int | None = None,
    m_block_size: int = 128,
    n_block_size: int = 128,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    q_layout: str | None = None,
    k_layout: str | None = None,
    w_layout: str | None = None,
    output_layout: str | None = None,
    cu_seqlens_q: Any | None = None,
    cu_seqlens_k: Any | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    q_causal_offsets: Any | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute fixed batch/sequence-major or packed THD indexer scores."""

    return IndexerForward(
        q,
        k,
        w,
        sample_cu_seqlens_q=cu_seqlens_q,
        sample_cu_seqlens_k=cu_seqlens_k,
        sample_q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        qhead_per_kv_head=qhead_per_kv_head,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        q_stage=q_stage,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        q_layout=q_layout,
        k_layout=k_layout,
        w_layout=w_layout,
        output_layout=output_layout,
        target_compute_capability=target_compute_capability,
    )(
        q,
        k,
        w,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
    )


__all__ = ["IndexerForward", "indexer_forward_wrapper"]
