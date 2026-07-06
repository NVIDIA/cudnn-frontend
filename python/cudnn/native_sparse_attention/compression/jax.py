# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 NSA compression attention."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp

from ..._jax.api_base import (
    ApiBaseJax,
    BufferSpec,
    TupleDict,
    call_cutedsl,
    require_dtype,
)
from ..jax_utils import (
    bhs_lse_as_bsh_spec,
    bhsd_storage_spec,
    require_bhsd_qkv,
)


def _launch(
    stream,
    q,
    k,
    v,
    output,
    lse=None,
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    enable_lse: bool,
    is_persistent: bool,
    scale_softmax: float,
    scale_output: float,
):
    from cutlass import Float32, Int32

    from .fmha import BlackwellFusedMultiHeadAttentionForward
    from .fmha_helpers import MaskType

    kernel = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=Float32,
        pv_acc_dtype=Float32,
        mma_tiler=(128, 128, head_dim),
        is_persistent=is_persistent,
        mask_type=MaskType.COMPRESSED_CAUSAL_MASK,
    )
    problem_size = tuple(
        Int32(value)
        for value in (
            batch,
            seqlen_q,
            seqlen_q,
            seqlen_k,
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
    )

    kernel(
        q,
        k,
        v,
        output,
        problem_size,
        None,
        None,
        lse if enable_lse else None,
        Float32(scale_softmax * math.log2(math.e)),
        Float32(scale_softmax),
        Float32(scale_output),
        None,
        Int32(0),
        stream,
    )


def _compression_attention_impl(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    enable_lse: bool = False,
    o_dtype: Any = None,
    qk_acc_dtype: Any = None,
    pv_acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    is_persistent: bool = False,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    inv_scale_o: float = 1.0,
    scale_softmax: float | None = None,
    *,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute fixed-shape BHSD compression attention on SM100.

    Q, K, and V use logical shapes ``(B, H, S, D)`` and a shared ``float16``
    or ``bfloat16`` dtype. The head dimension must be 32, 64, or 128. The
    returned O has Q's shape and ``o_dtype``; optional LSE has shape
    ``(B, H_q, S_q)`` and dtype ``float32``.

    Variable-length THD inputs are not part of this API. Configuration values
    must be static while tracing with :func:`jax.jit`.
    """

    (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        head_dim,
        input_dtype,
    ) = require_bhsd_qkv(q_tensor, k_tensor, v_tensor)

    require_dtype(
        qk_acc_dtype,
        (jnp.float32,),
        name="qk_acc_dtype",
        default=jnp.float32,
    )
    require_dtype(
        pv_acc_dtype,
        (jnp.float32,),
        name="pv_acc_dtype",
        default=jnp.float32,
    )
    output_dtype = require_dtype(
        o_dtype,
        (jnp.float16, jnp.bfloat16),
        name="o_dtype",
        default=input_dtype,
    )
    if mma_tiler_mn != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
    if seqlen_q < seqlen_k or seqlen_q % seqlen_k:
        raise ValueError("Compression attention requires S_q to be an integer multiple of " f"S_k, got S_q={seqlen_q} and S_k={seqlen_k}")

    base_softmax_scale = 1.0 / math.sqrt(head_dim) if scale_softmax is None else float(scale_softmax)
    resolved_softmax_scale = float(scale_q) * float(scale_k) * base_softmax_scale
    resolved_output_scale = float(scale_v) * float(inv_scale_o)
    if _validate_only:
        return None

    input_spec = bhsd_storage_spec(present_as_bshd=True)
    lse_spec = bhs_lse_as_bsh_spec()

    outputs = [
        BufferSpec(
            "o_tensor",
            tuple(q_tensor.shape),
            output_dtype,
            tensor_spec=input_spec,
        )
    ]
    if enable_lse:
        outputs.append(
            BufferSpec(
                "lse_tensor",
                (batch, num_query_heads, seqlen_q),
                jnp.float32,
                tensor_spec=lse_spec,
            )
        )

    results = call_cutedsl(
        _launch,
        (q_tensor, k_tensor, v_tensor),
        outputs=tuple(outputs),
        input_specs=(input_spec, input_spec, input_spec),
        static_args={
            "batch": batch,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "enable_lse": bool(enable_lse),
            "is_persistent": bool(is_persistent),
            "scale_softmax": resolved_softmax_scale,
            "scale_output": resolved_output_scale,
        },
    )
    return TupleDict(
        o_tensor=results[0],
        lse_tensor=results[1] if enable_lse else None,
    )


class CompressionAttention(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 compression attention."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        enable_lse: bool = False,
        o_dtype: Any = None,
        qk_acc_dtype: Any = None,
        pv_acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        is_persistent: bool = False,
        scale_q: float = 1.0,
        scale_k: float = 1.0,
        scale_v: float = 1.0,
        inv_scale_o: float = 1.0,
        scale_softmax: float | None = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self.make_tensor_desc(sample_v, name="sample_v")
        self.enable_lse = enable_lse
        self.o_dtype = self.as_optional_dtype(o_dtype)
        self.qk_acc_dtype = self.as_optional_dtype(qk_acc_dtype)
        self.pv_acc_dtype = self.as_optional_dtype(pv_acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.is_persistent = is_persistent
        self.scale_q = scale_q
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.inv_scale_o = inv_scale_o
        self.scale_softmax = scale_softmax

    def _check_support(self) -> None:
        _compression_attention_impl(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.enable_lse,
            self.o_dtype,
            self.qk_acc_dtype,
            self.pv_acc_dtype,
            self.mma_tiler_mn,
            self.is_persistent,
            self.scale_q,
            self.scale_k,
            self.scale_v,
            self.inv_scale_o,
            self.scale_softmax,
            _validate_only=True,
        )

    def __call__(self, q_tensor: Any, k_tensor: Any, v_tensor: Any) -> TupleDict:
        return super().__call__(q_tensor, k_tensor, v_tensor)

    def _call_impl(self, q_tensor: Any, k_tensor: Any, v_tensor: Any) -> TupleDict:
        self.check_tensor_signature(q_tensor, self.q_desc, name="Q")
        self.check_tensor_signature(k_tensor, self.k_desc, name="K")
        self.check_tensor_signature(v_tensor, self.v_desc, name="V")
        return _compression_attention_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            self.enable_lse,
            self.o_dtype,
            self.qk_acc_dtype,
            self.pv_acc_dtype,
            self.mma_tiler_mn,
            self.is_persistent,
            self.scale_q,
            self.scale_k,
            self.scale_v,
            self.inv_scale_o,
            self.scale_softmax,
        )


def compression_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    enable_lse: bool = False,
    o_dtype: Any = None,
    qk_acc_dtype: Any = None,
    pv_acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    is_persistent: bool = False,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    inv_scale_o: float = 1.0,
    scale_softmax: float | None = None,
) -> TupleDict:
    """Compute fixed-shape BHSD compression attention on SM100."""

    return CompressionAttention(
        q_tensor,
        k_tensor,
        v_tensor,
        enable_lse=enable_lse,
        o_dtype=o_dtype,
        qk_acc_dtype=qk_acc_dtype,
        pv_acc_dtype=pv_acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_persistent=is_persistent,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
        inv_scale_o=inv_scale_o,
        scale_softmax=scale_softmax,
    )(q_tensor, k_tensor, v_tensor)


__all__ = [
    "CompressionAttention",
    "compression_attention_wrapper",
]
