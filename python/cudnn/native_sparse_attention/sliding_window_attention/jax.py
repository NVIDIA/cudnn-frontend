# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape NSA sliding-window attention inference."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..._jax.api_base import ApiBaseJax, TupleDict, require_dtype
from ..jax_utils import require_bhsd_qkv


def _sliding_window_attention_impl(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    left_bound: int = 1,
    right_bound: int = 0,
    is_infer: bool = True,
    attn_scale: float | None = None,
    o_dtype: Any = None,
    *,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute fixed-shape BHSD sliding-window attention through cuDNN.

    The JAX binding covers the inference subset of the Torch API. Inputs use
    logical ``(B, H, S, D)`` shapes and are transposed to JAX's public BTNH
    attention contract before lowering to its registered cuDNN custom call.
    Packed THD inputs and training statistics are not supported.
    """

    (
        _,
        _,
        _,
        seqlen_q,
        seqlen_k,
        _,
        input_dtype,
    ) = require_bhsd_qkv(q_tensor, k_tensor, v_tensor)
    output_dtype = require_dtype(
        o_dtype,
        (jnp.float16, jnp.bfloat16),
        name="o_dtype",
        default=input_dtype,
    )

    if left_bound < 1:
        raise ValueError(f"left_bound must be at least 1, got {left_bound}")
    if seqlen_q != seqlen_k:
        raise NotImplementedError("JAX sliding-window attention currently requires S_q == S_k")
    if right_bound != 0:
        raise NotImplementedError("JAX cuDNN sliding-window attention currently requires right_bound=0")
    if not is_infer:
        raise NotImplementedError("JAX sliding-window attention currently supports inference only")

    scale = None if attn_scale is None else float(attn_scale)
    if _validate_only:
        return None

    q_btnh = jnp.transpose(q_tensor, (0, 2, 1, 3))
    k_btnh = jnp.transpose(k_tensor, (0, 2, 1, 3))
    v_btnh = jnp.transpose(v_tensor, (0, 2, 1, 3))
    output_btnh = jax.nn.dot_product_attention(
        q_btnh,
        k_btnh,
        v_btnh,
        scale=scale,
        is_causal=True,
        local_window_size=(int(left_bound) - 1, 0),
        implementation="cudnn",
    )
    output = jnp.transpose(output_btnh, (0, 2, 1, 3)).astype(output_dtype)
    return TupleDict(o_tensor=output, stats_tensor=None)


class SlidingWindowAttention(ApiBaseJax):
    """Sample-signature-bound JAX callable for sliding-window inference."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        left_bound: int = 1,
        right_bound: int = 0,
        is_infer: bool = True,
        attn_scale: float | None = None,
        o_dtype: Any = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self.make_tensor_desc(sample_v, name="sample_v")
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.is_infer = is_infer
        self.attn_scale = attn_scale
        self.o_dtype = self.as_optional_dtype(o_dtype)

    def _check_support(self) -> None:
        _sliding_window_attention_impl(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.left_bound,
            self.right_bound,
            self.is_infer,
            self.attn_scale,
            self.o_dtype,
            _validate_only=True,
        )

    def __call__(self, q_tensor: Any, k_tensor: Any, v_tensor: Any) -> TupleDict:
        return super().__call__(q_tensor, k_tensor, v_tensor)

    def _call_impl(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
    ) -> TupleDict:
        self.check_tensor_signature(q_tensor, self.q_desc, name="Q")
        self.check_tensor_signature(k_tensor, self.k_desc, name="K")
        self.check_tensor_signature(v_tensor, self.v_desc, name="V")
        return _sliding_window_attention_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            self.left_bound,
            self.right_bound,
            self.is_infer,
            self.attn_scale,
            self.o_dtype,
        )


def sliding_window_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    left_bound: int = 1,
    right_bound: int = 0,
    is_infer: bool = True,
    attn_scale: float | None = None,
    o_dtype: Any = None,
) -> TupleDict:
    """Compute fixed-shape BHSD sliding-window attention inference."""

    return SlidingWindowAttention(
        q_tensor,
        k_tensor,
        v_tensor,
        left_bound=left_bound,
        right_bound=right_bound,
        is_infer=is_infer,
        attn_scale=attn_scale,
        o_dtype=o_dtype,
    )(q_tensor, k_tensor, v_tensor)


__all__ = ["SlidingWindowAttention", "sliding_window_attention_wrapper"]
