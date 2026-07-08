# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed and fully packed NSA sliding-window attention."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._jax import JaxApiBase, TupleDict
from ..jax_utils import (
    FIXED_LAYOUTS,
    describe_fixed_data,
    fixed_data_mode,
    normalize_attention_layout,
    normalize_supported_dtype,
    require_fixed_qkv,
)


class SlidingWindowAttention(JaxApiBase):
    """JAX callable for fixed BHSD/BSHD or fully packed THD attention.

    Packed inputs use cumulative sequence lengths and static maxima. Their
    dynamic values must start at zero, be monotonic, end at the packed token
    count, and not exceed the supplied maxima.
    """

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_cum_seqlen_q: Any | None = None,
        sample_cum_seqlen_k: Any | None = None,
        max_s_q: int | None = None,
        max_s_k: int | None = None,
        left_bound: int = 1,
        right_bound: int = 0,
        is_infer: bool = False,
        attn_scale: float | None = None,
        o_dtype: Any = None,
        layout: str | None = None,
    ) -> None:
        ranks = tuple(len(sample.shape) for sample in (sample_q, sample_k, sample_v))
        if len(set(ranks)) != 1:
            raise ValueError(f"Q, K, and V must use the same rank, got {ranks}")
        self.input_layout = normalize_attention_layout(layout, ranks[0])
        if self.input_layout in FIXED_LAYOUTS:
            if sample_cum_seqlen_q is not None or sample_cum_seqlen_k is not None:
                raise ValueError(
                    "cumulative sequence lengths must be omitted for fixed inputs"
                )
            if max_s_q is not None or max_s_k is not None:
                raise ValueError("max_s_q and max_s_k are only valid for THD layout")
            self._init_fixed(sample_q, sample_k, sample_v)
        else:
            self._init_packed(
                sample_q,
                sample_k,
                sample_v,
                sample_cum_seqlen_q,
                sample_cum_seqlen_k,
                max_s_q,
                max_s_k,
            )
        self.left_bound = int(left_bound)
        self.right_bound = int(right_bound)
        self.is_infer = bool(is_infer)
        self.attn_scale = None if attn_scale is None else float(attn_scale)
        self.output_dtype = normalize_supported_dtype(
            o_dtype,
            sample_q.dtype,
            "o_dtype",
            (jnp.float16, jnp.bfloat16),
        )
        if self.left_bound < 1:
            raise ValueError(f"left_bound must be at least 1, got {self.left_bound}")
        if self.right_bound < 0:
            raise ValueError(
                f"right_bound must be non-negative, got {self.right_bound}"
            )

    def _init_fixed(self, sample_q: Any, sample_k: Any, sample_v: Any) -> None:
        self.data_mode = fixed_data_mode(self.input_layout, kernel_axes="BHSD")
        self.q_desc = describe_fixed_data(
            sample_q,
            "sample_q",
            layout=self.input_layout,
            kernel_axes="BHSD",
        )
        self.k_desc = describe_fixed_data(
            sample_k,
            "sample_k",
            layout=self.input_layout,
            kernel_axes="BHSD",
        )
        self.v_desc = describe_fixed_data(
            sample_v,
            "sample_v",
            layout=self.input_layout,
            kernel_axes="BHSD",
        )
        require_fixed_qkv(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            operation_name="SlidingWindowAttention",
            kernel_axes="BHSD",
        )
        self.cum_q_desc = self.cum_k_desc = None
        self.max_s_q = self.max_s_k = None

    def _init_packed(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_cum_seqlen_q: Any | None,
        sample_cum_seqlen_k: Any | None,
        max_s_q: int | None,
        max_s_k: int | None,
    ) -> None:
        if sample_cum_seqlen_q is None or sample_cum_seqlen_k is None:
            raise ValueError(
                "packed THD inputs require cumulative Q and K sequence lengths"
            )
        if max_s_q is None or max_s_k is None:
            raise ValueError("packed THD inputs require max_s_q and max_s_k")

        self.data_mode = None
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.v_desc = self._to_tensor_desc(sample_v, "sample_v")
        self.cum_q_desc = self._to_tensor_desc(
            sample_cum_seqlen_q, "sample_cum_seqlen_q"
        )
        self.cum_k_desc = self._to_tensor_desc(
            sample_cum_seqlen_k, "sample_cum_seqlen_k"
        )
        self.total_q, self.num_query_heads, self.head_dim = self.q_desc.shape
        self.total_k, self.num_kv_heads, k_head_dim = self.k_desc.shape
        v_total, v_heads, value_dim = self.v_desc.shape
        dimensions = (
            self.total_q,
            self.total_k,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(
                f"SlidingWindowAttention dimensions must be positive, got {dimensions}"
            )
        if (v_total, v_heads, value_dim) != (
            self.total_k,
            self.num_kv_heads,
            self.head_dim,
        ) or k_head_dim != self.head_dim:
            raise ValueError("packed K and V metadata must match Q/K head dimensions")
        if self.q_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
            raise ValueError("packed Q, K, and V must use float16 or bfloat16")
        if (
            self.k_desc.cudnn_dtype != self.q_desc.cudnn_dtype
            or self.v_desc.cudnn_dtype != self.q_desc.cudnn_dtype
        ):
            raise ValueError("packed Q, K, and V must have the same dtype")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("H_q must be divisible by H_kv")
        if (
            self.cum_q_desc.ndim != 1
            or self.cum_q_desc.shape != self.cum_k_desc.shape
            or self.cum_q_desc.shape[0] < 2
        ):
            raise ValueError(
                "cumulative Q and K sequence lengths must have matching (B + 1,) shapes"
            )
        if (
            self.cum_q_desc.cudnn_dtype != data_type.INT32
            or self.cum_k_desc.cudnn_dtype != data_type.INT32
        ):
            raise ValueError("cumulative sequence lengths must use int32")
        self.batch = self.cum_q_desc.shape[0] - 1
        self.max_s_q = int(max_s_q)
        self.max_s_k = int(max_s_k)
        if (
            self.max_s_q <= 0
            or self.max_s_k <= 0
            or self.max_s_q > self.total_q
            or self.max_s_k > self.total_k
        ):
            raise ValueError(
                "max_s_q and max_s_k must be positive and no larger than packed token counts"
            )

    def check_support(self) -> bool:
        return True

    def __call__(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        cum_seqlen_q_tensor: Any | None = None,
        cum_seqlen_k_tensor: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        for value, desc in (
            (q_tensor, self.q_desc),
            (k_tensor, self.k_desc),
            (v_tensor, self.v_desc),
        ):
            self._check_tensor_signature(value, desc, mode=self.data_mode)

        if self.input_layout == "BHSD":
            if cum_seqlen_q_tensor is not None or cum_seqlen_k_tensor is not None:
                raise ValueError(
                    "cumulative sequence lengths must be omitted for fixed inputs"
                )
            q_btnh = jnp.transpose(q_tensor, (0, 2, 1, 3))
            k_btnh = jnp.transpose(k_tensor, (0, 2, 1, 3))
            v_btnh = jnp.transpose(v_tensor, (0, 2, 1, 3))
            query_lengths = key_value_lengths = None
        elif self.input_layout == "BSHD":
            if cum_seqlen_q_tensor is not None or cum_seqlen_k_tensor is not None:
                raise ValueError(
                    "cumulative sequence lengths must be omitted for fixed inputs"
                )
            q_btnh, k_btnh, v_btnh = q_tensor, k_tensor, v_tensor
            query_lengths = key_value_lengths = None
        else:
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError(
                    "packed THD inputs require cumulative Q and K sequence lengths"
                )
            self._check_tensor_signature(cum_seqlen_q_tensor, self.cum_q_desc)
            self._check_tensor_signature(cum_seqlen_k_tensor, self.cum_k_desc)
            query_lengths = jnp.diff(cum_seqlen_q_tensor)
            key_value_lengths = jnp.diff(cum_seqlen_k_tensor)
            q_btnh = self._pad_packed(q_tensor, cum_seqlen_q_tensor[:-1], self.max_s_q)
            k_btnh = self._pad_packed(k_tensor, cum_seqlen_k_tensor[:-1], self.max_s_k)
            v_btnh = self._pad_packed(v_tensor, cum_seqlen_k_tensor[:-1], self.max_s_k)
        attention_result = jax.nn.dot_product_attention(
            q_btnh,
            k_btnh,
            v_btnh,
            scale=self.attn_scale,
            query_seq_lengths=query_lengths,
            key_value_seq_lengths=key_value_lengths,
            local_window_size=(self.left_bound - 1, self.right_bound),
            implementation="cudnn",
            return_residual=not self.is_infer,
        )
        if self.is_infer:
            output_btnh = attention_result
            stats = None
        else:
            output_btnh, residual_btn = attention_result
            stats = residual_btn
        if self.input_layout == "THD":
            output = self._unpad_packed(
                output_btnh, cum_seqlen_q_tensor, self.total_q
            ).astype(self.output_dtype)
            if stats is not None:
                stats = self._unpad_packed(
                    stats[..., None], cum_seqlen_q_tensor, self.total_q
                )
        else:
            output = (
                jnp.transpose(output_btnh, (0, 2, 1, 3))
                if self.input_layout == "BHSD"
                else output_btnh
            ).astype(self.output_dtype)
            if stats is not None:
                stats = jnp.transpose(stats, (0, 2, 1))[..., None]
        return TupleDict(o_tensor=output, stats_tensor=stats)

    @staticmethod
    def _pad_packed(values: Any, starts: Any, max_seqlen: int) -> Any:
        padded = jnp.pad(values, ((0, max_seqlen), (0, 0), (0, 0)))
        slice_shape = (max_seqlen, values.shape[1], values.shape[2])
        return jax.vmap(
            lambda start: jax.lax.dynamic_slice(padded, (start, 0, 0), slice_shape)
        )(starts)

    @staticmethod
    def _unpad_packed(values: Any, cumulative: Any, total_tokens: int) -> Any:
        token = jnp.arange(total_tokens, dtype=cumulative.dtype)
        batch = jnp.searchsorted(cumulative[1:], token, side="right")
        position = token - cumulative[batch]
        return values[batch, position]


@partial(
    jax.jit,
    static_argnames=(
        "left_bound",
        "right_bound",
        "is_infer",
        "attn_scale",
        "o_dtype",
        "max_s_q",
        "max_s_k",
        "layout",
    ),
)
def sliding_window_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    cum_seqlen_q_tensor: Any | None = None,
    cum_seqlen_k_tensor: Any | None = None,
    max_s_q: int | None = None,
    max_s_k: int | None = None,
    left_bound: int = 1,
    right_bound: int = 0,
    is_infer: bool = False,
    attn_scale: float | None = None,
    o_dtype: Any = None,
    layout: str | None = None,
) -> TupleDict:
    """Compute fixed BHSD/BSHD or fully packed THD attention.

    Packed THD calls require cumulative sequence lengths and static
    ``max_s_q``/``max_s_k`` bounds. Arbitrary ragged byte-offset tensors are
    intentionally outside this functional JAX API.
    """

    samples = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype)
        for value in (q_tensor, k_tensor, v_tensor)
    )
    sample_cum_seqlen_q = (
        None
        if cum_seqlen_q_tensor is None
        else jax.ShapeDtypeStruct(cum_seqlen_q_tensor.shape, cum_seqlen_q_tensor.dtype)
    )
    sample_cum_seqlen_k = (
        None
        if cum_seqlen_k_tensor is None
        else jax.ShapeDtypeStruct(cum_seqlen_k_tensor.shape, cum_seqlen_k_tensor.dtype)
    )
    return SlidingWindowAttention(
        *samples,
        sample_cum_seqlen_q,
        sample_cum_seqlen_k,
        max_s_q=max_s_q,
        max_s_k=max_s_k,
        left_bound=left_bound,
        right_bound=right_bound,
        is_infer=is_infer,
        attn_scale=attn_scale,
        o_dtype=o_dtype,
        layout=layout,
    )(
        q_tensor,
        k_tensor,
        v_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
    )


__all__ = ["SlidingWindowAttention", "sliding_window_attention_wrapper"]
