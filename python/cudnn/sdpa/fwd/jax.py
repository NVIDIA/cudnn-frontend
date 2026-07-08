# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed BHSD/BSHD and packed THD SM100 d=256 SDPA forward."""

from __future__ import annotations

from functools import partial
import math
from typing import Any

import jax
import jax.numpy as jnp

from ... import data_type
from ..._cute_compiler import compile_options_for_target
from ..._jax import JaxApiBase, TupleDict
from ..jax_utils import (
    FIXED_LAYOUTS,
    describe_fixed_data,
    fixed_data_mode,
    make_fixed_output,
    normalize_sdpa_layout,
    require_fixed_qkv,
    require_float32_dtype,
    resolve_sdpa_config,
)

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)
_PACKED_LAYOUT = "THD"


class SdpafwdSm100D256(JaxApiBase):
    """JAX callable specialized from fixed BHSD/BSHD or packed THD metadata.

    Packed THD inputs require cumulative sequence lengths and explicit static
    ``max_s_q``/``max_s_k`` bounds. JAX samples expose array metadata rather
    than cumulative-length values, so these bounds cannot be inferred while
    tracing.
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
        qk_acc_dtype: Any = None,
        pv_acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        is_causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        scale_softmax: float | None = None,
        scale_output: float = 1.0,
        layout: str | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        ranks = tuple(
            len(tuple(sample.shape)) for sample in (sample_q, sample_k, sample_v)
        )
        if len(set(ranks)) != 1:
            raise ValueError(f"Q, K, and V must use the same rank, got {ranks}")
        self.input_layout = normalize_sdpa_layout(layout, ranks[0])
        if self.input_layout in FIXED_LAYOUTS:
            if max_s_q is not None or max_s_k is not None:
                raise ValueError("max_s_q and max_s_k are only valid for THD layout")
            self.data_mode = fixed_data_mode(self.input_layout)
            self._init_fixed_shape(
                sample_q,
                sample_k,
                sample_v,
                sample_cum_seqlen_q,
                sample_cum_seqlen_k,
            )
        elif self.input_layout == _PACKED_LAYOUT:
            self.data_mode = None
            self._init_packed_shape(
                sample_q,
                sample_k,
                sample_v,
                sample_cum_seqlen_q,
                sample_cum_seqlen_k,
                max_s_q,
                max_s_k,
            )

        require_float32_dtype(qk_acc_dtype, "qk_acc_dtype")
        require_float32_dtype(pv_acc_dtype, "pv_acc_dtype")
        if tuple(mma_tiler_mn) != (128, 128):
            raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
        self.is_causal = bool(is_causal)
        (
            self.scale_softmax,
            self.window_size_left,
            self.window_size_right,
            self.mask_kind,
        ) = resolve_sdpa_config(
            seqlen_q=self.max_s_q,
            seqlen_k=self.max_s_k,
            # The Torch API always selects the residual mask for non-causal
            # varlen inputs, independent of the packed token count.
            tile_extent=self.seqlen_k if self.input_layout in FIXED_LAYOUTS else 1,
            is_causal=self.is_causal,
            window_size=tuple(window_size),
            scale_softmax=scale_softmax,
        )
        self.scale_output = float(scale_output)
        self.target_compute_capability = target_compute_capability
        self.compute_capability: int | None = None

        if self.input_layout in FIXED_LAYOUTS:
            self.o_desc = make_fixed_output(
                tuple(sample_q.shape),
                sample_q.dtype,
                "o_tensor",
                layout=self.input_layout,
            )
            self.o_kernel_desc = self.o_desc
            self.lse_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(
                    (self.batch, self.num_query_heads, self.seqlen_q),
                    jnp.float32,
                ),
                "lse_tensor",
            )
            self.lse_kernel_desc = self.lse_desc
        else:
            self.o_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(tuple(sample_q.shape), sample_q.dtype),
                "o_tensor",
            )
            self.o_kernel_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, *sample_q.shape), sample_q.dtype),
                "o_kernel_tensor",
            )
            self.lse_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(
                    (self.total_q_tokens, self.num_query_heads), jnp.float32
                ),
                "lse_tensor",
                # The kernel writes sequence-contiguous LSE. TensorSpec asks
                # XLA for that physical order while preserving public (T, H).
                public_stride_order=(0, 1),
            )
            # Forward consumes packed LSE through its iterator, just as the
            # Torch wrapper passes the rank-2 (T, H) buffer directly.
            self.lse_kernel_desc = self.lse_desc

    def _init_fixed_shape(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_cum_seqlen_q: Any | None,
        sample_cum_seqlen_k: Any | None,
    ) -> None:
        if sample_cum_seqlen_q is not None or sample_cum_seqlen_k is not None:
            raise ValueError(
                "cum_seqlen_q and cum_seqlen_k must be omitted for fixed layout"
            )
        self.q_desc = describe_fixed_data(
            sample_q, "sample_q", layout=self.input_layout
        )
        self.k_desc = describe_fixed_data(
            sample_k, "sample_k", layout=self.input_layout
        )
        self.v_desc = describe_fixed_data(
            sample_v, "sample_v", layout=self.input_layout
        )
        (
            self.batch,
            self.num_query_heads,
            self.num_kv_heads,
            self.seqlen_q,
            self.seqlen_k,
            self.head_dim,
        ) = require_fixed_qkv(self.q_desc, self.k_desc, self.v_desc)
        self.total_q_tokens = self.batch * self.seqlen_q
        self.total_k_tokens = self.batch * self.seqlen_k
        self.max_s_q = self.seqlen_q
        self.max_s_k = self.seqlen_k
        self.cum_q_desc = None
        self.cum_k_desc = None
        self.q_kernel_desc = self.q_desc
        self.k_kernel_desc = self.k_desc
        self.v_kernel_desc = self.v_desc

    def _init_packed_shape(
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
                "cum_seqlen_q and cum_seqlen_k are both required for THD layout"
            )
        if max_s_q is None or max_s_k is None:
            raise ValueError("max_s_q and max_s_k are both required for THD layout")

        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
        self.v_desc = self._to_tensor_desc(sample_v, "sample_v")
        self.cum_q_desc = self._to_tensor_desc(
            sample_cum_seqlen_q, "sample_cum_seqlen_q"
        )
        self.cum_k_desc = self._to_tensor_desc(
            sample_cum_seqlen_k, "sample_cum_seqlen_k"
        )

        self.total_q_tokens, self.num_query_heads, self.head_dim = self.q_desc.shape
        self.total_k_tokens, self.num_kv_heads, k_head_dim = self.k_desc.shape
        v_total, v_heads, value_dim = self.v_desc.shape
        dimensions = (
            self.total_q_tokens,
            self.total_k_tokens,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError(f"SDPA dimensions must be positive, got {dimensions}")
        if (v_total, v_heads, value_dim) != (
            self.total_k_tokens,
            self.num_kv_heads,
            self.head_dim,
        ) or k_head_dim != self.head_dim:
            raise ValueError(
                "K and V must share packed token, head, and head-dim metadata"
            )
        if self.q_desc.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
            raise ValueError(
                f"SDPA requires float16 or bfloat16 inputs, got {self.q_desc.dtype}"
            )
        if (
            self.k_desc.cudnn_dtype != self.q_desc.cudnn_dtype
            or self.v_desc.cudnn_dtype != self.q_desc.cudnn_dtype
        ):
            raise ValueError("Q, K, and V must have the same dtype")
        if self.head_dim != 256:
            raise ValueError(f"head dimension must be 256, got {self.head_dim}")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"H_q ({self.num_query_heads}) must be divisible by H_kv ({self.num_kv_heads})"
            )
        if (
            self.cum_q_desc.ndim != 1
            or self.cum_q_desc.shape != self.cum_k_desc.shape
            or self.cum_q_desc.shape[0] < 2
        ):
            raise ValueError(
                "cum_seqlen_q and cum_seqlen_k must have the same shape (B + 1,) with B > 0"
            )
        if (
            self.cum_q_desc.cudnn_dtype != data_type.INT32
            or self.cum_k_desc.cudnn_dtype != data_type.INT32
        ):
            raise ValueError("cum_seqlen_q and cum_seqlen_k must have dtype int32")

        self.batch = self.cum_q_desc.shape[0] - 1
        self.seqlen_q = self.total_q_tokens
        self.seqlen_k = self.total_k_tokens
        self.max_s_q = int(max_s_q)
        self.max_s_k = int(max_s_k)
        if (
            self.max_s_q <= 0
            or self.max_s_k <= 0
            or self.max_s_q > self.total_q_tokens
            or self.max_s_k > self.total_k_tokens
        ):
            raise ValueError(
                "max_s_q and max_s_k must be positive and no larger than their packed token counts"
            )

        # Public THD remains row-major; adding the singleton batch dimension
        # yields the exact (1, T, H, D) ABI used by the Torch wrapper.
        self.q_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *sample_q.shape), sample_q.dtype), "q_tensor"
        )
        self.k_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *sample_k.shape), sample_k.dtype), "k_tensor"
        )
        self.v_kernel_desc = self._to_tensor_desc(
            jax.ShapeDtypeStruct((1, *sample_v.shape), sample_v.dtype), "v_tensor"
        )

    def check_support(self) -> bool:
        self.compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "SdpafwdSm100D256",
        )
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
        signature_mode = self.data_mode
        for value, desc in (
            (q_tensor, self.q_desc),
            (k_tensor, self.k_desc),
            (v_tensor, self.v_desc),
        ):
            self._check_tensor_signature(value, desc, mode=signature_mode)

        if self.input_layout in FIXED_LAYOUTS:
            if cum_seqlen_q_tensor is not None or cum_seqlen_k_tensor is not None:
                raise ValueError(
                    "cum_seqlen_q and cum_seqlen_k must be omitted for fixed layout"
                )
            inputs = (q_tensor, k_tensor, v_tensor)
            input_spec = (
                self._to_tensor_spec(self.q_kernel_desc, mode=self.data_mode),
                self._to_tensor_spec(self.k_kernel_desc, mode=self.data_mode),
                self._to_tensor_spec(self.v_kernel_desc, mode=self.data_mode),
            )
            output_spec = (
                self._to_tensor_spec(self.o_kernel_desc, mode=self.data_mode),
                self._to_tensor_spec(self.lse_kernel_desc),
            )
            launch = self._launch_kernel
        else:
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError(
                    "cum_seqlen_q and cum_seqlen_k are both required for THD layout"
                )
            self._check_tensor_signature(cum_seqlen_q_tensor, self.cum_q_desc)
            self._check_tensor_signature(cum_seqlen_k_tensor, self.cum_k_desc)
            inputs = (
                jnp.reshape(q_tensor, self.q_kernel_desc.shape),
                jnp.reshape(k_tensor, self.k_kernel_desc.shape),
                jnp.reshape(v_tensor, self.v_kernel_desc.shape),
                cum_seqlen_q_tensor,
                cum_seqlen_k_tensor,
            )
            input_spec = (
                self._to_tensor_spec(self.q_kernel_desc),
                self._to_tensor_spec(self.k_kernel_desc),
                self._to_tensor_spec(self.v_kernel_desc),
                self._to_tensor_spec(self.cum_q_desc),
                self._to_tensor_spec(self.cum_k_desc),
            )
            output_spec = (
                self._to_tensor_spec(self.o_kernel_desc),
                self._to_tensor_spec(self.lse_kernel_desc),
            )
            launch = self._launch_varlen_kernel

        output, lse = self._call_kernel(
            inputs,
            launch=launch,
            output_descs=(self.o_kernel_desc, self.lse_kernel_desc),
            input_spec=input_spec,
            output_spec=output_spec,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        if self.input_layout == _PACKED_LAYOUT:
            output = jnp.reshape(output, self.o_desc.shape)
        return TupleDict(o_tensor=output, lse_tensor=lse)

    def _launch_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        output: Any,
        lse: Any,
    ) -> None:
        self._invoke_kernel(stream, q, k, v, output, lse, None, None)

    def _launch_varlen_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        cum_seqlen_q: Any,
        cum_seqlen_k: Any,
        output: Any,
        lse: Any,
    ) -> None:
        self._invoke_kernel(
            stream,
            q,
            k,
            v,
            output,
            lse,
            cum_seqlen_q,
            cum_seqlen_k,
        )

    def _invoke_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        output: Any,
        lse: Any,
        cum_seqlen_q: Any | None,
        cum_seqlen_k: Any | None,
    ) -> None:
        from cutlass import Float32, Int32

        from ..fmha_utils import MaskEnum
        from .fmha_forward_sm100_d256 import BlackwellFusedMultiHeadAttentionForward

        mask_type = {
            "residual": MaskEnum.RESIDUAL_MASK,
            "window": MaskEnum.WINDOW_MASK_INFERENCE,
        }[self.mask_kind]
        kernel = BlackwellFusedMultiHeadAttentionForward(
            qk_acc_dtype=Float32,
            pv_acc_dtype=Float32,
            mma_tiler=(128, 128, 256),
            is_persistent=False,
            mask_type=mask_type,
        )
        problem_size = tuple(
            Int32(value)
            for value in (
                self.batch,
                self.max_s_q,
                self.max_s_k,
                self.num_query_heads,
                self.num_kv_heads,
                256,
            )
        )
        left = None if self.window_size_left < 0 else Int32(self.window_size_left)
        right = None if self.window_size_right < 0 else Int32(self.window_size_right)
        kernel(
            q,
            k,
            v,
            output,
            problem_size,
            cum_seqlen_q,
            cum_seqlen_k,
            lse,
            Float32(self.scale_softmax * math.log2(math.e)),
            Float32(self.scale_softmax),
            Float32(self.scale_output),
            left,
            right,
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "max_s_q",
        "max_s_k",
        "qk_acc_dtype",
        "pv_acc_dtype",
        "mma_tiler_mn",
        "is_causal",
        "window_size",
        "scale_softmax",
        "scale_output",
        "layout",
        "target_compute_capability",
    ),
)
def sdpa_fwd_wrapper_sm100_d256(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    cum_seqlen_q_tensor: Any | None = None,
    cum_seqlen_k_tensor: Any | None = None,
    max_s_q: int | None = None,
    max_s_k: int | None = None,
    qk_acc_dtype: Any = None,
    pv_acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: float | None = None,
    scale_output: float = 1.0,
    layout: str | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute fixed BHSD/BSHD or packed THD SDPA forward.

    Packed calls must provide both cumulative sequence-length arrays and static
    ``max_s_q``/``max_s_k`` bounds. Fixed outputs follow ``layout``; LSE remains
    ``(B, H, S)``.
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
    return SdpafwdSm100D256(
        *samples,
        sample_cum_seqlen_q,
        sample_cum_seqlen_k,
        max_s_q=max_s_q,
        max_s_k=max_s_k,
        qk_acc_dtype=qk_acc_dtype,
        pv_acc_dtype=pv_acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_causal=is_causal,
        window_size=window_size,
        scale_softmax=scale_softmax,
        scale_output=scale_output,
        layout=layout,
        target_compute_capability=target_compute_capability,
    )(
        q_tensor,
        k_tensor,
        v_tensor,
        cum_seqlen_q_tensor,
        cum_seqlen_k_tensor,
    )


__all__ = [
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SdpafwdSm100D256",
    "sdpa_fwd_wrapper_sm100_d256",
]
