# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed BHSD/BSHD and packed THD NSA compression attention."""

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
    BHS_TO_BSH_MODE,
    FIXED_LAYOUTS,
    describe_bhs_as_bsh,
    describe_fixed_data,
    fixed_data_mode,
    make_fixed_output,
    normalize_attention_layout,
    normalize_supported_dtype,
    require_fixed_qkv,
)

SUPPORTED_COMPUTE_CAPABILITIES = (100, 103, 107)


class CompressionAttention(JaxApiBase):
    """JAX callable specialized from fixed BHSD/BSHD or packed THD metadata."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_cum_seqlen_q: Any | None = None,
        sample_cum_seqlen_k: Any | None = None,
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
        max_s_q: int | None = None,
        max_s_k: int | None = None,
        layout: str | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        input_dtypes = (
            data_type.HALF,
            data_type.BFLOAT16,
            data_type.FP8_E4M3,
        )
        ranks = tuple(len(sample.shape) for sample in (sample_q, sample_k, sample_v))
        if len(set(ranks)) != 1:
            raise ValueError(f"Q, K, and V must all use the same rank, got {ranks}")
        self.input_layout = normalize_attention_layout(layout, ranks[0])
        if self.input_layout in FIXED_LAYOUTS:
            self.data_mode = fixed_data_mode(self.input_layout, kernel_axes="BSHD")
            if sample_cum_seqlen_q is not None or sample_cum_seqlen_k is not None:
                raise ValueError(
                    "cumulative sequence lengths are only valid for packed THD inputs"
                )
            if max_s_q is not None or max_s_k is not None:
                raise ValueError("max_s_q and max_s_k are only valid for packed THD")
            self.q_desc = describe_fixed_data(
                sample_q,
                "sample_q",
                layout=self.input_layout,
                kernel_axes="BSHD",
            )
            self.k_desc = describe_fixed_data(
                sample_k,
                "sample_k",
                layout=self.input_layout,
                kernel_axes="BSHD",
            )
            self.v_desc = describe_fixed_data(
                sample_v,
                "sample_v",
                layout=self.input_layout,
                kernel_axes="BSHD",
            )
            (
                self.batch,
                self.num_query_heads,
                self.num_kv_heads,
                self.seqlen_q,
                self.seqlen_k,
                self.head_dim,
            ) = require_fixed_qkv(
                self.q_desc,
                self.k_desc,
                self.v_desc,
                operation_name="CompressionAttention",
                kernel_axes="BSHD",
                input_dtypes=input_dtypes,
            )
            self.cum_q_desc = self.cum_k_desc = None
            self.lse_extent = self.seqlen_q
            self.q_kernel_desc = self.q_desc
            self.k_kernel_desc = self.k_desc
            self.v_kernel_desc = self.v_desc
        elif self.input_layout == "THD":
            self.data_mode = None
            if len(sample_k.shape) != 3 or len(sample_v.shape) != 3:
                raise ValueError("Q, K, and V must all use the same rank")
            if sample_cum_seqlen_q is None or sample_cum_seqlen_k is None:
                raise ValueError(
                    "packed THD inputs require cumulative Q and K sequence lengths"
                )
            if max_s_q is None or max_s_k is None:
                raise ValueError("packed THD inputs require max_s_q and max_s_k")
            self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
            self.k_desc = self._to_tensor_desc(sample_k, "sample_k")
            self.v_desc = self._to_tensor_desc(sample_v, "sample_v")
            total_q, self.num_query_heads, self.head_dim = self.q_desc.shape
            total_k, self.num_kv_heads, k_head_dim = self.k_desc.shape
            v_total, value_heads, value_dim = self.v_desc.shape
            if min(total_q, total_k, self.num_query_heads, self.num_kv_heads) <= 0:
                raise ValueError("packed THD dimensions must be positive")
            if (v_total, value_heads, value_dim) != (
                total_k,
                self.num_kv_heads,
                self.head_dim,
            ) or k_head_dim != self.head_dim:
                raise ValueError("packed K and V shapes must match Q/K head metadata")
            if self.q_desc.cudnn_dtype not in input_dtypes:
                raise ValueError(f"unsupported Q dtype {self.q_desc.dtype}")
            if (
                self.k_desc.cudnn_dtype != self.q_desc.cudnn_dtype
                or self.v_desc.cudnn_dtype != self.q_desc.cudnn_dtype
            ):
                raise ValueError("Q, K, and V must have the same dtype")
            if self.head_dim not in (32, 64, 128):
                raise ValueError("head dimension must be 32, 64, or 128")
            if self.num_query_heads % self.num_kv_heads:
                raise ValueError("H_q must be divisible by H_kv")
            self.cum_q_desc = self._to_tensor_desc(
                sample_cum_seqlen_q, "sample_cum_seqlen_q"
            )
            self.cum_k_desc = self._to_tensor_desc(
                sample_cum_seqlen_k, "sample_cum_seqlen_k"
            )
            if (
                self.cum_q_desc.ndim != 1
                or self.cum_q_desc.shape != self.cum_k_desc.shape
                or self.cum_q_desc.shape[0] < 2
            ):
                raise ValueError(
                    "cumulative Q and K sequence lengths must have matching "
                    "(B + 1,) shapes"
                )
            if (
                self.cum_q_desc.cudnn_dtype not in (data_type.INT32, data_type.INT64)
                or self.cum_k_desc.cudnn_dtype != self.cum_q_desc.cudnn_dtype
            ):
                raise ValueError(
                    "cumulative sequence lengths must share int32 or int64 dtype"
                )
            self.batch = self.cum_q_desc.shape[0] - 1
            self.seqlen_q = int(max_s_q)
            self.seqlen_k = int(max_s_k)
            self.lse_extent = total_q
            if (
                self.seqlen_q <= 0
                or self.seqlen_k <= 0
                or self.seqlen_q > total_q
                or self.seqlen_k > total_k
            ):
                raise ValueError(
                    "max_s_q and max_s_k must be positive and no larger than "
                    "their packed token counts"
                )
            self.q_kernel_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, *sample_q.shape), sample_q.dtype),
                "q_tensor",
                public_stride_order=(3, 2, 0, 1),
            )
            self.k_kernel_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, *sample_k.shape), sample_k.dtype),
                "k_tensor",
                public_stride_order=(3, 2, 0, 1),
            )
            self.v_kernel_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, *sample_v.shape), sample_v.dtype),
                "v_tensor",
                public_stride_order=(3, 2, 0, 1),
            )
        if self.seqlen_q < self.seqlen_k or self.seqlen_q % self.seqlen_k:
            raise ValueError(
                "Compression attention requires S_q to be an integer multiple of "
                f"S_k, got S_q={self.seqlen_q} and S_k={self.seqlen_k}"
            )
        if tuple(mma_tiler_mn) != (128, 128):
            raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")

        self.enable_lse = bool(enable_lse)
        self.is_persistent = bool(is_persistent)
        self.output_dtype = normalize_supported_dtype(
            o_dtype,
            sample_q.dtype,
            "o_dtype",
            (jnp.float16, jnp.bfloat16, jnp.float8_e4m3fn),
        )
        normalize_supported_dtype(
            qk_acc_dtype, jnp.float32, "qk_acc_dtype", (jnp.float32,)
        )
        normalize_supported_dtype(
            pv_acc_dtype, jnp.float32, "pv_acc_dtype", (jnp.float32,)
        )
        base_scale = (
            1.0 / math.sqrt(self.head_dim)
            if scale_softmax is None
            else float(scale_softmax)
        )
        self.scale_softmax = float(scale_q) * float(scale_k) * base_scale
        self.scale_output = float(scale_v) * float(inv_scale_o)
        self.target_compute_capability = target_compute_capability
        self.compute_capability: int | None = None
        self.persistent_sm_count: int | None = None

        if self.input_layout in FIXED_LAYOUTS:
            self.o_desc = make_fixed_output(
                tuple(sample_q.shape),
                self.output_dtype,
                "o_tensor",
                layout=self.input_layout,
                kernel_axes="BSHD",
            )
            self.o_kernel_desc = self.o_desc
            self.lse_desc = (
                describe_bhs_as_bsh(
                    jax.ShapeDtypeStruct(
                        (self.batch, self.num_query_heads, self.seqlen_q),
                        jnp.float32,
                    ),
                    "lse_tensor",
                )
                if self.enable_lse
                else None
            )
            self.lse_kernel_desc = self.lse_desc
        else:
            output_shape = (
                self.q_desc.shape[0],
                self.num_query_heads,
                self.head_dim,
            )
            self.o_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(output_shape, self.output_dtype), "o_tensor"
            )
            self.o_kernel_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((1, *output_shape), self.output_dtype),
                "o_kernel_tensor",
                public_stride_order=(3, 2, 0, 1),
            )
            self.lse_desc = (
                self._to_tensor_desc(
                    jax.ShapeDtypeStruct(output_shape[:2], jnp.float32),
                    "lse_tensor",
                )
                if self.enable_lse
                else None
            )
            self.lse_kernel_desc = (
                self._to_tensor_desc(
                    jax.ShapeDtypeStruct((1, *output_shape[:2]), jnp.float32),
                    "lse_kernel_tensor",
                )
                if self.enable_lse
                else None
            )

    def check_support(self) -> bool:
        self.compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            SUPPORTED_COMPUTE_CAPABILITIES,
            "CompressionAttention",
        )
        if self.is_persistent:
            self.persistent_sm_count = self._get_device_multiprocessor_count()
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
                    "cumulative sequence lengths must be omitted for fixed inputs"
                )
            inputs = (q_tensor, k_tensor, v_tensor)
            input_specs = (
                self._to_tensor_spec(self.q_kernel_desc, mode=self.data_mode),
                self._to_tensor_spec(self.k_kernel_desc, mode=self.data_mode),
                self._to_tensor_spec(self.v_kernel_desc, mode=self.data_mode),
            )
            output_specs = (
                self._to_tensor_spec(self.o_kernel_desc, mode=self.data_mode),
            )
            if self.lse_kernel_desc is not None:
                output_specs += (
                    self._to_tensor_spec(self.lse_kernel_desc, mode=BHS_TO_BSH_MODE),
                )
            launch = self._launch_kernel
        else:
            if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
                raise ValueError(
                    "packed THD inputs require cumulative Q and K sequence lengths"
                )
            self._check_tensor_signature(cum_seqlen_q_tensor, self.cum_q_desc)
            self._check_tensor_signature(cum_seqlen_k_tensor, self.cum_k_desc)
            q_tensor = jnp.reshape(q_tensor, self.q_kernel_desc.shape)
            k_tensor = jnp.reshape(k_tensor, self.k_kernel_desc.shape)
            v_tensor = jnp.reshape(v_tensor, self.v_kernel_desc.shape)
            inputs = (
                q_tensor,
                k_tensor,
                v_tensor,
                cum_seqlen_q_tensor,
                cum_seqlen_k_tensor,
            )
            input_specs = (
                self._to_tensor_spec(self.q_kernel_desc),
                self._to_tensor_spec(self.k_kernel_desc),
                self._to_tensor_spec(self.v_kernel_desc),
                self._to_tensor_spec(self.cum_q_desc),
                self._to_tensor_spec(self.cum_k_desc),
            )
            output_specs = (self._to_tensor_spec(self.o_kernel_desc),)
            if self.lse_kernel_desc is not None:
                output_specs += (self._to_tensor_spec(self.lse_kernel_desc),)
            launch = self._launch_varlen_kernel

        results = self._call_kernel(
            inputs,
            launch=launch,
            output_descs=(self.o_kernel_desc,)
            if self.lse_kernel_desc is None
            else (self.o_kernel_desc, self.lse_kernel_desc),
            input_spec=input_specs,
            output_spec=output_specs,
            compile_options=compile_options_for_target(self.compute_capability),
        )
        output = results[0]
        lse = results[1] if self.enable_lse else None
        if self.input_layout == "THD":
            output = jnp.reshape(output, self.o_desc.shape)
            if lse is not None:
                lse = jnp.reshape(lse, self.lse_desc.shape)
        return TupleDict(
            o_tensor=output,
            lse_tensor=lse,
        )

    def _launch_kernel(
        self, stream: Any, q: Any, k: Any, v: Any, output: Any, *optional: Any
    ) -> None:
        lse = optional[0] if optional else None
        self._run_kernel(stream, q, k, v, output, lse, None, None)

    def _launch_varlen_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        cum_seqlen_q: Any,
        cum_seqlen_k: Any,
        output: Any,
        *optional: Any,
    ) -> None:
        lse = optional[0] if optional else None
        self._run_kernel(
            stream,
            q,
            k,
            v,
            output,
            lse,
            cum_seqlen_q,
            cum_seqlen_k,
        )

    def _run_kernel(
        self,
        stream: Any,
        q: Any,
        k: Any,
        v: Any,
        output: Any,
        lse: Any,
        cum_seqlen_q: Any,
        cum_seqlen_k: Any,
    ) -> None:
        from cutlass import Float32, Int32

        from .fmha import BlackwellFusedMultiHeadAttentionForward
        from .fmha_helpers import MaskType

        kernel = BlackwellFusedMultiHeadAttentionForward(
            qk_acc_dtype=Float32,
            pv_acc_dtype=Float32,
            mma_tiler=(128, 128, self.head_dim),
            is_persistent=self.is_persistent,
            mask_type=MaskType.COMPRESSED_CAUSAL_MASK,
            persistent_sm_count=self.persistent_sm_count,
        )
        problem_size = tuple(
            Int32(value)
            for value in (
                self.batch,
                self.seqlen_q,
                self.lse_extent,
                self.seqlen_k,
                self.num_query_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        )
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
            None,
            Int32(0),
            stream,
        )


@partial(
    jax.jit,
    static_argnames=(
        "enable_lse",
        "o_dtype",
        "qk_acc_dtype",
        "pv_acc_dtype",
        "mma_tiler_mn",
        "is_persistent",
        "scale_q",
        "scale_k",
        "scale_v",
        "inv_scale_o",
        "scale_softmax",
        "max_s_q",
        "max_s_k",
        "layout",
        "target_compute_capability",
    ),
)
def compression_attention_wrapper(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    cum_seqlen_q_tensor: Any | None = None,
    cum_seqlen_k_tensor: Any | None = None,
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
    max_s_q: int | None = None,
    max_s_k: int | None = None,
    layout: str | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute fixed BHSD/BSHD or packed THD compression attention.

    Packed THD calls must provide both cumulative sequence-length arrays and
    static ``max_s_q``/``max_s_k`` bounds because JAX traces array values.
    Fixed outputs follow ``layout``; LSE remains ``(B, H, S)``.
    """

    qkv_samples = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype)
        for value in (q_tensor, k_tensor, v_tensor)
    )
    cum_q_sample = (
        None
        if cum_seqlen_q_tensor is None
        else jax.ShapeDtypeStruct(cum_seqlen_q_tensor.shape, cum_seqlen_q_tensor.dtype)
    )
    cum_k_sample = (
        None
        if cum_seqlen_k_tensor is None
        else jax.ShapeDtypeStruct(cum_seqlen_k_tensor.shape, cum_seqlen_k_tensor.dtype)
    )
    return CompressionAttention(
        *qkv_samples,
        cum_q_sample,
        cum_k_sample,
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
        max_s_q=max_s_q,
        max_s_k=max_s_k,
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
    "CompressionAttention",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "compression_attention_wrapper",
]
