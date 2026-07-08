# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 d=256 SDPA forward."""

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
from ..jax_utils import bhsd_tensor_spec, require_bhsd_qkv, resolve_sdpa_config


def _launch(
    stream,
    q,
    k,
    v,
    output,
    lse,
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_query_heads: int,
    num_kv_heads: int,
    scale_softmax: float,
    scale_output: float,
    window_size_left: int,
    window_size_right: int,
    mask_kind: str,
):
    from cutlass import Float32, Int32

    from ..fmha_utils import MaskEnum
    from .fmha_forward_sm100_d256 import BlackwellFusedMultiHeadAttentionForward

    mask_type = {
        "residual": MaskEnum.RESIDUAL_MASK,
        "window": MaskEnum.WINDOW_MASK_INFERENCE,
    }[mask_kind]
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
            batch,
            seqlen_q,
            seqlen_k,
            num_query_heads,
            num_kv_heads,
            256,
        )
    )
    left = None if window_size_left < 0 else Int32(window_size_left)
    right = None if window_size_right < 0 else Int32(window_size_right)
    kernel(
        q,
        k,
        v,
        output,
        problem_size,
        None,
        None,
        lse,
        Float32(scale_softmax * math.log2(math.e)),
        Float32(scale_softmax),
        Float32(scale_output),
        left,
        right,
        stream,
    )


def _sdpa_fwd_impl(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    qk_acc_dtype: Any = None,
    pv_acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: float | None = None,
    scale_output: float = 1.0,
    *,
    _validate_only: bool = False,
) -> TupleDict:
    """Compute fixed-shape BHSD SDPA forward with the SM100 d=256 kernel.

    ``q_tensor``, ``k_tensor``, and ``v_tensor`` use logical JAX shapes
    ``(B, H, S, 256)`` and must share a ``float16`` or ``bfloat16`` dtype.
    The result contains ``o_tensor`` with the same shape and dtype as Q and
    ``lse_tensor`` with shape ``(B, H_q, S_q)`` and dtype ``float32``.

    Variable-length THD inputs are not part of this API. Configuration values
    must be static while tracing with :func:`jax.jit`.
    """

    (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        _head_dim,
        dtype,
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
    if mma_tiler_mn != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")

    scale_softmax, window_size_left, window_size_right, mask_kind = resolve_sdpa_config(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_extent=seqlen_k,
        is_causal=bool(is_causal),
        window_size=window_size,
        scale_softmax=scale_softmax,
    )
    scale_output = float(scale_output)
    if _validate_only:
        return None

    bhsd_spec = bhsd_tensor_spec()

    o_tensor, lse_tensor = call_cutedsl(
        _launch,
        (q_tensor, k_tensor, v_tensor),
        outputs=(
            BufferSpec(
                "o_tensor",
                tuple(q_tensor.shape),
                dtype,
                tensor_spec=bhsd_spec,
            ),
            BufferSpec(
                "lse_tensor",
                (batch, num_query_heads, seqlen_q),
                jnp.float32,
            ),
        ),
        input_specs=(bhsd_spec, bhsd_spec, bhsd_spec),
        static_args={
            "batch": batch,
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "scale_softmax": scale_softmax,
            "scale_output": scale_output,
            "window_size_left": window_size_left,
            "window_size_right": window_size_right,
            "mask_kind": mask_kind,
        },
    )
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


class SdpafwdSm100D256(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 d=256 SDPA forward."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        qk_acc_dtype: Any = None,
        pv_acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        is_causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        scale_softmax: float | None = None,
        scale_output: float = 1.0,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self.make_tensor_desc(sample_v, name="sample_v")
        self.qk_acc_dtype = self.as_optional_dtype(qk_acc_dtype)
        self.pv_acc_dtype = self.as_optional_dtype(pv_acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.is_causal = is_causal
        self.window_size = tuple(window_size)
        self.scale_softmax = scale_softmax
        self.scale_output = scale_output

    def _check_support(self) -> None:
        _sdpa_fwd_impl(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.qk_acc_dtype,
            self.pv_acc_dtype,
            self.mma_tiler_mn,
            self.is_causal,
            self.window_size,
            self.scale_softmax,
            self.scale_output,
            _validate_only=True,
        )

    def __call__(self, q_tensor: Any, k_tensor: Any, v_tensor: Any) -> TupleDict:
        return super().__call__(q_tensor, k_tensor, v_tensor)

    def _call_impl(self, q_tensor: Any, k_tensor: Any, v_tensor: Any) -> TupleDict:
        self.check_tensor_signature(q_tensor, self.q_desc, name="Q")
        self.check_tensor_signature(k_tensor, self.k_desc, name="K")
        self.check_tensor_signature(v_tensor, self.v_desc, name="V")
        return _sdpa_fwd_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            self.qk_acc_dtype,
            self.pv_acc_dtype,
            self.mma_tiler_mn,
            self.is_causal,
            self.window_size,
            self.scale_softmax,
            self.scale_output,
        )


def sdpa_fwd_wrapper_sm100_d256(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    qk_acc_dtype: Any = None,
    pv_acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: float | None = None,
    scale_output: float = 1.0,
) -> TupleDict:
    """Compute fixed-shape BHSD SDPA forward with the SM100 d=256 kernel."""

    return SdpafwdSm100D256(
        q_tensor,
        k_tensor,
        v_tensor,
        qk_acc_dtype=qk_acc_dtype,
        pv_acc_dtype=pv_acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        is_causal=is_causal,
        window_size=window_size,
        scale_softmax=scale_softmax,
        scale_output=scale_output,
    )(q_tensor, k_tensor, v_tensor)


__all__ = ["SdpafwdSm100D256", "sdpa_fwd_wrapper_sm100_d256"]
