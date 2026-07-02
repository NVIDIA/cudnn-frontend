# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 d=256 SDPA forward."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, NamedTuple

import jax.numpy as jnp

from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype
from ..jax_utils import bhsd_tensor_spec, require_bhsd_qkv, resolve_sdpa_config


class SdpaFwdResult(NamedTuple):
    """Functional outputs from :func:`sdpa_fwd_wrapper_sm100_d256`."""

    o_tensor: Any
    lse_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
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

    def launch(stream, q, k, v, output, lse):
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

    return launch


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
) -> SdpaFwdResult:
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

    require_dtype("qk_acc_dtype", qk_acc_dtype, (jnp.float32,), default=jnp.float32)
    require_dtype("pv_acc_dtype", pv_acc_dtype, (jnp.float32,), default=jnp.float32)
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
    bhsd_spec = bhsd_tensor_spec()

    o_tensor, lse_tensor = call_cutedsl(
        _make_launcher(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            scale_softmax=scale_softmax,
            scale_output=scale_output,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mask_kind=mask_kind,
        ),
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
        use_static_tensors=True,
    )
    return SdpaFwdResult(o_tensor=o_tensor, lse_tensor=lse_tensor)


__all__ = ["SdpaFwdResult", "sdpa_fwd_wrapper_sm100_d256"]
