# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 d=256 SDPA backward."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import jax.numpy as jnp
from cutlass.jax import jax_to_cutlass_dtype

from ..._jax.api_base import ApiBaseJax
from ..._jax.cutedsl import BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype
from ..jax_utils import (
    bhsd_tensor_spec,
    require_array,
    require_bhsd_qkv,
    resolve_sdpa_config,
)


class SdpaBwdResult(NamedTuple):
    """Functional outputs from :func:`sdpa_bwd_wrapper_sm100_d256`."""

    dq_tensor: Any
    dk_tensor: Any
    dv_tensor: Any


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_query_heads: int,
    num_kv_heads: int,
    element_dtype: Any,
    scale_softmax: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    mask_kind: str,
):
    from cutlass import Float32, Int32

    from ..fmha_utils import MaskEnum
    from .fmha_backward_sm100_2kernel import BlackwellFusedMultiHeadAttentionBackward

    mask_type = {
        "residual": MaskEnum.RESIDUAL_MASK,
        "window": MaskEnum.WINDOW_MASK_INFERENCE,
    }[mask_kind]
    kernel = BlackwellFusedMultiHeadAttentionBackward(
        element_dtype=element_dtype,
        acc_dtype=Float32,
        mma_tiler=(128, 128, 256),
        dkdv_mma_tiler=(128, 64, 256),
        varlen=False,
        is_causal=is_causal,
        mask_type=mask_type,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )

    def launch(stream, q, k, v, output, doutput, lse, dq, dk, dv, workspace):
        problem_shape = (
            Int32(seqlen_q),
            Int32(seqlen_k),
            Int32(256),
            (
                (Int32(num_query_heads // num_kv_heads), Int32(num_kv_heads)),
                Int32(batch),
            ),
        )
        kernel(
            problem_shape,
            q,
            k,
            v,
            output,
            dq,
            dk,
            dv,
            doutput,
            lse,
            None,
            None,
            Float32(scale_softmax),
            workspace,
            stream,
        )

    return launch


def _sdpa_bwd_impl(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    o_tensor: Any,
    do_tensor: Any,
    lse_tensor: Any,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    dkdv_mma_tiler_mn: tuple[int, int] = (128, 64),
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: float | None = None,
    *,
    _validate_only: bool = False,
) -> SdpaBwdResult:
    """Compute fixed-shape BHSD SDPA gradients with the SM100 d=256 kernel.

    Q, K, V, O, and dO use logical JAX shapes ``(B, H, S, 256)``.
    ``lse_tensor`` uses shape ``(B, H_q, S_q)`` and dtype ``float32``.
    The returned dQ, dK, and dV arrays match Q, K, and V respectively.

    The multi-kernel implementation uses a hidden, zero-initialized workspace
    owned by XLA. Variable-length THD inputs are not part of this API.
    """

    (
        batch,
        num_query_heads,
        num_kv_heads,
        seqlen_q,
        seqlen_k,
        head_dim,
        dtype,
    ) = require_bhsd_qkv(q_tensor, k_tensor, v_tensor)
    require_array("o_tensor", o_tensor, tuple(q_tensor.shape), dtype)
    require_array("do_tensor", do_tensor, tuple(q_tensor.shape), dtype)
    require_array(
        "lse_tensor",
        lse_tensor,
        (batch, num_query_heads, seqlen_q),
        jnp.float32,
    )

    acc_dtype = require_dtype("acc_dtype", acc_dtype, (jnp.float32,), default=jnp.float32)
    if mma_tiler_mn != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
    if dkdv_mma_tiler_mn != (128, 64):
        raise ValueError(f"dkdv_mma_tiler_mn must be (128, 64), got {dkdv_mma_tiler_mn}")

    scale_softmax, window_size_left, window_size_right, mask_kind = resolve_sdpa_config(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_extent=seqlen_q,
        is_causal=bool(is_causal),
        window_size=window_size,
        scale_softmax=scale_softmax,
    )
    if _validate_only:
        return None

    from cutlass import Float32

    from .fmha_backward_sm100_2kernel import BlackwellFusedMultiHeadAttentionBackward

    workspace_shape = BlackwellFusedMultiHeadAttentionBackward.get_workspace_size(
        seqlen_q,
        head_dim,
        num_query_heads,
        batch,
        Float32,
    )
    bhsd_spec = bhsd_tensor_spec()
    dq_tensor, dk_tensor, dv_tensor = call_cutedsl(
        _make_launcher(
            batch=batch,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            element_dtype=jax_to_cutlass_dtype(dtype),
            scale_softmax=scale_softmax,
            is_causal=bool(is_causal),
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            mask_kind=mask_kind,
        ),
        (q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, lse_tensor),
        outputs=(
            BufferSpec(
                "dq_tensor",
                tuple(q_tensor.shape),
                dtype,
                tensor_spec=bhsd_spec,
            ),
            BufferSpec(
                "dk_tensor",
                tuple(k_tensor.shape),
                dtype,
                tensor_spec=bhsd_spec,
            ),
            BufferSpec(
                "dv_tensor",
                tuple(v_tensor.shape),
                dtype,
                tensor_spec=bhsd_spec,
            ),
        ),
        workspaces=(
            BufferSpec(
                "workspace",
                workspace_shape,
                jnp.uint8,
                fill_value=0,
            ),
        ),
        input_specs=(bhsd_spec, bhsd_spec, bhsd_spec, bhsd_spec, bhsd_spec, None),
        use_static_tensors=True,
    )
    return SdpaBwdResult(
        dq_tensor=dq_tensor,
        dk_tensor=dk_tensor,
        dv_tensor=dv_tensor,
    )


class SdpabwdSm100D256(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 d=256 SDPA backward."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_o: Any,
        sample_do: Any,
        sample_lse: Any,
        acc_dtype: Any = None,
        mma_tiler_mn: tuple[int, int] = (128, 128),
        dkdv_mma_tiler_mn: tuple[int, int] = (128, 64),
        is_causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        scale_softmax: float | None = None,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self.make_tensor_desc(sample_k, name="sample_k")
        self.v_desc = self.make_tensor_desc(sample_v, name="sample_v")
        self.o_desc = self.make_tensor_desc(sample_o, name="sample_o")
        self.do_desc = self.make_tensor_desc(sample_do, name="sample_do")
        self.lse_desc = self.make_tensor_desc(sample_lse, name="sample_lse")
        self.acc_dtype = self.as_optional_dtype(acc_dtype)
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.dkdv_mma_tiler_mn = tuple(dkdv_mma_tiler_mn)
        self.is_causal = is_causal
        self.window_size = tuple(window_size)
        self.scale_softmax = scale_softmax

    def _check_support(self) -> bool:
        _sdpa_bwd_impl(
            self.q_desc,
            self.k_desc,
            self.v_desc,
            self.o_desc,
            self.do_desc,
            self.lse_desc,
            self.acc_dtype,
            self.mma_tiler_mn,
            self.dkdv_mma_tiler_mn,
            self.is_causal,
            self.window_size,
            self.scale_softmax,
            _validate_only=True,
        )
        return True

    def __call__(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        o_tensor: Any,
        do_tensor: Any,
        lse_tensor: Any,
    ) -> SdpaBwdResult:
        return super().__call__(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, lse_tensor)

    def _call_impl(
        self,
        q_tensor: Any,
        k_tensor: Any,
        v_tensor: Any,
        o_tensor: Any,
        do_tensor: Any,
        lse_tensor: Any,
    ) -> SdpaBwdResult:
        for value, expected, name in (
            (q_tensor, self.q_desc, "Q"),
            (k_tensor, self.k_desc, "K"),
            (v_tensor, self.v_desc, "V"),
            (o_tensor, self.o_desc, "O"),
            (do_tensor, self.do_desc, "dO"),
            (lse_tensor, self.lse_desc, "LSE"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        return _sdpa_bwd_impl(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            do_tensor,
            lse_tensor,
            self.acc_dtype,
            self.mma_tiler_mn,
            self.dkdv_mma_tiler_mn,
            self.is_causal,
            self.window_size,
            self.scale_softmax,
        )


def sdpa_bwd_wrapper_sm100_d256(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    o_tensor: Any,
    do_tensor: Any,
    lse_tensor: Any,
    acc_dtype: Any = None,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    dkdv_mma_tiler_mn: tuple[int, int] = (128, 64),
    is_causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    scale_softmax: float | None = None,
) -> SdpaBwdResult:
    """Compute fixed-shape BHSD SDPA gradients with the SM100 d=256 kernel."""

    return SdpabwdSm100D256(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        lse_tensor,
        acc_dtype=acc_dtype,
        mma_tiler_mn=mma_tiler_mn,
        dkdv_mma_tiler_mn=dkdv_mma_tiler_mn,
        is_causal=is_causal,
        window_size=window_size,
        scale_softmax=scale_softmax,
    )(q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, lse_tensor)


__all__ = ["SdpaBwdResult", "SdpabwdSm100D256", "sdpa_bwd_wrapper_sm100_d256"]
