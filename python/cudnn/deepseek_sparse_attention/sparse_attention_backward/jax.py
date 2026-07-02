# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for fixed-shape SM100 DSA sparse-attention backward."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, NamedTuple, Optional

import jax.numpy as jnp
from cutlass.jax import TensorSpec

from ..._jax.api_base import ApiBaseJax, BufferSpec, call_cutedsl
from ..._jax.validation import require_dtype


class SparseAttentionBackwardResult(NamedTuple):
    """Functional gradients from :func:`sparse_attention_backward_wrapper`."""

    dq: Any
    dkv: Any
    d_sink: Any


def require_array(
    name: str,
    value: Any,
    *,
    rank: int,
    shape: Optional[tuple[int, ...]] = None,
    dtype: Any,
) -> None:
    """Require an array with the expected rank, optional shape, and dtype."""

    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be a JAX array with shape and dtype metadata")
    if len(value.shape) != rank:
        raise ValueError(f"{name} must have rank {rank}, got shape {value.shape}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    require_dtype(f"{name}.dtype", value, (dtype,))


@lru_cache(maxsize=None)
def _make_launcher(
    *,
    total_seqlen_q: int,
    total_seqlen_kv: int,
    num_heads: int,
    head_dim: int,
    block_tile: int,
    softmax_scale: float,
    has_topk_length: bool,
):
    from cutlass import Float32, Int32

    from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

    kernel = FlashAttentionDSABackwardSm100(
        head_dim=head_dim,
        head_dim_v=head_dim,
        block_tile=block_tile,
    )

    if has_topk_length:

        def launch(
            stream,
            q,
            kv,
            output,
            doutput,
            lse,
            attn_sink,
            topk_idxs,
            topk_length,
            dq,
            dkv,
            d_sink,
            workspace_lse_odo,
            workspace_dkv,
        ):
            problem_shape = (
                Int32(total_seqlen_q),
                Int32(total_seqlen_kv),
                Int32(head_dim),
                (Int32(num_heads), Int32(1)),
            )
            kernel(
                problem_shape,
                q,
                kv,
                output,
                doutput,
                lse,
                attn_sink,
                topk_idxs,
                topk_length,
                dq,
                dkv,
                d_sink,
                workspace_lse_odo,
                workspace_dkv,
                Float32(softmax_scale),
                stream,
            )

    else:

        def launch(
            stream,
            q,
            kv,
            output,
            doutput,
            lse,
            attn_sink,
            topk_idxs,
            dq,
            dkv,
            d_sink,
            workspace_lse_odo,
            workspace_dkv,
        ):
            problem_shape = (
                Int32(total_seqlen_q),
                Int32(total_seqlen_kv),
                Int32(head_dim),
                (Int32(num_heads), Int32(1)),
            )
            kernel(
                problem_shape,
                q,
                kv,
                output,
                doutput,
                lse,
                attn_sink,
                topk_idxs,
                None,
                dq,
                dkv,
                d_sink,
                workspace_lse_odo,
                workspace_dkv,
                Float32(softmax_scale),
                stream,
            )

    return launch


def _sparse_attention_backward_impl(
    q: Any,
    kv: Any,
    out: Any,
    dout: Any,
    lse: Any,
    attn_sink: Any,
    topk_idxs: Any,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[Any] = None,
    block_tile: int = 64,
    _validate_only: bool = False,
) -> SparseAttentionBackwardResult:
    """Compute fixed-shape DSA sparse-attention gradients on SM100.

    Inputs follow the Torch wrapper's flat MQA contract: Q, O, and dO have
    shape ``(S_q, H, 512)``, KV has shape ``(S_kv, 512)``, LSE has shape
    ``(S_q, H)``, the attention sink has shape ``(H,)``, and global top-K
    indices have shape ``(S_q, topk)``. All floating inputs except LSE and the
    sink use ``bfloat16``; LSE and the sink use ``float32``.

    ``topk_length``, when present, is an ``int32`` vector of shape ``(S_q,)``.
    Runtime indices must be ``-1`` or in ``[0, S_kv)``. Runtime lengths must be
    in ``[1, topk]``; those value constraints are trusted while tracing.

    This is a standalone functional backward operation, not a custom VJP.
    SM90 and packed variable-length/batched layouts are not supported.
    Configuration values must be static under :func:`jax.jit`.
    """

    require_array("q", q, rank=3, dtype=jnp.bfloat16)
    require_array("kv", kv, rank=2, dtype=jnp.bfloat16)

    total_seqlen_q, num_heads, head_dim = tuple(q.shape)
    total_seqlen_kv, kv_head_dim = tuple(kv.shape)
    dimensions = {
        "S_q": total_seqlen_q,
        "S_kv": total_seqlen_kv,
        "H": num_heads,
        "D": head_dim,
    }
    nonpositive = [f"{name}={value}" for name, value in dimensions.items() if value <= 0]
    if nonpositive:
        raise ValueError("Sparse-attention dimensions must be positive, got " + ", ".join(nonpositive))
    if head_dim != 512 or kv_head_dim != 512:
        raise ValueError("The JAX SM100 sparse-attention backward API requires Q and KV head " f"dimensions of 512, got {head_dim} and {kv_head_dim}")
    if num_heads % 64:
        raise ValueError(f"H must be divisible by 64, got {num_heads}")
    if block_tile != 64:
        raise ValueError(f"block_tile must be 64, got {block_tile}")

    q_shape = (total_seqlen_q, num_heads, head_dim)
    require_array("out", out, rank=3, shape=q_shape, dtype=jnp.bfloat16)
    require_array("dout", dout, rank=3, shape=q_shape, dtype=jnp.bfloat16)
    require_array(
        "lse",
        lse,
        rank=2,
        shape=(total_seqlen_q, num_heads),
        dtype=jnp.float32,
    )
    require_array(
        "attn_sink",
        attn_sink,
        rank=1,
        shape=(num_heads,),
        dtype=jnp.float32,
    )
    require_array("topk_idxs", topk_idxs, rank=2, dtype=jnp.int32)
    topk_shape = tuple(topk_idxs.shape)
    if topk_shape[0] != total_seqlen_q:
        raise ValueError(f"topk_idxs leading dimension must be S_q ({total_seqlen_q}), got {topk_shape[0]}")
    if topk_shape[1] <= 0:
        raise ValueError(f"topk_idxs must contain at least one index per row, got {topk_shape}")

    if topk_length is not None:
        require_array(
            "topk_length",
            topk_length,
            rank=1,
            shape=(total_seqlen_q,),
            dtype=jnp.int32,
        )

    resolved_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None or softmax_scale == 0.0 else float(softmax_scale)
    if _validate_only:
        return None

    from cutlass import Float32

    from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

    workspace_lse_odo_shape = FlashAttentionDSABackwardSm100.get_workspace_size_lse_odo(
        total_seqlen_q,
        head_dim,
        num_heads,
        1,
        Float32,
    )
    workspace_dkv_shape = FlashAttentionDSABackwardSm100.get_workspace_size_dkv(
        total_seqlen_kv,
        head_dim,
        1,
        Float32,
    )
    tensor_spec = TensorSpec(divisibility=head_dim)

    inputs = (q, kv, out, dout, lse, attn_sink, topk_idxs)
    input_specs = (
        tensor_spec,
        tensor_spec,
        tensor_spec,
        tensor_spec,
        None,
        None,
        None,
    )
    if topk_length is not None:
        inputs += (topk_length,)
        input_specs += (None,)

    dq, dkv, d_sink = call_cutedsl(
        _make_launcher(
            total_seqlen_q=total_seqlen_q,
            total_seqlen_kv=total_seqlen_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            block_tile=block_tile,
            softmax_scale=resolved_scale,
            has_topk_length=topk_length is not None,
        ),
        inputs,
        outputs=(
            BufferSpec("dq", q_shape, jnp.bfloat16, tensor_spec=tensor_spec),
            BufferSpec(
                "dkv",
                (total_seqlen_kv, head_dim),
                jnp.bfloat16,
                tensor_spec=tensor_spec,
                fill_value=0,
            ),
            BufferSpec("d_sink", (num_heads,), jnp.float32, fill_value=0.0),
        ),
        workspaces=(
            BufferSpec(
                "workspace_lse_odo",
                workspace_lse_odo_shape,
                jnp.uint8,
                fill_value=0,
            ),
            BufferSpec(
                "workspace_dkv",
                workspace_dkv_shape,
                jnp.uint8,
                fill_value=0,
            ),
        ),
        input_specs=input_specs,
        use_static_tensors=True,
    )
    return SparseAttentionBackwardResult(dq=dq, dkv=dkv, d_sink=d_sink)


class SparseAttentionBackward(ApiBaseJax):
    """Sample-signature-bound JAX callable for SM100 sparse-attention backward."""

    def __init__(
        self,
        sample_q: Any,
        sample_kv: Any,
        sample_out: Any,
        sample_dout: Any,
        sample_lse: Any,
        sample_attn_sink: Any,
        sample_topk_idxs: Any,
        softmax_scale: Optional[float] = None,
        sample_topk_length: Optional[Any] = None,
        block_tile: int = 64,
    ) -> None:
        super().__init__()
        self.q_desc = self.make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self.make_tensor_desc(sample_kv, name="sample_kv")
        self.out_desc = self.make_tensor_desc(sample_out, name="sample_out")
        self.dout_desc = self.make_tensor_desc(sample_dout, name="sample_dout")
        self.lse_desc = self.make_tensor_desc(sample_lse, name="sample_lse")
        self.attn_sink_desc = self.make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_idxs_desc = self.make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.topk_length_desc = self.make_optional_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = softmax_scale
        self.block_tile = block_tile

    def _check_support(self) -> bool:
        _sparse_attention_backward_impl(
            self.q_desc,
            self.kv_desc,
            self.out_desc,
            self.dout_desc,
            self.lse_desc,
            self.attn_sink_desc,
            self.topk_idxs_desc,
            self.softmax_scale,
            self.topk_length_desc,
            self.block_tile,
            _validate_only=True,
        )
        return True

    def __call__(
        self,
        q: Any,
        kv: Any,
        out: Any,
        dout: Any,
        lse: Any,
        attn_sink: Any,
        topk_idxs: Any,
        topk_length: Optional[Any] = None,
    ) -> SparseAttentionBackwardResult:
        return super().__call__(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)

    def _call_impl(
        self,
        q: Any,
        kv: Any,
        out: Any,
        dout: Any,
        lse: Any,
        attn_sink: Any,
        topk_idxs: Any,
        topk_length: Optional[Any] = None,
    ) -> SparseAttentionBackwardResult:
        for value, expected, name in (
            (q, self.q_desc, "Q"),
            (kv, self.kv_desc, "KV"),
            (out, self.out_desc, "O"),
            (dout, self.dout_desc, "dO"),
            (lse, self.lse_desc, "LSE"),
            (attn_sink, self.attn_sink_desc, "attn_sink"),
            (topk_idxs, self.topk_idxs_desc, "topk_idxs"),
        ):
            self.check_tensor_signature(value, expected, name=name)
        self.check_optional_tensor_signature(topk_length, self.topk_length_desc, name="topk_length")
        return _sparse_attention_backward_impl(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            self.softmax_scale,
            topk_length,
            self.block_tile,
        )


def sparse_attention_backward_wrapper(
    q: Any,
    kv: Any,
    out: Any,
    dout: Any,
    lse: Any,
    attn_sink: Any,
    topk_idxs: Any,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[Any] = None,
    block_tile: int = 64,
) -> SparseAttentionBackwardResult:
    """Compute fixed-shape DSA sparse-attention gradients on SM100."""

    return SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        softmax_scale=softmax_scale,
        sample_topk_length=topk_length,
        block_tile=block_tile,
    )(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)


__all__ = [
    "SparseAttentionBackward",
    "SparseAttentionBackwardResult",
    "sparse_attention_backward_wrapper",
]
