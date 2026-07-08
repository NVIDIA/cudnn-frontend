# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral DeepSeek sparse-attention backward operation."""

from __future__ import annotations

import math
from typing import Any, Optional

from ... import data_type
from ..._op import Op
from ..._tensor_desc import TensorDesc

SUPPORTED_HEAD_DIMS = (512, 576)
BLOCK_TILE = 64


class SparseAttentionBackwardOp(Op):
    """Complete logical signature for DeepSeek sparse-attention backward.

    The operation describes the flat MQA interface shared by the Hopper and
    Blackwell implementations. Framework adapters own device selection,
    allocation, and lowering.
    """

    def __init__(
        self,
        *,
        q: TensorDesc[Any],
        kv: TensorDesc[Any],
        output: TensorDesc[Any],
        doutput: TensorDesc[Any],
        lse: TensorDesc[Any],
        attn_sink: TensorDesc[Any],
        topk_idxs: TensorDesc[Any],
        dq: TensorDesc[Any],
        dkv: TensorDesc[Any],
        d_sink: TensorDesc[Any],
        topk_length: Optional[TensorDesc[Any]] = None,
        softmax_scale: Optional[float] = None,
        block_tile: int = BLOCK_TILE,
    ) -> None:
        descriptors = (
            ("q", q),
            ("kv", kv),
            ("output", output),
            ("doutput", doutput),
            ("lse", lse),
            ("attn_sink", attn_sink),
            ("topk_idxs", topk_idxs),
            ("dq", dq),
            ("dkv", dkv),
            ("d_sink", d_sink),
        )
        for name, desc in descriptors:
            if not isinstance(desc, TensorDesc):
                raise TypeError(f"{name} must be a TensorDesc, got {type(desc).__name__}")
        if topk_length is not None and not isinstance(topk_length, TensorDesc):
            raise TypeError(f"topk_length must be a TensorDesc or None, got {type(topk_length).__name__}")

        self.q = q
        self.kv = kv
        self.output = output
        self.doutput = doutput
        self.lse = lse
        self.attn_sink = attn_sink
        self.topk_idxs = topk_idxs
        self.topk_length = topk_length
        self.dq = dq
        self.dkv = dkv
        self.d_sink = d_sink
        self.requested_softmax_scale = softmax_scale
        self.block_tile = int(block_tile)

        self.total_seqlen_q: Optional[int] = None
        self.total_seqlen_kv: Optional[int] = None
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.max_topk: Optional[int] = None
        self.softmax_scale: Optional[float] = None

    def check_support(self) -> bool:
        """Validate the complete tensor signature and static configuration."""

        self.total_seqlen_q = None
        self.total_seqlen_kv = None
        self.num_heads = None
        self.head_dim = None
        self.head_dim_v = None
        self.max_topk = None
        self.softmax_scale = None

        self._check_ranks_and_dtypes()

        total_seqlen_q, num_heads, head_dim = self.q.shape
        total_seqlen_kv, kv_head_dim = self.kv.shape
        max_topk = self.topk_idxs.shape[1]
        head_dim_v = 512 if head_dim == 576 else head_dim

        dimensions = {
            "total_seqlen_q": total_seqlen_q,
            "total_seqlen_kv": total_seqlen_kv,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "max_topk": max_topk,
        }
        invalid = ", ".join(f"{name}={value}" for name, value in dimensions.items() if value <= 0)
        if invalid:
            raise ValueError(f"Sparse-attention dimensions must be positive, got {invalid}")
        if kv_head_dim != head_dim:
            raise ValueError(f"KV head dimension must match Q ({head_dim}), got {kv_head_dim}")
        if head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}, got {head_dim}")
        if num_heads % BLOCK_TILE != 0:
            raise ValueError(f"num_heads must be divisible by {BLOCK_TILE}, got {num_heads}")
        if self.block_tile != BLOCK_TILE:
            raise ValueError(f"block_tile must be {BLOCK_TILE}, got {self.block_tile}")

        self._check_shapes(total_seqlen_q, total_seqlen_kv, num_heads, head_dim, head_dim_v, max_topk)
        self._check_compact_layouts()

        if self.requested_softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        else:
            try:
                softmax_scale = float(self.requested_softmax_scale)
            except (TypeError, ValueError) as error:
                raise TypeError(f"softmax_scale must be a real scalar or None, got {self.requested_softmax_scale!r}") from error
            if not math.isfinite(softmax_scale):
                raise ValueError(f"softmax_scale must be finite, got {softmax_scale}")

        self.total_seqlen_q = total_seqlen_q
        self.total_seqlen_kv = total_seqlen_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.max_topk = max_topk
        self.softmax_scale = softmax_scale
        return True

    def _check_ranks_and_dtypes(self) -> None:
        expected_ranks = (
            (self.q, 3, "Q"),
            (self.kv, 2, "KV"),
            (self.output, 3, "O"),
            (self.doutput, 3, "dO"),
            (self.lse, 2, "LSE"),
            (self.attn_sink, 1, "attn_sink"),
            (self.topk_idxs, 2, "topk_idxs"),
            (self.dq, 3, "dQ"),
            (self.dkv, 2, "dKV"),
            (self.d_sink, 1, "d_sink"),
        )
        if self.topk_length is not None:
            expected_ranks += ((self.topk_length, 1, "topk_length"),)
        for desc, rank, name in expected_ranks:
            if desc.ndim != rank:
                raise ValueError(f"{name} must have rank {rank}, got shape {desc.shape}")

        if self.q.cudnn_dtype not in (data_type.HALF, data_type.BFLOAT16):
            raise ValueError(f"Q must have dtype float16 or bfloat16, got {self.q.dtype}")
        for desc, name in (
            (self.kv, "KV"),
            (self.output, "O"),
            (self.doutput, "dO"),
            (self.dq, "dQ"),
            (self.dkv, "dKV"),
        ):
            if desc.cudnn_dtype != self.q.cudnn_dtype:
                raise ValueError(f"{name} must have the same dtype as Q, got {desc.dtype}")
        for desc, name in ((self.lse, "LSE"), (self.attn_sink, "attn_sink"), (self.d_sink, "d_sink")):
            if desc.cudnn_dtype != data_type.FLOAT:
                raise ValueError(f"{name} must have dtype float32, got {desc.dtype}")
        for desc, name in ((self.topk_idxs, "topk_idxs"), (self.topk_length, "topk_length")):
            if desc is not None and desc.cudnn_dtype != data_type.INT32:
                raise ValueError(f"{name} must have dtype int32, got {desc.dtype}")

    def _check_shapes(
        self,
        total_seqlen_q: int,
        total_seqlen_kv: int,
        num_heads: int,
        head_dim: int,
        head_dim_v: int,
        max_topk: int,
    ) -> None:
        expected_shapes = (
            (self.output, (total_seqlen_q, num_heads, head_dim_v), "O"),
            (self.doutput, (total_seqlen_q, num_heads, head_dim_v), "dO"),
            (self.lse, (total_seqlen_q, num_heads), "LSE"),
            (self.attn_sink, (num_heads,), "attn_sink"),
            (self.topk_idxs, (total_seqlen_q, max_topk), "topk_idxs"),
            (self.dq, (total_seqlen_q, num_heads, head_dim), "dQ"),
            (self.dkv, (total_seqlen_kv, head_dim), "dKV"),
            (self.d_sink, (num_heads,), "d_sink"),
        )
        if self.topk_length is not None:
            expected_shapes += ((self.topk_length, (total_seqlen_q,), "topk_length"),)
        for desc, expected, name in expected_shapes:
            if desc.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {desc.shape}")

    def _check_compact_layouts(self) -> None:
        descriptors = (
            self.q,
            self.kv,
            self.output,
            self.doutput,
            self.lse,
            self.attn_sink,
            self.topk_idxs,
            self.dq,
            self.dkv,
            self.d_sink,
        )
        if self.topk_length is not None:
            descriptors += (self.topk_length,)
        for desc in descriptors:
            if not desc.is_compact(tuple(reversed(range(desc.ndim)))):
                name = desc.name or "tensor"
                raise ValueError(f"{name} must be row-major contiguous, got stride {desc.stride}")


__all__ = ["BLOCK_TILE", "SUPPORTED_HEAD_DIMS", "SparseAttentionBackwardOp"]
