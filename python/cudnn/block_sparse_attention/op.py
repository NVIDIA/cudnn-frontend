# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Framework-neutral contracts for block-sparse attention kernels."""

from __future__ import annotations

from typing import Any

from .. import data_type
from .._op import Op
from .._tensor_desc import TensorDesc


SUPPORTED_FORWARD_COMPUTE_CAPABILITIES = (90, 100, 103, 107, 110, 120)
SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES = (90, 100, 103, 107, 110)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def compute_capability_family(compute_capability: int) -> int:
    if compute_capability == 90:
        return 90
    if 100 <= compute_capability < 120:
        return 100
    if compute_capability == 120:
        return 120
    raise ValueError(
        f"Unsupported block-sparse attention target SM{compute_capability}"
    )


def _require_shape(desc: TensorDesc[Any], expected: tuple[int, ...]) -> None:
    if desc.shape != expected:
        raise ValueError(
            f"{desc.name or 'tensor'} shape must be {expected}, got {desc.shape}"
        )


def _require_dtype(desc: TensorDesc[Any], expected: data_type) -> None:
    if desc.cudnn_dtype != expected:
        raise ValueError(
            f"{desc.name or 'tensor'} dtype must be {expected}, got {desc.cudnn_dtype}"
        )


class BlockSparseAttentionForwardOp(Op):
    """Complete, framework-neutral signature for BSA forward."""

    def __init__(
        self,
        *,
        q: TensorDesc[Any],
        k: TensorDesc[Any],
        v: TensorDesc[Any],
        block_index: TensorDesc[Any],
        output: TensorDesc[Any],
        lse: TensorDesc[Any],
        block_sizes: TensorDesc[Any] | None,
        block_nums: TensorDesc[Any] | None,
        block_sparse_num: int,
        sparse_block_size: int,
        softmax_scale: float,
        pack_gqa: bool | None,
        allow_empty_block_nums: bool,
        kv_splits: int,
        use_clc: bool | None,
        target_compute_capability: int,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.block_index = block_index
        self.output = output
        self.lse = lse
        self.block_sizes = block_sizes
        self.block_nums = block_nums
        self.block_sparse_num = block_sparse_num
        self.sparse_block_size = sparse_block_size
        self.softmax_scale = softmax_scale
        self.pack_gqa = pack_gqa
        self.allow_empty_block_nums = allow_empty_block_nums
        self.kv_splits = kv_splits
        self.use_clc = use_clc
        self.target_compute_capability = target_compute_capability

        self.batch = 0
        self.num_q_heads = 0
        self.num_kv_heads = 0
        self.seqlen_q = 0
        self.seqlen_k = 0
        self.head_dim = 0
        self.value_dim = 0
        self.num_q_blocks = 0
        self.num_kv_blocks = 0
        self.gqa_ratio = 0
        self.pack_gqa_effective = False

    def check_support(self) -> bool:
        if self.target_compute_capability not in SUPPORTED_FORWARD_COMPUTE_CAPABILITIES:
            raise ValueError(
                f"BlockSparseAttentionForward has no kernel for SM{self.target_compute_capability}; "
                f"supported targets are {SUPPORTED_FORWARD_COMPUTE_CAPABILITIES}"
            )
        family = compute_capability_family(self.target_compute_capability)
        if self.q.ndim != 4 or self.k.ndim != 4 or self.v.ndim != 4:
            raise ValueError("q, k, and v must all be rank-4 tensors")

        batch, num_q_heads, seqlen_q, head_dim = self.q.shape
        batch_k, num_kv_heads, seqlen_k, head_dim_k = self.k.shape
        batch_v, num_v_heads, seqlen_v, value_dim = self.v.shape
        if (
            min(
                batch,
                num_q_heads,
                num_kv_heads,
                seqlen_q,
                seqlen_k,
                head_dim,
                value_dim,
            )
            <= 0
        ):
            raise ValueError("q, k, and v dimensions must all be positive")
        if batch_k != batch or batch_v != batch:
            raise ValueError("q, k, and v batch dimensions must match")
        if num_v_heads != num_kv_heads or seqlen_v != seqlen_k:
            raise ValueError("k and v head and sequence dimensions must match")
        if head_dim_k != head_dim:
            raise ValueError("q and k head dimensions must match")
        if num_q_heads % num_kv_heads:
            raise ValueError(
                "the number of query heads must be divisible by the number of KV heads"
            )
        if self.q.cudnn_dtype not in {data_type.HALF, data_type.BFLOAT16}:
            raise ValueError("q, k, and v must use float16 or bfloat16")
        if (
            self.k.cudnn_dtype != self.q.cudnn_dtype
            or self.v.cudnn_dtype != self.q.cudnn_dtype
        ):
            raise ValueError("q, k, and v must have the same dtype")
        if self.q.stride[3] != 1 or self.k.stride[3] != 1 or self.v.stride[3] != 1:
            raise ValueError("q, k, and v must have a contiguous head dimension")

        if self.sparse_block_size not in {64, 128}:
            raise ValueError("sparse_block_size must be 64 or 128")
        if family in {90, 120} and self.sparse_block_size != 64:
            raise ValueError(
                f"SM{self.target_compute_capability} forward requires sparse_block_size=64"
            )
        if family == 90:
            if head_dim not in {64, 96, 128} or value_dim not in {64, 96, 128}:
                raise ValueError(
                    "SM90 forward supports QK and V dimensions 64, 96, or 128"
                )
            if seqlen_q % 64:
                raise ValueError(
                    "SM90 forward requires seqlen_q to be a multiple of 64"
                )
        elif family == 120:
            if (head_dim, value_dim) != (128, 128):
                raise ValueError("SM120 forward requires QK and V dimensions of 128")
        elif self.sparse_block_size == 64:
            if self.q.cudnn_dtype != data_type.BFLOAT16 or (head_dim, value_dim) != (
                128,
                128,
            ):
                raise ValueError(
                    "SM100-family blk64 forward requires BF16 and QK=V=128"
                )
            if num_q_heads != num_kv_heads:
                raise ValueError("SM100-family blk64 forward supports MHA only")
        elif (head_dim, value_dim) not in {(64, 64), (96, 96), (128, 128)}:
            raise ValueError(
                "SM100-family blk128 forward supports (QK, V) dimensions "
                f"(64, 64), (96, 96), or (128, 128); got ({head_dim}, {value_dim})"
            )

        gqa_ratio = num_q_heads // num_kv_heads
        if self.pack_gqa is True and not (
            family == 100 and self.sparse_block_size == 128
        ):
            raise ValueError(
                "pack_gqa is available only on the SM100-family blk128 path"
            )
        if self.pack_gqa is True and 128 % gqa_ratio:
            raise ValueError("pack_gqa=True requires the GQA ratio to divide 128")
        pack_gqa_effective = (
            family == 100
            and self.sparse_block_size == 128
            and (gqa_ratio > 1 if self.pack_gqa is None else self.pack_gqa)
            and 128 % gqa_ratio == 0
        )
        metadata_heads = num_kv_heads if pack_gqa_effective else num_q_heads
        metadata_tokens = seqlen_q * gqa_ratio if pack_gqa_effective else seqlen_q
        num_q_blocks = ceil_div(metadata_tokens, self.sparse_block_size)
        num_kv_blocks = ceil_div(seqlen_k, self.sparse_block_size)

        if self.block_index.ndim != 4:
            raise ValueError("block_index must be rank 4")
        _require_dtype(self.block_index, data_type.INT32)
        if self.block_index.shape[:3] != (batch, metadata_heads, num_q_blocks):
            raise ValueError(
                "block_index shape prefix must be "
                f"{(batch, metadata_heads, num_q_blocks)}, got {self.block_index.shape[:3]}"
            )
        capacity = self.block_index.shape[3]
        if capacity <= 0:
            raise ValueError("block_index must have non-empty KV-block capacity")

        if self.block_nums is None:
            minimum = 2 if family == 100 and self.sparse_block_size == 128 else 1
            if not minimum <= self.block_sparse_num <= capacity:
                raise ValueError(
                    f"block_sparse_num must be in [{minimum}, {capacity}], got {self.block_sparse_num}"
                )
            if minimum == 2 and self.block_sparse_num % 2:
                raise ValueError(
                    "block_sparse_num must be even for the SM100-family blk128 kernel"
                )
        else:
            _require_dtype(self.block_nums, data_type.INT32)
            _require_shape(self.block_nums, (batch, metadata_heads, num_q_blocks))

        if self.block_sizes is not None:
            _require_dtype(self.block_sizes, data_type.INT32)
            allowed_ranks = {1, 2, 3} if family in {90, 120} else {1}
            if self.block_sizes.ndim not in allowed_ranks:
                raise ValueError(
                    f"block_sizes rank must be one of {tuple(sorted(allowed_ranks))}"
                )
            expected = {
                1: (num_kv_blocks,),
                2: (batch, num_kv_blocks),
                3: (batch, metadata_heads, num_kv_blocks),
            }[self.block_sizes.ndim]
            _require_shape(self.block_sizes, expected)

        if (
            isinstance(self.kv_splits, bool)
            or not isinstance(self.kv_splits, int)
            or not 1 <= self.kv_splits <= 256
        ):
            raise ValueError(
                f"kv_splits must be an integer in [1, 256], got {self.kv_splits!r}"
            )
        if self.use_clc is not None and not isinstance(self.use_clc, bool):
            raise TypeError(
                f"use_clc must be a bool or None, got {type(self.use_clc).__name__}"
            )
        if self.use_clc is not None and not (
            family == 100 and self.sparse_block_size == 64
        ):
            raise ValueError("use_clc is available only on the SM100-family blk64 path")
        if family == 120 and self.kv_splits != 1:
            raise ValueError("SM120 forward does not support kv_splits")
        if family == 100 and self.sparse_block_size == 128 and self.kv_splits != 1:
            raise ValueError("SM100-family blk128 forward does not support kv_splits")
        if self.kv_splits > 1 and self.use_clc:
            raise ValueError("kv_splits > 1 does not support use_clc=True")

        _require_shape(self.output, (batch, num_q_heads, seqlen_q, value_dim))
        _require_dtype(self.output, self.q.cudnn_dtype)
        _require_shape(self.lse, (batch, num_q_heads, seqlen_q))
        _require_dtype(self.lse, data_type.FLOAT)

        self.batch = batch
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.num_q_blocks = num_q_blocks
        self.num_kv_blocks = num_kv_blocks
        self.gqa_ratio = gqa_ratio
        self.pack_gqa_effective = pack_gqa_effective
        return True


class BlockSparseAttentionBackwardOp(Op):
    """Complete, framework-neutral signature for BSA backward."""

    def __init__(
        self,
        *,
        dout: TensorDesc[Any],
        q: TensorDesc[Any],
        k: TensorDesc[Any],
        v: TensorDesc[Any],
        output: TensorDesc[Any],
        lse: TensorDesc[Any],
        block_index: TensorDesc[Any],
        dq: TensorDesc[Any],
        dk: TensorDesc[Any],
        dv: TensorDesc[Any],
        block_sizes: TensorDesc[Any] | None,
        block_nums: TensorDesc[Any] | None,
        block_sparse_num: int,
        sparse_block_size: int,
        softmax_scale: float,
        bucket_size_blocks: int,
        target_compute_capability: int,
    ) -> None:
        self.dout = dout
        self.q = q
        self.k = k
        self.v = v
        self.output = output
        self.lse = lse
        self.block_index = block_index
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.block_sizes = block_sizes
        self.block_nums = block_nums
        self.block_sparse_num = block_sparse_num
        self.sparse_block_size = sparse_block_size
        self.softmax_scale = softmax_scale
        self.bucket_size_blocks = bucket_size_blocks
        self.target_compute_capability = target_compute_capability

        self.batch = 0
        self.num_heads = 0
        self.seqlen_q = 0
        self.seqlen_k = 0
        self.head_dim = 0
        self.num_q_blocks = 0
        self.num_kv_blocks = 0
        self.num_q_groups = 0
        self.max_edges = 0

    def check_support(self) -> bool:
        if (
            self.target_compute_capability
            not in SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES
        ):
            raise ValueError(
                f"BlockSparseAttentionBackward has no kernel for SM{self.target_compute_capability}; "
                f"supported targets are {SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES}"
            )
        family = compute_capability_family(self.target_compute_capability)
        if any(
            desc.ndim != 4
            for desc in (
                self.dout,
                self.q,
                self.k,
                self.v,
                self.output,
                self.dq,
                self.dk,
                self.dv,
            )
        ):
            raise ValueError("BSA backward data tensors must all be rank 4")
        batch, num_heads, seqlen_q, head_dim = self.q.shape
        batch_k, num_kv_heads, seqlen_k, head_dim_k = self.k.shape
        if min(batch, num_heads, seqlen_q, seqlen_k, head_dim) <= 0:
            raise ValueError("BSA backward dimensions must be positive")
        if batch_k != batch or num_kv_heads != num_heads or head_dim_k != head_dim:
            raise ValueError(
                "BSA backward supports MHA with matching q and k dimensions"
            )
        _require_shape(self.v, self.k.shape)
        _require_shape(self.dout, self.q.shape)
        _require_shape(self.output, self.q.shape)
        _require_shape(self.dq, self.q.shape)
        _require_shape(self.dk, self.k.shape)
        _require_shape(self.dv, self.v.shape)
        if self.q.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError("BSA backward requires bfloat16 data tensors")
        for desc in (self.dout, self.k, self.v, self.output, self.dq, self.dk, self.dv):
            _require_dtype(desc, data_type.BFLOAT16)
        if any(
            desc.stride[3] != 1
            for desc in (
                self.dout,
                self.q,
                self.k,
                self.v,
                self.output,
                self.dq,
                self.dk,
                self.dv,
            )
        ):
            raise ValueError(
                "BSA backward data tensors must have a contiguous head dimension"
            )
        _require_shape(self.lse, (batch, num_heads, seqlen_q))
        _require_dtype(self.lse, data_type.FLOAT)

        if family == 90:
            if self.sparse_block_size != 64 or head_dim != 128:
                raise ValueError(
                    "SM90 backward requires sparse_block_size=64 and head_dim=128"
                )
        elif self.sparse_block_size == 64:
            if head_dim != 128:
                raise ValueError("SM100-family blk64 backward requires head_dim=128")
        elif self.sparse_block_size == 128:
            if head_dim not in {64, 128}:
                raise ValueError(
                    "SM100-family blk128 backward requires head_dim=64 or 128"
                )
            if self.block_sizes is not None:
                raise ValueError(
                    "SM100-family blk128 backward does not support block_sizes"
                )
        else:
            raise ValueError("sparse_block_size must be 64 or 128")

        num_q_blocks = ceil_div(seqlen_q, self.sparse_block_size)
        num_kv_blocks = ceil_div(seqlen_k, self.sparse_block_size)
        if self.block_index.ndim != 4:
            raise ValueError("block_index must be rank 4")
        _require_dtype(self.block_index, data_type.INT32)
        if self.block_index.shape[:3] != (batch, num_heads, num_q_blocks):
            raise ValueError(
                f"block_index shape prefix must be {(batch, num_heads, num_q_blocks)}, got {self.block_index.shape[:3]}"
            )
        capacity = self.block_index.shape[3]
        if capacity <= 0:
            raise ValueError("block_index must have non-empty KV-block capacity")
        if self.block_nums is None:
            if not 1 <= self.block_sparse_num <= capacity:
                raise ValueError(
                    f"block_sparse_num must be in [1, {capacity}], got {self.block_sparse_num}"
                )
            if self.sparse_block_size == 128 and self.block_sparse_num % 2:
                raise ValueError(
                    "block_sparse_num must be even for the blk128 backward kernel"
                )
            max_edges = num_q_blocks * self.block_sparse_num
        else:
            _require_dtype(self.block_nums, data_type.INT32)
            _require_shape(self.block_nums, (batch, num_heads, num_q_blocks))
            max_edges = num_q_blocks * capacity

        if self.block_sizes is not None:
            _require_dtype(self.block_sizes, data_type.INT32)
            if self.block_sizes.ndim not in {1, 2}:
                raise ValueError("block_sizes must have rank 1 or 2 for BSA backward")
            expected = (
                (num_kv_blocks,)
                if self.block_sizes.ndim == 1
                else (batch, num_kv_blocks)
            )
            _require_shape(self.block_sizes, expected)
        if (
            isinstance(self.bucket_size_blocks, bool)
            or not isinstance(self.bucket_size_blocks, int)
            or self.bucket_size_blocks <= 0
        ):
            raise ValueError(
                f"bucket_size_blocks must be positive, got {self.bucket_size_blocks!r}"
            )

        self.batch = batch
        self.num_heads = num_heads
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.head_dim = head_dim
        self.num_q_blocks = num_q_blocks
        self.num_kv_blocks = num_kv_blocks
        self.num_q_groups = ceil_div(num_q_blocks, self.bucket_size_blocks)
        self.max_edges = max(1, max_edges)
        return True


__all__ = [
    "BlockSparseAttentionBackwardOp",
    "BlockSparseAttentionForwardOp",
    "SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES",
    "SUPPORTED_FORWARD_COMPUTE_CAPABILITIES",
    "ceil_div",
    "compute_capability_family",
]
