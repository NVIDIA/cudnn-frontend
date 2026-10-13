# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-order reduction for deterministic SM100 indexer dK shards."""

from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor

NUM_DK_SHARDS = 128
_FOLD_GROUP_SIZE = 8
_NUM_FOLD_GROUPS = NUM_DK_SHARDS // _FOLD_GROUP_SIZE
_compile_cache: dict = {}


def deterministic_dk_workspace_bytes(batch: int, seqlen_k: int, head_dim: int) -> int:
    """Return the exact byte count for the fixed FP32 dK shard array."""
    return NUM_DK_SHARDS * batch * seqlen_k * head_dim * torch.float32.itemsize


class DeterministicDkReduceSm100:
    """Fold 128 CTA-private FP32 shards in a fixed two-level order."""

    arch = 100

    def __init__(self, head_dim: int):
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        mdK_shards: cute.Tensor,
        mdK: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Both tensors are contiguous. Keep the shard-major scratch layout so
        # producer CTA i and reducer group i address the same dense plane.
        seqlen_k = cute.size(mdK.shape[0]) * cute.size(mdK.shape[1])
        shards = cute.make_tensor(
            mdK_shards.iterator,
            cute.make_layout(
                (self.head_dim, seqlen_k * NUM_DK_SHARDS),
                stride=(1, cutlass.Int64(self.head_dim)),
            ),
        )
        out = cute.make_tensor(
            mdK.iterator,
            cute.make_layout(
                (self.head_dim, seqlen_k),
                stride=(1, cutlass.Int64(self.head_dim)),
            ),
        )
        self.fold_groups(shards, seqlen_k).launch(
            grid=(seqlen_k, _NUM_FOLD_GROUPS, 1),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )
        self.store_output(shards, out, seqlen_k).launch(
            grid=(seqlen_k, 1, 1),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def fold_groups(self, mdK_shards: cute.Tensor, seqlen_k: Int32):
        """Fold each adjacent group of eight shards into its first shard."""
        row_idx, group_idx, _ = cute.arch.block_idx()
        dim_idx, _, _ = cute.arch.thread_idx()
        if row_idx < seqlen_k and dim_idx < self.head_dim:
            group_begin = group_idx * _FOLD_GROUP_SIZE
            acc = Float32(0.0)
            for local_shard in cutlass.range_constexpr(_FOLD_GROUP_SIZE):
                shard_idx = group_begin + local_shard
                acc += mdK_shards[dim_idx, row_idx + shard_idx * seqlen_k]
            mdK_shards[dim_idx, row_idx + group_begin * seqlen_k] = acc

    @cute.kernel
    def store_output(self, mdK_shards: cute.Tensor, mdK: cute.Tensor, seqlen_k: Int32):
        """Fold the 16 group totals and convert once to the output dtype."""
        row_idx, _, _ = cute.arch.block_idx()
        dim_idx, _, _ = cute.arch.thread_idx()
        if row_idx < seqlen_k and dim_idx < self.head_dim:
            acc = Float32(0.0)
            for group_idx in cutlass.range_constexpr(_NUM_FOLD_GROUPS):
                shard_idx = group_idx * _FOLD_GROUP_SIZE
                acc += mdK_shards[dim_idx, row_idx + shard_idx * seqlen_k]
            mdK[dim_idx, row_idx] = mdK.element_type(acc)


def reduce_deterministic_dk(
    d_index_k_shards: torch.Tensor,
    d_index_k: torch.Tensor,
    current_stream=None,
) -> None:
    """Reduce caller-owned shards into ``d_index_k`` without allocation."""
    stream = resolve_stream(current_stream)
    key = (d_index_k.device, d_index_k.shape[-1], d_index_k.dtype)
    fn = _compile_cache.get(key)
    if fn is None:
        kernel = DeterministicDkReduceSm100(d_index_k.shape[-1])
        fn = cute.compile(
            kernel,
            to_cute_tensor(d_index_k_shards),
            to_cute_tensor(d_index_k),
            stream,
            options=compile_options("--opt-level 2"),
        )
        _compile_cache[key] = fn
    fn(d_index_k_shards, d_index_k, stream)
