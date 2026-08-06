# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import torch


class BucketedK2QCsrUniversal:
    """Build the bucketed K-to-Q CSR task layout on the GPU."""

    def __init__(
        self,
        block_sparse_num: int,
        bucket_size_blocks: int,
        has_variable_block_nums: bool,
        max_kv_blocks: int,
    ):
        self.block_sparse_num = block_sparse_num
        self.bucket_size_blocks = bucket_size_blocks
        self.has_variable_block_nums = has_variable_block_nums
        self.num_edge_threads = 256
        self.edge_width = max_kv_blocks if has_variable_block_nums else block_sparse_num
        self.num_edge_tiles = (self.edge_width + self.num_edge_threads - 1) // self.num_edge_threads

    @cute.jit
    def __call__(
        self,
        mCounts: cute.Tensor,
        mLocalOffsets: cute.Tensor,
        mGroupTotals: cute.Tensor,
        mBucketedK2qOffsets: cute.Tensor,
        mCursors: cute.Tensor,
        mBucketedK2qIndices: cute.Tensor,
        mQ2kBlockIndex: cute.Tensor,
        mQ2kBlockNums: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        if const_expr(
            any(
                tensor.element_type != Int32
                for tensor in (
                    mCounts,
                    mLocalOffsets,
                    mGroupTotals,
                    mBucketedK2qOffsets,
                    mCursors,
                    mBucketedK2qIndices,
                    mQ2kBlockIndex,
                    mQ2kBlockNums,
                )
            )
        ):
            raise TypeError("Bucketed K-to-Q CSR tensors must use int32")

        batch_size, num_heads, num_q_blocks, _ = mQ2kBlockIndex.shape
        num_q_groups = mCounts.shape[2]
        edge_grid = (
            num_q_blocks * self.num_edge_tiles,
            num_heads,
            batch_size,
        )
        if const_expr(self.edge_width > 0):
            self._count_edges_kernel(
                mCounts,
                mQ2kBlockIndex,
                mQ2kBlockNums,
            ).launch(
                grid=edge_grid,
                block=(self.num_edge_threads, 1, 1),
                stream=stream,
            )

        group_grid = (num_q_groups, num_heads, batch_size)
        self._local_offsets_kernel(
            mCounts,
            mLocalOffsets,
            mGroupTotals,
        ).launch(
            grid=group_grid,
            block=(1, 1, 1),
            stream=stream,
        )
        self._finalize_offsets_kernel(
            mLocalOffsets,
            mGroupTotals,
            mBucketedK2qOffsets,
            mCursors,
        ).launch(
            grid=group_grid,
            block=(1, 1, 1),
            stream=stream,
        )

        if const_expr(self.edge_width > 0):
            self._scatter_q_indices_kernel(
                mCursors,
                mBucketedK2qIndices,
                mQ2kBlockIndex,
                mQ2kBlockNums,
            ).launch(
                grid=edge_grid,
                block=(self.num_edge_threads, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def _count_edges_kernel(
        self,
        mCounts: cute.Tensor,
        mQ2kBlockIndex: cute.Tensor,
        mQ2kBlockNums: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_edge_tile_idx, head_idx, batch_idx = cute.arch.block_idx()
        q_block_idx = q_edge_tile_idx // self.num_edge_tiles
        edge_tile_idx = q_edge_tile_idx - q_block_idx * self.num_edge_tiles
        block_idx = edge_tile_idx * self.num_edge_threads + tidx
        num_kv_blocks = mCounts.shape[3]
        if block_idx < self.edge_width:
            num_active_blocks = Int32(self.block_sparse_num)
            if const_expr(self.has_variable_block_nums):
                num_active_blocks = mQ2kBlockNums[batch_idx, head_idx, q_block_idx]
            if block_idx < num_active_blocks:
                kv_block_idx = mQ2kBlockIndex[batch_idx, head_idx, q_block_idx, block_idx]
                if kv_block_idx >= 0:
                    if kv_block_idx < num_kv_blocks:
                        q_group_idx = q_block_idx // self.bucket_size_blocks
                        count_ptr = mCounts.iterator + cute.crd2idx(
                            (
                                batch_idx,
                                head_idx,
                                q_group_idx,
                                kv_block_idx,
                            ),
                            mCounts.layout,
                        )
                        cute.arch.atomic_add(
                            count_ptr.llvm_ptr,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )

    @cute.kernel
    def _local_offsets_kernel(
        self,
        mCounts: cute.Tensor,
        mLocalOffsets: cute.Tensor,
        mGroupTotals: cute.Tensor,
    ):
        q_group_idx, head_idx, batch_idx = cute.arch.block_idx()
        num_kv_blocks = mCounts.shape[3]
        running = Int32(0)
        for kv_block_idx in cutlass.range(num_kv_blocks, unroll=1):
            mLocalOffsets[batch_idx, head_idx, q_group_idx, kv_block_idx] = running
            running += mCounts[batch_idx, head_idx, q_group_idx, kv_block_idx]
        mLocalOffsets[batch_idx, head_idx, q_group_idx, num_kv_blocks] = running
        mGroupTotals[batch_idx, head_idx, q_group_idx] = running

    @cute.kernel
    def _finalize_offsets_kernel(
        self,
        mLocalOffsets: cute.Tensor,
        mGroupTotals: cute.Tensor,
        mBucketedK2qOffsets: cute.Tensor,
        mCursors: cute.Tensor,
    ):
        q_group_idx, head_idx, batch_idx = cute.arch.block_idx()
        num_kv_blocks = mCursors.shape[3]

        group_base = Int32(0)
        for previous_group_idx in cutlass.range(q_group_idx, unroll=1):
            group_base += mGroupTotals[batch_idx, head_idx, previous_group_idx]

        for kv_block_idx in cutlass.range(num_kv_blocks, unroll=1):
            offset = group_base + mLocalOffsets[batch_idx, head_idx, q_group_idx, kv_block_idx]
            mBucketedK2qOffsets[batch_idx, head_idx, q_group_idx, kv_block_idx] = offset
            mCursors[batch_idx, head_idx, q_group_idx, kv_block_idx] = offset
        mBucketedK2qOffsets[batch_idx, head_idx, q_group_idx, num_kv_blocks] = group_base + mLocalOffsets[batch_idx, head_idx, q_group_idx, num_kv_blocks]

    @cute.kernel
    def _scatter_q_indices_kernel(
        self,
        mCursors: cute.Tensor,
        mBucketedK2qIndices: cute.Tensor,
        mQ2kBlockIndex: cute.Tensor,
        mQ2kBlockNums: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_edge_tile_idx, head_idx, batch_idx = cute.arch.block_idx()
        q_block_idx = q_edge_tile_idx // self.num_edge_tiles
        edge_tile_idx = q_edge_tile_idx - q_block_idx * self.num_edge_tiles
        block_idx = edge_tile_idx * self.num_edge_threads + tidx
        num_kv_blocks = mCursors.shape[3]
        if block_idx < self.edge_width:
            num_active_blocks = Int32(self.block_sparse_num)
            if const_expr(self.has_variable_block_nums):
                num_active_blocks = mQ2kBlockNums[batch_idx, head_idx, q_block_idx]
            if block_idx < num_active_blocks:
                kv_block_idx = mQ2kBlockIndex[batch_idx, head_idx, q_block_idx, block_idx]
                if kv_block_idx >= 0:
                    if kv_block_idx < num_kv_blocks:
                        q_group_idx = q_block_idx // self.bucket_size_blocks
                        cursor_ptr = mCursors.iterator + cute.crd2idx(
                            (
                                batch_idx,
                                head_idx,
                                q_group_idx,
                                kv_block_idx,
                            ),
                            mCursors.layout,
                        )
                        position = cute.arch.atomic_add(
                            cursor_ptr.llvm_ptr,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                        mBucketedK2qIndices[batch_idx, head_idx, position] = Int32(q_block_idx)


def _bucketed_k2q_csr_compile_key(
    device_capability: tuple[int, int],
    block_sparse_num: int,
    bucket_size_blocks: int,
    has_variable_block_nums: bool,
    max_kv_blocks: int,
) -> tuple:
    """Return only the configuration that changes generated device code."""
    edge_width = max_kv_blocks if has_variable_block_nums else block_sparse_num
    return (
        device_capability,
        int(bucket_size_blocks),
        bool(has_variable_block_nums),
        int(edge_width),
    )


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(
        tensor.detach(),
        assumed_align=4,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def build_bucketed_k2q_csr_cutedsl(
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    num_kv_blocks: int,
    *,
    bucket_size_blocks: int,
    q2k_block_nums: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build bucketed K-to-Q CSR metadata with CuTe DSL kernels."""
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.is_cuda
    assert q2k_block_index.ndim == 4
    assert bucket_size_blocks > 0
    assert num_kv_blocks > 0

    batch_size, num_heads, num_q_blocks, max_kv_blocks = q2k_block_index.shape
    assert num_q_blocks > 0
    q2k_block_index = q2k_block_index.contiguous()
    has_variable_block_nums = q2k_block_nums is not None
    if has_variable_block_nums:
        assert q2k_block_nums is not None
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)
        assert q2k_block_nums.is_cuda
        assert q2k_block_nums.device == q2k_block_index.device
        q2k_block_nums = q2k_block_nums.contiguous()
        max_edges = num_q_blocks * max_kv_blocks
    else:
        assert 0 <= block_sparse_num <= max_kv_blocks
        q2k_block_nums = q2k_block_index
        max_edges = num_q_blocks * int(block_sparse_num)

    assert max_edges <= torch.iinfo(torch.int32).max
    max_edges = max(1, max_edges)
    num_q_groups = (num_q_blocks + bucket_size_blocks - 1) // bucket_size_blocks
    device = q2k_block_index.device
    counts = torch.zeros(
        (batch_size, num_heads, num_q_groups, num_kv_blocks),
        dtype=torch.int32,
        device=device,
    )
    local_offsets = torch.empty(
        (batch_size, num_heads, num_q_groups, num_kv_blocks + 1),
        dtype=torch.int32,
        device=device,
    )
    group_totals = torch.empty(
        (batch_size, num_heads, num_q_groups),
        dtype=torch.int32,
        device=device,
    )
    bucketed_k2q_offsets = torch.empty_like(local_offsets)
    cursors = torch.empty_like(counts)
    bucketed_k2q_indices = torch.empty(
        (batch_size, num_heads, max_edges),
        dtype=torch.int32,
        device=device,
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream(q2k_block_index.device).cuda_stream)
    tensors = (
        counts,
        local_offsets,
        group_totals,
        bucketed_k2q_offsets,
        cursors,
        bucketed_k2q_indices,
        q2k_block_index,
        q2k_block_nums,
    )
    device_capability = torch.cuda.get_device_capability(q2k_block_index.device)
    compile_key = _bucketed_k2q_csr_compile_key(
        device_capability,
        block_sparse_num,
        bucket_size_blocks,
        has_variable_block_nums,
        max_kv_blocks,
    )
    if compile_key not in build_bucketed_k2q_csr_cutedsl.compile_cache:
        kernel = BucketedK2QCsrUniversal(
            int(block_sparse_num),
            int(bucket_size_blocks),
            has_variable_block_nums,
            max_kv_blocks,
        )
        build_bucketed_k2q_csr_cutedsl.compile_cache[compile_key] = cute.compile(
            kernel,
            *(_to_cute_tensor(tensor) for tensor in tensors),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    build_bucketed_k2q_csr_cutedsl.compile_cache[compile_key](
        *tensors,
        current_stream,
    )
    return (
        bucketed_k2q_offsets,
        bucketed_k2q_indices,
        num_q_groups,
        num_kv_blocks,
    )


build_bucketed_k2q_csr_cutedsl.compile_cache = {}
