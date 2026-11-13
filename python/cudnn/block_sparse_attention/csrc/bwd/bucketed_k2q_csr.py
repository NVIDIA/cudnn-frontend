# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Arch, BaseDSL
import cuda.bindings.driver as cuda

from cudnn._cutlass_compat import SmemAllocator
from cudnn.tensor_adapter import get_compute_capability
from ..utils.kernel_utils import warp_prefix_sum


@lru_cache(maxsize=None)
def _device_capability(device_index: int):
    return get_compute_capability(device_index)


class BucketedK2QCsrUniversal:
    """Build the bucketed K-to-Q CSR task layout on the GPU.

    Three launches per call:

    1. ``_count_edges_kernel``: one 128-thread CTA per (q_block, head, batch) strides over that row's active KV
       list and bumps ``counts[b, h, q_group, kv]``, so the work scales with the number of edges instead of
       ``num_q_blocks * max_kv_blocks`` threads.
    2. ``_offsets_kernel``: one CTA per (q_group, head, batch) sums the counts of the previous groups of its
       (batch, head) for the group base, then runs a block-wide exclusive scan over its own group's counts and
       writes the global offsets, the per-group end entry at index ``num_kv_blocks`` and the scatter cursors.
    3. ``_scatter_q_indices_kernel``: one CTA per row again, appending the row's Q-block id behind the cursors.

    On sm_90+ the kernels are launched with programmatic dependent launch (PDL): each one waits for the
    predecessor grid before touching global memory and only then lets its own dependent launch early, so the
    launch latency of the chain overlaps the predecessor's tail.
    """

    num_edge_threads = 128
    num_scan_threads = 512
    scan_items_per_thread = 8
    base_groups_per_round = 4

    def __init__(
        self,
        block_sparse_num: int,
        bucket_size_blocks: int,
        has_variable_block_nums: bool,
        max_kv_blocks: int,
        parallel_offsets: bool = False,
    ):
        # ``parallel_offsets`` is accepted for signature compatibility only: the offsets are always built by one
        # block-wide scan CTA per (q_group, head, batch) now, so it changes neither the generated code nor the
        # compile key.
        self.parallel_offsets = parallel_offsets
        self.block_sparse_num = block_sparse_num
        self.bucket_size_blocks = bucket_size_blocks
        self.has_variable_block_nums = has_variable_block_nums
        self.edge_width = max_kv_blocks if has_variable_block_nums else block_sparse_num
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a

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
        # mLocalOffsets / mGroupTotals were the scratch of the former two-kernel offsets build. They are no longer
        # read or written, but stay in the positional signature so the XLA adapter (jax_kernels.Backward), which
        # plumbs those buffers through its custom call, keeps working unchanged.
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
        row_grid = (num_q_blocks, num_heads, batch_size)
        if const_expr(self.edge_width > 0):
            self._count_edges_kernel(
                mCounts,
                mQ2kBlockIndex,
                mQ2kBlockNums,
            ).launch(
                grid=row_grid,
                block=(self.num_edge_threads, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        self._offsets_kernel(
            mCounts,
            mBucketedK2qOffsets,
            mCursors,
        ).launch(
            grid=(mCounts.shape[2], num_heads, batch_size),
            block=(self.num_scan_threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

        if const_expr(self.edge_width > 0):
            self._scatter_q_indices_kernel(
                mCursors,
                mBucketedK2qIndices,
                mQ2kBlockIndex,
                mQ2kBlockNums,
            ).launch(
                grid=row_grid,
                block=(self.num_edge_threads, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.kernel
    def _count_edges_kernel(
        self,
        mCounts: cute.Tensor,
        mQ2kBlockIndex: cute.Tensor,
        mQ2kBlockNums: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_block_idx, head_idx, batch_idx = cute.arch.block_idx()
        num_kv_blocks = mCounts.shape[3]
        q_group_idx = q_block_idx // self.bucket_size_blocks
        if const_expr(self.use_pdl):
            # No global access before this wait: the counts memset and the index producers may still be in flight.
            cute.arch.griddepcontrol_wait()
            # Trigger only after the wait, so a chain of early launches can never skip a level and park dependents
            # on SM slots while this grid's own CTAs are still pending.
            cute.arch.griddepcontrol_launch_dependents()
        num_active_blocks = Int32(self.block_sparse_num)
        if const_expr(self.has_variable_block_nums):
            # Same bounds as before: a row's count is clamped to the width of the index tensor.
            num_active_blocks = cute.min(mQ2kBlockNums[batch_idx, head_idx, q_block_idx], self.edge_width)
        for edge in cutlass.range(tidx, num_active_blocks, self.num_edge_threads):
            kv_block_idx = mQ2kBlockIndex[batch_idx, head_idx, q_block_idx, edge]
            if kv_block_idx >= 0 and kv_block_idx < num_kv_blocks:
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
    def _offsets_kernel(
        self,
        mCounts: cute.Tensor,
        mBucketedK2qOffsets: cute.Tensor,
        mCursors: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_group_idx, head_idx, batch_idx = cute.arch.block_idx()
        lane = tidx % 32
        warp_idx = tidx // 32
        num_warps = self.num_scan_threads // 32
        num_kv_blocks = mCounts.shape[3]
        # The group's offsets row holds num_kv_blocks counts plus one zero-count slot at index num_kv_blocks: its
        # exclusive prefix is base_g + total_g, the group end entry the former finalize kernel wrote there.
        row_entries = num_kv_blocks + 1
        tile_entries = self.num_scan_threads * self.scan_items_per_thread

        smem = SmemAllocator()
        warp_sums = smem.allocate_tensor(Int32, cute.make_layout(num_warps), byte_alignment=16)
        values = cute.make_rmem_tensor((self.scan_items_per_thread,), Int32)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # Group base = number of edges in the previous groups of this (batch, head). Their count rows are read
        # base_groups_per_round at a time so that 8 x base_groups_per_round independent loads per thread overlap.
        partial = Int32(0)
        for prev_group_base in cutlass.range(0, q_group_idx, self.base_groups_per_round, unroll=1):
            for tile_start in cutlass.range(0, num_kv_blocks, tile_entries, unroll=1):
                first_entry = tile_start + tidx * self.scan_items_per_thread
                for round_slot in cutlass.range_constexpr(self.base_groups_per_round):
                    prev_group_idx = prev_group_base + round_slot
                    for item in cutlass.range_constexpr(self.scan_items_per_thread):
                        if prev_group_idx < q_group_idx and first_entry + item < num_kv_blocks:
                            partial += mCounts[batch_idx, head_idx, prev_group_idx, first_entry + item]
        partial_inclusive = warp_prefix_sum(partial, lane)
        if lane == 31:
            warp_sums[warp_idx] = partial_inclusive
        cute.arch.sync_threads()
        carry = Int32(0)
        for other_warp in cutlass.range_constexpr(num_warps):
            carry += warp_sums[other_warp]
        cute.arch.sync_threads()

        # Exclusive scan of this group's counts (tiles of tile_entries with a running carry; one tile in practice).
        for tile_start in cutlass.range(0, row_entries, tile_entries, unroll=1):
            first_entry = tile_start + tidx * self.scan_items_per_thread
            thread_sum = Int32(0)
            for item in cutlass.range_constexpr(self.scan_items_per_thread):
                value = Int32(0)
                if first_entry + item < num_kv_blocks:
                    value = mCounts[batch_idx, head_idx, q_group_idx, first_entry + item]
                values[item] = value
                thread_sum += value
            # Block-wide exclusive scan of the per-thread sums: warp scan, then warp totals through smem.
            inclusive = warp_prefix_sum(thread_sum, lane)
            if lane == 31:
                warp_sums[warp_idx] = inclusive
            cute.arch.sync_threads()
            warp_exclusive = Int32(0)
            tile_total = Int32(0)
            for other_warp in cutlass.range_constexpr(num_warps):
                other_total = warp_sums[other_warp]
                tile_total += other_total
                if other_warp < warp_idx:
                    warp_exclusive += other_total
            running = carry + warp_exclusive + inclusive - thread_sum
            for item in cutlass.range_constexpr(self.scan_items_per_thread):
                entry = first_entry + item
                if entry < row_entries:
                    mBucketedK2qOffsets[batch_idx, head_idx, q_group_idx, entry] = running
                if entry < num_kv_blocks:
                    mCursors[batch_idx, head_idx, q_group_idx, entry] = running
                running += values[item]
            carry += tile_total
            # warp_sums is rewritten by the next tile.
            cute.arch.sync_threads()

    @cute.kernel
    def _scatter_q_indices_kernel(
        self,
        mCursors: cute.Tensor,
        mBucketedK2qIndices: cute.Tensor,
        mQ2kBlockIndex: cute.Tensor,
        mQ2kBlockNums: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_block_idx, head_idx, batch_idx = cute.arch.block_idx()
        num_kv_blocks = mCursors.shape[3]
        q_group_idx = q_block_idx // self.bucket_size_blocks
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        num_active_blocks = Int32(self.block_sparse_num)
        if const_expr(self.has_variable_block_nums):
            num_active_blocks = cute.min(mQ2kBlockNums[batch_idx, head_idx, q_block_idx], self.edge_width)
        for edge in cutlass.range(tidx, num_active_blocks, self.num_edge_threads):
            kv_block_idx = mQ2kBlockIndex[batch_idx, head_idx, q_block_idx, edge]
            if kv_block_idx >= 0 and kv_block_idx < num_kv_blocks:
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
    parallel_offsets: bool = False,
) -> tuple:
    """Return only the configuration that changes generated device code (``parallel_offsets`` no longer does)."""
    edge_width = max_kv_blocks if has_variable_block_nums else block_sparse_num
    return (
        device_capability,
        int(bucket_size_blocks),
        bool(has_variable_block_nums),
        int(edge_width),
    )


def _to_cute_tensor(tensor: "torch.Tensor") -> cute.Tensor:
    return from_dlpack(
        tensor.detach(),
        assumed_align=4,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def build_bucketed_k2q_csr_cutedsl(
    q2k_block_index: "torch.Tensor",
    block_sparse_num: int,
    num_kv_blocks: int,
    *,
    bucket_size_blocks: int,
    q2k_block_nums: Optional["torch.Tensor"] = None,
    parallel_offsets: bool = False,
) -> Tuple["torch.Tensor", "torch.Tensor", int, int]:
    """Build bucketed K-to-Q CSR metadata with CuTe DSL kernels."""
    import torch

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
    bucketed_k2q_offsets = torch.empty(
        (batch_size, num_heads, num_q_groups, num_kv_blocks + 1),
        dtype=torch.int32,
        device=device,
    )
    cursors = torch.empty_like(counts)
    bucketed_k2q_indices = torch.empty(
        (batch_size, num_heads, max_edges),
        dtype=torch.int32,
        device=device,
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream(q2k_block_index.device.index).cuda_stream)
    tensors = (
        counts,
        counts,  # placeholder for the unused mLocalOffsets scratch (see BucketedK2QCsrUniversal.__call__)
        counts,  # placeholder for the unused mGroupTotals scratch
        bucketed_k2q_offsets,
        cursors,
        bucketed_k2q_indices,
        q2k_block_index,
        q2k_block_nums,
    )
    device_capability = _device_capability(q2k_block_index.device.index)
    compile_key = _bucketed_k2q_csr_compile_key(
        device_capability,
        block_sparse_num,
        bucket_size_blocks,
        has_variable_block_nums,
        max_kv_blocks,
        parallel_offsets,
    )
    if compile_key not in build_bucketed_k2q_csr_cutedsl.compile_cache:
        kernel = BucketedK2QCsrUniversal(
            int(block_sparse_num),
            int(bucket_size_blocks),
            has_variable_block_nums,
            max_kv_blocks,
            parallel_offsets,
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
