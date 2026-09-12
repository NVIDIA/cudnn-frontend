# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL builder for private HSTU arbitrary-mask block metadata."""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Int8, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from .block_sparsity import (
    HSTUD256BwdBlockSparseBuilderWorkspace,
    HSTUBlockSparseBuilderWorkspace,
    HSTUBlockSparseTensorsTorch,
    HSTUK2QBlockSparseBuilderWorkspace,
)
from . import utils

_SCAN_THREADS = 256
_SCAN_LEVEL_COUNT = 4


def _scan_workspace_numel(num_rows: int) -> int:
    """Return packed storage for every level of the device scan hierarchy.

    Level 0 lives in the public offset tensor.  The workspace stores block sums
    for four successive 256-way reductions.  With the int32 edge-capacity
    contract this is enough for every possible row count:

        ceil((2**31 - 1) / 256**3) <= 128.
    """

    level_size = max(int(num_rows), 1)
    workspace_numel = 0
    for _ in range(_SCAN_LEVEL_COUNT):
        level_size = max((level_size + _SCAN_THREADS - 1) // _SCAN_THREADS, 1)
        workspace_numel += level_size
    return workspace_numel


class HSTUQ2KBlockSparseBuilder:
    """Build compact Q-to-K mask/full CSR tensors entirely on the device."""

    def __init__(self, tile_m: int, tile_n: int, func_num: int):
        if tile_m not in (128, 256):
            raise ValueError(f"unsupported HSTU Q tile size: {tile_m}")
        if tile_n != 128:
            raise ValueError(f"unsupported HSTU K tile size: {tile_n}")
        if func_num <= 0 or func_num % 2 == 0:
            raise ValueError("func_num must be positive and odd")
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.func_num = func_num
        self.num_intervals = (func_num + 1) // 2
        self.num_threads = tile_m
        self.num_warps = tile_m // cute.arch.WARP_SIZE
        self.scatter_threads = 128
        self.scan_rows_per_block = _SCAN_THREADS
        self.scan_threads = _SCAN_THREADS
        self.scan_log2_threads = 8
        self.scan_add_threads = _SCAN_THREADS

    @cute.jit
    def __call__(
        self,
        mMaskCnt: cute.Tensor,
        mMaskOffset: cute.Tensor,
        mMaskIdx: cute.Tensor,
        mFullCnt: cute.Tensor,
        mFullOffset: cute.Tensor,
        mFullIdx: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFunc: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tensors = (
            mMaskCnt,
            mMaskOffset,
            mMaskIdx,
            mFullCnt,
            mFullOffset,
            mFullIdx,
            mMaskStaging,
            mFullStaging,
            mMaskScanBlocks,
            mFullScanBlocks,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        )
        if const_expr(any(tensor.element_type != Int32 for tensor in tensors)):
            raise TypeError("HSTU block-sparse builder tensors must use int32")

        batch_size = mMaskCnt.shape[0]
        num_m_blocks = mMaskCnt.shape[2]
        num_rows = batch_size * num_m_blocks

        self._classify_q2k_kernel(
            mMaskCnt,
            mFullCnt,
            mMaskStaging,
            mFullStaging,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        ).launch(
            # Keep the batch coordinate out of grid.y: CUDA limits grid.y to
            # 65535, while a packed workload may legally contain more
            # sequences than that.  grid.x supports the full flattened row
            # domain used by the capacity CSR.
            grid=(num_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

        self._scan_counts_to_offsets(
            mMaskCnt,
            mMaskOffset,
            mFullCnt,
            mFullOffset,
            mMaskScanBlocks,
            mFullScanBlocks,
            num_rows,
            stream,
        )

        self._scatter_compact_kernel(
            mMaskCnt,
            mMaskOffset,
            mMaskIdx,
            mFullCnt,
            mFullOffset,
            mFullIdx,
            mMaskStaging,
            mFullStaging,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.scatter_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _scan_counts_to_offsets(
        self,
        mMaskCnt: cute.Tensor,
        mMaskOffset: cute.Tensor,
        mFullCnt: cute.Tensor,
        mFullOffset: cute.Tensor,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        num_rows: Int32,
        stream: cuda.CUstream,
    ):
        """Run a fixed-depth, 256-way hierarchical exclusive scan.

        The public offsets are level 0.  Four packed workspace segments hold
        successively reduced block sums.  Three scan levels are sufficient to
        make level 3 globally scanned for every row count permitted by the
        int32 capacity contract; level 4 only receives the final total.
        """

        level_1_count = cute.ceil_div(num_rows, self.scan_rows_per_block)
        level_2_count = cute.ceil_div(level_1_count, self.scan_rows_per_block)
        level_3_count = cute.ceil_div(level_2_count, self.scan_rows_per_block)
        level_4_count = cute.ceil_div(level_3_count, self.scan_rows_per_block)
        level_1_base = Int32(0)
        level_2_base = level_1_base + level_1_count
        level_3_base = level_2_base + level_2_count
        level_4_base = level_3_base + level_3_count

        self._exclusive_offsets_blocks_kernel(
            mMaskCnt,
            mMaskOffset,
            mFullCnt,
            mFullOffset,
            mMaskScanBlocks,
            mFullScanBlocks,
        ).launch(
            grid=(level_1_count, 1, 1),
            block=(self.scan_threads, 1, 1),
            stream=stream,
        )

        self._exclusive_scan_workspace_level_kernel(
            mMaskScanBlocks,
            mFullScanBlocks,
            level_1_base,
            level_1_count,
            level_2_base,
        ).launch(
            grid=(level_2_count, 1, 1),
            block=(self.scan_threads, 1, 1),
            stream=stream,
        )
        self._exclusive_scan_workspace_level_kernel(
            mMaskScanBlocks,
            mFullScanBlocks,
            level_2_base,
            level_2_count,
            level_3_base,
        ).launch(
            grid=(level_3_count, 1, 1),
            block=(self.scan_threads, 1, 1),
            stream=stream,
        )
        self._exclusive_scan_workspace_level_kernel(
            mMaskScanBlocks,
            mFullScanBlocks,
            level_3_base,
            level_3_count,
            level_4_base,
        ).launch(
            grid=(level_4_count, 1, 1),
            block=(self.scan_threads, 1, 1),
            stream=stream,
        )

        # level_3_count is at most 128 under the int32 capacity contract, so
        # its preceding one-CTA scan is already global.  Propagate those
        # prefixes down the packed hierarchy in launch order.
        self._add_scan_hierarchy_prefixes_kernel(
            mMaskScanBlocks,
            mFullScanBlocks,
            level_2_base,
            level_2_count,
            level_3_base,
        ).launch(
            grid=(cute.ceil_div(level_2_count, self.scan_add_threads), 1, 1),
            block=(self.scan_add_threads, 1, 1),
            stream=stream,
        )
        self._add_scan_hierarchy_prefixes_kernel(
            mMaskScanBlocks,
            mFullScanBlocks,
            level_1_base,
            level_1_count,
            level_2_base,
        ).launch(
            grid=(cute.ceil_div(level_1_count, self.scan_add_threads), 1, 1),
            block=(self.scan_add_threads, 1, 1),
            stream=stream,
        )

        self._add_scan_block_prefixes_kernel(
            mMaskOffset,
            mFullOffset,
            mMaskScanBlocks,
            mFullScanBlocks,
            num_rows,
            level_1_count,
        ).launch(
            grid=(
                cute.ceil_div(num_rows + 1, self.scan_add_threads),
                1,
                1,
            ),
            block=(self.scan_add_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _classify_q2k_kernel(
        self,
        mMaskCnt: cute.Tensor,
        mFullCnt: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFunc: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        flat_row, _, _ = cute.arch.block_idx()
        num_m_blocks = mMaskCnt.shape[2]
        batch_idx = flat_row // num_m_blocks
        m_block = flat_row - batch_idx * num_m_blocks

        @cute.struct
        class SharedStorage:
            reduction: cute.struct.Align[
                cute.struct.MemRange[Int8, self.num_warps * 2],
                16,
            ]
            candidate_bounds: cute.struct.Align[
                cute.struct.MemRange[Int32, self.num_warps * 2],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        reduction = storage.reduction.get_tensor(cute.make_layout((self.num_warps, 2)))
        candidate_bounds = storage.candidate_bounds.get_tensor(cute.make_layout((self.num_warps, 2)))

        offset_q = mCuSeqlensQ[batch_idx]
        offset_k = mCuSeqlensK[batch_idx]
        seqlen_q = mCuSeqlensQ[batch_idx + 1] - offset_q
        seqlen_k = mCuSeqlensK[batch_idx + 1] - offset_k
        q_idx = m_block * self.tile_m + tidx
        q_in_bounds = Boolean(q_idx < seqlen_q)
        func_idx = offset_q + q_idx
        num_n_blocks = cute.ceil_div(seqlen_k, self.tile_n)
        interval_begins = cute.make_rmem_tensor(
            (self.num_intervals,),
            Int32,
        )
        interval_ends = cute.make_rmem_tensor(
            (self.num_intervals,),
            Int32,
        )
        for interval in cutlass.range_constexpr(
            self.num_intervals,
            unroll_full=True,
        ):
            interval_begins[interval] = Int32(0)
            interval_ends[interval] = Int32(0)
        if q_in_bounds:
            for interval in cutlass.range_constexpr(
                self.num_intervals,
                unroll_full=True,
            ):
                if const_expr(interval > 0):
                    interval_begins[interval] = mFunc[
                        0,
                        2 * interval - 1,
                        func_idx,
                    ]
                interval_ends[interval] = mFunc[
                    0,
                    2 * interval,
                    func_idx,
                ]

        thread_candidate_begin = cutlass.Int32.max
        thread_candidate_end = Int32(0)
        if q_in_bounds:
            for interval in cutlass.range_constexpr(
                self.num_intervals,
                unroll_full=True,
            ):
                interval_begin = interval_begins[interval]
                interval_end = interval_ends[interval]
                if interval_end > interval_begin and interval_end > Int32(0) and interval_begin < seqlen_k:
                    thread_candidate_begin = min(
                        thread_candidate_begin,
                        max(interval_begin, Int32(0)),
                    )
                    thread_candidate_end = max(
                        thread_candidate_end,
                        min(interval_end, seqlen_k),
                    )

        warp_candidate_begin = utils.warp_reduce(
            thread_candidate_begin,
            cutlass.min,
        )
        warp_candidate_end = utils.warp_reduce(
            thread_candidate_end,
            cutlass.max,
        )
        if lane_idx == 0:
            candidate_bounds[warp_idx, 0] = warp_candidate_begin
            candidate_bounds[warp_idx, 1] = warp_candidate_end
        cute.arch.sync_threads()

        if warp_idx == 0:
            lane_candidate_begin = cutlass.Int32.max
            lane_candidate_end = Int32(0)
            if lane_idx < self.num_warps:
                lane_candidate_begin = candidate_bounds[lane_idx, 0]
                lane_candidate_end = candidate_bounds[lane_idx, 1]
            block_candidate_begin = utils.warp_reduce(
                lane_candidate_begin,
                cutlass.min,
            )
            block_candidate_end = utils.warp_reduce(
                lane_candidate_end,
                cutlass.max,
            )
            if lane_idx == 0:
                candidate_bounds[0, 0] = block_candidate_begin
                candidate_bounds[0, 1] = block_candidate_end
        cute.arch.sync_threads()

        candidate_begin = Int32(0)
        candidate_end = candidate_bounds[0, 1]
        if candidate_end > Int32(0):
            candidate_begin = candidate_bounds[0, 0]
        first_n_block = candidate_begin // self.tile_n
        last_n_block = cute.ceil_div(candidate_end, self.tile_n)

        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)
        for n_block in cutlass.range(
            first_n_block,
            min(last_n_block, num_n_blocks),
            unroll=1,
        ):
            k_begin = n_block * self.tile_n
            k_end = min(k_begin + self.tile_n, seqlen_k)

            thread_has_allowed = Boolean(False)
            thread_is_full = Boolean(False)
            if q_in_bounds:
                covered_until = k_begin
                for interval in cutlass.range_constexpr(
                    self.num_intervals,
                    unroll_full=True,
                ):
                    interval_begin = interval_begins[interval]
                    interval_end = interval_ends[interval]
                    overlaps = Boolean(interval_begin < k_end and interval_end > k_begin)
                    thread_has_allowed |= overlaps
                    # Adjacent intervals are semantically gap-free (for
                    # example [0, 64) + [64, 128)), so accumulate their union
                    # instead of requiring one interval to cover the tile.
                    if interval_begin <= covered_until and interval_end > covered_until:
                        covered_until = interval_end
                thread_is_full = Boolean(covered_until >= k_end)

            thread_has_blocked = Boolean(q_in_bounds and not thread_is_full)
            warp_has_allowed = cute.arch.vote_any_sync(Boolean(q_in_bounds and thread_has_allowed))
            warp_has_blocked = cute.arch.vote_any_sync(thread_has_blocked)
            if lane_idx == 0:
                reduction[warp_idx, 0] = Int8(1) if warp_has_allowed else Int8(0)
                reduction[warp_idx, 1] = Int8(1) if warp_has_blocked else Int8(0)
            cute.arch.sync_threads()

            has_allowed = Boolean(False)
            has_blocked = Boolean(False)
            if warp_idx == 0:
                lane_has_allowed = Boolean(False)
                lane_has_blocked = Boolean(False)
                if lane_idx < self.num_warps:
                    lane_has_allowed = reduction[lane_idx, 0] != Int8(0)
                    lane_has_blocked = reduction[lane_idx, 1] != Int8(0)
                has_allowed = cute.arch.vote_any_sync(lane_has_allowed)
                has_blocked = cute.arch.vote_any_sync(lane_has_blocked)

            if tidx == 0:
                if has_allowed:
                    if has_blocked:
                        mMaskStaging[flat_row, num_mask_blocks] = n_block
                        num_mask_blocks += 1
                    else:
                        mFullStaging[flat_row, num_full_blocks] = n_block
                        num_full_blocks += 1
            # Prevent a fast warp from overwriting the reduction slots while
            # warp 0 is still consuming the previous K block's values.
            cute.arch.sync_threads()

        if tidx == 0:
            mMaskCnt[batch_idx, 0, m_block] = num_mask_blocks
            mFullCnt[batch_idx, 0, m_block] = num_full_blocks

    @cute.jit
    def _exclusive_scan_shared(
        self,
        scan_storage: cute.Tensor,
        tidx: Int32,
    ):
        """Blelloch-scan two int32 channels resident in shared memory."""

        cute.arch.sync_threads()
        for step in cutlass.range_constexpr(
            self.scan_log2_threads,
            unroll_full=True,
        ):
            stride = 1 << step
            tree_idx = (tidx + 1) * (2 * stride) - 1
            if tree_idx < self.scan_threads:
                scan_storage[tree_idx, 0] += scan_storage[tree_idx - stride, 0]
                scan_storage[tree_idx, 1] += scan_storage[tree_idx - stride, 1]
            cute.arch.sync_threads()

        if tidx == 0:
            scan_storage[self.scan_threads, 0] = scan_storage[self.scan_threads - 1, 0]
            scan_storage[self.scan_threads, 1] = scan_storage[self.scan_threads - 1, 1]
            scan_storage[self.scan_threads - 1, 0] = Int32(0)
            scan_storage[self.scan_threads - 1, 1] = Int32(0)
        cute.arch.sync_threads()

        for step in cutlass.range_constexpr(
            self.scan_log2_threads,
            unroll_full=True,
        ):
            stride = 1 << (self.scan_log2_threads - 1 - step)
            tree_idx = (tidx + 1) * (2 * stride) - 1
            if tree_idx < self.scan_threads:
                mask_prefix = scan_storage[tree_idx - stride, 0]
                full_prefix = scan_storage[tree_idx - stride, 1]
                scan_storage[tree_idx - stride, 0] = scan_storage[tree_idx, 0]
                scan_storage[tree_idx - stride, 1] = scan_storage[tree_idx, 1]
                scan_storage[tree_idx, 0] += mask_prefix
                scan_storage[tree_idx, 1] += full_prefix
            cute.arch.sync_threads()

    @cute.kernel
    def _exclusive_offsets_blocks_kernel(
        self,
        mMaskCnt: cute.Tensor,
        mMaskOffset: cute.Tensor,
        mFullCnt: cute.Tensor,
        mFullOffset: cute.Tensor,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        scan_block, _, _ = cute.arch.block_idx()
        num_m_blocks = mMaskCnt.shape[2]
        num_rows = mMaskCnt.shape[0] * num_m_blocks
        flat_row = scan_block * self.scan_rows_per_block + tidx

        @cute.struct
        class SharedStorage:
            scan: cute.struct.Align[
                cute.struct.MemRange[
                    Int32,
                    (self.scan_threads + 1) * 2,
                ],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        scan_storage = storage.scan.get_tensor(cute.make_layout((self.scan_threads + 1, 2)))

        mask_count = Int32(0)
        full_count = Int32(0)
        if flat_row < num_rows:
            batch_idx = flat_row // num_m_blocks
            m_block = flat_row - batch_idx * num_m_blocks
            mask_count = mMaskCnt[batch_idx, 0, m_block]
            full_count = mFullCnt[batch_idx, 0, m_block]
        scan_storage[tidx, 0] = mask_count
        scan_storage[tidx, 1] = full_count
        self._exclusive_scan_shared(scan_storage, tidx)

        if flat_row < num_rows:
            mMaskOffset[flat_row] = scan_storage[tidx, 0]
            mFullOffset[flat_row] = scan_storage[tidx, 1]
        if tidx == 0:
            mask_block_total = scan_storage[self.scan_threads, 0]
            full_block_total = scan_storage[self.scan_threads, 1]
            mMaskScanBlocks[scan_block] = mask_block_total
            mFullScanBlocks[scan_block] = full_block_total
            if scan_block + 1 == cute.ceil_div(num_rows, self.scan_rows_per_block):
                # The final global prefix is added to this local terminal value
                # after the hierarchy has been scanned.
                mMaskOffset[num_rows] = mask_block_total
                mFullOffset[num_rows] = full_block_total

    @cute.kernel
    def _exclusive_scan_workspace_level_kernel(
        self,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        input_base: Int32,
        input_count: Int32,
        output_base: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        scan_block, _, _ = cute.arch.block_idx()
        input_idx = scan_block * self.scan_rows_per_block + tidx

        @cute.struct
        class SharedStorage:
            scan: cute.struct.Align[
                cute.struct.MemRange[
                    Int32,
                    (self.scan_threads + 1) * 2,
                ],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        scan_storage = storage.scan.get_tensor(cute.make_layout((self.scan_threads + 1, 2)))

        mask_count = Int32(0)
        full_count = Int32(0)
        if input_idx < input_count:
            mask_count = mMaskScanBlocks[input_base + input_idx]
            full_count = mFullScanBlocks[input_base + input_idx]
        scan_storage[tidx, 0] = mask_count
        scan_storage[tidx, 1] = full_count
        self._exclusive_scan_shared(scan_storage, tidx)

        if input_idx < input_count:
            mMaskScanBlocks[input_base + input_idx] = scan_storage[tidx, 0]
            mFullScanBlocks[input_base + input_idx] = scan_storage[tidx, 1]
        if tidx == 0:
            mMaskScanBlocks[output_base + scan_block] = scan_storage[self.scan_threads, 0]
            mFullScanBlocks[output_base + scan_block] = scan_storage[self.scan_threads, 1]

    @cute.kernel
    def _add_scan_hierarchy_prefixes_kernel(
        self,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        child_base: Int32,
        child_count: Int32,
        parent_base: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        child_idx = block_idx * self.scan_add_threads + tidx
        if child_idx < child_count:
            parent_idx = child_idx // self.scan_rows_per_block
            mMaskScanBlocks[child_base + child_idx] += mMaskScanBlocks[parent_base + parent_idx]
            mFullScanBlocks[child_base + child_idx] += mFullScanBlocks[parent_base + parent_idx]

    @cute.kernel
    def _add_scan_block_prefixes_kernel(
        self,
        mMaskOffset: cute.Tensor,
        mFullOffset: cute.Tensor,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        num_rows: Int32,
        num_scan_blocks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        flat_offset = block_idx * self.scan_add_threads + tidx
        if flat_offset <= num_rows:
            scan_block = min(
                flat_offset // self.scan_rows_per_block,
                num_scan_blocks - 1,
            )
            mMaskOffset[flat_offset] += mMaskScanBlocks[scan_block]
            mFullOffset[flat_offset] += mFullScanBlocks[scan_block]

    @cute.kernel
    def _scatter_compact_kernel(
        self,
        mMaskCnt: cute.Tensor,
        mMaskOffset: cute.Tensor,
        mMaskIdx: cute.Tensor,
        mFullCnt: cute.Tensor,
        mFullOffset: cute.Tensor,
        mFullIdx: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        flat_row, _, _ = cute.arch.block_idx()
        num_m_blocks = mMaskCnt.shape[2]
        batch_idx = flat_row // num_m_blocks
        m_block = flat_row - batch_idx * num_m_blocks
        mask_count = mMaskCnt[batch_idx, 0, m_block]
        full_count = mFullCnt[batch_idx, 0, m_block]
        mask_offset = mMaskOffset[flat_row]
        full_offset = mFullOffset[flat_row]

        for entry in cutlass.range(
            tidx,
            mask_count,
            self.scatter_threads,
            unroll=1,
        ):
            mMaskIdx[mask_offset + entry] = mMaskStaging[flat_row, entry]
        for entry in cutlass.range(
            tidx,
            full_count,
            self.scatter_threads,
            unroll=1,
        ):
            mFullIdx[full_offset + entry] = mFullStaging[flat_row, entry]


class HSTUK2QBlockSparseBuilder(HSTUQ2KBlockSparseBuilder):
    """Build compact K-to-Q mask/full CSR tensors entirely on the device.

    Classification is performed once in Q-major order so each Q token's
    endpoint vector is loaded only for the K blocks in its candidate range.
    A second device kernel compacts the transposed state matrix into
    deterministic, strictly ascending local Q-block rows.
    """

    def __init__(self, tile_q: int, tile_k: int, func_num: int):
        if tile_q not in (128, 256) or tile_k != 128:
            raise ValueError("HSTU K2Q supports block_size=(128, 128) or (256, 128)")
        super().__init__(tile_q, tile_k, func_num)

    @cute.jit
    def __call__(
        self,
        mMaskCnt: cute.Tensor,
        mMaskOffset: cute.Tensor,
        mMaskIdx: cute.Tensor,
        mFullCnt: cute.Tensor,
        mFullOffset: cute.Tensor,
        mFullIdx: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
        mMaskScanBlocks: cute.Tensor,
        mFullScanBlocks: cute.Tensor,
        mBlockStates: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFunc: cute.Tensor,
        stream: cuda.CUstream,
    ):
        int32_tensors = (
            mMaskCnt,
            mMaskOffset,
            mMaskIdx,
            mFullCnt,
            mFullOffset,
            mFullIdx,
            mMaskStaging,
            mFullStaging,
            mMaskScanBlocks,
            mFullScanBlocks,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        )
        if const_expr(any(tensor.element_type != Int32 for tensor in int32_tensors)):
            raise TypeError("HSTU block-sparse builder tensors must use int32")
        if const_expr(mBlockStates.element_type != Int8):
            raise TypeError("HSTU K2Q block states must use int8")

        batch_size = mMaskCnt.shape[0]
        num_k_blocks = mMaskCnt.shape[2]
        num_q_blocks = mBlockStates.shape[1]
        num_rows = batch_size * num_k_blocks

        self._classify_q2k_states_kernel(
            mBlockStates,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        ).launch(
            grid=(batch_size * num_q_blocks, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

        self._compact_k2q_rows_kernel(
            mMaskCnt,
            mFullCnt,
            mMaskStaging,
            mFullStaging,
            mBlockStates,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )

        self._scan_counts_to_offsets(
            mMaskCnt,
            mMaskOffset,
            mFullCnt,
            mFullOffset,
            mMaskScanBlocks,
            mFullScanBlocks,
            num_rows,
            stream,
        )

        self._scatter_compact_kernel(
            mMaskCnt,
            mMaskOffset,
            mMaskIdx,
            mFullCnt,
            mFullOffset,
            mFullIdx,
            mMaskStaging,
            mFullStaging,
        ).launch(
            grid=(num_rows, 1, 1),
            block=(self.scatter_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _classify_q2k_states_kernel(
        self,
        mBlockStates: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFunc: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        flat_row, _, _ = cute.arch.block_idx()
        num_q_blocks = mBlockStates.shape[1]
        batch_idx = flat_row // num_q_blocks
        q_block = flat_row - batch_idx * num_q_blocks

        @cute.struct
        class SharedStorage:
            reduction: cute.struct.Align[
                cute.struct.MemRange[Int8, self.num_warps * 2],
                16,
            ]
            candidate_bounds: cute.struct.Align[
                cute.struct.MemRange[Int32, self.num_warps * 2],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        reduction = storage.reduction.get_tensor(cute.make_layout((self.num_warps, 2)))
        candidate_bounds = storage.candidate_bounds.get_tensor(cute.make_layout((self.num_warps, 2)))

        # Every capacity edge is rewritten on every invocation.  This is
        # required when func is mutated in place between CUDA Graph replays.
        for k_block in cutlass.range(
            tidx,
            mBlockStates.shape[2],
            self.num_threads,
            unroll=1,
        ):
            mBlockStates[batch_idx, q_block, k_block] = Int8(0)
        cute.arch.sync_threads()

        offset_q = mCuSeqlensQ[batch_idx]
        offset_k = mCuSeqlensK[batch_idx]
        seqlen_q = mCuSeqlensQ[batch_idx + 1] - offset_q
        seqlen_k = mCuSeqlensK[batch_idx + 1] - offset_k
        q_idx = q_block * self.tile_m + tidx
        q_in_bounds = Boolean(q_idx < seqlen_q)
        func_idx = offset_q + q_idx
        num_k_blocks = cute.ceil_div(seqlen_k, self.tile_n)

        interval_begins = cute.make_rmem_tensor(
            (self.num_intervals,),
            Int32,
        )
        interval_ends = cute.make_rmem_tensor(
            (self.num_intervals,),
            Int32,
        )
        for interval in cutlass.range_constexpr(
            self.num_intervals,
            unroll_full=True,
        ):
            interval_begins[interval] = Int32(0)
            interval_ends[interval] = Int32(0)
        if q_in_bounds:
            for interval in cutlass.range_constexpr(
                self.num_intervals,
                unroll_full=True,
            ):
                if const_expr(interval > 0):
                    interval_begins[interval] = mFunc[
                        0,
                        2 * interval - 1,
                        func_idx,
                    ]
                interval_ends[interval] = mFunc[
                    0,
                    2 * interval,
                    func_idx,
                ]

        # The Q-row endpoint prepass restricts expensive token-wise
        # classification to K blocks that intersect the row's allowed union.
        thread_candidate_begin = cutlass.Int32.max
        thread_candidate_end = Int32(0)
        if q_in_bounds:
            for interval in cutlass.range_constexpr(
                self.num_intervals,
                unroll_full=True,
            ):
                interval_begin = interval_begins[interval]
                interval_end = interval_ends[interval]
                if interval_end > interval_begin and interval_end > Int32(0) and interval_begin < seqlen_k:
                    thread_candidate_begin = min(
                        thread_candidate_begin,
                        max(interval_begin, Int32(0)),
                    )
                    thread_candidate_end = max(
                        thread_candidate_end,
                        min(interval_end, seqlen_k),
                    )

        warp_candidate_begin = utils.warp_reduce(
            thread_candidate_begin,
            cutlass.min,
        )
        warp_candidate_end = utils.warp_reduce(
            thread_candidate_end,
            cutlass.max,
        )
        if lane_idx == 0:
            candidate_bounds[warp_idx, 0] = warp_candidate_begin
            candidate_bounds[warp_idx, 1] = warp_candidate_end
        cute.arch.sync_threads()

        if warp_idx == 0:
            lane_candidate_begin = cutlass.Int32.max
            lane_candidate_end = Int32(0)
            if lane_idx < self.num_warps:
                lane_candidate_begin = candidate_bounds[lane_idx, 0]
                lane_candidate_end = candidate_bounds[lane_idx, 1]
            block_candidate_begin = utils.warp_reduce(
                lane_candidate_begin,
                cutlass.min,
            )
            block_candidate_end = utils.warp_reduce(
                lane_candidate_end,
                cutlass.max,
            )
            if lane_idx == 0:
                candidate_bounds[0, 0] = block_candidate_begin
                candidate_bounds[0, 1] = block_candidate_end
        cute.arch.sync_threads()

        candidate_begin = Int32(0)
        candidate_end = candidate_bounds[0, 1]
        if candidate_end > Int32(0):
            candidate_begin = candidate_bounds[0, 0]
        first_k_block = candidate_begin // self.tile_n
        last_k_block = cute.ceil_div(candidate_end, self.tile_n)

        for k_block in cutlass.range(
            first_k_block,
            min(last_k_block, num_k_blocks),
            unroll=1,
        ):
            k_begin = k_block * self.tile_n
            k_end = min(k_begin + self.tile_n, seqlen_k)

            thread_has_allowed = Boolean(False)
            thread_is_full = Boolean(False)
            if q_in_bounds:
                covered_until = k_begin
                for interval in cutlass.range_constexpr(
                    self.num_intervals,
                    unroll_full=True,
                ):
                    interval_begin = interval_begins[interval]
                    interval_end = interval_ends[interval]
                    overlaps = Boolean(interval_begin < k_end and interval_end > k_begin)
                    thread_has_allowed |= overlaps
                    if interval_begin <= covered_until and interval_end > covered_until:
                        covered_until = interval_end
                thread_is_full = Boolean(covered_until >= k_end)

            thread_has_blocked = Boolean(q_in_bounds and not thread_is_full)
            warp_has_allowed = cute.arch.vote_any_sync(Boolean(q_in_bounds and thread_has_allowed))
            warp_has_blocked = cute.arch.vote_any_sync(thread_has_blocked)
            if lane_idx == 0:
                reduction[warp_idx, 0] = Int8(1) if warp_has_allowed else Int8(0)
                reduction[warp_idx, 1] = Int8(1) if warp_has_blocked else Int8(0)
            cute.arch.sync_threads()

            has_allowed = Boolean(False)
            has_blocked = Boolean(False)
            if warp_idx == 0:
                lane_has_allowed = Boolean(False)
                lane_has_blocked = Boolean(False)
                if lane_idx < self.num_warps:
                    lane_has_allowed = reduction[lane_idx, 0] != Int8(0)
                    lane_has_blocked = reduction[lane_idx, 1] != Int8(0)
                has_allowed = cute.arch.vote_any_sync(lane_has_allowed)
                has_blocked = cute.arch.vote_any_sync(lane_has_blocked)

            if tidx == 0 and has_allowed:
                mBlockStates[batch_idx, q_block, k_block] = Int8(1) if has_blocked else Int8(2)
            cute.arch.sync_threads()

    @cute.kernel
    def _compact_k2q_rows_kernel(
        self,
        mMaskCnt: cute.Tensor,
        mFullCnt: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
        mBlockStates: cute.Tensor,
    ):
        num_k_blocks = mMaskCnt.shape[2]
        flat_row, _, _ = cute.arch.block_idx()
        batch_idx = flat_row // num_k_blocks
        k_block = flat_row - batch_idx * num_k_blocks
        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)

        # Ascending traversal makes both staging rows deterministic and
        # satisfies the CSR contract without atomics or a host-side sort.
        for q_block in cutlass.range(
            mBlockStates.shape[1],
            unroll=1,
        ):
            state = mBlockStates[batch_idx, q_block, k_block]
            if state == Int8(1):
                mMaskStaging[flat_row, num_mask_blocks] = q_block
                num_mask_blocks += 1
            elif state == Int8(2):
                mFullStaging[flat_row, num_full_blocks] = q_block
                num_full_blocks += 1

        mMaskCnt[batch_idx, 0, k_block] = num_mask_blocks
        mFullCnt[batch_idx, 0, k_block] = num_full_blocks

    @cute.kernel
    def _compact_q2k_rows_kernel(
        self,
        mMaskCnt: cute.Tensor,
        mFullCnt: cute.Tensor,
        mMaskStaging: cute.Tensor,
        mFullStaging: cute.Tensor,
        mBlockStates: cute.Tensor,
    ):
        num_q_blocks = mMaskCnt.shape[2]
        flat_row, _, _ = cute.arch.block_idx()
        batch_idx = flat_row // num_q_blocks
        q_block = flat_row - batch_idx * num_q_blocks
        num_mask_blocks = Int32(0)
        num_full_blocks = Int32(0)

        # The state matrix is Q-major, so this is also a stable ascending
        # traversal.  Keeping both orientations ordered makes the D256
        # cluster consumers deterministic without atomics or sorting.
        for k_block in cutlass.range(
            mBlockStates.shape[2],
            unroll=1,
        ):
            state = mBlockStates[batch_idx, q_block, k_block]
            if state == Int8(1):
                mMaskStaging[flat_row, num_mask_blocks] = k_block
                num_mask_blocks += 1
            elif state == Int8(2):
                mFullStaging[flat_row, num_full_blocks] = k_block
                num_full_blocks += 1

        mMaskCnt[batch_idx, 0, q_block] = num_mask_blocks
        mFullCnt[batch_idx, 0, q_block] = num_full_blocks


class HSTUD256BwdBlockSparseBuilder(HSTUK2QBlockSparseBuilder):
    """Build D256 backward Q2K and K2Q CSR from one coarse classification.

    The shared state matrix is indexed by ``[batch, q256, k128]``.  This is
    the cluster work-unit contract used by both D256 backward kernels: dQ
    consumes rows directly, while dK/dV consumes the transposed K2Q rows and
    expands every Q256 index into two Q128 compute iterations.
    """

    def __init__(self, func_num: int):
        super().__init__(256, 128, func_num)

    @cute.jit
    def __call__(
        self,
        qMaskCnt: cute.Tensor,
        qMaskOffset: cute.Tensor,
        qMaskIdx: cute.Tensor,
        qFullCnt: cute.Tensor,
        qFullOffset: cute.Tensor,
        qFullIdx: cute.Tensor,
        qMaskStaging: cute.Tensor,
        qFullStaging: cute.Tensor,
        qMaskScanBlocks: cute.Tensor,
        qFullScanBlocks: cute.Tensor,
        kMaskCnt: cute.Tensor,
        kMaskOffset: cute.Tensor,
        kMaskIdx: cute.Tensor,
        kFullCnt: cute.Tensor,
        kFullOffset: cute.Tensor,
        kFullIdx: cute.Tensor,
        kMaskStaging: cute.Tensor,
        kFullStaging: cute.Tensor,
        kMaskScanBlocks: cute.Tensor,
        kFullScanBlocks: cute.Tensor,
        mBlockStates: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mFunc: cute.Tensor,
        stream: cuda.CUstream,
    ):
        int32_tensors = (
            qMaskCnt,
            qMaskOffset,
            qMaskIdx,
            qFullCnt,
            qFullOffset,
            qFullIdx,
            qMaskStaging,
            qFullStaging,
            qMaskScanBlocks,
            qFullScanBlocks,
            kMaskCnt,
            kMaskOffset,
            kMaskIdx,
            kFullCnt,
            kFullOffset,
            kFullIdx,
            kMaskStaging,
            kFullStaging,
            kMaskScanBlocks,
            kFullScanBlocks,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        )
        if const_expr(any(tensor.element_type != Int32 for tensor in int32_tensors)):
            raise TypeError("HSTU block-sparse builder tensors must use int32")
        if const_expr(mBlockStates.element_type != Int8):
            raise TypeError("HSTU D256 block states must use int8")

        batch_size = qMaskCnt.shape[0]
        num_q_blocks = qMaskCnt.shape[2]
        num_k_blocks = kMaskCnt.shape[2]
        num_q_rows = batch_size * num_q_blocks
        num_k_rows = batch_size * num_k_blocks

        # This is the only func_tensor classification in the paired builder.
        # Every capacity state is overwritten, which is essential when func
        # is mutated in place between CUDA Graph replays.
        self._classify_q2k_states_kernel(
            mBlockStates,
            mCuSeqlensQ,
            mCuSeqlensK,
            mFunc,
        ).launch(
            grid=(num_q_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

        self._compact_q2k_rows_kernel(
            qMaskCnt,
            qFullCnt,
            qMaskStaging,
            qFullStaging,
            mBlockStates,
        ).launch(
            grid=(num_q_rows, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )
        self._compact_k2q_rows_kernel(
            kMaskCnt,
            kFullCnt,
            kMaskStaging,
            kFullStaging,
            mBlockStates,
        ).launch(
            grid=(num_k_rows, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )

        self._scan_counts_to_offsets(
            qMaskCnt,
            qMaskOffset,
            qFullCnt,
            qFullOffset,
            qMaskScanBlocks,
            qFullScanBlocks,
            num_q_rows,
            stream,
        )

        self._scan_counts_to_offsets(
            kMaskCnt,
            kMaskOffset,
            kFullCnt,
            kFullOffset,
            kMaskScanBlocks,
            kFullScanBlocks,
            num_k_rows,
            stream,
        )

        self._scatter_compact_kernel(
            qMaskCnt,
            qMaskOffset,
            qMaskIdx,
            qFullCnt,
            qFullOffset,
            qFullIdx,
            qMaskStaging,
            qFullStaging,
        ).launch(
            grid=(num_q_rows, 1, 1),
            block=(self.scatter_threads, 1, 1),
            stream=stream,
        )
        self._scatter_compact_kernel(
            kMaskCnt,
            kMaskOffset,
            kMaskIdx,
            kFullCnt,
            kFullOffset,
            kFullIdx,
            kMaskStaging,
            kFullStaging,
        ).launch(
            grid=(num_k_rows, 1, 1),
            block=(self.scatter_threads, 1, 1),
            stream=stream,
        )


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(
        tensor.detach(),
        assumed_align=16,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def allocate_hstu_q2k_workspace(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int],
    device: torch.device,
) -> HSTUBlockSparseBuilderWorkspace:
    """Allocate capacity-backed Q2K CSR storage without reading device data."""

    tile_m, tile_n = block_size
    num_m_blocks = (int(max_seqlen_q) + tile_m - 1) // tile_m
    num_n_blocks = (int(max_seqlen_k) + tile_n - 1) // tile_n
    num_rows = int(batch_size) * num_m_blocks
    capacity = num_rows * num_n_blocks
    if capacity > torch.iinfo(torch.int32).max:
        raise ValueError("HSTU Q2K metadata capacity exceeds int32: " f"{capacity} block edges")

    count_shape = (int(batch_size), 1, num_m_blocks)
    counts = [torch.empty(count_shape, dtype=torch.int32, device=device) for _ in range(2)]
    offsets = [torch.empty(num_rows + 1, dtype=torch.int32, device=device) for _ in range(2)]
    # max_seqlen_{q,k} are positive in the public API, so capacity is nonzero.
    indices = [torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(2)]
    staging = [
        torch.empty(
            (num_rows, num_n_blocks),
            dtype=torch.int32,
            device=device,
        )
        for _ in range(2)
    ]
    tensors = HSTUBlockSparseTensorsTorch(
        mask_block_cnt=counts[0],
        mask_block_offset=offsets[0],
        mask_block_idx=indices[0],
        full_block_cnt=counts[1],
        full_block_offset=offsets[1],
        full_block_idx=indices[1],
        block_size=(tile_m, tile_n),
        orientation="q2k",
    )
    return HSTUBlockSparseBuilderWorkspace(
        tensors=tensors,
        mask_staging=staging[0],
        full_staging=staging[1],
        mask_scan_blocks=torch.empty(
            _scan_workspace_numel(num_rows),
            dtype=torch.int32,
            device=device,
        ),
        full_scan_blocks=torch.empty(
            _scan_workspace_numel(num_rows),
            dtype=torch.int32,
            device=device,
        ),
    )


def allocate_hstu_k2q_workspace(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int],
    device: torch.device,
) -> HSTUK2QBlockSparseBuilderWorkspace:
    """Allocate capacity-backed K2Q CSR storage without device value reads."""

    tile_q, tile_k = map(int, block_size)
    if tile_q not in (128, 256) or tile_k != 128:
        raise ValueError("HSTU K2Q supports block_size=(128, 128) or (256, 128)")
    num_q_blocks = (int(max_seqlen_q) + tile_q - 1) // tile_q
    num_k_blocks = (int(max_seqlen_k) + tile_k - 1) // tile_k
    num_rows = int(batch_size) * num_k_blocks
    capacity = num_rows * num_q_blocks
    if capacity > torch.iinfo(torch.int32).max:
        raise ValueError("HSTU K2Q metadata capacity exceeds int32: " f"{capacity} block edges")

    count_shape = (int(batch_size), 1, num_k_blocks)
    counts = [torch.empty(count_shape, dtype=torch.int32, device=device) for _ in range(2)]
    offsets = [torch.empty(num_rows + 1, dtype=torch.int32, device=device) for _ in range(2)]
    indices = [torch.empty(capacity, dtype=torch.int32, device=device) for _ in range(2)]
    staging = [
        torch.empty(
            (num_rows, num_q_blocks),
            dtype=torch.int32,
            device=device,
        )
        for _ in range(2)
    ]
    tensors = HSTUBlockSparseTensorsTorch(
        mask_block_cnt=counts[0],
        mask_block_offset=offsets[0],
        mask_block_idx=indices[0],
        full_block_cnt=counts[1],
        full_block_offset=offsets[1],
        full_block_idx=indices[1],
        block_size=(tile_q, tile_k),
        orientation="k2q",
    )
    return HSTUK2QBlockSparseBuilderWorkspace(
        tensors=tensors,
        mask_staging=staging[0],
        full_staging=staging[1],
        mask_scan_blocks=torch.empty(
            _scan_workspace_numel(num_rows),
            dtype=torch.int32,
            device=device,
        ),
        full_scan_blocks=torch.empty(
            _scan_workspace_numel(num_rows),
            dtype=torch.int32,
            device=device,
        ),
        block_states=torch.empty(
            (int(batch_size), num_q_blocks, num_k_blocks),
            dtype=torch.int8,
            device=device,
        ),
    )


def allocate_hstu_d256_bwd_workspace(
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    device: torch.device,
) -> HSTUD256BwdBlockSparseBuilderWorkspace:
    """Allocate paired Q256-by-K128 backward metadata workspaces."""

    block_size = (256, 128)
    return HSTUD256BwdBlockSparseBuilderWorkspace(
        q2k=allocate_hstu_q2k_workspace(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=block_size,
            device=device,
        ),
        k2q=allocate_hstu_k2q_workspace(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            block_size=block_size,
            device=device,
        ),
    )


def build_hstu_q2k_block_sparse(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int],
    compile_only: bool = False,
) -> HSTUBlockSparseTensorsTorch:
    """Build compact Q2K metadata on the caller's current CUDA stream."""

    if func.dtype != torch.int32 or func.ndim != 3:
        raise ValueError("func must be a rank-3 torch.int32 tensor")
    if not func.is_cuda:
        raise ValueError("func must be a CUDA tensor")
    if func.shape[1] <= 0 or func.shape[1] % 2 == 0:
        raise ValueError("func.shape[1] must be positive and odd")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32 or not cu_seqlens_q.is_cuda or not cu_seqlens_k.is_cuda:
        raise ValueError("cu_seqlens_q/k must be CUDA int32 tensors")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q/k must have matching shapes")
    if func.device != cu_seqlens_q.device or func.device != cu_seqlens_k.device:
        raise ValueError("func and cu_seqlens_q/k must be on the same device")

    batch_size = int(cu_seqlens_q.numel() - 1)
    workspace = allocate_hstu_q2k_workspace(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size,
        device=func.device,
    )
    tensors = workspace.tensors
    runtime_tensors = (
        tensors.mask_block_cnt,
        tensors.mask_block_offset,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_offset,
        tensors.full_block_idx,
        workspace.mask_staging,
        workspace.full_staging,
        workspace.mask_scan_blocks,
        workspace.full_scan_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        func,
    )

    device_capability = torch.cuda.get_device_capability(func.device)
    compile_key = (
        func.device,
        device_capability,
        tuple(map(int, block_size)),
        int(func.shape[1]),
    )
    if compile_key not in build_hstu_q2k_block_sparse.compile_cache:
        kernel = HSTUQ2KBlockSparseBuilder(
            int(block_size[0]),
            int(block_size[1]),
            int(func.shape[1]),
        )
        build_hstu_q2k_block_sparse.compile_cache[compile_key] = cute.compile(
            kernel,
            *(_to_cute_tensor(tensor) for tensor in runtime_tensors),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    if not compile_only:
        current_stream = cuda.CUstream(torch.cuda.current_stream(func.device).cuda_stream)
        build_hstu_q2k_block_sparse.compile_cache[compile_key](
            *runtime_tensors,
            current_stream,
        )
    return tensors


build_hstu_q2k_block_sparse.compile_cache = {}


def build_hstu_k2q_block_sparse(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size: tuple[int, int] = (128, 128),
    compile_only: bool = False,
) -> HSTUBlockSparseTensorsTorch:
    """Build compact K2Q metadata on the caller's current CUDA stream."""

    if func.dtype != torch.int32 or func.ndim != 3:
        raise ValueError("func must be a rank-3 torch.int32 tensor")
    if not func.is_cuda:
        raise ValueError("func must be a CUDA tensor")
    if func.shape[1] <= 0 or func.shape[1] % 2 == 0:
        raise ValueError("func.shape[1] must be positive and odd")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32 or not cu_seqlens_q.is_cuda or not cu_seqlens_k.is_cuda:
        raise ValueError("cu_seqlens_q/k must be CUDA int32 tensors")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q/k must have matching shapes")
    if func.device != cu_seqlens_q.device or func.device != cu_seqlens_k.device:
        raise ValueError("func and cu_seqlens_q/k must be on the same device")

    block_size = tuple(map(int, block_size))
    if block_size not in ((128, 128), (256, 128)):
        raise ValueError("HSTU K2Q supports block_size=(128, 128) or (256, 128)")
    batch_size = int(cu_seqlens_q.numel() - 1)
    workspace = allocate_hstu_k2q_workspace(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_size=block_size,
        device=func.device,
    )
    tensors = workspace.tensors
    runtime_tensors = (
        tensors.mask_block_cnt,
        tensors.mask_block_offset,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_offset,
        tensors.full_block_idx,
        workspace.mask_staging,
        workspace.full_staging,
        workspace.mask_scan_blocks,
        workspace.full_scan_blocks,
        workspace.block_states,
        cu_seqlens_q,
        cu_seqlens_k,
        func,
    )

    device_capability = torch.cuda.get_device_capability(func.device)
    compile_key = (
        func.device,
        device_capability,
        "k2q",
        block_size,
        int(func.shape[1]),
    )
    if compile_key not in build_hstu_k2q_block_sparse.compile_cache:
        kernel = HSTUK2QBlockSparseBuilder(
            block_size[0],
            block_size[1],
            int(func.shape[1]),
        )
        build_hstu_k2q_block_sparse.compile_cache[compile_key] = cute.compile(
            kernel,
            *(_to_cute_tensor(tensor) for tensor in runtime_tensors),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    if not compile_only:
        current_stream = cuda.CUstream(torch.cuda.current_stream(func.device).cuda_stream)
        build_hstu_k2q_block_sparse.compile_cache[compile_key](
            *runtime_tensors,
            current_stream,
        )
    return tensors


build_hstu_k2q_block_sparse.compile_cache = {}


def build_hstu_d256_bwd_block_sparse(
    func: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    compile_only: bool = False,
) -> Tuple[HSTUBlockSparseTensorsTorch, HSTUBlockSparseTensorsTorch]:
    """Build both D256 backward CSR orientations from one device classify.

    The returned tuple is ``(q2k, k2q)``.  Both orientations use the same
    Q256-by-K128 coarse states, so a supertile whose two Q128 subtiles have
    different states is conservatively emitted as MASK in both views.
    """

    if func.dtype != torch.int32 or func.ndim != 3:
        raise ValueError("func must be a rank-3 torch.int32 tensor")
    if not func.is_cuda:
        raise ValueError("func must be a CUDA tensor")
    if func.shape[1] <= 0 or func.shape[1] % 2 == 0:
        raise ValueError("func.shape[1] must be positive and odd")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32 or not cu_seqlens_q.is_cuda or not cu_seqlens_k.is_cuda:
        raise ValueError("cu_seqlens_q/k must be CUDA int32 tensors")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens_q/k must have matching shapes")
    if func.device != cu_seqlens_q.device or func.device != cu_seqlens_k.device:
        raise ValueError("func and cu_seqlens_q/k must be on the same device")

    batch_size = int(cu_seqlens_q.numel() - 1)
    workspace = allocate_hstu_d256_bwd_workspace(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        device=func.device,
    )
    q2k_workspace = workspace.q2k
    k2q_workspace = workspace.k2q
    q2k = q2k_workspace.tensors
    k2q = k2q_workspace.tensors
    runtime_tensors = (
        q2k.mask_block_cnt,
        q2k.mask_block_offset,
        q2k.mask_block_idx,
        q2k.full_block_cnt,
        q2k.full_block_offset,
        q2k.full_block_idx,
        q2k_workspace.mask_staging,
        q2k_workspace.full_staging,
        q2k_workspace.mask_scan_blocks,
        q2k_workspace.full_scan_blocks,
        k2q.mask_block_cnt,
        k2q.mask_block_offset,
        k2q.mask_block_idx,
        k2q.full_block_cnt,
        k2q.full_block_offset,
        k2q.full_block_idx,
        k2q_workspace.mask_staging,
        k2q_workspace.full_staging,
        k2q_workspace.mask_scan_blocks,
        k2q_workspace.full_scan_blocks,
        k2q_workspace.block_states,
        cu_seqlens_q,
        cu_seqlens_k,
        func,
    )

    device_capability = torch.cuda.get_device_capability(func.device)
    compile_key = (
        func.device,
        device_capability,
        "d256_bwd_paired",
        (256, 128),
        int(func.shape[1]),
    )
    if compile_key not in build_hstu_d256_bwd_block_sparse.compile_cache:
        kernel = HSTUD256BwdBlockSparseBuilder(int(func.shape[1]))
        build_hstu_d256_bwd_block_sparse.compile_cache[compile_key] = cute.compile(
            kernel,
            *(_to_cute_tensor(tensor) for tensor in runtime_tensors),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )

    if not compile_only:
        current_stream = cuda.CUstream(torch.cuda.current_stream(func.device).cuda_stream)
        build_hstu_d256_bwd_block_sparse.compile_cache[compile_key](
            *runtime_tensors,
            current_stream,
        )
    return q2k, k2q


build_hstu_d256_bwd_block_sparse.compile_cache = {}
