# SPDX-License-Identifier: BSD-3-Clause
"""Architecture-neutral forward work descriptor materialization."""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Boolean, Int32, Int64, const_expr

import cuda.bindings.driver as cuda

_L2_SECTION_BYTES = 50 * 1024 * 1024


class ForwardSchedulePlan:
    """Materialize exact forward work and locality keys once per mask plan."""

    def __init__(
        self,
        *,
        plan_tile_m: int,
        tile_n: int,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        is_varlen: bool,
    ) -> None:
        self.plan_tile_m = plan_tile_m
        self.tile_n = tile_n
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.is_varlen = is_varlen

    @cute.jit
    def __call__(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mSequenceDesc: Optional[cute.Tensor],
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        batch_size: Int32,
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        max_m_blocks: Int32,
        head_dim: Int32,
        head_dim_v: Int32,
        element_size: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mPartialCount,
            mFullCount,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            mSequenceDesc,
            mWorkDesc,
            mTaskCost,
            mSectionId,
            num_scheduled_heads,
            num_kv_heads,
            seqlen_q_fixed,
            seqlen_k_fixed,
            max_m_blocks,
            head_dim,
            head_dim_v,
            element_size,
        ).launch(
            grid=(cutlass.max(max_m_blocks, Int32(1)), num_scheduled_heads, batch_size),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        mSequenceDesc: Optional[cute.Tensor],
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        max_m_blocks: Int32,
        head_dim: Int32,
        head_dim_v: Int32,
        element_size: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        q_offset = batch_idx * seqlen_q_fixed
        k_offset = batch_idx * seqlen_k_fixed
        q_len = seqlen_q_fixed
        k_len = seqlen_k_fixed
        q_plan_row_begin = batch_idx * max_m_blocks
        if const_expr(self.is_varlen):
            assert mCuSeqlensQ is not None
            assert mCuSeqlensK is not None
            assert mCuTotalMBlocks is not None
            assert mSequenceDesc is not None
            q_offset = mCuSeqlensQ[batch_idx]
            k_offset = mCuSeqlensK[batch_idx]
            q_len = mCuSeqlensQ[batch_idx + Int32(1)] - q_offset
            k_len = mCuSeqlensK[batch_idx + Int32(1)] - k_offset
            q_plan_row_begin = mCuTotalMBlocks[batch_idx]

        physical_q_len = q_len
        if const_expr(self.pack_gqa):
            physical_q_len *= Int32(self.qhead_per_kvhead)
        q_plan_row_count = cute.ceil_div(physical_q_len, self.plan_tile_m)
        num_k_blocks = cute.ceil_div(k_len, self.tile_n)

        if const_expr(mSequenceDesc is not None):
            if tidx == Int32(0) and m_block == Int32(0) and head_idx == Int32(0):
                mSequenceDesc[batch_idx, Int32(0)] = q_offset
                mSequenceDesc[batch_idx, Int32(1)] = k_offset
                mSequenceDesc[batch_idx, Int32(2)] = q_len
                mSequenceDesc[batch_idx, Int32(3)] = k_len
                mSequenceDesc[batch_idx, Int32(4)] = q_plan_row_begin
                mSequenceDesc[batch_idx, Int32(5)] = q_plan_row_count
                mSequenceDesc[batch_idx, Int32(6)] = num_k_blocks
                mSequenceDesc[batch_idx, Int32(7)] = Int32(0)

        if tidx == Int32(0) and m_block < q_plan_row_count:
            outer_row = q_plan_row_begin + m_block
            task_idx = outer_row * num_scheduled_heads + head_idx
            q_valid_rows = cutlass.min(
                Int32(self.plan_tile_m),
                physical_q_len - m_block * Int32(self.plan_tile_m),
            )
            plan_head = Int32(0)
            if mPartialCount.shape[0] != 1:
                plan_head = head_idx
            task_cost = mPartialCount[plan_head, outer_row] + mFullCount[plan_head, outer_row]
            kv_head_idx = head_idx
            if const_expr(not self.pack_gqa):
                kv_head_idx = head_idx // Int32(self.qhead_per_kvhead)

            kv_head_bytes = Int64(k_len) * Int64(head_dim + head_dim_v) * Int64(element_size)
            heads_per_section = Int32(1)
            while heads_per_section * Int32(2) <= num_kv_heads and kv_head_bytes * Int64(heads_per_section * Int32(2)) <= Int64(_L2_SECTION_BYTES):
                heads_per_section *= Int32(2)
            section_id = batch_idx * num_kv_heads + kv_head_idx // heads_per_section

            mWorkDesc[task_idx, Int32(0)] = m_block
            mWorkDesc[task_idx, Int32(1)] = head_idx
            mWorkDesc[task_idx, Int32(2)] = batch_idx
            mWorkDesc[task_idx, Int32(3)] = q_valid_rows
            mTaskCost[task_idx] = task_cost
            mSectionId[task_idx] = section_id


class ForwardScheduleOrder:
    """Order forward work with an exact device-side counting sort."""

    def __init__(
        self,
        *,
        qhead_per_kvhead: int,
        pack_gqa: bool,
        is_varlen: bool,
    ) -> None:
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.is_varlen = is_varlen

    @cute.jit
    def __call__(
        self,
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        mHistogram: cute.Tensor,
        mSectionCost: cute.Tensor,
        mSectionOrder: cute.Tensor,
        mPositiveBase: cute.Tensor,
        mZeroBase: cute.Tensor,
        mSortedWorkDesc: cute.Tensor,
        mCuTotalMBlocks: Optional[cute.Tensor],
        num_tasks: Int32,
        batch_size: Int32,
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        max_m_blocks: Int32,
        max_task_cost: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.histogram_kernel(
            mTaskCost,
            mSectionId,
            mHistogram,
            mSectionCost,
            num_tasks,
            max_task_cost,
        ).launch(
            grid=(
                cutlass.max(
                    (num_tasks + Int32(255)) // Int32(256),
                    Int32(1),
                ),
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )
        self.prepare_kernel(
            mHistogram,
            mSectionCost,
            mSectionOrder,
            mPositiveBase,
            mZeroBase,
            batch_size,
            num_kv_heads,
            max_task_cost,
        ).launch(
            grid=(1, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )
        self.scatter_kernel(
            mWorkDesc,
            mTaskCost,
            mSectionId,
            mHistogram,
            mSortedWorkDesc,
            mCuTotalMBlocks,
            num_scheduled_heads,
            num_kv_heads,
            max_m_blocks,
        ).launch(
            grid=(cutlass.max(batch_size * num_kv_heads, Int32(1)), 1, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def histogram_kernel(
        self,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        mHistogram: cute.Tensor,
        mSectionCost: cute.Tensor,
        num_tasks: Int32,
        max_task_cost: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * Int32(256) + tidx
        if task_idx < num_tasks:
            task_cost = mTaskCost[task_idx]
            section_id = mSectionId[task_idx]
            histogram_idx = section_id * (max_task_cost + Int32(1)) + task_cost
            cute.arch.atomic_add(
                ptr=mHistogram.iterator + histogram_idx,
                val=Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            cute.arch.atomic_add(
                ptr=mSectionCost.iterator + section_id,
                val=Int64(task_cost),
                sem="relaxed",
                scope="gpu",
            )

    @cute.kernel
    def prepare_kernel(
        self,
        mHistogram: cute.Tensor,
        mSectionCost: cute.Tensor,
        mSectionOrder: cute.Tensor,
        mPositiveBase: cute.Tensor,
        mZeroBase: cute.Tensor,
        batch_size: Int32,
        num_kv_heads: Int32,
        max_task_cost: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        num_sections = batch_size * num_kv_heads

        section = tidx
        while section < num_sections:
            mSectionOrder[section] = section
            section += Int32(256)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            order_idx = Int32(1)
            while order_idx < num_sections:
                key = mSectionOrder[order_idx]
                key_cost = mSectionCost[key]
                insert_idx = order_idx
                inserting = Boolean(True)
                while insert_idx > Int32(0) and inserting:
                    previous = mSectionOrder[insert_idx - Int32(1)]
                    if mSectionCost[previous] < key_cost:
                        mSectionOrder[insert_idx] = previous
                        insert_idx -= Int32(1)
                    else:
                        inserting = Boolean(False)
                mSectionOrder[insert_idx] = key
                order_idx += Int32(1)

            positive_end = Int32(0)
            order_idx = Int32(0)
            while order_idx < num_sections:
                section_id = mSectionOrder[order_idx]
                mPositiveBase[section_id] = positive_end
                cost = max_task_cost
                while cost > Int32(0):
                    positive_end += mHistogram[section_id, cost]
                    cost -= Int32(1)
                order_idx += Int32(1)

            zero_end = positive_end
            section_id = Int32(0)
            while section_id < num_sections:
                mZeroBase[section_id] = zero_end
                zero_end += mHistogram[section_id, Int32(0)]
                section_id += Int32(1)
        cute.arch.sync_threads()

        section = tidx
        while section < num_sections:
            cursor = mPositiveBase[section]
            cost = max_task_cost
            while cost > Int32(0):
                count = mHistogram[section, cost]
                mHistogram[section, cost] = cursor
                cursor += count
                cost -= Int32(1)
            mHistogram[section, Int32(0)] = mZeroBase[section]
            section += Int32(256)

    @cute.kernel
    def scatter_kernel(
        self,
        mWorkDesc: cute.Tensor,
        mTaskCost: cute.Tensor,
        mSectionId: cute.Tensor,
        mHistogram: cute.Tensor,
        mSortedWorkDesc: cute.Tensor,
        mCuTotalMBlocks: Optional[cute.Tensor],
        num_scheduled_heads: Int32,
        num_kv_heads: Int32,
        max_m_blocks: Int32,
    ) -> None:
        lane_idx, _, _ = cute.arch.thread_idx()
        section, _, _ = cute.arch.block_idx()
        smem = cutlass_utils.SmemAllocator()
        sCost = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
        )
        sBase = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((32,)),
            byte_alignment=16,
        )

        batch_idx = section // num_kv_heads
        outer_begin = batch_idx * max_m_blocks
        outer_end = outer_begin + max_m_blocks
        if const_expr(self.is_varlen):
            assert mCuTotalMBlocks is not None
            outer_begin = mCuTotalMBlocks[batch_idx]
            outer_end = mCuTotalMBlocks[batch_idx + Int32(1)]

        if outer_begin < outer_end:
            kv_head_idx = Int32(0)
            while kv_head_idx < num_kv_heads:
                head_begin = kv_head_idx
                heads_per_kv = Int32(1)
                if const_expr(not self.pack_gqa):
                    head_begin *= Int32(self.qhead_per_kvhead)
                    heads_per_kv = Int32(self.qhead_per_kvhead)
                first_task = outer_begin * num_scheduled_heads + head_begin
                if mSectionId[first_task] == section:
                    tasks_for_kv = (outer_end - outer_begin) * heads_per_kv
                    chunk_begin = Int32(0)
                    while chunk_begin < tasks_for_kv:
                        task_offset = chunk_begin + lane_idx
                        valid = task_offset < tasks_for_kv
                        source_idx = Int32(0)
                        task_cost = Int32(-1)
                        if valid:
                            outer_row = outer_begin + task_offset // heads_per_kv
                            head_offset = task_offset - (outer_row - outer_begin) * heads_per_kv
                            source_idx = outer_row * num_scheduled_heads + head_begin + head_offset
                            task_cost = mTaskCost[source_idx]
                        sCost[lane_idx] = task_cost
                        cute.arch.sync_warp()

                        rank = Int32(0)
                        first_lane = Int32(0)
                        found_first = Boolean(False)
                        if valid:
                            for compare_lane in cutlass.range_constexpr(32):
                                if sCost[compare_lane] == task_cost:
                                    if Int32(compare_lane) < lane_idx:
                                        rank += Int32(1)
                                    if not found_first:
                                        first_lane = Int32(compare_lane)
                                        found_first = Boolean(True)
                            if rank == Int32(0):
                                group_size = Int32(0)
                                for compare_lane in cutlass.range_constexpr(32):
                                    if sCost[compare_lane] == task_cost:
                                        group_size += Int32(1)
                                group_base = mHistogram[section, task_cost]
                                mHistogram[section, task_cost] = group_base + group_size
                                sBase[lane_idx] = group_base
                        cute.arch.sync_warp()

                        if valid:
                            destination_idx = sBase[first_lane] + rank
                            for field in cutlass.range_constexpr(4):
                                mSortedWorkDesc[destination_idx, field] = mWorkDesc[source_idx, field]
                        cute.arch.sync_warp()
                        chunk_begin += Int32(32)
                kv_head_idx += Int32(1)


__all__ = ["ForwardScheduleOrder", "ForwardSchedulePlan"]
