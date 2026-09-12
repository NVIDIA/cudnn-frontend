# SPDX-License-Identifier: BSD-3-Clause
"""Architecture-neutral stable compaction for arbitrary-mask plan rows."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Boolean, Int32, Uint32, const_expr

from cudnn.flex_attention.plan.kernels.common import (
    _ArbitraryPlanCommonSm90,
    _ArbitraryPlanK2QCommonSm90,
    _shr_u32,
)


class _ArbitraryPlanQ2KCompact(_ArbitraryPlanCommonSm90):
    @cute.kernel
    def compact_kernel(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalMBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_m_blocks: Int32,
        max_n_blocks: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        upper_outer_row, mask_head, _ = cute.arch.block_idx()
        (
            batch_idx,
            local_m_block,
            compact_outer_row,
            _,
            _,
            _,
            _,
            valid_m_block,
        ) = self._sample_info(
            upper_outer_row,
            max_m_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
        )
        smem = cutlass_utils.SmemAllocator()
        sWarpPartial = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sWarpFull = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sRunning = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((2,)),
            byte_alignment=8,
        )
        if planner_tidx == Int32(0):
            sRunning[0] = Int32(0)
            sRunning[1] = Int32(0)
        cute.arch.sync_threads()

        total_m_blocks = (mPartialOffsets.shape[0] - Int32(1)) // mVisibleBits.shape[0]
        plan_row = mask_head * total_m_blocks + compact_outer_row
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        block_base = Int32(0)
        while block_base < max_n_blocks:
            block_id = block_base + planner_tidx
            visible = Boolean(False)
            full = Boolean(False)
            if valid_m_block and block_id < max_n_blocks:
                word_idx = block_id // Int32(32)
                bit_idx = block_id - word_idx * Int32(32)
                bit = Uint32(1) << Uint32(bit_idx)
                visible = (mVisibleBits[mask_head, compact_outer_row, word_idx] & bit) != Uint32(0)
                full = (mFullBits[mask_head, compact_outer_row, word_idx] & bit) != Uint32(0)
            partial = visible & ~full
            partial_ballot = Uint32(cute.arch.vote_ballot_sync(partial))
            full_ballot = Uint32(cute.arch.vote_ballot_sync(full))
            if lane_idx == Int32(0):
                sWarpPartial[warp_idx] = Int32(cute.arch.popc(partial_ballot))
                sWarpFull[warp_idx] = Int32(cute.arch.popc(full_ballot))
            cute.arch.sync_threads()

            partial_prefix = Int32(0)
            full_prefix = Int32(0)
            for prior_warp in cutlass.range_constexpr(self.num_warps):
                if Int32(prior_warp) < warp_idx:
                    partial_prefix += sWarpPartial[prior_warp]
                    full_prefix += sWarpFull[prior_warp]
            low_lanes = _shr_u32(
                Uint32(0xFFFF_FFFF),
                Uint32(Int32(32) - lane_idx),
            )
            if partial:
                partial_rank = sRunning[0] + partial_prefix + Int32(cute.arch.popc(partial_ballot & low_lanes))
                payload_idx = mPartialOffsets[plan_row] + partial_rank
                mPartialIndices[payload_idx] = block_id
                mPartialWorkDesc[payload_idx, 0] = mask_head
                mPartialWorkDesc[payload_idx, 1] = batch_idx
                mPartialWorkDesc[payload_idx, 2] = local_m_block
                mPartialWorkDesc[payload_idx, 3] = block_id
            if full:
                full_rank = sRunning[1] + full_prefix + Int32(cute.arch.popc(full_ballot & low_lanes))
                mFullIndices[mFullOffsets[plan_row] + full_rank] = block_id
            cute.arch.sync_threads()
            if planner_tidx == Int32(0):
                chunk_partial = Int32(0)
                chunk_full = Int32(0)
                for warp in cutlass.range_constexpr(self.num_warps):
                    chunk_partial += sWarpPartial[warp]
                    chunk_full += sWarpFull[warp]
                sRunning[0] += chunk_partial
                sRunning[1] += chunk_full
            cute.arch.sync_threads()
            block_base += Int32(self.num_threads)


class _ArbitraryPlanK2QCompact(_ArbitraryPlanK2QCommonSm90):
    @cute.kernel
    def compact_kernel(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mQPartialCounts: cute.Tensor,
        mQFullCounts: cute.Tensor,
        mPartialOffsets: cute.Tensor,
        mPartialIndices: cute.Tensor,
        mPartialWorkDesc: cute.Tensor,
        mPartialDQOrder: cute.Tensor,
        mFullOffsets: cute.Tensor,
        mFullIndices: cute.Tensor,
        mFullDQOrder: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuTotalQBlocks: Optional[cute.Tensor],
        mCuTotalKBlocks: Optional[cute.Tensor],
        batch_size: Int32,
        seqlen_q_fixed: Int32,
        seqlen_k_fixed: Int32,
        total_q: Int32,
        total_k: Int32,
        max_n_blocks: Int32,
    ):
        planner_tidx, _, _ = cute.arch.thread_idx()
        upper_n_row, mask_head, _ = cute.arch.block_idx()
        (
            batch_idx,
            local_n_block,
            compact_n_row,
            compact_q_begin,
            num_q_blocks,
            _,
            _,
            _,
            _,
            valid_n_block,
        ) = self._sample_info_k(
            upper_n_row,
            max_n_blocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
        )
        smem = cutlass_utils.SmemAllocator()
        sWarpPartial = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sWarpFull = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        sRunning = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((2,)),
            byte_alignment=8,
        )
        if planner_tidx == Int32(0):
            sRunning[0] = Int32(0)
            sRunning[1] = Int32(0)
        cute.arch.sync_threads()

        total_n_blocks = (mPartialOffsets.shape[0] - Int32(1)) // mVisibleBits.shape[0]
        plan_row = mask_head * total_n_blocks + compact_n_row
        word_idx = local_n_block // Int32(32)
        bit_idx = local_n_block - word_idx * Int32(32)
        bit = Uint32(1) << Uint32(bit_idx)
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        q_block_base = Int32(0)
        while q_block_base < num_q_blocks:
            local_q_block = q_block_base + planner_tidx
            partial = Boolean(False)
            full = Boolean(False)
            compact_q_row = compact_q_begin + local_q_block
            if valid_n_block and local_q_block < num_q_blocks:
                visible = (mVisibleBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                full = (mFullBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                partial = visible & ~full
            partial_ballot = Uint32(cute.arch.vote_ballot_sync(partial))
            full_ballot = Uint32(cute.arch.vote_ballot_sync(full))
            if lane_idx == Int32(0):
                sWarpPartial[warp_idx] = Int32(cute.arch.popc(partial_ballot))
                sWarpFull[warp_idx] = Int32(cute.arch.popc(full_ballot))
            cute.arch.sync_threads()

            partial_prefix = Int32(0)
            full_prefix = Int32(0)
            for prior_warp in cutlass.range_constexpr(self.num_warps):
                if Int32(prior_warp) < warp_idx:
                    partial_prefix += sWarpPartial[prior_warp]
                    full_prefix += sWarpFull[prior_warp]
            low_lanes = _shr_u32(
                Uint32(0xFFFF_FFFF),
                Uint32(Int32(32) - lane_idx),
            )
            if partial or full:
                rank = Int32(0)
                if const_expr(self.store_dq_order):
                    rank = self._dq_write_rank(
                        mVisibleBits,
                        mask_head,
                        compact_q_row,
                        local_n_block,
                        mQPartialCounts[mask_head, compact_q_row] + mQFullCounts[mask_head, compact_q_row],
                    )
                if partial:
                    partial_rank = sRunning[0] + partial_prefix + Int32(cute.arch.popc(partial_ballot & low_lanes))
                    output_idx = mPartialOffsets[plan_row] + partial_rank
                    mPartialIndices[output_idx] = local_q_block
                    mPartialWorkDesc[output_idx, 0] = mask_head
                    mPartialWorkDesc[output_idx, 1] = batch_idx
                    mPartialWorkDesc[output_idx, 2] = local_q_block
                    mPartialWorkDesc[output_idx, 3] = local_n_block
                    if const_expr(self.store_dq_order):
                        mPartialDQOrder[output_idx] = rank
                if full:
                    full_rank = sRunning[1] + full_prefix + Int32(cute.arch.popc(full_ballot & low_lanes))
                    output_idx = mFullOffsets[plan_row] + full_rank
                    mFullIndices[output_idx] = local_q_block
                    if const_expr(self.store_dq_order):
                        mFullDQOrder[output_idx] = rank
            cute.arch.sync_threads()
            if planner_tidx == Int32(0):
                chunk_partial = Int32(0)
                chunk_full = Int32(0)
                for warp in cutlass.range_constexpr(self.num_warps):
                    chunk_partial += sWarpPartial[warp]
                    chunk_full += sWarpFull[warp]
                sRunning[0] += chunk_partial
                sRunning[1] += chunk_full
            cute.arch.sync_threads()
            q_block_base += Int32(self.num_threads)
