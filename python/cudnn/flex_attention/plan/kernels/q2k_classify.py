# SPDX-License-Identifier: BSD-3-Clause
"""Q-major empty/full/partial block classification kernels."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Boolean, Int32, Uint32, const_expr
import cuda.bindings.driver as cuda

from cudnn.flex_attention.plan.kernels.common import (
    _ArbitraryPlanCommonSm90,
    _shr_u32,
)


class _ArbitraryPlanClassifySm90(_ArbitraryPlanCommonSm90):
    """Validate intervals and classify candidate QK tiles."""

    @cute.jit
    def __call__(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
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
        nfunc: Int32,
        stream: cuda.CUstream = None,
    ):
        upper_total_m_blocks = mPartialCounts.shape[1]
        hmask = mArbitraryFunc.shape[0]
        self.kernel(
            mArbitraryFunc,
            mVisibleBits,
            mFullBits,
            mPartialCounts,
            mFullCounts,
            mIntervalInvalid,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_m_blocks,
            max_n_blocks,
            nfunc,
        ).launch(
            grid=(upper_total_m_blocks, hmask, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _mark_interval_invalid(self, mIntervalInvalid: cute.Tensor) -> None:
        cute.arch.atomic_or(
            mIntervalInvalid.iterator.llvm_ptr,
            Uint32(1),
            sem="relaxed",
            scope="gpu",
        )

    @cute.jit
    def _set_candidate_range(
        self,
        mVisibleBits: cute.Tensor,
        mask_head: Int32,
        compact_outer_row: Int32,
        block_begin: Int32,
        block_end: Int32,
    ) -> None:
        first_word = block_begin // Int32(32)
        last_word = (block_end - Int32(1)) // Int32(32)
        word_idx = first_word
        while word_idx <= last_word:
            lo = Int32(0)
            if word_idx == first_word:
                lo = block_begin - first_word * Int32(32)
            hi = Int32(32)
            if word_idx == last_word:
                hi = block_end - last_word * Int32(32)
            upper = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - hi))
            lower = _shr_u32(Uint32(0xFFFF_FFFF), Uint32(Int32(32) - lo))
            offset = cute.crd2idx((mask_head, compact_outer_row, word_idx), mVisibleBits.layout)
            cute.arch.atomic_or(
                (mVisibleBits.iterator + offset).llvm_ptr,
                upper ^ lower,
                sem="relaxed",
                scope="gpu",
            )
            word_idx += Int32(1)

    @cute.kernel
    def kernel(
        self,
        mArbitraryFunc: cute.Tensor,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
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
        nfunc: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        upper_outer_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
            local_m_block,
            compact_outer_row,
            q_begin,
            q_len,
            _,
            k_len,
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
        logical_q_rows = self.tile_m // self.payload_qhead_per_kvhead
        logical_row = tidx
        # A cooperative forward row can cover Q512 while the planner launch
        # intentionally stays at 256 threads.  Walk rows thread-stride so a K
        # block visible only to stage 1 / CTA rank 1 is still admitted to the
        # candidate union before the full/partial classification pass below.
        while logical_row < Int32(logical_q_rows):
            physical_row = logical_row * Int32(self.payload_qhead_per_kvhead)
            q_global, _, q_valid = self._physical_q_info(
                local_m_block,
                physical_row,
                q_begin,
                q_len,
            )
            q_valid = q_valid & valid_m_block

            if q_valid:
                previous_end = Int32(0)
                num_intervals = (nfunc + Int32(1)) // Int32(2)
                for interval_idx in cutlass.range(num_intervals, unroll=1):
                    endpoint_begin, endpoint_end, local_begin, local_end = self._safe_interval(
                        mArbitraryFunc,
                        mask_head,
                        interval_idx,
                        q_global,
                        k_len,
                    )
                    invalid = (
                        (endpoint_begin < Int32(0))
                        | (endpoint_end < Int32(0))
                        | (endpoint_begin > k_len)
                        | (endpoint_end > k_len)
                        | (endpoint_end < endpoint_begin)
                        | (endpoint_begin < previous_end)
                    )
                    if invalid:
                        self._mark_interval_invalid(mIntervalInvalid)
                    previous_end = endpoint_end
                    if local_end > local_begin:
                        block_begin = local_begin // Int32(self.tile_n)
                        block_end = cute.ceil_div(local_end, self.tile_n)
                        self._set_candidate_range(
                            mVisibleBits,
                            mask_head,
                            compact_outer_row,
                            block_begin,
                            block_end,
                        )
            logical_row += Int32(self.num_threads)

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
        cute.arch.sync_threads()

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        warp_partial_count = Int32(0)
        warp_full_count = Int32(0)
        word_idx = warp_idx
        num_words = cute.ceil_div(max_n_blocks, 32)
        while word_idx < num_words:
            candidate_word = Uint32(0)
            if valid_m_block:
                candidate_word = mVisibleBits[mask_head, compact_outer_row, word_idx]
            for bit_idx in cutlass.range_constexpr(32):
                block_id = word_idx * Int32(32) + Int32(bit_idx)
                candidate = (block_id < max_n_blocks) & ((candidate_word & Uint32(1 << bit_idx)) != Uint32(0))
                if candidate:
                    lane_visible = Boolean(False)
                    lane_full = Boolean(True)
                    logical_row = lane_idx
                    while logical_row < Int32(logical_q_rows):
                        row_in_tile = logical_row * Int32(self.payload_qhead_per_kvhead)
                        row_q_global, _, row_q_valid = self._physical_q_info(
                            local_m_block,
                            row_in_tile,
                            q_begin,
                            q_len,
                        )
                        if row_q_valid & valid_m_block:
                            row_visible, row_full = self._row_block_state(
                                mArbitraryFunc,
                                mask_head,
                                row_q_global,
                                block_id,
                                nfunc,
                                k_len,
                            )
                            lane_visible |= row_visible
                            lane_full &= row_full
                        logical_row += Int32(32)
                    warp_visible = cute.arch.vote_ballot_sync(lane_visible)
                    warp_full = cute.arch.vote_ballot_sync(lane_full)
                    if lane_idx == Int32(0):
                        is_full = Uint32(warp_full) == Uint32(0xFFFF_FFFF)
                        physical_q_len = q_len
                        if const_expr(self.pack_gqa):
                            physical_q_len *= Int32(self.qhead_per_kvhead)
                        if (local_m_block + Int32(1)) * Int32(self.tile_m) > physical_q_len:
                            is_full = Boolean(False)
                        if (block_id + Int32(1)) * Int32(self.tile_n) > k_len:
                            is_full = Boolean(False)
                        if Uint32(warp_visible) != Uint32(0):
                            if is_full:
                                mFullBits[mask_head, compact_outer_row, word_idx] |= Uint32(1 << bit_idx)
                                warp_full_count += Int32(1)
                            else:
                                warp_partial_count += Int32(1)
            word_idx += Int32(8)

        if lane_idx == Int32(0):
            sWarpPartial[warp_idx] = warp_partial_count
            sWarpFull[warp_idx] = warp_full_count
        cute.arch.sync_threads()

        if tidx == Int32(0) and valid_m_block:
            partial_count = Int32(0)
            full_count = Int32(0)
            for warp in cutlass.range_constexpr(self.num_warps):
                partial_count += sWarpPartial[warp]
                full_count += sWarpFull[warp]
            mPartialCounts[mask_head, compact_outer_row] = partial_count
            mFullCounts[mask_head, compact_outer_row] = full_count
