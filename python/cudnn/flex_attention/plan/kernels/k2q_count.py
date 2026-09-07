# SPDX-License-Identifier: BSD-3-Clause
"""K-major backward contributor counting kernel."""

from __future__ import annotations

from typing import Optional

import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32
import cuda.bindings.driver as cuda

from cudnn.flex_attention.plan.kernels.common import (
    _ArbitraryPlanK2QCommonSm90,
)


class _ArbitraryPlanK2QCountSm90(_ArbitraryPlanK2QCommonSm90):
    """Count K-major partial/full Q lists from Q-major arbitrary topology."""

    @cute.jit
    def __call__(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
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
        stream: cuda.CUstream = None,
    ):
        upper_total_n_blocks = mPartialCounts.shape[1]
        hmask = mPartialCounts.shape[0]
        self.kernel(
            mVisibleBits,
            mFullBits,
            mPartialCounts,
            mFullCounts,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalQBlocks,
            mCuTotalKBlocks,
            batch_size,
            seqlen_q_fixed,
            seqlen_k_fixed,
            total_q,
            total_k,
            max_n_blocks,
        ).launch(
            grid=(upper_total_n_blocks, hmask, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCounts: cute.Tensor,
        mFullCounts: cute.Tensor,
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
        tidx, _, _ = cute.arch.thread_idx()
        upper_n_row, mask_head, _ = cute.arch.block_idx()
        (
            _,
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

        if tidx < Int32(32):
            lane_idx = cute.arch.lane_idx()
            partial_count = Int32(0)
            full_count = Int32(0)
            q_group = Int32(0)
            word_idx = local_n_block // Int32(32)
            bit_idx = local_n_block - word_idx * Int32(32)
            bit = Uint32(1) << Uint32(bit_idx)
            while q_group * Int32(32) < num_q_blocks:
                local_q_block = q_group * Int32(32) + lane_idx
                partial = Boolean(False)
                full = Boolean(False)
                if valid_n_block and local_q_block < num_q_blocks:
                    compact_q_row = compact_q_begin + local_q_block
                    visible = (mVisibleBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                    full = (mFullBits[mask_head, compact_q_row, word_idx] & bit) != Uint32(0)
                    partial = visible & ~full
                partial_ballot = cute.arch.vote_ballot_sync(partial)
                full_ballot = cute.arch.vote_ballot_sync(full)
                if lane_idx == Int32(0):
                    partial_count += Int32(cute.arch.popc(partial_ballot))
                    full_count += Int32(cute.arch.popc(full_ballot))
                q_group += Int32(1)
            if lane_idx == Int32(0) and valid_n_block:
                mPartialCounts[mask_head, compact_n_row] = partial_count
                mFullCounts[mask_head, compact_n_row] = full_count
