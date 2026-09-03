# SPDX-License-Identifier: BSD-3-Clause
"""Architecture-neutral plan scans and exact-allocation header materialization."""

from __future__ import annotations

from typing import Optional

import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64, const_expr

import cuda.bindings.driver as cuda


class VarlenGeometry:
    """Build compact Varlen block prefixes and validate sequence metadata."""

    def __init__(
        self,
        *,
        fwd_tile_m: int,
        fwd_tile_n: int,
        qhead_per_kvhead: int,
        build_backward: bool,
        bwd_tile_m: int,
        bwd_tile_n: int,
    ) -> None:
        self.fwd_tile_m = fwd_tile_m
        self.fwd_tile_n = fwd_tile_n
        self.qhead_per_kvhead = qhead_per_kvhead
        self.build_backward = build_backward
        self.bwd_tile_m = bwd_tile_m
        self.bwd_tile_n = bwd_tile_n

    @cute.jit
    def __call__(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mCuTotalMBlocks: cute.Tensor,
        mCuTotalFwdNBlocks: cute.Tensor,
        mCuTotalBwdMBlocks: Optional[cute.Tensor],
        mCuTotalBwdNBlocks: Optional[cute.Tensor],
        mMetadataInvalid: cute.Tensor,
        total_q: Int32,
        total_k: Int32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuTotalMBlocks,
            mCuTotalFwdNBlocks,
            mCuTotalBwdMBlocks,
            mCuTotalBwdNBlocks,
            mMetadataInvalid,
            total_q,
            total_k,
            max_seqlen_q,
            max_seqlen_k,
        ).launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCuSeqlensQ: cute.Tensor,
        mCuSeqlensK: cute.Tensor,
        mCuTotalMBlocks: cute.Tensor,
        mCuTotalFwdNBlocks: cute.Tensor,
        mCuTotalBwdMBlocks: Optional[cute.Tensor],
        mCuTotalBwdNBlocks: Optional[cute.Tensor],
        mMetadataInvalid: cute.Tensor,
        total_q: Int32,
        total_k: Int32,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
    ) -> None:
        fwd_tile_m = Int32(self.fwd_tile_m)
        fwd_tile_n = Int32(self.fwd_tile_n)
        qhead_per_kvhead = Int32(self.qhead_per_kvhead)
        bwd_tile_m = Int32(self.bwd_tile_m)
        bwd_tile_n = Int32(self.bwd_tile_n)
        batch_size = cute.size(mCuSeqlensQ) - Int32(1)

        invalid = Boolean(False)
        if mCuSeqlensQ[0] != Int32(0) or mCuSeqlensK[0] != Int32(0):
            invalid = Boolean(True)
        mCuTotalMBlocks[0] = Int32(0)
        mCuTotalFwdNBlocks[0] = Int32(0)
        if const_expr(self.build_backward):
            assert mCuTotalBwdMBlocks is not None
            assert mCuTotalBwdNBlocks is not None
            mCuTotalBwdMBlocks[0] = Int32(0)
            mCuTotalBwdNBlocks[0] = Int32(0)

        batch_idx = Int32(0)
        while batch_idx < batch_size:
            q_len = mCuSeqlensQ[batch_idx + Int32(1)] - mCuSeqlensQ[batch_idx]
            k_len = mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[batch_idx]
            if q_len < Int32(0) or q_len > max_seqlen_q:
                invalid = Boolean(True)
            if k_len < Int32(0) or k_len > max_seqlen_k:
                invalid = Boolean(True)

            q_len_clamped = q_len
            k_len_clamped = k_len
            if q_len_clamped < Int32(0):
                q_len_clamped = Int32(0)
            if q_len_clamped > max_seqlen_q:
                q_len_clamped = max_seqlen_q
            if k_len_clamped < Int32(0):
                k_len_clamped = Int32(0)
            if k_len_clamped > max_seqlen_k:
                k_len_clamped = max_seqlen_k

            physical_q_len = q_len_clamped * qhead_per_kvhead
            fwd_m_blocks = (physical_q_len + fwd_tile_m - Int32(1)) // fwd_tile_m
            fwd_n_blocks = (k_len_clamped + fwd_tile_n - Int32(1)) // fwd_tile_n
            mCuTotalMBlocks[batch_idx + Int32(1)] = mCuTotalMBlocks[batch_idx] + fwd_m_blocks
            mCuTotalFwdNBlocks[batch_idx + Int32(1)] = mCuTotalFwdNBlocks[batch_idx] + fwd_n_blocks

            if const_expr(self.build_backward):
                assert mCuTotalBwdMBlocks is not None
                assert mCuTotalBwdNBlocks is not None
                bwd_m_blocks = (q_len_clamped + bwd_tile_m - Int32(1)) // bwd_tile_m
                bwd_n_blocks = (k_len_clamped + bwd_tile_n - Int32(1)) // bwd_tile_n
                mCuTotalBwdMBlocks[batch_idx + Int32(1)] = mCuTotalBwdMBlocks[batch_idx] + bwd_m_blocks
                mCuTotalBwdNBlocks[batch_idx + Int32(1)] = mCuTotalBwdNBlocks[batch_idx] + bwd_n_blocks
            batch_idx += Int32(1)

        if mCuSeqlensQ[batch_size] != total_q or mCuSeqlensK[batch_size] != total_k:
            invalid = Boolean(True)
        mMetadataInvalid[0] = invalid


class FixedScanHeader:
    """Scan retained fixed-shape count arrays and write the allocation header."""

    def __init__(self, *, build_backward: bool, build_dq: bool) -> None:
        self.build_backward = build_backward
        self.build_dq = build_dq

    @cute.jit
    def __call__(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialOffset: cute.Tensor,
        mFullOffset: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialOffset: Optional[cute.Tensor],
        mBwdFullOffset: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialOffset: Optional[cute.Tensor],
        mDqFullOffset: Optional[cute.Tensor],
        mIntervalInvalid: cute.Tensor,
        mHeader: cute.Tensor,
        total_m_blocks: Int32,
        bwd_total_m_blocks: Int32,
        bwd_total_n_blocks: Int32,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mPartialCount,
            mFullCount,
            mPartialOffset,
            mFullOffset,
            mBwdPartialCount,
            mBwdFullCount,
            mBwdPartialOffset,
            mBwdFullOffset,
            mDqPartialCount,
            mDqFullCount,
            mDqPartialOffset,
            mDqFullOffset,
            mIntervalInvalid,
            mHeader,
            total_m_blocks,
            bwd_total_m_blocks,
            bwd_total_n_blocks,
        ).launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _scan_counts(
        self,
        mCount: cute.Tensor,
        mOffset: cute.Tensor,
    ) -> Int64:
        mCountFlat = cute.make_tensor(
            mCount.iterator,
            cute.make_layout(cute.size(mCount)),
        )
        total = Int64(0)
        idx = Int32(0)
        mOffset[0] = Int32(0)
        while idx < cute.size(mCountFlat):
            total += Int64(mCountFlat[idx])
            mOffset[idx + Int32(1)] = Int32(total)
            idx += Int32(1)
        return total

    @cute.kernel
    def kernel(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialOffset: cute.Tensor,
        mFullOffset: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialOffset: Optional[cute.Tensor],
        mBwdFullOffset: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialOffset: Optional[cute.Tensor],
        mDqFullOffset: Optional[cute.Tensor],
        mIntervalInvalid: cute.Tensor,
        mHeader: cute.Tensor,
        total_m_blocks: Int32,
        bwd_total_m_blocks: Int32,
        bwd_total_n_blocks: Int32,
    ) -> None:
        partial_total = self._scan_counts(mPartialCount, mPartialOffset)
        full_total = self._scan_counts(mFullCount, mFullOffset)

        bwd_partial_total = Int64(0)
        bwd_full_total = Int64(0)
        if const_expr(self.build_backward):
            assert mBwdPartialCount is not None
            assert mBwdFullCount is not None
            assert mBwdPartialOffset is not None
            assert mBwdFullOffset is not None
            bwd_partial_total = self._scan_counts(
                mBwdPartialCount,
                mBwdPartialOffset,
            )
            bwd_full_total = self._scan_counts(
                mBwdFullCount,
                mBwdFullOffset,
            )

        if const_expr(self.build_dq):
            assert mDqPartialCount is not None
            assert mDqFullCount is not None
            assert mDqPartialOffset is not None
            assert mDqFullOffset is not None
            self._scan_counts(mDqPartialCount, mDqPartialOffset)
            self._scan_counts(mDqFullCount, mDqFullOffset)

        mHeader[0] = Int64(total_m_blocks)
        mHeader[1] = partial_total
        mHeader[2] = full_total
        mHeader[3] = Int64(bwd_total_m_blocks)
        mHeader[4] = Int64(bwd_total_n_blocks)
        mHeader[5] = bwd_partial_total
        mHeader[6] = bwd_full_total
        mHeader[7] = Int64(mIntervalInvalid[0])
        mHeader[8] = Int64(0)


class VarlenScanHeader:
    """Scan upper-bound Varlen counts and write one exact-allocation header."""

    def __init__(self, *, build_backward: bool, build_dq: bool) -> None:
        self.build_backward = build_backward
        self.build_dq = build_dq

    @cute.jit
    def __call__(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialScan: cute.Tensor,
        mFullScan: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialScan: Optional[cute.Tensor],
        mBwdFullScan: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialScan: Optional[cute.Tensor],
        mDqFullScan: Optional[cute.Tensor],
        mCuTotalMBlocks: cute.Tensor,
        mCuTotalBwdMBlocks: Optional[cute.Tensor],
        mCuTotalBwdNBlocks: Optional[cute.Tensor],
        mMetadataInvalid: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
        mHeader: cute.Tensor,
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mPartialCount,
            mFullCount,
            mPartialScan,
            mFullScan,
            mBwdPartialCount,
            mBwdFullCount,
            mBwdPartialScan,
            mBwdFullScan,
            mDqPartialCount,
            mDqFullCount,
            mDqPartialScan,
            mDqFullScan,
            mCuTotalMBlocks,
            mCuTotalBwdMBlocks,
            mCuTotalBwdNBlocks,
            mMetadataInvalid,
            mIntervalInvalid,
            mHeader,
        ).launch(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _scan_counts(
        self,
        mCount: cute.Tensor,
        mScan: cute.Tensor,
    ) -> Int64:
        mCountFlat = cute.make_tensor(
            mCount.iterator,
            cute.make_layout(cute.size(mCount)),
        )
        total = Int64(0)
        idx = Int32(0)
        while idx < cute.size(mCountFlat):
            total += Int64(mCountFlat[idx])
            mScan[idx] = total
            idx += Int32(1)
        return total

    @cute.kernel
    def kernel(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialScan: cute.Tensor,
        mFullScan: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialScan: Optional[cute.Tensor],
        mBwdFullScan: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialScan: Optional[cute.Tensor],
        mDqFullScan: Optional[cute.Tensor],
        mCuTotalMBlocks: cute.Tensor,
        mCuTotalBwdMBlocks: Optional[cute.Tensor],
        mCuTotalBwdNBlocks: Optional[cute.Tensor],
        mMetadataInvalid: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
        mHeader: cute.Tensor,
    ) -> None:
        partial_total = self._scan_counts(mPartialCount, mPartialScan)
        full_total = self._scan_counts(mFullCount, mFullScan)

        bwd_partial_total = Int64(0)
        bwd_full_total = Int64(0)
        bwd_total_m_blocks = Int64(0)
        bwd_total_n_blocks = Int64(0)
        if const_expr(self.build_backward):
            assert mBwdPartialCount is not None
            assert mBwdFullCount is not None
            assert mBwdPartialScan is not None
            assert mBwdFullScan is not None
            assert mCuTotalBwdMBlocks is not None
            assert mCuTotalBwdNBlocks is not None
            bwd_partial_total = self._scan_counts(
                mBwdPartialCount,
                mBwdPartialScan,
            )
            bwd_full_total = self._scan_counts(
                mBwdFullCount,
                mBwdFullScan,
            )
            bwd_total_m_blocks = Int64(mCuTotalBwdMBlocks[cute.size(mCuTotalBwdMBlocks) - Int32(1)])
            bwd_total_n_blocks = Int64(mCuTotalBwdNBlocks[cute.size(mCuTotalBwdNBlocks) - Int32(1)])

        if const_expr(self.build_dq):
            assert mDqPartialCount is not None
            assert mDqFullCount is not None
            assert mDqPartialScan is not None
            assert mDqFullScan is not None
            self._scan_counts(mDqPartialCount, mDqPartialScan)
            self._scan_counts(mDqFullCount, mDqFullScan)

        total_m_blocks = Int64(mCuTotalMBlocks[cute.size(mCuTotalMBlocks) - Int32(1)])
        mHeader[0] = total_m_blocks
        mHeader[1] = partial_total
        mHeader[2] = full_total
        mHeader[3] = bwd_total_m_blocks
        mHeader[4] = bwd_total_n_blocks
        mHeader[5] = bwd_partial_total
        mHeader[6] = bwd_full_total
        mHeader[7] = Int64(mIntervalInvalid[0])
        mHeader[8] = Int64(mMetadataInvalid[0])


class VarlenCompactMetadata:
    """Materialize exact compact counts and offsets from upper-bound scans."""

    def __init__(self, *, build_backward: bool, build_dq: bool) -> None:
        self.build_backward = build_backward
        self.build_dq = build_dq

    @cute.jit
    def __call__(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialScan: cute.Tensor,
        mFullScan: cute.Tensor,
        mCompactPartialCount: cute.Tensor,
        mCompactFullCount: cute.Tensor,
        mCompactPartialOffset: cute.Tensor,
        mCompactFullOffset: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialScan: Optional[cute.Tensor],
        mBwdFullScan: Optional[cute.Tensor],
        mCompactBwdPartialCount: Optional[cute.Tensor],
        mCompactBwdFullCount: Optional[cute.Tensor],
        mCompactBwdPartialOffset: Optional[cute.Tensor],
        mCompactBwdFullOffset: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialScan: Optional[cute.Tensor],
        mDqFullScan: Optional[cute.Tensor],
        mCompactDqPartialCount: Optional[cute.Tensor],
        mCompactDqFullCount: Optional[cute.Tensor],
        mCompactDqPartialOffset: Optional[cute.Tensor],
        mCompactDqFullOffset: Optional[cute.Tensor],
        stream: cuda.CUstream = None,
    ) -> None:
        self.kernel(
            mPartialCount,
            mFullCount,
            mPartialScan,
            mFullScan,
            mCompactPartialCount,
            mCompactFullCount,
            mCompactPartialOffset,
            mCompactFullOffset,
            mBwdPartialCount,
            mBwdFullCount,
            mBwdPartialScan,
            mBwdFullScan,
            mCompactBwdPartialCount,
            mCompactBwdFullCount,
            mCompactBwdPartialOffset,
            mCompactBwdFullOffset,
            mDqPartialCount,
            mDqFullCount,
            mDqPartialScan,
            mDqFullScan,
            mCompactDqPartialCount,
            mCompactDqFullCount,
            mCompactDqPartialOffset,
            mCompactDqFullOffset,
        ).launch(
            grid=(1, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _compact_counts_offsets(
        self,
        mCount: cute.Tensor,
        mScan: cute.Tensor,
        mCompactCount: cute.Tensor,
        mCompactOffset: cute.Tensor,
        tidx: Int32,
    ) -> None:
        mCountFlat = cute.make_tensor(
            mCount.iterator,
            cute.make_layout(cute.size(mCount)),
        )
        mCompactCountFlat = cute.make_tensor(
            mCompactCount.iterator,
            cute.make_layout(cute.size(mCompactCount)),
        )
        upper_rows = mCount.shape[1]
        compact_rows = mCompactCount.shape[1]
        if tidx == Int32(0):
            mCompactOffset[0] = Int32(0)
        compact_idx = tidx
        while compact_idx < cute.size(mCompactCountFlat):
            mask_head = compact_idx // compact_rows
            local_row = compact_idx - mask_head * compact_rows
            upper_idx = mask_head * upper_rows + local_row
            mCompactCountFlat[compact_idx] = mCountFlat[upper_idx]
            mCompactOffset[compact_idx + Int32(1)] = Int32(mScan[upper_idx])
            compact_idx += Int32(256)

    @cute.kernel
    def kernel(
        self,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mPartialScan: cute.Tensor,
        mFullScan: cute.Tensor,
        mCompactPartialCount: cute.Tensor,
        mCompactFullCount: cute.Tensor,
        mCompactPartialOffset: cute.Tensor,
        mCompactFullOffset: cute.Tensor,
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        mBwdPartialScan: Optional[cute.Tensor],
        mBwdFullScan: Optional[cute.Tensor],
        mCompactBwdPartialCount: Optional[cute.Tensor],
        mCompactBwdFullCount: Optional[cute.Tensor],
        mCompactBwdPartialOffset: Optional[cute.Tensor],
        mCompactBwdFullOffset: Optional[cute.Tensor],
        mDqPartialCount: Optional[cute.Tensor],
        mDqFullCount: Optional[cute.Tensor],
        mDqPartialScan: Optional[cute.Tensor],
        mDqFullScan: Optional[cute.Tensor],
        mCompactDqPartialCount: Optional[cute.Tensor],
        mCompactDqFullCount: Optional[cute.Tensor],
        mCompactDqPartialOffset: Optional[cute.Tensor],
        mCompactDqFullOffset: Optional[cute.Tensor],
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        self._compact_counts_offsets(
            mPartialCount,
            mPartialScan,
            mCompactPartialCount,
            mCompactPartialOffset,
            tidx,
        )
        self._compact_counts_offsets(
            mFullCount,
            mFullScan,
            mCompactFullCount,
            mCompactFullOffset,
            tidx,
        )

        if const_expr(self.build_backward):
            assert mBwdPartialCount is not None
            assert mBwdFullCount is not None
            assert mBwdPartialScan is not None
            assert mBwdFullScan is not None
            assert mCompactBwdPartialCount is not None
            assert mCompactBwdFullCount is not None
            assert mCompactBwdPartialOffset is not None
            assert mCompactBwdFullOffset is not None
            self._compact_counts_offsets(
                mBwdPartialCount,
                mBwdPartialScan,
                mCompactBwdPartialCount,
                mCompactBwdPartialOffset,
                tidx,
            )
            self._compact_counts_offsets(
                mBwdFullCount,
                mBwdFullScan,
                mCompactBwdFullCount,
                mCompactBwdFullOffset,
                tidx,
            )

        if const_expr(self.build_dq):
            assert mDqPartialCount is not None
            assert mDqFullCount is not None
            assert mDqPartialScan is not None
            assert mDqFullScan is not None
            assert mCompactDqPartialCount is not None
            assert mCompactDqFullCount is not None
            assert mCompactDqPartialOffset is not None
            assert mCompactDqFullOffset is not None
            self._compact_counts_offsets(
                mDqPartialCount,
                mDqPartialScan,
                mCompactDqPartialCount,
                mCompactDqPartialOffset,
                tidx,
            )
            self._compact_counts_offsets(
                mDqFullCount,
                mDqFullScan,
                mCompactDqFullCount,
                mCompactDqFullOffset,
                tidx,
            )


__all__ = [
    "FixedScanHeader",
    "VarlenCompactMetadata",
    "VarlenGeometry",
    "VarlenScanHeader",
]
