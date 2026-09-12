# SPDX-License-Identifier: BSD-3-Clause
"""Initialize temporary storage used while building an arbitrary-mask plan."""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, const_expr

import cuda.bindings.driver as cuda


class PlanWorkspaceInit:
    """Initialize classification and forward-schedule workspaces in one launch."""

    def __init__(self, *, build_backward: bool) -> None:
        self.build_backward = build_backward
        self.num_warps = 8
        self.num_threads = cute.arch.WARP_SIZE * self.num_warps

    @cute.jit
    def __call__(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
        mScheduleHistogram: cute.Tensor,
        mScheduleSectionCost: cute.Tensor,
        mBwdVisibleBits: Optional[cute.Tensor],
        mBwdFullBits: Optional[cute.Tensor],
        mBwdQPartialCount: Optional[cute.Tensor],
        mBwdQFullCount: Optional[cute.Tensor],
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
        stream: cuda.CUstream = None,
    ) -> None:
        max_numel = cutlass.max(
            cute.size(mVisibleBits),
            cute.size(mFullBits),
            cute.size(mPartialCount),
            cute.size(mFullCount),
            cute.size(mIntervalInvalid),
            cute.size(mScheduleHistogram),
            cute.size(mScheduleSectionCost),
        )
        if const_expr(self.build_backward):
            assert mBwdVisibleBits is not None
            assert mBwdFullBits is not None
            assert mBwdQPartialCount is not None
            assert mBwdQFullCount is not None
            assert mBwdPartialCount is not None
            assert mBwdFullCount is not None
            max_numel = cutlass.max(
                max_numel,
                cute.size(mBwdVisibleBits),
                cute.size(mBwdFullBits),
                cute.size(mBwdQPartialCount),
                cute.size(mBwdQFullCount),
                cute.size(mBwdPartialCount),
                cute.size(mBwdFullCount),
            )
        self.kernel(
            mVisibleBits,
            mFullBits,
            mPartialCount,
            mFullCount,
            mIntervalInvalid,
            mScheduleHistogram,
            mScheduleSectionCost,
            mBwdVisibleBits,
            mBwdFullBits,
            mBwdQPartialCount,
            mBwdQFullCount,
            mBwdPartialCount,
            mBwdFullCount,
        ).launch(
            grid=(cutlass.max(cute.ceil_div(max_numel, self.num_threads), 1), 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mVisibleBits: cute.Tensor,
        mFullBits: cute.Tensor,
        mPartialCount: cute.Tensor,
        mFullCount: cute.Tensor,
        mIntervalInvalid: cute.Tensor,
        mScheduleHistogram: cute.Tensor,
        mScheduleSectionCost: cute.Tensor,
        mBwdVisibleBits: Optional[cute.Tensor],
        mBwdFullBits: Optional[cute.Tensor],
        mBwdQPartialCount: Optional[cute.Tensor],
        mBwdQFullCount: Optional[cute.Tensor],
        mBwdPartialCount: Optional[cute.Tensor],
        mBwdFullCount: Optional[cute.Tensor],
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        linear_idx = block_idx * Int32(self.num_threads) + tidx

        self._zero_u32(mVisibleBits, linear_idx)
        self._zero_u32(mFullBits, linear_idx)
        self._zero_i32(mPartialCount, linear_idx)
        self._zero_i32(mFullCount, linear_idx)
        self._zero_u32(mIntervalInvalid, linear_idx)
        self._zero_i32(mScheduleHistogram, linear_idx)
        self._zero_i64(mScheduleSectionCost, linear_idx)

        if const_expr(self.build_backward):
            assert mBwdVisibleBits is not None
            assert mBwdFullBits is not None
            assert mBwdQPartialCount is not None
            assert mBwdQFullCount is not None
            assert mBwdPartialCount is not None
            assert mBwdFullCount is not None
            self._zero_u32(mBwdVisibleBits, linear_idx)
            self._zero_u32(mBwdFullBits, linear_idx)
            self._zero_i32(mBwdQPartialCount, linear_idx)
            self._zero_i32(mBwdQFullCount, linear_idx)
            self._zero_i32(mBwdPartialCount, linear_idx)
            self._zero_i32(mBwdFullCount, linear_idx)

    @cute.jit
    def _zero_u32(self, mTensor: cute.Tensor, linear_idx: Int32) -> None:
        mTensorFlat = cute.make_tensor(
            mTensor.iterator,
            cute.make_layout(cute.size(mTensor)),
        )
        if linear_idx < cute.size(mTensorFlat):
            mTensorFlat[linear_idx] = Uint32(0)

    @cute.jit
    def _zero_i32(self, mTensor: cute.Tensor, linear_idx: Int32) -> None:
        mTensorFlat = cute.make_tensor(
            mTensor.iterator,
            cute.make_layout(cute.size(mTensor)),
        )
        if linear_idx < cute.size(mTensorFlat):
            mTensorFlat[linear_idx] = Int32(0)

    @cute.jit
    def _zero_i64(self, mTensor: cute.Tensor, linear_idx: Int32) -> None:
        mTensorFlat = cute.make_tensor(
            mTensor.iterator,
            cute.make_layout(cute.size(mTensor)),
        )
        if linear_idx < cute.size(mTensorFlat):
            mTensorFlat[linear_idx] = Int64(0)


__all__ = ["PlanWorkspaceInit"]
