# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-side adjacent-query top-k partitioning for SM100 DSA backward."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cudnn._cutlass_compat import SmemAllocator


class QPairTopKTransformSm100:
    """Partition adjacent top-k rows into shared and per-query segments.

    The hash table is sized from ``topk_capacity`` and never from the KV
    sequence length. Invalid or duplicate rows retain the ordinary path.
    """

    THREADS = 256

    def __init__(self, topk_capacity: int, shared_capacity: int, tile: int, threshold_divisor: int):
        """Configure a bounded hash table and tile-aligned sharing policy."""
        if not 0 < topk_capacity <= 2048:
            raise ValueError("topk_capacity must be in [1, 2048]")
        if not 0 < shared_capacity < topk_capacity or shared_capacity % tile:
            raise ValueError("shared_capacity must be tile-aligned and below topk_capacity")
        if threshold_divisor <= 0:
            raise ValueError("threshold_divisor must be positive")

        self.topk_capacity = topk_capacity
        self.shared_capacity = shared_capacity
        self.tile = tile
        self.threshold_divisor = threshold_divisor
        # The union contains at most 2K entries. Keep the table at or below
        # 50% occupancy so bounded linear probing remains cheap.
        self.table_capacity = 1 << (4 * topk_capacity - 1).bit_length()
        self.items_per_thread = (topk_capacity + self.THREADS - 1) // self.THREADS
        self.slots_per_thread = (self.table_capacity + self.THREADS - 1) // self.THREADS

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mUniqueLengths: cute.Tensor,
        mSharedLengths: cute.Tensor,
        mDSink: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        max_seqlen_kv: Int32,
        stream: cuda.CUstream,
    ):
        """Launch one independent CTA per adjacent query pair."""
        pairs = (mInput.shape[0] + 1) // 2
        self.kernel(mInput, mOutput, mUniqueLengths, mSharedLengths, mDSink, mTopkLength, max_seqlen_kv).launch(
            grid=(pairs, 1, 1), block=(self.THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        mUniqueLengths: cute.Tensor,
        mSharedLengths: cute.Tensor,
        mDSink: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        max_seqlen_kv: Int32,
    ):
        """Hash, classify, and emit one query pair entirely on device."""
        tidx, _, _ = cute.arch.thread_idx()
        pair, _, _ = cute.arch.block_idx()
        row0 = pair * Int32(2)
        row1 = row0 + Int32(1)

        # The downstream dSink reduction uses atomics. Clear its caller-owned
        # output in this already-required launch instead of adding a memset.
        if pair == Int32(0) and tidx < mDSink.shape[0]:
            mDSink[tidx] = cutlass.Float32(0.0)

        smem = SmemAllocator()
        keys = smem.allocate_tensor(Int32, cute.make_layout((self.table_capacity,)), byte_alignment=128)
        flags = smem.allocate_tensor(Int32, cute.make_layout((self.table_capacity,)), byte_alignment=128)
        # lengths[0:2], unique counts[2:4], intersection count[4], selected
        # shared length[5], and emission counters[6:9].
        meta = smem.allocate_tensor(Int32, cute.make_layout((9,)), byte_alignment=16)

        for group in cutlass.range_constexpr(self.slots_per_thread):
            slot = tidx + Int32(group * self.THREADS)
            if slot < Int32(self.table_capacity):
                keys[slot] = Int32(-1)
                flags[slot] = Int32(0)
        if tidx < Int32(9):
            meta[tidx] = Int32(0)

        if tidx < Int32(2):
            row = row0 + tidx
            length = Int32(0)
            if row < mInput.shape[0]:
                length = Int32(self.topk_capacity)
                if cutlass.const_expr(mTopkLength is not None):
                    length = mTopkLength[row]
                    if length < Int32(0):
                        length = Int32(0)
                    if length > Int32(self.topk_capacity):
                        length = Int32(self.topk_capacity)
            meta[tidx] = length
        cute.arch.sync_threads()

        # An odd final Q row cannot form a pair. Preserve it in the unique
        # segment so the ordinary kernel handles it without a host branch.
        if row1 >= mInput.shape[0]:
            for item in cutlass.range_constexpr(self.items_per_thread):
                pos = tidx + Int32(item * self.THREADS)
                if pos < Int32(self.topk_capacity):
                    mOutput[row0, Int32(self.shared_capacity) + pos] = mInput[row0, pos]
            if tidx == Int32(0):
                mUniqueLengths[row0] = meta[0]
                mSharedLengths[pair] = Int32(0)
            cute.arch.nvvm.exit()

        # Insert valid indices from both rows. The old membership bits let the
        # inserter count unique and common keys without scanning the table.
        for rank in cutlass.range_constexpr(2):
            row = row0 + Int32(rank)
            bit = Int32(1 << rank)
            other_bit = Int32(1 << (1 - rank))
            for item in cutlass.range_constexpr(self.items_per_thread):
                pos = tidx + Int32(item * self.THREADS)
                if pos < meta[rank]:
                    index = mInput[row, pos]
                    if index >= Int32(0) and index < max_seqlen_kv:
                        slot = Int32((cutlass.Uint32(index) * cutlass.Uint32(2654435761)) & cutlass.Uint32(self.table_capacity - 1))
                        probe = Int32(0)
                        inserted = Int32(0)
                        while probe < Int32(self.table_capacity) and inserted == Int32(0):
                            previous = cute.arch.atomic_cas(
                                (keys.iterator + slot).llvm_ptr,
                                cmp=Int32(-1),
                                val=index,
                                sem="relaxed",
                                scope="cta",
                            )
                            if previous == Int32(-1) or previous == index:
                                previous_flags = cute.arch.atomic_or((flags.iterator + slot).llvm_ptr, bit, sem="relaxed", scope="cta")
                                if (previous_flags & bit) == Int32(0):
                                    cute.arch.atomic_add((meta.iterator + Int32(2 + rank)).llvm_ptr, Int32(1), sem="relaxed", scope="cta")
                                    if previous_flags & other_bit:
                                        cute.arch.atomic_add((meta.iterator + Int32(4)).llvm_ptr, Int32(1), sem="relaxed", scope="cta")
                                inserted = Int32(1)
                            else:
                                slot = (slot + Int32(1)) & Int32(self.table_capacity - 1)
                                probe += Int32(1)
        cute.arch.sync_threads()

        if tidx == Int32(0):
            usable = meta[0] if meta[0] < meta[1] else meta[1]
            shared = meta[4] if meta[4] < Int32(self.shared_capacity) else Int32(self.shared_capacity)
            shared = shared if shared < usable else usable
            shared = (shared // Int32(self.tile)) * Int32(self.tile)
            threshold_span = Int32(self.threshold_divisor * self.tile)
            min_shared = ((usable + threshold_span - Int32(1)) // threshold_span) * Int32(self.tile) if usable > Int32(0) else Int32(0)
            if meta[2] != meta[0] or meta[3] != meta[1] or shared < min_shared:
                shared = Int32(0)
            meta[5] = shared
            mUniqueLengths[row0] = meta[0] - shared
            mUniqueLengths[row1] = meta[1] - shared
            mSharedLengths[pair] = shared
        cute.arch.sync_threads()

        if meta[5] == Int32(0):
            for rank in cutlass.range_constexpr(2):
                row = row0 + Int32(rank)
                for item in cutlass.range_constexpr(self.items_per_thread):
                    pos = tidx + Int32(item * self.THREADS)
                    if pos < Int32(self.topk_capacity):
                        mOutput[row, Int32(self.shared_capacity) + pos] = mInput[row, pos]
            cute.arch.nvvm.exit()

        # Select common keys while walking row 0, then let row 1 observe the
        # selection bit. This visits O(K) inputs instead of O(table_capacity)
        # slots and preserves every unselected common key in both tails.
        for item in cutlass.range_constexpr(self.items_per_thread):
            pos = tidx + Int32(item * self.THREADS)
            if pos < meta[0]:
                index = mInput[row0, pos]
                slot = Int32((cutlass.Uint32(index) * cutlass.Uint32(2654435761)) & cutlass.Uint32(self.table_capacity - 1))
                while keys[slot] != index:
                    slot = (slot + Int32(1)) & Int32(self.table_capacity - 1)
                membership = flags[slot] & Int32(3)
                selected = Int32(0)
                if membership == Int32(3):
                    shared_pos = cute.arch.atomic_add((meta.iterator + Int32(6)).llvm_ptr, Int32(1), sem="relaxed", scope="cta")
                    if shared_pos < meta[5]:
                        mOutput[row0, shared_pos] = index
                        cute.arch.atomic_or((flags.iterator + slot).llvm_ptr, Int32(4), sem="relaxed", scope="cta")
                        selected = Int32(1)
                if selected == Int32(0):
                    unique_pos = cute.arch.atomic_add((meta.iterator + Int32(7)).llvm_ptr, Int32(1), sem="relaxed", scope="cta")
                    mOutput[row0, Int32(self.shared_capacity) + unique_pos] = index
        cute.arch.sync_threads()

        for item in cutlass.range_constexpr(self.items_per_thread):
            pos = tidx + Int32(item * self.THREADS)
            if pos < meta[1]:
                index = mInput[row1, pos]
                slot = Int32((cutlass.Uint32(index) * cutlass.Uint32(2654435761)) & cutlass.Uint32(self.table_capacity - 1))
                while keys[slot] != index:
                    slot = (slot + Int32(1)) & Int32(self.table_capacity - 1)
                if (flags[slot] & Int32(4)) == Int32(0):
                    unique_pos = cute.arch.atomic_add((meta.iterator + Int32(8)).llvm_ptr, Int32(1), sem="relaxed", scope="cta")
                    mOutput[row1, Int32(self.shared_capacity) + unique_pos] = index
