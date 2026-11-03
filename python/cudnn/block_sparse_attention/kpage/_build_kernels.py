# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CuTe DSL kernels that build the KPage walk tables and clear padding rows.  The forward walk kernels take the sparse
# block size of the metadata (``block`` = 128 or 64); page ids are always ``block // PAGE`` consecutive Page16 ids per
# block.

from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint8
import cuda.bindings.driver as cuda
import torch
from torch import Tensor

from cudnn._cutlass_compat import SmemAllocator
from cudnn.block_sparse_attention.csrc.utils import copy_utils
from cudnn.block_sparse_attention.csrc.utils.cute_dsl_utils import assume_tensor_aligned
from cudnn.block_sparse_attention.csrc.utils.kernel_utils import warp_reduce
from cudnn.block_sparse_attention.csrc.utils.kpage_utils import BLOCK, FRAGS_PER_TILE, HEAD_DIM, PAGE
from cudnn.block_sparse_attention.kpage._compile import compile_key as make_compile_key, to_cute_tensor


class WalkStats:
    """Count the Page16 rows required by each forward walk."""

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mSharedCount: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mBlockIndex,
            mBlockCount,
            mQSizes,
            mBlockSizes,
            mSharedCount,
            SharedStorage,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mSharedCount: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cap = mBlockIndex.shape[1]
        n_q_blocks = mQSizes.shape[0]
        n_blocks = mBlockSizes.shape[0]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)

        page_count = Int32(0)
        edge_idx = thread_idx
        while edge_idx < edge_count:
            block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
            block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
            page_count += (block_rows + PAGE - 1) // PAGE
            edge_idx += self.num_threads

        page_count = warp_reduce(page_count, lambda x, y: x + y)

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))
        if lane_idx == 0:
            warp_pages[warp_idx] = page_count
        cute.arch.sync_threads()

        if warp_idx == 0:
            page_count = warp_pages[lane_idx] if lane_idx < self.num_warps else Int32(0)
            page_count = warp_reduce(page_count, lambda x, y: x + y)
            if lane_idx == 0:
                q_size = max(min(mQSizes[walk_idx % n_q_blocks], self.block), 0)
                shared_count = Int32(0)
                if q_size > 0:
                    if q_size <= 64:
                        num_iters = (page_count + 15) // 16
                        num_iters += num_iters & 1
                        shared_count = num_iters * 2
                    else:
                        shared_count = (page_count + 7) // 8
                        shared_count += shared_count & 1
                mSharedCount[walk_idx] = shared_count


class CompactQRoutes:
    """Build stable Q64/Q128 templates and the required tile count."""

    def __init__(self):
        self.num_threads = 1024
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mQSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mSmallQ: cute.Tensor,
        mLargeQ: cute.Tensor,
        mRouteCounts: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_small: cute.struct.MemRange[Int32, self.num_warps]
            warp_large: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mQSizes,
            mTileOffsets,
            mTileCounts,
            mSmallQ,
            mLargeQ,
            mRouteCounts,
            SharedStorage,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mSmallQ: cute.Tensor,
        mLargeQ: cute.Tensor,
        mRouteCounts: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_q_blocks = mQSizes.shape[0]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_small = storage.warp_small.get_tensor(cute.make_layout(self.num_warps))
        warp_large = storage.warp_large.get_tensor(cute.make_layout(self.num_warps))

        small_base = Int32(0)
        large_base = Int32(0)
        num_chunks = (n_q_blocks + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            q_block = chunk * self.num_threads + thread_idx
            is_small = Int32(0)
            is_large = Int32(0)
            if q_block < n_q_blocks:
                q_size = max(min(mQSizes[q_block], BLOCK), 0)
                if q_size > 0:
                    if q_size <= 64:
                        is_small = Int32(1)
                    else:
                        is_large = Int32(1)

            small_rank = is_small
            large_rank = is_large
            for delta in (1, 2, 4, 8, 16):
                other_small = cute.arch.shuffle_sync_up(small_rank, delta, mask_and_clamp=0)
                other_large = cute.arch.shuffle_sync_up(large_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    small_rank += other_small
                    large_rank += other_large

            if lane_idx == 31:
                warp_small[warp_idx] = small_rank
                warp_large[warp_idx] = large_rank
            cute.arch.sync_threads()

            small_rank += small_base
            large_rank += large_base
            for idx in cutlass.range(warp_idx):
                small_rank += warp_small[idx]
                large_rank += warp_large[idx]

            if is_small != 0:
                mSmallQ[small_rank - 1] = q_block
            if is_large != 0:
                mLargeQ[large_rank - 1] = q_block

            for idx in cutlass.range(self.num_warps):
                small_base += warp_small[idx]
                large_base += warp_large[idx]
            cute.arch.sync_threads()

        if thread_idx == 0:
            mRouteCounts[0] = small_base
            mRouteCounts[1] = large_base
            last_walk = mTileCounts.shape[0] - 1
            mRouteCounts[2] = mTileOffsets[last_walk] + mTileCounts[last_walk]


class WalkFill:
    """Fill Page16 walks, packed masks, and both M128/M64 metadata views.

    With ``block == 64`` every Q block is an M64 tile: the M64 views are indexed
    by the Q block itself (no second half) and the M128 view is unused.

    With ``block == 128`` every table is written one output word per thread,
    consecutive threads on consecutive words: a chunk's page ids (at most
    ``num_threads`` blocks of 8 pages) go to shared memory first and are copied
    out in order, and each mask word is its own thread's.  A blk64 block has at
    most 4 pages, and its direct writes were measured as fast (``staged``).
    """

    def __init__(self, block: int = BLOCK):
        self.block = block
        self.pages_per_block = block // PAGE
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE
        self.staged = block == 128

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mFullTiles: cute.Tensor,
        mQ64Offsets: cute.Tensor,
        mQ64IterCounts: cute.Tensor,
        mQ64FullIters: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]
            stage: cute.struct.MemRange[Int32, self.num_threads * self.pages_per_block if self.staged else 1]

        self.kernel(
            mBlockIndex,
            mBlockCount,
            mQSizes,
            mBlockSizes,
            mTileOffsets,
            mTileCounts,
            mFullTiles,
            mQ64Offsets,
            mQ64IterCounts,
            mQ64FullIters,
            mPageTable,
            mMaskTable,
            SharedStorage,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def write_page(self, page_pos, page_id, allocated_pages, output_page_base, mPageTable: cute.Tensor):
        """Write one page id at walk position ``page_pos`` when it is inside the allocation."""
        if page_pos < allocated_pages:
            flat_slot = output_page_base + page_pos
            if flat_slot < mPageTable.shape[0] * 8:
                mPageTable[flat_slot // 8, flat_slot % 8] = page_id

    @cute.jit
    def copy_out(self, stage: cute.Tensor, chunk_pages, page_base, allocated_pages, output_page_base, mPageTable: cute.Tensor):
        """Copy a chunk's staged page ids to walk positions ``page_base + i``, keeping those inside the allocation."""
        idx = cute.arch.thread_idx()[0]
        while idx < chunk_pages:
            if page_base + idx < allocated_pages:
                flat_slot = output_page_base + page_base + idx
                if flat_slot < mPageTable.shape[0] * 8:
                    mPageTable[flat_slot // 8, flat_slot % 8] = stage[idx]
            idx += self.num_threads

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mTileCounts: cute.Tensor,
        mFullTiles: cute.Tensor,
        mQ64Offsets: cute.Tensor,
        mQ64IterCounts: cute.Tensor,
        mQ64FullIters: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cap = mBlockIndex.shape[1]
        n_q_blocks = mQSizes.shape[0]
        n_blocks = mBlockSizes.shape[0]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)
        q_size = max(min(mQSizes[walk_idx % n_q_blocks], self.block), 0)
        is_small = Int32(0)
        if q_size > 0:
            if q_size <= 64:
                is_small = Int32(1)
        requested_tiles = mTileCounts[walk_idx]
        tile_capacity = mPageTable.shape[0]
        offset_tiles = min(mTileOffsets[walk_idx], tile_capacity)
        room = max(tile_capacity - offset_tiles, 0)
        if is_small != 0:
            room = room & -4
        else:
            room = room & -2
        allocated_tiles = min(requested_tiles, room)
        allocated_pages = allocated_tiles * 8
        output_page_base = offset_tiles * 8

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))
        stage = storage.stage.get_tensor(cute.make_layout(self.num_threads * self.pages_per_block if self.staged else 1))

        # Keep every fully valid Page16 in one prefix. Partial Page16 tails
        # follow in stable edge order, then dummy pages fill the tile-aligned
        # suffix. This gives the attention kernel a contiguous unmasked phase.
        page_base = Int32(0)
        num_chunks = (edge_count + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            edge_idx = chunk * self.num_threads + thread_idx
            page_count = Int32(0)
            block_idx = Int32(0)
            if edge_idx < edge_count:
                block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
                block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                page_count = block_rows // PAGE

            page_rank = page_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(page_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    page_rank += other

            if lane_idx == 31:
                warp_pages[warp_idx] = page_rank
            cute.arch.sync_threads()

            page_rank += page_base
            for idx in cutlass.range(warp_idx):
                page_rank += warp_pages[idx]
            page_pos = page_rank - page_count
            first_page = block_idx * self.pages_per_block
            for page_idx in cutlass.range_constexpr(self.pages_per_block):
                if page_count > page_idx:
                    if cutlass.const_expr(self.staged):
                        stage[page_pos - page_base + page_idx] = first_page + page_idx
                    else:
                        self.write_page(page_pos + page_idx, first_page + page_idx, allocated_pages, output_page_base, mPageTable)
            chunk_pages = Int32(0)
            for idx in cutlass.range(self.num_warps):
                chunk_pages += warp_pages[idx]
            if cutlass.const_expr(self.staged):
                cute.arch.sync_threads()
                self.copy_out(stage, chunk_pages, page_base, allocated_pages, output_page_base, mPageTable)
            page_base += chunk_pages
            cute.arch.sync_threads()

        full_pages = page_base
        for chunk in cutlass.range(num_chunks):
            edge_idx = chunk * self.num_threads + thread_idx
            page_count = Int32(0)
            block_idx = Int32(0)
            tail_page_idx = Int32(0)
            if edge_idx < edge_count:
                block_idx = max(min(mBlockIndex[walk_idx, edge_idx], n_blocks - 1), 0)
                block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                tail_page_idx = block_rows // PAGE
                if block_rows - tail_page_idx * PAGE > 0:
                    page_count = Int32(1)

            page_rank = page_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(page_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    page_rank += other

            if lane_idx == 31:
                warp_pages[warp_idx] = page_rank
            cute.arch.sync_threads()

            page_rank += page_base
            for idx in cutlass.range(warp_idx):
                page_rank += warp_pages[idx]
            page_pos = page_rank - page_count
            if page_count > 0:
                if cutlass.const_expr(self.staged):
                    stage[page_pos - page_base] = block_idx * self.pages_per_block + tail_page_idx
                else:
                    self.write_page(page_pos, block_idx * self.pages_per_block + tail_page_idx, allocated_pages, output_page_base, mPageTable)
            chunk_pages = Int32(0)
            for idx in cutlass.range(self.num_warps):
                chunk_pages += warp_pages[idx]
            if cutlass.const_expr(self.staged):
                cute.arch.sync_threads()
                self.copy_out(stage, chunk_pages, page_base, allocated_pages, output_page_base, mPageTable)
            page_base += chunk_pages
            cute.arch.sync_threads()

        dummy_page = page_base + thread_idx
        zero_page = n_blocks * self.pages_per_block
        while dummy_page < allocated_pages:
            flat_slot = output_page_base + dummy_page
            mPageTable[flat_slot // 8, flat_slot % 8] = zero_page + flat_slot % 8
            dummy_page += self.num_threads
        cute.arch.sync_threads()

        valid_full_pages = min(full_pages, allocated_pages)
        first_mask_row = valid_full_pages // FRAGS_PER_TILE
        if is_small != 0:
            first_mask_row = (valid_full_pages // (2 * FRAGS_PER_TILE)) * 2

        # One int32 word covers two adjacent Page16 entries. Only rows in a
        # partial N128/N256 suffix are read by the forward kernels.  Staged: one word per thread; else one row.
        words_per_thread = 1 if self.staged else 4
        mask_word = first_mask_row * 4 + thread_idx * words_per_thread
        while mask_word < allocated_tiles * 4:
            global_row = offset_tiles + mask_word // 4
            for word_offset in cutlass.range_constexpr(words_per_thread):
                word_idx = mask_word % 4 + word_offset
                word_mask = Int32(0)
                for half in cutlass.range_constexpr(2):
                    page_id = mPageTable[global_row, word_idx * 2 + half]
                    valid_rows = Int32(0)
                    if page_id < zero_page:
                        block_idx = page_id // self.pages_per_block
                        page_idx = page_id - block_idx * self.pages_per_block
                        block_rows = max(min(mBlockSizes[block_idx], self.block), 0)
                        valid_rows = max(min(block_rows - page_idx * PAGE, PAGE), 0)
                    page_mask = (Int32(1) << valid_rows) - 1
                    word_mask = word_mask | (page_mask << (half * PAGE))
                mMaskTable[global_row, word_idx] = word_mask
            mask_word += self.num_threads * words_per_thread

        if thread_idx == 0:
            mTileOffsets[walk_idx] = offset_tiles
            mTileCounts[walk_idx] = allocated_tiles
            mFullTiles[walk_idx] = valid_full_pages // FRAGS_PER_TILE
            q64_idx = walk_idx if cutlass.const_expr(self.block == 64) else 2 * walk_idx
            mQ64Offsets[q64_idx] = offset_tiles
            mQ64IterCounts[q64_idx] = allocated_tiles // 2 if is_small != 0 else 0
            mQ64FullIters[q64_idx] = valid_full_pages // (2 * FRAGS_PER_TILE) if is_small != 0 else 0
            if cutlass.const_expr(self.block != 64):
                mQ64Offsets[q64_idx + 1] = 0
                mQ64IterCounts[q64_idx + 1] = 0
                mQ64FullIters[q64_idx + 1] = 0


class _PairWalkBase:
    """Shared pieces of the blk64 Q-pair walk kernels (one CTA per pair of Q blocks 2p, 2p + 1).

    The pair's walk covers the union of both KV lists, ordered first-only, common, second-only (ascending block id
    within each class), each listed block contributing ceil(rows / 16) Page16 entries. Membership is kept as two
    bitsets per pair, so the lists need not be sorted.
    """

    pages_per_block = 64 // PAGE
    num_threads = 256
    num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def class_bits(self, mBits: cute.Tensor, pair_idx, word, cls: cutlass.Constexpr[int]):
        """Blocks of class ``cls`` (0 first-only, 1 common, 2 second-only) in bitset word ``word``."""
        a = mBits[pair_idx, 0, word]
        b = mBits[pair_idx, 1, word]
        if cutlass.const_expr(cls == 0):
            return a & ~b
        if cutlass.const_expr(cls == 1):
            return a & b
        return b & ~a

    @cute.jit
    def word_pages(self, bits, word, mBlockSizes: cute.Tensor):
        """Page16 entries of the blocks set in ``bits`` (word ``word`` of a bitset)."""
        pages = Int32(0)
        while bits != 0:
            low = bits & -bits
            block_idx = word * 32 + Int32(31) - Int32(cute.arch.clz(low))
            pages += (max(min(mBlockSizes[block_idx], 64), 0) + PAGE - 1) // PAGE
            bits = bits & (bits - 1)
        return pages

    @cute.jit
    def lane_pages(self, bits, word, mBlockSizes: cute.Tensor):
        """Page16 entries of block ``word * 32 + lane`` if its bit is set in ``bits``, else 0."""
        lane_idx = cute.arch.lane_idx()
        pages = Int32(0)
        if ((bits >> lane_idx) & 1) != 0:
            pages = (max(min(mBlockSizes[word * 32 + lane_idx], 64), 0) + PAGE - 1) // PAGE
        return pages

    @cute.jit
    def block_scan(self, value, warp_sums: cute.Tensor):
        """Exclusive prefix of ``value`` over the CTA, and the CTA total."""
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = value
        for delta in (1, 2, 4, 8, 16):
            other = cute.arch.shuffle_sync_up(rank, delta, mask_and_clamp=0)
            if lane_idx >= delta:
                rank += other
        if lane_idx == 31:
            warp_sums[warp_idx] = rank
        cute.arch.sync_threads()
        total = Int32(0)
        for idx in cutlass.range(self.num_warps):
            prior = warp_sums[idx]
            if idx < warp_idx:
                rank += prior
            total += prior
        cute.arch.sync_threads()
        return rank - value, total


class PairWalkStats(_PairWalkBase):
    """Build each Q pair's membership bitsets and count the Page16 entries of its three classes."""

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,  # [H, nQ, cap]
        mBlockCount: cute.Tensor,  # [H, nQ]
        mBlockSizes: cute.Tensor,  # [nKV]
        mBits: cute.Tensor,  # [H * P, 2, words], zeroed
        mClassPages: cute.Tensor,  # [H * P, 3]
        mTiles: cute.Tensor,  # [H * P] 128-row tiles of the pair's walk
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sums: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(mBlockIndex, mBlockCount, mBlockSizes, mBits, mClassPages, mTiles, SharedStorage).launch(
            grid=(mBits.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mBits: cute.Tensor,
        mClassPages: cute.Tensor,
        mTiles: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        pair_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_q_blocks = mBlockIndex.shape[1]
        cap = mBlockIndex.shape[2]
        n_blocks = mBlockSizes.shape[0]
        n_pairs = (n_q_blocks + 1) // 2
        head = pair_idx // n_pairs
        first_q = (pair_idx - head * n_pairs) * 2
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_sums = storage.warp_sums.get_tensor(cute.make_layout(self.num_warps))

        for half in cutlass.range_constexpr(2):
            q_block = first_q + half
            if q_block < n_q_blocks:
                edge_count = max(min(mBlockCount[head, q_block], cap), 0)
                edge_idx = thread_idx
                while edge_idx < edge_count:
                    block_idx = mBlockIndex[head, q_block, edge_idx]
                    if block_idx >= 0 and block_idx < n_blocks:
                        ptr = mBits.iterator + cute.crd2idx((pair_idx, half, block_idx // 32), mBits.layout)
                        cute.arch.atomic_or(ptr.llvm_ptr, Int32(1) << (block_idx % 32), sem="relaxed", scope="gpu")
                    edge_idx += self.num_threads
        cute.arch.sync_threads()

        words = mBits.shape[2]
        walk_pages = Int32(0)
        for cls in cutlass.range_constexpr(3):
            pages = Int32(0)
            word = thread_idx
            while word < words:
                pages += self.word_pages(self.class_bits(mBits, pair_idx, word, cls), word, mBlockSizes)
                word += self.num_threads
            _, total = self.block_scan(pages, warp_sums)
            walk_pages += total
            if thread_idx == 0:
                mClassPages[pair_idx, cls] = total
        if thread_idx == 0:
            mTiles[pair_idx] = (walk_pages + FRAGS_PER_TILE - 1) // FRAGS_PER_TILE


class PairWalkFill(_PairWalkBase):
    """Write each Q pair's Page16 walk, the per-WG packed keep words, and its tile ranges.

    Every table is written one output word per thread, consecutive threads on consecutive words, so each store
    sector is written once.  For a chunk of bitset words, one thread per word counts its pages and one block scan
    gives each word its first slot and each dense word (more than ``dense_bits`` blocks) its place in a compact
    list.  The page ids go into a shared-memory window of ``stage_slots`` slots that the CTA then copies out in
    order: a sparse word's thread writes them itself, a dense word is written by one warp, one lane per block and
    each lane's slot from a warp scan (local patterns put a pair's blocks in a few dense words, which one thread
    per word would serialize; scattered ones leave a few blocks per word, which a warp per word would waste).
    """

    stage_slots = 2048
    dense_bits = 4

    @cute.jit
    def __call__(
        self,
        mBlockSizes: cute.Tensor,  # [nKV]
        mBits: cute.Tensor,  # [H * P, 2, words]
        mClassPages: cute.Tensor,  # [H * P, 3]
        mTileEnds: cute.Tensor,  # [H * P] inclusive prefix sum of the pairs' tile counts
        mPageTable: cute.Tensor,  # [T, 8]
        mKeep: cute.Tensor,  # [T, 2, 4]
        mRanges: cute.Tensor,  # [H * P, 5]
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sums: cute.struct.MemRange[Int32, self.num_warps]
            list_word: cute.struct.MemRange[Int32, self.num_threads]
            list_bits: cute.struct.MemRange[Int32, self.num_threads]
            list_start: cute.struct.MemRange[Int32, self.num_threads]
            list_pages: cute.struct.MemRange[Int32, self.num_threads]
            stage: cute.struct.MemRange[Int32, self.stage_slots]

        self.kernel(mBlockSizes, mBits, mClassPages, mTileEnds, mPageTable, mKeep, mRanges, SharedStorage).launch(
            grid=(mBits.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockSizes: cute.Tensor,
        mBits: cute.Tensor,
        mClassPages: cute.Tensor,
        mTileEnds: cute.Tensor,
        mPageTable: cute.Tensor,
        mKeep: cute.Tensor,
        mRanges: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        pair_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_blocks = mBlockSizes.shape[0]
        words = mBits.shape[2]
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_sums = storage.warp_sums.get_tensor(cute.make_layout(self.num_warps))
        list_word = storage.list_word.get_tensor(cute.make_layout(self.num_threads))
        list_bits = storage.list_bits.get_tensor(cute.make_layout(self.num_threads))
        list_start = storage.list_start.get_tensor(cute.make_layout(self.num_threads))
        list_pages = storage.list_pages.get_tensor(cute.make_layout(self.num_threads))
        stage = storage.stage.get_tensor(cute.make_layout(self.stage_slots))
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        first_pages = mClassPages[pair_idx, 0]
        common_pages = mClassPages[pair_idx, 1]
        second_pages = mClassPages[pair_idx, 2]
        walk_pages = first_pages + common_pages + second_pages
        tiles = (walk_pages + FRAGS_PER_TILE - 1) // FRAGS_PER_TILE
        tile_offset = mTileEnds[pair_idx] - tiles
        slot_base = tile_offset * FRAGS_PER_TILE

        # Pages in class order; within a class, ascending block ids (words in order, bits low to high).
        page_base = Int32(0)
        num_chunks = (words + self.num_threads - 1) // self.num_threads
        for cls in cutlass.range_constexpr(3):
            for chunk in cutlass.range(num_chunks):
                word = chunk * self.num_threads + thread_idx
                bits = Int32(0)
                if word < words:
                    bits = self.class_bits(mBits, pair_idx, word, cls)
                pages = self.word_pages(bits, word, mBlockSizes)  # <= 32 * 4 per word, so a chunk's sum fits 16 bits
                dense = Int32(1) if cute.arch.popc(bits) > self.dense_bits else Int32(0)
                packed, packed_total = self.block_scan(pages | (dense << 16), warp_sums)
                start = packed & 0xFFFF
                total = packed_total & 0xFFFF
                if dense != 0:
                    entry = packed >> 16
                    list_word[entry] = word
                    list_bits[entry] = bits
                    list_start[entry] = start
                    list_pages[entry] = pages
                cute.arch.sync_threads()
                listed = packed_total >> 16

                window = Int32(0)
                while window < total:
                    window_end = min(window + self.stage_slots, total)
                    if dense == 0 and pages > 0 and start < window_end and start + pages > window:
                        slot = start
                        remaining = bits
                        while remaining != 0:
                            low = remaining & -remaining
                            block_idx = word * 32 + Int32(31) - Int32(cute.arch.clz(low))
                            block_pages = (max(min(mBlockSizes[block_idx], 64), 0) + PAGE - 1) // PAGE
                            for page_idx in cutlass.range(block_pages):
                                if slot >= window and slot < window_end:
                                    stage[slot - window] = block_idx * self.pages_per_block + page_idx
                                slot += 1
                            remaining = remaining & (remaining - 1)
                    entry = warp_idx
                    while entry < listed:
                        first = list_start[entry]
                        if first < window_end and first + list_pages[entry] > window:
                            dense_word = list_word[entry]
                            lane_pages = self.lane_pages(list_bits[entry], dense_word, mBlockSizes)
                            slot = lane_pages
                            for delta in (1, 2, 4, 8, 16):
                                other = cute.arch.shuffle_sync_up(slot, delta, mask_and_clamp=0)
                                if lane_idx >= delta:
                                    slot += other
                            slot += first - lane_pages  # exclusive: this block's first page in the class walk
                            for page_idx in cutlass.range(lane_pages):
                                if slot >= window and slot < window_end:
                                    stage[slot - window] = (dense_word * 32 + lane_idx) * self.pages_per_block + page_idx
                                slot += 1
                        entry += self.num_warps
                    cute.arch.sync_threads()
                    copied = thread_idx
                    while copied < window_end - window:
                        out = slot_base + page_base + window + copied
                        mPageTable[out // FRAGS_PER_TILE, out % FRAGS_PER_TILE] = stage[copied]
                        copied += self.num_threads
                    cute.arch.sync_threads()  # the window is refilled next
                    window = window_end
                page_base += total

        # Dummy pages past seqlen_k pad the last tile; TMA loads them as zeros.
        zero_page = n_blocks * self.pages_per_block
        dummy = walk_pages + thread_idx
        while dummy < tiles * FRAGS_PER_TILE:
            slot = slot_base + dummy
            mPageTable[slot // FRAGS_PER_TILE, slot % FRAGS_PER_TILE] = zero_page + slot % FRAGS_PER_TILE
            dummy += self.num_threads
        cute.arch.sync_threads()

        # One int32 keep word covers two adjacent Page16 entries; WG0 keeps the first block's rows, WG1 the second's.
        # A page's walk position gives its class (first-only, common, second-only), so membership needs no bitset.
        # One thread per (tile, keep word), writing that word for both WGs.
        common_end = first_pages + common_pages
        entry = thread_idx
        while entry < tiles * 4:
            global_row = tile_offset + entry // 4
            word_idx = entry % 4
            keep0 = Int32(0)
            keep1 = Int32(0)
            for half in cutlass.range_constexpr(2):
                page = (entry // 4) * FRAGS_PER_TILE + word_idx * 2 + half  # position in the pair's walk
                if page < walk_pages:
                    page_id = mPageTable[global_row, word_idx * 2 + half]
                    block_idx = page_id // self.pages_per_block
                    page_idx = page_id - block_idx * self.pages_per_block
                    valid_rows = max(min(max(min(mBlockSizes[block_idx], 64), 0) - page_idx * PAGE, PAGE), 0)
                    page_mask = ((Int32(1) << valid_rows) - 1) << (half * PAGE)
                    if page < common_end:
                        keep0 = keep0 | page_mask
                    if page >= first_pages:
                        keep1 = keep1 | page_mask
            mKeep[global_row, 0, word_idx] = keep0
            mKeep[global_row, 1, word_idx] = keep1
            entry += self.num_threads

        if thread_idx == 0:
            # WG0 computes the tiles holding first-only and common pages, WG1 those holding common and second-only.
            first_hi = (first_pages + common_pages + FRAGS_PER_TILE - 1) // FRAGS_PER_TILE
            second_lo = first_pages // FRAGS_PER_TILE
            mRanges[pair_idx, 0] = tile_offset
            mRanges[pair_idx, 1] = tiles
            mRanges[pair_idx, 2] = first_hi
            mRanges[pair_idx, 3] = second_lo
            mRanges[pair_idx, 4] = tiles if common_pages + second_pages > 0 else second_lo


class OrderQRouteWork:
    """Expand one Q route and stable-partition long walks first."""

    def __init__(self, heavy_threshold: int, is_q64: bool):
        self.heavy_threshold = heavy_threshold
        self.is_q64 = is_q64
        self.num_threads = 1024
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mQRoute: cute.Tensor,
        mCost: cute.Tensor,
        mWork: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(mQRoute, mCost, mWork, SharedStorage).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRoute: cute.Tensor,
        mCost: cute.Tensor,
        mWork: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_route_q = mQRoute.shape[0]
        n_work = mWork.shape[0]
        num_heads = n_work // n_route_q
        n_q_blocks = mCost.shape[0] // num_heads
        if cutlass.const_expr(self.is_q64):
            n_q_blocks = n_q_blocks // 2

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_count = storage.warp_count.get_tensor(cute.make_layout(self.num_warps))

        local_heavy = Int32(0)
        work_idx = thread_idx
        while work_idx < n_work:
            head_idx = work_idx // n_route_q
            rank = work_idx - head_idx * n_route_q
            q_rank = rank
            if cutlass.const_expr(not self.is_q64):
                q_rank = n_route_q - 1 - rank
            q_block = mQRoute[q_rank]
            if cutlass.const_expr(self.is_q64):
                work = head_idx * (2 * n_q_blocks) + 2 * q_block
            else:
                work = head_idx * n_q_blocks + q_block
            if mCost[work] > self.heavy_threshold:
                local_heavy += 1
            work_idx += self.num_threads

        local_heavy = warp_reduce(local_heavy, lambda x, y: x + y)
        if lane_idx == 0:
            warp_count[warp_idx] = local_heavy
        cute.arch.sync_threads()
        heavy_total = Int32(0)
        for idx in cutlass.range_constexpr(self.num_warps):
            heavy_total += warp_count[idx]
        cute.arch.sync_threads()

        heavy_base = Int32(0)
        normal_base = heavy_total
        num_chunks = (n_work + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            work_idx = chunk * self.num_threads + thread_idx
            work = Int32(0)
            is_heavy = Int32(0)
            if work_idx < n_work:
                head_idx = work_idx // n_route_q
                rank = work_idx - head_idx * n_route_q
                q_rank = rank
                if cutlass.const_expr(not self.is_q64):
                    q_rank = n_route_q - 1 - rank
                q_block = mQRoute[q_rank]
                if cutlass.const_expr(self.is_q64):
                    work = head_idx * (2 * n_q_blocks) + 2 * q_block
                else:
                    work = head_idx * n_q_blocks + q_block
                if mCost[work] > self.heavy_threshold:
                    is_heavy = Int32(1)
            is_normal = Int32(work_idx < n_work) - is_heavy

            heavy_rank = is_heavy
            normal_rank = is_normal
            for delta in (1, 2, 4, 8, 16):
                other_heavy = cute.arch.shuffle_sync_up(heavy_rank, delta, mask_and_clamp=0)
                other_normal = cute.arch.shuffle_sync_up(normal_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    heavy_rank += other_heavy
                    normal_rank += other_normal

            if lane_idx == 31:
                warp_count[warp_idx] = heavy_rank
            cute.arch.sync_threads()
            heavy_rank += heavy_base
            for idx in cutlass.range(warp_idx):
                heavy_rank += warp_count[idx]
            chunk_heavy = Int32(0)
            for idx in cutlass.range_constexpr(self.num_warps):
                chunk_heavy += warp_count[idx]
            cute.arch.sync_threads()

            if lane_idx == 31:
                warp_count[warp_idx] = normal_rank
            cute.arch.sync_threads()
            normal_rank += normal_base
            for idx in cutlass.range(warp_idx):
                normal_rank += warp_count[idx]
            chunk_normal = Int32(0)
            for idx in cutlass.range_constexpr(self.num_warps):
                chunk_normal += warp_count[idx]
            cute.arch.sync_threads()

            if is_heavy != 0:
                mWork[heavy_rank - 1] = work
            if is_normal != 0:
                mWork[normal_rank - 1] = work
            heavy_base += chunk_heavy
            normal_base += chunk_normal


class RefineBlk128ToBlk64:
    """Rewrite each blk128 KV list as the 64-row half blocks that hold valid rows, for both 64-row halves of its Q block.

    KV block ``j`` with ``s`` valid rows becomes half ``2j`` (``min(s, 64)`` rows) and half ``2j + 1`` (``s - 64``
    rows); a half without valid rows is dropped from the list.  Q half ``2i`` keeps the list when Q block ``i`` has
    a valid row and half ``2i + 1`` when it has more than 64; otherwise the half gets count 0.  The halves are
    written in the [heads, Q halves] row order of the blk64 metadata the kernels consume: ``mBlockCount64`` holds
    ``n_q64`` halves per head, so with ``n_q64 = 2 * nQ - 1`` (a sequence ending 64 rows into its final block) the
    last block's second half is not written at all.  The half sizes are written to ``mBlockSizes64``.

    One warp per walk: each lane takes one list entry per step, and two ballots place its halves after those of the
    lower lanes, so the halves keep the list order (``2j`` before ``2j + 1``).
    """

    num_threads = 128
    num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,  # [W, cap] int32, W = heads * nQ
        mBlockCount: cute.Tensor,  # [W]
        mQSizes: cute.Tensor,  # [nQ]
        mBlockSizes: cute.Tensor,  # [nKV]
        mBlockIndex64: cute.Tensor,  # [heads * n_q64, 2 * cap]; entries past a row's count are left unwritten
        mBlockCount64: cute.Tensor,  # [heads * n_q64], n_q64 = 2 * nQ or 2 * nQ - 1
        mBlockSizes64: cute.Tensor,  # [2 * nKV]
        stream: cuda.CUstream,
    ):
        grid = ((mBlockCount.shape[0] + self.num_warps - 1) // self.num_warps, 1, 1)
        self.kernel(mBlockIndex, mBlockCount, mQSizes, mBlockSizes, mBlockIndex64, mBlockCount64, mBlockSizes64).launch(
            grid=grid, block=(self.num_threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mBlockIndex64: cute.Tensor,
        mBlockCount64: cute.Tensor,
        mBlockSizes64: cute.Tensor,
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_walks = mBlockCount.shape[0]
        n_kv = mBlockSizes.shape[0]

        kv = cute.arch.block_idx()[0] * self.num_threads + thread_idx
        while kv < n_kv:
            valid = min(max(mBlockSizes[kv], 0), BLOCK)
            mBlockSizes64[2 * kv] = min(valid, BLOCK // 2)
            mBlockSizes64[2 * kv + 1] = max(valid - BLOCK // 2, 0)
            kv += cute.arch.grid_dim()[0] * self.num_threads

        walk = cute.arch.block_idx()[0] * self.num_warps + warp_idx
        if walk < n_walks:
            n_q_blocks = mQSizes.shape[0]
            head = walk // n_q_blocks
            q_block = walk - head * n_q_blocks
            n_q64 = mBlockCount64.shape[0] // (n_walks // n_q_blocks)
            row = head * n_q64 + 2 * q_block  # first half's row; the second half follows unless it is past the sequence
            has_second_row = 2 * q_block + 1 < n_q64
            # clamped: the metadata check is read back only after this kernel (on the walk-table sizing sync)
            edge_count = max(min(mBlockCount[walk], mBlockIndex.shape[1]), 0)
            lower_lanes = ~(Int32(-1) << lane_idx)
            count = Int32(0)
            base = Int32(0)
            while base < edge_count:
                edge_idx = base + lane_idx
                j = Int32(0)
                valid = Int32(0)
                if edge_idx < edge_count:
                    j = mBlockIndex[walk, edge_idx]
                    valid = min(max(mBlockSizes[min(max(j, 0), n_kv - 1)], 0), BLOCK)
                has_first = cute.arch.vote_ballot_sync(valid > 0)
                has_second = cute.arch.vote_ballot_sync(valid > BLOCK // 2)
                pos = count + cute.arch.popc(has_first & lower_lanes) + cute.arch.popc(has_second & lower_lanes)
                if valid > 0:
                    mBlockIndex64[row, pos] = 2 * j
                    if has_second_row:
                        mBlockIndex64[row + 1, pos] = 2 * j
                if valid > BLOCK // 2:
                    mBlockIndex64[row, pos + 1] = 2 * j + 1
                    if has_second_row:
                        mBlockIndex64[row + 1, pos + 1] = 2 * j + 1
                count += cute.arch.popc(has_first) + cute.arch.popc(has_second)
                base += cute.arch.WARP_SIZE
            if lane_idx == 0:
                q_valid = mQSizes[q_block]
                mBlockCount64[row] = count if q_valid > 0 else Int32(0)
                if has_second_row:
                    mBlockCount64[row + 1] = count if q_valid > BLOCK // 2 else Int32(0)


# Columns of ValidateMetadata's per-CTA result row.
(
    VALIDATE_COUNTS_MIN,
    VALIDATE_COUNTS_MAX,
    VALIDATE_HAS_OOB,
    VALIDATE_HAS_DUP,
    VALIDATE_HAS_EMPTY_ROW,
    VALIDATE_SIZES_MIN,
    VALIDATE_SIZES_MAX,
    VALIDATE_PARTIAL_Q,
    VALIDATE_PARTIAL_KV,
    VALIDATE_NUM_FLAGS,
) = range(10)


class ValidateMetadata:
    """Check the KPage metadata contract in one pass over the listed KV blocks.

    CTAs stride over the walks (rows of ``q2k_block_index``) and read only the first ``q2k_block_nums`` slots of each.
    Duplicates are found with a per-CTA membership bitset in global scratch: ``atomic_or`` returns the previous word, so
    a set bit means the id was already listed.  The CTA clears the words it touched before its next walk.  Every CTA
    reduces its flags into its own row of ``mResult`` (columns ``VALIDATE_*``); the host reduces the rows.
    """

    num_threads = 256
    num_warps = num_threads // cute.arch.WARP_SIZE
    int32_max = 2**31 - 1
    int32_min = -(2**31)

    def __init__(self, block: int = BLOCK):
        self.block = block

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,  # [W, cap] int32, W = heads * nQ
        mBlockCount: cute.Tensor,  # [W]
        mQSizes: cute.Tensor,  # [nQ]
        mBlockSizes: cute.Tensor,  # [nKV]
        mScratch: cute.Tensor,  # [G, ceil(nKV / 32)] int32, zeroed; G = number of CTAs
        mResult: cute.Tensor,  # [G, VALIDATE_NUM_FLAGS] int32
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            row_has_valid: cute.struct.MemRange[Int32, 1]

        self.kernel(mBlockIndex, mBlockCount, mQSizes, mBlockSizes, mScratch, mResult, SharedStorage).launch(
            grid=(mScratch.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mQSizes: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mScratch: cute.Tensor,
        mResult: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        cta = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        num_ctas = mScratch.shape[0]
        num_walks = mBlockIndex.shape[0]
        cap = mBlockIndex.shape[1]
        n_q_blocks = mQSizes.shape[0]
        n_blocks = mBlockSizes.shape[0]
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        row_has_valid = storage.row_has_valid.get_tensor(cute.make_layout(1))

        if thread_idx == 0:
            row_has_valid[0] = Int32(0)
            mResult[cta, VALIDATE_COUNTS_MIN] = Int32(self.int32_max)
            mResult[cta, VALIDATE_COUNTS_MAX] = Int32(self.int32_min)
            mResult[cta, VALIDATE_SIZES_MIN] = Int32(self.int32_max)
            mResult[cta, VALIDATE_SIZES_MAX] = Int32(self.int32_min)
            for col in (VALIDATE_HAS_OOB, VALIDATE_HAS_DUP, VALIDATE_HAS_EMPTY_ROW, VALIDATE_PARTIAL_Q, VALIDATE_PARTIAL_KV):
                mResult[cta, col] = Int32(0)
        cute.arch.sync_threads()

        counts_min = Int32(self.int32_max)
        counts_max = Int32(self.int32_min)
        has_oob = Int32(0)
        has_dup = Int32(0)
        has_empty_row = Int32(0)
        walk = cta
        while walk < num_walks:
            count = mBlockCount[walk]
            edge_count = max(min(count, cap), 0)
            has_valid = Int32(0)
            edge_idx = thread_idx
            while edge_idx < edge_count:
                block_idx = mBlockIndex[walk, edge_idx]
                if block_idx >= 0 and block_idx < n_blocks:
                    bit = Int32(1) << (block_idx % 32)
                    ptr = mScratch.iterator + cute.crd2idx((cta, block_idx // 32), mScratch.layout)
                    previous = cute.arch.atomic_or(ptr.llvm_ptr, bit, sem="relaxed", scope="gpu")
                    if (previous & bit) != 0:
                        has_dup = Int32(1)
                    if mBlockSizes[block_idx] > 0:
                        has_valid = Int32(1)
                else:
                    has_oob = Int32(1)
                edge_idx += self.num_threads
            if has_valid != 0:
                row_has_valid[0] = Int32(1)
            cute.arch.sync_threads()

            if thread_idx == 0:
                counts_min = min(counts_min, count)
                counts_max = max(counts_max, count)
                if row_has_valid[0] == 0:
                    has_empty_row = Int32(1)
                row_has_valid[0] = Int32(0)
            edge_idx = thread_idx
            while edge_idx < edge_count:  # clear the words this walk set, for the CTA's next walk
                block_idx = mBlockIndex[walk, edge_idx]
                if block_idx >= 0 and block_idx < n_blocks:
                    mScratch[cta, block_idx // 32] = Int32(0)
                edge_idx += self.num_threads
            cute.arch.sync_threads()
            walk += num_ctas

        sizes_min = Int32(self.int32_max)
        sizes_max = Int32(self.int32_min)
        partial_q = Int32(0)
        partial_kv = Int32(0)
        idx = cta * self.num_threads + thread_idx
        while idx < n_q_blocks:
            size = mQSizes[idx]
            sizes_min = min(sizes_min, size)
            sizes_max = max(sizes_max, size)
            if size < self.block:
                partial_q = Int32(1)
            idx += num_ctas * self.num_threads
        idx = cta * self.num_threads + thread_idx
        while idx < n_blocks:
            size = mBlockSizes[idx]
            sizes_min = min(sizes_min, size)
            sizes_max = max(sizes_max, size)
            if size < self.block:
                partial_kv = Int32(1)
            idx += num_ctas * self.num_threads

        counts_min = warp_reduce(counts_min, lambda x, y: min(x, y))
        counts_max = warp_reduce(counts_max, lambda x, y: max(x, y))
        sizes_min = warp_reduce(sizes_min, lambda x, y: min(x, y))
        sizes_max = warp_reduce(sizes_max, lambda x, y: max(x, y))
        has_oob = warp_reduce(has_oob, lambda x, y: x | y)
        has_dup = warp_reduce(has_dup, lambda x, y: x | y)
        has_empty_row = warp_reduce(has_empty_row, lambda x, y: x | y)
        partial_q = warp_reduce(partial_q, lambda x, y: x | y)
        partial_kv = warp_reduce(partial_kv, lambda x, y: x | y)
        if lane_idx == 0:

            def col(c):
                return (mResult.iterator + cute.crd2idx((cta, c), mResult.layout)).llvm_ptr

            cute.arch.atomic_min(col(VALIDATE_COUNTS_MIN), counts_min, sem="relaxed", scope="cta")
            cute.arch.atomic_max(col(VALIDATE_COUNTS_MAX), counts_max, sem="relaxed", scope="cta")
            cute.arch.atomic_min(col(VALIDATE_SIZES_MIN), sizes_min, sem="relaxed", scope="cta")
            cute.arch.atomic_max(col(VALIDATE_SIZES_MAX), sizes_max, sem="relaxed", scope="cta")
            cute.arch.atomic_or(col(VALIDATE_HAS_OOB), has_oob, sem="relaxed", scope="cta")
            cute.arch.atomic_or(col(VALIDATE_HAS_DUP), has_dup, sem="relaxed", scope="cta")
            cute.arch.atomic_or(col(VALIDATE_HAS_EMPTY_ROW), has_empty_row, sem="relaxed", scope="cta")
            cute.arch.atomic_or(col(VALIDATE_PARTIAL_Q), partial_q, sem="relaxed", scope="cta")
            cute.arch.atomic_or(col(VALIDATE_PARTIAL_KV), partial_kv, sem="relaxed", scope="cta")


class BwdKvPackSlots:
    """Page counts and the no-straddle packing of the K/V blocks into 8-page owner tiles, in one CTA.

    Same layout as packing the blocks in stable order of ``page_count % 8`` (full blocks first, then partial
    blocks from short to long) with next-fit that never lets a block cross a tile: full blocks take aligned
    tiles, and inside a group of equal page counts ``c`` next-fit is periodic.  A block ranked ``j`` in its
    group fills the tile the group starts in while ``j < r0 // c`` (``r0`` slots left there), then ``8 // c``
    blocks per fresh tile.  So only each block's rank in its group is needed (ballots per chunk) plus the
    groups' start states (a few scalar steps), not a sort.  Empty blocks sit where the full blocks before
    them end.  Writes ``page_counts``, ``q_page_counts``, ``start_slots``, ``first_tiles``, ``tile_slots``
    and the packed slot total.
    """

    num_threads = 1024
    num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mKvSizes: cute.Tensor,  # [nKV]
        mQSizes: cute.Tensor,  # [nQ]
        mPageCounts: cute.Tensor,  # [nKV]
        mQPageCounts: cute.Tensor,  # [nQ]
        mStartSlots: cute.Tensor,  # [nKV]
        mFirstTiles: cute.Tensor,  # [nKV]
        mTileSlots: cute.Tensor,  # [nKV]
        mTotalSlots: cute.Tensor,  # [1]
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_counts: cute.struct.MemRange[Int32, 8 * self.num_warps]

        self.kernel(mKvSizes, mQSizes, mPageCounts, mQPageCounts, mStartSlots, mFirstTiles, mTileSlots, mTotalSlots, SharedStorage).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )

    @cute.jit
    def page_count(self, size):
        return min(max((size + PAGE - 1) // PAGE, 0), FRAGS_PER_TILE)

    @cute.kernel
    def kernel(
        self,
        mKvSizes: cute.Tensor,
        mQSizes: cute.Tensor,
        mPageCounts: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mStartSlots: cute.Tensor,
        mFirstTiles: cute.Tensor,
        mTileSlots: cute.Tensor,
        mTotalSlots: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_kv = mKvSizes.shape[0]
        n_q = mQSizes.shape[0]
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_counts = storage.warp_counts.get_tensor(cute.make_layout((8, self.num_warps), stride=(self.num_warps, 1)))
        lower_lanes = ~(Int32(-1) << lane_idx)

        q = thread_idx
        while q < n_q:
            mQPageCounts[q] = self.page_count(mQSizes[q])
            q += self.num_threads

        # Rank of every block in its group: group 0 = full blocks (an empty block counts the full blocks before it),
        # group c = blocks of c pages (1 <= c <= 7).  Group totals accumulate in ``base``.
        base = cute.make_rmem_tensor((8,), Int32)
        for g in cutlass.range_constexpr(8):
            base[g] = Int32(0)
        num_chunks = (n_kv + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            block_idx = chunk * self.num_threads + thread_idx
            count = Int32(-1)  # -1: past the end, in no group
            if block_idx < n_kv:
                count = self.page_count(mKvSizes[block_idx])
            group = count & (FRAGS_PER_TILE - 1)  # 8 -> 0
            rank = Int32(0)
            for g in cutlass.range_constexpr(8):
                ballot = cute.arch.vote_ballot_sync(count == (FRAGS_PER_TILE if g == 0 else g))
                if lane_idx == 0:
                    warp_counts[g, warp_idx] = cute.arch.popc(ballot)
                if group == g:
                    rank = cute.arch.popc(ballot & lower_lanes)
            cute.arch.sync_threads()
            if block_idx < n_kv:
                for g in cutlass.range_constexpr(8):
                    if group == g:
                        rank += base[g]
                        for w in cutlass.range(warp_idx):
                            rank += warp_counts[g, w]
                mPageCounts[block_idx] = count
                mStartSlots[block_idx] = rank  # the rank for now; replaced by the slot below
            for g in cutlass.range_constexpr(8):
                for w in cutlass.range_constexpr(self.num_warps):
                    base[g] += warp_counts[g, w]
            cute.arch.sync_threads()

        # Where each group starts: full blocks from slot 0, then groups 1..7 in turn (every thread, a few steps).
        group_start = cute.make_rmem_tensor((8,), Int32)
        group_room = cute.make_rmem_tensor((8,), Int32)
        start = base[0] * FRAGS_PER_TILE
        room = Int32(FRAGS_PER_TILE)
        group_start[0] = Int32(0)
        group_room[0] = room
        for g in cutlass.range_constexpr(1, 8):
            group_start[g] = start
            group_room[g] = room
            n = base[g]
            if n > 0:
                fit = room // g
                per_tile = FRAGS_PER_TILE // g
                if n <= fit:
                    start += n * g
                    room -= n * g
                else:
                    last = n - 1 - fit
                    used = (last % per_tile + 1) * g
                    start = group_start[g] + group_room[g] + (last // per_tile) * FRAGS_PER_TILE + used
                    room = FRAGS_PER_TILE - used
                if room == 0:
                    room = Int32(FRAGS_PER_TILE)
        if thread_idx == 0:
            mTotalSlots[0] = start
        cute.arch.sync_threads()  # the ranks above were written by other threads

        block_idx = thread_idx
        while block_idx < n_kv:
            count = mPageCounts[block_idx]
            rank = mStartSlots[block_idx]
            slot = rank * FRAGS_PER_TILE  # full blocks, and empty blocks after the full ones before them
            for g in cutlass.range_constexpr(1, 8):
                if count == g:
                    fit = group_room[g] // g
                    if rank < fit:
                        slot = group_start[g] + rank * g
                    else:
                        later = rank - fit
                        slot = group_start[g] + group_room[g] + (later // (FRAGS_PER_TILE // g)) * FRAGS_PER_TILE + (later % (FRAGS_PER_TILE // g)) * g
            mStartSlots[block_idx] = slot
            mFirstTiles[block_idx] = slot // FRAGS_PER_TILE
            mTileSlots[block_idx] = slot % FRAGS_PER_TILE
            block_idx += self.num_threads


class BwdKvPageTableFill:
    """Fill the owned-KV page ids and the valid row count of every slot."""

    def __init__(self):
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mStartSlots: cute.Tensor,
        mPageCounts: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mKvPageTable: cute.Tensor,
        mKvValid: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid = (
            (mPageCounts.shape[0] + self.num_threads - 1) // self.num_threads,
            1,
            1,
        )
        self.kernel(mStartSlots, mPageCounts, mBlockSizes, mKvPageTable, mKvValid).launch(grid=grid, block=(self.num_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mStartSlots: cute.Tensor,
        mPageCounts: cute.Tensor,
        mBlockSizes: cute.Tensor,
        mKvPageTable: cute.Tensor,
        mKvValid: cute.Tensor,
    ):
        block_idx = cute.arch.block_idx()[0] * self.num_threads + cute.arch.thread_idx()[0]
        if block_idx < mPageCounts.shape[0]:
            start = mStartSlots[block_idx]
            page_count = mPageCounts[block_idx]
            valid_rows = mBlockSizes[block_idx]
            for page_idx in cutlass.range_constexpr(8):
                if page_count > page_idx:
                    mKvPageTable[start + page_idx] = block_idx * 8 + page_idx
                    mKvValid[start + page_idx] = Uint8(min(max(valid_rows - page_idx * PAGE, 0), PAGE))


class BwdKvEdgeBitmap:
    """Scatter forward edges into packed backward adjacency bytes."""

    def __init__(self):
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mFirstTile: cute.Tensor,
        mTileSlot: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mKvPageCounts: cute.Tensor,
        mBitmap: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mBlockIndex,
            mBlockCount,
            mFirstTile,
            mTileSlot,
            mQPageCounts,
            mKvPageCounts,
            mBitmap,
            mConfig,
        ).launch(
            grid=(mBlockIndex.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBlockIndex: cute.Tensor,
        mBlockCount: cute.Tensor,
        mFirstTile: cute.Tensor,
        mTileSlot: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mKvPageCounts: cute.Tensor,
        mBitmap: cute.Tensor,
        mConfig: cute.Tensor,
    ):
        walk_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_q_blocks = mConfig[0]
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        n_kv_blocks = mConfig[4]
        cap = mBlockIndex.shape[1]
        edge_count = max(min(mBlockCount[walk_idx], cap), 0)
        head_idx = walk_idx // n_q_blocks
        q_block = walk_idx - head_idx * n_q_blocks
        q_pages = mQPageCounts[q_block]
        group_idx = q_block // bucket
        q_relative = q_block - group_idx * bucket

        edge_idx = thread_idx
        while edge_idx < edge_count:
            kv_block = max(min(mBlockIndex[walk_idx, edge_idx], n_kv_blocks - 1), 0)
            kv_pages = mKvPageCounts[kv_block]
            if q_pages > 0:
                if kv_pages > 0:
                    first_slot = mTileSlot[kv_block]
                    tile_idx = mFirstTile[kv_block]
                    adjacency = ((Int32(1) << kv_pages) - 1) << first_slot
                    row = (head_idx * num_groups + group_idx) * n_kv_blocks + tile_idx
                    word_idx = row * words_per_bucket + q_relative // 4
                    shift = (q_relative % 4) * 8
                    cute.arch.atomic_or(
                        ptr=(mBitmap.iterator + word_idx).llvm_ptr,
                        val=adjacency << shift,
                    )
            edge_idx += self.num_threads


class BwdKvBitmapStats:
    """Count backward walk steps and distinct Q blocks per bitmap row."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBitmap: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mSteps: cute.Tensor,
        mNnz: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_full: cute.struct.MemRange[Int32, self.num_warps]
            warp_partial: cute.struct.MemRange[Int32, self.num_warps]
            warp_nnz: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(mBitmap, mQPageCounts, mSteps, mNnz, mConfig, SharedStorage).launch(
            grid=(mSteps.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBitmap: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mSteps: cute.Tensor,
        mNnz: cute.Tensor,
        mConfig: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_q_blocks = mConfig[0]
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        head_group = row_idx // mConfig[4]
        group_idx = head_group % num_groups
        q_base = group_idx * bucket

        full_count = Int32(0)
        partial_pages = Int32(0)
        nnz_count = Int32(0)
        word_in_bucket = thread_idx
        while word_in_bucket < words_per_bucket:
            word = mBitmap[row_idx * words_per_bucket + word_in_bucket]
            first_q = q_base + word_in_bucket * 4
            for byte_idx in cutlass.range_constexpr(4):
                adjacency = (word >> (byte_idx * 8)) & 0xFF
                if adjacency != 0:
                    q_block = first_q + byte_idx
                    if q_block < n_q_blocks:
                        q_pages = mQPageCounts[q_block]
                        if q_pages == 8:
                            full_count += 1
                        else:
                            partial_pages += q_pages
                        nnz_count += 1
            word_in_bucket += self.num_threads

        full_count = warp_reduce(full_count, lambda x, y: x + y)
        partial_pages = warp_reduce(partial_pages, lambda x, y: x + y)
        nnz_count = warp_reduce(nnz_count, lambda x, y: x + y)
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_full = storage.warp_full.get_tensor(cute.make_layout(self.num_warps))
        warp_partial = storage.warp_partial.get_tensor(cute.make_layout(self.num_warps))
        warp_nnz = storage.warp_nnz.get_tensor(cute.make_layout(self.num_warps))
        if lane_idx == 0:
            warp_full[warp_idx] = full_count
            warp_partial[warp_idx] = partial_pages
            warp_nnz[warp_idx] = nnz_count
        cute.arch.sync_threads()
        if warp_idx == 0:
            full_count = warp_full[lane_idx] if lane_idx < self.num_warps else Int32(0)
            partial_pages = warp_partial[lane_idx] if lane_idx < self.num_warps else Int32(0)
            nnz_count = warp_nnz[lane_idx] if lane_idx < self.num_warps else Int32(0)
            full_count = warp_reduce(full_count, lambda x, y: x + y)
            partial_pages = warp_reduce(partial_pages, lambda x, y: x + y)
            nnz_count = warp_reduce(nnz_count, lambda x, y: x + y)
            if lane_idx == 0:
                mSteps[row_idx] = full_count + (partial_pages + 7) // 8
                mNnz[row_idx] = nnz_count


class BwdKvScatterWork:
    """Write each active backward row (``mActiveEnd``: inclusive prefix count of rows with steps) to its stable rank.

    Multi-CTA: the rank comes from the prefix count, so no CTA waits on another (the former single-CTA compaction
    walked every row in one block).
    """

    num_threads = 256

    @cute.jit
    def __call__(self, mActiveEnd: cute.Tensor, mConfig: cute.Tensor, mWork: cute.Tensor, stream: cuda.CUstream):
        grid = ((mActiveEnd.shape[0] + self.num_threads - 1) // self.num_threads, 1, 1)
        self.kernel(mActiveEnd, mConfig, mWork).launch(grid=grid, block=(self.num_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mActiveEnd: cute.Tensor, mConfig: cute.Tensor, mWork: cute.Tensor):
        row = cute.arch.block_idx()[0] * self.num_threads + cute.arch.thread_idx()[0]
        if row < mActiveEnd.shape[0]:
            end = mActiveEnd[row]
            before = Int32(0)
            if row > 0:
                before = mActiveEnd[row - 1]
            if end > before:
                n_kv_blocks = mConfig[4]
                head_group = row // n_kv_blocks
                mWork[end - 1, 0] = head_group
                mWork[end - 1, 1] = row - head_group * n_kv_blocks


class StablePartitionScatter:
    """``mWork[k]``: the ``k``-th index of a stable partition, selected indices first.

    ``mSelectedEnd`` is the inclusive prefix count of selected indices; an unselected index ``i`` goes after all
    selected ones, at ``total + i - mSelectedEnd[i]``.
    """

    num_threads = 256

    @cute.jit
    def __call__(self, mSelectedEnd: cute.Tensor, mWork: cute.Tensor, stream: cuda.CUstream):
        grid = ((mSelectedEnd.shape[0] + self.num_threads - 1) // self.num_threads, 1, 1)
        self.kernel(mSelectedEnd, mWork).launch(grid=grid, block=(self.num_threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, mSelectedEnd: cute.Tensor, mWork: cute.Tensor):
        idx = cute.arch.block_idx()[0] * self.num_threads + cute.arch.thread_idx()[0]
        n = mSelectedEnd.shape[0]
        if idx < n:
            end = mSelectedEnd[idx]
            before = Int32(0)
            if idx > 0:
                before = mSelectedEnd[idx - 1]
            if end > before:
                mWork[end - 1] = idx
            else:
                mWork[mSelectedEnd[n - 1] + idx - end] = idx


class BwdKvBitmapCompact:
    """Compact nonzero bitmap bytes into stable Q-block entry lists."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mBitmap: cute.Tensor,
        mEntryOffsets: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mConfig: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_count: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mBitmap,
            mEntryOffsets,
            mQBlocks,
            mAdjacencyBytes,
            mConfig,
            SharedStorage,
        ).launch(
            grid=(mEntryOffsets.shape[0] - 1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBitmap: cute.Tensor,
        mEntryOffsets: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mConfig: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        num_groups = mConfig[1]
        bucket = mConfig[2]
        words_per_bucket = mConfig[3]
        head_group = row_idx // mConfig[4]
        group_idx = head_group % num_groups
        q_base = group_idx * bucket
        output_base = mEntryOffsets[row_idx]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_count = storage.warp_count.get_tensor(cute.make_layout(self.num_warps))
        entry_base = Int32(0)
        num_chunks = (words_per_bucket + self.num_threads - 1) // self.num_threads
        for chunk in cutlass.range(num_chunks):
            word_in_bucket = chunk * self.num_threads + thread_idx
            word = Int32(0)
            if word_in_bucket < words_per_bucket:
                word = mBitmap[row_idx * words_per_bucket + word_in_bucket]
            byte0 = word & 0xFF
            byte1 = (word >> 8) & 0xFF
            byte2 = (word >> 16) & 0xFF
            byte3 = (word >> 24) & 0xFF
            entry_count = Int32(byte0 != 0) + Int32(byte1 != 0) + Int32(byte2 != 0) + Int32(byte3 != 0)
            entry_rank = entry_count
            for delta in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(entry_rank, delta, mask_and_clamp=0)
                if lane_idx >= delta:
                    entry_rank += other
            if lane_idx == 31:
                warp_count[warp_idx] = entry_rank
            cute.arch.sync_threads()
            entry_rank += entry_base
            for idx in cutlass.range(warp_idx):
                entry_rank += warp_count[idx]
            output_idx = output_base + entry_rank - entry_count
            first_q = q_base + word_in_bucket * 4
            if byte0 != 0:
                mQBlocks[output_idx] = first_q
                mAdjacencyBytes[output_idx] = byte0
                output_idx += 1
            if byte1 != 0:
                mQBlocks[output_idx] = first_q + 1
                mAdjacencyBytes[output_idx] = byte1
                output_idx += 1
            if byte2 != 0:
                mQBlocks[output_idx] = first_q + 2
                mAdjacencyBytes[output_idx] = byte2
                output_idx += 1
            if byte3 != 0:
                mQBlocks[output_idx] = first_q + 3
                mAdjacencyBytes[output_idx] = byte3

            for idx in cutlass.range(self.num_warps):
                entry_base += warp_count[idx]
            cute.arch.sync_threads()


class BwdKvWalkFill:
    """Fill backward Q-page walks, full Q blocks before partial blocks."""

    def __init__(self):
        self.num_threads = 256
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        mSegments: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mPageTable: cute.Tensor,
        mSlotBytes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_pages: cute.struct.MemRange[Int32, self.num_warps]

        self.kernel(
            mSegments,
            mQBlocks,
            mAdjacencyBytes,
            mQPageCounts,
            mTileOffsets,
            mPageTable,
            mSlotBytes,
            SharedStorage,
        ).launch(
            grid=(mTileOffsets.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSegments: cute.Tensor,
        mQBlocks: cute.Tensor,
        mAdjacencyBytes: cute.Tensor,
        mQPageCounts: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mPageTable: cute.Tensor,
        mSlotBytes: cute.Tensor,
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        row_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        entry_begin = mSegments[row_idx]
        entry_count = mSegments[row_idx + 1] - entry_begin
        output_base = mTileOffsets[row_idx] * 8

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        warp_pages = storage.warp_pages.get_tensor(cute.make_layout(self.num_warps))
        page_base = Int32(0)
        num_chunks = (entry_count + self.num_threads - 1) // self.num_threads
        for pass_idx in cutlass.range_constexpr(2):
            for chunk in cutlass.range(num_chunks):
                entry_idx = chunk * self.num_threads + thread_idx
                q_block = Int32(0)
                adjacency = Int32(0)
                selected_pages = Int32(0)
                if entry_idx < entry_count:
                    q_block = mQBlocks[entry_begin + entry_idx]
                    adjacency = mAdjacencyBytes[entry_begin + entry_idx]
                    q_pages = mQPageCounts[q_block]
                    if pass_idx == 0:
                        if q_pages == 8:
                            selected_pages = Int32(8)
                    else:
                        if q_pages != 8:
                            selected_pages = q_pages

                page_rank = selected_pages
                for delta in (1, 2, 4, 8, 16):
                    other = cute.arch.shuffle_sync_up(page_rank, delta, mask_and_clamp=0)
                    if lane_idx >= delta:
                        page_rank += other
                if lane_idx == 31:
                    warp_pages[warp_idx] = page_rank
                cute.arch.sync_threads()
                page_rank += page_base
                for idx in cutlass.range(warp_idx):
                    page_rank += warp_pages[idx]
                page_pos = page_rank - selected_pages
                first_page = q_block * 8
                for page_idx in cutlass.range_constexpr(8):
                    if selected_pages > page_idx:
                        output_idx = output_base + page_pos + page_idx
                        mPageTable[output_idx] = first_page + page_idx
                        mSlotBytes[output_idx] = Uint8(adjacency)

                for idx in cutlass.range(self.num_warps):
                    page_base += warp_pages[idx]
                cute.arch.sync_threads()


class ZeroPadForwardState:
    """Clear padding Q rows of O and LSE before backward preprocess."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mQSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        mOut = assume_tensor_aligned(mOut)
        gmem_tiled_copy = copy_utils.tiled_copy_2d(
            mOut.element_type,
            threads_per_row=16,
            num_threads=self.num_threads,
            num_copy_elems=8,
        )
        chunks_per_block = BLOCK // PAGE
        self.kernel(mOut, mLse, mQSizes, gmem_tiled_copy).launch(
            grid=(
                mOut.shape[1] * mQSizes.shape[0] * chunks_per_block,
                1,
                1,
            ),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mLse: cute.Tensor,
        mQSizes: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        chunks_per_block = BLOCK // PAGE
        head_block = block_idx // chunks_per_block
        chunk_idx = block_idx - head_block * chunks_per_block
        n_blocks = mQSizes.shape[0]
        head_idx = head_block // n_blocks
        logical_block = head_block - head_idx * n_blocks
        valid_rows = max(min(mQSizes[logical_block], BLOCK), 0)
        row_base = chunk_idx * PAGE
        rows_in_tensor = min(mOut.shape[2] - logical_block * BLOCK, BLOCK)  # a final block may end before row 128

        if row_base + PAGE > valid_rows:
            tile_idx = logical_block * chunks_per_block + chunk_idx
            gOut = cute.local_tile(
                mOut[0, head_idx, None, None],
                (PAGE, HEAD_DIM),
                (tile_idx, 0),
            )
            cOut = cute.make_identity_tensor((PAGE, HEAD_DIM))
            gmem_thr_copy = gmem_tiled_copy.get_slice(thread_idx)
            tOgOut = gmem_thr_copy.partition_D(gOut)
            tOcOut = gmem_thr_copy.partition_D(cOut)
            zeros = cute.make_fragment_like(tOgOut[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tOgOut.shape[1]):
                row = row_base + tOcOut[0, rest_m, 0][0]
                if row >= valid_rows and row < rows_in_tensor:
                    for rest_k in cutlass.range_constexpr(tOgOut.shape[2]):
                        cute.copy(
                            gmem_tiled_copy,
                            zeros,
                            tOgOut[None, rest_m, rest_k],
                        )

            if thread_idx < PAGE and row_base + thread_idx >= valid_rows and row_base + thread_idx < rows_in_tensor:
                row_idx = logical_block * BLOCK + row_base + thread_idx
                mLse[0, head_idx, row_idx] = cutlass.Float32(0.0)


class SetPadQStats:
    """Give padding Q rows LSE = +inf and dPsum = 0 so P and dS vanish on their columns."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mLseLog2: cute.Tensor,  # [1, H, T] fp32
        mDPsum: cute.Tensor,  # [1, H, T] fp32
        mQSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mLseLog2, mDPsum, mQSizes).launch(
            grid=(mLseLog2.shape[1] * mQSizes.shape[0], 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLseLog2: cute.Tensor,
        mDPsum: cute.Tensor,
        mQSizes: cute.Tensor,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        n_blocks = mQSizes.shape[0]
        head_idx = block_idx // n_blocks
        logical_block = block_idx - head_idx * n_blocks
        valid_rows = max(min(mQSizes[logical_block], BLOCK), 0)
        if thread_idx >= valid_rows:
            row_idx = logical_block * BLOCK + thread_idx
            mLseLog2[0, head_idx, row_idx] = cutlass.Float32.inf
            mDPsum[0, head_idx, row_idx] = cutlass.Float32(0.0)


class ZeroPadRows:
    """Clear padding rows of dQ (Q partition) and dK/dV (KV partition, kv heads) with vectorized stores."""

    def __init__(self):
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mQSizes: cute.Tensor,
        mKvSizes: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mDq.element_type == mDk.element_type == mDv.element_type
        mDq = assume_tensor_aligned(mDq)
        mDk = assume_tensor_aligned(mDk)
        mDv = assume_tensor_aligned(mDv)
        gmem_tiled_copy = copy_utils.tiled_copy_2d(
            mDq.element_type,
            threads_per_row=16,
            num_threads=self.num_threads,
            num_copy_elems=8,
        )
        chunks_per_block = BLOCK // PAGE
        self.kernel(mDq, mDk, mDv, mQSizes, mKvSizes, gmem_tiled_copy).launch(
            grid=(mDq.shape[1] * max(mQSizes.shape[0], mKvSizes.shape[0]) * chunks_per_block, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDv: cute.Tensor,
        mQSizes: cute.Tensor,
        mKvSizes: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
    ):
        block_idx = cute.arch.block_idx()[0]
        thread_idx = cute.arch.thread_idx()[0]
        chunks_per_block = BLOCK // PAGE
        head_block = block_idx // chunks_per_block
        chunk_idx = block_idx - head_block * chunks_per_block
        n_q_blocks = mQSizes.shape[0]
        n_kv_blocks = mKvSizes.shape[0]
        n_blocks = max(n_q_blocks, n_kv_blocks)
        head_idx = head_block // n_blocks
        logical_block = head_block - head_idx * n_blocks
        row_base = chunk_idx * PAGE
        tile_idx = logical_block * chunks_per_block + chunk_idx
        gmem_thr_copy = gmem_tiled_copy.get_slice(thread_idx)
        cRows = gmem_thr_copy.partition_D(cute.make_identity_tensor((PAGE, HEAD_DIM)))

        valid_q = Int32(BLOCK)
        rows_q = Int32(0)
        if logical_block < n_q_blocks:
            valid_q = max(min(mQSizes[logical_block], BLOCK), 0)
            rows_q = min(mDq.shape[2] - logical_block * BLOCK, BLOCK)  # a final block may end before row 128
        if row_base + PAGE > valid_q:
            gDq = cute.local_tile(mDq[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0))
            tDqgDq = gmem_thr_copy.partition_D(gDq)
            zeros = cute.make_fragment_like(tDqgDq[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tDqgDq.shape[1]):
                row = row_base + cRows[0, rest_m, 0][0]
                if row >= valid_q and row < rows_q:
                    for rest_k in cutlass.range_constexpr(tDqgDq.shape[2]):
                        cute.copy(gmem_tiled_copy, zeros, tDqgDq[None, rest_m, rest_k])

        valid_kv = Int32(BLOCK)
        rows_kv = Int32(0)
        if logical_block < n_kv_blocks and head_idx < mDk.shape[1]:
            valid_kv = max(min(mKvSizes[logical_block], BLOCK), 0)
            rows_kv = min(mDk.shape[2] - logical_block * BLOCK, BLOCK)
        if row_base + PAGE > valid_kv:
            gDk = cute.local_tile(mDk[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0))
            gDv = cute.local_tile(mDv[0, head_idx, None, None], (PAGE, HEAD_DIM), (tile_idx, 0))
            tDk_gDk = gmem_thr_copy.partition_D(gDk)
            tDv_gDv = gmem_thr_copy.partition_D(gDv)
            zeros = cute.make_fragment_like(tDk_gDk[None, 0, 0])
            zeros.fill(0.0)
            for rest_m in cutlass.range_constexpr(tDk_gDk.shape[1]):
                row = row_base + cRows[0, rest_m, 0][0]
                if row >= valid_kv and row < rows_kv:
                    for rest_k in cutlass.range_constexpr(tDk_gDk.shape[2]):
                        cute.copy(gmem_tiled_copy, zeros, tDk_gDk[None, rest_m, rest_k])
                        cute.copy(gmem_tiled_copy, zeros, tDv_gDv[None, rest_m, rest_k])


def _compile_tensors(tensors: Tuple[Tensor, ...]):
    converted = []
    for tensor in tensors:
        dynamic_modes = tuple(range(tensor.ndim)) if tensor.ndim <= 2 else (0, 1, 2)
        converted.append(
            to_cute_tensor(
                tensor,
                dynamic_modes=dynamic_modes,
                assumed_align=16 if tensor.ndim > 1 else 4,
            )
        )
    return tuple(converted)


def _run_padding_kernel(
    name: str,
    kernel,
    tensors: Tuple[Tensor, ...],
    cache: dict,
) -> None:
    compile_key = make_compile_key(
        name,
        (tuple(tensor.dtype for tensor in tensors), HEAD_DIM),
        tensors,
    )
    if compile_key not in cache:
        converted = _compile_tensors(tensors)
        cache[compile_key] = cute.compile(
            kernel,
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache[compile_key](*tensors, stream)


def _run_metadata_kernel(
    name: str,
    kernel,
    tensors: Tuple[Tensor, ...],
    cache: dict,
    stream: cuda.CUstream = None,
) -> None:
    compile_key = make_compile_key(
        name,
        tuple(tensor.dtype for tensor in tensors),
        tensors,
    )
    if compile_key not in cache:
        converted = _compile_tensors(tensors)
        cache[compile_key] = cute.compile(
            kernel,
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    if stream is None:
        stream = current_stream()
    cache[compile_key](*tensors, stream)


def current_stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def walk_stats(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    shared_count: Tensor,
    stream: cuda.CUstream = None,
    block: int = BLOCK,
) -> None:
    _run_metadata_kernel(
        f"kpage_walk_stats_blk{block}",
        WalkStats(block),
        (block_index, block_count, q_sizes, block_sizes, shared_count),
        walk_stats.compile_cache,
        stream,
    )


walk_stats.compile_cache = {}


def compact_q_routes(
    q_sizes: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    small_q: Tensor,
    large_q: Tensor,
    route_counts: Tensor,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        "kpage_compact_q_routes",
        CompactQRoutes(),
        (
            q_sizes,
            tile_offsets,
            tile_counts,
            small_q,
            large_q,
            route_counts,
        ),
        compact_q_routes.compile_cache,
        stream,
    )


compact_q_routes.compile_cache = {}


def walk_fill(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    full_tiles: Tensor,
    q64_offsets: Tensor,
    q64_iter_counts: Tensor,
    q64_full_iters: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    stream: cuda.CUstream = None,
    block: int = BLOCK,
) -> None:
    _run_metadata_kernel(
        f"kpage_walk_fill_blk{block}",
        WalkFill(block),
        (
            block_index,
            block_count,
            q_sizes,
            block_sizes,
            tile_offsets,
            tile_counts,
            full_tiles,
            q64_offsets,
            q64_iter_counts,
            q64_full_iters,
            page_table,
            mask_table,
        ),
        walk_fill.compile_cache,
        stream,
    )


walk_fill.compile_cache = {}


def pair_walk_stats(
    block_index: Tensor,
    block_count: Tensor,
    block_sizes: Tensor,
    bits: Tensor,
    class_pages: Tensor,
    tiles: Tensor,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        "kpage_pair_walk_stats_blk64",
        PairWalkStats(),
        (block_index, block_count, block_sizes, bits, class_pages, tiles),
        pair_walk_stats.compile_cache,
        stream,
    )


pair_walk_stats.compile_cache = {}


def pair_walk_fill(
    block_sizes: Tensor,
    bits: Tensor,
    class_pages: Tensor,
    tile_ends: Tensor,
    page_table: Tensor,
    keep: Tensor,
    ranges: Tensor,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        "kpage_pair_walk_fill_blk64",
        PairWalkFill(),
        (block_sizes, bits, class_pages, tile_ends, page_table, keep, ranges),
        pair_walk_fill.compile_cache,
        stream,
    )


pair_walk_fill.compile_cache = {}


def refine_blk128_to_blk64(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    block_index64: Tensor,
    block_count64: Tensor,
    block_sizes64: Tensor,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        "kpage_refine_blk128_to_blk64",
        RefineBlk128ToBlk64(),
        (block_index, block_count, q_sizes, block_sizes, block_index64, block_count64, block_sizes64),
        refine_blk128_to_blk64.compile_cache,
        stream,
    )


refine_blk128_to_blk64.compile_cache = {}


def validate_metadata(
    block_index: Tensor,
    block_count: Tensor,
    q_sizes: Tensor,
    block_sizes: Tensor,
    scratch: Tensor,
    result: Tensor,
    block: int,
    stream: cuda.CUstream = None,
) -> None:
    _run_metadata_kernel(
        f"kpage_validate_metadata_blk{block}",
        ValidateMetadata(block),
        (block_index, block_count, q_sizes, block_sizes, scratch, result),
        validate_metadata.compile_cache,
        stream,
    )


validate_metadata.compile_cache = {}


def order_q_route_work(
    q_route: Tensor,
    cost: Tensor,
    work: Tensor,
    *,
    heavy_threshold: int,
    is_q64: bool,
    stream: cuda.CUstream = None,
) -> None:
    name = f"kpage_order_q_route_work_q{64 if is_q64 else 128}"
    compile_key = make_compile_key(
        name,
        (heavy_threshold, is_q64, *(tensor.dtype for tensor in (q_route, cost, work))),
        (q_route, cost, work),
    )
    if compile_key not in order_q_route_work.compile_cache:
        converted = _compile_tensors((q_route, cost, work))
        order_q_route_work.compile_cache[compile_key] = cute.compile(
            OrderQRouteWork(heavy_threshold, is_q64),
            *converted,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    if stream is None:
        stream = current_stream()
    order_q_route_work.compile_cache[compile_key](q_route, cost, work, stream)


order_q_route_work.compile_cache = {}


def bwd_kv_pack_slots(
    kv_sizes: Tensor,
    q_sizes: Tensor,
    page_counts: Tensor,
    q_page_counts: Tensor,
    start_slots: Tensor,
    first_tiles: Tensor,
    tile_slots: Tensor,
    total_slots: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_pack_slots",
        BwdKvPackSlots(),
        (kv_sizes, q_sizes, page_counts, q_page_counts, start_slots, first_tiles, tile_slots, total_slots),
        bwd_kv_pack_slots.compile_cache,
    )


bwd_kv_pack_slots.compile_cache = {}


def bwd_kv_page_table_fill(start_slots: Tensor, page_counts: Tensor, block_sizes: Tensor, kv_page_table: Tensor, kv_valid: Tensor) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_page_table_fill",
        BwdKvPageTableFill(),
        (start_slots, page_counts, block_sizes, kv_page_table, kv_valid),
        bwd_kv_page_table_fill.compile_cache,
    )


bwd_kv_page_table_fill.compile_cache = {}


def bwd_kv_edge_bitmap(
    block_index: Tensor,
    block_count: Tensor,
    first_tile: Tensor,
    tile_slot: Tensor,
    q_page_counts: Tensor,
    kv_page_counts: Tensor,
    bitmap: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_edge_bitmap",
        BwdKvEdgeBitmap(),
        (
            block_index,
            block_count,
            first_tile,
            tile_slot,
            q_page_counts,
            kv_page_counts,
            bitmap,
            config,
        ),
        bwd_kv_edge_bitmap.compile_cache,
    )


bwd_kv_edge_bitmap.compile_cache = {}


def bwd_kv_bitmap_stats(
    bitmap: Tensor,
    q_page_counts: Tensor,
    steps: Tensor,
    nnz: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_bitmap_stats",
        BwdKvBitmapStats(),
        (bitmap, q_page_counts, steps, nnz, config),
        bwd_kv_bitmap_stats.compile_cache,
    )


bwd_kv_bitmap_stats.compile_cache = {}


def bwd_kv_scatter_work(active_end: Tensor, config: Tensor, work: Tensor) -> None:
    _run_metadata_kernel("kpage_bwd_kv_scatter_work", BwdKvScatterWork(), (active_end, config, work), bwd_kv_scatter_work.compile_cache)


bwd_kv_scatter_work.compile_cache = {}


def stable_partition_scatter(selected_end: Tensor, work: Tensor) -> None:
    _run_metadata_kernel("kpage_stable_partition_scatter", StablePartitionScatter(), (selected_end, work), stable_partition_scatter.compile_cache)


stable_partition_scatter.compile_cache = {}


def bwd_kv_bitmap_compact(
    bitmap: Tensor,
    entry_offsets: Tensor,
    q_blocks: Tensor,
    adjacency_bytes: Tensor,
    config: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_bitmap_compact",
        BwdKvBitmapCompact(),
        (bitmap, entry_offsets, q_blocks, adjacency_bytes, config),
        bwd_kv_bitmap_compact.compile_cache,
    )


bwd_kv_bitmap_compact.compile_cache = {}


def bwd_kv_walk_fill(
    segments: Tensor,
    q_blocks: Tensor,
    adjacency_bytes: Tensor,
    q_page_counts: Tensor,
    tile_offsets: Tensor,
    page_table: Tensor,
    slot_bytes: Tensor,
) -> None:
    _run_metadata_kernel(
        "kpage_bwd_kv_walk_fill",
        BwdKvWalkFill(),
        (segments, q_blocks, adjacency_bytes, q_page_counts, tile_offsets, page_table, slot_bytes),
        bwd_kv_walk_fill.compile_cache,
    )


bwd_kv_walk_fill.compile_cache = {}


def zero_pad_forward_state(out: Tensor, lse: Tensor, block_sizes: Tensor) -> None:
    _run_padding_kernel(
        "kpage_zero_pad_forward_state",
        ZeroPadForwardState(),
        (out, lse, block_sizes),
        zero_pad_forward_state.compile_cache,
    )


zero_pad_forward_state.compile_cache = {}


def zero_pad_rows(dq: Tensor, dk: Tensor, dv: Tensor, q_sizes: Tensor, kv_sizes: Tensor) -> None:
    _run_padding_kernel(
        "kpage_zero_pad_rows",
        ZeroPadRows(),
        (dq, dk, dv, q_sizes, kv_sizes),
        zero_pad_rows.compile_cache,
    )


zero_pad_rows.compile_cache = {}


def set_pad_q_stats(lse_log2: Tensor, dpsum: Tensor, block_sizes: Tensor) -> None:
    _run_padding_kernel(
        "kpage_set_pad_q_stats",
        SetPadQStats(),
        (lse_log2, dpsum, block_sizes),
        set_pad_q_stats.compile_cache,
    )


set_pad_q_stats.compile_cache = {}
