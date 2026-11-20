# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""SM100 / SM103 scheduler schedule: places every routed row, one block per (peer, column block).

Expert e's segment is written in the order (j, p): j is the route's ordinal among peer p's routes to e, taken
column-major (route i is in column t = i % 1024 and row k = i // 1024; column 0's routes come first, then column
1's, ...), and dst = base_e + sum_r min(count[e][r], j) + #{r < p : count[e][r] > j}.  A block owns peer p's
routes in COLUMNS whole columns, and the ordinal of a route is
  the column's starting ordinal (routes of e from p in the earlier columns, from the counts of the earlier
    column blocks and of the earlier columns of this block, scheduler_count)
  + rows of e in this column from earlier register batches (the column's next ordinal, advanced per batch)
  + rows of e in this column from the earlier warps of the batch (scan over the warps)
  + rows of e in this column from the earlier rows of this thread (the thread's running ordinal).
Threads: lane = column, warp w = ROWS consecutive rows of a batch.  The block takes the local experts in groups of up to
_GROUP (one group for E <= _GROUP); per group it walks the peer's rows in batches of _WARPS * ROWS, each thread keeping
its ROWS routes in registers.  Per (column, group expert) the block keeps the column's next ordinal (32-bit, advanced
by each batch's column total) and per (column, warp, group expert) the 16-bit count, then prefix within the batch.
Every destination row has exactly one writer (the placement is a bijection), so there are no atomics; count -> pad ->
schedule order is the stream order of the three launches, and no grid-wide barrier is needed because no block reads
what another block of the same launch writes.
"""

# Tile constants (none derives from a size):
THREADS = 512  # 16 warps x 32 lanes: a lane per column, a warp per ROWS rows.
_WARPS = THREADS // 32
ROWS = 16  # Routes of one column a thread keeps in registers; a batch = _WARPS * ROWS = 256 rows.
_GROUP = 64  # Most local experts placed per pass over the peer's rows (the per-group tables below).
_BLOCKS_PER_SM = 2  # Launch bound: two blocks (32 warps) per SM; EP * 32 blocks run in one wave for EP <= 8.
_SMEM_BYTES = 110 * 1024  # Static dynamic-smem size of every launch (two fit an SM); the host checks the tables fit.
_ORDER_THREADS = 1024  # Order columns: the order key of route i is its column i % 1024.
from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int16, Int32, Int64
from cutlass._mlir.dialects import llvm
from .scheduler_count import CHUNKS, COLUMNS, THREADS as _COUNT_THREADS

assert COLUMNS == 32 and _ORDER_THREADS == _COUNT_THREADS  # a lane per column; scheduler_count's column blocks

# Shared memory (bytes): the fixed per-group tables, then the runtime-sized group counts [EP][_GROUP] and
# expert_base[E].
_OWN_BYTES = 2 * _GROUP * _WARPS * COLUMNS  # own16[lane][warp][g]: count, then prefix within the batch
_NEXT_BYTES = 4 * _GROUP * COLUMNS  # next32[lane][g]: the column's next ordinal
_TOTAL_BYTES = 2 * _GROUP * COLUMNS  # total16[lane][g]: the batch's column total
_MIN_BYTES = 4 * _GROUP  # the group's smallest peer count per expert
_SCAN_BYTES = 4 * _WARPS  # block scan warp totals
_FIXED_BYTES = _OWN_BYTES + _NEXT_BYTES + _TOTAL_BYTES + _MIN_BYTES + _SCAN_BYTES


def schedule_shared_memory_bytes(experts: int, world: int) -> int:
    """Shared memory the schedule kernel's tables need for E local experts and EP ranks; the launch always requests
    _SMEM_BYTES, so this must not exceed it (the host checks)."""
    return _FIXED_BYTES + 4 * _GROUP * world + 4 * experts


@cute.jit
def _load_column(topk: cute.Tensor, peer_routes: Int64, first_row: Int64, column: Int64, rank_stride: Int64):
    """The ROWS routes of this thread: rows first_row .. first_row + ROWS - 1 of its column; -1 past the peer's
    last route."""
    routes = []
    for row in cutlass.range_constexpr(ROWS):
        index = (first_row + Int64(row)) * Int64(_ORDER_THREADS) + column
        route = Int32(-1)
        if index < rank_stride:
            route = topk[peer_routes + index]
        routes.append(route)
    return tuple(routes)


@cute.jit
def _warp_exclusive_scan(value: Int32, lane: Int32) -> Int32:
    """Exclusive prefix of value over the 32 lanes."""
    inclusive = value
    for shift in cutlass.range_constexpr(5):
        offset = Int32(1 << shift)
        previous = cute.arch.shuffle_sync_up(inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= offset:
            inclusive += previous
    return inclusive - value


@cute.jit
def _block_exclusive_scan(value: Int32, warp_totals: cute.Tensor, lane: Int32, warp: Int32):
    """Exclusive prefix of value over the block's THREADS threads (thread order) and the block total.
    Two block syncs; the caller syncs again before warp_totals (_WARPS ints) is reused."""
    inclusive = value
    for shift in cutlass.range_constexpr(5):
        offset = Int32(1 << shift)
        previous = cute.arch.shuffle_sync_up(inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= offset:
            inclusive += previous
    if lane == Int32(31):
        warp_totals[warp] = inclusive
    cute.arch.sync_threads()
    if warp == Int32(0):
        warp_total = Int32(0)
        if lane < Int32(_WARPS):
            warp_total = warp_totals[lane]
        for shift in cutlass.range_constexpr(5):
            offset = Int32(1 << shift)
            previous = cute.arch.shuffle_sync_up(warp_total, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
            if lane >= offset:
                warp_total += previous
        if lane < Int32(_WARPS):
            warp_totals[lane] = warp_total
    cute.arch.sync_threads()
    previous_warps = Int32(0)
    if warp > Int32(0):
        previous_warps = warp_totals[warp - Int32(1)]
    return previous_warps + inclusive - value, warp_totals[Int32(_WARPS - 1)]


class _SchedulerSchedule:
    @cute.jit
    def __call__(
        self,
        topk: cute.Tensor,
        counts: cute.Tensor,
        min_counts: cute.Tensor,
        chunk_counts: cute.Tensor,
        column_counts: cute.Tensor,
        padded: cute.Tensor,
        total: cute.Tensor,
        peer_rank: cute.Tensor,
        peer_token: cute.Tensor,
        num_tokens: Int32,
        world: Int32,
        top_k: Int32,
        experts: Int32,
        capacity: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        flat = cute.make_tensor(topk.iterator, cute.make_layout(Int64(world) * Int64(num_tokens) * Int64(top_k)))
        self.kernel(
            flat, counts, min_counts, chunk_counts, column_counts, padded, total, peer_rank, peer_token, num_tokens, world, top_k, experts, capacity, rank
        ).launch(
            grid=(Int64(world) * Int64(CHUNKS), 1, 1),
            block=(THREADS, 1, 1),
            smem=_SMEM_BYTES,
            min_blocks_per_mp=_BLOCKS_PER_SM,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        topk: cute.Tensor,
        counts: cute.Tensor,
        min_counts: cute.Tensor,
        chunk_counts: cute.Tensor,
        column_counts: cute.Tensor,
        padded: cute.Tensor,
        total_out: cute.Tensor,
        peer_rank_out: cute.Tensor,
        peer_token_out: cute.Tensor,
        num_tokens: Int32,
        world: Int32,
        top_k: Int32,
        experts: Int32,
        capacity: Int32,
        rank: Int32,
    ):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        peer, chunk = block // Int32(CHUNKS), block % Int32(CHUNKS)
        lane, warp = thread % Int32(32), thread // Int32(32)
        base16 = cute.arch.get_dyn_smem(Int16, alignment=16)
        own = cute.make_tensor(base16, cute.make_layout((COLUMNS, _WARPS, _GROUP)))  # [lane, warp, g]
        # own as 8 B words
        own_words = cute.make_tensor(cute.recast_ptr(base16, dtype=Int64), cute.make_layout(_OWN_BYTES // 8))
        next32 = cute.make_tensor(cute.recast_ptr(base16 + Int64(_OWN_BYTES // 2), dtype=Int32), cute.make_layout((COLUMNS, _GROUP)))  # [lane, g]
        total16 = cute.make_tensor(
            cute.recast_ptr(base16 + Int64((_OWN_BYTES + _NEXT_BYTES) // 2), dtype=Int16), cute.make_layout((COLUMNS, _GROUP))
        )  # [lane, g]
        group_min = cute.make_tensor(cute.recast_ptr(base16 + Int64((_OWN_BYTES + _NEXT_BYTES + _TOTAL_BYTES) // 2), dtype=Int32), cute.make_layout(_GROUP))
        warp_totals = cute.make_tensor(group_min.iterator + Int64(_GROUP), cute.make_layout(_WARPS))
        # group_counts[r, g]
        group_counts = cute.make_tensor(warp_totals.iterator + Int64(_WARPS), cute.make_layout((world, _GROUP)))
        expert_base = cute.make_tensor(group_counts.iterator + Int64(world) * Int64(_GROUP), cute.make_layout(experts))
        rank_stride = Int64(num_tokens) * Int64(top_k)
        first_expert = Int64(rank) * Int64(experts)
        peer_routes = Int64(peer) * rank_stride
        column = Int64(chunk * Int32(COLUMNS) + lane)
        rows_total = (rank_stride + Int64(_ORDER_THREADS - 1)) // Int64(_ORDER_THREADS)
        total = Int32(0)
        group_first = Int32(0)
        while group_first < experts:
            group_size = cutlass.min(Int32(_GROUP), experts - group_first)
            group_start = first_expert + Int64(group_first)
            # The group's tables: every peer's count and the smallest one per expert; thread (lane, warp + i * _WARPS)
            # owns group expert warp + i * _WARPS in column lane and starts its next ordinal at the column's starting
            # ordinal = the expert's routes from this peer in the earlier column blocks + in the earlier columns of this
            # block (two warp scans over the counts scheduler_count wrote).
            for slot in cutlass.range(thread, group_size * world, THREADS, unroll=1):
                table_group, other_peer = slot // world, slot % world
                group_counts[other_peer, table_group] = counts[Int64(group_first + table_group) * Int64(world) + Int64(other_peer)]
            if thread < group_size:
                group_min[thread] = min_counts[Int64(group_first + thread)]
            for owned in cutlass.range(warp, group_size, _WARPS, unroll=1):
                pair = Int64(group_first + owned) * Int64(world) + Int64(peer)
                earlier_blocks = _warp_exclusive_scan(chunk_counts[pair * Int64(CHUNKS) + Int64(lane)], lane)
                earlier_columns = _warp_exclusive_scan(column_counts[pair * Int64(_ORDER_THREADS) + Int64(chunk * Int32(COLUMNS)) + Int64(lane)], lane)
                next32[lane, owned] = cute.arch.shuffle_sync(earlier_blocks, chunk) + earlier_columns
            if group_first == Int32(0):
                # Block scan over the padded counts of all local experts (THREADS at a time): the base row of every
                # expert and the padded total.  Every block computes the total; block 0 stores it.  After the group's
                # table loads so that their latency overlaps the scan.
                scan_base = Int32(0)
                while scan_base < experts:
                    expert = scan_base + thread
                    rows = Int32(0)
                    if expert < experts:
                        rows = padded[Int64(expert)]
                    prefix, scanned = _block_exclusive_scan(rows, warp_totals, lane, warp)
                    if expert < experts:
                        expert_base[expert] = total + prefix
                    total += scanned
                    scan_base += Int32(THREADS)
                    cute.arch.sync_threads()  # warp_totals is reused by the next round; expert_base visible to all
                if total > capacity:
                    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True)
                if (block == Int32(0)) & (thread == Int32(0)):
                    total_out[0] = total
            batch_row = Int64(0)
            while batch_row < rows_total:
                first_row = batch_row + Int64(warp * Int32(ROWS))
                routes = _load_column(topk, peer_routes, first_row, column, rank_stride)
                # 1. This thread's route counts per group expert (its own slots; own is zeroed cooperatively first).
                for word in cutlass.range_constexpr(_OWN_BYTES // 8 // THREADS):
                    own_words[thread + word * THREADS] = Int64(0)
                cute.arch.sync_threads()
                for row in cutlass.range_constexpr(ROWS):
                    hit = Int64(routes[row]) - group_start
                    if (hit >= Int64(0)) & (hit < Int64(group_size)):
                        own[lane, warp, Int32(hit)] = own[lane, warp, Int32(hit)] + Int16(1)
                cute.arch.sync_threads()
                # 2. The owner of (column lane, group expert) turns the warps' counts into the exclusive prefix over the
                #    warps, keeps the batch's column total and advances the column's next ordinal by it.
                for owned in cutlass.range(warp, group_size, _WARPS, unroll=1):
                    running = Int16(0)
                    for any_warp in cutlass.range_constexpr(_WARPS):
                        scanned_value = own[lane, any_warp, owned]
                        own[lane, any_warp, owned] = running
                        running += scanned_value
                    total16[lane, owned] = running
                    next32[lane, owned] = next32[lane, owned] + Int32(running)
                cute.arch.sync_threads()
                # 3. Place this thread's rows in row order: ordinal = next (already past this batch) - the batch's
                #    column total + the thread's running prefix (its own slot).  Below every peer's count the segment
                #    is a plain interleave (row j * EP + p); the per-peer loop runs only at or above the smallest
                #    count.
                for row in cutlass.range_constexpr(ROWS):
                    placed64 = Int64(routes[row]) - group_start
                    if (placed64 >= Int64(0)) & (placed64 < Int64(group_size)):
                        placed = Int32(placed64)
                        within = own[lane, warp, placed]
                        own[lane, warp, placed] = within + Int16(1)
                        j = next32[lane, placed] - Int32(total16[lane, placed]) + Int32(within)
                        destination = expert_base[group_first + placed]
                        if j < group_min[placed]:
                            destination += j * world + peer
                        else:
                            for other_peer in cutlass.range(world, unroll=1):
                                count = group_counts[other_peer, placed]
                                contribution = j
                                if count < j:
                                    contribution = count
                                destination += contribution
                                if (Int32(other_peer) < peer) & (count > j):
                                    destination += Int32(1)
                        peer_rank_out[Int64(destination)] = peer
                        peer_token_out[Int64(destination)] = Int32((first_row + Int64(row)) * Int64(_ORDER_THREADS) + column)
                batch_row += Int64(_WARPS * ROWS)
                if (batch_row < rows_total) | (group_first + Int32(_GROUP) < experts):
                    # The next batch's counts (step 1) overwrite own only after every owner read it.
                    cute.arch.sync_threads()
            group_first += Int32(_GROUP)
            if group_first < experts:
                # The next group's tables overwrite group_counts / next32 after every placement read them.
                cute.arch.sync_threads()


_KERNEL = _SchedulerSchedule()
