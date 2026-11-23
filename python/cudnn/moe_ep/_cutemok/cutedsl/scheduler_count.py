# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""SM100 / SM103 scheduler count: route counts per (local expert, peer, column) and per column block, and the
-1 fill of peer_rank.

The schedule orders the routes of one peer column-major: seen as a matrix with rows k = i // 1024 and columns
t = i % 1024, route i sits at (k, t), and the output lists column 0's routes, then column 1's, ...  This kernel
counts routes per column and per block of COLUMNS columns: scheduler_pad sums the column-block counts of every
(expert, peer), and the schedule kernel derives each column's starting ordinal from the earlier column blocks'
counts and the earlier columns' counts of its block.  Nothing is accumulated across
blocks: every entry has exactly one writer, so the output needs no zeroing.

No host fill precedes the scheduler: this kernel also writes -1 to every entry of peer_rank; the schedule
kernel, launched after it on the same stream, overwrites the routed rows.  The consumers read peer_rank on the
padding rows of every expert segment and branch on -1.
"""

# Constants are tile constants, none derives from a size:
THREADS = 1024  # Order columns: route i of a peer is in column i % THREADS (the schedule's order key).
COLUMNS = 32  # Columns per block = one warp of lanes, so a row load is 128 contiguous bytes.
CHUNKS = THREADS // COLUMNS  # Column blocks per peer (32).
BLOCK_THREADS = 512  # Threads per block (16 warps); two blocks per SM keep EP * 32 blocks in one wave for EP <= 8.
_WARPS = BLOCK_THREADS // 32  # Warps per block; warp w reads rows w, w + _WARPS, ...
_LOADS = 8  # Independent row loads a thread issues before using them (latency hiding).
from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


@cute.jit
def _load_rows(topk: cute.Tensor, peer_routes: Int64, index: Int64, rank_stride: Int64):
    """The routes at index, index + _WARPS rows, ... (_LOADS of them, one column); -1 past the peer's last route."""
    routes = []
    for load in cutlass.range_constexpr(_LOADS):
        route_index = index + Int64(load * _WARPS * THREADS)
        route = Int32(-1)
        if route_index < rank_stride:
            route = topk[peer_routes + route_index]
        routes.append(route)
    return tuple(routes)


@cute.jit
def _warp_sum(value: Int32, lane: Int32) -> Int32:
    """Sum of value over the 32 lanes (shuffle-up scan, lane 31 broadcast)."""
    inclusive = value
    for shift in cutlass.range_constexpr(5):
        offset = Int32(1 << shift)
        previous = cute.arch.shuffle_sync_up(inclusive, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
        if lane >= offset:
            inclusive += previous
    return cute.arch.shuffle_sync(inclusive, 31)


class _SchedulerCount:
    @cute.jit
    def __call__(
        self,
        topk: cute.Tensor,
        chunk_counts: cute.Tensor,
        column_counts: cute.Tensor,
        peer_rank: cute.Tensor,
        num_tokens: Int32,
        world: Int32,
        top_k: Int32,
        experts: Int32,
        capacity: Int32,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        flat = cute.make_tensor(topk.iterator, cute.make_layout(Int64(world) * Int64(num_tokens) * Int64(top_k)))
        self.kernel(flat, chunk_counts, column_counts, peer_rank, num_tokens, world, top_k, experts, capacity, rank).launch(
            grid=(Int64(world) * Int64(CHUNKS), 1, 1), block=(BLOCK_THREADS, 1, 1), smem=Int64(4) * Int64(experts) * Int64(COLUMNS), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        topk: cute.Tensor,
        chunk_counts: cute.Tensor,
        column_counts: cute.Tensor,
        peer_rank_out: cute.Tensor,
        num_tokens: Int32,
        world: Int32,
        top_k: Int32,
        experts: Int32,
        capacity: Int32,
        rank: Int32,
    ):
        # Block (peer, chunk): the routes of `peer` in columns chunk * COLUMNS .. + COLUMNS - 1.  Lane = column,
        # warp w = rows w, w + _WARPS, ... (row k holds routes k * THREADS .. k * THREADS + THREADS - 1 of the peer).
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        peer, chunk = block // Int32(CHUNKS), block % Int32(CHUNKS)
        lane, warp = thread % Int32(32), thread // Int32(32)
        # histogram[e * COLUMNS + lane]: routes of local expert e in this lane's column (all warps add to it).
        bins = Int64(experts) * Int64(COLUMNS)
        histogram = cute.make_tensor(cute.arch.get_dyn_smem(Int32, alignment=4), cute.make_layout(bins))
        bin_index = Int64(thread)
        while bin_index < bins:
            histogram[bin_index] = Int32(0)
            bin_index += Int64(BLOCK_THREADS)
        cute.arch.sync_threads()
        # peer_rank = -1 everywhere, grid-stride; the schedule kernel (next launch) writes the routed rows.
        grid_threads = Int64(cute.arch.grid_dim()[0]) * Int64(BLOCK_THREADS)
        row = Int64(block) * Int64(BLOCK_THREADS) + Int64(thread)
        while row < Int64(capacity):
            peer_rank_out[row] = Int32(-1)
            row += grid_threads
        rank_stride = Int64(num_tokens) * Int64(top_k)
        peer_routes = Int64(peer) * rank_stride
        first_expert = Int64(rank) * Int64(experts)
        index = Int64(warp) * Int64(THREADS) + Int64(chunk * Int32(COLUMNS) + lane)  # row `warp`, this lane's column
        while index < rank_stride:
            routes = _load_rows(topk, peer_routes, index, rank_stride)
            for load in cutlass.range_constexpr(_LOADS):
                local = Int64(routes[load]) - first_expert
                if (local >= Int64(0)) & (local < Int64(experts)):
                    cute.arch.atomic_add(histogram.iterator + local * Int64(COLUMNS) + Int64(lane), Int32(1), sem="relaxed", scope="cta")
            index += Int64(_LOADS * _WARPS * THREADS)
        cute.arch.sync_threads()
        # column_counts[(e * EP + peer) * THREADS + column] (thread -> expert thread / 32, this lane's column) and
        # chunk_counts[(e * EP + peer) * CHUNKS + chunk] = the expert's routes in the block's COLUMNS columns.
        bin_index = Int64(thread)
        while bin_index < bins:
            expert = bin_index // Int64(COLUMNS)
            column = Int64(chunk * Int32(COLUMNS)) + bin_index % Int64(COLUMNS)
            value = histogram[bin_index]
            column_counts[(expert * Int64(world) + Int64(peer)) * Int64(THREADS) + column] = value
            block_total = _warp_sum(value, lane)
            if lane == Int32(0):
                chunk_counts[(expert * Int64(world) + Int64(peer)) * Int64(CHUNKS) + Int64(chunk)] = block_total
            bin_index += Int64(BLOCK_THREADS)


_KERNEL = _SchedulerCount()
