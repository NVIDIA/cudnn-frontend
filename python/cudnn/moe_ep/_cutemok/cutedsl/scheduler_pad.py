# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""SM100 / SM103 scheduler pad: per-expert counts, 256-row padding and the smallest peer count.

One block per local expert, a warp per peer (round robin over EP), a lane per column block: the warp sums the
CHUNKS column-block counts of (expert, peer) that scheduler_count wrote (the (expert, peer) count the schedule
kernel needs), and the block sums the peers into the expert's row count, padded up to 256 rows, and takes the
smallest peer count (the schedule kernel's interleave fast path).  Every output entry has one writer; no atomics,
no zeroing.  The total of the padded counts is left to the schedule kernel, which needs the prefix over experts.
"""

THREADS = 256  # 8 warps: a warp per peer round robin; EP is a runtime value.
_WARPS = THREADS // 32
from ._version_gate import require_cutedsl

require_cutedsl()  # CuTe DSL >= 4.8 (cu13), before CUTLASS is imported

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from .scheduler_count import CHUNKS, _warp_sum

assert CHUNKS == 32  # one lane per column block


class _SchedulerPad:
    @cute.jit
    def __call__(
        self,
        chunk_counts: cute.Tensor,
        counts: cute.Tensor,
        min_counts: cute.Tensor,
        padded_out: cute.Tensor,
        valid_out: cute.Tensor,
        experts: Int32,
        world: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(chunk_counts, counts, min_counts, padded_out, valid_out, world).launch(
            grid=(experts, 1, 1), block=(THREADS, 1, 1), smem=Int64(4 * 2 * _WARPS), stream=stream
        )

    @cute.kernel
    def kernel(self, chunk_counts: cute.Tensor, counts_out: cute.Tensor, min_out: cute.Tensor, padded_out: cute.Tensor, valid_out: cute.Tensor, world: Int32):
        expert = cute.arch.block_idx()[0]
        thread = cute.arch.thread_idx()[0]
        lane, warp = thread % Int32(32), thread // Int32(32)
        warp_counts = cute.make_tensor(cute.arch.get_dyn_smem(Int32, alignment=4), cute.make_layout(_WARPS))
        warp_mins = cute.make_tensor(warp_counts.iterator + Int64(_WARPS), cute.make_layout(_WARPS))
        count = Int32(0)
        smallest = Int32(2**31 - 1)
        peer = warp
        while peer < world:
            peer_total = _warp_sum(chunk_counts[(Int64(expert) * Int64(world) + Int64(peer)) * Int64(CHUNKS) + Int64(lane)], lane)
            if lane == Int32(0):
                counts_out[Int64(expert) * Int64(world) + Int64(peer)] = peer_total
            count += peer_total
            if peer_total < smallest:
                smallest = peer_total
            peer += Int32(_WARPS)
        if lane == Int32(0):
            warp_counts[warp] = count
            warp_mins[warp] = smallest  # a warp without a peer (EP < _WARPS) keeps int32 max
        cute.arch.sync_threads()
        if thread == Int32(0):
            count = Int32(0)
            for other in cutlass.range_constexpr(_WARPS):
                count += warp_counts[other]
                if warp_mins[other] < smallest:
                    smallest = warp_mins[other]
            padded_out[Int64(expert)] = ((count + Int32(255)) // Int32(256)) * Int32(256)
            valid_out[Int64(expert)] = count
            min_out[Int64(expert)] = smallest


_KERNEL = _SchedulerPad()
