# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training expansion of selected pools: [valid pools' tokens][tail][-1], one warp per row."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32

WARPS = 4


class ExpandCompactKernel:
    def __init__(self, pool_size, global_ids):
        self.pool_size = pool_size
        self.global_ids = global_ids

    @cute.jit
    def __call__(self, pool_ids, positions, token_ids, lengths, seqlen_k: Int32, blocks: Int32, stride: Int32, iters: Int32, stream: cuda.CUstream):
        self.kernel(pool_ids, positions, token_ids, lengths, seqlen_k, stride, iters).launch(grid=(blocks, 1, 1), block=(WARPS * 32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, pool_ids, positions, token_ids, lengths, seqlen_k: Int32, stride: Int32, iters: Int32):
        warp = cute.arch.block_idx()[0] * WARPS + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        n_rows = cute.size(positions.shape[1])
        total = cute.size(positions.shape[0]) * n_rows
        k = cute.size(pool_ids.shape[2])
        width = cute.size(token_ids.shape[2])
        for it in range(0, iters):
            task = warp + it * stride
            if task < total:
                b = task // n_rows
                row = task % n_rows
                length = Int32(positions[b, row]) + 1
                completed = length // self.pool_size
                tail = length - completed * self.pool_size
                n_valid = Int32(0)
                for g in range(lane, k, 32):
                    if Int32(pool_ids[b, row, g]) >= 0:
                        n_valid += 1
                n_valid = cute.arch.warp_reduction_sum(n_valid)
                offset = Int32(0)
                if cutlass.const_expr(self.global_ids):
                    offset = b * seqlen_k
                for slot in range(lane, width, 32):
                    value = Int32(-1)
                    g = slot // self.pool_size
                    if g < n_valid:
                        value = Int32(pool_ids[b, row, g]) * self.pool_size + (slot - g * self.pool_size) + offset
                    else:
                        t = slot - n_valid * self.pool_size
                        if t < tail:
                            value = completed * self.pool_size + t + offset
                    token_ids[b, row, slot] = value
                if lane == 0:
                    lengths[b, row] = n_valid * self.pool_size + tail

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
