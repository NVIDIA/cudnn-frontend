# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expand pools and append the causal tail into caller-owned token indices."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


class ExpandIndicesKernel:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.threads = 256

    @cute.jit
    def __call__(self, pools: cute.Tensor, positions: cute.Tensor, output: cute.Tensor, row_map: cute.Tensor | None, stream: cuda.CUstream):
        self.kernel(pools, positions, output, row_map).launch(
            grid=(cute.size(output.shape[0]), cute.ceil_div(cute.size(output.shape[1]), self.threads), 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, pools: cute.Tensor, positions: cute.Tensor, output: cute.Tensor, row_map: cute.Tensor | None):
        row = cute.arch.block_idx()[0]
        col = cute.arch.block_idx()[1] * self.threads + cute.arch.thread_idx()[0]
        if col < cute.size(output.shape[1]):
            token = Int64(-1)
            if row < cute.size(positions.shape[0]):
                source_row = Int64(row)
                if cutlass.const_expr(row_map is not None):
                    source_row = Int64(row_map[row])
                if source_row >= 0 and source_row < cute.size(pools.shape[0]):
                    position = Int64(positions[row])
                    if position >= 0 and position < 2147483647:
                        length = position + 1
                        completed = length // self.pool_size
                        topk = cute.size(pools.shape[1]) * self.pool_size
                        if length <= topk:
                            if col < length:
                                token = Int64(col)
                        elif col < topk:
                            pool = Int64(pools[source_row, col // self.pool_size])
                            if pool >= 0 and pool < completed:
                                token = pool * self.pool_size + col % self.pool_size
                        else:
                            tail = Int64(col) - topk
                            if tail < length % self.pool_size:
                                token = completed * self.pool_size + tail
            output[row, col] = Int32(token)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
