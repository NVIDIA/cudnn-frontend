# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training-side gated pooling and its backward: one warp per pool, four channels per lane."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32

LOG2E = 1.4426950408889634
WARPS = 4


class PoolKeysKernel:
    def __init__(self, pool_size, out_dtype):
        self.pool_size = pool_size
        self.out_dtype = out_dtype

    @cute.jit
    def __call__(self, keys, gates, ape, out, blocks: Int32, stride: Int32, iters: Int32, stream: cuda.CUstream):
        self.kernel(keys, gates, ape, out, stride, iters).launch(grid=(blocks, 1, 1), block=(WARPS * 32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, keys, gates, ape, out, stride: Int32, iters: Int32):
        warp = cute.arch.block_idx()[0] * WARPS + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        n_pools = cute.size(out.shape[1])
        total = cute.size(out.shape[0]) * n_pools
        scores = cute.make_rmem_tensor((self.pool_size,), Float32)
        for it in range(0, iters):
            task = warp + it * stride
            if task < total:
                b = task // n_pools
                base = (task % n_pools) * self.pool_size
                for j in cutlass.range_constexpr(4):
                    dim = lane + j * 32
                    largest = Float32(-float("inf"))
                    for i in cutlass.range_constexpr(self.pool_size):
                        s = Float32(gates[b, base + i, dim]) + ape[i, dim]
                        scores[i] = s
                        largest = cute.arch.fmax(largest, s)
                    num, den = Float32(0), Float32(0)
                    for i in cutlass.range_constexpr(self.pool_size):
                        p = cute.math.exp2((scores[i] - largest) * LOG2E, fastmath=True)
                        num += p * Float32(keys[b, base + i, dim])
                        den += p
                    value = num / den
                    if cutlass.const_expr(self.out_dtype == "bf16"):
                        out[b, task % n_pools, dim] = value.to(BFloat16)
                    else:
                        out[b, task % n_pools, dim] = value

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


class UnpoolBackwardKernel:
    def __init__(self, pool_size, grad_dtype):
        self.pool_size = pool_size
        self.grad_dtype = grad_dtype

    @cute.jit
    def __call__(self, keys, gates, ape, d_pooled, d_keys, d_gates, ape_partials, blocks: Int32, stride: Int32, iters: Int32, stream: cuda.CUstream):
        self.kernel(keys, gates, ape, d_pooled, d_keys, d_gates, ape_partials, stride, iters).launch(
            grid=(blocks, 1, 1), block=(WARPS * 32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, keys, gates, ape, d_pooled, d_keys, d_gates, ape_partials, stride: Int32, iters: Int32):
        warp = cute.arch.block_idx()[0] * WARPS + cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        n_pools = cute.size(d_pooled.shape[1])
        total = cute.size(d_pooled.shape[0]) * n_pools
        scores = cute.make_rmem_tensor((self.pool_size,), Float32)
        values = cute.make_rmem_tensor((self.pool_size,), Float32)
        acc = cute.make_rmem_tensor((self.pool_size * 4,), Float32)
        for i in cutlass.range_constexpr(self.pool_size * 4):
            acc[i] = Float32(0)
        for it in range(0, iters):
            task = warp + it * stride
            if task < total:
                b = task // n_pools
                pool = task % n_pools
                base = pool * self.pool_size
                for j in cutlass.range_constexpr(4):
                    dim = lane + j * 32
                    largest = Float32(-float("inf"))
                    for i in cutlass.range_constexpr(self.pool_size):
                        s = Float32(gates[b, base + i, dim]) + ape[i, dim]
                        scores[i] = s
                        values[i] = Float32(keys[b, base + i, dim])
                        largest = cute.arch.fmax(largest, s)
                    den, mean = Float32(0), Float32(0)
                    for i in cutlass.range_constexpr(self.pool_size):
                        p = cute.math.exp2((scores[i] - largest) * LOG2E, fastmath=True)
                        scores[i] = p
                        den += p
                        mean += p * values[i]
                    inv = Float32(1) / den
                    mean = mean * inv
                    g = Float32(d_pooled[b, pool, dim])
                    for i in cutlass.range_constexpr(self.pool_size):
                        alpha = scores[i] * inv
                        dk = alpha * g
                        de = dk * (values[i] - mean)
                        acc[i * 4 + j] += de
                        if cutlass.const_expr(self.grad_dtype == "bf16"):
                            d_keys[b, base + i, dim] = dk.to(BFloat16)
                            d_gates[b, base + i, dim] = de.to(BFloat16)
                        else:
                            d_keys[b, base + i, dim] = dk
                            d_gates[b, base + i, dim] = de
        for i in cutlass.range_constexpr(self.pool_size):
            for j in cutlass.range_constexpr(4):
                ape_partials[warp, i, lane + j * 32] = acc[i * 4 + j]

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
