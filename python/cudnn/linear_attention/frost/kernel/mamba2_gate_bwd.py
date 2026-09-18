# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SiLU gate backward with FP32 dD partials before rounding the SSD gradient."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda


class Mamba2GateBackward:
    def __init__(self, batch, length, heads):
        self.batch, self.length, self.heads = batch, length, heads
        self.nchunks = (length + 31) // 32

    @cute.jit
    def __call__(self, x, z, out, dy, dx, dz, ddp, stream: cuda.CUstream):
        self.kernel(x, z, out, dy, dx, dz, ddp).launch(grid=(self.batch * self.heads * self.nchunks, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x, z, out, dy, dx, dz, ddp):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        bh, chunk = block // self.nchunks, block % self.nchunks
        batch, h = bh // self.heads, bh % self.heads
        partial = cutlass.Float32(0)
        for i in cutlass.range_constexpr(2):
            row, dim = tid // 8 + i * 16, tid % 8 * 8
            token = chunk * 32 + row
            if token < self.length:
                idx = ((batch * self.length + token) * self.heads + h) * 64 + dim
                zv = (z.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
                grad = (dy.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
                values = (x.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
                outputs = (out.iterator.raw_ptr() + idx).load(count=8, alignment=16).to(cutlass.Float32)
                dxs, dzs = [], []
                for j in cutlass.range_constexpr(8):
                    sigmoid = 1.0 / (1.0 + cute.math.exp(-zv[j], fastmath=True))
                    gated_grad = grad[j] * zv[j] * sigmoid
                    dxs.append(gated_grad.to(cutlass.BFloat16))
                    dzs.append((grad[j] * outputs[j] * sigmoid * (1.0 + zv[j] * (1.0 - sigmoid))).to(cutlass.BFloat16))
                    partial += gated_grad * values[j]
                (dx.iterator.raw_ptr() + idx).store(cutlass.Vector.from_elements(tuple(dxs), cutlass.BFloat16), alignment=16)
                (dz.iterator.raw_ptr() + idx).store(cutlass.Vector.from_elements(tuple(dzs), cutlass.BFloat16), alignment=16)
        for off in [16, 8, 4, 2, 1]:
            partial += nvvm.shfl_sync(0xFFFFFFFF, partial, off, 31, kind=nvvm.Shfl.BFLY)
        shared = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
        if tid % 32 == 0:
            shared[tid // 32] = partial
        nvvm.barrier_cta_sync_aligned()
        if tid == 0:
            total = cutlass.Float32(0)
            for i in cutlass.range_constexpr(4):
                total += shared[i]
            ddp[block] = total

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
