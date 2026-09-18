# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shared-head and parameter reductions for SSD backward."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda


class Mamba2BackwardReduce:
    def __init__(self, batch, length, heads, groups, mode=3):
        self.tokens, self.heads, self.groups = batch * length, heads, groups
        self.mode = mode
        self.batch, self.nchunks = batch, (length + 31) // 32

    @cute.jit
    def __call__(self, dbp, dcp, dap, ddp, dbiasp, db, dc, da, dd, dbias, stream: cuda.CUstream):
        if cutlass.const_expr(self.mode & 1):
            self.bc(dbp, dcp, db, dc).launch(grid=(self.tokens * self.groups, 1, 1), block=(128, 1, 1), stream=stream)
        if cutlass.const_expr(self.mode & 2):
            self.params(dap, ddp, dbiasp, da, dd, dbias).launch(grid=(self.heads, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def bc(self, dbp, dcp, db, dc):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        token, group = block // self.groups, block % self.groups
        dim, part = tid % 64, tid // 64
        hs = self.heads // self.groups
        vb, vc = cutlass.Float32(0), cutlass.Float32(0)
        for i in cutlass.range_constexpr((hs + 1) // 2):
            h = i * 2 + part
            if h < hs:
                idx = (token * self.heads + group * hs + h) * 64 + dim
                vb += dbp[idx].to(cutlass.Float32)
                vc += dcp[idx].to(cutlass.Float32)
        sb = cutlass.Array(cutlass.Float32, 128, space=cutlass.AddressSpace.smem, alignment=128)
        sc = cutlass.Array(cutlass.Float32, 128, space=cutlass.AddressSpace.smem, alignment=128)
        sb[tid], sc[tid] = vb, vc
        nvvm.barrier_cta_sync_aligned()
        if tid < 64:
            db[block * 64 + tid] = (sb[tid] + sb[tid + 64]).to(cutlass.BFloat16)
            dc[block * 64 + tid] = (sc[tid] + sc[tid + 64]).to(cutlass.BFloat16)

    bc.set_name_prefix("cudnn", remove_cutlass_symbol=True)

    @cute.kernel
    def params(self, dap, ddp, dbiasp, da, dd, dbias):
        tid, _, _ = cute.arch.thread_idx()
        h, _, _ = cute.arch.block_idx()
        va, vd, vb = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
        for step in cutlass.range((self.batch * self.nchunks + 127) // 128, unroll=1):
            partial = step * 128 + tid
            if partial < self.batch * self.nchunks:
                batch, chunk = partial // self.nchunks, partial % self.nchunks
                idx = (batch * self.heads + h) * self.nchunks + chunk
                va += dap[idx]
                vd += ddp[idx]
                vb += dbiasp[idx]
        for off in [16, 8, 4, 2, 1]:
            va += nvvm.shfl_sync(0xFFFFFFFF, va, off, 31, kind=nvvm.Shfl.BFLY)
            vb += nvvm.shfl_sync(0xFFFFFFFF, vb, off, 31, kind=nvvm.Shfl.BFLY)
            vd += nvvm.shfl_sync(0xFFFFFFFF, vd, off, 31, kind=nvvm.Shfl.BFLY)
        sa = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
        sd = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
        sb = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.smem, alignment=16)
        if tid % 32 == 0:
            sa[tid // 32], sd[tid // 32], sb[tid // 32] = va, vd, vb
        nvvm.barrier_cta_sync_aligned()
        if tid == 0:
            va, vd, vb = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
            for i in cutlass.range_constexpr(4):
                va += sa[i]
                vd += sd[i]
                vb += sb[i]
            da[h], dd[h], dbias[h] = va, vd, vb

    params.set_name_prefix("cudnn", remove_cutlass_symbol=True)
