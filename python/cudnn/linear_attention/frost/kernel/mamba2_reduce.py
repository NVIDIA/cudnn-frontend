# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic shared-head and parameter reductions for SSD backward."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda


class Mamba2BackwardReduce:
    def __init__(self, batch, length, heads, groups, mode=3, chunk_size=32, state_tiles=2):
        self.tokens, self.heads, self.groups = batch * length, heads, groups
        self.mode = mode
        self.state_tiles = state_tiles
        self.batch, self.nchunks = batch, (length + chunk_size - 1) // chunk_size

    @cute.jit
    def __call__(self, dbp, dcp, dap, ddp, dbiasp, db, dc, da, dd, dbias, dxp, ddtp, dx, ddt, stream: cuda.CUstream):
        if cutlass.const_expr(self.mode & 1):
            self.bc(dbp, dcp, db, dc).launch(grid=((self.tokens * self.groups + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)
        if cutlass.const_expr(dxp is not None):
            self.partials(dxp, ddtp, dx, ddt).launch(grid=((self.tokens * self.heads * 64 + 1023) // 1024, 1, 1), block=(128, 1, 1), stream=stream)
        if cutlass.const_expr(self.mode & 2):
            self.params(dap, ddp, dbiasp, da, dd, dbias).launch(grid=(self.heads, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def bc(self, dbp, dcp, db, dc):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        # Four (token, group) rows per CTA, four adjacent state columns per
        # thread. Each thread owns the full head reduction, so no SMEM handoff
        # or separate state-tile CTA is needed.
        logical = block * 4 + tid // 32
        dim = (tid % 32) * 4
        token, group = logical // self.groups, logical % self.groups
        if token < self.tokens:
            hs = self.heads // self.groups
            vb = cutlass.Vector.from_elements(tuple(cutlass.Float32(0) for _ in range(4)), cutlass.Float32)
            vc = vb
            for h in cutlass.range_constexpr(hs):
                idx = (token * self.heads + group * hs + h) * 128 + dim
                vb += (dbp.iterator.raw_ptr() + idx).load(count=4, alignment=8).to(cutlass.Float32)
                vc += (dcp.iterator.raw_ptr() + idx).load(count=4, alignment=8).to(cutlass.Float32)
            offset = logical * 128 + dim
            (db.iterator.raw_ptr() + offset).store(vb.to(cutlass.BFloat16), alignment=8)
            (dc.iterator.raw_ptr() + offset).store(vc.to(cutlass.BFloat16), alignment=8)

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
                vd += ddp[idx]
                if cutlass.const_expr(self.state_tiles == 2):
                    va += dap[idx * 2] + dap[idx * 2 + 1]
                    vb += dbiasp[idx * 2] + dbiasp[idx * 2 + 1]
                else:
                    va += dap[idx]
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

    @cute.kernel
    def partials(self, dxp, ddtp, dx, ddt):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        idx = (block * 128 + tid) * 8
        if idx < self.tokens * self.heads * 64:
            row, col = idx // 64, idx % 64
            offset = row * 128 + col
            high = (dxp.iterator.raw_ptr() + offset).load(count=8, alignment=16)
            low = (dxp.iterator.raw_ptr() + offset + 64).load(count=8, alignment=16)
            (dx.iterator.raw_ptr() + idx).store((high + low).to(cutlass.BFloat16), alignment=16)
            if col == 0:
                ddt[row] = (ddtp[row * 2] + ddtp[row * 2 + 1]).to(cutlass.BFloat16)

    partials.set_name_prefix("cudnn", remove_cutlass_symbol=True)
