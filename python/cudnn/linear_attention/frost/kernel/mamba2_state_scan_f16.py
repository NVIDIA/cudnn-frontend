# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint forward-state recomputation and reverse adjoint scan for SSD backward."""

import cutlass
from .mamba2_math import softplus
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ts
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, f16x2_to_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import cp_async_commit, cp_async_wait


class Mamba2StateScan:
    def __init__(self, batch, length, heads, groups, split_directions=False, reverse_only=False, early_tmem_release=False):
        self.batch, self.length, self.heads, self.groups = batch, length, heads, groups
        self.nchunks = (length + 31) // 32
        self.split_directions = split_directions
        self.reverse_only = reverse_only
        self.early_tmem_release = early_tmem_release

    @cute.jit
    def __call__(self, x, dy, dt, a, b, c, bias, initial, dfinal, seeds, adjoints, dinitial, stream: cuda.CUstream):
        if cutlass.const_expr(self.reverse_only):
            self.kernel(x, dy, dt, a, b, c, bias, initial, dfinal, seeds, adjoints, dinitial, 1).launch(
                grid=(self.batch * self.heads, 1, 1), block=(288, 1, 1), stream=stream
            )
        elif cutlass.const_expr(self.split_directions):
            # Compile-time directions avoid excessive live registers when both
            # optional initial-state tensors are present in the boundary case.
            for direction in cutlass.range_constexpr(2):
                self.kernel(x, dy, dt, a, b, c, bias, initial, dfinal, seeds, adjoints, dinitial, direction).launch(
                    grid=(self.batch * self.heads, 1, 1), block=(288, 1, 1), stream=stream
                )
        else:
            self.kernel(x, dy, dt, a, b, c, bias, initial, dfinal, seeds, adjoints, dinitial, -1).launch(
                grid=(2 * self.batch * self.heads, 1, 1), block=(288, 1, 1), stream=stream
            )

    @cute.kernel
    def kernel(self, x, dy, dt, a, b, c, bias, initial, dfinal, seeds, adjoints, dinitial, direction: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(direction >= 0):
            reverse = direction == 1
            bh = block
        else:
            reverse = block >= self.batch * self.heads
            bh = block % (self.batch * self.heads)
        batch, h = bh // self.heads, bh % self.heads
        group = h // (self.heads // self.groups)
        warp, lane = tid // 32, tid % 32
        sm = cutlass.AddressSpace.smem
        keys = cutlass.Array(cutlass.BFloat16, 3 * 2048, space=sm, alignment=1024)
        values = cutlass.Array(cutlass.BFloat16, 3 * 2048, space=sm, alignment=1024)
        weight = cutlass.Array(cutlass.Float32, 3 * 32, space=sm, alignment=128)
        decay = cutlass.Array(cutlass.Float32, 3, space=sm, alignment=16)
        slot = cutlass.Array(cutlass.Int32, 1, space=sm, alignment=16)
        raw = cutlass.Array(cutlass.Int64, 8, space=sm, alignment=16)
        ready = MBarrier(raw, 3, 128, Producer.THREAD)
        free = MBarrier(raw.subview(3), 3, 128, Producer.THREAD)
        operands = MBarrier(raw.subview(6), 1, 128, Producer.THREAD)
        updated = MBarrier(raw.subview(7), 1, 1, Producer.MMA_COMMIT)
        if tid == 0:
            for i in cutlass.range_constexpr(3):
                ready[i].init()
                free[i].init()
            operands.init()
            updated.init()
            nvvm.fence_mbarrier_init()
        if warp == 8:
            nvvm.tcgen05_alloc(slot, cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)
            if cutlass.const_expr(self.early_tmem_release):
                nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync_aligned()
        base = slot.load()
        if warp >= 4 and warp < 8:
            lt = tid - 128
            for step in cutlass.range(self.nchunks, unroll=1):
                stage, phase = step % 3, (step // 3) & 1
                chunk = self.nchunks - 1 - step if reverse else step
                if step >= 3:
                    free[stage].wait(phase ^ 1)
                for pack in cutlass.range_constexpr(2):
                    idx = lt * 8 + pack * 1024
                    row, col = idx // 64, idx % 64
                    token = chunk * 32 + row
                    valid = cutlass.Int32(token < self.length) * 16
                    ki = ((batch * self.length + token) * self.groups + group) * 64 + col
                    vi = ((batch * self.length + token) * self.heads + h) * 64 + col
                    dst = stage * 2048 + row * 64 + swizzle_xor_128b(row, col)
                    if reverse:
                        nvvm.cp_async_shared_global(keys.data_ptr() + dst, c.iterator.raw_ptr() + ki, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                        nvvm.cp_async_shared_global(values.data_ptr() + dst, dy.iterator.raw_ptr() + vi, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                    else:
                        nvvm.cp_async_shared_global(keys.data_ptr() + dst, b.iterator.raw_ptr() + ki, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                        nvvm.cp_async_shared_global(values.data_ptr() + dst, x.iterator.raw_ptr() + vi, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                cp_async_commit()
                if warp == 4:
                    token = chunk * 32 + lane
                    delta = cutlass.Float32(0)
                    if token < self.length:
                        delta = dt[(batch * self.length + token) * self.heads + h].to(cutlass.Float32)
                        if cutlass.const_expr(bias is not None):
                            delta += bias[h]
                        delta = softplus(delta)
                    p = delta * a[h] * 1.4426950408889634
                    for off in [1, 2, 4, 8, 16]:
                        v = nvvm.shfl_sync(0xFFFFFFFF, p, off, 0, kind=nvvm.Shfl.UP)
                        if lane >= off:
                            p += v
                    end = nvvm.shfl_sync(0xFFFFFFFF, p, 31, 31, kind=nvvm.Shfl.IDX)
                    w = cute.math.exp2(p, fastmath=True) if reverse else delta * cute.math.exp2(end - p, fastmath=True)
                    weight[stage * 32 + lane] = w
                    if lane == 0:
                        decay[stage] = cute.math.exp2(end, fastmath=True)
                cp_async_wait(0)
                # Release every producer's writes directly to the consumers.
                ready[stage].arrive()
        elif warp < 4:
            row = warp * 16 + lane % 16
            for part in cutlass.range_constexpr(2):
                initial_values = []
                for j in cutlass.range_constexpr(16):
                    r0 = warp * 16 + lane // 4 + (j // 2 % 2) * 8
                    col = part * 32 + lane % 4 * 2 + j // 4 * 8 + j % 2
                    value = cutlass.Float32(0)
                    if reverse:
                        if cutlass.const_expr(dfinal is not None):
                            value = dfinal[bh * 4096 + r0 * 64 + col]
                    else:
                        if cutlass.const_expr(initial is not None):
                            value = initial[bh * 4096 + r0 * 64 + col]
                    initial_values.append(value)
                nvvm.tcgen05_st(
                    "16x256b", nvvm.make_tmem_ptr(base + part * 32, cutlass.Float32), cutlass.Vector.from_elements(tuple(initial_values), cutlass.Float32)
                )
            nvvm.tcgen05_wait("store")
            vr = tid % 8 + (tid // 16 % 2) * 8
            vc = warp * 16 + (tid // 8 % 2) * 8
            for step in cutlass.range(self.nchunks, unroll=1):
                stage, phase = step % 3, step & 1
                chunk = self.nchunks - 1 - step if reverse else step
                ready[stage].wait((step // 3) & 1)
                if step > 0:
                    updated.wait(phase ^ 1)
                state = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), num=8)
                nvvm.tcgen05_wait("load")
                for j in cutlass.range_constexpr(16):
                    r = warp * 16 + lane // 4 + 8 * (j % 2)
                    col = (lane % 4) * 2 + (j // 2) * 8
                    ci = (bh * self.nchunks + chunk) * 4096 + r * 64 + col
                    pair = cutlass.Vector.from_elements((state[2 * j], state[2 * j + 1]), cutlass.Float32).to(seeds.element_type)
                    if reverse:
                        (adjoints.iterator.raw_ptr() + ci).store(pair, alignment=seeds.element_type.width // 4)
                    else:
                        (seeds.iterator.raw_ptr() + ci).store(pair, alignment=seeds.element_type.width // 4)
                nvvm.tcgen05_st("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), state * decay[stage])
                xwords = []
                for part in cutlass.range_constexpr(2):
                    words = nvvm.ldmatrix(
                        values.data_ptr() + stage * 2048 + (vr + part * 16) * 64 + swizzle_xor_128b(vr + part * 16, vc), 4, nvvm.MMALayout.COL
                    )
                    for j in cutlass.range_constexpr(4):
                        xwords.append(words[j])
                weighted = []
                for j in cutlass.range_constexpr(8):
                    v0, v1 = f16x2_to_f32(xwords[j], dtype=cutlass.BFloat16)
                    col = (lane % 4) * 2 + (j // 2) * 8
                    weighted.append(fp32_to_fp16(v0 * weight[stage * 32 + col], v1 * weight[stage * 32 + col + 1], dtype=cutlass.BFloat16))
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(base + 64, cutlass.Int32), cutlass.Vector.from_elements(tuple(weighted), cutlass.Int32))
                nvvm.tcgen05_wait("store")
                operands.arrive()
                updated.wait(phase)
                free[stage].arrive()
            if reverse:
                for part in cutlass.range_constexpr(2):
                    final_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + part * 32, cutlass.Float32), num=4)
                    nvvm.tcgen05_wait("load")
                    for j in cutlass.range_constexpr(8):
                        r0 = warp * 16 + lane // 4 + (j % 2) * 8
                        col = part * 32 + lane % 4 * 2 + j // 2 * 8
                        pair = cutlass.Vector.from_elements((final_vec[2 * j], final_vec[2 * j + 1]), cutlass.Float32)
                        (dinitial.iterator.raw_ptr() + bh * 4096 + r0 * 64 + col).store(pair, alignment=8)
        else:
            desc = MmaDesc(
                M=64,
                N=64,
                K=32,
                bpe_a=2,
                bpe_b=2,
                tile_k_hw=16,
                btranspose=True,
                idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, n_dim=64, m_dim=64, b_major=1),
                kind=nvvm.Tcgen05MMAKind.F16,
            )
            for step in cutlass.range(self.nchunks, unroll=1):
                stage = step % 3
                operands.wait(step & 1)
                kd = SmemTile(keys.data_ptr() + stage * 2048, 2048, 4096, 1024, 2).desc()
                mma_ts(desc, nvvm.make_tmem_ptr(base + 64, cutlass.Int8), kd, nvvm.make_tmem_ptr(base, cutlass.Float32), accumulate=True)
                if nvvm.elect_sync():
                    updated.arrive(cta_group=1)
        nvvm.barrier_cta_sync_aligned()
        if warp == 8:
            if cutlass.const_expr(not self.early_tmem_release):
                nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(base, cutlass.Float32), cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
