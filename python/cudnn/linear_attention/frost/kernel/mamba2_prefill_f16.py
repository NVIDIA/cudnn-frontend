# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSD specialization of GDN's warp roles and TMEM state path, logical BT=32.

Four compute warps form causal scores, four update state/output, four load
three input stages, and one issues asynchronous tensor-core work. The delta
rule's residual, triangular inverse and inverse-times-value GEMM are absent.
The state conversion and transposed output layouts follow gdn_prefill_f16.
"""

import cutlass
from .mamba2_math import softplus
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda

from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, f16x2_to_f32
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import cp_async_commit, cp_async_wait


class Mamba2Prefill:
    def __init__(self, batch, length, heads, groups, chunk=32, warps=13, parallel_scores=False, gate_in_loader=False):
        if chunk != 32:
            raise ValueError("GDN-derived SSD specializes logical chunk=32")
        self.batch, self.length, self.heads, self.groups = batch, length, heads, groups
        self.nchunks = (length + 31) // 32
        self.parallel_scores = parallel_scores
        self.gate_in_loader = gate_in_loader

    @cute.jit
    def __call__(self, x, dt, a, b, c, d, bias, z, initial, out, final, ungated_out, checkpoints, stream: cuda.CUstream):
        self.kernel(x, dt, a, b, c, d, bias, z, initial, out, final, ungated_out, checkpoints).launch(
            grid=(self.batch * self.heads, 1, 1), block=(416, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, x, dt, a, b, c, d, bias, z, initial, out, final, ungated_out, checkpoints):
        tid, _, _ = cute.arch.thread_idx()
        bh, _, _ = cute.arch.block_idx()
        batch, h = bh // self.heads, bh % self.heads
        group = h // (self.heads // self.groups)
        warp, lane = tid // 32, tid % 32
        sm = cutlass.AddressSpace.smem
        kq = cutlass.Array(cutlass.BFloat16, 3 * 4096, space=sm, alignment=1024)
        vx = cutlass.Array(cutlass.BFloat16, 3 * 2048, space=sm, alignment=1024)
        att = cutlass.Array(cutlass.BFloat16, 3 * 2048, space=sm, alignment=1024)
        yo = cutlass.Array(out.element_type, 2048, space=sm, alignment=1024)
        if cutlass.const_expr(z is not None):
            zg = cutlass.Array(cutlass.BFloat16, 3 * 2048, space=sm, alignment=1024)
            if cutlass.const_expr(self.gate_in_loader):
                za = cutlass.Array(cutlass.Float32, 3 * 2048, space=sm, alignment=1024)
        if cutlass.const_expr(ungated_out is not None):
            yu = cutlass.Array(cutlass.BFloat16, 2048, space=sm, alignment=1024)
        prefix = cutlass.Array(cutlass.Float32, 3 * 32, space=sm, alignment=128)
        delta = cutlass.Array(cutlass.Float32, 3 * 32, space=sm, alignment=128)
        prod = cutlass.Array(cutlass.Float32, 3 * 32, space=sm, alignment=128)
        weight = cutlass.Array(cutlass.Float32, 3 * 32, space=sm, alignment=128)
        slot = cutlass.Array(cutlass.Int32, 1, space=sm, alignment=16)
        mb = cutlass.Array(cutlass.Int64, 27, space=sm, alignment=16)
        inputs_ready = MBarrier(mb, 3, 128, Producer.THREAD)
        inputs_done = MBarrier(mb.subview(3), 3, 128, Producer.THREAD)
        scores_ready = MBarrier(mb.subview(6), 2, 1, Producer.MMA_COMMIT)
        scores_done = MBarrier(mb.subview(8), 2, 128, Producer.THREAD)
        a_ready = MBarrier(mb.subview(10), 3, 128, Producer.THREAD)
        a_done = MBarrier(mb.subview(13), 3, 1, Producer.MMA_COMMIT)
        state_ready = MBarrier(mb.subview(16), 1, 128, Producer.THREAD)
        x_ready = MBarrier(mb.subview(17), 1, 128, Producer.THREAD)
        inter_ready = MBarrier(mb.subview(18), 1, 1, Producer.MMA_COMMIT)
        inter_scaled = MBarrier(mb.subview(19), 1, 128, Producer.THREAD)
        out_ready = MBarrier(mb.subview(20), 1, 1, Producer.MMA_COMMIT)
        state_done = MBarrier(mb.subview(21), 1, 1, Producer.MMA_COMMIT)
        if cutlass.const_expr(z is not None and self.gate_in_loader):
            gate_ready = MBarrier(mb.subview(22), 3, 128, Producer.THREAD)
        if tid == 0:
            for i in cutlass.range_constexpr(3):
                inputs_ready[i].init()
                inputs_done[i].init()
                a_ready[i].init()
                a_done[i].init()
                if cutlass.const_expr(z is not None and self.gate_in_loader):
                    gate_ready[i].init()
            for i in cutlass.range_constexpr(2):
                scores_ready[i].init()
                scores_done[i].init()
            state_ready.init()
            x_ready.init()
            inter_ready.init()
            inter_scaled.init()
            out_ready.init()
            state_done.init()
            nvvm.fence_mbarrier_init()
        if warp == 12:
            nvvm.tcgen05_alloc(slot, cutlass.Int32(256), group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync_aligned()
        base = slot.load()
        # TMEM: state fp32 [0,64), bf16 state [64,96), output [96,128),
        # X [128,144), weighted X [144,160), two score slots [160,224).
        if warp >= 8 and warp < 12:
            lt = tid - 256
            for chunk in cutlass.range(self.nchunks, unroll=1):
                stage = chunk % 3
                phase = (chunk // 3) & 1
                if chunk >= 3:
                    inputs_done[stage].wait(phase ^ 1)
                for pack in cutlass.range_constexpr(2):
                    idx = lt * 8 + pack * 1024
                    row, col = idx // 64, idx % 64
                    token = chunk * 32 + row
                    bcidx = ((batch * self.length + token) * self.groups + group) * 64 + col
                    xidx = ((batch * self.length + token) * self.heads + h) * 64 + col
                    valid = cutlass.Int32(token < self.length) * 16
                    ki = stage * 4096 + row * 64 + swizzle_xor_128b(row, col)
                    qi = stage * 4096 + (row + 32) * 64 + swizzle_xor_128b(row + 32, col)
                    vi = stage * 2048 + row * 64 + swizzle_xor_128b(row, col)
                    nvvm.cp_async_shared_global(kq.data_ptr() + ki, b.iterator.raw_ptr() + bcidx, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                    nvvm.cp_async_shared_global(kq.data_ptr() + qi, c.iterator.raw_ptr() + bcidx, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                    nvvm.cp_async_shared_global(vx.data_ptr() + vi, x.iterator.raw_ptr() + xidx, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                    if cutlass.const_expr(z is not None):
                        nvvm.cp_async_shared_global(zg.data_ptr() + vi, z.iterator.raw_ptr() + xidx, 16, nvvm.LoadCacheModifier.CA, cp_size=valid)
                cp_async_commit()
                if warp == 8:
                    tok = chunk * 32 + lane
                    dv = cutlass.Float32(0)
                    if tok < self.length:
                        dv = dt[(batch * self.length + tok) * self.heads + h].to(cutlass.Float32)
                        if cutlass.const_expr(bias is not None):
                            dv += bias[h]
                        dv = softplus(dv)
                    pv = dv * a[h] * 1.4426950408889634
                    for offset in [1, 2, 4, 8, 16]:
                        prev = nvvm.shfl_sync(0xFFFFFFFF, pv, offset, 0, kind=nvvm.Shfl.UP)
                        if lane >= offset:
                            pv += prev
                    end = nvvm.shfl_sync(0xFFFFFFFF, pv, 31, 31, kind=nvvm.Shfl.IDX)
                    prefix[stage * 32 + lane] = pv
                    delta[stage * 32 + lane] = dv
                    prod[stage * 32 + lane] = cute.math.exp2(pv, fastmath=True)
                    weight[stage * 32 + lane] = dv * cute.math.exp2(end - pv, fastmath=True)
                cp_async_wait(0)
                # Release matrix operands independently of gate activation.
                # Each producer publishes its own scalar and async-copy writes.
                # A leader-only arrive after a subset barrier triggered races.
                inputs_ready[stage].arrive()
                if cutlass.const_expr(z is not None and self.gate_in_loader):
                    for pack in cutlass.range_constexpr(2):
                        idx = lt * 8 + pack * 1024
                        row, col = idx // 64, idx % 64
                        zi = stage * 2048 + row * 64 + swizzle_xor_128b(row, col)
                        raw_gate = (zg.data_ptr() + zi).load(count=8, alignment=16).to(cutlass.Float32)
                        activations = []
                        for j in cutlass.range_constexpr(8):
                            value = raw_gate[j]
                            activations.append(value / (1.0 + cute.math.exp(-value, fastmath=True)))
                        (za.data_ptr() + zi).store(cutlass.Vector.from_elements(tuple(activations), cutlass.Float32), alignment=32)
                    gate_ready[stage].arrive()
        elif warp < 4:
            row = warp * 16 + lane % 16
            for chunk in cutlass.range(self.nchunks, unroll=1):
                stage, astage = chunk % 3, chunk % 2
                inputs_ready[stage].wait((chunk // 3) & 1)
                scores_ready[astage].wait((chunk // 2) & 1)
                if cutlass.const_expr(self.parallel_scores):
                    scorev = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + 160 + astage * 32, cutlass.Float32), num=4)
                    nvvm.tcgen05_wait("load")
                    scores_done[astage].arrive()
                    if chunk >= 3:
                        a_done[stage].wait(((chunk // 3) & 1) ^ 1)
                    if warp >= 2:
                        for j in cutlass.range_constexpr(8):
                            qr = (warp - 2) * 16 + lane // 4 + (j % 2) * 8
                            col0 = lane % 4 * 2 + j // 2 * 8
                            pi = prefix[stage * 32 + qr]
                            pair_values = []
                            for sub in cutlass.range_constexpr(2):
                                col = col0 + sub
                                value = cutlass.Float32(0)
                                if qr >= col:
                                    value = scorev[2 * j + sub] * delta[stage * 32 + col] * cute.math.exp2(pi - prefix[stage * 32 + col], fastmath=True)
                                pair_values.append(value.to(cutlass.BFloat16))
                            ai = stage * 2048 + qr * 64 + swizzle_xor_128b(qr, col0)
                            (att.data_ptr() + ai).store(cutlass.Vector.from_elements(tuple(pair_values), cutlass.BFloat16), alignment=4)
                else:
                    sv = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 160 + astage * 32, cutlass.Float32), num=32)
                    nvvm.tcgen05_wait("load")
                    scores_done[astage].arrive()
                    if chunk >= 3:
                        a_done[stage].wait(((chunk // 3) & 1) ^ 1)
                    if row >= 32 and lane < 16:
                        qr = row - 32
                        pi = prefix[stage * 32 + qr]
                        for pack in cutlass.range_constexpr(4):
                            vals = []
                            for j in cutlass.range_constexpr(8):
                                col = pack * 8 + j
                                value = cutlass.Float32(0)
                                if qr >= col:
                                    value = sv[col] * delta[stage * 32 + col] * cute.math.exp2(pi - prefix[stage * 32 + col], fastmath=True)
                                vals.append(value.to(cutlass.BFloat16))
                            ai = stage * 2048 + qr * 64 + swizzle_xor_128b(qr, pack * 8)
                            (att.data_ptr() + ai).store(cutlass.Vector.from_elements(tuple(vals), cutlass.BFloat16), alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                a_ready[stage].arrive()
        elif warp < 8:
            ct = tid - 128
            row = (ct // 32) * 16 + lane % 16
            seed = []
            for j in cutlass.range_constexpr(32):
                r0 = (warp - 4) * 16 + lane // 4 + (j // 2 % 2) * 8
                col = lane % 4 * 2 + j // 4 * 8 + j % 2
                value = cutlass.Float32(0)
                if cutlass.const_expr(initial is not None):
                    value = initial[bh * 4096 + r0 * 64 + col]
                seed.append(value)
            nvvm.tcgen05_st("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), cutlass.Vector.from_elements(tuple(seed), cutlass.Float32))
            nvvm.tcgen05_wait("store")
            vo_row = ct % 8 + (ct // 16 % 2) * 8
            vo_col = (ct // 32) * 16 + (ct // 8 % 2) * 8
            for chunk in cutlass.range(self.nchunks, unroll=1):
                stage, phase = chunk % 3, chunk & 1
                inputs_ready[stage].wait((chunk // 3) & 1)
                if chunk > 0:
                    state_done.wait(phase ^ 1)
                statev = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), num=8)
                nvvm.tcgen05_wait("load")
                if cutlass.const_expr(checkpoints is not None):
                    for j in cutlass.range_constexpr(16):
                        r0 = (warp - 4) * 16 + lane // 4 + (j % 2) * 8
                        col = lane % 4 * 2 + j // 2 * 8
                        idx = (bh * self.nchunks + chunk) * 4096 + r0 * 64 + col
                        pair = cutlass.Vector.from_elements((statev[2 * j], statev[2 * j + 1]), cutlass.Float32).to(checkpoints.element_type)
                        (checkpoints.iterator.raw_ptr() + idx).store(pair, alignment=checkpoints.element_type.width // 4)
                packs = [fp32_to_fp16(statev[2 * j], statev[2 * j + 1], dtype=cutlass.BFloat16) for j in range(16)]
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(base + 64, cutlass.Int32), cutlass.Vector.from_elements(tuple(packs), cutlass.Int32))
                decay = prod[stage * 32 + 31]
                nvvm.tcgen05_st("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), statev * decay)
                nvvm.tcgen05_wait("store")
                state_ready.arrive()
                xv = []
                for block in cutlass.range_constexpr(2):
                    vv = nvvm.ldmatrix(
                        vx.data_ptr() + stage * 2048 + (vo_row + block * 16) * 64 + swizzle_xor_128b(vo_row + block * 16, vo_col), 4, nvvm.MMALayout.COL
                    )
                    for j in cutlass.range_constexpr(4):
                        xv.append(vv[j])
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(base + 128, cutlass.Int32), cutlass.Vector.from_elements(tuple(xv), cutlass.Int32))
                weighted = []
                for j in cutlass.range_constexpr(8):
                    l0, l1 = f16x2_to_f32(xv[j], dtype=cutlass.BFloat16)
                    col = (lane % 4) * 2 + ((2 * j // 4) * 8)
                    weighted.append(fp32_to_fp16(l0 * weight[stage * 32 + col], l1 * weight[stage * 32 + col + 1], dtype=cutlass.BFloat16))
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(base + 144, cutlass.Int32), cutlass.Vector.from_elements(tuple(weighted), cutlass.Int32))
                nvvm.tcgen05_wait("store")
                x_ready.arrive()
                inter_ready.wait(phase)
                ov = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + 96, cutlass.Float32), num=4)
                nvvm.tcgen05_wait("load")
                scaled = []
                for j in cutlass.range_constexpr(16):
                    col = (lane % 4) * 2 + (j // 4) * 8 + j % 2
                    scaled.append(ov[j] * prod[stage * 32 + col])
                nvvm.tcgen05_st("16x256b", nvvm.make_tmem_ptr(base + 96, cutlass.Float32), cutlass.Vector.from_elements(tuple(scaled), cutlass.Float32))
                nvvm.tcgen05_wait("store")
                inter_scaled.arrive()
                out_ready.wait(phase)
                if cutlass.const_expr(z is not None and self.gate_in_loader):
                    gate_ready[stage].wait((chunk // 3) & 1)
                ov = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + 96, cutlass.Float32), num=4)
                nvvm.tcgen05_wait("load")
                gate_words = []
                if cutlass.const_expr(z is not None and not self.gate_in_loader):
                    for block in cutlass.range_constexpr(2):
                        gate_pack = nvvm.ldmatrix(
                            zg.data_ptr() + stage * 2048 + (vo_row + block * 16) * 64 + swizzle_xor_128b(vo_row + block * 16, vo_col),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for j in cutlass.range_constexpr(4):
                            gate_words.append(gate_pack[j])
                epilogue = []
                pre_gate = []
                for j in cutlass.range_constexpr(8):
                    x0, x1 = f16x2_to_f32(xv[j], dtype=cutlass.BFloat16)
                    y0, y1 = ov[2 * j], ov[2 * j + 1]
                    if cutlass.const_expr(d is not None):
                        y0 += d[h] * x0
                        y1 += d[h] * x1
                    if cutlass.const_expr(z is not None):
                        if cutlass.const_expr(ungated_out is not None):
                            pre_gate.append(fp32_to_fp16(y0, y1, dtype=cutlass.BFloat16))
                        if cutlass.const_expr(self.gate_in_loader):
                            pr = (ct // 32) * 16 + lane // 4 + 8 * (j % 2)
                            zr = (lane % 4) * 2 + (j // 2) * 8
                            y0 *= za[stage * 2048 + zr * 64 + swizzle_xor_128b(zr, pr)]
                            y1 *= za[stage * 2048 + (zr + 1) * 64 + swizzle_xor_128b(zr + 1, pr)]
                        else:
                            z0, z1 = f16x2_to_f32(gate_words[j], dtype=cutlass.BFloat16)
                            y0 *= z0 / (1.0 + cute.math.exp(-z0, fastmath=True))
                            y1 *= z1 / (1.0 + cute.math.exp(-z1, fastmath=True))
                    if cutlass.const_expr(out.element_type == cutlass.Float32):
                        pr = (ct // 32) * 16 + lane // 4 + 8 * (j % 2)
                        token_col = (lane % 4) * 2 + (j // 2) * 8
                        yo[token_col * 64 + swizzle_xor_128b(token_col, pr)] = y0
                        yo[(token_col + 1) * 64 + swizzle_xor_128b(token_col + 1, pr)] = y1
                    else:
                        epilogue.append(fp32_to_fp16(y0, y1, dtype=cutlass.BFloat16))
                if cutlass.const_expr(out.element_type != cutlass.Float32):
                    for block in cutlass.range_constexpr(2):
                        packs = [epilogue[block * 4 + j] for j in range(4)]
                        nvvm.stmatrix(yo.data_ptr() + (vo_row + block * 16) * 64 + swizzle_xor_128b(vo_row + block * 16, vo_col), packs, nvvm.MMALayout.COL)
                        if cutlass.const_expr(ungated_out is not None):
                            saved = [pre_gate[block * 4 + j] for j in range(4)]
                            nvvm.stmatrix(yu.data_ptr() + (vo_row + block * 16) * 64 + swizzle_xor_128b(vo_row + block * 16, vo_col), saved, nvvm.MMALayout.COL)
                nvvm.barrier_cta_sync(2, thread_count=128)
                for pack in cutlass.range_constexpr(2):
                    idx = ct * 8 + pack * 1024
                    r, col = idx // 64, idx % 64
                    tok = chunk * 32 + r
                    if tok < self.length:
                        yi = r * 64 + swizzle_xor_128b(r, col)
                        yy = (yo.data_ptr() + yi).load(count=8, alignment=16)
                        oi = ((batch * self.length + tok) * self.heads + h) * 64 + col
                        (out.iterator.raw_ptr() + oi).store(yy, alignment=16)
                        if cutlass.const_expr(ungated_out is not None):
                            ungated_values = (yu.data_ptr() + yi).load(count=8, alignment=16)
                            (ungated_out.iterator.raw_ptr() + oi).store(ungated_values, alignment=16)
                nvvm.barrier_cta_sync(2, thread_count=128)
                state_done.wait(phase)
                inputs_done[stage].arrive()
            state_done.wait((self.nchunks - 1) & 1)
            sv = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), num=8)
            nvvm.tcgen05_wait("load")
            for j in cutlass.range_constexpr(16):
                r0 = (warp - 4) * 16 + lane // 4 + (j % 2) * 8
                col = lane % 4 * 2 + j // 2 * 8
                pair = cutlass.Vector.from_elements((sv[2 * j], sv[2 * j + 1]), cutlass.Float32)
                (final.iterator.raw_ptr() + bh * 4096 + r0 * 64 + col).store(pair, alignment=8)
        else:
            qk_desc = MmaDesc(
                M=64,
                N=32,
                K=64,
                bpe_a=2,
                bpe_b=2,
                tile_k_hw=16,
                idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, n_dim=32, m_dim=64),
                kind=nvvm.Tcgen05MMAKind.F16,
            )
            out_desc = MmaDesc(
                M=64,
                N=32,
                K=32,
                bpe_a=2,
                bpe_b=2,
                tile_k_hw=16,
                idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, n_dim=32, m_dim=64),
                kind=nvvm.Tcgen05MMAKind.F16,
            )
            kv_desc = MmaDesc(
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
            inputs_ready[0].wait(0)
            kd0 = SmemTile(kq.data_ptr(), 4096, 16, 1024, 2).desc()
            mma_ss(qk_desc, kd0, kd0, nvvm.make_tmem_ptr(base + 160, cutlass.Float32))
            if nvvm.elect_sync():
                scores_ready[0].arrive(cta_group=1)
            for chunk in cutlass.range(self.nchunks, unroll=1):
                stage, phase = chunk % 3, chunk & 1
                inputs_ready[stage].wait((chunk // 3) & 1)
                kd = SmemTile(kq.data_ptr() + stage * 4096, 4096, 16, 1024, 2).desc()
                qd = kd + 256
                state_ready.wait(phase)
                mma_ts(qk_desc, nvvm.make_tmem_ptr(base + 64, cutlass.Int8), qd, nvvm.make_tmem_ptr(base + 96, cutlass.Float32))
                if nvvm.elect_sync():
                    inter_ready.arrive(cta_group=1)
                if chunk + 1 < self.nchunks:
                    look = chunk + 1
                    inputs_ready[look % 3].wait((look // 3) & 1)
                    if look >= 2:
                        scores_done[look % 2].wait(((look // 2) & 1) ^ 1)
                    lookd = SmemTile(kq.data_ptr() + (look % 3) * 4096, 4096, 16, 1024, 2).desc()
                    mma_ss(qk_desc, lookd, lookd, nvvm.make_tmem_ptr(base + 160 + (look % 2) * 32, cutlass.Float32))
                    if nvvm.elect_sync():
                        scores_ready[look % 2].arrive(cta_group=1)
                a_ready[stage].wait((chunk // 3) & 1)
                x_ready.wait(phase)
                inter_scaled.wait(phase)
                ad = SmemTile(att.data_ptr() + stage * 2048, 2048, 16, 1024, 2).desc()
                mma_ts(out_desc, nvvm.make_tmem_ptr(base + 128, cutlass.Int8), ad, nvvm.make_tmem_ptr(base + 96, cutlass.Float32), accumulate=True)
                if nvvm.elect_sync():
                    a_done[stage].arrive(cta_group=1)
                    out_ready.arrive(cta_group=1)
                kt = SmemTile(kq.data_ptr() + stage * 4096, 4096, 8192, 1024, 2).desc()
                mma_ts(kv_desc, nvvm.make_tmem_ptr(base + 144, cutlass.Int8), kt, nvvm.make_tmem_ptr(base, cutlass.Float32), accumulate=True)
                if nvvm.elect_sync():
                    state_done.arrive(cta_group=1)
        nvvm.barrier_cta_sync_aligned()
        if warp == 12:
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(base, cutlass.Float32), cutlass.Int32(256), group=nvvm.CTAGroup.CTA_1)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
