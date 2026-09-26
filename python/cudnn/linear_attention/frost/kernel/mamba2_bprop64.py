# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BT=64 parallel-chunk SSD backward with both 64-column state tiles in one CTA.

One CTA owns a full (batch, head, 64-token chunk) and the whole N=128 state, so
dX / dDt need no cross-tile partials, the intra-chunk dY@X^T and W^T@dY are
computed once instead of twice, and every M=64 MMA sees 64 live rows.
"""

import cutlass
from .mamba2_math import softplus
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
LOG2E = 1.4426950408889634


@cute.jit
def sum8(values):
    return ((values[0] + values[1]) + (values[2] + values[3])) + ((values[4] + values[5]) + (values[6] + values[7]))


@cute.jit
def zeros_bf16():
    return cutlass.Vector.from_elements(tuple(cutlass.BFloat16(0) for _ in range(8)), cutlass.BFloat16)


@cute.jit
def issue_mma(
    sa_ptr,
    sb_ptr,
    address,
    n: cutlass.Constexpr[int] = 64,
    k: cutlass.Constexpr[int] = 64,
    transpose: cutlass.Constexpr[bool] = False,
    accumulate: cutlass.Constexpr[bool] = False,
):
    desc = MmaDesc(
        M=64,
        N=n,
        K=k,
        bpe_a=2,
        bpe_b=2,
        tile_k_hw=16,
        btranspose=transpose,
        idesc=nvvm.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=n, m_dim=64, b_major=int(transpose)),
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    ad = SmemTile(sa_ptr, 4096, 16, 1024, 2).desc()
    bd = SmemTile(sb_ptr, 4096, k * 128 if transpose else 16, 1024, 2).desc()
    mma_ss(desc, ad, bd, nvvm.make_tmem_ptr(address, F32), accumulate=accumulate)


class Mamba2BackwardChunks64:
    def __init__(self, batch, length, heads, groups):
        self.batch, self.length, self.heads, self.groups = batch, length, heads, groups
        self.nchunks = (length + 63) // 64

    @cute.jit
    def __call__(self, x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias, stream: cuda.CUstream):
        self.kernel(x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias).launch(
            grid=(self.batch * self.heads * self.nchunks, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias):
        tid, _, _ = cute.arch.thread_idx()
        work, _, _ = cute.arch.block_idx()
        nbh = cutlass.const_expr(self.batch * self.heads)
        bh, chunk = work % nbh, work // nbh
        batch, h = bh // self.heads, bh % self.heads
        group = h // (self.heads // self.groups)
        warp, lane = tid // 32, tid % 32
        L = cutlass.const_expr(self.length)
        H = cutlass.const_expr(self.heads)

        sm = cutlass.AddressSpace.smem
        sx = cutlass.Array(BF16, 4096, space=sm, alignment=1024)
        sz = cutlass.Array(BF16, 4096, space=sm, alignment=1024)
        sb = cutlass.Array(BF16, 8192, space=sm, alignment=1024)
        sc = cutlass.Array(BF16, 8192, space=sm, alignment=1024)
        ss = cutlass.Array(BF16, 8192, space=sm, alignment=1024)
        sg = cutlass.Array(BF16, 8192, space=sm, alignment=1024)
        sj = cutlass.Array(BF16, 4096, space=sm, alignment=1024)
        sjt = cutlass.Array(BF16, 4096, space=sm, alignment=1024)
        swt = cutlass.Array(BF16, 4096, space=sm, alignment=1024)

        sp = cutlass.Array(F32, 64, space=sm, alignment=256)
        sdelta = cutlass.Array(F32, 64, space=sm, alignment=256)
        sf = cutlass.Array(F32, 64, space=sm, alignment=256)
        se = cutlass.Array(F32, 64, space=sm, alignment=256)
        inter_gp = cutlass.Array(F32, 64, space=sm, alignment=256)
        state_gp = cutlass.Array(F32, 64, space=sm, alignment=256)
        write_gdt = cutlass.Array(F32, 64, space=sm, alignment=256)
        ddparts = cutlass.Array(F32, 64, space=sm, alignment=256)
        rgd = cutlass.Array(F32, 256, space=sm, alignment=256)
        rgl = cutlass.Array(F32, 256, space=sm, alignment=256)
        scratch = cutlass.Array(F32, 16, space=sm, alignment=64)

        slot = cutlass.Array(cutlass.Int32, 1, space=sm, alignment=16)
        bar = MBarrier(cutlass.Array(cutlass.Int64, 1, space=sm, alignment=16), 1, 1, Producer.MMA_COMMIT)
        if tid == 0:
            bar.init()
            nvvm.fence_mbarrier_init()
        if warp == 0:
            nvvm.tcgen05_alloc(slot, cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync_aligned()
        base = slot.load()
        phase = cutlass.Int32(0)

        # ---------------- prologue: stage operands ----------------
        block = bh * cutlass.const_expr(self.nchunks) + chunk
        for pack in cutlass.range_constexpr(4):
            idx = tid * 8 + pack * 1024
            r, col = idx // 64, idx % 64
            tok = chunk * 64 + r
            xv = zeros_bf16()
            zv = xv
            if tok < L:
                oi = ((batch * L + tok) * H + h) * 64 + col
                xv = (x.iterator.raw_ptr() + oi).load(count=8, alignment=16)
                zv = (dy.iterator.raw_ptr() + oi).load(count=8, alignment=16)
            si = r * 64 + swizzle_xor_128b(r, col)
            (sx.data_ptr() + si).store(xv, alignment=16)
            (sz.data_ptr() + si).store(zv, alignment=16)
        for pack in cutlass.range_constexpr(8):
            idx = tid * 8 + pack * 1024
            r, col = idx // 128, idx % 128
            tok = chunk * 64 + r
            bv = zeros_bf16()
            cv = bv
            if tok < L:
                bi = ((batch * L + tok) * self.groups + group) * 128 + col
                bv = (b.iterator.raw_ptr() + bi).load(count=8, alignment=16)
                cv = (c.iterator.raw_ptr() + bi).load(count=8, alignment=16)
            si = (col // 64) * 4096 + r * 64 + swizzle_xor_128b(r, col % 64)
            (sb.data_ptr() + si).store(bv, alignment=16)
            (sc.data_ptr() + si).store(cv, alignment=16)
        dot_sg = cutlass.Float32(0)
        for pack in cutlass.range_constexpr(8):
            idx = tid * 8 + pack * 1024
            r, col = idx // 128, idx % 128
            gi = block * 8192 + r * 128 + col
            sv = (seeds.iterator.raw_ptr() + gi).load(count=8, alignment=16)
            gv = (adjoints.iterator.raw_ptr() + gi).load(count=8, alignment=16)
            si = (col // 64) * 4096 + r * 64 + swizzle_xor_128b(r, col % 64)
            (ss.data_ptr() + si).store(sv.to(BF16), alignment=16)
            (sg.data_ptr() + si).store(gv.to(BF16), alignment=16)
            for j in cutlass.range_constexpr(8):
                dot_sg += sv[j] * gv[j]
        for off in [16, 8, 4, 2, 1]:
            dot_sg += nvvm.shfl_sync(0xFFFFFFFF, dot_sg, off, 31, kind=nvvm.Shfl.BFLY)
        if lane == 0:
            scratch[warp] = dot_sg

        if warp == 0:
            t0 = 2 * lane
            tok0 = chunk * 64 + t0
            d0, d1 = cutlass.Float32(0), cutlass.Float32(0)
            if tok0 < L:
                d0 = softplus(dt[(batch * L + tok0) * H + h].to(F32) + bias[h])
            if tok0 + 1 < L:
                d1 = softplus(dt[(batch * L + tok0 + 1) * H + h].to(F32) + bias[h])
            av = a[h] * LOG2E
            v0, v1 = d0 * av, d1 * av
            incl = v0 + v1
            for off in [1, 2, 4, 8, 16]:
                prev = nvvm.shfl_sync(0xFFFFFFFF, incl, off, 0, kind=nvvm.Shfl.UP)
                if lane >= off:
                    incl += prev
            last = nvvm.shfl_sync(0xFFFFFFFF, incl, 31, 31, kind=nvvm.Shfl.IDX)
            p1 = incl
            p0 = incl - v1
            sp[t0], sp[t0 + 1] = p0, p1
            sdelta[t0], sdelta[t0 + 1] = d0, d1
            sf[t0] = cute.math.exp2(p0, fastmath=True)
            sf[t0 + 1] = cute.math.exp2(p1, fastmath=True)
            se[t0] = cute.math.exp2(last - p0, fastmath=True)
            se[t0 + 1] = cute.math.exp2(last - p1, fastmath=True)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync_aligned()

        # ---------------- cb = C @ B^T, zx = dY @ X^T ----------------
        if tid < 32:
            issue_mma(sc.data_ptr(), sb.data_ptr(), base)
            issue_mma(sc.data_ptr() + 4096, sb.data_ptr() + 4096, base, accumulate=True)
            issue_mma(sz.data_ptr(), sx.data_ptr(), base + 64)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1

        cb = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base, F32), num=8)
        zx = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + 64, F32), num=8)
        nvvm.tcgen05_wait("load")
        pgd = [cutlass.Float32(0) for _ in range(16)]
        pgl = [cutlass.Float32(0) for _ in range(16)]
        for half in cutlass.range_constexpr(2):
            qr = warp * 16 + lane // 4 + half * 8
            pq = sp[qr]
            running = cutlass.Float32(0)
            for k in cutlass.range_constexpr(8):
                masses = []
                for sub in cutlass.range_constexpr(2):
                    col = lane % 4 * 2 + k * 8 + sub
                    idx = k * 4 + half * 2 + sub
                    l = cutlass.Float32(0)
                    if qr >= col:
                        l = cute.math.exp2(pq - sp[col], fastmath=True)
                    dl = sdelta[col]
                    uu = cb[idx] * zx[idx] * l
                    masses.append(uu * dl)
                    pgd[k * 2 + sub] += uu
                    jv = (zx[idx] * l * dl).to(BF16)
                    wv = (cb[idx] * l * dl).to(BF16)
                    sj[qr * 64 + swizzle_xor_128b(qr, col)] = jv
                    sjt[col * 64 + swizzle_xor_128b(col, qr)] = jv
                    swt[col * 64 + swizzle_xor_128b(col, qr)] = wv
                inclusive = masses[0] + masses[1]
                for off in [1, 2]:
                    previous = nvvm.shfl_sync(0xFFFFFFFF, inclusive, off, 0, kind=nvvm.Shfl.UP)
                    if lane % 4 >= off:
                        inclusive += previous
                prior = nvvm.shfl_sync(0xFFFFFFFF, inclusive, 1, 0, kind=nvvm.Shfl.UP)
                if lane % 4 == 0:
                    prior = cutlass.Float32(0)
                col0 = lane % 4 * 2 + k * 8
                c0 = running + prior
                if qr >= col0:
                    pgl[k * 2] += c0
                if qr >= col0 + 1:
                    pgl[k * 2 + 1] += c0 + masses[0]
                blocksum = nvvm.shfl_sync(0xFFFFFFFF, inclusive, (lane // 4) * 4 + 3, 31, kind=nvvm.Shfl.IDX)
                running += blocksum
        for off in [4, 8, 16]:
            for i in cutlass.range_constexpr(16):
                pgd[i] += nvvm.shfl_sync(0xFFFFFFFF, pgd[i], off, 31, kind=nvvm.Shfl.BFLY)
                pgl[i] += nvvm.shfl_sync(0xFFFFFFFF, pgl[i], off, 31, kind=nvvm.Shfl.BFLY)
        if lane < 4:
            for i in cutlass.range_constexpr(16):
                col = lane * 2 + (i // 2) * 8 + i % 2
                rgd[warp * 64 + col] = pgd[i]
                rgl[warp * 64 + col] = pgl[i]
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync_aligned()

        row = warp * 16 + lane % 16
        out_i = ((batch * L + chunk * 64 + row) * H + h) * 128

        # ---------------- dC = f * (dY @ S^T) + J @ B ----------------
        if tid < 32:
            issue_mma(sz.data_ptr(), ss.data_ptr(), base, transpose=True)
            issue_mma(sz.data_ptr(), ss.data_ptr() + 4096, base + 64, transpose=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        fac_c = sf[row]
        dotc = cutlass.Float32(0)
        for half in cutlass.range_constexpr(2):
            cvh = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), num=64)
            nvvm.tcgen05_wait("load")
            if lane < 16:
                parts = []
                for pack in cutlass.range_constexpr(8):
                    vals = (sc.data_ptr() + half * 4096 + row * 64 + swizzle_xor_128b(row, pack * 8)).load(count=8, alignment=16)
                    parts.append(sum8([cvh[pack * 8 + j] * vals[j].to(F32) for j in range(8)]))
                dotc += sum8(parts)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), cvh * fac_c)
        if lane < 16:
            inter_gp[row] = dotc * fac_c
        nvvm.tcgen05_wait("store")
        nvvm.barrier_cta_sync_aligned()
        if tid < 32:
            issue_mma(sj.data_ptr(), sb.data_ptr(), base, transpose=True, accumulate=True)
            issue_mma(sj.data_ptr(), sb.data_ptr() + 4096, base + 64, transpose=True, accumulate=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        for half in cutlass.range_constexpr(2):
            cvh = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), num=64)
            nvvm.tcgen05_wait("load")
            if lane < 16 and chunk * 64 + row < L:
                (dc.iterator.raw_ptr() + out_i + half * 64).store(cvh.to(dc.element_type), alignment=16)
        nvvm.barrier_cta_sync_aligned()

        # ---------------- dB = delta*e * (X @ G^T) + J^T @ C ----------------
        if tid < 32:
            issue_mma(sx.data_ptr(), sg.data_ptr(), base, transpose=True)
            issue_mma(sx.data_ptr(), sg.data_ptr() + 4096, base + 64, transpose=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        er = se[row]
        fac_b = sdelta[row] * er
        dot_b = cutlass.Float32(0)
        for half in cutlass.range_constexpr(2):
            bvh = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), num=64)
            nvvm.tcgen05_wait("load")
            if lane < 16:
                parts = []
                for pack in cutlass.range_constexpr(8):
                    vals = (sb.data_ptr() + half * 4096 + row * 64 + swizzle_xor_128b(row, pack * 8)).load(count=8, alignment=16)
                    parts.append(sum8([bvh[pack * 8 + j] * vals[j].to(F32) for j in range(8)]))
                dot_b += sum8(parts)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), bvh * fac_b)
        if lane < 16:
            state_gp[row] = dot_b * fac_b
            write_gdt[row] = dot_b * er
        nvvm.tcgen05_wait("store")
        nvvm.barrier_cta_sync_aligned()
        if tid < 32:
            issue_mma(sjt.data_ptr(), sc.data_ptr(), base, transpose=True, accumulate=True)
            issue_mma(sjt.data_ptr(), sc.data_ptr() + 4096, base + 64, transpose=True, accumulate=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        for half in cutlass.range_constexpr(2):
            bvh = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + half * 64, F32), num=64)
            nvvm.tcgen05_wait("load")
            if lane < 16 and chunk * 64 + row < L:
                (db.iterator.raw_ptr() + out_i + half * 64).store(bvh.to(db.element_type), alignment=16)
        nvvm.barrier_cta_sync_aligned()

        # ---------------- dX = delta*e * (B @ G^T) + W^T @ dY + D*dY ----------------
        if tid < 32:
            issue_mma(sb.data_ptr(), sg.data_ptr(), base)
            issue_mma(sb.data_ptr() + 4096, sg.data_ptr() + 4096, base, accumulate=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        xv = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base, F32), num=64)
        nvvm.tcgen05_wait("load")
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base, F32), xv * fac_b)
        nvvm.tcgen05_wait("store")
        nvvm.barrier_cta_sync_aligned()
        if tid < 32:
            issue_mma(swt.data_ptr(), sz.data_ptr(), base, transpose=True, accumulate=True)
            if nvvm.elect_sync():
                bar.arrive(cta_group=1)
        bar.wait(phase)
        phase ^= 1
        xvout = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base, F32), num=64)
        nvvm.tcgen05_wait("load")
        if lane < 16:
            dparts = []
            dh = d[h]
            oi = ((batch * L + chunk * 64 + row) * H + h) * 64
            for pack in cutlass.range_constexpr(8):
                offset = row * 64 + swizzle_xor_128b(row, pack * 8)
                yvalues = (sz.data_ptr() + offset).load(count=8, alignment=16)
                xvalues = (sx.data_ptr() + offset).load(count=8, alignment=16)
                packvals = []
                dpack = []
                for j in cutlass.range_constexpr(8):
                    yz = yvalues[j].to(F32)
                    packvals.append((xvout[pack * 8 + j] + dh * yz).to(dx.element_type))
                    dpack.append(yz * xvalues[j].to(F32))
                dparts.append(sum8(dpack))
                if chunk * 64 + row < L:
                    (dx.iterator.raw_ptr() + oi + pack * 8).store(cutlass.Vector.from_elements(tuple(packvals), dx.element_type), alignment=16)
            ddparts[row] = sum8(dparts)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync_aligned()

        # ---------------- scalar reductions ----------------
        if tid < 64:
            v_i, v_s = inter_gp[tid], state_gp[tid]
            inc_i, inc_s = v_i, v_s
            for off in [1, 2, 4, 8, 16]:
                qi = nvvm.shfl_sync(0xFFFFFFFF, inc_i, off, 0, kind=nvvm.Shfl.UP)
                qs = nvvm.shfl_sync(0xFFFFFFFF, inc_s, off, 0, kind=nvvm.Shfl.UP)
                if lane >= off:
                    inc_i += qi
                    inc_s += qs
            if lane == 31:
                scratch[8 + warp] = inc_i
                scratch[10 + warp] = inc_s
        nvvm.barrier_cta_sync_aligned()
        if tid < 64:
            v_i, v_s = inter_gp[tid], state_gp[tid]
            inc_i, inc_s = v_i, v_s
            for off in [1, 2, 4, 8, 16]:
                qi = nvvm.shfl_sync(0xFFFFFFFF, inc_i, off, 0, kind=nvvm.Shfl.UP)
                qs = nvvm.shfl_sync(0xFFFFFFFF, inc_s, off, 0, kind=nvvm.Shfl.UP)
                if lane >= off:
                    inc_i += qi
                    inc_s += qs
            if warp == 1:
                inc_i += scratch[8]
                inc_s += scratch[10]
            total_i = scratch[8] + scratch[9]
            gl = (scratch[0] + scratch[1] + scratch[2] + scratch[3]) * sf[63]
            gl += total_i - inc_i + v_i
            gl += inc_s - v_s
            gl += rgl[tid] + rgl[64 + tid] + rgl[128 + tid] + rgl[192 + tid]
            gd = write_gdt[tid] + rgd[tid] + rgd[64 + tid] + rgd[128 + tid] + rgd[192 + tid] + a[h] * gl
            tok = chunk * 64 + tid
            ga, gbias, gskip = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
            if tok < L:
                oi = (batch * L + tok) * H + h
                rawdt = dt[oi].to(F32) + bias[h]
                gbias = gd / (1.0 + cute.math.exp(-rawdt, fastmath=True))
                ddt[oi] = gbias.to(ddt.element_type)
                ga = gl * sdelta[tid]
                gskip = ddparts[tid]
            for off in [16, 8, 4, 2, 1]:
                ga += nvvm.shfl_sync(0xFFFFFFFF, ga, off, 31, kind=nvvm.Shfl.BFLY)
                gbias += nvvm.shfl_sync(0xFFFFFFFF, gbias, off, 31, kind=nvvm.Shfl.BFLY)
                gskip += nvvm.shfl_sync(0xFFFFFFFF, gskip, off, 31, kind=nvvm.Shfl.BFLY)
            if lane == 0:
                scratch[4 + warp] = ga
                scratch[6 + warp] = gbias
                scratch[12 + warp] = gskip
        nvvm.barrier_cta_sync_aligned()
        if tid == 0:
            da[block] = scratch[4] + scratch[5]
            dbias[block] = scratch[6] + scratch[7]
            dd[block] = scratch[12] + scratch[13]
        nvvm.barrier_cta_sync_aligned()
        if warp == 0:
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(base, F32), cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
