# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental CTA reuse across several independent logical BT=32 chunks."""

import cutlass
from .mamba2_math import softplus
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
import cuda.bindings.driver as cuda
from cudnn.frost.tile_dsl.barrier import MBarrier, Producer
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b


@cute.jit
def sum64(v):
    a = [v[2 * i] + v[2 * i + 1] for i in range(32)]
    b = [a[2 * i] + a[2 * i + 1] for i in range(16)]
    c = [b[2 * i] + b[2 * i + 1] for i in range(8)]
    d = [c[2 * i] + c[2 * i + 1] for i in range(4)]
    e = [d[2 * i] + d[2 * i + 1] for i in range(2)]
    return e[0] + e[1]


@cute.jit
def gemm(
    sa,
    sb,
    address,
    bar,
    phase,
    n: cutlass.Constexpr[int] = 64,
    k: cutlass.Constexpr[int] = 64,
    transpose: cutlass.Constexpr[bool] = False,
    accumulate: cutlass.Constexpr[bool] = False,
):
    tid, _, _ = cute.arch.thread_idx()
    if tid < 32:
        desc = MmaDesc(
            M=64,
            N=n,
            K=k,
            bpe_a=2,
            bpe_b=2,
            tile_k_hw=16,
            btranspose=transpose,
            idesc=nvvm.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32, a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, n_dim=n, m_dim=64, b_major=int(transpose)
            ),
            kind=nvvm.Tcgen05MMAKind.F16,
        )
        ad = SmemTile(sa.data_ptr(), 4096, 16, 1024, 2).desc()
        bd = SmemTile(sb.data_ptr(), 4096, k * 128 if transpose else 16, 1024, 2).desc()
        mma_ss(desc, ad, bd, nvvm.make_tmem_ptr(address, cutlass.Float32), accumulate=accumulate)
        if nvvm.elect_sync():
            bar.arrive(cta_group=1)
    bar.wait(phase)


class Mamba2BackwardChunks:
    def __init__(self, batch, length, heads, groups, chunks_per_cta=4, early_tmem_release=False, reuse_j_storage=False):
        self.batch, self.length, self.heads, self.groups = batch, length, heads, groups
        self.nchunks = (length + 31) // 32
        self.chunks_per_cta = chunks_per_cta
        self.early_tmem_release = early_tmem_release
        self.reuse_j_storage = reuse_j_storage
        self.chunk_groups = (self.nchunks + chunks_per_cta - 1) // chunks_per_cta

    @cute.jit
    def __call__(self, x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias, stream: cuda.CUstream):
        self.kernel(x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias).launch(
            grid=(self.batch * self.heads * self.chunk_groups, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, x, dy, dt, a, b, c, d, bias, seeds, adjoints, dx, db, dc, ddt, da, dd, dbias):
        tid, _, _ = cute.arch.thread_idx()
        work, _, _ = cute.arch.block_idx()
        bh, chunk_group = work // self.chunk_groups, work % self.chunk_groups
        batch, h = bh // self.heads, bh % self.heads
        group = h // (self.heads // self.groups)
        warp, lane = tid // 32, tid % 32
        row = warp * 16 + lane % 16
        sm = cutlass.AddressSpace.smem
        sx = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sz = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sb = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sc = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        ss = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sg = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sj = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        sjt = sj if cutlass.const_expr(self.reuse_j_storage) else cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        swt = cutlass.Array(cutlass.BFloat16, 4096, space=sm, alignment=1024)
        u = cutlass.Array(cutlass.Float32, 1024, space=sm, alignment=128)
        cross = cutlass.Array(cutlass.Float32, 1024, space=sm, alignment=128)
        p = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        delta = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        f = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        e = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        inter_gp = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        state_gp = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        write_gdt = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        if cutlass.const_expr(dd is not None):
            ddparts = cutlass.Array(cutlass.Float32, 32, space=sm, alignment=128)
        dotparts = cutlass.Array(cutlass.Float32, 4, space=sm, alignment=16)
        slot = cutlass.Array(cutlass.Int32, 1, space=sm, alignment=16)
        bar = MBarrier(cutlass.Array(cutlass.Int64, 1, space=sm, alignment=16), 1, 1, Producer.MMA_COMMIT)
        if tid == 0:
            bar.init()
            nvvm.fence_mbarrier_init()
        if warp == 0:
            nvvm.tcgen05_alloc(slot, cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)
            if cutlass.const_expr(self.early_tmem_release):
                nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync_aligned()
        base = slot.load()
        for step in cutlass.range(self.chunks_per_cta, unroll=1):
            chunk = chunk_group * self.chunks_per_cta + step
            block = bh * self.nchunks + chunk
            if chunk < self.nchunks:
                if warp == 0:
                    token = chunk * 32 + lane
                    dv = cutlass.Float32(0)
                    if token < self.length:
                        dv = dt[(batch * self.length + token) * self.heads + h].to(cutlass.Float32)
                        if cutlass.const_expr(bias is not None):
                            dv += bias[h]
                        dv = softplus(dv)
                    pv = dv * a[h] * 1.4426950408889634
                    for off in [1, 2, 4, 8, 16]:
                        prev = nvvm.shfl_sync(0xFFFFFFFF, pv, off, 0, kind=nvvm.Shfl.UP)
                        if lane >= off:
                            pv += prev
                    last = nvvm.shfl_sync(0xFFFFFFFF, pv, 31, 31, kind=nvvm.Shfl.IDX)
                    p[lane], delta[lane] = pv, dv
                    f[lane] = cute.math.exp2(pv, fastmath=True)
                    e[lane] = cute.math.exp2(last - pv, fastmath=True)
                dot_sg = cutlass.Float32(0)
                for pack in cutlass.range_constexpr(4):
                    idx = tid * 8 + pack * 1024
                    r, col = idx // 64, idx % 64
                    tok = chunk * 32 + r
                    xv = cutlass.Vector.from_elements(tuple(cutlass.BFloat16(0) for _ in range(8)), cutlass.BFloat16)
                    zv, bv, cv = xv, xv, xv
                    if r < 32 and tok < self.length:
                        oi = ((batch * self.length + tok) * self.heads + h) * 64 + col
                        bi = ((batch * self.length + tok) * self.groups + group) * 64 + col
                        xv = (x.iterator.raw_ptr() + oi).load(count=8, alignment=16)
                        zv = (dy.iterator.raw_ptr() + oi).load(count=8, alignment=16)
                        bv = (b.iterator.raw_ptr() + bi).load(count=8, alignment=16)
                        cv = (c.iterator.raw_ptr() + bi).load(count=8, alignment=16)
                    si = r * 64 + swizzle_xor_128b(r, col)
                    (sx.data_ptr() + si).store(xv, alignment=16)
                    (sz.data_ptr() + si).store(zv, alignment=16)
                    (sb.data_ptr() + si).store(bv, alignment=16)
                    (sc.data_ptr() + si).store(cv, alignment=16)
                    zero = cutlass.Vector.from_elements(tuple(cutlass.BFloat16(0) for _ in range(8)), cutlass.BFloat16)
                    (sj.data_ptr() + si).store(zero, alignment=16)
                    if cutlass.const_expr(not self.reuse_j_storage):
                        (sjt.data_ptr() + si).store(zero, alignment=16)
                    (swt.data_ptr() + si).store(zero, alignment=16)
                    sv = (seeds.iterator.raw_ptr() + block * 4096 + idx).load(count=8, alignment=16).to(cutlass.Float32)
                    gv = (adjoints.iterator.raw_ptr() + block * 4096 + idx).load(count=8, alignment=16).to(cutlass.Float32)
                    (ss.data_ptr() + si).store(sv.to(cutlass.BFloat16), alignment=16)
                    (sg.data_ptr() + si).store(gv.to(cutlass.BFloat16), alignment=16)
                    for j in cutlass.range_constexpr(8):
                        dot_sg += sv[j] * gv[j]
                for off in [16, 8, 4, 2, 1]:
                    dot_sg += nvvm.shfl_sync(0xFFFFFFFF, dot_sg, off, 31, kind=nvvm.Shfl.BFLY)
                if lane == 0:
                    dotparts[warp] = dot_sg
                nvvm.fence_proxy("async.shared", space="cta")
                nvvm.barrier_cta_sync_aligned()
                gemm(sc, sb, base, bar, 0, n=32)
                # All consumers must observe this phase before the MMA warp
                # can complete the next phase on the same single barrier.
                nvvm.barrier_cta_sync_aligned()
                gemm(sz, sx, base + 32, bar, 1, n=32)
                cb = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base, cutlass.Float32), num=4)
                zx = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(base + 32, cutlass.Float32), num=4)
                nvvm.tcgen05_wait("load")
                if warp < 2:
                    for half in cutlass.range_constexpr(2):
                        qr = warp * 16 + lane // 4 + half * 8
                        running = cutlass.Float32(0)
                        for k in cutlass.range_constexpr(4):
                            masses = []
                            for sub in cutlass.range_constexpr(2):
                                col = lane % 4 * 2 + k * 8 + sub
                                idx = k * 4 + half * 2 + sub
                                l = cutlass.Float32(0)
                                if qr >= col:
                                    l = cute.math.exp2(p[qr] - p[col], fastmath=True)
                                base_g = cb[idx] * zx[idx] * l
                                u[qr * 32 + col] = base_g
                                masses.append(base_g * delta[col])
                                jv = (zx[idx] * l * delta[col]).to(cutlass.BFloat16)
                                wv = (cb[idx] * l * delta[col]).to(cutlass.BFloat16)
                                sj[qr * 64 + swizzle_xor_128b(qr, col)] = jv
                                if cutlass.const_expr(not self.reuse_j_storage):
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
                            cross[qr * 32 + col0] = running + prior
                            cross[qr * 32 + col0 + 1] = running + prior + masses[0]
                            blocksum = nvvm.shfl_sync(0xFFFFFFFF, inclusive, (lane // 4) * 4 + 3, 31, kind=nvvm.Shfl.IDX)
                            running += blocksum
                nvvm.fence_proxy("async.shared", space="cta")
                nvvm.barrier_cta_sync_aligned()

                # dC = exp(p) * dY @ S + J @ B.
                gemm(sz, ss, base + 64, bar, 0, transpose=True)
                cv = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                factor = cutlass.Float32(0)
                if row < 32:
                    factor = f[row]
                if row < 32 and lane < 16:
                    cterms = []
                    for pack in cutlass.range_constexpr(8):
                        values = (sc.data_ptr() + row * 64 + swizzle_xor_128b(row, pack * 8)).load(count=8, alignment=16)
                        for j in cutlass.range_constexpr(8):
                            cterms.append(cv[pack * 8 + j] * values[j].to(cutlass.Float32))
                    inter_gp[row] = sum64(cterms) * factor
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), cv * factor)
                nvvm.tcgen05_wait("store")
                nvvm.barrier_cta_sync_aligned()
                gemm(sj, sb, base + 64, bar, 1, k=32, transpose=True, accumulate=True)
                cvout = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                if row < 32 and lane < 16 and chunk * 32 + row < self.length:
                    oi = ((batch * self.length + chunk * 32 + row) * self.heads + h) * 64
                    (dc.iterator.raw_ptr() + oi).store(cvout.to(dc.element_type), alignment=16)
                # Join TMEM readers before reusing both its output buffer and
                # the shared MMA completion barrier in the next operation.
                nvvm.barrier_cta_sync_aligned()

                # dB = delta*exp(p_end-p) * X @ G + J^T @ C.
                if cutlass.const_expr(self.reuse_j_storage):
                    # J is dead after dC. Read its active 32x32 tile before
                    # any thread overwrites it with the transpose for dB.
                    jr, jc = tid // 4, (tid % 4) * 8
                    jvalues = (sj.data_ptr() + jr * 64 + swizzle_xor_128b(jr, jc)).load(count=8, alignment=16)
                    nvvm.barrier_cta_sync_aligned()
                    for j in cutlass.range_constexpr(8):
                        sjt[(jc + j) * 64 + swizzle_xor_128b(jc + j, jr)] = jvalues[j]
                    nvvm.fence_proxy("async.shared", space="cta")
                    nvvm.barrier_cta_sync_aligned()
                gemm(sx, sg, base + 64, bar, 0, transpose=True)
                bv = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                factor = cutlass.Float32(0)
                if row < 32:
                    factor = delta[row] * e[row]
                if row < 32 and lane < 16:
                    bterms = []
                    for pack in cutlass.range_constexpr(8):
                        values = (sb.data_ptr() + row * 64 + swizzle_xor_128b(row, pack * 8)).load(count=8, alignment=16)
                        for j in cutlass.range_constexpr(8):
                            bterms.append(bv[pack * 8 + j] * values[j].to(cutlass.Float32))
                    dot_b = sum64(bterms)
                    state_gp[row] = dot_b * factor
                    write_gdt[row] = dot_b * e[row]
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), bv * factor)
                nvvm.tcgen05_wait("store")
                nvvm.barrier_cta_sync_aligned()
                gemm(sjt, sc, base + 64, bar, 1, k=32, transpose=True, accumulate=True)
                bvout = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                if row < 32 and lane < 16 and chunk * 32 + row < self.length:
                    oi = ((batch * self.length + chunk * 32 + row) * self.heads + h) * 64
                    (db.iterator.raw_ptr() + oi).store(bvout.to(db.element_type), alignment=16)
                nvvm.barrier_cta_sync_aligned()

                # dX = delta*exp(p_end-p) * B @ G^T + W^T @ dY + D*dY.
                gemm(sb, sg, base + 64, bar, 0)
                xv = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), xv * factor)
                nvvm.tcgen05_wait("store")
                nvvm.barrier_cta_sync_aligned()
                gemm(swt, sz, base + 64, bar, 1, k=32, transpose=True, accumulate=True)
                xvout = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(base + 64, cutlass.Float32), num=64)
                nvvm.tcgen05_wait("load")
                if row < 32 and lane < 16 and chunk * 32 + row < self.length:
                    oi = ((batch * self.length + chunk * 32 + row) * self.heads + h) * 64
                    dterms = []
                    for pack in cutlass.range_constexpr(8):
                        packvals = []
                        offset = row * 64 + swizzle_xor_128b(row, pack * 8)
                        yvalues = (sz.data_ptr() + offset).load(count=8, alignment=16)
                        if cutlass.const_expr(dd is not None):
                            xvalues = (sx.data_ptr() + offset).load(count=8, alignment=16)
                        for j in cutlass.range_constexpr(8):
                            col = pack * 8 + j
                            yz = yvalues[j].to(cutlass.Float32)
                            value = xvout[col]
                            if cutlass.const_expr(d is not None):
                                value += d[h] * yz
                            packvals.append(value.to(cutlass.BFloat16))
                            if cutlass.const_expr(dd is not None):
                                dterms.append(yz * xvalues[j].to(cutlass.Float32))
                        (dx.iterator.raw_ptr() + oi + pack * 8).store(cutlass.Vector.from_elements(tuple(packvals), cutlass.BFloat16), alignment=16)
                    if cutlass.const_expr(dd is not None):
                        ddparts[row] = sum64(dterms)
                nvvm.barrier_cta_sync_aligned()
                if warp == 0:
                    token = chunk * 32 + lane
                    ga, gd_bias, gd_skip = cutlass.Float32(0), cutlass.Float32(0), cutlass.Float32(0)
                    if token < self.length:
                        gl = f[31] * (dotparts[0] + dotparts[1] + dotparts[2] + dotparts[3])
                        gd = write_gdt[lane]
                        for i in cutlass.range_constexpr(32):
                            if i >= lane:
                                gl += inter_gp[i] + cross[i * 32 + lane]
                                gd += u[i * 32 + lane]
                            if i < lane:
                                gl += state_gp[i]
                        oi = (batch * self.length + token) * self.heads + h
                        rawdt = dt[oi].to(cutlass.Float32)
                        if cutlass.const_expr(bias is not None):
                            rawdt += bias[h]
                        gd += a[h] * gl
                        sigmoid = 1.0 / (1.0 + cute.math.exp(-rawdt, fastmath=True))
                        gd_bias = gd * sigmoid
                        ddt[oi] = gd_bias.to(ddt.element_type)
                        ga = gl * delta[lane]
                        if cutlass.const_expr(dd is not None):
                            gd_skip = ddparts[lane]
                    for off in [16, 8, 4, 2, 1]:
                        ga += nvvm.shfl_sync(0xFFFFFFFF, ga, off, 31, kind=nvvm.Shfl.BFLY)
                        gd_bias += nvvm.shfl_sync(0xFFFFFFFF, gd_bias, off, 31, kind=nvvm.Shfl.BFLY)
                        if cutlass.const_expr(dd is not None):
                            gd_skip += nvvm.shfl_sync(0xFFFFFFFF, gd_skip, off, 31, kind=nvvm.Shfl.BFLY)
                    if lane == 0:
                        da[block], dbias[block] = ga, gd_bias
                        if cutlass.const_expr(dd is not None):
                            dd[block] = gd_skip
                nvvm.barrier_cta_sync_aligned()
        if warp == 0:
            if cutlass.const_expr(not self.early_tmem_release):
                nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
            nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(base, cutlass.Float32), cutlass.Int32(128), group=nvvm.CTAGroup.CTA_1)

    kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
