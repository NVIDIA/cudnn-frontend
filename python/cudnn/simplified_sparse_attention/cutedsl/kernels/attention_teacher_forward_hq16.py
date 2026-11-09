# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse attention and teacher distribution for Hq16, D256, and top-k512."""

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack, make_fake_stream

I32 = cutlass.Int32

HQ = 16
D = 256
DCH = 128  # stream K/V in two halves for four-CTA occupancy
TOPK = 512
MT = 128  # support rows per QK MMA tile
NQK = TOPK // MT  # 4
KSUP = 128  # support per V chunk
NPV = TOPK // KSUP  # 4
KST = 1  # single stage: trade overlap for two-CTA occupancy
VST = 1
NW = 4
NT = NW * 32  # 128 threads
SCALE = 0.0625
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
INV_H = 0.0625
C1 = 0.0625 * 1.4426950408889634

# element counts
K_ELEMS = KST * MT * DCH  # 32768
Q_ELEMS = HQ * D
P_ELEMS = HQ * TOPK
SMEM_BYTES = (K_ELEMS + 2 * Q_ELEMS) * 2 + 2 * TOPK * 4 + 2 * HQ * 4 * 4 + HQ * 4 * 2 + 16 * 8 + 2048


@cute.jit
def _exp_accum8_raw(frg, neg_max, fs, ob):
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(0, 8, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[j], neg_max[j + 1]))
        x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
        e0 = cute.math.exp2(x0, fastmath=True)
        e1 = cute.math.exp2(x1, fastmath=True)
        fs[j], fs[j + 1] = cute.arch.add_packed_f32x2((fs[j], fs[j + 1]), (e0, e1))
        ob[j] = e0.to(cutlass.BFloat16)
        ob[j + 1] = e1.to(cutlass.BFloat16)


@cute.jit
def _exp_store8_normalized(frg, neg_max, norm, ob, teacher_sum):
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(0, 8, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[j], neg_max[j + 1]))
        x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
        e0 = cute.math.exp2(x0, fastmath=True)
        e1 = cute.math.exp2(x1, fastmath=True)
        p0 = e0 * norm[j]
        p1 = e1 * norm[j + 1]
        ob[j] = p0.to(cutlass.BFloat16)
        ob[j + 1] = p1.to(cutlass.BFloat16)
        teacher_sum[0] = teacher_sum[0] + p0 + p1


@cute.kernel
def _ssa(
    mQ8,
    mK8,
    mV8,
    mIdx,
    mVlen,
    mO,
    mLse,
    mTea,
    mSoftmaxMax,
    mSoftmaxInvSum,
    tmma_qk,
    tmma_pv,
    TQ: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    t = 2 * blk
    t1 = t + 1
    if t1 >= TQ:
        t1 = t
    warp_id = tidx // 32
    lane = tidx % 32
    sub = tidx % 128  # TMEM lane / support row within tile
    hg = tidx // 128  # head group of 16   (0..4)
    lg = sub // 32  # lane group (0..3)

    vlen = mVlen[t]

    # ---------------- static layouts ----------------
    a_l = sm100_utils.make_smem_layout_a(tmma_qk, (MT, HQ, DCH), cutlass.BFloat16, KST)
    b_l = sm100_utils.make_smem_layout_b(tmma_qk, (MT, HQ, D), cutlass.BFloat16, 1)
    va_l = sm100_utils.make_smem_layout_a(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, VST)
    pb_l = sm100_utils.make_smem_layout_b(tmma_pv, (MT, HQ, TOPK), cutlass.BFloat16, 1)

    # ---------------- shared memory ----------------
    smem = utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, b_l.outer, byte_alignment=1024, swizzle=b_l.inner)
    _pad = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(Q_ELEMS), byte_alignment=1024)
    sIdx = smem.allocate_tensor(cutlass.Int32, cute.make_layout(TOPK), byte_alignment=16)
    sIdx1 = smem.allocate_tensor(cutlass.Int32, cute.make_layout(TOPK), byte_alignment=16)
    sRedM = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sRedS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sMx = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sRl = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    mbart = smem.allocate_tensor(cutlass.Int64, cute.make_layout(16), byte_alignment=16)
    mbar = mbart.iterator

    kp = sK.iterator
    # gathered-K cp.async view: (stage, kc, m, kb, i) -> stage*32768 + kc*8192 + m*64 + kb*8 + i
    sK8 = cute.make_tensor(kp, cute.make_layout((KST, 2, MT, 8, 8), stride=(MT * DCH, 8192, 64, 8, 1)))
    qp = sQ.iterator
    # Q cp.async view: (kc, h, kb, i) -> kc*5120 + h*64 + kb*8 + i
    sQ8 = cute.make_tensor(qp, cute.make_layout((4, HQ, 8, 8), stride=(HQ * 64, 64, 8, 1)))

    # Slot-contiguous V^T stages alias the retired 64-KiB K stages exactly.
    sV = cute.make_tensor(kp, va_l.outer)
    # V cp.async view for one 128-wide output half: (stage, mh, m8, k, i)
    sV8 = cute.make_tensor(kp, cute.make_layout((VST, 2, 8, KSUP, 8), stride=(MT * KSUP, KSUP * 64, 8, 64, 1)))

    # P^T operand (aliases Q + pad)
    pp = cute.recast_ptr(qp, pb_l.inner)
    sP = cute.make_tensor(pp, pb_l.outer)
    # P^T store/load view: (hg, s, hsub, j) -> hg*8192 + s*16 + hsub*8 + j
    sP8 = cute.make_tensor(
        pp,
        cute.make_layout((HQ // 16, TOPK, 2, 8), stride=(8192, 16, 8, 1)),
    )

    cpa = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128)

    # ---------------- mbarriers ----------------
    if tidx == 0:
        for i in cutlass.range_constexpr(16):
            cute.arch.mbarrier_init(mbar + i, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- indices + Q ----------------
    for i in cutlass.range_constexpr(TOPK // NT):
        c = i * NT + tidx
        sIdx[c] = mIdx[t, c]
        sIdx1[c] = mIdx[t1, c]

    for i in cutlass.range_constexpr(HQ * D // 8 // NT):
        c = i * NT + tidx
        h = c // 32
        dc = c % 32
        cute.copy(cpa, mQ8[t, c, None], sQ8[dc // 8, h, dc % 8, None])
    cute.arch.cp_async_commit_group()

    # ---------------- TMEM ----------------
    tbar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tbar)
    tmem.allocate(128)
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(cutlass.Float32)
    tmem.relinquish_alloc_permit()

    accS_shape = tmma_qk.partition_shape_C((MT, HQ))
    fakeS = tmma_qk.make_fragment_C(cute.append(accS_shape, NQK))
    tS = cute.make_tensor(tptr, fakeS.layout)
    fakeO = tmma_pv.make_fragment_C(cute.append(accS_shape, 2))
    tO = cute.make_tensor(tptr + NQK * HQ, fakeO.layout)

    tCrK = tmma_qk.make_fragment_A(sK)
    tCrQ = tmma_qk.make_fragment_B(sQ)
    tCrV = tmma_pv.make_fragment_A(sV)
    tCrP = tmma_pv.make_fragment_B(sP)

    # TMEM load plumbing (32 lanes x 8 columns per warp-quad)
    ldop = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), cutlass.Float32)
    tprobe = cute.make_tensor(tptr, cute.make_layout((MT, 8), stride=(65536, 1)))
    tld = tcgen05.make_tmem_copy(ldop, tprobe)
    thr_ld = tld.get_slice(sub)
    cId = cute.make_identity_tensor((MT, 8))
    tCrd = thr_ld.partition_D(cId)
    frg = cute.make_fragment_like(tCrd, cutlass.Float32)
    row = tCrd[0][0]

    pv4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    ob = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    mp0 = cute.make_rmem_tensor((8,), cutlass.Float32)
    mp1 = cute.make_rmem_tensor((8,), cutlass.Float32)
    fs0 = cute.make_rmem_tensor((8,), cutlass.Float32)
    fs1 = cute.make_rmem_tensor((8,), cutlass.Float32)
    nr = cute.make_rmem_tensor((8,), cutlass.Float32)
    teacher_sum = cute.make_rmem_tensor((1,), cutlass.Float32)

    # Keep one partial maximum per head and 32-row lane group.  The QK result
    # is split across four independent TMEM tiles, so a stable softmax shift
    # must include every tile rather than assuming that tile 0 contains the
    # largest score.
    for j in cutlass.range_constexpr(8):
        mp0[j] = -cutlass.Float32.inf
        mp1[j] = -cutlass.Float32.inf

    cute.arch.barrier()

    # ---------------- QK ----------------
    # prologue: gather support tile 0
    for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
        c = i * NT + tidx
        r = c >> 4
        ch = c & 15
        cute.copy(cpa, mK8[sIdx[r], ch, None], sK8[0, ch >> 3, r, ch & 7, None])
    cute.arch.cp_async_commit_group()

    for nt in cutlass.range_constexpr(NQK):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(DCH // 16):
                tmma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
                cute.gemm(tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, nt % KST], tCrQ[None, None, kb, 0], tS[None, None, None, nt])
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 12 + nt)
        cute.arch.mbarrier_wait(mbar + 12 + nt, 0)
        for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mK8[sIdx[nt * MT + r], 16 + ch, None], sK8[0, ch >> 3, r, ch & 7, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(DCH // 16):
                tmma_qk.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, nt % KST], tCrQ[None, None, DCH // 16 + kb, 0], tS[None, None, None, nt])
            with cute.arch.elect_one():
                tcgen05.commit(mbar + nt)
        if cutlass.const_expr(nt + 1 < NQK):
            cute.arch.mbarrier_wait(mbar + nt, 0)
            base = (nt + 1) * MT
            st = (nt + 1) % KST
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mK8[sIdx[base + r], ch, None], sK8[st, ch >> 3, r, ch & 7, None])
        if cutlass.const_expr(nt == NQK - 1):
            # K stage 0 retired after QK tile 2; fill its exact-alias V stage
            # while the final QK tile continues asynchronously on K stage 1.
            cute.arch.mbarrier_wait(mbar + nt, 0)
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mV8[sIdx[r], ch, None], sV8[0, (ch >> 3) & 1, ch & 7, r, None])
        cute.arch.cp_async_commit_group()
        # The preceding wait makes QK tile nt-1 visible.  Reduce that tile now
        # while the next MMA is in flight, retaining a global maximum over all
        # four support tiles in registers.
        if cutlass.const_expr(nt >= 1):
            mt = nt - 1
            s = mt * MT + row
            for hsub in cutlass.range_constexpr(2):
                col = mt * HQ + hg * 16 + hsub * 8
                tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                mp = mp0 if hsub == 0 else mp1
                for j in cutlass.range_constexpr(8):
                    x = frg[j] if s < vlen else -cutlass.Float32.inf
                    x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
                    mp[j] = cute.arch.fmax(mp[j], x)

    cute.arch.mbarrier_wait(mbar + (NQK - 1), 0)

    # Fold in the final QK tile, which has no following pipeline iteration.
    s = (NQK - 1) * MT + row
    for hsub in cutlass.range_constexpr(2):
        col = (NQK - 1) * HQ + hg * 16 + hsub * 8
        tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
        cute.copy(tld, thr_ld.partition_S(tv), frg)
        mp = mp0 if hsub == 0 else mp1
        for j in cutlass.range_constexpr(8):
            x = frg[j] if s < vlen else -cutlass.Float32.inf
            x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
            mp[j] = cute.arch.fmax(mp[j], x)

    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedM[hg * 16 + j, lg] = mp0[j]
            sRedM[hg * 16 + 8 + j, lg] = mp1[j]

    cute.arch.barrier()

    if tidx < HQ:
        cute.autovec_copy(sRedM[tidx, None], pv4)
        sMx[tidx] = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))

    cute.arch.barrier()

    for j in cutlass.range_constexpr(8):
        mp0[j] = cutlass.Float32(0.0) - sMx[hg * 16 + j]
        mp1[j] = cutlass.Float32(0.0) - sMx[hg * 16 + 8 + j]

    for j in cutlass.range_constexpr(8):
        fs0[j] = cutlass.Float32(0.0)
        fs1[j] = cutlass.Float32(0.0)

    # Diagnostic ablation: determine the denominator before PV so the value
    # contraction can consume BF16(normalized_probability), matching the
    # reference's rounding point.  This intentionally adds a score/exp pass.
    for nt in cutlass.range_constexpr(NQK):
        s = nt * MT + row
        for hsub in cutlass.range_constexpr(2):
            col = nt * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            mp = mp0 if hsub == 0 else mp1
            fs = fs0 if hsub == 0 else fs1
            if s < vlen:
                _exp_accum8_raw(frg, mp, fs, ob)

    for j in cutlass.range_constexpr(8):
        fs0[j] = cute.arch.warp_reduction_sum(fs0[j])
        fs1[j] = cute.arch.warp_reduction_sum(fs1[j])
    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedS[hg * 16 + j, lg] = fs0[j]
            sRedS[hg * 16 + 8 + j, lg] = fs1[j]

    cute.arch.barrier()

    if tidx < HQ:
        cute.autovec_copy(sRedS[tidx, None], pv4)
        l = pv4[0] + pv4[1] + pv4[2] + pv4[3]
        sRl[tidx] = cute.rcp(l, approx=True, ftz=True)
        # Backward must reconstruct exactly the probability convention used
        # for the BF16 PV contraction.  LSE is a useful forward diagnostic,
        # but its log/rcp approximation path cannot reproduce this pair.
        mSoftmaxMax[t, tidx] = sMx[tidx]
        mSoftmaxInvSum[t, tidx] = sRl[tidx]
        mLse[t, tidx] = sMx[tidx] * SCALE + cute.log2(l, approx=True, ftz=True) * LN2

    cute.arch.barrier()

    for i in cutlass.range_constexpr(TOPK // NT):
        mTea[t, i * NT + tidx] = cutlass.Float32(0.0)
    cute.arch.barrier()

    # ---------------- pass B interleaved with PV ----------------
    for nt in cutlass.range_constexpr(NQK):
        s = nt * MT + row
        teacher_sum[0] = cutlass.Float32(0.0)
        for hsub in cutlass.range_constexpr(2):
            col = nt * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            mp = mp0 if hsub == 0 else mp1
            fs = fs0 if hsub == 0 else fs1
            if s < vlen:
                for jj in cutlass.range_constexpr(8):
                    nr[jj] = sRl[hg * 16 + hsub * 8 + jj]
                _exp_store8_normalized(frg, mp, nr, ob, teacher_sum)
            else:
                for j in cutlass.range_constexpr(8):
                    ob[j] = cutlass.BFloat16(0.0)
            cute.autovec_copy(ob, sP8[hg, s, hsub, None])
        if s < vlen:
            mTea[t, s] = teacher_sum[0] * INV_H
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(KSUP // 16):
                tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                cute.gemm(
                    tmma_pv, tO[None, None, None, 0], tCrV[None, None, kb, nt % VST], tCrP[None, None, nt * (KSUP // 16) + kb, 0], tO[None, None, None, 0]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 8 + nt)
        cute.arch.mbarrier_wait(mbar + 8 + nt, 0)
        for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mV8[sIdx[nt * KSUP + r], 16 + ch, None], sV8[0, (ch >> 3) & 1, ch & 7, r, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(KSUP // 16):
                tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                cute.gemm(
                    tmma_pv, tO[None, None, None, 1], tCrV[None, None, kb, nt % VST], tCrP[None, None, nt * (KSUP // 16) + kb, 0], tO[None, None, None, 1]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 4 + nt)
        if cutlass.const_expr(nt + 1 < NPV):
            cute.arch.mbarrier_wait(mbar + 4 + nt, 0)
            vbase = (nt + 1) * KSUP
            vst = (nt + 1) % VST
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mV8[sIdx[vbase + r], ch, None], sV8[vst, (ch >> 3) & 1, ch & 7, r, None])
        if cutlass.const_expr(nt == NQK - 1):
            # V stage 0 is dead after PV tile 2.  Refill the same 64-KiB slot
            # with the next query's first K tile while PV tile 3 drains.
            cute.arch.mbarrier_wait(mbar + 4 + nt, 0)
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mK8[sIdx1[r], ch, None], sK8[0, ch >> 3, r, ch & 7, None])
        cute.arch.cp_async_commit_group()

    # LSE/reciprocal were computed by the diagnostic sum prepass.
    cute.arch.barrier()

    # ---------------- teacher (overlaps the tail PV MMAs) ----------------
    # Teacher was accumulated from FP32 normalized probabilities above.

    cute.arch.mbarrier_wait(mbar + 4 + (NQK - 1), 0)
    cute.arch.barrier()

    # P aliases Q, so publish query 1 only after its teacher and final PV use.
    for i in cutlass.range_constexpr(HQ * D // 8 // NT):
        c = i * NT + tidx
        h = c // 32
        dc = c % 32
        cute.copy(cpa, mQ8[t1, c, None], sQ8[dc // 8, h, dc % 8, None])
    cute.arch.cp_async_commit_group()

    # ---------------- epilogue ----------------
    rl8 = cute.make_rmem_tensor((8,), cutlass.Float32)
    for hsub in cutlass.range_constexpr(2):
        for j in cutlass.range_constexpr(8):
            rl8[j] = sRl[hg * 16 + hsub * 8 + j]
        for ai in cutlass.range_constexpr(2):
            col = NQK * HQ + ai * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            dv = ai * MT + row
            for j in cutlass.range_constexpr(8):
                mO[t, hg * 16 + hsub * 8 + j, dv] = frg[j].to(cutlass.BFloat16)

    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier()

    # ---------------- query 1 (barrier phase 1) ----------------
    t = t1
    vlen = mVlen[t]

    for j in cutlass.range_constexpr(8):
        mp0[j] = -cutlass.Float32.inf
        mp1[j] = -cutlass.Float32.inf

    for nt in cutlass.range_constexpr(NQK):
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(DCH // 16):
                tmma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
                cute.gemm(tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, nt % KST], tCrQ[None, None, kb, 0], tS[None, None, None, nt])
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 12 + nt)
        cute.arch.mbarrier_wait(mbar + 12 + nt, 1)
        for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mK8[sIdx1[nt * MT + r], 16 + ch, None], sK8[0, ch >> 3, r, ch & 7, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(DCH // 16):
                tmma_qk.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, nt % KST], tCrQ[None, None, DCH // 16 + kb, 0], tS[None, None, None, nt])
            with cute.arch.elect_one():
                tcgen05.commit(mbar + nt)
        if cutlass.const_expr(nt + 1 < NQK):
            cute.arch.mbarrier_wait(mbar + nt, 1)
            base = (nt + 1) * MT
            st = (nt + 1) % KST
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mK8[sIdx1[base + r], ch, None], sK8[st, ch >> 3, r, ch & 7, None])
        if cutlass.const_expr(nt == NQK - 1):
            # K stage 0 retired after QK tile 2; fill its exact-alias V stage
            # while the final QK tile continues asynchronously on K stage 1.
            cute.arch.mbarrier_wait(mbar + nt, 1)
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mV8[sIdx1[r], ch, None], sV8[0, (ch >> 3) & 1, ch & 7, r, None])
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(nt >= 1):
            mt = nt - 1
            s = mt * MT + row
            for hsub in cutlass.range_constexpr(2):
                col = mt * HQ + hg * 16 + hsub * 8
                tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                mp = mp0 if hsub == 0 else mp1
                for j in cutlass.range_constexpr(8):
                    x = frg[j] if s < vlen else -cutlass.Float32.inf
                    x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
                    mp[j] = cute.arch.fmax(mp[j], x)

    cute.arch.mbarrier_wait(mbar + (NQK - 1), 1)

    s = (NQK - 1) * MT + row
    for hsub in cutlass.range_constexpr(2):
        col = (NQK - 1) * HQ + hg * 16 + hsub * 8
        tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
        cute.copy(tld, thr_ld.partition_S(tv), frg)
        mp = mp0 if hsub == 0 else mp1
        for j in cutlass.range_constexpr(8):
            x = frg[j] if s < vlen else -cutlass.Float32.inf
            x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
            mp[j] = cute.arch.fmax(mp[j], x)

    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedM[hg * 16 + j, lg] = mp0[j]
            sRedM[hg * 16 + 8 + j, lg] = mp1[j]

    cute.arch.barrier()

    if tidx < HQ:
        cute.autovec_copy(sRedM[tidx, None], pv4)
        sMx[tidx] = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))

    cute.arch.barrier()

    for j in cutlass.range_constexpr(8):
        mp0[j] = cutlass.Float32(0.0) - sMx[hg * 16 + j]
        mp1[j] = cutlass.Float32(0.0) - sMx[hg * 16 + 8 + j]

    for j in cutlass.range_constexpr(8):
        fs0[j] = cutlass.Float32(0.0)
        fs1[j] = cutlass.Float32(0.0)

    # Diagnostic ablation: determine the denominator before PV so the value
    # contraction can consume BF16(normalized_probability), matching the
    # reference's rounding point.  This intentionally adds a score/exp pass.
    for nt in cutlass.range_constexpr(NQK):
        s = nt * MT + row
        for hsub in cutlass.range_constexpr(2):
            col = nt * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            mp = mp0 if hsub == 0 else mp1
            fs = fs0 if hsub == 0 else fs1
            if s < vlen:
                _exp_accum8_raw(frg, mp, fs, ob)

    for j in cutlass.range_constexpr(8):
        fs0[j] = cute.arch.warp_reduction_sum(fs0[j])
        fs1[j] = cute.arch.warp_reduction_sum(fs1[j])
    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedS[hg * 16 + j, lg] = fs0[j]
            sRedS[hg * 16 + 8 + j, lg] = fs1[j]

    cute.arch.barrier()

    if tidx < HQ:
        cute.autovec_copy(sRedS[tidx, None], pv4)
        l = pv4[0] + pv4[1] + pv4[2] + pv4[3]
        sRl[tidx] = cute.rcp(l, approx=True, ftz=True)
        mSoftmaxMax[t, tidx] = sMx[tidx]
        mSoftmaxInvSum[t, tidx] = sRl[tidx]
        mLse[t, tidx] = sMx[tidx] * SCALE + cute.log2(l, approx=True, ftz=True) * LN2

    cute.arch.barrier()

    for i in cutlass.range_constexpr(TOPK // NT):
        mTea[t, i * NT + tidx] = cutlass.Float32(0.0)
    cute.arch.barrier()

    # ---------------- pass B interleaved with PV ----------------
    for nt in cutlass.range_constexpr(NQK):
        s = nt * MT + row
        teacher_sum[0] = cutlass.Float32(0.0)
        for hsub in cutlass.range_constexpr(2):
            col = nt * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            mp = mp0 if hsub == 0 else mp1
            fs = fs0 if hsub == 0 else fs1
            if s < vlen:
                for jj in cutlass.range_constexpr(8):
                    nr[jj] = sRl[hg * 16 + hsub * 8 + jj]
                _exp_store8_normalized(frg, mp, nr, ob, teacher_sum)
            else:
                for j in cutlass.range_constexpr(8):
                    ob[j] = cutlass.BFloat16(0.0)
            cute.autovec_copy(ob, sP8[hg, s, hsub, None])
        if s < vlen:
            mTea[t, s] = teacher_sum[0] * INV_H
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(KSUP // 16):
                tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                cute.gemm(
                    tmma_pv, tO[None, None, None, 0], tCrV[None, None, kb, nt % VST], tCrP[None, None, nt * (KSUP // 16) + kb, 0], tO[None, None, None, 0]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 8 + nt)
        cute.arch.mbarrier_wait(mbar + 8 + nt, 1)
        for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mV8[sIdx1[nt * KSUP + r], 16 + ch, None], sV8[0, (ch >> 3) & 1, ch & 7, r, None])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(KSUP // 16):
                tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                cute.gemm(
                    tmma_pv, tO[None, None, None, 1], tCrV[None, None, kb, nt % VST], tCrP[None, None, nt * (KSUP // 16) + kb, 0], tO[None, None, None, 1]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + 4 + nt)
        if cutlass.const_expr(nt + 1 < NPV):
            cute.arch.mbarrier_wait(mbar + 4 + nt, 1)
            vbase = (nt + 1) * KSUP
            vst = (nt + 1) % VST
            for i in cutlass.range_constexpr(MT * DCH // 8 // NT):
                c = i * NT + tidx
                r = c >> 4
                ch = c & 15
                cute.copy(cpa, mV8[sIdx1[vbase + r], ch, None], sV8[vst, (ch >> 3) & 1, ch & 7, r, None])
        cute.arch.cp_async_commit_group()

    # LSE/reciprocal were computed by the diagnostic sum prepass.
    cute.arch.barrier()

    # ---------------- teacher (overlaps the tail PV MMAs) ----------------
    # Teacher was accumulated from FP32 normalized probabilities above.

    cute.arch.mbarrier_wait(mbar + 4 + (NQK - 1), 1)

    # ---------------- epilogue ----------------
    rl8 = cute.make_rmem_tensor((8,), cutlass.Float32)
    for hsub in cutlass.range_constexpr(2):
        for j in cutlass.range_constexpr(8):
            rl8[j] = sRl[hg * 16 + hsub * 8 + j]
        for ai in cutlass.range_constexpr(2):
            col = NQK * HQ + ai * HQ + hg * 16 + hsub * 8
            tv = cute.make_tensor(tptr + col, cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            dv = ai * MT + row
            for j in cutlass.range_constexpr(8):
                mO[t, hg * 16 + hsub * 8 + j, dv] = frg[j].to(cutlass.BFloat16)

    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier()
    tmem.free(tptr)


@cute.jit
def _launch(
    pQ8,
    pK8,
    pV8,
    pIdx,
    pVlen,
    pO,
    pLse,
    pTea,
    pSoftmaxMax,
    pSoftmaxInvSum,
    TQ: I32,
    TK: I32,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    stream: cuda.CUstream,
):
    # The DLPack arguments use a fully dynamic ABI so one executable accepts
    # every packed THD extent.  Rebuild the logical views from their pointers
    # to keep the cp.async vector/head modes static while only TQ/TK and the
    # fused-QKV token pitches remain runtime values.
    q_token_stride = cute.assume(q_token_stride, 8)
    k_token_stride = cute.assume(k_token_stride, 8)
    v_token_stride = cute.assume(v_token_stride, 8)
    mQ8 = cute.make_tensor(
        pQ8.iterator,
        cute.make_layout((TQ, HQ * D // 8, 8), stride=(q_token_stride, 8, 1)),
    )
    mK8 = cute.make_tensor(
        pK8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(k_token_stride, 8, 1)),
    )
    mV8 = cute.make_tensor(
        pV8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(v_token_stride, 8, 1)),
    )
    mIdx = cute.make_tensor(pIdx.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mVlen = cute.make_tensor(pVlen.iterator, cute.make_layout((TQ,), stride=(1,)))
    mO = cute.make_tensor(
        pO.iterator,
        cute.make_layout((TQ, HQ, D), stride=(HQ * D, D, 1)),
    )
    mLse = cute.make_tensor(pLse.iterator, cute.make_layout((TQ, HQ), stride=(HQ, 1)))
    mTea = cute.make_tensor(pTea.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mSoftmaxMax = cute.make_tensor(
        pSoftmaxMax.iterator,
        cute.make_layout((TQ, HQ), stride=(HQ, 1)),
    )
    mSoftmaxInvSum = cute.make_tensor(
        pSoftmaxInvSum.iterator,
        cute.make_layout((TQ, HQ), stride=(HQ, 1)),
    )
    tmma_qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    tmma_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.MN, OperandMajorMode.MN, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    _ssa(
        mQ8,
        mK8,
        mV8,
        mIdx,
        mVlen,
        mO,
        mLse,
        mTea,
        mSoftmaxMax,
        mSoftmaxInvSum,
        tmma_qk,
        tmma_pv,
        TQ,
    ).launch(
        grid=[(TQ + 1) // 2, 1, 1],
        block=[NT, 1, 1],
        smem=SMEM_BYTES,
        stream=stream,
    )


_cache = {}


def _dynamic_tensor(tensor, alignment):
    return from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


def _validate_qkv_vector_layout(name, tensor, expected_heads):
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"SSA {name} must be BF16, got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[1:] != (expected_heads, D):
        raise ValueError(f"SSA {name} must be [T,{expected_heads},{D}], got {tuple(tensor.shape)}")
    # A singleton head axis has no observable stride.  a fused QKV view
    # may inherit the full token pitch there even though its only D-vector is
    # contiguous.  Multi-head Q still requires a compact head-to-head stride.
    if tensor.stride(2) != 1 or (expected_heads > 1 and tensor.stride(1) != D):
        raise ValueError(f"SSA {name} head/dimension storage must be contiguous; " f"got strides {tuple(tensor.stride())}")
    if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
        raise ValueError(f"SSA {name} base address and token stride must be 16-byte aligned " "for 128-bit Q/K/V copies")


@torch.no_grad()
def run(
    q,
    k,
    v,
    selected_indices,
    valid_lengths,
    out,
    lse,
    teacher,
    softmax_max,
    softmax_inv_sum,
):
    tq = q.shape[0]
    tk = k.shape[0]
    if v.shape[0] != tk:
        raise ValueError(f"SSA attention K/V row mismatch: Tk={tk}, Tv={v.shape[0]}")
    _validate_qkv_vector_layout("Q", q, HQ)
    _validate_qkv_vector_layout("K", k, 1)
    _validate_qkv_vector_layout("V", v, 1)
    # Preserve the fused QKV projection's native token pitch while exposing
    # each compact 128-bit vector to cp.async.  The gaps between same-field
    # token views hold the other QKV fields, rather than padding.  The adapter
    # validates the required head/dimension contiguity at this boundary.
    q8 = q.as_strided(
        (tq, HQ * D // 8, 8),
        (q.stride(0), 8, 1),
    )
    k8 = k.as_strided(
        (tk, D // 8, 8),
        (k.stride(0), 8, 1),
    )
    v8 = v.as_strided(
        (tk, D // 8, 8),
        (v.stride(0), 8, 1),
    )
    ov = out.view(tq, HQ, D)
    idx = selected_indices.view(tq, TOPK)
    vl = valid_lengths.view(tq)
    ls = lse.view(tq, HQ)
    te = teacher.view(tq, TOPK)
    mx = softmax_max.view(tq, HQ)
    inv = softmax_inv_sum.view(tq, HQ)
    if mx.dtype != torch.float32 or inv.dtype != torch.float32 or not mx.is_contiguous() or not inv.is_contiguous():
        raise ValueError("SSA saved softmax max/inv_sum must be contiguous FP32 [Tq, Hq]")

    key = q.device.index
    if key not in _cache:
        args = [
            _dynamic_tensor(q8, 16),
            _dynamic_tensor(k8, 16),
            _dynamic_tensor(v8, 16),
            _dynamic_tensor(idx, 16),
            _dynamic_tensor(vl, 4),
            _dynamic_tensor(ov, 16),
            _dynamic_tensor(ls, 16),
            _dynamic_tensor(te, 16),
            _dynamic_tensor(mx, 16),
            _dynamic_tensor(inv, 16),
        ]
        _cache[key] = cute.compile(
            _launch,
            *args,
            I32(tq),
            I32(tk),
            I32(q8.stride(0)),
            I32(k8.stride(0)),
            I32(v8.stride(0)),
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _cache[key](
        q8,
        k8,
        v8,
        idx,
        vl,
        ov,
        ls,
        te,
        mx,
        inv,
        I32(tq),
        I32(tk),
        I32(q8.stride(0)),
        I32(k8.stride(0)),
        I32(v8.stride(0)),
    )
