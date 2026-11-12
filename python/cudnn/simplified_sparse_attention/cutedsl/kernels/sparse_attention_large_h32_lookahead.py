# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SSA selected sparse-attention + teacher forward.

Hq in {16,32,96}, TopK in {512,1024,2048}, Hkv=1, Dqk=Dv=256, BF16 Q/K/V and
output, FP32 online-softmax state / LSE / teacher distribution.

One CTA owns a contiguous run of query rows and, for a given (Hq,TopK) bucket,
a compile-specialised head-group width.  The whole selected score block for the
head group stays resident in TMEM, so the gathered K stream is walked exactly
once per query row.  Tq, Tk, Q/K/V token pitches and the launch grid stay
runtime values.
"""

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

D = 256
MT = 128  # support rows per QK MMA tile
KSUP = 128  # support rows per PV chunk
KST = 2  # gathered-K pipeline stages
VST = 2  # gathered-V / probability pipeline stages
SCALE = 0.0625
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
NEG = float("-inf")
C1 = SCALE * LOG2E
NEGC = -1.0e30  # clamp: -inf minus -inf is NaN for an all-masked warp

SUPPORTED_HQ = (16, 32, 96)
SUPPORTED_TOPK = (512, 1024, 2048)

# (Hq, TopK) -> (head-group width, number of 128-thread head groups)
#   TMEM cost is NQK*HG columns: the two output accumulators alias support
#   tiles 0/1, which are drained before the first PV issues.  HPG = HG//NHG
#   must stay a multiple of the 8-column TMEM load.
# (Hq, TopK) -> (head-group width, 128-thread head groups, online-fold flag).
# ONL=1 folds the shifted sum online during the support walk, where it hides
# under the gathered-K stream and removes the separate denominator pass.  At
# HG=48 the walk is already close to compute bound, so that shape keeps the
# cheap max-only fold plus a denominator pass.
_SCHEDULE = {
    (16, 512): (16, 2, 1),
    (16, 1024): (16, 4, 1),
    (16, 2048): (16, 4, 1),
    (32, 512): (32, 4, 1),
    (32, 1024): (32, 4, 1),
    (32, 2048): (32, 4, 1),
    (96, 512): (96, 4, 0),
    (96, 1024): (48, 4, 0),
    (96, 2048): (32, 4, 1),
}


def _plan(hq, topk):
    hg, nhg, onl = _SCHEDULE[(hq, topk)]
    nqk = topk // MT
    nt = 128 * nhg
    cols = 32
    while cols < nqk * hg:
        cols *= 2
    smem = (
        (KST * MT * D + hg * D) * 2  # K/V union + Q/P union
        + 2 * topk * 4  # double-buffered support indices
        + 2 * hg * 4 * 4  # sRedM + sRedS
        + 2 * hg * 4  # sMx + sRl
        + nhg * topk * 4  # teacher partials
        + 2 * nqk * 8  # mbarriers
        + 4096
    )
    return hg, nhg, nt, cols, smem, onl


@cute.jit
def _fold_lane_sum(fs, lane, HPG: cutlass.Constexpr):
    """Fold all per-lane head sums with one ownership butterfly."""
    n = HPG
    off = 16
    while n > 1:
        h = n // 2
        sel = (lane & off) != 0
        for i in range(h):
            a = fs[i]
            b = fs[i + h]
            snd = a if sel else b
            kep = b if sel else a
            fs[i] = kep + cute.arch.shuffle_sync_bfly(snd, offset=off, mask=-1, mask_and_clamp=31)
        n = h
        off = off // 2
    while off > 0:
        fs[0] = fs[0] + cute.arch.shuffle_sync_bfly(fs[0], offset=off, mask=-1, mask_and_clamp=31)
        off = off // 2


@cute.jit
def _exp_accum8(frg, neg_max, fs, base: cutlass.Constexpr, REP: cutlass.Constexpr = 8):
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(0, REP, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[base + j], neg_max[base + j + 1]))
        x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
        e0 = cute.math.exp2(x0, fastmath=True)
        e1 = cute.math.exp2(x1, fastmath=True)
        fs[base + j], fs[base + j + 1] = cute.arch.add_packed_f32x2((fs[base + j], fs[base + j + 1]), (e0, e1))


@cute.jit
def _online(mp, fl, frg, base: cutlass.Constexpr, REP: cutlass.Constexpr):
    """Stable online (max, shifted-sum) update with one exp2 per score.

    ``(mo - mn) + (x - mn)`` with ``mn = max(mo, x)`` equals
    ``min(mo, x) - max(mo, x)``, so one FMNMX pair plus a packed subtract
    replaces two subtracts and an add.  ``mo`` starts at -inf, which is exact
    for any finite score and never clips a representable maximum.
    """
    for j in cutlass.range_constexpr(0, REP, 2):
        x0 = frg[j]
        x1 = frg[j + 1]
        o0 = mp[base + j]
        o1 = mp[base + j + 1]
        n0 = cute.arch.fmax(o0, x0)
        n1 = cute.arch.fmax(o1, x1)
        d0, d1 = cute.arch.sub_packed_f32x2((cute.arch.fmin(o0, x0), cute.arch.fmin(o1, x1)), (n0, n1))
        d0, d1 = cute.arch.mul_packed_f32x2((d0, d1), (C1, C1))
        e0 = cute.math.exp2(d0, fastmath=True)
        e1 = cute.math.exp2(d1, fastmath=True)
        l0 = fl[base + j]
        l1 = fl[base + j + 1]
        lo0, lo1 = cute.arch.add_packed_f32x2((l0, l1), (e0, e1))
        hi0, hi1 = cute.arch.fma_packed_f32x2((l0, l1), (e0, e1), (cutlass.Float32(1.0), cutlass.Float32(1.0)))
        fl[base + j] = hi0 if x0 > o0 else lo0
        fl[base + j + 1] = hi1 if x1 > o1 else lo1
        mp[base + j] = n0
        mp[base + j + 1] = n1


@cute.jit
def _exp_norm8(frg, neg_max, base: cutlass.Constexpr, nr, ob, tsum, REP: cutlass.Constexpr = 8):
    """p = exp2((s - max) * C1 + log2(1/sum)).

    ``s - max`` keeps its own rounding step so the shift still cancels
    exactly for adversarial raw logits, while log2 of the reciprocal sum
    rides in the FMA addend and removes the separate normalising multiply.
    """
    for j in cutlass.range_constexpr(0, REP, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[base + j], neg_max[base + j + 1]))
        p0, p1 = cute.arch.fma_packed_f32x2((x0, x1), (C1, C1), (nr[base + j], nr[base + j + 1]))
        p0 = cute.math.exp2(p0, fastmath=True)
        p1 = cute.math.exp2(p1, fastmath=True)
        ob[j] = p0.to(cutlass.BFloat16)
        ob[j + 1] = p1.to(cutlass.BFloat16)
        tsum[0], tsum[1] = cute.arch.add_packed_f32x2((tsum[0], tsum[1]), (p0, p1))


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
    mMx,
    mIv,
    tmma_qk,
    tmma_pv,
    TQ: I32,
    NBLK: I32,
    ACC: I32,
    inv_h: cutlass.Float32,
    HG: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    NHG: cutlass.Constexpr,
    TCOLS: cutlass.Constexpr,
    ONL: cutlass.Constexpr,
):
    NQK = TOPK // MT
    NT = 128 * NHG
    HPG = HG // NHG
    REP = 8 if (HPG % 8 == 0) else 4
    SHB = 0
    _p = HPG
    while _p < 32:
        _p = _p * 2
        SHB = SHB + 1
    SHM = 1 << SHB
    NSUB = HPG // REP
    PGRP = 32 if (HG % 32 == 0) else 16
    P_STAGE = HG * KSUP
    NCP = MT * 32 // NT
    NCP4 = MT * 8 // NT
    NCQ = HG * D // 8 // NT
    NIDX = TOPK // NT

    tidx, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32
    sub = tidx % MT
    hgi = tidx // MT
    lg = sub // 32
    h0g = hgi * HPG

    q_beg = (blk * TQ) // NBLK
    q_end = ((blk + 1) * TQ) // NBLK
    nloc = q_end - q_beg

    # ---------------- static layouts ----------------
    a_l = sm100_utils.make_smem_layout_a(tmma_qk, (MT, HG, D), cutlass.BFloat16, KST)
    b_l = sm100_utils.make_smem_layout_b(tmma_qk, (MT, HG, D), cutlass.BFloat16, 1)
    va_l = sm100_utils.make_smem_layout_a(tmma_pv, (MT, HG, KSUP), cutlass.BFloat16, 2 * VST)
    pb_l = sm100_utils.make_smem_layout_b(tmma_pv, (MT, HG, KSUP), cutlass.BFloat16, VST)

    # ---------------- shared memory ----------------
    smem = utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, b_l.outer, byte_alignment=1024, swizzle=b_l.inner)
    sIdx = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, TOPK), stride=(TOPK, 1)), byte_alignment=16)
    sRedM = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HG, 4), stride=(4, 1)), byte_alignment=16)
    sRedS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HG, 4), stride=(4, 1)), byte_alignment=16)
    sMx = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HG), byte_alignment=16)
    sRl = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HG), byte_alignment=16)
    sTea = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((NHG, TOPK), stride=(TOPK, 1)),
        byte_alignment=16,
    )
    mbart = smem.allocate_tensor(cutlass.Int64, cute.make_layout(2 * NQK), byte_alignment=16)
    mbar = mbart.iterator

    kp = sK.iterator
    sK8 = cute.make_tensor(kp, cute.make_layout((KST, 4, MT, 8, 8), stride=(MT * D, 8192, 64, 8, 1)))
    qp = sQ.iterator
    sQ8 = cute.make_tensor(qp, cute.make_layout((4, HG, 8, 8), stride=(HG * 64, 64, 8, 1)))
    sV = cute.make_tensor(kp, va_l.outer)
    sV8 = cute.make_tensor(
        kp,
        cute.make_layout(
            (2, VST, 2, 8, KSUP, 8),
            stride=(MT * KSUP, 2 * MT * KSUP, KSUP * 64, 8, 64, 1),
        ),
    )
    pp = cute.recast_ptr(qp, pb_l.inner)
    sP = cute.make_tensor(pp, pb_l.outer)
    sPv = cute.make_tensor(pp, cute.make_layout((VST * P_STAGE // REP, REP), stride=(REP, 1)))

    cpa = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )

    # ---------------- mbarriers ----------------
    if tidx == 0:
        for i in cutlass.range_constexpr(2 * NQK):
            cute.arch.mbarrier_init(mbar + i, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tbar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tbar)
    tmem.allocate(TCOLS)
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(cutlass.Float32)

    accS_shape = tmma_qk.partition_shape_C((MT, HG))
    fakeS = tmma_qk.make_fragment_C(cute.append(accS_shape, NQK))
    tS = cute.make_tensor(tptr, fakeS.layout)
    fakeO = tmma_pv.make_fragment_C(accS_shape)
    tO0 = cute.make_tensor(tptr, fakeO.layout)  # aliases support tile 0
    tO1 = cute.make_tensor(tptr + HG, fakeO.layout)  # aliases support tile 1

    tCrK = tmma_qk.make_fragment_A(sK)
    tCrQ = tmma_qk.make_fragment_B(sQ)
    tCrV = tmma_pv.make_fragment_A(sV)
    tCrP = tmma_pv.make_fragment_B(sP)

    ldop = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(REP)), cutlass.Float32)
    tprobe = cute.make_tensor(tptr, cute.make_layout((MT, REP), stride=(65536, 1)))
    tld = tcgen05.make_tmem_copy(ldop, tprobe)
    thr_ld = tld.get_slice(sub)
    cId = cute.make_identity_tensor((MT, REP))
    tCrd = thr_ld.partition_D(cId)
    frg = cute.make_fragment_like(tCrd, cutlass.Float32)
    row = tCrd[0][0]

    pv4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    pl4 = cute.make_rmem_tensor((4,), cutlass.Float32)
    ob = cute.make_rmem_tensor((REP,), cutlass.BFloat16)
    mp = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    fs = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    tsum = cute.make_rmem_tensor((2,), cutlass.Float32)

    # ---------------- prologue ----------------
    t0 = q_beg
    for i in cutlass.range_constexpr(NIDX):
        sIdx[0, i * NT + tidx] = mIdx[t0, i * NT + tidx]
    cute.arch.barrier()
    # Prime K0, Q, and K1 as separate ordered groups.  The first QK walk can
    # wait for K0+Q while K1 remains in flight.
    for i in cutlass.range_constexpr(NCP4):
        c = i * NT + tidx
        r = c >> 3
        ch = c & 7
        gi = sIdx[0, r]
        for q in cutlass.range_constexpr(4):
            cute.copy(cpa, mK8[gi, q * 8 + ch, None], sK8[0, q, r, ch, None])
    cute.arch.cp_async_commit_group()
    for i in cutlass.range_constexpr(NCQ):
        c = i * NT + tidx
        dc = c % 32
        cute.copy(cpa, mQ8[t0, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
    cute.arch.cp_async_commit_group()
    for i in cutlass.range_constexpr(NCP4):
        c = i * NT + tidx
        r = c >> 3
        ch = c & 7
        gi = sIdx[0, MT + r]
        for q in cutlass.range_constexpr(4):
            cute.copy(cpa, mK8[gi, q * 8 + ch, None], sK8[1, q, r, ch, None])
    cute.arch.cp_async_commit_group()

    for qi in cutlass.range(nloc):
        cur = qi % 2
        nxt = 1 - cur
        ph = cutlass.Int32(qi % 2)
        t = q_beg + qi
        tn = t + 1
        if tn >= TQ:
            tn = TQ - 1
        vlen = mVlen[t]

        cute.arch.barrier()
        for i in cutlass.range_constexpr(NIDX):
            sIdx[nxt, i * NT + tidx] = mIdx[tn, i * NT + tidx]

        for j in cutlass.range_constexpr(HPG):
            mp[j] = cutlass.Float32(NEG)
            fs[j] = cutlass.Float32(0.0)

        # ---------------- QK ----------------
        # TopK-dependent support walks stay runtime loops: constexpr-unrolling
        # all 16 chunks of TopK=2048 explodes the compiled IR.
        for nt in cutlass.range(NQK):
            if nt == 0:
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            if warp_id == 0:
                for kb in cutlass.range_constexpr(D // 16):
                    tmma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
                    cute.gemm(
                        tmma_qk,
                        tS[None, None, None, nt],
                        tCrK[None, None, kb, nt % KST],
                        tCrQ[None, None, kb, 0],
                        tS[None, None, None, nt],
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + nt)
            if nt + 1 < NQK:
                if nt >= 1:
                    cute.arch.mbarrier_wait(mbar + (nt - 1), ph)
                    base = (nt + 1) * MT
                    st = (nt + 1) % KST
                    for i in cutlass.range_constexpr(NCP4):
                        c = i * NT + tidx
                        r = c >> 3
                        ch = c & 7
                        gi = sIdx[cur, base + r]
                        for q in cutlass.range_constexpr(4):
                            cute.copy(cpa, mK8[gi, q * 8 + ch, None], sK8[st, q, r, ch, None])
            if nt == NQK - 1:
                # K stage 0 retired: refill its exactly aliased V stage 0.
                cute.arch.mbarrier_wait(mbar + (nt - 1), ph)
                for i in cutlass.range_constexpr(NCP4):
                    c = i * NT + tidx
                    r = c >> 3
                    ch = c & 7
                    gi = sIdx[cur, r]
                    for q in cutlass.range_constexpr(4):
                        cute.copy(cpa, mV8[gi, q * 8 + ch, None], sV8[q >> 1, 0, q & 1, ch, r, None])
            cute.arch.cp_async_commit_group()
            if nt >= 1:
                mt = nt - 1
                s = mt * MT + row
                for hsub in cutlass.range_constexpr(NSUB):
                    tv = cute.make_tensor(
                        tptr + (mt * HG + h0g + hsub * REP),
                        cute.make_layout((MT, REP), stride=(65536, 1)),
                    )
                    cute.copy(tld, thr_ld.partition_S(tv), frg)
                    if cutlass.const_expr(ONL):
                        if s < vlen:
                            _online(mp, fs, frg, hsub * REP, REP)
                    else:
                        for j in cutlass.range_constexpr(REP):
                            x = frg[j] if s < vlen else cutlass.Float32(NEG)
                            mp[hsub * REP + j] = cute.arch.fmax(mp[hsub * REP + j], x)

        cute.arch.mbarrier_wait(mbar + (NQK - 1), ph)
        # K stage 1 retired: refill its aliased V stage 1 while softmax runs.
        for i in cutlass.range_constexpr(NCP4):
            c = i * NT + tidx
            r = c >> 3
            ch = c & 7
            gi = sIdx[cur, KSUP + r]
            for q in cutlass.range_constexpr(4):
                cute.copy(cpa, mV8[gi, q * 8 + ch, None], sV8[q >> 1, 1, q & 1, ch, r, None])
        cute.arch.cp_async_commit_group()

        s = (NQK - 1) * MT + row
        for hsub in cutlass.range_constexpr(NSUB):
            tv = cute.make_tensor(
                tptr + ((NQK - 1) * HG + h0g + hsub * REP),
                cute.make_layout((MT, REP), stride=(65536, 1)),
            )
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            if cutlass.const_expr(ONL):
                if s < vlen:
                    _online(mp, fs, frg, hsub * REP, REP)
            else:
                for j in cutlass.range_constexpr(REP):
                    x = frg[j] if s < vlen else cutlass.Float32(NEG)
                    mp[hsub * REP + j] = cute.arch.fmax(mp[hsub * REP + j], x)

        # ---------------- softmax state ----------------
        if cutlass.const_expr(ONL):
            # The shifted sum was folded online inside the support walk, where
            # it hides under the gathered-K stream, so no separate denominator
            # pass re-reads the whole TMEM score block.
            for jj in cutlass.range_constexpr(HPG):
                mw = cute.arch.warp_redux_sync(mp[jj], "fmax", nan=True)
                dd = cute.arch.fmax(mp[jj] - mw, cutlass.Float32(NEGC))
                fs[jj] = fs[jj] * cute.math.exp2(dd * C1, fastmath=True)
                mp[jj] = mw
            if cutlass.const_expr(HPG in (8, 16, 32)):
                _fold_lane_sum(fs, lane, HPG)
                if lane == 0:
                    for jj in cutlass.range_constexpr(HPG):
                        sRedM[h0g + jj, lg] = mp[jj]
                if (lane & (SHM - 1)) == 0:
                    sRedS[h0g + (lane >> SHB), lg] = fs[0]
            else:
                for jj in cutlass.range_constexpr(HPG):
                    fs[jj] = cute.arch.warp_reduction_sum(fs[jj])
                if lane == 0:
                    for jj in cutlass.range_constexpr(HPG):
                        sRedM[h0g + jj, lg] = mp[jj]
                        sRedS[h0g + jj, lg] = fs[jj]
            cute.arch.barrier()
            if tidx < HG:
                cute.autovec_copy(sRedM[tidx, None], pv4)
                cute.autovec_copy(sRedS[tidx, None], pl4)
                m = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))
                l = (
                    pl4[0] * cute.math.exp2((pv4[0] - m) * C1, fastmath=True)
                    + pl4[1] * cute.math.exp2((pv4[1] - m) * C1, fastmath=True)
                    + pl4[2] * cute.math.exp2((pv4[2] - m) * C1, fastmath=True)
                    + pl4[3] * cute.math.exp2((pv4[3] - m) * C1, fastmath=True)
                )
                lgv = cute.log2(l, approx=True, ftz=True)
                sMx[tidx] = m
                sRl[tidx] = cutlass.Float32(0.0) - lgv
                mMx[t, tidx] = m
                mIv[t, tidx] = cute.rcp(l, approx=True, ftz=True)
                mLse[t, tidx] = m * SCALE + lgv * LN2
            cute.arch.barrier()
            for jj in cutlass.range_constexpr(HPG):
                mp[jj] = cutlass.Float32(0.0) - sMx[h0g + jj]
                fs[jj] = sRl[h0g + jj]
        else:
            for jj in cutlass.range_constexpr(HPG):
                mp[jj] = cute.arch.warp_redux_sync(mp[jj], "fmax", nan=True)
            if lane == 0:
                for jj in cutlass.range_constexpr(HPG):
                    sRedM[h0g + jj, lg] = mp[jj]
            cute.arch.barrier()
            if tidx < HG:
                cute.autovec_copy(sRedM[tidx, None], pv4)
                sMx[tidx] = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))
            cute.arch.barrier()
            for jj in cutlass.range_constexpr(HPG):
                mp[jj] = cutlass.Float32(0.0) - sMx[h0g + jj]
                fs[jj] = cutlass.Float32(0.0)
            for nt in cutlass.range(NQK):
                s = nt * MT + row
                for hsub in cutlass.range_constexpr(NSUB):
                    tv = cute.make_tensor(
                        tptr + (nt * HG + h0g + hsub * REP),
                        cute.make_layout((MT, REP), stride=(65536, 1)),
                    )
                    cute.copy(tld, thr_ld.partition_S(tv), frg)
                    if s < vlen:
                        _exp_accum8(frg, mp, fs, hsub * REP, REP)
            for jj in cutlass.range_constexpr(HPG):
                fs[jj] = cute.arch.warp_reduction_sum(fs[jj])
            if lane == 0:
                for jj in cutlass.range_constexpr(HPG):
                    sRedS[h0g + jj, lg] = fs[jj]
            cute.arch.barrier()
            if tidx < HG:
                cute.autovec_copy(sRedS[tidx, None], pv4)
                l = pv4[0] + pv4[1] + pv4[2] + pv4[3]
                lgv = cute.log2(l, approx=True, ftz=True)
                sRl[tidx] = cutlass.Float32(0.0) - lgv
                mMx[t, tidx] = sMx[tidx]
                mIv[t, tidx] = cute.rcp(l, approx=True, ftz=True)
                mLse[t, tidx] = sMx[tidx] * SCALE + lgv * LN2
            cute.arch.barrier()
            for jj in cutlass.range_constexpr(HPG):
                fs[jj] = sRl[h0g + jj]

        # ---------------- probabilities + PV ----------------
        # Support tiles 0 and 1 are drained from TMEM before any PV issues so
        # that both output accumulators can alias their columns.
        for nt in cutlass.range_constexpr(VST):
            s = nt * MT + row
            tsum[0] = cutlass.Float32(0.0)
            tsum[1] = cutlass.Float32(0.0)
            for hsub in cutlass.range_constexpr(NSUB):
                tv = cute.make_tensor(
                    tptr + (nt * HG + h0g + hsub * REP),
                    cute.make_layout((MT, REP), stride=(65536, 1)),
                )
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                if s < vlen:
                    _exp_norm8(frg, mp, hsub * REP, fs, ob, tsum, REP)
                else:
                    for j in cutlass.range_constexpr(REP):
                        ob[j] = cutlass.BFloat16(0.0)
                h = h0g + hsub * REP
                pidx = (nt * P_STAGE + (h // PGRP) * (PGRP * KSUP) + row * PGRP + (h % PGRP)) // REP
                cute.autovec_copy(ob, sPv[pidx, None])
            sTea[hgi, s] = tsum[0] + tsum[1]

        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier()
        if warp_id == 0:
            for pst in cutlass.range_constexpr(VST):
                for kb in cutlass.range_constexpr(KSUP // 16):
                    tmma_pv.set(tcgen05.Field.ACCUMULATE, not (pst == 0 and kb == 0))
                    cute.gemm(
                        tmma_pv,
                        tO0,
                        tCrV[None, None, kb, 2 * pst],
                        tCrP[None, None, kb, pst],
                        tO0,
                    )
                    cute.gemm(
                        tmma_pv,
                        tO1,
                        tCrV[None, None, kb, 2 * pst + 1],
                        tCrP[None, None, kb, pst],
                        tO1,
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + NQK + pst)

        if cutlass.const_expr(NQK > VST):
            cute.arch.mbarrier_wait(mbar + NQK, ph)
            for i in cutlass.range_constexpr(NCP4):
                c = i * NT + tidx
                r = c >> 3
                ch = c & 7
                gi = sIdx[cur, VST * KSUP + r]
                for q in cutlass.range_constexpr(4):
                    cute.copy(cpa, mV8[gi, q * 8 + ch, None], sV8[q >> 1, 0, q & 1, ch, r, None])
            cute.arch.cp_async_commit_group()

        for nti in cutlass.range(NQK - VST):
            nt = nti + VST
            s = nt * MT + row
            pst = nt % VST
            cute.arch.mbarrier_wait(mbar + NQK + (nt - VST), ph)
            tsum[0] = cutlass.Float32(0.0)
            tsum[1] = cutlass.Float32(0.0)
            for hsub in cutlass.range_constexpr(NSUB):
                tv = cute.make_tensor(
                    tptr + (nt * HG + h0g + hsub * REP),
                    cute.make_layout((MT, REP), stride=(65536, 1)),
                )
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                if s < vlen:
                    _exp_norm8(frg, mp, hsub * REP, fs, ob, tsum, REP)
                else:
                    for j in cutlass.range_constexpr(REP):
                        ob[j] = cutlass.BFloat16(0.0)
                h = h0g + hsub * REP
                pidx = (pst * P_STAGE + (h // PGRP) * (PGRP * KSUP) + row * PGRP + (h % PGRP)) // REP
                cute.autovec_copy(ob, sPv[pidx, None])
            sTea[hgi, s] = tsum[0] + tsum[1]
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
            if warp_id == 0:
                for kb in cutlass.range_constexpr(KSUP // 16):
                    tmma_pv.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tmma_pv,
                        tO0,
                        tCrV[None, None, kb, 2 * pst],
                        tCrP[None, None, kb, pst],
                        tO0,
                    )
                    cute.gemm(
                        tmma_pv,
                        tO1,
                        tCrV[None, None, kb, 2 * pst + 1],
                        tCrP[None, None, kb, pst],
                        tO1,
                    )
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + NQK + nt)
            if nt + 1 < NQK:
                cute.arch.mbarrier_wait(mbar + NQK + (nt - 1), ph)
                vbase = (nt + 1) * KSUP
                vst = (nt + 1) % VST
                for i in cutlass.range_constexpr(NCP4):
                    c = i * NT + tidx
                    r = c >> 3
                    ch = c & 7
                    gi = sIdx[cur, vbase + r]
                    for q in cutlass.range_constexpr(4):
                        cute.copy(cpa, mV8[gi, q * 8 + ch, None], sV8[q >> 1, vst, q & 1, ch, r, None])
            if nt == NQK - 1:
                # V stage 0 is dead: prefetch the next query row's first K tile.
                cute.arch.mbarrier_wait(mbar + NQK + (nt - 1), ph)
                for i in cutlass.range_constexpr(NCP4):
                    c = i * NT + tidx
                    r = c >> 3
                    ch = c & 7
                    gi = sIdx[nxt, r]
                    for q in cutlass.range_constexpr(4):
                        cute.copy(cpa, mK8[gi, q * 8 + ch, None], sK8[0, q, r, ch, None])
            cute.arch.cp_async_commit_group()

        cute.arch.mbarrier_wait(mbar + NQK + (NQK - 1), ph)
        cute.arch.barrier()

        # K0 was issued from the final PV iteration.  With P and V now dead,
        # issue Q and K1 before teacher/output so their latency hides under
        # the row tail and the next walk starts from three ordered groups.
        for i in cutlass.range_constexpr(NCQ):
            c = i * NT + tidx
            dc = c % 32
            cute.copy(cpa, mQ8[tn, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
        cute.arch.cp_async_commit_group()
        for i in cutlass.range_constexpr(NCP4):
            c = i * NT + tidx
            r = c >> 3
            ch = c & 7
            gi = sIdx[nxt, MT + r]
            for q in cutlass.range_constexpr(4):
                cute.copy(cpa, mK8[gi, q * 8 + ch, None], sK8[1, q, r, ch, None])
        cute.arch.cp_async_commit_group()

        # ---------------- teacher ----------------
        for i in cutlass.range_constexpr(NIDX):
            c = i * NT + tidx
            acc = sTea[0, c]
            for g in cutlass.range_constexpr(1, NHG):
                acc = acc + sTea[g, c]
            val = acc * inv_h
            if ACC != 0:
                mTea[t, c] = mTea[t, c] + val
            else:
                mTea[t, c] = val

        # ---------------- output ----------------
        for hsub in cutlass.range_constexpr(NSUB):
            h = h0g + hsub * REP
            for ai in cutlass.range_constexpr(2):
                tv = cute.make_tensor(
                    tptr + (ai * HG + h),
                    cute.make_layout((MT, REP), stride=(65536, 1)),
                )
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                dv = ai * MT + row
                for j in cutlass.range_constexpr(REP):
                    mO[t, h + j, dv] = frg[j].to(cutlass.BFloat16)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier()

    cute.arch.cp_async_wait_group(0)
    tmem.relinquish_alloc_permit()
    cute.arch.barrier()
    tmem.free(tptr)


@cute.jit
def _launch(
    rawQ8,
    rawK8,
    rawV8,
    rawIdx,
    rawVlen,
    rawO,
    rawLse,
    rawTea,
    rawMx,
    rawIv,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    o_token_stride: I32,
    s_token_stride: I32,
    TQ: I32,
    TK: I32,
    NBLK: I32,
    ACC: I32,
    inv_h: cutlass.Float32,
    stream: cuda.CUstream,
    HG: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    NHG: cutlass.Constexpr,
    ONL: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    SMEM: cutlass.Constexpr,
    TCOLS: cutlass.Constexpr,
):
    q_token_stride = cute.assume(q_token_stride, 8)
    k_token_stride = cute.assume(k_token_stride, 8)
    v_token_stride = cute.assume(v_token_stride, 8)
    o_token_stride = cute.assume(o_token_stride, 8)

    mQ8 = cute.make_tensor(
        rawQ8.iterator,
        cute.make_layout((TQ, HG * D // 8, 8), stride=(q_token_stride, 8, 1)),
    )
    mK8 = cute.make_tensor(
        rawK8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(k_token_stride, 8, 1)),
    )
    mV8 = cute.make_tensor(
        rawV8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(v_token_stride, 8, 1)),
    )
    mIdx = cute.make_tensor(rawIdx.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mVlen = cute.make_tensor(rawVlen.iterator, cute.make_layout((TQ,), stride=(1,)))
    mO = cute.make_tensor(rawO.iterator, cute.make_layout((TQ, HG, D), stride=(o_token_stride, D, 1)))
    mLse = cute.make_tensor(rawLse.iterator, cute.make_layout((TQ, HG), stride=(s_token_stride, 1)))
    mTea = cute.make_tensor(rawTea.iterator, cute.make_layout((TQ, TOPK), stride=(TOPK, 1)))
    mMx = cute.make_tensor(rawMx.iterator, cute.make_layout((TQ, HG), stride=(s_token_stride, 1)))
    mIv = cute.make_tensor(rawIv.iterator, cute.make_layout((TQ, HG), stride=(s_token_stride, 1)))

    tmma_qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        (MT, HG),
    )
    tmma_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.MN,
        OperandMajorMode.MN,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        (MT, HG),
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
        mMx,
        mIv,
        tmma_qk,
        tmma_pv,
        TQ,
        NBLK,
        ACC,
        inv_h,
        HG,
        TOPK,
        NHG,
        TCOLS,
        ONL,
    ).launch(grid=[NBLK, 1, 1], block=[NT, 1, 1], smem=SMEM, stream=stream)


_cache = {}


def _dynamic_tensor(tensor, alignment):
    return from_dlpack(tensor.detach(), assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


def _validate_qkv(name, tensor, heads):
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"SSA {name} must be BF16, got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[1:] != (heads, D):
        raise ValueError(f"SSA {name} must be [T,{heads},{D}], got {tuple(tensor.shape)}")
    if tensor.stride(2) != 1 or (heads > 1 and tensor.stride(1) != D):
        raise ValueError(f"SSA {name} head/dimension storage must be contiguous")
    if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
        raise ValueError(f"SSA {name} base/token pitch must be 16-byte aligned")


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
    if q.ndim != 3 or q.shape[1] not in SUPPORTED_HQ or q.shape[2] != D:
        raise ValueError("SSA Q must be BF16 [Tq,Hq,256] with Hq in {16,32,96}")
    tq, hq, _ = q.shape
    topk = selected_indices.shape[-1]
    if topk not in SUPPORTED_TOPK:
        raise ValueError("SSA TopK must be one of 512, 1024, 2048")
    tk = k.shape[0]
    if v.shape != k.shape or k.ndim != 3 or k.shape[1:] != (1, D):
        raise ValueError("SSA K/V must be matching BF16 [Tk,1,256] tensors")
    _validate_qkv("Q", q, hq)
    _validate_qkv("K", k, 1)
    _validate_qkv("V", v, 1)
    if selected_indices.shape != (tq, topk) or selected_indices.dtype != torch.int32:
        raise ValueError("SSA selected indices must be int32 [Tq,TopK]")
    if valid_lengths.shape != (tq,) or valid_lengths.dtype != torch.int32:
        raise ValueError("SSA valid lengths must be int32 [Tq]")
    if out.shape != q.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("SSA output must be compact BF16 shaped like Q")
    for tensor, shape in (
        (lse, (tq, hq)),
        (teacher, (tq, topk)),
        (softmax_max, (tq, hq)),
        (softmax_inv_sum, (tq, hq)),
    ):
        if tensor.shape != shape or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError(f"SSA FP32 state must be compact with shape {shape}")
    others = (
        k,
        v,
        selected_indices,
        valid_lengths,
        out,
        lse,
        teacher,
        softmax_max,
        softmax_inv_sum,
    )
    if not q.is_cuda or any(t.device != q.device for t in others):
        raise ValueError("all SSA tensors must live on one CUDA device")

    hg, nhg, nt, tcols, smem, onl = _plan(hq, topk)
    ngrp = hq // hg
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
    nblk = min(nsm, tq)
    inv_h = cutlass.Float32(1.0 / float(hq))

    idx = selected_indices.view(tq, topk)
    vl = valid_lengths.view(tq)
    k8 = k.as_strided((tk, D // 8, 8), (k.stride(0), 8, 1))
    v8 = v.as_strided((tk, D // 8, 8), (v.stride(0), 8, 1))

    key = (q.device.index, hq, topk)
    fn = _cache.get(key)

    for gi in range(ngrp):
        h0 = gi * hg
        qg = q[:, h0 : h0 + hg, :]
        og = out[:, h0 : h0 + hg, :]
        lsg = lse[:, h0 : h0 + hg]
        mxg = softmax_max[:, h0 : h0 + hg]
        ivg = softmax_inv_sum[:, h0 : h0 + hg]
        q8 = qg.as_strided((tq, hg * D // 8, 8), (q.stride(0), 8, 1))
        args = tuple(t.detach() for t in (q8, k8, v8, idx, vl, og, lsg, teacher, mxg, ivg))
        if fn is None:
            dl = [_dynamic_tensor(t, a) for t, a in zip(args, (16, 16, 16, 16, 4, 16, 16, 16, 16, 16))]
            fn = cute.compile(
                _launch,
                *dl,
                I32(q.stride(0)),
                I32(k.stride(0)),
                I32(v.stride(0)),
                I32(hq * D),
                I32(hq),
                I32(tq),
                I32(tk),
                I32(nblk),
                I32(0),
                inv_h,
                stream=make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
                HG=hg,
                TOPK=topk,
                NHG=nhg,
                ONL=onl,
                NT=nt,
                SMEM=smem,
                TCOLS=tcols,
            )
            _cache[key] = fn
        fn(
            *args,
            I32(q.stride(0)),
            I32(k.stride(0)),
            I32(v.stride(0)),
            I32(hq * D),
            I32(hq),
            I32(tq),
            I32(tk),
            I32(nblk),
            I32(1 if gi else 0),
            inv_h,
        )
