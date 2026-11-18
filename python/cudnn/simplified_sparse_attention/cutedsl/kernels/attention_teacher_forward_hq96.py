# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SSA sparse-attention + teacher forward for Hq=96, Hkv=1, D=256, TopK=512.

The compute schedule is the independently audited ``ssa_hq96_b57_repaired_v2``
seed.  Only its caller ABI is modernized here: packed token extents and
token pitches are runtime values, DLPack views are detached, and launches use
the caller's CUDA stream.
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

HQ = 96
D = 256
TOPK = 512
MT = 128  # support rows per QK MMA tile
NQK = TOPK // MT  # 4
KSUP = 128  # support rows per PV chunk
KST = 2  # K pipeline stages
VST = 2  # V (and P) pipeline stages
NW = 16
NT = NW * 32  # 512 threads
NHG = NT // MT  # 4 head groups
HPG = HQ // NHG  # 24 heads per group
NSUB = HPG // 8  # 3 eight-wide head chunks per thread

SCALE = 0.0625
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
INV_H = 1.0 / float(HQ)
NEG = float("-inf")

# ---- shared-memory element counts -------------------------------------------
K_ELEMS = KST * MT * D  # 65536  (also VST*2*MT*KSUP)
QP_ELEMS = HQ * D  # 24576  (also VST*HQ*KSUP)
P_STAGE = HQ * KSUP  # 12288
P_GRP = 32  # MN_SW64 atom width for HQ=96 bf16
SMEM_BYTES = (
    (K_ELEMS + QP_ELEMS) * 2  # K/V union + Q/P union
    + 2 * TOPK * 4  # double-buffered indices
    + 2 * HQ * 4 * 4  # sRedM + sRedS
    + 2 * HQ * 4  # sMx + sRl
    + NHG * TOPK * 4  # teacher partials
    + 16 * 8  # mbarriers
    + 2048  # alignment slack
)


@cute.jit
def _exp_accum8(frg, neg_max, fs, base: cutlass.Constexpr):
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(0, 8, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[base + j], neg_max[base + j + 1]))
        x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
        e0 = cute.math.exp2(x0, fastmath=True)
        e1 = cute.math.exp2(x1, fastmath=True)
        fs[base + j], fs[base + j + 1] = cute.arch.add_packed_f32x2((fs[base + j], fs[base + j + 1]), (e0, e1))


@cute.jit
def _exp_norm8(frg, neg_max, base: cutlass.Constexpr, nr, ob, tsum):
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(0, 8, 2):
        x0, x1 = cute.arch.add_packed_f32x2((frg[j], frg[j + 1]), (neg_max[base + j], neg_max[base + j + 1]))
        x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
        e0 = cute.math.exp2(x0, fastmath=True)
        e1 = cute.math.exp2(x1, fastmath=True)
        p0 = e0 * nr[base + j]
        p1 = e1 * nr[base + j + 1]
        ob[j] = p0.to(cutlass.BFloat16)
        ob[j + 1] = p1.to(cutlass.BFloat16)
        tsum[0] = tsum[0] + p0 + p1


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
    NBLK: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    warp_id = tidx // 32
    lane = tidx % 32
    sub = tidx % MT  # TMEM lane == support row inside a tile
    hg = tidx // MT  # head group 0..3
    lg = sub // 32  # lane group 0..3
    h0g = hg * HPG  # first head owned by this thread

    q_beg = (blk * TQ) // NBLK
    q_end = ((blk + 1) * TQ) // NBLK
    nloc = q_end - q_beg

    # ---------------- static layouts ----------------
    a_l = sm100_utils.make_smem_layout_a(tmma_qk, (MT, HQ, D), cutlass.BFloat16, KST)
    b_l = sm100_utils.make_smem_layout_b(tmma_qk, (MT, HQ, D), cutlass.BFloat16, 1)
    va_l = sm100_utils.make_smem_layout_a(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, 2 * VST)
    pb_l = sm100_utils.make_smem_layout_b(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, VST)

    # ---------------- shared memory ----------------
    smem = utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, b_l.outer, byte_alignment=1024, swizzle=b_l.inner)
    sIdx = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, TOPK), stride=(TOPK, 1)), byte_alignment=16)
    sRedM = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sRedS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sMx = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sRl = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sTea = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NHG, TOPK), stride=(TOPK, 1)), byte_alignment=16)
    mbart = smem.allocate_tensor(cutlass.Int64, cute.make_layout(16), byte_alignment=16)
    mbar = mbart.iterator

    kp = sK.iterator
    # gathered-K cp.async view: (stage, kc, m, kb, i)
    sK8 = cute.make_tensor(kp, cute.make_layout((KST, 4, MT, 8, 8), stride=(MT * D, 8192, 64, 8, 1)))
    qp = sQ.iterator
    # Q cp.async view: (kc, h, kb, i)
    sQ8 = cute.make_tensor(qp, cute.make_layout((4, HQ, 8, 8), stride=(HQ * 64, 64, 8, 1)))

    # V^T stages alias the retired K stages exactly (MN_SW128 == K_SW128 swizzle).
    sV = cute.make_tensor(kp, va_l.outer)
    # V cp.async view: (half, stage, mh, m8, k, i)
    sV8 = cute.make_tensor(kp, cute.make_layout((2, VST, 2, 8, KSUP, 8), stride=(MT * KSUP, 2 * MT * KSUP, KSUP * 64, 8, 64, 1)))

    # P^T operand aliases Q exactly (HQ*D == VST*HQ*KSUP elements).
    pp = cute.recast_ptr(qp, pb_l.inner)
    sP = cute.make_tensor(pp, pb_l.outer)
    # P^T 128-bit store view
    sPv = cute.make_tensor(pp, cute.make_layout((VST * P_STAGE // 8, 8), stride=(8, 1)))

    cpa = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128)

    # ---------------- mbarriers ----------------
    if tidx == 0:
        for i in cutlass.range_constexpr(16):
            cute.arch.mbarrier_init(mbar + i, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tbar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tbar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(cutlass.Float32)

    accS_shape = tmma_qk.partition_shape_C((MT, HQ))
    fakeS = tmma_qk.make_fragment_C(cute.append(accS_shape, NQK))
    tS = cute.make_tensor(tptr, fakeS.layout)
    fakeO = tmma_pv.make_fragment_C(accS_shape)
    tO0 = cute.make_tensor(tptr, fakeO.layout)  # aliases retired tS tile 0
    tO1 = cute.make_tensor(tptr + NQK * HQ, fakeO.layout)  # columns 384..479

    tCrK = tmma_qk.make_fragment_A(sK)
    tCrQ = tmma_qk.make_fragment_B(sQ)
    tCrV = tmma_pv.make_fragment_A(sV)
    tCrP = tmma_pv.make_fragment_B(sP)

    # TMEM load plumbing (32 lanes x 8 columns per warp)
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
    mp = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    fs = cute.make_rmem_tensor((HPG,), cutlass.Float32)
    tsum = cute.make_rmem_tensor((1,), cutlass.Float32)

    # ---------------- prologue ----------------
    t0 = q_beg if q_beg < TQ else TQ - 1
    sIdx[0, tidx] = mIdx[t0, tidx]
    for i in cutlass.range_constexpr(HQ * D // 8 // NT):
        c = i * NT + tidx
        dc = c % 32
        cute.copy(cpa, mQ8[t0, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
    cute.arch.cp_async_commit_group()
    cute.arch.barrier()
    for i in cutlass.range_constexpr(MT * 32 // NT):
        c = i * NT + tidx
        r = c >> 5
        ch = c & 31
        cute.copy(cpa, mK8[sIdx[0, r], ch, None], sK8[0, ch >> 3, r, ch & 7, None])
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

        # publish next query's indices (buffer free since previous iteration)
        cute.arch.barrier()
        sIdx[nxt, tidx] = mIdx[tn, tidx]

        for j in cutlass.range_constexpr(HPG):
            mp[j] = cutlass.Float32(NEG)

        # ---------------- QK ----------------
        for nt in cutlass.range_constexpr(NQK):
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()
            if warp_id == 0:
                for kb in cutlass.range_constexpr(D // 16):
                    tmma_qk.set(tcgen05.Field.ACCUMULATE, kb != 0)
                    cute.gemm(tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, nt % KST], tCrQ[None, None, kb, 0], tS[None, None, None, nt])
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + nt)
            if cutlass.const_expr(nt + 1 < NQK):
                if cutlass.const_expr(nt >= 1):
                    cute.arch.mbarrier_wait(mbar + (nt - 1), ph)
                base = (nt + 1) * MT
                st = (nt + 1) % KST
                for i in cutlass.range_constexpr(MT * 32 // NT):
                    c = i * NT + tidx
                    r = c >> 5
                    ch = c & 31
                    cute.copy(cpa, mK8[sIdx[cur, base + r], ch, None], sK8[st, ch >> 3, r, ch & 7, None])
            if cutlass.const_expr(nt == NQK - 1):
                # K stage 0 retired after QK tile 2; refill its V alias.
                cute.arch.mbarrier_wait(mbar + (nt - 1), ph)
                for i in cutlass.range_constexpr(MT * 32 // NT):
                    c = i * NT + tidx
                    r = c >> 5
                    ch = c & 31
                    cute.copy(cpa, mV8[sIdx[cur, r], ch, None], sV8[ch >> 4, 0, (ch >> 3) & 1, ch & 7, r, None])
            cute.arch.cp_async_commit_group()
            if cutlass.const_expr(nt >= 1):
                mt = nt - 1
                s = mt * MT + row
                for hsub in cutlass.range_constexpr(NSUB):
                    tv = cute.make_tensor(tptr + (mt * HQ + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
                    cute.copy(tld, thr_ld.partition_S(tv), frg)
                    for j in cutlass.range_constexpr(8):
                        x = frg[j] if s < vlen else cutlass.Float32(NEG)
                        x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
                        mp[hsub * 8 + j] = cute.arch.fmax(mp[hsub * 8 + j], x)

        cute.arch.mbarrier_wait(mbar + (NQK - 1), ph)
        s = (NQK - 1) * MT + row
        for hsub in cutlass.range_constexpr(NSUB):
            tv = cute.make_tensor(tptr + ((NQK - 1) * HQ + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
            cute.copy(tld, thr_ld.partition_S(tv), frg)
            for j in cutlass.range_constexpr(8):
                x = frg[j] if s < vlen else cutlass.Float32(NEG)
                x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
                mp[hsub * 8 + j] = cute.arch.fmax(mp[hsub * 8 + j], x)

        if lane == 0:
            for j in cutlass.range_constexpr(HPG):
                sRedM[h0g + j, lg] = mp[j]
        cute.arch.barrier()
        if tidx < HQ:
            cute.autovec_copy(sRedM[tidx, None], pv4)
            sMx[tidx] = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))
        cute.arch.barrier()

        for j in cutlass.range_constexpr(HPG):
            mp[j] = cutlass.Float32(0.0) - sMx[h0g + j]
            fs[j] = cutlass.Float32(0.0)

        # ---------------- denominator pass ----------------
        for nt in cutlass.range_constexpr(NQK):
            s = nt * MT + row
            for hsub in cutlass.range_constexpr(NSUB):
                tv = cute.make_tensor(tptr + (nt * HQ + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                if s < vlen:
                    _exp_accum8(frg, mp, fs, hsub * 8)

        for j in cutlass.range_constexpr(HPG):
            fs[j] = cute.arch.warp_reduction_sum(fs[j])
        if lane == 0:
            for j in cutlass.range_constexpr(HPG):
                sRedS[h0g + j, lg] = fs[j]
        cute.arch.barrier()
        if tidx < HQ:
            cute.autovec_copy(sRedS[tidx, None], pv4)
            l = pv4[0] + pv4[1] + pv4[2] + pv4[3]
            sRl[tidx] = cute.rcp(l, approx=True, ftz=True)
            mSoftmaxMax[t, tidx] = sMx[tidx]
            mSoftmaxInvSum[t, tidx] = sRl[tidx]
            mLse[t, tidx] = sMx[tidx] * SCALE + cute.log2(l, approx=True, ftz=True) * LN2
        cute.arch.barrier()
        for j in cutlass.range_constexpr(HPG):
            fs[j] = sRl[h0g + j]

        # ---------------- probability materialization + PV ----------------
        for nt in cutlass.range_constexpr(NQK):
            s = nt * MT + row
            pst = nt % VST
            if cutlass.const_expr(nt >= VST):
                cute.arch.mbarrier_wait(mbar + 4 + (nt - VST), ph)
            tsum[0] = cutlass.Float32(0.0)
            for hsub in cutlass.range_constexpr(NSUB):
                tv = cute.make_tensor(tptr + (nt * HQ + h0g + hsub * 8), cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                if s < vlen:
                    _exp_norm8(frg, mp, hsub * 8, fs, ob, tsum)
                else:
                    for j in cutlass.range_constexpr(8):
                        ob[j] = cutlass.BFloat16(0.0)
                h = h0g + hsub * 8
                pidx = (pst * P_STAGE + (h // P_GRP) * (P_GRP * KSUP) + row * P_GRP + (h % P_GRP)) // 8
                cute.autovec_copy(ob, sPv[pidx, None])
            sTea[hg, s] = tsum[0]
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
            if warp_id == 0:
                for kb in cutlass.range_constexpr(KSUP // 16):
                    tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                    cute.gemm(tmma_pv, tO0, tCrV[None, None, kb, 2 * pst], tCrP[None, None, kb, pst], tO0)
                    cute.gemm(tmma_pv, tO1, tCrV[None, None, kb, 2 * pst + 1], tCrP[None, None, kb, pst], tO1)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 4 + nt)
            if cutlass.const_expr(nt + 1 < NQK):
                if cutlass.const_expr(nt >= 1):
                    cute.arch.mbarrier_wait(mbar + 4 + (nt - 1), ph)
                vbase = (nt + 1) * KSUP
                vst = (nt + 1) % VST
                for i in cutlass.range_constexpr(MT * 32 // NT):
                    c = i * NT + tidx
                    r = c >> 5
                    ch = c & 31
                    cute.copy(cpa, mV8[sIdx[cur, vbase + r], ch, None], sV8[ch >> 4, vst, (ch >> 3) & 1, ch & 7, r, None])
            if cutlass.const_expr(nt == NQK - 1):
                # V stage 0 is dead after PV tile 2: prefetch next query's K tile 0.
                cute.arch.mbarrier_wait(mbar + 4 + (NQK - 2), ph)
                for i in cutlass.range_constexpr(MT * 32 // NT):
                    c = i * NT + tidx
                    r = c >> 5
                    ch = c & 31
                    cute.copy(cpa, mK8[sIdx[nxt, r], ch, None], sK8[0, ch >> 3, r, ch & 7, None])
            cute.arch.cp_async_commit_group()

        cute.arch.mbarrier_wait(mbar + 4 + (NQK - 1), ph)
        cute.arch.barrier()

        # ---------------- teacher reduction ----------------
        mTea[t, tidx] = (sTea[0, tidx] + sTea[1, tidx] + sTea[2, tidx] + sTea[3, tidx]) * INV_H

        # ---------------- output epilogue ----------------
        for hsub in cutlass.range_constexpr(NSUB):
            h = h0g + hsub * 8
            for ai in cutlass.range_constexpr(2):
                tv = cute.make_tensor(tptr + (ai * (NQK * HQ) + h), cute.make_layout((MT, 8), stride=(65536, 1)))
                cute.copy(tld, thr_ld.partition_S(tv), frg)
                dv = ai * MT + row
                for j in cutlass.range_constexpr(8):
                    mO[t, h + j, dv] = frg[j].to(cutlass.BFloat16)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier()

        # P is dead: refill the alias with the next query's Q.
        for i in cutlass.range_constexpr(HQ * D // 8 // NT):
            c = i * NT + tidx
            dc = c % 32
            cute.copy(cpa, mQ8[tn, c, None], sQ8[dc // 8, c // 32, dc % 8, None])
        cute.arch.cp_async_commit_group()

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
    rawSoftmaxMax,
    rawSoftmaxInvSum,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    TK: I32,
    NBLK: I32,
    stream: cuda.CUstream,
):
    # DLPack's fully dynamic descriptor also makes the 8-element vector mode
    # dynamic.  CuTe's 128-bit gmem->smem copy requires that mode to be
    # compile-time static.  Consume only the raw iterators and rebuild the
    # logical tensors with runtime token extents/pitches but static inner
    # layouts.  This preserves zero-copy pitched Q/K/V while allowing one
    # compiled executable to serve every packed THD physical length.
    # Runtime pitches are validated by ``run`` in BF16 elements.  Preserve
    # that 16-byte divisibility in the IR so vector-copy alignment is not
    # discarded merely because the pitch is no longer a constexpr.
    q_token_stride = cute.assume(q_token_stride, 8)
    k_token_stride = cute.assume(k_token_stride, 8)
    v_token_stride = cute.assume(v_token_stride, 8)

    mQ8 = cute.make_tensor(
        rawQ8.iterator,
        cute.make_layout((TQ, HQ * D // 8, 8), stride=(q_token_stride, 8, 1)),
    )
    mK8 = cute.make_tensor(
        rawK8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(k_token_stride, 8, 1)),
    )
    mV8 = cute.make_tensor(
        rawV8.iterator,
        cute.make_layout((TK, D // 8, 8), stride=(v_token_stride, 8, 1)),
    )
    mIdx = cute.make_tensor(
        rawIdx.iterator,
        cute.make_layout((TQ, TOPK), stride=(TOPK, 1)),
    )
    mVlen = cute.make_tensor(rawVlen.iterator, cute.make_layout((TQ,), stride=(1,)))
    mO = cute.make_tensor(
        rawO.iterator,
        cute.make_layout((TQ, HQ, D), stride=(HQ * D, D, 1)),
    )
    mLse = cute.make_tensor(rawLse.iterator, cute.make_layout((TQ, HQ), stride=(HQ, 1)))
    mTea = cute.make_tensor(
        rawTea.iterator,
        cute.make_layout((TQ, TOPK), stride=(TOPK, 1)),
    )
    mSoftmaxMax = cute.make_tensor(
        rawSoftmaxMax.iterator,
        cute.make_layout((TQ, HQ), stride=(HQ, 1)),
    )
    mSoftmaxInvSum = cute.make_tensor(
        rawSoftmaxInvSum.iterator,
        cute.make_layout((TQ, HQ), stride=(HQ, 1)),
    )

    tmma_qk = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.K, OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    tmma_pv = sm100_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16, OperandMajorMode.MN, OperandMajorMode.MN, cutlass.Float32, tcgen05.CtaGroup.ONE, (MT, HQ)
    )
    _ssa(mQ8, mK8, mV8, mIdx, mVlen, mO, mLse, mTea, mSoftmaxMax, mSoftmaxInvSum, tmma_qk, tmma_pv, TQ, NBLK).launch(
        grid=[NBLK, 1, 1],
        block=[NT, 1, 1],
        smem=SMEM_BYTES,
        stream=stream,
    )


_cache = {}


def _dynamic_tensor(tensor, alignment):
    """Expose runtime token counts/strides so packed THD shapes share one JIT."""
    return from_dlpack(tensor.detach(), assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


def _validate_qkv_vector_layout(name, tensor, expected_heads):
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"SSA {name} must be BF16, got {tensor.dtype}")
    if tensor.ndim != 3 or tensor.shape[1:] != (expected_heads, D):
        raise ValueError(f"SSA {name} must be [T,{expected_heads},{D}], got {tuple(tensor.shape)}")
    # A singleton head axis has no observable stride. a fused QKV view
    # may retain the full token pitch while its only D-vector stays compact.
    if tensor.stride(2) != 1 or (expected_heads > 1 and tensor.stride(1) != D):
        raise ValueError(f"SSA {name} head/dimension storage must be contiguous; " f"got strides {tuple(tensor.stride())}")
    if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
        raise ValueError(f"SSA {name} base address and token stride must be 16-byte aligned " "for 128-bit Q/K/V copies")


@torch.no_grad()
def run(q, k, v, selected_indices, valid_lengths, out, lse, teacher, softmax_max, softmax_inv_sum):
    if q.ndim != 3 or q.shape[1:] != (HQ, D):
        raise ValueError(f"SSA Q must be [Tq,{HQ},{D}], got {tuple(q.shape)}")
    tq = q.shape[0]
    if k.ndim != 3 or k.shape[1:] != (1, D):
        raise ValueError(f"SSA K must be [Tk,1,{D}], got {tuple(k.shape)}")
    tk = k.shape[0]
    if tq <= 0 or tk <= 0:
        raise ValueError("SSA query and key/value token streams must be nonempty")
    if v.shape != k.shape:
        raise ValueError(f"SSA V must match K shape {tuple(k.shape)}, got {tuple(v.shape)}")
    _validate_qkv_vector_layout("Q", q, HQ)
    _validate_qkv_vector_layout("K", k, 1)
    _validate_qkv_vector_layout("V", v, 1)

    if selected_indices.shape != (tq, TOPK) or selected_indices.dtype != torch.int32:
        raise ValueError(f"SSA selected indices must be contiguous int32 [Tq,{TOPK}]")
    if valid_lengths.shape != (tq,) or valid_lengths.dtype != torch.int32:
        raise ValueError("SSA valid lengths must be contiguous int32 [Tq]")
    if out.shape != q.shape or out.dtype != torch.bfloat16:
        raise ValueError("SSA output must be BF16 with the same shape as Q")
    fp32_states = {
        "LSE": (lse, (tq, HQ)),
        "teacher": (teacher, (tq, TOPK)),
        "softmax max": (softmax_max, (tq, HQ)),
        "softmax inverse sum": (softmax_inv_sum, (tq, HQ)),
    }
    for name, (tensor, shape) in fp32_states.items():
        if tensor.shape != shape or tensor.dtype != torch.float32:
            raise ValueError(f"SSA {name} must be contiguous FP32 {shape}")
    all_tensors = (
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
    )
    if not q.is_cuda or any(tensor.device != q.device for tensor in all_tensors[1:]):
        raise ValueError("all SSA forward tensors must be on one CUDA device")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            selected_indices,
            valid_lengths,
            out,
            lse,
            teacher,
            softmax_max,
            softmax_inv_sum,
        )
    ):
        raise ValueError("SSA forward outputs, state, and support metadata must be contiguous")

    q8 = q.as_strided((tq, HQ * D // 8, 8), (q.stride(0), 8, 1))
    k8 = k.as_strided((tk, D // 8, 8), (k.stride(0), 8, 1))
    v8 = v.as_strided((tk, D // 8, 8), (v.stride(0), 8, 1))
    ov = out.view(tq, HQ, D)
    idx = selected_indices.view(tq, TOPK)
    vl = valid_lengths.view(tq)
    ls = lse.view(tq, HQ)
    te = teacher.view(tq, TOPK)
    mx = softmax_max.view(tq, HQ)
    inv = softmax_inv_sum.view(tq, HQ)
    runtime_args = tuple(tensor.detach() for tensor in (q8, k8, v8, idx, vl, ov, ls, te, mx, inv))

    # Packed THD adds a data-dependent number of alignment rows between
    # documents.  Treat token extents and pitches as runtime values; otherwise
    # every new physical Tq/Tk pair synchronously recompiles all ranks and the
    # faster DP replicas appear to spend seconds inside the next collective.
    nsm = torch.cuda.get_device_properties(q.device).multi_processor_count
    nblk = min(nsm, tq)

    key = q.device.index
    fn = _cache.get(key)
    if fn is None:
        args = [_dynamic_tensor(tensor, alignment) for tensor, alignment in zip(runtime_args, (16, 16, 16, 16, 4, 16, 16, 16, 16, 16))]
        fn = cute.compile(
            _launch,
            *args,
            I32(q8.stride(0)),
            I32(k8.stride(0)),
            I32(v8.stride(0)),
            I32(tq),
            I32(tk),
            I32(nblk),
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _cache[key] = fn
    fn(
        *runtime_args,
        I32(q8.stride(0)),
        I32(k8.stride(0)),
        I32(v8.stride(0)),
        I32(tq),
        I32(tk),
        I32(nblk),
    )
