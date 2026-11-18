# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused SSA sparse-attention + teacher forward for Hq=32, Hkv=1, D=256, TopK=512.

The gather stream is the whole cost of this operator: every query row pulls
512 scattered 512-byte K rows plus 512 scattered V rows through the SM.  So the
kernel is organised entirely around keeping that stream saturated.

Each query row is decomposed into 16 uniform 32 KiB *granules*:

    g = 0..7   K[128 support, 128 of D]      -> QK UMMA (K-major A, 8 k-blocks)
    g = 8..15  V^T[128 of D, 128 support]    -> PV UMMA (MN-major A, 8 k-blocks)

They flow through one 5-slot circular SMEM buffer with `cp.async` group depth
control, so four granules are always in flight instead of the single fat 64 KiB
tile a two-stage pipeline can sustain.  The stream never stops at a query-row
boundary: the tail of row t issues the head of row t+1.

The whole 512x32 score tile lives in registers (32 FP32 per thread), so softmax
costs exactly one TMEM read pass and one exp2 pass.
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

HQ = 32
D = 256
DH = 128  # D elements per granule
TOPK = 512
MT = 128  # support rows per QK MMA tile
NQK = TOPK // MT  # 4
KSUP = 128  # support rows per PV chunk
NS = 5  # circular granule slots
NG = 16  # granules per query row
PST = 2  # probability stages
NW = 16
NT = NW * 32  # 512 threads
HGN = NT // 128  # 4 head groups
HPG = HQ // HGN  # 8 heads per group
NCP = MT * DH // 8 // NT  # 4 128-bit gather copies per thread per granule
NCQ = HQ * D // 8 // NT  # 2 128-bit Q copies per thread
SCALE = 0.0625
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
INV_H = 1.0 / HQ
NEG = float("-inf")

GE = MT * DH  # 16384 elements per granule

SMEM_BYTES = (
    NS * GE * 2  # circular granule buffer
    + HQ * D * 2  # Q
    + PST * KSUP * HQ * 2  # P stages
    + 2 * TOPK * 4  # two index buffers
    + 2 * HQ * HGN * 4  # partial max / sum
    + 2 * HQ * 4  # per-head max / recip
    + HGN * TOPK * 4  # teacher partials
    + 16 * 8  # mbarriers
    + 2048
)


@cute.jit
def _gather(sI, sKg, sVg, mK8, mV8, cpa, tidx, gid: cutlass.Constexpr, slot: cutlass.Constexpr):
    """Issue one 32 KiB granule of 128-bit cp.async gathers."""
    if cutlass.const_expr(gid < 8):
        base = (gid // 2) * MT
        half = gid % 2
        for i in cutlass.range_constexpr(NCP):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mK8[sI[base + r], half * 16 + ch, None], sKg[slot, ch >> 3, r, ch & 7, None])
    else:
        j = gid - 8
        base = (j // 2) * KSUP
        half = j % 2
        for i in cutlass.range_constexpr(NCP):
            c = i * NT + tidx
            r = c >> 4
            ch = c & 15
            cute.copy(cpa, mV8[sI[base + r], half * 16 + ch, None], sVg[slot, ch >> 3, ch & 7, r, None])


@cute.jit
def _query_body(
    mQ8,
    mK8,
    mV8,
    mVlen,
    mO,
    mLse,
    mTea,
    mSoftmaxMax,
    mSoftmaxInvSum,
    sI,
    sInext,
    sKg,
    sVg,
    sQ8,
    sP8,
    sTea,
    sRedM,
    sRedS,
    sMx,
    sRl,
    mbar,
    tptr,
    tS,
    tO,
    tCrK,
    tCrQ,
    tCrV,
    tCrP,
    tld,
    thr_ld,
    f0,
    f1,
    f2,
    f3,
    mp,
    fs,
    nr,
    ob,
    pv4,
    tsum,
    tmma_qk,
    tmma_pv,
    cpa,
    tidx,
    warp_id,
    lane,
    hg,
    lg,
    row,
    t,
    tnext,
    ph: cutlass.Constexpr,
    roff: cutlass.Constexpr,
    last: cutlass.Constexpr,
):
    frg = (f0, f1, f2, f3)
    vlen = mVlen[t]
    hbase = hg * HPG

    for j in cutlass.range_constexpr(8):
        mp[j] = cutlass.Float32(NEG)

    # ------------------------------------------------ QK : granules 0..7
    for g in cutlass.range_constexpr(8):
        nt = g // 2
        half = g % 2
        cute.arch.cp_async_wait_group(NS - 2)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(DH // 16):
                tmma_qk.set(tcgen05.Field.ACCUMULATE, not (half == 0 and kb == 0))
                cute.gemm(
                    tmma_qk, tS[None, None, None, nt], tCrK[None, None, kb, (roff + g) % NS], tCrQ[None, None, half * 8 + kb, 0], tS[None, None, None, nt]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + g)
        # Slot (roff+g-1) is about to be refilled: its MMA must have retired.
        if cutlass.const_expr(g >= 1):
            cute.arch.mbarrier_wait(mbar + (g - 1), ph)
        gid = g + NS - 1
        if cutlass.const_expr(gid < NG):
            _gather(sI, sKg, sVg, mK8, mV8, cpa, tidx, gid, (roff + gid) % NS)
        elif cutlass.const_expr(not last):
            _gather(sInext, sKg, sVg, mK8, mV8, cpa, tidx, gid - NG, (roff + gid) % NS)
        cute.arch.cp_async_commit_group()
        # The wait above retired support tile nt-1; fold it into the max now.
        if cutlass.const_expr(g >= 2 and half == 0):
            mt = nt - 1
            s = mt * MT + row
            tv = cute.make_tensor(tptr + (mt * HQ + hbase), cute.make_layout((MT, 8), stride=(65536, 1)))
            fm = frg[mt]
            cute.copy(tld, thr_ld.partition_S(tv), fm)
            for j in cutlass.range_constexpr(8):
                x = fm[j] if s < vlen else cutlass.Float32(NEG)
                x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
                mp[j] = cute.arch.fmax(mp[j], x)

    cute.arch.mbarrier_wait(mbar + 7, ph)
    s = 3 * MT + row
    tv = cute.make_tensor(tptr + (3 * HQ + hbase), cute.make_layout((MT, 8), stride=(65536, 1)))
    fm = frg[3]
    cute.copy(tld, thr_ld.partition_S(tv), fm)
    for j in cutlass.range_constexpr(8):
        x = fm[j] if s < vlen else cutlass.Float32(NEG)
        x = cute.arch.warp_redux_sync(x, "fmax", nan=True)
        mp[j] = cute.arch.fmax(mp[j], x)

    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedM[hbase + j, lg] = mp[j]
    cute.arch.barrier()
    if tidx < HQ:
        cute.autovec_copy(sRedM[tidx, None], pv4)
        sMx[tidx] = cute.arch.fmax(cute.arch.fmax(pv4[0], pv4[1]), cute.arch.fmax(pv4[2], pv4[3]))
    cute.arch.barrier()

    # ------------------------------------------------ single exp2 pass
    factor = SCALE * LOG2E
    for j in cutlass.range_constexpr(8):
        mp[j] = cutlass.Float32(0.0) - sMx[hbase + j]
        fs[j] = cutlass.Float32(0.0)

    for nt in cutlass.range_constexpr(NQK):
        s = nt * MT + row
        f = frg[nt]
        if s < vlen:
            for j in cutlass.range_constexpr(0, 8, 2):
                x0, x1 = cute.arch.add_packed_f32x2((f[j], f[j + 1]), (mp[j], mp[j + 1]))
                x0, x1 = cute.arch.mul_packed_f32x2((x0, x1), (factor, factor))
                e0 = cute.math.exp2(x0, fastmath=True)
                e1 = cute.math.exp2(x1, fastmath=True)
                f[j] = e0
                f[j + 1] = e1
                fs[j], fs[j + 1] = cute.arch.add_packed_f32x2((fs[j], fs[j + 1]), (e0, e1))
        else:
            for j in cutlass.range_constexpr(8):
                f[j] = cutlass.Float32(0.0)

    for j in cutlass.range_constexpr(8):
        fs[j] = cute.arch.warp_reduction_sum(fs[j])
    if lane == 0:
        for j in cutlass.range_constexpr(8):
            sRedS[hbase + j, lg] = fs[j]
    cute.arch.barrier()
    if tidx < HQ:
        cute.autovec_copy(sRedS[tidx, None], pv4)
        l = pv4[0] + pv4[1] + pv4[2] + pv4[3]
        sRl[tidx] = cute.rcp(l, approx=True, ftz=True)
        mSoftmaxMax[t, tidx] = sMx[tidx]
        mSoftmaxInvSum[t, tidx] = sRl[tidx]
        mLse[t, tidx] = sMx[tidx] * SCALE + cute.log2(l, approx=True, ftz=True) * LN2
    cute.arch.barrier()

    # ------------------------------------------------ PV : granules 8..15
    for g in cutlass.range_constexpr(8, 16):
        j8 = g - 8
        nt = j8 // 2
        half = j8 % 2
        if cutlass.const_expr(half == 0):
            s = nt * MT + row
            tsum[0] = cutlass.Float32(0.0)
            for jj in cutlass.range_constexpr(8):
                nr[jj] = sRl[hbase + jj]
            fn = frg[nt]
            for jj in cutlass.range_constexpr(8):
                p = fn[jj] * nr[jj]
                ob[jj] = p.to(cutlass.BFloat16)
                tsum[0] = tsum[0] + p
            cute.autovec_copy(ob, sP8[nt % PST, row, hg, None])
            sTea[hg, s] = tsum[0]
        cute.arch.cp_async_wait_group(NS - 2)
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()
        if warp_id == 0:
            for kb in cutlass.range_constexpr(KSUP // 16):
                tmma_pv.set(tcgen05.Field.ACCUMULATE, not (nt == 0 and kb == 0))
                cute.gemm(
                    tmma_pv, tO[None, None, None, half], tCrV[None, None, kb, (roff + g) % NS], tCrP[None, None, kb, nt % PST], tO[None, None, None, half]
                )
            with cute.arch.elect_one():
                tcgen05.commit(mbar + g)
        cute.arch.mbarrier_wait(mbar + (g - 1), ph)
        gid = g + NS - 1
        if cutlass.const_expr(gid < NG):
            _gather(sI, sKg, sVg, mK8, mV8, cpa, tidx, gid, (roff + gid) % NS)
        elif cutlass.const_expr(not last):
            _gather(sInext, sKg, sVg, mK8, mV8, cpa, tidx, gid - NG, (roff + gid) % NS)
        if cutlass.const_expr(g == 8 and not last):
            # Every QK MMA of this row has retired, so Q is free for row t+1.
            for i in cutlass.range_constexpr(NCQ):
                c = i * NT + tidx
                h = c // 32
                dc = c % 32
                cute.copy(cpa, mQ8[tnext, c, None], sQ8[dc // 8, h, dc % 8, None])
        cute.arch.cp_async_commit_group()

    cute.arch.barrier()
    if tidx < TOPK:
        acc = sTea[0, tidx] + sTea[1, tidx] + sTea[2, tidx] + sTea[3, tidx]
        mTea[t, tidx] = acc * INV_H

    cute.arch.mbarrier_wait(mbar + 15, ph)
    cute.arch.barrier()

    for ai in cutlass.range_constexpr(2):
        tv = cute.make_tensor(tptr + (NQK * HQ + ai * HQ + hbase), cute.make_layout((MT, 8), stride=(65536, 1)))
        fa = frg[ai]
        cute.copy(tld, thr_ld.partition_S(tv), fa)
        dv = ai * MT + row
        for j in cutlass.range_constexpr(8):
            mO[t, hbase + j, dv] = fa[j].to(cutlass.BFloat16)

    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier()


@cute.kernel
def _ssa(mQ8, mK8, mV8, mIdx, mVlen, mO, mLse, mTea, mSoftmaxMax, mSoftmaxInvSum, tmma_qk, tmma_pv, TQ: I32):
    tidx, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    t0 = 2 * blk
    t1 = t0 + 1
    if t1 >= TQ:
        t1 = t0
    warp_id = tidx // 32
    lane = tidx % 32
    sub = tidx % 128
    hg = tidx // 128
    lg = sub // 32

    a_l = sm100_utils.make_smem_layout_a(tmma_qk, (MT, HQ, DH), cutlass.BFloat16, NS)
    b_l = sm100_utils.make_smem_layout_b(tmma_qk, (MT, HQ, D), cutlass.BFloat16, 1)
    va_l = sm100_utils.make_smem_layout_a(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, NS)
    pb_l = sm100_utils.make_smem_layout_b(tmma_pv, (MT, HQ, KSUP), cutlass.BFloat16, PST)

    smem = utils.SmemAllocator()
    sK = smem.allocate_tensor(cutlass.BFloat16, a_l.outer, byte_alignment=1024, swizzle=a_l.inner)
    sQ = smem.allocate_tensor(cutlass.BFloat16, b_l.outer, byte_alignment=1024, swizzle=b_l.inner)
    sP = smem.allocate_tensor(cutlass.BFloat16, pb_l.outer, byte_alignment=1024, swizzle=pb_l.inner)
    sIdx = smem.allocate_tensor(cutlass.Int32, cute.make_layout(TOPK), byte_alignment=16)
    sIdx1 = smem.allocate_tensor(cutlass.Int32, cute.make_layout(TOPK), byte_alignment=16)
    sRedM = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, HGN), stride=(HGN, 1)), byte_alignment=16)
    sRedS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HQ, HGN), stride=(HGN, 1)), byte_alignment=16)
    sMx = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sRl = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HQ), byte_alignment=16)
    sTea = smem.allocate_tensor(cutlass.Float32, cute.make_layout((HGN, TOPK), stride=(TOPK, 1)), byte_alignment=16)
    mbart = smem.allocate_tensor(cutlass.Int64, cute.make_layout(16), byte_alignment=16)
    mbar = mbart.iterator

    kp = sK.iterator
    sKg = cute.make_tensor(kp, cute.make_layout((NS, 2, MT, 8, 8), stride=(GE, 8192, 64, 8, 1)))
    sV = cute.make_tensor(kp, va_l.outer)
    sVg = cute.make_tensor(kp, cute.make_layout((NS, 2, 8, KSUP, 8), stride=(GE, 8192, 8, 64, 1)))
    qp = sQ.iterator
    sQ8 = cute.make_tensor(qp, cute.make_layout((4, HQ, 8, 8), stride=(HQ * 64, 64, 8, 1)))
    pp = sP.iterator
    sP8 = cute.make_tensor(pp, cute.make_layout((PST, KSUP, HGN, HPG), stride=(KSUP * HQ, HQ, HPG, 1)))

    cpa = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128)

    if tidx == 0:
        for i in cutlass.range_constexpr(16):
            cute.arch.mbarrier_init(mbar + i, 1)
    cute.arch.mbarrier_init_fence()

    if tidx < TOPK:
        sIdx[tidx] = mIdx[t0, tidx]
        sIdx1[tidx] = mIdx[t1, tidx]

    for i in cutlass.range_constexpr(NCQ):
        c = i * NT + tidx
        h = c // 32
        dc = c % 32
        cute.copy(cpa, mQ8[t0, c, None], sQ8[dc // 8, h, dc % 8, None])
    cute.arch.cp_async_commit_group()

    tbar = pipeline.NamedBarrier(barrier_id=1, num_threads=NT)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tbar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tptr = tmem.retrieve_ptr(cutlass.Float32)

    accS_shape = tmma_qk.partition_shape_C((MT, HQ))
    fakeS = tmma_qk.make_fragment_C(cute.append(accS_shape, NQK))
    tS = cute.make_tensor(tptr, fakeS.layout)
    fakeO = tmma_pv.make_fragment_C(cute.append(accS_shape, 2))
    tO = cute.make_tensor(tptr + NQK * HQ, fakeO.layout)

    tCrK = tmma_qk.make_fragment_A(sK)
    tCrQ = tmma_qk.make_fragment_B(sQ)
    tCrV = tmma_pv.make_fragment_A(sV)
    tCrP = tmma_pv.make_fragment_B(sP)

    ldop = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), cutlass.Float32)
    tprobe = cute.make_tensor(tptr, cute.make_layout((MT, 8), stride=(65536, 1)))
    tld = tcgen05.make_tmem_copy(ldop, tprobe)
    thr_ld = tld.get_slice(sub)
    cId = cute.make_identity_tensor((MT, 8))
    tCrd = thr_ld.partition_D(cId)
    f0 = cute.make_fragment_like(tCrd, cutlass.Float32)
    f1 = cute.make_fragment_like(tCrd, cutlass.Float32)
    f2 = cute.make_fragment_like(tCrd, cutlass.Float32)
    f3 = cute.make_fragment_like(tCrd, cutlass.Float32)
    row = tCrd[0][0]

    pv4 = cute.make_rmem_tensor((HGN,), cutlass.Float32)
    ob = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    mp = cute.make_rmem_tensor((8,), cutlass.Float32)
    fs = cute.make_rmem_tensor((8,), cutlass.Float32)
    nr = cute.make_rmem_tensor((8,), cutlass.Float32)
    tsum = cute.make_rmem_tensor((1,), cutlass.Float32)

    cute.arch.barrier()

    for gid in cutlass.range_constexpr(NS - 1):
        _gather(sIdx, sKg, sVg, mK8, mV8, cpa, tidx, gid, gid % NS)
        cute.arch.cp_async_commit_group()

    _query_body(
        mQ8,
        mK8,
        mV8,
        mVlen,
        mO,
        mLse,
        mTea,
        mSoftmaxMax,
        mSoftmaxInvSum,
        sIdx,
        sIdx1,
        sKg,
        sVg,
        sQ8,
        sP8,
        sTea,
        sRedM,
        sRedS,
        sMx,
        sRl,
        mbar,
        tptr,
        tS,
        tO,
        tCrK,
        tCrQ,
        tCrV,
        tCrP,
        tld,
        thr_ld,
        f0,
        f1,
        f2,
        f3,
        mp,
        fs,
        nr,
        ob,
        pv4,
        tsum,
        tmma_qk,
        tmma_pv,
        cpa,
        tidx,
        warp_id,
        lane,
        hg,
        lg,
        row,
        t0,
        t1,
        0,
        0,
        False,
    )
    _query_body(
        mQ8,
        mK8,
        mV8,
        mVlen,
        mO,
        mLse,
        mTea,
        mSoftmaxMax,
        mSoftmaxInvSum,
        sIdx1,
        sIdx1,
        sKg,
        sVg,
        sQ8,
        sP8,
        sTea,
        sRedM,
        sRedS,
        sMx,
        sRl,
        mbar,
        tptr,
        tS,
        tO,
        tCrK,
        tCrQ,
        tCrV,
        tCrP,
        tld,
        thr_ld,
        f0,
        f1,
        f2,
        f3,
        mp,
        fs,
        nr,
        ob,
        pv4,
        tsum,
        tmma_qk,
        tmma_pv,
        cpa,
        tidx,
        warp_id,
        lane,
        hg,
        lg,
        row,
        t1,
        t1,
        1,
        NG % NS,
        True,
    )

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
    _ssa(mQ8, mK8, mV8, mIdx, mVlen, mO, mLse, mTea, mSoftmaxMax, mSoftmaxInvSum, tmma_qk, tmma_pv, TQ).launch(
        grid=[(TQ + 1) // 2, 1, 1],
        block=[NT, 1, 1],
        smem=SMEM_BYTES,
        stream=stream,
    )


_cache = {}


def _dynamic_tensor(tensor, alignment):
    """Expose runtime token counts/strides so packed THD shapes share one JIT."""
    return from_dlpack(tensor, assumed_align=alignment, enable_tvm_ffi=True).mark_layout_dynamic()


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
    tq = q.shape[0]
    tk = k.shape[0]
    if v.shape[0] != tk:
        raise ValueError(f"SSA attention K/V row mismatch: Tk={tk}, Tv={v.shape[0]}")
    _validate_qkv_vector_layout("Q", q, HQ)
    _validate_qkv_vector_layout("K", k, 1)
    _validate_qkv_vector_layout("V", v, 1)
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
    if mx.dtype != torch.float32 or inv.dtype != torch.float32 or not mx.is_contiguous() or not inv.is_contiguous():
        raise ValueError("SSA saved softmax max/inv_sum must be contiguous FP32 [Tq, Hq]")

    # Packed THD adds a data-dependent number of alignment rows between
    # documents.  Treat token extents and pitches as runtime values; otherwise
    # every new physical Tq/Tk pair synchronously recompiles all ranks and the
    # faster DP replicas appear to spend seconds inside the next collective.
    key = q.device.index
    fn = _cache.get(key)
    if fn is None:
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
        fn = cute.compile(
            _launch,
            *args,
            I32(q8.stride(0)),
            I32(k8.stride(0)),
            I32(v8.stride(0)),
            I32(tq),
            I32(tk),
            stream=make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _cache[key] = fn
    fn(
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
        I32(q8.stride(0)),
        I32(k8.stride(0)),
        I32(v8.stride(0)),
        I32(tq),
        I32(tk),
    )
