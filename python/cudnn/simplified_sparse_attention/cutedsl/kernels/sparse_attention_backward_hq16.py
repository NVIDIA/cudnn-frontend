# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSA sparse-attention backward for Hq=16, Hkv=1, D=256, TopK=512 (CuTe DSL).

One CTA owns one query token and all 16 query heads, so the MQA dK/dV scatter
is reduced over heads inside the CTA before it reaches global memory.  The CTA
walks the query's selected support slots in chunks of BR=128 and issues five
tcgen05 (UMMA) contractions per chunk.

At Hq=16 the tensor-core work per query token is 5x smaller than the Hq=80
design while the fp32 dK/dV scatter-add cost is unchanged, so this kernel is
almost entirely bound by L2 atomics.  dK and dV therefore live in one
interleaved fp32 workspace

    W[key, dim, 0] = dK[key, dim]      W[key, dim, 1] = dV[key, dim]

so a single thread owns both fp32 values for a (key, dim) pair and issues one
vector atomic instead of two scalar ones.

The cancellation-sensitive dQ/dK contractions use FP16 operands with FP32
accumulation when every Q/K/dS value consumed by the current support chunk is
inside FP16's finite range.  A CTA-uniform range vote selects a TF32
high+residual fallback otherwise.  The fallback preserves FP32's exponent
range instead of silently overflowing finite BF16 inputs.  QK/dP and dV keep
their BF16 operand contract.  After QK/dP retire, selected K and each Q feature
half are converted into dead shared-memory storage.  The dV contraction keeps
the forward BF16 probability convention.
"""

import math
import os

import cuda.bindings.driver as cuda
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from ._stream_lock import serialize_workspace

BF16 = cutlass.BFloat16
F16 = cutlass.Float16
TF32 = cutlass.TFloat32
F32 = cutlass.Float32
I32 = cutlass.Int32

HQ = 16  # query heads (exact for this campaign)
DIM = 256  # head dim (qk == v)
TOPK = 512
BR = 128  # support slots per chunk
NTHREADS = 512
NWG = NTHREADS // 128

SCALE = 0.0625
LOG2E = float(math.log2(math.e))
SCALE_LOG2E = SCALE * LOG2E
F16_MAX = 65504.0

# Test-only path accounting.  When enabled before module import, the first
# head slot of the internal norm workspace is overwritten after each query
# with that query's number of TF32-fallback chunks.  The production ABI and
# returned tensors are unchanged, and the branch is compile-time dead by
# default.
_PATH_DIAGNOSTIC = os.getenv("SSA_HQ16_PATH_DIAGNOSTIC", "0") == "1"

# TMEM column layout (512 columns).
TM_A = 0  # S^T (128x16) then dK^T dim-tile (128x128)
TM_B = 128  # dP^T (128x16) then dV^T dim-tile (128x128)
TM_DQ0 = 256  # dQ^T tile0 : 128 x 16 fp32 (persistent across chunks)
TM_DQ1 = 272  # dQ^T tile1

Q_T1_OFF = HQ * 64 * 2  # second 128-dim tile of a (16,256) K-major tile
K_T1_OFF = BR * 64 * 2  # second 128-dim tile of a (128,256) K-major tile
DS_ELEMS = BR * HQ  # one (128,16) K-major tile
K_HALF_ELEMS = BR * BR  # one (128,128) feature-half tile
# V is dead after dP: TF32 dS high/residual (4*DS), BF16 P (DS),
# TF32 Q-half (2*DS), explicit K-major TF32 dS transposes (4*DS), and
# branch-exclusive FP16 dS plus its K-major transpose (2*DS).
assert 13 * DS_ELEMS <= BR * DIM
# The original BF16 K allocation is exactly one TF32 K feature-half scratch.
assert K_HALF_ELEMS * 2 == BR * DIM

# two eight-head softmax groups, one per warpgroup
SM_WG = (0, 1)


# --------------------------------------------------------------------------
# helper kernels
# --------------------------------------------------------------------------
@cute.kernel
def _zero_kernel(mW: cute.Tensor, n: I32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 8
    if base + 8 <= n:
        fa = cute.make_rmem_tensor(cute.make_layout(8), F32)
        fa.fill(0.0)
        cute.autovec_copy(fa, cute.make_tensor(mW.iterator + base, cute.make_layout(8)))


@cute.kernel
def _cvt_kernel(mW: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, n: I32):
    """n = TK*DIM ; thread deinterleaves 4 FP32 (dK,dV) pairs."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 4
    if base + 4 <= n:
        f = cute.make_rmem_tensor(cute.make_layout(8), F32)
        ok = cute.make_rmem_tensor(cute.make_layout(4), F32)
        ov = cute.make_rmem_tensor(cute.make_layout(4), F32)
        cute.autovec_copy(cute.make_tensor(mW.iterator + base * 2, cute.make_layout(8)), f)
        for i in cutlass.range_constexpr(4):
            ok[i] = f[2 * i]
            ov[i] = f[2 * i + 1]
        cute.autovec_copy(ok, cute.make_tensor(mDK.iterator + base, cute.make_layout(4)))
        cute.autovec_copy(ov, cute.make_tensor(mDV.iterator + base, cute.make_layout(4)))


# --------------------------------------------------------------------------
# exact softmax delta prepass:  delta[h] = sum_s P_fp32[s,h] * dP[s,h]
# --------------------------------------------------------------------------
@cute.kernel
def _delta_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mDO: cute.Tensor,
    mSoftmaxMax: cute.Tensor,
    mSoftmaxInvSum: cute.Tensor,
    mIdx: cute.Tensor,
    mVlen: cute.Tensor,
    mDelta: cute.Tensor,
    mNorm: cute.Tensor,
    KQ_mma: cute.TiledMma,
    lQ: cute.ComposedLayout,
    lK: cute.ComposedLayout,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    nctas: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    wg = tidx // 128
    t128 = tidx % 128

    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sDO = smem.allocate_tensor(BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sK = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sV = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sIdx = smem.allocate_tensor(I32, cute.make_layout(BR), byte_alignment=16)
    sMax = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sInv = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sRed = smem.allocate_tensor(F32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sRedP = smem.allocate_tensor(F32, cute.make_layout((HQ, 4), stride=(4, 1)), byte_alignment=16)
    sPart = smem.allocate_tensor(F32, cute.make_layout((TOPK // BR, HQ), stride=(HQ, 1)), byte_alignment=16)
    sPartP = smem.allocate_tensor(F32, cute.make_layout((TOPK // BR, HQ), stride=(HQ, 1)), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=16)

    sQ2 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQ.inner), lQ.outer)
    sDO2 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lQ.inner), lQ.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner), lK.outer)
    sV2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lK.inner), lK.outer)

    kbase = mK.iterator.toint()
    vbase = mV.iterator.toint()
    q_stride_bytes = q_token_stride * I32(2)
    k_stride_bytes = k_token_stride * I32(2)
    v_stride_bytes = v_token_stride * I32(2)
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 32)), cute.make_layout((1, 8)))
    thr_g2s = tiled_g2s.get_slice(lane)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
    cute.arch.mbarrier_init_fence()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C((BR, HQ)))
    tS = cute.make_tensor(tmem_ptr + TM_A, tS_f.layout)
    tDP = cute.make_tensor(tmem_ptr + TM_B, tS_f.layout)
    fK = KQ_mma.make_fragment_A(sK)
    fQ = KQ_mma.make_fragment_B(sQ)
    fV = KQ_mma.make_fragment_A(sV)
    fDO = KQ_mma.make_fragment_B(sDO)

    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), F32)
    tS2 = tS[(None, None), 0, 0]
    tDP2 = tDP[(None, None), 0, 0]
    cSN = cute.make_identity_tensor((BR, HQ))
    cp_s = tcgen05.make_tmem_copy(ld8, tS2)
    thr_s = cp_s.get_slice(t128)
    tr_S = thr_s.partition_S(tS2)
    tr_DP = thr_s.partition_S(tDP2)
    tr_cS = thr_s.partition_D(cSN)
    slot_m = cute.get(tr_cS[0], mode=[0])

    nkQ = cute.size(fQ, mode=[2])
    ph = I32(0)
    d0 = lane * 8
    nq = (TQ - bidx + nctas - I32(1)) // nctas
    for qi in cutlass.range(nq):
        t = bidx + qi * nctas
        vlen = mVlen[t]
        nchunk = (vlen + (BR - 1)) // BR

        if tidx < (TOPK // BR) * HQ:
            sPart[tidx // HQ, tidx % HQ] = F32(0.0)
            sPartP[tidx // HQ, tidx % HQ] = F32(0.0)
        if tidx < HQ:
            sMax[tidx] = mSoftmaxMax[t * HQ + tidx]
            sInv[tidx] = mSoftmaxInvSum[t * HQ + tidx]
        cute.arch.barrier()

        if vlen > 0:
            qptr = mQ.iterator.toint() + t * q_stride_bytes
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                cute.copy(
                    tiled_g2s,
                    thr_g2s.partition_S(
                        cute.make_tensor(
                            cute.make_ptr(
                                BF16,
                                qptr + h * (DIM * 2),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((1, DIM)),
                        )
                    ),
                    thr_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
                )
            cute.arch.cp_async_commit_group()
            qbase = t * (HQ * DIM)
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                goff = qbase + h * DIM + d0
                rdo = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                cute.autovec_copy(cute.make_tensor(mDO.iterator + goff, cute.make_layout((1, 8))), rdo)
                cute.autovec_copy(rdo, cute.local_tile(sDO2, (1, 8), (h, lane)))

            for c in cutlass.range(nchunk):
                cbase = c * BR
                if tidx < BR:
                    index = I32(0)
                    if cbase + tidx < vlen:
                        index = mIdx[t * TOPK + cbase + tidx]
                    sIdx[tidx] = index
                cute.arch.barrier()

                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    kbase + sIdx[r] * k_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
                    )
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    vbase + sIdx[r] * v_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()

                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 0)

                cute.arch.mbarrier_wait(mbar + 0, ph)
                for j in cutlass.range_constexpr(HQ // 8):
                    if wg == SM_WG[j]:
                        rS = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                        rDP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                        rDelta = cute.make_rmem_tensor(cute.make_layout((1, 8)), F32)
                        rPSum = cute.make_rmem_tensor(cute.make_layout((1, 8)), F32)
                        cute.copy(cp_s, tr_S[None, None, j], rS)
                        cute.copy(cp_s, tr_DP[None, None, j], rDP)
                        for e in cutlass.range_constexpr(0, 8, 2):
                            h0 = j * 8 + e
                            h1 = h0 + 1
                            p0 = F32(0.0)
                            p1 = F32(0.0)
                            if cbase + slot_m < vlen:
                                x0, x1 = cute.arch.add_packed_f32x2(
                                    (rS[e], rS[e + 1]),
                                    (F32(0.0) - sMax[h0], F32(0.0) - sMax[h1]),
                                )
                                x0, x1 = cute.arch.mul_packed_f32x2(
                                    (x0, x1),
                                    (F32(SCALE_LOG2E), F32(SCALE_LOG2E)),
                                )
                                p0 = cute.math.exp2(x0, fastmath=True) * sInv[h0]
                                p1 = cute.math.exp2(x1, fastmath=True) * sInv[h1]
                            rDelta[0, e] = p0 * rDP[e]
                            rDelta[0, e + 1] = p1 * rDP[e + 1]
                            rPSum[0, e] = p0
                            rPSum[0, e + 1] = p1
                        for e in cutlass.range_constexpr(8):
                            total = cute.arch.warp_reduction_sum(rDelta[0, e])
                            total_p = cute.arch.warp_reduction_sum(rPSum[0, e])
                            if lane == 0:
                                sRed[j * 8 + e, t128 // 32] = total
                                sRedP[j * 8 + e, t128 // 32] = total_p
                    pass
                cute.arch.fence_view_async_tmem_load()
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()

                if tidx < HQ:
                    total = F32(0.0)
                    total_p = F32(0.0)
                    for w in cutlass.range_constexpr(4):
                        total = total + sRed[tidx, w]
                        total_p = total_p + sRedP[tidx, w]
                    sPart[c, tidx] = total
                    sPartP[c, tidx] = total_p
                cute.arch.barrier()
                ph = ph ^ I32(1)

        if tidx < HQ:
            total = F32(0.0)
            total_p = F32(0.0)
            for c in cutlass.range_constexpr(TOPK // BR):
                total = total + sPart[c, tidx]
                total_p = total_p + sPartP[c, tidx]
            norm = F32(0.0)
            if vlen > 0:
                norm = F32(1.0) / total_p
            mDelta[t * HQ + tidx] = total * norm
            mNorm[t * HQ + tidx] = norm
        cute.arch.barrier()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


# --------------------------------------------------------------------------
# main kernel
# --------------------------------------------------------------------------
@cute.kernel
def _bwd_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mDO: cute.Tensor,
    mSoftmaxMax: cute.Tensor,
    mSoftmaxInvSum: cute.Tensor,
    mIdx: cute.Tensor,
    mVlen: cute.Tensor,
    mDelta: cute.Tensor,
    mNorm: cute.Tensor,
    mDQ: cute.Tensor,
    mW: cute.Tensor,
    KQ_mma: cute.TiledMma,
    dSQ_mma: cute.TiledMma,
    dPV_mma: cute.TiledMma,
    dSK_mma: cute.TiledMma,
    dSQ16_mma: cute.TiledMma,
    dSK16_mma: cute.TiledMma,
    lQ: cute.ComposedLayout,
    lK: cute.ComposedLayout,
    lKtf: cute.ComposedLayout,
    lQscratch: cute.ComposedLayout,
    lDS: cute.ComposedLayout,
    lDSkt_store: cute.ComposedLayout,
    lK16: cute.ComposedLayout,
    lQ16: cute.ComposedLayout,
    lDS16: cute.ComposedLayout,
    lDS16kt_store: cute.ComposedLayout,
    lP: cute.ComposedLayout,
    lQT: cute.ComposedLayout,
    lDOT: cute.ComposedLayout,
    lKT: cute.ComposedLayout,
    lDSkt: cute.ComposedLayout,
    lQT16: cute.ComposedLayout,
    lKT16: cute.ComposedLayout,
    lDS16kt: cute.ComposedLayout,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lDSs: cute.ComposedLayout,
    lDS16s: cute.ComposedLayout,
    lPs: cute.ComposedLayout,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    nctas: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    wg = tidx // 128
    t128 = tidx % 128

    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sDO = smem.allocate_tensor(BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sK = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sV = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sIdx = smem.allocate_tensor(I32, cute.make_layout(TOPK), byte_alignment=16)
    sMax = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sInv = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sDel = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sNorm = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sBad = smem.allocate_tensor(I32, cute.make_layout(16), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 4, byte_alignment=16)

    # Once dP^T has retired V, its 64 KiB staging allocation holds TF32 dS
    # high/residual, BF16 P, one TF32 Q feature half, and K-major transposes
    # of both dS terms. K uses its dead 64 KiB allocation as TF32 scratch.
    sDSh = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSs.inner, dtype=TF32), lDSs.outer)
    sDSl = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * DS_ELEMS, lDSs.inner, dtype=TF32), lDSs.outer)
    sP = cute.make_tensor(cute.recast_ptr(sV.iterator + 4 * DS_ELEMS, lPs.inner, dtype=BF16), lPs.outer)
    sQscratch = cute.make_tensor(cute.recast_ptr(sV.iterator + 5 * DS_ELEMS, lQT.inner, dtype=TF32), lQT.outer)
    sDShkt = cute.make_tensor(cute.recast_ptr(sV.iterator + 7 * DS_ELEMS, lDSkt.inner, dtype=TF32), lDSkt.outer)
    sDSlkt = cute.make_tensor(cute.recast_ptr(sV.iterator + 9 * DS_ELEMS, lDSkt.inner, dtype=TF32), lDSkt.outer)
    sDS16 = cute.make_tensor(cute.recast_ptr(sV.iterator + 11 * DS_ELEMS, lDS16s.inner, dtype=F16), lDS16s.outer)
    sDS16kt = cute.make_tensor(cute.recast_ptr(sV.iterator + 12 * DS_ELEMS, lDS16kt.inner, dtype=F16), lDS16kt.outer)
    sQ16 = cute.make_tensor(cute.recast_ptr(sV.iterator + 5 * DS_ELEMS, lQT16.inner, dtype=F16), lQT16.outer)
    sK16_0 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT16.inner, dtype=F16), lKT16.outer)
    sK16_1 = cute.make_tensor(cute.recast_ptr(sK.iterator + K_T1_OFF, lKT16.inner, dtype=F16), lKT16.outer)

    sQ2 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQ.inner), lQ.outer)
    sDO2 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lQ.inner), lQ.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner), lK.outer)
    sV2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lK.inner), lK.outer)
    sKtf2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKtf.inner, dtype=TF32), lKtf.outer)
    sDSh2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lDS.inner, dtype=TF32), lDS.outer)
    sDSl2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * DS_ELEMS, lDS.inner, dtype=TF32), lDS.outer)
    sP2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 4 * DS_ELEMS, lP.inner, dtype=BF16), lP.outer)
    sQscratch2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 5 * DS_ELEMS, lQscratch.inner, dtype=TF32), lQscratch.outer)
    sDShkt2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 7 * DS_ELEMS, lDSkt_store.inner, dtype=TF32), lDSkt_store.outer)
    sDSlkt2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 9 * DS_ELEMS, lDSkt_store.inner, dtype=TF32), lDSkt_store.outer)
    sDS16_2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 11 * DS_ELEMS, lDS16.inner, dtype=F16), lDS16.outer)
    sDS16kt2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 12 * DS_ELEMS, lDS16kt_store.inner, dtype=F16), lDS16kt_store.outer)
    sQ16_2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 5 * DS_ELEMS, lQ16.inner, dtype=F16), lQ16.outer)
    sK16_2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK16.inner, dtype=F16), lK16.outer)

    sDOT0 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lDOT.inner), lDOT.outer)
    sDOT1 = cute.make_tensor(cute.recast_ptr(sDO.iterator + Q_T1_OFF, lDOT.inner), lDOT.outer)
    sKT = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT.inner, dtype=TF32), lKT.outer)

    kbase = mK.iterator.toint()
    vbase = mV.iterator.toint()
    q_stride_bytes = q_token_stride * I32(2)
    k_stride_bytes = k_token_stride * I32(2)
    v_stride_bytes = v_token_stride * I32(2)
    workspace_token_stride = I32(DIM * 2)
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 32)), cute.make_layout((1, 8)))
    thr_g2s = tiled_g2s.get_slice(lane)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init(mbar + 2, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C((BR, HQ)))
    tS = cute.make_tensor(tmem_ptr + TM_A, tS_f.layout)
    tDP = cute.make_tensor(tmem_ptr + TM_B, tS_f.layout)
    tKV_f = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C((BR, BR)))
    tKA = cute.make_tensor(tmem_ptr + TM_A, tKV_f.layout)
    tKB = cute.make_tensor(tmem_ptr + TM_B, tKV_f.layout)
    tKV16_f = dSQ16_mma.make_fragment_C(dSQ16_mma.partition_shape_C((BR, BR)))
    tKA16 = cute.make_tensor(tmem_ptr + TM_A, tKV16_f.layout)
    tDQ_f = dSK_mma.make_fragment_C(dSK_mma.partition_shape_C((BR, HQ)))
    tDQ0 = cute.make_tensor(tmem_ptr + TM_DQ0, tDQ_f.layout)
    tDQ1 = cute.make_tensor(tmem_ptr + TM_DQ1, tDQ_f.layout)
    tDQ16_f = dSK16_mma.make_fragment_C(dSK16_mma.partition_shape_C((BR, HQ)))
    tDQ0_16 = cute.make_tensor(tmem_ptr + TM_DQ0, tDQ16_f.layout)
    tDQ1_16 = cute.make_tensor(tmem_ptr + TM_DQ1, tDQ16_f.layout)

    fK = KQ_mma.make_fragment_A(sK)
    fQ = KQ_mma.make_fragment_B(sQ)
    fV = KQ_mma.make_fragment_A(sV)
    fDO = KQ_mma.make_fragment_B(sDO)
    fQT = dSQ_mma.make_fragment_A(sQscratch)
    fDOT0 = dPV_mma.make_fragment_A(sDOT0)
    fDOT1 = dPV_mma.make_fragment_A(sDOT1)
    fDSh = dSQ_mma.make_fragment_B(sDSh)
    fDSl = dSQ_mma.make_fragment_B(sDSl)
    fPb = dPV_mma.make_fragment_B(sP)
    fKT = dSK_mma.make_fragment_A(sKT)
    fDShkt = dSK_mma.make_fragment_B(sDShkt)
    fDSlkt = dSK_mma.make_fragment_B(sDSlkt)
    fQT16 = dSQ16_mma.make_fragment_A(sQ16)
    fDS16 = dSQ16_mma.make_fragment_B(sDS16)
    fKT16_0 = dSK16_mma.make_fragment_A(sK16_0)
    fKT16_1 = dSK16_mma.make_fragment_A(sK16_1)
    fDS16kt = dSK16_mma.make_fragment_B(sDS16kt)

    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), F32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), F32)

    tS2 = tS[(None, None), 0, 0]
    tDP2 = tDP[(None, None), 0, 0]
    tKA2 = tKA[(None, None), 0, 0]
    tKB2 = tKB[(None, None), 0, 0]
    tDQ0_2 = tDQ0[(None, None), 0, 0]
    tDQ1_2 = tDQ1[(None, None), 0, 0]

    cSN = cute.make_identity_tensor((BR, HQ))
    cp_s = tcgen05.make_tmem_copy(ld8, tS2)
    thr_s = cp_s.get_slice(t128)
    tr_S = thr_s.partition_S(tS2)
    tr_DP = thr_s.partition_S(tDP2)
    tr_cS = thr_s.partition_D(cSN)
    slot_m = cute.get(tr_cS[0], mode=[0])

    cKVN = cute.make_identity_tensor((BR, BR))
    cp_kv = tcgen05.make_tmem_copy(ld16, tKA2)
    thr_kv = cp_kv.get_slice(t128)
    tr_A = thr_kv.partition_S(tKA2)
    tr_B = thr_kv.partition_S(tKB2)
    tr_cKV = thr_kv.partition_D(cKVN)
    dim_m = cute.get(tr_cKV[0], mode=[0])

    cp_dq = tcgen05.make_tmem_copy(ld16, tDQ0_2)
    thr_dq = cp_dq.get_slice(t128)
    tr_DQ0 = thr_dq.partition_S(tDQ0_2)
    tr_DQ1 = thr_dq.partition_S(tDQ1_2)
    tr_cDQ = thr_dq.partition_D(cSN)
    dq_m = cute.get(tr_cDQ[0], mode=[0])

    kvsh = tr_cKV[None, None, 0].shape
    rA = cute.make_rmem_tensor(kvsh, F32)
    rB = cute.make_rmem_tensor(kvsh, F32)
    rPk = cute.make_rmem_tensor(cute.make_layout((2, 32), stride=(1, 2)), F32)
    pair = cute.make_rmem_tensor(cute.make_layout(2), F32)
    rC = cute.make_rmem_tensor(cute.make_layout(8), F32)
    rD = cute.make_rmem_tensor(cute.make_layout(8), F32)
    dbase = I32(0)

    cute.arch.barrier()

    ph0 = I32(0)
    ph1 = I32(0)
    ph2 = I32(0)
    base_lo = dim_m * 2
    base_hi = base_lo + 256
    nkQ = cute.size(fQ, mode=[2])
    nkDS = cute.size(fDSh, mode=[2])
    nkPV = cute.size(fPb, mode=[2])
    nkKT = cute.size(fKT, mode=[2])
    nkDS16 = cute.size(fDS16, mode=[2])
    nkKT16 = cute.size(fKT16_0, mode=[2])

    d0 = lane * 8

    # ---- per-query head: index stage, chunk-0 gather, Q stream, delta ----
    first_vlen = mVlen[bidx]
    first_index = I32(0)
    if tidx < first_vlen:
        first_index = mIdx[bidx * TOPK + tidx]
    sIdx[tidx] = first_index
    cute.arch.barrier()
    for i in cutlass.range_constexpr(BR // NWG // 4):
        r = i * (4 * NWG) + warp_idx
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, kbase + sIdx[r] * k_stride_bytes, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
            ),
            thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
        )
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, vbase + sIdx[r] * v_stride_bytes, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
            ),
            thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
        )
    qb0 = bidx * (HQ * DIM)
    qptr0 = mQ.iterator.toint() + bidx * q_stride_bytes
    for i in cutlass.range_constexpr(HQ // NWG // 4):
        h = i * (4 * NWG) + warp_idx
        cute.copy(
            tiled_g2s,
            thr_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, qptr0 + h * (DIM * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
            ),
            thr_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
        )
    cute.arch.cp_async_commit_group()
    for i in cutlass.range_constexpr(HQ // NWG // 4):
        h = i * (4 * NWG) + warp_idx
        goff0 = qb0 + h * DIM + d0
        rdo0 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
        cute.autovec_copy(cute.make_tensor(mDO.iterator + goff0, cute.make_layout((1, 8))), rdo0)
        cute.autovec_copy(rdo0, cute.local_tile(sDO2, (1, 8), (h, lane)))
    if tidx < HQ:
        sDel[tidx] = mDelta[bidx * HQ + tidx]
        sNorm[tidx] = mNorm[bidx * HQ + tidx]
        sMax[tidx] = mSoftmaxMax[bidx * HQ + tidx]
        sInv[tidx] = mSoftmaxInvSum[bidx * HQ + tidx]

    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if warp_idx == 0:
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + 0)

    nq = (TQ - bidx + nctas - I32(1)) // nctas
    for qi in cutlass.range(nq):
        t = bidx + qi * nctas
        qbase = t * (HQ * DIM)
        vlen = mVlen[t]
        nchunk = (vlen + (BR - 1)) // BR
        dbase = I32(0)
        fallback_chunks = I32(0)

        for c in cutlass.range(nchunk):
            cbase = c * BR
            cute.arch.mbarrier_wait(mbar + 0, ph0)

            # Inspect the original BF16 staging tiles before either is reused.
            # Every in-range normal BF16 value is exactly representable in
            # FP16; the vote prevents finite large-exponent values from
            # silently becoming infinities in the accurate fast path.
            unsafe = F32(0.0)
            for i in cutlass.range_constexpr(BR // NWG // 4):
                r = i * (4 * NWG) + warp_idx
                rk_guard = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                rk_guard16 = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                cute.autovec_copy(cute.local_tile(sK2, (1, 8), (r, lane)), rk_guard)
                for e in cutlass.range_constexpr(8):
                    if cbase + r < vlen:
                        rk_guard16[0, e] = rk_guard[0, e].to(F16)
                        x = rk_guard[0, e].to(F32)
                        if x > F32(F16_MAX):
                            unsafe = F32(1.0)
                        if x < F32(0.0 - F16_MAX):
                            unsafe = F32(1.0)
                    else:
                        # The tensor-core tile still consumes suffix rows.
                        # Store zero rather than allowing 0*dS times an
                        # out-of-range gathered sentinel to form a NaN.
                        rk_guard16[0, e] = F16(0.0)
                cute.autovec_copy(rk_guard16, cute.local_tile(sK16_2, (1, 8), (r, lane)))
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                rq_guard = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                cute.autovec_copy(cute.local_tile(sQ2, (1, 8), (h, lane)), rq_guard)
                for e in cutlass.range_constexpr(8):
                    x = rq_guard[0, e].to(F32)
                    if x > F32(F16_MAX):
                        unsafe = F32(1.0)
                    if x < F32(0.0 - F16_MAX):
                        unsafe = F32(1.0)

            # ---- softmax / FP16 dS plus range-safe TF32 dS ----
            for j in cutlass.range_constexpr(HQ // 8):
                if wg == SM_WG[j]:
                    rS = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    rDP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    cute.copy(cp_s, tr_S[None, None, j], rS)
                    cute.copy(cp_s, tr_DP[None, None, j], rDP)
                    oP = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    oS16 = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                    oSh = cute.make_rmem_tensor(cute.make_layout((1, 8)), TF32)
                    oSl = cute.make_rmem_tensor(cute.make_layout((1, 8)), TF32)
                    for e in cutlass.range_constexpr(0, 8, 2):
                        h0 = j * 8 + e
                        h1 = h0 + 1
                        p0 = F32(0.0)
                        p1 = F32(0.0)
                        if (cbase + slot_m) < vlen:
                            x0, x1 = cute.arch.add_packed_f32x2(
                                (rS[e], rS[e + 1]),
                                (F32(0.0) - sMax[h0], F32(0.0) - sMax[h1]),
                            )
                            x0, x1 = cute.arch.mul_packed_f32x2(
                                (x0, x1),
                                (F32(SCALE_LOG2E), F32(SCALE_LOG2E)),
                            )
                            p0 = cute.math.exp2(x0, fastmath=True) * sInv[h0] * sNorm[h0]
                            p1 = cute.math.exp2(x1, fastmath=True) * sInv[h1] * sNorm[h1]
                        ds0 = F32(0.0)
                        ds1 = F32(0.0)
                        if vlen > 1:
                            ds0 = p0 * (rDP[e] - sDel[h0]) * F32(SCALE)
                            ds1 = p1 * (rDP[e + 1] - sDel[h1]) * F32(SCALE)
                        oP[0, e] = p0.to(BF16)
                        oP[0, e + 1] = p1.to(BF16)
                        oS16[0, e] = ds0.to(F16)
                        oS16[0, e + 1] = ds1.to(F16)
                        if ds0 > F32(F16_MAX):
                            unsafe = F32(1.0)
                        if ds0 < F32(0.0 - F16_MAX):
                            unsafe = F32(1.0)
                        if ds1 > F32(F16_MAX):
                            unsafe = F32(1.0)
                        if ds1 < F32(0.0 - F16_MAX):
                            unsafe = F32(1.0)
                        # TFloat32 is an MMA operand type, not a rounded storage
                        # type: a plain ``to(TF32).to(F32)`` round-trip is a
                        # no-op.  Materialize the PTX TF32 conversion so the
                        # residual contains the low mantissa bits discarded by
                        # the high term.
                        ds0h = I32(cute.arch.cvt_f32_tf32(ds0)).bitcast(F32)
                        ds1h = I32(cute.arch.cvt_f32_tf32(ds1)).bitcast(F32)
                        oSh[0, e] = ds0h.to(TF32)
                        oSh[0, e + 1] = ds1h.to(TF32)
                        oSl[0, e] = (ds0 - ds0h).to(TF32)
                        oSl[0, e + 1] = (ds1 - ds1h).to(TF32)
                    cute.autovec_copy(oP, cute.local_tile(sP2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oS16, cute.local_tile(sDS16_2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oSh, cute.local_tile(sDSh2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oSl, cute.local_tile(sDSl2, (1, 8), (slot_m, j)))
                    for e in cutlass.range_constexpr(8):
                        sDS16kt2[j * 8 + e, slot_m] = oS16[0, e]
                        sDShkt2[j * 8 + e, slot_m] = oSh[0, e]
                        sDSlkt2[j * 8 + e, slot_m] = oSl[0, e]
                pass

            warp_bad = cute.arch.warp_reduction_sum(unsafe)
            if lane == 0:
                sBad[warp_idx] = I32(0)
                if warp_bad > F32(0.0):
                    sBad[warp_idx] = I32(1)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if tidx == 0:
                bad = I32(0)
                for w in cutlass.range_constexpr(16):
                    bad = bad + sBad[w]
                sBad[0] = bad
            cute.arch.barrier()
            use_f16 = sBad[0] == 0
            if _PATH_DIAGNOSTIC:
                if tidx == 0:
                    if sBad[0] != 0:
                        fallback_chunks = fallback_chunks + I32(1)

            # K/Q scratch is branch-exclusive.  FP16 can safely rewrite K in
            # place because both types are 16-bit; TF32 is reloaded from global
            # to avoid an in-place widening race in sK.
            qptr = mQ.iterator.toint() + t * q_stride_bytes
            if use_f16:
                for i in cutlass.range_constexpr(HQ // NWG // 4):
                    h = i * (4 * NWG) + warp_idx
                    rq_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rq_f = cute.make_rmem_tensor(cute.make_layout((1, 4)), F16)
                    cute.autovec_copy(cute.local_tile(sQ2, (1, 4), (h, lane)), rq_b)
                    for e in cutlass.range_constexpr(4):
                        rq_f[0, e] = rq_b[0, e].to(F16)
                    cute.autovec_copy(rq_f, cute.local_tile(sQ16_2, (1, 4), (h, lane)))
            else:
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    rk_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rk_t = cute.make_rmem_tensor(cute.make_layout((1, 4)), TF32)
                    cute.autovec_copy(
                        cute.make_tensor(
                            cute.make_ptr(BF16, kbase + sIdx[cbase + r] * k_stride_bytes + lane * 8, cute.AddressSpace.gmem, assumed_align=8),
                            cute.make_layout((1, 4)),
                        ),
                        rk_b,
                    )
                    for e in cutlass.range_constexpr(4):
                        rk_t[0, e] = rk_b[0, e].to(TF32)
                    cute.autovec_copy(rk_t, cute.local_tile(sKtf2, (4, 1), (lane, r)))
                for i in cutlass.range_constexpr(HQ // NWG // 4):
                    h = i * (4 * NWG) + warp_idx
                    rq_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rq_t = cute.make_rmem_tensor(cute.make_layout((1, 4)), TF32)
                    cute.autovec_copy(
                        cute.make_tensor(
                            cute.make_ptr(BF16, qptr + h * (DIM * 2) + lane * 8, cute.AddressSpace.gmem, assumed_align=8), cute.make_layout((1, 4))
                        ),
                        rq_b,
                    )
                    for e in cutlass.range_constexpr(4):
                        rq_t[0, e] = rq_b[0, e].to(TF32)
                    cute.autovec_copy(rq_t, cute.local_tile(sQscratch2, (4, 1), (lane, h)))
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # ---- dK^T / dV^T dims [0,128) + persistent dQ^T ----
            if warp_idx == 0:
                if use_f16:
                    dSQ16_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS16):
                        cute.gemm(dSQ16_mma, tKA16, fQT16[None, None, kb, 0], fDS16[None, None, kb, 0], tKA16)
                        dSQ16_mma.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKA, fQT[None, None, kb, 0], fDSh[None, None, kb, 0], tKA)
                        dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKA, fQT[None, None, kb, 0], fDSl[None, None, kb, 0], tKA)
                dPV_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkPV):
                    cute.gemm(dPV_mma, tKB, fDOT0[None, None, kb, 0], fPb[None, None, kb, 0], tKB)
                    dPV_mma.set(tcgen05.Field.ACCUMULATE, True)
                if use_f16:
                    dSK16_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                    for kb in cutlass.range_constexpr(nkKT16):
                        cute.gemm(dSK16_mma, tDQ0_16, fKT16_0[None, None, kb, 0], fDS16kt[None, None, kb, 0], tDQ0_16)
                        dSK16_mma.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    dSK_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_mma, tDQ0, fKT[None, None, kb, 0], fDShkt[None, None, kb, 0], tDQ0)
                        dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_mma, tDQ0, fKT[None, None, kb, 0], fDSlkt[None, None, kb, 0], tDQ0)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 1)

            if c > 0:
                for j in cutlass.range_constexpr(NWG):
                    if wg == j:
                        for e in cutlass.range_constexpr(8):
                            pair[0] = rC[e]
                            pair[1] = rD[e]
                            cute.arch.atomic_add(
                                mW.iterator + (sIdx[dbase + j * 32 + e] * workspace_token_stride + base_hi),
                                pair.load(),
                                scope="gpu",
                            )
                    pass
            cute.arch.mbarrier_wait(mbar + 1, ph1)

            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    cute.copy(cp_kv, tr_A[None, None, 2 * j], rA)
                    cute.copy(cp_kv, tr_B[None, None, 2 * j], rB)
                    for e in cutlass.range_constexpr(16):
                        rPk[0, e] = rA[e]
                        rPk[1, e] = rB[e]
                    cute.copy(cp_kv, tr_A[None, None, 2 * j + 1], rA)
                    cute.copy(cp_kv, tr_B[None, None, 2 * j + 1], rB)
                    for e in cutlass.range_constexpr(16):
                        rPk[0, 16 + e] = rA[e]
                        rPk[1, 16 + e] = rB[e]
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()

            # Overwrite the selected branch's scratch with feature dims
            # [128,256) only after stage-0 UMMA has completed and its results
            # are in registers.
            if use_f16:
                for i in cutlass.range_constexpr(HQ // NWG // 4):
                    h = i * (4 * NWG) + warp_idx
                    rq_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rq_f = cute.make_rmem_tensor(cute.make_layout((1, 4)), F16)
                    cute.autovec_copy(cute.local_tile(sQ2, (1, 4), (h, lane + 32)), rq_b)
                    for e in cutlass.range_constexpr(4):
                        rq_f[0, e] = rq_b[0, e].to(F16)
                    cute.autovec_copy(rq_f, cute.local_tile(sQ16_2, (1, 4), (h, lane)))
            else:
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    rk_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rk_t = cute.make_rmem_tensor(cute.make_layout((1, 4)), TF32)
                    cute.autovec_copy(
                        cute.make_tensor(
                            cute.make_ptr(BF16, kbase + sIdx[cbase + r] * k_stride_bytes + 256 + lane * 8, cute.AddressSpace.gmem, assumed_align=8),
                            cute.make_layout((1, 4)),
                        ),
                        rk_b,
                    )
                    for e in cutlass.range_constexpr(4):
                        rk_t[0, e] = rk_b[0, e].to(TF32)
                    cute.autovec_copy(rk_t, cute.local_tile(sKtf2, (4, 1), (lane, r)))
                for i in cutlass.range_constexpr(HQ // NWG // 4):
                    h = i * (4 * NWG) + warp_idx
                    rq_b = cute.make_rmem_tensor(cute.make_layout((1, 4)), BF16)
                    rq_t = cute.make_rmem_tensor(cute.make_layout((1, 4)), TF32)
                    cute.autovec_copy(
                        cute.make_tensor(
                            cute.make_ptr(BF16, qptr + h * (DIM * 2) + 256 + lane * 8, cute.AddressSpace.gmem, assumed_align=8), cute.make_layout((1, 4))
                        ),
                        rq_b,
                    )
                    for e in cutlass.range_constexpr(4):
                        rq_t[0, e] = rq_b[0, e].to(TF32)
                    cute.autovec_copy(rq_t, cute.local_tile(sQscratch2, (4, 1), (lane, h)))
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # ---- dK^T / dV^T for feature dims [128,256) ----
            if warp_idx == 0:
                if use_f16:
                    dSQ16_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS16):
                        cute.gemm(dSQ16_mma, tKA16, fQT16[None, None, kb, 0], fDS16[None, None, kb, 0], tKA16)
                        dSQ16_mma.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKA, fQT[None, None, kb, 0], fDSh[None, None, kb, 0], tKA)
                        dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKA, fQT[None, None, kb, 0], fDSl[None, None, kb, 0], tKA)
                dPV_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkPV):
                    cute.gemm(dPV_mma, tKB, fDOT1[None, None, kb, 0], fPb[None, None, kb, 0], tKB)
                    dPV_mma.set(tcgen05.Field.ACCUMULATE, True)
                if use_f16:
                    dSK16_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                    for kb in cutlass.range_constexpr(nkKT16):
                        cute.gemm(dSK16_mma, tDQ1_16, fKT16_1[None, None, kb, 0], fDS16kt[None, None, kb, 0], tDQ1_16)
                        dSK16_mma.set(tcgen05.Field.ACCUMULATE, True)
                else:
                    dSK_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_mma, tDQ1, fKT[None, None, kb, 0], fDShkt[None, None, kb, 0], tDQ1)
                        dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_mma, tDQ1, fKT[None, None, kb, 0], fDSlkt[None, None, kb, 0], tDQ1)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 2)

            # ---- interleaved (dK,dV) scatter, dims [0,128) (overlaps the MMAs) ----
            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    for e in cutlass.range_constexpr(32):
                        pair[0] = rPk[0, e]
                        pair[1] = rPk[1, e]
                        cute.arch.atomic_add(
                            mW.iterator + (sIdx[cbase + j * 32 + e] * workspace_token_stride + base_lo),
                            pair.load(),
                            scope="gpu",
                        )
                pass

            cute.arch.mbarrier_wait(mbar + 2, ph2)

            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    cute.copy(cp_kv, tr_A[None, None, 2 * j], rA)
                    cute.copy(cp_kv, tr_B[None, None, 2 * j], rB)
                    for e in cutlass.range_constexpr(16):
                        rPk[0, e] = rA[e]
                        rPk[1, e] = rB[e]
                    cute.copy(cp_kv, tr_A[None, None, 2 * j + 1], rA)
                    cute.copy(cp_kv, tr_B[None, None, 2 * j + 1], rB)
                    for e in cutlass.range_constexpr(16):
                        rPk[0, 16 + e] = rA[e]
                        rPk[1, 16 + e] = rB[e]
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()

            # ---- prefetch chunk c+1 ----
            if c + 1 < nchunk:
                nbase = cbase + BR
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(BF16, kbase + sIdx[nbase + r] * k_stride_bytes, cute.AddressSpace.gmem, assumed_align=16),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
                    )
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(BF16, vbase + sIdx[nbase + r] * v_stride_bytes, cute.AddressSpace.gmem, assumed_align=16),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                    )
                cute.arch.cp_async_commit_group()

            # ---- interleaved (dK,dV) scatter, dims [128,256) ----
            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    for e in cutlass.range_constexpr(8):
                        pair[0] = rPk[0, e]
                        pair[1] = rPk[1, e]
                        cute.arch.atomic_add(
                            mW.iterator + (sIdx[cbase + j * 32 + e] * workspace_token_stride + base_hi),
                            pair.load(),
                            scope="gpu",
                        )
                pass

            if c + 1 < nchunk:
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 0)

            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    for e in cutlass.range_constexpr(16):
                        pair[0] = rPk[0, e + 8]
                        pair[1] = rPk[1, e + 8]
                        cute.arch.atomic_add(
                            mW.iterator + (sIdx[cbase + j * 32 + e + 8] * workspace_token_stride + base_hi),
                            pair.load(),
                            scope="gpu",
                        )
                pass
            for e in cutlass.range_constexpr(8):
                rC[e] = rPk[0, e + 24]
                rD[e] = rPk[1, e + 24]
            dbase = cbase + I32(24)
            ph0 = ph0 ^ I32(1)
            ph1 = ph1 ^ I32(1)
            ph2 = ph2 ^ I32(1)

        if nchunk == 0:
            cute.arch.mbarrier_wait(mbar + 0, ph0)
            ph0 = ph0 ^ I32(1)

        if nchunk > 0:
            for j in cutlass.range_constexpr(NWG):
                if wg == j:
                    for e in cutlass.range_constexpr(8):
                        pair[0] = rC[e]
                        pair[1] = rD[e]
                        cute.arch.atomic_add(
                            mW.iterator + (sIdx[dbase + j * 32 + e] * workspace_token_stride + base_hi),
                            pair.load(),
                            scope="gpu",
                        )
                pass

        tn = t + nctas
        if tn < TQ:
            cute.arch.barrier()
            next_vlen = mVlen[tn]
            next_index = I32(0)
            if tidx < next_vlen:
                next_index = mIdx[tn * TOPK + tidx]
            sIdx[tidx] = next_index
            cute.arch.barrier()
            for i in cutlass.range_constexpr(BR // NWG // 4):
                r = i * (4 * NWG) + warp_idx
                cute.copy(
                    tiled_g2s,
                    thr_g2s.partition_S(
                        cute.make_tensor(
                            cute.make_ptr(BF16, kbase + sIdx[r] * k_stride_bytes, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM))
                        )
                    ),
                    thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
                )
                cute.copy(
                    tiled_g2s,
                    thr_g2s.partition_S(
                        cute.make_tensor(
                            cute.make_ptr(BF16, vbase + sIdx[r] * v_stride_bytes, cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM))
                        )
                    ),
                    thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                )
            qb1 = tn * (HQ * DIM)
            qptr1 = mQ.iterator.toint() + tn * q_stride_bytes
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                cute.copy(
                    tiled_g2s,
                    thr_g2s.partition_S(
                        cute.make_tensor(cute.make_ptr(BF16, qptr1 + h * (DIM * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
                    ),
                    thr_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
                )
            cute.arch.cp_async_commit_group()
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                goff1 = qb1 + h * DIM + d0
                rdo1 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                cute.autovec_copy(cute.make_tensor(mDO.iterator + goff1, cute.make_layout((1, 8))), rdo1)
                cute.autovec_copy(rdo1, cute.local_tile(sDO2, (1, 8), (h, lane)))
            if tidx < HQ:
                sDel[tidx] = mDelta[tn * HQ + tidx]
                sNorm[tidx] = mNorm[tn * HQ + tidx]
                sMax[tidx] = mSoftmaxMax[tn * HQ + tidx]
                sInv[tidx] = mSoftmaxInvSum[tn * HQ + tidx]

        # ---------------- dQ epilogue ----------------
        if vlen > 0:
            if wg == 0:
                r0 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                cute.copy(cp_dq, tr_DQ0[None, None, 0], r0)
                for e in cutlass.range_constexpr(16):
                    mDQ[qbase + e * DIM + dq_m] = r0[e]
            if wg == 1:
                r1 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                cute.copy(cp_dq, tr_DQ1[None, None, 0], r1)
                for e in cutlass.range_constexpr(16):
                    mDQ[qbase + e * DIM + dq_m + 128] = r1[e]
        else:
            if wg == 0:
                for e in cutlass.range_constexpr(16):
                    mDQ[qbase + e * DIM + dq_m] = F32(0.0)
            if wg == 1:
                for e in cutlass.range_constexpr(16):
                    mDQ[qbase + e * DIM + dq_m + 128] = F32(0.0)
        if _PATH_DIAGNOSTIC:
            if tidx == 0:
                mNorm[t * HQ] = fallback_chunks.to(F32)
        cute.arch.fence_view_async_tmem_load()
        if tn < TQ:
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 0)

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------
@cute.jit
def _launch(
    mQ,
    mK,
    mV,
    mDO,
    mSoftmaxMax,
    mSoftmaxInvSum,
    mIdx,
    mVlen,
    mDelta,
    mNorm,
    mDQ,
    mW,
    mDK,
    mDV,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    TK: I32,
    nctas: I32,
    stream: cuda.CUstream,
):
    cg = tcgen05.CtaGroup.ONE
    KQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, cg, (BR, HQ))
    dSQ_mma = sm100_utils.make_trivial_tiled_mma(TF32, TF32, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dPV_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dSK_mma = sm100_utils.make_trivial_tiled_mma(TF32, TF32, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, HQ))
    dSQ16_mma = sm100_utils.make_trivial_tiled_mma(F16, F16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dSK16_mma = sm100_utils.make_trivial_tiled_mma(F16, F16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, HQ))

    lQs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, HQ, DIM), BF16, 1)
    lKs = sm100_utils.make_smem_layout_a(KQ_mma, (BR, HQ, DIM), BF16, 1)
    lDSs = sm100_utils.make_smem_layout_b(dSQ_mma, (BR, BR, HQ), TF32, 1)
    lPs = sm100_utils.make_smem_layout_b(dPV_mma, (BR, BR, HQ), BF16, 1)
    lQT = sm100_utils.make_smem_layout_a(dSQ_mma, (BR, BR, HQ), TF32, 1)
    lDOT = sm100_utils.make_smem_layout_a(dPV_mma, (BR, BR, HQ), BF16, 1)
    lKT = sm100_utils.make_smem_layout_a(dSK_mma, (BR, HQ, BR), TF32, 1)
    lDSkt = sm100_utils.make_smem_layout_b(dSK_mma, (BR, HQ, BR), TF32, 1)
    lDS16s = sm100_utils.make_smem_layout_b(dSQ16_mma, (BR, BR, HQ), F16, 1)
    lQT16 = sm100_utils.make_smem_layout_a(dSQ16_mma, (BR, BR, HQ), F16, 1)
    lKT16 = sm100_utils.make_smem_layout_a(dSK16_mma, (BR, HQ, BR), F16, 1)
    lDS16kt = sm100_utils.make_smem_layout_b(dSK16_mma, (BR, HQ, BR), F16, 1)

    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    a128_mn_tf32 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW128_32B, TF32)
    a128_k_tf32 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, TF32)
    a32_p = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW32, BF16)
    a64_tf32 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, TF32)
    a128_f16 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, F16)
    a32_f16 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW32, F16)
    lQ = cute.tile_to_shape(a128, (HQ, DIM), order=(0, 1))
    lK = cute.tile_to_shape(a128, (BR, DIM), order=(0, 1))
    lKtf = cute.tile_to_shape(a128_mn_tf32, (BR, BR), order=(1, 0))
    lQscratch = cute.tile_to_shape(a128_mn_tf32, (BR, HQ), order=(1, 0))
    lDS = cute.tile_to_shape(a64_tf32, (BR, HQ), order=(0, 1))
    lDSkt_store = cute.tile_to_shape(a128_k_tf32, (HQ, BR), order=(0, 1))
    lK16 = cute.tile_to_shape(a128_f16, (BR, DIM), order=(0, 1))
    lQ16 = cute.tile_to_shape(a128_f16, (HQ, BR), order=(0, 1))
    lDS16 = cute.tile_to_shape(a32_f16, (BR, HQ), order=(0, 1))
    lDS16kt_store = cute.tile_to_shape(a128_f16, (HQ, BR), order=(0, 1))
    lP = cute.tile_to_shape(a32_p, (BR, HQ), order=(0, 1))

    nelem = TK * DIM
    _delta_kernel(
        mQ,
        mK,
        mV,
        mDO,
        mSoftmaxMax,
        mSoftmaxInvSum,
        mIdx,
        mVlen,
        mDelta,
        mNorm,
        KQ_mma,
        lQ,
        lK,
        lQs,
        lKs,
        q_token_stride,
        k_token_stride,
        v_token_stride,
        TQ,
        nctas,
    ).launch(grid=[nctas, 1, 1], block=[NTHREADS, 1, 1], stream=stream)

    _zero_kernel(mW, nelem * 2).launch(
        grid=[cute.ceil_div(nelem * 2, 2048), 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )

    _bwd_kernel(
        mQ,
        mK,
        mV,
        mDO,
        mSoftmaxMax,
        mSoftmaxInvSum,
        mIdx,
        mVlen,
        mDelta,
        mNorm,
        mDQ,
        mW,
        KQ_mma,
        dSQ_mma,
        dPV_mma,
        dSK_mma,
        dSQ16_mma,
        dSK16_mma,
        lQ,
        lK,
        lKtf,
        lQscratch,
        lDS,
        lDSkt_store,
        lK16,
        lQ16,
        lDS16,
        lDS16kt_store,
        lP,
        lQT,
        lDOT,
        lKT,
        lDSkt,
        lQT16,
        lKT16,
        lDS16kt,
        lQs,
        lKs,
        lDSs,
        lDS16s,
        lPs,
        q_token_stride,
        k_token_stride,
        v_token_stride,
        TQ,
        nctas,
    ).launch(grid=[nctas, 1, 1], block=[NTHREADS, 1, 1], stream=stream)

    _cvt_kernel(mW, mDK, mDV, nelem).launch(
        grid=[cute.ceil_div(nelem, 1024), 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )


_compiled = {}
_delta_ws = {}  # fp32 delta scratch (fully rewritten at every launch)
_norm_ws = {}  # fp32 probability renormalization (fully rewritten per launch)
_smc = {}


def _sm_count(dev):
    n = _smc.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _smc[dev.index] = n
    return n


def _cv(x, al=16):
    return from_dlpack(x.detach(), assumed_align=al).mark_layout_dynamic()


def _validate_16byte_token_rows(q, k, v):
    """Validate the precondition used by every 128-bit Q/K/V row copy."""
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
            raise ValueError(f"{name} base address and token stride must be 16-byte aligned")


@torch.no_grad()
@serialize_workspace
def run(
    q,
    k,
    v,
    dout,
    softmax_max,
    softmax_inv_sum,
    selected_indices,
    valid_lengths,
    dq,
    dk,
    dv,
):
    tq = q.shape[0]
    tk = k.shape[0]
    dev = q.device
    stream_id = torch.cuda.current_stream(dev).cuda_stream

    _validate_16byte_token_rows(q, k, v)
    max_view = softmax_max.view(tq, HQ)
    inv_view = softmax_inv_sum.view(tq, HQ)

    nctas = min(tq, _sm_count(dev))

    # Scratch buffers only: the dK/dV accumulator is zeroed and the delta buffer
    # is fully rewritten at every launch, so no computed result survives a call.
    # The Tk-sized accumulator lives only for this call; the small per-query
    # buffers stay cached (the path diagnostic reads the renormalization one).
    w = torch.empty(tk * DIM * 2, dtype=torch.float32, device=dev)
    delta_key = (dev.index, stream_id)
    query_nelems = tq * HQ
    delta = _delta_ws.get(delta_key)
    if delta is None or delta.numel() < query_nelems:
        delta = torch.empty(query_nelems, dtype=torch.float32, device=dev)
        _delta_ws[delta_key] = delta
    norm = _norm_ws.get(delta_key)
    if norm is None or norm.numel() < query_nelems:
        norm = torch.empty(query_nelems, dtype=torch.float32, device=dev)
        _norm_ws[delta_key] = norm

    args = (
        _cv(q),
        _cv(k),
        _cv(v),
        _cv(dout.view(-1)),
        _cv(max_view.reshape(-1)),
        _cv(inv_view.reshape(-1)),
        _cv(selected_indices.view(-1), 4),
        _cv(valid_lengths.view(-1), 4),
        _cv(delta),
        _cv(norm),
        _cv(dq.view(-1)),
        _cv(w),
        _cv(dk.view(-1)),
        _cv(dv.view(-1)),
        I32(q.stride(0)),
        I32(k.stride(0)),
        I32(v.stride(0)),
    )

    # All tensor layouts are dynamic and TQ/TK/token pitches are explicit
    # runtime arguments.  The diagnostic switch is compile-time module state;
    # no exact packed extent or pitch belongs in the executable cache key.
    compile_key = (dev.index, _PATH_DIAGNOSTIC)
    stream = cuda.CUstream(stream_id)
    if compile_key not in _compiled:
        _compiled[compile_key] = cute.compile(_launch, *args, I32(tq), I32(tk), I32(nctas), make_fake_stream())
    _compiled[compile_key](*args, I32(tq), I32(tk), I32(nctas), stream)


def _debug_fallback_counts(q):
    """Return per-query TF32-fallback chunk counts for the current stream.

    Import this module with ``SSA_HQ16_PATH_DIAGNOSTIC=1`` and synchronize
    the current CUDA stream after ``run`` before reading the returned view.
    """
    if not _PATH_DIAGNOSTIC:
        raise RuntimeError("set SSA_HQ16_PATH_DIAGNOSTIC=1 before importing this module")
    tq = q.shape[0]
    dev = q.device
    stream_id = torch.cuda.current_stream(dev).cuda_stream
    norm = _norm_ws.get((dev.index, stream_id))
    if norm is None:
        raise RuntimeError("no Hq16 sparse-backward launch recorded for this tensor")
    return norm[: tq * HQ : HQ].detach()
