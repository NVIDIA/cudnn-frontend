# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSA sparse-attention backward for Hq=32, Hkv=1, D=256, TopK=512 (CuTe DSL, SM100/103).

Schedule
--------
One persistent CTA owns one query token and *all* 32 query heads, so the MQA
dK/dV scatter is reduced across heads in registers before it reaches global
memory.  The CTA walks the token's selected support in chunks of BR=128 slots
and issues five tcgen05 (UMMA) contractions per chunk.

Why this shape at Hq=32
-----------------------
The operator is *not* flop bound: every (query, support) pair deposits
2 x 256 fp32 values, and Blackwell's L2 caps global atomics well below the
tensor cores.  dK and dV therefore share a single interleaved fp32 workspace

    W[key, dim, 0] = dK[key, dim]      W[key, dim, 1] = dV[key, dim]

and the dK / dV tiles for the *same* 128 feature dims are produced together so
one thread owns both fp32 values and issues one 8-byte vector atomic instead
of two scalar ones.  Accumulation stays fp32 all the way to the public
gradient; narrowing before a contended atomic changes the backward and
produces data-dependent outliers.

Numerics
--------
Q, K, dO and V are BF16 leaves, so every contraction that consumes them
directly (S = K Q^T, dP = V dO^T, dV = P^T dO) is *exact* under a BF16 x BF16
-> FP32 UMMA: two 8-bit significands multiply into 16 bits and only the FP32
accumulation rounds.  The one genuinely FP32 operand is the score gradient dS,
which is therefore split into an exact two-term BF16 pair

    dS_hi = rn_bf16(dS),   dS_lo = rn_bf16(dS - dS_hi)

and fed through *two* UMMA passes per dQ / dK contraction.  ``dS - dS_hi`` is
exact in FP32, so the pair carries ~17 significand bits - 32x tighter than a
single FP16 dS - and every operand keeps BF16's FP32-wide exponent range.  A
BF16 +/-2^17 activation paired with exact 2^-17 factors therefore stays
representable, where an FP16 fast path either overflows it or flushes its
partner to zero, and no prescale/guard is needed anywhere.  The extra UMMAs
are nearly free because this operator is atomic-throughput bound, not flop
bound.  dV deliberately keeps the forward BF16 probability convention.  Delta
comes from an exact FP32 prepass so dS is formed from a full-support reduction.

Hq=32 layout notes (these differ from an Hq=80 schedule)
--------------------------------------------------------
The dS / P staging tile is (BR, Hq) with Hq contiguous, so its UMMA swizzle
atom is selected from the *Hq* extent: 32 x 2B = 64B -> K_SW64.  (At Hq=80 the
extent is 160B and CUTLASS picks K_SW32.)  The hand-written store views must
use the same atom or every gradient silently decodes from the wrong bank
rotation.  Head groups are 8 wide, giving exactly one group per warpgroup.
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
import cutlass.experimental.primitives as prims
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack, make_fake_stream

BF16 = cutlass.BFloat16
F16 = cutlass.Float16
F32 = cutlass.Float32
I32 = cutlass.Int32

HQ = 32  # query heads
DIM = 256  # head dim (qk == v)
TOPK = 512
BR = 128  # support slots per chunk
NTHREADS = 512
NWG = NTHREADS // 128

SCALE = 0.0625
LOG2E = float(math.log2(math.e))
SCALE_LOG2E = SCALE * LOG2E

# TMEM column layout (512 columns), exactly saturated.
#
#   delta sweep :  [0,128) dV^T tile0 | [128,256) dV^T tile1
#   backward    :  [0,128) dK^T tile  | [128,160) dQ^T0 | [160,192) dQ^T1
#   both        :  [256,384) S^T -> P in place | [384,512) dP^T cache
#
# The score tile is written by the KQ UMMA straight into the probability
# cache slot and is overwritten in place by P, so no separate S^T scratch is
# needed.  Freeing those 32 columns is what lets both 128-dim dV^T tiles stay
# resident simultaneously: the delta sweep issues them under one commit again
# instead of serialising them through a single temporary tile.
TM_A = 0  # dV^T tile0 (delta) / dK^T dims [0,128) (backward)
# TM_B carries dV^T tile1 in the delta sweep and, in the backward sweep, the
# dK^T tile for dims [128,256).  Giving dK-hi its own accumulator instead of
# recycling TM_A deletes the CTA barrier that used to sit between the dK-lo
# drain and the dK-hi UMMA: the two tiles are now disjoint, so the drain of one
# can no longer race the UMMA of the other.  The dQ^T accumulators move out to
# the P/dP cache region to make room (see TM_DQ0 below).
TM_B = 128
TM_PC = 256  # chunk c score tile, overwritten in place by exact FP32 P
# dP^T is produced once per (query, chunk) by the fused delta sweep and is
# needed a second time by the backward sweep.  At Hq=32 a (BR,Hq) fp32
# accumulator is only 32 TMEM columns, so all TOPK/BR=4 chunks of dP^T fit in
# 128 otherwise-idle columns.  Keeping them resident removes the whole second
# V gather (512 B per (query, slot), a quarter of the kernel's L2 traffic) and
# one UMMA per backward chunk.
NCHUNK = TOPK // BR
# P/dP for chunk c are paired and laid out in REVERSE consumption order, so the
# chunk the backward sweep consumes first sits at the bottom of the cache.  Its
# 64 columns are dead the moment that chunk's dS window retires -- before any
# dQ^T UMMA has run -- which is exactly where the two 32-column dQ^T
# accumulators now live.  Nothing else moves and TMEM stays exactly saturated.
TM_DQ0 = TM_PC  # dQ^T tile0 : 128 x Hq fp32 (backward-sweep persistent)
TM_DQ1 = TM_PC + 32  # dQ^T tile1


# --------------------------------------------------------------------------
# gradient deposit primitive
# --------------------------------------------------------------------------
# ``atom.global.add.f32`` reserves a return slot for the pre-image even though
# the dK/dV scatter never reads it, so every deposit carries a dead L2 -> SM
# response.  ``red`` is the fire-and-forget form of the same L2 reduction.  The
# deposit is also given an L2 evict-last policy: the dK/dV workspace is the one
# resident structure in this kernel (8-32 MiB) while Q/dO/dQ stream ~250 MiB-1.5
# GiB past it per launch, so keeping the reduction destinations pinned is what
# makes the RMW stay an L2-internal operation.
_L2_EVICT_LAST = 0x14F0000000000000


def _red_add_f32(ptr, val):
    prims.inline_ptx(
        "red.global.add.L2::cache_hint.f32 [ {$r0} ], {$r1}, {$r2};\n",
        read_only_args=[ptr, val, cutlass.Int64(_L2_EVICT_LAST)],
    )


def _idx4(sIdx4, i4):
    """One 128-bit shared load feeding four REDs.

    Every drain burst reads four consecutive support indices from ``sIdx`` and
    all 128 threads of a warpgroup read the *same* index (only the feature dim
    varies per lane), so a scalar ``LDS.32`` per RED is pure issue overhead on
    the deposit critical path.  ``sIdx`` is 16-byte aligned and every burst base
    is a multiple of four slots, so the (TOPK/4, 4) view lets CuTe emit a single
    ``LDS.128`` per four deposits.
    """
    r = cute.make_rmem_tensor(cute.make_layout(4), I32)
    cute.autovec_copy(sIdx4[i4, None], r)
    return r


def _st_stream_f32(ptr, val):
    """dQ is write-once and never re-read by this kernel: 128 MiB (Tq=4096) to
    512 MiB (Tq=16384) of fp32 that currently allocates and dirties L2 lines it
    will never serve a hit from.  ``st.global.cs`` marks those lines
    evict-first, which is the exact complement of the evict-last policy the
    dK/dV RED scatter carries -- the reduction workspace is the one structure
    that has to stay L2-resident, and dQ is the largest thing racing it out.
    (The padding-row branch below keeps the plain store: inline PTX cannot take
    an immediate zero as a register operand.)"""
    prims.inline_ptx(
        "st.global.cs.f32 [ {$r0} ], {$r1};\n",
        read_only_args=[ptr, val],
    )


Q_T1_OFF = HQ * 64 * 2  # second dim tile of a (80,256) K-major tile
K_T1_OFF = BR * 64 * 2  # second dim tile of a (128,256) K-major tile
DS_ELEMS = BR * HQ  # one (128,32) K-major tile
Q_SCRATCH_ELEMS = HQ * DIM
assert 3 * DS_ELEMS + Q_SCRATCH_ELEMS <= BR * DIM
# Exactly four eight-head softmax groups, one per warpgroup.
SM_WG = (0, 1, 2, 3)


def _dp_tile(tmem_ptr, layout, c):
    """dP^T accumulator for physical chunk ``c`` (0 <= c < TOPK//BR).

    The fused delta sweep produces dP^T once per (query, chunk); the backward
    sweep re-reads it from TMEM instead of re-gathering V and re-issuing the
    V.dO^T UMMA.
    """
    return cute.make_tensor(tmem_ptr + (I32(TM_PC + 32) + (I32(NCHUNK - 1) - c) * I32(64)), layout)


def _p_tile(tmem_ptr, layout, c):
    """Exact FP32 probability tile cached by the delta sweep (reverse order)."""
    return cute.make_tensor(tmem_ptr + (I32(TM_PC) + (I32(NCHUNK - 1) - c) * I32(64)), layout)


# --------------------------------------------------------------------------
# helper kernels
# --------------------------------------------------------------------------
@cute.kernel
def _zero_kernel(mDK: cute.Tensor, mDV: cute.Tensor, n: I32):
    """Initialise the two fp32 atomic destinations (n = Tk * DIM each)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 8
    if base + 8 <= n:
        fa = cute.make_rmem_tensor(cute.make_layout(8), F32)
        fa.fill(0.0)
        cute.autovec_copy(fa, cute.make_tensor(mDK.iterator + base, cute.make_layout(8)))
        cute.autovec_copy(fa, cute.make_tensor(mDV.iterator + base, cute.make_layout(8)))


# --------------------------------------------------------------------------
# exact softmax delta prepass
# --------------------------------------------------------------------------
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
    mDQ: cute.Tensor,
    mDK: cute.Tensor,
    mDV: cute.Tensor,
    KQ_mma: cute.TiledMma,
    dSQ_mma: cute.TiledMma,
    dPV_mma: cute.TiledMma,
    dSK_mma: cute.TiledMma,
    dSQ_bf16_mma: cute.TiledMma,
    dSK_bf16_mma: cute.TiledMma,
    lQ: cute.ComposedLayout,
    lK: cute.ComposedLayout,
    lQscratch: cute.ComposedLayout,
    lDS: cute.ComposedLayout,
    lP: cute.ComposedLayout,
    lQT: cute.ComposedLayout,
    lDOT: cute.ComposedLayout,
    lKT: cute.ComposedLayout,
    lDSmn: cute.ComposedLayout,
    lDS_bf16: cute.ComposedLayout,
    lQT_bf16: cute.ComposedLayout,
    lKT_bf16: cute.ComposedLayout,
    lDSmn_bf16: cute.ComposedLayout,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lDSs: cute.ComposedLayout,
    lDSs_bf16: cute.ComposedLayout,
    lPs: cute.ComposedLayout,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    nctas: I32,
    qperm: I32,
    diagnostic_single_bf16_ds: cutlass.Constexpr,
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
    # BF16 P staging for the *delta* sweep's dV^T contraction.  It cannot alias
    # the V buffer the way the backward sweep's dS/P tiles do, because V is
    # still live (it feeds dP^T) throughout the delta phase.  8 KiB.
    sPd = smem.allocate_tensor(BF16, lPs.outer, byte_alignment=1024, swizzle=lPs.inner)
    sIdx = smem.allocate_tensor(I32, cute.make_layout(TOPK), byte_alignment=16)
    # Vector view of the same storage: four consecutive support slots per row.
    # NOTE: CuTe's default layout is column-major -- the row-of-four view
    # needs an explicit (4, 1) stride or slot s lands at sIdx[s // 4].
    sIdx4 = cute.make_tensor(sIdx.iterator, cute.make_layout((TOPK // 4, 4), stride=(4, 1)))
    sMax = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sInv = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    sDel = smem.allocate_tensor(F32, cute.make_layout(HQ), byte_alignment=16)
    # Fused delta reduction channels: sum_s P*dP and sum_s P, first per warp
    # (sRed) then per physical TopK chunk (sPart).
    # Chunk-parity double buffered: with the end-of-chunk CTA barrier gone a
    # warpgroup can reach chunk c+1's probability window while another thread is
    # still folding chunk c's warp partials into sPart.  Two buffers make that
    # legal; the intervening chunk's publication barrier separates c from c+2.
    sRed = smem.allocate_tensor(
        F32,
        cute.make_layout((2, 2, HQ, 4), stride=(2 * HQ * 4, HQ * 4, 4, 1)),
        byte_alignment=16,
    )
    sPart = smem.allocate_tensor(
        F32,
        cute.make_layout((2, TOPK // BR, HQ), stride=(HQ * (TOPK // BR), HQ, 1)),
        byte_alignment=16,
    )
    # [Q not exactly FP16-representable, K not exactly representable,
    #  at least one dS requiring the BF16 high/residual representation].
    sPath = smem.allocate_tensor(I32, cute.make_layout(3), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 5, byte_alignment=16)

    # The production path stores both representations before a CTA-uniform
    # choice: one FP16 dS operand for the high-precision common case, and the
    # proven BF16 high/residual pair for any exponent or conversion hazard.
    sDS = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSs.inner, dtype=F16), lDSs.outer)
    sDSB = cute.make_tensor(cute.recast_ptr(sV.iterator + DS_ELEMS, lDSs_bf16.inner, dtype=BF16), lDSs_bf16.outer)
    sDSBL = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * DS_ELEMS, lDSs_bf16.inner, dtype=BF16), lDSs_bf16.outer)
    sQ2 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQ.inner), lQ.outer)
    sDO2 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lQ.inner), lQ.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner), lK.outer)
    sV2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lK.inner), lK.outer)
    sQF2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 3 * DS_ELEMS, lQscratch.inner, dtype=F16), lQscratch.outer)
    sKF2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner, dtype=F16), lK.outer)
    sDS2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lDS.inner, dtype=F16), lDS.outer)
    sDSB2 = cute.make_tensor(cute.recast_ptr(sV.iterator + DS_ELEMS, lDS_bf16.inner, dtype=BF16), lDS_bf16.outer)
    sDSBL2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * DS_ELEMS, lDS_bf16.inner, dtype=BF16), lDS_bf16.outer)
    sPd2 = cute.make_tensor(cute.recast_ptr(sPd.iterator, lP.inner, dtype=BF16), lP.outer)

    # dSQ and dPV share operand-A geometry.  Reinterpret resident Q exactly as
    # dO is reinterpreted here, avoiding both per-query transpose relays.
    sQT0 = cute.make_tensor(cute.recast_ptr(sV.iterator + 3 * DS_ELEMS, lQT.inner, dtype=F16), lQT.outer)
    sQT1 = cute.make_tensor(cute.recast_ptr(sV.iterator + 3 * DS_ELEMS + Q_T1_OFF, lQT.inner, dtype=F16), lQT.outer)
    sQTB0 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQT_bf16.inner, dtype=BF16), lQT_bf16.outer)
    sQTB1 = cute.make_tensor(cute.recast_ptr(sQ.iterator + Q_T1_OFF, lQT_bf16.inner, dtype=BF16), lQT_bf16.outer)
    sDOT0 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lDOT.inner), lDOT.outer)
    sDOT1 = cute.make_tensor(cute.recast_ptr(sDO.iterator + Q_T1_OFF, lDOT.inner), lDOT.outer)
    sKT0 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT.inner, dtype=F16), lKT.outer)
    sKT1 = cute.make_tensor(cute.recast_ptr(sK.iterator + K_T1_OFF, lKT.inner, dtype=F16), lKT.outer)
    sDSmn = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSmn.inner, dtype=F16), lDSmn.outer)
    sKTB0 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT_bf16.inner, dtype=BF16), lKT_bf16.outer)
    sKTB1 = cute.make_tensor(cute.recast_ptr(sK.iterator + K_T1_OFF, lKT_bf16.inner, dtype=BF16), lKT_bf16.outer)
    sDSBmn = cute.make_tensor(cute.recast_ptr(sV.iterator + DS_ELEMS, lDSmn_bf16.inner, dtype=BF16), lDSmn_bf16.outer)
    sDSBLmn = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * DS_ELEMS, lDSmn_bf16.inner, dtype=BF16), lDSmn_bf16.outer)

    kbase = mK.iterator.toint()
    vbase = mV.iterator.toint()
    dobase = mDO.iterator.toint()
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
    # Q is consumed once per query. Keep K/V on the GLOBAL (L2-cached) path
    # because their selected rows are reused across adjacent qperm=1 CTAs,
    # while marking the large Q stream evict-first.
    q_g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.STREAMING),
        BF16,
        num_bits_per_copy=128,
    )
    tiled_q_g2s = cute.make_tiled_copy_tv(q_g2s_atom, cute.make_layout((1, 32)), cute.make_layout((1, 8)))
    thr_q_g2s = tiled_q_g2s.get_slice(lane)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init(mbar + 2, 1)
            cute.arch.mbarrier_init(mbar + 3, 1)
            cute.arch.mbarrier_init(mbar + 4, NTHREADS)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C((BR, HQ)))
    tS0 = _p_tile(tmem_ptr, tS_f.layout, I32(0))

    tDP0 = _dp_tile(tmem_ptr, tS_f.layout, I32(0))
    tKV_f = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C((BR, BR)))
    tKA = cute.make_tensor(tmem_ptr + TM_A, tKV_f.layout)
    tKB = cute.make_tensor(tmem_ptr + TM_B, tKV_f.layout)
    tDQ_f = dSK_mma.make_fragment_C(dSK_mma.partition_shape_C((BR, HQ)))
    tDQ0 = cute.make_tensor(tmem_ptr + TM_DQ0, tDQ_f.layout)
    tDQ1 = cute.make_tensor(tmem_ptr + TM_DQ1, tDQ_f.layout)

    fK = KQ_mma.make_fragment_A(sK)
    fQ = KQ_mma.make_fragment_B(sQ)
    fV = KQ_mma.make_fragment_A(sV)
    fDO = KQ_mma.make_fragment_B(sDO)
    fQT0 = dSQ_mma.make_fragment_A(sQT0)
    fQT1 = dSQ_mma.make_fragment_A(sQT1)
    fQTB0 = dSQ_bf16_mma.make_fragment_A(sQTB0)
    fQTB1 = dSQ_bf16_mma.make_fragment_A(sQTB1)
    fDOT0 = dPV_mma.make_fragment_A(sDOT0)
    fDOT1 = dPV_mma.make_fragment_A(sDOT1)
    fDSb = dSQ_mma.make_fragment_B(sDS)
    fDSBb = dSQ_bf16_mma.make_fragment_B(sDSB)
    fDSBLb = dSQ_bf16_mma.make_fragment_B(sDSBL)
    fPdb = dPV_mma.make_fragment_B(sPd)
    fKT0 = dSK_mma.make_fragment_A(sKT0)
    fKT1 = dSK_mma.make_fragment_A(sKT1)
    fDSmn = dSK_mma.make_fragment_B(sDSmn)
    fKTB0 = dSK_bf16_mma.make_fragment_A(sKTB0)
    fKTB1 = dSK_bf16_mma.make_fragment_A(sKTB1)
    fDSBmn = dSK_bf16_mma.make_fragment_B(sDSBmn)
    fDSBLmn = dSK_bf16_mma.make_fragment_B(sDSBLmn)

    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), F32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), F32)
    st8 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(8)), F32)

    tSd2 = tS0[(None, None), 0, 0]
    tKA2 = tKA[(None, None), 0, 0]
    tKB2 = tKB[(None, None), 0, 0]
    tDQ0_2 = tDQ0[(None, None), 0, 0]
    tDQ1_2 = tDQ1[(None, None), 0, 0]

    cSN = cute.make_identity_tensor((BR, HQ))
    cp_s = tcgen05.make_tmem_copy(ld8, tSd2)
    thr_s = cp_s.get_slice(t128)
    tr_Sd = thr_s.partition_S(tSd2)

    tr_cS = thr_s.partition_D(cSN)
    slot_m = cute.get(tr_cS[0], mode=[0])

    tP0_2 = _p_tile(tmem_ptr, tS_f.layout, I32(0))[(None, None), 0, 0]
    cp_pst = tcgen05.make_tmem_copy(st8, tP0_2)
    thr_pst = cp_pst.get_slice(t128)
    tr_cPst = thr_pst.partition_S(cSN)

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
    # Preserve the FP32 MMA results through the repeated-index scatter and at
    # the public gradient boundary.  the attention backward exposes
    # FP32 Q/K/V gradients for BF16 activations; narrowing here would change
    # that contract before the QKV projection can consume the gradient.
    rPk = cute.make_rmem_tensor(cute.make_layout((2, 32), stride=(1, 2)), F32)
    # The last eighth of each chunk's dK/dV deposit is carried in registers and
    # issued during the *next* chunk's dK/dV-lo MMA, which is otherwise a pure
    # stall with an idle atomic pipe.
    rC = cute.make_rmem_tensor(cute.make_layout(8), F32)
    dbase4 = I32(0)

    cute.arch.barrier()

    # Each transaction barrier has an independent arrival history.  A physical
    # THD padding row launches only the stage-0 KQ/dP work, so its parity must
    # not advance the stage-1/2 waits.
    ph0 = I32(0)
    ph1 = I32(0)
    ph2 = I32(0)
    ph3 = I32(0)
    ph4 = I32(0)
    dk_hi = dim_m + I32(128)
    nkQ = cute.size(fQ, mode=[2])
    nkDS = cute.size(fDSb, mode=[2])
    nkPV = cute.size(fPdb, mode=[2])
    nkKT = cute.size(fKT0, mode=[2])

    d0 = lane * 8
    first_t = (bidx * qperm) % TQ

    # ---- per-query head: index stage, chunk-0 gather, Q stream, delta ----
    first_vlen = mVlen[first_t]
    # Slots past ``valid_length`` carry an exactly-zero dS/P and therefore
    # deposit exact zeros.  Pointing every one of them at key 0 funnels those
    # no-op fp32 atomics through a single L2 address; ``TOPK <= Tk`` always
    # holds, so slot-indexed dummies spread them over 512 distinct rows.
    first_index = I32(tidx)
    if tidx < first_vlen:
        first_index = mIdx[first_t * TOPK + tidx]
    sIdx[tidx] = first_index
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
    qb0 = first_t * (HQ * DIM)
    qptr0 = mQ.iterator.toint() + first_t * q_stride_bytes
    for i in cutlass.range_constexpr(HQ // NWG // 4):
        h = i * (4 * NWG) + warp_idx
        cute.copy(
            tiled_q_g2s,
            thr_q_g2s.partition_S(
                cute.make_tensor(cute.make_ptr(BF16, qptr0 + h * (DIM * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
            ),
            thr_q_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
        )
        goff0 = qb0 + h * DIM + d0
        cute.copy(
            tiled_q_g2s,
            thr_q_g2s.partition_S(
                cute.make_tensor(
                    cute.make_ptr(
                        BF16,
                        dobase + (goff0 - d0) * I32(2),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((1, DIM)),
                )
            ),
            thr_q_g2s.partition_D(cute.local_tile(sDO2, (1, DIM), (h, 0))),
        )
    cute.arch.cp_async_commit_group()
    if tidx < HQ:
        sMax[tidx] = mSoftmaxMax[first_t * HQ + tidx]
        sInv[tidx] = mSoftmaxInvSum[first_t * HQ + tidx]

    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if warp_idx == 0:
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tS0, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS0)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tDP0, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP0)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + 0)

    nq = (TQ - bidx + nctas - I32(1)) // nctas
    for qi in cutlass.range(nq):
        u = bidx + qi * nctas
        t = (u * qperm) % TQ
        qbase = t * (HQ * DIM)
        vlen = mVlen[t]
        nchunk = (vlen + (BR - 1)) // BR
        dbase4 = I32(0)

        # ---------------- fused exact-FP32 delta sweep ----------------
        # The persistent CTA derives its own delta immediately before the
        # backward sweep rather than reading it from a separate prepass
        # kernel.  The two sweeps have complementary bottlenecks - the delta
        # sweep is L2-gather / UMMA bound while the backward sweep is global
        # atomic bound - and the CTAs desynchronise, so the L2 atomic pipe now
        # stays busy through the delta work instead of idling for a whole
        # kernel launch.  The gather for chunk c+1 is issued the moment the
        # S^T/dP^T contraction for chunk c retires, which the standalone
        # prepass could not do.
        if tidx < 2 * (TOPK // BR) * HQ:
            sPart[tidx // ((TOPK // BR) * HQ), (tidx // HQ) % (TOPK // BR), tidx % HQ] = F32(0.0)
        cute.arch.barrier()

        for c in cutlass.range(nchunk):
            cbase = c * BR
            cpar = c % I32(2)
            cute.arch.mbarrier_wait(mbar + 0, ph0)
            ph0 = ph0 ^ I32(1)

            if c + 1 < nchunk:
                nbase = cbase + BR
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    kbase + sIdx[nbase + r] * k_stride_bytes,
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
                                    vbase + sIdx[nbase + r] * v_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar + 4)

            trDPc = thr_s.partition_S(_dp_tile(tmem_ptr, tS_f.layout, c)[(None, None), 0, 0])
            trSc = thr_s.partition_S(_p_tile(tmem_ptr, tS_f.layout, c)[(None, None), 0, 0])
            tr_PcD = thr_pst.partition_D(_p_tile(tmem_ptr, tS_f.layout, c)[(None, None), 0, 0])
            for j in cutlass.range_constexpr(HQ // 8):
                if wg == SM_WG[j]:
                    rSd = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    rDPd = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    rDelta = cute.make_rmem_tensor(cute.make_layout(8), F32)
                    rPsum = cute.make_rmem_tensor(tr_cPst[None, None, 0].shape, F32)
                    oPd = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    cute.copy(cp_s, trSc[None, None, j], rSd)
                    cute.copy(cp_s, trDPc[None, None, j], rDPd)
                    for e in cutlass.range_constexpr(0, 8, 2):
                        h0 = j * 8 + e
                        h1 = h0 + 1
                        p0 = F32(0.0)
                        p1 = F32(0.0)
                        # A slot at or past ``valid_length`` carries no semantic
                        # support, so its probability is exactly zero -- but the
                        # dP^T tile for that slot was formed from a *non-semantic*
                        # gathered V row and is only bounded by the caller's BF16
                        # range.  Forming ``p * dP`` unconditionally therefore turns
                        # a legal (finite, large) padding row into 0 * inf = NaN and
                        # poisons the whole per-head delta.  Keep the product inside
                        # the same validity predicate that already zeroes p; for
                        # every semantic slot this is bit-identical.
                        pdp0 = F32(0.0)
                        pdp1 = F32(0.0)
                        if cbase + slot_m < vlen:
                            x0, x1 = cute.arch.add_packed_f32x2(
                                (rSd[e], rSd[e + 1]),
                                (F32(0.0) - sMax[h0], F32(0.0) - sMax[h1]),
                            )
                            x0, x1 = cute.arch.mul_packed_f32x2(
                                (x0, x1),
                                (F32(SCALE_LOG2E), F32(SCALE_LOG2E)),
                            )
                            p0 = cute.math.exp2(x0, fastmath=True)
                            p1 = cute.math.exp2(x1, fastmath=True)
                            p0, p1 = cute.arch.mul_packed_f32x2((p0, p1), (sInv[h0], sInv[h1]))
                            pdp0, pdp1 = cute.arch.mul_packed_f32x2((p0, p1), (rDPd[e], rDPd[e + 1]))
                        rDelta[e] = pdp0
                        rDelta[e + 1] = pdp1
                        rPsum[e] = p0
                        rPsum[e + 1] = p1
                        # dV = BF16(P)^T @ BF16(dO) needs no delta, so the whole dV
                        # half of the scatter can be produced here.
                        oPd[0, e] = p0.to(BF16)
                        oPd[0, e + 1] = p1.to(BF16)
                    # S and P share these columns: retire the score load before
                    # the probability store overwrites it.
                    cute.arch.fence_view_async_tmem_load()
                    cute.copy(cp_pst, rPsum, tr_PcD[None, None, j])
                    # Warp-sum both delta channels in 2 x 32 ops instead of eight
                    # 5-round butterflies (160 ops).  A ``warp_reduction_sum`` per
                    # value broadcasts every partial to all 32 lanes and then
                    # discards 31/32 of that work, because only lane 0 stores.
                    # Halve the *value* count with every butterfly step instead:
                    # each lane keeps the half its partner does not need and adds
                    # the partner's complementary half, so the eight head sums fan
                    # out over eight distinct lanes and eight lanes publish in
                    # parallel.  Lane l < 8 holds the head whose index is the 3-bit
                    # reversal of l.  rDelta is dead afterwards and rPsum has
                    # already been consumed by the TMEM store above, so both are
                    # reduced in place at zero register cost.
                    hip0 = (lane & 1) != 0
                    lo = rDelta[0]
                    hv = rDelta[4]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rDelta[0] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rDelta[1]
                    hv = rDelta[5]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rDelta[1] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rDelta[2]
                    hv = rDelta[6]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rDelta[2] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rDelta[3]
                    hv = rDelta[7]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rDelta[3] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    hip1 = (lane & 2) != 0
                    lo = rDelta[0]
                    hv = rDelta[2]
                    snd = hv
                    kep = lo
                    if hip1:
                        snd = lo
                        kep = hv
                    rDelta[0] = kep + cute.arch.shuffle_sync_bfly(snd, 2)
                    lo = rDelta[1]
                    hv = rDelta[3]
                    snd = hv
                    kep = lo
                    if hip1:
                        snd = lo
                        kep = hv
                    rDelta[1] = kep + cute.arch.shuffle_sync_bfly(snd, 2)
                    hip2 = (lane & 4) != 0
                    lo = rDelta[0]
                    hv = rDelta[1]
                    snd = hv
                    kep = lo
                    if hip2:
                        snd = lo
                        kep = hv
                    rDelta[0] = kep + cute.arch.shuffle_sync_bfly(snd, 4)
                    dtot = rDelta[0]
                    dtot = dtot + cute.arch.shuffle_sync_bfly(dtot, 8)
                    dtot = dtot + cute.arch.shuffle_sync_bfly(dtot, 16)
                    hip0 = (lane & 1) != 0
                    lo = rPsum[0]
                    hv = rPsum[4]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rPsum[0] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rPsum[1]
                    hv = rPsum[5]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rPsum[1] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rPsum[2]
                    hv = rPsum[6]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rPsum[2] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    lo = rPsum[3]
                    hv = rPsum[7]
                    snd = hv
                    kep = lo
                    if hip0:
                        snd = lo
                        kep = hv
                    rPsum[3] = kep + cute.arch.shuffle_sync_bfly(snd, 1)
                    hip1 = (lane & 2) != 0
                    lo = rPsum[0]
                    hv = rPsum[2]
                    snd = hv
                    kep = lo
                    if hip1:
                        snd = lo
                        kep = hv
                    rPsum[0] = kep + cute.arch.shuffle_sync_bfly(snd, 2)
                    lo = rPsum[1]
                    hv = rPsum[3]
                    snd = hv
                    kep = lo
                    if hip1:
                        snd = lo
                        kep = hv
                    rPsum[1] = kep + cute.arch.shuffle_sync_bfly(snd, 2)
                    hip2 = (lane & 4) != 0
                    lo = rPsum[0]
                    hv = rPsum[1]
                    snd = hv
                    kep = lo
                    if hip2:
                        snd = lo
                        kep = hv
                    rPsum[0] = kep + cute.arch.shuffle_sync_bfly(snd, 4)
                    dsum = rPsum[0]
                    dsum = dsum + cute.arch.shuffle_sync_bfly(dsum, 8)
                    dsum = dsum + cute.arch.shuffle_sync_bfly(dsum, 16)
                    if lane < 8:
                        brl = ((lane & 1) << 2) | (lane & 2) | ((lane & 4) >> 2)
                        sRed[cpar, 0, j * 8 + brl, t128 // 32] = dtot
                        sRed[cpar, 1, j * 8 + brl, t128 // 32] = dsum
                    cute.autovec_copy(oPd, cute.local_tile(sPd2, (1, 8), (slot_m, j)))
                pass
            # One publication barrier per delta chunk instead of two.  The old
            # schedule crossed a CTA barrier to hand sPd to the dV^T UMMA and
            # then, four instructions later, crossed a second one to hand the
            # chunk-(c+1) K/V gather to the score UMMA.  Both producers (the
            # probability window and the cp.async gather) are complete at the
            # same point, so a single barrier that carries both the async-proxy
            # fence and the cp.async wait publishes both, and warp 0 then issues
            # the two UMMAs back to back into the same MMA pipe.
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # ---- dV^T for both 128-dim tiles, into the idle TM_A / TM_B ----
            if warp_idx == 0:
                dPV_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkPV):
                    cute.gemm(dPV_mma, tKA, fDOT0[None, None, kb, 0], fPdb[None, None, kb, 0], tKA)
                    dPV_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 1)
                dPV_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkPV):
                    cute.gemm(dPV_mma, tKB, fDOT1[None, None, kb, 0], fPdb[None, None, kb, 0], tKB)
                    dPV_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 3)

            if tidx < HQ:
                dtot = F32(0.0)
                dsum = F32(0.0)
                for w in cutlass.range_constexpr(4):
                    dtot = dtot + sRed[cpar, 0, tidx, w]
                    dsum = dsum + sRed[cpar, 1, tidx, w]
                sPart[0, c, tidx] = dtot
                sPart[1, c, tidx] = dsum
                # On the final chunk this thread owns every sPart entry for
                # its head, so it can form delta immediately while the other
                # warps drain dV.  The post-drain CTA barrier below publishes
                # sDel to the backward sweep as well as protecting TMEM.
                if c + 1 == nchunk:
                    dtot = F32(0.0)
                    dsum = F32(0.0)
                    for cc in cutlass.range_constexpr(TOPK // BR):
                        dtot = dtot + sPart[0, cc, tidx]
                        dsum = dsum + sPart[1, cc, tidx]
                    if dsum > F32(0.0):
                        sDel[tidx] = dtot / dsum
                    else:
                        sDel[tidx] = F32(0.0)

            # Drain tile A as soon as it retires while tile B is still moving
            # through the MMA pipe.  Each feature tile is drained separately,
            # so only 16 fp32 values are live across a RED burst.
            wgb = cbase + wg * I32(32)
            wgb4 = c * I32(BR // 4) + wg * I32(8)
            cute.arch.mbarrier_wait(mbar + 1, ph1)
            ph1 = ph1 ^ I32(1)
            for h in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(NWG):
                    if wg == j:
                        cute.copy(cp_kv, tr_A[None, None, 2 * j + h], rA)
                        for e in cutlass.range_constexpr(16):
                            rPk[0, e] = rA[e]
                    pass
                cute.arch.fence_view_async_tmem_load()
                for qv in cutlass.range_constexpr(4):
                    ri = _idx4(sIdx4, wgb4 + I32(h * 4 + qv))
                    for e in cutlass.range_constexpr(4):
                        koff = ri[e] * I32(DIM) + dim_m
                        _red_add_f32(mDV.iterator + koff, rPk[0, qv * 4 + e])
            # The delta gather publishes to its own transaction barrier, so the
            # score/dP pair can wait in warp 0 while the other warps begin the
            # dV drain.  Issuing here overlaps the pair with tile-B deposits.
            if c + 1 < nchunk:
                tDPn = _dp_tile(tmem_ptr, tS_f.layout, c + I32(1))
                tSn = _p_tile(tmem_ptr, tS_f.layout, c + I32(1))
                if warp_idx == 0:
                    cute.arch.mbarrier_wait(mbar + 4, ph4)
                    ph4 = ph4 ^ I32(1)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tSn, fK[None, None, kb, 0], fQ[None, None, kb, 0], tSn)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tDPn, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDPn)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 0)
            cute.arch.mbarrier_wait(mbar + 3, ph3)
            ph3 = ph3 ^ I32(1)
            for h in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(NWG):
                    if wg == j:
                        cute.copy(cp_kv, tr_B[None, None, 2 * j + h], rB)
                        for e in cutlass.range_constexpr(16):
                            rPk[1, e] = rB[e]
                    pass
                cute.arch.fence_view_async_tmem_load()
                for qv in cutlass.range_constexpr(4):
                    ri = _idx4(sIdx4, wgb4 + I32(h * 4 + qv))
                    for e in cutlass.range_constexpr(4):
                        koff = ri[e] * I32(DIM) + dk_hi
                        _red_add_f32(mDV.iterator + koff, rPk[1, qv * 4 + e])
            # No end-of-chunk CTA barrier.  TM_A/TM_B are only rewritten by the
            # next chunk's dV^T UMMA, which is issued after that chunk's single
            # publication barrier -- a barrier every thread must cross after
            # this drain, so it dominates.  sPd is protected by the mbar+3 wait
            # above and sK/sV by the mbar+0 wait at the top of the next
            # iteration.  Dropping it lets a warpgroup that has finished its
            # deposits start the next chunk's probability work while its peers
            # are still draining, which is the overlap the reduction pipe needs.

        # A zero-length physical padding row has no final chunk in which to
        # publish delta.  Keep it finite before the shared post-drain barrier.
        if nchunk == 0 and tidx < HQ:
            sDel[tidx] = F32(0.0)
        cute.arch.fence_view_async_tmem_store()
        # Jointly protects the completed dV TMEM drain and publishes sDel.
        cute.arch.barrier()

        # The FP16 contraction is used only when every BF16 Q value survives an
        # exact BF16 -> FP16 -> BF16 round trip.  This rejects both overflow and
        # underflow (including finite BF16 subnormals) instead of using a
        # one-sided magnitude check.  The decision is CTA-uniform because all
        # heads participate in each dK tile.
        if tidx == 0:
            sPath[0] = I32(0)
        cute.arch.barrier()
        q_unsafe = I32(0)
        for i in cutlass.range_constexpr(HQ // NWG // 4):
            h = i * (4 * NWG) + warp_idx
            rq_b = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
            rq_f = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
            cute.autovec_copy(cute.local_tile(sQ2, (1, 8), (h, lane)), rq_b)
            for e in cutlass.range_constexpr(8):
                rq_f[0, e] = rq_b[0, e].to(F16)
                if rq_f[0, e].to(BF16) != rq_b[0, e]:
                    q_unsafe = I32(1)
            cute.autovec_copy(rq_f, cute.local_tile(sQF2, (1, 8), (h, lane)))
        if q_unsafe != 0:
            cute.arch.atomic_or(sPath.iterator, I32(1), sem="relaxed", scope="cta")
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        if nchunk == 0:
            # Retire the prologue's outstanding S^T/dP^T transaction.
            cute.arch.mbarrier_wait(mbar + 0, ph0)
            ph0 = ph0 ^ I32(1)
        # The delta sweep leaves its last K chunk resident.  Traverse backward
        # chunks in reverse and consume that tile without a redundant gather.
        for ci in cutlass.range(nchunk):
            c = nchunk - I32(1) - ci
            cbase = c * BR

            # ---- cached exact-FP32 P / dS ----
            trPc = thr_s.partition_S(_p_tile(tmem_ptr, tS_f.layout, c)[(None, None), 0, 0])
            trDPc = thr_s.partition_S(_dp_tile(tmem_ptr, tS_f.layout, c)[(None, None), 0, 0])
            if tidx == 0:
                sPath[1] = I32(0)
                sPath[2] = I32(0)
            cute.arch.barrier()
            ds_unsafe = I32(0)
            for j in cutlass.range_constexpr(HQ // 8):
                if wg == SM_WG[j]:
                    rP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    rDP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    cute.copy(cp_s, trPc[None, None, j], rP)
                    cute.copy(cp_s, trDPc[None, None, j], rDP)
                    oS = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                    oSB = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    oSBL = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    oMask = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    # Mirror forward's packed FP32 normalization arithmetic so
                    # ``oP`` is the same BF16 probability seen by its PV MMA.
                    for e in cutlass.range_constexpr(0, 8, 2):
                        h0 = j * 8 + e
                        h1 = h0 + 1
                        p0 = rP[e]
                        p1 = rP[e + 1]
                        # A softmax over zero or one semantic support has an
                        # exactly zero derivative.  Enforce that identity so the
                        # independently reduced delta cannot leak ulps into
                        # dQ/dK on causal length-one rows.
                        ds0 = F32(0.0)
                        ds1 = F32(0.0)
                        if vlen > 1:
                            # Same padding-slot hazard as the delta sweep: p is the
                            # cached exact zero for a slot past ``valid_length``,
                            # but its dP^T came from a non-semantic gathered V row.
                            # Predicate the product instead of relying on 0 * dP so
                            # a legal large-magnitude unselected row cannot turn dS
                            # (and hence dQ/dK) into NaN.  Bit-identical on every
                            # semantic slot.
                            if cbase + slot_m < vlen:
                                ddp0, ddp1 = cute.arch.sub_packed_f32x2((rDP[e], rDP[e + 1]), (sDel[h0], sDel[h1]))
                                ds0, ds1 = cute.arch.mul_packed_f32x2((p0, p1), (ddp0, ddp1))
                                ds0, ds1 = cute.arch.mul_packed_f32x2((ds0, ds1), (F32(SCALE), F32(SCALE)))
                        for ee in cutlass.range_constexpr(2):
                            ds = ds0 if ee == 0 else ds1
                            dsf = ds.to(F16)
                            dsh = ds.to(BF16)
                            dsl = ds - dsh.to(F32)
                            oSB[0, e + ee] = dsh
                            oSBL[0, e + ee] = dsl.to(BF16)

                            # FP16 must preserve every nonzero dS and approximate
                            # it to within 2^-10 relative error.  The latter is a
                            # documented two-ulp ceiling around normal FP16 values;
                            # it also rejects NaN/Inf and subnormal conversions that
                            # are finite but too lossy for the single-pass path.
                            ads = ds if ds >= F32(0.0) else F32(0.0) - ds
                            back = dsf.to(F32)
                            err = back - ds
                            aerr = err if err >= F32(0.0) else F32(0.0) - err
                            unsafe_elem = I32(0)
                            if ds != ds or ads > F32(65504.0):
                                unsafe_elem = I32(1)
                            elif ads != F32(0.0):
                                if back == F32(0.0) or aerr > ads * F32(0.0009765625):
                                    unsafe_elem = I32(1)
                            if unsafe_elem != 0:
                                ds_unsafe = I32(1)
                                oS[0, e + ee] = F16(0.0)
                                oMask[0, e + ee] = F32(1.0).to(BF16)
                            else:
                                oS[0, e + ee] = dsf
                                oMask[0, e + ee] = F32(0.0).to(BF16)
                    cute.autovec_copy(oS, cute.local_tile(sDS2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oSB, cute.local_tile(sDSB2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oSBL, cute.local_tile(sDSBL2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oMask, cute.local_tile(sPd2, (1, 8), (slot_m, j)))
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # K uses the same exact round-trip requirement as Q.  Q/K hazards
            # require a full BF16 contraction.  A dS-only hazard is handled
            # without dropping any term: safe dS values use the more accurate
            # FP16 contraction, while unsafe values (marked in dead sPd
            # storage) contribute through BF16 high/residual into the same
            # FP32 accumulator.
            k_unsafe = I32(0)
            for i in cutlass.range_constexpr(BR // NWG // 4):
                r = i * (4 * NWG) + warp_idx
                rk_b = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                rk_f = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                cute.autovec_copy(cute.local_tile(sK2, (1, 8), (r, lane)), rk_b)
                for e in cutlass.range_constexpr(8):
                    rk_f[0, e] = rk_b[0, e].to(F16)
                    if rk_f[0, e].to(BF16) != rk_b[0, e]:
                        k_unsafe = I32(1)
                # Store only after the CTA has decided that BF16 K is dead.
            if k_unsafe != 0:
                cute.arch.atomic_or(sPath.iterator + 1, I32(1), sem="relaxed", scope="cta")
            if ds_unsafe != 0:
                cute.arch.atomic_or(sPath.iterator + 2, I32(1), sem="relaxed", scope="cta")
            cute.arch.barrier()
            full_bf16 = sPath[0] != 0 or sPath[1] != 0
            use_fp16 = not full_bf16
            use_bf16 = full_bf16 or sPath[2] != 0
            if use_fp16:
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    rk_b = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    rk_f = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                    cute.autovec_copy(cute.local_tile(sK2, (1, 8), (r, lane)), rk_b)
                    for e in cutlass.range_constexpr(8):
                        rk_f[0, e] = rk_b[0, e].to(F16)
                    cute.autovec_copy(rk_f, cute.local_tile(sKF2, (1, 8), (r, lane)))
                # Retain only the dS elements that could not safely use FP16;
                # the others are already represented in sDS and must not be
                # counted twice by the BF16 correction MMA.
                for i in cutlass.range_constexpr(DS_ELEMS // NTHREADS):
                    linear = tidx + i * NTHREADS
                    row = linear // HQ
                    head = linear - row * HQ
                    if sPd2[row, head] == F32(0.0).to(BF16):
                        sDSB2[row, head] = F32(0.0).to(BF16)
                        sDSBL2[row, head] = F32(0.0).to(BF16)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # K needs no conversion at all now: the dQ contraction consumes the
            # BF16 K tile in place through an MN-major view.  Both Q^T halves
            # are already resident in the freed V staging tail.
            # ---- dK^T dims [0,128) + persistent dQ^T ----
            if warp_idx == 0:
                if use_fp16:
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKA, fQT0[None, None, kb, 0], fDSb[None, None, kb, 0], tKA)
                        dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                if use_bf16:
                    dSQ_bf16_mma.set(tcgen05.Field.ACCUMULATE, use_fp16)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_bf16_mma, tKA, fQTB0[None, None, kb, 0], fDSBb[None, None, kb, 0], tKA)
                        dSQ_bf16_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(not diagnostic_single_bf16_ds):
                            cute.gemm(dSQ_bf16_mma, tKA, fQTB0[None, None, kb, 0], fDSBLb[None, None, kb, 0], tKA)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 1)
                # ---- dK^T for feature dims [128,256), into its own tile ----
                # Queued in the SAME warp-0 block as dK-lo/dQ instead of after
                # the dK-lo drain.  Now that dK-lo and dK-hi own disjoint
                # accumulators there is no ordering reason to hold it back, and
                # the UMMA pipe runs it while the CTA is draining TM_A -- so the
                # mbar+2 wait below is usually already satisfied instead of
                # exposing a whole UMMA latency on the critical path.
                if use_fp16:
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_mma, tKB, fQT1[None, None, kb, 0], fDSb[None, None, kb, 0], tKB)
                        dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                if use_bf16:
                    dSQ_bf16_mma.set(tcgen05.Field.ACCUMULATE, use_fp16)
                    for kb in cutlass.range_constexpr(nkDS):
                        cute.gemm(dSQ_bf16_mma, tKB, fQTB1[None, None, kb, 0], fDSBb[None, None, kb, 0], tKB)
                        dSQ_bf16_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(not diagnostic_single_bf16_ds):
                            cute.gemm(dSQ_bf16_mma, tKB, fQTB1[None, None, kb, 0], fDSBLb[None, None, kb, 0], tKB)
                # dK-hi gets its own completion barrier.  It is 4 UMMAs; dQ that
                # follows it is 16 (two 128-dim tiles x two dS terms x four
                # K-steps).  Sharing one barrier made the whole dK-hi drain wait
                # out the dQ block even though dK-hi retires first, so the
                # deposit pipe sat idle for the length of the largest UMMA group
                # in the kernel.  The dQ wait moves down to just before the K
                # prefetch, which is the first thing that actually depends on it
                # (the prefetch overwrites the sK tile the dQ contraction reads).
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 2)
                # dQ is independent of both dK tiles.  Queue it after dK-hi so
                # the first dK REDs can run while the remaining UMMAs retire.
                if use_fp16:
                    dSK_mma.set(tcgen05.Field.ACCUMULATE, ci != 0)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_mma, tDQ0, fKT0[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ0)
                        cute.gemm(dSK_mma, tDQ1, fKT1[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ1)
                        dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
                # A hybrid chunk cannot consume BF16 K yet: its in-place FP16
                # scratch aliases sK.  The BF16 correction is issued below,
                # after the FP16 dQ group retires and K is exactly restored.
                if use_bf16 and not use_fp16:
                    dSK_bf16_mma.set(tcgen05.Field.ACCUMULATE, ci != 0)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_bf16_mma, tDQ0, fKTB0[None, None, kb, 0], fDSBmn[None, None, kb, 0], tDQ0)
                        cute.gemm(dSK_bf16_mma, tDQ1, fKTB1[None, None, kb, 0], fDSBmn[None, None, kb, 0], tDQ1)
                        dSK_bf16_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(not diagnostic_single_bf16_ds):
                            cute.gemm(dSK_bf16_mma, tDQ0, fKTB0[None, None, kb, 0], fDSBLmn[None, None, kb, 0], tDQ0)
                            cute.gemm(dSK_bf16_mma, tDQ1, fKTB1[None, None, kb, 0], fDSBLmn[None, None, kb, 0], tDQ1)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 3)

            if use_fp16 and use_bf16:
                # The hybrid path needs both FP16 and BF16 views of K, but the
                # 128x256 K tile fills its shared allocation.  Wait only on
                # this rare correction path, restore the original BF16 bits
                # exactly (K passed the round-trip classifier), then append the
                # unsafe-dS BF16 high/residual terms to persistent dQ.
                cute.arch.mbarrier_wait(mbar + 3, ph3)
                ph3 = ph3 ^ I32(1)
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    rk_f = cute.make_rmem_tensor(cute.make_layout((1, 8)), F16)
                    rk_b = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    cute.autovec_copy(cute.local_tile(sKF2, (1, 8), (r, lane)), rk_f)
                    for e in cutlass.range_constexpr(8):
                        rk_b[0, e] = rk_f[0, e].to(BF16)
                    cute.autovec_copy(rk_b, cute.local_tile(sK2, (1, 8), (r, lane)))
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    dSK_bf16_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkKT):
                        cute.gemm(dSK_bf16_mma, tDQ0, fKTB0[None, None, kb, 0], fDSBmn[None, None, kb, 0], tDQ0)
                        cute.gemm(dSK_bf16_mma, tDQ1, fKTB1[None, None, kb, 0], fDSBmn[None, None, kb, 0], tDQ1)
                        if cutlass.const_expr(not diagnostic_single_bf16_ds):
                            cute.gemm(dSK_bf16_mma, tDQ0, fKTB0[None, None, kb, 0], fDSBLmn[None, None, kb, 0], tDQ0)
                            cute.gemm(dSK_bf16_mma, tDQ1, fKTB1[None, None, kb, 0], fDSBLmn[None, None, kb, 0], tDQ1)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 3)

            wgb = cbase + wg * I32(32)
            wgb4 = c * I32(BR // 4) + wg * I32(8)
            if ci > 0:
                dbo4 = dbase4 + wg * I32(8)
                for qv in cutlass.range_constexpr(2):
                    ri = _idx4(sIdx4, dbo4 + I32(qv))
                    for e in cutlass.range_constexpr(4):
                        _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dk_hi), rC[qv * 4 + e])
            cute.arch.mbarrier_wait(mbar + 1, ph1)

            # Two 16-slot halves.  TM_A must be fully drained before the next
            # UMMA overwrites it, but only the second half has to stay live
            # across that MMA, so the pressure peak carries 16 fp32 not 32.
            for h in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(NWG):
                    if wg == j:
                        cute.copy(cp_kv, tr_A[None, None, 2 * j + h], rA)
                        for e in cutlass.range_constexpr(16):
                            rPk[0, e] = rA[e]
                    pass
                cute.arch.fence_view_async_tmem_load()
                if h == 0:
                    for qv in cutlass.range_constexpr(4):
                        ri = _idx4(sIdx4, wgb4 + I32(qv))
                        for e in cutlass.range_constexpr(4):
                            _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dim_m), rPk[0, qv * 4 + e])

            # ---- dK scatter, dims [0,128) second half (overlaps the MMA) ----
            for qv in cutlass.range_constexpr(4):
                ri = _idx4(sIdx4, wgb4 + I32(4 + qv))
                for e in cutlass.range_constexpr(4):
                    _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dim_m), rPk[0, qv * 4 + e])

            # Retire dQ as early as possible, then launch the previous-chunk K
            # gather before draining dK-hi.  The following RED burst covers the
            # cp.async latency without aliasing dQ's sK operand.
            cute.arch.mbarrier_wait(mbar + 3, ph3)
            ph3 = ph3 ^ I32(1)
            if c > 0:
                nbase = cbase - BR
                for i in cutlass.range_constexpr(BR // NWG // 4):
                    r = i * (4 * NWG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    kbase + sIdx[nbase + r] * k_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
                    )
                cute.arch.cp_async_commit_group()

            cute.arch.mbarrier_wait(mbar + 2, ph2)

            # Two 16-slot halves.  TM_A must be fully drained before the next
            # UMMA overwrites it, but only the second half has to stay live
            # across that MMA, so the pressure peak carries 16 fp32 not 32.
            for h in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(NWG):
                    if wg == j:
                        cute.copy(cp_kv, tr_B[None, None, 2 * j + h], rA)
                        for e in cutlass.range_constexpr(16):
                            rPk[0, e] = rA[e]
                    pass
                cute.arch.fence_view_async_tmem_load()
                if h == 0:
                    for qv in cutlass.range_constexpr(4):
                        ri = _idx4(sIdx4, wgb4 + I32(qv))
                        for e in cutlass.range_constexpr(4):
                            _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dk_hi), rPk[0, qv * 4 + e])
            # No barrier here either.  The next writer of TM_A is the following
            # chunk's dK-lo UMMA, issued only after that chunk's dS-publication
            # barrier, which every thread crosses after this drain.  Nothing
            # between here and that barrier touches shared state this drain
            # produced: the K prefetch targets sK, the trailing deposits read
            # registers, and sDS/sDSL were consumed by the mbar+2 UMMA above.
            # ---- dK scatter, dims [128,256), slots wgb+16..wgb+23 ----
            # These cover the cp.async gather issued just above; the final
            # eighth stays in rC and is issued inside the next chunk's dK-lo
            # MMA window, which is otherwise a pure stall.
            for qv in cutlass.range_constexpr(2):
                ri = _idx4(sIdx4, wgb4 + I32(4 + qv))
                for e in cutlass.range_constexpr(4):
                    _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dk_hi), rPk[0, qv * 4 + e])
            for e in cutlass.range_constexpr(8):
                rC[e] = rPk[0, e + 8]
            dbase4 = c * I32(BR // 4) + I32(6)
            ph1 = ph1 ^ I32(1)
            ph2 = ph2 ^ I32(1)

        # A zero-length physical THD padding row retires its outstanding
        # stage-0 transaction in the delta sweep above (it never re-stages a
        # chunk-0 contraction), so nothing is left dangling here.

        # A physical THD padding row has no selected support.  There is no
        # final deferred dK/dV fragment in that case; atomically writing the
        # uninitialized rC/rD registers would corrupt key zero (and can make
        # a later real row numerically wrong).
        if nchunk > 0:
            dbo4 = dbase4 + wg * I32(8)
            for qv in cutlass.range_constexpr(2):
                ri = _idx4(sIdx4, dbo4 + I32(qv))
                for e in cutlass.range_constexpr(4):
                    _red_add_f32(mDK.iterator + (ri[e] * I32(DIM) + dk_hi), rC[qv * 4 + e])

        un = u + nctas
        tn = (un * qperm) % TQ
        if un < TQ:
            cute.arch.barrier()

            # ---- per-query head: index stage, chunk-0 gather, Q stream, delta ----
            next_vlen = mVlen[tn]
            next_index = I32(tidx)
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
            qb1 = tn * (HQ * DIM)
            qptr1 = mQ.iterator.toint() + tn * q_stride_bytes
            for i in cutlass.range_constexpr(HQ // NWG // 4):
                h = i * (4 * NWG) + warp_idx
                cute.copy(
                    tiled_q_g2s,
                    thr_q_g2s.partition_S(
                        cute.make_tensor(cute.make_ptr(BF16, qptr1 + h * (DIM * 2), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((1, DIM)))
                    ),
                    thr_q_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
                )
                goff1 = qb1 + h * DIM + d0
                cute.copy(
                    tiled_q_g2s,
                    thr_q_g2s.partition_S(
                        cute.make_tensor(
                            cute.make_ptr(
                                BF16,
                                dobase + (goff1 - d0) * I32(2),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((1, DIM)),
                        )
                    ),
                    thr_q_g2s.partition_D(cute.local_tile(sDO2, (1, DIM), (h, 0))),
                )
            cute.arch.cp_async_commit_group()
            if tidx < HQ:
                sMax[tidx] = mSoftmaxMax[tn * HQ + tidx]
                sInv[tidx] = mSoftmaxInvSum[tn * HQ + tidx]

        # ---------------- dQ epilogue ----------------
        if vlen > 0:
            for j in cutlass.range_constexpr(HQ // 16):
                if wg == (j % NWG):
                    r0 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                    r1 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                    cute.copy(cp_dq, tr_DQ0[None, None, j], r0)
                    cute.copy(cp_dq, tr_DQ1[None, None, j], r1)
                    for e in cutlass.range_constexpr(16):
                        h = j * 16 + e
                        dqo = qbase + h * I32(DIM) + dq_m
                        _st_stream_f32(mDQ.iterator + dqo, r0[e])
                        _st_stream_f32(mDQ.iterator + (dqo + I32(128)), r1[e])
                pass
        else:
            # Do not read uninitialized TMEM when a padding query has no
            # chunks.  This is part of the sparse-backward kernel contract,
            # not a host-side masking workaround.
            for j in cutlass.range_constexpr(HQ // 16):
                if wg == (j % NWG):
                    for e in cutlass.range_constexpr(16):
                        h = j * 16 + e
                        mDQ[qbase + h * DIM + dq_m] = F32(0.0)
                        mDQ[qbase + h * DIM + dq_m + 128] = F32(0.0)
                pass
        cute.arch.fence_view_async_tmem_load()
        if un < TQ:

            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tS0, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS0)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tDP0, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP0)
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
    mDQ,
    mDK,
    mDV,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    TQ: I32,
    TK: I32,
    nctas: I32,
    qperm: I32,
    diagnostic_single_bf16_ds: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    cg = tcgen05.CtaGroup.ONE
    KQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, cg, (BR, HQ))
    dSQ_mma = sm100_utils.make_trivial_tiled_mma(F16, F16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dPV_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dSK_mma = sm100_utils.make_trivial_tiled_mma(F16, F16, OperandMajorMode.MN, OperandMajorMode.MN, F32, cg, (BR, HQ))
    dSQ_bf16_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (BR, BR))
    dSK_bf16_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.MN, F32, cg, (BR, HQ))

    lQs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, HQ, DIM), BF16, 1)
    lKs = sm100_utils.make_smem_layout_a(KQ_mma, (BR, HQ, DIM), BF16, 1)
    lDSs = sm100_utils.make_smem_layout_b(dSQ_mma, (BR, BR, HQ), F16, 1)
    lPs = sm100_utils.make_smem_layout_b(dPV_mma, (BR, BR, HQ), BF16, 1)
    lQT = sm100_utils.make_smem_layout_a(dSQ_mma, (BR, BR, HQ), F16, 1)
    lDOT = sm100_utils.make_smem_layout_a(dPV_mma, (BR, BR, HQ), BF16, 1)
    lKT = sm100_utils.make_smem_layout_a(dSK_mma, (BR, HQ, BR), F16, 1)
    lDSmn = sm100_utils.make_smem_layout_b(dSK_mma, (BR, HQ, BR), F16, 1)
    lDSs_bf16 = sm100_utils.make_smem_layout_b(dSQ_bf16_mma, (BR, BR, HQ), BF16, 1)
    lQT_bf16 = sm100_utils.make_smem_layout_a(dSQ_bf16_mma, (BR, BR, HQ), BF16, 1)
    lKT_bf16 = sm100_utils.make_smem_layout_a(dSK_bf16_mma, (BR, HQ, BR), BF16, 1)
    lDSmn_bf16 = sm100_utils.make_smem_layout_b(dSK_bf16_mma, (BR, HQ, BR), BF16, 1)

    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    a128_f16 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, F16)
    # Hq=32 -> 64B contiguous extent -> K_SW64 (an Hq=80 schedule needs K_SW32).
    a32_ds = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, F16)
    a32_ds_bf16 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, BF16)
    a32_p = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW64, BF16)
    lQ = cute.tile_to_shape(a128, (HQ, DIM), order=(0, 1))
    lK = cute.tile_to_shape(a128, (BR, DIM), order=(0, 1))
    lQscratch = cute.tile_to_shape(a128_f16, (HQ, DIM), order=(0, 1))
    lDS = cute.tile_to_shape(a32_ds, (BR, HQ), order=(0, 1))
    lDS_bf16 = cute.tile_to_shape(a32_ds_bf16, (BR, HQ), order=(0, 1))
    lP = cute.tile_to_shape(a32_p, (BR, HQ), order=(0, 1))

    # K/V are globally gathered under CP, while Q remains rank-local.
    # The dK/dV accumulation workspace is keyed by TK; mDelta is keyed by TQ.
    nelem = TK * DIM
    # The exact FP32 delta prepass is fused into the persistent backward now:
    # a separate launch left the L2 atomic pipe completely idle for its whole
    # duration and could not overlap its gathers with anything.

    _zero_kernel(mDK, mDV, nelem).launch(
        grid=[cute.ceil_div(nelem, 2048), 1, 1],
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
        mDQ,
        mDK,
        mDV,
        KQ_mma,
        dSQ_mma,
        dPV_mma,
        dSK_mma,
        dSQ_bf16_mma,
        dSK_bf16_mma,
        lQ,
        lK,
        lQscratch,
        lDS,
        lP,
        lQT,
        lDOT,
        lKT,
        lDSmn,
        lDS_bf16,
        lQT_bf16,
        lKT_bf16,
        lDSmn_bf16,
        lQs,
        lKs,
        lDSs,
        lDSs_bf16,
        lPs,
        q_token_stride,
        k_token_stride,
        v_token_stride,
        TQ,
        nctas,
        qperm,
        diagnostic_single_bf16_ds,
    ).launch(grid=[nctas, 1, 1], block=[NTHREADS, 1, 1], stream=stream)


_compiled = {}
_smc = {}
_DIAGNOSTIC_SINGLE_BF16_DS = os.getenv("SSA_HQ32_DIAGNOSTIC_SINGLE_BF16_DS", "0") == "1"


def _sm_count(dev):
    n = _smc.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _smc[dev.index] = n
    return n


def _cv(x, al=16):
    # ``run`` is called from the manually-defined backward of
    # _SparseAttentionTrainingFunction: CuTe consumes tensor storage here and the
    # caller explicitly returns dQ/dK/dV to PyTorch.  It must therefore not
    # export PyTorch autograd metadata through DLPack.  Newer PyTorch rejects
    # such an export for any saved tensor that still has requires_grad=True,
    # while detach keeps the identical storage and strides.
    return from_dlpack(x.detach(), assumed_align=al).mark_layout_dynamic()


def _align_token_rows(x):
    """Keep the zero-copy path unless every BF16 row is 16-byte aligned."""
    if ((x.stride(0) | x.storage_offset()) & 7) == 0:
        return x
    # contiguous() may preserve an already-contiguous view with a misaligned
    # storage offset, whereas clone guarantees fresh allocator alignment.
    return x.clone(memory_format=torch.contiguous_format)


@torch.no_grad()
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
    if v.shape[0] != tk or dk.shape[0] != tk or dv.shape[0] != tk:
        raise ValueError("SSA sparse backward K/V gradient rows must match Tk; " f"Tk={tk}, Tv={v.shape[0]}, dK={dk.shape[0]}, dV={dv.shape[0]}")
    if dout.shape[0] != tq or dq.shape[0] != tq:
        raise ValueError("SSA sparse backward Q-side tensors must all have Tq rows")
    if dq.dtype != torch.float32 or dk.dtype != torch.float32 or dv.dtype != torch.float32:
        raise ValueError("SSA sparse backward follows the FP32 Q/K/V gradient contract; " f"got dQ={dq.dtype}, dK={dk.dtype}, dV={dv.dtype}")
    max_view = softmax_max.view(tq, HQ)
    inv_view = softmax_inv_sum.view(tq, HQ)
    if max_view.dtype != torch.float32 or inv_view.dtype != torch.float32 or not max_view.is_contiguous() or not inv_view.is_contiguous():
        raise ValueError("SSA sparse backward saved softmax max/inv_sum must be contiguous FP32 [Tq, Hq]")
    dev = q.device
    # Two persistent waves instead of one.  With 1 CTA/SM (173 KiB smem) the
    # resident set is still one SM-count-wide contiguous query block, so the
    # qperm=1 key-stratum coincidence that feeds L2 reduction combining is
    # preserved inside a wave; the second wave backfills SMs as lighter CTAs
    # retire, which both rebalances the causal work skew (short rows 0..510)
    # and de-phases co-resident CTAs so their RED bursts stop landing on the
    # L2 reduction pipe simultaneously.
    nctas = min(tq, 2 * _sm_count(dev))
    # Public token pitches need only cover a logical row; they are not
    # required to preserve the 16-byte row alignment used by our 128-bit
    # cp.async gathers. Normalize Q/K/V independently, without touching the
    # aligned production path exercised by the timed workloads.
    qx = _align_token_rows(q)
    kx = _align_token_rows(k)
    vx = _align_token_rows(v)
    dox = _align_token_rows(dout)
    # Persistent CTA ``b`` visits query rows (b + i*nctas) * qperm (mod Tq).
    # Any ``qperm`` coprime to Tq is a bijection over [0,Tq), so every query row
    # is still visited exactly once; the stride simply desynchronises the
    # sorted-support dK/dV scatter of concurrently resident CTAs, which
    # otherwise all hammer the same key stratum at the same time.  Choosing
    # qperm near Tq/nctas puts co-resident CTAs in different support strata.
    # Rounds 7/8 picked a stride near Tq/nctas to *spread* the concurrently
    # resident queries across key strata, because the scatter then used
    # atom.global.add, whose same-address queueing also carried a pre-image
    # response back to the SM.  With the r17 fire-and-forget
    # red.global.add.L2::cache_hint deposits into an evict-last workspace the
    # cost model inverts: two CTAs aiming at the SAME reduction address are
    # combined inside one L2 slice on an already-resident line, while spread
    # addresses touch strictly more lines.  qperm = 1 gives CTA b query b, so
    # the resident CTAs walk one contiguous causal block and their supports
    # (whose bins are s*L/512, L = q+1) coincide almost exactly.
    # Measured on B300 wl0 (Tq=Tk=4096): 1.157 ms permuted -> 1.149 ms.
    qperm = 1

    args = (
        # DLPack forbids exporting tensors with ``requires_grad=True``. These
        # detached views share the same storage and preserve the native token
        # strides; gradients are written explicitly to dq/dk/dv below.
        _cv(qx.detach()),
        _cv(kx.detach()),
        _cv(vx.detach()),
        _cv(dox.detach().reshape(-1)),
        _cv(max_view.view(-1)),
        _cv(inv_view.view(-1)),
        _cv(selected_indices.view(-1), 4),
        _cv(valid_lengths.view(-1), 4),
        _cv(dq.view(-1)),
        _cv(dk.view(-1)),
        _cv(dv.view(-1)),
        I32(qx.stride(0)),
        I32(kx.stride(0)),
        I32(vx.stride(0)),
    )

    # _cv marks tensor layouts dynamic and token pitches are explicit runtime
    # arguments.  Reusing this specialization across packed physical lengths
    # avoids synchronous late JITs when the document mix changes Tq/Tk.
    compile_key = (dev.index, _DIAGNOSTIC_SINGLE_BF16_DS)
    stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    if compile_key not in _compiled:
        _compiled[compile_key] = cute.compile(
            _launch,
            *args,
            I32(tq),
            I32(tk),
            I32(nctas),
            I32(qperm),
            _DIAGNOSTIC_SINGLE_BF16_DS,
            make_fake_stream(),
        )
    _compiled[compile_key](*args, I32(tq), I32(tk), I32(nctas), I32(qperm), stream)
