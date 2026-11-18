# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse-attention backward for B=1, Hq=16/32/96, Hkv=1 and D=256.

The Hq96 schedule uses 64 selected key/value slots per chunk, packed shared
stores, and earlier preparation of the next score contraction. Backward
recomputes QK and dP, accumulates dV, and computes dK/dQ with BF16 high and
residual dS parts. Gradient accumulation stays FP32 until conversion.

Hq16/32 retain the original schedule. Set SSA_SPARSE_BWD_MAX_ACTIVE_SMS
before the first Hq96 call to limit its persistent grid. Unset uses the
device's physical SM count; 151 selects the measured grid on a 152-SM GB300.
"""

import math
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32
I64 = cutlass.Int64

DIM = 256  # head dim (qk == v)
BR = 64  # support slots per chunk
DF = 128  # feature rows per gradient tile
SUPPORTED_HQ = (16, 32, 96)
SUPPORTED_TOPK = (512, 1024, 2048)
NTHREADS = 512
NWG = NTHREADS // 128
NPRODUCER_WG = 2
NCONSUMER_WG = NWG - NPRODUCER_WG
CONSUMER_WG_BASE = NPRODUCER_WG

SCALE = 0.0625
LOG2E = float(math.log2(math.e))
SCALE_LOG2E = SCALE * LOG2E

# TMEM column layout (512 columns).
TM_A = 0  # S^T (BR, Hq), then dK^T (DF, BR)
TM_B = 128  # dP^T (BR, Hq), then dV^T (DF, BR)
TM_DQ0 = 256  # persistent dQ^T (DF, Hq), first feature tile
TM_DQ1 = 384  # dQ^T tile1

K_T1_OFF = BR * 64 * 2  # second feature tile of a (BR, DIM) K-major tile
# WG0/1 issue MMA and gather operands. WG2/3 prepare dS, scatter dK/dV,
# and store dQ.


# --------------------------------------------------------------------------
# helper kernels
# --------------------------------------------------------------------------


@dsl_user_op
def _store_packed_bf16(address, a, b, *, loc=None, ip=None):
    low = arith.bitcast(cutlass.Uint16.mlir_type, a.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    high = arith.bitcast(cutlass.Uint16.mlir_type, b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), low, high],
        "{ .reg .b32 word; mov.b32 word, {$1, $2}; st.shared.b32 [$0], word; }",
        "r,h,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _zero_kernel(mW: cute.Tensor, n: I32, nctas: I32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 8
    if bidx == 0 and tidx == 0:
        counter = cute.make_tensor(
            cute.make_ptr(I32, mW.iterator.toint() + I64(n) * I64(4), cute.AddressSpace.gmem, assumed_align=4),
            cute.make_layout(1),
        )
        counter[0] = nctas
    if base + 8 <= n:
        fa = cute.make_rmem_tensor(cute.make_layout(8), F32)
        fa.fill(0.0)
        cute.autovec_copy(fa, cute.make_tensor(mW.iterator + base, cute.make_layout(8)))


@cute.kernel
def _cvt_kernel(mW: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, n: I32):
    """n = T*DIM ; thread converts 4 (dK,dV) pairs."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 4
    if base + 4 <= n:
        f = cute.make_rmem_tensor(cute.make_layout(8), F32)
        ok = cute.make_rmem_tensor(cute.make_layout(4), BF16)
        ov = cute.make_rmem_tensor(cute.make_layout(4), BF16)
        cute.autovec_copy(cute.make_tensor(mW.iterator + base * 2, cute.make_layout(8)), f)
        for i in cutlass.range_constexpr(4):
            ok[i] = f[2 * i].to(BF16)
            ov[i] = f[2 * i + 1].to(BF16)
        cute.autovec_copy(ok, cute.make_tensor(mDK.iterator + base, cute.make_layout(4)))
        cute.autovec_copy(ov, cute.make_tensor(mDV.iterator + base, cute.make_layout(4)))


# --------------------------------------------------------------------------
# main kernel
# --------------------------------------------------------------------------
@cute.jit
def _stage_do_half(sDO: cute.Tensor, dobase: I64, qbase: I32, hq: cutlass.Constexpr, half: cutlass.Constexpr):
    """Producer-only synchronous vector loads; caller fences and CTA-syncs."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), BF16, num_bits_per_copy=128)
    if tidx < NPRODUCER_WG * 128 and lane < 16:
        for i in cutlass.range_constexpr(hq // NPRODUCER_WG // 4):
            h = i * (4 * NPRODUCER_WG) + warp_idx
            offset = I64(qbase) + I64(h * DIM + half * 128 + lane * 8)
            reg = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
            cute.copy(
                atom,
                cute.make_tensor(
                    cute.make_ptr(BF16, dobase + offset * I64(2), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((1, 8)),
                ),
                reg,
            )
            cute.autovec_copy(reg, cute.local_tile(sDO, (1, 8), (h, lane)))


@dsl_user_op
def _early_score_producer_barrier(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "bar.sync 2,256;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _bwd_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mDO: cute.Tensor,
    mLse: cute.Tensor,
    mIdx: cute.Tensor,
    mVlen: cute.Tensor,
    mDQ: cute.Tensor,
    mW: cute.Tensor,
    KQ_mma: cute.TiledMma,
    dSQ_mma: cute.TiledMma,
    dSK_mma: cute.TiledMma,
    lQ: cute.ComposedLayout,
    lK: cute.ComposedLayout,
    lDS: cute.ComposedLayout,
    lQT: cute.ComposedLayout,
    lKT: cute.ComposedLayout,
    lDSmn: cute.ComposedLayout,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lDSs: cute.ComposedLayout,
    lDO: cute.ComposedLayout,
    lDOs: cute.ComposedLayout,
    q_token_stride: I32,
    k_token_stride: I32,
    v_token_stride: I32,
    T: I32,
    TK: I32,
    nctas: I32,
    hq: cutlass.Constexpr,
    full_hq: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    head_offset: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    wg = tidx // 128
    t128 = tidx % 128

    # Redistribute the CTA's 64K-register file from the two producer WGs to the
    # two consumer WGs.  2*80*128 + 2*176*128 == 65536 registers.
    if wg < NPRODUCER_WG:
        cute.arch.setmaxregister_decrease(80)
    else:
        cute.arch.setmaxregister_increase(176)

    ds_elems = BR * hq
    q_t1_off = hq * 64 * 2
    consumer_head_tiles = hq // 8 // NCONSUMER_WG
    score_head_tiles = hq // 16 // NCONSUMER_WG
    half_do = False
    do_t1_off = 0 if half_do else q_t1_off

    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sDO = smem.allocate_tensor(BF16, lDOs.outer, byte_alignment=1024, swizzle=lDOs.inner)
    sK = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    if cutlass.const_expr(hq == 96):
        sVraw = smem.allocate_tensor(BF16, cute.make_layout(3 * ds_elems), byte_alignment=1024)
        sV = cute.make_tensor(cute.recast_ptr(sVraw.iterator, lKs.inner), lKs.outer)
    else:
        sV = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    # Keep input V separate from the shared gradient operands.
    sVdata = smem.allocate_tensor(BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sIdx = smem.allocate_tensor(I32, cute.make_layout(topk), byte_alignment=16)
    sNextQuery = smem.allocate_tensor(I32, cute.make_layout(1), byte_alignment=4)
    counter = cute.make_ptr(
        I32,
        mW.iterator.toint() + I64(TK) * I64(DIM * 2 * 4),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    sLse = smem.allocate_tensor(F32, cute.make_layout(hq), byte_alignment=16)
    sDel = smem.allocate_tensor(F32, cute.make_layout(hq), byte_alignment=16)
    # Completion barriers for S^T, dK/dV-lo, dK/dV-hi, dP^T, and dQ.
    # Keeping S^T and dP^T separate lets the consumers start exp2 as soon as
    # the score tile is ready.  dQ has its own completion point so the next K
    # tile can be gathered while the consumers scatter dK/dV-lo.
    # H96 has an additional completion point before overwriting dO-high.
    mbar = smem.allocate_array(cutlass.Int64, 6, byte_alignment=16)

    # dS / P / dS-residual alias the V staging buffer (V is dead once dP^T
    # has been computed). H96 needs 3*128*96 BF16 elements, i.e. 72 KiB.
    sDS = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSs.inner), lDSs.outer)
    sP = cute.make_tensor(cute.recast_ptr(sV.iterator + ds_elems, lDSs.inner), lDSs.outer)
    sDSlo = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDSs.inner), lDSs.outer)

    sQ2 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQ.inner), lQ.outer)
    sDO2 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lDO.inner), lDO.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner), lK.outer)
    sV2 = cute.make_tensor(cute.recast_ptr(sVdata.iterator, lK.inner), lK.outer)
    sDS2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lDS.inner), lDS.outer)
    sP2 = cute.make_tensor(cute.recast_ptr(sV.iterator + ds_elems, lDS.inner), lDS.outer)
    sDSlo2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDS.inner), lDS.outer)

    sQT0 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQT.inner), lQT.outer)
    sQT1 = cute.make_tensor(cute.recast_ptr(sQ.iterator + q_t1_off, lQT.inner), lQT.outer)
    sDOT0 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lQT.inner), lQT.outer)
    sDOT1 = cute.make_tensor(cute.recast_ptr(sDO.iterator + do_t1_off, lQT.inner), lQT.outer)
    sKT0 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT.inner), lKT.outer)
    sKT1 = cute.make_tensor(cute.recast_ptr(sK.iterator + K_T1_OFF, lKT.inner), lKT.outer)
    sDSmn = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSmn.inner), lDSmn.outer)
    sDSlomn = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDSmn.inner), lDSmn.outer)

    kbase = mK.iterator.toint()
    vbase = mV.iterator.toint()
    obase = mO.iterator.toint()
    dobase = mDO.iterator.toint()
    # Byte offsets for packed Q/O/dO cross 2 GiB at Hq=96, T=64K.  Keep all
    # address arithmetic in i64 even though tensor element indices still fit
    # in i32 for the currently supported shape matrix.
    q_stride_bytes = I64(q_token_stride) * I64(2)
    k_stride_bytes = I64(k_token_stride) * I64(2)
    v_stride_bytes = I64(v_token_stride) * I64(2)
    g2s_atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF16, num_bits_per_copy=128)
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 32)), cute.make_layout((1, 8)))
    thr_g2s = tiled_g2s.get_slice(lane)
    # Each lane owns eight contiguous BF16 values (16 bytes) of O and dO.
    # Spell out the vector width because autovec_copy loses the alignment
    # information through the dynamic per-query/per-head pointer arithmetic and
    # otherwise scalarizes this access into eight LDG.E.U16 instructions.
    g2r_128 = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), BF16, num_bits_per_copy=128)

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init(mbar + 2, 1)
            cute.arch.mbarrier_init(mbar + 3, 1)
            cute.arch.mbarrier_init(mbar + 4, 1)
            cute.arch.mbarrier_init(mbar + 5, NCONSUMER_WG * 4)
            if cutlass.const_expr(half_do):
                cute.arch.mbarrier_init(mbar + 5, 1)
    cute.arch.mbarrier_init_fence()

    # ---------------- TMEM ----------------
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=NTHREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(F32)
    tmem.relinquish_alloc_permit()

    tS_f = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C((BR, hq)))
    tS = cute.make_tensor(tmem_ptr + TM_A, tS_f.layout)
    tDP = cute.make_tensor(tmem_ptr + TM_B, tS_f.layout)
    tKV_f = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C((DF, BR)))
    tKA = cute.make_tensor(tmem_ptr + TM_A, tKV_f.layout)
    tKB = cute.make_tensor(tmem_ptr + 64, tKV_f.layout)
    # The64-column halves fit beside the original low-gradient banks.
    tKAHi = cute.make_tensor(tmem_ptr + 128, tKV_f.layout)
    tKBHi = cute.make_tensor(tmem_ptr + 192, tKV_f.layout)
    tDQ_f = dSK_mma.make_fragment_C(dSK_mma.partition_shape_C((DF, hq)))
    tDQ0 = cute.make_tensor(tmem_ptr + TM_DQ0, tDQ_f.layout)
    tDQ1 = cute.make_tensor(tmem_ptr + TM_DQ1, tDQ_f.layout)

    fK = KQ_mma.make_fragment_A(sK)
    fQ = KQ_mma.make_fragment_B(sQ)
    fV = KQ_mma.make_fragment_A(sVdata)
    fDO = KQ_mma.make_fragment_B(sDO)
    fQT0 = dSQ_mma.make_fragment_A(sQT0)
    fQT1 = dSQ_mma.make_fragment_A(sQT1)
    fDOT0 = dSQ_mma.make_fragment_A(sDOT0)
    fDOT1 = dSQ_mma.make_fragment_A(sDOT1)
    fDSb = dSQ_mma.make_fragment_B(sDS)
    fDSlob = dSQ_mma.make_fragment_B(sDSlo)
    fPb = dSQ_mma.make_fragment_B(sP)
    fKT0 = dSK_mma.make_fragment_A(sKT0)
    fKT1 = dSK_mma.make_fragment_A(sKT1)
    fDSmn = dSK_mma.make_fragment_B(sDSmn)
    fDSlomn = dSK_mma.make_fragment_B(sDSlomn)

    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), F32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), F32)

    tS2 = tS[(None, None), 0, 0]
    tDP2 = tDP[(None, None), 0, 0]
    tKA2 = tKA[(None, None), 0, 0]
    tKB2 = tKB[(None, None), 0, 0]
    tKAHi2 = tKAHi[(None, None), 0, 0]
    tKBHi2 = tKBHi[(None, None), 0, 0]
    tDQ0_2 = tDQ0[(None, None), 0, 0]
    tDQ1_2 = tDQ1[(None, None), 0, 0]

    cSN = cute.make_identity_tensor((BR, hq))
    cp_s = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition(2)), F32), tS2)
    thr_s = cp_s.get_slice(t128)
    tr_S = thr_s.partition_S(tS2)
    tr_DP = thr_s.partition_S(tDP2)
    tr_cS = thr_s.partition_D(cSN)
    slot_m = cute.get(tr_cS[0], mode=[0])
    print("KEY64_SCORE_LAYOUT", tr_S.shape, tr_S.layout, tr_cS.shape, tr_cS.layout)
    for audit_thread in cutlass.range_constexpr(2):
        audit_coords = cp_s.get_slice(audit_thread * 16).partition_D(cSN)
        for audit_j in cutlass.range_constexpr(2):
            for audit_e in cutlass.range_constexpr(2):
                print(
                    "KEY64_COORD",
                    audit_thread * 16,
                    audit_j,
                    audit_e * 4,
                    audit_coords[audit_e * 4, 0, audit_j],
                )

    cKVN = cute.make_identity_tensor((DF, BR))
    cp_kv = tcgen05.make_tmem_copy(ld8, tKA2)
    thr_kv = cp_kv.get_slice(t128)
    tr_A = thr_kv.partition_S(tKA2)
    tr_B = thr_kv.partition_S(tKB2)
    tr_AHi = thr_kv.partition_S(tKAHi2)
    tr_BHi = thr_kv.partition_S(tKBHi2)
    tr_cKV = thr_kv.partition_D(cKVN)
    dim_m = cute.get(tr_cKV[0], mode=[0])

    cp_dq = tcgen05.make_tmem_copy(ld8, tDQ0_2)
    thr_dq = cp_dq.get_slice(t128)
    tr_DQ0 = thr_dq.partition_S(tDQ0_2)
    tr_DQ1 = thr_dq.partition_S(tDQ1_2)
    tr_cDQ = thr_dq.partition_D(cute.make_identity_tensor((DF, hq)))
    dq_m = cute.get(tr_cDQ[0], mode=[0])

    kvsh = tr_cKV[None, None, 0].shape
    rA = cute.make_rmem_tensor(kvsh, F32)
    rB = cute.make_rmem_tensor(kvsh, F32)
    # Each consumer streams eight fp32 dK/dV pairs per TMEM load and reduces
    # them immediately.  The two consumers own 64 keys (eight fragments) each.
    pair = cute.make_rmem_tensor(cute.make_layout(2), F32)
    # Preserve fp32 P across the independent dP^T completion wait.  Reloading
    # the BF16 sP value here would change dS rounding relative to the original
    # fused softmax/dS implementation.
    rP = cute.make_rmem_tensor(cute.make_layout((score_head_tiles, 8)), F32)

    cute.arch.barrier()

    ph = I32(0)
    base_lo = dim_m * 2
    base_hi = base_lo + 256
    nkQ = cute.size(fQ, mode=[2])
    nkDO = cute.size(fDO, mode=[2])
    dp_first_k = nkDO if half_do else 0
    dp_first_bar = 5 if half_do else 3
    nkDS = cute.size(fDSb, mode=[2])
    nkKT = cute.size(fKT0, mode=[2])

    d0 = lane * 8

    # ---- per-query head: index stage, chunk-0 gather, Q stream, delta ----
    first_vlen = mVlen[bidx]
    if wg < NPRODUCER_WG:
        for i in cutlass.range_constexpr(topk // (NPRODUCER_WG * 128)):
            idx = i * (NPRODUCER_WG * 128) + tidx
            first_index = I32(0)
            if idx < first_vlen:
                first_index = mIdx[bidx * topk + idx]
            sIdx[idx] = first_index
    cute.arch.barrier()
    if wg < NPRODUCER_WG:
        for i in cutlass.range_constexpr(BR // NPRODUCER_WG // 4):
            r = i * (4 * NPRODUCER_WG) + warp_idx
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(
                            BF16,
                            kbase + I64(sIdx[r]) * k_stride_bytes,
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
                            vbase + I64(sIdx[r]) * v_stride_bytes,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((1, DIM)),
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
            )
    qb0 = bidx * (full_hq * DIM) + head_offset * DIM
    qptr0 = mQ.iterator.toint() + I64(bidx) * q_stride_bytes + I64(head_offset) * I64(DIM * 2)
    if wg < NPRODUCER_WG:
        for i in cutlass.range_constexpr(hq // NPRODUCER_WG // 4):
            h = i * (4 * NPRODUCER_WG) + warp_idx
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(BF16, qptr0 + h * (DIM * 2), cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout((1, DIM)),
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
            )
    cute.arch.cp_async_commit_group()
    if wg < NPRODUCER_WG:
        for i in cutlass.range_constexpr(hq // NPRODUCER_WG // 4):
            h = i * (4 * NPRODUCER_WG) + warp_idx
            goff0 = qb0 + h * DIM + d0
            rdo0 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
            ro0 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
            cute.copy(
                g2r_128,
                cute.make_tensor(
                    cute.make_ptr(BF16, dobase + I64(goff0) * I64(2), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((1, 8)),
                ),
                rdo0,
            )
            cute.copy(
                g2r_128,
                cute.make_tensor(
                    cute.make_ptr(BF16, obase + I64(goff0) * I64(2), cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((1, 8)),
                ),
                ro0,
            )
            prod0 = rdo0.load().to(F32) * ro0.load().to(F32)
            acc0 = prod0.reduce(cute.ReductionOp.ADD, F32(0.0), 0)
            acc0 = cute.arch.warp_reduction_sum(acc0)
            if lane == 0:
                sDel[h] = acc0
            if cutlass.const_expr(half_do):
                if lane >= 16:
                    cute.autovec_copy(rdo0, cute.local_tile(sDO2, (1, 8), (h, lane - 16)))
            else:
                cute.autovec_copy(rdo0, cute.local_tile(sDO2, (1, 8), (h, lane)))
    if tidx < hq:
        sLse[tidx] = mLse[bidx * full_hq + head_offset + tidx] * F32(LOG2E)

    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if warp_idx == 0:
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + 0)
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkDO):
            cute.gemm(
                KQ_mma,
                tDP,
                fV[None, None, (kb + nkDO // 2) % nkDO, 0],
                fDO[None, None, (kb + nkDO // 2) % nkDO, 0],
                tDP,
            )
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + dp_first_bar)

    t = bidx
    while t < T:
        if tidx == 0:
            sNextQuery[0] = cute.arch.atomic_add(counter, I32(1), sem="relaxed", scope="gpu")
        cute.arch.barrier()
        tn = sNextQuery[0]
        qbase = t * (full_hq * DIM) + head_offset * DIM
        vlen = mVlen[t]
        nchunk = (vlen + (BR - 1)) // BR
        for c in cutlass.range(nchunk):
            cbase = c * BR
            cute.arch.mbarrier_wait(mbar + 0, ph)

            if cutlass.const_expr(half_do):
                if wg < NPRODUCER_WG:
                    cute.arch.mbarrier_wait(mbar + 5, ph)
                    _stage_do_half(sDO2, dobase, qbase, hq, 0)
                # Consumers calculate P while producers replace dO-high.

            # ---- P = softmax(S); overlaps the independently committed dP^T ----
            for j in cutlass.range_constexpr(hq // 16):
                if wg == CONSUMER_WG_BASE + j // score_head_tiles and slot_m < BR:
                    rS = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    cute.copy(cp_s, tr_S[None, None, j], rS)
                    for e in cutlass.range_constexpr(8):
                        h = cute.get(tr_cS[e, 0, j], mode=[1])
                        sc = rS[e] * F32(SCALE_LOG2E) - sLse[h]
                        p = F32(0.0)
                        if (cbase + cute.get(tr_cS[e, 0, j], mode=[0])) < vlen:
                            p = cute.math.exp2(sc, fastmath=True)
                        rP[j % score_head_tiles, e] = p
                pass
            cute.arch.fence_view_async_tmem_load()

            if cutlass.const_expr(half_do):
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkDO):
                        cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 3)

            cute.arch.mbarrier_wait(mbar + 3, ph)

            # ---- prefetch chunk c+1 V (K was started after dQ completed) ----
            if c + 1 < nchunk:
                nbase = cbase + BR
                if wg < NPRODUCER_WG:
                    for i in cutlass.range_constexpr(BR // NPRODUCER_WG // 4):
                        r = i * (4 * NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        BF16,
                                        vbase + I64(sIdx[nbase + r]) * v_stride_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                        )
                cute.arch.cp_async_commit_group()

            # ---- dS = P * (dP - delta) ----
            for j in cutlass.range_constexpr(hq // 16):
                if wg == CONSUMER_WG_BASE + j // score_head_tiles and slot_m < BR:
                    rDP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, F32)
                    cute.copy(cp_s, tr_DP[None, None, j], rDP)
                    oP = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    oS = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    oSlo = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                    for e in cutlass.range_constexpr(8):
                        h = cute.get(tr_cS[e, 0, j], mode=[1])
                        ds = rP[j % score_head_tiles, e] * (rDP[e] - sDel[h]) * F32(SCALE)
                        oP[0, e] = rP[j % score_head_tiles, e].to(BF16)
                        ds_hi = ds.to(BF16)
                        oS[0, e] = ds_hi
                        oSlo[0, e] = (ds - ds_hi.to(F32)).to(BF16)
                    for pair_e in cutlass.range_constexpr(4):
                        row = cute.get(tr_cS[2 * pair_e, 0, j], mode=[0])
                        head = cute.get(tr_cS[2 * pair_e, 0, j], mode=[1])
                        linear = (row % I32(8)) * I32(32) + (row // I32(8)) * I32(256) + (head % I32(32)) + (head // I32(32)) * I32(2048)
                        byte = I32(2) * linear
                        physical = byte ^ ((byte >> I32(3)) & I32(48))
                        raw = sVraw.iterator.toint()
                        _store_packed_bf16(
                            raw + I32(2 * ds_elems) + physical,
                            oP[0, 2 * pair_e],
                            oP[0, 2 * pair_e + 1],
                        )
                        _store_packed_bf16(raw + physical, oS[0, 2 * pair_e], oS[0, 2 * pair_e + 1])
                        _store_packed_bf16(
                            raw + I32(4 * ds_elems) + physical,
                            oSlo[0, 2 * pair_e],
                            oSlo[0, 2 * pair_e + 1],
                        )
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()

            # ---- dK^T / dV^T for feature dims [0,128) ----
            if warp_idx == 0:
                dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT0[None, None, kb, 0], fDSb[None, None, kb, 0], tKA)
                    cute.gemm(dSQ_mma, tKB, fDOT0[None, None, kb, 0], fPb[None, None, kb, 0], tKB)
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT0[None, None, kb, 0], fDSlob[None, None, kb, 0], tKA)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 1)

                dSK_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                for kb in cutlass.range_constexpr(nkKT):
                    cute.gemm(dSK_mma, tDQ0, fKT0[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ0)
                    cute.gemm(dSK_mma, tDQ1, fKT1[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ1)
                    dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkKT):
                    cute.gemm(dSK_mma, tDQ0, fKT0[None, None, kb, 0], fDSlomn[None, None, kb, 0], tDQ0)
                    cute.gemm(dSK_mma, tDQ1, fKT1[None, None, kb, 0], fDSlomn[None, None, kb, 0], tDQ1)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 4)
                dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKAHi, fQT1[None, None, kb, 0], fDSb[None, None, kb, 0], tKAHi)
                    cute.gemm(dSQ_mma, tKBHi, fDOT1[None, None, kb, 0], fPb[None, None, kb, 0], tKBHi)
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKAHi, fQT1[None, None, kb, 0], fDSlob[None, None, kb, 0], tKAHi)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 2)

            # High gradients have independent banks; low scatter no longer
            # gates high MMA issue or requires retaining a full register tile.
            if wg < NPRODUCER_WG:
                cute.arch.mbarrier_wait(mbar + 4, ph)
                if c + 1 < nchunk:
                    nbase = cbase + BR
                    for i in cutlass.range_constexpr(BR // NPRODUCER_WG // 4):
                        r = i * (4 * NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        BF16,
                                        kbase + I64(sIdx[nbase + r]) * k_stride_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sK2, (1, DIM), (r, 0))),
                        )
                    cute.arch.cp_async_commit_group()
                if c + 1 < nchunk:
                    # Current dQ has retired K; all producer copies target next K/V.
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    _early_score_producer_barrier()
                    # Eight consumer warps independently retire low TMEM reads.
                    cute.arch.mbarrier_wait(mbar + 5, ph)
                    if warp_idx == 0:
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(nkQ):
                            cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar + 0)
            else:
                cute.arch.mbarrier_wait(mbar + 1, ph)
                for j in cutlass.range_constexpr(NCONSUMER_WG):
                    if wg == CONSUMER_WG_BASE + j:
                        for frag in cutlass.range(4, unroll=1):
                            cute.copy(cp_kv, tr_A[None, None, j * 4 + frag], rA)
                            cute.copy(cp_kv, tr_B[None, None, j * 4 + frag], rB)
                            for e in cutlass.range_constexpr(8):
                                pair[0] = rA[e]
                                pair[1] = rB[e]
                                cute.arch.atomic_add(
                                    mW.iterator + (sIdx[cbase + j * 32 + frag * 8 + e] * I32(DIM * 2) + base_lo),
                                    pair.load(),
                                    scope="gpu",
                                )
                    pass

                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar + 5)

            cute.arch.mbarrier_wait(mbar + 2, ph)

            # On the final chunk, Q/dO from the current query are dead after
            # mbar[2].  Fill that producer bubble with the next query's Q
            # stream and delta while the consumers scatter dK/dV-hi.
            if c + 1 == nchunk and tn < T:
                qb1 = tn * (full_hq * DIM) + head_offset * DIM
                qptr1 = mQ.iterator.toint() + I64(tn) * q_stride_bytes + I64(head_offset) * I64(DIM * 2)
                if wg < NPRODUCER_WG:
                    for i in cutlass.range_constexpr(hq // NPRODUCER_WG // 4):
                        h = i * (4 * NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        BF16,
                                        qptr1 + h * (DIM * 2),
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sQ2, (1, DIM), (h, 0))),
                        )
                    cute.arch.cp_async_commit_group()
                    for i in cutlass.range_constexpr(hq // NPRODUCER_WG // 4):
                        h = i * (4 * NPRODUCER_WG) + warp_idx
                        goff1 = qb1 + h * DIM + d0
                        rdo1 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                        ro1 = cute.make_rmem_tensor(cute.make_layout((1, 8)), BF16)
                        cute.copy(
                            g2r_128,
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    dobase + I64(goff1) * I64(2),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, 8)),
                            ),
                            rdo1,
                        )
                        cute.copy(
                            g2r_128,
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    obase + I64(goff1) * I64(2),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, 8)),
                            ),
                            ro1,
                        )
                        prod1 = rdo1.load().to(F32) * ro1.load().to(F32)
                        acc1 = prod1.reduce(cute.ReductionOp.ADD, F32(0.0), 0)
                        acc1 = cute.arch.warp_reduction_sum(acc1)
                        if lane == 0:
                            sDel[h] = acc1
                        if cutlass.const_expr(half_do):
                            if lane >= 16:
                                cute.autovec_copy(rdo1, cute.local_tile(sDO2, (1, 8), (h, lane - 16)))
                        else:
                            cute.autovec_copy(rdo1, cute.local_tile(sDO2, (1, 8), (h, lane)))
                if tidx < hq:
                    sLse[tidx] = mLse[tn * full_hq + head_offset + tidx] * F32(LOG2E)

            # ---- interleaved (dK,dV) scatter, dims [128,256) ----
            for j in cutlass.range_constexpr(NCONSUMER_WG):
                if wg == CONSUMER_WG_BASE + j:
                    for frag in cutlass.range(4, unroll=1):
                        cute.copy(cp_kv, tr_AHi[None, None, j * 4 + frag], rA)
                        cute.copy(cp_kv, tr_BHi[None, None, j * 4 + frag], rB)
                        for e in cutlass.range_constexpr(8):
                            pair[0] = rA[e]
                            pair[1] = rB[e]
                            cute.arch.atomic_add(
                                mW.iterator + (sIdx[cbase + j * 32 + frag * 8 + e] * I32(DIM * 2) + base_hi),
                                pair.load(),
                                scope="gpu",
                            )
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()

            if c + 1 < nchunk:
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDO):
                        cute.gemm(
                            KQ_mma,
                            tDP,
                            fV[None, None, (kb + nkDO // 2) % nkDO, 0],
                            fDO[None, None, (kb + nkDO // 2) % nkDO, 0],
                            tDP,
                        )
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + dp_first_bar)

            ph = ph ^ I32(1)

        if tn < T:
            # The preceding dK/dV-hi barrier already made sIdx reusable.
            # Stage the next query's indices and first K/V chunk; Q, dO,
            # delta, and LSE were prefetched during the final scatter above.
            next_vlen = mVlen[tn]
            if wg < NPRODUCER_WG:
                for i in cutlass.range_constexpr(topk // (NPRODUCER_WG * 128)):
                    idx = i * (NPRODUCER_WG * 128) + tidx
                    next_index = I32(0)
                    if idx < next_vlen:
                        next_index = mIdx[tn * topk + idx]
                    sIdx[idx] = next_index
            cute.arch.barrier()
            if wg < NPRODUCER_WG:
                for i in cutlass.range_constexpr(BR // NPRODUCER_WG // 4):
                    r = i * (4 * NPRODUCER_WG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    BF16,
                                    kbase + I64(sIdx[r]) * k_stride_bytes,
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
                                    vbase + I64(sIdx[r]) * v_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sV2, (1, DIM), (r, 0))),
                    )
            cute.arch.cp_async_commit_group()

        # ---------------- dQ epilogue ----------------
        # mbar[4] is independent of the two dK/dV result barriers.  Explicitly
        # consume the final dQ phase before loading its persistent TMEM tiles.
        cute.arch.mbarrier_wait(mbar + 4, ph ^ I32(1))
        for j in cutlass.range_constexpr(hq // 8):
            if wg == CONSUMER_WG_BASE + j // consumer_head_tiles:
                r0 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                r1 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, F32)
                cute.copy(cp_dq, tr_DQ0[None, None, j], r0)
                cute.copy(cp_dq, tr_DQ1[None, None, j], r1)
                for e in cutlass.range_constexpr(8):
                    h = j * 8 + e
                    mDQ[qbase + h * DIM + dq_m] = r0[e].to(BF16)
                    mDQ[qbase + h * DIM + dq_m + 128] = r1[e].to(BF16)
            pass
        cute.arch.fence_view_async_tmem_load()
        if tn < T:

            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 0)
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDO):
                    cute.gemm(
                        KQ_mma,
                        tDP,
                        fV[None, None, (kb + nkDO // 2) % nkDO, 0],
                        fDO[None, None, (kb + nkDO // 2) % nkDO, 0],
                        tDP,
                    )
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + dp_first_bar)

        t = tn

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
    mO,
    mDO,
    mLse,
    mIdx,
    mVlen,
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
    hq: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel_hq = hq
    do_dim = DIM
    cg = tcgen05.CtaGroup.ONE
    KQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.K, OperandMajorMode.K, F32, cg, (BR, kernel_hq))
    dSQ_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.K, F32, cg, (DF, BR))
    dSK_mma = sm100_utils.make_trivial_tiled_mma(BF16, BF16, OperandMajorMode.MN, OperandMajorMode.MN, F32, cg, (DF, kernel_hq))

    lQs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, kernel_hq, DIM), BF16, 1)
    lDOs = sm100_utils.make_smem_layout_b(KQ_mma, (BR, kernel_hq, do_dim), BF16, 1)
    lKs = sm100_utils.make_smem_layout_a(KQ_mma, (BR, kernel_hq, DIM), BF16, 1)
    lDSs = sm100_utils.make_smem_layout_b(dSQ_mma, (DF, BR, kernel_hq), BF16, 1)
    lQT = sm100_utils.make_smem_layout_a(dSQ_mma, (DF, BR, kernel_hq), BF16, 1)
    lKT = sm100_utils.make_smem_layout_a(dSK_mma, (DF, kernel_hq, BR), BF16, 1)
    lDSmn = sm100_utils.make_smem_layout_b(dSK_mma, (DF, kernel_hq, BR), BF16, 1)

    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    ds_atom_kind = tcgen05.SmemLayoutAtomKind.K_SW64 if kernel_hq == 32 or kernel_hq == 96 else tcgen05.SmemLayoutAtomKind.K_SW32
    a_ds = tcgen05.make_smem_layout_atom(ds_atom_kind, BF16)
    lQ = cute.tile_to_shape(a128, (kernel_hq, DIM), order=(0, 1))
    lDO = cute.tile_to_shape(a128, (kernel_hq, do_dim), order=(0, 1))
    lK = cute.tile_to_shape(a128, (BR, DIM), order=(0, 1))
    lDS = cute.tile_to_shape(a_ds, (BR, kernel_hq), order=(0, 1))

    nelem = TK * DIM
    _zero_kernel(mW, nelem * 2, nctas).launch(grid=[cute.ceil_div(nelem * 2, 2048), 1, 1], block=[256, 1, 1], stream=stream)

    for group in cutlass.range_constexpr(hq // kernel_hq):
        _bwd_kernel(
            mQ,
            mK,
            mV,
            mO,
            mDO,
            mLse,
            mIdx,
            mVlen,
            mDQ,
            mW,
            KQ_mma,
            dSQ_mma,
            dSK_mma,
            lQ,
            lK,
            lDS,
            lQT,
            lKT,
            lDSmn,
            lQs,
            lKs,
            lDSs,
            lDO,
            lDOs,
            q_token_stride,
            k_token_stride,
            v_token_stride,
            TQ,
            TK,
            nctas,
            kernel_hq,
            hq,
            topk,
            I32(group * kernel_hq),
        ).launch(grid=[nctas, 1, 1], block=[NTHREADS, 1, 1], stream=stream)

    _cvt_kernel(mW, mDK, mDV, nelem).launch(grid=[cute.ceil_div(nelem, 1024), 1, 1], block=[256, 1, 1], stream=stream)


_compiled = {}
_smc = {}


def _sm_count(dev):
    n = _smc.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        cap = os.getenv("SSA_SPARSE_BWD_MAX_ACTIVE_SMS")
        if cap is not None:
            if not cap.isdecimal() or int(cap) <= 0:
                raise ValueError("SSA_SPARSE_BWD_MAX_ACTIVE_SMS must be a positive integer")
            n = min(n, int(cap))
        _smc[dev.index] = n
    return n


def _cv(x, al=16):
    return from_dlpack(x.detach(), assumed_align=al).mark_layout_dynamic()


@torch.no_grad()
def run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv):
    if q.ndim != 3 or q.shape[-1] != DIM:
        raise ValueError(f"q must have shape [Tq, Hq, {DIM}], got {tuple(q.shape)}")
    tq = q.shape[0]
    tk = k.shape[0]
    hq = q.shape[-2]
    topk = selected_indices.shape[-1]
    dev = q.device
    if tq <= 0 or tk <= 0:
        raise ValueError("q and k/v token streams must be nonempty")
    if hq not in SUPPORTED_HQ:
        raise ValueError(f"Hq must be one of {SUPPORTED_HQ}, got {hq}")
    if topk not in SUPPORTED_TOPK:
        raise ValueError(f"topK must be one of {SUPPORTED_TOPK}, got {topk}")
    if k.ndim != 3 or k.shape[1:] != (1, DIM):
        raise ValueError(f"k must have shape [Tk, 1, {DIM}], got {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v must match k.shape={tuple(k.shape)}, got {tuple(v.shape)}")
    if out.shape != q.shape or dout.shape != q.shape or dq.shape != q.shape:
        raise ValueError("out, dout, and dq must match q.shape")
    if dk.shape != k.shape or dv.shape != k.shape:
        raise ValueError("dk and dv must match k.shape")
    if lse.shape != q.shape[:2] or lse.dtype != torch.float32 or not lse.is_contiguous():
        raise ValueError("lse must be contiguous FP32 [Tq, Hq]")
    if selected_indices.shape != (tq, topk):
        raise ValueError("selected_indices must have shape [Tq, topK]")
    if valid_lengths.shape != (tq,):
        raise ValueError("valid_lengths must have shape [Tq]")
    bf16_tensors = (q, k, v, out, dout, dq, dk, dv)
    if any(tensor.dtype != torch.bfloat16 for tensor in bf16_tensors):
        raise ValueError("q/k/v/out/dout and dq/dk/dv must all be BF16")
    if selected_indices.dtype != torch.int32 or valid_lengths.dtype != torch.int32:
        raise ValueError("selected_indices and valid_lengths must be int32")
    tensors = (*bf16_tensors, lse, selected_indices, valid_lengths)
    if not q.is_cuda or any(tensor.device != dev for tensor in tensors[1:]):
        raise ValueError("all sparse-backward tensors must be on one CUDA device")
    if not all(tensor.is_contiguous() for tensor in (out, dout, selected_indices, valid_lengths, dq, dk, dv)):
        raise ValueError("sparse-backward outputs, state, and support metadata must be contiguous")
    if q.stride()[1:] != (DIM, 1) or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q/k/v feature rows must be contiguous")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
            raise ValueError(f"{name} base address and token stride must be 16-byte aligned")
    nctas = min(tq, _sm_count(dev))

    stream_id = torch.cuda.current_stream(dev).cuda_stream
    # _zero_kernel clears the accumulator at every launch, so it lives only for this call.
    w = torch.empty(tk * DIM * 2 + 8, dtype=torch.float32, device=dev)

    args = (
        # DLPack forbids exporting tensors with ``requires_grad=True``. These
        # detached views share the same storage and preserve the native token
        # strides; gradients are written explicitly to dq/dk/dv below.
        _cv(q),
        _cv(k),
        _cv(v),
        _cv(out.view(-1)),
        _cv(dout.view(-1)),
        _cv(lse.view(-1)),
        _cv(selected_indices.view(-1), 4),
        _cv(valid_lengths.view(-1), 4),
        _cv(dq.view(-1)),
        _cv(w),
        _cv(dk.view(-1)),
        _cv(dv.view(-1)),
        I32(q.stride(0)),
        I32(k.stride(0)),
        I32(v.stride(0)),
    )

    compile_key = (dev.index, hq, topk)
    stream = cuda.CUstream(stream_id)
    if compile_key not in _compiled:
        _compiled[compile_key] = cute.compile(_launch, *args, I32(tq), I32(tk), I32(nctas), hq, topk, make_fake_stream())
    _compiled[compile_key](*args, I32(tq), I32(tk), I32(nctas), stream)


_run_h96 = run

"""SSA sparse-attention backward (B=1, Hq=16/32/96, Hkv=1, D=256) in CuTe DSL.

The query and key/value token extents are independent so physical packed-THD
indices can address the full gathered K/V stream under context parallelism.

One CTA owns one query token `t` and a query-head group, so the MQA dK/dV
scatter is reduced over heads before it ever touches global memory.  The CTA
walks the query's selected support slots in chunks of BR=128 and runs five
tcgen05 (UMMA) GEMMs per chunk.

All supported head counts use one head group. Hq=96 streams dO through a
128-feature shared-memory stage and expands the V/dS/P alias region to
72 KiB, keeping one 96-head CTA within the SM103 shared-memory limit.

The dominant cost of this operator is the fp32 scatter-add of dK/dV: every
(query, support) pair deposits 2 x 256 fp32 values, and Blackwell's L2 caps
global atomics at a few hundred G-ops/s.  v2 therefore keeps dK and dV in a
*single interleaved* fp32 workspace

    W[key, dim, 0] = dK[key, dim]      W[key, dim, 1] = dV[key, dim]

and computes the dK and dV tiles for the *same* 128 feature dims at the same
time, so one thread owns both fp32 values and issues one vector atomic instead
of two scalar ones.  Accumulation stays fp32 until the final BF16 conversion;
rounding every contribution to FP16 before a heavily contended atomic changes
the mathematical backward and produces data-dependent gradient outliers.

The dK contraction consumes dS as a BF16 high part plus a BF16 residual.  A
single BF16 dS tile loses enough information to move large real-model dK values
by one BF16 bin.  The two-part representation retains BF16's full exponent
range and avoids an input-distribution-dependent FP16 conversion.
"""
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

_stock_BF16 = cutlass.BFloat16
_stock_F32 = cutlass.Float32
_stock_I32 = cutlass.Int32
_stock_I64 = cutlass.Int64
_stock_DIM = 256
_stock_BR = 128
_stock_SUPPORTED_HQ = (16, 32, 96)
_stock_SUPPORTED_TOPK = (512, 1024, 2048)
_stock_NTHREADS = 512
_stock_NWG = _stock_NTHREADS // 128
_stock_NPRODUCER_WG = 2
_stock_NCONSUMER_WG = _stock_NWG - _stock_NPRODUCER_WG
_stock_CONSUMER_WG_BASE = _stock_NPRODUCER_WG
_stock_SCALE = 0.0625
_stock_LOG2E = float(math.log2(math.e))
_stock_SCALE_LOG2E = _stock_SCALE * _stock_LOG2E
_stock_TM_A = 0
_stock_TM_B = 128
_stock_TM_DQ0 = 256
_stock_TM_DQ1 = 384
_stock_K_T1_OFF = _stock_BR * 64 * 2


@cute.kernel
def _stock__zero_kernel(mW: cute.Tensor, n: _stock_I32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 8
    if base + 8 <= n:
        fa = cute.make_rmem_tensor(cute.make_layout(8), _stock_F32)
        fa.fill(0.0)
        cute.autovec_copy(fa, cute.make_tensor(mW.iterator + base, cute.make_layout(8)))


@cute.kernel
def _stock__cvt_kernel(mW: cute.Tensor, mDK: cute.Tensor, mDV: cute.Tensor, n: _stock_I32):
    """n = T*DIM ; thread converts 4 (dK,dV) pairs."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    base = (bidx * 256 + tidx) * 4
    if base + 4 <= n:
        f = cute.make_rmem_tensor(cute.make_layout(8), _stock_F32)
        ok = cute.make_rmem_tensor(cute.make_layout(4), _stock_BF16)
        ov = cute.make_rmem_tensor(cute.make_layout(4), _stock_BF16)
        cute.autovec_copy(cute.make_tensor(mW.iterator + base * 2, cute.make_layout(8)), f)
        for i in cutlass.range_constexpr(4):
            ok[i] = f[2 * i].to(_stock_BF16)
            ov[i] = f[2 * i + 1].to(_stock_BF16)
        cute.autovec_copy(ok, cute.make_tensor(mDK.iterator + base, cute.make_layout(4)))
        cute.autovec_copy(ov, cute.make_tensor(mDV.iterator + base, cute.make_layout(4)))


@cute.jit
def _stock__stage_do_half(
    sDO: cute.Tensor,
    dobase: _stock_I64,
    qbase: _stock_I32,
    hq: cutlass.Constexpr,
    half: cutlass.Constexpr,
):
    """Producer-only synchronous vector loads; caller fences and CTA-syncs."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    atom = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), _stock_BF16, num_bits_per_copy=128)
    if tidx < _stock_NPRODUCER_WG * 128 and lane < 16:
        for i in cutlass.range_constexpr(hq // _stock_NPRODUCER_WG // 4):
            h = i * (4 * _stock_NPRODUCER_WG) + warp_idx
            offset = _stock_I64(qbase) + _stock_I64(h * _stock_DIM + half * 128 + lane * 8)
            reg = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
            cute.copy(
                atom,
                cute.make_tensor(
                    cute.make_ptr(
                        _stock_BF16,
                        dobase + offset * _stock_I64(2),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((1, 8)),
                ),
                reg,
            )
            cute.autovec_copy(reg, cute.local_tile(sDO, (1, 8), (h, lane)))


@cute.kernel
def _stock__bwd_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mDO: cute.Tensor,
    mLse: cute.Tensor,
    mIdx: cute.Tensor,
    mVlen: cute.Tensor,
    mDQ: cute.Tensor,
    mW: cute.Tensor,
    KQ_mma: cute.TiledMma,
    dSQ_mma: cute.TiledMma,
    dSK_mma: cute.TiledMma,
    lQ: cute.ComposedLayout,
    lK: cute.ComposedLayout,
    lDS: cute.ComposedLayout,
    lQT: cute.ComposedLayout,
    lKT: cute.ComposedLayout,
    lDSmn: cute.ComposedLayout,
    lQs: cute.ComposedLayout,
    lKs: cute.ComposedLayout,
    lDSs: cute.ComposedLayout,
    lDO: cute.ComposedLayout,
    lDOs: cute.ComposedLayout,
    q_token_stride: _stock_I32,
    k_token_stride: _stock_I32,
    v_token_stride: _stock_I32,
    T: _stock_I32,
    nctas: _stock_I32,
    hq: cutlass.Constexpr,
    full_hq: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    head_offset: _stock_I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    wg = tidx // 128
    t128 = tidx % 128
    if wg < _stock_NPRODUCER_WG:
        cute.arch.setmaxregister_decrease(80)
    else:
        cute.arch.setmaxregister_increase(176)
    ds_elems = _stock_BR * hq
    q_t1_off = hq * 64 * 2
    consumer_head_tiles = hq // 8 // _stock_NCONSUMER_WG
    half_do = hq == 96
    do_t1_off = 0 if half_do else q_t1_off
    smem = utils.SmemAllocator()
    sQ = smem.allocate_tensor(_stock_BF16, lQs.outer, byte_alignment=1024, swizzle=lQs.inner)
    sDO = smem.allocate_tensor(_stock_BF16, lDOs.outer, byte_alignment=1024, swizzle=lDOs.inner)
    sK = smem.allocate_tensor(_stock_BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    if cutlass.const_expr(half_do):
        sVraw = smem.allocate_tensor(_stock_BF16, cute.make_layout(3 * ds_elems), byte_alignment=1024)
        sV = cute.make_tensor(cute.recast_ptr(sVraw.iterator, lKs.inner), lKs.outer)
    else:
        sV = smem.allocate_tensor(_stock_BF16, lKs.outer, byte_alignment=1024, swizzle=lKs.inner)
    sIdx = smem.allocate_tensor(_stock_I32, cute.make_layout(topk), byte_alignment=16)
    sLse = smem.allocate_tensor(_stock_F32, cute.make_layout(hq), byte_alignment=16)
    sDel = smem.allocate_tensor(_stock_F32, cute.make_layout(hq), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 6 if half_do else 5, byte_alignment=16)
    sDS = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSs.inner), lDSs.outer)
    sP = cute.make_tensor(cute.recast_ptr(sV.iterator + ds_elems, lDSs.inner), lDSs.outer)
    sDSlo = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDSs.inner), lDSs.outer)
    sQ2 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQ.inner), lQ.outer)
    sDO2 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lDO.inner), lDO.outer)
    sK2 = cute.make_tensor(cute.recast_ptr(sK.iterator, lK.inner), lK.outer)
    sV2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lK.inner), lK.outer)
    sDS2 = cute.make_tensor(cute.recast_ptr(sV.iterator, lDS.inner), lDS.outer)
    sP2 = cute.make_tensor(cute.recast_ptr(sV.iterator + ds_elems, lDS.inner), lDS.outer)
    sDSlo2 = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDS.inner), lDS.outer)
    sQT0 = cute.make_tensor(cute.recast_ptr(sQ.iterator, lQT.inner), lQT.outer)
    sQT1 = cute.make_tensor(cute.recast_ptr(sQ.iterator + q_t1_off, lQT.inner), lQT.outer)
    sDOT0 = cute.make_tensor(cute.recast_ptr(sDO.iterator, lQT.inner), lQT.outer)
    sDOT1 = cute.make_tensor(cute.recast_ptr(sDO.iterator + do_t1_off, lQT.inner), lQT.outer)
    sKT0 = cute.make_tensor(cute.recast_ptr(sK.iterator, lKT.inner), lKT.outer)
    sKT1 = cute.make_tensor(cute.recast_ptr(sK.iterator + _stock_K_T1_OFF, lKT.inner), lKT.outer)
    sDSmn = cute.make_tensor(cute.recast_ptr(sV.iterator, lDSmn.inner), lDSmn.outer)
    sDSlomn = cute.make_tensor(cute.recast_ptr(sV.iterator + 2 * ds_elems, lDSmn.inner), lDSmn.outer)
    kbase = mK.iterator.toint()
    vbase = mV.iterator.toint()
    obase = mO.iterator.toint()
    dobase = mDO.iterator.toint()
    q_stride_bytes = _stock_I64(q_token_stride) * _stock_I64(2)
    k_stride_bytes = _stock_I64(k_token_stride) * _stock_I64(2)
    v_stride_bytes = _stock_I64(v_token_stride) * _stock_I64(2)
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        _stock_BF16,
        num_bits_per_copy=128,
    )
    tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout((1, 32)), cute.make_layout((1, 8)))
    thr_g2s = tiled_g2s.get_slice(lane)
    g2r_128 = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), _stock_BF16, num_bits_per_copy=128)
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
            cute.arch.mbarrier_init(mbar + 2, 1)
            cute.arch.mbarrier_init(mbar + 3, 1)
            cute.arch.mbarrier_init(mbar + 4, 1)
            if cutlass.const_expr(half_do):
                cute.arch.mbarrier_init(mbar + 5, 1)
    cute.arch.mbarrier_init_fence()
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=_stock_NTHREADS)
    tmem = utils.TmemAllocator(barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(_stock_F32)
    tmem.relinquish_alloc_permit()
    tS_f = KQ_mma.make_fragment_C(KQ_mma.partition_shape_C((_stock_BR, hq)))
    tS = cute.make_tensor(tmem_ptr + _stock_TM_A, tS_f.layout)
    tDP = cute.make_tensor(tmem_ptr + _stock_TM_B, tS_f.layout)
    tKV_f = dSQ_mma.make_fragment_C(dSQ_mma.partition_shape_C((_stock_BR, _stock_BR)))
    tKA = cute.make_tensor(tmem_ptr + _stock_TM_A, tKV_f.layout)
    tKB = cute.make_tensor(tmem_ptr + _stock_TM_B, tKV_f.layout)
    tDQ_f = dSK_mma.make_fragment_C(dSK_mma.partition_shape_C((_stock_BR, hq)))
    tDQ0 = cute.make_tensor(tmem_ptr + _stock_TM_DQ0, tDQ_f.layout)
    tDQ1 = cute.make_tensor(tmem_ptr + _stock_TM_DQ1, tDQ_f.layout)
    fK = KQ_mma.make_fragment_A(sK)
    fQ = KQ_mma.make_fragment_B(sQ)
    fV = KQ_mma.make_fragment_A(sV)
    fDO = KQ_mma.make_fragment_B(sDO)
    fQT0 = dSQ_mma.make_fragment_A(sQT0)
    fQT1 = dSQ_mma.make_fragment_A(sQT1)
    fDOT0 = dSQ_mma.make_fragment_A(sDOT0)
    fDOT1 = dSQ_mma.make_fragment_A(sDOT1)
    fDSb = dSQ_mma.make_fragment_B(sDS)
    fDSlob = dSQ_mma.make_fragment_B(sDSlo)
    fPb = dSQ_mma.make_fragment_B(sP)
    fKT0 = dSK_mma.make_fragment_A(sKT0)
    fKT1 = dSK_mma.make_fragment_A(sKT1)
    fDSmn = dSK_mma.make_fragment_B(sDSmn)
    fDSlomn = dSK_mma.make_fragment_B(sDSlomn)
    ld8 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(8)), _stock_F32)
    ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), _stock_F32)
    tS2 = tS[(None, None), 0, 0]
    tDP2 = tDP[(None, None), 0, 0]
    tKA2 = tKA[(None, None), 0, 0]
    tKB2 = tKB[(None, None), 0, 0]
    tDQ0_2 = tDQ0[(None, None), 0, 0]
    tDQ1_2 = tDQ1[(None, None), 0, 0]
    cSN = cute.make_identity_tensor((_stock_BR, hq))
    cp_s = tcgen05.make_tmem_copy(ld8, tS2)
    thr_s = cp_s.get_slice(t128)
    tr_S = thr_s.partition_S(tS2)
    tr_DP = thr_s.partition_S(tDP2)
    tr_cS = thr_s.partition_D(cSN)
    slot_m = cute.get(tr_cS[0], mode=[0])
    cKVN = cute.make_identity_tensor((_stock_BR, _stock_BR))
    cp_kv = tcgen05.make_tmem_copy(ld8, tKA2)
    thr_kv = cp_kv.get_slice(t128)
    tr_A = thr_kv.partition_S(tKA2)
    tr_B = thr_kv.partition_S(tKB2)
    tr_cKV = thr_kv.partition_D(cKVN)
    dim_m = cute.get(tr_cKV[0], mode=[0])
    cp_dq = tcgen05.make_tmem_copy(ld8, tDQ0_2)
    thr_dq = cp_dq.get_slice(t128)
    tr_DQ0 = thr_dq.partition_S(tDQ0_2)
    tr_DQ1 = thr_dq.partition_S(tDQ1_2)
    tr_cDQ = thr_dq.partition_D(cSN)
    dq_m = cute.get(tr_cDQ[0], mode=[0])
    kvsh = tr_cKV[None, None, 0].shape
    rA = cute.make_rmem_tensor(kvsh, _stock_F32)
    rB = cute.make_rmem_tensor(kvsh, _stock_F32)
    pair = cute.make_rmem_tensor(cute.make_layout(2), _stock_F32)
    rP = cute.make_rmem_tensor(cute.make_layout((consumer_head_tiles, 8)), _stock_F32)
    cute.arch.barrier()
    ph = _stock_I32(0)
    base_lo = dim_m * 2
    base_hi = base_lo + 256
    nkQ = cute.size(fQ, mode=[2])
    nkDO = cute.size(fDO, mode=[2])
    dp_first_k = nkDO if half_do else 0
    dp_first_bar = 5 if half_do else 3
    nkDS = cute.size(fDSb, mode=[2])
    nkKT = cute.size(fKT0, mode=[2])
    d0 = lane * 8
    first_vlen = mVlen[bidx]
    if wg < _stock_NPRODUCER_WG:
        for i in cutlass.range_constexpr(topk // (_stock_NPRODUCER_WG * 128)):
            idx = i * (_stock_NPRODUCER_WG * 128) + tidx
            first_index = _stock_I32(0)
            if idx < first_vlen:
                first_index = mIdx[bidx * topk + idx]
            sIdx[idx] = first_index
    cute.arch.barrier()
    if wg < _stock_NPRODUCER_WG:
        for i in cutlass.range_constexpr(_stock_BR // _stock_NPRODUCER_WG // 4):
            r = i * (4 * _stock_NPRODUCER_WG) + warp_idx
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(
                            _stock_BF16,
                            kbase + _stock_I64(sIdx[r]) * k_stride_bytes,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((1, _stock_DIM)),
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sK2, (1, _stock_DIM), (r, 0))),
            )
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(
                            _stock_BF16,
                            vbase + _stock_I64(sIdx[r]) * v_stride_bytes,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((1, _stock_DIM)),
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sV2, (1, _stock_DIM), (r, 0))),
            )
    qb0 = bidx * (full_hq * _stock_DIM) + head_offset * _stock_DIM
    qptr0 = mQ.iterator.toint() + _stock_I64(bidx) * q_stride_bytes + _stock_I64(head_offset) * _stock_I64(_stock_DIM * 2)
    if wg < _stock_NPRODUCER_WG:
        for i in cutlass.range_constexpr(hq // _stock_NPRODUCER_WG // 4):
            h = i * (4 * _stock_NPRODUCER_WG) + warp_idx
            cute.copy(
                tiled_g2s,
                thr_g2s.partition_S(
                    cute.make_tensor(
                        cute.make_ptr(
                            _stock_BF16,
                            qptr0 + h * (_stock_DIM * 2),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((1, _stock_DIM)),
                    )
                ),
                thr_g2s.partition_D(cute.local_tile(sQ2, (1, _stock_DIM), (h, 0))),
            )
    cute.arch.cp_async_commit_group()
    if wg < _stock_NPRODUCER_WG:
        for i in cutlass.range_constexpr(hq // _stock_NPRODUCER_WG // 4):
            h = i * (4 * _stock_NPRODUCER_WG) + warp_idx
            goff0 = qb0 + h * _stock_DIM + d0
            rdo0 = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
            ro0 = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
            cute.copy(
                g2r_128,
                cute.make_tensor(
                    cute.make_ptr(
                        _stock_BF16,
                        dobase + _stock_I64(goff0) * _stock_I64(2),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((1, 8)),
                ),
                rdo0,
            )
            cute.copy(
                g2r_128,
                cute.make_tensor(
                    cute.make_ptr(
                        _stock_BF16,
                        obase + _stock_I64(goff0) * _stock_I64(2),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((1, 8)),
                ),
                ro0,
            )
            prod0 = rdo0.load().to(_stock_F32) * ro0.load().to(_stock_F32)
            acc0 = prod0.reduce(cute.ReductionOp.ADD, _stock_F32(0.0), 0)
            acc0 = cute.arch.warp_reduction_sum(acc0)
            if lane == 0:
                sDel[h] = acc0
            if cutlass.const_expr(half_do):
                if lane >= 16:
                    cute.autovec_copy(rdo0, cute.local_tile(sDO2, (1, 8), (h, lane - 16)))
            else:
                cute.autovec_copy(rdo0, cute.local_tile(sDO2, (1, 8), (h, lane)))
    if tidx < hq:
        sLse[tidx] = mLse[bidx * full_hq + head_offset + tidx] * _stock_F32(_stock_LOG2E)
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier()
    if warp_idx == 0:
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkQ):
            cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + 0)
        KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkDO):
            cute.gemm(KQ_mma, tDP, fV[None, None, kb + dp_first_k, 0], fDO[None, None, kb, 0], tDP)
            KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar + dp_first_bar)
    nq = (T - bidx + nctas - _stock_I32(1)) // nctas
    for qi in cutlass.range(nq):
        t = bidx + qi * nctas
        tn = t + nctas
        qbase = t * (full_hq * _stock_DIM) + head_offset * _stock_DIM
        vlen = mVlen[t]
        nchunk = (vlen + (_stock_BR - 1)) // _stock_BR
        for c in cutlass.range(nchunk):
            cbase = c * _stock_BR
            cute.arch.mbarrier_wait(mbar + 0, ph)
            if cutlass.const_expr(half_do):
                if wg < _stock_NPRODUCER_WG:
                    cute.arch.mbarrier_wait(mbar + 5, ph)
                    _stock__stage_do_half(sDO2, dobase, qbase, hq, 0)
            for j in cutlass.range_constexpr(hq // 8):
                if wg == _stock_CONSUMER_WG_BASE + j // consumer_head_tiles:
                    rS = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, _stock_F32)
                    cute.copy(cp_s, tr_S[None, None, j], rS)
                    for e in cutlass.range_constexpr(8):
                        h = j * 8 + e
                        sc = rS[e] * _stock_F32(_stock_SCALE_LOG2E) - sLse[h]
                        p = _stock_F32(0.0)
                        if cbase + slot_m < vlen:
                            p = cute.math.exp2(sc, fastmath=True)
                        rP[j % consumer_head_tiles, e] = p
                pass
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(half_do):
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier()
                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(nkDO):
                        cute.gemm(KQ_mma, tDP, fV[None, None, kb, 0], fDO[None, None, kb, 0], tDP)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 3)
            cute.arch.mbarrier_wait(mbar + 3, ph)
            for j in cutlass.range_constexpr(hq // 8):
                if wg == _stock_CONSUMER_WG_BASE + j // consumer_head_tiles:
                    rDP = cute.make_rmem_tensor(tr_cS[None, None, 0].shape, _stock_F32)
                    cute.copy(cp_s, tr_DP[None, None, j], rDP)
                    oP = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
                    oS = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
                    oSlo = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
                    for e in cutlass.range_constexpr(8):
                        h = j * 8 + e
                        ds = rP[j % consumer_head_tiles, e] * (rDP[e] - sDel[h]) * _stock_F32(_stock_SCALE)
                        oP[0, e] = rP[j % consumer_head_tiles, e].to(_stock_BF16)
                        ds_hi = ds.to(_stock_BF16)
                        oS[0, e] = ds_hi
                        oSlo[0, e] = (ds - ds_hi.to(_stock_F32)).to(_stock_BF16)
                    cute.autovec_copy(oP, cute.local_tile(sP2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oS, cute.local_tile(sDS2, (1, 8), (slot_m, j)))
                    cute.autovec_copy(oSlo, cute.local_tile(sDSlo2, (1, 8), (slot_m, j)))
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT0[None, None, kb, 0], fDSb[None, None, kb, 0], tKA)
                    cute.gemm(dSQ_mma, tKB, fDOT0[None, None, kb, 0], fPb[None, None, kb, 0], tKB)
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT0[None, None, kb, 0], fDSlob[None, None, kb, 0], tKA)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 1)
                dSK_mma.set(tcgen05.Field.ACCUMULATE, c != 0)
                for kb in cutlass.range_constexpr(nkKT):
                    cute.gemm(dSK_mma, tDQ0, fKT0[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ0)
                    cute.gemm(dSK_mma, tDQ1, fKT1[None, None, kb, 0], fDSmn[None, None, kb, 0], tDQ1)
                    dSK_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkKT):
                    cute.gemm(dSK_mma, tDQ0, fKT0[None, None, kb, 0], fDSlomn[None, None, kb, 0], tDQ0)
                    cute.gemm(dSK_mma, tDQ1, fKT1[None, None, kb, 0], fDSlomn[None, None, kb, 0], tDQ1)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 4)
            if wg < _stock_NPRODUCER_WG:
                if cutlass.const_expr(half_do):
                    cute.arch.mbarrier_wait(mbar + 1, ph)
                    _stock__stage_do_half(sDO2, dobase, qbase, hq, 1)
                if c + 1 < nchunk:
                    cute.arch.mbarrier_wait(mbar + 4, ph)
                    nbase = cbase + _stock_BR
                    for i in cutlass.range_constexpr(_stock_BR // _stock_NPRODUCER_WG // 4):
                        r = i * (4 * _stock_NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        _stock_BF16,
                                        kbase + _stock_I64(sIdx[nbase + r]) * k_stride_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, _stock_DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sK2, (1, _stock_DIM), (r, 0))),
                        )
                    cute.arch.cp_async_commit_group()
            else:
                cute.arch.mbarrier_wait(mbar + 1, ph)
                for j in cutlass.range_constexpr(_stock_NCONSUMER_WG):
                    if wg == _stock_CONSUMER_WG_BASE + j:
                        for frag in cutlass.range(8, unroll=1):
                            cute.copy(cp_kv, tr_A[None, None, j * 8 + frag], rA)
                            cute.copy(cp_kv, tr_B[None, None, j * 8 + frag], rB)
                            for e in cutlass.range_constexpr(8):
                                pair[0] = rA[e]
                                pair[1] = rB[e]
                                cute.arch.atomic_add(
                                    mW.iterator + (sIdx[cbase + j * 64 + frag * 8 + e] * _stock_I32(_stock_DIM * 2) + base_lo),
                                    pair.load(),
                                    scope="gpu",
                                )
                    pass
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(half_do):
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                dSQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT1[None, None, kb, 0], fDSb[None, None, kb, 0], tKA)
                    cute.gemm(dSQ_mma, tKB, fDOT1[None, None, kb, 0], fPb[None, None, kb, 0], tKB)
                    dSQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(nkDS):
                    cute.gemm(dSQ_mma, tKA, fQT1[None, None, kb, 0], fDSlob[None, None, kb, 0], tKA)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 2)
            cute.arch.mbarrier_wait(mbar + 2, ph)
            if c + 1 < nchunk:
                nbase = cbase + _stock_BR
                if wg < _stock_NPRODUCER_WG:
                    for i in cutlass.range_constexpr(_stock_BR // _stock_NPRODUCER_WG // 4):
                        r = i * (4 * _stock_NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        _stock_BF16,
                                        vbase + _stock_I64(sIdx[nbase + r]) * v_stride_bytes,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, _stock_DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sV2, (1, _stock_DIM), (r, 0))),
                        )
                cute.arch.cp_async_commit_group()
            if c + 1 == nchunk and tn < T:
                qb1 = tn * (full_hq * _stock_DIM) + head_offset * _stock_DIM
                qptr1 = mQ.iterator.toint() + _stock_I64(tn) * q_stride_bytes + _stock_I64(head_offset) * _stock_I64(_stock_DIM * 2)
                if wg < _stock_NPRODUCER_WG:
                    for i in cutlass.range_constexpr(hq // _stock_NPRODUCER_WG // 4):
                        h = i * (4 * _stock_NPRODUCER_WG) + warp_idx
                        cute.copy(
                            tiled_g2s,
                            thr_g2s.partition_S(
                                cute.make_tensor(
                                    cute.make_ptr(
                                        _stock_BF16,
                                        qptr1 + h * (_stock_DIM * 2),
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    cute.make_layout((1, _stock_DIM)),
                                )
                            ),
                            thr_g2s.partition_D(cute.local_tile(sQ2, (1, _stock_DIM), (h, 0))),
                        )
                    cute.arch.cp_async_commit_group()
                    for i in cutlass.range_constexpr(hq // _stock_NPRODUCER_WG // 4):
                        h = i * (4 * _stock_NPRODUCER_WG) + warp_idx
                        goff1 = qb1 + h * _stock_DIM + d0
                        rdo1 = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
                        ro1 = cute.make_rmem_tensor(cute.make_layout((1, 8)), _stock_BF16)
                        cute.copy(
                            g2r_128,
                            cute.make_tensor(
                                cute.make_ptr(
                                    _stock_BF16,
                                    dobase + _stock_I64(goff1) * _stock_I64(2),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, 8)),
                            ),
                            rdo1,
                        )
                        cute.copy(
                            g2r_128,
                            cute.make_tensor(
                                cute.make_ptr(
                                    _stock_BF16,
                                    obase + _stock_I64(goff1) * _stock_I64(2),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, 8)),
                            ),
                            ro1,
                        )
                        prod1 = rdo1.load().to(_stock_F32) * ro1.load().to(_stock_F32)
                        acc1 = prod1.reduce(cute.ReductionOp.ADD, _stock_F32(0.0), 0)
                        acc1 = cute.arch.warp_reduction_sum(acc1)
                        if lane == 0:
                            sDel[h] = acc1
                        if cutlass.const_expr(half_do):
                            if lane >= 16:
                                cute.autovec_copy(rdo1, cute.local_tile(sDO2, (1, 8), (h, lane - 16)))
                        else:
                            cute.autovec_copy(rdo1, cute.local_tile(sDO2, (1, 8), (h, lane)))
                if tidx < hq:
                    sLse[tidx] = mLse[tn * full_hq + head_offset + tidx] * _stock_F32(_stock_LOG2E)
            for j in cutlass.range_constexpr(_stock_NCONSUMER_WG):
                if wg == _stock_CONSUMER_WG_BASE + j:
                    for frag in cutlass.range(8, unroll=1):
                        cute.copy(cp_kv, tr_A[None, None, j * 8 + frag], rA)
                        cute.copy(cp_kv, tr_B[None, None, j * 8 + frag], rB)
                        for e in cutlass.range_constexpr(8):
                            pair[0] = rA[e]
                            pair[1] = rB[e]
                            cute.arch.atomic_add(
                                mW.iterator + (sIdx[cbase + j * 64 + frag * 8 + e] * _stock_I32(_stock_DIM * 2) + base_hi),
                                pair.load(),
                                scope="gpu",
                            )
                pass
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
            if c + 1 < nchunk:
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()
                if warp_idx == 0:
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkQ):
                        cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + 0)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkDO):
                        cute.gemm(
                            KQ_mma,
                            tDP,
                            fV[None, None, kb + dp_first_k, 0],
                            fDO[None, None, kb, 0],
                            tDP,
                        )
                        KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + dp_first_bar)
            ph = ph ^ _stock_I32(1)
        if tn < T:
            next_vlen = mVlen[tn]
            if wg < _stock_NPRODUCER_WG:
                for i in cutlass.range_constexpr(topk // (_stock_NPRODUCER_WG * 128)):
                    idx = i * (_stock_NPRODUCER_WG * 128) + tidx
                    next_index = _stock_I32(0)
                    if idx < next_vlen:
                        next_index = mIdx[tn * topk + idx]
                    sIdx[idx] = next_index
            cute.arch.barrier()
            if wg < _stock_NPRODUCER_WG:
                for i in cutlass.range_constexpr(_stock_BR // _stock_NPRODUCER_WG // 4):
                    r = i * (4 * _stock_NPRODUCER_WG) + warp_idx
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    _stock_BF16,
                                    kbase + _stock_I64(sIdx[r]) * k_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, _stock_DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sK2, (1, _stock_DIM), (r, 0))),
                    )
                    cute.copy(
                        tiled_g2s,
                        thr_g2s.partition_S(
                            cute.make_tensor(
                                cute.make_ptr(
                                    _stock_BF16,
                                    vbase + _stock_I64(sIdx[r]) * v_stride_bytes,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((1, _stock_DIM)),
                            )
                        ),
                        thr_g2s.partition_D(cute.local_tile(sV2, (1, _stock_DIM), (r, 0))),
                    )
            cute.arch.cp_async_commit_group()
        cute.arch.mbarrier_wait(mbar + 4, ph ^ _stock_I32(1))
        for j in cutlass.range_constexpr(hq // 8):
            if wg == _stock_CONSUMER_WG_BASE + j // consumer_head_tiles:
                r0 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, _stock_F32)
                r1 = cute.make_rmem_tensor(tr_cDQ[None, None, 0].shape, _stock_F32)
                cute.copy(cp_dq, tr_DQ0[None, None, j], r0)
                cute.copy(cp_dq, tr_DQ1[None, None, j], r1)
                for e in cutlass.range_constexpr(8):
                    h = j * 8 + e
                    mDQ[qbase + h * _stock_DIM + dq_m] = r0[e].to(_stock_BF16)
                    mDQ[qbase + h * _stock_DIM + dq_m + 128] = r1[e].to(_stock_BF16)
            pass
        cute.arch.fence_view_async_tmem_load()
        if tn < T:
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp_idx == 0:
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkQ):
                    cute.gemm(KQ_mma, tS, fK[None, None, kb, 0], fQ[None, None, kb, 0], tS)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + 0)
                KQ_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nkDO):
                    cute.gemm(KQ_mma, tDP, fV[None, None, kb + dp_first_k, 0], fDO[None, None, kb, 0], tDP)
                    KQ_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar + dp_first_bar)
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _stock__launch(
    mQ,
    mK,
    mV,
    mO,
    mDO,
    mLse,
    mIdx,
    mVlen,
    mDQ,
    mW,
    mDK,
    mDV,
    q_token_stride: _stock_I32,
    k_token_stride: _stock_I32,
    v_token_stride: _stock_I32,
    TQ: _stock_I32,
    TK: _stock_I32,
    nctas: _stock_I32,
    hq: cutlass.Constexpr,
    topk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    kernel_hq = hq
    do_dim = 128 if hq == 96 else _stock_DIM
    cg = tcgen05.CtaGroup.ONE
    KQ_mma = sm100_utils.make_trivial_tiled_mma(
        _stock_BF16,
        _stock_BF16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        _stock_F32,
        cg,
        (_stock_BR, kernel_hq),
    )
    dSQ_mma = sm100_utils.make_trivial_tiled_mma(
        _stock_BF16,
        _stock_BF16,
        OperandMajorMode.MN,
        OperandMajorMode.K,
        _stock_F32,
        cg,
        (_stock_BR, _stock_BR),
    )
    dSK_mma = sm100_utils.make_trivial_tiled_mma(
        _stock_BF16,
        _stock_BF16,
        OperandMajorMode.MN,
        OperandMajorMode.MN,
        _stock_F32,
        cg,
        (_stock_BR, kernel_hq),
    )
    lQs = sm100_utils.make_smem_layout_b(KQ_mma, (_stock_BR, kernel_hq, _stock_DIM), _stock_BF16, 1)
    lDOs = sm100_utils.make_smem_layout_b(KQ_mma, (_stock_BR, kernel_hq, do_dim), _stock_BF16, 1)
    lKs = sm100_utils.make_smem_layout_a(KQ_mma, (_stock_BR, kernel_hq, _stock_DIM), _stock_BF16, 1)
    lDSs = sm100_utils.make_smem_layout_b(dSQ_mma, (_stock_BR, _stock_BR, kernel_hq), _stock_BF16, 1)
    lQT = sm100_utils.make_smem_layout_a(dSQ_mma, (_stock_BR, _stock_BR, kernel_hq), _stock_BF16, 1)
    lKT = sm100_utils.make_smem_layout_a(dSK_mma, (_stock_BR, kernel_hq, _stock_BR), _stock_BF16, 1)
    lDSmn = sm100_utils.make_smem_layout_b(dSK_mma, (_stock_BR, kernel_hq, _stock_BR), _stock_BF16, 1)
    a128 = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, _stock_BF16)
    ds_atom_kind = tcgen05.SmemLayoutAtomKind.K_SW64 if kernel_hq == 32 or kernel_hq == 96 else tcgen05.SmemLayoutAtomKind.K_SW32
    a_ds = tcgen05.make_smem_layout_atom(ds_atom_kind, _stock_BF16)
    lQ = cute.tile_to_shape(a128, (kernel_hq, _stock_DIM), order=(0, 1))
    lDO = cute.tile_to_shape(a128, (kernel_hq, do_dim), order=(0, 1))
    lK = cute.tile_to_shape(a128, (_stock_BR, _stock_DIM), order=(0, 1))
    lDS = cute.tile_to_shape(a_ds, (_stock_BR, kernel_hq), order=(0, 1))
    nelem = TK * _stock_DIM
    _stock__zero_kernel(mW, nelem * 2).launch(grid=[cute.ceil_div(nelem * 2, 2048), 1, 1], block=[256, 1, 1], stream=stream)
    for group in cutlass.range_constexpr(hq // kernel_hq):
        _stock__bwd_kernel(
            mQ,
            mK,
            mV,
            mO,
            mDO,
            mLse,
            mIdx,
            mVlen,
            mDQ,
            mW,
            KQ_mma,
            dSQ_mma,
            dSK_mma,
            lQ,
            lK,
            lDS,
            lQT,
            lKT,
            lDSmn,
            lQs,
            lKs,
            lDSs,
            lDO,
            lDOs,
            q_token_stride,
            k_token_stride,
            v_token_stride,
            TQ,
            nctas,
            kernel_hq,
            hq,
            topk,
            _stock_I32(group * kernel_hq),
        ).launch(grid=[nctas, 1, 1], block=[_stock_NTHREADS, 1, 1], stream=stream)
    _stock__cvt_kernel(mW, mDK, mDV, nelem).launch(grid=[cute.ceil_div(nelem, 1024), 1, 1], block=[256, 1, 1], stream=stream)


_stock__compiled = {}
_stock__smc = {}


def _stock__sm_count(dev):
    n = _stock__smc.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _stock__smc[dev.index] = n
    return n


def _stock__cv(x, al=16):
    return from_dlpack(x.detach(), assumed_align=al).mark_layout_dynamic()


@torch.no_grad()
def _stock_run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv):
    if q.ndim != 3 or q.shape[-1] != _stock_DIM:
        raise ValueError(f"q must have shape [Tq, Hq, {_stock_DIM}], got {tuple(q.shape)}")
    tq = q.shape[0]
    tk = k.shape[0]
    hq = q.shape[-2]
    topk = selected_indices.shape[-1]
    dev = q.device
    if tq <= 0 or tk <= 0:
        raise ValueError("q and k/v token streams must be nonempty")
    if hq not in _stock_SUPPORTED_HQ:
        raise ValueError(f"Hq must be one of {_stock_SUPPORTED_HQ}, got {hq}")
    if topk not in _stock_SUPPORTED_TOPK:
        raise ValueError(f"topK must be one of {_stock_SUPPORTED_TOPK}, got {topk}")
    if k.ndim != 3 or k.shape[1:] != (1, _stock_DIM):
        raise ValueError(f"k must have shape [Tk, 1, {_stock_DIM}], got {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v must match k.shape={tuple(k.shape)}, got {tuple(v.shape)}")
    if out.shape != q.shape or dout.shape != q.shape or dq.shape != q.shape:
        raise ValueError("out, dout, and dq must match q.shape")
    if dk.shape != k.shape or dv.shape != k.shape:
        raise ValueError("dk and dv must match k.shape")
    if lse.shape != q.shape[:2] or lse.dtype != torch.float32 or (not lse.is_contiguous()):
        raise ValueError("lse must be contiguous FP32 [Tq, Hq]")
    if selected_indices.shape != (tq, topk):
        raise ValueError("selected_indices must have shape [Tq, topK]")
    if valid_lengths.shape != (tq,):
        raise ValueError("valid_lengths must have shape [Tq]")
    bf16_tensors = (q, k, v, out, dout, dq, dk, dv)
    if any((tensor.dtype != torch.bfloat16 for tensor in bf16_tensors)):
        raise ValueError("q/k/v/out/dout and dq/dk/dv must all be BF16")
    if selected_indices.dtype != torch.int32 or valid_lengths.dtype != torch.int32:
        raise ValueError("selected_indices and valid_lengths must be int32")
    tensors = (*bf16_tensors, lse, selected_indices, valid_lengths)
    if not q.is_cuda or any((tensor.device != dev for tensor in tensors[1:])):
        raise ValueError("all sparse-backward tensors must be on one CUDA device")
    if not all((tensor.is_contiguous() for tensor in (out, dout, selected_indices, valid_lengths, dq, dk, dv))):
        raise ValueError("sparse-backward outputs, state, and support metadata must be contiguous")
    if q.stride()[1:] != (_stock_DIM, 1) or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q/k/v feature rows must be contiguous")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.data_ptr() % 16 != 0 or tensor.stride(0) % 8 != 0:
            raise ValueError(f"{name} base address and token stride must be 16-byte aligned")
    nctas = min(tq, _stock__sm_count(dev))
    stream_id = torch.cuda.current_stream(dev).cuda_stream
    w = torch.empty(tk * _stock_DIM * 2, dtype=torch.float32, device=dev)
    args = (
        _stock__cv(q),
        _stock__cv(k),
        _stock__cv(v),
        _stock__cv(out.view(-1)),
        _stock__cv(dout.view(-1)),
        _stock__cv(lse.view(-1)),
        _stock__cv(selected_indices.view(-1), 4),
        _stock__cv(valid_lengths.view(-1), 4),
        _stock__cv(dq.view(-1)),
        _stock__cv(w),
        _stock__cv(dk.view(-1)),
        _stock__cv(dv.view(-1)),
        _stock_I32(q.stride(0)),
        _stock_I32(k.stride(0)),
        _stock_I32(v.stride(0)),
    )
    compile_key = (dev.index, hq, topk)
    stream = cuda.CUstream(stream_id)
    if compile_key not in _stock__compiled:
        _stock__compiled[compile_key] = cute.compile(
            _stock__launch,
            *args,
            _stock_I32(tq),
            _stock_I32(tk),
            _stock_I32(nctas),
            hq,
            topk,
            make_fake_stream(),
        )
    _stock__compiled[compile_key](*args, _stock_I32(tq), _stock_I32(tk), _stock_I32(nctas), stream)


@torch.no_grad()
def run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv):
    if q.ndim == 3 and q.shape[1] in (16, 32):
        return _stock_run(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv)
    return _run_h96(q, k, v, out, dout, lse, selected_indices, valid_lengths, dq, dk, dv)
