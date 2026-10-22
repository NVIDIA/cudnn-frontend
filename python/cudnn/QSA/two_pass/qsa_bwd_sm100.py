# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Sparse Attention (QSA) backward for SM100 / SM103, written with the CuTe DSL primitives.

The launch sequence is a dense key-tile kernel for the queries below the crossover, and for the remaining queries a
query-major kernel (S, dP, dQ and the P / dS records of every selected block) followed by a key-major kernel that
accumulates dK / dV from those records. ``api._QsaTwoPassBackward`` describes the index contract.
"""

import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

from cudnn.api_base import ws_align

I32 = cutlass.Int32
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

HQ = 24
HKV = 2
GQ = 12
D = 256
W = 2051
TOPK = 512
SCALE = 1.0 / 16.0
LOG2E = 1.4426950408889634

NTHREADS = 512
# Blocks per chunk fill 8 * CHB of the 128 MMA rows. With the K and V tiles copied on to TMEM, 12 blocks leave room
# for a 3-stage K ring. Rows past 8 * CHB alias the next region and are masked out.
CHB = 12
NKS = 3  # K stages, so K(c+1) streams in while S(c), dP(c) and dQ(c-1) run

# shared memory of the query-major kernel (bytes from a 1024-aligned base): [K0][K1][K2][V][Q][dO][dS]
BLK_SMEM = 4096  # a gathered block is exactly [4 slabs][8 rows x 128 B] of 128-byte-swizzle (SW128) atoms, so its TMA box needs no repack
SLAB = 1024
K_STAGE = CHB * BLK_SMEM  # blocks sit at the 8-row-group stride of the MMA operand
OFF_K = 0  # MMA row = 8 * block + 2 * token + head, so one block covers both KV heads
OFF_V = OFF_K + NKS * K_STAGE
QSLAB = 3072
OFF_Q = OFF_V + K_STAGE  # [4 slabs][24 head rows x 128 B] with SW128, rows 0-11 for KV head 0
OFF_DO = OFF_Q + 12288
OFF_DS = OFF_DO + 12288  # dS^T as [16 row groups][3 column chunks][8 rows x 16 B], the MN-major B operand of dQ^T
DS_BYTES = 6144
SMEM_BYTES = OFF_DS + DS_BYTES

# tensor memory columns of the query-major kernel
TM_S = 0  # S^T, double buffered at a 64-column pitch
TM_DP = 32  # dP^T, interleaved with S^T
TM_DQ = 128  # dQ^T halves at 128 and 160
TM_KA = 192  # the chunk's K tile as the TMEM A operand of S^T (128 columns of packed bf16 pairs)
TM_VA = 320  # the chunk's V tile as the TMEM A operand of dP^T
TM_COLS = 512
L2_EVICT_FIRST = 0x12F0000000000000  # createpolicy encoding of the L2 evict-first priority

Q_BOX_BYTES = 24 * 128


@cute.jit
def _wait(bar, parity):
    while not prims.mbarrier_try_wait_parity(bar, parity):
        pass


@cute.jit
def _st_async16(dst, v4, mbar):
    # st.async stores 16 B from registers into a peer CTA's shared memory and completes them on the peer's mbarrier.
    prims.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
        read_only_args=[dst, v4[0], v4[1], v4[2], v4[3], mbar],
    )


def _desc(addr, lbo, sbo, swz):
    return prims.Tcgen05SmemDesc.build(start_address=addr, leading_byte_offset=lbo, stride_byte_offset=sbo, layout=swz)


@cute.jit
def _qmeta(qi):
    ncb = (qi + 1) >> 2
    nsel = ncb
    if nsel > TOPK:
        nsel = I32(TOPK)
    ntail = (qi + 1) - (ncb << 2)
    nblk = nsel
    if ntail > 0:
        nblk = nsel + 1
    return ncb, nsel, ntail, nblk


# The tail block of the last queries may reduce exact zeros up to 3 tokens past T, so the sparse dK/dV
# accumulators carry PAD fp32 elements of slack.
PAD = 4 * HKV * D


# ----------------------------------------------------------------------------- query-major kernel (dQ, P / dS records)
@cute.kernel
def qsa_bwd_sparse_query_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_k: cutlass.GridConstant[cuda.TensorMap],
    tma_v: cutlass.GridConstant[cuda.TensorMap],
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dq: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_sparse_block_slots: cute.Tensor,
    g_pair_records: cute.Tensor,
    T: I32,
    nctas: I32,
    qlo: I32,
    NP: I32,
):
    # Q and dO of an item are read once per KV pass, while the pass's K/V blocks are reused by many queries.
    hint_q = cutlass.Int64(L2_EVICT_FIRST)
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, SMEM_BYTES // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_qf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_qe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_kf = cutlass.Array(cutlass.Int64, NKS, space=cutlass.AddressSpace.smem, alignment=8)
    b_ke = cutlass.Array(cutlass.Int64, NKS, space=cutlass.AddressSpace.smem, alignment=8)
    b_vf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ve = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_sf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_pf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_sde = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_dsf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ktf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_kte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_vtf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_vte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # zero the whole operand region once, so padding rows and stale rows stay finite
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    sbase = smem.data_ptr()
    for it in cutlass.range(tidx, SMEM_BYTES // 16, NTHREADS, unroll=1):
        (sbase + it * 4).store(zero4, alignment=16)

    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(tma_q.get_ptr())
            prims.prefetch_tensormap(tma_do.get_ptr())
            prims.prefetch_tensormap(tma_k.get_ptr())
            prims.prefetch_tensormap(tma_v.get_ptr())
            prims.mbarrier_init(b_qf, 1)
            prims.mbarrier_init(b_qe, 1)
            for s in cutlass.range_constexpr(NKS):
                prims.mbarrier_init(b_kf.subview(s), 1)
                prims.mbarrier_init(b_ke.subview(s), 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_pf.subview(s), 4)
            prims.mbarrier_init(b_sf.subview(0), 1)
            prims.mbarrier_init(b_sf.subview(1), 1)
            prims.mbarrier_init(b_vf, 1)
            prims.mbarrier_init(b_ve, 8)
            prims.mbarrier_init(b_dqf, 1)
            prims.mbarrier_init(b_dqe, 4)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_sde.subview(s), 4)
            prims.mbarrier_init(b_dsf, 1)
            for s in cutlass.range_constexpr(4):
                prims.mbarrier_init(b_ktf.subview(s), 8)
                prims.mbarrier_init(b_vtf.subview(s), 8)
            prims.mbarrier_init(b_kte, 1)
            prims.mbarrier_init(b_vte, 1)
    if warp == 2:
        prims.tcgen05_alloc(tmem_slot, TM_COLS, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    prims.fence_mbarrier_init()
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)

    tmem_raw = tmem_slot.load()
    tmem_col0 = tmem_raw & I32(0xFFFF)
    tmem_row0 = tmem_raw >> I32(16)

    p_lse = g_lse.iterator.raw_ptr()
    p_delta = g_delta.iterator.raw_ptr()
    p_dq = g_dq.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_pair_records = g_pair_records.iterator.raw_ptr()

    # warpgroup dispatch (all roles fit the 128-register budget of a 512-thread CTA)
    wg = warp // 4
    if wg == 0:
        # ================================================================ producer warp (TMA gathers)
        # lane j < CHB owns block slot j of every chunk, one 4 KB TMA box per block and tensor.
        # Tokens past T are zero-filled by the TMA unit.
        if warp == 0:
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qe, (nq & 1) ^ 1)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(b_qf, 8 * Q_BOX_BYTES)
                            for s in cutlass.range_constexpr(4):
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + (OFF_Q + s * QSLAB) // 4,
                                    tma_q.get_ptr(),
                                    (s * 64, 0, qi),
                                    b_qf,
                                    l2_cache_hint=hint_q,
                                )
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + (OFF_DO + s * QSLAB) // 4,
                                    tma_do.get_ptr(),
                                    (s * 64, 0, qi),
                                    b_qf,
                                    l2_cache_hint=hint_q,
                                )
                        # this lane's block slot of the next chunk, loaded one chunk ahead with the column clamped to a
                        # valid one (entries past nsel are replaced by the tail block below)
                        m0 = lane
                        if m0 >= nblk:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            st = cgc % NKS
                            kpar = (cgc // NKS) & 1
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + lane
                                if m1 >= nblk:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + lane
                            blk = xcur
                            ok = m < nblk
                            if lane >= CHB:
                                ok = False
                            nval = nblk - c * CHB
                            if nval > CHB:
                                nval = I32(CHB)
                            tbytes = nval * BLK_SMEM
                            # ---- K
                            _wait(b_ke.subview(st), kpar ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_kf.subview(st), tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + (OFF_K + st * K_STAGE + lane * BLK_SMEM) // 4,
                                    tma_k.get_ptr(),
                                    (0, 0, blk * 4, 0),
                                    b_kf.subview(st),
                                )
                            if c + 1 < nchunks:
                                okn = (c + 1) * CHB + lane < nblk
                                if lane >= CHB:
                                    okn = False
                                if okn:
                                    prims.cp_async_bulk_tensor_prefetch(tma_k.get_ptr(), [0, 0, xnext * 4, 0], [])
                        gc = gc + nchunks
                        nq = nq + 1

        # V has only one SMEM stage.  Giving it to the otherwise-idle warp 1 prevents its empty-barrier wait from
        # serialising warp 0's three-stage K producer, so K can run up to the actual ring depth ahead of the MMA.
        if warp == 1:
            gc = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        m0 = lane
                        if m0 >= nblk:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + lane
                                if m1 >= nblk:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + lane
                            blk = xcur
                            ok = m < nblk
                            if lane >= CHB:
                                ok = False
                            nval = nblk - c * CHB
                            if nval > CHB:
                                nval = I32(CHB)
                            tbytes = nval * BLK_SMEM
                            _wait(b_ve, (cgc & 1) ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_vf, tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + (OFF_V + lane * BLK_SMEM) // 4,
                                    tma_v.get_ptr(),
                                    (0, 0, blk * 4, 0),
                                    b_vf,
                                )
                            if c + 1 < nchunks:
                                okn = (c + 1) * CHB + lane < nblk
                                if lane >= CHB:
                                    okn = False
                                if okn:
                                    prims.cp_async_bulk_tensor_prefetch(tma_v.get_ptr(), [0, 0, xnext * 4, 0], [])
                        gc = gc + nchunks

        # ================================================================ score MMA warp (S^T and dP^T)
        if warp == 2:
            idesc_s = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=24,
                m_dim=128,
                a_major=0,
                b_major=0,
            )
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            # Q/dO as the K-major B operand of S^T / dP^T
            d_q_kmaj = _desc(sbase + OFF_Q // 4, 16, 1024, SW128)
            d_do_kmaj = _desc(sbase + OFF_DO // 4, 16, 1024, SW128)

            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qf, nq & 1)
                        for c in cutlass.range(nchunks + 1, unroll=1):
                            # S(c) and dP(c) only need softmax(c-1) to have read S/dP (signalled early through sde),
                            # so softmax(c) overlaps dQ(c-1) and S/dP(c+1) on the tensor pipe.
                            cgc = gc + c
                            st = cgc % NKS
                            if c < nchunks:
                                _wait(b_kf.subview(st), (cgc // NKS) & 1)
                                # softmax(c-1) has loaded S/dP into registers
                                _wait(b_sde.subview(cgc & 1), ((cgc >> 1) & 1) ^ 1)
                                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                # A = K(c) from TMEM (copied there slab by slab by warps 8-15), because the SMEM operand
                                # fetch, not the math, bounds these small-N MMAs
                                for pub in cutlass.range_constexpr(2):
                                    _wait(b_ktf.subview(pub), cgc & 1)
                                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                    stop = 1 if pub == 0 else 4
                                    for sl4 in cutlass.range(pub, stop, unroll=1):
                                        for k4 in cutlass.range_constexpr(4):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_S + (cgc & 1) * 64), F32),
                                                    prims.make_tmem_ptr(tmem_raw + (TM_KA + sl4 * 32 + k4 * 8), BF16),
                                                    d_q_kmaj.advance_start_address(sl4 * QSLAB + k4 * 32),
                                                    idesc_s,
                                                    (sl4 * 4 + k4) > 0,
                                                )
                                if prims.elect_sync():
                                    prims.tcgen05_commit(b_kte, group=prims.CTAGroup.CTA_1)
                            if c < nchunks:
                                # A = V(c) from TMEM (copied by warps 8-15, which also release the V SMEM stage)
                                for pub in cutlass.range_constexpr(2):
                                    _wait(b_vtf.subview(pub), cgc & 1)
                                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                    stop = 1 if pub == 0 else 4
                                    for sl4 in cutlass.range(pub, stop, unroll=1):
                                        for k4 in cutlass.range_constexpr(4):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_DP + (cgc & 1) * 64), F32),
                                                    prims.make_tmem_ptr(tmem_raw + (TM_VA + sl4 * 32 + k4 * 8), BF16),
                                                    d_do_kmaj.advance_start_address(sl4 * QSLAB + k4 * 32),
                                                    idesc_s,
                                                    (sl4 * 4 + k4) > 0,
                                                )
                                if prims.elect_sync():
                                    prims.tcgen05_commit(b_vte, group=prims.CTAGroup.CTA_1)
                                    prims.tcgen05_commit(b_sf.subview(cgc & 1), group=prims.CTAGroup.CTA_1)
                                    if c == nchunks - 1:
                                        prims.tcgen05_commit(b_qe, group=prims.CTAGroup.CTA_1)
                        gc = gc + nchunks
                        nq = nq + 1

        # ================================================================ dQ MMA warp
        # A second MMA-issuing warp: dQ(c) is issued as soon as softmax(c) has published dS^T, and S/dP of later
        # chunks no longer queue behind it in warp 2 (tcgen05.commit tracks the MMAs of the issuing thread only).
        if warp == 3:
            idesc_q3 = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=24,
                m_dim=128,
                a_major=1,
                b_major=1,
            )
            # K tile read as K^T (MN-major SW128) and dS^T (MN-major, no swizzle), as in warp 2
            d_k_mn3 = _desc(sbase + OFF_K // 4, SLAB, BLK_SMEM, prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
            d_ds_mn3 = _desc(sbase + OFF_DS // 4, 384, 128, prims.Tcgen05SmemSwizzle.NONE)
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        nchunks = (nblk + (CHB - 1)) // CHB
                        for c in cutlass.range(nchunks, unroll=1):
                            cgb = gc + c
                            pb = cgb & 1
                            kb = cgb % NKS
                            # softmax(c) has written dS^T to SMEM
                            _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            if c == 0:
                                # the previous query's dQ has been drained from TMEM
                                _wait(b_dqe, (nq & 1) ^ 1)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            acc_q = c > 0  # the first chunk of a query initialises the dQ accumulator
                            dk_mn = d_k_mn3.advance_start_address(kb * K_STAGE)
                            # K-steps of 16 rows = 2 blocks over the chunk's valid blocks only
                            nv3 = nblk - c * CHB
                            if nv3 > CHB:
                                nv3 = I32(CHB)
                            nkk = (nv3 + 1) >> 1
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range(nkk, unroll=1):
                                    if prims.elect_sync():
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                            dk_mn.advance_start_address(hh * 2 * SLAB + kk * 2 * BLK_SMEM),
                                            d_ds_mn3.advance_start_address(kk * 768),
                                            idesc_q3,
                                            acc_q | (kk > 0),
                                        )
                            if prims.elect_sync():
                                prims.tcgen05_commit(b_ke.subview(kb), group=prims.CTAGroup.CTA_1)
                                prims.tcgen05_commit(b_dsf, group=prims.CTAGroup.CTA_1)
                                if c == nchunks - 1:
                                    prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.bar_warp_sync(0xFFFFFFFF)
                        gc = gc + nchunks
                        nq = nq + 1

    elif wg == 1:
        # ================================================================ softmax warps (P, dS, records, dQ drain)
        qd = warp % 4
        row = qd * 32 + lane
        hrow = lane & 1  # kv head of this row
        trow = (lane & 7) >> 1  # token within block
        jrow = row >> 3  # block slot within chunk
        c_s = F32(SCALE * LOG2E)
        gc = I32(0)
        nq = I32(0)
        ds_off = (OFF_DS + jrow * 384 + (row & 7) * 16) // 4
        for pp in cutlass.range(NP, unroll=1):
            for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                o0 = prow.load()
                nblk = (prow + 1).load() - o0
                if nblk > 0:
                    lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                    _, _, ntail, _ = _qmeta(qi)
                    nchunks = (nblk + (CHB - 1)) // CHB
                    lbase = qi * HQ + hrow * GQ
                    # the tail block (present iff ntail > 0) is the last entry of the query's whole partition list
                    tl = (p_sparse_pass_offsets + ((qi - qlo) * POS + NP)).load() - 1 - o0
                    if ntail == 0:
                        tl = I32(-1)
                    # record slot of this row's block entry, loaded one chunk ahead
                    prow_pp = p_sparse_block_slots + ((qi - qlo) * PART_W + o0)
                    mpf = jrow
                    if mpf >= nblk:
                        mpf = I32(0)
                    pnx = (prow_pp + mpf).load()
                    lse_a = (p_lse + lbase).load(count=4, alignment=16)
                    lse_b = (p_lse + lbase + 4).load(count=4, alignment=16)
                    lse_c = (p_lse + lbase + 8).load(count=4, alignment=16)
                    del_a = (p_delta + lbase).load(count=4, alignment=16)
                    del_b = (p_delta + lbase + 4).load(count=4, alignment=16)
                    del_c = (p_delta + lbase + 8).load(count=4, alignment=16)
                    # The 12-head softmax state stays at vector level across the chunks, built once per query.
                    lse_v = cutlass.Vector.from_elements(
                        [
                            lse_a[0],
                            lse_a[1],
                            lse_a[2],
                            lse_a[3],
                            lse_b[0],
                            lse_b[1],
                            lse_b[2],
                            lse_b[3],
                            lse_c[0],
                            lse_c[1],
                            lse_c[2],
                            lse_c[3],
                        ],
                        F32,
                    )
                    del_v = cutlass.Vector.from_elements(
                        [
                            del_a[0],
                            del_a[1],
                            del_a[2],
                            del_a[3],
                            del_b[0],
                            del_b[1],
                            del_b[2],
                            del_b[3],
                            del_c[0],
                            del_c[1],
                            del_c[2],
                            del_c[3],
                        ],
                        F32,
                    )
                    zero12 = cutlass.Vector.from_elements([F32(0.0)] * GQ, F32)
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc & 1
                        _wait(b_sf.subview(cgc & 1), (cgc >> 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        t_s = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_S + (cgc & 1) * 64).as_ptr(F32)
                        t_dp = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DP + (cgc & 1) * 64).as_ptr(F32)
                        sv = prims.tcgen05_ld("32x32b", t_s, num=32)
                        dpv = prims.tcgen05_ld("32x32b", t_dp, num=32)
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_sde.subview(cgc & 1))

                        m = c * CHB + jrow
                        valid = m < nblk
                        if jrow >= CHB:
                            valid = False
                        slot_ok = valid
                        pcur = pnx
                        if c + 1 < nchunks:
                            mpf = (c + 1) * CHB + jrow
                            if mpf >= nblk:
                                mpf = I32(0)
                            pnx = (prow_pp + mpf).load()
                        if m == tl:
                            valid = valid & (trow < ntail)
                        score_v = sv[0:GQ]
                        dot_v = dpv[0:GQ]
                        if hrow == 1:
                            score_v = sv[GQ : 2 * GQ]
                            dot_v = dpv[GQ : 2 * GQ]
                        prob_v = zero12
                        ds_v = zero12
                        if valid:
                            prob_v = cute.math.exp2(score_v * c_s - lse_v * F32(LOG2E), fastmath=True)
                            ds_v = prob_v * (dot_v - del_v) * F32(SCALE)

                        # dS^T in block-diagonal placement (own head group, zeros elsewhere), built in packed bf16 words
                        prw = prob_v.to(BF16).bitcast(I32)
                        drw = ds_v.to(BF16).bitcast(I32)
                        dwords = []
                        for n in cutlass.range_constexpr(16):
                            dv_ = I32(0)
                            if cutlass.const_expr(n < 6):
                                if hrow == 0:
                                    dv_ = drw[n]
                            elif cutlass.const_expr(n < 12):
                                if hrow == 1:
                                    dv_ = drw[n - 6]
                            dwords.append(dv_)
                        dvec = cutlass.Vector.from_elements(dwords, I32)

                        # dQ(c-1) has finished reading the single dS^T SMEM buffer
                        _wait(b_dsf, (cgc & 1) ^ 1)
                        # dS^T row -> SMEM (MN-major core matrices: 3 n-chunks of 8 at 128 B)
                        for nc in cutlass.range_constexpr(3):
                            (sbase + ds_off + nc * 32).store(dvec[nc * 4 : nc * 4 + 4], alignment=16)
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_pf.subview(st))
                        # The key-major kernel runs after this one, so dS is published to the dQ MMA first and the pair
                        # record streams out after the ready signal.
                        if slot_ok:
                            roff = cutlass.Int64(pcur) * REC_W + cutlass.Int64(trow * 12 + hrow * 6)
                            rbase = p_pair_records + roff
                            for u in cutlass.range_constexpr(3):
                                prims.store_ext(prw[2 * u : 2 * u + 2], rbase + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)
                                prims.store_ext(drw[2 * u : 2 * u + 2], rbase + 48 + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)

                    # ---- dQ for this query: dQ^T[dim, qhead] from TMEM -> dq[qi, qhead, dim]
                    _wait(b_dqf, nq & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    dqv = []
                    for hh in cutlass.range_constexpr(2):
                        t_q = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DQ + hh * 32).as_ptr(F32)
                        dqv.append(prims.tcgen05_ld("32x32b", t_q, num=32))
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_dqe)
                    for hh in cutlass.range_constexpr(2):
                        dim = hh * 128 + qd * 32 + lane
                        qb = p_dq + (qi * (HQ * D) + dim)
                        if o0 == 0:
                            # The query's first non-empty KV pass stores dQ, later passes add to it.
                            for n in cutlass.range_constexpr(HQ):
                                (qb + n * D).store(dqv[hh][n])
                        else:
                            for n in cutlass.range_constexpr(HQ):
                                prims.red(
                                    prims.ReductionOp.ADD,
                                    prims.ReductionType.F32,
                                    qb + n * D,
                                    dqv[hh][n],
                                    mem_order=prims.MemOrder.RELAXED,
                                    mem_scope=prims.MemScope.GPU,
                                    cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                                )
                    gc = gc + nchunks
                    nq = nq + 1

    else:
        # ================================================================ K/V copy warps (SMEM -> TMEM)
        # The K and V tiles of every chunk become TMEM A operands, slab by slab so S can start on the first slab.
        # Lane = chunk row (8 * block + 2 * token + head), column = packed bf16 dim pair. Warp pair hh of a lane
        # quadrant moves 16-byte chunks 4hh..4hh+3 of each 128-byte SW128 row segment (chunk cc of row y sits at
        # physical chunk cc ^ y of its 1 KB atom).
        qd = warp % 4
        hh = (warp - 8) >> 2
        r = qd * 32 + lane
        yr = r & 7
        roff = (OFF_K + (r >> 3) * BLK_SMEM + yr * 128) // 4
        # zero every quadrant's K / V operand rows once, so rows a chunk does not copy keep finite data
        zw = []
        for e in cutlass.range_constexpr(16):
            zw.append(I32(0))
        for zb in cutlass.range_constexpr(2):
            for sl4 in cutlass.range_constexpr(4):
                prims.tcgen05_st(
                    "32x32b",
                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + zb * 128 + sl4 * 32 + hh * 16).as_ptr(I32),
                    cutlass.Vector.from_elements(zw, I32),
                )
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        gc = I32(0)
        for pp in cutlass.range(NP, unroll=1):
            for qi in cutlass.range(qlo + bid, T, nctas, unroll=1):
                prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                o0 = prow.load()
                nblk = (prow + 1).load() - o0
                if nblk > 0:
                    nchunks = (nblk + (CHB - 1)) // CHB
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc % NKS
                        # the last chunk of a (query, pass) may be partial, and quadrant qd holds blocks 4qd..4qd+3
                        nval_c = nblk - c * CHB
                        if nval_c > CHB:
                            nval_c = I32(CHB)
                        do_cp = qd * 4 < nval_c
                        # K(c): SMEM stage st -> TMEM (single buffer, free once S(c-1) has completed)
                        _wait(b_kf.subview(st), (cgc // NKS) & 1)
                        _wait(b_kte, (cgc & 1) ^ 1)
                        rb = sbase + (roff + st * (K_STAGE // 4))
                        for sl4 in cutlass.range_constexpr(4):
                            if do_cp:
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (rb + (sl4 * (SLAB // 4) + ((yr ^ cc) * 4))).load(count=4, alignment=16)
                                    for e in cutlass.range_constexpr(4):
                                        wv.append(v4[e])
                                prims.tcgen05_st(
                                    "32x32b",
                                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_KA + sl4 * 32 + hh * 16).as_ptr(I32),
                                    cutlass.Vector.from_elements(wv, I32),
                                )
                            if cutlass.const_expr(sl4 == 0):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_ktf.subview(0))
                            elif cutlass.const_expr(sl4 == 3):
                                # One wait retires the three outstanding stores for slabs 1-3.
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_ktf.subview(1))
                        # V(c): SMEM -> TMEM (free once dP(c-1) has completed), then release the V SMEM stage
                        _wait(b_vf, cgc & 1)
                        _wait(b_vte, (cgc & 1) ^ 1)
                        vb = sbase + (roff - OFF_K // 4 + OFF_V // 4)
                        for sl4 in cutlass.range_constexpr(4):
                            if do_cp:
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (vb + (sl4 * (SLAB // 4) + ((yr ^ cc) * 4))).load(count=4, alignment=16)
                                    for e in cutlass.range_constexpr(4):
                                        wv.append(v4[e])
                                prims.tcgen05_st(
                                    "32x32b",
                                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_VA + sl4 * 32 + hh * 16).as_ptr(I32),
                                    cutlass.Vector.from_elements(wv, I32),
                                )
                            if cutlass.const_expr(sl4 == 0):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_vtf.subview(0))
                            elif cutlass.const_expr(sl4 == 3):
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_vtf.subview(1))
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_ve)
                    gc = gc + nchunks

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 2:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), TM_COLS, group="cta_1")


qsa_bwd_sparse_query_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sparse_query_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    sparse_block_slots: cute.Tensor,
    pair_records: cute.Tensor,
    nctas: I32,
    qlo: I32,
    NP: I32,
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    kv_layout = cute.make_layout((T, HKV, 4, 64), stride=(HKV * D, D, 64, 1))
    tma_k = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, kv_layout),
        box_dims=(4, HKV, 4, 64),
        stride_order=(3, 1, 0, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    tma_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, kv_layout),
        box_dims=(4, HKV, 4, 64),
        stride_order=(3, 1, 0, 2),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    qsa_bwd_sparse_query_kernel(
        tma_q, tma_do, tma_k, tma_v, lse, delta, dq, sparse_blocks, sparse_pass_offsets, sparse_block_slots, pair_records, I32(T), nctas, qlo, NP
    ).launch(grid=(nctas, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- key-major kernel (dK, dV)
# One work item per key (range r, block b): the n queries of range r that selected block b. Their Q / dO rows
# (24 heads each) are gathered two queries per stage (48 K-rows) and, with the P / dS records expanded into
# MN-major B tiles, accumulated on the tensor core:
#   dV^T [256 dims, 16] += dO^T [256, 48] P [48, 16]      dK^T [256 dims, 16] += Q^T [256, 48] dS [48, 16]
# (column n < 8 = kv head n // 4, token n % 4, columns 8-15 are zero). The item's dV / dK (fp32, TMEM double
# buffered) are then reduced once into the fp32 accumulators.
PB_NST = 4  # stages of two queries: one landing, one being expanded, one on the tensor core, one refilling
PB_SLAB = 48 * 128  # one 64-dim slab of a stage tile (48 rows x 128 B, SW128)
PB_Q = 0
PB_DO = 4 * PB_SLAB
PB_REC = 8 * PB_SLAB  # 2 records (768 B)
PB_P = PB_REC + 1024  # P tile: [6 K-groups][2 N-chunks][8 rows x 16 B]
PB_DS = PB_P + 1536
PB_STAGE = PB_DS + 1536  # every stage starts on a 1 KB boundary, which the SW128 slabs require
PB_OFF_ST = PB_NST * PB_STAGE  # dK/dV staging: 4 drain warps x 4 boxes x [4 tokens][2 kv][32 dims] fp32
PB_ST_W = 4 * 1024
PB_SMEM = PB_OFF_ST + 4 * PB_ST_W
PB_THREADS = 256  # w0 producer, w1 MMA, w2-3 record expansion, w4-7 dK/dV drain
PB_STAGE_TX = 8 * 3072 + 384  # per query: Q + dO boxes + record


@cute.kernel
def qsa_bwd_sparse_key_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_rk: cutlass.GridConstant[cuda.TensorMap],
    tma_rv: cutlass.GridConstant[cuda.TensorMap],
    g_pair_records: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_slot_queries: cute.Tensor,
    num_dense_queries: I32,
    Ktot: I32,
    SK: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nctas, _, _ = cute.arch.grid_dim()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, PB_SMEM // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_full = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_exp = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_emp = cutlass.Array(cutlass.Int64, PB_NST, space=cutlass.AddressSpace.smem, alignment=8)
    b_af = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_ae = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # the dense pair grid is a programmatic dependent launch (PDL) secondary, so it backfills the SMs this grid leaves
    prims.griddepcontrol("launch_dependents")

    sbase = smem.data_ptr()
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    for it in cutlass.range(tidx, PB_SMEM // 16, PB_THREADS, unroll=1):
        (sbase + it * 4).store(zero4, alignment=16)
    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(tma_q.get_ptr())
            prims.prefetch_tensormap(tma_do.get_ptr())
            prims.prefetch_tensormap(tma_rk.get_ptr())
            prims.prefetch_tensormap(tma_rv.get_ptr())
            for s in cutlass.range_constexpr(PB_NST):
                prims.mbarrier_init(b_full.subview(s), 1)
                prims.mbarrier_init(b_exp.subview(s), 2)
                prims.mbarrier_init(b_emp.subview(s), 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_af.subview(s), 1)
                prims.mbarrier_init(b_ae.subview(s), 4)
    if warp == 1:
        prims.tcgen05_alloc(tmem_slot, 128, group="cta_1")
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")
    prims.fence_mbarrier_init()
    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(0)

    tmem_raw = tmem_slot.load()
    tmem_col0 = tmem_raw & I32(0xFFFF)
    tmem_row0 = tmem_raw >> I32(16)
    p_pair_records = g_pair_records.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_slot_queries = g_slot_queries.iterator.raw_ptr()

    if warp == 0:
        # ---------------------------------------------------------------- producer
        gs = I32(0)
        # key counts / offsets two keys ahead and the first 32 query ids one key ahead keep the slot_queries latency off
        # the start of every key
        n_nx = I32(0)
        o_nx = I32(0)
        q_nx = I32(0)
        n_n2 = I32(0)
        o_n2 = I32(0)
        if bid < Ktot:
            n_nx = (p_key_pair_counts + bid).load()
            o_nx = (p_key_slot_offsets + bid).load()
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
        if bid + nctas < Ktot:
            n_n2 = (p_key_pair_counts + (bid + nctas)).load()
            o_n2 = (p_key_slot_offsets + (bid + nctas)).load()
        for w in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            o = o_nx
            q0v = q_nx
            n_nx = n_n2
            o_nx = o_n2
            q_nx = I32(0)
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
            w2 = w + 2 * nctas
            if w2 < Ktot:
                n_n2 = (p_key_pair_counts + w2).load()
                o_n2 = (p_key_slot_offsets + w2).load()
            for qb in cutlass.range(0, n, 32, unroll=1):
                qv = q0v
                if qb > 0:
                    qv = I32(0)
                    if qb + lane < n:
                        qv = (p_slot_queries + (o + qb + lane)).load()
                nloc = n - qb
                if nloc > 32:
                    nloc = I32(32)
                for j2 in cutlass.range(0, nloc, 2, unroll=1):
                    q0 = prims.shfl_sync(0xFFFFFFFF, qv, j2, 0x1F, prims.Shfl.IDX)
                    q1 = prims.shfl_sync(0xFFFFFFFF, qv, j2 + 1, 0x1F, prims.Shfl.IDX)
                    nq = nloc - j2
                    if nq > 2:
                        nq = I32(2)
                    slot = gs % PB_NST
                    _wait(b_emp.subview(slot), ((gs // PB_NST) & 1) ^ 1)
                    if prims.elect_sync():
                        sb = sbase + (slot * PB_STAGE) // 4
                        prims.mbarrier_arrive_expect_tx(b_full.subview(slot), nq * PB_STAGE_TX)
                        for s in cutlass.range_constexpr(4):
                            prims.cp_async_bulk_tensor_shared_cta_global(sb + (PB_Q + s * PB_SLAB) // 4, tma_q.get_ptr(), (s * 64, 0, q0), b_full.subview(slot))
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sb + (PB_DO + s * PB_SLAB) // 4, tma_do.get_ptr(), (s * 64, 0, q0), b_full.subview(slot)
                            )
                        if nq == 2:
                            for s in cutlass.range_constexpr(4):
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sb + (PB_Q + s * PB_SLAB + 3072) // 4, tma_q.get_ptr(), (s * 64, 0, q1), b_full.subview(slot)
                                )
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sb + (PB_DO + s * PB_SLAB + 3072) // 4, tma_do.get_ptr(), (s * 64, 0, q1), b_full.subview(slot)
                                )
                        prims.cp_async_bulk_shared_cluster_global(
                            sb + PB_REC // 4,
                            p_pair_records + cutlass.Int64(o + qb + j2) * REC_W,
                            b_full.subview(slot),
                            nq * (REC_W * 4),
                            l2_cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                        )
                    prims.bar_warp_sync(0xFFFFFFFF)
                    gs = gs + 1
    elif warp == 1:
        # ---------------------------------------------------------------- MMA issuer
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=F32,
            a_dtype=BF16,
            b_dtype=BF16,
            n_dim=16,
            m_dim=128,
            a_major=1,
            b_major=1,
        )
        SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
        SWN = prims.Tcgen05SmemSwizzle.NONE
        # Q / dO tiles read as Q^T / dO^T (MN-major SW128): LBO = next 64-dim slab, SBO = next 8 rows
        d_q = _desc(sbase + PB_Q // 4, PB_SLAB, 1024, SW128)
        d_do = _desc(sbase + PB_DO // 4, PB_SLAB, 1024, SW128)
        # P / dS tiles (MN-major, no swizzle): LBO = next 8 K-rows (256 B), SBO = next 8 columns (128 B)
        d_p = _desc(sbase + PB_P // 4, 256, 128, SWN)
        d_ds = _desc(sbase + PB_DS // 4, 256, 128, SWN)
        gs = I32(0)
        ia = I32(0)
        n_nx = I32(0)
        if bid < Ktot:
            n_nx = (p_key_pair_counts + bid).load()
        for w in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            if w + nctas < Ktot:
                n_nx = (p_key_pair_counts + (w + nctas)).load()
            if n > 0:
                ab = ia & 1
                _wait(b_ae.subview(ab), ((ia >> 1) & 1) ^ 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                nst = (n + 1) >> 1
                for si in cutlass.range(nst, unroll=1):
                    slot = gs % PB_NST
                    ph = (gs // PB_NST) & 1
                    _wait(b_full.subview(slot), ph)
                    _wait(b_exp.subview(slot), ph)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    soff = slot * PB_STAGE
                    if cutlass.const_expr(SK > 0):
                        # a single-query stage (odd n) has zero P / dS rows 24-47, so K-step 2 (rows 32-47) is skipped
                        nk3 = I32(2)
                        if n - 2 * si > 1:
                            nk3 = I32(3)
                        if prims.elect_sync():
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range_constexpr(3):
                                    accf = (si > 0) | (kk > 0)
                                    if kk < nk3:
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                            d_do.advance_start_address(soff + hh * 2 * PB_SLAB + kk * 2048),
                                            d_p.advance_start_address(soff + kk * 512),
                                            idesc,
                                            accf,
                                        )
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                            d_q.advance_start_address(soff + hh * 2 * PB_SLAB + kk * 2048),
                                            d_ds.advance_start_address(soff + kk * 512),
                                            idesc,
                                            accf,
                                        )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    else:
                        if prims.elect_sync():
                            for hh in cutlass.range_constexpr(2):
                                for kk in cutlass.range_constexpr(3):
                                    accf = (si > 0) | (kk > 0)
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                        d_do.advance_start_address(soff + hh * 2 * PB_SLAB + kk * 2048),
                                        d_p.advance_start_address(soff + kk * 512),
                                        idesc,
                                        accf,
                                    )
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                        d_q.advance_start_address(soff + hh * 2 * PB_SLAB + kk * 2048),
                                        d_ds.advance_start_address(soff + kk * 512),
                                        idesc,
                                        accf,
                                    )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    gs = gs + 1
                if prims.elect_sync():
                    prims.tcgen05_commit(b_af.subview(ab), group=prims.CTAGroup.CTA_1)
                prims.bar_warp_sync(0xFFFFFFFF)
                ia = ia + 1
    elif warp < 4:
        # ---------------------------------------------------------------- record expansion into MN-major B tiles
        t = tidx - 64
        jq = I32(0)
        if t >= 24:
            jq = I32(1)
        hq = t - 24 * jq
        hkv = I32(0)
        if hq >= 12:
            hkv = I32(1)
        rw = hq >> 1
        rs = (hq & 1) * 16
        toff = ((t >> 3) * 256 + (t & 7) * 16) // 4
        gs = I32(0)
        n_nx = I32(0)
        if bid < Ktot:
            n_nx = (p_key_pair_counts + bid).load()
        for w in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            if w + nctas < Ktot:
                n_nx = (p_key_pair_counts + (w + nctas)).load()
            nst = (n + 1) >> 1
            for si in cutlass.range(nst, unroll=1):
                slot = gs % PB_NST
                ph = (gs // PB_NST) & 1
                nq = n - 2 * si
                _wait(b_full.subview(slot), ph)
                if t < 48:
                    sb = sbase + (slot * PB_STAGE) // 4
                    pw0 = I32(0)
                    pw1 = I32(0)
                    dw0 = I32(0)
                    dw1 = I32(0)
                    if jq < nq:
                        rb = sb + (PB_REC // 4 + jq * REC_W + rw)
                        pv = []
                        dv_ = []
                        for tt in cutlass.range_constexpr(4):
                            pv.append(((rb + tt * 12).load() >> rs) & I32(0xFFFF))
                            dv_.append(((rb + 48 + tt * 12).load() >> rs) & I32(0xFFFF))
                        pw0 = pv[0] | (pv[1] << 16)
                        pw1 = pv[2] | (pv[3] << 16)
                        dw0 = dv_[0] | (dv_[1] << 16)
                        dw1 = dv_[2] | (dv_[3] << 16)
                    e0 = pw0
                    e1 = pw1
                    e2 = I32(0)
                    e3 = I32(0)
                    f0 = dw0
                    f1 = dw1
                    f2 = I32(0)
                    f3 = I32(0)
                    if hkv == 1:
                        e2 = pw0
                        e3 = pw1
                        e0 = I32(0)
                        e1 = I32(0)
                        f2 = dw0
                        f3 = dw1
                        f0 = I32(0)
                        f1 = I32(0)
                    vp = cutlass.Vector.from_elements([e0, e1, e2, e3], I32)
                    vd = cutlass.Vector.from_elements([f0, f1, f2, f3], I32)
                    (sb + (PB_P // 4 + toff)).store(vp, alignment=16)
                    (sb + (PB_DS // 4 + toff)).store(vd, alignment=16)
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_exp.subview(slot))
                gs = gs + 1
    else:
        # ---------------------------------------------------------------- dK / dV drain (one TMEM lane quadrant per warp)
        qd = warp - 4
        stg = sbase + (PB_OFF_ST + qd * PB_ST_W) // 4
        ia = I32(0)
        r = I32(0)
        kb1 = _kbase(I32(1), num_dense_queries)
        kb0 = I32(0)
        n_nx = I32(0)
        if bid < Ktot:
            n_nx = (p_key_pair_counts + bid).load()
        for w in cutlass.range(bid, Ktot, nctas, unroll=1):
            while w >= kb1:
                r = r + 1
                kb0 = kb1
                kb1 = _kbase(r + 1, num_dense_queries)
            n = n_nx
            if w + nctas < Ktot:
                n_nx = (p_key_pair_counts + (w + nctas)).load()
            if n > 0:
                blk = w - kb0
                ab = ia & 1
                _wait(b_af.subview(ab), (ia >> 1) & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                va = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + ab * 64).as_ptr(F32), num=32)
                vb = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + ab * 64 + 32).as_ptr(F32), num=32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_ae.subview(ab))
                # The key's dV / dK slice of this warp's 32 dims is staged as four [4 tokens][2 kv][32 dims] boxes and
                # added with one full-line TMA tensor reduce each. The previous key's reduces must have read the boxes.
                prims.cp_async_bulk_wait_group(0, read=True)
                prims.bar_warp_sync(0xFFFFFFFF)
                for bx in cutlass.range_constexpr(4):
                    # box bx: tensor (0 dV, 1 dK) = bx >> 1, d-half = bx & 1
                    for kv in cutlass.range_constexpr(2):
                        for tt in cutlass.range_constexpr(4):
                            col = 4 * kv + tt + 16 * (bx & 1)
                            if cutlass.const_expr((bx >> 1) == 0):
                                val = va[col]
                            else:
                                val = vb[col]
                            (stg + (bx * 256 + (tt * 2 + kv) * 32 + lane)).store(prims.mov_b32(val, target_type=I32))
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    for bx in cutlass.range_constexpr(4):
                        if cutlass.const_expr((bx >> 1) == 0):
                            prims.cp_async_bulk_tensor_reduce(tma_rv.get_ptr(), stg + bx * 256, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, blk * 4])
                        else:
                            prims.cp_async_bulk_tensor_reduce(tma_rk.get_ptr(), stg + bx * 256, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, blk * 4])
                prims.cp_async_bulk_commit_group()
                ia = ia + 1
        prims.cp_async_bulk_wait_group(0)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), 128, group="cta_1")


qsa_bwd_sparse_key_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sparse_key_host(
    q: cute.Tensor,
    do: cute.Tensor,
    pair_records: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    key_pair_counts: cute.Tensor,
    slot_queries: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    num_dense_queries: I32,
    Ktot: I32,
    nctas: I32,
    SK: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    TP = sparse_dk_acc.shape[0] // (HKV * D)
    tma_rk = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(sparse_dk_acc.iterator, cute.make_layout((TP, HKV, D), stride=(HKV * D, D, 1))),
        box_dims=(4, HKV, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    tma_rv = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(sparse_dv_acc.iterator, cute.make_layout((TP, HKV, D), stride=(HKV * D, D, 1))),
        box_dims=(4, HKV, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    qsa_bwd_sparse_key_kernel(tma_q, tma_do, tma_rk, tma_rv, pair_records, key_slot_offsets, key_pair_counts, slot_queries, num_dense_queries, Ktot, SK).launch(
        grid=(nctas, 1, 1), block=(PB_THREADS, 1, 1), stream=stream
    )


# ----------------------------------------------------------------------------- prep kernel
PREP_THREADS = 256
PREP_ROWS = 4  # Delta rows in flight per warp, so all eight vector loads issue before the dot products


@cute.kernel
def qsa_bwd_prep_kernel(
    g_o: cute.Tensor,
    g_do: cute.Tensor,
    g_delta: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    T: I32,
    nz: I32,
    nsel: I32,
    n_key_pair_counts: I32,
    n_dq_acc: I32,
    n_dense_dkv: I32,
    bitmap_fill: I32,
):
    # Delta and the zero fill of every accumulator and counter share one launch.
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tidx % 32
    wib = tidx // 32
    p_o = g_o.iterator.raw_ptr()
    p_do = g_do.iterator.raw_ptr()
    p_delta = g_delta.iterator.raw_ptr()
    nrows = T * HQ
    nwarps = nb * (PREP_THREADS // 32)
    for r0 in cutlass.range(bid * (PREP_THREADS // 32) + wib, nrows, nwarps * PREP_ROWS, unroll=1):
        o_vecs = []
        do_vecs = []
        for u in cutlass.range_constexpr(PREP_ROWS):
            r = r0 + u * nwarps
            if r >= nrows:
                r = r0
            o_vecs.append((p_o + r * D + lane * 8).load(count=8, alignment=16))
            do_vecs.append((p_do + r * D + lane * 8).load(count=8, alignment=16))
        for u in cutlass.range_constexpr(PREP_ROWS):
            r = r0 + u * nwarps
            ov = o_vecs[u].to(F32)
            dv = do_vecs[u].to(F32)
            acc = F32(0.0)
            for j in cutlass.range_constexpr(8):
                acc = acc + ov[j] * dv[j]
            for sh in cutlass.range_constexpr(5):
                acc = acc + prims.shfl_sync(0xFFFFFFFF, acc, 16 >> sh, 0x1F, prims.Shfl.BFLY)
            if lane == 0:
                if r < nrows:
                    (p_delta + r).store(acc)
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    zf = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
    zi = cutlass.Vector.from_elements([I32(0)] * 4, I32)
    bitmap_words = cutlass.Vector.from_elements([bitmap_fill] * 4, I32)
    for it in cutlass.range(bid * PREP_THREADS + tidx, nz // 8, nb * PREP_THREADS, unroll=1):
        (p_sparse_dk_acc + it * 8).store(zf, alignment=16)
        (p_sparse_dv_acc + it * 8).store(zf, alignment=16)
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, nsel // 4, nb * PREP_THREADS, unroll=1):
        (p_dense_block_bitmap + it * 4).store(bitmap_words, alignment=16)
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_key_pair_counts // 4, nb * PREP_THREADS, unroll=1):
        (p_key_pair_counts + it * 4).store(zi, alignment=16)
    p_dq_acc = g_dq_acc.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_dq_acc // 8, nb * PREP_THREADS, unroll=1):
        (p_dq_acc + it * 8).store(zf, alignment=16)
    p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
    p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, n_dense_dkv // 8, nb * PREP_THREADS, unroll=1):
        (p_dense_dk_acc + it * 8).store(zf, alignment=16)
        (p_dense_dv_acc + it * 8).store(zf, alignment=16)


qsa_bwd_prep_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _prep_host(
    o: cute.Tensor,
    do: cute.Tensor,
    delta: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    dense_block_bitmap: cute.Tensor,
    key_pair_counts: cute.Tensor,
    dq_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    nblocks: I32,
    bitmap_fill: I32,
    stream: cuda_driver.CUstream,
):
    T = o.shape[0]
    nz = sparse_dk_acc.shape[0]
    qsa_bwd_prep_kernel(
        o,
        do,
        delta,
        sparse_dk_acc,
        sparse_dv_acc,
        dense_block_bitmap,
        key_pair_counts,
        dq_acc,
        dense_dk_acc,
        dense_dv_acc,
        I32(T),
        I32(nz),
        I32(dense_block_bitmap.shape[0]),
        I32(key_pair_counts.shape[0]),
        I32(dq_acc.shape[0]),
        I32(dense_dk_acc.shape[0]),
        bitmap_fill,
    ).launch(grid=(nblocks, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)


# ----------------------------------------------------------------------------- KV-pass partition and inverted index
# The query-major kernel walks (KV pass, query) items pass-major so that one pass's K/V blocks stay L2 resident.
# sparse_blocks[r] holds query qlo+r's selected blocks (and its tail block ncb) grouped by pass p = block // RB, and
# sparse_pass_offsets[r, p] .. sparse_pass_offsets[r, p+1] bounds pass p's segment.
PART_W = 516  # list row stride in int32: 512 blocks plus the tail block, rounded up to a multiple of 4 words
POS = 64  # offset row stride in int32, a power of two well above the 11 passes of the longest sequence, so NP < POS

# Sparse queries [num_dense_queries, T) are split into ranges of QR queries. Every (query, selected block) pair gets a slot in a
# key-major order: key = kbase(range) + block, and the slots of a key are contiguous
# (key_slot_offsets[key] .. key_slot_offsets[key] + key_pair_counts[key]).
# The query-major kernel writes the pair's P / dS record (bf16, [4 tokens][24 heads] each) into its slot, and the
# key-major kernel walks the keys.
QR = 2048  # queries per range: a range's Q / dO rows stay L2 resident while the key-major kernel walks its keys
QR_SH = 11  # log2(QR)
REC_W = 96  # int32 words per record (384 B: P[4][24] bf16, then dS[4][24] bf16)


@cute.jit
def _kbase(r, num_dense_queries):
    # first key of range r (ranges before the last one are never clipped by T, and num_dense_queries and QR are multiples of 4)
    return r * ((num_dense_queries >> 2) + 1) + (QR >> 2) * ((r * (r + 1)) >> 1)


@cute.jit
def _pairs_before(a, b):
    # number of (query, block) pairs of the queries in [a, b) (all >= 2051: 512 blocks + the tail block if any)
    return I32(513) * (b - a) - ((b >> 2) - (a >> 2))


@cute.kernel
def qsa_bwd_partition_kernel(
    g_idx: cute.Tensor, g_sparse_blocks: cute.Tensor, g_sparse_pass_offsets: cute.Tensor, g_key_pair_counts: cute.Tensor, qlo: I32, T: I32, RB: I32, NP: I32
):
    # per-query block lists split by KV pass, plus the per-key pair counts of the inverted index
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = qlo + bid * 8 + tidx // 32
    p_idx = g_idx.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    if qi < T:
        ncb, nsel, _, nblk = _qmeta(qi)
        r = qi - qlo
        kb = _kbase(r >> QR_SH, qlo)
        lt = (I32(1) << lane) - I32(1)
        off = I32(0)
        if lane == 0:
            (p_sparse_pass_offsets + r * POS).store(I32(0))
        for pp in cutlass.range(NP, unroll=1):
            lo = pp * RB
            hi = lo + RB
            for m0 in cutlass.range(0, nblk, 32, unroll=1):
                m = m0 + lane
                blk = ncb
                inp = m < nblk
                if m < nsel:
                    blk = (p_idx + (qi * W + m * 4)).load() >> 2
                    # a malformed id must not reach a future block or the dK/dV rows past T
                    if blk >= ncb:
                        inp = False
                if blk < lo:
                    inp = False
                if blk >= hi:
                    inp = False
                bal = prims.vote_sync(0xFFFFFFFF, inp, prims.VoteSync.BALLOT)
                if inp:
                    (p_sparse_blocks + (r * PART_W + off + cute.arch.popc(bal & lt))).store(blk)
                    prims.red(
                        prims.ReductionOp.ADD,
                        prims.ReductionType.U32,
                        p_key_pair_counts + (kb + blk),
                        I32(1),
                        mem_order=prims.MemOrder.RELAXED,
                        mem_scope=prims.MemScope.GPU,
                    )
                off = off + cute.arch.popc(bal)
            if lane == 0:
                (p_sparse_pass_offsets + (r * POS + pp + 1)).store(off)


qsa_bwd_partition_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _part_host(
    idx: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    key_pair_counts: cute.Tensor,
    qlo: I32,
    RB: I32,
    NP: I32,
    stream: cuda_driver.CUstream,
):
    T = idx.shape[0]
    qsa_bwd_partition_kernel(idx, sparse_blocks, sparse_pass_offsets, key_pair_counts, qlo, I32(T), RB, NP).launch(
        grid=((I32(T) - qlo + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )


SCAN_T = 1024  # the largest CTA, so one CTA scans a whole range's keys with a serial per-thread prefix


@cute.kernel
def qsa_bwd_index_scan_kernel(g_key_pair_counts: cute.Tensor, g_key_slot_offsets: cute.Tensor, g_key_slot_cursors: cute.Tensor, num_dense_queries: I32, T: I32):
    # one CTA per range: key_slot_offsets[key] = pairs of the earlier ranges + exclusive scan of the range's counts, cursors start there
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    sm = cutlass.Array(I32, SCAN_T, space=cutlass.AddressSpace.smem, alignment=16)
    sp = sm.data_ptr()
    p_key_pair_counts = g_key_pair_counts.iterator.raw_ptr()
    p_key_slot_offsets = g_key_slot_offsets.iterator.raw_ptr()
    p_key_slot_cursors = g_key_slot_cursors.iterator.raw_ptr()
    q0 = num_dense_queries + (r << QR_SH)
    q1 = q0 + QR
    if q1 > T:
        q1 = T
    nb = (q1 >> 2) + 1
    kb = _kbase(r, num_dense_queries)
    base = _pairs_before(num_dense_queries, q0)
    ch = (nb + (SCAN_T - 1)) // SCAN_T
    e0 = tidx * ch
    e1 = e0 + ch
    if e1 > nb:
        e1 = nb
    s = I32(0)
    for e in cutlass.range(e0, e1, unroll=1):
        s = s + (p_key_pair_counts + (kb + e)).load()
    (sp + tidx).store(s)
    prims.barrier_cta_sync(0)
    if tidx == 0:
        acc = I32(0)
        for t in cutlass.range(SCAN_T, unroll=1):
            x = (sp + t).load()
            (sp + t).store(acc)
            acc = acc + x
    prims.barrier_cta_sync(0)
    acc = base + (sp + tidx).load()
    for e in cutlass.range(e0, e1, unroll=1):
        (p_key_slot_offsets + (kb + e)).store(acc)
        (p_key_slot_cursors + (kb + e)).store(acc)
        acc = acc + (p_key_pair_counts + (kb + e)).load()


qsa_bwd_index_scan_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _scan_host(
    key_pair_counts: cute.Tensor,
    key_slot_offsets: cute.Tensor,
    key_slot_cursors: cute.Tensor,
    num_dense_queries: I32,
    T: I32,
    NR: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_scan_kernel(key_pair_counts, key_slot_offsets, key_slot_cursors, num_dense_queries, T).launch(
        grid=(NR, 1, 1), block=(SCAN_T, 1, 1), stream=stream
    )


@cute.kernel
def qsa_bwd_index_fill_kernel(
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    g_key_slot_cursors: cute.Tensor,
    g_sparse_block_slots: cute.Tensor,
    g_slot_queries: cute.Tensor,
    qlo: I32,
    T: I32,
    NP: I32,
):
    # one warp per sparse query claims a slot for every pair: sparse_block_slots (aligned with sparse_blocks) -> slot, slot_queries[slot] = query
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = qlo + bid * 8 + tidx // 32
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_key_slot_cursors = g_key_slot_cursors.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_slot_queries = g_slot_queries.iterator.raw_ptr()
    if qi < T:
        r = qi - qlo
        kb = _kbase(r >> QR_SH, qlo)
        tot = (p_sparse_pass_offsets + (r * POS + NP)).load()
        for L in cutlass.range(lane, tot, 32, unroll=1):
            blk = (p_sparse_blocks + (r * PART_W + L)).load()
            pos = prims.atomicrmw("add", p_key_slot_cursors + (kb + blk), I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU)
            (p_sparse_block_slots + (r * PART_W + L)).store(pos)
            (p_slot_queries + pos).store(qi)


qsa_bwd_index_fill_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fill_host(
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
    key_slot_cursors: cute.Tensor,
    sparse_block_slots: cute.Tensor,
    slot_queries: cute.Tensor,
    qlo: I32,
    T: I32,
    NP: I32,
    stream: cuda_driver.CUstream,
):
    qsa_bwd_index_fill_kernel(sparse_blocks, sparse_pass_offsets, key_slot_cursors, sparse_block_slots, slot_queries, qlo, T, NP).launch(
        grid=((T - qlo + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )


# ----------------------------------------------------------------------------- selection bitmap of the dense queries
@cute.kernel
def qsa_bwd_bitmap_kernel(g_idx: cute.Tensor, g_dense_block_bitmap: cute.Tensor, num_dense_queries: I32, NW: I32):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = bid * 8 + tidx // 32
    p_idx = g_idx.iterator.raw_ptr()
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    if qi < num_dense_queries:
        ncb = (qi + 1) >> 2
        nsel = ncb
        if nsel > TOPK:
            nsel = I32(TOPK)
        for m in cutlass.range(lane, nsel, 32, unroll=1):
            b = (p_idx + (qi * W + m * 4)).load() >> 2
            # a malformed id must not set a bit in another query's row or past the bitmap
            ok = b < ncb
            if b < 0:
                ok = False
            if ok:
                prims.red(
                    prims.ReductionOp.OR,
                    prims.ReductionType.B32,
                    p_dense_block_bitmap + (qi * NW + (b >> 5)),
                    I32(1) << (b & 31),
                    mem_order=prims.MemOrder.RELAXED,
                    mem_scope=prims.MemScope.GPU,
                )


qsa_bwd_bitmap_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _bitmap_host(idx: cute.Tensor, dense_block_bitmap: cute.Tensor, num_dense_queries: I32, NW: I32, stream: cuda_driver.CUstream):
    qsa_bwd_bitmap_kernel(idx, dense_block_bitmap, num_dense_queries, NW).launch(grid=((num_dense_queries + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)


FIN_THREADS = 256


@cute.kernel
def qsa_bwd_finalize_kernel(
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_dk: cute.Tensor,
    g_dv: cute.Tensor,
    g_dq: cute.Tensor,
    T: I32,
    num_dense_queries: I32,
    HAS_SPARSE: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
    p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
    p_dq_acc = g_dq_acc.iterator.raw_ptr()
    p_dk = g_dk.iterator.raw_ptr()
    p_dv = g_dv.iterator.raw_ptr()
    p_dq = g_dq.iterator.raw_ptr()
    nitems = T * (HKV * D // 8)
    for it in cutlass.range(bid * FIN_THREADS + tidx, nitems, nb * FIN_THREADS, unroll=1):
        j = it >> 6
        eo = it * 8
        ak = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
        av = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
        if cutlass.const_expr(HAS_SPARSE > 0):
            ak = ak + (p_sparse_dk_acc + eo).load(count=8, alignment=16)
            av = av + (p_sparse_dv_acc + eo).load(count=8, alignment=16)
        if j < num_dense_queries:
            ak = ak + (p_dense_dk_acc + eo).load(count=8, alignment=16)
            av = av + (p_dense_dv_acc + eo).load(count=8, alignment=16)
        (p_dk + eo).store(ak.to(BF16), alignment=16)
        (p_dv + eo).store(av.to(BF16), alignment=16)
    nq8 = num_dense_queries * (HQ * D // 8)
    if cutlass.const_expr(HAS_SPARSE > 0):
        nq8 = T * (HQ * D // 8)
    for it in cutlass.range(bid * FIN_THREADS + tidx, nq8, nb * FIN_THREADS, unroll=1):
        (p_dq + it * 8).store((p_dq_acc + it * 8).load(count=8, alignment=16).to(F32).to(BF16), alignment=16)


qsa_bwd_finalize_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _fin_host(
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    dq_acc: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dq: cute.Tensor,
    num_dense_queries: I32,
    nblocks: I32,
    HAS_SPARSE: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = dk.shape[0]
    qsa_bwd_finalize_kernel(sparse_dk_acc, sparse_dv_acc, dense_dk_acc, dense_dv_acc, dq_acc, dk, dv, dq, I32(T), num_dense_queries, HAS_SPARSE).launch(
        grid=(nblocks, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream
    )


@cute.jit
def _rot(k, off, nk):
    r = k + off
    if r >= nk:
        r = r - nk
    return r


# Key-tile-major path for the queries below num_dense_queries. The two CTAs of a cluster split each group's work by operand, so every
# MMA keeps M=128 and each CTA holds one [128 x 256] fp32 dV or dK accumulator in TMEM across all groups.
DP_TT = 128
DP_G = 4  # queries per group: with 12 heads N = 48, the widest N whose S^T / dP^T, dQ^T and P / dS slices fit the TMEM columns
DP_N = DP_G * GQ
DP_THREADS = 512
DP_KSLAB = DP_TT * 128
DP_QSLAB = DP_N * 128
DP_OFF_T = 0
DP_NE = 2  # early ring (B of the score MMA: Q on rank 0, dO on rank 1), released right after S^T / dP^T
DP_NL = 2  # late ring (B of dV / dK: dO on rank 0, Q on rank 1), held until the backward MMAs of the group
DP_HS = 4 * DP_QSLAB
DP_OFF_E = 2 * DP_KSLAB
DP_OFF_L = DP_OFF_E + DP_NE * DP_HS
DP_CM = 16 * 6 * 128
DP_NX = 3  # rank 0's dQ^T holds an exchange buffer for DQL = 2 more groups, so a third buffer keeps the exchange moving
DP_OFF_P = DP_OFF_L + DP_NL * DP_HS
DP_OFF_DS = DP_OFF_P + DP_NX * DP_CM
DP_ST_W = DP_N * 128
DP_OFF_ST = DP_OFF_DS + DP_NX * DP_CM
DP_SMEM = DP_OFF_ST + 4 * DP_ST_W
DP_TM_SD = 0
DP_TM_DQ = 64
DP_TM_A = 128
DP_TM_KV = 256
DP_TM_PA = (48, 56, 112)  # P^T / dS^T slices reuse the TMEM columns left free after S^T (0-47) and dQ^T (64-111)


@cute.kernel
def qsa_bwd_dense_pair_kernel(
    tma_kd: cutlass.GridConstant[cuda.TensorMap],
    tma_qd: cutlass.GridConstant[cuda.TensorMap],
    tma_dod: cutlass.GridConstant[cuda.TensorMap],
    tma_dq: cutlass.GridConstant[cuda.TensorMap],
    g_k: cute.Tensor,
    g_v: cute.Tensor,
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    g_dense_done_count: cute.Tensor,
    T: I32,
    num_dense_queries: I32,
    NW: I32,
    ntiles: I32,
    U: I32,
    PDL: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32
    rank = bid & 1
    pr = bid >> 1
    hk = pr & 1
    rest = pr >> 1
    part = rest // ntiles
    tile = rest - part * ntiles
    t0 = tile * DP_TT
    Gd = (num_dense_queries + (DP_G - 1)) // DP_G
    g_top = Gd - part * U
    g_lo = t0 // DP_G
    g_bot = g_top - U
    if g_bot > g_lo:
        g_lo = g_bot
    nk = g_top - g_lo
    if nk > 0:
        off = (tile * 3) % nk
        peer = rank ^ 1

        smem = cutlass.Array(I32, DP_SMEM // 4, space=cutlass.AddressSpace.smem, alignment=1024)
        b_tile = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_vt = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_ef = cutlass.Array(cutlass.Int64, DP_NE, space=cutlass.AddressSpace.smem, alignment=8)
        b_ee = cutlass.Array(cutlass.Int64, DP_NE, space=cutlass.AddressSpace.smem, alignment=8)
        b_lf = cutlass.Array(cutlass.Int64, DP_NL, space=cutlass.AddressSpace.smem, alignment=8)
        b_le = cutlass.Array(cutlass.Int64, DP_NL, space=cutlass.AddressSpace.smem, alignment=8)
        b_sf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_sr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_pf = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_pe = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_xr = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_xe = cutlass.Array(cutlass.Int64, DP_NX, space=cutlass.AddressSpace.smem, alignment=8)
        b_dqf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_dqe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        b_kvd = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        sbase = smem.data_ptr()
        if warp == 0:
            if prims.elect_sync():
                prims.prefetch_tensormap(tma_kd.get_ptr())
                prims.prefetch_tensormap(tma_qd.get_ptr())
                prims.prefetch_tensormap(tma_dod.get_ptr())
                prims.prefetch_tensormap(tma_dq.get_ptr())
                prims.mbarrier_init(b_tile, 1)
                prims.mbarrier_init(b_vt, 8)
                for s in cutlass.range_constexpr(DP_NE):
                    prims.mbarrier_init(b_ef.subview(s), 1)
                    prims.mbarrier_init(b_ee.subview(s), 1)
                for s in cutlass.range_constexpr(DP_NL):
                    prims.mbarrier_init(b_lf.subview(s), 1)
                    prims.mbarrier_init(b_le.subview(s), 1)
                prims.mbarrier_init(b_sf, 1)
                prims.mbarrier_init(b_sr, 8)
                for s in cutlass.range_constexpr(DP_NX):
                    prims.mbarrier_init(b_pf.subview(s), 8)
                    prims.mbarrier_init(b_pe.subview(s), 1)
                    prims.mbarrier_init(b_xr.subview(s), 1)
                    prims.mbarrier_init(b_xe.subview(s), 1)
                prims.fence_mbarrier_init()
                # receiver-armed exchange: each rank arms its own "received" barriers for the first DP_NX groups
                for s in cutlass.range_constexpr(DP_NX):
                    if s < nk:
                        prims.mbarrier_arrive_expect_tx(b_xr.subview(s), DP_CM)
                prims.mbarrier_init(b_dqf, 1)
                prims.mbarrier_init(b_dqe, 4)
                prims.mbarrier_init(b_kvd, 1)
        if warp == 1:
            prims.tcgen05_alloc(tmem_slot, 512, group="cta_1")
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        prims.fence_mbarrier_init()
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)
        # the peer's barriers must be initialised before any remote arrive / bulk copy
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        tmem_raw = tmem_slot.load()
        tmem_col0 = tmem_raw & I32(0xFFFF)
        tmem_row0 = tmem_raw >> I32(16)
        p_lse = g_lse.iterator.raw_ptr()
        p_delta = g_delta.iterator.raw_ptr()
        p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
        p_dense_dk_acc = g_dense_dk_acc.iterator.raw_ptr()
        p_dense_dv_acc = g_dense_dv_acc.iterator.raw_ptr()
        p_k = g_k.iterator.raw_ptr()
        p_v = g_v.iterator.raw_ptr()

        if warp == 0:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(b_tile, 2 * DP_KSLAB)
                for s in cutlass.range_constexpr(2):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sbase + (DP_OFF_T + s * DP_KSLAB) // 4, tma_kd.get_ptr(), ((rank * 2 + s) * 64, hk, t0), b_tile
                    )
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                se = k % DP_NE
                _wait(b_ee.subview(se), ((k // DP_NE) & 1) ^ 1)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_ef.subview(se), DP_HS)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_qd.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_ef.subview(se)
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_dod.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_ef.subview(se)
                            )
        elif warp == 3:
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                sl = k % DP_NL
                _wait(b_le.subview(sl), ((k // DP_NL) & 1) ^ 1)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_lf.subview(sl), DP_HS)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_dod.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_lf.subview(sl)
                            )
                        else:
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_qd.get_ptr(), (s * 64, hk * GQ, g * DP_G), b_lf.subview(sl)
                            )
        elif warp == 1:
            idesc_s = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=0, b_major=0)
            idesc_q = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=1, b_major=1)
            idesc_kv = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=256, m_dim=128, a_major=0, b_major=1)
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            SWN = prims.Tcgen05SmemSwizzle.NONE
            d_kmn = _desc(sbase + DP_OFF_T // 4, DP_KSLAB, 1024, SW128)
            d_e = _desc(sbase + DP_OFF_E // 4, 16, 1024, SW128)
            d_lmn = _desc(sbase + DP_OFF_L // 4, DP_QSLAB, 1024, SW128)
            d_dsb = _desc(sbase + DP_OFF_DS // 4, 768, 128, SWN)
            t_sd = prims.make_tmem_ptr(tmem_raw + DP_TM_SD, F32)
            t_dq = prims.make_tmem_ptr(tmem_raw + DP_TM_DQ, F32)
            t_kv = prims.make_tmem_ptr(tmem_raw + DP_TM_KV, F32)
            _wait(b_tile, 0)
            _wait(b_vt, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            # rank 0 issues dQ^T(k - DQL) after the backward MMAs of k - 1, so the dS round trip through rank 1
            # overlaps two groups of rank-0 work
            DQL = 2
            for k in cutlass.range(nk + DQL, unroll=1):
                se = k % DP_NE
                if k < nk:
                    if k >= 1:
                        _wait(b_sr, (k - 1) & 1)
                    _wait(b_ef.subview(se), (k // DP_NE) & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    qoff = se * DP_HS
                    if prims.elect_sync():
                        for ks in cutlass.range_constexpr(16):
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F16,
                                prims.CTAGroup.CTA_1,
                                t_sd,
                                prims.make_tmem_ptr(tmem_raw + (DP_TM_A + ks * 8), BF16),
                                d_e.advance_start_address(qoff + (ks >> 2) * DP_QSLAB + (ks & 3) * 32),
                                idesc_s,
                                ks > 0,
                            )
                        prims.tcgen05_commit(b_sf, group=prims.CTAGroup.CTA_1)
                        prims.tcgen05_commit(b_ee.subview(se), group=prims.CTAGroup.CTA_1)
                if k >= 1 and k <= nk:
                    kb = k - 1
                    sl = kb % DP_NL
                    qob = sl * DP_HS
                    xb = kb % DP_NX
                    acc0 = kb > 0
                    _wait(b_lf.subview(sl), (kb // DP_NL) & 1)
                    _wait(b_pf.subview(xb), (kb // DP_NX) & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    if rank == 0:
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(DP_N // 16):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_kv,
                                    prims.make_tmem_ptr(tmem_raw + DP_TM_PA[kk], BF16),
                                    d_lmn.advance_start_address(qob + kk * 2048),
                                    idesc_kv,
                                    acc0 | (kk > 0),
                                )
                            prims.tcgen05_commit(b_pe.subview(xb), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_le.subview(sl), group=prims.CTAGroup.CTA_1)
                            if kb == nk - 1:
                                prims.tcgen05_commit(b_kvd, group=prims.CTAGroup.CTA_1)
                    else:
                        if kb >= 1:
                            _wait(b_dqe, (kb - 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(DP_N // 16):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_kv,
                                    prims.make_tmem_ptr(tmem_raw + DP_TM_PA[kk], BF16),
                                    d_lmn.advance_start_address(qob + kk * 2048),
                                    idesc_kv,
                                    acc0 | (kk > 0),
                                )
                            for kk in cutlass.range_constexpr(8):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_dq,
                                    d_kmn.advance_start_address(kk * 2048),
                                    d_dsb.advance_start_address(xb * DP_CM + kk * 1536),
                                    idesc_q,
                                    kk > 0,
                                )
                            prims.tcgen05_commit(b_pe.subview(xb), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_le.subview(sl), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            if kb == nk - 1:
                                prims.tcgen05_commit(b_kvd, group=prims.CTAGroup.CTA_1)
                if rank == 0:
                    if k >= DQL:
                        # Rank 0's half of dQ^T for group k - DQL needs the dS that rank 1 sends back.
                        kq = k - DQL
                        xq = kq % DP_NX
                        _wait(b_xr.subview(xq), (kq // DP_NX) & 1)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        if kq + DP_NX < nk:
                            if prims.elect_sync():
                                # arm the buffer for dS(kq + DP_NX). Rank 1 only sends it after P(kq + DP_NX),
                                # which our softmax sends only after the dQ^T MMAs below have completed.
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xq), DP_CM)
                        if kq >= 1:
                            _wait(b_dqe, (kq - 1) & 1)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for kk in cutlass.range_constexpr(8):
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F16,
                                    prims.CTAGroup.CTA_1,
                                    t_dq,
                                    d_kmn.advance_start_address(kk * 2048),
                                    d_dsb.advance_start_address(xq * DP_CM + kk * 1536),
                                    idesc_q,
                                    kk > 0,
                                )
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_xe.subview(xq), group=prims.CTAGroup.CTA_1)
        elif warp >= 4 and warp < 12:
            hf = (warp - 4) >> 2
            qd = warp & 3
            r = qd * 32 + lane
            tok = t0 + r
            row_ok = tok < num_dense_queries
            blk = tok >> 2
            wsel = blk >> 5
            bsel = blk & 31
            c_s = F32(SCALE * LOG2E)
            xoff = (DP_OFF_P + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            xoffd = (DP_OFF_DS + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            # Lane r writes row t0 + r of K (rank 0) or V (rank 1) to TMEM as packed bf16 pairs, the score MMA's A operand.
            vok = tok < T
            asrc = p_k
            if rank == 1:
                asrc = p_v
            vrow = asrc + (tok * HKV + hk) * D
            for c2 in cutlass.range_constexpr(2):
                c4 = hf * 2 + c2
                stw = []
                for u in cutlass.range_constexpr(8):
                    ldv = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
                    if vok:
                        ldv = (vrow + c4 * 64 + u * 8).load(count=8, alignment=16).bitcast(I32)
                    for ee in cutlass.range_constexpr(4):
                        stw.append(ldv[ee])
                prims.tcgen05_st(
                    "32x32b",
                    prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_A + c4 * 32).as_ptr(I32),
                    cutlass.Vector.from_elements(stw, I32),
                )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(b_vt)
            scol = tmem_col0 + DP_TM_SD + hf * 24
            if rank == 0:
                # Selection words and lse load one group ahead to hide their latency.
                g_n = g_top - 1 - _rot(I32(0), off, nk)
                sw_n = []
                av_n = []
                for jj in cutlass.range_constexpr(2):
                    qc = g_n * DP_G + hf * 2 + jj
                    if qc >= num_dense_queries:
                        qc = num_dense_queries - 1
                    sw_n.append((p_dense_block_bitmap + (qc * NW + wsel)).load())
                    ab = qc * HQ + hk * GQ
                    av_n.append(
                        [
                            (p_lse + ab).load(count=4, alignment=16),
                            (p_lse + ab + 4).load(count=4, alignment=16),
                            (p_lse + ab + 8).load(count=4, alignment=16),
                        ]
                    )
                for k in cutlass.range(nk, unroll=1):
                    g = g_top - 1 - _rot(k, off, nk)
                    sw = sw_n
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    sw_n = []
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= num_dense_queries:
                            qc = num_dense_queries - 1
                        sw_n.append((p_dense_block_bitmap + (qc * NW + wsel)).load())
                        ab = qc * HQ + hk * GQ
                        av_n.append(
                            [
                                (p_lse + ab).load(count=4, alignment=16),
                                (p_lse + ab + 4).load(count=4, alignment=16),
                                (p_lse + ab + 8).load(count=4, alignment=16),
                            ]
                        )
                    _wait(b_sf, k & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    opk = []
                    for jj in cutlass.range_constexpr(2):
                        qi = g * DP_G + hf * 2 + jj
                        ncb = (qi + 1) >> 2
                        sel = (sw[jj] >> bsel) & 1
                        valid = row_ok & (qi < num_dense_queries) & (tok <= qi) & ((sel != 0) | (blk == ncb))
                        ow = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh // 4][hh % 4]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            e = cute.math.exp2(x * c_s - a * F32(LOG2E), fastmath=True)
                            pj = F32(0.0)
                            if valid:
                                pj = e
                            ow.append(pj)
                        pk = cutlass.Vector.from_elements(ow, F32).to(BF16).bitcast(I32)
                        for u in cutlass.range_constexpr(GQ // 2):
                            opk.append(pk[u])
                    ovec = cutlass.Vector.from_elements(opk, I32)
                    xb = k % DP_NX
                    if k >= 1:
                        # dV(k-1) has read the TMEM P^T slot
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if k >= DP_NX:
                        # dQ^T(k - DP_NX) has completed: rank 1 has read P(k - DP_NX) (dS(k - DP_NX) arrived) and may receive into its
                        # buffer again, and our dS buffer is free for dS(k)
                        _wait(b_xe.subview(xb), ((k // DP_NX) & 1) ^ 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), ovec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), ovec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), ovec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), ovec[4:12])
                    rb = prims.mapa(b_xr.subview(xb), peer)
                    for nc in cutlass.range_constexpr(3):
                        _st_async16(prims.mapa(sbase + xoff + xb * (DP_CM // 4) + nc * 32, peer), ovec[nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            else:
                g_n = g_top - 1 - _rot(I32(0), off, nk)
                av_n = []
                for jj in cutlass.range_constexpr(2):
                    qc = g_n * DP_G + hf * 2 + jj
                    if qc >= num_dense_queries:
                        qc = num_dense_queries - 1
                    ab = qc * HQ + hk * GQ
                    av_n.append(
                        [
                            (p_delta + ab).load(count=4, alignment=16),
                            (p_delta + ab + 4).load(count=4, alignment=16),
                            (p_delta + ab + 8).load(count=4, alignment=16),
                        ]
                    )
                for k in cutlass.range(nk, unroll=1):
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= num_dense_queries:
                            qc = num_dense_queries - 1
                        ab = qc * HQ + hk * GQ
                        av_n.append(
                            [
                                (p_delta + ab).load(count=4, alignment=16),
                                (p_delta + ab + 4).load(count=4, alignment=16),
                                (p_delta + ab + 8).load(count=4, alignment=16),
                            ]
                        )
                    _wait(b_sf, k & 1)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    xb = k % DP_NX
                    _wait(b_xr.subview(xb), (k // DP_NX) & 1)
                    pin = []
                    for nc in cutlass.range_constexpr(3):
                        pin.append((sbase + xoff + xb * (DP_CM // 4) + nc * 32).load(count=4, alignment=16))
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if k + DP_NX < nk:
                        if warp == 4:
                            if prims.elect_sync():
                                # arm this buffer for P(k + DP_NX): rank 0 sends it only after dS(k) (whose quadrant-0
                                # copy this warp issues after this arm) has been consumed by its dQ^T
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xb), DP_CM)
                    dpk = []
                    for jj in cutlass.range_constexpr(2):
                        dw = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh // 4][hh % 4]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            dw.append(x - a)
                        for hp in cutlass.range_constexpr(GQ // 2):
                            n0 = jj * GQ + 2 * hp
                            pw = pin[n0 // 8][(n0 % 8) // 2]
                            pv = cutlass.Vector.from_elements([pw], I32).bitcast(BF16).to(F32)
                            dpk.append(
                                cutlass.Vector.from_elements([pv[0] * dw[2 * hp] * F32(SCALE), pv[1] * dw[2 * hp + 1] * F32(SCALE)], F32)
                                .to(BF16)
                                .bitcast(I32)[0]
                            )
                    dvec = cutlass.Vector.from_elements(dpk, I32)
                    if k >= 1:
                        # dK / dQ^T(k-1) have read the TMEM dS^T slot (and k - DP_NX this SMEM buffer). The copy of dS(k - DP_NX)
                        # has landed and rank 0's dQ^T(k - DP_NX) has consumed it (both implied by the arrival of P(k)).
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), dvec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), dvec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), dvec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), dvec[4:12])
                    rb = prims.mapa(b_xr.subview(xb), peer)
                    for nc in cutlass.range_constexpr(3):
                        (sbase + xoffd + xb * (DP_CM // 4) + nc * 32).store(dvec[nc * 4 : nc * 4 + 4], alignment=16)
                        _st_async16(prims.mapa(sbase + xoffd + xb * (DP_CM // 4) + nc * 32, peer), dvec[nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            _wait(b_kvd, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            obase = (tok * HKV + hk) * D + hf * 128
            pdst = p_dense_dv_acc
            if rank == 1:
                pdst = p_dense_dk_acc
            for c8 in cutlass.range_constexpr(4):
                vv = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_KV + hf * 128 + c8 * 32).as_ptr(F32), num=32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                if row_ok:
                    for u in cutlass.range_constexpr(8):
                        prims.red(
                            prims.ReductionOp.ADD,
                            prims.ReductionType.F32,
                            pdst + (obase + c8 * 32 + u * 4),
                            vv[u * 4 : u * 4 + 4],
                            mem_order=prims.MemOrder.RELAXED,
                            mem_scope=prims.MemScope.GPU,
                        )
        elif warp >= 12:
            qd = warp - 12
            stg0 = sbase + (DP_OFF_ST + qd * DP_ST_W) // 4
            # rank 0 sends P straight into the peer (st.async) and never touches its own P buffers, so they hold a
            # second staging box per drain warp: the reduce of group k-1 may still be reading while k is staged
            stg1 = sbase + (DP_OFF_P + qd * DP_ST_W) // 4
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                _wait(b_dqf, k & 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                vh0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ).as_ptr(F32), num=32)
                vh1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ + 32).as_ptr(F32), num=DP_N - 32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_dqe)
                # the reduce that last used this staging box must have finished reading it. Rank 0 has two boxes,
                # so one reduce may stay in flight, and rank 1 has one.
                stg = stg0
                if rank == 0:
                    if (k & 1) == 1:
                        stg = stg1
                if rank == 0:
                    prims.cp_async_bulk_wait_group(1, read=True)
                else:
                    prims.cp_async_bulk_wait_group(0, read=True)
                prims.bar_warp_sync(0xFFFFFFFF)
                for n in cutlass.range_constexpr(DP_N):
                    if cutlass.const_expr(n < 32):
                        val = vh0[n]
                    else:
                        val = vh1[n - 32]
                    (stg + n * 32 + lane).store(prims.mov_b32(val, target_type=I32))
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    prims.cp_async_bulk_tensor_reduce(tma_dq.get_ptr(), stg, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GQ, g * DP_G])
                prims.cp_async_bulk_commit_group()
            prims.cp_async_bulk_wait_group(0)

        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        # nobody leaves while the peer may still touch our shared memory
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if warp == 1:
            prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), 512, group="cta_1")

    if cutlass.const_expr(PDL > 0):
        # Only the last dependent CTA waits for the sparse primary. Dense-grid completion then
        # orders the normally launched final combine after both accumulator producers.
        if tidx == 0:
            nb, _, _ = cute.arch.grid_dim()
            old = prims.atomicrmw("add", g_dense_done_count.iterator.raw_ptr(), I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU)
            if old == nb - 1:
                prims.griddepcontrol("wait")


qsa_bwd_dense_pair_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _dpair_host(
    k: cute.Tensor,
    v: cute.Tensor,
    q: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dense_block_bitmap: cute.Tensor,
    dq_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    dense_done_count: cute.Tensor,
    num_dense_queries: I32,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    PDL: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = k.shape[0]
    tma_kd = cuda.create_tensor_map_tiled_from_view(k, box_dims=(DP_TT, 1, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_qd = cuda.create_tensor_map_tiled_from_view(q, box_dims=(DP_G, GQ, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dod = cuda.create_tensor_map_tiled_from_view(do, box_dims=(DP_G, GQ, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dq = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(dq_acc.iterator, cute.make_layout((num_dense_queries, HQ, D), stride=(HQ * D, D, 1))),
        box_dims=(DP_G, GQ, 32),
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    qsa_bwd_dense_pair_kernel(
        tma_kd,
        tma_qd,
        tma_dod,
        tma_dq,
        k,
        v,
        lse,
        delta,
        dense_block_bitmap,
        dense_dk_acc,
        dense_dv_acc,
        dense_done_count,
        I32(T),
        num_dense_queries,
        NW,
        ntiles,
        U,
        PDL,
    ).launch(grid=(nparts * ntiles * 4, 1, 1), block=(DP_THREADS, 1, 1), cluster=(2, 1, 1), stream=stream, use_pdl=(PDL == 1))


# Tuned dense/sparse crossover: below it row i selects 2048 of its i + 1 tokens, dense enough that keeping dK/dV on
# chip beats the record round trip of a (query, block) pair. A sparse remainder under 2048 queries is folded in too.
DENSE_QUERIES_MAX = 20480
KV_PASS = 32768  # KV pass of the query-major kernel (tokens), so one pass's K/V blocks stay L2 resident
# From this length on a key holds few enough queries that single-query stages are common, and skipping their empty
# K-step pays for the per-stage branch.
KEY_SKIP_MIN_T = 196608
DP_U = 320  # tuned query groups per dense work unit, small enough to backfill the key-major grid's tail evenly
DP_U_MAX = 448  # query groups per dense work unit beyond which a larger unit no longer amortizes the key tile load and dK / dV flush
# The kernels address [T, 24, 256] tensors with 32-bit element offsets.
MAX_SEQ_LEN = (2**31 - 1) // (HQ * D)


class LaunchPlan:
    """Launch geometry and caller-workspace layout for one sequence length."""

    def __init__(self, seq_len: int, num_sms: int):
        T = int(seq_len)
        assert 1 <= T <= MAX_SEQ_LEN, f"sequence length {T} outside [1, {MAX_SEQ_LEN}]"
        dense_max = DENSE_QUERIES_MAX
        self.seq_len = T
        self.num_dense_queries = T if T - dense_max < 2048 else dense_max
        self.has_sparse = self.num_dense_queries < T
        # A query at position i has min((i + 1) // 4, 512) selected blocks, so up to T = 2051 every list is the whole
        # causal prefix: the bitmap is all ones and the bitmap kernel has nothing to add.
        self.every_block_selected = T <= W
        self.nw = (self.num_dense_queries + 127) // 128
        self.nsel = ((self.num_dense_queries + 2) * self.nw + 64 + 3) // 4 * 4
        gd = (self.num_dense_queries + DP_G - 1) // DP_G
        self.dense_tiles = (self.num_dense_queries + DP_TT - 1) // DP_TT
        # A (key tile, unit) pair of CTAs walks its query groups serially behind one key tile load and one dK / dV flush.
        # Without a sparse grid to backfill, the unit hands every pair slot one wave of the grid's causal work (key tile t
        # sees the groups from its first token up), so the fixed cost is paid once per slot and the waves stay balanced.
        causal_work = HKV * sum(gd - t * (DP_TT // DP_G) for t in range(self.dense_tiles))
        self.dense_unit = DP_U if self.has_sparse else max(1, min(DP_U_MAX, -(-causal_work // (num_sms // 2))))
        self.dense_parts = (gd + self.dense_unit - 1) // self.dense_unit
        self.pass_tokens = KV_PASS
        self.num_passes = (T + self.pass_tokens - 1) // self.pass_tokens
        assert self.num_passes < POS, f"sequence length {T} needs {self.num_passes} KV passes, the offset rows hold {POS - 1}"
        nsq = T - self.num_dense_queries
        self.sparse_ctas = min(num_sms, nsq)
        # inverted index of the sparse queries: ranges of QR queries, one key per (range, block), one slot per pair
        self.num_ranges = (nsq + QR - 1) // QR if self.has_sparse else 0
        self.num_keys = sum(min(self.num_dense_queries + (r + 1) * QR, T) // 4 + 1 for r in range(self.num_ranges))
        self.num_pairs = 513 * nsq - (T // 4 - self.num_dense_queries // 4) if self.has_sparse else 0
        self.key_name = f"key_skip{1 if T >= KEY_SKIP_MIN_T else 0}"
        self.key_ctas = num_sms
        nz = T * HKV * D if self.has_sparse else 0
        dq_zero_size = self.num_dense_queries * HQ * D  # the query-major kernel initializes the sparse rows of dQ itself
        dense_dkv_size = self.num_dense_queries * HKV * D + 64
        keys = (self.num_keys + 31) // 16 * 16
        prep_vecs = max(T * HQ // PREP_ROWS, nz // 8, (self.nsel + 4) // 4, keys // 4, dq_zero_size // 8, dense_dkv_size // 8)
        self.prep_blocks = min(num_sms * 8, (prep_vecs + 255) // 256)
        self.fin_blocks = min(num_sms * 8, (T * HKV * D // 8 + 255) // 256)
        sp = self.has_sparse
        # Without the sparse path its buffers shrink to nothing, except the dK/dV accumulators, which keep one
        # never-read PAD chunk that both kernels alias.
        self.chunks = [
            ("delta", (T, HQ), torch.float32),
            ("sparse_dk_acc", (nz + PAD,), torch.float32),
            ("sparse_dv_acc", (nz + PAD if sp else 0,), torch.float32),
            ("dense_block_bitmap", (self.nsel + 4,), torch.int32),
            ("dq_acc", (T * HQ * D,), torch.float32),
            ("dense_dk_acc", (dense_dkv_size,), torch.float32),
            ("dense_dv_acc", (dense_dkv_size,), torch.float32),
            ("sparse_blocks", (nsq * PART_W + 16 if sp else 0,), torch.int32),
            ("sparse_pass_offsets", (nsq * POS + 16 if sp else 0,), torch.int32),
            ("sparse_block_slots", (nsq * PART_W + 16 if sp else 0,), torch.int32),
            ("key_pair_counts", (keys if sp else 0,), torch.int32),
            ("key_slot_offsets", (keys if sp else 0,), torch.int32),
            ("key_slot_cursors", (keys if sp else 0,), torch.int32),
            ("slot_queries", ((self.num_pairs + 31) // 16 * 16 if sp else 0,), torch.int32),
            ("pair_records", (self.num_pairs * REC_W + 16 if sp else 0,), torch.int32),
        ]

    def workspace_bytes(self) -> int:
        return sum(ws_align(math.prod(shape) * dtype.itemsize) for _, shape, dtype in self.chunks)

    def carve(self, carver) -> dict:
        """Scratch views from a ``cudnn.api_base.WorkspaceCarver`` over the caller's workspace."""
        scratch = {name: carver.take(math.prod(shape), dtype).view(shape) for name, shape, dtype in self.chunks if math.prod(shape)}
        if not self.has_sparse:
            scratch["sparse_dv_acc"] = scratch["sparse_dk_acc"]
            scratch["key_pair_counts"] = scratch["dense_block_bitmap"][self.nsel :]
        return scratch


_COMPILED = {}


def _fake(dtype, shape, sym, align=16):
    full = tuple(sym if s is None else s for s in shape)
    order = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, full, stride_order=order, assumed_align=align)


def compile_kernels(plan: LaunchPlan) -> dict:
    """Compile the launch sequence of the plan's path once per process.

    Every extent is symbolic, so one set of artifacts serves all sequence lengths of a path.
    """
    if plan.has_sparse:
        names = ("prep", "bitmap", "part", "scan", "fill", "query", plan.key_name, "dense_pdl", "fin_sparse")
    else:
        names = ("prep", "bitmap", "dense", "fin")
    missing = [name for name in names if name not in _COMPILED]
    if missing:
        sym_t = cute.sym_int64()
        f_q = _fake(BF16, (None, HQ, D), sym_t)
        f_kv = _fake(BF16, (None, HKV, D), sym_t)
        f_idx = _fake(I32, (None, W), sym_t, align=4)
        f_lse = _fake(F32, (None, HQ), sym_t)
        f_h = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dq_acc = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dense_dkv = _fake(F32, (None,), cute.sym_int64(divisibility=8))

        def f_i():
            return _fake(I32, (None,), cute.sym_int64(divisibility=4))

        stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        dense_fakes = (f_kv, f_kv, f_q, f_q, f_lse, f_lse, f_i(), f_dq_acc, f_dense_dkv, f_dense_dkv, f_i(), I32(1), I32(1), I32(1), I32(1), I32(1))
        fin_fakes = (f_h, f_h, f_dense_dkv, f_dense_dkv, f_dq_acc, f_kv, f_kv, f_q, I32(1), I32(1))
        builders = {
            "prep": lambda: (_prep_host, f_q, f_q, f_lse, f_h, f_h, f_i(), f_i(), f_dq_acc, f_dense_dkv, f_dense_dkv, I32(1), I32(0), stream),
            "bitmap": lambda: (_bitmap_host, f_idx, f_i(), I32(1), I32(1), stream),
            "part": lambda: (_part_host, f_idx, f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), stream),
            "scan": lambda: (_scan_host, f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), stream),
            "fill": lambda: (_fill_host, f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), stream),
            "query": lambda: (_sparse_query_host, f_q, f_kv, f_kv, f_q, f_lse, f_lse, f_dq_acc, f_i(), f_i(), f_i(), f_i(), I32(1), I32(0), I32(1), stream),
            "dense": lambda: (_dpair_host, *dense_fakes, 0, stream),
            "dense_pdl": lambda: (_dpair_host, *dense_fakes, 1, stream),
            "fin": lambda: (_fin_host, *fin_fakes, 0, stream),
            "fin_sparse": lambda: (_fin_host, *fin_fakes, 1, stream),
        }
        for skip in (0, 1):
            builders[f"key_skip{skip}"] = lambda skip=skip: (
                _sparse_key_host,
                f_q,
                f_q,
                f_i(),
                f_i(),
                f_i(),
                f_i(),
                f_h,
                f_h,
                I32(1),
                I32(1),
                I32(1),
                skip,
                stream,
            )
        for name in missing:
            _COMPILED[name] = cute.compile(*builders[name](), options="--enable-tvm-ffi")
    return {name: _COMPILED[name] for name in names}


def launch(plan: LaunchPlan, fns: dict, q, k, v, out, dout, lse, topk_idxs, dq, dk, dv, scratch: dict, stream) -> None:
    """Enqueue the launch sequence on ``stream``. Inputs, outputs and scratch are caller-owned."""
    s = scratch
    dense_done_count = s["dense_block_bitmap"][plan.nsel :]
    dq_acc_dense = s["dq_acc"][: plan.num_dense_queries * HQ * D]
    fns["prep"](
        out,
        dout,
        s["delta"],
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_block_bitmap"],
        s["key_pair_counts"],
        dq_acc_dense,
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        plan.prep_blocks,
        I32(-1) if plan.every_block_selected else I32(0),
        stream,
    )
    if plan.has_sparse:
        fns["part"](
            topk_idxs,
            s["sparse_blocks"],
            s["sparse_pass_offsets"],
            s["key_pair_counts"],
            plan.num_dense_queries,
            plan.pass_tokens // 4,
            plan.num_passes,
            stream,
        )
        fns["scan"](s["key_pair_counts"], s["key_slot_offsets"], s["key_slot_cursors"], plan.num_dense_queries, plan.seq_len, plan.num_ranges, stream)
        fns["fill"](
            s["sparse_blocks"],
            s["sparse_pass_offsets"],
            s["key_slot_cursors"],
            s["sparse_block_slots"],
            s["slot_queries"],
            plan.num_dense_queries,
            plan.seq_len,
            plan.num_passes,
            stream,
        )
    if not plan.every_block_selected:
        fns["bitmap"](topk_idxs, s["dense_block_bitmap"], plan.num_dense_queries, plan.nw, stream)
    dense_args = (
        k,
        v,
        q,
        dout,
        lse,
        s["delta"],
        s["dense_block_bitmap"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        dense_done_count,
        plan.num_dense_queries,
        plan.nw,
        plan.dense_tiles,
        plan.dense_unit,
        plan.dense_parts,
    )
    fin_args = (
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        s["dq_acc"],
        dk,
        dv,
        dq,
        plan.num_dense_queries,
        plan.fin_blocks,
        stream,
    )
    if plan.has_sparse:
        fns["query"](
            q,
            k,
            v,
            dout,
            lse,
            s["delta"],
            s["dq_acc"],
            s["sparse_blocks"],
            s["sparse_pass_offsets"],
            s["sparse_block_slots"],
            s["pair_records"],
            plan.sparse_ctas,
            plan.num_dense_queries,
            plan.num_passes,
            stream,
        )
        # The key-major grid is a normal launch, so the PDL-launched dense grid sees every index table and record. It
        # holds every SM until its items run out, and the dense grid backfills the tail.
        fns[plan.key_name](
            q,
            dout,
            s["pair_records"],
            s["key_slot_offsets"],
            s["key_pair_counts"],
            s["slot_queries"],
            s["sparse_dk_acc"],
            s["sparse_dv_acc"],
            plan.num_dense_queries,
            plan.num_keys,
            plan.key_ctas,
            stream,
        )
        fns["dense_pdl"](*dense_args, stream)
        fns["fin_sparse"](*fin_args)
    else:
        fns["dense"](*dense_args, stream)
        fns["fin"](*fin_args)
