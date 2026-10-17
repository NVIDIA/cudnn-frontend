# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Sparse Attention (QSA) backward for SM100, written with the CuTe DSL primitives."""

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
# Blocks per chunk fill 8 * CHB of the 128 MMA rows, and 13 is the largest count whose two K stages and one V stage
# fit in shared memory. Rows past 8 * CHB alias the next region, which for V includes fp32 staging that may decode as NaN.
CHB = 13


OFF_Q = 0  # [4 slabs][24 head rows x 128 B] with a 128-byte swizzle (SW128), rows 0-11 for KV head 0
OFF_DO = 12288
QSLAB = 3072
OFF_K = 24576  # MMA row = 8 * block + 2 * token + head, so one block covers both KV heads
BLK_SMEM = 4096  # a gathered block is exactly [4 slabs][8 rows x 128 B] of SW128 atoms, so its TMA box needs no repack
SLAB = 1024
K_STAGE = CHB * BLK_SMEM  # blocks sit at the 8-row-group stride of the MMA operand
OFF_V = OFF_K + 2 * K_STAGE
OFF_DS = OFF_V + K_STAGE
DS_BYTES = 6144
NDRAIN = 8
EP_BYTES = 4096
OFF_EP = OFF_DS + DS_BYTES
SMEM_BYTES = OFF_EP + NDRAIN * EP_BYTES

TM_S = 0
TM_DP = 32
TM_PT = 64
TM_DST = 96
TM_DQ = 128
TM_SL = 192
NSL = 5  # the most 64-column dV/dK slices that fit in the 512 TMEM columns after TM_SL
L2_EVICT_FIRST = 0x12F0000000000000  # createpolicy encodings of the L2 evict-first and evict-last priorities
L2_EVICT_LAST = 0x14F0000000000000
TM_COLS = 512

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


@cute.jit
def _bwd_slice(j, cgb, pb, b_le, b_lf, tmem_raw, d_do_mn, d_q_mn, idesc_kv):
    gs = cgb * 8 + j
    slot = gs % NSL
    _wait(b_le.subview(slot), ((gs // NSL) & 1) ^ 1)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    d_sl = prims.make_tmem_ptr(tmem_raw + (TM_SL + slot * 64), F32)
    if prims.elect_sync():
        for k2 in cutlass.range_constexpr(2):
            if cutlass.const_expr(j < 4):
                a_op = prims.make_tmem_ptr(tmem_raw + (TM_PT + pb * 16 + k2 * 8), BF16)
                b_op = d_do_mn.advance_start_address(j * QSLAB + k2 * 2048)
            else:
                a_op = prims.make_tmem_ptr(tmem_raw + (TM_DST + pb * 16 + k2 * 8), BF16)
                b_op = d_q_mn.advance_start_address((j - 4) * QSLAB + k2 * 2048)
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                d_sl,
                a_op,
                b_op,
                idesc_kv,
                k2 > 0,
            )
        prims.tcgen05_commit(b_lf.subview(slot), group=prims.CTAGroup.CTA_1)


# Query-major path: both KV heads share each M=128 MMA through block-diagonal masking, so every gathered K/V
# block serves all 24 query heads with one load.
@cute.kernel
def qsa_bwd_sparse_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_k: cutlass.GridConstant[cuda.TensorMap],
    tma_v: cutlass.GridConstant[cuda.TensorMap],
    tma_sparse_dv_acc: cutlass.GridConstant[cuda.TensorMap],
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dq: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_sparse_pass_offsets: cute.Tensor,
    T: I32,
    nctas: I32,
    qlo: I32,
    NP: I32,
):
    # K/V blocks are streamed once per item, while the dK/dV accumulators of the pass are revisited
    hint_ld = cutlass.Int64(L2_EVICT_FIRST)
    hint_red = cutlass.Int64(L2_EVICT_LAST)
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    # The dense pair grid is a programmatic dependent launch (PDL) that backfills SMs released by sparse load imbalance.
    prims.griddepcontrol("launch_dependents")

    smem = cutlass.Array(I32, SMEM_BYTES // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_qf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_qe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_kf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_ke = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_vf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_ve = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_sf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_pf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_pe = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_lf = cutlass.Array(cutlass.Int64, NSL, space=cutlass.AddressSpace.smem, alignment=8)
    b_le = cutlass.Array(cutlass.Int64, NSL, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_dqe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    # dQ^T multiplies the K rows past 8 * CHB by zero dS, so they must be finite and the region is zeroed before any load.
    # The V rows past 8 * CHB need no such care because the softmax assigns zero to masked P and dS instead of scaling them.
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
            prims.prefetch_tensormap(tma_sparse_dv_acc.get_ptr())
            prims.mbarrier_init(b_qf, 1)
            prims.mbarrier_init(b_qe, 1)
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_kf.subview(s), 1)
                prims.mbarrier_init(b_ke.subview(s), 1)
                prims.mbarrier_init(b_pf.subview(s), 4)
                prims.mbarrier_init(b_pe.subview(s), 1)
            prims.mbarrier_init(b_sf, 1)
            prims.mbarrier_init(b_vf, 1)
            prims.mbarrier_init(b_ve, 1)
            for s in cutlass.range_constexpr(NSL):
                prims.mbarrier_init(b_lf.subview(s), 1)
                prims.mbarrier_init(b_le.subview(s), 4)
            prims.mbarrier_init(b_dqf, 1)
            prims.mbarrier_init(b_dqe, 4)
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
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()

    # Every role fits the 128-register budget of a 512-thread CTA, so no register reallocation is needed.
    wg = warp // 4
    if wg == 0:
        # The TMA unit zero-fills tokens past T, which keeps the masked rows of a tail block finite.
        if warp == 0:
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qq in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    qi = qq
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        ncb, nsel, ntail, _nb = _qmeta(qi)
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
                                )
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + (OFF_DO + s * QSLAB) // 4,
                                    tma_do.get_ptr(),
                                    (s * 64, 0, qi),
                                    b_qf,
                                )
                        # Index entries load one chunk ahead, clamped to a valid column so the load stays in bounds.
                        m0 = lane
                        if m0 >= nblk:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            st = cgc & 1
                            kpar = (cgc >> 1) & 1
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
                                    l2_cache_hint=hint_ld,
                                )
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
                                    l2_cache_hint=hint_ld,
                                )
                        gc = gc + nchunks
                        nq = nq + 1

        # Chunk c's dV/dK slices are issued with chunk c+1's score MMAs, so softmax(c+1) overlaps the drain of chunk c.
        # dQ(c) is issued before S(c+1), so b_sf(c+1) implies dQ(c) has read the single-buffered dS^T SMEM copy.
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
            idesc_kv = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=64,
                m_dim=128,
                a_major=0,
                b_major=1,
            )
            idesc_q = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=24,
                m_dim=128,
                a_major=1,
                b_major=1,
            )
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            SWN = prims.Tcgen05SmemSwizzle.NONE
            # K/V tile, K-major SW128: 8-row atoms (one block) at 4 KB, slabs of 64 dims at 1 KB inside a block
            d_k_kmaj = _desc(sbase + OFF_K // 4, 16, BLK_SMEM, SW128)
            d_v_kmaj = _desc(sbase + OFF_V // 4, 16, BLK_SMEM, SW128)
            # K tile read as K^T (MN-major SW128): leading byte offset = next 64-dim slab (1 KB), stride byte offset = next 8 rows
            d_k_mn = _desc(sbase + OFF_K // 4, SLAB, BLK_SMEM, SW128)
            # Q/dO as B of S^T/dP^T (K-major SW128, 24 rows) and as B of dK/dV (MN-major SW128)
            d_q_kmaj = _desc(sbase + OFF_Q // 4, 16, 1024, SW128)
            d_do_kmaj = _desc(sbase + OFF_DO // 4, 16, 1024, SW128)
            d_q_mn = _desc(sbase + OFF_Q // 4, QSLAB, 1024, SW128)
            d_do_mn = _desc(sbase + OFF_DO // 4, QSLAB, 1024, SW128)
            # dS^T as B of dQ^T (MN-major, no swizzle): leading byte offset = next 8 K-rows, stride byte offset = next 8 columns
            d_ds_mn = _desc(sbase + OFF_DS // 4, 384, 128, SWN)

            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qq in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    qi = qq
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        ncb, nsel, ntail, _nb = _qmeta(qi)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qf, nq & 1)
                        for c in cutlass.range(nchunks + 1, unroll=1):
                            # Slots j = 0..5 issue dQ(c-1), S(c) and dP(c) between the eight dV/dK slices of chunk c-1, so at most 8 small
                            # MMAs queue ahead of a slice. S/dP are single buffered, so S(c) waits for b_pf(c-1).
                            cgc = gc + c
                            st = cgc & 1
                            cgb = cgc - 1  # chunk whose backward half is issued this iteration
                            pb = cgb & 1
                            if c >= 1:
                                _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            if c < nchunks:
                                _wait(b_kf.subview(st), (cgc >> 1) & 1)
                                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            dk_st = d_k_kmaj.advance_start_address(st * K_STAGE)
                            dk_mn = d_k_mn.advance_start_address(pb * K_STAGE)
                            acc_q = c > 1  # the first chunk of a query initialises the dQ accumulator
                            for j in cutlass.range_constexpr(8):
                                if c >= 1:
                                    _bwd_slice(j, cgb, pb, b_le, b_lf, tmem_raw, d_do_mn, d_q_mn, idesc_kv)
                                if cutlass.const_expr(j >= 2 and j < 4):
                                    if c < nchunks:
                                        for kk in cutlass.range(8, unroll=1):
                                            k16 = kk + 8 * (j - 2)
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + TM_S, F32),
                                                    dk_st.advance_start_address((k16 >> 2) * SLAB + (k16 & 3) * 32),
                                                    d_q_kmaj.advance_start_address((k16 >> 2) * QSLAB + (k16 & 3) * 32),
                                                    idesc_s,
                                                    k16 > 0,
                                                )
                                elif cutlass.const_expr(j < 2):
                                    if c >= 1:
                                        if cutlass.const_expr(j == 0):
                                            if c == 1:
                                                _wait(b_dqe, (nq & 1) ^ 1)
                                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                        hh = j
                                        for kk in cutlass.range(8, unroll=1):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                                    dk_mn.advance_start_address(hh * 2 * SLAB + kk * 2 * BLK_SMEM),
                                                    d_ds_mn.advance_start_address(kk * 768),
                                                    idesc_q,
                                                    acc_q | (kk > 0),
                                                )
                                        if cutlass.const_expr(j == 1):
                                            if prims.elect_sync():
                                                prims.tcgen05_commit(b_ke.subview(pb), group=prims.CTAGroup.CTA_1)
                                                prims.tcgen05_commit(b_ve, group=prims.CTAGroup.CTA_1)
                                                if c == nchunks:
                                                    prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                                elif cutlass.const_expr(j < 6):
                                    if c < nchunks:
                                        if cutlass.const_expr(j == 4):
                                            _wait(b_vf, cgc & 1)
                                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                        for kk in cutlass.range(8, unroll=1):
                                            k16 = kk + 8 * (j - 4)
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + TM_DP, F32),
                                                    d_v_kmaj.advance_start_address((k16 >> 2) * SLAB + (k16 & 3) * 32),
                                                    d_do_kmaj.advance_start_address((k16 >> 2) * QSLAB + (k16 & 3) * 32),
                                                    idesc_s,
                                                    k16 > 0,
                                                )
                                        if cutlass.const_expr(j == 5):
                                            if prims.elect_sync():
                                                prims.tcgen05_commit(b_sf, group=prims.CTAGroup.CTA_1)
                            if c >= 1:
                                if prims.elect_sync():
                                    prims.tcgen05_commit(b_pe.subview(pb), group=prims.CTAGroup.CTA_1)
                        if prims.elect_sync():
                            prims.tcgen05_commit(b_qe, group=prims.CTAGroup.CTA_1)
                        gc = gc + nchunks
                        nq = nq + 1

    elif wg == 1:
        if True:
            qd = warp % 4
            row = qd * 32 + lane
            hrow = lane & 1
            trow = (lane & 7) >> 1
            jrow = row >> 3
            c_s = F32(SCALE * LOG2E)
            gc = I32(0)
            nq = I32(0)
            ds_off = (OFF_DS + jrow * 384 + (row & 7) * 16) // 4
            for pp in cutlass.range(NP, unroll=1):
                for qq in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    qi = qq
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        ncb, nsel, ntail, _nb = _qmeta(qi)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        lbase = qi * HQ + hrow * GQ
                        lse_a = (p_lse + lbase).load(count=4, alignment=16)
                        lse_b = (p_lse + lbase + 4).load(count=4, alignment=16)
                        lse_c = (p_lse + lbase + 8).load(count=4, alignment=16)
                        del_a = (p_delta + lbase).load(count=4, alignment=16)
                        del_b = (p_delta + lbase + 4).load(count=4, alignment=16)
                        del_c = (p_delta + lbase + 8).load(count=4, alignment=16)
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            st = cgc & 1
                            par = (cgc >> 1) & 1
                            _wait(b_sf, cgc & 1)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            t_s = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_S).as_ptr(F32)
                            t_dp = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DP).as_ptr(F32)
                            sv = prims.tcgen05_ld("32x32b", t_s, num=32)
                            dpv = prims.tcgen05_ld("32x32b", t_dp, num=32)
                            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

                            m = c * CHB + jrow
                            valid = m < nblk
                            if jrow >= CHB:
                                valid = False
                            if valid:
                                if (lrow + m).load() == ncb:
                                    valid = trow < ntail
                            pw = []
                            dw = []
                            for j in cutlass.range_constexpr(GQ):
                                if cutlass.const_expr(j < 4):
                                    lj = lse_a[j]
                                    ej = del_a[j]
                                elif cutlass.const_expr(j < 8):
                                    lj = lse_b[j - 4]
                                    ej = del_b[j - 4]
                                else:
                                    lj = lse_c[j - 8]
                                    ej = del_c[j - 8]
                                s0 = sv[j]
                                d0 = dpv[j]
                                if hrow == 1:
                                    s0 = sv[GQ + j]
                                    d0 = dpv[GQ + j]
                                pj = F32(0.0)
                                dsj = F32(0.0)
                                if valid:
                                    pj = cute.math.exp2(s0 * c_s - lj * F32(LOG2E), fastmath=True)
                                    dsj = pj * (d0 - ej) * F32(SCALE)
                                pw.append(pj)
                                dw.append(dsj)
                            # Zeroing the other KV head's columns makes the shared dV/dK MMAs block-diagonal.
                            zero = F32(0.0)
                            pfull = []
                            dfull = []
                            for n in cutlass.range_constexpr(32):
                                if cutlass.const_expr(n < GQ):
                                    pv = pw[n]
                                    dv_ = dw[n]
                                    if hrow == 1:
                                        pv = zero
                                        dv_ = zero
                                elif cutlass.const_expr(n < 2 * GQ):
                                    pv = zero
                                    dv_ = zero
                                    if hrow == 1:
                                        pv = pw[n - GQ]
                                        dv_ = dw[n - GQ]
                                else:
                                    pv = zero
                                    dv_ = zero
                                pfull.append(pv)
                                dfull.append(dv_)
                            pvec = cutlass.Vector.from_elements(pfull, F32).to(BF16).bitcast(I32)
                            dvec = cutlass.Vector.from_elements(dfull, F32).to(BF16).bitcast(I32)

                            # Buffer st was last read by the dV/dK MMAs of chunk c - 2.
                            _wait(b_pe.subview(st), par ^ 1)
                            t_pt = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_PT + st * 16).as_ptr(I32)
                            t_dst = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DST + st * 16).as_ptr(I32)
                            prims.tcgen05_st("32x32b", t_pt, pvec)
                            prims.tcgen05_st("32x32b", t_dst, dvec)
                            # dQ^T reads dS^T as an SMEM B operand, so the row is also stored as MN-major core matrices.
                            for nc in cutlass.range_constexpr(3):
                                (sbase + ds_off + nc * 32).store(dvec[nc * 4 : nc * 4 + 4], alignment=16)
                            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if prims.elect_sync():
                                prims.mbarrier_arrive(b_pf.subview(st))

                        # A query's dQ is split over KV passes, so each pass adds its part with fp32 reductions.
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
                            for n in cutlass.range_constexpr(HQ):
                                prims.red(
                                    prims.ReductionOp.ADD,
                                    prims.ReductionType.F32,
                                    qb + n * D,
                                    dqv[hh][n],
                                    mem_order=prims.MemOrder.RELAXED,
                                    mem_scope=prims.MemScope.GPU,
                                )
                        gc = gc + nchunks
                        nq = nq + 1

    else:
        # Slices stay fp32 all the way to the global reduction. A 64-dim row is two 128 B pages that reuse the four 1 KB
        # block boxes of the 4 KB per-warp staging area, which keeps the CTA within the shared memory limit.
        if True:
            grp = (warp - 8) >> 2
            qd = warp % 4
            g_l = lane >> 3
            y_l = lane & 7
            ep = sbase + (OFF_EP + (warp - 8) * EP_BYTES) // 4
            sofs = [(g_l * 1024 + y_l * 128 + ((y_l ^ cc) * 16)) // 4 for cc in range(8)]
            gc = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for qq in cutlass.range(qlo + bid, T, nctas, unroll=1):
                    qi = qq
                    prow = p_sparse_pass_offsets + ((qi - qlo) * POS + pp)
                    o0 = prow.load()
                    nblk = (prow + 1).load() - o0
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((qi - qlo) * PART_W + o0)
                        ncb, nsel, ntail, _nb = _qmeta(qi)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        # Index entries load one chunk ahead, clamped to a valid column so the load stays in bounds.
                        nx = []
                        for g in cutlass.range_constexpr(4):
                            m = qd * 4 + g
                            if m >= nblk:
                                m = I32(0)
                            nx.append((lrow + m).load())
                        nx0 = nx[0]
                        nx1 = nx[1]
                        nx2 = nx[2]
                        nx3 = nx[3]
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            cur = [nx0, nx1, nx2, nx3]
                            if c + 1 < nchunks:
                                nn = []
                                for g in cutlass.range_constexpr(4):
                                    m = (c + 1) * CHB + qd * 4 + g
                                    if m >= nblk:
                                        m = I32(0)
                                    nn.append((lrow + m).load())
                                nx0 = nn[0]
                                nx1 = nn[1]
                                nx2 = nn[2]
                                nx3 = nn[3]
                            grow = []
                            bok = []
                            for g in cutlass.range_constexpr(4):
                                m = c * CHB + qd * 4 + g
                                blk = cur[g]
                                ok = m < nblk
                                if qd * 4 + g >= CHB:
                                    ok = False
                                grow.append(blk * 8)
                                bok.append(ok)
                            for jj in cutlass.range_constexpr(4):
                                dblk = 2 * (jj % 2) + grp
                                gs = cgc * 8 + 2 * jj + grp
                                slot = gs % NSL
                                _wait(b_lf.subview(slot), (gs // NSL) & 1)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                                ta = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_SL + slot * 64)
                                va = prims.tcgen05_ld("32x32b", ta.as_ptr(F32), num=32)
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                ta1 = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_SL + slot * 64 + 32)
                                vb = prims.tcgen05_ld("32x32b", ta1.as_ptr(F32), num=32)
                                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_le.subview(slot))
                                for pg in cutlass.range_constexpr(2):
                                    # The previous four reductions must have read the four 1KB staging boxes.
                                    if cutlass.const_expr(jj >= 2):
                                        # dK goes through coalesced red.v4 from a line-major re-read of the staged page, so dK and dV
                                        # egress run on the load/store unit and the TMA unit in parallel.
                                        prims.cp_async_bulk_wait_group(0, read=True)
                                        prims.bar_warp_sync(0xFFFFFFFF)
                                        for cc in cutlass.range_constexpr(8):
                                            if cutlass.const_expr(pg == 0):
                                                w0 = va[4 * cc : 4 * cc + 4].bitcast(I32)
                                            else:
                                                w0 = vb[4 * cc : 4 * cc + 4].bitcast(I32)
                                            (ep + sofs[cc]).store(w0, alignment=16)
                                        prims.bar_warp_sync(0xFFFFFFFF)
                                        cl = lane & 7
                                        for kk in cutlass.range_constexpr(8):
                                            yy = ((kk & 1) << 2) + (lane >> 3)
                                            if bok[kk >> 1]:
                                                v4 = (ep + ((kk >> 1) * 256 + yy * 32 + ((yy ^ cl) * 4))).load(count=4, alignment=16).bitcast(F32)
                                                goff = (grow[kk >> 1] + yy) * D + (dblk * 64 + pg * 32) + cl * 4
                                                prims.red(
                                                    prims.ReductionOp.ADD, prims.ReductionType.F32, p_sparse_dk_acc + goff, v4, mem_order=prims.MemOrder.RELAXED
                                                )
                                        prims.bar_warp_sync(0xFFFFFFFF)
                                    else:
                                        prims.cp_async_bulk_wait_group(0, read=True)
                                        prims.bar_warp_sync(0xFFFFFFFF)
                                        for cc in cutlass.range_constexpr(8):
                                            if cutlass.const_expr(pg == 0):
                                                w0 = va[4 * cc : 4 * cc + 4].bitcast(I32)
                                            else:
                                                w0 = vb[4 * cc : 4 * cc + 4].bitcast(I32)
                                            (ep + sofs[cc]).store(w0, alignment=16)
                                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                        prims.bar_warp_sync(0xFFFFFFFF)
                                        if lane == 0:
                                            for g in cutlass.range_constexpr(4):
                                                if bok[g]:
                                                    prims.cp_async_bulk_tensor_reduce(
                                                        tma_sparse_dv_acc.get_ptr(),
                                                        ep + g * 256,
                                                        prims.TMARedux.ADD,
                                                        [dblk * 64 + pg * 32, grow[g]],
                                                        l2_cache_hint=hint_red,
                                                    )
                                        prims.cp_async_bulk_commit_group()
                        gc = gc + nchunks
            prims.cp_async_bulk_wait_group(0)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 2:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), TM_COLS, group="cta_1")


qsa_bwd_sparse_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _sparse_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq: cute.Tensor,
    sparse_dk_acc: cute.Tensor,
    sparse_dv_acc: cute.Tensor,
    sparse_blocks: cute.Tensor,
    sparse_pass_offsets: cute.Tensor,
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
    nrows = sparse_dv_acc.shape[0] // D
    h_layout = cute.make_layout((nrows, D), stride=(D, 1))
    tma_sparse_dv_acc = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(sparse_dv_acc.iterator, h_layout),
        box_dims=(8, 32),
        stride_order=(1, 0),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    qsa_bwd_sparse_kernel(
        tma_q, tma_do, tma_k, tma_v, tma_sparse_dv_acc, lse, delta, dq, sparse_dk_acc, sparse_blocks, sparse_pass_offsets, I32(T), nctas, qlo, NP
    ).launch(grid=(nctas, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


PREP_THREADS = 256


@cute.kernel
def qsa_bwd_prep_kernel(
    g_o: cute.Tensor,
    g_do: cute.Tensor,
    g_delta: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_dv_acc: cute.Tensor,
    g_dense_block_bitmap: cute.Tensor,
    g_dq_acc: cute.Tensor,
    g_dense_dk_acc: cute.Tensor,
    g_dense_dv_acc: cute.Tensor,
    T: I32,
    nz: I32,
    nsel: I32,
    n_dq_acc: I32,
    n_dense_dkv: I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    lane = tidx % 32
    wib = tidx // 32
    p_o = g_o.iterator.raw_ptr()
    p_do = g_do.iterator.raw_ptr()
    p_delta = g_delta.iterator.raw_ptr()
    nrows = T * HQ
    for r in cutlass.range(bid * (PREP_THREADS // 32) + wib, nrows, nb * (PREP_THREADS // 32), unroll=1):
        ov = (p_o + r * D + lane * 8).load(count=8, alignment=16).to(F32)
        dv = (p_do + r * D + lane * 8).load(count=8, alignment=16).to(F32)
        acc = F32(0.0)
        for j in cutlass.range_constexpr(8):
            acc = acc + ov[j] * dv[j]
        for sh in cutlass.range_constexpr(5):
            acc = acc + prims.shfl_sync(0xFFFFFFFF, acc, 16 >> sh, 0x1F, prims.Shfl.BFLY)
        if lane == 0:
            (p_delta + r).store(acc)
    p_sparse_dk_acc = g_sparse_dk_acc.iterator.raw_ptr()
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    zf = cutlass.Vector.from_elements([F32(0.0)] * 8, F32)
    zi = cutlass.Vector.from_elements([I32(0)] * 4, I32)
    for it in cutlass.range(bid * PREP_THREADS + tidx, nz // 8, nb * PREP_THREADS, unroll=1):
        (p_sparse_dk_acc + it * 8).store(zf, alignment=16)
        (p_sparse_dv_acc + it * 8).store(zf, alignment=16)
    p_dense_block_bitmap = g_dense_block_bitmap.iterator.raw_ptr()
    for it in cutlass.range(bid * PREP_THREADS + tidx, nsel // 4, nb * PREP_THREADS, unroll=1):
        (p_dense_block_bitmap + it * 4).store(zi, alignment=16)
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
    dq_acc: cute.Tensor,
    dense_dk_acc: cute.Tensor,
    dense_dv_acc: cute.Tensor,
    nblocks: I32,
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
        dq_acc,
        dense_dk_acc,
        dense_dv_acc,
        I32(T),
        I32(nz),
        I32(dense_block_bitmap.shape[0]),
        I32(dq_acc.shape[0]),
        I32(dense_dk_acc.shape[0]),
    ).launch(grid=(nblocks, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)


# Walking (KV pass, query) items pass-major keeps one pass's K/V and fp32 dK/dV accumulators L2 resident. sparse_blocks[r] holds
# query qlo + r's blocks grouped by pass p = block // RB, and sparse_pass_offsets[r, p] .. sparse_pass_offsets[r, p + 1] bounds pass p's segment.
PART_W = 516  # 512 blocks and the tail block, padded to a multiple of 4 ints
POS = 64  # offset row stride, which bounds the pass count NP by POS - 1


@cute.kernel
def qsa_bwd_partition_kernel(g_idx: cute.Tensor, g_sparse_blocks: cute.Tensor, g_sparse_pass_offsets: cute.Tensor, qlo: I32, T: I32, RB: I32, NP: I32):
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    qi = qlo + bid * 8 + tidx // 32
    p_idx = g_idx.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    if qi < T:
        ncb, nsel, ntail, nblk = _qmeta(qi)
        r = qi - qlo
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
                off = off + cute.arch.popc(bal)
            if lane == 0:
                (p_sparse_pass_offsets + (r * POS + pp + 1)).store(off)


qsa_bwd_partition_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _part_host(idx: cute.Tensor, sparse_blocks: cute.Tensor, sparse_pass_offsets: cute.Tensor, qlo: I32, RB: I32, NP: I32, stream: cuda_driver.CUstream):
    T = idx.shape[0]
    qsa_bwd_partition_kernel(idx, sparse_blocks, sparse_pass_offsets, qlo, I32(T), RB, NP).launch(
        grid=((I32(T) - qlo + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
    )


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
DP_G = 4
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
DP_OFF_Q = DP_OFF_L + DP_NL * DP_HS
DP_CM = 16 * 6 * 128
DP_NX = 3  # rank 0's dQ^T holds an exchange buffer for DQL = 2 more groups, so a third buffer keeps the exchange moving
DP_OFF_P = DP_OFF_Q
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


# The dense path computes every visible key of a 128-token tile, the sparse path only selected keys but with fp32 atomics.
# A selected key costs about SPARSE_COST_RATIO visible dense keys, so a query stays dense while it selects that share of its prefix.
SPARSE_COST_RATIO = 12
MAX_DENSE_QUERIES = SPARSE_COST_RATIO * TOPK * 4
KV_PASS = 16384  # sparse kernel KV pass (tokens): one pass's K/V + fp32 dK/dV (~96 MB) stays L2 resident
KV_PASS_MIN_T = 16384  # up to this T the whole sequence already fits one L2-resident pass
DP_U = 320  # tuned query groups per dense work unit, small enough to backfill the sparse grid's tail evenly
DP_U_ALL = (384, 448)  # without a sparse grid to backfill, larger units (for Gd < 4096 / >= 4096) amortize each key tile load
# The kernels address [T, 24, 256] tensors with 32-bit element offsets.
MAX_SEQ_LEN = (2**31 - 1) // (HQ * D)


class LaunchPlan:
    """Launch geometry and caller-workspace layout for one sequence length."""

    def __init__(self, seq_len: int, num_sms: int):
        T = int(seq_len)
        assert 1 <= T <= MAX_SEQ_LEN, f"sequence length {T} outside [1, {MAX_SEQ_LEN}]"
        self.seq_len = T
        self.num_dense_queries = min(T, MAX_DENSE_QUERIES)
        self.has_sparse = self.num_dense_queries < T
        self.nw = (self.num_dense_queries + 127) // 128
        self.nsel = ((self.num_dense_queries + 2) * self.nw + 64 + 3) // 4 * 4
        gd = (self.num_dense_queries + DP_G - 1) // DP_G
        self.dense_unit = DP_U if self.has_sparse else (DP_U_ALL[1] if gd >= 4096 else DP_U_ALL[0])
        self.dense_tiles = (self.num_dense_queries + DP_TT - 1) // DP_TT
        self.dense_parts = (gd + self.dense_unit - 1) // self.dense_unit
        rt = KV_PASS if T > KV_PASS_MIN_T else T
        self.pass_tokens = max(4, (rt + 3) // 4 * 4)
        self.num_passes = (T + self.pass_tokens - 1) // self.pass_tokens
        assert self.num_passes < POS, f"sequence length {T} needs {self.num_passes} KV passes, the offset rows hold {POS - 1}"
        self.sparse_queries = T - self.num_dense_queries
        self.sparse_ctas = min(num_sms, self.sparse_queries)
        nz = T * HKV * D if self.has_sparse else 0
        dq_acc_size = T * HQ * D
        dense_dkv_size = self.num_dense_queries * HKV * D + 64
        self.prep_blocks = min(num_sms * 8, (max(T * HQ, nz // 8, (self.nsel + 4) // 4, dq_acc_size // 8, dense_dkv_size // 8) + 255) // 256)
        self.fin_blocks = min(num_sms * 8, (T * HKV * D // 8 + 255) // 256)
        # Without the sparse path its dK/dV accumulators shrink to one never-read PAD chunk that both kernels alias.
        self.chunks = [
            ("delta", (T, HQ), torch.float32),
            ("sparse_dk_acc", (nz + PAD,), torch.float32),
            ("sparse_dv_acc", (nz + PAD if self.has_sparse else 0,), torch.float32),
            ("dense_block_bitmap", (self.nsel + 4,), torch.int32),
            ("dq_acc", (dq_acc_size,), torch.float32),
            ("dense_dk_acc", (dense_dkv_size,), torch.float32),
            ("dense_dv_acc", (dense_dkv_size,), torch.float32),
            ("sparse_blocks", (self.sparse_queries * PART_W + 16 if self.has_sparse else 0,), torch.int32),
            ("sparse_pass_offsets", (self.sparse_queries * POS + 16 if self.has_sparse else 0,), torch.int32),
        ]

    def workspace_bytes(self) -> int:
        return sum(ws_align(math.prod(shape) * dtype.itemsize) for _, shape, dtype in self.chunks)

    def carve(self, carver) -> dict:
        """Scratch views from a ``cudnn.api_base.WorkspaceCarver`` over the caller's workspace."""
        scratch = {name: carver.take(math.prod(shape), dtype).view(shape) for name, shape, dtype in self.chunks if math.prod(shape)}
        if not self.has_sparse:
            scratch["sparse_dv_acc"] = scratch["sparse_dk_acc"]
        return scratch


_COMPILED = {}


def _fake(dtype, shape, sym, align=16):
    full = tuple(sym if s is None else s for s in shape)
    order = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, full, stride_order=order, assumed_align=align)


def compile_kernels(has_sparse: bool) -> dict:
    """Compile the launch sequence of the dense-only or the dense + sparse path once per process.

    Every extent is symbolic, so one set of artifacts serves all sequence lengths.
    """
    names = ("prep", "bitmap", "dense", "fin") if not has_sparse else ("prep", "bitmap", "part", "sparse", "dense_pdl", "fin_sparse")
    missing = [name for name in names if name not in _COMPILED]
    if missing:
        sym_t = cute.sym_int64()
        f_q = _fake(BF16, (None, HQ, D), sym_t)
        f_kv = _fake(BF16, (None, HKV, D), sym_t)
        f_idx = _fake(I32, (None, W), sym_t, align=4)
        f_lse = _fake(F32, (None, HQ), sym_t)
        f_h = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dense_block_bitmap = _fake(I32, (None,), cute.sym_int64(divisibility=4))
        f_dense_done_count = _fake(I32, (None,), cute.sym_int64(divisibility=4))
        f_dq_acc = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_dense_dkv = _fake(F32, (None,), cute.sym_int64(divisibility=8))
        f_sparse_blocks = _fake(I32, (None,), cute.sym_int64(divisibility=4))
        f_sparse_pass_offsets = _fake(I32, (None,), cute.sym_int64(divisibility=4))
        stream = make_fake_stream(use_tvm_ffi_env_stream=False)
        builders = {
            "prep": lambda: (
                _prep_host,
                f_q,
                f_q,
                f_lse,
                f_h,
                f_h,
                f_dense_block_bitmap,
                f_dq_acc,
                f_dense_dkv,
                f_dense_dkv,
                I32(1),
                stream,
            ),
            "bitmap": lambda: (_bitmap_host, f_idx, f_dense_block_bitmap, I32(1), I32(1), stream),
            "part": lambda: (_part_host, f_idx, f_sparse_blocks, f_sparse_pass_offsets, I32(1), I32(1), I32(1), stream),
            "sparse": lambda: (
                _sparse_host,
                f_q,
                f_kv,
                f_kv,
                f_q,
                f_lse,
                f_lse,
                f_dq_acc,
                f_h,
                f_h,
                f_sparse_blocks,
                f_sparse_pass_offsets,
                I32(1),
                I32(0),
                I32(1),
                stream,
            ),
            "dense": lambda: (
                _dpair_host,
                f_kv,
                f_kv,
                f_q,
                f_q,
                f_lse,
                f_lse,
                f_dense_block_bitmap,
                f_dq_acc,
                f_dense_dkv,
                f_dense_dkv,
                f_dense_done_count,
                I32(1),
                I32(1),
                I32(1),
                I32(1),
                I32(1),
                0,
                stream,
            ),
            "dense_pdl": lambda: (
                _dpair_host,
                f_kv,
                f_kv,
                f_q,
                f_q,
                f_lse,
                f_lse,
                f_dense_block_bitmap,
                f_dq_acc,
                f_dense_dkv,
                f_dense_dkv,
                f_dense_done_count,
                I32(1),
                I32(1),
                I32(1),
                I32(1),
                I32(1),
                1,
                stream,
            ),
            "fin": lambda: (_fin_host, f_h, f_h, f_dense_dkv, f_dense_dkv, f_dq_acc, f_kv, f_kv, f_q, I32(1), I32(1), 0, stream),
            "fin_sparse": lambda: (_fin_host, f_h, f_h, f_dense_dkv, f_dense_dkv, f_dq_acc, f_kv, f_kv, f_q, I32(1), I32(1), 1, stream),
        }
        for name in missing:
            _COMPILED[name] = cute.compile(*builders[name](), options="--enable-tvm-ffi")
    return {name: _COMPILED[name] for name in names}


def launch(plan: LaunchPlan, fns: dict, q, k, v, out, dout, lse, topk_idxs, dq, dk, dv, scratch: dict, stream) -> None:
    """Enqueue the launch sequence on ``stream``. Inputs, outputs and scratch are caller-owned."""
    s = scratch
    dense_done_count = s["dense_block_bitmap"][plan.nsel :]
    fns["prep"](
        out,
        dout,
        s["delta"],
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_block_bitmap"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        plan.prep_blocks,
        stream,
    )
    if plan.has_sparse:
        fns["part"](topk_idxs, s["sparse_blocks"], s["sparse_pass_offsets"], plan.num_dense_queries, plan.pass_tokens // 4, plan.num_passes, stream)
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
        # The sparse grid is a normal launch, so the PDL-launched dense grid sees the partition lists and the bitmap.
        fns["sparse"](
            q,
            k,
            v,
            dout,
            lse,
            s["delta"],
            s["dq_acc"],
            s["sparse_dk_acc"],
            s["sparse_dv_acc"],
            s["sparse_blocks"],
            s["sparse_pass_offsets"],
            plan.sparse_ctas,
            plan.num_dense_queries,
            plan.num_passes,
            stream,
        )
        fns["dense_pdl"](*dense_args, stream)
        fns["fin_sparse"](*fin_args)
    else:
        fns["dense"](*dense_args, stream)
        fns["fin"](*fin_args)
