# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from sdpa/bwd/kernels/sm100/bprop_sparse_d256_f16.py.

"""Batched QSA backward for Blackwell and Rubin: NQ consecutive queries per CTA (two 12-head or four 6-head queries).

Each CTA gathers the union of its NQ queries' selected K/V blocks once; the queries' heads fill the 24 columns of one
MMA tile, and owner bits in the compact IDs keep every query's own mask.
"""

import hashlib
from pathlib import Path

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental import primitives as prims

from cudnn.frost.compiled_cache import compile_cached, template_key
from .._contract import BLK_BITS, PAIRED_BACKWARD_COLUMNS, paired_backward

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


I32 = cutlass.Int32
I64 = cutlass.Int64
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

D = 256
LOG2E = 1.4426950408889634

NTHREADS = 512
CHB = 24
# NQ = N_COLS // GQ consecutive queries fill the MMA columns: 2 x 12 heads (12/1) or 4 x 6 heads (6/1); 24/2 takes
# the single-query kernels. One owner bit per batched query follows the block index in the compact IDs.
N_COLS = PAIRED_BACKWARD_COLUMNS
BLK_MASK = (1 << BLK_BITS) - 1


OFF_Q = 0  # Reserve 32 shared rows per slab; MMA padding stays zero.
OFF_DO = 16384
QSLAB = 4096
OFF_K = 32768  # Two four-row blocks share each physical eight-row MMA core tile.
BLK_SMEM = 2048  # Four 512-byte slabs per gathered four-token block.
BLOCK_SLAB = 512
SLAB = 1024  # One physical eight-row core-matrix slab (two gathered blocks).
CORE_SMEM = 4096
K_STAGE = CHB * BLK_SMEM
OFF_V = OFF_K + 2 * K_STAGE
OFF_DS = OFF_V + K_STAGE
DS_BYTES = 6144
NDRAIN = 8
EP_BYTES = 4096
OFF_EP = OFF_DS + 3 * DS_BYTES
SMEM_BYTES = OFF_EP + NDRAIN * EP_BYTES

TM_S = 0
TM_DP = 32
TM_PT = 64
TM_DST = 96
TM_DQ = 128
TM_SL = 192
NSL = 5  # the most 64-column dV/dK slices that fit in the 512 TMEM columns after TM_SL
L2_EVICT_FIRST = 0x12F0000000000000
TM_COLS = 512

Q_BOX_BYTES = N_COLS * 128


@cute.jit
def _wait(bar, parity):
    while not prims.mbarrier_try_wait_parity(bar, parity):
        pass


def _desc(addr, lbo, sbo, swz):
    return prims.Tcgen05SmemDesc.build(start_address=addr, leading_byte_offset=lbo, stride_byte_offset=sbo, layout=swz)


@cute.jit
def _bwd_slice(j, cgb, pb, b_le, b_lf, tmem_raw, d_do_mn, d_q_mn, idesc_kv, HKV: cutlass.Constexpr[int]):
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


# Each gathered block serves its selected query columns.
@cute.kernel
def frost_qsa_bprop_sm100(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_k: cutlass.GridConstant[cuda.TensorMap],
    tma_v: cutlass.GridConstant[cuda.TensorMap],
    g_sparse_dv_acc: cute.Tensor,
    g_lse: cute.Tensor,
    g_delta: cute.Tensor,
    g_dq: cute.Tensor,
    g_sparse_dk_acc: cute.Tensor,
    g_sparse_blocks: cute.Tensor,
    g_counts: cute.Tensor,
    g_lo: cute.Tensor,
    g_hi: cute.Tensor,
    g_first: cute.Tensor,
    g_last: cute.Tensor,
    T: I32,
    NG: I32,
    nctas: I32,
    scale: F32,
    HQ: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    nonfinite,
):
    if cutlass.const_expr(nonfinite is not None):
        if nonfinite[0] != 0:
            cute.arch.nvvm.exit()
    GQ = HQ // HKV  # query heads per KV head: 12 (12/1) or 6 (6/1)
    NQ = N_COLS // GQ  # queries batched per CTA; their heads fill the MMA columns
    hint_ld = cutlass.Int64(L2_EVICT_FIRST)
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, SMEM_BYTES // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_qf = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_qe = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_kf = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_ke = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_kr = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
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

    # Keep unused K rows finite for the masked dQ MMA.
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
            for s in cutlass.range_constexpr(2):
                prims.mbarrier_init(b_kf.subview(s), 1)
                prims.mbarrier_init(b_ke.subview(s), 1)
                prims.mbarrier_init(b_kr.subview(s), 1)
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
    p_sparse_dv_acc = g_sparse_dv_acc.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_counts = g_counts.iterator.raw_ptr()
    p_lo = g_lo.iterator.raw_ptr()
    p_hi = g_hi.iterator.raw_ptr()
    p_first = g_first.iterator.raw_ptr()
    p_last = g_last.iterator.raw_ptr()
    capacity = I64(g_sparse_blocks.shape[1])

    wg = warp // 4
    if wg == 0:
        # The TMA unit zero-fills tokens past T, which keeps the masked rows of a tail block finite.
        if warp == 0:
            gc = I32(0)
            nq = I32(0)
            for qq in cutlass.range(bid, NG, nctas, unroll=1):
                qi = qq
                nblk = (p_counts + qi).load()
                if nblk > 0:
                    lrow = p_sparse_blocks + I64(qi) * capacity
                    nchunks = (nblk + (CHB - 1)) // CHB
                    _wait(b_qe, (nq & 1) ^ 1)
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(b_qf, 8 * Q_BOX_BYTES)
                        for s in cutlass.range_constexpr(4):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (OFF_Q + s * QSLAB) // 4,
                                tma_q.get_ptr(),
                                (s * 64, qi * (NQ * HQ)),
                                b_qf,
                            )
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sbase + (OFF_DO + s * QSLAB) // 4,
                                tma_do.get_ptr(),
                                (s * 64, qi * (NQ * HQ)),
                                b_qf,
                            )
                    # Prefetch the next chunk with in-bounds index loads.
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
                            first = (p_first + (blk & BLK_MASK)).load()
                            pair = lane >> 1
                            half = lane & 1
                            for s in cutlass.range_constexpr(4):
                                dst = OFF_K + st * K_STAGE + pair * CORE_SMEM + s * SLAB + half * BLOCK_SLAB
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + dst // 4,
                                    tma_k.get_ptr(),
                                    (0, 0, first, s),
                                    b_kf.subview(st),
                                )
                        _wait(b_ve, (cgc & 1) ^ 1)
                        if prims.elect_sync():
                            prims.mbarrier_arrive_expect_tx(b_vf, tbytes)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if ok:
                            first = (p_first + (blk & BLK_MASK)).load()
                            pair = lane >> 1
                            half = lane & 1
                            for s in cutlass.range_constexpr(4):
                                dst = OFF_V + pair * CORE_SMEM + s * SLAB + half * BLOCK_SLAB
                                prims.cp_async_bulk_tensor_shared_cta_global(
                                    sbase + dst // 4,
                                    tma_v.get_ptr(),
                                    (0, 0, first, s),
                                    b_vf,
                                )
                    gc = gc + nchunks
                    nq = nq + 1

        # Mask K before the dQ MMA: multiplying zero dS by NaN must not leak across a packed boundary.
        if warp == 1:
            gc = I32(0)
            for qi in cutlass.range(bid, NG, nctas, unroll=1):
                nblk = (p_counts + qi).load()
                if nblk > 0:
                    lrow = p_sparse_blocks + I64(qi) * capacity
                    qbase = qi * NQ
                    bnd_lo = []
                    bnd_hi = []
                    for i in cutlass.range_constexpr(NQ):
                        qrow = qbase + i
                        if qrow >= T:
                            qrow = T - 1
                        bnd_lo.append((p_lo + qrow).load())
                        bnd_hi.append((p_hi + qrow).load())
                    nchunks = (nblk + CHB - 1) // CHB
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc & 1
                        _wait(b_kf.subview(st), (cgc >> 1) & 1)
                        # Fully valid chunks need no shared-memory writes.
                        needs_mask = False
                        if lane < CHB:
                            needs_mask = True
                            if c * CHB + lane < nblk:
                                code = (lrow + c * CHB + lane).load()
                                block = code & BLK_MASK
                                begin, end = (p_first + block).load(), (p_last + block).load()
                                covered = False
                                for i in cutlass.range_constexpr(NQ):
                                    covered = covered | ((((code >> (BLK_BITS + i)) & 1) != 0) & (begin >= bnd_lo[i]) & (begin + 3 <= bnd_hi[i]))
                                spans = covered & (begin + 3 <= end) & (begin >= 0) & (begin + 3 < T)
                                needs_mask = not spans
                        if cute.arch.vote_ballot_sync(needs_mask) != 0:
                            for b in cutlass.range(CHB, unroll=1):
                                selected = c * CHB + b < nblk
                                first, last = I32(0), I32(-1)
                                code = I32(0)
                                if selected:
                                    code = (lrow + c * CHB + b).load()
                                    block = code & BLK_MASK
                                    first, last = (p_first + block).load(), (p_last + block).load()
                                for row in cutlass.range_constexpr(4):
                                    token = first + row
                                    span_ok = selected & (token <= last) & (token >= 0) & (token < T)
                                    valid = False
                                    for i in cutlass.range_constexpr(NQ):
                                        valid = valid | (span_ok & (((code >> (BLK_BITS + i)) & 1) != 0) & (token >= bnd_lo[i]) & (token <= bnd_hi[i]))
                                    if not valid:
                                        pair = b >> 1
                                        half = b & 1
                                        slab = lane >> 3
                                        offset = OFF_K + st * K_STAGE + pair * CORE_SMEM + slab * SLAB + half * BLOCK_SLAB + row * 128 + (lane & 7) * 16
                                        (sbase + offset // 4).store(zero4, alignment=16)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_kr.subview(st))
                    gc = gc + nchunks

        # Interleave dV/dK(c) with scores(c+1). Finish dQ(c) before reusing dS shared memory.
        if warp == 2:
            idesc_s = prims.Tcgen05InstrDesc.build(
                c_dtype=F32,
                a_dtype=BF16,
                b_dtype=BF16,
                n_dim=N_COLS,
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
                n_dim=N_COLS,
                m_dim=128,
                a_major=1,
                b_major=1,
            )
            SW128 = prims.Tcgen05SmemSwizzle.SWIZZLE_128B
            SWN = prims.Tcgen05SmemSwizzle.NONE
            # K/V tile: two gathered four-row blocks share each 8-row, 4-KiB core matrix.
            d_k_kmaj = _desc(sbase + OFF_K // 4, 16, CORE_SMEM, SW128)
            d_v_kmaj = _desc(sbase + OFF_V // 4, 16, CORE_SMEM, SW128)
            # K tile read as K^T (MN-major SW128): leading byte offset = next 64-dim slab (1 KB), stride byte offset = next 8 rows
            d_k_mn = _desc(sbase + OFF_K // 4, SLAB, CORE_SMEM, SW128)
            # Q/dO as B of S^T/dP^T (K-major SW128, 24 rows) and as B of dK/dV (MN-major SW128)
            d_q_kmaj = _desc(sbase + OFF_Q // 4, 16, 1024, SW128)
            d_do_kmaj = _desc(sbase + OFF_DO // 4, 16, 1024, SW128)
            d_q_mn = _desc(sbase + OFF_Q // 4, QSLAB, 1024, SW128)
            d_do_mn = _desc(sbase + OFF_DO // 4, QSLAB, 1024, SW128)
            # dS^T as B of dQ^T (MN-major, no swizzle): leading byte offset = next 8 K-rows, stride byte offset = next 8 columns
            d_ds_mn = _desc(sbase + OFF_DS // 4, 384, 128, SWN)

            gc = I32(0)
            nq = I32(0)
            for qq in cutlass.range(bid, NG, nctas, unroll=1):
                qi = qq
                nblk = (p_counts + qi).load()
                if nblk > 0:
                    lrow = p_sparse_blocks + I64(qi) * capacity
                    nchunks = (nblk + (CHB - 1)) // CHB
                    _wait(b_qf, nq & 1)
                    for c in cutlass.range(nchunks + 1, unroll=1):
                        # Interleave dQ, S, and dP with eight dV/dK slices. Wait before reusing S/dP.
                        cgc = gc + c
                        st = cgc & 1
                        cgb = cgc - 1  # chunk whose backward half is issued this iteration
                        pb = cgb & 1
                        if c >= 1:
                            _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if c < nchunks:
                            _wait(b_kr.subview(st), (cgc >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        dk_st = d_k_kmaj.advance_start_address(st * K_STAGE)
                        dk_mn = d_k_mn.advance_start_address(pb * K_STAGE)
                        acc_q = c > 1  # the first chunk of a query initialises the dQ accumulator
                        for j in cutlass.range_constexpr(8):
                            if c >= 1:
                                _bwd_slice(j, cgb, pb, b_le, b_lf, tmem_raw, d_do_mn, d_q_mn, idesc_kv, HKV)
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
                                    for component in cutlass.range_constexpr(3):
                                        for kk in cutlass.range(CHB // 4, unroll=1):
                                            if prims.elect_sync():
                                                prims.tcgen05_mma(
                                                    prims.Tcgen05MMAKind.F16,
                                                    prims.CTAGroup.CTA_1,
                                                    prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                                    dk_mn.advance_start_address(hh * 2 * SLAB + kk * 2 * CORE_SMEM),
                                                    d_ds_mn.advance_start_address(component * DS_BYTES + kk * 768),
                                                    idesc_q,
                                                    acc_q | (kk > 0) | (component > 0),
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
        qd = warp % 4
        row = qd * 32 + lane
        trow = row & 3
        jrow = row >> 2
        gc = I32(0)
        nq = I32(0)
        ds_off = (OFF_DS + (row >> 3) * 384 + (row & 7) * 16) // 4
        for qq in cutlass.range(bid, NG, nctas, unroll=1):
            qi = qq
            nblk = (p_counts + qi).load()
            if nblk > 0:
                lrow = p_sparse_blocks + I64(qi) * capacity
                nchunks = (nblk + (CHB - 1)) // CHB
                qbase = qi * NQ
                # A row past T reads the first query's bounds; no owner bit is set for it, so no key is valid for it.
                qrows, bnd_lo, bnd_hi = [], [], []
                for i in cutlass.range_constexpr(NQ):
                    qrow = qbase + i
                    if cutlass.const_expr(i > 0):
                        if qrow >= T:
                            qrow = qbase
                    qrows.append(qrow)
                    bnd_lo.append((p_lo + qrow).load())
                    bnd_hi.append((p_hi + qrow).load())
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
                    okslot = m < nblk
                    if jrow >= CHB:
                        okslot = False
                    # Values assigned under a dynamic branch must be plain names (the DSL yields names, not list items),
                    # so the per-query validity is four names of which the first NQ are used.
                    ok0 = okslot
                    ok1 = okslot
                    ok2 = okslot
                    ok3 = okslot
                    if okslot:
                        code = (lrow + m).load()
                        blk = code & BLK_MASK
                        token = (p_first + blk).load() + trow
                        span = (token <= (p_last + blk).load()) & (token >= 0) & (token < T)
                        ok0 = span & (((code >> BLK_BITS) & 1) != 0) & (token >= bnd_lo[0]) & (token <= bnd_hi[0])
                        ok1 = span & (((code >> (BLK_BITS + 1)) & 1) != 0) & (token >= bnd_lo[1]) & (token <= bnd_hi[1])
                        if cutlass.const_expr(NQ > 2):
                            ok2 = span & (((code >> (BLK_BITS + 2)) & 1) != 0) & (token >= bnd_lo[2]) & (token <= bnd_hi[2])
                            ok3 = span & (((code >> (BLK_BITS + 3)) & 1) != 0) & (token >= bnd_lo[3]) & (token <= bnd_hi[3])
                    oks = [ok0, ok1, ok2, ok3]
                    pw = []
                    dw = []
                    for i in cutlass.range_constexpr(NQ):
                        vi = oks[i]
                        qrow = qrows[i]
                        for j in cutlass.range_constexpr(GQ):
                            lj = (p_lse + j * T + qrow).load()
                            ej = (p_delta + qrow * HQ + j).load()
                            s0 = sv[i * GQ + j]
                            d0 = dpv[i * GQ + j]
                            pj = F32(0.0)
                            dsj = F32(0.0)
                            if vi:
                                pj = cute.math.exp2((s0 * scale - lj) * F32(LOG2E), fastmath=True)
                                dsj = pj * (d0 - ej) * scale
                            pw.append(pj)
                            dw.append(dsj)
                    # Columns past the batched query heads stay zero so the dV/dK MMAs ignore them.
                    zero = F32(0.0)
                    pfull = []
                    dfull = []
                    for n in cutlass.range_constexpr(32):
                        if cutlass.const_expr(n < N_COLS):
                            pfull.append(pw[n])
                            dfull.append(dw[n])
                        else:
                            pfull.append(zero)
                            dfull.append(zero)
                    pvec = cutlass.Vector.from_elements(pfull, F32).to(BF16).bitcast(I32)
                    dvec = cutlass.Vector.from_elements(dfull, F32).to(BF16).bitcast(I32)

                    # Buffer st was last read by the dV/dK MMAs of chunk c - 2.
                    _wait(b_pe.subview(st), par ^ 1)
                    t_pt = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_PT + st * 16).as_ptr(I32)
                    t_dst = prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + TM_DST + st * 16).as_ptr(I32)
                    prims.tcgen05_st("32x32b", t_pt, pvec)
                    prims.tcgen05_st("32x32b", t_dst, dvec)
                    # dQ^T reads dS^T as an SMEM B operand, so the row is also stored as MN-major core matrices.
                    ds_float = cutlass.Vector.from_elements(dfull, F32)
                    ds_high = ds_float.to(BF16)
                    ds_middle = (ds_float - ds_high.to(F32)).to(BF16)
                    ds_low = ((ds_float - ds_high.to(F32)) - ds_middle.to(F32)).to(BF16)
                    dq_components = [ds_high.bitcast(I32), ds_middle.bitcast(I32), ds_low.bitcast(I32)]
                    for component in cutlass.range_constexpr(3):
                        for nc in cutlass.range_constexpr(3):
                            (sbase + ds_off + component * DS_BYTES // 4 + nc * 32).store(dq_components[component][nc * 4 : nc * 4 + 4], alignment=16)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(st))

                # Each CTA owns its batched queries' complete dQ.
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
                for i in cutlass.range_constexpr(NQ):
                    qout = qbase + i
                    if qout < T:
                        for hh in cutlass.range_constexpr(2):
                            dim = hh * 128 + qd * 32 + lane
                            qb = p_dq + (I64(qout) * (HQ * D) + I64(dim))
                            for n in cutlass.range_constexpr(GQ):
                                (qb + n * D).store(dqv[hh][i * GQ + n])
                gc = gc + nchunks
                nq = nq + 1

    else:
        # Stage FP32 slices in coalesced 16-byte vectors for global reduction.
        grp = (warp - 8) >> 2
        qd = warp % 4
        ep = sbase + (OFF_EP + (warp - 8) * EP_BYTES) // 4
        gc = I32(0)
        for qq in cutlass.range(bid, NG, nctas, unroll=1):
            qi = qq
            nblk = (p_counts + qi).load()
            if nblk > 0:
                lrow = p_sparse_blocks + I64(qi) * capacity
                nchunks = (nblk + (CHB - 1)) // CHB
                # Prefetch the next chunk with in-bounds index loads.
                nx = []
                for g in cutlass.range_constexpr(8):
                    m = qd * 8 + g
                    if m >= nblk:
                        m = I32(0)
                    nx.append((lrow + m).load())
                nx0 = nx[0]
                nx1 = nx[1]
                nx2 = nx[2]
                nx3 = nx[3]
                nx4 = nx[4]
                nx5 = nx[5]
                nx6 = nx[6]
                nx7 = nx[7]
                for c in cutlass.range(nchunks, unroll=1):
                    cgc = gc + c
                    cur = [nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7]
                    if c + 1 < nchunks:
                        nn = []
                        for g in cutlass.range_constexpr(8):
                            m = (c + 1) * CHB + qd * 8 + g
                            if m >= nblk:
                                m = I32(0)
                            nn.append((lrow + m).load())
                        nx0 = nn[0]
                        nx1 = nn[1]
                        nx2 = nn[2]
                        nx3 = nn[3]
                        nx4 = nn[4]
                        nx5 = nn[5]
                        nx6 = nn[6]
                        nx7 = nn[7]
                    grow = []
                    bok = []
                    for g in cutlass.range_constexpr(8):
                        m = c * CHB + qd * 8 + g
                        blk = cur[g]
                        ok = m < nblk
                        if qd * 8 + g >= CHB:
                            ok = False
                        first = I32(0)
                        if ok:
                            first = (p_first + (blk & BLK_MASK)).load()
                        grow.append(first)
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
                            prims.bar_warp_sync(0xFFFFFFFF)
                            srow = lane
                            for cc in cutlass.range_constexpr(8):
                                if cutlass.const_expr(pg == 0):
                                    w0 = va[4 * cc : 4 * cc + 4].bitcast(I32)
                                else:
                                    w0 = vb[4 * cc : 4 * cc + 4].bitcast(I32)
                                (ep + srow * 32 + ((srow ^ cc) & 7) * 4).store(w0, alignment=16)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            cl = lane & 7
                            for kk in cutlass.range_constexpr(8):
                                srow = kk * 4 + (lane >> 3)
                                token = grow[kk] + (lane >> 3)
                                if bok[kk] & (token >= 0) & (token < T):
                                    v4 = (ep + srow * 32 + ((srow ^ cl) & 7) * 4).load(count=4, alignment=16).bitcast(F32)
                                    goff = token * D + dblk * 64 + pg * 32 + cl * 4
                                    if cutlass.const_expr(jj >= 2):
                                        ptr = p_sparse_dk_acc + goff
                                    else:
                                        ptr = p_sparse_dv_acc + goff
                                    prims.red(prims.ReductionOp.ADD, prims.ReductionType.F32, ptr, v4, mem_order=prims.MemOrder.RELAXED)
                            prims.bar_warp_sync(0xFFFFFFFF)
                gc = gc + nchunks

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 2:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), TM_COLS, group="cta_1")


frost_qsa_bprop_sm100.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _bprop_host(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    dout: cute.Tensor,
    lse: cute.Tensor,
    delta: cute.Tensor,
    dq_acc: cute.Tensor,
    dk_acc: cute.Tensor,
    dv_acc: cute.Tensor,
    ids: cute.Tensor,
    counts: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    first: cute.Tensor,
    last: cute.Tensor,
    scale: F32,
    nctas: I32,
    stream: cuda_driver.CUstream,
    nonfinite=None,
):
    total = q.shape[0]
    hq, hkv = cutlass.const_expr(q.shape[1]), cutlass.const_expr(k.shape[1])
    nq = N_COLS // (hq // hkv)
    # One box holds the NQ consecutive queries' heads; the TMA unit zero-fills the rows past T * Hq of the last group.
    flat = cute.make_layout((total * hq, D), stride=(D, 1))
    tma_q = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(q.iterator, flat), box_dims=(N_COLS, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
    )
    tma_do = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(dout.iterator, flat), box_dims=(N_COLS, 64), stride_order=(1, 0), swizzle=cuda.TensorMapSwizzle.s128b
    )
    layout = cute.make_layout((total, hkv, 4, 64), stride=(hkv * D, D, 64, 1))
    tma_k = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, layout), box_dims=(4, 1, 1, 64), stride_order=(3, 1, 0, 2), swizzle=cuda.TensorMapSwizzle.s128b
    )
    tma_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, layout), box_dims=(4, 1, 1, 64), stride_order=(3, 1, 0, 2), swizzle=cuda.TensorMapSwizzle.s128b
    )
    frost_qsa_bprop_sm100(
        tma_q,
        tma_do,
        tma_k,
        tma_v,
        dv_acc,
        lse,
        delta,
        dq_acc,
        dk_acc,
        ids,
        counts,
        lo,
        hi,
        first,
        last,
        I32(total),
        (I32(total) + I32(nq - 1)) // I32(nq),
        nctas,
        scale,
        hq,
        hkv,
        nonfinite,
    ).launch(grid=(nctas, 1, 1), block=(NTHREADS, 1, 1), min_blocks_per_mp=1, stream=stream)


def compile_kernel(dtype, query_heads, kv_heads, arch):
    """Compile native THD geometry; tensor extents remain dynamic."""
    if not paired_backward(query_heads, kv_heads):
        from .bprop_sm100_base import compile_kernel as compile_baseline

        return compile_baseline(dtype, query_heads, kv_heads, arch)
    from .bprop_helpers import compile_helpers, fake

    if dtype != BF16:
        raise ValueError("Batched QSA backward requires BF16, D256 and 12/1 or 6/1 heads")
    if arch not in (100, 103, 107):
        raise ValueError(f"Batched QSA backward requires SM100, SM103 or SM107, got SM{arch}")
    key = template_key(globals(), {"query_heads": query_heads, "kv_heads": kv_heads, "arch": arch}, "compile_kernel")
    total, capacity, blocks, groups = cute.sym_int64(), cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    q, kv = fake(BF16, (total, query_heads, D)), fake(BF16, (total, kv_heads, D))
    aq, ak = fake(F32, (total, query_heads, D)), fake(F32, (total, kv_heads, D))
    lse, delta = fake(F32, (query_heads, total), 4), fake(F32, (total, query_heads))
    ids = fake(I32, (groups, capacity), 4)
    rows, spans = fake(I32, (total,), 4), fake(I32, (blocks,), 4)
    gcounts = fake(I32, (groups,), 4)
    compiled = dict(compile_helpers(query_heads, kv_heads, arch))
    compiled["main"] = compile_cached(
        _bprop_host,
        q,
        kv,
        kv,
        q,
        lse,
        delta,
        aq,
        ak,
        ak,
        ids,
        gcounts,
        rows,
        rows,
        spans,
        spans,
        F32(1),
        I32(1),
        make_fake_stream(use_tvm_ffi_env_stream=False),
        options=f"--enable-tvm-ffi --gpu-arch=sm_{arch}a",
        cache_key=key,
        symbol="qsa_backward",
    )
    from . import bprop_dispatch

    compiled["dispatch"] = bprop_dispatch.compile_kernel(arch, query_heads, kv_heads)
    return compiled


def launch(compiled, grad_out, q, k, v, out_fp32, lse, ids, counts, lo, hi, first, last, dq, dk, dv, scratch, scale, stream, num_sms):
    """Launch with caller-owned outputs and scratch."""
    if not paired_backward(q.shape[1], k.shape[1]):
        from .bprop_sm100_base import launch as launch_baseline

        return launch_baseline(compiled, grad_out, q, k, v, out_fp32, lse, ids, counts, lo, hi, first, last, dq, dk, dv, scratch, scale, stream, num_sms)
    from .bprop_dispatch import launch as dispatch

    return dispatch(compiled["dispatch"], grad_out, q, k, v, out_fp32, lse, ids, counts, lo, hi, first, last, dq, dk, dv, scratch, scale, stream, num_sms)
