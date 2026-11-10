# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM107 (Rubin) query-major, key-major and dense pair kernels of the Qwen Sparse Attention (QSA) backward.

``cudnn.QSA._backward`` compiles them in place of the SM100 kernels on SM107 when ``supported``. Each kernel is its
SM100 twin in ``bprop_sm100``, which documents the shared logic, with these changes:

- The oversized shared memory carveout (up to 327 KiB per CTA, leaving 8 KB of L1) holds a third K stage and a second
  buffer of the three dS^T parts in the query-major kernel, and a fifth stage of the key-major kernel with two KV heads.
- The dense pair kernel keeps its K^T half in TMEM, so its dQ^T MMAs read no K from shared memory.
- The dense pair kernel with twelve-head tiles and three dS parts spins on its barriers instead of suspending.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cuda.bindings import driver as cuda_driver
from cutlass.experimental import primitives as prims

from cudnn._device import device_info

from .bprop_sm100 import (
    BF16,
    BLK_SMEM,
    CHB,
    D,
    DP_CM,
    DP_G,
    DP_HS,
    DP_KSLAB,
    DP_N,
    DP_NE,
    DP_NL,
    DP_NX,
    DP_OFF_DS,
    DP_OFF_E,
    DP_OFF_L,
    DP_OFF_P,
    DP_OFF_ST,
    DP_OFF_T,
    DP_QSLAB,
    DP_SMEM,
    DP_ST_W,
    DP_THREADS,
    DP_TM_A,
    DP_TM_DQ,
    DP_TM_KV,
    DP_TM_PA,
    DP_TM_SD,
    DP_TT,
    DS_BYTES,
    F32,
    GAP01,
    GAP12,
    GQ,
    HALF_BLK,
    I32,
    K_STAGE,
    L2_EVICT_FIRST,
    LOG2E,
    NTHREADS,
    OFF_K,
    PART_W,
    PB_THREADS,
    POS,
    Q_BYTES,
    QSLAB,
    SLAB,
    TM_COLS,
    TM_DP,
    TM_DQ,
    TM_KA,
    TM_S,
    TM_VA,
    _PbLayout as _PbLayout100,
    _active_key,
    _chunk_bytes,
    _chunk_valid,
    _delta_rows,
    _dense_rows_load,
    _dense_sel,
    _desc,
    _item_meta,
    _koff,
    _lane_meta,
    _load_block,
    _lse_rows,
    _pair_meta,
    _prefetch_block,
    _rec_w,
    _rot,
    _st_async8,
    _st_async16,
)

# the dense pair kernel's K^T half follows the 512 columns of the SM100 TMEM map
DP_TM_KT = 512
DP_TM_COLS = 576


def supported(device_index: int) -> bool:
    """Whether the device's driver offers the oversized shared memory carveout these kernels need (CUDA 13.4+)."""
    # the largest layout plus barriers and static storage
    return device_info(device_index).oversized_shared_memory_per_block >= _PbLayout(2).smem + 2048


@cute.jit
def _wait(bar, parity, spin: cutlass.Constexpr[bool] = False):
    if cutlass.const_expr(spin):
        # sm_107a issues the hint-less try-wait as a uniform spin, and the DSL wrapper always passes a suspend hint
        prims.inline_ptx(
            "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [{$r0}], {$r1};\n\t"
            "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}",
            read_only_args=[bar.data_ptr(), I32(parity)],
        )
    else:
        while not prims.mbarrier_try_wait_parity(bar, parity):
            pass


def _query_layout(parts):
    # [K0][pad][K1][dO][K2][Q][dS x parts x buffers][V]. With three parts a second dS^T buffer lets the softmax write
    # dS(c + 1) while dQ(c) reads dS(c).
    nds = 2 if parts > 1 else 1
    off_do = OFF_K + 2 * K_STAGE + GAP01
    off_q = off_do + GAP12 + K_STAGE
    off_ds = off_q + Q_BYTES
    off_v = off_ds + nds * parts * DS_BYTES
    return 3, nds, off_do, off_q, off_v, off_ds


@cute.kernel
def qsa_bwd_sparse_query_sm107_kernel(
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
    g_sparse_queries: cute.Tensor,
    g_lo: cute.Tensor,
    g_hi: cute.Tensor,
    g_sparse_count: cute.Tensor,
    nctas: I32,
    NP: I32,
    scale: F32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
):
    NKS, NDS, OFF_DO, OFF_Q, OFF_V, OFF_DS = _query_layout(DQ_PARTS)
    HQ = GH * HKV
    PAIR = HKV == 1
    QSTEP = 2 if PAIR else 1
    REC_W = _rec_w(HKV)
    nsq = g_sparse_count.iterator.raw_ptr().load()
    p_sparse_queries = g_sparse_queries.iterator.raw_ptr()
    p_lo = g_lo.iterator.raw_ptr()
    p_hi = g_hi.iterator.raw_ptr()
    hint_q = cutlass.Int64(L2_EVICT_FIRST)
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, (OFF_V + K_STAGE) // 4, space=cutlass.AddressSpace.smem, alignment=1024)
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
    b_dsf = cutlass.Array(cutlass.Int64, NDS, space=cutlass.AddressSpace.smem, alignment=8)
    b_ktf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_kte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b_vtf = cutlass.Array(cutlass.Int64, 4, space=cutlass.AddressSpace.smem, alignment=8)
    b_vte = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    sbase = smem.data_ptr()
    for it in cutlass.range(tidx, (OFF_V + K_STAGE) // 16, NTHREADS, unroll=1):
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
            for s in cutlass.range_constexpr(NDS):
                prims.mbarrier_init(b_dsf.subview(s), 1)
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

    p_delta = g_delta.iterator.raw_ptr()
    p_dq = g_dq.iterator.raw_ptr()
    p_sparse_blocks = g_sparse_blocks.iterator.raw_ptr()
    p_sparse_pass_offsets = g_sparse_pass_offsets.iterator.raw_ptr()
    p_sparse_block_slots = g_sparse_block_slots.iterator.raw_ptr()
    p_pair_records = g_pair_records.iterator.raw_ptr()

    wg = warp // 4
    if wg == 0:
        if cutlass.const_expr(PAIR):
            jl = lane >> 1
            hl = lane & 1
            lane_ok = lane < 2 * CHB
        else:
            jl = lane
            hl = I32(0)
            lane_ok = lane < CHB
        if warp == 0:
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, o0_l, nblk_l = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hl)
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((j + hl) * PART_W + o0_l)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qe, (nq & 1) ^ 1)
                        if prims.elect_sync():
                            nqq = I32(1)
                            if cutlass.const_expr(PAIR):
                                if j + 1 < nsq:
                                    nqq = I32(2)
                            prims.mbarrier_arrive_expect_tx(b_qf, nqq * (2 * 4 * GH * HKV * 128))
                            for h in cutlass.range_constexpr(QSTEP):
                                if h < nqq:
                                    qh = (p_sparse_queries + (j + h)).load()
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sbase + (OFF_Q + s * QSLAB + h * (QSLAB // 2)) // 4,
                                            tma_q.get_ptr(),
                                            (s * 64, 0, qh),
                                            b_qf,
                                            l2_cache_hint=hint_q,
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sbase + (OFF_DO + s * QSLAB + h * (QSLAB // 2)) // 4,
                                            tma_do.get_ptr(),
                                            (s * 64, 0, qh),
                                            b_qf,
                                            l2_cache_hint=hint_q,
                                        )
                        m0 = jl
                        if m0 >= nblk_l:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            st = cgc % NKS
                            kpar = (cgc // NKS) & 1
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + jl
                                if m1 >= nblk_l:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + jl
                            ok = (m < nblk_l) & lane_ok
                            tbytes = _chunk_bytes(nblk_0, nblk_1, c, PAIR)
                            _wait(b_ke.subview(st), kpar ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_kf.subview(st), tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                _load_block(tma_k, sbase + (OFF_K + _koff(st) + jl * BLK_SMEM + hl * HALF_BLK) // 4, xcur, b_kf.subview(st), PAIR)
                            if c + 1 < nchunks:
                                okn = ((c + 1) * CHB + jl < nblk_l) & lane_ok
                                if okn:
                                    _prefetch_block(tma_k, xnext, PAIR)
                        gc = gc + nchunks
                        nq = nq + 1

        if warp == 1:
            gc = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, o0_l, nblk_l = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hl)
                    if nblk > 0:
                        lrow = p_sparse_blocks + ((j + hl) * PART_W + o0_l)
                        nchunks = (nblk + (CHB - 1)) // CHB
                        m0 = jl
                        if m0 >= nblk_l:
                            m0 = I32(0)
                        xnext = (lrow + m0).load()
                        for c in cutlass.range(nchunks, unroll=1):
                            cgc = gc + c
                            xcur = xnext
                            if c + 1 < nchunks:
                                m1 = (c + 1) * CHB + jl
                                if m1 >= nblk_l:
                                    m1 = I32(0)
                                xnext = (lrow + m1).load()
                            m = c * CHB + jl
                            ok = (m < nblk_l) & lane_ok
                            tbytes = _chunk_bytes(nblk_0, nblk_1, c, PAIR)
                            _wait(b_ve, (cgc & 1) ^ 1)
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_vf, tbytes)
                            prims.bar_warp_sync(0xFFFFFFFF)
                            if ok:
                                _load_block(tma_v, sbase + (OFF_V + jl * BLK_SMEM + hl * HALF_BLK) // 4, xcur, b_vf, PAIR)
                            if c + 1 < nchunks:
                                okn = ((c + 1) * CHB + jl < nblk_l) & lane_ok
                                if okn:
                                    _prefetch_block(tma_v, xnext, PAIR)
                        gc = gc + nchunks

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
            d_q_kmaj = _desc(sbase + OFF_Q // 4, 16, 1024, SW128)
            d_do_kmaj = _desc(sbase + OFF_DO // 4, 16, 1024, SW128)

            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                    if nblk > 0:
                        nchunks = (nblk + (CHB - 1)) // CHB
                        _wait(b_qf, nq & 1)
                        for c in cutlass.range(nchunks + 1, unroll=1):
                            cgc = gc + c
                            st = cgc % NKS
                            if c < nchunks:
                                _wait(b_kf.subview(st), (cgc // NKS) & 1)
                                _wait(b_sde.subview(cgc & 1), ((cgc >> 1) & 1) ^ 1)
                                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
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
            if cutlass.const_expr(PAIR):
                d_k_mn3 = _desc(sbase + OFF_K // 4, 512, HALF_BLK, prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
            else:
                d_k_mn3 = _desc(sbase + OFF_K // 4, SLAB, BLK_SMEM, prims.Tcgen05SmemSwizzle.SWIZZLE_128B)
            d_ds_mn3 = _desc(sbase + OFF_DS // 4, 384, 128, prims.Tcgen05SmemSwizzle.NONE)
            gc = I32(0)
            nq = I32(0)
            for pp in cutlass.range(NP, unroll=1):
                for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                    o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                    o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                    nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                    if nblk > 0:
                        nchunks = (nblk + (CHB - 1)) // CHB
                        for c in cutlass.range(nchunks, unroll=1):
                            cgb = gc + c
                            pb = cgb & 1
                            kb = cgb % NKS
                            nv3 = _chunk_valid(nblk, c)
                            _wait(b_pf.subview(pb), (cgb >> 1) & 1)
                            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            if c == 0:
                                _wait(b_dqe, (nq & 1) ^ 1)
                                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                            acc_q = c > 0
                            dk_mn = d_k_mn3.advance_start_address(_koff(kb))
                            if cutlass.const_expr(PAIR):
                                nkk = nv3
                            else:
                                nkk = (nv3 + 1) >> 1
                            for hh in cutlass.range_constexpr(2):
                                for part in cutlass.range_constexpr(DQ_PARTS):
                                    for kk in cutlass.range(nkk, unroll=1):
                                        if cutlass.const_expr(PAIR):
                                            a_k = dk_mn.advance_start_address(hh * 2 * 512 + kk * BLK_SMEM)
                                        else:
                                            a_k = dk_mn.advance_start_address(hh * 2 * SLAB + kk * 2 * BLK_SMEM)
                                        if prims.elect_sync():
                                            prims.tcgen05_mma(
                                                prims.Tcgen05MMAKind.F16,
                                                prims.CTAGroup.CTA_1,
                                                prims.make_tmem_ptr(tmem_raw + (TM_DQ + hh * 32), F32),
                                                a_k,
                                                d_ds_mn3.advance_start_address(((cgb % NDS) * DQ_PARTS + part) * DS_BYTES + kk * 768),
                                                idesc_q3,
                                                acc_q | (part > 0) | (kk > 0),
                                            )
                            if prims.elect_sync():
                                prims.tcgen05_commit(b_ke.subview(kb), group=prims.CTAGroup.CTA_1)
                                prims.tcgen05_commit(b_dsf.subview(cgb % NDS), group=prims.CTAGroup.CTA_1)
                                if c == nchunks - 1:
                                    prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.bar_warp_sync(0xFFFFFFFF)
                        gc = gc + nchunks
                        nq = nq + 1

    elif wg == 1:
        qd = warp % 4
        row = qd * 32 + lane
        if cutlass.const_expr(PAIR):
            hrow = (lane >> 2) & 1
            trow = lane & 3
        else:
            hrow = lane & 1
            trow = (lane & 7) >> 1
        jrow = row >> 3
        c_s = scale * F32(LOG2E)
        gc = I32(0)
        nq = I32(0)
        if cutlass.const_expr(PAIR):
            ds_off = (OFF_DS + (2 * jrow + hrow) * 384 + trow * 16) // 4
        else:
            ds_off = (OFF_DS + jrow * 384 + (row & 7) * 16) // 4
        for pp in cutlass.range(NP, unroll=1):
            for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                if cutlass.const_expr(PAIR):
                    nblk, o0, nblk_r = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, hrow)
                else:
                    nblk, o0, nblk_r = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                if nblk > 0:
                    jr = j
                    row_ok = jrow < CHB
                    if cutlass.const_expr(PAIR):
                        jr = j + hrow
                        row_ok = row_ok & (jr < nsq)
                        if jr >= nsq:
                            jr = j
                    qr = (p_sparse_queries + jr).load()
                    h0 = I32(0)
                    if cutlass.const_expr(not PAIR):
                        h0 = hrow * GH
                    lbase = qr * HQ + h0
                    lo_q = (p_lo + qr).load()
                    hi_q = (p_hi + qr).load()
                    nchunks = (nblk + (CHB - 1)) // CHB
                    prow_pp = p_sparse_block_slots + (jr * PART_W + o0)
                    trow_pp = p_sparse_blocks + (jr * PART_W + o0)
                    mpf = jrow
                    if mpf >= nblk_r:
                        mpf = I32(0)
                    pnx = (prow_pp + mpf).load()
                    tnx = (trow_pp + mpf).load()
                    lse_v = _lse_rows(g_lse, qr, h0, GH)
                    del_v = _delta_rows(p_delta, lbase, GH)
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
                        valid = (m < nblk_r) & row_ok
                        slot_ok = valid
                        pcur = pnx
                        tcur = tnx
                        if c + 1 < nchunks:
                            mpf = (c + 1) * CHB + jrow
                            if mpf >= nblk_r:
                                mpf = I32(0)
                            pnx = (prow_pp + mpf).load()
                            tnx = (trow_pp + mpf).load()
                        tk = tcur + trow
                        score_v = sv[0:GQ]
                        dot_v = dpv[0:GQ]
                        if hrow == 1:
                            score_v = sv[GQ : 2 * GQ]
                            dot_v = dpv[GQ : 2 * GQ]
                        bad = valid & (~cute.math.isfinite(score_v[0]))
                        if cute.arch.vote_any_sync(bad):
                            if bad:
                                st_k = cgc % NKS
                                if cutlass.const_expr(PAIR):
                                    krow = sbase + (OFF_K + _koff(st_k) + jrow * BLK_SMEM + hrow * HALF_BLK + trow * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * 512) // 4 + u * 4).store(zero4, alignment=16)
                                else:
                                    krow = sbase + (OFF_K + _koff(st_k) + jrow * BLK_SMEM + (2 * trow + hrow) * 128) // 4
                                    for s4 in cutlass.range_constexpr(4):
                                        for u in cutlass.range_constexpr(8):
                                            (krow + (s4 * SLAB) // 4 + u * 4).store(zero4, alignment=16)
                        valid = valid & (tk >= lo_q) & (tk <= hi_q)
                        prob_v = zero12
                        ds_v = zero12
                        if valid:
                            prob_v = cute.math.exp2(score_v * c_s - lse_v * F32(LOG2E), fastmath=True)
                            ds_v = prob_v * (dot_v - del_v) * scale

                        prw = prob_v.to(BF16).bitcast(I32)
                        drw = ds_v.to(BF16).bitcast(I32)
                        dvecs = []
                        res = ds_v
                        for part in cutlass.range_constexpr(DQ_PARTS):
                            bp = res.to(BF16)
                            if cutlass.const_expr(part + 1 < DQ_PARTS):
                                res = res - bp.to(F32)
                            bpw = bp.bitcast(I32)
                            dwords = []
                            for n in cutlass.range_constexpr(16):
                                dv_ = I32(0)
                                if cutlass.const_expr(n < 6):
                                    if hrow == 0:
                                        dv_ = bpw[n]
                                elif cutlass.const_expr(n < 12):
                                    if hrow == 1:
                                        dv_ = bpw[n - 6]
                                dwords.append(dv_)
                            dvecs.append(cutlass.Vector.from_elements(dwords, I32))

                        # dQ(c - NDS) has finished reading this chunk's dS^T buffer
                        _wait(b_dsf.subview(cgc % NDS), ((cgc // NDS) & 1) ^ 1)
                        if jrow < CHB:
                            ds_buf = ds_off + (cgc % NDS) * (DQ_PARTS * DS_BYTES // 4)
                            for part in cutlass.range_constexpr(DQ_PARTS):
                                for nc in cutlass.range_constexpr(3):
                                    (sbase + ds_buf + part * (DS_BYTES // 4) + nc * 32).store(dvecs[part][nc * 4 : nc * 4 + 4], alignment=16)
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        prims.bar_warp_sync(0xFFFFFFFF)
                        if prims.elect_sync():
                            prims.mbarrier_arrive(b_pf.subview(st))
                        if slot_ok:
                            if cutlass.const_expr(PAIR):
                                roff = cutlass.Int64(pcur) * REC_W + cutlass.Int64(trow * 6)
                            else:
                                roff = cutlass.Int64(pcur) * REC_W + cutlass.Int64(trow * 12 + hrow * 6)
                            rbase = p_pair_records + roff
                            for u in cutlass.range_constexpr(3):
                                prims.store_ext(prw[2 * u : 2 * u + 2], rbase + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)
                                prims.store_ext(drw[2 * u : 2 * u + 2], rbase + REC_W // 2 + 2 * u, cache_modifier=prims.StoreCacheModifier.CS)

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
                    for h in cutlass.range_constexpr(2 if PAIR else 1):
                        o0_h = o0_0
                        write_h = (p_sparse_pass_offsets + (j * POS + NP)).load() > 0
                        if cutlass.const_expr(h == 1):
                            o0_h = o0_1
                            write_h = False
                            if j + 1 < nsq:
                                write_h = (p_sparse_pass_offsets + ((j + 1) * POS + NP)).load() > 0
                        if write_h:
                            qrow = (p_sparse_queries + (j + h)).load()
                            for hh in cutlass.range_constexpr(2):
                                dim = hh * 128 + qd * 32 + lane
                                qb = p_dq + (qrow * (HQ * D) + dim)
                                if o0_h == 0:
                                    for n in cutlass.range_constexpr(GH if PAIR else HQ):
                                        (qb + n * D).store(dqv[hh][h * GQ + n])
                                else:
                                    for n in cutlass.range_constexpr(GH if PAIR else HQ):
                                        prims.red(
                                            prims.ReductionOp.ADD,
                                            prims.ReductionType.F32,
                                            qb + n * D,
                                            dqv[hh][h * GQ + n],
                                            mem_order=prims.MemOrder.RELAXED,
                                            mem_scope=prims.MemScope.GPU,
                                            cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                                        )
                    gc = gc + nchunks
                    nq = nq + 1

    else:
        qd = warp % 4
        hh = (warp - 8) >> 2
        r = qd * 32 + lane
        if cutlass.const_expr(PAIR):
            yr = r & 3
            roff = (OFF_K + (r >> 3) * BLK_SMEM + ((r >> 2) & 1) * HALF_BLK + yr * 128) // 4
            slab_w = 512 // 4
        else:
            yr = r & 7
            roff = (OFF_K + (r >> 3) * BLK_SMEM + yr * 128) // 4
            slab_w = SLAB // 4
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
            for j in cutlass.range(QSTEP * bid, nsq, QSTEP * nctas, unroll=1):
                o0_0, nblk_0 = _item_meta(p_sparse_pass_offsets, j, pp)
                o0_1, nblk_1 = _pair_meta(p_sparse_pass_offsets, j, pp, nsq, PAIR)
                nblk, _, _ = _lane_meta(o0_0, nblk_0, o0_1, nblk_1, I32(0))
                if nblk > 0:
                    nchunks = (nblk + (CHB - 1)) // CHB
                    for c in cutlass.range(nchunks, unroll=1):
                        cgc = gc + c
                        st = cgc % NKS
                        nval_c = _chunk_valid(nblk, c)
                        do_cp = qd * 4 < nval_c
                        _wait(b_kf.subview(st), (cgc // NKS) & 1)
                        _wait(b_kte, (cgc & 1) ^ 1)
                        for sl4 in cutlass.range_constexpr(4):
                            ys = yr
                            if cutlass.const_expr(PAIR and sl4 % 2 == 1):
                                ys = yr + 4
                            if do_cp:
                                off = roff + _koff(st) // 4 + sl4 * slab_w
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (sbase + (off + ((ys ^ cc) * 4))).load(count=4, alignment=16)
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
                                if do_cp:
                                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                prims.bar_warp_sync(0xFFFFFFFF)
                                if prims.elect_sync():
                                    prims.mbarrier_arrive(b_ktf.subview(1))
                        _wait(b_vf, cgc & 1)
                        _wait(b_vte, (cgc & 1) ^ 1)
                        vb = sbase + (roff - OFF_K // 4 + OFF_V // 4)
                        for sl4 in cutlass.range_constexpr(4):
                            ys = yr
                            if cutlass.const_expr(PAIR and sl4 % 2 == 1):
                                ys = yr + 4
                            if do_cp:
                                wv = []
                                for i4 in cutlass.range_constexpr(4):
                                    cc = hh * 4 + i4
                                    v4 = (vb + (sl4 * slab_w + ((ys ^ cc) * 4))).load(count=4, alignment=16)
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


qsa_bwd_sparse_query_sm107_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    sparse_queries: cute.Tensor,
    lo: cute.Tensor,
    hi: cute.Tensor,
    sparse_count: cute.Tensor,
    nctas: I32,
    NP: I32,
    scale: F32,
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    qbox = (1, GH * HKV, 64)
    tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    if cutlass.const_expr(HKV == 1):
        kv_layout = cute.make_layout((T, 4, 64), stride=(D, 64, 1))
        kv_box = (4, 4, 64)
        kv_order = (2, 0, 1)
    else:
        kv_layout = cute.make_layout((T, HKV, 4, 64), stride=(HKV * D, D, 64, 1))
        kv_box = (4, HKV, 4, 64)
        kv_order = (3, 1, 0, 2)
    tma_k = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(k.iterator, kv_layout), box_dims=kv_box, stride_order=kv_order, swizzle=cuda.TensorMapSwizzle.s128b
    )
    tma_v = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(v.iterator, kv_layout), box_dims=kv_box, stride_order=kv_order, swizzle=cuda.TensorMapSwizzle.s128b
    )
    qsa_bwd_sparse_query_sm107_kernel(
        tma_q,
        tma_do,
        tma_k,
        tma_v,
        lse,
        delta,
        dq,
        sparse_blocks,
        sparse_pass_offsets,
        sparse_block_slots,
        pair_records,
        sparse_queries,
        lo,
        hi,
        sparse_count,
        nctas,
        NP,
        scale,
        HKV,
        GH,
        DQ_PARTS,
    ).launch(grid=(nctas, 1, 1), block=(NTHREADS, 1, 1), stream=stream)


class _PbLayout(_PbLayout100):
    def __init__(self, hkv, gh=GQ):
        super().__init__(hkv, gh)
        # a fifth stage for two KV heads fits the oversized carveout, and SM107 descriptors address past 256 KiB
        self.nst = 5 if hkv == 2 else 4
        self.off_st = self.nst * self.stage
        self.smem = self.off_st + 4 * self.st_w * 4


@cute.kernel
def qsa_bwd_sparse_key_sm107_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_do: cutlass.GridConstant[cuda.TensorMap],
    tma_rk: cutlass.GridConstant[cuda.TensorMap],
    tma_rv: cutlass.GridConstant[cuda.TensorMap],
    g_pair_records: cute.Tensor,
    g_key_slot_offsets: cute.Tensor,
    g_key_pair_counts: cute.Tensor,
    g_slot_queries: cute.Tensor,
    g_active_keys: cute.Tensor,
    g_active_count: cute.Tensor,
    KR: I32,
    SK: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
):
    PB = _PbLayout(HKV, GH)
    n0 = g_active_count.iterator.raw_ptr().load()
    Ktot = n0 + (g_active_count.iterator.raw_ptr() + 1).load()
    p_active = g_active_keys.iterator.raw_ptr()
    QPS = PB.qps
    REC_W = PB.rec_w
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nctas, _, _ = cute.arch.grid_dim()
    warp = cute.arch.warp_idx()
    lane = tidx % 32

    smem = cutlass.Array(I32, PB.smem // 4, space=cutlass.AddressSpace.smem, alignment=1024)
    b_full = cutlass.Array(cutlass.Int64, PB.nst, space=cutlass.AddressSpace.smem, alignment=8)
    b_exp = cutlass.Array(cutlass.Int64, PB.nst, space=cutlass.AddressSpace.smem, alignment=8)
    b_emp = cutlass.Array(cutlass.Int64, PB.nst, space=cutlass.AddressSpace.smem, alignment=8)
    b_af = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    b_ae = cutlass.Array(cutlass.Int64, 2, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    prims.griddepcontrol("launch_dependents")

    sbase = smem.data_ptr()
    zero4 = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
    for it in cutlass.range(tidx, PB.smem // 16, PB_THREADS, unroll=1):
        (sbase + it * 4).store(zero4, alignment=16)
    if warp == 0:
        if prims.elect_sync():
            prims.prefetch_tensormap(tma_q.get_ptr())
            prims.prefetch_tensormap(tma_do.get_ptr())
            prims.prefetch_tensormap(tma_rk.get_ptr())
            prims.prefetch_tensormap(tma_rv.get_ptr())
            for s in cutlass.range_constexpr(PB.nst):
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
        gs = I32(0)
        n_nx = I32(0)
        o_nx = I32(0)
        q_nx = I32(0)
        n_n2 = I32(0)
        o_n2 = I32(0)
        if bid < Ktot:
            w0 = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + w0).load()
            o_nx = (p_key_slot_offsets + w0).load()
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
        if bid + nctas < Ktot:
            w1 = _active_key(p_active, n0, KR, bid + nctas)
            n_n2 = (p_key_pair_counts + w1).load()
            o_n2 = (p_key_slot_offsets + w1).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            o = o_nx
            q0v = q_nx
            n_nx = n_n2
            o_nx = o_n2
            q_nx = I32(0)
            if lane < n_nx:
                q_nx = (p_slot_queries + o_nx + lane).load()
            i2 = i + 2 * nctas
            if i2 < Ktot:
                w2 = _active_key(p_active, n0, KR, i2)
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
                for j2 in cutlass.range(0, nloc, QPS, unroll=1):
                    qids = []
                    for i in cutlass.range_constexpr(QPS):
                        qids.append(prims.shfl_sync(0xFFFFFFFF, qv, j2 + i, 0x1F, prims.Shfl.IDX))
                    nq = nloc - j2
                    if nq > QPS:
                        nq = I32(QPS)
                    slot = gs % PB.nst
                    _wait(b_emp.subview(slot), ((gs // PB.nst) & 1) ^ 1)
                    if prims.elect_sync():
                        sb = sbase + (slot * PB.stage) // 4
                        prims.mbarrier_arrive_expect_tx(b_full.subview(slot), nq * PB.tx_per_query)
                        for i in cutlass.range_constexpr(QPS):
                            if i < nq:
                                if cutlass.const_expr(HKV == 2):
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_q + s * PB.slab + i * 3072) // 4, tma_q.get_ptr(), (s * 64, 0, qids[i]), b_full.subview(slot)
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_do + s * PB.slab + i * 3072) // 4, tma_do.get_ptr(), (s * 64, 0, qids[i]), b_full.subview(slot)
                                        )
                                elif cutlass.const_expr(GH == GQ):
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_q + i * 4 * PB.slab) // 4, tma_q.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                                    prims.cp_async_bulk_tensor_shared_cta_global(
                                        sb + (PB.off_do + i * 4 * PB.slab) // 4, tma_do.get_ptr(), (0, 0, 0, qids[i]), b_full.subview(slot)
                                    )
                                else:
                                    for s in cutlass.range_constexpr(4):
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_q + (i * 4 + s) * PB.slab) // 4, tma_q.get_ptr(), (0, 0, s, qids[i]), b_full.subview(slot)
                                        )
                                        prims.cp_async_bulk_tensor_shared_cta_global(
                                            sb + (PB.off_do + (i * 4 + s) * PB.slab) // 4, tma_do.get_ptr(), (0, 0, s, qids[i]), b_full.subview(slot)
                                        )
                        prims.cp_async_bulk_shared_cluster_global(
                            sb + PB.off_rec // 4,
                            p_pair_records + cutlass.Int64(o + qb + j2) * REC_W,
                            b_full.subview(slot),
                            nq * (REC_W * 4),
                            l2_cache_hint=cutlass.Int64(L2_EVICT_FIRST),
                        )
                    prims.bar_warp_sync(0xFFFFFFFF)
                    gs = gs + 1
    elif warp == 1:
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
        d_q = _desc(sbase + PB.off_q // 4, PB.slab, 1024, SW128)
        d_do = _desc(sbase + PB.off_do // 4, PB.slab, 1024, SW128)
        d_p = _desc(sbase + PB.off_p // 4, 256, 128, SWN)
        d_ds = _desc(sbase + PB.off_ds // 4, 256, 128, SWN)
        gs = I32(0)
        ia = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            if n > 0:
                ab = ia & 1
                _wait(b_ae.subview(ab), ((ia >> 1) & 1) ^ 1)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                nst = (n + (QPS - 1)) // QPS
                for si in cutlass.range(nst, unroll=1):
                    slot = gs % PB.nst
                    ph = (gs // PB.nst) & 1
                    _wait(b_full.subview(slot), ph)
                    _wait(b_exp.subview(slot), ph)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    soff = slot * PB.stage
                    if cutlass.const_expr(HKV == 1):
                        nq_s = n - QPS * si
                        if nq_s > QPS:
                            nq_s = I32(QPS)
                        if prims.elect_sync():
                            for qq in cutlass.range(nq_s, unroll=1):
                                accf = (si > 0) | (qq > 0)
                                for hh in cutlass.range_constexpr(2):
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + hh * 16), F32),
                                        d_do.advance_start_address(soff + qq * 4 * PB.slab + hh * 2 * PB.slab),
                                        d_p.advance_start_address(soff + qq * 512),
                                        idesc,
                                        accf,
                                    )
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                        d_q.advance_start_address(soff + qq * 4 * PB.slab + hh * 2 * PB.slab),
                                        d_ds.advance_start_address(soff + qq * 512),
                                        idesc,
                                        accf,
                                    )
                            prims.tcgen05_commit(b_emp.subview(slot), group=prims.CTAGroup.CTA_1)
                    elif cutlass.const_expr(SK > 0):
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
                                            d_do.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                            d_p.advance_start_address(soff + kk * 512),
                                            idesc,
                                            accf,
                                        )
                                        prims.tcgen05_mma(
                                            prims.Tcgen05MMAKind.F16,
                                            prims.CTAGroup.CTA_1,
                                            prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                            d_q.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
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
                                        d_do.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
                                        d_p.advance_start_address(soff + kk * 512),
                                        idesc,
                                        accf,
                                    )
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        prims.make_tmem_ptr(tmem_raw + (ab * 64 + 32 + hh * 16), F32),
                                        d_q.advance_start_address(soff + hh * 2 * PB.slab + kk * 2048),
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
        t = tidx - 64
        if cutlass.const_expr(HKV == 2):
            jq = I32(0)
            if t >= 24:
                jq = I32(1)
            hq = t - 24 * jq
            hkv = I32(0)
            if hq >= 12:
                hkv = I32(1)
            row_on = t < 48
        else:
            jq = t >> 4
            hq = t & 15
            hkv = I32(0)
            row_on = hq < 12
        rw = hq >> 1
        rs = (hq & 1) * 16
        toff = ((t >> 3) * 256 + (t & 7) * 16) // 4
        gs = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            nst = (n + (QPS - 1)) // QPS
            for si in cutlass.range(nst, unroll=1):
                slot = gs % PB.nst
                ph = (gs // PB.nst) & 1
                nq = n - QPS * si
                _wait(b_full.subview(slot), ph)
                if row_on:
                    sb = sbase + (slot * PB.stage) // 4
                    pw0 = I32(0)
                    pw1 = I32(0)
                    dw0 = I32(0)
                    dw1 = I32(0)
                    if jq < nq:
                        rb = sb + (PB.off_rec // 4 + jq * REC_W + rw)
                        pv = []
                        dv_ = []
                        for tt in cutlass.range_constexpr(4):
                            pv.append(((rb + tt * (6 * HKV)).load() >> rs) & I32(0xFFFF))
                            dv_.append(((rb + REC_W // 2 + tt * (6 * HKV)).load() >> rs) & I32(0xFFFF))
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
                    (sb + (PB.off_p // 4 + toff)).store(vp, alignment=16)
                    (sb + (PB.off_ds // 4 + toff)).store(vd, alignment=16)
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_exp.subview(slot))
                gs = gs + 1
    else:
        qd = warp - 4
        stg = sbase + (PB.off_st + qd * PB.st_w * 4) // 4
        ia = I32(0)
        n_nx = I32(0)
        k_nx = I32(0)
        if bid < Ktot:
            k_nx = _active_key(p_active, n0, KR, bid)
            n_nx = (p_key_pair_counts + k_nx).load()
        for i in cutlass.range(bid, Ktot, nctas, unroll=1):
            n = n_nx
            w = k_nx
            if i + nctas < Ktot:
                k_nx = _active_key(p_active, n0, KR, i + nctas)
                n_nx = (p_key_pair_counts + k_nx).load()
            if n > 0:
                tok = w % KR
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
                prims.cp_async_bulk_wait_group(0, read=True)
                prims.bar_warp_sync(0xFFFFFFFF)
                for bx in cutlass.range_constexpr(4):
                    for kv in cutlass.range_constexpr(HKV):
                        for tt in cutlass.range_constexpr(4):
                            col = 4 * kv + tt + 16 * (bx & 1)
                            if cutlass.const_expr((bx >> 1) == 0):
                                val = va[col]
                            else:
                                val = vb[col]
                            (stg + (bx * PB.box_w + (tt * HKV + kv) * 32 + lane)).store(prims.mov_b32(val, target_type=I32))
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                prims.bar_warp_sync(0xFFFFFFFF)
                if lane == 0:
                    for bx in cutlass.range_constexpr(4):
                        if cutlass.const_expr((bx >> 1) == 0):
                            prims.cp_async_bulk_tensor_reduce(tma_rv.get_ptr(), stg + bx * PB.box_w, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, tok])
                        else:
                            prims.cp_async_bulk_tensor_reduce(tma_rk.get_ptr(), stg + bx * PB.box_w, prims.TMARedux.ADD, [(bx & 1) * 128 + qd * 32, 0, tok])
                prims.cp_async_bulk_commit_group()
                ia = ia + 1
        prims.cp_async_bulk_wait_group(0)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == 1:
        prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), 128, group="cta_1")


qsa_bwd_sparse_key_sm107_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    active_keys: cute.Tensor,
    active_count: cute.Tensor,
    nctas: I32,
    KR: I32,
    SK: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    T = q.shape[0]
    if cutlass.const_expr(HKV == 2):
        tma_q = cuda.create_tensor_map_tiled_from_view(q, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
        tma_do = cuda.create_tensor_map_tiled_from_view(do, box_dims=(1, 24, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    else:
        q_layout = cute.make_layout((T, GH, 4, 64), stride=(GH * D, D, 64, 1))
        if cutlass.const_expr(GH == GQ):
            qbox4 = (1, GH, 4, 64)
        else:
            qbox4 = (1, GH, 1, 64)
        tma_q = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(q.iterator, q_layout), box_dims=qbox4, stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
        tma_do = cuda.create_tensor_map_tiled_from_view(
            cute.make_tensor(do.iterator, q_layout), box_dims=qbox4, stride_order=(3, 1, 2, 0), swizzle=cuda.TensorMapSwizzle.s128b
        )
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
    qsa_bwd_sparse_key_sm107_kernel(
        tma_q, tma_do, tma_rk, tma_rv, pair_records, key_slot_offsets, key_pair_counts, slot_queries, active_keys, active_count, KR, SK, HKV, GH
    ).launch(grid=(nctas, 1, 1), block=(PB_THREADS, 1, 1), stream=stream)


@cute.kernel
def qsa_bwd_dense_pair_sm107_kernel(
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
    g_seq_start: cute.Tensor,
    g_dense_hi: cute.Tensor,
    g_unit: cute.Tensor,
    g_first_group: cute.Tensor,
    T: I32,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
):
    HQ = GH * HKV
    # the TMEM K^T and the spinning waits only pay off in these geometries
    KT_TMEM = GH == GQ or DQ_PARTS > 1
    SPIN = GH == GQ and DQ_PARTS > 1
    TM_NCOLS = DP_TM_COLS if KT_TMEM else 512
    P_LO = DQ_PARTS > 1
    PXB = DP_CM + DP_CM // 2 if P_LO else DP_CM
    R0 = min(DP_NX, (2 * DP_NX) // DQ_PARTS, (DP_NX * DP_CM) // PXB)
    DQL = R0 - 1
    tidx, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    warp = cute.arch.warp_idx()
    lane = tidx % 32
    rank = bid & 1
    pr = bid >> 1
    if cutlass.const_expr(HKV == 2):
        hk = pr & 1
        rest = pr >> 1
    else:
        hk = I32(0)
        rest = pr
    part = rest // ntiles
    tile = rest - part * ntiles
    a0 = tile * DP_TT
    p_seq_start = g_seq_start.iterator.raw_ptr()
    p_dense_hi = g_dense_hi.iterator.raw_ptr()
    dh = I32(0)
    for u in cutlass.range_constexpr(DP_TT // 32):
        tt = a0 + u * 32 + lane
        if tt < T:
            x = (p_dense_hi + tt).load()
            if x > dh:
                dh = x
    dh = cute.arch.warp_reduction_max(dh)
    g_cap = (T + (DP_G - 1)) // DP_G
    g_top = (dh + (DP_G - 1)) // DP_G
    if g_top > g_cap:
        g_top = g_cap
    g_lo = a0 // DP_G
    fg = (g_first_group.iterator.raw_ptr() + tile).load()
    if fg < g_lo:
        g_lo = fg
    Ut = U
    if adapt != 0:
        Ut = g_unit.iterator.raw_ptr().load()
        if Ut > U:
            Ut = U
    need = (g_top - g_lo + nparts - 1) // nparts
    if need > Ut:
        Ut = need
    g_top = g_top - part * Ut
    g_bot = g_top - Ut
    if g_bot > g_lo:
        g_lo = g_bot
    nk = g_top - g_lo
    if a0 >= dh:
        nk = I32(0)
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
        b_ks = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
        tmem_slot = cutlass.Array(I32, 1, space=cutlass.AddressSpace.smem, alignment=4)

        sbase = smem.data_ptr()
        if cutlass.const_expr(GH < GQ):
            zq = cutlass.Vector.from_elements([I32(0), I32(0), I32(0), I32(0)], I32)
            for it in cutlass.range(tidx, (DP_OFF_P - DP_OFF_E) // 16, DP_THREADS, unroll=1):
                (sbase + DP_OFF_E // 4 + it * 4).store(zq, alignment=16)
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
                for s in cutlass.range_constexpr(DP_NX):
                    if s < nk:
                        if rank == 0:
                            if cutlass.const_expr(s < R0):
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(s), DQ_PARTS * DP_CM)
                        else:
                            if cutlass.const_expr(s < R0):
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(s), PXB)
                prims.mbarrier_init(b_dqf, 1)
                prims.mbarrier_init(b_dqe, 4)
                prims.mbarrier_init(b_kvd, 1)
                prims.mbarrier_init(b_ks, 4)
        if warp == 1:
            prims.tcgen05_alloc(tmem_slot, TM_NCOLS, is_exclusive=KT_TMEM, group="cta_1")
            prims.tcgen05_relinquish_alloc_permit(group="cta_1")
        prims.fence_mbarrier_init()
        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

        tmem_raw = tmem_slot.load()
        tmem_col0 = tmem_raw & I32(0xFFFF)
        tmem_row0 = tmem_raw >> I32(16)
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
                        sbase + (DP_OFF_T + s * DP_KSLAB) // 4, tma_kd.get_ptr(), ((rank * 2 + s) * 64, hk, a0), b_tile
                    )
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                se = k % DP_NE
                _wait(b_ee.subview(se), ((k // DP_NE) & 1) ^ 1, SPIN)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_ef.subview(se), 4 * DP_G * GH * 128)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            _dense_rows_load(sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_qd, s, hk, g, b_ef.subview(se), GH)
                        else:
                            _dense_rows_load(sbase + (DP_OFF_E + se * DP_HS + s * DP_QSLAB) // 4, tma_dod, s, hk, g, b_ef.subview(se), GH)
        elif warp == 3:
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                sl = k % DP_NL
                _wait(b_le.subview(sl), ((k // DP_NL) & 1) ^ 1, SPIN)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(b_lf.subview(sl), 4 * DP_G * GH * 128)
                    for s in cutlass.range_constexpr(4):
                        if rank == 0:
                            _dense_rows_load(sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_dod, s, hk, g, b_lf.subview(sl), GH)
                        else:
                            _dense_rows_load(sbase + (DP_OFF_L + sl * DP_HS + s * DP_QSLAB) // 4, tma_qd, s, hk, g, b_lf.subview(sl), GH)
        elif warp == 1:
            idesc_s = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=0, b_major=0)
            idesc_q = prims.Tcgen05InstrDesc.build(c_dtype=F32, a_dtype=BF16, b_dtype=BF16, n_dim=DP_N, m_dim=128, a_major=0 if KT_TMEM else 1, b_major=1)
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
            _wait(b_tile, 0, SPIN)
            _wait(b_vt, 0, SPIN)
            _wait(b_ks, 0, SPIN)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            d_x0 = _desc(sbase + DP_OFF_P // 4, 768, 128, SWN)
            for k in cutlass.range(nk + DQL, unroll=1):
                se = k % DP_NE
                if k < nk:
                    if k >= 1:
                        _wait(b_sr, (k - 1) & 1, SPIN)
                    _wait(b_ef.subview(se), (k // DP_NE) & 1, SPIN)
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
                    _wait(b_lf.subview(sl), (kb // DP_NL) & 1, SPIN)
                    _wait(b_pf.subview(xb), (kb // DP_NX) & 1, SPIN)
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
                            _wait(b_dqe, (kb - 1) & 1, SPIN)
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
                            for part in cutlass.range_constexpr(DQ_PARTS):
                                for kk in cutlass.range_constexpr(8):
                                    if cutlass.const_expr(KT_TMEM):
                                        a_q = prims.make_tmem_ptr(tmem_raw + (DP_TM_KT + kk * 8), BF16)
                                    else:
                                        a_q = d_kmn.advance_start_address(kk * 2048)
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        t_dq,
                                        a_q,
                                        d_dsb.advance_start_address(part * DP_CM + kk * 1536),
                                        idesc_q,
                                        (part > 0) | (kk > 0),
                                    )
                            prims.tcgen05_commit(b_pe.subview(xb), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_le.subview(sl), group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            if kb == nk - 1:
                                prims.tcgen05_commit(b_kvd, group=prims.CTAGroup.CTA_1)
                if rank == 0:
                    if k >= DQL:
                        kq = k - DQL
                        xq = kq % R0
                        _wait(b_xr.subview(xq), (kq // R0) & 1, SPIN)
                        prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                        if kq + R0 < nk:
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xq), DQ_PARTS * DP_CM)
                        if kq >= 1:
                            _wait(b_dqe, (kq - 1) & 1, SPIN)
                        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if prims.elect_sync():
                            for part in cutlass.range_constexpr(DQ_PARTS):
                                for kk in cutlass.range_constexpr(8):
                                    if cutlass.const_expr(KT_TMEM):
                                        a_q = prims.make_tmem_ptr(tmem_raw + (DP_TM_KT + kk * 8), BF16)
                                    else:
                                        a_q = d_kmn.advance_start_address(kk * 2048)
                                    prims.tcgen05_mma(
                                        prims.Tcgen05MMAKind.F16,
                                        prims.CTAGroup.CTA_1,
                                        t_dq,
                                        a_q,
                                        d_x0.advance_start_address((xq * DQ_PARTS + part) * DP_CM + kk * 1536),
                                        idesc_q,
                                        (part > 0) | (kk > 0),
                                    )
                            prims.tcgen05_commit(b_dqf, group=prims.CTAGroup.CTA_1)
                            prims.tcgen05_commit(b_xe.subview(xq), group=prims.CTAGroup.CTA_1)
        elif warp >= 4 and warp < 12:
            hf = (warp - 4) >> 2
            qd = warp & 3
            r = qd * 32 + lane
            tok = a0 + r
            row_ok = tok < T
            c_s = scale * F32(LOG2E)
            xoff = (DP_OFF_P + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            xoffd = (DP_OFF_DS + (r >> 3) * 768 + (r & 7) * 16) // 4 + hf * 96
            xlo4 = (DP_OFF_P + DP_CM) // 4 + (hf * 128 + r) * 4
            xlo2 = (DP_OFF_P + DP_CM + 4096) // 4 + (hf * 128 + r) * 2
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
                g_n = g_top - 1 - _rot(I32(0), off, nk)
                sw_n = []
                bq_n = []
                av_n = []
                for jj in cutlass.range_constexpr(2):
                    qc = g_n * DP_G + hf * 2 + jj
                    if qc >= T:
                        qc = T - 1
                    sw_j, bq_j = _dense_sel(p_dense_block_bitmap, p_seq_start, qc, tok, NW)
                    sw_n.append(sw_j)
                    bq_n.append(bq_j)
                    ab = qc * HQ + hk * GH
                    av_n.append(_lse_rows(g_lse, qc, hk * GH, GH))
                for k in cutlass.range(nk, unroll=1):
                    g = g_top - 1 - _rot(k, off, nk)
                    sw = sw_n
                    bqs = bq_n
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    sw_n = []
                    bq_n = []
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= T:
                            qc = T - 1
                        sw_j, bq_j = _dense_sel(p_dense_block_bitmap, p_seq_start, qc, tok, NW)
                        sw_n.append(sw_j)
                        bq_n.append(bq_j)
                        ab = qc * HQ + hk * GH
                        av_n.append(_lse_rows(g_lse, qc, hk * GH, GH))
                    _wait(b_sf, k & 1, SPIN)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    opk = []
                    pall = []
                    for jj in cutlass.range_constexpr(2):
                        qi = g * DP_G + hf * 2 + jj
                        sel = (sw[jj] >> (bqs[jj] & 31)) & 1
                        valid = row_ok & (qi < T) & (sel != 0)
                        ow = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            e = cute.math.exp2(x * c_s - a * F32(LOG2E), fastmath=True)
                            pj = F32(0.0)
                            if valid:
                                pj = e
                            ow.append(pj)
                            pall.append(pj)
                        pk = cutlass.Vector.from_elements(ow, F32).to(BF16).bitcast(I32)
                        for u in cutlass.range_constexpr(GQ // 2):
                            opk.append(pk[u])
                    ovec = cutlass.Vector.from_elements(opk, I32)
                    xvec = ovec
                    if cutlass.const_expr(P_LO):
                        pb = cutlass.Vector.from_elements(pall, F32).bitcast(I32)
                        xw = []
                        for w in cutlass.range_constexpr(GQ):
                            xw.append(prims.prmt(pb[2 * w], 0x7632, prims.PermuteMode.DEFAULT, hi=pb[2 * w + 1]))
                        lw = []
                        for w in cutlass.range_constexpr(GQ // 2):
                            t0 = prims.prmt(pb[4 * w], 0x0051, prims.PermuteMode.DEFAULT, hi=pb[4 * w + 1])
                            t1 = prims.prmt(pb[4 * w + 2], 0x5100, prims.PermuteMode.DEFAULT, hi=pb[4 * w + 3])
                            lw.append(prims.prmt(t0, 0x7610, prims.PermuteMode.DEFAULT, hi=t1))
                        xvec = cutlass.Vector.from_elements(xw, I32)
                        lvec = cutlass.Vector.from_elements(lw, I32)
                    xb = k % DP_NX
                    xp = k % R0
                    if k >= 1:
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1, SPIN)
                    if k >= R0:
                        _wait(b_xe.subview(k % R0), ((k // R0) & 1) ^ 1, SPIN)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), ovec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), ovec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), ovec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), ovec[4:12])
                    rb = prims.mapa(b_xr.subview(xp), peer)
                    for nc in cutlass.range_constexpr(3):
                        _st_async16(prims.mapa(sbase + xoff + xp * (PXB // 4) + nc * 32, peer), xvec[nc * 4 : nc * 4 + 4], rb)
                    if cutlass.const_expr(P_LO):
                        _st_async16(prims.mapa(sbase + xlo4 + xp * (PXB // 4), peer), lvec[0:4], rb)
                        _st_async8(prims.mapa(sbase + xlo2 + xp * (PXB // 4), peer), lvec[4:6], rb)
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
                    if qc >= T:
                        qc = T - 1
                    ab = qc * HQ + hk * GH
                    av_n.append(_delta_rows(p_delta, ab, GH))
                for k in cutlass.range(nk, unroll=1):
                    av = av_n
                    kn = k + 1
                    if kn >= nk:
                        kn = nk - 1
                    g_n = g_top - 1 - _rot(kn, off, nk)
                    av_n = []
                    for jj in cutlass.range_constexpr(2):
                        qc = g_n * DP_G + hf * 2 + jj
                        if qc >= T:
                            qc = T - 1
                        ab = qc * HQ + hk * GH
                        av_n.append(_delta_rows(p_delta, ab, GH))
                    _wait(b_sf, k & 1, SPIN)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                    xv0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol).as_ptr(F32), num=16)
                    xv1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, scol + 16).as_ptr(F32), num=8)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_sr)
                    xb = k % DP_NX
                    xp = k % R0
                    _wait(b_xr.subview(xp), (k // R0) & 1, SPIN)
                    pin = []
                    for nc in cutlass.range_constexpr(3):
                        pin.append((sbase + xoff + xp * (PXB // 4) + nc * 32).load(count=4, alignment=16))
                    if cutlass.const_expr(P_LO):
                        lo4 = (sbase + xlo4 + xp * (PXB // 4)).load(count=4, alignment=16)
                        lo2 = (sbase + xlo2 + xp * (PXB // 4)).load(count=2, alignment=8)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if k + R0 < nk:
                        if warp == 4:
                            if prims.elect_sync():
                                prims.mbarrier_arrive_expect_tx(b_xr.subview(xp), PXB)
                    dpk = []
                    for jj in cutlass.range_constexpr(2):
                        dw = []
                        for hh in cutlass.range_constexpr(GQ):
                            n = jj * GQ + hh
                            a = av[jj][hh]
                            if cutlass.const_expr(n < 16):
                                x = xv0[n]
                            else:
                                x = xv1[n - 16]
                            dw.append(x - a)
                        for hp in cutlass.range_constexpr(GQ // 2):
                            n0 = jj * GQ + 2 * hp
                            pw = pin[n0 // 8][(n0 % 8) // 2]
                            if cutlass.const_expr(P_LO):
                                if cutlass.const_expr(n0 < 16):
                                    lw = lo4[n0 // 4]
                                else:
                                    lw = lo2[n0 // 4 - 4]
                                sb = (n0 % 4) * 0x11
                                p0 = prims.prmt(lw, 0x5400 + sb, prims.PermuteMode.DEFAULT, hi=pw)
                                p1 = prims.prmt(lw, 0x7611 + sb, prims.PermuteMode.DEFAULT, hi=pw)
                                pv = cutlass.Vector.from_elements([p0, p1], I32).bitcast(F32)
                            else:
                                pv = cutlass.Vector.from_elements([pw], I32).bitcast(BF16).to(F32)
                            d0 = F32(0.0)
                            d1 = F32(0.0)
                            if pv[0] != 0.0:
                                d0 = pv[0] * dw[2 * hp] * scale
                            if pv[1] != 0.0:
                                d1 = pv[1] * dw[2 * hp + 1] * scale
                            dpk.append(d0)
                            dpk.append(d1)
                    dparts = []
                    res = cutlass.Vector.from_elements(dpk, F32)
                    for part in cutlass.range_constexpr(DQ_PARTS):
                        bp = res.to(BF16)
                        if cutlass.const_expr(part + 1 < DQ_PARTS):
                            res = res - bp.to(F32)
                        dparts.append(bp.bitcast(I32))
                    dvec = dparts[0]
                    if k >= 1:
                        kp = k - 1
                        _wait(b_pe.subview(kp % DP_NX), (kp // DP_NX) & 1, SPIN)
                    if hf == 0:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[0]).as_ptr(I32), dvec[0:8])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1]).as_ptr(I32), dvec[8:12])
                    else:
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[1] + 4).as_ptr(I32), dvec[0:4])
                        prims.tcgen05_st("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_PA[2]).as_ptr(I32), dvec[4:12])
                    x0 = k % R0
                    rb = prims.mapa(b_xr.subview(x0), peer)
                    for part in cutlass.range_constexpr(DQ_PARTS):
                        for nc in cutlass.range_constexpr(3):
                            (sbase + xoffd + part * (DP_CM // 4) + nc * 32).store(dparts[part][nc * 4 : nc * 4 + 4], alignment=16)
                            _st_async16(prims.mapa(sbase + xoff + (x0 * DQ_PARTS + part) * (DP_CM // 4) + nc * 32, peer), dparts[part][nc * 4 : nc * 4 + 4], rb)
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
                    prims.bar_warp_sync(0xFFFFFFFF)
                    if prims.elect_sync():
                        prims.mbarrier_arrive(b_pf.subview(xb))
            _wait(b_kvd, 0, SPIN)
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
            _wait(b_tile, 0, SPIN)
            kt = sbase + DP_OFF_T // 4
            for it in cutlass.range_constexpr((2 * DP_KSLAB) // (16 * 128)):
                koff = (it * 128 + (tidx - 384)) * 4
                w4 = (kt + koff).load(count=4, alignment=16)
                ws = []
                nbad = I32(0)
                for e in cutlass.range_constexpr(4):
                    w = w4[e]
                    if (w & I32(0x7F80)) == I32(0x7F80):
                        w = w & I32(-65536)
                        nbad = nbad + 1
                    if (w & I32(0x7F800000)) == I32(0x7F800000):
                        w = w & I32(0xFFFF)
                        nbad = nbad + 1
                    ws.append(w)
                if nbad > 0:
                    (kt + koff).store(cutlass.Vector.from_elements(ws, I32), alignment=16)
            if cutlass.const_expr(KT_TMEM):
                # each warp's dims read keys that the other warps sanitized
                prims.barrier_cta_sync(1, thread_count=128)
                kmi = lane >> 3
                krho = lane & 7
                # ldmatrix tile 0 holds keys {0, 1, 4, 5, 8, 9, 12, 13} of a 16-key step and tile 1 the rest, so the
                # 16x256b store puts keys 2j and 2j + 1 in TMEM column j, the row order of dS^T
                kkey = 4 * (krho >> 1) + (krho & 1) + 2 * (kmi & 1)
                for half in cutlass.range_constexpr(2):
                    kchunk = (qd & 1) * 4 + half * 2 + (kmi >> 1)
                    for kk in cutlass.range_constexpr(DP_TT // 16):
                        key = kk * 16 + kkey
                        frag = prims.ldmatrix(kt + ((qd >> 1) * DP_KSLAB + key * 128 + ((kchunk ^ (key & 7)) << 4)) // 4, 4, prims.MMALayout.COL)
                        prims.tcgen05_st(
                            "16x256b",
                            prims.TmemAddr.from_row_col(tmem_row0 + qd * 32 + half * 16, tmem_col0 + DP_TM_KT + kk * 8).as_ptr(I32),
                            frag,
                        )
                prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            else:
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.bar_warp_sync(0xFFFFFFFF)
            if prims.elect_sync():
                prims.mbarrier_arrive(b_ks)
            stg0 = sbase + (DP_OFF_ST + qd * DP_ST_W) // 4
            STG2 = R0 * DQ_PARTS * DP_CM + 4 * DP_ST_W <= 2 * DP_NX * DP_CM
            stg1 = sbase + (DP_OFF_P + R0 * DQ_PARTS * DP_CM + qd * DP_ST_W) // 4
            for k in cutlass.range(nk, unroll=1):
                g = g_top - 1 - _rot(k, off, nk)
                _wait(b_dqf, k & 1, SPIN)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                vh0 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ).as_ptr(F32), num=32)
                vh1 = prims.tcgen05_ld("32x32b", prims.TmemAddr.from_row_col(tmem_row0 + qd * 32, tmem_col0 + DP_TM_DQ + 32).as_ptr(F32), num=DP_N - 32)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                prims.bar_warp_sync(0xFFFFFFFF)
                if prims.elect_sync():
                    prims.mbarrier_arrive(b_dqe)
                stg = stg0
                if cutlass.const_expr(STG2):
                    if rank == 0:
                        if (k & 1) == 1:
                            stg = stg1
                        prims.cp_async_bulk_wait_group(1, read=True)
                    else:
                        prims.cp_async_bulk_wait_group(0, read=True)
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
                    if cutlass.const_expr(GH == GQ):
                        prims.cp_async_bulk_tensor_reduce(tma_dq.get_ptr(), stg, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GQ, g * DP_G])
                    else:
                        for qq in cutlass.range_constexpr(DP_G):
                            prims.cp_async_bulk_tensor_reduce(
                                tma_dq.get_ptr(), stg + qq * GQ * 32, prims.TMARedux.ADD, [rank * 128 + qd * 32, hk * GH, g * DP_G + qq]
                            )
                prims.cp_async_bulk_commit_group()
            prims.cp_async_bulk_wait_group(0)

        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()
        if warp == 1:
            prims.tcgen05_dealloc(prims.make_tmem_ptr(tmem_raw, F32), TM_NCOLS, is_exclusive=KT_TMEM, group="cta_1")

    if cutlass.const_expr(PDL > 0):
        if tidx == 0:
            nb, _, _ = cute.arch.grid_dim()
            old = prims.atomicrmw("add", g_dense_done_count.iterator.raw_ptr(), I32(1), mem_order=prims.MemOrder.RELAXED, syncscope=prims.MemScope.GPU)
            if old == nb - 1:
                prims.griddepcontrol("wait")


qsa_bwd_dense_pair_sm107_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


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
    seq_start: cute.Tensor,
    dense_hi: cute.Tensor,
    unit: cute.Tensor,
    first_group: cute.Tensor,
    NW: I32,
    ntiles: I32,
    U: I32,
    nparts: I32,
    adapt: I32,
    scale: F32,
    PDL: cutlass.Constexpr[int],
    HKV: cutlass.Constexpr[int],
    GH: cutlass.Constexpr[int],
    DQ_PARTS: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    HQ = GH * HKV
    T = k.shape[0]
    tma_kd = cuda.create_tensor_map_tiled_from_view(k, box_dims=(DP_TT, 1, 64), stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    if cutlass.const_expr(GH == GQ):
        qbox = (DP_G, GQ, 64)
        dqbox = (DP_G, GQ, 32)
    else:
        qbox = (1, GH, 64)
        dqbox = (1, GH, 32)
    tma_qd = cuda.create_tensor_map_tiled_from_view(q, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dod = cuda.create_tensor_map_tiled_from_view(do, box_dims=qbox, stride_order=(2, 1, 0), swizzle=cuda.TensorMapSwizzle.s128b)
    tma_dq = cuda.create_tensor_map_tiled_from_view(
        cute.make_tensor(dq_acc.iterator, cute.make_layout((T, HQ, D), stride=(HQ * D, D, 1))),
        box_dims=dqbox,
        stride_order=(2, 1, 0),
        swizzle=cuda.TensorMapSwizzle.none,
    )
    qsa_bwd_dense_pair_sm107_kernel(
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
        seq_start,
        dense_hi,
        unit,
        first_group,
        I32(T),
        NW,
        ntiles,
        U,
        nparts,
        adapt,
        scale,
        PDL,
        HKV,
        GH,
        DQ_PARTS,
    ).launch(grid=(nparts * ntiles * 2 * HKV, 1, 1), block=(DP_THREADS, 1, 1), cluster=(2, 1, 1), stream=stream, use_pdl=(PDL == 1))
