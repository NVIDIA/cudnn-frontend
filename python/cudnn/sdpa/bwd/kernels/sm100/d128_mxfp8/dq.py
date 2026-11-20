# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent dQ kernel of the SM100 / SM103 D128 MXFP8 backward chain (two-CTA clusters)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, GridConstant, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl import barrier
from cudnn.frost.tile_dsl.consume import consume_then_arrive, consume_then_arrive_on_peer, release_zero
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.tmem import tmem_alloc
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.common import (
    _G,
    _abs_ftz,
    _add2,
    _atom_of,
    _bulk_g2s,
    _bulk_g2s_mc,
    _dsc,
    _e8m0_ceil,
    _fma2,
    _fma_ftz,
    _gword_ptr,
    _i32_ptr,
    _lds128,
    _lds_f32,
    _lds_u32,
    _lds_u32_ordered,
    _mask_above_if,
    _max3abs,
    _max_ftz,
    _mbar_arrive,
    _mbar_arrive_if,
    _mbar_expect_tx,
    _mbar_ptr,
    _mbar_wait_if,
    _mul2,
    _neg_ftz,
    _pack_e4m3x4,
    _select_u32,
    _sts128,
    _tc_fence_after,
    _tc_fence_before,
    _tcw_commit,
    _tcw_commit_mc,
    _tcw_cp_sf,
    _tcw_elect,
    _tcw_mma,
    _tm_ld32,
    _tm_st1,
    _tm_st16,
    _tm_st8,
    _tm_wait_ld,
    _tm_wait_st,
    _tma_2d,
    _tma_3d,
    _tma_rows_mc,
    _tma_store_rows,
    _trap,
    _wsrow,
)

# ================================================================ persistent dQ kernel
# A persistent grid of 2-CTA clusters pulls unit pairs (head h, query blocks 2m, 2m+1; rank r of the cluster owns
# block 2m+r) from a global counter (reset by the prep kernel), so the per-unit fixed cost of a fresh CTA (launch,
# TMEM / barrier setup, Q / dO load, pipeline fill, epilogue) is paid once per cluster and the unit transitions
# overlap:
#  - warp 2 (unit warp) of rank 0 grabs units and publishes them to both CTAs (2-slot ring, st.async); every role reads
#    each unit index once. It also stages the next unit's Q / dO tiles, their scale atoms and stats / delta rows in SMEM.
#  - the softmax warps copy the staged Q / dO into TMEM as soon as S/dP of the unit's last tile have landed (the
#    previous Q / dO are dead then), so the MMA warp issues S/dP of the next unit's first tile while the softmax warps
#    finish the last tile and the dQ epilogue; the next unit's first dQ MMA waits only for the epilogue's TMEM read.
# Unit p = (query head, pair m of 128-row blocks 2m, 2m+1). Causal: the walk stops at the pair's higher diagonal block,
# so units are handed out longest first (pair index descending across heads) to balance the persistent grid.
_OFF = _G.SMEM_P_OFF


@cute.jit
def _umin(a, b):
    return _select_u32(Uint32(a < b), Uint32(a), Uint32(b))


# ---- device helpers: @cute.jit (inlined into the kernels; no closures over staged values); sm: SmemP base (u32)
@cute.jit
def _a(sm, name, i=0, stride=0):
    """shared address of SmemP field ``name`` (element i of ``stride`` bytes)"""
    if cutlass.const_expr(isinstance(i, int)):
        return sm + Uint32(_OFF[name] + i * stride)
    return sm + Uint32(_OFF[name]) + i * Uint32(stride)


@cute.jit
def _word_ptr(addr):
    """raw shared address -> u32 pointer (st.async destination)"""
    return cute.make_ptr(cutlass.Uint32, addr, cute.AddressSpace.smem, assumed_align=4)


@cute.jit
def _read_unit(sm, k):
    """unit index k of this cluster, read once by every reader warp (one arrival per warp on rank 0's sched_empty).
    The index lands by st.async with complete_tx on this CTA's sched_full, so a CTA-scope wait sees it (as for TMA)."""
    slot = Uint32(k) & Uint32(1)
    barrier.wait(_mbar_ptr(_a(sm, "sched_full", slot, 8)), (Uint32(k) >> 1) & Uint32(1), spin=True)
    return Int32(_lds_u32_ordered(_a(sm, "sched", slot, 4)))


@cute.jit
def _release_unit(sm, k, p, rz, pred=None):
    """hand back slot k on rank 0's sched_empty (Rule 11: the arrive's address is computed from the loaded index, through
    the kernel's release-zero). Relaxed: a cluster-scope release would wait for the softmax warps' outstanding dQ stores.
    ``pred``: the arriving lane of a warp whose lanes all loaded the index; None where one lane loads and releases."""
    consume_then_arrive_on_peer(_mbar_ptr(_a(sm, "sched_empty", Uint32(k) & Uint32(1), 8)), 0, p, rz, pred=pred, release_cta=False)


@cute.jit
def _unit_head(g, p):
    pu = Uint32(p)
    if cutlass.const_expr(g.causal):
        return pu % Uint32(g.nhq)
    return pu // Uint32(g.p_uph)


@cute.jit
def _unit_pair(g, p):
    pu = Uint32(p)
    if cutlass.const_expr(g.causal):
        return Uint32(g.p_uph - 1) - pu // Uint32(g.nhq)
    return pu % Uint32(g.p_uph)


@cute.jit
def _unit_nkt(g, p):
    if cutlass.const_expr(g.causal):
        return Int32(_umin(_unit_pair(g, p) * 2 + 2, g.nb))
    return Int32(g.p_nkt)


@cute.jit
def _tma_rows(g, dst, m, x, h, s0, mbar):
    """a 128-row tile (rows s0.. of head h): 2D maps over H*S rows in aligned builds, 3D maps clipped at S in tail
    builds (rows >= S arrive as zeros; the full box still counts toward complete_tx)"""
    if cutlass.const_expr(g.tail):
        _tma_3d(dst, m, x, s0, h, mbar)
    else:
        _tma_2d(dst, m, x, h * Uint32(g.seq) + s0, mbar)


@cute.jit
def _unit_s0(g, pu, crank):
    """unit pu: rows s0.. of query head unit_head(pu), s0 = (2 pair + rank) * 128 (tail builds: possibly >= S)"""
    return (_unit_pair(g, pu) * 2 + crank) * Uint32(128)


@cute.jit
def _publish(sm, k, p):
    slot = Uint32(k) & Uint32(1)
    _mbar_wait_if(Uint32(k >= 2), _a(sm, "sched_empty", slot, 8), ((Uint32(k) >> 1) - Uint32(1)) & Uint32(1))
    bar, dst = _a(sm, "sched_full", slot, 8), _a(sm, "sched", slot, 4)
    _mbar_expect_tx(bar, 4)
    prims.mbarrier_arrive_expect_tx(prims.mapa(_mbar_ptr(bar), 1), 4, scope=prims.MemScope.CLUSTER, relaxed=True)
    cute.arch.store_async_dsmem(_word_ptr(dst), Uint32(p), _mbar_ptr(bar), 0)
    cute.arch.store_async_dsmem(_word_ptr(dst), Uint32(p), _mbar_ptr(bar), 1)


@cute.jit
def _store_dq(g, sm, dq_map, crank, ku, pu):
    """the staging buffer also carries each unit's bf16 dQ tile out (TMA store issued by the unit warp, off the
    softmax path): unit k-2's tile is written by the softmax epilogue after unit k-1 was copied in, and stored before
    unit k is staged"""
    barrier.wait(_mbar_ptr(_a(sm, "dq_staged")), Uint32(ku) & Uint32(1), spin=True)
    cute.arch.fence_proxy("async.shared", space="cta")
    h, s0 = _unit_head(g, pu), _unit_s0(g, pu, crank)
    _tma_store_rows(g, dq_map, 0, h, s0, _a(sm, "stq"))
    _tma_store_rows(g, dq_map, 64, h, s0, _a(sm, "stq", 1, _G.B_TILE))
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _fire_sdp(sm, tmu, k_lo):
    """S = Q K^T and dP = dO V^T of one 128-key tile (4 K-steps each), then s_full"""
    til = _G.B_TILE >> 4
    for ks in cutlass.range_constexpr(4):
        _tcw_mma(
            tmu + _G.TB_S,
            tmu + (_G.TB_Q + ks * 8),
            _dsc(k_lo + Uint32(2 * ks)),
            _G.idesc_mx(128, _G.B_BK, 0, ks, ks),
            tmu + _G.TB_SFQ,
            tmu + _G.TB_SFK,
            int(ks > 0),
        )
    for ks in cutlass.range_constexpr(4):
        _tcw_mma(
            tmu + _G.TB_DP,
            tmu + (_G.TB_DO + ks * 8),
            _dsc(k_lo + Uint32(til + 2 * ks)),
            _G.idesc_mx(128, _G.B_BK, 0, ks, ks),
            tmu + _G.TB_SFDO,
            tmu + _G.TB_SFV,
            int(ks > 0),
        )
    _tcw_commit(_a(sm, "s_full"))


@cute.jit
def _copy_unit(sm, k, cq, r, trow, is_l0, rz):
    """staged unit -> this warp's 16 words of Q (cq 0, 1) or dO (cq 2, 3) into TMEM; returns the row's stats / delta"""
    barrier.wait(_mbar_ptr(_a(sm, "st_full")), Uint32(k) & Uint32(1), spin=True)
    rowb = _a(sm, "stq") + (cq >> 1) * Uint32(_G.B_TILE) + (r >> 3) * Uint32(1024) + (r & Uint32(7)) * Uint32(128)
    w = []
    for i in cutlass.range_constexpr(4):
        c = (cq & Uint32(1)) * Uint32(4) + Uint32(i)  # 16-byte chunk of the 128-byte row
        w += _lds128(rowb + ((c ^ (r & Uint32(7))) << 4))
    _tm_st16(trow + Uint32(_G.TB_Q) + (cq >> 1) * Uint32(_G.TB_DO - _G.TB_Q) + (cq & Uint32(1)) * Uint32(16), w)
    st_v = _lds_f32(_a(sm, "strow") + r * Uint32(4))
    dl_v = _lds_f32(_a(sm, "strow") + Uint32(512) + r * Uint32(4))
    _tm_wait_st()
    _tc_fence_before()
    cute.arch.sync_warp()
    _mbar_arrive_if(is_l0, _a(sm, "q_ready"))
    # Rule 11: the staging buffer is refilled (TMA / bulk) after st_empty. Every lane arrives with an address computed from
    # every word it loaded from the stage, so each lane's own reads are released before the refill (an elected arrive
    # after sync_warp leaves lanes 1-31 without a release of their own: a Compute Sanitizer racecheck WAR hazard).
    consume_then_arrive(_mbar_ptr(_a(sm, "st_empty")), (w, st_v, dl_v), rz)
    return st_v, dl_v


@cute.kernel
def dq_kernel(
    tm_k: GridConstant[tmap.TensorMap],
    tm_v: GridConstant[tmap.TensorMap],
    tm_kt: GridConstant[tmap.TensorMap],
    tm_q: GridConstant[tmap.TensorMap],
    tm_do: GridConstant[tmap.TensorMap],
    tm_dq: GridConstant[tmap.TensorMap],
    sfq: Int64,
    sfdo: Int64,
    sfk: Int64,
    sfv: Int64,
    sfkt: Int64,
    stats: Int64,  # tail builds: the prep kernel's padded LSE copy (+inf at rows >= S)
    delta: Int64,
    counter: Int64,
    amax_dq: Int64,
    g: cutlass.Constexpr,
    rz_arg: Int32,  # the release-zero (Rule 11): the host passes the literal 0; its only use is release_zero()
):
    NHQ, GRP, NB, P_UPH, P_NUNITS, P_NKT, SEQ, S_ALLOC = g.nhq, g.grp, g.nb, g.p_uph, g.p_nunits, g.p_nkt, g.seq, g.s_alloc
    TAIL, CAUSAL, AMAX = g.tail, g.causal, g.amax
    P_KV, P_KT, B_TILE = _G.P_KV, _G.P_KT, _G.B_TILE

    rz = release_zero(rz_arg)
    sm = Uint32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())  # SmemP: 1024-aligned

    k_map, v_map, kt_map = tm_k.get_ptr().toint(), tm_v.get_ptr().toint(), tm_kt.get_ptr().toint()
    q_map, do_map, dq_map = tm_q.get_ptr().toint(), tm_do.get_ptr().toint(), tm_dq.get_ptr().toint()

    tidx = Uint32(cute.arch.thread_idx()[0])
    warp = cute.arch.shuffle_sync(tidx >> 5, 0)
    lane = tidx & Uint32(31)
    crank = Uint32(cute.arch.block_idx_in_cluster())
    if tidx == 0:
        for s in cutlass.range_constexpr(P_KV):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kv_full", s, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kv_empty", s, 8)), 2)  # released by the MMA warps of both CTAs
        for s in cutlass.range_constexpr(P_KT):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kt_full", s, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "kt_empty", s, 8)), 2)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "s_full")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "s_empty")), _G.NSMX_B * 32)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "p_full")), _G.NSMX_B * 32)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "p_empty")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "done")), 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "dq_free")), _G.NSMX_B)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "st_full")), 1)
        # every lane of the softmax warps (Q / dO / rows read) + the MMA warp's sf copy commit
        prims.mbarrier_init(_mbar_ptr(_a(sm, "st_empty")), _G.NSMX_B * 32 + 1)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "q_ready")), _G.NSMX_B)
        prims.mbarrier_init(_mbar_ptr(_a(sm, "dq_staged")), _G.NSMX_B)  # softmax warps: unit's bf16 dQ tile written into the staging buffer
        for i in cutlass.range_constexpr(2):
            prims.mbarrier_init(_mbar_ptr(_a(sm, "sched_full", i, 8)), 1)
            prims.mbarrier_init(_mbar_ptr(_a(sm, "sched_empty", i, 8)), _G.P_READERS)
        cute.arch.mbarrier_init_fence()
    if warp == 1:
        tmem_alloc(_i32_ptr(_a(sm, "tmem_base")), 512, prims.CTAGroup.CTA_1)
    _tc_fence_before()
    cute.arch.sync_threads()
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()
    _tc_fence_after()
    if _lds_u32(_a(sm, "tmem_base")) != 0:
        _trap()

    if warp == 2:
        # ------------------------------------------------------------ unit warp: scheduler (rank 0) + Q/dO staging
        if lane == 0:
            p = Int32(-1)
            pn = Int32(-1)
            if crank == 0:
                p = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
                p = Int32(_select_u32(Uint32(p >= P_NUNITS), Uint32(0xFFFFFFFF), Uint32(p)))
                _publish(sm, 0, p)
                if p >= 0:
                    pn = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
            else:
                p = _read_unit(sm, 0)
                _release_unit(sm, 0, p, rz)
            pm2 = Int32(-1)  # unit indices k-2, k-1
            pm1 = Int32(-1)
            k = Int32(0)
            while p >= 0:
                _mbar_wait_if(Uint32(k > 0), _a(sm, "st_empty"), Uint32(k - 1) & Uint32(1))  # unit k-1 copied out of staging
                if k >= 2:
                    _store_dq(g, sm, dq_map, crank, k - 2, pm2)
                qh, qs0 = _unit_head(g, p), _unit_s0(g, p, crank)
                if cutlass.const_expr(TAIL):  # an empty block's scale atom: any valid one
                    qsf = _select_u32(Uint32(qs0 >= SEQ), Uint32((NB - 1) * 128), qs0)
                else:
                    qsf = qs0
                fb = _a(sm, "st_full")
                _mbar_expect_tx(fb, 2 * B_TILE + 2 * 512 + 2 * 512)
                _tma_rows(g, _a(sm, "stq"), q_map, 0, qh, qs0, fb)
                _tma_rows(g, _a(sm, "stq", 1, B_TILE), do_map, 0, qh, qs0, fb)
                _bulk_g2s(_a(sm, "stsf"), sfq + _atom_of(g, qh, qsf) * 512, 512, fb)
                _bulk_g2s(_a(sm, "stsf", 1, 512), sfdo + _atom_of(g, qh, qsf) * 512, 512, fb)
                _bulk_g2s(_a(sm, "strow"), stats + _wsrow(g, qh, qs0) * 4, 512, fb)
                _bulk_g2s(_a(sm, "strow", 1, 512), delta + _wsrow(g, qh, qs0) * 4, 512, fb)
                pm2 = pm1
                pm1 = p
                if crank == 0:
                    pk = Int32(_select_u32(Uint32(pn >= P_NUNITS) | Uint32(pn < 0), Uint32(0xFFFFFFFF), Uint32(pn)))
                    _publish(sm, k + 1, pk)
                    p = pk
                    if p >= 0:
                        pn = Int32(cute.arch.atomic_add(_gword_ptr(counter), Uint32(1)))
                else:
                    p = _read_unit(sm, k + 1)
                    _release_unit(sm, k + 1, p, rz)
                k = k + 1
            # k units in all: the last two dQ tiles are still to be stored
            if k >= 2:
                _store_dq(g, sm, dq_map, crank, k - 2, pm2)
            if k >= 1:
                _store_dq(g, sm, dq_map, crank, k - 1, pm1)
            cute.arch.cp_async_bulk_wait_group(0, read=False)
    elif warp == 0:
        # ------------------------------------------------------------ K / V / Kt ring producer (both CTAs, multicast)
        if lane == 0:
            t = Uint32(0)
            k = Int32(0)
            p = _read_unit(sm, 0)
            _release_unit(sm, 0, p, rz)
            while p >= 0:
                kvh = _unit_head(g, p) // Uint32(GRP)
                nkt = _unit_nkt(g, p)
                for j in cutlass.range(nkt):
                    k0 = Uint32(j) * Uint32(_G.B_BK)
                    ra = _atom_of(g, kvh, k0) * 512
                    s = t % Uint32(P_KV)  # K, V (+ scales): multicast by rank 0
                    _mbar_wait_if(Uint32(t >= P_KV), _a(sm, "kv_empty", s, 8), ((t // Uint32(P_KV)) - Uint32(1)) & Uint32(1))
                    fb = _a(sm, "kv_full", s, 8)
                    _mbar_expect_tx(fb, 2 * B_TILE + 2 * 512)
                    if crank == 0:
                        _tma_rows_mc(g, _a(sm, "kv", s, 2 * B_TILE), k_map, 0, kvh, k0, fb, 3)
                        _tma_rows_mc(g, _a(sm, "kv", s, 2 * B_TILE) + Uint32(B_TILE), v_map, 0, kvh, k0, fb, 3)
                        _bulk_g2s_mc(_a(sm, "sfkv", s, 1024), sfk + ra, 512, fb, 3)
                        _bulk_g2s_mc(_a(sm, "sfkv", s, 1024) + Uint32(512), sfv + ra, 512, fb, 3)
                    s2 = t % Uint32(P_KT)  # Kt (+ scales): multicast by rank 1
                    _mbar_wait_if(Uint32(t >= P_KT), _a(sm, "kt_empty", s2, 8), ((t // Uint32(P_KT)) - Uint32(1)) & Uint32(1))
                    fb2 = _a(sm, "kt_full", s2, 8)
                    _mbar_expect_tx(fb2, B_TILE + 512)
                    if crank == 1:
                        _tma_rows_mc(g, _a(sm, "kt", s2, B_TILE), kt_map, 0, kvh, k0, fb2, 3)
                        _bulk_g2s_mc(_a(sm, "sfkt", s2, 512), sfkt + ra, 512, fb2, 3)
                    t = t + 1
                k = k + 1
                p = _read_unit(sm, k)
                _release_unit(sm, k, p, rz)
    elif warp == 1:
        # ------------------------------------------------------------ MMA issuer (converged warp, elected issue)
        _tcw_elect()
        tmu = cute.arch.shuffle_sync(_lds_u32(_a(sm, "tmem_base")), 0)  # runtime uniform (== 0)
        kv_lo = (_a(sm, "kv") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
        kt_lo0 = (_a(sm, "kt") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
        sfkv0, sfkt0, stsf0 = _a(sm, "sfkv"), _a(sm, "sfkt"), _a(sm, "stsf")
        KVS, TIL, SFKVS = (2 * B_TILE) >> 4, B_TILE >> 4, 1024

        p = _read_unit(sm, 0)
        _release_unit(sm, 0, p, rz, pred=(lane == Uint32(0)))
        pn = Int32(-1)
        if p >= 0:
            barrier.wait(_mbar_ptr(_a(sm, "kv_full")), 0, spin=True)
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TB_SFK, sfkv0)
            _tcw_cp_sf(tmu + _G.TB_SFV, sfkv0 + Uint32(512))
            barrier.wait(_mbar_ptr(_a(sm, "q_ready")), 0, spin=True)  # Q/dO of unit 0 in TMEM; staged scale atoms valid
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TB_SFQ, stsf0)
            _tcw_cp_sf(tmu + _G.TB_SFDO, stsf0 + Uint32(512))
            _tcw_commit(_a(sm, "st_empty"))
            _fire_sdp(sm, tmu, kv_lo)
            _tcw_commit_mc(_a(sm, "kv_empty"), 3)
            pn = _read_unit(sm, 1)
            _release_unit(sm, 1, pn, rz, pred=(lane == Uint32(0)))
        t = Uint32(0)
        k = Int32(0)
        while p >= 0:
            nkt = _unit_nkt(g, p)
            for j in cutlass.range(nkt):
                u = t % Uint32(P_KT)
                last = Uint32(j + 1 == nkt)
                if ((last ^ Uint32(1)) | Uint32(pn >= 0)) != 0:
                    t1 = t + Uint32(1)
                    s1 = t1 % Uint32(P_KV)
                    barrier.wait(_mbar_ptr(_a(sm, "kv_full", s1, 8)), (t1 // Uint32(P_KV)) & Uint32(1), spin=True)
                    _tc_fence_after()
                    _tcw_cp_sf(tmu + _G.TB_SFK, sfkv0 + s1 * Uint32(SFKVS))  # in-order behind S/dP(t)
                    _tcw_cp_sf(tmu + _G.TB_SFV, sfkv0 + s1 * Uint32(SFKVS) + Uint32(512))
                    barrier.wait(_mbar_ptr(_a(sm, "s_empty")), t & Uint32(1), spin=True)
                    _tc_fence_after()
                    if last != 0:
                        barrier.wait(_mbar_ptr(_a(sm, "q_ready")), Uint32(k + 1) & Uint32(1), spin=True)  # next unit's Q/dO in TMEM
                        _tc_fence_after()
                        _tcw_cp_sf(tmu + _G.TB_SFQ, stsf0)
                        _tcw_cp_sf(tmu + _G.TB_SFDO, stsf0 + Uint32(512))
                        _tcw_commit(_a(sm, "st_empty"))
                    _fire_sdp(sm, tmu, kv_lo + s1 * Uint32(KVS))
                    _tcw_commit_mc(_a(sm, "kv_empty", s1, 8), 3)  # K/V slot free once S/dP(t+1) are done
                barrier.wait(_mbar_ptr(_a(sm, "kt_full", u, 8)), (t // Uint32(P_KT)) & Uint32(1), spin=True)
                _tc_fence_after()
                _tcw_cp_sf(tmu + _G.TB_SFKT, sfkt0 + u * Uint32(512))  # in-order behind dQ(t-1)
                barrier.wait(_mbar_ptr(_a(sm, "p_full")), t & Uint32(1), spin=True)
                _tc_fence_after()
                if (Uint32(j == 0) & Uint32(k > 0)) != 0:
                    barrier.wait(_mbar_ptr(_a(sm, "dq_free")), Uint32(k - 1) & Uint32(1), spin=True)  # the epilogue of unit k-1 has read the accumulator
                    _tc_fence_after()
                kt_lo = kt_lo0 + u * Uint32(TIL)
                for ks in cutlass.range_constexpr(4):
                    _tcw_mma(
                        tmu + _G.TB_DQ,
                        tmu + (_G.TB_DS + ks * 8),
                        _dsc(kt_lo + Uint32(ks * 256)),
                        _G.idesc_mx(128, 128, 1, 1, ks),
                        tmu + (_G.TB_SFDS + 2 * ks),
                        tmu + _G.TB_SFKT,
                        1 if ks > 0 else Uint32(j > 0),
                    )
                _tcw_commit(_a(sm, "p_empty"))
                _tcw_commit_mc(_a(sm, "kt_empty", u, 8), 3)
                if last != 0:
                    _tcw_commit(_a(sm, "done"))
                t = t + Uint32(1)
            p = pn
            if p >= 0:
                pn = _read_unit(sm, k + 2)
                _release_unit(sm, k + 2, pn, rz, pred=(lane == Uint32(0)))
            k = k + 1
    else:
        # ------------------------------------------------------------ softmax / quantization warps
        quad = warp & Uint32(3)
        cq = (warp - Uint32(_G.P_W0)) >> 2  # which 32-key block of the 128-key tile
        r = quad * Uint32(32) + lane  # query row within the block == TMEM lane
        trow = (quad * Uint32(32)) << 16  # TMEM address of this lane quadrant (the allocation starts at column 0)
        c1 = Float32(g.c1)
        tsfds = trow + Uint32(_G.TB_SFDS) + Uint32(2) * cq + quad
        is_l0 = Uint32(lane == 0)

        p = _read_unit(sm, 0)
        _release_unit(sm, 0, p, rz, pred=(lane == Uint32(0)))
        pn = Int32(-1)
        stv = Float32(0.0)
        dlv = Float32(0.0)
        if p >= 0:
            stv, dlv = _copy_unit(sm, 0, cq, r, trow, is_l0, rz)
            pn = _read_unit(sm, 1)
            _release_unit(sm, 1, pn, rz, pred=(lane == Uint32(0)))
        amq = Float32(0.0)  # amax: max |x| of this thread's dQ values before the cast
        t = Uint32(0)
        k = Int32(0)
        while p >= 0:
            # p*scale = 2^(s*C1 + nLb)
            nlb = _neg_ftz(_fma_ftz(stv, Float32(_G.LOG2E), Float32(g.neg_log2_scale)))
            ndl = _neg_ftz(dlv)
            nkt = _unit_nkt(g, p)
            qb = _unit_pair(g, p) * Uint32(2) + crank  # causal: this rank's query block
            for j in cutlass.range(nkt):
                barrier.wait(_mbar_ptr(_a(sm, "s_full")), t & Uint32(1), spin=True)
                _tc_fence_after()
                if (Uint32(j + 1 == nkt) & Uint32(pn >= 0)) != 0:
                    stv, dlv = _copy_unit(sm, k + 1, cq, r, trow, is_l0, rz)  # S/dP(t) landed: this unit's Q/dO are dead
                sw = _tm_ld32(trow + Uint32(_G.TB_S) + cq * Uint32(32))
                dw_in = _tm_ld32(trow + Uint32(_G.TB_DP) + cq * Uint32(32))
                _tm_wait_ld()
                _tc_fence_before()
                _mbar_arrive(_a(sm, "s_empty"))
                if cutlass.const_expr(CAUSAL):
                    # key (j*128 + cq*32 + i) > query (qb*128 + r): masked; tiles past the pair's lower diagonal are
                    # wholly masked
                    lim = (Int32(qb) - Int32(j)) * _G.B_BK + Int32(r) - Int32(cq) * 32  # entries i > lim are masked
                    sw = _mask_above_if(Uint32(lim < 31), sw, lim)
                if cutlass.const_expr(SEQ % 128 != 0):
                    # keys >= S of the last key tile (zero-filled K / V): -inf scores give P = 0 and dS = 0, so they
                    # neither reach dQ nor the MX scale of this 32-key block of dS (as cudnn-frontend's residual mask)
                    valid = Int32(SEQ - (P_NKT - 1) * _G.B_BK) - Int32(cq) * 32  # keys < S in this warp's 32-key block
                    sw = _mask_above_if(Uint32(j == P_NKT - 1), sw, valid - 1)
                sv = [Uint32(x).bitcast(Float32) for x in sw]
                dv = [Uint32(x).bitcast(Float32) for x in dw_in]
                m0 = Float32(0.0)
                m1 = Float32(0.0)
                for b0 in cutlass.range_constexpr(0, 32, 8):
                    score01 = _fma2((sv[b0], sv[b0 + 1]), (c1, c1), (nlb, nlb))
                    score23 = _fma2((sv[b0 + 2], sv[b0 + 3]), (c1, c1), (nlb, nlb))
                    score45 = _fma2((sv[b0 + 4], sv[b0 + 5]), (c1, c1), (nlb, nlb))
                    score67 = _fma2((sv[b0 + 6], sv[b0 + 7]), (c1, c1), (nlb, nlb))
                    prob01 = (cute.arch.exp2(score01[0]), cute.arch.exp2(score01[1]))
                    prob23 = (cute.arch.exp2(score23[0]), cute.arch.exp2(score23[1]))
                    prob45 = (cute.arch.exp2(score45[0]), cute.arch.exp2(score45[1]))
                    prob67 = (cute.arch.exp2(score67[0]), cute.arch.exp2(score67[1]))
                    grad01 = _mul2(prob01, _add2((dv[b0], dv[b0 + 1]), (ndl, ndl)))
                    grad23 = _mul2(prob23, _add2((dv[b0 + 2], dv[b0 + 3]), (ndl, ndl)))
                    grad45 = _mul2(prob45, _add2((dv[b0 + 4], dv[b0 + 5]), (ndl, ndl)))
                    grad67 = _mul2(prob67, _add2((dv[b0 + 6], dv[b0 + 7]), (ndl, ndl)))
                    dv[b0], dv[b0 + 1] = grad01
                    dv[b0 + 2], dv[b0 + 3] = grad23
                    dv[b0 + 4], dv[b0 + 5] = grad45
                    dv[b0 + 6], dv[b0 + 7] = grad67
                    m0 = _max3abs(m0, grad01[0], grad01[1])
                    m1 = _max3abs(m1, grad23[0], grad23[1])
                    m0 = _max3abs(m0, grad45[0], grad45[1])
                    m1 = _max3abs(m1, grad67[0], grad67[1])
                e, rcp = _e8m0_ceil(_max_ftz(m0, m1))
                rc = (rcp, rcp)
                dw = []
                for b0 in cutlass.range_constexpr(0, 32, 8):
                    quant01 = _mul2((dv[b0], dv[b0 + 1]), rc)
                    quant23 = _mul2((dv[b0 + 2], dv[b0 + 3]), rc)
                    quant45 = _mul2((dv[b0 + 4], dv[b0 + 5]), rc)
                    quant67 = _mul2((dv[b0 + 6], dv[b0 + 7]), rc)
                    dw.append(_pack_e4m3x4(quant01[0], quant01[1], quant23[0], quant23[1]))
                    dw.append(_pack_e4m3x4(quant45[0], quant45[1], quant67[0], quant67[1]))
                _mbar_wait_if(Uint32(t > 0), _a(sm, "p_empty"), (t - Uint32(1)) & Uint32(1))
                _tc_fence_after()
                _tm_st8(trow + Uint32(_G.TB_DS) + cq * Uint32(8), dw)
                _tm_st1(tsfds, e)
                _tm_wait_st()
                _tc_fence_before()
                _mbar_arrive(_a(sm, "p_full"))
                t = t + Uint32(1)
            # epilogue of unit k: read the accumulator, release it, convert, and hand the bf16 tile to the unit warp
            # (TMA store) through the staging buffer (free: the next unit was copied out of it at k's last tile)
            barrier.wait(_mbar_ptr(_a(sm, "done")), Uint32(k) & Uint32(1), spin=True)
            _tc_fence_after()
            acc = [Uint32(x).bitcast(Float32) for x in _tm_ld32(trow + Uint32(_G.TB_DQ) + cq * Uint32(32))]
            _tm_wait_ld()
            _tc_fence_before()
            cute.arch.sync_warp()
            _mbar_arrive_if(is_l0, _a(sm, "dq_free"))
            if cutlass.const_expr(AMAX):
                for i in cutlass.range_constexpr(32):
                    amq = _max_ftz(amq, _abs_ftz(acc[i]))
            pk = [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=cutlass.BFloat16) for i in range(16)]
            hb = _a(sm, "stq") + (cq >> 1) * Uint32(B_TILE) + r * Uint32(128)
            for i in cutlass.range_constexpr(4):
                c = (cq & Uint32(1)) * Uint32(4) + Uint32(i)
                _sts128(hb + ((c ^ (r & Uint32(7))) << 4), pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3])
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            _mbar_arrive_if(is_l0, _a(sm, "dq_staged"))
            p = pn
            if p >= 0:
                pn = _read_unit(sm, k + 2)
                _release_unit(sm, k + 2, pn, rz, pred=(lane == Uint32(0)))
            k = k + 1
        if cutlass.const_expr(AMAX):
            m = amq
            for off in cutlass.range_constexpr(5):
                m = _max_ftz(m, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=m, offset=16 >> off, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
            if (is_l0 & Uint32(m > Float32(0.0))) != 0:
                cute.arch.atomic_max(_gword_ptr(amax_dq), Uint32(Float32(m).bitcast(Uint32)))
    _tc_fence_before()
    cute.arch.sync_threads()
    if warp == 1:
        cute.arch.sync_warp()
        _tc_fence_after()
        prims.tcgen05_dealloc(prims.make_tmem_ptr(Int32(0), cutlass.Int8), 512, group=prims.CTAGroup.CTA_1)
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()  # the peer may still signal our barriers until it is done
    # chain completion: this grid must not complete before its PDL primary (dK/dV)
    if tidx == 0:
        cute.arch.griddepcontrol_wait()


dq_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
