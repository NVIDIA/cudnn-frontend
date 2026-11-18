# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dK / dV kernel of the SM100 / SM103 D128 MXFP8 backward chain (KV-stationary query walk)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, GridConstant, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl import barrier
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16
from cudnn.frost.tile_dsl.tmem import tmem_alloc
from cudnn.sdpa.bwd.kernels.sm100.d128_mxfp8.common import (
    _G,
    _abs_ftz,
    _add_ftz,
    _atom_of,
    _bulk_g2s,
    _bulk_g2s_mc,
    _dsc,
    _e8m0_ceil,
    _ex2_unless_below,
    _fma2,
    _gword_ptr,
    _i32_ptr,
    _ldg_nc_v4_plain,
    _ldg_nc_v4_plain_if,
    _lds128,
    _lds_u32,
    _lds_v4f32,
    _max3abs,
    _max_ftz,
    _mbar_arrive,
    _mbar_arrive_if,
    _mbar_expect_tx,
    _mbar_ptr,
    _mbar_wait_if,
    _mul2,
    _pack_e4m3x4,
    _select_u32,
    _select_u64,
    _smem_ptr,
    _sts128,
    _tc_fence_after,
    _tc_fence_before,
    _tcw_commit,
    _tcw_commit_mc,
    _tcw_cp_sf,
    _tcw_elect,
    _tcw_mma,
    _tcw_mma_ss,
    _tm_ld32,
    _tm_ld32_f,
    _tm_st1,
    _tm_st32,
    _tm_st8,
    _tm_wait_ld,
    _tm_wait_st,
    _tma_rows_mc,
    _tma_store_rows,
    _trap,
    _wsrow,
)

# ================================================================ dK / dV kernel (KV-stationary)
# Each CTA owns 128 keys of one KV head and sweeps every query head of its group in 128-query tiles, so every MMA is
# 128x128. S^T and dP^T are produced in turn into one shared TMEM region. P^T (TMEM) feeds dV and dS^T (TMEM) feeds dK.
# 16 softmax warps: 4 per TMEM lane quadrant, each owning one 32-query MX block. The softmax loop is rotated: P(j) is
# handed to dV right after dP^T(j) is pulled, and the requantization of dS(j) is stored before S^T(j+1) is pulled.
#   MMA order per tile: S^T(j+1), dV(j), dK(j), dP^T(j+1)
_OA = _G.SMEM_A_OFF


@cute.jit
def _aa(sm, name, i=0, stride=0):
    """shared address of SmemA field ``name`` (element i of ``stride`` bytes)"""
    if cutlass.const_expr(isinstance(i, int)):
        return sm + Uint32(_OA[name] + i * stride)
    return sm + Uint32(_OA[name]) + i * Uint32(stride)


@cute.jit
def _exp_tile(g, sm, v, stage, cb, c1, causal_thr):
    """P = 2^(s*C1 + nlse) for this thread's 32 queries of one key row (P*256: nlse carries the +8): (fp32 values,
    packed e4m3 words). Causal: entries i < causal_thr are masked (P = 0)."""
    nl = _aa(sm, "lse", stage, _G.A_BQ * 4) + cb * Uint32(128)
    p, wp = [], []
    for i in cutlass.range_constexpr(0, 32, 4):
        L = _lds_v4f32(nl + Uint32(4 * i))
        t01 = _fma2((v[i], v[i + 1]), (c1, c1), (L[0], L[1]))
        t23 = _fma2((v[i + 2], v[i + 3]), (c1, c1), (L[2], L[3]))
        if cutlass.const_expr(g.causal):  # query (qb*128 + cb*32 + i) < key (k0 + r): masked, so dV, dS and the dS block scale see nothing
            ts = (t01[0], t01[1], t23[0], t23[1])
            pv = [_ex2_unless_below(ts[u], causal_thr, i + u) for u in range(4)]  # a comprehension: trace-time Python, not preprocessed
        else:
            pv = [cute.arch.exp2(t01[0]), cute.arch.exp2(t01[1]), cute.arch.exp2(t23[0]), cute.arch.exp2(t23[1])]
        p += pv
        wp.append(_pack_e4m3x4(pv[0], pv[1], pv[2], pv[3]))
    return p, wp


@cute.jit
def _stage_out_words(acc):
    """32 fp32 values -> 16 bf16x2 words"""
    return [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=cutlass.BFloat16) for i in range(16)]  # a comprehension: trace-time Python, not preprocessed


@cute.jit
def _sts_row(ra, r, c, pk):
    """this thread's 64-byte half row (block c) of a 128B-swizzled bf16 staging tile"""
    for ii in cutlass.range_constexpr(4):
        ch = (c & Uint32(1)) * Uint32(4) + Uint32(ii)
        _sts128(ra + ((ch ^ (r & Uint32(7))) << 4), pk[4 * ii], pk[4 * ii + 1], pk[4 * ii + 2], pk[4 * ii + 3])


@cute.jit
def _amax32(m, acc):
    for i in cutlass.range_constexpr(32):
        m = _max_ftz(m, _abs_ftz(acc[i]))
    return m


@cute.kernel
def dkdv_kernel(
    tm_q: GridConstant[tmap.TensorMap],
    tm_do: GridConstant[tmap.TensorMap],
    tm_dot: GridConstant[tmap.TensorMap],
    tm_qt: GridConstant[tmap.TensorMap],
    tm_dk: GridConstant[tmap.TensorMap],
    tm_dv: GridConstant[tmap.TensorMap],
    kg: Int64,
    vg: Int64,
    sfq: Int64,
    sfdo: Int64,
    sfdot: Int64,
    sfqt: Int64,
    sfk: Int64,
    sfv: Int64,
    nlse: Int64,
    ndelta: Int64,
    amax_dk: Int64,
    amax_dv: Int64,
    g: cutlass.Constexpr,
    ks: cutlass.Constexpr[int],  # walk split (1 or 2)
    cl: cutlass.Constexpr[int],  # cluster size
):
    """KS == 2: cluster c owns key blocks 2c and 2c+1 (rank & 1); rank >> 1 selects the half of the query walk. Walk
    group = ranks {0,1} (half 0) or {2,3} (half 1): each group multicasts its own Q / dO tiles within itself, and the
    two halves exchange fp32 partials through DSMEM (half 0 finalizes dV, half 1 dK). KS == 1: the CL CTAs of a cluster
    own CL consecutive key blocks of one KV head and multicast across the cluster. CTA b runs tasks b, b + gridDim.x, ..."""
    NHKV, GRP, NB, NBT, SEQ = g.nhkv, g.grp, g.nb, g.nbt, g.seq
    TAIL, CAUSAL, AMAX = g.tail, g.causal, g.amax
    A_NST, A_TILE, A_BQ = _G.A_NST, _G.A_TILE, _G.A_BQ
    A_NIT = g.a_nit
    NW0, NW1 = (A_NIT + ks - 1) // ks, A_NIT // ks  # walk split: half 0 takes the extra tile when GRP * NB is odd
    NTASK = NHKV * NBT * ks
    WGROUP = 2 if ks == 2 else cl
    NSMXT = _G.A_NSMX * 32

    sm = Uint32(cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024).toint())  # SmemA: 1024-aligned

    tidx = Uint32(cute.arch.thread_idx()[0])
    warp = cute.arch.shuffle_sync(tidx >> 5, 0)
    lane = tidx & Uint32(31)
    if warp == 1:
        tmem_alloc(_i32_ptr(_aa(sm, "tmem_base")), 512, prims.CTAGroup.CTA_1)
    _tc_fence_before()
    cute.arch.sync_threads()
    # all 512 columns are allocated, so the allocation necessarily starts at TMEM address 0
    if _lds_u32(_aa(sm, "tmem_base")) != 0:
        _trap()
    crank = Uint32(cute.arch.block_idx_in_cluster())
    whalf = (crank >> 1) if ks == 2 else Uint32(0)
    if cutlass.const_expr(NW0 == NW1):
        nw_plain = Uint32(NW0)
    else:
        nw_plain = _select_u32(whalf, Uint32(NW1), Uint32(NW0))
    if cutlass.const_expr(ks == 2):
        wmask = _select_u32(whalf, Uint32(0xC), Uint32(0x3))
        wrole = crank & Uint32(1)  # walk split: 0 loads half a (q, dO), 1 half b (the rest)
    else:
        wmask = Uint32((1 << cl) - 1)
        wrole = crank
    j0 = whalf * Uint32(NW0)
    amx = Float32(0.0)  # amax: max |x| of this thread's final dV (cb 0, 1) or dK (cb 2, 3) values before the cast
    bidx = Uint32(cute.arch.block_idx()[0])
    gdim = Uint32(cute.arch.grid_dim()[0])
    for task_i in cutlass.range(Int32(bidx), Int32(NTASK), Int32(gdim)):
        task = Uint32(task_i)
        if cutlass.const_expr(ks == 2):
            ctile = (task >> 2) * Uint32(2) + (crank & Uint32(1))
        else:
            ctile = task
        if cutlass.const_expr(CAUSAL):
            # Causal walks shrink with the key block (a group starting at block b walks NB - b query blocks per head).
            # Deal the cluster tasks longest first in snake order: cluster k takes position k in even waves and
            # nc-1-k in odd full waves, so long and short walks pair up across waves (static and deterministic; a
            # partial last wave is not reversed).
            GS = 2 if ks == 2 else cl  # key blocks per cluster task (walk group)
            NG = NHKV * (NBT // GS)  # cluster tasks
            nc = gdim // Uint32(cl)  # clusters
            tc = task // Uint32(cl)
            wave = tc // nc
            kk = tc - wave * nc
            flip = (wave & Uint32(1)) & Uint32((wave + Uint32(1)) * nc <= Uint32(NG))
            pos = wave * nc + _select_u32(flip, nc - Uint32(1) - kk, kk)
            c_ = pos // Uint32(NHKV)
            gh = pos - c_ * Uint32(NHKV)  # longest first: group c of every KV head, then c + 1
            ctile = gh * Uint32(NBT) + c_ * Uint32(GS) + ((crank & Uint32(1)) if ks == 2 else crank)
        gkv = ctile // Uint32(NBT)
        k0 = (ctile - gkv * Uint32(NBT)) * Uint32(128)  # tail builds: a block with k0 >= S is empty (zero K / V, nothing stored)
        if cutlass.const_expr(TAIL):
            ksf = _select_u32(Uint32(k0 >= SEQ), Uint32((NB - 1) * 128), k0)  # its scale atom: any valid one
        else:
            ksf = k0
        # Causal: query blocks below a key block's diagonal see none of its keys. The CTAs of a walk group multicast
        # the same tiles, so the group walks query blocks kbw..NB-1 of each head (kbw = its first key block); a CTA whose
        # own block is later masks the tiles above its diagonal. Non-causal: kbw = 0, wl = NB (the constant walk).
        if cutlass.const_expr(CAUSAL):
            kfirst = (ctile & Uint32(0xFFFFFFFE)) if ks == 2 else ctile - crank
            kbw = kfirst - (kfirst // Uint32(NBT)) * Uint32(NBT)
            wl = Uint32(NB) - kbw
            anit = Uint32(GRP) * wl
            if cutlass.const_expr(ks == 2):
                nw = _select_u32(whalf, anit >> 1, anit - (anit >> 1))
                jb = whalf * (anit - (anit >> 1))
            else:
                nw = anit
                jb = Uint32(0)
        else:
            kbw = Uint32(0)
            wl = Uint32(NB)
            nw = nw_plain
            jb = j0

        if tidx == 0:
            if task != bidx:
                for st in cutlass.range_constexpr(A_NST):
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "full_a", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "full_b", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "empty_a", st, 8)))
                    prims.mbarrier_inval(_mbar_ptr(_aa(sm, "empty_b", st, 8)))
                for nm in cutlass.range_constexpr(10):
                    prims.mbarrier_inval(
                        _mbar_ptr(
                            _aa(sm, ("sd_full", "s_pulled", "dp_pulled", "ds_full", "ds_empty", "kv_ready", "sfkv_full", "done", "rx_ready", "rx_full")[nm])
                        )
                    )
            for st in cutlass.range_constexpr(A_NST):
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "full_a", st, 8)), 1)
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "full_b", st, 8)), 1)
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "empty_a", st, 8)), WGROUP)  # released by the MMA warps of all CTAs of the walk group
                prims.mbarrier_init(_mbar_ptr(_aa(sm, "empty_b", st, 8)), WGROUP)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "sd_full")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "s_pulled")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "dp_pulled")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "ds_full")), NSMXT)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "ds_empty")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "kv_ready")), 8)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "sfkv_full")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "done")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "rx_ready")), 1)
            prims.mbarrier_init(_mbar_ptr(_aa(sm, "rx_full")), 1)
            cute.arch.mbarrier_init_fence()
        _tc_fence_before()
        cute.arch.sync_threads()
        prims.barrier_cluster_arrive_aligned()
        prims.barrier_cluster_wait_aligned()  # peer barriers initialised before any multicast lands
        _tc_fence_after()
        if task == bidx:
            # PDL dependent of prep: the prologue above overlapped prep's tail; prep's rows are visible after the wait,
            # and only then may dQ (which reads prep's delta and unit counter) launch
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        if warp == 0:
            # ------------------------------------------------------------ TMA producer
            if prims.elect_sync():
                # tensor-map addresses formed where they are used (kept out of the softmax warps' register budget)
                q_map, do_map, dot_map, qt_map = tm_q.get_ptr().toint(), tm_do.get_ptr().toint(), tm_dot.get_ptr().toint(), tm_qt.get_ptr().toint()
                kvb = _aa(sm, "sfkv_full")
                _mbar_expect_tx(kvb, 1024)
                _bulk_g2s(_aa(sm, "sfkv"), sfk + _atom_of(g, gkv, ksf) * 512, 512, kvb)
                _bulk_g2s(_aa(sm, "sfkv", 1, 512), sfv + _atom_of(g, gkv, ksf) * 512, 512, kvb)
                # Each stage is split in two halves with their own full/empty barriers: half a (q, dO, scales) is
                # multicast by rank 0 and recycled as soon as dP of its tile is done; half b (dOt, qt, scales, lse,
                # delta) is multicast by rank 1 and recycled after dK. Each CTA also arms halves delivered by other ranks
                # once the previous phase of that barrier has completed (a peer's bytes may land before the arming; the
                # tx-count absorbs that).
                BYTES_A = 2 * A_TILE + 2 * 512
                BYTES_B = 2 * A_TILE + 2 * 512 + 2 * A_BQ * 4
                # tile j = jb + jj walks query block kbw + j % wl of query head gkv * GRP + j / wl; both are advanced
                # incrementally (wl is a run-time divisor in causal builds)
                hcur = gkv * Uint32(GRP) + jb // wl
                qcur = jb - (jb // wl) * wl
                for jj in cutlass.range(nw):
                    jju = Uint32(jj)
                    s = jju % Uint32(A_NST)
                    pph = ((jju // Uint32(A_NST)) - Uint32(1)) & Uint32(1)  # parity of the previous use of stage s
                    hq_ = hcur
                    q0 = (kbw + qcur) * Uint32(A_BQ)
                    wr = _wsrow(g, hq_, q0)  # lse / delta rows (tail builds: rows >= S give P = 0)
                    fa, fbb = _aa(sm, "full_a", s, 8), _aa(sm, "full_b", s, 8)
                    ra = _atom_of(g, hq_, q0) * 512
                    again = Uint32(jju >= A_NST)
                    tiles_s = _aa(sm, "tiles", s, 4 * A_TILE)
                    sf_s = _aa(sm, "sf", s, 4 * 512)
                    if wrole == 0:
                        _mbar_wait_if(again, _aa(sm, "empty_a", s, 8), pph)
                        _mbar_expect_tx(fa, BYTES_A)
                        _tma_rows_mc(g, tiles_s, q_map, 0, hq_, q0, fa, wmask)
                        _tma_rows_mc(g, tiles_s + Uint32(A_TILE), do_map, 0, hq_, q0, fa, wmask)
                        _bulk_g2s_mc(sf_s, sfq + ra, 512, fa, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(512), sfdo + ra, 512, fa, wmask)
                        _mbar_wait_if(again, fbb, pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                    elif wrole == 1:
                        _mbar_wait_if(again, _aa(sm, "empty_b", s, 8), pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                        _tma_rows_mc(g, tiles_s + Uint32(2 * A_TILE), dot_map, 0, hq_, q0, fbb, wmask)
                        _tma_rows_mc(g, tiles_s + Uint32(3 * A_TILE), qt_map, 0, hq_, q0, fbb, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(1024), sfdot + ra, 512, fbb, wmask)
                        _bulk_g2s_mc(sf_s + Uint32(1536), sfqt + ra, 512, fbb, wmask)
                        _bulk_g2s_mc(_aa(sm, "lse", s, A_BQ * 4), nlse + wr * 4, A_BQ * 4, fbb, wmask)
                        _bulk_g2s_mc(_aa(sm, "dlt", s, A_BQ * 4), ndelta + wr * 4, A_BQ * 4, fbb, wmask)
                        _mbar_wait_if(again, fa, pph)
                        _mbar_expect_tx(fa, BYTES_A)
                    else:
                        _mbar_wait_if(again, fa, pph)
                        _mbar_expect_tx(fa, BYTES_A)
                        _mbar_wait_if(again, fbb, pph)
                        _mbar_expect_tx(fbb, BYTES_B)
                    qcur = qcur + Uint32(1)
                    wrapq = Uint32(qcur == wl)
                    qcur = _select_u32(wrapq, Uint32(0), qcur)
                    hcur = hcur + wrapq
        elif warp == 1:
            # ------------------------------------------------------------ MMA issuer (converged warp, elected issue)
            _tcw_elect()
            tmu = cute.arch.shuffle_sync(_lds_u32(_aa(sm, "tmem_base")), 0)  # runtime uniform (== 0)
            barrier.wait(_mbar_ptr(_aa(sm, "kv_ready")), 0, spin=True)
            _tc_fence_after()
            # SW128 operand descriptors are affine in the SMEM address: precompute the low words and keep every
            # scale-factor copy and stage wait ahead of the softmax handoff it does not depend on, so only the MMAs
            # themselves are issued after each handoff (tcgen05.cp/mma from one thread execute in order).
            t_lo = (_aa(sm, "tiles") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
            k_lo = (_aa(sm, "kmat") & Uint32(0x3FFFF)) >> 4 | Uint32(_G.SDESC_LO_FLAGS)
            sf0 = _aa(sm, "sf")
            STG, TIL, SFS = (4 * A_TILE) >> 4, A_TILE >> 4, 4 * 512
            # s_pulled completes once per S pull, dp_pulled once per dP pull (which also hands over P)
            barrier.wait(_mbar_ptr(_aa(sm, "full_a")), 0, spin=True)
            _tc_fence_after()
            _tcw_cp_sf(tmu + _G.TA_SFQ, sf0)
            _tcw_cp_sf(tmu + _G.TA_SFDO, sf0 + Uint32(512))
            for kq in cutlass.range_constexpr(4):  # S^T = K Q^T
                _tcw_mma_ss(
                    tmu + _G.TA_SD,
                    _dsc(k_lo + Uint32(2 * kq)),
                    _dsc(t_lo + Uint32(2 * kq)),
                    _G.idesc_mx(128, A_BQ, 0, kq, kq),
                    tmu + _G.TA_SFK,
                    tmu + _G.TA_SFQ,
                    1 if kq > 0 else 0,
                )
            _tcw_commit(_aa(sm, "sd_full"))
            barrier.wait(_mbar_ptr(_aa(sm, "s_pulled")), 0, spin=True)
            _tc_fence_after()
            for kq in cutlass.range_constexpr(4):  # dP^T = V dO^T
                _tcw_mma(
                    tmu + _G.TA_SD,
                    tmu + (_G.TA_V + kq * 8),
                    _dsc(t_lo + Uint32(TIL + 2 * kq)),
                    _G.idesc_mx(128, A_BQ, 0, kq, kq),
                    tmu + _G.TA_SFV,
                    tmu + _G.TA_SFDO,
                    1 if kq > 0 else 0,
                )
            _tcw_commit(_aa(sm, "sd_full"))
            _tcw_commit_mc(_aa(sm, "empty_a"), wmask)
            s = Uint32(0)  # stages of tiles j and j + 1
            s1 = Uint32(1)
            ph = Uint32(0)  # full_a/full_b parities of tiles j and j + 1
            ph1 = Uint32(0)
            for j in cutlass.range(nw):
                ju = Uint32(j)
                more = Uint32(ju + Uint32(1) < nw)
                lo = t_lo + s * Uint32(STG)
                lo1 = t_lo + s1 * Uint32(STG)
                if more != 0:
                    barrier.wait(_mbar_ptr(_aa(sm, "full_a", s1, 8)), ph1, spin=True)
                    _tc_fence_after()
                    _tcw_cp_sf(tmu + _G.TA_SFQ, sf0 + s1 * Uint32(SFS))  # S(j) is complete: its dP has been issued
                barrier.wait(_mbar_ptr(_aa(sm, "dp_pulled")), ju & Uint32(1), spin=True)  # dP^T(j) pulled out and P(j) in TMEM
                _tc_fence_after()
                if more != 0:
                    for kq in cutlass.range_constexpr(4):
                        _tcw_mma_ss(
                            tmu + _G.TA_SD,
                            _dsc(k_lo + Uint32(2 * kq)),
                            _dsc(lo1 + Uint32(2 * kq)),
                            _G.idesc_mx(128, A_BQ, 0, kq, kq),
                            tmu + _G.TA_SFK,
                            tmu + _G.TA_SFQ,
                            1 if kq > 0 else 0,
                        )
                    _tcw_commit(_aa(sm, "sd_full"))
                barrier.wait(_mbar_ptr(_aa(sm, "full_b", s, 8)), ph, spin=True)
                _tc_fence_after()
                _tcw_cp_sf(tmu + _G.TA_SFDOT, sf0 + s * Uint32(SFS) + Uint32(1024))
                for kq in cutlass.range_constexpr(4):  # dV += P(j)^T-scaled dOt
                    _tcw_mma(
                        tmu + _G.TA_DV,
                        tmu + (_G.TA_P + kq * 8),
                        _dsc(lo + Uint32(2 * TIL + kq * 256)),
                        _G.idesc_mx(128, 128, 1, 0, kq),
                        tmu + _G.TA_SFP,
                        tmu + _G.TA_SFDOT,
                        1 if kq > 0 else Uint32(ju > 0),
                    )
                # dP^T(j+1) goes ahead of dK(j) in the pipe; dK(j) only gates reuse of the dS buffer (ds_empty)
                if more != 0:
                    _tcw_cp_sf(tmu + _G.TA_SFDO, sf0 + s1 * Uint32(SFS) + Uint32(512))  # dP(j) is complete: it was pulled
                    barrier.wait(_mbar_ptr(_aa(sm, "s_pulled")), (ju + Uint32(1)) & Uint32(1), spin=True)
                    _tc_fence_after()
                    for kq in cutlass.range_constexpr(4):
                        _tcw_mma(
                            tmu + _G.TA_SD,
                            tmu + (_G.TA_V + kq * 8),
                            _dsc(lo1 + Uint32(TIL + 2 * kq)),
                            _G.idesc_mx(128, A_BQ, 0, kq, kq),
                            tmu + _G.TA_SFV,
                            tmu + _G.TA_SFDO,
                            1 if kq > 0 else 0,
                        )
                    _tcw_commit(_aa(sm, "sd_full"))
                    _tcw_commit_mc(_aa(sm, "empty_a", s1, 8), wmask)
                # dK += dS(j)^T qt
                _tcw_cp_sf(tmu + _G.TA_SFQT, sf0 + s * Uint32(SFS) + Uint32(1536))
                barrier.wait(_mbar_ptr(_aa(sm, "ds_full")), ju & Uint32(1), spin=True)
                _tc_fence_after()
                for kq in cutlass.range_constexpr(4):
                    _tcw_mma(
                        tmu + _G.TA_DK,
                        tmu + (_G.TA_DS + kq * 8),
                        _dsc(lo + Uint32(3 * TIL + kq * 256)),
                        _G.idesc_mx(128, 128, 1, 1, kq),
                        tmu + (_G.TA_SFDS + 2 * kq),
                        tmu + _G.TA_SFQT,
                        1 if kq > 0 else Uint32(ju > 0),
                    )
                _tcw_commit(_aa(sm, "ds_empty"))
                _tcw_commit_mc(_aa(sm, "empty_b", s, 8), wmask)
                s = s1
                ph = ph1
                s1 = s1 + Uint32(1)
                wrap = Uint32(s1 == A_NST)
                s1 = _select_u32(wrap, Uint32(0), s1)
                ph1 = ph1 ^ wrap
            _tcw_commit(_aa(sm, "done"))
        else:
            # ------------------------------------------------------------ softmax / quantization warps
            quad = warp & Uint32(3)
            cb = (warp - Uint32(2)) >> 2  # which 32-query MX block of the 128-query tile
            r = quad * Uint32(32) + lane  # key row within the 128-key block == TMEM lane
            trow = (quad * Uint32(32)) << 16
            is_l0 = Uint32(lane == 0)
            if cb < 2:
                kp = _select_u64(Uint32(cb == 0), kg, vg) + (Int64(gkv) * SEQ + Int64(k0 + r)) * _G.HD
                w = []
                if cutlass.const_expr(TAIL):
                    krow = Uint32(k0 + r < SEQ)  # tail builds: key rows >= S are zero
                    for i in cutlass.range_constexpr(8):
                        w += _ldg_nc_v4_plain_if(krow, kp + Int64(16 * i))
                else:
                    for i in cutlass.range_constexpr(8):
                        w += _ldg_nc_v4_plain(kp + Int64(16 * i))
                if cb == 0:
                    krow_s = _aa(sm, "kmat") + (r >> 3) * Uint32(1024) + (r & Uint32(7)) * Uint32(128)
                    for i in cutlass.range_constexpr(8):
                        _sts128(krow_s + ((Uint32(i) ^ (r & Uint32(7))) << 4), w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3])
                    cute.arch.fence_proxy("async.shared", space="cta")
                else:
                    _tm_st32(trow + Uint32(_G.TA_V), w)
                barrier.wait(_mbar_ptr(_aa(sm, "sfkv_full")), 0, spin=True)
                _tm_st1(
                    trow + Uint32(_G.TA_SFK) + cb * Uint32(_G.TA_SFV - _G.TA_SFK) + quad,
                    _lds_u32(_aa(sm, "sfkv") + cb * Uint32(512) + (r & Uint32(31)) * Uint32(16) + (r >> 5) * Uint32(4)),
                )
                if cb == 0:
                    _tm_st1(trow + Uint32(_G.TA_SFP) + quad, Uint32(0x77777777))  # 2^-8 for every P block
                _tm_wait_st()
                _tc_fence_before()
                cute.arch.sync_warp()
                _mbar_arrive_if(is_l0, _aa(sm, "kv_ready"))
            tsfds = trow + Uint32(_G.TA_SFDS) + Uint32(2) * cb + quad
            c1 = Float32(g.c1)
            c2 = Float32(g.c2)
            tsd = trow + Uint32(_G.TA_SD) + cb * Uint32(32)
            tp = trow + Uint32(_G.TA_P) + cb * Uint32(8)
            tds = trow + Uint32(_G.TA_DS) + cb * Uint32(8)
            if cutlass.const_expr(CAUSAL):
                thr0 = Int32(k0 + r) - Int32(cb) * 32  # entries i < thr are masked, thr = k0 + r - qb*128 - cb*32
            else:
                thr0 = Int32(0)
            # Tile j pulls S(j) -> P(j), then dP(j) -> dS(j): P(j) never crosses an iteration, so it stays in registers.
            phase = Uint32(0)
            qm = jb - (jb // wl) * wl  # causal: (jb + j) % wl of the current tile, advanced per tile
            for base_i in cutlass.range(Int32(0), Int32(nw), Int32(A_NST), unroll=1):
                for stage in cutlass.range_constexpr(A_NST):
                    j = Uint32(base_i) + Uint32(stage)
                    if j < nw:
                        # ---- pull S^T(j) and compute P(j)
                        barrier.wait(_mbar_ptr(_aa(sm, "sd_full")), 0, spin=True)
                        _tc_fence_after()
                        sv = _tm_ld32_f(tsd)
                        _tm_wait_ld()
                        if cutlass.const_expr(stage == 0):
                            if j == 0:  # the first tile hands S^T back before waiting for its half b
                                _tc_fence_before()
                                _mbar_arrive(_aa(sm, "s_pulled"))
                                barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                            else:
                                barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                                _tc_fence_before()
                                _mbar_arrive(_aa(sm, "s_pulled"))
                        else:
                            barrier.wait(_mbar_ptr(_aa(sm, "full_b", stage, 8)), phase, spin=True)
                            _tc_fence_before()
                            _mbar_arrive(_aa(sm, "s_pulled"))
                        if cutlass.const_expr(CAUSAL):
                            pf, wp = _exp_tile(g, sm, sv, stage, cb, c1, thr0 - Int32(kbw + qm) * 128)
                            qm = qm + Uint32(1)
                            qm = _select_u32(Uint32(qm == wl), Uint32(0), qm)
                        else:
                            pf, wp = _exp_tile(g, sm, sv, stage, cb, c1, thr0)
                        # ---- pull dP^T(j) and hand P(j) to the dV MMA (dP^T(j) done implies dV(j-1) done reading P)
                        barrier.wait(_mbar_ptr(_aa(sm, "sd_full")), 1, spin=True)
                        _tc_fence_after()
                        d = _tm_ld32_f(tsd)
                        _tm_st8(tp, wp)
                        _tm_wait_ld()
                        _tm_wait_st()
                        _tc_fence_before()
                        _mbar_arrive(_aa(sm, "dp_pulled"))
                        # ---- dS*scale = (P*256) * (dP*C2 + ndelta)
                        m0 = Float32(0.0)
                        m1 = Float32(0.0)
                        nd_addr = _aa(sm, "dlt", stage, A_BQ * 4) + cb * Uint32(128)
                        for i in cutlass.range_constexpr(0, 32, 4):
                            Dl = _lds_v4f32(nd_addr + Uint32(4 * i))
                            a01 = _fma2((d[i], d[i + 1]), (c2, c2), (Dl[0], Dl[1]))
                            a23 = _fma2((d[i + 2], d[i + 3]), (c2, c2), (Dl[2], Dl[3]))
                            d01 = _mul2((pf[i], pf[i + 1]), a01)
                            d23 = _mul2((pf[i + 2], pf[i + 3]), a23)
                            d[i], d[i + 1] = d01
                            d[i + 2], d[i + 3] = d23
                            m0 = _max3abs(m0, d01[0], d01[1])
                            m1 = _max3abs(m1, d23[0], d23[1])
                        e, rcp = _e8m0_ceil(_max_ftz(m0, m1))
                        wd = []
                        for i in cutlass.range_constexpr(0, 32, 4):
                            q01 = _mul2((d[i], d[i + 1]), (rcp, rcp))
                            q23 = _mul2((d[i + 2], d[i + 3]), (rcp, rcp))
                            wd.append(_pack_e4m3x4(q01[0], q01[1], q23[0], q23[1]))
                        # ---- hand dS(j) to the dK MMA once dK(j-1) has finished reading the buffer
                        if j > 0:
                            barrier.wait(_mbar_ptr(_aa(sm, "ds_empty")), (j - Uint32(1)) & Uint32(1), spin=True)
                            _tc_fence_after()
                        _tm_st8(tds, wd)
                        _tm_st1(tsfds, e)
                        _tm_wait_st()
                        _tc_fence_before()
                        _mbar_arrive(_aa(sm, "ds_full"))
                phase = phase ^ Uint32(1)

            # ------------------------------------------------------------ epilogue: dV (cb 0,1), dK (cb 2,3) -> bf16
            # Each thread owns one 64-column half row (r, cb & 1) of dV (cb < 2) or dK. The bf16 tile goes through the
            # idle stage buffers (4 regions of 128 rows x 128 B, 128B-swizzled) and leaves by TMA stores.
            barrier.wait(_mbar_ptr(_aa(sm, "done")), 0, spin=True)
            _tc_fence_after()
            stage0 = _aa(sm, "tiles")
            colbase = _select_u32(Uint32(cb < 2), Uint32(_G.TA_DV), Uint32(_G.TA_DK))
            region = _select_u32(Uint32(cb < 2), Uint32(0), Uint32(2)) + (cb & Uint32(1))
            if cutlass.const_expr(TAIL):
                amax_row = Uint32(k0 + r < SEQ)  # key rows >= S are never stored
            else:
                amax_row = Uint32(1)
            c0 = (cb & Uint32(1)) * Uint32(2)
            if cutlass.const_expr(ks == 2):
                # Bidirectional exchange of the fp32 partials within the pair (half 0 = rank r, half 1 = rank r + 2):
                # half 0 finalizes dV, half 1 dK. Each CTA dumps the partial it does not keep ([blk][row][32 floats],
                # 128B-swizzled, offsets 0..64K of its idle stage buffers) and bulk-copies it into the peer's 64K..128K;
                # the kept half adds the received partial (fp32, commutative: identical bits in either CTA) and is
                # staged as bf16 at 128K..160K.
                peer = _select_u32(whalf, crank - Uint32(2), crank + Uint32(2))
                keep = Uint32(Uint32(cb < 2) == Uint32(whalf == 0))  # dV threads keep in half 0, dK threads in half 1
                if keep == 0:
                    for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                        c = Uint32(c_i)
                        acc = _tm_ld32(trow + colbase + c * Uint32(32))
                        _tm_wait_ld()
                        ra = stage0 + c * Uint32(16384) + r * Uint32(128)
                        for i in cutlass.range_constexpr(8):
                            _sts128(ra + ((Uint32(i) ^ (r & Uint32(7))) << 4), acc[4 * i], acc[4 * i + 1], acc[4 * i + 2], acc[4 * i + 3])
                    cute.arch.fence_proxy("async.shared", space="cta")
                # first warp of the kept output (half 0: warp 2 = cb 0, half 1: warp 10 = cb 2) arms the receive; the
                # first warp of the other output ships the dump
                keep_w0 = _select_u32(Uint32(whalf == 0), Uint32(2), Uint32(10))
                send_w0 = _select_u32(Uint32(whalf == 0), Uint32(10), Uint32(2))
                if (Uint32(warp == keep_w0) & is_l0) != 0:
                    _mbar_expect_tx(_aa(sm, "rx_full"), 4 * 16384)
                    prims.mbarrier_arrive(
                        prims.mapa(_mbar_ptr(_aa(sm, "rx_ready")), peer), scope=prims.MemScope.CLUSTER, relaxed=False
                    )  # this CTA's MMAs are done: 64K..128K free
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == send_w0) & is_l0) != 0:
                    barrier.wait_cluster_acquire(_mbar_ptr(_aa(sm, "rx_ready")), 0)
                    dst = prims.mapa(_smem_ptr(stage0 + Uint32(65536)), peer)
                    rbar = prims.mapa(_mbar_ptr(_aa(sm, "rx_full")), peer)
                    for i_i in cutlass.range(Int32(4), unroll=1):
                        iu = Uint32(i_i)
                        prims.cp_async_bulk_shared_cluster_shared_cta(dst.subview(iu * Uint32(16384)), _smem_ptr(stage0 + iu * Uint32(16384)), rbar, 16384)
                if keep != 0:
                    barrier.wait(_mbar_ptr(_aa(sm, "rx_full")), 0, spin=True)
                    out0 = stage0 + Uint32(131072) - _select_u32(Uint32(cb < 2), Uint32(0), Uint32(2 * 16384))  # regions 0,1 at 128K
                    for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                        c = Uint32(c_i)
                        accw = _tm_ld32(trow + colbase + c * Uint32(32))
                        ra = stage0 + Uint32(65536) + c * Uint32(16384) + r * Uint32(128)
                        prw = []
                        for i in cutlass.range_constexpr(8):
                            prw += _lds128(ra + ((Uint32(i) ^ (r & Uint32(7))) << 4))
                        _tm_wait_ld()
                        acc = [_add_ftz(Uint32(accw[i]).bitcast(Float32), Uint32(prw[i]).bitcast(Float32)) for i in range(32)]
                        if cutlass.const_expr(AMAX):
                            if amax_row != 0:
                                amx = _amax32(amx, acc)
                        _sts_row(out0 + region * Uint32(16384) + r * Uint32(128), r, c, _stage_out_words(acc))
                    cute.arch.fence_proxy("async.shared", space="cta")
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == 2) & is_l0) != 0:
                    out0 = stage0 + Uint32(131072)
                    om = _select_u64(Uint32(whalf == 0), tm_dv.get_ptr().toint(), tm_dk.get_ptr().toint())
                    _tma_store_rows(g, om, 0, gkv, k0, out0)
                    _tma_store_rows(g, om, 64, gkv, k0, out0 + Uint32(16384))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            else:
                for c_i in cutlass.range(Int32(c0), Int32(c0) + 2, unroll=1):
                    c = Uint32(c_i)
                    accw = _tm_ld32(trow + colbase + c * Uint32(32))
                    _tm_wait_ld()
                    acc = [Uint32(x).bitcast(Float32) for x in accw]
                    if cutlass.const_expr(AMAX):
                        if amax_row != 0:
                            amx = _amax32(amx, acc)
                    _sts_row(stage0 + region * Uint32(16384) + r * Uint32(128), r, c, _stage_out_words(acc))
                cute.arch.fence_proxy("async.shared", space="cta")
                barrier.named_barrier_sync_unaligned(1, NSMXT)
                if (Uint32(warp == 2) & is_l0) != 0:
                    dv_map, dk_map = tm_dv.get_ptr().toint(), tm_dk.get_ptr().toint()
                    _tma_store_rows(g, dv_map, 0, gkv, k0, stage0)
                    _tma_store_rows(g, dv_map, 64, gkv, k0, stage0 + Uint32(16384))
                    _tma_store_rows(g, dk_map, 0, gkv, k0, stage0 + Uint32(32768))
                    _tma_store_rows(g, dk_map, 64, gkv, k0, stage0 + Uint32(49152))
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)  # the next task's loads may overwrite the stage buffers after the end-of-task barrier
        if cutlass.const_expr(AMAX):
            if warp >= 2:  # softmax warps; max is order-free, so the reduction stays deterministic
                m = amx
                for off in cutlass.range_constexpr(5):
                    m = _max_ftz(m, prims.shfl_sync(thread_mask=0xFFFFFFFF, val=m, offset=16 >> off, mask_and_clamp=0x1F, kind=prims.Shfl.BFLY))
                if (Uint32(lane == 0) & Uint32(m > Float32(0.0))) != 0:
                    cute.arch.atomic_max(_gword_ptr(_select_u64(Uint32(((warp - Uint32(2)) >> 2) < 2), amax_dv, amax_dk)), Uint32(Float32(m).bitcast(Uint32)))
        _tc_fence_before()
        cute.arch.sync_threads()
        prims.barrier_cluster_arrive_aligned()
        prims.barrier_cluster_wait_aligned()
    if warp == 1:
        cute.arch.sync_warp()
        _tc_fence_after()
        prims.tcgen05_dealloc(prims.make_tmem_ptr(Int32(0), cutlass.Int8), 512, group=prims.CTAGroup.CTA_1)
    if (Uint32(warp == 2) & Uint32(lane == 0)) != 0:
        cute.arch.cp_async_bulk_wait_group(0, read=False)  # dK / dV TMA stores complete before the grid does
    prims.barrier_cluster_arrive_aligned()
    prims.barrier_cluster_wait_aligned()


dkdv_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
