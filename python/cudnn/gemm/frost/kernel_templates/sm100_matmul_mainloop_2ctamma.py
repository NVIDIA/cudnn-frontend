# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm100 cta_group=2 GEMM kernel: 2-CTA MMA cluster pair + mainloop fusion.

Extends ``sm100_matmul_2ctamma.py`` with 4 dedicated mainloop warps per CTA
that apply a unary/scalar op chain to the A and/or B operand tile in SMEM
(in place, preserving the TMA swizzle) before the MMA reads it. The compiler
picks this template when the graph has mainloop fusion and
``TileConfig.cta_group == 2``.

Warp layout (12 warps × 32 = 384 threads/CTA, 8 + 4 mainloop):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (leader CTA runs MMA; follower CTA CLC-consumes only)  — setmaxnreg.dec 40
  warp  5   : TMA producer (both CTAs load their slice)  — setmaxnreg.dec 40
  warp  6   : CLC scheduler (leader CTA issues queries; every CTA waits + reads + arrives empty)  — setmaxnreg.dec 40
  warps 8–11: mainloop-fusion warps (transform A in SMEM)  — setmaxnreg.inc 216
  warp  7   : unused donor — setmaxnreg.dec 40
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.kernel_templates._tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# @@INJECT_TILE_CONSTANTS@@

# Rank decomposition below uses shifts and masks instead of runtime integer
# division.  The catalog satisfies this; keep synthesized configs from silently
# taking the fast path with a non-power-of-two cluster dimension.
if any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: cluster M/N dimensions must be powers of two")
if fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: fallback cluster M/N dimensions must be powers of two")

# Keep the two launch alternatives as host constants and spell the preferred /
# fallback operations at each use site. This exposes constant masks and shift
# alternatives before backend canonicalization.
_preferred_cluster_m_shift = cluster_shape_mnk[0].bit_length() - 1
_preferred_cluster_n_shift = cluster_shape_mnk[1].bit_length() - 1
_fallback_cluster_m_shift = _preferred_cluster_m_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[0].bit_length() - 1
_fallback_cluster_n_shift = _preferred_cluster_n_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[1].bit_length() - 1


# Scheduler ring depth.
CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue handoff around the TMA store
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc handoff
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch.

    ``tile_swizzle_n > 0`` pins it. Otherwise: the walk keeps one operand slice
    resident and re-reads the other every super-block, so block along the SHORTER
    problem side. Once that side outgrows what L2 can hold onto while C streams
    through it, keeping it is no longer free -- fall back to the widest N block the
    budget does cover.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


@cute.kernel
def _kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    # @@INJECT_KERNEL_TAP_PARAMS@@
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_KERNEL_TMA_C_PARAMS@@
    # @@TMA_STORE_ONLY:END@@
) -> None:
    # @@INJECT_AB_DESC_LISTS@@
    tma_a_desc = tma_a_descs[0]
    tma_b_desc = tma_b_descs[0]
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_TMA_C_LISTS@@
    # @@TMA_STORE_ONLY:END@@

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    mainloop_warp_id_start = 8
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    # Mixed CGA: the launch carries a preferred (wide) cluster plus a smaller
    # fallback one, and the device picks per cluster — a CTA can only tell which
    # by reading the hardware cluster dims. Everything cluster-shaped below then
    # follows from those, so the two kinds share one body; only the multicast bit
    # patterns are loop-built and come in precomputed per shape.
    a_mcast_pattern = mixed_a_pattern_pref
    b_mcast_pattern = mixed_b_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        b_mcast_pattern = cutlass.Int32(mixed_b_pattern_pref)
        # Bitwise, not `or`: both operands are runtime Booleans (this is the form
        # cutlass.cute.experimental.is_preferred_cluster uses).
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
            b_mcast_pattern = cutlass.Int32(mixed_b_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    # Every catalog cluster dimension is a power of two.  Mixed-CGA makes the
    # divisor runtime-visible, so spelling rank decomposition as div/mod would
    # otherwise lower to reciprocal-based integer division in every warp.
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift
    pair_member = m_rank % 2
    pair_m_idx = m_rank // 2
    is_pair_leader = pair_member == 0
    pair_leader_rank = pair_m_idx * 2 + n_rank * cluster_m

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        nvvm.prefetch_tensormap(tma_a_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_b_desc.get_ptr())

        # @@TMA_STORE_ONLY:BEGIN@@
        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())
        # @@TMA_STORE_ONLY:END@@

    init_raw_m = bidx >> _preferred_cluster_m_shift
    init_raw_n = bidy >> _preferred_cluster_n_shift
    init_nt_m = gridx >> _preferred_cluster_m_shift
    init_nt_n = gridy >> _preferred_cluster_n_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            init_raw_m = bidx >> _fallback_cluster_m_shift
            init_raw_n = bidy >> _fallback_cluster_n_shift
            init_nt_m = gridx >> _fallback_cluster_m_shift
            init_nt_n = gridy >> _fallback_cluster_n_shift
    swizzle_w = _auto_swizzle_w(m, n, k, init_nt_n)
    init_tile_m, init_tile_n = _l2_swizzle_tile(
        init_raw_m,
        init_raw_n,
        init_nt_m,
        init_nt_n,
        swizzle_w,
        identity=tile_swizzle_n == 1,
    )
    init_tile_l = bidz

    if cutlass.const_expr(multicast_a):
        tma_mcast_mask_a = cutlass.Int16(a_mcast_pattern << m_rank)
    else:
        tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
    if cutlass.const_expr(multicast_b):
        tma_mcast_mask_b = cutlass.Int16((b_mcast_pattern << pair_member) << (n_rank * cluster_m))
    else:
        tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    a_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    b_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    mainloop_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    _clc_response_raw = cutlass.Array(cutlass.Int128, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    clc_response_ptr_base = cute.make_ptr(
        cutlass.Int128,
        _clc_response_raw.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )
    clc_full_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, CLC_SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    clc_full_mbar_cute_base = cute.make_ptr(
        cutlass.Int64,
        clc_full_mbar_ptr.data_ptr(),
        mem_space=cute.AddressSpace.smem,
    )

    sA_elems = cta_tile_mnk[0] * cgrp_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cgrp_tile_mnk[2]
    smem_a = cutlass.Array(ab_dtype, sA_elems * ab_stages, space=cutlass.AddressSpace.smem, alignment=1024)
    smem_b = cutlass.Array(ab_dtype, sB_elems * ab_stages, space=cutlass.AddressSpace.smem, alignment=1024)
    if cutlass.const_expr(mainloop_a_cast):
        smem_a_load = cutlass.Array(
            ab_load_a_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
    if cutlass.const_expr(mainloop_b_cast):
        smem_b_load = cutlass.Array(
            ab_load_b_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )

    # @@TMA_STORE_ONLY:BEGIN@@
    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    # The ring slot is indexed by `tidx`, so its row count is the EPILOGUE THREAD
    # count -- which is epi_tile_mn[0] only when the MMA M block is 128.
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # @@TMA_STORE_ONLY:END@@

    acc_empty_count = num_epilogue_warps * 2
    cta_group = 2
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_count = cluster_size // cta_group
    else:
        ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    num_consumer_warps_per_cta = 7 + num_mainloop_warps
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    mainloop_full_count = num_mainloop_warps * 2
    if warp_idx == 0:
        if elect_one:
            nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(a_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(b_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(mainloop_full_mbar_ptr.subview(i), mainloop_full_count)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), clc_empty_count)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    if cutlass.const_expr(mainloop_a_cast):
        sA_tma_bytes = sA_elems * (ab_load_a_dtype.width // 8)
    else:
        sA_tma_bytes = sA_bytes
    if cutlass.const_expr(mainloop_b_cast):
        sB_tma_bytes = sB_elems * (ab_load_b_dtype.width // 8)
    else:
        sB_tma_bytes = sB_bytes

    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    # Per-CTA logical tile — the cluster cancels out, so these stay compile-time
    # constants even when the cluster shape is only known at runtime.
    logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
    logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
    pair_n_size = logical_cta_tile_n
    # Per-CTA output rows one MMA-M block covers. The pair splits M, so this is
    # the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // num_mma_m
    if cutlass.const_expr(epi_rows_per_mma_m == 64):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        epi_cols_per_mma_m = pair_n_size // 2
    else:
        epi_cols_per_mma_m = pair_n_size
    # M block mi owns columns [mi*epi_cols_per_mma_m, +epi_cols_per_mma_m) of its
    # GEMM's region, all at TMEM lane base 0. N is NOT split across instructions
    # here: the pair already splits B's N, so an N sub-block would not be
    # contiguous in output N (the CTA tile is never split along N).
    cols_per_acc_stage = num_mma_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    nvvm.barrier_cluster_wait()
    nvvm.barrier_cta_sync(0)

    # @@INJECT_TAP_PTRS@@

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cgrp_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
    cgrp_tile_n_cur = logical_cta_tile_n * cluster_n
    num_k_blocks = cgrp_tile_mnk[2] // mma_inst_shape_mnk[2]

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        sched_iter = cutlass.Int32(0)
        clc_empty_phase = cutlass.Int32(1)
        clc_full_phase = cutlass.Int32(0)
        is_valid_sched = cutlass.Int32(1)
        while is_valid_sched != 0:
            stage = sched_iter % CLC_SCHED_STAGES
            if stage == 0 and sched_iter != 0:
                clc_empty_phase = clc_empty_phase ^ 1
                clc_full_phase = clc_full_phase ^ 1

            if is_cluster_leader_cta:
                while not nvvm.mbarrier_try_wait_parity(clc_empty_mbar_ptr.subview(stage), clc_empty_phase, time_limit=10_000_000):
                    pass

            if elect_one:
                nvvm.mbarrier_arrive_expect_tx(clc_full_mbar_ptr.subview(stage), 16)

            if is_cluster_leader_cta:
                if elect_one:
                    cute_clc.issue_clc_query(
                        clc_full_mbar_cute_base + stage,
                        clc_response_ptr_base + stage,
                        multicast=True,
                    )

            while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(stage), clc_full_phase, time_limit=10_000_000):
                pass

            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid_sched = vld

            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            sched_iter += 1

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            if is_cluster_leader_cta:
                for _ in range(CLC_SCHED_STAGES):
                    stage = sched_iter % CLC_SCHED_STAGES
                    if stage == 0 and sched_iter != 0:
                        clc_empty_phase = clc_empty_phase ^ 1
                    while not nvvm.mbarrier_try_wait_parity(
                        clc_empty_mbar_ptr.subview(stage),
                        clc_empty_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    sched_iter += 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_tma = cutlass.Int32(0)
        while is_valid != 0:
            coord_m_per_cta = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
            if cutlass.const_expr(matmul_a_batch == 1):
                tile_l_a = cutlass.Int32(0)
            else:
                tile_l_a = tile_l
            if cutlass.const_expr(matmul_b_batch == 1):
                tile_l_b = cutlass.Int32(0)
            else:
                tile_l_b = tile_l

            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                    pass

                sA_stage = smem_a.subview(sA_elems * stage)
                sB_stage = smem_b.subview(sB_elems * stage)
                coord_k = k_tile_idx * cgrp_tile_mnk[2]
                if cutlass.const_expr(mainloop_a_cast):
                    sA_tma_dst = smem_a_load.subview(sA_elems * stage)
                else:
                    sA_tma_dst = sA_stage
                if cutlass.const_expr(mainloop_b_cast):
                    sB_tma_dst = smem_b_load.subview(sB_elems * stage)
                else:
                    sB_tma_dst = sB_stage

                if cutlass.const_expr(mainloop_fuse_a):
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(stage), sA_tma_bytes)
                else:
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(stage), sA_tma_bytes * 2)
                if cutlass.const_expr(mainloop_fuse_b):
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(b_full_mbar_ptr.subview(stage), sB_tma_bytes)
                else:
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(b_full_mbar_ptr.subview(stage), sB_tma_bytes * 2)
                if cutlass.const_expr(mainloop_fuse_a):
                    a_self_mask = cutlass.Int16(1) << cta_rank_in_cluster
                    if cutlass.const_expr(a_is_m_major):
                        for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_tma_dst.subview(m_group * a_tma_group_elems * cgrp_tile_mnk[2]),
                                    tma_a_desc.get_ptr(),
                                    (
                                        coord_m_per_cta + m_group * a_tma_group_elems,
                                        coord_k,
                                        tile_l_a,
                                    ),
                                    a_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=a_self_mask,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sA_tma_dst,
                                tma_a_desc.get_ptr(),
                                (coord_k, coord_m_per_cta, tile_l_a),
                                a_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=a_self_mask,
                                group=nvvm.CTAGroup.CTA_1,
                            )
                elif cutlass.const_expr(a_mcast_slices > 1):
                    _a_rows = cta_tile_mnk[0] // a_mcast_slices
                    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sA_tma_dst.subview(n_rank * _a_rows * cta_tile_mnk[2]),
                                tma_a_desc.get_ptr(),
                                (coord_k, coord_m_per_cta + n_rank * _a_rows, tile_l_a),
                                a_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_a,
                                group=nvvm.CTAGroup.CTA_2,
                            )
                    else:
                        _a_per_cta = a_mcast_slices >> _preferred_cluster_n_shift
                        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                            _a_per_cta = a_mcast_slices >> _fallback_cluster_n_shift
                        for _asl in cutlass.range(_a_per_cta):
                            _a_idx = n_rank * _a_per_cta + _asl
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_tma_dst.subview(_a_idx * _a_rows * cta_tile_mnk[2]),
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta + _a_idx * _a_rows, tile_l_a),
                                    a_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                elif cutlass.const_expr(multicast_a):
                    if n_rank == 0:
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_tma_dst.subview(m_group * a_tma_group_elems * cgrp_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (
                                            coord_m_per_cta + m_group * a_tma_group_elems,
                                            coord_k,
                                            tile_l_a,
                                        ),
                                        a_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=nvvm.CTAGroup.CTA_2,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_tma_dst,
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta, tile_l_a),
                                    a_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                else:
                    if cutlass.const_expr(a_is_m_major):
                        for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_tma_dst.subview(m_group * a_tma_group_elems * cgrp_tile_mnk[2]),
                                    tma_a_desc.get_ptr(),
                                    (
                                        coord_m_per_cta + m_group * a_tma_group_elems,
                                        coord_k,
                                        tile_l_a,
                                    ),
                                    a_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sA_tma_dst,
                                tma_a_desc.get_ptr(),
                                (coord_k, coord_m_per_cta, tile_l_a),
                                a_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_a,
                                group=nvvm.CTAGroup.CTA_2,
                            )

                if cutlass.const_expr(mainloop_fuse_b):
                    b_self_mask = cutlass.Int16(1) << cta_rank_in_cluster
                    if cutlass.const_expr(b_is_n_major):
                        for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_tma_dst.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                    tma_b_desc.get_ptr(),
                                    (
                                        coord_n_per_cta + n_group * b_tma_group_elems,
                                        coord_k,
                                        tile_l_b,
                                    ),
                                    b_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=b_self_mask,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sB_tma_dst,
                                tma_b_desc.get_ptr(),
                                (coord_k, coord_n_per_cta, tile_l_b),
                                b_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=b_self_mask,
                                group=nvvm.CTAGroup.CTA_1,
                            )
                elif cutlass.const_expr(b_mcast_slices > 1):
                    _b_rows = cta_tile_mnk[1] // b_mcast_slices
                    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sB_tma_dst.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                tma_b_desc.get_ptr(),
                                (coord_k, coord_n_per_cta + pair_m_idx * _b_rows, tile_l_b),
                                b_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=nvvm.CTAGroup.CTA_2,
                            )
                    else:
                        _b_per_cta = b_mcast_slices >> (_preferred_cluster_m_shift - 1)
                        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                            _b_per_cta = b_mcast_slices >> (_fallback_cluster_m_shift - 1)
                        for _bsl in cutlass.range(_b_per_cta):
                            _b_idx = pair_m_idx * _b_per_cta + _bsl
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_tma_dst.subview(_b_idx * _b_rows * cta_tile_mnk[2]),
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                    b_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                elif cutlass.const_expr(multicast_b):
                    if pair_m_idx == 0:
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_tma_dst.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k,
                                            tile_l_b,
                                        ),
                                        b_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_2,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_tma_dst,
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta, tile_l_b),
                                    b_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                else:
                    if cutlass.const_expr(b_is_n_major):
                        for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_tma_dst.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                    tma_b_desc.get_ptr(),
                                    (
                                        coord_n_per_cta + n_group * b_tma_group_elems,
                                        coord_k,
                                        tile_l_b,
                                    ),
                                    b_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sB_tma_dst,
                                tma_b_desc.get_ptr(),
                                (coord_k, coord_n_per_cta, tile_l_b),
                                b_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=nvvm.CTAGroup.CTA_2,
                            )

                ab_iter += 1

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_tma = clc_full_phase_tma ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_tma,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            tma_raw_m = m_idx >> _preferred_cluster_m_shift
            tma_raw_n = n_idx >> _preferred_cluster_n_shift
            tma_nt_m = gridx >> _preferred_cluster_m_shift
            tma_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    tma_raw_m = m_idx >> _fallback_cluster_m_shift
                    tma_raw_n = n_idx >> _fallback_cluster_n_shift
                    tma_nt_m = gridx >> _fallback_cluster_m_shift
                    tma_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                tma_raw_m,
                tma_raw_n,
                tma_nt_m,
                tma_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1

        tail_stage = ab_iter % ab_stages
        tail_phase = ab_empty_phase_bit
        if tail_stage == 0 and ab_iter != 0:
            tail_phase = tail_phase ^ 1
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(ab_stages):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == ab_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    pair_mask = cutlass.Int16(3) << pair_leader_rank
    a_arrive_pattern = a_mcast_pattern
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_arrive_pattern = (1 << cluster_m) - 1
    else:
        b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
    a_part = a_arrive_pattern << m_rank
    a_part = a_part | (a_part << 1)
    b_part = b_arrive_pattern << (n_rank * cluster_m)
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=nvvm.CTAGroup.CTA_2,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        peer_cta_rank = cta_rank_in_cluster ^ 1
        if is_pair_leader:
            a_full_phase_bit = cutlass.Int32(0)
            b_full_phase_bit = cutlass.Int32(0)
            mainloop_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_mma = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            # Descriptor metadata and the SMEM allocation base are invariant
            # across persistent tiles.  Only the encoded start address advances.
            desc_a_root = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_a,
                leading_byte_offset=a_smem_desc_leading_byte_offset,
                stride_byte_offset=a_smem_desc_stride_byte_offset,
                layout=ab_smem_swizzle,
            )
            desc_b_root = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_b,
                leading_byte_offset=b_smem_desc_leading_byte_offset,
                stride_byte_offset=b_smem_desc_stride_byte_offset,
                layout=ab_smem_swizzle,
            )
            while is_valid != 0:
                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(acc_stage),
                    acc_empty_phase_bit,
                    time_limit=10_000_000,
                ):
                    pass

                acc_col_id = base_col_id_root + acc_stage * cols_per_acc_stage
                # One accumulator per M block, all at TMEM lane base 0.
                tmem_addr_mmas = [cutlass.inttoptr((base_row_id << 16) | (acc_col_id + mi * epi_cols_per_mma_m), 6, cutlass.Int32) for mi in range(num_mma_m)]

                scale_d = cutlass.Boolean(False)
                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        a_full_phase_bit = a_full_phase_bit ^ 1
                        b_full_phase_bit = b_full_phase_bit ^ 1
                        mainloop_full_phase_bit = mainloop_full_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        mainloop_full_mbar_ptr.subview(stage),
                        mainloop_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass
                    if cutlass.const_expr(not mainloop_fuse_a):
                        while not nvvm.mbarrier_try_wait_parity(
                            a_full_mbar_ptr.subview(stage),
                            a_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass
                    if cutlass.const_expr(not mainloop_fuse_b):
                        while not nvvm.mbarrier_try_wait_parity(
                            b_full_mbar_ptr.subview(stage),
                            b_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass

                    desc_a_base = desc_a_root.advance_start_address(sA_bytes * stage)
                    desc_b_base = desc_b_root.advance_start_address(sB_bytes * stage)

                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        desc_a_k = desc_a_base.advance_start_address(a_smem_k_step_bytes * k_block_idx)
                        desc_b = desc_b_base.advance_start_address(b_smem_k_step_bytes * k_block_idx)
                        for mi in cutlass.range_constexpr(num_mma_m):
                            # The M sub-block offset is a whole SMEM swizzle atom, so the
                            # descriptor's swizzle phase is preserved. B is shared.
                            desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                            if elect_one:
                                nvvm.tcgen05_mma(
                                    mma_kind,
                                    nvvm.CTAGroup.CTA_2,
                                    tmem_addr_mmas[mi],
                                    desc_a,
                                    desc_b,
                                    idesc,
                                    scale_d,
                                )
                        # Every accumulator sees scale_d=False on exactly the first
                        # k_block of the tile, so the flip stays outside mi.
                        scale_d = cutlass.Boolean(True)

                    if elect_one:
                        nvvm.tcgen05_commit(
                            ab_empty_mbar_ptr.subview(stage),
                            multicast_mask=ab_empty_arrive_mask,
                            group=nvvm.CTAGroup.CTA_2,
                        )
                    ab_iter += 1

                if elect_one:
                    nvvm.tcgen05_commit(
                        acc_full_mbar_ptr.subview(acc_stage),
                        multicast_mask=pair_mask,
                        group=nvvm.CTAGroup.CTA_2,
                    )

                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_mma = clc_full_phase_mma ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_mma,
                    time_limit=10_000_000,
                ):
                    pass
                _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            tail_stage = acc_stage
            tail_phase = acc_empty_phase_bit
            for _ in range(acc_stages):
                tail_stage = tail_stage + 1
                if tail_stage == acc_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    acc_empty_mbar_ptr.subview(tail_stage),
                    tail_phase,
                    time_limit=10_000_000,
                ):
                    pass
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
            peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
            while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                pass
            nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=nvvm.CTAGroup.CTA_2,
            )
        else:
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_mma = cutlass.Int32(0)
            while is_valid != 0:
                consumer_stage = tile_iter % CLC_SCHED_STAGES
                if consumer_stage == 0 and tile_iter != 0:
                    clc_full_phase_mma = clc_full_phase_mma ^ 1
                while not nvvm.mbarrier_try_wait_parity(
                    clc_full_mbar_ptr.subview(consumer_stage),
                    clc_full_phase_mma,
                    time_limit=10_000_000,
                ):
                    pass
                _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                cute.arch.fence_proxy("async.shared", space="cta")
                is_valid = vld
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
            peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
            nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                pass
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=nvvm.CTAGroup.CTA_2,
            )

    if warp_idx >= mainloop_warp_id_start:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        lane = tidx % 32
        ml_local = warp_idx - mainloop_warp_id_start
        ml_threads = num_mainloop_warps * 32
        ml_tid = ml_local * 32 + lane
        ml_vec_bytes = 16
        ml_vec_elems = ml_vec_bytes // (ab_dtype.width // 8)
        # Full per-thread rounds + the remainder vectors (threads < ml_tail_b
        # each take one extra). A floor alone would leave the tile's tail
        # untransformed whenever sB_elems isn't a multiple of 128 vectors —
        # e.g. any cta_tile_n with (N/2)*K_BYTES % 2048 != 0 on the B side.
        # A's extent (cta_m in {64,128} × K_BYTES) always divides exactly.
        ml_chunks_a = (sA_elems // ml_vec_elems) // ml_threads
        ml_chunks_b = (sB_elems // ml_vec_elems) // ml_threads
        ml_tail_b = (sB_elems // ml_vec_elems) % ml_threads

        a_full_phase_bit_ml = cutlass.Int32(0)
        b_full_phase_bit_ml = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_ml = cutlass.Int32(0)
        while is_valid != 0:
            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    a_full_phase_bit_ml = a_full_phase_bit_ml ^ 1
                    b_full_phase_bit_ml = b_full_phase_bit_ml ^ 1

                if cutlass.const_expr(mainloop_fuse_a):
                    while not nvvm.mbarrier_try_wait_parity(
                        a_full_mbar_ptr.subview(stage),
                        a_full_phase_bit_ml,
                        time_limit=10_000_000,
                    ):
                        pass
                    sA_stage = smem_a.subview(sA_elems * stage)
                    if cutlass.const_expr(mainloop_a_cast):
                        sA_load_stage = smem_a_load.subview(sA_elems * stage)

                        ml_offsets = []
                        ml_vecs = []
                        for i in cutlass.range_constexpr(ml_chunks_a):
                            ml_L = (ml_tid + i * ml_threads) * ml_vec_elems
                            ml_offsets.append(ml_L)
                            ml_vecs.append(
                                (sA_load_stage.subview(ml_L)).load(
                                    vector_size=ml_vec_elems,
                                    alignment=ml_vec_elems * (ab_load_a_dtype.width // 8),
                                )
                            )

                        ml_outs = []
                        for i in cutlass.range_constexpr(ml_chunks_a):
                            ml_vec_a = ml_vecs[i]
                            # @@INJECT_MAINLOOP_A@@
                            ml_outs.append(ml_out_a)

                        for i in cutlass.range_constexpr(ml_chunks_a):
                            (sA_stage.subview(ml_offsets[i])).data_ptr().store_swizzled(ml_outs[i], ab_swizzle, alignment=ml_vec_bytes)
                    elif cutlass.const_expr(mainloop_koob_fix):
                        a_kvalid = cutlass.Int32(k - k_tile_idx * cgrp_tile_mnk[2])
                        for chunk in cutlass.range_constexpr(ml_chunks_a):
                            ml_L = (ml_tid + chunk * ml_threads) * ml_vec_elems
                            ml_ptr_a = (sA_stage.subview(ml_L)).data_ptr()
                            ml_vec_a = ml_ptr_a.load_swizzled(ab_swizzle, alignment=ml_vec_bytes, count=ml_vec_elems)

                            # @@INJECT_MAINLOOP_A@@

                            # K coordinate of the flat SMEM index: K is the fast dim of a K-major
                            # tile, but an M-major tile is stored as M-groups of a_tma_group_elems.
                            if cutlass.const_expr(a_is_m_major):
                                ml_k_of_L = (ml_L % (a_tma_group_elems * cgrp_tile_mnk[2])) // a_tma_group_elems
                            else:
                                ml_k_of_L = ml_L % cgrp_tile_mnk[2]
                            ml_kloc = cutlass.full_like(ml_f32_a, cutlass.Float32(ml_k_of_L))
                            ml_klim = cutlass.full_like(ml_f32_a, cutlass.Float32(a_kvalid))
                            ml_out_a = _cvec.where(
                                ml_kloc >= ml_klim,
                                ml_f32_a - ml_f32_a,
                                ml_out_a.to(cutlass.Float32),
                            ).to(ab_dtype)
                            ml_ptr_a.store_swizzled(ml_out_a, ab_swizzle, alignment=ml_vec_bytes)
                    else:
                        for chunk in cutlass.range_constexpr(ml_chunks_a):
                            ml_idx = (ml_tid + chunk * ml_threads) * ml_vec_elems
                            ml_ptr_a = sA_stage.subview(ml_idx)
                            ml_vec_a = ml_ptr_a.load(vector_size=ml_vec_elems, alignment=ml_vec_bytes)

                            # @@INJECT_MAINLOOP_A@@

                            ml_ptr_a.store(ml_out_a, alignment=ml_vec_bytes)

                if cutlass.const_expr(mainloop_fuse_b):
                    while not nvvm.mbarrier_try_wait_parity(
                        b_full_mbar_ptr.subview(stage),
                        b_full_phase_bit_ml,
                        time_limit=10_000_000,
                    ):
                        pass
                    sB_stage = smem_b.subview(sB_elems * stage)
                    if cutlass.const_expr(mainloop_b_cast):
                        sB_load_stage = smem_b_load.subview(sB_elems * stage)

                        ml_offsets = []
                        ml_vecs = []
                        for i in cutlass.range_constexpr(ml_chunks_b):
                            ml_L = (ml_tid + i * ml_threads) * ml_vec_elems
                            ml_offsets.append(ml_L)
                            ml_vecs.append(
                                (sB_load_stage.subview(ml_L)).load(
                                    vector_size=ml_vec_elems,
                                    alignment=ml_vec_elems * (ab_load_b_dtype.width // 8),
                                )
                            )

                        ml_outs = []
                        for i in cutlass.range_constexpr(ml_chunks_b):
                            ml_vec_b = ml_vecs[i]
                            # @@INJECT_MAINLOOP_B@@
                            ml_outs.append(ml_out_b)

                        for i in cutlass.range_constexpr(ml_chunks_b):
                            (sB_stage.subview(ml_offsets[i])).data_ptr().store_swizzled(ml_outs[i], ab_swizzle, alignment=ml_vec_bytes)

                        if cutlass.const_expr(ml_tail_b != 0):
                            # Tail: every thread loads/transforms a clamped
                            # position (out-of-range threads re-read the last
                            # vector); only the tail threads store. Keeps every
                            # name defined on both paths of the dynamic guard.
                            ml_L = cutlass.min(
                                cutlass.Int32((ml_tid + ml_chunks_b * ml_threads) * ml_vec_elems),
                                cutlass.Int32(sB_elems - ml_vec_elems),
                            )
                            ml_vec_b = (sB_load_stage.subview(ml_L)).load(
                                vector_size=ml_vec_elems,
                                alignment=ml_vec_elems * (ab_load_b_dtype.width // 8),
                            )
                            # @@INJECT_MAINLOOP_B@@
                            if ml_tid < ml_tail_b:
                                (sB_stage.subview(ml_L)).data_ptr().store_swizzled(ml_out_b, ab_swizzle, alignment=ml_vec_bytes)
                    else:
                        for chunk in cutlass.range_constexpr(ml_chunks_b):
                            ml_idx = (ml_tid + chunk * ml_threads) * ml_vec_elems
                            ml_ptr_b = sB_stage.subview(ml_idx)
                            ml_vec_b = ml_ptr_b.load(vector_size=ml_vec_elems, alignment=ml_vec_bytes)

                            # @@INJECT_MAINLOOP_B@@

                            ml_ptr_b.store(ml_out_b, alignment=ml_vec_bytes)

                        if cutlass.const_expr(ml_tail_b != 0):
                            # Tail: clamped unconditional load/transform, store
                            # under the predicate (see the cast-path tail note).
                            ml_idx = cutlass.min(
                                cutlass.Int32((ml_tid + ml_chunks_b * ml_threads) * ml_vec_elems),
                                cutlass.Int32(sB_elems - ml_vec_elems),
                            )
                            ml_ptr_b = sB_stage.subview(ml_idx)
                            ml_vec_b = ml_ptr_b.load(vector_size=ml_vec_elems, alignment=ml_vec_bytes)

                            # @@INJECT_MAINLOOP_B@@

                            if ml_tid < ml_tail_b:
                                ml_ptr_b.store(ml_out_b, alignment=ml_vec_bytes)

                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if elect_one:
                    ml_full_remote = nvvm.mapa(mainloop_full_mbar_ptr.subview(stage), pair_leader_rank)
                    nvvm.mbarrier_arrive(ml_full_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                ab_iter += 1

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_ml = clc_full_phase_ml ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_ml,
                time_limit=10_000_000,
            ):
                pass
            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1

    if warp_idx < num_epilogue_warps:
        nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)

        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@
        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # @@TMA_STORE_ONLY:END@@

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * pair_n_size
            if cutlass.const_expr(epi_rows_per_mma_m == 64):
                coord_n_c = coord_n_c + (warp_idx // 2) * epi_cols_per_mma_m

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            for mi in cutlass.range_constexpr(num_mma_m):
                coord_m = coord_m_tile + mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * cols_per_acc_stage) for g in range(num_gemms)]

                if cutlass.const_expr(epi_rows_per_mma_m == 64):
                    row = coord_m + (warp_idx % 2) * 32 + lane
                    row_active = True
                else:
                    row = coord_m + tidx
                    row_active = True

                # @@INJECT_AUX_VIEWS@@

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                    c_rmem_vecs = []
                    for g in cutlass.range_constexpr(num_gemms):
                        subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                        tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                        _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w)
                        # INT8 int32 accumulate → widen to fp32 (skipped for int32 output).
                        if cutlass.const_expr(acc_widen_to_fp32):
                            _accf = _cv.to(cutlass.Float32)
                            # `+ 0.0` forces a fresh fp32 register so int32->fp32 isn't folded into an invalid int32->fp8 cast.
                            _cv = _accf + cutlass.full_like(_accf, 0.0)
                        c_rmem_vecs.append(_cv)
                    c_rmem_vec = c_rmem_vecs[0]

                    if mi == num_mma_m - 1 and subtile_idx == subtile_cnt - 1:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if elect_one:
                            nvvm.mbarrier_arrive(
                                nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                scope=nvvm.MemScope.CLUSTER,
                                relaxed=True,
                            )

                    col = coord_n_c + subtile_col_offset

                    # @@TMA_STORE_ONLY:BEGIN@@
                    vec_f32 = c_rmem_vec
                    col_j = col
                    linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                    # @@INJECT_EPILOGUE@@

                    # @@INJECT_TMA_STORE_SEQUENCE@@
                    # @@TMA_STORE_ONLY:END@@

                    # @@STG_ONLY:BEGIN@@
                    if row_active and row < M:
                        for j in cutlass.range_constexpr(subtile_w // vsize):
                            col_j = col + j * vsize
                            if col_j + vsize <= N:
                                vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                                # @@INJECT_STG_VEC_BINDINGS@@

                                # @@INJECT_EPILOGUE@@
                    # @@STG_ONLY:END@@

            # @@EPILOGUE_DRAIN:END@@
            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_epi = clc_full_phase_epi ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_epi,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            epi_raw_m = m_idx >> _preferred_cluster_m_shift
            epi_raw_n = n_idx >> _preferred_cluster_n_shift
            epi_nt_m = gridx >> _preferred_cluster_m_shift
            epi_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    epi_raw_m = m_idx >> _fallback_cluster_m_shift
                    epi_raw_n = n_idx >> _fallback_cluster_n_shift
                    epi_nt_m = gridx >> _fallback_cluster_m_shift
                    epi_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                epi_raw_m,
                epi_raw_n,
                epi_nt_m,
                epi_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            tile_iter += 1

        # @@TMA_STORE_ONLY:BEGIN@@
        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)
        # @@TMA_STORE_ONLY:END@@

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _host(
    problem_size: tuple,
    # @@INJECT_HOST_AB_PARAMS@@
    # @@INJECT_HOST_TAP_PARAMS@@
    # @@INJECT_HOST_AUX_PARAMS@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_HOST_TMA_C_PARAMS@@
    # @@TMA_STORE_ONLY:END@@
    stream: _cuda.CUstream,
) -> None:
    # @@INJECT_HOST_AB_LISTS@@
    a = _a_operands[0]
    b = _b_operands[0]
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    batch = problem_size[3]
    a_stride_m = problem_size[4]
    a_stride_k = problem_size[5]
    a_stride_l = problem_size[6]
    b_stride_n = problem_size[7]
    b_stride_k = problem_size[8]
    b_stride_l = problem_size[9]
    _stride_idx = 10
    # @@INJECT_HOST_REDUCTION_STRIDES@@

    if cutlass.const_expr(matmul_a_batch == 1):
        a_batch = 1
    else:
        a_batch = batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_batch = 1
    else:
        b_batch = batch
    if cutlass.const_expr(a_is_m_major):
        a_tma_tensor = cute.make_tensor(
            a.iterator,
            cute.make_layout((m, k_sym, a_batch), stride=(1, a_stride_k, a_stride_l)),
        )
        a_box_dims = (a_tma_group_elems, cgrp_tile_mnk[2], 1)
        a_stride_order = (0, 1, 2)
    else:
        a_tma_tensor = cute.make_tensor(
            a.iterator,
            cute.make_layout((m, k_sym, a_batch), stride=(a_stride_m, 1, a_stride_l)),
        )
        a_box_dims = (cta_tile_mnk[0] // a_mcast_slices, cgrp_tile_mnk[2], 1)
        a_stride_order = (1, 0, 2)
    if cutlass.const_expr(b_is_n_major):
        b_tma_tensor = cute.make_tensor(
            b.iterator,
            cute.make_layout((n, k_sym, b_batch), stride=(1, b_stride_k, b_stride_l)),
        )
        b_box_dims = (b_tma_group_elems, cgrp_tile_mnk[2], 1)
        b_stride_order = (0, 1, 2)
    else:
        b_tma_tensor = cute.make_tensor(
            b.iterator,
            cute.make_layout((n, k_sym, b_batch), stride=(b_stride_n, 1, b_stride_l)),
        )
        b_box_dims = (cta_tile_mnk[1] // b_mcast_slices, cgrp_tile_mnk[2], 1)
        b_stride_order = (1, 0, 2)
    if cutlass.const_expr(mainloop_a_cast):
        a_desc_dtype = ab_load_a_dtype
        a_desc_swizzle = _tma.TensorMapSwizzle.none
    else:
        a_desc_dtype = ab_tma_dtype
        a_desc_swizzle = ab_tma_swizzle
    if cutlass.const_expr(mainloop_b_cast):
        b_desc_dtype = ab_load_b_dtype
        b_desc_swizzle = _tma.TensorMapSwizzle.none
    else:
        b_desc_dtype = ab_tma_dtype
        b_desc_swizzle = ab_tma_swizzle
    tma_a_desc = _tma.create_tensor_map_tiled_from_view(
        a_tma_tensor,
        dtype=a_desc_dtype,
        box_dims=a_box_dims,
        stride_order=a_stride_order,
        swizzle=a_desc_swizzle,
    )
    tma_b_desc = _tma.create_tensor_map_tiled_from_view(
        b_tma_tensor,
        dtype=b_desc_dtype,
        box_dims=b_box_dims,
        stride_order=b_stride_order,
        swizzle=b_desc_swizzle,
    )
    tma_a_desc_list = [tma_a_desc]
    tma_b_desc_list = [tma_b_desc]

    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_HOST_TMA_C_LISTS@@
    # @@INJECT_HOST_TMA_C_DESCS@@
    # @@TMA_STORE_ONLY:END@@

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, batch)
    launch = _kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        # @@INJECT_HOST_TAP_PASS@@
        # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        # @@INJECT_HOST_AUX_PASS@@
        # @@TMA_STORE_ONLY:BEGIN@@
        # @@INJECT_HOST_TMA_C_PASS@@
        # @@TMA_STORE_ONLY:END@@
    )
    # Mixed CGA: `cluster` is the preferred (wide) shape and `fallback_cluster`
    # the regular one the device groups blocks into when a preferred cluster does
    # not fit. The grid is already a multiple of the preferred shape, which the
    # driver requires.
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )
    else:
        launch.launch(
            grid=grid_shape,
            block=(threads_per_cta, 1, 1),
            cluster=cluster_shape_mnk,
            fallback_cluster=fallback_cluster_shape_mnk,
            use_pdl=USE_PDL,
            stream=stream,
        )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    sym_l = cute.sym_int64()
    if matmul_a_batch == 1:
        sym_a_l = 1
    else:
        sym_a_l = sym_l
    if matmul_b_batch == 1:
        sym_b_l = 1
    else:
        sym_b_l = sym_l

    def _make_fake_a():
        return make_fake_compact_tensor(
            ab_load_a_dtype if mainloop_a_cast else mma_a_dtype,
            (sym_m, sym_k, sym_a_l),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            ab_load_b_dtype if mainloop_b_cast else mma_b_dtype,
            (sym_n, sym_k, sym_b_l),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # @@INJECT_COMPILE_AB_FAKES@@
    # @@TMA_STORE_ONLY:BEGIN@@
    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, sym_n // _div, sym_l),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
            assumed_align=16,
        )

    # @@INJECT_COMPILE_TMA_C_FAKES@@
    # @@TMA_STORE_ONLY:END@@
    def _sym_operand_strides(is_mn_major: bool, load_dtype) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        # Strides are in elements of the LOADED dtype, which a mixed-input mainloop makes narrower than the MMA dtype.
        stride_elems = 16 // (load_dtype.width // 8)
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=stride_elems) for i in range(3))

    sym_a_strides = _sym_operand_strides(a_is_m_major, ab_load_a_dtype)
    sym_b_strides = _sym_operand_strides(b_is_n_major, ab_load_b_dtype)
    # @@INJECT_COMPILE_REDUCTION_STRIDE_DECLS@@
    # @@INJECT_COMPILE_TAP_FAKES@@
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_l,
        *sym_a_strides,
        *sym_b_strides,
        # @@INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS@@
    )
    # @@INJECT_COMPILE_AUX_FAKES@@
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(
        _host,
        problem_size,
        # @@INJECT_COMPILE_AB_PASS@@
        # @@INJECT_COMPILE_TAP_PASS@@
        # @@INJECT_COMPILE_AUX_PASS@@
        # @@TMA_STORE_ONLY:BEGIN@@
        # @@INJECT_COMPILE_TMA_C_PASS@@
        # @@TMA_STORE_ONLY:END@@
        stream=_fake_stream,
        options=frost_compile_options,
    )
