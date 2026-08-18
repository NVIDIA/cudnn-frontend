# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm100 CTA_1 GEMM kernel: persistent + double-TMEM + CLC dynamic scheduler (2-stage ring).

Single-CTA MMA — every CTA runs its own MMA on its own SMEM/TMEM (no
leader-follower pair). Compiler picks this when TileConfig.cta_group == 1.

The CTA tile may span several MMA instructions along M (``num_mma_m``, each
``mma_inst_shape_mnk``): the TMA still loads the whole tile in one go, the MMA
warp issues one instruction per (M, N) sub-block into its own TMEM column
region, and the epilogue drains one MMA-M block per pass.

Warp layout (8 warps × 32 = 256 threads/CTA):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (every CTA runs MMA — no pair structure)  — setmaxnreg.dec 40
  warp  5   : TMA producer  — setmaxnreg.dec 40
  warp  6   : CLC scheduler (leader CTA issues queries; every CTA waits + reads + arrives empty)  — setmaxnreg.dec 40
  warp  7   : unused donor — setmaxnreg.dec 40, idle to dealloc barrier
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
from cutlass import apply_swizzle as _apply_smem_swizzle
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# @@INJECT_TILE_CONSTANTS@@


CLC_SCHED_STAGES = 2

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path
EPI_SMEM_STAGES = 2

# Named barrier id for cross-warp sync of the 4 epilogue warps
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
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_TMA_C_LISTS@@
    tma_c_desc = tma_c_descs[0]
    # @@TMA_STORE_ONLY:END@@

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
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
    # pattern is loop-built and comes in precomputed per shape.
    a_mcast_pattern = mixed_a_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        # Bitwise, not `or`: both operands are runtime Booleans (this is the form
        # cutlass.cute.experimental.is_preferred_cluster uses).
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        # @@TMA_STORE_ONLY:BEGIN@@
        nvvm.prefetch_tensormap(tma_c_desc.get_ptr())
        # @@TMA_STORE_ONLY:END@@

    swizzle_w = _auto_swizzle_w(m, n, k, gridy // cluster_n)
    init_tile_m, init_tile_n = _l2_swizzle_tile(
        bidx // cluster_m,
        bidy // cluster_n,
        gridx // cluster_m,
        gridy // cluster_n,
        swizzle_w,
    )
    init_tile_l = bidz

    a_pattern = a_mcast_pattern
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_pattern = (1 << cluster_m) - 1
    else:
        b_pattern = (cutlass.Int32(1) << cluster_m) - 1

    if cutlass.const_expr(multicast_a):
        tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
    else:
        tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
    if cutlass.const_expr(multicast_b):
        tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
    else:
        tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster

    a_part_arrive = cutlass.Int16(a_pattern) << m_rank
    b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
    ab_empty_arrive_mask = a_part_arrive | b_part_arrive

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
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

    sA_elems = cta_tile_mnk[0] * cta_tile_mnk[2]
    sB_elems = cta_tile_mnk[1] * cta_tile_mnk[2]
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # @@TMA_STORE_ONLY:BEGIN@@
    # One epilogue subtile = one MMA-M block x 32 cols; the M blocks reuse it.
    epi_subtile_elems = epi_tile_mn[0] * epi_tile_mn[1]
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # @@TMA_STORE_ONLY:END@@

    ab_empty_count = cluster_m + cluster_n - 1
    num_consumer_warps_per_cta = 7
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_epilogue_warps)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), clc_empty_count)
    nvvm.fence_mbarrier_init()

    if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
        nvvm.barrier_cluster_arrive_relaxed()
        nvvm.barrier_cluster_wait()
    else:
        nvvm.barrier_cta_sync(0)

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes

    # One descriptor for every MMA instruction of the tile — the CTA tile spans
    # num_mma_m of them, all the same shape.
    idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
        a_dtype=mma_a_dtype,
        b_dtype=mma_b_dtype,
        c_dtype=mma_c_dtype,
        n_dim=mma_inst_shape_mnk[1],
        m_dim=mma_inst_shape_mnk[0],
        a_major=mma_a_major,
        b_major=mma_b_major,
    )

    # TMEM accumulator layout, per acc stage:
    #   gemm g, M block mi, N block ni  ->  columns
    #   [g*cols_per_acc_stage + mi*epi_cols_per_mma_m + ni*mma_inst_shape_mnk[1], +N)
    # all at TMEM lane base 0. The N blocks tile one M block's column range, so
    # the epilogue drains a whole M block as one contiguous span.
    epi_cols_per_mma_m = cta_tile_mnk[1]
    cols_per_acc_stage = num_mma_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    # @@INJECT_TAP_PTRS@@

    VEC_BYTES = vec_bytes_epi
    vsize = (VEC_BYTES * 8) // cd_dtype.width

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = cta_tile_mnk[0] * cluster_m
    cgrp_tile_n_cur = cta_tile_mnk[1] * cluster_n
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]

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
            if elect_one:
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
            coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
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

                coord_k = k_tile_idx * cta_tile_mnk[2]
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)

                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    if cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_desc.get_ptr(),
                                            (
                                                coord_m_per_cta + m_group * a_tma_group_elems,
                                                coord_k,
                                                tile_l_a,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=nvvm.CTAGroup.CTA_1,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage,
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta, tile_l_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                    else:
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (
                                            coord_m_per_cta + m_group * a_tma_group_elems,
                                            coord_k,
                                            tile_l_a,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage,
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_1,
                                )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    if cutlass.const_expr(multicast_b):
                        if m_rank == 0:
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                tile_l_b,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=nvvm.CTAGroup.CTA_1,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage,
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                    else:
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(n_group * b_tma_group_elems * cta_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k,
                                            tile_l_b,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage,
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_1,
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
            tile_m, tile_n = _l2_swizzle_tile(
                m_idx // cluster_m,
                n_idx // cluster_n,
                gridx // cluster_m,
                gridy // cluster_n,
                swizzle_w,
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
        for _ in range(ab_stages - 1):
            tail_stage = tail_stage + 1
            if tail_stage == ab_stages:
                tail_stage = cutlass.Int32(0)
                tail_phase = tail_phase ^ 1
        if elect_one:
            while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                pass

    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=nvvm.CTAGroup.CTA_1,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        acc_empty_phase_bit = cutlass.Int32(1)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_mma = cutlass.Int32(0)
        acc_stage = cutlass.Int32(0)
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

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            # One accumulator per (gemm, M block). Column arithmetic stays on the
            # encoded (row << 16) | col integer.
            tmem_addr_mmas = [
                [
                    cutlass.inttoptr(
                        (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                        6,
                        cutlass.Int32,
                    )
                    for mi in range(num_mma_m)
                ]
                for g in range(num_gemms)
            ]

            scale_d = cutlass.Boolean(False)
            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                    for g in cutlass.range_constexpr(num_gemms):
                        sA_stage = smem_a_list[gemm_a_idx[g]].subview(sA_elems * stage)
                        sB_stage = smem_b_list[gemm_b_idx[g]].subview(sB_elems * stage)
                        desc_b = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                            start_address=sB_stage,
                            leading_byte_offset=b_smem_desc_leading_byte_offset,
                            stride_byte_offset=b_smem_desc_stride_byte_offset,
                            layout=ab_smem_swizzle,
                        ).advance_start_address(b_smem_k_step_bytes * k_block_idx)
                        for mi in cutlass.range_constexpr(num_mma_m):
                            # The M sub-block offset is a whole SMEM swizzle atom
                            # (mma_inst_m x cta_tile_k_bytes), so the descriptor's
                            # swizzle phase is preserved.
                            desc_a = cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                                start_address=sA_stage,
                                leading_byte_offset=a_smem_desc_leading_byte_offset,
                                stride_byte_offset=a_smem_desc_stride_byte_offset,
                                layout=ab_smem_swizzle,
                            ).advance_start_address(a_smem_m_step_bytes * mi + a_smem_k_step_bytes * k_block_idx)
                            if elect_one:
                                nvvm.tcgen05_mma(
                                    mma_kind,
                                    nvvm.CTAGroup.CTA_1,
                                    tmem_addr_mmas[g][mi],
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
                        group=nvvm.CTAGroup.CTA_1,
                    )
                ab_iter += 1

            if elect_one:
                nvvm.tcgen05_commit(
                    acc_full_mbar_ptr.subview(acc_stage),
                    group=nvvm.CTAGroup.CTA_1,
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
            if elect_one:
                nvvm.griddepcontrol("launch_dependents")

        tail_stage = acc_stage
        tail_phase = acc_empty_phase_bit
        if elect_one:
            for _ in range(acc_stages):
                tail_stage = tail_stage + 1
                if tail_stage == acc_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
                while not nvvm.mbarrier_try_wait_parity(acc_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass

        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
        _tcgen05_dealloc(
            alloc_ptr,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=nvvm.CTAGroup.CTA_1,
        )

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

        # The drain layout follows the MMA INSTRUCTION's M (the TMEM row
        # organisation), not the CTA tile's: at hardware M=64 the accumulator
        # occupies data paths 0-15 of each sub-partition, so the 32-data-path
        # LDTM layout cannot be used to read it.
        if cutlass.const_expr(mma_inst_shape_mnk[0] == 64):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        # One M block's accumulator columns are contiguous (the N-direction MMAs
        # tile them), so one span list drains all of them.
        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(mma_inst_shape_mnk[0] == 64):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32

        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # @@TMA_STORE_ONLY:END@@

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols

            # One pass per MMA-M block: the 4 epilogue warps cover mma_inst M
            # rows at a time, so a CTA tile of num_mma_m blocks drains in
            # num_mma_m passes over its own column region.
            for mi in cutlass.range_constexpr(num_mma_m):
                coord_m = coord_m_tile + mi * mma_inst_shape_mnk[0]
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * cols_per_acc_stage) for g in range(num_gemms)]

                if cutlass.const_expr(mma_inst_shape_mnk[0] == 64):
                    row = coord_m + warp_idx * 16 + lane
                    row_active = lane < 16
                else:
                    row = coord_m + tidx
                    row_active = True

                # @@INJECT_AUX_VIEWS@@

                for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                    subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                    if cutlass.const_expr(not (use_tma_store_epi and cd_out_is_m_major)):
                        c_rmem_vecs = []
                        for g in cutlass.range_constexpr(num_gemms):
                            subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                            tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                            _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w, offset=ld_half_off)
                            # INT8 int32 accumulate → widen to fp32 (skipped for int32 output).
                            if cutlass.const_expr(acc_widen_to_fp32):
                                _accf = _cv.to(cutlass.Float32)
                                # `+ 0.0` forces a fresh fp32 register so int32->fp32 isn't folded into an invalid int32->fp8 cast.
                                _cv = _accf + cutlass.full_like(_accf, 0.0)
                            c_rmem_vecs.append(_cv)
                        c_rmem_vec = c_rmem_vecs[0]

                    # Exactly one acc_empty arrive per epilogue warp per tile —
                    # the mbar counts warps, not M blocks.
                    if (not use_tma_store_epi) and mi == num_mma_m - 1 and subtile_idx == subtile_cnt - 1:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if elect_one:
                            nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                    col = coord_n + subtile_col_offset

                    # @@TMA_STORE_ONLY:BEGIN@@
                    epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                    smem_subtile_ptr = smem_d_ptr.subview(epi_stage_idx * epi_subtile_elems)
                    smem_thr_ptr = smem_subtile_ptr.subview(tidx * subtile_w)

                    if cutlass.const_expr(cd_out_is_m_major):
                        ld_col = mi_col_base + subtile_col_offset
                        for _h in cutlass.range(2, unroll_full=True):
                            ld_row = base_row_id + warp_idx * 32 + _h * 16
                            ld_addr = (ld_row << 16) | ld_col
                            ld_tmem = cutlass.inttoptr(ld_addr, 6, mma_c_dtype)
                            _lv = nvvm.tcgen05_ld(nvvm.Tcgen05LdStShape.SHAPE_16X256B, ld_tmem, num=4)
                            if cutlass.const_expr(acc_widen_to_fp32):
                                _accf = _lv.to(cutlass.Float32)
                                _lv = _accf + cutlass.full_like(_accf, 0.0)
                            vec_f32 = _lv
                            col_j = col
                            linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                            # @@INJECT_EPILOGUE@@

                            _i32 = vec_out.bitcast(cutlass.Int32)
                            for _blk in cutlass.range_constexpr(2):
                                _regs = [_i32[_blk * 4 + _j] for _j in range(4)]
                                _n_full = (lane % 8) + 8 * (lane // 16) + 16 * _blk
                                _m_base = warp_idx * 32 + _h * 16 + 8 * ((lane // 8) % 2)
                                _stm_off = (
                                    (_m_base // cd_mmajor_atom_m) * (cd_mmajor_atom_m * epi_tile_mn[1])
                                    + (_m_base % cd_mmajor_atom_m)
                                    + _n_full * cd_mmajor_atom_m
                                )
                                nvvm.stmatrix(
                                    _apply_smem_swizzle(
                                        smem_subtile_ptr.data_ptr() + _stm_off,
                                        cutlass.Swizzle(3, 4, 3),
                                    ),
                                    _regs,
                                    nvvm.MMALayout.COL,
                                    shape=nvvm.StoreShape.M8N8,
                                )
                    else:
                        vec_f32 = c_rmem_vec
                        col_j = col
                        linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                        # @@INJECT_EPILOGUE@@

                        smem_thr_ptr.data_ptr().store_swizzled(vec_out, alignment=64, swizzle=cutlass.Swizzle(2, 4, 3))

                    cute.arch.fence_view_async_shared()
                    nvvm.barrier_cta_sync(
                        barrier_id=EPI_SYNC_BAR_ID,
                        thread_count=num_epilogue_warps * 32,
                    )

                    if warp_idx == 0:
                        if cutlass.const_expr(cd_out_is_m_major):
                            for _mb in cutlass.range_constexpr(epi_tile_mn[0] // cd_mmajor_atom_m):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_global_shared_cta(
                                        tma_c_desc.get_ptr(),
                                        smem_subtile_ptr.subview(_mb * (cd_mmajor_atom_m * epi_tile_mn[1])),
                                        (coord_m + _mb * cd_mmajor_atom_m, col, tile_l),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_global_shared_cta(
                                    tma_c_desc.get_ptr(),
                                    smem_subtile_ptr,
                                    (col, coord_m, tile_l),
                                )
                        if elect_one:
                            nvvm.cp_async_bulk_commit_group()
                        nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)

                    nvvm.barrier_cta_sync(
                        barrier_id=EPI_SYNC_BAR_ID,
                        thread_count=num_epilogue_warps * 32,
                    )
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

            if cutlass.const_expr(use_tma_store_epi):
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if elect_one:
                    nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

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
            tile_m, tile_n = _l2_swizzle_tile(
                m_idx // cluster_m,
                n_idx // cluster_n,
                gridx // cluster_m,
                gridy // cluster_n,
                swizzle_w,
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
    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    batch = problem_size[3]
    _stride_idx = 4
    _a_stride_sets = []
    for _ in cutlass.range_constexpr(num_a_operands):
        _a_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    _b_stride_sets = []
    for _ in cutlass.range_constexpr(num_b_operands):
        _b_stride_sets.append(
            (
                problem_size[_stride_idx],
                problem_size[_stride_idx + 1],
                problem_size[_stride_idx + 2],
            )
        )
        _stride_idx += 3
    # @@INJECT_HOST_REDUCTION_STRIDES@@

    if cutlass.const_expr(matmul_a_batch == 1):
        a_batch = 1
    else:
        a_batch = batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_batch = 1
    else:
        b_batch = batch

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[m, k_sym, a_batch],
                    global_strides=[
                        a_stride_k * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, m, a_batch],
                    global_strides=[
                        a_stride_m * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[n, k_sym, b_batch],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_dtype,
                    global_dims=[k_sym, n, b_batch],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1], 1],
                    swizzle=ab_tma_swizzle,
                )
            )

    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_HOST_TMA_C_LISTS@@
    c = _tma_c_outputs[0]
    if cutlass.const_expr(cd_out_is_m_major):
        tma_c_desc = _tma.create_tensor_map_tiled(
            global_address=c.iterator.toint(),
            dtype=cd_tma_dtype,
            global_dims=[m, n, batch],
            global_strides=[
                out_stride_n_0 * cd_dtype.width // 128,
                out_stride_l_0 * cd_dtype.width // 128,
            ],
            box_dims=[cd_mmajor_atom_m, epi_tile_mn[1], 1],
            swizzle=(_tma.TensorMapSwizzle.s128b if cutlass.const_expr(use_tma_store_epi) else _tma.TensorMapSwizzle.none),
        )
    else:
        tma_c_desc = _tma.create_tensor_map_tiled(
            global_address=c.iterator.toint(),
            dtype=cd_tma_dtype,
            global_dims=[n, m, batch],
            global_strides=[
                out_stride_m_0 * cd_dtype.width // 128,
                out_stride_l_0 * cd_dtype.width // 128,
            ],
            box_dims=[epi_tile_mn[1], epi_tile_mn[0], 1],
            swizzle=(_tma.TensorMapSwizzle.s64b if cutlass.const_expr(use_tma_store_epi) else _tma.TensorMapSwizzle.none),
        )
    tma_c_desc_list = [tma_c_desc]
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
    ab_stride_elems = 16 // (ab_dtype.width // 8)
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
            mma_a_dtype,
            (sym_m, sym_k, sym_a_l),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            mma_b_dtype,
            (sym_n, sym_k, sym_b_l),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # @@TMA_STORE_ONLY:BEGIN@@
    def _make_fake_c():
        return make_fake_compact_tensor(
            cd_dtype,
            (sym_m, sym_n // cd_fake_n_div, sym_l),
            stride_order=(0, 1, 2) if cd_out_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    # @@INJECT_COMPILE_TMA_C_FAKES@@
    # @@TMA_STORE_ONLY:END@@
    def _sym_operand_strides(is_mn_major: bool) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=ab_stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major))
    # @@INJECT_COMPILE_REDUCTION_STRIDE_DECLS@@
    # @@INJECT_COMPILE_AB_FAKES@@
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
