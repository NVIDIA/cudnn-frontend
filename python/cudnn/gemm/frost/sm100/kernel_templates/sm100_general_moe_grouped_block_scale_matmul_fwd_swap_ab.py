# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# SM100-family MoE NONE/SCATTER/COMBINE; GATHER uses its dedicated template.

"""sm100 general MoE grouped block-scale matmul fwd (weight-by-token): linear scales of any dtype and block shape.

The grouped pipeline of sm100_moe_grouped_block_scale_matmul_fwd_swap_ab.py, with the scales applied as sm100_general_block_scale_matmul.py applies them:
the plain ``tcgen05.mma`` on the raw operands goes into a PARTIAL TMEM accumulator, one promotion window (the
K extent over which every scale is constant) at a time, and two warpgroups of promotion warps fold
``partial * token_scale * weight_scale`` into the tile's running sum, which reaches the MAIN TMEM accumulator
that the epilogue drains. The token and weight loads, the group scheduler and the epilogue are the source
template's. A promotion thread owns one TMEM lane and its warpgroup's half of the accumulator columns; the
running sum stays in its registers when that share fits (``promo_in_registers``) and in the main accumulator
otherwise."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.kernel_templates.dynamic_scheduler_counter_initialization import (
    dynamic_scheduler_counter_initialization as _dynamic_scheduler_counter_initialization,
)
from cudnn.gemm.frost.tile_helpers import (
    moe_scatter_row,
    moe_combine_add,
    moe_combine_swap_bulk,
    tma_scatter4,
    copy_tensormap_to_workspace as _copy_tensormap_to_workspace,
    epi_subtile_spans as _epi_subtile_spans,
    fence_tensormap_acquire as _fence_tensormap_acquire,
    fence_tensormap_release as _fence_tensormap_release,
    moe_swizzle_tile as _moe_swizzle_tile,
    replace_tensormap_global_address as _replace_tensormap_global_address,
    replace_tensormap_global_dim_0 as _replace_tensormap_global_dim_0,
    replace_tensormap_global_dim_1 as _replace_tensormap_global_dim_1,
    replace_tensormap_global_dim_2 as _replace_tensormap_global_dim_2,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
    TENSOR_MAP_QWORDS,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
from cudnn.frost.compiled_cache import compile_cached as _compile_cached
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda

# @@INJECT_TILE_CONSTANTS@@

moe_desc_slots = num_b_operands + n_tma_outputs
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
# The scheduler warp of cluster rank 0 publishes each claimed tile into every
# cluster CTA's broadcast slot through mapa and arrives on that CTA's barrier,
# a cluster-scope release by a thread of another CTA. The waits on the
# broadcast ring poll that barrier at cluster scope with relaxed ordering and
# then acquire through a fence restricted to shared::cluster memory. That
# orders the slot read (and the producer's reuse of the slot) after the peer's
# access just like an acquire-ordered cluster-scope wait, without the L1
# invalidation such a wait costs on every poll.
_SCHED_BCAST_CLUSTER = cluster_shape_mnk[0] * cluster_shape_mnk[1] * cluster_shape_mnk[2] > 1
_SCHED_BCAST_SCOPE = nvvm.MBarrierScope.CLUSTER if _SCHED_BCAST_CLUSTER else None
_SCHED_BCAST_ORDER = nvvm.MemOrder.RELAXED if _SCHED_BCAST_CLUSTER else None

if use_acc_overlap and any(_w != epi_n for _, _w in _epi_subtile_spans(epi_cols_per_mma_m, epi_n)):
    raise NotImplementedError(f"{__name__}: acc overlap reverses subtiles by index, which needs a uniform drain width")


# Per-CTA scheduler ring (replaces CLC): 2 stages, 8 int32 slot words.
SCHED_STAGES = 2
SCHED_BCAST_STAGES = 2
SCHED_SLOT_WORDS = 8

USE_PDL = True
EPI_SMEM_STAGES = 2
EPI_SYNC_BAR_ID = 1
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _moe_auto_swizzle_w(m, group_tokens, k, nt_n):
    """N-super-block width for one routed group."""
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_max_data_bits) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(m)
    if cutlass.min(rows, group_tokens) * row_bytes <= budget and rows <= group_tokens:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _a_collector_op(g):
    if cutlass.const_expr(num_gemms == 1 or num_a_operands != 1 or mma_size_m != 1):
        return None
    if cutlass.const_expr(g == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(g == num_gemms - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


def _b_collector_op(mi):
    if cutlass.const_expr(not b_collector_ok or mma_size_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == mma_size_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


# fma.rn.f32x2 (sm_100+): two FMAs per instruction. Older DSL wheels lack the wrapper; the promotion then
# multiplies and adds element by element.
_fma_f32x2 = getattr(nvvm, "fma_packed_f32x2", None)


def _promo_fma_pair(scale0, scale1, part0, part1, sum0, sum1):
    """``(scale0 * part0 + sum0, scale1 * part1 + sum1)`` for two neighbouring columns of one lane."""
    if cutlass.const_expr(_fma_f32x2 is not None):
        return _fma_f32x2((scale0, scale1), (part0, part1), (sum0, sum1), rnd=nvvm.FPRoundingMode.RN)
    return scale0 * part0 + sum0, scale1 * part1 + sum1


def _load_sfa(ptr, stride_m, stride_k, stride_l, l_idx, row, k_idx, m, k):
    """A's raw scale for ``row`` at K index ``k_idx``. An index past the extent reads the last block, so the
    load needs no bounds branch; the promotion discards what was read for such a row or window."""
    blk_m = cutlass.min(cutlass.Int64(row), m - 1) // sfa_block_m
    blk_k = cutlass.min(cutlass.Int64(k_idx), k - 1) // sfa_block_k
    return (ptr + blk_m * stride_m + blk_k * stride_k + l_idx * stride_l).load()


def _load_sfb(ptr, stride_k, stride_n, stride_l, l_idx, col, k_idx, n, k):
    """B's raw scale for ``col`` at K index ``k_idx``, bounded like :func:`_load_sfa`."""
    blk_k = cutlass.min(cutlass.Int64(k_idx), k - 1) // sfb_block_k
    blk_n = cutlass.min(cutlass.Int64(col), n - 1) // sfb_block_n
    return (ptr + blk_k * stride_k + blk_n * stride_n + l_idx * stride_l).load()


# A scale that is the same for every lane of a warp is loaded once and shuffled to the warp. Each lane loads one
# value of a (column run, window) grid, so one global load covers 32 // runs windows of this thread's column runs
# (or 32 runs of one window when it has more than 32 of them).
_sfb_shfl_cols = min(sfb_groups, 32)
_sfb_loads = -(-sfb_groups // 32)
_sfb_windows = max(1, 32 // sfb_groups)
# The A scale qualifies when a whole MMA row block lies in one of its blocks: the weight rows of a tile start at a multiple of an MMA row block.
_sfa_uniform = sfa_present and (sfa_block_m % (cta_tile_mnk[0] // mma_size_m) == 0)
_sfa_windows = max(1, 32 // promo_mi_count)
# A warp-uniform A scale of a thread with one MMA row block is folded into its B scales: a grid slot or a table
# entry holds the product of the two, so a column costs the thread no multiply.
_sfa_folded = _sfa_uniform and sfb_present and promo_mi_count == 1
# A thread with many column runs (promo_scale_table) reads their scales from a shared-memory table instead: the first
# warp of its warpgroup loads the window's runs (lane = run), and all four warps read what they fold.
_promo_tbl_runs = 32 * _sfb_loads


@cute.kernel
def _kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    first_token_offset: cute.Tensor,
    tma_workspace: cute.Tensor,
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    # @@INJECT_MOE_KERNEL_MA_PARAMS@@
    # @@INJECT_KERNEL_TAP_PARAMS@@
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_KERNEL_TMA_C_PARAMS@@
    # @@TMA_STORE_ONLY:END@@
) -> None:
    """Run the persistent grouped GEMM with promoted linear scales and the selected output epilogue."""
    # @@INJECT_AB_DESC_LISTS@@

    # @@INJECT_MOE_MA_LIST@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_TMA_C_LISTS@@
    # @@TMA_STORE_ONLY:END@@

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    # Warps 8.. promote the partial accumulators: warp_idx % 4 selects a warp's TMEM lane quarter, its warpgroup
    # the share of the accumulator columns.
    promo_warp_id_start = 8
    num_promo_warps = 4 * num_promo_groups
    prod_reg_count = 24

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    gridx = cute.arch.grid_dim()[0]

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m
    if cutlass.const_expr(cta_group == 2):
        pair_member = m_rank % cta_group
        pair_m_idx = m_rank // cta_group
        is_pair_leader = pair_member == 0
        pair_leader_rank = pair_m_idx * cta_group + n_rank * cluster_m
    else:
        pair_member = 0
        pair_m_idx = m_rank
        is_pair_leader = True
        pair_leader_rank = cta_rank_in_cluster

    if cutlass.const_expr(cta_group == 2):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
            mem_space=cute.AddressSpace.generic,
        )

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        # @@TMA_STORE_ONLY:BEGIN@@
        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())
        # @@TMA_STORE_ONLY:END@@

    if cutlass.const_expr(cta_group == 1):
        sched_counter_ptr = cute.make_ptr(
            cutlass.Int32,
            (tma_workspace.iterator.raw_ptr() + grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS).toint(),
            mem_space=cute.AddressSpace.generic,
        )

    a_pattern = 0
    for n_idx in cutlass.range_constexpr(cluster_n):
        a_pattern = a_pattern | (1 << (n_idx * cluster_m))
    if cutlass.const_expr(cta_group == 1):
        b_pattern = (1 << cluster_m) - 1
    else:
        b_pattern = 0
        for pm_idx in cutlass.range_constexpr(cluster_m // 2):
            b_pattern = b_pattern | (1 << (pm_idx * 2))

    if cutlass.const_expr(multicast_a):
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(a_pattern) << m_rank
        else:
            tma_mcast_mask_a = cutlass.Int16(a_pattern << m_rank)
    else:
        if cutlass.const_expr(cta_group == 1):
            tma_mcast_mask_a = cutlass.Int16(1) << cta_rank_in_cluster
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
    if cutlass.const_expr(cta_group == 1):
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        else:
            tma_mcast_mask_b = cutlass.Int16(1) << cta_rank_in_cluster

        a_part_arrive = cutlass.Int16(a_pattern) << m_rank
        b_part_arrive = cutlass.Int16(b_pattern) << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = a_part_arrive | b_part_arrive
    else:
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)
    if cutlass.const_expr(moe_combine_swap_bulk_enabled):
        combine_staging = cutlass.Array(cutlass.BFloat16, 128 * mma_size_m * epi_n, space=cutlass.AddressSpace.smem, alignment=16)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    partial_full_mbar_ptr = cutlass.Array(cutlass.Int64, partial_stages, space=cutlass.AddressSpace.smem)
    partial_empty_mbar_ptr = cutlass.Array(cutlass.Int64, partial_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(promo_scale_table):
        # One scale table and one "table written" barrier per (promotion warpgroup, partial stage).
        promo_tbl_full_mbar_ptr = cutlass.Array(cutlass.Int64, num_promo_groups * partial_stages, space=cutlass.AddressSpace.smem)
        promo_scale_tbl = cutlass.Array(cutlass.Float32, num_promo_groups * partial_stages * _promo_tbl_runs, space=cutlass.AddressSpace.smem, alignment=16)

    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    sched_storage = cutlass.Array(
        cutlass.Int32,
        SCHED_STAGES * SCHED_SLOT_WORDS,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sched_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_bcast_slot = cutlass.Array(cutlass.Int32, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    sched_bcast_full_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    sched_bcast_empty_mbar_ptr = cutlass.Array(cutlass.Int64, SCHED_BCAST_STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    tma_b_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_b_operands)
    ]

    # @@TMA_STORE_ONLY:BEGIN@@
    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    tma_c_desc_smem = cutlass.Array(
        cutlass.Int64,
        TENSOR_MAP_QWORDS * n_tma_outputs,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    # @@TMA_STORE_ONLY:END@@

    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
    smem_a_list = [
        cutlass.Array(
            a_smem_dtype,
            sA_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            b_smem_dtype,
            sB_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    if cutlass.const_expr(cta_group == 2):
        acc_empty_count = num_epilogue_warps * 2
    if cutlass.const_expr(ab_empty_full_mask):
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_size
        else:
            ab_empty_count = cluster_size // cta_group
    else:
        if cutlass.const_expr(cta_group == 1):
            ab_empty_count = cluster_m + cluster_n - 1
        else:
            ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    sched_empty_count = 1 + 1 + num_epilogue_warps + num_promo_warps
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
            else:
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        else:
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(acc_stages):
                if elect_one:
                    nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), num_promo_warps)
                if elect_one:
                    nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_epilogue_warps)
            if cutlass.const_expr(use_acc_overlap):
                if elect_one:
                    nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
        if cutlass.const_expr(cta_group == 2):
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(acc_stages):
                if elect_one:
                    nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), num_promo_warps)
                if elect_one:
                    nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(partial_stages):
            if elect_one:
                nvvm.mbarrier_init(partial_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(partial_empty_mbar_ptr.subview(i), num_promo_warps * cta_group)
        if cutlass.const_expr(promo_scale_table):
            for i in range(num_promo_groups * partial_stages):
                if elect_one:
                    nvvm.mbarrier_init(promo_tbl_full_mbar_ptr.subview(i), 1)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), sched_empty_count)
        for i in range(SCHED_BCAST_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_bcast_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_bcast_empty_mbar_ptr.subview(i), cluster_size)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (a_smem_dtype.width // 8)
    sB_bytes = sB_elems * (b_smem_dtype.width // 8)
    # The pair leader issues ONE expect_tx for both CTAs, so it counts twice.
    ab_only_copy_bytes = (num_a_operands * sA_tma_bytes + num_b_operands * sB_tma_bytes) * cta_group
    if cutlass.const_expr(cta_group == 2):
        pair_n_size = cgrp_tile_mnk[1] // cluster_n
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    tmem_alloc_bar_count = (num_epilogue_warps + num_promo_warps + 1) * 32
    if cutlass.const_expr(cta_group == 2):

        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    # @@INJECT_TAP_PTRS@@

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_m = cute.ceil_div(cutlass.Int32(M), cgrp_tile_mnk[0])
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(first_token_arr[num_groups])
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        bcast_stage = cutlass.Int32(0)
        bcast_full_phase = cutlass.Int32(0)
        bcast_empty_phase = cutlass.Int32(1)
        last_bcast_stage = cutlass.Int32(0)
        last_bcast_empty_done_phase = cutlass.Int32(0)
        linear_idx = cutlass.Int32(0)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        start_sf_block_n = cutlass.Int32(0)
        total_sf_blocks_n = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)
        cached_next_end = cutlass.Int32(0)
        if lane < num_groups:
            cached_next_end = cutlass.Int32(first_token_arr[lane + 1])
        else:
            cached_next_end = gemm_s
        tile_lower_bound = nvvm.shfl_sync(full_warp_mask, cached_next_end, 1, shfl_up_clamp, nvvm.Shfl.UP)
        cached_next_begin = cutlass.Int32(0)
        if lane != 0:
            cached_next_begin = tile_lower_bound

        while is_tile_valid != 0:
            if cta_rank_in_cluster == 0:
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_empty_mbar_ptr.subview(bcast_stage),
                    bcast_empty_phase,
                    time_limit=10_000_000,
                    scope=_SCHED_BCAST_SCOPE,
                    order=_SCHED_BCAST_ORDER,
                ):
                    pass
                if cutlass.const_expr(_SCHED_BCAST_CLUSTER):
                    nvvm.fence_sync_restrict(nvvm.MemOrder.ACQUIRE)
                claimed = cutlass.Int32(0)
                if lane == 0:
                    claimed = nvvm.atomicrmw(
                        "add",
                        sched_counter_ptr,
                        cutlass.Int32(1),
                        mem_order="relaxed",
                        syncscope="gpu",
                    )
                claimed = nvvm.shfl_sync(full_warp_mask, claimed, 0, shfl_idx_clamp, nvvm.Shfl.IDX)
                if lane < cluster_size:
                    (nvvm.mapa(sched_bcast_slot.subview(bcast_stage), lane)).store(claimed)
                    nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_full_mbar_ptr.subview(bcast_stage), lane))
            while not nvvm.mbarrier_try_wait_parity(
                sched_bcast_full_mbar_ptr.subview(bcast_stage),
                bcast_full_phase,
                time_limit=10_000_000,
                scope=_SCHED_BCAST_SCOPE,
                order=_SCHED_BCAST_ORDER,
            ):
                pass
            if cutlass.const_expr(_SCHED_BCAST_CLUSTER):
                nvvm.fence_sync_restrict(nvvm.MemOrder.ACQUIRE)
            linear_idx = (sched_bcast_slot.subview(bcast_stage)).load()
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            # Release orders the mailbox reads before the producer may reuse it.
            if lane == 0:
                nvvm.mbarrier_arrive(nvvm.mapa(sched_bcast_empty_mbar_ptr.subview(bcast_stage), 0))
            if cutlass.const_expr(cluster_size > 1):
                last_bcast_stage = bcast_stage
                last_bcast_empty_done_phase = bcast_empty_phase ^ 1
            bcast_stage += 1
            if bcast_stage == SCHED_BCAST_STAGES:
                bcast_stage = cutlass.Int32(0)
                bcast_full_phase = bcast_full_phase ^ 1
                bcast_empty_phase = bcast_empty_phase ^ 1

            group_begin = cached_next_begin
            group_end = cached_next_end

            if linear_idx >= start_linear_idx + total_tiles:
                group_idx += lane
                is_search_live = cutlass.Int32(1)
                while is_search_live != 0:
                    cached_group_begin = cached_next_begin
                    cached_group_end = cached_next_end
                    tile_start_idx = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        31,
                        shfl_idx_clamp,
                        nvvm.Shfl.IDX,
                    )
                    next_end_group = group_idx + 32 + 1
                    if next_end_group <= num_groups:
                        cached_next_end = cutlass.Int32(first_token_arr[next_end_group])
                    else:
                        cached_next_end = gemm_s
                    tile_lower_bound = nvvm.shfl_sync(
                        full_warp_mask,
                        cached_next_end,
                        1,
                        shfl_up_clamp,
                        nvvm.Shfl.UP,
                    )
                    if lane != 0:
                        cached_next_begin = tile_lower_bound
                    else:
                        cached_next_begin = tile_start_idx

                    group_tokens = cached_group_end - cached_group_begin
                    total_tiles = cute.ceil_div(group_tokens, cgrp_tile_mnk[1]) * clusters_along_m
                    total_sf_blocks_n = cute.ceil_div(group_tokens, 128)
                    prefix_tiles = total_tiles
                    prefix_sf = total_sf_blocks_n
                    for delta in (1, 2, 4, 8, 16):
                        prefix_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_tiles,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        prefix_sf_delta = nvvm.shfl_sync(
                            full_warp_mask,
                            prefix_sf,
                            delta,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane >= delta:
                            prefix_tiles += prefix_delta
                            prefix_sf += prefix_sf_delta
                    start_linear_idx += prefix_tiles - total_tiles
                    start_sf_block_n += prefix_sf - total_sf_blocks_n
                    thread_succeed = nvvm.vote_sync(
                        full_warp_mask,
                        linear_idx < start_linear_idx + total_tiles,
                        nvvm.VoteSync.BALLOT,
                    )
                    if thread_succeed != 0:
                        winning_lane = cutlass.Int32(31) - cute.arch.bfind(cute.arch.brev(thread_succeed)).to(cutlass.Int32)
                        group_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_linear_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            start_linear_idx,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        total_tiles = nvvm.shfl_sync(
                            full_warp_mask,
                            total_tiles,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        start_sf_block_n = nvvm.shfl_sync(
                            full_warp_mask,
                            start_sf_block_n,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        tile_start_idx = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_group_begin,
                            winning_lane,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        group_end_idx = group_idx + lane + 1
                        if group_end_idx <= num_groups:
                            cached_next_end = cutlass.Int32(first_token_arr[group_end_idx])
                        else:
                            cached_next_end = gemm_s
                        tile_lower_bound = nvvm.shfl_sync(
                            full_warp_mask,
                            cached_next_end,
                            1,
                            shfl_up_clamp,
                            nvvm.Shfl.UP,
                        )
                        if lane != 0:
                            cached_next_begin = tile_lower_bound
                        else:
                            cached_next_begin = tile_start_idx
                        group_begin = cached_next_begin
                        group_end = cached_next_end
                        is_search_live = cutlass.Int32(0)
                    else:
                        group_idx += 32
                        first_lane_group = nvvm.shfl_sync(
                            full_warp_mask,
                            group_idx,
                            0,
                            shfl_idx_clamp,
                            nvvm.Shfl.IDX,
                        )
                        if first_lane_group >= num_groups:
                            is_tile_valid = cutlass.Int32(0)
                            is_search_live = cutlass.Int32(0)
                        else:
                            next_start_linear_idx = start_linear_idx + total_tiles
                            start_linear_idx = nvvm.shfl_sync(
                                full_warp_mask,
                                next_start_linear_idx,
                                31,
                                shfl_idx_clamp,
                                nvvm.Shfl.IDX,
                            )
                            next_start_sf = start_sf_block_n + total_sf_blocks_n
                            start_sf_block_n = nvvm.shfl_sync(
                                full_warp_mask,
                                next_start_sf,
                                31,
                                shfl_idx_clamp,
                                nvvm.Shfl.IDX,
                            )

            coord_expert = cutlass.Int32(0)
            cluster_tile_m = cutlass.Int32(0)
            coord_n = cutlass.Int32(0)
            if is_tile_valid != 0:
                local_linear_idx = linear_idx - start_linear_idx
                group_nt_n = total_tiles // clusters_along_m
                cluster_tile_m, coord_n = _moe_swizzle_tile(
                    local_linear_idx,
                    clusters_along_m,
                    group_nt_n,
                    _moe_auto_swizzle_w(M, group_nt_n * cgrp_tile_mnk[1], k, group_nt_n),
                )
                coord_expert = group_idx % num_experts

            while not nvvm.mbarrier_try_wait_parity(
                sched_empty_mbar_ptr.subview(sched_stage),
                sched_empty_phase,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                (slot.subview(0)).store(coord_expert)
                (slot.subview(1)).store(cluster_tile_m)
                (slot.subview(2)).store(coord_n)
                (slot.subview(3)).store(is_tile_valid)
                (slot.subview(4)).store(group_begin)
                (slot.subview(5)).store(group_end)
                (slot.subview(6)).store(start_sf_block_n)
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

        if cutlass.const_expr(cluster_size > 1):
            if cta_rank_in_cluster == 0:
                while not nvvm.mbarrier_try_wait_parity(
                    sched_bcast_empty_mbar_ptr.subview(last_bcast_stage),
                    last_bcast_empty_done_phase,
                    time_limit=10_000_000,
                    scope=_SCHED_BCAST_SCOPE,
                    order=_SCHED_BCAST_ORDER,
                ):
                    pass
                if cutlass.const_expr(_SCHED_BCAST_CLUSTER):
                    nvvm.fence_sync_restrict(nvvm.MemOrder.ACQUIRE)

    if warp_idx == tma_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        if cutlass.const_expr(cta_group == 2):
            logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_b_operands)]
        b_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_b_operands)
        ]
        previous_group_begin = cutlass.Int32(-1)
        if cutlass.const_expr(moe_aligned_offsets or moe_combine_swap_bulk_enabled):
            b_desc_load_list = [tma_b_descs[_ai].get_ptr() for _ai in range(num_b_operands)]
        else:
            b_desc_load_list = b_desc_tma_ptr_list
        if elect_one and cutlass.const_expr(not moe_aligned_offsets and (not moe_combine_swap_bulk_enabled)):
            for _ai in cutlass.range_constexpr(num_b_operands):
                _copy_tensormap_to_workspace(tma_b_descs[_ai].get_ptr(), tma_b_desc_smem_list[_ai])
        nvvm.bar_warp_sync(0xFFFFFFFF)

        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            coord_expert = (slot.subview(0)).load()
            tile_m = (slot.subview(1)).load()
            tile_n = (slot.subview(2)).load()
            is_valid = (slot.subview(3)).load()
            group_begin = (slot.subview(4)).load()
            group_end = (slot.subview(5)).load()
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_weight = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_group = tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
                if cutlass.const_expr(moe_aligned_offsets or moe_combine_swap_bulk_enabled):
                    coord_n_desc = group_begin + coord_n_group
                else:
                    coord_n_desc = coord_n_group
                coord_n_per_cta = coord_n_desc + pair_member * cta_tile_mnk[1]

                if group_begin != previous_group_begin and cutlass.const_expr(not moe_aligned_offsets and (not moe_combine_swap_bulk_enabled)):
                    previous_group_begin = group_begin
                    for _ai in cutlass.range_constexpr(num_b_operands):
                        _fence_tensormap_acquire(b_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_b_operands):
                        if elect_one:
                            row_base = mB_list[_ai].iterator.raw_ptr().toint() + ((group_begin * b_stride_n_list[_ai] * b_dtype.width) >> 3)
                            _replace_tensormap_global_address(tma_b_desc_smem_list[_ai], row_base)
                            _replace_tensormap_global_dim_1(tma_b_desc_smem_list[_ai], group_end - group_begin)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (cta_desc_base_list[_ai] + lane).store((tma_b_desc_smem_list[_ai].subview(lane)).load())
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()

                for k_tile_idx in range(num_k_tiles):
                    stage = ab_iter % ab_stages
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_empty_mbar_ptr.subview(stage),
                        ab_empty_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2]
                    if cutlass.const_expr(cta_group == 1):
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                    else:

                        if is_pair_leader:
                            if elect_one:
                                nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                    a_issue = (not multicast_a) or (n_rank == 0)
                    if cutlass.const_expr(a_mcast_slices > 1):
                        a_data_issue = True
                        _a_off = n_rank * (cta_tile_mnk[0] // a_mcast_slices)
                    else:
                        a_data_issue = a_issue
                        _a_off = 0
                    b_issue = (not multicast_b) or (pair_m_idx == 0)
                    if cutlass.const_expr(b_mcast_slices > 1):
                        b_data_issue = True
                        _b_off = pair_m_idx * (cta_tile_mnk[1] // b_mcast_slices)
                    else:
                        b_data_issue = b_issue
                        _b_off = 0

                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (coord_m_weight + m_group * a_tma_group_elems, coord_k, coord_expert),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                            else:
                                for _am in cutlass.range_constexpr(cta_tile_mnk[0] // a_mcast_slices // a_tma_box_m):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(_a_off * a_packed_per_row + _am * a_tma_box_m * a_packed_per_row),
                                            tma_a_descs[_ai].get_ptr(),
                                            (coord_k, coord_m_weight + _a_off + _am * a_tma_box_m, coord_expert),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=_CTA_GROUP,
                                        )
                    if b_data_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_b_list[_bj].subview(sB_elems * stage + _b_off * b_packed_per_row),
                                    b_desc_load_list[_bj],
                                    (coord_k, coord_n_per_cta + _b_off, cutlass.Int32(0)),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=_CTA_GROUP,
                                )
                    ab_iter += 1

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

    if cutlass.const_expr(cta_group == 2):
        pair_mask = cutlass.Int16(3) << pair_leader_rank
        a_arrive_pattern = 0
        for n_idx in cutlass.range_constexpr(cluster_n):
            a_arrive_pattern = a_arrive_pattern | (1 << (n_idx * cluster_m))
        b_arrive_pattern = 0
        for m_idx in cutlass.range_constexpr(cluster_m):
            b_arrive_pattern = b_arrive_pattern | (1 << m_idx)
        a_part = a_arrive_pattern << m_rank
        a_part = a_part | (a_part << 1)
        b_part = b_arrive_pattern << (n_rank * cluster_m)
        if cutlass.const_expr(ab_empty_full_mask):
            ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
        else:
            ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if warp_idx == mma_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        _tcgen05_alloc(
            tmem_ptr_i32,
            cutlass.Int32(num_tmem_alloc_cols),
            is_exclusive=tmem_alloc_exclusive,
            group=_CTA_GROUP,
        )
        nvvm.bar_warp_sync(0xFFFFFFFF)
        nvvm.barrier_cta_arrive(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        if cutlass.const_expr(cta_group == 1):
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            sched_stage = cutlass.Int32(0)
            sched_full_phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)

            idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
                a_dtype=mma_a_dtype,
                b_dtype=mma_b_dtype,
                c_dtype=mma_acc_dtype,
                n_dim=mma_n_dim,
                m_dim=mma_m_dim,
                a_major=mma_a_major,
                b_major=mma_b_major,
            )
            desc_a_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_a_list[i],
                    leading_byte_offset=a_smem_desc_leading_byte_offset,
                    stride_byte_offset=a_smem_desc_stride_byte_offset,
                    layout=a_smem_swizzle,
                )
                for i in range(num_a_operands)
            ]
            desc_b_roots = [
                cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                    start_address=smem_b_list[j],
                    leading_byte_offset=b_smem_desc_leading_byte_offset,
                    stride_byte_offset=b_smem_desc_stride_byte_offset,
                    layout=b_smem_swizzle,
                )
                for j in range(num_b_operands)
            ]
            # The partial ring runs across tiles: a window takes the next buffer once the promotion warps free it.
            partial_iter = cutlass.Int32(0)
            partial_slot = cutlass.Int32(0)
            partial_empty_phase_bit = cutlass.Int32(1)

            while is_valid != 0:
                while not nvvm.mbarrier_try_wait_parity(
                    sched_full_mbar_ptr.subview(sched_stage),
                    sched_full_phase,
                    time_limit=10_000_000,
                ):
                    pass
                is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                # Finish every lane's slot reads before the elected release.
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                sched_stage += 1
                if sched_stage == SCHED_STAGES:
                    sched_stage = cutlass.Int32(0)
                    sched_full_phase = sched_full_phase ^ 1

                if is_valid != 0:
                    acc_stage = tile_iter % acc_stages
                    if acc_stage == 0 and tile_iter != 0:
                        acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr.subview(acc_stage),
                        acc_empty_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    last_inst = num_k_tiles * num_k_insts - 1
                    scale_d = cutlass.Boolean(False)
                    for k_tile_idx in range(num_k_tiles):
                        stage = ab_iter % ab_stages
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1

                        desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                        desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]

                        while not nvvm.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass

                        for mma_k in cutlass.range_constexpr(num_k_insts):
                            inst = k_tile_idx * num_k_insts + mma_k
                            # A window opens at each multiple of insts_per_window.
                            if inst % insts_per_window == 0:
                                partial_slot = partial_iter % partial_stages
                                if partial_slot == 0 and partial_iter != 0:
                                    partial_empty_phase_bit = partial_empty_phase_bit ^ 1
                                while not nvvm.mbarrier_try_wait_parity(
                                    partial_empty_mbar_ptr.subview(partial_slot),
                                    partial_empty_phase_bit,
                                    time_limit=10_000_000,
                                ):
                                    pass
                                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                                scale_d = cutlass.Boolean(False)
                            partial_col = base_col_id_root + partial_col_base + partial_slot * acc_region_cols
                            desc_a_k = desc_a_bases[gemm_a_idx[0]].advance_start_address(a_smem_k_step_bytes * mma_k)
                            desc_b = desc_b_bases[gemm_b_idx[0]].advance_start_address(b_smem_k_step_bytes * mma_k)
                            for mma_m in cutlass.range_constexpr(mma_size_m):
                                desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                if elect_one:
                                    _tcgen05_mma(
                                        mma_kind,
                                        _CTA_GROUP,
                                        nvvm.make_tmem_ptr((base_row_id << 16) | (partial_col + mma_m * epi_cols_per_mma_m), mma_acc_dtype),
                                        desc_a,
                                        desc_b,
                                        idesc,
                                        scale_d,
                                        collector_op=_a_collector_op(0),
                                        b_collector_op=_b_collector_op(mma_m),
                                    )
                            scale_d = cutlass.Boolean(True)
                            # The window closes before the next one opens or at the tile's last instruction.
                            if ((inst + 1) % insts_per_window == 0) | (inst == last_inst):
                                if elect_one:
                                    nvvm.tcgen05_commit(
                                        partial_full_mbar_ptr.subview(partial_slot),
                                        group=_CTA_GROUP,
                                    )
                                partial_iter += 1

                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1

                    # The promotion warps arrive on acc_full once the tile's last window is folded in.
                    tile_iter += 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if tile_iter != 0:
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
                if cutlass.const_expr(use_acc_overlap):
                    while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                        pass

            nvvm.bar_warp_sync(0xFFFFFFFF)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            peer_cta_rank = cta_rank_in_cluster ^ 1
            if is_pair_leader:
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                idesc = cutlass.experimental.primitives.Tcgen05InstrDesc.build(
                    a_dtype=mma_a_dtype,
                    b_dtype=mma_b_dtype,
                    c_dtype=mma_acc_dtype,
                    n_dim=mma_n_dim,
                    m_dim=mma_m_dim,
                    a_major=mma_a_major,
                    b_major=mma_b_major,
                )
                desc_a_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_a_list[i],
                        leading_byte_offset=a_smem_desc_leading_byte_offset,
                        stride_byte_offset=a_smem_desc_stride_byte_offset,
                        layout=a_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                        start_address=smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        layout=b_smem_swizzle,
                    )
                    for j in range(num_b_operands)
                ]
                # The partial ring runs across tiles: a window takes the next buffer once the promotion warps free it.
                partial_iter = cutlass.Int32(0)
                partial_slot = cutlass.Int32(0)
                partial_empty_phase_bit = cutlass.Int32(1)

                while is_valid != 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                    # Finish every lane's slot reads before the elected release.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

                    if is_valid != 0:
                        acc_stage = tile_iter % acc_stages
                        if acc_stage == 0 and tile_iter != 0:
                            acc_empty_phase_bit = acc_empty_phase_bit ^ 1

                        while not nvvm.mbarrier_try_wait_parity(
                            acc_empty_mbar_ptr.subview(acc_stage),
                            acc_empty_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass

                        last_inst = num_k_tiles * num_k_insts - 1
                        scale_d = cutlass.Boolean(False)
                        for k_tile_idx in range(num_k_tiles):
                            stage = ab_iter % ab_stages
                            if stage == 0 and ab_iter != 0:
                                ab_full_phase_bit = ab_full_phase_bit ^ 1

                            desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                            desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]

                            while not nvvm.mbarrier_try_wait_parity(
                                ab_full_mbar_ptr.subview(stage),
                                ab_full_phase_bit,
                                time_limit=10_000_000,
                            ):
                                pass

                            for mma_k in cutlass.range_constexpr(num_k_insts):
                                inst = k_tile_idx * num_k_insts + mma_k
                                # A window opens at each multiple of insts_per_window.
                                if inst % insts_per_window == 0:
                                    partial_slot = partial_iter % partial_stages
                                    if partial_slot == 0 and partial_iter != 0:
                                        partial_empty_phase_bit = partial_empty_phase_bit ^ 1
                                    while not nvvm.mbarrier_try_wait_parity(
                                        partial_empty_mbar_ptr.subview(partial_slot),
                                        partial_empty_phase_bit,
                                        time_limit=10_000_000,
                                    ):
                                        pass
                                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                                    scale_d = cutlass.Boolean(False)
                                partial_col = base_col_id_root + partial_col_base + partial_slot * acc_region_cols
                                desc_a_k = desc_a_bases[gemm_a_idx[0]].advance_start_address(a_smem_k_step_bytes * mma_k)
                                desc_b = desc_b_bases[gemm_b_idx[0]].advance_start_address(b_smem_k_step_bytes * mma_k)
                                for mma_m in cutlass.range_constexpr(mma_size_m):
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                    if elect_one:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            nvvm.make_tmem_ptr((base_row_id << 16) | (partial_col + mma_m * epi_cols_per_mma_m), mma_acc_dtype),
                                            desc_a,
                                            desc_b,
                                            idesc,
                                            scale_d,
                                            collector_op=_a_collector_op(0),
                                            b_collector_op=_b_collector_op(mma_m),
                                        )
                                scale_d = cutlass.Boolean(True)
                                # The window closes before the next one opens or at the tile's last instruction.
                                if ((inst + 1) % insts_per_window == 0) | (inst == last_inst):
                                    if elect_one:
                                        nvvm.tcgen05_commit(
                                            partial_full_mbar_ptr.subview(partial_slot),
                                            multicast_mask=pair_mask,
                                            group=_CTA_GROUP,
                                        )
                                    partial_iter += 1

                            if elect_one:
                                nvvm.tcgen05_commit(
                                    ab_empty_mbar_ptr.subview(stage),
                                    multicast_mask=ab_empty_arrive_mask,
                                    group=_CTA_GROUP,
                                )
                            ab_iter += 1

                        # The promotion warps arrive on acc_full once the tile's last window is folded in.
                        tile_iter += 1

                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")

                if tile_iter != 0:
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
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                if cutlass.const_expr(not use_acc_overlap):
                    nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )
            else:
                is_valid = cutlass.Int32(1)
                sched_stage = cutlass.Int32(0)
                sched_full_phase = cutlass.Int32(0)
                while is_valid != 0:
                    while not nvvm.mbarrier_try_wait_parity(
                        sched_full_mbar_ptr.subview(sched_stage),
                        sched_full_phase,
                        time_limit=10_000_000,
                    ):
                        pass
                    is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
                    # Finish every lane's slot reads before the elected release.
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                    sched_stage += 1
                    if sched_stage == SCHED_STAGES:
                        sched_stage = cutlass.Int32(0)
                        sched_full_phase = sched_full_phase ^ 1

                if cutlass.const_expr(USE_PDL):
                    nvvm.griddepcontrol("launch_dependents")

                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                if cutlass.const_expr(not use_acc_overlap):
                    nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
                _tcgen05_dealloc(
                    alloc_ptr,
                    cutlass.Int32(num_tmem_alloc_cols),
                    is_exclusive=tmem_alloc_exclusive,
                    group=_CTA_GROUP,
                )

    if warp_idx < num_epilogue_warps:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(epi_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        tile_iter = cutlass.Int32(0)
        acc_full_phase_bit = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        # Routed rows the groups cover (the final offset, at most the capacity N): the count an
        # ungrouped AVG over tokens divides by. At least 1, so an empty routing flushes 0, not NaN.
        moe_used_rows = cutlass.max(cutlass.min(cutlass.Int64(first_token_arr[num_groups]), N), cutlass.Int64(1))

        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        tile_l = cutlass.Int32(0)
        epi_block_linear = bidx + bidy * gridx
        d_desc_base_list = [
            tma_workspace.iterator.raw_ptr() + (epi_block_linear * moe_desc_slots + num_b_operands + _di) * TENSOR_MAP_QWORDS for _di in range(n_tma_outputs)
        ]
        d_desc_ptr_list = [cute.make_ptr(cutlass.Int64, _b.toint(), mem_space=cute.AddressSpace.generic) for _b in d_desc_base_list]
        previous_group_end = cutlass.Int32(-1)
        if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets and not moe_scatter):
            for _di in cutlass.range_constexpr(n_tma_outputs):
                if elect_one:
                    _copy_tensormap_to_workspace(tma_c_descs[_di].get_ptr(), tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS))
            nvvm.bar_warp_sync(0xFFFFFFFF)
        # @@TMA_STORE_ONLY:END@@

        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            if cutlass.const_expr(cta_group == 1):
                _slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                tile_m = (_slot.subview(1)).load()
                tile_n = (_slot.subview(2)).load()
                is_valid = (_slot.subview(3)).load()
                group_begin = (_slot.subview(4)).load()
                group_end = (_slot.subview(5)).load()
                start_sf_block_n = (_slot.subview(6)).load()
                group_idx = (_slot.subview(7)).load()
            else:
                slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
                tile_m = (slot.subview(1)).load()
                tile_n = (slot.subview(2)).load()
                is_valid = (slot.subview(3)).load()
                group_begin = (slot.subview(4)).load()
                group_end = (slot.subview(5)).load()
                start_sf_block_n = (slot.subview(6)).load()
                group_idx = (slot.subview(7)).load()
            nvvm.bar_warp_sync(0xFFFFFFFF)
            sched_stage = cute.arch.make_warp_uniform(sched_stage)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_tile = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                # @@TMA_STORE_ONLY:BEGIN@@
                if warp_idx == 0 and cutlass.const_expr(not moe_aligned_offsets and not moe_scatter):
                    if group_end != previous_group_end:
                        previous_group_end = group_end
                        nvvm.cp_async_bulk_wait_group(0, read=True)
                        for _di in cutlass.range_constexpr(n_tma_outputs):
                            _scratch = tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS)
                            _fence_tensormap_acquire(d_desc_ptr_list[_di])
                            if elect_one:
                                if cutlass.const_expr(tma_c_m_major[_di]):
                                    _replace_tensormap_global_dim_1(_scratch, group_end)
                                else:
                                    _replace_tensormap_global_dim_0(_scratch, group_end)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            if lane < TENSOR_MAP_QWORDS:
                                (d_desc_base_list[_di] + lane).store((_scratch.subview(lane)).load())
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()
                # @@TMA_STORE_ONLY:END@@
                # @@EPILOGUE_DRAIN:BEGIN@@
                coord_n_c = group_begin + tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)

                acc_stage = tile_iter % acc_stages
                if acc_stage == 0 and tile_iter != 0:
                    acc_full_phase_bit = acc_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                    pass

                if cutlass.const_expr(use_acc_overlap):
                    acc_buf_parity = tile_iter % 2
                    acc_base_col = base_col_id_root + acc_buf_parity * acc_stage_stride
                else:
                    acc_buf_parity = cutlass.Int32(0)
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols

                if cutlass.const_expr(moe_combine_swap_bulk_enabled):
                    for combine_subtile in cutlass.range_constexpr(subtile_cnt):
                        combine_col = coord_n_c + combine_subtile * epi_n
                        combine_weights = []
                        for combine_batch in cutlass.range_constexpr(epi_n // 32):
                            combine_route = combine_col + combine_batch * 32 + lane
                            combine_weight = cutlass.Float32(0.0)
                            if combine_route < group_end:
                                combine_token = token_index[combine_route]
                                combine_slot = token_ks[combine_route]
                                if (combine_token >= 0) & (combine_token < top_k_scores.shape[0]) & (combine_slot >= 0) & (combine_slot < moe_top_k):
                                    combine_weight = top_k_scores[combine_token, combine_slot]
                            combine_weights.append(combine_weight)
                        for combine_mi in cutlass.range_constexpr(mma_size_m):
                            for combine_chunk in cutlass.range_constexpr(epi_n // 64):
                                combine_tmem_col = acc_base_col + combine_mi * epi_cols_per_mma_m + combine_subtile * epi_n + combine_chunk * 64
                                combine_tmem_addr = (row_id_with_warp_offset << 16) | combine_tmem_col
                                combine_tmem = cutlass.inttoptr(combine_tmem_addr, 6, mma_c_dtype)
                                combine_values = nvvm.tcgen05_ld(shape, combine_tmem, num=64)
                                if cutlass.const_expr(combine_mi == mma_size_m - 1 and combine_subtile == subtile_cnt - 1 and combine_chunk == epi_n // 64 - 1):
                                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                    if elect_one:
                                        nvvm.mbarrier_arrive(
                                            nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                            scope=nvvm.MemScope.CLUSTER,
                                            relaxed=True,
                                        )
                                for combine_j in cutlass.range_constexpr(64):
                                    combine_logical_j = combine_chunk * 64 + combine_j
                                    combine_score = cute.arch.shuffle_sync(combine_weights[combine_logical_j // 32], cutlass.Int32(combine_logical_j % 32))
                                    combine_value = combine_values[combine_j] * combine_score
                                    combine_offset = combine_logical_j * (128 * mma_size_m) + combine_mi * 128 + tidx
                                    combine_staging.data_ptr(combine_offset).store(combine_value.to(cutlass.BFloat16))
                        moe_combine_swap_bulk(
                            combine_staging.data_ptr(0),
                            gC_tap_0_ptr,
                            token_index,
                            token_ks,
                            top_k_scores,
                            coord_m_tile,
                            combine_col,
                            group_end,
                            M,
                            out_stride_m_0,
                            out_stride_n_0,
                            tidx,
                            moe_top_k,
                            epi_n,
                            128 * mma_size_m,
                        )
                else:
                    for mi in cutlass.range_constexpr(mma_size_m):
                        if cutlass.const_expr(use_acc_overlap and mma_size_m > 1):
                            _mi = mi + (1 - acc_buf_parity) * (mma_size_m - 1 - 2 * mi)
                        else:
                            _mi = mi
                        coord_m = coord_m_tile + _mi * epi_rows_per_mma_m
                        mi_col_base = acc_base_col + _mi * epi_cols_per_mma_m
                        tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * acc_gemm_stride) for g in range(num_gemms)]

                        row = coord_m + tidx
                        row_active = True

                        # @@INJECT_AUX_VIEWS@@

                        for subtile_idx in cutlass.range_constexpr(subtile_cnt):
                            if cutlass.const_expr(use_acc_overlap):
                                _sub = subtile_idx + (1 - acc_buf_parity) * (subtile_cnt - 1 - 2 * subtile_idx)
                                subtile_col_offset = _sub * epi_n
                                subtile_w = epi_n
                            else:
                                subtile_col_offset, subtile_w = epi_spans[subtile_idx]
                            c_rmem_vecs = []
                            for g in cutlass.range_constexpr(num_gemms):
                                subtile_tmem_addr = tmem_col_addr_gemms[g] + subtile_col_offset
                                tmem = cutlass.inttoptr(subtile_tmem_addr, 6, mma_c_dtype)
                                _cv = nvvm.tcgen05_ld(shape, tmem, num=subtile_w)
                                c_rmem_vecs.append(_cv)
                            c_rmem_vec = c_rmem_vecs[0]

                            if cutlass.const_expr(not use_acc_overlap):
                                if cutlass.const_expr(mi == mma_size_m - 1 and subtile_idx == subtile_cnt - 1):
                                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                    if elect_one:
                                        if cutlass.const_expr(cta_group == 2):
                                            nvvm.mbarrier_arrive(
                                                nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                                scope=nvvm.MemScope.CLUSTER,
                                                relaxed=True,
                                            )
                                        else:
                                            nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                            if use_acc_overlap and mi * subtile_cnt + subtile_idx == acc_overlap_subtiles - 1:
                                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                                if elect_one:
                                    if cutlass.const_expr(cta_group == 2):
                                        nvvm.mbarrier_arrive(
                                            nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                            scope=nvvm.MemScope.CLUSTER,
                                            relaxed=True,
                                        )
                                    else:
                                        nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                            col = coord_n_c + subtile_col_offset

                            # @@TMA_STORE_ONLY:BEGIN@@
                            vec_f32 = c_rmem_vec
                            col_j = col

                            # @@INJECT_EPILOGUE@@

                            # @@INJECT_TMA_STORE_SEQUENCE@@
                            # @@TMA_STORE_ONLY:END@@

                            # @@STG_ONLY:BEGIN@@
                            if row_active and row < M:
                                for j in cutlass.range_constexpr(subtile_w // vsize):
                                    col_j = col + j * vsize
                                    if col_j < group_end:
                                        vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                                        # @@INJECT_STG_VEC_BINDINGS@@

                                        # @@INJECT_EPILOGUE@@
                            # @@STG_ONLY:END@@

                # The M-major TMA path loads its accumulator inside the store loop, so its release cannot move up.
                # @@EPILOGUE_DRAIN:END@@
                tile_iter += 1

        if cutlass.const_expr(use_acc_overlap):
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if elect_one:
                nvvm.mbarrier_arrive(tmem_dealloc_mbar_ptr)

    if warp_idx >= promo_warp_id_start:
        nvvm.setmaxregister(promo_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        nvvm.barrier_cta_sync(barrier_id=TMEM_ALLOC_BARRIER_ID, thread_count=tmem_alloc_bar_count)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id_root = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16

        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        # This thread's TMEM lane, which is its row inside an MMA row block ...
        promo_warp_lane = tidx % 32
        promo_lane = (warp_idx % 4) * 32 + promo_warp_lane
        promo_row = (base_row_id + (warp_idx % 4) * 32) << 16
        promo_shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        # ... and its share of an accumulator's columns: promo_mi_count MMA row blocks from promo_mi0 on, and in
        # each of them promo_n_cols tile columns from promo_n0 on.
        promo_group = (warp_idx - promo_warp_id_start) // 4
        promo_col0 = promo_group * promo_cols
        promo_mi0 = promo_col0 // epi_cols_per_mma_m
        promo_n0 = promo_col0 % epi_cols_per_mma_m
        # The scale tensors arrive in their operands' kernel order, in blocks: A's is (m // block_m, k // block_k, l)
        # and B's is (n // block_n, k // block_k, l).
        if cutlass.const_expr(sfa_present):
            sfa_ptr = sfa_tensors[0].iterator
            sfa_stride_m = cutlass.Int64(sfa_tensors[0].stride[0])
            sfa_stride_k = cutlass.Int64(sfa_tensors[0].stride[1])
            sfa_stride_l = cutlass.Int64(sfa_tensors[0].stride[2])
        if cutlass.const_expr(sfb_present):
            sfb_ptr = sfb_tensors[0].iterator
            sfb_stride_n = cutlass.Int64(sfb_tensors[0].stride[0])
            sfb_stride_k = cutlass.Int64(sfb_tensors[0].stride[1])
            sfb_stride_l = cutlass.Int64(sfb_tensors[0].stride[2])
        if cutlass.const_expr(promo_in_registers):
            # The running sum of this thread's share of the tile, one fp32 per accumulator column.
            promo_sum = cutlass.Array(cutlass.Float32, promo_cols, space=cutlass.AddressSpace.rmem)
        # Scales in flight. A per-lane A scale is loaded a window early (its raw value waits here); a warp-uniform
        # scale is this lane's share of the warp's (run, window) grid, fp32 already.
        if cutlass.const_expr(sfa_present):
            if cutlass.const_expr(_sfa_uniform):
                if cutlass.const_expr(not _sfa_folded):
                    sfa_grid = cutlass.Array(cutlass.Float32, 1, space=cutlass.AddressSpace.rmem)
            else:
                sfa_next = cutlass.Array(sfa_dtype, promo_mi_count, space=cutlass.AddressSpace.rmem)
        if cutlass.const_expr(sfb_present):
            if cutlass.const_expr(not promo_scale_table):
                sfb_grid = cutlass.Array(cutlass.Float32, _sfb_loads, space=cutlass.AddressSpace.rmem)
            # The raw values of the next grid period (or window, for the table), loaded ahead of their use.
            sfb_ahead = cutlass.Array(sfb_dtype, _sfb_loads, space=cutlass.AddressSpace.rmem)
            if cutlass.const_expr(_sfa_folded):
                sfa_ahead = cutlass.Array(sfa_dtype, 1, space=cutlass.AddressSpace.rmem)
        # The column run a lane loads: itself for the table, its grid slot's otherwise.
        if cutlass.const_expr(promo_scale_table):
            promo_lane_run = promo_warp_lane
        else:
            promo_lane_run = promo_warp_lane % _sfb_shfl_cols
        if cutlass.const_expr(promo_scale_table):
            # The warpgroup's table (one per partial stage), and whether this warp writes it. The writers are
            # warps 11 and 14 (18 under GATHER): away from the MMA warp's (4) and each other's slot modulo four,
            # which measured 2..3% faster on B200 than the first warp of each group.
            promo_is_writer = (warp_idx % 4) == 3 - promo_group
            promo_tbl_full = promo_tbl_full_mbar_ptr.subview(promo_group * partial_stages)
            promo_tbl = promo_scale_tbl.subview(promo_group * partial_stages * _promo_tbl_runs)

        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        partial_iter = cutlass.Int32(0)
        partial_full_phase_bit = cutlass.Int32(0)
        last_window = cutlass.Int32((num_k_tiles * num_k_insts - 1) // insts_per_window)
        while is_valid != 0:
            while not nvvm.mbarrier_try_wait_parity(
                sched_full_mbar_ptr.subview(sched_stage),
                sched_full_phase,
                time_limit=10_000_000,
            ):
                pass
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            coord_expert = (slot.subview(0)).load()
            tile_m = (slot.subview(1)).load()
            tile_n = (slot.subview(2)).load()
            is_valid = (slot.subview(3)).load()
            group_begin = (slot.subview(4)).load()
            group_end = (slot.subview(5)).load()
            # Finish every lane's slot reads before the elected release.
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_tile = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_c = group_begin + tile_n * cgrp_tile_mnk[1] + n_rank * (cta_tile_mnk[1] * cta_group)
                if cutlass.const_expr(sfa_present):
                    # One weight scale matrix per expert, unless every expert shares one.
                    if cutlass.const_expr(sfa_batch_broadcast):
                        sfa_l = cutlass.Int64(0)
                    else:
                        sfa_l = cutlass.Int64(coord_expert)
                acc_stage = tile_iter % acc_stages
                acc_col = base_col_id_root + acc_stage * acc_region_cols + promo_col0
                if cutlass.const_expr(promo_in_registers):
                    for c in cutlass.range_constexpr(promo_cols):
                        promo_sum[c] = cutlass.Float32(0.0)
                if cutlass.const_expr(sfa_present and not _sfa_uniform):
                    # The first window's per-lane A scales.
                    for mi in cutlass.range_constexpr(promo_mi_count):
                        row = coord_m_tile + (promo_mi0 + mi) * epi_rows_per_mma_m + promo_lane
                        sfa_next[mi] = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, row, 0, M, k)
                if cutlass.const_expr(sfb_present):
                    # The first grid period's loads (the first window's, for the table).
                    if cutlass.const_expr(promo_scale_table):
                        prime_k = cutlass.Int64(0)
                    else:
                        prime_k = cutlass.Int64(promo_warp_lane // _sfb_shfl_cols) * (insts_per_window * mma_inst_k_elems)
                    if cutlass.const_expr(_sfa_folded):
                        prime_row = coord_m_tile + promo_mi0 * epi_rows_per_mma_m
                        sfa_ahead[0] = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, prime_row, prime_k, M, k)
                    for load in cutlass.range_constexpr(_sfb_loads):
                        prime_run = load * 32 + promo_lane_run
                        prime_col = coord_n_c + promo_n0 + prime_run * sfb_group_cols
                        sfb_ahead[load] = _load_sfb(sfb_ptr, sfb_stride_k, sfb_stride_n, sfb_stride_l, 0, prime_col, prime_k, N, k)
                for window in range(last_window + 1):
                    partial_slot = partial_iter % partial_stages
                    if partial_slot == 0 and partial_iter != 0:
                        partial_full_phase_bit = partial_full_phase_bit ^ 1
                    # Every scale is constant over the window, so its first K index addresses them all. A window
                    # past K (the zero-filled tail of the last K tile) holds an exact zero partial: scale it by zero
                    # instead of by a scale block that does not exist.
                    window_k = cutlass.Int64(window) * (insts_per_window * mma_inst_k_elems)
                    window_live = window_k < k

                    row_scales = []
                    if cutlass.const_expr(_sfa_folded):
                        # The B grid below carries this thread's A scale.
                        row_scales.append(None)
                    elif cutlass.const_expr(sfa_present and _sfa_uniform):
                        # This lane's slot of the warp's (row block, window) grid, reloaded every _sfa_windows windows.
                        if window % _sfa_windows == 0:
                            grid_mi = promo_warp_lane % promo_mi_count
                            grid_k = window_k + (promo_warp_lane // promo_mi_count) * (insts_per_window * mma_inst_k_elems)
                            grid_row = coord_m_tile + (promo_mi0 + grid_mi) * epi_rows_per_mma_m
                            grid_a_raw = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, grid_row, grid_k, M, k)
                            grid_a_scale = cutlass.Float32(0.0)
                            if (grid_row < M) & (grid_k < k):
                                grid_a_scale = grid_a_raw.to(cutlass.Float32)
                            sfa_grid[0] = grid_a_scale
                        for mi in cutlass.range_constexpr(promo_mi_count):
                            row_scales.append(nvvm.shfl_sync(0xFFFFFFFF, sfa_grid[0], (window % _sfa_windows) * promo_mi_count + mi, 0x1F, nvvm.Shfl.IDX))
                    else:
                        # Take this window's per-lane A scales, loaded during the previous one, and issue the next window's loads.
                        next_k = window_k + insts_per_window * mma_inst_k_elems
                        for mi in cutlass.range_constexpr(promo_mi_count):
                            row_scale = cutlass.Float32(1.0)
                            if cutlass.const_expr(sfa_present):
                                row = coord_m_tile + (promo_mi0 + mi) * epi_rows_per_mma_m + promo_lane
                                raw = sfa_next[mi]
                                sfa_next[mi] = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, row, next_k, M, k)
                                row_scale = cutlass.Float32(0.0)
                                if (row < M) & window_live:
                                    row_scale = raw.to(cutlass.Float32)
                            row_scales.append(row_scale)
                    if cutlass.const_expr(sfb_present and not promo_scale_table):
                        # This lane's slots of the warp's (column run, window) grid, a period of _sfb_windows windows at a time: the
                        # period's raw values were loaded during the previous one, and the next period's loads are issued here. A
                        # slot past the thread's runs, the operand or K holds zero, so no user of the grid checks bounds.
                        if window % _sfb_windows == 0:
                            grid_b_k = window_k + (promo_warp_lane // _sfb_shfl_cols) * (insts_per_window * mma_inst_k_elems)
                            grid_next_k = grid_b_k + _sfb_windows * (insts_per_window * mma_inst_k_elems)
                            if cutlass.const_expr(_sfa_folded):
                                # The A scale of the slot's window, the same for every row of this thread's MMA row block.
                                grid_frow = coord_m_tile + promo_mi0 * epi_rows_per_mma_m
                                grid_fa_scale = cutlass.Float32(0.0)
                                if (grid_frow < M) & (grid_b_k < k):
                                    grid_fa_scale = sfa_ahead[0].to(cutlass.Float32)
                                sfa_ahead[0] = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, grid_frow, grid_next_k, M, k)
                            for load in cutlass.range_constexpr(_sfb_loads):
                                grid_run = load * 32 + promo_lane_run
                                grid_col = coord_n_c + promo_n0 + grid_run * sfb_group_cols
                                grid_b_scale = cutlass.Float32(0.0)
                                if (grid_col < group_end) & (grid_b_k < k) & (grid_run < sfb_groups):
                                    grid_b_scale = sfb_ahead[load].to(cutlass.Float32)
                                if cutlass.const_expr(_sfa_folded):
                                    grid_b_scale = grid_b_scale * grid_fa_scale
                                sfb_grid[load] = grid_b_scale
                                sfb_ahead[load] = _load_sfb(sfb_ptr, sfb_stride_k, sfb_stride_n, sfb_stride_l, 0, grid_col, grid_next_k, N, k)
                    # scales[mi][run]: what this lane's partial of row block mi is multiplied by in column run ``run``. A thread
                    # whose registers hold all of its runs shuffles them out before it waits on the MMA; otherwise it takes a
                    # column pair's as it folds the pair.
                    grid_base = (window % _sfb_windows) * _sfb_shfl_cols
                    scales = []
                    for mi in cutlass.range_constexpr(promo_mi_count):
                        scales.append({})
                        if cutlass.const_expr(not sfb_present):
                            scales[mi][0] = row_scales[mi]
                        elif cutlass.const_expr(promo_scales_ahead):
                            for run in cutlass.range_constexpr(sfb_groups):
                                scales[mi][run] = nvvm.shfl_sync(0xFFFFFFFF, sfb_grid[run // 32], grid_base + run % 32, 0x1F, nvvm.Shfl.IDX)
                                if cutlass.const_expr(not _sfa_folded):
                                    scales[mi][run] = row_scales[mi] * scales[mi][run]

                    while not nvvm.mbarrier_try_wait_parity(
                        partial_full_mbar_ptr.subview(partial_slot),
                        partial_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

                    if cutlass.const_expr(promo_scale_table):
                        # The partial is full, so every warp is done with what this stage's table held: the warpgroup's first warp
                        # writes the window's scales (loaded during the previous window) and issues the next window's loads; the
                        # others wait for the table.
                        if promo_is_writer:
                            tbl_next_k = window_k + insts_per_window * mma_inst_k_elems
                            if cutlass.const_expr(_sfa_folded):
                                tbl_frow = coord_m_tile + promo_mi0 * epi_rows_per_mma_m
                                tbl_fa_scale = cutlass.Float32(0.0)
                                if (tbl_frow < M) & window_live:
                                    tbl_fa_scale = sfa_ahead[0].to(cutlass.Float32)
                                sfa_ahead[0] = _load_sfa(sfa_ptr, sfa_stride_m, sfa_stride_k, sfa_stride_l, sfa_l, tbl_frow, tbl_next_k, M, k)
                            for load in cutlass.range_constexpr(_sfb_loads):
                                tbl_run = load * 32 + promo_warp_lane
                                tbl_col = coord_n_c + promo_n0 + tbl_run * sfb_group_cols
                                tbl_b_scale = cutlass.Float32(0.0)
                                if (tbl_col < group_end) & window_live & (tbl_run < sfb_groups):
                                    tbl_b_scale = sfb_ahead[load].to(cutlass.Float32)
                                if cutlass.const_expr(_sfa_folded):
                                    tbl_b_scale = tbl_b_scale * tbl_fa_scale
                                (promo_tbl.subview(partial_slot * _promo_tbl_runs + tbl_run)).store(tbl_b_scale)
                                sfb_ahead[load] = _load_sfb(sfb_ptr, sfb_stride_k, sfb_stride_n, sfb_stride_l, 0, tbl_col, tbl_next_k, N, k)
                            # Every lane's stores precede the elected release.
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            if elect_one:
                                nvvm.mbarrier_arrive(promo_tbl_full.subview(partial_slot))
                        else:
                            while not nvvm.mbarrier_try_wait_parity(
                                promo_tbl_full.subview(partial_slot),
                                partial_full_phase_bit,
                                time_limit=10_000_000,
                            ):
                                pass

                    partial_col = base_col_id_root + partial_col_base + partial_slot * acc_region_cols + promo_col0
                    for mi in cutlass.range_constexpr(promo_mi_count):
                        for ch in cutlass.range_constexpr(promo_chunks):
                            col_off = mi * epi_cols_per_mma_m + ch * promo_chunk_cols
                            sum_off = mi * promo_n_cols + ch * promo_chunk_cols
                            part = nvvm.tcgen05_ld(promo_shape, cutlass.inttoptr(promo_row | (partial_col + col_off), 6, mma_acc_dtype), num=promo_chunk_cols)
                            if cutlass.const_expr(mma_acc_dtype is cutlass.Int32):
                                part = part.to(cutlass.Float32)
                            if cutlass.const_expr(promo_in_registers):
                                for e in cutlass.range_constexpr(0, promo_chunk_cols, 2):
                                    c = ch * promo_chunk_cols + e
                                    if cutlass.const_expr(sfb_present and not promo_scales_ahead):
                                        # This column pair's runs.
                                        for run in cutlass.range_constexpr(c // sfb_group_cols, (c + 1) // sfb_group_cols + 1):
                                            if cutlass.const_expr(run not in scales[mi]):
                                                if cutlass.const_expr(promo_scale_table):
                                                    scales[mi][run] = (promo_tbl.subview(partial_slot * _promo_tbl_runs + run)).load()
                                                else:
                                                    scales[mi][run] = nvvm.shfl_sync(0xFFFFFFFF, sfb_grid[run // 32], grid_base + run % 32, 0x1F, nvvm.Shfl.IDX)
                                                if cutlass.const_expr(not _sfa_folded):
                                                    scales[mi][run] = row_scales[mi] * scales[mi][run]
                                    sum0, sum1 = _promo_fma_pair(
                                        scales[mi][c // sfb_group_cols],
                                        scales[mi][(c + 1) // sfb_group_cols],
                                        part[e],
                                        part[e + 1],
                                        promo_sum[sum_off + e],
                                        promo_sum[sum_off + e + 1],
                                    )
                                    promo_sum[sum_off + e] = sum0
                                    promo_sum[sum_off + e + 1] = sum1
                            else:
                                if cutlass.const_expr(sfb_present and not promo_scales_ahead):
                                    # This load's column runs.
                                    for run in cutlass.range_constexpr(
                                        (ch * promo_chunk_cols) // sfb_group_cols, ((ch + 1) * promo_chunk_cols - 1) // sfb_group_cols + 1
                                    ):
                                        if cutlass.const_expr(run not in scales[mi]):
                                            if cutlass.const_expr(promo_scale_table):
                                                scales[mi][run] = (promo_tbl.subview(partial_slot * _promo_tbl_runs + run)).load()
                                            else:
                                                scales[mi][run] = nvvm.shfl_sync(0xFFFFFFFF, sfb_grid[run // 32], grid_base + run % 32, 0x1F, nvvm.Shfl.IDX)
                                            if cutlass.const_expr(not _sfa_folded):
                                                scales[mi][run] = row_scales[mi] * scales[mi][run]
                                scale_vec = cutlass.Vector.from_elements(
                                    tuple(scales[mi][(ch * promo_chunk_cols + c) // sfb_group_cols] for c in range(promo_chunk_cols)), cutlass.Float32
                                )
                                main = nvvm.tcgen05_ld(promo_shape, cutlass.inttoptr(promo_row | (acc_col + col_off), 6, cutlass.Float32), num=promo_chunk_cols)
                                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                                total = part * scale_vec
                                # The first window overwrites whatever an earlier tile left in this stage.
                                if window != 0:
                                    total = total + main
                                nvvm.tcgen05_st(promo_shape, cutlass.inttoptr(promo_row | (acc_col + col_off), 6, cutlass.Float32), total)

                    # Partial reads (and main writes) complete before the MMA may refill the partial buffer.
                    if cutlass.const_expr(promo_in_registers):
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                    else:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                    if elect_one:
                        if cutlass.const_expr(cta_group == 2):
                            nvvm.mbarrier_arrive(
                                nvvm.mapa(partial_empty_mbar_ptr.subview(partial_slot), pair_leader_rank),
                                scope=nvvm.MemScope.CLUSTER,
                                relaxed=True,
                            )
                        else:
                            nvvm.mbarrier_arrive(partial_empty_mbar_ptr.subview(partial_slot))
                    partial_iter += 1

                if cutlass.const_expr(promo_in_registers):
                    # The MMA warp committed this tile's windows only after the epilogue drained the stage's previous tile.
                    for mi in cutlass.range_constexpr(promo_mi_count):
                        for ch in cutlass.range_constexpr(promo_chunks):
                            col_off = mi * epi_cols_per_mma_m + ch * promo_chunk_cols
                            sum_off = mi * promo_n_cols + ch * promo_chunk_cols
                            total = cutlass.Vector.from_elements(tuple(promo_sum[sum_off + c] for c in range(promo_chunk_cols)), cutlass.Float32)
                            nvvm.tcgen05_st(promo_shape, cutlass.inttoptr(promo_row | (acc_col + col_off), 6, cutlass.Float32), total)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                # The stores are fenced above: the stage now holds the whole K range of the tile.
                if elect_one:
                    nvvm.mbarrier_arrive(acc_full_mbar_ptr.subview(acc_stage))
                tile_iter += 1

    if warp_idx == unused_warp_id:
        if cutlass.const_expr(not moe_combine_swap_bulk_enabled):
            nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    tma_workspace: cute.Tensor,
    # @@INJECT_HOST_AB_PARAMS@@
    # @@INJECT_HOST_TAP_PARAMS@@
    # @@INJECT_HOST_AUX_PARAMS@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_HOST_TMA_C_PARAMS@@
    # @@TMA_STORE_ONLY:END@@
    stream: _cuda.CUstream,
) -> None:
    """Bind the token and weight tensor maps, then launch the grouped GEMM."""
    # @@INJECT_HOST_AB_LISTS@@

    m = problem_size[0]
    n = problem_size[1]
    k_sym = problem_size[2]
    num_experts = problem_size[3]
    num_groups = problem_size[4]
    _stride_idx = 5
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

    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_HOST_TMA_C_LISTS@@
    # @@INJECT_HOST_TMA_C_DESCS@@
    # @@TMA_STORE_ONLY:END@@

    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[m, k_sym, num_experts],
                    global_strides=[a_stride_k * a_dtype.width // 128, a_stride_l * a_dtype.width // 128],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=a_tma_desc_dtype,
                    global_dims=[k_sym, m, num_experts],
                    global_strides=[a_stride_m * a_dtype.width // 128, a_stride_l * a_dtype.width // 128],
                    box_dims=[cta_tile_mnk[2], a_tma_box_m, 1],
                    swizzle=a_tma_swizzle,
                    tma_format=a_tma_format,
                )
            )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        token_dims = [k_sym, n, 1]
        token_strides = [b_stride_n * b_dtype.width // 128, b_stride_l * b_dtype.width // 128]
        token_box = [cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1]
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=b_tma_desc_dtype,
                global_dims=token_dims,
                global_strides=token_strides,
                box_dims=token_box,
                swizzle=b_tma_swizzle,
                tma_format=b_tma_format,
            )
        )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    counter_qword = grid_num_clusters * cluster_m * cluster_n * moe_desc_slots * TENSOR_MAP_QWORDS
    _dynamic_scheduler_counter_initialization(tma_workspace, cutlass.Int32(counter_qword)).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)
    _kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        tma_workspace,
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        # @@INJECT_MOE_HOST_MA_PASS@@
        # @@INJECT_HOST_TAP_PASS@@
        # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        # @@INJECT_HOST_AUX_PASS@@
        # @@TMA_STORE_ONLY:BEGIN@@
        # @@INJECT_HOST_TMA_C_PASS@@
        # @@TMA_STORE_ONLY:END@@
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        # One CTA per SM fixes the register count at entry, which is what lets the assembler give the
        # promotion warps the registers they raise themselves to (setmaxnreg) for the running sum.
        min_blocks_per_mp=1,
        use_pdl=USE_PDL,
        # @@INJECT_SWAP_COMBINE_LAUNCH@@
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    a_stride_elems = 16 // (a_dtype.width // 8)
    b_stride_elems = 16 // (b_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=min(out_vec_elems, moe_token_alignment))
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_akp = cute.sym_int64()
    sym_bkp = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()
    sym_source_n = sym_n

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_m, sym_akp, sym_e),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_source_n, sym_bkp, 1),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sfa_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=sfa_dtype.width // 8,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sfb_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=sfb_dtype.width // 8,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (cute.sym_int64(),),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_ctas = grid_num_clusters * cluster_m * cluster_n
    fake_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * moe_desc_slots * 16 + 16,),
        stride_order=(0,),
        assumed_align=128,
    )

    def _sym_operand_strides(is_mn_major: bool, stride_elems: int) -> tuple:
        # Operand is permuted to (M|N, K, L): the unit stride is mode 0 when MN-major, mode 1 when K-major, and never reaches TMA.
        unit = 0 if is_mn_major else 1
        return tuple(cute.sym_int64() if i == unit else cute.sym_int64(divisibility=stride_elems) for i in range(3))

    sym_a_strides = []
    for _ in range(num_a_operands):
        sym_a_strides.extend(_sym_operand_strides(a_is_m_major, a_stride_elems))
    sym_b_strides = []
    for _ in range(num_b_operands):
        sym_b_strides.extend(_sym_operand_strides(b_is_n_major, b_stride_elems))

    # @@INJECT_COMPILE_REDUCTION_STRIDE_DECLS@@

    # @@INJECT_COMPILE_AB_FAKES@@

    # @@INJECT_COMPILE_TAP_FAKES@@

    # @@TMA_STORE_ONLY:BEGIN@@
    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, sym_n // _div, 1),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
            assumed_align=16,
        )

    # @@INJECT_COMPILE_TMA_C_FAKES@@
    # @@TMA_STORE_ONLY:END@@

    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_e,
        sym_g,
        *sym_a_strides,
        *sym_b_strides,
        # @@INJECT_COMPILE_REDUCTION_STRIDE_SYMBOLS@@
    )

    # @@INJECT_COMPILE_AUX_FAKES@@

    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return _compile_cached(
        _host,
        problem_size,
        fake_first_token_offset,
        fake_tma_workspace,
        # @@INJECT_COMPILE_AB_PASS@@
        # @@INJECT_COMPILE_TAP_PASS@@
        # @@INJECT_COMPILE_AUX_PASS@@
        # @@TMA_STORE_ONLY:BEGIN@@
        # @@INJECT_COMPILE_TMA_C_PASS@@
        # @@TMA_STORE_ONLY:END@@
        stream=_fake_stream,
        options=frost_compile_options,
        # persistent object across processes (cudnn.frost.compiled_cache); the digest of THIS source is the key
        cache_key=globals().get("FROST_SOURCE_DIGEST"),
        symbol="frost_gemm",
    )
