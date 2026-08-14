# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm100 cta_group=2 **block-scaled** GEMM kernel: cluster MMA + STATIC scheduler (no CLC) + double-TMEM.

Computes ``C = (descale_a ⊙ A) @ (descale_b ⊙ B)`` with per-block K scales
dequantized inside the 2-CTA MMA pair; supports nvfp4 / mxfp4 / mxfp8. No CLC —
the grid launches one cluster per cgrp_tile and each MMA pair runs its blockIdx
tile once and exits, for small (≤1-wave) shapes. Compiler picks this when
``TileConfig.cta_group == 2 and static_sched``.

Warp layout (8 warps × 32 = 256 threads/CTA):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (leader CTA runs MMA; follower CTA only allocs/frees TMEM)  — setmaxnreg.dec 40
  warp  5   : TMA producer (both CTAs load their slice)  — setmaxnreg.dec 40
  warp  6   : idle (no CLC scheduler) — donates register budget  — setmaxnreg.dec 40
  warp  7   : unused donor — setmaxnreg.dec 40, then idle.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.kernel_templates._tile_helpers import (
    l2_swizzle_tile as _l2_swizzle_tile,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
from cutlass import apply_swizzle as _apply_smem_swizzle
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda

# @@INJECT_TILE_CONSTANTS@@


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


def _b_collector_op(mi):
    """B is identical across the M sub-blocks (only A's address advances), so the
    first MMA fills the B collector and the rest read it back instead of
    re-fetching the same operand from SMEM. `.collector::b::*` is silicon-gated
    (sm_107a only), hence `b_collector_ok`."""
    if cutlass.const_expr(not b_collector_ok or num_mma_m == 1):
        return None
    if cutlass.const_expr(mi == 0):
        return nvvm.Tcgen05MMACollectorOp.FILL
    if cutlass.const_expr(mi == num_mma_m - 1):
        return nvvm.Tcgen05MMACollectorOp.LASTUSE
    return nvvm.Tcgen05MMACollectorOp.USE


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

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster % cluster_m
    n_rank = cta_rank_in_cluster // cluster_m
    pair_member = m_rank % 2
    pair_m_idx = m_rank // 2
    is_pair_leader = pair_member == 0
    pair_leader_rank = pair_m_idx * 2 + n_rank * cluster_m

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

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

    a_pattern = 0
    for n_idx in cutlass.range_constexpr(cluster_n):
        a_pattern = a_pattern | (1 << (n_idx * cluster_m))
    b_pattern = 0
    for pm_idx in cutlass.range_constexpr(cluster_m // 2):
        b_pattern = b_pattern | (1 << (pm_idx * 2))

    if cutlass.const_expr(multicast_a):
        tma_mcast_mask_a = cutlass.Int16(a_pattern << m_rank)
    else:
        tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
    if cutlass.const_expr(multicast_b):
        tma_mcast_mask_b = cutlass.Int16((b_pattern << pair_member) << (n_rank * cluster_m))
    else:
        tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    sA_elems = sA_packed_elems
    sB_elems = sB_packed_elems
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
    smem_sfa_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfa_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_sfb_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfb_smem_bytes * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    # @@TMA_STORE_ONLY:BEGIN@@
    epi_subtile_elems = epi_tile_mn[0] * epi_tile_mn[1]
    smem_d_ptr = cutlass.Array(
        cd_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    # @@TMA_STORE_ONLY:END@@

    acc_empty_count = num_epilogue_warps * 2
    cta_group = 2
    ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    if warp_idx == 0:
        if cutlass.const_expr(use_acc_overlap):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
        else:
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = (num_a_operands * (sA_bytes + sfa_smem_bytes) + num_b_operands * (sB_bytes + sfb_smem_bytes)) * 2

    pair_n_size = cgrp_tile_mnk[1] // cluster_n
    # Per-CTA output rows one MMA-M block covers. The pair splits M, so this is
    # the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // num_mma_m
    if cutlass.const_expr(epi_rows_per_mma_m == 64):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        cols_per_acc_stage = pair_n_size // 2
    else:
        cols_per_acc_stage = pair_n_size
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    nvvm.barrier_cluster_wait()
    nvvm.barrier_cta_sync(0)

    # @@INJECT_TAP_PTRS@@

    VEC_BYTES = vec_bytes_epi
    vsize = (VEC_BYTES * 8) // cd_dtype.width

    M = m
    N = n
    num_tile_m = cute.ceil_div(M, cgrp_tile_mnk[0])
    num_tile_n = cute.ceil_div(N, cgrp_tile_mnk[1])
    num_k_tiles = cute.ceil_div(k, cgrp_tile_mnk[2])

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)

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
        while is_valid != 0:
            logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n
            coord_m_per_cta = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
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

                coord_k = k_tile_idx * cgrp_tile_mnk[2]
                coord_sf_k = k_tile_idx * sf_tma_box_k
                coord_n_pair = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n
                sfb_n_block = coord_n_pair // 128

                if is_pair_leader:
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)

                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    sSFA_stage = smem_sfa_list[_ai].subview(sfa_smem_bytes * stage)
                    tma_sfa_desc = tma_sfa_descs[_ai]
                    sfa_m_block = coord_m_per_cta // 128
                    if cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if cutlass.const_expr(a_is_m_major):
                                for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sA_stage.subview(m_group * a_tma_group_elems * cgrp_tile_mnk[2]),
                                            tma_a_desc.get_ptr(),
                                            (
                                                coord_m_per_cta + m_group * a_tma_group_elems,
                                                coord_k,
                                                tile_l_a,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=nvvm.CTAGroup.CTA_2,
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
                                        group=nvvm.CTAGroup.CTA_2,
                                    )
                    else:
                        if cutlass.const_expr(a_is_m_major):
                            for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(m_group * a_tma_group_elems * cgrp_tile_mnk[2]),
                                        tma_a_desc.get_ptr(),
                                        (
                                            coord_m_per_cta + m_group * a_tma_group_elems,
                                            coord_k,
                                            tile_l_a,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=nvvm.CTAGroup.CTA_2,
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
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    if cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFA_stage,
                                    tma_sfa_desc.get_ptr(),
                                    (0, coord_sf_k, sfa_m_block, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFA_stage,
                                tma_sfa_desc.get_ptr(),
                                (0, coord_sf_k, sfa_m_block, tile_l_a),
                                ab_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_a,
                                group=nvvm.CTAGroup.CTA_2,
                            )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    sSFB_stage = smem_sfb_list[_bj].subview(sfb_smem_bytes * stage)
                    tma_sfb_desc = tma_sfb_descs[_bj]
                    if cutlass.const_expr(multicast_b):
                        if pair_m_idx == 0:
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                tile_l_b,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=nvvm.CTAGroup.CTA_2,
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
                                        group=nvvm.CTAGroup.CTA_2,
                                    )
                    else:
                        if cutlass.const_expr(b_is_n_major):
                            for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (
                                            coord_n_per_cta + n_group * b_tma_group_elems,
                                            coord_k,
                                            tile_l_b,
                                        ),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_2,
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
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    if cutlass.const_expr(multicast_b):
                        if pair_m_idx == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFB_stage,
                                    tma_sfb_desc.get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFB_stage,
                                tma_sfb_desc.get_ptr(),
                                (0, coord_sf_k, sfb_n_block, tile_l_b),
                                ab_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=nvvm.CTAGroup.CTA_2,
                            )

                ab_iter += 1

            is_valid = cutlass.Int32(0)
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
    ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.tcgen05_alloc(
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
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            acc_stage = cutlass.Int32(0)

            idesc_by_j = [
                cutlass.experimental.primitives.Tcgen05MxInstrDesc.build(
                    a_dtype=idesc_a_dtype,
                    b_dtype=idesc_b_dtype,
                    scale_format=sf_scale_format,
                    n_dim=mma_n_dim,
                    m_dim=mma_m_dim,
                    a_major=mma_a_major,
                    b_major=mma_b_major,
                    a_sf_id=j * sf_scales_per_inst,
                    b_sf_id=j * sf_scales_per_inst,
                )
                for j in range(sf_insts_per_atom)
            ]

            sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
            sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
            s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
            sfa_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfa_tmem_bases]
            sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
            sfa_dst_ptrs = [
                [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block, cutlass.Float32) for m in range(num_blocks_m)] for i in range(num_a_operands)
            ]
            sfb_dst_ptrs = [
                [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + m * registers_per_block, cutlass.Float32) for m in range(num_blocks_n)] for j in range(num_b_operands)
            ]
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

                if cutlass.const_expr(use_acc_overlap):
                    acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
                else:
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                # One accumulator per (gemm, M block); M block mi sits
                # epi_cols_per_mma_m columns further into its GEMM's region and
                # reads SF word block mi (SF words are one per 128 rows).
                acc_tmem_ptrs = [
                    [
                        nvvm.make_tmem_ptr(
                            (base_row_id << 16) | (acc_base_col + g * acc_gemm_stride + mi * epi_cols_per_mma_m),
                            cutlass.Float32,
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

                    while not nvvm.mbarrier_try_wait_parity(
                        ab_full_mbar_ptr.subview(stage),
                        ab_full_phase_bit,
                        time_limit=10_000_000,
                    ):
                        pass

                    desc_a_bases = [
                        cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                            start_address=smem_a_list[i].subview(sA_elems * stage),
                            leading_byte_offset=a_smem_desc_leading_byte_offset,
                            stride_byte_offset=a_smem_desc_stride_byte_offset,
                            layout=ab_smem_swizzle,
                        )
                        for i in range(num_a_operands)
                    ]
                    desc_b_bases = [
                        cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                            start_address=smem_b_list[j].subview(sB_elems * stage),
                            leading_byte_offset=b_smem_desc_leading_byte_offset,
                            stride_byte_offset=b_smem_desc_stride_byte_offset,
                            layout=ab_smem_swizzle,
                        )
                        for j in range(num_b_operands)
                    ]

                    desc_sfa_bases = [
                        cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                            start_address=smem_sfa_list[i].subview(sfa_smem_bytes * stage),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                        )
                        for i in range(num_a_operands)
                    ]
                    desc_sfb_bases = [
                        cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                            start_address=smem_sfb_list[j].subview(sfb_smem_bytes * stage),
                            leading_byte_offset=16,
                            stride_byte_offset=128,
                            layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
                        )
                        for j in range(num_b_operands)
                    ]

                    for atom_r in cutlass.range(num_sf_atoms, unroll_full=True):
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            for _m in cutlass.range_constexpr(num_blocks_m):
                                if elect_one:
                                    nvvm.tcgen05_cp(
                                        s2t_shape,
                                        sfa_dst_ptrs[_ai][_m],
                                        desc_sfa_bases[_ai] + (sf_atom_desc_stride * atom_r + sf_block_desc_stride * _m),
                                        group=nvvm.CTAGroup.CTA_2,
                                        multicast=s2t_multicast,
                                    )
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            for _m in cutlass.range_constexpr(num_blocks_n):
                                if elect_one:
                                    nvvm.tcgen05_cp(
                                        s2t_shape,
                                        sfb_dst_ptrs[_bj][_m],
                                        desc_sfb_bases[_bj] + (sf_atom_desc_stride * atom_r + sf_block_desc_stride * _m),
                                        group=nvvm.CTAGroup.CTA_2,
                                        multicast=s2t_multicast,
                                    )
                        for j in cutlass.range_constexpr(sf_insts_per_atom):
                            k_block_idx = atom_r * sf_insts_per_atom + j
                            idesc_k = idesc_by_j[j]
                            for g in cutlass.range_constexpr(num_gemms):
                                _ai = gemm_a_idx[g]
                                _bj = gemm_b_idx[g]
                                desc_a_k = desc_a_bases[_ai].advance_start_address(a_smem_k_step_bytes * k_block_idx)
                                desc_b = desc_b_bases[_bj].advance_start_address(b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(num_mma_m):
                                    # The M sub-block offset is a whole SMEM swizzle atom, so
                                    # the descriptor's swizzle phase is preserved. B and its SF
                                    # are shared; A's SF word block follows the M block.
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if elect_one:
                                        nvvm.tcgen05_mma_block_scale(
                                            mma_block_scale_kind,
                                            nvvm.CTAGroup.CTA_2,
                                            acc_tmem_ptrs[g][mi],
                                            desc_a,
                                            desc_b,
                                            idesc_k,
                                            enable_input_d=scale_d,
                                            scale_a=sfa_dst_ptrs[_ai][mi],
                                            scale_b=sfb_scale_ptrs[_bj],
                                            scale_vec_size=scale_vec_size,
                                            b_collector_op=_b_collector_op(mi),
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

                is_valid = cutlass.Int32(0)
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
                    while not nvvm.mbarrier_try_wait_parity(
                        acc_empty_mbar_ptr.subview(tail_stage),
                        tail_phase,
                        time_limit=10_000_000,
                    ):
                        pass
            nvvm.bar_warp_sync(0xFFFFFFFF)

            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
            peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
            while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                pass
            if cutlass.const_expr(not use_acc_overlap):
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            nvvm.tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=nvvm.CTAGroup.CTA_2,
            )
        else:
            if cutlass.const_expr(USE_PDL):
                if elect_one:
                    nvvm.griddepcontrol("launch_dependents")

            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
            peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
            if cutlass.const_expr(not use_acc_overlap):
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                pass
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            nvvm.tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=nvvm.CTAGroup.CTA_2,
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

        row_id_with_warp_offset = base_row_id + warp_idx * 32
        if cutlass.const_expr(cols_per_acc_stage >= 32):
            t2r_inst_repx = 32
            subtile_cnt = cols_per_acc_stage // 32
        else:
            t2r_inst_repx = cols_per_acc_stage
            subtile_cnt = 1
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # @@TMA_STORE_ONLY:END@@

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
            coord_n_c = tile_n * cgrp_tile_mnk[1] + n_rank * pair_n_size
            if cutlass.const_expr(epi_rows_per_mma_m == 64):
                coord_n_c = coord_n_c + (warp_idx // 2) * cols_per_acc_stage

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
            # The 2-CTA epilogue drains its own half of the instruction's M,
            # epi_rows_per_mma_m rows at a time, so a CTA tile of num_mma_m blocks
            # drains in num_mma_m passes over its own column region.
            for mi in cutlass.range_constexpr(num_mma_m):
                coord_m = coord_m_tile + mi * epi_rows_per_mma_m
                mi_col_base = acc_base_col + mi * epi_cols_per_mma_m
                tmem_col_addr_gemms = [(row_id_with_warp_offset << 16) | (mi_col_base + g * acc_gemm_stride) for g in range(num_gemms)]

                if cutlass.const_expr(epi_rows_per_mma_m == 64):
                    row = coord_m + (warp_idx % 2) * 32 + lane
                    row_active = True
                else:
                    row = coord_m + tidx
                    row_active = True

                # @@INJECT_AUX_VIEWS@@

                for subtile_idx in cutlass.range(subtile_cnt, unroll_full=True):
                    if cutlass.const_expr(use_acc_overlap):
                        _sub = subtile_idx + (1 - acc_buf_parity) * (subtile_cnt - 1 - 2 * subtile_idx)
                        subtile_col_offset = _sub * t2r_inst_repx
                    else:
                        subtile_col_offset = subtile_idx * t2r_inst_repx

                    if cutlass.const_expr(not (use_tma_store_epi and cd_out_is_m_major)):
                        c_rmem_vecs = []
                        for g in cutlass.range_constexpr(num_gemms):
                            tmem = cutlass.inttoptr(
                                tmem_col_addr_gemms[g] + subtile_col_offset,
                                6,
                                cutlass.Float32,
                            )
                            c_rmem_vecs.append(nvvm.tcgen05_ld(shape, tmem, num=t2r_inst_repx))
                        c_rmem_vec = c_rmem_vecs[0]

                    if use_acc_overlap and (not cd_out_is_m_major) and mi == num_mma_m - 1 and subtile_idx == acc_overlap_subtiles - 1:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if elect_one:
                            mbar_pair_ptr = nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank)
                            nvvm.mbarrier_arrive(mbar_pair_ptr, scope=nvvm.MemScope.CLUSTER, relaxed=True)

                    col = coord_n_c + subtile_col_offset

                    # @@TMA_STORE_ONLY:BEGIN@@
                    epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                    smem_subtile_ptr = smem_d_ptr.subview(epi_stage_idx * epi_subtile_elems)
                    smem_thr_ptr = smem_subtile_ptr.subview(tidx * t2r_inst_repx)

                    if cutlass.const_expr(cd_out_is_m_major):
                        ld_col = mi_col_base + subtile_col_offset
                        for _h in cutlass.range(2, unroll_full=True):
                            ld_row = base_row_id + warp_idx * 32 + _h * 16
                            ld_addr = (ld_row << 16) | ld_col
                            ld_tmem = cutlass.inttoptr(ld_addr, 6, cutlass.Float32)
                            _lv = nvvm.tcgen05_ld(nvvm.Tcgen05LdStShape.SHAPE_16X256B, ld_tmem, num=4)
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
                        for j in cutlass.range_constexpr(t2r_inst_repx // vsize):
                            col_j = col + j * vsize
                            if col_j + vsize <= N:
                                vec_f32 = c_rmem_vec[j * vsize : (j + 1) * vsize]

                                # @@INJECT_STG_VEC_BINDINGS@@

                                # @@INJECT_EPILOGUE@@
                    # @@STG_ONLY:END@@

            if cutlass.const_expr((not use_acc_overlap) or cd_out_is_m_major):
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if elect_one:
                    mbar_pair_ptr = nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank)
                    nvvm.mbarrier_arrive(mbar_pair_ptr, scope=nvvm.MemScope.CLUSTER, relaxed=True)

            is_valid = cutlass.Int32(0)

            tile_iter += 1

        if cutlass.const_expr(use_acc_overlap):
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if elect_one:
                nvvm.mbarrier_arrive(tmem_dealloc_mbar_ptr)

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
    a_stride_m = problem_size[4]
    a_stride_k = problem_size[5]
    a_stride_l = problem_size[6]
    b_stride_n = problem_size[7]
    b_stride_k = problem_size[8]
    b_stride_l = problem_size[9]
    # @@INJECT_HOST_REDUCTION_STRIDES@@

    if cutlass.const_expr(matmul_a_batch == 1):
        a_batch = 1
    else:
        a_batch = batch
    if cutlass.const_expr(matmul_b_batch == 1):
        b_batch = 1
    else:
        b_batch = batch

    rest_k = ((k_sym // block_size) + 3) // 4
    rest_m = (m + 127) // 128
    rest_n = (n + 127) // 128
    tma_a_desc_list = []
    tma_sfa_desc_list = []
    for _a_op, _sfa_op in zip(_a_operands, _sfa_operands):
        if cutlass.const_expr(a_is_m_major):
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_desc_dtype,
                    global_dims=[m, k_sym, a_batch],
                    global_strides=[
                        a_stride_k * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[a_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                    tma_format=ab_tma_format,
                )
            )
        else:
            tma_a_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_a_op.iterator.toint(),
                    dtype=ab_tma_desc_dtype,
                    global_dims=[k_sym, m, a_batch],
                    global_strides=[
                        a_stride_m * ab_dtype.width // 128,
                        a_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0], 1],
                    swizzle=ab_tma_swizzle,
                    tma_format=ab_tma_format,
                )
            )
        sfa_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_m, batch),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_m, 8),
                ),
            ),
        )
        tma_sfa_desc_list.append(
            _tma.create_tensor_map_tiled_from_view(
                sfa_fp16_tensor,
                dtype=cutlass.Uint16,
                box_dims=(256, sf_tma_box_k, sfa_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
            )
        )
    tma_b_desc_list = []
    tma_sfb_desc_list = []
    for _b_op, _sfb_op in zip(_b_operands, _sfb_operands):
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_desc_dtype,
                    global_dims=[n, k_sym, b_batch],
                    global_strides=[
                        b_stride_k * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[b_tma_group_elems, cta_tile_mnk[2], 1],
                    swizzle=ab_tma_swizzle,
                    tma_format=ab_tma_format,
                )
            )
        else:
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_desc_dtype,
                    global_dims=[k_sym, n, b_batch],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1], 1],
                    swizzle=ab_tma_swizzle,
                    tma_format=ab_tma_format,
                )
            )
        sfb_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_n, batch),
                stride=(
                    1,
                    256,
                    cute.assume(256 * rest_k, 8),
                    cute.assume(256 * rest_k * rest_n, 8),
                ),
            ),
        )
        tma_sfb_desc_list.append(
            _tma.create_tensor_map_tiled_from_view(
                sfb_fp16_tensor,
                dtype=cutlass.Uint16,
                box_dims=(256, sf_tma_box_k, sfb_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
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
    _kernel(
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
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 128 // ab_dtype.width
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_kp = cute.sym_int64()
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
            a_fake_dtype,
            (sym_m, sym_kp, sym_a_l),
            stride_order=(0, 1, 2) if a_is_m_major else (1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_kp, sym_b_l),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # SF reaches the kernel as a base pointer only; the host rebuilds the
    # F8_128x4 view from problem_size, so no SF mode carries a layout contract.
    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
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
    # @@INJECT_COMPILE_AB_FAKES@@
    # The operand's unit stride (m/n when MN-major, k when K-major) never reaches TMA, so it carries no 16B contract.
    sym_a_stride_m = cute.sym_int64() if a_is_m_major else cute.sym_int64(divisibility=ab_stride_elems)
    sym_a_stride_k = cute.sym_int64(divisibility=ab_stride_elems) if a_is_m_major else cute.sym_int64()
    sym_a_stride_l = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_n = cute.sym_int64() if b_is_n_major else cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_k = cute.sym_int64(divisibility=ab_stride_elems) if b_is_n_major else cute.sym_int64()
    sym_b_stride_l = cute.sym_int64(divisibility=ab_stride_elems)
    # @@INJECT_COMPILE_REDUCTION_STRIDE_DECLS@@
    # @@INJECT_COMPILE_TAP_FAKES@@
    problem_size = (
        sym_m,
        sym_n,
        sym_k,
        sym_l,
        sym_a_stride_m,
        sym_a_stride_k,
        sym_a_stride_l,
        sym_b_stride_n,
        sym_b_stride_k,
        sym_b_stride_l,
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
