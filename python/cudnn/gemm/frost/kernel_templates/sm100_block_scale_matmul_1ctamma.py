# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm100 CTA_1 **block-scaled** GEMM kernel: persistent + CLC dynamic scheduler.

Computes ``C = (descale_a ⊙ A) @ (descale_b ⊙ B)`` where A/B are narrow (FP4
e2m1 packed 2-per-byte, or FP8 e4m3/e5m2), dequantized by a per-block scale
factor along K inside the MMA. Supports nvfp4 (fp4/e4m3/block16), mxfp4
(fp4/e8m0/block32), and mxfp8 (fp8/e8m0/block32). Single-CTA MMA; the compiler
picks this when ``TileConfig.cta_group == 1``.

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
    tcgen05_mma_block_scale as _tcgen05_mma_block_scale,
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

if use_acc_overlap and any(_w != epi_n for _, _w in _epi_subtile_spans(epi_cols_per_mma_m, epi_n)):
    raise NotImplementedError(f"{__name__}: acc overlap reverses subtiles by index, which needs a uniform drain width")


CLC_SCHED_STAGES = 1

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
    # Every catalog cluster dimension is a power of two.  Mixed-CGA makes the
    # divisor runtime-visible, so spelling rank decomposition as div/mod would
    # otherwise lower to reciprocal-based integer division in every warp.
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

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
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = a_part_arrive | b_part_arrive

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    sf_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
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

    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_count = cluster_size
    else:
        ab_empty_count = cluster_m + cluster_n - 1
    num_consumer_warps_per_cta = 7
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), num_epilogue_warps)
        if cutlass.const_expr(use_acc_overlap):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, num_epilogue_warps)
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
    ab_only_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes
    sf_only_copy_bytes = num_a_operands * sfa_smem_bytes + num_b_operands * sfb_smem_bytes

    epi_rows_per_mma_m = cta_tile_mnk[0] // num_mma_m
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    # @@INJECT_TAP_PTRS@@

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    # The tile this cluster owns spans its OWN cluster shape; both shapes walk
    # the grid as the identity map (tile == blockIdx), so they tile the problem
    # identically and every output tile is still covered exactly once.
    cgrp_tile_m_cur = cta_tile_mnk[0] * cluster_m
    cgrp_tile_n_cur = cta_tile_mnk[1] * cluster_n

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
                coord_sf_k = k_tile_idx * sf_tma_box_k
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)

                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    sSFA_stage = smem_sfa_list[_ai].subview(sfa_smem_bytes * stage)
                    tma_sfa_desc = tma_sfa_descs[_ai]
                    sfa_m_block = coord_m_per_cta // 128
                    if cutlass.const_expr(multicast_a):
                        if n_rank == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFA_stage,
                                    tma_sfa_desc.get_ptr(),
                                    (0, coord_sf_k, sfa_m_block, tile_l_a),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFA_stage,
                                tma_sfa_desc.get_ptr(),
                                (0, coord_sf_k, sfa_m_block, tile_l_a),
                                sf_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_a,
                                group=nvvm.CTAGroup.CTA_1,
                            )

                for _bj in cutlass.range_constexpr(num_b_operands):
                    sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                    tma_b_desc = tma_b_descs[_bj]
                    sSFB_stage = smem_sfb_list[_bj].subview(sfb_smem_bytes * stage)
                    tma_sfb_desc = tma_sfb_descs[_bj]
                    sfb_n_block = coord_n_per_cta // 128
                    if cutlass.const_expr(multicast_b):
                        if m_rank == 0:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sSFB_stage,
                                    tma_sfb_desc.get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, tile_l_b),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                sSFB_stage,
                                tma_sfb_desc.get_ptr(),
                                (0, coord_sf_k, sfb_n_block, tile_l_b),
                                sf_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=nvvm.CTAGroup.CTA_1,
                            )

                for _ai in cutlass.range_constexpr(num_a_operands):
                    sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                    tma_a_desc = tma_a_descs[_ai]
                    sSFA_stage = smem_sfa_list[_ai].subview(sfa_smem_bytes * stage)
                    tma_sfa_desc = tma_sfa_descs[_ai]
                    sfa_m_block = coord_m_per_cta // 128
                    if cutlass.const_expr(a_mcast_slices > 1):
                        _a_rows = cta_tile_mnk[0] // a_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sA_stage.subview(n_rank * _a_rows * ab_packed_per_row),
                                    tma_a_desc.get_ptr(),
                                    (coord_k, coord_m_per_cta + n_rank * _a_rows, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                        else:
                            _a_per_cta = a_mcast_slices >> _preferred_cluster_n_shift
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _a_per_cta = a_mcast_slices >> _fallback_cluster_n_shift
                            for _asl in cutlass.range(_a_per_cta):
                                _a_idx = n_rank * _a_per_cta + _asl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(_a_idx * _a_rows * ab_packed_per_row),
                                        tma_a_desc.get_ptr(),
                                        (coord_k, coord_m_per_cta + _a_idx * _a_rows, tile_l_a),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_a,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                    elif cutlass.const_expr(multicast_a):
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
                    sSFB_stage = smem_sfb_list[_bj].subview(sfb_smem_bytes * stage)
                    tma_sfb_desc = tma_sfb_descs[_bj]
                    sfb_n_block = coord_n_per_cta // 128
                    if cutlass.const_expr(b_mcast_slices > 1):
                        _b_rows = cta_tile_mnk[1] // b_mcast_slices
                        if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    sB_stage.subview(m_rank * _b_rows * ab_packed_per_row),
                                    tma_b_desc.get_ptr(),
                                    (coord_k, coord_n_per_cta + m_rank * _b_rows, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_1,
                                )
                        else:
                            _b_per_cta = b_mcast_slices >> _preferred_cluster_m_shift
                            if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                _b_per_cta = b_mcast_slices >> _fallback_cluster_m_shift
                            for _bsl in cutlass.range(_b_per_cta):
                                _b_idx = m_rank * _b_per_cta + _bsl
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_idx * _b_rows * ab_packed_per_row),
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_1,
                                    )
                    elif cutlass.const_expr(multicast_b):
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
        s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
        sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
        sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
        sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
        sfa_dst_ptrs = [
            [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block, cutlass.Float32) for m in range(num_mma_m)] for i in range(num_a_operands)
        ]
        sfb_dst_ptrs = [
            [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + m * registers_per_block, cutlass.Float32) for m in range(num_blocks_n)] for j in range(num_b_operands)
        ]
        # Descriptor metadata and the SMEM allocation base are invariant across
        # persistent tiles. The K loop only advances the encoded start address.
        desc_a_roots = [
            cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_a_list[i],
                leading_byte_offset=a_smem_desc_leading_byte_offset,
                stride_byte_offset=a_smem_desc_stride_byte_offset,
                layout=ab_smem_swizzle,
            )
            for i in range(num_a_operands)
        ]
        desc_b_roots = [
            cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_b_list[j],
                leading_byte_offset=b_smem_desc_leading_byte_offset,
                stride_byte_offset=b_smem_desc_stride_byte_offset,
                layout=ab_smem_swizzle,
            )
            for j in range(num_b_operands)
        ]
        desc_sfa_roots = [
            cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_sfa_list[i],
                leading_byte_offset=16,
                stride_byte_offset=128,
                layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
            )
            for i in range(num_a_operands)
        ]
        desc_sfb_roots = [
            cutlass.experimental.primitives.Tcgen05SmemDesc.build(
                start_address=smem_sfb_list[j],
                leading_byte_offset=16,
                stride_byte_offset=128,
                layout=cutlass.experimental.primitives.Tcgen05SmemSwizzle.NONE,
            )
            for j in range(num_b_operands)
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
            # epi_cols_per_mma_m columns further into its GEMM's region and reads
            # SF word block mi (SF words are one per 128 rows).
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

                desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]
                desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_smem_bytes * stage) for i in range(num_a_operands)]
                desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_smem_bytes * stage) for j in range(num_b_operands)]

                while not nvvm.mbarrier_try_wait_parity(sf_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                for sf_word in cutlass.range_constexpr(num_sf_atoms):
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        for block_n in cutlass.range_constexpr(num_blocks_n):
                            if elect_one:
                                nvvm.tcgen05_cp(
                                    s2t_shape,
                                    sfb_dst_ptrs[_bj][block_n],
                                    desc_sfb_bases[_bj] + (sf_atom_desc_stride * sf_word + sf_block_desc_stride * block_n),
                                    group=nvvm.CTAGroup.CTA_1,
                                    multicast=s2t_multicast,
                                )
                    if cutlass.const_expr(sf_word == 0):
                        while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                            pass
                    for mma_k_in_word in cutlass.range_constexpr(sf_insts_per_atom):
                        mma_k = sf_word * sf_insts_per_atom + mma_k_in_word
                        idesc_k = idesc_by_j[mma_k_in_word]
                        for gemm_i in cutlass.range_constexpr(num_gemms):
                            _ai = gemm_a_idx[gemm_i]
                            _bj = gemm_b_idx[gemm_i]
                            desc_a_k = desc_a_bases[_ai].advance_start_address(a_smem_k_step_bytes * mma_k)
                            desc_b = desc_b_bases[_bj].advance_start_address(b_smem_k_step_bytes * mma_k)
                            for mma_m in cutlass.range_constexpr(num_mma_m):
                                if cutlass.const_expr(mma_k_in_word == 0 and _ai not in gemm_a_idx[:gemm_i]):
                                    if elect_one:
                                        nvvm.tcgen05_cp(
                                            s2t_shape,
                                            sfa_dst_ptrs[_ai][mma_m],
                                            desc_sfa_bases[_ai] + (sf_atom_desc_stride * sf_word + sf_block_desc_stride * mma_m),
                                            group=nvvm.CTAGroup.CTA_1,
                                            multicast=s2t_multicast,
                                        )
                                # The M sub-block offset is a whole SMEM swizzle atom, so the
                                # descriptor's swizzle phase is preserved. B and its SF are
                                # shared; A's SF word block follows the M block.
                                desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                if elect_one:
                                    _tcgen05_mma_block_scale(
                                        mma_block_scale_kind,
                                        nvvm.CTAGroup.CTA_1,
                                        acc_tmem_ptrs[gemm_i][mma_m],
                                        desc_a,
                                        desc_b,
                                        idesc_k,
                                        enable_input_d=scale_d,
                                        scale_a=sfa_dst_ptrs[_ai][mma_m],
                                        scale_b=sfb_scale_ptrs[_bj],
                                        scale_vec_size=scale_vec_size,
                                        b_collector_op=_b_collector_op(mma_m),
                                    )
                        # Every accumulator sees scale_d=False on exactly the first
                        # k_block of the tile, so the flip stays outside mma_m.
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
            nvvm.griddepcontrol("launch_dependents")
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        tail_stage = acc_stage
        tail_phase = acc_empty_phase_bit
        for _ in range(acc_stages):
            tail_stage = tail_stage + 1
            if tail_stage == acc_stages:
                tail_stage = cutlass.Int32(0)
                tail_phase = tail_phase ^ 1
            while not nvvm.mbarrier_try_wait_parity(acc_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
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
            coord_n_c = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]

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

            for mi in cutlass.range_constexpr(num_mma_m):
                if cutlass.const_expr(use_acc_overlap and num_mma_m > 1):
                    _mi = mi + (1 - acc_buf_parity) * (num_mma_m - 1 - 2 * mi)
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
                        if cutlass.const_expr(mi == num_mma_m - 1 and subtile_idx == subtile_cnt - 1):
                            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                            if elect_one:
                                nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

                    if use_acc_overlap and mi * subtile_cnt + subtile_idx == acc_overlap_subtiles - 1:
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        if elect_one:
                            nvvm.mbarrier_arrive(acc_empty_mbar_ptr.subview(acc_stage))

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
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[0] // a_mcast_slices, 1],
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
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
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
    def _make_fake_c(_dt, _div, _mm):
        return make_fake_compact_tensor(
            _dt,
            (sym_m, sym_n // _div, sym_l),
            stride_order=(0, 1, 2) if _mm else (1, 0, 2),
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
