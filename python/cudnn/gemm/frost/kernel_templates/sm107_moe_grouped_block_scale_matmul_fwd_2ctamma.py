# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm107 cta_group=2 MoE grouped block-scale matmul fwd: grouped persistent
scheduler + per-group A TMA descriptor patch + block-scaled MMA.

Per routed group g: ``out[fto[g]:fto[g+1]] = deq(token[range]) @ deq(weight[g%E]).T``,
token/weight FP4/FP8 dequantized by per-block scale factors inside the MMA
(supports nvfp4 / mxfp4 / mxfp8). 2-CTA MMA cluster pair (cluster2x1 reference
design): the leader CTA issues the MMA, the follower consumes; both load their
operand slice. The TMA warp patches A's descriptor on each routed-group change.

The pipeline is the sm100 one; SM 10.7's block-scale MMA reads a **64-byte K**
per instruction instead of 32, which shows up in exactly two places (both
driven by injected constants, so the rest of the file stays in lockstep with
``sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py``):

  * half as many MMAs per K-tile, each consuming ``sf_scales_per_inst`` scales
    (8 at K-block 16, 4 at 32 — it follows the BLOCK SIZE, not the scale
    dtype). When that exceeds the 4 scales one 128x4 utccp atom holds, a scale
    *word* spans ``word_atoms`` atoms, and the two SF regions then lay them
    out DIFFERENTLY: SFB atom-major across its N-blocks, SFA block-major.
    At K-block 32 ``word_atoms == 1`` (identical to sm100).
  * fp4 rides the OMMA instruction descriptor (``Tcgen05MxOmmaInstrDesc``,
    K-mode 2 = 128 fp4 elements); mxfp8 stays on ``Tcgen05MxInstrDesc``
    (K-mode 1 = 64 fp8 elements). Both take the real operand dtype.

The K-tile itself is unchanged (128 bytes), so the grouped scheduler, the
per-group A-tensormap patch and the per-group-128-padded SF blob layout are
byte-for-byte the sm100 ones.

Warp layout (8 warps × 32 = 256 threads/CTA):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (leader CTA runs MMA; follower CTA consumes only)  — setmaxnreg.dec 40
  warp  5   : TMA producer (both CTAs load their slice; per-group A descriptor patch)  — setmaxnreg.dec 40
  warp  6   : grouped persistent scheduler  — setmaxnreg.dec 40
  warp  7   : unused donor — setmaxnreg.dec 40, idle to dealloc barrier
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.kernel_templates._tile_helpers import (
    copy_tensormap_to_workspace as _copy_tensormap_to_workspace,
    epi_subtile_spans as _epi_subtile_spans,
    fence_tensormap_acquire as _fence_tensormap_acquire,
    fence_tensormap_release as _fence_tensormap_release,
    moe_swizzle_tile as _moe_swizzle_tile,
    replace_tensormap_global_address as _replace_tensormap_global_address,
    replace_tensormap_global_dim_1 as _replace_tensormap_global_dim_1,
    replace_tensormap_global_dim_2 as _replace_tensormap_global_dim_2,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma_block_scale as _tcgen05_mma_block_scale,
    TENSOR_MAP_QWORDS,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda

# @@INJECT_TILE_CONSTANTS@@

# Tensormap workspace slots per CTA: the A operands, plus the output descriptor
# when the TMA-store epilogue re-dimensions it per routed group.
moe_desc_slots = num_a_operands * 2 + n_tma_outputs

if use_acc_overlap and any(_w != epi_n for _, _w in _epi_subtile_spans(epi_cols_per_mma_m, epi_n)):
    raise NotImplementedError(f"{__name__}: acc overlap reverses subtiles by index, which needs a uniform drain width")


SCHED_STAGES = 2
SCHED_SLOT_WORDS = 8

USE_PDL = True
EPI_SMEM_STAGES = 2
EPI_SYNC_BAR_ID = 1
TMEM_ALLOC_BARRIER_ID = 2


@cute.jit
def _moe_auto_swizzle_w(group_rows, n, k, nt_n):
    """N-super-block width for one routed group, resolved per group.

    Same rule as the dense path, but the "M side" is THIS group's token slice, not the
    whole token tensor: block along the shorter of (group tokens, expert weight), capped
    by what L2 can hold onto. A group spanning one m-tile makes both orders identical.
    """
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    rows = cutlass.Int64(group_rows)
    if cutlass.min(rows, n) * row_bytes <= budget and rows <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _b_collector_op(mi):
    """B is identical across the M sub-blocks (only A's address advances), so the
    first MMA fills the B collector and the rest read it back instead of
    re-fetching the same operand from SMEM."""
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
    num_experts: cutlass.Int32,
    num_groups: cutlass.Int32,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    # @@INJECT_MOE_KERNEL_MA_PARAMS@@
    # @@INJECT_MOE_KERNEL_MSFA_PARAMS@@
    # @@INJECT_KERNEL_TAP_PARAMS@@
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
    # @@TMA_STORE_ONLY:BEGIN@@
    # @@INJECT_KERNEL_TMA_C_PARAMS@@
    # @@TMA_STORE_ONLY:END@@
) -> None:
    # @@INJECT_AB_DESC_LISTS@@

    # @@INJECT_MOE_MA_LIST@@
    # @@INJECT_MOE_MSFA_LIST@@
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
    gridx = cute.arch.grid_dim()[0]

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

    cluster_linear_init = bidx // cluster_m

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
    sf_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
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
    tma_a_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_a_operands)
    ]
    tma_sfa_desc_smem_list = [
        cutlass.Array(
            cutlass.Int64,
            TENSOR_MAP_QWORDS,
            space=cutlass.AddressSpace.smem,
            alignment=128,
        )
        for _ in range(num_a_operands)
    ]

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

    acc_empty_count = num_epilogue_warps * 2
    cta_group = 2
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_count = cluster_size // cta_group
    else:
        ab_empty_count = (cluster_m // cta_group) + cluster_n - 1
    sched_empty_count = 1 + 1 + num_epilogue_warps
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
                nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(acc_stages):
            if elect_one:
                nvvm.mbarrier_init(acc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(acc_empty_mbar_ptr.subview(i), acc_empty_count)
        for i in range(SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(sched_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sched_empty_mbar_ptr.subview(i), sched_empty_count)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    ab_only_copy_bytes = (num_a_operands * sA_bytes + num_b_operands * sB_bytes) * 2
    sf_only_copy_bytes = (num_a_operands * sfa_smem_bytes + num_b_operands * sfb_smem_bytes) * 2

    pair_n_size = cgrp_tile_mnk[1] // cluster_n
    # Per-CTA output rows one MMA-M block covers. The pair splits M, so this is
    # the per-CTA mma_inst_m — half the instruction's hardware M.
    epi_rows_per_mma_m = cta_tile_mnk[0] // num_mma_m
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    nvvm.barrier_cluster_wait()
    nvvm.barrier_cta_sync(0)

    # @@INJECT_TAP_PTRS@@

    vsize = epi_chunk_elems

    M = m
    N = n
    clusters_along_n = cute.ceil_div(cutlass.Int32(N), cgrp_tile_mnk[1])
    num_k_tiles = cute.ceil_div(k, cgrp_tile_mnk[2])
    first_token_arr = cutlass.make_array_view(first_token_offset)

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        full_warp_mask = 0xFFFFFFFF
        shfl_idx_clamp = 0x1F
        shfl_up_clamp = 0
        lane = cute.arch.lane_idx()
        gemm_s = cutlass.Int32(M)
        sched_stage = cutlass.Int32(0)
        sched_empty_phase = cutlass.Int32(1)
        linear_idx = cutlass.Int32(cluster_linear_init)
        start_linear_idx = cutlass.Int32(0)
        total_tiles = cutlass.Int32(0)
        start_sf_block_m = cutlass.Int32(0)
        total_sf_blocks_m = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_tile_valid = cutlass.Int32(1)
        cached_next_end = cutlass.Int32(0)
        if lane + 1 < num_groups:
            cached_next_end = cutlass.Int32(first_token_arr[lane + 1])
        else:
            cached_next_end = gemm_s
        tile_lower_bound = nvvm.shfl_sync(full_warp_mask, cached_next_end, 1, shfl_up_clamp, nvvm.Shfl.UP)
        cached_next_begin = cutlass.Int32(0)
        if lane != 0:
            cached_next_begin = tile_lower_bound

        while is_tile_valid != 0:
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
                    if next_end_group < num_groups:
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

                    group_m = cached_group_end - cached_group_begin
                    total_tiles = cute.ceil_div(group_m, cgrp_tile_mnk[0]) * clusters_along_n
                    total_sf_blocks_m = cute.ceil_div(group_m, 128)
                    prefix_tiles = total_tiles
                    prefix_sf = total_sf_blocks_m
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
                    start_sf_block_m += prefix_sf - total_sf_blocks_m
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
                        start_sf_block_m = nvvm.shfl_sync(
                            full_warp_mask,
                            start_sf_block_m,
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
                        if group_end_idx < num_groups:
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
                            next_start_sf = start_sf_block_m + total_sf_blocks_m
                            start_sf_block_m = nvvm.shfl_sync(
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
                group_nt_m = total_tiles // clusters_along_n
                cluster_tile_m, coord_n = _moe_swizzle_tile(
                    local_linear_idx,
                    group_nt_m,
                    clusters_along_n,
                    _moe_auto_swizzle_w(group_nt_m * cgrp_tile_mnk[0], N, k, clusters_along_n),
                )
                coord_expert = group_idx % num_experts
                linear_idx += grid_num_clusters

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
                (slot.subview(6)).store(start_sf_block_m)
                (slot.subview(7)).store(group_idx)
                nvvm.mbarrier_arrive(sched_full_mbar_ptr.subview(sched_stage))

            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_empty_phase = sched_empty_phase ^ 1

    if warp_idx == tma_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ab_empty_phase_bit = cutlass.Int32(1)
        ab_iter = cutlass.Int32(0)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_n

        lane = tidx % 32
        block_linear = bidx + bidy * gridx
        cta_desc_base_list = [a_tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_a_operands)]
        a_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                cta_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_a_operands)
        ]
        sfa_desc_base_list = [
            a_tma_workspace.iterator.raw_ptr() + (block_linear * moe_desc_slots + num_a_operands + _ai) * TENSOR_MAP_QWORDS for _ai in range(num_a_operands)
        ]
        sfa_desc_tma_ptr_list = [
            cute.make_ptr(
                cutlass.Int64,
                sfa_desc_base_list[_ai].toint(),
                mem_space=cute.AddressSpace.generic,
            )
            for _ai in range(num_a_operands)
        ]
        sfa_block_bytes = 512 * (((k // block_size) + 3) // 4)
        previous_group_begin = cutlass.Int32(-1)
        if elect_one:
            for _ai in cutlass.range_constexpr(num_a_operands):
                _copy_tensormap_to_workspace(tma_a_descs[_ai].get_ptr(), tma_a_desc_smem_list[_ai])
            for _ai in cutlass.range_constexpr(num_a_operands):
                _copy_tensormap_to_workspace(tma_sfa_descs[_ai].get_ptr(), tma_sfa_desc_smem_list[_ai])
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
            start_sf_block_m = (slot.subview(6)).load()
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_group = tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                coord_n_per_cta = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
                coord_n_pair = tile_n * cgrp_tile_mnk[1] + n_rank * logical_cta_tile_n
                sfa_m_block = coord_m_group // 128
                sfb_n_block = coord_n_pair // 128

                if group_begin != previous_group_begin:
                    previous_group_begin = group_begin
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        _fence_tensormap_acquire(a_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if elect_one:
                            row_base = mA_list[_ai].iterator.raw_ptr().toint() + ((group_begin * a_stride_m_list[_ai] * ab_dtype.width) >> 3)
                            _replace_tensormap_global_address(tma_a_desc_smem_list[_ai], row_base)
                            _replace_tensormap_global_dim_1(tma_a_desc_smem_list[_ai], group_end - group_begin)
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (cta_desc_base_list[_ai] + lane).store((tma_a_desc_smem_list[_ai].subview(lane)).load())
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        _fence_tensormap_acquire(sfa_desc_tma_ptr_list[_ai])
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        if elect_one:
                            sfa_base = mSFA_list[_ai].iterator.raw_ptr().toint() + start_sf_block_m * sfa_block_bytes
                            _replace_tensormap_global_address(tma_sfa_desc_smem_list[_ai], sfa_base)
                            _replace_tensormap_global_dim_2(tma_sfa_desc_smem_list[_ai], cute.ceil_div(group_end - group_begin, 128))
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if lane < TENSOR_MAP_QWORDS:
                            (sfa_desc_base_list[_ai] + lane).store((tma_sfa_desc_smem_list[_ai].subview(lane)).load())
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

                    coord_k = k_tile_idx * cgrp_tile_mnk[2]
                    coord_sf_k = k_tile_idx * sf_tma_box_k

                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), ab_only_copy_bytes)
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), sf_only_copy_bytes)
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
                    if a_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_sfa_list[_ai].subview(sfa_smem_bytes * stage),
                                    sfa_desc_tma_ptr_list[_ai],
                                    (0, coord_sf_k, sfa_m_block, cutlass.Int32(0)),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    if b_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_sfb_list[_bj].subview(sfb_smem_bytes * stage),
                                    tma_sfb_descs[_bj].get_ptr(),
                                    (0, coord_sf_k, sfb_n_block, coord_expert),
                                    sf_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_b,
                                    group=nvvm.CTAGroup.CTA_2,
                                )

                    if a_data_issue:
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                    smem_a_list[_ai].subview(sA_elems * stage + _a_off * ab_packed_per_row),
                                    a_desc_tma_ptr_list[_ai],
                                    (coord_k, coord_m_group + _a_off, cutlass.Int32(0)),
                                    ab_full_mbar_ptr.subview(stage),
                                    [],
                                    multicast_mask=tma_mcast_mask_a,
                                    group=nvvm.CTAGroup.CTA_2,
                                )
                    if b_data_issue:
                        for _bj in cutlass.range_constexpr(num_b_operands):
                            sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                            if cutlass.const_expr(b_is_n_major):
                                for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(n_group * b_tma_group_elems * cgrp_tile_mnk[2]),
                                            tma_b_descs[_bj].get_ptr(),
                                            (
                                                coord_n_per_cta + n_group * b_tma_group_elems,
                                                coord_k,
                                                coord_expert,
                                            ),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=nvvm.CTAGroup.CTA_2,
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(_b_off * ab_packed_per_row),
                                        tma_b_descs[_bj].get_ptr(),
                                        (coord_k, coord_n_per_cta + _b_off, coord_expert),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=nvvm.CTAGroup.CTA_2,
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
            ab_full_phase_bit = cutlass.Int32(0)
            ab_iter = cutlass.Int32(0)
            acc_empty_phase_bit = cutlass.Int32(1)
            tile_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            sched_stage = cutlass.Int32(0)
            sched_full_phase = cutlass.Int32(0)
            acc_stage = cutlass.Int32(0)
            # fp4 packs its K-mode into the OMMA descriptor's 2-bit split field; fp8
            # keeps the MX descriptor's 1-bit one. Both are built once, outside the
            # loops — the fields depend only on j (the scale id within a word).
            if cutlass.const_expr(idesc_is_omma):
                idesc_by_j = [
                    cutlass.experimental.primitives.Tcgen05MxOmmaInstrDesc.build(
                        a_dtype=idesc_a_dtype,
                        b_dtype=idesc_b_dtype,
                        scale_format=sf_scale_format,
                        n_dim=mma_n_dim,
                        m_dim=mma_m_dim,
                        a_major=mma_a_major,
                        b_major=mma_b_major,
                        a_sf_id=j * sf_scales_per_inst,
                        b_sf_id=j * sf_scales_per_inst,
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]
            else:
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
                        k_dim=mma_k_dim_mode,
                    )
                    for j in range(sf_insts_per_atom)
                ]
            sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
            sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
            s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
            sfb_scale_ptrs = [nvvm.make_tmem_ptr(b, cutlass.Float32) for b in sfb_tmem_bases]
            # utccp destination per (MN-block, atom within the scale word). SFB
            # is atom-MAJOR across the N-blocks because ONE instruction walks
            # all of them; SFA is block-major because one instruction covers
            # exactly one 128-row block, so that word has to be contiguous.
            # Both collapse to the same addresses at a single block, and to
            # sm100's layout at word_atoms == 1.
            sfa_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfa_tmem_bases[i] + m * registers_per_block + a * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(num_mma_m)
                ]
                for i in range(num_a_operands)
            ]
            sfb_dst_ptrs = [
                [
                    [nvvm.make_tmem_ptr(sfb_tmem_bases[j] + (a * num_blocks_n + m) * registers_per_atom, cutlass.Float32) for a in range(word_atoms)]
                    for m in range(num_blocks_n)
                ]
                for j in range(num_b_operands)
            ]
            # Per-group TMA replacement changes the GMEM source, not these
            # invariant MMA-side SMEM descriptor roots.
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
                while not nvvm.mbarrier_try_wait_parity(
                    sched_full_mbar_ptr.subview(sched_stage),
                    sched_full_phase,
                    time_limit=10_000_000,
                ):
                    pass
                is_valid = (sched_storage.subview(sched_stage * SCHED_SLOT_WORDS).subview(3)).load()
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

                        desc_a_bases = [desc_a_roots[i].advance_start_address(sA_bytes * stage) for i in range(num_a_operands)]
                        desc_b_bases = [desc_b_roots[j].advance_start_address(sB_bytes * stage) for j in range(num_b_operands)]
                        desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_smem_bytes * stage) for i in range(num_a_operands)]
                        desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_smem_bytes * stage) for j in range(num_b_operands)]

                        # One SF word per group of MMAs, refreshed right before they
                        # read it. A word spans word_atoms consecutive K-atoms in SMEM.
                        while not nvvm.mbarrier_try_wait_parity(
                            sf_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass

                        for sf_word in cutlass.range_constexpr(num_sf_atoms):
                            for _bj in cutlass.range_constexpr(num_b_operands):
                                for block_n in cutlass.range_constexpr(num_blocks_n):
                                    for _a in cutlass.range_constexpr(word_atoms):
                                        if elect_one:
                                            nvvm.tcgen05_cp(
                                                s2t_shape,
                                                sfb_dst_ptrs[_bj][block_n][_a],
                                                desc_sfb_bases[_bj] + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * block_n),
                                                group=nvvm.CTAGroup.CTA_2,
                                                multicast=s2t_multicast,
                                            )
                            if cutlass.const_expr(sf_word == 0):
                                while not nvvm.mbarrier_try_wait_parity(
                                    ab_full_mbar_ptr.subview(stage),
                                    ab_full_phase_bit,
                                    time_limit=10_000_000,
                                ):
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
                                            for _a in cutlass.range_constexpr(word_atoms):
                                                if elect_one:
                                                    nvvm.tcgen05_cp(
                                                        s2t_shape,
                                                        sfa_dst_ptrs[_ai][mma_m][_a],
                                                        desc_sfa_bases[_ai]
                                                        + (sf_atom_desc_stride * (sf_word * word_atoms + _a) + sf_block_desc_stride * mma_m),
                                                        group=nvvm.CTAGroup.CTA_2,
                                                        multicast=s2t_multicast,
                                                    )
                                        # The M sub-block offset is a whole SMEM swizzle atom, so
                                        # the descriptor's swizzle phase is preserved. B and its SF
                                        # are shared; A's SF word block follows the M block.
                                        desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mma_m)
                                        if elect_one:
                                            _tcgen05_mma_block_scale(
                                                mma_block_scale_kind,
                                                nvvm.CTAGroup.CTA_2,
                                                acc_tmem_ptrs[gemm_i][mma_m],
                                                desc_a,
                                                desc_b,
                                                idesc_k,
                                                enable_input_d=scale_d,
                                                scale_a=sfa_dst_ptrs[_ai][mma_m][0],
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
                                group=nvvm.CTAGroup.CTA_2,
                            )
                        ab_iter += 1

                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            multicast_mask=pair_mask,
                            group=nvvm.CTAGroup.CTA_2,
                        )
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
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
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
                group=nvvm.CTAGroup.CTA_2,
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
                if elect_one:
                    nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
                sched_stage += 1
                if sched_stage == SCHED_STAGES:
                    sched_stage = cutlass.Int32(0)
                    sched_full_phase = sched_full_phase ^ 1

            if cutlass.const_expr(USE_PDL):
                nvvm.griddepcontrol("launch_dependents")

            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_2)
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
        is_valid = cutlass.Int32(1)
        sched_stage = cutlass.Int32(0)
        sched_full_phase = cutlass.Int32(0)

        # @@EPILOGUE_SETUP:BEGIN@@
        row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        # @@TMA_STORE_ONLY:BEGIN@@
        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)
        # The routed output is a single (1, S, N) tensor, so the batch coord is fixed.
        tile_l = cutlass.Int32(0)
        epi_block_linear = bidx + bidy * gridx
        d_desc_base_list = [
            a_tma_workspace.iterator.raw_ptr() + (epi_block_linear * moe_desc_slots + num_a_operands * 2 + _di) * TENSOR_MAP_QWORDS
            for _di in range(n_tma_outputs)
        ]
        d_desc_ptr_list = [cute.make_ptr(cutlass.Int64, _b.toint(), mem_space=cute.AddressSpace.generic) for _b in d_desc_base_list]
        previous_group_end = cutlass.Int32(-1)
        if warp_idx == 0:
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
            slot = sched_storage.subview(sched_stage * SCHED_SLOT_WORDS)
            tile_m = (slot.subview(1)).load()
            tile_n = (slot.subview(2)).load()
            is_valid = (slot.subview(3)).load()
            group_begin = (slot.subview(4)).load()
            group_end = (slot.subview(5)).load()
            group_idx = (slot.subview(7)).load()
            if elect_one:
                nvvm.mbarrier_arrive(sched_empty_mbar_ptr.subview(sched_stage))
            sched_stage += 1
            if sched_stage == SCHED_STAGES:
                sched_stage = cutlass.Int32(0)
                sched_full_phase = sched_full_phase ^ 1

            if is_valid != 0:
                coord_m_tile = group_begin + tile_m * cgrp_tile_mnk[0] + m_rank * cta_tile_mnk[0]
                # @@TMA_STORE_ONLY:BEGIN@@
                # Re-dimension D to this group's last row so the hardware clips the
                # ragged tail; the base stays put, so the store coords are global.
                if warp_idx == 0:
                    if group_end != previous_group_end:
                        previous_group_end = group_end
                        # One drain retires the in-flight stores of EVERY descriptor,
                        # so it does not repeat per output.
                        nvvm.cp_async_bulk_wait_group(0, read=True)
                        for _di in cutlass.range_constexpr(n_tma_outputs):
                            _scratch = tma_c_desc_smem.subview(_di * TENSOR_MAP_QWORDS)
                            _fence_tensormap_acquire(d_desc_ptr_list[_di])
                            if elect_one:
                                _replace_tensormap_global_dim_1(_scratch, group_end)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            if lane < TENSOR_MAP_QWORDS:
                                (d_desc_base_list[_di] + lane).store((_scratch.subview(lane)).load())
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                        _fence_tensormap_release()
                # @@TMA_STORE_ONLY:END@@
                # @@EPILOGUE_DRAIN:BEGIN@@
                coord_n_c = tile_n * cgrp_tile_mnk[1] + n_rank * pair_n_size

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
                                    nvvm.mbarrier_arrive(
                                        nvvm.mapa(acc_empty_mbar_ptr.subview(acc_stage), pair_leader_rank),
                                        scope=nvvm.MemScope.CLUSTER,
                                        relaxed=True,
                                    )

                        if use_acc_overlap and mi * subtile_cnt + subtile_idx == acc_overlap_subtiles - 1:
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
                        if row_active and row < group_end:
                            for j in cutlass.range_constexpr(subtile_w // vsize):
                                col_j = col + j * vsize
                                if col_j + vsize <= N:
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

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _host(
    problem_size: tuple,
    first_token_offset: cute.Tensor,
    a_tma_workspace: cute.Tensor,
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
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_desc_dtype,
                global_dims=[k_sym, m, 1],
                global_strides=[
                    a_stride_m * ab_dtype.width // 128,
                    a_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[cta_tile_mnk[2], cta_tile_mnk[0] // a_mcast_slices, 1],
                swizzle=ab_tma_swizzle,
                tma_format=ab_tma_format,
            )
        )
    tma_b_desc_list = []
    for _b_idx, _b_op in enumerate(_b_operands):
        b_stride_n, b_stride_k, b_stride_l = _b_stride_sets[_b_idx]
        if cutlass.const_expr(b_is_n_major):
            tma_b_desc_list.append(
                _tma.create_tensor_map_tiled(
                    global_address=_b_op.iterator.toint(),
                    dtype=ab_tma_desc_dtype,
                    global_dims=[n, k_sym, num_experts],
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
                    global_dims=[k_sym, n, num_experts],
                    global_strides=[
                        b_stride_n * ab_dtype.width // 128,
                        b_stride_l * ab_dtype.width // 128,
                    ],
                    box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
                    swizzle=ab_tma_swizzle,
                    tma_format=ab_tma_format,
                )
            )
    rest_k = ((k_sym // block_size) + 3) // 4
    rest_m = (m + 127) // 128 + num_groups
    rest_n = (n + 127) // 128
    tma_sfa_desc_list = []
    for _sfa_op in _sfa_operands:
        sfa_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfa_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_m, 1),
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
    tma_sfb_desc_list = []
    for _sfb_op in _sfb_operands:
        sfb_fp16_tensor = cute.make_tensor(
            cute.recast_ptr(_sfb_op.iterator, dtype=cutlass.Float16),
            cute.make_layout(
                (256, rest_k, rest_n, num_experts),
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

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_shape = (grid_num_clusters * cluster_m, cluster_n, 1)
    _kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        cutlass.Int32(num_experts),
        cutlass.Int32(num_groups),
        first_token_offset,
        a_tma_workspace,
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        # @@INJECT_MOE_HOST_MA_PASS@@
        # @@INJECT_MOE_HOST_MSFA_PASS@@
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
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
    # K tails are supported: the K loop is ceil_div and the TMA descriptor's global K
    # extent makes a partial box HW zero-filled. The only real K rule is the 16-byte
    # TMA contiguous-extent one, already gated by _tma_alignment_reject.
    sym_k = cute.sym_int64()
    # Packed K extent: same reasoning as sym_k -- no CTA-tile multiple is required.
    sym_kp = cute.sym_int64()
    sym_e = cute.sym_int64()
    sym_g = cute.sym_int64()

    def _make_fake_a():
        return make_fake_compact_tensor(
            a_fake_dtype,
            (sym_m, sym_kp, 1),
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_kp, sym_e),
            stride_order=(0, 1, 2) if b_is_n_major else (1, 0, 2),
            assumed_align=16,
        )

    # SF reaches the kernel as a base pointer only; the host rebuilds the
    # F8_128x4 view from problem_size. Modes 0/1 and all strides carry no
    # contract; mode 2 keeps its literal plane count (1 for sfa, sym_e for sfb).
    def _make_fake_sfa():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), 1),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return cute.runtime.make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), sym_e),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    fake_first_token_offset = make_fake_compact_tensor(
        offset_cutlass_dtype,
        (sym_g,),
        stride_order=(0,),
        assumed_align=offset_cutlass_dtype.width // 8,
    )
    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    grid_ctas = grid_num_clusters * cluster_m * cluster_n
    fake_a_tma_workspace = make_fake_compact_tensor(
        cutlass.Int64,
        (grid_ctas * moe_desc_slots * 16,),
        stride_order=(0,),
        assumed_align=128,
    )

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
    return cute.compile(
        _host,
        problem_size,
        fake_first_token_offset,
        fake_a_tma_workspace,
        # @@INJECT_COMPILE_AB_PASS@@
        # @@INJECT_COMPILE_TAP_PASS@@
        # @@INJECT_COMPILE_AUX_PASS@@
        # @@TMA_STORE_ONLY:BEGIN@@
        # @@INJECT_COMPILE_TMA_C_PASS@@
        # @@TMA_STORE_ONLY:END@@
        stream=_fake_stream,
        options=frost_compile_options,
    )
