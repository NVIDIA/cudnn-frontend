# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm103 CTA_1 **block-scaled** GEMM kernel: persistent + CLC dynamic scheduler.

Computes ``C = (descale_a ⊙ A) @ (descale_b ⊙ B)`` where A/B are FP4 e2m1
(packed 2-per-byte), dequantized by a per-block scale factor along K inside the
MMA. FP4-only pipeline: any of the e4m3 / e8m0 / e5m3 scale dtypes at either
K-block (the two axes are orthogonal; e5m3 and e4m3-at-block-32 need SM 10.7+).

sm103's fp4 UTCOMMA instruction K width is 48 BYTES (96 fp4 elements) vs
sm100's 32 B (64 elements). 48 does not divide the 128-B swizzled SMEM line, so
the pipeline follows the CUTLASS sm103 blockscaled design:

  * the logical K-tile is lcm(128, 48) = 384 B = 768 fp4 elements (8 K=96 MMAs)
    but the **AB pipeline stage is ONE 128-B-K chunk** (3 stages per K-tile) —
    the MMA starts as soon as the first 128 B land and releases each chunk as
    its last reader issues;
  * MMA k-blocks 2 and 5 straddle a chunk (= stage) boundary — every MMA
    passes a **circular SMEM descriptor**: leading-dim-mode bit 52 set and the
    *next* stage's start address packed into desc bits [16:32);
  * **SF rides its own pipeline**: a dedicated SF-load warp fills an
    (sf_stages)-deep ring at 12-SFs-per-row group granularity (4 groups per
    K-tile at VS16, 2 at VS32); the MMA warp utccp's each group into TMEM
    right before the group's first MMA and releases the SF stage;
  * the instruction descriptor is ``Tcgen05MxOmmaInstrDesc`` with ``k_dim=1``
    (the K=96 mode) and ``sparsity_version=0``;
  * a whole K-tile's SF is resident in TMEM (sf_k byte-cols per 128-row block
    per operand); each MMA j reads scales_per_inst SF bytes from byte offset
    ``spi*j`` — the TMEM pointer is the word-aligned col and the idesc
    ``a_sf_id/b_sf_id`` picks the byte.

Warp layout (8 warps × 32 = 256 threads/CTA):
  warps 0–3 : epilogue (warp 0 also allocates TMEM)  — setmaxnreg.inc 216
  warp  4   : MMA driver (every CTA runs MMA — no pair structure)  — setmaxnreg.dec 88
  warp  5   : TMA producer, A/B data (128-B chunk stages)  — setmaxnreg.dec 24
  warp  6   : CLC scheduler (leader CTA issues queries; every CTA waits + reads + arrives empty)  — setmaxnreg.dec 24
  warp  7   : TMA producer, SFA/SFB (own sf ring)  — setmaxnreg.dec 24
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

# K-walk-derived schedule points: slot c must be FULL before its first reader
# (a straddling MMA reads slot c via the circular next-desc) and is RELEASED
# after its last reader issues. Derived from the injected walk tables so the
# 8-MMA interleave never hardcodes j indices.
_WAIT_SLOTS_AT: dict[int, list[int]] = {}
_seen_slot = -1
for _j in range(num_kblocks):
    _hi = mma_next_chunk_by_j[_j]
    if _hi > _seen_slot:
        _WAIT_SLOTS_AT[_j] = list(range(_seen_slot + 1, _hi + 1))
        _seen_slot = _hi
_RELEASE_SLOTS_AT: dict[int, list[int]] = {}
for _c in range(chunks_per_ktile):
    _last = max(_j for _j in range(num_kblocks) if mma_chunk_by_j[_j] == _c or mma_next_chunk_by_j[_j] == _c)
    _RELEASE_SLOTS_AT.setdefault(_last, []).append(_c)
del _j, _c, _hi, _last, _seen_slot

# The idesc only varies with the SF byte-id, which cycles with this period
# (2 at VS16, 4 at VS32) — build one descriptor per cycle position, not per j.
_SF_ID_PERIOD = next(p for p in range(1, num_kblocks + 1) if all(sf_id_by_j[j] == sf_id_by_j[j % p] for j in range(num_kblocks)))

# The MMA-side slot chain below is written out for exactly 3 chunks per K-tile.
if chunks_per_ktile != 3:
    raise NotImplementedError(f"sm103 block-scale: the MMA slot chain is written out for exactly 3 chunks per K-tile, got {chunks_per_ktile}")


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


def _sm103_circular_mma_desc_base(current_desc):
    """Precompute invariant fields of an SM103 K=96 circular MMA SMEM desc:
    set the circular leading-dim mode bit (52) and clear the next-chunk
    address field (bits [16:32))."""
    return nvvm.Tcgen05SmemDesc((current_desc | (1 << 52)) & -4294901761)


def _sm103_circular_mma_next_bits(next_desc):
    """Next-chunk SMEM start address (16-B units) packed for desc bits [16:32)."""
    return (next_desc & 0xFFFF) << 16


def _sm103_make_circular_mma_desc(current_desc_circular, phase_k16, next_addr_bits):
    """Patch the in-chunk K phase (16-B units) + next-chunk bits before OMMA."""
    desc_with_phase = current_desc_circular.advance_start_address(phase_k16 * 16)
    return nvvm.Tcgen05SmemDesc(desc_with_phase | next_addr_bits)


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
    sf_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 216
    mma_reg_count = 88
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
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    sf_full_mbar_ptr = cutlass.Array(cutlass.Int64, sf_stages, space=cutlass.AddressSpace.smem)
    sf_empty_mbar_ptr = cutlass.Array(cutlass.Int64, sf_stages, space=cutlass.AddressSpace.smem)
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

    # An AB stage holds ONE 128-B-K chunk per operand; SF has its own ring of
    # 12-SF-per-row groups.
    smem_a_list = [
        cutlass.Array(
            ab_dtype,
            a_chunk_packed_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            b_chunk_packed_elems * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]
    smem_sfa_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfa_group_bytes * sf_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_sfb_list = [
        cutlass.Array(
            cutlass.Uint8,
            sfb_group_bytes * sf_stages,
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
    num_consumer_warps_per_cta = 8
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
        for i in range(sf_stages):
            if elect_one:
                nvvm.mbarrier_init(sf_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(sf_empty_mbar_ptr.subview(i), ab_empty_count)
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

    a_chunk_bytes = a_chunk_packed_elems * (ab_dtype.width // 8)
    b_chunk_bytes = b_chunk_packed_elems * (ab_dtype.width // 8)
    num_tma_ab_chunk_bytes = num_a_operands * a_chunk_bytes + num_b_operands * b_chunk_bytes
    num_tma_sf_group_bytes = num_a_operands * sfa_group_bytes + num_b_operands * sfb_group_bytes

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
        ab_stage_cur = cutlass.Int32(0)  # incremental ring walk — no div in the loop
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
                for _kc in cutlass.range_constexpr(chunks_per_ktile):
                    stage = ab_stage_cur

                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_k = k_tile_idx * cta_tile_mnk[2] + _kc * ab_tma_box_k_elems
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_ab_chunk_bytes)

                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_stage = smem_a_list[_ai].subview(a_chunk_packed_elems * stage)
                        tma_a_desc = tma_a_descs[_ai]
                        if cutlass.const_expr(a_mcast_slices > 1):
                            _a_rows = cta_tile_mnk[0] // a_mcast_slices
                            _a_row_elems = a_chunk_packed_elems // cta_tile_mnk[0]
                            if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sA_stage.subview(n_rank * _a_rows * _a_row_elems),
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
                                            sA_stage.subview(_a_idx * _a_rows * _a_row_elems),
                                            tma_a_desc.get_ptr(),
                                            (coord_k, coord_m_per_cta + _a_idx * _a_rows, tile_l_a),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_a,
                                            group=nvvm.CTAGroup.CTA_1,
                                        )
                        elif cutlass.const_expr(multicast_a):
                            if n_rank == 0:
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
                        sB_stage = smem_b_list[_bj].subview(b_chunk_packed_elems * stage)
                        tma_b_desc = tma_b_descs[_bj]
                        if cutlass.const_expr(b_mcast_slices > 1):
                            _b_rows = cta_tile_mnk[1] // b_mcast_slices
                            _b_row_elems = b_chunk_packed_elems // cta_tile_mnk[1]
                            if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(m_rank * _b_rows * _b_row_elems),
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
                                            sB_stage.subview(_b_idx * _b_rows * _b_row_elems),
                                            tma_b_desc.get_ptr(),
                                            (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=nvvm.CTAGroup.CTA_1,
                                        )
                        elif cutlass.const_expr(multicast_b):
                            if m_rank == 0:
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

                    _wrap = ab_stage_cur == (ab_stages - 1)
                    ab_stage_cur = ab_stage_cur + 1 - _wrap * ab_stages
                    ab_empty_phase_bit = ab_empty_phase_bit ^ _wrap

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

        # (ab_stage_cur, ab_empty_phase_bit) is the next-to-use pair — the
        # incremental advance already flipped the phase on wrap.
        tail_stage = ab_stage_cur
        tail_phase = ab_empty_phase_bit
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(ab_stages):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == ab_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    if warp_idx == sf_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        sf_empty_phase_bit = cutlass.Int32(1)
        sf_stage_cur = cutlass.Int32(0)  # incremental ring walk — no div in the loop
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_sf = cutlass.Int32(0)
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
            sfa_m_block = coord_m_per_cta // 128
            sfb_n_block = coord_n_per_cta // 128

            for k_tile_idx in range(num_k_tiles):
                for _grp in cutlass.range_constexpr(sf_groups_per_ktile):
                    stage = sf_stage_cur

                    while not nvvm.mbarrier_try_wait_parity(sf_empty_mbar_ptr.subview(stage), sf_empty_phase_bit, time_limit=10_000_000):
                        pass

                    coord_sf_k = k_tile_idx * sf_tma_box_k + _grp * sf_atoms_per_group
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(sf_full_mbar_ptr.subview(stage), num_tma_sf_group_bytes)

                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sSFA_stage = smem_sfa_list[_ai].subview(sfa_group_bytes * stage)
                        tma_sfa_desc = tma_sfa_descs[_ai]
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
                        sSFB_stage = smem_sfb_list[_bj].subview(sfb_group_bytes * stage)
                        tma_sfb_desc = tma_sfb_descs[_bj]
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

                    _wrap = sf_stage_cur == (sf_stages - 1)
                    sf_stage_cur = sf_stage_cur + 1 - _wrap * sf_stages
                    sf_empty_phase_bit = sf_empty_phase_bit ^ _wrap

            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_sf = clc_full_phase_sf ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_sf,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            sf_raw_m = m_idx >> _preferred_cluster_m_shift
            sf_raw_n = n_idx >> _preferred_cluster_n_shift
            sf_nt_m = gridx >> _preferred_cluster_m_shift
            sf_nt_n = gridy >> _preferred_cluster_n_shift
            if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                    sf_raw_m = m_idx >> _fallback_cluster_m_shift
                    sf_raw_n = n_idx >> _fallback_cluster_n_shift
                    sf_nt_m = gridx >> _fallback_cluster_m_shift
                    sf_nt_n = gridy >> _fallback_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                sf_raw_m,
                sf_raw_n,
                sf_nt_m,
                sf_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1

        tail_stage = sf_stage_cur
        tail_phase = sf_empty_phase_bit
        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            for _ in range(sf_stages):
                while not nvvm.mbarrier_try_wait_parity(sf_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == sf_stages:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1

    if warp_idx == mma_warp_id:
        nvvm.setmaxregister(mma_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
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
        mma_slot_stage = cutlass.Int32(0)  # ring pos of the tile's first chunk
        mma_slot_phase = cutlass.Int32(0)
        mma_sf_stage = cutlass.Int32(0)
        mma_sf_phase = cutlass.Int32(0)
        acc_empty_phase_bit = cutlass.Int32(1)
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_mma = cutlass.Int32(0)
        acc_stage = cutlass.Int32(0)
        # One OMMA idesc per K-block j: k_dim=1 selects the K=96 mode
        # (sparsity_version stays 0 for dense sm103); a_sf_id/b_sf_id pick the
        # SF byte within the word-aligned TMEM col the scale pointer names.
        idesc_cycle = [
            cutlass.experimental.primitives.Tcgen05MxOmmaInstrDesc.build(
                a_dtype=cutlass.Float4E2M1FN,
                b_dtype=cutlass.Float4E2M1FN,
                scale_format=sf_scale_format,
                n_dim=mma_n_dim,
                m_dim=mma_m_dim,
                a_major=mma_a_major,
                b_major=mma_b_major,
                a_sf_id=sf_id_by_j[j],
                b_sf_id=sf_id_by_j[j],
                k_dim=1,
                sparsity_version=0,
            )
            for j in range(_SF_ID_PERIOD)
        ]
        s2t_shape, s2t_multicast = nvvm.S2TCopyMode.S2T_32x128b_WARPX4
        # Only the per-operand bases stay live; scale / utccp-dst TMEM pointers
        # are one immediate add away and computed at each use site.
        sfa_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfa_col_bases[i]) for i in range(num_a_operands)]
        sfb_tmem_bases = [(base_row_id << 16) | (base_col_id_root + sfb_col_bases[j]) for j in range(num_b_operands)]
        # Descriptor metadata and allocation bases are invariant. The SM103
        # circular current/next slot addresses remain runtime values below.
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

            # CLC consume coords for THIS tile's successor are read at tile
            # END — flip + peek here so the response poll resolves during
            # the K-loop.
            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_mma = clc_full_phase_mma ^ 1
            clc_peek = nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_mma, time_limit=1)

            while not nvvm.mbarrier_try_wait_parity(
                acc_empty_mbar_ptr.subview(acc_stage),
                acc_empty_phase_bit,
                time_limit=10_000_000,
            ):
                pass

            is_mma_leader = elect_one
            if cutlass.const_expr(use_acc_overlap):
                acc_base_col = base_col_id_root + (tile_iter % 2) * acc_stage_stride
            else:
                acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            # One accumulator per (gemm, M block); M block mi sits
            # epi_cols_per_mma_m columns further into its GEMM's region.
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
                # The K-tile's three ring slots — branchless incremental walk
                # (no div/mod on the MMA dispatch path).
                _wrap1 = mma_slot_stage == (ab_stages - 1)
                slot_stage_1 = mma_slot_stage + 1 - _wrap1 * ab_stages
                slot_phase_1 = mma_slot_phase ^ _wrap1
                _wrap2 = slot_stage_1 == (ab_stages - 1)
                slot_stage_2 = slot_stage_1 + 1 - _wrap2 * ab_stages
                slot_phase_2 = slot_phase_1 ^ _wrap2
                slot_stages = [mma_slot_stage, slot_stage_1, slot_stage_2]
                slot_phases = [mma_slot_phase, slot_phase_1, slot_phase_2]

                # Per-slot MMA descriptors + circular halves (address math
                # only — built before the data lands, consumed after the wait).
                # A gains an M sub-block axis: the circular descriptor's base AND
                # its next-chunk address both shift by the same whole SMEM swizzle
                # atom, so the wrap lands on the next chunk of the SAME M rows.
                desc_a_slots = [
                    [
                        [
                            desc_a_roots[i].advance_start_address(a_chunk_bytes * slot_stages[c]).advance_start_address(a_smem_m_step_bytes * mi)
                            for c in range(chunks_per_ktile)
                        ]
                        for mi in range(num_mma_m)
                    ]
                    for i in range(num_a_operands)
                ]
                desc_b_slots = [
                    [desc_b_roots[j].advance_start_address(b_chunk_bytes * slot_stages[c]) for c in range(chunks_per_ktile)] for j in range(num_b_operands)
                ]
                desc_a_circ = [[[_sm103_circular_mma_desc_base(d) for d in blk] for blk in op] for op in desc_a_slots]
                desc_a_next = [[[_sm103_circular_mma_next_bits(d) for d in blk] for blk in op] for op in desc_a_slots]
                desc_b_circ = [[_sm103_circular_mma_desc_base(d) for d in row] for row in desc_b_slots]
                desc_b_next = [[_sm103_circular_mma_next_bits(d) for d in row] for row in desc_b_slots]

                # Rotated schedule (matches the cuBLAS sm103 SASS): MMA j-1
                # is dispatched FIRST, then j's sf-utccp / ab-slot waits run
                # while the queued OMMAs drain the tensor pipe — the
                # PHASECHK wait latency hides behind TC work instead of
                # gating the next dispatch.
                for _kj in cutlass.range_constexpr(num_kblocks):
                    if cutlass.const_expr(_kj > 0):
                        _pj = _kj - 1
                        _kc = mma_chunk_by_j[_pj]
                        _kn = mma_next_chunk_by_j[_pj]
                        _ph = mma_phase16_by_j[_pj]
                        idesc_k = idesc_cycle[_pj % _SF_ID_PERIOD]
                        for g in cutlass.range_constexpr(num_gemms):
                            _ai = gemm_a_idx[g]
                            _bj = gemm_b_idx[g]
                            desc_b = _sm103_make_circular_mma_desc(desc_b_circ[_bj][_kc], _ph, desc_b_next[_bj][_kn])
                            for mi in cutlass.range_constexpr(num_mma_m):
                                # B and its SF are shared by every M block; A's SF word block
                                # follows the M block (SF words are one per 128 rows, packed
                                # M-fastest, one utccp 128x4 atom apart -- registers_per_ATOM, which
                                # is NOT registers_per_block once a K=96 MMA needs more
                                # than 4 scales (nvfp4: 6). The utccp destination below is
                                # the same stride; both read one name so they cannot drift.
                                desc_a = _sm103_make_circular_mma_desc(desc_a_circ[_ai][mi][_kc], _ph, desc_a_next[_ai][mi][_kn])
                                if is_mma_leader:
                                    _tcgen05_mma_block_scale(
                                        mma_block_scale_kind,
                                        nvvm.CTAGroup.CTA_1,
                                        acc_tmem_ptrs[g][mi],
                                        desc_a,
                                        desc_b,
                                        idesc_k,
                                        enable_input_d=scale_d,
                                        scale_a=nvvm.make_tmem_ptr(sfa_tmem_bases[_ai] + sfa_mma_col_off_by_j[_pj] + mi * registers_per_atom, cutlass.Float32),
                                        scale_b=nvvm.make_tmem_ptr(sfb_tmem_bases[_bj] + sfb_mma_col_off_by_j[_pj], cutlass.Float32),
                                        scale_vec_size=scale_vec_size,
                                        b_collector_op=_b_collector_op(mi),
                                    )
                        scale_d = cutlass.Boolean(True)
                        for _rs in _RELEASE_SLOTS_AT.get(_pj, []):
                            if is_mma_leader:
                                nvvm.tcgen05_commit(
                                    ab_empty_mbar_ptr.subview(slot_stages[_rs]),
                                    multicast_mask=ab_empty_arrive_mask,
                                    group=nvvm.CTAGroup.CTA_1,
                                )

                    # utccp the group feeding MMAs [_kj, _kj + mmas_per_sf_group)
                    # from the SF ring into the K-tile-resident TMEM region.
                    if cutlass.const_expr(_kj % mmas_per_sf_group == 0):
                        _grp = _kj // mmas_per_sf_group
                        while not nvvm.mbarrier_try_wait_parity(sf_full_mbar_ptr.subview(mma_sf_stage), mma_sf_phase, time_limit=10_000_000):
                            pass
                        desc_sfa_bases = [desc_sfa_roots[i].advance_start_address(sfa_group_bytes * mma_sf_stage) for i in range(num_a_operands)]
                        desc_sfb_bases = [desc_sfb_roots[j].advance_start_address(sfb_group_bytes * mma_sf_stage) for j in range(num_b_operands)]
                        for _at in cutlass.range_constexpr(sf_atoms_per_group):
                            for _ai in cutlass.range_constexpr(num_a_operands):
                                for _mh in cutlass.range_constexpr(num_blocks_m):
                                    if is_mma_leader:
                                        nvvm.tcgen05_cp(
                                            s2t_shape,
                                            nvvm.make_tmem_ptr(
                                                sfa_tmem_bases[_ai] + (num_blocks_m * (_grp * sf_atoms_per_group + _at) + _mh) * registers_per_atom,
                                                cutlass.Float32,
                                            ),
                                            desc_sfa_bases[_ai] + (sf_atom_desc_stride * _at + sf_group_block_desc_stride * _mh),
                                            group=nvvm.CTAGroup.CTA_1,
                                            multicast=s2t_multicast,
                                        )
                            for _bj in cutlass.range_constexpr(num_b_operands):
                                for _nh in cutlass.range_constexpr(num_blocks_n):
                                    if is_mma_leader:
                                        nvvm.tcgen05_cp(
                                            s2t_shape,
                                            nvvm.make_tmem_ptr(
                                                sfb_tmem_bases[_bj] + (num_blocks_n * (_grp * sf_atoms_per_group + _at) + _nh) * registers_per_atom,
                                                cutlass.Float32,
                                            ),
                                            desc_sfb_bases[_bj] + (sf_atom_desc_stride * _at + sf_group_block_desc_stride * _nh),
                                            group=nvvm.CTAGroup.CTA_1,
                                            multicast=s2t_multicast,
                                        )
                        if is_mma_leader:
                            nvvm.tcgen05_commit(
                                sf_empty_mbar_ptr.subview(mma_sf_stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=nvvm.CTAGroup.CTA_1,
                            )
                        _sfw = mma_sf_stage == (sf_stages - 1)
                        mma_sf_stage = mma_sf_stage + 1 - _sfw * sf_stages
                        mma_sf_phase = mma_sf_phase ^ _sfw

                    # Wait each slot right before its first reader.
                    for _ws in _WAIT_SLOTS_AT.get(_kj, []):
                        while not nvvm.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr.subview(slot_stages[_ws]),
                            slot_phases[_ws],
                            time_limit=10_000_000,
                        ):
                            pass

                # Tail of the rotation: the K-tile's last MMA + releases.
                _pj = num_kblocks - 1
                _kc = mma_chunk_by_j[_pj]
                _kn = mma_next_chunk_by_j[_pj]
                _ph = mma_phase16_by_j[_pj]
                idesc_k = idesc_cycle[_pj % _SF_ID_PERIOD]
                for g in cutlass.range_constexpr(num_gemms):
                    _ai = gemm_a_idx[g]
                    _bj = gemm_b_idx[g]
                    desc_b = _sm103_make_circular_mma_desc(desc_b_circ[_bj][_kc], _ph, desc_b_next[_bj][_kn])
                    for mi in cutlass.range_constexpr(num_mma_m):
                        # B and its SF are shared by every M block; A's SF word block
                        # follows the M block (SF words are one per 128 rows, packed
                        # M-fastest, one utccp 128x4 atom apart -- registers_per_ATOM, which
                        # is NOT registers_per_block once a K=96 MMA needs more
                        # than 4 scales (nvfp4: 6). The utccp destination below is
                        # the same stride; both read one name so they cannot drift.
                        desc_a = _sm103_make_circular_mma_desc(desc_a_circ[_ai][mi][_kc], _ph, desc_a_next[_ai][mi][_kn])
                        if is_mma_leader:
                            _tcgen05_mma_block_scale(
                                mma_block_scale_kind,
                                nvvm.CTAGroup.CTA_1,
                                acc_tmem_ptrs[g][mi],
                                desc_a,
                                desc_b,
                                idesc_k,
                                enable_input_d=scale_d,
                                scale_a=nvvm.make_tmem_ptr(sfa_tmem_bases[_ai] + sfa_mma_col_off_by_j[_pj] + mi * registers_per_atom, cutlass.Float32),
                                scale_b=nvvm.make_tmem_ptr(sfb_tmem_bases[_bj] + sfb_mma_col_off_by_j[_pj], cutlass.Float32),
                                scale_vec_size=scale_vec_size,
                                b_collector_op=_b_collector_op(mi),
                            )
                scale_d = cutlass.Boolean(True)
                for _rs in _RELEASE_SLOTS_AT.get(_pj, []):
                    if is_mma_leader:
                        nvvm.tcgen05_commit(
                            ab_empty_mbar_ptr.subview(slot_stages[_rs]),
                            multicast_mask=ab_empty_arrive_mask,
                            group=nvvm.CTAGroup.CTA_1,
                        )

                # Advance the tile's first-chunk ring position past slot 2.
                _wrap3 = slot_stage_2 == (ab_stages - 1)
                mma_slot_stage = slot_stage_2 + 1 - _wrap3 * ab_stages
                mma_slot_phase = slot_phase_2 ^ _wrap3

            if is_mma_leader:
                nvvm.tcgen05_commit(
                    acc_full_mbar_ptr.subview(acc_stage),
                    group=nvvm.CTAGroup.CTA_1,
                )

            # consumer_stage / phase flip + peek happened at tile head.
            if not clc_peek:
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
        # FP4 is K-major only; the TMA box covers ONE 128-B-K chunk (= one AB
        # pipeline stage; the kernel issues chunks_per_ktile boxes per K-tile).
        tma_a_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_a_op.iterator.toint(),
                dtype=ab_tma_desc_dtype,
                global_dims=[k_sym, m, a_batch],
                global_strides=[
                    a_stride_m * ab_dtype.width // 128,
                    a_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[ab_tma_box_k_elems, cta_tile_mnk[0] // a_mcast_slices, 1],
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
                box_dims=(256, sf_atoms_per_group, sfa_tma_box_mn, 1),
                stride_order=(0, 1, 2, 3),
                swizzle=_tma.TensorMapSwizzle.none,
            )
        )
    tma_b_desc_list = []
    tma_sfb_desc_list = []
    for _b_op, _sfb_op in zip(_b_operands, _sfb_operands):
        tma_b_desc_list.append(
            _tma.create_tensor_map_tiled(
                global_address=_b_op.iterator.toint(),
                dtype=ab_tma_desc_dtype,
                global_dims=[k_sym, n, b_batch],
                global_strides=[
                    b_stride_n * ab_dtype.width // 128,
                    b_stride_l * ab_dtype.width // 128,
                ],
                box_dims=[ab_tma_box_k_elems, cta_tile_mnk[1] // b_mcast_slices, 1],
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
                box_dims=(256, sf_atoms_per_group, sfb_tma_box_mn, 1),
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
            stride_order=(1, 0, 2),
            assumed_align=16,
        )

    def _make_fake_b():
        return make_fake_compact_tensor(
            b_fake_dtype,
            (sym_n, sym_kp, sym_b_l),
            stride_order=(1, 0, 2),
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

    # Operands are always K-major here, so the k stride is the unit stride and never reaches TMA.
    sym_a_stride_m = cute.sym_int64(divisibility=ab_stride_elems)
    sym_a_stride_k = cute.sym_int64()
    sym_a_stride_l = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_n = cute.sym_int64(divisibility=ab_stride_elems)
    sym_b_stride_k = cute.sym_int64()
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
