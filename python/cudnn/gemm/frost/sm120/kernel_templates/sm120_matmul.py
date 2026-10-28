# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm120 (GeForce/consumer Blackwell, CC 12.0) GEMM kernel: persistent + CLC dynamic scheduler (2-stage ring) + warp-level MMA."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
from cudnn.gemm.frost.tile_helpers import l2_swizzle_tile as _l2_swizzle_tile
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass._mlir_helpers.vector as _cvec
from cutlass import apply_swizzle as _apply_smem_swizzle
import cutlass
from cudnn.frost.compiled_cache import compile_cached as _compile_cached
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# @@INJECT_TILE_CONSTANTS@@


CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+; supported on sm_120).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for cross-warp sync of the 8 compute warps around TMA stores.
EPI_SYNC_BAR_ID = 1

WARPS_M = cta_tile_mnk[0] // (mma_size_m * mma_inst_shape_mnk[0])
WARPS_N = cta_tile_mnk[1] // (mma_size_n * mma_inst_shape_mnk[1])
NUM_COMPUTE_WARPS = WARPS_M * WARPS_N

TMA_WARP_ID = NUM_COMPUTE_WARPS
SCHEDULER_WARP_ID = NUM_COMPUTE_WARPS + 1
NUM_WARPS = threads_per_cta // 32

NUM_CLC_CONSUMER_WARPS = NUM_COMPUTE_WARPS + 2

EPI_REG_COUNT = 232
PROD_REG_COUNT = 24
assert NUM_COMPUTE_WARPS % 4 == 0, "the producer warps must form whole warpgroups"


_ELEM_BITS = ab_dtype.width
_ELEM_BYTES = _ELEM_BITS // 8
_ELEMS_16B = 16 // _ELEM_BYTES
_K_BLK_ELEMS = (32 * 8) // _ELEM_BITS
_NUM_K_BLOCKS = (cta_tile_mnk[2] * _ELEM_BITS) // (32 * 8)
_CTA_K_ELEMS = cta_tile_mnk[2]

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

_AB_SMEM_SWIZZLE_BYTES = _CTA_K_ELEMS * _ELEM_BYTES
_AB_SW_BBITS = (_AB_SMEM_SWIZZLE_BYTES // 16).bit_length() - 1
_AB_SWIZZLE = cutlass.Swizzle(_AB_SW_BBITS, 4, 3)
# Epilogue staging tile swizzle — matches the s64b TMA-store descriptor.
_EPI_SWIZZLE = cutlass.Swizzle(2, 4, 3)

_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

_RPL_SUBPASSES = 2 if _M_FRAGS % 2 == 0 else 1
_RPL_SLABS = _M_FRAGS // _RPL_SUBPASSES


def _rpl_swz(r):
    """Return the row-major staging swizzle key."""
    rr = r % 8
    return ((rr & 3) << 1) | (rr >> 2)


_STG_EPI_BYTES = 4 * _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS
# One SMEM tile per DISTINCT operand per stage (multi-GEMM holds several A / B tiles).
_AB_STAGE_BYTES = (num_a_operands * cta_tile_mnk[0] + num_b_operands * cta_tile_mnk[1]) * _CTA_K_ELEMS * _ELEM_BYTES + 16
if not stg_epi_prefunded:
    ab_stages = ab_stages - -(-_STG_EPI_BYTES // _AB_STAGE_BYTES)


_PTX_AB_TAG = {
    cutlass.BFloat16: "bf16",
    cutlass.Float16: "f16",
    cutlass.Float8E4M3FN: "e4m3",
    cutlass.Float8E5M2: "e5m2",
    cutlass.Int8: "s8",
}
assert mma_a_dtype in _PTX_AB_TAG and mma_b_dtype in _PTX_AB_TAG, f"unsupported sm120 MMA input dtypes: {mma_a_dtype} x {mma_b_dtype}"
_MMA_SHAPE = "m16n8k16" if _ELEM_BITS == 16 else "m16n8k32"
_MMA_C_TAG = "f32" if mma_c_dtype == cutlass.Float32 else "s32"
_MMA_PTX = (
    f"mma.sync.aligned.{_MMA_SHAPE}.row.col"
    f".{_MMA_C_TAG}.{_PTX_AB_TAG[mma_a_dtype]}.{_PTX_AB_TAG[mma_b_dtype]}.{_MMA_C_TAG} "
    "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
)


@cute.jit
def _mma_16x8_k32b(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    """One warp-wide mma.sync on a (16, 8, 32-byte-K) fragment."""
    return cute.arch.inline_ptx(
        _MMA_PTX,
        write_only_types=[mma_c_dtype, mma_c_dtype, mma_c_dtype, mma_c_dtype],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch."""
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(ab_dtype.width) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


# @@SPLITK_ONLY:BEGIN@@
from cudnn.gemm.frost.kernel_templates.split_k_reduction_epilogue_fusion import (
    SPLITK_REDUCE_THREADS,
    SPLITK_REDUCE_TILE_M,
    _splitk_reduce_kernel,
    splitk_reduce_tile_n,
)


@cute.jit
def _splitk_epilogue(vec_f32, row, col_j, tile_l, M, N, vsize, taps, strides, aux):
    """Apply the split-K reducer epilogue."""
    # @@INJECT_SPLITK_EPILOGUE_BINDINGS@@
    # @@INJECT_REDUCE_AUX_VIEWS@@

    # @@INJECT_REDUCE_EPILOGUE@@


# @@SPLITK_ONLY:END@@


@cute.kernel
def _kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    # @@INJECT_KERNEL_TAP_PARAMS@@
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
    # @@SPLITK_ONLY:BEGIN@@
    mSplitK_partials: cute.Tensor,
    # @@SPLITK_ONLY:END@@
) -> None:
    # @@INJECT_AB_DESC_LISTS@@

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    if warp_idx == TMA_WARP_ID:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, ab_stages, space=cutlass.AddressSpace.smem)

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

    smem_stg_epi = cutlass.Array(
        mma_c_dtype,
        _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    if warp_idx == 0:
        for i in range(ab_stages):
            if elect_one:
                nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), NUM_COMPUTE_WARPS)
        for i in range(CLC_SCHED_STAGES):
            if elect_one:
                nvvm.mbarrier_init(clc_full_mbar_ptr.subview(i), 1)
            if elect_one:
                nvvm.mbarrier_init(clc_empty_mbar_ptr.subview(i), NUM_CLC_CONSUMER_WARPS)
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0)

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    num_tma_copy_bytes = num_a_operands * sA_bytes + num_b_operands * sB_bytes

    # @@INJECT_TAP_PTRS@@

    # @@SPLITK_ONLY:BEGIN@@
    gSplitK_partials_ptr = mSplitK_partials.iterator.raw_ptr()
    # @@SPLITK_ONLY:END@@

    VEC_BYTES = vec_bytes_epi
    vsize = epi_chunk_elems

    M = m
    N = n

    if warp_idx >= NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)

    if warp_idx == SCHEDULER_WARP_ID:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        sched_iter = cutlass.Int32(0)
        clc_empty_phase = cutlass.Int32(1)
        clc_full_phase = cutlass.Int32(0)
        is_valid_sched = cutlass.Int32(1)
        while is_valid_sched != 0:
            stage = sched_iter % CLC_SCHED_STAGES
            if stage == 0 and sched_iter != 0:
                clc_empty_phase = clc_empty_phase ^ 1
                clc_full_phase = clc_full_phase ^ 1

            while not nvvm.mbarrier_try_wait_parity(clc_empty_mbar_ptr.subview(stage), clc_empty_phase, time_limit=10_000_000):
                pass

            if elect_one:
                nvvm.mbarrier_arrive_expect_tx(clc_full_mbar_ptr.subview(stage), 16)
            if elect_one:
                cute_clc.issue_clc_query(
                    clc_full_mbar_cute_base + stage,
                    clc_response_ptr_base + stage,
                    multicast=False,
                )

            while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(stage), clc_full_phase, time_limit=10_000_000):
                pass

            _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid_sched = vld

            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(clc_empty_mbar_ptr.subview(stage))

            sched_iter += 1

    # -- TMA producer warp ----------------------------------------------------
    if warp_idx == TMA_WARP_ID:
        swizzle_w = _auto_swizzle_w(m, n, k, gridy)
        init_tile_m, init_tile_n = _l2_swizzle_tile(bidx, bidy, gridx, gridy, swizzle_w)
        init_tile_l = bidz
        num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
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
            coord_m = tile_m * cgrp_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_mnk[1]
            # Split-K: grid z carries batch*S
            if cutlass.const_expr(split_k_slices > 1):
                batch_tile_l = tile_l // split_k_slices
                split_idx = cutlass.Int64(tile_l % split_k_slices)
                k_tiles_per_split = num_k_tiles // split_k_slices
                k_tiles_remainder = num_k_tiles % split_k_slices
                k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
            else:
                batch_tile_l = tile_l
                k_begin = cutlass.Int64(0)
                k_end = num_k_tiles
            if cutlass.const_expr(matmul_a_batch == 1):
                tile_l_a = cutlass.Int32(0)
            else:
                tile_l_a = batch_tile_l
            if cutlass.const_expr(matmul_b_batch == 1):
                tile_l_b = cutlass.Int32(0)
            else:
                tile_l_b = batch_tile_l

            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_empty_phase_bit = ab_empty_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                    pass

                coord_k = k_tile_idx * cta_tile_mnk[2]
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                for _ai in cutlass.range_constexpr(num_a_operands):
                    if cutlass.const_expr(a_is_m_major):
                        for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_a_list[_ai].subview(sA_elems * stage + m_group * a_tma_group_elems * _CTA_K_ELEMS),
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_m + m_group * a_tma_group_elems, coord_k, tile_l_a),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cta_global(
                                smem_a_list[_ai].subview(sA_elems * stage),
                                tma_a_descs[_ai].get_ptr(),
                                (coord_k, coord_m, tile_l_a),
                                ab_full_mbar_ptr.subview(stage),
                            )
                for _bj in cutlass.range_constexpr(num_b_operands):
                    if cutlass.const_expr(b_is_n_major):
                        for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_b_list[_bj].subview(sB_elems * stage + n_group * b_tma_group_elems * _CTA_K_ELEMS),
                                    tma_b_descs[_bj].get_ptr(),
                                    (coord_n + n_group * b_tma_group_elems, coord_k, tile_l_b),
                                    ab_full_mbar_ptr.subview(stage),
                                )
                    else:
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cta_global(
                                smem_b_list[_bj].subview(sB_elems * stage),
                                tma_b_descs[_bj].get_ptr(),
                                (coord_k, coord_n, tile_l_b),
                                ab_full_mbar_ptr.subview(stage),
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
            tile_m, tile_n = _l2_swizzle_tile(m_idx, n_idx, gridx, gridy, swizzle_w)
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(clc_empty_mbar_ptr.subview(consumer_stage))
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

    # -- Compute warps: mma.sync mainloop + epilogue --------------------------
    if warp_idx < NUM_COMPUTE_WARPS:
        nvvm.setmaxregister(EPI_REG_COUNT, nvvm.SetMaxRegisterAction.INCREASE)
        swizzle_w = _auto_swizzle_w(m, n, k, gridy)
        init_tile_m, init_tile_n = _l2_swizzle_tile(bidx, bidy, gridx, gridy, swizzle_w)
        init_tile_l = bidz
        num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        lane = tidx % 32
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        warp_row = warp_idx % WARPS_M
        warp_col = warp_idx // WARPS_M

        a_ldm_row = (lane % 8) + 8 * ((lane // 8) % 2)
        a_ldm_col16 = lane // 16
        b_ldm_pair_row = (lane % 8) + 8 * (lane // 16)
        b_ldm_pair_col16 = (lane // 8) % 2
        # B x2 tail: one n-frag (rows 0-7 x two 16B cols; lanes 16-31 unused).
        b_ldm_tail_row = lane % 8
        b_ldm_tail_col16 = (lane // 8) % 2
        at_ldm_k = (lane % 8) + 8 * (lane // 16)
        at_ldm_m8 = (lane // 8) % 2
        bt_ldm_k = (lane % 8) + 8 * ((lane // 8) % 2)
        bt_ldm_n8 = lane // 16

        acc_list = [cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16) for _g in range(num_gemms)]

        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)
        _red_part = cute.make_rmem_tensor(red_part_slots, cutlass.Float32)
        red_part_seed(_red_part)
        while is_valid != 0:
            coord_m = tile_m * cgrp_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_mnk[1]
            if cutlass.const_expr(split_k_slices > 1):
                split_idx = cutlass.Int64(tile_l % split_k_slices)
                k_tiles_per_split = num_k_tiles // split_k_slices
                k_tiles_remainder = num_k_tiles % split_k_slices
                k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
            else:
                k_begin = cutlass.Int64(0)
                k_end = num_k_tiles

            for _g in cutlass.range_constexpr(num_gemms):
                _acc_g = acc_list[_g]
                for _z in cutlass.range_constexpr(_ACC_REGS):
                    _acc_g[_z] = mma_c_dtype(0)

            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                sA_ptrs = [smem_a_list[_ai].subview(sA_elems * stage).data_ptr() for _ai in range(num_a_operands)]
                sB_ptrs = [smem_b_list[_bj].subview(sB_elems * stage).data_ptr() for _bj in range(num_b_operands)]

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    kb_base = k_blk * _K_BLK_ELEMS
                    a_frags_list = []
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_ptr = sA_ptrs[_ai]
                        a_frags = []
                        if cutlass.const_expr(a_is_m_major and _ELEM_BITS == 8):
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_m = warp_row * _WARP_TILE_M + mf * 16
                                a_off = (
                                    (a_m // a_tma_group_elems) * (a_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + lane) * a_tma_group_elems
                                    + a_m % a_tma_group_elems
                                )
                                a_frags.append(
                                    nvvm.ldmatrix(
                                        _apply_smem_swizzle(sA_ptr + a_off, _AB_SWIZZLE),
                                        4,
                                        nvvm.MMALayout.COL,
                                        shape=nvvm.LoadShape.M16N16,
                                        src_format=nvvm.LoadSrcFormat.B8,
                                    )
                                )
                        elif cutlass.const_expr(a_is_m_major):
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_m = warp_row * _WARP_TILE_M + mf * 16 + at_ldm_m8 * 8
                                a_off = (
                                    (a_m // a_tma_group_elems) * (a_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + at_ldm_k) * a_tma_group_elems
                                    + a_m % a_tma_group_elems
                                )
                                a_frags.append(
                                    nvvm.ldmatrix(
                                        _apply_smem_swizzle(sA_ptr + a_off, _AB_SWIZZLE),
                                        4,
                                        nvvm.MMALayout.COL,
                                    )
                                )
                        else:
                            for mf in cutlass.range_constexpr(_M_FRAGS):
                                a_row = warp_row * _WARP_TILE_M + mf * 16 + a_ldm_row
                                a_off = a_row * _CTA_K_ELEMS + kb_base + a_ldm_col16 * _ELEMS_16B
                                a_frags.append(
                                    nvvm.ldmatrix(
                                        _apply_smem_swizzle(sA_ptr + a_off, _AB_SWIZZLE),
                                        4,
                                        nvvm.MMALayout.ROW,
                                    )
                                )
                        a_frags_list.append(a_frags)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_ptr = sB_ptrs[_bj]
                        b_frags = []
                        if cutlass.const_expr(b_is_n_major and _ELEM_BITS == 8):
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_n = warp_col * _WARP_TILE_N + npair * 16
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + lane) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.COL,
                                    shape=nvvm.LoadShape.M16N16,
                                    src_format=nvvm.LoadSrcFormat.B8,
                                )
                                b_frags.append((bv[0], bv[2]))
                                b_frags.append((bv[1], bv[3]))
                        elif cutlass.const_expr(b_is_n_major):
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_n = warp_col * _WARP_TILE_N + npair * 16 + bt_ldm_n8 * 8
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + bt_ldm_k) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.COL,
                                )
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_n = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8
                                b_off = (
                                    (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_ELEMS)
                                    + (kb_base + bt_ldm_k) * b_tma_group_elems
                                    + b_n % b_tma_group_elems
                                )
                                bt = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    2,
                                    nvvm.MMALayout.COL,
                                )
                                b_frags.append((bt[0], bt[1]))
                        else:
                            for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                                b_row = warp_col * _WARP_TILE_N + npair * 16 + b_ldm_pair_row
                                b_off = b_row * _CTA_K_ELEMS + kb_base + b_ldm_pair_col16 * _ELEMS_16B
                                bv = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bv[0], bv[1]))
                                b_frags.append((bv[2], bv[3]))
                            if cutlass.const_expr(_N_FRAGS % 2 == 1):
                                b_row = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8 + b_ldm_tail_row
                                b_off = b_row * _CTA_K_ELEMS + kb_base + b_ldm_tail_col16 * _ELEMS_16B
                                bt = nvvm.ldmatrix(
                                    _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                    2,
                                    nvvm.MMALayout.ROW,
                                )
                                b_frags.append((bt[0], bt[1]))

                        # Every GEMM fed by this B operand (trace-time selection).
                        for _g in cutlass.range_constexpr(num_gemms):
                            if cutlass.const_expr(gemm_b_idx[_g] == _bj):
                                _acc_g = acc_list[_g]
                                a_frags = a_frags_list[gemm_a_idx[_g]]
                                for mf in cutlass.range_constexpr(_M_FRAGS):
                                    av = a_frags[mf]
                                    for nf in cutlass.range_constexpr(_N_FRAGS):
                                        b0, b1 = b_frags[nf]
                                        _o = (mf * _N_FRAGS + nf) * 4
                                        _acc_g[_o:4] = _mma_16x8_k32b(
                                            av[0],
                                            av[1],
                                            av[2],
                                            av[3],
                                            b0,
                                            b1,
                                            _acc_g[_o + 0],
                                            _acc_g[_o + 1],
                                            _acc_g[_o + 2],
                                            _acc_g[_o + 3],
                                        )

                nvvm.bar_warp_sync(0xFFFFFFFF)
                cute.arch.fence_proxy("async.shared", space="cta")
                if elect_one:
                    nvvm.mbarrier_arrive(ab_empty_mbar_ptr.subview(stage))
                ab_iter += 1

            # -- Epilogue: accumulators are already in registers ------------------

            # @@INJECT_AUX_VIEWS@@

            _stg_stage = smem_stg_epi.subview(warp_idx * _STG_EPI_WARP_ELEMS)
            if cutlass.const_expr(not row_per_lane_drain):
                for mf in cutlass.range_constexpr(_M_FRAGS):
                    for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                        _nf0 = grp * _STG_EPI_GROUP_FRAGS
                        _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                        _seg_list = []
                        for _g in cutlass.range_constexpr(num_gemms):
                            _acc_g = acc_list[_g]
                            for b in cutlass.range_constexpr(_grp_frags):
                                _o = (mf * _N_FRAGS + _nf0 + b) * 4
                                _s_off = b * _STG_EPI_BATCH_STRIDE + lane * _STG_EPI_LANE_QUAD
                                (_stg_stage.data_ptr() + _s_off).store(_acc_g[_o:4], alignment=16)
                            nvvm.bar_warp_sync(0xFFFFFFFF)
                            _seg_list.append((_stg_stage.data_ptr() + lane_mod4 * _STG_EPI_BATCH_STRIDE + lane_div4 * 16).load(alignment=16, count=16))
                            if cutlass.const_expr(_g + 1 < num_gemms):
                                nvvm.bar_warp_sync(0xFFFFFFFF)  # every lane has its segment before the buffer is reused
                        _lane_active = True if _grp_frags == _STG_EPI_GROUP_FRAGS else lane_mod4 < _grp_frags
                        if _lane_active:
                            for half in cutlass.range_constexpr(2):
                                row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4
                                row = coord_m + row_in_cta
                                if row < M:
                                    _row_list = []
                                    for _g in cutlass.range_constexpr(num_gemms):
                                        _seg = _seg_list[_g]
                                        _row = cutlass.Array(mma_c_dtype, 8, alignment=16)
                                        for sj in cutlass.range_constexpr(4):
                                            _row[2 * sj] = _seg[4 * sj + 2 * half]
                                            _row[2 * sj + 1] = _seg[4 * sj + 2 * half + 1]
                                        _row_list.append(_row)
                                    for sv in cutlass.range_constexpr(8 // _STG_V):
                                        col = coord_n + warp_col * _WARP_TILE_N + (_nf0 + lane_mod4) * 8 + sv * _STG_V
                                        col_j = col
                                        if col_j + vsize <= N:
                                            c_rmem_vecs = []
                                            for _g in cutlass.range_constexpr(num_gemms):
                                                _vec = _row_list[_g][sv * _STG_V : _STG_V]
                                                if cutlass.const_expr(acc_widen_to_fp32):
                                                    _pf = _vec.to(cutlass.Float32)
                                                    c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                                else:
                                                    c_rmem_vecs.append(_vec)
                                            vec_f32 = c_rmem_vecs[0]

                                            # @@INJECT_STG_VEC_BINDINGS@@

                                            # @@INJECT_EPILOGUE@@
                        nvvm.bar_warp_sync(0xFFFFFFFF)
            else:
                for slab in cutlass.range_constexpr(_RPL_SLABS):
                    for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                        _nf0 = grp * _STG_EPI_GROUP_FRAGS
                        _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                        _run_list = []
                        for _g in cutlass.range_constexpr(num_gemms):
                            _acc_g = acc_list[_g]
                            _run = cutlass.Array(mma_c_dtype, 32, alignment=16)
                            for _sp in cutlass.range_constexpr(_RPL_SUBPASSES):
                                _mf = slab * _RPL_SUBPASSES + _sp
                                for b in cutlass.range_constexpr(_grp_frags):
                                    for _hr in cutlass.range_constexpr(2):
                                        _rr = lane_div4 + 8 * _hr
                                        _off = _rr * 32 + ((2 * b + lane_mod4 // 2) ^ _rpl_swz(_rr)) * 4 + (lane_mod4 % 2) * 2
                                        _o = (_mf * _N_FRAGS + _nf0 + b) * 4 + 2 * _hr
                                        (_stg_stage.data_ptr() + _off).store(_acc_g[_o:2], alignment=8)
                                nvvm.bar_warp_sync(0xFFFFFFFF)
                                # LDS.128: lanes 16*_sp .. 16*_sp + 15 take this fragment's rows.
                                if lane // 16 == _sp:
                                    _r = lane % 16
                                    for _j in cutlass.range_constexpr(2 * _grp_frags):
                                        _run[_j * 4 : 4] = (_stg_stage.data_ptr() + _r * 32 + (_j ^ _rpl_swz(_r)) * 4).load(alignment=16, count=4)
                                nvvm.bar_warp_sync(0xFFFFFFFF)  # the next sub-pass / GEMM reuses the buffer
                            _run_list.append(_run)
                        # Lanes 16..31 own no row when the slab is a single 16-row fragment.
                        _rows_owned = True if _RPL_SUBPASSES == 2 else lane < 16
                        if _rows_owned:
                            row_in_cta = warp_row * _WARP_TILE_M + slab * (16 * _RPL_SUBPASSES) + lane
                            row = coord_m + row_in_cta
                            if cutlass.const_expr(rpl_bounded_rows):
                                _row_live = True
                            else:
                                _row_live = row < M
                            if _row_live:
                                for _jv in cutlass.range_constexpr((_grp_frags * 8) // vsize):
                                    col = coord_n + warp_col * _WARP_TILE_N + _nf0 * 8 + _jv * vsize
                                    col_j = col
                                    if col_j + vsize <= N:
                                        c_rmem_vecs = []
                                        for _g in cutlass.range_constexpr(num_gemms):
                                            _vec = _run_list[_g][_jv * vsize : vsize]
                                            if cutlass.const_expr(acc_widen_to_fp32):
                                                _pf = _vec.to(cutlass.Float32)
                                                c_rmem_vecs.append(_pf + cutlass.full_like(_pf, 0.0))
                                            else:
                                                c_rmem_vecs.append(_vec)
                                        vec_f32 = c_rmem_vecs[0]

                                        # @@INJECT_STG_VEC_BINDINGS@@

                                        # @@INJECT_EPILOGUE@@
            red_part_flush_tile(_red_part, lane, M, N, (tile_l,), _red_part_args)

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
            tile_m, tile_n = _l2_swizzle_tile(m_idx, n_idx, gridx, gridy, swizzle_w)
            tile_l = l_idx
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                nvvm.mbarrier_arrive(clc_empty_mbar_ptr.subview(consumer_stage))

            tile_iter += 1

        red_part_flush_cta(_red_part, lane, M, N, (tile_l,), _red_part_args)
        if cutlass.const_expr(USE_PDL and split_k_slices == 1):
            if warp_idx == 0:
                if elect_one:
                    nvvm.griddepcontrol("launch_dependents")


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    problem_size: tuple,
    # @@INJECT_HOST_AB_PARAMS@@
    # @@INJECT_HOST_TAP_PARAMS@@
    # @@INJECT_HOST_AUX_PARAMS@@
    # @@SPLITK_ONLY:BEGIN@@
    splitk_partials: cute.Tensor,
    # @@SPLITK_ONLY:END@@
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
                )
            )

    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_shape = (num_tile_m_host, num_tile_n_host, batch * split_k_slices)
    _kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        # @@INJECT_HOST_TAP_PASS@@
        # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        # @@INJECT_HOST_AUX_PASS@@
        # @@SPLITK_ONLY:BEGIN@@
        splitk_partials,
        # @@SPLITK_ONLY:END@@
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        use_pdl=USE_PDL,
        stream=stream,
        min_blocks_per_mp=1,
    )

    # @@SPLITK_ONLY:BEGIN@@
    _splitk_reduce_kernel(
        m,
        n,
        batch,
        splitk_partials,
        (
            # @@INJECT_HOST_TAP_PASS@@
        ),
        (
            # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        ),
        (
            # @@INJECT_HOST_AUX_PASS@@
        ),
        _splitk_epilogue,
        split_k_slices,
        splitk_reduce_elems,
        USE_PDL,
    ).launch(
        grid=(
            (m + SPLITK_REDUCE_TILE_M - 1) // SPLITK_REDUCE_TILE_M,
            (n + splitk_reduce_tile_n(split_k_slices, splitk_reduce_elems) - 1) // splitk_reduce_tile_n(split_k_slices, splitk_reduce_elems),
            batch,
        ),
        block=(SPLITK_REDUCE_THREADS, 1, 1),
        use_pdl=USE_PDL,
        stream=stream,
    )
    # @@SPLITK_ONLY:END@@


@lru_cache(maxsize=None)
def compile() -> Callable:
    out_vec_elems = vec_bytes_epi // (cd_dtype.width // 8)
    ab_stride_elems = 16 // (ab_dtype.width // 8)
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
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
    # @@SPLITK_ONLY:BEGIN@@
    sym_partials_elems = cute.sym_int64()
    fake_splitk_partials = make_fake_tensor(cutlass.Float32, (sym_partials_elems,), stride=(1,), assumed_align=16)
    # @@SPLITK_ONLY:END@@
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return _compile_cached(
        _host,
        problem_size,
        # @@INJECT_COMPILE_AB_PASS@@
        # @@INJECT_COMPILE_TAP_PASS@@
        # @@INJECT_COMPILE_AUX_PASS@@
        # @@SPLITK_ONLY:BEGIN@@
        fake_splitk_partials,
        # @@SPLITK_ONLY:END@@
        stream=_fake_stream,
        options=frost_compile_options,
        # persistent object across processes (cudnn.frost.compiled_cache); the digest of THIS source is the key
        cache_key=globals().get("FROST_SOURCE_DIGEST"),
        symbol="frost_gemm",
    )
