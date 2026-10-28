# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Sm120 block scale matmul."""

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
from cutlass.cutlass_dsl import T as _T
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects.nvvm import BlockScaleFormat as _BlockScaleFormat
from cutlass._mlir.dialects.nvvm import MMABlockScaleKind as _MMABlockScaleKind
from cutlass._mlir.dialects.nvvm import MMATypes as _MMATypes
from cutlass._mlir.dialects.nvvm import ScaleVecSize as _ScaleVecSize
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


_NATIVE_BITS = ab_tma_desc_dtype.width  # 4 (fp4) or 8 (fp8)
_PACK = 8 // _NATIVE_BITS  # native elements per packed byte
_CTA_K_ELEMS = cta_tile_mnk[2]  # native K elements per tile
_CTA_K_PACKED = ab_packed_per_row  # packed bytes per K row in SMEM (= cta_tile_k_bytes)
assert _CTA_K_PACKED * _PACK == _CTA_K_ELEMS, (_CTA_K_PACKED, _PACK, _CTA_K_ELEMS)
assert ab_dtype.width == 8, f"the packed carrier must be one byte wide, got {ab_dtype}"
_K_BLK_PACKED = 32
_K_BLK_ELEMS = _K_BLK_PACKED * _PACK
_NUM_K_BLOCKS = _CTA_K_PACKED // _K_BLK_PACKED
assert mma_inst_shape_mnk[2] == _K_BLK_ELEMS, (mma_inst_shape_mnk, _K_BLK_ELEMS)
_ELEMS_16B = 16  # packed bytes per 16-byte ldmatrix row

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

_AB_SMEM_SWIZZLE_BYTES = _CTA_K_PACKED
_AB_SW_BBITS = (_AB_SMEM_SWIZZLE_BYTES // 16).bit_length() - 1
_AB_SWIZZLE = cutlass.Swizzle(_AB_SW_BBITS, 4, 3)
assert (
    _AB_SMEM_SWIZZLE_BYTES == 128 and ab_tma_swizzle == _tma.TensorMapSwizzle.s128b
), "block-scale K rows are one 128-byte swizzle span (validate_block_scale_config_sm120)"
# Epilogue staging tile swizzle -- matches the s64b TMA-store descriptor.
_EPI_SWIZZLE = cutlass.Swizzle(2, 4, 3)

_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

assert ab_stages >= 1, f"{__name__}: the renderer emitted an empty AB ring"


_SF_WORDS = sf_tma_box_k  # 32-bit SF words per row per K-tile (= sf_k4)
_SF_SPI = sf_scales_per_inst  # scales one MMA consumes along K: 1 (1X), 2 (2X), 4 (4X)
_SF_ATOM_WORDS = 128  # 512 B
_SF_BLOCK_WORDS = _SF_WORDS * _SF_ATOM_WORDS  # one 128-row block of the stage
_SFA_STAGE_WORDS = sfa_smem_bytes // 4
_SFB_STAGE_WORDS = sfb_smem_bytes // 4
assert _SF_WORDS * 4 == _NUM_K_BLOCKS * _SF_SPI, (_SF_WORDS, _NUM_K_BLOCKS, _SF_SPI)
_SF_WORD_BY_KBLK = tuple((j * _SF_SPI) // 4 for j in range(_NUM_K_BLOCKS))
_SF_BYTE_BY_KBLK = tuple((j * _SF_SPI) % 4 for j in range(_NUM_K_BLOCKS))
_SFA_GROUPS = -(-_M_FRAGS // 2)
_SFB_GROUPS = -(-_N_FRAGS // 4)
_M_BLOCK_ALIGNED = cta_tile_mnk[0] % 128 == 0
_N_BLOCK_ALIGNED = cta_tile_mnk[1] % 128 == 0


_MMA_KIND = getattr(_MMABlockScaleKind, mma_block_scale_kind)
_MMA_SCALE_VEC = getattr(_ScaleVecSize, mma_scale_vec_size)
_MMA_SF_FORMAT = getattr(_BlockScaleFormat, mma_sf_format)
_MMA_A_TYPE = getattr(_MMATypes, mma_a_ptx_type)
_MMA_B_TYPE = getattr(_MMATypes, mma_b_ptx_type)
_MMA_SHAPE_ATTR = f"#nvvm.shape<m = 16, n = 8, k = {_K_BLK_ELEMS}>"
assert mma_c_dtype == cutlass.Float32, "the block-scaled warp MMA accumulates in fp32"


def _iv(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def _mma_16x8_bs(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, byte_id_a, thread_id_a, sfb, byte_id_b, thread_id_b):
    """One warp-wide block-scaled mma.sync on a (16, 8, 32-byte-K) fragment."""
    i16 = _T.i16()
    res = nvvm.mma_block_scale(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        _ir.Attribute.parse(_MMA_SHAPE_ATTR),
        _MMA_SCALE_VEC,
        _MMA_SF_FORMAT,
        _MMA_KIND,
        [_iv(a0), _iv(a1), _iv(a2), _iv(a3)],
        [_iv(b0), _iv(b1)],
        [_iv(c0), _iv(c1), _iv(c2), _iv(c3)],
        _iv(sfa),
        _arith.constant(i16, byte_id_a),
        _arith.constant(i16, thread_id_a),
        _iv(sfb),
        _arith.constant(i16, byte_id_b),
        _arith.constant(i16, thread_id_b),
        multiplicand_a_ptx_type=_MMA_A_TYPE,
        multiplicand_b_ptx_type=_MMA_B_TYPE,
    )
    return tuple(cutlass.Float32(_llvm.extractvalue(_T.f32(), res, [i])) for i in range(4))


@cute.jit
def _auto_swizzle_w(m, n, k, nt_n):
    """N-super-block width for the tile rasterization, resolved per launch."""
    if cutlass.const_expr(tile_swizzle_n > 0):
        return tile_swizzle_n
    budget = cutlass.Int64(swizzle_l2_budget_bytes)
    row_bytes = (cutlass.Int64(_NATIVE_BITS) * k) // 8
    cap = cutlass.max(budget // (row_bytes * cgrp_tile_mnk[1]), cutlass.Int64(1))
    w = cutlass.min(cutlass.Int64(nt_n), cap)
    if cutlass.min(m, n) * row_bytes <= budget and m <= n:
        w = cutlass.Int64(1)
    return cutlass.Int32(w)


def _sf_word_offset(r, r_in_block):
    """Return the scale-word offset within one shared-memory stage."""
    return (r_in_block // 128) * _SF_BLOCK_WORDS + (r_in_block % 32) * 4 + (r_in_block // 32) % 4


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
            nvvm.prefetch_tensormap(tma_sfa_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())
            nvvm.prefetch_tensormap(tma_sfb_descs[_j].get_ptr())

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
            cutlass.Int32,
            _SFA_STAGE_WORDS * ab_stages,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_sfb_list = [
        cutlass.Array(
            cutlass.Int32,
            _SFB_STAGE_WORDS * ab_stages,
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
    num_tma_copy_bytes = num_a_operands * (sA_bytes + sfa_smem_bytes) + num_b_operands * (sB_bytes + sfb_smem_bytes)

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
            sfa_m_block = coord_m // 128
            sfb_n_block = coord_n // 128
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
                coord_sf_k = k_tile_idx * sf_tma_box_k
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                for _ai in cutlass.range_constexpr(num_a_operands):
                    if cutlass.const_expr(a_is_m_major):
                        for m_group in cutlass.range_constexpr(cta_tile_mnk[0] // a_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_a_list[_ai].subview(sA_elems * stage + m_group * a_tma_group_elems * _CTA_K_PACKED),
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
                    if elect_one:
                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                            smem_sfa_list[_ai].subview(_SFA_STAGE_WORDS * stage),
                            tma_sfa_descs[_ai].get_ptr(),
                            (0, coord_sf_k, sfa_m_block, tile_l_a),
                            ab_full_mbar_ptr.subview(stage),
                        )
                for _bj in cutlass.range_constexpr(num_b_operands):
                    if cutlass.const_expr(b_is_n_major):
                        for n_group in cutlass.range_constexpr(cta_tile_mnk[1] // b_tma_group_elems):
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    smem_b_list[_bj].subview(sB_elems * stage + n_group * b_tma_group_elems * _CTA_K_PACKED),
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
                    if elect_one:
                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                            smem_sfb_list[_bj].subview(_SFB_STAGE_WORDS * stage),
                            tma_sfb_descs[_bj].get_ptr(),
                            (0, coord_sf_k, sfb_n_block, tile_l_b),
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

    # -- Compute warps: block-scaled mma.sync mainloop + epilogue ------------
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

        sf_lane_row = lane_div4 + 8 * (lane % 2)
        sfa_lane_sel = (lane // 2) % 2
        sfb_lane_sel = lane_mod4

        acc = cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16)

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

            if cutlass.const_expr(_M_BLOCK_ALIGNED):
                m_in_block = cutlass.Int32(0)
            else:
                m_in_block = coord_m % 128
            if cutlass.const_expr(_N_BLOCK_ALIGNED):
                n_in_block = cutlass.Int32(0)
            else:
                n_in_block = coord_n % 128
            sfa_word_off = []
            for g in cutlass.range_constexpr(_SFA_GROUPS):
                # Lane group 1 of the last (odd) pair duplicates frag 2g.
                mf_sel = cutlass.min(cutlass.Int32(2 * g) + sfa_lane_sel, cutlass.Int32(_M_FRAGS - 1))
                r = warp_row * _WARP_TILE_M + mf_sel * 16 + sf_lane_row
                sfa_word_off.append(_sf_word_offset(r, m_in_block + r))
            sfb_word_off = []
            for g in cutlass.range_constexpr(_SFB_GROUPS):
                nf_sel = cutlass.min(cutlass.Int32(4 * g) + sfb_lane_sel, cutlass.Int32(_N_FRAGS - 1))
                c = warp_col * _WARP_TILE_N + nf_sel * 8 + lane_div4
                sfb_word_off.append(_sf_word_offset(c, n_in_block + c))

            if cutlass.const_expr(split_k_slices > 1):
                split_idx = cutlass.Int64(tile_l % split_k_slices)
                k_tiles_per_split = num_k_tiles // split_k_slices
                k_tiles_remainder = num_k_tiles % split_k_slices
                k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
            else:
                k_begin = cutlass.Int64(0)
                k_end = num_k_tiles

            for _z in cutlass.range_constexpr(_ACC_REGS):
                acc[_z] = mma_c_dtype(0)

            for k_tile_idx in range(k_begin, k_end):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                sA_ptr = smem_a_list[0].subview(sA_elems * stage).data_ptr()
                sB_ptr = smem_b_list[0].subview(sB_elems * stage).data_ptr()
                sSFA_ptr = smem_sfa_list[0].subview(_SFA_STAGE_WORDS * stage).data_ptr()
                sSFB_ptr = smem_sfb_list[0].subview(_SFB_STAGE_WORDS * stage).data_ptr()

                sfa_words = []
                for g in cutlass.range_constexpr(_SFA_GROUPS):
                    _ws = []
                    for w in cutlass.range_constexpr(_SF_WORDS):
                        _ws.append((sSFA_ptr + sfa_word_off[g] + w * _SF_ATOM_WORDS).load())
                    sfa_words.append(_ws)
                sfb_words = []
                for g in cutlass.range_constexpr(_SFB_GROUPS):
                    _ws = []
                    for w in cutlass.range_constexpr(_SF_WORDS):
                        _ws.append((sSFB_ptr + sfb_word_off[g] + w * _SF_ATOM_WORDS).load())
                    sfb_words.append(_ws)

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    kb_base = k_blk * _K_BLK_PACKED
                    _sf_w = _SF_WORD_BY_KBLK[k_blk]
                    _sf_byte = _SF_BYTE_BY_KBLK[k_blk]
                    a_frags = []
                    if cutlass.const_expr(a_is_m_major):
                        for mf in cutlass.range_constexpr(_M_FRAGS):
                            a_m = warp_row * _WARP_TILE_M + mf * 16
                            a_off = (
                                (a_m // a_tma_group_elems) * (a_tma_group_elems * _CTA_K_PACKED)
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
                    else:
                        for mf in cutlass.range_constexpr(_M_FRAGS):
                            a_row = warp_row * _WARP_TILE_M + mf * 16 + a_ldm_row
                            a_off = a_row * _CTA_K_PACKED + kb_base + a_ldm_col16 * _ELEMS_16B
                            a_frags.append(
                                nvvm.ldmatrix(
                                    _apply_smem_swizzle(sA_ptr + a_off, _AB_SWIZZLE),
                                    4,
                                    nvvm.MMALayout.ROW,
                                )
                            )
                    b_frags = []
                    if cutlass.const_expr(b_is_n_major):
                        for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                            b_n = warp_col * _WARP_TILE_N + npair * 16
                            b_off = (
                                (b_n // b_tma_group_elems) * (b_tma_group_elems * _CTA_K_PACKED)
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
                    else:
                        for npair in cutlass.range_constexpr(_N_FRAG_PAIRS):
                            b_row = warp_col * _WARP_TILE_N + npair * 16 + b_ldm_pair_row
                            b_off = b_row * _CTA_K_PACKED + kb_base + b_ldm_pair_col16 * _ELEMS_16B
                            bv = nvvm.ldmatrix(
                                _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                4,
                                nvvm.MMALayout.ROW,
                            )
                            b_frags.append((bv[0], bv[1]))
                            b_frags.append((bv[2], bv[3]))
                        if cutlass.const_expr(_N_FRAGS % 2 == 1):
                            b_row = warp_col * _WARP_TILE_N + (_N_FRAGS - 1) * 8 + b_ldm_tail_row
                            b_off = b_row * _CTA_K_PACKED + kb_base + b_ldm_tail_col16 * _ELEMS_16B
                            bt = nvvm.ldmatrix(
                                _apply_smem_swizzle(sB_ptr + b_off, _AB_SWIZZLE),
                                2,
                                nvvm.MMALayout.ROW,
                            )
                            b_frags.append((bt[0], bt[1]))

                    for mf in cutlass.range_constexpr(_M_FRAGS):
                        av = a_frags[mf]
                        _sfa = sfa_words[mf // 2][_sf_w]
                        for nf in cutlass.range_constexpr(_N_FRAGS):
                            b0, b1 = b_frags[nf]
                            _sfb = sfb_words[nf // 4][_sf_w]
                            _o = (mf * _N_FRAGS + nf) * 4
                            acc[_o:4] = _mma_16x8_bs(
                                av[0],
                                av[1],
                                av[2],
                                av[3],
                                b0,
                                b1,
                                acc[_o + 0],
                                acc[_o + 1],
                                acc[_o + 2],
                                acc[_o + 3],
                                _sfa,
                                _sf_byte,
                                mf % 2,
                                _sfb,
                                _sf_byte,
                                nf % 4,
                            )

                nvvm.bar_warp_sync(0xFFFFFFFF)
                cute.arch.fence_proxy("async.shared", space="cta")
                if elect_one:
                    nvvm.mbarrier_arrive(ab_empty_mbar_ptr.subview(stage))
                ab_iter += 1

            # -- Epilogue: accumulators are already in registers ------------------

            # @@INJECT_AUX_VIEWS@@

            _stg_stage = smem_stg_epi.subview(warp_idx * _STG_EPI_WARP_ELEMS)
            for mf in cutlass.range_constexpr(_M_FRAGS):
                for grp in cutlass.range_constexpr(_STG_EPI_NGRP):
                    _nf0 = grp * _STG_EPI_GROUP_FRAGS
                    _grp_frags = min(_STG_EPI_GROUP_FRAGS, _N_FRAGS - _nf0)
                    # -- STS_128: reg-index-order dump, one batch per fragment --
                    for b in cutlass.range_constexpr(_grp_frags):
                        _o = (mf * _N_FRAGS + _nf0 + b) * 4
                        _s_off = b * _STG_EPI_BATCH_STRIDE + lane * _STG_EPI_LANE_QUAD
                        (_stg_stage.data_ptr() + _s_off).store(acc[_o:4], alignment=16)
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    # -- LDS: 16 contiguous elems = both row-halves of one frag --
                    _seg = (_stg_stage.data_ptr() + lane_mod4 * _STG_EPI_BATCH_STRIDE + lane_div4 * 16).load(alignment=16, count=16)
                    _lane_active = True if _grp_frags == _STG_EPI_GROUP_FRAGS else lane_mod4 < _grp_frags
                    if _lane_active:
                        for half in cutlass.range_constexpr(2):
                            row_in_cta = warp_row * _WARP_TILE_M + mf * 16 + half * 8 + lane_div4
                            row = coord_m + row_in_cta
                            if row < M:
                                _row = cutlass.Array(mma_c_dtype, 8, alignment=16)
                                for sj in cutlass.range_constexpr(4):
                                    _row[2 * sj] = _seg[4 * sj + 2 * half]
                                    _row[2 * sj + 1] = _seg[4 * sj + 2 * half + 1]
                                for sv in cutlass.range_constexpr(8 // _STG_V):
                                    col = coord_n + warp_col * _WARP_TILE_N + (_nf0 + lane_mod4) * 8 + sv * _STG_V
                                    col_j = col
                                    if col_j + vsize <= N:
                                        _vec = _row[sv * _STG_V : _STG_V]
                                        if cutlass.const_expr(acc_widen_to_fp32):
                                            _pf = _vec.to(cutlass.Float32)
                                            vec_f32 = _pf + cutlass.full_like(_pf, 0.0)
                                        else:
                                            vec_f32 = _vec

                                        # @@INJECT_STG_VEC_BINDINGS@@

                                        # @@INJECT_EPILOGUE@@
                    nvvm.bar_warp_sync(0xFFFFFFFF)
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
                    l2_promotion=_tma.TensorMapL2Promotion.l2_128b,
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
    ab_stride_elems = 128 // ab_dtype.width
    sym_m = cute.sym_int64()
    sym_n = cute.sym_int64(divisibility=out_vec_elems)
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

    def _make_fake_sfa():
        return make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

    def _make_fake_sfb():
        return make_fake_tensor(
            sf_cutlass_dtype,
            (cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            stride=(cute.sym_int64(), cute.sym_int64(), cute.sym_int64()),
            assumed_align=16,
        )

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
