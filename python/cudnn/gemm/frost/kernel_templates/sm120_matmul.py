# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""sm120 (GeForce/consumer Blackwell, CC 12.0) GEMM kernel: persistent + CLC
dynamic scheduler (2-stage ring) + warp-level MMA.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass.experimental.primitives as nvvm
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


CLC_SCHED_STAGES = 1

# Programmatic Dependent Launch (PDL, sm_90+; supported on sm_120).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for cross-warp sync of the 8 compute warps around TMA stores.
EPI_SYNC_BAR_ID = 1

# Compute-warp grid over the CTA tile (warp_row x warp_col).
NUM_COMPUTE_WARPS = 8
WARPS_M = 4
WARPS_N = 2

TMA_WARP_ID = 8
SCHEDULER_WARP_ID = 9
NUM_WARPS = 12

# CLC-ring consumers: every compute warp + the TMA producer + the scheduler
# itself each arrive once (elected) per consumed response slot.
NUM_CLC_CONSUMER_WARPS = NUM_COMPUTE_WARPS + 2

EPI_REG_COUNT = 232
PROD_REG_COUNT = 24

# ---------------------------------------------------------------------------
# Geometry derived from the injected tile constants (all plain Python ints —
# resolved at render/import time, traced as constants).
# ---------------------------------------------------------------------------

_ELEM_BITS = ab_dtype.width
_ELEM_BYTES = _ELEM_BITS // 8
_ELEMS_16B = 16 // _ELEM_BYTES
# One k-block = 32 bytes of K = the K extent of one mma.sync (k16 for 16-bit,
# k32 for 8-bit operands) — fort's UNIT_MATRIX_{A,B} column span.
_K_BLK_ELEMS = (32 * 8) // _ELEM_BITS
_NUM_K_BLOCKS = (cta_tile_mnk[2] * _ELEM_BITS) // (32 * 8)
_CTA_K_ELEMS = cta_tile_mnk[2]

_WARP_TILE_M = cta_tile_mnk[0] // WARPS_M
_WARP_TILE_N = cta_tile_mnk[1] // WARPS_N
_M_FRAGS = _WARP_TILE_M // 16
_N_FRAGS = _WARP_TILE_N // 8
_N_FRAG_PAIRS = _N_FRAGS // 2
_ACC_REGS = _M_FRAGS * _N_FRAGS * 4

_EPI_N = epi_tile_mn[1]

# SMEM K-row swizzle: the K-row width IS the swizzle span (the renderer derives
# ab_tma_swizzle from cta_tile_k_bytes; cross-checked against it below). The TMA
# s{128,64,32}b pattern == cutlass.Swizzle(b, 4, 3) with b = log2(row_bytes / 16).
# ldmatrix addresses below apply the same XOR
# (fort: swizzled_bank_id = bank ^ ((bank / 8) % SWIZZLE_SCALE)).
_AB_SMEM_SWIZZLE_BYTES = _CTA_K_ELEMS * _ELEM_BYTES
_AB_SW_BBITS = (_AB_SMEM_SWIZZLE_BYTES // 16).bit_length() - 1
_AB_SWIZZLE = cutlass.Swizzle(_AB_SW_BBITS, 4, 3)
# Epilogue staging tile swizzle — matches the s64b TMA-store descriptor.
_EPI_SWIZZLE = cutlass.Swizzle(2, 4, 3)

# ---- Transposed STG epilogue staging (fort "Sheet3" scheme) -----------------
_STG_EPI_LANE_QUAD = 4  # one STS.128 = 4 x 32-bit acc regs per lane
_STG_EPI_PAD = 4  # 16B skew after each 128-element batch (sheet's X cells)
_STG_EPI_BATCH_STRIDE = 32 * _STG_EPI_LANE_QUAD + _STG_EPI_PAD  # 132
_STG_EPI_GROUP_FRAGS = 4  # fragments (= STS batches) per 32-column group
_STG_EPI_WARP_ELEMS = _STG_EPI_GROUP_FRAGS * _STG_EPI_BATCH_STRIDE  # 528
_STG_EPI_NGRP = (_N_FRAGS + _STG_EPI_GROUP_FRAGS - 1) // _STG_EPI_GROUP_FRAGS
_STG_V = (vec_bytes_epi * 8) // cd_dtype.width

_STG_EPI_BYTES = 4 * _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS
_AB_STAGE_BYTES = (cta_tile_mnk[0] + cta_tile_mnk[1]) * _CTA_K_ELEMS * _ELEM_BYTES + 16
ab_stages = ab_stages - -(-_STG_EPI_BYTES // _AB_STAGE_BYTES)
assert ab_stages >= 1, "transposed STG epilogue: staging stream cannot be funded from the AB pipeline"

# ---- v1 scope guards (fail at render/import, not at runtime) ---------------
assert cluster_shape_mnk == (1, 1, 1), "sm120 has no thread-block clusters (CC 12.0): cluster_shape must be (1,1,1)"
assert threads_per_cta == NUM_WARPS * 32, f"sm120 template is a fixed 12-warp kernel (384 threads), got {threads_per_cta}"
assert not multicast_a and not multicast_b, "sm120 has no TMA multicast (no clusters)"
assert not a_is_m_major and not b_is_n_major, "sm120 template v1 supports K-major A and K-major B only (TN GEMM)"
# N-major, non-fp4 output only in v1 — enforced upstream by Sm120KernelTemplate._extra_reject.
assert num_gemms == 1 and num_a_operands == 1 and num_b_operands == 1, "sm120 template v1 is single-GEMM only"
assert cta_tile_mnk[0] % WARPS_M == 0 and _WARP_TILE_M % 16 == 0, f"cta_tile_m={cta_tile_mnk[0]} must be a multiple of {WARPS_M * 16}"
assert cta_tile_mnk[1] % WARPS_N == 0 and _WARP_TILE_N % 8 == 0, f"cta_tile_n={cta_tile_mnk[1]} must be a multiple of {WARPS_N * 8}"
_TMA_SWIZZLE_BY_BYTES = {32: _tma.TensorMapSwizzle.s32b, 64: _tma.TensorMapSwizzle.s64b, 128: _tma.TensorMapSwizzle.s128b}
assert ab_tma_swizzle == _TMA_SWIZZLE_BY_BYTES.get(_AB_SMEM_SWIZZLE_BYTES), "SMEM K-row width must equal the swizzle span"
assert _NUM_K_BLOCKS * _K_BLK_ELEMS == _CTA_K_ELEMS, "cta_tile_k must be a multiple of 32 bytes"
assert _STG_V >= 2 and _STG_V % 2 == 0 and 8 % _STG_V == 0, "transposed STG epilogue: the store vector must be whole pairs tiling the 8-column row run"

# ---------------------------------------------------------------------------
# The warp MMA instruction, resolved from the injected MMA dtypes.
# sm120 tensor cores are warp-scoped: mma.sync.aligned.m16n8k16 (16-bit A/B)
# or .m16n8k32 (8-bit A/B), row.col (both operands K-major), fp32/s32 acc —
# fort emits the same instruction pair per XMMA (bf16mma_fp32_16x16x16).
# ---------------------------------------------------------------------------

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
    """One warp-wide mma.sync on a (16, 8, 32-byte-K) fragment.

    A carrier: 4x b32 regs (ldmatrix.x4 of a [16 x 32B] K-major SMEM region).
    B carrier: 2x b32 regs. D/C: 4 accumulator regs (f32 or s32).
    """
    return cute.arch.inline_ptx(
        _MMA_PTX,
        write_only_types=[mma_c_dtype, mma_c_dtype, mma_c_dtype, mma_c_dtype],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


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


def _l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w):
    """N-direction super-block rasterization of the (m, n) tile coord, for
    L2 reuse. Applied identically to the launch-grid coords and to every CLC
    response, so a stolen CTA id lands on the same logical tile the canceled
    CTA would have computed (fort's ``swizzle()`` plays the same role).
    ``swizzle_w == 1`` falls out of the math as the identity mapping.
    """
    t = raw_n * nt_m + raw_m
    blk = nt_m * swizzle_w
    sb = t // blk
    off = t - sb * blk
    base_n = sb * swizzle_w
    cur_S = cutlass.min(cutlass.Int32(swizzle_w), nt_n - base_n)
    log_m = off // cur_S
    log_n = base_n + off - log_m * cur_S
    return log_m, log_n


@cute.kernel
def _kernel(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    # @@INJECT_KERNEL_AB_DESC_PARAMS@@
    # @@INJECT_KERNEL_TAP_PARAMS@@
    # @@INJECT_KERNEL_REDUCTION_STRIDE_PARAMS@@
    # @@INJECT_KERNEL_AUX_PARAMS@@
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

    # First tile from the launch grid (grid == tile grid); later tiles come
    # from canceled-CTA ids delivered through the CLC response ring.
    swizzle_w = _auto_swizzle_w(m, n, k, gridy)
    init_tile_m, init_tile_n = _l2_swizzle_tile(bidx, bidy, gridx, gridy, swizzle_w)
    init_tile_l = bidz

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

    # Per-compute-warp staging stream for the transposed STG epilogue (raw
    # accumulator dtype; 4 batches x (128 elems + 16B pad) = 528 elems, one
    # 32-column group of one m-frag at a time). Slices are warp-private, so
    # the round trip only needs bar.warp syncs — no CTA barrier.
    smem_stg_epi = cutlass.Array(
        mma_c_dtype,
        _STG_EPI_WARP_ELEMS * NUM_COMPUTE_WARPS,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )

    # ab full: one producer-elected arrive_expect_tx per stage.
    # ab empty: one elected arrive per compute warp per stage (fort inits this
    # to GROUPS_M * WARPS_PER_GROUP = the 8 math warps).
    # clc full: tx-count armed by the scheduler; completed by the response.
    # clc empty: one elected arrive per consumer warp per slot.
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

    VEC_BYTES = vec_bytes_epi
    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])

    # -- CLC scheduler warp ---------------------------------------------------
    # fort's scheduler warp: wait empty(slot) -> arm 16 tx bytes -> try_cancel
    # into the slot -> wait full(slot) -> read validity -> arrive empty. No
    # cluster: every CTA is its own leader and the response is CTA-local.
    if warp_idx == SCHEDULER_WARP_ID:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)
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
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)
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
                # One elected lane only: the barrier's arrival count is 1, and
                # the TMA copies deliver exactly num_tma_copy_bytes once.
                if elect_one:
                    nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                # K-major A: TMA box [K_tile, cta_m] at (k, m, l); OOB rows/cols
                # are hardware zero-filled (K tails contribute 0 to the MMA).
                for _ai in cutlass.range_constexpr(num_a_operands):
                    if elect_one:
                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                            smem_a_list[_ai].subview(sA_elems * stage),
                            tma_a_descs[_ai].get_ptr(),
                            (coord_k, coord_m, tile_l_a),
                            ab_full_mbar_ptr.subview(stage),
                        )
                # K-major B: TMA box [K_tile, cta_n] at (k, n, l).
                for _bj in cutlass.range_constexpr(num_b_operands):
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

        # # Drain: wait until the compute warps have consumed the final stage so
        # # the producer never exits with a stage it would have re-armed pending.
        # tail_stage = ab_iter % ab_stages
        # tail_phase = ab_empty_phase_bit
        # if tail_stage == 0 and ab_iter != 0:
        #     tail_phase = tail_phase ^ 1
        # for _ in range(ab_stages - 1):
        #     tail_stage = tail_stage + 1
        #     if tail_stage == ab_stages:
        #         tail_stage = cutlass.Int32(0)
        #         tail_phase = tail_phase ^ 1
        # if elect_one:
        #     while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
        #         pass

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
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")

        lane = tidx % 32
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        warp_row = warp_idx % WARPS_M
        warp_col = warp_idx // WARPS_M

        # ldmatrix lane->address maps (see PTX ldmatrix; addresses are 16B rows).
        # A x4 tile order = (rows 0-7, rows 8-15) x (16B col 0, 16B col 1) —
        # matching the a0..a3 fragment order of mma.sync (fort Lds_tile_8).
        a_ldm_row = (lane % 8) + 8 * ((lane // 8) % 2)
        a_ldm_col16 = lane // 16
        # B x4 covers TWO 8-col n-frags: (n rows 0-7, n rows 8-15) each split
        # over (16B col 0, 16B col 1) -> regs (b0,b1) frag0 + (b0,b1) frag1
        # (fort Lds_tile_10).
        b_ldm_pair_row = (lane % 8) + 8 * (lane // 16)
        b_ldm_pair_col16 = (lane // 8) % 2
        # B x2 tail: one n-frag (rows 0-7 x two 16B cols; lanes 16-31 unused).
        b_ldm_tail_row = lane % 8
        b_ldm_tail_col16 = (lane // 8) % 2

        acc = cutlass.Array(mma_c_dtype, _ACC_REGS, alignment=16)

        ab_full_phase_bit = cutlass.Int32(0)
        ab_iter = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_epi = cutlass.Int32(0)
        while is_valid != 0:
            coord_m = tile_m * cgrp_tile_mnk[0]
            coord_n = tile_n * cgrp_tile_mnk[1]

            for _z in cutlass.range_constexpr(_ACC_REGS):
                acc[_z] = mma_c_dtype(0)

            for k_tile_idx in range(num_k_tiles):
                stage = ab_iter % ab_stages
                if stage == 0 and ab_iter != 0:
                    ab_full_phase_bit = ab_full_phase_bit ^ 1

                while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                    pass

                sA_ptr = smem_a_list[0].subview(sA_elems * stage).data_ptr()
                sB_ptr = smem_b_list[0].subview(sB_elems * stage).data_ptr()

                for k_blk in cutlass.range_constexpr(_NUM_K_BLOCKS):
                    kb_base = k_blk * _K_BLK_ELEMS
                    a_frags = []
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
                    b_frags = []
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

                    for mf in cutlass.range_constexpr(_M_FRAGS):
                        av = a_frags[mf]
                        for nf in cutlass.range_constexpr(_N_FRAGS):
                            b0, b1 = b_frags[nf]
                            _o = (mf * _N_FRAGS + nf) * 4
                            acc[_o:4] = _mma_16x8_k32b(
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
                            )

                # Stage fully consumed by this warp (ldmatrix is synchronous).
                nvvm.bar_warp_sync(0xFFFFFFFF)
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
                    # Short tail group: trailing lanes own no fragment there
                    # (True at trace time for full groups — no guard emitted).
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
                                        # NB: Array slices are [start:COUNT], not
                                        # [start:stop] (matches acc[_o:2] above).
                                        _vec = _row[sv * _STG_V : _STG_V]
                                        if cutlass.const_expr(acc_widen_to_fp32):
                                            _pf = _vec.to(cutlass.Float32)
                                            vec_f32 = _pf + cutlass.full_like(_pf, 0.0)
                                        else:
                                            vec_f32 = _vec
                                        linear_idx = tile_l * out_stride_l_0 + row * out_stride_m_0 + col_j * out_stride_n_0

                                        # @@INJECT_STG_VEC_BINDINGS@@

                                        # @@INJECT_EPILOGUE@@
                    nvvm.bar_warp_sync(0xFFFFFFFF)

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

        # No more tiles for this CTA: all its global A/B reads have been issued
        # (fort fires launch_dependent_grids at the same point).
        if cutlass.const_expr(USE_PDL):
            if warp_idx == 0:
                if elect_one:
                    nvvm.griddepcontrol("launch_dependents")

    # -- Unused donor warps ---------------------------------------------------
    if warp_idx > SCHEDULER_WARP_ID:
        nvvm.setmaxregister(PROD_REG_COUNT, nvvm.SetMaxRegisterAction.DECREASE)


@cute.jit
def _host(
    problem_size: tuple,
    # @@INJECT_HOST_AB_PARAMS@@
    # @@INJECT_HOST_TAP_PARAMS@@
    # @@INJECT_HOST_AUX_PARAMS@@
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

    # K-major A/B only (asserted at render time): TMA boxes [K_tile, cta_{m,n}].
    tma_a_desc_list = []
    for _a_idx, _a_op in enumerate(_a_operands):
        a_stride_m, a_stride_k, a_stride_l = _a_stride_sets[_a_idx]
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

    # CLC persistent grid: launch the full tile grid (fort launches the same);
    # CTAs that finish early cancel not-yet-launched blocks and steal their
    # (m, n, l) coordinates through the response ring. No cluster launch on
    # sm120 (CC 12.0 has no thread-block clusters).
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    grid_shape = (num_tile_m_host, num_tile_n_host, batch)
    _kernel(
        problem_size[0],
        problem_size[1],
        problem_size[2],
        # @@INJECT_HOST_KERNEL_DESC_PASS@@
        # @@INJECT_HOST_TAP_PASS@@
        # @@INJECT_HOST_REDUCTION_STRIDE_PASS@@
        # @@INJECT_HOST_AUX_PASS@@
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
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
        stream=_fake_stream,
        options=frost_compile_options,
    )
