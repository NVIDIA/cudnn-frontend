# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""LayerNorm-prologue gated dual GEMM.

Protocol: python/cudnn/frost/README.md#layernorm-prologue-dual-gemm.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from cutlass._mlir.dialects import arith

import cutlass.experimental.primitives as nvvm
from cudnn.frost.tile_dsl.barrier import arrive_on_leader_release
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, fmul2, fp32_to_fp16, sigmoid, sigmoid2, sigmoid2_biased_half
from cudnn.frost.tile_dsl.tma import ld_global_v4, ld_shared_v4, st_global_v2, st_shared_v4
from cudnn.gemm.frost.ln_prologue import (
    A_STAGE_BYTES,
    B_STAGE_BYTES,
    PAIR_TILE_ROWS,
    SMEM_BUDGET_BYTES_BY_ARCH,
    LnDualGemmParams,
    ln_once_a_slots,
    ln_once_b_stages_effective,
    ln_once_n_tiles_per_item,
    ln_once_tiles_per_item,
    ring_stages,
    tcgen05_smem_desc_builder,
    validate_ln_dual_gemm_params,
)
from cudnn.gemm.frost.tile_helpers import (
    epi_subtile_spans as _epi_subtile_spans,
    l2_swizzle_tile as _l2_swizzle_tile,
    tcgen05_alloc as _tcgen05_alloc,
    tcgen05_dealloc as _tcgen05_dealloc,
    tcgen05_mma as _tcgen05_mma,
)
import cutlass.experimental.cuda.tensor_map as _tma
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream
from cuda.bindings import driver as _cuda
from cutlass.cute.arch import clc as cute_clc

# Tile config: CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma
mma_inst_shape_mnk = (256, 256, 16)
cta_group = 2
cgrp_tile_mnk = (256, 256, 64)
cta_tile_mnk = (128, 128, 64)
epi_tile_mn = (128, 32)
threads_per_cta = 384  # 8 template warps + the 4 LayerNorm-prologue warps
cluster_shape_mnk = (2, 1, 1)
matmul_a_batch = 1
matmul_b_batch = 0  # set below from PARAMS.passes: the pass axis is the GEMM batch of B and of the output
a_is_m_major = False
b_is_n_major = False
mma_a_major = 0
mma_b_major = 0
ab_stages = 7
b_collector_ok = False
multicast_a = False
multicast_b = False
a_mcast_slices = 1
b_mcast_slices = 1
ab_empty_full_mask = False
ab_smem_swizzle = cutlass.experimental.primitives.Tcgen05SmemSwizzle.SWIZZLE_128B
a_smem_desc_leading_byte_offset = 16
a_smem_desc_stride_byte_offset = 1024
a_smem_k_step_bytes = 32
a_smem_m_step_bytes = 16384
a_tma_group_elems = 1
b_smem_desc_leading_byte_offset = 16
b_smem_desc_stride_byte_offset = 1024
b_smem_k_step_bytes = 32
b_tma_group_elems = 1
mma_size_m = 1
mma_size_n = 1
mma_size_k = 4
ab_tma_swizzle = _tma.TensorMapSwizzle.s128b

# Dtype family: A=bf16->MMAbf16, B=bf16->MMAbf16, out=bf16 (K_BYTES=128)
ab_dtype = cutlass.BFloat16
cd_dtype = cutlass.BFloat16
epi_store_dtype = cutlass.BFloat16
mma_a_dtype = cutlass.BFloat16
mma_b_dtype = cutlass.BFloat16
mma_c_dtype = cutlass.Float32
acc_widen_to_fp32 = False
ab_tma_dtype = cutlass.BFloat16
mma_kind = nvvm.Tcgen05MMAKind.F16
epi_n = 32
epi_row_elems = 32
tile_swizzle_n = 0
swizzle_l2_budget_bytes = 44214954
num_gemms = 1
num_a_operands = 1
num_b_operands = 1
gemm_a_idx = (0,)
gemm_b_idx = (0,)
num_tmem_alloc_cols = 512
tmem_alloc_exclusive = False
acc_stages = 2  # 256 acc cols/stage
vec_bytes_epi = 32
split_k_slices = 1
frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_100a"
n_tma_outputs = 1
moe_aligned_offsets = False
epi_slot_widen = 1
epi_packed_lanes = False
epi_dp22 = False
epi_stage_rows = 128
epi_chunk_elems = 32
ab_stages = 6  # SMEM-D 16400B fixed + cast LOAD 0B/stage + multi-GEMM 0B/stage
fallback_cluster_shape_mnk = None
mixed_a_pattern_pref = 1
mixed_b_pattern_pref = 1
mixed_a_pattern_fb = 1
mixed_b_pattern_fb = 1

BODY_ARCH: str = "sm100"
PARAMS: LnDualGemmParams = globals().get("FROST_TEMPLATE_PARAMS", LnDualGemmParams())
validate_ln_dual_gemm_params(PARAMS, BODY_ARCH)
_K: int = PARAMS.k  # the LayerNorm width = GEMM K
_NK: int = PARAMS.k_stages  # K stages per tile (64 wide); the LN warps hold a whole row = all of them at once
_DESC_BUILD = tcgen05_smem_desc_builder(BODY_ARCH)
_C_OUT: int = PARAMS.c_out  # output channels per pass; N = 2 * c_out, interleaved [value block | gate block] per 256
_N_OUT: int = PARAMS.n_out  # the GEMM N per tile: 2 c_out (value | gate interleaved) in the columns arm, c_out per GEMM in the accumulators arm
_BIAS_PER_PASS: int = 2 * _C_OUT  # the interleaved [value block | gate block] bias entries per pass (BOTH arms: value + gate per output channel)
_PASSES: int = PARAMS.passes
HAS_BIAS: bool = PARAMS.has_bias
WANT_STATS: bool = PARAMS.want_stats
HAS_MASK: bool = PARAMS.has_mask
SKIP_MASKED_TILES: bool = PARAMS.skip_masked_tiles
_EPS: float = PARAMS.eps
_DIAG: str = PARAMS.diag  # diagnostic relief arms ("" in production)
_INV_K: float = 1.0 / PARAMS.k
_PASS_IN_M: bool = PARAMS.pass_order == "interleaved"
_LOG2_PASSES: int = _PASSES.bit_length() - 1  # the validator pins passes to a power of two when the pass is folded into M
matmul_b_batch = _PASSES
_VALUE_BLOCK: int = 128  # value columns [0, 128) and gate columns [128, 256) of every 256-wide N tile
_EPI_SUBTILES: int = _VALUE_BLOCK // epi_n  # 4 subtiles of 32 output channels per tile
# The facts the arms stand on, checked against THIS rendering's constants (the K3 arm re-resolves the tile below, from the same base).
if cta_group != 2 or cgrp_tile_mnk != (256, 256, 64) or cta_tile_mnk != (128, 128, 64) or cluster_shape_mnk != (2, 1, 1) or epi_dp22:
    raise ValueError(f"{__name__}: the LN-prologue dual GEMM is written for the cta_group::2 256x256x64 pair tile at cluster 2x1; re-render for another config")
if cd_dtype is not cutlass.BFloat16 or ab_dtype is not cutlass.BFloat16 or epi_packed_lanes or mma_size_m != 1 or num_gemms != 1 or split_k_slices != 1:
    raise ValueError(f"{__name__}: this fork was rendered for a plain dense bf16 single-GEMM, no split-K; the rendering's constants say otherwise")
if _K % cta_tile_mnk[2] != 0 or _NK > ab_stages:
    raise ValueError(
        f"{__name__}: K={_K} must be a multiple of the {cta_tile_mnk[2]}-wide stage and span at most ab_stages={ab_stages} stages (a row is normalised with every stage resident)"
    )
if epi_n != 32 or epi_stage_rows != 128 or epi_row_elems != 32:
    raise ValueError(f"{__name__}: the transposed staging buffer is laid out for 128-row x 32-column subtiles; the rendering says otherwise")
_IO_BPE: int = cd_dtype.width // 8
_DUAL: bool = PARAMS.pairing == "accumulators"
_A0_TRANSPOSE: bool = _DUAL and PARAMS.a0_prologue == "transpose"  # operand 0 normalised INTO the K-major layout (the fallback form)
_AFFINE_FOLD: bool = _DUAL and PARAMS.affine_fold  # operand 0's LayerNorm without gamma / beta (folded into W / bias on the host)
_FUSE_RESIDUAL: bool = _DUAL and PARAMS.fuse_residual
_PREFETCH_A_L2: bool = PARAMS.prefetch_a_l2
_LN_SMEM: bool = PARAMS.ln_pass_form != "registers"
_LN_ROLLED: bool = PARAMS.ln_pass_form == "smem_rolled"
_LN_PF: bool = PARAMS.ln_pass_form in ("smem_rolled_pf", "smem_rolled_tree")
_LN_TREE: bool = PARAMS.ln_pass_form == "smem_rolled_tree"
# Packed epilogue arithmetic.
_EPI_PACKED: bool = PARAMS.epi_math in ("packed", "packed_hoist", "packed_biasfold")
_EPI_BIAS_HOIST: bool = PARAMS.epi_math in ("packed_hoist", "packed_biasfold")
_EPI_BIASFOLD: bool = PARAMS.epi_math == "packed_biasfold"
_MIN_BLOCKS_PER_MP: int = PARAMS.min_blocks_per_mp
if _DUAL:
    mma_inst_shape_mnk = (256, 128, 16)  # two GEMMs of N = 128 per pair tile
    cgrp_tile_mnk = (256, 128, 64)
    cta_tile_mnk = (128, 64, 64)  # each CTA loads its 64-row half of BOTH B tiles
    num_gemms = 2
    num_a_operands = 2
    num_b_operands = 2
    gemm_a_idx = (0, 1)
    gemm_b_idx = (0, 1)
    ab_stages = (ab_stages * (A_STAGE_BYTES + B_STAGE_BYTES["columns"])) // (2 * A_STAGE_BYTES + 2 * B_STAGE_BYTES["accumulators"])
    if _K % cta_tile_mnk[2] != 0 or _NK > ab_stages:
        raise ValueError(f"{__name__}: the K3 arm keeps every K stage of a tile resident: K={_K} must be a multiple of 64 and at most {64 * ab_stages}")
_A_LANDS_MN: tuple = tuple((i == 0 and PARAMS.a0_major == "mn") for i in range(num_a_operands))
_A_MMA_MAJOR: tuple = tuple(1 if (_A_LANDS_MN[i] and not _A0_TRANSPOSE) else 0 for i in range(num_a_operands))
_A_LBO: tuple = tuple(8192 if mj else a_smem_desc_leading_byte_offset for mj in _A_MMA_MAJOR)
_A_SBO: tuple = tuple(a_smem_desc_stride_byte_offset for _ in _A_MMA_MAJOR)
_A_KSTEP: tuple = tuple(2048 if mj else a_smem_k_step_bytes for mj in _A_MMA_MAJOR)
_MN_GROUP: int = 64  # the 64-element (128-B) MN group of the SWIZZLE_128B atom; one TMA box per group
_LN_ONCE: bool = PARAMS.ln_once != ""
_N_TILES: int = PARAMS.n_tiles  # GEMM N tiles per row block (c_out / 128 in both arms)
_NTI: int = ln_once_n_tiles_per_item(PARAMS)  # N tiles per work item (every one under "block", one under "passes"; 1 when off)
_LOG2_NTI: int = _NTI.bit_length() - 1
_TPI: int = ln_once_tiles_per_item(PARAMS)  # GEMM tiles per work item (passes x _NTI; 1 when off)
_A_SLOTS: int = ln_once_a_slots(PARAMS, BODY_ARCH)  # resident A region slots: 2 where the arch's ring bytes hold two, else 1 (host module); 0 off
_A_REGION_STAGES: int = _A_SLOTS * _NK if _LN_ONCE else ab_stages  # the A buffers' stage count: the region's, or the shipped ring's
_B_STAGES: int = ln_once_b_stages_effective(PARAMS, BODY_ARCH) if _LN_ONCE else ab_stages
if ring_stages(BODY_ARCH, PARAMS.pairing) != ab_stages:
    raise ValueError(
        f"{__name__}: the host module's RING_STAGES_BY_ARCH[{BODY_ARCH!r}][{PARAMS.pairing!r}] = {ring_stages(BODY_ARCH, PARAMS.pairing)} but this body's ring is {ab_stages} deep -- the resident region and the K cap were sized against a ring the body does not have"
    )
if _LN_ONCE:
    if b_mcast_slices != 1 or multicast_b or b_is_n_major or matmul_a_batch != 1 or split_k_slices != 1 or fallback_cluster_shape_mnk is not None:
        raise ValueError(
            f"{__name__}: the LayerNorm-once arm is written for the rendering's live operand path (one B slice per CTA, no B multicast, K-major B, one A batch, no split-K, one cluster shape)"
        )

if any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: cluster M/N dimensions must be powers of two")
if fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2]):
    raise NotImplementedError(f"{__name__}: fallback cluster M/N dimensions must be powers of two")

_preferred_cluster_m_shift = cluster_shape_mnk[0].bit_length() - 1
_preferred_cluster_n_shift = cluster_shape_mnk[1].bit_length() - 1
_fallback_cluster_m_shift = _preferred_cluster_m_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[0].bit_length() - 1
_fallback_cluster_n_shift = _preferred_cluster_n_shift if fallback_cluster_shape_mnk is None else fallback_cluster_shape_mnk[1].bit_length() - 1
_CTA_GROUP = nvvm.CTAGroup.CTA_2 if cta_group == 2 else nvvm.CTAGroup.CTA_1
_cta_group_shift = cta_group.bit_length() - 1


# Scheduler ring depth: PARAMS.clc_stages (the rendering's value is 1; see LnDualGemmParams.clc_stages).
CLC_SCHED_STAGES = PARAMS.clc_stages

# Programmatic Dependent Launch (PDL, sm_90+).
USE_PDL = True

# Double-buffer for the TMA-store epilogue path.
EPI_SMEM_STAGES = 2

# Named barrier id for the 4-warp epilogue hand-over around the TMA store.
EPI_SYNC_BAR_ID = 1

# Named barrier id for the TMEM-alloc hand-over.
TMEM_ALLOC_BARRIER_ID = 2


def _smem_estimate_bytes() -> int:
    """Estimate the rendered kernel shared-memory allocation."""
    bpe = ab_dtype.width // 8
    rings = (num_a_operands * cta_tile_mnk[0] * _A_REGION_STAGES + num_b_operands * cta_tile_mnk[1] * _B_STAGES) * cta_tile_mnk[2] * bpe
    staging = epi_stage_rows * epi_row_elems * epi_slot_widen * EPI_SMEM_STAGES * (epi_store_dtype.width // 8)
    small = 4 * (2 * _K * num_a_operands + (_PASSES * _BIAS_PER_PASS if HAS_BIAS else 4) + _A_REGION_STAGES + 8)
    mbars = 8 * (2 * _B_STAGES + 2 * _A_REGION_STAGES + 2 * acc_stages + 2 * CLC_SCHED_STAGES + 1 + (_A_SLOTS if _LN_ONCE else 0)) + 16 * CLC_SCHED_STAGES + 16
    aligned = 1024 * (num_a_operands + num_b_operands + 1)
    return rings + staging + small + mbars + aligned


_SMEM_ESTIMATE_BYTES: int = _smem_estimate_bytes()
if _SMEM_ESTIMATE_BYTES > SMEM_BUDGET_BYTES_BY_ARCH[BODY_ARCH]:
    raise ValueError(
        f"{__name__}: this rendering declares ~{_SMEM_ESTIMATE_BYTES} B of dynamic SMEM but the {BODY_ARCH} part gives a CTA "
        f"{SMEM_BUDGET_BYTES_BY_ARCH[BODY_ARCH]} B -- a ring depth ({_B_STAGES}) or a resident A region ({_A_REGION_STAGES} stages) sized past the carveout; "
        "the launch would fail with an invalid-argument error (or a later buffer would overlap an earlier one), never a typed decline"
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


def _a0_chunk_words(a0w, wb, xow, chunk):
    """Load one MN-major operand chunk for the LayerNorm prologue."""
    words = []
    for e in range(4):
        ca = chunk * 8 + 2 * e
        words.append(a0w.data_ptr(wb + xow[2 * e] + ca * 32).load())
        words.append(a0w.data_ptr(wb + xow[2 * e + 1] + (ca + 1) * 32).load())
    return tuple(words)


def _a0_pair_f32(wa, wc, sel):
    """The packed (c, c + 1) bf16 pair of row j out of the two channel words (``prmt`` with the thread's half selector), unpacked to two fp32."""
    w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[wa, wc, sel])
    return f16x2_to_f32(w0, dtype=cutlass.BFloat16)


@cute.jit
def _row_coords(tile_m, n_kb, lkb, r_pair):
    """Map a pair row to its row-block coordinates."""
    ib = tile_m // n_kb
    kb_idx = tile_m - ib * n_kb
    kbw = cutlass.Int32(1) << lkb
    k_idx = kb_idx * kbw + (r_pair & (kbw - 1))
    bi_idx = ib * (cutlass.Int32(PAIR_TILE_ROWS) >> lkb) + (r_pair >> lkb)
    return k_idx, bi_idx


@cute.jit
def _cta_box_origin(tile_m, n_kb, lkb, pair_member):
    """The ``(k, bi)`` origin of the TMA box covering rows ``128 * pair_member .. + 127`` of the pair tile."""
    return _row_coords(tile_m, n_kb, lkb, pair_member * 128)


@cute.jit
def _decode_pass(tile_m, tile_l):
    """Decode the scheduler row-block and pass indices."""
    row_tile = (tile_m >> _LOG2_PASSES) if cutlass.const_expr(_PASS_IN_M) else tile_m
    pass_idx = (tile_m & (_PASSES - 1)) if cutlass.const_expr(_PASS_IN_M) else tile_l
    return row_tile, pass_idx


@cute.jit
def _item_tile(item_n, sub):
    """Decode a work-item subtile into its N tile and pass."""
    tile_n = item_n * _NTI + (sub & (_NTI - 1))
    pass_idx = sub >> _LOG2_NTI
    return tile_n, pass_idx


@cute.jit
def _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w):
    """Decode a CLC work item."""
    raw_m = m_idx >> _preferred_cluster_m_shift
    raw_n = n_idx >> _preferred_cluster_n_shift
    nt_m = gridx >> _preferred_cluster_m_shift
    nt_n = gridy >> _preferred_cluster_n_shift
    tile_m, item_n = _l2_swizzle_tile(raw_m, raw_n, nt_m, nt_n, swizzle_w, identity=tile_swizzle_n == 1)
    return tile_m, item_n


@cute.jit
def _row_mask(mask, row_valid, bi_idx, k_idx, n_res):
    """The row's mask as fp32: the bf16 ``mask[bi * n_res + k]`` for a valid row, 0 for a padded one (1 with no mask tensor)."""
    m_val = cutlass.Float32(0.0)
    if row_valid:
        if cutlass.const_expr(HAS_MASK):
            _byte_off = (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(2)
            _half = nvvm.inline_ptx("ld.global.u16 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[mask.iterator.toint() + _byte_off])
            m_val, _ = f16x2_to_f32(_half, dtype=cutlass.BFloat16)  # the low half of the zero-extended word IS the element
        else:
            m_val = cutlass.Float32(1.0)
    return m_val


@cute.kernel
def frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma(
    m: cutlass.Int64,
    n: cutlass.Int64,
    k: cutlass.Int64,
    tma_a_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_a_desc_1: cutlass.GridConstant[_tma.TensorMap],  # K3: operand 1 (z rows); K1 passes operand 0's descriptor again (unused)
    tma_b_desc_0: cutlass.GridConstant[_tma.TensorMap],
    tma_b_desc_1: cutlass.GridConstant[_tma.TensorMap],  # K3: the gate weights; K1 passes operand 0's descriptor again (unused)
    tma_c_desc_0: cutlass.GridConstant[_tma.TensorMap],
    n_res: cutlass.Int32,  # inner row index extent (k < n_res is a valid row)
    bn: cutlass.Int32,  # outer row index extent (bi < bn)
    n_pad: cutlass.Int32,  # output k extent (multiple of 8); k in [n_res, n_pad) is written as zeros
    lkb: cutlass.Int32,  # log2(k's per 256-row pair tile)
    n_kb: cutlass.Int32,  # k blocks per bi row
    ln_w: cute.Tensor,  # [K] bf16 LayerNorm weight of operand 0
    ln_b: cute.Tensor,  # [K] bf16 LayerNorm bias of operand 0
    ln_w1: Optional[cute.Tensor],  # [K] bf16 LayerNorm weight of operand 1 (K3); None in the single-operand arm
    ln_b1: Optional[cute.Tensor],  # [K] bf16 LayerNorm bias of operand 1 (K3); None in the single-operand arm
    bias: Optional[cute.Tensor],  # [passes * 2 * c_out] bf16, interleaved like the weight rows; None iff not HAS_BIAS
    mask: Optional[cute.Tensor],  # [bn * n_res] bf16 per-row mask; None iff not HAS_MASK (every valid row counts as 1)
    stats: Optional[cute.Tensor],  # [bn * n_res * 2] fp32 (mean, rstd) of operand 0's LayerNorm; None iff not WANT_STATS
    resid: Optional[cute.Tensor],  # [bn * n_res * c_out] bf16 pair-major residual added before the rounding (K3 fuse_residual); else None
) -> None:
    tma_a_descs = [tma_a_desc_0, tma_a_desc_1][:num_a_operands]
    tma_b_descs = [tma_b_desc_0, tma_b_desc_1][:num_b_operands]
    tma_c_descs = [tma_c_desc_0]
    ln_ws = [ln_w, ln_w1][:num_a_operands]
    ln_bs = [ln_b, ln_b1][:num_a_operands]

    mma_warp_id = 4
    tma_warp_id = 5
    scheduler_warp_id = 6
    unused_warp_id = 7
    num_epilogue_warps = 4
    epi_reg_count = 232
    prod_reg_count = 24
    ln_warp_base = 8
    num_ln_warps = 4
    ln_reg_count = 232
    LN_SYNC_BAR_ID = 3

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    elect_one = nvvm.elect_sync()

    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    gridx = cute.arch.grid_dim()[0]
    gridy = cute.arch.grid_dim()[1]

    a_mcast_pattern = mixed_a_pattern_pref
    if cutlass.const_expr(cta_group == 2):
        b_mcast_pattern = mixed_b_pattern_pref
    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        cluster_m = cluster_shape_mnk[0]
        cluster_n = cluster_shape_mnk[1]
    else:
        cdim_x, cdim_y, _cdim_z = cute.arch.block_in_cluster_dim()
        cluster_m = cdim_x
        cluster_n = cdim_y
        a_mcast_pattern = cutlass.Int32(mixed_a_pattern_pref)
        if cutlass.const_expr(cta_group == 2):
            b_mcast_pattern = cutlass.Int32(mixed_b_pattern_pref)
        if (cdim_x != cluster_shape_mnk[0]) | (cdim_y != cluster_shape_mnk[1]):
            a_mcast_pattern = cutlass.Int32(mixed_a_pattern_fb)
            if cutlass.const_expr(cta_group == 2):
                b_mcast_pattern = cutlass.Int32(mixed_b_pattern_fb)
    cluster_size = cluster_m * cluster_n * cluster_shape_mnk[2]

    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    m_rank = cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)
    n_rank = cta_rank_in_cluster >> _preferred_cluster_m_shift
    if cutlass.const_expr(fallback_cluster_shape_mnk is not None):
        if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
            m_rank = cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)
            n_rank = cta_rank_in_cluster >> _fallback_cluster_m_shift

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

    is_cluster_leader_cta = cta_rank_in_cluster == 0

    if warp_idx == mma_warp_id:
        for _i in cutlass.range_constexpr(num_a_operands):
            nvvm.prefetch_tensormap(tma_a_descs[_i].get_ptr())
        for _j in cutlass.range_constexpr(num_b_operands):
            nvvm.prefetch_tensormap(tma_b_descs[_j].get_ptr())

        for _ci in cutlass.range_constexpr(n_tma_outputs):
            nvvm.prefetch_tensormap(tma_c_descs[_ci].get_ptr())

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
    init_tile_m, init_tile_l = _decode_pass(init_tile_m, init_tile_l)

    if cutlass.const_expr(cta_group == 1):
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
    else:
        if cutlass.const_expr(multicast_a):
            tma_mcast_mask_a = cutlass.Int16(a_mcast_pattern << m_rank)
        else:
            tma_mcast_mask_a = cutlass.Int16(1 << cta_rank_in_cluster)
        if cutlass.const_expr(multicast_b):
            tma_mcast_mask_b = cutlass.Int16((b_mcast_pattern << pair_member) << (n_rank * cluster_m))
        else:
            tma_mcast_mask_b = cutlass.Int16(1 << cta_rank_in_cluster)

    _smem_sys_reserved = cutlass.Array(cutlass.Int8, 1024, space=cutlass.AddressSpace.smem, alignment=1)

    ab_full_mbar_ptr = cutlass.Array(cutlass.Int64, _B_STAGES, space=cutlass.AddressSpace.smem)
    ab_empty_mbar_ptr = cutlass.Array(cutlass.Int64, _B_STAGES, space=cutlass.AddressSpace.smem)
    acc_empty_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    acc_full_mbar_ptr = cutlass.Array(cutlass.Int64, acc_stages, space=cutlass.AddressSpace.smem)
    if cutlass.const_expr(cta_group == 2):
        tmem_dealloc_mbar_ptr = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)

    # CLC scheduler SMEM — 2-stage ring.
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
            sA_elems * _A_REGION_STAGES,  # LN ONCE: the resident A region (_A_SLOTS x _NK stages); the shipped ring when off
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_a_operands)
    ]
    smem_b_list = [
        cutlass.Array(
            ab_dtype,
            sB_elems * _B_STAGES,
            space=cutlass.AddressSpace.smem,
            alignment=1024,
        )
        for _ in range(num_b_operands)
    ]

    epi_subtile_elems = epi_stage_rows * epi_row_elems * epi_slot_widen
    smem_d_ptr = cutlass.Array(
        epi_store_dtype,
        epi_subtile_elems * EPI_SMEM_STAGES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    a_full_mbar_ptr = cutlass.Array(cutlass.Int64, _A_REGION_STAGES, space=cutlass.AddressSpace.smem)
    a_norm_full_mbar_ptr = cutlass.Array(cutlass.Int64, _A_REGION_STAGES, space=cutlass.AddressSpace.smem)
    smem_ln_params = cutlass.Array(cutlass.Float32, 2 * _K * num_a_operands, space=cutlass.AddressSpace.smem, alignment=16)  # per operand: gamma | beta, fp32
    smem_bias = cutlass.Array(cutlass.Float32, _PASSES * _BIAS_PER_PASS if HAS_BIAS else 4, space=cutlass.AddressSpace.smem, alignment=16)
    smem_flags = cutlass.Array(cutlass.Int32, _A_REGION_STAGES, space=cutlass.AddressSpace.smem, alignment=16)
    smem_vote = cutlass.Array(cutlass.Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    if cutlass.const_expr(_LN_ONCE):
        a_empty_mbar_ptr = cutlass.Array(cutlass.Int64, _A_SLOTS, space=cutlass.AddressSpace.smem)
    ln_norm_arrivers = num_ln_warps * cta_group  # 4 leader LN warps + 4 follower LN warps, one elected lane each

    acc_empty_count = num_epilogue_warps * cta_group
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
    num_consumer_warps_per_cta = 7 + num_ln_warps  # the LN warps consume the CLC ring too
    clc_empty_count = num_consumer_warps_per_cta * cluster_size
    if warp_idx == 0:
        if cutlass.const_expr(cta_group == 2):
            if elect_one:
                nvvm.mbarrier_init(tmem_dealloc_mbar_ptr, 32)
        if cutlass.const_expr(_LN_ONCE):
            # LN ONCE: the B ring's rows, then the A region's (a_full / a_norm_full per region stage), then the slots' release barrier
            for i in range(_B_STAGES):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
            for i in range(_A_REGION_STAGES):
                if elect_one:
                    nvvm.mbarrier_init(a_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(a_norm_full_mbar_ptr.subview(i), ln_norm_arrivers)
            for i in range(_A_SLOTS):
                if elect_one:
                    nvvm.mbarrier_init(a_empty_mbar_ptr.subview(i), ab_empty_count)
        else:
            for i in range(ab_stages):
                if elect_one:
                    nvvm.mbarrier_init(ab_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(ab_empty_mbar_ptr.subview(i), ab_empty_count)
                if elect_one:
                    nvvm.mbarrier_init(a_full_mbar_ptr.subview(i), 1)
                if elect_one:
                    nvvm.mbarrier_init(a_norm_full_mbar_ptr.subview(i), ln_norm_arrivers)
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
    if warp_idx >= ln_warp_base:
        if cutlass.const_expr(USE_PDL):
            nvvm.griddepcontrol("wait")
        ln_tid_stage = tidx - ln_warp_base * 32
        for _op in cutlass.range_constexpr(num_a_operands):  # operand op's gamma | beta at [2 K op, 2 K op + 2 K)
            for _rep in cutlass.range_constexpr((2 * _K // 8 + num_ln_warps * 32 - 1) // (num_ln_warps * 32)):
                _chunk = ln_tid_stage + _rep * (num_ln_warps * 32)
                if _chunk < 2 * _K // 8:
                    _is_beta = _chunk >= _K // 8
                    _src_base = cutlass.Int64(arith.select(_is_beta.ir_value(), ln_bs[_op].iterator.toint().ir_value(), ln_ws[_op].iterator.toint().ir_value()))
                    _src_chunk = cutlass.Int32(arith.select(_is_beta.ir_value(), (_chunk - _K // 8).ir_value(), _chunk.ir_value()))
                    _words = ld_global_v4(_src_base + _src_chunk.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Int32)
                    _f = []
                    for _w in _words:
                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                        _f.append(_lo)
                        _f.append(_hi)
                    st_shared_v4(smem_ln_params.data_ptr(2 * _K * _op + _chunk * 8), _f[0:4], cutlass.Float32)
                    st_shared_v4(smem_ln_params.data_ptr(2 * _K * _op + _chunk * 8 + 4), _f[4:8], cutlass.Float32)
        if cutlass.const_expr(HAS_BIAS):
            for _rep in cutlass.range_constexpr((_PASSES * _BIAS_PER_PASS // 8 + num_ln_warps * 32 - 1) // (num_ln_warps * 32)):
                _chunk = ln_tid_stage + _rep * (num_ln_warps * 32)
                if _chunk < _PASSES * _BIAS_PER_PASS // 8:
                    _words = ld_global_v4(bias.iterator.toint() + _chunk.to(cutlass.Int64) * cutlass.Int64(16), cutlass.Int32)
                    _f = []
                    for _w in _words:
                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                        _f.append(_lo)
                        _f.append(_hi)
                    if cutlass.const_expr(_EPI_BIASFOLD):
                        _gate_half = cutlass.Float32(
                            arith.select(((_chunk % 32) >= 16).ir_value(), cutlass.Float32(0.5).ir_value(), cutlass.Float32(1.0).ir_value())
                        )
                        _f = [_x * _gate_half for _x in _f]
                    st_shared_v4(smem_bias.data_ptr(_chunk * 8), _f[0:4], cutlass.Float32)
                    st_shared_v4(smem_bias.data_ptr(_chunk * 8 + 4), _f[4:8], cutlass.Float32)
    nvvm.fence_mbarrier_init()
    if cutlass.const_expr(cta_group == 1):

        if cutlass.const_expr(cluster_shape_mnk[0] * cluster_shape_mnk[1] > 1):
            nvvm.barrier_cluster_arrive_relaxed()
            nvvm.barrier_cluster_wait()
        else:
            nvvm.barrier_cta_sync(0)
    else:
        nvvm.barrier_cluster_arrive_relaxed()

    sA_bytes = sA_elems * (ab_dtype.width // 8)
    sB_bytes = sB_elems * (ab_dtype.width // 8)
    if cutlass.const_expr(cta_group == 1):
        num_tma_copy_bytes = num_b_operands * sB_bytes
    else:
        num_tma_copy_bytes = num_b_operands * sB_bytes * 2

    idescs = [
        cutlass.experimental.primitives.Tcgen05InstrDesc.build(
            a_dtype=mma_a_dtype,
            b_dtype=mma_b_dtype,
            c_dtype=mma_c_dtype,
            n_dim=mma_inst_shape_mnk[1],
            m_dim=mma_inst_shape_mnk[0],
            a_major=_A_MMA_MAJOR[gemm_a_idx[_g]],
            b_major=mma_b_major,
        )
        for _g in range(num_gemms)
    ]

    logical_cta_tile_m = cgrp_tile_mnk[0] // cluster_shape_mnk[0]
    logical_cta_tile_n = cgrp_tile_mnk[1] // cluster_shape_mnk[1]
    pair_n_size = logical_cta_tile_n
    epi_rows_per_mma_m = cta_tile_mnk[0] // mma_size_m
    if cutlass.const_expr(epi_dp22):
        # cluster-MMA m=128: the pair also splits N, so each CTA drains N/2.
        epi_cols_per_mma_m = pair_n_size // 2
    else:
        epi_cols_per_mma_m = pair_n_size
    cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m
    acc_region_cols = num_gemms * cols_per_acc_stage
    tmem_alloc_bar_count = (num_epilogue_warps + 1) * 32

    if cutlass.const_expr(cta_group == 2):
        nvvm.barrier_cluster_wait()
        nvvm.barrier_cta_sync(0)

    pass

    vsize = epi_chunk_elems

    M = m
    N = n
    num_k_tiles = cute.ceil_div(k, cta_tile_mnk[2])
    cgrp_tile_m_cur = logical_cta_tile_m * cluster_m
    cgrp_tile_n_cur = logical_cta_tile_n * cluster_n
    num_k_blocks = cta_tile_mnk[2] // mma_inst_shape_mnk[2]

    if warp_idx == scheduler_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)
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
        if cutlass.const_expr(_LN_ONCE):
            ab_empty_phase_bit = cutlass.Int32(1)
            ab_iter = cutlass.Int32(0)
            a_empty_phase_tma = cutlass.Int32(1)  # pre-armed (P5): parity 1 on a fresh barrier returns at once
            a_item_iter_tma = cutlass.Int32(0)
            a_slot_tma = cutlass.Int32(0)
            tile_m = init_tile_m
            item_n = init_tile_n
            sub = cutlass.Int32(0)
            item_iter = cutlass.Int32(0)
            is_valid = cutlass.Int32(1)
            clc_full_phase_tma = cutlass.Int32(0)
            while is_valid != 0:
                if sub == 0:
                    a_slot_tma = a_item_iter_tma % _A_SLOTS
                    if a_slot_tma == 0 and a_item_iter_tma != 0:
                        a_empty_phase_tma = a_empty_phase_tma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(a_empty_mbar_ptr.subview(a_slot_tma), a_empty_phase_tma, time_limit=10_000_000):
                        pass
                    a_coord_k, a_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
                    a_mn_k = []
                    a_mn_bi = []
                    for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                        _mk, _mbi = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + _mg * _MN_GROUP)
                        a_mn_k.append(_mk)
                        a_mn_bi.append(_mbi)
                    for _kt in cutlass.range_constexpr(_NK):
                        a_reg = a_slot_tma * _NK + _kt
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(a_reg), num_a_operands * sA_bytes)
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            sA_stage = smem_a_list[_ai].subview(sA_elems * a_reg)
                            if cutlass.const_expr(_A_LANDS_MN[_ai]):
                                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cta_global(
                                            sA_stage.subview(_mg * _MN_GROUP * cta_tile_mnk[2]),
                                            tma_a_descs[_ai].get_ptr(),
                                            (a_mn_k[_mg], a_mn_bi[_mg], _kt * cta_tile_mnk[2]),
                                            a_full_mbar_ptr.subview(a_reg),
                                        )
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        sA_stage,
                                        tma_a_descs[_ai].get_ptr(),
                                        (_kt * cta_tile_mnk[2], a_coord_k, a_coord_bi),
                                        a_full_mbar_ptr.subview(a_reg),
                                    )
                    a_item_iter_tma += 1
                tile_n, tile_l = _item_tile(item_n, sub)
                coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
                if cutlass.const_expr(matmul_b_batch == 1):
                    tile_l_b = cutlass.Int32(0)
                else:
                    tile_l_b = tile_l
                for _kt in cutlass.range_constexpr(_NK):
                    stage = ab_iter % _B_STAGES
                    if stage == 0 and ab_iter != 0:
                        ab_empty_phase_bit = ab_empty_phase_bit ^ 1
                    while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(stage), ab_empty_phase_bit, time_limit=10_000_000):
                        pass
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    for _bj in cutlass.range_constexpr(num_b_operands):
                        if elect_one:
                            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                smem_b_list[_bj].subview(sB_elems * stage),
                                tma_b_descs[_bj].get_ptr(),
                                (_kt * cta_tile_mnk[2], coord_n_per_cta, tile_l_b),
                                ab_full_mbar_ptr.subview(stage),
                                [],
                                multicast_mask=tma_mcast_mask_b,
                                group=_CTA_GROUP,
                            )
                    ab_iter += 1
                if sub == _TPI - 1:
                    # the item's last sub-tile: the per-ITEM CLC consume (the shipped per-tile block, once per item)
                    sub = cutlass.Int32(0)
                    consumer_stage = item_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and item_iter != 0:
                        clc_full_phase_tma = clc_full_phase_tma ^ 1
                    while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_tma, time_limit=10_000_000):
                        pass
                    m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    tile_m, item_n = _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w)
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    item_iter += 1
                else:
                    sub += 1
            tail_stage = ab_iter % _B_STAGES
            tail_phase = ab_empty_phase_bit
            if tail_stage == 0 and ab_iter != 0:
                tail_phase = tail_phase ^ 1
            for _ in range(_B_STAGES):
                while not nvvm.mbarrier_try_wait_parity(ab_empty_mbar_ptr.subview(tail_stage), tail_phase, time_limit=10_000_000):
                    pass
                tail_stage = tail_stage + 1
                if tail_stage == _B_STAGES:
                    tail_stage = cutlass.Int32(0)
                    tail_phase = tail_phase ^ 1
            a_tail_slot = a_item_iter_tma % _A_SLOTS
            a_tail_phase = a_empty_phase_tma
            if a_tail_slot == 0 and a_item_iter_tma != 0:
                a_tail_phase = a_tail_phase ^ 1
            for _ in range(_A_SLOTS):
                while not nvvm.mbarrier_try_wait_parity(a_empty_mbar_ptr.subview(a_tail_slot), a_tail_phase, time_limit=10_000_000):
                    pass
                a_tail_slot = a_tail_slot + 1
                if a_tail_slot == _A_SLOTS:
                    a_tail_slot = cutlass.Int32(0)
                    a_tail_phase = a_tail_phase ^ 1
        else:
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
                # Locate this CTA's LayerNorm rows.
                a_coord_k, a_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
                # K3 ARM: an MN-major operand lands as two 64-row boxes -- the (k, bi) of pair rows 128 pm + 0 and 128 pm + 64 (one bi each)
                a_mn_k = []
                a_mn_bi = []
                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                    _mk, _mbi = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + _mg * _MN_GROUP)
                    a_mn_k.append(_mk)
                    a_mn_bi.append(_mbi)
                if cutlass.const_expr(_PREFETCH_A_L2):
                    # the idle-TMA-warp L2 prefetch: every A box of this tile BEFORE the first ring-slot wait (the loads below then hit L2)
                    for _kt in cutlass.range_constexpr(_NK):
                        for _ai in cutlass.range_constexpr(num_a_operands):
                            if cutlass.const_expr(_A_LANDS_MN[_ai]):
                                for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_prefetch(tma_a_descs[_ai].get_ptr(), (a_mn_k[_mg], a_mn_bi[_mg], _kt * cta_tile_mnk[2]), [])
                            else:
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_prefetch(tma_a_descs[_ai].get_ptr(), (_kt * cta_tile_mnk[2], a_coord_k, a_coord_bi), [])
                if cutlass.const_expr(cta_group == 1):
                    coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * cta_tile_mnk[1]
                else:
                    coord_n_per_cta = tile_n * cgrp_tile_n_cur + n_rank * logical_cta_tile_n + pair_member * cta_tile_mnk[1]
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
                    if is_pair_leader:
                        if elect_one:
                            nvvm.mbarrier_arrive_expect_tx(ab_full_mbar_ptr.subview(stage), num_tma_copy_bytes)
                    if elect_one:
                        nvvm.mbarrier_arrive_expect_tx(a_full_mbar_ptr.subview(stage), num_a_operands * sA_bytes)
                    for _ai in cutlass.range_constexpr(num_a_operands):
                        sA_stage = smem_a_list[_ai].subview(sA_elems * stage)
                        if cutlass.const_expr(_A_LANDS_MN[_ai]):
                            for _mg in cutlass.range_constexpr(cta_tile_mnk[0] // _MN_GROUP):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cta_global(
                                        sA_stage.subview(_mg * _MN_GROUP * cta_tile_mnk[2]),
                                        tma_a_descs[_ai].get_ptr(),
                                        (a_mn_k[_mg], a_mn_bi[_mg], coord_k),
                                        a_full_mbar_ptr.subview(stage),
                                    )
                        else:
                            if elect_one:
                                nvvm.cp_async_bulk_tensor_shared_cta_global(
                                    sA_stage,
                                    tma_a_descs[_ai].get_ptr(),
                                    (coord_k, a_coord_k, a_coord_bi),
                                    a_full_mbar_ptr.subview(stage),
                                )

                    for _bj in cutlass.range_constexpr(num_b_operands):
                        sB_stage = smem_b_list[_bj].subview(sB_elems * stage)
                        tma_b_desc = tma_b_descs[_bj]
                        if cutlass.const_expr(b_mcast_slices > 1):
                            _b_rows = cta_tile_mnk[1] // b_mcast_slices
                            if cutlass.const_expr(fallback_cluster_shape_mnk is None):
                                if elect_one:
                                    nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                        sB_stage.subview(pair_m_idx * _b_rows * cta_tile_mnk[2]),
                                        tma_b_desc.get_ptr(),
                                        (coord_k, coord_n_per_cta + pair_m_idx * _b_rows, tile_l_b),
                                        ab_full_mbar_ptr.subview(stage),
                                        [],
                                        multicast_mask=tma_mcast_mask_b,
                                        group=_CTA_GROUP,
                                    )
                            else:
                                _b_per_cta = b_mcast_slices >> (_preferred_cluster_m_shift - _cta_group_shift)
                                if (cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1]):
                                    _b_per_cta = b_mcast_slices >> (_fallback_cluster_m_shift - _cta_group_shift)
                                for _bsl in cutlass.range(_b_per_cta):
                                    _b_idx = pair_m_idx * _b_per_cta + _bsl
                                    if elect_one:
                                        nvvm.cp_async_bulk_tensor_shared_cluster_global(
                                            sB_stage.subview(_b_idx * _b_rows * cta_tile_mnk[2]),
                                            tma_b_desc.get_ptr(),
                                            (coord_k, coord_n_per_cta + _b_idx * _b_rows, tile_l_b),
                                            ab_full_mbar_ptr.subview(stage),
                                            [],
                                            multicast_mask=tma_mcast_mask_b,
                                            group=_CTA_GROUP,
                                        )
                        elif cutlass.const_expr(multicast_b):
                            if pair_m_idx == 0:
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
                                                group=_CTA_GROUP,
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
                                            group=_CTA_GROUP,
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
                                            group=_CTA_GROUP,
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
                                        group=_CTA_GROUP,
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
                tile_m, tile_l = _decode_pass(tile_m, tile_l)
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

    if cutlass.const_expr(fallback_cluster_shape_mnk is None):
        b_arrive_pattern = (1 << cluster_m) - 1
    else:
        b_arrive_pattern = (cutlass.Int32(1) << cluster_m) - 1
    a_part = a_mcast_pattern << m_rank
    if cutlass.const_expr(cta_group == 2):
        a_part = a_part | (a_part << 1)
    b_part = b_arrive_pattern << (n_rank * cluster_m)
    if cutlass.const_expr(ab_empty_full_mask):
        ab_empty_arrive_mask = cutlass.Int16((1 << cluster_size) - 1)
    else:
        ab_empty_arrive_mask = cutlass.Int16(a_part | b_part)
    if cutlass.const_expr(cta_group == 2):
        acc_full_mcast = cutlass.Int16(3) << pair_leader_rank
    else:
        acc_full_mcast = None
    if warp_idx == mma_warp_id:
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
        if cutlass.const_expr(cta_group == 2):
            peer_cta_rank = cta_rank_in_cluster ^ 1

        if is_pair_leader:
            if cutlass.const_expr(_LN_ONCE):
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                acc_iter = cutlass.Int32(0)
                a_item_iter_mma = cutlass.Int32(0)
                a_slot_mma = cutlass.Int32(0)
                a_norm_phase_mma = cutlass.Int32(0)
                sub = cutlass.Int32(0)
                tile_skip = cutlass.Int32(0)
                item_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                desc_a_roots = [
                    _DESC_BUILD(
                        smem_a_list[i],
                        leading_byte_offset=_A_LBO[i],
                        stride_byte_offset=_A_SBO[i],
                        layout=ab_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    _DESC_BUILD(
                        smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        layout=ab_smem_swizzle,
                    )
                    for j in range(num_b_operands)
                ]
                while is_valid != 0:
                    acc_stage = acc_iter % acc_stages
                    if acc_stage == 0 and acc_iter != 0:
                        acc_empty_phase_bit = acc_empty_phase_bit ^ 1
                    while not nvvm.mbarrier_try_wait_parity(acc_empty_mbar_ptr.subview(acc_stage), acc_empty_phase_bit, time_limit=10_000_000):
                        pass
                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    tmem_addr_mmas = [
                        [
                            cutlass.inttoptr(
                                (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                6,
                                cutlass.Int32,
                            )
                            for mi in range(mma_size_m)
                        ]
                        for g in range(num_gemms)
                    ]
                    if sub == 0:
                        a_slot_mma = a_item_iter_mma % _A_SLOTS
                        if a_slot_mma == 0 and a_item_iter_mma != 0:
                            a_norm_phase_mma = a_norm_phase_mma ^ 1
                    scale_d = cutlass.Boolean(False)
                    for _kt in cutlass.range_constexpr(_NK):
                        stage = ab_iter % _B_STAGES
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1
                        while not nvvm.mbarrier_try_wait_parity(ab_full_mbar_ptr.subview(stage), ab_full_phase_bit, time_limit=10_000_000):
                            pass
                        a_reg = a_slot_mma * _NK + _kt
                        if sub == 0:
                            # the item's first sub-tile: both CTAs' A rows of this k-stage normalised in place (8 release arrives on the region stage)
                            while not nvvm.mbarrier_try_wait_parity(a_norm_full_mbar_ptr.subview(a_reg), a_norm_phase_mma, time_limit=10_000_000):
                                pass
                            if cutlass.const_expr(SKIP_MASKED_TILES and _kt == 0):
                                tile_skip = smem_flags.data_ptr(a_reg).load()
                        issue_mma = elect_one & (tile_skip == 0)
                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * a_reg + _A_KSTEP[gemm_a_idx[g]] * k_block_idx)
                                desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if issue_mma:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b,
                                            idescs[g],
                                            scale_d,
                                            collector_op=_a_collector_op(g),
                                            b_collector_op=_b_collector_op(mi),
                                        )
                            scale_d = cutlass.Boolean(True)
                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1
                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            multicast_mask=acc_full_mcast,
                            group=_CTA_GROUP,
                        )
                    acc_iter += 1
                    if sub == _TPI - 1:
                        # the item's last sub-tile: release the region slot to BOTH TMA warps (one pair-multicast commit = 1 arrive per CTA = a_empty's init)
                        if elect_one:
                            nvvm.tcgen05_commit(
                                a_empty_mbar_ptr.subview(a_slot_mma),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        a_item_iter_mma += 1
                        sub = cutlass.Int32(0)
                        consumer_stage = item_iter % CLC_SCHED_STAGES
                        if consumer_stage == 0 and item_iter != 0:
                            clc_full_phase_mma = clc_full_phase_mma ^ 1
                        while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_mma, time_limit=10_000_000):
                            pass
                        _m_idx, _n_idx, _l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                        cute.arch.fence_proxy("async.shared", space="cta")
                        is_valid = vld
                        nvvm.bar_warp_sync(0xFFFFFFFF)
                        if elect_one:
                            empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                            nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                        item_iter += 1
                    else:
                        sub += 1
            else:
                ab_full_phase_bit = cutlass.Int32(0)
                ab_iter = cutlass.Int32(0)
                acc_empty_phase_bit = cutlass.Int32(1)
                tile_iter = cutlass.Int32(0)
                is_valid = cutlass.Int32(1)
                clc_full_phase_mma = cutlass.Int32(0)
                acc_stage = cutlass.Int32(0)
                if cutlass.const_expr(split_k_slices > 1):
                    tile_l = bidz  # this tile's z coord
                desc_a_roots = [
                    _DESC_BUILD(
                        smem_a_list[i],
                        leading_byte_offset=_A_LBO[i],
                        stride_byte_offset=_A_SBO[i],
                        layout=ab_smem_swizzle,
                    )
                    for i in range(num_a_operands)
                ]
                desc_b_roots = [
                    _DESC_BUILD(
                        smem_b_list[j],
                        leading_byte_offset=b_smem_desc_leading_byte_offset,
                        stride_byte_offset=b_smem_desc_stride_byte_offset,
                        layout=ab_smem_swizzle,
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

                    acc_base_col = base_col_id_root + acc_stage * acc_region_cols
                    tmem_addr_mmas = [
                        [
                            cutlass.inttoptr(
                                (base_row_id << 16) | (acc_base_col + g * cols_per_acc_stage + mi * epi_cols_per_mma_m),
                                6,
                                cutlass.Int32,
                            )
                            for mi in range(mma_size_m)
                        ]
                        for g in range(num_gemms)
                    ]

                    if cutlass.const_expr(split_k_slices > 1):
                        split_idx = cutlass.Int64(tile_l % split_k_slices)
                        k_tiles_per_split = num_k_tiles // split_k_slices
                        k_tiles_remainder = num_k_tiles % split_k_slices
                        k_begin = split_idx * k_tiles_per_split + cutlass.min(split_idx, k_tiles_remainder)
                        k_end = (split_idx + 1) * k_tiles_per_split + cutlass.min(split_idx + 1, k_tiles_remainder)
                    else:
                        k_begin = cutlass.Int64(0)
                        k_end = num_k_tiles
                    scale_d = cutlass.Boolean(False)
                    tile_skip = cutlass.Int32(0)
                    for k_tile_idx in range(k_begin, k_end):
                        stage = ab_iter % ab_stages
                        if stage == 0 and ab_iter != 0:
                            ab_full_phase_bit = ab_full_phase_bit ^ 1

                        while not nvvm.mbarrier_try_wait_parity(
                            ab_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass
                        # LN-PROLOGUE DUAL GEMM: both CTAs' A rows normalised in place (8 release arrives, same slot / phase as ab_full)
                        while not nvvm.mbarrier_try_wait_parity(
                            a_norm_full_mbar_ptr.subview(stage),
                            ab_full_phase_bit,
                            time_limit=10_000_000,
                        ):
                            pass
                        if cutlass.const_expr(SKIP_MASKED_TILES):
                            if k_tile_idx == k_begin:
                                tile_skip = smem_flags.data_ptr(stage).load()
                        issue_mma = elect_one & (tile_skip == 0)

                        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                            for g in cutlass.range_constexpr(num_gemms):
                                desc_a_k = desc_a_roots[gemm_a_idx[g]].advance_start_address(sA_bytes * stage + _A_KSTEP[gemm_a_idx[g]] * k_block_idx)
                                desc_b = desc_b_roots[gemm_b_idx[g]].advance_start_address(sB_bytes * stage + b_smem_k_step_bytes * k_block_idx)
                                for mi in cutlass.range_constexpr(mma_size_m):
                                    desc_a = desc_a_k.advance_start_address(a_smem_m_step_bytes * mi)
                                    if issue_mma:
                                        _tcgen05_mma(
                                            mma_kind,
                                            _CTA_GROUP,
                                            tmem_addr_mmas[g][mi],
                                            desc_a,
                                            desc_b,
                                            idescs[g],
                                            scale_d,
                                            collector_op=_a_collector_op(g),
                                            b_collector_op=_b_collector_op(mi),
                                        )
                            scale_d = cutlass.Boolean(True)

                        if elect_one:
                            nvvm.tcgen05_commit(
                                ab_empty_mbar_ptr.subview(stage),
                                multicast_mask=ab_empty_arrive_mask,
                                group=_CTA_GROUP,
                            )
                        ab_iter += 1

                    if elect_one:
                        nvvm.tcgen05_commit(
                            acc_full_mbar_ptr.subview(acc_stage),
                            multicast_mask=acc_full_mcast,
                            group=_CTA_GROUP,
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
                    if cutlass.const_expr(split_k_slices > 1):
                        tile_l = _l_idx
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    tile_iter += 1

            if cutlass.const_expr(USE_PDL and split_k_slices == 1):
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
            nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
            if cutlass.const_expr(cta_group == 2):
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
                while not nvvm.mbarrier_try_wait_parity(tmem_dealloc_mbar_ptr, 0, time_limit=10_000_000):
                    pass
                nvvm.mbarrier_arrive(peer_mbar, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            alloc_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Int32)
            _tcgen05_dealloc(
                alloc_ptr,
                cutlass.Int32(num_tmem_alloc_cols),
                is_exclusive=tmem_alloc_exclusive,
                group=_CTA_GROUP,
            )
        else:
            if cutlass.const_expr(cta_group == 2):
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
                # Dense only, same reason as above.
                if cutlass.const_expr(USE_PDL and split_k_slices == 1):
                    nvvm.griddepcontrol("launch_dependents")
                nvvm.tcgen05_relinquish_alloc_permit(group=_CTA_GROUP)
                peer_mbar = nvvm.mapa(tmem_dealloc_mbar_ptr, peer_cta_rank)
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
        if cutlass.const_expr(_LN_ONCE):
            # Decode this work item's epilogue tile.
            item_n = init_tile_n
            sub = cutlass.Int32(0)
            item_iter = cutlass.Int32(0)
            tile_n, tile_l = _item_tile(item_n, sub)

        # @@EPILOGUE_SETUP:BEGIN@@
        if cutlass.const_expr(epi_packed_lanes):
            row_id_with_warp_offset = base_row_id
        else:
            row_id_with_warp_offset = base_row_id + warp_idx * 32

        epi_spans = _epi_subtile_spans(epi_cols_per_mma_m, epi_n)
        subtile_cnt = len(epi_spans)
        if cutlass.const_expr(epi_packed_lanes):
            shape = nvvm.Tcgen05LdStShape.SHAPE_16X32BX2
            ld_half_off = 0
        else:
            shape = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            ld_half_off = None
        lane = tidx % 32
        # @@EPILOGUE_SETUP:END@@

        epi_stage_idx = cutlass.Int32(EPI_SMEM_STAGES - 1)

        while is_valid != 0:
            coord_m_tile = tile_m * cgrp_tile_m_cur + m_rank * cta_tile_mnk[0]
            # @@EPILOGUE_DRAIN:BEGIN@@
            k_idx, bi_idx = _row_coords(tile_m, n_kb, lkb, pair_member * cta_tile_mnk[0] + tidx)
            row_valid = (k_idx < n_res) & (bi_idx < bn)
            m_val = _row_mask(mask, row_valid, bi_idx, k_idx, n_res)
            st_coord_k, st_coord_bi = _cta_box_origin(tile_m, n_kb, lkb, pair_member)
            out_ch_base = tile_n * _VALUE_BLOCK

            acc_stage = tile_iter % acc_stages
            if acc_stage == 0 and tile_iter != 0:
                acc_full_phase_bit = acc_full_phase_bit ^ 1

            while not nvvm.mbarrier_try_wait_parity(acc_full_mbar_ptr.subview(acc_stage), acc_full_phase_bit, time_limit=10_000_000):
                pass

            acc_base_col = base_col_id_root + acc_stage * acc_region_cols
            tmem_col_addr = (row_id_with_warp_offset << 16) | acc_base_col

            for subtile_idx in cutlass.range_constexpr(_EPI_SUBTILES):
                _col0 = subtile_idx * epi_n
                _outs = []
                _bp = []
                _bg = []
                if cutlass.const_expr(HAS_BIAS and _EPI_BIAS_HOIST and _DIAG != "no_epi_math"):
                    _bias_base = tile_l * _BIAS_PER_PASS + tile_n * (2 * _VALUE_BLOCK) + _col0
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bp.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _q * 4), cutlass.Float32))
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bg.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _VALUE_BLOCK + _q * 4), cutlass.Float32))
                if cutlass.const_expr(_DIAG == "no_epi_math"):
                    for _i in cutlass.range_constexpr(epi_n):
                        _outs.append(cutlass.Float32(0.0) * m_val)  # DIAGNOSTIC: no TMEM read, no sigmoid; the store path stays
                else:
                    _tm_v = cutlass.inttoptr(tmem_col_addr + _col0, 6, mma_c_dtype)
                    _tm_g = cutlass.inttoptr(tmem_col_addr + _VALUE_BLOCK + _col0, 6, mma_c_dtype)
                    _v = nvvm.tcgen05_ld(shape, _tm_v, num=epi_n, offset=ld_half_off)
                    _g = nvvm.tcgen05_ld(shape, _tm_g, num=epi_n, offset=ld_half_off)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(HAS_BIAS and not _EPI_BIAS_HOIST and _DIAG != "no_epi_math"):
                    _bias_base = (
                        tile_l * _BIAS_PER_PASS + tile_n * (2 * _VALUE_BLOCK) + _col0
                    )  # the interleaved [value block | gate block] per 128 output channels
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bp.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _q * 4), cutlass.Float32))
                    for _q in cutlass.range_constexpr(epi_n // 4):
                        _bg.extend(ld_shared_v4(smem_bias.data_ptr(_bias_base + _VALUE_BLOCK + _q * 4), cutlass.Float32))
                _rz = []
                if cutlass.const_expr(_FUSE_RESIDUAL and _DIAG != "no_epi_math"):
                    _rrow = cutlass.Int32(arith.select(row_valid.ir_value(), (bi_idx * n_res + k_idx).ir_value(), cutlass.Int32(0).ir_value()))
                    _raddr = resid.iterator.toint() + _rrow.to(cutlass.Int64) * cutlass.Int64(_C_OUT * _IO_BPE) + cutlass.Int64((out_ch_base + _col0) * _IO_BPE)
                    for _q in cutlass.range_constexpr(epi_n // 8):
                        for _w in ld_global_v4(_raddr + cutlass.Int64(_q * 16), cutlass.Int32):
                            _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                            _rz.append(_lo)
                            _rz.append(_hi)
                if cutlass.const_expr(_EPI_PACKED and _DIAG != "no_epi_math"):
                    for _h in cutlass.range_constexpr(epi_n // 2):
                        _i = 2 * _h
                        if cutlass.const_expr(HAS_BIAS and _EPI_BIASFOLD):
                            # "packed_biasfold": the value add as shipped; the gate bias (staged halved) folded into the tanh argument's fma.
                            _vl, _vh = fadd2(_v[_i], _v[_i + 1], _bp[_i], _bp[_i + 1])
                            _sl, _sh = sigmoid2_biased_half(_g[_i], _g[_i + 1], _bg[_i], _bg[_i + 1])
                        else:
                            _gl, _gh = fadd2(_g[_i], _g[_i + 1], _bg[_i], _bg[_i + 1]) if cutlass.const_expr(HAS_BIAS) else (_g[_i], _g[_i + 1])
                            _vl, _vh = fadd2(_v[_i], _v[_i + 1], _bp[_i], _bp[_i + 1]) if cutlass.const_expr(HAS_BIAS) else (_v[_i], _v[_i + 1])
                            _sl, _sh = sigmoid2(_gl, _gh)
                        _pl, _ph = fmul2(_sl, _sh, _vl, _vh)
                        _ol, _oh = fmul2(_pl, _ph, m_val, m_val)
                        if cutlass.const_expr(_FUSE_RESIDUAL):
                            _ol, _oh = fadd2(_ol, _oh, _rz[_i], _rz[_i + 1])
                        # A masked / padded row is EXACTLY +0.0 (a select, never x * 0: the residue may be anything).
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _ol.ir_value())))
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _oh.ir_value())))
                else:
                    for _i in cutlass.range_constexpr(epi_n if _DIAG != "no_epi_math" else 0):
                        if cutlass.const_expr(HAS_BIAS):
                            _gate = sigmoid(_g[_i] + _bg[_i])
                            _o = _gate * (_v[_i] + _bp[_i]) * m_val
                        else:
                            _gate = sigmoid(_g[_i])
                            _o = _gate * _v[_i] * m_val
                        if cutlass.const_expr(_FUSE_RESIDUAL):
                            _o = _o + _rz[_i]  # one rounding for update + residual (the fused block form; the API's unfused form rounds twice)
                        # A masked / padded row is EXACTLY +0.0 (a select, never x * 0: the residue may be anything).
                        _outs.append(cutlass.Float32(arith.select((m_val == cutlass.Float32(0.0)).ir_value(), cutlass.Float32(0.0).ir_value(), _o.ir_value())))

                if subtile_idx == _EPI_SUBTILES - 1:
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

                epi_stage_idx = (epi_stage_idx + 1) % EPI_SMEM_STAGES
                _tsv_0 = cutlass.Array(base=smem_d_ptr.data_ptr(epi_stage_idx * epi_subtile_elems), shape=4096, dtype=cutlass.BFloat16)
                if cutlass.const_expr(_DUAL):
                    _tsv_0.data_ptr(tidx * 32).store_swizzled(
                        cutlass.Vector.from_elements(tuple(_outs), cutlass.Float32).to(cutlass.BFloat16), alignment=64, swizzle=cutlass.Swizzle(2, 4, 3)
                    )
                else:
                    # Transposed staging: sD[channel][row] bf16 -- 32 lanes of a warp write 64 contiguous bytes per channel.
                    for _i in cutlass.range_constexpr(epi_n):
                        _tsv_0.data_ptr(_i * epi_stage_rows + tidx).store(_outs[_i].to(cutlass.BFloat16))
                cute.arch.fence_view_async_shared()
                nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
                if warp_idx == 0:
                    if elect_one:
                        if cutlass.const_expr(_DUAL):
                            # the 3-D (channel, k, bi) box of the pair-major [bn][n_res][c_out] output: rows k >= n_res / bi >= bn are clipped
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_descs[0].get_ptr(),
                                _tsv_0.data_ptr(),
                                (out_ch_base + _col0, st_coord_k, st_coord_bi),
                            )
                        else:
                            nvvm.cp_async_bulk_tensor_global_shared_cta(
                                tma_c_descs[0].get_ptr(),
                                _tsv_0.data_ptr(),
                                (st_coord_k, st_coord_bi, out_ch_base + _col0, tile_l),
                            )
                    if elect_one:
                        nvvm.cp_async_bulk_commit_group()
                    nvvm.cp_async_bulk_wait_group(EPI_SMEM_STAGES - 1, read=True)
                nvvm.barrier_cta_sync(barrier_id=EPI_SYNC_BAR_ID, thread_count=num_epilogue_warps * 32)
            # @@EPILOGUE_DRAIN:END@@
            if cutlass.const_expr(_LN_ONCE):
                # LN ONCE: the next sub-tile of the item, or -- after its last -- the per-ITEM CLC consume (the shipped block in the else arm, once per item)
                if sub == _TPI - 1:
                    sub = cutlass.Int32(0)
                    consumer_stage = item_iter % CLC_SCHED_STAGES
                    if consumer_stage == 0 and item_iter != 0:
                        clc_full_phase_epi = clc_full_phase_epi ^ 1
                    while not nvvm.mbarrier_try_wait_parity(clc_full_mbar_ptr.subview(consumer_stage), clc_full_phase_epi, time_limit=10_000_000):
                        pass
                    m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    is_valid = vld
                    tile_m, item_n = _item_decode(m_idx, n_idx, gridx, gridy, swizzle_w)
                    nvvm.bar_warp_sync(0xFFFFFFFF)
                    if elect_one:
                        empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                        nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
                    item_iter += 1
                else:
                    sub += 1
                tile_n, tile_l = _item_tile(item_n, sub)
                tile_iter += 1
            else:
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
                tile_m, tile_l = _decode_pass(tile_m, tile_l)
                nvvm.bar_warp_sync(0xFFFFFFFF)
                if elect_one:
                    empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                    nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)

                tile_iter += 1

        if warp_idx == 0:
            nvvm.cp_async_bulk_wait_group(0, read=True)

    if warp_idx == unused_warp_id:
        nvvm.setmaxregister(prod_reg_count, nvvm.SetMaxRegisterAction.DECREASE)

    if warp_idx >= ln_warp_base:
        nvvm.setmaxregister(ln_reg_count, nvvm.SetMaxRegisterAction.INCREASE)
        ln_tid = tidx - ln_warp_base * 32  # this CTA's row 0..127 == the SMEM A-stage row normalised by the calling lane
        ln_warp = warp_idx - ln_warp_base
        ln_lane = ln_tid & 31
        ln_swz = ln_tid & 7  # the SWIZZLE_128B chunk XOR of this row
        _KOP = 1 if _DUAL else 0  # the K-MAJOR operand this thread normalises as a row: K3's operand 1 (z); K1's only operand
        if cutlass.const_expr(_DUAL):
            _xw = (ln_tid >> 6) * 2048 + ((ln_tid & 63) >> 1)
            _xe = (ln_tid >> 6) * 4096 + (ln_tid & 63)
            _xow = []
            _xoe = []
            for _kx in cutlass.range_constexpr(8):
                _xow.append(_xw ^ (_kx << 2))
                _xoe.append(_xe ^ (_kx << 3))
            # prmt selector: an even j takes the LOW halves of the two channel words, an odd j the HIGH halves -> the packed (c, c + 1) pair of row j
            _sel = cutlass.Int32(arith.select(((ln_tid & 1) != 0).ir_value(), cutlass.Int32(0x7632).ir_value(), cutlass.Int32(0x5410).ir_value()))
            _a0w = cutlass.Array(base=smem_a_list[0].data_ptr(), shape=sA_elems * ab_stages // 2, dtype=cutlass.Int32)
        ab_iter = cutlass.Int32(0)
        a_full_phase_ln = cutlass.Int32(0)
        tile_m = init_tile_m
        tile_n = init_tile_n
        tile_l = init_tile_l
        tile_iter = cutlass.Int32(0)
        is_valid = cutlass.Int32(1)
        clc_full_phase_ln = cutlass.Int32(0)
        while is_valid != 0:
            r_pair = pair_member * cta_tile_mnk[0] + ln_tid
            k_idx, bi_idx = _row_coords(tile_m, n_kb, lkb, r_pair)
            row_valid = (k_idx < n_res) & (bi_idx < bn)
            # (1) the ring slots / phases of this tile's _NK stages (the TMA warp's stepping), then wait for all of them
            _sts = []
            _phs = []
            if cutlass.const_expr(_LN_ONCE):
                a_slot_ln = tile_iter % _A_SLOTS
                if a_slot_ln == 0 and tile_iter != 0:
                    a_full_phase_ln = a_full_phase_ln ^ 1
                for _j in cutlass.range_constexpr(_NK):
                    _sts.append(a_slot_ln * _NK + _j)
                    _phs.append(a_full_phase_ln)
            else:
                for _j in cutlass.range_constexpr(_NK):
                    _st = ab_iter % ab_stages
                    if _st == 0 and ab_iter != 0:
                        a_full_phase_ln = a_full_phase_ln ^ 1
                    _sts.append(_st)
                    _phs.append(a_full_phase_ln)
                    ab_iter += 1
            for _j in cutlass.range_constexpr(_NK):
                while not nvvm.mbarrier_try_wait_parity(a_full_mbar_ptr.subview(_sts[_j]), _phs[_j], time_limit=10_000_000):
                    pass
            if cutlass.const_expr(_DIAG == "ln_first_tile_only"):
                _ln_this_tile = (tile_n == 0) & (tile_l == 0)
            else:
                _ln_this_tile = cutlass.Boolean(True)
            if _ln_this_tile:
                # (2) pass A on the K-MAJOR operand: the row into registers (K/2 packed bf16x2 words, chunks in LOGICAL channel order), fp32 sum
                _ln_work = cutlass.const_expr(_DIAG != "no_ln")
                _row_words = []
                _sum = cutlass.Float32(0.0)
                _sumsq = cutlass.Float32(0.0)
                if cutlass.const_expr(_LN_PF):
                    _ta0 = cutlass.Float32(0.0)
                    _ta1 = cutlass.Float32(0.0)
                    _ta2 = cutlass.Float32(0.0)
                    _ta3 = cutlass.Float32(0.0)
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        _pa0, _pa1, _pa2, _pa3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0 (0 ^ swz = swz)
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _nqa = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                            _na0, _na1, _na2, _na3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqa ^ ln_swz) * 8), cutlass.Int32)
                            if cutlass.const_expr(_LN_TREE):
                                _tl0, _th0 = f16x2_to_f32(_pa0, dtype=cutlass.BFloat16)
                                _tl1, _th1 = f16x2_to_f32(_pa1, dtype=cutlass.BFloat16)
                                _tl2, _th2 = f16x2_to_f32(_pa2, dtype=cutlass.BFloat16)
                                _tl3, _th3 = f16x2_to_f32(_pa3, dtype=cutlass.BFloat16)
                                _ta0 = _ta0 + _tl0 + _th0
                                _ta1 = _ta1 + _tl1 + _th1
                                _ta2 = _ta2 + _tl2 + _th2
                                _ta3 = _ta3 + _tl3 + _th3
                                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                    for _tlx, _thx in ((_tl0, _th0), (_tl1, _th1), (_tl2, _th2), (_tl3, _th3)):
                                        _sumsq = _sumsq + _tlx * _tlx + _thx * _thx
                            else:
                                for _pax in (_pa0, _pa1, _pa2, _pa3):
                                    _plo, _phi = f16x2_to_f32(_pax, dtype=cutlass.BFloat16)
                                    _sum = _sum + _plo + _phi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _plo * _plo + _phi * _phi
                            _pa0, _pa1, _pa2, _pa3 = _na0, _na1, _na2, _na3
                    if cutlass.const_expr(_LN_TREE):
                        _sum = (_ta0 + _ta1) + (_ta2 + _ta3)
                elif cutlass.const_expr(_LN_SMEM):
                    # Re-read shared chunks for LayerNorm.
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        if cutlass.const_expr(_LN_ROLLED):
                            # a runtime loop carries SCALARS only (the DSL joins every assignment of a loop-body name to one type: no list names here)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _aw0, _aw1, _aw2, _aw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                for _awx in (_aw0, _aw1, _aw2, _aw3):
                                    _alo, _ahi = f16x2_to_f32(_awx, dtype=cutlass.BFloat16)
                                    _sum = _sum + _alo + _ahi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _alo * _alo + _ahi * _ahi
                        else:
                            for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                for _w in _words:
                                    _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                    _sum = _sum + _lo + _hi
                                    if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                        _sumsq = _sumsq + _lo * _lo + _hi * _hi
                else:
                    for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                        _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                            _phys = _q ^ ln_swz
                            _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), cutlass.Int32)
                            for _w in _words:
                                _row_words.append(_w)
                                _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                _sum = _sum + _lo + _hi
                                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                                    _sumsq = _sumsq + _lo * _lo + _hi * _hi
                _mean = _sum * cutlass.Float32(_INV_K)
                # (3) pass B: sum of squared deviations (two-pass: the mean is subtracted before squaring)
                _sq = cutlass.Float32(0.0)
                if cutlass.const_expr(_DIAG == "one_pass_stats"):
                    _sq = _sumsq - _sum * _mean  # DIAGNOSTIC: E[x^2] - mean^2 from the first pass
                else:
                    if cutlass.const_expr(_LN_PF):
                        _tb0 = cutlass.Float32(0.0)
                        _tb1 = cutlass.Float32(0.0)
                        _tb2 = cutlass.Float32(0.0)
                        _tb3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                            _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                            _pb0, _pb1, _pb2, _pb3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqb = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nb0, _nb1, _nb2, _nb3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqb ^ ln_swz) * 8), cutlass.Int32)
                                if cutlass.const_expr(_LN_TREE):
                                    _ul0, _uh0 = f16x2_to_f32(_pb0, dtype=cutlass.BFloat16)
                                    _ul1, _uh1 = f16x2_to_f32(_pb1, dtype=cutlass.BFloat16)
                                    _ul2, _uh2 = f16x2_to_f32(_pb2, dtype=cutlass.BFloat16)
                                    _ul3, _uh3 = f16x2_to_f32(_pb3, dtype=cutlass.BFloat16)
                                    _udl0 = _ul0 - _mean
                                    _udh0 = _uh0 - _mean
                                    _udl1 = _ul1 - _mean
                                    _udh1 = _uh1 - _mean
                                    _udl2 = _ul2 - _mean
                                    _udh2 = _uh2 - _mean
                                    _udl3 = _ul3 - _mean
                                    _udh3 = _uh3 - _mean
                                    _tb0 = _tb0 + _udl0 * _udl0 + _udh0 * _udh0
                                    _tb1 = _tb1 + _udl1 * _udl1 + _udh1 * _udh1
                                    _tb2 = _tb2 + _udl2 * _udl2 + _udh2 * _udh2
                                    _tb3 = _tb3 + _udl3 * _udl3 + _udh3 * _udh3
                                else:
                                    for _pbx in (_pb0, _pb1, _pb2, _pb3):
                                        _qlo, _qhi = f16x2_to_f32(_pbx, dtype=cutlass.BFloat16)
                                        _qdl = _qlo - _mean
                                        _qdh = _qhi - _mean
                                        _sq = _sq + _qdl * _qdl + _qdh * _qdh
                                _pb0, _pb1, _pb2, _pb3 = _nb0, _nb1, _nb2, _nb3
                        if cutlass.const_expr(_LN_TREE):
                            _sq = (_tb0 + _tb1) + (_tb2 + _tb3)
                    elif cutlass.const_expr(_LN_SMEM):
                        # the re-read of pass A's chunks (the stage is unchanged until pass C): the same words, the same order -> bitwise the register form
                        for _j in cutlass.range_constexpr(_NK if _ln_work else 0):
                            _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    _bw0, _bw1, _bw2, _bw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                    for _bwx in (_bw0, _bw1, _bw2, _bw3):
                                        _blo, _bhi = f16x2_to_f32(_bwx, dtype=cutlass.BFloat16)
                                        _bdl = _blo - _mean
                                        _bdh = _bhi - _mean
                                        _sq = _sq + _bdl * _bdl + _bdh * _bdh
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_q ^ ln_swz) * 8), cutlass.Int32)
                                    for _w in _words:
                                        _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                                        _dl = _lo - _mean
                                        _dh = _hi - _mean
                                        _sq = _sq + _dl * _dl + _dh * _dh
                    else:
                        for _w in _row_words:
                            _lo, _hi = f16x2_to_f32(_w, dtype=cutlass.BFloat16)
                            _dl = _lo - _mean
                            _dh = _hi - _mean
                            _sq = _sq + _dl * _dl + _dh * _dh
                _rstd = cute.math.rsqrt(_sq * cutlass.Float32(_INV_K) + cutlass.Float32(_EPS), fastmath=True)
                if cutlass.const_expr(WANT_STATS and not _DUAL):
                    if row_valid & (tile_l == 0) & (tile_n == 0):
                        _stats_addr = stats.iterator.toint() + (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(8)
                        st_global_v2(_stats_addr, [_mean, _rstd], cutlass.Float32)
                _wi = 0
                for _j in cutlass.range_constexpr(_NK if (_DIAG != "no_ln_norm" and _ln_work) else 0):
                    _row_base = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                    if cutlass.const_expr(_LN_SMEM and _LN_ROLLED):
                        # the re-read form, rolled: the chunk index is a loop variable; its XOR and its gamma / beta offset are computed per chunk
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _cphys = _q ^ ln_swz
                            _cc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _q * 8
                            _cg0, _cg1, _cg2, _cg3 = ld_shared_v4(smem_ln_params.data_ptr(_cc0), cutlass.Float32)
                            _cg4, _cg5, _cg6, _cg7 = ld_shared_v4(smem_ln_params.data_ptr(_cc0 + 4), cutlass.Float32)
                            _cb0, _cb1, _cb2, _cb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _cc0), cutlass.Float32)
                            _cb4, _cb5, _cb6, _cb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _cc0 + 4), cutlass.Float32)
                            _cw0, _cw1, _cw2, _cw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _cphys * 8), cutlass.Int32)
                            _cl0, _ch0 = f16x2_to_f32(_cw0, dtype=cutlass.BFloat16)
                            _cl1, _ch1 = f16x2_to_f32(_cw1, dtype=cutlass.BFloat16)
                            _cl2, _ch2 = f16x2_to_f32(_cw2, dtype=cutlass.BFloat16)
                            _cl3, _ch3 = f16x2_to_f32(_cw3, dtype=cutlass.BFloat16)
                            _cpk0 = fp32_to_fp16((_cl0 - _mean) * _rstd * _cg0 + _cb0, (_ch0 - _mean) * _rstd * _cg1 + _cb1, dtype=cutlass.BFloat16)
                            _cpk1 = fp32_to_fp16((_cl1 - _mean) * _rstd * _cg2 + _cb2, (_ch1 - _mean) * _rstd * _cg3 + _cb3, dtype=cutlass.BFloat16)
                            _cpk2 = fp32_to_fp16((_cl2 - _mean) * _rstd * _cg4 + _cb4, (_ch2 - _mean) * _rstd * _cg5 + _cb5, dtype=cutlass.BFloat16)
                            _cpk3 = fp32_to_fp16((_cl3 - _mean) * _rstd * _cg6 + _cb6, (_ch3 - _mean) * _rstd * _cg7 + _cb7, dtype=cutlass.BFloat16)
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _cphys * 8), [_cpk0, _cpk1, _cpk2, _cpk3], cutlass.Int32)
                    elif cutlass.const_expr(_LN_PF):
                        _pcc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2]
                        _pg0, _pg1, _pg2, _pg3 = ld_shared_v4(smem_ln_params.data_ptr(_pcc0), cutlass.Float32)
                        _pg4, _pg5, _pg6, _pg7 = ld_shared_v4(smem_ln_params.data_ptr(_pcc0 + 4), cutlass.Float32)
                        _pk0, _pk1, _pk2, _pk3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _pcc0), cutlass.Float32)
                        _pk4, _pk5, _pk6, _pk7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _pcc0 + 4), cutlass.Float32)
                        _pw0, _pw1, _pw2, _pw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + ln_swz * 8), cutlass.Int32)  # chunk 0
                        for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                            _nqc = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                            _ncc0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _nqc * 8
                            _ng0, _ng1, _ng2, _ng3 = ld_shared_v4(smem_ln_params.data_ptr(_ncc0), cutlass.Float32)
                            _ng4, _ng5, _ng6, _ng7 = ld_shared_v4(smem_ln_params.data_ptr(_ncc0 + 4), cutlass.Float32)
                            _nk0, _nk1, _nk2, _nk3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _ncc0), cutlass.Float32)
                            _nk4, _nk5, _nk6, _nk7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _ncc0 + 4), cutlass.Float32)
                            _nw0, _nw1, _nw2, _nw3 = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + (_nqc ^ ln_swz) * 8), cutlass.Int32)
                            _pphys = _q ^ ln_swz
                            _pl0, _ph0 = f16x2_to_f32(_pw0, dtype=cutlass.BFloat16)
                            _pl1, _ph1 = f16x2_to_f32(_pw1, dtype=cutlass.BFloat16)
                            _pl2, _ph2 = f16x2_to_f32(_pw2, dtype=cutlass.BFloat16)
                            _pl3, _ph3 = f16x2_to_f32(_pw3, dtype=cutlass.BFloat16)
                            _ppk0 = fp32_to_fp16((_pl0 - _mean) * _rstd * _pg0 + _pk0, (_ph0 - _mean) * _rstd * _pg1 + _pk1, dtype=cutlass.BFloat16)
                            _ppk1 = fp32_to_fp16((_pl1 - _mean) * _rstd * _pg2 + _pk2, (_ph1 - _mean) * _rstd * _pg3 + _pk3, dtype=cutlass.BFloat16)
                            _ppk2 = fp32_to_fp16((_pl2 - _mean) * _rstd * _pg4 + _pk4, (_ph2 - _mean) * _rstd * _pg5 + _pk5, dtype=cutlass.BFloat16)
                            _ppk3 = fp32_to_fp16((_pl3 - _mean) * _rstd * _pg6 + _pk6, (_ph3 - _mean) * _rstd * _pg7 + _pk7, dtype=cutlass.BFloat16)
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _pphys * 8), [_ppk0, _ppk1, _ppk2, _ppk3], cutlass.Int32)
                            _pg0, _pg1, _pg2, _pg3, _pg4, _pg5, _pg6, _pg7 = _ng0, _ng1, _ng2, _ng3, _ng4, _ng5, _ng6, _ng7
                            _pk0, _pk1, _pk2, _pk3, _pk4, _pk5, _pk6, _pk7 = _nk0, _nk1, _nk2, _nk3, _nk4, _nk5, _nk6, _nk7
                            _pw0, _pw1, _pw2, _pw3 = _nw0, _nw1, _nw2, _nw3
                    else:
                        for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                            _phys = _q ^ ln_swz
                            _c0 = 2 * _K * _KOP + _j * cta_tile_mnk[2] + _q * 8
                            _gam = []
                            _bet = []
                            _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_c0), cutlass.Float32))
                            _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_c0 + 4), cutlass.Float32))
                            _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _c0), cutlass.Float32))
                            _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _c0 + 4), cutlass.Float32))
                            if cutlass.const_expr(_LN_SMEM):
                                _words = ld_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), cutlass.Int32)  # the re-read form, unrolled
                            else:
                                _words = _row_words[_wi : _wi + 4]
                            _packed = []
                            for _e in cutlass.range_constexpr(4):
                                _lo, _hi = f16x2_to_f32(_words[_e], dtype=cutlass.BFloat16)
                                _ylo = (_lo - _mean) * _rstd * _gam[2 * _e] + _bet[2 * _e]
                                _yhi = (_hi - _mean) * _rstd * _gam[2 * _e + 1] + _bet[2 * _e + 1]
                                _packed.append(fp32_to_fp16(_ylo, _yhi, dtype=cutlass.BFloat16))
                            _wi += 4
                            st_shared_v4(smem_a_list[_KOP].data_ptr(_row_base + _phys * 8), _packed, cutlass.Int32)
                if cutlass.const_expr(_DUAL and _ln_work):
                    _row_words0 = []
                    _sum0 = cutlass.Float32(0.0)
                    if cutlass.const_expr(_LN_PF):
                        _tp0 = cutlass.Float32(0.0)
                        _tp1 = cutlass.Float32(0.0)
                        _tp2 = cutlass.Float32(0.0)
                        _tp3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            _pya0, _pyc0, _pya1, _pyc1, _pya2, _pyc2, _pya3, _pyc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqy = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nya0, _nyc0, _nya1, _nyc1, _nya2, _nyc2, _nya3, _nyc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqy)
                                if cutlass.const_expr(_LN_TREE):
                                    _yl0, _yh0 = _a0_pair_f32(_pya0, _pyc0, _sel)
                                    _yl1, _yh1 = _a0_pair_f32(_pya1, _pyc1, _sel)
                                    _yl2, _yh2 = _a0_pair_f32(_pya2, _pyc2, _sel)
                                    _yl3, _yh3 = _a0_pair_f32(_pya3, _pyc3, _sel)
                                    _tp0 = _tp0 + _yl0 + _yh0
                                    _tp1 = _tp1 + _yl1 + _yh1
                                    _tp2 = _tp2 + _yl2 + _yh2
                                    _tp3 = _tp3 + _yl3 + _yh3
                                else:
                                    for _pyax, _pycx in ((_pya0, _pyc0), (_pya1, _pyc1), (_pya2, _pyc2), (_pya3, _pyc3)):
                                        _ylx, _yhx = _a0_pair_f32(_pyax, _pycx, _sel)
                                        _sum0 = _sum0 + _ylx + _yhx
                                _pya0, _pyc0, _pya1, _pyc1, _pya2, _pyc2, _pya3, _pyc3 = _nya0, _nyc0, _nya1, _nyc1, _nya2, _nyc2, _nya3, _nyc3
                        if cutlass.const_expr(_LN_TREE):
                            _sum0 = (_tp0 + _tp1) + (_tp2 + _tp3)
                    elif cutlass.const_expr(_LN_SMEM):
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    for _e in cutlass.range_constexpr(4):
                                        _oaca = _q * 8 + 2 * _e
                                        _oawa = _a0w.data_ptr(_wb + _xow[2 * _e] + _oaca * 32).load()
                                        _oawc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_oaca + 1) * 32).load()
                                        _oaw0 = nvvm.inline_ptx(
                                            "prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_oawa, _oawc, _sel]
                                        )
                                        _oalo, _oahi = f16x2_to_f32(_oaw0, dtype=cutlass.BFloat16)
                                        _sum0 = _sum0 + _oalo + _oahi
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    for _e in cutlass.range_constexpr(4):
                                        _ca = _q * 8 + 2 * _e
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _ca * 32).load()
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_ca + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                        _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                        _sum0 = _sum0 + _lo + _hi
                    else:
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            for _p in cutlass.range_constexpr(cta_tile_mnk[2] // 2):
                                _ca = 2 * _p
                                _wa = _a0w.data_ptr(_wb + _xow[_ca & 7] + _ca * 32).load()
                                _wc = _a0w.data_ptr(_wb + _xow[(_ca + 1) & 7] + (_ca + 1) * 32).load()
                                _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                _row_words0.append(_w0)
                                _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                _sum0 = _sum0 + _lo + _hi
                    _mean0 = _sum0 * cutlass.Float32(_INV_K)
                    _sq0 = cutlass.Float32(0.0)
                    if cutlass.const_expr(_LN_PF):
                        _tq0 = cutlass.Float32(0.0)
                        _tq1 = cutlass.Float32(0.0)
                        _tq2 = cutlass.Float32(0.0)
                        _tq3 = cutlass.Float32(0.0)
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            _pza0, _pzc0, _pza1, _pzc1, _pza2, _pzc2, _pza3, _pzc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqz = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                _nza0, _nzc0, _nza1, _nzc1, _nza2, _nzc2, _nza3, _nzc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqz)
                                if cutlass.const_expr(_LN_TREE):
                                    _zl0, _zh0 = _a0_pair_f32(_pza0, _pzc0, _sel)
                                    _zl1, _zh1 = _a0_pair_f32(_pza1, _pzc1, _sel)
                                    _zl2, _zh2 = _a0_pair_f32(_pza2, _pzc2, _sel)
                                    _zl3, _zh3 = _a0_pair_f32(_pza3, _pzc3, _sel)
                                    _zdl0 = _zl0 - _mean0
                                    _zdh0 = _zh0 - _mean0
                                    _zdl1 = _zl1 - _mean0
                                    _zdh1 = _zh1 - _mean0
                                    _zdl2 = _zl2 - _mean0
                                    _zdh2 = _zh2 - _mean0
                                    _zdl3 = _zl3 - _mean0
                                    _zdh3 = _zh3 - _mean0
                                    _tq0 = _tq0 + _zdl0 * _zdl0 + _zdh0 * _zdh0
                                    _tq1 = _tq1 + _zdl1 * _zdl1 + _zdh1 * _zdh1
                                    _tq2 = _tq2 + _zdl2 * _zdl2 + _zdh2 * _zdh2
                                    _tq3 = _tq3 + _zdl3 * _zdl3 + _zdh3 * _zdh3
                                else:
                                    for _pzax, _pzcx in ((_pza0, _pzc0), (_pza1, _pzc1), (_pza2, _pzc2), (_pza3, _pzc3)):
                                        _zlx, _zhx = _a0_pair_f32(_pzax, _pzcx, _sel)
                                        _zdlx = _zlx - _mean0
                                        _zdhx = _zhx - _mean0
                                        _sq0 = _sq0 + _zdlx * _zdlx + _zdhx * _zdhx
                                _pza0, _pzc0, _pza1, _pzc1, _pza2, _pzc2, _pza3, _pzc3 = _nza0, _nzc0, _nza1, _nzc1, _nza2, _nzc2, _nza3, _nzc3
                        if cutlass.const_expr(_LN_TREE):
                            _sq0 = (_tq0 + _tq1) + (_tq2 + _tq3)
                    elif cutlass.const_expr(_LN_SMEM):
                        for _j in cutlass.range_constexpr(_NK):
                            _wb = (sA_elems // 2) * _sts[_j]
                            if cutlass.const_expr(_LN_ROLLED):
                                for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                    for _e in cutlass.range_constexpr(4):
                                        _obca = _q * 8 + 2 * _e
                                        _obwa = _a0w.data_ptr(_wb + _xow[2 * _e] + _obca * 32).load()
                                        _obwc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_obca + 1) * 32).load()
                                        _obw0 = nvvm.inline_ptx(
                                            "prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_obwa, _obwc, _sel]
                                        )
                                        _oblo, _obhi = f16x2_to_f32(_obw0, dtype=cutlass.BFloat16)
                                        _obdl = _oblo - _mean0
                                        _obdh = _obhi - _mean0
                                        _sq0 = _sq0 + _obdl * _obdl + _obdh * _obdh
                            else:
                                for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                    for _e in cutlass.range_constexpr(4):
                                        _ca = _q * 8 + 2 * _e
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _ca * 32).load()
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_ca + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                        _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                        _dl = _lo - _mean0
                                        _dh = _hi - _mean0
                                        _sq0 = _sq0 + _dl * _dl + _dh * _dh
                    else:
                        for _w0 in _row_words0:
                            _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                            _dl = _lo - _mean0
                            _dh = _hi - _mean0
                            _sq0 = _sq0 + _dl * _dl + _dh * _dh
                    _rstd0 = cute.math.rsqrt(_sq0 * cutlass.Float32(_INV_K) + cutlass.Float32(_EPS), fastmath=True)
                    if cutlass.const_expr(WANT_STATS):
                        if row_valid & (tile_n == 0):
                            _stats_addr0 = stats.iterator.toint() + (bi_idx * n_res + k_idx).to(cutlass.Int64) * cutlass.Int64(8)
                            st_global_v2(_stats_addr0, [_mean0, _rstd0], cutlass.Float32)
                    if cutlass.const_expr(_A0_TRANSPOSE):
                        nvvm.barrier_cta_sync(barrier_id=LN_SYNC_BAR_ID, thread_count=num_ln_warps * 32)
                    _wi = 0
                    for _j in cutlass.range_constexpr(_NK if _DIAG != "no_ln_norm" else 0):
                        _row_base0 = sA_elems * _sts[_j] + ln_tid * cta_tile_mnk[2]
                        _wb = (sA_elems // 2) * _sts[_j]
                        if cutlass.const_expr(_LN_SMEM and _LN_ROLLED):
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _occg = _j * cta_tile_mnk[2] + _q * 8
                                if cutlass.const_expr(_AFFINE_FOLD):
                                    _hg0 = _hg1 = _hg2 = _hg3 = _hg4 = _hg5 = _hg6 = _hg7 = _rstd0  # unused under the fold (the pure (x - mean) * rstd)
                                    _hb0 = _hb1 = _hb2 = _hb3 = _hb4 = _hb5 = _hb6 = _hb7 = _rstd0
                                else:
                                    _hg0, _hg1, _hg2, _hg3 = ld_shared_v4(smem_ln_params.data_ptr(_occg), cutlass.Float32)
                                    _hg4, _hg5, _hg6, _hg7 = ld_shared_v4(smem_ln_params.data_ptr(_occg + 4), cutlass.Float32)
                                    _hb0, _hb1, _hb2, _hb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _occg), cutlass.Float32)
                                    _hb4, _hb5, _hb6, _hb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _occg + 4), cutlass.Float32)
                                for _e in cutlass.range_constexpr(4):
                                    _occl = _q * 8 + 2 * _e
                                    _ocwa = _a0w.data_ptr(_wb + _xow[2 * _e] + _occl * 32).load()
                                    _ocwc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_occl + 1) * 32).load()
                                    _ocw0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_ocwa, _ocwc, _sel])
                                    _oclo, _ochi = f16x2_to_f32(_ocw0, dtype=cutlass.BFloat16)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _ocylo = (_oclo - _mean0) * _rstd0
                                        _ocyhi = (_ochi - _mean0) * _rstd0
                                    else:
                                        _oc_gl = (_hg0, _hg1, _hg2, _hg3, _hg4, _hg5, _hg6, _hg7)[2 * _e]
                                        _oc_gh = (_hg0, _hg1, _hg2, _hg3, _hg4, _hg5, _hg6, _hg7)[2 * _e + 1]
                                        _ocbl = (_hb0, _hb1, _hb2, _hb3, _hb4, _hb5, _hb6, _hb7)[2 * _e]
                                        _ocbh = (_hb0, _hb1, _hb2, _hb3, _hb4, _hb5, _hb6, _hb7)[2 * _e + 1]
                                        _ocylo = (_oclo - _mean0) * _rstd0 * _oc_gl + _ocbl
                                        _ocyhi = (_ochi - _mean0) * _rstd0 * _oc_gh + _ocbh
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e] + _occl * cta_tile_mnk[2]).store(_ocylo.to(cutlass.BFloat16))
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e + 1] + (_occl + 1) * cta_tile_mnk[2]).store(
                                        _ocyhi.to(cutlass.BFloat16)
                                    )
                        elif cutlass.const_expr(_LN_PF):
                            if cutlass.const_expr(_AFFINE_FOLD):
                                _pyg0 = _pyg1 = _pyg2 = _pyg3 = _pyg4 = _pyg5 = _pyg6 = _pyg7 = _rstd0  # unused under the fold (the pure (x - mean) * rstd)
                                _pyb0 = _pyb1 = _pyb2 = _pyb3 = _pyb4 = _pyb5 = _pyb6 = _pyb7 = _rstd0
                            else:
                                _pyg0, _pyg1, _pyg2, _pyg3 = ld_shared_v4(smem_ln_params.data_ptr(_j * cta_tile_mnk[2]), cutlass.Float32)
                                _pyg4, _pyg5, _pyg6, _pyg7 = ld_shared_v4(smem_ln_params.data_ptr(_j * cta_tile_mnk[2] + 4), cutlass.Float32)
                                _pyb0, _pyb1, _pyb2, _pyb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _j * cta_tile_mnk[2]), cutlass.Float32)
                                _pyb4, _pyb5, _pyb6, _pyb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _j * cta_tile_mnk[2] + 4), cutlass.Float32)
                            _pva0, _pvc0, _pva1, _pvc1, _pva2, _pvc2, _pva3, _pvc3 = _a0_chunk_words(_a0w, _wb, _xow, 0)
                            for _q in cutlass.range(cta_tile_mnk[2] // 8, unroll=1):  # unroll=1 = NO unrolling (the rolled form's point)
                                _nqv = (_q + 1) & (cta_tile_mnk[2] // 8 - 1)
                                if cutlass.const_expr(_AFFINE_FOLD):
                                    _nyg0 = _nyg1 = _nyg2 = _nyg3 = _nyg4 = _nyg5 = _nyg6 = _nyg7 = _rstd0
                                    _nyb0 = _nyb1 = _nyb2 = _nyb3 = _nyb4 = _nyb5 = _nyb6 = _nyb7 = _rstd0
                                else:
                                    _nccg = _j * cta_tile_mnk[2] + _nqv * 8
                                    _nyg0, _nyg1, _nyg2, _nyg3 = ld_shared_v4(smem_ln_params.data_ptr(_nccg), cutlass.Float32)
                                    _nyg4, _nyg5, _nyg6, _nyg7 = ld_shared_v4(smem_ln_params.data_ptr(_nccg + 4), cutlass.Float32)
                                    _nyb0, _nyb1, _nyb2, _nyb3 = ld_shared_v4(smem_ln_params.data_ptr(_K + _nccg), cutlass.Float32)
                                    _nyb4, _nyb5, _nyb6, _nyb7 = ld_shared_v4(smem_ln_params.data_ptr(_K + _nccg + 4), cutlass.Float32)
                                _nva0, _nvc0, _nva1, _nvc1, _nva2, _nvc2, _nva3, _nvc3 = _a0_chunk_words(_a0w, _wb, _xow, _nqv)
                                for _e in cutlass.range_constexpr(4):
                                    _pvcl = _q * 8 + 2 * _e
                                    _pvlo, _pvhi = _a0_pair_f32((_pva0, _pva1, _pva2, _pva3)[_e], (_pvc0, _pvc1, _pvc2, _pvc3)[_e], _sel)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _pvylo = (_pvlo - _mean0) * _rstd0
                                        _pvyhi = (_pvhi - _mean0) * _rstd0
                                    else:
                                        _pvgl = (_pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7)[2 * _e]
                                        _pvgh = (_pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7)[2 * _e + 1]
                                        _pvbl = (_pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7)[2 * _e]
                                        _pvbh = (_pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7)[2 * _e + 1]
                                        _pvylo = (_pvlo - _mean0) * _rstd0 * _pvgl + _pvbl
                                        _pvyhi = (_pvhi - _mean0) * _rstd0 * _pvgh + _pvbh
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e] + _pvcl * cta_tile_mnk[2]).store(_pvylo.to(cutlass.BFloat16))
                                    smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[2 * _e + 1] + (_pvcl + 1) * cta_tile_mnk[2]).store(
                                        _pvyhi.to(cutlass.BFloat16)
                                    )
                                _pyg0, _pyg1, _pyg2, _pyg3, _pyg4, _pyg5, _pyg6, _pyg7 = _nyg0, _nyg1, _nyg2, _nyg3, _nyg4, _nyg5, _nyg6, _nyg7
                                _pyb0, _pyb1, _pyb2, _pyb3, _pyb4, _pyb5, _pyb6, _pyb7 = _nyb0, _nyb1, _nyb2, _nyb3, _nyb4, _nyb5, _nyb6, _nyb7
                                _pva0, _pvc0, _pva1, _pvc1, _pva2, _pvc2, _pva3, _pvc3 = _nva0, _nvc0, _nva1, _nvc1, _nva2, _nvc2, _nva3, _nvc3
                        else:
                            for _q in cutlass.range_constexpr(cta_tile_mnk[2] // 8):
                                _cg = _j * cta_tile_mnk[2] + _q * 8  # the global channel of this 8-chunk (operand 0's gamma / beta sit at [0, 2 K))
                                _gam = []
                                _bet = []
                                if cutlass.const_expr(not _AFFINE_FOLD):
                                    _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_cg), cutlass.Float32))
                                    _gam.extend(ld_shared_v4(smem_ln_params.data_ptr(_cg + 4), cutlass.Float32))
                                    _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _cg), cutlass.Float32))
                                    _bet.extend(ld_shared_v4(smem_ln_params.data_ptr(_K + _cg + 4), cutlass.Float32))
                                _packed = []
                                for _e in cutlass.range_constexpr(4):
                                    _cl = _q * 8 + 2 * _e  # the stage-local channel of _lo; _hi is the next channel
                                    if cutlass.const_expr(_LN_SMEM):
                                        _wa = _a0w.data_ptr(_wb + _xow[2 * _e] + _cl * 32).load()  # the re-read form, unrolled
                                        _wc = _a0w.data_ptr(_wb + _xow[2 * _e + 1] + (_cl + 1) * 32).load()
                                        _w0 = nvvm.inline_ptx("prmt.b32 $0, $1, $2, $3;", write_only_types=[cutlass.Int32], read_only_args=[_wa, _wc, _sel])
                                    else:
                                        _w0 = _row_words0[_wi + _e]
                                    _lo, _hi = f16x2_to_f32(_w0, dtype=cutlass.BFloat16)
                                    if cutlass.const_expr(_AFFINE_FOLD):
                                        _ylo = (_lo - _mean0) * _rstd0
                                        _yhi = (_hi - _mean0) * _rstd0
                                    else:
                                        _ylo = (_lo - _mean0) * _rstd0 * _gam[2 * _e] + _bet[2 * _e]
                                        _yhi = (_hi - _mean0) * _rstd0 * _gam[2 * _e + 1] + _bet[2 * _e + 1]
                                    if cutlass.const_expr(_A0_TRANSPOSE):
                                        _packed.append(fp32_to_fp16(_ylo, _yhi, dtype=cutlass.BFloat16))
                                    else:
                                        smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[_cl & 7] + _cl * cta_tile_mnk[2]).store(_ylo.to(cutlass.BFloat16))
                                        smem_a_list[0].data_ptr(sA_elems * _sts[_j] + _xoe[(_cl + 1) & 7] + (_cl + 1) * cta_tile_mnk[2]).store(
                                            _yhi.to(cutlass.BFloat16)
                                        )
                                _wi += 4
                                if cutlass.const_expr(_A0_TRANSPOSE):
                                    st_shared_v4(smem_a_list[0].data_ptr(_row_base0 + (_q ^ ln_swz) * 8), _packed, cutlass.Int32)
            if cutlass.const_expr(SKIP_MASKED_TILES):
                if is_pair_leader:
                    _m_self = _row_mask(mask, row_valid, bi_idx, k_idx, n_res)
                    _k2, _bi2 = _row_coords(tile_m, n_kb, lkb, r_pair ^ cta_tile_mnk[0])
                    _valid2 = (_k2 < n_res) & (_bi2 < bn)
                    _m_peer = _row_mask(mask, _valid2, _bi2, _k2, n_res)
                    _ballot = cute.arch.vote_ballot_sync((_m_self != cutlass.Float32(0.0)) | (_m_peer != cutlass.Float32(0.0)))
                    _vslot = (tile_iter & 1) * 4 + ln_warp
                    if ln_lane == 0:
                        smem_vote.data_ptr(_vslot).store(cutlass.Int32(_ballot))
                    nvvm.barrier_cta_sync(barrier_id=LN_SYNC_BAR_ID, thread_count=num_ln_warps * 32)
                    _vbase = (tile_iter & 1) * 4
                    _any = (
                        smem_vote.data_ptr(_vbase).load()
                        | smem_vote.data_ptr(_vbase + 1).load()
                        | smem_vote.data_ptr(_vbase + 2).load()
                        | smem_vote.data_ptr(_vbase + 3).load()
                    )
                    if ln_warp == 0:
                        if elect_one:  # the arriving lane stores the flag: its release covers its own store directly
                            smem_flags.data_ptr(_sts[0]).store(
                                cutlass.Int32(arith.select((_any == 0).ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
                            )
            cute.arch.fence_proxy("async.shared", space="cta")
            nvvm.bar_warp_sync(0xFFFFFFFF)
            for _j in cutlass.range_constexpr(_NK):
                if elect_one:
                    arrive_on_leader_release(a_norm_full_mbar_ptr.subview(_sts[_j]), pair_leader_rank, cta_group)
            # (8) consume the CLC response (the same protocol as every consumer warp)
            consumer_stage = tile_iter % CLC_SCHED_STAGES
            if consumer_stage == 0 and tile_iter != 0:
                clc_full_phase_ln = clc_full_phase_ln ^ 1
            while not nvvm.mbarrier_try_wait_parity(
                clc_full_mbar_ptr.subview(consumer_stage),
                clc_full_phase_ln,
                time_limit=10_000_000,
            ):
                pass
            m_idx, n_idx, l_idx, vld = cute_clc.clc_response(clc_response_ptr_base + consumer_stage)
            cute.arch.fence_proxy("async.shared", space="cta")
            is_valid = vld
            ln_raw_m = m_idx >> _preferred_cluster_m_shift
            ln_raw_n = n_idx >> _preferred_cluster_n_shift
            ln_nt_m = gridx >> _preferred_cluster_m_shift
            ln_nt_n = gridy >> _preferred_cluster_n_shift
            tile_m, tile_n = _l2_swizzle_tile(
                ln_raw_m,
                ln_raw_n,
                ln_nt_m,
                ln_nt_n,
                swizzle_w,
                identity=tile_swizzle_n == 1,
            )
            tile_l = l_idx
            tile_m, tile_l = _decode_pass(tile_m, tile_l)
            nvvm.bar_warp_sync(0xFFFFFFFF)
            if elect_one:
                empty_remote = nvvm.mapa(clc_empty_mbar_ptr.subview(consumer_stage), 0)
                nvvm.mbarrier_arrive(empty_remote, scope=nvvm.MemScope.CLUSTER, relaxed=True)
            tile_iter += 1


frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host(
    geom: tuple,
    a0: cute.Tensor,  # operand 0: K1 = z [bn * n_res * K] bf16 (pair-major rows); K3 = y [c * bn * n_pad] bf16 (channel-major, MN-major rows)
    a1: Optional[cute.Tensor],  # operand 1 (K3): z [bn * n_res * K] bf16; None in the single-operand arm
    w0: cute.Tensor,  # K1: [passes * 2 c_out * K] (per pass: rows interleaved per 128-channel block); K3: W_value [c_out * K]
    w1: Optional[cute.Tensor],  # K3: W_gate [c_out * K]; None in the single-operand arm
    out: cute.Tensor,  # K1: [passes * c_out * bn * n_pad] channel-major planes; K3: [bn * n_res * c_out] pair-major
    ln_w: cute.Tensor,
    ln_b: cute.Tensor,
    ln_w1: Optional[cute.Tensor],
    ln_b1: Optional[cute.Tensor],
    bias: Optional[cute.Tensor],
    mask: Optional[cute.Tensor],
    stats: Optional[cute.Tensor],
    resid: Optional[cute.Tensor],
    stream: _cuda.CUstream,
) -> None:
    m_rows = geom[0]  # the caller's flat M: pair tiles x 256
    n = geom[1]  # the GEMM N: K1 = 2 * c_out (value | gate interleaved), K3 = c_out per GEMM
    k_sym = geom[2]  # K
    batch = geom[3]  # passes
    # Select scheduler grid axes.
    m = (m_rows * _PASSES) if cutlass.const_expr(_PASS_IN_M) else m_rows
    grid_z = 1 if cutlass.const_expr(_PASS_IN_M or _LN_ONCE) else batch  # LN ONCE: the pass lives inside the work item
    n_res = cutlass.Int32(geom[4])
    bn = cutlass.Int32(geom[5])
    n_pad = cutlass.Int32(geom[6])
    lkb = cutlass.Int32(geom[7])
    n_kb = cutlass.Int32(geom[8])
    kbc = cutlass.Int32(geom[9])  # k's per CTA box (min(kb, 128))
    rc = cutlass.Int32(geom[10])  # bi's per CTA box (128 // kbc)
    bpe = ab_dtype.width // 8

    if cutlass.const_expr(_DUAL):
        tma_a_desc_0 = _tma.create_tensor_map_tiled(
            global_address=a0.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[n_res, bn, k_sym],
            global_strides=[n_pad.to(cutlass.Int64) * bpe // 16, bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16],
            box_dims=[_MN_GROUP, 1, cta_tile_mnk[2]],
            swizzle=ab_tma_swizzle,
        )
        # K3 operand 1: z viewed [bn][n_res][K] (channel, k, bi) -- the K-major box (64, kbc, rc) lands as the 128 x 64 stage (K1's A).
        tma_a_desc_1 = _tma.create_tensor_map_tiled(
            global_address=a1.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n_res, bn],
            global_strides=[k_sym * bpe // 16, n_res.to(cutlass.Int64) * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], kbc, rc],
            swizzle=ab_tma_swizzle,
        )
    else:
        # A: z viewed [bn][n_res][K] -- dims innermost first (channel, k, bi); the CTA box (64, kbc, rc) lands as the 128 x 64 stage.
        tma_a_desc_0 = _tma.create_tensor_map_tiled(
            global_address=a0.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n_res, bn],
            global_strides=[k_sym * bpe // 16, n_res.to(cutlass.Int64) * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], kbc, rc],  # dynamic box dims: the builder takes them as Int32 (an Int8 cannot hold 128)
            swizzle=ab_tma_swizzle,
        )
        tma_a_desc_1 = tma_a_desc_0  # unused by the single-operand arm (the kernel signature is shared)
    tma_b_desc_0 = _tma.create_tensor_map_tiled(
        global_address=w0.iterator.toint(),
        dtype=ab_tma_dtype,
        global_dims=[k_sym, n, batch],
        global_strides=[k_sym * bpe // 16, n * k_sym * bpe // 16],
        box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
        swizzle=ab_tma_swizzle,
    )
    if cutlass.const_expr(_DUAL):
        tma_b_desc_1 = _tma.create_tensor_map_tiled(
            global_address=w1.iterator.toint(),
            dtype=ab_tma_dtype,
            global_dims=[k_sym, n, batch],
            global_strides=[k_sym * bpe // 16, n * k_sym * bpe // 16],
            box_dims=[cta_tile_mnk[2], cta_tile_mnk[1] // b_mcast_slices, 1],
            swizzle=ab_tma_swizzle,
        )
    else:
        tma_b_desc_1 = tma_b_desc_0
    if cutlass.const_expr(_DUAL):
        # C (K3): out viewed [bn][n_res][c_out] -- dims (channel, k, bi); the s64b box (32, kbc, rc) from the [128 rows][32 ch] swizzled staging.
        tma_c_desc_0 = _tma.create_tensor_map_tiled(
            global_address=out.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=[n, n_res, bn],
            global_strides=[n * bpe // 16, n_res.to(cutlass.Int64) * n * bpe // 16],
            box_dims=[epi_n, kbc, rc],
            swizzle=_tma.TensorMapSwizzle.s64b,
        )
    else:
        # C: out viewed [passes][c_out][bn][n_pad] -- dims (k, bi, channel, pass); the staging buffer is [32 channels][rc][kbc] dense.
        c_out_i64 = n // 2
        tma_c_desc_0 = _tma.create_tensor_map_tiled(
            global_address=out.iterator.toint(),
            dtype=cutlass.BFloat16,
            global_dims=[n_pad, bn, cutlass.Int32(c_out_i64), cutlass.Int32(batch)],
            global_strides=[
                n_pad.to(cutlass.Int64) * bpe // 16,
                bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16,
                c_out_i64 * bn.to(cutlass.Int64) * n_pad.to(cutlass.Int64) * bpe // 16,
            ],
            box_dims=[kbc, rc, epi_n, 1],
            swizzle=_tma.TensorMapSwizzle.none,
        )

    cluster_m = cluster_shape_mnk[0]
    cluster_n = cluster_shape_mnk[1]
    cgrp_tile_m = cgrp_tile_mnk[0]
    cgrp_tile_n = cgrp_tile_mnk[1]
    num_tile_m_host = (m + cgrp_tile_m - 1) // cgrp_tile_m
    num_tile_n_host = (n + cgrp_tile_n - 1) // cgrp_tile_n
    if cutlass.const_expr(_LN_ONCE):
        num_tile_n_host = num_tile_n_host // _NTI  # LN ONCE: one scheduler item per (row block, N-tile group)
    grid_x = num_tile_m_host * cluster_m
    grid_y = num_tile_n_host * cluster_n
    grid_shape = (grid_x, grid_y, grid_z)
    launch = frost_ln_dual_gemm_128x256x128_cluster2x1_2ctamma(
        m,
        n,
        k_sym,
        tma_a_desc_0,
        tma_a_desc_1,
        tma_b_desc_0,
        tma_b_desc_1,
        tma_c_desc_0,
        n_res,
        bn,
        n_pad,
        lkb,
        n_kb,
        ln_w,
        ln_b,
        ln_w1,
        ln_b1,
        bias,
        mask,
        stats,
        resid,
    )
    launch.launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
        cluster=cluster_shape_mnk,
        use_pdl=USE_PDL,
        min_blocks_per_mp=_MIN_BLOCKS_PER_MP,
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile() -> Callable:
    """cute-compile ``_host`` for this module's PARAMS.  Shapes are symbolic; the geometry tuple is
    ``(m_sched, gemm_n, K, passes, n_res, bn, n_pad, lkb, n_kb, kbc, rc)`` -- the host module builds it."""
    sym = [cute.sym_int64() for _ in range(11)]
    flat = lambda dt, align: make_fake_compact_tensor(dt, (cute.sym_int64(),), stride_order=(0,), assumed_align=align)  # noqa: E731
    fake_a0 = flat(cutlass.BFloat16, 16)
    fake_a1 = flat(cutlass.BFloat16, 16) if _DUAL else None
    fake_w0 = flat(cutlass.BFloat16, 16)
    fake_w1 = flat(cutlass.BFloat16, 16) if _DUAL else None
    fake_out = flat(cutlass.BFloat16, 16)
    fake_ln_w = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16)
    fake_ln_b = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16)
    fake_ln_w1 = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16) if _DUAL else None
    fake_ln_b1 = make_fake_compact_tensor(cutlass.BFloat16, (_K,), stride_order=(0,), assumed_align=16) if _DUAL else None
    fake_bias = make_fake_compact_tensor(cutlass.BFloat16, (_PASSES * _BIAS_PER_PASS,), stride_order=(0,), assumed_align=16) if HAS_BIAS else None
    fake_mask = flat(cutlass.BFloat16, 16) if HAS_MASK else None
    fake_stats = flat(cutlass.Float32, 16) if WANT_STATS else None
    fake_resid = flat(cutlass.BFloat16, 16) if _FUSE_RESIDUAL else None
    _fake_stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    return cute.compile(
        _host,
        tuple(sym),
        fake_a0,
        fake_a1,
        fake_w0,
        fake_w1,
        fake_out,
        fake_ln_w,
        fake_ln_b,
        fake_ln_w1,
        fake_ln_b1,
        fake_bias,
        fake_mask,
        fake_stats,
        fake_resid,
        stream=_fake_stream,
        options=frost_compile_options,
    )
