# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL prefill SDPA kernel — block-scale MXFP8 (E4M3 / E5M2), d=128, SM100 (Blackwell).

Classic 2-sub-tile pipeline (TILES_Q=2, two softmax warpgroups, four correction
warps, persistent try_cancel scheduler); shares the f16 / fp8 SM100 d=128 layout
plus the MXFP8 TMEM scale-factor (SF) relayout:
  1. **cga2-only, STAGES_KV=4** (config).
  2. **512-col TMEM.**  S(256)+O(256) fill the cap, so the SF operands cannot
     park above 512; instead they are reloaded every BMM1 into transiently-free
     S_acc scratch — prologue BMM1 uses the O region (O not yet accumulated),
     steady-state BMM1 ping-pongs into the *other* sub-tile's S_acc lower region,
     gated on that tile's softmax-LDTM-done (``mb_softmax_ldtm_done``).  BMM2
     SF_P/V ride the existing bmm2_ready wait.  Stats sit on the S_acc heads
     (col 0/128, SF_STATS_OFFSET=8 clears them).
  3. **MXFP8 MMA uses the Blackwell K=32 QMMA path** — ``_MXFP8_K_DIM=0`` +
     TILE_K_HW=32.
  4. **Row-max**: cc10.0 uses a manual tcgen05_ld + software reduction; cc10.3+
     (FUSED_LDTM_STAT, set from the device capability) fuses load + row-max into
     one tcgen05.ld.red.f32.max for unmasked tiles.
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass, replace
from typing import NamedTuple

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, make_cfg_d128

# The template loader (api_dsl._load_kernel_module) injects FROST_TEMPLATE_PARAMS
# as a module global before this body runs; the default keeps direct import usable.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d128(PARAMS)
if PARAMS.softmax_f16:
    raise ValueError("prefill_d128_mxfp8_sm100: softmax_f16 is per-tensor-FP8-on-SM107 only (softmax_precision knob domain)")
Cfg = type(CFG)
# LDTM.STAT — fused `tcgen05.ld.red.f32.max` (S_acc load + row-max in one op) — is a
# cc10.3+ capability; cc10.0 lacks it and uses the manual tcgen05_ld + software
# row_max_reduction_64 (default 0). api_dsl sets this from the device capability at
# compile time; a distinct value yields a distinct kernel specialization (cache key).
FUSED_LDTM_STAT = int(PARAMS.fused_ldtm_stat)
# THD / varlen (CFG.THD_VARLEN=1) follows the device-built-metadata +
# plan-time-envelope design (write_thd_meta, issue #552 / PRs #606, #608) used
# by the f16 kernels: packed [1,T,H,D] Q/K/V/O with DYNAMIC token extents, the
# setup kernel builds the [kv|cu_q|cu_k] metadata + per-batch O TMA descriptors
# device-side, and the launch grid is the plan-time envelope (dead units decode
# the batch == n_batch sentinel and drain).  MXFP8-only addition: the SF
# tensors travel PACKED per-sequence-TILE-padded (the shared MXFP8 quantizer
# writes SF tiles at cu_sf[b] + local_tile; this kernel reads them with the
# matching tile base from _thd_sf_tile_bases) as [1, H, Σ_b ceil(S_b/TILE),
# SF_SMEM] with a DYNAMIC packed tile extent — the SF descriptors use B=1 and
# the bound view's tile extent.  Dense path byte-identical.
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# O TMA box params follow O's swizzle, not V's — required when V_SWZ_B != O_SWZ_B.
# Sized in BPE_O — O may be written at BF16/FP16 when DTYPE_O != DTYPE_QKV.
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) — still used for sched.mb_* + mb_softmax_ldtm (not in Bars).
    wait,
    arrive_on_leader,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    read_tile_id_arrive,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    # cc10.0: the MASK_NONE fast path uses manual tcgen05_ld + row_max_reduction_64.
    # cc10.3+ (FUSED_LDTM_STAT) fuses load + row-max into tmem_load_max_reduction_x64.
    row_reduction_pair_64,
    row_max_reduction_64,
    tmem_load_max_reduction_x64,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    bulk_copy,
    bulk_copy_multicast,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, GmemTileLinear, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)

# MXFP8 storage dtype dispatch — keyed off CFG.DTYPE_QKV (0=E4M3, 1=E5M2).
if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
else:
    raise ValueError(f"prefill_sdpa_mxfp8: DTYPE_QKV={CFG.DTYPE_QKV} not supported " f"(expected 0=E4M3 or 1=E5M2)")

# P -> fp8 cast bias (BAKED constant — MXFP8's block SFs cover Q/K/V inputs,
# not P). P is quantized as P * 2**P_CAST_LOG2_SCALE: the lazy-rescale skip
# bounds P by 2**RESCALE_THRESHOLD (4.0 for fp8 dtypes), so the cast peaks at
# 2^(4+4) = 256 < 448 (e4m3 max) — no saturation — while flat-row entries
# (P ~ 1/S) sit four binades above e4m3's subnormal cliff. The bias rides the
# exp2 argument, so total_sum accumulates in the SAME 2^4-scaled units and
# the O normalization (O_acc / total_sum) cancels it exactly; only the LSE
# subtracts the constant (and the sink denominator term is scaled to match).
# Invariant: RESCALE_THRESHOLD + P_CAST_LOG2_SCALE <= log2(448).
P_CAST_LOG2_SCALE = 4.0

MMA_KIND = nvvm.MMABlockScaleKind.MXF8F6F4
SCALE_VEC_SIZE = nvvm.Tcgen05MMAScaleVecSize.BLOCK32

# SM100/Blackwell MXFP8: k_dim=0 = K=32 QMMA path (NOT the k_dim=1 K=64
# fast path, which is silently WRONG on Blackwell — cuda-kernels rules §16).
# BMM1 = TILE_K/32 k-steps, BMM2 = TILE_N/32 (both derive from _MXFP8_TILE_K_HW).
_MXFP8_K_DIM = 0
_MXFP8_TILE_K_HW = 32 if _MXFP8_K_DIM == 0 else 64

# DTYPE_O is independent of DTYPE_QKV — MXFP8 input may write BF16/FP16 O so a
# downstream consumer skips a dequant.  BPE_O ∈ {1, 2}; the epilogue already
# casts via .to(OUT_STORAGE_DTYPE) + store_swizzled (dtype-generic), so only the
# O-buffer / TMA-box sizing needs to follow BPE_O.
if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_mxfp8: DTYPE_O={CFG.DTYPE_O} not supported " f"(expected 0=E4M3 / 1=E5M2 / 2=BF16 / 3=FP16)")


# MXFP8 SF constants — E8M0 scale factor, 32 elems share one SF.
BITS_PER_SF_ELEMENT = 8
BLOCK_SCALE_BLOCK_SIZE = 32
SF_BLOCK_DIM_NON_K = 128
SF_BLOCK_DIM_K = 4
SF_SWIZZLED_BLOCK_DIM_K = 16
SF_BYTES_PER_BLOCK = SF_BLOCK_DIM_NON_K * SF_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 8


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


# SF_NUM_BLOCKS_K: 1 (TILE_K=128).
SF_NUM_BLOCKS_M = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_N = CFG.TILE_N // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K = _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K

SF_REGISTERS_PER_BLOCK = SF_SWIZZLED_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 32

SF_TMEM_COLS_Q = SF_NUM_BLOCKS_M * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK
SF_TMEM_COLS_K = SF_NUM_BLOCKS_N * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK

SF_SMEM_SIZE_Q = CFG.TILE_M * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_K = CFG.TILE_N * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE

# BMM2 SF: P is constant 1.0 (softmax doesn't scale P); V comes from GMEM.
SF_NUM_BLOCKS_P = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_V = _round_up(CFG.TILE_O, 128) // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K_BMM2 = CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K
SF_TMEM_COLS_P = SF_NUM_BLOCKS_P * SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK
SF_SMEM_SIZE_P = _round_up(CFG.TILE_M, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_V = _round_up(CFG.TILE_O, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE

# 0x7F = E8M0 representation of 2^0 = 1.0.
SF_CONST_VALUE = 0x7F


from cudnn.sdpa.fwd.kernels._common_sm100 import (
    make_split_helpers,
    Bars,
    KvLoopBounds,
    make_classic_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
    assert_tile_n_supported,
)

assert_tile_n_supported(CFG)


CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA


# SM100 is always cga2 → LPT reverse-row count in CGA-tile units.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
_bounds_for_tile = _sdpa_h.bounds_for_tile
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; the setup kernel + TENSOR_MAP_QWORDS from the shared
# kernels thd_sm100.py.  Gated by CFG.THD_VARLEN (folds out: _thd_tma_offsets
# is (0, 0, batch_idx) and _thd_sf_tile_bases (0, 0) dense — TMA coords
# byte-identical).  Supported at cga1 and cga2 (TILES_Q=2 → two Q slabs /
# O stores per tile).  seq_kv_lens overloaded as the THD metadata buffer
# (int32 len 3B+2): [0..B-1]=seq_kv_lens [B..2B]=cu_q(B+1) [2B+1..3B+1]=cu_k(B+1)
# — built DEVICE-side by the setup kernel (issue #552).
# MXFP8-only: _thd_sf_tile_bases returns the per-sequence SF-tile prefix bases
# (cu_sf_q_base / cu_sf_k_base) for the packed scale-factor layout.
from cudnn.sdpa.fwd.kernels.thd_sm100 import build_thd_meta_o_kv_descs_kernel as _build_thd_meta_o_kv_descs_kernel, TENSOR_MAP_QWORDS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

# === PackGQA ===
# PackGQA is not supported for MXFP8 with the current SF contract: sf_q arrives
# as one opaque 512-byte F8_128x4 atom per (batch, head, 128-row tile) with
# intra-atom byte order (r%32)*16 + (r//32)*4 + c. The finest TMA-addressable
# unit inside an atom (16 B) bundles source rows {m, m+32, m+64, m+96} of one
# head, while a packed tile row j needs token j//G of head j%G — expressing that
# gather needs a 4-byte-stride dim (illegal: TMA global strides are 16-B units)
# and 6 dims (> the 5-dim TMA max).
# Unblock options (all change the contract or add a pass): a packed upstream SF
# layout, a device-side repack kernel, or an in-kernel quadrant gather + SMEM
# permute.
if CFG.PACK_GQA:
    raise ValueError("prefill_d128_mxfp8_sm100: PACK_GQA is unsupported (F8_128x4 sf_q atom is not TMA-gatherable at token granularity)")

# === KV split ===
#
# Mechanics live in _common_sm100.make_split_helpers, shared with the other
# SM100 prefill flavors: each Q tile's KV loop range is cut into SPLIT_KV
# contiguous chunks, each run as its own persistent tile, and each writing a
# normalized partial O + its own LSE into a split-major workspace that
# split_combine_sm100 folds with the exact log-sum-exp identity.  At
# SPLIT_KV == 1 every closure folds away and this is the classic kernel.


@cute.jit
def _bounds_for_tile_uniform(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_tensor, batch_idx, qh_per_kh: int = 1):
    """Uniform bounds signature for make_split_helpers.

    This flavor has no dead-Q-tile trim, so the seq-lens args are accepted
    and ignored (callers pass None for them).
    """
    return _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh)


_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile_uniform,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch
_thd_sf_tile_bases = _sdpa_h.thd_sf_tile_bases


# Kernel TMEM layout — P aliases S_acc tail (4:1 fp8 pack); SF tiles cols 512..543; Stats outside S/O range.
@dataclass(frozen=True)
class KernelTmemLayout:
    # Blackwell SM10.0 512-col TMEM cap.
    TOTAL_COLS: int = 512

    S0_OFF: int = 0
    S1_OFF: int = 128

    O0_OFF: int = 256
    O1_OFF: int = 384

    # FP8 P is 4:1-packed at the S_acc tails (96 / 224); the lower halves of each
    # S_acc slot (< P) are free for stats + reloaded SF (cuDNN mxfp8 scheme).
    P0_OFF: int = 96
    P1_OFF: int = 224

    # Stats on the S_acc heads (col 0 / 128); SF_STATS_OFFSET clears them.
    STATS_OFF: int = 0
    STATS_STRIDE: int = 128
    SF_STATS_OFFSET: int = 8
    SF_AFTER_P_OFFSET: int = 32

    # --- MXFP8 SF operands.  512-col cap leaves no room above S∪O, so SF is
    # RELOADED every MMA into transiently-free scratch (all offsets derive from
    # SF_TMEM_COLS_*).
    #
    # Steady-state BMM1: SF_Q/K ping-pong into the OTHER sub-tile's S_acc lower
    # region, gated on that tile's softmax-LDTM-done (mb_softmax_ldtm_done):
    #   sub0 (writes S_acc_0) -> hides SF in S_acc_1 head; sub1 -> S_acc_0 head.
    SF_Q0_OFF: int = 128 + SF_STATS_OFFSET  # 136
    SF_K0_OFF: int = 128 + SF_STATS_OFFSET + SF_TMEM_COLS_Q  # 140
    SF_Q1_OFF: int = SF_STATS_OFFSET  # 8
    SF_K1_OFF: int = SF_STATS_OFFSET + SF_TMEM_COLS_Q  # 12
    # Prologue BMM1: O not yet accumulated -> O_0 region is dead scratch (the
    # first BMM2 overwrites it with accumulate=False).  No LDTM wait needed.
    SF_Q0_PRO_OFF: int = 256  # O_0 head
    SF_K0_PRO_OFF: int = 256 + SF_TMEM_COLS_Q  # 260
    SF_Q1_PRO_OFF: int = 256 + SF_TMEM_COLS_Q + SF_TMEM_COLS_K  # 264
    SF_K1_PRO_OFF: int = 256 + 2 * SF_TMEM_COLS_Q + SF_TMEM_COLS_K  # 268
    # BMM2 SF_P/V per sub-tile (S_acc region + SF_AFTER_P_OFFSET); the existing
    # bmm2_ready wait already proves softmax read all of S_acc, so no extra gate.
    SF_P0_OFF: int = SF_AFTER_P_OFFSET  # 32
    SF_V0_OFF: int = SF_AFTER_P_OFFSET + SF_TMEM_COLS_P  # 36
    SF_P1_OFF: int = 128 + SF_AFTER_P_OFFSET  # 160
    SF_V1_OFF: int = 128 + SF_AFTER_P_OFFSET + SF_TMEM_COLS_P  # 164


LAYOUT = KernelTmemLayout()


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

# SF SmemDesc layout: NONE=0 is real no-swizzle (NOT value 1 which is SWIZZLE_128B_ATOM_32B).
SMEM_LAYOUT_SF = 0
SF_LEADING_BYTE_OFFSET = 16
SF_STRIDE_BYTE_OFFSET = 128

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)
LEADING_BYTE_OFFSET_QK = 0

_MMA_K_FP8 = _MXFP8_TILE_K_HW
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N*V_SWZ_BYTES
_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // _MMA_K_FP8
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


# === Kernel ===


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    # SF rides cta_group=CFG.CTA_MMA TMA → bytes route through mb_q/k/v_full like Q/K/V; no peer Q_SF DSMEM forward.
    tma_q_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,  # GQA group size = n_qh / n_kh.
    scale_softmax_log2: cutlass.Float32,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.TILES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sO_raw = cutlass.Array(OUT_STORAGE_DTYPE, CFG.TILES_Q * oBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    # SF buffers sized in BYTES (uint8 storage).
    sQ_SF_raw = cutlass.Array(cutlass.Int8, CFG.TILES_Q * SF_SMEM_SIZE_Q, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_K, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_P, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_V, alignment=1024, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.TILES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=CFG.TILES_Q,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
    )

    # SF SmemTiles: no-swizzle, leading=16, stride=128. TMA-LDG warp issues one 5-D bulk.tensor per slab.
    K_SF_BYTES_PER_PEER = SF_SMEM_SIZE_K // CFG.CTA_MMA
    V_SF_BYTES_PER_PEER = SF_SMEM_SIZE_V // CFG.CTA_MMA
    sQ_SF = SmemTile(
        base=sQ_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_Q,
        stages=CFG.TILES_Q,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sK_SF = SmemTile(
        base=sK_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_K,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sP_SF = SmemTile(
        base=sP_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_P,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )
    sV_SF = SmemTile(
        base=sV_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_V,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
    )

    bars = make_classic_bars(CFG)

    # SM100 mxfp8-only cross-CTA barrier (not in the shared Bars NamedTuple):
    # softmax[sub] signals "S_acc[sub] LDTM into registers is done" so the leader
    # MMA may reload SF_Q/K into that tile's S_acc scratch (steady-state ping-pong).
    # Leader-waited (P8): both CTAs' softmax arrive_on_leader → init = CTA_MMA.
    mb_softmax_ldtm = cutlass.Array(cutlass.Int64, CFG.TILES_Q, alignment=16, space=cutlass.AddressSpace.smem)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    # READ_TILE_ARRIVERS = 15 × CGA_M × CGA_N (softmax(8) + corr(4) + MMA + TMALDG + TMASTG).
    READ_TILE_ARRIVERS_TOTAL = ((CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS) + CFG.CORRECTION_WARPS + 1 + 1 + 1) * CGA_SIZE

    if warp_idx == 0:
        if nvvm.elect_sync():
            # mb_q_full init=1: leader's arrive_expect_tx covers Q TMA + both peers' Q_SF TMA via cta_group::2.
            # range_constexpr → Python-int loop var (required for the
            # mb_bmm2_ready tuple-init lookup).  Bounds are small —
            # unroll is free.
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_q_full[qs].init()
                bars.mb_q_empty[qs].init()
                bars.mb_bmm1_done[qs].init()
                bars.mb_bmm2_done[qs].init()
                bars.mb_stat_full[qs].init()
                bars.mb_stat_empty[qs].init()
                bars.mb_stats_read[qs].init()
                bars.mb_o_full[qs].init()
                bars.mb_o_empty[qs].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + c].init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOTAL)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                # Leader-waited (P8): each softmax warp elect-arrives after its own
                # S_acc LDTM+wait → SOFTMAX_WG_WARPS arrives per CTA × CTA_MMA CTAs.
                nvvm.mbarrier_init(mb_softmax_ldtm.subview(qs), CFG.SOFTMAX_WG_WARPS * CFG.CTA_MMA)
            bars.mb_tmem_dealloc.init()
            bars.mb_empty_mainloop.init()

    # P_SF SMEM fill 0x7F (constant 1.0); MUST happen BEFORE cga_arrive so both peers see filled SMEM.
    SF_TOTAL_BYTES_P = SF_SMEM_SIZE_P * CFG.STAGES_KV
    _SF_P_ITERS = (SF_TOTAL_BYTES_P + CFG.THREADS_PER_CTA - 1) // CFG.THREADS_PER_CTA
    for _i in cutlass.range_constexpr(_SF_P_ITERS):
        _off = tidx + cutlass.Int32(_i * CFG.THREADS_PER_CTA)
        if _off < cutlass.Int32(SF_TOTAL_BYTES_P):
            sP_SF_raw.subview(_off).store(cutlass.Int8(0x7F))

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    # 0b11 delivers each peer's K_SF/V_SF half to BOTH peers' SMEM.
    sf_mcast_mask = cutlass.Int16(3) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            sub_tile_id=0,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            bars=bars,
            sched=sched,
            mb_softmax_ldtm=mb_softmax_ldtm,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx >= CFG.SOFTMAX_WG1_BASE and warp_idx < CFG.SOFTMAX_WG1_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            sub_tile_id=1,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            bars=bars,
            sched=sched,
            mb_softmax_ldtm=mb_softmax_ldtm,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            sO=sO,
            tmem_ptr_i32=tmem_ptr_i32,
            tidx=tidx,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            amax_o_tensor=amax_o_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(CFG.CTA_MMA == 2):
            if is_leader:
                _mma_warp_group(
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    sQ=sQ,
                    sK=sK,
                    sV=sV,
                    sQ_SF=sQ_SF,
                    sK_SF=sK_SF,
                    sP_SF=sP_SF,
                    sV_SF=sV_SF,
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    mb_softmax_ldtm=mb_softmax_ldtm,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
                    qh_per_kh=qh_per_kh,
                )
            else:
                _mma_warp_quiet(
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    cta_in_pair=cta_in_pair,
                    cta_id_x=cta_id_x,
                )
        else:
            _mma_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sQ_SF=sQ_SF,
                sK_SF=sK_SF,
                sP_SF=sP_SF,
                sV_SF=sV_SF,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                mb_softmax_ldtm=mb_softmax_ldtm,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
                qh_per_kh=qh_per_kh,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_q_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_sf_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_sf_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            tma_q_sf_desc=tma_q_sf_desc,
            tma_k_sf_desc=tma_k_sf_desc,
            tma_v_sf_desc=tma_v_sf_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sQ_SF=sQ_SF,
            sK_SF=sK_SF,
            sV_SF=sV_SF,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            sf_mcast_mask=sf_mcast_mask,
        )

    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


# === Warp-group functions ===


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    tma_q_sf_desc,
    tma_k_sf_desc,
    tma_v_sf_desc,
    sQ,
    sK,
    sV,
    sQ_SF,
    sK_SF,
    sV_SF,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    o_desc_words,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    sf_mcast_mask,
):
    """TMA-LDG warp — Q/K/V TMA loads + per-iter SF TMA loads.

    Under cga2: Q_SF — each peer pulls its own M-half (one peer in mcast),
    bytes route to leader via cta_group::2.  K_SF/V_SF — both peers
    cooperatively load HALF the slab each at peer-offset with sf_mcast_mask=3.
    """
    q_empty_phase = cutlass.Int32(1)
    kv_state = PipelineState.start(phase=1)

    # SF TMA handles 5-D (innermost = 128-B row); tma_load_tile auto-selects on coord_0.
    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's RUNTIME descriptors (o_desc_words
        # slots n_batch+1 / n_batch+2), whose seq extent is clamped to the
        # packed KV total cu_k[B]. The last sequence's tile steps past that
        # total into the buffer's capacity tail; through the clamped
        # descriptors those rows are TMA-OOB and land as EXACT ZEROS — a NaN
        # tail (test_mhas_v2 poisons it) would otherwise wipe the tile via
        # BMM2's P·V (0 · NaN == NaN) and, on cc10.3, the pre-mask fused-LDTM
        # row-max. The SF descriptors stay grid-constant: THD SF buffers are
        # exactly the packed per-sequence-TILE-padded layout (caller
        # contract), so SF loads never leave caller-quantized bytes. Same
        # closure shape as the dense GmemTileTma — load sites are branch-free.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)
    tma_q_sf = GmemTileTma(tma_q_sf_desc)
    tma_k_sf = GmemTileTma(tma_k_sf_desc)
    tma_v_sf = GmemTileTma(tma_v_sf_desc)

    # cga2 split-half offsets fold to 0 under cga1.
    SF_TMA_ROW_BYTES = 128
    K_SF_BYTES_PER_PEER = SF_SMEM_SIZE_K // CFG.CTA_MMA
    V_SF_BYTES_PER_PEER = SF_SMEM_SIZE_V // CFG.CTA_MMA
    K_SF_ROWS_PER_PEER = K_SF_BYTES_PER_PEER // SF_TMA_ROW_BYTES
    V_SF_ROWS_PER_PEER = V_SF_BYTES_PER_PEER // SF_TMA_ROW_BYTES
    k_sf_peer_off = cta_in_pair * cutlass.Int32(K_SF_BYTES_PER_PEER)
    v_sf_peer_off = cta_in_pair * cutlass.Int32(V_SF_BYTES_PER_PEER)
    k_sf_peer_row = cta_in_pair * cutlass.Int32(K_SF_ROWS_PER_PEER)
    v_sf_peer_row = cta_in_pair * cutlass.Int32(V_SF_ROWS_PER_PEER)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    # GQA: K/V indexed by kv-head.
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    # Shared-helper seq offsets / SF-tile bases — fold to (0, 0, batch_idx) and
    # (0, 0) at THD_VARLEN=0 (dense-identity; legacy THD leg removed).
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
    cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    # SF expect_tx scales by CTA_MMA under cga2 (2× SMEM size on leader); cga1 → SF_SMEM_SIZE.
    Q_SF_EXPECT_BYTES = SF_SMEM_SIZE_Q * CFG.CTA_MMA
    K_SF_EXPECT_BYTES = SF_SMEM_SIZE_K * CFG.CTA_MMA
    V_SF_EXPECT_BYTES = SF_SMEM_SIZE_V * CFG.CTA_MMA

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Empty-kv tile: MMA's empty-mainloop branch handles the matching mbar phases.
        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            pass
        else:
            q_sf_tile_base = q_row_base // cutlass.Int32(CFG.TILE_M)
            kv_row_base = kv_left * CFG.TILE_N

            # Prologue: Q[0] / K[first] / Q[1] / V[first] interleaved.
            bars.mb_q_empty[0].wait(q_empty_phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                if is_leader:
                    if nvvm.elect_sync():
                        bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES)
            else:
                if nvvm.elect_sync():
                    bars.mb_q_full[0].arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES)
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(0 * CFG.TILE_M) + q_seq_off, tma_batch),
                bars.mb_q_full[0].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sQ_SF[0],
                tma_q_sf(cutlass.Int32(0), cu_sf_q_base + q_sf_tile_base + cutlass.Int32(0), head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_q_full[0].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                if is_leader:
                    if nvvm.elect_sync():
                        bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES)
            else:
                if nvvm.elect_sync():
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES)
            tma_load_tile(
                sK[kv_state.idx],
                tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                bars.mb_k_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sK_SF[kv_state.idx].shifted(k_sf_peer_off),
                tma_k_sf(k_sf_peer_row, cu_sf_k_base + kv_left, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_k_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=sf_mcast_mask,
            )

            bars.mb_q_empty[1].wait(q_empty_phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                if is_leader:
                    if nvvm.elect_sync():
                        bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES)
            else:
                if nvvm.elect_sync():
                    bars.mb_q_full[1].arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES)
            tma_load_tile(
                sQ[1],
                tma_q(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(1 * CFG.TILE_M) + q_seq_off, tma_batch),
                bars.mb_q_full[1].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sQ_SF[1],
                tma_q_sf(cutlass.Int32(0), cu_sf_q_base + q_sf_tile_base + cutlass.Int32(1), head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_q_full[1].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_phase = q_empty_phase ^ 1

            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                if is_leader:
                    if nvvm.elect_sync():
                        bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES)
            else:
                if nvvm.elect_sync():
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES)
            tma_load_tile(
                sV[kv_state.idx],
                tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                bars.mb_v_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sV_SF[kv_state.idx].shifted(v_sf_peer_off),
                tma_v_sf(v_sf_peer_row, cu_sf_k_base + kv_left, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_v_full[kv_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=sf_mcast_mask,
            )
            kv_state = advance(kv_state, CFG.STAGES_KV)

            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                kv_row_base_iter = kv_loop * CFG.TILE_N

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    if is_leader:
                        if nvvm.elect_sync():
                            bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES)
                else:
                    if nvvm.elect_sync():
                        bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES)
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base_iter + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                tma_load_tile(
                    sK_SF[kv_state.idx].shifted(k_sf_peer_off),
                    tma_k_sf(k_sf_peer_row, cu_sf_k_base + kv_loop, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=sf_mcast_mask,
                )

                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    if is_leader:
                        if nvvm.elect_sync():
                            bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES)
                else:
                    if nvvm.elect_sync():
                        bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES)
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base_iter + kv_seq_off, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                tma_load_tile(
                    sV_SF[kv_state.idx].shifted(v_sf_peer_off),
                    tma_v_sf(v_sf_peer_row, cu_sf_k_base + kv_loop, kv_head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=sf_mcast_mask,
                )

                kv_state = advance(kv_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # cga2 drain — TMA warp cga2 drain at kernel exit (cga2-mma.md).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_q_empty[_qs].wait(q_empty_phase)
        q_empty_phase = q_empty_phase ^ cutlass.Int32(1)
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def _tmastg_warp_group(
    tma_o_desc,
    sO,
    bars,
    sched,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    seq_kv_lens_tensor,
    o_desc_words,
    seqlen_kv,
    qh_per_kh,
):
    """TMA-STG warp — O SMEM→GMEM store."""
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_base = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        # KV split: partials are stacked split-major on the workspace BATCH axis
        # (extent B*SPLIT_KV), so the store needs no new descriptor — only a
        # shifted batch coord.  Folds to batch_idx at SPLIT_KV == 1.
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            bars.mb_o_full[qs].wait(o_full_phase)

            # O TMA params follow O's swizzle, NOT V's — required when V_SWZ_B != O_SWZ_B.
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: store each Q slab through this batch's pre-built descriptor
                # (base at the sequence's packed row, seq extent = S_q_b → a box
                # past S_q_b is OOB-clipped).  q_row coord is sequence-local; the
                # batch coord collapses to 0.  Both slabs share one descriptor.
                # (split_kv is dense-only — the config backstop rejects THD —
                # so o_batch never applies here.)
                # DEAD unit (batch == n_batch, over-launched envelope grid —
                # issue #552): no O rows exist and descriptor slot n_batch is
                # never built, so skip the store; the barrier protocol below
                # still runs.
                if batch_idx < n_batch:
                    o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                    o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(qs * CFG.TILE_M), cutlass.Int32(0))
                    tma_store_tile(sO[qs], o_slice)
            else:
                tma_store_tile(
                    sO[qs],
                    tma_o(cutlass.Int32(0), head_idx, q_row_base + cutlass.Int32(qs * CFG.TILE_M), o_batch),
                )

            tma_store_commit()
            tma_store_wait(0)

            bars.mb_o_empty[qs].arrive()

        o_full_phase = o_full_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _mma_warp_quiet(
    tmem_ptr_i32,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    cta_id_x,
):
    """Quiet MMA warp — non-leader CTA's MMA slot under cga2.

    Persistent loop keeps peer's mb_read_tile_id contribution + scheduler
    advance in lockstep with leader (MXFP8 quiet warp counts in
    READ_TILE_ARRIVERS=15×CGA, unlike FP8).
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
    sQ,
    sK,
    sV,
    sQ_SF,
    sK_SF,
    sP_SF,
    sV_SF,
    tmem_ptr_i32,
    bars,
    sched,
    mb_softmax_ldtm,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
):
    """Leader MMA warp — block-scale MMA stream for BMM1 + BMM2.

    Per-iter copy_sf into SF TMEM tiles ahead of each mma; P_SF tile is
    populated once at warp start (constant 1.0).  The inner loop covers
    SF_NUM_BLOCKS_K K-blocks.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk_bs = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=_MXFP8_K_DIM,
    )
    idesc_pv_bs = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=_MXFP8_K_DIM,
    )
    # sf_blocks_per_step = TILE_K_HW/32 (SF cols per MMA k-step); NOT SF_NUM_BLOCKS_K (total across K-tile).
    _MXFP8_SF_BLOCKS_PER_STEP = _MXFP8_TILE_K_HW // 32
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=_MXFP8_TILE_K_HW,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk_bs,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=_MXFP8_SF_BLOCKS_PER_STEP,
        scale_vec_size=SCALE_VEC_SIZE,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=_MXFP8_TILE_K_HW,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv_bs,
        kind=MMA_KIND,
        is_block_scale=True,
        sf_blocks_per_step=_MXFP8_SF_BLOCKS_PER_STEP,
        scale_vec_size=SCALE_VEC_SIZE,
    )

    desc_Q0 = sQ[0].desc()
    desc_Q1 = sQ[1].desc()

    desc_Q_SF_0 = sQ_SF[0].desc()
    desc_Q_SF_1 = sQ_SF[1].desc()
    desc_P_SF_0 = sP_SF[0].desc()

    # SF TMEM scratch (Blackwell 512-col cap → SF reloaded every MMA, cuDNN
    # ping-pong): steady BMM1 SF_Q/K in the OTHER sub-tile's S_acc lower region;
    # prologue BMM1 SF_Q/K in the (dead) O region; BMM2 SF_P/V per sub-tile.
    tmem_SF_Q0 = tmem_raw.subview(LAYOUT.SF_Q0_OFF)  # sub0 steady (S_acc_1)
    tmem_SF_K0 = tmem_raw.subview(LAYOUT.SF_K0_OFF)
    tmem_SF_Q1 = tmem_raw.subview(LAYOUT.SF_Q1_OFF)  # sub1 steady (S_acc_0)
    tmem_SF_K1 = tmem_raw.subview(LAYOUT.SF_K1_OFF)
    tmem_SF_Q0_pro = tmem_raw.subview(LAYOUT.SF_Q0_PRO_OFF)  # prologue (O_0, dead)
    tmem_SF_K0_pro = tmem_raw.subview(LAYOUT.SF_K0_PRO_OFF)
    tmem_SF_Q1_pro = tmem_raw.subview(LAYOUT.SF_Q1_PRO_OFF)
    tmem_SF_K1_pro = tmem_raw.subview(LAYOUT.SF_K1_PRO_OFF)
    tmem_SF_P0 = tmem_raw.subview(LAYOUT.SF_P0_OFF)
    tmem_SF_V0 = tmem_raw.subview(LAYOUT.SF_V0_OFF)
    tmem_SF_P1 = tmem_raw.subview(LAYOUT.SF_P1_OFF)
    tmem_SF_V1 = tmem_raw.subview(LAYOUT.SF_V1_OFF)

    # Trace-time UTCCP macros (emit IR at the call site; caller elect-gates).
    def _utccp_bmm1_sf(tmem_sf_q, tmem_sf_k, smem_desc_q, smem_desc_k):
        # SF_NUM_BLOCKS_K: 1 (TILE_K=128).
        for _sf_k in cutlass.range_constexpr(SF_NUM_BLOCKS_K):
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_q.subview(_sf_k * SF_REGISTERS_PER_BLOCK),
                smem_desc_q + _sf_k * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )
            nvvm.tcgen05_cp(
                nvvm.Tcgen05CpShape.SHAPE_32X128B,
                tmem_sf_k.subview(_sf_k * SF_REGISTERS_PER_BLOCK),
                smem_desc_k + _sf_k * (SF_BYTES_PER_BLOCK // 16),
                group=CTA_GROUP_KIND,
                multicast=nvvm.Tcgen05CpMulticast.WARPX4,
            )

    def _utccp_bmm2_sf(tmem_sf_p, tmem_sf_v, smem_desc_p, smem_desc_v):
        # SF_P is the constant P_SF; SF_V is per-kv-tile.  Both single-block (llama).
        # desc_P_SF_0 passed explicitly (DSL-value closure capture is unstaged).
        nvvm.tcgen05_cp(nvvm.Tcgen05CpShape.SHAPE_32X128B, tmem_sf_p, smem_desc_p, group=CTA_GROUP_KIND, multicast=nvvm.Tcgen05CpMulticast.WARPX4)
        nvvm.tcgen05_cp(nvvm.Tcgen05CpShape.SHAPE_32X128B, tmem_sf_v, smem_desc_v, group=CTA_GROUP_KIND, multicast=nvvm.Tcgen05CpMulticast.WARPX4)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx, split_idx = _decode_initial_split(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=0)
    bmm2_ready_phase = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)
    # Steady-state BMM1[sub_i] waits mb_softmax_ldtm[1-sub_i] before reloading
    # SF_Q/K into that tile's S_acc scratch (one iter behind softmax).  Per tile
    # softmax fires n_kv arrives on EACH sub-tile's ldtm mbar: [0] is consumed
    # n_kv times (prologue + steady loop), [1] only n_kv-1 times (steady loop
    # only) — the epilogue P14 drain (below) consumes the last [1] so neither
    # phase accumulates across persistent-loop wraps (else: multi-wave deadlock).
    ldtm_phase0 = cutlass.Int32(0)  # consumes softmax[0] arrives (for BMM1[sub1])
    ldtm_phase1 = cutlass.Int32(0)  # consumes softmax[1] arrives (for BMM1[sub0])
    # Stats-consumed gate (one flip per tile per sub-tile): the prologue BMM1
    # overwrites the S_acc HEAD where the PREVIOUS tile's final
    # (total_max, total_sum) stats live until the correction epilogue has read
    # them.  q_full/k_full alone do NOT order that read before the BMM1, and
    # mb_softmax_ldtm only orders BMM1 after SOFTMAX's S reads — never after
    # CORRECTION's epilogue stats tcgen05_ld.  Bootstrap phase 1: the first
    # tile has no prior stats to protect, so the wait passes immediately.
    stats_read_phase = cutlass.Int32(1)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty mainloop — keep softmax/correction phase trackers in lockstep with non-empty path.
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            # Keep the stats-read gate in lockstep — correction's epilogue
            # (and its mb_stats_read arrive) runs for empty tiles too.
            bars.mb_stats_read[0].wait(stats_read_phase)
            bars.mb_stats_read[1].wait(stats_read_phase)
            if nvvm.elect_sync():
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
        else:
            # Prologue: BMM1[sub0] + BMM1[sub1] for kv=kv_left.  SF_Q/K land in
            # the O_0 region (O not yet accumulated → dead scratch; the first BMM2
            # overwrites it with accumulate=False).  No softmax-LDTM wait needed.
            # mb_stats_read[qs] gates each prologue BMM1 on the correction
            # epilogue having READ the previous tile's final stats from the
            # S_acc head this BMM1 is about to overwrite (q_full/k_full and
            # mb_softmax_ldtm don't order that).
            bars.mb_q_full[0].wait(q_full_phase)
            bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
            bars.mb_stats_read[0].wait(stats_read_phase)
            desc_K = sK[kv_state.idx].desc()
            desc_K_SF = sK_SF[kv_state.idx].desc()
            if nvvm.elect_sync():
                _utccp_bmm1_sf(tmem_SF_Q0_pro, tmem_SF_K0_pro, desc_Q_SF_0, desc_K_SF)
            mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)), tmem_sf_a=tmem_SF_Q0_pro, tmem_sf_b=tmem_SF_K0_pro)
            if nvvm.elect_sync():
                bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

            bars.mb_q_full[1].wait(q_full_phase)
            # Consume softmax[0]'s prologue-iter LDTM arrive so the steady-state
            # BMM1[sub1] ldtm[0] waits align to the SAME-iter softmax[0] (cuDNN:
            # "tile 1 waits for tile 0's softmax; tile 0 starts immediately").
            # Prologue SF is O-region-dead, so this wait is alignment-only.
            wait(mb_softmax_ldtm.subview(0), ldtm_phase0)
            ldtm_phase0 = ldtm_phase0 ^ 1
            bars.mb_stats_read[1].wait(stats_read_phase)
            if nvvm.elect_sync():
                _utccp_bmm1_sf(tmem_SF_Q1_pro, tmem_SF_K1_pro, desc_Q_SF_1, desc_K_SF)
            mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)), tmem_sf_a=tmem_SF_Q1_pro, tmem_sf_b=tmem_SF_K1_pro)
            if nvvm.elect_sync():
                bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
                bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

            q_full_phase = q_full_phase ^ 1

            for kv_loop in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                old_state = kv_state
                kv_state = advance(kv_state, CFG.STAGES_KV)

                bars.mb_v_full[old_state.idx].wait(old_state.phase)
                desc_V = sV[old_state.idx].desc()
                desc_V_SF = sV_SF[old_state.idx].desc()
                is_not_first_bmm2 = cutlass.Boolean(kv_loop != (kv_left + cutlass.Int32(1)))

                # BMM2[sub0] → O_0.  SF_P0/V0 reload into S_acc_0 scratch (32/36);
                # the bmm2_ready wait already proves softmax[0] read all of S_acc_0.
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
                if nvvm.elect_sync():
                    _utccp_bmm2_sf(tmem_SF_P0, tmem_SF_V0, desc_P_SF_0, desc_V_SF)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P0_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O0_OFF)),
                        local_k,
                        accum_b2,
                        tmem_sf_a=tmem_SF_P0,
                        tmem_sf_b=tmem_SF_V0,
                    )
                    accum_b2 = cutlass.Boolean(True)
                bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P0_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O0_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                        tmem_sf_a=tmem_SF_P0,
                        tmem_sf_b=tmem_SF_V0,
                    )
                if nvvm.elect_sync():
                    bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

                # BMM1[sub0] next-kv → S_acc_0.  SF_Q0/K0 ping-pong into S_acc_1
                # scratch (136/140); wait tile1's softmax LDTM (ONE-behind) so its
                # read of S_acc_1 completed before the leader-broadcast UTCCP
                # (cta_group::2 self-fills both CTAs) overwrites it.
                bars.mb_k_full[kv_state.idx].wait(kv_state.phase)
                desc_K = sK[kv_state.idx].desc()
                desc_K_SF = sK_SF[kv_state.idx].desc()
                wait(mb_softmax_ldtm.subview(1), ldtm_phase1)
                ldtm_phase1 = ldtm_phase1 ^ 1
                if nvvm.elect_sync():
                    _utccp_bmm1_sf(tmem_SF_Q0, tmem_SF_K0, desc_Q_SF_0, desc_K_SF)
                mma_ss(bmm1_desc, desc_Q0, desc_K, (tmem_raw.subview(LAYOUT.S0_OFF)), tmem_sf_a=tmem_SF_Q0, tmem_sf_b=tmem_SF_K0)
                if nvvm.elect_sync():
                    bars.mb_bmm1_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

                # BMM2[sub1] → O_1.  SF_P1/V1 reload into S_acc_1 scratch (160/164).
                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
                if nvvm.elect_sync():
                    _utccp_bmm2_sf(tmem_SF_P1, tmem_SF_V1, desc_P_SF_0, desc_V_SF)
                accum_b2 = is_not_first_bmm2
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P1_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O1_OFF)),
                        local_k,
                        accum_b2,
                        tmem_sf_a=tmem_SF_P1,
                        tmem_sf_b=tmem_SF_V1,
                    )
                    accum_b2 = cutlass.Boolean(True)
                bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
                for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                    mma_ts_step(
                        bmm2_desc,
                        (tmem_raw.subview(LAYOUT.P1_OFF)),
                        desc_V,
                        (tmem_raw.subview(LAYOUT.O1_OFF)),
                        NUM_KPHASES_PV_PER_CHUNK + local_k,
                        cutlass.Boolean(True),
                        tmem_sf_a=tmem_SF_P1,
                        tmem_sf_b=tmem_SF_V1,
                    )
                if nvvm.elect_sync():
                    bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
                    bars.mb_v_empty[old_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

                # BMM1[sub1] next-kv → S_acc_1.  SF_Q1/K1 ping-pong into S_acc_0
                # scratch (8/12); wait tile0's softmax LDTM (SAME-iter — BMM1[sub0]
                # above just wrote this iter's S_acc_0, which softmax[0] must read).
                wait(mb_softmax_ldtm.subview(0), ldtm_phase0)
                ldtm_phase0 = ldtm_phase0 ^ 1
                if nvvm.elect_sync():
                    _utccp_bmm1_sf(tmem_SF_Q1, tmem_SF_K1, desc_Q_SF_1, desc_K_SF)
                mma_ss(bmm1_desc, desc_Q1, desc_K, (tmem_raw.subview(LAYOUT.S1_OFF)), tmem_sf_a=tmem_SF_Q1, tmem_sf_b=tmem_SF_K1)
                if nvvm.elect_sync():
                    bars.mb_bmm1_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
                    bars.mb_k_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

                bmm2_ready_phase = bmm2_ready_phase ^ 1

            # Epilogue: BMM2 for last kv (always runs — n_kv >= 1).
            if nvvm.elect_sync():
                for qs in cutlass.range_constexpr(CFG.TILES_Q):
                    bars.mb_q_empty[qs].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

            bars.mb_v_full[kv_state.idx].wait(kv_state.phase)
            desc_V = sV[kv_state.idx].desc()
            desc_V_SF = sV_SF[kv_state.idx].desc()
            is_not_first_bmm2_epi = cutlass.Boolean((kv_right - kv_left) != cutlass.Int32(1))

            # Epilogue BMM2[sub0] → O_0.  SF_P0/V0 reload into S_acc_0 scratch.
            bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
            if nvvm.elect_sync():
                _utccp_bmm2_sf(tmem_SF_P0, tmem_SF_V0, desc_P_SF_0, desc_V_SF)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P0_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O0_OFF)),
                    local_k,
                    accum_b2,
                    tmem_sf_a=tmem_SF_P0,
                    tmem_sf_b=tmem_SF_V0,
                )
                accum_b2 = cutlass.Boolean(True)
            bars.mb_bmm2_ready[0 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P0_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O0_OFF)),
                    NUM_KPHASES_PV_PER_CHUNK + local_k,
                    cutlass.Boolean(True),
                    tmem_sf_a=tmem_SF_P0,
                    tmem_sf_b=tmem_SF_V0,
                )
            if nvvm.elect_sync():
                bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

            # Epilogue BMM2[sub1] → O_1.  SF_P1/V1 reload into S_acc_1 scratch.
            bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 0].wait(bmm2_ready_phase)
            if nvvm.elect_sync():
                _utccp_bmm2_sf(tmem_SF_P1, tmem_SF_V1, desc_P_SF_0, desc_V_SF)
            accum_b2 = is_not_first_bmm2_epi
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P1_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O1_OFF)),
                    local_k,
                    accum_b2,
                    tmem_sf_a=tmem_SF_P1,
                    tmem_sf_b=tmem_SF_V1,
                )
                accum_b2 = cutlass.Boolean(True)
            bars.mb_bmm2_ready[1 * CFG.N_BMM2_CHUNKS + 1].wait(bmm2_ready_phase)
            for local_k in cutlass.range_constexpr(NUM_KPHASES_PV_PER_CHUNK):
                mma_ts_step(
                    bmm2_desc,
                    (tmem_raw.subview(LAYOUT.P1_OFF)),
                    desc_V,
                    (tmem_raw.subview(LAYOUT.O1_OFF)),
                    NUM_KPHASES_PV_PER_CHUNK + local_k,
                    cutlass.Boolean(True),
                    tmem_sf_a=tmem_SF_P1,
                    tmem_sf_b=tmem_SF_V1,
                )
            if nvvm.elect_sync():
                bars.mb_bmm2_done[1].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)
                bars.mb_v_empty[kv_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

            # P14 drain — rebalance mb_softmax_ldtm[1].  softmax fires one
            # arrive_on_leader(mb_softmax_ldtm[1]) per kv-iter (= n_kv/tile), but the
            # consumer flips ldtm_phase1 only n_kv-1 times (steady loop; the prologue
            # consumes softmax[0] not [1], and the epilogue has no BMM1).  ldtm_phase1
            # lives OUTSIDE the persistent tile loop, so that +1/tile surplus
            # ACCUMULATES across waves and laps the 1-bit phase -> circular NANOSLEEP
            # deadlock at high tile-count (multi-wave; small single-wave problems never
            # trip it).  This catch-up wait returns immediately — softmax's last [1]
            # arrive precedes the mb_bmm2_ready[1] arrives already awaited above — and
            # balances [1] to n_kv.  Non-empty else only (empty tiles fire/consume 0).
            wait(mb_softmax_ldtm.subview(1), ldtm_phase1)
            ldtm_phase1 = ldtm_phase1 ^ 1

            bmm2_ready_phase = bmm2_ready_phase ^ 1
            kv_state = advance(kv_state, CFG.STAGES_KV)

        # One correction-epilogue arrive per tile per sub-tile — flip once per tile.
        stats_read_phase = stats_read_phase ^ 1

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
            nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
            nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
            nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
            q_super_idx, _hd, batch_idx, split_idx = _decode_payload_split(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
                qh_per_kh,
                seqlen_kv,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
            else:
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
                bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)
                kv_left = bounds_next.left
                kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_kv_body(
    apply_mask: bool,
    sub_tile_id: int,
    kv_loop,
    tmem_ptr_i32,
    bars,
    mb_softmax_ldtm,
    q_abs,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    total_max,
    total_sum,
    bmm1_phase,
    stat_empty_phase,
    leader_cta_id,
):
    """Per-iter softmax body — returns (total_max, total_sum, bmm1_phase, stat_empty_phase).

    Compile-time apply_mask picks fast HW max (tcgen05.ld.red.f32.max)
    vs slow path (load → apply_mask_chunk → software row_max).  The HW
    max can't observe NEG_INF written after load, so the kv-loop is
    split into LEFT/unmasked/RIGHT segments to keep the fast path on
    every interior iter.
    """
    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE
    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 4
    NEG_INF = cutlass.Float32(-3.4028235e38)
    RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

    bars.mb_bmm1_done[sub_tile_id].wait(bmm1_phase)
    bmm1_phase = bmm1_phase ^ 1

    # Hoist TMEM address math to a single base load per kv-iter.
    tmem_base = tmem_ptr_i32.load()
    s_addr_a = tmem_base + cutlass.Int32(tmem_S_off + 0)
    s_addr_b = tmem_base + cutlass.Int32(tmem_S_off + CHUNK)
    p_addr_a = tmem_base + cutlass.Int32(tmem_P_off + 0)
    p_addr_b = tmem_base + cutlass.Int32(tmem_P_off + P_COLS_PER_CHUNK)
    stats_addr = tmem_base + cutlass.Int32(stats_off)

    # Pre-declare reg_S_*/max_* — without it MLIR cf.if NameErrors the tracer post-branch.
    reg_S_a = cutlass.Vector.from_elements(
        tuple(cutlass.Float32(0.0) for _ in range(CHUNK)),
        cutlass.Float32,
    )
    reg_S_b = reg_S_a
    max_a = cutlass.Float32(0.0)
    max_b = cutlass.Float32(0.0)

    if apply_mask:
        # Masked slow path — load, mask, software row-max.
        reg_S_a = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_a, cutlass.Float32),
            num=CHUNK,
        )
        reg_S_b = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_b, cutlass.Float32),
            num=CHUNK,
        )
        # No tcgen05_wait(LOAD) — reg consumers chain through deps; no downstream mbar arrive depends on it.

        kv_col_base_a = kv_loop * cutlass.Int32(CFG.TILE_N)
        kv_col_base_b = kv_col_base_a + cutlass.Int32(CHUNK)
        # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
        # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
        reg_S_a = apply_mask_chunk(
            reg_S_a,
            q_abs,
            kv_col_base_a,
            eff_seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            N=CHUNK,
            bottom_right=CFG.BOTTOM_RIGHT,
            causal_diag=causal_diag,
            window_right=CFG.WINDOW_RIGHT,
        )
        reg_S_b = apply_mask_chunk(
            reg_S_b,
            q_abs,
            kv_col_base_b,
            eff_seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            N=CHUNK,
            bottom_right=CFG.BOTTOM_RIGHT,
            causal_diag=causal_diag,
            window_right=CFG.WINDOW_RIGHT,
        )

        max_a = row_max_reduction_64(reg_S_a)
        max_b = row_max_reduction_64(reg_S_b)
    elif cutlass.const_expr(FUSED_LDTM_STAT != 0):
        # cc10.3+: fused HW row-max — one tcgen05.ld.red.f32.max does the S_acc
        # load AND the row-max in one op, returning 64 data regs + the max at
        # index CHUNK. Unmasked tiles only (the fused .max reduces before the
        # mask would be applied; masked tiles took the software path above).
        res_a = tmem_load_max_reduction_x64(s_addr_a)
        res_b = tmem_load_max_reduction_x64(s_addr_b)
        reg_S_a = cutlass.Vector.from_elements(
            tuple(res_a[:CHUNK]),
            cutlass.Int32,
        ).bitcast(cutlass.Float32)
        reg_S_b = cutlass.Vector.from_elements(
            tuple(res_b[:CHUNK]),
            cutlass.Int32,
        ).bitcast(cutlass.Float32)
        max_a = cutlass.Vector.from_elements(
            (res_a[CHUNK],),
            cutlass.Int32,
        ).bitcast(
            cutlass.Float32
        )[0]
        max_b = cutlass.Vector.from_elements(
            (res_b[CHUNK],),
            cutlass.Int32,
        ).bitcast(
            cutlass.Float32
        )[0]
    else:
        # cc10.0: no fused LDTM.STAT (tmem_load_max_reduction_x64) — manual
        # tcgen05_ld + software row_max_reduction_64 (masked path sans mask).
        reg_S_a = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_a, cutlass.Float32),
            num=CHUNK,
        )
        reg_S_b = nvvm.tcgen05_ld(
            "32x32b",
            nvvm.make_tmem_ptr(s_addr_b, cutlass.Float32),
            num=CHUNK,
        )
        max_a = row_max_reduction_64(reg_S_a)
        max_b = row_max_reduction_64(reg_S_b)

    # SM100 mxfp8: S_acc[sub] is fully LDTM'd into registers now.  Each softmax
    # warp elect-arrives on the leader (cross-CTA, P8) so the leader MMA may
    # reload SF_Q/K into this tile's S_acc scratch (steady-state ping-pong);
    # tcgen05_wait(LOAD) proves the TMEM read drained before MMA overwrites it.
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
    if nvvm.elect_sync():
        arrive_on_leader(mb_softmax_ldtm.subview(sub_tile_id), leader_cta_id, CFG.CTA_MMA)

    current_max = cute.math.max(max_a, max_b) * scale_log2

    # sync the two softmax warpgroups before the stat-store.
    if sub_tile_id == 1:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    old_total_max = total_max
    is_first = total_max == NEG_INF
    update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
    total_max = cutlass.Float32(
        arith.select(
            update_cond.ir_value(),
            current_max.ir_value(),
            total_max.ir_value(),
        )
    )
    exp_input = cutlass.Float32(
        arith.select(
            is_first.ir_value(),
            NEG_INF.ir_value(),
            (old_total_max - total_max).ir_value(),
        )
    )
    alpha = cute.math.exp2(exp_input, fastmath=True)
    new_total_max = total_max

    alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[sub_tile_id].arrive()

    # P-cast bias: exp2(x + P_CAST_LOG2_SCALE) = 2^4 * P — the sum picks up
    # the same factor, so normalization cancels it (see P_CAST_LOG2_SCALE).
    reg_S_a = reg_S_a * scale_log2 - (new_total_max - cutlass.Float32(P_CAST_LOG2_SCALE))
    reg_S_b = reg_S_b * scale_log2 - (new_total_max - cutlass.Float32(P_CAST_LOG2_SCALE))
    reg_P_a = cute.math.exp2(reg_S_a, fastmath=True)
    sum_a_pair = row_reduction_pair_64(reg_P_a)
    p_a_fp16 = reg_P_a.to(STORAGE_DTYPE)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_a, cutlass.Float32), p_a_fp16)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[sub_tile_id * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    reg_P_b = cute.math.exp2(reg_S_b, fastmath=True)
    p_b_fp16 = reg_P_b.to(STORAGE_DTYPE)
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_b, cutlass.Float32), p_b_fp16)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_bmm2_ready[sub_tile_id * CFG.N_BMM2_CHUNKS + 1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    if sub_tile_id == 0:
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=256)

    sum_b_pair = row_reduction_pair_64(reg_P_b)
    new_p_sum_pair = sum_a_pair + sum_b_pair
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum = total_sum * alpha_pair + new_p_sum_pair

    bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
    stat_empty_phase = stat_empty_phase ^ 1

    return total_max, total_sum, bmm1_phase, stat_empty_phase


@cute.jit
def _softmax_warp_group(
    sub_tile_id: int,
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    sQ,
    bars,
    sched,
    mb_softmax_ldtm,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
):
    """Softmax warp group — online softmax + α publish (LSE write owned by correction)."""
    # Wait on MMA's TMEM-publish named barrier BEFORE tmem_ptr_i32.load() — else stale base pointer.
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    tmem_S_off = LAYOUT.S0_OFF if sub_tile_id == 0 else LAYOUT.S1_OFF
    tmem_P_off = LAYOUT.P0_OFF if sub_tile_id == 0 else LAYOUT.P1_OFF

    NEG_INF = cutlass.Float32(-3.4028235e38)

    CHUNK = 64
    P_COLS_PER_CHUNK = CHUNK // 4
    stats_off = LAYOUT.STATS_OFF + sub_tile_id * LAYOUT.STATS_STRIDE

    bmm1_phase = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately
    epilogue_state = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately

    # total_sum kept as Vector[Float32,2] so per-iter update lowers to packed FMUL2 + FADD2.
    total_max = NEG_INF
    total_sum = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)

    softmax_wg_base_const = CFG.SOFTMAX_WG0_BASE if sub_tile_id == 0 else CFG.SOFTMAX_WG1_BASE
    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(softmax_wg_base_const * 32)

    while is_valid_tile > cutlass.Int32(0):
        if cutlass.const_expr(not (CFG.MASK_FLAGS & MASK_CAUSAL)):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Both softmax wgs wait on slot [0]; without this softmax races ahead while TMA-STG drains prior tile.
        bars.mb_o_empty[0].wait(epilogue_state)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        total_max = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + cutlass.Int32(sub_tile_id * CFG.TILE_M) + tid_in_wg
        # Bootstrap stat_empty wait lifts the wait out of per-iter critical path.
        bars.mb_stat_empty[sub_tile_id].wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ 1
        # 3-segment kv loop: LEFT-masked / unmasked (fast HW max) / RIGHT-masked.
        # MASK_NONE: bounds collapse so masked sub-loops fold out at trace time.
        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                total_max, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
                    False,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    mb_softmax_ldtm,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                total_max, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
                    True,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    mb_softmax_ldtm,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                total_max, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
                    False,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    mb_softmax_ldtm,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )
            if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
                read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                total_max, total_sum, bmm1_phase, stat_empty_phase = _softmax_kv_body(
                    True,
                    sub_tile_id,
                    kv_loop,
                    tmem_ptr_i32,
                    bars,
                    mb_softmax_ldtm,
                    q_abs,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    total_max,
                    total_sum,
                    bmm1_phase,
                    stat_empty_phase,
                    leader_cta_id,
                )

        # Per-tile balance: softmax 1 bootstrap + n_kv end-of-body waits = n_kv+1 = corr fires.
        total_sum_scalar = total_sum[0] + total_sum[1]

        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(stats_off)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[sub_tile_id].arrive()

        # make_warp_uniform on scheduler loads keeps payload in uniform regs across back-edge (no STL spill).
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
    sO,
    tmem_ptr_i32,
    tidx,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
):
    """Correction warp group — α-rescale O + epilogue cast/store + LSE.

    P14 catch-up flip on bmm2_done_phase at end of tile is REQUIRED for
    multi-wave at n_kv=1.
    """
    # Wait on MMA's TMEM-publish named barrier BEFORE tmem_ptr_i32.load() — else stale base pointer.
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    # O_CHUNK=16 keeps live range short enough to avoid spilling (O_CHUNK=32 spilled correction regs).
    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    # D_BLOCK_SIZE must use O_SWZ_B not V_SWZ_B — fp8 V drops to Swz64B while half O stays Swz128B.
    # Sized in BPE_O so it stays consistent with the BPE_O-derived O_SWZ_BYTES (BF16/FP16 O).
    TMA_O_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS
    TMA_O_GRANU_ELEMS = CFG.TILE_M * D_BLOCK_SIZE

    stat_full_phase = cutlass.Int32(0)
    bmm2_done_phase = cutlass.Int32(0)
    o_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1 so first wait passes immediately

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
        qh_per_kh,
        seqlen_kv,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)

    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # Iter-0 skip — MMA's iter-0 BMM2 uses init_d=False to overwrite O, no α-rescale needed.
        if bounds.right > bounds.left:
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                bars.mb_stat_full[qs].wait(stat_full_phase)
                bars.mb_stat_empty[qs].arrive()
            stat_full_phase = stat_full_phase ^ 1
        else:
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            tmem_base_iter = tmem_ptr_i32.load()
            for qs in cutlass.range_constexpr(CFG.TILES_Q):
                stats_off = LAYOUT.STATS_OFF + qs * LAYOUT.STATS_STRIDE
                tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

                bars.mb_stat_full[qs].wait(stat_full_phase)

                stats_addr = tmem_base_iter + cutlass.Int32(stats_off)
                stats_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                    num=2,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                alpha = stats_vec[0]

                # all_alpha_one ballot skips α-rescale once softmax stops bumping total_max.
                alpha_is_one = alpha == cutlass.Float32(1.0)
                all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                bars.mb_stat_empty[qs].arrive()

                bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

                # vec_scale_pair emits mul_packed_f32x2; without it the DSL lowers to scalar FMUL inside this runtime-if.
                if ~all_alpha_one:
                    for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                        o_addr = tmem_base_iter + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                        o_chunk = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                            num=O_CHUNK,
                        )
                        o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                bars.mb_bmm2_ready[qs * CFG.N_BMM2_CHUNKS + 0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_full_phase = stat_full_phase ^ 1
            bmm2_done_phase = bmm2_done_phase ^ 1

        tmem_base_epi = tmem_ptr_i32.load()
        for qs in cutlass.range_constexpr(CFG.TILES_Q):
            stats_off = LAYOUT.STATS_OFF + qs * LAYOUT.STATS_STRIDE
            tmem_O_off = LAYOUT.O0_OFF if qs == 0 else LAYOUT.O1_OFF

            bars.mb_bmm2_done[qs].wait(bmm2_done_phase)

            bars.mb_stat_full[qs].wait(stat_full_phase)

            stats_addr = tmem_base_epi + cutlass.Int32(stats_off)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec[0]
            total_sum = stats_vec[1]

            bars.mb_stat_empty[qs].arrive()
            # Release MMA's NEXT-tile prologue BMM1 into this S_acc slot: the
            # final stats ride the slot HEAD and are now safely in registers
            # (tcgen05_wait LOAD above).  Cross-CTA arrive on the leader under
            # cga2 — the collective BMM1 writes both peers' TMEM.
            bars.mb_stats_read[qs].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # Pre-declare inv_sum/lse_val — without it MLIR cf.if NameErrors the tracer post-branch.
            inv_sum = cutlass.Float32(0.0)
            lse_val = cutlass.Float32(0.0)
            LN2 = cutlass.Float32(0.6931471805599453)
            total_max_nat = total_max_scaled * LN2
            # Dead row (no valid KV column at all): O := 0 in BOTH branches
            # (with a sink the denominator is finite but the O numerator is an
            # empty sum).  total_sum >= 2^P_CAST_LOG2_SCALE for any alive row
            # so this never fires spuriously.
            row_dead = total_sum <= cutlass.Float32(0.0)
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = sinks_arr[head_idx]
                new_max = cute.math.max(total_max_nat, sink_logit)
                scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
                # total_sum is in 2^P_CAST_LOG2_SCALE units — lift the sink term
                # into the same units, then take the constant back out of the LSE.
                new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True) * cutlass.Float32(2.0**P_CAST_LOG2_SCALE)
                lse_val = new_max + cute.math.log(new_sum, fastmath=True) - cutlass.Float32(P_CAST_LOG2_SCALE) * LN2
                inv_sum = scale / new_sum
            else:
                # total_sum carries 2^P_CAST_LOG2_SCALE — subtract the constant.
                lse_val = total_max_nat + cute.math.log(total_sum, fastmath=True) - cutlass.Float32(P_CAST_LOG2_SCALE) * LN2
                # Safe inverse: avoid div by 0 (rows fully masked).
                inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))
                # Dead row without a sink: LSE := -inf on top of O := 0.
                neg_inf_lse = cutlass.Float32(float("-inf"))
                lse_val = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse_val.ir_value()))
                inv_sum = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), inv_sum.ir_value()))

            # OOB-row guard: under cga2 cluster Q rows can exceed seqlen_q (else write aliases next head's LSE slot).
            q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + cutlass.Int32(qs * CFG.TILE_M) + tid_in_wg
            # has_lse=False: the Stats store is compiled out; the amax_o
            # atomicMax below is independent of it, gated only on _row_valid.
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: q_row_global is sequence-local; the row is valid against
                # the per-sequence Q length from the device metadata (negative
                # for the dead-unit sentinel batch == n_batch — issue #552 —
                # so no dead-unit row ever writes LSE or feeds amax_o). The
                # packed ragged-Stats LSE is written in the caller's declared
                # layout — token-major rank-2 [T, QH] (index
                # [cu_q[b] + local, head]) or head-major rank-3
                # [1, QH, head_stride] (index [0, head, cu_q[b] + local]).
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                _row_valid = q_row_global < _s_q_b
                if cutlass.const_expr(lse_tensor is not None):
                    if _row_valid:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        if cutlass.const_expr(len(lse_tensor.shape) == 2):
                            lse_row = lse_arr[_cu_q_b + q_row_global, :]
                            lse_row[head_idx] = lse_val
                        else:
                            lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                            lse_row[_cu_q_b + q_row_global] = lse_val
            else:
                _row_valid = q_row_global < seqlen_q
                if cutlass.const_expr(lse_tensor is not None):
                    if _row_valid:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        # This chunk's LSE goes to its own split-major slot, matching where
                        # TMA-STG put the chunk's O.  The pair (O_s, lse_s) is everything
                        # the combine needs.  Folds to batch_idx at SPLIT_KV == 1.
                        lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                        lse_arr[lse_batch, head_idx, q_row_global] = lse_val

            sO_sub_base = sO[qs].base

            _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
            _amax_o_local = cutlass.Float32(0.0)

            for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                o_addr = tmem_base_epi + cutlass.Int32(tmem_O_off + chunk_idx * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                # Dead-row sanitize: an empty mainloop never writes O TMEM,
                # so o_chunk is garbage (possibly NaN) and `* inv_sum(=0)`
                # cannot zero it — select 0 explicitly (keeps amax_o clean).
                _zero_f = cutlass.Float32(0.0)
                o_scaled = cutlass.Vector.from_elements(
                    tuple(cutlass.Float32(arith.select(row_dead.ir_value(), _zero_f.ir_value(), o_scaled[i].ir_value())) for i in range(O_CHUNK)),
                    cutlass.Float32,
                )
                for _i in cutlass.range_constexpr(O_CHUNK):
                    _e = o_scaled[_i]
                    _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))
                o_out = o_scaled.to(OUT_STORAGE_DTYPE)

                col_offset_const = (chunk_idx * O_CHUNK) % D_BLOCK_SIZE
                block_idx_const = (chunk_idx * O_CHUNK) // D_BLOCK_SIZE
                block_offset_const = block_idx_const * TMA_O_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)

                smem_ptr = sO_sub_base.subview(smem_offset).data_ptr()
                # Gate FIRST SMEM store (not earlier TMEM-load loop) — keeps load/FFMA/cast overlapped with prior TMA-STG drain.
                if chunk_idx == 0:
                    bars.mb_o_empty[qs].wait(o_empty_phase)
                smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # One atomic per valid row (invalid/OOB rows must not poison the global amax).
            #
            # Under KV split this epilogue sees only its OWN partial, and the
            # recombined O is a convex combination of the partials -- so a max
            # over partials over-reports the output amax.  split_combine_sm100
            # computes it over the recombined O instead; this write has to stay
            # out of the way, since atomicMax only grows.
            if cutlass.const_expr(SPLIT_KV == 1):
                if _row_valid:
                    nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))

            # fence_proxy needed before TMA reads SMEM written by tcgen05_st.
            nvvm.fence_proxy("async.shared", space="cta")

            bars.mb_o_full[qs].arrive()

        stat_full_phase = stat_full_phase ^ 1
        o_empty_phase = o_empty_phase ^ 1
        # P14 catch-up flip — bmm2_done_phase ^= 1 AFTER epilogue wait (else n_kv=1 multi-wave deadlocks on tile 2).
        bmm2_done_phase = bmm2_done_phase ^ 1

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
            qh_per_kh,
            seqlen_kv,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair, None, None, split_idx)

    # mb_tmem_dealloc fan-out — all-lanes arrive + DSMEM-arrive on cross-pair peer under cga2.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


# === Host launcher ===


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    sf_q_tensor: cute.Tensor,
    sf_k_tensor: cute.Tensor,
    sf_v_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    # THD device metadata build (issue #552): the CALLER's Q/KV length
    # tensors — (B,) per-batch lengths or (B+1,) cu prefix sums, per side via
    # thd_lens_form (bit 0: Q is cu, bit 1: KV is cu) — consumed only by the
    # setup kernel, which writes the [kv|cu_q|cu_k] metadata buffer
    # (seq_kv_lens_tensor) device-side. None (folded out of the ABI) for
    # dense graphs.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """MXFP8 host launcher — builds TMA descriptors for Q/K/V/O + SF tensors.

    THD/varlen: q/k/v/o/lse + SF tensors are PACKED with batch dim 1 and
    DYNAMIC token / SF-tile extents; the SF descriptors use B=1 + the bound
    SF views' packed tile extents (per-sequence-TILE-padded layout); O uses a
    per-batch descriptor array built by the setup kernel."""
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are runtime values (dynamic extents); the
        # problem_size slots are 0 by contract.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O  # O box sized in BPE_O (BF16/FP16 O)
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

    # SF TMA: 5-D, box=(128, num_rows, 1, 1, 1); cga2 K_SF/V_SF narrows to (128, num_rows//CTA_MMA, ...).
    # 5-D layout assumes SF SMEM flat-contiguous (single block-K); d>128 (SF_NUM_BLOCKS_V=2) would need split TMA or 6-D.
    SF_TMA_ROW_BYTES = 128
    SF_NUM_ROWS_Q = SF_SMEM_SIZE_Q // SF_TMA_ROW_BYTES
    SF_NUM_ROWS_K = SF_SMEM_SIZE_K // SF_TMA_ROW_BYTES
    SF_NUM_ROWS_V = SF_SMEM_SIZE_V // SF_TMA_ROW_BYTES

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qk_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor,
        box_dims=qk_box_k,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # SF TMA strides in 16-B units (TMA convention); reinterpret 4-D runner SF
    # tensor as 5-D split-byte view.  Dense: per-batch tile counts (SQ/TILE_M,
    # SKV/TILE_N), B-extent = B.  THD: the SF tensor is PACKED (batch dim 1)
    # and per-sequence-TILE-padded, so the descriptor uses the bound view's
    # packed tile extent (Σ_b ceil(S/TILE), a DYNAMIC value carried by the SF
    # tensor's shape) and B = 1.
    sq_sf_tiles = (SQ + CFG.TILE_M - 1) // CFG.TILE_M
    skv_sf_tiles = (SKV + CFG.TILE_N - 1) // CFG.TILE_N
    if cutlass.const_expr(CFG.THD_VARLEN):
        _B_SF = 1
        _q_sf_num_tiles = sf_q_tensor.shape[2]
        _kv_sf_num_tiles = sf_k_tensor.shape[2]
    else:
        _B_SF = B
        _q_sf_num_tiles = sq_sf_tiles
        _kv_sf_num_tiles = skv_sf_tiles

    def _build_sf_desc(sf_tensor, num_tiles, sf_smem_size, num_rows_box, num_heads):
        sf_base = cutlass.Int64(sf_tensor.iterator.toint())
        tile_stride_16 = sf_smem_size // 16
        return tmap.create_tensor_map_tiled(
            global_address=sf_base,
            dtype=cutlass.Uint8,
            global_dims=[
                SF_TMA_ROW_BYTES,
                sf_smem_size // SF_TMA_ROW_BYTES,
                num_tiles,
                num_heads,
                _B_SF,
            ],
            global_strides=[
                SF_TMA_ROW_BYTES // 16,
                tile_stride_16,
                num_tiles * tile_stride_16,
                num_heads * num_tiles * tile_stride_16,
            ],
            box_dims=[SF_TMA_ROW_BYTES, num_rows_box, 1, 1, 1],
            swizzle=tmap.TensorMapSwizzle.none,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_sf_desc = _build_sf_desc(sf_q_tensor, _q_sf_num_tiles, SF_SMEM_SIZE_Q, SF_NUM_ROWS_Q, QH)
    tma_k_sf_desc = _build_sf_desc(sf_k_tensor, _kv_sf_num_tiles, SF_SMEM_SIZE_K, SF_NUM_ROWS_K // CFG.CTA_MMA, KH)
    tma_v_sf_desc = _build_sf_desc(sf_v_tensor, _kv_sf_num_tiles, SF_SMEM_SIZE_V, SF_NUM_ROWS_V // CFG.CTA_MMA, KH)

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD setup launch: build the [kv|cu_q|cu_k] metadata buffer
        # DEVICE-side from the caller's length tensors (no host cumsum, no
        # H2D — issue #552), then the per-batch O descriptor array (reuse
        # tma_o_desc over the packed [1,T,QH,D_v] O as base). Main grid: the
        # PLAN-TIME ENVELOPE (n_thd_units = B * ceil(S_q_decl/CGA_TILE_M) * QH,
        # from the DECLARED S_q — no runtime length reaches the host); units
        # past a sequence's live tiles decode the batch == n_batch sentinel
        # and drain without loads or stores. grid_x = n_thd_units * CGA_M.
        # Per-token element stride of packed O (o_tensor.stride[1] = QH * d_v)
        # — NOT CFG.TILE_O, which is only coincidentally right at QH == 1.
        # The FP8-family setup variant also clamps runtime K/V descriptors to
        # the packed KV total (slots n_batch+1/+2 of o_desc_words) so
        # tile-tail loads past it zero-fill instead of reading the buffer's
        # capacity tail — a NaN tail would poison BMM2's P·V (0 · NaN == NaN).
        _build_thd_meta_o_kv_descs_kernel(
            o_tensor,
            tma_o_desc,
            tma_k_desc,
            tma_v_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            thd_q_lens_tensor,
            thd_kv_lens_tensor,
            thd_lens_form,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(o_tensor.stride[1]),
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        # KV split rides the BATCH axis: z = batch + split*B.  The decode
        # already recovers the batch coord on both the blockIdx and the
        # scheduler-handout paths, so the split travels with it for free.
        grid_shape = (
            (grid_q_supers, QH, B * SPLIT_KV) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B * SPLIT_KV, 1, 1)
        )
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        tma_q_sf_desc,
        tma_k_sf_desc,
        tma_v_sf_desc,
        lse_tensor,
        amax_o_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(q_supers),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CTA_MMA, 1, 1),
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 128,
    has_lse: bool = True,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple[int, int, int]] = None,
) -> Callable:
    """Compile a kernel with concrete dims; 3 SF tensors layout [B, H, num_seq_tiles, SF_SMEM_SIZE_*].

    THD/varlen: q/k/v/o/lse + SF tensors are PACKED with batch dim 1; ``b`` is
    the LOGICAL batch (sequence count).  ``sq``/``skv`` are IGNORED under THD —
    the packed token totals AND the packed SF tile extents (Σ_b ceil(S/TILE),
    per-sequence-TILE-padded) are runtime values, so they compile DYNAMIC
    (``cute.sym_int``) and the cache key stays plan-time-only (issue #552).
    THD Stats layouts: token-major packed rank-2 (T, H) by default;
    ``lse_head_major=True`` = rank-3 [1, QH, head_stride] (0 → compact).
    ``has_lse=False`` compiles the LSE store out (the kernel specializes on a
    ``None`` LSE argument) — callers without a Stats output pass no LSE buffer
    at all; the amax_o atomicMax write is independent and unchanged."""
    if SPLIT_KV > 1 and not has_lse:
        # Each split's LSE is not optional under KV split — it IS the weight
        # the combine reduces with.  Without it the partials cannot be recombined.
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    _fake_batch = 1 if CFG.THD_VARLEN else b
    # KV split: O and LSE are the PARTIAL workspaces, stacked split-major on
    # the batch axis (B*SPLIT_KV).  Q/K/V keep the real batch.  THD packs the
    # batch away (dim 1) and split_kv is dense-only (config backstop), so the
    # THD fakes see SPLIT_KV == 1.
    _o_batch = _fake_batch * SPLIT_KV
    _lse_batch = b * SPLIT_KV
    if CFG.THD_VARLEN:
        # Dynamic packed extents: one symbol per ragged group (Q/O and a
        # token-major LSE share t_q; K/V share t_kv; the Q and K/V SF packed
        # tile totals carry their own), so a new packed partition re-binds the
        # same compiled artifact instead of minting a new one (issue #552).
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)
        _q_sf_tiles = cute.sym_int(divisibility=1)
        _kv_sf_tiles = cute.sym_int(divisibility=1)
    else:
        # Q SF tiles TILE_M-row wide → num_tiles = SQ/TILE_M; K/V SF TILE_N-row wide → num_tiles = SKV/TILE_N.
        _q_sf_tiles = (sq + CFG.TILE_M - 1) // CFG.TILE_M
        _kv_sf_tiles = (skv + CFG.TILE_N - 1) // CFG.TILE_N

    fake_q = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, sq, qh, CFG.TILE_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, CFG.TILE_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, CFG.TILE_O),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        OUT_STORAGE_DTYPE,
        (_o_batch, sq, qh, CFG.TILE_O),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )

    fake_sf_q = cute.runtime.make_fake_compact_tensor(
        cutlass.Int8,
        (_fake_batch, qh, _q_sf_tiles, SF_SMEM_SIZE_Q),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_sf_k = cute.runtime.make_fake_compact_tensor(
        cutlass.Int8,
        (_fake_batch, kh, _kv_sf_tiles, SF_SMEM_SIZE_K),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_sf_v = cute.runtime.make_fake_compact_tensor(
        cutlass.Int8,
        (_fake_batch, kh, _kv_sf_tiles, SF_SMEM_SIZE_V),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )

    if not has_lse:
        # No Stats output: the LSE argument is None-specialized and the store
        # is compiled out entirely — no dummy buffer exists at any level.
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        # Packed ragged-Stats LSE in the caller's declared layout (align 4:
        # the store is scalar f32 and the caller's Stats buffer only
        # guarantees element alignment). The epilogue store branches on the
        # STATIC rank, so the layout is fully encoded in this fake tensor.
        if lse_head_major:
            _lse_hs = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (1, qh, _lse_hs),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only (token-major (T, H) is compact)")
            fake_lse = cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (sq, qh),
                stride_order=(1, 0),
                assumed_align=4,
            )
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are THD-only (dense LSE is compact (B, H, Sq))")
        fake_lse = (
            cute.runtime.make_fake_tensor(cutlass.Float32, (_lse_batch, qh, sq), lse_stride, assumed_align=4)
            if lse_stride is not None and SPLIT_KV == 1
            else cute.runtime.make_fake_compact_tensor(
                cutlass.Float32,
                (_lse_batch, qh, sq),
                stride_order=(2, 1, 0),
                assumed_align=16,
            )
        )
    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    # seq_kv_lens always part of the ABI; unread when CFG.SEQ_KV_LENS_PRESENT
    # == 0.  THD overloads it as [seq_kv_lens(B)|cu_q(B+1)|cu_k(B+1)] (len 3B+2).
    _skv_len = (3 * b + 2) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Per-batch O TMA-descriptor array (16 int64 = 128 B each) + 1 pad slot;
    # dummy 1-elem when THD off (kernel never reads it).
    # +2 slots after the pad slot: the packed-total-clamped K and V runtime
    # descriptors (see the THD tma_k/tma_v closures).
    _odesc_len = ((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # THD: the caller's Q/KV length tensors, consumed by the setup kernel's
    # device-side metadata build. DYNAMIC extents — (B,) per-batch lengths and
    # (B+1,) cu prefix sums bind the same artifact; the form rides the runtime
    # thd_lens_form bitmask, so no compile key grows (Rule 4). align 4: bound
    # directly, only natural int32 alignment is guaranteed.
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = None
        fake_thd_kv_lens = None
        fake_thd_lens_form = None
    return cute.compile(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_sf_q,
        fake_sf_k,
        fake_sf_v,
        fake_lse,
        fake_amax_o,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        # THD: the packed totals are runtime values carried by the (dynamic)
        # tensor extents — _host reads them from the views' shapes.
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
