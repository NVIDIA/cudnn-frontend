# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""pre-upstream DSL qwen prefill SDPA MXFP8 kernel (d_qk = d_v = 256).

Qwen pipeline: TILES_Q=1, single softmax wg, Q∪O SMEM alias, Q*K(i+1)→S*V(i)
lookahead, parity-keyed S_acc/P, bootstrap-only-lo-parity bmm2_ready in
correction, P14 end-of-tile catch-up flip on bmm2_done_phase, LSE written in
correction.

MXFP8 layer: 4 SF SMEM buffers (Q/K/P/V Int8, alignment=1024); SF TMEM cols
parked after O accumulator and before Stats; SF TMA rides the same
cta_group=CTA_MMA collective TMA as Q/K/V; block-scale mma_ss / mma_ts_step
with is_block_scale=True; MMA kind MXF8F6F4; only DTYPE_QKV ∈ {0,1} accepted.

d=256 BMM2 N-block split: under cga1 issues TWO mma_ts_step calls per kv iter
(one per 128 N-rows of O with shifted tmem_O / tmem_SF_V views + shifted
desc_V); under cga2 the collective MMA covers full d_v in one call.

READ_TILE_ARRIVERS overridden here: ((SOFTMAX_WG*4)+CORR+3)*CGA_M*CGA_N —
the qwen flavor config's value is correct only for F16/FP8.

THD / varlen (CFG.THD_VARLEN=1) supported (E4M3/E5M2) at cga2 via the shared
thd helpers: Q/K/V data ride the packed [1,T,H,D] + cu_seqlens coord path;
the MXFP8 SF tensors use B=1 + total_*_sf_tiles descriptors with a per-sequence
SF-tile prefix base (cu_sf_q/k from _thd_sf_tile_bases) added to the SF tile
coord.  cga2 within-tile peer SF byte offsets are orthogonal and unchanged.
Dense path (THD_VARLEN=0) folds to identity (cu_sf=0, tma_batch=batch_idx) —
byte-identical.  Public-API MXFP8 THD glue is a follow-up (THD-aware quantizer).
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental.cuda import tensor_map as tmap
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass
from typing import NamedTuple

# Config comes from the FROST template loader, NOT an env var: the pre-upstream
# kernels picked a flavor with an env var + a sdpa_config_<flavor> module, which the
# FROST engine contract forbids ("no environment variables for configuration" --
# parameters travel as typed dataclasses). The loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a plain
# `import` usable as a standalone driver.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256_mxfp8

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d256_mxfp8(PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE
# decision point, wired into every construction below rather than repeated as a
# per-tile literal.  A version-0 descriptor's ``start_address`` is 14 bits = a
# 256 KiB window; Rubin raises the per-CTA SMEM cap to 327 KiB, so an operand
# buffer at or above 256 KiB wraps to offset 0 and the MMA multiplies whatever
# sits at the bottom of SMEM.  This flavor's buffers all stay below the line.
#
# Do NOT re-literal this at a call site: the d512 MXFP8 sibling shipped NaN on
# 100% of cells because its scale-factor tiles were declared UNDER a comment
# claiming "every operand tile here carries desc_version=1" -- without the
# kwarg.  A single constant makes that class of drift impossible, and
# test_sm107_descriptor_version_matches_the_smem_budget asserts it.
DESC_VERSION: int = 0
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# MXFP8 K-step pin -- k_dim=1 selects 64-element K chunks, halving the
# tcgen05.mma.block_scale issue count versus k_dim=0.  ``config_sm107`` derives
# TILE_K_HW from the dtype; this guard pins the two together, because the
# pairing is arch-OPPOSITE (Blackwell wants k_dim=0 with 32) and a mismatch
# scrambles accumulator rows SILENTLY (rules/mma-tma-matrix.md S1).
_MXFP8_K_DIM = 1
_MXFP8_TILE_K_HW = 32 if _MXFP8_K_DIM == 0 else 64
if CFG.TILE_K_HW_BMM1 != _MXFP8_TILE_K_HW or CFG.TILE_K_HW_BMM2 != _MXFP8_TILE_K_HW:
    raise ValueError(
        f"{__name__}: idesc k_dim={_MXFP8_K_DIM} requires TILE_K_HW={_MXFP8_TILE_K_HW}; " f"got BMM1={CFG.TILE_K_HW_BMM1} BMM2={CFG.TILE_K_HW_BMM2}"
    )

# MXFP8 inputs are FP8 storage with E8M0 SF — only E4M3 (0) / E5M2 (1) valid.
if CFG.DTYPE_QKV not in (0, 1):
    raise ValueError(f"prefill_sdpa_d256_mxfp8: DTYPE_QKV must be 0 (E4M3) or 1 (E5M2); " f"got {CFG.DTYPE_QKV}.  Use prefill_sdpa_d256_f16.py for BF16/FP16.")

# O TMA box follows O swizzle, not V.
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) — still used for sched.mb_* (Sched not in Bars).
    wait,
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
    tmem_load_max_reduction_x64,
    row_reduction_pair_64,
    row_max_reduction_64,
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
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)

if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2

MMA_KIND = nvvm.MMABlockScaleKind.MXF8F6F4
SCALE_VEC_SIZE = nvvm.Tcgen05MMAScaleVecSize.BLOCK32


# DTYPE_O independent of DTYPE_QKV — MXFP8 input may write BF16/FP16 O so a
# downstream consumer skips a dequant.  BPE_O ∈ {1, 2}; epilogue already casts
# via .to(OUT_STORAGE_DTYPE) + store_swizzled (dtype-generic).  The Q∪O SMEM
# alias is byte-sized for max(Q@BPE, O@BPE_O) so BF16 O doesn't overflow it.
if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_d256_mxfp8: DTYPE_O={CFG.DTYPE_O} not supported " f"(expected 0=E4M3 / 1=E5M2 / 2=BF16 / 3=FP16)")


# === MXFP8 SF constants (trace-time folded) ===
BITS_PER_SF_ELEMENT = 8  # E8M0
BLOCK_SCALE_BLOCK_SIZE = 32  # 32 elems share one SF byte
SF_BLOCK_DIM_NON_K = 128
SF_BLOCK_DIM_K = 4
SF_SWIZZLED_BLOCK_DIM_K = 16
SF_BYTES_PER_BLOCK = SF_BLOCK_DIM_NON_K * SF_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 8  # 512


def _round_up(a: int, b: int) -> int:
    return (a + b - 1) // b * b


SF_NUM_BLOCKS_M = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_N = CFG.TILE_N // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K = _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K

SF_REGISTERS_PER_BLOCK = SF_SWIZZLED_BLOCK_DIM_K * BITS_PER_SF_ELEMENT // 32

SF_TMEM_COLS_Q = SF_NUM_BLOCKS_M * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK
SF_TMEM_COLS_K = SF_NUM_BLOCKS_N * SF_NUM_BLOCKS_K * SF_REGISTERS_PER_BLOCK

SF_SMEM_SIZE_Q = CFG.TILE_M * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_K = CFG.TILE_N * _round_up(CFG.TILE_K, 128) // BLOCK_SCALE_BLOCK_SIZE

# BMM2 SF: P softmax-filled with constant 1.0; V from GMEM.
SF_NUM_BLOCKS_P = CFG.TILE_M // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_V = _round_up(CFG.TILE_O, 128) // SF_BLOCK_DIM_NON_K
SF_NUM_BLOCKS_K_BMM2 = CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE // SF_BLOCK_DIM_K
SF_TMEM_COLS_P = SF_NUM_BLOCKS_P * SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK
SF_TMEM_COLS_V = SF_NUM_BLOCKS_V * SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK
SF_SMEM_SIZE_P = _round_up(CFG.TILE_M, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE
SF_SMEM_SIZE_V = _round_up(CFG.TILE_O, 128) * CFG.TILE_N // BLOCK_SCALE_BLOCK_SIZE

# V's SF D-planes are the cga peer split unit (see the V SF descriptor in
# _host): each CTA of the pair owns a contiguous run of D-planes, matching the
# V data columns it holds.
if SF_NUM_BLOCKS_V % CFG.CTA_MMA != 0:
    raise ValueError(
        f"prefill_sdpa_d256_mxfp8: V SF D-plane count {SF_NUM_BLOCKS_V} is not " f"divisible by CTA_MMA={CFG.CTA_MMA}; the cga peer split walks whole D-planes"
    )
V_SF_PLANES_PER_PEER = SF_NUM_BLOCKS_V // CFG.CTA_MMA

# E8M0 1.0 (=2^0) — fills P_SF SMEM.
SF_CONST_VALUE = 0x7F


# === d=256 BMM2 N-block split ===
# Block-scale mma_ts_step cycles SF B only along K, so V's d_v > 128 needs one
# MMA per V SF N-block (cga1 d_v=256 → 2 calls; cga2 collective covers full).
SF_BMM2_N_BLOCKS = _round_up(CFG.TILE_O, 128) // 128
BMM2_LOOP_N_BLOCKS = SF_BMM2_N_BLOCKS // CFG.CTA_MMA
BMM2_N_PER_CALL = CFG.TILE_O // BMM2_LOOP_N_BLOCKS
BMM2_N_PER_CALL_PER_CTA = BMM2_N_PER_CALL // CFG.CTA_MMA
BMM2_N_BLOCK_BYTE_STRIDE = CFG.TILE_N * CFG.V_SWZ_BYTES


from cudnn.sdpa.fwd.kernels._common_sm100 import (
    D256Bars as Bars,
    KvLoopBounds,
    make_d256_bars,
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
qoAliasBytes = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

# SF expect_tx bytes per TMA call.
Q_SF_EXPECT_BYTES = SF_SMEM_SIZE_Q * CFG.CTA_MMA
K_SF_EXPECT_BYTES = SF_SMEM_SIZE_K * CFG.CTA_MMA
V_SF_EXPECT_BYTES = SF_SMEM_SIZE_V * CFG.CTA_MMA

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

_sdpa_h = make_sdpa_helpers(CFG)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
_bounds_for_tile = _sdpa_h.bounds_for_tile
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; O-descriptor builder + TENSOR_MAP_QWORDS from the shared
# the shared pre-upstream THD helper.  Gated by CFG.THD_VARLEN (folds out otherwise).
# Supported at cga1 and cga2 (the per-batch O descriptor's seq extent OOB-clips
# the 256-row cga2 store box).  seq_kv_lens overloaded as the THD metadata buffer
# (int32 len 3B+2): [0..B-1]=seq_kv_lens [B..2B]=cu_q(B+1) [2B+1..3B+1]=cu_k(B+1).
# MXFP8-only: _thd_sf_tile_bases returns the per-sequence SF-tile prefix bases
# (cu_sf_q_base / cu_sf_k_base) for the packed scale-factor layout.
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets
_thd_sf_tile_bases = _sdpa_h.thd_sf_tile_bases


# Kernel TMEM layout (TOTAL=576 Rubin cap; Stats parked outside S_acc/O range,
# P aliases S_acc tail, SF cols between O accumulator and Stats).
@dataclass(frozen=True)
class KernelTmemLayout:
    TOTAL_COLS: int = 576

    S_ACC_EVEN_OFF: int = 0
    S_ACC_ODD_OFF: int = 128

    P_EVEN_OFF: int = 96
    P_ODD_OFF: int = 224

    O_OFF: int = 256

    SF_Q_OFF: int = 512
    SF_K_OFF: int = 512 + SF_TMEM_COLS_Q
    SF_P_OFF: int = 512 + SF_TMEM_COLS_Q + SF_TMEM_COLS_K
    SF_V_OFF: int = 512 + SF_TMEM_COLS_Q + SF_TMEM_COLS_K + SF_TMEM_COLS_P

    STATS_OFF: int = 544


LAYOUT = KernelTmemLayout()
assert LAYOUT.SF_V_OFF + SF_TMEM_COLS_V <= LAYOUT.STATS_OFF, "TMEM SF region collides with Stats — re-check SF_*_TMEM_COLS"


# Overrides CFG.READ_TILE_ARRIVERS (which is the F16/FP8 qwen-quiet-minimal
# formula).  Qwen MXFP8 quiet warp keeps the persistent loop so every CTA
# contributes the full per-CTA arriver count.
READ_TILE_ARRIVERS_MXFP8 = (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + CFG.CORRECTION_WARPS + 3) * CFG.CGA_M * CFG.CGA_N


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

# SF SmemDesc — no swizzle, leading=16, stride=128 (utccp_32x128b_warpx4).
SMEM_LAYOUT_SF = 0
SF_LEADING_BYTE_OFFSET = 16
SF_STRIDE_BYTE_OFFSET = 128

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N*V_SWZ_BYTES
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

NUM_KPHASES_PV = CFG.TILE_N // _MXFP8_TILE_K_HW
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS

SF_V_COLS_PER_NBLOCK = SF_NUM_BLOCKS_K_BMM2 * SF_REGISTERS_PER_BLOCK


# === Kernel entry ===


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_q_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_sf_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    # FROST's MXFP8 ABI puts amax_o RIGHT AFTER lse (position 9).
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # Q∪O alias byte-sized to max(Q@BPE, O@BPE_O) — STORAGE_DTYPE is FP8 (1 B)
    # so element count == byte count; BF16/FP16 O (BPE_O=2) would otherwise
    # overflow a qBufferElems-only allocation.
    _QO_ALIAS_ELEMS = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)
    sQO_raw = cutlass.Array(STORAGE_DTYPE, _QO_ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # OUT_STORAGE_DTYPE view of the Q∪O backing — epilogue store offsets then
    # advance in BPE_O units (no-op recast for FP8 output).
    sO_raw = cutlass.Array(sQO_raw.data_ptr(), shape=oBufferElems, dtype=OUT_STORAGE_DTYPE)

    # SF buffers Int8 alignment=1024; Q_SF/P_SF single-stage.
    sQ_SF_raw = cutlass.Array(cutlass.Int8, SF_SMEM_SIZE_Q, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_K, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_SF_raw = cutlass.Array(cutlass.Int8, SF_SMEM_SIZE_P, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_SF_raw = cutlass.Array(cutlass.Int8, CFG.STAGES_KV * SF_SMEM_SIZE_V, alignment=1024, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
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
        desc_version=DESC_VERSION,
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
        desc_version=DESC_VERSION,
    )
    # sO shares backing with sQ (Q∪O alias) — OUT_STORAGE_DTYPE view for BPE_O offsets.
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
        desc_version=DESC_VERSION,
    )

    sQ_SF = SmemTile(
        base=sQ_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_Q,
        stages=1,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sK_SF = SmemTile(
        base=sK_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_K,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sP_SF = SmemTile(
        base=sP_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_P,
        stages=1,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )
    sV_SF = SmemTile(
        base=sV_SF_raw,
        elems_per_stage=SF_SMEM_SIZE_V,
        stages=CFG.STAGES_KV,
        leading_byte_offset=SF_LEADING_BYTE_OFFSET,
        stride_byte_offset=SF_STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SF,
        desc_version=DESC_VERSION,
    )

    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS)

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

    if warp_idx == 0:
        if nvvm.elect_sync():
            # Init counts baked into make_d256_bars(...); kernel calls .init()
            # per stage.  range_constexpr keeps loop vars Python int.
            bars.mb_q_full.init()
            bars.mb_q_o_alias.init()
            bars.mb_tmastg_go.init()
            for p in cutlass.range_constexpr(2):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()
            bars.mb_stat_full.init()
            bars.mb_stat_empty.init()
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].init()
            bars.mb_o_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                # MXFP8 quiet warp matches FP8 minimal — keep CFG arriver count.
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), CFG.READ_TILE_ARRIVERS)
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            # Bootstrap pre-armed so first wait passes immediately (no prior O).
            bars.mb_q_o_alias.arrive()

    # P_SF SMEM fill with 0x7F (E8M0 1.0) — MUST precede cga_arrive/cga_wait
    # so both cga2 peers see filled SMEM before any cta_group::2 MMA.
    _SF_P_ITERS = (SF_SMEM_SIZE_P + CFG.THREADS_PER_CTA - 1) // CFG.THREADS_PER_CTA
    for _i in cutlass.range_constexpr(_SF_P_ITERS):
        _off = tidx + cutlass.Int32(_i * CFG.THREADS_PER_CTA)
        if _off < cutlass.Int32(SF_SMEM_SIZE_P):
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
    sf_mcast_mask = cutlass.Int16(3) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Warp layout: 0..3 softmax wg, 4..7 correction, 8 MMA, 9 TMALDG,
    # 10 TMASTG, 11 scheduler.
    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
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
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
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
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
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
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


# === TMA-LDG warp group ===


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
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    sf_mcast_mask,
):
    """Q/K/V/SF TMA loads with cga2 SF split-half multicast.

    Per-tile SF TMA loads ride the SAME mb_q/k/v_full mbar — SF expect_tx
    adds Q_SF_EXPECT_BYTES / K_SF_EXPECT_BYTES / V_SF_EXPECT_BYTES.  Under
    cga2 each peer pulls HALF of K_SF / V_SF via bulk_copy_multicast
    (mcast=Int16(3) distributes the half to both peers).
    """
    q_o_alias_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_v = GmemTileTma(tma_v_desc)
    tma_q_sf = GmemTileTma(tma_q_sf_desc)
    tma_k_sf = GmemTileTma(tma_k_sf_desc)
    tma_v_sf = GmemTileTma(tma_v_sf_desc)

    SF_TMA_ROW_BYTES = 128
    K_SF_BYTES_PER_PEER = SF_SMEM_SIZE_K // CFG.CTA_MMA
    V_SF_BYTES_PER_PEER = SF_SMEM_SIZE_V // CFG.CTA_MMA
    K_SF_ROWS_PER_PEER = K_SF_BYTES_PER_PEER // SF_TMA_ROW_BYTES
    k_sf_peer_off = cta_in_pair * cutlass.Int32(K_SF_BYTES_PER_PEER)
    v_sf_peer_off = cta_in_pair * cutlass.Int32(V_SF_BYTES_PER_PEER)
    k_sf_peer_row = cta_in_pair * cutlass.Int32(K_SF_ROWS_PER_PEER)
    # V's SF descriptor is D-plane-major (host), so the peer split is a plane
    # coord, not a row offset (K, being rowwise, keeps the row form).
    v_sf_peer_plane = cta_in_pair * cutlass.Int32(V_SF_PLANES_PER_PEER)
    n_kh = n_qh // qh_per_kh

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    # THD packed-tensor seq offsets (fold to (0,0,batch_idx) dense); SF-tile bases
    # fold to (0,0) dense.
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
    cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        bounds_init = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Wait q_o_alias before clobbering Q∪O with next tile's Q.
        bars.mb_q_o_alias.wait(q_o_alias_phase)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            pass
        else:
            q_sf_tile_base = q_row_base // cutlass.Int32(CFG.TILE_M)

            # P9 — cga2 collective TMA routes mbar arrives to leader.
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes + Q_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            tma_load_tile(
                sQ_SF[0],
                tma_q_sf(cutlass.Int32(0), cu_sf_q_base + q_sf_tile_base, head_idx, tma_batch, coord_0=cutlass.Int32(0)),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes + K_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
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
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes + V_SF_EXPECT_BYTES, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )
                tma_load_tile(
                    sV_SF[kv_state.idx].shifted(v_sf_peer_off),
                    tma_v_sf(cutlass.Int32(0), v_sf_peer_plane, cu_sf_k_base + kv_loop, tma_batch * n_kh + kv_head_idx, coord_0=cutlass.Int32(0)),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=sf_mcast_mask,
                )

                kv_state = advance(kv_state, CFG.STAGES_KV)

        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        cu_sf_q_base, cu_sf_k_base = _thd_sf_tile_bases(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            bounds_next = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# === TMA-STG warp group ===


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
):
    """TMA-STG warp — no SF channel on the O store path."""
    tmastg_go_phase = cutlass.Int32(0)
    o_full_phase = cutlass.Int32(0)
    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        for chunk in cutlass.range_constexpr(N_O_CHUNKS):
            bars.mb_o_full[chunk].wait(o_full_phase)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)

        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: store through this batch's pre-built descriptor (seq extent
            # = S_q_b → box past S_q_b OOB-clipped).  q_row_coord seq-local; batch→0.
            o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
            o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), head_idx, q_row_coord, cutlass.Int32(0))
            tma_store_tile(sO[0], o_slice)
        else:
            tma_store_tile(
                sO[0],
                tma_o(cutlass.Int32(0), head_idx, q_row_coord, batch_idx),
            )
        tma_store_commit()
        tma_store_wait(0)

        bars.mb_o_empty.arrive()
        if nvvm.elect_sync():
            bars.mb_q_o_alias.arrive()

        o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# === MMA warp group (+ quiet warp for cga2 non-leader) ===


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
    """Quiet MMA warp (cga2 non-leader).

    P9 — cga2 collective TMA routes ALL byte arrives to the leader's mbar,
    so peer's mb_q_full never receives bytes.  SF split-half multicast
    (sf_mcast_mask=Int16(3)) delivers SF to both peers, so no peer DSMEM
    forward needed.  Minimal: tmem_alloc → 2× barrier_arrive → wait dealloc
    → tmem_dealloc.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
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
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
):
    """Block-scale MMA stream for BMM1 + BMM2 (Q*K(i+1) → S*V(i) lookahead).

    Prologue: copy_sf(SF_Q,SF_K) → Q*K(lo) → bmm1_done[lo&1].
    Mainloop: copy_sf(SF_K) → Q*K(kv+1); copy_sf(SF_V) → BMM2 N-block loop
    with bmm2_ready[parity_cur][chunk] gates.
    Epilogue: copy_sf(SF_V) → S*V(hi-1) BMM2 N-block loop.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    # Int8 pointer typing — Float16 typing strides in fp16 elems not cols.
    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk_bs = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=_MXFP8_K_DIM,
    )
    # BMM2 N = BMM2_N_PER_CALL (per N-block).
    idesc_pv_bs = prims.Tcgen05MxInstrDesc.build(
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=_MXFP8_K_DIM,
    )

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
        N=BMM2_N_PER_CALL,
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

    desc_Q = sQ[0].desc()
    desc_Q_SF = sQ_SF[0].desc()
    desc_P_SF = sP_SF[0].desc()

    tmem_SF_Q = tmem_raw.subview(LAYOUT.SF_Q_OFF)
    tmem_SF_K = tmem_raw.subview(LAYOUT.SF_K_OFF)
    tmem_SF_P = tmem_raw.subview(LAYOUT.SF_P_OFF)
    tmem_SF_V = tmem_raw.subview(LAYOUT.SF_V_OFF)

    # One-shot copy_sf SMEM(P_SF) → TMEM(SF_P) — P_SF constant 1.0.
    if nvvm.elect_sync():
        nvvm.tcgen05_cp(
            nvvm.Tcgen05CpShape.SHAPE_32X128B,
            tmem_SF_P,
            desc_P_SF,
            group=CTA_GROUP_KIND,
            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
        )

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx = _dispatch_decode_initial(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        bounds_init = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    bmm2_ready_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            bars.mb_q_full.wait(q_full_phase)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            lo_parity_runtime = kv_left & cutlass.Int32(1)
            parity_lo_is_even = lo_parity_runtime == cutlass.Int32(0)
            tmem_S_acc_lo_addr = cutlass.Int32(
                arith.select(
                    parity_lo_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                )
            )

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
            desc_K = sK[kv_state_K.idx].desc()
            desc_K_SF = sK_SF[kv_state_K.idx].desc()
            if nvvm.elect_sync():
                for _sf_k in cutlass.range_constexpr(SF_NUM_BLOCKS_K):
                    nvvm.tcgen05_cp(
                        nvvm.Tcgen05CpShape.SHAPE_32X128B,
                        tmem_SF_Q.subview(_sf_k * SF_REGISTERS_PER_BLOCK),
                        desc_Q_SF + _sf_k * (SF_BYTES_PER_BLOCK // 16),
                        group=CTA_GROUP_KIND,
                        multicast=nvvm.Tcgen05CpMulticast.WARPX4,
                    )
                    nvvm.tcgen05_cp(
                        nvvm.Tcgen05CpShape.SHAPE_32X128B,
                        tmem_SF_K.subview(_sf_k * SF_REGISTERS_PER_BLOCK),
                        desc_K_SF + _sf_k * (SF_BYTES_PER_BLOCK // 16),
                        group=CTA_GROUP_KIND,
                        multicast=nvvm.Tcgen05CpMulticast.WARPX4,
                    )
            mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_lo_addr)), tmem_sf_a=tmem_SF_Q, tmem_sf_b=tmem_SF_K)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            k_per_chunk = NUM_KPHASES_PV_PER_CHUNK

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                cur_is_even = parity_cur_rt == cutlass.Int32(0)
                next_is_even = parity_next_rt == cutlass.Int32(0)

                tmem_S_acc_next_addr = cutlass.Int32(
                    arith.select(
                        next_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                    )
                )
                tmem_P_cur_addr = cutlass.Int32(
                    arith.select(
                        cur_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                    )
                )
                bmm2_ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)

                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                desc_K_SF = sK_SF[kv_state_K.idx].desc()
                if nvvm.elect_sync():
                    for _sf_k in cutlass.range_constexpr(SF_NUM_BLOCKS_K):
                        nvvm.tcgen05_cp(
                            nvvm.Tcgen05CpShape.SHAPE_32X128B,
                            tmem_SF_K.subview(_sf_k * SF_REGISTERS_PER_BLOCK),
                            desc_K_SF + _sf_k * (SF_BYTES_PER_BLOCK // 16),
                            group=CTA_GROUP_KIND,
                            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
                        )
                mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_next_addr)), tmem_sf_a=tmem_SF_Q, tmem_sf_b=tmem_SF_K)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                desc_V = sV[kv_state_V.idx].desc()
                desc_V_SF = sV_SF[kv_state_V.idx].desc()
                if nvvm.elect_sync():
                    # SF V SMEM→TMEM needs per-N-block loop when SF_NUM_BLOCKS_V>1.
                    for _sf_n in cutlass.range_constexpr(SF_NUM_BLOCKS_V):
                        nvvm.tcgen05_cp(
                            nvvm.Tcgen05CpShape.SHAPE_32X128B,
                            tmem_SF_V.subview(_sf_n * SF_REGISTERS_PER_BLOCK),
                            desc_V_SF + _sf_n * (SF_BYTES_PER_BLOCK // 16),
                            group=CTA_GROUP_KIND,
                            multicast=nvvm.Tcgen05CpMulticast.WARPX4,
                        )

                scaleC = cutlass.Boolean(kv_loop != kv_left)
                for nblk in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                    tmem_O_n = tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + nblk * BMM2_N_PER_CALL))
                    tmem_SF_V_n = tmem_SF_V.subview(cutlass.Int32(nblk * SF_V_COLS_PER_NBLOCK))
                    accum_b2 = scaleC
                    for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                        if nblk == 0 and k % k_per_chunk == 0:
                            chunk_id = k // k_per_chunk
                            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                        mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_cur_addr)), desc_V, tmem_O_n, k, accum_b2, tmem_sf_a=tmem_SF_P, tmem_sf_b=tmem_SF_V_n)
                        accum_b2 = cutlass.Boolean(True)
                    if cutlass.const_expr(BMM2_LOOP_N_BLOCKS > 1):
                        if nblk + 1 < BMM2_LOOP_N_BLOCKS:
                            desc_V = desc_V + cutlass.Int32(BMM2_N_BLOCK_BYTE_STRIDE // 16)
                if cutlass.const_expr(BMM2_LOOP_N_BLOCKS > 1):
                    desc_V = desc_V - cutlass.Int32((BMM2_LOOP_N_BLOCKS - 1) * BMM2_N_BLOCK_BYTE_STRIDE // 16)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            last_is_even = parity_last_rt == cutlass.Int32(0)
            tmem_P_last_addr = cutlass.Int32(
                arith.select(
                    last_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                )
            )
            bmm2_ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)

            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
            desc_V = sV[kv_state_V.idx].desc()
            desc_V_SF = sV_SF[kv_state_V.idx].desc()
            if nvvm.elect_sync():
                for _sf_n in cutlass.range_constexpr(SF_NUM_BLOCKS_V):
                    nvvm.tcgen05_cp(
                        nvvm.Tcgen05CpShape.SHAPE_32X128B,
                        tmem_SF_V.subview(_sf_n * SF_REGISTERS_PER_BLOCK),
                        desc_V_SF + _sf_n * (SF_BYTES_PER_BLOCK // 16),
                        group=CTA_GROUP_KIND,
                        multicast=nvvm.Tcgen05CpMulticast.WARPX4,
                    )

            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            for nblk in cutlass.range_constexpr(BMM2_LOOP_N_BLOCKS):
                tmem_O_n = tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + nblk * BMM2_N_PER_CALL))
                tmem_SF_V_n = tmem_SF_V.subview(cutlass.Int32(nblk * SF_V_COLS_PER_NBLOCK))
                accum_b2 = scaleC_epi
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if nblk == 0 and k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_last_addr)), desc_V, tmem_O_n, k, accum_b2, tmem_sf_a=tmem_SF_P, tmem_sf_b=tmem_SF_V_n)
                    accum_b2 = cutlass.Boolean(True)
                if cutlass.const_expr(BMM2_LOOP_N_BLOCKS > 1):
                    if nblk + 1 < BMM2_LOOP_N_BLOCKS:
                        desc_V = desc_V + cutlass.Int32(BMM2_N_BLOCK_BYTE_STRIDE // 16)
            if cutlass.const_expr(BMM2_LOOP_N_BLOCKS > 1):
                desc_V = desc_V - cutlass.Int32((BMM2_LOOP_N_BLOCKS - 1) * BMM2_N_BLOCK_BYTE_STRIDE // 16)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
            nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
            nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
            q_super_idx, _hd, batch_idx = _dispatch_decode_payload(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            bounds_next = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# === Softmax warp group (single wg — TILES_Q=1) ===


@cute.jit
def _softmax_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    # FROST's MXFP8 ABI puts amax_o RIGHT AFTER lse (position 9).
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
):
    """Softmax: reads S_acc, writes P (FP8 cast).

    SF_P is the constant 1.0 already in tmem.SF_P (filled once at MMA
    warp start).  3-segment kv loop: LEFT-masked / unmasked / RIGHT-masked.
    """
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    bmm1_done_phase_pair = cutlass.Int32(0)
    stat_empty_phase = cutlass.Int32(1)
    epilogue_state = cutlass.Int32(1)

    NEG_INF = cutlass.Float32(-3.4028235e38)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    bounds = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + tid_in_wg

        bars.mb_o_empty.wait(epilogue_state)
        bars.mb_stat_empty.wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        # Body inlined (no nested @cute.jit helper) so the DSL tracer can
        # dispatch through cutlass.range without hitting the closure check.
        CHUNK = 64
        P_COLS_PER_CHUNK = CHUNK // 4  # fp8 packed 4:1 into FP32 cells
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                tmem_base = tmem_ptr_i32.load()
                s_addr_a = tmem_base + s_off_rt
                s_addr_b = tmem_base + s_off_rt + cutlass.Int32(CHUNK)
                p_addr_a = tmem_base + p_off_rt
                p_addr_b = tmem_base + p_off_rt + cutlass.Int32(P_COLS_PER_CHUNK)
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)

                res_a = tmem_load_max_reduction_x64(s_addr_a)
                res_b = tmem_load_max_reduction_x64(s_addr_b)
                reg_S_a = cutlass.Vector.from_elements(tuple(res_a[:CHUNK]), cutlass.Int32).bitcast(cutlass.Float32)
                reg_S_b = cutlass.Vector.from_elements(tuple(res_b[:CHUNK]), cutlass.Int32).bitcast(cutlass.Float32)
                max_a = cutlass.Vector.from_elements((res_a[CHUNK],), cutlass.Int32).bitcast(cutlass.Float32)[0]
                max_b = cutlass.Vector.from_elements((res_b[CHUNK],), cutlass.Int32).bitcast(cutlass.Float32)[0]
                current_max = cute.math.max(max_a, max_b) * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()

                reg_S_a = reg_S_a * scale_log2 - new_total_max
                reg_P_a = cute.math.exp2(reg_S_a, fastmath=True)
                p_a_fp8 = reg_P_a.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_a, cutlass.Float32), p_a_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                reg_S_b = reg_S_b * scale_log2 - new_total_max
                reg_P_b = cute.math.exp2(reg_S_b, fastmath=True)
                p_b_fp8 = reg_P_b.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_b, cutlass.Float32), p_b_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                sum_a_pair = row_reduction_pair_64(reg_P_a)
                sum_b_pair = row_reduction_pair_64(reg_P_b)
                new_p_sum_pair = sum_a_pair + sum_b_pair
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair

                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_a = tmem_base + s_off_rt
                s_addr_b = tmem_base + s_off_rt + cutlass.Int32(CHUNK)
                p_addr_a = tmem_base + p_off_rt
                p_addr_b = tmem_base + p_off_rt + cutlass.Int32(P_COLS_PER_CHUNK)
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                reg_S_a = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_a, cutlass.Float32), num=CHUNK)
                reg_S_b = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_b, cutlass.Float32), num=CHUNK)
                kv_col_base_a = kv_loop * cutlass.Int32(CFG.TILE_N)
                kv_col_base_b = kv_col_base_a + cutlass.Int32(CHUNK)
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
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
                current_max = cute.math.max(max_a, max_b) * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S_a = reg_S_a * scale_log2 - new_total_max
                reg_P_a = cute.math.exp2(reg_S_a, fastmath=True)
                p_a_fp8 = reg_P_a.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_a, cutlass.Float32), p_a_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                reg_S_b = reg_S_b * scale_log2 - new_total_max
                reg_P_b = cute.math.exp2(reg_S_b, fastmath=True)
                p_b_fp8 = reg_P_b.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_b, cutlass.Float32), p_b_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                sum_a_pair = row_reduction_pair_64(reg_P_a)
                sum_b_pair = row_reduction_pair_64(reg_P_b)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + (sum_a_pair + sum_b_pair)
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_a = tmem_base + s_off_rt
                s_addr_b = tmem_base + s_off_rt + cutlass.Int32(CHUNK)
                p_addr_a = tmem_base + p_off_rt
                p_addr_b = tmem_base + p_off_rt + cutlass.Int32(P_COLS_PER_CHUNK)
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                res_a = tmem_load_max_reduction_x64(s_addr_a)
                res_b = tmem_load_max_reduction_x64(s_addr_b)
                reg_S_a = cutlass.Vector.from_elements(tuple(res_a[:CHUNK]), cutlass.Int32).bitcast(cutlass.Float32)
                reg_S_b = cutlass.Vector.from_elements(tuple(res_b[:CHUNK]), cutlass.Int32).bitcast(cutlass.Float32)
                max_a = cutlass.Vector.from_elements((res_a[CHUNK],), cutlass.Int32).bitcast(cutlass.Float32)[0]
                max_b = cutlass.Vector.from_elements((res_b[CHUNK],), cutlass.Int32).bitcast(cutlass.Float32)[0]
                current_max = cute.math.max(max_a, max_b) * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S_a = reg_S_a * scale_log2 - new_total_max
                reg_P_a = cute.math.exp2(reg_S_a, fastmath=True)
                p_a_fp8 = reg_P_a.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_a, cutlass.Float32), p_a_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                reg_S_b = reg_S_b * scale_log2 - new_total_max
                reg_P_b = cute.math.exp2(reg_S_b, fastmath=True)
                p_b_fp8 = reg_P_b.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_b, cutlass.Float32), p_b_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                sum_a_pair = row_reduction_pair_64(reg_P_a)
                sum_b_pair = row_reduction_pair_64(reg_P_b)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + (sum_a_pair + sum_b_pair)
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_a = tmem_base + s_off_rt
                s_addr_b = tmem_base + s_off_rt + cutlass.Int32(CHUNK)
                p_addr_a = tmem_base + p_off_rt
                p_addr_b = tmem_base + p_off_rt + cutlass.Int32(P_COLS_PER_CHUNK)
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                reg_S_a = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_a, cutlass.Float32), num=CHUNK)
                reg_S_b = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_b, cutlass.Float32), num=CHUNK)
                kv_col_base_a = kv_loop * cutlass.Int32(CFG.TILE_N)
                kv_col_base_b = kv_col_base_a + cutlass.Int32(CHUNK)
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
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
                current_max = cute.math.max(max_a, max_b) * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S_a = reg_S_a * scale_log2 - new_total_max
                reg_P_a = cute.math.exp2(reg_S_a, fastmath=True)
                p_a_fp8 = reg_P_a.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_a, cutlass.Float32), p_a_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                reg_S_b = reg_S_b * scale_log2 - new_total_max
                reg_P_b = cute.math.exp2(reg_S_b, fastmath=True)
                p_b_fp8 = reg_P_b.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_b, cutlass.Float32), p_b_fp8)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                sum_a_pair = row_reduction_pair_64(reg_P_a)
                sum_b_pair = row_reduction_pair_64(reg_P_b)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + (sum_a_pair + sum_b_pair)
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)

        total_sum_scalar = total_sum[0] + total_sum[1]
        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full.arrive()

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
        nxt_hb = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
        nxt_v = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load())
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        bounds = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)


# === Correction warp group ===


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    tmem_ptr_i32,
    tidx,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    # FROST's MXFP8 ABI puts amax_o RIGHT AFTER lse (position 9).
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
    """Correction: α-rescale O, sink-aware LSE, cast→store_swizzled.

    MXFP8 inv_sum drops o_scale_fused (per-block SF replaces per-tensor
    scale).  P14 catch-up flip on bmm2_done_phase_pair at end-of-tile is
    REQUIRED for multi-wave n_kv=1.
    """
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    bmm2_done_phase_pair = cutlass.Int32(0)
    stat_mbar_state = cutlass.Int32(0)
    epilogue_state = cutlass.Int32(1)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    bounds = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if bounds.right > bounds.left:
            # Bootstrap-only-lo-parity: arm bmm2_ready[lo_parity * N_CHUNKS + 0]
            # (NOT both parities).
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            bars.mb_stat_full.wait(stat_mbar_state)
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
        else:
            # Empty-mainloop — MMA waits then fires bmm2_done[0] via multicast.
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)
            tmem_base_iter = tmem_ptr_i32.load()

            bars.mb_stat_full.wait(stat_mbar_state)

            stats_addr = tmem_base_iter + cutlass.Int32(LAYOUT.STATS_OFF)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            alpha = stats_vec[0]

            alpha_is_one = alpha == cutlass.Float32(1.0)
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

            bars.mb_stat_empty.arrive()

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

            if ~all_alpha_one:
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        tmem_base_epi = tmem_ptr_i32.load()

        # Pre-declare for DSL if-staging — names used after the conditional
        # must be bound on every path.
        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        if bounds.right > bounds.left:
            bars.mb_stat_full.wait(stat_mbar_state)
            stats_addr_epi = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_OFF)
            stats_vec_epi = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec_epi[0]
            total_sum = stats_vec_epi[1]
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = sinks_arr[head_idx]
            new_max = cute.math.max(total_max_nat, sink_logit)
            scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            inv_sum = scale / new_sum
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            inv_sum = cutlass.Float32(1.0) / cute.math.max(total_sum, cutlass.Float32(1e-30))

        # --- empty KV range (zero-length sequence under the padding mask) ---
        # Same guard as the d256 FP8 sibling: with an empty mainloop
        # total_max/total_sum stay 0, so the 1e-30 floor makes LSE log(1e-30) =
        # -69.08 and inv_sum +inf, and O becomes (TMEM residue) * inf -- NaN,
        # since the residue can be a NaN bit pattern.  Zero O with a SELECT.
        _kv_empty = bounds.right <= bounds.left
        if cutlass.const_expr(not CFG.HAS_SINK):
            # A sink leaves real mass and the branch above already yields
            # LSE = sink_logit there; without one an empty row is -inf.
            lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))

        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg
        # ONE row bound for BOTH the LSE write and the amax atomic below.
        # They must stay tied: amax is an atomicMax, which only GROWS, so a
        # single padded row folded in permanently inflates the graph's Amax_O
        # for the whole tensor and no per-row check ever shows it.
        q_row_limit = seqlen_q
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: sequence-local row; LSE packed [1,QH,T] → [0, head, cu_q[b]+local].
            _cu = cutlass.make_array_view(seq_kv_lens_tensor)
            _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
            _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
            q_row_limit = _s_q_b
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < q_row_limit:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                    lse_row[_cu_q_b + q_row_global] = lse_val
        else:
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < q_row_limit:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_row = lse_arr[batch_idx, head_idx, :]
                    lse_row[q_row_global] = lse_val

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
        # P14 catch-up flip — bmm2_done_phase ^= 1 AFTER epilogue wait, BEFORE
        # next-tile sched advance.
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        # Block grouping = 64 elems (= 4 O_CHUNK loads) to preserve the
        # mb_o_full[chunk/2] firing schedule (one barrier per 2 epi blocks).
        O_EPI_BLK = 64
        N_BLOCKS_EPI = CFG.TILE_O // O_EPI_BLK
        CHUNKS_PER_BLK = O_EPI_BLK // O_CHUNK

        # amax_o over VALID rows of |o| (fp32, pre-cast).  atomicMax only grows,
        # so a padded row would permanently inflate the graph's Amax_O -- gate
        # the atomic exactly like the LSE write above.
        _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
        _amax_o_local = cutlass.Float32(0.0)

        sO_base = sO[0].base
        epi_o_full_block_idx = 0

        for block_idx in cutlass.range_constexpr(N_BLOCKS_EPI):
            for sub in cutlass.range_constexpr(CHUNKS_PER_BLK):
                chunk_idx_total = block_idx * CHUNKS_PER_BLK + sub
                o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx_total * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                # SELECT the zero for an empty KV range, then fold amax over the
                # substituted values -- an ungated NaN here would propagate into
                # the graph's Amax_O for every other row via atomicMax.
                # Statement-level loop: range_constexpr is not preprocessed
                # inside a comprehension.
                _o_elems = []
                for _i in cutlass.range_constexpr(O_CHUNK):
                    _e = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value()))
                    _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))
                    _o_elems.append(_e)
                o_out = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)

                col_offset_const = (chunk_idx_total * O_CHUNK) % D_BLOCK_SIZE
                block_offset_const = ((chunk_idx_total * O_CHUNK) // D_BLOCK_SIZE) * TMA_O_GRANU_ELEMS_LOCAL
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()

                if block_idx == 0 and sub == 0:
                    bars.mb_o_empty.wait(epilogue_state)
                smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # One mb_o_full per TMA-O store chunk.  BF16/FP16 O (BPE_O=2) doubles
            # N_O_CHUNKS so _BLOCKS_PER_OCHUNK drops 2→1 (fire every block);
            # keeps the TMA-STG consumer's N_O_CHUNKS waits balanced (FP8 unchanged).
            _BLOCKS_PER_OCHUNK = N_BLOCKS_EPI // N_O_CHUNKS
            fire_now = (block_idx + 1) % _BLOCKS_PER_OCHUNK == 0
            if cutlass.const_expr(fire_now):
                # fence_proxy needed before TMA reads SMEM written by swizzled store.
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_full[block_idx // _BLOCKS_PER_OCHUNK].arrive()

        if q_row_global < q_row_limit:
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))

        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(0))).load()
        nxt_hb = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(1))).load()
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        bounds = _bounds_for_tile(q_super_idx, seqlen_q, eff_seqlen_kv, cta_in_pair)

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
    # FROST's MXFP8 ABI puts amax_o RIGHT AFTER lse (position 9).
    amax_o_tensor: cute.Tensor,
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    # FROST passes the dense padded-Q trim tensor POSITIONALLY on every call.
    # These kernels have no Q-trim epilogue (Capabilities.dense_seq_q_trim=False)
    # so it is accepted and unused -- but the SLOT is mandatory: without it the
    # adapter's 12th positional lands on `stream` and execute dies with
    # "got multiple values for argument 'stream'".
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
    # Keyword-only in effect: FROST never passes these positionally (its SF
    # extents are compile-time), and they are read only on the THD path, which
    # this kernel declines.  They MUST sit after seq_q_lens_tensor: the adapter
    # passes seq_q_lens positionally for the d256 quantized ABI, so an SF total
    # in that slot would swallow it.
    total_q_sf_tiles: cutlass.Int32 = 0,
    total_kv_sf_tiles: cutlass.Int32 = 0,
    # FROST plans must run on the caller's stream (engine contract; there is a
    # dedicated stream-respect test).  Threaded exactly as the shipped
    # prefill_d128_fp8_sm107.py sibling does.
    stream: _cuda_driver.CUstream = None,
) -> None:
    """MXFP8 host launcher — builds TMA descriptors for Q/K/V/O + SF tensors.

    THD/varlen: q/k/v/o/lse/SF are PACKED with batch dim 1; the SF descriptors use
    B=1 + total_*_sf_tiles (per-sequence-TILE-padded SF layout); O uses a per-batch
    descriptor array (build_thd_meta_o_descs_kernel)."""
    B, QH, KH, SQ, SKV, _ = problem_size

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

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

    # SF TMA — 5-D explicit layout.  V_SF SMEM concatenates SF_NUM_BLOCKS_V
    # non-K blocks contiguously per tile (no padding between blocks at
    # TILE_N=128); the kernel's mma_ts_step nblk loop walks blocks via
    # tmem_SF_V_n = tmem_SF_V + nblk * SF_V_COLS_PER_NBLOCK.
    # Dense: per-batch tile counts (SQ/TILE_M, SKV/TILE_N), B-extent = B.  THD:
    # the SF tensor is PACKED (batch dim 1) and per-sequence-TILE-padded, so the
    # descriptor uses num_tiles = total_*_sf_tiles (Σ_b ceil(S/TILE)) and B = 1.
    # _B_SF folds to the const 1 under THD so the descriptor's batch extent is correct.
    sq_sf_tiles = (SQ + CFG.TILE_M - 1) // CFG.TILE_M
    skv_sf_tiles = (SKV + CFG.TILE_N - 1) // CFG.TILE_N
    _B_SF = 1 if cutlass.const_expr(CFG.THD_VARLEN) else B
    _q_sf_num_tiles = total_q_sf_tiles if cutlass.const_expr(CFG.THD_VARLEN) else sq_sf_tiles
    _kv_sf_num_tiles = total_kv_sf_tiles if cutlass.const_expr(CFG.THD_VARLEN) else skv_sf_tiles

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
    # V's SF is the COLUMNWISE (transposed-operand) quantization, and its GMEM
    # layout is NOT the per-tile-contiguous one Q/K use.  The F8_128x4 atom rule
    # is applied to the transposed scale matrix [D, S/32], so the atom grid is
    # (D/128) x (b*KH*S/128) laid out ROW-MAJOR -- the D-block index is the
    # OUTER one, and the SF_NUM_BLOCKS_V D-planes are separated by a whole plane
    # of `v_sf_groups` atoms, a stride that GROWS WITH S.  Q/K are rowwise:
    # their atom grid is (b*QH*S/128) x (D/128), so THEIR D-blocks are per-tile
    # contiguous and _build_sf_desc is right for them.
    #
    # At d=128 there is exactly ONE D-plane, so the two layouts coincide -- which
    # is why the d128 MXFP8 sibling passes with the per-tile form, and why d256
    # was correct at S=128 (one KV tile => one group) and wrong from S=256 on.
    # Mirrors prefill_d256_mxfp8_sm100.py's dense/THD stride split.
    _v_sf_groups = _B_SF * KH * _kv_sf_num_tiles
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD packs both D-planes of a (head, sequence tile) contiguously.
        _v_plane_stride_16 = SF_BYTES_PER_BLOCK // 16
        _v_tile_stride_16 = SF_SMEM_SIZE_V // 16
    else:
        _v_plane_stride_16 = (_v_sf_groups * SF_BYTES_PER_BLOCK) // 16
        _v_tile_stride_16 = SF_BYTES_PER_BLOCK // 16
    tma_v_sf_desc = tmap.create_tensor_map_tiled(
        global_address=cutlass.Int64(sf_v_tensor.iterator.toint()),
        dtype=cutlass.Uint8,
        global_dims=[
            SF_TMA_ROW_BYTES,
            SF_BYTES_PER_BLOCK // SF_TMA_ROW_BYTES,
            SF_NUM_BLOCKS_V,
            _kv_sf_num_tiles,
            KH * _B_SF,
        ],
        global_strides=[
            SF_TMA_ROW_BYTES // 16,
            _v_plane_stride_16,
            _v_tile_stride_16,
            _kv_sf_num_tiles * _v_tile_stride_16,
        ],
        box_dims=[SF_TMA_ROW_BYTES, SF_BYTES_PER_BLOCK // SF_TMA_ROW_BYTES, V_SF_PLANES_PER_PEER, 1, 1],
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the per-batch O descriptor array (reuse tma_o_desc over the
        # packed [1,T,QH,D_v] O as base), then launch the exact flat
        # batch-outermost grid (n_thd_units host-computed); grid_x = units*CGA_M.
        _build_thd_meta_o_descs_kernel(
            o_tensor,
            tma_o_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(CFG.TILE_O),
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (grid_q_supers, QH, B) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B, 1, 1)
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
    total_q_sf_tiles: int = 0,
    total_kv_sf_tiles: int = 0,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_stride: Optional[tuple] = None,
) -> Callable:
    """Compile a kernel with concrete dims.

    THD/varlen: q/k/v/o/lse + SF tensors are PACKED with batch dim 1; ``b`` is the
    LOGICAL batch (sequence count).  The SF tensors are per-sequence-TILE-padded so
    their packed tile extent is ``total_q_sf_tiles`` / ``total_kv_sf_tiles`` (=
    Σ_b ceil(S/TILE)); pass those (they vary with the cu_seqlens partition, hence
    part of the lru_cache key).  Default 0 → fall back to the dense per-batch tile
    counts so the dense path is unaffected."""
    # THD/varlen is NOT ported for this kernel yet.  The setup-kernel call site
    # below still speaks the pre-upstream 7-arg contract, while FROST's
    # thd_helpers.build_thd_meta_o_descs_kernel takes 14 args and a different
    # metadata layout (4B+4 with batch_remap + a claim counter, vs the 3B+2
    # here), and the scheduler decode differs to match.  Raise here rather than
    # let it fail as an arity error deep in the trace -- and so no engine row
    # can advertise thd=True for this kernel and appear to work.  The dense
    # path is unaffected: CFG.THD_VARLEN is 0 and every THD branch folds out.
    if CFG.THD_VARLEN:
        raise NotImplementedError(f"{__name__}: THD/varlen not ported to the FROST setup-kernel contract")
    # ---- FROST adapter ABI ------------------------------------------------
    # lower_dsl_prefill calls EVERY kernel with the full forward signature.
    # This body carries only what the port brought over, so anything
    # it cannot honor RAISES rather than being silently ignored: a raise here
    # means the engine's Capabilities row is lying, which is the failure we
    # want loud.  (Capabilities: lse_optional=False, no strided Stats.)
    if lse_stride is not None:
        raise NotImplementedError(f"{__name__}: strided Stats not ported (contiguous [B, H, S] only)")
    if d_qk > CFG.TILE_K or d_v > CFG.TILE_O or d_qk <= 0 or d_v <= 0:
        raise ValueError(f"{__name__}: envelope is 0 < d_qk <= {CFG.TILE_K}, 0 < d_v <= {CFG.TILE_O}; " f"got ({d_qk}, {d_v})")
    _fake_batch = 1 if CFG.THD_VARLEN else b

    # Q SF tiles TILE_M-row wide → num_tiles = SQ/TILE_M; K/V SF TILE_N-row wide → num_tiles = SKV/TILE_N.
    # THD: packed (batch dim 1) + per-sequence-TILE-padded → total_*_sf_tiles tiles.
    sq_tiles = (sq + CFG.TILE_M - 1) // CFG.TILE_M
    skv_tiles = (skv + CFG.TILE_N - 1) // CFG.TILE_N
    if CFG.THD_VARLEN:
        _q_sf_tiles = total_q_sf_tiles if total_q_sf_tiles > 0 else b * sq_tiles
        _kv_sf_tiles = total_kv_sf_tiles if total_kv_sf_tiles > 0 else b * skv_tiles
    else:
        _q_sf_tiles = sq_tiles
        _kv_sf_tiles = skv_tiles

    fake_q = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, sq, qh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_k = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, d_qk),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_compact_tensor(
        STORAGE_DTYPE,
        (_fake_batch, skv, kh, d_v),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    fake_o = cute.runtime.make_fake_compact_tensor(
        OUT_STORAGE_DTYPE,
        (_fake_batch, sq, qh, d_v),
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

    # has_lse=False (no Stats output): the LSE argument is None-specialized and
    # the store is compiled out entirely -- no dummy buffer exists at any level,
    # which is what lets the dense graph report get_workspace_size() == 0.
    # Mirrors the shipped prefill_d128_fp8_sm107.py.
    fake_lse = (
        cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (_fake_batch, qh, sq),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        if has_lse
        else None
    )
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )
    # seq_kv_lens always part of the ABI; THD overloads it as the
    # [seq_kv_lens(B)|cu_q(B+1)|cu_k(B+1)] metadata buffer (length 3B+2).
    _skv_len = (3 * b + 2) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Per-batch O TMA-descriptor array (TENSOR_MAP_QWORDS int64 = 128 B each) + 1
    # pad slot; dummy 1-elem when THD off (kernel never reads it).
    _odesc_len = (b * _TENSOR_MAP_QWORDS + _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )
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
        (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        # BY KEYWORD, not position: the SF totals now sit AFTER
        # seq_q_lens_tensor in _host's signature (the adapter passes that slot
        # positionally for the d256 quantized ABI).  Passing them positionally
        # here lands an Int32 in the tensor slot --
        # "expects argument #16 (seq_q_lens_tensor) ... but got Int32".
        total_q_sf_tiles=cutlass.Int32(_q_sf_tiles),
        total_kv_sf_tiles=cutlass.Int32(_kv_sf_tiles),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def _main():
    """Minimal CLI for compile-check / perf bring-up."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=1)
    parser.add_argument("--hk", type=int, default=1)
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=128)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    print(f"[d256_mxfp8] compile b={args.b} qh={args.hq} kh={args.hk} " f"sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[d256_mxfp8] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d256_mxfp8] --validate: see tests/run_python_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
