# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin SM107 DSv4 SDPA prefill — d_qk = d_v = 512, FP8 (E4M3 / E5M2).

THD / varlen (``CFG.THD_VARLEN=1``) is supported — FP8 is element-addressed (no
block-scale SF), so it rides the same THD path as f16: packed ``[1,T,H,D]`` +
``cu_seqlens`` coord offset, per-batch O TMA-descriptor array (shared
the shared pre-upstream THD helper), packed ``[1,QH,T]`` LSE.  Dense path
byte-identical.  Public-API fp8 THD glue is a follow-up (THD-aware quantizer);
drive via the kernel module directly.

Python port (from the pre-upstream DSL) of the C++ Rubin baseline
``kernels/c++/sm107/sdpa/prefill/prefill_sdpa_d512_fp8.cu`` — same kernel,
different language.  Pipeline shape, TMEM/SMEM layout, barrier inventory,
scheduler, warp dispatch, and register split mirror the C++ source 1:1.

Pipeline shape (cluster (4, 1, 1) = two cga2 sub-groups):
  sg0 (CTAs 0, 1):  TMA-LDG Q + K_ring; BMM1 = Q*K^T → S_acc; streaming
                    softmax; ship alpha + P + stats to sg1 via DSMEM.
  sg1 (CTAs 2, 3):  TMA-LDG V_ring; recv alpha + P + stats via DSMEM;
                    correction (apply alpha to O TMEM); BMM2 = P*V^T → O;
                    epilogue normalize + fp32→fp8/half cast + TMA-STG O + LSE.

FP8 deltas vs d512_f16 (CFG fields DTYPE_QKV-driven via sdpa_config_dsv4.py):
  - STAGES_KV = 3, XFER_STAGES = 4 (BPE=1 fits more stages in 327 KiB).
  - TILE_K_HW_BMM1/2 ∈ {32, 64}; MMA_KIND = F8F6F4.
  - P stays FP8-typed (P_STORAGE_DTYPE).
  - BMM2_V_NBLOCK_ADVANCE = TILE_N * V_SWZ_BYTES (NO BPE factor at FP8 —
    per-CTA V inner = (TILE_O/CTA_MMA)*BPE = 256*1 = 256 B = 2 swz lines).
  - DTYPE_O independent of DTYPE_QKV: E4M3 / E5M2 / BF16 / FP16 (BPE_O ∈ {1, 2}).
  - Epilogue cast loop must handle BPE_O ∈ {1, 2}:
      epilogue_block_size = 64 / sizeof(ElementO)  (fp8: 64, half: 32)
      CHUNK_ELEMS         = 128 / sizeof(ElementO) (fp8: 128, half: 64)
      BLOCKS_PER_SUBTILE  = 128 / epilogue_block_size (fp8: 2, half: 4)
  - 2-bit sub-tile permutation [0,2,1,3] unchanged — formula stays generic.

SM107 deltas vs SM100:
  - No Q∪K_ring alias — SMEM cap is 327 KiB, Q stays resident.
  - No UTCCP(Q→TMEM) — Q feeds BMM1 directly from SMEM.
  - No B1 / B2 / B3 alias-seam mbars (those are SM100-only).
  - TMEM cap is 576 cols (is_exclusive=True on tcgen05_alloc).

P12 / P13 patterns (cf. mbarrier-patterns.md, also project memory):
  - mb_p_xfer_full (P12): sg1 leader's collective BMM2 reads BOTH peers' P
    SMEM via crossbar.  Each sg1 CTA's mbar gets bytes from its own cross-sg
    sg0 partner; non-leader sg1 forwards its `arrive` to the leader via
    DSMEM so leader's wait flips only after the full pair has landed.
    Leader init = CTA_MMA (own + DSMEM-forwarded); non-leader init = 1.
  - mb_p_xfer_empty (P13): sg1 leader MMA fires a single
    tcgen05.commit.cta_group::2.multicast with sg0_mcast_mask (= 0x3) so
    BOTH sg0 CTAs see one arrive — pre-armed via PipelineState(0, 1).

DSL-only adjustments applied (per the C++-to-DSL porting notes):
  - is_exclusive=True on tmem_alloc (SM107 >512-col TMEM).
  - tile_id_smem stride 8 Int32/stage.
  - cga_arrive/wait wrapped with relaxed=True (per project memory:
    dsv3 cga2 47%→92% SOL with this fix).
  - DSMEM SMEM→peer-SMEM via cp_async_bulk_shared_cluster_shared_cta
    (per project memory: bare nvvm op silently fires sender's local mbar).
  - RegTile(Float8E4M3FN) is illegal at JIT — FP8 P_cast skips the RegTile
    wrap; pass raw vec to store_swizzled / slice via vec_slice on Float32.
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from typing import NamedTuple


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental.cuda import tensor_map as tmap
from cudnn.frost.tile_dsl.tma import cp_async_bulk_shared_cluster_shared_cta
import cuda.bindings.driver as _cuda_driver  # noqa: F401  (cute.compile pulls cuda)

# DSv4 has only one config flavor (dsv4); no env-var flavor switch needed.
# Config comes from the FROST template loader, NOT an env var: the pre-upstream
# kernels picked a flavor with an env var + a sdpa_config_<flavor> module, which the
# FROST engine contract forbids ("no environment variables for configuration" --
# parameters travel as typed dataclasses). The loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a plain
# `import` usable as a standalone driver.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d512

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d512(PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE
# decision point, wired into every construction below rather than repeated as a
# per-tile literal.  A version-0 descriptor's ``start_address`` is 14 bits = a
# 256 KiB window; Rubin raises the per-CTA SMEM cap to 327 KiB, and THIS flavor
# crosses the line (the d512 slabs put the P transfer ring at exactly 262144),
# so a version-0 descriptor would wrap to offset 0 and the MMA would multiply
# whatever sits at the bottom of SMEM -- the accumulator comes out EXACTLY
# zero, or the SF columns come back as data bytes (LSE = +inf, O = NaN).  No
# crash either way.  Matches what C++ SmemTile::make_desc always emitted.
#
# Do NOT re-literal this at a call site: this kernel's MXFP8 sibling shipped
# NaN on 100% of cells because its scale-factor tiles were declared UNDER a
# comment claiming "every operand tile here carries desc_version=1" -- without
# the kwarg.  A single constant makes that class of drift impossible, and
# test_sm107_descriptor_version_matches_the_smem_budget asserts it.
DESC_VERSION: int = 1
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# Rubin's dense-FP8 MMA runs K=64 per instruction, and every
# ``Tcgen05InstrDesc.build`` site in this body hardcodes the matching idesc
# ``k_dim=1``.  ``config_sm107.tile_k_hw()`` derives the 64; this guard is the
# tripwire for the pairing, which is arch-OPPOSITE (Blackwell wants k_dim=0 with
# TILE_K_HW=32) and fails SILENTLY -- a mismatch scrambles rows of the
# accumulator rather than raising (rules/mma-tma-matrix.md S1).
if CFG.TILE_K_HW_BMM1 != 64 or CFG.TILE_K_HW_BMM2 != 64:
    raise ValueError(f"{__name__}: this body's idesc k_dim=1 requires TILE_K_HW=64 on Rubin; " f"got BMM1={CFG.TILE_K_HW_BMM1} BMM2={CFG.TILE_K_HW_BMM2}")

# O TMA box / store params follow O's swizzle (independent of V under cga2).
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES


# Generic tile primitives — never hand-roll equivalents.
from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    MBarrier,
    Producer,
    Scope,
    # `wait` (free fn) — still used for sched.mb_* (Sched not in Bars).
    wait,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    tmem_load_max_reduction_tile,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
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

# Reuse sm107 SDPA pipeline-shared helpers (KvLoopBounds + closures factory).
# Note: dsv4 forks Bars in this file (NOT imported from _sdpa_common) per the
# per-pipeline fork pattern in the C++-to-DSL porting notes.
from cudnn.sdpa.fwd.kernels._common_blackwell import (
    KvLoopBounds,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
)

# ----------------------------------------------------------------------------
# Storage dtype dispatch — folded at trace time on CFG.DTYPE_QKV.
# FP8 (E4M3 / E5M2) only — BF16/FP16 ships in prefill_sdpa_d512_f16.py.
# ----------------------------------------------------------------------------
if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
else:
    raise ValueError(
        f"prefill_sdpa_d512_fp8 (SM107): DTYPE_QKV={CFG.DTYPE_QKV} not supported " f"(expected 0=E4M3 or 1=E5M2; BF16/FP16 ship in prefill_sdpa_d512_f16.py)"
    )

# DTYPE_O is independent of DTYPE_QKV (C++ Cfg::DTYPE_O — defaults to DTYPE_QKV
# but can be promoted to BF16/FP16 to skip downstream dequant).  Dispatch on
# Cfg::DTYPE_O.  BPE_O ∈ {1, 2}; epilogue cast loop generic over the result.
if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_d512_fp8 (SM107): DTYPE_O={CFG.DTYPE_O} not supported " f"(expected 0=E4M3/1=E5M2/2=BF16/3=FP16)")


# ----------------------------------------------------------------------------
# Derived constants — mirror prefill_sdpa_d512_fp8.cu:120-140.
# ----------------------------------------------------------------------------
CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

# Per-CTA buffer element counts + collective TMA transaction byte counts.
qBufferElems = CFG.TILE_M * CFG.TILE_K  # 65536  → 128 KiB @ BPE=2
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA  # 32768  →  64 KiB
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA  # 32768  →  64 KiB
oBufferElems = CFG.TILE_M * CFG.TILE_O  # 65536  → 128 KiB @ BPE_O=2
pXferElems = CFG.TILE_M * CFG.TILE_N  # 16384  →  32 KiB / xfer stage

# Per-stage xfer SMEM sizes (DSMEM peer-to-peer payloads).
pXferBytes = pXferElems * CFG.BPE  # 32 KiB / stage
alphaXferBytes = CFG.TILE_M * 4  # 512 B  / stage (FP32)
statsXferBytes = 2 * CFG.TILE_M * 4  # 1 KiB total (ell + max)

# Collective TMA transaction bytes (leader expect_tx under cga2).
qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

# Per-call BMM2 V SMEM advance (between the BMM2_LOOP_N_BLOCKS=2 BMM2 calls).
# C++ prefill_sdpa_d512_fp8.cu:87 — at FP8 (BPE=1) the per-CTA V inner =
# (TILE_O/CTA_MMA)*BPE = 256 B = 2 swizzle lines, so the slab stride works
# out to TILE_N * V_SWZ_BYTES bytes per N-block (NO extra BPE factor).
BMM2_V_NBLOCK_ADVANCE = CFG.TILE_N * CFG.V_SWZ_BYTES

# O store chunks (TMA-STG arrive granularity — 128 B per chunk).
N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128  # FP8 out: 4; half out: 8

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

# ----------------------------------------------------------------------------
# Softmax-body constants — module-level so the top-level @cute.jit helper
# `_sg0_softmax_kv_iter` can read them without a closure capture (pre-upstream DSL
# forbids closures inside @cute.jit traced under dynamic control flow).
# ----------------------------------------------------------------------------
# P SMEM layout helpers (TILE_M=128 rows × TILE_N=128 cols, Swz128B).
# 1 row of TILE_N elements = (TILE_N*BPE/SWZ_BYTES) swizzle chunks of P_D_BLOCK each.
P_TMA_ITERS = (CFG.TILE_N * CFG.BPE) // CFG.Q_SWZ_BYTES  # 2
P_D_BLOCK = CFG.TILE_N // P_TMA_ITERS  # 64 elements / chunk
P_BLOCK_BYTES = CFG.TILE_M * P_D_BLOCK  # 8192 elements / TMA chunk
# Mask-path load chunk width (mirrors canonical _softmax_kv_body CHUNK=64).
SOFTMAX_CHUNK = 64
SOFTMAX_N_CHUNKS_LOAD = CFG.TILE_N // SOFTMAX_CHUNK  # 2 for TILE_N=128
# Swizzle for P SMEM stores (Swz128B at d=512 — same as Q/O).
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
# Streaming-softmax constants.
NEG_INF_F32 = cutlass.Float32(-3.4028235e38)
RESCALE_THRESHOLD_F32 = cutlass.Float32(CFG.RESCALE_THRESHOLD)


# ----------------------------------------------------------------------------
# Named arrival-count constants — mirror C++ prefill_sdpa_d512_fp8.cu:1500-1520.
# Per mbarrier-patterns.md P3, ALL init(...) counts must reference named
# constants — never inline literals.
# ----------------------------------------------------------------------------
COMPUTE_LANES = CFG.SOFTMAX_WG_WARPS * 32  # 4 warps * 32 lanes = 128
SM_LANES_TOTAL = 2 * COMPUTE_LANES  # 256 — both sg0 CTAs * 128 sm threads
TWO_LANES_TOTAL = 2  # both sg1 CTAs * 1 elect-one to leader
KV_EMPTY_ARRIVERS = CFG.CGA_N  # = 1 — cga2 commit multicast counts as 1

# READ_TILE_ARRIVERS already derived in CFG (= 25 for dsv4 cga4x1).  Re-export
# for inline visibility next to the init block.
READ_TILE_ARRIVERS = CFG.READ_TILE_ARRIVERS


# ----------------------------------------------------------------------------
# Compile-time budget checks — Rubin SM107 caps (NOT 227/512 SM100 caps).
# ----------------------------------------------------------------------------
# Per-CTA SMEM budget under Rubin SM107 327 KiB oversized cap.  Each physical
# CTA runs ONE role (sg0 or sg1), so the SharedStorage union covers
# max(sg0_bytes, sg1_bytes), NOT the sum.  sg0 holds Q + K_ring + xfer
# staging; sg1 holds V_ring + O + xfer staging + LSE.  Mirrors the C++
# `union { sg0; sg1; }` in SharedStorage.
_SG0_SMEM_DATA_KIB = (
    qBufferElems * CFG.BPE  # Q (sg0)
    + CFG.STAGES_KV * kBufferElems * CFG.BPE  # K_ring (sg0)
    + CFG.XFER_STAGES * pXferBytes  # P xfer (source side staging)
    + CFG.XFER_STAGES * alphaXferBytes  # alpha xfer (source side staging)
    + statsXferBytes  # stats xfer
) / 1024
_SG1_SMEM_DATA_KIB = (
    CFG.STAGES_KV * vBufferElems * CFG.BPE  # V_ring (sg1)
    + oBufferElems * CFG.BPE_O  # O (sg1)
    + CFG.XFER_STAGES * pXferBytes  # P xfer (sink side staging)
    + CFG.XFER_STAGES * alphaXferBytes  # alpha xfer (sink side staging)
    + statsXferBytes  # stats xfer
    + CFG.TILE_M * 4  # LSE staging (sg1)
) / 1024
# RESOURCE-NOTE: SM107 oversized cap is 327 KiB; allow ~4 KiB scaffolding +
# mbar headroom.  Per-sg data budgets must each fit under 323 KiB.
assert _SG0_SMEM_DATA_KIB <= 323, f"sg0 SMEM data {_SG0_SMEM_DATA_KIB:.1f} KiB exceeds 323 KiB headroom under 327 KiB cap"
assert _SG1_SMEM_DATA_KIB <= 323, f"sg1 SMEM data {_SG1_SMEM_DATA_KIB:.1f} KiB exceeds 323 KiB headroom under 327 KiB cap"

# TMEM 576-col cap — Rubin SM107 (NOT SM100's 512).  is_exclusive=True required.
_TMEM_SG0_COLS = CFG.XFER_STAGES * CFG.TILE_N  # FP8: 4 * 128 = 512 cols for S_acc parities
_TMEM_SG1_COLS = CFG.TILE_O  # 512 cols for O
assert _TMEM_SG0_COLS <= 576, f"sg0 S_acc TMEM overflow: {_TMEM_SG0_COLS} > 576"
assert _TMEM_SG1_COLS <= 576, f"sg1 O TMEM overflow: {_TMEM_SG1_COLS} > 576"


# ----------------------------------------------------------------------------
# Bars — DSv4 role-split inventory, forked in this kernel file per the
# per-pipeline fork pattern.  Mirrors C++ prefill_sdpa_d512_fp8.cu:320-373.
# ----------------------------------------------------------------------------
# Barrier table (cga4x1 / CTA_MMA=2 — DSv4 role-split, SM107 baseline)
# ============================================================================
# | Name                     | Stages           | Init                       | Producer / arrive site                          | Consumer / wait site                  | Scope / bootstrap                                   |
# |--------------------------|------------------|----------------------------|-------------------------------------------------|---------------------------------------|-----------------------------------------------------|
# | mb_tma_q_full            | 1                | ONE_LANE                   | sg0 TMA-LDG : arrive_expect_tx (Q bytes)        | sg0 MMA (BMM1)                        | cga2 leader-only                                    |
# | mb_tma_q_empty           | 1                | ONE_LANE                   | sg0 MMA : arrive_mma after last-iter BMM1 commit| sg0 TMA-LDG (next-tile Q load)        | leader-multicast; pre-arm via q_empty_state=1       |
# | mb_tma_k_full[s]         | STAGES_KV (=2)   | ONE_LANE                   | sg0 TMA-LDG : arrive_expect_tx (K bytes)        | sg0 MMA (BMM1)                        | cga2 leader-only                                    |
# | mb_tma_k_empty[s]        | STAGES_KV        | KV_EMPTY_ARRIVERS (=1)     | sg0 MMA : arrive_mma after BMM1 commit          | sg0 TMA-LDG (K_ring back-pressure)    | leader-multicast; pre-arm via PipelineState(0,1)    |
# | mb_tma_v_full[s]         | STAGES_KV        | ONE_LANE                   | sg1 TMA-LDG : arrive_expect_tx (V bytes)        | sg1 MMA (BMM2)                        | cga2 leader-only                                    |
# | mb_tma_v_empty[s]        | STAGES_KV        | KV_EMPTY_ARRIVERS (=1)     | sg1 MMA : arrive_mma after BMM2 commit          | sg1 TMA-LDG (V_ring back-pressure)    | leader-multicast; pre-arm via PipelineState(0,1)    |
# | mb_bmm1_done[p]          | XFER_STAGES (=2) | ONE_LANE                   | sg0 MMA : arrive_mma after BMM1 commit          | sg0 softmax                           | local; not bootstrapped                             |
# | mb_bmm2_done[p]          | XFER_STAGES      | ONE_LANE                   | sg1 MMA : arrive_mma after BMM2 commit          | sg1 corr (epilogue gate)              | local; not bootstrapped                             |
# | mb_bmm2_ready[p*2+c]     | XFER_STAGES*2(=4)| SM_LANES_TOTAL (=256)      | sg1 corr (all-thread) : arrive_on_peer→leader   | sg1 MMA leader (per-half-N-block)     | cga2 leader-waited                                  |
# | mb_s_acc_empty[p]        | XFER_STAGES      | SM_LANES_TOTAL (=256)      | sg0 softmax (all-thread) : arrive_on_peer→leader| sg0 MMA leader (S_acc TMEM free)      | cga2 leader-waited; pre-arm via PipelineState(0,1)  |
# | mb_p_xfer_full[p]        | XFER_STAGES      | leader: CTA_MMA(=2)        | sg0 softmax bulk_copy→peer; sg1 peer→leader DSMEM| sg1 MMA leader (BMM2 P operand)       | P12; leader=2 / non-leader=1                        |
# |                          |                  | non-leader: ONE_LANE(=1)   |                                                  |                                       |                                                     |
# | mb_p_xfer_empty[p]       | XFER_STAGES      | ONE_LANE                   | sg1 MMA leader : arrive_mma multicast→sg0       | sg0 softmax (next P slot reuse)       | P13 MMA-commit multicast; pre-arm via PipelineState(0,1) |
# | mb_alpha_xfer_full[p]    | XFER_STAGES      | ONE_LANE                   | sg1 corr (local) : arrive_expect_tx (alpha)     | sg1 corr (apply alpha)                | local; not bootstrapped                             |
# | mb_alpha_xfer_empty[p]   | XFER_STAGES      | COMPUTE_LANES (=128)       | sg1 corr (all-thread) : arrive_on_peer→sg0      | sg0 softmax (next alpha slot reuse)   | local; pre-arm via PipelineState(0,1)               |
# | mb_stats_xfer_full       | 1                | ONE_LANE                   | sg1 corr (local) : arrive_expect_tx (ell+max)   | sg1 corr (final stats consume)        | local; not bootstrapped                             |
# | mb_stats_xfer_empty      | 1                | ONE_LANE                   | sg1 corr (elect) : arrive_on_peer→sg0           | sg0 softmax (next tile stats slot)    | local; pre-arm via stats_xfer_empty_phase=1         |
# | mb_tma_o_full[c]         | N_O_CHUNKS (=8)  | COMPUTE_LANES (=128)       | sg1 corr (all-thread) : arrive per chunk        | sg1 TMA-STG (per-chunk TMA)           | local; not bootstrapped                             |
# | mb_tma_o_empty           | 1                | ONE_WARP (=32)             | sg1 TMA-STG warp : arrive after STG drain       | sg1 corr (next-tile O staging reuse)  | local; pre-arm via PipelineState(0,1)               |
# | mb_empty_mainloop        | 1                | TWO_LANES_TOTAL (=2)       | sg1 corr (both CTAs, elect) : arrive_on_peer→leader| sg1 MMA leader (empty-kv branch)   | cga2 leader-waited                                  |
# | mb_tmem_dealloc          | 1                | ONE_LANE                   | both sgs compute (elect) : arrive_on_peer→partner| both sgs MMA (alloc/dealloc fence)   | local; not bootstrapped                             |
# ============================================================================
class Bars(NamedTuple):
    """DSv4 role-split mbarrier inventory (Rubin SM107 baseline)."""

    # ---- TMA Q/K/V handshakes (sg0: Q + K; sg1: V) ----------------------
    mb_tma_q_full: object  # [1]            sg0  TMA-LDG → MMA
    mb_tma_q_empty: object  # [1]            sg0  MMA → TMA-LDG (next-tile Q)
    mb_tma_k_full: object  # [STAGES_KV]    sg0  TMA-LDG → MMA
    mb_tma_k_empty: object  # [STAGES_KV]    sg0  MMA → TMA-LDG (back-pressure)
    mb_tma_v_full: object  # [STAGES_KV]    sg1  TMA-LDG → MMA
    mb_tma_v_empty: object  # [STAGES_KV]    sg1  MMA → TMA-LDG

    # ---- MMA → softmax / corr handshakes -------------------------------
    mb_bmm1_done: object  # [XFER_STAGES]      sg0 MMA → softmax
    mb_bmm2_done: object  # [XFER_STAGES]      sg1 MMA → corr (epilogue gate)
    mb_bmm2_ready: object  # [XFER_STAGES * 2]  sg1 corr → MMA (per half-N-block)
    mb_s_acc_empty: object  # [XFER_STAGES]      sg0 softmax → MMA (S_acc TMEM free)

    # ---- DSMEM xfer rings (sg0 → sg1) ----------------------------------
    mb_p_xfer_full: object  # [XFER_STAGES]   sg0 → sg1 (P12 leader=2 / non-leader=1)
    mb_p_xfer_empty: object  # [XFER_STAGES]   sg1 leader MMA → sg0 (P13 multicast)
    mb_alpha_xfer_full: object  # [XFER_STAGES]   sg0 softmax (local expect_tx) → sg1 corr
    mb_alpha_xfer_empty: object  # [XFER_STAGES]   sg1 corr (all-thread DSMEM) → sg0
    mb_stats_xfer_full: object  # [1]             sg1 corr (local expect_tx) one-shot per tile
    mb_stats_xfer_empty: object  # [1]             sg0 softmax (elect DSMEM) → sg1

    # ---- sg1 TMA-STG handshakes -----------------------------------------
    mb_tma_o_full: object  # [N_O_CHUNKS]   sg1 corr → TMA-STG (per-128B chunk)
    mb_tma_o_empty: object  # [1]            sg1 TMA-STG warp → corr

    # ---- End-of-kernel teardown -----------------------------------------
    mb_empty_mainloop: object  # [1]            sg1 corr → sg1 MMA leader (empty-kv branch)
    mb_tmem_dealloc: object  # [1]            both sgs compute → cga2 partner MMA


# ----------------------------------------------------------------------------
# KernelTmemLayout — sg-conditional carves total ≤ 576 cols on SM107.
# Mirrors C++ prefill_sdpa_d512_fp8.cu KernelTmemLayout (XFER_STAGES=4).
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelTmemLayout:
    """SM107 TMEM carves: sg0 = S_acc(XFER_STAGES parities); sg1 = O.

    sg0 uses TMEM cols [0, XFER_STAGES * 128) for the parity S_acc slots.
        FP8 (XFER_STAGES=4) → [0, 512) = full S_acc area.
    sg1 uses TMEM cols [0, 512) for the O accumulator (full d_v = 512 FP32 cols).
    Both fit under the Rubin 576-col cap.

    Use S_acc_at(parity) to pick the runtime parity slot (do NOT introduce
    per-slot constexpr names; the formula scales over XFER_STAGES generically).
    """

    # Rubin SM107 cap; requires is_exclusive=True on tcgen05_alloc.
    TOTAL_COLS: int = 576

    # sg0 layout — per-parity S_acc slot at offset parity * S_ACC_COLS.
    S_ACC_COLS: int = 128  # = CFG.TILE_N — one parity slot
    # Per-parity offsets (formula = parity * S_ACC_COLS).
    S_ACC_PARITY0_OFF: int = 0
    S_ACC_PARITY1_OFF: int = 128

    # sg1 layout — single fixed-offset O accumulator @ cols [0, TILE_O).
    O_OFF: int = 0
    O_COLS: int = 512  # = CFG.TILE_O


LAYOUT = KernelTmemLayout()
assert CFG.XFER_STAGES * LAYOUT.S_ACC_COLS <= LAYOUT.TOTAL_COLS, f"sg0 TMEM overflow: {CFG.XFER_STAGES * LAYOUT.S_ACC_COLS} > {LAYOUT.TOTAL_COLS}"
assert LAYOUT.O_OFF + LAYOUT.O_COLS <= LAYOUT.TOTAL_COLS, f"sg1 TMEM overflow: {LAYOUT.O_OFF + LAYOUT.O_COLS} > {LAYOUT.TOTAL_COLS}"


# ----------------------------------------------------------------------------
# Bars factory — dsv4 role-split mbarrier inventory typed with MBarrier
# wrapper.  Producer / scope tags + init counts derived from the barrier
# table comment above + the explicit init logic below.  Note: mb_p_xfer_full
# uses a RUNTIME-conditional init count (leader = CTA_MMA, non-leader = 1),
# so the static init_count here is a placeholder — the kernel passes an
# explicit override_count to .init() per CTA.
# ----------------------------------------------------------------------------
def _make_dsv4_bars(CFG, N_O_CHUNKS: int):
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        # TMA Q/K/V
        mb_tma_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_q_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tma_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tma_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tma_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        # MMA → softmax / corr
        mb_bmm1_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS),
            stages=CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS,
            init_count=SM_LANES_TOTAL,
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_s_acc_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=SM_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        # DSMEM xfer rings — mb_p_xfer_full has RUNTIME init count (P12):
        # leader = CTA_MMA arrives (own + DSMEM forwarder); non-leader = 1.
        # Static init_count here is CFG.CTA_MMA (the leader value); the
        # kernel's init loop passes override_count=p_full_init for each stage.
        mb_p_xfer_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.CTA_MMA, producer=Producer.TMA_LOAD),
        mb_p_xfer_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_alpha_xfer_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_alpha_xfer_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_stats_xfer_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_stats_xfer_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        # sg1 TMA-STG
        mb_tma_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=COMPUTE_LANES, producer=Producer.THREAD),
        mb_tma_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        # Teardown
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=TWO_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
    )


# ----------------------------------------------------------------------------
# SMEM swizzle / layout enums.
# ----------------------------------------------------------------------------
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q
# P xfer SMEM layout matches Q swizzle (cf. C++ prefill_sdpa_d512_fp8.cu:60).
SMEM_LAYOUT_P = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

# BMM2 manual k-step iters.
NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2


# ----------------------------------------------------------------------------
# Helpers bound to CFG (scheduler topology + mask plumbing — unchanged from
# the sm107 classic kernels; role-split doesn't change tile decoding).
# ----------------------------------------------------------------------------
_sdpa_h = make_sdpa_helpers(CFG)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
_bounds_for_tile = _sdpa_h.bounds_for_tile
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
# Q-side per-sequence length.  Under THD `seqlen_q` is the PACKED TOTAL, so
# every BOTTOM_RIGHT diagonal needs the sequence's own S_q_b instead.
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — flat-grid decode + tma-offset closures (CFG-bound) from the
# factory; O-descriptor builder + TENSOR_MAP_QWORDS from the shared
# the shared pre-upstream THD helper.  Gated by CFG.THD_VARLEN (folds out otherwise).
# FP8 is element-addressed (per-tensor dequant scalars, no block-scale SF), so
# THD rides the same path as f16.  seq_kv_lens overloaded as the THD metadata
# buffer (int32 len 3B+2): [0..B-1]=seq_kv_lens [B..2B]=cu_q [2B+1..3B+1]=cu_k.
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets


# ============================================================================
# Kernel entry.
# ============================================================================
@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
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
    o_scale_fused: cutlass.Float32,
    # FROST's quantized ABI: the four 1-element fp32 scales arrive as DEVICE
    # TENSORS and the amax of O is an OUTPUT.  The pre-upstream contract pre-folded on the host
    # and produced no amax.  Template: sm107/prefill_d128_fp8.py (same lineage).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # Device-scale fold (same as the shipped d128 sibling): every thread reads
    # the four 1-element fp32 scales -- identical addresses, so L2 broadcast --
    # and folds them into the softmax and output scales.
    _dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
    _dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
    _dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
    _scl_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
    scale_softmax_log2 = scale_softmax_log2 * _dsc_q * _dsc_k
    o_scale_fused = o_scale_fused * _dsc_v * _scl_o

    # ------------------------------------------------------------------
    # SMEM allocations — natural Q / K / V / O order, NO Q∪K_ring or
    # O∪V_ring alias (Rubin SM107 cap fits everything; aliases are
    # SM100-only).  Mirror C++ SharedStorage at prefill_sdpa_d512_fp8.cu:148.
    # ------------------------------------------------------------------
    # RESOURCE-NOTE: each physical CTA runs ONE role (sg0 OR sg1), so the
    # Q (sg0) ∪ O (sg1) and K_ring (sg0) ∪ V_ring (sg1) regions alias at
    # the same SMEM offset.  Mirrors C++ `union QOAlias`/`union KVAlias` at
    # prefill_sdpa_d512_fp8.cu:149-161 — without these unions the per-CTA
    # SMEM budget doubles (~580 KiB) and busts the 327 KiB Rubin cap.
    # Q∪O alias byte-sized to max(Q@BPE, O@BPE_O) — STORAGE_DTYPE is FP8 (1 B)
    # so element count == byte count.  BF16/FP16 O (BPE_O=2) would overflow a
    # qBufferElems-only allocation; size for the larger of the two views.
    _QO_ELEMS = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)
    _KV_ELEMS = CFG.STAGES_KV * kBufferElems if kBufferElems >= vBufferElems else CFG.STAGES_KV * vBufferElems
    sQO_raw = cutlass.Array(STORAGE_DTYPE, _QO_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, _KV_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # sQ / sO / sK / sV alias into sQO_raw / sKV_raw respectively.  sO's view
    # reinterprets the backing as OUT_STORAGE_DTYPE so epilogue store offsets
    # advance in BPE_O units (no-op recast for FP8 output).
    sQ_raw = sQO_raw  # sg0 view
    sO_raw = cutlass.Array(sQO_raw.data_ptr(), shape=oBufferElems, dtype=OUT_STORAGE_DTYPE)  # sg1 view
    sK_raw = sKV_raw  # sg0 view
    sV_raw = sKV_raw  # sg1 view

    # P xfer (sg0 → sg1) SMEM — XFER_STAGES rings, kept resident.
    sP_xfer_raw = cutlass.Array(P_STORAGE_DTYPE, CFG.XFER_STAGES * pXferElems, alignment=128, space=cutlass.AddressSpace.smem)
    # alpha xfer (sg0 → sg1) SMEM.
    sAlpha_xfer_raw = cutlass.Array(cutlass.Float32, CFG.XFER_STAGES * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    # stats xfer (sg0 → sg1) SMEM — 1-shot per tile, holds ell + max (2 * TILE_M floats).
    sStats_xfer_raw = cutlass.Array(cutlass.Float32, 2 * CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)
    # LSE staging on sg1 (sub-tile write-back from corr; cf. C++ line 198).
    sLSE_raw = cutlass.Array(cutlass.Float32, CFG.TILE_M, alignment=128, space=cutlass.AddressSpace.smem)

    # Typed SmemTile handles.  EVERY tile takes the module-level DESC_VERSION
    # (= 1 here) -- see its declaration for why that is one constant and not a
    # per-tile literal.  The short version: Rubin raises the per-CTA SMEM cap
    # to 327 KiB, so a buffer can land past the 256 KiB a version-0 tcgen05
    # descriptor can address (14-bit start_address), and the tiles that cross
    # the line are not the ones you would guess.
    sQ = SmemTile(
        base=sQ_raw,
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

    # ------------------------------------------------------------------
    # Bars allocation — per the barrier table above.
    # ------------------------------------------------------------------
    bars = _make_dsv4_bars(CFG, N_O_CHUNKS)

    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # tile_id_smem stride 8 Int32/stage (32 B per stage; 16 B payload + 16 B padding).
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

    # ------------------------------------------------------------------
    # cga2 / role-split identity — hoisted ABOVE init because mb_p_xfer_full
    # init count is leader-vs-non-leader runtime-conditional (P12 — leader
    # sees CTA_MMA arrives, non-leader sees 1).
    # ------------------------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    # sg0_mcast_mask = 0x3 — both sg0 CTAs (used by sg1 leader's P13 multicast on mb_p_xfer_empty).
    sg0_mcast_mask = cutlass.Int32(0x3)
    # TMA-load self-bit mask: per cga2-mma.md and a reference fp16 GEMM sample, each
    # peer's `cta_group::2` TMA targets ITS OWN cluster bit (= cta_id_x);
    # cga2 routing strips bit-24 so bytes land on the cga2 pair leader's
    # mbar.  Use cluster-rank (cta_id_x), NOT pair-rank (cta_in_pair) — for
    # cga2 (CGA_M=CTA_MMA=2) these coincide, but for cga4×1 (dsv4) the sg1
    # pair has cta_in_pair ∈ {0,1} while cta_id_x ∈ {2,3}, and using
    # cta_in_pair would target CTAs 0,1 (sg0) instead of CTAs 2,3 (sg1).
    tma_mcast_mask = (cutlass.Int16(1) << cta_id_x.to(cutlass.Int16)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Sub-group identity (sg0 = CTAs 0, 1; sg1 = CTAs 2, 3).
    sg_id = cta_id_x // cutlass.Int32(CFG.CTA_MMA)
    is_sg0 = sg_id == cutlass.Int32(0)
    is_sg1 = sg_id == cutlass.Int32(1)
    # Cross-sg peer = this CTA XOR CTA_MMA (CTAs 0↔2, 1↔3).
    cross_sg_peer = cta_id_x ^ cutlass.Int32(CFG.CTA_MMA)

    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # ------------------------------------------------------------------
    # Mbar init — Phase 1 of cluster init (P4).  No Phase-2 bootstrap arrives:
    # every consumer that would need one is pre-armed via PipelineState(0, 1)
    # or per-warp `phase = 1` initial values.  MmaProducer<> typing forbids
    # plain arrive() anyway.
    #
    # Per-CTA leader / non-leader split for mb_p_xfer_full follows P12:
    # leader = CTA_MMA arrives (own + (CTA_MMA-1) DSMEM forwarder); non-leader = 1.
    # ------------------------------------------------------------------
    if warp_idx == 0:
        if nvvm.elect_sync():
            # ---- TMA Q/K/V handshakes ----
            bars.mb_tma_q_full.init()
            bars.mb_tma_q_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_full[ks].init()
                bars.mb_tma_k_empty[ks].init()
                bars.mb_tma_v_full[ks].init()
                bars.mb_tma_v_empty[ks].init()

            # ---- MMA → softmax / corr handshakes ----
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                bars.mb_s_acc_empty[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()

                # ---- DSMEM xfer rings (sg0 → sg1) ----
                # P12: leader = CTA_MMA arrives; non-leader = 1.  Runtime-
                # conditional init count passed via override_count=.
                p_full_init = cutlass.Int32(
                    arith.select(
                        is_leader.ir_value(),
                        cutlass.Int32(CFG.CTA_MMA).ir_value(),
                        cutlass.Int32(CFG.ONE_LANE).ir_value(),
                    )
                )
                bars.mb_p_xfer_full[p].init(override_count=p_full_init)
                bars.mb_p_xfer_empty[p].init()
                bars.mb_alpha_xfer_full[p].init()
                bars.mb_alpha_xfer_empty[p].init()

            # ---- one-shot stats ring ----
            bars.mb_stats_xfer_full.init()
            bars.mb_stats_xfer_empty.init()

            # ---- sg1 TMA-STG handshakes ----
            for c in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[c].init()
            bars.mb_tma_o_empty.init()

            # ---- end-of-kernel teardown ----
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            # ---- scheduler ring (NOT in Bars; still raw nvvm.mbarrier_init) ----
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4 cluster fence (cga2-aware) — required because Phase 2 of the kernel
    # body issues cross-CTA arrive_on_peer BEFORE the receiving CTA has
    # necessarily flushed its mbar init through the cluster.  relaxed=True
    # (per project memory: dsv3 cga2 47%→92% SOL).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    # ------------------------------------------------------------------
    # Per-warp role dispatch.  Role-split topology:
    #   compute wg (warps 0..3): sg-conditional softmax (sg0) OR corr (sg1).
    #   MMA       (warp 4): sg-conditional BMM1 (sg0) OR BMM2 (sg1);
    #                       non-leader CTAs run quiet (sg0) or forwarder (sg1).
    #   TMA-LDG   (warp 5): sg-conditional Q+K (sg0) OR V (sg1).
    #   TMA-STG   (warp 6): sg1 only — O + LSE.  sg0 spins scheduler only.
    #   Scheduler (warp 7): try_cancel; both sgs.
    # ------------------------------------------------------------------
    if warp_idx >= cutlass.Int32(CFG.SOFTMAX_WG0_BASE) and warp_idx < cutlass.Int32(CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp_group(
            is_sg0=is_sg0,
            is_sg1=is_sg1,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sQ=sQ,
            sO=sO,
            sP_xfer_raw=sP_xfer_raw,
            sAlpha_xfer_raw=sAlpha_xfer_raw,
            sStats_xfer_raw=sStats_xfer_raw,
            sLSE_raw=sLSE_raw,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            cross_sg_peer=cross_sg_peer,
            o_scale_fused=o_scale_fused,
            amax_o_tensor=amax_o_tensor,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        # cga2 non-leader paths fork — sg0 non-leader is quiet, sg1 non-leader
        # forwards mb_p_xfer_full to leader (P12) in a persistent loop.
        if is_leader:
            _mma_warp_group(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sP_xfer_raw=sP_xfer_raw,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                sg0_mcast_mask=sg0_mcast_mask,
                cta_in_pair=cta_in_pair,
            )
        else:
            _mma_warp_non_leader(
                is_sg0=is_sg0,
                is_sg1=is_sg1,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                cta_in_pair=cta_in_pair,
                leader_cta_id=leader_cta_id,
            )

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            is_sg0=is_sg0,
            is_sg1=is_sg1,
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            o_desc_words=o_desc_words,
            sQ=sQ,
            sK=sK,
            sV=sV,
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
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            is_sg1=is_sg1,
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
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + the device claim counter in the 4B+4
            # metadata's last two slots.  The dense CLC path hands out tiles
            # against a grid that is the persistent cluster count, not the work
            # list, so every tile->unit mapping is off without this.
            scheduler_warp_loop_persistent(
                sched,
                CFG.SCHEDULER_STAGES,
                is_cga_first_cta,
                seq_kv_lens_tensor,
                cutlass.Int32(4) * n_batch + cutlass.Int32(3),
                cutlass.Int32(4) * n_batch + cutlass.Int32(2),
                CGA_SIZE,
                CFG.CGA_M,
            )
        else:
            scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta)


# ============================================================================
# sg0 softmax-iter helper — per-iter body for the 3-segment kv loop
# (LEFT-masked / unmasked center / RIGHT-masked).  Mirrors canonical
# `_softmax_kv_body` (prefill_sdpa_f16.py:923) and C++
# `softmax_kv_body<ApplyMask>` (prefill_sdpa_d512_fp8.cu:432).
#
# apply_mask=False uses fused HW row-max (tcgen05.ld.red.f32.max);
# apply_mask=True uses tcgen05.ld + apply_mask_chunk + software row-max
# (HW max can't observe -inf written by the mask after the load).
#
# Top-level @cute.jit so the dynamic call-graph contains no closure-capture
# violations (the pre-upstream DSL forbids closures under dynamic control flow).  All
# kernel-local vars are passed as explicit args; constants come from
# module-level (NEG_INF_F32, RESCALE_THRESHOLD_F32, P_*, SOFTMAX_*, etc.).
# ============================================================================
@cute.jit
def _sg0_softmax_kv_iter(
    apply_mask: cutlass.Constexpr[bool],
    kv_loop,
    # State (threaded, returned updated):
    sg0_xfer_state,
    bmm1_done_state,
    total_max,
    total_sum_vec,
    # TMEM / SMEM bases:
    tmem_base_addr,
    bars,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    # Per-tile / per-lane:
    q_abs,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    tid_in_wg,
    is_lead_warp,
    leader_cta_id,
    cross_sg_peer,
):
    cur_parity_S = sg0_xfer_state.idx
    cur_phase_S = sg0_xfer_state.phase
    bars.mb_alpha_xfer_empty[cur_parity_S].wait(cur_phase_S)
    bars.mb_p_xfer_empty[cur_parity_S].wait(cur_phase_S)
    bars.mb_bmm1_done[bmm1_done_state.idx].wait(bmm1_done_state.phase)
    sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
    bmm1_done_state = advance(bmm1_done_state, CFG.XFER_STAGES)

    s_addr_base = tmem_base_addr + cur_parity_S * cutlass.Int32(LAYOUT.S_ACC_COLS)

    if cutlass.const_expr(apply_mask):
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
        raw_chunks = [
            nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * SOFTMAX_CHUNK), cutlass.Float32),
                num=SOFTMAX_CHUNK,
            )
            for c in range(SOFTMAX_N_CHUNKS_LOAD)
        ]
        # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
        # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
        chunks_S = [
            apply_mask_chunk(
                raw_chunks[c],
                q_abs,
                kv_col_base + cutlass.Int32(c * SOFTMAX_CHUNK),
                eff_seqlen_kv,
                CFG.WINDOW_LEFT,
                CFG.MASK_FLAGS,
                N=SOFTMAX_CHUNK,
                bottom_right=CFG.BOTTOM_RIGHT,
                causal_diag=causal_diag,
                window_right=CFG.WINDOW_RIGHT,
            )
            for c in range(SOFTMAX_N_CHUNKS_LOAD)
        ]
        chunks_max = [row_max_reduction(chunks_S[c]) for c in range(SOFTMAX_N_CHUNKS_LOAD)]
        reg_S_vec = vec_concat(chunks_S)
        current_max_raw = chunks_max[0]
        for m in chunks_max[1:]:
            current_max_raw = cute.math.max(current_max_raw, m)
        reg_S_tile = RegTile(reg_S_vec, size=CFG.TILE_N)
    else:
        reg_S_tile, current_max_raw = tmem_load_max_reduction_tile(
            s_addr_base,
            num_elems=CFG.TILE_N,
        )
    current_max = current_max_raw * scale_log2

    # Online softmax (RESCALE_THRESHOLD skip).
    old_total_max = total_max
    is_first = total_max == NEG_INF_F32
    update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD_F32)
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
            NEG_INF_F32.ir_value(),
            (old_total_max - total_max).ir_value(),
        )
    )
    alpha = cute.math.exp2(exp_input, fastmath=True)

    # reg_S = reg_S * scale_log2 - total_max; then exp2.  Keep the FP32
    # RegTile (slice via .vec slicing) and cast each chunk to FP8 inside the
    # P_TMA_ITERS loop — RegTile(Float8E4M3FN) is illegal at JIT, but slicing
    # the FP32 RegTile then casting the slice is fine (per d256_fp8 pattern).
    reg_S_scaled = reg_S_tile.vec * scale_log2 - total_max
    reg_P_fp32 = cute.math.exp2(reg_S_scaled, fastmath=True)
    reg_P_tile = RegTile(reg_P_fp32, size=CFG.TILE_N)

    # ---- Write P[tid, :] to SMEM xfer ring slot[parity] ----
    p_xfer_slot = sP_xfer_raw.subview(cur_parity_S * cutlass.Int32(pXferElems))
    for chunk in cutlass.range_constexpr(P_TMA_ITERS):
        smem_off = cutlass.Int32(chunk * P_BLOCK_BYTES) + tid_in_wg * cutlass.Int32(P_D_BLOCK)
        smem_ptr = p_xfer_slot.subview(smem_off).data_ptr()
        # Slice FP32 RegTile chunk, then cast → FP8.  Avoids RegTile(FP8).
        chunk_P_fp32 = reg_P_tile[chunk * P_D_BLOCK : (chunk + 1) * P_D_BLOCK].vec
        chunk_P = chunk_P_fp32.to(P_STORAGE_DTYPE)
        smem_ptr.store_swizzled(
            chunk_P,
            alignment=64,
            swizzle=P_SMEM_SWIZZLE,
        )

    # ---- Write alpha[tid] (FP32) to alpha xfer slot[parity] ----
    alpha_slot = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
    alpha_slot.store(alpha)

    # Update total_sum = total_sum * alpha + row_reduction(reg_P).
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    iter_sum_pair = row_reduction_pair(reg_P_fp32)
    total_sum_vec = total_sum_vec * alpha_pair + iter_sum_pair

    # Fence SMEM→async; sync the 4 compute warps so all 128 rows of
    # alpha + P are written before bulk_copy issues.
    nvvm.fence_proxy("async.shared", space="cta")
    nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

    # DSMEM bulk_copy alpha + P → cross-sg sg1 peer's SMEM, firing peer's
    # mb_alpha_xfer_full + mb_p_xfer_full.  PREDICATED form: emit `@p
    # cp.async.bulk` (matches the C++ reference's uniform predicated form) instead of an
    # `if is_lead_warp: if elect_sync():` branch — that branch lowers to a
    # warp-divergent BSSY/BSYNC + reconverge that also poisons ptxas's uniform
    # analysis of nearby mbar waits (measured: removing the ship branches flips
    # ~5 SYNCS.PHASECHK → USYNCS and is worth ~+12% on dsv4 d512).  All 128
    # lanes compute the cheap mapa addresses up-front; only the lead warp's
    # elected lane issues the bulk copies under ship_pred.
    ship_pred = is_lead_warp & nvvm.elect_sync()
    local_alpha_src = sAlpha_xfer_raw.subview(cur_parity_S * cutlass.Int32(CFG.TILE_M))
    peer_alpha_dst = nvvm.mapa(local_alpha_src, cross_sg_peer, addrspace=7)
    peer_alpha_full_mbar = nvvm.mapa(bars.mb_alpha_xfer_full[cur_parity_S].smem_ptr, cross_sg_peer, addrspace=7)
    local_p_src = sP_xfer_raw.subview(cur_parity_S * cutlass.Int32(pXferElems))
    peer_p_dst = nvvm.mapa(local_p_src, cross_sg_peer, addrspace=7)
    peer_p_full_mbar = nvvm.mapa(bars.mb_p_xfer_full[cur_parity_S].smem_ptr, cross_sg_peer, addrspace=7)
    cp_async_bulk_shared_cluster_shared_cta(
        peer_alpha_dst,
        local_alpha_src,
        peer_alpha_full_mbar,
        alphaXferBytes,
        pred=ship_pred,
    )
    cp_async_bulk_shared_cluster_shared_cta(
        peer_p_dst,
        local_p_src,
        peer_p_full_mbar,
        pXferBytes,
        pred=ship_pred,
    )

    # All-thread DSMEM arrive on sg0 leader's mb_s_acc_empty — fired at the
    # END of the iter (after the P/alpha DSMEM ship), matching the C++
    # baseline's perf-tuned placement (softmax_kv_body tail:
    # `mb_s_acc_empty[parity].arrive_on_peer(0u,1u)`).  The barrier_id=8 sync
    # above already guarantees all 128 lanes finished reading S_acc from TMEM,
    # so freeing the slot here is safe; firing it early (right after the TMEM
    # read) measurably regressed perf — the MMA reclaims the S_acc slot too
    # eagerly and contends with the in-flight softmax.
    bars.mb_s_acc_empty[cur_parity_S].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

    return (sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec)


# ============================================================================
# Compute warp group — sg-conditional softmax (sg0) or correction (sg1).
# 4 warps × 32 lanes = 128 threads.  Port of C++ compute_warp_group
# (prefill_sdpa_d512_fp8.cu:529-1013).
# ============================================================================
@cute.jit
def _compute_warp_group(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    scale_log2,
    tmem_ptr_i32,
    sQ,
    sO,
    sP_xfer_raw,
    sAlpha_xfer_raw,
    sStats_xfer_raw,
    sLSE_raw,
    bars,
    sched,
    lse_tensor,
    sinks_tensor,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    cross_sg_peer,
    o_scale_fused,
    amax_o_tensor,
):
    """sg-conditional compute body (4 warps × 32 lanes = 128 threads).

    sg0 path (port of C++ lines 599-749):
        Streaming softmax — per kv iter wait BMM1 done, tmem_load S_acc + HW
        row-max, online softmax (exp2 + alpha + new total_max / total_sum),
        publish alpha (DSMEM bulk_copy to sg1), publish P (DSMEM bulk_copy
        to sg1), all-thread arrive on mb_s_acc_empty.  End-of-tile ship
        (max, ell) stats over DSMEM (1-shot).  Per-tile pre-armed empty
        phases via PipelineState(0, 1).

    sg1 path (port of C++ lines 750-1024):
        Correction + epilogue — per kv iter wait alpha (local mbar), apply
        alpha to O TMEM in two halves (gates BMM2 sub-tile 0 / 1 via
        mb_bmm2_ready), DSMEM-arrive empty alpha on sg0.  Epilogue: wait
        final BMM2, wait stats, normalize O by 1/ell + cast to half, store
        to sO with per-chunk mb_tma_o_full arrive, write LSE to GMEM,
        DSMEM-arrive empty stats on sg0.

    Wait pattern: MMA warp publishes TMEM base via named barrier 1 (count =
    32 * (SOFTMAX_WG_WARPS + 1) = 160) BEFORE any tmem_ptr_i32.load().
    """
    # Wait MMA's TMEM-alloc publish (C++ line 592 ``named_barrier_wait(TMEM_ALLOC_SOFTMAX_BARRIER, 160)``).
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base_addr = tmem_ptr_i32.load()
    tmem_raw = nvvm.make_tmem_ptr(tmem_base_addr, cutlass.Int8)

    # Per-thread / per-warp identifiers — compute warps occupy CTA tid_x in
    # [0, 128); tid_in_wg == tid_x, wid_in_wg = tid_x // 32 ∈ {0..3}.
    tid_in_wg = cute.arch.thread_idx()[0]
    wid_in_wg = tid_in_wg // cutlass.Int32(32)
    is_lead_warp = wid_in_wg == cutlass.Int32(0)

    # Common state — persistent-tile scheduler decode.
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

    # kv_loop bounds — at MASK_NONE collapse to [0, seqlen_kv / TILE_N).
    # When MASK_FLAGS != 0 the kv loop splits into 3 segments
    # (LEFT-masked / fully-unmasked / RIGHT-masked) on unmasked_lo / unmasked_hi.
    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_unmasked_lo = cutlass.Int32(0)
        kv_unmasked_hi = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        eff_seqlen_kv = seqlen_kv
        # Bind the Q side on this arm too.  The unmasked sg0 loop below passes
        # eff_seqlen_q to _sg0_softmax_kv_iter, and the only other binding is in
        # the masked `else`.  It happens to trace today, but relying on that is
        # relying on how the DSL preprocessor treats a const_expr branch's
        # locals; at MASK_FLAGS == 0 the per-sequence length IS the scalar, so
        # say so explicitly.
        eff_seqlen_q = seqlen_q
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_unmasked_lo = bounds_init.unmasked_lo
        kv_unmasked_hi = bounds_init.unmasked_hi
        kv_right = bounds_init.right

    # ---- Per-tensor swizzle for SMEM stores (same Swz128B as Q/O at d=512) ----
    # P SMEM swizzle hoisted to module-level (P_SMEM_SWIZZLE) — see top of file.
    _O_EPI_SWIZZLE = cutlass.Swizzle(3, 4, 3)

    # Epilogue tile params (O SMEM stride).  TILE_O=512, BPE_O=2 → 8 chunks.
    O_EPI_BLOCK_SIZE = 64 // CFG.BPE_O  # 32 fp16, 64 fp8
    O_TMA_ITERS = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES  # 8 chunks
    O_D_BLOCK = CFG.TILE_O // O_TMA_ITERS  # 64 elements / chunk
    O_TMA_GRANU_ELEMS = CFG.TILE_M * O_D_BLOCK  # 8192 elements / TMA chunk
    O_BLOCKS_PER_SUB = 128 // O_EPI_BLOCK_SIZE  # 4 (f16) / 2 (fp8)
    O_CHUNK_ELEMS = 128 // CFG.BPE_O  # 64 (f16) / 128 (fp8)

    # ---- sg0 state — streaming softmax + DSMEM ship ----
    bmm1_done_state = PipelineState.start(phase=0)
    # PipelineState(0, 1) → both mb_p_xfer_empty and mb_alpha_xfer_empty are
    # pre-armed; iter-0/1 wait passes immediately on fresh barriers.
    sg0_xfer_state = PipelineState.start(phase=1)
    stats_xfer_empty_phase = cutlass.Int32(1)

    # ---- sg1 state — recv alpha + correction + epilogue ----
    alpha_full_state = PipelineState.start(phase=0)
    sg1_bmm2_done_state = PipelineState.start(phase=0)
    stats_xfer_full_phase = cutlass.Int32(0)
    epilogue_state = cutlass.Int32(0)

    # ---- streaming-softmax accumulators (sg0) — Vector[Float32,2] for
    #      packed FMUL2/FADD2 lowering, per the canonical f16 port pattern.
    total_max = NEG_INF_F32
    total_sum_vec = cutlass.Vector.from_elements(
        (cutlass.Float32(0.0), cutlass.Float32(0.0)),
        cutlass.Float32,
    )

    # ==================================================================
    # TODO(main agent): port sg0 streaming-softmax + DSMEM-ship body from
    # C++ prefill_sdpa_d512_fp8.cu lines 432-522 (softmax_kv_body) and
    # 599-749 (per-tile loop), and sg1 correction + epilogue body from
    # C++ lines 750-1024.
    #
    # Must preserve:
    #   - sg0 mb_p_xfer_empty / mb_alpha_xfer_empty pre-armed at phase=1
    #     (PipelineState(0, 1)) — matches C++ ``Sg0KvState s.sg0_state(0, 1)``.
    #   - sg0 mb_bmm1_done_state NOT pre-armed (phase=0) — BMM1 commit must
    #     land before sg0 sm reads S_acc[0].
    #   - sg0 mb_s_acc_empty.arrive_on_peer(0u, 1u) — all-thread DSMEM-arrive
    #     to sg0 leader (256 lanes total).  Cross-warp sync between
    #     tmem_load_fence and the arrive is BUILT INTO the all-thread arrive
    #     (P8 + P11) — no extra named_barrier_wait needed.
    #   - sg0 DSMEM bulk_copy to peer SMEM uses
    #     ``cp_async_bulk_shared_cluster_shared_cta`` (per project memory:
    #     bare nvvm op silently fires sender's local mbar instead of receiver).
    #   - sg1 mb_alpha_xfer_full[parity_cur].arrive(ALPHA_XFER_TILE_BYTES, lead_warp_elect)
    #     — wrapper @p-gates on elect, the wait below spins on phase parity.
    #   - sg1 iter-0 BOTH mb_bmm2_ready half slots fired immediately (BMM2(0)
    #     doesn't depend on alpha — C++ lines 798-805).
    #   - sg1 correction split into 2 halves of CORR_BLOCKS_PER_HALF blocks
    #     each (TILE_O / correction_block_size / 2 blocks per half), with
    #     mb_bmm2_ready[parity*2+0] fired after half-1, [parity*2+1] after
    #     half-2.  Lets BMM2 sub-tile 0 start while corr works on half-2.
    #   - sg1 epilogue: TMEM layout SCRAMBLED by cga2 N-split + 2-call N-block
    #     split (cf. C++ lines 964-983).  Iterate in OUTPUT order with bit-
    #     swap [0,2,1,3] permutation.
    #   - sg1 mb_tma_o_full arrives per-128B chunk in output order.
    #   - sg1 mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ 1u) — DSMEM fan-out
    #     between cga2 partners (cf. C++ lines 1023, 748).
    # ==================================================================
    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            # ============================================================
            # sg0 — streaming softmax + ship alpha + P + stats.
            # Port of C++ prefill_sdpa_d512_fp8.cu lines 432-522 + 585-735.
            # ============================================================
            total_max = NEG_INF_F32
            total_sum_vec = cutlass.Vector.from_elements(
                (cutlass.Float32(0.0), cutlass.Float32(0.0)),
                cutlass.Float32,
            )

            # Per-lane absolute Q row (used by apply_mask_chunk in
            # LEFT/RIGHT segments).  Same anchor C++ uses at
            # prefill_sdpa_d512_fp8.cu:473 (`tc.q_row_coord + tid`).
            q_abs = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg

            if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
                pass
            else:
                # 3-segment kv loop: LEFT-masked / unmasked center / RIGHT-masked.
                # MASK_NONE collapses the two masked sub-loops to empty range
                # (kv_left == kv_unmasked_lo and kv_unmasked_hi == kv_right),
                # so the LEFT/RIGHT branches fold out at trace time.
                if cutlass.const_expr(CFG.MASK_FLAGS == 0):
                    for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec = _sg0_softmax_kv_iter(
                            False,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                else:
                    for _kv in cutlass.range(kv_left, kv_unmasked_lo, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec = _sg0_softmax_kv_iter(
                            True,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                    for _kv in cutlass.range(kv_unmasked_lo, kv_unmasked_hi, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec = _sg0_softmax_kv_iter(
                            False,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )
                    for _kv in cutlass.range(kv_unmasked_hi, kv_right, 1, unroll=1):
                        sg0_xfer_state, bmm1_done_state, total_max, total_sum_vec = _sg0_softmax_kv_iter(
                            True,
                            _kv,
                            sg0_xfer_state,
                            bmm1_done_state,
                            total_max,
                            total_sum_vec,
                            tmem_base_addr,
                            bars,
                            sP_xfer_raw,
                            sAlpha_xfer_raw,
                            q_abs,
                            eff_seqlen_kv,
                            eff_seqlen_q,
                            scale_log2,
                            tid_in_wg,
                            is_lead_warp,
                            leader_cta_id,
                            cross_sg_peer,
                        )

            # ---- End-of-tile: ship final stats (max, ell) ----
            bars.mb_stats_xfer_empty.wait(stats_xfer_empty_phase)
            stats_xfer_empty_phase = stats_xfer_empty_phase ^ cutlass.Int32(1)

            final_sum = total_sum_vec[0] + total_sum_vec[1]
            # Stats layout in SMEM: ell[TILE_M] then max[TILE_M].
            stats_sum_slot = sStats_xfer_raw.subview(tid_in_wg)
            stats_max_slot = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            stats_sum_slot.store(final_sum)
            stats_max_slot.store(total_max)

            nvvm.fence_proxy("async.shared", space="cta")
            nvvm.barrier_cta_sync(barrier_id=8, thread_count=128)

            # Predicated stats ship — same rationale as the per-iter alpha/P
            # ship above (predicated, no warp-divergent branch).
            ship_pred = is_lead_warp & nvvm.elect_sync()
            peer_stats_dst = nvvm.mapa(sStats_xfer_raw, cross_sg_peer, addrspace=7)
            peer_stats_full_mbar = nvvm.mapa(bars.mb_stats_xfer_full.smem_ptr, cross_sg_peer, addrspace=7)
            cp_async_bulk_shared_cluster_shared_cta(
                peer_stats_dst,
                sStats_xfer_raw,
                peer_stats_full_mbar,
                statsXferBytes,
                pred=ship_pred,
            )
        else:
            # ============================================================
            # sg1 — correction (apply alpha) + epilogue (normalize, cast,
            # store O, LSE).  Port of C++ lines 750-1024.
            # ============================================================
            tmem_O_base = tmem_base_addr + cutlass.Int32(LAYOUT.O_OFF)

            if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
                # Empty-kv branch: signal MMA's empty-mainloop wait
                # (each sg1 CTA's elect lane DSMEM-arrives sg1 leader).
                bars.mb_empty_mainloop.arrive_on_peer(leader_cta_id, pred=is_lead_warp & nvvm.elect_sync())
            else:
                # --- iter 0: BMM2(0) doesn't depend on alpha; fire both
                #     bmm2_ready half slots immediately (C++ 798-805).
                cur_parity_0 = alpha_full_state.idx
                bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                bars.mb_bmm2_ready[cur_parity_0 * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
                    leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA
                )

                # Arm + wait alpha_xfer_full[parity_0].  Wrapper @p-gates
                # on elect; wait spins on phase parity (atomically observable).
                bars.mb_alpha_xfer_full[cur_parity_0].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
                bars.mb_alpha_xfer_full[cur_parity_0].wait(alpha_full_state.phase)
                alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

                # Notify sg0 (cross-sg peer) alpha slot is empty.
                bars.mb_alpha_xfer_empty[cur_parity_0].arrive_on_peer(cross_sg_peer)

                # --- iter 1..n_kv-1: apply alpha to O before BMM2(kv) ---
                CORR_BLOCK_SIZE = 16
                CORR_BLOCKS_TOTAL = CFG.TILE_O // CORR_BLOCK_SIZE  # 32
                CORR_BLOCKS_PER_HALF = CORR_BLOCKS_TOTAL // 2  # 16

                for _kv in cutlass.range(kv_left + cutlass.Int32(1), kv_right, 1, unroll=1):
                    cur_parity = alpha_full_state.idx

                    bars.mb_alpha_xfer_full[cur_parity].arrive(n_bytes=alphaXferBytes, pred=is_lead_warp & nvvm.elect_sync())
                    bars.mb_alpha_xfer_full[cur_parity].wait(alpha_full_state.phase)
                    alpha_full_state = advance(alpha_full_state, CFG.XFER_STAGES)

                    # Wait prior BMM2 done so O is committed.
                    bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
                    sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

                    # Read alpha[tid] from xfer_in[cur_parity].
                    alpha_addr = sAlpha_xfer_raw.subview(cur_parity * cutlass.Int32(CFG.TILE_M) + tid_in_wg)
                    alpha_corr = alpha_addr.load()

                    # all_alpha_one ballot: skip rescale entirely if every
                    # lane's alpha == 1.0.
                    alpha_is_one = alpha_corr == cutlass.Float32(1.0)
                    all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

                    # ---- Half-1: cols [0..TILE_O/2) ----
                    if ~all_alpha_one:
                        for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                            o_off = tmem_O_base + cutlass.Int32(block * CORR_BLOCK_SIZE)
                            o_chunk = nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                num=CORR_BLOCK_SIZE,
                            )
                            o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                o_scaled,
                            )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                    # Notify sg0 alpha slot empty.
                    bars.mb_alpha_xfer_empty[cur_parity].arrive_on_peer(cross_sg_peer)
                    # Half-1 ready — sg1 leader BMM2 sub-tile 0 unblocked.
                    bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                    # ---- Half-2: cols [TILE_O/2..TILE_O) ----
                    if ~all_alpha_one:
                        for block in cutlass.range_constexpr(CORR_BLOCKS_PER_HALF):
                            block_idx = CORR_BLOCKS_PER_HALF + block
                            o_off = tmem_O_base + cutlass.Int32(block_idx * CORR_BLOCK_SIZE)
                            o_chunk = nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                num=CORR_BLOCK_SIZE,
                            )
                            o_scaled = vec_scale_pair(o_chunk, alpha_corr, CORR_BLOCK_SIZE)
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(o_off, cutlass.Float32),
                                o_scaled,
                            )
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

                    # Half-2 ready — sg1 leader BMM2 sub-tile 1 unblocked.
                    bars.mb_bmm2_ready[cur_parity * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(
                        leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA
                    )

            # ---- Final BMM2-done wait (always runs — empty path covered) ----
            bars.mb_bmm2_done[sg1_bmm2_done_state.idx].wait(sg1_bmm2_done_state.phase)
            sg1_bmm2_done_state = advance(sg1_bmm2_done_state, CFG.XFER_STAGES)

            # ---- Epilogue ----
            epilogue_state = epilogue_state ^ cutlass.Int32(1)
            bars.mb_tma_o_empty.wait(epilogue_state)

            # Arm stats_xfer_full (sg0 delivers via DSMEM bulk_copy).
            bars.mb_stats_xfer_full.arrive(n_bytes=statsXferBytes, pred=is_lead_warp & nvvm.elect_sync())
            bars.mb_stats_xfer_full.wait(stats_xfer_full_phase)
            stats_xfer_full_phase = stats_xfer_full_phase ^ cutlass.Int32(1)

            # Per-thread lds_32 of final_ell / final_max.
            ell_addr = sStats_xfer_raw.subview(tid_in_wg)
            max_addr = sStats_xfer_raw.subview(cutlass.Int32(CFG.TILE_M) + tid_in_wg)
            final_ell = ell_addr.load()
            final_max = max_addr.load()

            # Stats consumed → notify sg0 stats_xfer.out free to overwrite.
            bars.mb_stats_xfer_empty.arrive_on_peer(cross_sg_peer, pred=is_lead_warp & nvvm.elect_sync())

            # Compute beta, lse — HAS_SINK fold (dsv4 today: HAS_SINK=0,
            # collapses to plain 1/total_sum normalization).
            LN2 = cutlass.Float32(0.6931471805599453)
            if cutlass.const_expr(CFG.HAS_SINK):
                sinks_arr = cutlass.make_array_view(sinks_tensor)
                sink_logit = sinks_arr[head_idx]
                final_max_nat = final_max * LN2
                new_max_nat = cute.math.max(final_max_nat, sink_logit)
                scale_sink = cute.math.exp(final_max_nat - new_max_nat, fastmath=True)
                new_sum = final_ell * scale_sink + cute.math.exp(sink_logit - new_max_nat, fastmath=True)
                beta = (scale_sink * o_scale_fused) / new_sum
                lse = new_max_nat + cute.math.log(new_sum, fastmath=True)
            else:
                final_ell_safe = cute.math.max(final_ell, cutlass.Float32(1e-30))
                beta = o_scale_fused / final_ell_safe
                lse = final_max * LN2 + cute.math.log(final_ell_safe, fastmath=True)

            # Empty KV range (fully-masked / zero-length sequence): the mainloop
            # ran ZERO iterations, so final_ell is exactly 0 and the O
            # accumulator is RESIDUE (the suite poisons TMEM on purpose, so it
            # can be a NaN bit pattern).  The 1e-30 floor above keeps the
            # division finite but LEAKS: beta becomes ~1e30 -> +inf after the
            # o_scale fold, and residue * inf = NaN; lse becomes log(1e-30) =
            # -69.08 instead of -inf.  SELECT the answer -- never `* 0`, which
            # a NaN residue survives.  Mirrors the d512 f16 / MXFP8 siblings.
            # (sdpa-invariants.md S2/S3; the sink arm already yields
            # sink_logit at total_sum == 0, so it needs no LSE substitution.)
            _row_empty = final_ell == cutlass.Float32(0.0)
            if cutlass.const_expr(not CFG.HAS_SINK):
                lse = cutlass.Float32(arith.select(_row_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse.ir_value()))

            # Cast O fp32 → fp8/bf16/fp16 (CFG.DTYPE_O) with bit-permuted TMEM block index.
            # TMEM layout is scrambled by cga2 N-split + 2-call N-block:
            #   TMEM[ 0:128] = d_v[  0:128]   (call 0, leader half)
            #   TMEM[128:256] = d_v[256:384]   (call 0, peer half)
            #   TMEM[256:384] = d_v[128:256]   (call 1, leader half)
            #   TMEM[384:512] = d_v[384:512]   (call 1, peer half)
            # Iterate in OUTPUT order; bit-swap b_sub ∈ {0..3} → {0,2,1,3}.
            sO_base = sO[0].base

            # amax_o = max over VALID rows of |o| (fp32, pre-cast).  The atomic
            # itself is gated on row validity further down, where q_row_global
            # exists -- atomicMax only grows, so a padded row would permanently
            # inflate the graph's Amax_O with no error anywhere.
            _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
            _amax_o_local = cutlass.Float32(0.0)

            for b in cutlass.range_constexpr(CFG.TILE_O // O_EPI_BLOCK_SIZE):
                b_intra = b & (O_BLOCKS_PER_SUB - 1)
                b_sub = b // O_BLOCKS_PER_SUB
                tmem_sub = ((b_sub & 1) << 1) | ((b_sub & 2) >> 1)
                tmem_block = tmem_sub * O_BLOCKS_PER_SUB + b_intra

                o_addr = tmem_O_base + cutlass.Int32(tmem_block * O_EPI_BLOCK_SIZE)
                o_fp32 = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_EPI_BLOCK_SIZE,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_fp32 * beta
                # Zero an empty row with a SELECT (see above), and fold amax
                # over the SUBSTITUTED values so a dead row cannot poison Amax_O.
                _o_elems = []
                for _i in cutlass.range_constexpr(O_EPI_BLOCK_SIZE):
                    _e = cutlass.Float32(arith.select(_row_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value()))
                    _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))
                    _o_elems.append(_e)
                o_half = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)

                # SMEM store — TMA-O grain layout (O_D_BLOCK elements / chunk).
                col_offset_const = (b * O_EPI_BLOCK_SIZE) % O_D_BLOCK
                block_idx_const = (b * O_EPI_BLOCK_SIZE) // O_D_BLOCK
                block_offset_const = block_idx_const * O_TMA_GRANU_ELEMS
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(O_D_BLOCK)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()
                smem_ptr.store_swizzled(
                    o_half,
                    alignment=64,
                    swizzle=_O_EPI_SWIZZLE,
                )

                # Per-128B-chunk arrive on TMA-STG.
                if ((b + 1) * O_EPI_BLOCK_SIZE) % O_CHUNK_ELEMS == 0:
                    chunk = (b * O_EPI_BLOCK_SIZE) // O_CHUNK_ELEMS
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_tma_o_full[chunk].arrive()

            # Write LSE — under cga2 each sg1 peer writes its half of O+LSE
            # rows (leader = [0:128], peer = [128:256]).  Per-thread row.
            q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg
            # Gated exactly like the LSE write below.
            if q_row_global < seqlen_q:
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: q_row_global is sequence-local; LSE is packed [1,QH,T] →
                # index [0, head, cu_q[b] + local], bound by per-sequence Q len S_q_b.
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                if cutlass.const_expr(lse_tensor is not None):
                    if q_row_global < _s_q_b:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        # Written in the CALLER's layout, picked by the STATIC rank compile()
                        # baked in: token-major rank-2 [T, QH] (the DEFAULT) or head-major
                        # rank-3 [1, QH, head_stride].  Serving only the rank-3 arm
                        # transposes every LSE on the common path.
                        if cutlass.const_expr(len(lse_tensor.shape) == 2):
                            lse_row = lse_arr[_cu_q_b + q_row_global, :]
                            lse_row[head_idx] = lse
                        else:
                            lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                            lse_row[_cu_q_b + q_row_global] = lse
            else:
                if cutlass.const_expr(lse_tensor is not None):
                    if q_row_global < seqlen_q:
                        lse_arr = cutlass.make_array_view(lse_tensor)
                        lse_row = lse_arr[batch_idx, head_idx, :]
                        lse_row[q_row_global] = lse

        # End-of-tile: advance scheduler.
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
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_unmasked_lo = bounds_next.unmasked_lo
            kv_unmasked_hi = bounds_next.unmasked_hi
            kv_right = bounds_next.right

    # End-of-kernel: trailing-arrive drain (sg0 only — sg1's late
    # arrive_on_peer lands on a still-live mbar via this drain).  Per
    # C++ lines 722-744: drain XFER_STAGES empty arrives + stats_xfer_empty
    # before sg0 sm exits.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            if is_lead_warp:
                if nvvm.elect_sync():
                    for _p in cutlass.range_constexpr(CFG.XFER_STAGES):
                        bars.mb_p_xfer_empty[sg0_xfer_state.idx].wait(sg0_xfer_state.phase)
                        bars.mb_alpha_xfer_empty[sg0_xfer_state.idx].wait(sg0_xfer_state.phase)
                        sg0_xfer_state = advance(sg0_xfer_state, CFG.XFER_STAGES)
                    bars.mb_stats_xfer_empty.wait(stats_xfer_empty_phase)
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        # Each CTA's compute warp fires 1 elect-arrive_on_peer to its cga2
        # partner → 1 arrive per CTA (init = ONE_LANE matches).
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta, pred=is_lead_warp & nvvm.elect_sync())


# ============================================================================
# MMA warp group — sg-conditional BMM1 (sg0 leader) / BMM2 (sg1 leader).
# Port of C++ mma_warp_group (prefill_sdpa_d512_fp8.cu:1021-1233).
# ============================================================================
@cute.jit
def _mma_warp_group(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    sP_xfer_raw,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    sg0_mcast_mask,
    cta_in_pair,
):
    """MMA warp leader — sg-conditional BMM1 (sg0 leader) or BMM2 (sg1 leader).

    sg0 leader (port of C++ lines 1101-1163):
        For each tile: wait Q full, then for each kv iter: wait S_acc[parity]
        empty (pre-armed at iter 0/1 via PipelineState(0, 1)), wait K[stage]
        full, mma_ss(Q, K) → S_acc[parity] (collective N = TILE_M*CTA_MMA),
        commit_mma on bmm1_done[parity] AND k_empty[stage] (P13 multicast,
        sg0_mcast_mask).  After last BMM1: commit_mma on q_empty (multicast).

    sg1 leader (port of C++ lines 1165-1246):
        For each tile: if n_kv > 0: for each kv iter: arrive own
        mb_p_xfer_full (own + DSMEM-forwarded from non-leader), wait
        mb_p_xfer_full, wait V[stage] full, for each of BMM2_LOOP_N_BLOCKS=2
        N-block calls: wait mb_bmm2_ready[parity*2+nblock], mma_ss(P, V) →
        O_block (per-call N=256, advance V desc by BMM2_V_NBLOCK_ADVANCE),
        commit bmm2_done[parity], v_empty[stage], p_xfer_empty[parity]
        (P13 multicast with sg0_mcast_mask).
        Empty-kv branch: wait mb_empty_mainloop, commit bmm2_done[parity].

    Both leaders share the TMEM alloc + named-barrier publish:
        tmem_alloc(576 cols, is_exclusive=True) → barrier_arrive(barrier_id=1,
        count = 32 * (SOFTMAX_WG_WARPS + 1) = 160).  No barrier_id=2 here
        (DSv4 has no separate correction-warp publish; correction is fused
        into the compute warp group's sg1 branch).
    """
    # SM107 cap = 576 cols; is_exclusive=True enforced inside tmem_alloc wrapper.
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    # Publish to compute warpgroup (waits on barrier_id=1 with count 160).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    # Do column arithmetic on raw Int8 ptr; retype at use site.  (Per
    # C++-to-DSL porting notes: make_tmem_ptr(addr, Float16) + N strides FP16 elems
    # not columns — silent-wrong-result bug.)
    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    # idesc M is COLLECTIVE (per-CTA M * CTA_MMA) under cga2.
    # k_dim=1 = Rubin K=64 2-chunk fast path (TILE_K_HW_BMM1=64).  Without it the
    # QMMA descriptor encodes K=32 → each tcgen05.mma processes half the K bytes
    # per step → silently-scrambled per-row S_acc.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=1,
    )
    # BMM2 idesc — N per call = 256 (NOT TILE_O); 2 calls per BMM2.
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=1,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2 with N per call = 256, B-transpose, manual k-step driven by kernel.
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.BMM2_N_PER_CALL,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    # Persistent-tile scheduler state.
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

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    # sg0 leader state (BMM1).
    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    # pre-armed: PipelineState(0, 1) so iter-0/1 wait passes on fresh barriers.
    s_acc_empty_state = PipelineState.start(phase=1)

    # sg1 leader state (BMM2).
    kv_state_V = PipelineState.start(phase=0)
    sg1_mma_state = PipelineState.start(phase=0)
    # bmm2_ready_state walks XFER_STAGES * N_BMM2_CHUNKS slots linearly
    # (parity*2 + nblock); phase flips every XFER_STAGES*N_BMM2_CHUNKS arrives.
    bmm2_ready_state = PipelineState.start(phase=0)
    bmm2_done_prod_state = PipelineState.start(phase=0)
    empty_mainloop_phase = cutlass.Int32(0)

    # Pre-build a SmemTile wrapper for the P xfer ring (sg1 leader only — sg0
    # builds its descriptor on its own SMEM, but for parity_cur lookup we use
    # the same swizzle/layout that sg0 used when writing).  P xfer SMEM is
    # TILE_M × TILE_N (cf. C++ P_Layout); matches Q swizzle (Swz128B at d=512).
    sP_xfer = SmemTile(
        base=sP_xfer_raw,
        elems_per_stage=pXferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_P,
        desc_version=DESC_VERSION,
    )

    # BMM1 leader: desc_Q is tile-static (Q stays resident under d=512 / no Q∪K alias).
    desc_Q = sQ[0].desc()
    # BMM2 leader: V N-block advance in ELEMENTS (BMM2_V_NBLOCK_ADVANCE is bytes).
    V_NBLOCK_ADVANCE_ELEMS = BMM2_V_NBLOCK_ADVANCE // CFG.BPE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if is_sg0:
            # ============================================================
            # sg0 leader — BMM1: Q × K^T → S_acc[parity]
            # ============================================================
            if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
                pass
            else:
                bars.mb_tma_q_full.wait(q_full_phase)
                q_full_phase = q_full_phase ^ cutlass.Int32(1)

                for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    cur_parity_K = s_acc_empty_state.idx
                    bars.mb_s_acc_empty[cur_parity_K].wait(s_acc_empty_state.phase)
                    s_acc_empty_state = advance(s_acc_empty_state, CFG.XFER_STAGES)

                    bars.mb_tma_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                    desc_K = sK[kv_state_K.idx].desc()
                    # S_acc TMEM offset = parity * S_ACC_COLS (TmemTile layout).
                    s_acc_off = cur_parity_K * cutlass.Int32(LAYOUT.S_ACC_COLS)
                    mma_ss(bmm1_desc, desc_Q, desc_K, tmem_raw.subview(s_acc_off), accumulate=False)
                    elect_p = nvvm.elect_sync()
                    bars.mb_bmm1_done[cur_parity_K].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    bars.mb_tma_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                # After last BMM1: multicast Q-empty so both peers' TMA warps
                # advance.  tcgen05.commit fires AFTER MMA drain (P13) so TMA
                # can't overwrite Q SMEM mid-read.
                bars.mb_tma_q_empty.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        else:
            # ============================================================
            # sg1 leader — BMM2: P × V → O via mma_ss, 2 N-block calls
            # ============================================================
            if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
                # Empty-kv branch — keep bmm2_done producer/consumer states
                # in lockstep across mixed empty/normal sequences.
                bars.mb_empty_mainloop.wait(empty_mainloop_phase)
                empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
                bars.mb_bmm2_done[bmm2_done_prod_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
                bmm2_done_prod_state = advance(bmm2_done_prod_state, CFG.XFER_STAGES)
            else:
                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    cur_parity = sg1_mma_state.idx

                    # Leader's own arrive on mb_p_xfer_full[parity] (own
                    # P-bytes were just delivered by cross-sg sg0 partner
                    # via DSMEM bulk_copy).  Init = CTA_MMA arrives — this
                    # contributes 1, the non-leader forwarder contributes
                    # the other 1 (P12).
                    bars.mb_p_xfer_full[cur_parity].arrive(n_bytes=pXferBytes, pred=nvvm.elect_sync())
                    bars.mb_p_xfer_full[cur_parity].wait(sg1_mma_state.phase)
                    sg1_mma_state = advance(sg1_mma_state, CFG.XFER_STAGES)

                    bars.mb_tma_v_full[kv_state_V.idx].wait(kv_state_V.phase)

                    accum_b2 = kv_loop > kv_left

                    # Rebuild SmemDescs for this iter's P parity and V stage.
                    desc_P = sP_xfer[cur_parity].desc()
                    desc_V_n0 = sV[kv_state_V.idx].desc()
                    desc_V_n1 = sV[kv_state_V.idx].shifted(V_NBLOCK_ADVANCE_ELEMS).desc()

                    # N-block 0: writes O TMEM cols [0..BMM2_N_PER_CALL).
                    bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                    bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                    mma_ss(bmm2_desc, desc_P, desc_V_n0, tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF)), accumulate=accum_b2)

                    # N-block 1: writes O TMEM cols [BMM2_N_PER_CALL..TILE_O).
                    bars.mb_bmm2_ready[bmm2_ready_state.idx].wait(bmm2_ready_state.phase)
                    bmm2_ready_state = advance(bmm2_ready_state, CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS)
                    mma_ss(bmm2_desc, desc_P, desc_V_n1, tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + CFG.BMM2_N_PER_CALL)), accumulate=accum_b2)

                    elect_p = nvvm.elect_sync()
                    bars.mb_bmm2_done[bmm2_done_prod_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    bars.mb_tma_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    # P13 multicast on mb_p_xfer_empty — sg0_mcast_mask
                    # (= 0x3) targets BOTH sg0 CTAs so sg0 sm's next P
                    # slot reuse is unblocked.
                    bars.mb_p_xfer_empty[cur_parity].arrive(mcast_mask=sg0_mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    kv_state_V = advance(kv_state_V, CFG.STAGES_KV)
                    bmm2_done_prod_state = advance(bmm2_done_prod_state, CFG.XFER_STAGES)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

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
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # End-of-warp tmem_dealloc — wait fan-in from both compute warps then free.
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# ============================================================================
# MMA warp non-leader paths — sg0 minimal quiet vs sg1 P12 forwarder.
# Port of C++ mma_warp_group non-leader paths (prefill_sdpa_d512_fp8.cu:1036-1082).
# ============================================================================
@cute.jit
def _mma_warp_non_leader(
    is_sg0,
    is_sg1,
    seqlen_q,
    seqlen_kv,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    leader_cta_id,
):
    """sg0 non-leader: minimal quiet (alloc + wait dealloc + free).
    sg1 non-leader: persistent loop forwarding mb_p_xfer_full to leader (P12).

    sg0 quiet (port of C++ lines 1049-1054):
        Just participates in TMEM alloc + named-barrier publish (sm warps
        read the published base ptr from peer's TMEM under cga2 collective
        MMA), then waits on mb_tmem_dealloc and frees.

    sg1 forwarder (port of C++ lines 1055-1093):
        Persistent tile loop — at iter scope, per kv_local:
          - arrive(mb_p_xfer_full[parity], P_XFER_TILE_BYTES, elect) — local mbar
          - wait(mb_p_xfer_full[parity], nlmma_state.phase) — local mbar
          - arrive_on_peer(mb_p_xfer_full[parity], leader_cta_id) — DSMEM forward
        Required because BMM2's collective mma_ss reads BOTH peers' P SMEM via
        crossbar; leader's local mbar only sees its own cross-sg sg0 partner's
        bytes, so non-leader must DSMEM-arrive on leader after observing own bytes.
    """
    # All-lanes warp-collective tmem_alloc (no elect_sync gating).
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    # Match leader's named-barrier arrive count (lead is +1 on barrier_id=1).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))

    # sg0 non-leader is quiet (just wait dealloc); sg1 non-leader runs the
    # persistent P12 forwarder loop.  `return` is forbidden inside @cute.jit
    # (DSL gotcha), so we gate the forwarder on `is_sg1` and let both arms
    # fall through to the shared dealloc wait + free at the bottom.
    if is_sg1:
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

        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            kv_left = cutlass.Int32(0)
            kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
        else:
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_init.left
            kv_right = bounds_init.right

        nlmma_state = PipelineState.start(phase=0)

        while is_valid_tile > cutlass.Int32(0):
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
                pass
            else:
                for _kv in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    # Arm own expect_tx for P-bytes delivered by cross-sg sg0
                    # partner (via DSMEM bulk_copy).  Wait local mbar
                    # (init = ONE_LANE — own arrive only).
                    bars.mb_p_xfer_full[nlmma_state.idx].arrive(n_bytes=pXferBytes, pred=nvvm.elect_sync())
                    bars.mb_p_xfer_full[nlmma_state.idx].wait(nlmma_state.phase)
                    cur_parity = nlmma_state.idx
                    nlmma_state = advance(nlmma_state, CFG.XFER_STAGES)
                    # DSMEM-arrive on leader's mb_p_xfer_full[parity] — leader's
                    # init = CTA_MMA (own + this forwarded arrive), so wait
                    # flips only when both peers' bytes have landed (P12).
                    bars.mb_p_xfer_full[cur_parity].arrive_on_peer(leader_cta_id, pred=nvvm.elect_sync())

            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

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
            if cutlass.const_expr(CFG.MASK_FLAGS != 0):
                eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
                eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
                bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
                kv_left = bounds_next.left
                kv_right = bounds_next.right

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# ============================================================================
# TMA-LDG warp group — sg-conditional Q+K (sg0) or V (sg1).
# Port of C++ tmaldg_warp_group (prefill_sdpa_d512_fp8.cu:1241-1385).
# ============================================================================
@cute.jit
def _tmaldg_warp_group(
    is_sg0,
    is_sg1,
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    o_desc_words,
    sQ,
    sK,
    sV,
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
):
    """sg0: Q + K TMA-LDG.  sg1: V TMA-LDG.

    Both sgs decode the persistent-tile scheduler payload.  Leader of each
    pair issues expect_tx for the collective TMA byte count; both peers
    issue the cta_group::2 multicast TMA (mcast_mask bits = bit_per_peer).

    sg0 (port of C++ lines 1265-1338):
        Per tile, if n_kv > 0:
            wait(q_empty, q_empty_phase); leader arrive_expect_tx(q_full, qBytes);
            tma_load_tile(sQ[0], q_gmem, q_full, multicast).
            For each kv:
              wait(k_empty[stage]); leader arrive_expect_tx(k_full[stage], kBytes);
              tma_load_tile(sK[stage], k_gmem(kv * TILE_N + k_row_off), k_full, multicast).
        q_empty pre-armed via q_empty_state=1 (iter-0 wait passes on phase 0 != 1).

    sg1 (port of C++ lines 1339-1397):
        Per tile, if n_kv > 0, per kv:
          wait(v_empty[stage]); leader arrive_expect_tx(v_full[stage], vBytes);
          tma_load_tile(sV[stage], v_gmem(v_col_off + cta_in_pair * (TILE_O/CTA_MMA) * BPE,
                                        kv * TILE_N), v_full, multicast).
        Note: C++ uses BYTE coord for V because UINT8 TMA descriptors; the DSL's
        typed descriptor uses ELEMENT coord — so multiply only when descriptor
        is UINT8.  Current DSL descriptors are element-typed, so plain cta_in_pair *
        (TILE_O/CTA_MMA) suffices.

    Both sgs drain trailing K/V/Q empty arrives at end-of-kernel under cga2.
    """
    q_empty_phase = cutlass.Int32(1)  # pre-armed (iter-0 wait passes)
    kv_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's packed-total-CLAMPED runtime
        # descriptors (o_desc_words slots n_batch+1 / n_batch+2).  The plan-time
        # descriptors describe the buffer's CAPACITY, so the last sequence's
        # tile-tail would read caller-owned bytes past cu_k[B]; a NaN there wipes
        # the tile through BMM2 (0 * NaN == NaN).  Clamped, those rows are
        # TMA-OOB and land as EXACT ZEROS.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

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
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # Per-peer K row offset (sg0 only — sg1's V offset is along d_v).
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    # Per-peer V d_v offset (sg1 only — sg0's K offset is along seq).
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # n_kv > 0 gate at MASK_NONE collapses to "always" since kv_right > 0.
        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            pass
        else:
            if is_sg0:
                # ---- sg0: Q (one-shot per tile) + K_ring ----------------
                bars.mb_tma_q_empty.wait(q_empty_phase)
                q_empty_phase = q_empty_phase ^ cutlass.Int32(1)
                bars.mb_tma_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                tma_load_tile(
                    sQ[0],
                    tma_q(cutlass.Int32(0), head_idx, q_row_base + q_seq_off, tma_batch),
                    bars.mb_tma_q_full.smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=None,
                )

                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                    bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                    bars.mb_tma_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                    tma_load_tile(
                        sK[kv_state.idx],
                        tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                        bars.mb_tma_k_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=None,
                    )
                    kv_state = advance(kv_state, CFG.STAGES_KV)
            else:
                # ---- sg1: V_ring only -----------------------------------
                # V split along d_v: per-CTA inner = TILE_O / CTA_MMA.  the pre-upstream DSL
                # TMA descriptors are element-typed so the per-peer offset
                # is in elements (NOT bytes — C++ uses bytes because its
                # TMA descriptors are UINT8-typed).
                for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                    bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                    bars.mb_tma_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                    tma_load_tile(
                        sV[kv_state.idx],
                        tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                        bars.mb_tma_v_full[kv_state.idx].smem_ptr,
                        cta_group=CFG.CTA_MMA,
                        mcast_mask=None,
                    )
                    kv_state = advance(kv_state, CFG.STAGES_KV)

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
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # cga2: drain trailing K/V/Q empty arrives so SMEM isn't torn down while
    # leader's multicast commits are still in-flight.
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        if is_sg0:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_k_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
            bars.mb_tma_q_empty.wait(q_empty_phase)
        else:
            for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_tma_v_empty[kv_state.idx].wait(kv_state.phase)
                kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# ============================================================================
# TMA-STG warp — sg1 only: O TMA store + LSE write back.
# Port of C++ tmastg_warp_group (prefill_sdpa_d512_fp8.cu:1392-1453).
# ============================================================================
@cute.jit
def _tmastg_warp_group(
    is_sg1,
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
    """sg1: persistent O-store + per-chunk mb_tma_o_empty drain.

    sg0 TMASTG (port of C++ lines 1417-1428): idle — just spin scheduler so
    persistent tile claims advance in lockstep with sg1.  Does NOT call
    read_tile_id_arrive (per the READ_TILE_ARRIVERS = 25 derivation — sg0's
    TMASTG slot is NOT in the arriver list).

    sg1 TMASTG (port of C++ lines 1430-1467):
        Per tile:
          - read_tile_id_arrive on the scheduler's read_tile_id mbar.
          - For each chunk c in [0, N_O_CHUNKS):
              wait(mb_tma_o_full[c], epilogue_state).
          - tma_store_tile(sO[0], o_gmem(0, q_row_coord, head_idx, batch_idx)).
          - tma_store_commit; tma_store_wait(0).
          - arrive(mb_tma_o_empty).
          - Flip epilogue_state.
    """
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
        # sg0's TMA-STG slot is idle but spins the scheduler so persistent
        # tile claims advance in lockstep with sg1.  sg0 does NOT call
        # read_tile_id_arrive (per the READ_TILE_ARRIVERS=25 derivation —
        # sg0's TMA-STG slot is intentionally NOT in the arriver list).
        if is_sg1:
            read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

            # Wait every chunk of O ready.
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_tma_o_full[chunk].wait(o_full_phase)

            # Both sg1 peers issue the O TMA store with their own q_row_coord —
            # cga2 splits C along M, so leader writes rows [0:128] and peer
            # writes [128:256].
            q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
            if cutlass.const_expr(CFG.THD_VARLEN):
                # THD: store through this batch's pre-built descriptor (base at
                # the sequence's packed row, seq extent = S_q_b → box past S_q_b
                # OOB-clipped).  q_row_coord sequence-local; batch coord → 0.
                # DEAD unit (batch_idx == n_batch, the over-launched persistent grid's
                # sentinel): no O rows exist and descriptor slot n_batch is never
                # built, so skip only the store -- the barrier protocol still runs.
                if batch_idx < n_batch:
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

            bars.mb_tma_o_empty.arrive()
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)

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


# ============================================================================
# Host launcher.
# ============================================================================
@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    # FROST's quantized ABI: the four 1-element fp32 scales arrive as DEVICE
    # TENSORS and the amax of O is an OUTPUT.  The pre-upstream contract pre-folded on the host
    # and produced no amax.  Template: sm107/prefill_d128_fp8.py (same lineage).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
    # FROST plans must run on the caller's stream (engine contract; there is a
    # dedicated stream-respect test).  Threaded exactly as the shipped
    # sm107/prefill_d128_fp8.py sibling does.
    # THD device metadata build: the CALLER's Q/KV length tensors, (B,) lengths
    # or (B+1,) cu prefix sums per side via thd_lens_form.  Consumed only by the
    # setup kernel, which writes the metadata buffer device-side.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    # FROST passes the dense padded-Q trim tensor POSITIONALLY on every call.
    # These kernels have no Q-trim epilogue (Capabilities.dense_seq_q_trim=False)
    # so it is accepted and unused -- but the SLOT is mandatory: without it the
    # adapter's 12th positional lands on `stream` and execute dies with
    # "got multiple values for argument 'stream'".
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are RUNTIME values: the adapter passes 0 in the
        # problem_size seq slots by THD contract.  Without this the kernel runs
        # at SQ = SKV = 0 -- descriptor extents, loop bounds and the tile count
        # all collapse, and the output is uniformly wrong with no crash.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    # Tensors are [B, S, H, D] with stride_order=(3, 2, 1, 0); D is fastest.
    # Under cga2 K is split along seq (per-CTA box rows are TILE_N / CTA_MMA),
    # V is split along d_v (per-CTA inner is TILE_O / CTA_MMA).  O's TMA box
    # inner follows O's swizzle, NOT V's (V may drop to narrower under cga2).
    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

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

    # Each cluster (CGA_M = 4 CTAs; CTA_MMA = 2 sg pairs collectively share Q)
    # covers CGA_TILE_M = TILES_Q * TILE_M * CTA_MMA = 256 Q rows.  grid_x
    # must scale by CGA_M (cluster x-dim) so the cluster shape divides the
    # grid — for non-dsv4 flavors CGA_M == CTA_MMA and this collapses to the
    # historical formula.  Mirrors C++ runner
    # (the pre-upstream host runner).
    rows_per_cluster = CGA_TILE_M
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CGA_M
    # n_q_supers passed to kernel is the LOGICAL q-tile count (not the grid x
    # span).  For SCHED_LPT it's `q_clusters * CTA_MMA` (each cluster owns
    # CTA_MMA rows of distinct q_super_idx values); for NATURAL the kernel
    # reads bidx directly so this value is unused.
    q_supers = q_clusters * CFG.CTA_MMA
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the per-batch O descriptor array (reuse tma_o_desc over the
        # packed [1,T,QH,D_v] O as base), then launch the exact flat
        # batch-outermost grid (n_thd_units host-computed); grid_x = units*CGA_M.
        _build_thd_meta_o_descs_kernel(
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
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (grid_q_supers, QH, B) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B, 1, 1)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse_tensor,
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
        o_scale_fused,
        descale_q_t,
        descale_k_t,
        descale_v_t,
        scale_o_t,
        amax_o_tensor,
    ).launch(
        # cluster = (CGA_M, CGA_N, 1) = (4, 1, 1) under dsv4.
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 128,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    # The f16 THD compile key carries the caller's DECLARED packed strides
    # (api_dsl._thd_compile_kwargs) and the ragged-Stats layout.
    q_stride: Optional[tuple] = None,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    o_stride: Optional[tuple] = None,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple] = None,
) -> Callable:
    """Compile a kernel with ALL dims concrete to pin TMA descriptor strides at compile time.

    THD/varlen: q/k/v/o/lse PACKED with batch dim 1 ([1,T,H,D]); ``b`` is the
    LOGICAL batch (sequence count) driving n_batch / metadata + O-desc sizes."""
    # THD/varlen is NOT ported for this kernel yet.  The setup-kernel call site
    # below still speaks the pre-upstream 7-arg contract, while FROST's
    # thd_helpers.build_thd_meta_o_descs_kernel takes 14 args and a different
    # metadata layout (4B+4 with batch_remap + a claim counter, vs the 3B+2
    # here), and the scheduler decode differs to match.  Raise here rather than
    # let it fail as an arity error deep in the trace -- and so no engine row
    # can advertise thd=True for this kernel and appear to work.  The dense
    # path is unaffected: CFG.THD_VARLEN is 0 and every THD branch folds out.
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
    if CFG.THD_VARLEN:
        # Dynamic packed token totals: a new packed total RE-BINDS the same
        # artifact instead of minting one per shape.
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)

    def _fake_bshd(shape, stride, dtype=STORAGE_DTYPE, bpe=CFG.BPE):
        """BSHD fake tensor, compact or at the caller's DECLARED strides.

        The head dim must be innermost-contiguous, and the seq/head global
        strides feed TMA so they obey the 16-byte global-stride rule.  Under THD
        the fake binds the token stride for the extent-1 batch dim, as _thd_view
        does at runtime: tokens * token_stride is never stepped and overflows
        the int32 stride slot on long packed KV (GitHub #980)."""
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if stride[3] != 1:
            raise ValueError(f"declared stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
        for axis in (1, 2):
            if (stride[axis] * bpe) % 16 != 0:
                raise ValueError(f"declared stride {stride} axis {axis} must be a 16-byte multiple at BPE={bpe} (TMA global-stride rule)")
        if CFG.THD_VARLEN:
            # Extent-1 batch dim: bind the token stride, as _thd_view does at
            # runtime -- T * token_stride is never stepped and overflows the int32
            # stride slot on long packed KV with wide tokens (GitHub #980).
            return cute.runtime.make_fake_tensor(dtype, shape, (stride[1], stride[1], stride[2], stride[3]), assumed_align=16)
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd((_fake_batch, sq, qh, d_qk), q_stride)
    fake_k = _fake_bshd((_fake_batch, skv, kh, d_qk), k_stride)
    fake_v = _fake_bshd((_fake_batch, skv, kh, d_v), v_stride)
    fake_o = _fake_bshd((_fake_batch, sq, qh, d_v), o_stride, dtype=OUT_STORAGE_DTYPE, bpe=CFG.BPE_O)
    # has_lse=False (no Stats output): the LSE argument is None-specialized and
    # the store is compiled out entirely -- no dummy buffer exists at any level,
    # which is what lets the dense graph report get_workspace_size() == 0.
    # Mirrors the shipped sm107/prefill_d128_fp8.py.
    if not has_lse:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        # Packed ragged Stats in the CALLER's declared layout; the epilogue store
        # branches on the STATIC rank, so the layout is fully encoded here.
        if lse_head_major:
            _lse_hs = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1, qh, _lse_hs), stride_order=(2, 1, 0), assumed_align=4)
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only (token-major (T, H) is compact)")
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (sq, qh), stride_order=(1, 0), assumed_align=4)
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are THD-only (dense LSE is compact (B, H, Sq))")
        fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (_fake_batch, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    # THD overloads seq_kv_lens as [seq_kv_lens(B)|cu_q(B+1)|cu_k(B+1)] (len 3B+2).
    _skv_len = (4 * b + 4) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Per-batch O TMA-descriptor array (16 int64 = 128 B each) + 1 pad slot.
    _odesc_len = ((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )

    # The four 1-element fp32 device scales + the amax_o output slot; the
    # compiled artifact must declare them or the adapter's positional call
    # cannot bind (FROST quantized ABI).
    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (1,),
            stride_order=(0,),
            assumed_align=4,
        )

    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Caller Q/KV length tensors, read only by the setup kernel.  DYNAMIC extent:
    # the accepted forms differ in length ((B,) vs (B+1,)) and which one arrives
    # rides the runtime thd_lens_form bitmask.
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = fake_thd_kv_lens = fake_thd_lens_form = None
    return cute.compile(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        fake_amax_o,
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        # seq_q_lens_tensor is accepted-but-unused AND LAST on this flavor's ABI:
        # the adapter sends a leading None only for d256 quantized, so d512 takes
        # the slot after the THD lens args (mirrors sm100/prefill_d512_fp8.py).
        None,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )
