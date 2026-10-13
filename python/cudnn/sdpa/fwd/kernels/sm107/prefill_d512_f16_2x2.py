# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin SM107 bf16/fp16 d=512 forward on the 2x2 DATAPATH (TemplateParams.mma_2x2).

The SM100 body (sm100/prefill_d512_f16_2x2.py -- the per-arch sibling-file convention of every flavor; a diff against
it is the review surface) with the Rubin deltas of the design (section 13), re-probed on a cc 10.7 board (
2026-10-01: the 2SM M=128 lane map, the TS duplication and the full-rate 2x2 atoms are IDENTICAL to the B200 facts):
  * 3-deep K and V sub-chunk rings (config_sm107.make_cfg_d512_2x2: STAGES_K_SUB = STAGES_V_SUB = 3, one extra KV
    iteration of TMA allowance; the O staging keeps aliasing the V ring).
  * EVERY SmemTile takes the module constant DESC_VERSION (= 1) as its descriptor version: the layout puts the P ring
    at exactly 262144 B, the 256 KiB window of a version-0 tcgen05 descriptor (the 2026-09-04 silent-zero bug class).
  * O u V alias gate made PAIR-WIDE across the twins (mirrors fix-lane FATAL-1 of the SM100 body): the TMA-STG warp
    arrives on its own ``mb_o_empty`` AND on the twin's (``arrive_on_peer(cta_id_x ^ 2)``), init ONE_WARP x KV_SHARE,
    so neither twin's first V(t+1) multicast (which lands in BOTH twins' sVO) can be issued before BOTH O(t) stores
    have read their staging; the TMA-LDG drains the final phase at kernel end so the twin's remote arrive never
    targets an exited CTA.
  * The SMEM budget is checked against config_sm107.SMEM_USABLE_BYTES (320 KiB), never the 327 KiB capacity.
  * The dense (unmasked) softmax arm reads S with ``tmem_load_max_reduction_tile(num_elems=64)`` = one
    ``tcgen05.ld.red.max`` (cc 10.7 has it; cc 10.0 does not) -- its result is the HALF-row max, so the lane-half
    exchange below is still mandatory; the masked arms keep ``tcgen05.ld`` + ``apply_mask_chunk`` + the software max.
  * ``SPIN_RING_WAITS`` on every per-KV-iteration ring wait (the sm107 discipline, pinned by
    test_sm107_ring_waits_take_the_module_spin_constant); the whole-tile idle waits and the drains keep the default.
  * TMEM: one 512-column cta_group::2 allocation (388 used) -- the 576-col ``is_exclusive`` form is not needed.
  * TILE_K_HW stays 16 for f16/bf16 (config_sm107.tile_k_hw; the K=64 2-chunk form is the FP8 QMMA path).
The O_STORE_STREAM / O_EPI_PIPELINE / CORR_READY_BEFORE_DONE levers are module constants here too (both arms trace).
Softmax lever, per plan from TemplateParams: ``SCALE_PREFOLDED`` (softmax_scale_prefolded / graph.sdpa
``attn_scale_prefolded``) -- Q carries attn_scale * log2 e, so the softmax takes the RAW row max and shifts with
S - m (two const_expr sites in ``_softmax_kv_iter``; the scaled chain stays the default and ``scale_softmax_log2``
stays in every signature).  The f16x2 exponent arm (softmax_f16) is a quantized-kernel specialization: declined here.

ONE pipeline per CTA on ``tcgen05.mma.cta_group::2`` with collective M = 128 = 64 Q rows per
CTA, every CTA running TMA-LDG(Q,K,V) + BMM1 + softmax + BMM2 + correction + epilogue + TMA-STG
for its own 64 rows.  Cluster (CGA_M, 1, 1) = CGA_M // 2 cta_group::2 pairs {0,1},{2,3}; CTA c
owns q rows [cluster*CGA_M*64 + 64c, +64); CTAs c and c ^ 2 ("twins", same cta_in_pair) share
every K/V sub-chunk by TMA multicast (each issues half of each 32 KiB sub-chunk with mask
(1 << c) | (1 << (c ^ 2)); KV_SHARE = 2).  The CGA_M=2 bring-up arm is the same body with
KV_SHARE=1 and own-bit loads (FROST_D512_2X2_CGA_M injected next to FROST_TEMPLATE_PARAMS).

The 2x2 accumulator atom (probe_layout, 2026-09-30): fp32 D row m, column n of a 2SM M=128
MMA lands on TMEM lane (m % 64) + 64 * (n // (N/2)), column n % (N/2) -- warps w and w + 2
hold the SAME 64 rows, one column half each.  Thread (r, h) = (tid & 63, tid >> 6) of each
compute warpgroup therefore owns row r's columns of half h:
  S parity p   : cols [256 + 64p, +64)  = kv [64h, +64) of the 128-wide tile      (ONE 32x32b x64 ld)
  O            : cols [0, 256)          = d_v [256h, +256)                        (BMM2 N-block c at cols [128c, +128))
  alpha[p]     : col 384 + p (both halves write the SAME value); tile stats: cols 386 (max) / 387 (sum)
Per-row max / sum are HALF-row values until exchanged with lane r + 64 through SMEM (sXchgMax
per parity, sXchgSum at tile end) and a 128-thread named barrier -- both halves then run the
identical scalar chain (RESCALE_THRESHOLD, row_max_for_exp2, alpha, beta, lse), so alpha / beta
agree bitwise across the lane halves.  P is written lane-wise into SMEM (K-major SW128, one
128-B row segment per lane into subtile h) and read IN PLACE across the pair by the leader's
cta_group::2 BMM2 after fence.proxy.async + a per-lane .release.cta arrive on the leader's
mb_p_full (probe release_p: 0/1920 mismatches, SASS MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S;
SYNCS.ARRIVE).  Q is SMEM-resident (SS BMM1; a TS A operand under 2SM M=128 is DUPLICATED
and saves nothing), K comes in 32 KiB d-half sub-chunks (SS B), V in 32 KiB 128-col sub-chunks
(MN-major SS B, LBO 16384 / SBO 1024: LBO 0 silently corrupts the second 64-d subtile).

SMEM (SM107, 297880 B incl. the 1008 B base pad vs the 327680 B usable line): sQ 64 KiB | sK ring
3 x 32 KiB | sV ring 3 x 32 KiB u sO staging 64 KiB | sP ring 2 x 16 KiB at 262144 B | sXchgMax
1 KiB | sXchgSum 512 B | 39 mbarriers + scheduler.  TMEM: one 512-col cta_group::2 alloc per CTA.
The mbarrier ledger is the docstring of ``_common_blackwell.make_d512_2x2_bars``; every init
count is a Cfg constant pinned by test_sdpa_fwd_d512_2x2_sm100.py / test_sdpa_fwd_dsl_sm107.py.
"""

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

from cudnn.sdpa.fwd.config_sm100 import TemplateParams, CfgD512X2, d512_2x2_smem_bytes, d512_2x2_p_ring_start_bytes
from cudnn.sdpa.fwd.config_sm107 import make_cfg_d512_2x2, SMEM_USABLE_BYTES, TCGEN05_V0_ADDR_LIMIT

# The per-graph params are injected as a module global by the loader before this body executes
# (api_dsl._load_sm100_kernel_module's rubin arm routes mma_2x2=True records here); a plain import
# gets the all-defaults 2x2 config (dense fp16).  FROST_D512_2X2_CGA_M (4 = shipped twin-multicast
# arm, 2 = bring-up / bitwise-twin arm) is a second loader-style global, NOT a knob: a test that
# wants the 2-CTA arm execs this file with it set, exactly as the loader sets FROST_TEMPLATE_PARAMS.
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(mma_2x2=True))
CGA_M_ARM: int = int(globals().get("FROST_D512_2X2_CGA_M", 4))
CFG, _TMA = make_cfg_d512_2x2(PARAMS, cga_m=CGA_M_ARM)
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS


def _require(cond, msg):
    """Geometry sanity check; raises instead of assert (asserts vanish under -O)."""
    if not cond:
        raise ValueError(f"prefill_sdpa_d512_f16_2x2_sm107: {msg}")


_require(isinstance(CFG, CfgD512X2), f"expected a CfgD512X2, got {type(CFG).__name__}")

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point, wired into every
# construction below rather than repeated as a per-tile literal (the sm107 discipline; the d512 MXFP8 sibling shipped
# NaN on 100 % of cells from one re-literalled tile).  A version-0 descriptor's start_address is 14 bits = a 256 KiB
# window; this layout puts the P ring (BMM2's A operand) at exactly 262144 B, so a version-0 descriptor would wrap to
# offset 0 and BMM2 would multiply the bottom of SMEM (O comes out EXACTLY zero, no crash).  The Cfg derives the value
# from the layout (config_sm107._d512_2x2_desc_version_for); the module constant is what the tiles read, and the two
# must agree (test_sm107_descriptor_version_matches_the_smem_budget pins the constant against _NEEDS_DESC_V1).
DESC_VERSION: int = 1
_require(CFG.DESC_VERSION == DESC_VERSION, f"Cfg.DESC_VERSION {CFG.DESC_VERSION} disagrees with the module constant {DESC_VERSION}")
_require(
    d512_2x2_p_ring_start_bytes(CFG) >= TCGEN05_V0_ADDR_LIMIT,
    f"the P ring starts at {d512_2x2_p_ring_start_bytes(CFG)} B, below the version-0 window: DESC_VERSION=1 would then be a silent widening",
)
# Retry form of the per-KV-iteration RING waits (k/v/q _full/_empty, bmm1_done / bmm2_ready / bmm2_done / p_full,
# stat_*, o_empty, empty_mainloop): every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``; the waits a warp
# parks in for a whole tile (the scheduler payload wait of every role, mb_tmem_dealloc, the TMA-STG's mb_o_full) and the
# end-of-kernel drains keep the default sleeping form.  ``spin=True`` is the hint-less uniform spin
# (tile_dsl.barrier.wait); like DESC_VERSION it lives here once and never as a literal at a call site
# (test_sm107_ring_waits_take_the_module_spin_constant).  Carried over from the role-split sibling's measured value
# (+1.1 / +1.6 % at S=8K dense / causal there); THIS kernel's own A/B/A is owed (lane notes).
SPIN_RING_WAITS: bool = True
# CROSS-PAIR barriers are POLLED, never parked (mirrors the fix-lane poll-wait rule, bprop fix @ bd6bed12c, 2026-10-01).  A
# barrier whose completing event is issued from OUTSIDE the pair -- under KV_SHARE=2: mb_k_empty / mb_v_empty (both pair
# leaders' tcgen05.commit multicast 0xF, waited by the TMA-LDG warp), mb_k_full / mb_v_full (the twin's TMA complete_tx on
# the leader's copy, waited by the leader MMA warp) and, under the pair-wide O u V gate, mb_o_empty (the twin's remote
# arrive_on_peer) -- is waited with the NON-BLOCKING ``mbarrier.test_wait.parity`` poll (_poll_wait), every site including
# the end-of-kernel drains.  MEASURED by the d512 backward 2x2 lane under GPU time-slicing: tile_dsl wait() (try_wait.parity +
# time_limit, NANOSLEEP.SYNCS) hung at launch 2/300, the hint-less spin (wait(spin=True)) at 74/200 -- ANY try_wait variant can
# park the warp and miss the wake-up of an event from outside the pair -- while the test_wait poll ran 200/200 + 300/300.
# Pair-local / CTA-local barriers keep tile_dsl wait() with the SPIN_RING_WAITS lever.  POLL_CROSS_PAIR_WAITS is a correctness
# constant (the module refuses to trace with it False), not a lever; the sm107 structural test counts the _poll_wait sites.
POLL_CROSS_PAIR_WAITS: bool = True
_require(POLL_CROSS_PAIR_WAITS is True, "cross-pair-released barriers must be polled with mbarrier.test_wait.parity (fix-lane poll-wait rule)")
# The poll's SHAPE, handed to the shared tile_dsl ``barrier.wait_poll`` (``tight_iters`` back-to-back ``test_wait``s, then a
# timer ``nanosleep.u32 sleep_ns`` between tests; ``sleep_ns = 0`` = the pure tight loop).  The forward's pollers (TMA-LDG,
# leader MMA, correction) wait briefly, so the shape is immaterial here -- MEASURED on the board (H128 S8192 bf16, one clean
# slot each, cudaEvent medians of 30 trials): tight 1 / 0 -> 5138.0 us dense / 2842.3 us causal; 32 / 128 -> 5136.4 / 2846.4
# (within 0.15 %, inside slot noise; the SM100 2x2 forward measured its tight loop 0.3-1.1 % faster than 32 / 128, also at
# noise).  Both forwards ship the shared default 32 / 128; the backward's 2.2x tight-poll loss (long waits on an SMSP shared
# with the softmax warps) is why the shape exists at all, and it ships 128 / 128.  The 12 x 100 contention hygiene and the
# +21.1 % / +18.1 % headline were taken with the tight loop; the hang-safety argument is the same for both shapes (a timer
# NANOSLEEP never parks on the barrier).
POLL_TIGHT_ITERS: int = 32
POLL_SLEEP_NS: int = 128

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    wait,
    wait_poll,
    cga_arrive,
    cga_wait,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
)
from cudnn.frost.tile_dsl.pointwise import (
    row_reduction_pair,
    row_max_reduction,
    tmem_load_max_reduction_tile,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile
from cudnn.frost.tile_dsl.mma import mma_ss, desc_opaque
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_load_subtiles,
    tma_store_tile,
    tma_store_subtile,
    tma_store_commit,
    tma_store_wait,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import apply_mask_chunk, MASK_NONE

from cudnn.sdpa.fwd.kernels._common_blackwell import (
    sdpa_operand_tensors,
    make_split_helpers,
    make_d512_2x2_bars,
    row_max_for_exp2,
    make_sdpa_helpers,
)

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
elif CFG.DTYPE_QKV == 3:
    STORAGE_DTYPE = cutlass.Float16
    MMA_KIND = nvvm.Tcgen05MMAKind.F16
else:
    raise ValueError(f"prefill_sdpa_d512_f16_2x2 (SM100): DTYPE_QKV={CFG.DTYPE_QKV} not supported (BF16=2 / FP16=3 only)")
P_STORAGE_DTYPE = STORAGE_DTYPE
OUT_STORAGE_DTYPE = STORAGE_DTYPE

# softmax_scale_prefolded (graph.sdpa ``attn_scale_prefolded``): the caller multiplied Q by attn_scale * log2(e), so the
# raw QK^T already sits in the log2 domain.  The softmax then takes the RAW row max (no FMUL by scale_log2) and shifts
# with S - m (an FADD2 for the FFMA2) -- two const_expr sites in _softmax_kv_iter; nothing else moves: the published
# Stats, the sink fold, the dead-row select and the RESCALE_THRESHOLD compare consume the log2-domain max they consumed
# before.  Numerically neutral by construction: under the contract the adapter pins scale_log2 to exactly 1.0, so the
# elided multiplies were exact identities (``scale_softmax_log2`` stays in every signature as a dead runtime argument --
# the direct-ABI oracle launches the fold with a garbage value and expects the scaled chain's bits).  Both softmax arms
# mask with true -inf, so a fully-masked iteration leaves the raw max at -inf exactly as the scaled chain does.
SCALE_PREFOLDED = int(PARAMS.softmax_scale_prefolded)
# softmax_precision=HALF (TemplateParams.softmax_f16) is a quantized-kernel arm; the half kernels keep the f32 exponent.
# config_sm107 declines it before this file loads -- the guard keeps the request from ever tracing the default chain.
_require(not PARAMS.softmax_f16, "softmax_f16 is a quantized-kernel (FP8 / MXFP8) specialization; half inputs run the f32 exponent")

# ---------------------------------------------------------------------------- module-constant levers
# (measured per kernel, never knobs: a value must not differ per plan)
# Stream the O store subtile by subtile behind the epilogue (tma_store_subtile) instead of one
# whole-tile store after all eight chunks.
O_STORE_STREAM: bool = True
# Ballot-first correction: when every lane of the warp has alpha == 1 the two bmm2_ready credits
# are arrived BEFORE the unconditional bmm2_done(i-1) wait (no rescale needed, so BMM2(i) may
# start early); the slow arm waits first, rescales, then credits per N-block.
CORR_READY_BEFORE_DONE: bool = True
# Epilogue: issue the next 32-col TMEM batch's tcgen05.ld before processing the current one.
O_EPI_PIPELINE: bool = True

CGA_SIZE = CFG.CGA_M * CFG.CGA_N
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2
KV_SHARE = CFG.KV_SHARE
_require(CFG.CTA_MMA == 2 and CFG.CGA_N == 1, "2x2 datapath: cta_group::2 pairs, CGA_N == 1")
_require(KV_SHARE in (1, 2) and KV_SHARE * CFG.CTA_MMA == CFG.CGA_M, "KV_SHARE must be CGA_M // CTA_MMA in {1, 2}")
# The O u V alias gate is PAIR-WIDE: the TMA-STG warp below arrives on its own mb_o_empty and (KV_SHARE=2) its twin's.
_require(
    CFG.O_EMPTY_ARRIVERS == CFG.ONE_WARP * KV_SHARE, f"mb_o_empty init must be ONE_WARP x KV_SHARE = {CFG.ONE_WARP * KV_SHARE}, got {CFG.O_EMPTY_ARRIVERS}"
)

# ---------------------------------------------------------------------------- geometry (elements)
qBufferElems = CFG.TILE_M * CFG.TILE_K  # 64 x 512 resident Q
K_SUB_ROWS = CFG.TILE_N // CFG.CTA_MMA  # this CTA's 64 kv rows of a K tile
K_SUB_COLS = CFG.TILE_K // 2  # one d-half sub-chunk = 256 d
kSubElems = K_SUB_ROWS * K_SUB_COLS  # 32 KiB
V_SUB_COLS = CFG.TILE_O // CFG.CTA_MMA // CFG.N_BMM2_CHUNKS  # this CTA's 128 d_v cols of one N-block
vSubElems = CFG.TILE_N * V_SUB_COLS  # 128 kv rows x 128 d_v = 32 KiB
oBufferElems = CFG.TILE_M * CFG.TILE_O  # 64 x 512 staging
pSlotElems = CFG.TILE_M * CFG.TILE_N  # 64 x 128 P
_require(CFG.STAGES_V_SUB * vSubElems * CFG.BPE >= oBufferElems * CFG.BPE_O, "sO staging must fit inside the V ring it aliases")

# Bytes landing in the PAIR per barrier phase (expect_tx on the pair leader; P9 routing: every
# cta_group::2 tensor load counts its bytes on the destination's pair leader).
qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kSubTransactionBytes = kSubElems * CFG.BPE * CFG.CTA_MMA
vSubTransactionBytes = vSubElems * CFG.BPE * CFG.CTA_MMA

# TMA subtiles: Q 8 x (64 rows x 64 d) = 8 KiB; K sub-chunk 4 x (64 rows x 64 d) = 8 KiB;
# V sub-chunk 2 x (128 rows x 64 d_v) = 16 KiB; O 8 x (64 rows x 64 d_v) = 8 KiB.
K_SUBTILES_PER_SUB = K_SUB_COLS // TMA_QK_GRANU_ELEMS  # 4
V_SUBTILES_PER_SUB = V_SUB_COLS // TMA_VO_GRANU_ELEMS  # 2
_require(K_SUBTILES_PER_SUB % KV_SHARE == 0 and V_SUBTILES_PER_SUB % KV_SHARE == 0, "every K/V sub-chunk must split evenly across the KV_SHARE issuers")
K_SUBTILES_PER_ISSUER = K_SUBTILES_PER_SUB // KV_SHARE  # 2 (KV_SHARE=2) / 4
V_SUBTILES_PER_ISSUER = V_SUBTILES_PER_SUB // KV_SHARE  # 1 / 2
Q_SUBTILE_STRIDE_ELEMS = CFG.TILE_M * TMA_QK_GRANU_ELEMS  # 4096
K_SUBTILE_STRIDE_ELEMS = K_SUB_ROWS * TMA_QK_GRANU_ELEMS  # 4096
V_SUBTILE_STRIDE_ELEMS = CFG.TILE_N * TMA_VO_GRANU_ELEMS  # 8192
TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O  # 64
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES  # 8
O_SUBTILE_STRIDE_ELEMS = CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST  # 4096
N_O_CHUNKS = TMA_O_ITERS_HOST  # 8 x 8 KiB O subtiles, one mb_o_full each
_require(N_O_CHUNKS == 8, f"the O drain mapping assumes 8 subtiles, got {N_O_CHUNKS}")

# P: lane (r, h) writes 64 values = 128 B = one SW128 row segment at elem offset h*4096 + r*64.
P_SUBTILE_ELEMS = CFG.TILE_M * 64  # 4096: one 8 KiB subtile = 64 rows x 64 kv cols
P_ROW_ELEMS = 64
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_O_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)

# Softmax: each lane holds 64 S columns (one half of the 128-wide tile).
SOFTMAX_COLS = CFG.TILE_N // 2  # 64
# BMM2: 8 k-steps of 16 kv; k-steps 0-3 read P subtile 0 (kv 0..63), 4-7 subtile 1.
NUM_KPHASES_PV = CFG.TILE_N // CFG.TILE_K_HW_BMM2  # 8
NUM_KPHASES_QK_SUB = K_SUB_COLS // CFG.TILE_K_HW_BMM1  # 16 per d-half
STAT_STAGES = 2  # alpha / tile-stats ring depth (cols 384 + idx)

NEG_INF_F32 = cutlass.Float32(float("-inf"))
RESCALE_THRESHOLD_F32 = cutlass.Float32(CFG.RESCALE_THRESHOLD)

# Rows per cluster / THD unit (the setup kernel's rows-per-cluster argument; 256 at CGA_M=4).
CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CGA_M * CFG.CGA_N
_require(CGA_TILE_M == CFG.ROWS_PER_CLUSTER, "CGA_TILE_M must equal Cfg.ROWS_PER_CLUSTER")
CTA_MMA = CFG.CTA_MMA
THD_PERSISTENT = True

# ---------------------------------------------------------------------------- SMEM budget
# Rubin: validate against the 320 KiB USABLE line (config_sm107.SMEM_USABLE_BYTES), never the 327 KiB capacity --
# overflowing it does not fail the launch, it clobbers the last buffer allocated (d192 f16: O 50 % zeros, exact LSE).
_SMEM = d512_2x2_smem_bytes(CFG, n_o_chunks=N_O_CHUNKS)
_require(CFG.SMEM_CAP_BYTES == SMEM_USABLE_BYTES, f"the Cfg budget line {CFG.SMEM_CAP_BYTES} is not the Rubin usable carveout {SMEM_USABLE_BYTES}")
_require(
    _SMEM["total"] <= SMEM_USABLE_BYTES,
    f"SMEM {_SMEM['total']} B (incl. {CFG.SMEM_ALIGN_PAD} B pad) exceeds the {SMEM_USABLE_BYTES // 1024} KiB usable Rubin carveout: {_SMEM}",
)
# Streamed O store: one 8 KiB subtile per mb_o_full chunk (the SASS pin's expected longest UTMASTG run).
_O_SUBTILES_PER_CHUNK = TMA_O_ITERS_HOST // N_O_CHUNKS
_require(_O_SUBTILES_PER_CHUNK == 1, "the O drain publishes one TMA subtile per chunk")


# ---------------------------------------------------------------------------- TMEM map
@dataclass(frozen=True)
class KernelTmemLayout:
    TOTAL_COLS: int = 512
    O_OFF: int = 0
    O_COLS: int = 256  # 64 x 512 fp32 in the 2x2 atom: lane r + 64*(d_v >> 8), col d_v & 255
    S_ACC_OFF: int = 256  # parity p at S_ACC_OFF + 64p: lane r + 64*(kv >> 6), col kv & 63
    S_ACC_COLS: int = 64
    ALPHA_OFF: int = 384  # + stat ring idx
    # Tile stats (total_max_safe, final_sum) PER RING SLOT: cols STATS_OFF + 2*idx, +1.  The SM100 body keeps them in the two
    # FIXED columns 386 / 387; that is a race (FOUND on this lane, 2026-10-01, reproduced on the B200 with the SM100 body too):
    # the stats ride the alpha ring (mb_stat_full / mb_stat_empty, 2 slots) but a fixed column is protected by the ring only
    # when the NEXT writer of that column waits the SAME slot's empty.  An EMPTY tile is ONE ring step, so its stats store
    # waits mb_stat_empty[1 - s] (the correction's consumption of the previous tile's LAST ALPHA), not mb_stat_empty[s] (its
    # read of the previous tile's stats): a softmax lane that enters an empty tile while its correction lane is still in the
    # slow arm (two O rescales between that alpha consume and the stats read) overwrites cols 386/387 with (-, 0) and the
    # correction reads final_sum = 0 -> the row is published DEAD (O = 0, LSE = -inf).  Only rows whose alpha != 1 on the
    # last iteration (the slow arm) lose the race -- test_sm107_d512_correction_handoff_under_mixed_rescale's +1 stair rows,
    # test_dsl_sm100_empty_tiles_between_live_tiles_multiwave (lane_d512_fprop/sm107/stair_probe.py: `repro padded qtrim qlens
    # gqa sq1024 multi` fails, `step6` passes, `allpos` kills every row).  Slot-indexed stats put every ring step's payload
    # in its own columns, so the ring's full/empty protocol covers the stats exactly as it covers alpha.
    STATS_OFF: int = 386
    STATS_COLS_PER_SLOT: int = 2


LAYOUT = KernelTmemLayout()
_require(LAYOUT.TOTAL_COLS == CFG.TMEM_COLS and LAYOUT.O_COLS == CFG.O_TMEM_COLS and LAYOUT.S_ACC_COLS == CFG.S_TMEM_COLS, "TMEM layout disagrees with the Cfg")
_require(LAYOUT.O_OFF + LAYOUT.O_COLS == LAYOUT.S_ACC_OFF, "S parities must abut O")
_require(LAYOUT.S_ACC_OFF + CFG.XFER_STAGES * LAYOUT.S_ACC_COLS == LAYOUT.ALPHA_OFF, "alpha ring must abut the S parities")
_require(
    LAYOUT.ALPHA_OFF + STAT_STAGES == LAYOUT.STATS_OFF and LAYOUT.STATS_OFF + STAT_STAGES * LAYOUT.STATS_COLS_PER_SLOT <= LAYOUT.TOTAL_COLS,
    "alpha/stats columns overflow",
)
_require(CFG.XFER_STAGES >= CFG.BMM1_LOOKAHEAD + 1, "XFER_STAGES >= BMM1_LOOKAHEAD + 1 (S/P slot reuse rides the MMA issue order)")
_require(CFG.BMM1_LOOKAHEAD == 1 and CFG.XFER_STAGES == 2, "v1 issue schedule is written for one iteration of BMM1 lookahead over two S/P parities")

# ---------------------------------------------------------------------------- descriptor constants
_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_QKP = _SWZ_ENUM[CFG.Q_SWZ_BYTES]  # K-major SW128
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]  # MN-major SW128
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES  # 1024
_CORE_MATRIX_ROWS = 8
# V (B transposed / MN-major): this CTA's N/CTA = 128 d_v cols of a 256-wide N-block = 2 column
# subtiles of 64 -> LBO = the byte distance between them = TILE_N * 128 B = 16384 (probe ss_slabs
# S4: LBO 0 corrupts output columns 64..127 of each CTA's slice).
LEADING_BYTE_OFFSET_PV = 0 if (V_SUB_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES
_require(LEADING_BYTE_OFFSET_PV == 16384, f"V descriptor LBO must be 16384 for 128 d_v cols per CTA, got {LEADING_BYTE_OFFSET_PV}")

# ---------------------------------------------------------------------------- shared helpers
# kv_shared_cluster: cluster-UNION bounds (height 256 rows, base = the cluster's first row) and
# CGA_M-unit Q super decodes; the "cta_in_pair" argument of every helper is this CTA's CLUSTER rank.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True, kv_shared_cluster=True)
_bounds_for_tile = _sdpa_h.bounds_for_tile_qtrim
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets

from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS

# === PackGQA === whole-group packing only (G | 64): row r <-> token r // G, head r % G.
HEADS_PER_TILE = CFG.PACK_G if CFG.PACK_GQA else 1
TOKENS_PER_TILE = CFG.TILE_M // HEADS_PER_TILE
_require(not CFG.PACK_GQA or CFG.PACK_G == CFG.QH_PER_KH, "2x2 d512 packs the whole GQA group (PACK_G == QH_PER_KH)")

# === KV split === (phase 1.5: the body carries the half-aware fp32-partials arm; the adapter twin
# keeps split_kv > 1 on the role-split kernel)
_split_h = make_split_helpers(
    CFG,
    bounds_for_tile=_bounds_for_tile,
    dispatch_decode_initial=_dispatch_decode_initial,
    dispatch_decode_payload=_dispatch_decode_payload,
)
SPLIT_KV = _split_h.SPLIT_KV
_FP32_PARTIALS = SPLIT_KV > 1
MAY_BE_EMPTY = _split_h.MAY_BE_EMPTY
_decode_initial_split = _split_h.decode_initial_split
_decode_payload_split = _split_h.decode_payload_split
_bounds_for_tile_split = _split_h.bounds_for_tile_split
_nomask_range_split = _split_h.nomask_range_split
_partial_batch = _split_h.partial_batch


@cute.jit
def _poll_wait(mb, phase):
    """A wait that NEVER parks the warp: the shared tile_dsl ``barrier.wait_poll`` (an inline-PTX ``mbarrier.test_wait.parity``
    loop; the DSL's ``nvvm.mbarrier_test_wait`` wrapper raises a TypeError on the public 4.7.0 build) in THIS module's shape
    (``POLL_TIGHT_ITERS`` / ``POLL_SLEEP_NS``) -- the form every barrier whose completing event is issued from OUTSIDE the pair
    must take (see POLL_CROSS_PAIR_WAITS).  ``mb`` is the barrier's SMEM pointer (``MBarrier.smem_ptr`` /
    ``MBarrier[idx].smem_ptr``)."""
    wait_poll(mb, phase, tight_iters=POLL_TIGHT_ITERS, sleep_ns=POLL_SLEEP_NS)


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
    seq_q_lens_addr: cutlass.Int64 = 0,
    o_partial_f32: Optional[cute.Tensor] = None,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # SMEM, in the order of the budget ledger (every array a multiple of 1 KiB, so no inter-array pad).
    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_K_SUB * kSubElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sVO_raw = cutlass.Array(STORAGE_DTYPE, max(CFG.STAGES_V_SUB * vSubElems, oBufferElems), alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(P_STORAGE_DTYPE, CFG.XFER_STAGES * pSlotElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # Row-max exchange [parity][half][row] and the tile-end row-sum exchange [half][row] (dedicated).
    sXchgMax = cutlass.Array(cutlass.Float32, CFG.XFER_STAGES * 2 * CFG.TILE_M, alignment=16, space=cutlass.AddressSpace.smem)
    sXchgSum = cutlass.Array(cutlass.Float32, 2 * CFG.TILE_M, alignment=16, space=cutlass.AddressSpace.smem)

    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKP,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=Q_SUBTILE_STRIDE_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kSubElems,
        stages=CFG.STAGES_K_SUB,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKP,
        tma_loads_per_tile=K_SUBTILES_PER_SUB,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=K_SUBTILE_STRIDE_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sVO_raw,
        elems_per_stage=vSubElems,
        stages=CFG.STAGES_V_SUB,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=V_SUBTILES_PER_SUB,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=V_SUBTILE_STRIDE_ELEMS,
        desc_version=DESC_VERSION,
    )
    sO = SmemTile(
        base=sVO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=O_SUBTILE_STRIDE_ELEMS,
        desc_version=DESC_VERSION,
    )
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pSlotElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKP,
        desc_version=DESC_VERSION,
    )

    bars = make_d512_2x2_bars(CFG, N_O_CHUNKS=N_O_CHUNKS, STAT_STAGES=STAT_STAGES)

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

    # Cluster ids.  cta_id_x is this CTA's cluster rank AND its Q super within the cluster.
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    mcast_mask = cutlass.Int32(3) << leader_cta_id  # the pair (commits)
    is_leader = cta_in_pair == cutlass.Int32(0)
    # K/V share group: the twin c ^ 2 (KV_SHARE=2) or just me.  TMA masks are Int16 register values.
    if cutlass.const_expr(KV_SHARE == 2):
        kv_share_mask_i32 = (cutlass.Int32(1) << cta_id_x) | (cutlass.Int32(1) << (cta_id_x ^ cutlass.Int32(2)))
        kv_empty_mask = cutlass.Int32(0xF)  # every CTA of the cluster
    else:
        kv_share_mask_i32 = cutlass.Int32(1) << cta_id_x
        kv_empty_mask = mcast_mask
    kv_mcast_mask = kv_share_mask_i32.to(cutlass.Int16)
    q_mcast_mask = (cutlass.Int32(1) << cta_id_x).to(cutlass.Int16)
    pair_idx = cta_id_x >> cutlass.Int32(1)  # which share of each sub-chunk this CTA issues
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            bars.mb_q_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_K_SUB):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
            for vs in cutlass.range_constexpr(CFG.STAGES_V_SUB):
                bars.mb_v_full[vs].init()
                bars.mb_v_empty[vs].init()
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                bars.mb_p_full[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()
            for s in cutlass.range_constexpr(STAT_STAGES):
                bars.mb_stat_full[s].init()
                bars.mb_stat_empty[s].init()
            for c in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[c].init()
            bars.mb_o_empty.init()
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), CFG.READ_TILE_ARRIVERS)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    if warp_idx >= cutlass.Int32(CFG.SOFTMAX_WG0_BASE) and warp_idx < cutlass.Int32(CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            sP_raw=sP_raw,
            sXchgMax=sXchgMax,
            sXchgSum=sXchgSum,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_id_x=cta_id_x,
            qh_per_kh=qh_per_kh,
        )

    elif warp_idx >= cutlass.Int32(CFG.CORR_WARP_BASE) and warp_idx < cutlass.Int32(CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS):
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            sO=sO,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_id_x=cta_id_x,
            qh_per_kh=qh_per_kh,
            o_partial_f32=o_partial_f32,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                sP=sP,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                seq_q_lens_addr=seq_q_lens_addr,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                kv_empty_mask=kv_empty_mask,
                cta_id_x=cta_id_x,
                qh_per_kh=qh_per_kh,
            )
        else:
            _mma_warp_quiet(tmem_ptr_i32, bars)

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seq_q_lens_addr=seq_q_lens_addr,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_id_x=cta_id_x,
            cta_in_pair=cta_in_pair,
            pair_idx=pair_idx,
            q_mcast_mask=q_mcast_mask,
            kv_mcast_mask=kv_mcast_mask,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            cta_id_x=cta_id_x,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
            qh_per_kh=qh_per_kh,
            seqlen_kv=seqlen_kv,
        )

    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(CFG.THD_VARLEN):
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
            scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _tma_issue_k(sK, tma_k, bars, k_state, kv_loop, kv_head_idx, row_off, tma_batch, is_leader, kv_mcast_mask, k_share_col, k_share_smem):
    """Both 32 KiB d-half sub-chunks of K tile ``kv_loop`` (this CTA's 64 kv rows): wait the slot, leader expect_tx of
    the PAIR's 64 KiB, issue my share (2 of 4 subtiles to me + twin under KV_SHARE=2, all 4 own-bit otherwise)."""
    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
    for dh in cutlass.range_constexpr(2):
        _poll_wait(bars.mb_k_empty[k_state.idx].smem_ptr, k_state.phase)
        bars.mb_k_full[k_state.idx].arrive(n_bytes=kSubTransactionBytes, pred=is_leader & nvvm.elect_sync())
        if cutlass.const_expr(KV_SHARE == 2):
            tma_load_subtiles(
                sK[k_state.idx].shifted(k_share_smem),
                tma_k(cutlass.Int32(dh * K_SUB_COLS) + k_share_col, kv_head_idx, kv_row_base + row_off, tma_batch),
                bars.mb_k_full[k_state.idx].smem_ptr,
                0,
                K_SUBTILES_PER_ISSUER,
                cta_group=CFG.CTA_MMA,
                mcast_mask=kv_mcast_mask,
            )
        else:
            tma_load_tile(
                sK[k_state.idx],
                tma_k(cutlass.Int32(dh * K_SUB_COLS), kv_head_idx, kv_row_base + row_off, tma_batch),
                bars.mb_k_full[k_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=kv_mcast_mask,
            )
        k_state = advance(k_state, CFG.STAGES_K_SUB)
    return k_state


@cute.jit
def _tma_issue_v(sV, tma_v, bars, v_state, kv_loop, kv_head_idx, v_col_off, kv_seq_off, tma_batch, is_leader, kv_mcast_mask, v_share_col, v_share_smem):
    """Both 32 KiB 128-col sub-chunks of V tile ``kv_loop`` (128 kv rows x this CTA's d_v [256*cta_in_pair + 128c, +128))."""
    kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)
    for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
        _poll_wait(bars.mb_v_empty[v_state.idx].smem_ptr, v_state.phase)
        bars.mb_v_full[v_state.idx].arrive(n_bytes=vSubTransactionBytes, pred=is_leader & nvvm.elect_sync())
        if cutlass.const_expr(KV_SHARE == 2):
            tma_load_subtiles(
                sV[v_state.idx].shifted(v_share_smem),
                tma_v(v_col_off + cutlass.Int32(c * V_SUB_COLS) + v_share_col, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                bars.mb_v_full[v_state.idx].smem_ptr,
                0,
                V_SUBTILES_PER_ISSUER,
                cta_group=CFG.CTA_MMA,
                mcast_mask=kv_mcast_mask,
            )
        else:
            tma_load_tile(
                sV[v_state.idx],
                tma_v(v_col_off + cutlass.Int32(c * V_SUB_COLS), kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                bars.mb_v_full[v_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=kv_mcast_mask,
            )
        v_state = advance(v_state, CFG.STAGES_V_SUB)
    return v_state


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    o_desc_words,
    qh_per_kh,
    is_leader,
    cta_id_x,
    cta_in_pair,
    pair_idx,
    q_mcast_mask,
    kv_mcast_mask,
):
    q_empty_state = PipelineState.start(phase=1)
    k_state = PipelineState.start(phase=1)
    v_state = PipelineState.start(phase=1)
    o_empty_for_v_state = PipelineState.start(phase=1)

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's packed-total-clamped runtime descriptors (o_desc_words
        # slots n_batch+1 / n_batch+2) so the last sequence's tile tail lands as exact zeros.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
    )
    q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
    kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    # This CTA's K rows [64 * cta_in_pair, +64) and V columns [256 * cta_in_pair, +256) of every tile.
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(K_SUB_ROWS)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)
    # My share of each sub-chunk: K subtiles [pair_idx * K_SUBTILES_PER_ISSUER, +K_SUBTILES_PER_ISSUER)
    # (d cols 64 * that), V subtile pair_idx * V_SUBTILES_PER_ISSUER.
    k_share_first = pair_idx * cutlass.Int32(K_SUBTILES_PER_ISSUER)
    v_share_first = pair_idx * cutlass.Int32(V_SUBTILES_PER_ISSUER)
    k_share_col = k_share_first * cutlass.Int32(TMA_QK_GRANU_ELEMS)
    v_share_col = v_share_first * cutlass.Int32(TMA_VO_GRANU_ELEMS)
    k_share_smem = k_share_first * cutlass.Int32(K_SUBTILE_STRIDE_ELEMS)
    v_share_smem = v_share_first * cutlass.Int32(V_SUBTILE_STRIDE_ELEMS)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty KV loop: no loads, but TMA-STG still drains the (zeroed) O of this tile and flips
            # mb_o_empty once, so the O u V alias gate must advance to stay in phase.
            o_empty_for_v_state = advance(o_empty_for_v_state, 1)
        else:
            bars.mb_q_empty.wait(q_empty_state.phase, spin=SPIN_RING_WAITS)
            q_empty_state = advance(q_empty_state, 1)
            bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), q_head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=q_mcast_mask,
            )

            # Issue order mirrors the MMA's one-iteration BMM1 lookahead: K(kv_left) prologue, then per iteration
            # K(kv+1) BEFORE V(kv), V(kv_right-1) tail.  K(kv+1) must never queue behind V(kv): V(kv)'s slot is
            # released by BMM2(kv-1), and the K(kv+1) issue would then wait for that completion and land its
            # whole L2 round trip on BMM1(kv+1) (MEASURED: the K-after-V order ran dense H128 S8K at 24.86 ms,
            # 0.70x of the role split).  The K prologue precedes the O u V gate so the next tile's K / BMM1
            # overlap the previous tile's epilogue and O store.
            k_state = _tma_issue_k(
                sK, tma_k, bars, k_state, kv_left, kv_head_idx, K_ROW_OFFSET_PEER + kv_seq_off, tma_batch, is_leader, kv_mcast_mask, k_share_col, k_share_smem
            )

            # O u V alias: the first V load of this tile overwrites the previous tile's O staging.
            _poll_wait(bars.mb_o_empty.smem_ptr, o_empty_for_v_state.phase)
            o_empty_for_v_state = advance(o_empty_for_v_state, 1)

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                k_state = _tma_issue_k(
                    sK,
                    tma_k,
                    bars,
                    k_state,
                    kv_loop + cutlass.Int32(1),
                    kv_head_idx,
                    K_ROW_OFFSET_PEER + kv_seq_off,
                    tma_batch,
                    is_leader,
                    kv_mcast_mask,
                    k_share_col,
                    k_share_smem,
                )
                v_state = _tma_issue_v(
                    sV,
                    tma_v,
                    bars,
                    v_state,
                    kv_loop,
                    kv_head_idx,
                    V_COL_OFFSET_PEER,
                    kv_seq_off,
                    tma_batch,
                    is_leader,
                    kv_mcast_mask,
                    v_share_col,
                    v_share_smem,
                )
            v_state = _tma_issue_v(
                sV,
                tma_v,
                bars,
                v_state,
                kv_right - cutlass.Int32(1),
                kv_head_idx,
                V_COL_OFFSET_PEER,
                kv_seq_off,
                tma_batch,
                is_leader,
                kv_mcast_mask,
                v_share_col,
                v_share_smem,
            )

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q, nxt_hb, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
        )
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        kv_head_idx = cute.arch.make_warp_uniform(head_idx if cutlass.const_expr(CFG.PACK_GQA) else head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # End-of-kernel drain: every ring slot's last consumer (the pair's -- and under KV_SHARE=2 the
    # twin pair's -- MMA) must have committed before this CTA exits, so multicast targets stay alive.
    for _ks in cutlass.range_constexpr(CFG.STAGES_K_SUB):
        _poll_wait(bars.mb_k_empty[k_state.idx].smem_ptr, k_state.phase)
        k_state = advance(k_state, CFG.STAGES_K_SUB)
    for _vs in cutlass.range_constexpr(CFG.STAGES_V_SUB):
        _poll_wait(bars.mb_v_empty[v_state.idx].smem_ptr, v_state.phase)
        v_state = advance(v_state, CFG.STAGES_V_SUB)
    bars.mb_q_empty.wait(q_empty_state.phase)
    if cutlass.const_expr(KV_SHARE == 2):
        # Pair-wide O u V gate: the last tile's mb_o_empty phase completes only when the TWIN's TMA-STG warp has
        # arrived on MY copy too -- waiting for it here keeps this CTA alive for that remote arrive (every other
        # cross-CTA arrive of the kernel is likewise waited by its target before exit).
        _poll_wait(bars.mb_o_empty.smem_ptr, o_empty_for_v_state.phase)
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
    cta_id_x,
    seq_kv_lens_tensor,
    o_desc_words,
    seqlen_kv,
    qh_per_kh,
):
    o_full_phase = cutlass.Int32(0)
    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE)
        q_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE)
        o_batch = _partial_batch(batch_idx, split_idx, n_batch)

        if cutlass.const_expr(_FP32_PARTIALS):
            # fp32 partials went straight to the workspace: wait the chunks, skip the store.
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].wait(o_full_phase)
        elif cutlass.const_expr(CFG.THD_VARLEN):
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].wait(o_full_phase)
            # DEAD unit (batch == n_batch): no O rows exist, descriptor slot n_batch is never built.
            if batch_idx < n_batch:
                o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), q_head_idx, q_row_coord, cutlass.Int32(0))
                tma_store_tile(sO[0], o_slice)
                tma_store_commit()
                tma_store_wait(0)
        else:
            o_slice = tma_o(cutlass.Int32(0), q_head_idx, q_row_coord, o_batch)
            if cutlass.const_expr(O_STORE_STREAM):
                for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                    bars.mb_o_full[chunk].wait(o_full_phase)
                    tma_store_subtile(sO[0], o_slice, chunk)
            else:
                for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                    bars.mb_o_full[chunk].wait(o_full_phase)
                tma_store_tile(sO[0], o_slice)
            tma_store_commit()
            tma_store_wait(0)

        # PAIR-WIDE O u V alias gate (fix-lane FATAL-1): this warp's 32 lanes arrive on MY mb_o_empty and, under
        # KV_SHARE=2, on the TWIN's (cta_id_x ^ 2) -- init ONE_WARP x KV_SHARE -- so the twin's first V(t+1) multicast
        # (which lands in MY sVO too) waits for MY O(t) store as well as its own.  Ledger: make_d512_2x2_bars.
        bars.mb_o_empty.arrive()
        if cutlass.const_expr(KV_SHARE == 2):
            bars.mb_o_empty.arrive_on_peer(cta_id_x ^ cutlass.Int32(2))
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q, nxt_hb, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    """The pair's non-leader MMA warp: allocates / publishes / frees the TMEM, issues nothing."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _bmm1_issue(bmm1_desc, desc_Q_lo, desc_Q_hi, sK, tmem_S, k_state, bars, mcast_mask, kv_empty_mask, parity_rt):
    """S[parity] = Q K^T as two K=256 d-half sub-chunk calls (16 k-steps each, bitwise identical to
    one K=512 chain: probe ss_slabs S3).  After dh=0 the K slot is released; after dh=1 bmm1_done[p]
    (pair) and the second slot.  Returns the advanced K ring state."""
    _poll_wait(bars.mb_k_full[k_state.idx].smem_ptr, k_state.phase)
    mma_ss(bmm1_desc, desc_Q_lo, sK[k_state.idx].desc(), tmem_S, accumulate=False)
    elect_p = nvvm.elect_sync()
    bars.mb_k_empty[k_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=kv_empty_mask, pred=elect_p)
    k_state = advance(k_state, CFG.STAGES_K_SUB)

    _poll_wait(bars.mb_k_full[k_state.idx].smem_ptr, k_state.phase)
    mma_ss(bmm1_desc, desc_Q_hi, sK[k_state.idx].desc(), tmem_S, accumulate=True)
    elect_p = nvvm.elect_sync()
    bars.mb_bmm1_done[parity_rt].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
    bars.mb_k_empty[k_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=kv_empty_mask, pred=elect_p)
    k_state = advance(k_state, CFG.STAGES_K_SUB)
    return k_state


@cute.jit
def _bmm2_issue(bmm2_desc, desc_P, sV, tmem_raw, v_state, bars, mcast_mask, kv_empty_mask, parity_rt, ready_phase, accumulate):
    """O += P V as two collective N=256 blocks (one 32 KiB V sub-chunk each; block c lands at TMEM
    cols [128c, +128)).  Waits bmm2_ready[p*2+c] + v_full before each block; releases the V slot after
    each and bmm2_done[p] (pair) after the second.  Returns the advanced V ring state."""
    for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
        bars.mb_bmm2_ready[parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(c)].wait(ready_phase, spin=SPIN_RING_WAITS)
        _poll_wait(bars.mb_v_full[v_state.idx].smem_ptr, v_state.phase)
        tmem_O_c = tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF + c * (LAYOUT.O_COLS // CFG.N_BMM2_CHUNKS)))
        mma_ss(bmm2_desc, desc_P, sV[v_state.idx].desc(), tmem_O_c, accumulate=accumulate)
        elect_p = nvvm.elect_sync()
        if cutlass.const_expr(c == CFG.N_BMM2_CHUNKS - 1):
            bars.mb_bmm2_done[parity_rt].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=elect_p)
        bars.mb_v_empty[v_state.idx].arrive(cta_group=CFG.CTA_MMA, mcast_mask=kv_empty_mask, pred=elect_p)
        v_state = advance(v_state, CFG.STAGES_V_SUB)
    return v_state


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    sP,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    kv_empty_mask,
    cta_id_x,
    qh_per_kh,
):
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        a_negate=int(PARAMS.negate_scores),
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.BMM2_N_PER_CALL,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
    )
    # BMM1 per d-half sub-chunk: M=128 collective (64 rows/CTA -> 8 KiB A subtiles), N=128 (64 kv
    # rows/CTA), K=256 (16 steps of 16); Q is re-based by 4 subtiles for the second half OUTSIDE
    # mma_ss so both operands index k in [0, 16) relative to the sub-chunk.
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=K_SUB_COLS,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2 per N-block: M=128, N=256 collective (128 d_v/CTA), K=128 kv (8 steps); P K-major
    # (subtile h = kv [64h, +64) -> k-steps 4h..4h+3), V MN-major with k_subtile 64.
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

    # The two Q descriptor bases are static (one-stage SMEM array): left as plain values their 2 x 16 per-k-step adds
    # fold to immediates.  Re-anchoring them with desc_opaque(..., anchor=<loop-variant>) -- the sm107 d256 backward's
    # remedy for a RUNTIME base -- makes them opaque and ADDS live 64-bit adds: MEASURED sm_107a 2026-10-01, 17 STL /
    # 17 LDL unanchored -> 27 / 31 anchored.  Do not anchor these.
    desc_Q_lo = sQ[0].desc()
    desc_Q_hi = sQ[0].shifted(K_SUBTILES_PER_SUB * Q_SUBTILE_STRIDE_ELEMS).desc()

    q_super_idx, _hd, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV == 1):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    elif cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds_init = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    k_state = PipelineState.start(phase=0)
    v_state = PipelineState.start(phase=0)
    # Per-parity phase bits (slot p is consumed once every two iterations).
    p_full_phase_pair = cutlass.Int32(0)
    bmm2_ready_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(MAY_BE_EMPTY) and (kv_right <= kv_left):
            # Empty tile: the correction lanes of both CTAs arrive mb_empty_mainloop; one bmm2_done[0]
            # commit flips the epilogue's final wait (d256 protocol).  No BMM, no ring traffic.
            bars.mb_empty_mainloop.wait(empty_mainloop_phase, spin=SPIN_RING_WAITS)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            bars.mb_bmm2_done[0].arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=nvvm.elect_sync())
        else:
            bars.mb_q_full.wait(q_full_phase, spin=SPIN_RING_WAITS)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            # Prologue: BMM1(kv_left) into S[parity(kv_left)].
            parity_lo_rt = kv_left & cutlass.Int32(1)
            k_state = _bmm1_issue(
                bmm1_desc,
                desc_Q_lo,
                desc_Q_hi,
                sK,
                tmem_raw.subview(cutlass.Int32(LAYOUT.S_ACC_OFF) + parity_lo_rt * cutlass.Int32(LAYOUT.S_ACC_COLS)),
                k_state,
                bars,
                mcast_mask,
                kv_empty_mask,
                parity_lo_rt,
            )

            # Steady state: BMM1(i+1) one iteration ahead of BMM2(i).  S/P slot reuse: BMM1(i+2) is
            # issued (next body) after BMM2(i), which waited p_full(i) = every softmax lane's
            # wait::ld of S[p] + its P(i) store -- so XFER_STAGES (2) >= BMM1_LOOKAHEAD (1) + 1.
            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                k_state = _bmm1_issue(
                    bmm1_desc,
                    desc_Q_lo,
                    desc_Q_hi,
                    sK,
                    tmem_raw.subview(cutlass.Int32(LAYOUT.S_ACC_OFF) + parity_next_rt * cutlass.Int32(LAYOUT.S_ACC_COLS)),
                    k_state,
                    bars,
                    mcast_mask,
                    kv_empty_mask,
                    parity_next_rt,
                )
                p_full_phase_cur = (p_full_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
                bars.mb_p_full[parity_cur_rt].wait(p_full_phase_cur, spin=SPIN_RING_WAITS)
                p_full_phase_pair = p_full_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)
                v_state = _bmm2_issue(
                    bmm2_desc,
                    sP[parity_cur_rt].desc(),
                    sV,
                    tmem_raw,
                    v_state,
                    bars,
                    mcast_mask,
                    kv_empty_mask,
                    parity_cur_rt,
                    ready_phase_cur,
                    cutlass.Boolean(kv_loop != kv_left),
                )
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)

            # Tail: BMM2(kv_right - 1), then release Q for the next tile's load.
            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            p_full_phase_last = (p_full_phase_pair >> parity_last_rt) & cutlass.Int32(1)
            bars.mb_p_full[parity_last_rt].wait(p_full_phase_last, spin=SPIN_RING_WAITS)
            p_full_phase_pair = p_full_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)
            v_state = _bmm2_issue(
                bmm2_desc,
                sP[parity_last_rt].desc(),
                sV,
                tmem_raw,
                v_state,
                bars,
                mcast_mask,
                kv_empty_mask,
                parity_last_rt,
                ready_phase_last,
                cutlass.Boolean(kv_last != kv_left),
            )
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

            bars.mb_q_empty.arrive(cta_group=CFG.CTA_MMA, mcast_mask=mcast_mask, pred=nvvm.elect_sync())

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, _hd, batch_idx, split_idx = _decode_payload_split(
            nxt_q, nxt_hb, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0 and SPLIT_KV > 1):
            kv_left, kv_right = _nomask_range_split(seqlen_kv, split_idx)
        elif cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
            bounds_next = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_kv_iter(
    apply_mask: bool,
    kv_loop,
    bmm1_done_phase_pair,
    stat_state,
    total_max,
    total_max_safe,
    total_sum_vec,
    tmem_base,
    bars,
    sP_raw,
    sXchgMax,
    q_abs,
    row_r,
    half_h,
    eff_seqlen_kv,
    eff_seqlen_q,
    scale_log2,
    leader_cta_id,
):
    """One KV iteration of lane (r, h): S half -> masked -> HALF-row max -> exchange with lane r + 64
    through sXchgMax[parity][half][row] + named barrier 8 -> identical online-softmax chain on both
    halves -> alpha to TMEM col ALPHA_OFF + ring idx -> P = exp2(S - max) into sP[parity] subtile h
    row r (128-B swizzled row segment) -> fence.proxy.async + per-lane .release.cta arrive on the
    leader's mb_p_full[parity]."""
    parity_rt = kv_loop & cutlass.Int32(1)
    bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
    bars.mb_bmm1_done[parity_rt].wait(bmm1_phase, spin=SPIN_RING_WAITS)
    bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

    s_addr = tmem_base + cutlass.Int32(LAYOUT.S_ACC_OFF) + parity_rt * cutlass.Int32(LAYOUT.S_ACC_COLS)
    if cutlass.const_expr(apply_mask):
        raw_S = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr, cutlass.Float32), num=SOFTMAX_COLS)
        # Pin the load before anything else runs: the S slot's reuse (BMM1(i+2)) is ordered through this
        # lane's p_full arrive, so the ld must have landed before that arrive.
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N) + half_h * cutlass.Int32(SOFTMAX_COLS)
        causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
        reg_S = apply_mask_chunk(
            raw_S,
            q_abs,
            kv_col_base,
            eff_seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            N=SOFTMAX_COLS,
            bottom_right=CFG.BOTTOM_RIGHT,
            causal_diag=causal_diag,
            window_right=CFG.WINDOW_RIGHT,
            mask_value=float("-inf"),
        )
        half_max = row_max_reduction(reg_S)
    else:
        # Rubin dense arm: ONE tcgen05.ld.red.max over this lane's 64 S columns (the fused LDTM.STAT, cc 10.7) returns
        # the data AND the HALF-row max -- the exchange with lane r + 64 below is still mandatory (the 2x2 atom's lane
        # half holds one column half; AGENTS.md: every row_max_reduction / ld.red.max result under a TILE_M=64, CTA_MMA=2
        # Cfg is a half-row value until combined).  The same wait::ld pins the load ahead of the p_full arrive.
        reg_S_tile, half_max = tmem_load_max_reduction_tile(s_addr, num_elems=SOFTMAX_COLS)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        reg_S = reg_S_tile.vec

    # Row-max exchange: slot [parity][half][row]; one barrier per iteration suffices (each lane writes
    # before and reads after it; slot p is next written two iterations later, after barrier i+1 has
    # ordered the partner's read; the tile-end barriers order the cross-tile reuse).  fmax is
    # commutative -> both halves hold the bitwise-same max and run the identical chain below.
    xchg_mine = (parity_rt * cutlass.Int32(2) + half_h) * cutlass.Int32(CFG.TILE_M) + row_r
    xchg_other = (parity_rt * cutlass.Int32(2) + (cutlass.Int32(1) - half_h)) * cutlass.Int32(CFG.TILE_M) + row_r
    sXchgMax.subview(xchg_mine).store(half_max)
    nvvm.barrier_cta_sync(barrier_id=8, thread_count=CFG.SOFTMAX_LANES)
    other_max = sXchgMax.subview(xchg_other).load()
    current_max_raw = cute.math.max(half_max, other_max, ftz=True)
    if cutlass.const_expr(SCALE_PREFOLDED):
        # Pre-folded scale: Q carries attn_scale * log2 e, so the raw max IS the log2-domain max (-inf when the whole
        # iteration is masked, exactly as below: both arms mask with true -inf).
        current_max = current_max_raw
    else:
        current_max = current_max_raw * scale_log2  # -inf when the whole iteration is masked

    # total_max starts at -inf: a live iteration always clears the threshold, a fully-masked one never
    # does (-inf - x = -inf or NaN; ordered > is false for both).
    update_cond = (current_max - total_max) > RESCALE_THRESHOLD_F32
    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
    new_total_max_safe = row_max_for_exp2(total_max)
    alpha = cute.math.exp2(cute.math.min(total_max_safe - new_total_max_safe, cutlass.Float32(0.0)), fastmath=True)
    total_max_safe = new_total_max_safe

    # alpha -> TMEM col ALPHA_OFF + ring idx (lanes r and r + 64 write the same value).
    bars.mb_stat_empty[stat_state.idx].wait(stat_state.phase, spin=SPIN_RING_WAITS)
    alpha_addr = tmem_base + cutlass.Int32(LAYOUT.ALPHA_OFF) + stat_state.idx
    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(alpha_addr, cutlass.Float32), cutlass.Vector.from_elements((alpha,), cutlass.Float32))
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
    bars.mb_stat_full[stat_state.idx].arrive()
    stat_state = advance(stat_state, STAT_STAGES)

    # P for this lane's 64 columns -> sP[parity] subtile h, row r.
    if cutlass.const_expr(SCALE_PREFOLDED):
        reg_P = cute.math.exp2(reg_S - total_max_safe, fastmath=True)  # raw scores are log2-domain: the shift alone (FADD2)
    else:
        reg_P = cute.math.exp2(reg_S * scale_log2 - total_max_safe, fastmath=True)
    alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
    total_sum_vec = total_sum_vec * alpha_pair + row_reduction_pair(reg_P)
    reg_P_half = reg_P.to(P_STORAGE_DTYPE)
    p_off = parity_rt * cutlass.Int32(pSlotElems) + half_h * cutlass.Int32(P_SUBTILE_ELEMS) + row_r * cutlass.Int32(P_ROW_ELEMS)
    sP_raw.subview(p_off).data_ptr().store_swizzled(reg_P_half, alignment=64, swizzle=P_SMEM_SWIZZLE)
    # Publish to the async proxy (the leader's cta_group::2 BMM2 reads this slab in place across the
    # pair), then the per-lane release arrive on the leader's barrier; keep the fence on EVERY lane
    # immediately before the arrive (probe release_p: the MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S pair is
    # the only hardware ordering present).
    nvvm.fence_proxy("async.shared", space="cta")
    bars.mb_p_full[parity_rt].arrive(cta_group=CFG.CTA_MMA, leader_cta_id=leader_cta_id)

    return bmm1_done_phase_pair, stat_state, total_max, total_max_safe, total_sum_vec


@cute.jit
def _softmax_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    sP_raw,
    sXchgMax,
    sXchgSum,
    bars,
    sched,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_id_x,
    qh_per_kh,
):
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WG_WARPS + 1))
    tmem_base = tmem_ptr_i32.load()

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)
    row_r = tid_in_wg & cutlass.Int32(CFG.TILE_M - 1)
    half_h = tid_in_wg >> cutlass.Int32(6)

    bmm1_done_phase_pair = cutlass.Int32(0)
    stat_state = PipelineState.start(phase=1)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF_F32
        total_max_safe = NEG_INF_F32
        total_sum_vec = cutlass.Vector.from_elements((cutlass.Float32(0.0), cutlass.Float32(0.0)), cutlass.Float32)
        # PackGQA: q_abs is the row's TOKEN index (row // G).
        q_abs = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (row_r // cutlass.Int32(HEADS_PER_TILE))

        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                bmm1_done_phase_pair, stat_state, total_max, total_max_safe, total_sum_vec = _softmax_kv_iter(
                    False,
                    kv_loop,
                    bmm1_done_phase_pair,
                    stat_state,
                    total_max,
                    total_max_safe,
                    total_sum_vec,
                    tmem_base,
                    bars,
                    sP_raw,
                    sXchgMax,
                    q_abs,
                    row_r,
                    half_h,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    leader_cta_id,
                )
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                bmm1_done_phase_pair, stat_state, total_max, total_max_safe, total_sum_vec = _softmax_kv_iter(
                    True,
                    kv_loop,
                    bmm1_done_phase_pair,
                    stat_state,
                    total_max,
                    total_max_safe,
                    total_sum_vec,
                    tmem_base,
                    bars,
                    sP_raw,
                    sXchgMax,
                    q_abs,
                    row_r,
                    half_h,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                bmm1_done_phase_pair, stat_state, total_max, total_max_safe, total_sum_vec = _softmax_kv_iter(
                    False,
                    kv_loop,
                    bmm1_done_phase_pair,
                    stat_state,
                    total_max,
                    total_max_safe,
                    total_sum_vec,
                    tmem_base,
                    bars,
                    sP_raw,
                    sXchgMax,
                    q_abs,
                    row_r,
                    half_h,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    leader_cta_id,
                )
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                bmm1_done_phase_pair, stat_state, total_max, total_max_safe, total_sum_vec = _softmax_kv_iter(
                    True,
                    kv_loop,
                    bmm1_done_phase_pair,
                    stat_state,
                    total_max,
                    total_max_safe,
                    total_sum_vec,
                    tmem_base,
                    bars,
                    sP_raw,
                    sXchgMax,
                    q_abs,
                    row_r,
                    half_h,
                    eff_seqlen_kv,
                    eff_seqlen_q,
                    scale_log2,
                    leader_cta_id,
                )

        # Tile end: row-sum exchange through the DEDICATED sXchgSum (fixed operand order -> identical on
        # both halves); a second barrier so no fast lane's next write (sum of the next tile, or the max
        # slot of its first iteration) races the partner's read; then the tile stats to THIS ring step's
        # stats columns (STATS_OFF + 2 * slot: the ring's empty wait below protects exactly these two
        # columns, see KernelTmemLayout) as the next ring step -- unconditional, empty tiles included
        # (the correction consumes one ring step per tile end).
        partial_sum = total_sum_vec[0] + total_sum_vec[1]
        sXchgSum.subview(half_h * cutlass.Int32(CFG.TILE_M) + row_r).store(partial_sum)
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=CFG.SOFTMAX_LANES)
        final_sum = sXchgSum.subview(row_r).load() + sXchgSum.subview(cutlass.Int32(CFG.TILE_M) + row_r).load()
        nvvm.barrier_cta_sync(barrier_id=8, thread_count=CFG.SOFTMAX_LANES)
        bars.mb_stat_empty[stat_state.idx].wait(stat_state.phase, spin=SPIN_RING_WAITS)
        stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF) + stat_state.idx * cutlass.Int32(LAYOUT.STATS_COLS_PER_SLOT)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), cutlass.Vector.from_elements((total_max_safe, final_sum), cutlass.Float32))
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full[stat_state.idx].arrive()
        stat_state = advance(stat_state, STAT_STAGES)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q, nxt_hb, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)


@cute.jit
def _rescale_o_half(tmem_base, alpha, first_block: cutlass.Constexpr[int], n_blocks: cutlass.Constexpr[int]):
    """O[lane][cols of blocks first_block .. +n_blocks) *= alpha (16-col ld / packed mul / st), then wait::st."""
    for blk in cutlass.range_constexpr(n_blocks):
        o_addr = tmem_base + cutlass.Int32(LAYOUT.O_OFF + (first_block + blk) * CORR_BLOCK_COLS)
        o_chunk = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), num=CORR_BLOCK_COLS)
        o_scaled = vec_scale_pair(o_chunk, alpha, CORR_BLOCK_COLS)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)


CORR_BLOCK_COLS = 16
CORR_BLOCKS_PER_NBLOCK = (LAYOUT.O_COLS // CFG.N_BMM2_CHUNKS) // CORR_BLOCK_COLS  # 8
O_EPI_BLOCK_COLS = 32
N_O_EPI_BLOCKS = LAYOUT.O_COLS // O_EPI_BLOCK_COLS  # 8 per lane: d_v = 256h + 32b .. +32
_require(N_O_EPI_BLOCKS == N_O_CHUNKS, "the epilogue drains 8 blocks per lane into the 8 O subtiles (two blocks per subtile)")


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    tmem_ptr_i32,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    seq_q_lens_addr,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_id_x,
    qh_per_kh,
    o_partial_f32=None,
):
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))
    tmem_base = tmem_ptr_i32.load()

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.CORR_WARP_BASE * 32)
    row_r = tid_in_wg & cutlass.Int32(CFG.TILE_M - 1)
    half_h = tid_in_wg >> cutlass.Int32(6)  # warp-uniform: warps 4,5 -> 0, warps 6,7 -> 1

    bmm2_done_phase_pair = cutlass.Int32(0)
    stat_state = PipelineState.start(phase=0)
    o_empty_phase = cutlass.Int32(1)

    q_super_idx, head_idx, batch_idx, split_idx = _decode_initial_split(
        sched.bidx_init, sched.bidy_init, sched.bidz_init, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
    bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    sO_base = sO[0].base

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if bounds.right > bounds.left:
            # BMM2(kv_left) needs no rescale: credit both N-blocks now, consume (and ignore) alpha(kv_left).
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            bars.mb_stat_full[stat_state.idx].wait(stat_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_stat_empty[stat_state.idx].arrive()
            stat_state = advance(stat_state, STAT_STAGES)
        else:
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)

            bars.mb_stat_full[stat_state.idx].wait(stat_state.phase, spin=SPIN_RING_WAITS)
            alpha_addr = tmem_base + cutlass.Int32(LAYOUT.ALPHA_OFF) + stat_state.idx
            alpha_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(alpha_addr, cutlass.Float32), num=1)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            alpha = alpha_vec[0]
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha == cutlass.Float32(1.0), VoteSync.ALL)
            bars.mb_stat_empty[stat_state.idx].arrive()
            stat_state = advance(stat_state, STAT_STAGES)

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            ready_c0 = parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)
            ready_c1 = ready_c0 + cutlass.Int32(1)
            # Both arms arrive each credit exactly once per lane per iteration (ledger: 128 x 2 CTAs).
            if cutlass.const_expr(CORR_READY_BEFORE_DONE):
                if all_alpha_one:
                    bars.mb_bmm2_ready[ready_c0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                    bars.mb_bmm2_ready[ready_c1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                    bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev, spin=SPIN_RING_WAITS)
                else:
                    bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev, spin=SPIN_RING_WAITS)
                    _rescale_o_half(tmem_base, alpha, 0, CORR_BLOCKS_PER_NBLOCK)
                    bars.mb_bmm2_ready[ready_c0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                    _rescale_o_half(tmem_base, alpha, CORR_BLOCKS_PER_NBLOCK, CORR_BLOCKS_PER_NBLOCK)
                    bars.mb_bmm2_ready[ready_c1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            else:
                bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev, spin=SPIN_RING_WAITS)
                if ~all_alpha_one:
                    _rescale_o_half(tmem_base, alpha, 0, CORR_BLOCKS_PER_NBLOCK)
                bars.mb_bmm2_ready[ready_c0].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
                if ~all_alpha_one:
                    _rescale_o_half(tmem_base, alpha, CORR_BLOCKS_PER_NBLOCK, CORR_BLOCKS_PER_NBLOCK)
                bars.mb_bmm2_ready[ready_c1].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

        # ---- tile end: stats (UNCONDITIONAL ring step, empty tiles included; read from THIS ring step's columns) ----
        bars.mb_stat_full[stat_state.idx].wait(stat_state.phase, spin=SPIN_RING_WAITS)
        stats_addr_rd = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF) + stat_state.idx * cutlass.Int32(LAYOUT.STATS_COLS_PER_SLOT)
        stats_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(stats_addr_rd, cutlass.Float32), num=2)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        final_max = stats_vec[0]
        final_ell = stats_vec[1]
        bars.mb_stat_empty[stat_state.idx].arrive()
        stat_state = advance(stat_state, STAT_STAGES)

        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * TOKENS_PER_TILE) + (row_r // cutlass.Int32(HEADS_PER_TILE))
        row_head_idx = head_idx * cutlass.Int32(HEADS_PER_TILE) + (row_r % cutlass.Int32(HEADS_PER_TILE))
        LN2 = cutlass.Float32(0.6931471805599453)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = cutlass.Float32(sinks_arr[row_head_idx])
            final_max_nat = final_max * LN2
            # Keyless row (final_ell == 0): the sink is the row's whole mass, O := 0 / LSE := sink.
            kv_empty = final_ell <= cutlass.Float32(0.0)
            new_max_nat = cutlass.Float32(arith.select(kv_empty.ir_value(), sink_logit.ir_value(), cute.math.max(final_max_nat, sink_logit).ir_value()))
            scale_sink = cutlass.Float32(
                arith.select(kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), cute.math.exp(final_max_nat - new_max_nat, fastmath=True).ir_value())
            )
            new_sum = final_ell * scale_sink + cute.math.exp(sink_logit - new_max_nat, fastmath=True)
            beta = scale_sink / new_sum
            lse = new_max_nat + cute.math.log(new_sum, fastmath=True)
            # With a sink no row is "dead": a keyless row's O is zeroed through beta = 0 (scale_sink).
            row_dead = cutlass.Boolean(False)
        else:
            final_ell_safe = cute.math.max(final_ell, cutlass.Float32(1e-30))
            beta = cutlass.Float32(1.0) / final_ell_safe
            lse = final_max * LN2 + cute.math.log(final_ell_safe, fastmath=True)
            # Dead row (no valid KV column at all): O := 0, LSE := -inf.
            row_dead = final_ell <= cutlass.Float32(0.0)
            neg_inf_lse = cutlass.Float32(float("-inf"))
            beta = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))
            lse = cutlass.Float32(arith.select(row_dead.ir_value(), neg_inf_lse.ir_value(), lse.ir_value()))

        if cutlass.const_expr(CFG.SEQ_Q_LENS_PRESENT):
            # Dense padded-Q trim: q rows >= seq_len_q[b] write O := 0 / LSE := -inf (after the sink fold).
            _sq_arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            _q_len_b = cutlass.Int32(_sq_arr[batch_idx])
            row_trim = q_row_global >= _q_len_b
            neg_inf_trim = cutlass.Float32(float("-inf"))
            beta = cutlass.Float32(arith.select(row_trim.ir_value(), cutlass.Float32(0.0).ir_value(), beta.ir_value()))
            lse = cutlass.Float32(arith.select(row_trim.ir_value(), neg_inf_trim.ir_value(), lse.ir_value()))
            row_dead = row_dead | row_trim
        if cutlass.const_expr(CFG.STATS_LOG2):
            lse = lse * cutlass.Float32(1.4426950408889634)

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last, spin=SPIN_RING_WAITS)
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        # ---- O drain: lane (r, h) owns d_v [256h, +256) = TMEM cols [0, 256) of its lane; block b =
        # cols [32b, +32) -> O subtile s = 4h + (b >> 1), column (b & 1) * 32 -> sO elem offset
        # s * 4096 + r * 64 + (b & 1) * 32 (64-B swizzled half-row segment); after each odd b the 64
        # lanes of half h publish subtile s (fence.proxy.async + mb_o_full[s]) ----
        _poll_wait(bars.mb_o_empty.smem_ptr, o_empty_phase)
        o_empty_phase = o_empty_phase ^ cutlass.Int32(1)
        tile_live = cutlass.Boolean(True)
        if cutlass.const_expr(MAY_BE_EMPTY):
            tile_live = bounds.right > bounds.left
        if cutlass.const_expr(_FP32_PARTIALS):
            # fp32 split partials straight to the workspace (one writer per element: this lane's d_v half),
            # bounded by the slab's real d_v (envelope flavor); the staged path's chunk publishes still run.
            _op32 = cutlass.make_array_view(o_partial_f32)
            _d_v32 = cutlass.const_expr(o_partial_f32.shape[3])
            _o_b32 = _partial_batch(batch_idx, split_idx, n_batch)
            _valid32 = q_row_global < seqlen_q
            for b in cutlass.range_constexpr(N_O_EPI_BLOCKS):
                o_fp32 = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(O_EPI_BLOCK_COLS)), cutlass.Float32)
                if tile_live:
                    o_addr = tmem_base + cutlass.Int32(LAYOUT.O_OFF + b * O_EPI_BLOCK_COLS)
                    o_fp32 = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), num=O_EPI_BLOCK_COLS)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_fp32 * beta
                col_base = half_h * cutlass.Int32(LAYOUT.O_COLS) + cutlass.Int32(b * O_EPI_BLOCK_COLS)
                if _valid32:
                    _row_out = _op32[_o_b32, q_row_global, row_head_idx, :]
                    for _j in cutlass.range_constexpr(O_EPI_BLOCK_COLS):
                        col = col_base + cutlass.Int32(_j)
                        if col < cutlass.Int32(_d_v32):
                            _row_out[col] = cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_j].ir_value()))
                if cutlass.const_expr(b % 2 == 1):
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_o_full[half_h * cutlass.Int32(N_O_CHUNKS // 2) + cutlass.Int32(b // 2)].arrive()
        else:
            o_zeros = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(O_EPI_BLOCK_COLS)), cutlass.Float32)
            o_cur = o_zeros
            if tile_live:
                o_cur = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.O_OFF), cutlass.Float32), num=O_EPI_BLOCK_COLS)
            for b in cutlass.range_constexpr(N_O_EPI_BLOCKS):
                o_nxt = o_cur
                if cutlass.const_expr(O_EPI_PIPELINE and b + 1 < N_O_EPI_BLOCKS):
                    # Issue the next batch's load before waiting on this one (two live 32-fp32 batches).
                    if tile_live:
                        o_nxt = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.O_OFF + (b + 1) * O_EPI_BLOCK_COLS), cutlass.Float32),
                            num=O_EPI_BLOCK_COLS,
                        )
                if tile_live:
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                # Dead / trimmed rows are zeroed by a SELECT, never by `* beta` with beta = 0: a q-trimmed row whose Q memory
                # holds NaN (a poisoned padded tail) has S = P = O = NaN in TMEM, and NaN * 0 = NaN would reach the output
                # (test_two_by_two_trimmed_rows_with_nan_inputs_store_zero, the SM100 body's fix carried over: review P2).
                # The fp32-partials arm above selects per element.
                o_scaled = o_cur * beta
                if row_dead:
                    o_scaled = o_zeros
                o_half = o_scaled.to(OUT_STORAGE_DTYPE)
                subtile_rt = half_h * cutlass.Int32(N_O_CHUNKS // 2) + cutlass.Int32(b // 2)
                smem_off = (
                    subtile_rt * cutlass.Int32(O_SUBTILE_STRIDE_ELEMS)
                    + row_r * cutlass.Int32(TMA_O_GRANU_ELEMS_HOST)
                    + cutlass.Int32((b % 2) * O_EPI_BLOCK_COLS)
                )
                sO_base.subview(smem_off).data_ptr().store_swizzled(o_half, alignment=64, swizzle=_O_SMEM_SWIZZLE)
                if cutlass.const_expr(b % 2 == 1):
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_o_full[subtile_rt].arrive()
                if cutlass.const_expr(O_EPI_PIPELINE and b + 1 < N_O_EPI_BLOCKS):
                    o_cur = o_nxt
                elif cutlass.const_expr(b + 1 < N_O_EPI_BLOCKS):
                    if tile_live:
                        o_cur = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.O_OFF + (b + 1) * O_EPI_BLOCK_COLS), cutlass.Float32),
                            num=O_EPI_BLOCK_COLS,
                        )

        # LSE / Stats: one writer per row (column half 0; warp-uniform).
        if half_h == cutlass.Int32(0):
            if cutlass.const_expr(lse_tensor is None):
                pass  # has_lse=False: the Stats store is compiled out
            elif cutlass.const_expr(CFG.THD_VARLEN):
                _cu = cutlass.make_array_view(seq_kv_lens_tensor)
                _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
                _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
                if q_row_global < _s_q_b:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        lse_arr[_cu_q_b + q_row_global, head_idx] = lse  # token-major packed (T, H)
                    else:
                        if cutlass.const_expr(len(lse_tensor.shape) == 4):
                            lse_arr[batch_idx, head_idx, q_row_global, 0] = lse  # per-batch padded Stats
                        else:
                            lse_arr[cutlass.Int32(0), head_idx, _cu_q_b + q_row_global] = lse  # head-major packed
            else:
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_batch = _partial_batch(batch_idx, split_idx, n_batch)
                    lse_arr[lse_batch, row_head_idx, q_row_global] = lse

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx, split_idx = _decode_payload_split(
            nxt_q, nxt_hb, cta_id_x, n_q_supers, n_qh, n_batch, seq_kv_lens_tensor, qh_per_kh, seqlen_kv
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch, seq_q_lens_addr)
        bounds = _bounds_for_tile_split(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_id_x, seq_q_lens_addr, batch_idx, split_idx, HEADS_PER_TILE)

    # 128 local + 128 peer arrives per CTA = PAIR_LANES; both MMA warps wait before tcgen05.dealloc.
    bars.mb_tmem_dealloc.arrive_on_peer(cta_id_x ^ cutlass.Int32(1))
    bars.mb_tmem_dealloc.arrive()


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: cute.Pointer,
    meta_ptr: cute.Pointer,
    o_desc_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_ext: cutlass.Int64,
    scale_softmax_log2: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    seq_q_lens_addr: cutlass.Int64,
    thd_q_lens_ptr: Optional[cute.Pointer],
    thd_kv_lens_ptr: Optional[cute.Pointer],
    thd_lens_form: Optional[cutlass.Int32],
    o_partial_ptr: Optional[cute.Pointer],
    block_table_ptr: Optional[cute.Pointer],
    block_table_v_ptr: Optional[cute.Pointer],
    table_strides: Tuple[cutlass.Int64, cutlass.Int64],
    n_pages: cutlass.Int32,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    lse_kind: cutlass.Constexpr[str],
    paged_hnd: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Host entry (same positional ABI as sm100/prefill_d512_f16.py so the prepared binder serves both):
    device pointers, runtime extents and Int64 strides in, TMA encodes and launches out.  ``problem_size``
    = (B, QH, KH, SQ, SKV, 0); ``lse_kind`` in dense / token / head / padded; ``lse_ptr`` None compiles the
    Stats store out.  Paged slots are dead on this flavor (None)."""
    B, QH, KH, SQ, SKV, _ = problem_size
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        thd_q_lens_tensor,
        thd_kv_lens_tensor,
        o_partial_f32,
        _block_table_tensor,
        _block_table_v_tensor,
    ) = sdpa_operand_tensors(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        sinks_ptr,
        meta_ptr,
        o_desc_ptr,
        problem_size,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
        lse_strides,
        lse_ext,
        thd_q_lens_ptr,
        thd_kv_lens_ptr,
        thd_lens_form,
        o_partial_ptr,
        d_qk=d_qk,
        d_v=d_v,
        lse_kind=lse_kind,
        thd=CFG.THD_VARLEN,
        split_kv=SPLIT_KV,
        tensor_map_qwords=_TENSOR_MAP_QWORDS,
    )
    stride_order = (3, 2, 1, 0)
    # 64-row Q / O boxes (TILE_M), K sub-chunk boxes of this CTA's 64 kv rows, V boxes of 128 kv rows.
    qk_box_q = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, K_SUB_ROWS, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M // HEADS_PER_TILE, HEADS_PER_TILE, TMA_O_GRANU_ELEMS_HOST)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    def _create_tma_desc(tensor: cute.Tensor, box_dims, swizzle, order=stride_order):
        # Strides widened to Int64 explicitly (create_tensor_map_tiled_from_view scales dynamic strides in i32).
        return tmap.create_tensor_map_tiled(
            global_address=tensor.iterator.toint(),
            dtype=tensor.element_type,
            global_dims=tuple(tensor.shape[i] for i in order),
            global_strides=tuple(cutlass.Int64(tensor.stride[i]) * tensor.element_type.width // 128 for i in order[1:]),
            box_dims=tuple(box_dims[i] for i in order),
            swizzle=swizzle,
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_q_desc = _create_tma_desc(q_tensor, qk_box_q, _tma_swz(CFG.Q_SWZ_BYTES))
    tma_k_desc = _create_tma_desc(k_tensor, qk_box_k, _tma_swz(CFG.K_SWZ_BYTES))
    tma_v_desc = _create_tma_desc(v_tensor, vo_box_v, _tma_swz(CFG.V_SWZ_BYTES))
    _o_box = list(vo_box_o)
    if _FP32_PARTIALS:
        _o_box[-1] = max(1, _o_box[-1] * CFG.BPE_O // 4)
    tma_o_desc = _create_tma_desc(o_tensor, tuple(_o_box), _tma_swz(CFG.O_SWZ_BYTES))

    # One 64-row Q super per CTA; CGA_M supers per cluster.  PackGQA: SQ*G packed rows per packed head.
    rows_per_cluster = CGA_TILE_M
    q_clusters = (SQ * HEADS_PER_TILE + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CGA_M
    q_supers = q_clusters * CFG.Q_SUPERS_PER_CLUSTER
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD setup launch (the one declared auxiliary launch): metadata + per-batch O descriptors +
        # the persistent scheduler's live-unit total / claim counter.  Units are CGA_TILE_M rows tall.
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
            cutlass.Int32(QH // HEADS_PER_TILE),
            cutlass.Int32(B),
            cutlass.Int64(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,  # persistent cluster count; also seeds the claim counter
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (
            (grid_q_supers, QH // HEADS_PER_TILE, B * SPLIT_KV)
            if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL)
            else (grid_q_supers * (QH // HEADS_PER_TILE) * B * SPLIT_KV, 1, 1)
        )
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
        cutlass.Int32(QH // HEADS_PER_TILE),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        seq_q_lens_addr,
        o_partial_f32,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself
LSE_KINDS = ("dense", "token", "head", "padded")


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    lse_kind: str = "dense",
    paged_hnd: bool = False,
) -> Callable:
    """Compile the host entry for one layout kind (same key surface as sm100/prefill_d512_f16.py: the
    head-dim ENVELOPE, whether the Stats store exists, the Stats layout kind)."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if not (0 < d_qk <= CFG.TILE_K and 0 < d_v <= CFG.TILE_O):
        raise ValueError(f"d512 envelope: need 0 < d_qk <= {CFG.TILE_K} and 0 < d_v <= {CFG.TILE_O}; got ({d_qk}, {d_v})")
    if (d_qk * CFG.BPE) % 16 != 0 or (d_v * CFG.BPE_O) % 16 != 0:
        raise ValueError(f"d512 envelope: d_qk*BPE and d_v*BPE must be 16-byte multiples (TMA global-stride rule); got ({d_qk}, {d_v}) at BPE={CFG.BPE}")
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("split_kv > 1 requires has_lse=True (the per-split LSE drives the combine)")
    if lse_kind not in LSE_KINDS:
        raise ValueError(f"lse_kind must be one of {LSE_KINDS}; got {lse_kind!r}")
    if has_lse and (lse_kind == "dense") == bool(CFG.THD_VARLEN):
        raise ValueError("lse_kind 'dense' is the dense form; 'token' / 'head' / 'padded' are the THD forms")
    if paged_hnd:
        raise ValueError("paged_hnd is a paged-KV specialization (not served by the d512 2x2 kernel)")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i32 = cutlass.Int32(0)
    i64_3 = (cutlass.Int64(0),) * 3
    thd = bool(CFG.THD_VARLEN)
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(cutlass.Float32 if _FP32_PARTIALS else OUT_STORAGE_DTYPE),
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Float32),
        P(cutlass.Int32),
        P(cutlass.Int64),
        (0, 0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        cutlass.Int64(0),
        cutlass.Float32(0.0),
        i32,
        cutlass.Int64(0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        i32 if thd else None,
        P(cutlass.Float32) if _FP32_PARTIALS else None,
        None,
        None,
        (cutlass.Int64(0), cutlass.Int64(0)),
        i32,
        d_qk,
        d_v,
        lse_kind,
        paged_hnd,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd",
    )


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    lse_kind = "token" if CFG.THD_VARLEN else "dense"
    print(f"[d512_f16_2x2_sm107] CGA_M={CFG.CGA_M} KV_SHARE={KV_SHARE} DESC_VERSION={DESC_VERSION} smem={_SMEM} compile lse_kind={lse_kind}", flush=True)
    fn = compile(lse_kind=lse_kind)
    print(f"[d512_f16_2x2_sm107] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d512_f16_2x2_sm107] compiled -- run validation via test/python/sdpa/frost/test_sdpa_fwd_d512_2x2_sm107.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
