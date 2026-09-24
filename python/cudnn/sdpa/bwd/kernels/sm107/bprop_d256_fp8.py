# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SM107 (Rubin) SDPA backward, d_qk = d_v = 256, per-tensor FP8 E4M3: dV in-kernel + bf16 dS workspace.

One cga2 pair (CGA_M=2, CGA_N=1, CTA_MMA=2) owns one 256-row kv block of one (batch, head) and walks the q tiles that
attend it.  It computes **dV in-kernel** (TMEM accumulator over the q loop, stored per Q-head) and writes **dS** to a
GMEM workspace ``[B, H_chunk, S_kv, S_q]`` in **bf16**; the adapter runs the dK = dS.Q and dQ = dS^T.K GEMMs over that
workspace (``kernels/bprop_matmul_blackwell.py``), folds GQA (``bprop_chain_common.dkv_reduce``) and quantizes the
gradients.  lane = kv row (S = [kv, q]).  Twelve warps per CTA, ONE warp-specialized body:

    MMA leader (lookahead order, perf-critical):
        Q.K[q_lo] -> { dO.V[i] ; Q.K[i+1] ; P.dO[i] } -> dO.V[last] ; P.dO[last]
        BMM1 S  = K . Q^T   -> S_acc[kv, q]   TMEM [  0, 128)   A = K (fixed per tile), B = Q[i]          mma_ss
        BMM1 dP = V . dO^T  -> dP[kv, q]      TMEM [128, 256)   A = V (fixed per tile), B = dO[i]         mma_ss
        BMM2 dV = P . dO    -> dV[kv, d_v]    TMEM [256, 512)   A = fp8 P (TMEM ring), B = dO_dv[i] (BT)  mma_ts
    8 compute warps (2 warpgroups x 4, each wg owns a 64-wide q half), per q iteration:
        softmax : P_s = exp2(S * scale_log2 - lse_s)         (lse_s = lse*log2e - log2(scale_s), so P_s = P * scale_s)
                  -> transposed per-cell mask -> e4m3 -> TMEM P ring slot -> mb_p_ready (the dV BMM2 starts NOW)
        dsoftmax: dS = (dP * dp_scale - delta_s) * P_s        (= attn_scale * P * (dP_true - delta), fp32)
                  -> amax_dP fold -> bf16 -> sdS SMEM ring -> TMASTG -> GMEM workspace
    per kv tile (post q loop): dV epilogue = dV_acc * descale_s * descale_dO -> amax_dV fold -> (* scale_dV -> e4m3 |
        bf16 / fp16) -> sdV SMEM (aliases K+V, dead by then) -> TMASTG -> GMEM dV
    TMALDG: K, V once per kv tile; Q, dO (dP view) and dO_dv (dV view, second load of the SAME dO) per q iteration.
    TMASTG: dS ring slot -> workspace per q iteration; dV -> GMEM per kv tile.
    Scheduler warp: try_cancel protocol FUSED with the lse / delta SMEM prefetch ring (lane = q col).

FP8 contract (cuDNN ``sdpa_fp8_backward``; every scale is a 1-element fp32 DEVICE tensor read in-kernel, never a
host fold):  S_true = S_acc * descale_q * descale_k;  P_s = e4m3(P * scale_s) is the BMM2 A operand;
dV_true = dV_acc * descale_s * descale_dO;  dP_true = dP_acc * descale_v * descale_dO;  dS = attn_scale * P *
(dP_true - delta) with delta = rowsum(dO * O) in TRUE units (the adapter's fp8 ``dot_do_o`` arm applies
descale_o * descale_dO);  amax_dV = max |dV_true|;  amax_dP = max |dS| over the fp32 value the bf16 workspace
holds (attn_scale folded, pre-cast).  ``DTYPE_O`` = E4M3 (default: dV_q = e4m3(dV_true * scale_dV)) or BF16 / FP16
(the pre-quantization output for the bitwise A/B against the pre-port kernel; ``scale_dV`` is then not applied).

Workspace head-chunking: the dS workspace is the dominant allocation (B * H * S_kv * S_q * 2 bytes).  ``compile(...,
qh_chunk=)`` sizes the dS descriptor + the grid head axis to a chunk of heads; the runtime ``head_base`` offsets ALL
full-tensor I/O (Q / K / V / dO / dV / lse / delta -> head_idx + head_base) while dS stays chunk-local.  The adapter
loops ``H_q // qh_chunk`` launches (per batch entry when B > 1, or over the full batch when it fits).

--------------------------------------------------------------------------------------------------------------------
TMEM (one ``tcgen05.alloc.cta_group::2`` of TOTAL_COLS = 576 per CTA, ``is_exclusive=True``; ``config_sm107.tmem_layout``)
    [  0, 128)  S_acc   fp32 [kv, q]     BMM1 S  ->  softmax  ``tmem_load_tile(S_OFF + q_half_off, 64)``
    [128, 256)  dP      fp32 [kv, q]     BMM1 dP ->  dsoftmax ``tmem_load_tile(dP_OFF + q_half_off, 64)``
    [256, 512)  dV_acc  fp32 [kv, d_v]   BMM2 dV (accumulate=(q_iter > q_lo)) -> epilogue
    [512, 576)  fp8 P ring, 2 x P_COLS=32 (= TILE_N * BPE / 4)   softmax ``tcgen05_st`` -> BMM2 A operand (``mma_ts``)

SMEM (declaration order == ``config_sm107.smem_layout``; every slab 1024-B aligned; byte offsets for the fp8 body)
    #  buffer      dtype x elems               KiB @off   writer (how)            reader (how)           lane stride     swizzle + WHY
    1  sQ          e4m3 x 3 x (64 x 256)        48 @  0   TMA (box 1,64,1,128 x2)  MMA desc, B of BMM1 S  --              128 B s128b: TMA write + MMA desc read (job 1)
    2  sdO         e4m3 x 3 x (64 x 256)        48 @ 48   TMA (box 1,64,1,128 x2)  MMA desc, B of BMM1 dP --              128 B, job 1
    3  sdOdv       e4m3 x 3 x (128 x 128)       48 @ 96   TMA (box 1,128,1,128)    MMA desc, B (BT) of BMM2 --            128 B, job 1 (a SECOND copy of dO: the s128b XOR of a
                                                                                                                          256-B-row view and a 128-B-row view differ)
    4a sK          e4m3 x 128 x 256             32 @144   TMA (box 1,128,1,128 x2) MMA desc, A of BMM1 S  --              128 B, job 1
    4b sV          e4m3 x 128 x 256             32 @176   TMA                      MMA desc, A of BMM1 dP --              128 B, job 1  (alias offset kBufferElems in ELEMENTS)
    4c sdV (=4a+4b post-loop) OUT x 128 x 256   32|64@144 compute lanes store_swizzled(Swizzle(3,4,3))  TMA store  64 elems = 128 B (bf16) / 64 B (e4m3) per lane per block
                                                                                                                          Swizzle(3,4,3): row 128 B -> banks spread (job 2) AND matches the s128b store descriptor (job 1)
    5  sStats      fp32 x 2 x 256                2 @208   scheduler lanes (4 B stride, conflict-free)  compute lanes, same address on 32 lanes (broadcast)  LINEAR (job 2 by arithmetic; no descriptor)
    6  sdS         bf16 x 3 x (128 x 128)       96 @210   compute lanes store_swizzled(Swizzle(3,4,3)) at slot + wg*8192 + tid*64   TMA store box (1,1,128,64) x2 subtiles
                                                                                                          128 B per lane   Swizzle(3,4,3) row 128 B: job 2 spread + job 1 matches s128b
    ~0.6 KiB scaffolding (mbarriers, scheduler slots, TMEM pointer) -> 306 KiB slabs + 2 KiB budget < 327 KiB (Rubin
    oversized cap; the launcher sets ALLOW_OVERSIZED_SHARED_MEMORY).  Every tcgen05 descriptor ROOT (sQ / sdO / sdOdv
    stages, sK, sV) is < 256 KiB -> DESC_VERSION = 0 (derived, never a literal).

BARRIER TABLE (lane ledger: SUM(issuing lanes) == init, per CTA.  A phase = one q iteration for the per-q rings, one kv
tile for the per-tile bars.  ``ldr+elect`` = ``pred=is_leader & elect_sync()``; ``pred`` = ``pred=elect_p`` (one lane);
``bare`` = every lane of the calling warps; ``elect`` = ``if elect_sync():``.  cga2 tensor-TMA bytes land on the LEADER's
mbar (P9), so only the leader arms ``expect_tx``; the follower's TMA_LOAD bars are initialised but never armed or waited.)

  bar                 stages  producer            guard        issuing lanes / phase    init            consumer (waits)          scope
  mb_q_full           3       TMA_LOAD (TMALDG)   ldr+elect    1 (leader CTA only)      ONE_LANE=1      MMA leader                LOCAL  tx = 2 CTAs x 16 KiB = 32768
  mb_do_full          3       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 32768
  mb_dodv_full        3       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 32768
  mb_k_full / v_full  1       TMA_LOAD            ldr+elect    1                        1               MMA leader                LOCAL  tx 65536 each
  mb_q_empty          3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both CTAs)        LOCAL  drained x3 at exit (P15)
  mb_do_empty         3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x3
  mb_dodv_empty       3       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x3
  mb_k_empty/v_empty  1       MMA_COMMIT mcast    pred         1 per target CTA         1               TMALDG (both)             LOCAL  drained x1 (port fix F2)
  mb_s_acc_full       1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 lanes (both)  LOCAL
  mb_dp_full          1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 (both)        LOCAL
  mb_dv_ready         1       MMA_COMMIT mcast    pred         1 per target CTA         1               softmax 256 (both)        LOCAL
  mb_s_acc_empty      1       LEADER (softmax)    bare         256 lanes x 2 CTAs       512             MMA leader (pre-armed)    LEADER drained x1 at exit (F3)
  mb_dp_empty         1       LEADER (softmax)    bare         256 x 2                  512             MMA leader (pre-armed)    LEADER drained x1 (F3)
  mb_p_ready          2       LEADER (softmax)    bare         256 x 2                  512             MMA leader                LEADER
  mb_dv_acc_empty     1       LEADER (softmax)    bare         256 x 2                  512             MMA leader                LEADER
  mb_stats_full       2       THREAD (scheduler)  bare         32 (one warp)            ONE_WARP=32     softmax 256               LOCAL
  mb_stats_empty      2       THREAD (softmax)    bare         256                      256             scheduler (pre-armed)     LOCAL
  mb_ds_smem_full     3       THREAD (softmax)    bare         256                      256             TMASTG                    LOCAL
  mb_ds_smem_empty    3       THREAD (TMASTG)     elect        1                        1               softmax (pre-armed)       LOCAL
  mb_dv_stg_full      1       THREAD (softmax)    bare         256                      256             TMASTG                    LOCAL
  mb_dv_stg_empty     1       THREAD (TMASTG)     elect        1                        1               TMALDG (pre-armed; F1)    LOCAL  drained x1
  mb_tmem_dealloc     1       THREAD (softmax)    bare local + bare arrive_on_peer  256 local + 256 peer = 512  MMA warp of each CTA  LOCAL
  sched.mb_scheduler  2       expect_tx 16 B      elect (cga-first CTA arms BOTH)  1 per CTA          1               every persistent warp     LOCAL
  sched.mb_read_tile_id 2     read_tile_id_arrive one predicated arrive per calling warp on EVERY CTA   READ_TILE_ARRIVERS_TOT=21  scheduler
        = leader (8 softmax + MMA + TMALDG + TMASTG = 11) + follower (8 + TMALDG + TMASTG = 10); the scheduler warp never credits.

Named barrier 1 (288 = 32 x 9 threads): MMA warp ``barrier_cta_arrive`` <-> softmax ``barrier_cta_sync`` publishes the
TMEM base after ``tmem_alloc``.  Every LEADER-scope arrive is preceded by ``tcgen05_wait(LOAD)`` (the arrive has no data
dependency on the loaded registers; without the wait ptxas may schedule it between two LDTMs).  ``tcgen05_wait(STORE)``
precedes ``mb_p_ready``.  P15 drains at the tail of the CONSUMER: TMALDG drains the five multicast ``_empty`` rings, the
MMA leader drains the two pre-armed LEADER-scope rings, the softmax's ``mb_tmem_dealloc`` is waited by both CTAs'
MMA warps.  Port deltas vs the pre-port body (its barrier ledger: ``frost_dev/plans/bprop_d256_sm107_BARRIERS_fp8.md``):
F1 the TMALDG now WAITS ``mb_dv_stg_empty`` before the K load (the dV staging aliases K+V; the K load could land under
the dV TMA store's SMEM read); F2 K/V ``_empty`` drained; F3 ``s_acc_empty`` / ``dp_empty`` drained; F4 the TMA_LOAD
arms and the K/V release commits are value-predicated (P16); dead knobs and the dead named barrier 2 deleted.

## Launch ABI

``compile(b, qh, kh, sq, skv, qh_chunk=0, has_amax=True)`` (all ``lru_cache``-d Python ints; ``qh_chunk=0`` means
``qh``) returns a callable taking, POSITIONALLY (torch tensors bind through tvm-ffi):

    fn(q, do, k, v, dv, ds_ws, lse, delta,
       descale_q, descale_k, descale_v, descale_do, descale_s, scale_s, scale_dv,
       amax_dv | None, amax_dp | None,
       (b, qh, kh, sq, skv, qh_chunk),          # the compile-time problem_size tuple, repeated at the call
       attn_scale, attn_scale_log2e,            # cutlass.Float32: softmax scale, and attn_scale * log2(e)
       head_base, seqlen_kv_real,               # cutlass.Int32: first full-tensor head of this chunk; REAL S_kv (<= skv)
       stream=<CUstream>)

    q, k        e4m3  [B, S_q, H_q, 256] / [B, S_kv, H_kv, 256]   BSHD, d contiguous, seq/head strides 16-B multiples
    v           e4m3  [B, S_kv, H_kv, 256]
    do          e4m3  [B, S_q, H_q, 256]                          (loaded twice: dP view and dV view)
    dv          OUT   [B, S_kv, H_q, 256]   per Q-HEAD partial (GQA fold is the adapter's dkv_reduce); e4m3 by default
    ds_ws       bf16  [B, qh_chunk, S_kv, S_q]   chunk-local head axis; holds attn_scale * P * (dP_true - delta)
    lse         fp32  [B, H_q, S_q]  NATURAL-log LSE of the forward (the kernel applies log2e); rows past the real
                                     S_q must read +inf (P = 0) when the adapter pads S_q up to a multiple of 128
    delta       fp32  [B, H_q, S_q]  rowsum(dO * O) in TRUE units (fp8 inputs: * descale_o * descale_dO), UNSCALED
    descale_* / scale_*   fp32 [1]  device tensors (never host-folded)
    amax_dv, amax_dp      fp32 [1]  ZEROED by the caller on the launch stream; atomicMax'd (as int32 bit patterns of
                                     non-negative fp32) over rows kv < seqlen_kv_real; None when has_amax=False
    Shapes: sq % 128 == 0, skv % 256 == 0 (the adapter pads; a padded S_kv REQUIRES the MASK_PADDED specialization
    -- ``TemplateParams.seq_kv_lens_present=True`` -- with ``seqlen_kv_real`` the uniform real length, or the zero-
    filled pad rows produce P != 0 and a non-zero dS), qh % kh == 0, qh % qh_chunk == 0.
    Grid: NATURAL ``(skv/256 * 2, qh_chunk, b)``; LPT / LPT_L2 the flat ``(skv/256 * qh_chunk * b * 2, 1, 1)``;
    cluster (2, 1, 1); 384 threads; SMEM ``config_sm107.kernel_smem_bytes(CFG)`` (oversized mode).
    Masks are TemplateParams (module-load time): ``window_right=0`` causal, ``window_left=W`` SWA, ``bottom_right``.
"""

from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

import cuda.bindings.driver as _cuda_driver  # noqa: F401
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from cudnn.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, Scope, advance, arrive_expect_tx, cga_arrive, cga_wait, wait
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_E4M3, DTYPE_FP16
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import (
    MASK_CAUSAL,
    MASK_FORM_BITS,
    MASK_FORM_CELLS,
    MASK_FORMS,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    apply_mask_words,
    band_mask_words,
    compute_q_loop_bounds,
)
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import abs_max_tree, fmax_f32, opaque_f32_zero, tmem_load_tile
from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, Sched, read_clc_payload, read_tile_id_arrive
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.sdpa.bwd.config_sm107 import FAMILY_FP8, TMEM_IS_EXCLUSIVE, TemplateParams, buffer_elems, desc_version, make_cfg_d256_bwd, tmem_layout

# Config comes from the FROST template loader, never an environment variable: the loader injects FROST_TEMPLATE_PARAMS
# before this body runs; the default keeps a plain `import` usable as a standalone driver (E4M3 is the only io this body
# serves, so the standalone default names it -- the shared record's BF16 default belongs to the f16 body).
PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=DTYPE_E4M3))
CFG = make_cfg_d256_bwd(PARAMS, FAMILY_FP8)
_B = buffer_elems(CFG)
LAYOUT = tmem_layout(CFG)


def _needs_desc_v1(cfg) -> bool:
    """True when any tcgen05 descriptor ROOT of the SMEM table (``config_sm107.desc_roots``) starts at or past the 14-bit
    version-0 window (256 KiB).  The fp8 body's last root is sV at 176 KiB, so this is False -- derived, not assumed."""
    return desc_version(cfg) == 1


# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module: ONE decision point wired into every construction
# below, never a per-tile literal (rules/mma-tma-matrix.md S6; the d512 MXFP8 sibling shipped NaN on 100 % of cells from
# one SmemTile declared under a comment claiming the version it did not pass).
DESC_VERSION: int = 1 if _needs_desc_v1(CFG) else 0

# Retry form of the per-q-iteration RING waits (q / dO / dO_dv _full/_empty, s_acc_full/_empty, dp_full/_empty, p_ready,
# stats_*, ds_smem_*): every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``.  Whole-tile waits (k/v, dv_*,
# tmem_dealloc, the scheduler payload) and the end-of-kernel drains keep the default sleeping form.  The sign of the
# hint-less spin is a MEASURED per-kernel fact (rules/frost-tile-dsl.md S8b); False until this body has its own A/B/A.
SPIN_RING_WAITS: bool = False

# Per-cell mask lowering, ONE constant per kernel (the DESC_VERSION discipline): every masked site passes form=MASK_FORM.
# "bits" = one keep-word per 32 q columns + register-to-predicate select (R2P + 1 FSEL per cell); "cells" = the per-cell
# compare + select the pre-port body carried.  Same masked set, same zero -> bitwise identical dS / dV.
MASK_FORM: str = MASK_FORM_BITS

# --- dtype dispatch (the config validated the codes; these are the DSL types they name) ---------------------------------
STORAGE_DTYPE = cutlass.Float8E4M3FN  # CFG.DTYPE_QKV == DTYPE_E4M3 (the fp8 body is E4M3-only)
MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
DS_STORAGE_DTYPE = cutlass.BFloat16  # CFG.DTYPE_DS == DTYPE_BF16: the bf16 stage-3 GEMMs read the workspace unchanged
_OUT_DTYPES = {DTYPE_E4M3: cutlass.Float8E4M3FN, DTYPE_BF16: cutlass.BFloat16, DTYPE_FP16: cutlass.Float16}
OUT_STORAGE_DTYPE = _OUT_DTYPES[CFG.DTYPE_O]
OUT_IS_FP8: bool = CFG.DTYPE_O == DTYPE_E4M3

# Rubin's dense-FP8 MMA runs K=64 per instruction and every idesc below passes k_dim=CFG.IDESC_K_DIM (validated == 1 with
# TILE_K_HW == 64 by the config).  The pairing is arch-OPPOSITE of Blackwell and a mismatch scrambles accumulator ROWS
# silently, so the tripwire is repeated here where the idescs are built (rules/mma-tma-matrix.md S1).
if CFG.TILE_K_HW_BMM1 != 64 or CFG.TILE_K_HW_BMM2 != 64 or CFG.IDESC_K_DIM != 1:
    raise ValueError(
        f"{__name__}: Rubin dense FP8 needs TILE_K_HW=64 with idesc k_dim=1; got {CFG.TILE_K_HW_BMM1}/{CFG.TILE_K_HW_BMM2}, k_dim={CFG.IDESC_K_DIM}"
    )
if MASK_FORM not in MASK_FORMS:
    raise ValueError(f"{__name__}: MASK_FORM must be one of {MASK_FORMS}, got {MASK_FORM!r}")

# --- per-CTA buffer geometry (config_sm107.buffer_elems; never re-derived here) ------------------------------------------
_M_PER_CTA = _B._M_PER_CTA  # 64 q rows per CTA of the Q / dO N-split
qBufferElems = _B.qBufferElems  # 64 x 256
dOBufferElems = _B.dOBufferElems  # 64 x 256 (== TILE_N x TILE_O / CTA_MMA for the dV view)
kBufferElems = _B.kBufferElems  # 128 x 256
vBufferElems = _B.vBufferElems  # 128 x 256
dSBufferElems = _B.dSBufferElems  # 128 kv x 128 q, elements of DS_STORAGE_DTYPE
dVBufferElems = _B.dVBufferElems  # 128 kv x 256 d_v, elements of OUT_STORAGE_DTYPE
qTmaTransactionBytes = _B.qTmaTransactionBytes  # both peers' bytes land on the leader's mbar (x CTA_MMA)
dOTmaTransactionBytes = _B.dOTmaTransactionBytes  # also the dO_dv view: TILE_N x (TILE_O / CTA_MMA) x BPE x CTA_MMA
kTmaTransactionBytes = _B.kTmaTransactionBytes
vTmaTransactionBytes = _B.vTmaTransactionBytes
TMA_QK_ITERS = _B.TMA_QK_ITERS  # 2 x 128-elem subtiles per 256-B fp8 row
TMA_VO_ITERS = _B.TMA_VO_ITERS
TMA_QK_GRANU_ELEMS = _B.TMA_QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _B.TMA_VO_GRANU_ELEMS
TMA_VO_SG1_ITERS = _B.TMA_VO_SG1_ITERS  # the dV view: 128 d_v per CTA = 1 subtile
TMA_VO_SG1_GRANU_ELEMS = _B.TMA_VO_SG1_GRANU_ELEMS
DV_D_BLOCK = _B.DV_D_BLOCK  # d_v elems per 128-B store subtile: 128 at e4m3 out, 64 at bf16 / fp16 out
TMA_DV_ITERS = _B.TMA_DV_ITERS  # 2 (e4m3) / 4 (bf16) subtiles per dV row
DV_BLOCK_SLAB = _B.DV_BLOCK_SLAB  # TILE_M x DV_D_BLOCK: one store subtile's slab
P_TMA_ITERS = _B.P_TMA_ITERS  # 2: a 128-col bf16 dS row is two 128-B subtiles
P_D_BLOCK = _B.P_D_BLOCK  # 64 q cols per dS subtile == one warpgroup's q half
P_BLOCK_ELEMS = _B.P_BLOCK_BYTES  # TILE_M x P_D_BLOCK elems per dS subtile (the pre-port misnomer is the config's)
STATS_SLOT_ELEMS = _B.STATS_SLOT_ELEMS
STATS_LSE_OFF = _B.STATS_LSE_OFF
STATS_DOT_OFF = _B.STATS_DOT_OFF
_SMX_CHUNK = _B._SMX_CHUNK  # 64 q cols per compute warpgroup
_LDTM_NUM = _B._LDTM_NUM  # tcgen05.ld.32x32b.x64 granule
_KV_BLOCK_ROWS = _B._KV_BLOCK_ROWS  # 256 kv rows per cga2 pair
CGA_SIZE = _B.CGA_SIZE  # 2
LEADING_BYTE_OFFSET_QK = _B.LEADING_BYTE_OFFSET_QK
STRIDE_BYTE_OFFSET_QK = _B.STRIDE_BYTE_OFFSET_QK
LEADING_BYTE_OFFSET_dO = _B.LEADING_BYTE_OFFSET_dO
STRIDE_BYTE_OFFSET_dO = _B.STRIDE_BYTE_OFFSET_dO
LEADING_BYTE_OFFSET_dS = _B.LEADING_BYTE_OFFSET_dS
STRIDE_BYTE_OFFSET_dS = _B.STRIDE_BYTE_OFFSET_dS
LEADING_BYTE_OFFSET_dO_SG1 = _B.LEADING_BYTE_OFFSET_dO_SG1  # BT=true B operand: K x swz (B_PC_COLS // 8 > 8)
STRIDE_BYTE_OFFSET_dO_SG1 = _B.STRIDE_BYTE_OFFSET_dO_SG1
SMEM_LAYOUT_Q = _B.SMEM_LAYOUT_Q
SMEM_LAYOUT_dO = _B.SMEM_LAYOUT_dO
SMEM_LAYOUT_K = _B.SMEM_LAYOUT_K
SMEM_LAYOUT_V = _B.SMEM_LAYOUT_V
SMEM_LAYOUT_dS = _B.SMEM_LAYOUT_dS
SMEM_LAYOUT_dV = _B.SMEM_LAYOUT_dV

# The lane-written staging buffers (dS ring, dV epilogue) are laid out as 128-B-row subtiles under Swizzle(3, 4, 3):
# MBase + SShift = 4 + 3 = 7 = log2(128 B), so 32 lanes writing one row each hit 32 different bank groups (job 2), and it
# is the s128b pattern the TMA-store descriptors of both buffers decode (job 1).  Both jobs by ONE swizzle.
STAGING_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_DV_EPI_CHUNK = 64  # d_v cols per epilogue TMEM load / store (register cap); each wg owns TILE_O / SOFTMAX_WARPGROUPS
_DV_CHUNKS_PER_WG = (CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS) // _DV_EPI_CHUNK  # 2
if (CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS) % _DV_EPI_CHUNK or DV_D_BLOCK % _DV_EPI_CHUNK:
    raise ValueError(
        f"{__name__}: the dV epilogue walks {_DV_EPI_CHUNK}-col chunks; TILE_O/WGS={CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS}, DV_D_BLOCK={DV_D_BLOCK}"
    )
if P_D_BLOCK != _SMX_CHUNK:
    raise ValueError(f"{__name__}: a dS store subtile ({P_D_BLOCK} q cols) must be exactly one warpgroup's q half ({_SMX_CHUNK})")

# log2(e): the softmax uses exp2, so P * scale_s = exp2(S_acc * (attn_scale * descale_q * descale_k * log2e) - (lse * log2e
# - log2(scale_s))).  attn_scale * log2e rides in the `attn_scale_log2e` scalar; the lse fold happens IN-KERNEL in the
# scheduler-stats warp (the host passes natural-log lse).
_LOG2E = 1.4426950408889634
# CLC response payload the scheduler ring's expect_tx arms (tile_dsl/scheduler.py uses the same 16).
_CLC_RESPONSE_BYTES = 16

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2  # CTA_MMA == 2 (validated by the config)

# --- named arrival-count constants (P3) --------------------------------------------------------------------------------------
ONE_LANE = CFG.ONE_LANE  # 1
ONE_WARP = CFG.ONE_WARP  # 32
SOFTMAX_LANES = CFG.SOFTMAX_LANES  # 256 = every compute lane of ONE CTA (both warpgroups)
SOFT_X_CTA_MMA = CFG.SOFT_X_CTA_MMA  # 512 = every compute lane of BOTH CTAs (the LEADER-scope counts)
MMA_COMMIT_ARRIVES = CFG.MMA_COMMIT_ARRIVES  # 1 per target CTA per multicast commit (one elected lane)
READ_TILE_ARRIVERS_TOT = CFG.READ_TILE_ARRIVERS_TOT  # 21, derivation in config_sm107.read_tile_arrivers_tot
_COMPUTE_WARPS = CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS  # 8
_NAMED_BAR_TMEM_ID = 1  # MMA warp <-> compute warps: TMEM base publish
_NAMED_BAR_TMEM_THREADS = 32 * (_COMPUTE_WARPS + 1)  # 288


# === Bars -- the mbarrier inventory (see the BARRIER TABLE in the module docstring) ==================================
class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_do_full: object
    mb_do_empty: object
    mb_dodv_full: object
    mb_dodv_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object
    mb_s_acc_full: object
    mb_s_acc_empty: object
    mb_dp_full: object
    mb_dp_empty: object
    mb_p_ready: object
    mb_stats_full: object
    mb_stats_empty: object
    mb_ds_smem_full: object
    mb_ds_smem_empty: object
    mb_dv_ready: object
    mb_dv_acc_empty: object
    mb_dv_stg_full: object
    mb_dv_stg_empty: object
    mb_tmem_dealloc: object


def _make_bars(CFG) -> Bars:
    """Allocate the SMEM mbarriers with the init counts of the BARRIER TABLE.  The LEADER-scope bars are initialised to
    the cluster-wide count on BOTH CTAs (the follower never waits its copy; an unconditional constant beats a runtime
    select, P8).  Depths come from the config (``config_sm107.mbar_stage_counts`` lists the same inventory)."""

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_do_full=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_empty=MBarrier(_alloc(CFG.STAGES_dO), stages=CFG.STAGES_dO, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dodv_full=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_dodv_empty=MBarrier(_alloc(CFG.STAGES_dO_DV), stages=CFG.STAGES_dO_DV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # S / dP accumulators are single-buffered (STAGES_TMEM_S == 1): the cross-iteration overlap is the MMA order.
        mb_s_acc_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_s_acc_empty=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dp_full=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dp_empty=MBarrier(_alloc(CFG.STAGES_TMEM_S), stages=CFG.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        # fp8 P ring (2 TMEM stages): softmax -> MMA "P[slot] stored".  No p_empty: slot i % 2 is rewritten at q-iter
        # i + 2 only after mb_s_acc_full[i + 2], whose commit orders after P.dO[i] read the slot (in-order MMA stream).
        mb_p_ready=MBarrier(_alloc(CFG.STAGES_TMEM_P), stages=CFG.STAGES_TMEM_P, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        # lse / delta SMEM prefetch ring: scheduler warp (32 lanes) -> compute lanes (256).
        mb_stats_full=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=ONE_WARP, producer=Producer.THREAD),
        mb_stats_empty=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        # dS SMEM ring: compute lanes store_swizzled -> TMASTG TMA store (one elected lane releases the slot).
        mb_ds_smem_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_ds_smem_empty=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=ONE_LANE, producer=Producer.THREAD),
        # dV epilogue (once per kv tile): MMA -> compute warps -> TMASTG -> (F1) TMALDG.
        mb_dv_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dv_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dv_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_dv_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.THREAD),
        # 256 local bare arrives + 256 relaxed cluster arrives from the partner's compute lanes, on EACH CTA.
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.THREAD),
    )


# === Mask plumbing (the backward is the TRANSPOSE of the forward) =====================================================
# lane = kv row, inner loop = q tile.  A fixed kv block [kvb, kvb + 256) bounds WHICH q tiles attend (`_q_loop_bounds`
# skips the rest) and `_mask_p_chunk` zeroes P on the masked (kv, q) cells of the boundary tiles.  P = 0 makes dV / dS /
# dK / dQ inherit the mask.  Every MASK_* arm is a const_expr: the dense specialization emits no mask IR at all.


def _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv):
    """``[q_lo, q_hi)``: the q tiles that attend this kv block, via ``tile_dsl.mask.compute_q_loop_bounds`` (the same band
    the forward applied: causal keeps kv <= q + diag, SWA keeps kv >= q + diag - W with the bottom-right anchor on BOTH
    edges; padding masks kv ROWS per lane and leaves the q range alone), then the empty-kv-block clamp: a block no q attends
    (top-left causal with S_kv > S_q) would run zero q iterations, and an N = 0 tile hangs the UNCONDITIONAL MMA prologue on
    mb_q_full (the TMA loop made no load) -- a runtime `if` cannot skip it without breaking every ring's per-tile balance.
    So q_lo is clamped in range and N is FORCED >= 1: the single forced tile is fully masked (P = 0 -> dV += 0, dS = 0:
    correct zeros, dV_acc overwritten via accumulate=False) and every per-iteration ring stays balanced (P14).  A no-op for
    non-empty blocks.  Uniform across the pair (kv_block_base is the cluster's kv base)."""
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return cutlass.Int32(0), n_q_tiles
    b = compute_q_loop_bounds(
        kv_block_base,
        seqlen_q,
        seqlen_kv,
        n_q_tiles,
        CFG.SWA_WINDOW,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        _KV_BLOCK_ROWS,
        bottom_right=bool(CFG.CAUSAL_BOTTOM_RIGHT),
        window_right=0,
    )
    q_lo = cute.math.min(b.lo, n_q_tiles - cutlass.Int32(1))
    q_hi = cute.math.max(b.hi, q_lo + cutlass.Int32(1))
    return q_lo, q_hi


def _mask_p_chunk(reg_P, kv_abs, q_col_base, seqlen_kv, causal_diag, N: int):
    """Zero P on masked (kv = lane, q = col) cells of an ``N``-wide q chunk starting at absolute q ``q_col_base``.

    The TRANSPOSE of ``tile_dsl.mask.apply_mask_chunk`` (row = kv, col = q), with ZERO as the masked value:
      causal : kv_abs > q + diag          (key past the query; diag = S_kv - S_q under bottom-right, else 0)
      SWA    : kv_abs < q + diag - W      (key left of the window; the same bottom-right anchor as the forward)
      padded : kv_abs >= seq_kv_len       (per-lane pad row -> the whole row is masked)
    ``form=MASK_FORM``: "cells" is the per-cell compare + select; "bits" maps the terms onto ONE q band [lo, hi) per lane
    -- causal lo = kv_abs - diag, SWA hi = kv_abs - diag + W + 1, padded hi = q_col_base (all masked) -- and reuses the
    library's ``band_mask_words`` / ``apply_mask_words`` (keep-word + R2P + 1 FSEL per cell).  Same masked set, same zero,
    so the two forms are bitwise identical.  MASK_NONE returns reg_P unchanged (no IR).
    TODO(plan s13 PR-4): hoist this transposed arm into tile_dsl.mask as the kv-major twin of apply_mask_chunk_bits."""
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return reg_P
    if cutlass.const_expr(MASK_FORM == MASK_FORM_BITS):
        lo = None
        hi = None
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
            lo = kv_abs - causal_diag
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
            hi = kv_abs - causal_diag + cutlass.Int32(CFG.SWA_WINDOW + 1)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
            row_dead = kv_abs >= seqlen_kv
            hi_pad = cutlass.Int32(arith.select(row_dead.ir_value(), q_col_base.ir_value(), (q_col_base + cutlass.Int32(N)).ir_value()))
            hi = hi_pad if hi is None else cute.math.min(hi, hi_pad)
        words = band_mask_words(lo, hi, q_col_base, N)
        return apply_mask_words(reg_P, words, mask_value=0.0, n_cols=N)
    zero = cutlass.Float32(0.0)
    elems = []
    for i in range(N):
        q_abs = q_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
            t = kv_abs >= seqlen_kv
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
            t = kv_abs > (q_abs + causal_diag)
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
            t = kv_abs < (q_abs + causal_diag - cutlass.Int32(CFG.SWA_WINDOW))
            masked = t if masked is None else (masked | t)
        elems.append(cutlass.Float32(arith.select(masked.ir_value(), zero.ir_value(), reg_P[i].ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


# === Tile decode (one cga2 cluster per (kv block, head, batch) tile) ===================================================


@cute.jit
def _decode_linear_bprop(linear, n_qh_grid, n_batch):
    """Flat 1-D (LPT / LPT_L2) decode: linear cluster id -> (kv_super, head, batch).  kv_super is the OUTER axis so the
    heaviest causal kv blocks (small kv_super, attended by the most queries) land in the first SM wave.  ``n_qh_grid`` is
    the GRID head extent (== qh_chunk under head-chunking)."""
    hb = n_qh_grid * n_batch
    kv_super = linear // hb
    within = linear % hb
    head = within % n_qh_grid
    batch = within // n_qh_grid
    return kv_super, head, batch


@cute.jit
def _boot_tile(sched):
    """The FIRST tile (kv_super, head, batch) from the launch blockIdx: natural 3-D grid (bidx // CGA_M, bidy, bidz); LPT /
    LPT_L2 flat grid: bidx is the linear cluster base and bidy_init / bidz_init are REPURPOSED to carry (n_qh_grid,
    n_batch) -- dead under an (N, 1, 1) grid (see _kernel)."""
    linear = sched.bidx_init // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv = linear
        h = sched.bidy_init
        b = sched.bidz_init
    else:
        kv, h, b = _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)
    return cute.arch.make_warp_uniform(kv), cute.arch.make_warp_uniform(h), cute.arch.make_warp_uniform(b)


@cute.jit
def _decode_tile_payload(sched, sched_idx):
    """Decode (kv_super, head, batch, is_valid) from the try_cancel response slot (ONE 128-bit load, ``read_clc_payload``).
    Natural grid: word 0 = kv_super * CGA_M, word 1 = head (low 16) | batch (high 16).  Flat grid: word 0 = linear * CGA_M,
    decoded with the (n_qh_grid, n_batch) stashed in bidy_init / bidz_init."""
    t0, t1, valid = read_clc_payload(sched, sched_idx * cutlass.Int32(8))
    t0 = cute.arch.make_warp_uniform(t0)
    t1 = cute.arch.make_warp_uniform(t1)
    valid = cute.arch.make_warp_uniform(valid)
    linear = t0 // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv = linear
        h = t1 & cutlass.Int32(0xFFFF)
        b = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
    else:
        kv, h, b = _decode_linear_bprop(linear, sched.bidy_init, sched.bidz_init)
    return kv, h, b, valid


# === Kernel entry ========================================================================================================


@cute.kernel
def _kernel(
    # TMA descriptors -- loads
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q   (BMM1 S B, q-split box)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  (BMM1 dP B, q-split box)
    tma_do_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  (BMM2 dV B, d_v-split box, BT)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    # TMA descriptors -- stores
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dV -> [B, S_kv, H_q, d_v] OUT
    tma_ds_desc: cutlass.GridConstant[tmap.TensorMap],  # dS -> workspace [B, H_chunk, S_kv, S_q] bf16
    # GMEM vectors
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32, natural log
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 delta = rowsum(dO * O), true units, unscaled
    # FP8 contract: 1-element fp32 device scales (never host-folded) + the two amax outputs (None-specialized off).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    descale_do_t: cute.Tensor,
    descale_s_t: cute.Tensor,
    scale_s_t: cute.Tensor,
    scale_dv_t: cute.Tensor,
    amax_dv_tensor: Optional[cute.Tensor],
    amax_dp_tensor: Optional[cute.Tensor],
    # Scalars
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,  # the REAL kv length (drives the PADDED mask and the amax row gate)
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,  # workspace chunking: full-tensor head = grid head + head_base
    n_qh_grid: cutlass.Int32,  # grid head extent (== qh_chunk); the flat-grid decode
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- device-scale fold (every thread loads the same 1-element scales: identical addresses -> L2 broadcast) -------
    _dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
    _dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
    _dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
    _dsc_do = cutlass.Float32(cutlass.make_array_view(descale_do_t)[0])
    _dsc_s = cutlass.Float32(cutlass.make_array_view(descale_s_t)[0])
    _scl_s = cutlass.Float32(cutlass.make_array_view(scale_s_t)[0])
    _scl_dv = cutlass.Float32(cutlass.make_array_view(scale_dv_t)[0])
    # S_acc -> P * scale_s in ONE exp2: exp2(S_acc * s_scale_log2 - (lse * log2e - log2(scale_s))).
    s_scale_log2 = attn_scale_log2e * _dsc_q * _dsc_k
    lse_log2_shift = cute.math.log2(_scl_s)
    # dS = (dP_acc * dp_scale - delta * dot_scale) * P_s  ==  attn_scale * P * (dP_acc * descale_v * descale_do - delta),
    # because P_s = P * scale_s and both terms carry descale_s.
    dp_scale = attn_scale * _dsc_v * _dsc_do * _dsc_s
    dot_scale = attn_scale * _dsc_s
    # dV_true = dV_acc * descale_s * descale_dO (P_s and dO_q both dequantize); the e4m3 output then applies scale_dV.
    dv_scale = _dsc_s * _dsc_do
    dv_out_scale = _scl_dv

    # --- SharedStorage: DECLARATION ORDER == config_sm107.smem_layout (the desc-root tally describes this layout) ------
    # Q ring (3 stages): B operand of BMM1 S (q-split, 64 q x 256 d_qk per CTA).
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #1 (3 stages): B operand of BMM1 dP (BT=false, q-split, 64 q x 256 d_v; leading 0, 2 swizzle subtiles).
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dO ring #2 (3 stages): B operand of BMM2 dV (BT=true, d_v-split, 128 q x 128 d_v; leading TILE_N x swz, 1 subtile).
    # The SAME dO GMEM data with a DIFFERENT per-CTA box + swizzle layout: it cannot share sdO_raw (the s128b XOR maps
    # cells to different bytes for the 256-B-row and the 128-B-row interpretations).
    sdOdv_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO_DV * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # K + V backing: sK [0, kBufferElems) | sV [kBufferElems, +vBufferElems) during the q loop; the dV epilogue staging
    # ALIASES it post-loop (K, V dead).  Byte-sized for max(K + V, dV @ BPE_O) -- equal at bf16 out, dV smaller at e4m3.
    _KV_ALIAS_ELEMS = max((kBufferElems + vBufferElems) * CFG.BPE, dVBufferElems * CFG.BPE_O)
    sExcl_raw = cutlass.Array(STORAGE_DTYPE, _KV_ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(sExcl_raw.data_ptr(), shape=kBufferElems, dtype=STORAGE_DTYPE)
    # ELEMENT offset (subview is element-addressed; a `* BPE` here is a no-op at BPE = 1 and doubles at BPE = 2).
    sV_raw = cutlass.Array(sExcl_raw.subview(kBufferElems).data_ptr(), shape=vBufferElems, dtype=STORAGE_DTYPE)
    sdV_raw = cutlass.Array(sExcl_raw.data_ptr(), shape=dVBufferElems, dtype=OUT_STORAGE_DTYPE)
    # lse / delta prefetch ring (fp32, ~2 KiB).  K + V are live through the whole q loop, so it has its own backing.
    sStats_raw = cutlass.Array(cutlass.Float32, CFG.STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # dS SMEM ring (bf16, XFER_STAGES deep): compute lanes store_swizzled dS[kv, q], the TMASTG TMA-stores each slot to
    # the GMEM workspace.  LOCAL to each CTA (each CTA owns its 128 kv rows of the pair's block).
    sdS_raw = cutlass.Array(DS_STORAGE_DTYPE, CFG.XFER_STAGES * dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    # --- SmemTile wrappers (every one takes desc_version=DESC_VERSION) --------------------------------------------------
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.STAGES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_Q,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sdO = SmemTile(
        base=sdO_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sdO_dv = SmemTile(
        base=sdOdv_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO_DV,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO_SG1,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO_SG1,
        layout=SMEM_LAYOUT_dO,
        tma_loads_per_tile=TMA_VO_SG1_ITERS,
        tma_granu_elems=TMA_VO_SG1_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_SG1_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_K,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # dS staging: TMA-store source only (no MMA reads it -> leading / stride are irrelevant, kept at the dS constants).
    # A 128-col bf16 row is P_TMA_ITERS = 2 subtiles of P_D_BLOCK = 64 cols; subtile s of stage k starts at
    # k * dSBufferElems + s * P_BLOCK_ELEMS.
    sdS = SmemTile(
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_dS,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dS,
        layout=SMEM_LAYOUT_dS,
        tma_loads_per_tile=P_TMA_ITERS,
        tma_granu_elems=P_D_BLOCK,
        tma_subtile_stride_elems=P_BLOCK_ELEMS,
        desc_version=DESC_VERSION,
    )
    # dV staging: TMA_DV_ITERS subtiles of (TILE_M kv x DV_D_BLOCK d_v) under the 128-B swizzle; aliases K + V.
    sdV = SmemTile(
        base=sdV_raw,
        elems_per_stage=dVBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_dV,
        tma_loads_per_tile=TMA_DV_ITERS,
        tma_granu_elems=DV_D_BLOCK,
        tma_subtile_stride_elems=DV_BLOCK_SLAB,
        desc_version=DESC_VERSION,
    )

    bars = _make_bars(CFG)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            # NATURAL: bidy / bidz = blockIdx.{y, z} (head, batch).  LPT / LPT_L2 flat 1-D grid: blockIdx.{y, z} = 0 (dead)
            # -> REPURPOSE these slots to carry (n_qh_grid, n_batch) for the linear decode (no extra Sched fields).
            "bidy_init": (n_qh_grid if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidy),
            "bidz_init": (n_batch if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidz),
        }
    )

    # --- cluster role identity (one cga2 pair) ----------------------------------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    partner_cta_id = cta_id_x ^ cutlass.Int32(1)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # --- mbarrier init: ONE warp, ONE lane (P4); every stage of every ring; then fence -> CTA sync -> cluster sync ------
    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(CFG.STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_dO):
                bars.mb_do_full[s].init()
                bars.mb_do_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_dO_DV):
                bars.mb_dodv_full[s].init()
                bars.mb_dodv_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[s].init()
                bars.mb_v_full[s].init()
                bars.mb_k_empty[s].init()
                bars.mb_v_empty[s].init()
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_S):
                bars.mb_s_acc_full[p].init()
                bars.mb_dp_full[p].init()
                bars.mb_s_acc_empty[p].init()
                bars.mb_dp_empty[p].init()
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_P):
                bars.mb_p_ready[p].init()
            bars.mb_dv_ready.init()
            bars.mb_dv_acc_empty.init()
            bars.mb_dv_stg_full.init()
            bars.mb_dv_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            for p in cutlass.range_constexpr(CFG.STATS_STAGES):
                bars.mb_stats_full[p].init()
                bars.mb_stats_empty[p].init()
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_ds_smem_full[p].init()
                bars.mb_ds_smem_empty[p].init()
            # Scheduler rings: init on EVERY CTA (the try_cancel multicast targets all of them).
            for s in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    # P4 order: init -> fence -> within-CTA sync -> cluster sync (BEFORE any cross-CTA arrive).  No bootstrap arrives: every
    # first wait on a consumer-side ring is pre-armed by PipelineState.start(phase=1) (P5b).
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    # Pair-scoped multicast mask for the MMA commits (both CTAs of the pair); TMA multicast = this CTA only.
    mcast_mask = cutlass.Int32(3) << leader_cta_id
    tma_mcast_mask = cutlass.Int16(1) << cta_in_pair
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # --- warp dispatch: 0..7 compute (2 wg) ; 8 MMA ; 9 TMALDG ; 10 TMASTG ; 11 scheduler + stats ------------------------
    if warp_idx < cutlass.Int32(_COMPUTE_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            warp_idx=warp_idx,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            sdS_raw=sdS_raw,
            sStats_raw=sStats_raw,
            sdV_raw=sdV_raw,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            s_scale_log2=s_scale_log2,
            dp_scale=dp_scale,
            dv_scale=dv_scale,
            dv_out_scale=dv_out_scale,
            amax_dv_tensor=amax_dv_tensor,
            amax_dp_tensor=amax_dp_tensor,
            cta_in_pair=cta_in_pair,
            leader_cta_id=leader_cta_id,
            partner_cta_id=partner_cta_id,
        )

    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp(
                sQ=sQ,
                sdO=sdO,
                sdO_dv=sdO_dv,
                sK=sK,
                sV=sV,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                mcast_mask=mcast_mask,
            )
        else:
            _mma_warp_quiet(tmem_ptr_i32, bars)

    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp(
            tma_q_desc=tma_q_desc,
            tma_do_desc=tma_do_desc,
            tma_do_dv_desc=tma_do_dv_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sQ=sQ,
            sdO=sdO,
            sdO_dv=sdO_dv,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            qh_per_kh=qh_per_kh,
            head_base=head_base,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_ds_desc.get_ptr())
        _tmastg_warp(
            tma_dv_desc=tma_dv_desc,
            tma_ds_desc=tma_ds_desc,
            sdV=sdV,
            sdS=sdS,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            cta_in_pair=cta_in_pair,
            head_base=head_base,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _scheduler_stats_warp(
            sched=sched,
            is_cga_first_cta=is_cga_first_cta,
            bars=bars,
            sStats_raw=sStats_raw,
            lse_tensor=lse_tensor,
            do_dot_tensor=do_dot_tensor,
            lse_log2_shift=lse_log2_shift,
            dot_scale=dot_scale,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            head_base=head_base,
        )


# === Compute warps: softmax + dsoftmax per q iteration, dV epilogue per kv tile ========================================


@cute.jit
def _softmax_warp_group(
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sdS_raw,
    sStats_raw,
    sdV_raw,
    seqlen_q,
    seqlen_kv,
    s_scale_log2,
    dp_scale,
    dv_scale,
    dv_out_scale,
    amax_dv_tensor: Optional[cute.Tensor],
    amax_dp_tensor: Optional[cute.Tensor],
    cta_in_pair,
    leader_cta_id,
    partner_cta_id,
) -> None:
    """8 compute warps (2 wg x 4), lane = kv row; wg0 -> q[0:64], wg1 -> q[64:128] (q_half_off).  Per kv tile, per q iter:
      softmax : wait mb_s_acc_full; tmem_load S[q half]; tcgen05_wait(LOAD); arrive mb_s_acc_empty (frees S for Q.K[i+1]);
                P_s = exp2(S * s_scale_log2 - lse_s); mask; e4m3 -> TMEM P ring slot; tcgen05_wait(STORE); arrive mb_p_ready
                (EARLY, before dsoftmax, so the dV BMM2 overlaps the whole dsoftmax below).
      dsoftmax: wait mb_dp_full; tmem_load dP[q half]; tcgen05_wait(LOAD); arrive mb_dp_empty (frees dP for dO.V[i+1]);
                dS = (dP * dp_scale - delta_s) * P_s; amax_dP fold; bf16 -> sdS ring slot (this wg's 64-col subtile);
                fence_proxy; arrive mb_ds_smem_full; arrive mb_stats_empty.
    Post q loop (per kv tile): wait mb_dv_ready; per 64-col chunk of this wg's d_v half: tmem_load dV; tcgen05_wait(LOAD);
    dV_true = acc * dv_scale; amax_dV fold; (* scale_dV ->) OUT dtype -> sdV (store_swizzled); fence_proxy; arrive
    mb_dv_stg_full; arrive mb_dv_acc_empty (frees dV TMEM for the next tile's P.dO[0], accumulate=False); the two amax
    atomics, gated on the lane's kv row being real."""
    # Pair with the MMA warp's barrier_cta_arrive: the TMEM base is published after its tmem_alloc.
    nvvm.barrier_cta_sync(barrier_id=_NAMED_BAR_TMEM_ID, thread_count=_NAMED_BAR_TMEM_THREADS)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)

    # tid_in_wg: 0..127 = this lane's kv row within its warpgroup; wg_id (0 / 1) -> q half.
    tid_in_wg = cute.arch.thread_idx()[0] & cutlass.Int32(127)
    wg_id = (warp_idx - cutlass.Int32(CFG.SOFTMAX_WG0_BASE)) // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    q_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)  # 0 or 64 q cols
    p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)  # fp8 P TMEM col offset: 0 or 16
    ds_wg_off = wg_id * cutlass.Int32(P_BLOCK_ELEMS)  # this wg's dS store subtile inside a ring slot
    dv_col_base = wg_id * cutlass.Int32(CFG.TILE_O // CFG.SOFTMAX_WARPGROUPS)  # this wg's d_v half

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()  # consume mb_s_acc_full
    dp_full_state = PipelineState.start()  # consume mb_dp_full
    ds_empty_state = PipelineState.start(phase=1)  # dS ring slot free (pre-armed)
    p_ready_state = PipelineState.start()  # produce mb_p_ready (2-stage P ring)
    dv_ready_state = PipelineState.start()  # consume mb_dv_ready
    stats_full_state = PipelineState.start()  # consume mb_stats_full

    # Bottom-right causal diagonal (kv <= q + (S_kv - S_q)); 0 for top-left / dense.
    causal_diag = (seqlen_kv - seqlen_q) if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT) else cutlass.Int32(0)
    # Per-lane absolute kv row base: this CTA's M slice of the pair's kv block.
    kv_lane_base0 = cta_in_pair * cutlass.Int32(CFG.TILE_M) + tid_in_wg

    # amax outputs: int32-bit-pattern atomicMax over non-negative fp32 (the forward's Amax_O idiom); None-specialized away
    # under has_amax=False.  The locals are opaque zeros: they feed fmax_f32's inline_ptx (a folded constant ICEs libNVVM).
    _amax_dv_ptr = Pointer(amax_dv_tensor.iterator.raw_ptr(), dtype=cutlass.Int32) if cutlass.const_expr(amax_dv_tensor is not None) else None
    _amax_dp_ptr = Pointer(amax_dp_tensor.iterator.raw_ptr(), dtype=cutlass.Int32) if cutlass.const_expr(amax_dp_tensor is not None) else None

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        tmem_base = tmem_ptr_i32.load()  # TMEM col base (published by the MMA warp's alloc)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)
        kv_abs = kv_block_base + kv_lane_base0
        # Row validity for the amax atomics (sdpa-invariants S5): a kv row past the REAL length is padding.  Its P is 0
        # under MASK_PADDED, so the fold below would add 0 anyway; the gate keeps an un-masked pad row out regardless.
        row_valid = kv_abs < seqlen_kv
        _amax_dp_tile = opaque_f32_zero()
        _amax_dv_tile = opaque_f32_zero()

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ---- stats ring: this q tile's lse_s / delta_s (lane = kv row reads the SAME q cols -> broadcast LDS) ----
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase, spin=SPIN_RING_WAITS)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS) + q_half_off

            # ---- 1) softmax: S[q half] -> P_s (registers) ----
            bars.mb_s_acc_full[s_full_state.idx].wait(s_full_state.phase, spin=SPIN_RING_WAITS)
            s_full_state = advance(s_full_state, CFG.STAGES_TMEM_S)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            # The LOAD-side wait ORDERS the arrive after the last LDTM (frost-kernels S3): the arrive has no data dependency
            # on the loaded registers, and without it ptxas may schedule it between two LDTMs -> a parked Q.K[i+1] overwrites
            # S under the pending second read.
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # Free S for Q.K[i+1] (the lookahead gate) -- every compute lane of both CTAs arrives on the LEADER (512).
            bars.mb_s_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            lse_elems = []
            for j in cutlass.range_constexpr(_SMX_CHUNK // 4):
                lse_elems.extend(sStats_raw.load(stats_base + cutlass.Int32(STATS_LSE_OFF + 4 * j), vector_size=4, alignment=16).to_elements())
            lse_vec = cutlass.Vector.from_elements(tuple(lse_elems), cutlass.Float32)
            # P_s = P * scale_s (the scale folded into the exponent shift by the stats warp).
            chunk_P = cute.math.exp2(reg_S.vec * s_scale_log2 - lse_vec, fastmath=True)
            # Transposed mask: zero P on masked (kv = lane, q = col) cells -> the fp8 P (dV) AND dS inherit it.  No IR at NONE.
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                chunk_P = _mask_p_chunk(chunk_P, kv_abs, q_iter * cutlass.Int32(CFG.TILE_N) + q_half_off, seqlen_kv, causal_diag, _SMX_CHUNK)

            # ---- 2) e4m3 P -> TMEM ring slot -> notify the MMA (EARLY, before dsoftmax, so the dV BMM2 overlaps the dP
            #         load + dS compute + dS store below).  Disjoint cols per wg + the all-lane mb_p_ready gate the MMA. ----
            p_slot = p_ready_state.idx
            chunk_P_fp8 = chunk_P.to(STORAGE_DTYPE)
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_slot * cutlass.Int32(LAYOUT.P_COLS) + p_col_off, cutlass.Float32),
                chunk_P_fp8,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bars.mb_p_ready[p_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)

            # ---- 3) dsoftmax: dP -> free the dP slot -> dS = (dP * dp_scale - delta_s) * P_s -> bf16 -> sdS ring ----
            bars.mb_dp_full[dp_full_state.idx].wait(dp_full_state.phase, spin=SPIN_RING_WAITS)
            dp_full_state = advance(dp_full_state, CFG.STAGES_TMEM_S)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dP read into registers -> the slot is free for dO.V[i+1] (every lane arrives on the leader).
            bars.mb_dp_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            dot_elems = []
            for j in cutlass.range_constexpr(_SMX_CHUNK // 4):
                dot_elems.extend(sStats_raw.load(stats_base + cutlass.Int32(STATS_DOT_OFF + 4 * j), vector_size=4, alignment=16).to_elements())
            dot_vec = cutlass.Vector.from_elements(tuple(dot_elems), cutlass.Float32)
            chunk_dS = (reg_dP.vec * dp_scale - dot_vec) * chunk_P
            if cutlass.const_expr(amax_dp_tensor is not None):
                # amax_dP = max |dS| of the fp32 value the workspace holds (attn_scale folded, pre-bf16-cast); masked cells
                # are exact zeros.  Ternary abs-max tree on max.f32 (FMNMX3), never cute.math.max (compare + select).
                _amax_dp_tile = fmax_f32(_amax_dp_tile, abs_max_tree([chunk_dS[i] for i in range(_SMX_CHUNK)]))
            chunk_dS_bf16 = chunk_dS.to(DS_STORAGE_DTYPE)
            ds_slot = ds_empty_state.idx
            bars.mb_ds_smem_empty[ds_slot].wait(ds_empty_state.phase, spin=SPIN_RING_WAITS)
            ds_empty_state = advance(ds_empty_state, CFG.XFER_STAGES)
            # dS SMEM [kv, q] as two 128-B-row subtiles per slot: this wg's subtile, row = this lane's kv row (64 bf16 =
            # 128 B under Swizzle(3, 4, 3) -- job 2 bank spread, job 1 the s128b store descriptor).
            (sdS_raw.subview(ds_slot * cutlass.Int32(dSBufferElems) + ds_wg_off + tid_in_wg * cutlass.Int32(P_D_BLOCK))).data_ptr().store_swizzled(
                chunk_dS_bf16, alignment=128, swizzle=STAGING_SMEM_SWIZZLE
            )
            # Generic SMEM writes -> async-proxy TMA store: the real proxy fence (rules/frost-tile-dsl.md S1).
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ds_smem_full[ds_slot].arrive()
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, CFG.STATS_STAGES)

        # ---- dV epilogue (per kv tile): dV TMEM -> dV_true -> OUT dtype -> sdV SMEM (aliases K + V, dead by now) ----
        # sdV is TMA_DV_ITERS subtiles of (TILE_M kv x DV_D_BLOCK d_v) under the 128-B swizzle; this wg owns
        # TILE_O / SOFTMAX_WARPGROUPS d_v cols, walked in 64-col chunks: chunk cols [gcol, gcol + 64) land at
        # (gcol // DV_D_BLOCK) * DV_BLOCK_SLAB + row * DV_D_BLOCK + (gcol % DV_D_BLOCK), with the swizzle XOR.
        bars.mb_dv_ready.wait(dv_ready_state.phase)
        dv_ready_state = advance(dv_ready_state, 1)
        for _c in cutlass.range_constexpr(_DV_CHUNKS_PER_WG):
            gcol = dv_col_base + cutlass.Int32(_c * _DV_EPI_CHUNK)
            reg_dV = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dV_OFF) + gcol, num_elems=_DV_EPI_CHUNK)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            # dV_acc = sum P_s_q . dO_q  ->  dV_true = dV_acc * descale_s * descale_dO.
            dv_true = reg_dV.vec * dv_scale
            if cutlass.const_expr(amax_dv_tensor is not None):
                _amax_dv_tile = fmax_f32(_amax_dv_tile, abs_max_tree([dv_true[i] for i in range(_DV_EPI_CHUNK)]))
            if cutlass.const_expr(OUT_IS_FP8):
                dv_out = (dv_true * dv_out_scale).to(OUT_STORAGE_DTYPE)  # dV_q = e4m3(dV_true * scale_dV)
            else:
                dv_out = dv_true.to(OUT_STORAGE_DTYPE)  # pre-quantization bf16 / fp16 output (the A/B path)
            dv_blk = gcol // cutlass.Int32(DV_D_BLOCK)
            dv_col_in_blk = gcol - dv_blk * cutlass.Int32(DV_D_BLOCK)
            (sdV_raw.subview(dv_blk * cutlass.Int32(DV_BLOCK_SLAB) + tid_in_wg * cutlass.Int32(DV_D_BLOCK) + dv_col_in_blk)).data_ptr().store_swizzled(
                dv_out, alignment=_DV_EPI_CHUNK * CFG.BPE_O, swizzle=STAGING_SMEM_SWIZZLE
            )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dv_stg_full.arrive()
        # Free the dV TMEM for the next kv tile's P.dO[0] (accumulate=False) overwrite.
        bars.mb_dv_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        if cutlass.const_expr(amax_dv_tensor is not None):
            if row_valid:
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_dv_ptr, _amax_dv_tile.bitcast(cutlass.Int32))
        if cutlass.const_expr(amax_dp_tensor is not None):
            if row_valid:
                nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_dp_ptr, _amax_dp_tile.bitcast(cutlass.Int32))

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # Release the MMA warps' TMEM allocation on BOTH CTAs (256 local + 256 from the partner's compute lanes = 512 each).
    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


# === MMA warp (pair leader) + the follower's quiet MMA warp ==============================================================


@cute.jit
def _mma_warp(sQ, sdO, sdO_dv, sK, sV, tmem_ptr_i32, bars, sched, seqlen_q, seqlen_kv, mcast_mask) -> None:
    """MMA leader: the 3-matmul lookahead stream.  Per kv tile (K, V one-shot) the issue order is FIXED for perf (Q.K[i+1]
    ahead of P.dO[i] overlaps the dV BMM2 with the softmax and hides the SMEM-latency stall):

        Q.K[q_lo]
        for i in q_lo .. q_hi-2:  dO.V[i] ; Q.K[i+1] ; P.dO[i]
        dO.V[q_hi-1] ; P.dO[q_hi-1]

    Handshakes (single-buffer S / dP; the fp8 P ring is its own 2-stage TMEM region):
      mb_s_acc_empty  gates Q.K[i+1]  (softmax loaded S[i])       pre-armed (PipelineState.start(phase=1))
      mb_s_acc_full   Q.K  -> softmax "S[i] ready"
      mb_dp_full      dO.V -> softmax "dP[i] ready"
      mb_dp_empty     gates dO.V[i+1] (dsoftmax loaded dP[i])     pre-armed
      mb_p_ready[2]   softmax -> P.dO[i] "P[i] in ring slot i % 2"
    P-slot reuse is gated by the 2-stage ring + the in-order MMA stream (s_acc_full[i+2] fires after P.dO[i] read slot
    i % 2) -- no p_empty.  At exit the two pre-armed LEADER-scope rings hold one completed, un-waited phase each (256 of
    its 512 arrives cross-CTA); they are drained here so the follower's last relaxed cluster arrive lands on a resident CTA
    (P15, port fix F3)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=TMEM_IS_EXCLUSIVE)
    # Publish the TMEM base: pair with the compute warps' barrier_cta_sync at their top of body.
    nvvm.barrier_cta_arrive(_NAMED_BAR_TMEM_ID, _NAMED_BAR_TMEM_THREADS)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()  # dO ring #1 (BMM1 dP)
    dodv_full_state = PipelineState.start()  # dO ring #2 (BMM2 dV)
    # Consumer handshakes (single buffer): pre-armed so the prologue Q.K[q_lo] / the first dO.V pass with no softmax load
    # yet.  p_ready is a real consumer wait.
    s_acc_empty_state = PipelineState.start(phase=1)
    dp_empty_state = PipelineState.start(phase=1)
    p_ready_state = PipelineState.start(phase=0)
    dv_empty_state = PipelineState.start(phase=0)  # epilogue drained dV_acc

    kv_super_idx, _, _ = _boot_tile(sched)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_OFF))
    tmem_P = tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))  # fp8 P 2-stage ring (the dV mma_ts A operand)

    # ---- descriptors (Rubin dense FP8: K = 64 per instruction, idesc k_dim = 1) ----
    # BMM1 S = K . Q^T   (A = K, M = kv; B = Q, N = q; K = d_qk).
    idesc_bmm1_s = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=CFG.IDESC_K_DIM
    )
    bmm1_s_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_s,
        kind=MMA_KIND,
    )
    # BMM1 dP = V . dO^T  (A = V, M = kv; B = dO, N = q; K = d_v).
    idesc_bmm1_dp = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_N, m_dim=CFG.TILE_M * CFG.CTA_MMA, k_dim=CFG.IDESC_K_DIM
    )
    bmm1_dp_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_O,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_dp,
        kind=MMA_KIND,
    )
    # BMM2 dV = P . dO  (A = fp8 P in TMEM, M = kv, K = q; B = dO_dv, N = d_v, BT).
    idesc_bmm2_dv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        a_major=0,
        b_major=1,
        k_dim=CFG.IDESC_K_DIM,
    )
    bmm2_dv_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm2_dv,
        kind=MMA_KIND,
    )

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # ONE elect per work item for every predicated commit of this tile (K3): the warp is converged here and the
        # predicated arrives below never diverge it (each is a branch round the native op that reconverges).
        elect_p = nvvm.elect_sync()

        # The q range that attends this kv block (uniform across the pair); the prologue handles q_lo, the loop runs
        # [q_lo, q_hi); the dV accumulate restarts at q_lo, NOT 0 (rules/frost-tile-dsl.md S2).
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        # K + V: one-shot per kv tile.
        bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase)
        bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase)
        desc_K = sK[k_full_state.idx].desc()
        desc_V = sV[v_full_state.idx].desc()

        # ---- prologue: Q.K[q_lo] -> S_acc ----
        bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
        s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
        bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
        mma_ss(bmm1_s_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S)
        bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        q_full_state = advance(q_full_state, CFG.STAGES_Q)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ----- dO.V[i] -> dP (BMM1 dP): gated on the dP slot being free (dsoftmax[i-1] read it); pre-armed for i = q_lo -----
            bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase, spin=SPIN_RING_WAITS)
            dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)
            bars.mb_do_full[do_full_state.idx].wait(do_full_state.phase, spin=SPIN_RING_WAITS)
            mma_ss(bmm1_dp_desc, desc_V, sdO[do_full_state.idx].desc(), tmem_dP)
            bars.mb_dp_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_do_empty[do_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            do_full_state = advance(do_full_state, CFG.STAGES_dO)

            # ----- Q.K[i+1] -> S_acc (lookahead; not on the last iteration) -----
            if (q_iter + cutlass.Int32(1)) < q_hi:
                bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
                bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
                mma_ss(bmm1_s_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S)
                bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ----- P.dO[i] -> dV += P[i] . dO_dv[i] (BMM2 dV, mma_ts; A = the P ring slot the softmax filled EARLY) -----
            p_slot = p_ready_state.idx
            bars.mb_p_ready[p_slot].wait(p_ready_state.phase, spin=SPIN_RING_WAITS)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)
            bars.mb_dodv_full[dodv_full_state.idx].wait(dodv_full_state.phase, spin=SPIN_RING_WAITS)
            mma_ts(
                bmm2_dv_desc,
                tmem_P.subview(p_slot * cutlass.Int32(LAYOUT.P_COLS)),
                sdO_dv[dodv_full_state.idx].desc(),
                tmem_dV,
                accumulate=(q_iter > q_lo),
            )
            bars.mb_dodv_empty[dodv_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dodv_full_state = advance(dodv_full_state, CFG.STAGES_dO_DV)

        # dV accumulation complete for this kv tile -> epilogue (compute warps).
        bars.mb_dv_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        # The epilogue drained dV TMEM before the next tile's P.dO[0] (accumulate=False) overwrites it -- gate here.
        bars.mb_dv_acc_empty[dv_empty_state.idx].wait(dv_empty_state.phase)
        dv_empty_state = advance(dv_empty_state, 1)

        # End of tile: release K + V (the commit orders after every Q.K / dO.V of the tile that read them).
        bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        k_full_state = advance(k_full_state, CFG.STAGES_KV)
        v_full_state = advance(v_full_state, CFG.STAGES_KV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, _, _, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15 (F3): the two pre-armed LEADER-scope rings each hold ONE completed phase this warp never waited (the protocol runs
    # one phase ahead); drain it so the follower's relaxed cluster arrives have landed before either CTA can exit.
    bars.mb_s_acc_empty[s_acc_empty_state.idx].wait(s_acc_empty_state.phase)
    bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase)

    # ---- TMEM dealloc (after every compute lane of both CTAs has arrived) ----
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    """The follower CTA's MMA warp: collective tmem_alloc (the leader's collective MMAs write this CTA's TMEM half), publish
    the TMEM base to this CTA's compute warps, then wait mb_tmem_dealloc and release TMEM.  No persistent loop, no
    scheduler credit (READ_TILE_ARRIVERS_TOT counts it out)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=TMEM_IS_EXCLUSIVE)
    nvvm.barrier_cta_arrive(_NAMED_BAR_TMEM_ID, _NAMED_BAR_TMEM_THREADS)
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# === TMA-store warp ====================================================================================================


@cute.jit
def _tmastg_warp(tma_dv_desc, tma_ds_desc, sdV, sdS, bars, sched, seqlen_q, seqlen_kv, cta_in_pair, head_base) -> None:
    """Per q iteration: store the dS ring slot -> GMEM workspace [B, H_chunk, S_kv, S_q] (mb_ds_smem_full wait -> TMA store
    -> commit -> wait(0) -> mb_ds_smem_empty).  Per kv tile: store dV -> GMEM [B, S_kv, H_q, d_v] (mb_dv_stg_full wait ->
    TMA store -> commit -> wait(0) -> mb_dv_stg_empty, which the TMALDG waits before it reloads K over the same SMEM).
    Both stores are per-CTA (this CTA's 128 kv rows of the pair's block).  The lse / delta prefetch lives on the scheduler
    warp so it runs concurrently with these stores."""
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_ds = GmemTileTma(tma_ds_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    ds_full_state = PipelineState.start()  # consume mb_ds_smem_full
    dv_full_state = PipelineState.start()  # consume mb_dv_stg_full
    KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        # Only the in-range q tiles produce dS; the skipped (out-of-band) q tiles' workspace regions stay ZERO (the adapter
        # zero-initialises the workspace under a mask) so dK / dQ = dS.Q / dS^T.K are correct.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            ds_slot = ds_full_state.idx
            bars.mb_ds_smem_full[ds_slot].wait(ds_full_state.phase, spin=SPIN_RING_WAITS)
            ds_full_state = advance(ds_full_state, CFG.XFER_STAGES)
            # Workspace [B, H_chunk, S_kv, S_q] -> coords innermost-first (S_q, S_kv, H, B); box (1, 1, TILE_M, P_D_BLOCK),
            # walked over the P_TMA_ITERS subtiles by tma_store_tile.  dS stays CHUNK-local (grid head, no head_base).
            tma_store_tile(sdS[ds_slot], tma_ds(q_col_base, kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_idx))
            tma_store_commit()
            tma_store_wait(0)
            if nvvm.elect_sync():
                bars.mb_ds_smem_empty[ds_slot].arrive()

        # dV for THIS kv tile (after the epilogue) -> the FULL output [B, S_kv, H_q, d_v]: full-tensor head = head + head_base.
        bars.mb_dv_stg_full.wait(dv_full_state.phase)
        dv_full_state = advance(dv_full_state, 1)
        tma_store_tile(sdV[0], tma_dv(cutlass.Int32(0), head_idx + head_base, kv_block_base + KV_ROW_OFFSET_PEER, batch_idx))
        tma_store_commit()
        # wait_group.read: the SMEM source has been READ (not: the GMEM write has landed) -- exactly what the alias needs.
        tma_store_wait(0)
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# === TMA-load warp =====================================================================================================


@cute.jit
def _tmaldg_warp(
    tma_q_desc,
    tma_do_desc,
    tma_do_dv_desc,
    tma_k_desc,
    tma_v_desc,
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
    head_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
) -> None:
    """Per kv tile: K, V one-shot (M-split kv, full d).  Per q iteration: Q (BMM1 S B: TILE_N / CTA_MMA q rows x full d_qk),
    dO (BMM1 dP B: same N-split) and dO_dv (BMM2 dV B: full TILE_N q rows x TILE_O / CTA_MMA d_v cols, BT -- a SECOND load
    of the same dO GMEM into its own ring with the dV-box descriptor).  Only the pair LEADER arms expect_tx (the
    cga2 tensor TMA routes both CTAs' bytes to its mbar, P9); both CTAs issue their own loads.

    Port fix F1: the dV epilogue staging ALIASES K + V.  The MMA's K/V release (mb_k_empty / mb_v_empty) fires when the
    epilogue has finished READING dV from TMEM (mb_dv_acc_empty), not when the TMASTG has finished reading the staged dV
    from SMEM -- so this warp additionally waits its CTA's mb_dv_stg_empty (armed after the dV store's wait_group.read)
    before the K load, or tile t+1's K could land under tile t's dV store.  Pre-armed so tile 0 passes.  At exit every
    cross-CTA multicast _empty ring is drained here (its consumer), the K/V rings included (F2)."""
    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_do_dv = GmemTileTma(tma_do_dv_desc)
    tma_v = GmemTileTma(tma_v_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    # Q / K / V / dO are the FULL [B, H, ...] tensors: index by the full-tensor head (grid head + head_base).
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)
    Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_M_PER_CTA)
    DO_DV_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)  # dO_dv: per-CTA d_v offset, full q rows

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    dodv_empty_state = PipelineState.start(phase=1)
    dv_stg_empty_state = PipelineState.start(phase=1)  # F1: this CTA's dV staging has left SMEM

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)

        # ---- K + V: one-shot per kv tile (after the previous tile's dV has left the aliased SMEM, F1) ----
        bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase)
        dv_stg_empty_state = advance(dv_stg_empty_state, 1)

        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
        bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sK[k_empty_state.idx],
            tma_k(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_k_full[k_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
        bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sV[v_empty_state.idx],
            tma_v(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, batch_idx),
            bars.mb_v_full[v_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

        # ---- Q + dO + dO_dv per q iteration (the mask-bounded range) ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_N)

            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_q_full[q_empty_state.idx].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sQ[q_empty_state.idx],
                tma_q(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_q_full[q_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)

            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_do_full[do_empty_state.idx].arrive(n_bytes=dOTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO[do_empty_state.idx],
                tma_do(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, batch_idx),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)

            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_dodv_full[dodv_empty_state.idx].arrive(n_bytes=dOTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO_dv[dodv_empty_state.idx],
                tma_do_dv(DO_DV_OFFSET_PEER, full_head, q_row_base, batch_idx),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        kv_super_idx, head_idx, batch_idx, is_valid_tile = _decode_tile_payload(sched, sched_state.idx)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: drain every cross-CTA multicast _empty ring OUTSIDE the persistent loop, on its consumer (this warp), so the
    # leader's last commits land on a resident CTA.  Q / dO / dO_dv (3 deep each) and -- port fix F2 -- K / V (1 deep).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _qs in cutlass.range_constexpr(CFG.STAGES_Q):
            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase)
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)
        for _ds in cutlass.range_constexpr(CFG.STAGES_dO):
            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase)
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)
        for _ds in cutlass.range_constexpr(CFG.STAGES_dO_DV):
            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase)
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
            k_empty_state = advance(k_empty_state, CFG.STAGES_KV)
            bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
            v_empty_state = advance(v_empty_state, CFG.STAGES_KV)
    # The LOCAL dV-staging ring: the last tile's arrive is unconsumed (its wait guards the NEXT tile's K load).  Harmless,
    # but a symmetric 1-deep drain keeps an init-count imbalance a localizable hang.
    bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase)


# === Scheduler warp FUSED with the lse / delta prefetch =================================================================


@cute.jit
def _scheduler_stats_warp(
    sched, is_cga_first_cta, bars, sStats_raw, lse_tensor, do_dot_tensor, lse_log2_shift, dot_scale, seqlen_q, seqlen_kv, head_base
) -> None:
    """Persistent tile scheduler (try_cancel protocol, the shape of ``tile_dsl.scheduler.scheduler_warp_loop``) fused with
    the lse / delta stats prefetch.  The compute lanes (lane = kv row) all read the SAME TILE_N q values of lse and delta
    per q tile; straight from GMEM that was 128x redundant and the dominant long-scoreboard stall, so this otherwise idle
    warp prefetches them into a STATS_STAGES-deep SMEM ring (LDG -> STS -> arrive) and the compute lanes read SMEM.

    Per loop iteration: (A) prefetch the CURRENTLY processed tile's rows -- one ring slot per q iteration, tile context
    tracked one behind the scheduling (bootstrap = blockIdx, then the decoded payload); the folds ``lse * log2e -
    log2(scale_s)`` and ``delta * attn_scale * descale_s`` happen here (the host passes RAW natural-log lse and RAW delta).
    (B) the try_cancel protocol for the NEXT tile: the cga-first CTA's elected lane arms expect_tx(16) on EVERY CTA's
    mb_scheduler (program-ordered before its multicast try_cancel; CTA scope suffices -- a cluster-scope release is a
    GPU drain), then every CTA's warp waits its own response.  This warp does NOT credit mb_read_tile_id."""
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    _PER_LANE = CFG.TILE_N // 32  # 4 q cols per lane per q tile

    # Context of the tile the consumers are CURRENTLY processing (one behind the scheduling).
    cur_kv_super, cur_head, cur_batch = _boot_tile(sched)

    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        # lse / delta are the FULL [B, H_q, S_q] tensors: index by the full-tensor head.
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        # Prefetch ONLY the q tiles the consumers process, so the ring count matches their [q_lo, q_hi).
        kv_block_base = cur_kv_super * cutlass.Int32(_KV_BLOCK_ROWS)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_kv)
        # ---- (A) stats prefetch for the CURRENT tile ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase, spin=SPIN_RING_WAITS)
            stats_empty_state = advance(stats_empty_state, CFG.STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                lse_s = lse_tensor[cur_batch, cur_full_head, q_col_base + col] * cutlass.Float32(_LOG2E) - lse_log2_shift
                dot_s = do_dot_tensor[cur_batch, cur_full_head, q_col_base + col] * dot_scale
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_LSE_OFF) + col).store(lse_s)
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_DOT_OFF) + col).store(dot_s)
            # Generic STS -> generic LDS on the consumer: the mbarrier orders it; the proxy fence here is harmless (carried).
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        # ---- (B) try_cancel protocol for the NEXT tile ----
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync() and is_cga_first_cta:
            for i in cutlass.range_constexpr(CGA_SIZE):
                if cutlass.const_expr(i == 0):
                    arrive_expect_tx(sched.mb_scheduler.subview(state.idx), _CLC_RESPONSE_BYTES)
                else:
                    peer_mb = nvvm.mapa(sched.mb_scheduler.subview(state.idx), cutlass.Int32(i))
                    nvvm.mbarrier_arrive_expect_tx(peer_mb, _CLC_RESPONSE_BYTES, scope=nvvm.MemScope.CTA)
            nvvm.clusterlaunchcontrol_try_cancel(sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)), sched.mb_scheduler.subview(state.idx), multicast=1)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        # The NEXT tile's context becomes "current" for the next loop.
        cur_kv_super, cur_head, cur_batch, is_valid = _decode_tile_payload(sched, state.idx)
        state = advance(state, CFG.SCHEDULER_STAGES)


# === Host launcher =========================================================================================================


@cute.jit
def _host(
    q_tensor: cute.Tensor,  # [B, S_q, H_q, d_qk]   e4m3
    do_tensor: cute.Tensor,  # [B, S_q, H_q, d_v]    e4m3
    k_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_qk] e4m3
    v_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_v]  e4m3
    dv_tensor: cute.Tensor,  # out [B, S_kv, H_q, d_v] OUT dtype (per Q-head partial)
    ds_tensor: cute.Tensor,  # out [B, H_chunk, S_kv, S_q] bf16 workspace (the dK / dQ GEMM A operand)
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 natural-log LSE
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32 delta, true units
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    descale_do_t: cute.Tensor,
    descale_s_t: cute.Tensor,
    scale_s_t: cute.Tensor,
    scale_dv_t: cute.Tensor,
    amax_dv_tensor: Optional[cute.Tensor],
    amax_dp_tensor: Optional[cute.Tensor],
    problem_size: Tuple[int, int, int, int, int, int],  # (B, QH, KH, SQ, SKV, QH_CHUNK)
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    seqlen_kv_real: cutlass.Int32,  # REAL kv length (the PADDED mask + amax gate); == SKV when dense
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK = problem_size

    # TMA boxes ([B, S, H, D] layout: box = (1 batch, S rows, 1 head, D cols); the workspace is [B, H, S_kv, S_q]).
    qk_box_q = (1, _M_PER_CTA, 1, TMA_QK_GRANU_ELEMS)  # Q   (BMM1 S B, q-split)
    qk_box_k = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)  # K   (M-split kv)
    do_box = (1, _M_PER_CTA, 1, TMA_VO_GRANU_ELEMS)  # dO  (BMM1 dP B, q-split)
    v_box = (1, CFG.TILE_M, 1, TMA_VO_GRANU_ELEMS)  # V   (M-split kv)
    do_dv_box = (1, CFG.TILE_N, 1, CFG.TILE_O // CFG.CTA_MMA)  # dO  (BMM2 dV B, BT): full TILE_N q x TILE_O / CTA_MMA d_v
    dv_box = (1, CFG.TILE_M, 1, DV_D_BLOCK)  # dV store subtile: TILE_M kv x DV_D_BLOCK d_v = 128 B rows
    ds_box = (1, 1, CFG.TILE_M, P_D_BLOCK)  # dS store subtile: TILE_M kv x P_D_BLOCK q bf16 = 128 B rows
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor, box_dims=qk_box_q, stride_order=stride_order, swizzle=_tma_swz(CFG.Q_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=do_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_do_dv_desc = tmap.create_tensor_map_tiled_from_view(
        do_tensor, box_dims=do_dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dO_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor, box_dims=qk_box_k, stride_order=stride_order, swizzle=_tma_swz(CFG.K_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor, box_dims=v_box, stride_order=stride_order, swizzle=_tma_swz(CFG.V_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    # dV / dS stores: the compute lanes store_swizzled with the 128-B pattern, so both descriptors decode s128b (one unit).
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(
        dv_tensor, box_dims=dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dV_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_tensor, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )

    # Grid: one cga2 cluster per (kv block, head, batch) tile; the head axis spans QH_CHUNK heads per launch.  NATURAL =
    # 3-D grid; LPT / LPT_L2 = flat 1-D (kv_super OUTER: the heaviest causal kv blocks first).  Both ride the same persistent
    # try_cancel scheduler -- the policy only sets the launch shape and the per-tile decode.
    kv_blocks = (SKV + _KV_BLOCK_ROWS - 1) // _KV_BLOCK_ROWS
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        grid_shape = (kv_blocks * CFG.CGA_M, QH_CHUNK, B)
    else:
        grid_shape = (kv_blocks * QH_CHUNK * B * CFG.CGA_M, 1, 1)

    _kernel(
        tma_q_desc,
        tma_do_desc,
        tma_do_dv_desc,
        tma_k_desc,
        tma_v_desc,
        tma_dv_desc,
        tma_ds_desc,
        lse_tensor,
        do_dot_tensor,
        descale_q_t,
        descale_k_t,
        descale_v_t,
        descale_do_t,
        descale_s_t,
        scale_s_t,
        scale_dv_t,
        amax_dv_tensor,
        amax_dp_tensor,
        cutlass.Int32(SQ),
        # kernel seqlen_kv = the REAL length (drives the PADDED mask); grid / kv_blocks / descriptors stay on the padded SKV.
        seqlen_kv_real,
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        cutlass.Int32(QH_CHUNK),
    ).launch(
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
    skv: int = 256,
    qh_chunk: int = 0,
    has_amax: bool = True,
) -> Callable:
    """Compile the kernel with ALL dims concrete (pins the TMA descriptor strides); see the module docstring's Launch ABI.

    ``qh`` is the FULL Q-head count (the Q / K / V / dO / dV / lse / delta descriptors span it); ``qh_chunk`` (0 = qh) is the
    per-launch head extent: the grid head axis and the dS workspace head dim.  ``has_amax=False`` None-specializes the two
    amax outputs and folds their |.| trees and atomics out (the bf16 A/B path pays nothing for them)."""
    _cache_key = _template_key(globals(), locals(), "compile")
    if qh_chunk == 0:
        qh_chunk = qh
    if sq % CFG.TILE_N or skv % _KV_BLOCK_ROWS or sq <= 0 or skv <= 0:
        raise ValueError(f"{__name__}: sq must be a multiple of {CFG.TILE_N} and skv of {_KV_BLOCK_ROWS} (the adapter pads); got sq={sq}, skv={skv}")
    if kh <= 0 or qh % kh:
        raise ValueError(f"{__name__}: qh ({qh}) must be a positive multiple of kh ({kh})")
    if qh_chunk <= 0 or qh % qh_chunk:
        raise ValueError(f"{__name__}: qh_chunk ({qh_chunk}) must be a positive divisor of qh ({qh})")

    def _fake_bshd(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), stride_order=(0,), assumed_align=4)

    fake_q = _fake_bshd((b, sq, qh, CFG.TILE_K), STORAGE_DTYPE)
    fake_do = _fake_bshd((b, sq, qh, CFG.TILE_O), STORAGE_DTYPE)
    fake_k = _fake_bshd((b, skv, kh, CFG.TILE_K), STORAGE_DTYPE)
    fake_v = _fake_bshd((b, skv, kh, CFG.TILE_O), STORAGE_DTYPE)
    fake_dv = _fake_bshd((b, skv, qh, CFG.TILE_O), OUT_STORAGE_DTYPE)
    # dS workspace [B, H_chunk, S_kv, S_q] bf16: the kv-major layout the dK = dS.Q GEMM reads un-permuted.
    fake_ds = _fake_bshd((b, qh_chunk, skv, sq), DS_STORAGE_DTYPE)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_dot = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_amax_dv = _fake_scale() if has_amax else None
    fake_amax_dp = _fake_scale() if has_amax else None

    return _compile_cached(
        _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dv,
        fake_ds,
        fake_lse,
        fake_dot,
        _fake_scale(),  # descale_q
        _fake_scale(),  # descale_k
        _fake_scale(),  # descale_v
        _fake_scale(),  # descale_do
        _fake_scale(),  # descale_s
        _fake_scale(),  # scale_s
        _fake_scale(),  # scale_dv
        fake_amax_dv,
        fake_amax_dp,
        (b, qh, kh, sq, skv, qh_chunk),
        cutlass.Float32(0.0),  # attn_scale
        cutlass.Float32(0.0),  # attn_scale_log2e
        cutlass.Int32(0),  # head_base
        cutlass.Int32(skv),  # seqlen_kv_real (default = the allocated SKV; the adapter passes the real length under padding)
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_bwd_d256_fp8",
    )


def _main():
    """Minimal CLI for a standalone compile check."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=1)
    parser.add_argument("--hk", type=int, default=1)
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=256)
    args = parser.parse_args()
    print(f"[bprop_d256_fp8] compile b={args.b} qh={args.hq} kh={args.hk} sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[bprop_d256_fp8] compile OK: {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
