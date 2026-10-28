# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SDPA backward main kernel, d_qk = d_v = 256, bf16/fp16, on the **2x2 tcgen05 datapath** (``cta_group::2``,
collective M = 128, 64 kv rows per CTA per sub-block).  SM100 (cc 10.0-10.6) and the Rubin line share this ONE body;
``config_d256_2x2.CfgBwdD256x2.DATAPATH_2X2_PROFILE`` (from ``TemplateParams.datapath_2x2_profile``) folds every
arm at trace time.

One cga2 pair (2 CTAs, 12 warps each) owns a ``KV_BLOCK_ROWS``-row kv block of ONE (batch, head) -- 128 rows on
profile 1 (one 64-row sub-block per CTA), 256 rows on profile 2 (two sub-blocks per CTA, warpgroup g owns sub-block
g) -- and walks the q tiles that attend the 256-row kv WRITE PAIR the block belongs to.  It computes **dV in TMEM**
and writes **dS to a GMEM workspace** ``[B, H, S_kv, S_q]``; the adapter then runs dK = dS . Q and dQ = dS^T . K as the
``bprop_matmul_blackwell`` GEMMs at the (256, 256) cluster tile over that workspace.  lane = kv row within the 64-row
sub-block tile, the lane HALF = the q (or d_v) column half (the 2x2 accumulator atom).

Pipeline (per kv block; profile 1 uses NATURAL for D256 and lookahead for native D192x128 -- ``MMA_LOOKAHEAD`` from
``CFG.MMA_LOOKAHEAD``. The
lookahead arm (``S(q_lo)`` prologue, then ``dP(i); S(i+1); BMM2(i)``, the fp8 twin's form) is profile 2's default and
the A/B arm for profile-1 D256: B200 2026-10-01 stage 2 4525 us lookahead vs 3781 us NATURAL dense; ``[s]`` = per sub-block)::

    MMA (leader CTA):  per q_iter i:  S(i)[s]; dP(i)[s]; BMM2(i)[s]        (lookahead: S(q_lo)[s]; dP(i); S(i+1); BMM2(i))
        BMM1 S  = K . Q[i]^T      -> S  TMEM slot (64 cols)  mma_ss(sK[s] 64 x 256 SW128, sQ N-split 64 q rows)
        BMM1 dP = V . dO[i]^T     -> dP TMEM slot (64 cols)  mma_ss(sV[s], sdO)
        BMM2 dV += P[i] . dO[i]   -> dV TMEM (128 cols)      mma_ss(sP[s][pslot] 64 kv x 128 q bf16, sdO_dv BT 128 q x 128 d_v)
    8 compute warps (lane = kv row of its quadrant, lane half = q column half), per q_iter:
        softmax  P = exp2(S * scale * log2e - lse[q] * log2e), masked -> bf16 -> sP[s][pslot] (store_swizzled, 128-B
                 swizzled K-major rows) -> fence_proxy -> mb_p_full (LEADER_RELEASE)
        dSoftmax dS = (dP * scale - do_dot[q]) * P -> sdS ring -> TMA-STG -> workspace
    per kv block (post q-loop): dV TMEM -> half -> sdV (aliases sK) -> TMA-STG -> dV [B, S_kv, H_q, d_v]

Every MMA operand is an SMEM SS operand (fact (c) of the design: a 2x2 TS A operand is lane-duplicated and saves no
TMEM; the quadrant warps cannot write the duplicated image).  No UTCCP, no K-split, no TMEM P alias, no DSMEM data
ship: the only cross-CTA traffic is the cga2 TMA routing to the leader's mbar, the tcgen05.commit multicasts and the
mbarrier arrives.  Every slot reuse is an EXPLICIT barrier (``mb_s_acc_empty``, ``mb_dp_empty``, ``mb_p_empty``),
never an issue-order invariant (Rule S3 class).

Warp roles (per CTA, 12 warps; ids from ``CFG``)::

    0..7   compute   -- qd = w & 3 (lane quadrant: kv row 32*(qd&1)+t, q half qd>>1), g = w >> 2 (warpgroup:
                        sub-block g // SUBBLOCK_WGS, column sub-chunk g % SUBBLOCK_WGS of COLS_PER_LANE q cols)
    8      MMA       -- leader CTA: the 3-MMA stream and every commit; follower: quiet (alloc / dealloc only)
    9      TMA-LDG   -- K, V once per kv block (KV_SUBBLOCKS x 4 boxes each); Q, dO, dO_dv per q tile
    10     TMA-STG   -- dS slot -> workspace per q tile; dV -> GMEM per kv block
    11     scheduler -- CLC try_cancel protocol FUSED with the lse / do_dot SMEM prefetch

    Register split per profile (config_d256_2x2): 8 x 176 + 4 x 152 on profile 1 (the service warps carry the headroom
    because ptxas hoists the 1-stage B operands' k-step descriptors into the MMA warp), 8 x 224 + 4 x 56 on profile 2 (64 q
    columns per compute lane: 91 / 129 STL / LDL at 176 on sm_107a, 0 / 0 at 224); both = 2016 = 12 warps x the 168-register
    ENTRY count (config_sm107.reg_entry_pool).

TMEM (512 columns, one ``tcgen05.alloc.cta_group::2`` per CTA, NOT exclusive; the 2x2 D atom: row m, column n of a
64 x N fp32 tile -> lane m + 64 * (n // (N/2)), column n % (N/2)); sub-block s at ``s * SUBBLOCK_STRIDE_COLS``::

    profile 1:  [0, 64) S slot 0 | [64, 128) S slot 1 | [128, 192) dP slot 0 | [192, 256) dP slot 1 | [256, 384) dV | [384, 512) free
    profile 2:  per sub-block [0, 64) S | [64, 128) dP | [128, 256) dV, x 2 (= 512)

SMEM (per CTA, declaration order == ``config_d256_2x2.smem_layout_2x2``; every slab 1024-B aligned, 128-B swizzled)::

    slab        profile 1 (210 KiB, SM100 227 KiB cap)           profile 2 (322 KiB, Rubin 327 KiB cap)
    sQ          1 x 32 KiB  [64 q x 256]  B of BMM1 S            2 x 32 KiB
    sdO         1 x 32 KiB  [64 q x 256]  B of BMM1 dP           1 x 32 KiB
    sdOdv       1 x 32 KiB  [128 q x 128 d_v] BT B of BMM2       1 x 32 KiB
    sK          1 x 32 KiB  [64 kv x 256] A of BMM1 S            2 sub-blocks x 32 KiB   (+ the post-loop sdV alias)
    sV          1 x 32 KiB  A of BMM1 dP                          2 x 32 KiB
    sP          2 x 16 KiB  [64 kv x 128 q] bf16, A of BMM2      2 sub-blocks x 1 x 16 KiB  (root 262144 -> DESC_VERSION 1)
    sStats      2 KiB       lse*log2e | do_dot*scale per slot     2 KiB
    sdS         1 x 16 KiB  [64 kv x 128 q] io dtype -> TMA-STG   1 x 32 KiB [128 kv x 128 q]

BARRIER TABLE (lane ledger; L = leader CTA, F = follower; x N = per q tile, x T = per kv block; rings with a
sub-block axis are indexed ``[s * STAGES + idx]``; ``L_CNT`` = the sub-block's compute lanes x 2 CTAs = 512 (profile 1)
/ 256 (profile 2); ``SOFT_X_CTA_MMA`` = every compute lane x 2 CTAs = 512)::

    mbar[stages]                 producer / scope    init L / F        arrive site (guard)                              issuing lanes      waiter(s)                    phase  fires
    mb_q_full[STAGES_Q]          TMA_LOAD            1 / 1(-)          TMALDG expect_tx, pred=is_leader & elect         1, L only          MMA(L)                       st(0)  x N
    mb_q_empty[STAGES_Q]         MMA_COMMIT          1 / 1             MMA(L) after the LAST sub-block's S, pred=elect  1 x mcast(2 CTAs)  TMALDG L+F; DRAIN x STAGES_Q st(1)  x N
    mb_do_full / empty[1]        as q                                  (empty: after the last sub-block's dP)                                                                   x N
    mb_dodv_full / empty[1]      as q                                  (empty: after the last sub-block's BMM2)                                                                 x N
    mb_k_full, mb_v_full[1]      TMA_LOAD            1 / 1(-)          TMALDG expect_tx (KV_SUBBLOCKS x 64 KiB)         1, L only          MMA(L)                       st(0)  x T
    mb_k_empty, mb_v_empty[1]    MMA_COMMIT          1 / 1             MMA(L) end of block, pred=elect                  1 x mcast          TMALDG L+F; DRAIN x1         st(1)  x T
    mb_s_full[KV_SUB x S_ST]     MMA_COMMIT          1 / 1             MMA(L) after S[s], pred=elect                    1 x mcast          compute of s, L+F            st(0)  x N
    mb_s_acc_empty[same]         LEADER (relaxed)    L_CNT / L_CNT(-)  compute of s, BARE, after the S LOAD wait        L_CNT/2 x 2 -> L   MMA(L) pre-armed; DRAIN x S_ST st(1) x N
    mb_dp_full[same]             MMA_COMMIT          1 / 1             MMA(L) after dP[s], pred=elect                   1 x mcast          compute of s, L+F            st(0)  x N
    mb_dp_empty[same]            LEADER (relaxed)    L_CNT / L_CNT(-)  compute of s, BARE, after the dP LOAD wait       L_CNT/2 x 2 -> L   MMA(L) pre-armed; DRAIN x S_ST st(1) x N
    mb_p_full[KV_SUB x P_ST]     LEADER_RELEASE      L_CNT / L_CNT(-)  compute of s, BARE, after store_swizzled + fence L_CNT/2 x 2 -> L   MMA(L)                       st(0)  x N
    mb_p_empty[same]             MMA_COMMIT          1 / 1             MMA(L) after BMM2[s], pred=elect                 1 x mcast          compute of s pre-armed; DRAIN x P_ST st(1) x N
    mb_stats_full[2]             THREAD              32 / 32           scheduler warp, BARE                             32 (local)         compute (256 lanes)          st(0)  x N
    mb_stats_empty[2]            THREAD              256 / 256         compute, BARE, 8 warps                           256 (local)        scheduler warp               st(1)  x N
    mb_ds_smem_full[1]           THREAD              256 / 256         compute, BARE, 8 warps                           256 (local)        TMASTG                       st(0)  x N
    mb_ds_smem_empty[1]          THREAD              1 / 1             TMASTG, `if elect_sync():`                       1 (local)          compute                      st(1)  x N
    mb_dv_ready[1]               MMA_COMMIT          1 / 1             MMA(L) after the last sub-block's last BMM2      1 x mcast          compute L+F (256)            st(0)  x T
    mb_dv_acc_empty[1]           LEADER (relaxed)    512 / 512(-)      compute, BARE, 8 warps x 2 CTAs after the dV LOAD wait  256 x 2 -> L  MMA(L)                   st(0)  x T
    mb_dv_stg_full[1]            THREAD              256 / 256         compute, BARE, 8 warps                           256 (local)        TMASTG                       st(0)  x T
    mb_dv_stg_empty[1]           THREAD              1 / 1             TMASTG, `if elect_sync():`                       1 (local)          TMALDG (pre-armed); DRAIN x1 st(1)  x T
    mb_tmem_dealloc[1]           THREAD (+peer)      512 / 512         compute BARE local + BARE arrive_on_peer         256 + 256          MMA (each CTA)               0      x 1
    sched.mb_scheduler[2]        expect_tx 16 B      1 / 1             scheduler, CTA 0 elect arms EVERY CTA            1                  every persistent warp        st(0)  x T
    sched.mb_read_tile_id[2]     read_tile_id_arrive 21 / 21           8 compute + TMALDG + TMASTG on both CTAs + MMA(L) = 21 calls, 1 arrive per CTA per call

    SUM(issuing lanes) == init on every row and both CTAs; a "(-)" follower init is a copy that is never armed and
    never waited (cga2 tensor TMAs deliver both peers' bytes to the LEADER's mbar; LEADER-scope arrives all land on the
    leader).  P15 drains (cross-CTA ASYNC arrives whose last fire is in flight at exit) are OWNED by the consumer:
    TMALDG drains q/do/dodv/k/v_empty after its loop; the leader MMA drains the two pre-armed LEADER rings
    (s_acc_empty, dp_empty: STAGES_TMEM_S waits each, the un-waited batches of the last tiles); every compute warp
    drains p_empty (STAGES_SMEM_P waits); the mb_tmem_dealloc wait itself is the drain of the peer arrives.
    ``mb_p_full`` is the first ``Producer.LEADER_RELEASE`` consumer in the tree: the ``.release.cta`` arrive after a
    per-lane ``fence_proxy("async.shared", "cta")`` publishes the lane-written sP slab to the peer-issued MMA (probe
    release_p, 2026-10-01: 0 / 1920 mismatches, SASS ``MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S; SYNCS.ARRIVE``, 0 CGAERRBAR).

Deadlock check of the lookahead with the explicit rings: S(i+1) needs ``s_acc_empty`` from softmax(i-1)'s LOAD (needs
``s_full(i-1)``, committed earlier); BMM2(i) needs ``p_full(i)`` from softmax(i) (needs ``s_full(i)``, committed, and
``p_empty`` from BMM2(i-2), committed); dP(i) needs ``dp_empty`` from dSoftmax(i-2).  No cycle.

Masks are the TRANSPOSE of the forward (lane = kv row, inner loop = q tile): the kv WRITE PAIR bounds WHICH q tiles
attend (``compute_q_loop_bounds`` over ``kv_write_base = kv_block_base & ~255`` with 256 rows -- both 128-row blocks
of a pair walk the identical, pair-rounded range, the extra tiles fully masked by the per-cell mask on the TRUE
``kv_abs``), and the per-cell mask zeroes P on masked (kv, q) cells, so dV / dS / dK / dQ inherit it.  An empty q
range is FORCED to one fully-masked q tile.  The per-cell mask is the bit-word form (``band_mask_words`` /
``apply_mask_words``, one keep-word per 32 q columns).

## Launch ABI (identical to ``sm107/bprop_d256_f16.py`` so ``kernels/sm107/prepared_host.host_f16`` runs this body)

``compile(b, qh, kh, sq, skv, qh_chunk=0, b_chunk=0, sq_real=0, skv_real=0)``; ``sq % 128 == 0``, ``skv % 256 == 0``
(the kv write pair; the adapter pads).  Call, positionally::

    fn(q, do, k, v, dv, ds, lse, do_dot, seq_kv_lens, problem_size, attn_scale, head_base, batch_base, stream)

Raw templates default to ``fuse_dk=False`` and retain this frame. An explicitly
fused D192 template appends ``dk_tensor`` and, for THD, ``dk_maps`` (``(2*B+1)*16``
int64 words aligned to 128 bytes). Prepared graph and standalone plans select
this fusion internally and carve the maps from caller workspace.

Grid: NATURAL ``(ceil(SKV / KV_BLOCK_ROWS) * 2, QH_CHUNK, B_CHUNK)``, cluster (2, 1, 1); LPT / LPT_L2 flatten to
``(kv_blocks * QH_CHUNK * B_CHUNK * 2, 1, 1)`` (``config_d256_2x2.launch_grid_2x2``).
"""

from functools import lru_cache
from typing import Callable, NamedTuple, Optional

import cuda.bindings.driver as _cuda_driver  # noqa: F401
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass._mlir.dialects import arith
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap

from cudnn.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    Scope,
    advance,
    arrive_expect_tx,
    cga_arrive,
    cga_wait,
    wait,
)
from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mask import (
    MASK_CAUSAL,
    MASK_NONE,
    MASK_PADDED,
    MASK_SWA,
    apply_mask_words,
    band_mask_words,
    compute_q_loop_bounds,
)
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile
from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, Sched, read_clc_payload, read_tile_id_arrive
from cudnn.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_BWD_META_WORDS,
    THD_BWD_MAPS_META_WORDS,
    THD_BWD_MAPS_OFF,
    THD_BWD_WORK_PREFIX_OFF,
    THD_SETUP_THREADS,
    emit_clamped_desc,
    emit_seq_descs,
    thd_decode_unit_prefix,
    thd_port_origin,
    write_thd_work_prefix,
)
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc

# Config comes from the FROST template loader, never an env var: the loader injects FROST_TEMPLATE_PARAMS before this
# body runs; the default keeps a plain import usable (profile 1 = the SM100 body; profile 0 is the 4x1 bodies' and the
# 2x2 factory refuses it).
from cudnn.sdpa.bwd.config_d256_2x2 import (
    FAMILY_F16,
    PROFILE_SM100,
    TemplateParams,
    buffer_elems_2x2,
    desc_version_2x2,
    make_cfg_d256_2x2,
    q_write_tiles_2x2,
    tmem_layout_2x2,
    validate_head_chunk,
)

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams(datapath_2x2_profile=PROFILE_SM100))
CFG = make_cfg_d256_2x2(PARAMS, FAMILY_F16)
_B = buffer_elems_2x2(CFG)
LAYOUT = tmem_layout_2x2(CFG)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point, derived from THIS config's
# declaration-order slab table (profile 1: largest root sP[1] at 176 KiB -> 0; profile 2: sP at 256 KiB -> 1).  A
# version-0 descriptor's start_address is 14 bits = a 256 KiB window; an operand at or past it wraps to offset 0 and the
# MMA multiplies the bottom of SMEM -- an EXACTLY-zero accumulator, no crash.  Never re-literal it at a call site.
DESC_VERSION: int = desc_version_2x2(CFG)

# Retry form of the per-q-iteration / per-kv-block RING waits; the waits a warp parks in for a whole tile (scheduler
# payload, mb_tmem_dealloc) and the end-of-kernel drains keep the default sleeping form.  Unmeasured on this body -> the
# default (rules/frost-tile-dsl.md S8b).
SPIN_RING_WAITS: bool = False

# MMA issue order: True = lookahead (S(i+1) between dP(i) and BMM2(i): softmax(i) gets a dP(i) + S(i+1) window); False
# = NATURAL (S(i); dP(i); BMM2(i)), kept for A/B.  Bound from the config's validated default; a test may flip the module
# attribute before compile() (the SPIN_RING_WAITS idiom).  Both arms keep the explicit s_acc_empty / p_empty rings.
MMA_LOOKAHEAD: bool = bool(CFG.MMA_LOOKAHEAD)


_THD = bool(CFG.THD_VARLEN)
THD_Q_SLOT, THD_DO_SLOT, THD_DODV_SLOT, THD_K_SLOT, THD_V_SLOT = 0, 1, 2, 3, 4
THD_DV_SLOT_BASE = 5  # dV descriptor of sequence b at slot THD_DV_SLOT_BASE + b
THD_INPUT_SLOTS = 5
_THD_SEQ_ORD = 2  # (d, head, seq, batch): the sequence axis of every packed descriptor
_THD_MAP_WORDS = TENSOR_MAP_QWORDS * 2  # int32 words per 128-B tensor map


# --- dtype dispatch (tile_dsl DTYPE_* codes) -------------------------------------------
if CFG.DTYPE_QKV == DTYPE_BF16:
    STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_QKV == DTYPE_FP16:
    STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"{__name__}: DTYPE_QKV must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {CFG.DTYPE_QKV}")
MMA_KIND = nvvm.Tcgen05MMAKind.F16
OUT_STORAGE_DTYPE = cutlass.BFloat16 if CFG.DTYPE_O == DTYPE_BF16 else cutlass.Float16
DS_STORAGE_DTYPE = STORAGE_DTYPE  # the config pins DTYPE_DS == DTYPE_QKV

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2  # CFG.CTA_MMA == 2, pinned by the config
CGA_SIZE = _B.CGA_SIZE  # 2

# --- per-CTA geometry (all from the config, never re-derived) ------------------------------
KV_SUBBLOCKS = CFG.KV_SUBBLOCKS  # 64-row sub-blocks per CTA (1 / 2)
SUB_ROWS = CFG.TILE_M  # 64 kv rows per sub-block = the MMA's m_per_cta
ROWS_PER_CTA = CFG.ROWS_PER_CTA  # the contiguous kv rows one CTA owns (64 / 128)
SUBBLOCK_WGS = CFG.SUBBLOCK_WGS  # compute warpgroups per sub-block (2 / 1)
COLS_PER_LANE = CFG.COLS_PER_LANE  # q columns per compute lane (32 / 64)
_M_PER_CTA = _B._M_PER_CTA  # 64 q rows per CTA (Q / dO N-split across the pair)
qBufferElems = _B.qBufferElems
dOBufferElems = _B.dOBufferElems
dOdvBufferElems = _B.dOdvBufferElems
kSubElems = _B.kSubElems
kBufferElems = _B.kBufferElems
vSubElems = _B.vSubElems
vBufferElems = _B.vBufferElems
pSlabElems = _B.pSlabElems
dSBufferElems = _B.dSBufferElems
dVBufferElems = _B.dVBufferElems

qTmaTransactionBytes = _B.qTmaTransactionBytes
dOTmaTransactionBytes = _B.dOTmaTransactionBytes
dOdvTmaTransactionBytes = _B.dOdvTmaTransactionBytes
kTmaTransactionBytes = _B.kTmaTransactionBytes
vTmaTransactionBytes = _B.vTmaTransactionBytes

TMA_QK_ITERS = _B.TMA_QK_ITERS  # 4 subtiles of 64 elems (128 B) per 256-elem row
TMA_VO_ITERS = _B.TMA_VO_ITERS
TMA_QK_GRANU_ELEMS = _B.TMA_QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _B.TMA_VO_GRANU_ELEMS
TMA_VO_SG1_ITERS = _B.TMA_VO_SG1_ITERS  # dO_dv (BT view): 2 subtiles of 64 d_v cols per CTA
TMA_VO_SG1_GRANU_ELEMS = _B.TMA_VO_SG1_GRANU_ELEMS
DV_D_BLOCK = _B.DV_D_BLOCK  # 64 d_v cols per dV store subtile (128 B)
TMA_DV_ITERS = _B.TMA_DV_ITERS  # 4
DV_BLOCK_SLAB = _B.DV_BLOCK_SLAB  # ROWS_PER_CTA * DV_D_BLOCK elems per dV subtile slab
P_TMA_ITERS = _B.P_TMA_ITERS  # 2 dS store subtiles of 64 q cols (128 B)
P_D_BLOCK = _B.P_D_BLOCK  # 64
DS_BLOCK_SLAB = _B.DS_BLOCK_SLAB  # ROWS_PER_CTA * P_D_BLOCK elems per dS subtile slab
P_SUB_SLAB = _B.P_SUB_SLAB  # SUB_ROWS * 64 elems: one 64-q-column subtile (8 KiB) of a P slab
P_STORE_ALIGN = _B.P_STORE_ALIGN  # a lane's P / dS segment: 64 B (32 cols) or 128 B (64 cols)

LEADING_BYTE_OFFSET_QK = _B.LEADING_BYTE_OFFSET_QK
STRIDE_BYTE_OFFSET_QK = _B.STRIDE_BYTE_OFFSET_QK
# BT=true B operand (dO_dv): per-CTA N = TILE_O / CTA_MMA = 128 -> 128 // 8 > 8 -> leading = K x swizzle (16384); probe
# ss_slabs S4: LBO = 0 silently corrupts the second 64-d_v subtile.
LEADING_BYTE_OFFSET_dO_SG1 = _B.LEADING_BYTE_OFFSET_dO_SG1
STRIDE_BYTE_OFFSET_dO_SG1 = _B.STRIDE_BYTE_OFFSET_dO_SG1
LEADING_BYTE_OFFSET_P = _B.LEADING_BYTE_OFFSET_P
STRIDE_BYTE_OFFSET_P = _B.STRIDE_BYTE_OFFSET_P
SMEM_LAYOUT_SW128 = _B.SMEM_LAYOUT_SW128

# P / dS / dV lane stores: 128-B swizzled rows (Swizzle(3, 4, 3): MBase + SShift = 7 = log2(128 B)), the UMMA K-major
# SW128 canonical layout the sP descriptors read AND the s128b TMA-store descriptors read; the swizzle XOR is applied to
# the ABSOLUTE SMEM address, so a 64-B half-row store at a 64-B-aligned sub-row pointer inside a 1024-B-aligned slab
# lands in the same image (probe release_p M2b, verified).
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)

STATS_SLOT_ELEMS = _B.STATS_SLOT_ELEMS
STATS_LSE_OFF = _B.STATS_LSE_OFF
STATS_DOT_OFF = _B.STATS_DOT_OFF
_KV_BLOCK_ROWS = _B._KV_BLOCK_ROWS  # kv rows per cga2 pair (128 / 256): the scheduler's unit
_KV_WRITE_ROWS = _B._KV_WRITE_ROWS  # 256: the kv write pair = the stage-3 GEMMs' cluster M tile / K-trim granularity
_THD_BLOCKS_PER_WRITE = _KV_WRITE_ROWS // _KV_BLOCK_ROWS
_Q_WRITE_TILES = q_write_tiles_2x2(CFG)  # 2
_LOG2E = 1.4426950408889634
_CLC_RESPONSE_BYTES = 16

# --- named arrive counts (P3) -------------------------------------------------------------
ONE_LANE = CFG.ONE_LANE
ONE_WARP = CFG.ONE_WARP
SOFTMAX_LANES_ALL_WG = CFG.SOFTMAX_LANES  # 8 warps x 32 = every compute lane of ONE CTA
SOFT_X_CTA_MMA = CFG.SOFT_X_CTA_MMA  # every compute lane of BOTH CTAs -> the leader (dv_acc_empty, tmem_dealloc)
L_CNT = CFG.L_CNT  # the compute lanes of ONE sub-block on BOTH CTAs -> the leader (s_acc_empty / dp_empty / p_full)
MMA_COMMIT_ARRIVES = CFG.MMA_COMMIT_ARRIVES  # one predicated tcgen05.commit multicast = 1 arrive per target CTA
READ_TILE_ARRIVERS_TOT = CFG.READ_TILE_ARRIVERS_TOT  # 21, derivation in config_d256_2x2.read_tile_arrivers_tot_2x2
N_S_BARS = KV_SUBBLOCKS * CFG.STAGES_TMEM_S  # sub-block-indexed S / dP rings
N_P_BARS = KV_SUBBLOCKS * CFG.STAGES_SMEM_P  # sub-block-indexed P ring
# dV epilogue: x64 loads per lane per block (1 on profile 1 at +64*cs; 2 on profile 2 at +0 / +64).
DV_LOADS_PER_LANE = 2 // SUBBLOCK_WGS
DV_LOAD_COLS = 64


# ============================================================================
# Mask plumbing (the TRANSPOSE of the forward: lane = kv row, inner loop = q tile)
# ============================================================================


def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real):
    """Real kv length of this batch entry: ``seq_kv_lens[batch_idx + batch_base]`` under the PADDED arm, else the scalar."""
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
        arr = cutlass.make_array_view(seq_kv_lens_tensor)
        return cutlass.Int32(arr[batch_idx + batch_base])
    return seqlen_kv_real


def _thd_maps_off_words(n_batch):
    """int32 word offset of the tensor-map array inside the metadata buffer (``tile_dsl.thd.THD_BWD_MAPS_OFF`` on a runtime B)."""
    return (THD_BWD_META_WORDS(n_batch) * cutlass.Int32(4) + cutlass.Int32(127)) // cutlass.Int32(128) * cutlass.Int32(_THD_MAP_WORDS)


def _thd_map_ptr(meta_t, n_batch, slot):
    """Generic-space pointer to tensor map ``slot`` of the THD metadata buffer (``_THD`` only)."""
    return (meta_t.iterator.raw_ptr() + _thd_maps_off_words(n_batch) + slot * cutlass.Int32(_THD_MAP_WORDS)).tospace(cutlass.AddressSpace.generic)


def _dk_map_ptr(maps, slot):
    return (maps.iterator.raw_ptr() + slot * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)


@cute.jit
def _qdk_slice(desc, maps, n_batch, batch_idx, physical_origins: cutlass.Constexpr, *coords):
    if cutlass.const_expr(_THD):
        slot = cutlass.Int32(0)
        if cutlass.const_expr(physical_origins):
            slot = batch_idx + cutlass.Int32(1) if batch_idx < n_batch else cutlass.Int32(0)
        return tma_slice_runtime_desc(_dk_map_ptr(maps, slot), *coords)
    return GmemTileTma(desc)(*coords)


@cute.jit
def _input_slice(desc, meta_t, n_batch, slot: cutlass.Constexpr, batch_idx, physical_origins: cutlass.Constexpr, *coords):
    if cutlass.const_expr(_THD):
        map_slot = cutlass.Int32(slot)
        if cutlass.const_expr(physical_origins):
            map_slot = cutlass.Int32(THD_INPUT_SLOTS) + slot * n_batch + batch_idx if batch_idx < n_batch else cutlass.Int32(slot)
        return tma_slice_runtime_desc(_thd_map_ptr(meta_t, n_batch, map_slot), *coords)
    return GmemTileTma(desc)(*coords)


def _seq_ctx(meta_t, batch_idx, n_batch, seqlen_q_real, seqlen_kv_real, physical_origins: cutlass.Constexpr = False):
    """``(q_tok, kv_tok, ws_row, s_q, s_kv, q_load, kv_load)`` of the tile's sequence -- THD reads them from the metadata
    buffer, dense returns ``(0, 0, 0, SQ_REAL, SKV_REAL, 0, 0)`` and every use folds.  The per-sequence values ride WITH
    the tile decode (every warp refreshes them from the same ``(batch_idx)`` the decode returned), so a stale sequence
    after a tile change is impossible by construction -- that omission would not fault, it would address the previous
    sequence.  A DEAD unit (the occupancy-sized grid hands out ``uid >= live``) decodes ``batch_idx == n_batch``: every
    read stays IN BOUNDS of the buffer (``cu_q[B+1]`` is ``cu_k[0]``, ``cu_k[B+1]`` is ``batch_remap[0]``, ``row_off[B]``
    the blocked total) and yields ``s_q <= 0``. Its forced tile is fully masked and both dS/dV stores are suppressed;
    all pipeline arrivals still occur, so dead clusters drain the same rings without racing on a shared slack row.

    ``q_load`` / ``kv_load`` are the sequence offsets the TMA LOADS take (``_thd_tma_offsets``): ``cu_q[b]`` / ``cu_k[b]``
    for a unit with query rows, and for one WITHOUT (an ``s_q[b] = 0`` sequence, a dead unit) the clamped descriptors'
    own extent ``max(cu_q[B], 1)`` / ``max(cu_k[B], 1)`` -- OUT OF BOUNDS by construction, so TMA zero-fills every operand
    row.  The setup kernel cannot clamp an extent to 0 (invalid, traps), so an all-empty side keeps ONE addressable row,
    the packed buffer's row 0; a tile that addressed ``cu_q[b] = 0`` there would load that row, and a NaN in it (a
    poisoned capacity with no live token) would reach ``dV = P^T . dO`` as ``0 * NaN`` -- P is select-masked, the MMA
    is a multiply.  Routing the load past the extent makes the operand itself zero (sdpa-invariants: a dead range must
    not depend on what the residue holds).  The stats index keeps ``q_tok`` (its own select, in the scheduler warp)."""
    if cutlass.const_expr(_THD):
        meta = cutlass.make_array_view(meta_t)
        cu_q0 = n_batch
        cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
        row0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
        q_tok = cute.arch.make_warp_uniform(cutlass.Int32(meta[cu_q0 + batch_idx]))
        kv_tok = cute.arch.make_warp_uniform(cutlass.Int32(meta[cu_k0 + batch_idx]))
        s_q = cute.arch.make_warp_uniform(cutlass.Int32(meta[cu_q0 + batch_idx + cutlass.Int32(1)]) - q_tok)
        s_kv = cute.arch.make_warp_uniform(cutlass.Int32(meta[cu_k0 + batch_idx + cutlass.Int32(1)]) - kv_tok)
        ws_row = cute.arch.make_warp_uniform(cutlass.Int32(meta[row0 + batch_idx]))
        q_cap = cute.math.max(cutlass.Int32(meta[cu_q0 + n_batch]), cutlass.Int32(1))  # cu_q[B], the clamped Q / dO extent
        kv_cap = cute.math.max(cutlass.Int32(meta[cu_k0 + n_batch]), cutlass.Int32(1))  # cu_k[B], the clamped K / V extent
        has_q = s_q > cutlass.Int32(0)
        q_load = cute.arch.make_warp_uniform(cutlass.Int32(arith.select(has_q.ir_value(), q_tok.ir_value(), q_cap.ir_value())))
        kv_load = cute.arch.make_warp_uniform(cutlass.Int32(arith.select(has_q.ir_value(), kv_tok.ir_value(), kv_cap.ir_value())))
        if cutlass.const_expr(physical_origins):
            # Per-sequence maps start at each port's own base. Empty/dead work
            # uses an OOB coordinate even though an empty map has extent one.
            q_load = cute.arch.make_warp_uniform(
                cutlass.Int32(arith.select(has_q.ir_value(), cutlass.Int32(0).ir_value(), cute.math.max(s_q, cutlass.Int32(1)).ir_value()))
            )
            kv_load = cute.arch.make_warp_uniform(
                cutlass.Int32(arith.select((has_q & (s_kv > 0)).ir_value(), cutlass.Int32(0).ir_value(), cute.math.max(s_kv, cutlass.Int32(1)).ir_value()))
            )
        return q_tok, kv_tok, ws_row, s_q, s_kv, q_load, kv_load
    return cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), seqlen_q_real, seqlen_kv_real, cutlass.Int32(0), cutlass.Int32(0)


def _eff_lengths(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_q_real, seqlen_kv_real, ctx):
    """``(eff_seqlen_q, eff_seqlen_kv)`` every role masks and bounds with: the sequence's own lengths under THD (the
    decode's ``_seq_ctx``), the uniform real q length and the padded arm's per-batch kv length otherwise."""
    if cutlass.const_expr(_THD):
        return ctx[3], ctx[4]
    return seqlen_q_real, _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real)


def _thd_tma_offsets(ctx, full_batch):
    """``(q_seq_off, kv_seq_off, tma_batch)`` for EVERY packed-operand TMA coordinate (rules/frost-tile-dsl.md S5: the
    detector ``check_thd_load_coords`` requires a ``seq_off`` term at every Q / K / V / dO load): THD adds ``cu_q[b]`` /
    ``cu_k[b]`` to the sequence coordinate of a one-batch packed descriptor (the clamped extent itself -- out of bounds,
    zero-filled -- for a unit without query rows; ``_seq_ctx``), dense keeps the full-tensor batch."""
    if cutlass.const_expr(_THD):
        return ctx[5], ctx[6], cutlass.Int32(0)
    return cutlass.Int32(0), cutlass.Int32(0), full_batch


def _causal_diag(seqlen_q_real, eff_seqlen_kv):
    """Bottom-right diagonal offset (S_kv - S_q); 0 for top-left / dense."""
    if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT):
        return eff_seqlen_kv - seqlen_q_real
    return cutlass.Int32(0)


def _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv):
    """[q_lo, q_hi) -- the q-tile range that attends the 256-row kv WRITE PAIR this block belongs to, rounded OUTWARD
    to a q pair, FORCED non-empty.

    The bounds are computed from ``kv_write_base = kv_block_base & ~(_KV_WRITE_ROWS - 1)`` with ``_KV_WRITE_ROWS``
    rows, NOT from the block's own rows: on profile 1 the two 128-row blocks of a write pair then walk the IDENTICAL q
    range (the block's extra tiles are fully masked by the per-cell mask on the true ``kv_abs``: P = 0 -> dS = 0 stored,
    dV += 0), so the written set per 256-row stage-3 M tile equals the 4x1 body's and ``_causal_k_range``'s tight-trim
    invariant holds without a workspace zero-fill (the write-pair invariant; the poisoned-workspace tests pin it).  On
    profile 2 the block IS the write pair and this is a no-op.  Then the 4x1 body's pair rounding and the forced
    N >= 1 clamp (an empty range would leave the UNCONDITIONAL MMA prologue waiting on a mb_q_full nobody arms).
    Every warp derives the SAME bounds from the same (kv_block_base, lengths) -- the P14 balance of six loop bodies
    depends on it.
    """
    kv_write_base = kv_block_base & cutlass.Int32(~(_KV_WRITE_ROWS - 1) & 0xFFFFFFFF)
    if cutlass.const_expr(_THD):
        n_q_tiles = cute.math.max((seqlen_q_real + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N), cutlass.Int32(1))
    else:
        n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    b = compute_q_loop_bounds(
        kv_write_base,
        seqlen_q_real,
        eff_seqlen_kv,
        n_q_tiles,
        CFG.SWA_WINDOW,
        CFG.MASK_FLAGS,
        CFG.TILE_N,
        _KV_WRITE_ROWS,
        bottom_right=bool(CFG.CAUSAL_BOTTOM_RIGHT),
        window_right=0,
    )
    lo, hi = b.lo, b.hi
    if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
        pair = cutlass.Int32(_Q_WRITE_TILES)
        lo = (lo // pair) * pair
        hi = cute.math.min(((hi + pair - cutlass.Int32(1)) // pair) * pair, n_q_tiles)
    q_lo = cute.math.min(lo, n_q_tiles - cutlass.Int32(1))
    q_hi = cute.math.max(hi, q_lo + cutlass.Int32(1))
    return q_lo, q_hi


def _mask_p_chunk(reg_P, kv_abs, q_col_base, eff_seqlen_kv, causal_diag, N: int, eff_seqlen_q=None):
    """Zero P on masked (kv = lane, q = col) cells of an ``N``-wide q chunk starting at absolute q ``q_col_base``.

    The TRANSPOSE of ``tile_dsl.mask.apply_mask_chunk`` (row = kv, col = q), with ZERO as the masked value:
      causal : kv_abs >  q_abs + diag       (key past the query; diag = S_kv - S_q under bottom-right, else 0)
      SWA    : kv_abs <  q_abs + diag - W   (key left of the window, bottom-right-anchored like the forward)
      padded : kv_abs >= seq_kv_len         (per-lane pad row; the whole row goes to 0)
      THD    : q_abs  >= s_q[b]             (the q pad columns of the sequence's last tile are the NEXT sequence's tokens,
                                             or the clamped zeros past cu_q[B]; a dead unit's s_q <= 0 masks every column)
    The bit-word form (rules/frost-tile-dsl.md S10d): the terms map onto ONE q band [lo, hi) per lane -- causal
    lo = kv_abs - diag, SWA hi = kv_abs - diag + W + 1, padded hi = q_col_base (all masked), THD hi = s_q[b] -- and the
    library's ``band_mask_words`` / ``apply_mask_words`` do the rest (one keep-word per 32 q columns, R2P + 1 FSEL per cell).
    The masked set is the per-cell compare's, so dS / dV are bitwise what the pre-port body produced.  MASK_NONE
    returns reg_P unchanged (no IR).  P on a masked cell is a SELECT-zero, never a multiply: ``dS = 0`` and ``dV += 0 . dO``
    whatever the Stats row holds (a ``-inf`` LSE of a query with no key gives ``exp2(..) = +inf`` before the select).
    TODO(plan s13 PR-4): hoist this transposed arm into tile_dsl.mask as the kv-major twin of apply_mask_chunk.
    """
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return reg_P
    lo = None
    hi = None
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
        lo = kv_abs - causal_diag
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
        hi = kv_abs - causal_diag + cutlass.Int32(CFG.SWA_WINDOW + 1)
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
        row_dead = kv_abs >= eff_seqlen_kv
        hi_pad = cutlass.Int32(arith.select(row_dead.ir_value(), q_col_base.ir_value(), (q_col_base + cutlass.Int32(N)).ir_value()))
        if cutlass.const_expr(_THD):
            # The q band: uniform across lanes (an absolute q column), clamped at 0 so a dead unit's negative s_q keeps the
            # keep-word shift in range (the saturating shift masks everything either way).
            hi_pad = cute.math.min(hi_pad, cute.math.max(eff_seqlen_q, cutlass.Int32(0)))
        hi = hi_pad if hi is None else cute.math.min(hi, hi_pad)
    words = band_mask_words(lo, hi, q_col_base, N)
    return apply_mask_words(reg_P, words, mask_value=0.0, n_cols=N)


# ============================================================================
# Bars -- barrier inventory (init counts are the named constants above; see the table)
# ============================================================================


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
    # S / dP TMEM parity rings, sub-block indexed [s * STAGES_TMEM_S + slot]: MMA commit -> compute; compute LOAD-wait
    # -> MMA (the lookahead's explicit write-after-read handshake: the fp8 twin's mb_s_acc_empty, never an
    # issue-order invariant).
    mb_s_full: object
    mb_s_acc_empty: object
    mb_dp_full: object
    mb_dp_empty: object
    # SMEM P ring, sub-block indexed [s * STAGES_SMEM_P + pslot]: compute store + fence + .release.cta arrive -> MMA
    # (LEADER_RELEASE: the lane-written slab the peer-issued cta_group::2 MMA reads); MMA commit after BMM2 -> compute.
    mb_p_full: object
    mb_p_empty: object
    mb_stats_full: object
    mb_stats_empty: object
    mb_ds_smem_full: object
    mb_ds_smem_empty: object
    mb_dv_ready: object
    mb_dv_acc_empty: object
    mb_dv_stg_full: object
    mb_dv_stg_empty: object
    mb_qdk_full: object
    mb_qdk_empty: object
    mb_ds_mma_full: object
    mb_ds_mma_empty: object
    mb_tmem_dealloc: object


def _make_bars(cfg) -> Bars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        mb_qdk_full=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.TMA_LOAD) if cfg.FUSE_DK else None,
        mb_qdk_empty=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT) if cfg.FUSE_DK else None,
        mb_ds_mma_full=(
            MBarrier(_alloc(cfg.XFER_STAGES), stages=cfg.XFER_STAGES, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER_RELEASE, scope=Scope.LEADER)
            if cfg.FUSE_DK
            else None
        ),
        mb_ds_mma_empty=(
            MBarrier(_alloc(cfg.XFER_STAGES), stages=cfg.XFER_STAGES, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT) if cfg.FUSE_DK else None
        ),
        mb_q_full=MBarrier(_alloc(cfg.STAGES_Q), stages=cfg.STAGES_Q, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(cfg.STAGES_Q), stages=cfg.STAGES_Q, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_do_full=MBarrier(_alloc(cfg.STAGES_dO), stages=cfg.STAGES_dO, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_empty=MBarrier(_alloc(cfg.STAGES_dO), stages=cfg.STAGES_dO, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dodv_full=MBarrier(_alloc(cfg.STAGES_dO_DV), stages=cfg.STAGES_dO_DV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_dodv_empty=MBarrier(_alloc(cfg.STAGES_dO_DV), stages=cfg.STAGES_dO_DV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(cfg.STAGES_KV), stages=cfg.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(cfg.STAGES_KV), stages=cfg.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(cfg.STAGES_KV), stages=cfg.STAGES_KV, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(cfg.STAGES_KV), stages=cfg.STAGES_KV, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(N_S_BARS), stages=N_S_BARS, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # P8: the leader-waited fan-ins take the cluster-wide count on BOTH CTAs (the follower's copy is never armed
        # and never waited).  L_CNT = the sub-block's compute lanes x 2 CTAs, one bare arrive per lane after its LOAD wait.
        mb_s_acc_empty=MBarrier(_alloc(N_S_BARS), stages=N_S_BARS, init_count=L_CNT, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dp_full=MBarrier(_alloc(N_S_BARS), stages=N_S_BARS, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dp_empty=MBarrier(_alloc(N_S_BARS), stages=N_S_BARS, init_count=L_CNT, producer=Producer.LEADER, scope=Scope.LEADER),
        # The lane-written sP slab the peer-issued MMA reads: one .release.cta arrive per writer lane after its
        # fence_proxy (L_CNT = the sub-block's writer lanes x 2 CTAs; probe release_p: 512 with 8 warps).
        mb_p_full=MBarrier(_alloc(N_P_BARS), stages=N_P_BARS, init_count=L_CNT, producer=Producer.LEADER_RELEASE, scope=Scope.LEADER),
        mb_p_empty=MBarrier(_alloc(N_P_BARS), stages=N_P_BARS, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # lse / do_dot prefetch ring: scheduler warp (32 lanes, bare) -> compute; compute (256 lanes, bare) -> scheduler.
        mb_stats_full=MBarrier(_alloc(cfg.STATS_STAGES), stages=cfg.STATS_STAGES, init_count=ONE_WARP, producer=Producer.THREAD),
        mb_stats_empty=MBarrier(_alloc(cfg.STATS_STAGES), stages=cfg.STATS_STAGES, init_count=SOFTMAX_LANES_ALL_WG, producer=Producer.THREAD),
        # dS SMEM ring: compute store_swizzled (all 256 lanes, bare) -> TMA-STG; TMA-STG `if elect_sync():` -> compute.
        mb_ds_smem_full=MBarrier(_alloc(cfg.XFER_STAGES), stages=cfg.XFER_STAGES, init_count=SOFTMAX_LANES_ALL_WG, producer=Producer.THREAD),
        mb_ds_smem_empty=MBarrier(_alloc(cfg.XFER_STAGES), stages=cfg.XFER_STAGES, init_count=ONE_LANE, producer=Producer.THREAD),
        # dV epilogue (per kv block): MMA(dV) -> compute warps -> TMA-STG -> TMA-LDG (sdV aliases sK).
        mb_dv_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dv_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dv_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES_ALL_WG, producer=Producer.THREAD),
        mb_dv_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.THREAD),
        # 256 local bare arrives + 256 bare arrive_on_peer from the partner's compute lanes, on EACH CTA.
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.THREAD),
    )


# ============================================================================
# Tile decode (trace-time helpers; NATURAL 3-D grid or the LPT / LPT_L2 flat grid)
# ============================================================================


def _decode_linear(linear, n_qh_grid, n_batch):
    """Flat 1-D (LPT / LPT_L2) decode: linear cluster id -> (kv_super, head, batch); kv_super is the OUTER axis."""
    hb = n_qh_grid * n_batch
    kv_super = linear // hb
    within = linear % hb
    head = within % n_qh_grid
    batch = within // n_qh_grid
    return kv_super, head, batch


def _thd_decode(linear, meta_t, n_batch, n_qh):
    """THD: (sequence-local kv block, head, batch) of unit ``linear // CGA_M`` -- ``tile_dsl.thd.thd_decode_unit`` over the KV
    prefix (``cu0 = 2B+1``). Each 256-row write pair expands into two 128-row compute units per head, including a
    masked tail half, matching the setup's ``live`` count. Sequences run longest-first through ``batch_remap``. A unit past ``live`` decodes
    ``batch == n_batch`` (the dead-unit sentinel ``_seq_ctx`` turns into a fully masked tile)."""
    meta = cutlass.make_array_view(meta_t)
    uid = cute.arch.make_warp_uniform(linear) // cutlass.Int32(CFG.CGA_M)
    kv_blk, batch, head = thd_decode_unit_prefix(
        meta,
        n_batch,
        uid,
        n_qh * cutlass.Int32(_THD_BLOCKS_PER_WRITE),
        cutlass.Int32(_KV_WRITE_ROWS),
        False,
        cu0=cutlass.Int32(2) * n_batch + cutlass.Int32(1),
        prefix0=THD_BWD_WORK_PREFIX_OFF(n_batch),
    )
    # Every 256-row write pair needs both 128-row blocks, including its masked tail.
    kv_blk = kv_blk * cutlass.Int32(_THD_BLOCKS_PER_WRITE) + head % cutlass.Int32(_THD_BLOCKS_PER_WRITE)
    head = head // cutlass.Int32(_THD_BLOCKS_PER_WRITE)
    return cute.arch.make_warp_uniform(kv_blk), cute.arch.make_warp_uniform(head), cute.arch.make_warp_uniform(batch)


def _boot_tile(sched, meta_t, n_batch, n_qh_grid):
    """The FIRST tile (kv_super, head, batch) from the launch blockIdx.  NATURAL: (bidx // CGA_M,
    bidy, bidz).  LPT / LPT_L2: bidx is the linear cluster base and bidy_init / bidz_init are
    REPURPOSED to carry (n_qh_grid, n_batch) -- dead under a (N, 1, 1) grid.  THD: bidx // CGA_M
    is the cluster's first UNIT id (the claim counter starts at the cluster count), decoded through
    the metadata (``_thd_decode``); ``meta_t`` / ``n_batch`` / ``n_qh_grid`` are unused otherwise."""
    if cutlass.const_expr(_THD):
        return _thd_decode(sched.bidx_init, meta_t, n_batch, n_qh_grid)
    linear = sched.bidx_init // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv, h, b = linear, sched.bidy_init, sched.bidz_init
    else:
        kv, h, b = _decode_linear(linear, sched.bidy_init, sched.bidz_init)
    return cute.arch.make_warp_uniform(kv), cute.arch.make_warp_uniform(h), cute.arch.make_warp_uniform(b)


def _decode_tile_words(sched, t0, t1, meta_t, n_batch, n_qh_grid):
    """(kv_super, head, batch) from the scheduler payload words (``read_clc_payload``): t0 = first
    ctaid.x of the cluster (= kv_super * CGA_M under NATURAL, linear * CGA_M under LPT), t1 =
    packed (head = low 16, batch = high 16) under NATURAL (y = z = 0 on the flat grid).  THD: the
    claim loop puts ``unit * CGA_M`` in t0 and leaves t1 unused (``_thd_decode``)."""
    if cutlass.const_expr(_THD):
        return _thd_decode(t0, meta_t, n_batch, n_qh_grid)
    linear = cute.arch.make_warp_uniform(t0) // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        t1u = cute.arch.make_warp_uniform(t1)
        return linear, t1u & cutlass.Int32(0xFFFF), (t1u >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
    return _decode_linear(linear, sched.bidy_init, sched.bidz_init)


# ============================================================================
# Kernel entry
# ============================================================================


@cute.kernel
def _kernel(
    # TMA descriptors -- loads
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q   [B, S_q, H_q, d]    box (1, 64, 1, 64)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  dP view            box (1, 64, 1, 64)
    tma_do_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  dV view (BT)       box (1, 128, 1, 64)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],  # K   [B, S_kv, H_kv, d] box (1, 64, 1, 64)
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],  # V
    # TMA descriptors -- stores
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dV  [B, S_kv, H_q, d_v]  box (1, ROWS_PER_CTA, 1, 64)
    tma_ds_desc: cutlass.GridConstant[tmap.TensorMap],  # dS  [B_chunk, H_chunk, S_kv, S_q]  box (1, 1, ROWS_PER_CTA, 64)
    # GMEM vectors
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32, natural log
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32, raw rowsum(dO * O)
    seq_kv_lens_tensor: cute.Tensor,  # [B] int32 (PADDED arm only)
    # Scalars
    seqlen_q: cutlass.Int32,  # padded S_q (drives n_q_tiles)
    seqlen_q_real: cutlass.Int32,
    seqlen_kv_real: cutlass.Int32,
    n_batch: cutlass.Int32,  # grid batch extent (B_CHUNK); LPT flat-grid decode
    n_qh_grid: cutlass.Int32,  # grid head extent (QH_CHUNK); LPT flat-grid decode
    qh_per_kh: cutlass.Int32,
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    dv_head_base: cutlass.Int32,
    dv_batch_base: cutlass.Int32,
    thd_origins,
    thd_origin_rows: cutlass.Constexpr,
    tma_qdk_desc: Optional[cutlass.GridConstant[tmap.TensorMap]],
    tma_dk_desc: Optional[cutlass.GridConstant[tmap.TensorMap]],
    dk_maps,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- SharedStorage: DECLARATION ORDER == config_d256_2x2.smem_layout_2x2 (sQ | sdO | sdOdv | sK | sV | sP | sStats | sdS) ---
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdOdv_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO_DV * dOdvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(STORAGE_DTYPE, N_P_BARS * pSlabElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sStats_raw = cutlass.Array(cutlass.Float32, CFG.STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sdS_raw = cutlass.Array(DS_STORAGE_DTYPE, CFG.XFER_STAGES * dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sQdk_raw = None
    sdK_raw = None
    if cutlass.const_expr(CFG.FUSE_DK):
        sQdk_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
        # All P reads finish before the joint dK/dV epilogue. The existing output
        # empty credit gates the next K load until both stores consume these aliases.
        sdK_raw = cutlass.Array(sP_raw.data_ptr(), shape=kBufferElems, dtype=OUT_STORAGE_DTYPE)
    # dV epilogue staging: an OUT_STORAGE_DTYPE view of sK (dVBufferElems * BPE_O <= kBufferElems * BPE, config-pinned), live
    # only after the q loop (the TMA-LDG waits mb_dv_stg_empty before the next block's K load).
    sdV_raw = cutlass.Array(sK_raw.data_ptr(), shape=dVBufferElems, dtype=OUT_STORAGE_DTYPE)

    # --- SmemTile wrappers (every one takes desc_version=DESC_VERSION) ----------------------
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=CFG.STAGES_Q,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # dO, dP view: BMM1 dP B operand (BT=false, 64 q x 256 d_v per CTA, 4 subtiles).
    sdO = SmemTile(
        base=sdO_raw,
        elems_per_stage=dOBufferElems,
        stages=CFG.STAGES_dO,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=_M_PER_CTA * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # dO, dV view: BMM2 B operand (BT=true, 128 q x 128 d_v per CTA, 2 subtiles of 16 KiB, leading = TILE_N x swz).
    sdO_dv = SmemTile(
        base=sdOdv_raw,
        elems_per_stage=dOdvBufferElems,
        stages=CFG.STAGES_dO_DV,
        leading_byte_offset=LEADING_BYTE_OFFSET_dO_SG1,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dO_SG1,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=TMA_VO_SG1_ITERS,
        tma_granu_elems=TMA_VO_SG1_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_SG1_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # K / V: KV_SUBBLOCKS "stages" = the per-sub-block 64 x 256 slabs (4 subtiles of 8 KiB each); the kv ring itself is
    # STAGES_KV = 1 deep (one block of K / V resident at a time).
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kSubElems,
        stages=KV_SUBBLOCKS,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=SUB_ROWS * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vSubElems,
        stages=KV_SUBBLOCKS,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=TMA_VO_ITERS,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=SUB_ROWS * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # P ring: [64 kv x 128 q] bf16 K-major SW128 per (sub-block, stage) = 2 subtiles of 8 KiB; the BMM2 A operand (SS).
    # Lane-written (store_swizzled), never TMA-loaded.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pSlabElems,
        stages=N_P_BARS,
        leading_byte_offset=LEADING_BYTE_OFFSET_P,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_SW128,
        desc_version=DESC_VERSION,
    )
    # dS ring: [ROWS_PER_CTA kv x 128 q] io dtype, P_TMA_ITERS store subtiles of P_D_BLOCK q cols (128 B rows).
    sdS = SmemTile(
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_SW128,
        tma_loads_per_tile=P_TMA_ITERS,
        tma_granu_elems=P_D_BLOCK,
        tma_subtile_stride_elems=DS_BLOCK_SLAB,
        desc_version=DESC_VERSION,
    )
    sQdk = None
    sdS_mma = None
    sdK = None
    if cutlass.const_expr(CFG.FUSE_DK):
        sQdk = SmemTile(
            base=sQdk_raw,
            elems_per_stage=qBufferElems,
            stages=1,
            leading_byte_offset=128 * 64,
            stride_byte_offset=8 * 64,
            layout=4,
            tma_loads_per_tile=3,
            tma_granu_elems=32,
            tma_subtile_stride_elems=128 * 32,
            desc_version=DESC_VERSION,
        )
        sdS_mma = SmemTile(
            base=sdS_raw,
            elems_per_stage=dSBufferElems,
            stages=CFG.XFER_STAGES,
            leading_byte_offset=LEADING_BYTE_OFFSET_P,
            stride_byte_offset=STRIDE_BYTE_OFFSET_P,
            layout=SMEM_LAYOUT_SW128,
            desc_version=DESC_VERSION,
        )
        sdK = SmemTile(
            base=sdK_raw,
            elems_per_stage=kBufferElems,
            stages=1,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=SMEM_LAYOUT_SW128,
            tma_loads_per_tile=3,
            tma_granu_elems=64,
            tma_subtile_stride_elems=64 * 64,
            desc_version=DESC_VERSION,
        )
    # dV staging: TMA_DV_ITERS subtiles of (ROWS_PER_CTA kv x DV_D_BLOCK d_v), 128 B swizzled; a TMA-STG source only.
    sdV = SmemTile(
        base=sdV_raw,
        elems_per_stage=dVBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_SW128,
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
            "bidy_init": (n_qh_grid if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidy),
            "bidz_init": (n_batch if cutlass.const_expr(CFG.SCHEDULER_POLICY != SCHED_NATURAL) else bidz),
        }
    )

    # --- cluster identity (one cga2 pair) ---------------------------------------------------
    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    partner_cta_id = cta_id_x ^ cutlass.Int32(1)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # --- mbarrier init (P4: ONE warp, ONE lane; every stage of every ring) ----------------
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
            for p in cutlass.range_constexpr(N_S_BARS):
                bars.mb_s_full[p].init()
                bars.mb_s_acc_empty[p].init()
                bars.mb_dp_full[p].init()
                bars.mb_dp_empty[p].init()
            for p in cutlass.range_constexpr(N_P_BARS):
                bars.mb_p_full[p].init()
                bars.mb_p_empty[p].init()
            if cutlass.const_expr(CFG.FUSE_DK):
                bars.mb_qdk_full.init()
                bars.mb_qdk_empty.init()
                for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                    bars.mb_ds_mma_full[p].init()
                    bars.mb_ds_mma_empty[p].init()
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
            for s in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    # P4 order: init -> fence -> CTA sync -> cluster sync, all OUTSIDE the warp branch.
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    # Pair-scoped tcgen05.commit multicast mask and this CTA's TMA multicast bit (self-only: a cga2 tensor TMA routes both
    # peers' bytes to the leader's mbar, P9).
    mcast_mask = cutlass.Int32(3) << leader_cta_id
    tma_mcast_mask = cutlass.Int16(1) << cta_in_pair
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # --- warp dispatch ------------------------------------------------------------------------
    if warp_idx < cutlass.Int32(CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _compute_warp(
            warp_idx=warp_idx,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            sP_raw=sP_raw,
            sdS_raw=sdS_raw,
            sStats_raw=sStats_raw,
            sdV_raw=sdV_raw,
            sdK_raw=sdK_raw,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            n_qh_grid=n_qh_grid,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            attn_scale=attn_scale,
            attn_scale_log2e=attn_scale_log2e,
            batch_base=batch_base,
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
                sP=sP,
                sQdk=sQdk,
                sdS_mma=sdS_mma,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_batch=n_batch,
                n_qh_grid=n_qh_grid,
                seqlen_q=seqlen_q,
                seqlen_q_real=seqlen_q_real,
                seqlen_kv_real=seqlen_kv_real,
                batch_base=batch_base,
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
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            n_qh_grid=n_qh_grid,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            qh_per_kh=qh_per_kh,
            head_base=head_base,
            batch_base=batch_base,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
            physical_origins=thd_origins is not None,
        )

    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_ds_desc.get_ptr())
        _tmastg_warp(
            dk_maps=dk_maps,
            tma_qdk_desc=tma_qdk_desc,
            sQdk=sQdk,
            tma_dk_desc=tma_dk_desc,
            sdK=sdK,
            tma_dv_desc=tma_dv_desc,
            tma_ds_desc=tma_ds_desc,
            sdV=sdV,
            sdS=sdS,
            bars=bars,
            sched=sched,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            n_qh_grid=n_qh_grid,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            head_base=head_base,
            batch_base=batch_base,
            cta_in_pair=cta_in_pair,
            dv_head_base=dv_head_base,
            dv_batch_base=dv_batch_base,
            physical_origins=thd_origins is not None,
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
            thd_origins=thd_origins,
            stats_origin_row=thd_origin_rows[8],
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            n_qh_grid=n_qh_grid,
            attn_scale=attn_scale,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            head_base=head_base,
            batch_base=batch_base,
        )


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ============================================================================
# Warp bodies, in pipeline order: TMA-LDG -> MMA -> compute -> TMA-STG -> scheduler
# ============================================================================


@cute.jit
def _load_q_tile(
    sQ,
    bars,
    q_state,
    tma_q_desc,
    seq_kv_lens_tensor,
    n_batch,
    batch_idx,
    physical_origins: cutlass.Constexpr,
    full_head,
    q_row,
    tma_batch,
    is_leader,
    tma_mcast_mask,
):
    """Issue one Q tile after its last reader and advance the producer ring."""
    bars.mb_q_empty[q_state.idx].wait(q_state.phase, spin=SPIN_RING_WAITS)
    bars.mb_q_full[q_state.idx].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
    tma_load_tile(
        sQ[q_state.idx],
        _input_slice(
            tma_q_desc,
            seq_kv_lens_tensor,
            n_batch,
            THD_Q_SLOT,
            batch_idx,
            physical_origins,
            cutlass.Int32(0),
            full_head,
            q_row,
            tma_batch,
        ),
        bars.mb_q_full[q_state.idx].smem_ptr,
        cta_group=CFG.CTA_MMA,
        mcast_mask=tma_mcast_mask,
        acquire=not _THD,
    )
    return advance(q_state, CFG.STAGES_Q)


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
    seq_kv_lens_tensor,
    n_batch,
    n_qh_grid,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    qh_per_kh,
    head_base,
    batch_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
    physical_origins: cutlass.Constexpr = False,
) -> None:
    """TMA-LDG warp (both CTAs).  Per kv block: K, V once (KV_SUBBLOCKS sub-block slabs of 64 rows at kv
    ``kv_block_base + cta_in_pair * ROWS_PER_CTA + 64 * s``, full d).  Per q tile: Q (N-split on q), dO dP view (same
    split), dO dV view (N-split on d_v, full q rows, BT).  Only the LEADER arms expect_tx (P9: cga2 tensor TMAs deliver
    both peers' bytes to the leader's mbar); both CTAs issue their own loads.
    """
    if cutlass.const_expr(_THD and not physical_origins):
        _q_rt = _thd_map_ptr(seq_kv_lens_tensor, n_batch, cutlass.Int32(THD_Q_SLOT))
        _do_rt = _thd_map_ptr(seq_kv_lens_tensor, n_batch, cutlass.Int32(THD_DO_SLOT))
        _dodv_rt = _thd_map_ptr(seq_kv_lens_tensor, n_batch, cutlass.Int32(THD_DODV_SLOT))
        _k_rt = _thd_map_ptr(seq_kv_lens_tensor, n_batch, cutlass.Int32(THD_K_SLOT))
        _v_rt = _thd_map_ptr(seq_kv_lens_tensor, n_batch, cutlass.Int32(THD_V_SLOT))
        tma_tensormap_acquire(_q_rt)
        tma_tensormap_acquire(_do_rt)
        tma_tensormap_acquire(_dodv_rt)
        tma_tensormap_acquire(_k_rt)
        tma_tensormap_acquire(_v_rt)
        # Each loader warp acquires these immutable maps on both CTAs on every
        # launch. Physical-origin maps are acquired below at each work item.
        # The tile helpers therefore suppress only their redundant THD acquire.

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched, seq_kv_lens_tensor, n_batch, n_qh_grid)
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
    full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)

    DO_DV_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)  # this CTA's d_v half of the BT view
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(ROWS_PER_CTA)  # this CTA's contiguous kv rows of the block
    Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_M_PER_CTA)  # this CTA's q half of the tile

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    dodv_empty_state = PipelineState.start(phase=1)
    # sdV aliases sK: the K load of block t must not land while the TMA-STG still READS the dV staging of block t-1
    # (its tma_store_wait -> mb_dv_stg_empty is the only signal).  Pre-armed: block 0 has nothing to wait for.
    dv_stg_empty_state = PipelineState.start(phase=1)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        ctx = _seq_ctx(seq_kv_lens_tensor, batch_idx, n_batch, seqlen_q_real, seqlen_kv_real, physical_origins)
        eff_seqlen_q, eff_seqlen_kv = _eff_lengths(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_q_real, seqlen_kv_real, ctx)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, eff_seqlen_q, eff_seqlen_kv)
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(ctx, full_batch)

        if cutlass.const_expr(_THD and physical_origins):
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (cutlass.Int32(THD_INPUT_SLOTS) + cutlass.Int32(THD_Q_SLOT) * n_batch + batch_idx) if batch_idx < n_batch else cutlass.Int32(THD_Q_SLOT),
                )
            )
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (cutlass.Int32(THD_INPUT_SLOTS) + cutlass.Int32(THD_DO_SLOT) * n_batch + batch_idx) if batch_idx < n_batch else cutlass.Int32(THD_DO_SLOT),
                )
            )
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (
                        (cutlass.Int32(THD_INPUT_SLOTS) + cutlass.Int32(THD_DODV_SLOT) * n_batch + batch_idx)
                        if batch_idx < n_batch
                        else cutlass.Int32(THD_DODV_SLOT)
                    ),
                )
            )
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (cutlass.Int32(THD_INPUT_SLOTS) + cutlass.Int32(THD_K_SLOT) * n_batch + batch_idx) if batch_idx < n_batch else cutlass.Int32(THD_K_SLOT),
                )
            )
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (cutlass.Int32(THD_INPUT_SLOTS) + cutlass.Int32(THD_V_SLOT) * n_batch + batch_idx) if batch_idx < n_batch else cutlass.Int32(THD_V_SLOT),
                )
            )

        # ---- K + V, once per kv block: KV_SUBBLOCKS x 4 boxes each; one expect_tx per ring slot with BOTH peers' bytes ----
        bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase, spin=SPIN_RING_WAITS)
        dv_stg_empty_state = advance(dv_stg_empty_state, 1)
        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        for s in cutlass.range_constexpr(KV_SUBBLOCKS):
            tma_load_tile(
                sK[s],
                _input_slice(
                    tma_k_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    THD_K_SLOT,
                    batch_idx,
                    physical_origins,
                    cutlass.Int32(0),
                    kv_head_idx,
                    kv_seq_off + kv_block_base + K_ROW_OFFSET_PEER + cutlass.Int32(s * SUB_ROWS),
                    tma_batch,
                ),
                bars.mb_k_full[k_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=not _THD,
            )
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        for s in cutlass.range_constexpr(KV_SUBBLOCKS):
            tma_load_tile(
                sV[s],
                _input_slice(
                    tma_v_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    THD_V_SLOT,
                    batch_idx,
                    physical_origins,
                    cutlass.Int32(0),
                    kv_head_idx,
                    kv_seq_off + kv_block_base + K_ROW_OFFSET_PEER + cutlass.Int32(s * SUB_ROWS),
                    tma_batch,
                ),
                bars.mb_v_full[v_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=not _THD,
            )
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

        # ---- Q + dO + dO_dv, per q tile of the pair-rounded range ----
        if cutlass.const_expr(MMA_LOOKAHEAD):
            if q_lo < q_hi:
                q_empty_state = _load_q_tile(
                    sQ,
                    bars,
                    q_empty_state,
                    tma_q_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    batch_idx,
                    physical_origins,
                    full_head,
                    q_seq_off + q_lo * cutlass.Int32(CFG.TILE_N) + Q_ROW_OFFSET_PEER,
                    tma_batch,
                    is_leader,
                    tma_mcast_mask,
                )

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_N)

            if cutlass.const_expr(not MMA_LOOKAHEAD):
                q_empty_state = _load_q_tile(
                    sQ,
                    bars,
                    q_empty_state,
                    tma_q_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    batch_idx,
                    physical_origins,
                    full_head,
                    q_seq_off + q_row_base + Q_ROW_OFFSET_PEER,
                    tma_batch,
                    is_leader,
                    tma_mcast_mask,
                )

            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_do_full[do_empty_state.idx].arrive(n_bytes=dOTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO[do_empty_state.idx],
                _input_slice(
                    tma_do_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    THD_DO_SLOT,
                    batch_idx,
                    physical_origins,
                    cutlass.Int32(0),
                    full_head,
                    q_seq_off + q_row_base + Q_ROW_OFFSET_PEER,
                    tma_batch,
                ),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=not _THD,
            )
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)

            if cutlass.const_expr(MMA_LOOKAHEAD):
                # Match the MMA's S(i+1)-before-dV(i) order. Waiting for the
                # dO_dV slot first would delay the next Q even with a deeper ring.
                if q_iter + cutlass.Int32(1) < q_hi:
                    q_empty_state = _load_q_tile(
                        sQ,
                        bars,
                        q_empty_state,
                        tma_q_desc,
                        seq_kv_lens_tensor,
                        n_batch,
                        batch_idx,
                        physical_origins,
                        full_head,
                        q_seq_off + (q_iter + cutlass.Int32(1)) * cutlass.Int32(CFG.TILE_N) + Q_ROW_OFFSET_PEER,
                        tma_batch,
                        is_leader,
                        tma_mcast_mask,
                    )

            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_dodv_full[dodv_empty_state.idx].arrive(n_bytes=dOdvTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO_dv[dodv_empty_state.idx],
                _input_slice(
                    tma_do_dv_desc,
                    seq_kv_lens_tensor,
                    n_batch,
                    THD_DODV_SLOT,
                    batch_idx,
                    physical_origins,
                    DO_DV_OFFSET_PEER,
                    full_head,
                    q_seq_off + q_row_base,
                    tma_batch,
                ),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
                acquire=not _THD,
            )
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1, seq_kv_lens_tensor, n_batch, n_qh_grid)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: drain EVERY cross-CTA _empty ring this warp consumes, OUTSIDE the persistent loop (the leader's last
    # multicast commits are still in flight when it leaves its loop; this CTA must stay resident until they land).
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _s in cutlass.range_constexpr(CFG.STAGES_Q):
            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase)
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)
        for _s in cutlass.range_constexpr(CFG.STAGES_dO):
            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase)
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)
        for _s in cutlass.range_constexpr(CFG.STAGES_dO_DV):
            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase)
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)
        for _s in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
            k_empty_state = advance(k_empty_state, CFG.STAGES_KV)
            bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
            v_empty_state = advance(v_empty_state, CFG.STAGES_KV)
    # The LOCAL dV-staging ring: the last kv block's TMA-STG arrive is unconsumed; a symmetric 1-deep drain keeps an
    # init-count imbalance a localizable hang.
    bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase)


@cute.jit
def _mma_warp(
    sQdk,
    sdS_mma,
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    sP,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_batch,
    n_qh_grid,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    batch_base,
    mcast_mask,
) -> None:
    """MMA leader -- the 3-MMA stream, all SS, every commit.

    Per kv block (K, V once): S(q_lo)[s]; then per q tile i: dP(i)[s]; S(i+1)[s] if i+1 < q_hi (lookahead; the NATURAL
    arm issues S(i) at the top of the iteration instead); BMM2(i)[s] (accumulate = i > q_lo).  Handshakes: s_acc_empty
    (pre-armed) -> S -> s_full; dp_empty (pre-armed) -> dP -> dp_full; p_full -> BMM2 -> p_empty; the *_empty commits
    -> TMA-LDG; dv_ready / dv_acc_empty around the epilogue.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    # Publish tmem_ptr_i32 to the compute warps (their barrier_cta_sync(1) at top of body).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()
    dodv_full_state = PipelineState.start()
    if cutlass.const_expr(CFG.FUSE_DK):
        qdk_full_state = PipelineState.start()
        ds_mma_full_state = PipelineState.start()
    s_acc_empty_state = PipelineState.start(phase=1)  # pre-armed: both S slots start free
    dp_empty_state = PipelineState.start(phase=1)  # pre-armed: both dP slots start free
    p_full_state = PipelineState.start(phase=0)
    dv_empty_state = PipelineState.start(phase=0)

    kv_super_idx, _head_idx, batch_idx = _boot_tile(sched, seq_kv_lens_tensor, n_batch, n_qh_grid)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_BASE))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_BASE))
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_BASE))
    if cutlass.const_expr(CFG.FUSE_DK):
        tmem_dK = tmem_raw.subview(cutlass.Int32(LAYOUT.dK_BASE))
        idesc_dk = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=192, m_dim=CFG.MMA_M, a_major=0, b_major=1, k_dim=CFG.IDESC_K_DIM
        )
        dk_desc = MmaDesc(
            M=CFG.MMA_M,
            N=192,
            K=CFG.TILE_N,
            bpe_a=CFG.BPE,
            bpe_b=CFG.BPE,
            tile_k_hw=CFG.TILE_K_HW_BMM2,
            btranspose=True,
            cta_group=CFG.CTA_MMA,
            idesc=idesc_dk,
            kind=MMA_KIND,
        )

    # ---- instruction / operand descriptors (k_dim = CFG.IDESC_K_DIM, the f16 default 0; m_dim = the collective 128) ----
    # BMM1 S = K . Q^T and dP = V . dO^T share one shape: A = K / V (M = kv, 64 per CTA), B = Q / dO (N = q, 64 per CTA),
    # K = d = 256 (16 k-steps; 4 SW128 subtiles of 8 KiB per operand).
    idesc_bmm1 = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.MMA_M,
        k_dim=CFG.IDESC_K_DIM,
    )
    bmm1_desc = MmaDesc(
        M=CFG.MMA_M,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1,
        kind=MMA_KIND,
    )
    # dP contracts the value width, independently of the score contraction.
    dp_desc = MmaDesc(
        M=CFG.MMA_M,
        N=CFG.TILE_N,
        K=CFG.TILE_O,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1,
        kind=MMA_KIND,
    )
    # BMM2 dV += P . dO (A = P in SMEM [64 kv x 128 q] K-major, 2 subtiles of 8 KiB; B = dO_dv BT [128 q x 128 d_v per CTA],
    # N = d_v = 256 collective, K = q = 128: 8 k-steps of 2 KiB B advances inside one 16 KiB column subtile).
    idesc_bmm2 = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.MMA_M,
        a_major=0,
        b_major=1,
        k_dim=CFG.IDESC_K_DIM,
    )
    bmm2_desc = MmaDesc(
        M=CFG.MMA_M,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm2,
        kind=MMA_KIND,
    )

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        # ONE elect per work item for every predicated commit of this tile: the warp is converged here and the
        # predicated arrives below never diverge it (each is a branch round the native op that reconverges).
        elect_p = nvvm.elect_sync()

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        ctx = _seq_ctx(seq_kv_lens_tensor, batch_idx, n_batch, seqlen_q_real, seqlen_kv_real)
        eff_seqlen_q, eff_seqlen_kv = _eff_lengths(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_q_real, seqlen_kv_real, ctx)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, eff_seqlen_q, eff_seqlen_kv)

        # K + V once per kv block.
        bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase, spin=SPIN_RING_WAITS)
        # K / V descriptor bases, pinned per kv block (desc_opaque + the kv-block counter as the loop-variant anchor, the
        # spill lesson of the 4x1 body: unpinned, LLVM hoists every `base + k_step` to the persistent loop's preheader).
        desc_K = [desc_opaque(sK[s].desc(), anchor=kv_super_idx) for s in range(KV_SUBBLOCKS)]
        desc_V = [desc_opaque(sV[s].desc(), anchor=kv_super_idx) for s in range(KV_SUBBLOCKS)]

        if cutlass.const_expr(MMA_LOOKAHEAD):
            # ---- prologue: S(q_lo)[s] -> the first S slot of every sub-block ----
            bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
            # The B descriptor is anchored on the tile index too: a 1-stage ring's slab address is STATIC, so an unanchored
            # `sQ[0].desc() + k_step` is kernel-invariant and LLVM hoists all 16 adds to the preheader (measured 70 STL /
            # 79 LDL in the 56-register MMA warp, 2026-10-01, sm_100a); anchored, they live inside this tile only.
            desc_Q = desc_opaque(sQ[q_full_state.idx].desc(), anchor=q_lo)
            for s in cutlass.range_constexpr(KV_SUBBLOCKS):
                s_bar = s * CFG.STAGES_TMEM_S + s_acc_empty_state.idx
                bars.mb_s_acc_empty[s_bar].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                mma_ss(
                    bmm1_desc,
                    desc_K[s],
                    desc_Q,
                    tmem_S.subview(cutlass.Int32(s * LAYOUT.SUB_STRIDE) + s_acc_empty_state.idx * cutlass.Int32(LAYOUT.S_COLS)),
                    accumulate=False,
                    elect_once=True,
                )
                bars.mb_s_full[s_bar].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
            bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            q_full_state = advance(q_full_state, CFG.STAGES_Q)
        else:
            # NATURAL: `desc_Q` / `s_bar` are (re)defined at the top of every iteration below, but the DSL types a name
            # assigned under a branch at the branch's join, so each must exist with the SAME type on the other path too
            # (TYPE_UNSTABLE_JOIN otherwise; the lookahead arm defines both in its prologue).  One mov.b64 of slot 0's
            # base and a folded constant, never consumed as these values.
            desc_Q = desc_opaque(sQ[0].desc(), anchor=q_lo)
            s_bar = cutlass.Int32(0)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            if cutlass.const_expr(not MMA_LOOKAHEAD):
                # ---- NATURAL: S(i)[s] at the top of the iteration ----
                bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
                desc_Q = desc_opaque(sQ[q_full_state.idx].desc(), anchor=q_iter)
                for s in cutlass.range_constexpr(KV_SUBBLOCKS):
                    s_bar = s * CFG.STAGES_TMEM_S + s_acc_empty_state.idx
                    bars.mb_s_acc_empty[s_bar].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                    mma_ss(
                        bmm1_desc,
                        desc_K[s],
                        desc_Q,
                        tmem_S.subview(cutlass.Int32(s * LAYOUT.SUB_STRIDE) + s_acc_empty_state.idx * cutlass.Int32(LAYOUT.S_COLS)),
                        accumulate=False,
                        elect_once=True,
                    )
                    bars.mb_s_full[s_bar].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
                bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ---- dP(i)[s] = V . dO[i]^T: gated on the dP slot being free (dSoftmax(i-2) read it; pre-armed for the first two) ----
            bars.mb_do_full[do_full_state.idx].wait(do_full_state.phase, spin=SPIN_RING_WAITS)
            desc_dO = desc_opaque(sdO[do_full_state.idx].desc(), anchor=q_iter)
            for s in cutlass.range_constexpr(KV_SUBBLOCKS):
                dp_bar = s * CFG.STAGES_TMEM_S + dp_empty_state.idx
                bars.mb_dp_empty[dp_bar].wait(dp_empty_state.phase, spin=SPIN_RING_WAITS)
                mma_ss(
                    dp_desc,
                    desc_V[s],
                    desc_dO,
                    tmem_dP.subview(cutlass.Int32(s * LAYOUT.SUB_STRIDE) + dp_empty_state.idx * cutlass.Int32(LAYOUT.S_COLS)),
                    accumulate=False,
                    elect_once=True,
                )
                bars.mb_dp_full[dp_bar].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)
            bars.mb_do_empty[do_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            do_full_state = advance(do_full_state, CFG.STAGES_dO)

            if cutlass.const_expr(MMA_LOOKAHEAD):
                # ---- S(i+1)[s] (lookahead; not on the last iteration): the fp8 twin's form ----
                if (q_iter + cutlass.Int32(1)) < q_hi:
                    bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
                    desc_Q = desc_opaque(sQ[q_full_state.idx].desc(), anchor=q_iter)
                    for s in cutlass.range_constexpr(KV_SUBBLOCKS):
                        s_bar = s * CFG.STAGES_TMEM_S + s_acc_empty_state.idx
                        bars.mb_s_acc_empty[s_bar].wait(s_acc_empty_state.phase, spin=SPIN_RING_WAITS)
                        mma_ss(
                            bmm1_desc,
                            desc_K[s],
                            desc_Q,
                            tmem_S.subview(cutlass.Int32(s * LAYOUT.SUB_STRIDE) + s_acc_empty_state.idx * cutlass.Int32(LAYOUT.S_COLS)),
                            accumulate=False,
                            elect_once=True,
                        )
                        bars.mb_s_full[s_bar].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
                    bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                    q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # ---- BMM2(i)[s]: dV += P[i] . dO[i] (A = the lane-written sP slot the compute warps published) ----
            # accumulate restarts at q_lo (NOT 0): the first q tile of every kv block OVERWRITES dV (rules/frost-tile-dsl.md S2).
            bars.mb_dodv_full[dodv_full_state.idx].wait(dodv_full_state.phase, spin=SPIN_RING_WAITS)
            desc_dOdv = desc_opaque(sdO_dv[dodv_full_state.idx].desc(), anchor=q_iter)
            for s in cutlass.range_constexpr(KV_SUBBLOCKS):
                p_bar = s * CFG.STAGES_SMEM_P + p_full_state.idx
                bars.mb_p_full[p_bar].wait(p_full_state.phase, spin=SPIN_RING_WAITS)
                mma_ss(
                    bmm2_desc,
                    desc_opaque(sP[p_bar].desc(), anchor=q_iter),
                    desc_dOdv,
                    tmem_dV.subview(cutlass.Int32(s * LAYOUT.SUB_STRIDE)),
                    accumulate=(q_iter > q_lo),
                    elect_once=True,
                )
                bars.mb_p_empty[p_bar].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            p_full_state = advance(p_full_state, CFG.STAGES_SMEM_P)
            bars.mb_dodv_empty[dodv_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dodv_full_state = advance(dodv_full_state, CFG.STAGES_dO_DV)
            if cutlass.const_expr(CFG.FUSE_DK):
                ds_slot = ds_mma_full_state.idx
                bars.mb_ds_mma_full[ds_slot].wait(ds_mma_full_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_qdk_full.wait(qdk_full_state.phase, spin=SPIN_RING_WAITS)
                mma_ss(
                    dk_desc,
                    desc_opaque(sdS_mma[ds_slot].desc(), anchor=q_iter),
                    desc_opaque(sQdk[0].desc(), anchor=q_iter),
                    tmem_dK,
                    accumulate=(q_iter > q_lo),
                    elect_once=True,
                )
                bars.mb_ds_mma_empty[ds_slot].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_qdk_empty.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                ds_mma_full_state = advance(ds_mma_full_state, CFG.XFER_STAGES)
                qdk_full_state = advance(qdk_full_state, 1)

        # dV complete for this kv block -> epilogue (compute warps); wait for it to drain dV before the next block's
        # BMM2(q_lo) (accumulate=False) overwrites it.
        bars.mb_dv_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_dv_acc_empty[dv_empty_state.idx].wait(dv_empty_state.phase, spin=SPIN_RING_WAITS)
        dv_empty_state = advance(dv_empty_state, 1)

        # End of block: release K + V (after every MMA that reads them has committed in order).
        bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        k_full_state = advance(k_full_state, CFG.STAGES_KV)
        v_full_state = advance(v_full_state, CFG.STAGES_KV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, _head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1, seq_kv_lens_tensor, n_batch, n_qh_grid)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: the pre-armed LEADER-scope rings leave the LAST tiles' batches (L_CNT arrives each, half of them DSMEM from the
    # follower) un-waited; drain STAGES_TMEM_S of each (the first drain wait falls through on an untouched slot when N = 1)
    # so this CTA is resident when they land.
    for _s in cutlass.range_constexpr(CFG.STAGES_TMEM_S):
        for s in cutlass.range_constexpr(KV_SUBBLOCKS):
            bars.mb_s_acc_empty[s * CFG.STAGES_TMEM_S + s_acc_empty_state.idx].wait(s_acc_empty_state.phase)
            bars.mb_dp_empty[s * CFG.STAGES_TMEM_S + dp_empty_state.idx].wait(dp_empty_state.phase)
        s_acc_empty_state = advance(s_acc_empty_state, CFG.STAGES_TMEM_S)
        dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)

    # ---- TMEM dealloc (the wait IS the drain of both CTAs' compute-lane arrives) ----
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    """Follower CTA's MMA warp: collective tmem_alloc (the leader's cga2 MMAs write this CTA's TMEM half), publish the
    pointer, wait mb_tmem_dealloc, release TMEM.  No persistent loop, no credit arrive (READ_TILE_ARRIVERS_TOT counts it out)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _compute_warp(
    sdK_raw,
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sP_raw,
    sdS_raw,
    sStats_raw,
    sdV_raw,
    seq_kv_lens_tensor,
    n_batch,
    n_qh_grid,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    attn_scale,
    attn_scale_log2e,
    batch_base,
    cta_in_pair,
    leader_cta_id,
    partner_cta_id,
) -> None:
    """8 compute warps.  Warp w: lane quadrant ``qd = w & 3`` -> TMEM lanes 32qd..+31 = kv row ``r = 32*(qd&1) + t`` of
    the 64-row sub-block tile and q-column half ``h = qd >> 1`` (the 2x2 D atom: lane r + 64h holds row r, columns
    [64h, 64h+64)); warpgroup ``g = w >> 2`` -> sub-block ``s = g // SUBBLOCK_WGS``, column sub-chunk ``cs = g %
    SUBBLOCK_WGS`` (COLS_PER_LANE columns from q 64h + COLS_PER_LANE*cs).  Warps w and w+2 hold the SAME kv rows at
    different q columns: no exchange is needed, every op is pointwise given the per-COLUMN lse / do_dot.

    Per q tile:  softmax  -- wait s_full; ld S; LOAD-wait; arrive s_acc_empty; P = exp2(scale*log2e*S - lse*log2e); mask;
                 bf16 -> wait p_empty -> store_swizzled into sP -> fence_proxy -> arrive p_full (.release.cta).
                 dSoftmax -- wait dp_full; ld dP; LOAD-wait; arrive dp_empty; dS = (dP*scale - do_dot) * P -> sdS ring
                 (store_swizzled) -> fence -> arrive ds_smem_full + stats_empty.
    Per kv block: dV epilogue -- wait dv_ready; ld dV (x64 per load, DV_LOADS_PER_LANE of them); LOAD-wait; OUT dtype ->
                 sdV subtile (2h + cs*DV_LOADS_PER_LANE + b), row 64s + r; fence; arrive dv_stg_full; arrive dv_acc_empty.
    A tcgen05_wait(LOAD) precedes every arrive that frees a slot.
    """
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    kv_super_idx, _head_idx, batch_idx = _boot_tile(sched, seq_kv_lens_tensor, n_batch, n_qh_grid)

    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    qd = warp_idx & cutlass.Int32(3)  # lane quadrant
    wg = warp_idx >> cutlass.Int32(2)  # warpgroup
    s_sub = wg // cutlass.Int32(SUBBLOCK_WGS)  # this warp's sub-block (0 on profile 1; g on profile 2)
    cs = wg % cutlass.Int32(SUBBLOCK_WGS)  # this warp's q column sub-chunk (g on profile 1; 0 on profile 2)
    kv_row = (qd & cutlass.Int32(1)) * cutlass.Int32(32) + lane  # kv row within the 64-row sub-block tile (= TMEM lane & 63)
    q_half = qd >> cutlass.Int32(1)  # q column half (= TMEM lane >> 6)
    q_col_local = q_half * cutlass.Int32(CFG.TILE_N // 2) + cs * cutlass.Int32(COLS_PER_LANE)  # first q col of this lane in the tile
    sub_base = s_sub * cutlass.Int32(LAYOUT.SUB_STRIDE)  # this sub-block's TMEM column base
    s_bar0 = s_sub * cutlass.Int32(CFG.STAGES_TMEM_S)  # this sub-block's first S / dP bar
    p_bar0 = s_sub * cutlass.Int32(CFG.STAGES_SMEM_P)  # this sub-block's first P bar
    # per-lane absolute kv row base (this CTA's rows of the pair's block, then this sub-block, then the lane's row)
    kv_lane_base0 = cta_in_pair * cutlass.Int32(ROWS_PER_CTA) + s_sub * cutlass.Int32(SUB_ROWS) + kv_row
    # this lane's row inside the CTA's ROWS_PER_CTA-row dS / dV staging tiles
    stg_row = s_sub * cutlass.Int32(SUB_ROWS) + kv_row
    # this lane's P segment inside a P slab: subtile q_half (64 q cols = 8 KiB), row kv_row (128 B), sub-chunk cs
    p_seg_off = q_half * cutlass.Int32(P_SUB_SLAB) + kv_row * cutlass.Int32(TMA_QK_GRANU_ELEMS) + cs * cutlass.Int32(COLS_PER_LANE)
    # this lane's dS segment inside a dS ring stage: subtile q_half (ROWS_PER_CTA x 64 q), row stg_row, sub-chunk cs
    ds_seg_off = q_half * cutlass.Int32(DS_BLOCK_SLAB) + stg_row * cutlass.Int32(P_D_BLOCK) + cs * cutlass.Int32(COLS_PER_LANE)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()
    dp_full_state = PipelineState.start()
    p_empty_state = PipelineState.start(phase=1)  # P ring slots free (pre-armed)
    ds_empty_state = PipelineState.start(phase=1)  # dS ring slot free (pre-armed)
    dv_ready_state = PipelineState.start()
    stats_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        tmem_base = tmem_ptr_i32.load()

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        ctx = _seq_ctx(seq_kv_lens_tensor, batch_idx, n_batch, seqlen_q_real, seqlen_kv_real)
        eff_seqlen_q, eff_seqlen_kv = _eff_lengths(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_q_real, seqlen_kv_real, ctx)
        causal_diag = _causal_diag(eff_seqlen_q, eff_seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, eff_seqlen_q, eff_seqlen_kv)
        kv_abs = kv_block_base + kv_lane_base0

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_abs = q_iter * cutlass.Int32(CFG.TILE_N) + q_col_local
            # ---- stats ring: this q tile's lse / do_dot (prefetched by the scheduler warp) ----
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase, spin=SPIN_RING_WAITS)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS) + q_col_local

            # ---- 1) softmax: S slot -> P ----
            s_slot = s_full_state.idx
            bars.mb_s_full[s_bar0 + s_slot].wait(s_full_state.phase, spin=SPIN_RING_WAITS)
            reg_S = tmem_load_tile(
                tmem_base + cutlass.Int32(LAYOUT.S_BASE) + sub_base + s_slot * cutlass.Int32(LAYOUT.S_COLS) + cs * cutlass.Int32(COLS_PER_LANE),
                num_elems=COLS_PER_LANE,
                ld_num=COLS_PER_LANE,
            )
            # The LOAD wait ORDERS the arrive after the TMEM read (an arrive with no data dependency on the loaded
            # registers may otherwise be scheduled between two LDTM chunks).
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.mb_s_acc_empty[s_bar0 + s_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            s_full_state = advance(s_full_state, CFG.STAGES_TMEM_S)
            lse_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw.subview(stats_base + cutlass.Int32(STATS_LSE_OFF + i)).load() for i in range(COLS_PER_LANE)),
                cutlass.Float32,
            )
            chunk_P = cute.math.exp2(reg_S.vec * attn_scale_log2e - lse_vec, fastmath=True)
            # Mask (the transpose of the forward): zero P on masked (kv = lane, q = col) cells so dV AND dS inherit it.
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                chunk_P = _mask_p_chunk(chunk_P, kv_abs, q_col_abs, eff_seqlen_kv, causal_diag, COLS_PER_LANE, eff_seqlen_q)

            # ---- 2) P -> sP[s][pslot] (the SS A operand of BMM2); publish to the leader's MMA with the .release.cta arrive ----
            chunk_P_half = chunk_P.to(STORAGE_DTYPE)
            p_slot = p_empty_state.idx
            bars.mb_p_empty[p_bar0 + p_slot].wait(p_empty_state.phase, spin=SPIN_RING_WAITS)
            p_empty_state = advance(p_empty_state, CFG.STAGES_SMEM_P)
            sP_raw.subview((p_bar0 + p_slot) * cutlass.Int32(pSlabElems) + p_seg_off).data_ptr().store_swizzled(
                chunk_P_half, alignment=P_STORE_ALIGN, swizzle=P_SMEM_SWIZZLE
            )
            nvvm.fence_proxy("async.shared", space="cta")  # generic SMEM stores -> the async-proxy MMA read (per lane, before the arrive)
            bars.mb_p_full[p_bar0 + p_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # ---- 3) dSoftmax: dP -> free the slot -> dS = (scale*dP - do_dot) * P -> sdS ring ----
            dp_slot = dp_full_state.idx
            bars.mb_dp_full[s_bar0 + dp_slot].wait(dp_full_state.phase, spin=SPIN_RING_WAITS)
            reg_dP = tmem_load_tile(
                tmem_base + cutlass.Int32(LAYOUT.dP_BASE) + sub_base + dp_slot * cutlass.Int32(LAYOUT.S_COLS) + cs * cutlass.Int32(COLS_PER_LANE),
                num_elems=COLS_PER_LANE,
                ld_num=COLS_PER_LANE,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.mb_dp_empty[s_bar0 + dp_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            dp_full_state = advance(dp_full_state, CFG.STAGES_TMEM_S)
            dot_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw.subview(stats_base + cutlass.Int32(STATS_DOT_OFF + i)).load() for i in range(COLS_PER_LANE)),
                cutlass.Float32,
            )
            chunk_dS = (reg_dP.vec * attn_scale - dot_vec) * chunk_P
            chunk_dS_half = chunk_dS.to(DS_STORAGE_DTYPE)
            ds_slot = ds_empty_state.idx
            bars.mb_ds_smem_empty[ds_slot].wait(ds_empty_state.phase, spin=SPIN_RING_WAITS)
            ds_empty_state = advance(ds_empty_state, CFG.XFER_STAGES)
            sdS_raw.subview(ds_slot * cutlass.Int32(dSBufferElems) + ds_seg_off).data_ptr().store_swizzled(
                chunk_dS_half, alignment=P_STORE_ALIGN, swizzle=P_SMEM_SWIZZLE
            )
            nvvm.fence_proxy("async.shared", space="cta")  # generic SMEM stores -> async-proxy TMA store
            bars.mb_ds_smem_full[ds_slot].arrive()
            if cutlass.const_expr(CFG.FUSE_DK):
                # dS is read by both TMA-STG and the peer-issued dK MMA. Its store
                # warp returns the compute credit only after both consumers finish.
                bars.mb_ds_mma_full[ds_slot].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, CFG.STATS_STAGES)

        # ---- dV epilogue (per kv block): dV TMEM -> OUT dtype -> sdV (aliases sK) ----
        # Lane half q_half holds d_v [128*q_half, +128); load b covers d_v cols 128*q_half + 64*(cs*DV_LOADS_PER_LANE + b)
        # + [0, 64) = the 128-B-row store subtile gblk = 2*q_half + cs*DV_LOADS_PER_LANE + b, at row stg_row.
        bars.mb_dv_ready.wait(dv_ready_state.phase, spin=SPIN_RING_WAITS)
        dv_ready_state = advance(dv_ready_state, 1)
        # N128 occupies one 64-column half per lane quadrant. The other
        # compute group skips its dV load/store but retains every arrival.
        if cutlass.const_expr(CFG.TILE_O == 256) or cs == cutlass.Int32(0):
            for _b in cutlass.range_constexpr(DV_LOADS_PER_LANE):
                dcol = cs * cutlass.Int32(DV_LOADS_PER_LANE * DV_LOAD_COLS) + cutlass.Int32(_b * DV_LOAD_COLS)
                reg_dV = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dV_BASE) + sub_base + dcol, num_elems=DV_LOAD_COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                dV_out = reg_dV.vec.to(OUT_STORAGE_DTYPE)
                gblk = q_half * cutlass.Int32(CFG.TILE_O // 128) + cs * cutlass.Int32(DV_LOADS_PER_LANE) + cutlass.Int32(_b)
                sdV_raw.subview(gblk * cutlass.Int32(DV_BLOCK_SLAB) + stg_row * cutlass.Int32(DV_D_BLOCK)).data_ptr().store_swizzled(
                    dV_out, alignment=128, swizzle=P_SMEM_SWIZZLE
                )
        if cutlass.const_expr(CFG.FUSE_DK):
            # P's last MMA read precedes dv_ready; stage dK into that now-dead ring.
            dk_col = q_half * cutlass.Int32(96) + cs * cutlass.Int32(32)
            dk_reg = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dK_BASE) + cs * cutlass.Int32(32), num_elems=32, ld_num=32)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            dk_offset = (dk_col // cutlass.Int32(64)) * cutlass.Int32(64 * 64) + stg_row * cutlass.Int32(64) + dk_col % cutlass.Int32(64)
            sdK_raw.subview(dk_offset).data_ptr().store_swizzled(dk_reg.vec.to(OUT_STORAGE_DTYPE), alignment=64, swizzle=P_SMEM_SWIZZLE)
            if cs == cutlass.Int32(0):
                dk_tail_col = q_half * cutlass.Int32(96) + cutlass.Int32(64)
                dk_tail = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dK_BASE + 64), num_elems=32, ld_num=32)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                dk_tail_offset = (dk_tail_col // cutlass.Int32(64)) * cutlass.Int32(64 * 64) + stg_row * cutlass.Int32(64) + dk_tail_col % cutlass.Int32(64)
                sdK_raw.subview(dk_tail_offset).data_ptr().store_swizzled(dk_tail.vec.to(OUT_STORAGE_DTYPE), alignment=64, swizzle=P_SMEM_SWIZZLE)
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dv_stg_full.arrive()
        # dV read -> the next block's BMM2(q_lo) (accumulate=False) may overwrite it.
        bars.mb_dv_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, _head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1, seq_kv_lens_tensor, n_batch, n_qh_grid)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: the pre-armed mb_p_empty consumer leaves the LAST tiles' BMM2 commits (multicast onto this CTA's own bar)
    # un-waited; drain STAGES_SMEM_P so this CTA is resident when they land.
    for _s in cutlass.range_constexpr(CFG.STAGES_SMEM_P):
        bars.mb_p_empty[p_bar0 + p_empty_state.idx].wait(p_empty_state.phase)
        p_empty_state = advance(p_empty_state, CFG.STAGES_SMEM_P)

    # ---- release the MMA warps' TMEM alloc on BOTH CTAs (bare: every compute lane, local + peer) ----
    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


@cute.jit
def _tmastg_warp(
    dk_maps,
    tma_qdk_desc,
    sQdk,
    tma_dk_desc,
    sdK,
    tma_dv_desc,
    tma_ds_desc,
    sdV,
    sdS,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_batch,
    n_qh_grid,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    head_base,
    batch_base,
    cta_in_pair,
    dv_head_base,
    dv_batch_base,
    physical_origins: cutlass.Constexpr = False,
) -> None:
    """TMA-store warp (both CTAs): per q tile the dS slot -> workspace [B_chunk, H_chunk, S_kv, S_q] (one contiguous
    ROWS_PER_CTA-row box at kv_block_base + cta_in_pair * ROWS_PER_CTA); per kv block dV -> [B, S_kv, H_q, d_v]."""
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_ds = GmemTileTma(tma_ds_desc)
    if cutlass.const_expr(CFG.FUSE_DK):
        tma_dk = GmemTileTma(tma_dk_desc)
        qdk_empty_state = PipelineState.start(phase=1)
        is_leader = cta_in_pair == cutlass.Int32(0)
        tma_mcast_mask = cutlass.Int16(1) << cta_in_pair
    if cutlass.const_expr(_THD):
        for sequence in cutlass.range(0, n_batch, 1, unroll=1):
            if cutlass.const_expr(CFG.FUSE_DK):
                tma_tensormap_acquire(_dk_map_ptr(dk_maps, n_batch + cutlass.Int32(1) + sequence))
            tma_tensormap_acquire(
                _thd_map_ptr(
                    seq_kv_lens_tensor,
                    n_batch,
                    (cutlass.Int32(THD_INPUT_SLOTS) * (n_batch + 1) if cutlass.const_expr(physical_origins) else cutlass.Int32(THD_DV_SLOT_BASE)) + sequence,
                )
            )

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched, seq_kv_lens_tensor, n_batch, n_qh_grid)
    full_batch = cute.arch.make_warp_uniform(batch_idx + dv_batch_base)
    KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(ROWS_PER_CTA)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    ds_full_state = PipelineState.start()
    if cutlass.const_expr(CFG.FUSE_DK):
        ds_mma_done_state = PipelineState.start()
    dv_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        ctx = _seq_ctx(seq_kv_lens_tensor, batch_idx, n_batch, seqlen_q_real, seqlen_kv_real, physical_origins)
        eff_seqlen_q, eff_seqlen_kv = _eff_lengths(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_q_real, seqlen_kv_real, ctx)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, eff_seqlen_q, eff_seqlen_kv)
        ws_row = ctx[2]
        batch_ws = cutlass.Int32(0) if cutlass.const_expr(_THD) else batch_idx
        if cutlass.const_expr(CFG.FUSE_DK):
            q_seq_off, _kv_seq_off, qdk_batch = _thd_tma_offsets(ctx, batch_idx + batch_base)
            if cutlass.const_expr(_THD):
                # Qdk belongs to this store warp. Acquire its current descriptor
                # on each CTA before the q loop, including the dead-unit map.
                qdk_slot = cutlass.Int32(0)
                if cutlass.const_expr(physical_origins):
                    qdk_slot = batch_idx + cutlass.Int32(1) if batch_idx < n_batch else cutlass.Int32(0)
                tma_tensormap_acquire(_dk_map_ptr(dk_maps, qdk_slot))

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            if cutlass.const_expr(CFG.FUSE_DK):
                # Qdk loads share this warp with the dS store, freeing the Q/dO prefetch warp.
                bars.mb_qdk_empty.wait(qdk_empty_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_qdk_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                tma_load_tile(
                    sQdk[0],
                    _qdk_slice(
                        tma_qdk_desc,
                        dk_maps,
                        n_batch,
                        batch_idx,
                        physical_origins,
                        cta_in_pair * cutlass.Int32(96),
                        head_idx + head_base,
                        q_seq_off + q_col_base,
                        qdk_batch,
                    ),
                    bars.mb_qdk_full.smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                    acquire=not _THD,
                )
                qdk_empty_state = advance(qdk_empty_state, 1)
            ds_slot = ds_full_state.idx
            bars.mb_ds_smem_full[ds_slot].wait(ds_full_state.phase, spin=SPIN_RING_WAITS)
            ds_full_state = advance(ds_full_state, CFG.XFER_STAGES)
            # workspace [B_chunk, H_chunk, S_kv, S_q] -> coords innermost-first (S_q, S_kv, H, B), chunk-local head / batch
            if cutlass.const_expr(_THD):
                if batch_idx < n_batch:
                    tma_store_tile(sdS[ds_slot], tma_ds(q_col_base, ws_row + kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_ws))
            else:
                tma_store_tile(sdS[ds_slot], tma_ds(q_col_base, ws_row + kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_ws))
            tma_store_commit()
            tma_store_wait(0)
            if cutlass.const_expr(CFG.FUSE_DK):
                bars.mb_ds_mma_empty[ds_slot].wait(ds_mma_done_state.phase, spin=SPIN_RING_WAITS)
                ds_mma_done_state = advance(ds_mma_done_state, CFG.XFER_STAGES)
            if nvvm.elect_sync():
                bars.mb_ds_smem_empty[ds_slot].arrive()

        # dV uses independent output origins: full tensor or current chunk workspace.
        bars.mb_dv_stg_full.wait(dv_full_state.phase, spin=SPIN_RING_WAITS)
        dv_full_state = advance(dv_full_state, 1)
        if cutlass.const_expr(_THD):
            if batch_idx < n_batch:
                tma_store_tile(
                    sdV[0],
                    tma_slice_runtime_desc(
                        _thd_map_ptr(
                            seq_kv_lens_tensor,
                            n_batch,
                            (cutlass.Int32(THD_INPUT_SLOTS) * (n_batch + 1) if cutlass.const_expr(physical_origins) else cutlass.Int32(THD_DV_SLOT_BASE))
                            + batch_idx,
                        ),
                        cutlass.Int32(0),
                        head_idx + dv_head_base,
                        kv_block_base + KV_ROW_OFFSET_PEER,
                        cutlass.Int32(0),
                    ),
                    acquire=False,
                )
        else:
            tma_store_tile(sdV[0], tma_dv(cutlass.Int32(0), head_idx + dv_head_base, kv_block_base + KV_ROW_OFFSET_PEER, full_batch))
        if cutlass.const_expr(CFG.FUSE_DK):
            if cutlass.const_expr(_THD):
                if batch_idx < n_batch:
                    tma_store_tile(
                        sdK[0],
                        tma_slice_runtime_desc(
                            _dk_map_ptr(dk_maps, n_batch + cutlass.Int32(1) + batch_idx),
                            cutlass.Int32(0),
                            head_idx + dv_head_base,
                            kv_block_base + KV_ROW_OFFSET_PEER,
                            cutlass.Int32(0),
                        ),
                        acquire=False,
                    )
            else:
                tma_store_tile(sdK[0], tma_dk(cutlass.Int32(0), head_idx + dv_head_base, kv_block_base + KV_ROW_OFFSET_PEER, full_batch))
        tma_store_commit()
        tma_store_wait(0)  # source SMEM reads done -> the TMA-LDG may reload K over sdV
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1, seq_kv_lens_tensor, n_batch, n_qh_grid)
        full_batch = cute.arch.make_warp_uniform(batch_idx + dv_batch_base)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    if cutlass.const_expr(CFG.FUSE_DK):
        # Drain the final cross-CTA Qdk release before this consumer exits.
        bars.mb_qdk_empty.wait(qdk_empty_state.phase)


@cute.jit
def _scheduler_stats_warp(
    sched,
    is_cga_first_cta,
    bars,
    sStats_raw,
    lse_tensor,
    do_dot_tensor,
    seq_kv_lens_tensor,
    attn_scale,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    n_batch,
    n_qh_grid,
    head_base,
    batch_base,
    thd_origins=None,
    stats_origin_row: cutlass.Constexpr = -1,
) -> None:
    """Persistent tile scheduler (CLC try_cancel protocol) FUSED with the lse / do_dot prefetch.

    Per loop iteration: (A) fill the sStats ring for the tile the consumers are CURRENTLY
    processing (one slot per q tile: lse * log2e | do_dot * attn_scale, TILE_N each; the
    consumers read the SAME slot broadcast on every lane), tracked one tile behind the
    scheduling; (B) the canonical CLC protocol for the NEXT tile (tile_dsl.scheduler
    .scheduler_warp_loop): the cluster's first CTA arms expect_tx on EVERY CTA's mb_scheduler,
    program-ordered before its multicast try_cancel.  This warp never credits
    mb_read_tile_id (it is the waiter), so READ_TILE_ARRIVERS_TOT excludes it.

    THD: (A) reads the PACKED Stats / delta at ``cu_q[b] + q_col`` for ``q_col < s_q[b]`` and
    substitutes ``(+inf, 0)`` past the sequence (the dense adapter's +inf-LSE pad, moved
    in-kernel: ``P = exp2(.. - inf) = 0`` on the pad columns; the q band of ``_mask_p_chunk`` is the
    belt to this brace), the GMEM index SELECTED to row 0 where invalid so a dead unit's
    ``cu_q[B] + col`` -- one past a buffer whose capacity equals the live total -- is never
    dereferenced.  Stats is the caller's in either packing the forward emits, selected on the
    fake tensor's STATIC rank: token-major ``(T, H)`` or head-major ``(1, QH, head_stride)``; delta
    is ours, head-major ``(1, QH, ceil128(T))``.  (B) becomes the device CLAIM protocol
    (``tile_dsl.scheduler.scheduler_warp_loop_persistent``'s body on this warp's loop): EACH
    CTA's elected lane arms its OWN ``mb_scheduler`` with ``expect_tx(16)`` (1 lane -> init ONE_LANE,
    unchanged), the cluster-first CTA's elected lane claims ``uid = atomicAdd(ctr, 1)``, compares it
    with the device live total and ships the 4-word payload ``(uid * CGA_M, 0, valid, 0)`` into every
    CTA's ``tile_id_smem`` slot by DSMEM stores that complete each target's own mbarrier's 16
    transaction bytes -- the two waits and the payload decode are the SAME sites as the CLC arm,
    so every ring count and ``_RING_WAIT_SITES`` stay as they are.  No CLC try_cancel under THD:
    the grid is occupancy-sized and the work list is a device value.
    """
    lse_arr = cutlass.make_array_view(lse_tensor)
    dot_arr = cutlass.make_array_view(do_dot_tensor)
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    _PER_LANE = CFG.TILE_N // 32
    log2e = cutlass.Float32(_LOG2E)

    cur_kv_super, cur_head, cur_batch = _boot_tile(sched, seq_kv_lens_tensor, n_batch, n_qh_grid)
    ctx = _seq_ctx(seq_kv_lens_tensor, cur_batch, n_batch, seqlen_q_real, seqlen_kv_real)
    if cutlass.const_expr(_THD):
        _ctr_ptr = Pointer(seq_kv_lens_tensor.iterator.raw_ptr(), dtype=cutlass.Int32) + (cutlass.Int32(4) * n_batch + cutlass.Int32(3))
        _live_off = cutlass.Int32(4) * n_batch + cutlass.Int32(2)
        _meta = cutlass.make_array_view(seq_kv_lens_tensor)

    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        cur_full_batch = cute.arch.make_warp_uniform(cur_batch + batch_base)
        kv_block_base = cur_kv_super * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_q, eff_seqlen_kv = _eff_lengths(seq_kv_lens_tensor, cur_batch, batch_base, seqlen_q_real, seqlen_kv_real, ctx)
        # The SAME [q_lo, q_hi) the consumers walk, so the ring fill count matches their waits.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, eff_seqlen_q, eff_seqlen_kv)

        if cutlass.const_expr(thd_origins is not None):
            stats_base = thd_port_origin(thd_origins, stats_origin_row, cute.math.min(cur_batch, n_batch - 1), ctx[0])

        # ---- (A) stats prefetch for the CURRENT tile ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase, spin=SPIN_RING_WAITS)
            stats_empty_state = advance(stats_empty_state, CFG.STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                if cutlass.const_expr(_THD):
                    col_abs = q_col_base + col
                    valid = col_abs < eff_seqlen_q
                    row_pk = cutlass.Int32(arith.select(valid.ir_value(), (ctx[0] + col_abs).ir_value(), cutlass.Int32(0).ir_value()))
                    lse_row = cutlass.Int64(row_pk)
                    if cutlass.const_expr(thd_origins is not None):
                        lse_row = stats_base + cutlass.Int64(col_abs) if valid else cutlass.Int64(0)
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        lse_v = lse_arr[lse_row, cur_full_head]
                    else:
                        lse_v = lse_arr[cutlass.Int32(0), cur_full_head, lse_row]
                    dot_v = dot_arr[cutlass.Int32(0), cur_full_head, row_pk]
                    lse_s = cutlass.Float32(arith.select(valid.ir_value(), (lse_v * log2e).ir_value(), cutlass.Float32(float("inf")).ir_value()))
                    dot_s = cutlass.Float32(arith.select(valid.ir_value(), (dot_v * attn_scale).ir_value(), cutlass.Float32(0.0).ir_value()))
                    sStats_raw.subview(slot_base + cutlass.Int32(STATS_LSE_OFF) + col).store(lse_s)
                    sStats_raw.subview(slot_base + cutlass.Int32(STATS_DOT_OFF) + col).store(dot_s)
                else:
                    sStats_raw.subview(slot_base + cutlass.Int32(STATS_LSE_OFF) + col).store(lse_arr[cur_full_batch, cur_full_head, q_col_base + col] * log2e)
                    sStats_raw.subview(slot_base + cutlass.Int32(STATS_DOT_OFF) + col).store(
                        dot_arr[cur_full_batch, cur_full_head, q_col_base + col] * attn_scale
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        # ---- (B) the NEXT tile: the CLC protocol (dense) or the device claim counter (THD) ----
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if cutlass.const_expr(_THD):
            # Every CTA expects the 16-byte payload on its OWN mbarrier (one elected lane -> init ONE_LANE); the cluster-first CTA's
            # DSMEM stores deliver it and complete the barrier through the transaction count (a complete_tx that lands
            # before the expect_tx only parks the count negative within the phase).
            if nvvm.elect_sync():
                arrive_expect_tx(sched.mb_scheduler.subview(state.idx), _CLC_RESPONSE_BYTES)
            if nvvm.elect_sync() and is_cga_first_cta:
                uid = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, _ctr_ptr, cutlass.Int32(1)))
                live = cutlass.Int32(_meta[_live_off])
                valid_u = cutlass.Int32(arith.select((uid < live).ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
                # Stride by CGA_M: the consumers decode the unit back out as t0 // CGA_M (_decode_tile_words).
                linear = uid * cutlass.Int32(CFG.CGA_M)
                _tile_ptr = cute.make_ptr(
                    cutlass.Int32,
                    sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)).data_ptr().toint(cutlass.Int32),
                    cutlass.AddressSpace.smem,
                    assumed_align=16,
                )
                _mbar_ptr = cute.make_ptr(
                    cutlass.Int64,
                    sched.mb_scheduler.subview(state.idx).data_ptr().toint(cutlass.Int32),
                    cutlass.AddressSpace.smem,
                    assumed_align=8,
                )
                # Four scalar DSMEM stores per target CTA = the 16 bytes its expect_tx counts (store_async_dsmem takes one word).
                _payload = (linear, cutlass.Int32(0), valid_u, cutlass.Int32(0))
                for i in cutlass.range_constexpr(CGA_SIZE):
                    for w in cutlass.range_constexpr(4):
                        cute.arch.store_async_dsmem(_tile_ptr + w, _payload[w], _mbar_ptr, i)
        else:
            if nvvm.elect_sync() and is_cga_first_cta:
                for i in cutlass.range_constexpr(CGA_SIZE):
                    if cutlass.const_expr(i == 0):
                        arrive_expect_tx(sched.mb_scheduler.subview(state.idx), _CLC_RESPONSE_BYTES)
                    else:
                        peer_mb = nvvm.mapa(sched.mb_scheduler.subview(state.idx), cutlass.Int32(i))
                        # Program-ordered before the multicast try_cancel by the SAME thread; CTA scope
                        # (a cluster-scope release is a GPU-scope drain per arrive).
                        nvvm.mbarrier_arrive_expect_tx(peer_mb, _CLC_RESPONSE_BYTES, scope=nvvm.MemScope.CTA)
                nvvm.clusterlaunchcontrol_try_cancel(
                    sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                    sched.mb_scheduler.subview(state.idx),
                    multicast=1,
                )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, state.idx * cutlass.Int32(8))
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        cur_kv_super, cur_head, cur_batch = _decode_tile_words(sched, nxt_t0, nxt_t1, seq_kv_lens_tensor, n_batch, n_qh_grid)
        ctx = _seq_ctx(seq_kv_lens_tensor, cur_batch, n_batch, seqlen_q_real, seqlen_kv_real)
        state = advance(state, CFG.SCHEDULER_STAGES)


# ============================================================================
# Host launcher
# ============================================================================


def _tma_swz(byte_w: int):
    return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b


@cute.kernel
def _thd_setup_kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_do_dv_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],
    maps_t: cute.Tensor,  # the (THD_INPUT_SLOTS + B) x 16 int64 tensor-map array inside the metadata buffer
    meta_t: cute.Tensor,  # the metadata buffer (int32 words)
    dv_tensor: cute.Tensor,  # the packed dV the per-sequence descriptors are based on
    n_batch: cutlass.Int32,
    n_qh: cutlass.Int32,  # the LAUNCH's head count (QH_CHUNK): the unit total is per launch
    n_clusters: cutlass.Int32,
    dv_row_stride: cutlass.Int64,  # dV's token stride in ELEMENTS
    input_tensors,
    thd_origins,
    thd_origin_rows: cutlass.Constexpr,
    tma_qdk_desc: Optional[cutlass.GridConstant[tmap.TensorMap]],
    tma_dk_desc: Optional[cutlass.GridConstant[tmap.TensorMap]],
    dk_tensor,
    dk_maps,
) -> None:
    """This kernel's OWN THD descriptors and per-launch scheduler words, from the metadata the chain's setup launch wrote.

    * warp 0's elected lane: the five input descriptors copied with their sequence extent CLAMPED to the current packed
      total ``cu_q[B]`` / ``cu_k[B]`` (at least 1: a zero extent is INVALID and traps; the one row an all-empty side then
      keeps addressable is never addressed -- ``_seq_ctx`` routes every load of a unit without query rows to the extent
      itself, out of bounds, zero-filled) -- a declared ``max_total_seq_len`` cannot do this job, it is a maximum
      while the row that must read as zero is the current total;
    * every warp's elected lane: one CLIPPED dV descriptor per sequence (base ``cu_k[b]``, extent ``max(s_kv[b], 1)``), the
      sequences shared round-robin over the warps; each writer publishes to the TMA proxy;
    * the whole block: ``live`` = this launch's write pairs times ``n_qh * _THD_BLOCKS_PER_WRITE`` and the claim counter
      reset to the cluster count.  Per LAUNCH, not per execute: a head-chunked THD chain would otherwise claim past its
      live total on the second chunk and hand out dead units on the first.
    The kernel boundary orders all of it before the main launch reads it.
    """
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    warp = cutlass.Int32(tidx) // cutlass.Int32(32)
    meta = cutlass.make_array_view(meta_t)
    cu_k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    if nvvm.elect_sync():
        if warp == cutlass.Int32(0):
            t_q = cute.math.max(cutlass.Int32(meta[cutlass.Int32(2) * n_batch]), cutlass.Int32(1))  # cu_q[B]
            t_kv = cute.math.max(cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)]), cutlass.Int32(1))  # cu_k[B]
            emit_clamped_desc(tma_q_desc, maps_t, cutlass.Int32(THD_Q_SLOT), t_q, seq_ord=_THD_SEQ_ORD)
            if cutlass.const_expr(CFG.FUSE_DK):
                emit_clamped_desc(
                    tma_qdk_desc, dk_maps, cutlass.Int32(0), cutlass.Int32(1) if cutlass.const_expr(thd_origins is not None) else t_q, seq_ord=_THD_SEQ_ORD
                )
            emit_clamped_desc(tma_do_desc, maps_t, cutlass.Int32(THD_DO_SLOT), t_q, seq_ord=_THD_SEQ_ORD)
            emit_clamped_desc(tma_do_dv_desc, maps_t, cutlass.Int32(THD_DODV_SLOT), t_q, seq_ord=_THD_SEQ_ORD)
            emit_clamped_desc(tma_k_desc, maps_t, cutlass.Int32(THD_K_SLOT), t_kv, seq_ord=_THD_SEQ_ORD)
            emit_clamped_desc(tma_v_desc, maps_t, cutlass.Int32(THD_V_SLOT), t_kv, seq_ord=_THD_SEQ_ORD)
        if cutlass.const_expr(thd_origins is not None):
            input_descs = (tma_q_desc, tma_do_desc, tma_do_dv_desc, tma_k_desc, tma_v_desc)
            port_rows = (thd_origin_rows[0], thd_origin_rows[4], thd_origin_rows[4], thd_origin_rows[1], thd_origin_rows[2])
            for port in cutlass.range_constexpr(THD_INPUT_SLOTS):
                # The first five maps are empty/dead-unit fallbacks. Every live
                # sequence gets a separately clipped map for each input port.
                if warp == cutlass.Int32(0):
                    emit_clamped_desc(input_descs[port], maps_t, cutlass.Int32(port), cutlass.Int32(1), seq_ord=_THD_SEQ_ORD)
                emit_seq_descs(
                    input_descs[port],
                    maps_t,
                    meta,
                    n_batch if cutlass.const_expr(port < 3) else cu_k0,
                    input_tensors[port],
                    n_batch,
                    cutlass.Int64(input_tensors[port].stride[1]),
                    seq_ord=_THD_SEQ_ORD,
                    slot_base=THD_INPUT_SLOTS + port * n_batch,
                    first_batch=warp,
                    batch_step=cutlass.Int32(nthreads) // 32,
                    origins=thd_origins,
                    origin_row=port_rows[port],
                )
        emit_seq_descs(
            tma_dv_desc,
            maps_t,
            meta,
            cu_k0,
            dv_tensor,
            n_batch,
            dv_row_stride,
            seq_ord=_THD_SEQ_ORD,
            slot_base=THD_INPUT_SLOTS * (n_batch + 1) if cutlass.const_expr(thd_origins is not None) else THD_DV_SLOT_BASE,
            origins=thd_origins,
            origin_row=thd_origin_rows[7],
            first_batch=warp,
            batch_step=cutlass.Int32(nthreads) // 32,
        )
        if cutlass.const_expr(CFG.FUSE_DK):
            if cutlass.const_expr(thd_origins is not None):
                emit_seq_descs(
                    tma_qdk_desc,
                    dk_maps,
                    meta,
                    n_batch,
                    input_tensors[0],
                    n_batch,
                    cutlass.Int64(input_tensors[0].stride[1]),
                    seq_ord=_THD_SEQ_ORD,
                    slot_base=1,
                    first_batch=warp,
                    batch_step=cutlass.Int32(nthreads) // 32,
                    origins=thd_origins,
                    origin_row=thd_origin_rows[0],
                )
            emit_seq_descs(
                tma_dk_desc,
                dk_maps,
                meta,
                cu_k0,
                dk_tensor,
                n_batch,
                cutlass.Int64(dk_tensor.stride[1]),
                seq_ord=_THD_SEQ_ORD,
                slot_base=n_batch + cutlass.Int32(1),
                first_batch=warp,
                batch_step=cutlass.Int32(nthreads) // 32,
                origins=thd_origins,
                origin_row=thd_origin_rows[6],
            )
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    cute.arch.barrier()
    write_thd_work_prefix(
        meta,
        n_batch,
        n_qh * cutlass.Int32(_THD_BLOCKS_PER_WRITE),
        cutlass.Int32(_KV_WRITE_ROWS),
        n_clusters,
        cutlass.Int32(tidx),
        1,
        cu0=cu_k0,
        prefix0=THD_BWD_WORK_PREFIX_OFF(n_batch),
    )


_thd_setup_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def _host_chain(
    q_tensor: cute.Tensor,  # [B, S_q, H_q, d]
    do_tensor: cute.Tensor,  # [B, S_q, H_q, d_v]
    k_tensor: cute.Tensor,  # [B, S_kv, H_kv, d]
    v_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_v]
    dv_tensor: cute.Tensor,  # out [B, S_kv, H_q, d_v] OUT dtype
    ds_tensor: cute.Tensor,  # out [B_chunk, H_chunk, S_kv, S_q] io dtype (workspace)
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32
    seq_kv_lens_tensor: cute.Tensor,  # [B] int32 (PADDED arm) or a 1-element dummy
    problem_size: tuple,  # (B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL)
    attn_scale: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    stream: _cuda_driver.CUstream = None,
    dv_head_base=None,
    dv_batch_base=None,
    thd_origins=None,
    thd_origin_rows: cutlass.Constexpr = (-1,) * 9,
    dk_tensor=None,
    dk_maps=None,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL = problem_size[:9]
    N_THD_UNITS = problem_size[9] if len(problem_size) > 9 else 0
    # Inputs retain their full-tensor origins; only the output partial may be local.
    if cutlass.const_expr(dv_head_base is None):
        dv_head_base = head_base
    if cutlass.const_expr(dv_batch_base is None):
        dv_batch_base = batch_base

    # [B, S, H, D] operands: box = (1 batch, S rows, 1 head, D cols); stride_order innermost-first so the kernel's coords
    # are (d, head, seq, batch).  Workspace [B, H, S_kv, S_q]: (q, kv, head, batch).
    stride_order = (3, 2, 1, 0)
    qk_box_q = (1, _M_PER_CTA, 1, TMA_QK_GRANU_ELEMS)  # Q, dO (dP view): per-CTA q half
    kv_box = (1, SUB_ROWS, 1, TMA_QK_GRANU_ELEMS)  # K, V: one 64-row sub-block slab
    do_box = (1, _M_PER_CTA, 1, TMA_VO_GRANU_ELEMS)
    v_box = (1, SUB_ROWS, 1, TMA_VO_GRANU_ELEMS)
    do_dv_box = (1, CFG.TILE_N, 1, TMA_VO_SG1_GRANU_ELEMS)  # dO dV view (BT): full q tile x one 64-d_v granule
    dv_box = (1, ROWS_PER_CTA, 1, DV_D_BLOCK)  # dV store: this CTA's contiguous rows x 64 d_v
    ds_box = (1, 1, ROWS_PER_CTA, P_D_BLOCK)  # dS store: this CTA's contiguous rows x 64 q

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
        k_tensor, box_dims=kv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.K_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor, box_dims=v_box, stride_order=stride_order, swizzle=_tma_swz(CFG.V_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(
        dv_tensor, box_dims=dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dV_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_tensor, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )

    tma_qdk_desc = None
    tma_dk_desc = None
    if cutlass.const_expr(CFG.FUSE_DK):
        tma_qdk_desc = tmap.create_tensor_map_tiled_from_view(
            q_tensor, box_dims=(1, 128, 1, 32), stride_order=stride_order, swizzle=_tma_swz(64), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
        tma_dk_desc = tmap.create_tensor_map_tiled_from_view(
            dk_tensor, box_dims=(1, 64, 1, 64), stride_order=stride_order, swizzle=_tma_swz(128), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
        )
    # One cga2 cluster per (kv block, head, batch) tile of this chunk (kv blocks of KV_BLOCK_ROWS rows).
    kv_blocks = (SKV + _KV_BLOCK_ROWS - 1) // _KV_BLOCK_ROWS
    if cutlass.const_expr(_THD):
        # The problem_size entries are RUNTIME values on the compile() path (and Python ints on the prepared host's), so the
        # cluster count is validated by the caller (`config_sm107.launch_grid` / the adapter), not raised on here.
        grid_shape = (N_THD_UNITS * CFG.CGA_M, 1, 1)
        # The tensor-map array inside the metadata buffer (tile_dsl.thd.THD_BWD_MAPS_OFF: the next 128-B boundary after the
        # 5B+5 words), as an int64 view the descriptor patchers write through.  Non-negative arithmetic only: a runtime
        # `//` truncates toward zero, so the Python negation trick for a ceil would floor instead.
        _maps_off_bytes = THD_BWD_MAPS_OFF(B) * 4
        _maps = cute.make_tensor(
            cute.make_ptr(cutlass.Int64, seq_kv_lens_tensor.iterator.toint() + cutlass.Int64(_maps_off_bytes), cute.AddressSpace.gmem, assumed_align=128),
            cute.make_layout(((THD_INPUT_SLOTS + (6 if cutlass.const_expr(thd_origins is not None) else 1) * B) * TENSOR_MAP_QWORDS,), stride=(1,)),
        )
        # Ahead of the main launch, on the same stream: kernel-boundary ordering makes the descriptors and the per-launch
        # live / ctr visible to it.
        _thd_setup_kernel(
            tma_q_desc,
            tma_do_desc,
            tma_do_dv_desc,
            tma_k_desc,
            tma_v_desc,
            tma_dv_desc,
            _maps,
            seq_kv_lens_tensor,
            dv_tensor,
            cutlass.Int32(B),
            cutlass.Int32(QH_CHUNK),
            cutlass.Int32(N_THD_UNITS),
            cutlass.Int64(dv_tensor.stride[1]),
            (q_tensor, do_tensor, do_tensor, k_tensor, v_tensor),
            thd_origins,
            thd_origin_rows,
            tma_qdk_desc,
            tma_dk_desc,
            dk_tensor,
            dk_maps,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
    elif cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        grid_shape = (kv_blocks * CFG.CGA_M, QH_CHUNK, B_CHUNK)
    else:
        grid_shape = (kv_blocks * QH_CHUNK * B_CHUNK * CFG.CGA_M, 1, 1)

    attn_scale_log2e = attn_scale * cutlass.Float32(_LOG2E)

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
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SQ_REAL),
        cutlass.Int32(SKV_REAL),
        cutlass.Int32(B) if _THD else cutlass.Int32(B_CHUNK),
        cutlass.Int32(QH_CHUNK),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        batch_base,
        dv_head_base,
        dv_batch_base,
        thd_origins,
        thd_origin_rows,
        tma_qdk_desc,
        tma_dk_desc,
        dk_maps,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CGA_M, CFG.CGA_N, 1),
        stream=stream,
    )


@cute.jit
def _host(
    q_tensor: cute.Tensor,  # [B, S_q, H_q, d]
    do_tensor: cute.Tensor,  # [B, S_q, H_q, d_v]
    k_tensor: cute.Tensor,  # [B, S_kv, H_kv, d]
    v_tensor: cute.Tensor,  # [B, S_kv, H_kv, d_v]
    dv_tensor: cute.Tensor,  # out [B, S_kv, H_q, d_v] OUT dtype
    ds_tensor: cute.Tensor,  # out [B_chunk, H_chunk, S_kv, S_q] io dtype (workspace)
    lse_tensor: cute.Tensor,  # [B, H_q, S_q] fp32
    do_dot_tensor: cute.Tensor,  # [B, H_q, S_q] fp32
    seq_kv_lens_tensor: cute.Tensor,  # [B] int32 (PADDED arm) or a 1-element dummy
    problem_size: tuple,  # (B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL)
    attn_scale: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    stream: _cuda_driver.CUstream = None,
    dv_head_base=None,
    dv_batch_base=None,
    dk_tensor=None,
    dk_maps=None,
) -> None:
    """Preserve the compact exported runtime frame."""
    _host_chain(
        q_tensor,
        do_tensor,
        k_tensor,
        v_tensor,
        dv_tensor,
        ds_tensor,
        lse_tensor,
        do_dot_tensor,
        seq_kv_lens_tensor,
        problem_size,
        attn_scale,
        head_base,
        batch_base,
        stream,
        dv_head_base,
        dv_batch_base,
        dk_tensor=dk_tensor,
        dk_maps=dk_maps,
    )


_host_origins = _host_chain


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 256,
    qh_chunk: int = 0,
    b_chunk: int = 0,
    sq_real: int = 0,
    skv_real: int = 0,
    thd_units: int = 0,
    sq_env: int = 0,
    lse_token_major: bool = False,
    origin_rows: tuple | None = None,
) -> Callable:
    """Compile the main kernel with ALL extents concrete (pins the TMA strides); see '## Launch ABI'.

    ``qh_chunk`` (default ``qh``) / ``b_chunk`` (default ``b``) size the dS workspace's head /
    batch extents and the grid; the runtime ``head_base`` / ``batch_base`` walk the chunks so
    one artifact serves every launch.  ``sq_real`` / ``skv_real`` (default = the padded
    extents) are the real lengths the bottom-right diagonal uses.

    THD (``CFG.THD_VARLEN``; the bring-up / SASS-probe entry -- the prepared chain bakes ``_host``
    itself): ``b`` is the SEQUENCE count, ``sq`` / ``skv`` the packed token CAPACITIES (any
    positive value, no tile rule), ``sq_env`` (default ``sq``) the q envelope the workspace
    columns are padded from, ``thd_units`` (default: the unit upper bound) the persistent grid's
    cluster count, ``lse_token_major`` the Stats packing (rank 2 ``(T, H)`` vs rank 3 ``(1, H, T)``).
    """
    _cache_key = _template_key(globals(), locals(), "compile")
    if qh_chunk == 0:
        qh_chunk = qh
    if b_chunk == 0:
        b_chunk = b
    if sq_real == 0:
        sq_real = sq
    if skv_real == 0:
        skv_real = skv
    if sq_env == 0:
        sq_env = sq
    if not _THD:
        if sq % CFG.TILE_N != 0:
            raise ValueError(f"bwd d256 2x2 f16: S_q must be a multiple of the q tile ({CFG.TILE_N}); got {sq} -- the adapter pads")
        if skv % _KV_WRITE_ROWS != 0:
            raise ValueError(f"bwd d256 2x2 f16: S_kv must be a multiple of the kv write pair ({_KV_WRITE_ROWS}); got {skv} -- the adapter pads")
        if not (0 < sq_real <= sq and 0 < skv_real <= skv):
            raise ValueError(f"bwd d256 2x2 f16: real lengths must satisfy 0 < sq_real <= sq, 0 < skv_real <= skv; got {sq_real}/{sq}, {skv_real}/{skv}")
    elif sq < 1 or skv < 1 or sq_env < 1:
        raise ValueError(f"bwd d256 2x2 f16 THD: the packed capacities and the q envelope must be >= 1; got sq={sq}, skv={skv}, sq_env={sq_env}")
    validate_head_chunk(qh, kh, qh_chunk)
    if b_chunk < 1 or b % b_chunk != 0:
        raise ValueError(f"bwd d256 2x2 f16: b_chunk ({b_chunk}) must be a positive divisor of B ({b})")

    def _fake_bshd(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

    if origin_rows is not None and (not _THD or len(origin_rows) != 9):
        raise ValueError("physical origins require a THD rendering and nine per-port row declarations")

    if _THD:
        # Packed operands, ONE batch; the kv-blocked workspace at every sequence padded to a kv block; the metadata + maps buffer.
        if b_chunk != b:
            raise ValueError(f"bwd d256 2x2 f16 THD: no batch chunking under THD (b_chunk must equal b = {b}); got {b_chunk}")
        sq_pad = -(-sq_env // CFG.TILE_N) * CFG.TILE_N
        ws_rows = -(-(skv + b * _KV_WRITE_ROWS) // _KV_WRITE_ROWS) * _KV_WRITE_ROWS
        if thd_units == 0:
            thd_units = (-(-skv // _KV_WRITE_ROWS) + b) * _THD_BLOCKS_PER_WRITE * qh_chunk
        fake_q = _fake_bshd((1, sq, qh, CFG.TILE_K), STORAGE_DTYPE)
        fake_do = _fake_bshd((1, sq, qh, CFG.TILE_O), STORAGE_DTYPE)
        fake_k = _fake_bshd((1, skv, kh, CFG.TILE_K), STORAGE_DTYPE)
        fake_v = _fake_bshd((1, skv, kh, CFG.TILE_O), STORAGE_DTYPE)
        fake_dv = _fake_bshd((1, skv, qh, CFG.TILE_O), OUT_STORAGE_DTYPE)
        fake_ds = _fake_bshd((1, qh_chunk, ws_rows, sq_pad), DS_STORAGE_DTYPE)
        if lse_token_major:
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (sq, qh), stride_order=(1, 0), assumed_align=16)
        else:
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
        fake_do_dot = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1, qh, -(-sq // 128) * 128), stride_order=(2, 1, 0), assumed_align=16)
        fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (THD_BWD_MAPS_META_WORDS(b, THD_INPUT_SLOTS + (6 if origin_rows is not None else 1) * b),), stride_order=(0,), assumed_align=128
        )
        problem = (b, qh, kh, sq_pad, ws_rows, qh_chunk, 1, sq_env, skv, thd_units)
    else:
        fake_q = _fake_bshd((b, sq, qh, CFG.TILE_K), STORAGE_DTYPE)
        fake_do = _fake_bshd((b, sq, qh, CFG.TILE_O), STORAGE_DTYPE)
        fake_k = _fake_bshd((b, skv, kh, CFG.TILE_K), STORAGE_DTYPE)
        fake_v = _fake_bshd((b, skv, kh, CFG.TILE_O), STORAGE_DTYPE)
        fake_dv = _fake_bshd((b, skv, qh, CFG.TILE_O), OUT_STORAGE_DTYPE)
        # dS workspace, chunk-local: [B_chunk, H_chunk, S_kv, S_q] io dtype (the dK = dS . Q GEMM's A operand un-permuted).
        fake_ds = _fake_bshd((b_chunk, qh_chunk, skv, sq), DS_STORAGE_DTYPE)
        fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
        fake_do_dot = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
        fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (b,), stride_order=(0,), assumed_align=16)
        problem = (b, qh, kh, sq, skv, qh_chunk, b_chunk, sq_real, skv_real)
    origin_args = {}
    if origin_rows is not None:
        origin_args = dict(
            thd_origins=cute.runtime.make_fake_compact_tensor(cutlass.Int64, (9, b), stride_order=(1, 0), assumed_align=16),
            thd_origin_rows=origin_rows,
        )
    if CFG.FUSE_DK:
        origin_args["dk_tensor"] = _fake_bshd((1 if _THD else b, skv, qh, CFG.TILE_K), OUT_STORAGE_DTYPE)
        if _THD:
            origin_args["dk_maps"] = cute.runtime.make_fake_compact_tensor(
                cutlass.Int64, ((2 * b + 1) * TENSOR_MAP_QWORDS,), stride_order=(0,), assumed_align=128
            )
    return _compile_cached(
        _host_origins if origin_rows is not None else _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dv,
        fake_ds,
        fake_lse,
        fake_do_dot,
        fake_seq_kv_lens,
        problem,
        cutlass.Float32(0.0),  # attn_scale
        cutlass.Int32(0),  # head_base
        cutlass.Int32(0),  # batch_base
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        **origin_args,
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_bwd_d256_2x2_f16",
    )


__all__ = [
    "CFG",
    "PARAMS",
    "LAYOUT",
    "Bars",
    "DESC_VERSION",
    "SPIN_RING_WAITS",
    "MMA_LOOKAHEAD",
    "STORAGE_DTYPE",
    "OUT_STORAGE_DTYPE",
    "DS_STORAGE_DTYPE",
    "_kernel",
    "_host",
    "compile",
]
