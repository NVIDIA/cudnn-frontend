# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""SDPA backward main kernel, d_qk = d_v = 256, bf16/fp16, SM107 (Rubin).

One cga2 pair (2 CTAs, 12 warps each) owns a 256-row kv block of ONE (batch,
head) and walks the q tiles that attend it.  It computes **dV in TMEM** and
writes **dS to a GMEM workspace** ``[B, H, S_kv, S_q]``; the adapter then runs
dK = dS . Q and dQ = dS^T . K as the ``bprop_matmul_blackwell`` GEMMs over that
workspace (kv-major so dK reads it un-permuted).  lane = kv row (S = [kv, q]).

Pipeline (per kv block; per q tile the MMA stream is the NATURAL order)::

    MMA (leader CTA), per q_iter i:
        BMM1 S  = K . Q[i]^T   -> S_acc  TMEM [0, 128)     K-split: mma_ss(K_front SMEM)
                                                           + mma_ts(K_back TMEM [512, 576))
        BMM1 dP = V . dO[i]^T  -> dP     TMEM [128, 256)   mma_ss
        BMM2 dV += P[i] . dO[i] -> dV_acc TMEM [256, 512)  mma_ts, A = P inside S_acc[32, 96)
    8 compute warps (2 wg x 4, wg = q half), per q_iter:
        softmax  P = exp2(S * scale * log2e - lse[q] * log2e) -> half P INTO S_acc[P_OFF + wg half]
        dSoftmax dS = (dP * scale - do_dot[q]) * P -> half dS -> SMEM ring -> TMA-STG -> workspace
    per kv block (post q-loop): dV_acc -> half -> SMEM (aliases K) -> TMA-STG -> dV [B, S_kv, H_q, d_v]

The BMM1 K-split: per kv block the leader MMA UTCCPs K's d_qk[128:256] (SMEM
subtiles 2, 3 = a contiguous 32 KiB back-half) into TMEM [512, 576), so S =
mma_ss(front, overwrite) + mma_ts(back, accumulate) contracts the full d_qk.  The
K back-half SMEM is dead once the UTCCP commits (``mb_k_utccp_done``), so the
second ``dO_dv`` ring stage ALIASES it -- a 2-deep dO_dv ring at zero net SMEM.
P is written back INTO S_acc: in-order MMA (Q.K[i+1] issues after P.dO[i]) covers
the S_acc WAR, so there is no ``s_acc_empty`` and no ``p_empty``.

Warp roles (per CTA, 12 warps; ids from ``CFG``)::

    0..3   softmax wg0  -- q[0:64]   of every q tile        (SOFTMAX_REGS = 224)
    4..7   softmax wg1  -- q[64:128]                          (SOFTMAX_REGS = 224)
    8      MMA          -- leader CTA: the 3-matmul stream; follower: quiet (alloc / dealloc only)
    9      TMA-LDG      -- K, V once per kv block; Q, dO, dO_dv per q tile   (OTHER_REGS = 56)
    10     TMA-STG      -- dS ring -> workspace per q tile; dV -> GMEM per kv block
    11     scheduler    -- CLC try_cancel protocol FUSED with the lse / do_dot SMEM prefetch

    8 x 224 + 4 x 56 = 2016 = 12 warps x the 168-register ENTRY count: the pool ``setmaxnreg``
    redistributes is the LAUNCH allocation (``USETMAXREG.*.CTAPOOL``), not the 2048-register file;
    a split summing past it parks the last softmax INCREASE forever (``config_sm107.reg_entry_pool``).

TMEM (576 columns, one ``tcgen05.alloc.cta_group::2`` per CTA, ``is_exclusive``)::

    [  0, 128)  S_acc   fp32 [kv, q]     BMM1 S; P (half, 64 cols) written back into [32, 96):
                                         wg0 -> [32, 64), wg1 -> [64, 96) (inside its own S q-half)
    [128, 256)  dP      fp32 [kv, q]     BMM1 dP, single buffer, ``mb_dp_empty`` gates reuse
    [256, 512)  dV_acc  fp32 [kv, d_v]   BMM2, ``accumulate = (q_iter > q_lo)``, freed per kv block
    [512, 576)  K_back  half [kv, d_qk[128:256]] UTCCP'd per kv block (leader only, self-fill)

SMEM (per CTA, declaration order == ``config_sm107.smem_layout``; every slab 1024-B aligned)::

    slab                          bytes    offset   writer / reader                 swizzle + why
    sQ[2]      64 q x 256 d      65536   0        TMA / MMA desc B (front + back)  128 B, descriptor-matched
    sdO[2]     64 q x 256 d_v    65536   65536    TMA / MMA desc B (dP)            128 B, descriptor-matched
    sCombined = [sdO_dv[0] | K | V]     163840   131072
      sdO_dv[0] 128 q x 128 d_v  32768   131072   TMA / MMA desc B (dV, BT)        128 B, descriptor-matched
      sK        128 kv x 256 d   65536   163840   TMA / MMA desc A + UTCCP source  128 B, descriptor-matched
      sdO_dv[1] == sK back half  32768   196608   TMA (after mb_k_utccp_done)      alias seam
      sV        128 kv x 256 d_v 65536   229376   TMA / MMA desc A (dP)            128 B; LAST desc root
      sdV (post-loop alias of sK) 65536  163840   lanes store_swizzled / TMA-STG   Swizzle(3,4,3): row = 128 B
    sStats[2]  2 x 256 fp32       2048   294912   scheduler lanes / softmax lanes  none (4-B stride, broadcast reads)
    sdS[1]     128 kv x 128 q    32768   296960   lanes store_swizzled / TMA-STG   Swizzle(3,4,3): row = 128 B
    mbarriers + tmem ptr + scheduler ring            ~0.6 KiB              -> 322.5 KiB + scaffold, under 327 KiB

    The largest tcgen05 ``build()`` root is sV at 229376 < 262144 -> ``DESC_VERSION = 0``
    (derived, ``config_sm107.desc_version``); sStats / sdS sit past 256 KiB but no
    descriptor reads them.

BARRIER TABLE (lane ledger; L = leader CTA, F = follower; x N = per q tile, x T = per kv block)::

    mbar[stages]        producer / scope     init L / F  arrive site (guard)                          issuing lanes      waiter(s)                 phase   fires
    mb_q_full[2]        TMA_LOAD / LOCAL     1 / 1(-)    TMALDG, pred=is_leader & elect               1, L only          MMA(L)                    st(0)   x N
    mb_q_empty[2]       MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast(2 CTAs)  TMALDG L+F; DRAIN x2      st(1)   x N
    mb_do_full[2]       TMA_LOAD             1 / 1(-)    TMALDG, pred=is_leader & elect               1, L only          MMA(L)                    st(0)   x N
    mb_do_empty[2]      MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          TMALDG L+F; DRAIN x2      st(1)   x N
    mb_dodv_full[2]     TMA_LOAD             1 / 1(-)    TMALDG, pred=is_leader & elect               1, L only          MMA(L)                    st(0)   x N
    mb_dodv_empty[2]    MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          TMALDG L+F; DRAIN x2      st(1)   x N
    mb_k_full[1]        TMA_LOAD             1 / 1(-)    TMALDG, pred=is_leader & elect               1, L only          MMA(L)                    st(0)   x T
    mb_k_empty[1]       MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          TMALDG L+F; DRAIN x1      st(1)   x T
    mb_v_full[1]        TMA_LOAD             1 / 1(-)    TMALDG, pred=is_leader & elect               1, L only          MMA(L)                    st(0)   x T
    mb_v_empty[1]       MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          TMALDG L+F; DRAIN x1      st(1)   x T
    mb_k_utccp_done[1]  MMA_COMMIT           1 / 1       MMA(L), inside `if elect_p:` with the UTCCP  1 x mcast          TMALDG L+F (same tile)    st(0)   x T
    mb_s_acc_full[1]    MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          softmax L+F (256 lanes)   st(0)   x N
    mb_dp_full[1]       MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          softmax L+F               st(0)   x N
    mb_dp_empty[1]      LEADER / LEADER      512 / 1(-)  softmax, BARE, 8 warps x 2 CTAs              256 x 2 -> L       MMA(L); DRAIN x1          st(1)   x N
    mb_p_ready[1]       LEADER / LEADER      512 / 1(-)  softmax, BARE, 8 warps x 2 CTAs              256 x 2 -> L       MMA(L)                    st(0)   x N
    mb_stats_full[2]    THREAD               32 / 32     scheduler warp, BARE                         32 (local)         softmax (256 lanes)       st(0)   x N
    mb_stats_empty[2]   THREAD               256 / 256   softmax, BARE, 8 warps                       256 (local)        scheduler warp            st(1)   x N
    mb_ds_smem_full[1]  THREAD               256 / 256   softmax, BARE, 8 warps                       256 (local)        TMASTG                    st(0)   x N
    mb_ds_smem_empty[1] THREAD               1 / 1       TMASTG, `if elect_sync():`                   1 (local)          softmax                   st(1)   x N
    mb_dv_ready[1]      MMA_COMMIT           1 / 1       MMA(L), pred=elect_p                         1 x mcast          softmax L+F               st(0)   x T
    mb_dv_acc_empty[1]  LEADER / LEADER      512 / 1(-)  softmax, BARE, 8 warps x 2 CTAs              256 x 2 -> L       MMA(L)                    st(0)   x T
    mb_dv_stg_full[1]   THREAD               256 / 256   softmax, BARE, 8 warps                       256 (local)        TMASTG                    st(0)   x T
    mb_dv_stg_empty[1]  THREAD               1 / 1       TMASTG, `if elect_sync():`                   1 (local)          TMALDG (pre-armed)        st(1)   x T
    mb_tmem_dealloc[1]  THREAD (+peer)       512 / 512   softmax BARE local + BARE arrive_on_peer     256 + 256          MMA (each CTA)            0       x 1
    sched.mb_scheduler[2]     expect_tx 16 B  1 / 1     scheduler, CTA 0 elect arms EVERY CTA        1                  every persistent warp     st(0)   x T
    sched.mb_read_tile_id[2]  read_tile_id_arrive  21 / 21  8 softmax + TMALDG + TMASTG on both CTAs + MMA(L) = 21 calls, 1 arrive per CTA per call

    SUM(issuing lanes) == init on every row and both CTAs; a "1(-)" follower init is a
    copy that is never armed and never waited (cga2 tensor TMAs deliver both peers' bytes
    to the LEADER's mbar; LEADER-scope arrives all land on the leader).

    P15 drains (cross-CTA ASYNC arrives whose last fire is in flight at exit) are OWNED by
    the consumer: TMALDG drains q/do/dodv/k/v_empty (STAGES_x waits each) after its loop;
    the leader MMA drains the one un-waited pre-armed batch of mb_dp_empty before
    mb_tmem_dealloc; the mb_tmem_dealloc wait itself is the drain of the peer arrives.
    Every other cross-CTA commit (k_utccp_done, s_acc_full, dp_full, dv_ready, p_ready,
    dv_acc_empty) is consumed in-loop within the tile that fired it.

Masks are the TRANSPOSE of the forward (lane = kv row, inner loop = q tile): the kv
block bounds WHICH q tiles attend (``compute_q_loop_bounds``) and the per-cell mask
zeroes P on masked (kv, q) cells, so dV / dS / dK / dQ inherit the mask.  An empty
q range is FORCED to one fully-masked q tile (P = 0 -> dV = 0 written with
``accumulate=False``, dS = 0): the MMA prologue cannot be skipped by a runtime ``if``
and every per-tile ring stays balanced (sdpa-invariants S1-2).  SWA under
bottom-right alignment anchors the window to the bottom-right diagonal
(kv >= q + (S_kv - S_q) - W), the same anchor ``tile_dsl.mask.apply_mask_chunk`` uses.
TODO(plan s13 PR-4): a transposed ``apply_mask_chunk_bits`` arm; this port keeps the
validated per-cell select form (``_mask_p_chunk``).

## Launch ABI

``compile(b, qh, kh, sq, skv, qh_chunk=0, b_chunk=0)`` returns a launchable ``fn``;
every extent is compile-time (pins the TMA strides).  ``sq % 128 == 0``,
``skv % 256 == 0`` (the adapter pads; a q row past the real length is handled by
zero-filling Q / dO and setting ``lse = +inf``, ``do_dot = 0`` there -> P = 0), ``qh %
kh == 0``, ``qh_chunk`` divides ``qh`` and is a multiple of ``qh // kh`` (default
``qh``), ``b_chunk`` divides ``b`` (default ``b``).  Call, positionally::

    fn(q, do, k, v, dv, ds, lse, do_dot, seq_kv_lens, problem_size, attn_scale, head_base, batch_base, stream)

    q       [B, S_q,  H_q,  256] io dtype (BSHD)     do   [B, S_q,  H_q,  256] io dtype
    k       [B, S_kv, H_kv, 256] io dtype             v    [B, S_kv, H_kv, 256] io dtype
    dv      [B, S_kv, H_q,  256] DTYPE_O (OUT)        -- per-Q-head partial; the adapter folds GQA groups
    ds      [b_chunk, qh_chunk, S_kv, S_q] io dtype (OUT, kv-major workspace; chunk-local heads / batches)
    lse     [B, H_q, S_q] fp32 natural-log Stats of the forward (log2e applied in-kernel)
    do_dot  [B, H_q, S_q] fp32 RAW delta = rowsum(dO * O) (``dot_do_o`` output; attn_scale applied in-kernel)
    seq_kv_lens [B] int32 per-batch real kv lengths; read ONLY under the PADDED mask arm
                (``TemplateParams.seq_kv_lens_present``); pass a 1-element dummy otherwise
    problem_size  (B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL) -- the compile() extents plus the
                REAL lengths: SQ / SKV are the padded extents (grid, workspace, n_q_tiles); SQ_REAL / SKV_REAL
                only feed the bottom-right causal diagonal (S_kv_real - S_q_real)
    attn_scale  fp32 softmax scale (the kernel derives attn_scale * log2e itself)
    head_base   int32: full-tensor head = grid head + head_base (Q/K/V/dO/dV/lse/do_dot); ds stays chunk-local
    batch_base  int32: full-tensor batch = grid batch + batch_base (same rule)
    stream      CUstream

Grid: NATURAL ``(ceil(SKV / 256) * 2, QH_CHUNK, B_CHUNK)``, cluster (2, 1, 1); LPT / LPT_L2
flatten to ``(kv_blocks * QH_CHUNK * B_CHUNK * 2, 1, 1)`` (``config_sm107.launch_grid``).
Under a mask the adapter must zero the ds workspace region a launch does not touch
(the q tiles a kv block skips) before the dK / dQ GEMMs read it.

Q6 (open A/B): the pre-port body passed idesc ``k_dim=1`` on its f16 descriptors; this
port uses the f16 default ``k_dim=0`` (``CFG.IDESC_K_DIM``, rules/mma-tma-matrix.md S2)
like every FROST f16 kernel -- the bitwise A/B against the pre-port kernel decides.
"""

from functools import lru_cache
from typing import Callable, NamedTuple, Tuple

import cuda.bindings.driver as _cuda_driver  # noqa: F401
import cutlass
import cutlass.cute as cute
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
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from cudnn.frost.tile_dsl.mask import MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA, compute_q_loop_bounds
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile
from cudnn.frost.tile_dsl.scheduler import SCHED_NATURAL, Sched, read_clc_payload, read_tile_id_arrive
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc

# Config comes from the FROST template loader, never an env var: the loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a plain import usable.
from cudnn.sdpa.bwd.config_sm107 import FAMILY_F16, TemplateParams, buffer_elems, desc_version, make_cfg_d256_bwd, tmem_layout, validate_head_chunk

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG = make_cfg_d256_bwd(PARAMS, FAMILY_F16)
_B = buffer_elems(CFG)
LAYOUT = tmem_layout(CFG)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision
# point, derived from the config's declaration-order slab table (the largest build()
# root, sV, sits at 224 KiB -> 0).  A version-0 descriptor's start_address is 14 bits =
# a 256 KiB window; an operand at or past it wraps to offset 0 and the MMA multiplies
# the bottom of SMEM -- an EXACTLY-zero accumulator, no crash.  Never re-literal it at
# a call site (test: every SmemTile takes desc_version=DESC_VERSION).
DESC_VERSION: int = desc_version(CFG)

# Retry form of the per-q-iteration / per-kv-block RING waits (q/do/dodv/k/v full + empty,
# k_utccp_done, s_acc_full, dp_full, dp_empty, p_ready, stats_*, ds_smem_*, dv_ready,
# dv_acc_empty, dv_stg_*): every such site spells ``spin=SPIN_RING_WAITS``.  The waits a
# warp parks in for a whole tile (scheduler payload, mb_tmem_dealloc) and the
# end-of-kernel drains keep the default sleeping form.  The sign is a MEASURED per-kernel
# fact (rules/frost-tile-dsl.md S8b); unmeasured on this body -> the default.
SPIN_RING_WAITS: bool = False

# --- dtype dispatch (tile_dsl DTYPE_* codes) -------------------------------------------
if CFG.DTYPE_QKV == DTYPE_BF16:
    STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_QKV == DTYPE_FP16:
    STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"{__name__}: DTYPE_QKV must be DTYPE_BF16 ({DTYPE_BF16}) or DTYPE_FP16 ({DTYPE_FP16}); got {CFG.DTYPE_QKV}")
MMA_KIND = nvvm.Tcgen05MMAKind.F16
# dV grad storage dtype (BPE_O == BPE, so the dV alias of sK never resizes).
OUT_STORAGE_DTYPE = cutlass.BFloat16 if CFG.DTYPE_O == DTYPE_BF16 else cutlass.Float16
# dS workspace dtype == the io dtype on this body (config pins DTYPE_DS == DTYPE_QKV).
DS_STORAGE_DTYPE = STORAGE_DTYPE

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2  # CFG.CTA_MMA == 2, pinned by the config
CGA_SIZE = _B.CGA_SIZE  # 2

# --- per-CTA SMEM element counts / TMA geometry (all from the config, never re-derived) ---
_M_PER_CTA = _B._M_PER_CTA  # 64 q rows per CTA (Q / dO N-split across the pair)
qBufferElems = _B.qBufferElems  # 64 x 256
dOBufferElems = _B.dOBufferElems  # 64 x 256
kBufferElems = _B.kBufferElems  # 128 x 256
vBufferElems = _B.vBufferElems  # 128 x 256
dSBufferElems = _B.dSBufferElems  # 128 x 128
dVBufferElems = _B.dVBufferElems  # 128 x 256
K_SPLIT_TILE_K = CFG.TILE_K // 2  # 128: half the d_qk contraction per BMM1 MMA
K_BACK_OFF_ELEMS = _B.K_BACK_OFF_ELEMS  # sK subtiles 2, 3 (d_qk[128:256])
Q_BACK_OFF_ELEMS = _B.Q_BACK_OFF_ELEMS  # sQ subtiles 2, 3
_DODV_STAGE_ELEMS = _B._DODV_STAGE_ELEMS  # dOBufferElems + K_BACK_OFF_ELEMS: sdO_dv[1] == sK back half

qTmaTransactionBytes = _B.qTmaTransactionBytes
dOTmaTransactionBytes = _B.dOTmaTransactionBytes
kTmaTransactionBytes = _B.kTmaTransactionBytes
vTmaTransactionBytes = _B.vTmaTransactionBytes
# dO_dv: per CTA TILE_N q rows x TILE_O / CTA_MMA d_v cols; cga2 tensor bytes of both peers land on the leader.
dOdvTmaTransactionBytes = CFG.TILE_N * (CFG.TILE_O // CFG.CTA_MMA) * CFG.BPE * CFG.CTA_MMA

TMA_QK_ITERS = _B.TMA_QK_ITERS  # 4 subtiles of 64 elems (128 B) per 256-elem half row
TMA_VO_ITERS = _B.TMA_VO_ITERS
TMA_QK_GRANU_ELEMS = _B.TMA_QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _B.TMA_VO_GRANU_ELEMS
TMA_VO_SG1_ITERS = _B.TMA_VO_SG1_ITERS  # dO_dv (BT view): 2 subtiles of 64 d_v cols per CTA
TMA_VO_SG1_GRANU_ELEMS = _B.TMA_VO_SG1_GRANU_ELEMS
DV_D_BLOCK = _B.DV_D_BLOCK  # 64 d_v cols per dV store subtile (128 B)
TMA_DV_ITERS = _B.TMA_DV_ITERS  # 4
TMA_DV_GRANU_ELEMS = _B.TMA_DV_GRANU_ELEMS
DV_BLOCK_SLAB = _B.DV_BLOCK_SLAB  # TILE_M * DV_D_BLOCK elems per dV subtile slab
P_TMA_ITERS = _B.P_TMA_ITERS  # 2 dS store subtiles of 64 q cols (128 B)
P_D_BLOCK = _B.P_D_BLOCK  # 64
P_BLOCK_ELEMS = _B.P_BLOCK_BYTES  # TILE_M * P_D_BLOCK ELEMS per dS subtile slab (the config keeps the pre-port name)

LEADING_BYTE_OFFSET_QK = _B.LEADING_BYTE_OFFSET_QK
STRIDE_BYTE_OFFSET_QK = _B.STRIDE_BYTE_OFFSET_QK
LEADING_BYTE_OFFSET_dO = _B.LEADING_BYTE_OFFSET_dO
STRIDE_BYTE_OFFSET_dO = _B.STRIDE_BYTE_OFFSET_dO
LEADING_BYTE_OFFSET_dS = _B.LEADING_BYTE_OFFSET_dS
STRIDE_BYTE_OFFSET_dS = _B.STRIDE_BYTE_OFFSET_dS
# BT=true B operand (dO_dv): per-CTA N = TILE_O / CTA_MMA = 128 -> 128 // 8 > 8 -> leading = K x swizzle.
LEADING_BYTE_OFFSET_dO_SG1 = _B.LEADING_BYTE_OFFSET_dO_SG1
STRIDE_BYTE_OFFSET_dO_SG1 = _B.STRIDE_BYTE_OFFSET_dO_SG1
SMEM_LAYOUT_Q = _B.SMEM_LAYOUT_Q
SMEM_LAYOUT_dO = _B.SMEM_LAYOUT_dO
SMEM_LAYOUT_K = _B.SMEM_LAYOUT_K
SMEM_LAYOUT_V = _B.SMEM_LAYOUT_V
SMEM_LAYOUT_dS = _B.SMEM_LAYOUT_dS
SMEM_LAYOUT_dV = _B.SMEM_LAYOUT_dV

# dS ring and dV staging lane stores: one 64-elem (128 B) row per lane = the 128 B bank
# cycle, so an unswizzled store would be a 32-way conflict; Swizzle(3, 4, 3) has
# MBase + SShift = 7 = log2(128 B) AND matches the s128b TMA-store descriptors (both jobs,
# rules/frost-kernels.md S5).
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)

# lse / do_dot prefetch ring: STATS_STAGES slots of [lse[TILE_N] | do_dot[TILE_N]] fp32.
STATS_SLOT_ELEMS = _B.STATS_SLOT_ELEMS
STATS_LSE_OFF = _B.STATS_LSE_OFF
STATS_DOT_OFF = _B.STATS_DOT_OFF
_SMX_CHUNK = _B._SMX_CHUNK  # 64 q cols per softmax wg (lane = kv row -> 64 q elems per lane)
_LDTM_NUM = _B._LDTM_NUM  # tcgen05.ld.32x32b.x64 per q half
_KV_BLOCK_ROWS = _B._KV_BLOCK_ROWS  # 256 kv rows per cga2 pair
# log2(e): P = exp(scale * S - lse) = exp2(scale * log2e * S - lse * log2e); the lse
# factor is applied by the scheduler warp at prefetch time (the host passes natural-log lse).
_LOG2E = 1.4426950408889634
# CLC response payload the scheduler ring's expect_tx arms (tile_dsl/scheduler.py uses the same 16).
_CLC_RESPONSE_BYTES = 16

# --- named arrive counts (P3) -------------------------------------------------------------
ONE_LANE = CFG.ONE_LANE
ONE_WARP = CFG.ONE_WARP
SOFTMAX_LANES_ALL_WG = CFG.SOFTMAX_LANES  # 8 warps x 32 = every compute lane of ONE CTA
SOFT_X_CTA_MMA = CFG.SOFT_X_CTA_MMA  # x CTA_MMA: every compute lane of BOTH CTAs -> the leader
MMA_COMMIT_ARRIVES = CFG.MMA_COMMIT_ARRIVES  # one predicated tcgen05.commit multicast = 1 arrive per target CTA
READ_TILE_ARRIVERS_TOT = CFG.READ_TILE_ARRIVERS_TOT  # 21, derivation in config_sm107.read_tile_arrivers_tot


# ============================================================================
# Mask plumbing (the TRANSPOSE of the forward: lane = kv row, inner loop = q tile)
# ============================================================================


def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real):
    """Real kv length of this batch entry: ``seq_kv_lens[batch_idx + batch_base]`` under the PADDED arm, else the
    scalar.  The full-tensor batch is formed HERE so the MMA / compute warps (which never address a full tensor)
    carry no extra live register for it -- the 40-register MMA warp is one register from spilling."""
    if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
        arr = cutlass.make_array_view(seq_kv_lens_tensor)
        return cutlass.Int32(arr[batch_idx + batch_base])
    return seqlen_kv_real


def _causal_diag(seqlen_q_real, eff_seqlen_kv):
    """Bottom-right diagonal offset (S_kv - S_q); 0 for top-left / dense."""
    if cutlass.const_expr(CFG.CAUSAL_BOTTOM_RIGHT):
        return eff_seqlen_kv - seqlen_q_real
    return cutlass.Int32(0)


def _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv):
    """[q_lo, q_hi) -- the q-tile range that attends this kv block, FORCED non-empty.

    ``compute_q_loop_bounds`` (tile_dsl.mask) trims causal from below and SWA from above
    with the bottom-right anchor; MASK_NONE -> [0, n_q_tiles).  An empty range (a kv
    block past the last query under top-left causal with S_kv > S_q, a window that
    ends before this block) would leave the UNCONDITIONAL MMA prologue waiting on a
    mb_q_full the TMA warp never armed, and a runtime ``if`` cannot skip it (loop-carried
    PipelineState reassignments do not propagate out of a runtime branch).  So q_lo is
    clamped in range and N >= 1 is forced: the single forced tile is fully masked by
    ``_mask_p_chunk`` (P = 0 -> dV = 0 via accumulate=False, dS = 0) and every per-iter ring
    stays balanced.  A no-op for non-empty ranges.  Every warp derives the SAME bounds from
    the same (kv_block_base, lengths) -- the P14 balance of six loop bodies depends on it.
    """
    n_q_tiles = seqlen_q // cutlass.Int32(CFG.TILE_N)
    b = compute_q_loop_bounds(
        kv_block_base,
        seqlen_q_real,
        eff_seqlen_kv,
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


def _mask_p_chunk(reg_P, kv_abs, q_col_base, eff_seqlen_kv, causal_diag, N: int):
    """Zero P on masked (kv = lane, q = col) cells; MASK_NONE returns reg_P unchanged (no IR).

    kv_abs = this lane's absolute kv row; q_col_base = absolute q of column 0; N columns.
      causal : kv_abs >  q_abs + diag       (key past the query; diag = S_kv - S_q under bottom-right)
      SWA    : kv_abs <  q_abs + diag - W   (key left of the window, bottom-right-anchored like the forward)
      padded : kv_abs >= seq_kv_len         (per-lane pad row; the whole row goes to 0)
    Per-cell compare + select (the validated form).  TODO(plan s13 PR-4): the transposed
    keep-word / register-to-predicate arm (tile_dsl.mask.band_mask_words) -- per lane the
    band is [kv_abs - diag, kv_abs + diag' + W] in q, so it maps onto band_mask_words directly.
    """
    if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
        return reg_P
    zero = cutlass.Float32(0.0)
    elems = []
    for i in range(N):
        q_abs = q_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_PADDED):
            t = kv_abs >= eff_seqlen_kv
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_CAUSAL):
            t = kv_abs > (q_abs + causal_diag)
            masked = t if masked is None else (masked | t)
        if cutlass.const_expr(CFG.MASK_FLAGS & MASK_SWA):
            t = kv_abs < (q_abs + causal_diag - cutlass.Int32(CFG.SWA_WINDOW))
            masked = t if masked is None else (masked | t)
        elems.append(cutlass.Float32(arith.select(masked.ir_value(), zero.ir_value(), reg_P[i].ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


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
    # BMM1 K-split alias seam: leader MMA commit after the UTCCP -> TMA-LDG may clobber the
    # K back-half SMEM with the sdO_dv stage-1 load.
    mb_k_utccp_done: object
    mb_s_acc_full: object
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


def _make_bars(cfg) -> Bars:
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
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
        # init = 1 per CTA BECAUSE exactly one lane issues the commit (it sits inside the
        # MMA warp's `if elect_p:` block with the UTCCP loop); an un-elected commit would be
        # 32 arrives against this count (rules/mbarrier-patterns.md P3).
        mb_k_utccp_done=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # S single buffer; NO mb_s_acc_empty on this body (P lives in S_acc, in-order MMA covers the WAR).
        mb_s_acc_full=MBarrier(_alloc(cfg.STAGES_TMEM_S), stages=cfg.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dp_full=MBarrier(_alloc(cfg.STAGES_TMEM_S), stages=cfg.STAGES_TMEM_S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # dP slot freed by dSoftmax (after its tmem_load); every compute lane of BOTH CTAs
        # arrives on the leader (bare LEADER arrive, 256 x 2 = 512); follower init = ONE_LANE (never waited).
        mb_dp_empty=MBarrier(_alloc(cfg.STAGES_TMEM_S), stages=cfg.STAGES_TMEM_S, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_p_ready=MBarrier(_alloc(cfg.STAGES_TMEM_P), stages=cfg.STAGES_TMEM_P, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        # lse / do_dot prefetch ring: scheduler warp (32 lanes, bare) -> softmax; softmax (256 lanes, bare) -> scheduler.
        mb_stats_full=MBarrier(_alloc(cfg.STATS_STAGES), stages=cfg.STATS_STAGES, init_count=ONE_WARP, producer=Producer.THREAD),
        mb_stats_empty=MBarrier(_alloc(cfg.STATS_STAGES), stages=cfg.STATS_STAGES, init_count=SOFTMAX_LANES_ALL_WG, producer=Producer.THREAD),
        # dS SMEM ring: softmax store_swizzled (all 256 lanes, bare) -> TMA-STG; TMA-STG `if elect_sync():` -> softmax.
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
    """Flat 1-D (LPT / LPT_L2) decode: linear cluster id -> (kv_super, head, batch); kv_super
    is the OUTER axis so the heaviest causal kv blocks (small kv_super, attended by the most
    queries) land in the first wave.  ``n_qh_grid`` is the GRID head extent (== QH_CHUNK)."""
    hb = n_qh_grid * n_batch
    kv_super = linear // hb
    within = linear % hb
    head = within % n_qh_grid
    batch = within // n_qh_grid
    return kv_super, head, batch


def _boot_tile(sched):
    """The FIRST tile (kv_super, head, batch) from the launch blockIdx.  NATURAL: (bidx // CGA_M,
    bidy, bidz).  LPT / LPT_L2: bidx is the linear cluster base and bidy_init / bidz_init are
    REPURPOSED to carry (n_qh_grid, n_batch) -- dead under a (N, 1, 1) grid."""
    linear = sched.bidx_init // cutlass.Int32(CFG.CGA_M)
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
        kv, h, b = linear, sched.bidy_init, sched.bidz_init
    else:
        kv, h, b = _decode_linear(linear, sched.bidy_init, sched.bidz_init)
    return cute.arch.make_warp_uniform(kv), cute.arch.make_warp_uniform(h), cute.arch.make_warp_uniform(b)


def _decode_tile_words(sched, t0, t1):
    """(kv_super, head, batch) from the CLC response words (``read_clc_payload``): t0 = first
    ctaid.x of the cluster (= kv_super * CGA_M under NATURAL, linear * CGA_M under LPT), t1 =
    packed (head = low 16, batch = high 16) under NATURAL (y = z = 0 on the flat grid)."""
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
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q   [B, S_q, H_q, d]  box (1, 64, 1, 64)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  dP view          box (1, 64, 1, 64)
    tma_do_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  dV view (BT)     box (1, 128, 1, 64)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],  # K   [B, S_kv, H_kv, d] box (1, 128, 1, 64)
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],  # V
    # TMA descriptors -- stores
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],  # dV  [B, S_kv, H_q, d_v]  box (1, 128, 1, 64)
    tma_ds_desc: cutlass.GridConstant[tmap.TensorMap],  # dS  [B_chunk, H_chunk, S_kv, S_q]  box (1, 1, 128, 64)
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
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- SharedStorage: DECLARATION ORDER == config_sm107.smem_layout (sQ | sdO | sCombined | sStats | sdS) ---
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_dO * dOBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # [sdO_dv[0] | K | V] share ONE backing so the sdO_dv stage-1 alias of the (post-UTCCP
    # dead) K back-half is a COMPILE-TIME element offset: the ring stride _DODV_STAGE_ELEMS =
    # dOBufferElems + K_BACK_OFF_ELEMS puts sdO_dv[1] exactly on sK + K_BACK_OFF_ELEMS.  The
    # K -> V offset is kBufferElems ELEMENTS (subview advances in elements; a `* BPE` here
    # doubled it on the pre-port body and pushed sV past the cap -> TMA OOB).
    _sCombined_raw = cutlass.Array(STORAGE_DTYPE, dOBufferElems + kBufferElems + vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = _sCombined_raw.subview(dOBufferElems)
    sV_raw = _sCombined_raw.subview(dOBufferElems + kBufferElems)
    # dV epilogue staging: an OUT_STORAGE_DTYPE view of sK (dVBufferElems * BPE_O == kBufferElems * BPE:
    # the alias covers EXACTLY sK, not K + V), live only after the q loop.
    sdV_raw = cutlass.Array(sK_raw.data_ptr(), shape=dVBufferElems, dtype=OUT_STORAGE_DTYPE)
    # lse / do_dot prefetch ring (fp32; K + V are live through the q loop, so no free corner to alias).
    sStats_raw = cutlass.Array(cutlass.Float32, CFG.STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # dS SMEM ring (io dtype): softmax store_swizzled -> TMA-STG -> GMEM workspace.  LOCAL per CTA.
    sdS_raw = cutlass.Array(DS_STORAGE_DTYPE, CFG.XFER_STAGES * dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)

    # --- SmemTile wrappers (every one takes desc_version=DESC_VERSION) ----------------------
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
    # dO, dP view: BMM1 dP B operand (BT=false, 64 q x 256 d_v per CTA, 4 subtiles).
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
    # dO, dV view: BMM2 dV B operand (BT=true, 128 q x 128 d_v per CTA, leading = TILE_N x swz).
    # The SAME dO GMEM under a DIFFERENT box + swizzle interpretation, so it cannot share sdO
    # (the 128 B XOR maps cells differently for 256 B and 128 B rows).  elems_per_stage is the
    # ALIAS stride (stage 1 == sK back half); the tile geometry stays dOBufferElems-shaped.
    sdO_dv = SmemTile(
        base=_sCombined_raw,
        elems_per_stage=_DODV_STAGE_ELEMS,
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
    # dS ring: [kv, q] io dtype, TILE_N x BPE = 256 B per row = P_TMA_ITERS subtiles of P_D_BLOCK q cols.
    sdS = SmemTile(
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=CFG.XFER_STAGES,
        leading_byte_offset=LEADING_BYTE_OFFSET_dS,
        stride_byte_offset=STRIDE_BYTE_OFFSET_dS,
        layout=SMEM_LAYOUT_dS,
        tma_loads_per_tile=P_TMA_ITERS,
        tma_granu_elems=P_D_BLOCK,
        tma_subtile_stride_elems=CFG.TILE_M * P_D_BLOCK,
        desc_version=DESC_VERSION,
    )
    # dV staging: TMA_DV_ITERS subtiles of (TILE_M kv x DV_D_BLOCK d_v), 128 B swizzled.  A TMA-STG
    # source only (never an MMA operand), so leading / stride are 0.
    sdV = SmemTile(
        base=sdV_raw,
        elems_per_stage=dVBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_dV,
        tma_loads_per_tile=TMA_DV_ITERS,
        tma_granu_elems=TMA_DV_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_DV_GRANU_ELEMS,
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
            # NATURAL: bidy / bidz = (head, batch).  LPT / LPT_L2 flat grid: blockIdx.{y,z} = 0 (dead),
            # so these slots carry (n_qh_grid, n_batch) for the linear decode (_boot_tile / _decode_tile_words).
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
            bars.mb_k_utccp_done.init()
            # P8: leader-waited fan-ins (dp_empty / p_ready / dv_acc_empty): the leader's copy
            # takes every compute lane of BOTH CTAs, the follower's copy is never waited.
            LEADER_INIT = cutlass.Int32(
                arith.select(
                    is_leader.ir_value(),
                    cutlass.Int32(SOFT_X_CTA_MMA).ir_value(),
                    cutlass.Int32(ONE_LANE).ir_value(),
                )
            )
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_S):
                bars.mb_s_acc_full[p].init()
                bars.mb_dp_full[p].init()
                bars.mb_dp_empty[p].init(override_count=LEADER_INIT)
            for p in cutlass.range_constexpr(CFG.STAGES_TMEM_P):
                bars.mb_p_ready[p].init(override_count=LEADER_INIT)
            bars.mb_dv_ready.init()
            bars.mb_dv_acc_empty.init(override_count=LEADER_INIT)
            bars.mb_dv_stg_full.init()
            bars.mb_dv_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            for p in cutlass.range_constexpr(CFG.STATS_STAGES):
                bars.mb_stats_full[p].init()
                bars.mb_stats_empty[p].init()
            for p in cutlass.range_constexpr(CFG.XFER_STAGES):
                bars.mb_ds_smem_full[p].init()
                bars.mb_ds_smem_empty[p].init()
            # Scheduler ring on EVERY CTA (the try_cancel multicast targets all of them).
            for s in cutlass.range_constexpr(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    # P4 order: init -> fence -> CTA sync -> cluster sync, all OUTSIDE the warp branch.  Every
    # consumer whose producer fires after it is pre-armed by PipelineState.start(phase=1); no
    # bootstrap arrives.
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    # Pair-scoped tcgen05.commit multicast mask and this CTA's TMA multicast bit.
    mcast_mask = cutlass.Int32(3) << leader_cta_id
    tma_mcast_mask = cutlass.Int16(1) << cta_in_pair
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    # --- warp dispatch ------------------------------------------------------------------------
    if warp_idx < cutlass.Int32(CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            warp_idx=warp_idx,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            sdS_raw=sdS_raw,
            sStats_raw=sStats_raw,
            sdV_raw=sdV_raw,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
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
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
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
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            qh_per_kh=qh_per_kh,
            head_base=head_base,
            batch_base=batch_base,
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
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            head_base=head_base,
            batch_base=batch_base,
            cta_in_pair=cta_in_pair,
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
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            attn_scale=attn_scale,
            seqlen_q=seqlen_q,
            seqlen_q_real=seqlen_q_real,
            seqlen_kv_real=seqlen_kv_real,
            head_base=head_base,
            batch_base=batch_base,
        )


# ============================================================================
# Warp bodies, in pipeline order: TMA-LDG -> MMA -> softmax / dSoftmax -> TMA-STG -> scheduler
# ============================================================================


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
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    qh_per_kh,
    head_base,
    batch_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
) -> None:
    """TMA-LDG warp (both CTAs).  Per kv block: K, V once (M-split kv, full d).  Per q tile:
    Q (N-split on q), dO dP view (same split), dO dV view (N-split on d_v, full q rows, BT).
    Only the LEADER arms expect_tx (P9: cga2 tensor TMAs deliver both peers' bytes to the
    leader's mbar); both CTAs issue their own loads.
    """
    tma_q = GmemTileTma(tma_q_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_do_dv = GmemTileTma(tma_do_dv_desc)
    tma_v = GmemTileTma(tma_v_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    # Full-tensor coordinates: the grid's (head, batch) are chunk-local.
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
    full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)

    # dO_dv per-CTA inner (d_v) offset; its q axis is FULL (no per-CTA q offset).
    DO_DV_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)
    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)
    Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_M_PER_CTA)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    dodv_empty_state = PipelineState.start(phase=1)
    # sdV aliases sK: the K load of block t must not land while the TMA-STG still READS the
    # dV staging of block t-1 (its tma_store_wait -> mb_dv_stg_empty is the only signal that
    # the read finished).  Pre-armed: block 0 has nothing to wait for.
    dv_stg_empty_state = PipelineState.start(phase=1)
    # K-split alias seam: a real producer (the leader's UTCCP commit of THIS block), not pre-armed.
    utccp_done_state = PipelineState.start(phase=0)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)

        # ---- K + V, once per kv block ----
        bars.mb_dv_stg_empty.wait(dv_stg_empty_state.phase, spin=SPIN_RING_WAITS)
        dv_stg_empty_state = advance(dv_stg_empty_state, 1)
        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sK[k_empty_state.idx],
            tma_k(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, full_batch),
            bars.mb_k_full[k_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
        tma_load_tile(
            sV[v_empty_state.idx],
            tma_v(cutlass.Int32(0), kv_head_idx, kv_block_base + K_ROW_OFFSET_PEER, full_batch),
            bars.mb_v_full[v_empty_state.idx].smem_ptr,
            cta_group=CFG.CTA_MMA,
            mcast_mask=tma_mcast_mask,
        )
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

        # K-split alias seam: sdO_dv stage 1 IS the K back-half, so no dO_dv load (hence the
        # whole q loop) may start before the leader MMA has UTCCP'd it into TMEM.  Once per
        # block; within-block dO_dv prefetch still pipelines.  No deadlock against mb_q_full:
        # the MMA issues UTCCP -> commit before it needs Q.
        bars.mb_k_utccp_done.wait(utccp_done_state.phase, spin=SPIN_RING_WAITS)
        utccp_done_state = advance(utccp_done_state, 1)

        # ---- Q + dO + dO_dv, per q tile of the mask-bounded range ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_N)

            bars.mb_q_empty[q_empty_state.idx].wait(q_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_q_full[q_empty_state.idx].arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sQ[q_empty_state.idx],
                tma_q(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, full_batch),
                bars.mb_q_full[q_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)

            bars.mb_do_empty[do_empty_state.idx].wait(do_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_do_full[do_empty_state.idx].arrive(n_bytes=dOTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO[do_empty_state.idx],
                tma_do(cutlass.Int32(0), full_head, q_row_base + Q_ROW_OFFSET_PEER, full_batch),
                bars.mb_do_full[do_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            do_empty_state = advance(do_empty_state, CFG.STAGES_dO)

            bars.mb_dodv_empty[dodv_empty_state.idx].wait(dodv_empty_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_dodv_full[dodv_empty_state.idx].arrive(n_bytes=dOdvTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            tma_load_tile(
                sdO_dv[dodv_empty_state.idx],
                tma_do_dv(DO_DV_OFFSET_PEER, full_head, q_row_base, full_batch),
                bars.mb_dodv_full[dodv_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            dodv_empty_state = advance(dodv_empty_state, CFG.STAGES_dO_DV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head_idx = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: drain EVERY cross-CTA _empty ring this warp consumes, OUTSIDE the persistent loop.
    # The leader's last multicast commits are still in flight when it leaves its loop; this
    # CTA must stay resident until they land (else CUDA_EXCEPTION_17 at teardown).  Balanced:
    # T pre-armed waits consumed commits 0..T-2 (the first wait was free), the drain takes T-1.
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


@cute.jit
def _mma_warp(
    sQ,
    sdO,
    sdO_dv,
    sK,
    sV,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    batch_base,
    mcast_mask,
) -> None:
    """MMA leader -- the NATURAL-order 3-matmul stream.

    Per kv block (K, V once): UTCCP K back-half -> TMEM, commit mb_k_utccp_done; then per
    q tile i:  Q.K[i] (mma_ss front + mma_ts back) ; dO.V[i] ; P.dO[i] (accumulate = i > q_lo).
    Handshakes: mb_s_acc_full / mb_dp_full (-> softmax), mb_dp_empty (dSoftmax read dP,
    pre-armed), mb_p_ready (P in S_acc), mb_dv_ready / mb_dv_acc_empty (epilogue), the
    *_empty commits (-> TMA-LDG).  No s_acc_empty and no p_empty: the in-order pipeline
    issues Q.K[i+1] (the S_acc writer) after P.dO[i] (the P reader), which waited
    mb_p_ready[i], arrived after softmax[i]'s S read completed.
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    # Publish tmem_ptr_i32 to the compute warps (their barrier_cta_sync(1) at top of body).
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()
    dodv_full_state = PipelineState.start()
    dp_empty_state = PipelineState.start(phase=1)  # pre-armed: dO.V[q_lo] finds the dP slot free
    p_ready_state = PipelineState.start(phase=0)
    dv_empty_state = PipelineState.start(phase=0)

    kv_super_idx, _head_idx, batch_idx = _boot_tile(sched)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_OFF))
    tmem_P = tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))  # half P inside S_acc (mma_ts A of BMM2)
    tmem_K_back = tmem_raw.subview(cutlass.Int32(LAYOUT.RSVD_OFF))  # UTCCP'd K d_qk[128:256]

    # ---- instruction / operand descriptors (k_dim = CFG.IDESC_K_DIM, the f16 default 0) ----
    # BMM1 S = K . Q^T (A = K, M = kv; B = Q, N = q; K = d_qk).  The idesc shape is K-independent,
    # so the K-split half desc below reuses it.
    idesc_bmm1_s = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=CFG.IDESC_K_DIM,
    )
    # BMM1 dP = V . dO^T (A = V, M = kv; B = dO, N = q; K = d_v).
    idesc_bmm1_dp = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=CFG.IDESC_K_DIM,
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
    # BMM2 dV = P . dO (A = P in TMEM, M = kv, K = q; B = dO_dv, N = d_v, BT).
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
    # BMM1 S K-split: ONE K=128 MmaDesc drives BOTH the front mma_ss (A = K front, SMEM) and the
    # back mma_ts (A = K back-half, TMEM); the operand source is the call, not the desc.
    # tmem_advance_A = TILE_K_HW * BPE / 4 = 8 cols per k-step x 8 steps = 64 cols = RSVD_COLS.
    bmm1_s_half_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=K_SPLIT_TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_bmm1_s,
        kind=MMA_KIND,
    )
    # Subtile-major UTCCP walk (tcgen05.cp.128x128b, 16 B per call = 4 TMEM cols): 16 calls
    # cover the 64-col K back-half; desc offsets are in 16-B units, subtile-major.
    _UTCCP_BYTES_PER_CALL = 16
    _UTCCP_TMEM_COLS_PER_CALL = _UTCCP_BYTES_PER_CALL // 4
    _UTCCP_PER_SUBTILE = CFG.K_SWZ_BYTES // _UTCCP_BYTES_PER_CALL
    _UTCCP_LOG2_PER_SUBTILE = _UTCCP_PER_SUBTILE.bit_length() - 1  # the runtime loop below splits the call index by shift / mask
    _UTCCP_SUBTILE_DESC_STRIDE = (CFG.TILE_M * CFG.K_SWZ_BYTES) // _UTCCP_BYTES_PER_CALL
    _UTCCP_N_CALLS = LAYOUT.RSVD_COLS // _UTCCP_TMEM_COLS_PER_CALL

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)

        # K + V once per kv block.
        bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase, spin=SPIN_RING_WAITS)
        # K / V descriptor bases, pinned per kv block (desc_opaque + the kv-block counter as the
        # loop-variant anchor).  They are statically addressed (static SMEM, and STAGES_KV = 1
        # folds the ring index to 0), so unpinned every `base + k_step` of the k-loops below is
        # KERNEL-invariant and LLVM hoists all of them to the persistent loop's preheader, live for
        # the whole kernel; pinned, the 8 + 8 adds live inside this kv block only.
        desc_K = desc_opaque(sK[k_full_state.idx].desc(), anchor=kv_super_idx)
        desc_V = desc_opaque(sV[v_full_state.idx].desc(), anchor=kv_super_idx)

        # K-split: UTCCP K's d_qk[128:256] (subtiles 2, 3) SMEM -> TMEM [RSVD], then commit so the
        # TMA-LDG learns the K back-half SMEM is dead (sdO_dv stage-1 alias seam).  Leader-only:
        # a cga2 tcgen05.cp self-fills BOTH peers' TMEM from their OWN SMEM.  The loop AND its
        # commit sit in ONE elected block (one lane issues both -> init 1 per CTA).
        #
        # A RUNTIME loop, deliberately not range_constexpr: unrolled, the 16 `base + off` 64-bit
        # uniform adds are all front-loaded by ptxas (the PTX has each add right before its
        # tcgen05.cp; the SASS had all 16 at the top of the kv block), which together with the
        # K / V k-step descriptors overflows the 63-UR file and parks them through GPR spill slots
        # -- measured 38 STL / 45 LDL in this 40-register warp (2026-09-23, sm_107a).  One
        # descriptor in flight per iteration keeps it flat; once per 256-row kv block, so the loop
        # overhead is nil against the 16 UTCCPs it issues.
        elect_p = nvvm.elect_sync()
        if elect_p:
            desc_K_back = desc_opaque(sK[k_full_state.idx].shifted(K_BACK_OFF_ELEMS).desc(), anchor=kv_super_idx)
            for _qk in cutlass.range(0, _UTCCP_N_CALLS, 1, unroll=1):
                _s = _qk >> cutlass.Int32(_UTCCP_LOG2_PER_SUBTILE)
                _kk = _qk & cutlass.Int32(_UTCCP_PER_SUBTILE - 1)
                _desc_off = _s * cutlass.Int32(_UTCCP_SUBTILE_DESC_STRIDE) + _kk
                nvvm.tcgen05_cp(
                    nvvm.Tcgen05CpShape.SHAPE_128X128B,
                    tmem_K_back.subview(_qk * cutlass.Int32(_UTCCP_TMEM_COLS_PER_CALL)),
                    desc_K_back + _desc_off,
                    group=CTA_GROUP_KIND,
                )
            bars.mb_k_utccp_done.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA)

        # ---- NATURAL order: Q.K[i] ; dO.V[i] ; P.dO[i] ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # BMM1 S = K_front . Q_front^T (mma_ss, overwrite) + K_back . Q_back^T (mma_ts,
            # accumulate) = exact K . Q^T.  Both consume sQ[stage], so q_empty fires after the back half.
            bars.mb_q_full[q_full_state.idx].wait(q_full_state.phase, spin=SPIN_RING_WAITS)
            mma_ss(bmm1_s_half_desc, desc_K, sQ[q_full_state.idx].desc(), tmem_S, accumulate=False)
            mma_ts(bmm1_s_half_desc, tmem_K_back, sQ[q_full_state.idx].shifted(Q_BACK_OFF_ELEMS).desc(), tmem_S, accumulate=True)
            elect_p = nvvm.elect_sync()
            bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_q_empty[q_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            q_full_state = advance(q_full_state, CFG.STAGES_Q)

            # BMM1 dP = V . dO[i]^T; the single dP slot is gated by dSoftmax[i-1]'s read (pre-armed for i = q_lo).
            bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase, spin=SPIN_RING_WAITS)
            dp_empty_state = advance(dp_empty_state, CFG.STAGES_TMEM_S)
            bars.mb_do_full[do_full_state.idx].wait(do_full_state.phase, spin=SPIN_RING_WAITS)
            mma_ss(bmm1_dp_desc, desc_V, sdO[do_full_state.idx].desc(), tmem_dP)
            elect_p = nvvm.elect_sync()
            bars.mb_dp_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_do_empty[do_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            do_full_state = advance(do_full_state, CFG.STAGES_dO)

            # BMM2 dV += P[i] . dO[i] (A = P in S_acc[P_OFF]).  accumulate restarts at q_lo (NOT 0):
            # the first q tile of every kv block OVERWRITES dV_acc (rules/frost-tile-dsl.md S2).
            bars.mb_p_ready[p_ready_state.idx].wait(p_ready_state.phase, spin=SPIN_RING_WAITS)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)
            bars.mb_dodv_full[dodv_full_state.idx].wait(dodv_full_state.phase, spin=SPIN_RING_WAITS)
            mma_ts(bmm2_dv_desc, tmem_P, sdO_dv[dodv_full_state.idx].desc(), tmem_dV, accumulate=(q_iter > q_lo))
            elect_p = nvvm.elect_sync()
            bars.mb_dodv_empty[dodv_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            dodv_full_state = advance(dodv_full_state, CFG.STAGES_dO_DV)

        # dV complete for this kv block -> epilogue (compute warps); wait for it to drain dV_acc
        # before the next block's P.dO[q_lo] (accumulate=False) overwrites it.
        bars.mb_dv_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=nvvm.elect_sync())
        bars.mb_dv_acc_empty[dv_empty_state.idx].wait(dv_empty_state.phase, spin=SPIN_RING_WAITS)
        dv_empty_state = advance(dv_empty_state, 1)

        # End of block: release K + V (after every MMA that reads them has committed in order).
        elect_p = nvvm.elect_sync()
        bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        k_full_state = advance(k_full_state, CFG.STAGES_KV)
        v_full_state = advance(v_full_state, CFG.STAGES_KV)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, _head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: the pre-armed mb_dp_empty consumer leaves the LAST dSoftmax's batch (512 arrives,
    # 256 of them DSMEM from the follower) un-waited; drain it so this CTA is resident when
    # they land.  One wait per kernel.
    bars.mb_dp_empty[dp_empty_state.idx].wait(dp_empty_state.phase)

    # ---- TMEM dealloc (the wait IS the drain of both CTAs' compute-lane arrives) ----
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    """Follower CTA's MMA warp: collective tmem_alloc (the leader's cga2 MMAs write this CTA's
    TMEM half), publish the pointer, wait mb_tmem_dealloc, release TMEM.  No persistent loop,
    no credit arrive (READ_TILE_ARRIVERS_TOT counts it out)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _softmax_warp_group(
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sdS_raw,
    sStats_raw,
    sdV_raw,
    seq_kv_lens_tensor,
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
    """8 compute warps (2 wg x 4), lane = kv row; wg0 -> q[0:64], wg1 -> q[64:128].

    Per q tile:  softmax  -- wait mb_s_acc_full; tmem_load S[wg q-half]; P = exp2(scale*log2e*S -
                 lse*log2e); mask; half P -> tcgen05_st S_acc[P_OFF + wg half]; arrive mb_p_ready.
                 dSoftmax -- wait mb_dp_full; tmem_load dP[wg q-half]; arrive mb_dp_empty;
                 dS = (dP*scale - do_dot) * P; half dS -> sdS ring (store_swizzled); arrive
                 mb_ds_smem_full + mb_stats_empty.
    Per kv block: dV epilogue -- wait mb_dv_ready; tmem_load dV[wg d_v half] -> OUT dtype -> sdV;
                 arrive mb_dv_stg_full; arrive mb_dv_acc_empty.
    Each wg's P write stays inside its OWN S-read q-half, so the S tmem_load -> P tmem_st needs
    no cross-wg sync; a tcgen05_wait(LOAD) precedes every arrive that frees a slot.
    """
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    kv_super_idx, _head_idx, batch_idx = _boot_tile(sched)

    tid_in_wg = cute.arch.thread_idx()[0] & cutlass.Int32(CFG.SOFTMAX_WG_LANES - 1)  # this lane's kv row within its wg
    wg_id = (warp_idx - cutlass.Int32(CFG.SOFTMAX_WG0_BASE)) // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    q_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)  # q col offset of this wg's half
    p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)  # half P TMEM col offset within S_acc[P_OFF..]
    # per-lane absolute kv row base (this CTA's M slice of the pair's kv block)
    kv_lane_base0 = cta_in_pair * cutlass.Int32(CFG.TILE_M) + tid_in_wg

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()
    dp_full_state = PipelineState.start()
    ds_empty_state = PipelineState.start(phase=1)  # dS ring slot free (pre-armed)
    p_ready_state = PipelineState.start()
    dv_ready_state = PipelineState.start()
    stats_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        tmem_base = tmem_ptr_i32.load()

        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real)
        causal_diag = _causal_diag(seqlen_q_real, eff_seqlen_kv)
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)
        kv_abs = kv_block_base + kv_lane_base0

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            # ---- stats ring: this q tile's lse / do_dot (prefetched by the scheduler warp) ----
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase, spin=SPIN_RING_WAITS)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS)

            # ---- 1) softmax: S[wg q-half] -> P ----
            bars.mb_s_acc_full[s_full_state.idx].wait(s_full_state.phase, spin=SPIN_RING_WAITS)
            s_full_state = advance(s_full_state, CFG.STAGES_TMEM_S)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            lse_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw.subview(stats_base + cutlass.Int32(STATS_LSE_OFF + i) + q_half_off).load() for i in range(_SMX_CHUNK)),
                cutlass.Float32,
            )
            chunk_P = cute.math.exp2(reg_S.vec * attn_scale_log2e - lse_vec, fastmath=True)
            # Mask (the transpose of the forward): zero P on masked (kv = lane, q = col) cells so the
            # half P (dV) AND dS (dK / dQ) inherit it.  No IR at MASK_NONE.
            if cutlass.const_expr(CFG.MASK_FLAGS != MASK_NONE):
                chunk_P = _mask_p_chunk(chunk_P, kv_abs, q_iter * cutlass.Int32(CFG.TILE_N) + q_half_off, eff_seqlen_kv, causal_diag, _SMX_CHUNK)

            # ---- 2) half P -> S_acc[P_OFF + p_col_off]; notify the MMA EARLY so BMM2 overlaps the dSoftmax ----
            chunk_P_half = chunk_P.to(STORAGE_DTYPE)
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_col_off, cutlass.Float32), chunk_P_half)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bars.mb_p_ready[p_ready_state.idx].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            p_ready_state = advance(p_ready_state, CFG.STAGES_TMEM_P)

            # ---- 3) dSoftmax: dP -> free the slot -> dS = (scale*dP - do_dot) * P -> sdS ring ----
            bars.mb_dp_full[dp_full_state.idx].wait(dp_full_state.phase, spin=SPIN_RING_WAITS)
            dp_full_state = advance(dp_full_state, CFG.STAGES_TMEM_S)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=_LDTM_NUM)
            # The LOAD wait ORDERS the arrive after the TMEM read (an arrive with no data dependency
            # on the loaded registers may otherwise be scheduled between two LDTM chunks).
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.mb_dp_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            dot_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw.subview(stats_base + cutlass.Int32(STATS_DOT_OFF + i) + q_half_off).load() for i in range(_SMX_CHUNK)),
                cutlass.Float32,
            )
            chunk_dS = (reg_dP.vec * attn_scale - dot_vec) * chunk_P
            chunk_dS_half = chunk_dS.to(DS_STORAGE_DTYPE)
            ds_slot = ds_empty_state.idx
            bars.mb_ds_smem_empty[ds_slot].wait(ds_empty_state.phase, spin=SPIN_RING_WAITS)
            ds_empty_state = advance(ds_empty_state, CFG.XFER_STAGES)
            # dS SMEM [kv, q], 128 B swizzled: a 256 B row = P_TMA_ITERS subtiles of P_D_BLOCK q cols; each
            # wg owns ONE subtile (wg0 -> q[0:64], wg1 -> q[64:128]) at slab wg * P_BLOCK_ELEMS, row tid * P_D_BLOCK.
            sdS_raw.subview(
                ds_slot * cutlass.Int32(dSBufferElems) + wg_id * cutlass.Int32(P_BLOCK_ELEMS) + tid_in_wg * cutlass.Int32(P_D_BLOCK)
            ).data_ptr().store_swizzled(chunk_dS_half, alignment=128, swizzle=P_SMEM_SWIZZLE)
            nvvm.fence_proxy("async.shared", space="cta")  # generic SMEM stores -> async-proxy TMA store
            bars.mb_ds_smem_full[ds_slot].arrive()
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, CFG.STATS_STAGES)

        # ---- dV epilogue (per kv block): dV_acc TMEM -> OUT dtype -> sdV (aliases sK) ----
        # sdV is 128 B swizzled: TILE_O d_v = TMA_DV_ITERS subtiles of DV_D_BLOCK cols; each wg owns
        # TMA_DV_ITERS / SOFTMAX_WARPGROUPS of them, block b at slab b * DV_BLOCK_SLAB, row tid * DV_D_BLOCK.
        bars.mb_dv_ready.wait(dv_ready_state.phase, spin=SPIN_RING_WAITS)
        dv_ready_state = advance(dv_ready_state, 1)
        _DV_BLOCKS_PER_WG = TMA_DV_ITERS // CFG.SOFTMAX_WARPGROUPS
        for _b in cutlass.range_constexpr(_DV_BLOCKS_PER_WG):
            gblk = wg_id * cutlass.Int32(_DV_BLOCKS_PER_WG) + cutlass.Int32(_b)
            reg_dV = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dV_OFF) + gblk * cutlass.Int32(DV_D_BLOCK), num_elems=DV_D_BLOCK)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            dV_out = reg_dV.vec.to(OUT_STORAGE_DTYPE)
            sdV_raw.subview(gblk * cutlass.Int32(DV_BLOCK_SLAB) + tid_in_wg * cutlass.Int32(DV_D_BLOCK)).data_ptr().store_swizzled(
                dV_out, alignment=128, swizzle=P_SMEM_SWIZZLE
            )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dv_stg_full.arrive()
        # dV_acc read -> the next block's P.dO[q_lo] (accumulate=False) may overwrite it.
        bars.mb_dv_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, _head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # ---- release the MMA warps' TMEM alloc on BOTH CTAs (bare: every compute lane, local + peer) ----
    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


@cute.jit
def _tmastg_warp(
    tma_dv_desc,
    tma_ds_desc,
    sdV,
    sdS,
    bars,
    sched,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_q_real,
    seqlen_kv_real,
    head_base,
    batch_base,
    cta_in_pair,
) -> None:
    """TMA-store warp (both CTAs): per q tile the dS slot -> workspace [B_chunk, H_chunk, S_kv, S_q]
    (wait mb_ds_smem_full -> store -> commit / wait -> arrive mb_ds_smem_empty); per kv block dV ->
    [B, S_kv, H_q, d_v] (wait mb_dv_stg_full -> store -> commit / wait -> arrive mb_dv_stg_empty).
    The lse / do_dot prefetch lives on the scheduler warp so it overlaps these stores.
    """
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_ds = GmemTileTma(tma_ds_desc)

    kv_super_idx, head_idx, batch_idx = _boot_tile(sched)
    full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)
    KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_M)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    ds_full_state = PipelineState.start()
    dv_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        kv_block_base = kv_super_idx * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, batch_base, seqlen_kv_real)
        # Only the in-range q tiles produce dS; the skipped tiles' workspace regions must be zero
        # (the adapter zeroes them under a mask) so dK / dQ = dS . Q / dS^T . K stay correct.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)

        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            ds_slot = ds_full_state.idx
            bars.mb_ds_smem_full[ds_slot].wait(ds_full_state.phase, spin=SPIN_RING_WAITS)
            ds_full_state = advance(ds_full_state, CFG.XFER_STAGES)
            # workspace [B_chunk, H_chunk, S_kv, S_q] -> coords innermost-first (S_q, S_kv, H, B), chunk-local head / batch
            tma_store_tile(sdS[ds_slot], tma_ds(q_col_base, kv_block_base + KV_ROW_OFFSET_PEER, head_idx, batch_idx))
            tma_store_commit()
            tma_store_wait(0)
            if nvvm.elect_sync():
                bars.mb_ds_smem_empty[ds_slot].arrive()

        # dV -> the FULL output [B, S_kv, H_q, d_v] (full-tensor head / batch).
        bars.mb_dv_stg_full.wait(dv_full_state.phase, spin=SPIN_RING_WAITS)
        dv_full_state = advance(dv_full_state, 1)
        tma_store_tile(sdV[0], tma_dv(cutlass.Int32(0), head_idx + head_base, kv_block_base + KV_ROW_OFFSET_PEER, full_batch))
        tma_store_commit()
        tma_store_wait(0)  # source SMEM reads done -> the TMA-LDG may reload K over sdV
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()

        # ---- scheduler tail ----
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_t0, nxt_t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        is_valid_tile = cute.arch.make_warp_uniform(nxt_v)
        kv_super_idx, head_idx, batch_idx = _decode_tile_words(sched, nxt_t0, nxt_t1)
        full_batch = cute.arch.make_warp_uniform(batch_idx + batch_base)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


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
    head_base,
    batch_base,
) -> None:
    """Persistent tile scheduler (CLC try_cancel protocol) FUSED with the lse / do_dot prefetch.

    Per loop iteration: (A) fill the sStats ring for the tile the consumers are CURRENTLY
    processing (one slot per q tile: lse * log2e | do_dot * attn_scale, TILE_N each; the
    consumers read the SAME slot broadcast on every lane), tracked one tile behind the
    scheduling; (B) the canonical CLC protocol for the NEXT tile (tile_dsl.scheduler
    .scheduler_warp_loop): the cluster's first CTA arms expect_tx on EVERY CTA's mb_scheduler,
    program-ordered before its multicast try_cancel.  This warp never credits
    mb_read_tile_id (it is the waiter), so READ_TILE_ARRIVERS_TOT excludes it.
    """
    lse_arr = cutlass.make_array_view(lse_tensor)
    dot_arr = cutlass.make_array_view(do_dot_tensor)
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    _PER_LANE = CFG.TILE_N // 32
    log2e = cutlass.Float32(_LOG2E)

    cur_kv_super, cur_head, cur_batch = _boot_tile(sched)

    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        cur_full_batch = cute.arch.make_warp_uniform(cur_batch + batch_base)
        kv_block_base = cur_kv_super * cutlass.Int32(_KV_BLOCK_ROWS)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, cur_batch, batch_base, seqlen_kv_real)
        # The SAME [q_lo, q_hi) the consumers walk, so the ring fill count matches their waits.
        q_lo, q_hi = _q_loop_bounds(kv_block_base, seqlen_q, seqlen_q_real, eff_seqlen_kv)

        # ---- (A) stats prefetch for the CURRENT tile ----
        for q_iter in cutlass.range(q_lo, q_hi, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_N)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase, spin=SPIN_RING_WAITS)
            stats_empty_state = advance(stats_empty_state, CFG.STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_LSE_OFF) + col).store(lse_arr[cur_full_batch, cur_full_head, q_col_base + col] * log2e)
                sStats_raw.subview(slot_base + cutlass.Int32(STATS_DOT_OFF) + col).store(dot_arr[cur_full_batch, cur_full_head, q_col_base + col] * attn_scale)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        # ---- (B) CLC protocol for the NEXT tile ----
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
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
        cur_kv_super, cur_head, cur_batch = _decode_tile_words(sched, nxt_t0, nxt_t1)
        state = advance(state, CFG.SCHEDULER_STAGES)


# ============================================================================
# Host launcher
# ============================================================================


def _tma_swz(byte_w: int):
    return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b


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
    problem_size: Tuple[int, int, int, int, int, int, int, int, int],  # (B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL)
    attn_scale: cutlass.Float32,
    head_base: cutlass.Int32,
    batch_base: cutlass.Int32,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK, B_CHUNK, SQ_REAL, SKV_REAL = problem_size

    # [B, S, H, D] operands: box = (1 batch, S rows, 1 head, D cols); stride_order innermost-first
    # so the kernel's coords are (d, head, seq, batch).  Workspace [B, H, S_kv, S_q]: (q, kv, head, batch).
    stride_order = (3, 2, 1, 0)
    qk_box_q = (1, _M_PER_CTA, 1, TMA_QK_GRANU_ELEMS)  # Q, dO (dP view): per-CTA q half
    qk_box_k = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)  # K, V: per-CTA kv half
    do_box = (1, _M_PER_CTA, 1, TMA_VO_GRANU_ELEMS)
    v_box = (1, CFG.TILE_M, 1, TMA_VO_GRANU_ELEMS)
    # dO dV view (BT): full TILE_N q rows x the 64-col swizzle granule, TMA_VO_SG1_ITERS subtiles per CTA half.
    do_dv_box = (1, CFG.TILE_N, 1, TMA_VO_SG1_GRANU_ELEMS)
    dv_box = (1, CFG.TILE_M, 1, TMA_DV_GRANU_ELEMS)  # dV store: 128 B swizzled subtile
    ds_box = (1, 1, CFG.TILE_M, P_D_BLOCK)  # dS store: TILE_M kv x 64 q per subtile (a 256 B row exceeds the atom)

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
    # Stores: the SMEM side is written with the 128 B store_swizzled pattern, so both descriptors are s128b.
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(
        dv_tensor, box_dims=dv_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dV_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )
    tma_ds_desc = tmap.create_tensor_map_tiled_from_view(
        ds_tensor, box_dims=ds_box, stride_order=stride_order, swizzle=_tma_swz(CFG.dS_SWZ_BYTES), l2_promotion=tmap.TensorMapL2Promotion.l2_128b
    )

    # One cga2 cluster per (kv block, head, batch) tile of this chunk.  NATURAL: 3-D grid; LPT /
    # LPT_L2: a flat 1-D grid decoded kv-super-outermost.  Same persistent try_cancel scheduler.
    kv_blocks = (SKV + _KV_BLOCK_ROWS - 1) // _KV_BLOCK_ROWS
    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL):
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
        cutlass.Int32(B_CHUNK),
        cutlass.Int32(QH_CHUNK),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        batch_base,
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
    b_chunk: int = 0,
    sq_real: int = 0,
    skv_real: int = 0,
) -> Callable:
    """Compile the main kernel with ALL extents concrete (pins the TMA strides); see '## Launch ABI'.

    ``qh_chunk`` (default ``qh``) / ``b_chunk`` (default ``b``) size the dS workspace's head /
    batch extents and the grid; the runtime ``head_base`` / ``batch_base`` walk the chunks so
    one artifact serves every launch.  ``sq_real`` / ``skv_real`` (default = the padded
    extents) are the real lengths the bottom-right diagonal uses.
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
    if sq % CFG.TILE_N != 0:
        raise ValueError(f"bwd d256 f16: S_q must be a multiple of the q tile ({CFG.TILE_N}); got {sq} -- the adapter pads")
    if skv % _KV_BLOCK_ROWS != 0:
        raise ValueError(f"bwd d256 f16: S_kv must be a multiple of the cga2 kv block ({_KV_BLOCK_ROWS}); got {skv} -- the adapter pads")
    if not (0 < sq_real <= sq and 0 < skv_real <= skv):
        raise ValueError(f"bwd d256 f16: real lengths must satisfy 0 < sq_real <= sq, 0 < skv_real <= skv; got {sq_real}/{sq}, {skv_real}/{skv}")
    validate_head_chunk(qh, kh, qh_chunk)
    if b_chunk < 1 or b % b_chunk != 0:
        raise ValueError(f"bwd d256 f16: b_chunk ({b_chunk}) must be a positive divisor of B ({b})")

    def _fake_bshd(shape, dtype):
        return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)

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
    return _compile_cached(
        _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dv,
        fake_ds,
        fake_lse,
        fake_do_dot,
        fake_seq_kv_lens,
        (b, qh, kh, sq, skv, qh_chunk, b_chunk, sq_real, skv_real),
        cutlass.Float32(0.0),  # attn_scale
        cutlass.Int32(0),  # head_base
        cutlass.Int32(0),  # batch_base
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_bwd_d256_f16",
    )


__all__ = [
    "CFG",
    "PARAMS",
    "LAYOUT",
    "Bars",
    "DESC_VERSION",
    "SPIN_RING_WAITS",
    "STORAGE_DTYPE",
    "OUT_STORAGE_DTYPE",
    "DS_STORAGE_DTYPE",
    "_kernel",
    "_host",
    "compile",
]
