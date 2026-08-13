# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Chunked Gated Delta Net (GDN) BPROP kernel for Blackwell SM100 (Cutlass primitives).

Algorithm overview (per chunk c, iterated c = NT-1 .. 0):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], dO[BT,DV], S_c[DK,DV] (= H[c-1],
           the forward state ENTERING chunk c), gate[BT], beta[BT]
  State  : dH[DK,DV]  (state gradient, held in TMEM, accumulated backward)

  MMA order.  A = the staged attention matrix
  (CG0's qk epilogue, sQk); dO' = dO * gCumprod * scale (CG1 restage):
  kk          : W_kk[BT,BT]  = K  @ K^T        -> shared acc   (for T)
  qk          : W_qk[BT,BT]  = Q  @ K^T        -> shared acc
  dV inter    : dV[DV,BT]    = dH^T(TMEM) @ K^T -> the dV/dK slot
                (acc=False; CG1 then scales it by gDecayScale IN PLACE)
  ks          : KS[BT,DV]    = S^T(SMEM) @ K^T  -> shared acc
  dU intra    : dV/dK slot  += dO^T(SMEM) @ A(SMEM)  (waits CG1's
                in-place scale -> the accumulate is the reference's
                dv2 = dv_intra + exp2(g_last-g)*(k@dh))
  dH q-term   : dH          += dO'^T(TMEM) @ Q  (acc=False on the
                first backward chunk = dH init; the reference's
                dh = dh*exp2(g_last) + (q*scale*exp2(g))^T @ do)
  dY          : dY[DV,BT]    = dU^T(TMEM f16) @ T(SMEM) -> shared acc
                (dY == dV == d(delta); T read TRANSPOSED vs prefill)
  dA          : dA_eff[BT,BT]= dO(SMEM) @ delta^T(SMEM) -> shared acc
                (sV holds delta; CG0 masks it -> sDa for dQ/dK)
  dM core     : dM[BT,BT]    = dY(SMEM) @ U^T(SMEM) -- the WY
                inverse backward T^T dT T^T collapsed via
                T^T dU = dY and T Y = U (beta folds through
                sAinv's column scale into both factors); CG0
                applies -strict(2^{g_i-g_j}) -> sDm (the reference's dA22;
                both betas cancel against the K rows in the dM-terms)
  dH ds-term  : dH          += dY'^T(TMEM f16) @ K  (dY' = -gCumprod
                * dY: the -(w^T @ d(delta)) term; ready commits HERE)
  dK dM-terms : dV/dK slot  += K^T @ dM^T + K^T @ dM  (the reference's
                dk += dA22^T@K_beta / dk_beta += dA22@K folded, the
                beta row scale cancelling with K_beta = beta*K)
  dK s-path   : SY[BT,DV]    = S^T(SMEM) @ dY^T(SMEM) -> the shared f16
                input columns (dO'/dU/dY' dead by then); CG1 reads it as
                -gCumprod[t] * (dY @ S^T)[t,:] and adds it to the banked
                inter+attn dK terms

  Per chunk CG1: restages dO' and dU and dY' -> the f16 input columns
  (dY' overwrites dU), computes delta = V - gCumprod*(K @ S) in registers and stages
  Y (= delta) and gks to their dedicated f16 TMEM slots (the u-GEMM and
  the dV-pass read them), stages Q^T over the Y slot after the dV pass,
  loads dY and stages it plain to sdV (the dV output).

SMEM layout (~226 KB of the 227 KB SM100 cap):
  Buffer                           Size (B)  Stages
  q                                16384     1
  k                                32768     2      <-- double-buffered (prefetch next chunk)
  v                                16384     1      <-- overwritten in place by u
  dO                               16384     1
  S (forward state H[c-1])         32768     1      <-- bf16 [DK,DV], TMA-loaded
  A_inverse / T                    8192      1      <-- inverse OUTPUT (upper tri = kernel-start zeros)
  KK (pristine M_kk)               8192      1      <-- kk_epi's only store; inverse input + dG/dBeta
  QK staging / sDa                 8192      1      <-- ALIAS: A then the masked dA
  dM staging (sDm)                 8192      1      <-- Step 8 -> dK dM-terms
  dH_entry (sdH)                   32768     1      <-- f16 restage, dK-inter's A
  dQ store staging                 16384     1
  dK store staging                 16384     1
  dV store staging                 16384     1
  cumsumlog / cumprod / beta       3 x 512   2      <-- in-place dG/dBeta staging

TMEM layout (512 cols; EXACTLY the prefill map, S->dH, O->dV):
  cols 0-128   : dH accumulator (fp32)         <-- prefill: state (S)
  cols 128-192 : dV/dK accumulator (fp32)      <-- one slot, five sequential
                 per-chunk productions (dV inter -> dU intra -> dK inter
                 -> dK attn -> dK dM-terms), each with its own mbar pair
  cols 192-256 : dH input (f16 packed)         <-- prefill: state_inp
  cols 256-384 : shared accumulators x2        <-- kk / qk / ks / u / dY / dA
                                                    / dM core
  cols 384-448 : shared inputs x2 (f16 packed) <-- dO' / dU / dY'; the dK
                 s-path acc overwrites them after their last GEMM reads
                 (CG1's s-path readout precedes its next-chunk restages)
  cols 448-512 : Y (448) + gks (480) f16 slots until the dV pass, then
                 Q^T (448) until CG1's dQ dot reads it

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise, kk_epi, qk_epi, inverse
  warps 4-7     : compute group 1 - dV epilogue, dH scale + f16 restage
                                    (later: V-K*S -> SMEM, dY staging)
  warp  8       : MMA warp       - issues the GEMMs
  warp  9       : TMA load warp  - loads q, k (double-buf), v, dO, S(=H)
  warp  10      : gate warp      - loads gate/beta (double-buffered,
                                    backward order) + stores dG/dBeta
  warp  11      : epilogue warp   - store dQ, dK, dV to global memory
"""

import functools
from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as _tma
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import min as _cutlass_min

from ..common.thd import build_h_descs_kernel, build_qkv_load_descs_kernel, build_state_descs_kernel, TENSOR_MAP_QWORDS
from ..common.split_k import decode_work_item

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
    arrive,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts, mma_step
from cudnn.frost.tile_dsl.pointwise import fp32_to_fp16, f16x2_to_f32, fmul2, fadd2, ffma2, opaque_f32_zero, sub_f16x2
from cudnn.frost.tile_dsl.swizzle import swizzle_lin_128b, swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .gdn_bprop_config import CFG


class GdnBwdBars(NamedTuple):
    """GDN bprop pipeline mbarrier inventory.

    Every pipeline is a ``_ready``/``_done`` MBarrier pair over one ring: a
    slot is acquired for filling by waiting ``_done`` and committed by
    arriving ``_ready``; the reading side waits ``_ready`` and releases the
    slot by arriving ``_done``.

    Operand buffers read by both the MMA warp and a compute group carry a
    SPLIT done pair (``_mma_done`` MMA_COMMIT + ``_cg?_done`` THREAD) and the
    TMA warp waits BOTH: a plain arrive from the MMA warp fires at MMA issue,
    not completion, which would release the buffer for reload mid-GEMM.
    """

    mb_q_ready: MBarrier
    mb_q_mma_done: MBarrier
    mb_q_cg1_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_mma_done: MBarrier
    mb_k_cg0_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_mma_done: MBarrier
    mb_v_cg1_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_do_cg1_done: MBarrier
    mb_s_ready: MBarrier
    mb_s_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_dh_acc_ready: MBarrier
    mb_dh_acc_done: MBarrier
    mb_du_scale_ready: MBarrier
    mb_du_scale_done: MBarrier
    mb_du_total_ready: MBarrier
    mb_dk_scale_ready: MBarrier
    mb_dk_scale_done: MBarrier
    mb_dk_attn_ready: MBarrier
    mb_dk_attn_done: MBarrier
    mb_dk_total_ready: MBarrier
    mb_dk_total_done: MBarrier
    mb_kk_acc_ready: MBarrier
    mb_kk_acc_done: MBarrier
    mb_a_acc_ready: MBarrier
    mb_ks_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier

    mb_ainv_ready: MBarrier
    mb_ainv_done: MBarrier
    mb_qk_ready: MBarrier
    mb_qk_done: MBarrier
    mb_dh_inp_ready: MBarrier
    mb_dh_inp_done: MBarrier
    mb_dop_inp_ready: MBarrier
    mb_dop_inp_done: MBarrier
    mb_du_inp_ready: MBarrier
    mb_dyp_inp_ready: MBarrier
    mb_dyp_inp_done: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier

    mb_y_ready: MBarrier
    mb_sdv_done: MBarrier
    mb_u_ready: MBarrier
    mb_dhs_ready: MBarrier
    mb_dhs_done: MBarrier
    mb_da_ready: MBarrier
    mb_dq_acc_scale_ready: MBarrier
    mb_dq_acc_scale_done: MBarrier
    mb_dq_acc_total_ready: MBarrier
    mb_dq_acc_total_done: MBarrier
    mb_da_acc_ready: MBarrier
    mb_dm_ready: MBarrier
    mb_dm_done: MBarrier
    mb_dbeta_cg1_ready: MBarrier
    mb_dgate_cg1_ready: MBarrier
    mb_hdh_done: MBarrier
    mb_dk_spath_ready: MBarrier
    mb_tmem_done: MBarrier
    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_gdn_bars(cfg) -> GdnBwdBars:
    """GdnBwdBars factory.  MUST be called from inside ``_kernel`` (allocates SMEM;
    the mbar rings sit ahead of the gate scalar arrays and data buffers)."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBwdBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_mma_done=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_q_cg1_done=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_mma_done=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_k_cg0_done=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_mma_done=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_cg1_done=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_do_ready=MBarrier(alloc(cfg.smem_do_stages), stages=cfg.smem_do_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_do_stages), stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_cg1_done=MBarrier(alloc(cfg.smem_do_stages), stages=cfg.smem_do_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_s_ready=MBarrier(alloc(cfg.smem_s_stages), stages=cfg.smem_s_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_s_done=MBarrier(alloc(cfg.smem_s_stages), stages=cfg.smem_s_stages, init_count=ONE_LANE + CG0_THREADS, producer=Producer.MMA_COMMIT),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=CG0_PLUS_CG1, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=CG0_PLUS_CG1, producer=Producer.THREAD),
        mb_dh_acc_ready=MBarrier(alloc(cfg.tmem_dh_acc_stages), stages=cfg.tmem_dh_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dh_acc_done=MBarrier(alloc(cfg.tmem_dh_acc_stages), stages=cfg.tmem_dh_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        # five sequential per-chunk productions share the dV/dK TMEM slot
        mb_du_scale_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_du_scale_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_du_total_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_scale_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_scale_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dk_attn_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_attn_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dk_total_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_total_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        # shared accumulators at STATIC columns: group A holds
        # kk -> ks -> dY -> dM core, group B holds A -> U -> dA.
        mb_kk_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_kk_acc_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_ks_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dy_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_ainv_ready=MBarrier(alloc(cfg.smem_ainv_stages), stages=cfg.smem_ainv_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_ainv_done=MBarrier(alloc(cfg.smem_ainv_stages), stages=cfg.smem_ainv_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_qk_ready=MBarrier(alloc(cfg.smem_qk_stages), stages=cfg.smem_qk_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_qk_done=MBarrier(alloc(cfg.smem_qk_stages), stages=cfg.smem_qk_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dh_inp_ready=MBarrier(alloc(cfg.tmem_dh_inp_stages), stages=cfg.tmem_dh_inp_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dh_inp_done=MBarrier(alloc(cfg.tmem_dh_inp_stages), stages=cfg.tmem_dh_inp_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        # f16 input restages at STATIC columns: dO' alone; dU and dY' overlap
        mb_dop_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dop_inp_done=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_du_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dyp_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dyp_inp_done=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dq_tmastg_ready=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dq_tmastg_done=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_dk_tmastg_ready=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dk_tmastg_done=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_dv_tmastg_ready=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dv_tmastg_done=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_y_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_sdv_done=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_u_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dhs_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dhs_done=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_da_ready=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dq_acc_scale_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dq_acc_scale_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dq_acc_total_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dq_acc_total_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_da_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dm_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dm_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dbeta_cg1_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dgate_cg1_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_hdh_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dk_spath_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS + CG1_THREADS, producer=Producer.THREAD),
        mb_sched_ready=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_sched_done=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=11, producer=Producer.THREAD),
    )


# ---------------------------------------------------------------------------
# Dynamic tile scheduler: global-ticket work-stealing ring
# ---------------------------------------------------------------------------


@cute.jit
def _sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_done[sched_state.idx].wait(sched_state.phase)
        if nvvm.elect_sync():
            fetched = cutlass.Int32(nvvm.atomicrmw("add", mSched.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
            sSched[sched_state.idx] = num_ctas + fetched
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_ready[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


@cute.jit
def _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_ready[sched_state.idx].wait(sched_state.phase)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_done[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


# ---------------------------------------------------------------------------
# Device-side helpers / warp bodies
# ---------------------------------------------------------------------------


@cute.jit
def _invert_diagonal_NxN(cfg, in_base, out_base, d, tidx, N: int = 8):
    """Stage 1: Gauss-Jordan inversion of one diagonal NxN block, reading the
    raw block from ``in_base`` and writing the inverse to ``out_base`` (same
    swizzled offsets).

    The tile swizzle re-homes whole rows only, so a diagonal block's row stays
    a contiguous N-element run at ``swz(row_lin_base)``.
    """
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_lin_base = (d * N + tidx_in_group) * BT + d * N
    row_phys = swizzle_lin_128b(row_lin_base, row_stride_log2=6)
    row_ptr_in = (
        cute.make_ptr(
            cfg.io_dtype,
            in_base,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )
    row_ptr = (
        cute.make_ptr(
            cfg.io_dtype,
            out_base,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )

    row = [(row_ptr_in + j).load().to(cutlass.Float32) for j in range(N)]
    for i in cutlass.range_constexpr(N):
        row[i] = cutlass.Float32(1.0) if tidx_in_group == i else row[i]
    for src_row in cutlass.range_constexpr(N - 1):
        row_scale = -row[src_row]
        for i in cutlass.range_constexpr(src_row):
            shfl_val = nvvm.shfl_sync(0xFFFFFFFF, row[i], src_row, 0b1100000011111, kind=nvvm.Shfl.IDX)
            row[i] = row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
        row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

    for j in cutlass.range_constexpr(N):
        (row_ptr + j).store(row[j].to(cfg.io_dtype))


@cute.jit
def _mma_m16n8k8(a0, a1, b0, c_regs, dtype: cutlass.Constexpr):
    """One m16n8k8 reg-reg mma (``mma_step`` only emits the k16 form),
    accumulating into the ``c_regs`` buffer in place."""
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    c_regs[0], c_regs[1], c_regs[2], c_regs[3] = inline_ptx(
        f"mma.sync.aligned.m16n8k8.row.col.f32.{tag}.{tag}.f32" " {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
        write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
        read_only_args=[a0, a1, b0, c_regs[0], c_regs[1], c_regs[2], c_regs[3]],
    )


@cute.jit
def _warp_reduce_scatter_frag16(vals, lane_id):
    """Reduce-scatter 16 fragment token-partials (tcol = (lane%4)*2 +
    (j//2)*8 + (j%2)) over the 8 lane-groups: step k exchanges with
    lane^(4<<k), keeping the j half whose bit k+1 matches the lane bit.
    Lane L returns the sums of tokens (L//4)*8 + (L%4)*2 and +1."""
    cur = vals
    for k in cutlass.range_constexpr(3):
        off = cutlass.const_expr(4 << k)
        hi_lane = (lane_id // off) % 2 == 1
        nxt = []
        for i in cutlass.range_constexpr(len(cur)):
            if cutlass.const_expr((i & 2) == 0):
                lo = cur[i]
                hi = cur[i + 2]
                send = lo if hi_lane else hi
                recv = nvvm.shfl_sync(0xFFFFFFFF, send, off, 31, kind=nvvm.Shfl.BFLY)
                keep = hi if hi_lane else lo
                nxt.append(keep + recv)
        cur = nxt
    return cur[0], cur[1]


@cute.jit
def _blockwise_diagonal_8x8_to_16x16(cfg, base_int, raw_base, d0, lane_id):
    """Stage 2: off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}).
    Raw C blocks read from ``raw_base`` (sKK); corrected blocks and all
    writes on ``base_int`` (sAinv).

    Keep the per-lane ldmatrix offset (``lds1``/``lds4``) a SEPARATE sum term
    from the warp-uniform (row, col) origin in all the diagonal helpers —
    folding it into the row term costs ~2% (per-lane address datapath).
    """
    bpe = cfg.io_dtype.width // 8
    lds1 = (lane_id % 8) * 64
    d = nvvm.ldmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b((d0 + 8) * 64 + d0 + 8 + lds1, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        1,
        nvvm.MMALayout.ROW,
    )
    c = nvvm.ldmatrix(
        cutlass.inttoptr(raw_base + swizzle_lin_128b((d0 + 8) * 64 + d0 + lds1, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        1,
        nvvm.MMALayout.COL,
    )
    c_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    _mma_m16n8k8(d, d, c, c_regs, cfg.io_dtype)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_f16 = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(2)]
    ai = nvvm.ldmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b(d0 * 64 + d0 + lds1, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        1,
        nvvm.MMALayout.COL,
    )
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    _mma_m16n8k8(a_f16[0], a_f16[1], ai, o_regs, cfg.io_dtype)
    o_f16 = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)
    nvvm.stmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b((d0 + 8) * 64 + d0 + lds1, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        o_f16,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def _blockwise_diagonal_16x16_to_32x32(cfg, base_int, raw_base, d0, lane_id):
    """Stage 3: off-diagonal correction 16x16 -> 32x32 (raw C from
    ``raw_base``)."""
    bpe = cfg.io_dtype.width // 8
    lds4 = (lane_id % 16) * 64 + (lane_id // 16) * 8
    d = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int + swizzle_lin_128b((d0 + 16) * 64 + d0 + 16 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16
            ),
            4,
            nvvm.MMALayout.ROW,
        )
    )
    c = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(raw_base + swizzle_lin_128b((d0 + 16) * 64 + d0 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
            4,
            nvvm.MMALayout.COL,
        )
    )
    c_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d, c, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_f16 = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
    ai = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(base_int + swizzle_lin_128b(d0 * 64 + d0 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
            4,
            nvvm.MMALayout.COL,
        )
    )
    o_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_f16, ai, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    ow = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
    nvvm.stmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b((d0 + 16) * 64 + d0 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        ow,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def _blockwise_diagonal_32x32_to_64x64(cfg, base_int, raw_base, warp_id, lane_id):
    """Stage 4: off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row
    M-band each; raw C from ``raw_base``)."""
    band = warp_id % 2
    bpe = cfg.io_dtype.width // 8
    lds4 = (lane_id % 16) * 64 + (lane_id // 16) * 8
    a_regs = []
    for vs in cutlass.range_constexpr(2):
        a_regs += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int + swizzle_lin_128b((32 + band * 16) * 64 + 32 + vs * 16 + lds4, row_stride_log2=6) * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.ROW,
            )
        )
    b_regs = []
    for vs in cutlass.range_constexpr(4):
        b_regs += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    raw_base + swizzle_lin_128b((32 + (vs // 2) * 16) * 64 + (vs % 2) * 16 + lds4, row_stride_log2=6) * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.COL,
            )
        )
    c_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(c_regs, a_regs, b_regs[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_f16 = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]
    ai_regs = []
    for vs in cutlass.range_constexpr(4):
        ai_regs += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int + swizzle_lin_128b(((vs // 2) * 16) * 64 + (vs % 2) * 16 + lds4, row_stride_log2=6) * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.COL,
            )
        )
    o_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(o_regs, a_f16, ai_regs[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    ow = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_inner_barrier_id,
        thread_count=cfg.inverse_inner_barrier_threads,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b((32 + band * 16) * 64 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        ow[0:4],
        nvvm.MMALayout.ROW,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(base_int + swizzle_lin_128b((32 + band * 16) * 64 + 16 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
        ow[4:8],
        nvvm.MMALayout.ROW,
    )


@cute.jit
def _tmastg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sdQ_raw,
    sdK_raw,
    sdV_raw,
    desc_dq_base,
    desc_dk_base,
    desc_dv_base,
    sSched,
    bars,
):
    """TMA-STG warp role (warp 11): persistent tile-scheduler loop + per-chunk
    dQ/dK/dV TMA bulk-stores from the SMEM staging buffers to global memory."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2

    bpe = cfg.io_dtype.width // 8
    granu = 128 // bpe
    sdQ_tma = SmemTile(
        base=sdQ_raw,
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages),
        stages=cfg.smem_dq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    sdK_tma = SmemTile(
        base=sdK_raw,
        elems_per_stage=(cfg.dk_cosize // cfg.smem_dk_stages),
        stages=cfg.smem_dk_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    sdV_tma = SmemTile(
        base=sdV_raw,
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages),
        stages=cfg.smem_dv_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )

        slot = (batch_idx * heads_out + head_idx) * desc_qwords
        desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
        if nvvm.elect_sync():
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)

        for rev_idx in cutlass.range(cend - wstart):
            chunk_idx = cend - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            dv_idx = dv_index.idx
            bars.mb_dv_tmastg_ready[dv_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), tok_coord)
            if cutlass.const_expr(cfg.split_k):
                # right-warmup chunks stage grads to SMEM but never store them
                if chunk_idx < wend:
                    tma_store_tile(sdV_tma[dv_idx], dv_slice, acquire=False)
                    tma_store_commit()
            else:
                tma_store_tile(sdV_tma[dv_idx], dv_slice, acquire=False)
                tma_store_commit()

            dq_idx = dq_index.idx
            bars.mb_dq_tmastg_ready[dq_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), tok_coord)
            if cutlass.const_expr(cfg.split_k):
                if chunk_idx < wend:
                    tma_store_tile(sdQ_tma[dq_idx], dq_slice, acquire=False)
                    tma_store_commit()
            else:
                tma_store_tile(sdQ_tma[dq_idx], dq_slice, acquire=False)
                tma_store_commit()

            dk_idx = dk_index.idx
            bars.mb_dk_tmastg_ready[dk_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), tok_coord)
            if cutlass.const_expr(cfg.split_k):
                if chunk_idx < wend:
                    tma_store_tile(sdK_tma[dk_idx], dk_slice, acquire=False)
                    tma_store_commit()
            else:
                tma_store_tile(sdK_tma[dk_idx], dk_slice, acquire=False)
                tma_store_commit()

            tma_store_wait(2)
            bars.mb_dv_tmastg_done[dv_idx].arrive()
            tma_store_wait(1)
            bars.mb_dq_tmastg_done[dq_idx].arrive()
            tma_store_wait(0)
            bars.mb_dk_tmastg_done[dk_idx].arrive()
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _gate_beta_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    mGate,
    mBeta,
    mDg,
    mDbeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sSched,
    bars,
):
    """Gate/beta LOAD + STORE warp role (warp 10): per-chunk gate/beta G->S
    loads (BACKWARD order, one-chunk prefetch) and the dG/dBeta stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    gate_store_index = PipelineState.start(phase=0)
    beta_store_index = PipelineState.start(phase=0)
    lidx = tidx % cfg.threads_per_warp
    n_cols = cfg.b_t // cfg.threads_per_warp
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = cend - wstart
        if cutlass.const_expr(cfg.split_k):
            # dG/dBeta ownership: mask stores past the item's write range
            write_end = batch_start + wend * cfg.b_t
            write_end = write_end if write_end < batch_end else batch_end
        else:
            write_end = batch_end
        # ---- prefetch: the FIRST backward chunk's gate/beta ------------
        if sk_nt > 0:
            chunk_offset = batch_start + (cend - 1) * cfg.b_t
            gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
            gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
            gate_idx = gate_index.idx
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            pos_valid = [None] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                pos_valid[col] = cute.elem_less(chunk_offset + pos, batch_end)

            # --- Gate load (OOB neutral: 1.0 -> log2 = 0.0) ---
            tGrGate = [cutlass.Float32(0.0)] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
                tGrGate[col] = gGate[pos] if pos_valid[col] else oob_neutral

            if cutlass.const_expr(cfg.log_gate):
                for col in cutlass.range_constexpr(n_cols):
                    tGrGate[col] = tGrGate[col] * cutlass.Float32(RCP_LN2)
            else:
                for col in cutlass.range_constexpr(n_cols):
                    tGrGate[col] = cute.math.log2(tGrGate[col] + 1e-10, fastmath=True)
            for offset in [1, 2, 4, 8, 16]:
                for col in cutlass.range_constexpr(n_cols):
                    n = nvvm.shfl_sync(0xFFFFFFFF, tGrGate[col], offset, 0, kind=nvvm.Shfl.UP)
                    if lidx >= offset:
                        tGrGate[col] = tGrGate[col] + n
            for col in cutlass.range_constexpr(1, n_cols):
                last_v = nvvm.shfl_sync(
                    0xFFFFFFFF,
                    tGrGate[col - 1],
                    cfg.threads_per_warp - 1,
                    cfg.threads_per_warp - 1,
                    kind=nvvm.Shfl.IDX,
                )
                tGrGate[col] += last_v

            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                sCumsumlog[pos, 0, gate_idx] = tGrGate[col]
                sCumprod[pos, 0, gate_idx] = cute.math.exp2(tGrGate[col], fastmath=True)

            bars.mb_gate_ready[gate_idx].arrive()

            # --- Beta load (per-element async G->S cp.async) ---
            beta_idx = beta_index.idx
            beta_index = advance(beta_index, cfg.smem_beta_stages)
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                src = gBeta.iterator + gBeta.layout((pos,))
                dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                nvvm.cp_async_shared_global(dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size)
            nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)

        for rev_idx in cutlass.range(sk_nt):
            # ---- prefetch the NEXT chunk's gate/beta -------------------
            if rev_idx + 1 < sk_nt:
                chunk_offset = batch_start + (cend - 2 - rev_idx) * cfg.b_t
                gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
                gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
                gate_idx = gate_index.idx
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                pos_valid = [None] * n_cols
                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    pos_valid[col] = cute.elem_less(chunk_offset + pos, batch_end)

                tGrGate = [cutlass.Float32(0.0)] * n_cols
                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
                    tGrGate[col] = gGate[pos] if pos_valid[col] else oob_neutral

                if cutlass.const_expr(cfg.log_gate):
                    for col in cutlass.range_constexpr(n_cols):
                        tGrGate[col] = tGrGate[col] * cutlass.Float32(RCP_LN2)
                else:
                    for col in cutlass.range_constexpr(n_cols):
                        tGrGate[col] = cute.math.log2(tGrGate[col] + 1e-10, fastmath=True)
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(0xFFFFFFFF, tGrGate[col], offset, 0, kind=nvvm.Shfl.UP)
                        if lidx >= offset:
                            tGrGate[col] = tGrGate[col] + n
                for col in cutlass.range_constexpr(1, n_cols):
                    last_v = nvvm.shfl_sync(
                        0xFFFFFFFF,
                        tGrGate[col - 1],
                        cfg.threads_per_warp - 1,
                        cfg.threads_per_warp - 1,
                        kind=nvvm.Shfl.IDX,
                    )
                    tGrGate[col] += last_v

                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, gate_idx] = tGrGate[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(tGrGate[col], fastmath=True)

                bars.mb_gate_ready[gate_idx].arrive()

                beta_idx = beta_index.idx
                beta_index = advance(beta_index, cfg.smem_beta_stages)
                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    src = gBeta.iterator + gBeta.layout((pos,))
                    dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                    cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                    nvvm.cp_async_shared_global(dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size)
                nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)

            # ---- store-ready wait + in-place store back ----------------
            st_offset = batch_start + (cend - 1 - rev_idx) * cfg.b_t
            gGate_st = cute.domain_offset((st_offset,), mDg[None, head_idx])
            gBeta_st = cute.domain_offset((st_offset,), mDbeta[None, head_idx])
            g_st_idx = gate_store_index.idx
            bars.mb_gate_done[g_st_idx].wait(gate_store_index.phase)
            gate_store_index = advance(gate_store_index, cfg.smem_gate_stages)
            tGrDg = [cutlass.Float32(0.0)] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                tGrDg[col] = sCumsumlog[pos, 0, g_st_idx]
            for offset in [1, 2, 4, 8, 16]:
                for col in cutlass.range_constexpr(n_cols):
                    n = nvvm.shfl_sync(0xFFFFFFFF, tGrDg[col], offset, 31, kind=nvvm.Shfl.DOWN)
                    if lidx < cfg.threads_per_warp - offset:
                        tGrDg[col] = tGrDg[col] + n
            for col in cutlass.range_constexpr(n_cols - 1):
                cc = cutlass.const_expr(n_cols - 2 - col)
                later_total = nvvm.shfl_sync(0xFFFFFFFF, tGrDg[cc + 1], 0, 0, kind=nvvm.Shfl.IDX)
                tGrDg[cc] = tGrDg[cc] + later_total
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                if cute.elem_less(st_offset + pos, write_end):
                    gGate_st[pos] = tGrDg[col]
            b_st_idx = beta_store_index.idx
            bars.mb_beta_done[b_st_idx].wait(beta_store_index.phase)
            beta_store_index = advance(beta_store_index, cfg.smem_beta_stages)
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                if cute.elem_less(st_offset + pos, write_end):
                    gBeta_st[pos] = sBeta[pos, 0, b_st_idx]
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_hold,
    sQ,
    sQ_trans,
    sK,
    sK_trans,
    sV,
    sV_kmaj,
    sdO,
    sdO_kmaj,
    sS,
    sS_kmaj,
    sAinv,
    sAinv_trans,
    sQk,
    sQk_trans,
    sDa,
    sDa_trans,
    sdH,
    sDm,
    sDm_trans,
    sdV,
    sdV_kmaj,
    sSched,
    bars,
):
    """MMA (UMMA issuer) warp role (warp 8): persistent scheduler loop +
    per-chunk (BACKWARD order) issue of the full GEMM pipeline (see the
    module docstring for the per-chunk MMA order); owns the TMEM lifecycle
    (alloc up front, dealloc once CG0 and CG1 both signal mb_tmem_done)."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    kk_read = PipelineState.start(phase=0)
    dk_total_free = PipelineState.start(phase=1)
    du_scale_read = PipelineState.start(phase=0)
    dk_scale_read = PipelineState.start(phase=0)
    dk_attn_read = PipelineState.start(phase=0)
    dh_acc_index = PipelineState.start(phase=0 if cfg.use_dht else 1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)
    s_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    ainv_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    y_rdy_index = PipelineState.start(phase=0)
    u_index = PipelineState.start(phase=0)
    da_rdy_index = PipelineState.start(phase=0)
    dv_rdy_index = PipelineState.start(phase=0)
    dm_index = PipelineState.start(phase=0)
    dhs_index = PipelineState.start(phase=0)
    dq_scale_index = PipelineState.start(phase=0)
    dq_total_index = PipelineState.start(phase=1)
    qk_index = PipelineState.start(phase=0)
    dop_inp_rdy = PipelineState.start(phase=0)
    du_inp_rdy = PipelineState.start(phase=0)
    dyp_inp_rdy = PipelineState.start(phase=0)
    dh_inp_index = PipelineState.start(phase=0)

    nvvm.tcgen05_alloc(tmem_hold, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

    # ---- chunk-invariant GEMM descriptors ------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_qk_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_dop_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_ks_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_spath_col = tmem_shared_inp_col

    idesc_dv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_dh_inp_col = tmem_base + cfg.tmem_dh_inp_offset
    tmem_dvdk_col = tmem_base + cfg.tmem_dvdk_offset
    DH_INP_STAGE_COLS = cfg.d_k // 2

    idesc_ks = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
    )
    bmm_ks_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_ks,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    idesc_du = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_du_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_du,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dh = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_dh_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dh,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_dop_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    tmem_dh_col = tmem_base + cfg.tmem_dh_offset
    tmem_y_col = tmem_base + cfg.tmem_y_offset

    idesc_dy = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dy_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dy,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_da = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_da_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_da,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    idesc_u = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_u_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_u,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_dstate_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dstate,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dka = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_dka_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dka,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dqa = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dqa_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dqa,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = cend - wstart

        # ---- first backward chunk (c = NT-1): no dH yet; with a dht seed
        # every chunk takes the steady path ----
        if cutlass.const_expr(not cfg.use_dht):
            if sk_nt > 0:
                k_idx = k_index.idx
                bars.mb_k_ready[k_idx].wait(k_index.phase)
                k_index = advance(k_index, cfg.smem_k_stages)

                desc_k = sK[k_idx].desc()
                mma_ss(
                    bmm_qk_desc,
                    desc_k,
                    desc_k,
                    nvvm.make_tmem_ptr(tmem_kk_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_kk_acc_ready[0].arrive(cta_group=1)

                q_idx = q_index.idx
                bars.mb_q_ready[q_idx].wait(q_index.phase)
                q_index = advance(q_index, cfg.smem_q_stages)

                desc_q = sQ[q_idx].desc()
                mma_ss(
                    bmm_qk_desc,
                    desc_q,
                    desc_k,
                    nvvm.make_tmem_ptr(tmem_a_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_a_acc_ready[0].arrive(cta_group=1)

                # ---- ks (K @ S) first, then dQ inter (S @ dO) ----------------------
                s_idx = s_index.idx
                if cend >= SFIRST_MIN:
                    bars.mb_s_ready[s_idx].wait(s_index.phase)
                    s_index = advance(s_index, cfg.smem_s_stages)
                    bars.mb_kk_acc_done[0].wait(kk_read.phase)
                    kk_read = advance(kk_read, 1)
                    desc_s = sS[s_idx].desc()
                    mma_ss(
                        bmm_ks_desc,
                        desc_s,
                        desc_k,
                        nvvm.make_tmem_ptr(tmem_ks_col, cutlass.Float32),
                        accumulate=False,
                    )
                    if nvvm.elect_sync():
                        bars.mb_ks_acc_ready[0].arrive(cta_group=1)
                do_idx = do_index.idx
                bars.mb_do_ready[do_idx].wait(do_index.phase)
                do_index = advance(do_index, cfg.smem_do_stages)
                if cend >= SFIRST_MIN:
                    bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                    dq_total_index = advance(dq_total_index, 1)
                    desc_s_k = sS_kmaj[s_idx].desc()
                    desc_do_k2 = sdO_kmaj[do_idx].desc()
                    mma_ss(
                        bmm_dstate_desc,
                        desc_s_k,
                        desc_do_k2,
                        nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                        accumulate=False,
                    )
                    if nvvm.elect_sync():
                        bars.mb_dq_acc_scale_ready[0].arrive(cta_group=1)

                # ---- dU intra: dO^T @ A -> the dV/dK slot ------------------
                du_qk_idx = qk_index.idx
                bars.mb_qk_ready[du_qk_idx].wait(qk_index.phase)
                qk_index = advance(qk_index, cfg.smem_qk_stages)
                bars.mb_dk_total_done[0].wait(dk_total_free.phase)
                dk_total_free = advance(dk_total_free, 1)

                desc_do_mn = sdO[do_idx].desc()
                desc_qk_t = sQk_trans[du_qk_idx].desc()
                mma_ss(
                    bmm_du_desc,
                    desc_do_mn,
                    desc_qk_t,
                    nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_du_total_ready[0].arrive(cta_group=1)

                # ---- dH init: dO'^T @ Q -------------------------------------
                bars.mb_dop_inp_ready[0].wait(dop_inp_rdy.phase)
                dop_inp_rdy = advance(dop_inp_rdy, 1)
                dh_idx = dh_acc_index.idx
                bars.mb_dh_acc_done[dh_idx].wait(dh_acc_index.phase)
                dh_acc_index = advance(dh_acc_index, cfg.tmem_dh_acc_stages)

                do_a_ptr = nvvm.make_tmem_ptr(
                    tmem_dop_col,
                    cutlass.Int8,
                )
                desc_q_t = sQ_trans[q_idx].desc()
                mma_ts(
                    bmm_dh_desc,
                    do_a_ptr,
                    desc_q_t,
                    nvvm.make_tmem_ptr(tmem_dh_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dop_inp_done[0].arrive(cta_group=1)

                # ---- Y operand acquire: y_ready = CG1's delta --------------
                bars.mb_y_ready[0].wait(y_rdy_index.phase)
                y_rdy_index = advance(y_rdy_index, 1)
                v_idx = v_index.idx
                v_index = advance(v_index, cfg.smem_v_stages)

                # ---- U recompute: U^T = Y^T @ T^T -> shared acc ------------
                ainv_idx = ainv_index.idx
                bars.mb_ainv_ready[ainv_idx].wait(ainv_index.phase)
                ainv_index = advance(ainv_index, cfg.smem_ainv_stages)

                y_a_ptr = nvvm.make_tmem_ptr(tmem_y_col, cutlass.Int8)
                desc_t_plain = sAinv[ainv_idx].desc()
                mma_ts(
                    bmm_u_desc,
                    y_a_ptr,
                    desc_t_plain,
                    nvvm.make_tmem_ptr(tmem_u_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_u_acc_ready[0].arrive(cta_group=1)

                # ---- dY: dU^T(TMEM f16) @ T ---------------------------------
                bars.mb_du_inp_ready[0].wait(du_inp_rdy.phase)
                du_inp_rdy = advance(du_inp_rdy, 1)

                du_a_ptr = nvvm.make_tmem_ptr(
                    tmem_du_col,
                    cutlass.Int8,
                )
                desc_ainv_t = sAinv_trans[ainv_idx].desc()
                mma_ts(
                    bmm_dy_desc,
                    du_a_ptr,
                    desc_ainv_t,
                    nvvm.make_tmem_ptr(tmem_dy_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dy_acc_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_ainv_done[ainv_idx].arrive(cta_group=1)

                # ---- dA_eff: dO @ U^T -> shared acc -------------------------
                bars.mb_u_ready[0].wait(u_index.phase)
                u_index = advance(u_index, 1)

                desc_do_k = sdO_kmaj[do_idx].desc()
                desc_u = sV_kmaj[v_idx].desc()
                mma_ss(
                    bmm_da_desc,
                    desc_do_k,
                    desc_u,
                    nvvm.make_tmem_ptr(tmem_da_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_da_acc_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_do_mma_done[do_idx].arrive(cta_group=1)

                # ---- dM core: dY @ U^T --------------------------------------
                dv_stg_x_idx = dv_rdy_index.idx
                bars.mb_dv_tmastg_ready[dv_stg_x_idx].wait(dv_rdy_index.phase)
                dv_rdy_index = advance(dv_rdy_index, cfg.smem_dv_stages)

                desc_dy_dm = sdV_kmaj[0].desc()
                desc_u_dm = sV_kmaj[v_idx].desc()
                mma_ss(
                    bmm_da_desc,
                    desc_dy_dm,
                    desc_u_dm,
                    nvvm.make_tmem_ptr(tmem_dm_core_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dm_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_v_mma_done[v_idx].arrive(cta_group=1)

                # ---- dH ds-term: dH += dY'^T @ K ----------------------------
                bars.mb_dyp_inp_ready[0].wait(dyp_inp_rdy.phase)
                dyp_inp_rdy = advance(dyp_inp_rdy, 1)

                dyp_a_ptr = nvvm.make_tmem_ptr(
                    tmem_dyp_col,
                    cutlass.Int8,
                )
                desc_k_t = sK_trans[k_idx].desc()
                mma_ts(
                    bmm_dh_desc,
                    dyp_a_ptr,
                    desc_k_t,
                    nvvm.make_tmem_ptr(tmem_dh_col, cutlass.Float32),
                    accumulate=True,
                )
                if nvvm.elect_sync():
                    bars.mb_dh_acc_ready[dh_idx].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_dyp_inp_done[0].arrive(cta_group=1)

                # ---- dK attn: Q^T @ dA(sDa) -> the dV/dK slot --------------
                bars.mb_da_ready[0].wait(da_rdy_index.phase)
                da_rdy_index = advance(da_rdy_index, 1)

                desc_q_mn = sQ_trans[q_idx].desc()
                desc_da_t = sDa_trans[0].desc()
                mma_ss(
                    bmm_dka_desc,
                    desc_q_mn,
                    desc_da_t,
                    nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dk_attn_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_q_mma_done[q_idx].arrive(cta_group=1)

                # ---- dQ attn: K^T @ dA onto the rescaled dQ acc -------------------
                if cend >= SFIRST_MIN:
                    bars.mb_dq_acc_scale_done[0].wait(dq_scale_index.phase)
                    dq_scale_index = advance(dq_scale_index, 1)
                if cend < SFIRST_MIN:
                    bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                    dq_total_index = advance(dq_total_index, 1)
                desc_k_mn = sK_trans[k_idx].desc()
                desc_da_p = sDa[0].desc()
                if cend >= SFIRST_MIN:
                    mma_ss(
                        bmm_dqa_desc,
                        desc_k_mn,
                        desc_da_p,
                        nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                        accumulate=True,
                    )
                if cend < SFIRST_MIN:
                    mma_ss(
                        bmm_dqa_desc,
                        desc_k_mn,
                        desc_da_p,
                        nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                        accumulate=False,
                    )
                if nvvm.elect_sync():
                    bars.mb_dq_acc_total_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_qk_done[du_qk_idx].arrive(cta_group=1)

                # ---- dK s-path: S @ dY^T -> shared acc, before the dM-terms --------
                if cend >= SFIRST_MIN:
                    desc_s_k5 = sS_kmaj[s_idx].desc()
                    desc_dy_k5 = sdV_kmaj[0].desc()
                    mma_ss(
                        bmm_dstate_desc,
                        desc_s_k5,
                        desc_dy_k5,
                        nvvm.make_tmem_ptr(tmem_dk_spath_col, cutlass.Float32),
                        accumulate=False,
                    )
                    if nvvm.elect_sync():
                        bars.mb_dk_spath_ready[0].arrive(cta_group=1)
                    if nvvm.elect_sync():
                        bars.mb_s_done[s_idx].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_sdv_done[0].arrive(cta_group=1)

                # ---- dK dM-terms: K^T @ dM^T + K^T @ dM onto the slot ------
                bars.mb_dm_done[0].wait(dm_index.phase)
                dm_index = advance(dm_index, 1)
                bars.mb_dk_attn_done[0].wait(dk_attn_read.phase)
                dk_attn_read = advance(dk_attn_read, 1)
                desc_k_mn2 = sK_trans[k_idx].desc()
                desc_dm_t = sDm_trans[0].desc()
                mma_ss(
                    bmm_dka_desc,
                    desc_k_mn2,
                    desc_dm_t,
                    nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                    accumulate=True,
                )
                desc_dm_p = sDm[0].desc()
                mma_ss(
                    bmm_dqa_desc,
                    desc_k_mn2,
                    desc_dm_p,
                    nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                    accumulate=True,
                )
                if nvvm.elect_sync():
                    bars.mb_dk_total_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_k_mma_done[k_idx].arrive(cta_group=1)

        # ---- chunks NT-2 .. 0 (backward): full body ----------------------
        for rev_idx in cutlass.range(0 if cfg.use_dht else 1, sk_nt):
            chunk_idx = cend - 1 - rev_idx
            # ---- kk: K @ K^T -> shared acc ------------------------------
            k_idx = k_index.idx
            bars.mb_k_ready[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)

            desc_k = sK[k_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_k,
                desc_k,
                nvvm.make_tmem_ptr(tmem_kk_col, cutlass.Float32),
                accumulate=False,
            )

            if nvvm.elect_sync():
                bars.mb_kk_acc_ready[0].arrive(cta_group=1)

            # ---- qk: Q @ K^T -> shared acc ------------------------------
            q_idx = q_index.idx
            bars.mb_q_ready[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)

            desc_q = sQ[q_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_q,
                desc_k,
                nvvm.make_tmem_ptr(tmem_a_col, cutlass.Float32),
                accumulate=False,
            )

            if nvvm.elect_sync():
                bars.mb_a_acc_ready[0].arrive(cta_group=1)

            # ---- ks (K @ S): hoisted above dV inter so CG1's delta chain
            # is not serialized behind the prev chunk's dK readout ----------
            s_idx = s_index.idx
            if chunk_idx >= S_MIN:
                bars.mb_s_ready[s_idx].wait(s_index.phase)
                s_index = advance(s_index, cfg.smem_s_stages)
                bars.mb_kk_acc_done[0].wait(kk_read.phase)
                kk_read = advance(kk_read, 1)
                desc_s = sS[s_idx].desc()
                mma_ss(
                    bmm_ks_desc,
                    desc_s,
                    desc_k,
                    nvvm.make_tmem_ptr(tmem_ks_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_ks_acc_ready[0].arrive(cta_group=1)

            # ---- dV inter: dH^T @ K -> the dV/dK slot -------------------
            dh_inp_idx = dh_inp_index.idx
            bars.mb_dh_inp_ready[dh_inp_idx].wait(dh_inp_index.phase)
            dh_inp_index = advance(dh_inp_index, cfg.tmem_dh_inp_stages)
            bars.mb_dk_total_done[0].wait(dk_total_free.phase)
            dk_total_free = advance(dk_total_free, 1)

            dh_a_ptr = nvvm.make_tmem_ptr(tmem_dh_inp_col + dh_inp_idx * DH_INP_STAGE_COLS, cutlass.Int8)
            mma_ts(
                bmm_dv_desc,
                dh_a_ptr,
                desc_k,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_du_scale_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_dh_inp_done[dh_inp_idx].arrive(cta_group=1)

            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if chunk_idx >= S_MIN:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
                desc_s_k = sS_kmaj[s_idx].desc()
                desc_do_k2 = sdO_kmaj[do_idx].desc()
                mma_ss(
                    bmm_dstate_desc,
                    desc_s_k,
                    desc_do_k2,
                    nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dq_acc_scale_ready[0].arrive(cta_group=1)

            # ---- dU intra: dO^T @ A accumulated onto the scaled dV inter ----
            du_qk_idx = qk_index.idx
            bars.mb_qk_ready[du_qk_idx].wait(qk_index.phase)
            qk_index = advance(qk_index, cfg.smem_qk_stages)
            bars.mb_du_scale_done[0].wait(du_scale_read.phase)
            du_scale_read = advance(du_scale_read, 1)

            desc_do_mn = sdO[do_idx].desc()
            desc_qk_t = sQk_trans[du_qk_idx].desc()
            mma_ss(
                bmm_du_desc,
                desc_do_mn,
                desc_qk_t,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_du_total_ready[0].arrive(cta_group=1)

            # ---- dH update: dO'^T @ Q -----------------------------------
            bars.mb_dop_inp_ready[0].wait(dop_inp_rdy.phase)
            dop_inp_rdy = advance(dop_inp_rdy, 1)
            dh_idx = dh_acc_index.idx
            bars.mb_dh_acc_done[dh_idx].wait(dh_acc_index.phase)
            dh_acc_index = advance(dh_acc_index, cfg.tmem_dh_acc_stages)

            do_a_ptr = nvvm.make_tmem_ptr(
                tmem_dop_col,
                cutlass.Int8,
            )
            desc_q_t = sQ_trans[q_idx].desc()
            mma_ts(
                bmm_dh_desc,
                do_a_ptr,
                desc_q_t,
                nvvm.make_tmem_ptr(tmem_dh_col, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_dop_inp_done[0].arrive(cta_group=1)

            # ---- Y operand acquire: y_ready = CG1's delta --------------
            bars.mb_y_ready[0].wait(y_rdy_index.phase)
            y_rdy_index = advance(y_rdy_index, 1)
            v_idx = v_index.idx
            v_index = advance(v_index, cfg.smem_v_stages)

            # ---- U recompute: U^T = Y^T @ T^T -> shared acc ------------
            ainv_idx = ainv_index.idx
            bars.mb_ainv_ready[ainv_idx].wait(ainv_index.phase)
            ainv_index = advance(ainv_index, cfg.smem_ainv_stages)

            y_a_ptr = nvvm.make_tmem_ptr(tmem_y_col, cutlass.Int8)
            desc_t_plain = sAinv[ainv_idx].desc()
            mma_ts(
                bmm_u_desc,
                y_a_ptr,
                desc_t_plain,
                nvvm.make_tmem_ptr(tmem_u_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_u_acc_ready[0].arrive(cta_group=1)

            # ---- dY: dU^T(TMEM f16) @ T ---------------------------------
            bars.mb_du_inp_ready[0].wait(du_inp_rdy.phase)
            du_inp_rdy = advance(du_inp_rdy, 1)

            du_a_ptr = nvvm.make_tmem_ptr(
                tmem_du_col,
                cutlass.Int8,
            )
            desc_ainv_t = sAinv_trans[ainv_idx].desc()
            mma_ts(
                bmm_dy_desc,
                du_a_ptr,
                desc_ainv_t,
                nvvm.make_tmem_ptr(tmem_dy_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_dy_acc_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_ainv_done[ainv_idx].arrive(cta_group=1)

            # ---- dK inter: dH_entry(SMEM) @ U^T -> the dV/dK slot -------
            bars.mb_u_ready[0].wait(u_index.phase)
            u_index = advance(u_index, 1)
            bars.mb_dhs_ready[0].wait(dhs_index.phase)
            dhs_index = advance(dhs_index, 1)

            desc_dh_k = sdH[0].desc()
            desc_u_t = sV_kmaj[v_idx].desc()
            mma_ss(
                bmm_dstate_desc,
                desc_dh_k,
                desc_u_t,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_dk_scale_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_dhs_done[0].arrive(cta_group=1)

            # ---- dA_eff: dO @ U^T -> shared acc (U gated at dK-inter) ----
            desc_do_k = sdO_kmaj[do_idx].desc()
            desc_u = sV_kmaj[v_idx].desc()
            mma_ss(
                bmm_da_desc,
                desc_do_k,
                desc_u,
                nvvm.make_tmem_ptr(tmem_da_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_da_acc_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_do_mma_done[do_idx].arrive(cta_group=1)

            # ---- dM core: dY @ U^T --------------------------------------
            dv_stg_x_idx = dv_rdy_index.idx
            bars.mb_dv_tmastg_ready[dv_stg_x_idx].wait(dv_rdy_index.phase)
            dv_rdy_index = advance(dv_rdy_index, cfg.smem_dv_stages)

            desc_dy_dm = sdV_kmaj[0].desc()
            desc_u_dm = sV_kmaj[v_idx].desc()
            mma_ss(
                bmm_da_desc,
                desc_dy_dm,
                desc_u_dm,
                nvvm.make_tmem_ptr(tmem_dm_core_col, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_dm_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_v_mma_done[v_idx].arrive(cta_group=1)

            # ---- dH ds-term: dH += dY'^T @ K ----------------------------
            bars.mb_dyp_inp_ready[0].wait(dyp_inp_rdy.phase)
            dyp_inp_rdy = advance(dyp_inp_rdy, 1)

            dyp_a_ptr = nvvm.make_tmem_ptr(
                tmem_dyp_col,
                cutlass.Int8,
            )
            desc_k_t = sK_trans[k_idx].desc()
            mma_ts(
                bmm_dh_desc,
                dyp_a_ptr,
                desc_k_t,
                nvvm.make_tmem_ptr(tmem_dh_col, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_dh_acc_ready[dh_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_dyp_inp_done[0].arrive(cta_group=1)

            # ---- dK attn: Q^T @ dA(sDa) onto the rescaled dK inter ------
            bars.mb_da_ready[0].wait(da_rdy_index.phase)
            da_rdy_index = advance(da_rdy_index, 1)
            bars.mb_dk_scale_done[0].wait(dk_scale_read.phase)
            dk_scale_read = advance(dk_scale_read, 1)

            desc_q_mn = sQ_trans[q_idx].desc()
            desc_da_t = sDa_trans[0].desc()
            mma_ss(
                bmm_dka_desc,
                desc_q_mn,
                desc_da_t,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_dk_attn_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_q_mma_done[q_idx].arrive(cta_group=1)

            # ---- dQ attn: K^T @ dA onto the rescaled dQ acc -------------------
            if chunk_idx >= S_MIN:
                bars.mb_dq_acc_scale_done[0].wait(dq_scale_index.phase)
                dq_scale_index = advance(dq_scale_index, 1)
            if chunk_idx < S_MIN:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
            desc_k_mn = sK_trans[k_idx].desc()
            desc_da_p = sDa[0].desc()
            if chunk_idx >= S_MIN:
                mma_ss(
                    bmm_dqa_desc,
                    desc_k_mn,
                    desc_da_p,
                    nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < S_MIN:
                mma_ss(
                    bmm_dqa_desc,
                    desc_k_mn,
                    desc_da_p,
                    nvvm.make_tmem_ptr(tmem_dh_inp_col, cutlass.Float32),
                    accumulate=False,
                )
            if nvvm.elect_sync():
                bars.mb_dq_acc_total_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_qk_done[du_qk_idx].arrive(cta_group=1)

            # ---- dK s-path: S @ dY^T -> shared acc, before the dM-terms --------
            if chunk_idx >= S_MIN:
                desc_s_k5 = sS_kmaj[s_idx].desc()
                desc_dy_k5 = sdV_kmaj[0].desc()
                mma_ss(
                    bmm_dstate_desc,
                    desc_s_k5,
                    desc_dy_k5,
                    nvvm.make_tmem_ptr(tmem_dk_spath_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_dk_spath_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_s_done[s_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_sdv_done[0].arrive(cta_group=1)

            # ---- dK dM-terms: K^T @ dM^T + K^T @ dM onto the slot -------
            bars.mb_dm_done[0].wait(dm_index.phase)
            dm_index = advance(dm_index, 1)
            bars.mb_dk_attn_done[0].wait(dk_attn_read.phase)
            dk_attn_read = advance(dk_attn_read, 1)
            desc_k_mn2 = sK_trans[k_idx].desc()
            desc_dm_t = sDm_trans[0].desc()
            mma_ss(
                bmm_dka_desc,
                desc_k_mn2,
                desc_dm_t,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=True,
            )
            desc_dm_p = sDm[0].desc()
            mma_ss(
                bmm_dqa_desc,
                desc_k_mn2,
                desc_dm_p,
                nvvm.make_tmem_ptr(tmem_dvdk_col, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_dk_total_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_k_mma_done[k_idx].arrive(cta_group=1)

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_dk_total_done[0].wait(dk_total_free.phase)

    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def _tmaldg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sQ_raw,
    sK_raw,
    sV_raw,
    sdO_raw,
    sS_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_do_base,
    desc_s_base,
    desc_s0_base,
    sSched,
    mSched,
    bars,
):
    """TMA-LDG warp role (warp 9): persistent tile loop + per-chunk (BACKWARD
    order) Q/K/V/dO loads, plus S = H[c-1] for chunks c >= S_MIN (with an
    initial state, chunk 0's S comes from the io-dtype S0 descriptors
    instead).  The per-(b,h) runtime descriptors fold the sequence start and
    head, so Q/K/V/dO load at token ``chunk_idx*BT`` and S at the
    sequence-local index ``c - 1``."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    do_index = PipelineState.start(phase=1)
    s_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2

    bpe = cfg.io_dtype.width // 8
    granu = 128 // bpe
    bt = cfg.b_t
    q_stage_elems = cfg.q_cosize // cfg.smem_q_stages
    k_stage_elems = cfg.k_cosize // cfg.smem_k_stages
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=q_stage_elems,
        stages=cfg.smem_q_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=bt * granu,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=k_stage_elems,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=bt * granu,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=0,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    sdO_tma = SmemTile(
        base=sdO_raw,
        elems_per_stage=0,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    sS_tma = SmemTile(
        base=sS_raw,
        elems_per_stage=(cfg.s_cosize // cfg.smem_s_stages),
        stages=cfg.smem_s_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // 64,
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.d_k * 64,
    )
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )

        slot = (batch_idx * heads_out + head_idx) * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_s_slot = (desc_s_base + slot).tospace(cutlass.AddressSpace.generic)
        if cutlass.const_expr(cfg.use_initial_state):
            desc_s0_slot = (desc_s0_base + slot).tospace(cutlass.AddressSpace.generic)
        if nvvm.elect_sync():
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_s_slot)
            if cutlass.const_expr(cfg.use_initial_state):
                tma_tensormap_acquire(desc_s0_slot)

        for rev_idx in cutlass.range(cend - wstart):
            chunk_idx = cend - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            # ----------------------------------------------------------
            # K  (B operand of GEMMs 1/2/3, double-buffered)
            # ----------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_mma_done[k_idx].wait(k_index.phase)
            bars.mb_k_cg0_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if nvvm.elect_sync():
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # Q  (A operand of the qk GEMM, single-buffered)
            # ----------------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_mma_done[q_idx].wait(q_index.phase)
            bars.mb_q_cg1_done[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            if nvvm.elect_sync():
                bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # V  (transposed [DV, T] descriptor)
            # ----------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_mma_done[v_idx].wait(v_index.phase)
            bars.mb_v_cg1_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if nvvm.elect_sync():
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # dO  (transposed [DV, T] descriptor, like V)
            # ----------------------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_mma_done[do_idx].wait(do_index.phase)
            bars.mb_do_cg1_done[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if nvvm.elect_sync():
                bars.mb_do_ready[do_idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sdO_tma[do_idx], do_slice, bars.mb_do_ready[do_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # S = H[c-1]  (forward state entering chunk c; S0 for c = 0
            # when an initial state is given, none otherwise)
            # ----------------------------------------------------------
            if chunk_idx >= S_MIN:
                s_idx = s_index.idx
                bars.mb_s_done[s_idx].wait(s_index.phase)
                s_index = advance(s_index, cfg.smem_s_stages)
                if nvvm.elect_sync():
                    bars.mb_s_ready[s_idx].arrive(n_bytes=cfg.tma_s_bytes)
                if cutlass.const_expr(cfg.use_initial_state):
                    if chunk_idx == 0:
                        s0_slice = tma_slice_runtime_desc(desc_s0_slot, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))
                        tma_load_tile(sS_tma[s_idx], s0_slice, bars.mb_s_ready[s_idx].smem_ptr, acquire=False)
                    else:
                        s_slice = tma_slice_runtime_desc(desc_s_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx - 1)
                        tma_load_tile(sS_tma[s_idx], s_slice, bars.mb_s_ready[s_idx].smem_ptr, acquire=False)
                else:
                    s_slice = tma_slice_runtime_desc(desc_s_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx - S_MIN)
                    tma_load_tile(sS_tma[s_idx], s_slice, bars.mb_s_ready[s_idx].smem_ptr, acquire=False)

        tile_idx, sched_state = _sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_mma_done[q_index.idx].wait(q_index.phase)
        bars.mb_q_cg1_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_mma_done[k_index.idx].wait(k_index.phase)
        bars.mb_k_cg0_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_mma_done[v_index.idx].wait(v_index.phase)
        bars.mb_v_cg1_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    for _ in range(cfg.smem_do_stages):
        bars.mb_do_mma_done[do_index.idx].wait(do_index.phase)
        bars.mb_do_cg1_done[do_index.idx].wait(do_index.phase)
        do_index = advance(do_index, cfg.smem_do_stages)


@cute.jit
def _compute0_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    tmem_hold,
    scale,
    sCumsumlog,
    sCumprod,
    sBeta,
    sAinv,
    sKK,
    sQk,
    sDa,
    sDm,
    sK,
    sdQ,
    sdH,
    ss_flat,
    sdh_flat,
    sSched,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): persistent scheduler loop +
    per-chunk T-pairwise, kk_epi, hierarchical blockwise inverse, qk_epi."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    qk_index = PipelineState.start(phase=1)
    ainv_index = PipelineState.start(phase=1)
    da_acc_index = PipelineState.start(phase=0)
    dm_rdy_index = PipelineState.start(phase=0)
    cg0_dbeta_index = PipelineState.start(phase=0)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_id = cg0_tidx % cfg.threads_per_warp
    bpe = cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    store_row = warp_id * 16 + lane_id % 16
    store_col = (lane_id // 16) * 8
    tmem_warp_row = warp_id * cfg.threads_per_warp
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_dop_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_ks_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_spath_col = tmem_shared_inp_col
    acc_zero = cfg.acc_dtype(0.0)
    mask_zero = opaque_f32_zero()
    ov_tok = cg0_tidx % 8 + (cg0_tidx // 16 % 2) * 8
    ov_col = (cg0_tidx // 8 % 2) * 8 + (cg0_tidx // 32 % 2) * 32
    ov_slab = (cg0_tidx // 64) * 4096
    sK_base_p = cute.make_ptr(cfg.io_dtype, sK[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    k_stage_elems_cg0 = cfg.k_cosize // cfg.smem_k_stages
    sdQ_base = cute.make_ptr(cfg.io_dtype, sdQ[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sdH_parts_p = cute.make_ptr(cfg.io_dtype, sdH[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    # dG fold scratch in sKK (NOT sAinv: its upper triangle is kernel-start
    # zeros, never rewritten); kk_epi fully rewrites sKK every chunk and its
    # readers (inverse, M-terms dot) precede the fold in CG0 program order
    skk_red = cute.make_ptr(cutlass.Float32, sKK[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    cg0_k_index = PipelineState.start(phase=0)
    cg0_dgate_index = PipelineState.start(phase=0)
    tmem_dvdk_col = tmem_base + cfg.tmem_dvdk_offset
    tmem_dh_inp_col = tmem_base + cfg.tmem_dh_inp_offset
    cg0_kk_rdy = PipelineState.start(phase=0)
    cg0_a_rdy = PipelineState.start(phase=0)
    cg0_dk_scale_rdy = PipelineState.start(phase=0)
    cg0_dq_scale_rdy = PipelineState.start(phase=0)
    cg0_dhs_rdy = PipelineState.start(phase=0)
    cg0_dk_attn_rdy = PipelineState.start(phase=0)
    DHT0 = 1 if cfg.use_dht else 0

    ainv_zero_ptr = cute.make_ptr(cutlass.Int32, sAinv[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    for z in cutlass.range_constexpr(cfg.ainv_cosize * bpe // 4 // num_threads_cg0):
        (ainv_zero_ptr + cg0_tidx + z * num_threads_cg0).store(cutlass.Int32(0))

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = cend - wstart

        for chunk_idx in cutlass.range(sk_nt):
            # ---- Step 1: T-pairwise ------------------------------------
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            gT = []
            gT_strict = []
            for k in cutlass.range_constexpr(num_vals):
                crow = warp_id * 16 + lane_id // 4 + ((k // 2) % 2) * 8
                ccol = (lane_id % 4) * 2 + ((k // 4) * 8 + k % 2)
                gT.append(cute.math.exp2(sCumsumlog[crow, 0, gate_idx] - sCumsumlog[ccol, 0, gate_idx], fastmath=True) if crow >= ccol else mask_zero)
                gT_strict.append(mask_zero if crow == ccol else gT[k])
            gDecayScale = []
            last_cs = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
            for k in cutlass.range_constexpr(num_vals):
                gDecayScale.append(cute.math.exp2(last_cs - sCumsumlog[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx], fastmath=True))
            cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

            beta_idx = beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)

            gBeta = []
            for k in cutlass.range_constexpr(num_vals):
                crow = warp_id * 16 + lane_id // 4 + ((k // 2) % 2) * 8
                gBeta.append(sBeta[crow, 0, beta_idx])

            # ---- Step 2: kk_epi:  M_kk[i,j] = W_kk[i,j] * T[i,j] * beta[i] ----
            ainv_idx = ainv_index.idx
            ainv_phase = ainv_index.phase
            ainv_index = advance(ainv_index, cfg.smem_ainv_stages)
            bars.mb_kk_acc_ready[0].wait(cg0_kk_rdy.phase)
            cg0_kk_rdy = advance(cg0_kk_rdy, 1)

            ainv_base = sAinv[ainv_idx].base
            kk_base = sKK[ainv_idx].base
            kk_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_kk_col, cutlass.Float32), num=8)
            kk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(kk_vec[2 * k], kk_vec[2 * k + 1], gT[2 * k], gT[2 * k + 1])
                v0, v1 = fmul2(p0, p1, gBeta[2 * k], gBeta[2 * k + 1])
                kk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        kk_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [kk_f16[c * 4 + 0], kk_f16[c * 4 + 1], kk_f16[c * 4 + 2], kk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            if chunk_idx < cend - S_MIN:
                bars.mb_kk_acc_done[0].arrive()

            # ---- Step 3: qk_epi:  W_qkv[i,j] = W_qk[i,j] * T[i,j] * scale ----
            qk_idx = qk_index.idx
            bars.mb_qk_done[qk_idx].wait(qk_index.phase)
            qk_index = advance(qk_index, cfg.smem_qk_stages)
            bars.mb_a_acc_ready[0].wait(cg0_a_rdy.phase)
            cg0_a_rdy = advance(cg0_a_rdy, 1)

            qk_base = sQk[qk_idx].base
            qk_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_a_col, cutlass.Float32), num=8)
            qk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(qk_vec[2 * k], qk_vec[2 * k + 1], gT[2 * k], gT[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                qk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        qk_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [qk_f16[c * 4 + 0], qk_f16[c * 4 + 1], qk_f16[c * 4 + 2], qk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_ready[qk_idx].arrive()

            # ---- Step 4: blockwise inverse:  A_inv = -------------------
            bars.mb_ainv_done[ainv_idx].wait(ainv_phase)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )
            if warp_id < 2:
                _invert_diagonal_NxN(cfg, kk_base, ainv_base, cg0_tidx // 8, cg0_tidx, 8)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            _blockwise_diagonal_8x8_to_16x16(cfg, ainv_base, kk_base, warp_id * 16, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                _blockwise_diagonal_16x16_to_32x32(cfg, ainv_base, kk_base, warp_id * 32, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                _blockwise_diagonal_32x32_to_64x64(cfg, ainv_base, kk_base, warp_id, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # ---- beta column-scaling in place:  A_inv[i,j] *= beta[j] ----
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta_idx])
            ainv_f16 = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                ainv_f16 += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            ainv_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                            cutlass.AddressSpace.smem,
                            cutlass.BFloat16,
                        ),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                )
            ainv_scaled = []
            for j in cutlass.range_constexpr(num_vals // 2):
                lo, hi = f16x2_to_f32(ainv_f16[j], dtype=cfg.io_dtype)
                s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
                ainv_scaled.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        ainv_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [ainv_scaled[c * 4 + 0], ainv_scaled[c * 4 + 1], ainv_scaled[c * 4 + 2], ainv_scaled[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )

            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ainv_ready[ainv_idx].arrive()

            # ---- dQ inter rescale: gCumprod * scale in place (CG0-owned;
            # the scale_ready wait also covers the hdh sS read via ks) ----
            if chunk_idx < cend - S_MIN:
                gCumprod = []
                for k in cutlass.range_constexpr(num_vals):
                    gCumprod.append(sCumprod[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])
                bars.mb_dq_acc_scale_ready[0].wait(cg0_dq_scale_rdy.phase)
                cg0_dq_scale_rdy = advance(cg0_dq_scale_rdy, 1)
                # all loads issue before the first store: a TMEM store between
                # loads pins ptxas to one load latency per sub
                dqi_ptrs = []
                dqi_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dqi_ptrs.append(nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dh_inp_col, cutlass.Float32))
                    dqi_vecs.append(nvvm.tcgen05_ld("16x256b", dqi_ptrs[sub], num=8))
                for sub in cutlass.range_constexpr(2):
                    dqi_scaled = []
                    for j in cutlass.range_constexpr(16):
                        p0, p1 = fmul2(dqi_vecs[sub][2 * j], dqi_vecs[sub][2 * j + 1], gCumprod[2 * j], gCumprod[2 * j + 1])
                        s0, s1 = fmul2(p0, p1, scale, scale)
                        dqi_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dqi_ptrs[sub], cutlass.Vector.from_elements(tuple(dqi_scaled), cutlass.Float32))
                nvvm.tcgen05_wait("store")
                bars.mb_dq_acc_scale_done[0].arrive()

            # ---- dg_last H ⊙ dH term (octet-vectorized LDS.128); the sS
            # read is released via the mb_s_done arrive below.  The sdH
            # staging is gated by a dhs_ready mirror-wait (non-consuming) ----
            if chunk_idx + DHT0 >= 1:
                bars.mb_dhs_ready[0].wait(cg0_dhs_rdy.phase)
                cg0_dhs_rdy = advance(cg0_dhs_rdy, 1)
            sdh_base = sdh_flat.iterator.toint()
            ss_base = ss_flat.iterator.toint()
            hdh_lo = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
            hdh_hi = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
            for oct_ in cutlass.range_constexpr(cfg.d_v // 8):
                hs0 = (oct_ // 8) * (cfg.d_k * 64) + cg0_tidx * 64 + swizzle_xor_128b(cg0_tidx, (oct_ % 8) * 8)
                dw = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, sdh_base + hs0 * bpe, mem_space=cute.AddressSpace.smem, assumed_align=16),
                    cute.make_layout(4),
                ).load()
                sw = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, ss_base + hs0 * bpe, mem_space=cute.AddressSpace.smem, assumed_align=16),
                    cute.make_layout(4),
                ).load()
                for w in cutlass.range_constexpr(4):
                    d_lo, d_hi = f16x2_to_f32(dw[w], dtype=cfg.io_dtype)
                    s_lo, s_hi = f16x2_to_f32(sw[w], dtype=cfg.io_dtype)
                    hdh_lo[w], hdh_hi[w] = ffma2(d_lo, d_hi, s_lo, s_hi, hdh_lo[w], hdh_hi[w])
            hdh = ((hdh_lo[0] + hdh_lo[1]) + (hdh_lo[2] + hdh_lo[3])) + ((hdh_hi[0] + hdh_hi[1]) + (hdh_hi[2] + hdh_hi[3]))
            for off in [1, 2, 4, 8, 16]:
                hdh += nvvm.shfl_sync(0xFFFFFFFF, hdh, off, 31, kind=nvvm.Shfl.BFLY)
            bars.mb_hdh_done[0].arrive()
            if chunk_idx < cend - S_MIN:
                nvvm.fence_proxy("async.shared", space="cta")
                nvvm.mbarrier_arrive(bars.mb_s_done[0].smem_ptr)

            # ---- dK inter rescale: gDecayScale in place + the skd dot ----
            cg0_k_idx = cg0_k_index.idx
            cg0_k_index = advance(cg0_k_index, cfg.smem_k_stages)
            skd = cutlass.Float32(0.0)
            if chunk_idx + DHT0 >= 1:
                bars.mb_dk_scale_ready[0].wait(cg0_dk_scale_rdy.phase)
                cg0_dk_scale_rdy = advance(cg0_dk_scale_rdy, 1)
                skd_lo = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
                skd_hi = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
                dki_ptrs = []
                dki_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dki_ptrs.append(nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32))
                    dki_vecs.append(nvvm.tcgen05_ld("16x256b", dki_ptrs[sub], num=8))
                for sub in cutlass.range_constexpr(2):
                    dki_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(dki_vecs[sub][2 * j], dki_vecs[sub][2 * j + 1], gDecayScale[2 * j], gDecayScale[2 * j + 1])
                        dki_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dki_ptrs[sub], cutlass.Vector.from_elements(tuple(dki_scaled), cutlass.Float32))
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        k_f16 = nvvm.ldmatrix((sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(), 4, nvvm.MMALayout.COL)
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_f16[i], dtype=cfg.io_dtype)
                            skd_lo[i], skd_hi[i] = ffma2(dki_scaled[8 * m0 + 2 * i], dki_scaled[8 * m0 + 2 * i + 1], k_lo, k_hi, skd_lo[i], skd_hi[i])
                nvvm.tcgen05_wait("store")
                bars.mb_dk_scale_done[0].arrive()
                skd = ((skd_lo[0] + skd_lo[1]) + (skd_lo[2] + skd_lo[3])) + ((skd_hi[0] + skd_hi[1]) + (skd_hi[2] + skd_hi[3]))
                for off in [1, 2, 4, 8, 16]:
                    skd += nvvm.shfl_sync(0xFFFFFFFF, skd, off, 31, kind=nvvm.Shfl.BFLY)

            # ---- Step 6: dA epilogue -----------------------------------
            bars.mb_da_acc_ready[0].wait(da_acc_index.phase)
            da_acc_index = advance(da_acc_index, 1)

            da_base = sDa[0].base
            da_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_da_col, cutlass.Float32), num=8)
            da_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(da_vec[2 * k], da_vec[2 * k + 1], gT[2 * k], gT[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                da_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        da_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [da_f16[c * 4 + 0], da_f16[c * 4 + 1], da_f16[c * 4 + 2], da_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[0].arrive()

            # ---- Step 8: dM epilogue -----------------------------------
            bars.mb_dm_ready[0].wait(dm_rdy_index.phase)
            dm_rdy_index = advance(dm_rdy_index, 1)
            dm_base = sDm[0].base
            dm_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dm_core_col, cutlass.Float32), num=8)

            dm_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(dm_vec[2 * k], dm_vec[2 * k + 1], gT_strict[2 * k], gT_strict[2 * k + 1])
                v0 = cutlass.Float32(0.0) - p0
                v1 = cutlass.Float32(0.0) - p1
                dm_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        dm_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [dm_f16[c * 4 + 0], dm_f16[c * 4 + 1], dm_f16[c * 4 + 2], dm_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_done[0].arrive()

            # ---- dK attn read: 16x256b fragment view for the part_k dot ----
            bars.mb_dk_attn_ready[0].wait(cg0_dk_attn_rdy.phase)
            cg0_dk_attn_rdy = advance(cg0_dk_attn_rdy, 1)
            dks_frag = []
            for sub in cutlass.range_constexpr(2):
                dks_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32),
                    num=8,
                )
                dks_frag.append([dks_vec[k] for k in range(32)])
            nvvm.tcgen05_wait("load")
            bars.mb_dk_attn_done[0].arrive()

            # ---- dBeta/dGate M-terms: E = strict ⊙ dm_core ⊙ M_kk(sKK). ----
            kk_frag = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                kk_frag += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            kk_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                            cutlass.AddressSpace.smem,
                            cutlass.BFloat16,
                        ),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                )
            binv_row = [
                cute.math.rcp(gBeta[0] + cutlass.Float32(1e-10), approx=True, ftz=True),
                cute.math.rcp(gBeta[2] + cutlass.Float32(1e-10), approx=True, ftz=True),
            ]
            row_acc = [cutlass.Float32(0.0)] * 8
            col_part = [opaque_f32_zero() for _ in range(16)]
            for j in cutlass.range_constexpr(num_vals // 2):
                klo, khi = f16x2_to_f32(kk_frag[j], dtype=cfg.io_dtype)
                binv_j = binv_row[j % 2]
                p_lo, p_hi = fmul2(dm_vec[2 * j], dm_vec[2 * j + 1], klo, khi)
                e_lo, e_hi = fmul2(p_lo, p_hi, binv_j, binv_j)
                crow = warp_id * 16 + lane_id // 4 + (j % 2) * 8
                ccol = (lane_id % 4) * 2 + (j // 2) * 8
                e_val_lo = e_lo if crow > ccol else acc_zero
                e_val_hi = e_hi if crow > ccol + 1 else acc_zero
                row_acc[(j % 2) * 4 + (j // 2) % 4] += e_val_lo + e_val_hi
                c0 = cutlass.const_expr((j // 2) * 2)
                col_part[c0], col_part[c0 + 1] = fadd2(col_part[c0], col_part[c0 + 1], e_val_lo, e_val_hi)
            row_part = [
                (row_acc[0] + row_acc[1]) + (row_acc[2] + row_acc[3]),
                (row_acc[4] + row_acc[5]) + (row_acc[6] + row_acc[7]),
            ]
            for off in [1, 2]:
                for rp in cutlass.range_constexpr(2):
                    row_part[rp] += nvvm.shfl_sync(0xFFFFFFFF, row_part[rp], off, 31, kind=nvvm.Shfl.BFLY)
            # CG1's v-terms assign the dBeta base into sBeta; fold the k-term on top
            bars.mb_dbeta_cg1_ready[0].wait(cg0_dbeta_index.phase)
            cg0_dbeta_index = advance(cg0_dbeta_index, 1)
            if lane_id % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                    sBeta[crow_r, 0, beta_idx] = sBeta[crow_r, 0, beta_idx] - row_part[rp] * binv_row[rp]
            # ---- part reductions: fragment dot (K frags from sK, dks from
            # the 16x256b r4 view), col_part folded in, reduce-scatter ----
            if chunk_idx + DHT0 >= S_MIN:
                part_k = [acc_zero] * 16
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        k_f16 = nvvm.ldmatrix((sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(), 4, nvvm.MMALayout.COL)
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_f16[i], dtype=cfg.io_dtype)
                            kk0 = cutlass.const_expr(8 * m0 + 2 * i)
                            j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                            if cutlass.const_expr(sub == 0 and i % 2 == 0):
                                part_k[j0], part_k[j0 + 1] = fmul2(dks_frag[sub][kk0], dks_frag[sub][kk0 + 1], k_lo, k_hi)
                            else:
                                part_k[j0], part_k[j0 + 1] = ffma2(dks_frag[sub][kk0], dks_frag[sub][kk0 + 1], k_lo, k_hi, part_k[j0], part_k[j0 + 1])
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_k_cg0_done[cg0_k_idx].arrive()

                am = [col_part[j] - part_k[j] for j in range(16)]
                am_lo, am_hi = _warp_reduce_scatter_frag16(am, lane_id)
                dg_last_w = skd + ((cumprod_total * hdh if chunk_idx + DHT0 >= 1 else acc_zero) if chunk_idx < cend - S_MIN else acc_zero)
                tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                (skk_red + warp_id * 64 + tok0).store(am_lo)
                (skk_red + warp_id * 64 + tok0 + 1).store(am_hi + dg_last_w if lane_id == 31 else am_hi)

                # CG1 adds part_q to dGate first; fold CG0's terms on top
                bars.mb_dgate_cg1_ready[0].wait(cg0_dgate_index.phase)
                cg0_dgate_index = advance(cg0_dgate_index, 1)
                if lane_id % 4 == 0:
                    for rp in cutlass.range_constexpr(2):
                        crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                        sCumsumlog[crow_r, 0, gate_idx] = sCumsumlog[crow_r, 0, gate_idx] - row_part[rp]
                nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
                if cg0_tidx < 64:
                    dg_sum = (
                        (skk_red + cg0_tidx).load() + (skk_red + 64 + cg0_tidx).load() + (skk_red + 128 + cg0_tidx).load() + (skk_red + 192 + cg0_tidx).load()
                    )
                    sCumsumlog[cg0_tidx, 0, gate_idx] = sCumsumlog[cg0_tidx, 0, gate_idx] + dg_sum

            if chunk_idx + DHT0 < S_MIN:
                part_k = [acc_zero] * 16
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        k_f16 = nvvm.ldmatrix((sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(), 4, nvvm.MMALayout.COL)
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_f16[i], dtype=cfg.io_dtype)
                            kk0 = cutlass.const_expr(8 * m0 + 2 * i)
                            j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                            if cutlass.const_expr(sub == 0 and i % 2 == 0):
                                part_k[j0], part_k[j0 + 1] = fmul2(dks_frag[sub][kk0], dks_frag[sub][kk0 + 1], k_lo, k_hi)
                            else:
                                part_k[j0], part_k[j0 + 1] = ffma2(dks_frag[sub][kk0], dks_frag[sub][kk0 + 1], k_lo, k_hi, part_k[j0], part_k[j0 + 1])
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_k_cg0_done[cg0_k_idx].arrive()

                am = [col_part[j] - part_k[j] for j in range(16)]
                am_lo, am_hi = _warp_reduce_scatter_frag16(am, lane_id)
                tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                (skk_red + warp_id * 64 + tok0).store(am_lo)
                (skk_red + warp_id * 64 + tok0 + 1).store(am_hi)

                # CG1 adds part_q to dGate first; fold CG0's terms on top
                bars.mb_dgate_cg1_ready[0].wait(cg0_dgate_index.phase)
                cg0_dgate_index = advance(cg0_dgate_index, 1)
                if lane_id % 4 == 0:
                    for rp in cutlass.range_constexpr(2):
                        crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                        sCumsumlog[crow_r, 0, gate_idx] = sCumsumlog[crow_r, 0, gate_idx] - row_part[rp]
                nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
                if cg0_tidx < 64:
                    dg_sum = (
                        (skk_red + cg0_tidx).load() + (skk_red + 64 + cg0_tidx).load() + (skk_red + 128 + cg0_tidx).load() + (skk_red + 192 + cg0_tidx).load()
                    )
                    sCumsumlog[cg0_tidx, 0, gate_idx] = sCumsumlog[cg0_tidx, 0, gate_idx] + dg_sum

            bars.mb_gate_done[gate_idx].arrive()
            bars.mb_beta_done[beta_idx].arrive()
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
    for _ in range(cfg.smem_qk_stages):
        bars.mb_qk_done[qk_index.idx].wait(qk_index.phase)
        qk_index = advance(qk_index, cfg.smem_qk_stages)
    for _ in range(cfg.smem_ainv_stages):
        bars.mb_ainv_done[ainv_index.idx].wait(ainv_index.phase)
        ainv_index = advance(ainv_index, cfg.smem_ainv_stages)

    # CG0 done with TMEM: release the MMA warp's dealloc
    bars.mb_tmem_done[0].arrive()


@cute.jit
def _compute1_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    mDs0_out,
    mDht,
    tidx,
    warp_idx,
    tmem_hold,
    scale,
    sQ,
    sK,
    sV,
    sdO,
    sCumsumlog,
    sCumprod,
    sBeta,
    sdQ,
    sdK,
    sdV,
    sdH,
    sDa,
    sDm,
    sSched,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop and
    per-chunk (BACKWARD order): dV-acc / dQ-acc
    in-place decay scales, dO'/dU/dY' TMEM restages, Y and gks staging,
    u readout over sV, the dV-pass dG/dBeta v-terms, dV/dK staging with the
    dK s-path fold, and the next-chunk dH prep at the chunk tail."""

    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    cg1_ks_rdy = PipelineState.start(phase=0)
    cg1_u_rdy = PipelineState.start(phase=0)
    cg1_dy_rdy = PipelineState.start(phase=0)
    cg1_du_scale_rdy = PipelineState.start(phase=0)
    cg1_du_total_rdy = PipelineState.start(phase=0)
    cg1_dk_total_rdy = PipelineState.start(phase=0)
    dh_acc_index = PipelineState.start(phase=0)
    dop_inp_free = PipelineState.start(phase=1)
    dyp_inp_free = PipelineState.start(phase=1)
    dhs_index = PipelineState.start(phase=1)
    cg1_hdh_index = PipelineState.start(phase=0)
    dq_index = PipelineState.start(phase=1)
    cg1_beta_index = PipelineState.start(phase=0)
    sdv_done_index = PipelineState.start(phase=1)
    cg1_dk_spath_rdy = PipelineState.start(phase=0)
    dq_total_rdy_index = PipelineState.start(phase=0)
    dh_inp_index = PipelineState.start(phase=1)
    dk_index = PipelineState.start(phase=1)
    dv_index = PipelineState.start(phase=1)

    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_id = cg1_tidx % cfg.threads_per_warp
    tmem_warp_row = (cg1_tidx // cfg.threads_per_warp) * cfg.threads_per_warp
    ldtm_width = 32
    sttm_width = ldtm_width // 2
    num_state_subs = cutlass.const_expr(cfg.d_v // ldtm_width)
    tmem_dh_col = tmem_base + cfg.tmem_dh_offset
    tmem_dh_inp_col = tmem_base + cfg.tmem_dh_inp_offset
    tmem_dvdk_col = tmem_base + cfg.tmem_dvdk_offset
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_dop_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_y_col = tmem_base + cfg.tmem_y_offset
    tmem_gks_col = tmem_y_col + SHARED_INP_STAGE_COLS
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_ks_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_spath_col = tmem_shared_inp_col
    ov_tok = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    ov_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
    ov_slab = (cg1_tidx // 64) * 4096
    dv_stage_elems = cfg.dv_cosize // cfg.smem_dv_stages
    sdV_base = cute.make_ptr(cfg.io_dtype, sdV[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    v_stage_elems = cfg.v_cosize // cfg.smem_v_stages
    sV_base = cute.make_ptr(cfg.io_dtype, sV[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sQ_base = cute.make_ptr(cfg.io_dtype, sQ[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    do_stage_elems = cfg.do_cosize // cfg.smem_do_stages
    sdO_base = cute.make_ptr(cfg.io_dtype, sdO[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    dk_stage_elems = cfg.dk_cosize // cfg.smem_dk_stages
    dq_stage_elems = cfg.dq_cosize // cfg.smem_dq_stages
    sdQ_base = cute.make_ptr(cfg.io_dtype, sdQ[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sdK_base = cute.make_ptr(cfg.io_dtype, sdK[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sdH_base_int = sdH[0].base
    # dBeta/dGate reduction scratch: f32 view of the owned dQ stage (rebased per chunk)
    sred_base = cute.make_ptr(cutlass.Float32, sdQ[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    # part_q scratch in sdH: consumed by dK-inter (covered by total_ready =
    # the dQ-attn commit) and re-staged by CG1 itself later in the iteration
    sdh_red = cute.make_ptr(cutlass.Float32, sdH[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    dh_done_idx = cutlass.Int32(0)
    cg1w = cg1_tidx // cfg.threads_per_warp

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    S_MIN = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = cend - wstart
        # ---- d_final_state prologue: seed the dH acc (f32) + the f16
        # entry staging (dh_inp + sdH) ----
        if cutlass.const_expr(cfg.use_dht):
            if sk_nt > 0:
                # split-K: only the item owning the sequence tail receives
                # the true d_final_state; warmup items rebuild dH from zero
                gDht = mDht[None, None, head_idx, batch_idx]
                seed_from_dht = cend == num_chunks_b
                dh_inp_idx = dh_inp_index.idx
                bars.mb_dh_inp_done[dh_inp_idx].wait(dh_inp_index.phase)
                dh_inp_index = advance(dh_inp_index, cfg.tmem_dh_inp_stages)
                for sub in cutlass.range_constexpr(num_state_subs):
                    dht_vals = []
                    for kk in cutlass.range_constexpr(ldtm_width):
                        v = gDht[sub * ldtm_width + kk, cg1_tidx]
                        if cutlass.const_expr(cfg.split_k):
                            v = v if seed_from_dht else cutlass.Float32(0.0)
                        dht_vals.append(v)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(dht_vals), cutlass.Float32),
                    )
                    dht_f16 = [fp32_to_fp16(dht_vals[2 * j], dht_vals[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_inp_col + sub * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dht_f16), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dh_inp_ready[dh_inp_idx].arrive()

                # sdH staging read back from the just-seeded acc
                bars.mb_dhs_done[0].wait(dhs_index.phase)
                dhs_index = advance(dhs_index, 1)
                dhs_vecs = []
                for b in cutlass.range_constexpr(2):
                    for hh in cutlass.range_constexpr(2):
                        dhs_vecs.append(
                            nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(((tmem_warp_row + b * 16) << 16) + tmem_dh_col + hh * 64, cutlass.Float32),
                                num=8,
                            )
                        )
                for b in cutlass.range_constexpr(2):
                    for hh in cutlass.range_constexpr(2):
                        dhs_vec = dhs_vecs[b * 2 + hh]
                        dhs_f16 = [fp32_to_fp16(dhs_vec[2 * j], dhs_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                        for c in cutlass.range_constexpr(4):
                            dhs_row = hh * 64 + ov_tok + c * 16
                            nvvm.stmatrix(
                                cutlass.inttoptr(
                                    sdH_base_int + ((cg1_tidx // 64) * cfg.d_k * 64 + dhs_row * 64 + swizzle_xor_128b(dhs_row, ov_col + b * 16)) * 2,
                                    cutlass.AddressSpace.smem,
                                    cutlass.BFloat16,
                                ),
                                [dhs_f16[c * 4 + 0], dhs_f16[c * 4 + 1], dhs_f16[c * 4 + 2], dhs_f16[c * 4 + 3]],
                                nvvm.MMALayout.COL,
                            )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dhs_ready[0].arrive()

        if cutlass.const_expr(not cfg.use_dht):
            if sk_nt > 0:
                # ---- first backward chunk (c = NT-1): no dH yet -------------------
                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                beta_idx = cg1_beta_index.idx
                bars.mb_beta_ready[beta_idx].wait(cg1_beta_index.phase)
                cg1_beta_index = advance(cg1_beta_index, cfg.smem_beta_stages)
                num_vals = 32
                gCumprod = []
                for k in cutlass.range_constexpr(num_vals):
                    gCumprod.append(sCumprod[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])

                # ---- dO' restage: dO * gCumprod * scale -> shared_inp TMEM ----
                bars.mb_dop_inp_done[0].wait(dop_inp_free.phase)
                dop_inp_free = advance(dop_inp_free, 1)
                do_idx = do_index.idx
                bars.mb_do_ready[do_idx].wait(do_index.phase)
                do_index = advance(do_index, cfg.smem_do_stages)
                do_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
                for c in cutlass.range_constexpr(8):
                    m0 = cutlass.const_expr(c % 4)
                    sub = cutlass.const_expr(c // 4)
                    do_f16 = nvvm.ldmatrix(
                        (
                            sdO_base + do_idx * do_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        ).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        lo, hi = f16x2_to_f32(do_f16[i], dtype=cfg.io_dtype)
                        p0, p1 = fmul2(lo, hi, gCumprod[8 * m0 + 2 * i], gCumprod[8 * m0 + 2 * i + 1])
                        do_regs[8 * m0 + 2 * i][sub], do_regs[8 * m0 + 2 * i + 1][sub] = fmul2(p0, p1, scale, scale)
                for sub in cutlass.range_constexpr(2):
                    do_pack = [fp32_to_fp16(do_regs[2 * j][sub], do_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dop_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(do_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dop_inp_ready[0].arrive()

                # ---- v - k*state:  delta = V - cumprod*(K @ S), in place ----
                v_idx = v_index.idx
                bars.mb_v_ready[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)
                if cend >= SFIRST_MIN:
                    # V stays PACKED in the io dtype; the scaled KS is packed
                    # (needed for its own TMEM store anyway) and subtracted
                    # with packed 16-bit ops.
                    v_words = [[cutlass.Int32(0), cutlass.Int32(0)] for _ in range(16)]
                    for c in cutlass.range_constexpr(8):
                        m0 = cutlass.const_expr(c % 4)
                        sub = cutlass.const_expr(c // 4)
                        v_f16 = nvvm.ldmatrix(
                            (
                                sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            v_words[4 * m0 + i][sub] = v_f16[i]
                    bars.mb_ks_acc_ready[0].wait(cg1_ks_rdy.phase)
                    cg1_ks_rdy = advance(cg1_ks_rdy, 1)
                    ks_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        ks_vecs.append(
                            nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_ks_col, cutlass.Float32),
                                num=8,
                            )
                        )
                    for sub in cutlass.range_constexpr(2):
                        gks_pack = []
                        y_pack = []
                        for j in cutlass.range_constexpr(16):
                            g0, g1 = fmul2(ks_vecs[sub][2 * j], ks_vecs[sub][2 * j + 1], gCumprod[2 * j], gCumprod[2 * j + 1])
                            gks_word = fp32_to_fp16(g0, g1, dtype=cfg.io_dtype)
                            gks_pack.append(gks_word)
                            y_pack.append(sub_f16x2(v_words[j][sub], gks_word, cfg.io_dtype))
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(y_pack), cutlass.Int32),
                        )
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_gks_col, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(gks_pack), cutlass.Int32),
                        )
                if cend < SFIRST_MIN:
                    for sub in cutlass.range_constexpr(2):
                        v_pack = []
                        for m0 in cutlass.range_constexpr(4):
                            v_f16 = nvvm.ldmatrix(
                                (
                                    sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                                ).raw_ptr(),
                                4,
                                nvvm.MMALayout.COL,
                            )
                            for i in cutlass.range_constexpr(4):
                                v_pack.append(v_f16[i])
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(v_pack), cutlass.Int32),
                        )
                nvvm.tcgen05_wait("store")
                bars.mb_y_ready[0].arrive()

                # ---- dU restage: dv acc ------------------------------------
                bars.mb_dyp_inp_done[0].wait(dyp_inp_free.phase)
                dyp_inp_free = advance(dyp_inp_free, 1)
                bars.mb_du_total_ready[0].wait(cg1_du_total_rdy.phase)
                cg1_du_total_rdy = advance(cg1_du_total_rdy, 1)
                du_vecs = []
                for sub in cutlass.range_constexpr(2):
                    du_vecs.append(nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8))
                for sub in cutlass.range_constexpr(2):
                    du_pack = [fp32_to_fp16(du_vecs[sub][2 * j], du_vecs[sub][2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_du_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(du_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_du_inp_ready[0].arrive()
                bars.mb_do_cg1_done[do_idx].arrive()

                # ---- U readout ---------------------------------------------
                bars.mb_u_acc_ready[0].wait(cg1_u_rdy.phase)
                cg1_u_rdy = advance(cg1_u_rdy, 1)
                u_regs = []
                for sub in cutlass.range_constexpr(2):
                    u_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_u_col, cutlass.Float32), num=8)
                    u_regs.append([u_vec[k] for k in range(32)])
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        u_f16 = [fp32_to_fp16(u_regs[sub][8 * m0 + 2 * j], u_regs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            u_f16,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_u_ready[0].arrive()
                bars.mb_v_cg1_done[v_idx].arrive()

                # ---- dY ----------------------------------------------------
                bars.mb_dy_acc_ready[0].wait(cg1_dy_rdy.phase)
                cg1_dy_rdy = advance(cg1_dy_rdy, 1)
                dy_regs = []
                for sub in cutlass.range_constexpr(2):
                    dy_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dy_col, cutlass.Float32),
                        num=8,
                    )
                    dy_regs.append([dy_vec[k] for k in range(32)])

                # ---- dY' = -gCumprod * dY -> f16 shared_inp ----------------
                neg_one = cutlass.Float32(-1.0)
                gCumprodNeg = [gCumprod[k] * neg_one for k in range(32)]
                for sub in cutlass.range_constexpr(2):
                    dyp = []
                    for j in cutlass.range_constexpr(16):
                        n0, n1 = fmul2(dy_regs[sub][2 * j], dy_regs[sub][2 * j + 1], gCumprodNeg[2 * j], gCumprodNeg[2 * j + 1])
                        dyp += [n0, n1]
                    dyp_pack = [fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dyp_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dyp_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dyp_inp_ready[0].arrive()

                # ---- dV staging: dV = dY -> the sdV slot -------------------
                dv_stg_idx = dv_index.idx
                bars.mb_dv_tmastg_done[dv_stg_idx].wait(dv_index.phase)
                dv_index = advance(dv_index, cfg.smem_dv_stages)
                bars.mb_sdv_done[0].wait(sdv_done_index.phase)
                sdv_done_index = advance(sdv_done_index, 1)
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        dv_f16 = [fp32_to_fp16(dy_regs[sub][8 * m0 + 2 * j], dy_regs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sdV_base
                                + dv_stg_idx * dv_stage_elems
                                + ov_slab
                                + (ov_tok + m0 * 16) * 64
                                + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            dv_f16,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dv_tmastg_ready[dv_stg_idx].arrive()
                dk_stg_idx = dk_index.idx
                bars.mb_dk_tmastg_done[dk_stg_idx].wait(dk_index.phase)
                dk_index = advance(dk_index, cfg.smem_dk_stages)
                dq_stg_idx = dq_index.idx
                bars.mb_dq_tmastg_done[dq_stg_idx].wait(dq_index.phase)
                dq_index = advance(dq_index, cfg.smem_dq_stages)
                sred = sred_base + dq_stg_idx * (dq_stage_elems // 2)

                # ---- dBeta/dGate v-terms: dbeta_t += rowsum(dV ⊙ Y)_t / beta_t ----
                part_y = [cutlass.Float32(0.0)] * 16
                for sub in cutlass.range_constexpr(2):
                    y_pk = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32), num=8)
                    for j in cutlass.range_constexpr(16):
                        lo, hi = f16x2_to_f32(y_pk[j], dtype=cfg.io_dtype)
                        kk0 = cutlass.const_expr(2 * j)
                        j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                        if cutlass.const_expr(sub == 0 and j % 2 == 0):
                            part_y[j0], part_y[j0 + 1] = fmul2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi)
                        else:
                            part_y[j0], part_y[j0 + 1] = ffma2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi, part_y[j0], part_y[j0 + 1])
                py_lo, py_hi = _warp_reduce_scatter_frag16(part_y, lane_id)
                vt_tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                if cend >= SFIRST_MIN:
                    part_g = [cutlass.Float32(0.0)] * 16
                    for sub in cutlass.range_constexpr(2):
                        g_pk = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_gks_col, cutlass.Int32), num=8)
                        for j in cutlass.range_constexpr(16):
                            lo, hi = f16x2_to_f32(g_pk[j], dtype=cfg.io_dtype)
                            kk0 = cutlass.const_expr(2 * j)
                            j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                            if cutlass.const_expr(sub == 0 and j % 2 == 0):
                                part_g[j0], part_g[j0 + 1] = fmul2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi)
                            else:
                                part_g[j0], part_g[j0 + 1] = ffma2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi, part_g[j0], part_g[j0 + 1])
                    pg_lo, pg_hi = _warp_reduce_scatter_frag16(part_g, lane_id)
                    (sred + cg1w * 64 + vt_tok0).store(py_lo)
                    (sred + cg1w * 64 + vt_tok0 + 1).store(py_hi)
                    (sred + 256 + cg1w * 64 + vt_tok0).store(pg_lo)
                    (sred + 256 + cg1w * 64 + vt_tok0 + 1).store(pg_hi)
                    nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                    if cg1_tidx < 64:
                        binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                        ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                        gsum = (sred + 256 + cg1_tidx).load() + (sred + 320 + cg1_tidx).load() + (sred + 384 + cg1_tidx).load() + (sred + 448 + cg1_tidx).load()
                        sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                        sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0) - gsum
                if cend < SFIRST_MIN:
                    (sred + cg1w * 64 + vt_tok0).store(py_lo)
                    (sred + cg1w * 64 + vt_tok0 + 1).store(py_hi)
                    nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                    if cg1_tidx < 64:
                        binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                        ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                        sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                        sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0)
                bars.mb_beta_done[beta_idx].arrive()
                bars.mb_dbeta_cg1_ready[0].arrive()
                # sred reads must land before the dq stmatrix below reuses the stage
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                # ---- Q fragments held in registers over the dq stage (sQ
                # free once read; the dQ dot consumes them -- no TMEM trip) ----
                q_frag = []
                for sub in cutlass.range_constexpr(2):
                    q_words = []
                    for m0 in cutlass.range_constexpr(4):
                        q_f16 = nvvm.ldmatrix(
                            (sQ_base + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            q_words.append(q_f16[i])
                    q_frag.append(q_words)
                    # the store+wait order the sQ LDSM reads before the release arrive
                    # (a bare register consume can be hoisted past by ptxas); the dot
                    # below reads the registers, so the TMEM read-back is still gone
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(q_words), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_q_cg1_done[0].arrive()

                # ---- dq final read -> sdQ (output staging; the fragments are
                # held for the dQ dot below) ---------
                bars.mb_dq_acc_total_ready[0].wait(dq_total_rdy_index.phase)
                dq_total_rdy_index = advance(dq_total_rdy_index, 1)
                dq_frag = []
                for sub in cutlass.range_constexpr(2):
                    dqv = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dh_inp_col, cutlass.Float32), num=8)
                    dq_frag.append([dqv[k] for k in range(32)])
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        dq_f16 = [fp32_to_fp16(dq_frag[sub][8 * m0 + 2 * j], dq_frag[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix((sdQ_base + dq_stg_idx * dq_stage_elems + frag_addr).raw_ptr(), dq_f16, nvvm.MMALayout.COL)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dq_acc_total_done[0].arrive()
                bars.mb_dq_tmastg_ready[dq_stg_idx].arrive()

                # ---- dQ dot (part_q): fragment dot of the held dq acc with the
                # staged Q^T, added to dGate FIRST; CG0 adds after parts_ready ----
                part_q = [cutlass.Float32(0.0)] * 16
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(4):
                            q_lo, q_hi = f16x2_to_f32(q_frag[sub][4 * m0 + i], dtype=cfg.io_dtype)
                            kk0 = cutlass.const_expr(8 * m0 + 2 * i)
                            j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                            if cutlass.const_expr(sub == 0 and i % 2 == 0):
                                part_q[j0], part_q[j0 + 1] = fmul2(dq_frag[sub][kk0], dq_frag[sub][kk0 + 1], q_lo, q_hi)
                            else:
                                part_q[j0], part_q[j0 + 1] = ffma2(dq_frag[sub][kk0], dq_frag[sub][kk0 + 1], q_lo, q_hi, part_q[j0], part_q[j0 + 1])
                amq_lo, amq_hi = _warp_reduce_scatter_frag16(part_q, lane_id)
                tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                (sdh_red + (cg1_tidx // 32) * 64 + tok0).store(amq_lo)
                (sdh_red + (cg1_tidx // 32) * 64 + tok0 + 1).store(amq_hi)
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                if cg1_tidx < 64:
                    pq_sum = (
                        (sdh_red + cg1_tidx).load() + (sdh_red + 64 + cg1_tidx).load() + (sdh_red + 128 + cg1_tidx).load() + (sdh_red + 192 + cg1_tidx).load()
                    )
                    sCumsumlog[cg1_tidx, 0, gate_idx] = sCumsumlog[cg1_tidx, 0, gate_idx] + pq_sum
                bars.mb_gate_done[gate_idx].arrive()
                bars.mb_dgate_cg1_ready[0].arrive()

                # ---- NEXT-CHUNK dH prep (>= 2: single-chunk tiles have no
                # consumer and the dh_acc_ready consume would starve the drain) ----
                if sk_nt >= 2:
                    dh_idx = dh_acc_index.idx
                    bars.mb_dh_acc_ready[dh_idx].wait(dh_acc_index.phase)
                    dh_acc_index = advance(dh_acc_index, cfg.tmem_dh_acc_stages)
                    dh_done_idx = dh_idx
                    dh_inp_idx = dh_inp_index.idx
                    bars.mb_dh_inp_done[dh_inp_idx].wait(dh_inp_index.phase)
                    dh_inp_index = advance(dh_inp_index, cfg.tmem_dh_inp_stages)
                    dh_regs = [[cutlass.Float32(0.0) for _ in range(num_state_subs)] for _ in range(32)]
                    # all loads issue before the first store: a TMEM store
                    # between loads pins ptxas to one load latency per sub
                    dh_vecs = []
                    for sub in cutlass.range_constexpr(num_state_subs):
                        dh_vecs.append(
                            nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32), num=32)
                        )
                    for sub in cutlass.range_constexpr(num_state_subs):
                        for k in cutlass.range_constexpr(32):
                            dh_regs[k][sub] = dh_vecs[sub][k]

                        dh_f16 = [fp32_to_fp16(dh_regs[2 * j][sub], dh_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_inp_col + sub * sttm_width, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(dh_f16), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_dh_inp_ready[dh_inp_idx].arrive()

                # ---- dK s-path fold: read while the dM-terms GEMMs run ------------
                if cend >= SFIRST_MIN:
                    bars.mb_dk_spath_ready[0].wait(cg1_dk_spath_rdy.phase)
                    cg1_dk_spath_rdy = advance(cg1_dk_spath_rdy, 1)
                    dk_spath_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        dk_spath_vecs.append(
                            nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dk_spath_col, cutlass.Float32),
                                num=8,
                            )
                        )
                    nvvm.tcgen05_wait("load")
                    dk_spath_regs = []
                    for sub in cutlass.range_constexpr(2):
                        dk_spath_row = []
                        for j in cutlass.range_constexpr(16):
                            n0, n1 = fmul2(dk_spath_vecs[sub][2 * j], dk_spath_vecs[sub][2 * j + 1], gCumprodNeg[2 * j], gCumprodNeg[2 * j + 1])
                            dk_spath_row += [n0, n1]
                        dk_spath_regs.append(dk_spath_row)

                    bars.mb_dk_total_ready[0].wait(cg1_dk_total_rdy.phase)
                    cg1_dk_total_rdy = advance(cg1_dk_total_rdy, 1)
                    dmr_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        dmr_vecs.append(
                            nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8)
                        )
                    nvvm.tcgen05_wait("load")
                    bars.mb_dk_total_done[0].arrive()
                    for sub in cutlass.range_constexpr(2):
                        dk_sum = [dmr_vecs[sub][k] + dk_spath_regs[sub][k] for k in range(32)]
                        for m0 in cutlass.range_constexpr(4):
                            dk_f16 = [fp32_to_fp16(dk_sum[8 * m0 + 2 * j], dk_sum[8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                            nvvm.stmatrix(
                                (
                                    sdK_base
                                    + dk_stg_idx * dk_stage_elems
                                    + ov_slab
                                    + (ov_tok + m0 * 16) * 64
                                    + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                                ).raw_ptr(),
                                dk_f16,
                                nvvm.MMALayout.COL,
                            )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()
                if cend < SFIRST_MIN:
                    bars.mb_dk_total_ready[0].wait(cg1_dk_total_rdy.phase)
                    cg1_dk_total_rdy = advance(cg1_dk_total_rdy, 1)
                    dmr_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        dmr_vecs.append(
                            nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8)
                        )
                    nvvm.tcgen05_wait("load")
                    bars.mb_dk_total_done[0].arrive()
                    for sub in cutlass.range_constexpr(2):
                        for m0 in cutlass.range_constexpr(4):
                            dk_f16 = [fp32_to_fp16(dmr_vecs[sub][8 * m0 + 2 * j], dmr_vecs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                            nvvm.stmatrix(
                                (
                                    sdK_base
                                    + dk_stg_idx * dk_stage_elems
                                    + ov_slab
                                    + (ov_tok + m0 * 16) * 64
                                    + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                                ).raw_ptr(),
                                dk_f16,
                                nvvm.MMALayout.COL,
                            )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()

                # ---- dH prep, sdH restage half (the dK readout above fires
                # dk_total_done before this sweep) --------------------------
                if sk_nt >= 2:
                    # the sdH overwrite below waits only CG0's hdh read
                    bars.mb_hdh_done[0].wait(cg1_hdh_index.phase)
                    cg1_hdh_index = advance(cg1_hdh_index, 1)
                    bars.mb_dhs_done[0].wait(dhs_index.phase)
                    dhs_index = advance(dhs_index, 1)
                    for b in cutlass.range_constexpr(2):
                        for hh in cutlass.range_constexpr(2):
                            dhs_vec = nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(((tmem_warp_row + b * 16) << 16) + tmem_dh_col + hh * 64, cutlass.Float32),
                                num=8,
                            )
                            dhs_f16 = [fp32_to_fp16(dhs_vec[2 * j], dhs_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                            for c in cutlass.range_constexpr(4):
                                dhs_row = hh * 64 + ov_tok + c * 16
                                nvvm.stmatrix(
                                    cutlass.inttoptr(
                                        sdH_base_int + ((cg1_tidx // 64) * cfg.d_k * 64 + dhs_row * 64 + swizzle_xor_128b(dhs_row, ov_col + b * 16)) * 2,
                                        cutlass.AddressSpace.smem,
                                        cutlass.BFloat16,
                                    ),
                                    [dhs_f16[c * 4 + 0], dhs_f16[c * 4 + 1], dhs_f16[c * 4 + 2], dhs_f16[c * 4 + 3]],
                                    nvvm.MMALayout.COL,
                                )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_dhs_ready[0].arrive()

                if sk_nt < 2:
                    cg1_hdh_index = advance(cg1_hdh_index, 1)

        # ---- chunks NT-2 .. 0 (backward): full body ----------------------
        for rev_idx in cutlass.range(0 if cfg.use_dht else 1, sk_nt):
            chunk_idx = cend - 1 - rev_idx
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            beta_idx = cg1_beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(cg1_beta_index.phase)
            cg1_beta_index = advance(cg1_beta_index, cfg.smem_beta_stages)

            # ---- dH rescale: dH *= this chunk's cumprod ----
            cumprod_top = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]
            dhr_vecs = []
            for sub in cutlass.range_constexpr(num_state_subs):
                dhr_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32), num=32))
            for sub in cutlass.range_constexpr(num_state_subs):
                dhr_scaled = []
                for j in cutlass.range_constexpr(16):
                    h0, h1 = fmul2(dhr_vecs[sub][2 * j], dhr_vecs[sub][2 * j + 1], cumprod_top, cumprod_top)
                    dhr_scaled += [h0, h1]
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32),
                    cutlass.Vector.from_elements(tuple(dhr_scaled), cutlass.Float32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dh_acc_done[dh_done_idx].arrive()

            num_vals = 32
            gDecayScale = []
            last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
            for k in cutlass.range_constexpr(num_vals):
                gDecayScale.append(cute.math.exp2(last_cumsumlog - sCumsumlog[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx], fastmath=True))
            gCumprod = []
            for k in cutlass.range_constexpr(num_vals):
                gCumprod.append(sCumprod[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])

            # ---- dO' restage: dO * gCumprod * scale -> shared_inp TMEM ----
            bars.mb_dop_inp_done[0].wait(dop_inp_free.phase)
            dop_inp_free = advance(dop_inp_free, 1)
            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            do_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
            for c in cutlass.range_constexpr(8):
                m0 = cutlass.const_expr(c % 4)
                sub = cutlass.const_expr(c // 4)
                do_f16 = nvvm.ldmatrix(
                    (sdO_base + do_idx * do_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                    4,
                    nvvm.MMALayout.COL,
                )
                for i in cutlass.range_constexpr(4):
                    lo, hi = f16x2_to_f32(do_f16[i], dtype=cfg.io_dtype)
                    p0, p1 = fmul2(lo, hi, gCumprod[8 * m0 + 2 * i], gCumprod[8 * m0 + 2 * i + 1])
                    do_regs[8 * m0 + 2 * i][sub], do_regs[8 * m0 + 2 * i + 1][sub] = fmul2(p0, p1, scale, scale)
            for sub in cutlass.range_constexpr(2):
                do_pack = [fp32_to_fp16(do_regs[2 * j][sub], do_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dop_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(do_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dop_inp_ready[0].arrive()

            # ---- dV inter: in-place decay scale -------------------------
            bars.mb_du_scale_ready[0].wait(cg1_du_scale_rdy.phase)
            cg1_du_scale_rdy = advance(cg1_du_scale_rdy, 1)
            dv_ptrs = []
            dv_vecs = []
            for sub in cutlass.range_constexpr(2):
                dv_ptrs.append(nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32))
                dv_vecs.append(nvvm.tcgen05_ld("16x256b", dv_ptrs[sub], num=8))
            for sub in cutlass.range_constexpr(2):
                dv_scaled = []
                for j in cutlass.range_constexpr(16):
                    s0, s1 = fmul2(dv_vecs[sub][2 * j], dv_vecs[sub][2 * j + 1], gDecayScale[2 * j], gDecayScale[2 * j + 1])
                    dv_scaled += [s0, s1]
                nvvm.tcgen05_st("16x256b", dv_ptrs[sub], cutlass.Vector.from_elements(tuple(dv_scaled), cutlass.Float32))
            nvvm.tcgen05_wait("store")
            bars.mb_du_scale_done[0].arrive()

            # ---- v - k*state:  delta = V - cumprod*(K @ S), in place ----
            v_idx = v_index.idx
            bars.mb_v_ready[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if chunk_idx >= S_MIN:
                v_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
                gks_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
                for c in cutlass.range_constexpr(8):
                    m0 = cutlass.const_expr(c % 4)
                    sub = cutlass.const_expr(c // 4)
                    v_f16 = nvvm.ldmatrix(
                        (sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        lo, hi = f16x2_to_f32(v_f16[i], dtype=cfg.io_dtype)
                        v_regs[8 * m0 + 2 * i][sub] = lo
                        v_regs[8 * m0 + 2 * i + 1][sub] = hi
                bars.mb_ks_acc_ready[0].wait(cg1_ks_rdy.phase)
                cg1_ks_rdy = advance(cg1_ks_rdy, 1)
                for sub in cutlass.range_constexpr(2):
                    ks_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_ks_col, cutlass.Float32),
                        num=8,
                    )
                    for j in cutlass.range_constexpr(16):
                        g0, g1 = fmul2(ks_vec[2 * j], ks_vec[2 * j + 1], gCumprod[2 * j], gCumprod[2 * j + 1])
                        gks_regs[2 * j][sub] = g0
                        gks_regs[2 * j + 1][sub] = g1
                        v_regs[2 * j][sub] = v_regs[2 * j][sub] - g0
                        v_regs[2 * j + 1][sub] = v_regs[2 * j + 1][sub] - g1
                for sub in cutlass.range_constexpr(2):
                    y_pack = [fp32_to_fp16(v_regs[2 * j][sub], v_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(y_pack), cutlass.Int32),
                    )
                    gks_pack = [fp32_to_fp16(gks_regs[2 * j][sub], gks_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_gks_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(gks_pack), cutlass.Int32),
                    )
            if chunk_idx < S_MIN:
                for sub in cutlass.range_constexpr(2):
                    v_pack = []
                    for m0 in cutlass.range_constexpr(4):
                        v_f16 = nvvm.ldmatrix(
                            (
                                sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            v_lo, v_hi = f16x2_to_f32(v_f16[i], dtype=cfg.io_dtype)
                            v_pack.append(fp32_to_fp16(v_lo, v_hi, dtype=cfg.io_dtype))
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(v_pack), cutlass.Int32),
                    )
            nvvm.tcgen05_wait("store")
            bars.mb_y_ready[0].arrive()

            # ---- dU restage: dv acc ------------------------------------
            bars.mb_dyp_inp_done[0].wait(dyp_inp_free.phase)
            dyp_inp_free = advance(dyp_inp_free, 1)
            bars.mb_du_total_ready[0].wait(cg1_du_total_rdy.phase)
            cg1_du_total_rdy = advance(cg1_du_total_rdy, 1)
            for sub in cutlass.range_constexpr(2):
                du_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8)
                du_pack = [fp32_to_fp16(du_vec[2 * j], du_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_du_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(du_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_du_inp_ready[0].arrive()
            bars.mb_do_cg1_done[do_idx].arrive()

            # ---- U readout ---------------------------------------------
            bars.mb_u_acc_ready[0].wait(cg1_u_rdy.phase)
            cg1_u_rdy = advance(cg1_u_rdy, 1)
            u_regs = []
            for sub in cutlass.range_constexpr(2):
                u_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_u_col, cutlass.Float32), num=8)
                u_regs.append([u_vec[k] for k in range(32)])
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    u_f16 = [fp32_to_fp16(u_regs[sub][8 * m0 + 2 * j], u_regs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix(
                        (sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                        u_f16,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_ready[0].arrive()
            bars.mb_v_cg1_done[v_idx].arrive()

            # ---- dY ----------------------------------------------------
            bars.mb_dy_acc_ready[0].wait(cg1_dy_rdy.phase)
            cg1_dy_rdy = advance(cg1_dy_rdy, 1)
            dy_regs = []
            for sub in cutlass.range_constexpr(2):
                dy_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dy_col, cutlass.Float32),
                    num=8,
                )
                dy_regs.append([dy_vec[k] for k in range(32)])

            # ---- dY' = -gCumprod * dY -> f16 shared_inp ----------------
            neg_one = cutlass.Float32(-1.0)
            gCumprodNeg = [gCumprod[k] * neg_one for k in range(32)]
            for sub in cutlass.range_constexpr(2):
                dyp = []
                for j in cutlass.range_constexpr(16):
                    n0, n1 = fmul2(dy_regs[sub][2 * j], dy_regs[sub][2 * j + 1], gCumprodNeg[2 * j], gCumprodNeg[2 * j + 1])
                    dyp += [n0, n1]
                dyp_pack = [fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dyp_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(dyp_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dyp_inp_ready[0].arrive()

            # ---- dV staging: dV = dY -> the sdV slot -------------------
            dv_stg_idx = dv_index.idx
            bars.mb_dv_tmastg_done[dv_stg_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            bars.mb_sdv_done[0].wait(sdv_done_index.phase)
            sdv_done_index = advance(sdv_done_index, 1)
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    dv_f16 = [fp32_to_fp16(dy_regs[sub][8 * m0 + 2 * j], dy_regs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix(
                        (
                            sdV_base + dv_stg_idx * dv_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                        ).raw_ptr(),
                        dv_f16,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stg_idx].arrive()
            dk_stg_idx = dk_index.idx
            bars.mb_dk_tmastg_done[dk_stg_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dq_stg_idx = dq_index.idx
            bars.mb_dq_tmastg_done[dq_stg_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            sred = sred_base + dq_stg_idx * (dq_stage_elems // 2)

            # ---- dBeta/dGate v-terms: dbeta_t += rowsum(dV ⊙ Y)_t / beta_t ----
            part_y = [cutlass.Float32(0.0)] * 16
            for sub in cutlass.range_constexpr(2):
                y_pk = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32), num=8)
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(y_pk[j], dtype=cfg.io_dtype)
                    kk0 = cutlass.const_expr(2 * j)
                    j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                    if cutlass.const_expr(sub == 0 and j % 2 == 0):
                        part_y[j0], part_y[j0 + 1] = fmul2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi)
                    else:
                        part_y[j0], part_y[j0 + 1] = ffma2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi, part_y[j0], part_y[j0 + 1])
            py_lo, py_hi = _warp_reduce_scatter_frag16(part_y, lane_id)
            vt_tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
            if chunk_idx >= S_MIN:
                part_g = [cutlass.Float32(0.0)] * 16
                for sub in cutlass.range_constexpr(2):
                    g_pk = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_gks_col, cutlass.Int32), num=8)
                    for j in cutlass.range_constexpr(16):
                        lo, hi = f16x2_to_f32(g_pk[j], dtype=cfg.io_dtype)
                        kk0 = cutlass.const_expr(2 * j)
                        j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                        if cutlass.const_expr(sub == 0 and j % 2 == 0):
                            part_g[j0], part_g[j0 + 1] = fmul2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi)
                        else:
                            part_g[j0], part_g[j0 + 1] = ffma2(dy_regs[sub][kk0], dy_regs[sub][kk0 + 1], lo, hi, part_g[j0], part_g[j0 + 1])
                pg_lo, pg_hi = _warp_reduce_scatter_frag16(part_g, lane_id)
                (sred + cg1w * 64 + vt_tok0).store(py_lo)
                (sred + cg1w * 64 + vt_tok0 + 1).store(py_hi)
                (sred + 256 + cg1w * 64 + vt_tok0).store(pg_lo)
                (sred + 256 + cg1w * 64 + vt_tok0 + 1).store(pg_hi)
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                    ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                    gsum = (sred + 256 + cg1_tidx).load() + (sred + 320 + cg1_tidx).load() + (sred + 384 + cg1_tidx).load() + (sred + 448 + cg1_tidx).load()
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                    sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0) - gsum
            if chunk_idx < S_MIN:
                (sred + cg1w * 64 + vt_tok0).store(py_lo)
                (sred + cg1w * 64 + vt_tok0 + 1).store(py_hi)
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                    ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                    sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0)
            bars.mb_beta_done[beta_idx].arrive()
            bars.mb_dbeta_cg1_ready[0].arrive()
            # sred reads must land before the dq stmatrix below reuses the stage
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
            # ---- Q fragments held in registers over the dq stage (sQ
            # free once read; the dQ dot consumes them -- no TMEM trip) ----
            q_frag = []
            for sub in cutlass.range_constexpr(2):
                q_words = []
                for m0 in cutlass.range_constexpr(4):
                    q_f16 = nvvm.ldmatrix(
                        (sQ_base + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        q_lo, q_hi = f16x2_to_f32(q_f16[i], dtype=cfg.io_dtype)
                        q_words.append(fp32_to_fp16(q_lo, q_hi, dtype=cfg.io_dtype))
                q_frag.append(q_words)
                # the store+wait order the sQ LDSM reads before the release arrive
                # (a bare register consume can be hoisted past by ptxas); the dot
                # below reads the registers, so the TMEM read-back is still gone
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(q_words), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_q_cg1_done[0].arrive()

            # ---- dq final read -> sdQ (output staging; the fragments are
            # held for the dQ dot below) ---------
            bars.mb_dq_acc_total_ready[0].wait(dq_total_rdy_index.phase)
            dq_total_rdy_index = advance(dq_total_rdy_index, 1)
            dq_frag = []
            for sub in cutlass.range_constexpr(2):
                dqv = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dh_inp_col, cutlass.Float32), num=8)
                dq_frag.append([dqv[k] for k in range(32)])
                for m0 in cutlass.range_constexpr(4):
                    frag_addr = ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                    dq_f16 = [fp32_to_fp16(dqv[8 * m0 + 2 * j], dqv[8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix((sdQ_base + dq_stg_idx * dq_stage_elems + frag_addr).raw_ptr(), dq_f16, nvvm.MMALayout.COL)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_acc_total_done[0].arrive()
            bars.mb_dq_tmastg_ready[dq_stg_idx].arrive()

            # ---- dQ dot (part_q): fragment dot of the held dq acc with the
            # staged Q^T, added to dGate FIRST; CG0 adds after parts_ready ----
            part_q = [cutlass.Float32(0.0)] * 16
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        q_lo, q_hi = f16x2_to_f32(q_frag[sub][4 * m0 + i], dtype=cfg.io_dtype)
                        kk0 = cutlass.const_expr(8 * m0 + 2 * i)
                        j0 = cutlass.const_expr((kk0 // 4) * 2 + (kk0 % 2))
                        if cutlass.const_expr(sub == 0 and i % 2 == 0):
                            part_q[j0], part_q[j0 + 1] = fmul2(dq_frag[sub][kk0], dq_frag[sub][kk0 + 1], q_lo, q_hi)
                        else:
                            part_q[j0], part_q[j0 + 1] = ffma2(dq_frag[sub][kk0], dq_frag[sub][kk0 + 1], q_lo, q_hi, part_q[j0], part_q[j0 + 1])
            amq_lo, amq_hi = _warp_reduce_scatter_frag16(part_q, lane_id)
            tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
            (sdh_red + (cg1_tidx // 32) * 64 + tok0).store(amq_lo)
            (sdh_red + (cg1_tidx // 32) * 64 + tok0 + 1).store(amq_hi)
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
            if cg1_tidx < 64:
                pq_sum = (sdh_red + cg1_tidx).load() + (sdh_red + 64 + cg1_tidx).load() + (sdh_red + 128 + cg1_tidx).load() + (sdh_red + 192 + cg1_tidx).load()
                sCumsumlog[cg1_tidx, 0, gate_idx] = sCumsumlog[cg1_tidx, 0, gate_idx] + pq_sum
            bars.mb_gate_done[gate_idx].arrive()
            bars.mb_dgate_cg1_ready[0].arrive()

            # ---- NEXT-CHUNK dH prep ------------------------------------
            if chunk_idx >= wstart + 1:
                dh_idx = dh_acc_index.idx
                bars.mb_dh_acc_ready[dh_idx].wait(dh_acc_index.phase)
                dh_acc_index = advance(dh_acc_index, cfg.tmem_dh_acc_stages)
                dh_done_idx = dh_idx
                dh_inp_idx = dh_inp_index.idx
                bars.mb_dh_inp_done[dh_inp_idx].wait(dh_inp_index.phase)
                dh_inp_index = advance(dh_inp_index, cfg.tmem_dh_inp_stages)
                dh_regs = [[cutlass.Float32(0.0) for _ in range(num_state_subs)] for _ in range(32)]
                for sub in cutlass.range_constexpr(num_state_subs):
                    dh_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32), num=32)
                    for k in cutlass.range_constexpr(32):
                        dh_regs[k][sub] = dh_vec[k]

                    dh_f16 = [fp32_to_fp16(dh_regs[2 * j][sub], dh_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_inp_col + sub * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dh_f16), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dh_inp_ready[dh_inp_idx].arrive()

            # ---- dK s-path fold: read while the dM-terms GEMMs run ------------
            if chunk_idx >= S_MIN:
                bars.mb_dk_spath_ready[0].wait(cg1_dk_spath_rdy.phase)
                cg1_dk_spath_rdy = advance(cg1_dk_spath_rdy, 1)
                dk_spath_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dk_spath_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dk_spath_col, cutlass.Float32),
                            num=8,
                        )
                    )
                nvvm.tcgen05_wait("load")
                dk_spath_regs = []
                for sub in cutlass.range_constexpr(2):
                    dk_spath_row = []
                    for j in cutlass.range_constexpr(16):
                        n0, n1 = fmul2(dk_spath_vecs[sub][2 * j], dk_spath_vecs[sub][2 * j + 1], gCumprodNeg[2 * j], gCumprodNeg[2 * j + 1])
                        dk_spath_row += [n0, n1]
                    dk_spath_regs.append(dk_spath_row)

                bars.mb_dk_total_ready[0].wait(cg1_dk_total_rdy.phase)
                cg1_dk_total_rdy = advance(cg1_dk_total_rdy, 1)
                dmr_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dmr_vecs.append(nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8))
                nvvm.tcgen05_wait("load")
                bars.mb_dk_total_done[0].arrive()
                for sub in cutlass.range_constexpr(2):
                    dk_sum = [dmr_vecs[sub][k] + dk_spath_regs[sub][k] for k in range(32)]
                    for m0 in cutlass.range_constexpr(4):
                        dk_f16 = [fp32_to_fp16(dk_sum[8 * m0 + 2 * j], dk_sum[8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elems
                                + ov_slab
                                + (ov_tok + m0 * 16) * 64
                                + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            dk_f16,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()
            if chunk_idx < S_MIN:
                bars.mb_dk_total_ready[0].wait(cg1_dk_total_rdy.phase)
                cg1_dk_total_rdy = advance(cg1_dk_total_rdy, 1)
                dmr_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dmr_vecs.append(nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_col, cutlass.Float32), num=8))
                nvvm.tcgen05_wait("load")
                bars.mb_dk_total_done[0].arrive()
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        dk_f16 = [fp32_to_fp16(dmr_vecs[sub][8 * m0 + 2 * j], dmr_vecs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elems
                                + ov_slab
                                + (ov_tok + m0 * 16) * 64
                                + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                            ).raw_ptr(),
                            dk_f16,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()

            # ---- dH prep, sdH restage half (the dK readout above fires
            # dk_total_done before this sweep) --------------------------
            if chunk_idx >= wstart + 1:
                # the sdH overwrite below waits only CG0's hdh read
                bars.mb_hdh_done[0].wait(cg1_hdh_index.phase)
                cg1_hdh_index = advance(cg1_hdh_index, 1)
                bars.mb_dhs_done[0].wait(dhs_index.phase)
                dhs_index = advance(dhs_index, 1)
                dhs_vecs = []
                for b in cutlass.range_constexpr(2):
                    for hh in cutlass.range_constexpr(2):
                        dhs_vecs.append(
                            nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(((tmem_warp_row + b * 16) << 16) + tmem_dh_col + hh * 64, cutlass.Float32),
                                num=8,
                            )
                        )
                for b in cutlass.range_constexpr(2):
                    for hh in cutlass.range_constexpr(2):
                        dhs_vec = dhs_vecs[b * 2 + hh]
                        dhs_f16 = [fp32_to_fp16(dhs_vec[2 * j], dhs_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                        for c in cutlass.range_constexpr(4):
                            dhs_row = hh * 64 + ov_tok + c * 16
                            nvvm.stmatrix(
                                cutlass.inttoptr(
                                    sdH_base_int + ((cg1_tidx // 64) * cfg.d_k * 64 + dhs_row * 64 + swizzle_xor_128b(dhs_row, ov_col + b * 16)) * 2,
                                    cutlass.AddressSpace.smem,
                                    cutlass.BFloat16,
                                ),
                                [dhs_f16[c * 4 + 0], dhs_f16[c * 4 + 1], dhs_f16[c * 4 + 2], dhs_f16[c * 4 + 3]],
                                nvvm.MMALayout.COL,
                            )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dhs_ready[0].arrive()

            if chunk_idx < wstart + 1:
                cg1_hdh_index = advance(cg1_hdh_index, 1)

        # ---- dH drain: with an initial state this is dL/dS0 ----
        if sk_nt > 0:
            dh_idx = dh_acc_index.idx
            bars.mb_dh_acc_ready[dh_idx].wait(dh_acc_index.phase)
            dh_acc_index = advance(dh_acc_index, cfg.tmem_dh_acc_stages)
            if cutlass.const_expr(cfg.use_initial_state):
                # split-K: only the item owning chunk 0 drains dL/dS0
                if cutlass.const_expr(cfg.split_k):
                    if wstart == 0:
                        gDs0 = mDs0_out[None, None, head_idx, batch_idx]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            ds0_vec = nvvm.tcgen05_ld(
                                "32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32), num=32
                            )
                            for kk in cutlass.range_constexpr(32):
                                gDs0[sub * ldtm_width + kk, cg1_tidx] = ds0_vec[kk]
                else:
                    gDs0 = mDs0_out[None, None, head_idx, batch_idx]
                    for sub in cutlass.range_constexpr(num_state_subs):
                        ds0_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dh_col + sub * ldtm_width, cutlass.Float32), num=32)
                        for kk in cutlass.range_constexpr(32):
                            gDs0[sub * ldtm_width + kk, cg1_tidx] = ds0_vec[kk]
            if cutlass.const_expr(not cfg.use_dht):
                bars.mb_dh_acc_done[dh_idx].arrive()
        else:
            # zero-length sequence: the gradient passes straight through
            # (dS0 = dHt when given, zeros otherwise); pure GMEM, no TMEM
            if cutlass.const_expr(cfg.use_initial_state):
                write_passthrough = True
                if cutlass.const_expr(cfg.split_k):
                    write_passthrough = wstart == 0
                if write_passthrough:
                    gDs0 = mDs0_out[None, None, head_idx, batch_idx]
                    if cutlass.const_expr(cfg.use_dht):
                        gDht = mDht[None, None, head_idx, batch_idx]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for kk in cutlass.range_constexpr(32):
                                gDs0[sub * ldtm_width + kk, cg1_tidx] = gDht[sub * ldtm_width + kk, cg1_tidx]
                    else:
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for kk in cutlass.range_constexpr(32):
                                gDs0[sub * ldtm_width + kk, cg1_tidx] = cutlass.Float32(0.0)

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    # CG1 done with TMEM: release the MMA warp's dealloc
    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.tmem_dh_inp_stages):
        bars.mb_dh_inp_done[dh_inp_index.idx].wait(dh_inp_index.phase)
        dh_inp_index = advance(dh_inp_index, cfg.tmem_dh_inp_stages)
    for _ in range(cfg.smem_dk_stages):
        bars.mb_dk_tmastg_done[dk_index.idx].wait(dk_index.phase)
        dk_index = advance(dk_index, cfg.smem_dk_stages)
    for _ in range(cfg.smem_dv_stages):
        bars.mb_dv_tmastg_done[dv_index.idx].wait(dv_index.phase)
        dv_index = advance(dv_index, cfg.smem_dv_stages)


@cute.jit
def _build_descs(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do_: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    h: cute.Tensor,
    cu_seqlens: cute.Tensor,
    s0: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """Build the per-(b,h) TMA-descriptor arrays (Q, K, V, dO, H loads;
    dQ, dK, dV stores; the io-dtype S0 loads when ``s0`` is given) into
    ``tensormap_workspace``. Launched on every execute: the descriptors fold cu_seqlens contents into
    GLOBAL_ADDRESS and GLOBAL_DIM, which the host cannot read without a D2H sync.

    The H descriptor is 3-D ``(dv, dk, h)`` over the packed
    ``[total_h, HO, DK, DV]`` H tensor; ``build_h_descs_kernel`` derives the
    per-sequence H offset from the token ``cu_seqlens`` ((seqlen-1)//B_T,
    prefix-summed), folds it and the head into GLOBAL_ADDRESS, and caps
    GLOBAL_DIM[2] to the per-sequence H count, so the load coordinate is the
    sequence-local H index.  The S0 descriptors share the H descriptor
    format (one ``[DK, DV]`` entry per slot over the dense
    ``[N, HO, DK, DV]`` buffer), so the load path is interchangeable."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    q_group = heads_out // h_q
    k_group = heads_out // h_k
    v_group = heads_out // h_v
    d_v = v.shape[2]
    d_k_h = h.shape[2]
    d_v_h = h.shape[3]
    bpe = io_dtype.width // 8
    granu = 128 // bpe
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]
    do_row_stride, do_head_stride = do_.stride[0], do_.stride[1]
    dq_row_stride, dq_head_stride = dq.stride[0], dq.stride[1]
    dk_row_stride, dk_head_stride = dk.stride[0], dk.stride[1]
    dv_row_stride, dv_head_stride = dv.stride[0], dv.stride[1]

    q_head0 = q[None, 0, None]
    k_head0 = k[None, 0, None]

    def _trans_head0(t, d):
        view = t[None, 0, None]
        return cute.make_tensor(
            view.iterator,
            cute.make_layout((d, view.shape[0]), stride=(view.stride[1], view.stride[0])),
        )

    v_head0 = _trans_head0(v, d_v)
    do_head0 = _trans_head0(do_, d_v)
    dq_head0 = _trans_head0(dq, dq.shape[2])
    dk_head0 = _trans_head0(dk, dk.shape[2])
    dv_head0 = _trans_head0(dv, d_v)
    swz128 = _tma.TensorMapSwizzle.s128b
    base_desc_q = _tma.create_tensor_map_tiled_from_view(q_head0, box_dims=(bt, granu), stride_order=(1, 0), swizzle=swz128)
    base_desc_k = _tma.create_tensor_map_tiled_from_view(k_head0, box_dims=(bt, granu), stride_order=(1, 0), swizzle=swz128)
    base_desc_v = _tma.create_tensor_map_tiled_from_view(v_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
    base_desc_do = _tma.create_tensor_map_tiled_from_view(do_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
    base_desc_dq = _tma.create_tensor_map_tiled_from_view(dq_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
    base_desc_dk = _tma.create_tensor_map_tiled_from_view(dk_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
    base_desc_dv = _tma.create_tensor_map_tiled_from_view(dv_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
    h_view = cute.make_tensor(
        h.iterator,
        cute.make_layout(
            (d_v_h, d_k_h, h.shape[0]),
            stride=(h.stride[3], h.stride[2], h.stride[0]),
        ),
    )
    base_desc_h = _tma.create_tensor_map_tiled_from_view(h_view, box_dims=(64, d_k_h, 1), stride_order=(0, 1, 2), swizzle=swz128)

    arr_words = (batch_size * heads_out) * TENSOR_MAP_QWORDS
    ws_iter = tensormap_workspace.iterator

    def sub_array(i):
        return cute.make_tensor(ws_iter + i * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    build_qkv_load_descs_kernel(
        base_desc_q,
        sub_array(0),
        cu_seqlens,
        q,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(q_group),
        cutlass.Int32(q_head_stride),
        cutlass.Int32(q_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_k,
        sub_array(1),
        cu_seqlens,
        k,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(k_group),
        cutlass.Int32(k_head_stride),
        cutlass.Int32(k_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_v,
        sub_array(2),
        cu_seqlens,
        v,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(v_group),
        cutlass.Int32(v_head_stride),
        cutlass.Int32(v_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_do,
        sub_array(3),
        cu_seqlens,
        do_,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(1),
        cutlass.Int32(do_head_stride),
        cutlass.Int32(do_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_h_descs_kernel(
        base_desc_h,
        sub_array(4),
        cu_seqlens,
        h,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(h.stride[1]),
        cutlass.Int32(h.stride[0]),
        cutlass.Int32(b_t),
        2,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_dq,
        sub_array(5),
        cu_seqlens,
        dq,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(1),
        cutlass.Int32(dq_head_stride),
        cutlass.Int32(dq_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_dk,
        sub_array(6),
        cu_seqlens,
        dk,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(1),
        cutlass.Int32(dk_head_stride),
        cutlass.Int32(dk_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_desc_dv,
        sub_array(7),
        cu_seqlens,
        dv,
        cutlass.Int32(batch_size),
        cutlass.Int32(heads_out),
        cutlass.Int32(1),
        cutlass.Int32(dv_head_stride),
        cutlass.Int32(dv_row_stride),
        1,
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    if cutlass.const_expr(s0 is not None):
        s0_view = cute.make_tensor(
            s0.iterator,
            cute.make_layout(
                (d_v_h, d_k_h, 1),
                stride=(s0.stride[3], s0.stride[2], s0.stride[1]),
            ),
        )
        base_desc_s0 = _tma.create_tensor_map_tiled_from_view(s0_view, box_dims=(64, d_k_h, 1), stride_order=(0, 1, 2), swizzle=swz128)
        build_state_descs_kernel(
            base_desc_s0,
            sub_array(8),
            s0,
            cutlass.Int32(batch_size),
            cutlass.Int32(heads_out),
            cutlass.Int32(s0.stride[1]),
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    do_: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    cu_seqlens: cute.Tensor,
    ds0: Optional[cute.Tensor],
    dht: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    sched_ctr: Optional[cute.Tensor],
    scale: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    h_q = q.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v

    # ---- GQA reshapes: fold the head group into a ----------------------
    if cutlass.const_expr(cfg.is_GQA):
        h_r = h_q // h_v
        h_qv = h_v
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[2], (h_r, h_v)),
                stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
            ),
        )
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (k.shape[0], k.shape[2], (h_r, h_v)),
                stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
            ),
        )
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[2], v.shape[0], (h_r, h_v)),
                stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
            ),
        )
    else:
        h_r = h_v // h_q
        h_qv = h_q
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[2], (h_r, h_q)),
                stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
            ),
        )
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (k.shape[0], k.shape[2], (h_r, h_q)),
                stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
            ),
        )
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[2], v.shape[0], (h_r, h_q)),
                stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
            ),
        )

    gate = cute.make_tensor(
        gate.iterator,
        cute.make_layout(
            (gate.shape[0], (h_r, h_qv)),
            stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1])),
        ),
    )
    beta = cute.make_tensor(
        beta.iterator,
        cute.make_layout(
            (beta.shape[0], (h_r, h_qv)),
            stride=(beta.stride[0], (beta.stride[1], h_r * beta.stride[1])),
        ),
    )
    dg = cute.make_tensor(
        dg.iterator,
        cute.make_layout(
            (dg.shape[0], (h_r, h_qv)),
            stride=(dg.stride[0], (dg.stride[1], h_r * dg.stride[1])),
        ),
    )
    dbeta = cute.make_tensor(
        dbeta.iterator,
        cute.make_layout(
            (dbeta.shape[0], (h_r, h_qv)),
            stride=(dbeta.stride[0], (dbeta.stride[1], h_r * dbeta.stride[1])),
        ),
    )
    if cutlass.const_expr(ds0 is not None):
        ds0 = cute.make_tensor(
            ds0.iterator,
            cute.make_layout(
                (ds0.shape[2], ds0.shape[3], (h_r, h_qv), ds0.shape[0]),
                stride=(
                    ds0.stride[2],
                    ds0.stride[3],
                    (ds0.stride[1], h_r * ds0.stride[1]),
                    ds0.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(dht is not None):
        dht = cute.make_tensor(
            dht.iterator,
            cute.make_layout(
                (dht.shape[2], dht.shape[3], (h_r, h_qv), dht.shape[0]),
                stride=(
                    dht.stride[2],
                    dht.stride[3],
                    (dht.stride[1], h_r * dht.stride[1]),
                    dht.stride[0],
                ),
            ),
        )

    # ---- SMEM sizing: per-buffer element cosizes -----------------------
    bpe = cfg.io_dtype.width // 8
    q_tile_elems = cfg.b_t * cfg.d_k
    k_tile_elems = cfg.b_t * cfg.d_k
    v_tile_elems = cfg.d_v * cfg.b_t
    do_tile_elems = cfg.d_v * cfg.b_t
    s_tile_elems = cfg.d_k * cfg.d_v
    ainv_tile_elems = cfg.b_t * cfg.b_t
    qk_tile_elems = cfg.b_t * cfg.b_t
    dq_tile_elems = cfg.b_t * cfg.d_k
    dk_tile_elems = cfg.b_t * cfg.d_k
    dv_tile_elems = cfg.d_v * cfg.b_t
    cfg.q_cosize = q_tile_elems * cfg.smem_q_stages
    cfg.k_cosize = k_tile_elems * cfg.smem_k_stages
    cfg.v_cosize = v_tile_elems * cfg.smem_v_stages
    cfg.do_cosize = do_tile_elems * cfg.smem_do_stages
    cfg.s_cosize = s_tile_elems * cfg.smem_s_stages
    cfg.ainv_cosize = ainv_tile_elems * cfg.smem_ainv_stages
    cfg.qk_cosize = qk_tile_elems * cfg.smem_qk_stages
    cfg.dq_cosize = dq_tile_elems * cfg.smem_dq_stages
    cfg.dk_cosize = dk_tile_elems * cfg.smem_dk_stages
    cfg.dv_cosize = dv_tile_elems * cfg.smem_dv_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))

    cfg.tma_q_bytes = q_tile_elems * bpe
    cfg.tma_k_bytes = k_tile_elems * bpe
    cfg.tma_v_bytes = v_tile_elems * bpe
    cfg.tma_do_bytes = do_tile_elems * bpe
    cfg.tma_s_bytes = s_tile_elems * bpe

    cfg.n_heads_out = heads_out
    num_descs = batch_size * heads_out

    # ---- launch --------------------------------------------------------
    total_tiles = batch_size * heads_out
    # CUDA-graph-stable launch: fixed SM-count grid
    grid_shape = (cfg.max_active_clusters, 1, 1)

    _kernel(
        cfg,
        gate,
        beta,
        dg,
        dbeta,
        cu_seqlens,
        scale,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        total_tiles,
        q,
        k,
        v,
        do_,
        dq,
        dk,
        dv,
        ds0,
        dht,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        cutlass.Int32(num_descs),
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        cluster=cfg.cluster_shape_mnk,
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.kernel
def _kernel(
    cfg: cutlass.Constexpr,
    mGate: cute.Tensor,
    mBeta: cute.Tensor,
    mDg: cute.Tensor,
    mDbeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    total_tiles: cutlass.Int32,
    mQ,
    mK,
    mV,
    mdO,
    mdQ,
    mdK,
    mdV,
    mDs0,
    mDht,
    mWorkItems: Optional[cute.Tensor],
    mCount: Optional[cute.Tensor],
    mSched: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """
    Main GDN bprop chunked kernel.

    Warp specialization is the outermost control flow: each warp role owns
    its own persistent tile-scheduler loop, iterating over (batch, head)
    tiles and then over chunks within each tile in BACKWARD order.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    if cutlass.const_expr(cfg.split_k):
        total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None, "mSched must be provided if dyn_sched is True"

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_s_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(7) * arr_words
    desc_s0_base = desc_base_words + cutlass.Int32(8) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_gdn_bars(cfg)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=cutlass.AddressSpace.smem, alignment=16)
    tmem_hold = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=128)

    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = (cfg.d_v // 2) * 128
    V_LEAD = (cfg.d_v // 2) * 128
    S_LEAD = cfg.d_k * 128
    sQ_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.q_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sQ = SmemTile(
        base=sQ_raw.data_ptr().toint(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages) * bpe,
        stages=cfg.smem_q_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.k_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sK = SmemTile(
        base=sK_raw.data_ptr().toint(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages) * bpe,
        stages=cfg.smem_k_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_trans = SmemTile(
        base=sK_raw.data_ptr().toint(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages) * bpe,
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQ_trans = SmemTile(
        base=sQ_raw.data_ptr().toint(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages) * bpe,
        stages=cfg.smem_q_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.do_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdO = SmemTile(
        base=sdO_raw.data_ptr().toint(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages) * bpe,
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO_kmaj = SmemTile(
        base=sdO_raw.data_ptr().toint(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages) * bpe,
        stages=cfg.smem_do_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sS_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.s_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sS = SmemTile(
        base=sS_raw.data_ptr().toint(),
        elems_per_stage=(cfg.s_cosize // cfg.smem_s_stages) * bpe,
        stages=cfg.smem_s_stages,
        leading_byte_offset=S_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sS_kmaj = SmemTile(
        base=sS_raw.data_ptr().toint(),
        elems_per_stage=(cfg.s_cosize // cfg.smem_s_stages) * bpe,
        stages=cfg.smem_s_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sAinv_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.ainv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sAinv = SmemTile(
        base=sAinv_raw.data_ptr().toint(),
        elems_per_stage=(cfg.ainv_cosize // cfg.smem_ainv_stages) * bpe,
        stages=cfg.smem_ainv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sAinv_trans = SmemTile(
        base=sAinv_raw.data_ptr().toint(),
        elems_per_stage=(cfg.ainv_cosize // cfg.smem_ainv_stages) * bpe,
        stages=cfg.smem_ainv_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKK_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.ainv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sKK = SmemTile(
        base=sKK_raw.data_ptr().toint(),
        elems_per_stage=(cfg.ainv_cosize // cfg.smem_ainv_stages) * bpe,
        stages=cfg.smem_ainv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQk_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.qk_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sQk = SmemTile(
        base=sQk_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=cfg.smem_qk_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQk_trans = SmemTile(
        base=sQk_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=cfg.smem_qk_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa = SmemTile(
        base=sQk_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa_trans = SmemTile(
        base=sQk_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.qk_cosize // cfg.smem_qk_stages,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sDm = SmemTile(
        base=sDm_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_trans = SmemTile(
        base=sDm_raw.data_ptr().toint(),
        elems_per_stage=(cfg.qk_cosize // cfg.smem_qk_stages) * bpe,
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    # sub-bank split: V + sdH + dQ + dK + dV (the LDSM/LDS/STS/STSM-heavy
    # set) allocated last -> all past the 128KB sub-bank boundary
    sV_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.v_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sV = SmemTile(
        base=sV_raw.data_ptr().toint(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages) * bpe,
        stages=cfg.smem_v_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV_kmaj = SmemTile(
        base=sV_raw.data_ptr().toint(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages) * bpe,
        stages=cfg.smem_v_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdH_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.d_k * cfg.d_v,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdH = SmemTile(
        base=sdH_raw.data_ptr().toint(),
        elems_per_stage=cfg.d_k * cfg.d_v * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdQ_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.dq_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdQ = SmemTile(
        base=sdQ_raw.data_ptr().toint(),
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages) * bpe,
        stages=cfg.smem_dq_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdK_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.dk_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdK = SmemTile(
        base=sdK_raw.data_ptr().toint(),
        elems_per_stage=(cfg.dk_cosize // cfg.smem_dk_stages) * bpe,
        stages=cfg.smem_dk_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdV_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.dv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdV = SmemTile(
        base=sdV_raw.data_ptr().toint(),
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages) * bpe,
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdV_kmaj = SmemTile(
        base=sdV_raw.data_ptr().toint(),
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages) * bpe,
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdh_flat = cute.make_tensor(
        cute.make_ptr(cfg.io_dtype, sdH_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes),
        cute.make_layout(cfg.d_k * cfg.d_v),
    )
    ss_flat = cute.make_tensor(
        cute.make_ptr(cfg.io_dtype, sS_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes),
        cute.make_layout(cfg.s_cosize),
    )
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumsumlog_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sCumprod = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumprod_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sBeta = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, beta_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        beta_smem_layout_staged,
    )

    # ------------------------------------------------------------------
    # mbarrier init (all threads)
    # ------------------------------------------------------------------
    for s_ in range(cfg.sched_stages):
        bars.mb_sched_ready[s_].init()
        bars.mb_sched_done[s_].init()
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_mma_done[s].init()
        bars.mb_q_cg1_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_mma_done[s].init()
        bars.mb_k_cg0_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_mma_done[s].init()
        bars.mb_v_cg1_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
        bars.mb_do_cg1_done[s].init()
    for s in range(cfg.smem_s_stages):
        bars.mb_s_ready[s].init()
        bars.mb_s_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.smem_beta_stages):
        bars.mb_beta_ready[s].init()
        bars.mb_beta_done[s].init()
    for s in range(cfg.tmem_dh_acc_stages):
        bars.mb_dh_acc_ready[s].init()
        bars.mb_dh_acc_done[s].init()
    for b in (
        bars.mb_du_scale_ready,
        bars.mb_du_scale_done,
        bars.mb_du_total_ready,
        bars.mb_dk_scale_ready,
        bars.mb_dk_scale_done,
        bars.mb_dk_attn_ready,
        bars.mb_dk_attn_done,
        bars.mb_dk_total_ready,
        bars.mb_dk_total_done,
    ):
        b[0].init()
    for b in (
        bars.mb_kk_acc_ready,
        bars.mb_kk_acc_done,
        bars.mb_a_acc_ready,
        bars.mb_ks_acc_ready,
        bars.mb_u_acc_ready,
        bars.mb_dy_acc_ready,
    ):
        b[0].init()
    for s in range(cfg.smem_ainv_stages):
        bars.mb_ainv_ready[s].init()
        bars.mb_ainv_done[s].init()
    for s in range(cfg.smem_qk_stages):
        bars.mb_qk_ready[s].init()
        bars.mb_qk_done[s].init()
    for s in range(cfg.tmem_dh_inp_stages):
        bars.mb_dh_inp_ready[s].init()
        bars.mb_dh_inp_done[s].init()
    for b in (
        bars.mb_dop_inp_ready,
        bars.mb_dop_inp_done,
        bars.mb_du_inp_ready,
        bars.mb_dyp_inp_ready,
        bars.mb_dyp_inp_done,
    ):
        b[0].init()
    for s in range(cfg.smem_dq_stages):
        bars.mb_dq_tmastg_ready[s].init()
        bars.mb_dq_tmastg_done[s].init()
    for s in range(cfg.smem_dk_stages):
        bars.mb_dk_tmastg_ready[s].init()
        bars.mb_dk_tmastg_done[s].init()
    for s in range(cfg.smem_dv_stages):
        bars.mb_dv_tmastg_ready[s].init()
        bars.mb_dv_tmastg_done[s].init()
    bars.mb_y_ready[0].init()
    bars.mb_sdv_done[0].init()
    bars.mb_u_ready[0].init()
    bars.mb_dhs_ready[0].init()
    bars.mb_dhs_done[0].init()
    bars.mb_da_ready[0].init()
    bars.mb_dq_acc_scale_ready[0].init()
    bars.mb_dq_acc_scale_done[0].init()
    bars.mb_dq_acc_total_ready[0].init()
    bars.mb_dq_acc_total_done[0].init()
    bars.mb_da_acc_ready[0].init()
    bars.mb_dm_ready[0].init()
    bars.mb_dm_done[0].init()
    bars.mb_dbeta_cg1_ready[0].init()
    bars.mb_dgate_cg1_ready[0].init()
    bars.mb_hdh_done[0].init()
    bars.mb_dk_spath_ready[0].init()
    bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ------------------------------------------------------------------
    # 2. Warp specialization - each warp role owns its own scheduler loop
    # ------------------------------------------------------------------

    if warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        _compute0_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            tmem_hold=tmem_hold,
            scale=scale,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sAinv=sAinv,
            sKK=sKK,
            sQk=sQk,
            sDa=sDa,
            sDm=sDm,
            sK=sK,
            sdQ=sdQ,
            sdH=sdH,
            ss_flat=ss_flat,
            sdh_flat=sdh_flat,
            sSched=sSched,
            bars=bars,
        )

    if warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        _compute1_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mDs0,
            mDht,
            tidx,
            warp_idx=warp_idx,
            tmem_hold=tmem_hold,
            scale=scale,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sdO=sdO,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sdQ=sdQ,
            sdK=sdK,
            sdV=sdV,
            sdH=sdH,
            sDa=sDa,
            sDm=sDm,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.mma_warp_id:
        _mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tmem_hold=tmem_hold,
            sQ=sQ,
            sQ_trans=sQ_trans,
            sK=sK,
            sK_trans=sK_trans,
            sV=sV,
            sV_kmaj=sV_kmaj,
            sdO=sdO,
            sdO_kmaj=sdO_kmaj,
            sS=sS,
            sS_kmaj=sS_kmaj,
            sAinv=sAinv,
            sAinv_trans=sAinv_trans,
            sQk=sQk,
            sQk_trans=sQk_trans,
            sDa=sDa,
            sDa_trans=sDa_trans,
            sdH=sdH,
            sDm=sDm,
            sDm_trans=sDm_trans,
            sdV=sdV,
            sdV_kmaj=sdV_kmaj,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.tma_qkv_warp_id:
        _tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mSched=mSched,
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            sdO_raw=sdO_raw,
            sS_raw=sS_raw,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            desc_do_base=desc_do_base,
            desc_s_base=desc_s_base,
            desc_s0_base=desc_s0_base,
            sSched=sSched,
            bars=bars,
        )

    if warp_idx == cfg.load_gate_beta_warp_id:
        _gate_beta_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            mGate=mGate,
            mBeta=mBeta,
            mDg=mDg,
            mDbeta=mDbeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sSched=sSched,
            bars=bars,
        )
    if warp_idx == cfg.epilogue_warp_id:
        _tmastg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sdQ_raw=sdQ_raw,
            sdK_raw=sdK_raw,
            sdV_raw=sdV_raw,
            desc_dq_base=desc_dq_base,
            desc_dk_base=desc_dk_base,
            desc_dv_base=desc_dv_base,
            sSched=sSched,
            bars=bars,
        )


@dataclass
class GdnBwdCfg:
    """Per-compile GDN bprop kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA) are the ``cute.compile`` cache
    keys; the rest is derived from the module-global ``CFG`` constants.
    ``_host`` stamps the shape-derived fields at trace time.
    """

    use_initial_state: bool
    use_dht: bool
    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    is_GQA: bool
    # split-K: tiles come from a work-item table (see common/split_k.py);
    # each item computes chunks [wstart, cend) backward, writes [wstart, wend)
    split_k: bool = False
    # gate input domain: natural-log decay instead of raw
    # linear alpha; the gate warp then skips its log2 and rescales by 1/ln2
    log_gate: bool = False

    # --- fixed constants stamped from CFG by build_cfg ---
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    mma_warp_id: int = CFG.MMA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK
    dyn_sched: bool = False
    sched_stages: int = CFG.SMEM_SCHED_STAGES

    # --- named barrier slots (ids 1-6; 0 is the CTA-wide sync) ---
    tmem_alloc_barrier_id: int = 1
    tmem_alloc_barrier_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0
    inverse_inner_barrier_id: int = 3
    inverse_inner_barrier_threads: int = 0
    init_state_store_barrier_id: int = 4
    init_state_store_barrier_threads: int = 0
    cg1_barrier_id: int = 5
    cg1_barrier_threads: int = 0

    # --- SMEM / TMEM stage counts + TMEM column offsets ---
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_do_stages: int = 1
    smem_s_stages: int = 1
    smem_ainv_stages: int = CFG.SMEM_AINV_STAGES
    smem_qk_stages: int = CFG.SMEM_QK_STAGES
    smem_dq_stages: int = 1
    smem_dk_stages: int = 1
    smem_dv_stages: int = 1
    smem_gate_stages: int = 2
    smem_beta_stages: int = 2
    tmem_dh_acc_stages: int = CFG.TMEM_DH_ACC_STAGES
    tmem_dvdk_acc_stages: int = CFG.TMEM_DVDK_ACC_STAGES
    tmem_dh_inp_stages: int = CFG.TMEM_DH_INP_STAGES
    tmem_shared_inp_stages: int = CFG.TMEM_SHARED_INP_STAGES
    tmem_shared_acc_stages: int = CFG.TMEM_SHARED_ACC_STAGES
    tmem_dh_offset: int = 0
    tmem_dvdk_offset: int = 0
    tmem_dh_inp_offset: int = 0
    tmem_shared_acc_offset: int = 0
    tmem_shared_inp_offset: int = 0
    tmem_y_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # --- stamped by _host at trace time (shape-derived) ---
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    do_cosize: int = 0
    s_cosize: int = 0
    ainv_cosize: int = 0
    qk_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dv_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_do_bytes: int = 0
    tma_s_bytes: int = 0
    n_heads_out: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dht: bool = False,
    split_k: bool = False,
    log_gate: bool = False,
    dyn_sched: bool = False,
) -> GdnBwdCfg:
    """Build the per-compile ``GdnBwdCfg`` (io_dtype in {Float16, BFloat16};
    acc is always Float32)."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = GdnBwdCfg(
        use_initial_state=use_initial_state,
        use_dht=use_dht,
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        max_active_clusters=max_active_clusters,
        is_GQA=is_GQA,
        split_k=split_k,
        log_gate=log_gate,
        dyn_sched=dyn_sched,
    )
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1)
    cfg.tmem_alloc_barrier_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1)
    cfg.inverse_barrier_threads = cfg.threads_per_warp * n_cg0
    cfg.inverse_inner_barrier_threads = cfg.threads_per_warp * 2
    cfg.init_state_store_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.cg1_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.tmem_dh_offset = 0
    cfg.tmem_dvdk_offset = cfg.tmem_dh_offset + cfg.tmem_dh_acc_stages * 128
    cfg.tmem_dh_inp_offset = cfg.tmem_dvdk_offset + cfg.tmem_dvdk_acc_stages * 64
    cfg.tmem_shared_acc_offset = cfg.tmem_dh_inp_offset + cfg.tmem_dh_inp_stages * 64
    cfg.tmem_shared_inp_offset = cfg.tmem_shared_acc_offset + cfg.tmem_shared_acc_stages * 64
    cfg.tmem_y_offset = cfg.tmem_shared_inp_offset + cfg.tmem_shared_inp_stages * (cfg.b_t // 2)
    return cfg


def get_workspace_size(B: int, HQ: int, HV: int):
    HO = HQ if HQ >= HV else HV
    return CFG.BYTES_PER_TENSORMAP * (9 * B * HO) + 128


def _check_cuda(err):
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")


def _data_ptr(t) -> int:
    """Device address of a tensor-like (``data_ptr()`` or the CUDA array
    interface)."""
    fn = getattr(t, "data_ptr", None)
    if fn is not None:
        return fn()
    return t.__cuda_array_interface__["data"][0]


def _device_sm_count() -> int:
    """Multiprocessor count of the current device (runtime API: auto-inits
    the primary context, so this works before any other CUDA call)."""
    from cuda.bindings import runtime as _rt

    err, dev = _rt.cudaGetDevice()
    if int(err) != 0:
        raise RuntimeError(f"cudaGetDevice failed: {err}")
    err, count = _rt.cudaDeviceGetAttribute(_rt.cudaDeviceAttr.cudaDevAttrMultiProcessorCount, dev)
    if int(err) != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute failed: {err}")
    return count


def _cutlass_io_dtype(dtype):
    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float16" in name or "half" in name:
        return cutlass.Float16
    raise ValueError(f"Unsupported dtype {dtype}, expected bfloat16 or float16")


@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dht: bool = False,
    split_k: bool = False,
    log_gate: bool = False,
    dyn_sched: bool = False,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dht: bool = False,
    split_k: bool = False,
    log_gate: bool = False,
    dyn_sched: bool = False,
    *,
    num_sm: int,
    q_cute,
    k_cute,
    v_cute,
    gate_cute,
    beta_cute,
    dg_cute,
    dbeta_cute,
    do_cute,
    dq_cute,
    dk_cute,
    dv_cute,
    cu_seqlens_cute,
    ds0_cute=None,
    dht_cute=None,
    work_items_cute=None,
    work_count_cute=None,
    sched_ctr_cute=None,
    scale=None,
    workspace_cute=None,
    stream=None,
):
    """JIT-compile the chunked GDN bprop kernel for one static config."""
    cfg = build_cfg(
        io_dtype,
        max_active_clusters=num_sm,
        is_GQA=is_GQA,
        use_initial_state=use_initial_state,
        use_dht=use_dht,
        split_k=split_k,
        log_gate=log_gate,
        dyn_sched=dyn_sched,
    )

    return cute.compile(
        _host,
        cfg,
        q_cute,
        k_cute,
        v_cute,
        gate_cute,
        beta_cute,
        dg_cute,
        dbeta_cute,
        do_cute,
        dq_cute,
        dk_cute,
        dv_cute,
        cu_seqlens_cute,
        ds0_cute,
        dht_cute,
        work_items_cute,
        work_count_cute,
        sched_ctr_cute,
        scale,
        workspace_cute,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_gdn_bwd_sm100(
    q,
    k,
    v,
    gate,
    beta,
    do,
    h,
    dq,
    dk,
    dv,
    dg,
    dbeta,
    cu_seqlens,
    scale: float,
    *,
    initial_state=None,
    d_initial_state=None,
    d_final_state=None,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    log_gate: bool = False,
    workspace,
    stream,
) -> None:
    """Execute the Blackwell chunked GDN bprop kernel (THD / varlen entry).

    Produces dQ/dK/dV/dG/dBeta at ``HO = max(HQ, HV)`` heads (the caller
    reduces over the head group; dgate = dL/d(ln alpha)).  With
    ``initial_state`` (io dtype ``(num_seqs, HO, DK, DV)``, K-major — the
    caller downcasts its fp32 state), chunk 0's forward state loads from it
    through a dedicated per-(b,h) descriptor set; ``d_initial_state`` (fp32,
    same shape) then also receives dL/dS0.  The two go together.  ``h`` is
    always the PLAIN per-chunk series.  All tensors are contiguous,
    DLPack-compatible CUDA tensors on the same device.
    Compile-cache-and-replay.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate — raw linear
            alpha, or the natural-log decay when ``log_gate``
        beta: ``(total_tokens, HO)`` float32, update gate
        do: ``(total_tokens, HO, DV)`` float16/bfloat16, output gradient
        h: ``(total_h, HO, DK, DV)`` bfloat16, per-chunk forward states from
            the prefill kernel's H output (``checkpoint_every_n_tokens=B_T``)
        dq/dk/dv: pre-allocated output gradients, shaped/typed like q/k/v at
            HO heads
        dg/dbeta: pre-allocated ``(total_tokens, HO)`` float32 gate/beta
            gradients
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DV)`` io dtype (matching ``h``),
            or None
        scale: attention scale factor (must not be 0)
        work_items: ``(max_items, 6)`` int32 split-K work-item table from
            ``common/split_k.py``, or None for the one-tile-per-(b,h)
            layout.  With a table, each item computes chunks
            ``[wstart, cend)`` backward and writes gradients only for
            ``[wstart, wend)``.
        work_count: ``(1,)`` int32 device-side item count (required with
            work_items)
        workspace: ``(>= get_workspace_size(B, HQ, HV) // 8,)`` int64,
            128-byte aligned; holds the per-(b,h) TMA descriptors
        stream: CUDA stream handle (``cudaStream_t`` as an int)
    """
    HQ = q.shape[1]
    HV = v.shape[1]
    DK = q.shape[2]
    B = cu_seqlens.shape[0] - 1
    is_GQA = HQ >= HV
    split_k = work_items is not None
    io_dtype = _cutlass_io_dtype(q.dtype)
    if str(h.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
        raise ValueError(f"h dtype must match the io dtype (the prefill H output): got {h.dtype} with io {q.dtype}")
    if (initial_state is None) != (d_initial_state is None):
        raise ValueError("initial_state and d_initial_state go together (chunk 0 reads S0; the drain produces dS0)")
    if initial_state is not None and str(initial_state.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
        raise ValueError(f"initial_state must be io dtype (the caller downcasts): got {initial_state.dtype} with io {q.dtype}")
    if split_k:
        if work_count is None:
            raise ValueError("work_count is required with work_items")
    elif work_count is not None:
        raise ValueError("work_count must be None without work_items")

    ws_words = get_workspace_size(B, HQ, HV) // 8
    if workspace.shape[0] < ws_words:
        raise ValueError(f"workspace too small: need {ws_words} int64 words, " f"got {workspace.shape[0]}")
    if _data_ptr(workspace) % 128 != 0:
        raise ValueError("workspace must be 128-byte aligned")
    cu_stream = cuda.CUstream(int(stream))

    dyn_sched = sched_ctr is not None
    cache = _get_compiled_cache(str(q.dtype), HQ, k.shape[1], HV, is_GQA, d_initial_state is not None, d_final_state is not None, split_k, log_gate, dyn_sched)

    if "compiled" not in cache:

        def _tok3(t):
            c = from_dlpack(t, assumed_align=16)
            c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            return c

        def _tok2(t):
            c = from_dlpack(t, assumed_align=16)
            c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            return c

        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        workspace_cute = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()

        ds0_cute = None
        if d_initial_state is not None:
            # prefill s_out marking (the drain reuses the fs-store indexing)
            ds0_cute = from_dlpack(d_initial_state, assumed_align=16)
            ds0_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=CFG.D_K)
        dht_cute = None
        if d_final_state is not None:
            dht_cute = from_dlpack(d_final_state, assumed_align=16)
            dht_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=CFG.D_K)
        work_items_cute = None
        work_count_cute = None
        if split_k:
            work_items_cute = from_dlpack(work_items, assumed_align=4)
            work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        sched_ctr_cute = None
        if dyn_sched:
            sched_ctr_cute = from_dlpack(sched_ctr, assumed_align=4).mark_layout_dynamic()
        cache["compiled"] = compile(
            io_dtype,
            is_GQA,
            use_initial_state=d_initial_state is not None,
            use_dht=d_final_state is not None,
            split_k=split_k,
            log_gate=log_gate,
            dyn_sched=dyn_sched,
            num_sm=_device_sm_count(),
            q_cute=_tok3(q),
            k_cute=_tok3(k),
            v_cute=_tok3(v),
            gate_cute=_tok2(gate),
            beta_cute=_tok2(beta),
            dg_cute=_tok2(dg),
            dbeta_cute=_tok2(dbeta),
            do_cute=_tok3(do),
            dq_cute=_tok3(dq),
            dk_cute=_tok3(dk),
            dv_cute=_tok3(dv),
            cu_seqlens_cute=cu_seqlens_cute,
            ds0_cute=ds0_cute,
            dht_cute=dht_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            sched_ctr_cute=sched_ctr_cute,
            scale=scale,
            workspace_cute=workspace_cute,
            stream=cu_stream,
        )

    compiled = cache["compiled"]

    # The descriptors encode cu_seqlens' CONTENTS, which no key built from the
    # buffers can track. The skip this replaces asked torch's _version counter,
    # so it was sound for a torch caller and silently stale for every other
    # producer. Rebuilding unconditionally measures free: 131 vs 135 us of host
    # time, and 157 either way once the launches are waited on.
    if "build_descs" not in cache:

        def _tok3_bc(t):
            c = from_dlpack(t, assumed_align=16)
            c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            return c

        h_bc = from_dlpack(h, assumed_align=16)
        h_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3), divisibility=1)
        cu_bc = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        s0_bc = None
        if initial_state is not None:
            s0_bc = from_dlpack(initial_state, assumed_align=16)
            s0_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3), divisibility=1)

        ws_bc = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
        cache["build_descs"] = cute.compile(
            _build_descs,
            io_dtype,
            CFG.B_T,
            _tok3_bc(q),
            _tok3_bc(k),
            _tok3_bc(v),
            _tok3_bc(do),
            _tok3_bc(dq),
            _tok3_bc(dk),
            _tok3_bc(dv),
            h_bc,
            cu_bc,
            s0_bc,
            ws_bc,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["build_descs"](q, k, v, do, dq, dk, dv, h, cu_seqlens, initial_state, workspace, cu_stream)

    compiled(
        q,
        k,
        v,
        gate,
        beta,
        dg,
        dbeta,
        do,
        dq,
        dk,
        dv,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        sched_ctr,
        scale,
        workspace,
        cu_stream,
    )
