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
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], dO[BT,DV], S_c[DK,DV] (= checkpoint entry c-1,
           the forward state ENTERING chunk c), Gate[BT], Beta[BT]
  State  : dstate[DK,DV]  (state gradient, held in TMEM, accumulated backward)

  MMA order.  A = the staged attention matrix
  (CG0's A epilogue, sA); dO' = dO * cumprod_vals * scale (CG1 restage):
  KK          : W_kk[BT,BT]  = K  @ K^T        -> shared acc   (for T)
  QK          : W_qk[BT,BT]  = Q  @ K^T        -> shared acc   (for the A tile)
  dV inter    : dV[DV,BT]    = dstate^T(TMEM) @ K^T -> the dV/dK slot
                (acc=False; CG1 then scales it by decay_scale_vals IN PLACE)
  KS          : KS[BT,DV]    = S^T(SMEM) @ K^T  -> shared acc
  dU intra    : dV/dK slot  += dO^T(SMEM) @ A(SMEM)  (waits CG1's
                in-place scale -> the accumulate is the reference's
                dv2 = dv_intra + exp2(g_last-g)*(k@dh))
  dstate Q-term : dstate    += dO'^T(TMEM) @ Q  (acc=False on the
                first backward chunk = dstate init; the reference's
                dh = dh*exp2(g_last) + (q*scale*exp2(g))^T @ do)
  dY          : dY[DV,BT]    = dU^T(TMEM f16) @ T(SMEM) -> shared acc
                (dY == dV == d(Y); T read TRANSPOSED vs prefill)
  dA          : dA_eff[BT,BT]= dO(SMEM) @ U^T(SMEM) -> shared acc
                (sV holds U; CG0 masks it -> sDa for dQ/dK)
  dM core     : dM[BT,BT]    = dY(SMEM) @ U^T(SMEM) -- the WY
                inverse backward T^T dT T^T collapsed via
                T^T dU = dY and T Y = U (Beta folds through
                sTinv's column scale into both factors); CG0
                applies -strict(2^{g_i-g_j}) -> sDm (the reference's dA22;
                both betas cancel against the K rows in the dM-terms)
  dstate update-term : dstate += dY'^T(TMEM f16) @ K  (dY' = -cumprod_vals
                * dY: the -(W^T @ dY) term; ready commits HERE)
  dK dM-terms : dV/dK slot  += K^T @ dM^T + K^T @ dM  (the reference's
                dk += dA22^T@K_beta / dk_beta += dA22@K folded, the
                beta row scale cancelling with K_beta = beta*K)
  dK state-path : SY[BT,DV]    = S^T(SMEM) @ dY^T(SMEM) -> the shared f16
                input columns (dO'/dU/dY' dead by then); CG1 reads it as
                -cumprod_vals[t] * (dY @ S^T)[t,:] and adds it to the banked
                inter+attn dK terms

  Per chunk CG1: restages dO' and dU and dY' -> the f16 input columns
  (dY' overwrites dU), computes Y = V - cumprod_vals*(K @ state) in registers and stages
  Y and g_k_state to their dedicated f16 TMEM slots (the U GEMM and
  the dV-pass read them), stages Q^T over the Y slot after the dV pass,
  loads dY and stages it plain to sdV (the dV output).

SMEM layout (~226 KB of the 227 KB SM100 cap):
  Buffer                           Size (B)  Stages
  Q                                16384     1
  K                                32768     2      <-- double-buffered (prefetch next chunk)
  V                                16384     1      <-- overwritten in place by U
  dO                               16384     1
  state (checkpoint entry c-1)     32768     1      <-- io-dtype [DK,DV], TMA-loaded
  T_inv                            8192      1      <-- inverse OUTPUT (upper tri = kernel-start zeros)
  KK (strict-masked M_kk)          8192      1      <-- KK epi's only store; inverse input + dGate/dBeta
  A staging / sDa                  8192      1      <-- ALIAS: A then the masked dA
  dM staging (sDm)                 8192      1      <-- Step 8 -> dK dM-terms
  dstate_entry (sDstate)           32768     1      <-- f16 restage, dK-inter's A
  dQ store staging                 16384     1
  dK store staging                 16384     1
  dV store staging                 16384     1
  cumsumlog / cumprod / Beta       3 x 512   2      <-- in-place dGate/dBeta staging

TMEM layout (512 cols; EXACTLY the prefill map, state->dstate, O->dV):
  cols 0-128   : dstate accumulator (fp32)     <-- prefill: state acc
  cols 128-192 : dV/dK accumulator (fp32)      <-- one slot, five sequential
                 per-chunk productions (dV inter -> dU intra -> dK inter
                 -> dK attn -> dK dM-terms), each with its own mbar pair
  cols 192-256 : dstate input (f16 packed)     <-- prefill: state_input
  cols 256-384 : shared accumulators x2        <-- KK / A / k_state / U / dY / dA
                                                    / dM core
  cols 384-448 : shared inputs x2 (f16 packed) <-- dO' / dU / dY'; the dK
                 state-path acc overwrites them after their last GEMM reads
                 (CG1's state-path readout precedes its next-chunk restages)
  cols 448-512 : Y (448) + g_k_state (480) f16 slots until the dV pass, then
                 Q^T (448) until CG1's dQ dot reads it

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise, KK epi, A epi, inverse
  warps 4-7     : compute group 1 - dV epilogue, dstate scale + f16 restage
                                    (later: Y = V - K*state -> TMEM, dY staging)
  warp  8       : MMA warp       - issues the GEMMs
  warp  9       : TMA load warp  - loads Q, K (double-buf), V, dO, state(=checkpoint)
  warp  10      : gate warp      - loads Gate/Beta (double-buffered,
                                    backward order) + stores dGate/dBeta
  warp  11      : epilogue warp   - store dQ, dK, dV to global memory
"""

import functools
from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma
from cutlass.cute.runtime import from_dlpack

from ..common.thd import emit_checkpoint_seq_descs, emit_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_work_item, order_body
from ..common.host import get_dtype
from cudnn.frost.buffers import data_ptr
from cudnn.frost.device import current_device, multiprocessor_count

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_ss, mma_step_k8, mma_ts_step, mma_step
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, ffma2, fmul2, fp32_to_fp16, opaque_f32_zero, sigmoid, softplus, softplus2, sub_f16x2
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .gdn_bprop_config import CFG


class GdnBwdBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_mma_done: MBarrier
    mb_q_cg1_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_mma_done: MBarrier
    mb_k_cg2_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_mma_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_state_ready: MBarrier
    mb_state_mma_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_scale_acc_done: MBarrier
    mb_du_scale_acc_ready: MBarrier
    mb_du_scale_acc_done: MBarrier
    mb_du_total_acc_ready: MBarrier
    mb_dk_scale_acc_ready: MBarrier
    mb_dk_scale_acc_done: MBarrier
    mb_dk_attn_acc_ready: MBarrier
    mb_dk_attn_acc_done: MBarrier
    mb_dk_total_acc_ready: MBarrier
    mb_dk_total_acc_done: MBarrier
    mb_dq_acc_scale_ready: MBarrier
    mb_dq_acc_scale_done: MBarrier
    mb_dq_acc_total_ready: MBarrier
    mb_dq_acc_total_done: MBarrier
    mb_kk_acc_ready: MBarrier
    mb_kk_acc_done: MBarrier
    mb_a_acc_ready: MBarrier
    mb_k_state_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier
    mb_da_acc_ready: MBarrier
    mb_dm_acc_ready: MBarrier
    mb_dm_acc_done: MBarrier
    mb_dk_state_path_acc_ready: MBarrier

    mb_dstate_input_ready: MBarrier
    mb_dstate_input_done: MBarrier
    mb_do_prime_input_ready: MBarrier
    mb_du_input_ready: MBarrier
    mb_dyp_input_ready: MBarrier
    mb_y_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier
    mb_u_ready: MBarrier
    mb_dstate_smem_ready: MBarrier
    mb_state_dot_dstate_done: MBarrier
    mb_da_ready: MBarrier

    mb_dbeta_cg1_ready: MBarrier
    mb_dgate_cg1_ready: MBarrier
    mb_dgate_cg0_ready: MBarrier
    mb_cg2_chunk_ready: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier
    mb_sdv_done: MBarrier

    mb_tmem_done: MBarrier
    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_gdn_bars(cfg) -> GdnBwdBars:
    """GdnBwdBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG2_THREADS = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS
    ALL_COMPUTE = CG0_THREADS + CG1_THREADS + CG2_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBwdBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_mma_done=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_q_cg1_done=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_mma_done=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_k_cg2_done=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_mma_done=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_ready=MBarrier(alloc(cfg.smem_do_stages), stages=cfg.smem_do_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_do_stages), stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_state_ready=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_state_mma_done=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=ALL_COMPUTE, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=CG0_PLUS_CG1, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages), stages=cfg.tmem_dstate_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_scale_acc_done=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages), stages=cfg.tmem_dstate_acc_stages, init_count=CG2_THREADS, producer=Producer.THREAD
        ),
        mb_du_scale_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_du_scale_acc_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_du_total_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_scale_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_scale_acc_done=MBarrier(alloc(1), stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dk_attn_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_attn_acc_done=MBarrier(alloc(1), stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dk_total_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dk_total_acc_done=MBarrier(alloc(1), stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dq_acc_scale_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dq_acc_scale_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dq_acc_total_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dq_acc_total_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_kk_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_kk_acc_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_k_state_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dy_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_da_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dm_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dm_acc_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dk_state_path_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dstate_input_ready=MBarrier(
            alloc(cfg.tmem_dstate_input_stages), stages=cfg.tmem_dstate_input_stages, init_count=CG2_THREADS, producer=Producer.THREAD
        ),
        mb_dstate_input_done=MBarrier(
            alloc(cfg.tmem_dstate_input_stages), stages=cfg.tmem_dstate_input_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_do_prime_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_du_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dyp_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_y_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_t_inv_stages), stages=cfg.smem_t_inv_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_ready=MBarrier(alloc(cfg.smem_a_stages), stages=cfg.smem_a_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_a_stages), stages=cfg.smem_a_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_u_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dstate_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_state_dot_dstate_done=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_da_ready=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dbeta_cg1_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dgate_cg1_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dgate_cg0_ready=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_cg2_chunk_ready=MBarrier(alloc(1), stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dq_tmastg_ready=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dq_tmastg_done=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_dk_tmastg_ready=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dk_tmastg_done=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_dv_tmastg_ready=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dv_tmastg_done=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_sdv_done=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=ALL_COMPUTE, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
    )


# ---- Dynamic tile scheduler ----------------------------------------------------------


@cute.jit
def scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, tail_base, tail_row, elect_one):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    if cutlass.const_expr(cfg.dynamic_scheduling):
        sentinel = cutlass.Int32(1 << 28)
        pinned = tail_row if tile_idx < tail_base else sentinel
        bars.mb_scheduler_done[scheduler_state.idx].wait(scheduler_state.phase)
        if elect_one:
            fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
            granted = num_ctas + fetched
            sScheduler[scheduler_state.idx] = granted if granted < tail_base else pinned
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        next_tile = sScheduler[scheduler_state.idx]
        if elect_one:
            bars.mb_scheduler_ready[scheduler_state.idx].arrive()
        return next_tile, advance(scheduler_state, cfg.scheduler_stages)
    return tile_idx + num_ctas, scheduler_state


@cute.jit
def scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    if cutlass.const_expr(cfg.dynamic_scheduling):
        bars.mb_scheduler_ready[scheduler_state.idx].wait(scheduler_state.phase)
        next_tile = sScheduler[scheduler_state.idx]
        if elect_one:
            bars.mb_scheduler_done[scheduler_state.idx].arrive()
        return next_tile, advance(scheduler_state, cfg.scheduler_stages)
    return tile_idx + num_ctas, scheduler_state


@cute.jit
def invert_diagonal_NxN(cfg, in_base, out_base, d_idx, tidx, N: int = 8):
    """Gauss-Jordan inversion of one diagonal NxN block, ``in_base`` -> ``out_base`` (f16 SMEM)."""
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_coord = d_idx * N + tidx_in_group
    row_off = row_coord * BT + swizzle_xor_128b(row_coord, d_idx * N)
    row_ptr_in = in_base + row_off
    row_ptr = out_base + row_off

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
def warp_reduce_scatter_sum_16_elements(vals, lane_idx):
    """Reduce-scatter 16 fragment token-partials (tcol = (lane%4)*2 +
    (j//2)*8 + (j%2)); lane L returns the sums of tokens (L//4)*8 + (L%4)*2 and +1."""
    level = vals
    for k in cutlass.range_constexpr(3):
        off = cutlass.const_expr(4 << k)
        hi_lane = (lane_idx // off) % 2 == 1
        next_level = []
        for i in cutlass.range_constexpr(len(level)):
            if cutlass.const_expr((i & 2) == 0):
                lo = level[i]
                hi = level[i + 2]
                send = lo if hi_lane else hi
                recv = nvvm.shfl_sync(0xFFFFFFFF, send, off, 31, kind=nvvm.Shfl.BFLY)
                keep = hi if hi_lane else lo
                next_level.append(keep + recv)
        level = next_level
    return level[0], level[1]


@cute.jit
def blockwise_diagonal_8x8_to_16x16(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}); raw C from ``raw_base``, writes on ``base``."""
    BT = cfg.b_t
    row_lo = d_idx + lane_idx % 8
    row_hi = row_lo + 8
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 8)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx)
    d_inv_frag = nvvm.ldmatrix(base + off_d_inv, 1, nvvm.MMALayout.ROW)
    c_frag = nvvm.ldmatrix(raw_base + off_c, 1, nvvm.MMALayout.COL)

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(c_regs, [d_inv_frag, d_inv_frag], [c_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(2)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frag = nvvm.ldmatrix(base + off_a_inv, 1, nvvm.MMALayout.COL)
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [a_inv_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    o_pack = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_16x16_to_32x32(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 16x16 -> 32x32 (raw C from ``raw_base``)."""
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_lo = d_idx + lane_row
    row_hi = row_lo + 16
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 16 + lane_col)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + lane_col)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx + lane_col)
    d_inv_frags = list(nvvm.ldmatrix(base + off_d_inv, 4, nvvm.MMALayout.ROW))
    c_frags = list(nvvm.ldmatrix(raw_base + off_c, 4, nvvm.MMALayout.COL))

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d_inv_frags, c_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = list(nvvm.ldmatrix(base + off_a_inv, 4, nvvm.MMALayout.COL))
    o_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_pack, a_inv_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_32x32_to_64x64(cfg, base, raw_base, warp_id, lane_idx):
    """Off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row M-band each; raw C from ``raw_base``)."""
    band = warp_id % 2
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_d_inv = 32 + band * 16 + lane_row
    d_inv_frags = []
    for vs in cutlass.range_constexpr(2):
        d_inv_frags += list(nvvm.ldmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 32 + vs * 16 + lane_col), 4, nvvm.MMALayout.ROW))
    c_frags = []
    for vs in cutlass.range_constexpr(4):
        row_c = 32 + (vs // 2) * 16 + lane_row
        c_frags += list(nvvm.ldmatrix(raw_base + row_c * BT + swizzle_xor_128b(row_c, (vs % 2) * 16 + lane_col), 4, nvvm.MMALayout.COL))

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(c_regs, d_inv_frags, c_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = []
    for vs in cutlass.range_constexpr(4):
        row_a_inv = (vs // 2) * 16 + lane_row
        a_inv_frags += list(nvvm.ldmatrix(base + row_a_inv * BT + swizzle_xor_128b(row_a_inv, (vs % 2) * 16 + lane_col), 4, nvvm.MMALayout.COL))
    o_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(o_regs, a_pack, a_inv_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_inner_barrier_id,
        thread_count=cfg.inverse_inner_barrier_threads,
    )
    nvvm.stmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, lane_col), o_pack[0:4], nvvm.MMALayout.ROW)
    nvvm.stmatrix(base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 16 + lane_col), o_pack[4:8], nvvm.MMALayout.ROW)


@cute.jit
def tmastg_warp(
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
    sScheduler,
    bars,
):
    """TMA-STG warp role (warp 11): persistent tile-scheduler loop + per-chunk
    dQ/dK/dV TMA bulk-stores from the SMEM staging buffers to global memory."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    bpe = cfg.io_dtype.width // 8
    granule_elements = 128 // bpe
    sdQ_tma = SmemTile(
        base=sdQ_raw,
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages),
        stages=cfg.smem_dq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elements,
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
        tma_granu_elems=granule_elements,
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
        tma_granu_elems=granule_elements,
        tma_subtile_stride_elems=4096,
    )
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )

        head_o = head_idx
        slot = batch_idx * desc_qwords
        desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)

        for rev_idx in cutlass.range(compute_end - write_start):
            chunk_idx = compute_end - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            dv_idx = dv_index.idx
            bars.mb_dv_tmastg_ready[dv_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_o, tok_coord)
            if chunk_idx < write_end:
                tma_store_tile(sdV_tma[dv_idx], dv_slice, acquire=False)
                tma_store_commit()

            dq_idx = dq_index.idx
            bars.mb_dq_tmastg_ready[dq_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, tok_coord)
            if chunk_idx < write_end:
                tma_store_tile(sdQ_tma[dq_idx], dq_slice, acquire=False)
                tma_store_commit()

            dk_idx = dk_index.idx
            bars.mb_dk_tmastg_ready[dk_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, tok_coord)
            if chunk_idx < write_end:
                tma_store_tile(sdK_tma[dk_idx], dk_slice, acquire=False)
                tma_store_commit()

            tma_store_wait(2)
            bars.mb_dv_tmastg_done[dv_idx].arrive()
            tma_store_wait(1)
            bars.mb_dq_tmastg_done[dq_idx].arrive()
            tma_store_wait(0)
            bars.mb_dk_tmastg_done[dk_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


@cute.jit
def gate_beta_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    mGate,
    mA_log,
    mDt_bias,
    mBeta,
    mDgate,
    mDbeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sScheduler,
    bars,
):
    """Gate/beta LOAD + STORE warp role (warp 10): per-chunk Gate/Beta G->S
    loads and the dGate/dBeta stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    gate_store_index = PipelineState.start(phase=0)
    beta_store_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    lane_idx = tidx % cfg.threads_per_warp

    n_cols = cfg.b_t // cfg.threads_per_warp
    a = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start
        if cutlass.const_expr(cfg.safe_gate):
            if num_item_chunks > 0:
                a = -cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
                bias = mDt_bias[head_idx].to(cutlass.Float32)
        write_end_token = batch_start + write_end * cfg.b_t
        write_end_token = write_end_token if write_end_token < batch_end else batch_end

        # ---- prefetch: the FIRST backward chunk's Gate/Beta --------------------------
        if num_item_chunks > 0:
            chunk_offset = batch_start + (compute_end - 1) * cfg.b_t
            gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
            gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
            gate_idx = gate_index.idx
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            pos_valid = [(chunk_offset + lane_idx + col * cfg.threads_per_warp) < batch_end for col in range(n_cols)]

            # ---- Gate load: GMEM -> SMEM (OOB neutral: 1.0 -> log2 = 0.0) ------------
            oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
            gate_vals = [gGate[lane_idx + col * cfg.threads_per_warp] if pos_valid[col] else oob_neutral for col in range(n_cols)]

            if cutlass.const_expr(cfg.safe_gate):
                for col in cutlass.range_constexpr(0, n_cols, 2):
                    biased_lo, biased_hi = fadd2(gate_vals[col], gate_vals[col + 1], bias, bias)
                    sp_lo, sp_hi = softplus2(biased_lo, biased_hi)
                    contrib_lo, contrib_hi = fmul2(sp_lo, sp_hi, a, a)
                    gate_vals[col] = contrib_lo if pos_valid[col] else cutlass.Float32(0.0)
                    gate_vals[col + 1] = contrib_hi if pos_valid[col + 1] else cutlass.Float32(0.0)
            elif cutlass.const_expr(cfg.log_gate):
                rcp_ln2 = opaque_f32_zero() + cutlass.Float32(RCP_LN2)
                for col in cutlass.range_constexpr(0, n_cols, 2):
                    gate_vals[col], gate_vals[col + 1] = fmul2(gate_vals[col], gate_vals[col + 1], rcp_ln2, rcp_ln2)
            else:
                floor = cutlass.Float32(1e-10)
                for col in cutlass.range_constexpr(0, n_cols, 2):
                    shifted_lo, shifted_hi = fadd2(gate_vals[col], gate_vals[col + 1], floor, floor)
                    gate_vals[col] = cute.math.log2(shifted_lo, fastmath=True)
                    gate_vals[col + 1] = cute.math.log2(shifted_hi, fastmath=True)
            for offset in [1, 2, 4, 8, 16]:
                for col in cutlass.range_constexpr(n_cols):
                    n = nvvm.shfl_sync(0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP)
                    if lane_idx >= offset:
                        gate_vals[col] = gate_vals[col] + n
            for col in cutlass.range_constexpr(1, n_cols):
                last_v = nvvm.shfl_sync(
                    0xFFFFFFFF,
                    gate_vals[col - 1],
                    cfg.threads_per_warp - 1,
                    cfg.threads_per_warp - 1,
                    kind=nvvm.Shfl.IDX,
                )
                gate_vals[col] += last_v

            for col in cutlass.range_constexpr(n_cols):
                pos = lane_idx + col * cfg.threads_per_warp
                sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

            bars.mb_gate_ready[gate_idx].arrive()

            # ---- Beta load: GMEM -> SMEM (per-element cp.async) ----------------------
            beta_idx = beta_index.idx
            beta_index = advance(beta_index, cfg.smem_beta_stages)
            if cutlass.const_expr(cfg.beta_sigmoid):
                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    beta_value = cutlass.Float32(0.0)
                    if pos_valid[col]:
                        beta_value = gBeta[pos].to(cutlass.Float32)
                        beta_value = sigmoid(beta_value).to(mBeta.element_type).to(cutlass.Float32)
                    sBeta[pos, 0, beta_idx] = beta_value
                bars.mb_beta_ready[beta_idx].arrive()
            else:
                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    src = gBeta.iterator + gBeta.layout((pos,))
                    dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                    cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                    nvvm.cp_async_shared_global(dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size)
                nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)

        for rev_idx in cutlass.range(num_item_chunks):
            # ---- prefetch the NEXT chunk's Gate/Beta ---------------------------------
            if rev_idx + 1 < num_item_chunks:
                chunk_offset = batch_start + (compute_end - 2 - rev_idx) * cfg.b_t
                gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
                gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
                gate_idx = gate_index.idx
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                pos_valid = [(chunk_offset + lane_idx + col * cfg.threads_per_warp) < batch_end for col in range(n_cols)]

                oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
                gate_vals = [gGate[lane_idx + col * cfg.threads_per_warp] if pos_valid[col] else oob_neutral for col in range(n_cols)]

                if cutlass.const_expr(cfg.safe_gate):
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        biased_lo, biased_hi = fadd2(gate_vals[col], gate_vals[col + 1], bias, bias)
                        sp_lo, sp_hi = softplus2(biased_lo, biased_hi)
                        contrib_lo, contrib_hi = fmul2(sp_lo, sp_hi, a, a)
                        gate_vals[col] = contrib_lo if pos_valid[col] else cutlass.Float32(0.0)
                        gate_vals[col + 1] = contrib_hi if pos_valid[col + 1] else cutlass.Float32(0.0)
                elif cutlass.const_expr(cfg.log_gate):
                    rcp_ln2 = opaque_f32_zero() + cutlass.Float32(RCP_LN2)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        gate_vals[col], gate_vals[col + 1] = fmul2(gate_vals[col], gate_vals[col + 1], rcp_ln2, rcp_ln2)
                else:
                    floor = cutlass.Float32(1e-10)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        shifted_lo, shifted_hi = fadd2(gate_vals[col], gate_vals[col + 1], floor, floor)
                        gate_vals[col] = cute.math.log2(shifted_lo, fastmath=True)
                        gate_vals[col + 1] = cute.math.log2(shifted_hi, fastmath=True)
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP)
                        if lane_idx >= offset:
                            gate_vals[col] = gate_vals[col] + n
                for col in cutlass.range_constexpr(1, n_cols):
                    last_v = nvvm.shfl_sync(
                        0xFFFFFFFF,
                        gate_vals[col - 1],
                        cfg.threads_per_warp - 1,
                        cfg.threads_per_warp - 1,
                        kind=nvvm.Shfl.IDX,
                    )
                    gate_vals[col] += last_v

                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

                bars.mb_gate_ready[gate_idx].arrive()

                beta_idx = beta_index.idx
                beta_index = advance(beta_index, cfg.smem_beta_stages)
                if cutlass.const_expr(cfg.beta_sigmoid):
                    for col in cutlass.range_constexpr(n_cols):
                        pos = lane_idx + col * cfg.threads_per_warp
                        beta_value = cutlass.Float32(0.0)
                        if pos_valid[col]:
                            beta_value = gBeta[pos].to(cutlass.Float32)
                            beta_value = sigmoid(beta_value).to(mBeta.element_type).to(cutlass.Float32)
                        sBeta[pos, 0, beta_idx] = beta_value
                    bars.mb_beta_ready[beta_idx].arrive()
                else:
                    for col in cutlass.range_constexpr(n_cols):
                        pos = lane_idx + col * cfg.threads_per_warp
                        src = gBeta.iterator + gBeta.layout((pos,))
                        dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                        cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                        nvvm.cp_async_shared_global(dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size)
                    nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)

            # ---- dGate/dBeta store: SMEM -> GMEM (in place) --------------------------
            st_offset = batch_start + (compute_end - 1 - rev_idx) * cfg.b_t
            gDgate = cute.domain_offset((st_offset,), mDgate[None, head_idx])
            gDbeta = cute.domain_offset((st_offset,), mDbeta[None, head_idx])

            # ---- dGate store ---------------------------------------------------------
            gate_store_idx = gate_store_index.idx
            bars.mb_gate_done[gate_store_idx].wait(gate_store_index.phase)
            gate_store_index = advance(gate_store_index, cfg.smem_gate_stages)
            dgate_vals = [sCumsumlog[lane_idx + col * cfg.threads_per_warp, 0, gate_store_idx] for col in range(n_cols)]
            for offset in [1, 2, 4, 8, 16]:
                for col in cutlass.range_constexpr(n_cols):
                    n = nvvm.shfl_sync(0xFFFFFFFF, dgate_vals[col], offset, 31, kind=nvvm.Shfl.DOWN)
                    if lane_idx < cfg.threads_per_warp - offset:
                        dgate_vals[col] = dgate_vals[col] + n
            for col in cutlass.range_constexpr(n_cols - 1):
                rev_col = cutlass.const_expr(n_cols - 2 - col)
                later_total = nvvm.shfl_sync(0xFFFFFFFF, dgate_vals[rev_col + 1], 0, 0, kind=nvvm.Shfl.IDX)
                dgate_vals[rev_col] = dgate_vals[rev_col] + later_total
            for col in cutlass.range_constexpr(n_cols):
                pos = lane_idx + col * cfg.threads_per_warp
                if st_offset + pos < write_end_token:
                    gDgate[pos] = dgate_vals[col]

            # ---- dBeta store ---------------------------------------------------------
            beta_store_idx = beta_store_index.idx
            bars.mb_beta_done[beta_store_idx].wait(beta_store_index.phase)
            beta_store_index = advance(beta_store_index, cfg.smem_beta_stages)
            for col in cutlass.range_constexpr(n_cols):
                pos = lane_idx + col * cfg.threads_per_warp
                if st_offset + pos < write_end_token:
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        gDbeta[pos] = sBeta[pos, 0, beta_store_idx].to(mDbeta.element_type)
                    else:
                        gDbeta[pos] = sBeta[pos, 0, beta_store_idx]
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_base_slot,
    sQ,
    sQ_trans,
    sK,
    sK_trans,
    sV_trans,
    sV,
    sdO_trans,
    sdO,
    sState_trans,
    sState,
    sTinv,
    sTinv_trans,
    sA,
    sA_trans,
    sDa,
    sDa_trans,
    sDstate_trans,
    sDm,
    sDm_trans,
    sdV,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 8): persistent scheduler loop issuing every
    tcgen05 GEMM."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    kk_acc_index = PipelineState.start(phase=0)
    dk_total_index = PipelineState.start(phase=1)
    du_scale_index = PipelineState.start(phase=0)
    dk_scale_index = PipelineState.start(phase=0)
    dk_attn_index = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0 if cfg.use_dstate_in else 1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)
    state_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    y_ready_index = PipelineState.start(phase=0)
    u_index = PipelineState.start(phase=0)
    da_ready_index = PipelineState.start(phase=0)
    dv_ready_index = PipelineState.start(phase=0)
    dm_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    dq_scale_index = PipelineState.start(phase=0)
    dq_total_index = PipelineState.start(phase=1)
    a_index = PipelineState.start(phase=0)
    do_prime_input_ready = PipelineState.start(phase=0)
    du_input_ready = PipelineState.start(phase=0)
    dyp_input_ready = PipelineState.start(phase=0)
    dstate_input_index = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_k_k_desc = MmaDesc(
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
    bmm_q_k_desc = bmm_k_k_desc
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_input_col
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_state_path_col = tmem_shared_input_col

    idesc_dv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dstate_k_desc = MmaDesc(
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
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset
    tmem_dvdk_acc_col = tmem_col + cfg.tmem_dvdk_acc_offset
    DSTATE_INP_STAGE_COLS = cfg.d_k // 2

    idesc_k_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
    )
    bmm_state_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_k_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_k_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_state_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_k_state,
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
    bmm_do_a_desc = MmaDesc(
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
    idesc_dstate_upd = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_do_prime_q_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dstate_upd,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_dy_prime_k_desc = bmm_do_prime_q_desc
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_input_col
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_dstate_acc_col = tmem_col + cfg.tmem_dstate_acc_offset
    tmem_y_col = tmem_col + cfg.tmem_y_offset

    idesc_dy = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_du_t_inv_trans_desc = MmaDesc(
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
    bmm_do_u_desc = MmaDesc(
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
    bmm_dy_u_desc = bmm_do_u_desc

    idesc_u = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_y_t_inv_desc = MmaDesc(
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
    idesc_dqdk_inter_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_state_do_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dqdk_inter_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_dstate_u_desc = bmm_state_do_desc
    bmm_state_dy_desc = bmm_state_do_desc
    idesc_dka = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_k_dm_trans_desc = MmaDesc(
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
    bmm_q_da_trans_desc = bmm_k_dm_trans_desc
    idesc_dqa = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_k_dm_desc = MmaDesc(
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
    bmm_k_da_desc = bmm_k_dm_desc

    do_prime_input_ptr = nvvm.make_tmem_ptr(tmem_do_prime_col, cutlass.Int8)
    y_input_ptr = nvvm.make_tmem_ptr(tmem_y_col, cutlass.Int8)
    du_input_ptr = nvvm.make_tmem_ptr(tmem_du_col, cutlass.Int8)
    dyp_input_ptr = nvvm.make_tmem_ptr(tmem_dyp_col, cutlass.Int8)
    kk_acc_ptr = nvvm.make_tmem_ptr(tmem_kk_col, cutlass.Float32)
    a_acc_ptr = nvvm.make_tmem_ptr(tmem_a_col, cutlass.Float32)
    k_state_acc_ptr = nvvm.make_tmem_ptr(tmem_k_state_col, cutlass.Float32)
    u_acc_ptr = nvvm.make_tmem_ptr(tmem_u_col, cutlass.Float32)
    dy_acc_ptr = nvvm.make_tmem_ptr(tmem_dy_col, cutlass.Float32)
    da_acc_ptr = nvvm.make_tmem_ptr(tmem_da_col, cutlass.Float32)
    dm_core_acc_ptr = nvvm.make_tmem_ptr(tmem_dm_core_col, cutlass.Float32)
    dk_state_path_acc_ptr = nvvm.make_tmem_ptr(tmem_dk_state_path_col, cutlass.Float32)
    dstate_acc_ptr = nvvm.make_tmem_ptr(tmem_dstate_acc_col, cutlass.Float32)
    dq_acc_ptr = nvvm.make_tmem_ptr(tmem_dstate_input_col, cutlass.Float32)
    dvdk_acc_ptr = nvvm.make_tmem_ptr(tmem_dvdk_acc_col, cutlass.Float32)

    # ---- warp-top descriptors (1-stage tiles are loop-constant; K advances) ----------
    d_q0 = sQ[0].desc()
    d_k0 = sK[0].desc()
    d_k_trans0 = sK_trans[0].desc()
    d_q_trans0 = sQ_trans[0].desc()
    d_do_trans0 = sdO_trans[0].desc()
    d_do0 = sdO[0].desc()
    d_state_trans0 = sState_trans[0].desc()
    d_state0 = sState[0].desc()
    d_tinv0 = sTinv[0].desc()
    d_tinv_trans0 = sTinv_trans[0].desc()
    d_a_trans0 = sA_trans[0].desc()
    d_v0 = sV[0].desc()
    d_dv0 = sdV[0].desc()
    d_da0 = sDa[0].desc()
    d_da_trans0 = sDa_trans[0].desc()
    d_dm0 = sDm[0].desc()
    d_dm_trans0 = sDm_trans[0].desc()
    d_dstate_trans0 = sDstate_trans[0].desc()
    K_STAGE_BYTES = (cfg.k_cosize // cfg.smem_k_stages) * (cfg.io_dtype.width // 8)

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- chunks NT-2 .. 0 (backward): full body ----------------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            chunk_idx = compute_end - 1 - rev_idx
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0

            # ---- KK = K(S) @ K^T -----------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_ready[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)

            desc_k = d_k0.advance_start_address(k_idx * K_STAGE_BYTES)
            mma_ss(
                bmm_k_k_desc,
                desc_k,
                desc_k,
                kk_acc_ptr,
                accumulate=False,
            )

            if elect_one:
                bars.mb_kk_acc_ready[0].arrive(cta_group=1)

            # ---- QK = Q(S) @ K^T -----------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_ready[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)

            desc_q = d_q0
            mma_ss(
                bmm_q_k_desc,
                desc_q,
                desc_k,
                a_acc_ptr,
                accumulate=False,
            )

            if elect_one:
                bars.mb_a_acc_ready[0].arrive(cta_group=1)

            # ---- k state = state^T(S) @ K^T ------------------------------------------
            state_idx = state_index.idx
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                bars.mb_kk_acc_done[0].wait(kk_acc_index.phase)
                kk_acc_index = advance(kk_acc_index, 1)
                desc_state = d_state0
                mma_ss(
                    bmm_state_k_desc,
                    desc_state,
                    desc_k,
                    k_state_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_k_state_acc_ready[0].arrive(cta_group=1)

            # ---- dV inter = dstate^T(T) @ K ------------------------------------------
            dstate_input_idx = dstate_input_index.idx
            if have_dstate:
                bars.mb_dstate_input_ready[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
            bars.mb_dk_total_acc_done[0].wait(dk_total_index.phase)
            dk_total_index = advance(dk_total_index, 1)

            if have_dstate:
                dstate_a_ptr = nvvm.make_tmem_ptr(tmem_dstate_input_col + dstate_input_idx * DSTATE_INP_STAGE_COLS, cutlass.Int8)
                for i in cutlass.range_constexpr(bmm_dstate_k_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_k_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_k_desc,
                            dstate_a_ptr.subview(i * bmm_dstate_k_desc.sps_B * bmm_dstate_k_desc.tmem_advance_A),
                            desc_k + i * (bmm_dstate_k_desc.smem_subtile_B >> 4),
                            dvdk_acc_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )
                if elect_one:
                    bars.mb_du_scale_acc_ready[0].arrive(cta_group=1)
                    bars.mb_dstate_input_done[dstate_input_idx].arrive(cta_group=1)

            # ---- dQ inter = state(S) @ dO^T ------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
                desc_state_dq_inter = d_state_trans0
                desc_do_dq_inter = d_do0
                mma_ss(
                    bmm_state_do_desc,
                    desc_state_dq_inter,
                    desc_do_dq_inter,
                    dq_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dq_acc_scale_ready[0].arrive(cta_group=1)

            # ---- dU intra += dO^T(S) @ A ---------------------------------------------
            du_a_idx = a_index.idx
            bars.mb_a_ready[du_a_idx].wait(a_index.phase)
            a_index = advance(a_index, cfg.smem_a_stages)
            if have_dstate:
                bars.mb_du_scale_acc_done[0].wait(du_scale_index.phase)
                du_scale_index = advance(du_scale_index, 1)

            desc_do_trans = d_do_trans0
            desc_a_trans = d_a_trans0
            mma_ss(
                bmm_do_a_desc,
                desc_do_trans,
                desc_a_trans,
                dvdk_acc_ptr,
                accumulate=have_dstate,
            )
            if elect_one:
                bars.mb_du_total_acc_ready[0].arrive(cta_group=1)

            # ---- dstate Q-term += dO'^T(T) @ Q ---------------------------------------
            bars.mb_do_prime_input_ready[0].wait(do_prime_input_ready.phase)
            do_prime_input_ready = advance(do_prime_input_ready, 1)
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_scale_acc_done[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)

            desc_q_trans = d_q_trans0
            for i in cutlass.range_constexpr(bmm_do_prime_q_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_do_prime_q_desc.sps_B):
                    mma_ts_step(
                        bmm_do_prime_q_desc,
                        do_prime_input_ptr.subview(i * bmm_do_prime_q_desc.sps_B * bmm_do_prime_q_desc.tmem_advance_A),
                        desc_q_trans + i * (bmm_do_prime_q_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True) if cutlass.const_expr(i + k > 0) else have_dstate,
                    )

            # ---- U^T recompute = Y^T(T) @ T^T ----------------------------------------
            tinv_idx = tinv_index.idx
            bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_y_ready[0].wait(y_ready_index.phase)
            y_ready_index = advance(y_ready_index, 1)
            v_idx = v_index.idx
            v_index = advance(v_index, cfg.smem_v_stages)

            desc_tinv = d_tinv0
            for i in cutlass.range_constexpr(bmm_y_t_inv_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_y_t_inv_desc.sps_B):
                    mma_ts_step(
                        bmm_y_t_inv_desc,
                        y_input_ptr.subview(i * bmm_y_t_inv_desc.sps_B * bmm_y_t_inv_desc.tmem_advance_A),
                        desc_tinv + i * (bmm_y_t_inv_desc.smem_subtile_B >> 4),
                        u_acc_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_u_acc_ready[0].arrive(cta_group=1)

            # ---- dY = dU^T(T) @ T ----------------------------------------------------
            bars.mb_du_input_ready[0].wait(du_input_ready.phase)
            du_input_ready = advance(du_input_ready, 1)

            desc_tinv_trans = d_tinv_trans0
            for i in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.sps_B):
                    mma_ts_step(
                        bmm_du_t_inv_trans_desc,
                        du_input_ptr.subview(i * bmm_du_t_inv_trans_desc.sps_B * bmm_du_t_inv_trans_desc.tmem_advance_A),
                        desc_tinv_trans + i * (bmm_du_t_inv_trans_desc.smem_subtile_B >> 4),
                        dy_acc_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready[0].arrive(cta_group=1)

            # ---- dK inter = dstate entry(S) @ U^T ------------------------------------
            bars.mb_u_ready[0].wait(u_index.phase)
            u_index = advance(u_index, 1)
            if have_dstate:
                bars.mb_dstate_smem_ready[0].wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)

                desc_dstate = d_dstate_trans0
                desc_u_dk_inter = d_v0
                mma_ss(
                    bmm_dstate_u_desc,
                    desc_dstate,
                    desc_u_dk_inter,
                    dvdk_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_scale_acc_ready[0].arrive(cta_group=1)

            # ---- dA eff = dO(S) @ U^T ------------------------------------------------
            desc_do_da = d_do0
            desc_u_da = d_v0
            mma_ss(
                bmm_do_u_desc,
                desc_do_da,
                desc_u_da,
                da_acc_ptr,
                accumulate=False,
            )
            if elect_one:
                bars.mb_da_acc_ready[0].arrive(cta_group=1)
                bars.mb_do_mma_done[do_idx].arrive(cta_group=1)

            # ---- dM core = dY(S) @ U^T -----------------------------------------------
            dv_ready_idx = dv_ready_index.idx
            bars.mb_dv_tmastg_ready[dv_ready_idx].wait(dv_ready_index.phase)
            dv_ready_index = advance(dv_ready_index, cfg.smem_dv_stages)

            desc_dy_dm = d_dv0
            desc_u_dm = d_v0
            mma_ss(
                bmm_dy_u_desc,
                desc_dy_dm,
                desc_u_dm,
                dm_core_acc_ptr,
                accumulate=False,
            )
            if elect_one:
                bars.mb_dm_acc_ready[0].arrive(cta_group=1)
                bars.mb_v_mma_done[v_idx].arrive(cta_group=1)

            # ---- dstate K-term += dY'^T(T) @ K ---------------------------------------
            bars.mb_dyp_input_ready[0].wait(dyp_input_ready.phase)
            dyp_input_ready = advance(dyp_input_ready, 1)

            desc_k_trans = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            for i in cutlass.range_constexpr(bmm_dy_prime_k_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dy_prime_k_desc.sps_B):
                    mma_ts_step(
                        bmm_dy_prime_k_desc,
                        dyp_input_ptr.subview(i * bmm_dy_prime_k_desc.sps_B * bmm_dy_prime_k_desc.tmem_advance_A),
                        desc_k_trans + i * (bmm_dy_prime_k_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready[dstate_idx].arrive(cta_group=1)

            # ---- dK attn += Q^T(S) @ dA ----------------------------------------------
            if have_dstate:
                bars.mb_dk_scale_acc_done[0].wait(dk_scale_index.phase)
                dk_scale_index = advance(dk_scale_index, 1)
            bars.mb_da_ready[0].wait(da_ready_index.phase)
            da_ready_index = advance(da_ready_index, 1)

            desc_q_trans_dk_attn = d_q_trans0
            desc_da_trans = d_da_trans0
            mma_ss(
                bmm_q_da_trans_desc,
                desc_q_trans_dk_attn,
                desc_da_trans,
                dvdk_acc_ptr,
                accumulate=have_dstate,
            )
            if elect_one:
                bars.mb_dk_attn_acc_ready[0].arrive(cta_group=1)
                bars.mb_q_mma_done[q_idx].arrive(cta_group=1)

            # ---- dQ attn += K^T(S) @ dA ----------------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dq_acc_scale_done[0].wait(dq_scale_index.phase)
                dq_scale_index = advance(dq_scale_index, 1)
            if chunk_idx < FIRST_STATE_CHUNK:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
            desc_k_trans_dq_attn = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            desc_da = d_da0
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_da_desc,
                    desc_k_trans_dq_attn,
                    desc_da,
                    dq_acc_ptr,
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_da_desc,
                    desc_k_trans_dq_attn,
                    desc_da,
                    dq_acc_ptr,
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dq_acc_total_ready[0].arrive(cta_group=1)
                bars.mb_a_done[du_a_idx].arrive(cta_group=1)

            # ---- dK state-path = state(S) @ dY^T -------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                desc_state_spath = d_state_trans0
                desc_dy_spath = d_dv0
                mma_ss(
                    bmm_state_dy_desc,
                    desc_state_spath,
                    desc_dy_spath,
                    dk_state_path_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_state_path_acc_ready[0].arrive(cta_group=1)
                    bars.mb_state_mma_done[state_idx].arrive(cta_group=1)
            if elect_one:
                bars.mb_sdv_done[0].arrive(cta_group=1)

            # ---- dK dM-terms += K^T(S) @ dM^T + K^T(S) @ dM --------------------------
            bars.mb_dm_acc_done[0].wait(dm_index.phase)
            dm_index = advance(dm_index, 1)
            bars.mb_dk_attn_acc_done[0].wait(dk_attn_index.phase)
            dk_attn_index = advance(dk_attn_index, 1)
            desc_k_trans_dm_terms = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            desc_dm_trans = d_dm_trans0
            mma_ss(
                bmm_k_dm_trans_desc,
                desc_k_trans_dm_terms,
                desc_dm_trans,
                dvdk_acc_ptr,
                accumulate=True,
            )
            desc_dm = d_dm0
            mma_ss(
                bmm_k_dm_desc,
                desc_k_trans_dm_terms,
                desc_dm,
                dvdk_acc_ptr,
                accumulate=True,
            )
            if elect_one:
                bars.mb_dk_total_acc_ready[0].arrive(cta_group=1)
                bars.mb_k_mma_done[k_idx].arrive(cta_group=1)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)

    bars.mb_dk_total_acc_done[0].wait(dk_total_index.phase)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_col, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def tmaldg_warp(
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
    sState_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_do_base,
    desc_checkpoint_base,
    sScheduler,
    mScheduler,
    bars,
):
    """TMA-LDG warp role (warp 9): persistent tile loop issuing every
    Q/K/V/dO/state TMA load."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    do_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)
    tail_count = ((total_tiles - cutlass.Int32(1)) % num_ctas) + cutlass.Int32(1)
    tail_base = (total_tiles - tail_count) if tail_count * 2 >= num_ctas else total_tiles
    tail_row = tail_base + cute.arch.smid()
    tail_row = tail_row if tail_row < total_tiles else cutlass.Int32(1 << 28)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    bpe = cfg.io_dtype.width // 8
    granule_elements = 128 // bpe
    bt = cfg.b_t
    q_stage_elements = cfg.q_cosize // cfg.smem_q_stages
    k_stage_elements = cfg.k_cosize // cfg.smem_k_stages
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=q_stage_elements,
        stages=cfg.smem_q_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elements,
        tma_subtile_stride_elems=bt * granule_elements,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=k_stage_elements,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elements,
        tma_subtile_stride_elems=bt * granule_elements,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=0,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elements,
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
        tma_granu_elems=granule_elements,
        tma_subtile_stride_elems=4096,
    )
    sCheckpoint_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages),
        stages=cfg.smem_state_stages,
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
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_checkpoint_slot)

        for rev_idx in cutlass.range(compute_end - write_start):
            chunk_idx = compute_end - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            # ---- K load --------------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_mma_done[k_idx].wait(k_index.phase)
            bars.mb_k_cg2_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if elect_one:
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr, acquire=False)

            # ---- Q load --------------------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_mma_done[q_idx].wait(q_index.phase)
            bars.mb_q_cg1_done[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            if elect_one:
                bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, tok_coord)
            tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr, acquire=False)

            # ---- V load --------------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_mma_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, tok_coord)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

            # ---- dO load -------------------------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_mma_done[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if elect_one:
                bars.mb_do_ready[do_idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, tok_coord)
            tma_load_tile(sdO_tma[do_idx], do_slice, bars.mb_do_ready[do_idx].smem_ptr, acquire=False)

            # ---- entering state ------------------------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                state_idx = state_index.idx
                bars.mb_state_mma_done[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                if elect_one:
                    bars.mb_state_ready[state_idx].arrive(n_bytes=cfg.tma_state_bytes)
                checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx, head_o)
                tma_load_tile(sCheckpoint_tma[state_idx], checkpoint_slice, bars.mb_state_ready[state_idx].smem_ptr, acquire=False)

        next_tile, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, tail_base, tail_row, elect_one
        )
        tile_idx = next_tile

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_mma_done[q_index.idx].wait(q_index.phase)
        bars.mb_q_cg1_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_mma_done[k_index.idx].wait(k_index.phase)
        bars.mb_k_cg2_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_mma_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    for _ in range(cfg.smem_do_stages):
        bars.mb_do_mma_done[do_index.idx].wait(do_index.phase)
        do_index = advance(do_index, cfg.smem_do_stages)


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    tmem_base_slot,
    scale,
    sCumsumlog,
    sCumprod,
    sBeta,
    sTinv,
    sKK,
    sA,
    sDa,
    sDm,
    sstate_flat,
    sdstate_flat,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): persistent scheduler loop
    building each chunk's blockwise-inverse T and attention matrices."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=1)
    tinv_index = PipelineState.start(phase=1)
    da_acc_index = PipelineState.start(phase=0)
    dm_ready_index = PipelineState.start(phase=0)
    cg0_dbeta_index = PipelineState.start(phase=0)
    cg0_kk_ready = PipelineState.start(phase=0)
    cg0_a_ready = PipelineState.start(phase=0)
    cg0_dstate_smem_ready = PipelineState.start(phase=0)
    cg0_state_ready_index = PipelineState.start(phase=0)

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_idx = cg0_tidx % cfg.threads_per_warp
    store_row = warp_id * 16 + lane_idx % 16
    store_col = (lane_idx // 16) * 8

    bpe = cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    ACC_STAGE_COLS = cfg.b_t
    acc_zero = cfg.acc_dtype(0.0)
    mask_zero = opaque_f32_zero()
    DSTATE_IN0 = 1 if cfg.use_dstate_in else 0

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_do_prime_col = tmem_shared_input_col
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_state_path_col = tmem_shared_input_col
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset

    skk_red = cute.make_ptr(cutlass.Float32, sKK[0].base.toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    tinv_zero_ptr = cute.make_ptr(cutlass.Int32, sTinv[0].base.toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    for z in cutlass.range_constexpr(cfg.t_inv_cosize * bpe // 4 // num_threads_cg0):
        (tinv_zero_ptr + cg0_tidx + z * num_threads_cg0).store(cutlass.Int32(0))

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        for chunk_idx in cutlass.range(num_item_chunks):
            # ---- T-pairwise ----------------------------------------------------------
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            row_cumsums = []
            for r in cutlass.range_constexpr(2):
                row_cumsums.append(sCumsumlog[warp_id * 16 + lane_idx // 4 + r * 8, 0, gate_idx])
            col_cumsums = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    col_cumsums.append(sCumsumlog[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
            decay_t = []
            decay_t_strict = []
            for k in cutlass.range_constexpr(num_vals):
                chunk_row = warp_id * 16 + lane_idx // 4 + ((k // 2) % 2) * 8
                chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
                decay_t.append(
                    cute.math.exp2(row_cumsums[(k // 2) % 2] - col_cumsums[(k // 4) * 2 + (k % 2)], fastmath=True) if chunk_row >= chunk_col else mask_zero
                )
                decay_t_strict.append(mask_zero if chunk_row == chunk_col else decay_t[k])
            cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

            beta_idx = beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)

            gBeta = []
            for k in cutlass.range_constexpr(num_vals):
                chunk_row = warp_id * 16 + lane_idx // 4 + ((k // 2) % 2) * 8
                gBeta.append(sBeta[chunk_row, 0, beta_idx])

            # ---- KK epilogue: M kk[i,j] = W kk[i,j] * T strict[i,j] * Beta[i] --------
            tinv_idx = tinv_index.idx
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_kk_acc_ready[0].wait(cg0_kk_ready.phase)
            cg0_kk_ready = advance(cg0_kk_ready, 1)

            tinv_base = sTinv[tinv_idx].base
            kk_base = sKK[tinv_idx].base
            kk_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + tmem_kk_col, cutlass.Float32), num=8)
            kk_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(kk_vec[2 * k], kk_vec[2 * k + 1], decay_t_strict[2 * k], decay_t_strict[2 * k + 1])
                v0, v1 = fmul2(p0, p1, gBeta[2 * k], gBeta[2 * k + 1])
                kk_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    kk_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [kk_pack[c * 4 + 0], kk_pack[c * 4 + 1], kk_pack[c * 4 + 2], kk_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            if chunk_idx < compute_end - FIRST_STATE_CHUNK:
                bars.mb_kk_acc_done[0].arrive()

            # ---- A epilogue: A[i,j] = W qk[i,j] * T[i,j] * scale ---------------------
            a_idx = a_index.idx
            a_phase = a_index.phase
            a_index = advance(a_index, cfg.smem_a_stages)
            bars.mb_a_acc_ready[0].wait(cg0_a_ready.phase)
            cg0_a_ready = advance(cg0_a_ready, 1)

            a_base = sA[a_idx].base
            a_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + tmem_a_col, cutlass.Float32), num=8)
            a_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(a_vec[2 * k], a_vec[2 * k + 1], decay_t[2 * k], decay_t[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                a_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            bars.mb_a_done[a_idx].wait(a_phase)
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    a_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [a_pack[c * 4 + 0], a_pack[c * 4 + 1], a_pack[c * 4 + 2], a_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[a_idx].arrive()

            # ---- blockwise inverse ---------------------------------------------------
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )
            if warp_id < 2:
                invert_diagonal_NxN(cfg, kk_base, tinv_base, cg0_tidx // 8, cg0_tidx, 8)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            blockwise_diagonal_8x8_to_16x16(cfg, tinv_base, kk_base, warp_id * 16, lane_idx)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                blockwise_diagonal_16x16_to_32x32(cfg, tinv_base, kk_base, warp_id * 32, lane_idx)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                blockwise_diagonal_32x32_to_64x64(cfg, tinv_base, kk_base, warp_id, lane_idx)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # ---- Beta column-scaling in place: T^-1[i,j] *= Beta[j] ------------------
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta_idx])
            tinv_frags = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                tinv_frags += list(
                    nvvm.ldmatrix(
                        tinv_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                )
            tinv_pack = []
            for j in cutlass.range_constexpr(num_vals // 2):
                lo, hi = f16x2_to_f32(tinv_frags[j], dtype=cfg.io_dtype)
                s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
                tinv_pack.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    tinv_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [tinv_pack[c * 4 + 0], tinv_pack[c * 4 + 1], tinv_pack[c * 4 + 2], tinv_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )

            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_t_inv_ready[tinv_idx].arrive()

            # ---- dGate last state dot dstate term ------------------------------------
            if chunk_idx + DSTATE_IN0 >= 1:
                bars.mb_dstate_smem_ready[0].wait(cg0_dstate_smem_ready.phase)
                cg0_dstate_smem_ready = advance(cg0_dstate_smem_ready, 1)
            if chunk_idx < compute_end - FIRST_STATE_CHUNK:
                bars.mb_state_ready[cg0_state_ready_index.idx].wait(cg0_state_ready_index.phase)
                cg0_state_ready_index = advance(cg0_state_ready_index, cfg.smem_state_stages)
            sdstate_base = sdstate_flat.iterator.toint()
            sstate_base = sstate_flat.iterator.toint()
            state_dot_dstate_lo = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
            state_dot_dstate_hi = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
            for oct_ in cutlass.range_constexpr(cfg.d_v // 8):
                state_dot_off = (oct_ // 8) * (cfg.d_k * 64) + cg0_tidx * 64 + swizzle_xor_128b(cg0_tidx, (oct_ % 8) * 8)
                dstate_frag = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, sdstate_base + state_dot_off * bpe, mem_space=cute.AddressSpace.smem, assumed_align=16),
                    cute.make_layout(4),
                ).load()
                state_frag = cute.make_tensor(
                    cute.make_ptr(cutlass.Int32, sstate_base + state_dot_off * bpe, mem_space=cute.AddressSpace.smem, assumed_align=16),
                    cute.make_layout(4),
                ).load()
                for w in cutlass.range_constexpr(4):
                    dstate_lo, dstate_hi = f16x2_to_f32(dstate_frag[w], dtype=cfg.io_dtype)
                    state_lo, state_hi = f16x2_to_f32(state_frag[w], dtype=cfg.io_dtype)
                    state_dot_dstate_lo[w], state_dot_dstate_hi[w] = ffma2(
                        dstate_lo, dstate_hi, state_lo, state_hi, state_dot_dstate_lo[w], state_dot_dstate_hi[w]
                    )
            state_dot_dstate = ((state_dot_dstate_lo[0] + state_dot_dstate_lo[1]) + (state_dot_dstate_lo[2] + state_dot_dstate_lo[3])) + (
                (state_dot_dstate_hi[0] + state_dot_dstate_hi[1]) + (state_dot_dstate_hi[2] + state_dot_dstate_hi[3])
            )
            for off in [1, 2, 4, 8, 16]:
                state_dot_dstate += nvvm.shfl_sync(0xFFFFFFFF, state_dot_dstate, off, 31, kind=nvvm.Shfl.BFLY)
            hdh_last_w = cumprod_total * state_dot_dstate if chunk_idx + DSTATE_IN0 >= 1 and chunk_idx < compute_end - FIRST_STATE_CHUNK else acc_zero
            bars.mb_state_dot_dstate_done[0].arrive()

            # ---- dA epilogue ---------------------------------------------------------
            bars.mb_da_acc_ready[0].wait(da_acc_index.phase)
            da_acc_index = advance(da_acc_index, 1)

            da_base = sDa[0].base
            da_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + tmem_da_col, cutlass.Float32), num=8)
            da_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(da_vec[2 * k], da_vec[2 * k + 1], decay_t[2 * k], decay_t[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                da_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    da_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [da_pack[c * 4 + 0], da_pack[c * 4 + 1], da_pack[c * 4 + 2], da_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[0].arrive()

            # ---- dM epilogue ---------------------------------------------------------
            bars.mb_dm_acc_ready[0].wait(dm_ready_index.phase)
            dm_ready_index = advance(dm_ready_index, 1)
            dm_base = sDm[0].base
            dm_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dm_core_col, cutlass.Float32), num=8)

            dm_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(dm_vec[2 * k], dm_vec[2 * k + 1], decay_t_strict[2 * k], decay_t_strict[2 * k + 1])
                v0 = cutlass.Float32(0.0) - p0
                v1 = cutlass.Float32(0.0) - p1
                dm_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    dm_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [dm_pack[c * 4 + 0], dm_pack[c * 4 + 1], dm_pack[c * 4 + 2], dm_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_acc_done[0].arrive()

            # ---- dBeta/dGate M-terms: E = dM core ⊙ M kk(sKK, strict-masked) ---------
            kk_frag = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                kk_frag += list(
                    nvvm.ldmatrix(
                        kk_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                )
            binv_row = [
                cute.math.rcp(gBeta[0] + cutlass.Float32(1e-10), approx=True, ftz=True),
                cute.math.rcp(gBeta[2] + cutlass.Float32(1e-10), approx=True, ftz=True),
            ]
            row_acc = [None] * 8
            col_part = [None] * 16
            for j in cutlass.range_constexpr(num_vals // 2):
                k_lo, k_hi = f16x2_to_f32(kk_frag[j], dtype=cfg.io_dtype)
                binv_j = binv_row[j % 2]
                lo, hi = fmul2(dm_vec[2 * j], dm_vec[2 * j + 1], k_lo, k_hi)
                lo, hi = fmul2(lo, hi, binv_j, binv_j)
                r = cutlass.const_expr((j % 2) * 4 + (j // 2) % 4)
                if cutlass.const_expr(j < 8):
                    row_acc[r] = lo + hi
                else:
                    row_acc[r] += lo + hi
                c0 = cutlass.const_expr((j // 2) * 2)
                if cutlass.const_expr(j % 2 == 0):
                    col_part[c0], col_part[c0 + 1] = lo, hi
                else:
                    col_part[c0], col_part[c0 + 1] = fadd2(col_part[c0], col_part[c0 + 1], lo, hi)
            row_part = [
                (row_acc[0] + row_acc[1]) + (row_acc[2] + row_acc[3]),
                (row_acc[4] + row_acc[5]) + (row_acc[6] + row_acc[7]),
            ]
            for off in [1, 2]:
                for rp in cutlass.range_constexpr(2):
                    row_part[rp] += nvvm.shfl_sync(0xFFFFFFFF, row_part[rp], off, 31, kind=nvvm.Shfl.BFLY)
            # ---- dGate column fold ---------------------------------------------------
            nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
            col_lo, col_hi = warp_reduce_scatter_sum_16_elements(col_part, lane_idx)
            tok0 = (lane_idx // 4) * 8 + (lane_idx % 4) * 2
            (skk_red + warp_id * 64 + tok0).store(col_lo)
            (skk_red + warp_id * 64 + tok0 + 1).store(col_hi + hdh_last_w if lane_idx == 31 else col_hi)

            if lane_idx % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow_r = warp_id * 16 + lane_idx // 4 + rp * 8
                    sCumsumlog[crow_r, 0, gate_idx] = acc_zero - row_part[rp]
            nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
            if cg0_tidx < 64:
                col_sum = (skk_red + cg0_tidx).load() + (skk_red + 64 + cg0_tidx).load() + (skk_red + 128 + cg0_tidx).load() + (skk_red + 192 + cg0_tidx).load()
                sCumsumlog[cg0_tidx, 0, gate_idx] = sCumsumlog[cg0_tidx, 0, gate_idx] + col_sum
            bars.mb_dgate_cg0_ready[0].arrive()
            bars.mb_gate_done[gate_idx].arrive()

            # ---- dBeta M-term --------------------------------------------------------
            bars.mb_dbeta_cg1_ready[0].wait(cg0_dbeta_index.phase)
            cg0_dbeta_index = advance(cg0_dbeta_index, 1)
            if lane_idx % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow_r = warp_id * 16 + lane_idx // 4 + rp * 8
                    db = sBeta[crow_r, 0, beta_idx] - row_part[rp] * binv_row[rp]
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        b = gBeta[2 * rp]
                        db = db * (b - b * b)
                    sBeta[crow_r, 0, beta_idx] = db

            bars.mb_beta_done[beta_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)
    for _ in range(cfg.smem_a_stages):
        bars.mb_a_done[a_index.idx].wait(a_index.phase)
        a_index = advance(a_index, cfg.smem_a_stages)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    warp_idx,
    tmem_base_slot,
    scale,
    sQ,
    sV,
    sdO,
    sCumsumlog,
    sCumprod,
    sBeta,
    sdQ,
    sdV,
    sdK,
    sDa,
    sDm,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop
    running each chunk's gradient epilogues and stagings."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    cg1_k_state_ready = PipelineState.start(phase=0)
    cg1_u_ready = PipelineState.start(phase=0)
    cg1_dq_scale_ready = PipelineState.start(phase=0)
    cg1_dstate_smem_index = PipelineState.start(phase=0)
    cg1_dstate_scale_index = PipelineState.start(phase=0)
    cg1_dy_ready = PipelineState.start(phase=0)
    cg1_du_scale_ready = PipelineState.start(phase=0)
    cg1_du_total_ready = PipelineState.start(phase=0)
    cg1_dgate_cg0_index = PipelineState.start(phase=0)
    cg1_cg2_chunk_index = PipelineState.start(phase=0)
    dq_index = PipelineState.start(phase=1)
    cg1_beta_index = PipelineState.start(phase=0)
    sdv_done_index = PipelineState.start(phase=1)
    dq_total_ready_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = cg1_tidx % cfg.threads_per_warp

    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    ACC_STAGE_COLS = cfg.b_t
    frag_row = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    frag_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
    frag_segment = (cg1_tidx // 64) * 4096
    dv_stage_elements = cfg.dv_cosize // cfg.smem_dv_stages
    sdV_base = sdV[0].base
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    sV_base = sV[0].base
    sQ_base = sQ[0].base
    do_stage_elements = cfg.do_cosize // cfg.smem_do_stages
    sdO_base = sdO[0].base
    dq_stage_elements = cfg.dq_cosize // cfg.smem_dq_stages
    sdQ_base = sdQ[0].base
    cg1_warp_id = cg1_tidx // cfg.threads_per_warp
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset
    tmem_dvdk_acc_col = tmem_col + cfg.tmem_dvdk_acc_offset
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_do_prime_col = tmem_shared_input_col
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_y_col = tmem_col + cfg.tmem_y_offset
    tmem_g_k_state_col = tmem_y_col + SHARED_INP_STAGE_COLS
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b

    sred_base = cute.make_ptr(cutlass.Float32, sdQ[0].base.toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sdk_red = cute.make_ptr(cutlass.Float32, sdK[0].base.toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    PART_Q_SCRATCH_OFF = 4 * 64
    tmem_dstate_acc_col = tmem_col + cfg.tmem_dstate_acc_offset
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.d_v // ldtm_width)
    cg1_dk_index = PipelineState.start(phase=1)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- chunks NT-1 .. 0 (backward) ---------------------------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            chunk_idx = compute_end - 1 - rev_idx
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            beta_idx = cg1_beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(cg1_beta_index.phase)
            cg1_beta_index = advance(cg1_beta_index, cfg.smem_beta_stages)

            num_vals = 32
            cumprod_fp32 = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    cumprod_fp32.append(sCumprod[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
            cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]

            # ---- dstate rescale: dstate *= this chunk's cumprod ----------------------
            if have_dstate:
                bars.mb_dstate_smem_ready[0].wait(cg1_dstate_smem_index.phase)
                cg1_dstate_smem_index = advance(cg1_dstate_smem_index, 1)
                cumprod_top = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_rescale_vec = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32), num=32
                    )
                    dstate_rescaled = []
                    for j in cutlass.range_constexpr(16):
                        h0, h1 = fmul2(dstate_rescale_vec[2 * j], dstate_rescale_vec[2 * j + 1], cumprod_top, cumprod_top)
                        dstate_rescaled += [h0, h1]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(dstate_rescaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_scale_acc_done[cg1_dstate_scale_index.idx].arrive()
                cg1_dstate_scale_index = advance(cg1_dstate_scale_index, cfg.tmem_dstate_acc_stages)

            # ---- V fragment preload --------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_ready[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            v_frag = []
            for half in cutlass.range_constexpr(2):
                v_words = []
                for block in cutlass.range_constexpr(4):
                    v_raw = nvvm.ldmatrix(
                        (
                            sV_base
                            + v_idx * v_stage_elements
                            + frag_segment
                            + (frag_row + block * 16) * 64
                            + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                        ),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        v_words.append(v_raw[i])
                v_frag.append(v_words)

            # ---- dO fragment preload -------------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            do_raw_frag = []
            for half in cutlass.range_constexpr(2):
                do_words_raw = []
                for block in cutlass.range_constexpr(4):
                    do_raw = nvvm.ldmatrix(
                        (
                            sdO_base
                            + do_idx * do_stage_elements
                            + frag_segment
                            + (frag_row + block * 16) * 64
                            + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                        ),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        do_words_raw.append(do_raw[i])
                do_raw_frag.append(do_words_raw)

            # ---- Y stage: Y = V - cumprod * k state -> TMEM f16 ----------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_k_state_acc_ready[0].wait(cg1_k_state_ready.phase)
                cg1_k_state_ready = advance(cg1_k_state_ready, 1)
                for half in cutlass.range_constexpr(2):
                    k_state_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_k_state_col, cutlass.Float32),
                        num=8,
                    )
                    g_k_state_pack = []
                    y_pack = []
                    for j in cutlass.range_constexpr(16):
                        g0, g1 = fmul2(k_state_vec[2 * j], k_state_vec[2 * j + 1], cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                        g_k_state_word = fp32_to_fp16(g0, g1, dtype=cfg.io_dtype)
                        g_k_state_pack.append(g_k_state_word)
                        y_pack.append(sub_f16x2(v_frag[half][j], g_k_state_word, cfg.io_dtype))
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_y_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(y_pack), cutlass.Int32),
                    )
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_g_k_state_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(g_k_state_pack), cutlass.Int32),
                    )
            if chunk_idx < FIRST_STATE_CHUNK:
                for half in cutlass.range_constexpr(2):
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_y_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(v_frag[half]), cutlass.Int32),
                    )
            nvvm.tcgen05_wait("store")
            bars.mb_y_ready[0].arrive()

            # ---- dV inter ------------------------------------------------------------
            if have_dstate:
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                col_cumsumlog_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        col_cumsumlog_fp32.append(sCumsumlog[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
                decay_scale_fp32 = []
                for i in cutlass.range_constexpr(16):
                    decay_scale_fp32.append(cute.math.exp2(last_cumsumlog - col_cumsumlog_fp32[i], fastmath=True))
                decay_scale_vals = [decay_scale_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
                bars.mb_du_scale_acc_ready[0].wait(cg1_du_scale_ready.phase)
                cg1_du_scale_ready = advance(cg1_du_scale_ready, 1)
                dv_ptrs = []
                dv_vecs = []
                for half in cutlass.range_constexpr(2):
                    dv_ptrs.append(nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32))
                    dv_vecs.append(nvvm.tcgen05_ld("16x256b", dv_ptrs[half], num=8))
                for half in cutlass.range_constexpr(2):
                    dv_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(dv_vecs[half][2 * j], dv_vecs[half][2 * j + 1], decay_scale_vals[2 * j], decay_scale_vals[2 * j + 1])
                        dv_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dv_ptrs[half], cutlass.Vector.from_elements(tuple(dv_scaled), cutlass.Float32))
                nvvm.tcgen05_wait("store")
                bars.mb_du_scale_acc_done[0].arrive()

            # ---- dO' stage: dO * cumprod * scale -> shared input TMEM ----------------
            bars.mb_cg2_chunk_ready[0].wait(cg1_cg2_chunk_index.phase)
            cg1_cg2_chunk_index = advance(cg1_cg2_chunk_index, 1)
            for half in cutlass.range_constexpr(2):
                do_pack = []
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(do_raw_frag[half][j], dtype=cfg.io_dtype)
                    p0, p1 = fmul2(lo, hi, cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                    q0, q1 = fmul2(p0, p1, scale, scale)
                    do_pack.append(fp32_to_fp16(q0, q1, dtype=cfg.io_dtype))
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_do_prime_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(do_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_do_prime_input_ready[0].arrive()

            # ---- dU stage: dV acc -> TMEM f16 A operand ------------------------------
            bars.mb_du_total_acc_ready[0].wait(cg1_du_total_ready.phase)
            cg1_du_total_ready = advance(cg1_du_total_ready, 1)
            for half in cutlass.range_constexpr(2):
                du_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32), num=8)
                du_pack = [fp32_to_fp16(du_vec[2 * j], du_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_du_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(du_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_du_input_ready[0].arrive()

            # ---- dQ inter rescale ----------------------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dq_acc_scale_ready[0].wait(cg1_dq_scale_ready.phase)
                cg1_dq_scale_ready = advance(cg1_dq_scale_ready, 1)
                dqi_ptrs = []
                dqi_vecs = []
                for half in cutlass.range_constexpr(2):
                    dqi_ptrs.append(nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dstate_input_col, cutlass.Float32))
                    dqi_vecs.append(nvvm.tcgen05_ld("16x256b", dqi_ptrs[half], num=8))
                for half in cutlass.range_constexpr(2):
                    dqi_scaled = []
                    for j in cutlass.range_constexpr(16):
                        p0, p1 = fmul2(dqi_vecs[half][2 * j], dqi_vecs[half][2 * j + 1], cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                        s0, s1 = fmul2(p0, p1, scale, scale)
                        dqi_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dqi_ptrs[half], cutlass.Vector.from_elements(tuple(dqi_scaled), cutlass.Float32))
                nvvm.tcgen05_wait("store")
                bars.mb_dq_acc_scale_done[0].arrive()

            # ---- U read: TMEM -> sU --------------------------------------------------
            bars.mb_u_acc_ready[0].wait(cg1_u_ready.phase)
            cg1_u_ready = advance(cg1_u_ready, 1)
            u_regs = []
            for half in cutlass.range_constexpr(2):
                u_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_u_col, cutlass.Float32), num=8)
                u_regs.append([u_vec[k] for k in range(32)])
            for half in cutlass.range_constexpr(2):
                for block in cutlass.range_constexpr(4):
                    u_pack = [fp32_to_fp16(u_regs[half][8 * block + 2 * j], u_regs[half][8 * block + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix(
                        (
                            sV_base
                            + v_idx * v_stage_elements
                            + frag_segment
                            + (frag_row + block * 16) * 64
                            + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                        ),
                        u_pack,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_ready[0].arrive()

            # ---- dY ------------------------------------------------------------------
            bars.mb_dy_acc_ready[0].wait(cg1_dy_ready.phase)
            cg1_dy_ready = advance(cg1_dy_ready, 1)
            dy_regs = []
            for half in cutlass.range_constexpr(2):
                dy_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dy_col, cutlass.Float32),
                    num=8,
                )
                dy_regs.append([dy_vec[k] for k in range(32)])

            # ---- dY' = -cumprod * dY -> f16 shared input -----------------------------
            neg_one = cutlass.Float32(-1.0)
            cumprod_neg_vals = [cumprod_vals[k] * neg_one for k in range(32)]
            for half in cutlass.range_constexpr(2):
                dyp = []
                for j in cutlass.range_constexpr(16):
                    n0, n1 = fmul2(dy_regs[half][2 * j], dy_regs[half][2 * j + 1], cumprod_neg_vals[2 * j], cumprod_neg_vals[2 * j + 1])
                    dyp += [n0, n1]
                dyp_pack = [fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dyp_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(dyp_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dyp_input_ready[0].arrive()

            # ---- dV stage: dV = dY -> sdV --------------------------------------------
            dv_stg_idx = dv_index.idx
            bars.mb_dv_tmastg_done[dv_stg_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            bars.mb_sdv_done[0].wait(sdv_done_index.phase)
            sdv_done_index = advance(sdv_done_index, 1)
            for half in cutlass.range_constexpr(2):
                for block in cutlass.range_constexpr(4):
                    dv_pack = [fp32_to_fp16(dy_regs[half][8 * block + 2 * j], dy_regs[half][8 * block + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix(
                        (
                            sdV_base
                            + dv_stg_idx * dv_stage_elements
                            + frag_segment
                            + (frag_row + block * 16) * 64
                            + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                        ),
                        dv_pack,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stg_idx].arrive()
            dq_stg_idx = dq_index.idx
            bars.mb_dq_tmastg_done[dq_stg_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            sred = sred_base + dq_stg_idx * (dq_stage_elements // 2)

            # ---- dBeta/dGate V-terms: dBeta += rowsum(dV ⊙ Y) / Beta -----------------
            gsum = cutlass.Float32(0.0)
            part_y = [None] * 16
            for half in cutlass.range_constexpr(2):
                y_vec = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_y_col, cutlass.Int32), num=8)
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(y_vec[j], dtype=cfg.io_dtype)
                    frag_e0 = cutlass.const_expr(2 * j)
                    part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                    if cutlass.const_expr(half == 0 and j % 2 == 0):
                        part_y[part_e0], part_y[part_e0 + 1] = fmul2(dy_regs[half][frag_e0], dy_regs[half][frag_e0 + 1], lo, hi)
                    else:
                        part_y[part_e0], part_y[part_e0 + 1] = ffma2(
                            dy_regs[half][frag_e0], dy_regs[half][frag_e0 + 1], lo, hi, part_y[part_e0], part_y[part_e0 + 1]
                        )
            py_lo, py_hi = warp_reduce_scatter_sum_16_elements(part_y, lane_idx)
            vt_tok0 = (lane_idx // 4) * 8 + (lane_idx % 4) * 2
            if chunk_idx >= FIRST_STATE_CHUNK:
                part_g = [None] * 16
                for half in cutlass.range_constexpr(2):
                    g_k_state_vec = nvvm.tcgen05_ld("16x128b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_g_k_state_col, cutlass.Int32), num=8)
                    for j in cutlass.range_constexpr(16):
                        lo, hi = f16x2_to_f32(g_k_state_vec[j], dtype=cfg.io_dtype)
                        frag_e0 = cutlass.const_expr(2 * j)
                        part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                        if cutlass.const_expr(half == 0 and j % 2 == 0):
                            part_g[part_e0], part_g[part_e0 + 1] = fmul2(dy_regs[half][frag_e0], dy_regs[half][frag_e0 + 1], lo, hi)
                        else:
                            part_g[part_e0], part_g[part_e0 + 1] = ffma2(
                                dy_regs[half][frag_e0], dy_regs[half][frag_e0 + 1], lo, hi, part_g[part_e0], part_g[part_e0 + 1]
                            )
                pg_lo, pg_hi = warp_reduce_scatter_sum_16_elements(part_g, lane_idx)
                (sred + cg1_warp_id * 64 + vt_tok0).store(py_lo)
                (sred + cg1_warp_id * 64 + vt_tok0 + 1).store(py_hi)
                (sred + 256 + cg1_warp_id * 64 + vt_tok0).store(pg_lo)
                (sred + 256 + cg1_warp_id * 64 + vt_tok0 + 1).store(pg_hi)
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                    ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                    gsum = (sred + 256 + cg1_tidx).load() + (sred + 320 + cg1_tidx).load() + (sred + 384 + cg1_tidx).load() + (sred + 448 + cg1_tidx).load()
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
            if chunk_idx < FIRST_STATE_CHUNK:
                (sred + cg1_warp_id * 64 + vt_tok0).store(py_lo)
                (sred + cg1_warp_id * 64 + vt_tok0 + 1).store(py_hi)
                nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10), approx=True, ftz=True)
                    ysum = (sred + cg1_tidx).load() + (sred + 64 + cg1_tidx).load() + (sred + 128 + cg1_tidx).load() + (sred + 192 + cg1_tidx).load()
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
            bars.mb_beta_done[beta_idx].arrive()
            bars.mb_dbeta_cg1_ready[0].arrive()
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)

            # ---- Q fragments held in registers ---------------------------------------
            q_frag = []
            for half in cutlass.range_constexpr(2):
                q_words = []
                for block in cutlass.range_constexpr(4):
                    q_raw = nvvm.ldmatrix(
                        sQ_base + frag_segment + (frag_row + block * 16) * 64 + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        q_words.append(q_raw[i])
                q_frag.append(q_words)
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_y_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(q_words), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_q_cg1_done[0].arrive()

            # ---- dQ final read -> sdQ ------------------------------------------------
            bars.mb_dq_acc_total_ready[0].wait(dq_total_ready_index.phase)
            dq_total_ready_index = advance(dq_total_ready_index, 1)
            dq_regs = []
            for half in cutlass.range_constexpr(2):
                dq_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dstate_input_col, cutlass.Float32), num=8)
                dq_regs.append([dq_vec[k] for k in range(32)])
            nvvm.tcgen05_wait("load")
            bars.mb_dq_acc_total_done[0].arrive()
            for half in cutlass.range_constexpr(2):
                for block in cutlass.range_constexpr(4):
                    frag_addr = frag_segment + (frag_row + block * 16) * 64 + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                    dq_pack = [fp32_to_fp16(dq_regs[half][8 * block + 2 * j], dq_regs[half][8 * block + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                    nvvm.stmatrix(sdQ_base + dq_stg_idx * dq_stage_elements + frag_addr, dq_pack, nvvm.MMALayout.COL)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_tmastg_ready[dq_stg_idx].arrive()

            # ---- dQ dot (part q) -----------------------------------------------------
            part_q = [None] * 16
            for half in cutlass.range_constexpr(2):
                for block in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        q_lo, q_hi = f16x2_to_f32(q_frag[half][4 * block + i], dtype=cfg.io_dtype)
                        frag_e0 = cutlass.const_expr(8 * block + 2 * i)
                        part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                        if cutlass.const_expr(half == 0 and i % 2 == 0):
                            part_q[part_e0], part_q[part_e0 + 1] = fmul2(dq_regs[half][frag_e0], dq_regs[half][frag_e0 + 1], q_lo, q_hi)
                        else:
                            part_q[part_e0], part_q[part_e0 + 1] = ffma2(
                                dq_regs[half][frag_e0], dq_regs[half][frag_e0 + 1], q_lo, q_hi, part_q[part_e0], part_q[part_e0 + 1]
                            )
            bars.mb_dk_tmastg_done[cg1_dk_index.idx].wait(cg1_dk_index.phase)
            cg1_dk_index = advance(cg1_dk_index, cfg.smem_dk_stages)
            part_q_lo, part_q_hi = warp_reduce_scatter_sum_16_elements(part_q, lane_idx)
            tok0 = (lane_idx // 4) * 8 + (lane_idx % 4) * 2
            (sdk_red + PART_Q_SCRATCH_OFF + (cg1_tidx // 32) * 64 + tok0).store(part_q_lo)
            (sdk_red + PART_Q_SCRATCH_OFF + (cg1_tidx // 32) * 64 + tok0 + 1).store(part_q_hi)
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
            bars.mb_dgate_cg0_ready[0].wait(cg1_dgate_cg0_index.phase)
            cg1_dgate_cg0_index = advance(cg1_dgate_cg0_index, 1)
            if cg1_tidx < 64:
                pq_sum = (
                    (sdk_red + PART_Q_SCRATCH_OFF + cg1_tidx).load()
                    + (sdk_red + PART_Q_SCRATCH_OFF + 64 + cg1_tidx).load()
                    + (sdk_red + PART_Q_SCRATCH_OFF + 128 + cg1_tidx).load()
                    + (sdk_red + PART_Q_SCRATCH_OFF + 192 + cg1_tidx).load()
                )
                sCumsumlog[cg1_tidx, 0, gate_idx] = sCumsumlog[cg1_tidx, 0, gate_idx] + pq_sum - gsum
            bars.mb_gate_done[gate_idx].arrive()
            bars.mb_dgate_cg1_ready[0].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.smem_dv_stages):
        bars.mb_dv_tmastg_done[dv_index.idx].wait(dv_index.phase)
        dv_index = advance(dv_index, cfg.smem_dv_stages)


@cute.jit
def compute2_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    mWorkItems,
    mDstate0_out,
    mDstate_in,
    tidx,
    tmem_base_slot,
    sCumsumlog,
    sCumprod,
    sK,
    sdK,
    sDstate_trans,
    sScheduler,
    bars,
):
    """Compute warp-group 2 role (warps 8-11): the whole read side of the
    dV/dK accumulator slot.  One chunk walks the slot's last three
    productions in place -- inter rescale, attn read, dM-terms fold -- so the
    dGate K part and the dK store come off the same registers."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    cg2_k_index = PipelineState.start(phase=0)
    cg2_kk_scale_ready = PipelineState.start(phase=0)
    cg2_dk_attn_ready = PipelineState.start(phase=0)
    cg2_dgate_index = PipelineState.start(phase=0)
    cg2_dk_state_path_ready = PipelineState.start(phase=0)
    cg2_dk_total_ready = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0)
    dstate_input_index = PipelineState.start(phase=1)
    cg2_state_dot_dstate_index = PipelineState.start(phase=0)
    cg2_dq_total_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg2 = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    cg2_tidx = tidx % num_threads_cg2
    warp_id = cg2_tidx // cfg.threads_per_warp
    lane_idx = cg2_tidx % cfg.threads_per_warp

    num_vals = 32
    acc_zero = cfg.acc_dtype(0.0)
    neg_one = cutlass.Float32(-1.0)
    frag_row = cg2_tidx % 8 + (cg2_tidx // 16 % 2) * 8
    frag_col = (cg2_tidx // 8 % 2) * 8 + (cg2_tidx // 32 % 2) * 32
    frag_segment = (cg2_tidx // 64) * 4096
    sK_base = sK[0].base
    k_stage_elements_cg2 = cfg.k_cosize // cfg.smem_k_stages
    sdK_base = sdK[0].base
    dk_stage_elements = cfg.dk_cosize // cfg.smem_dk_stages
    ldtm_width = 32
    sttm_width = 16
    num_ldtms = cutlass.const_expr(cfg.d_v // ldtm_width)
    dstate_done_idx = cutlass.Int32(0)
    sDstate_base = sDstate_trans[0].base
    DSTATE_IN0 = 1 if cfg.use_dstate_in else 0
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    tmem_dvdk_acc_col = tmem_col + cfg.tmem_dvdk_acc_offset
    tmem_dk_state_path_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_dstate_acc_col = tmem_col + cfg.tmem_dstate_acc_offset
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset

    # dGate fold scratch in the dK staging tile
    sdk_red = cute.make_ptr(cutlass.Float32, sdK[0].base.toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- dstate seed prologue ----------------------------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if num_item_chunks > 0:
                gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                seed_from_dstate_in = compute_end == batch_num_chunks
                dstate_input_idx = dstate_input_index.idx
                bars.mb_dstate_input_done[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_in_vals = []
                    for kk in cutlass.range_constexpr(ldtm_width):
                        v = gDstate_in[cg2_tidx, i * ldtm_width + kk]
                        v = v if seed_from_dstate_in else cutlass.Float32(0.0)
                        dstate_in_vals.append(v)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(dstate_in_vals), cutlass.Float32),
                    )
                    dstate_in_pack = [fp32_to_fp16(dstate_in_vals[2 * j], dstate_in_vals[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_input_col + i * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dstate_in_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready[dstate_input_idx].arrive()

                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_smem_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(ldtm_width // 8):
                        dstate_smem_pack = tuple(
                            fp32_to_fp16(dstate_smem_vec[g * 8 + 2 * t], dstate_smem_vec[g * 8 + 2 * t + 1], dtype=cfg.io_dtype) for t in range(4)
                        )
                        dstate_dk = i * ldtm_width + g * 8
                        dstate_addr = (dstate_dk // 64) * (cfg.d_v * 64) + cg2_tidx * 64 + swizzle_xor_128b(cg2_tidx, dstate_dk % 64)
                        (sDstate_base + dstate_addr).store(cutlass.Vector.from_elements(dstate_smem_pack, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready[0].arrive()

        for rev_idx in cutlass.range(num_item_chunks):
            abs_chunk = compute_end - 1 - rev_idx
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0

            # ---- decay-scale latch ---------------------------------------------------
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
            decay_scale_fp32 = []
            for i in cutlass.range_constexpr(16):
                decay_scale_fp32.append(cute.math.exp2(last_cumsumlog - sCumsumlog[(lane_idx % 4) * 2 + (i // 2) * 8 + i % 2, 0, gate_idx], fastmath=True))
            decay_scale_vals = [decay_scale_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
            bars.mb_cg2_chunk_ready[0].arrive()

            # ---- dK inter rescale ----------------------------------------------------
            cg2_k_idx = cg2_k_index.idx
            cg2_k_index = advance(cg2_k_index, cfg.smem_k_stages)
            k_dot_dk_inter = cutlass.Float32(0.0)
            if rev_idx + DSTATE_IN0 >= 1:
                bars.mb_dk_scale_acc_ready[0].wait(cg2_kk_scale_ready.phase)
                cg2_kk_scale_ready = advance(cg2_kk_scale_ready, 1)
                k_dot_dk_inter_lo = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
                k_dot_dk_inter_hi = [opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero(), opaque_f32_zero()]
                for half in cutlass.range_constexpr(2):
                    dki_ptr = nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32)
                    dki_vec = nvvm.tcgen05_ld("16x256b", dki_ptr, num=8)
                    dki_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(dki_vec[2 * j], dki_vec[2 * j + 1], decay_scale_vals[2 * j], decay_scale_vals[2 * j + 1])
                        dki_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dki_ptr, cutlass.Vector.from_elements(tuple(dki_scaled), cutlass.Float32))
                    for block in cutlass.range_constexpr(4):
                        frag_addr = frag_segment + (frag_row + block * 16) * 64 + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                        k_frag = nvvm.ldmatrix(sK_base + cg2_k_idx * k_stage_elements_cg2 + frag_addr, 4, nvvm.MMALayout.COL)
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=cfg.io_dtype)
                            k_dot_dk_inter_lo[i], k_dot_dk_inter_hi[i] = ffma2(
                                dki_scaled[8 * block + 2 * i], dki_scaled[8 * block + 2 * i + 1], k_lo, k_hi, k_dot_dk_inter_lo[i], k_dot_dk_inter_hi[i]
                            )
                nvvm.tcgen05_wait("store")
                bars.mb_dk_scale_acc_done[0].arrive()
                k_dot_dk_inter = ((k_dot_dk_inter_lo[0] + k_dot_dk_inter_lo[1]) + (k_dot_dk_inter_lo[2] + k_dot_dk_inter_lo[3])) + (
                    (k_dot_dk_inter_hi[0] + k_dot_dk_inter_hi[1]) + (k_dot_dk_inter_hi[2] + k_dot_dk_inter_hi[3])
                )
                for off in [1, 2, 4, 8, 16]:
                    k_dot_dk_inter += nvvm.shfl_sync(0xFFFFFFFF, k_dot_dk_inter, off, 31, kind=nvvm.Shfl.BFLY)

            # ---- dK attn read: TMEM -> registers -------------------------------------
            bars.mb_dk_attn_acc_ready[0].wait(cg2_dk_attn_ready.phase)
            cg2_dk_attn_ready = advance(cg2_dk_attn_ready, 1)
            dks_regs = []
            for half in cutlass.range_constexpr(2):
                dks_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32),
                    num=8,
                )
                dks_regs.append([dks_vec[k] for k in range(32)])
            nvvm.tcgen05_wait("load")
            bars.mb_dk_attn_acc_done[0].arrive()

            # ---- dK stage slot -------------------------------------------------------
            dk_stg_idx = dk_index.idx
            bars.mb_dk_tmastg_done[dk_stg_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)

            # ---- part reductions -----------------------------------------------------
            part_k = [acc_zero] * 16
            for half in cutlass.range_constexpr(2):
                for block in cutlass.range_constexpr(4):
                    frag_addr = frag_segment + (frag_row + block * 16) * 64 + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                    k_frag = nvvm.ldmatrix(sK_base + cg2_k_idx * k_stage_elements_cg2 + frag_addr, 4, nvvm.MMALayout.COL)
                    for i in cutlass.range_constexpr(4):
                        k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=cfg.io_dtype)
                        frag_e0 = cutlass.const_expr(8 * block + 2 * i)
                        part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                        if cutlass.const_expr(half == 0 and i % 2 == 0):
                            part_k[part_e0], part_k[part_e0 + 1] = fmul2(dks_regs[half][frag_e0], dks_regs[half][frag_e0 + 1], k_lo, k_hi)
                        else:
                            part_k[part_e0], part_k[part_e0 + 1] = ffma2(
                                dks_regs[half][frag_e0], dks_regs[half][frag_e0 + 1], k_lo, k_hi, part_k[part_e0], part_k[part_e0 + 1]
                            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_cg2_done[cg2_k_idx].arrive()

            part_k_lo, part_k_hi = warp_reduce_scatter_sum_16_elements(part_k, lane_idx)
            tok0 = (lane_idx // 4) * 8 + (lane_idx % 4) * 2
            (sdk_red + warp_id * 64 + tok0).store(part_k_lo)
            (sdk_red + warp_id * 64 + tok0 + 1).store(part_k_hi - k_dot_dk_inter if lane_idx == 31 else part_k_hi)

            bars.mb_dq_acc_total_done[0].wait(cg2_dq_total_index.phase)
            cg2_dq_total_index = advance(cg2_dq_total_index, 1)

            # ---- NEXT-CHUNK dstate prep ----------------------------------------------
            if abs_chunk >= write_start + 1:
                dstate_idx = dstate_acc_index.idx
                bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
                dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
                dstate_done_idx = dstate_idx
                dstate_input_idx = dstate_input_index.idx
                bars.mb_dstate_input_done[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32), num=32)
                    dstate_pack = [fp32_to_fp16(dstate_vec[2 * j], dstate_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_input_col + i * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dstate_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready[dstate_input_idx].arrive()

            # ---- dstate prep: TMEM -> SMEM -------------------------------------------
            if abs_chunk >= write_start + 1:
                bars.mb_state_dot_dstate_done[0].wait(cg2_state_dot_dstate_index.phase)
                cg2_state_dot_dstate_index = advance(cg2_state_dot_dstate_index, 1)
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_smem_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(ldtm_width // 8):
                        dstate_smem_pack = tuple(
                            fp32_to_fp16(dstate_smem_vec[g * 8 + 2 * t], dstate_smem_vec[g * 8 + 2 * t + 1], dtype=cfg.io_dtype) for t in range(4)
                        )
                        dstate_dk = i * ldtm_width + g * 8
                        dstate_addr = (dstate_dk // 64) * (cfg.d_v * 64) + cg2_tidx * 64 + swizzle_xor_128b(cg2_tidx, dstate_dk % 64)
                        (sDstate_base + dstate_addr).store(cutlass.Vector.from_elements(dstate_smem_pack, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready[0].arrive()

            if abs_chunk < write_start + 1:
                cg2_state_dot_dstate_index = advance(cg2_state_dot_dstate_index, 1)

            bars.mb_dgate_cg1_ready[0].wait(cg2_dgate_index.phase)
            cg2_dgate_index = advance(cg2_dgate_index, 1)
            nvvm.barrier_cta_sync_aligned(cfg.cg2_barrier_id, thread_count=cfg.cg2_barrier_threads)
            if cg2_tidx < 64:
                dgate_sum = (
                    (sdk_red + cg2_tidx).load() + (sdk_red + 64 + cg2_tidx).load() + (sdk_red + 128 + cg2_tidx).load() + (sdk_red + 192 + cg2_tidx).load()
                )
                sCumsumlog[cg2_tidx, 0, gate_idx] = sCumsumlog[cg2_tidx, 0, gate_idx] - dgate_sum
            nvvm.barrier_cta_sync_aligned(cfg.cg2_barrier_id, thread_count=cfg.cg2_barrier_threads)

            # ---- dK fold -------------------------------------------------------------
            if abs_chunk >= FIRST_STATE_CHUNK:
                cumprod_neg = []
                for i in cutlass.range_constexpr(16):
                    cumprod_neg.append(sCumprod[(lane_idx % 4) * 2 + (i // 2) * 8 + i % 2, 0, gate_idx] * neg_one)
                cumprod_neg_vals = [cumprod_neg[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
                bars.mb_dk_state_path_acc_ready[0].wait(cg2_dk_state_path_ready.phase)
                cg2_dk_state_path_ready = advance(cg2_dk_state_path_ready, 1)
                bars.mb_dk_total_acc_ready[0].wait(cg2_dk_total_ready.phase)
                cg2_dk_total_ready = advance(cg2_dk_total_ready, 1)
                for half in cutlass.range_constexpr(2):
                    dk_state_path_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dk_state_path_col, cutlass.Float32),
                        num=8,
                    )
                    dmr_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32), num=8)
                    nvvm.tcgen05_wait("load")
                    if cutlass.const_expr(half == 1):
                        bars.mb_dk_total_acc_done[0].arrive()
                    for block in cutlass.range_constexpr(4):
                        dk_row = []
                        for j in cutlass.range_constexpr(4):
                            e = cutlass.const_expr(8 * block + 2 * j)
                            n0, n1 = fmul2(dk_state_path_vec[e], dk_state_path_vec[e + 1], cumprod_neg_vals[e], cumprod_neg_vals[e + 1])
                            dk_row += [dmr_vec[e] + n0, dmr_vec[e + 1] + n1]
                        dk_pack = [fp32_to_fp16(dk_row[2 * j], dk_row[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elements
                                + frag_segment
                                + (frag_row + block * 16) * 64
                                + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                            ),
                            dk_pack,
                            nvvm.MMALayout.COL,
                        )
            if abs_chunk < FIRST_STATE_CHUNK:
                bars.mb_dk_total_acc_ready[0].wait(cg2_dk_total_ready.phase)
                cg2_dk_total_ready = advance(cg2_dk_total_ready, 1)
                for half in cutlass.range_constexpr(2):
                    dmr_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32), num=8)
                    nvvm.tcgen05_wait("load")
                    if cutlass.const_expr(half == 1):
                        bars.mb_dk_total_acc_done[0].arrive()
                    for block in cutlass.range_constexpr(4):
                        dk_pack = [fp32_to_fp16(dmr_vec[8 * block + 2 * j], dmr_vec[8 * block + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elements
                                + frag_segment
                                + (frag_row + block * 16) * 64
                                + swizzle_xor_128b(frag_row + block * 16, frag_col + half * 16)
                            ),
                            dk_pack,
                            nvvm.MMALayout.COL,
                        )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()
            bars.mb_gate_done[gate_idx].arrive()

        # ---- dstate store: with an initial state this is its gradient ----------------
        if num_item_chunks > 0:
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
            if cutlass.const_expr(cfg.use_dstate0):
                if write_start == 0:
                    gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                    for i in cutlass.range_constexpr(num_ldtms):
                        dstate0_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32), num=32)
                        for kk in cutlass.range_constexpr(32):
                            gDstate0[cg2_tidx, i * ldtm_width + kk] = dstate0_vec[kk]
            if cutlass.const_expr(not cfg.use_dstate_in):
                bars.mb_dstate_scale_acc_done[dstate_idx].arrive()
        else:
            if cutlass.const_expr(cfg.use_dstate0):
                write_passthrough = write_start == 0
                if write_passthrough:
                    gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                    if cutlass.const_expr(cfg.use_dstate_in):
                        gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                        for i in cutlass.range_constexpr(num_ldtms):
                            for kk in cutlass.range_constexpr(32):
                                gDstate0[cg2_tidx, i * ldtm_width + kk] = gDstate_in[cg2_tidx, i * ldtm_width + kk]
                    else:
                        for i in cutlass.range_constexpr(num_ldtms):
                            for kk in cutlass.range_constexpr(32):
                                gDstate0[cg2_tidx, i * ldtm_width + kk] = cutlass.Float32(0.0)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.tmem_dstate_input_stages):
        bars.mb_dstate_input_done[dstate_input_index.idx].wait(dstate_input_index.phase)
        dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
    for _ in range(cfg.smem_dk_stages):
        bars.mb_dk_tmastg_done[dk_index.idx].wait(dk_index.phase)
        dk_index = advance(dk_index, cfg.smem_dk_stages)


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_do,
    base_checkpoint,
    base_dq,
    base_dk,
    base_dv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do_: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    n_batch: cutlass.Int32,
    q_rs: cutlass.Int32,
    k_rs: cutlass.Int32,
    v_rs: cutlass.Int32,
    do_rs: cutlass.Int32,
    checkpoint_rs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    dq_rs: cutlass.Int32,
    dk_rs: cutlass.Int32,
    dv_rs: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub2 = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub3 = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub4 = cute.make_tensor(desc_workspace.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub5 = cute.make_tensor(desc_workspace.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub6 = cute.make_tensor(desc_workspace.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub7 = cute.make_tensor(desc_workspace.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, q_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, k_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_descs(base_v, sub2, cu_seqlens, v, n_batch, v_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        if nvvm.elect_sync():
            emit_seq_descs(base_do, sub3, cu_seqlens, do_, n_batch, do_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        if nvvm.elect_sync():
            emit_checkpoint_seq_descs(base_checkpoint, sub4, cu_seqlens, state_checkpoints, n_batch, checkpoint_rs, checkpoint_every_n, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 5:
        if nvvm.elect_sync():
            emit_seq_descs(base_dq, sub5, cu_seqlens, dq, n_batch, dq_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 6:
        if nvvm.elect_sync():
            emit_seq_descs(base_dk, sub6, cu_seqlens, dk, n_batch, dk_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 7:
        if nvvm.elect_sync():
            emit_seq_descs(base_dv, sub7, cu_seqlens, dv, n_batch, dv_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def prologue_kernel(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do_: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    q_rs: cutlass.Int32,
    k_rs: cutlass.Int32,
    v_rs: cutlass.Int32,
    do_rs: cutlass.Int32,
    checkpoint_rs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    dq_rs: cutlass.Int32,
    dk_rs: cutlass.Int32,
    dv_rs: cutlass.Int32,
) -> None:
    """Single-CTA prologue. Under ``run_order`` this kernel is the first
    work-item-table consumer, so it LPT-orders the table and zeroes both
    consumers' sched rings via :func:`order_body`; it then builds the
    per-batch TMA-descriptor arrays via :func:`build_descs_body`, one warp
    per array (the extra warps only take part in the order phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    if cutlass.const_expr(run_order):
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(do_.shape[1])
        order_body(
            order_gen,
            True,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_heads_out * n_batch,
            cu_seqlens,
            mStaging,
            mCount,
            mWorkItems,
            mScheduler,
            sKey,
            sIdx,
            sSpread,
        )
    build_descs_body(
        widx,
        base_q,
        base_k,
        base_v,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        desc_workspace,
        cu_seqlens,
        q,
        k,
        v,
        do_,
        state_checkpoints,
        dq,
        dk,
        dv,
        n_batch,
        q_rs,
        k_rs,
        v_rs,
        do_rs,
        checkpoint_rs,
        checkpoint_every_n,
        dq_rs,
        dk_rs,
        dv_rs,
    )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do_: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    state_checkpoints: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """One-launch prologue: LPT-order the work items (with ``run_order``, when
    this kernel is the backward pair's first table consumer) and build the
    per-(b,h) TMA-descriptor arrays (Q, K, V, dO, checkpoint loads; dQ, dK, dV
    stores) into ``tensormap_workspace``."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    d_v = v.shape[2]
    d_k_state = state_checkpoints.shape[2]
    d_v_state = state_checkpoints.shape[3]
    bpe = io_dtype.width // 8
    granule_elements = 128 // bpe
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]
    do_row_stride, do_head_stride = do_.stride[0], do_.stride[1]
    dq_row_stride, dq_head_stride = dq.stride[0], dq.stride[1]
    dk_row_stride, dk_head_stride = dk.stride[0], dk.stride[1]
    dv_row_stride, dv_head_stride = dv.stride[0], dv.stride[1]

    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q_row_stride, q_head_stride, 1)))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k_row_stride, k_head_stride, 1)))

    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    do_headed = cute.make_tensor(do_.iterator, cute.make_layout((d_v, heads_out, seqlen), stride=(1, do_.stride[1], do_.stride[0])))
    dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((dq.shape[2], heads_out, seqlen), stride=(1, dq.stride[1], dq.stride[0])))
    dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((dk.shape[2], heads_out, seqlen), stride=(1, dk.stride[1], dk.stride[0])))
    dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((d_v, heads_out, seqlen), stride=(1, dv.stride[1], dv.stride[0])))
    swz128 = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, granule_elements), stride_order=(2, 1, 0), swizzle=swz128)
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, granule_elements), stride_order=(2, 1, 0), swizzle=swz128)
    base_desc_v = tma.create_tensor_map_tiled_from_view(v_headed, box_dims=(granule_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swz128)
    base_desc_do = tma.create_tensor_map_tiled_from_view(do_headed, box_dims=(granule_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swz128)
    base_desc_dq = tma.create_tensor_map_tiled_from_view(dq_headed, box_dims=(granule_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swz128)
    base_desc_dk = tma.create_tensor_map_tiled_from_view(dk_headed, box_dims=(granule_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swz128)
    base_desc_dv = tma.create_tensor_map_tiled_from_view(dv_headed, box_dims=(granule_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swz128)
    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (d_v_state, d_k_state, state_checkpoints.shape[0], heads_out),
            stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
        ),
    )
    base_desc_checkpoint = tma.create_tensor_map_tiled_from_view(checkpoint_view, box_dims=(64, d_k_state, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swz128)

    prologue_kernel(
        run_order,
        order_gen,
        b_t,
        base_desc_q,
        base_desc_k,
        base_desc_v,
        base_desc_do,
        base_desc_checkpoint,
        base_desc_dq,
        base_desc_dk,
        base_desc_dv,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        do_,
        state_checkpoints,
        dq,
        dk,
        dv,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
        cutlass.Int32(q_row_stride),
        cutlass.Int32(k_row_stride),
        cutlass.Int32(v_row_stride),
        cutlass.Int32(do_row_stride),
        cutlass.Int32(state_checkpoints.stride[0]),
        cutlass.Int32(b_t),
        cutlass.Int32(dq_row_stride),
        cutlass.Int32(dk_row_stride),
        cutlass.Int32(dv_row_stride),
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    do_: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    cu_seqlens: cute.Tensor,
    dstate0: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    scheduler_counter: Optional[cute.Tensor],
    scale: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    h_q = cfg.h_q
    h_k = cfg.h_k
    h_v = cfg.h_v
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v

    # ---- SMEM sizing: per-buffer element cosizes -------------------------------------
    bpe = cfg.io_dtype.width // 8
    q_tile_elements = cfg.b_t * cfg.d_k
    k_tile_elements = cfg.b_t * cfg.d_k
    v_tile_elements = cfg.d_v * cfg.b_t
    do_tile_elements = cfg.d_v * cfg.b_t
    state_tile_elements = cfg.d_k * cfg.d_v
    tinv_tile_elements = cfg.b_t * cfg.b_t
    a_tile_elements = cfg.b_t * cfg.b_t
    dq_tile_elements = cfg.b_t * cfg.d_k
    dk_tile_elements = cfg.b_t * cfg.d_k
    dv_tile_elements = cfg.d_v * cfg.b_t
    cfg.q_cosize = q_tile_elements * cfg.smem_q_stages
    cfg.k_cosize = k_tile_elements * cfg.smem_k_stages
    cfg.v_cosize = v_tile_elements * cfg.smem_v_stages
    cfg.do_cosize = do_tile_elements * cfg.smem_do_stages
    cfg.state_cosize = state_tile_elements * cfg.smem_state_stages
    cfg.t_inv_cosize = tinv_tile_elements * cfg.smem_t_inv_stages
    cfg.a_cosize = a_tile_elements * cfg.smem_a_stages
    cfg.dq_cosize = dq_tile_elements * cfg.smem_dq_stages
    cfg.dk_cosize = dk_tile_elements * cfg.smem_dk_stages
    cfg.dv_cosize = dv_tile_elements * cfg.smem_dv_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))

    cfg.tma_q_bytes = q_tile_elements * bpe
    cfg.tma_k_bytes = k_tile_elements * bpe
    cfg.tma_v_bytes = v_tile_elements * bpe
    cfg.tma_do_bytes = do_tile_elements * bpe
    cfg.tma_state_bytes = state_tile_elements * bpe

    cfg.n_heads_out = heads_out
    cfg.q_ratio = heads_out // h_q
    cfg.k_ratio = heads_out // h_k
    cfg.v_ratio = heads_out // h_v
    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)

    kernel(
        cfg,
        gate,
        a_log,
        dt_bias,
        beta,
        dgate,
        dbeta,
        cu_seqlens,
        scale,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        q,
        k,
        v,
        do_,
        dq,
        dk,
        dv,
        dstate0,
        dstate_in,
        work_items,
        work_count,
        scheduler_counter,
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
def kernel(
    cfg: cutlass.Constexpr,
    mGate: cute.Tensor,
    mA_log: Optional[cute.Tensor],
    mDt_bias: Optional[cute.Tensor],
    mBeta: cute.Tensor,
    mDgate: cute.Tensor,
    mDbeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    scale: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    mQ,
    mK,
    mV,
    mdO,
    mdQ,
    mdK,
    mdV,
    mDstate0,
    mDstate_in,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main GDN bprop chunked kernel (warp-specialized persistent body)."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dynamic_scheduling):
        assert mScheduler is not None, "mScheduler must be provided if dynamic_scheduling is True"

    if cutlass.const_expr(cfg.is_GQA):
        h_r = cfg.h_q // cfg.h_v
        h_qv = cfg.h_v
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], mQ.shape[2], (h_r, h_qv)),
                stride=(mQ.stride[0], mQ.stride[2], (mQ.stride[1], h_r * mQ.stride[1])),
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], mK.shape[2], (h_r, h_qv)),
                stride=(mK.stride[0], mK.stride[2], (0, mK.stride[1])),
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (mV.shape[2], mV.shape[0], (h_r, h_qv)),
                stride=(mV.stride[2], mV.stride[0], (0, mV.stride[1])),
            ),
        )
    else:
        h_r = cfg.h_v // cfg.h_q
        h_qv = cfg.h_q
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[0], mQ.shape[2], (h_r, h_qv)),
                stride=(mQ.stride[0], mQ.stride[2], (0, mQ.stride[1])),
            ),
        )
        mK = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[0], mK.shape[2], (h_r, h_qv)),
                stride=(mK.stride[0], mK.stride[2], (0, mK.stride[1])),
            ),
        )
        mV = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (mV.shape[2], mV.shape[0], (h_r, h_qv)),
                stride=(mV.stride[2], mV.stride[0], (mV.stride[1], h_r * mV.stride[1])),
            ),
        )
    mGate = cute.make_tensor(
        mGate.iterator,
        cute.make_layout(
            (mGate.shape[0], (h_r, h_qv)),
            stride=(mGate.stride[0], (mGate.stride[1], h_r * mGate.stride[1])),
        ),
    )
    mBeta = cute.make_tensor(
        mBeta.iterator,
        cute.make_layout(
            (mBeta.shape[0], (h_r, h_qv)),
            stride=(mBeta.stride[0], (mBeta.stride[1], h_r * mBeta.stride[1])),
        ),
    )
    mDgate = cute.make_tensor(
        mDgate.iterator,
        cute.make_layout(
            (mDgate.shape[0], (h_r, h_qv)),
            stride=(mDgate.stride[0], (mDgate.stride[1], h_r * mDgate.stride[1])),
        ),
    )
    mDbeta = cute.make_tensor(
        mDbeta.iterator,
        cute.make_layout(
            (mDbeta.shape[0], (h_r, h_qv)),
            stride=(mDbeta.stride[0], (mDbeta.stride[1], h_r * mDbeta.stride[1])),
        ),
    )
    if cutlass.const_expr(mDstate0 is not None):
        mDstate0 = cute.make_tensor(
            mDstate0.iterator,
            cute.make_layout(
                (mDstate0.shape[2], mDstate0.shape[3], (h_r, h_qv), mDstate0.shape[0]),
                stride=(
                    mDstate0.stride[2],
                    mDstate0.stride[3],
                    (mDstate0.stride[1], h_r * mDstate0.stride[1]),
                    mDstate0.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(mDstate_in is not None):
        mDstate_in = cute.make_tensor(
            mDstate_in.iterator,
            cute.make_layout(
                (mDstate_in.shape[2], mDstate_in.shape[3], (h_r, h_qv), mDstate_in.shape[0]),
                stride=(
                    mDstate_in.stride[2],
                    mDstate_in.stride[3],
                    (mDstate_in.stride[1], h_r * mDstate_in.stride[1]),
                    mDstate_in.stride[0],
                ),
            ),
        )

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(7) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_gdn_bars(cfg)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=cutlass.AddressSpace.smem, alignment=16)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=128)

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = (cfg.d_v // 2) * 128
    V_LEAD = (cfg.d_v // 2) * 128
    STATE_LEAD = cfg.d_k * 128
    sQ_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.q_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sQ = SmemTile(
        base=sQ_raw.data_ptr(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages),
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
        base=sK_raw.data_ptr(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_trans = SmemTile(
        base=sK_raw.data_ptr(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQ_trans = SmemTile(
        base=sQ_raw.data_ptr(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages),
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
    sdO_trans = SmemTile(
        base=sdO_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO = SmemTile(
        base=sdO_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.state_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sState_trans = SmemTile(
        base=sState_raw.data_ptr(),
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages),
        stages=cfg.smem_state_stages,
        leading_byte_offset=STATE_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState = SmemTile(
        base=sState_raw.data_ptr(),
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages),
        stages=cfg.smem_state_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.t_inv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sTinv = SmemTile(
        base=sTinv_raw.data_ptr(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_trans = SmemTile(
        base=sTinv_raw.data_ptr(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKK_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.t_inv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sKK = SmemTile(
        base=sKK_raw.data_ptr(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.a_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sA = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=cfg.smem_a_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_trans = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=cfg.smem_a_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa_trans = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.a_cosize // cfg.smem_a_stages,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sDm = SmemTile(
        base=sDm_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_trans = SmemTile(
        base=sDm_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    # sub-bank split: V + dstate_entry + dQ + dK + dV allocated last
    sV_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.v_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sV_trans = SmemTile(
        base=sV_raw.data_ptr(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages),
        stages=cfg.smem_v_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV = SmemTile(
        base=sV_raw.data_ptr(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages),
        stages=cfg.smem_v_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDstate_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.d_k * cfg.d_v,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sDstate_trans = SmemTile(
        base=sDstate_raw.data_ptr(),
        elems_per_stage=cfg.d_k * cfg.d_v,
        stages=1,
        leading_byte_offset=STATE_LEAD,
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
        base=sdQ_raw.data_ptr(),
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages),
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
        base=sdK_raw.data_ptr(),
        elems_per_stage=(cfg.dk_cosize // cfg.smem_dk_stages),
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
        base=sdV_raw.data_ptr(),
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages),
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdstate_flat = cute.make_tensor(
        cute.make_ptr(cfg.io_dtype, sDstate_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes),
        cute.make_layout(cfg.d_k * cfg.d_v),
    )
    sstate_flat = cute.make_tensor(
        cute.make_ptr(cfg.io_dtype, sState_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes),
        cute.make_layout(cfg.state_cosize),
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

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_mma_done[s].init()
        bars.mb_q_cg1_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_mma_done[s].init()
        bars.mb_k_cg2_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_mma_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
    for s in range(cfg.smem_state_stages):
        bars.mb_state_ready[s].init()
        bars.mb_state_mma_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.smem_beta_stages):
        bars.mb_beta_ready[s].init()
        bars.mb_beta_done[s].init()
    for s in range(cfg.tmem_dstate_acc_stages):
        bars.mb_dstate_acc_ready[s].init()
        bars.mb_dstate_scale_acc_done[s].init()
    for b in (
        bars.mb_du_scale_acc_ready,
        bars.mb_du_scale_acc_done,
        bars.mb_du_total_acc_ready,
        bars.mb_dk_scale_acc_ready,
        bars.mb_dk_scale_acc_done,
        bars.mb_dk_attn_acc_ready,
        bars.mb_dk_attn_acc_done,
        bars.mb_dk_total_acc_ready,
        bars.mb_dk_total_acc_done,
    ):
        b[0].init()
    for b in (
        bars.mb_kk_acc_ready,
        bars.mb_kk_acc_done,
        bars.mb_a_acc_ready,
        bars.mb_k_state_acc_ready,
        bars.mb_u_acc_ready,
        bars.mb_dy_acc_ready,
    ):
        b[0].init()
    for s in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_ready[s].init()
    for s in range(cfg.smem_a_stages):
        bars.mb_a_ready[s].init()
        bars.mb_a_done[s].init()
    for s in range(cfg.tmem_dstate_input_stages):
        bars.mb_dstate_input_ready[s].init()
        bars.mb_dstate_input_done[s].init()
    for b in (
        bars.mb_do_prime_input_ready,
        bars.mb_du_input_ready,
        bars.mb_dyp_input_ready,
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
    bars.mb_dstate_smem_ready[0].init()
    bars.mb_da_ready[0].init()
    bars.mb_dq_acc_scale_ready[0].init()
    bars.mb_dq_acc_scale_done[0].init()
    bars.mb_dq_acc_total_ready[0].init()
    bars.mb_dq_acc_total_done[0].init()
    bars.mb_da_acc_ready[0].init()
    bars.mb_dm_acc_ready[0].init()
    bars.mb_dm_acc_done[0].init()
    bars.mb_dbeta_cg1_ready[0].init()
    bars.mb_dgate_cg1_ready[0].init()
    bars.mb_dgate_cg0_ready[0].init()
    bars.mb_cg2_chunk_ready[0].init()
    bars.mb_state_dot_dstate_done[0].init()
    bars.mb_dk_state_path_acc_ready[0].init()
    bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization ---------------------------------------------------------

    if warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        compute0_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sTinv=sTinv,
            sKK=sKK,
            sA=sA,
            sDa=sDa,
            sDm=sDm,
            sstate_flat=sstate_flat,
            sdstate_flat=sdstate_flat,
            sScheduler=sScheduler,
            bars=bars,
        )

    if warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        compute1_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            warp_idx=warp_idx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sQ=sQ,
            sV=sV,
            sdO=sdO,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sdQ=sdQ,
            sdV=sdV,
            sdK=sdK,
            sDa=sDa,
            sDm=sDm,
            sScheduler=sScheduler,
            bars=bars,
        )

    if warp_idx >= cfg.compute_group_2_warp_ids[0] and warp_idx <= cfg.compute_group_2_warp_ids[-1]:
        compute2_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            mWorkItems,
            mDstate0,
            mDstate_in,
            tidx,
            tmem_base_slot=tmem_base_slot,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sK=sK,
            sdK=sdK,
            sDstate_trans=sDstate_trans,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tcgen05_mma_warp_id:
        tcgen05_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tmem_base_slot=tmem_base_slot,
            sQ=sQ,
            sQ_trans=sQ_trans,
            sK=sK,
            sK_trans=sK_trans,
            sV_trans=sV_trans,
            sV=sV,
            sdO_trans=sdO_trans,
            sdO=sdO,
            sState_trans=sState_trans,
            sState=sState,
            sTinv=sTinv,
            sTinv_trans=sTinv_trans,
            sA=sA,
            sA_trans=sA_trans,
            sDa=sDa,
            sDa_trans=sDa_trans,
            sDstate_trans=sDstate_trans,
            sDm=sDm,
            sDm_trans=sDm_trans,
            sdV=sdV,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tma_qkv_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mScheduler=mScheduler,
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            sdO_raw=sdO_raw,
            sState_raw=sState_raw,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            desc_do_base=desc_do_base,
            desc_checkpoint_base=desc_checkpoint_base,
            sScheduler=sScheduler,
            bars=bars,
        )

    if warp_idx == cfg.load_gate_beta_warp_id:
        gate_beta_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            mGate=mGate,
            mA_log=mA_log,
            mDt_bias=mDt_bias,
            mBeta=mBeta,
            mDgate=mDgate,
            mDbeta=mDbeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sScheduler=sScheduler,
            bars=bars,
        )
    if warp_idx == cfg.epilogue_warp_id:
        tmastg_warp(
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
            sScheduler=sScheduler,
            bars=bars,
        )


@dataclass
class GdnBwdCfg:
    """Per-compile GDN bprop kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA) are the ``cute.compile`` cache
    keys; the rest is derived from the module-global ``CFG`` constants.
    ``host`` stamps the shape-derived fields at trace time.
    """

    use_initial_state: bool
    use_dstate_in: bool
    use_dstate0: bool
    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    is_GQA: bool
    log_gate: bool = False
    safe_gate: bool = False
    beta_sigmoid: bool = False

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    compute_group_2_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_2_WARP_IDS
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK
    dynamic_scheduling: bool = False
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- named barrier slots (ids 1-6; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0
    inverse_inner_barrier_id: int = 3
    inverse_inner_barrier_threads: int = 0
    init_state_store_barrier_id: int = 4
    init_state_store_barrier_threads: int = 0
    cg1_barrier_id: int = 5
    cg1_barrier_threads: int = 0
    cg2_barrier_id: int = 6
    cg2_barrier_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_do_stages: int = 1
    smem_state_stages: int = 1
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_a_stages: int = CFG.SMEM_A_STAGES
    smem_dq_stages: int = 1
    smem_dk_stages: int = 1
    smem_dv_stages: int = 1
    smem_gate_stages: int = 2
    smem_beta_stages: int = 2
    tmem_dstate_acc_stages: int = CFG.TMEM_DH_ACC_STAGES
    tmem_dvdk_acc_stages: int = CFG.TMEM_DVDK_ACC_STAGES
    tmem_dstate_input_stages: int = CFG.TMEM_DH_INP_STAGES
    tmem_shared_input_stages: int = CFG.TMEM_SHARED_INP_STAGES
    tmem_shared_acc_stages: int = CFG.TMEM_SHARED_ACC_STAGES
    tmem_dstate_acc_offset: int = 0
    tmem_dvdk_acc_offset: int = 0
    tmem_dstate_input_offset: int = 0
    tmem_shared_acc_offset: int = 0
    tmem_shared_input_offset: int = 0
    tmem_y_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    do_cosize: int = 0
    state_cosize: int = 0
    t_inv_cosize: int = 0
    a_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dv_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_do_bytes: int = 0
    tma_state_bytes: int = 0
    n_heads_out: int = 0
    q_ratio: int = 1
    k_ratio: int = 1
    v_ratio: int = 1


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dstate_in: bool = False,
    use_dstate0: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    dynamic_scheduling: bool = False,
) -> GdnBwdCfg:
    """Build the per-compile ``GdnBwdCfg`` (io_dtype in {Float16, BFloat16};
    acc is always Float32)."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = GdnBwdCfg(
        use_initial_state=use_initial_state,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        max_active_clusters=max_active_clusters,
        is_GQA=is_GQA,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        dynamic_scheduling=dynamic_scheduling,
    )
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    n_cg2 = len(cfg.compute_group_2_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1 + n_cg2)
    cfg.tmem_user_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1 + n_cg2)
    cfg.inverse_barrier_threads = cfg.threads_per_warp * n_cg0
    cfg.inverse_inner_barrier_threads = cfg.threads_per_warp * 2
    cfg.init_state_store_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.cg1_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.cg2_barrier_threads = cfg.threads_per_warp * n_cg2
    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dvdk_acc_offset = cfg.tmem_dstate_acc_offset + cfg.tmem_dstate_acc_stages * 128
    cfg.tmem_dstate_input_offset = cfg.tmem_dvdk_acc_offset + cfg.tmem_dvdk_acc_stages * 64
    cfg.tmem_shared_acc_offset = cfg.tmem_dstate_input_offset + cfg.tmem_dstate_input_stages * 64
    cfg.tmem_shared_input_offset = cfg.tmem_shared_acc_offset + cfg.tmem_shared_acc_stages * 64
    cfg.tmem_y_offset = cfg.tmem_shared_input_offset + cfg.tmem_shared_input_stages * (cfg.b_t // 2)
    return cfg


TENSORMAP_DESC_ARRAYS = 8  # per-batch runtime TMA descriptors: Q, K, V, dO, checkpoints, dQ, dK, dV
TENSORMAP_STATIC_SLOTS = 0


@functools.cache
def get_compiled_cache(
    io_dtype_str: str,
    cu_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dstate_in: bool = False,
    use_dstate0: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    dynamic_scheduling: bool = False,
    run_order: bool = False,
    order_gen: bool = False,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    is_GQA: bool,
    use_initial_state: bool = False,
    use_dstate_in: bool = False,
    use_dstate0: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    dynamic_scheduling: bool = False,
    *,
    num_sm: int,
    h_q: int,
    h_k: int,
    h_v: int,
    q_cute,
    k_cute,
    v_cute,
    gate_cute,
    a_log_cute=None,
    dt_bias_cute=None,
    beta_cute,
    dgate_cute,
    dbeta_cute,
    do_cute,
    dq_cute,
    dk_cute,
    dv_cute,
    cu_seqlens_cute,
    dstate0_cute=None,
    dstate_in_cute=None,
    work_items_cute=None,
    work_count_cute=None,
    scheduler_counter_cute=None,
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
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        dynamic_scheduling=dynamic_scheduling,
    )
    cfg.h_q = h_q
    cfg.h_k = h_k
    cfg.h_v = h_v

    return cute.compile(
        host,
        cfg,
        q_cute,
        k_cute,
        v_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        beta_cute,
        dgate_cute,
        dbeta_cute,
        do_cute,
        dq_cute,
        dk_cute,
        dv_cute,
        cu_seqlens_cute,
        dstate0_cute,
        dstate_in_cute,
        work_items_cute,
        work_count_cute,
        scheduler_counter_cute,
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
    state_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    cu_seqlens,
    scale: float,
    *,
    use_initial_state: bool = False,
    d_initial_state=None,
    d_final_state=None,
    work_items=None,
    work_count=None,
    scheduler_counter=None,
    scheduler_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    workspace,
    stream,
) -> None:
    """Execute the Blackwell chunked GDN bprop kernel (THD / varlen entry).

    Produces dQ/dK/dV/dGate/dBeta at ``HO = max(HQ, HV)`` heads (the caller
    reduces over the head group; dGate = dL/d(ln alpha)).  With
    ``use_initial_state``, chunk 0 reads its entering state from
    ``state_checkpoints`` row 0 like every other chunk (the forward writes the
    initial state there); ``d_initial_state`` (fp32) then also receives the
    initial-state gradient.  The two go together.  ``state_checkpoints`` is
    always the PLAIN per-chunk checkpoint series.  All tensors are DLPack-compatible
    CUDA tensors on the same device with a stride-1 innermost dim (outer
    strides are runtime arguments).
    Compile-cache-and-replay.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate — raw linear
            alpha, or the natural-log decay when ``log_gate``, or raw logits
            when ``safe_gate``, which applies the safe-gate transform
            ``-exp(a_log) * softplus(gate + dt_bias)``
        beta: ``(total_tokens, HO)`` float32, update gate — post-sigmoid, or
            io-dtype logits when ``use_beta_sigmoid``
        do: ``(total_tokens, HO, DV)`` float16/bfloat16, output gradient
        state_checkpoints: ``(total_checkpoints, HO, DV, DK)`` io dtype, per-chunk
            forward states from the prefill kernel's checkpoint output (``checkpoint_every_n_tokens=B_T``)
        dq/dk/dv: pre-allocated output gradients, shaped/typed like q/k/v at
            HO heads
        dgate/dbeta: pre-allocated ``(total_tokens, HO)`` float32 gate/beta
            gradients (``safe_gate`` leaves dgate in the transformed gate
            space; ``use_beta_sigmoid`` leaves dbeta in post-sigmoid space,
            io dtype)
        cu_seqlens: ``(num_seqs + 1,)`` int32
        use_initial_state: the forward ran with an initial state, so chunk 0
            has an entering state to load from ``state_checkpoints`` row 0
        scale: attention scale factor (must not be 0)
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO,)`` float32, safe-gate per-head bias (None = 0)
        use_beta_sigmoid: ``beta`` holds logits; sigmoid in-kernel
        work_items: ``(max_items, 8)`` int32 work-item table from
            ``common/split_k.py`` (REQUIRED; an uncut table row is the whole
            (b, h) sequence).  Each item computes chunks ``[write_start, compute_end)``
            backward and writes gradients only for ``[write_start, write_end)``.
        work_count: ``(1,)`` int32 device-side item count (REQUIRED)
        workspace: ``(>= tensormap_workspace_bytes(module, B) // 8,)`` int64,
            128-byte aligned; holds the per-(b,h) TMA descriptors
        stream: CUDA stream handle (``cudaStream_t`` as an int)
    """
    HQ = q.shape[1]
    HK = k.shape[1]
    HV = v.shape[1]
    HO = max(HQ, HV)
    DK = q.shape[2]
    B = cu_seqlens.shape[0] - 1
    is_GQA = HQ >= HV
    if work_items is None or work_count is None:
        raise ValueError(
            "work_items/work_count are required (built by the split-table stage, or by an order-generating prologue when work_item_scratch is None)"
        )
    io_dtype = get_dtype(q.dtype)
    for name, hh in (("HQ", HQ), ("HK", HK), ("HV", HV)):
        if HO % hh != 0:
            raise ValueError(f"{name}={hh} must divide {HO}")

    cu_stream = cuda.CUstream(int(stream))

    dynamic_scheduling = scheduler_counter is not None
    run_order = bool(order_in_prologue)
    order_gen = run_order and work_item_scratch is None
    if run_order and scheduler_all is None:
        raise ValueError("order_in_prologue requires scheduler_all (the prologue zeroes both consumers' sched rings)")
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if not safe_gate:
        a_log = None
        dt_bias = None
    cache = get_compiled_cache(
        str(q.dtype),
        str(cu_seqlens.dtype),
        HQ,
        HK,
        HV,
        is_GQA,
        use_initial_state,
        d_final_state is not None,
        d_initial_state is not None,
        log_gate,
        safe_gate,
        use_beta_sigmoid,
        dynamic_scheduling,
        run_order,
        order_gen,
    )

    if "compiled" not in cache:
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        workspace_cute = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()

        dstate0_cute = None
        if d_initial_state is not None:
            dstate0_cute = from_dlpack(d_initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate_in_cute = None
        if d_final_state is not None:
            dstate_in_cute = from_dlpack(d_final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        work_items_cute = from_dlpack(work_items, assumed_align=16)
        work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        scheduler_counter_cute = None
        if dynamic_scheduling:
            scheduler_counter_cute = from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic()
        a_log_cute = from_dlpack(a_log, assumed_align=4) if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=4) if dt_bias is not None else None
        cache["compiled"] = compile(
            io_dtype,
            is_GQA,
            use_initial_state=use_initial_state,
            use_dstate_in=d_final_state is not None,
            use_dstate0=d_initial_state is not None,
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            dynamic_scheduling=dynamic_scheduling,
            num_sm=multiprocessor_count(current_device()),
            h_q=HQ,
            h_k=HK,
            h_v=HV,
            q_cute=from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            k_cute=from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            v_cute=from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            gate_cute=from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            beta_cute=from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            dgate_cute=from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            dbeta_cute=from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            do_cute=from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            dq_cute=from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            dk_cute=from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            dv_cute=from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            cu_seqlens_cute=cu_seqlens_cute,
            dstate0_cute=dstate0_cute,
            dstate_in_cute=dstate_in_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            scheduler_counter_cute=scheduler_counter_cute,
            scale=scale,
            workspace_cute=workspace_cute,
            stream=cu_stream,
        )

    compiled = cache["compiled"]

    if "prologue" not in cache:
        checkpoints_pl = from_dlpack(state_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        cu_pl = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        staging_pl = None
        if run_order and not order_gen:
            staging_pl = from_dlpack(work_item_scratch, assumed_align=16)
            staging_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_pl = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        work_items_pl = from_dlpack(work_items, assumed_align=16)
        work_items_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        scheduler_all_pl = None
        if run_order:
            scheduler_all_pl = from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic()
        workspace_pl = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
        cache["prologue"] = cute.compile(
            prologue,
            io_dtype,
            CFG.B_T,
            run_order,
            order_gen,
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            checkpoints_pl,
            cu_pl,
            staging_pl,
            work_count_pl,
            work_items_pl,
            scheduler_all_pl,
            workspace_pl,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["prologue"](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        state_checkpoints,
        cu_seqlens,
        work_item_scratch if (run_order and not order_gen) else None,
        work_count,
        work_items,
        scheduler_all if run_order else None,
        workspace,
        cu_stream,
    )
    compiled(
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        dgate,
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
        scheduler_counter,
        scale,
        workspace,
        cu_stream,
    )
    return cache


def run_bwd(
    cache,
    q,
    k,
    v,
    gate,
    beta,
    do,
    state_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    cu_seqlens,
    d_initial_state,
    d_final_state,
    work_items,
    work_count,
    scheduler_counter,
    scheduler_all,
    work_item_scratch,
    tensormap_workspace,
    scale,
    stream,
    a_log=None,
    dt_bias=None,
) -> None:
    """Replay the compiled plan: the prologue launch, then the main launch.
    The caller owns the contract, which the plan validated at build, so
    nothing here raises."""
    cu_stream = cuda.CUstream(int(stream))
    cache["prologue"](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        state_checkpoints,
        cu_seqlens,
        work_item_scratch,
        work_count,
        work_items,
        scheduler_all,
        tensormap_workspace,
        cu_stream,
    )
    cache["compiled"](
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        dgate,
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
        scheduler_counter,
        scale,
        tensormap_workspace,
        cu_stream,
    )
