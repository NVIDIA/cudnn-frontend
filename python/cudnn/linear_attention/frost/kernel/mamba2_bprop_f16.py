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
Chunked Mamba-2 SSD BPROP kernel for SM100 / SM103 / SM107 (Cutlass primitives).

Algorithm overview (chunks c = NT-1 .. 0 of C = 128 tokens, natural orientation; both score tiles are computed
transposed, key token j on the TMEM lane):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], dO[BT,DV], S_c[DK,DV] (the entering state, from the checkpoint series),
           Gate[BT] (raw dt), A[H], Dt_bias[H], D[H] or D[H,DV]
  State  : dS[DK,DV] (the gradient of the entering state, fp32 in TMEM, carried backward)

  Gate warp: cumsumlog[t] = sum_{l<=t} dt_l A / ln 2, beta[t] = dt_t (the forward's tables)

  MMA order (per chunk, the operands become ready in this order):
    -  dA^T      = V @ dO^T                -> score region            (CG0 dA pass: E beta_j dA^T -> dA_raw^T bf16 SMEM)
    -  dK_state  = U'(T) @ dS_c+1^T        -> dK fresh                (U' = beta_j Lambda_j V_j, bf16 in TMEM over dQ)
    -  dU'       = K @ dS_c+1              -> dV fresh                (CG1 dU' pass: rows * beta_j Lambda_j + D dO_j)
    -  A_raw^T   = K @ Q^T                 -> score region            (CG0 A pass: E beta_j A_raw^T -> bf16 overlay)
    -  dS_c     += Q^T @ dO'               -> dstate (pre-scaled by gamma_c; dO' = Gamma_i dO_i)
    -  dQ_inter  = dO' @ S_c^T             -> dQ fresh
    -  dQ_attn  += dA_raw @ K              -> dQ
    -  dK_attn  += dA_raw^T @ Q            -> dK
    -  dV_intra += A^T(T) @ dO             -> dV

  CG2 keeps one scalar per chunk and head, <dS_c, S_c>, for the post-pass gate gradient, and the D-skip partials (per
  head the dA^T diagonal, per channel the raw-row dots); the per-token dots and the
  parameter gradients are the helpers' (common/gate_bwd.py, common/head_reduce.py) over the written dQ, dK, dV and the
  inputs.  The dQ and dK written here are per head; the fold to the B / C groups takes the row dots on the way.

SMEM layout (227 KB cap):
  Buffer                           Size (B)  Stages
  Q                                32768     1
  K                                32768     1
  V                                16384     1
  dO                               16384     2      <-- the two buffers alternate between dO (TMA) and dO' (CG1)
  state S_c (checkpoint tile)      16384     1
  dS bf16 SMEM copy                16384     1
  dK store staging                 32768     1
  dQ store staging / dA_raw^T      32768     1      <-- ALIAS: dA_raw^T until the dA_raw^T Q GEMM commits, then dQ
  dV store staging                 16384     1
  cumsumlog / beta                   512     2
  D[h, :] (per-channel D)            256     1
  exit-dot scratch, scheduler ring, mbarriers (8 B aligned)

TMEM layout (512 cols):
  cols 0-64    : dstate accumulator (fp32), lane = d_k row
  cols 64-128  : dV accumulator, lane = key token
  cols 128-256 : dK accumulator, lane = key token
  cols 256-384 : dQ accumulator, lane = query token; cols 256-288 hold the bf16 U' tile until the dO' S^T GEMM overwrites
  cols 384-512 : score region: dA^T, then A_raw^T, then the bf16 A^T overlay in cols 384-448

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - the dA pass, the A pass (the mask factor E beta_j held across both), the dQ drain
  warps 4-7     : compute group 1 - U' build, dO' build, dU' pass, dK drain
  warps 8-11    : compute group 2 - dstate pass (seed, exit dot, bf16 SMEM copy, pre-scale, dS_0 store), dV drain
  warp  12      : MMA warp       - the nine GEMMs; TMEM lifecycle
  warp  13      : TMA load warp  - loads Q, K, V, dO, S_c
  warp  14      : gate warp      - dt transform + cumsum tables, the exit-row prefix
  warp  15      : epilogue warp  - dQ, dK, dV TMA stores
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma

from ..common.thd import emit_checkpoint_seq_descs, emit_seq_descs, TENSOR_MAP_QWORDS
from ..common.gate_bwd import exit_slot_table_body
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, WORK_ITEM_DSTATE_DST, decode_head, decode_work_item, order_body

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log decays -> the kernel's log2 domain
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
    wait_on_dependent_grids,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, ffma2, fmul2, fp32_to_fp16, opaque_f32_zero, softplus
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .mamba2_bprop_config import CFG

USE_PDL = True


class Mamba2BpropBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_mma_done: MBarrier
    mb_v_cg1_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_do_skip_done: MBarrier
    mb_do_prime_ready: MBarrier
    mb_do_prime_mma_done: MBarrier
    mb_state_ready: MBarrier
    mb_state_mma_done: MBarrier
    mb_state_cg2_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_da_acc_ready: MBarrier
    mb_da_acc_done: MBarrier
    mb_da_ready: MBarrier
    mb_araw_acc_ready: MBarrier
    mb_a_ready: MBarrier

    mb_u_prime_ready: MBarrier
    mb_dstate_smem_ready: MBarrier
    mb_dstate_scaled: MBarrier
    mb_dstate_acc_ready: MBarrier
    mb_dv_unscaled_ready: MBarrier
    mb_dv_scaled_done: MBarrier

    mb_dq_acc_done: MBarrier
    mb_dk_acc_ready: MBarrier
    mb_dv_acc_ready: MBarrier
    mb_dv_acc_done: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> Mamba2BpropBars:
    """Mamba2BpropBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    EPILOGUE_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG2_THREADS = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    ALL_COMPUTE = CG0_THREADS + CG1_THREADS + CG2_THREADS
    GATE_CONSUMERS = ALL_COMPUTE
    SKIP_ROW_READERS = CG2_THREADS if (cfg.skip_bwd_partials and cfg.d_per_channel) else 0
    SKIP_DIAGONAL_READERS = CG2_THREADS if (cfg.skip_bwd_partials and not cfg.d_per_channel) else 0

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    return Mamba2BpropBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_mma_done=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_cg1_done=MBarrier(
            alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=CG1_THREADS + SKIP_ROW_READERS, producer=Producer.THREAD
        ),
        mb_do_ready=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_skip_done=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_do_prime_ready=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_do_prime_mma_done=MBarrier(
            alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_state_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_state_mma_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_state_cg2_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_CONSUMERS, producer=Producer.THREAD),
        mb_da_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_da_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS + SKIP_DIAGONAL_READERS, producer=Producer.THREAD),
        mb_da_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_araw_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_a_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_u_prime_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dstate_smem_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dstate_scaled=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dv_unscaled_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dv_scaled_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dq_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dk_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dv_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dv_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dq_tmastg_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dq_tmastg_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
        mb_dk_tmastg_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dk_tmastg_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
        mb_dv_tmastg_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dv_tmastg_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=ALL_COMPUTE, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
    )


@cute.jit
def scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    bars.mb_scheduler_done[scheduler_state.idx].wait(scheduler_state.phase)
    if elect_one:
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
        sScheduler[scheduler_state.idx] = num_ctas + fetched
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    next_tile = sScheduler[scheduler_state.idx]
    if elect_one:
        bars.mb_scheduler_ready[scheduler_state.idx].arrive()
    return next_tile, advance(scheduler_state, cfg.scheduler_stages)


@cute.jit
def scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    bars.mb_scheduler_ready[scheduler_state.idx].wait(scheduler_state.phase)
    next_tile = sScheduler[scheduler_state.idx]
    if elect_one:
        bars.mb_scheduler_done[scheduler_state.idx].arrive()
    return next_tile, advance(scheduler_state, cfg.scheduler_stages)


@cute.jit
def tmastg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    sdQ_raw,
    sdK_raw,
    sdV_raw,
    desc_dq_base,
    desc_dk_base,
    desc_dv_base,
    sScheduler,
    bars,
):
    """TMA-STG warp role (warp 15): the per-chunk dQ / dK / dV TMA stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    sdQ_tma = SmemTile(
        base=sdQ_raw,
        elems_per_stage=cfg.dq_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=cfg.b_t * box_elements,
    )
    sdK_tma = SmemTile(
        base=sdK_raw,
        elems_per_stage=cfg.dk_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=cfg.b_t * box_elements,
    )
    sdV_tma = SmemTile(
        base=sdV_raw,
        elems_per_stage=cfg.dv_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=cfg.b_t * box_elements,
    )
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )

        head_o, v_offset = decode_head(cfg, head_idx)
        slot = batch_idx * desc_qwords
        desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
        n_local = compute_end - write_start
        if elect_one:
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)

        for reverse_idx in cutlass.range(n_local):
            chunk_idx = compute_end - 1 - reverse_idx
            token_coordinate = chunk_idx * cutlass.Int32(cfg.b_t)
            writes = chunk_idx < write_end

            dq_idx = dq_index.idx
            bars.mb_dq_tmastg_ready[dq_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, 1)
            if writes:
                dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, token_coordinate)
                tma_store_tile(sdQ_tma[dq_idx], dq_slice, acquire=False)
                tma_store_commit()

            dk_idx = dk_index.idx
            bars.mb_dk_tmastg_ready[dk_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, 1)
            if writes:
                dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, token_coordinate)
                tma_store_tile(sdK_tma[dk_idx], dk_slice, acquire=False)
                tma_store_commit()

            dv_idx = dv_index.idx
            bars.mb_dv_tmastg_ready[dv_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, 1)
            if writes:
                dv_slice = tma_slice_runtime_desc(desc_dv_slot, v_offset, head_o, token_coordinate)
                tma_store_tile(sdV_tma[dv_idx], dv_slice, acquire=False)
                tma_store_commit()

            tma_store_wait(2)
            bars.mb_dq_tmastg_done[dq_idx].arrive()
            tma_store_wait(1)
            bars.mb_dk_tmastg_done[dk_idx].arrive()
            tma_store_wait(0)
            bars.mb_dv_tmastg_done[dv_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def gate_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    mGate,
    mA,
    mDt_bias,
    dt_min,
    dt_max,
    sCumsumlog,
    sBeta,
    mTransition,
    sScheduler,
    bars,
):
    """Gate warp (warp 14): the per-chunk dt transform and cumsum tables; with ``mTransition`` bound, the tile's log2
    decay product per (sequence, head)."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    lane_idx = tidx % cfg.threads_per_warp
    n_cols = cfg.b_t // cfg.threads_per_warp

    a = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = compute_end - write_start
        tile_log2 = cutlass.Float32(0.0)
        if n_local > 0:
            a = mA[head_o].to(cutlass.Float32) * cutlass.Float32(RCP_LN2)
            if cutlass.const_expr(mDt_bias is not None):
                bias = mDt_bias[head_o].to(cutlass.Float32)
            gGateSeq = mGate[None, head_o]
            for reverse_idx in cutlass.range(n_local):
                # ---- dt load + transform ----------------------------------------------------
                chunk_idx = compute_end - 1 - reverse_idx
                chunk_offset = batch_start + chunk_idx * cfg.b_t
                tokens = [chunk_offset + lane_idx + col * cfg.threads_per_warp for col in range(n_cols)]
                position_valid = [token < batch_end for token in tokens]
                dt_values = [gGateSeq[cutlass.min(token, batch_end - 1)].to(cutlass.Float32) + bias for token in tokens]
                if cutlass.const_expr(cfg.dt_softplus):
                    for col in cutlass.range_constexpr(n_cols):
                        dt_values[col] = softplus(dt_values[col])
                for col in cutlass.range_constexpr(n_cols):
                    dt_clamped = cutlass.max(cutlass.min(dt_values[col], dt_max), dt_min)
                    dt_values[col] = dt_clamped if position_valid[col] else cutlass.Float32(0.0)
                gate_values = [dt_values[col] * a for col in range(n_cols)]
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(0xFFFFFFFF, gate_values[col], offset, 0, kind=nvvm.Shfl.UP)
                        if lane_idx >= offset:
                            gate_values[col] = gate_values[col] + n
                for col in cutlass.range_constexpr(1, n_cols):
                    last_v = nvvm.shfl_sync(
                        0xFFFFFFFF,
                        gate_values[col - 1],
                        cfg.threads_per_warp - 1,
                        cfg.threads_per_warp - 1,
                        kind=nvvm.Shfl.IDX,
                    )
                    gate_values[col] += last_v
                if cutlass.const_expr(mTransition is not None):
                    tile_log2 = tile_log2 + nvvm.shfl_sync(
                        0xFFFFFFFF, gate_values[n_cols - 1], cfg.threads_per_warp - 1, cfg.threads_per_warp - 1, kind=nvvm.Shfl.IDX
                    )

                gate_idx = gate_index.idx
                gate_phase = gate_index.phase
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                bars.mb_gate_done[gate_idx].wait(gate_phase)
                for col in cutlass.range_constexpr(n_cols):
                    position = lane_idx + col * cfg.threads_per_warp
                    sCumsumlog[position, 0, gate_idx] = gate_values[col]
                    sBeta[position, 0, gate_idx] = dt_values[col]
                bars.mb_gate_ready[gate_idx].arrive()
        if cutlass.const_expr(mTransition is not None):
            if lane_idx == 0 and write_start == 0 and write_end == batch_num_chunks:
                mTransition[batch_idx, head_o] = tile_log2.to(mTransition.element_type)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    for _ in range(cfg.smem_gate_stages):
        bars.mb_gate_done[gate_index.idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)


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
    sV,
    sDo,
    sDo_trans,
    sState,
    sDstate,
    sDstate_trans,
    sDa,
    sDa_trans,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 12): the nine GEMMs of every chunk in operand-ready order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=0)
    k_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    do_prime_index = PipelineState.start(phase=0)
    state_index = PipelineState.start(phase=0)
    da_done_index = PipelineState.start(phase=0)  # same-chunk ready dependency
    da_ready_index = PipelineState.start(phase=0)
    a_ready_index = PipelineState.start(phase=0)
    u_prime_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    dstate_scaled_index = PipelineState.start(phase=0)
    dv_scaled_index = PipelineState.start(phase=0)
    dv_done_index = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8

    # ---- dA^T = V @ dO^T descriptor --------------------------------------------------------------
    instruction_descriptor_vdo = nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.b_t)
    bmm_v_do_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_vdo,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dK_state = U' @ dS^T descriptor ---------------------------------------------------------
    instruction_descriptor_up_ds = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.b_t
    )
    bmm_up_ds_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_k,
        K=cfg.d_v,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_up_ds,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dU' = K @ dS descriptor -----------------------------------------------------------------
    instruction_descriptor_k_ds = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_v, m_dim=cfg.b_t, b_major=1
    )
    bmm_k_ds_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_v,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_k_ds,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- A_raw^T = K @ Q^T descriptor ------------------------------------------------------------
    instruction_descriptor_kq = nvvm.Tcgen05InstrDesc.build(c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.b_t)
    bmm_k_q_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_kq,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dS += Q^T @ dO' descriptor --------------------------------------------------------------
    instruction_descriptor_qt_do_prime = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_v, m_dim=cfg.d_k, a_major=1, b_major=1
    )
    bmm_qt_do_prime_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_qt_do_prime,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dQ_inter = dO' @ S^T descriptor ---------------------------------------------------------
    instruction_descriptor_do_prime_s = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.b_t, b_major=1
    )
    bmm_do_prime_s_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_k,
        K=cfg.d_v,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_do_prime_s,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dQ_attn += dA_raw @ K descriptor --------------------------------------------------------
    instruction_descriptor_da_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.b_t, a_major=1, b_major=1
    )
    bmm_da_k_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_da_k,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dK_attn += dA_raw^T @ Q descriptor ------------------------------------------------------
    instruction_descriptor_dat_q = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.b_t, b_major=1
    )
    bmm_dat_q_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_dat_q,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    # ---- dV_intra += A^T @ dO descriptor ---------------------------------------------------------
    instruction_descriptor_at_do = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_v, m_dim=cfg.b_t, b_major=1
    )
    bmm_at_do_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_at_do,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    tmem_col = tmem_base_slot.load() & 0xFFFF
    dstate_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dstate_acc_offset, cutlass.Float32)
    dv_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dv_acc_offset, cutlass.Float32)
    dk_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dk_acc_offset, cutlass.Float32)
    dq_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dq_acc_offset, cutlass.Float32)
    u_prime_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dq_acc_offset, cutlass.Int8)
    score_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_score_acc_offset, cutlass.Float32)
    a_overlay_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_score_acc_offset, cutlass.Int8)
    desc_q = desc_opaque(sQ[0].desc())
    desc_q_trans = desc_opaque(sQ_trans[0].desc())
    desc_k = desc_opaque(sK[0].desc())
    desc_k_trans = desc_opaque(sK_trans[0].desc())
    desc_v = desc_opaque(sV[0].desc())
    desc_state = desc_opaque(sState[0].desc())
    desc_dstate = desc_opaque(sDstate[0].desc())
    desc_dstate_trans = desc_opaque(sDstate_trans[0].desc())
    desc_da = desc_opaque(sDa[0].desc())
    desc_da_trans = desc_opaque(sDa_trans[0].desc())

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = compute_end - write_start

        for reverse_idx in cutlass.range(n_local):  # noqa: B007
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else reverse_idx > 0
            do_buffer = do_index.idx
            do_prime_buffer = cutlass.Int32(1) - do_buffer
            desc_do = desc_opaque(sDo[do_buffer].desc())
            desc_do_trans = desc_opaque(sDo_trans[do_buffer].desc())
            desc_do_prime = desc_opaque(sDo[do_prime_buffer].desc())
            desc_do_prime_trans = desc_opaque(sDo_trans[do_prime_buffer].desc())

            # ---- dA^T = V(S) @ dO(S)^T -------------------------------------------------
            bars.mb_v_ready[0].wait(v_index.phase)
            v_index = advance(v_index, 1)
            bars.mb_do_ready[do_buffer].wait(do_index.phase)
            mma_ss(bmm_v_do_desc, desc_v, desc_do, score_acc_ptr, accumulate=False)
            if elect_one:
                bars.mb_da_acc_ready[0].arrive(cta_group=1)
                bars.mb_v_mma_done[0].arrive(cta_group=1)

            # ---- dK_state = U'(T) @ dS^T -------------------------------------------------
            if have_dstate:
                bars.mb_dstate_smem_ready[0].wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)
                bars.mb_u_prime_ready[0].wait(u_prime_index.phase)
                u_prime_index = advance(u_prime_index, 1)
                for k in cutlass.range_constexpr(cfg.d_v // 16):
                    mma_ts_step(bmm_up_ds_desc, u_prime_ptr, desc_dstate, dk_acc_ptr, k, cutlass.Boolean(k > 0))

            # ---- dU' = K(S) @ dS -------------------------------------------------------
            bars.mb_k_ready[0].wait(k_index.phase)
            k_index = advance(k_index, 1)
            bars.mb_dv_acc_done[0].wait(dv_done_index.phase)
            dv_done_index = advance(dv_done_index, 1)
            if have_dstate:
                mma_ss(bmm_k_ds_desc, desc_k, desc_dstate_trans, dv_acc_ptr, accumulate=False)
            if elect_one:
                bars.mb_dv_unscaled_ready[0].arrive(cta_group=1)

            # ---- A_raw^T = K(S) @ Q(S)^T -------------------------------------------------
            bars.mb_q_ready[0].wait(q_index.phase)
            q_index = advance(q_index, 1)
            bars.mb_da_acc_done[0].wait(da_done_index.phase)
            da_done_index = advance(da_done_index, 1)
            mma_ss(bmm_k_q_desc, desc_k, desc_q, score_acc_ptr, accumulate=False)
            if elect_one:
                bars.mb_araw_acc_ready[0].arrive(cta_group=1)

            # ---- dS += Q(S)^T @ dO'(S) ----------------------------------------------------
            if have_dstate:
                bars.mb_dstate_scaled[0].wait(dstate_scaled_index.phase)
                dstate_scaled_index = advance(dstate_scaled_index, 1)
            bars.mb_do_prime_ready[do_prime_buffer].wait(do_prime_index.phase)
            mma_ss(bmm_qt_do_prime_desc, desc_q_trans, desc_do_prime_trans, dstate_acc_ptr, accumulate=have_dstate)
            if elect_one:
                bars.mb_dstate_acc_ready[0].arrive(cta_group=1)

            # ---- dQ_inter = dO'(S) @ S^T --------------------------------------------------
            bars.mb_state_ready[0].wait(state_index.phase)
            state_index = advance(state_index, 1)
            mma_ss(bmm_do_prime_s_desc, desc_do_prime, desc_state, dq_acc_ptr, accumulate=False)
            if elect_one:
                bars.mb_state_mma_done[0].arrive(cta_group=1)
                bars.mb_do_prime_mma_done[do_prime_buffer].arrive(cta_group=1)

            # ---- dQ += dA_raw(S) @ K(S) ---------------------------------------------------
            bars.mb_da_ready[0].wait(da_ready_index.phase)
            da_ready_index = advance(da_ready_index, 1)
            mma_ss(bmm_da_k_desc, desc_da_trans, desc_k_trans, dq_acc_ptr, accumulate=True)
            if elect_one:
                bars.mb_k_done[0].arrive(cta_group=1)

            # ---- dK += dA_raw^T(S) @ Q(S) --------------------------------------------------
            mma_ss(bmm_dat_q_desc, desc_da, desc_q_trans, dk_acc_ptr, accumulate=have_dstate)
            if elect_one:
                bars.mb_dk_acc_ready[0].arrive(cta_group=1)
                bars.mb_q_done[0].arrive(cta_group=1)

            # ---- dV += A^T(T) @ dO(S) ------------------------------------------------------
            bars.mb_dv_scaled_done[0].wait(dv_scaled_index.phase)
            dv_scaled_index = advance(dv_scaled_index, 1)
            bars.mb_a_ready[0].wait(a_ready_index.phase)
            a_ready_index = advance(a_ready_index, 1)
            for k in cutlass.range_constexpr(cfg.b_t // 16):
                mma_ts_step(
                    bmm_at_do_desc,
                    a_overlay_ptr,
                    desc_do_trans,
                    dv_acc_ptr,
                    k,
                    cutlass.Boolean(True) if cutlass.const_expr(k > 0 or cfg.has_d) else have_dstate,
                )
            if elect_one:
                bars.mb_dv_acc_ready[0].arrive(cta_group=1)
                bars.mb_do_mma_done[do_buffer].arrive(cta_group=1)

            do_index = advance(do_index, cfg.smem_do_stages)
            do_prime_index = advance(do_prime_index, cfg.smem_do_stages)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base_slot.load() & 0xFFFF, cutlass.Int8),
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
    sDo_raw,
    sState_raw,
    tidx,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_do_base,
    desc_checkpoint_base,
    mScheduler,
    sScheduler,
    bars,
    q_ratio,
    k_ratio,
    v_ratio,
):
    """TMA-LDG warp role (warp 13): the per-chunk V, dO, K, Q and S_c loads of the reverse walk."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)
    do_step = cutlass.Int32(0)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    bt = cfg.b_t
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=cfg.q_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=cfg.k_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=cfg.v_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sDo_tma = SmemTile(
        base=sDo_raw,
        elems_per_stage=cfg.do_cosize // cfg.smem_do_stages,
        stages=cfg.smem_do_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sState_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=cfg.state_cosize,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=cfg.d_v * box_elements,
    )
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )

        head_o, v_offset = decode_head(cfg, head_idx)
        head_q = head_idx // q_ratio
        head_k = head_idx // k_ratio
        head_v = head_idx // v_ratio
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_checkpoint_slot)

        for reverse_idx in cutlass.range(compute_end - write_start):
            chunk_idx = compute_end - 1 - reverse_idx
            token_coordinate = chunk_idx * cutlass.Int32(cfg.b_t)

            # ---- V load --------------------------------------------------------------------
            bars.mb_v_mma_done[0].wait(v_index.phase)
            bars.mb_v_cg1_done[0].wait(v_index.phase)
            v_index = advance(v_index, 1)
            if elect_one:
                bars.mb_v_ready[0].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, token_coordinate)
            tma_load_tile(sV_tma[0], v_slice, bars.mb_v_ready[0].smem_ptr, acquire=False)

            # ---- dO load -------------------------------------------------------------------
            do_buffer = do_step & cutlass.Int32(1)
            prior_do_prime = (do_step + do_buffer) >> 1
            bars.mb_do_prime_mma_done[do_buffer].wait((prior_do_prime & cutlass.Int32(1)) ^ cutlass.Int32(1))
            if elect_one:
                bars.mb_do_ready[do_buffer].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, v_offset, head_o, token_coordinate)
            tma_load_tile(sDo_tma[do_buffer], do_slice, bars.mb_do_ready[do_buffer].smem_ptr, acquire=False)
            do_step = do_step + cutlass.Int32(1)

            # ---- K load --------------------------------------------------------------------
            bars.mb_k_done[0].wait(k_index.phase)
            k_index = advance(k_index, 1)
            if elect_one:
                bars.mb_k_ready[0].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, token_coordinate)
            tma_load_tile(sK_tma[0], k_slice, bars.mb_k_ready[0].smem_ptr, acquire=False)

            # ---- Q load --------------------------------------------------------------------
            bars.mb_q_done[0].wait(q_index.phase)
            q_index = advance(q_index, 1)
            if elect_one:
                bars.mb_q_ready[0].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, token_coordinate)
            tma_load_tile(sQ_tma[0], q_slice, bars.mb_q_ready[0].smem_ptr, acquire=False)

            # ---- S_c load ------------------------------------------------------------------
            bars.mb_state_mma_done[0].wait(state_index.phase)
            bars.mb_state_cg2_done[0].wait(state_index.phase)
            state_index = advance(state_index, 1)
            if elect_one:
                bars.mb_state_ready[0].arrive(n_bytes=cfg.tma_state_bytes)
            checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), v_offset, chunk_idx, head_o)
            tma_load_tile(sState_tma[0], checkpoint_slice, bars.mb_state_ready[0].smem_ptr, acquire=False)

        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)

    bars.mb_q_done[0].wait(q_index.phase)
    bars.mb_k_done[0].wait(k_index.phase)
    bars.mb_v_mma_done[0].wait(v_index.phase)
    bars.mb_v_cg1_done[0].wait(v_index.phase)
    bars.mb_state_mma_done[0].wait(state_index.phase)
    bars.mb_state_cg2_done[0].wait(state_index.phase)
    for b in cutlass.range_constexpr(cfg.smem_do_stages):
        buffer = cutlass.Int32(b)
        loads = (do_step + cutlass.Int32(1) - buffer) >> 1
        bars.mb_do_mma_done[buffer].wait((loads & cutlass.Int32(1)) ^ cutlass.Int32(1))
        if cutlass.const_expr(cfg.skip_bwd_partials and cfg.d_per_channel):
            bars.mb_do_skip_done[buffer].wait((loads & cutlass.Int32(1)) ^ cutlass.Int32(1))


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    warp_idx,
    tmem_base_slot,
    sDa_raw,
    sCumsumlog,
    sBeta,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): the dA pass, the A pass and the dQ drain, one token per thread."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    da_acc_index = PipelineState.start(phase=0)
    araw_acc_index = PipelineState.start(phase=0)
    dq_tmastg_index = PipelineState.start(phase=1)
    dk_ready_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.b_t // ldtm_width)
    sttm_width = 16
    num_dq_pieces = cutlass.const_expr(cfg.d_k // ldtm_width)
    bytes_per_element = cfg.io_dtype.width // 8
    vector_width = 16 // bytes_per_element
    block_elements = cfg.b_t * 64
    neg_inf = cutlass.Float32(float("-inf"))
    zero = opaque_f32_zero()
    sDa_base = sDa_raw.data_ptr()
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_score_col = tmem_col + cfg.tmem_score_acc_offset
    tmem_dq_col = tmem_col + cfg.tmem_dq_acc_offset
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = compute_end - write_start

        for reverse_idx in cutlass.range(n_local):  # noqa: B007
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            g_j = sCumsumlog[cg0_tidx, 0, gate_idx]
            beta_j = sBeta[cg0_tidx, 0, gate_idx]

            # ---- dA pass, two halves: w[j, i] = [i >= j] exp2(g_i - g_j) beta_j, dA_raw^T = w dA^T -> bf16 SMEM ----
            bars.mb_da_acc_ready[0].wait(da_acc_index.phase)
            da_acc_index = advance(da_acc_index, 1)
            w_values = []
            da_pack = []
            pieces_per_half = num_ldtms // 2
            vectors_per_piece = ldtm_width // vector_width
            half_base = row_lo_addr + tmem_score_col
            g_j_half = g_j
            for half in cutlass.range_constexpr(2):
                if cutlass.const_expr(half == 1):
                    half_base = half_base + cutlass.Int32(d_hi * zero)
                    g_j_half = g_j + zero * d_hi
                da_vectors = []
                for piece in cutlass.range_constexpr(half * pieces_per_half, (half + 1) * pieces_per_half):
                    da_vectors.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(half_base + piece * ldtm_width, cutlass.Float32), num=32))
                nvvm.tcgen05_wait("load")
                if cutlass.const_expr(half == 1):
                    bars.mb_da_acc_done[0].arrive()
                    bars.mb_dq_tmastg_done[0].wait(dq_tmastg_index.phase)
                    dq_tmastg_index = advance(dq_tmastg_index, 1)
                    for q in cutlass.range_constexpr(pieces_per_half * vectors_per_piece):
                        i0 = q * vector_width
                        (sDa_base + (i0 // 64) * block_elements + cg0_tidx * 64 + swizzle_xor_128b(cg0_tidx, i0 % 64)).store(
                            cutlass.Vector.from_elements(tuple(da_pack[q * (vector_width // 2) : (q + 1) * (vector_width // 2)]), cutlass.Int32).bitcast(
                                cfg.io_dtype
                            ),
                            alignment=16,
                        )
                    nvvm.fence_proxy("async.shared", space="cta")
                for piece in cutlass.range_constexpr(half * pieces_per_half, (half + 1) * pieces_per_half):
                    for ii in cutlass.range_constexpr(ldtm_width):
                        i = piece * ldtm_width + ii
                        d = sCumsumlog[i, 0, gate_idx] - g_j_half
                        d = d if cutlass.Int32(i) >= cg0_tidx else neg_inf
                        w_values.append(cute.math.exp2(d, fastmath=True) * beta_j)
                    da_vector = da_vectors[piece - half * pieces_per_half]
                    for m in cutlass.range_constexpr(ldtm_width // 2):
                        i = piece * ldtm_width + 2 * m
                        d_lo, d_hi = fmul2(w_values[i], w_values[i + 1], da_vector[2 * m], da_vector[2 * m + 1])
                        da_pack.append(fp32_to_fp16(d_lo, d_hi, dtype=cfg.io_dtype))
                    if cutlass.const_expr(half == 1):
                        for q in cutlass.range_constexpr(piece * vectors_per_piece, (piece + 1) * vectors_per_piece):
                            i0 = q * vector_width
                            (sDa_base + (i0 // 64) * block_elements + cg0_tidx * 64 + swizzle_xor_128b(cg0_tidx, i0 % 64)).store(
                                cutlass.Vector.from_elements(tuple(da_pack[q * (vector_width // 2) : (q + 1) * (vector_width // 2)]), cutlass.Int32).bitcast(
                                    cfg.io_dtype
                                ),
                                alignment=16,
                            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[0].arrive()

            # ---- A pass: A^T = w A_raw^T -> bf16 TMEM overlay -----------------------------------------------------
            bars.mb_araw_acc_ready[0].wait(araw_acc_index.phase)
            araw_acc_index = advance(araw_acc_index, 1)
            araw_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_score_col, cutlass.Float32), num=32)
            for piece in cutlass.range_constexpr(num_ldtms):
                if cutlass.const_expr(piece + 1 < num_ldtms):
                    araw_next = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_score_col + (piece + 1) * ldtm_width, cutlass.Float32), num=32)
                nvvm.tcgen05_wait("load")
                a_pack = []
                for m in cutlass.range_constexpr(ldtm_width // 2):
                    a_lo, a_hi = fmul2(
                        w_values[piece * ldtm_width + 2 * m], w_values[piece * ldtm_width + 2 * m + 1], araw_vector[2 * m], araw_vector[2 * m + 1]
                    )
                    a_pack.append(fp32_to_fp16(a_lo, a_hi, dtype=cfg.io_dtype))
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + tmem_score_col + piece * sttm_width, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(a_pack), cutlass.Int32),
                )
                if cutlass.const_expr(piece + 1 < num_ldtms):
                    araw_vector = araw_next
            nvvm.tcgen05_wait("store")
            bars.mb_a_ready[0].arrive()
            bars.mb_gate_done[gate_idx].arrive()

            # ---- dQ drain --------------------------------------------------------------------
            bars.mb_dk_acc_ready[0].wait(dk_ready_index.phase)
            dk_ready_index = advance(dk_ready_index, 1)
            acc_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dq_col, cutlass.Float32), num=32)
            for piece in cutlass.range_constexpr(num_dq_pieces):
                if cutlass.const_expr(piece + 1 < num_dq_pieces):
                    acc_next = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dq_col + (piece + 1) * ldtm_width, cutlass.Float32), num=32)
                nvvm.tcgen05_wait("load")
                if cutlass.const_expr(piece + 1 == num_dq_pieces):
                    bars.mb_dq_acc_done[0].arrive()
                for q in cutlass.range_constexpr(ldtm_width // vector_width):
                    c0 = piece * ldtm_width + q * vector_width
                    swizzle_offset = (c0 // 64) * block_elements + cg0_tidx * 64 + swizzle_xor_128b(cg0_tidx, c0 % 64)
                    words = []
                    for m in cutlass.range_constexpr(vector_width // 2):
                        words.append(fp32_to_fp16(acc_vector[q * vector_width + 2 * m], acc_vector[q * vector_width + 2 * m + 1], dtype=cfg.io_dtype))
                    (sDa_base + swizzle_offset).store(cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                if cutlass.const_expr(piece + 1 < num_dq_pieces):
                    acc_vector = acc_next
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_tmastg_ready[0].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()
    bars.mb_dq_tmastg_done[0].wait(dq_tmastg_index.phase)


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
    sV_raw,
    sDo_raw,
    sdK_raw,
    sD,
    sExit,
    sCumsumlog,
    sBeta,
    mD,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): the U' and dO' builds, the dU' pass and the dK drain, one token per thread."""
    if cutlass.const_expr(cfg.num_regs_compute_group_1 > cfg.launch_regs):
        nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    elif cutlass.const_expr(cfg.num_regs_compute_group_1 < cfg.launch_regs):
        nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.DECREASE)

    gate_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    dq_done_index = PipelineState.start(phase=1)
    dv_unscaled_index = PipelineState.start(phase=0)
    dk_ready_index = PipelineState.start(phase=0)
    dk_tmastg_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = tidx % cfg.threads_per_warp
    warp_in_group = cg1_tidx // cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    bytes_per_element = cfg.io_dtype.width // 8
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.d_v // ldtm_width)
    sttm_width = 16
    num_sttms = cutlass.const_expr(cfg.d_v // 2 // sttm_width)
    vector_width = 16 // bytes_per_element
    num_vectors = cutlass.const_expr(cfg.d_v // vector_width)
    num_dk_pieces = cutlass.const_expr(cfg.d_k // ldtm_width)
    words_per_vector = cutlass.const_expr(vector_width // 2)
    words_layout = cute.make_layout(words_per_vector)
    block_elements = cfg.b_t * 64
    do_stage_elements = cfg.do_cosize // cfg.smem_do_stages
    do_stage_bytes = do_stage_elements * bytes_per_element
    row_bytes = cfg.d_v * bytes_per_element
    row_swizzle_bytes = (cg1_tidx & 7) << 4
    row_swizzle_elements = (cg1_tidx & 7) * vector_width
    sdK_base = sdK_raw.data_ptr()
    sDo_base = sDo_raw.data_ptr()
    v_row_address = sV_raw.data_ptr().toint(cutlass.Int32) + cg1_tidx * row_bytes + row_swizzle_bytes
    do_address = sDo_raw.data_ptr().toint(cutlass.Int32) + cg1_tidx * row_bytes + row_swizzle_bytes
    do_step = cutlass.Int32(0)
    d_scalar = cutlass.Float32(0.0)
    zero = opaque_f32_zero()

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_dv_col = tmem_col + cfg.tmem_dv_acc_offset
    tmem_u_prime_col = tmem_col + cfg.tmem_dq_acc_offset
    tmem_dk_col = tmem_col + cfg.tmem_dk_acc_offset

    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = compute_end - write_start
        if cutlass.const_expr(cfg.has_d):
            if cutlass.const_expr(cfg.d_per_channel):
                if n_local > 0:
                    if cg1_tidx < cfg.d_v:
                        sD[cg1_tidx] = mD[head_o, cg1_tidx + v_offset].to(cutlass.Float32)
                    nvvm.barrier_cta_sync_aligned(cfg.cg1_sync_barrier_id, thread_count=num_threads_cg1)
            else:
                if n_local > 0:
                    d_scalar = mD[head_o].to(cutlass.Float32)
        for reverse_idx in cutlass.range(n_local):  # noqa: B007
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else reverse_idx > 0
            chunk_idx = compute_end - 1 - reverse_idx
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            g_last = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
            g_token = sCumsumlog[cg1_tidx, 0, gate_idx]
            beta_token = sBeta[cg1_tidx, 0, gate_idx]
            lambda_token = cute.math.exp2(g_last - g_token, fastmath=True)
            gamma_token = cute.math.exp2(g_token, fastmath=True)

            # ---- U' build: U'[j, :] = beta_j Lambda_j V[j, :] -> bf16 TMEM ----------------------
            bars.mb_v_ready[0].wait(v_index.phase)
            v_index = advance(v_index, 1)
            s_j = beta_token * lambda_token
            if have_dstate:
                bars.mb_dq_acc_done[0].wait(dq_done_index.phase)
                u_pack = []
                for q in cutlass.range_constexpr(num_vectors):
                    v_words = cute.make_tensor(
                        cute.make_ptr(cutlass.Int32, v_row_address ^ (q * 16), mem_space=cute.AddressSpace.smem, assumed_align=16), words_layout
                    ).load()
                    for w in cutlass.range_constexpr(words_per_vector):
                        v_lo, v_hi = f16x2_to_f32(v_words[w], dtype=cfg.io_dtype)
                        u0, u1 = fmul2(v_lo, v_hi, s_j, s_j)
                        u_pack.append(fp32_to_fp16(u0, u1, dtype=cfg.io_dtype))
                for piece in cutlass.range_constexpr(num_sttms):
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_u_prime_col + piece * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(u_pack[piece * sttm_width : (piece + 1) * sttm_width]), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_u_prime_ready[0].arrive()
            dq_done_index = advance(dq_done_index, 1)

            # ---- dO' build: dO'[i, :] = Gamma_i dO[i, :] -> the other dO buffer ---------------
            do_buffer = do_step & cutlass.Int32(1)
            do_prime_buffer = cutlass.Int32(1) - do_buffer
            bars.mb_do_ready[do_buffer].wait(do_index.phase)
            prior_do = (do_step + do_buffer) >> 1
            bars.mb_do_mma_done[do_prime_buffer].wait((prior_do & cutlass.Int32(1)) ^ cutlass.Int32(1))
            if cutlass.const_expr(cfg.skip_bwd_partials and cfg.d_per_channel):
                bars.mb_do_skip_done[do_prime_buffer].wait((prior_do & cutlass.Int32(1)) ^ cutlass.Int32(1))
            do_row_offset = do_buffer * do_stage_elements + cg1_tidx * cfg.d_v + row_swizzle_elements
            do_prime_row_offset = do_prime_buffer * do_stage_elements + cg1_tidx * cfg.d_v + row_swizzle_elements
            for q in cutlass.range_constexpr(num_vectors):
                slot = q * vector_width
                do_words = (sDo_base + (do_row_offset ^ slot)).load(count=vector_width, alignment=16).bitcast(cutlass.Int32)
                words = []
                for w in cutlass.range_constexpr(words_per_vector):
                    do_lo, do_hi = f16x2_to_f32(do_words[w], dtype=cfg.io_dtype)
                    d0, d1 = fmul2(do_lo, do_hi, gamma_token, gamma_token)
                    words.append(fp32_to_fp16(d0, d1, dtype=cfg.io_dtype))
                (sDo_base + (do_prime_row_offset ^ slot)).store(cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_do_prime_ready[do_prime_buffer].arrive()
            do_index = advance(do_index, cfg.smem_do_stages)
            do_step = do_step + cutlass.Int32(1)

            # ---- dU' pass: dV[j, :] = beta_j Lambda_j dU'[j, :] + D dO[j, :] -------------------
            bars.mb_dv_unscaled_ready[0].wait(dv_unscaled_index.phase)
            dv_unscaled_index = advance(dv_unscaled_index, 1)
            do_row_address = do_address + do_buffer * do_stage_bytes
            final_tile = reverse_idx == 0 and write_end == batch_num_chunks
            u_dot_lo = zero
            u_dot_hi = zero
            for i in cutlass.range_constexpr(num_ldtms):
                du_vector = cutlass.Vector.from_elements(tuple(zero for _ in range(ldtm_width)), cutlass.Float32)
                if have_dstate:
                    du_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dv_col + i * ldtm_width, cutlass.Float32), num=32)
                    nvvm.tcgen05_wait("load")
                if cutlass.const_expr(cfg.use_dstate_in):
                    # ---- <U', dU'> row dot of the final tile, packed pairs ----
                    if final_tile:
                        for q in cutlass.range_constexpr(ldtm_width // vector_width):
                            p0 = i * ldtm_width + q * vector_width
                            v_words = cute.make_tensor(
                                cute.make_ptr(cutlass.Int32, v_row_address ^ (p0 * bytes_per_element), mem_space=cute.AddressSpace.smem, assumed_align=16),
                                words_layout,
                            ).load()
                            for w in cutlass.range_constexpr(words_per_vector):
                                v_lo, v_hi = f16x2_to_f32(v_words[w], dtype=cfg.io_dtype)
                                u_dot_lo, u_dot_hi = ffma2(
                                    v_lo, v_hi, du_vector[q * vector_width + 2 * w], du_vector[q * vector_width + 2 * w + 1], u_dot_lo, u_dot_hi
                                )
                dv_values = []
                for m in cutlass.range_constexpr(ldtm_width // 2):
                    d0, d1 = fmul2(du_vector[2 * m], du_vector[2 * m + 1], s_j, s_j)
                    dv_values += [d0, d1]
                if cutlass.const_expr(cfg.has_d):
                    for q in cutlass.range_constexpr(ldtm_width // vector_width):
                        p0 = i * ldtm_width + q * vector_width
                        do_words = cute.make_tensor(
                            cute.make_ptr(cutlass.Int32, do_row_address ^ (p0 * bytes_per_element), mem_space=cute.AddressSpace.smem, assumed_align=16),
                            words_layout,
                        ).load()
                        for w in cutlass.range_constexpr(words_per_vector):
                            do_lo, do_hi = f16x2_to_f32(do_words[w], dtype=cfg.io_dtype)
                            if cutlass.const_expr(cfg.d_per_channel):
                                d_lo = sD[p0 + 2 * w]
                                d_hi = sD[p0 + 2 * w + 1]
                            else:
                                d_lo = d_scalar
                                d_hi = d_scalar
                            dv_values[q * vector_width + 2 * w], dv_values[q * vector_width + 2 * w + 1] = ffma2(
                                d_lo, d_hi, do_lo, do_hi, dv_values[q * vector_width + 2 * w], dv_values[q * vector_width + 2 * w + 1]
                            )
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + tmem_dv_col + i * ldtm_width, cutlass.Float32),
                    cutlass.Vector.from_elements(tuple(dv_values), cutlass.Float32),
                )
            nvvm.tcgen05_wait("store")
            if cutlass.const_expr(cfg.use_dstate_in):
                if final_tile:
                    u_dot_du = (u_dot_lo + u_dot_hi) * s_j
                    for offset in [16, 8, 4, 2, 1]:
                        u_dot_du = u_dot_du + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, u_dot_du, offset, 31, kind=nvvm.Shfl.BFLY))
                    if lane_idx == 0:
                        sExit[8 + warp_in_group] = u_dot_du
            bars.mb_dv_scaled_done[0].arrive()
            bars.mb_v_cg1_done[0].arrive()

            # ---- dK drain --------------------------------------------------------------------
            bars.mb_dk_acc_ready[0].wait(dk_ready_index.phase)
            dk_ready_index = advance(dk_ready_index, 1)
            acc_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dk_col, cutlass.Float32), num=32)
            for piece in cutlass.range_constexpr(num_dk_pieces):
                if cutlass.const_expr(piece + 1 < num_dk_pieces):
                    acc_next = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dk_col + (piece + 1) * ldtm_width, cutlass.Float32), num=32)
                nvvm.tcgen05_wait("load")
                if cutlass.const_expr(piece == 0):
                    bars.mb_dk_tmastg_done[0].wait(dk_tmastg_index.phase)
                    dk_tmastg_index = advance(dk_tmastg_index, 1)
                for q in cutlass.range_constexpr(ldtm_width // vector_width):
                    c0 = piece * ldtm_width + q * vector_width
                    swizzle_offset = (c0 // 64) * block_elements + cg1_tidx * 64 + swizzle_xor_128b(cg1_tidx, c0 % 64)
                    words = []
                    for m in cutlass.range_constexpr(vector_width // 2):
                        words.append(fp32_to_fp16(acc_vector[q * vector_width + 2 * m], acc_vector[q * vector_width + 2 * m + 1], dtype=cfg.io_dtype))
                    (sdK_base + swizzle_offset).store(cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                if cutlass.const_expr(piece + 1 < num_dk_pieces):
                    acc_vector = acc_next
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dk_tmastg_ready[0].arrive()
            bars.mb_gate_done[gate_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()
    bars.mb_dk_tmastg_done[0].wait(dk_tmastg_index.phase)


@cute.jit
def compute2_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    warp_idx,
    tmem_base_slot,
    sState_raw,
    sDstate_raw,
    sdV_raw,
    sV_raw,
    sDo_raw,
    sExit,
    sCumsumlog,
    mDstate_in,
    mDstate0,
    mExit,
    mSkipPart,
    sScheduler,
    bars,
):
    """Compute warp-group 2 role (warps 8-11): the D-skip partials, the dstate pass (the dS seed, the exit dot,
    the bf16 SMEM copy, the gamma pre-scale, the dS_0 store; one d_k row per thread) and the dV drain, one token per
    thread."""
    if cutlass.const_expr(cfg.num_regs_compute_group_2 > cfg.launch_regs):
        nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.INCREASE)
    elif cutlass.const_expr(cfg.num_regs_compute_group_2 < cfg.launch_regs):
        nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.DECREASE)
    gate_index = PipelineState.start(phase=0)
    da_acc_index = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0)
    state_index = PipelineState.start(phase=0)
    dv_scaled_index = PipelineState.start(phase=0)
    dv_ready_index = PipelineState.start(phase=0)
    dv_tmastg_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    do_step = cutlass.Int32(0)
    num_threads_cg2 = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    cg2_tidx = tidx % num_threads_cg2
    lane_idx = tidx % cfg.threads_per_warp
    warp_in_group = cg2_tidx // cfg.threads_per_warp
    elect_one = nvvm.elect_sync()
    ldtm_width = 32
    bytes_per_element = cfg.io_dtype.width // 8
    vector_width = 16 // bytes_per_element
    num_vectors = cutlass.const_expr(cfg.d_v // vector_width)
    box_elements = 128 // bytes_per_element
    block_elements = cfg.b_t * 64
    num_dv_pieces = cutlass.const_expr(cfg.d_v // ldtm_width)
    sdV_base = sdV_raw.data_ptr()
    sState_base = sState_raw.data_ptr()
    sDstate_base = sDstate_raw.data_ptr()
    row_bytes = cfg.d_v * bytes_per_element
    do_stage_bytes = (cfg.do_cosize // cfg.smem_do_stages) * bytes_per_element
    v_base = sV_raw.data_ptr().toint(cutlass.Int32) + warp_in_group * (32 * row_bytes)
    do_base = sDo_raw.data_ptr().toint(cutlass.Int32) + warp_in_group * (32 * row_bytes)
    lane_word_offset = lane_idx * 4
    word_layout = cute.make_layout(1)
    zero = opaque_f32_zero()
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_dstate_col = tmem_col + cfg.tmem_dstate_acc_offset
    tmem_dv_col = tmem_col + cfg.tmem_dv_acc_offset
    tmem_score_col = tmem_col + cfg.tmem_score_acc_offset
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = compute_end - write_start
        if n_local > 0:
            # ---- dS seed: d_final_state or zeros -> TMEM + bf16 SMEM ------------------------
            if cutlass.const_expr(cfg.use_dstate_in):
                seed_values = []
                seed_true = compute_end == batch_num_chunks
                gDstate_in = mDstate_in[None, None, head_o, batch_idx]
                for seed_p in cutlass.range_constexpr(cfg.d_v):
                    seed_value = gDstate_in[seed_p + v_offset, cg2_tidx].to(cutlass.Float32)
                    seed_values.append(seed_value if seed_true else zero)
                bars.mb_gate_ready[gate_index.idx].wait(gate_index.phase)
                gamma_first = cute.math.exp2(sCumsumlog[cfg.b_t - 1, 0, gate_index.idx], fastmath=True)
                seed_acc = [seed_values[seed_p] * gamma_first for seed_p in range(cfg.d_v)]
                for seed_i in cutlass.range_constexpr(num_dv_pieces):
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_col + seed_i * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(seed_acc[seed_i * ldtm_width : (seed_i + 1) * ldtm_width]), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_scaled[0].arrive()
                for seed_q in cutlass.range_constexpr(num_vectors):
                    seed_words = []
                    for seed_m in cutlass.range_constexpr(vector_width // 2):
                        seed_p = seed_q * vector_width + 2 * seed_m
                        seed_words.append(fp32_to_fp16(seed_values[seed_p], seed_values[seed_p + 1], dtype=cfg.io_dtype))
                    (sDstate_base + cg2_tidx * cfg.d_v + swizzle_xor_128b(cg2_tidx, seed_q * vector_width)).store(
                        cutlass.Vector.from_elements(tuple(seed_words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                    )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready[0].arrive()

            for reverse_idx in cutlass.range(n_local):  # noqa: B007
                chunk_idx = compute_end - 1 - reverse_idx
                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                g_last = sCumsumlog[cfg.b_t - 1, 0, gate_idx]

                # ---- D-skip partials: per head the warp's 32 diagonal entries of dA^T = V dO^T (the per-token <x_j, dO_j>),
                # per channel the raw-row dots of the warp's 32 tokens, one channel pair per lane; one row per warp per slot ----
                if cutlass.const_expr(cfg.skip_bwd_partials):
                    skip_row = (batch_start // cutlass.Int32(cfg.b_t) + 2 * batch_idx + chunk_idx) * 4 + warp_in_group
                    if cutlass.const_expr(cfg.d_per_channel):
                        bars.mb_v_ready[0].wait(v_index.phase)
                        v_index = advance(v_index, 1)
                        do_buffer = do_step & cutlass.Int32(1)
                        bars.mb_do_ready[do_buffer].wait(do_index.phase)
                        do_index = advance(do_index, cfg.smem_do_stages)
                        do_step = do_step + cutlass.Int32(1)
                        do_rows = do_base + do_buffer * do_stage_bytes
                        skip_lo = zero
                        skip_hi = zero
                        for i in cutlass.range_constexpr(32):
                            word_offset = i * row_bytes + (lane_word_offset ^ ((i & 7) << 4))
                            x_word = cute.make_tensor(
                                cute.make_ptr(cutlass.Int32, v_base + word_offset, mem_space=cute.AddressSpace.smem, assumed_align=4), word_layout
                            ).load()
                            do_word = cute.make_tensor(
                                cute.make_ptr(cutlass.Int32, do_rows + word_offset, mem_space=cute.AddressSpace.smem, assumed_align=4), word_layout
                            ).load()
                            x_lo, x_hi = f16x2_to_f32(x_word[0], dtype=cfg.io_dtype)
                            do_lo, do_hi = f16x2_to_f32(do_word[0], dtype=cfg.io_dtype)
                            skip_lo, skip_hi = ffma2(do_lo, do_hi, x_lo, x_hi, skip_lo, skip_hi)
                        bars.mb_v_cg1_done[0].arrive()
                        bars.mb_do_skip_done[do_buffer].arrive()
                        if chunk_idx < write_end:
                            mSkipPart[skip_row, head_o, 2 * lane_idx + v_offset] = skip_lo
                            mSkipPart[skip_row, head_o, 2 * lane_idx + 1 + v_offset] = skip_hi
                    else:
                        bars.mb_da_acc_ready[0].wait(da_acc_index.phase)
                        da_acc_index = advance(da_acc_index, 1)
                        diagonal_block = nvvm.tcgen05_ld(
                            "32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_score_col + warp_in_group * ldtm_width, cutlass.Float32), num=32
                        )
                        nvvm.tcgen05_wait("load")
                        bars.mb_da_acc_done[0].arrive()
                        values = [diagonal_block[i] for i in range(32)]
                        for bit in cutlass.range_constexpr(5):
                            take_high = (lane_idx & cutlass.Int32(1 << bit)) != 0
                            folded = []
                            for i in cutlass.range_constexpr(16 >> bit):
                                picked = values[2 * i + 1] if take_high else values[2 * i]
                                folded.append(picked)
                            values = folded
                        diagonal_sum = values[0]
                        for offset in [16, 8, 4, 2, 1]:
                            diagonal_sum = diagonal_sum + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, diagonal_sum, offset, 31, kind=nvvm.Shfl.BFLY))
                        if lane_idx == 0 and chunk_idx < write_end:
                            mSkipPart[skip_row, head_o] = diagonal_sum

                # ---- dstate pass: exit dot, bf16 SMEM copy, pre-scale ----------------------------
                bars.mb_state_ready[0].wait(state_index.phase)
                state_index = advance(state_index, 1)
                final_tile = reverse_idx == 0 and write_end == batch_num_chunks
                state_panel = sState_base + (cg2_tidx // box_elements) * (cfg.d_v * box_elements)
                state_col = cg2_tidx % box_elements
                final_dot = zero
                if cutlass.const_expr(cfg.use_dstate_in):
                    # ---- final-state dot of the final tile, ahead of the dS wait, the 64 seed loads in one burst ----
                    if final_tile:
                        gDstate_final = mDstate_in[None, None, head_o, batch_idx]
                        final_values = []
                        for p in cutlass.range_constexpr(cfg.d_v):
                            final_values.append(gDstate_final[p + v_offset, cg2_tidx].to(cutlass.Float32))
                        final_lo = zero
                        final_hi = zero
                        for p in cutlass.range_constexpr(0, cfg.d_v, 2):
                            s_lo = (state_panel + p * box_elements + swizzle_xor_128b(p, state_col)).load().to(cutlass.Float32)
                            s_hi = (state_panel + (p + 1) * box_elements + swizzle_xor_128b(p + 1, state_col)).load().to(cutlass.Float32)
                            final_lo, final_hi = ffma2(final_values[p], final_values[p + 1], s_lo, s_hi, final_lo, final_hi)
                        final_dot = final_lo + final_hi
                bars.mb_dstate_acc_ready[0].wait(dstate_acc_index.phase)
                dstate_acc_index = advance(dstate_acc_index, 1)
                dstate_vectors = []
                for i in cutlass.range_constexpr(num_dv_pieces):
                    dstate_vectors.append(
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_col + i * ldtm_width, cutlass.Float32), num=32)
                    )
                nvvm.tcgen05_wait("load")
                exit_lo = zero
                exit_hi = zero
                for p in cutlass.range_constexpr(0, cfg.d_v, 2):
                    s_lo = (state_panel + p * box_elements + swizzle_xor_128b(p, state_col)).load().to(cutlass.Float32)
                    s_hi = (state_panel + (p + 1) * box_elements + swizzle_xor_128b(p + 1, state_col)).load().to(cutlass.Float32)
                    exit_lo, exit_hi = ffma2(
                        dstate_vectors[p // ldtm_width][p % ldtm_width], dstate_vectors[p // ldtm_width][p % ldtm_width + 1], s_lo, s_hi, exit_lo, exit_hi
                    )
                exit_dot = exit_lo + exit_hi
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_state_cg2_done[0].arrive()
                if cutlass.const_expr(cfg.use_dstate_in):
                    final_dot = final_dot * cute.math.exp2(g_last, fastmath=True)
                for offset in [16, 8, 4, 2, 1]:
                    exit_dot = exit_dot + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, exit_dot, offset, 31, kind=nvvm.Shfl.BFLY))
                    final_dot = final_dot + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, final_dot, offset, 31, kind=nvvm.Shfl.BFLY))
                if lane_idx == 0:
                    sExit[warp_in_group] = exit_dot
                    sExit[4 + warp_in_group] = final_dot
                if cutlass.const_expr(cfg.use_dstate_in):
                    bars.mb_dv_scaled_done[0].wait(dv_scaled_index.phase)
                    dv_scaled_index = advance(dv_scaled_index, 1)
                nvvm.barrier_cta_sync_aligned(cfg.cg2_sync_barrier_id, thread_count=num_threads_cg2)
                if cg2_tidx == 0:
                    exit_slot = batch_start // cutlass.Int32(cfg.b_t) + 2 * batch_idx + chunk_idx
                    if chunk_idx < write_end:
                        mExit[exit_slot, head_o] = ((sExit[0] + sExit[1]) + sExit[2]) + sExit[3]
                    if final_tile:
                        final_sum = ((sExit[4] + sExit[5]) + sExit[6]) + sExit[7]
                        if cutlass.const_expr(cfg.use_dstate_in):
                            final_sum = final_sum + (((sExit[8] + sExit[9]) + sExit[10]) + sExit[11])
                        mExit[exit_slot + 1, head_o] = final_sum
                nvvm.barrier_cta_sync_aligned(cfg.cg2_sync_barrier_id, thread_count=num_threads_cg2)

                if chunk_idx > write_start:
                    for q in cutlass.range_constexpr(num_vectors):
                        words = []
                        for m in cutlass.range_constexpr(vector_width // 2):
                            p = q * vector_width + 2 * m
                            words.append(
                                fp32_to_fp16(
                                    dstate_vectors[p // ldtm_width][p % ldtm_width], dstate_vectors[p // ldtm_width][p % ldtm_width + 1], dtype=cfg.io_dtype
                                )
                            )
                        (sDstate_base + cg2_tidx * cfg.d_v + swizzle_xor_128b(cg2_tidx, q * vector_width)).store(
                            cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                        )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_dstate_smem_ready[0].arrive()

                    # ---- pre-scale by gamma_{c-1} ----------------------------------------------------
                    bars.mb_gate_ready[gate_index.idx].wait(gate_index.phase)
                    gamma_previous = cute.math.exp2(sCumsumlog[cfg.b_t - 1, 0, gate_index.idx], fastmath=True)
                    for i in cutlass.range_constexpr(num_dv_pieces):
                        scaled = []
                        for m in cutlass.range_constexpr(ldtm_width // 2):
                            s0, s1 = fmul2(dstate_vectors[i][2 * m], dstate_vectors[i][2 * m + 1], gamma_previous, gamma_previous)
                            scaled += [s0, s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_col + i * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_dstate_scaled[0].arrive()
                elif cutlass.const_expr(cfg.use_dstate0):
                    # ---- dS_0 store ------------------------------------------------------------------
                    dstate_destination = mWorkItems[tile_idx, WORK_ITEM_DSTATE_DST]
                    if chunk_idx == 0 and dstate_destination >= 0:
                        gDstate0 = mDstate0[None, None, head_o, dstate_destination]
                        for p in cutlass.range_constexpr(cfg.d_v):
                            gDstate0[p + v_offset, cg2_tidx] = dstate_vectors[p // ldtm_width][p % ldtm_width].to(mDstate0.element_type)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_gate_done[gate_idx].arrive()

                # ---- dV drain ----------------------------------------------------------------------
                bars.mb_dv_tmastg_done[0].wait(dv_tmastg_index.phase)
                dv_tmastg_index = advance(dv_tmastg_index, 1)
                bars.mb_dv_acc_ready[0].wait(dv_ready_index.phase)
                dv_ready_index = advance(dv_ready_index, 1)
                acc_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dv_col, cutlass.Float32), num=32)
                for piece in cutlass.range_constexpr(num_dv_pieces):
                    if cutlass.const_expr(piece + 1 < num_dv_pieces):
                        acc_next = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dv_col + (piece + 1) * ldtm_width, cutlass.Float32), num=32)
                    nvvm.tcgen05_wait("load")
                    if cutlass.const_expr(piece + 1 == num_dv_pieces):
                        bars.mb_dv_acc_done[0].arrive()
                    for q in cutlass.range_constexpr(ldtm_width // vector_width):
                        c0 = piece * ldtm_width + q * vector_width
                        swizzle_offset = (c0 // 64) * block_elements + cg2_tidx * 64 + swizzle_xor_128b(cg2_tidx, c0 % 64)
                        words = []
                        for m in cutlass.range_constexpr(vector_width // 2):
                            words.append(fp32_to_fp16(acc_vector[q * vector_width + 2 * m], acc_vector[q * vector_width + 2 * m + 1], dtype=cfg.io_dtype))
                        (sdV_base + swizzle_offset).store(cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                    if cutlass.const_expr(piece + 1 < num_dv_pieces):
                        acc_vector = acc_next
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dv_tmastg_ready[0].arrive()
        elif cutlass.const_expr(cfg.use_dstate0):
            # ---- zero-length sequence: dS_0 = d_final_state or zeros ----------------------------
            dstate_destination = mWorkItems[tile_idx, WORK_ITEM_DSTATE_DST]
            if dstate_destination >= 0:
                gDstate0 = mDstate0[None, None, head_o, dstate_destination]
                if cutlass.const_expr(cfg.use_dstate_in):
                    gDstate_in = mDstate_in[None, None, head_o, batch_idx]
                    for p in cutlass.range_constexpr(cfg.d_v):
                        gDstate0[p + v_offset, cg2_tidx] = gDstate_in[p + v_offset, cg2_tidx].to(mDstate0.element_type)
                else:
                    for p in cutlass.range_constexpr(cfg.d_v):
                        gDstate0[p + v_offset, cg2_tidx] = cutlass.Float32(0.0).to(mDstate0.element_type)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)
    bars.mb_tmem_done[0].arrive()
    bars.mb_dv_tmastg_done[0].wait(dv_tmastg_index.phase)


@cute.kernel
def frost_mamba2_bprop(
    cfg: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA: cute.Tensor,
    mDt_bias: Optional[cute.Tensor],
    mD: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    mDstate_in: Optional[cute.Tensor],
    mDstate0: Optional[cute.Tensor],
    mExit: cute.Tensor,
    mSkipPart: Optional[cute.Tensor],
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    mTransition: Optional[cute.Tensor],
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    mQ,
    mK,
    mV,
    mDo,
    mDq,
    mDk,
    mDv,
    mCheckpoints,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main Mamba-2 chunked backward kernel: warp-specialized persistent tile loop over the reverse chunk walk."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]

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

    SWIZZLE_LAYOUT = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
    STATE_LEAD = cfg.d_v * 128
    DSTATE_LEAD = cfg.d_k * 128
    sQ_raw = cutlass.Array(cfg.io_dtype, cfg.q_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ = SmemTile(base=sQ_raw.data_ptr(), elems_per_stage=cfg.q_cosize, stages=1, leading_byte_offset=LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT)
    sQ_trans = SmemTile(
        base=sQ_raw.data_ptr(), elems_per_stage=cfg.q_cosize, stages=1, leading_byte_offset=KT_LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sK_raw = cutlass.Array(cfg.io_dtype, cfg.k_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK = SmemTile(base=sK_raw.data_ptr(), elems_per_stage=cfg.k_cosize, stages=1, leading_byte_offset=LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT)
    sK_trans = SmemTile(
        base=sK_raw.data_ptr(), elems_per_stage=cfg.k_cosize, stages=1, leading_byte_offset=KT_LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sV_raw = cutlass.Array(cfg.io_dtype, cfg.v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sV = SmemTile(base=sV_raw.data_ptr(), elems_per_stage=cfg.v_cosize, stages=1, leading_byte_offset=LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT)
    sDo_raw = cutlass.Array(cfg.io_dtype, cfg.do_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDo = SmemTile(
        base=sDo_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
    sDo_trans = SmemTile(
        base=sDo_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
    sState_raw = cutlass.Array(cfg.io_dtype, cfg.state_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sState = SmemTile(
        base=sState_raw.data_ptr(), elems_per_stage=cfg.state_cosize, stages=1, leading_byte_offset=STATE_LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sDstate_raw = cutlass.Array(cfg.io_dtype, cfg.dstate_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDstate = SmemTile(
        base=sDstate_raw.data_ptr(), elems_per_stage=cfg.dstate_cosize, stages=1, leading_byte_offset=LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sDstate_trans = SmemTile(
        base=sDstate_raw.data_ptr(),
        elems_per_stage=cfg.dstate_cosize,
        stages=1,
        leading_byte_offset=DSTATE_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
    sdK_raw = cutlass.Array(cfg.io_dtype, cfg.dk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sdQ_raw = cutlass.Array(cfg.io_dtype, cfg.dq_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDa = SmemTile(base=sdQ_raw.data_ptr(), elems_per_stage=cfg.dq_cosize, stages=1, leading_byte_offset=LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT)
    sDa_trans = SmemTile(
        base=sdQ_raw.data_ptr(), elems_per_stage=cfg.dq_cosize, stages=1, leading_byte_offset=KT_LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sdV_raw = cutlass.Array(cfg.io_dtype, cfg.dv_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=16)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=16)
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumsumlog_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=16),
        cumsumlog_smem_layout_staged,
    )
    sBeta = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, beta_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=16),
        beta_smem_layout_staged,
    )
    sD = cutlass.Array(cutlass.Float32, cfg.d_v, space=SMEM, alignment=16)
    sExit = cutlass.Array(cutlass.Float32, 12, space=SMEM, alignment=16)

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_mma_done[s].init()
        bars.mb_v_cg1_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
        bars.mb_do_skip_done[s].init()
        bars.mb_do_prime_ready[s].init()
        bars.mb_do_prime_mma_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for b in (
        bars.mb_state_ready,
        bars.mb_state_mma_done,
        bars.mb_state_cg2_done,
        bars.mb_da_acc_ready,
        bars.mb_da_acc_done,
        bars.mb_da_ready,
        bars.mb_araw_acc_ready,
        bars.mb_a_ready,
        bars.mb_u_prime_ready,
        bars.mb_dstate_smem_ready,
        bars.mb_dstate_scaled,
        bars.mb_dstate_acc_ready,
        bars.mb_dv_unscaled_ready,
        bars.mb_dv_scaled_done,
        bars.mb_dq_acc_done,
        bars.mb_dk_acc_ready,
        bars.mb_dv_acc_ready,
        bars.mb_dv_acc_done,
        bars.mb_dq_tmastg_ready,
        bars.mb_dq_tmastg_done,
        bars.mb_dk_tmastg_ready,
        bars.mb_dk_tmastg_done,
        bars.mb_dv_tmastg_ready,
        bars.mb_dv_tmastg_done,
        bars.mb_tmem_done,
    ):
        b[0].init()
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()

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
            warp_idx,
            tmem_base_slot=tmem_base_slot,
            sDa_raw=sdQ_raw,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            sScheduler=sScheduler,
            bars=bars,
        )
    elif warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
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
            sV_raw=sV_raw,
            sDo_raw=sDo_raw,
            sdK_raw=sdK_raw,
            sD=sD,
            sExit=sExit,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            mD=mD,
            sScheduler=sScheduler,
            bars=bars,
        )
    elif warp_idx >= cfg.compute_group_2_warp_ids[0] and warp_idx <= cfg.compute_group_2_warp_ids[-1]:
        compute2_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            warp_idx=warp_idx,
            tmem_base_slot=tmem_base_slot,
            sState_raw=sState_raw,
            sDstate_raw=sDstate_raw,
            sdV_raw=sdV_raw,
            sV_raw=sV_raw,
            sDo_raw=sDo_raw,
            sExit=sExit,
            sCumsumlog=sCumsumlog,
            mDstate_in=mDstate_in,
            mDstate0=mDstate0,
            mExit=mExit,
            mSkipPart=mSkipPart,
            sScheduler=sScheduler,
            bars=bars,
        )
    elif warp_idx == cfg.load_gate_warp_id:
        gate_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx=tidx,
            mGate=mGate,
            mA=mA,
            mDt_bias=mDt_bias,
            dt_min=dt_min,
            dt_max=dt_max,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            mTransition=mTransition,
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
            sV=sV,
            sDo=sDo,
            sDo_trans=sDo_trans,
            sState=sState,
            sDstate=sDstate,
            sDstate_trans=sDstate_trans,
            sDa=sDa,
            sDa_trans=sDa_trans,
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
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            sDo_raw=sDo_raw,
            sState_raw=sState_raw,
            tidx=tidx,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            desc_do_base=desc_do_base,
            desc_checkpoint_base=desc_checkpoint_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
        )
    elif warp_idx == cfg.epilogue_warp_id:
        tmastg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx=tidx,
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
class Mamba2BpropCfg:
    """Per-compile Mamba-2 backward kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, the state-gradient ports, the D skip) are the ``cute.compile`` cache keys;
    the rest is derived from the module-global ``CFG`` constants.  ``host`` stamps the shape-derived fields at trace
    time.  Passed ``cfg``-first (a ``cutlass.Constexpr``) into ``host`` / ``kernel`` and every warp body.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    use_dstate_in: bool
    use_dstate0: bool
    d_k: int
    d_v: int
    dt_softplus: bool = True
    has_d: bool = False
    d_per_channel: bool = False
    skip_bwd_partials: bool = False
    tiles_per_head: int = 1
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    compute_group_2_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_2_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0
    cg1_sync_barrier_id: int = 2
    cg2_sync_barrier_id: int = 3
    launch_regs: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_do_stages: int = CFG.SMEM_DO_STAGES
    smem_state_stages: int = CFG.SMEM_STATE_STAGES
    smem_dstate_stages: int = CFG.SMEM_DSTATE_STAGES
    smem_dq_stages: int = CFG.SMEM_DQ_STAGES
    smem_dk_stages: int = CFG.SMEM_DK_STAGES
    smem_dv_stages: int = CFG.SMEM_DV_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_dstate_acc_offset: int = 0
    tmem_dv_acc_offset: int = 0
    tmem_dk_acc_offset: int = 0
    tmem_dq_acc_offset: int = 0
    tmem_score_acc_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    do_cosize: int = 0
    state_cosize: int = 0
    dstate_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dv_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_do_bytes: int = 0
    tma_state_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_dstate_in: bool,
    use_dstate0: bool,
    dt_softplus: bool = True,
    has_d: bool = False,
    d_per_channel: bool = False,
    skip_bwd_partials: bool = False,
    d_k: int,
    d_v: int,
) -> Mamba2BpropCfg:
    """Build the per-compile ``Mamba2BpropCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32)."""
    cfg = Mamba2BpropCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        dt_softplus=dt_softplus,
        has_d=has_d,
        d_per_channel=d_per_channel,
        skip_bwd_partials=skip_bwd_partials,
        d_k=d_k,
        d_v=d_v,
    )
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    n_cg2 = len(cfg.compute_group_2_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1 + n_cg2)
    cfg.tmem_user_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1 + n_cg2)
    cfg.launch_regs = 65536 // cfg.threads_per_cta
    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dv_acc_offset = cfg.tmem_dstate_acc_offset + cfg.d_v
    cfg.tmem_dk_acc_offset = cfg.tmem_dv_acc_offset + cfg.d_v
    cfg.tmem_dq_acc_offset = cfg.tmem_dk_acc_offset + cfg.d_k
    cfg.tmem_score_acc_offset = cfg.tmem_dq_acc_offset + cfg.d_k
    assert cfg.tmem_score_acc_offset + cfg.b_t <= 512, "the backward TMEM map needs d_k = 128 and d_v <= 64"
    return cfg


TENSORMAP_DESC_ARRAYS = 8  # per-batch runtime TMA descriptors: Q, K, V, dO, checkpoints, dQ, dK, dV


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
    do: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array, inside the prologue kernel after its order pass."""
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
        emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        emit_seq_descs(base_v, sub2, cu_seqlens, v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        emit_seq_descs(base_do, sub3, cu_seqlens, do, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        emit_checkpoint_seq_descs(base_checkpoint, sub4, cu_seqlens, state_checkpoints, n_batch, cutlass.Int32(b_t), 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 5:
        emit_seq_descs(base_dq, sub5, cu_seqlens, dq, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 6:
        emit_seq_descs(base_dk, sub6, cu_seqlens, dk, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 7:
        emit_seq_descs(base_dv, sub7, cu_seqlens, dv, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_mamba2_bprop_prologue(
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
    do: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
    mSlots: cute.Tensor,
) -> None:
    """Two-CTA prologue: block 0 LPT-orders the work-item table and zeroes the scheduler rings via
    :func:`order_body`, block 1 builds the per-batch TMA-descriptor arrays via :func:`build_descs_body`,
    one warp per array, then the gate backward's exit-slot table."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(q.shape[1] if q.shape[1] >= do.shape[1] else do.shape[1])
        order_body(
            order_gen,
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
    else:
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
            do,
            state_checkpoints,
            dq,
            dk,
            dv,
            n_batch,
            b_t,
        )
        exit_slot_table_body(b_t, tidx, ORDER_THREADS, n_batch, cu_seqlens, mSlots)


@cute.jit
def base_descriptors(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    state_checkpoints: cute.Tensor,
):
    """The whole-tensor TMA descriptors (Q, K, V, dO, checkpoints, dQ, dK, dV) the prologue's per-batch arrays are built
    from."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    heads_out = h_q if h_q >= do.shape[1] else do.shape[1]
    d_v = do.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    bt = b_t
    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q.stride[0], q.stride[1], 1)))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k.stride[0], k.stride[1], 1)))
    do_headed = cute.make_tensor(do.iterator, cute.make_layout((d_v, heads_out, seqlen), stride=(1, do.stride[1], do.stride[0])))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_do = tma.create_tensor_map_tiled_from_view(do_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, v.shape[1], seqlen), stride=(1, v.stride[1], v.stride[0])))
    base_desc_v = tma.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((seqlen, dq.shape[1], d_k), stride=(dq.stride[0], dq.stride[1], 1)))
    base_desc_dq = tma.create_tensor_map_tiled_from_view(dq_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((seqlen, dk.shape[1], d_k), stride=(dk.stride[0], dk.stride[1], 1)))
    base_desc_dk = tma.create_tensor_map_tiled_from_view(dk_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((d_v, heads_out, seqlen), stride=(1, dv.stride[1], dv.stride[0])))
    base_desc_dv = tma.create_tensor_map_tiled_from_view(dv_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    d_v_state = state_checkpoints.shape[2]
    d_k_state = state_checkpoints.shape[3]
    checkpoint_box_elements = 128 // (state_checkpoints.element_type.width // 8)
    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (d_k_state, d_v_state, state_checkpoints.shape[0], heads_out),
            stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
        ),
    )
    base_desc_checkpoint = tma.create_tensor_map_tiled_from_view(
        checkpoint_view, box_dims=(checkpoint_box_elements, d_v_state, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle_128b
    )
    return base_desc_q, base_desc_k, base_desc_v, base_desc_do, base_desc_checkpoint, base_desc_dq, base_desc_dk, base_desc_dv


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_checkpoints: cute.Tensor,
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    exit_slots: cute.Tensor,
    stream: cuda.CUstream,
):
    """One-launch prologue: LPT-order the work items, build the per-(b,h) TMA-descriptor arrays
    (Q, K, V, dO, checkpoints, dQ, dK, dV) into ``tensormap_workspace`` and write the gate backward's exit-slot table."""
    batch_size = cu_seqlens.shape[0] - 1
    base_desc_q, base_desc_k, base_desc_v, base_desc_do, base_desc_checkpoint, base_desc_dq, base_desc_dk, base_desc_dv = base_descriptors(
        io_dtype, b_t, q, k, v, do, dq, dk, dv, state_checkpoints
    )
    frost_mamba2_bprop_prologue(
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
        do,
        state_checkpoints,
        dq,
        dk,
        dv,
        work_item_staging,
        work_count,
        work_items,
        scheduler_counter,
        cutlass.Int32(batch_size),
        exit_slots,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    state_checkpoints: cute.Tensor,
    exit_dots: cute.Tensor,
    skip_bwd_part: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    dstate0: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    scheduler_counter: cute.Tensor,
    transition: Optional[cute.Tensor],
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = cutlass.Int32(gate.shape[1])
    q_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(q.shape[1]))
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))

    # ---- head-major views: (tokens, dim, heads) q / k / dq / dk, (dim, tokens, heads) v / dO / dv, (V, K, heads, seqs) states ----
    q = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], q.shape[2], q.shape[1]), stride=(q.stride[0], q.stride[2], q.stride[1])))
    k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[2], k.shape[1]), stride=(k.stride[0], k.stride[2], k.stride[1])))
    dq = cute.make_tensor(dq.iterator, cute.make_layout((dq.shape[0], dq.shape[2], dq.shape[1]), stride=(dq.stride[0], dq.stride[2], dq.stride[1])))
    dk = cute.make_tensor(dk.iterator, cute.make_layout((dk.shape[0], dk.shape[2], dk.shape[1]), stride=(dk.stride[0], dk.stride[2], dk.stride[1])))
    v = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[0], v.shape[1]), stride=(v.stride[2], v.stride[0], v.stride[1])))
    do = cute.make_tensor(do.iterator, cute.make_layout((do.shape[2], do.shape[0], do.shape[1]), stride=(do.stride[2], do.stride[0], do.stride[1])))
    dv = cute.make_tensor(dv.iterator, cute.make_layout((dv.shape[2], dv.shape[0], dv.shape[1]), stride=(dv.stride[2], dv.stride[0], dv.stride[1])))
    if cutlass.const_expr(dstate_in is not None):
        dstate_in = cute.make_tensor(
            dstate_in.iterator,
            cute.make_layout(
                (dstate_in.shape[2], dstate_in.shape[3], dstate_in.shape[1], dstate_in.shape[0]),
                stride=(dstate_in.stride[2], dstate_in.stride[3], dstate_in.stride[1], dstate_in.stride[0]),
            ),
        )
    if cutlass.const_expr(dstate0 is not None):
        dstate0 = cute.make_tensor(
            dstate0.iterator,
            cute.make_layout(
                (dstate0.shape[2], dstate0.shape[3], dstate0.shape[1], dstate0.shape[0]),
                stride=(dstate0.stride[2], dstate0.stride[3], dstate0.stride[1], dstate0.stride[0]),
            ),
        )

    # ---- SMEM sizing: per-buffer element cosizes -------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    q_tile_elements = cfg.b_t * cfg.d_k
    v_tile_elements = cfg.d_v * cfg.b_t
    cfg.q_cosize = q_tile_elements * cfg.smem_q_stages
    cfg.k_cosize = q_tile_elements * cfg.smem_k_stages
    cfg.v_cosize = v_tile_elements * cfg.smem_v_stages
    cfg.do_cosize = v_tile_elements * cfg.smem_do_stages
    cfg.state_cosize = cfg.d_k * cfg.d_v
    cfg.dstate_cosize = cfg.d_k * cfg.d_v
    cfg.dq_cosize = q_tile_elements * cfg.smem_dq_stages
    cfg.dk_cosize = q_tile_elements * cfg.smem_dk_stages
    cfg.dv_cosize = v_tile_elements * cfg.smem_dv_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    cfg.tma_q_bytes = q_tile_elements * bytes_per_element
    cfg.tma_k_bytes = q_tile_elements * bytes_per_element
    cfg.tma_v_bytes = v_tile_elements * bytes_per_element
    cfg.tma_do_bytes = v_tile_elements * bytes_per_element
    cfg.tma_state_bytes = cfg.d_k * cfg.d_v * bytes_per_element

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cutlass.min(cutlass.Int32(cfg.max_active_clusters), cutlass.max(cutlass.Int32(work_items.shape[0]), cutlass.Int32(1))), 1, 1)

    frost_mamba2_bprop(
        cfg,
        q_ratio,
        k_ratio,
        v_ratio,
        gate,
        a,
        dt_bias,
        d,
        cu_seqlens,
        dstate_in,
        dstate0,
        exit_dots,
        skip_bwd_part,
        work_items,
        work_count,
        scheduler_counter,
        transition,
        dt_min,
        dt_max,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        state_checkpoints,
        tensormap_workspace,
        cutlass.Int32(num_descs),
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        cluster=cfg.cluster_shape_mnk,
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


# ---------------------------------------------------------------------------


frost_mamba2_bprop_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_bprop.set_name_prefix("cudnn", remove_cutlass_symbol=False)
