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
Chunked Mamba-2 SSD prefill kernel for SM100 / SM103 / SM107 (Cutlass primitives) with optional
per-chunk state-checkpoint output.

Mamba-2 in the GDN vocabulary: Q = C, K = B, V = x, the gate is dt and the write strength is dt too.
The recurrence is the gated delta rule without the erase term, S_t = alpha_t S_t-1 + dt_t k_t v_t^T,
o_t = q_t S_t, with alpha_t = exp(A_h dt_t) and dt_t = clamp(softplus(dt_raw_t + dt_bias_h)).

Algorithm overview (per chunk c of C = 128 tokens, natural orientation, tokens on M):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], Gate[BT] (raw dt), A[H], Dt_bias[H]
  State  : S[DK,DV]  (recurrent state, fp32 in TMEM)

  Gate warp:
    dt_t          = clamp(softplus(gate_t + dt_bias), dt_min, dt_max)       (softplus optional)
    cumsumlog[t]  = sum_{l=0}^{t} dt_l A / ln 2                            log2 of the running decay
    beta[t]       = dt_t

  QK GEMM        : W[BT,BT]   = Q @ K^T                        (hoisted one chunk ahead)
  A epilogue     : A[i,j]     = [i >= j] exp2(cumsumlog_i - cumsumlog_j) beta_j W[i,j]   (bf16, OVERLAID
                   on the first 64 columns of its own accumulator stage)
  Q*state GEMM   : O_inter    = Q @ S
  state GEMM     : S         += K^T @ U'    where U'[j,:] = beta_j exp2(cumsumlog_127 - cumsumlog_j) V[j,:]
  A*V GEMM       : O_intra    = A @ V

  Epilogues:
    S            <- exp2(cumsumlog_127) S  before the state GEMM accumulates (in place, CG1), with the
                    bf16 copy of S for the Q*state GEMM (the copy doubles as the checkpoint tile)
    O[i,:]        = exp2(cumsumlog_i) O_inter[i,:] + O_intra[i,:] + D x[i,:]  (D per head or per channel)
                    (* z sigmoid(z) when the z gate is fused; the pre-gate O is stored too under grad)

SMEM layout (227 KB SM100 / SM103 / SM107 cap; stage counts live in mamba2_prefill_config.py):
  Buffer                       Size (B)  Stages
  Q                               32768       2
  K                               32768       2
  V                               16384       2
  z                               16384       1     <-- has_z only
  S^T bf16 copy / checkpoint tile 16384       1   (V-major: [DV][DK], two 64-column panels)
  U'                              16384       1
  O store                         16384       1
  cumsumlog / beta                  512       3     <-- 2 with z
  scheduler ticket ring               4       2

TMEM layout (512 columns):
  Buffer                  Cols
  state                    64     <-- DKxDV fp32, lane = d_k row
  O_inter acc              64     <-- BTxDV fp32, lane = token
  O_intra acc              64
  (free)                   64
  QK acc                  128 x 2 <-- 2-stage ring; the bf16 A tile overlays columns 0-63 of its stage
                                      once CG0 has loaded the fp32 tile; the stage is reused after
                                      the A*V GEMM read it (the in-order tensor pipe)

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - A epilogue
  warps 4-7     : compute group 1 - state copy / rescale, U' build, state I/O, checkpoints
  warps 8-11    : compute group 2 - output epilogue (Gamma fold, D skip, z gate)
  warp  12      : MMA warp       - QK (hoisted) + Q*state / state / A*V per chunk; TMEM lifecycle
  warp  13      : TMA load warp  - loads Q, K, V (+ z)
  warp  14      : Gate loads     - dt softplus / clamp, cumsum
  warp  15      : epilogue warp  - O (+ pre-gate O) then checkpoint TMA stores
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma

from ..common.thd import emit_checkpoint_seq_descs, emit_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, WORK_ITEM_FINAL_DST, decode_head, decode_work_item, order_body

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
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fadd2, ffma2, fmul2, fp32_to_fp16, sigmoid, softplus
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .mamba2_prefill_config import CFG

USE_PDL = True


class Mamba2PrefillBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_z_ready: MBarrier
    mb_z_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_qk_acc_ready: MBarrier
    mb_qk_acc_done: MBarrier
    mb_a_ready: MBarrier

    mb_state_acc_ready: MBarrier
    mb_state_ready: MBarrier
    mb_state_copy_done: MBarrier
    mb_u_ready: MBarrier
    mb_u_done: MBarrier

    mb_o_inter_acc_ready: MBarrier
    mb_o_inter_acc_done: MBarrier
    mb_o_intra_acc_ready: MBarrier
    mb_o_intra_acc_done: MBarrier

    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier
    mb_pre_tmastg_ready: MBarrier
    mb_pre_tmastg_done: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> Mamba2PrefillBars:
    """Mamba2PrefillBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    EPILOGUE_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG2_THREADS = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    ALL_COMPUTE = CG0_THREADS + CG1_THREADS + CG2_THREADS
    GATE_CONSUMERS = ALL_COMPUTE
    V_CONSUMERS = CG2_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    return Mamba2PrefillBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=V_CONSUMERS, producer=Producer.THREAD),
        mb_z_ready=MBarrier(alloc(cfg.smem_z_stages), try_wait=True, stages=cfg.smem_z_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_z_done=MBarrier(alloc(cfg.smem_z_stages), try_wait=True, stages=cfg.smem_z_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_CONSUMERS, producer=Producer.THREAD),
        mb_qk_acc_ready=MBarrier(
            alloc(cfg.tmem_qk_acc_stages), try_wait=True, stages=cfg.tmem_qk_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_qk_acc_done=MBarrier(alloc(cfg.tmem_qk_acc_stages), try_wait=True, stages=cfg.tmem_qk_acc_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_ready=MBarrier(alloc(cfg.tmem_qk_acc_stages), try_wait=True, stages=cfg.tmem_qk_acc_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_state_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_state_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_state_copy_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_u_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_u_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_o_inter_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_o_inter_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_o_intra_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_o_intra_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), try_wait=True, stages=cfg.smem_o_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), try_wait=True, stages=cfg.smem_o_stages, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
        mb_pre_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), try_wait=True, stages=cfg.smem_o_stages, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_pre_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), try_wait=True, stages=cfg.smem_o_stages, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
        mb_checkpoint_tmastg_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=len(cfg.compute_group_1_warp_ids), producer=Producer.THREAD),
        mb_checkpoint_tmastg_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=EPILOGUE_WARP, producer=Producer.THREAD),
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
    checkpoint_every_n_tokens,
    tidx,
    sO_raw,
    sState_raw,
    desc_o_base,
    desc_pre_base,
    desc_checkpoint_base,
    sScheduler,
    bars,
):
    """Epilogue warp role (warp 15): persistent scheduler loop issuing the per-chunk O (and pre-gate O) TMA
    stores and the state-checkpoint TMA stores out of the S copy tile."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    o_index = PipelineState.start(phase=0)
    pre_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=cfg.b_t * box_elements,
    )
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_tma = SmemTile(
            base=sState_raw,
            elems_per_stage=cfg.d_k * cfg.d_v,
            stages=1,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=cfg.d_k // box_elements,
            tma_granu_elems=box_elements,
            tma_subtile_stride_elems=cfg.d_v * box_elements,
        )
        checkpoint_store_count = cutlass.Int32(0)
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = write_end - compute_start

        head_o, v_offset = decode_head(cfg, head_idx)
        slot = batch_idx * desc_qwords
        desc_o_slot = (desc_o_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_o_slot)
        if cutlass.const_expr(cfg.store_pre_gate):
            desc_pre_slot = (desc_pre_base + slot).tospace(cutlass.AddressSpace.generic)
            if elect_one:
                tma_tensormap_acquire(desc_pre_slot)
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
            checkpoint_coordinate = (write_start + checkpoint_chunks - cutlass.Int32(1)) // checkpoint_chunks
            checkpoint_mod = compute_start % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)

        if n_local > 0:
            for local_idx in cutlass.range(n_local):
                chunk_idx = compute_start + local_idx
                writes = chunk_idx >= write_start

                # ---- checkpoint store: the S copy of chunk c is the state entering it -----
                if cutlass.const_expr(cfg.enable_checkpoints):
                    if checkpoint_mod == 0 and writes:
                        checkpoint_phase = checkpoint_store_count & cutlass.Int32(1)
                        bars.mb_checkpoint_tmastg_ready[0].wait(checkpoint_phase)
                        checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), v_offset, checkpoint_coordinate, head_o)
                        tma_store_tile(sCheckpoint_tma[0], checkpoint_slice, acquire=False)
                        tma_store_commit()
                        tma_store_wait(0)
                        bars.mb_checkpoint_tmastg_done[0].arrive()
                        checkpoint_coordinate += 1
                        checkpoint_store_count = checkpoint_store_count + 1
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    checkpoint_mod = cutlass.Int32(0) if checkpoint_mod == checkpoint_chunks else checkpoint_mod

                token_coordinate = chunk_idx * cutlass.Int32(cfg.b_t)

                # ---- pre-gate O store (z fused under grad) ---------------------------------
                if cutlass.const_expr(cfg.store_pre_gate):
                    pre_idx = pre_index.idx
                    bars.mb_pre_tmastg_ready[pre_idx].wait(pre_index.phase)
                    pre_index = advance(pre_index, cfg.smem_o_stages)
                    if writes:
                        pre_slice = tma_slice_runtime_desc(desc_pre_slot, v_offset, head_o, token_coordinate)
                        tma_store_tile(sO_tma[pre_idx], pre_slice, acquire=False)
                        tma_store_commit()
                        tma_store_wait(0)
                    bars.mb_pre_tmastg_done[pre_idx].arrive()

                # ---- O store -------------------------------------------------------------------
                o_idx = o_index.idx
                bars.mb_o_tmastg_ready[o_idx].wait(o_index.phase)
                o_index = advance(o_index, cfg.smem_o_stages)
                if writes:
                    o_slice = tma_slice_runtime_desc(desc_o_slot, v_offset, head_o, token_coordinate)
                    tma_store_tile(sO_tma[o_idx], o_slice, acquire=False)
                    tma_store_commit()
                    tma_store_wait(0)
                bars.mb_o_tmastg_done[o_idx].arrive()

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
    """Gate producer (warp 14): the per-chunk dt transform and cumsum tables (cumsumlog[t], beta[t] = dt_t); with
    ``mTransition`` bound, the tile's log2 decay product per (sequence, head)."""

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
        n_local = write_end - compute_start
        tile_log2 = cutlass.Float32(0.0)
        if n_local > 0:
            a = mA[head_o].to(cutlass.Float32) * cutlass.Float32(RCP_LN2)
            if cutlass.const_expr(mDt_bias is not None):
                bias = mDt_bias[head_o].to(cutlass.Float32)
            gGateSeq = mGate[None, head_o]
            for local_idx in cutlass.range(n_local):
                # ---- dt load + transform ----------------------------------------------------
                chunk_idx = compute_start + local_idx
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
    sK,
    sK_trans,
    sV_trans,
    sState,
    sU_trans,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 12): every tcgen05 GEMM; the Q K^T GEMM of chunk c + 1 is hoisted into chunk c."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=0)
    k_index = PipelineState.start(phase=0)
    q_look_index = PipelineState.start(phase=0)
    k_look_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    qk_issue_index = PipelineState.start(phase=1)
    qk_use_index = PipelineState.start(phase=0)
    state_ready_index = PipelineState.start(phase=0)
    u_ready_index = PipelineState.start(phase=0)
    o_inter_done_index = PipelineState.start(phase=1)
    o_intra_done_index = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_qk = nvvm.Tcgen05InstrDesc.build(
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
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_qs = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.b_t,
    )
    bmm_qs_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_v,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_qs,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_kv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_kv_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_kv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_av = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.b_t,
        b_major=1,
    )
    bmm_av_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_av,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    QK_STAGE_COLS = cfg.b_t

    tmem_col = tmem_base_slot.load() & 0xFFFF
    tmem_qk_col = tmem_col + cfg.tmem_qk_acc_offset
    state_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_state_acc_offset, cutlass.Float32)
    o_inter_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_o_inter_acc_offset, cutlass.Float32)
    o_intra_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_o_intra_acc_offset, cutlass.Float32)
    desc_state = desc_opaque(sState[0].desc())
    desc_u_trans = desc_opaque(sU_trans[0].desc())

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = write_end - compute_start

        # ---- Q K^T of the first chunk ----------------------------------------------------
        if n_local > 0:
            qk_idx = qk_issue_index.idx
            bars.mb_qk_acc_done[qk_idx].wait(qk_issue_index.phase)
            qk_issue_index = advance(qk_issue_index, cfg.tmem_qk_acc_stages)
            q_look_idx = q_look_index.idx
            k_look_idx = k_look_index.idx
            bars.mb_q_ready[q_look_idx].wait(q_look_index.phase)
            q_look_index = advance(q_look_index, cfg.smem_q_stages)
            bars.mb_k_ready[k_look_idx].wait(k_look_index.phase)
            k_look_index = advance(k_look_index, cfg.smem_k_stages)
            acc_qk = nvvm.make_tmem_ptr(tmem_qk_col + qk_idx * QK_STAGE_COLS, cutlass.Float32)
            mma_ss(bmm_qk_desc, desc_opaque(sQ[q_look_idx].desc()), desc_opaque(sK[k_look_idx].desc()), acc_qk, accumulate=False)
            if elect_one:
                bars.mb_qk_acc_ready[qk_idx].arrive(cta_group=1)

        for peel in cutlass.range_constexpr(1 if cfg.use_initial_state else 2):
            have_state = True if cfg.use_initial_state else peel == 1
            peel_stop = n_local if cutlass.const_expr(peel == 1 or cfg.use_initial_state) else cutlass.min(n_local, 1)
            for local_idx in cutlass.range(peel, peel_stop, 1):  # noqa: B007
                q_idx = q_index.idx
                q_index = advance(q_index, cfg.smem_q_stages)
                k_idx = k_index.idx
                k_index = advance(k_index, cfg.smem_k_stages)
                desc_q = desc_opaque(sQ[q_idx].desc())
                desc_k_trans = desc_opaque(sK_trans[k_idx].desc())

                # ---- Q K^T lookahead (chunk c + 1) ---------------------------------------
                if local_idx + 1 < n_local:
                    qk_idx = qk_issue_index.idx
                    bars.mb_qk_acc_done[qk_idx].wait(qk_issue_index.phase)
                    qk_issue_index = advance(qk_issue_index, cfg.tmem_qk_acc_stages)
                    q_look_idx = q_look_index.idx
                    k_look_idx = k_look_index.idx
                    bars.mb_q_ready[q_look_idx].wait(q_look_index.phase)
                    q_look_index = advance(q_look_index, cfg.smem_q_stages)
                    bars.mb_k_ready[k_look_idx].wait(k_look_index.phase)
                    k_look_index = advance(k_look_index, cfg.smem_k_stages)
                    acc_qk = nvvm.make_tmem_ptr(tmem_qk_col + qk_idx * QK_STAGE_COLS, cutlass.Float32)
                    mma_ss(bmm_qk_desc, desc_opaque(sQ[q_look_idx].desc()), desc_opaque(sK[k_look_idx].desc()), acc_qk, accumulate=False)
                    if elect_one:
                        bars.mb_qk_acc_ready[qk_idx].arrive(cta_group=1)

                # ---- O_inter = Q @ S ---------------------------------------------------
                if cutlass.const_expr(have_state):
                    bars.mb_state_ready[0].wait(state_ready_index.phase)
                    state_ready_index = advance(state_ready_index, 1)
                    bars.mb_o_inter_acc_done[0].wait(o_inter_done_index.phase)
                    o_inter_done_index = advance(o_inter_done_index, 1)
                    mma_ss(bmm_qs_desc, desc_q, desc_state, o_inter_acc_ptr, accumulate=False)
                    if elect_one:
                        bars.mb_o_inter_acc_ready[0].arrive(cta_group=1)
                        bars.mb_state_copy_done[0].arrive(cta_group=1)
                if elect_one:
                    bars.mb_q_done[q_idx].arrive(cta_group=1)

                # ---- S += K^T @ U' -------------------------------------------------------
                bars.mb_u_ready[0].wait(u_ready_index.phase)
                u_ready_index = advance(u_ready_index, 1)
                mma_ss(bmm_kv_desc, desc_k_trans, desc_u_trans, state_acc_ptr, accumulate=have_state)
                if elect_one:
                    bars.mb_state_acc_ready[0].arrive(cta_group=1)
                    bars.mb_u_done[0].arrive(cta_group=1)
                    bars.mb_k_done[k_idx].arrive(cta_group=1)

                # ---- O_intra = A(T) @ V ----------------------------------------------------
                bars.mb_o_intra_acc_done[0].wait(o_intra_done_index.phase)
                o_intra_done_index = advance(o_intra_done_index, 1)
                a_idx = qk_use_index.idx
                bars.mb_a_ready[a_idx].wait(qk_use_index.phase)
                qk_use_index = advance(qk_use_index, cfg.tmem_qk_acc_stages)
                v_idx = v_index.idx
                v_index = advance(v_index, cfg.smem_v_stages)
                a_ptr = nvvm.make_tmem_ptr(tmem_qk_col + a_idx * QK_STAGE_COLS, cutlass.Int8)
                desc_v_trans = desc_opaque(sV_trans[v_idx].desc())
                for k in cutlass.range_constexpr(cfg.b_t // 16):
                    mma_ts_step(bmm_av_desc, a_ptr, desc_v_trans, o_intra_acc_ptr, k, cutlass.Boolean(k > 0))
                if elect_one:
                    bars.mb_o_intra_acc_ready[0].arrive(cta_group=1)

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
    sZ_raw,
    tidx,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_z_base,
    mScheduler,
    sScheduler,
    bars,
    q_ratio,
    k_ratio,
    v_ratio,
):
    """TMA-LDG warp role (warp 13): persistent scheduler loop + per-chunk Q / K / V (+ z) G->S TMA loads."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    z_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    bt = cfg.b_t
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=cfg.q_cosize // cfg.smem_q_stages,
        stages=cfg.smem_q_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=cfg.k_cosize // cfg.smem_k_stages,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=cfg.v_cosize // cfg.smem_v_stages,
        stages=cfg.smem_v_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elements,
        tma_granu_elems=box_elements,
        tma_subtile_stride_elems=bt * box_elements,
    )
    if cutlass.const_expr(cfg.has_z):
        sZ_tma = SmemTile(
            base=sZ_raw,
            elems_per_stage=cfg.z_cosize // cfg.smem_z_stages,
            stages=cfg.smem_z_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=cfg.d_v // box_elements,
            tma_granu_elems=box_elements,
            tma_subtile_stride_elems=bt * box_elements,
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
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
        if cutlass.const_expr(cfg.has_z):
            desc_z_slot = (desc_z_base + slot).tospace(cutlass.AddressSpace.generic)
            if elect_one:
                tma_tensormap_acquire(desc_z_slot)

        for chunk_idx in cutlass.range(compute_start, write_end):
            token_coordinate = chunk_idx * cutlass.Int32(cfg.b_t)

            # ---- Q load --------------------------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_done[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            if elect_one:
                bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, token_coordinate)
            tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr, acquire=False)

            # ---- K load --------------------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if elect_one:
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, token_coordinate)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr, acquire=False)

            # ---- V load --------------------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, token_coordinate)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

            # ---- z load --------------------------------------------------------------------
            if cutlass.const_expr(cfg.has_z):
                z_idx = z_index.idx
                bars.mb_z_done[z_idx].wait(z_index.phase)
                z_index = advance(z_index, cfg.smem_z_stages)
                if elect_one:
                    bars.mb_z_ready[z_idx].arrive(n_bytes=cfg.tma_z_bytes)
                z_slice = tma_slice_runtime_desc(desc_z_slot, v_offset, head_o, token_coordinate)
                tma_load_tile(sZ_tma[z_idx], z_slice, bars.mb_z_ready[z_idx].smem_ptr, acquire=False)

        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    if cutlass.const_expr(cfg.has_z):
        for _ in range(cfg.smem_z_stages):
            bars.mb_z_done[z_index.idx].wait(z_index.phase)
            z_index = advance(z_index, cfg.smem_z_stages)


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
    sCumsumlog,
    sBeta,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): the A epilogue, one query token per thread; the bf16 A row overlays the
    first 64 columns of its own Q K^T stage."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    qk_acc_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.b_t // ldtm_width)
    sttm_width = 16
    QK_STAGE_COLS = cfg.b_t
    neg_inf = cutlass.Float32(float("-inf"))
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_qk_col = tmem_col + cfg.tmem_qk_acc_offset
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = write_end - compute_start

        for local_idx in cutlass.range(n_local):
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            qk_idx = qk_acc_index.idx
            bars.mb_qk_acc_ready[qk_idx].wait(qk_acc_index.phase)
            qk_acc_index = advance(qk_acc_index, cfg.tmem_qk_acc_stages)
            g_i = sCumsumlog[cg0_tidx, 0, gate_idx]

            # ---- Q K^T row: TMEM -> RMEM, stage released -----------------------------------
            w_vectors = []
            for i in cutlass.range_constexpr(num_ldtms):
                w_vectors.append(
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_qk_col + qk_idx * QK_STAGE_COLS + i * ldtm_width, cutlass.Float32), num=32)
                )
            nvvm.tcgen05_wait("load")
            bars.mb_qk_acc_done[qk_idx].arrive()

            # ---- A epilogue: A[i,j] = [i >= j] exp2(g_i - g_j) beta_j W[i,j] ----------------
            for b in cutlass.range_constexpr(num_ldtms):
                a_pack = []
                for jj in cutlass.range_constexpr(0, ldtm_width, 2):
                    j = b * ldtm_width + jj
                    d0, d1 = fadd2(g_i, g_i, -sCumsumlog[j, 0, gate_idx], -sCumsumlog[j + 1, 0, gate_idx])
                    d0 = d0 if cutlass.Int32(j) <= cg0_tidx else neg_inf
                    d1 = d1 if cutlass.Int32(j + 1) <= cg0_tidx else neg_inf
                    e0 = cute.math.exp2(d0, fastmath=True)
                    e1 = cute.math.exp2(d1, fastmath=True)
                    w0, w1 = fmul2(e0, e1, sBeta[j, 0, gate_idx], sBeta[j + 1, 0, gate_idx])
                    a0, a1 = fmul2(w_vectors[b][jj], w_vectors[b][jj + 1], w0, w1)
                    a_pack.append(fp32_to_fp16(a0, a1, dtype=cfg.io_dtype))
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + tmem_qk_col + qk_idx * QK_STAGE_COLS + b * sttm_width, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(a_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_a_ready[qk_idx].arrive()
            bars.mb_gate_done[gate_idx].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

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
    sV_raw,
    sState_raw,
    sU_raw,
    sCumsumlog,
    sBeta,
    mState_init,
    mState_out,
    mSeedIndices,
    mFinalIndices,
    checkpoint_every_n_tokens,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): the state epilogue (bf16 copy, checkpoint hand-off, in-place rescale), the
    U' build and the per-tile state seed / final-state store; one state row per thread, one key token per thread in the
    U' build."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)

    gate_index = PipelineState.start(phase=0)
    state_acc_index = PipelineState.start(phase=0)
    state_copy_done_index = PipelineState.start(phase=1)
    u_done_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = tidx % cfg.threads_per_warp
    warp_in_group = cg1_tidx // cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    bytes_per_element = cfg.io_dtype.width // 8
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.d_v // ldtm_width)
    vector_width = 16 // bytes_per_element
    num_vectors = cutlass.const_expr(cfg.d_v // vector_width)
    box_elements = 128 // bytes_per_element
    words_per_vector = cutlass.const_expr(vector_width // 2)
    row_swizzle_elements = (cg1_tidx & 7) * vector_width
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    sV_base = sV_raw.data_ptr()
    sState_base = sState_raw.data_ptr()
    sU_base = sU_raw.data_ptr()
    checkpoint_count = cutlass.Int32(0)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    tmem_state_col = tmem_col + cfg.tmem_state_acc_offset

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        seed_row = batch_idx
        if cutlass.const_expr(mSeedIndices is not None):
            seed_row = cutlass.Int32(mSeedIndices[batch_idx])
        n_local = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_mod = compute_start % checkpoint_chunks
        if n_local > 0:
            # ---- state seed: initial state (V-major in GMEM) or zeros -> TMEM -------------
            if cutlass.const_expr(cfg.use_initial_state):
                gState_init = mState_init[None, None, head_o, seed_row]
                seed_state = compute_start == 0
                if seed_state:
                    for i in cutlass.range_constexpr(num_ldtms):
                        words = []
                        for t in cutlass.range_constexpr(ldtm_width):
                            words.append(gState_init[i * ldtm_width + t + v_offset, cg1_tidx].to(cfg.acc_dtype))
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")
                else:
                    for i in cutlass.range_constexpr(num_ldtms):
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0) for _ in range(ldtm_width)), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")

            for peel in cutlass.range_constexpr(1 if cfg.use_initial_state else 2):
                have_state = True if cfg.use_initial_state else peel == 1
                peel_stop = n_local if cutlass.const_expr(peel == 1 or cfg.use_initial_state) else cutlass.min(n_local, 1)
                for local_idx in cutlass.range(peel, peel_stop, 1):  # noqa: B007
                    chunk_idx = compute_start + local_idx
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        do_checkpoint_now = checkpoint_mod == 0
                        checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                        checkpoint_mod = cutlass.Int32(0) if checkpoint_mod == checkpoint_chunks else checkpoint_mod
                    if cutlass.const_expr(cfg.enable_checkpoints and not cfg.use_initial_state):
                        # ---- checkpoint row 0 = the zero incoming state -----------------------
                        if chunk_idx == 0 and write_start == 0:
                            bars.mb_checkpoint_tmastg_done[0].wait((checkpoint_count & cutlass.Int32(1)) ^ cutlass.Int32(1))
                            checkpoint_zero_ptr = cutlass.inttoptr(sState_base.toint(), cutlass.AddressSpace.smem, cutlass.Int32)
                            for z in cutlass.range_constexpr(cfg.d_k * cfg.d_v * bytes_per_element // 4 // num_threads_cg1):
                                (checkpoint_zero_ptr + cg1_tidx + z * num_threads_cg1).store(cutlass.Int32(0))
                            nvvm.fence_proxy("async.shared", space="cta")
                            if elect_one:
                                bars.mb_checkpoint_tmastg_ready[0].arrive()
                            checkpoint_count = checkpoint_count + 1

                    gate_idx = gate_index.idx
                    bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                    gate_index = advance(gate_index, cfg.smem_gate_stages)
                    g_last = sCumsumlog[cfg.b_t - 1, 0, gate_idx]

                    # ---- state epilogue: bf16 copy, checkpoint, rescale ----------------------
                    if cutlass.const_expr(have_state):
                        if cutlass.const_expr(cfg.use_initial_state):
                            if local_idx > 0:
                                bars.mb_state_acc_ready[0].wait(state_acc_index.phase)
                                state_acc_index = advance(state_acc_index, 1)
                        else:
                            bars.mb_state_acc_ready[0].wait(state_acc_index.phase)
                            state_acc_index = advance(state_acc_index, 1)
                        state_halves = []
                        for half in cutlass.range_constexpr(2):
                            half_addr = row_lo_addr if half == 0 else row_hi_addr
                            state_halves.append(nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(half_addr + tmem_state_col, cutlass.Float32), num=8))
                        nvvm.tcgen05_wait("load")

                        bars.mb_state_copy_done[0].wait(state_copy_done_index.phase)
                        state_copy_done_index = advance(state_copy_done_index, 1)
                        if cutlass.const_expr(cfg.enable_checkpoints):
                            bars.mb_checkpoint_tmastg_done[0].wait((checkpoint_count & cutlass.Int32(1)) ^ cutlass.Int32(1))
                        # ---- S^T bf16 copy (V-major): the Q S operand and the checkpoint tile ----
                        matrix = lane_idx // 8
                        panel_base = (warp_in_group // 2) * (cfg.d_v * box_elements)
                        for half in cutlass.range_constexpr(2):
                            n_col = (warp_in_group % 2) * 32 + half * 16 + (matrix % 2) * 8
                            copy_words = [fp32_to_fp16(state_halves[half][2 * j], state_halves[half][2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                            for c in cutlass.range_constexpr(4):
                                p_row = (2 * c + matrix // 2) * 8 + lane_idx % 8
                                nvvm.stmatrix(
                                    sState_base + panel_base + p_row * box_elements + swizzle_xor_128b(p_row, n_col),
                                    [copy_words[4 * c], copy_words[4 * c + 1], copy_words[4 * c + 2], copy_words[4 * c + 3]],
                                    nvvm.MMALayout.COL,
                                )
                        nvvm.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(cfg.enable_checkpoints):
                            do_checkpoint = do_checkpoint_now and chunk_idx < write_end
                            do_checkpoint = do_checkpoint and chunk_idx >= write_start
                            if do_checkpoint:
                                if elect_one:
                                    bars.mb_checkpoint_tmastg_ready[0].arrive()
                                checkpoint_count = checkpoint_count + 1

                        gamma = cute.math.exp2(g_last, fastmath=True)
                        for half in cutlass.range_constexpr(2):
                            state_scaled = []
                            for m in cutlass.range_constexpr(16):
                                s0, s1 = fmul2(state_halves[half][2 * m], state_halves[half][2 * m + 1], gamma, gamma)
                                state_scaled += [s0, s1]
                            nvvm.tcgen05_st(
                                "16x256b",
                                nvvm.make_tmem_ptr((row_lo_addr if half == 0 else row_hi_addr) + tmem_state_col, cutlass.Float32),
                                cutlass.Vector.from_elements(tuple(state_scaled), cutlass.Float32),
                            )
                        nvvm.tcgen05_wait("store")
                        bars.mb_state_ready[0].arrive()

                    # ---- U' build: U'[j,:] = beta_j exp2(g_127 - g_j) V[j,:] ---------------
                    v_idx = v_index.idx
                    bars.mb_v_ready[v_idx].wait(v_index.phase)
                    v_index = advance(v_index, cfg.smem_v_stages)
                    bars.mb_u_done[0].wait(u_done_index.phase)
                    u_done_index = advance(u_done_index, 1)
                    s_j = sBeta[cg1_tidx, 0, gate_idx] * cute.math.exp2(g_last - sCumsumlog[cg1_tidx, 0, gate_idx], fastmath=True)
                    v_row_offset = v_idx * v_stage_elements + cg1_tidx * cfg.d_v + row_swizzle_elements
                    u_row_offset = cg1_tidx * cfg.d_v + row_swizzle_elements
                    for q in cutlass.range_constexpr(num_vectors):
                        slot = q * vector_width
                        v_words = (sV_base + (v_row_offset ^ slot)).load(count=vector_width, alignment=16).bitcast(cutlass.Int32)
                        words = []
                        for w in cutlass.range_constexpr(words_per_vector):
                            v_lo, v_hi = f16x2_to_f32(v_words[w], dtype=cfg.io_dtype)
                            u0, u1 = fmul2(v_lo, v_hi, s_j, s_j)
                            words.append(fp32_to_fp16(u0, u1, dtype=cfg.io_dtype))
                        (sU_base + (u_row_offset ^ slot)).store(cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_u_ready[0].arrive()
                    bars.mb_gate_done[gate_idx].arrive()

        # ---- final state store: TMEM -> GMEM (V-major) ----------------------------------
        if n_local > 0:
            bars.mb_state_acc_ready[0].wait(state_acc_index.phase)
            state_acc_index = advance(state_acc_index, 1)
            if cutlass.const_expr(cfg.store_final_state):
                final_destination = mWorkItems[tile_idx, WORK_ITEM_FINAL_DST]
                if final_destination >= 0:
                    final_row = final_destination
                    if cutlass.const_expr(mFinalIndices is not None):
                        final_row = cutlass.Int32(mFinalIndices[final_destination])
                    gState_out = mState_out[None, None, head_o, final_row]
                    for i in cutlass.range_constexpr(num_ldtms):
                        state_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32), num=32)
                        nvvm.tcgen05_wait("load")
                        for t in cutlass.range_constexpr(ldtm_width):
                            gState_out[i * ldtm_width + t + v_offset, cg1_tidx] = state_vector[t].to(mState_out.element_type)
        else:
            if cutlass.const_expr(cfg.store_final_state):
                final_destination = mWorkItems[tile_idx, WORK_ITEM_FINAL_DST]
                if final_destination >= 0:
                    final_row = final_destination
                    if cutlass.const_expr(mFinalIndices is not None):
                        final_row = cutlass.Int32(mFinalIndices[final_destination])
                    gState_out = mState_out[None, None, head_o, final_row]
                    if cutlass.const_expr(cfg.use_initial_state):
                        gState_in = mState_init[None, None, head_o, seed_row]
                        for p in cutlass.range_constexpr(cfg.d_v):
                            seed_value = gState_in[p + v_offset, cg1_tidx]
                            if cutlass.const_expr(mState_init.element_type != mState_out.element_type):
                                seed_value = seed_value.to(mState_out.element_type)
                            gState_out[p + v_offset, cg1_tidx] = seed_value
                    else:
                        for p in cutlass.range_constexpr(cfg.d_v):
                            gState_out[p + v_offset, cg1_tidx] = cutlass.Float32(0.0).to(mState_out.element_type)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()

    if cutlass.const_expr(cfg.enable_checkpoints):
        bars.mb_checkpoint_tmastg_done[0].wait((checkpoint_count & cutlass.Int32(1)) ^ cutlass.Int32(1))


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
    sV_raw,
    sZ_raw,
    sO_raw,
    sCumsumlog,
    mD,
    sScheduler,
    bars,
):
    """Compute warp-group 2 role (warps 8-11): the output epilogue, one token per thread, staged for the TMA store; the
    pre-gate O is stored first when the backward needs it."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.INCREASE)

    gate_index = PipelineState.start(phase=0)
    o_inter_index = PipelineState.start(phase=0)
    o_intra_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    z_index = PipelineState.start(phase=0)
    o_index = PipelineState.start(phase=1)
    pre_index = PipelineState.start(phase=0)

    num_threads_cg2 = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    cg2_tidx = tidx % num_threads_cg2

    elect_one = nvvm.elect_sync()
    bytes_per_element = cfg.io_dtype.width // 8
    ldtm_width = 32
    num_ldtms = cutlass.const_expr(cfg.d_v // ldtm_width)
    vector_width = 16 // bytes_per_element
    num_vectors = cutlass.const_expr(cfg.d_v // vector_width)
    words_per_vector = cutlass.const_expr(vector_width // 2)
    words_layout = cute.make_layout(words_per_vector)
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    v_stage_bytes = v_stage_elements * bytes_per_element
    o_stage_elements = cfg.o_cosize // cfg.smem_o_stages
    v_row_address = sV_raw.data_ptr().toint(cutlass.Int32) + cg2_tidx * (cfg.d_v * bytes_per_element) + ((cg2_tidx & 7) << 4)
    sO_base = sO_raw.data_ptr()
    if cutlass.const_expr(cfg.has_z):
        z_stage_elements = cfg.z_cosize // cfg.smem_z_stages
        sZ_base = sZ_raw.data_ptr()
    d_scalar = cutlass.Float32(0.0)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_o_inter_col = tmem_col + cfg.tmem_o_inter_acc_offset
    tmem_o_intra_col = tmem_col + cfg.tmem_o_intra_acc_offset

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = write_end - compute_start
        if cutlass.const_expr(cfg.has_d and not cfg.d_per_channel):
            if n_local > 0:
                d_scalar = mD[head_o].to(cutlass.Float32)
        if n_local > 0:
            for peel in cutlass.range_constexpr(1 if cfg.use_initial_state else 2):
                have_state = True if cfg.use_initial_state else peel == 1
                peel_stop = n_local if cutlass.const_expr(peel == 1 or cfg.use_initial_state) else cutlass.min(n_local, 1)
                for local_idx in cutlass.range(peel, peel_stop, 1):  # noqa: B007
                    gate_idx = gate_index.idx
                    bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                    gate_index = advance(gate_index, cfg.smem_gate_stages)
                    gamma_i = cute.math.exp2(sCumsumlog[cg2_tidx, 0, gate_idx], fastmath=True)
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_gate_done[gate_idx].arrive()

                    # ---- O = Gamma O_inter + O_intra ----------------------------------------
                    if cutlass.const_expr(have_state):
                        bars.mb_o_inter_acc_ready[0].wait(o_inter_index.phase)
                        o_inter_index = advance(o_inter_index, 1)
                    bars.mb_o_intra_acc_ready[0].wait(o_intra_index.phase)
                    o_intra_index = advance(o_intra_index, 1)
                    o_values = []
                    for i in cutlass.range_constexpr(num_ldtms):
                        intra_vector = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_o_intra_col + i * ldtm_width, cutlass.Float32), num=32)
                        if cutlass.const_expr(have_state):
                            inter_vector = nvvm.tcgen05_ld(
                                "32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_o_inter_col + i * ldtm_width, cutlass.Float32), num=32
                            )
                            nvvm.tcgen05_wait("load")
                            for m in cutlass.range_constexpr(ldtm_width // 2):
                                o_lo, o_hi = ffma2(gamma_i, gamma_i, inter_vector[2 * m], inter_vector[2 * m + 1], intra_vector[2 * m], intra_vector[2 * m + 1])
                                o_values += [o_lo, o_hi]
                        else:
                            nvvm.tcgen05_wait("load")
                            for m in cutlass.range_constexpr(ldtm_width):
                                o_values.append(intra_vector[m])
                    bars.mb_o_intra_acc_done[0].arrive()
                    if cutlass.const_expr(have_state):
                        bars.mb_o_inter_acc_done[0].arrive()

                    # ---- D skip: O += D x ------------------------------------------------------
                    v_idx = v_index.idx
                    bars.mb_v_ready[v_idx].wait(v_index.phase)
                    v_index = advance(v_index, cfg.smem_v_stages)
                    if cutlass.const_expr(cfg.has_d):
                        v_stage_address = v_row_address + v_idx * v_stage_bytes
                        for q in cutlass.range_constexpr(num_vectors):
                            x_words = cute.make_tensor(
                                cute.make_ptr(cutlass.Int32, v_stage_address ^ (q * 16), mem_space=cute.AddressSpace.smem, assumed_align=16), words_layout
                            ).load()
                            for w in cutlass.range_constexpr(words_per_vector):
                                p = q * vector_width + 2 * w
                                x_lo, x_hi = f16x2_to_f32(x_words[w], dtype=cfg.io_dtype)
                                if cutlass.const_expr(cfg.d_per_channel):
                                    d_lo = mD[head_o, p + v_offset].to(cutlass.Float32)
                                    d_hi = mD[head_o, p + 1 + v_offset].to(cutlass.Float32)
                                else:
                                    d_lo = d_scalar
                                    d_hi = d_scalar
                                o_values[p], o_values[p + 1] = ffma2(d_lo, d_hi, x_lo, x_hi, o_values[p], o_values[p + 1])
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_v_done[v_idx].arrive()

                    # ---- pre-gate O: staged and stored before the gate ---------------------
                    o_idx = o_index.idx
                    bars.mb_o_tmastg_done[o_idx].wait(o_index.phase)
                    o_index = advance(o_index, cfg.smem_o_stages)
                    o_row = o_idx * o_stage_elements + cg2_tidx * cfg.d_v
                    if cutlass.const_expr(cfg.store_pre_gate):
                        for q in cutlass.range_constexpr(num_vectors):
                            words = []
                            for m in cutlass.range_constexpr(vector_width // 2):
                                p = q * vector_width + 2 * m
                                words.append(fp32_to_fp16(o_values[p], o_values[p + 1], dtype=cfg.io_dtype))
                            (sO_base + o_row + swizzle_xor_128b(cg2_tidx, q * vector_width)).store(
                                cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                            )
                        nvvm.fence_proxy("async.shared", space="cta")
                        bars.mb_pre_tmastg_ready[o_idx].arrive()

                    # ---- z gate: O *= z sigmoid(z) -------------------------------------------
                    if cutlass.const_expr(cfg.has_z):
                        z_idx = z_index.idx
                        bars.mb_z_ready[z_idx].wait(z_index.phase)
                        z_index = advance(z_index, cfg.smem_z_stages)
                        z_row = z_idx * z_stage_elements + cg2_tidx * cfg.d_v
                        for q in cutlass.range_constexpr(num_vectors):
                            z_vector = (sZ_base + z_row + swizzle_xor_128b(cg2_tidx, q * vector_width)).load(count=vector_width, alignment=16)
                            for m in cutlass.range_constexpr(vector_width // 2):
                                p = q * vector_width + 2 * m
                                z_lo = z_vector[2 * m].to(cutlass.Float32)
                                z_hi = z_vector[2 * m + 1].to(cutlass.Float32)
                                o_values[p], o_values[p + 1] = fmul2(o_values[p], o_values[p + 1], z_lo * sigmoid(z_lo), z_hi * sigmoid(z_hi))
                        nvvm.fence_proxy("async.shared", space="cta")
                        bars.mb_z_done[z_idx].arrive()

                    # ---- output store: RMEM -> SMEM staging ------------------------------------
                    if cutlass.const_expr(cfg.store_pre_gate):
                        bars.mb_pre_tmastg_done[o_idx].wait(pre_index.phase)
                        pre_index = advance(pre_index, cfg.smem_o_stages)
                    for q in cutlass.range_constexpr(num_vectors):
                        words = []
                        for m in cutlass.range_constexpr(vector_width // 2):
                            p = q * vector_width + 2 * m
                            words.append(fp32_to_fp16(o_values[p], o_values[p + 1], dtype=cfg.io_dtype))
                        (sO_base + o_row + swizzle_xor_128b(cg2_tidx, q * vector_width)).store(
                            cutlass.Vector.from_elements(tuple(words), cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                        )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_o_tmastg_ready[o_idx].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.smem_o_stages):
        bars.mb_o_tmastg_done[o_index.idx].wait(o_index.phase)
        o_index = advance(o_index, cfg.smem_o_stages)


@cute.kernel
def frost_mamba2_prefill(
    cfg: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA: cute.Tensor,
    mDt_bias: Optional[cute.Tensor],
    mD: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    mState_init: Optional[cute.Tensor],
    mState_out: Optional[cute.Tensor],
    mSeedIndices: Optional[cute.Tensor],
    mFinalIndices: Optional[cute.Tensor],
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    mTransition: Optional[cute.Tensor],
    checkpoint_every_n_tokens: cutlass.Int32,
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    mQ,
    mK,
    mV,
    mZ,
    mO,
    mO_pre,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main Mamba-2 chunked kernel: warp-specialized persistent tile loop."""
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
    desc_o_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_z_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_pre_base = desc_base_words + cutlass.Int32(6) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWIZZLE_LAYOUT = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
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
        layout=SWIZZLE_LAYOUT,
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
        layout=SWIZZLE_LAYOUT,
    )
    sK_trans = SmemTile(
        base=sK_raw.data_ptr(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
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
        layout=SWIZZLE_LAYOUT,
    )
    if cutlass.const_expr(cfg.has_z):
        sZ_raw = cutlass.Array(
            cfg.io_dtype,
            cfg.z_cosize,
            space=cutlass.AddressSpace.smem,
            alignment=cfg.buffer_align_bytes,
        )
    else:
        sZ_raw = None
    sState_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.state_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sState = SmemTile(
        base=sState_raw.data_ptr(),
        elems_per_stage=cfg.state_cosize,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
    sU_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.u_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sU_trans = SmemTile(
        base=sU_raw.data_ptr(),
        elems_per_stage=cfg.u_cosize,
        stages=1,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
    sO_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.o_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=128)
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumsumlog_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sBeta = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, beta_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        beta_smem_layout_staged,
    )

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_done[s].init()
    if cutlass.const_expr(cfg.has_z):
        for s in range(cfg.smem_z_stages):
            bars.mb_z_ready[s].init()
            bars.mb_z_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.tmem_qk_acc_stages):
        bars.mb_qk_acc_ready[s].init()
        bars.mb_qk_acc_done[s].init()
        bars.mb_a_ready[s].init()
    for b in (
        bars.mb_state_acc_ready,
        bars.mb_state_ready,
        bars.mb_state_copy_done,
        bars.mb_u_ready,
        bars.mb_u_done,
        bars.mb_o_inter_acc_ready,
        bars.mb_o_inter_acc_done,
        bars.mb_o_intra_acc_ready,
        bars.mb_o_intra_acc_done,
    ):
        b[0].init()
    for s in range(cfg.smem_o_stages):
        bars.mb_o_tmastg_ready[s].init()
        bars.mb_o_tmastg_done[s].init()
        bars.mb_pre_tmastg_ready[s].init()
        bars.mb_pre_tmastg_done[s].init()
    bars.mb_checkpoint_tmastg_ready[0].init()
    bars.mb_checkpoint_tmastg_done[0].init()
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()
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
            warp_idx,
            tmem_base_slot=tmem_base_slot,
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
            sState_raw=sState_raw,
            sU_raw=sU_raw,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            mState_init=mState_init,
            mState_out=mState_out,
            mSeedIndices=mSeedIndices,
            mFinalIndices=mFinalIndices,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
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
            sV_raw=sV_raw,
            sZ_raw=sZ_raw,
            sO_raw=sO_raw,
            sCumsumlog=sCumsumlog,
            mD=mD,
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
            sK=sK,
            sK_trans=sK_trans,
            sV_trans=sV_trans,
            sState=sState,
            sU_trans=sU_trans,
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
            sZ_raw=sZ_raw,
            tidx=tidx,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            desc_z_base=desc_z_base,
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
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            tidx=tidx,
            sO_raw=sO_raw,
            sState_raw=sState_raw,
            desc_o_base=desc_o_base,
            desc_pre_base=desc_pre_base,
            desc_checkpoint_base=desc_checkpoint_base,
            sScheduler=sScheduler,
            bars=bars,
        )


@dataclass
class Mamba2PrefillCfg:
    """Per-compile Mamba-2 kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA, state flags, the D / z fusions) are the
    ``cute.compile`` cache keys; the rest is derived from the module-global ``CFG``
    constants.  ``host`` stamps the shape-derived fields at trace time.  Passed
    ``cfg``-first (a ``cutlass.Constexpr``) into ``host`` / ``kernel`` and every warp body.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    d_k: int
    d_v: int
    dt_softplus: bool = True
    has_d: bool = False
    d_per_channel: bool = False
    has_z: bool = False
    store_pre_gate: bool = False
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

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_z_stages: int = CFG.SMEM_Z_STAGES
    smem_state_stages: int = CFG.SMEM_STATE_STAGES
    smem_u_stages: int = CFG.SMEM_U_STAGES
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_state_acc_stages: int = CFG.TMEM_STATE_ACC_STAGES
    tmem_o_inter_acc_stages: int = CFG.TMEM_O_INTER_ACC_STAGES
    tmem_o_intra_acc_stages: int = CFG.TMEM_O_INTRA_ACC_STAGES
    tmem_qk_acc_stages: int = CFG.TMEM_QK_ACC_STAGES
    tmem_state_acc_offset: int = 0
    tmem_o_inter_acc_offset: int = 0
    tmem_o_intra_acc_offset: int = 0
    tmem_qk_acc_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    z_cosize: int = 0
    state_cosize: int = 0
    u_cosize: int = 0
    o_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_z_bytes: int = 0
    tma_o_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_initial_state: bool,
    store_final_state: bool = True,
    enable_checkpoints: bool = False,
    dt_softplus: bool = True,
    has_d: bool = False,
    d_per_channel: bool = False,
    has_z: bool = False,
    store_pre_gate: bool = False,
    d_k: int,
    d_v: int,
) -> Mamba2PrefillCfg:
    """Build the per-compile ``Mamba2PrefillCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32); ``has_z``
    compiles run two gate-table stages."""
    cfg = Mamba2PrefillCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        dt_softplus=dt_softplus,
        has_d=has_d,
        d_per_channel=d_per_channel,
        has_z=has_z,
        store_pre_gate=store_pre_gate and has_z,
        d_k=d_k,
        d_v=d_v,
    )
    if has_z:
        cfg.smem_gate_stages = 2
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    n_cg2 = len(cfg.compute_group_2_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1 + n_cg2)
    cfg.tmem_user_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1 + n_cg2)
    cfg.tmem_state_acc_offset = 0
    cfg.tmem_o_inter_acc_offset = cfg.tmem_state_acc_offset + cfg.tmem_state_acc_stages * cfg.d_v
    cfg.tmem_o_intra_acc_offset = cfg.tmem_o_inter_acc_offset + cfg.tmem_o_inter_acc_stages * cfg.d_v
    cfg.tmem_qk_acc_offset = 512 - cfg.tmem_qk_acc_stages * cfg.b_t
    return cfg


TENSORMAP_DESC_ARRAYS = 7  # per-batch runtime TMA descriptors: Q, K, V, O, checkpoints, z, pre-gate O


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_o,
    base_checkpoint,
    base_z,
    base_pre,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    state_checkpoints_out: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
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
        emit_seq_descs(base_o, sub3, cu_seqlens, o if o is not None else v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(state_checkpoints_out is not None):
        if widx == 4:
            emit_checkpoint_seq_descs(base_checkpoint, sub4, cu_seqlens, state_checkpoints_out, n_batch, checkpoint_every_n, 2, lanes=32)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(z is not None):
        if widx == 5:
            emit_seq_descs(base_z, sub5, cu_seqlens, z, n_batch, 2, lanes=32)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(o_pre is not None):
        if widx == 6:
            emit_seq_descs(base_pre, sub6, cu_seqlens, o_pre, n_batch, 2, lanes=32)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_mamba2_prefill_prologue(
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_o: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_z: cutlass.GridConstant[tma.TensorMap],
    base_pre: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    state_checkpoints_out: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Two-CTA prologue: block 0 LPT-orders the work-item table and zeroes the scheduler rings via
    :func:`order_body`, block 1 builds the per-batch TMA-descriptor arrays via :func:`build_descs_body`,
    one warp per array."""
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
        n_heads_out = cutlass.Int32(q.shape[1] if q.shape[1] >= v.shape[1] else v.shape[1])
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
            base_o,
            base_checkpoint,
            base_z,
            base_pre,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            v,
            o,
            state_checkpoints_out,
            z,
            o_pre,
            n_batch,
            checkpoint_every_n,
            b_t,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    state_checkpoints_out: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_counter: cute.Tensor,
    checkpoint_every_n: cutlass.Int32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """One-launch prologue: LPT-order the work items and build the per-(b,h) TMA-descriptor arrays
    (Q, K, V, O, checkpoints, z, pre-gate O) into ``tensormap_workspace``."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]

    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q_row_stride, q_head_stride, 1)))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k_row_stride, k_head_stride, 1)))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v_head_stride, v_row_stride)))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_v = tma.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    base_desc_o = base_desc_v
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(o.iterator, cute.make_layout((d_v, heads_out, o.shape[0]), stride=(1, o.stride[1], o.stride[0])))
        base_desc_o = tma.create_tensor_map_tiled_from_view(o_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    base_desc_checkpoint = base_desc_v
    if cutlass.const_expr(state_checkpoints_out is not None):
        d_v_state = state_checkpoints_out.shape[2]
        d_k_state = state_checkpoints_out.shape[3]
        checkpoint_box_elements = 128 // (state_checkpoints_out.element_type.width // 8)
        checkpoint_view = cute.make_tensor(
            state_checkpoints_out.iterator,
            cute.make_layout(
                (d_k_state, d_v_state, state_checkpoints_out.shape[0], heads_out),
                stride=(state_checkpoints_out.stride[3], state_checkpoints_out.stride[2], state_checkpoints_out.stride[0], state_checkpoints_out.stride[1]),
            ),
        )
        base_desc_checkpoint = tma.create_tensor_map_tiled_from_view(
            checkpoint_view, box_dims=(checkpoint_box_elements, d_v_state, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle_128b
        )

    base_desc_z = base_desc_v
    if cutlass.const_expr(z is not None):
        z_headed = cute.make_tensor(z.iterator, cute.make_layout((d_v, heads_out, z.shape[0]), stride=(1, z.stride[1], z.stride[0])))
        base_desc_z = tma.create_tensor_map_tiled_from_view(z_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    base_desc_pre = base_desc_o
    if cutlass.const_expr(o_pre is not None):
        pre_headed = cute.make_tensor(o_pre.iterator, cute.make_layout((d_v, heads_out, o_pre.shape[0]), stride=(1, o_pre.stride[1], o_pre.stride[0])))
        base_desc_pre = tma.create_tensor_map_tiled_from_view(pre_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    frost_mamba2_prefill_prologue(
        order_gen,
        b_t,
        base_desc_q,
        base_desc_k,
        base_desc_v,
        base_desc_o,
        base_desc_checkpoint,
        base_desc_z,
        base_desc_pre,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        o,
        state_checkpoints_out,
        z,
        o_pre,
        work_item_staging,
        work_count,
        work_items,
        scheduler_counter,
        cutlass.Int32(batch_size),
        checkpoint_every_n,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    d: Optional[cute.Tensor],
    z: Optional[cute.Tensor],
    o: Optional[cute.Tensor],
    o_pre: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    scheduler_counter: cute.Tensor,
    transition: Optional[cute.Tensor],
    checkpoint_every_n_tokens: cutlass.Int32,
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

    # ---- head-major views: (tokens, dim, heads) k / q, (dim, tokens, heads) v / o, (V, K, heads, seqs) states ----
    q = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], q.shape[2], q.shape[1]), stride=(q.stride[0], q.stride[2], q.stride[1])))
    k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[2], k.shape[1]), stride=(k.stride[0], k.stride[2], k.stride[1])))
    v = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[0], v.shape[1]), stride=(v.stride[2], v.stride[0], v.stride[1])))
    if cutlass.const_expr(o is not None):
        o = cute.make_tensor(o.iterator, cute.make_layout((o.shape[2], o.shape[0], o.shape[1]), stride=(o.stride[2], o.stride[0], o.stride[1])))
    if cutlass.const_expr(state_in is not None):
        state_in = cute.make_tensor(
            state_in.iterator,
            cute.make_layout(
                (state_in.shape[2], state_in.shape[3], state_in.shape[1], state_in.shape[0]),
                stride=(state_in.stride[2], state_in.stride[3], state_in.stride[1], state_in.stride[0]),
            ),
        )
    if cutlass.const_expr(state_out is not None):
        state_out = cute.make_tensor(
            state_out.iterator,
            cute.make_layout(
                (state_out.shape[2], state_out.shape[3], state_out.shape[1], state_out.shape[0]),
                stride=(state_out.stride[2], state_out.stride[3], state_out.stride[1], state_out.stride[0]),
            ),
        )

    # ---- SMEM sizing: per-buffer element cosizes -------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    q_tile_elements = cfg.b_t * cfg.d_k
    k_tile_elements = cfg.b_t * cfg.d_k
    v_tile_elements = cfg.d_v * cfg.b_t
    o_tile_elements = cfg.d_v * cfg.b_t
    cfg.q_cosize = q_tile_elements * cfg.smem_q_stages
    cfg.k_cosize = k_tile_elements * cfg.smem_k_stages
    cfg.v_cosize = v_tile_elements * cfg.smem_v_stages
    cfg.z_cosize = v_tile_elements * cfg.smem_z_stages
    cfg.state_cosize = cfg.d_k * cfg.d_v
    cfg.u_cosize = v_tile_elements
    cfg.o_cosize = o_tile_elements * cfg.smem_o_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    cfg.tma_q_bytes = q_tile_elements * bytes_per_element
    cfg.tma_k_bytes = k_tile_elements * bytes_per_element
    cfg.tma_v_bytes = v_tile_elements * bytes_per_element
    cfg.tma_z_bytes = v_tile_elements * bytes_per_element
    cfg.tma_o_bytes = o_tile_elements * bytes_per_element

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cutlass.min(cutlass.Int32(cfg.max_active_clusters), cutlass.max(cutlass.Int32(work_items.shape[0]), cutlass.Int32(1))), 1, 1)

    frost_mamba2_prefill(
        cfg,
        q_ratio,
        k_ratio,
        v_ratio,
        gate,
        a,
        dt_bias,
        d,
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler_counter,
        transition,
        checkpoint_every_n_tokens,
        dt_min,
        dt_max,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        q,
        k,
        v,
        z,
        o,
        o_pre,
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


frost_mamba2_prefill_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_prefill.set_name_prefix("cudnn", remove_cutlass_symbol=False)
