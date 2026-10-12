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
Chunked Mamba-2 SSD backward summary kernel for SM100 / SM103 / SM107 (Cutlass primitives): the state-gradient
recurrence alone, with no attention scores and no dQ / dK / dV.  Serves the chain's backward summary (every piece's
start gradient from a zero end gradient) and the standalone MAMBA2_SUMMARY_BWD node (every sequence's d_initial_state
from d_final_state).

Algorithm overview (chunks c = NT-1 .. 0 of C = 128 tokens):
  Inputs : Q[BT,DK], dO[BT,DV], Gate[BT] (raw dt), A[H], Dt_bias[H], dS_NT = d_final_state (or zero)
  State  : dS[DK,DV] (fp32 in TMEM, carried backward)

  Gate warp: cumsumlog[t] = sum_{l<=t} dt_l A / ln 2, beta[t] = dt_t; with ``mTransition`` bound, the tile's log2
  decay product per (sequence, head)

  Per chunk:
    dO'[i,:]  = exp2(cumsumlog_i) dO[i,:]                  (bf16 SMEM, CG1: the other dO buffer)
    dS       += Q^T @ dO'                                   (the one GEMM; dS entered pre-scaled by gamma_c)
    dS       <- exp2(cumsumlog_127 of chunk c-1) dS         (in place, CG2: the next chunk's gate stage, peeked)
  dS_0 of every item goes to its own row of ``d_initial_state``.

SMEM layout:
  Buffer                           Size (B)  Stages
  Q                                32768     1
  dO                               16384     2      <-- the two buffers alternate between dO (TMA) and dO' (CG1)
  cumsumlog / beta                   512     2
  scheduler ring, mbarriers

TMEM layout (``d_v`` columns allocated, 32 at least):
  cols 0-d_v   : dS accumulator (fp32), lane = d_k row

Warp assignments (11 warps = 352 threads, one register budget):
  warps 0-3     : compute group 1 - the dO' build, one token per thread
  warps 4-7     : compute group 2 - the dstate pass (seed, gamma pre-scale, dS_0 store), one d_k row per thread
  warp  8       : gate warp      - dt transform + cumsum tables, the transition
  warp  9       : TMA load warp  - loads Q, dO
  warp  10      : MMA warp       - dS += Q^T dO'; TMEM lifecycle
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma

from ..common.thd import emit_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_head, decode_work_item, order_body
from cudnn.frost.tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
    wait_on_dependent_grids,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ss
from cudnn.frost.tile_dsl.pointwise import f16x2_to_f32, fmul2, fp32_to_fp16, opaque_f32_zero, softplus
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from .mamba2_bprop_summary_config import CFG

USE_PDL = True
RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log decays -> the kernel's log2 domain


class Mamba2BpropSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_do_prime_ready: MBarrier
    mb_do_prime_mma_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_dstate_scaled: MBarrier
    mb_dstate_acc_ready: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> Mamba2BpropSummaryBars:
    """Mamba2BpropSummaryBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG2_THREADS = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    GATE_CONSUMERS = CG1_THREADS + CG2_THREADS
    SCHEDULER_CONSUMERS = len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_2_warp_ids) + 2

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    return Mamba2BpropSummaryBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_ready=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_prime_ready=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_do_prime_mma_done=MBarrier(
            alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_CONSUMERS, producer=Producer.THREAD),
        mb_dstate_scaled=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG2_THREADS, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(
            alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=SCHEDULER_CONSUMERS, producer=Producer.THREAD
        ),
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
    """Gate producer (warp 8): the per-chunk dt transform and cumsum tables in walk order; with ``mTransition`` bound,
    the tile's log2 decay product per (sequence, head)."""
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
    sQ_trans,
    sDo_trans,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 10): dS += Q^T dO' of every chunk; dO' alternates between the two dO buffers."""
    q_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    do_prime_index = PipelineState.start(phase=0)
    dstate_scaled_index = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(cfg.tmem_columns), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)

    # ---- dS += Q^T @ dO' descriptor --------------------------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
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

    tmem_col = tmem_base_slot.load() & 0xFFFF
    dstate_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_dstate_acc_offset, cutlass.Float32)
    desc_q_trans = desc_opaque(sQ_trans[0].desc())

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
            desc_do_prime_trans = desc_opaque(sDo_trans[do_prime_buffer].desc())

            # ---- dS += Q(S)^T @ dO'(S) ----------------------------------------------------
            bars.mb_q_ready[0].wait(q_index.phase)
            q_index = advance(q_index, 1)
            if have_dstate:
                bars.mb_dstate_scaled[0].wait(dstate_scaled_index.phase)
                dstate_scaled_index = advance(dstate_scaled_index, 1)
            bars.mb_do_prime_ready[do_prime_buffer].wait(do_prime_index.phase)
            mma_ss(bmm_qt_do_prime_desc, desc_q_trans, desc_do_prime_trans, dstate_acc_ptr, accumulate=have_dstate)
            if elect_one:
                bars.mb_dstate_acc_ready[0].arrive(cta_group=1)
                bars.mb_q_done[0].arrive(cta_group=1)
                bars.mb_do_prime_mma_done[do_prime_buffer].arrive(cta_group=1)
                bars.mb_do_mma_done[do_buffer].arrive(cta_group=1)

            do_index = advance(do_index, cfg.smem_do_stages)
            do_prime_index = advance(do_prime_index, cfg.smem_do_stages)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base_slot.load() & 0xFFFF, cutlass.Int8),
        cutlass.Int32(cfg.tmem_columns),
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
    sDo_raw,
    tidx,
    desc_q_base,
    desc_do_base,
    mScheduler,
    sScheduler,
    bars,
    q_ratio,
):
    """TMA-LDG warp role (warp 9): the per-chunk dO and Q loads of the reverse walk; dO(c) lands in buffer
    ``do_step & 1``."""
    q_index = PipelineState.start(phase=1)
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
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )

        head_o, v_offset = decode_head(cfg, head_idx)
        head_q = head_idx // q_ratio
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_do_slot)

        for reverse_idx in cutlass.range(compute_end - write_start):
            chunk_idx = compute_end - 1 - reverse_idx
            token_coordinate = chunk_idx * cutlass.Int32(cfg.b_t)

            # ---- dO load -------------------------------------------------------------------
            do_buffer = do_step & cutlass.Int32(1)
            prior_do_prime = (do_step + do_buffer) >> 1
            bars.mb_do_prime_mma_done[do_buffer].wait((prior_do_prime & cutlass.Int32(1)) ^ cutlass.Int32(1))
            if elect_one:
                bars.mb_do_ready[do_buffer].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, v_offset, head_o, token_coordinate)
            tma_load_tile(sDo_tma[do_buffer], do_slice, bars.mb_do_ready[do_buffer].smem_ptr, acquire=False)
            do_step = do_step + cutlass.Int32(1)

            # ---- Q load --------------------------------------------------------------------
            bars.mb_q_done[0].wait(q_index.phase)
            q_index = advance(q_index, 1)
            if elect_one:
                bars.mb_q_ready[0].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, token_coordinate)
            tma_load_tile(sQ_tma[0], q_slice, bars.mb_q_ready[0].smem_ptr, acquire=False)

        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)

    bars.mb_q_done[0].wait(q_index.phase)
    for b in cutlass.range_constexpr(cfg.smem_do_stages):
        buffer = cutlass.Int32(b)
        loads = (do_step + cutlass.Int32(1) - buffer) >> 1
        bars.mb_do_mma_done[buffer].wait((loads & cutlass.Int32(1)) ^ cutlass.Int32(1))


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
    sDo_raw,
    sCumsumlog,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 0-3): the dO' build, one token per thread."""
    gate_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1

    elect_one = nvvm.elect_sync()
    bytes_per_element = cfg.io_dtype.width // 8
    vector_width = 16 // bytes_per_element
    num_vectors = cutlass.const_expr(cfg.d_v // vector_width)
    words_per_vector = cutlass.const_expr(vector_width // 2)
    row_swizzle_elements = (cg1_tidx & 7) * vector_width
    do_stage_elements = cfg.do_cosize // cfg.smem_do_stages
    sDo_base = sDo_raw.data_ptr()
    do_step = cutlass.Int32(0)

    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = compute_end - write_start
        for reverse_idx in cutlass.range(n_local):  # noqa: B007
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            gamma_token = cute.math.exp2(sCumsumlog[cg1_tidx, 0, gate_idx], fastmath=True)

            # ---- dO' build: dO'[i, :] = Gamma_i dO[i, :] -> the other dO buffer ---------------
            do_buffer = do_step & cutlass.Int32(1)
            do_prime_buffer = cutlass.Int32(1) - do_buffer
            bars.mb_do_ready[do_buffer].wait(do_index.phase)
            prior_do = (do_step + do_buffer) >> 1
            bars.mb_do_mma_done[do_prime_buffer].wait((prior_do & cutlass.Int32(1)) ^ cutlass.Int32(1))
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
            bars.mb_gate_done[gate_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


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
    sCumsumlog,
    mDstate_in,
    mDstate0,
    sScheduler,
    bars,
):
    """Compute warp-group 2 role (warps 4-7): the dstate pass (the dS seed, the gamma pre-scale, the dS_0 store), one
    d_k row per thread."""
    gate_index = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)
    num_threads_cg2 = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    cg2_tidx = tidx % num_threads_cg2
    elect_one = nvvm.elect_sync()
    ldtm_width = 32
    num_dv_pieces = cutlass.const_expr(cfg.d_v // ldtm_width)
    zero = opaque_f32_zero()
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_dstate_col = tmem_col + cfg.tmem_dstate_acc_offset
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = compute_end - write_start
        if n_local > 0:
            # ---- dS seed: d_final_state or zeros -> TMEM, pre-scaled by gamma_{NT-1} ---------
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

            for reverse_idx in cutlass.range(n_local):  # noqa: B007
                chunk_idx = compute_end - 1 - reverse_idx
                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)

                # ---- dstate pass -----------------------------------------------------------------
                bars.mb_dstate_acc_ready[0].wait(dstate_acc_index.phase)
                dstate_acc_index = advance(dstate_acc_index, 1)
                dstate_vectors = []
                for i in cutlass.range_constexpr(num_dv_pieces):
                    dstate_vectors.append(
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_col + i * ldtm_width, cutlass.Float32), num=32)
                    )
                nvvm.tcgen05_wait("load")
                if chunk_idx > write_start:
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
                else:
                    # ---- dS_0 store ------------------------------------------------------------------
                    if chunk_idx == 0:
                        gDstate0 = mDstate0[None, None, head_o, batch_idx]
                        for p in cutlass.range_constexpr(cfg.d_v):
                            gDstate0[p + v_offset, cg2_tidx] = dstate_vectors[p // ldtm_width][p % ldtm_width].to(mDstate0.element_type)
                bars.mb_gate_done[gate_idx].arrive()
        else:
            # ---- zero-length sequence: dS_0 = d_final_state or zeros ----------------------------
            gDstate0 = mDstate0[None, None, head_o, batch_idx]
            if cutlass.const_expr(cfg.use_dstate_in):
                gDstate_in = mDstate_in[None, None, head_o, batch_idx]
                for p in cutlass.range_constexpr(cfg.d_v):
                    gDstate0[p + v_offset, cg2_tidx] = gDstate_in[p + v_offset, cg2_tidx].to(mDstate0.element_type)
            else:
                for p in cutlass.range_constexpr(cfg.d_v):
                    gDstate0[p + v_offset, cg2_tidx] = cutlass.Float32(0.0).to(mDstate0.element_type)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)
    bars.mb_tmem_done[0].arrive()


@cute.kernel
def frost_mamba2_bprop_summary(
    cfg: cutlass.Constexpr,
    desc_q_array: cutlass.Constexpr[int],
    desc_do_array: cutlass.Constexpr[int],
    q_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA: cute.Tensor,
    mDt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    mDstate_in: Optional[cute.Tensor],
    mDstate0: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    mTransition: Optional[cute.Tensor],
    dt_min: cutlass.Float32,
    dt_max: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Mamba-2 backward summary kernel: warp-specialized persistent tile loop over the dS recurrence alone.  The
    descriptor array indices select Q and dO inside ``tensormap_workspace``: (0, 1) for this module's own prologue, the
    bprop's positions when the chain shares its arrays."""
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
    desc_q_base = desc_base_words + cutlass.Int32(desc_q_array) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(desc_do_array) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWIZZLE_LAYOUT = 2
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
    sQ_raw = cutlass.Array(cfg.io_dtype, cfg.q_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_trans = SmemTile(
        base=sQ_raw.data_ptr(), elems_per_stage=cfg.q_cosize, stages=1, leading_byte_offset=KT_LEAD, stride_byte_offset=STRIDE, layout=SWIZZLE_LAYOUT
    )
    sDo_raw = cutlass.Array(cfg.io_dtype, cfg.do_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDo_trans = SmemTile(
        base=sDo_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWIZZLE_LAYOUT,
    )
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

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
        bars.mb_do_prime_ready[s].init()
        bars.mb_do_prime_mma_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for b in (
        bars.mb_dstate_scaled,
        bars.mb_dstate_acc_ready,
        bars.mb_tmem_done,
    ):
        b[0].init()
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization ---------------------------------------------------------
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
            sDo_raw=sDo_raw,
            sCumsumlog=sCumsumlog,
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
            sCumsumlog=sCumsumlog,
            mDstate_in=mDstate_in,
            mDstate0=mDstate0,
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
            sQ_trans=sQ_trans,
            sDo_trans=sDo_trans,
            sScheduler=sScheduler,
            bars=bars,
        )
    elif warp_idx == cfg.tma_qdo_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sQ_raw=sQ_raw,
            sDo_raw=sDo_raw,
            tidx=tidx,
            desc_q_base=desc_q_base,
            desc_do_base=desc_do_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
            q_ratio=q_ratio,
        )


@dataclass
class Mamba2BpropSummaryCfg:
    """Per-compile Mamba-2 backward summary kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, the d_final_state port) are the ``cute.compile`` cache keys; the rest is
    derived from the module-global ``CFG`` constants.  ``host`` stamps the shape-derived fields at trace time.  Passed
    ``cfg``-first (a ``cutlass.Constexpr``) into ``host`` / ``kernel`` and every warp body.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    use_dstate_in: bool
    d_k: int
    d_v: int
    dt_softplus: bool = True
    tiles_per_head: int = 1
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    compute_group_2_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_2_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_qdo_warp_id: int = CFG.TMA_QDO_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_do_stages: int = CFG.SMEM_DO_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_dstate_acc_offset: int = 0
    tmem_columns: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    q_cosize: int = 0
    do_cosize: int = 0
    tma_q_bytes: int = 0
    tma_do_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_dstate_in: bool,
    dt_softplus: bool = True,
    d_k: int,
    d_v: int,
) -> Mamba2BpropSummaryCfg:
    """Build the per-compile ``Mamba2BpropSummaryCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32)."""
    cfg = Mamba2BpropSummaryCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_dstate_in=use_dstate_in,
        dt_softplus=dt_softplus,
        d_k=d_k,
        d_v=d_v,
    )
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    n_cg2 = len(cfg.compute_group_2_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (3 + n_cg1 + n_cg2)
    cfg.tmem_user_threads = cfg.threads_per_warp * (1 + n_cg2)
    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_columns = max(32, cfg.d_v)
    assert cfg.d_k == cfg.threads_per_warp * n_cg2, "the dstate pass holds one d_k row per compute group 2 thread"
    assert cfg.tmem_columns & (cfg.tmem_columns - 1) == 0 and cfg.tmem_columns <= 512, "the TMEM allocation is a power of two"
    return cfg


TENSORMAP_DESC_ARRAYS = 2  # per-batch runtime TMA descriptors: Q, dO


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_do,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    do: cute.Tensor,
    n_batch: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array, inside the prologue kernel after its order pass."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_seq_descs(base_do, sub1, cu_seqlens, do, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_mamba2_bprop_summary_prologue(
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    do: cute.Tensor,
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
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
            base_do,
            desc_workspace,
            cu_seqlens,
            q,
            do,
            n_batch,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    do: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """One-launch prologue: LPT-order the work items and build the per-(b,h) TMA-descriptor arrays (Q, dO) into
    ``tensormap_workspace``."""
    h_q = q.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= do.shape[1] else do.shape[1]
    d_v = do.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elements = 128 // bytes_per_element
    bt = b_t
    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q.stride[0], q.stride[1], 1)))
    do_headed = cute.make_tensor(do.iterator, cute.make_layout((d_v, heads_out, seqlen), stride=(1, do.stride[1], do.stride[0])))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, box_elements), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_do = tma.create_tensor_map_tiled_from_view(do_headed, box_dims=(box_elements, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)
    frost_mamba2_bprop_summary_prologue(
        order_gen,
        b_t,
        base_desc_q,
        base_desc_do,
        tensormap_workspace,
        cu_seqlens,
        q,
        do,
        work_item_staging,
        work_count,
        work_items,
        scheduler_counter,
        cutlass.Int32(batch_size),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    desc_q_array: cutlass.Constexpr[int],
    desc_do_array: cutlass.Constexpr[int],
    q: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    a: cute.Tensor,
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    dstate0: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
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

    # ---- head-major views: (V, K, heads, seqs) state gradients ---------------------------------
    if cutlass.const_expr(dstate_in is not None):
        dstate_in = cute.make_tensor(
            dstate_in.iterator,
            cute.make_layout(
                (dstate_in.shape[2], dstate_in.shape[3], dstate_in.shape[1], dstate_in.shape[0]),
                stride=(dstate_in.stride[2], dstate_in.stride[3], dstate_in.stride[1], dstate_in.stride[0]),
            ),
        )
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
    cfg.do_cosize = v_tile_elements * cfg.smem_do_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    cfg.tma_q_bytes = q_tile_elements * bytes_per_element
    cfg.tma_do_bytes = v_tile_elements * bytes_per_element

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cutlass.min(cutlass.Int32(cfg.max_active_clusters), cutlass.max(cutlass.Int32(work_items.shape[0]), cutlass.Int32(1))), 1, 1)

    frost_mamba2_bprop_summary(
        cfg,
        desc_q_array,
        desc_do_array,
        q_ratio,
        gate,
        a,
        dt_bias,
        cu_seqlens,
        dstate_in,
        dstate0,
        work_items,
        work_count,
        scheduler_counter,
        transition,
        dt_min,
        dt_max,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
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


frost_mamba2_bprop_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_mamba2_bprop_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
