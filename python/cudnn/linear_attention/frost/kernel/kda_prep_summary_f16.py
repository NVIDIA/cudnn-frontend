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
Prep-fed chunked Kimi Delta Attention (KDA) fused state-summary kernel for SM100 / SM103 / SM107 (Cutlass primitives): the
BT = 16 recurrence run twice per chunk over the records of kda_prep_f16 (K decay k_decay, T, diag), chain H (the state from
zero or initial_state, consuming V) and chain M (the transition, from the identity with V = 0), one persistent CTA per
(sequence, head, d_v half) at the tile counts where the slot budget is 2.

Algorithm overview (per chunk c, tokens [cC, (c+1)C), for each chain X in {H, M}; beta and the chunk inverse are folded
into the records):
  Inputs : k_decay[BT,DK], T[BT,DK], diag[DK] = exp2(g[BT-1,:]), V[BT,DV] (chain H only)
  State  : S_X[DK,DV]  (recurrent state, held in TMEM, fp32 carry; a b16 copy is the K*state GEMM's A operand; H from
                        zero / initial_state, M from I)
  K*state GEMM   : KS_X[BT,DV] = k_decay @ S_X
  Y              : Y_H = V - KS_H, Y_M = 0 - KS_M   (BF16 path subtracts in fp32, then rounds once; beta is already folded
                                                     into the records)
  KV update GEMM : S_upd_X[DK,DV] = T^T @ Y_X       (the T record in the K restore slot; left then right key half, H then
                                                     M per stage, so chain M trails H by one stage)
  Epilogue:
    S_X       = diag .* S_X + S_upd_X               (per-channel decay of the state in TMEM, then the update)
    output_state = S_H, output_transition = S_M     (stored domain M_buf = M^T; X_final = X_init @ M_buf + X_H)

An item loads its initial state when compute_start == 0 and stores when write_end == batch_num_chunks; an empty item passes
initial_state (or zero) through as H and writes M = I.  Under tiles_per_head = 2 each CTA owns half of d_v's rows of H and,
with a square state, half of M's rows (d_m = d_k / 2); otherwise chain M runs whole and the first tile stores it.

prep_summary_host issues the prologue, kda_prep_f16 (summary_records: the k_decay, T and diag records only, no Q input) and
this kernel from one host.

SMEM layout (stage counts live in kda_summary_config.py and build_cfg; sizes at DK = 128, DV = 64, bf16 io):
  Buffer                       Size (B)  Stages
  V (raw)                          2048       8
  diag (fp32)                       512       8
  k_decay / T records          2 x 4096       4
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated; chain H at 0, chain M at 240):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 (doubles as the final state acc)
  state input              64     <-- b16 state staging (K*state A operand)
  K*state acc              16
  (unused)                 24
  U input                   8     <-- b16 packed Y

Warp assignments (16 warps = 512 threads):
  warps 0-7     : compute group 0 - left key halves of both states: b16 pack, fp32 decay (two ping-pong groups)
  warps 8-11    : compute group 1 - initial states, right key halves, Y of both chains, H / M stores
  warp  12      : scheduler warp - the scheduler protocol only
  warp  13      : MMA warp       - every tcgen05 GEMM of both chains; TMEM lifecycle
  warp  14      : TMA load warp  - V and the k_decay, T and diag records
  warp  15      : scheduler warp - the scheduler protocol only
"""

from dataclasses import dataclass
import functools
from typing import NamedTuple, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_head, decode_work_item, order_body
from ..common.host import get_dtype
from ..common.thd import TENSOR_MAP_QWORDS, emit_seq_descs, emit_tile_seq_descs
from . import kda_prep_f16
from .gdn_tinv_f16 import emit_tinv_rows
from .kda_summary_config import CFG

from cudnn.frost.tile_dsl.barrier import (
    wait_on_dependent_grids,
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import desc_opaque, mma_ts_step
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from cudnn.frost.tile_dsl.pointwise import (
    opaque_f32_zero,
    opaque_i32_zero,
    fmul2,
    fp32_to_fp16,
)

from ..common.residual import residual_operand

USE_PDL = True
STATE_DIMS = (64, 128)

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0


class KdaPrepSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring: the shared
    stream and operand rings, then each recurrence's own handoffs, chain H then chain M."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier

    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier

    mb_state_k_acc_h_ready: MBarrier
    mb_state_input_h_cg1_ready: MBarrier
    mb_state_input_h_cg0_ready: MBarrier
    mb_u_input_h_ready: MBarrier
    mb_state_acc_h_cg0_done: MBarrier
    mb_state_acc_h_cg1_done: MBarrier
    mb_state_k_acc_m_ready: MBarrier
    mb_state_input_m_cg1_ready: MBarrier
    mb_state_input_m_cg0_ready: MBarrier
    mb_u_input_m_ready: MBarrier
    mb_state_acc_m_cg0_done: MBarrier
    mb_state_acc_m_cg1_done: MBarrier


def make_bars(cfg) -> KdaPrepSummaryBars:
    """KdaPrepSummaryBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    CG0_GROUP_WARPS = cfg.cg0_warps_per_group
    CG1_WARPS = len(cfg.compute_group_1_warp_ids)

    return KdaPrepSummaryBars(
        mb_raw_ready=MBarrier(alloc(cfg.smem_raw_stages), spin=True, stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_raw_done=MBarrier(
            alloc(cfg.smem_raw_stages), spin=True, stages=cfg.smem_raw_stages, init_count=CG0_GROUP_WARPS + CG1_WARPS, producer=Producer.THREAD
        ),
        mb_k_decay_inv_cg0_ready=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_decay_tcgen05_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_k_restore_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), spin=True, stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), spin=True, stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
        mb_state_k_acc_h_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_input_h_cg1_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD),
        mb_state_input_h_cg0_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG0_GROUP_WARPS, producer=Producer.THREAD),
        mb_u_input_h_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG1_WARPS + CG0_GROUP_WARPS, producer=Producer.THREAD),
        mb_state_acc_h_cg0_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_h_cg1_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_k_acc_m_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_input_m_cg1_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD),
        mb_state_input_m_cg0_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG0_GROUP_WARPS, producer=Producer.THREAD),
        mb_u_input_m_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG1_WARPS + CG0_GROUP_WARPS, producer=Producer.THREAD),
        mb_state_acc_m_cg0_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_m_cg1_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
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
def scheduler_warp(
    cfg,
    total_tiles,
    bidx,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    bars,
) -> None:
    """Scheduler warp role (warps 12 and 15): the scheduler protocol only."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    sTmem_base,
    sK_decay,
    sK_restore_trans,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): issues every state GEMM of both chains (H first, M second at every stage) and
    owns the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(sTmem_base, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    state_update_n = cutlass.const_expr(cfg.d_k // 2 if cfg.d_k // 2 >= 64 else cfg.d_k)
    state_update_split = cutlass.const_expr(state_update_n < cfg.d_k)
    k_restore_right_bytes = cutlass.const_expr(cfg.b_t * 128)
    state_input_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_h_offset, cutlass.Int8)
    state_k_acc_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_h_offset, cutlass.Float32)
    u_input_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_h_offset, cutlass.Int8)
    state_dst_cg0_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_h_offset, cutlass.Float32)
    state_dst_cg1_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_h_offset + state_update_n, cutlass.Float32)
    state_input_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_m_offset, cutlass.Int8)
    state_k_acc_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_m_offset, cutlass.Float32)
    u_input_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_m_offset, cutlass.Int8)
    state_dst_cg0_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_m_offset, cutlass.Float32)
    state_dst_cg1_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_m_offset + state_update_n, cutlass.Float32)
    state_input_cg1_index_h = PipelineState.start(phase=0)
    state_input_cg0_index_h = PipelineState.start(phase=0)
    state_input_cg1_index_m = PipelineState.start(phase=0)
    state_input_cg0_index_m = PipelineState.start(phase=0)
    u_input_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)

    # ---- chunk-invariant GEMM descriptors (rows = DV for chain H, d_m for chain M) ----
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_acc_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.d_v, b_major=0
    )
    instruction_descriptor_acc_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.d_m, b_major=0
    )
    instruction_descriptor_final_state_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=state_update_n, m_dim=cfg.d_v, b_major=1
    )
    instruction_descriptor_final_state_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=state_update_n, m_dim=cfg.d_m, b_major=1
    )
    bmm_state_k_decay_desc_h = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_k_decay_desc_m = MmaDesc(
        M=cfg.d_m,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc_h = MmaDesc(
        M=cfg.d_v,
        N=state_update_n,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_final_state_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc_m = MmaDesc(
        M=cfg.d_m,
        N=state_update_n,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_final_state_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    STATE_A_SEG = bmm_state_k_decay_desc_h.sps_B * bmm_state_k_decay_desc_h.tmem_advance_A
    STATE_B_SEG = bmm_state_k_decay_desc_h.smem_subtile_B >> 4
    STATE_K_STEPS_CG0 = bmm_state_k_decay_desc_h.num_k_steps // 2
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        seed_state = compute_start == 0
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            if cutlass.const_expr(cfg.use_initial_state):
                have_state_h = local_chunk_idx > 0 or seed_state
            else:
                have_state_h = local_chunk_idx > 0
            have_state_m = local_chunk_idx > 0 or seed_state
            decay_stage = k_decay_ready.idx
            sK_decay_stage = sK_decay[decay_stage]
            sK_restore_stage = sK_restore_trans[decay_stage]
            desc_k_decay = desc_opaque(sK_decay_stage.desc())
            desc_k_restore = desc_opaque(sK_restore_stage.desc())
            if cutlass.const_expr(state_update_split):
                desc_k_restore_right = desc_k_restore.advance_start_address(k_restore_right_bytes)

            # ---- k state H = state H(T) @ K decay^T, k state M = state M(T) @ K decay^T --
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state_h:
                bars.mb_state_input_h_cg0_ready.wait(state_input_cg0_index_h.phase)
                state_input_cg0_index_h = advance(state_input_cg0_index_h, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc_h.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_h_cg1_ready.wait(state_input_cg1_index_h.phase)
                        state_input_cg1_index_h = advance(state_input_cg1_index_h, 1)
                    s = f // bmm_state_k_decay_desc_h.sps_B
                    k = f - s * bmm_state_k_decay_desc_h.sps_B
                    mma_ts_step(
                        bmm_state_k_decay_desc_h,
                        state_input_ptr_h.subview(s * STATE_A_SEG),
                        desc_k_decay + s * STATE_B_SEG,
                        state_k_acc_ptr_h,
                        k,
                        cutlass.Boolean(f > 0),
                        issue_mma=elect_one,
                    )

                if elect_one:
                    bars.mb_state_k_acc_h_ready.arrive(cta_group=1)
            if have_state_m:
                bars.mb_state_input_m_cg0_ready.wait(state_input_cg0_index_m.phase)
                state_input_cg0_index_m = advance(state_input_cg0_index_m, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc_m.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_m_cg1_ready.wait(state_input_cg1_index_m.phase)
                        state_input_cg1_index_m = advance(state_input_cg1_index_m, 1)
                    s = f // bmm_state_k_decay_desc_m.sps_B
                    k = f - s * bmm_state_k_decay_desc_m.sps_B
                    mma_ts_step(
                        bmm_state_k_decay_desc_m,
                        state_input_ptr_m.subview(s * STATE_A_SEG),
                        desc_k_decay + s * STATE_B_SEG,
                        state_k_acc_ptr_m,
                        k,
                        cutlass.Boolean(f > 0),
                        issue_mma=elect_one,
                    )

                if elect_one:
                    bars.mb_state_k_acc_m_ready.arrive(cta_group=1)

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            # ---- state H += U H(T) @ K restore, then state M: left then right key half --
            bars.mb_u_input_h_ready.wait(u_input_index.phase)
            mma_ts_step(bmm_u_k_restore_desc_h, u_input_ptr_h, desc_k_restore, state_dst_cg0_ptr_h, 0, have_state_h, issue_mma=elect_one)
            if elect_one:
                bars.mb_state_acc_h_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(bmm_u_k_restore_desc_h, u_input_ptr_h, desc_k_restore_right, state_dst_cg1_ptr_h, 0, have_state_h, issue_mma=elect_one)
            if elect_one:
                bars.mb_state_acc_h_cg1_done[decay_stage].arrive(cta_group=1)
            bars.mb_u_input_m_ready.wait(u_input_index.phase)
            mma_ts_step(bmm_u_k_restore_desc_m, u_input_ptr_m, desc_k_restore, state_dst_cg0_ptr_m, 0, have_state_m, issue_mma=elect_one)
            if elect_one:
                bars.mb_state_acc_m_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(bmm_u_k_restore_desc_m, u_input_ptr_m, desc_k_restore_right, state_dst_cg1_ptr_m, 0, have_state_m, issue_mma=elect_one)
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_m_cg1_done[decay_stage].arrive(cta_group=1)
            u_input_index = advance(u_input_index, 1)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
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
    mScheduler,
    sScheduler,
    lane_idx,
    sV_raw,
    sGate_raw,
    desc_v_base,
    desc_k_decay_base,
    desc_t_base,
    desc_diag_base,
    sK_decay_raw,
    sK_restore_raw,
    bars,
    v_ratio,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing the per-chunk V load and three records of
    kda_prep_f16 (k_decay -> K decay, T -> K restore, diag -> the gate stage)."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.gate_cosize // cfg.smem_raw_stages),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(1),
        tma_granu_elems=(cfg.d_k),
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    sK_decay_tma = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sK_restore_tma = SmemTile(
        base=sK_restore_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        head_v = head_idx // v_ratio
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_decay_slot = (desc_k_decay_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_t_slot = (desc_t_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_diag_slot = (desc_diag_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_decay_slot)
            tma_tensormap_acquire(desc_t_slot)
            tma_tensormap_acquire(desc_diag_slot)
            tma_tensormap_acquire(desc_v_slot)

        for chunk_idx in cutlass.range(compute_start, write_end, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t

            # ---- V + diag record: one transaction barrier per stage ------------------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_index.idx].arrive(n_bytes=cfg.tma_v_bytes + cfg.tma_gate_bytes)
            raw_ready_ptr = bars.mb_raw_ready[raw_index.idx].smem_ptr
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, raw_ready_ptr, acquire=False)

            chunk_count = cutlass.Uint32(cum_chunk_base + (chunk_idx - compute_start))
            decay_stage = cutlass.Int32(chunk_count % cfg.smem_decay_stages)
            decay_free_parity = cutlass.Int32(((chunk_count // cfg.smem_decay_stages) + 1) % 2)

            # ---- diag record into the gate stage -------------------------------------
            diag_slice = tma_slice_runtime_desc(desc_diag_slot, cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sGate_tma[raw_index.idx], diag_slice, raw_ready_ptr, acquire=False)

            # ---- k_decay, T records on the decay stage's one barrier -----------------
            bars.mb_decay_tcgen05_done[decay_stage].wait(decay_free_parity)
            bars.mb_k_restore_done[decay_stage].wait(decay_free_parity)
            if elect_one:
                bars.mb_k_decay_inv_cg0_ready[decay_stage].arrive(n_bytes=2 * cfg.tma_k_bytes)
            k_decay_slice = tma_slice_runtime_desc(desc_k_decay_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sK_decay_tma[decay_stage], k_decay_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr, acquire=False)
            t_slice = tma_slice_runtime_desc(desc_t_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sK_restore_tma[decay_stage], t_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr, acquire=False)

            raw_index = advance(raw_index, cfg.smem_raw_stages)
        cum_chunk_base += write_end - compute_start
        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    warp_idx,
    sGate_exchange_raw,
    sTmem_base,
    bars,
) -> None:
    """CG0 warp-group role (warps 0-7): persistent scheduler loop carrying the left key half of both states, b16 pack and
    fp32 decay by the diag record."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_0,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_0 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    row_lo_addr = (tmem_base >> 16) << 16
    state_col_id_h = tmem_col + cfg.tmem_state_acc_h_offset
    packed_col_id_h = tmem_col + cfg.tmem_state_input_h_offset
    state_col_id_m = tmem_col + cfg.tmem_state_acc_m_offset
    packed_col_id_m = tmem_col + cfg.tmem_state_input_m_offset

    scheduler_state = PipelineState.start(phase=0)

    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    dk_halves = cutlass.const_expr(cfg.d_k // 64)

    cg0_group_id = cg0_warp // cfg.cg0_warps_per_group
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        nvvm.barrier_cta_sync(cfg.cg0_tile_entry_barrier_id, thread_count=cfg.cg0_group_count * cfg.cg0_threads_per_group)
        for local_chunk_idx in cutlass.range(cg0_group_id, num_chunks_tile, cfg.cg0_group_count, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_count = cutlass.Uint32(cum_chunk)
            raw_stage = cutlass.Int32(chunk_count % cfg.smem_raw_stages)
            raw_parity = cutlass.Int32((chunk_count // cfg.smem_raw_stages) % 2)
            sGate_exchange_ptr = sGate_exchange_raw.data_ptr() + raw_stage * cfg.d_k
            bars.mb_raw_ready[raw_stage].wait(raw_parity)

            # ---- state stage, left key half: pack, publish, fp32 decay ---------------
            if cum_chunk > 0:
                update_count = chunk_count - cutlass.Uint32(1)
                bars.mb_state_acc_h_cg0_done[cutlass.Int32(update_count % cfg.smem_decay_stages)].wait(
                    cutlass.Int32((update_count // cfg.smem_decay_stages) % 2)
                )
            if local_chunk_idx > 0:
                l_state_vecs = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + b * 32, cutlass.Float32), num=32))
                for b in cutlass.range_constexpr(dk_halves):
                    l_packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
                    for packed_col in cutlass.range_constexpr(16):
                        l_packed[packed_col] = fp32_to_fp16(l_state_vecs[b][2 * packed_col], l_state_vecs[b][2 * packed_col + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_h + b * 16, cutlass.Int8), l_packed[0:16])
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_h_cg0_ready.arrive()

                # ---- fp32 decay of the left key half: state *= exp2(g last) ----------
                for b in cutlass.range_constexpr(dk_halves):
                    l_scaled = []
                    for scale_group in cutlass.range_constexpr(8):
                        scale_dim = b * 32 + scale_group * 4
                        l_scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(2):
                            l_s0, l_s1 = fmul2(
                                l_state_vecs[b][scale_group * 4 + 2 * t],
                                l_state_vecs[b][scale_group * 4 + 2 * t + 1],
                                l_scale_frag[2 * t],
                                l_scale_frag[2 * t + 1],
                            )
                            l_scaled += [l_s0, l_s1]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + b * 32, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")

            # ---- state stage, left key half: pack, publish, fp32 decay ---------------
            if cum_chunk > 0:
                update_count = chunk_count - cutlass.Uint32(1)
                bars.mb_state_acc_m_cg0_done[cutlass.Int32(update_count % cfg.smem_decay_stages)].wait(
                    cutlass.Int32((update_count // cfg.smem_decay_stages) % 2)
                )
            if local_chunk_idx > 0:
                l_state_vecs = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + b * 32, cutlass.Float32), num=32))
                for b in cutlass.range_constexpr(dk_halves):
                    l_packed = cutlass.Array(cutlass.Int32, 16, alignment=16)
                    for packed_col in cutlass.range_constexpr(16):
                        l_packed[packed_col] = fp32_to_fp16(l_state_vecs[b][2 * packed_col], l_state_vecs[b][2 * packed_col + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_m + b * 16, cutlass.Int8), l_packed[0:16])
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_m_cg0_ready.arrive()

                # ---- fp32 decay of the left key half: state *= exp2(g last) ----------
                for b in cutlass.range_constexpr(dk_halves):
                    l_scaled = []
                    for scale_group in cutlass.range_constexpr(8):
                        scale_dim = b * 32 + scale_group * 4
                        l_scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(2):
                            l_s0, l_s1 = fmul2(
                                l_state_vecs[b][scale_group * 4 + 2 * t],
                                l_state_vecs[b][scale_group * 4 + 2 * t + 1],
                                l_scale_frag[2 * t],
                                l_scale_frag[2 * t + 1],
                            )
                            l_scaled += [l_s0, l_s1]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + b * 32, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_stage].arrive()
                bars.mb_u_input_h_ready.arrive()
                bars.mb_u_input_m_ready.arrive()
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    sTmem_base,
    warp_idx,
    mState_out,
    mTransition,
    mState_init,
    mComposed,
    sV_raw,
    sGate_exchange_raw,
    bars,
) -> None:
    """CG1 warp-group role (warps 8-11): persistent scheduler loop seeding both states, packing and rescaling the right
    key half of both, staging the Y inputs of both chains (H first, M second at every stage) and storing H and M.

    H uses V - projection, M uses 0 - projection; the prep records already
    incorporate beta, so Y is not scaled again. Both BF16 paths retain FP32
    projection/residual arithmetic until Y packing; FP16 preserves packed
    arithmetic. Initial and steady stages retain their projection-completion
    waits and packed-operand publication protocol.
    """
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()

    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16

    # ---- ldmatrix.x4 COL lane decode for the V loads ---------------------------------
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    cg1_warp = warp_idx - cfg.compute_group_1_warp_ids[0]
    row_lanes_h = cutlass.const_expr(cfg.threads_per_warp if cfg.d_v == 128 else 16)
    row_lanes_m = cutlass.const_expr(cfg.threads_per_warp if cfg.d_m == 128 else 16)
    value_dim = cg1_warp * row_lanes_h + lane_idx % row_lanes_h
    value_dim_base = cg1_warp * row_lanes_h
    row_valid_h = lane_idx < cutlass.Int32(row_lanes_h)
    key_dim = cg1_warp * row_lanes_m + lane_idx % row_lanes_m
    row_valid_m = lane_idx < cutlass.Int32(row_lanes_m)
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    state_blocks_per_half = cutlass.const_expr(cfg.d_k // 32)
    state_col_id_h = tmem_col + cfg.tmem_state_acc_h_offset
    packed_col_id_h = tmem_col + cfg.tmem_state_input_h_offset
    statek_col_id_h = tmem_col + cfg.tmem_state_k_acc_h_offset
    y_input_col_id_h = tmem_col + cfg.tmem_u_input_h_offset
    state_col_id_m = tmem_col + cfg.tmem_state_acc_m_offset
    packed_col_id_m = tmem_col + cfg.tmem_state_input_m_offset
    statek_col_id_m = tmem_col + cfg.tmem_state_k_acc_m_offset
    y_input_col_id_m = tmem_col + cfg.tmem_u_input_m_offset
    v_swizzle_off_lo = (
        (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2)
    )
    v_swizzle_off_hi = (
        (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2)
    )
    state_k_acc_index_h = PipelineState.start(phase=0)
    state_k_acc_index_m = PipelineState.start(phase=0)
    state_update_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        m_offset = cutlass.Int32(0)
        row_store_m = row_valid_m
        if cutlass.const_expr(cfg.d_m != cfg.d_k):
            m_offset = v_offset
        elif cutlass.const_expr(cfg.tiles_per_head > 1):
            row_store_m = row_valid_m and v_offset == cutlass.Int32(0)
        num_chunks_tile = write_end - compute_start

        if num_chunks_tile > 0:
            seed_state = compute_start == 0
            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)

            if seed_state:
                bars.mb_raw_ready[raw_index.idx].wait(raw_index.phase)
                seed_exchange_ptr = sGate_exchange_raw.data_ptr() + raw_index.idx * cfg.d_k

                # ---- state seed: initial state GMEM -> packed b16 TMEM + fp32 state TMEM ----
                if cutlass.const_expr(cfg.use_initial_state):
                    if cutlass.const_expr(cfg.state_table):
                        seed_vw = 16 // (mState_init.element_type.width // 8)
                        seed_src = (mState_init.iterator + mState_init.layout((batch_idx, head_o, value_dim + v_offset, 0))).raw_ptr()
                    if cutlass.const_expr(cfg.has_composed):
                        is_boundary = batch_idx == cutlass.Int32(0)
                        composed_src = (mComposed.iterator + mComposed.layout((cutlass.Int32(0), head_o, value_dim + v_offset, 0))).raw_ptr()
                    for seed_half in cutlass.range_constexpr(2):
                        seed_blocks_lo = seed_half * state_blocks_per_half
                        seed_blocks_hi = seed_blocks_lo + state_blocks_per_half
                        seed_vecs = []
                        for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                            seed_block = []
                            if cutlass.const_expr(cfg.state_table):
                                for g in cutlass.range_constexpr(16 // seed_vw):
                                    seed_chunk = (seed_src + i * 16 + g * seed_vw).load(count=seed_vw, alignment=16)
                                    for t in cutlass.range_constexpr(seed_vw):
                                        seed_block.append(seed_chunk[t].to(cutlass.Float32))
                            else:
                                for _ in cutlass.range_constexpr(16):
                                    seed_block.append(cutlass.Float32(0.0))
                            if cutlass.const_expr(cfg.has_composed):
                                for g in cutlass.range_constexpr(4):
                                    composed_chunk = (composed_src + i * 16 + g * 4).load(count=4, alignment=16)
                                    for t in cutlass.range_constexpr(4):
                                        seed_block[g * 4 + t] = composed_chunk[t] if is_boundary else seed_block[g * 4 + t]
                            seed_vecs.append(seed_block)

                        for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                            seed_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                            for packed_col in cutlass.range_constexpr(8):
                                seed_pack[packed_col] = fp32_to_fp16(
                                    seed_vecs[i - seed_blocks_lo][2 * packed_col], seed_vecs[i - seed_blocks_lo][2 * packed_col + 1], dtype=cfg.io_dtype
                                )
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_h + i * 8, cutlass.Int8),
                                seed_pack[0:8],
                            )
                        nvvm.tcgen05_wait("store")
                        if cutlass.const_expr(seed_half == 0):
                            if nvvm.elect_sync():
                                bars.mb_state_input_h_cg0_ready.arrive()
                        else:
                            if nvvm.elect_sync():
                                bars.mb_state_input_h_cg1_ready.arrive()

                        # ---- fp32 decay of the seed half: state = seed * exp2(g last) ----
                        for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                            seed_scaled = []
                            for seed_scale_group in cutlass.range_constexpr(4):
                                seed_scale_dim = i * 16 + seed_scale_group * 4
                                seed_scale_frag = (seed_exchange_ptr + seed_scale_dim).load(count=4, alignment=16)
                                for t in cutlass.range_constexpr(2):
                                    seed_s0, seed_s1 = fmul2(
                                        seed_vecs[i - seed_blocks_lo][seed_scale_group * 4 + 2 * t],
                                        seed_vecs[i - seed_blocks_lo][seed_scale_group * 4 + 2 * t + 1],
                                        seed_scale_frag[2 * t],
                                        seed_scale_frag[2 * t + 1],
                                    )
                                    seed_scaled += [seed_s0, seed_s1]
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + i * 16, cutlass.Float32),
                                cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                            )
                    nvvm.tcgen05_wait("store")

                # ---- identity seed: one -> packed b16 TMEM + fp32 state TMEM ---------
                seed_zero = opaque_f32_zero()
                seed_one = seed_zero + cutlass.Float32(1.0)
                for seed_half in cutlass.range_constexpr(2):
                    seed_blocks_lo = seed_half * state_blocks_per_half
                    seed_blocks_hi = seed_blocks_lo + state_blocks_per_half
                    seed_vecs = []
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_block = []
                        for j in cutlass.range_constexpr(16):
                            seed_block.append(seed_one if key_dim + m_offset == i * 16 + j else seed_zero)
                        seed_vecs.append(seed_block)

                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                        for packed_col in cutlass.range_constexpr(8):
                            seed_pack[packed_col] = fp32_to_fp16(
                                seed_vecs[i - seed_blocks_lo][2 * packed_col], seed_vecs[i - seed_blocks_lo][2 * packed_col + 1], dtype=cfg.io_dtype
                            )
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_m + i * 8, cutlass.Int8),
                            seed_pack[0:8],
                        )
                    nvvm.tcgen05_wait("store")
                    if cutlass.const_expr(seed_half == 0):
                        if nvvm.elect_sync():
                            bars.mb_state_input_m_cg0_ready.arrive()
                    else:
                        if nvvm.elect_sync():
                            bars.mb_state_input_m_cg1_ready.arrive()

                    # ---- fp32 decay of the seed half: state = seed * exp2(g last) ----
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_scaled = []
                        for seed_scale_group in cutlass.range_constexpr(4):
                            seed_scale_dim = i * 16 + seed_scale_group * 4
                            seed_scale_frag = (seed_exchange_ptr + seed_scale_dim).load(count=4, alignment=16)
                            for t in cutlass.range_constexpr(2):
                                seed_s0, seed_s1 = fmul2(
                                    seed_vecs[i - seed_blocks_lo][seed_scale_group * 4 + 2 * t],
                                    seed_vecs[i - seed_blocks_lo][seed_scale_group * 4 + 2 * t + 1],
                                    seed_scale_frag[2 * t],
                                    seed_scale_frag[2 * t + 1],
                                )
                                seed_scaled += [seed_s0, seed_s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + i * 16, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                        )
                nvvm.tcgen05_wait("store")

            # ---- Y stage: Y = V - k state --------------------------------------------
            bars.mb_raw_ready[raw_index.idx].wait(raw_index.phase)
            raw_v_frag_lo = nvvm.ldmatrix(sV_ptr + v_swizzle_off_lo, 4, nvvm.MMALayout.COL)
            raw_v_frag_hi = raw_v_frag_lo
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(sV_ptr + v_swizzle_off_hi, 4, nvvm.MMALayout.COL)
            if cutlass.const_expr(cfg.use_initial_state):
                have_state_h = seed_state
            else:
                have_state_h = cutlass.Boolean(False)
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state_h:
                bars.mb_state_k_acc_h_ready.wait(state_k_acc_index_h.phase)
                state_k_acc_index_h = advance(state_k_acc_index_h, 1)
                state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_h, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = residual_operand(
                        raw_v_frag_lo[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_v == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_h, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = residual_operand(
                            raw_v_frag_hi[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = raw_v_frag_lo[reg_idx]
                    y_hi[reg_idx] = raw_v_frag_hi[reg_idx]

            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_h, cutlass.Int8), y_input_pack_lo[0:4])
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_h, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_h_ready.arrive()

            # ---- Y stage: Y = 0 - k state --------------------------------------------
            zero_word = opaque_i32_zero()
            zero_v_frag_lo = [zero_word for _ in range(4)]
            zero_v_frag_hi = [zero_word for _ in range(4)]
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if seed_state:
                bars.mb_state_k_acc_m_ready.wait(state_k_acc_index_m.phase)
                state_k_acc_index_m = advance(state_k_acc_index_m, 1)
                state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_m, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = residual_operand(
                        zero_v_frag_lo[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_m == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_m, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = residual_operand(
                            zero_v_frag_hi[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = zero_v_frag_lo[reg_idx]
                    y_hi[reg_idx] = zero_v_frag_hi[reg_idx]

            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_m, cutlass.Int8), y_input_pack_lo[0:4])
            if cutlass.const_expr(cfg.d_m == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_m, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_m_ready.arrive()
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_index.idx].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            raw_stage = raw_index.idx
            raw_phase = raw_index.phase
            sV_ptr = sV_raw.data_ptr() + raw_stage * (cfg.d_v * cfg.b_t)
            bars.mb_raw_ready[raw_stage].wait(raw_phase)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            sGate_exchange_ptr = sGate_exchange_raw.data_ptr() + raw_stage * cfg.d_k
            bars.mb_state_acc_h_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_vecs = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + i * 16, cutlass.Float32), num=16))

            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                packed_state = cutlass.Array(cutlass.Int32, 8, alignment=16)
                for packed_col in cutlass.range_constexpr(8):
                    packed_state[packed_col] = fp32_to_fp16(
                        state_vecs[i - state_blocks_per_half][2 * packed_col], state_vecs[i - state_blocks_per_half][2 * packed_col + 1], dtype=cfg.io_dtype
                    )
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_h + i * 8, cutlass.Int8),
                    packed_state[0:8],
                )

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
            scaled_blocks = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                scaled = []
                for scale_group in cutlass.range_constexpr(4):
                    scale_dim = i * 16 + scale_group * 4
                    scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t],
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t + 1],
                            scale_frag[2 * t],
                            scale_frag[2 * t + 1],
                        )
                        scaled += [s0, s1]
                scaled_blocks.append(scaled)
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_state_input_h_cg1_ready.arrive()
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + i * 16, cutlass.Float32),
                    cutlass.Vector.from_elements(tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32),
                )

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            bars.mb_state_acc_m_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_vecs = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + i * 16, cutlass.Float32), num=16))

            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                packed_state = cutlass.Array(cutlass.Int32, 8, alignment=16)
                for packed_col in cutlass.range_constexpr(8):
                    packed_state[packed_col] = fp32_to_fp16(
                        state_vecs[i - state_blocks_per_half][2 * packed_col], state_vecs[i - state_blocks_per_half][2 * packed_col + 1], dtype=cfg.io_dtype
                    )
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_m + i * 8, cutlass.Int8),
                    packed_state[0:8],
                )

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
            scaled_blocks = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                scaled = []
                for scale_group in cutlass.range_constexpr(4):
                    scale_dim = i * 16 + scale_group * 4
                    scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t],
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t + 1],
                            scale_frag[2 * t],
                            scale_frag[2 * t + 1],
                        )
                        scaled += [s0, s1]
                scaled_blocks.append(scaled)
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_state_input_m_cg1_ready.arrive()
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + i * 16, cutlass.Float32),
                    cutlass.Vector.from_elements(tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32),
                )
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

            # ---- Y stage: Y = V - k state --------------------------------------------
            have_state = cutlass.Boolean(True)
            raw_v_frag_lo = nvvm.ldmatrix(sV_ptr + v_swizzle_off_lo, 4, nvvm.MMALayout.COL)
            raw_v_frag_hi = raw_v_frag_lo
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(sV_ptr + v_swizzle_off_hi, 4, nvvm.MMALayout.COL)
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state:
                bars.mb_state_k_acc_h_ready.wait(state_k_acc_index_h.phase)
                state_k_acc_index_h = advance(state_k_acc_index_h, 1)
                state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_h, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = residual_operand(
                        raw_v_frag_lo[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_v == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_h, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = residual_operand(
                            raw_v_frag_hi[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = raw_v_frag_lo[reg_idx]
                    y_hi[reg_idx] = raw_v_frag_hi[reg_idx]

            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_h, cutlass.Int8), y_input_pack_lo[0:4])
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_h, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_h_ready.arrive()

            # ---- Y stage: Y = 0 - k state --------------------------------------------
            zero_word = opaque_i32_zero()
            zero_v_frag_lo = [zero_word for _ in range(4)]
            zero_v_frag_hi = [zero_word for _ in range(4)]
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state:
                bars.mb_state_k_acc_m_ready.wait(state_k_acc_index_m.phase)
                state_k_acc_index_m = advance(state_k_acc_index_m, 1)
                state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_m, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = residual_operand(
                        zero_v_frag_lo[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_m == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_m, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = residual_operand(
                            zero_v_frag_hi[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = zero_v_frag_lo[reg_idx]
                    y_hi[reg_idx] = zero_v_frag_hi[reg_idx]

            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_m, cutlass.Int8), y_input_pack_lo[0:4])
            if cutlass.const_expr(cfg.d_m == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_m, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_m_ready.arrive()
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_stage].arrive()

        if num_chunks_tile > 0:
            bars.mb_state_acc_h_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            bars.mb_state_acc_m_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

        owns_final = write_end == batch_num_chunks

        # ---- final state stores; empty items pass the seeds through ------------------
        if batch_seqlen > 0:
            if owns_final:
                # ---- final state store: TMEM -> GMEM ---------------------------------
                state_vw = 16 // (mState_out.element_type.width // 8)
                state_dst = (mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim + v_offset, 0))).raw_ptr()
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    loaded = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + key_block_start, cutlass.Float32),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(32 // state_vw):
                        if row_valid_h:
                            (state_dst + key_block_start + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(loaded[g * state_vw + t].to(mState_out.element_type) for t in range(state_vw)),
                                    mState_out.element_type,
                                ),
                                alignment=16,
                            )

                # ---- final state store: TMEM -> GMEM ---------------------------------
                state_vw = 16 // (mTransition.element_type.width // 8)
                state_dst = (mTransition.iterator + mTransition.layout((batch_idx, head_o, key_dim + m_offset, 0))).raw_ptr()
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    loaded = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + key_block_start, cutlass.Float32),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(32 // state_vw):
                        if row_store_m:
                            (state_dst + key_block_start + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(loaded[g * state_vw + t].to(mTransition.element_type) for t in range(state_vw)),
                                    mTransition.element_type,
                                ),
                                alignment=16,
                            )
        else:
            h_vw = 16 // (mState_out.element_type.width // 8)
            h_dst = (mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim + v_offset, 0))).raw_ptr()
            if cutlass.const_expr(cfg.has_composed):
                is_boundary = batch_idx == cutlass.Int32(0)
                composed_src = (mComposed.iterator + mComposed.layout((cutlass.Int32(0), head_o, value_dim + v_offset, 0))).raw_ptr()
                if cutlass.const_expr(cfg.state_table):
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (mState_init.iterator + mState_init.layout((batch_idx, head_o, value_dim + v_offset, 0))).raw_ptr()
                for g in cutlass.range_constexpr(cfg.d_k // 4):
                    composed_chunk = (composed_src + g * 4).load(count=4, alignment=16)
                    if cutlass.const_expr(cfg.state_table):
                        table_chunk = (seed_src + g * 4).load(count=4, alignment=16)
                    for t in cutlass.range_constexpr(4):
                        pass_val = cutlass.Float32(0.0)
                        if cutlass.const_expr(cfg.state_table):
                            pass_val = table_chunk[t].to(cutlass.Float32)
                        pass_val = composed_chunk[t] if is_boundary else pass_val
                        if row_valid_h:
                            (h_dst + g * 4 + t).store(pass_val.to(mState_out.element_type))
            elif cutlass.const_expr(cfg.use_initial_state):
                seed_vw = 16 // (mState_init.element_type.width // 8)
                seed_src = (mState_init.iterator + mState_init.layout((batch_idx, head_o, value_dim + v_offset, 0))).raw_ptr()
                for g in cutlass.range_constexpr(cfg.d_k // seed_vw):
                    seed_chunk = (seed_src + g * seed_vw).load(count=seed_vw, alignment=16)
                    for t in cutlass.range_constexpr(seed_vw):
                        if row_valid_h:
                            (h_dst + g * seed_vw + t).store(seed_chunk[t].to(mState_out.element_type))
            else:
                h_zero = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0).to(mState_out.element_type) for _ in range(h_vw)), mState_out.element_type)
                for g in cutlass.range_constexpr(cfg.d_k // h_vw):
                    if row_valid_h:
                        (h_dst + g * h_vw).store(h_zero, alignment=16)
            m_vw = 16 // (mTransition.element_type.width // 8)
            m_dst = (mTransition.iterator + mTransition.layout((batch_idx, head_o, key_dim + m_offset, 0))).raw_ptr()
            m_zero = cutlass.Vector.from_elements(tuple(cutlass.Float32(0.0).to(mTransition.element_type) for _ in range(m_vw)), mTransition.element_type)
            for g in cutlass.range_constexpr(cfg.d_k // m_vw):
                if row_store_m:
                    (m_dst + g * m_vw).store(m_zero, alignment=16)
            if row_store_m:
                mTransition[batch_idx, head_o, key_dim + m_offset, key_dim + m_offset] = cutlass.Float32(1.0).to(mTransition.element_type)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    if nvvm.elect_sync():
        bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_v,
    base_k_decay,
    base_t,
    base_diag,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    v: cute.Tensor,
    prep_k_decay: cute.Tensor,
    prep_t: cute.Tensor,
    prep_diag: cute.Tensor,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build inside the prologue kernel after its order pass, one warp per array; warps past
    the array count fall through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_words_v = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_k_decay = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_t = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_diag = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        emit_seq_descs(base_v, desc_words_v, cu_seqlens, v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_tile_seq_descs(base_k_decay, desc_words_k_decay, cu_seqlens, prep_k_decay, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        emit_tile_seq_descs(base_t, desc_words_t, cu_seqlens, prep_t, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        emit_tile_seq_descs(base_diag, desc_words_diag, cu_seqlens, prep_diag, n_batch, b_t, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_kda_prep_summary_prologue(
    b_t: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    tiles_per_head: cutlass.Constexpr[int],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_t: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_diag: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    prep_k_decay: cute.Tensor,
    prep_t: cute.Tensor,
    prep_diag: cute.Tensor,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
    prep_base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_k_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_t: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_desc_words: cute.Tensor,
    prep_rows: cute.Tensor,
    prep_count: cute.Tensor,
) -> None:
    """Three-CTA prologue (block 2 emits kda_prep's descriptor arrays and its row table, so the prep launches without a
    prologue of its own): block 0 synthesizes and LPT-orders the uncut work-item table and zeroes the scheduler rings
    (:func:`order_body`); block 1 builds the per-batch TMA-descriptor arrays (:func:`build_descs_body`), one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(gate.shape[1])
        if cutlass.const_expr(tiles_per_head > 1):
            n_heads_out = n_heads_out * cutlass.Int32(tiles_per_head)
        order_body(
            True,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_heads_out * n_batch,
            cu_seqlens,
            None,
            mCount,
            mWorkItems,
            mScheduler,
            sKey,
            sIdx,
            sSpread,
            num_ctas=num_ctas,
        )
    elif bidx == cutlass.Int32(1):
        build_descs_body(
            widx,
            base_v,
            base_k_decay,
            base_t,
            base_diag,
            desc_workspace,
            cu_seqlens,
            v,
            prep_k_decay,
            prep_t,
            prep_diag,
            n_batch,
            b_t,
        )
    else:
        kda_prep_f16.build_descs_body(
            widx,
            None,
            prep_base_k,
            prep_base_gate,
            prep_base_k_decay,
            None,
            prep_base_t,
            prep_desc_words,
            cu_seqlens,
            None,
            k,
            gate,
            prep_k_decay,
            None,
            prep_t,
            n_batch,
            b_t,
        )
        if widx == cutlass.Int32(0):
            emit_tinv_rows(b_t, 1, cu_seqlens, prep_rows, prep_count, lane_idx)


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
    tiles_per_head: cutlass.Constexpr[int],
    prep_k_decay: cute.Tensor,
    prep_t: cute.Tensor,
    prep_diag: cute.Tensor,
    prep_cfg: cutlass.Constexpr,
    prep_desc_words: cute.Tensor,
    prep_rows: cute.Tensor,
    prep_row_count: cute.Tensor,
):
    """One-launch prologue: synthesize and LPT-order the uncut work items, build the per-batch V / k_decay / t / diag
    TMA-descriptor arrays into ``tensormap_workspace`` and kda_prep's descriptor arrays and row table."""
    h_v = v.shape[1]
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = k.shape[2]
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    seqlen = k.shape[0]

    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))

    swizzle = cuda.TensorMapSwizzle.s128b
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)
    rows = prep_k_decay.shape[0]
    record_maps = []
    for rec in (prep_k_decay, prep_t):
        rec_view = cute.make_tensor(rec.iterator, cute.make_layout((d_k, b_t, n_heads_out, rows), stride=(1, rec.stride[2], rec.stride[1], rec.stride[0])))
        record_maps.append(cuda.create_tensor_map_tiled_from_view(rec_view, box_dims=(box_elems, b_t, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle))
    base_k_decay, base_t = record_maps
    diag_view = cute.make_tensor(prep_diag.iterator, cute.make_layout((d_k, n_heads_out, rows), stride=(1, prep_diag.stride[1], prep_diag.stride[0])))
    base_diag = cuda.create_tensor_map_tiled_from_view(diag_view, box_dims=(d_k, 1, 1), stride_order=(0, 1, 2), swizzle=cuda.TensorMapSwizzle.none)
    _, prep_base_k, prep_base_gate, prep_record_maps = kda_prep_f16.prep_base_maps(prep_cfg, None, k, gate, prep_k_decay, None, prep_t)

    frost_kda_prep_summary_prologue(
        b_t,
        num_ctas,
        tiles_per_head,
        base_v,
        base_k_decay,
        base_t,
        base_diag,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        prep_k_decay,
        prep_t,
        prep_diag,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
        prep_base_k,
        prep_base_gate,
        prep_record_maps[0],
        prep_record_maps[2],
        prep_desc_words,
        prep_rows,
        prep_row_count,
    ).launch(grid=(3, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    composed: cute.Tensor | None,
    final_state: cute.Tensor,
    transition: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    stream,
) -> None:
    heads_out = cutlass.Int32(raw_gate.shape[1])
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        heads_out = heads_out * cutlass.Int32(cfg.tiles_per_head)
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))
    num_sequences = cu_seqlens.shape[0] - 1

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cutlass.min(cutlass.Int32(cfg.max_active_clusters), cutlass.max(cutlass.Int32(work_items.shape[0]), cutlass.Int32(1))), 1, 1)
    frost_kda_prep_summary(
        cfg,
        v_ratio,
        tensormap_workspace,
        cutlass.Int32(num_sequences),
        v,
        cu_seqlens,
        initial_state,
        composed,
        final_state,
        transition,
        work_items,
        work_count,
        scheduler_counter,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.jit
def prep_summary_host(
    cfg: cutlass.Constexpr,
    prep_cfg: cutlass.Constexpr,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    composed: cute.Tensor | None,
    final_state: cute.Tensor,
    transition: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler_counter: cute.Tensor,
    scheduler_all: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    prep_k_decay: cute.Tensor,
    prep_t: cute.Tensor,
    prep_diag: cute.Tensor,
    prep_words: cute.Tensor,
    prep_rows: cute.Tensor,
    prep_row_count: cute.Tensor,
    stream,
) -> None:
    """One compiled launch of the prep-fed summary: the prologue, kda_prep_f16 (``scheduler_counter`` as its steep slot) and
    the summary issued from a single host."""
    prologue(
        cfg.io_dtype,
        cfg.b_t,
        cfg.max_active_clusters,
        k,
        v,
        raw_gate,
        cu_seqlens,
        work_count,
        work_items,
        scheduler_all,
        tensormap_workspace,
        stream,
        cfg.tiles_per_head,
        prep_k_decay,
        prep_t,
        prep_diag,
        prep_cfg,
        prep_words,
        prep_rows,
        prep_row_count,
    )
    kda_prep_f16.host(
        prep_cfg,
        None,
        k,
        prep_words,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        prep_k_decay,
        None,
        prep_t,
        None,
        prep_diag,
        prep_rows,
        prep_row_count,
        scheduler_counter,
        stream,
    )
    host(
        cfg,
        v,
        raw_gate,
        cu_seqlens,
        initial_state,
        composed,
        final_state,
        transition,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        stream,
    )


@cute.kernel
def frost_kda_prep_summary(
    cfg: cutlass.Constexpr,
    v_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mV: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mComposed: cute.Tensor | None,
    mState_out: cute.Tensor,
    mTransition: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
) -> None:
    """BT=16 prep-fed KDA fused H + M summary persistent kernel body: every warp role runs a
    tile-scheduler loop over the tiles."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_v_base = desc_base_words
    desc_k_decay_base = desc_base_words + arr_words
    desc_t_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_diag_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    sTmem_base = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    sK_decay_raw = cutlass.Array(cfg.io_dtype, cfg.k_decay_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_restore_raw = cutlass.Array(cfg.io_dtype, cfg.k_restore_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sV_raw = cutlass.Array(mV.element_type, cfg.v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sGate_raw = cutlass.Array(cutlass.Float32, cfg.gate_cosize, space=SMEM, alignment=1024)
    sGate_exchange_raw = sGate_raw
    sK_decay = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sK_restore_trans = SmemTile(
        base=sK_restore_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=(cfg.b_t * 128),
        stride_byte_offset=(8 * 128),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )

    elect_one = nvvm.elect_sync()

    # ---- mbarrier init (one lane per owning role) ------------------------------------
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_raw_ready[stage].init()
                bars.mb_raw_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_h_ready.init()
            bars.mb_state_input_h_cg1_ready.init()
            bars.mb_state_input_h_cg0_ready.init()
            bars.mb_u_input_h_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_h_cg0_done[stage].init()
                bars.mb_state_acc_h_cg1_done[stage].init()
            bars.mb_state_k_acc_m_ready.init()
            bars.mb_state_input_m_cg1_ready.init()
            bars.mb_state_input_m_cg0_ready.init()
            bars.mb_u_input_m_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_m_cg0_done[stage].init()
                bars.mb_state_acc_m_cg1_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.scheduler_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_cg0_ready[stage].init()
    elif warp_idx == cfg.scheduler_twin_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.scheduler_stages):
                bars.mb_scheduler_ready[stage].init()
                bars.mb_scheduler_done[stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0, thread_count=cfg.threads_per_cta)

    # ---- warp specialization ---------------------------------------------------------
    if warp_idx == cfg.tma_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mScheduler,
            sScheduler,
            lane_idx,
            sV_raw,
            sGate_raw,
            desc_v_base,
            desc_k_decay_base,
            desc_t_base,
            desc_diag_base,
            sK_decay_raw,
            sK_restore_raw,
            bars,
            v_ratio=v_ratio,
        )
    elif warp_idx == cfg.scheduler_warp_id:
        scheduler_warp(
            cfg,
            total_tiles,
            bidx,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            bars,
        )
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        tcgen05_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            sTmem_base,
            sK_decay,
            sK_restore_trans,
            bars,
        )
    elif warp_idx == cfg.scheduler_twin_warp_id:
        scheduler_warp(
            cfg,
            total_tiles,
            bidx,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            bars,
        )
    elif warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        compute0_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            warp_idx,
            sGate_exchange_raw,
            sTmem_base,
            bars,
        )
    elif warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        compute1_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            sTmem_base,
            warp_idx,
            mState_out,
            mTransition,
            mState_init,
            mComposed,
            sV_raw,
            sGate_exchange_raw,
            bars,
        )


@dataclass
class KdaPrepSummaryCfg:
    """Kernel cfg: fixed BT=16 schedule constants plus the TMEM column offsets and SMEM cosizes stamped by ``build_cfg``;
    passed ``cfg``-first as a ``cutlass.Constexpr`` into ``host`` / ``kernel`` and every warp body."""

    io_dtype: Type[cutlass.Numeric]
    gate_dtype: Type[cutlass.Numeric]
    use_initial_state: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    log_gate: bool
    beta_sigmoid: bool
    allow_neg_eigval: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    state_table: bool = True
    has_composed: bool = False
    tiles_per_head: int = 1
    d_m: int = 0
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    scheduler_warp_id: int = CFG.REGISTER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    scheduler_twin_warp_id: int = CFG.REGISTER_MMA_TWIN_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0
    cg0_group_count: int = 2
    cg0_warps_per_group: int = 4
    cg0_threads_per_group: int = 0
    cg0_tile_entry_barrier_id: int = 5  # CG0-wide (both groups) work-item entry sync
    tmem_user_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -----------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES

    # ---- TMEM column offsets of the two chains ---------------------------------------
    tmem_state_acc_h_offset: int = 0
    tmem_state_input_h_offset: int = 0
    tmem_state_k_acc_h_offset: int = 0
    tmem_u_acc_h_offset: int = 0
    tmem_y_input_h_offset: int = 0
    tmem_u_input_h_offset: int = 0
    tmem_state_acc_m_offset: int = 0
    tmem_state_input_m_offset: int = 0
    tmem_state_k_acc_m_offset: int = 0
    tmem_u_acc_m_offset: int = 0
    tmem_y_input_m_offset: int = 0
    tmem_u_input_m_offset: int = 0

    # ---- SMEM buffer cosizes ---------------------------------------------------------
    v_cosize: int = 0
    gate_cosize: int = 0
    k_decay_cosize: int = 0
    k_restore_cosize: int = 0

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_gate_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    gate_dtype: Type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
    state_table: bool | None = None,
    has_composed: bool = False,
    tiles_per_head: int = 1,
) -> KdaPrepSummaryCfg:
    """Build the per-compile ``KdaPrepSummaryCfg`` (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    state_table = use_initial_state if state_table is None else bool(state_table)
    has_composed = bool(has_composed)
    cfg = KdaPrepSummaryCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        use_initial_state=use_initial_state,
        state_table=state_table,
        has_composed=has_composed,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
        tiles_per_head=tiles_per_head,
        d_m=d_v if tiles_per_head > 1 and d_k == tiles_per_head * d_v else d_k,
    )
    if cfg.d_k not in STATE_DIMS or cfg.d_v not in STATE_DIMS:
        raise ValueError(f"the prep-fed KDA summary serves DK, DV in {STATE_DIMS}, got DK={cfg.d_k} DV={cfg.d_v}")
    if cfg.smem_raw_stages % 2 != 0:
        raise ValueError("smem_raw_stages must be even: the CG0 ping-pong groups alias parity waits on odd rings")
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.cg0_threads_per_group = cfg.cg0_warps_per_group * cfg.threads_per_warp
    cfg.tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)) * cfg.threads_per_warp
    if cfg.cg0_warps_per_group != len(cfg.compute_group_1_warp_ids):
        raise ValueError("the state halves are packed by one CG0 group and by CG1: their warp counts must match")

    cfg.tmem_state_acc_h_offset = 0
    cfg.tmem_state_input_h_offset = cfg.tmem_state_acc_h_offset + cfg.d_k
    cfg.tmem_state_k_acc_h_offset = cfg.tmem_state_input_h_offset + cfg.d_k // 2
    cfg.tmem_u_acc_h_offset = cfg.tmem_state_k_acc_h_offset + cfg.b_t
    cfg.tmem_y_input_h_offset = cfg.tmem_u_acc_h_offset + cfg.b_t
    cfg.tmem_u_input_h_offset = cfg.tmem_y_input_h_offset + cfg.b_t // 2
    cfg.tmem_state_acc_m_offset = cfg.tmem_u_input_h_offset + cfg.b_t // 2
    cfg.tmem_state_input_m_offset = cfg.tmem_state_acc_m_offset + cfg.d_k
    cfg.tmem_state_k_acc_m_offset = cfg.tmem_state_input_m_offset + cfg.d_k // 2
    cfg.tmem_u_acc_m_offset = cfg.tmem_state_k_acc_m_offset + cfg.b_t
    cfg.tmem_y_input_m_offset = cfg.tmem_u_acc_m_offset + cfg.b_t
    cfg.tmem_u_input_m_offset = cfg.tmem_y_input_m_offset + cfg.b_t // 2
    if cfg.tmem_u_input_m_offset + cfg.b_t // 2 > 512:
        raise ValueError(f"TMEM layout exceeds 512 columns: {cfg.tmem_u_input_m_offset + cfg.b_t // 2}")

    cfg.v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k
    cfg.smem_decay_stages = 4
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.tma_k_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_v_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_gate_bytes = cfg.d_k * 4
    return cfg


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: V, k_decay, t, diag


# ---------------------------------------------------------------------------


@functools.cache
def get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    transition_dtype_str: str,
    gate_dtype_str: str,
    a_log_dtype_str: str,
    dt_bias_dtype_str: str,
    cu_dtype_str: str,
    beta_dtype_str: str,
    device: int,
    num_sm: int,
    DK: int,
    DV: int,
    use_initial_state: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    log_gate: bool,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    run_order: bool,
    has_composed: bool = False,
    state_table: bool = False,
    tiles_per_head: int = 1,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    gate_dtype,
    use_initial_state: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    *,
    d_k: int,
    d_v: int,
    num_sm: int,
    log_gate: bool = True,
    tiles_per_head: int = 1,
    k_cute,
    v_cute,
    gate_cute,
    a_log_cute,
    dt_bias_cute,
    beta_cute,
    cu_seqlens_cute,
    state_in_cute,
    composed_cute=None,
    state_table=None,
    has_composed=False,
    state_out_cute,
    transition_cute,
    work_items_cute,
    work_count_cute,
    scheduler_counter_cute,
    tensormap_workspace_cute,
    stream,
    scheduler_all_cute=None,
    prep_k_decay_cute=None,
    prep_t_cute=None,
    prep_diag_cute=None,
    prep_words_cute=None,
    prep_rows_cute=None,
    prep_row_count_cute=None,
):
    """JIT-compile the prep-fed KDA summary launch (prologue, prep, summary) for one static config."""
    cfg = build_cfg(
        io_dtype,
        gate_dtype,
        use_initial_state=use_initial_state,
        state_table=state_table,
        has_composed=has_composed,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=num_sm,
        d_k=d_k,
        d_v=d_v,
        tiles_per_head=tiles_per_head,
    )
    prep_cfg = kda_prep_f16.build_cfg(
        io_dtype,
        gate_dtype,
        num_sm=num_sm,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
        summary_records=True,
    )

    return cute.compile(
        prep_summary_host,
        cfg,
        prep_cfg,
        k_cute,
        v_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        beta_cute,
        cu_seqlens_cute,
        state_in_cute,
        composed_cute,
        state_out_cute,
        transition_cute,
        work_items_cute,
        work_count_cute,
        scheduler_counter_cute,
        scheduler_all_cute,
        tensormap_workspace_cute,
        prep_k_decay_cute,
        prep_t_cute,
        prep_diag_cute,
        prep_words_cute,
        prep_rows_cute,
        prep_row_count_cute,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_kda_summary(
    k,
    v,
    gate,
    beta,
    cu_seqlens,
    initial_state,
    output_state,
    output_transition,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    work_items=None,
    work_count=None,
    scheduler_counter=None,
    scheduler_all=None,
    order_in_prologue: bool = False,
    *,
    log_gate: bool = True,
    tensormap_workspace,
    device: int,
    num_sm: int,
    stream,
    own_prologue: bool = True,
    composed=None,
    tiles_per_head: int = 1,
    item_limit_word=None,
    prep_k_decay=None,
    prep_t=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
) -> None:
    """BT=16 prep-fed KDA state summary: one launch writes H (``output_state``, fp32; ``initial_state`` or zero on an empty
    sequence) and M (``output_transition``, fp32, stored domain ``M_buf = M^T`` with ``X_final = X_init @ M_buf + X_H``; identity on
    an empty sequence) for every work item.  Innermost strides 1; DK, DV in {64, 128}.  log_gate: ``gate`` is the natural-log decay,
    else alpha in (0, 1] floored at 1e-10; safe_gate overrides with ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``.
    use_beta_sigmoid: ``beta`` holds io-dtype logits, else fp32 post-sigmoid.  work_items / work_count: REQUIRED; an item seeds when
    ``compute_start == 0`` and stores when ``write_end == batch_num_chunks``.  scheduler_counter: ``[ticket, done]`` work-stealing
    scratch zeroed before every launch (REQUIRED; the ordering prologue zeroes ``scheduler_all``).  The launch orders its
    items in its own prologue (order_in_prologue and own_prologue REQUIRED).  prep_k_decay / prep_t / prep_diag: the
    kda_prep_f16 record buffers; prep_words, prep_rows, prep_row_count: its descriptor arrays and row table."""
    k.shape[1]
    HO = gate.shape[1]
    DK = k.shape[2]
    v.shape[1]
    DV = v.shape[2]
    if output_state is None or output_transition is None:
        raise ValueError("the fused summary writes both output_state (H) and output_transition (M)")
    if tuple(output_transition.shape) != (cu_seqlens.shape[0] - 1, HO, DK, DK):
        raise ValueError(f"output_transition must be (num_seqs, HO, DK, DK), got {tuple(output_transition.shape)}")
    if work_items is None or work_count is None or scheduler_counter is None:
        raise ValueError("work_items, work_count and scheduler_counter are required")
    state_table = initial_state is not None
    has_composed = composed is not None
    use_initial_state = state_table or has_composed
    run_order = order_in_prologue
    if run_order and scheduler_all is None:
        raise ValueError("order in the prologue requires scheduler_all (the prologue zeroes the scheduler rings)")
    if not run_order or not own_prologue:
        raise ValueError("the prep-fed summary orders its items in its own prologue")
    if any(buffer is None for buffer in (prep_k_decay, prep_t, prep_diag, prep_words, prep_rows, prep_row_count)):
        raise ValueError("the prep-fed summary requires the prep record buffers, descriptor arrays and row table")

    gate_scale_log2 = gate_lower_bound * LOG2_E

    if not safe_gate:
        a_log = None
        dt_bias = None
    cu_stream = cuda_driver.CUstream(int(stream))

    cache = get_compiled_cache(
        str(k.dtype),
        str(output_state.dtype),
        str(output_transition.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(beta.dtype),
        device,
        num_sm,
        DK,
        DV,
        use_initial_state,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        log_gate,
        use_beta_sigmoid,
        allow_neg_eigval,
        run_order,
        has_composed=has_composed,
        state_table=state_table,
        tiles_per_head=tiles_per_head,
    )

    if "compiled" not in cache:
        io_dtype = get_dtype(k.dtype)
        gate_dtype = get_dtype(gate.dtype)
        k_cute = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        v_cute = from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        gate_cute = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        a_log_cute = from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None
        beta_cute = from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic()

        state_in_cute = None
        if state_table:
            state_in_cute = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        composed_cute = None
        if has_composed:
            composed_cute = from_dlpack(composed, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        state_out_cute = from_dlpack(output_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        transition_cute = from_dlpack(output_transition, assumed_align=16).mark_layout_dynamic(leading_dim=3)

        work_items_cute = from_dlpack(work_items, assumed_align=16)
        work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()

        scheduler_counter_cute = from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic()
        scheduler_all_cute = from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic()

        tensormap_workspace_cute = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()
        prep_k_decay_cute = from_dlpack(prep_k_decay, assumed_align=128).mark_layout_dynamic(leading_dim=3)
        prep_t_cute = from_dlpack(prep_t, assumed_align=128).mark_layout_dynamic(leading_dim=3)
        prep_diag_cute = from_dlpack(prep_diag, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        prep_words_cute = from_dlpack(prep_words, assumed_align=128).mark_layout_dynamic()
        prep_rows_cute = from_dlpack(prep_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        prep_row_count_cute = from_dlpack(prep_row_count, assumed_align=4).mark_layout_dynamic()

        cache["compiled"] = compile(
            io_dtype,
            gate_dtype,
            use_initial_state,
            use_qk_l2norm_in_kernel,
            safe_gate,
            gate_scale_log2,
            use_beta_sigmoid,
            allow_neg_eigval,
            d_k=DK,
            d_v=DV // tiles_per_head,
            log_gate=log_gate,
            num_sm=num_sm,
            tiles_per_head=tiles_per_head,
            k_cute=k_cute,
            v_cute=v_cute,
            gate_cute=gate_cute,
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            beta_cute=beta_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            state_in_cute=state_in_cute,
            composed_cute=composed_cute,
            state_table=state_table,
            has_composed=has_composed,
            state_out_cute=state_out_cute,
            transition_cute=transition_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            scheduler_counter_cute=scheduler_counter_cute,
            tensormap_workspace_cute=tensormap_workspace_cute,
            stream=cu_stream,
            scheduler_all_cute=scheduler_all_cute,
            prep_k_decay_cute=prep_k_decay_cute,
            prep_t_cute=prep_t_cute,
            prep_diag_cute=prep_diag_cute,
            prep_words_cute=prep_words_cute,
            prep_rows_cute=prep_rows_cute,
            prep_row_count_cute=prep_row_count_cute,
        )

    compiled = cache["compiled"]
    compiled(
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state if state_table else None,
        composed if has_composed else None,
        output_state,
        output_transition,
        work_items,
        work_count,
        scheduler_counter,
        scheduler_all,
        tensormap_workspace,
        prep_k_decay,
        prep_t,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        cu_stream,
    )
    return cache


def run_summary(
    cache,
    k,
    v,
    gate,
    a_log,
    dt_bias,
    beta,
    cu_seqlens,
    initial_state,
    output_state,
    output_transition,
    work_items,
    work_count,
    scheduler_counter,
    scheduler_all,
    tensormap_workspace,
    stream,
    own_prologue=True,
    composed=None,
    item_limit_word=None,
    prep_k_decay=None,
    prep_t=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
) -> None:
    """Replay the compiled plan: one crossing into the DSL for the prologue, prep and summary launches.  The plan validated
    the contract at build, so nothing here raises."""
    cu_stream = cuda_driver.CUstream(int(stream))
    cache["compiled"](
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        composed,
        output_state,
        output_transition,
        work_items,
        work_count,
        scheduler_counter,
        scheduler_all,
        tensormap_workspace,
        prep_k_decay,
        prep_t,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        cu_stream,
    )


frost_kda_prep_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_kda_prep_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
