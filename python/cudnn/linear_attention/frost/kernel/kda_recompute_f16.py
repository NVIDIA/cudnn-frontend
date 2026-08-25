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

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_work_item, order_body
from ..common.host import get_dtype
from cudnn.frost.buffers import data_ptr
from cudnn.frost.device import current_device, multiprocessor_count
from ..common.thd import TENSOR_MAP_QWORDS, emit_checkpoint_seq_descs, emit_seq_descs
from .kda_recompute_config import CFG

from cudnn.frost.tile_dsl.barrier import (
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_step, mma_ts_step
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b, swizzle_xor_32b
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.pointwise import (
    sigmoid,
    opaque_f32_zero,
    f16x2_to_f32,
    fadd2,
    fmul2,
    ffma2,
    movmatrix_16b,
    mul_f16x2,
    fp32_to_fp16,
    sub_f16x2,
)

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class KdaBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_state_k_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier

    mb_state_input_ready: MBarrier
    mb_y_input_ready: MBarrier
    mb_u_input_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_qk_scale_ready: MBarrier
    mb_state_scale_diag_done: MBarrier
    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_decay_super_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_state_acc_done: MBarrier
    mb_state_acc_read_done: MBarrier
    mb_tmem_done: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_kda_bars(cfg) -> KdaBars:
    """KdaBars factory."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0_GROUP_THREADS = cfg.cg0_warps_per_group * WARP
    CG1_THREADS = len(cfg.compute_group_1_warp_ids) * WARP

    return KdaBars(
        mb_k_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_v_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=WARP + CG1_THREADS, producer=Producer.THREAD),
        mb_state_k_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_y_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_u_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_t_inv_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_qk_scale_ready=MBarrier(
            alloc(cfg.qk_scale_ready_stages),
            stages=cfg.qk_scale_ready_stages,
            init_count=CG0_GROUP_THREADS,
            producer=Producer.THREAD,
        ),
        mb_state_scale_diag_done=MBarrier(
            alloc(cfg.smem_state_scale_diag_stages),
            stages=cfg.smem_state_scale_diag_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_k_decay_inv_cg0_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_decay_tcgen05_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_decay_super_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=WARP, producer=Producer.THREAD),
        mb_k_restore_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_done=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_read_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_checkpoint_tmastg_ready=MBarrier(
            alloc(cfg.smem_checkpoint_stages), stages=cfg.smem_checkpoint_stages, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_checkpoint_tmastg_done=MBarrier(alloc(cfg.smem_checkpoint_stages), stages=cfg.smem_checkpoint_stages, init_count=WARP, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
    )


# ---- Dynamic tile scheduler ----------------------------------------------------------


@cute.jit
def scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, elect_one):
    """TMA-warp side: pull the next tile off the global ticket, publish it."""
    if cutlass.const_expr(cfg.dynamic_scheduling):
        bars.mb_scheduler_done[scheduler_state.idx].wait(scheduler_state.phase)
        if elect_one:
            fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
            sScheduler[scheduler_state.idx] = num_ctas + fetched
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        next_tile = sScheduler[scheduler_state.idx]
        if elect_one:
            bars.mb_scheduler_ready[scheduler_state.idx].arrive()
        return next_tile, advance(scheduler_state, cfg.scheduler_stages)
    return tile_idx + num_ctas, scheduler_state


@cute.jit
def scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one):
    """Consumer side: read the TMA warp's published next tile."""
    if cutlass.const_expr(cfg.dynamic_scheduling):
        bars.mb_scheduler_ready[scheduler_state.idx].wait(scheduler_state.phase)
        next_tile = sScheduler[scheduler_state.idx]
        if elect_one:
            bars.mb_scheduler_done[scheduler_state.idx].arrive()
        return next_tile, advance(scheduler_state, cfg.scheduler_stages)
    return tile_idx + num_ctas, scheduler_state


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
    sK_raw,
    sV_raw,
    sGate_raw,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing the
    per-chunk K/V/Gate G->S loads."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 32),
        tma_granu_elems=32,
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o = head_idx
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
        for chunk_idx in cutlass.range(compute_start, write_end, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t

            # ---- K load --------------------------------------------------------------
            bars.mb_k_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_k_ready[raw_index.idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, bars.mb_k_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Gate load: GMEM -> SMEM ---------------------------------------------
            bars.mb_gate_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_gate_ready[raw_index.idx].arrive(n_bytes=cfg.tma_gate_bytes)
            gate_slice = tma_slice_runtime_desc(desc_gate_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, bars.mb_gate_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- V load --------------------------------------------------------------
            bars.mb_v_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_v_ready[raw_index.idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, bars.mb_v_ready[raw_index.idx].smem_ptr, acquire=False)

            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


@cute.jit
def super_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    sK_inv_raw,
    sIntermediate_raw,
    sBeta_raw,
    sK_decay_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): persistent scheduler loop computing the
    register-MMA Neumann-series T_inv."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()
    raw_index = PipelineState.start(phase=0)
    t_inv_free = PipelineState.start(phase=1)
    k_decay_ready = PipelineState.start(phase=0)

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    k_inv_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    k_inv_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    k_decay_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    k_decay_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    t_inv_row_coord = lane_idx & 7
    t_inv_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        t_inv_row_coord = t_inv_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        t_inv_col_coord = cutlass.Int32(8)
    t_inv_idx = t_inv_row_coord * cfg.b_t + swizzle_xor_32b(t_inv_row_coord, t_inv_col_coord)
    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            decay_stage = k_decay_ready.idx
            intermediate_stage = t_inv_free.idx
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sIntermediate_ptr = sIntermediate_raw.data_ptr() + intermediate_stage * (2 * cfg.b_t * cfg.b_t)

            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)

            # ---- KK = K decay @ K inv^T ----------------------------------------------
            kk_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)

            for i in cutlass.range_constexpr((cfg.d_k // 16)):
                k_inv_col = i * 16 + k_inv_col_offset
                k_inv_segment = k_inv_col // 64
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr
                    + k_inv_segment * (cfg.b_t * 64)
                    + k_inv_row_coord * 64
                    + swizzle_xor_128b(k_inv_row_coord, k_inv_col - k_inv_segment * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                k_decay_col = i * 16 + k_decay_col_offset
                k_decay_segment = k_decay_col // 64
                k_decay_frag = nvvm.ldmatrix(
                    sK_decay_ptr
                    + k_decay_segment * (cfg.b_t * 64)
                    + swizzle_xor_128b(k_decay_row_coord, k_decay_row_coord * 64 + k_decay_col - k_decay_segment * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )

                mma_step(
                    kk_acc,
                    (k_decay_frag[0], k_decay_frag[1], k_decay_frag[2], k_decay_frag[3]),
                    (k_inv_frag[0], k_inv_frag[1], k_inv_frag[2], k_inv_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- L = Beta * tril(KK, -1) ---------------------------------------------
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)
            row_lo = lane_idx // 4
            row_hi = row_lo + cutlass.Int32(8)
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            l_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
                col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                l_regs[accum_idx] = kk_acc[accum_idx] if row_coord > col_coord else cutlass.Float32(0.0)
            for pair in cutlass.range_constexpr(4):
                beta_scale = beta_hi if cutlass.const_expr(pair % 2 == 1) else beta_lo
                l_regs[2 * pair], l_regs[2 * pair + 1] = fmul2(l_regs[2 * pair], l_regs[2 * pair + 1], beta_scale, beta_scale)
            bars.mb_beta_done[raw_index.idx].arrive()
            l_a0 = fp32_to_fp16(l_regs[0], l_regs[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_regs[4], l_regs[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_regs[6], l_regs[7], dtype=cfg.io_dtype)
            l_values = cutlass.Vector.from_elements((l_a0, l_a1, l_a2, l_a3), cutlass.Int32).bitcast(cfg.io_dtype).to(cutlass.Float32)

            # ---- T^-1 = I - L, then three Neumann doubling rounds --------------------
            tinv_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_lo
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_hi
                col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                eye = cutlass.Float32(1.0) if row_coord == col_coord else cutlass.Float32(0.0)
                tinv_acc[accum_idx] = eye - l_values[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(l_a0), movmatrix_16b(l_a1), movmatrix_16b(l_a2), movmatrix_16b(l_a3)
            for neumann_round in cutlass.range_constexpr(3):
                # ---- Lpow = Lpow @ Lpow ----------------------------------------------
                sq_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
                for accum_idx in cutlass.range_constexpr(8):
                    sq_acc[accum_idx] = cutlass.Float32(0.0)
                mma_step(
                    sq_acc,
                    (lpow_a0, lpow_a1, lpow_a2, lpow_a3),
                    (mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                lpow_a0 = fp32_to_fp16(sq_acc[0], sq_acc[1], dtype=cfg.io_dtype)
                lpow_a1 = fp32_to_fp16(sq_acc[2], sq_acc[3], dtype=cfg.io_dtype)
                lpow_a2 = fp32_to_fp16(sq_acc[4], sq_acc[5], dtype=cfg.io_dtype)
                lpow_a3 = fp32_to_fp16(sq_acc[6], sq_acc[7], dtype=cfg.io_dtype)
                mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(lpow_a0), movmatrix_16b(lpow_a1), movmatrix_16b(lpow_a2), movmatrix_16b(lpow_a3)
                # ---- T^-1 += T^-1 @ Lpow ---------------------------------------------
                upd_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
                for accum_idx in cutlass.range_constexpr(8):
                    upd_acc[accum_idx] = cutlass.Float32(0.0)
                tinv_p0 = fp32_to_fp16(tinv_acc[0], tinv_acc[1], dtype=cfg.io_dtype)
                tinv_p1 = fp32_to_fp16(tinv_acc[2], tinv_acc[3], dtype=cfg.io_dtype)
                tinv_p2 = fp32_to_fp16(tinv_acc[4], tinv_acc[5], dtype=cfg.io_dtype)
                tinv_p3 = fp32_to_fp16(tinv_acc[6], tinv_acc[7], dtype=cfg.io_dtype)
                mma_step(
                    upd_acc,
                    (tinv_p0, tinv_p1, tinv_p2, tinv_p3),
                    (mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                tinv_lo0, tinv_hi0 = f16x2_to_f32(tinv_p0, dtype=cfg.io_dtype)
                tinv_lo1, tinv_hi1 = f16x2_to_f32(tinv_p1, dtype=cfg.io_dtype)
                tinv_lo2, tinv_hi2 = f16x2_to_f32(tinv_p2, dtype=cfg.io_dtype)
                tinv_lo3, tinv_hi3 = f16x2_to_f32(tinv_p3, dtype=cfg.io_dtype)
                tinv_acc[0], tinv_acc[1] = fadd2(tinv_lo0, tinv_hi0, upd_acc[0], upd_acc[1])
                tinv_acc[2], tinv_acc[3] = fadd2(tinv_lo1, tinv_hi1, upd_acc[2], upd_acc[3])
                tinv_acc[4], tinv_acc[5] = fadd2(tinv_lo2, tinv_hi2, upd_acc[4], upd_acc[5])
                tinv_acc[6], tinv_acc[7] = fadd2(tinv_lo3, tinv_hi3, upd_acc[6], upd_acc[7])

            bars.mb_t_inv_done[intermediate_stage].wait(t_inv_free.phase)
            t_inv_free = advance(t_inv_free, cfg.smem_intermediate_stages)
            nvvm.stmatrix(
                sIntermediate_ptr + (cfg.b_t * cfg.b_t) + t_inv_idx,
                [
                    fp32_to_fp16(tinv_acc[0], tinv_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[2], tinv_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[4], tinv_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[6], tinv_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_t_inv_ready[intermediate_stage].arrive()
            bars.mb_decay_super_done[decay_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


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
    sIntermediate,
    sK_decay,
    sK_restore_trans,
    sState_scale_diag,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): persistent scheduler loop issuing
    every state GEMM and owning the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(sTmem_base, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    state_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_offset, cutlass.Int8)
    state_dsts = tuple(nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset + k * 16, cutlass.Float32) for k in range(cfg.d_k // 16))
    state_k_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_offset, cutlass.Float32)
    u_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_acc_offset, cutlass.Float32)
    y_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_y_input_offset, cutlass.Int8)
    u_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_offset, cutlass.Int8)
    state_dst_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset, cutlass.Float32)
    state_input_index = PipelineState.start(phase=0)
    state_read_index = PipelineState.start(phase=0)
    y_input_index = PipelineState.start(phase=0)
    u_input_index = PipelineState.start(phase=0)
    qk_scale_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)
    t_inv_ready = PipelineState.start(phase=0)

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_acc = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=0,
    )
    idesc_diag = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=16,
        m_dim=cfg.d_v,
        b_major=0,
    )
    idesc_final_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_state_k_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_diag_desc = MmaDesc(
        M=cfg.d_v,
        N=16,
        K=16,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_diag,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_y_t_inv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=idesc_final_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    STATE_A_SEG = bmm_state_k_decay_desc.sps_B * bmm_state_k_decay_desc.tmem_advance_A
    STATE_B_SEG = bmm_state_k_decay_desc.smem_subtile_B >> 4
    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.use_initial_state):
            seed_state = compute_start == 0
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            if cutlass.const_expr(cfg.use_initial_state):
                have_state = local_chunk_idx > 0 or seed_state
            else:
                have_state = local_chunk_idx > 0
            decay_stage = k_decay_ready.idx
            state_scale_diag_stage = qk_scale_index.idx
            intermediate_stage = t_inv_ready.idx
            sK_decay_stage = sK_decay[decay_stage]
            sK_restore_stage = sK_restore_trans[decay_stage]
            sState_scale_diag_stage = sState_scale_diag[state_scale_diag_stage]
            sIntermediate_stage = sIntermediate[intermediate_stage]

            # ---- k state = state(T) @ K decay^T --------------------------------------
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state:
                bars.mb_state_input_ready.wait(state_input_index.phase)
                state_input_index = advance(state_input_index, 1)
                desc_k_decay = sK_decay_stage.desc()

                for s in cutlass.range_constexpr(bmm_state_k_decay_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_k_decay_desc.sps_B):
                        mma_ts_step(
                            bmm_state_k_decay_desc,
                            state_input_ptr.subview(s * STATE_A_SEG),
                            desc_k_decay + s * STATE_B_SEG,
                            state_k_acc_ptr,
                            k,
                            cutlass.Boolean(s + k > 0),
                        )

                if elect_one:
                    bars.mb_state_k_acc_ready.arrive(cta_group=1)

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)
            if cutlass.const_expr(cfg.enable_checkpoints):
                if have_state:
                    bars.mb_state_acc_read_done.wait(state_read_index.phase)
                    state_read_index = advance(state_read_index, 1)

            # ---- state decay = state(T) @ diag(exp2(g last)) (per-k-atom blocks) -----
            if have_state:
                desc_diag = sState_scale_diag_stage.desc()
                for i in cutlass.range_constexpr(cfg.d_k // 16):
                    mma_ts_step(
                        bmm_state_diag_desc,
                        state_input_ptr.subview(i * bmm_state_diag_desc.tmem_advance_A),
                        desc_diag.advance_start_address(i * 256 * 2),
                        state_dsts[i],
                        0,
                        cutlass.Boolean(False),
                    )

            if elect_one:
                bars.mb_state_scale_diag_done[state_scale_diag_stage].arrive(cta_group=1)

            # ---- U = Y(T) @ T^-1 -----------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(t_inv_ready.phase)
            bars.mb_y_input_ready.wait(y_input_index.phase)
            y_input_index = advance(y_input_index, 1)
            desc_t_inv = sIntermediate_stage.shifted((cfg.b_t * cfg.b_t)).desc()
            mma_ts_step(bmm_y_t_inv_desc, y_input_ptr, desc_t_inv, u_acc_ptr, 0, cutlass.Boolean(False))
            if elect_one:
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)
                bars.mb_u_acc_ready.arrive(cta_group=1)

            # ---- final state += U(T) @ K restore -------------------------------------
            bars.mb_u_input_ready.wait(u_input_index.phase)
            u_input_index = advance(u_input_index, 1)
            desc_k_restore = sK_restore_stage.desc()

            mma_ts_step(bmm_u_k_restore_desc, u_input_ptr, desc_k_restore, state_dst_ptr, 0, have_state)
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_done.arrive(cta_group=1)

            t_inv_ready = advance(t_inv_ready, cfg.smem_intermediate_stages)
            qk_scale_index = advance(qk_scale_index, cfg.smem_state_scale_diag_stages)

        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    sCheckpoint_raw,
    desc_checkpoint_base,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """Epilogue warp role (warp 15): persistent scheduler loop issuing the
    per-chunk checkpoint TMA stores."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_tma = SmemTile(
            base=sCheckpoint_raw,
            elems_per_stage=(cfg.d_k * cfg.d_v),
            stages=cfg.smem_checkpoint_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=(cfg.d_v // 64),
            tma_granu_elems=64,
            tma_subtile_stride_elems=cfg.d_k * 64,
        )
    checkpoint_ready_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        if cutlass.const_expr(cfg.enable_checkpoints):
            head_o = head_idx
            checkpoint_slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
            desc_checkpoint_slot = (desc_checkpoint_base + checkpoint_slot).tospace(cutlass.AddressSpace.generic)
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_quot = (compute_start + cutlass.Int32(1)) // checkpoint_chunks
            checkpoint_mod = (compute_start + cutlass.Int32(1)) % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)
            num_chunks_tile = write_end - compute_start
            if num_chunks_tile > 0 and write_start == 0:
                checkpoint_stage = checkpoint_ready_index.idx
                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(checkpoint_ready_index.phase)
                checkpoint_ready_index = advance(checkpoint_ready_index, cfg.smem_checkpoint_stages)
                checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), head_o)
                tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice, acquire=False)
                tma_store_commit()
                tma_store_wait(0)
                bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
            for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
                chunk_idx = compute_start + local_chunk_idx
                if local_chunk_idx > 0:
                    # ---- checkpoint store --------------------------------------------
                    do_checkpoint = checkpoint_mod == 0
                    do_checkpoint = do_checkpoint and chunk_idx >= write_start
                    if do_checkpoint:
                        checkpoint_stage = checkpoint_ready_index.idx
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(checkpoint_ready_index.phase)
                        checkpoint_ready_index = advance(checkpoint_ready_index, cfg.smem_checkpoint_stages)
                        checkpoint_entry = checkpoint_quot
                        checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), checkpoint_entry, head_o)
                        tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice, acquire=False)
                        tma_store_commit()
                        tma_store_wait(0)
                        bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    if checkpoint_mod == checkpoint_chunks:
                        checkpoint_mod = cutlass.Int32(0)
                        checkpoint_quot = checkpoint_quot + cutlass.Int32(1)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    # Default ABI: Gate arrives in natural-log space
    return raw_gate * cutlass.Float32(LOG2_E)


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
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_raw,
    mBeta,
    sBeta_raw,
    sK_raw,
    sK_decay_raw,
    sK_restore_raw,
    sState_scale_diag_raw,
    bars,
) -> None:
    """CG0 warp-group role (warps 0-7): persistent scheduler loop running the
    Gate prefix scan and staging the decay/restore operands."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    scheduler_state = PipelineState.start(phase=0)

    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    cg0_local_warp = cg0_warp % cfg.cg0_warps_per_group

    cg0_group_id = cg0_warp // cfg.cg0_warps_per_group
    channel_dim = cg0_local_warp * cfg.threads_per_warp + lane_idx
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    opaque_one = opaque_f32_zero() + cutlass.Float32(1.0)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o = head_idx
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.safe_gate):
            if num_chunks_tile > 0:
                cg0_a_log_exp = cute.math.exp2(mA_log[head_o].to(cutlass.Float32) * LOG2_E, fastmath=True)
                cg0_dt_bias_value = mDt_bias[head_o, channel_dim].to(cutlass.Float32)
        nvvm.barrier_cta_sync(cfg.cg0_tile_entry_barrier_id, thread_count=cfg.cg0_group_count * cfg.cg0_threads_per_group)
        group_cum_chunk_start = cum_chunk_base + cutlass.Int32(cg0_group_id)
        diag_ring_stage = group_cum_chunk_start % cutlass.Int32(cfg.smem_state_scale_diag_stages)
        diag_ring_phase = (group_cum_chunk_start // cutlass.Int32(cfg.smem_state_scale_diag_stages)) % cutlass.Int32(2)
        for local_chunk_idx in cutlass.range(cg0_group_id, num_chunks_tile, cfg.cg0_group_count, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = cum_chunk % cfg.smem_decay_stages
            raw_stage = cum_chunk % cfg.smem_raw_stages
            state_scale_diag_stage = diag_ring_stage
            qk_scale_ready_stage = state_scale_diag_stage
            sK_ptr = sK_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = sGate_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = sK_restore_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sState_scale_diag_ptr = sState_scale_diag_raw.data_ptr() + state_scale_diag_stage * ((cfg.d_k // 16) * 256)

            # ---- Beta scalars --------------------------------------------------------
            if cg0_local_warp == 0:
                bars.mb_beta_done[raw_stage].wait(((cum_chunk // cfg.smem_raw_stages) + 1) % 2)
                if lane_idx < cfg.b_t:
                    token_idx = chunk_idx * cfg.b_t + lane_idx
                    beta_value = cutlass.Float32(0.0)
                    if token_idx < batch_seqlen:
                        beta_value = mBeta[batch_start + token_idx, head_o].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.beta_sigmoid):
                            beta_value = sigmoid(beta_value).to(mBeta.element_type).to(cutlass.Float32)
                    sBeta_raw[raw_stage * cfg.b_t + lane_idx] = beta_value
                bars.mb_beta_ready[raw_stage].arrive()
            bars.mb_gate_ready[raw_stage].wait((cum_chunk // cfg.smem_raw_stages) % 2)

            row_group_start = cg0_local_warp * (cfg.b_t // cfg.cg0_warps_per_group)
            lane_row_group = lane_idx // 8
            lane_in_row_group = lane_idx - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            channel_dim = cg0_local_warp * cfg.threads_per_warp + lane_idx

            # ---- Gate prefix scan ----------------------------------------------------
            f32_segment = channel_dim // 32
            prefix_seg_base = f32_segment * (cfg.b_t * 32)
            prefix_col = channel_dim - f32_segment * 32
            gate_raw = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            for row in cutlass.range_constexpr(cfg.b_t):
                prefix_idx = prefix_seg_base + swizzle_xor_128b(row, row * 32 + prefix_col, elem_bytes=4)
                gate_raw[row] = (sGate_ptr + prefix_idx).load()
            g_prefix_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            if cutlass.const_expr(cfg.safe_gate):
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate = gate_raw[row]
                    token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cutlass.Int32(row)
                    if token_idx < batch_seqlen:
                        gate = gate_scale(
                            cfg,
                            cg0_a_log_exp * (gate + cg0_dt_bias_value),
                        )
                    else:
                        gate = cutlass.Float32(0.0)
                    g_prefix_regs[row] = gate
            else:
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate = gate_raw[row]
                    token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cutlass.Int32(row)
                    if token_idx < batch_seqlen:
                        gate = gate_scale(
                            cfg,
                            gate,
                        )
                    else:
                        gate = cutlass.Float32(0.0)
                    g_prefix_regs[row] = gate

            prefix_acc = cutlass.Float32(0.0)
            for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                row0 = row_pair * 2
                row1 = row0 + 1
                gate0 = g_prefix_regs[row0]
                gate1 = g_prefix_regs[row1]
                prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
                prefix1 = prefix_acc + row_pair_sum
                g_prefix_regs[row0] = prefix0
                g_prefix_regs[row1] = prefix1
                prefix_acc = prefix1

            # ---- exp2(g): stage prefixes + final-token decay -------------------------
            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            exp_g_last = g_prefix_regs[cfg.b_t - 1]
            for row in cutlass.range_constexpr(cfg.b_t):
                prefix_idx = prefix_seg_base + swizzle_xor_128b(row, row * 32 + prefix_col, elem_bytes=4)
                (sGate_ptr + prefix_idx).store(g_prefix_regs[row])

            # ---- state-scale diag: stage exp2(g last) decay blocks -------------------
            bars.mb_state_scale_diag_done[state_scale_diag_stage].wait(diag_ring_phase ^ cutlass.Int32(1))
            block = channel_dim // cutlass.Int32(16)
            coord = channel_dim - block * cutlass.Int32(16)
            diag_idx = block * cutlass.Int32(256) + coord * cutlass.Int32(16) + swizzle_xor_32b(channel_dim, coord)
            sState_scale_diag_ptr[diag_idx] = exp_g_last.to(cfg.io_dtype)

            nvvm.barrier_cta_sync(cfg.cg0_group_sync_barrier_base_id + cg0_group_id, thread_count=cfg.cg0_threads_per_group)

            bars.mb_k_ready[raw_stage].wait((cum_chunk // cfg.smem_raw_stages) % 2)
            k_inv_pack = cutlass.Array(cutlass.Int32, 2 * 4, alignment=16)
            raw_k_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)

            # ---- optional K L2-norm + K inv stage ------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                kk_lo = opaque_f32_zero()
                kk_hi = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_vec_f32 = raw_k_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    k_val = raw_k_vec_f32[dim_offset]
                    raw_k_regs[reg_base + dim_offset] = k_val
                if cutlass.const_expr(cfg.l2norm):
                    for dim_pair in cutlass.range_constexpr(4):
                        k_even = raw_k_vec_f32[2 * dim_pair]
                        k_odd = raw_k_vec_f32[2 * dim_pair + 1]
                        kk_lo, kk_hi = ffma2(k_even, k_odd, k_even, k_odd, kk_lo, kk_hi)

            k_inv_norm = opaque_one
            if cutlass.const_expr(cfg.l2norm):
                k_sum_sq = kk_lo + kk_hi
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)

            # ---- decay/restore operands: exp2(+-g) applied per key channel -----------
            exp_g_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            exp_g_last_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    g_prefix_idx = f32_segment * (cfg.b_t * 32) + decay_row * 32 + swizzle_xor_128b(decay_row, f32_segment_dim, elem_bytes=4)
                    exp_g_frag = (sGate_ptr + g_prefix_idx).load(count=4, alignment=16)
                    exp_g_last_idx = f32_segment * (cfg.b_t * 32) + (cfg.b_t - 1) * 32 + swizzle_xor_128b((cfg.b_t - 1), f32_segment_dim, elem_bytes=4)
                    exp_g_last_frag = (sGate_ptr + exp_g_last_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    for j in cutlass.range_constexpr(4):
                        exp_g_regs[f32_reg_base + j] = exp_g_frag[j]
                        exp_g_last_regs[f32_reg_base + j] = exp_g_last_frag[j]

            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8

                # ---- K decay + K inv operands: K * exp2(+g) and K * exp2(-g) ---------
                k_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
                    k_pair = fp32_to_fp16(k_value0, k_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_decay_pack[pair_idx] = mul_f16x2(k_pair, exp_g_pair, cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    exp_neg_pair = fp32_to_fp16(exp_neg_g0, exp_neg_g1, dtype=cfg.io_dtype)
                    k_inv_pack[dim_half * 4 + pair_idx] = mul_f16x2(k_pair, exp_neg_pair, cfg.io_dtype)

                k_inv_vec = cutlass.Vector.from_elements(
                    (
                        k_inv_pack[dim_half * 4],
                        k_inv_pack[dim_half * 4 + 1],
                        k_inv_pack[dim_half * 4 + 2],
                        k_inv_pack[dim_half * 4 + 3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                k_decay_vec = cutlass.Vector.from_elements(
                    (
                        k_decay_pack[0],
                        k_decay_pack[1],
                        k_decay_pack[2],
                        k_decay_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                if cutlass.const_expr(dim_half == 0):
                    operand_done_phase = ((cum_chunk // cfg.smem_decay_stages) + 1) % 2
                    bars.mb_decay_super_done[decay_stage].wait(operand_done_phase)
                    bars.mb_decay_tcgen05_done[decay_stage].wait(operand_done_phase)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_inv_swizzled_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sK_inv_ptr + k_inv_swizzled_idx).store(k_inv_vec, alignment=16)
                decay_col = dim_base
                decay_segment = decay_col // 64
                decay_swizzled_idx = decay_segment * (cfg.b_t * 64) + swizzle_xor_128b(decay_row, decay_row * 64 + decay_col - decay_segment * 64, elem_bytes=2)
                (sK_decay_ptr + decay_swizzled_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_inv_cg0_ready[decay_stage].arrive()

            # ---- K restore operand: K inv * exp2(g last) -----------------------------
            bars.mb_k_restore_done[decay_stage].wait(((cum_chunk // cfg.smem_decay_stages + 1) % 2))
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                k_restore_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    exp_g_last_pair = fp32_to_fp16(exp_g_last_regs[reg_base + dim0], exp_g_last_regs[reg_base + dim1], dtype=cfg.io_dtype)
                    k_restore_pack[pair_idx] = mul_f16x2(k_inv_pack[dim_half * 4 + pair_idx], exp_g_last_pair, cfg.io_dtype)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_restore_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                k_restore_vec = cutlass.Vector.from_elements(
                    (
                        k_restore_pack[0],
                        k_restore_pack[1],
                        k_restore_pack[2],
                        k_restore_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                (sK_restore_ptr + k_restore_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_scale_ready[qk_scale_ready_stage].arrive()
            bars.mb_k_done[raw_stage].arrive()
            bars.mb_gate_done[raw_stage].arrive()
            diag_ring_stage = diag_ring_stage + cutlass.Int32(cfg.cg0_group_count)
            wrapped = diag_ring_stage >= cutlass.Int32(cfg.smem_state_scale_diag_stages)
            diag_ring_stage = diag_ring_stage - cutlass.Int32(cfg.smem_state_scale_diag_stages) if wrapped else diag_ring_stage
            diag_ring_phase = diag_ring_phase ^ (cutlass.Int32(1) if wrapped else cutlass.Int32(0))
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)


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
    mState_init,
    sBeta_raw,
    sV_raw,
    sCheckpoint_raw,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """CG1 warp-group role (warps 8-11): persistent scheduler loop staging the
    value-side TMEM operands and storing the checkpoint/final states."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    checkpoint_done_index = PipelineState.start(phase=1)

    sCheckpoint_ptr = sCheckpoint_raw.data_ptr() if cutlass.const_expr(cfg.enable_checkpoints) else sV_raw.data_ptr()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    # ldmatrix.x4 COL lane decode for the V loads
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    value_dim = tmem_subpartition * cfg.threads_per_warp + lane_idx
    value_dim_base = tmem_subpartition * cfg.threads_per_warp
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_input_offset
    statek_col_id = tmem_col + cfg.tmem_state_k_acc_offset
    y_input_col_id = tmem_col + cfg.tmem_y_input_offset
    u_acc_addr = row_lo_addr + tmem_col + cfg.tmem_u_acc_offset
    u_input_addr = row_lo_addr + tmem_col + cfg.tmem_u_input_offset
    v_swz_off_lo = (
        (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2)
    )
    v_swz_off_hi = (
        (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2)
    )
    checkpoint_swz_off_lo = (value_dim_base + ov_col_offset) // 64 * (cfg.d_k * 64)
    checkpoint_swz_col_lo = (value_dim_base + ov_col_offset) % 64
    checkpoint_swz_off_hi = (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.d_k * 64)
    checkpoint_swz_col_hi = (value_dim_base + 16 + ov_col_offset) % 64
    state_k_acc_index = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    state_upd_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o = head_idx
        num_chunks_tile = write_end - compute_start

        if num_chunks_tile > 0:
            # ---- first chunk: seed state TMEM from mState init -----------------------
            seed_from_initial_state = compute_start == 0
            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t

            # ---- state seed: initial state GMEM -> state TMEM ------------------------
            if cutlass.const_expr(mState_init is not None):
                if seed_from_initial_state:
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (mState_init.iterator + mState_init.layout((batch_idx, head_o, value_dim, 0))).raw_ptr()
                    state_vecs = []
                    for i in cutlass.range_constexpr(cfg.d_k // 16):
                        state_block = []
                        for g in cutlass.range_constexpr(16 // seed_vw):
                            seed_chunk = (seed_src + i * 16 + g * seed_vw).load(count=seed_vw, alignment=16)
                            for t in cutlass.range_constexpr(seed_vw):
                                state_block.append(seed_chunk[t].to(cutlass.Float32))
                        state_vecs.append(state_block)

                    for i in cutlass.range_constexpr(cfg.d_k // 16):
                        state_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                        for packed_col in cutlass.range_constexpr(8):
                            state_pack[packed_col] = fp32_to_fp16(state_vecs[i][2 * packed_col], state_vecs[i][2 * packed_col + 1], dtype=cfg.io_dtype)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                            state_pack[0:8],
                        )

                    nvvm.tcgen05_wait("store")
                    bars.mb_state_input_ready.arrive()
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        if write_start == 0:
                            checkpoint_stage = checkpoint_done_index.idx
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_done_index.phase)
                            checkpoint_done_index = advance(checkpoint_done_index, cfg.smem_checkpoint_stages)
                            checkpoint_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                            for i in cutlass.range_constexpr(cfg.d_k // 16):
                                for g in cutlass.range_constexpr(2):
                                    packs = tuple(
                                        fp32_to_fp16(state_vecs[i][g * 8 + 2 * t], state_vecs[i][g * 8 + 2 * t + 1], dtype=cfg.io_dtype) for t in range(4)
                                    )
                                    dk = i * 16 + g * 8
                                    checkpoint_addr = (
                                        checkpoint_stage_base
                                        + (dk // 64) * (cfg.d_v * 64)
                                        + value_dim * 64
                                        + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                                    )
                                    (sCheckpoint_ptr + checkpoint_addr).store(
                                        cutlass.Vector.from_elements(packs, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                                    )
                            nvvm.fence_proxy("async.shared", space="cta")
                            bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                        bars.mb_state_acc_read_done.arrive()

            if cutlass.const_expr(cfg.enable_checkpoints and mState_init is None):
                if write_start == 0:
                    checkpoint_stage = checkpoint_done_index.idx
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_done_index.phase)
                    checkpoint_done_index = advance(checkpoint_done_index, cfg.smem_checkpoint_stages)
                    checkpoint_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    zero_packs = tuple(cutlass.Int32(0) for _ in range(4))
                    for i in cutlass.range_constexpr(cfg.d_k // 16):
                        for g in cutlass.range_constexpr(2):
                            dk = i * 16 + g * 8
                            checkpoint_addr = (
                                checkpoint_stage_base + (dk // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                            )
                            (sCheckpoint_ptr + checkpoint_addr).store(
                                cutlass.Vector.from_elements(zero_packs, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                            )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            bars.mb_v_ready[raw_index.idx].wait(raw_index.phase)
            raw_v_frag_lo = nvvm.ldmatrix(
                sV_ptr + v_swz_off_lo,
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag_hi = nvvm.ldmatrix(
                sV_ptr + v_swz_off_hi,
                4,
                nvvm.MMALayout.COL,
            )
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)
            beta_pack = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_pack[reg_idx] = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)

            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if cutlass.const_expr(mState_init is not None):
                if seed_from_initial_state:
                    bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
                    state_k_acc_index = advance(state_k_acc_index, 1)
                    state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id, cutlass.Float32), num=2)
                    state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        state_k_lo = fp32_to_fp16(state_k_vec_lo[frag_pair], state_k_vec_lo[frag_pair + 1], dtype=cfg.io_dtype)
                        state_k_hi = fp32_to_fp16(state_k_vec_hi[frag_pair], state_k_vec_hi[frag_pair + 1], dtype=cfg.io_dtype)
                        y_lo[reg_idx] = mul_f16x2(beta_pack[reg_idx], sub_f16x2(raw_v_frag_lo[reg_idx], state_k_lo, cfg.io_dtype), cfg.io_dtype)
                        y_hi[reg_idx] = mul_f16x2(beta_pack[reg_idx], sub_f16x2(raw_v_frag_hi[reg_idx], state_k_hi, cfg.io_dtype), cfg.io_dtype)
                else:
                    for reg_idx in cutlass.range_constexpr(4):
                        y_lo[reg_idx] = mul_f16x2(beta_pack[reg_idx], raw_v_frag_lo[reg_idx], cfg.io_dtype)
                        y_hi[reg_idx] = mul_f16x2(beta_pack[reg_idx], raw_v_frag_hi[reg_idx], cfg.io_dtype)
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = mul_f16x2(beta_pack[reg_idx], raw_v_frag_lo[reg_idx], cfg.io_dtype)
                    y_hi[reg_idx] = mul_f16x2(beta_pack[reg_idx], raw_v_frag_hi[reg_idx], cfg.io_dtype)

            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id, cutlass.Int8), y_input_pack_lo[0:4])
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_v_done[raw_index.idx].arrive()
            bars.mb_beta_done[raw_index.idx].arrive()
            bars.mb_y_input_ready.arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_input_pack = cutlass.Array(cutlass.Int32, (cfg.b_t // 2), alignment=16)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr, cutlass.Int8),
                u_input_pack[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            bars.mb_u_input_ready.arrive()
            if cutlass.const_expr(cfg.enable_checkpoints):
                bars.mb_state_acc_done.wait(state_upd_index.phase)
                state_upd_index = advance(state_upd_index, 1)

            raw_index = advance(raw_index, cfg.smem_raw_stages)

        if cutlass.const_expr(cfg.enable_checkpoints):
            cg1_checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            cg1_checkpoint_mod = (compute_start + cutlass.Int32(1)) % cg1_checkpoint_chunks
        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t

            do_checkpoint = False
            if cutlass.const_expr(cfg.enable_checkpoints):
                do_checkpoint = cg1_checkpoint_mod == 0
                cg1_checkpoint_mod = cg1_checkpoint_mod + cutlass.Int32(1)
                cg1_checkpoint_mod = cutlass.Int32(0) if cg1_checkpoint_mod == cg1_checkpoint_chunks else cg1_checkpoint_mod
                do_checkpoint = do_checkpoint and chunk_idx >= write_start

            # ---- state stage: acc TMEM -> packed b16 TMEM ----------------------------
            if cutlass.const_expr(not cfg.enable_checkpoints):
                bars.mb_state_acc_done.wait(state_upd_index.phase)
                state_upd_index = advance(state_upd_index, 1)
            state_vecs = []
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32), num=16))

            for i in cutlass.range_constexpr(cfg.d_k // 16):
                state_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                for packed_col in cutlass.range_constexpr(8):
                    state_pack[packed_col] = fp32_to_fp16(state_vecs[i][2 * packed_col], state_vecs[i][2 * packed_col + 1], dtype=cfg.io_dtype)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                    state_pack[0:8],
                )
            nvvm.tcgen05_wait("store")
            bars.mb_state_input_ready.arrive()

            # ---- checkpoint store ----------------------------------------------------
            if cutlass.const_expr(cfg.enable_checkpoints):
                if do_checkpoint:
                    checkpoint_stage = checkpoint_done_index.idx
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_done_index.phase)
                    checkpoint_done_index = advance(checkpoint_done_index, cfg.smem_checkpoint_stages)
                    checkpoint_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    for k_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                        checkpoint_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id + k_base, cutlass.Float32), num=32)
                        for g in cutlass.range_constexpr(4):
                            packs = tuple(fp32_to_fp16(checkpoint_vec[g * 8 + 2 * t], checkpoint_vec[g * 8 + 2 * t + 1], dtype=cfg.io_dtype) for t in range(4))
                            dk = k_base + g * 8
                            checkpoint_addr = (
                                checkpoint_stage_base + (dk // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                            )
                            (sCheckpoint_ptr + checkpoint_addr).store(cutlass.Vector.from_elements(packs, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16)
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                    nvvm.tcgen05_wait("load")
                    bars.mb_state_acc_read_done.arrive()
                else:
                    bars.mb_state_acc_read_done.arrive()

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            bars.mb_v_ready[raw_index.idx].wait(raw_index.phase)
            raw_v_frag_lo = nvvm.ldmatrix(
                sV_ptr + v_swz_off_lo,
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag_hi = nvvm.ldmatrix(
                sV_ptr + v_swz_off_hi,
                4,
                nvvm.MMALayout.COL,
            )
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)

            # ---- read k state acc ----------------------------------------------------
            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
            state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id, cutlass.Float32), num=2)
            state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + statek_col_id, cutlass.Float32), num=2)

            beta_pack = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_pack[reg_idx] = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
            y_input_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                state_k_val0, state_k_val1 = state_k_vec_lo[frag_pair], state_k_vec_lo[frag_pair + 1]
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_frag_lo[reg_idx],
                    state_k_pair,
                    cfg.io_dtype,
                )
                y_input_pack_lo[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            y_input_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                state_k_val0, state_k_val1 = state_k_vec_hi[frag_pair], state_k_vec_hi[frag_pair + 1]
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_frag_hi[reg_idx],
                    state_k_pair,
                    cfg.io_dtype,
                )
                y_input_pack_hi[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id, cutlass.Int8), y_input_pack_lo[0:4])
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id, cutlass.Int8), y_input_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            bars.mb_v_done[raw_index.idx].arrive()
            bars.mb_beta_done[raw_index.idx].arrive()
            bars.mb_y_input_ready.arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_input_pack = cutlass.Array(cutlass.Int32, (cfg.b_t // 2), alignment=16)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr, cutlass.Int8),
                u_input_pack[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            bars.mb_u_input_ready.arrive()

            if cutlass.const_expr(cfg.enable_checkpoints):
                bars.mb_state_acc_done.wait(state_upd_index.phase)
                state_upd_index = advance(state_upd_index, 1)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        if num_chunks_tile > 0:
            if cutlass.const_expr(not cfg.enable_checkpoints):
                bars.mb_state_acc_done.wait(state_upd_index.phase)
                state_upd_index = advance(state_upd_index, 1)

        owns_final = write_end == batch_num_chunks

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if cutlass.const_expr(mState_out is not None):
            if batch_seqlen > 0:
                if owns_final:
                    state_vw = 16 // (mState_out.element_type.width // 8)
                    state_dst = (mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim, 0))).raw_ptr()
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        loaded = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_state_acc_offset + key_block_start), cutlass.Float32),
                            num=32,
                        )

                        for g in cutlass.range_constexpr(32 // state_vw):
                            (state_dst + key_block_start + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(loaded[g * state_vw + t].to(mState_out.element_type) for t in range(state_vw)),
                                    mState_out.element_type,
                                ),
                                alignment=16,
                            )
            else:
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for col in cutlass.range_constexpr(32):
                        key_dim = key_block_start + col
                        if cutlass.const_expr(mState_init is not None):
                            mState_out[batch_idx, head_o, value_dim, key_dim] = mState_init[batch_idx, head_o, value_dim, key_dim]
                        else:
                            mState_out[batch_idx, head_o, value_dim, key_dim] = cutlass.Float32(0.0).to(mState_out.element_type)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, tile_idx, num_ctas, elect_one)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    stream,
) -> None:
    num_sequences = cu_seqlens.shape[0] - 1
    grid_shape = (cfg.max_active_clusters, 1, 1)
    kernel(
        cfg,
        tensormap_workspace,
        cutlass.Int32(num_sequences),
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        final_state,
        work_items,
        work_count,
        scheduler_counter,
        checkpoint_every_n_tokens,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.kernel
def kernel(
    cfg: cutlass.Constexpr,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mState_out: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor | None,
    checkpoint_every_n_tokens: cutlass.Int32,
) -> None:
    """BT=16 KDA recompute (state/checkpoints-only) persistent kernel body: every warp
    role runs a tile-scheduler loop over the tiles."""

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dynamic_scheduling):
        assert mScheduler is not None and mScheduler.element_type == cutlass.Int32
    assert mK.element_type == cfg.io_dtype and mV.element_type == cfg.io_dtype
    assert mGate.element_type == cutlass.Float32
    beta_expected = cfg.io_dtype if cutlass.const_expr(cfg.beta_sigmoid) else cutlass.Float32
    assert mBeta.element_type == beta_expected
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    if cutlass.const_expr(cfg.use_initial_state):
        assert mState_init is not None and mState_init.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mState_init is None, "mState_init must be None if use_initial_state is False"
    if cutlass.const_expr(cfg.store_final_state):
        assert mState_out is not None and mState_out.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mState_out is None, "mState_out must be None if store_final_state is False"
    if cutlass.const_expr(mState_init is not None and mState_out is not None):
        assert mState_init.element_type == mState_out.element_type
    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_k_base = desc_base_words
    desc_v_base = desc_base_words + arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_kda_bars(cfg)
    sTmem_base = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    sK_decay_raw = cutlass.Array(cfg.io_dtype, cfg.k_decay_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_restore_raw = cutlass.Array(cfg.io_dtype, cfg.k_restore_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sIntermediate_raw = cutlass.Array(cfg.io_dtype, cfg.intermediate_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_raw = cutlass.Array(mK.element_type, cfg.k_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sV_raw = cutlass.Array(mV.element_type, cfg.v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sGate_raw = cutlass.Array(cutlass.Float32, cfg.gate_cosize, space=SMEM, alignment=1024)
    sState_scale_diag_raw = cutlass.Array(cfg.io_dtype, cfg.state_scale_diag_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_inv_raw = cutlass.Array(cfg.io_dtype, cfg.k_inv_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sBeta_raw = cutlass.Array(cutlass.Float32, cfg.beta_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sCheckpoint_raw = (
        cutlass.Array(cfg.io_dtype, cfg.smem_checkpoint_stages * cfg.d_k * cfg.d_v, space=SMEM, alignment=cfg.buffer_align_bytes)
        if cutlass.const_expr(cfg.enable_checkpoints)
        else sV_raw
    )
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
        leading_byte_offset=(cfg.b_t * (cfg.d_v // 2) * 2),
        stride_byte_offset=(8 * (cfg.d_v // 2) * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sState_scale_diag = SmemTile(
        base=sState_scale_diag_raw,
        elems_per_stage=((cfg.d_k // 16) * 256),
        stages=cfg.smem_state_scale_diag_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * 16 * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    sIntermediate = SmemTile(
        base=sIntermediate_raw,
        elems_per_stage=(2 * cfg.b_t * cfg.b_t),
        stages=cfg.smem_intermediate_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * cfg.b_t * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )

    elect_one = nvvm.elect_sync()
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_k_ready[stage].init()
                bars.mb_v_ready[stage].init()
                bars.mb_gate_ready[stage].init()
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
                bars.mb_k_done[stage].init()
                bars.mb_v_done[stage].init()
                bars.mb_gate_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_ready.init()
            bars.mb_u_acc_ready.init()
            bars.mb_state_acc_done.init()
            bars.mb_state_input_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_state_scale_diag_stages):
                bars.mb_state_scale_diag_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_decay_super_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_y_input_ready.init()
            bars.mb_u_input_ready.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.super_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_t_inv_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.qk_scale_ready_stages):
                bars.mb_qk_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_cg0_ready[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.scheduler_stages):
                bars.mb_scheduler_ready[stage].init()
                bars.mb_scheduler_done[stage].init()
            if cutlass.const_expr(cfg.enable_checkpoints):
                for stage in cutlass.range_constexpr(cfg.smem_checkpoint_stages):
                    bars.mb_checkpoint_tmastg_ready[stage].init()
                    bars.mb_checkpoint_tmastg_done[stage].init()
                bars.mb_state_acc_read_done.init()
    diag_zero = cfg.io_dtype(0.0)
    for diag_idx in cutlass.range(tidx, cfg.state_scale_diag_cosize, cfg.threads_per_cta, unroll=1):
        sState_scale_diag_raw[diag_idx] = diag_zero
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0, thread_count=cfg.threads_per_cta)
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
            sK_raw,
            sV_raw,
            sGate_raw,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
            bars,
        )
    elif warp_idx == cfg.super_mma_warp_id:
        super_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            sK_inv_raw,
            sIntermediate_raw,
            sBeta_raw,
            sK_decay_raw,
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
            sIntermediate,
            sK_decay,
            sK_restore_trans,
            sState_scale_diag,
            bars,
        )
    elif warp_idx == cfg.epilogue_warp_id:
        epilogue_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            sCheckpoint_raw,
            desc_checkpoint_base,
            checkpoint_every_n_tokens,
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
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_raw,
            mBeta,
            sBeta_raw,
            sK_raw,
            sK_decay_raw,
            sK_restore_raw,
            sState_scale_diag_raw,
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
            mState_init,
            sBeta_raw,
            sV_raw,
            sCheckpoint_raw,
            checkpoint_every_n_tokens,
            bars,
        )


@dataclass
class KdaRecomputeCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``; per-stage sizes are
    inlined at the use sites).  Passed ``cfg``-first (a ``cutlass.Constexpr``)
    into ``host`` / ``kernel`` and every warp body."""

    io_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    dynamic_scheduling: bool = False
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    super_mma_warp_id: int = CFG.SUPER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0
    cg0_group_count: int = 2
    cg0_warps_per_group: int = 4
    cg0_threads_per_group: int = 0
    cg0_group_sync_barrier_base_id: int = 1  # CG0 group g syncs on named-barrier id 1 + g
    cg0_tile_entry_barrier_id: int = 5  # CG0-wide (both groups) work-item entry sync
    tmem_user_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -----------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_checkpoint_stages: int = 1
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_state_scale_diag_stages: int = CFG.SMEM_STATE_SCALE_DIAG_STAGES
    qk_scale_ready_stages: int = CFG.QK_SCALE_READY_STAGES

    # ---- TMEM column offsets (state doubles as the final state acc) ------------------
    tmem_state_acc_offset: int = 0
    tmem_state_input_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_y_input_offset: int = 0
    tmem_u_input_offset: int = 0

    # ---- SMEM buffer cosizes ---------------------------------------------------------
    k_cosize: int = 0
    v_cosize: int = 0
    gate_cosize: int = 0
    beta_cosize: int = 0
    k_inv_cosize: int = 0
    k_decay_cosize: int = 0
    k_restore_cosize: int = 0
    state_scale_diag_cosize: int = 0

    # TMA transaction bytes per stage
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_gate_bytes: int = 0
    intermediate_cosize: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    dynamic_scheduling: bool = False,
) -> KdaRecomputeCfg:
    """Build the per-compile ``KdaRecomputeCfg`` (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = KdaRecomputeCfg(
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        dynamic_scheduling=dynamic_scheduling,
    )
    if enable_checkpoints:
        cfg.smem_raw_stages = 6
        cfg.smem_checkpoint_stages = 2
    if cfg.smem_raw_stages % 2 != 0:
        raise ValueError("smem_raw_stages must be even: the CG0 ping-pong groups alias parity waits on odd rings")
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.cg0_threads_per_group = cfg.cg0_warps_per_group * cfg.threads_per_warp
    cfg.tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids)) * cfg.threads_per_warp
    if cfg.smem_state_scale_diag_stages != cfg.qk_scale_ready_stages:
        raise ValueError("diag and qk-scale ready rings must share their rolling stage")

    cfg.tmem_state_input_offset = cfg.tmem_state_acc_offset + cfg.d_k
    cfg.tmem_state_k_acc_offset = cfg.tmem_state_input_offset + (cfg.d_k // 2)
    cfg.tmem_u_acc_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    cfg.tmem_y_input_offset = cfg.tmem_u_acc_offset + cfg.b_t
    cfg.tmem_u_input_offset = cfg.tmem_y_input_offset + (cfg.b_t // 2)
    assert (cfg.tmem_u_input_offset + (cfg.b_t // 2)) <= 512

    cfg.k_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.beta_cosize = cfg.smem_raw_stages * cfg.b_t
    cfg.k_inv_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.state_scale_diag_cosize = cfg.smem_state_scale_diag_stages * (cfg.d_k // 16) * 256
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * 2 * cfg.b_t * cfg.b_t
    cfg.tma_k_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_v_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_gate_bytes = cfg.d_k * cfg.b_t * 4
    return cfg


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: K, V, Gate, state_checkpoints
TENSORMAP_STATIC_SLOTS = 0


@cute.jit
def build_descs_body(
    widx,
    base_k,
    base_v,
    base_gate,
    base_checkpoint,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    n_batch: cutlass.Int32,
    k_row_stride: cutlass.Int32,
    v_row_stride: cutlass.Int32,
    gate_row_stride: cutlass.Int32,
    checkpoint_row_stride: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_words_k = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_v = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_gate = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_words_checkpoint = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_descs(base_k, desc_words_k, cu_seqlens, k, n_batch, k_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_descs(base_v, desc_words_v, cu_seqlens, v, n_batch, v_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_descs(base_gate, desc_words_gate, cu_seqlens, gate, n_batch, gate_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(state_checkpoints is not None):
        if widx == 3:
            if nvvm.elect_sync():
                emit_checkpoint_seq_descs(
                    base_checkpoint, desc_words_checkpoint, cu_seqlens, state_checkpoints, n_batch, checkpoint_row_stride, checkpoint_every_n, 2
                )
                nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def prologue_kernel(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_scheduler: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor | None,
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
    k_row_stride: cutlass.Int32,
    v_row_stride: cutlass.Int32,
    gate_row_stride: cutlass.Int32,
    checkpoint_row_stride: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Single-CTA prologue. Under ``run_order`` this kernel is the first
    work-item-table consumer, so it LPT-orders the table and zeroes both
    consumers' scheduler rings via :func:`order_body`; it then builds the
    per-batch TMA-descriptor arrays via :func:`build_descs_body`, one warp
    per array (the extra warps only take part in the order phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    if cutlass.const_expr(run_order):
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(gate.shape[1])
        order_body(
            order_gen,
            has_scheduler,
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
        base_k,
        base_v,
        base_gate,
        base_checkpoint,
        desc_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        state_checkpoints,
        n_batch,
        k_row_stride,
        v_row_stride,
        gate_row_stride,
        checkpoint_row_stride,
        checkpoint_every_n,
    )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_scheduler: cutlass.Constexpr[bool],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when ``run_order``) and
    build the per-batch K/V/Gate/checkpoint TMA-descriptor arrays into
    ``tensormap_workspace``."""
    h_k = k.shape[1]
    h_v = v.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = k.shape[2]
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    tma_granu_elems = 128 // bpe
    seqlen = k.shape[0]

    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])))

    swz = cuda.TensorMapSwizzle.s128b
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)

    base_checkpoint = base_gate
    if cutlass.const_expr(state_checkpoints is not None):
        checkpoint_view = cute.make_tensor(
            state_checkpoints.iterator,
            cute.make_layout(
                (state_checkpoints.shape[3], state_checkpoints.shape[2], state_checkpoints.shape[0], ho),
                stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
            ),
        )
        base_checkpoint = cuda.create_tensor_map_tiled_from_view(checkpoint_view, box_dims=(tma_granu_elems, d_k, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swz)
    prologue_kernel(
        run_order,
        order_gen,
        has_scheduler,
        b_t,
        base_k,
        base_v,
        base_gate,
        base_checkpoint,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        state_checkpoints,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
        cutlass.Int32(k.stride[0]),
        cutlass.Int32(v.stride[0]),
        cutlass.Int32(gate.stride[0]),
        cutlass.Int32(state_checkpoints.stride[0] if state_checkpoints is not None else 0),
        checkpoint_every_n,
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


# ---- Torch adapter / host-side compilation -------------------------------------------


@lru_cache(maxsize=None)
def get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    cu_dtype_str: str,
    HO: int,
    HK: int,
    HV: int,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    beta_sigmoid: bool,
    dynamic_scheduling: bool,
    run_order: bool,
    order_gen: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    state_dtype,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    dynamic_scheduling: bool = False,
    *,
    num_sm: int,
    k_cute,
    v_cute,
    gate_cute,
    a_log_cute,
    dt_bias_cute,
    beta_cute,
    cu_seqlens_cute,
    state_in_cute,
    state_out_cute,
    work_items_cute=None,
    work_count_cute=None,
    scheduler_counter_cute=None,
    tensormap_workspace_cute,
    checkpoint_every_n_tokens,
    stream,
):
    """JIT-compile the chunked KDA recompute kernel for one static config."""
    cfg = build_cfg(
        io_dtype,
        state_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=num_sm,
        dynamic_scheduling=dynamic_scheduling,
    )

    return cute.compile(
        host,
        cfg,
        k_cute,
        v_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        beta_cute,
        cu_seqlens_cute,
        state_in_cute,
        state_out_cute,
        work_items_cute,
        work_count_cute,
        scheduler_counter_cute,
        tensormap_workspace_cute,
        checkpoint_every_n_tokens,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_kda_recompute_sm100(
    k,
    v,
    gate,
    beta,
    cu_seqlens,
    initial_state,
    output_state,
    checkpoint_every_n_tokens: int = 0,
    output_state_checkpoints=None,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    work_items=None,
    work_count=None,
    scheduler_counter=None,
    scheduler_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    *,
    tensormap_workspace,
    stream,
) -> None:
    """Execute the Blackwell BT=16 chunked KDA recompute (state/checkpoints-only)
    kernel.

    All tensors must be on the same CUDA device with a stride-1 innermost
    dim; outer strides are free (padded / permuted views are read through
    the TMA descriptors and dynamic layouts).

    Args:
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO, DK)`` float32.  Natural-log decay unless
              ``safe_gate``, which applies the safe-gate transform
              ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``.
        beta: ``(total_tokens, HO)``.  Post-sigmoid float32, or io-dtype
              logits when ``use_beta_sigmoid``
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DV, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DV, DK)`` float32/bfloat16, or None
        checkpoint_every_n_tokens: emit a state checkpoint every N tokens (0 = off).
            state_checkpoints[j] is the state after ``(j + 1) * N`` tokens, STRICTLY BEFORE
            the sequence end - the end-of-sequence state is only
            ``output_state``.  With ``N == B_T`` this is the per-chunk checkpoint
            series the backward pass consumes.
        output_state_checkpoints: ``(total_checkpoints, HO, DV, DK)`` io-dtype (VK, K
            contiguous); the per-sequence entry offsets
            are derived on device from ``cu_seqlens`` ((seqlen-1)//N,
            prefix-summed), so there is no cu_checkpoints array
        use_qk_l2norm_in_kernel: L2-normalize k rows inside the kernel
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO, DK)`` float32, safe-gate channel bias (None = 0)
        use_beta_sigmoid: ``beta`` holds logits; sigmoid in-kernel
        work_items: ``(max_items, 8)`` int32 work-item table from
            ``common/split_k.py`` (REQUIRED; an uncut table row is the whole
            (b, h) sequence).  Each item computes chunks ``[compute_start, write_end)``
            and writes checkpoints only for ``[write_start, write_end)``.
        work_count: ``(1,)`` int32 device-side item count (REQUIRED)
        scheduler_counter: ``(2,)`` int32 device scratch ``[ticket, done]`` enabling
            the dynamic (work-stealing) tile scheduler; must be zeroed before
            every launch (the split-table stage and the order-generating
            prologue both zero it when passed as ``scheduler_counter``).
            None keeps the static CTA stride.
    """
    HK = k.shape[1]
    HV = v.shape[1]
    HO = gate.shape[1]
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if enable_checkpoints:
        if output_state_checkpoints is None:
            raise ValueError("checkpoint_every_n_tokens > 0 requires output_state_checkpoints")
        if str(output_state_checkpoints.dtype).split(".")[-1] != str(k.dtype).split(".")[-1]:
            raise ValueError(
                f"output_state_checkpoints dtype must match the io dtype (fp32 state belongs to output_state): got {output_state_checkpoints.dtype} with io {k.dtype}"
            )
    if work_items is None or work_count is None:
        raise ValueError(
            "work_items/work_count are required (built by the split-table stage, or by an order-generating prologue when work_item_scratch is None)"
        )
    dynamic_scheduling = scheduler_counter is not None
    run_order = order_in_prologue
    order_gen = order_in_prologue and work_item_scratch is None
    if run_order and scheduler_all is None:
        raise ValueError("order in the prologue requires scheduler_all (the prologue zeroes both consumers' scheduler rings)")

    if initial_state is not None:
        state_dtype_src = initial_state.dtype
    elif output_state is not None:
        state_dtype_src = output_state.dtype
    else:
        state_dtype_src = "float32"

    for name, h in (("HK", HK), ("HV", HV)):
        if HO % h != 0:
            raise ValueError(f"{name}={h} must divide sab heads {HO}")
    k_ratio = HO // HK
    v_ratio = HO // HV
    gate_scale_log2 = gate_lower_bound * LOG2_E

    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if not safe_gate:
        a_log = None
        dt_bias = None
    cu_stream = cuda_driver.CUstream(int(stream))

    cache = get_compiled_cache(
        str(k.dtype),
        str(state_dtype_src),
        str(cu_seqlens.dtype),
        HO,
        HK,
        HV,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid,
        dynamic_scheduling,
        run_order,
        order_gen,
    )

    if "compiled" not in cache:
        io_dtype = get_dtype(k.dtype)
        state_dtype = get_dtype(state_dtype_src)
        k_cute = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        v_cute = from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        gate_cute = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        a_log_cute = from_dlpack(a_log, assumed_align=4) if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16) if dt_bias is not None else None
        beta_cute = from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()

        state_in_cute = None
        if use_initial_state:
            state_in_cute = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)

        state_out_cute = None
        if store_final_state:
            state_out_cute = from_dlpack(output_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)

        work_items_cute = from_dlpack(work_items, assumed_align=16)
        work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()

        scheduler_counter_cute = None
        if dynamic_scheduling:
            scheduler_counter_cute = from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic()

        tensormap_workspace_cute = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()

        cache["compiled"] = compile(
            io_dtype,
            state_dtype,
            use_initial_state,
            store_final_state,
            enable_checkpoints,
            use_qk_l2norm_in_kernel,
            safe_gate,
            gate_scale_log2,
            use_beta_sigmoid,
            k_ratio,
            v_ratio,
            HO,
            dynamic_scheduling,
            num_sm=multiprocessor_count(current_device()),
            k_cute=k_cute,
            v_cute=v_cute,
            gate_cute=gate_cute,
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            beta_cute=beta_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            state_in_cute=state_in_cute,
            state_out_cute=state_out_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            scheduler_counter_cute=scheduler_counter_cute,
            tensormap_workspace_cute=tensormap_workspace_cute,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            stream=cu_stream,
        )

    compiled = cache["compiled"]
    state_checkpoints_for_descs = output_state_checkpoints if enable_checkpoints else None
    if "prologue" not in cache:
        io_dtype = get_dtype(k.dtype)
        k_pl = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        v_pl = from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        gate_pl = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        cu_pl = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        workspace_pl = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()
        state_checkpoints_pl = None
        if state_checkpoints_for_descs is not None:
            state_checkpoints_pl = from_dlpack(state_checkpoints_for_descs, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        staging_pl = None
        if run_order and not order_gen:
            staging_pl = from_dlpack(work_item_scratch, assumed_align=16)
            staging_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_pl = from_dlpack(work_items, assumed_align=16)
        work_items_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_pl = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        scheduler_pl = None
        if run_order:
            scheduler_pl = from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic()
        cache["prologue"] = cute.compile(
            prologue,
            io_dtype,
            CFG.B_T,
            run_order,
            order_gen,
            run_order,
            k_pl,
            v_pl,
            gate_pl,
            state_checkpoints_pl,
            cu_pl,
            staging_pl,
            work_count_pl,
            work_items_pl,
            scheduler_pl,
            workspace_pl,
            cutlass.Int32(checkpoint_every_n_tokens),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["prologue"](
        k,
        v,
        gate,
        state_checkpoints_for_descs,
        cu_seqlens,
        work_item_scratch if run_order else None,
        work_count,
        work_items,
        scheduler_all if run_order else None,
        tensormap_workspace,
        checkpoint_every_n_tokens,
        cu_stream,
    )
    compiled(
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state if use_initial_state else None,
        output_state if store_final_state else None,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        checkpoint_every_n_tokens,
        cu_stream,
    )
    return cache


def run_recompute(
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
    output_state_checkpoints,
    work_items,
    work_count,
    scheduler_counter,
    scheduler_all,
    work_item_scratch,
    tensormap_workspace,
    checkpoint_every_n_tokens,
    stream,
) -> None:
    """Replay the compiled plan: the prologue launch, then the main launch.
    The caller owns the contract, which the plan validated at build, so
    nothing here raises."""
    cu_stream = cuda_driver.CUstream(int(stream))
    cache["prologue"](
        k,
        v,
        gate,
        output_state_checkpoints,
        cu_seqlens,
        work_item_scratch,
        work_count,
        work_items,
        scheduler_all,
        tensormap_workspace,
        checkpoint_every_n_tokens,
        cu_stream,
    )
    cache["compiled"](
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        output_state,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        checkpoint_every_n_tokens,
        cu_stream,
    )
