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
Chunked Gated Delta Net 2 (GDN-2) bprop state-summary kernel for Blackwell SM100 (Cutlass primitives): the BT = 16
reverse state-gradient recurrence alone over q, k, Gate (+ a_log / dt_bias under safe_gate), Beta, dO and an optional
d_final_state seed (zero when absent); the single output is d_initial_state (the gradient-state pass G, or the full dh0
when d_final_state is given).

Algorithm overview (per chunk c, iterated c = compute_end-1 .. write_start):
  Inputs : Q[BT,DK], K[BT,DK], dO[BT,DV], Gate[BT,DK] (per-channel gate), Beta[BT,DK] (erase gate)
  State  : dstate[DK,DV]  (state gradient, held in TMEM, seeded from d_final_state or zero, accumulated backward)

  Preprocessing (compute group 0): g, K decay = Beta * K * exp2(+g), K inv, Q decay as in the prefill, the decay scale
  exp2(g[BT-1,:]), the Beta guard.
  Register MMA (warp 12): KK = K decay @ K inv^T; L = tril(KK, -1); T_inv by three Neumann doubling rounds.
  Register MMA (warp 15): A = tril(Q decay @ K inv^T, 0).

  MMA order (tcgen05, warp 13; (S) = SMEM operand, (T) = TMEM operand):
  dU inter      : dU inter = dstate input(T) @ K inv
  dU intra      : dU intra += dO^T(S) @ A
  dstate Q-term : dstate Q-term += dO^T(S) @ Q decay
  dY            : dY = dU(T) @ T^-1
  dstate K-term : dstate K-term += -dY(T) @ K decay

  Epilogue (compute group 1): dstate captured for the next chunk (f16 pack, decayed carry); the item owning chunk 0
  stores d_initial_state.

SMEM layout (stage counts live in gdn2_bprop_config.py and the kernel cfg; sizes at DK = DV = 128, bf16 io, fp32 Gate):
  Buffer                       Size (B)  Stages
  Q / K / Beta (raw)           3 x 4096       2
  dO (raw)                         4096       2
  Gate (raw)                       8192       2
  K decay / K inv / Q decay    3 x 4096       2
  decay scale                       512       2
  T_inv / A (intermediate)         2560       2    <-- five BT x BT tiles per stage, as the bprop
  sub-bank fill                   17408       1    <-- 1024 B plus two operand rings keeps the high group at 128 KB
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated):
  Buffer                  Cols
  dstate acc              128     <-- DKxDV fp32
  dstate input             64     <-- f16 packed
  dU acc                   16
  dY acc                   16
  dU input                  8     <-- f16 packed
  -dY input                 8

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - Gate prefix scan, decay operands, Beta guard
  warps 4-7     : compute group 1 - dstate seed, dU / -dY stagings, dstate capture, d_initial_state store
  warps 8-11    : exit after init
  warp  12      : super MMA warp - register-MMA KK and T_inv
  warp  13      : MMA warp       - every tcgen05 GEMM; TMEM lifecycle
  warp  14      : TMA load warp  - loads Q, K, Gate, Beta, dO
  warp  15      : epilogue warp  - register-MMA A tile
"""

from dataclasses import dataclass
import functools
from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.beta_guard import beta_guard
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_work_item, order_body
from ..common.host import get_dtype
from cudnn.frost.buffers import data_ptr
from ..common.thd import TENSOR_MAP_QWORDS, emit_seq_descs
from .gdn2_bprop_config import CFG

from cudnn.frost.tile_dsl.barrier import (
    launch_dependent_grids,
    wait_on_dependent_grids,
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_ss, mma_step, mma_ts_step
from cudnn.frost.tile_dsl.swizzle import swizzle_xor_128b, swizzle_xor_32b
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from cudnn.frost.tile_dsl.pointwise import (
    sigmoid,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    movmatrix_16b,
    opaque_f32_zero,
)

USE_PDL = True

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class Gdn2BpropSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier

    mb_k_decay_inv_ready: MBarrier
    mb_q_decay_ready: MBarrier
    mb_decay_done: MBarrier
    mb_decay_scale_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier
    mb_t_inv_done: MBarrier

    mb_du_acc_ready: MBarrier
    mb_du_input_ready: MBarrier
    mb_dy_acc_ready: MBarrier
    mb_neg_dy_input_ready: MBarrier
    mb_dstate_acc_ready: MBarrier
    mb_dstate_input_ready: MBarrier
    mb_dstate_input_done: MBarrier

    mb_dstate0_acc_stored: MBarrier
    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> Gdn2BpropSummaryBars:
    """Gdn2BpropSummaryBars factory."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0 = len(cfg.compute_group_0_warp_ids) * WARP
    CG1 = len(cfg.compute_group_1_warp_ids) * WARP
    MMA = 1

    return Gdn2BpropSummaryBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_do_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_k_decay_inv_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_q_decay_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_decay_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_decay_scale_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_t_inv_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_du_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_du_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dy_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_neg_dy_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_input_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_input_done=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate0_acc_stored=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        # scheduler consumers: CG0 x4 + CG1 x4 + super + tcgen05 + epilogue
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), stages=cfg.scheduler_stages, init_count=11, producer=Producer.THREAD),
    )


@cute.jit
def scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, tail_base, tail_row, elect_one):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
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


@cute.jit
def scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    bars.mb_scheduler_ready[scheduler_state.idx].wait(scheduler_state.phase)
    next_tile = sScheduler[scheduler_state.idx]
    if elect_one:
        bars.mb_scheduler_done[scheduler_state.idx].arrive()
    return next_tile, advance(scheduler_state, cfg.scheduler_stages)


@cute.jit
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    sK_inv_raw,
    sQ_decay_raw,
    sIntermediate_raw,
    bars,
) -> None:
    """Epilogue warp role (warp 15): the register-MMA A tile build, in chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    elect_one = nvvm.elect_sync()

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    b_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    b_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    a_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    a_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    intermediate_row_coord = lane_idx & 7
    intermediate_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        intermediate_row_coord = intermediate_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        intermediate_col_coord = cutlass.Int32(8)
    intermediate_idx = intermediate_row_coord * cfg.b_t + swizzle_xor_32b(intermediate_row_coord, intermediate_col_coord)
    row_lo = lane_idx // 4
    row_hi = row_lo + cutlass.Int32(8)

    # tril bitmask: bit i = row >= col for accum index i
    tril_incl_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_incl_mask = tril_incl_mask | (cutlass.Int32(1 << accum_idx) if row_coord >= col_coord else cutlass.Int32(0))
    gbase = cutlass.Int32(0)

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        sk_nt = compute_end - write_start
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            gc = gbase + rev_idx
            decay_stage = gc % cfg.smem_decay_stages
            intermediate_stage = gc % cfg.smem_intermediate_stages
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = sIntermediate_raw.data_ptr() + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            # ---- A = tril(Q decay @ K inv^T, 0) --------------------------------------
            bars.mb_a_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)
            bars.mb_q_decay_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            a_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = i * 16 + a_col_offset
                a_seg = a_col // 64
                q_decay_frag = nvvm.ldmatrix(
                    sQ_decay_ptr + a_seg * (cfg.b_t * 64) + a_row_coord * 64 + swizzle_xor_128b(a_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = i * 16 + b_col_offset
                b_seg = b_col // 64
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + b_row_coord * 64 + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    a_acc,
                    (q_decay_frag[0], q_decay_frag[1], q_decay_frag[2], q_decay_frag[3]),
                    (k_inv_frag[0], k_inv_frag[1], k_inv_frag[2], k_inv_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = a_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            nvvm.stmatrix(
                sIntermediate_ptr + intermediate_idx,
                [
                    fp32_to_fp16(a_acc[0], a_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[2], a_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[4], a_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[6], a_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[intermediate_stage].arrive()

        gbase += sk_nt
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


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
    sK_decay_raw,
    sK_inv_raw,
    sIntermediate_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): the Neumann T_inv staging tile, in chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    b_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    b_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    a_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    a_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    intermediate_row_coord = lane_idx & 7
    intermediate_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        intermediate_row_coord = intermediate_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        intermediate_col_coord = cutlass.Int32(8)
    intermediate_idx = intermediate_row_coord * cfg.b_t + swizzle_xor_32b(intermediate_row_coord, intermediate_col_coord)
    row_lo = lane_idx // 4
    row_hi = row_lo + cutlass.Int32(8)

    # tril bitmasks: bit i = row > col / row == col for accum index i
    tril_strict_mask = cutlass.Int32(0)
    eye_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_strict_mask = tril_strict_mask | (cutlass.Int32(1 << accum_idx) if row_coord > col_coord else cutlass.Int32(0))
        eye_mask = eye_mask | (cutlass.Int32(1 << accum_idx) if row_coord == col_coord else cutlass.Int32(0))

    gbase = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        sk_nt = compute_end - write_start
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            gc = gbase + rev_idx
            decay_stage = gc % cfg.smem_decay_stages
            intermediate_stage = gc % cfg.smem_intermediate_stages
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = sIntermediate_raw.data_ptr() + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            bars.mb_t_inv_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)

            # ---- KK = K decay @ K inv^T ----------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            kk_a_row = a_row_coord
            kk_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = i * 16 + a_col_offset
                a_seg = a_col // 64
                k_decay_frag = nvvm.ldmatrix(
                    sK_decay_ptr + a_seg * (cfg.b_t * 64) + kk_a_row * 64 + swizzle_xor_128b(kk_a_row, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = i * 16 + b_col_offset
                b_seg = b_col // 64
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + b_row_coord * 64 + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
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

            # ---- L = tril(KK, -1) ----------------------------------------------------
            l_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                lower = kk_acc[accum_idx] if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                l_regs[accum_idx] = lower
            l_a0 = fp32_to_fp16(l_regs[0], l_regs[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_regs[4], l_regs[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_regs[6], l_regs[7], dtype=cfg.io_dtype)

            # ---- T^-1 = I - L, then three Neumann doubling rounds --------------------
            tinv_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                eye = cutlass.Float32(1.0) if (eye_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                tinv_acc[accum_idx] = eye - l_regs[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(l_a0), movmatrix_16b(l_a1), movmatrix_16b(l_a2), movmatrix_16b(l_a3)
            for neumann_round in cutlass.range(3, unroll=1):
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
                tinv_acc[0], tinv_acc[1] = fadd2(tinv_acc[0], tinv_acc[1], upd_acc[0], upd_acc[1])
                tinv_acc[2], tinv_acc[3] = fadd2(tinv_acc[2], tinv_acc[3], upd_acc[2], upd_acc[3])
                tinv_acc[4], tinv_acc[5] = fadd2(tinv_acc[4], tinv_acc[5], upd_acc[4], upd_acc[5])
                tinv_acc[6], tinv_acc[7] = fadd2(tinv_acc[6], tinv_acc[7], upd_acc[6], upd_acc[7])

            nvvm.stmatrix(
                sIntermediate_ptr + 1 * (cfg.b_t * cfg.b_t) + intermediate_idx,
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
        gbase += sk_nt
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
    tmem_base_slot,
    sK_inv,
    sDo_trans,
    sQ_decay_trans,
    sK_decay_trans,
    sIntermediate,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): issues every tcgen05 GEMM and owns the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    bpe = cfg.io_dtype.width // 8

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    idesc_mv_nt = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dstate_k_inv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_mv_nt,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_du_at = nvvm.Tcgen05InstrDesc.build(
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
        idesc=idesc_du_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_q_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_do_q_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dstate_q_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_mv_nt_t = nvvm.Tcgen05InstrDesc.build(
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
        cta_group=1,
        idesc=idesc_mv_nt_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dy_k_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=idesc_dstate_k,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    dstate_input_index = PipelineState.start(phase=0)
    du_input_index = PipelineState.start(phase=0)
    neg_dy_index = PipelineState.start(phase=0)

    do_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    op_seg = (cfg.b_t * cfg.d_k * (cfg.io_dtype.width // 8)) >> 4
    intermediate_seg = (cfg.intermediate_tiles * cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    intermediate_slot = (cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    d_do_trans0 = sDo_trans[0].desc()
    d_qd_trans0 = sQ_decay_trans[0].desc()
    d_kd_trans0 = sK_decay_trans[0].desc()
    d_int0 = sIntermediate[0].desc()
    d_ki0 = sK_inv[0].desc()
    dstate0_index = PipelineState.start(phase=0)

    gbase = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        sk_nt = compute_end - write_start
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            gc = gbase + rev_idx
            decay_stage = gc % cfg.smem_decay_stages
            intermediate_stage = gc % cfg.smem_intermediate_stages
            decay_phase = (gc // cfg.smem_decay_stages) % 2
            intermediate_phase = (gc // cfg.smem_intermediate_stages) % 2
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)
            raw_stage_idx = gc % cfg.smem_raw_stages

            # ---- stage-derived operand descriptors -----------------------------------
            decay_op_off = decay_stage * op_seg
            d_do_trans = d_do_trans0 + raw_stage_idx * do_seg
            d_qd_trans = d_qd_trans0 + decay_op_off
            d_kd_trans = d_kd_trans0 + decay_op_off
            d_int = d_int0 + intermediate_stage * intermediate_seg
            d_int_tinv = d_int + intermediate_slot

            # ---- operand gates -------------------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(decay_phase)
            bars.mb_do_ready[raw_stage_idx].wait((gc // cfg.smem_raw_stages) % 2)

            # ---- dU inter = dstate input(T) @ K inv ----------------------------------
            if has_dstate:
                bars.mb_dstate_input_ready.wait(dstate_input_index.phase)
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_input_offset), cutlass.Int8)
                b_desc = d_ki0 + decay_op_off
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32)
                for i in cutlass.range_constexpr(bmm_dstate_k_inv_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_k_inv_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_k_inv_desc,
                            a_ptr.subview(i * bmm_dstate_k_inv_desc.sps_B * bmm_dstate_k_inv_desc.tmem_advance_A),
                            b_desc + i * (bmm_dstate_k_inv_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )
                dstate_input_index = advance(dstate_input_index, 1)
                if elect_one:
                    bars.mb_dstate_input_done.arrive(cta_group=1)

            # ---- dU intra += dO^T(S) @ A ---------------------------------------------
            bars.mb_a_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_do_a_desc,
                d_do_trans,
                d_int,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )
            if elect_one:
                bars.mb_du_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate Q-term += dO^T(S) @ Q decay ----------------------------------
            bars.mb_q_decay_ready[decay_stage].wait(decay_phase)
            mma_ss(
                bmm_do_q_decay_desc,
                d_do_trans,
                d_qd_trans,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )
            if elect_one:
                bars.mb_do_mma_done[raw_stage_idx].arrive(cta_group=1)

            # ---- dY = dU(T) @ T^-1 ---------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_phase)
            bars.mb_du_input_ready.wait(du_input_index.phase)
            du_input_index = advance(du_input_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_input_offset), cutlass.Int8)
            b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dy_acc_offset), cutlass.Float32)
            for i in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.sps_B):
                    mma_ts_step(
                        bmm_du_t_inv_trans_desc,
                        a_ptr.subview(i * bmm_du_t_inv_trans_desc.sps_B * bmm_du_t_inv_trans_desc.tmem_advance_A),
                        b_desc + i * (bmm_du_t_inv_trans_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready.arrive(cta_group=1)
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate K-term += -dY(T) @ K decay -----------------------------------
            bars.mb_neg_dy_input_ready.wait(neg_dy_index.phase)
            neg_dy_index = advance(neg_dy_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_neg_dy_input_offset), cutlass.Int8)
            b_desc = d_kd_trans
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32)
            for i in cutlass.range_constexpr(bmm_dy_k_decay_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dy_k_decay_desc.sps_B):
                    mma_ts_step(
                        bmm_dy_k_decay_desc,
                        a_ptr.subview(i * bmm_dy_k_decay_desc.sps_B * bmm_dy_k_decay_desc.tmem_advance_A),
                        b_desc + i * (bmm_dy_k_decay_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready.arrive(cta_group=1)
                bars.mb_decay_done[decay_stage].arrive(cta_group=1)

        # ---- tile end: dstate0 store wait --------------------------------------------
        if sk_nt > 0:
            bars.mb_dstate0_acc_stored.wait(dstate0_index.phase)
            dstate0_index = advance(dstate0_index, 1)
        gbase += sk_nt
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
    q_tx_bytes,
    k_tx_bytes,
    gate_tx_bytes,
    beta_tx_bytes,
    do_tx_bytes,
    sQ_raw,
    sK_raw,
    sGate_raw,
    sDo_raw,
    sBeta_raw,
    desc_q_base,
    desc_k_base,
    desc_gate_base,
    desc_do_base,
    desc_beta_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): every G->S operand load."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)
    tail_count = ((total_tiles - cutlass.Int32(1)) % num_ctas) + cutlass.Int32(1)
    tail_base = (total_tiles - tail_count) if tail_count * 2 >= num_ctas else total_tiles
    tail_row = tail_base + bidx
    tail_row = tail_row if tail_row < total_tiles else cutlass.Int32(1 << 28)

    elect_one = nvvm.elect_sync()
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
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
    gate_granu = cutlass.const_expr(128 // (cfg.gate_dtype.width // 8))
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // gate_granu),
        tma_granu_elems=gate_granu,
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    sDo_tma = SmemTile(
        base=sDo_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sBeta_tma = SmemTile(
        base=sBeta_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        next_tile, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, tile_idx, num_ctas, tail_base, tail_row, elect_one
        )
        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_beta_slot = (desc_beta_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_gate_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_beta_slot)
        sk_nt = compute_end - write_start
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t

            # ---- Q load --------------------------------------------------------------
            bars.mb_q_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_q_ready[raw_index.idx].arrive(n_bytes=q_tx_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, chunk_start)
            tma_load_tile(sQ_tma[raw_index.idx], q_slice, bars.mb_q_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- K load --------------------------------------------------------------
            bars.mb_k_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_k_ready[raw_index.idx].arrive(n_bytes=k_tx_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, bars.mb_k_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Gate load: GMEM -> SMEM ---------------------------------------------
            bars.mb_gate_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_gate_ready[raw_index.idx].arrive(n_bytes=gate_tx_bytes)
            gate_slice = tma_slice_runtime_desc(desc_gate_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, bars.mb_gate_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Beta load: GMEM -> SMEM ---------------------------------------------
            bars.mb_beta_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_beta_ready[raw_index.idx].arrive(n_bytes=beta_tx_bytes)
            beta_slice = tma_slice_runtime_desc(desc_beta_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sBeta_tma[raw_index.idx], beta_slice, bars.mb_beta_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- dO load -------------------------------------------------------------
            bars.mb_do_mma_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_do_ready[raw_index.idx].arrive(n_bytes=do_tx_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sDo_tma[raw_index.idx], do_slice, bars.mb_do_ready[raw_index.idx].smem_ptr, acquire=False)
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx = next_tile
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    if cutlass.const_expr(cfg.log_gate):
        return raw_gate * cutlass.Float32(LOG2_E)
    return cute.math.log2(raw_gate + cutlass.Float32(1e-10), fastmath=True)


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
    scale,
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_raw,
    sGate_load_ptr,
    sK_raw,
    sQ_raw,
    sBeta_raw,
    sK_decay_raw,
    sQ_decay_raw,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG0 warp role (warps 0-3): the gate prefix scan and the decay / restore operands in tcgen05 SMEM for every chunk."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_0,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_0 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    channel_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_0_warp_ids))
    channel_active = cutlass.Boolean(True)
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        channel_active = lane_idx < cutlass.Int32(channel_rows)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    cg0_channel_dim = cg0_warp * cfg.threads_per_warp + lane_idx
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        cg0_channel_dim = cg0_warp * channel_rows + lane_idx % channel_rows
    cg0_a_log_exp = opaque_f32_zero() + cutlass.Float32(1.0)
    cg0_dt_bias_value = opaque_f32_zero()
    gbase = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        sk_nt = compute_end - write_start
        if cutlass.const_expr(cfg.safe_gate and (mA_log is not None or mDt_bias is not None)):
            if sk_nt > 0:
                if cutlass.const_expr(mA_log is not None):
                    cg0_a_log_exp = cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * LOG2_E, fastmath=True)
                if cutlass.const_expr(mDt_bias is not None):
                    cg0_dt_bias_value = mDt_bias[head_idx, cg0_channel_dim].to(cutlass.Float32)
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            gc = gbase + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = gc % cfg.smem_decay_stages
            raw_stage = gc % cfg.smem_raw_stages
            sQ_ptr = sQ_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = sK_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sBetaP_ptr = sBeta_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = sGate_load_ptr + raw_stage * cfg.gate_stage_elems
            sGate_exchange_ptr = sGate_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sDecay_scale_ptr = sDecay_scale_raw.data_ptr() + decay_stage * cfg.d_k

            bars.mb_gate_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_q_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_k_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_beta_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)

            row_group_start = cg0_warp * (cfg.b_t // len(cfg.compute_group_0_warp_ids))
            lane_row_group = lane_idx // 8
            lane_in_row_group = lane_idx - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            gate_prefix_ptr = sGate_exchange_ptr
            channel_dim = cg0_channel_dim

            # ---- Gate prefix scan: cumulative log-gate per key channel ---------------
            f32_segment = channel_dim // 32
            f32_segment_dim = channel_dim - f32_segment * 32
            if cutlass.const_expr(cfg.gate_dtype != cutlass.Float32):
                raw_segment = channel_dim // 64
                raw_seg_base = raw_segment * (cfg.b_t * 64)
                raw_col = channel_dim - raw_segment * 64
            prefix_acc = cutlass.Float32(0.0)
            exp_g_last = cutlass.Float32(0.0)
            if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    prefix_idx0 = f32_segment * (cfg.b_t * 32) + row0 * 32 + swizzle_xor_128b(row0, f32_segment_dim, elem_bytes=4)
                    prefix_idx1 = f32_segment * (cfg.b_t * 32) + row1 * 32 + swizzle_xor_128b(row1, f32_segment_dim, elem_bytes=4)
                    gate0 = (sGate_ptr + prefix_idx0).load()
                    gate1 = (sGate_ptr + prefix_idx1).load()
                    token_idx0 = chunk_idx * cutlass.Int32(cfg.b_t) + row0
                    token_idx1 = chunk_idx * cutlass.Int32(cfg.b_t) + row1
                    if cutlass.const_expr(cfg.safe_gate):
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, cg0_a_log_exp * (gate0 + cg0_dt_bias_value))
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, cg0_a_log_exp * (gate1 + cg0_dt_bias_value))
                        else:
                            gate1 = cutlass.Float32(0.0)
                    else:
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, gate0)
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, gate1)
                        else:
                            gate1 = cutlass.Float32(0.0)
                    pair_vec = nvvm.add_packed_f32x2(
                        cutlass.Vector.from_elements((prefix_acc, gate0), cutlass.Float32),
                        cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32),
                        ftz=False,
                        rnd="rn",
                    )
                    prefix0, row_pair_sum = cutlass.Float32(pair_vec[0]), cutlass.Float32(pair_vec[1])
                    prefix1 = prefix_acc + row_pair_sum
                    exp_g0 = cute.math.exp2(prefix0, fastmath=True)
                    exp_g1 = cute.math.exp2(prefix1, fastmath=True)
                    if channel_active:
                        (
                            sGate_exchange_ptr + f32_segment * (cfg.b_t * 32) + row0 * 32 + swizzle_xor_128b(row0 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                        ).store(exp_g0)
                        (
                            sGate_exchange_ptr + f32_segment * (cfg.b_t * 32) + row1 * 32 + swizzle_xor_128b(row1 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                        ).store(exp_g1)
                    prefix_acc = prefix1
                    exp_g_last = exp_g1
            else:
                gate_conditioned = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
                for row in cutlass.range_constexpr(cfg.b_t):
                    raw_idx = raw_seg_base + swizzle_xor_128b(row, row * 64 + raw_col, elem_bytes=2)
                    gate_row = (sGate_ptr + raw_idx).load().to(cutlass.Float32)
                    token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + row
                    if cutlass.const_expr(cfg.safe_gate):
                        if token_idx < batch_seqlen:
                            gate_row = gate_scale(cfg, cg0_a_log_exp * (gate_row + cg0_dt_bias_value))
                        else:
                            gate_row = cutlass.Float32(0.0)
                    else:
                        if token_idx < batch_seqlen:
                            gate_row = gate_scale(cfg, gate_row)
                        else:
                            gate_row = cutlass.Float32(0.0)
                    gate_conditioned[row] = gate_row
                nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    prefix_idx0 = f32_segment * (cfg.b_t * 32) + row0 * 32 + swizzle_xor_128b(row0 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    prefix_idx1 = f32_segment * (cfg.b_t * 32) + row1 * 32 + swizzle_xor_128b(row1 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    gate0 = gate_conditioned[row0]
                    gate1 = gate_conditioned[row1]
                    pair_vec = nvvm.add_packed_f32x2(
                        cutlass.Vector.from_elements((prefix_acc, gate0), cutlass.Float32),
                        cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32),
                        ftz=False,
                        rnd="rn",
                    )
                    prefix0, row_pair_sum = cutlass.Float32(pair_vec[0]), cutlass.Float32(pair_vec[1])
                    prefix1 = prefix_acc + row_pair_sum
                    exp_g0 = cute.math.exp2(prefix0, fastmath=True)
                    exp_g1 = cute.math.exp2(prefix1, fastmath=True)
                    if channel_active:
                        (sGate_exchange_ptr + prefix_idx0).store(exp_g0)
                        (sGate_exchange_ptr + prefix_idx1).store(exp_g1)
                    prefix_acc = prefix1
                    exp_g_last = exp_g1

            # ---- decay-slot guard ----------------------------------------------------
            operand_done_phase = ((gc // cfg.smem_decay_stages) + 1) % 2
            bars.mb_decay_done[decay_stage].wait(operand_done_phase)

            # ---- decay scale: exp2(g last) per key channel ---------------------------
            if channel_active:
                (sDecay_scale_ptr + channel_dim).store(exp_g_last)
            bars.mb_decay_scale_ready[decay_stage].arrive()

            nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)

            k_inv_pack = cutlass.Array(cutlass.Int32, dk_halves * 4, alignment=16)
            raw_q_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)
            raw_k_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)
            raw_beta_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)

            # ---- optional Q/K L2-norm ------------------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                qk0_lo = opaque_f32_zero()
                qk0_hi = opaque_f32_zero()
                qk1_lo = opaque_f32_zero()
                qk1_hi = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                raw_q_frag = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_beta_frag = (sBetaP_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_q_frag_f32 = raw_q_frag.to(cutlass.Float32)
                raw_k_frag_f32 = raw_k_frag.to(cutlass.Float32)
                raw_beta_frag_f32 = raw_beta_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    q_val = raw_q_frag_f32[dim_offset]
                    k_val = raw_k_frag_f32[dim_offset]
                    if cutlass.const_expr(not cfg.beta_guard):
                        raw_q_regs[reg_base + dim_offset] = q_val
                    raw_k_regs[reg_base + dim_offset] = k_val
                    beta_val = raw_beta_frag_f32[dim_offset]
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        beta_val = (sigmoid(beta_val) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(cfg.io_dtype).to(cutlass.Float32)
                    raw_beta_regs[reg_base + dim_offset] = beta_val
                    if cutlass.const_expr(cfg.l2norm):
                        if cutlass.const_expr(dim_offset % 2 == 0):
                            qk0_lo, qk0_hi = ffma2(q_val, k_val, q_val, k_val, qk0_lo, qk0_hi)
                        else:
                            qk1_lo, qk1_hi = ffma2(q_val, k_val, q_val, k_val, qk1_lo, qk1_hi)

            nvvm.fence_proxy("async.shared", space="cta")
            if cutlass.const_expr(not cfg.beta_guard):
                bars.mb_q_done[raw_stage].arrive()
            bars.mb_k_done[raw_stage].arrive()
            if cutlass.const_expr(not cfg.beta_guard):
                bars.mb_beta_done[raw_stage].arrive()

            q_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            k_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            if cutlass.const_expr(cfg.l2norm):
                q_sum_sq = qk0_lo + qk1_lo
                k_sum_sq = qk0_hi + qk1_hi
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                q_inv_norm = cute.math.rsqrt(cute.math.max(q_sum_sq, norm_floor_sq), fastmath=True)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)
            q_stage_norm = q_inv_norm * scale

            # ---- Beta guard ----------------------------------------------------------
            if cutlass.const_expr(cfg.beta_guard):
                beta_guard(cfg, raw_beta_regs, raw_k_regs, k_inv_norm, gate_prefix_ptr, decay_row, lane_in_row_group)
                for dim_half in cutlass.range_constexpr(dk_halves):
                    dim_base = dim_half * 64 + lane_in_row_group * 8
                    reg_base = dim_half * 8
                    beta_eff_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                    for pair_idx in cutlass.range_constexpr(4):
                        dim0 = pair_idx * 2
                        beta_eff_pack[pair_idx] = fp32_to_fp16(raw_beta_regs[reg_base + dim0], raw_beta_regs[reg_base + dim0 + 1], dtype=cfg.io_dtype)
                    beta_eff_vec = cutlass.Vector.from_elements(
                        (beta_eff_pack[0], beta_eff_pack[1], beta_eff_pack[2], beta_eff_pack[3]),
                        cutlass.Int32,
                    ).bitcast(cfg.io_dtype)
                    f16_segment = dim_base // 64
                    f16_segment_dim = dim_base - f16_segment * 64
                    raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                    (sBetaP_ptr + raw_f16_idx).store(beta_eff_vec, alignment=16)

            # ---- decay/restore operands: exp2(+-g) applied per key channel -----------
            exp_g_regs = cutlass.Array(cutlass.Float32, dk_halves * 8, alignment=16)
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    g_prefix_idx = f32_segment * (cfg.b_t * 32) + decay_row * 32 + swizzle_xor_128b(decay_row ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    exp_g_frag = (gate_prefix_ptr + g_prefix_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    exp_g_regs[f32_reg_base] = exp_g_frag[0]
                    exp_g_regs[f32_reg_base + 1] = exp_g_frag[1]
                    exp_g_regs[f32_reg_base + 2] = exp_g_frag[2]
                    exp_g_regs[f32_reg_base + 3] = exp_g_frag[3]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_gate_done[raw_stage].arrive()
            if cutlass.const_expr(cfg.beta_guard):
                bars.mb_beta_done[raw_stage].arrive()

            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8

                # ---- K decay + K inv operands: Beta * K * exp2(+g), K * exp2(-g) -----
                k_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
                    k_beta0, k_beta1 = fmul2(k_value0, k_value1, raw_beta_regs[raw_reg_idx0], raw_beta_regs[raw_reg_idx1])
                    k_decay0, k_decay1 = fmul2(k_beta0, k_beta1, exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1])
                    k_decay_pack[pair_idx] = fp32_to_fp16(k_decay0, k_decay1, dtype=cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    k_inv0, k_inv1 = fmul2(k_value0, k_value1, exp_neg_g0, exp_neg_g1)
                    k_inv_pack[dim_half * 4 + pair_idx] = fp32_to_fp16(k_inv0, k_inv1, dtype=cfg.io_dtype)

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
                    (k_decay_pack[0], k_decay_pack[1], k_decay_pack[2], k_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sK_inv_ptr + op_idx).store(k_inv_vec, alignment=16)
                (sK_decay_ptr + op_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_inv_ready[decay_stage].arrive()

            # ---- Q decay operand -----------------------------------------------------
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                if cutlass.const_expr(cfg.beta_guard):
                    f16_segment = dim_base // 64
                    f16_segment_dim = dim_base - f16_segment * 64
                    raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                    raw_q_frag = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
                    raw_q_frag_f32 = raw_q_frag.to(cutlass.Float32)
                    for dim_offset in cutlass.range_constexpr(8):
                        raw_q_regs[reg_base + dim_offset] = raw_q_frag_f32[dim_offset]
                q_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_stage_norm, q_stage_norm)
                    q_decay0, q_decay1 = fmul2(q_value0, q_value1, exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1])
                    q_decay_pack[pair_idx] = fp32_to_fp16(q_decay0, q_decay1, dtype=cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (q_decay_pack[0], q_decay_pack[1], q_decay_pack[2], q_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sQ_decay_ptr + op_idx).store(q_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_decay_ready[decay_stage].arrive()
            if cutlass.const_expr(cfg.beta_guard):
                bars.mb_q_done[raw_stage].arrive()
        gbase += sk_nt
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
    tmem_base_slot,
    warp_idx,
    mDstate0,
    mDstate_in,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG1 warp role (warps 4-7): the value-side TMEM staging (dstate seed, dU / -dY restages, dstate capture, dstate0 store)."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_sp = warp_idx % len(cfg.compute_group_1_warp_ids)
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    if cutlass.const_expr(cfg.d_v == 128):
        state_gmem_row = tmem_sp * cfg.threads_per_warp + lane_idx
        state_row_valid = cutlass.Boolean(True)
    else:
        state_gmem_row = tmem_sp * 16 + lane_idx % 16
        state_row_valid = lane_idx < 16

    du_acc_index = PipelineState.start(phase=0)
    dy_acc_index = PipelineState.start(phase=0)
    dstate_ready_index = PipelineState.start(phase=0)
    dstate_input_done_index = PipelineState.start(phase=1)

    gbase = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        sk_nt = compute_end - write_start

        # ---- dstate seed: GMEM -> TMEM -----------------------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if sk_nt > 0:
                seed_true = compute_end == batch_num_chunks
                bars.mb_dstate_input_done.wait(dstate_input_done_index.phase)
                dstate_input_done_index = advance(dstate_input_done_index, 1)
                seed_stage = gbase % cfg.smem_decay_stages
                seed_phase = (gbase // cfg.smem_decay_stages) % 2
                bars.mb_decay_scale_ready[seed_stage].wait(seed_phase)
                seed_scale_ptr = sDecay_scale_raw.data_ptr() + seed_stage * cfg.d_k
                row_lo_addr = tmem_row << 16
                seed_vw = 16 // (mDstate_in.element_type.width // 8)
                dstate_src = (mDstate_in.iterator + mDstate_in.layout((batch_idx, head_idx, state_gmem_row, 0))).raw_ptr()
                for i in cutlass.range(cfg.d_k // 16, unroll=1):
                    seed_block = cutlass.Array(cutlass.Float32, 16, alignment=16)
                    for g in cutlass.range_constexpr(16 // seed_vw):
                        seed_chunk = (dstate_src + i * 16 + g * seed_vw).load(count=seed_vw, alignment=16)
                        for t in cutlass.range_constexpr(seed_vw):
                            dval = seed_chunk[t].to(cutlass.Float32)
                            seed_block[g * seed_vw + t] = dval if seed_true else cutlass.Float32(0.0)

                    # ---- seed * exp2(g last) of the first processed chunk ------------
                    for group in cutlass.range_constexpr(4):
                        seed_frag = (seed_scale_ptr + i * 16 + group * 4).load(count=4, alignment=16)
                        for pair in cutlass.range_constexpr(2):
                            seed_block[group * 4 + 2 * pair], seed_block[group * 4 + 2 * pair + 1] = fmul2(
                                seed_block[group * 4 + 2 * pair], seed_block[group * 4 + 2 * pair + 1], seed_frag[2 * pair], seed_frag[2 * pair + 1]
                            )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 16), cutlass.Float32),
                        seed_block[0:16],
                    )
                    seed_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                    for pc in cutlass.range_constexpr(8):
                        seed_pack[pc] = fp32_to_fp16(seed_block[2 * pc], seed_block[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 8), cutlass.Int8),
                        seed_pack[0:8],
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready.arrive()

        for rev_idx in cutlass.range(sk_nt, unroll=1):
            gc = gbase + rev_idx
            row_lo_addr = tmem_row << 16
            row_hi_addr = (tmem_row + 16) << 16

            # ---- dU stage: dU acc -> TMEM f16 ----------------------------------------
            bars.mb_du_acc_ready.wait(du_acc_index.phase)
            du_acc_index = advance(du_acc_index, 1)
            du_col_id = tmem_col + cfg.tmem_du_acc_offset
            du_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + du_col_id, cutlass.Float32), num=2)
            if cutlass.const_expr(dv_halves == 2):
                du_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + du_col_id, cutlass.Float32), num=2)

            du_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            du_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                du_pack_lo[reg_idx] = fp32_to_fp16(du_vec_lo[frag_pair], du_vec_lo[frag_pair + 1], dtype=cfg.io_dtype)
                if cutlass.const_expr(dv_halves == 2):
                    du_pack_hi[reg_idx] = fp32_to_fp16(du_vec_hi[frag_pair], du_vec_hi[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_du_input_offset), cutlass.Int8), du_pack_lo[0:4])
            if cutlass.const_expr(dv_halves == 2):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + (tmem_col + cfg.tmem_du_input_offset), cutlass.Int8), du_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_du_input_ready.arrive()

            # ---- dY read -------------------------------------------------------------
            bars.mb_dy_acc_ready.wait(dy_acc_index.phase)
            dy_acc_index = advance(dy_acc_index, 1)
            dy_col_id = tmem_col + cfg.tmem_dy_acc_offset
            dy_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + dy_col_id, cutlass.Float32), num=2)
            if cutlass.const_expr(dv_halves == 2):
                dy_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + dy_col_id, cutlass.Float32), num=2)

            # ---- -dY -> TMEM ---------------------------------------------------------
            neg_dy_regs_lo = cutlass.Array(cutlass.Float32, 8, alignment=16)
            neg_dy_regs_hi = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for e in cutlass.range_constexpr(8):
                neg_dy_regs_lo[e] = -dy_vec_lo[e]
                if cutlass.const_expr(dv_halves == 2):
                    neg_dy_regs_hi[e] = -dy_vec_hi[e]
            neg_dy_pack_lo = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            neg_dy_pack_hi = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                neg_dy_pack_lo[reg_idx] = fp32_to_fp16(neg_dy_regs_lo[frag_pair], neg_dy_regs_lo[frag_pair + 1], dtype=cfg.io_dtype)
                if cutlass.const_expr(dv_halves == 2):
                    neg_dy_pack_hi[reg_idx] = fp32_to_fp16(neg_dy_regs_hi[frag_pair], neg_dy_regs_hi[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_neg_dy_input_offset), cutlass.Int8), neg_dy_pack_lo[0:4])
            if cutlass.const_expr(dv_halves == 2):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + (tmem_col + cfg.tmem_neg_dy_input_offset), cutlass.Int8), neg_dy_pack_hi[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_neg_dy_input_ready.arrive()

            # ---- dstate capture for the next chunk -----------------------------------
            bars.mb_dstate_acc_ready.wait(dstate_ready_index.phase)
            dstate_ready_index = advance(dstate_ready_index, 1)
            if rev_idx + cutlass.Int32(1) < sk_nt:
                bars.mb_dstate_input_done.wait(dstate_input_done_index.phase)
                dstate_input_done_index = advance(dstate_input_done_index, 1)
                next_serial = gc + cutlass.Int32(1)
                scale_stage = next_serial % cfg.smem_decay_stages
                scale_phase = (next_serial // cfg.smem_decay_stages) % 2
                bars.mb_decay_scale_ready[scale_stage].wait(scale_phase)
                next_scale_ptr = sDecay_scale_raw.data_ptr() + scale_stage * cfg.d_k
                row_lo_addr = tmem_row << 16
                for i in cutlass.range(cfg.d_k // 32, unroll=1):
                    dstate_vec = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32), cutlass.Float32), num=32
                    )

                    # ---- decayed carry: dstate * exp2(g last) of the next chunk ------
                    dstate_decayed = cutlass.Array(cutlass.Float32, 32, alignment=16)
                    for group in cutlass.range_constexpr(8):
                        decay_frag = (next_scale_ptr + i * 32 + group * 4).load(count=4, alignment=16)
                        for pair in cutlass.range_constexpr(2):
                            dstate_decayed[group * 4 + 2 * pair], dstate_decayed[group * 4 + 2 * pair + 1] = fmul2(
                                dstate_vec[group * 4 + 2 * pair], dstate_vec[group * 4 + 2 * pair + 1], decay_frag[2 * pair], decay_frag[2 * pair + 1]
                            )
                    dstate_pack = cutlass.Array(cutlass.Int32, 16, alignment=16)
                    for pc in cutlass.range_constexpr(16):
                        dstate_pack[pc] = fp32_to_fp16(dstate_decayed[2 * pc], dstate_decayed[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 16), cutlass.Int8),
                        dstate_pack[0:16],
                    )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32), cutlass.Float32),
                        dstate_decayed[0:32],
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready.arrive()

        # ---- tile end: dstate0 store / zero-length pass-through ----------------------
        if cutlass.const_expr(mDstate0 is not None):
            if sk_nt > 0:
                if write_start == 0:
                    row_lo_addr = tmem_row << 16
                    dstate0_vw = 16 // (mDstate0.element_type.width // 8)
                    dstate0_dst = (mDstate0.iterator + mDstate0.layout((batch_idx, head_idx, state_gmem_row, 0))).raw_ptr()
                    for i in cutlass.range_constexpr(cfg.d_k // 32):
                        dstate0_vec = nvvm.tcgen05_ld(
                            "32x32b", nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32), cutlass.Float32), num=32
                        )
                        for g in cutlass.range_constexpr(32 // dstate0_vw):
                            if state_row_valid:
                                (dstate0_dst + i * 32 + g * dstate0_vw).store(
                                    cutlass.Vector.from_elements(
                                        tuple(dstate0_vec[g * dstate0_vw + t].to(mDstate0.element_type) for t in range(dstate0_vw)),
                                        mDstate0.element_type,
                                    ),
                                    alignment=16,
                                )
            else:
                for key_dim_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for kk_i in cutlass.range_constexpr(32):
                        kd = key_dim_base + kk_i
                        if state_row_valid:
                            if cutlass.const_expr(cfg.use_dstate_in):
                                mDstate0[batch_idx, head_idx, state_gmem_row, kd] = mDstate_in[batch_idx, head_idx, state_gmem_row, kd]
                            else:
                                mDstate0[batch_idx, head_idx, state_gmem_row, kd] = cutlass.Float32(0.0).to(mDstate0.element_type)
        if sk_nt > 0:
            bars.mb_dstate0_acc_stored.arrive()
        gbase += sk_nt
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_gate,
    base_do,
    base_beta,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    n_batch: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array, run inside the prologue kernel after its order pass;
    warps past the array count fall through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub2 = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub3 = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub4 = cute.make_tensor(desc_workspace.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        emit_seq_descs(base_gate, sub2, cu_seqlens, gate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        emit_seq_descs(base_do, sub3, cu_seqlens, do, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        emit_seq_descs(base_beta, sub4, cu_seqlens, beta, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_gdn2_bprop_summary_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_do: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_beta: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
) -> None:
    """Two-CTA prologue: under ``run_order`` (this kernel is the table's first consumer) block 0 LPT-orders the work-item
    table and zeroes both consumers' scheduler rings via :func:`order_body`; block 1 builds the per-batch TMA-descriptor
    arrays via :func:`build_descs_body`, one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        if cutlass.const_expr(run_order):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            n_heads_out = cutlass.Int32(gate.shape[1])
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
            base_gate,
            base_do,
            base_beta,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            gate,
            do,
            beta,
            n_batch,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when ``run_order``) and build the 5 per-(batch, head) TMA-descriptor
    arrays into ``tensormap_workspace``."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = do.shape[2]
    bpe = io_dtype.width // 8
    granule = 128 // bpe
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0])))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])))
    do_headed = cute.make_tensor(do.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, do.stride[1], do.stride[0])))
    beta_headed = cute.make_tensor(beta.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, beta.stride[1], beta.stride[0])))

    swz = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(q_headed, box_dims=(granule, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(granule, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    gate_granu_elems = 128 // (gate.element_type.width // 8)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(gate_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_do = cuda.create_tensor_map_tiled_from_view(do_headed, box_dims=(granule, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_beta = cuda.create_tensor_map_tiled_from_view(beta_headed, box_dims=(granule, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)

    frost_gdn2_bprop_summary_prologue(
        run_order,
        order_gen,
        b_t,
        base_q,
        base_k,
        base_gate,
        base_do,
        base_beta,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        gate,
        do,
        beta,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    d_initial_state: cute.Tensor | None,
    d_final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    scale: cutlass.Float32,
    stream,
) -> None:
    num_sequences = cu_seqlens.shape[0] - 1

    # ---- launch ----------------------------------------------------------------------
    n_desc = num_sequences
    grid_shape = (cfg.max_active_clusters, 1, 1)
    frost_gdn2_bprop_summary(
        cfg,
        tensormap_workspace,
        n_desc,
        cu_seqlens,
        a_log,
        dt_bias,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        scheduler_counter,
        scale,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_gdn2_bprop_summary(
    cfg: cutlass.Constexpr,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mDstate0: cute.Tensor | None,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    scale: cutlass.Float32,
) -> None:
    """BT=16 GDN-2 backward state-summary persistent kernel body (16 warps; warps 8-11 idle past the entry sync)."""
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
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_beta_base = desc_base_words + cutlass.Int32(4) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128

    sK_decay_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_inv_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_decay_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sIntermediate_raw = cutlass.Array(cfg.io_dtype, cfg.intermediate_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDo_raw = cutlass.Array(cfg.io_dtype, cfg.raw_v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sBeta_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    # sub-bank fill: high group starts at the 128KB midpoint
    smem_bank_fill = cutlass.Array(cfg.io_dtype, cfg.bank_fill_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sGate_raw = cutlass.Array(cutlass.Float32, cfg.raw_gate_cosize, space=SMEM, alignment=1024)
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sGate_load_ptr = sGate_raw.data_ptr()
    else:
        sGate_load_ptr = cute.make_ptr(cfg.gate_dtype, sGate_raw.data_ptr().toint(), mem_space=SMEM, assumed_align=1024)
    sDecay_scale_raw = cutlass.Array(cutlass.Float32, cfg.decay_scale_cosize, space=SMEM, alignment=16)

    sK_inv = SmemTile(
        base=sK_inv_raw.data_ptr().toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_trans = SmemTile(
        base=sDo_raw.data_ptr().toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bpe,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )

    sQ_decay_trans = SmemTile(
        base=sQ_decay_raw.data_ptr().toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_decay_trans = SmemTile(
        base=sK_decay_raw.data_ptr().toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sIntermediate = SmemTile(
        base=sIntermediate_raw,
        elems_per_stage=(cfg.intermediate_tiles * cfg.b_t * cfg.b_t),
        stages=cfg.smem_intermediate_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * cfg.b_t * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    q_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * bpe)
    k_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * bpe)
    gate_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * (cfg.gate_dtype.width // 8))
    beta_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * bpe)
    do_tx_bytes = cutlass.const_expr(cfg.d_v * cfg.b_t * bpe)

    elect_one = nvvm.elect_sync()

    # ---- mbarrier init (one lane per owning role) ------------------------------------
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_q_ready[stage].init()
                bars.mb_q_done[stage].init()
                bars.mb_k_ready[stage].init()
                bars.mb_k_done[stage].init()
                bars.mb_gate_ready[stage].init()
                bars.mb_gate_done[stage].init()
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
                bars.mb_do_ready[stage].init()
                bars.mb_do_mma_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_du_acc_ready.init()
            bars.mb_du_input_ready.init()
            bars.mb_dy_acc_ready.init()
            bars.mb_neg_dy_input_ready.init()
            bars.mb_dstate_acc_ready.init()
            bars.mb_dstate_input_ready.init()
            bars.mb_dstate_input_done.init()
            bars.mb_dstate0_acc_stored.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.super_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_ready[stage].init()
                bars.mb_q_decay_ready[stage].init()
                bars.mb_decay_done[stage].init()
                bars.mb_decay_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_a_ready[stage].init()
                bars.mb_a_done[stage].init()
                bars.mb_t_inv_done[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
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
            q_tx_bytes,
            k_tx_bytes,
            gate_tx_bytes,
            beta_tx_bytes,
            do_tx_bytes,
            sQ_raw,
            sK_raw,
            sGate_raw,
            sDo_raw,
            sBeta_raw,
            desc_q_base,
            desc_k_base,
            desc_gate_base,
            desc_do_base,
            desc_beta_base,
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
            sK_decay_raw,
            sK_inv_raw,
            sIntermediate_raw,
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
            tmem_base_slot,
            sK_inv,
            sDo_trans,
            sQ_decay_trans,
            sK_decay_trans,
            sIntermediate,
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
            lane_idx,
            sK_inv_raw,
            sQ_decay_raw,
            sIntermediate_raw,
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
            scale,
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_raw,
            sGate_load_ptr,
            sK_raw,
            sQ_raw,
            sBeta_raw,
            sK_decay_raw,
            sQ_decay_raw,
            sDecay_scale_raw,
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
            tmem_base_slot,
            warp_idx,
            mDstate0,
            mDstate_in,
            sDecay_scale_raw,
            bars,
        )


@dataclass
class Gdn2BpropSummaryCfg:
    """Kernel cfg: fixed BT=16 schedule constants; ``build_cfg`` stamps the derived TMEM column offsets and SMEM cosizes."""

    io_dtype: Type[cutlass.Numeric]
    gate_dtype: Type[cutlass.Numeric]
    use_dstate_in: bool
    use_dstate0: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    allow_neg_eigval: bool
    beta_guard: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    d_k: int
    d_v: int
    log_gate: bool = True
    scheduler_stages: int = 8

    # ---- fixed constants stamped from CFG at build time ------------------------------
    compute_group_0_warp_ids: tuple = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple = CFG.COMPUTE_GROUP_1_WARP_IDS
    super_mma_warp_id: int = CFG.SUPER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    cg0_sync_barrier_id: int = 1
    cg0_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    tmem_user_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    intermediate_tiles: int = 5
    tmem_dstate_acc_offset: int = 0
    tmem_dstate_input_offset: int = 0
    tmem_du_acc_offset: int = 0
    tmem_dy_acc_offset: int = 0
    tmem_du_input_offset: int = 0
    tmem_neg_dy_input_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- buffer cosizes stamped at build time ----------------------------------------
    raw_qk_cosize: int = 0
    raw_v_cosize: int = 0
    raw_gate_cosize: int = 0
    gate_stage_elems: int = 0
    operand_cosize: int = 0
    bank_fill_cosize: int = 0
    decay_scale_cosize: int = 0
    intermediate_cosize: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    gate_dtype: Type[cutlass.Numeric],
    *,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    beta_guard: bool = False,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
    log_gate: bool = True,
) -> Gdn2BpropSummaryCfg:
    cfg = Gdn2BpropSummaryCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        l2norm=l2norm,
        safe_gate=safe_gate,
        log_gate=log_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        beta_guard=beta_guard,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
    )
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.cg0_threads = len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp
    cfg.tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)) * cfg.threads_per_warp

    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dstate_input_offset = cfg.d_k
    cfg.tmem_du_acc_offset = cfg.tmem_dstate_input_offset + cfg.d_k // 2
    cfg.tmem_dy_acc_offset = cfg.tmem_du_acc_offset + cfg.b_t
    cfg.tmem_du_input_offset = cfg.tmem_dy_acc_offset + cfg.b_t
    cfg.tmem_neg_dy_input_offset = cfg.tmem_du_input_offset + cfg.b_t // 2
    assert cfg.tmem_neg_dy_input_offset + cfg.b_t // 2 <= 512

    cfg.raw_qk_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.raw_v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.raw_gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.gate_stage_elems = (cfg.d_k * cfg.b_t) * (4 // (cfg.gate_dtype.width // 8))
    cfg.operand_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k

    # ---- sub-bank fill: 1024 B plus two operand rings keeps the high group at 128KB ----
    cfg.bank_fill_cosize = 1024 // (io_dtype.width // 8) + 2 * cfg.operand_cosize
    cfg.decay_scale_cosize = cfg.smem_decay_stages * cfg.d_k
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * cfg.intermediate_tiles * cfg.b_t * cfg.b_t
    return cfg


TENSORMAP_DESC_ARRAYS = 5  # per-batch runtime TMA descriptors: Q, K, Gate, dO, Beta


# ---------------------------------------------------------------------------


@functools.cache
def get_compiled_cache(
    io_dtype_str: str,
    dstate_in_dtype_str: str,
    dstate0_dtype_str: str,
    gate_dtype_str: str,
    a_log_dtype_str: str,
    dt_bias_dtype_str: str,
    cu_dtype_str: str,
    device: int,
    num_sm: int,
    HQ: int,
    HK: int,
    HV: int,
    DK: int,
    DV: int,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    beta_guard: bool,
    order_in_prologue: bool,
    order_gen: bool,
    log_gate: bool = True,
):
    return {}


def compile(
    io_dtype,
    gate_dtype,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    log_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    beta_guard: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    *,
    d_k: int,
    d_v: int,
    num_sm: int,
    a_log_cute,
    dt_bias_cute,
    cu_seqlens_cute,
    dstate0_cute,
    dstate_in_cute,
    work_items_cute,
    work_count_cute,
    scheduler_counter_cute,
    tensormap_workspace_cute,
    scale,
    stream,
):
    """JIT-compile the chunked GDN-2 bprop summary kernel for one static config."""
    cfg = build_cfg(
        io_dtype,
        gate_dtype,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        l2norm=l2norm,
        safe_gate=safe_gate,
        log_gate=log_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        beta_guard=beta_guard,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=num_sm,
        d_k=d_k,
        d_v=d_v,
    )

    return cute.compile(
        host,
        cfg,
        a_log_cute,
        dt_bias_cute,
        cu_seqlens_cute,
        dstate0_cute,
        dstate_in_cute,
        work_items_cute,
        work_count_cute,
        scheduler_counter_cute,
        tensormap_workspace_cute,
        scale,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_gdn2_bwd_summary_sm100(
    q,
    k,
    gate,
    beta,
    do,
    cu_seqlens,
    scale: float,
    *,
    d_initial_state,
    d_final_state=None,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    log_gate: bool = True,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    beta_guard: bool = False,
    work_items=None,
    work_count=None,
    scheduler_counter=None,
    scheduler_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    tensormap_workspace,
    device: int,
    num_sm: int,
    stream,
    own_prologue: bool = True,
) -> None:
    """Blackwell BT=16 chunked GDN-2 backward state summary: the reverse state-gradient recurrence alone, writing
    ``d_initial_state`` = dL/dS0 (``(num_seqs, HO, DV, DK)``; the gradient-state pass G when ``d_final_state`` is None, else the full
    dh0; both state gradients share one dtype).  All tensors contiguous on one device; HO = max(HQ, do heads).  gate: natural-log
    decay (``log_gate``), alpha floored at 1e-10 (``log_gate=False``), or logits under safe_gate = ``lower_bound * sigmoid(exp(a_log) *
    (gate + dt_bias))``; use_beta_sigmoid: ``beta`` holds logits; use_qk_l2norm_in_kernel: q / k arrive raw; work_items / work_count:
    REQUIRED split-K table, each item runs chunks ``[write_start, compute_end)`` backward, only the item owning the sequence tail seeds
    ``d_final_state`` and only the item with ``write_start == 0`` stores ``d_initial_state``; scheduler_counter: zeroed scratch enabling
    the work-stealing scheduler; tensormap_workspace: ``tensormap_workspace_bytes(module, B)`` bytes, 128-byte aligned.
    """
    HQ = q.shape[1]
    HK = k.shape[1]
    HV = do.shape[1]
    HO = max(HQ, HV)
    DK = q.shape[2]
    DV = do.shape[2]
    use_dstate_in = d_final_state is not None
    use_dstate0 = d_initial_state is not None
    if not use_dstate0:
        raise ValueError("chunk_gdn2_bwd_summary_sm100 requires d_initial_state (its single output)")
    if scheduler_counter is None:
        raise ValueError("scheduler_counter is required")
    order_gen = work_item_scratch is None
    if order_in_prologue and scheduler_all is None:
        raise ValueError("order_in_prologue requires scheduler_all (the prologue zeroes both consumers' scheduler rings)")
    B = cu_seqlens.shape[0] - 1
    gate_scale_log2 = gate_lower_bound * LOG2_E

    if not safe_gate:
        a_log = None
        dt_bias = None

    cu_stream = cuda_driver.CUstream(int(stream))
    cache = get_compiled_cache(
        str(q.dtype),
        str(d_final_state.dtype) if d_final_state is not None else "none",
        str(d_initial_state.dtype) if d_initial_state is not None else "none",
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        device,
        num_sm,
        HQ,
        HK,
        HV,
        DK,
        DV,
        use_dstate_in,
        use_dstate0,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid,
        allow_neg_eigval,
        beta_guard,
        order_in_prologue,
        order_gen,
        log_gate=log_gate,
    )

    if "compiled" not in cache:
        io_dtype = get_dtype(q.dtype)
        gate_dtype = get_dtype(gate.dtype)

        dstate0_cute = None
        if use_dstate0:
            dstate0_cute = from_dlpack(d_initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate_in_cute = None
        if use_dstate_in:
            dstate_in_cute = from_dlpack(d_final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        work_items_cute = from_dlpack(work_items, assumed_align=16)
        work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        scheduler_counter_cute = from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic()

        tensormap_workspace_cute = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()

        a_log_cute = from_dlpack(a_log, assumed_align=4) if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16) if dt_bias is not None else None
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        cache["compiled"] = compile(
            io_dtype,
            gate_dtype,
            use_dstate_in=use_dstate_in,
            use_dstate0=use_dstate0,
            l2norm=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            log_gate=log_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            beta_guard=beta_guard,
            q_ratio=HO // HQ,
            k_ratio=HO // HK,
            v_ratio=HO // HV,
            n_heads_out=HO,
            d_k=DK,
            d_v=DV,
            num_sm=num_sm,
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            dstate0_cute=dstate0_cute,
            dstate_in_cute=dstate_in_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            scheduler_counter_cute=scheduler_counter_cute,
            tensormap_workspace_cute=tensormap_workspace_cute,
            scale=scale,
            stream=cu_stream,
        )

    if own_prologue and "prologue" not in cache:
        io_dtype = get_dtype(q.dtype)
        q_placeholder = from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        k_placeholder = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        gate_placeholder = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        do_placeholder = from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        beta_placeholder = from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        cu_placeholder = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        workspace_placeholder = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()
        staging_placeholder = None
        if not order_gen:
            staging_placeholder = from_dlpack(work_item_scratch, assumed_align=16)
            staging_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_placeholder = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        cache["prologue_scheduler_all"] = order_in_prologue
        scheduler_all_placeholder = None
        if order_in_prologue:
            scheduler_all_placeholder = from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic()
        cache["prologue"] = cute.compile(
            prologue,
            io_dtype,
            CFG.B_T,
            order_in_prologue,
            order_gen,
            q_placeholder,
            k_placeholder,
            gate_placeholder,
            do_placeholder,
            beta_placeholder,
            cu_placeholder,
            staging_placeholder,
            work_count_placeholder,
            work_items_placeholder,
            scheduler_all_placeholder,
            workspace_placeholder,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    if own_prologue:
        cache["prologue"](
            q,
            k,
            gate,
            do,
            beta,
            cu_seqlens,
            work_item_scratch if not order_gen else None,
            work_count,
            work_items,
            scheduler_all if cache["prologue_scheduler_all"] else None,
            tensormap_workspace,
            cu_stream,
        )
    cache["compiled"](
        a_log,
        dt_bias,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        scale,
        cu_stream,
    )
    return cache


def run_bwd_summary(
    cache,
    q,
    k,
    gate,
    beta,
    do,
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
    own_prologue=True,
) -> None:
    """Replay the compiled plan: the prologue launch, then the main launch.  The plan validated the contract at build, so
    nothing here raises."""
    cu_stream = cuda_driver.CUstream(int(stream))
    if own_prologue:
        cache["prologue"](
            q,
            k,
            gate,
            do,
            beta,
            cu_seqlens,
            work_item_scratch,
            work_count,
            work_items,
            scheduler_all if cache["prologue_scheduler_all"] else None,
            tensormap_workspace,
            cu_stream,
        )
    cache["compiled"](
        a_log,
        dt_bias,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        scale,
        cu_stream,
    )


frost_gdn2_bprop_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn2_bprop_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
