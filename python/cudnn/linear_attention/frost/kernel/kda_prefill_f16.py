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

"""Chunked Kimi Delta Attention (KDA) prefill kernel for Blackwell SM100
(Cutlass DSL), BT=16 tiling with a per-key-channel decay.  Framework-neutral
entry ``chunk_kda_sm100``.

Persistent kernel: the grid is the SM count and every warp role
runs a tile-scheduler loop (``decode_work_item``); a tile is one (batch,
head) sequence, or one split-K work item computing chunks ``[cstart, wend)``
and writing O / checkpoints only for the owned ``[wstart, wend)`` (see
``common/split_k.py``; warmup chunks rebuild the incoming state from
zero).  All ring stage/phase bookkeeping runs on cumulative per-CTA chunk
counters so pipelines flow seamlessly across tiles.

Pipeline (direct CUTLASS primitives, chunk_idx-size 16 KDA schedule):

  load q/k/v/gate/beta
  optional in-kernel L2-norm of q/k (L2NORM specialization)
  exp2(g), exp2(-g), stage final-token exp2(g) as exp2(g_last)
  super-MMA: kk/qk/Neumann inverse + apply beta
  tcgen05-MMA: state*k/state*q/new-v/kv-update/qkv
  store o, periodic state checkpoints, final state

ABI: q `[T, HQ, DK]`, k `[T, HK, DK]`, v `[T, HV, DV]`, gate
`[T, HO, DK]` fp32 (natural-log decay unless SAFE_GATE, which applies the
safe-gate transform from raw gate + a_log/dt_bias), beta `[T, HO]` fp32
post-sigmoid, cu_seqlens int32, states/checkpoints `[N, HO, DV, DK]` (VK).
GQA/GVA head broadcast follows repeat_interleave: source head =
head_idx // (HO // H_x).  State presence, L2NORM, SAFE_GATE, checkpoints, and the
head ratios are compile-time specializations.

Warp assignments (16 warps = 512 threads):
  warps 0-7  : compute group 0 - gate prefix scan + decay/restore operands
  warps 8-11 : compute group 1 - TMEM value side, o drain, state stores
  warp  12   : super-MMA       - register-MMA kk^T + Neumann inverse
  warp  13   : tcgen05-MMA     - the six state GEMMs + the TMEM lifecycle
  warp  14   : TMA load        - per-chunk input G->S loads
  warp  15   : epilogue        - register-MMA qk + the O TMA store

SMEM layout (~221 KB total):
  Buffer                    Bytes  Stages
  q / k / v raw             32768  8       <-- SW128 TMA ring (io dtype)
  gate raw                  65536  8       <-- fp32 prefix-scan source
  beta                        512  8       <-- fp32 per-token scalars
  dt_bias (+a_log slot)       516  1       <-- SAFE_GATE only
  K_inv                      8192  2       <-- token-major ldmatrix/tcgen05 B operand
  K decay / Q decay       2x 8192  2       <-- tcgen05 SW128 K-box-major A/B operands
  K restore                  8192  2       <-- tcgen05 B operand for the state update
  state-scale diag          12288  3       <-- per-k-atom decay diagonal blocks
  pairwise (A_inv / qk)      2048  2       <-- SW32 16x16 register-MMA tiles
  o staging                  8192  2       <-- W128 output drain

TMEM layout (272 of 512 columns):
  Buffer          Cols     Purpose
  state           0-127    S[DK,DV] fp32 recurrent state
  state inp       128-191  packed b16 A operand view of the state
  q_state_acc      192-223  2-stage state*q -> o accumulator
  state_k_acc     224-239  state*k fp32 accumulator
  update_acc      240-255  update fp32 accumulator
  rhs input       256-263  packed b16 A operand: beta * (v - state*k)
  update input    264-271  packed b16 A operand: the update readback

GEMM schedule (tcgen05-MMA warp, in issue order per chunk):
  state*k -> state_k_acc
  state*q -> q_state_acc (the o acc)
  state decay (diag blocks)
  update = A_inv @ rhs -> update_acc
  final_state += update @ k_restore
  o += qk @ update -> q_state_acc

Requires a cutlass DSL build providing `cutlass.experimental.*`; not
available in the pip nvidia-cutlass-dsl releases.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.split_k import decode_work_item
from ..common.thd import TENSOR_MAP_QWORDS, build_h_descs_kernel, build_qkv_load_descs_kernel
from .kda_prefill_config import CFG
from cudnn.frost.tile_dsl.barrier import (
    advance,
    arrive,
    MBarrier,
    PipelineState,
    Producer,
    wait,
)
from cudnn.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_step, mma_ts
from cudnn.frost.tile_dsl.swizzle import swizzle_lin_128b, swizzle_lin_S, swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.pointwise import (
    opaque_f32_zero,
    fadd2,
    fmul2,
    movmatrix_16b,
    mul_f16x2,
    fp32_to_fp16,
    sub_f16x2,
)

LOG2_E: float = 1.4426950408889634


DEFAULT_GATE_LOWER_BOUND: float = -5.0


# Host-side API defaults.


L2_NORM_EPS: float = 1.0e-12


class KdaBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring (mirrors GDN's
    ``GdnBars``).  Consumers track ``(idx, phase)`` inline; the producer tag selects
    the arrive lowering (``TMA_LOAD``/``MMA_COMMIT``/``THREAD``)."""

    mb_tma_done: MBarrier
    mb_inputs_ready: MBarrier
    mb_inputs_done: MBarrier
    mb_o_acc_ready: MBarrier
    mb_o_acc_done: MBarrier
    mb_state_k_acc_ready: MBarrier
    mb_update_acc_ready: MBarrier
    mb_state_inp_ready: MBarrier
    mb_state_scale_diag_done: MBarrier
    mb_kk_qk_super_mma_done: MBarrier
    mb_kk_qk_mma_done: MBarrier
    mb_k_restore_done: MBarrier
    mb_rhs_ready: MBarrier
    mb_update_ready: MBarrier
    mb_final_state_stored: MBarrier
    mb_a_ready: MBarrier
    mb_qk_acc_ready: MBarrier
    mb_a_done: MBarrier
    mb_qk_scale_ready: MBarrier
    mb_k_decay_cg0_ready: MBarrier
    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier
    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier
    mb_h_tmastg_ready: MBarrier
    mb_h_tmastg_done: MBarrier


def make_kda_bars(cfg) -> KdaBars:
    """Bars factory.  MUST be called from inside ``_kernel`` (allocates the
    mbarrier rings in SMEM ahead of the data buffers)."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0_GROUP_THREADS = cfg.cg0_warps_per_group * WARP
    CG1_THREADS = len(cfg.compute_group_1_warp_ids) * WARP

    return KdaBars(
        mb_tma_done=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.TMA_LOAD),
        mb_inputs_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=WARP, producer=Producer.THREAD),
        mb_inputs_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_o_acc_done=MBarrier(alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_state_k_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_update_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_state_scale_diag_done=MBarrier(
            alloc(cfg.smem_state_scale_diag_stages),
            stages=cfg.smem_state_scale_diag_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_kk_qk_super_mma_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=2 * WARP, producer=Producer.THREAD),
        mb_kk_qk_mma_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_k_restore_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_rhs_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_update_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_final_state_stored=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_a_ready=MBarrier(alloc(cfg.smem_pairwise_stages), stages=cfg.smem_pairwise_stages, init_count=WARP, producer=Producer.THREAD),
        mb_qk_acc_ready=MBarrier(alloc(cfg.smem_pairwise_stages), stages=cfg.smem_pairwise_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_pairwise_stages), stages=cfg.smem_pairwise_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_qk_scale_ready=MBarrier(
            alloc(cfg.qk_scale_ready_stages),
            stages=cfg.qk_scale_ready_stages,
            init_count=CG0_GROUP_THREADS,
            producer=Producer.THREAD,
        ),
        mb_k_decay_cg0_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=WARP, producer=Producer.THREAD),
        mb_sched_ready=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=1, producer=Producer.THREAD),
        mb_sched_done=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=15, producer=Producer.THREAD),
        # H staging handshake: CG1 fills sH (ready), the epilogue TMA-stores
        # and frees it (done); single stage
        mb_h_tmastg_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_h_tmastg_done=MBarrier(alloc(1), stages=1, init_count=WARP, producer=Producer.THREAD),
    )


# ---------------------------------------------------------------------------
# Device-side helpers / warp bodies
# ---------------------------------------------------------------------------


@cute.jit
def _gate_log2(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        half = cutlass.Float32(0.5)
        sigmoid = cute.math.tanh(raw_gate * half, approx=True) * half + half
        return cfg.gate_scale_log2 * sigmoid
    # Default ABI: gate arrives in natural-log space
    return raw_gate * cutlass.Float32(LOG2_E)


# ---------------------------------------------------------------------------
# Dynamic tile scheduler: global-ticket ring
# ---------------------------------------------------------------------------


@cute.jit
def _sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas):
    """TMA-warp side: pull the next tile off the global ticket, publish it."""
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
    """Consumer side: read the TMA warp's published next tile."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_ready[sched_state.idx].wait(sched_state.phase)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_done[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


@cute.jit
def _tmaldg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    mSched,
    sSched,
    tma_tx_bytes,
    lane,
    mBeta,
    sQ_raw,
    sK_raw,
    sV_raw,
    sGate_raw,
    sBeta_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): persistent tile-scheduler loop + per-chunk
    q/k/v/gate G->S loads on one shared tx-count mbarrier plus the per-token
    beta scalar stage.  Loads go through the per-(batch, head) descriptor
    array: head grouping and the sequence base live in each descriptor and
    the token extent is capped per sequence, so coordinates are
    sequence-relative and tail chunks zero-fill in hardware."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
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
    tma_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        head_o = head_idx
        slot = (batch_idx * cutlass.Int32(cfg.n_heads_out) + head_idx) * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        if nvvm.elect_sync():
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
        for chunk_idx in cutlass.range(cstart, wend, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t
            bars.mb_inputs_done[raw_index.idx].wait(raw_index.phase)
            # ---- q/k/v/gate TMA loads + per-token beta scalar stage ------------
            if nvvm.elect_sync():
                bars.mb_tma_done.arrive(n_bytes=tma_tx_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), chunk_start)
            tma_load_tile(sQ_tma[raw_index.idx], q_slice, bars.mb_tma_done.smem_ptr, acquire=False)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, bars.mb_tma_done.smem_ptr, acquire=False)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, bars.mb_tma_done.smem_ptr, acquire=False)
            gate_slice = tma_slice_runtime_desc(desc_gate_slot, cutlass.Int32(0), chunk_start)
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, bars.mb_tma_done.smem_ptr, acquire=False)

            if lane < cfg.b_t:
                token_idx = chunk_start + lane
                beta_value = cutlass.Float32(0.0)
                if token_idx < seqlen_b:
                    beta_value = mBeta[batch_start + token_idx, head_o].to(cutlass.Float32)
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        # Roundtrip through the io dtype to bit-match host-side mBeta.sigmoid()
                        half = cutlass.Float32(0.5)
                        beta_value = (cute.math.tanh(beta_value * half, approx=True) * half + half).to(mBeta.element_type).to(cutlass.Float32)
                sBeta_raw[raw_index.idx * cfg.b_t + lane] = beta_value

            bars.mb_tma_done.wait(tma_index.phase)
            tma_index = advance(tma_index, 1)
            bars.mb_inputs_ready[raw_index.idx].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx, sched_state = _sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _super_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    sK_inv_raw,
    sPairwise_raw,
    sBeta_raw,
    sK_decay_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): persistent tile-scheduler loop +
    register-MMA kk^T, L = beta*tril(kk), and the Neumann-series A_inv,
    staged to pairwise SMEM."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    raw_index = PipelineState.start(phase=0)
    # ---- ldmatrix/stmatrix lane decode ---------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    decay_key_mask = cutlass.Int32(8) ^ ((lhs_row_coord & cutlass.Int32(2)) * cutlass.Int32(16))
    elems_per_128b = cutlass.Int32(64)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + (stsm_col_coord ^ (cfg.b_t // 2)), bbits=1, mbase=3, sshift=3)
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = wend - cstart  # processed chunks; ring bookkeeping runs on gbase + li
        for li in cutlass.range(sk_nt, unroll=1):
            gc = gbase + li
            decay_stage = gc % cfg.smem_decay_stages
            pairwise_stage = gc % cfg.smem_pairwise_stages
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sPairwise_ptr = sPairwise_raw.data_ptr() + pairwise_stage * (2 * cfg.b_t * cfg.b_t)

            bars.mb_a_done[pairwise_stage].wait(((gc // cfg.smem_pairwise_stages) + 1) % 2)
            bars.mb_k_decay_cg0_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            # ---- kk^T register MMA over the K blocks ---------------------------
            kk_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)

            for k_block in cutlass.range_constexpr((cfg.d_k // 16)):
                # Load B operand
                k_inv_col = k_block * 16 + rhs_col_offset
                k_inv_segment = k_inv_col // 64
                rhs_vec = nvvm.ldmatrix(
                    sK_inv_ptr
                    + k_inv_segment * (cfg.b_t * 64)
                    + rhs_row_coord * 64
                    + swizzle_xor_128b(rhs_row_coord, k_inv_col - k_inv_segment * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                # Load A operand
                storage_key = (k_block * 16 + lhs_col_offset) ^ decay_key_mask
                storage_slice = storage_key // elems_per_128b
                key_in_slice = storage_key - storage_slice * elems_per_128b
                storage_phase = key_in_slice // cutlass.Int32(16)
                byte_in_slice = (
                    lhs_row_coord * cutlass.Int32(128)
                    + storage_phase * cutlass.Int32(32)
                    + (key_in_slice - storage_phase * cutlass.Int32(16)) * cutlass.Int32(2)
                )
                kk_lhs_vec = nvvm.ldmatrix(
                    sK_decay_ptr
                    + storage_slice * cutlass.Int32(cfg.b_t) * elems_per_128b
                    + ((byte_in_slice ^ ((lhs_row_coord & cutlass.Int32(7)) << 4)) // cutlass.Int32(2)),
                    4,
                    nvvm.MMALayout.ROW,
                )

                mma_step(
                    kk_acc,
                    (kk_lhs_vec[0], kk_lhs_vec[1], kk_lhs_vec[2], kk_lhs_vec[3]),
                    (rhs_vec[0], rhs_vec[1], rhs_vec[2], rhs_vec[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            # ---- L = beta * tril(kk, -1) fragment ------------------------------
            row_lo = lane // 4
            row_hi = row_lo + cutlass.Int32(8)
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            l_frag = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_lo
                beta_scale = beta_lo
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_hi
                    beta_scale = beta_hi
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                lower = kk_acc[accum_idx] if row_coord > col_coord else cutlass.Float32(0.0)
                l_frag[accum_idx] = lower * beta_scale
            l_a0 = fp32_to_fp16(l_frag[0], l_frag[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_frag[2], l_frag[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_frag[4], l_frag[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_frag[6], l_frag[7], dtype=cfg.io_dtype)
            l_values = cutlass.Vector.from_elements((l_a0, l_a1, l_a2, l_a3), cutlass.Int32).bitcast(cfg.io_dtype).to(cutlass.Float32)

            # ---- A_inv = I - L, then three Neumann doubling rounds -------------
            ainv_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_lo
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_hi
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                eye = cutlass.Float32(1.0) if row_coord == col_coord else cutlass.Float32(0.0)
                ainv_acc[accum_idx] = eye - l_values[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            for _round in cutlass.range_constexpr(3):
                # Lpow <- Lpow @ Lpow (packed A-layout fragments, B via movmatrix)
                sq_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
                for accum_idx in cutlass.range_constexpr(8):
                    sq_acc[accum_idx] = cutlass.Float32(0.0)
                mma_step(
                    sq_acc,
                    (lpow_a0, lpow_a1, lpow_a2, lpow_a3),
                    (movmatrix_16b(lpow_a0), movmatrix_16b(lpow_a1), movmatrix_16b(lpow_a2), movmatrix_16b(lpow_a3)),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                lpow_a0 = fp32_to_fp16(sq_acc[0], sq_acc[1], dtype=cfg.io_dtype)
                lpow_a1 = fp32_to_fp16(sq_acc[2], sq_acc[3], dtype=cfg.io_dtype)
                lpow_a2 = fp32_to_fp16(sq_acc[4], sq_acc[5], dtype=cfg.io_dtype)
                lpow_a3 = fp32_to_fp16(sq_acc[6], sq_acc[7], dtype=cfg.io_dtype)
                # A_inv <- A_inv + A_inv @ Lpow, keeping A_inv in registers
                upd_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
                for accum_idx in cutlass.range_constexpr(8):
                    upd_acc[accum_idx] = cutlass.Float32(0.0)
                mma_step(
                    upd_acc,
                    (
                        fp32_to_fp16(ainv_acc[0], ainv_acc[1], dtype=cfg.io_dtype),
                        fp32_to_fp16(ainv_acc[2], ainv_acc[3], dtype=cfg.io_dtype),
                        fp32_to_fp16(ainv_acc[4], ainv_acc[5], dtype=cfg.io_dtype),
                        fp32_to_fp16(ainv_acc[6], ainv_acc[7], dtype=cfg.io_dtype),
                    ),
                    (movmatrix_16b(lpow_a0), movmatrix_16b(lpow_a1), movmatrix_16b(lpow_a2), movmatrix_16b(lpow_a3)),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                for accum_idx in cutlass.range_constexpr(8):
                    ainv_acc[accum_idx] = ainv_acc[accum_idx].to(cfg.io_dtype).to(cutlass.Float32) + upd_acc[accum_idx]

            nvvm.stmatrix(
                sPairwise_ptr + (cfg.b_t * cfg.b_t) + stsm_idx,
                [
                    fp32_to_fp16(ainv_acc[0], ainv_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(ainv_acc[2], ainv_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(ainv_acc[4], ainv_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(ainv_acc[6], ainv_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[pairwise_stage].arrive()
            bars.mb_kk_qk_super_mma_done[decay_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        gbase += sk_nt
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    tmem_hold,
    sPairwise,
    sK_decay,
    sK_restore,
    sQ_decay,
    sState_scale_diag,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): persistent tile-scheduler loop, issues
    all six state GEMMs in dependency order and owns the TMEM lifecycle."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    tmem_base = tmem_hold.load()
    state_inp_index = PipelineState.start(phase=0)
    rhs_index = PipelineState.start(phase=0)
    update_index = PipelineState.start(phase=0)
    final_state_index = PipelineState.start(phase=0)
    qk_scale_index = PipelineState.start(phase=0)
    # ---- chunk-invariant GEMM descriptors ------------------------------
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
    bmm_state_desc = MmaDesc(
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
    bmm_diag_desc = MmaDesc(
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
    bmm_pairwise_desc = MmaDesc(
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
    bmm_final_state_desc = MmaDesc(
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
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = wend - cstart
        for li in cutlass.range(sk_nt, unroll=1):
            gc = gbase + li
            q_state_acc_stage = gc % cfg.tmem_q_state_acc_stages
            decay_stage = gc % cfg.smem_decay_stages
            state_scale_diag_stage = qk_scale_index.idx
            pairwise_stage = gc % cfg.smem_pairwise_stages
            sK_decay_stage = sK_decay[decay_stage]
            sQ_decay_stage = sQ_decay[decay_stage]
            sK_restore_stage = sK_restore[decay_stage]
            sState_scale_diag_stage = sState_scale_diag[state_scale_diag_stage]
            sPairwise_stage = sPairwise[pairwise_stage]

            # ---- state*k -> state_k_acc ----------------------------------------
            bars.mb_k_decay_cg0_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            bars.mb_state_inp_ready.wait(state_inp_index.phase)
            state_inp_index = advance(state_inp_index, 1)
            desc_k_decay = sK_decay_stage.desc()

            state_a_tmem_base = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_inp_offset, cutlass.Int8)
            mma_ts(
                bmm_state_desc,
                state_a_tmem_base,
                desc_k_decay,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_offset, cutlass.Float32),
                accumulate=False,
            )

            if nvvm.elect_sync():
                bars.mb_state_k_acc_ready.arrive(cta_group=1)

            # ---- state*q -> q_state_acc (stays live until qk@update fuses into o)
            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)
            bars.mb_o_acc_done[q_state_acc_stage].wait(((gc // cfg.tmem_q_state_acc_stages + cutlass.Int32(1)) % cutlass.Int32(2)))
            desc_q_decay = sQ_decay_stage.desc()

            state_a_tmem_base = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_inp_offset, cutlass.Int8)
            mma_ts(
                bmm_state_desc,
                state_a_tmem_base,
                desc_q_decay,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_q_state_acc_offset + q_state_acc_stage * cfg.b_t, cutlass.Float32),
                accumulate=False,
            )

            if nvvm.elect_sync():
                bars.mb_kk_qk_mma_done[decay_stage].arrive(cta_group=1)

            # ---- state decay (per-k-atom diag blocks) ----------------------------
            desc_diag = sState_scale_diag_stage.desc()

            for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                mma_ts(
                    bmm_diag_desc,
                    nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_inp_offset + k_block * 8, cutlass.Int8),
                    desc_diag.advance_start_address(k_block * 256 * 2),
                    nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_offset + k_block * 16, cutlass.Float32),
                    accumulate=False,
                )

            if nvvm.elect_sync():
                bars.mb_state_scale_diag_done[state_scale_diag_stage].arrive(cta_group=1)

            # ---- update = A_inv @ rhs -> update_acc ------------------------------
            bars.mb_a_ready[pairwise_stage].wait((gc // cfg.smem_pairwise_stages) % 2)
            bars.mb_rhs_ready.wait(rhs_index.phase)
            rhs_index = advance(rhs_index, 1)
            lhs_tmem = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_rhs_inp_offset, cutlass.Int8)
            desc_pairwise = sPairwise_stage.shifted((cfg.b_t * cfg.b_t)).desc()
            mma_ts(
                bmm_pairwise_desc,
                lhs_tmem,
                desc_pairwise,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_update_acc_offset, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_update_acc_ready.arrive(cta_group=1)

            # ---- final_state += update @ k_restore -------------------------------
            bars.mb_update_ready.wait(update_index.phase)
            update_index = advance(update_index, 1)
            update_tmem = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_update_inp_offset, cutlass.Int8)
            desc_k_restore = sK_restore_stage.desc()

            mma_ts(
                bmm_final_state_desc,
                update_tmem,
                desc_k_restore,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_offset, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)

            # ---- o += qk @ update -> q_state_acc ----------------------------------
            bars.mb_qk_acc_ready[pairwise_stage].wait((gc // cfg.smem_pairwise_stages) % 2)
            lhs_tmem = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_update_inp_offset, cutlass.Int8)
            desc_pairwise = sPairwise_stage.desc()
            mma_ts(
                bmm_pairwise_desc,
                lhs_tmem,
                desc_pairwise,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_q_state_acc_offset + q_state_acc_stage * cfg.b_t, cutlass.Float32),
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_o_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[pairwise_stage].arrive(cta_group=1)
            qk_scale_index = advance(qk_scale_index, cfg.smem_state_scale_diag_stages)

        bars.mb_final_state_stored.wait(final_state_index.phase)
        final_state_index = advance(final_state_index, 1)
        gbase += sk_nt
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def _epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    mO,
    sK_inv_raw,
    sO_raw,
    sPairwise_raw,
    sQ_decay_raw,
    sH_raw,
    desc_o_base,
    desc_h_base,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """Epilogue warp role (warp 15): persistent tile-scheduler loop +
    register-MMA qk (causal), A_inv qk staging, the O TMA store drain
    (split-K warmup chunks drain the SMEM stage but never store), and the
    per-chunk H TMA store when checkpoints are enabled.  O and H stores go
    through the per-(batch, head) descriptor arrays: the token / entry
    extents are capped per sequence, so partial tails are clipped by the
    hardware and the H coordinate is sequence-local."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    if cutlass.const_expr(cfg.enable_checkpoints):
        sH_tma = SmemTile(
            base=sH_raw,
            elems_per_stage=(cfg.d_k * cfg.d_v),
            stages=1,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=(cfg.d_v // 64),
            tma_granu_elems=64,
            tma_subtile_stride_elems=cfg.d_k * 64,
        )
    h_ready_index = PipelineState.start(phase=0)
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.b_t * cfg.d_v),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    qk_scale_index = PipelineState.start(phase=0)
    # ---- ldmatrix/stmatrix lane decode ---------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    decay_key_mask = cutlass.Int32(8) ^ ((lhs_row_coord & cutlass.Int32(2)) * cutlass.Int32(16))
    elems_per_128b = cutlass.Int32(64)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + (stsm_col_coord ^ (cfg.b_t // 2)), bbits=1, mbase=3, sshift=3)
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        head_o = head_idx
        o_slot = (batch_idx * cutlass.Int32(cfg.n_heads_out) + head_idx) * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_o_slot = (desc_o_base + o_slot).tospace(cutlass.AddressSpace.generic)
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_h_slot = (desc_h_base + o_slot).tospace(cutlass.AddressSpace.generic)
            if nvvm.elect_sync():
                tma_tensormap_acquire(desc_h_slot)
        if nvvm.elect_sync():
            tma_tensormap_acquire(desc_o_slot)
        sk_nt = wend - cstart
        for li in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cstart + li
            gc = gbase + li
            decay_stage = gc % cfg.smem_decay_stages
            pairwise_stage = gc % cfg.smem_pairwise_stages
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sPairwise_ptr = sPairwise_raw.data_ptr() + pairwise_stage * (2 * cfg.b_t * cfg.b_t)

            bars.mb_a_done[pairwise_stage].wait(((gc // cfg.smem_pairwise_stages) + 1) % 2)
            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)
            # ---- qk register MMA (inclusive-causal) ----------------------------
            qk_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                qk_acc[accum_idx] = cutlass.Float32(0.0)

            for k_block in cutlass.range_constexpr((cfg.d_k // 16)):
                # Load B operand
                k_inv_col = k_block * 16 + rhs_col_offset
                k_inv_segment = k_inv_col // 64
                rhs_vec = nvvm.ldmatrix(
                    sK_inv_ptr
                    + k_inv_segment * (cfg.b_t * 64)
                    + rhs_row_coord * 64
                    + swizzle_xor_128b(rhs_row_coord, k_inv_col - k_inv_segment * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                # Load A operand
                storage_key = (k_block * 16 + lhs_col_offset) ^ decay_key_mask
                storage_slice = storage_key // elems_per_128b
                key_in_slice = storage_key - storage_slice * elems_per_128b
                storage_phase = key_in_slice // cutlass.Int32(16)
                byte_in_slice = (
                    lhs_row_coord * cutlass.Int32(128)
                    + storage_phase * cutlass.Int32(32)
                    + (key_in_slice - storage_phase * cutlass.Int32(16)) * cutlass.Int32(2)
                )
                qk_lhs_vec = nvvm.ldmatrix(
                    sQ_decay_ptr
                    + storage_slice * cutlass.Int32(cfg.b_t) * elems_per_128b
                    + ((byte_in_slice ^ ((lhs_row_coord & cutlass.Int32(7)) << 4)) // cutlass.Int32(2)),
                    4,
                    nvvm.MMALayout.ROW,
                )

                mma_step(
                    qk_acc,
                    (qk_lhs_vec[0], qk_lhs_vec[1], qk_lhs_vec[2], qk_lhs_vec[3]),
                    (rhs_vec[0], rhs_vec[1], rhs_vec[2], rhs_vec[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            for accum_idx in cutlass.range_constexpr(8):
                row_coord = lane // 4
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_coord + cutlass.Int32(8)
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                qk_acc[accum_idx] = qk_acc[accum_idx] if row_coord >= col_coord else cutlass.Float32(0.0)

            nvvm.stmatrix(
                sPairwise_ptr + stsm_idx,
                [
                    fp32_to_fp16(qk_acc[0], qk_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(qk_acc[2], qk_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(qk_acc[4], qk_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(qk_acc[6], qk_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_acc_ready[pairwise_stage].arrive()
            bars.mb_kk_qk_super_mma_done[decay_stage].arrive()
            qk_scale_index = advance(qk_scale_index, cfg.qk_scale_ready_stages)

            # ---- O drain: staged output tile -> GMEM TMA store -----------------
            if li > 0:
                output_chunk = chunk_idx - cutlass.Int32(1)
                output_chunk_start = output_chunk * cfg.b_t
                o_stage = (gc - cutlass.Int32(1)) % cfg.smem_o_stages
                bars.mb_o_tmastg_ready[o_stage].wait(((gc - cutlass.Int32(1)) // cfg.smem_o_stages) % 2)
                o_slice = tma_slice_runtime_desc(desc_o_slot, cutlass.Int32(0), output_chunk_start)
                if cutlass.const_expr(cfg.split_k):
                    # warmup chunks stage O to SMEM but never store it
                    if output_chunk >= wstart:
                        tma_store_tile(sO_tma[o_stage], o_slice, acquire=False)
                        tma_store_commit()
                else:
                    tma_store_tile(sO_tma[o_stage], o_slice, acquire=False)
                    tma_store_commit()
                tma_store_wait(0)
                bars.mb_o_tmastg_done[o_stage].arrive()
            # ---- H store: CG1 staged the state entering chunk_idx ----------
            if cutlass.const_expr(cfg.enable_checkpoints):
                if li > 0:
                    tokens_done = chunk_idx * cutlass.Int32(cfg.b_t)
                    do_h = tokens_done % checkpoint_every_n_tokens == 0
                    if cutlass.const_expr(cfg.split_k):
                        do_h = do_h and chunk_idx >= wstart
                    if do_h:
                        bars.mb_h_tmastg_ready.wait(h_ready_index.phase)
                        h_ready_index = advance(h_ready_index, 1)
                        # sequence-local entry: the per-(b,h) descriptor folds the
                        # sequence base into GLOBAL_ADDRESS and caps the extent
                        h_entry = tokens_done // checkpoint_every_n_tokens - cutlass.Int32(1)
                        h_slice = tma_slice_runtime_desc(desc_h_slot, cutlass.Int32(0), cutlass.Int32(0), h_entry)
                        tma_store_tile(sH_tma[0], h_slice, acquire=False)
                        tma_store_commit()
                        tma_store_wait(0)
                        bars.mb_h_tmastg_done.arrive()
        # ---- last computed chunk drain (always owned: it is wend - 1) ------
        if sk_nt > 0:
            output_chunk = wend - cutlass.Int32(1)
            og = gbase + sk_nt - cutlass.Int32(1)
            output_chunk_start = output_chunk * cfg.b_t
            o_stage = og % cfg.smem_o_stages
            bars.mb_o_tmastg_ready[o_stage].wait((og // cfg.smem_o_stages) % 2)
            # a partial last chunk is clipped by the descriptor's token extent
            o_slice = tma_slice_runtime_desc(desc_o_slot, cutlass.Int32(0), output_chunk_start)
            tma_store_tile(sO_tma[o_stage], o_slice, acquire=False)
            tma_store_commit()
            tma_store_wait(0)
            bars.mb_o_tmastg_done[o_stage].arrive()
        gbase += sk_nt
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    warp_idx,
    mQ,
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_raw,
    sK_raw,
    sQ_raw,
    sV_raw,
    sK_decay_raw,
    sK_restore_raw,
    sQ_decay_raw,
    sState_scale_diag_raw,
    bars,
) -> None:
    """CG0 warp role (warps 0-7, two ping-pong groups): persistent
    tile-scheduler loop + gate prefix scan and the decay/restore operand
    materialization into tcgen05 SMEM."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    cg0_group_id = cg0_warp // cfg.cg0_warps_per_group
    cg0_local_warp = cg0_warp % cfg.cg0_warps_per_group
    prefix_dim = cg0_local_warp * cfg.threads_per_warp + lane
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        head_o = head_idx
        sk_nt = wend - cstart
        if cutlass.const_expr(cfg.safe_gate):
            # per-head safe-gate constants straight from GMEM (the head
            # changes per tile, so there is no SMEM staging to handshake)
            if sk_nt > 0:
                cg0_a_log_exp = cute.math.exp2(mA_log[head_o].to(cutlass.Float32) * LOG2_E, fastmath=True)
                cg0_dt_bias_value = mDt_bias[head_o, prefix_dim].to(cutlass.Float32)
        for li in cutlass.range(cg0_group_id, sk_nt, cfg.cg0_group_count, unroll=1):
            chunk_idx = cstart + li
            gc = gbase + li
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = gc % cfg.smem_decay_stages
            raw_stage = gc % cfg.smem_raw_stages
            state_scale_diag_stage = gc % cfg.smem_state_scale_diag_stages
            qk_scale_ready_stage = state_scale_diag_stage
            sQ_ptr = sQ_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = sK_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sV_ptr = sV_raw.data_ptr() + raw_stage * (cfg.d_v * cfg.b_t)
            sGate_ptr = sGate_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = sK_restore_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sState_scale_diag_ptr = sState_scale_diag_raw.data_ptr() + state_scale_diag_stage * ((cfg.d_k // 16) * 256)

            bars.mb_inputs_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)

            # ---- tail chunk: zero-fill raw staging past seqlen -----------------
            if chunk_start + cutlass.Int32(cfg.b_t) > seqlen_b:
                if cg0_local_warp == 0:
                    f16_zero = mQ.element_type(0.0)
                    f16_zero_vec = cutlass.Vector.from_elements(
                        (
                            f16_zero,
                            f16_zero,
                            f16_zero,
                            f16_zero,
                            f16_zero,
                            f16_zero,
                            f16_zero,
                            f16_zero,
                        ),
                        mQ.element_type,
                    )
                    f32_zero = cutlass.Float32(0.0)
                    f32_zero_vec = cutlass.Vector.from_elements(
                        (f32_zero, f32_zero, f32_zero, f32_zero),
                        cutlass.Float32,
                    )
                    for row in cutlass.range_constexpr(cfg.b_t):
                        token_idx = chunk_start + cutlass.Int32(row)
                        if token_idx >= seqlen_b:
                            if lane < (cfg.d_k // 8):
                                f16_dim_base = lane * 8
                                f16_segment = f16_dim_base // 64
                                f16_segment_dim = f16_dim_base - f16_segment * 64
                                f16_idx = f16_segment * (cfg.b_t * 64) + row * 64 + swizzle_xor_128b(row, f16_segment_dim, elem_bytes=2)
                                (sQ_ptr + f16_idx).store(f16_zero_vec, alignment=16)
                                (sK_ptr + f16_idx).store(f16_zero_vec, alignment=16)
                                (sV_ptr + f16_idx).store(f16_zero_vec, alignment=16)
                            if lane < (cfg.d_k // 4):
                                f32_dim_base = lane * 4
                                f32_segment = f32_dim_base // 32
                                f32_segment_dim = f32_dim_base - f32_segment * 32
                                f32_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                                (sGate_ptr + f32_idx).store(f32_zero_vec, alignment=16)
                nvvm.barrier_cta_sync(cfg.nbar_cg0_group0_id + cg0_group_id, thread_count=cfg.cg0_threads_per_group)

            row_group_start = cg0_local_warp * (cfg.b_t // cfg.cg0_warps_per_group)
            lane_row_group = lane // 8
            lane_in_row_group = lane - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            g_prefix_ptr = sGate_ptr

            prefix_dim = cg0_local_warp * cfg.threads_per_warp + lane
            # ---- gate prefix scan: cumulative log-gate per key channel --------
            # gathers first, math second: a load consumed in the same iteration
            # serializes on LDS latency
            gate_raw = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            for row in cutlass.range_constexpr(cfg.b_t):
                f32_segment = prefix_dim // 32
                f32_segment_dim = prefix_dim - f32_segment * 32
                prefix_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                gate_raw[row] = (sGate_ptr + prefix_idx).load()
            g_prefix_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            if cutlass.const_expr(cfg.safe_gate):
                valid_rows = seqlen_b - chunk_idx * cutlass.Int32(cfg.b_t)
                valid_mask = cutlass.vector.create_mask([cfg.b_t], [valid_rows])
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    gate0 = cg0_a_log_exp * (gate_raw[row0] + cg0_dt_bias_value)
                    gate1 = cg0_a_log_exp * (gate_raw[row1] + cg0_dt_bias_value)
                    gate0 = _gate_log2(
                        cfg,
                        gate0,
                    )
                    gate1 = _gate_log2(
                        cfg,
                        gate1,
                    )
                    gate_pair = cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32)
                    gate_pair = cutlass.vector.where(valid_mask[row0 : row1 + 1], gate_pair, 0.0)
                    g_prefix_regs[row0] = gate_pair[0]
                    g_prefix_regs[row1] = gate_pair[1]
            else:
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate = gate_raw[row]
                    token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cutlass.Int32(row)
                    if token_idx < seqlen_b:
                        gate = _gate_log2(
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

            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            # ---- exp2(g): stage prefixes + final-token decay ------------------
            exp_g_last = g_prefix_regs[cfg.b_t - 1]
            for row in cutlass.range_constexpr(cfg.b_t):
                f32_segment = prefix_dim // 32
                f32_segment_dim = prefix_dim - f32_segment * 32
                prefix_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                (sGate_ptr + prefix_idx).store(g_prefix_regs[row])

            # ---- state-scale diag: stage exp2(g_last) decay blocks -------------
            bars.mb_state_scale_diag_done[state_scale_diag_stage].wait((gc // cfg.smem_state_scale_diag_stages + 1) % 2)
            block = prefix_dim // cutlass.Int32(16)
            coord = prefix_dim - block * cutlass.Int32(16)
            storage_col = coord ^ cutlass.Int32((cfg.b_t // 2))
            linear_idx = block * cutlass.Int32(256) + coord * cutlass.Int32(16) + storage_col
            diag_idx = swizzle_lin_S(linear_idx, bbits=1, mbase=3, sshift=3)
            sState_scale_diag_ptr[diag_idx] = exp_g_last.to(cfg.io_dtype)

            nvvm.barrier_cta_sync(cfg.nbar_cg0_group0_id + cg0_group_id, thread_count=cfg.cg0_threads_per_group)

            k_inv_words = cutlass.Array(cutlass.Int32, 2 * 4, alignment=16)
            raw_q_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            raw_k_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            # ---- optional q/k L2-norm + K_inv staging --------------------------
            q_sum_sq = cutlass.Float32(0.0)
            k_sum_sq = cutlass.Float32(0.0)
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                raw_q_vec = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_vec = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_q_vec_f32 = raw_q_vec.to(cutlass.Float32)
                raw_k_vec_f32 = raw_k_vec.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    q_val = raw_q_vec_f32[dim_offset]
                    k_val = raw_k_vec_f32[dim_offset]
                    raw_q_regs[reg_base + dim_offset] = q_val
                    raw_k_regs[reg_base + dim_offset] = k_val
                    q_sum_sq = q_sum_sq + q_val * q_val
                    k_sum_sq = k_sum_sq + k_val * k_val

            # opaque 1.0: keeps the no-l2norm packed-mul operands out of libNVVM's
            # constant folder (the documented inline_ptx "n"-constraint ICE)
            q_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            k_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            if cutlass.const_expr(cfg.l2norm):
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                q_inv_norm = cute.math.rsqrt(cute.math.max(q_sum_sq, norm_floor_sq), fastmath=True)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)

            # ---- decay/restore operands: exp2(+-g) applied per key channel -----
            exp_g_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            exp_g_last_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            # both halves' prefix gathers up front: the sK_inv/sK_decay stores
            # below would otherwise pin the second half's loads behind them
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    g_prefix_idx = f32_segment * (cfg.b_t * 32) + decay_row * 32 + swizzle_xor_128b(decay_row, f32_segment_dim, elem_bytes=4)
                    exp_g_vec = (g_prefix_ptr + g_prefix_idx).load(count=4, alignment=16)
                    exp_g_last_idx = f32_segment * (cfg.b_t * 32) + (cfg.b_t - 1) * 32 + swizzle_xor_128b((cfg.b_t - 1), f32_segment_dim, elem_bytes=4)
                    exp_g_last_vec = (g_prefix_ptr + exp_g_last_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    exp_g_regs[f32_reg_base] = exp_g_vec[0]
                    exp_g_regs[f32_reg_base + 1] = exp_g_vec[1]
                    exp_g_regs[f32_reg_base + 2] = exp_g_vec[2]
                    exp_g_regs[f32_reg_base + 3] = exp_g_vec[3]
                    exp_g_last_regs[f32_reg_base] = exp_g_last_vec[0]
                    exp_g_last_regs[f32_reg_base + 1] = exp_g_last_vec[1]
                    exp_g_last_regs[f32_reg_base + 2] = exp_g_last_vec[2]
                    exp_g_last_regs[f32_reg_base + 3] = exp_g_last_vec[3]

            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                # ---- k decay operand: exp2(g) * k ------------------------------
                k_decay_words = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
                    k_pair = fp32_to_fp16(k_value0, k_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_decay_words[pair_idx] = mul_f16x2(k_pair, exp_g_pair, cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    k_inv_words[dim_half * 4 + pair_idx] = fp32_to_fp16(k_value0 * exp_neg_g0, k_value1 * exp_neg_g1, dtype=cfg.io_dtype)

                k_inv_vec = cutlass.Vector.from_elements(
                    (
                        k_inv_words[dim_half * 4],
                        k_inv_words[dim_half * 4 + 1],
                        k_inv_words[dim_half * 4 + 2],
                        k_inv_words[dim_half * 4 + 3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                k_decay_vec = cutlass.Vector.from_elements(
                    (
                        k_decay_words[0],
                        k_decay_words[1],
                        k_decay_words[2],
                        k_decay_words[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                if cutlass.const_expr(dim_half == 0):
                    operand_done_phase = ((gc // cfg.smem_decay_stages) + 1) % 2
                    bars.mb_kk_qk_super_mma_done[decay_stage].wait(operand_done_phase)
                    bars.mb_kk_qk_mma_done[decay_stage].wait(operand_done_phase)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_inv_swizzled_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sK_inv_ptr + k_inv_swizzled_idx).store(k_inv_vec, alignment=16)
                key_mask = cutlass.Int32(8) ^ (decay_row & cutlass.Int32(2)) * cutlass.Int32(16)
                decay_storage_dim_base = dim_base ^ key_mask
                decay_linear_idx_base = decay_row * cfg.d_k + decay_storage_dim_base
                sw128_elems_per_128b = 128 // 2
                sw128_row = decay_linear_idx_base // cfg.d_k
                sw128_col = decay_linear_idx_base - sw128_row * cfg.d_k
                sw128_slice = sw128_col // sw128_elems_per_128b
                sw128_col_in_slice = sw128_col - sw128_slice * sw128_elems_per_128b
                sw128_slice_linear = sw128_row * sw128_elems_per_128b + sw128_col_in_slice
                sw128_byte = sw128_slice_linear * 2
                sw128_mask = (sw128_byte >> 7 & 7) << 4
                decay_swizzled_idx_base = sw128_slice * cfg.b_t * sw128_elems_per_128b + (sw128_byte ^ sw128_mask) // 2
                (sK_decay_ptr + decay_swizzled_idx_base).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_cg0_ready[decay_stage].arrive()

            # ---- q decay operand -----------------------------------------------
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                q_decay_words = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_inv_norm, q_inv_norm)
                    q_pair = fp32_to_fp16(q_value0, q_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    q_decay_words[pair_idx] = mul_f16x2(q_pair, exp_g_pair, cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (
                        q_decay_words[0],
                        q_decay_words[1],
                        q_decay_words[2],
                        q_decay_words[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                key_mask = cutlass.Int32(8) ^ (decay_row & cutlass.Int32(2)) * cutlass.Int32(16)
                decay_storage_dim_base = dim_base ^ key_mask
                decay_linear_idx_base = decay_row * cfg.d_k + decay_storage_dim_base
                sw128_elems_per_128b = 128 // 2
                sw128_row = decay_linear_idx_base // cfg.d_k
                sw128_col = decay_linear_idx_base - sw128_row * cfg.d_k
                sw128_slice = sw128_col // sw128_elems_per_128b
                sw128_col_in_slice = sw128_col - sw128_slice * sw128_elems_per_128b
                sw128_slice_linear = sw128_row * sw128_elems_per_128b + sw128_col_in_slice
                sw128_byte = sw128_slice_linear * 2
                sw128_mask = (sw128_byte >> 7 & 7) << 4
                decay_swizzled_idx_base = sw128_slice * cfg.b_t * sw128_elems_per_128b + (sw128_byte ^ sw128_mask) // 2
                (sQ_decay_ptr + decay_swizzled_idx_base).store(q_decay_vec, alignment=16)

            bars.mb_k_restore_done[decay_stage].wait(((gc // cfg.smem_decay_stages + 1) % 2))

            # ---- k_restore operand ----------------------------------------------
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                k_restore_words = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    exp_g_last_pair = fp32_to_fp16(exp_g_last_regs[reg_base + dim0], exp_g_last_regs[reg_base + dim1], dtype=cfg.io_dtype)
                    k_restore_words[pair_idx] = mul_f16x2(k_inv_words[dim_half * 4 + pair_idx], exp_g_last_pair, cfg.io_dtype)
                storage_row = decay_row ^ (cfg.b_t // 2)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_restore_idx = f16_segment * (cfg.b_t * 64) + storage_row * 64 + swizzle_xor_128b(storage_row, f16_segment_dim, elem_bytes=2)
                k_restore_vec = cutlass.Vector.from_elements(
                    (
                        k_restore_words[0],
                        k_restore_words[1],
                        k_restore_words[2],
                        k_restore_words[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                (sK_restore_ptr + k_restore_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_scale_ready[qk_scale_ready_stage].arrive()
        gbase += sk_nt
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    tmem_hold,
    warp_idx,
    mS_out,
    mS_init,
    mO,
    sO_raw,
    sBeta_raw,
    sV_raw,
    sH_raw,
    checkpoint_every_n_tokens,
    scale,
    bars,
) -> None:
    """CG1 warp role (warps 8-11): persistent tile-scheduler loop + value-side
    TMEM staging (state input, rhs, update), output drain to SMEM, and
    checkpoint/final-state stores (split-K: only owned entries)."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    sO_ptr = sO_raw.data_ptr()
    sH_ptr = sH_raw.data_ptr() if cutlass.const_expr(cfg.enable_checkpoints) else sO_raw.data_ptr()
    h_done_index = PipelineState.start(phase=1)  # sH starts free
    tmem_base = tmem_hold.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_sp = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    # ldmatrix.x4/stmatrix.x4 COL lane decode shared by the v loads and o stores
    ov_tok = (lane // 16) * 8 + (lane & 7)
    ov_col = ((lane // 8) & 1) * 8
    row_id = tmem_row + tmem_sp * cfg.threads_per_warp
    value_dim = tmem_sp * cfg.threads_per_warp + lane
    state_k_acc_index = PipelineState.start(phase=0)
    update_acc_index = PipelineState.start(phase=0)
    o_acc_index = PipelineState.start(phase=0)
    kr_index = PipelineState.start(phase=0)  # CG1's per-chunk mb_k_restore_done wait slot
    raw_index = PipelineState.start(phase=0)  # raw-ring slot for the sV/sBeta reads + inputs_done arrives
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        head_o = head_idx
        sk_nt = wend - cstart

        if sk_nt > 0:
            # ---- first chunk: seed state TMEM from mS_init (else zeros);
            # split-K warmup items (cstart > 0) rebuild their state from zero
            seed_from_s0 = cstart == 0
            for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                state_block = cutlass.Array(cutlass.Float32, 32, alignment=16)
                for col in cutlass.range_constexpr(32):
                    key_dim = key_block_start + col
                    state_value = cutlass.Float32(0.0)
                    if cutlass.const_expr(mS_init is not None):
                        state_value = mS_init[batch_idx, head_o, key_dim, value_dim].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.split_k):
                            state_value = state_value if seed_from_s0 else cutlass.Float32(0.0)
                    state_block[col] = state_value

                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr((row_id << 16) + (tmem_col + cfg.tmem_state_offset + key_block_start), cutlass.Float32),
                    state_block[0:32],
                )

            nvvm.tcgen05_wait("store")
            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t

            row_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
            state_col_id = tmem_col + cfg.tmem_state_offset
            # ---- state -> packed b16 A operand (TMEM roundtrip) ----------------
            state_blocks = []
            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                state_blocks.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + state_col_id + sub * 16, cutlass.Float32), num=16))
            nvvm.tcgen05_wait("load")

            packed_col_id = tmem_col + cfg.tmem_state_inp_offset
            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                packed_state = cutlass.Array(cutlass.Int32, 8, alignment=16)
                for packed_col in cutlass.range_constexpr(8):
                    source_pair = packed_col ^ 4
                    packed_state[packed_col] = fp32_to_fp16(state_blocks[sub][2 * source_pair], state_blocks[sub][2 * source_pair + 1], dtype=cfg.io_dtype)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr((tmem_row << 16) + packed_col_id + sub * 8, cutlass.Int8),
                    packed_state[0:8],
                )

            nvvm.tcgen05_wait("store")
            bars.mb_state_inp_ready.arrive()

            # ---- rhs staging: rhs input = beta * (v - state*k) -----------------
            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
            projection_col_id = tmem_col + cfg.tmem_state_k_acc_offset
            input_col_id = tmem_col + cfg.tmem_rhs_inp_offset
            value_dim_base = tmem_sp * cfg.threads_per_warp

            # ---- read back state*k acc + raw v fragments -----------------------
            row_id0 = tmem_row + value_dim_base
            state_k0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)

            row_id1 = row_id0 + 16
            state_k1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)

            raw_v_regs0 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_regs1 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            nvvm.tcgen05_wait("load")

            packed_rhs0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                packed_col = (reg_idx // 2) * 4 + (lane & 3)
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token1).load().to(cutlass.Float32)
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k0[frag_pair], state_k0[frag_pair + 1]
                beta_pair = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_regs0[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                packed_rhs0[reg_idx] = mul_f16x2(
                    beta_pair,
                    diff_pair,
                    cfg.io_dtype,
                )

            packed_rhs1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                packed_col = (reg_idx // 2) * 4 + (lane & 3)
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token1).load().to(cutlass.Float32)
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k1[frag_pair], state_k1[frag_pair + 1]
                beta_pair = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_regs1[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                packed_rhs1[reg_idx] = mul_f16x2(
                    beta_pair,
                    diff_pair,
                    cfg.io_dtype,
                )

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr((tmem_row << 16) + input_col_id, cutlass.Int8), packed_rhs0[0:4])

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr((tmem_row + 16 << 16) + input_col_id, cutlass.Int8), packed_rhs1[0:4])

            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            bars.mb_inputs_done[raw_index.idx].arrive()
            bars.mb_rhs_ready.arrive()

            # ---- update readback -> packed b16 A operand -----------------------
            bars.mb_update_acc_ready.wait(update_acc_index.phase)
            update = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr((tmem_row + tmem_sp * cfg.threads_per_warp << 16) + (tmem_col + cfg.tmem_update_acc_offset), cutlass.Float32),
                num=cfg.b_t,
            )
            nvvm.tcgen05_wait("load")

            packed_update = cutlass.Array(cutlass.Int32, (cfg.b_t // 2), alignment=16)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                packed_update[packed_col] = fp32_to_fp16(update[token0], update[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr((tmem_row << 16) + (tmem_col + cfg.tmem_update_inp_offset), cutlass.Int8),
                packed_update[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_wait("store")
            update_acc_index = advance(update_acc_index, 1)
            bars.mb_update_ready.arrive()

            bars.mb_k_restore_done[kr_index.idx].wait(kr_index.phase)
            kr_index = advance(kr_index, cfg.smem_decay_stages)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        # the first chunk is peeled above so this steady-state loop always
        # drains the prior chunk's output
        for li in cutlass.range(1, sk_nt, 1, unroll=1):
            chunk_idx = cstart + li
            gc = gbase + li
            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = sBeta_raw.data_ptr() + raw_index.idx * cfg.b_t

            prev_output_chunk = chunk_idx - cutlass.Int32(1)
            prev_og = gc - cutlass.Int32(1)
            prev_o_stage = prev_og % cfg.smem_o_stages
            prev_q_state_acc_stage = prev_og % cfg.tmem_q_state_acc_stages
            prev_o_stage_base = prev_o_stage * (cfg.b_t * cfg.d_v)
            # H entry gate: the state read by this restage entered chunk_idx,
            # i.e. the state after chunk_idx * b_t tokens (strictly before the
            # sequence end -- the end state is only final_state); split-K
            # warmup reconstructions below wstart belong to the previous item
            do_h = False
            if cutlass.const_expr(cfg.enable_checkpoints):
                do_h = (chunk_idx * cutlass.Int32(cfg.b_t)) % checkpoint_every_n_tokens == 0
                if cutlass.const_expr(cfg.split_k):
                    do_h = do_h and chunk_idx >= wstart
                if do_h:
                    # sH is free once the epilogue's previous TMA store retired
                    bars.mb_h_tmastg_done.wait(h_done_index.phase)
                    h_done_index = advance(h_done_index, 1)
            row_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
            state_col_id = tmem_col + cfg.tmem_state_offset
            # ---- state -> packed b16 A operand (TMEM roundtrip) ----------------
            state_blocks = []
            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                state_blocks.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + state_col_id + sub * 16, cutlass.Float32), num=16))
            bars.mb_o_tmastg_done[prev_o_stage].wait(((prev_og // cfg.smem_o_stages) + 1) % 2)
            nvvm.tcgen05_wait("load")

            packed_col_id = tmem_col + cfg.tmem_state_inp_offset
            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                packed_state = cutlass.Array(cutlass.Int32, 8, alignment=16)
                for packed_col in cutlass.range_constexpr(8):
                    source_pair = packed_col ^ 4
                    packed_state[packed_col] = fp32_to_fp16(state_blocks[sub][2 * source_pair], state_blocks[sub][2 * source_pair + 1], dtype=cfg.io_dtype)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr((tmem_row << 16) + packed_col_id + sub * 8, cutlass.Int8),
                    packed_state[0:8],
                )
                if cutlass.const_expr(cfg.enable_checkpoints):
                    # stage this sub's state to sH TRANSPOSED (KV: k rows, v
                    # contiguous in 64-v slabs, swizzled — the GDN H layout);
                    # each thread scatters its 16 k values down one v column
                    if do_h:
                        h_col = value_dim % cutlass.Int32(64)
                        h_seg = (value_dim // cutlass.Int32(64)) * (cfg.d_k * cutlass.Int32(64))
                        k_base = sub * 16
                        for j in cutlass.range_constexpr(16):
                            hv = state_blocks[sub][j].to(cfg.io_dtype)
                            (sH_ptr + h_seg + cutlass.Int32((k_base + j) * 64) + swizzle_xor_128b(cutlass.Int32(k_base + j), h_col, elem_bytes=2)).store(hv)

            nvvm.tcgen05_wait("store")
            bars.mb_state_inp_ready.arrive()
            if cutlass.const_expr(cfg.enable_checkpoints):
                if do_h:
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_h_tmastg_ready.arrive()
            bars.mb_o_acc_ready.wait(o_acc_index.phase)
            o_acc_index = advance(o_acc_index, 1)
            projection_col_id = tmem_col + cfg.tmem_q_state_acc_offset + prev_q_state_acc_stage * cfg.b_t
            value_dim_base = tmem_sp * cfg.threads_per_warp

            row_id0 = tmem_row + value_dim_base
            row_id1 = row_id0 + 16
            loaded0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)
            loaded1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)

            # ---- output drain: q_state_acc -> scaled b16 -> SMEM stmatrix -------
            stsm_regs0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            stsm_regs1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                scaled0_0, scaled0_1 = fmul2(loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1], scale, scale)
                scaled1_0, scaled1_1 = fmul2(loaded1[2 * reg_idx], loaded1[2 * reg_idx + 1], scale, scale)
                stsm_regs0[reg_idx] = fp32_to_fp16(scaled0_0, scaled0_1, dtype=mO.element_type)
                stsm_regs1[reg_idx] = fp32_to_fp16(scaled1_0, scaled1_1, dtype=mO.element_type)

            nvvm.stmatrix(
                sO_ptr
                + prev_o_stage_base
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                stsm_regs0.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                sO_ptr
                + prev_o_stage_base
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                stsm_regs1.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            # release only after the stmatrix pair: the STSM->F2FP->FMUL2->LDTM
            # register chain pins the TMEM reads complete without a wait("load")
            bars.mb_o_acc_done[prev_q_state_acc_stage].arrive()
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_o_tmastg_ready[prev_o_stage].arrive()

            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
            # ---- rhs staging: rhs input = beta * (v - state*k) -----------------
            projection_col_id = tmem_col + cfg.tmem_state_k_acc_offset
            input_col_id = tmem_col + cfg.tmem_rhs_inp_offset
            value_dim_base = tmem_sp * cfg.threads_per_warp

            # ---- read back state*k acc + raw v fragments -----------------------
            row_id0 = tmem_row + value_dim_base
            state_k0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)

            row_id1 = row_id0 + 16
            state_k1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)

            raw_v_regs0 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_regs1 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            nvvm.tcgen05_wait("load")

            packed_rhs0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                packed_col = (reg_idx // 2) * 4 + (lane & 3)
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token1).load().to(cutlass.Float32)
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k0[frag_pair], state_k0[frag_pair + 1]
                beta_pair = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_regs0[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                packed_rhs0[reg_idx] = mul_f16x2(
                    beta_pair,
                    diff_pair,
                    cfg.io_dtype,
                )

            packed_rhs1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                packed_col = (reg_idx // 2) * 4 + (lane & 3)
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token1).load().to(cutlass.Float32)
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k1[frag_pair], state_k1[frag_pair + 1]
                beta_pair = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_regs1[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                packed_rhs1[reg_idx] = mul_f16x2(
                    beta_pair,
                    diff_pair,
                    cfg.io_dtype,
                )

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr((tmem_row << 16) + input_col_id, cutlass.Int8), packed_rhs0[0:4])

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr((tmem_row + 16 << 16) + input_col_id, cutlass.Int8), packed_rhs1[0:4])

            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            bars.mb_inputs_done[raw_index.idx].arrive()
            bars.mb_rhs_ready.arrive()

            # ---- update readback -> packed b16 A operand -----------------------
            bars.mb_update_acc_ready.wait(update_acc_index.phase)
            update = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr((tmem_row + tmem_sp * cfg.threads_per_warp << 16) + (tmem_col + cfg.tmem_update_acc_offset), cutlass.Float32),
                num=cfg.b_t,
            )
            nvvm.tcgen05_wait("load")

            packed_update = cutlass.Array(cutlass.Int32, (cfg.b_t // 2), alignment=16)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                packed_update[packed_col] = fp32_to_fp16(update[token0], update[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr((tmem_row << 16) + (tmem_col + cfg.tmem_update_inp_offset), cutlass.Int8),
                packed_update[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_wait("store")
            update_acc_index = advance(update_acc_index, 1)
            bars.mb_update_ready.arrive()

            bars.mb_k_restore_done[kr_index.idx].wait(kr_index.phase)
            kr_index = advance(kr_index, cfg.smem_decay_stages)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        if sk_nt > 0:
            og = gbase + sk_nt - cutlass.Int32(1)
            final_o_stage = og % cfg.smem_o_stages
            final_q_state_acc_stage = og % cfg.tmem_q_state_acc_stages
            final_o_stage_base = final_o_stage * (cfg.b_t * cfg.d_v)
            bars.mb_o_tmastg_done[final_o_stage].wait(((og // cfg.smem_o_stages) + 1) % 2)

            bars.mb_o_acc_ready.wait(o_acc_index.phase)
            o_acc_index = advance(o_acc_index, 1)
            projection_col_id = tmem_col + cfg.tmem_q_state_acc_offset + final_q_state_acc_stage * cfg.b_t
            value_dim_base = tmem_sp * cfg.threads_per_warp

            row_id0 = tmem_row + value_dim_base
            row_id1 = row_id0 + 16
            loaded0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)
            loaded1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)

            # ---- output drain: q_state_acc -> scaled b16 -> SMEM stmatrix -------
            stsm_regs0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            stsm_regs1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                scaled0_0, scaled0_1 = fmul2(loaded0[2 * reg_idx], loaded0[2 * reg_idx + 1], scale, scale)
                scaled1_0, scaled1_1 = fmul2(loaded1[2 * reg_idx], loaded1[2 * reg_idx + 1], scale, scale)
                stsm_regs0[reg_idx] = fp32_to_fp16(scaled0_0, scaled0_1, dtype=mO.element_type)
                stsm_regs1[reg_idx] = fp32_to_fp16(scaled1_0, scaled1_1, dtype=mO.element_type)

            nvvm.stmatrix(
                sO_ptr
                + final_o_stage_base
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                stsm_regs0.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                sO_ptr
                + final_o_stage_base
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                stsm_regs1.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            # release only after the stmatrix pair: the STSM->F2FP->FMUL2->LDTM
            # register chain pins the TMEM reads complete without a wait("load")
            bars.mb_o_acc_done[final_q_state_acc_stage].arrive()
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_o_tmastg_ready[final_o_stage].arrive()

        # split-K: only the item owning the sequence's last chunk holds the
        # true end-of-sequence state (legacy tiles always do: wend == nc)
        owns_final = wend == num_chunks_b
        # ---- final-state drain: final_state acc -> GMEM --------------------
        if cutlass.const_expr(mS_out is not None):
            if seqlen_b > 0:
                if owns_final:
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        loaded = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr((row_id << 16) + (tmem_col + cfg.tmem_state_offset + key_block_start), cutlass.Float32),
                            num=32,
                        )
                        nvvm.tcgen05_wait("load")

                        for col in cutlass.range_constexpr(32):
                            key_dim = key_block_start + col
                            mS_out[batch_idx, head_o, key_dim, value_dim] = loaded[col].to(mS_out.element_type)
            else:
                # zero-length sequence: the state passes through untouched
                # (S0 when seeded, zeros otherwise); pure GMEM, no TMEM
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for col in cutlass.range_constexpr(32):
                        key_dim = key_block_start + col
                        if cutlass.const_expr(mS_init is not None):
                            mS_out[batch_idx, head_o, key_dim, value_dim] = mS_init[batch_idx, head_o, key_dim, value_dim]
                        else:
                            mS_out[batch_idx, head_o, key_dim, value_dim] = cutlass.Float32(0.0).to(mS_out.element_type)
        bars.mb_final_state_stored.arrive()
        gbase += sk_nt
        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def _host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    sched_ctr: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    stream,
) -> None:
    num_sequences = cu_seqlens.shape[0] - 1
    ho = raw_gate.shape[1]
    # ---- persistent launch: the grid only needs to cover the tiles ------
    total_tiles = num_sequences * ho
    # CUDA-graph-stable launch: fixed SM-count grid
    grid_shape = (cfg.max_active_clusters, 1, 1)
    _kernel(
        cfg,
        tensormap_workspace,
        cutlass.Int32(num_sequences * ho),
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        out,
        final_state,
        work_items,
        work_count,
        sched_ctr,
        total_tiles,
        scale,
        checkpoint_every_n_tokens,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.kernel
def _kernel(
    cfg: cutlass.Constexpr,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mS_init: cute.Tensor | None,
    mO: cute.Tensor,
    mS_out: cute.Tensor | None,
    mWorkItems: cute.Tensor | None,
    mCount: cute.Tensor | None,
    mSched: cute.Tensor | None,
    total_tiles: cutlass.Int32,
    scale: cutlass.Float32,
    checkpoint_every_n_tokens: cutlass.Int32,
) -> None:
    """BT=16 KDA forward kernel (persistent).

    Grid: `(min(tiles, SM count), 1, 1)`.  Every warp role runs a
    tile-scheduler loop over the tiles — one packed sequence/head each, or
    one split-K work item — iterating its chunks in order.  Source heads
    follow repeat_interleave: head_x = head_idx // X_RATIO.
    """

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % cfg.threads_per_warp

    if cutlass.const_expr(cfg.split_k):
        total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None and mSched.element_type == cutlass.Int32
    assert mQ.element_type == cfg.io_dtype and mK.element_type == cfg.io_dtype and mV.element_type == cfg.io_dtype
    assert mGate.element_type == cutlass.Float32
    beta_expected = cfg.io_dtype if cutlass.const_expr(cfg.beta_sigmoid) else cutlass.Float32
    assert mBeta.element_type == beta_expected
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    if cutlass.const_expr(cfg.use_initial_state):
        assert mS_init is not None and mS_init.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mS_init is None, "mS_init must be None if use_initial_state is False"
    if cutlass.const_expr(cfg.store_final_state):
        assert mS_out is not None and mS_out.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mS_out is None, "mS_out must be None if store_final_state is False"
    if cutlass.const_expr(mS_init is not None and mS_out is not None):
        assert mS_init.element_type == mS_out.element_type
    # per-(batch, head) TMA-descriptor arrays: [q, k, v, gate, o]
    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_h_base = desc_base_words + cutlass.Int32(5) * arr_words

    # Buffers are declaration-ordered and intentionally non-aliased.
    SMEM = cutlass.AddressSpace.smem
    bars = make_kda_bars(cfg)
    # The hand-written K-box-major SW128 mapping is normalized to phase 0,
    # so both tcgen05 and ldmatrix can share 1KB-aligned operand buffers.
    tmem_hold = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    sK_decay_raw = cutlass.Array(cfg.io_dtype, cfg.k_decay_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_decay_raw = cutlass.Array(cfg.io_dtype, cfg.q_decay_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_restore_raw = cutlass.Array(cfg.io_dtype, cfg.k_restore_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sPairwise_raw = cutlass.Array(cfg.io_dtype, cfg.pairwise_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_raw = cutlass.Array(mQ.element_type, cfg.q_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_raw = cutlass.Array(mK.element_type, cfg.k_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sV_raw = cutlass.Array(mV.element_type, cfg.v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sGate_raw = cutlass.Array(cutlass.Float32, cfg.gate_cosize, space=SMEM, alignment=1024)
    sState_scale_diag_raw = cutlass.Array(cfg.io_dtype, cfg.state_scale_diag_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_inv_raw = cutlass.Array(cfg.io_dtype, cfg.k_inv_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sO_raw = cutlass.Array(
        mO.element_type,
        cfg.o_cosize,
        space=SMEM,
        # The scalar CG1 store computes W128 offsets relative to this buffer.
        # Align to the full s128b period so absolute SMEM address bits do not
        # add a hidden phase to the TMA store-side swizzle.
        alignment=cfg.buffer_align_bytes,
    )
    sBeta_raw = cutlass.Array(cutlass.Float32, cfg.beta_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sH_raw = (
        cutlass.Array(cfg.io_dtype, cfg.d_k * cfg.d_v, space=SMEM, alignment=cfg.buffer_align_bytes) if cutlass.const_expr(cfg.enable_checkpoints) else sO_raw
    )
    # K-box-major SW128 staging: 16B leading offset, 1KB stride; the decay
    # stores apply a row-group key xor so tcgen05 B reads logical [DK, BT].
    sK_decay = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sQ_decay = SmemTile(
        base=sQ_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sK_restore = SmemTile(
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
    sPairwise = SmemTile(
        base=sPairwise_raw,
        elems_per_stage=(2 * cfg.b_t * cfg.b_t),
        stages=cfg.smem_pairwise_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * cfg.b_t * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )
    tma_tx_bytes = cutlass.const_expr(
        cfg.d_k * cfg.b_t * mQ.element_type.width // 8
        + cfg.d_k * cfg.b_t * mK.element_type.width // 8
        + cfg.d_v * cfg.b_t * mV.element_type.width // 8
        + cfg.d_k * cfg.b_t * mGate.element_type.width // 8
    )

    if warp_idx == cfg.tma_warp_id:
        if nvvm.elect_sync():
            bars.mb_tma_done.init()
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_inputs_ready[stage].init()
                bars.mb_inputs_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if nvvm.elect_sync():
            bars.mb_o_acc_ready.init()
            for stage in cutlass.range_constexpr(cfg.tmem_q_state_acc_stages):
                bars.mb_o_acc_done[stage].init()
            bars.mb_state_k_acc_ready.init()
            bars.mb_update_acc_ready.init()
            bars.mb_state_inp_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_state_scale_diag_stages):
                bars.mb_state_scale_diag_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_kk_qk_super_mma_done[stage].init()
                bars.mb_kk_qk_mma_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_rhs_ready.init()
            bars.mb_update_ready.init()
            bars.mb_final_state_stored.init()
    elif warp_idx == cfg.super_mma_warp_id:
        if nvvm.elect_sync():
            for stage in cutlass.range_constexpr(cfg.smem_pairwise_stages):
                bars.mb_a_ready[stage].init()
                bars.mb_qk_acc_ready[stage].init()
                bars.mb_a_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.qk_scale_ready_stages):
                bars.mb_qk_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_cg0_ready[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if nvvm.elect_sync():
            for stage in cutlass.range_constexpr(cfg.smem_o_stages):
                bars.mb_o_tmastg_ready[stage].init()
                bars.mb_o_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.sched_stages):
                bars.mb_sched_ready[stage].init()
                bars.mb_sched_done[stage].init()
            if cutlass.const_expr(cfg.enable_checkpoints):
                bars.mb_h_tmastg_ready.init()
                bars.mb_h_tmastg_done.init()
    diag_zero = cfg.io_dtype(0.0)
    for diag_idx in cutlass.range(tidx, cfg.state_scale_diag_cosize, cfg.threads_per_cta, unroll=1):
        sState_scale_diag_raw[diag_idx] = diag_zero
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0, thread_count=cfg.threads_per_cta)
    if (warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]) or warp_idx == cfg.tcgen05_mma_warp_id:
        if warp_idx == cfg.tcgen05_mma_warp_id:
            nvvm.tcgen05_alloc(tmem_hold, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync(cfg.nbar_tmem_lifecycle_id, thread_count=cfg.tmem_user_threads)
        if warp_idx == cfg.tcgen05_mma_warp_id:
            nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
        nvvm.barrier_cta_sync(cfg.nbar_tmem_lifecycle_id, thread_count=cfg.tmem_user_threads)

    # Actual SMEM/TMEM buffers for the BT=16 schedule:
    #   q/k/v              : 16 x 128 each
    #   gate_log2          : 16 x 128
    #   beta               : 16
    #   q/k inverse norm   : 16 each, staged once per decay stage
    #   exp_g_last         : 128, CG0-local and staged once per decay stage
    #   state-scale diag   : 8 x 16 x 16 input dtype, zeroed once in the prologue
    #   q_decay/k_decay    : tcgen05 SW128 operands shared with super-MMA
    #   k_restore          : tcgen05 SW128 N-major final-state operand
    #   k_inv              : 16 x 128 token-major for super-MMA RHS
    #   A inverse/QK       : 16 x 16 each, plus transposed tcgen05 operands
    #   state              : external/kernel ABI is VK `[DV, DK]`; reference
    #                        math can view it as KV `[DK, DV]` by transposing.
    #                        The TS A-staging path keeps VK in TMEM so state*k
    #                        is `[DV, DK] @ [DK, BT] -> [DV, BT]` with M=128.

    if warp_idx == cfg.tma_warp_id:
        _tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mSched,
            sSched,
            tma_tx_bytes,
            lane,
            mBeta,
            sQ_raw,
            sK_raw,
            sV_raw,
            sGate_raw,
            sBeta_raw,
            desc_q_base,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
            bars,
        )
    elif warp_idx == cfg.super_mma_warp_id:
        _super_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            lane,
            sK_inv_raw,
            sPairwise_raw,
            sBeta_raw,
            sK_decay_raw,
            bars,
        )
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        _tcgen05_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            tmem_hold,
            sPairwise,
            sK_decay,
            sK_restore,
            sQ_decay,
            sState_scale_diag,
            bars,
        )
    elif warp_idx == cfg.epilogue_warp_id:
        _epilogue_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            lane,
            mO,
            sK_inv_raw,
            sO_raw,
            sPairwise_raw,
            sQ_decay_raw,
            sH_raw,
            desc_o_base,
            desc_h_base,
            checkpoint_every_n_tokens,
            bars,
        )
    elif warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        _compute0_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            lane,
            warp_idx,
            mQ,
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_raw,
            sK_raw,
            sQ_raw,
            sV_raw,
            sK_decay_raw,
            sK_restore_raw,
            sQ_decay_raw,
            sState_scale_diag_raw,
            bars,
        )
    elif warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        _compute1_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            lane,
            tmem_hold,
            warp_idx,
            mS_out,
            mS_init,
            mO,
            sO_raw,
            sBeta_raw,
            sV_raw,
            sH_raw,
            checkpoint_every_n_tokens,
            scale,
            bars,
        )


@dataclass
class KdaCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``; per-stage sizes are
    inlined at the use sites).  Passed ``cfg``-first (a ``cutlass.Constexpr``)
    into ``_host`` / ``_kernel`` and every warp body, mirroring GDN's
    ``GdnCfg``."""

    io_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    # split-K: tiles come from a work-item table (see common/split_k.py);
    # each item computes chunks [cstart, wend) and writes only [wstart, wend)
    split_k: bool = False
    dyn_sched: bool = False
    sched_stages: int = CFG.SMEM_SCHED_STAGES

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
    nbar_cg0_group0_id: int = 1  # CG0 group g syncs on nbar id 1 + g
    tmem_user_threads: int = 0
    nbar_tmem_lifecycle_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # --- SMEM / TMEM ring stage counts ---
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_pairwise_stages: int = CFG.SMEM_PAIRWISE_STAGES
    smem_state_scale_diag_stages: int = CFG.SMEM_STATE_SCALE_DIAG_STAGES
    qk_scale_ready_stages: int = CFG.QK_SCALE_READY_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES

    # --- TMEM column offsets (state doubles as the final_state acc) ---
    tmem_state_offset: int = 0
    tmem_state_inp_offset: int = 0
    tmem_q_state_acc_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_update_acc_offset: int = 0
    tmem_rhs_inp_offset: int = 0
    tmem_update_inp_offset: int = 0

    # --- SMEM buffer cosizes ---
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    gate_cosize: int = 0
    beta_cosize: int = 0
    k_inv_cosize: int = 0
    k_decay_cosize: int = 0
    q_decay_cosize: int = 0
    k_restore_cosize: int = 0
    state_scale_diag_cosize: int = 0
    o_cosize: int = 0
    pairwise_cosize: int = 0


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
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    split_k: bool = False,
    dyn_sched: bool = False,
) -> KdaCfg:
    """Build the per-compile ``KdaCfg`` (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = KdaCfg(
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        split_k=split_k,
        dyn_sched=dyn_sched,
    )
    if enable_checkpoints:
        # the 32 KB H staging buffer must fit next to the raw ring: trim the
        # q/k/v/gate TMA lookahead for H compiles
        cfg.smem_raw_stages = 6
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.cg0_threads_per_group = cfg.cg0_warps_per_group * cfg.threads_per_warp
    cfg.tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids)) * cfg.threads_per_warp
    if cfg.smem_state_scale_diag_stages != cfg.qk_scale_ready_stages:
        raise ValueError("diag and qk-scale ready rings must share their rolling stage")

    cfg.tmem_state_inp_offset = cfg.tmem_state_offset + cfg.d_k
    cfg.tmem_q_state_acc_offset = cfg.tmem_state_inp_offset + (cfg.d_k // 2)
    cfg.tmem_state_k_acc_offset = cfg.tmem_q_state_acc_offset + cfg.tmem_q_state_acc_stages * cfg.b_t
    cfg.tmem_update_acc_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    cfg.tmem_rhs_inp_offset = cfg.tmem_update_acc_offset + cfg.b_t
    cfg.tmem_update_inp_offset = cfg.tmem_rhs_inp_offset + (cfg.b_t // 2)
    assert (cfg.tmem_update_inp_offset + (cfg.b_t // 2)) <= 512

    cfg.q_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.k_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.beta_cosize = cfg.smem_raw_stages * cfg.b_t
    cfg.k_inv_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.q_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.state_scale_diag_cosize = cfg.smem_state_scale_diag_stages * (cfg.d_k // 16) * 256
    cfg.o_cosize = cfg.smem_o_stages * cfg.b_t * cfg.d_v
    cfg.pairwise_cosize = cfg.smem_pairwise_stages * 2 * cfg.b_t * cfg.b_t
    return cfg


def get_workspace_size(B: int, HQ: int, HV: int) -> int:
    """Bytes for the per-(batch, head) TMA-descriptor arrays (q, k, v, gate,
    o, h) + 128 alignment slack."""
    HO = HQ if HQ >= HV else HV
    return TENSOR_MAP_QWORDS * 8 * (6 * B * HO) + 128


@cute.jit
def _build_descs(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    h: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    h_every_n: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    """Build the 6 per-(batch, head) TMA-descriptor arrays (q, k, v, gate,
    o, h) into ``tensormap_workspace``.

    Launched on every execute: the descriptors fold cu_seqlens contents into
    GLOBAL_ADDRESS and GLOBAL_DIM, which the host cannot read without a D2H sync.  Each descriptor
    folds the sequence base + head offset into GLOBAL_ADDRESS (Int64) and
    caps the token GLOBAL_DIM to the sequence length, so the main kernel's
    coordinates are sequence-relative and tail chunks clip in hardware.  The
    H descriptor is 3-D ``(dv, dk, entry)`` over the packed ``[total_h, HO,
    DK, DV]`` series; its per-sequence entry offsets ((seqlen-1)//N,
    prefix-summed) are derived on device and its entry extent is capped per
    sequence, so H store coordinates are sequence-local."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    granu = 128 // bpe
    seqlen = q.shape[0]

    def _head0(t, dim, heads):
        # 2-D (dim, token) head-0 view: box (granu, b_t) matches the main
        # kernel's SMEM staging byte-for-byte
        return cute.make_tensor(t.iterator, cute.make_layout((dim, seqlen), stride=(1, heads * dim)))

    swz = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(_head0(q, d_k, h_q), box_dims=(granu, b_t), stride_order=(0, 1), swizzle=swz)
    base_k = cuda.create_tensor_map_tiled_from_view(_head0(k, d_k, h_k), box_dims=(granu, b_t), stride_order=(0, 1), swizzle=swz)
    base_v = cuda.create_tensor_map_tiled_from_view(_head0(v, d_v, h_v), box_dims=(granu, b_t), stride_order=(0, 1), swizzle=swz)
    base_gate = cuda.create_tensor_map_tiled_from_view(_head0(gate, d_k, ho), box_dims=(32, b_t), stride_order=(0, 1), swizzle=swz)
    base_o = cuda.create_tensor_map_tiled_from_view(_head0(o, d_v, ho), box_dims=(granu, b_t), stride_order=(0, 1), swizzle=swz)

    arr_words = (batch_size * ho) * TENSOR_MAP_QWORDS
    ws_iter = tensormap_workspace.iterator

    def _sub(i):
        return cute.make_tensor(ws_iter + i * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    build_qkv_load_descs_kernel(
        base_q, _sub(0), cu_seqlens, q, cutlass.Int32(batch_size), cutlass.Int32(ho), cutlass.Int32(ho // h_q), cutlass.Int32(d_k), cutlass.Int32(h_q * d_k), 1
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_k, _sub(1), cu_seqlens, k, cutlass.Int32(batch_size), cutlass.Int32(ho), cutlass.Int32(ho // h_k), cutlass.Int32(d_k), cutlass.Int32(h_k * d_k), 1
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_v, _sub(2), cu_seqlens, v, cutlass.Int32(batch_size), cutlass.Int32(ho), cutlass.Int32(ho // h_v), cutlass.Int32(d_v), cutlass.Int32(h_v * d_v), 1
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_gate, _sub(3), cu_seqlens, gate, cutlass.Int32(batch_size), cutlass.Int32(ho), cutlass.Int32(1), cutlass.Int32(d_k), cutlass.Int32(ho * d_k), 1
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    build_qkv_load_descs_kernel(
        base_o, _sub(4), cu_seqlens, o, cutlass.Int32(batch_size), cutlass.Int32(ho), cutlass.Int32(1), cutlass.Int32(d_v), cutlass.Int32(ho * d_v), 1
    ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)
    if cutlass.const_expr(h is not None):
        h_view = cute.make_tensor(
            h.iterator,
            cute.make_layout(
                (h.shape[3], h.shape[2], h.shape[0]),
                stride=(h.stride[3], h.stride[2], h.stride[0]),
            ),
        )
        base_h = cuda.create_tensor_map_tiled_from_view(h_view, box_dims=(granu, d_k, 1), stride_order=(0, 1, 2), swizzle=swz)
        build_h_descs_kernel(
            base_h,
            _sub(5),
            cu_seqlens,
            h,
            cutlass.Int32(batch_size),
            cutlass.Int32(ho),
            cutlass.Int32(h.stride[1]),
            cutlass.Int32(h.stride[0]),
            h_every_n,
            2,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


# ---------------------------------------------------------------------------
# Torch adapter / host-side compilation
# ---------------------------------------------------------------------------


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


def _data_ptr(t) -> int:
    """Device address of a tensor-like (``data_ptr()`` or the CUDA array
    interface)."""
    fn = getattr(t, "data_ptr", None)
    if fn is not None:
        return fn()
    return t.__cuda_array_interface__["data"][0]


def _cutlass_io_dtype(dtype):
    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float16" in name or "half" in name:
        return cutlass.Float16
    raise ValueError(f"Unsupported dtype {dtype}, expected bfloat16 or float16")


def _cutlass_state_dtype(dtype):
    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float32" in name:
        return cutlass.Float32
    raise ValueError(f"Unsupported state dtype {dtype}, expected float32 or bfloat16")


@lru_cache(maxsize=None)
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    cu_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    beta_sigmoid: bool,
    split_k: bool,
    dyn_sched: bool,
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
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    split_k: bool = False,
    dyn_sched: bool = False,
    *,
    num_sm: int,
    q_cute,
    k_cute,
    v_cute,
    gate_cute,
    a_log_cute,
    dt_bias_cute,
    beta_cute,
    cu_seqlens_cute,
    s_in_cute,
    o_cute,
    s_out_cute,
    work_items_cute=None,
    work_count_cute=None,
    sched_ctr_cute=None,
    tensormap_ws_cute,
    checkpoint_every_n_tokens,
    scale,
    stream,
):
    """JIT-compile the chunked KDA prefill kernel for one static config."""
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
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=num_sm,
        split_k=split_k,
        dyn_sched=dyn_sched,
    )

    return cute.compile(
        _host,
        cfg,
        q_cute,
        k_cute,
        v_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        beta_cute,
        cu_seqlens_cute,
        s_in_cute,
        o_cute,
        s_out_cute,
        work_items_cute,
        work_count_cute,
        sched_ctr_cute,
        tensormap_ws_cute,
        checkpoint_every_n_tokens,
        scale,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_kda_sm100(
    q,
    k,
    v,
    gate,
    beta,
    output,
    cu_seqlens,
    initial_state,
    output_state,
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    output_checkpoints=None,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid_in_kernel: bool = False,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    *,
    tensormap_workspace,
    stream,
) -> None:
    """Execute the Blackwell BT=16 chunked KDA prefill kernel.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO, DK)`` float32.  Natural-log decay unless
              ``safe_gate``, which applies the safe-gate transform
              ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``.
        beta: ``(total_tokens, HO)``.  Post-sigmoid float32, or io-dtype
              logits when ``use_beta_sigmoid_in_kernel``
        output: ``(total_tokens, HO, DV)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DV)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DK, DV)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: emit an H entry every N tokens (0 = off).
            H[j] is the state after ``(j + 1) * N`` tokens, STRICTLY BEFORE
            the sequence end — the end-of-sequence state is only
            ``output_state``.  With ``N == B_T`` this is the per-chunk state
            series the backward pass consumes.
        output_checkpoints: ``(total_h, HO, DK, DV)`` io-dtype (KV, v
            contiguous — the GDN H layout); the per-sequence entry offsets
            are derived on device from ``cu_seqlens`` ((seqlen-1)//N,
            prefix-summed), so there is no cu_checkpoints array
        use_qk_l2norm_in_kernel: L2-normalize q/k rows inside the kernel
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO, DK)`` float32, safe-gate channel bias (None = 0)
        use_beta_sigmoid_in_kernel: ``beta`` holds logits; sigmoid in-kernel
        work_items: ``(max_items, 6)`` int32 split-K work-item table from
            ``common/split_k.py``, or None for the one-tile-per-(b,h)
            layout.  With a table, each item computes chunks
            ``[cstart, wend)`` and writes O/checkpoints only for
            ``[wstart, wend)``.
        work_count: ``(1,)`` int32 device-side item count (required with
            work_items)
        sched_ctr: ``(2,)`` int32 device scratch ``[ticket, done]`` enabling
            the dynamic (work-stealing) tile scheduler; must be zeroed before
            every launch (``build_split_table`` does this when it is passed as
            ``sched_ctr``).  None keeps the static CTA stride.
    """
    HQ = q.shape[1]
    HK = k.shape[1]
    HV = v.shape[1]
    HO = max(HQ, HV)
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if enable_checkpoints:
        if output_checkpoints is None:
            raise ValueError("checkpoint_every_n_tokens > 0 requires output_checkpoints")
        if str(output_checkpoints.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
            raise ValueError(
                f"output_checkpoints dtype must match the io dtype (fp32 state belongs to output_state): got {output_checkpoints.dtype} with io {q.dtype}"
            )
    split_k = work_items is not None
    dyn_sched = sched_ctr is not None
    if split_k:
        if work_count is None:
            raise ValueError("work_count is required with work_items")
        if enable_checkpoints and checkpoint_every_n_tokens != CFG.B_T:
            raise ValueError(f"split-K checkpoints require checkpoint_every_n_tokens == {CFG.B_T}, got {checkpoint_every_n_tokens}")
    elif work_count is not None:
        raise ValueError("work_count must be None without work_items")

    if initial_state is not None:
        state_dtype_src = initial_state.dtype
    elif output_state is not None:
        state_dtype_src = output_state.dtype
    else:
        state_dtype_src = "float32"

    for name, h in (("HQ", HQ), ("HK", HK), ("HV", HV)):
        if HO % h != 0:
            raise ValueError(f"{name}={h} must divide sab heads {HO}")
    q_ratio = HO // HQ
    k_ratio = HO // HK
    v_ratio = HO // HV
    gate_scale_log2 = gate_lower_bound * LOG2_E

    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if not safe_gate:
        a_log = None
        dt_bias = None
    if _data_ptr(tensormap_workspace) % 128 != 0:
        raise ValueError("tensormap_workspace must be 128-byte aligned")
    cu_stream = cuda_driver.CUstream(int(stream))

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_dtype_src),
        str(cu_seqlens.dtype),
        HQ,
        HK,
        HV,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid_in_kernel,
        split_k,
        dyn_sched,
    )

    if "compiled" not in cache:
        io_dtype = _cutlass_io_dtype(q.dtype)
        state_dtype = _cutlass_state_dtype(state_dtype_src)
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        a_log_cute = from_dlpack(a_log, assumed_align=4) if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16) if dt_bias is not None else None
        beta_cute = from_dlpack(beta, assumed_align=4)
        beta_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(initial_state, assumed_align=16)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=CFG.D_K)

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(output_state, assumed_align=16)
            s_out_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=CFG.D_K)

        work_items_cute = None
        work_count_cute = None
        if split_k:
            work_items_cute = from_dlpack(work_items, assumed_align=4)
            work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()

        sched_ctr_cute = None
        if dyn_sched:
            sched_ctr_cute = from_dlpack(sched_ctr, assumed_align=4).mark_layout_dynamic()

        tensormap_ws_cute = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()

        cache["compiled"] = compile(
            io_dtype,
            state_dtype,
            use_initial_state,
            store_final_state,
            enable_checkpoints,
            use_qk_l2norm_in_kernel,
            safe_gate,
            gate_scale_log2,
            use_beta_sigmoid_in_kernel,
            q_ratio,
            k_ratio,
            v_ratio,
            HO,
            split_k,
            dyn_sched,
            num_sm=_device_sm_count(),
            q_cute=q_cute,
            k_cute=k_cute,
            v_cute=v_cute,
            gate_cute=gate_cute,
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            beta_cute=beta_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            s_in_cute=s_in_cute,
            o_cute=o_cute,
            s_out_cute=s_out_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            sched_ctr_cute=sched_ctr_cute,
            tensormap_ws_cute=tensormap_ws_cute,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            scale=scale,
            stream=cu_stream,
        )

    compiled = cache["compiled"]

    # The descriptors encode cu_seqlens' CONTENTS, which no key built from the
    # buffers can track. The skip this replaces asked torch's _version counter,
    # so it was sound for a torch caller and silently stale for every other
    # producer. Rebuilding unconditionally measures free: 131 vs 135 us of host
    # time, and 157 either way once the launches are waited on.
    h_for_descs = output_checkpoints if enable_checkpoints else None
    if cache.get("build_descs_has_h") != (h_for_descs is not None):
        cache.pop("build_descs", None)
        cache["build_descs_has_h"] = h_for_descs is not None
    if "build_descs" not in cache:
        io_dtype = _cutlass_io_dtype(q.dtype)
        q_bd = from_dlpack(q, assumed_align=16)
        q_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        k_bd = from_dlpack(k, assumed_align=16)
        k_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        v_bd = from_dlpack(v, assumed_align=16)
        v_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        gate_bd = from_dlpack(gate, assumed_align=16)
        gate_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        o_bd = from_dlpack(output, assumed_align=16)
        o_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        cu_bd = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        ws_bd = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()
        h_bd = None
        if h_for_descs is not None:
            h_bd = from_dlpack(h_for_descs, assumed_align=16)
            h_bd.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3), divisibility=1)
        cache["build_descs"] = cute.compile(
            _build_descs,
            io_dtype,
            CFG.B_T,
            q_bd,
            k_bd,
            v_bd,
            gate_bd,
            o_bd,
            h_bd,
            cu_bd,
            ws_bd,
            cutlass.Int32(checkpoint_every_n_tokens),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["build_descs"](
        q,
        k,
        v,
        gate,
        output,
        h_for_descs,
        cu_seqlens,
        tensormap_workspace,
        checkpoint_every_n_tokens,
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
        cu_seqlens,
        initial_state if use_initial_state else None,
        output,
        output_state if store_final_state else None,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        checkpoint_every_n_tokens,
        scale,
        cu_stream,
    )
