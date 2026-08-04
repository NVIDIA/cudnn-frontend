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
Chunked Gated Delta Net (GDN) prefill kernel for Blackwell SM100 (Cutlass primitives)
with optional per-chunk state (H) output.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], gate[BT] (scalar gate), beta[BT] (scalar LR)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (compute warp group 0):
    cumsumlog[t]     = sum_{l=0}^{t} log(gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates
    T_pairwise[i,j]  = cumprod[i] / cumprod[j]  (i>=j)       inter-token transfer weights
    (stored in registers; 128 regs/thread)

  GEMM 1 - kk   : W_kk[BT,BT]  = K  @ K^T       (lower-triangular intra scores)
  GEMM 2 - qk   : W_qk[BT,BT]  = Q  @ K^T       (output attention scores)
  GEMM 3 - k*state : KS[BT,DV] = K  @ S_prev    (key applied to state)
  GEMM 4 - q*state : QS[BT,DV] = Q  @ S_prev    (inter-chunk output, before T scaling)
  GEMM 5 - new v   : NV[BT,DV] = A_inv @ V       (corrected value vectors)
                      where A_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*beta[i]*W_kk[i,j]  (lower-tri, hierarchical blockwise inverse)
  GEMM 6 - qkv  : O_intra[BT,DV] = W_qkv @ NV   (intra-chunk output)
                   where W_qkv = T*beta*W_qk (scaled qk scores)
  GEMM 7 - kv update : dS[DK,DV] = K^T @ delta       (state update, BT contraction)
                        where delta[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    O[BT,DV]  = O_intra + T_col * QS             (combine intra + inter)
    S_next    = cumprod[BT-1] * S_prev + dS        (update state in TMEM)

Chunks run in PAIRS (CG0 warp halves invert chunk 0 / chunk 1 in parallel);
odd counts pad with a neutral zero-filled chunk.

SMEM layout (227 KB = full; stage counts live in gdn_prefill_config.py;
enable_h compiles trim K/V stages to fit the H buffer):
  Buffer                       Size (B)  Stages
  q                               16384       3
  k                               16384       4
  v                               16384       3
  A_inverse / new_v                8192       2
  QK output                        8192       2
  O store                         16384       2
  H staging                     DK*DV*2       1    <-- enable_h only
  cumsumlog / cumprod / beta        256       3
  sched ticket ring                   4       2    <-- dyn_sched publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  state (S)               128     <-- DKxDV fp32 = 128x128x4B
  q*state / O acc          64     <-- BTxDV fp32 accumulator
  state inp                64     <-- fp16 state staging (GEMMs 3/4 A operand)
  cg0 shared acc          128     <-- 2-stage ring: KK0/KK1 then QK0/QK1
  cg1 shared acc           64     <-- 1-stage ring: KS then NV
  vks+nv / decay_v inp     64     <-- slot 0 = VKS then NV, slot 1 = decay_v

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise x2, kk_epi x2, pair inverse,
                                    qk_epi x2
  warps 4-7     : compute group 1 - state restage/rescale, v-k*state,
                                    state*q_epi, new_v_epi, qkv_epilogue
  warp  8       : CG0 MMA issuer - KK0/KK1/QK0/QK1 per pair; TMEM lifecycle
  warp  9       : TMA load warp  - loads q, k, v
  warp  10      : CG1 MMA issuer - KS/QS/NV/QKV/KV per chunk
  warp  11      : epilogue warp   - gate/beta loads (4-chunk lookahead) +
                                    O then H TMA stores
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

from ..common.thd import build_h_descs_kernel, build_qkv_load_descs_kernel, downcast_state_kernel, TENSOR_MAP_QWORDS
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
from cudnn.frost.tile_dsl.pointwise import fadd2, fp32_to_fp16, f16x2_to_f32, fmul2, opaque_f32_zero, sub_f16x2
from cudnn.frost.tile_dsl.swizzle import swizzle_lin_128b, swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import (
    tma_load_tile,
    tma_store_tile,
    tma_store_commit,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .gdn_prefill_config import CFG


class GdnBars(NamedTuple):
    """GDN pipeline mbarrier inventory.

    Every pipeline is a ``_ready``/``_done`` MBarrier pair over one ring: a
    slot is acquired for filling by waiting ``_done`` and committed by
    arriving ``_ready``; the reading side waits ``_ready`` and releases the
    slot by arriving ``_done``.
    """

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_kv_acc_ready: MBarrier
    mb_kv_acc_scale_done: MBarrier
    mb_o_acc_ready: MBarrier
    mb_o_acc_done: MBarrier
    mb_o_state_scale_acc_ready: MBarrier
    mb_o_state_scale_acc_done: MBarrier
    mb_cg0_acc_ready: MBarrier
    mb_cg0_acc_done: MBarrier
    mb_ks_ready: MBarrier
    mb_nv_ready: MBarrier

    mb_ainv_ready: MBarrier
    mb_ainv_done: MBarrier
    mb_qk_ready: MBarrier
    mb_qk_done: MBarrier
    mb_state_inp_ready: MBarrier
    mb_vks_inp_ready: MBarrier
    mb_nv_inp_ready: MBarrier
    mb_decay_v_inp_ready: MBarrier

    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier

    mb_h_tmastg_ready: MBarrier
    mb_h_tmastg_done: MBarrier

    mb_tmem_done: MBarrier
    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_gdn_bars(cfg) -> GdnBars:
    """GdnBars factory.  MUST be called from inside ``_kernel`` (allocates SMEM;
    the mbar rings sit ahead of the gate scalar arrays and data buffers)."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.mma_warp_id])
    BOTH_ISSUERS = len([cfg.mma_warp_id, cfg.mma_cg1_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_q_stages), stages=cfg.smem_q_stages, init_count=BOTH_ISSUERS, producer=Producer.MMA_COMMIT),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_k_stages), stages=cfg.smem_k_stages, init_count=BOTH_ISSUERS, producer=Producer.MMA_COMMIT),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_v_stages), stages=cfg.smem_v_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), stages=cfg.smem_gate_stages, init_count=CG0_PLUS_CG1, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_kv_acc_ready=MBarrier(alloc(cfg.tmem_kv_acc_stages), stages=cfg.tmem_kv_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_kv_acc_scale_done=MBarrier(alloc(cfg.tmem_kv_acc_stages), stages=cfg.tmem_kv_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_acc_ready=MBarrier(alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_o_acc_done=MBarrier(alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_state_scale_acc_ready=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_o_state_scale_acc_done=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_cg0_acc_ready=MBarrier(alloc(cfg.tmem_cg0_acc_stages), stages=cfg.tmem_cg0_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_cg0_acc_done=MBarrier(alloc(cfg.tmem_cg0_acc_stages), stages=cfg.tmem_cg0_acc_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_ks_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_nv_ready=MBarrier(alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_ainv_ready=MBarrier(alloc(cfg.smem_ainv_stages), stages=cfg.smem_ainv_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_ainv_done=MBarrier(alloc(cfg.smem_ainv_stages), stages=cfg.smem_ainv_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_qk_ready=MBarrier(alloc(cfg.smem_qk_stages), stages=cfg.smem_qk_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_qk_done=MBarrier(alloc(cfg.smem_qk_stages), stages=cfg.smem_qk_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_state_inp_ready=MBarrier(alloc(cfg.tmem_state_inp_stages), stages=cfg.tmem_state_inp_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_vks_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_nv_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_decay_v_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_h_tmastg_ready=MBarrier(alloc(cfg.smem_h_stages), stages=cfg.smem_h_stages, init_count=len(cfg.compute_group_1_warp_ids), producer=Producer.THREAD),
        mb_h_tmastg_done=MBarrier(alloc(cfg.smem_h_stages), stages=cfg.smem_h_stages, init_count=EPI_WARP, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_sched_ready=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=1, producer=Producer.THREAD),
        mb_sched_done=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=11, producer=Producer.THREAD),
    )


# ---------------------------------------------------------------------------
# Device-side helpers / warp bodies
# ---------------------------------------------------------------------------


@cute.jit
def _invert_diagonal_NxN(cfg, base_int, d, tidx, N: int = 8):
    """Stage 1: Gauss-Jordan inversion of one diagonal NxN block in-place (f16 SMEM).

    The tile swizzle re-homes whole rows only, so a diagonal block's row stays
    a contiguous N-element run at ``swz(row_lin_base)``.
    """
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_lin_base = (d * N + tidx_in_group) * BT + d * N
    row_phys = swizzle_lin_128b(row_lin_base, row_stride_log2=6)
    row_ptr = (
        cute.make_ptr(
            cfg.io_dtype,
            base_int,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )

    row = [(row_ptr + j).load().to(cutlass.Float32) for j in range(N)]
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
def _blockwise_diagonal_8x8_to_16x16(cfg, base_int, d0, lane_id):
    """Stage 2: off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}).

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
        cutlass.inttoptr(base_int + swizzle_lin_128b((d0 + 8) * 64 + d0 + lds1, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
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
def _blockwise_diagonal_16x16_to_32x32(cfg, base_int, d0, lane_id):
    """Stage 3: off-diagonal correction 16x16 -> 32x32."""
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
            cutlass.inttoptr(base_int + swizzle_lin_128b((d0 + 16) * 64 + d0 + lds4, row_stride_log2=6) * bpe, cutlass.AddressSpace.smem, cutlass.BFloat16),
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
def _blockwise_diagonal_32x32_to_64x64(cfg, base_int, warp_id, lane_id):
    """Stage 4: off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row
    M-band each)."""
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
                    base_int + swizzle_lin_128b((32 + (vs // 2) * 16) * 64 + (vs % 2) * 16 + lds4, row_stride_log2=6) * bpe,
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
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
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
def _load_gate_beta_chunk(
    cfg,
    lidx,
    mGate,
    mBeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    gate_index,
    beta_index,
    head_idx,
    batch_start,
    batch_end,
    chunk_idx,
    bars,
):
    """Load gate[BT]/beta[BT] for one chunk (epilogue warp).

    The OOB predicate is RUNTIME (covers the last valid chunk AND padded
    pair chunks), keeping one body in SASS.  OOB gate positions read a
    clamped in-bounds address and select the neutral value (gate=1 ->
    log 0); OOB beta lanes zero-fill via cp_size=0."""
    n_cols = cfg.b_t // cfg.threads_per_warp
    chunk_offset = batch_start + chunk_idx * cfg.b_t
    gGateSeq = mGate[None, head_idx]
    gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])

    gate_idx = gate_index.idx
    gate_phase = gate_index.phase
    gate_index = advance(gate_index, cfg.smem_gate_stages)

    pos_valid = [None] * n_cols
    tGrGate = [cutlass.Float32(0.0)] * n_cols
    oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
    for col in cutlass.range_constexpr(n_cols):
        tok = chunk_offset + lidx + col * cfg.threads_per_warp
        pos_valid[col] = cute.elem_less(tok, batch_end)
        # batch_end >= 1 whenever chunks exist, so the clamp stays in bounds
        tok_clamped = _cutlass_min(tok, batch_end - 1)
        tGrGate[col] = gGateSeq[tok_clamped] if pos_valid[col] else oob_neutral

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

    bars.mb_gate_done[gate_idx].wait(gate_phase)
    for col in cutlass.range_constexpr(n_cols):
        pos = lidx + col * cfg.threads_per_warp
        sCumsumlog[pos, 0, gate_idx] = tGrGate[col]
        sCumprod[pos, 0, gate_idx] = cute.math.exp2(tGrGate[col], fastmath=True)

    bars.mb_gate_ready[gate_idx].arrive()

    # --- Beta load (per-element async G->S cp.async) ---
    beta_idx = beta_index.idx
    bars.mb_beta_done[beta_idx].wait(beta_index.phase)
    beta_index = advance(beta_index, cfg.smem_beta_stages)
    for col in cutlass.range_constexpr(n_cols):
        pos = lidx + col * cfg.threads_per_warp
        src = gBeta.iterator + gBeta.layout((pos,))
        dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
        cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
        nvvm.cp_async_shared_global(dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size)
    nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)
    return gate_index, beta_index


# ---------------------------------------------------------------------------
# Dynamic tile scheduler: global-ticket work-stealing ring


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


@cute.jit
def _tmastg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    checkpoint_every_n_tokens,
    tidx,
    mGate,
    mBeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sO_raw,
    sH_raw,
    desc_o_base,
    desc_h_base,
    sSched,
    bars,
):
    """Epilogue warp role (warp 11): persistent tile-scheduler loop; loads
    gate/beta with a four-chunk lookahead and issues the per-chunk O and
    H-state TMA bulk-stores from SMEM staging to global memory."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    o_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    lidx = tidx % cfg.threads_per_warp
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)

    bpe = cfg.io_dtype.width // 8
    granu = 128 // bpe
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granu,
        tma_subtile_stride_elems=4096,
    )
    if cutlass.const_expr(cfg.enable_h):
        h_granu = 64
        sH_tma = SmemTile(
            base=sH_raw,
            elems_per_stage=(cfg.h_cosize // cfg.smem_h_stages),
            stages=cfg.smem_h_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=cfg.d_v // h_granu,
            tma_granu_elems=h_granu,
            tma_subtile_stride_elems=cfg.d_k * h_granu,
        )
        hr_cnt = cutlass.Int32(0)
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        n_local = wend - cstart
        n_padded = ((n_local + 1) // 2) * 2

        slot = (batch_idx * heads_out + head_idx) * desc_qwords
        if cutlass.const_expr(cfg.enable_o):
            desc_o_slot = (desc_o_base + slot).tospace(cutlass.AddressSpace.generic)
            if nvvm.elect_sync():
                tma_tensormap_acquire(desc_o_slot)
        if cutlass.const_expr(cfg.enable_h):
            desc_h_slot = (desc_h_base + slot).tospace(cutlass.AddressSpace.generic)
            # sequence-local H index of this item's first owned entry
            h_coord = wstart - 1 if wstart > 0 else cutlass.Int32(0)
            if nvvm.elect_sync():
                tma_tensormap_acquire(desc_h_slot)

        if n_local > 0:
            # ---- gate/beta lookahead: chunk count is padded even, so the
            # first two stages always exist and chunks 2/3 come together ----
            for pf in range(2):
                gate_index, beta_index = _load_gate_beta_chunk(
                    cfg, lidx, mGate, mBeta, sCumsumlog, sCumprod, sBeta, gate_index, beta_index, head_idx, batch_start, batch_end, cstart + pf, bars
                )
            if n_padded > 2:
                for pf in range(2, 4):
                    gate_index, beta_index = _load_gate_beta_chunk(
                        cfg,
                        lidx,
                        mGate,
                        mBeta,
                        sCumsumlog,
                        sCumprod,
                        sBeta,
                        gate_index,
                        beta_index,
                        head_idx,
                        batch_start,
                        batch_end,
                        cstart + pf,
                        bars,
                    )

            for local_idx in cutlass.range(n_padded):
                if local_idx + 4 < n_padded:
                    gate_index, beta_index = _load_gate_beta_chunk(
                        cfg,
                        lidx,
                        mGate,
                        mBeta,
                        sCumsumlog,
                        sCumprod,
                        sBeta,
                        gate_index,
                        beta_index,
                        head_idx,
                        batch_start,
                        batch_end,
                        cstart + local_idx + 4,
                        bars,
                    )
                chunk_idx = cstart + local_idx

                did_o = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_o):
                    o_idx = o_index.idx
                    bars.mb_o_tmastg_ready[o_idx].wait(o_index.phase)
                    o_index = advance(o_index, cfg.smem_o_stages)

                    # padded / warmup chunks stage O but never store it
                    if chunk_idx >= wstart and chunk_idx < wend:
                        tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
                        o_slice = tma_slice_runtime_desc(desc_o_slot, cutlass.Int32(0), tok_coord)
                        tma_store_tile(sO_tma[o_idx], o_slice, acquire=False)
                        tma_store_commit()
                        did_o = cutlass.Int32(1)

                did_h = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_h):
                    if chunk_idx >= wstart - 1 and chunk_idx < wend - 1:
                        if (cfg.b_t * (chunk_idx + 1)) % checkpoint_every_n_tokens == 0:
                            bars.mb_h_tmastg_ready[hr_cnt % cfg.smem_h_stages].wait((hr_cnt // cfg.smem_h_stages) & cutlass.Int32(1))
                            h_slice = tma_slice_runtime_desc(desc_h_slot, cutlass.Int32(0), cutlass.Int32(0), h_coord)
                            tma_store_tile(sH_tma[hr_cnt % cfg.smem_h_stages], h_slice, acquire=False)
                            tma_store_commit()
                            h_coord += 1
                            did_h = cutlass.Int32(1)

                if cutlass.const_expr(cfg.enable_o):
                    if cutlass.const_expr(cfg.enable_h):
                        if did_o == 1 and did_h == 1:
                            tma_store_wait(1)
                            bars.mb_o_tmastg_done[o_idx].arrive()
                            tma_store_wait(0)
                            bars.mb_h_tmastg_done[hr_cnt % cfg.smem_h_stages].arrive()
                            hr_cnt = hr_cnt + 1
                        if did_o == 1 and did_h == 0:
                            tma_store_wait(0)
                            bars.mb_o_tmastg_done[o_idx].arrive()
                        if did_o == 0:
                            if did_h == 1:
                                tma_store_wait(0)
                                bars.mb_h_tmastg_done[hr_cnt % cfg.smem_h_stages].arrive()
                                hr_cnt = hr_cnt + 1
                            bars.mb_o_tmastg_done[o_idx].arrive()
                    else:
                        if did_o == 1:
                            tma_store_wait(0)
                        bars.mb_o_tmastg_done[o_idx].arrive()
                else:
                    if cutlass.const_expr(cfg.enable_h):
                        if did_h == 1:
                            tma_store_wait(0)
                            bars.mb_h_tmastg_done[hr_cnt % cfg.smem_h_stages].arrive()
                            hr_cnt = hr_cnt + 1

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    for _ in range(cfg.smem_gate_stages):
        bars.mb_gate_done[gate_index.idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)
    for _ in range(cfg.smem_beta_stages):
        bars.mb_beta_done[beta_index.idx].wait(beta_index.phase)
        beta_index = advance(beta_index, cfg.smem_beta_stages)


@cute.jit
def _mma0_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_hold,
    sQ,
    sK,
    sSched,
    bars,
):
    """CG0 MMA issuer role (warp 8): persistent scheduler loop + per-pair
    KK0/KK1/QK0/QK1 issue into CG0's private accumulator ring; owns the TMEM
    lifecycle (alloc up front, dealloc once CG1 signals mb_tmem_done)."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    cg0_acc_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)

    nvvm.tcgen05_alloc(tmem_hold, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

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
    tmem_cg0_acc_col = tmem_base + cfg.tmem_cg0_acc_offset
    ACC_STAGE_COLS = cfg.b_t

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        n_pairs = (wend - cstart + 1) // 2
        for _pair in cutlass.range(n_pairs):  # noqa: B007
            # ---- GEMM 1 (chunk 0 of pair): kk0 -------------------------
            kk0_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[kk0_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            k0_idx = k_index.idx
            bars.mb_k_ready[k0_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)

            desc_k0 = sK[k0_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_k0,
                desc_k0,
                nvvm.make_tmem_ptr(tmem_cg0_acc_col + kk0_acc_idx * ACC_STAGE_COLS, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_cg0_acc_ready[kk0_acc_idx].arrive(cta_group=1)

            # ---- GEMM 1 (chunk 1 of pair): kk1 -------------------------
            kk1_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[kk1_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            k1_idx = k_index.idx
            bars.mb_k_ready[k1_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)

            desc_k1 = sK[k1_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_k1,
                desc_k1,
                nvvm.make_tmem_ptr(tmem_cg0_acc_col + kk1_acc_idx * ACC_STAGE_COLS, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_cg0_acc_ready[kk1_acc_idx].arrive(cta_group=1)

            # ---- GEMM 2 (chunk 0 of pair): qk0 -------------------------
            q0_idx = q_index.idx
            bars.mb_q_ready[q0_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            qk0_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[qk0_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)

            desc_q0 = sQ[q0_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_q0,
                desc_k0,
                nvvm.make_tmem_ptr(tmem_cg0_acc_col + qk0_acc_idx * ACC_STAGE_COLS, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_cg0_acc_ready[qk0_acc_idx].arrive(cta_group=1)

            # ---- GEMM 2 (chunk 1 of pair): qk1 -------------------------
            q1_idx = q_index.idx
            bars.mb_q_ready[q1_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            qk1_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[qk1_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)

            desc_q1 = sQ[q1_idx].desc()
            mma_ss(
                bmm_qk_desc,
                desc_q1,
                desc_k1,
                nvvm.make_tmem_ptr(tmem_cg0_acc_col + qk1_acc_idx * ACC_STAGE_COLS, cutlass.Float32),
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_cg0_acc_ready[qk1_acc_idx].arrive(cta_group=1)

            # this issuer's Q/K releases (the CG1 issuer commits its own)
            if nvvm.elect_sync():
                bars.mb_q_done[q0_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_q_done[q1_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_k_done[k0_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_k_done[k1_idx].arrive(cta_group=1)

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def _mma1_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_hold,
    sQ,
    sK,
    sK_trans,
    sAinv,
    sQk,
    sSched,
    bars,
):
    """CG1 MMA issuer role (warp 10): persistent scheduler loop + per-chunk
    issue of the five state/output GEMMs (KS/QS/NV/QKV/KV) in dependency
    order."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    o_acc_index = PipelineState.start(phase=1)
    o_state_scale_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)
    ainv_index = PipelineState.start(phase=0)
    qk_index = PipelineState.start(phase=0)
    state_inp_index = PipelineState.start(phase=0)
    vks_inp_rdy = PipelineState.start(phase=0)
    nv_inp_rdy = PipelineState.start(phase=0)
    decay_v_inp_rdy = PipelineState.start(phase=0)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

    # ---- chunk-invariant GEMM descriptors ------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_qs = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_qs_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_qs,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_qkv_ts = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_qkv_ts_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_qkv_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_kv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_kv_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_kv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    tmem_state_col = tmem_base + cfg.tmem_state_offset
    tmem_q_state_col = tmem_base + cfg.tmem_q_state_offset
    tmem_state_inp_col = tmem_base + cfg.tmem_state_inp_offset
    tmem_inp_col = tmem_base + cfg.tmem_inp_offset
    ACC_STAGE_COLS = cfg.b_t
    KV_ACC_STAGE_COLS = cfg.d_v
    STATE_INP_STAGE_COLS = cfg.d_k // 2
    INP_SLOT_COLS = cfg.b_t // 2
    tmem_ks_col = tmem_base + cfg.tmem_cg1_acc_offset
    tmem_nv_col = tmem_ks_col
    tmem_vks_col = tmem_inp_col
    # NV overwrites the VKS slot once GEMM 5 has consumed it
    tmem_nv_inp_col = tmem_inp_col
    tmem_decay_v_col = tmem_inp_col + INP_SLOT_COLS

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        n_local = wend - cstart
        n_padded = ((n_local + 1) // 2) * 2

        first_loop = 0
        if cutlass.const_expr(not cfg.use_initial_state):
            # ---- peeled first chunk: S_prev = 0, GEMMs 3/4 skipped -----
            if n_local > 0:
                k_idx = k_index.idx
                bars.mb_k_ready[k_idx].wait(k_index.phase)
                k_index = advance(k_index, cfg.smem_k_stages)
                q_idx = q_index.idx
                bars.mb_q_ready[q_idx].wait(q_index.phase)
                q_index = advance(q_index, cfg.smem_q_stages)
                if nvvm.elect_sync():
                    bars.mb_q_done[q_idx].arrive(cta_group=1)

                # ---- GEMM 5: new_v ---------------------------------
                bars.mb_vks_inp_ready[0].wait(vks_inp_rdy.phase)
                vks_inp_rdy = advance(vks_inp_rdy, 1)
                ainv_idx = ainv_index.idx
                bars.mb_ainv_ready[ainv_idx].wait(ainv_index.phase)
                ainv_index = advance(ainv_index, cfg.smem_ainv_stages)

                desc_ainv = sAinv[ainv_idx].desc()
                vks_a_ptr = nvvm.make_tmem_ptr(tmem_vks_col, cutlass.Int8)
                mma_ts(
                    bmm_qkv_ts_desc,
                    vks_a_ptr,
                    desc_ainv,
                    nvvm.make_tmem_ptr(tmem_nv_col, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_nv_ready[0].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_ainv_done[ainv_idx].arrive(cta_group=1)

                # ---- GEMM 6: qkv -----------------------------------
                bars.mb_nv_inp_ready[0].wait(nv_inp_rdy.phase)
                nv_inp_rdy = advance(nv_inp_rdy, 1)
                qk_idx = qk_index.idx
                bars.mb_qk_ready[qk_idx].wait(qk_index.phase)
                qk_index = advance(qk_index, cfg.smem_qk_stages)
                qs_idx = o_acc_index.idx
                bars.mb_o_acc_done[qs_idx].wait(o_acc_index.phase)
                o_acc_index = advance(o_acc_index, cfg.tmem_q_state_acc_stages)

                qkv_a_ptr = nvvm.make_tmem_ptr(tmem_nv_inp_col, cutlass.Int8)
                desc_nv = sQk[qk_idx].desc()
                mma_ts(
                    bmm_qkv_ts_desc,
                    qkv_a_ptr,
                    desc_nv,
                    nvvm.make_tmem_ptr(tmem_q_state_col + qs_idx * ACC_STAGE_COLS, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_qk_done[qk_idx].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_o_state_scale_acc_ready[qs_idx].arrive(cta_group=1)

                # ---- GEMM 7: kv_update ---------------------------------------
                bars.mb_decay_v_inp_ready[0].wait(decay_v_inp_rdy.phase)
                decay_v_inp_rdy = advance(decay_v_inp_rdy, 1)
                kv_acc_idx = kv_acc_index.idx
                bars.mb_kv_acc_scale_done[kv_acc_idx].wait(kv_acc_index.phase)
                kv_acc_index = advance(kv_acc_index, cfg.tmem_kv_acc_stages)

                delta_a_ptr = nvvm.make_tmem_ptr(tmem_decay_v_col, cutlass.Int8)
                desc_kt = sK_trans[k_idx].desc()
                mma_ts(
                    bmm_kv_desc,
                    delta_a_ptr,
                    desc_kt,
                    nvvm.make_tmem_ptr(tmem_state_col + kv_acc_idx * KV_ACC_STAGE_COLS, cutlass.Float32),
                    accumulate=False,
                )
                if nvvm.elect_sync():
                    bars.mb_kv_acc_ready[kv_acc_idx].arrive(cta_group=1)
                if nvvm.elect_sync():
                    bars.mb_k_done[k_idx].arrive(cta_group=1)

            first_loop = 1

        for local_idx in cutlass.range(first_loop, n_padded):  # noqa: B007
            if cutlass.const_expr(cfg.use_initial_state):
                if local_idx == 0:
                    kv_acc_index = advance(kv_acc_index, cfg.tmem_kv_acc_stages)

            k_idx = k_index.idx
            q_idx = q_index.idx
            s_idx = state_inp_index.idx
            q_state_acc_idx = o_acc_index.idx
            ainv_idx = ainv_index.idx
            qk_idx = qk_index.idx
            qs2_idx = o_state_scale_index.idx
            kv_acc_idx = kv_acc_index.idx
            desc_k = sK[k_idx].desc()
            desc_q = sQ[q_idx].desc()
            desc_ainv = sAinv[ainv_idx].desc()
            desc_nv = sQk[qk_idx].desc()
            desc_kt = sK_trans[k_idx].desc()
            state_a_ptr = nvvm.make_tmem_ptr(tmem_state_inp_col + s_idx * STATE_INP_STAGE_COLS, cutlass.Int8)
            vks_a_ptr = nvvm.make_tmem_ptr(tmem_vks_col, cutlass.Int8)
            qkv_a_ptr = nvvm.make_tmem_ptr(tmem_nv_inp_col, cutlass.Int8)
            delta_a_ptr = nvvm.make_tmem_ptr(tmem_decay_v_col, cutlass.Int8)
            ks_acc_ptr = nvvm.make_tmem_ptr(tmem_ks_col, cutlass.Float32)
            qs_acc_ptr = nvvm.make_tmem_ptr(tmem_q_state_col + q_state_acc_idx * ACC_STAGE_COLS, cutlass.Float32)
            nv_acc_ptr = nvvm.make_tmem_ptr(tmem_nv_col, cutlass.Float32)
            qkv_acc_ptr = nvvm.make_tmem_ptr(tmem_q_state_col + qs2_idx * ACC_STAGE_COLS, cutlass.Float32)
            kv_acc_ptr = nvvm.make_tmem_ptr(tmem_state_col + kv_acc_idx * KV_ACC_STAGE_COLS, cutlass.Float32)

            bars.mb_k_ready[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            bars.mb_q_ready[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)

            # ---- GEMM 3: k*state ---------------------------------------
            bars.mb_state_inp_ready[s_idx].wait(state_inp_index.phase)
            state_inp_index = advance(state_inp_index, cfg.tmem_state_inp_stages)

            mma_ts(
                bmm_qs_desc,
                state_a_ptr,
                desc_k,
                ks_acc_ptr,
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_ks_ready[0].arrive(cta_group=1)

            # ---- GEMM 4: q*state ---------------------------------------
            bars.mb_o_acc_done[q_state_acc_idx].wait(o_acc_index.phase)
            o_acc_index = advance(o_acc_index, cfg.tmem_q_state_acc_stages)
            mma_ts(
                bmm_qs_desc,
                state_a_ptr,
                desc_q,
                qs_acc_ptr,
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_o_acc_ready[q_state_acc_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_q_done[q_idx].arrive(cta_group=1)

            # ---- GEMM 5: new_v -----------------------------------------
            # vks_inp_ready also proves CG1 read KS out of the shared column
            bars.mb_vks_inp_ready[0].wait(vks_inp_rdy.phase)
            vks_inp_rdy = advance(vks_inp_rdy, 1)
            bars.mb_ainv_ready[ainv_idx].wait(ainv_index.phase)
            ainv_index = advance(ainv_index, cfg.smem_ainv_stages)
            mma_ts(
                bmm_qkv_ts_desc,
                vks_a_ptr,
                desc_ainv,
                nv_acc_ptr,
                accumulate=False,
            )
            if nvvm.elect_sync():
                bars.mb_nv_ready[0].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_ainv_done[ainv_idx].arrive(cta_group=1)

            # ---- GEMM 6: qkv -------------------------------------------
            bars.mb_nv_inp_ready[0].wait(nv_inp_rdy.phase)
            nv_inp_rdy = advance(nv_inp_rdy, 1)
            bars.mb_qk_ready[qk_idx].wait(qk_index.phase)
            qk_index = advance(qk_index, cfg.smem_qk_stages)
            bars.mb_o_state_scale_acc_done[qs2_idx].wait(o_state_scale_index.phase)
            o_state_scale_index = advance(o_state_scale_index, cfg.tmem_q_state_acc_stages)
            mma_ts(
                bmm_qkv_ts_desc,
                qkv_a_ptr,
                desc_nv,
                qkv_acc_ptr,
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_qk_done[qk_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_o_state_scale_acc_ready[qs2_idx].arrive(cta_group=1)

            # ---- GEMM 7: kv_update -------------------------------------------
            bars.mb_decay_v_inp_ready[0].wait(decay_v_inp_rdy.phase)
            decay_v_inp_rdy = advance(decay_v_inp_rdy, 1)
            bars.mb_kv_acc_scale_done[kv_acc_idx].wait(kv_acc_index.phase)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_kv_acc_stages)
            mma_ts(
                bmm_kv_desc,
                delta_a_ptr,
                desc_kt,
                kv_acc_ptr,
                accumulate=True,
            )
            if nvvm.elect_sync():
                bars.mb_kv_acc_ready[kv_acc_idx].arrive(cta_group=1)
            if nvvm.elect_sync():
                bars.mb_k_done[k_idx].arrive(cta_group=1)

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


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
    desc_q_base,
    desc_k_base,
    desc_v_base,
    mSched,
    sSched,
    bars,
):
    """TMA-LDG warp role (warp 9): persistent scheduler loop + per-chunk
    Q/K/V G->S TMA loads.

    Each load goes through a per-(b,h) runtime descriptor (built once on
    host) whose GLOBAL_ADDRESS already folds the sequence start and head, so
    the only load coordinate is token = ``chunk_idx*BT``."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)

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
        elems_per_stage=cfg.v_cosize // cfg.smem_v_stages,
        stages=cfg.smem_v_stages,
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
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        if nvvm.elect_sync():
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)

        # padded loads zero-fill (descriptor token extent is capped)
        wend_padded = cstart + ((wend - cstart + 1) // 2) * 2
        for chunk_idx in cutlass.range(cstart, wend_padded):
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            # ----------------------------------------------------------
            # K  (B operand of GEMM-kk / GEMM-qk, double-buffered)
            # ----------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if nvvm.elect_sync():
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # Q  (A operand of GEMM-qk, single-buffered)
            # ----------------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_done[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            if nvvm.elect_sync():
                bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr, acquire=False)

            # ----------------------------------------------------------
            # V  (A operand of GEMM-new_v; transposed [DV, T] descriptor)
            # ----------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if nvvm.elect_sync():
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), tok_coord)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

        tile_idx, sched_state = _sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)


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
    sBeta,
    sAinv,
    sQk,
    sH_raw,
    checkpoint_every_n_tokens,
    sSched,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): persistent scheduler loop +
    per-PAIR T-pairwise x2, kk_epi x2, pair inverse (warps 0-1 invert chunk
    0's matrix while warps 2-3 invert chunk 1's), qk_epi x2, and (enable_h)
    the H checkpoint readouts state TMEM -> sH woven around the QK epilogues."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    cg0_acc_rdy = PipelineState.start(phase=0)
    ainv_index = PipelineState.start(phase=1)
    qk_index = PipelineState.start(phase=1)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_hold.load()

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_id = cg0_tidx % cfg.threads_per_warp
    inverse_local_warp = warp_id % 2
    bpe = cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    store_row = warp_id * 16 + lane_id % 16
    store_col = (lane_id // 16) * 8
    tmem_warp_row = warp_id * cfg.threads_per_warp
    tmem_cg0_acc_col = tmem_base + cfg.tmem_cg0_acc_offset
    ACC_STAGE_COLS = cfg.b_t
    mask_zero = opaque_f32_zero()
    crow_lo = warp_id * 16 + lane_id // 4
    crow_hi = crow_lo + 8

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        n_local = wend - cstart
        n_pairs = (n_local + 1) // 2
        n_padded = n_pairs * 2

        for pair_i in cutlass.range(n_pairs):
            chunk0 = cstart + pair_i * 2
            chunk1 = chunk0 + 1
            # ---- Step 1: T-pairwise, both chunks in one traversal --------
            gate0_idx = gate_index.idx
            bars.mb_gate_ready[gate0_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            gate1_idx = gate_index.idx
            bars.mb_gate_ready[gate1_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            row_cs0_lo = sCumsumlog[crow_lo, 0, gate0_idx]
            row_cs0_hi = sCumsumlog[crow_hi, 0, gate0_idx]
            row_cs1_lo = sCumsumlog[crow_lo, 0, gate1_idx]
            row_cs1_hi = sCumsumlog[crow_hi, 0, gate1_idx]

            gT0 = []
            gT1 = []
            for k in cutlass.range_constexpr(num_vals):
                hi_row = cutlass.const_expr(((k // 2) % 2) == 1)
                crow = crow_hi if cutlass.const_expr(hi_row) else crow_lo
                ccol = (lane_id % 4) * 2 + ((k // 4) * 8 + k % 2)
                is_lower = crow >= ccol
                cs0 = row_cs0_hi if cutlass.const_expr(hi_row) else row_cs0_lo
                cs1 = row_cs1_hi if cutlass.const_expr(hi_row) else row_cs1_lo
                gT0.append(cute.math.exp2(cs0 - sCumsumlog[ccol, 0, gate0_idx], fastmath=True) if is_lower else mask_zero)
                gT1.append(cute.math.exp2(cs1 - sCumsumlog[ccol, 0, gate1_idx], fastmath=True) if is_lower else mask_zero)
            bars.mb_gate_done[gate0_idx].arrive()
            bars.mb_gate_done[gate1_idx].arrive()

            beta0_idx = beta_index.idx
            bars.mb_beta_ready[beta0_idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)
            beta1_idx = beta_index.idx
            bars.mb_beta_ready[beta1_idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)
            # beta row scaling only needs the two per-thread row scalars
            beta0_lo = sBeta[crow_lo, 0, beta0_idx]
            beta0_hi = sBeta[crow_hi, 0, beta0_idx]
            beta1_lo = sBeta[crow_lo, 0, beta1_idx]
            beta1_hi = sBeta[crow_hi, 0, beta1_idx]

            # ---- Step 2: kk_epi0 + kk_epi1 ------------------------------
            ainv0_idx = ainv_index.idx
            bars.mb_ainv_done[ainv0_idx].wait(ainv_index.phase)
            ainv_index = advance(ainv_index, cfg.smem_ainv_stages)
            kk0_acc_idx = cg0_acc_rdy.idx
            bars.mb_cg0_acc_ready[kk0_acc_idx].wait(cg0_acc_rdy.phase)
            cg0_acc_rdy = advance(cg0_acc_rdy, cfg.tmem_cg0_acc_stages)

            ainv0_base = sAinv[ainv0_idx].base
            kk_vec = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_cg0_acc_col + kk0_acc_idx * ACC_STAGE_COLS, cutlass.Float32), num=8
            )
            nvvm.tcgen05_wait("load")
            bars.mb_cg0_acc_done[kk0_acc_idx].arrive()
            kk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                b0 = beta0_hi if cutlass.const_expr((k % 2) == 1) else beta0_lo
                p0, p1 = fmul2(kk_vec[2 * k], kk_vec[2 * k + 1], gT0[2 * k], gT0[2 * k + 1])
                v0, v1 = fmul2(p0, p1, b0, b0)
                kk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        ainv0_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [kk_f16[c * 4 + 0], kk_f16[c * 4 + 1], kk_f16[c * 4 + 2], kk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )

            ainv1_idx = ainv_index.idx
            bars.mb_ainv_done[ainv1_idx].wait(ainv_index.phase)
            ainv_index = advance(ainv_index, cfg.smem_ainv_stages)
            kk1_acc_idx = cg0_acc_rdy.idx
            bars.mb_cg0_acc_ready[kk1_acc_idx].wait(cg0_acc_rdy.phase)
            cg0_acc_rdy = advance(cg0_acc_rdy, cfg.tmem_cg0_acc_stages)

            ainv1_base = sAinv[ainv1_idx].base
            kk_vec = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_cg0_acc_col + kk1_acc_idx * ACC_STAGE_COLS, cutlass.Float32), num=8
            )
            nvvm.tcgen05_wait("load")
            bars.mb_cg0_acc_done[kk1_acc_idx].arrive()
            kk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                b1 = beta1_hi if cutlass.const_expr((k % 2) == 1) else beta1_lo
                p0, p1 = fmul2(kk_vec[2 * k], kk_vec[2 * k + 1], gT1[2 * k], gT1[2 * k + 1])
                v0, v1 = fmul2(p0, p1, b1, b1)
                kk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        ainv1_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [kk_f16[c * 4 + 0], kk_f16[c * 4 + 1], kk_f16[c * 4 + 2], kk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )

            # ---- pair inverse: warps 0-1 own matrix 0, warps 2-3 matrix 1
            inv_base = ainv1_base if warp_id >= 2 else ainv0_base

            # Stage 1: diagonal 8x8 Gauss-Jordan, all four warps
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )
            _invert_diagonal_NxN(cfg, inv_base, (inverse_local_warp * cfg.threads_per_warp + lane_id) // 8, cg0_tidx, 8)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # Stage 2: 8x8 -> 16x16 (both matrices per warp)
            _blockwise_diagonal_8x8_to_16x16(cfg, ainv0_base, warp_id * 16, lane_id)
            _blockwise_diagonal_8x8_to_16x16(cfg, ainv1_base, warp_id * 16, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # Stage 3: 16x16 -> 32x32, one tile per warp within the group
            _blockwise_diagonal_16x16_to_32x32(cfg, inv_base, inverse_local_warp * 32, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # Stage 4: 32x32 -> 64x64, two warps per matrix
            _blockwise_diagonal_32x32_to_64x64(cfg, inv_base, inverse_local_warp, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # ---- beta column-scaling + publish, stage 0 then stage 1 ----
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta0_idx])
            ainv_f16 = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                ainv_f16 += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            ainv0_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
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
                        ainv0_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [ainv_scaled[c * 4 + 0], ainv_scaled[c * 4 + 1], ainv_scaled[c * 4 + 2], ainv_scaled[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ainv_ready[ainv0_idx].arrive()
            bars.mb_beta_done[beta0_idx].arrive()

            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta1_idx])
            ainv_f16 = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                ainv_f16 += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            ainv1_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
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
                        ainv1_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [ainv_scaled[c * 4 + 0], ainv_scaled[c * 4 + 1], ainv_scaled[c * 4 + 2], ainv_scaled[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ainv_ready[ainv1_idx].arrive()
            bars.mb_beta_done[beta1_idx].arrive()

            # ---- Step 3: qk_epi0 ----------------------------------------
            qk0_idx = qk_index.idx
            bars.mb_qk_done[qk0_idx].wait(qk_index.phase)
            qk_index = advance(qk_index, cfg.smem_qk_stages)
            qk0_acc_idx = cg0_acc_rdy.idx
            bars.mb_cg0_acc_ready[qk0_acc_idx].wait(cg0_acc_rdy.phase)
            cg0_acc_rdy = advance(cg0_acc_rdy, cfg.tmem_cg0_acc_stages)

            qk0_base = sQk[qk0_idx].base
            qk_vec = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_cg0_acc_col + qk0_acc_idx * ACC_STAGE_COLS, cutlass.Float32), num=8
            )
            nvvm.tcgen05_wait("load")
            bars.mb_cg0_acc_done[qk0_acc_idx].arrive()
            qk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(qk_vec[2 * k], qk_vec[2 * k + 1], gT0[2 * k], gT0[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                qk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        qk0_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [qk_f16[c * 4 + 0], qk_f16[c * 4 + 1], qk_f16[c * 4 + 2], qk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_ready[qk0_idx].arrive()

            # ---- Step 4: qk_epi1 ----------------------------------------
            qk1_idx = qk_index.idx
            bars.mb_qk_done[qk1_idx].wait(qk_index.phase)
            qk_index = advance(qk_index, cfg.smem_qk_stages)
            qk1_acc_idx = cg0_acc_rdy.idx
            bars.mb_cg0_acc_ready[qk1_acc_idx].wait(cg0_acc_rdy.phase)
            cg0_acc_rdy = advance(cg0_acc_rdy, cfg.tmem_cg0_acc_stages)

            qk1_base = sQk[qk1_idx].base
            qk_vec = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_cg0_acc_col + qk1_acc_idx * ACC_STAGE_COLS, cutlass.Float32), num=8
            )
            nvvm.tcgen05_wait("load")
            bars.mb_cg0_acc_done[qk1_acc_idx].arrive()
            qk_f16 = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(qk_vec[2 * k], qk_vec[2 * k + 1], gT1[2 * k], gT1[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                qk_f16.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        qk1_base + (store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)) * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [qk_f16[c * 4 + 0], qk_f16[c * 4 + 1], qk_f16[c * 4 + 2], qk_f16[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_ready[qk1_idx].arrive()

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
    for _ in range(cfg.smem_ainv_stages):
        bars.mb_ainv_done[ainv_index.idx].wait(ainv_index.phase)
        ainv_index = advance(ainv_index, cfg.smem_ainv_stages)
    for _ in range(cfg.smem_qk_stages):
        bars.mb_qk_done[qk_index.idx].wait(qk_index.phase)
        qk_index = advance(qk_index, cfg.smem_qk_stages)


@cute.jit
def _compute1_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    warp_idx,
    tmem_hold,
    scale,
    sV,
    sCumsumlog,
    sCumprod,
    sBeta,
    sO,
    sH_raw,
    mS_init,
    mS_out,
    checkpoint_every_n_tokens,
    sSched,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop,
    initial-state seed, then one uniform per-chunk body: fused state
    restage+rescale, V-K*S, state*q_epi, new_v_epi + decay_v publish, qkv
    epilogue; final-state store per item."""

    v_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=0)
    o_acc_rdy_index = PipelineState.start(phase=0)
    o_scale_rdy_index = PipelineState.start(phase=0)
    ks_rdy_index = PipelineState.start(phase=0)
    nv_rdy_index = PipelineState.start(phase=0)
    kv_acc_seed_index = PipelineState.start(phase=1)
    o_index = PipelineState.start(phase=1)
    si_cnt = cutlass.Int32(0)
    kv_done_idx = cutlass.Int32(0)

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
    tmem_state_col = tmem_base + cfg.tmem_state_offset
    tmem_state_inp_col = tmem_base + cfg.tmem_state_inp_offset
    tmem_q_state_col = tmem_base + cfg.tmem_q_state_offset
    tmem_inp_col = tmem_base + cfg.tmem_inp_offset
    ACC_STAGE_COLS = cfg.b_t
    INP_SLOT_COLS = cfg.b_t // 2
    tmem_ks_col = tmem_base + cfg.tmem_cg1_acc_offset
    tmem_nv_col = tmem_ks_col
    tmem_vks_col = tmem_inp_col
    tmem_nv_inp_col = tmem_inp_col
    tmem_decay_v_col = tmem_inp_col + INP_SLOT_COLS
    ov_tok = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    ov_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
    ov_slab = (cg1_tidx // 64) * 4096
    v_stage_elems = cfg.v_cosize // cfg.smem_v_stages
    o_stage_elems = cfg.o_cosize // cfg.smem_o_stages
    sV_base = cute.make_ptr(cfg.io_dtype, sV[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    sO_base = cute.make_ptr(cfg.io_dtype, sO[0].base, mem_space=cute.AddressSpace.smem, assumed_align=cfg.buffer_align_bytes)
    num_vals = 32
    if cutlass.const_expr(cfg.enable_h):
        sH_base_int = sH_raw.data_ptr().toint()
        h_cnt = cutlass.Int32(0)
        h_ov_tok = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
        h_ov_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(
            cfg, tile_idx, cu_seqlens, mWorkItems
        )
        sk_nt = wend - cstart
        n_padded = ((sk_nt + 1) // 2) * 2
        if sk_nt > 0:
            if cutlass.const_expr(cfg.use_initial_state):
                # ---- initial-state seed: S_init GMEM -> state TMEM
                # (split-K warmup items seed zeros through the same path) ----
                gS_init = mS_init[None, None, head_idx, batch_idx]
                kv_init_idx = kv_acc_seed_index.idx
                bars.mb_kv_acc_scale_done[kv_init_idx].wait(kv_acc_seed_index.phase)
                kv_acc_seed_index = advance(kv_acc_seed_index, cfg.tmem_kv_acc_stages)
                seed_from_s0 = cstart == 0
                for sub in cutlass.range_constexpr(num_state_subs):
                    words = []
                    for k in cutlass.range_constexpr(32):
                        v = gS_init[sub * ldtm_width + k, cg1_tidx]
                        if cutlass.const_expr(cfg.state_dtype != cfg.acc_dtype):
                            v = v.to(cfg.acc_dtype)
                        if cutlass.const_expr(cfg.split_k):
                            v = v if seed_from_s0 else cutlass.Float32(0.0)
                        words.append(v)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")

                nvvm.barrier_cta_sync_aligned(
                    cfg.init_state_store_barrier_id,
                    thread_count=cfg.init_state_store_barrier_threads,
                )
                if cg1_tidx == 0:
                    arrive(bars.mb_kv_acc_ready[kv_init_idx].smem_ptr)

            for local_idx in cutlass.range(n_padded):  # noqa: B007
                chunk_idx = cstart + local_idx
                valid_state = local_idx > 0
                if cutlass.const_expr(cfg.use_initial_state):
                    valid_state = cutlass.Boolean(True)
                    kv_acc_seed_index = advance(kv_acc_seed_index, cfg.tmem_kv_acc_stages)

                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

                # ---- fused state restage + rescale (one read serves both) ----
                if valid_state:
                    kv_idx = kv_acc_index.idx
                    bars.mb_kv_acc_ready[kv_idx].wait(kv_acc_index.phase)
                    kv_acc_index = advance(kv_acc_index, cfg.tmem_kv_acc_stages)
                    kv_done_idx = kv_idx

                    state_regs = [[cutlass.Float32(0.0) for _ in range(num_state_subs)] for _ in range(32)]
                    siu_idx = si_cnt % cfg.tmem_state_inp_stages
                    # loads before stores: a TMEM st blocks ld hoisting
                    state_vecs = []
                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_vecs.append(
                            nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width, cutlass.Float32), num=32)
                        )
                    for sub in cutlass.range_constexpr(num_state_subs):
                        for k in cutlass.range_constexpr(32):
                            state_regs[k][sub] = state_vecs[sub][k]
                        state_f16 = [fp32_to_fp16(state_regs[2 * j][sub], state_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_inp_col + sub * sttm_width, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(state_f16), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_state_inp_ready[siu_idx].arrive()
                    si_cnt = si_cnt + 1

                    if cutlass.const_expr(cfg.enable_h):
                        # ---- H checkpoint: state TMEM -> sH before the decayed
                        # write-back overwrites it ----
                        do_h = (cfg.b_t * chunk_idx - 1) % checkpoint_every_n_tokens == checkpoint_every_n_tokens - 1 and chunk_idx < wend
                        if cutlass.const_expr(cfg.split_k):
                            do_h = do_h and chunk_idx >= wstart
                        if do_h:
                            h_regs = [[cutlass.Int32(0) for _ in range(16)] for _ in range(4)]
                            for b in cutlass.range_constexpr(2):
                                for hh in cutlass.range_constexpr(2):
                                    h_vec = nvvm.tcgen05_ld(
                                        "16x256b",
                                        nvvm.make_tmem_ptr(((tmem_warp_row + b * 16) << 16) + tmem_state_col + hh * 64, cutlass.Float32),
                                        num=8,
                                    )
                                    for j in cutlass.range_constexpr(16):
                                        h_regs[b * 2 + hh][j] = fp32_to_fp16(h_vec[2 * j], h_vec[2 * j + 1], dtype=cfg.io_dtype)
                            bars.mb_h_tmastg_done[h_cnt % cfg.smem_h_stages].wait(cutlass.Int32(1) ^ ((h_cnt // cfg.smem_h_stages) & cutlass.Int32(1)))
                            for b in cutlass.range_constexpr(2):
                                for hh in cutlass.range_constexpr(2):
                                    h_base = (h_cnt % cfg.smem_h_stages) * cfg.d_k * cfg.d_v + (cg1_tidx // 64) * cfg.d_k * 64
                                    for c in cutlass.range_constexpr(4):
                                        h_row = hh * 64 + h_ov_tok + c * 16
                                        nvvm.stmatrix(
                                            cutlass.inttoptr(
                                                sH_base_int + (h_base + h_row * 64 + swizzle_xor_128b(h_row, h_ov_col + b * 16)) * 2,
                                                cutlass.AddressSpace.smem,
                                                cfg.io_dtype,
                                            ),
                                            [
                                                h_regs[b * 2 + hh][c * 4 + 0],
                                                h_regs[b * 2 + hh][c * 4 + 1],
                                                h_regs[b * 2 + hh][c * 4 + 2],
                                                h_regs[b * 2 + hh][c * 4 + 3],
                                            ],
                                            nvvm.MMALayout.COL,
                                        )
                            nvvm.fence_proxy("async.shared", space="cta")
                            if nvvm.elect_sync():
                                bars.mb_h_tmastg_ready[h_cnt % cfg.smem_h_stages].arrive()
                            h_cnt = h_cnt + 1

                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_scaled = []
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(state_regs[2 * j][sub], state_regs[2 * j + 1][sub], cumprod_total, cumprod_total)
                            state_scaled += [s0, s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(state_scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_kv_acc_scale_done[kv_idx].arrive()

                # ---- deferred per-row gate register builds ---------------
                gCumprod = []
                for k in cutlass.range_constexpr(num_vals):
                    gCumprod.append(sCumprod[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                # gathers before math: interleaving pins one LDS latency per pair
                gCs = []
                for k in cutlass.range_constexpr(num_vals):
                    gCs.append(sCumsumlog[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])
                gDecayScale = []
                for k in cutlass.range_constexpr(0, num_vals, 2):
                    d0, d1 = fadd2(last_cumsumlog, last_cumsumlog, -gCs[k], -gCs[k + 1])
                    gDecayScale.append(cute.math.exp2(d0, fastmath=True))
                    gDecayScale.append(cute.math.exp2(d1, fastmath=True))
                bars.mb_gate_done[gate_idx].arrive()

                # ---- v - k*state (packed 16-bit; V ring cursor survives
                # item boundaries, so no fixed SMEM stage assumption) -------
                v_idx = v_index.idx
                bars.mb_v_ready[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)

                v_words = [[cutlass.Int32(0), cutlass.Int32(0)] for _ in range(16)]
                for c in cutlass.range_constexpr(8):
                    m0 = cutlass.const_expr(c % 4)
                    sub = cutlass.const_expr(c // 4)
                    v_f16 = nvvm.ldmatrix(
                        (sV_base + v_idx * v_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        v_words[4 * m0 + i][sub] = v_f16[i]
                if valid_state:
                    bars.mb_ks_ready[0].wait(ks_rdy_index.phase)
                    ks_rdy_index = advance(ks_rdy_index, 1)

                    for sub in cutlass.range_constexpr(2):
                        ks_vec = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_ks_col, cutlass.Float32),
                            num=8,
                        )
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(ks_vec[2 * j], ks_vec[2 * j + 1], gCumprod[2 * j], gCumprod[2 * j + 1])
                            ks_word = fp32_to_fp16(s0, s1, dtype=cfg.io_dtype)
                            v_words[j][sub] = sub_f16x2(v_words[j][sub], ks_word, cfg.io_dtype)
                for sub in cutlass.range_constexpr(2):
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_vks_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(v_words[j][sub] for j in range(16)), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_vks_inp_ready[0].arrive()

                # ---- state*q_epi:  QS *= cumprod * scale, in place -------
                if valid_state:
                    qs_idx = o_acc_rdy_index.idx
                    bars.mb_o_acc_ready[qs_idx].wait(o_acc_rdy_index.phase)
                    o_acc_rdy_index = advance(o_acc_rdy_index, cfg.tmem_q_state_acc_stages)

                    qs_ptrs = []
                    qs_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        qs_ptrs.append(nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_q_state_col + qs_idx * ACC_STAGE_COLS, cutlass.Float32))
                        qs_vecs.append(nvvm.tcgen05_ld("16x256b", qs_ptrs[sub], num=8))
                    for sub in cutlass.range_constexpr(2):
                        qs_scaled = []
                        for j in cutlass.range_constexpr(16):
                            p0, p1 = fmul2(qs_vecs[sub][2 * j], qs_vecs[sub][2 * j + 1], gCumprod[2 * j], gCumprod[2 * j + 1])
                            s0, s1 = fmul2(p0, p1, scale, scale)
                            qs_scaled += [s0, s1]
                        nvvm.tcgen05_st("16x256b", qs_ptrs[sub], cutlass.Vector.from_elements(tuple(qs_scaled), cutlass.Float32))
                    nvvm.tcgen05_wait("store")

                    bars.mb_o_state_scale_acc_done[qs_idx].arrive()

                # ---- new_v_epi + decay_v publish -------------------------
                bars.mb_nv_ready[0].wait(nv_rdy_index.phase)
                nv_rdy_index = advance(nv_rdy_index, 1)
                bars.mb_v_done[v_idx].arrive()

                nv_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
                # NV reuses the VKS slot; both publishes share one fence
                nv_vecs = []
                for sub in cutlass.range_constexpr(2):
                    nv_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_nv_col, cutlass.Float32),
                            num=8,
                        )
                    )
                for sub in cutlass.range_constexpr(2):
                    for k in cutlass.range_constexpr(32):
                        nv_regs[k][sub] = nv_vecs[sub][k]

                    nv_f16 = [fp32_to_fp16(nv_regs[2 * j][sub], nv_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_nv_inp_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(nv_f16), cutlass.Int32),
                    )
                    for j in cutlass.range_constexpr(16):
                        nv_regs[2 * j][sub], nv_regs[2 * j + 1][sub] = fmul2(
                            nv_regs[2 * j][sub], nv_regs[2 * j + 1][sub], gDecayScale[2 * j], gDecayScale[2 * j + 1]
                        )
                    decay_f16 = [fp32_to_fp16(nv_regs[2 * j][sub], nv_regs[2 * j + 1][sub], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_decay_v_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(decay_f16), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_nv_inp_ready[0].arrive()
                bars.mb_decay_v_inp_ready[0].arrive()

                # ---- qkv_epilogue: O acc -> sO ---------------------------
                if cutlass.const_expr(cfg.enable_o):
                    o_idx = o_index.idx
                    bars.mb_o_tmastg_done[o_idx].wait(o_index.phase)
                    o_index = advance(o_index, cfg.smem_o_stages)
                qs2_idx = o_scale_rdy_index.idx
                bars.mb_o_state_scale_acc_ready[qs2_idx].wait(o_scale_rdy_index.phase)
                o_scale_rdy_index = advance(o_scale_rdy_index, cfg.tmem_q_state_acc_stages)

                if cutlass.const_expr(cfg.enable_o):
                    o_regs = []
                    for sub in cutlass.range_constexpr(2):
                        o_vec = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_warp_row + sub * 16) << 16) + tmem_q_state_col + qs2_idx * ACC_STAGE_COLS, cutlass.Float32),
                            num=8,
                        )
                        o_regs.append([o_vec[k] for k in range(32)])
                # REQUIRED: an arrive does not order in-flight tcgen05 loads
                nvvm.tcgen05_wait("load")
                bars.mb_o_acc_done[qs2_idx].arrive()
                if cutlass.const_expr(cfg.enable_o):
                    for sub in cutlass.range_constexpr(2):
                        for m0 in cutlass.range_constexpr(4):
                            o_f16 = [fp32_to_fp16(o_regs[sub][8 * m0 + 2 * j], o_regs[sub][8 * m0 + 2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]
                            nvvm.stmatrix(
                                (
                                    sO_base + o_idx * o_stage_elems + ov_slab + (ov_tok + m0 * 16) * 64 + swizzle_xor_128b(ov_tok + m0 * 16, ov_col + sub * 16)
                                ).raw_ptr(),
                                o_f16,
                                nvvm.MMALayout.COL,
                            )
                    nvvm.fence_proxy("async.shared", space="cta")

                    bars.mb_o_tmastg_ready[o_idx].arrive()

        # ---- final state S: state TMEM -> GMEM -------------------------
        if sk_nt > 0:
            # required even unstored: next item's decay_v must not race GEMM 7
            kv_last_idx = kv_acc_index.idx
            bars.mb_kv_acc_ready[kv_last_idx].wait(kv_acc_index.phase)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_kv_acc_stages)
            if cutlass.const_expr(cfg.store_final_state):
                # split-K: only the last chunk's owner holds the final state
                if cutlass.const_expr(cfg.split_k):
                    if wend == num_chunks_b:
                        gS_out = mS_out[None, None, head_idx, batch_idx]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            state_vec = nvvm.tcgen05_ld(
                                "32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width, cutlass.Float32), num=32
                            )
                            for k in cutlass.range_constexpr(32):
                                val = state_vec[k]
                                if cutlass.const_expr(cfg.state_dtype != cfg.acc_dtype):
                                    val = val.to(cfg.state_dtype)
                                gS_out[sub * ldtm_width + k, cg1_tidx] = val
                else:
                    gS_out = mS_out[None, None, head_idx, batch_idx]
                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b", nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width, cutlass.Float32), num=32
                        )
                        for k in cutlass.range_constexpr(32):
                            val = state_vec[k]
                            if cutlass.const_expr(cfg.state_dtype != cfg.acc_dtype):
                                val = val.to(cfg.state_dtype)
                            gS_out[sub * ldtm_width + k, cg1_tidx] = val
                bars.mb_kv_acc_scale_done[kv_last_idx].arrive()
            else:
                bars.mb_kv_acc_scale_done[kv_last_idx].arrive()
        else:
            # zero-length sequence: state passes through, pure GMEM
            if cutlass.const_expr(cfg.store_final_state):
                write_passthrough = True
                if cutlass.const_expr(cfg.split_k):
                    write_passthrough = wend == num_chunks_b
                if write_passthrough:
                    gS_out = mS_out[None, None, head_idx, batch_idx]
                    if cutlass.const_expr(cfg.use_initial_state):
                        gS_in = mS_init[None, None, head_idx, batch_idx]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for k in cutlass.range_constexpr(32):
                                gS_out[sub * ldtm_width + k, cg1_tidx] = gS_in[sub * ldtm_width + k, cg1_tidx]
                    else:
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for k in cutlass.range_constexpr(32):
                                gS_out[sub * ldtm_width + k, cg1_tidx] = cutlass.Float32(0.0).to(cfg.state_dtype)

        tile_idx, sched_state = _sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    # CG1 done with TMEM: release the MMA warp's dealloc
    bars.mb_tmem_done[0].arrive()

    if cutlass.const_expr(cfg.enable_o):
        for _ in range(cfg.smem_o_stages):
            bars.mb_o_tmastg_done[o_index.idx].wait(o_index.phase)
            o_index = advance(o_index, cfg.smem_o_stages)
    if cutlass.const_expr(cfg.enable_h):
        for _ in range(cfg.smem_h_stages):
            bars.mb_h_tmastg_done[h_cnt % cfg.smem_h_stages].wait(cutlass.Int32(1) ^ ((h_cnt // cfg.smem_h_stages) & cutlass.Int32(1)))
            h_cnt = h_cnt + 1


@cute.jit
def _build_descs(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    h_out: Optional[cute.Tensor],
    h_every_n: cutlass.Int32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """Build the 5 per-(b,h) TMA-descriptor arrays (Q, K, V, O, S) into
    ``tensormap_workspace``.  Compiled + launched separately from the main
    kernel and cached by input identity in the host bridge, so the builder
    launches do not recur in steady-state replay.

    The H descriptor is 3-D ``(dv, dk, h)`` over the packed
    ``[total_h, HO, DK, DV]`` H tensor; ``build_h_descs_kernel`` derives the
    per-sequence H offset from the token ``cu_seqlens`` ((seqlen-1)//h_every_n,
    prefix-summed), folds it and the head into GLOBAL_ADDRESS, and caps
    GLOBAL_DIM[2] to the per-sequence H count, so the store coordinate is the
    sequence-local H index."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    q_group = heads_out // h_q
    k_group = heads_out // h_k
    v_group = heads_out // h_v
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    granu = 128 // bpe
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]

    q_head0 = q[None, 0, None]
    k_head0 = k[None, 0, None]
    v_view = v[None, 0, None]
    v_head0 = cute.make_tensor(
        v_view.iterator,
        cute.make_layout((d_v, v_view.shape[0]), stride=(v_view.stride[1], v_view.stride[0])),
    )
    swz128 = _tma.TensorMapSwizzle.s128b
    base_desc_q = _tma.create_tensor_map_tiled_from_view(q_head0, box_dims=(bt, granu), stride_order=(1, 0), swizzle=swz128)
    base_desc_k = _tma.create_tensor_map_tiled_from_view(k_head0, box_dims=(bt, granu), stride_order=(1, 0), swizzle=swz128)
    base_desc_v = _tma.create_tensor_map_tiled_from_view(v_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)

    arr_words = (batch_size * heads_out) * TENSOR_MAP_QWORDS
    ws_iter = tensormap_workspace.iterator

    def sub_array(k):
        return cute.make_tensor(ws_iter + k * arr_words, cute.make_layout((arr_words,), stride=(1,)))

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
    if cutlass.const_expr(o is not None):
        o_row_stride, o_head_stride = o.stride[0], o.stride[1]
        o_view = o[None, 0, None]
        o_head0 = cute.make_tensor(
            o_view.iterator,
            cute.make_layout((d_v, o_view.shape[0]), stride=(o_view.stride[1], o_view.stride[0])),
        )
        base_desc_o = _tma.create_tensor_map_tiled_from_view(o_head0, box_dims=(granu, bt), stride_order=(0, 1), swizzle=swz128)
        build_qkv_load_descs_kernel(
            base_desc_o,
            sub_array(3),
            cu_seqlens,
            o,
            cutlass.Int32(batch_size),
            cutlass.Int32(heads_out),
            cutlass.Int32(1),
            cutlass.Int32(o_head_stride),
            cutlass.Int32(o_row_stride),
            1,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    if cutlass.const_expr(h_out is not None):
        d_k_s = h_out.shape[2]
        d_v_s = h_out.shape[3]
        h_granu = 128 // (h_out.element_type.width // 8)
        h_view = cute.make_tensor(
            h_out.iterator,
            cute.make_layout(
                (d_v_s, d_k_s, h_out.shape[0]),
                stride=(h_out.stride[3], h_out.stride[2], h_out.stride[0]),
            ),
        )
        base_desc_h = _tma.create_tensor_map_tiled_from_view(h_view, box_dims=(h_granu, d_k_s, 1), stride_order=(0, 1, 2), swizzle=swz128)
        build_h_descs_kernel(
            base_desc_h,
            sub_array(4),
            cu_seqlens,
            h_out,
            cutlass.Int32(batch_size),
            cutlass.Int32(heads_out),
            cutlass.Int32(h_out.stride[1]),
            cutlass.Int32(h_out.stride[0]),
            h_every_n,
            2,
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _downcast_state(
    s0: cute.Tensor,
    out: cute.Tensor,
    n: cutlass.Int32,
    n_blocks: cutlass.Int32,
    stream: cuda.CUstream,
):
    downcast_state_kernel(
        s0,
        out,
        n,
    ).launch(grid=(n_blocks, 1, 1), block=(128, 1, 1), stream=stream)


@functools.cache
def _downcast_state_cache(key):
    return {}


def downcast_state(initial_state, out, *, stream):
    """Copy the fp32 initial state ``[N, HO, K, V]`` into ``out`` (io dtype,
    same shape) — the buffer the backward's per-(b,h) state descriptors
    read."""
    n = 1
    for s_ in out.shape:
        n *= int(s_)
    key = (str(initial_state.dtype), str(out.dtype))
    cache = _downcast_state_cache(key)
    cu_stream = cuda.CUstream(int(stream))
    n_blocks = (n + 127) // 128
    s0_flat = initial_state.reshape(n)
    out_flat = out.reshape(n)
    if "compiled" not in cache:
        s0_c = from_dlpack(s0_flat, assumed_align=16).mark_layout_dynamic()
        out_c = from_dlpack(out_flat, assumed_align=16).mark_layout_dynamic()
        cache["compiled"] = cute.compile(
            _downcast_state,
            s0_c,
            out_c,
            cutlass.Int32(n),
            cutlass.Int32(n_blocks),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](s0_flat, out_flat, n, n_blocks, cu_stream)


@cute.jit
def _host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    o: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    s_in: Optional[cute.Tensor],
    s_out: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    sched_ctr: Optional[cute.Tensor],
    checkpoint_every_n_tokens: cutlass.Int32,
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
    if cutlass.const_expr(o is not None):
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
    if cutlass.const_expr(s_in is not None):
        s_in = cute.make_tensor(
            s_in.iterator,
            cute.make_layout(
                (s_in.shape[2], s_in.shape[3], (h_r, h_qv), s_in.shape[0]),
                stride=(
                    s_in.stride[2],
                    s_in.stride[3],
                    (s_in.stride[1], h_r * s_in.stride[1]),
                    s_in.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(s_out is not None):
        s_out = cute.make_tensor(
            s_out.iterator,
            cute.make_layout(
                (s_out.shape[2], s_out.shape[3], (h_r, h_qv), s_out.shape[0]),
                stride=(
                    s_out.stride[2],
                    s_out.stride[3],
                    (s_out.stride[1], h_r * s_out.stride[1]),
                    s_out.stride[0],
                ),
            ),
        )
    # ---- SMEM sizing: per-buffer element cosizes -----------------------
    bpe = cfg.io_dtype.width // 8
    q_tile_elems = cfg.b_t * cfg.d_k
    k_tile_elems = cfg.b_t * cfg.d_k
    v_tile_elems = cfg.d_v * cfg.b_t
    ainv_tile_elems = cfg.b_t * cfg.b_t
    qk_tile_elems = cfg.b_t * cfg.b_t
    o_tile_elems = cfg.d_v * cfg.b_t
    cfg.q_cosize = q_tile_elems * cfg.smem_q_stages
    cfg.k_cosize = k_tile_elems * cfg.smem_k_stages
    cfg.v_cosize = v_tile_elems * cfg.smem_v_stages
    cfg.ainv_cosize = ainv_tile_elems * cfg.smem_ainv_stages
    cfg.qk_cosize = qk_tile_elems * cfg.smem_qk_stages
    cfg.o_cosize = o_tile_elems * cfg.smem_o_stages
    cfg.h_cosize = cfg.d_k * cfg.d_v * cfg.smem_h_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))

    cfg.tma_q_bytes = q_tile_elems * bpe
    cfg.tma_k_bytes = k_tile_elems * bpe
    cfg.tma_v_bytes = v_tile_elems * bpe
    cfg.tma_o_bytes = o_tile_elems * bpe

    cfg.n_heads_out = heads_out
    num_descs = batch_size * heads_out

    # ---- launch --------------------------------------------------------
    total_tiles = batch_size * heads_out
    # CUDA-graph-stable launch: fixed SM-count grid; shapes ride on buffer contents
    grid_shape = (cfg.max_active_clusters, 1, 1)

    _kernel(
        cfg,
        gate,
        beta,
        cu_seqlens,
        s_in,
        s_out,
        work_items,
        work_count,
        sched_ctr,
        checkpoint_every_n_tokens,
        scale,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        total_tiles,
        q,
        k,
        v,
        o,
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
    cu_seqlens: cute.Tensor,
    mS_init: Optional[cute.Tensor],
    mS_out: Optional[cute.Tensor],
    mWorkItems: Optional[cute.Tensor],
    mCount: Optional[cute.Tensor],
    mSched: Optional[cute.Tensor],
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    total_tiles: cutlass.Int32,
    mQ,
    mK,
    mV,
    mO,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """
    Main GDN chunked kernel.

    Warp specialization is the outermost control flow: each warp role owns
    its own persistent tile-scheduler loop, iterating over (batch, head)
    tiles and then over chunks within each tile.
    """
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    if cutlass.const_expr(cfg.split_k):
        total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None, "mSched must be provided if dyn_sched is True"

    if cutlass.const_expr(cfg.use_initial_state):
        assert mS_init is not None, "mS_init must be provided if use_initial_state is True"
    else:
        assert mS_init is None, "mS_init must be None if use_initial_state is False"
    if cutlass.const_expr(cfg.store_final_state):
        assert mS_out is not None, "mS_out must be provided if store_final_state is True"
    else:
        assert mS_out is None, "mS_out must be None if store_final_state is False"

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_h_base = desc_base_words + cutlass.Int32(4) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_gdn_bars(cfg)
    tmem_hold = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=128)

    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = (cfg.d_v // 2) * 128
    V_LEAD = (cfg.d_v // 2) * 128
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
    # descriptor operands total 136 KB: V/O/H land past the sub-bank midpoint
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
    sO_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.o_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sO = SmemTile(
        base=sO_raw.data_ptr().toint(),
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages) * bpe,
        stages=cfg.smem_o_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    if cutlass.const_expr(cfg.enable_h):
        sH_raw = cutlass.Array(
            cfg.io_dtype,
            cfg.h_cosize,
            space=cutlass.AddressSpace.smem,
            alignment=cfg.buffer_align_bytes,
        )
    else:
        sH_raw = None
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
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.smem_beta_stages):
        bars.mb_beta_ready[s].init()
        bars.mb_beta_done[s].init()
    for s in range(cfg.tmem_kv_acc_stages):
        bars.mb_kv_acc_ready[s].init()
        bars.mb_kv_acc_scale_done[s].init()
    for s in range(cfg.tmem_q_state_acc_stages):
        bars.mb_o_acc_ready[s].init()
        bars.mb_o_acc_done[s].init()
        bars.mb_o_state_scale_acc_ready[s].init()
        bars.mb_o_state_scale_acc_done[s].init()
    for s in range(cfg.tmem_cg0_acc_stages):
        bars.mb_cg0_acc_ready[s].init()
        bars.mb_cg0_acc_done[s].init()
    bars.mb_ks_ready[0].init()
    bars.mb_nv_ready[0].init()
    for s in range(cfg.smem_ainv_stages):
        bars.mb_ainv_ready[s].init()
        bars.mb_ainv_done[s].init()
    for s in range(cfg.smem_qk_stages):
        bars.mb_qk_ready[s].init()
        bars.mb_qk_done[s].init()
    for s in range(cfg.tmem_state_inp_stages):
        bars.mb_state_inp_ready[s].init()
    for b in (bars.mb_vks_inp_ready, bars.mb_nv_inp_ready, bars.mb_decay_v_inp_ready):
        b[0].init()
    for s in range(cfg.smem_o_stages):
        bars.mb_o_tmastg_ready[s].init()
        bars.mb_o_tmastg_done[s].init()
    for s in range(cfg.smem_h_stages):
        bars.mb_h_tmastg_ready[s].init()
        bars.mb_h_tmastg_done[s].init()
    for s_ in range(cfg.sched_stages):
        bars.mb_sched_ready[s_].init()
        bars.mb_sched_done[s_].init()
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
            sBeta=sBeta,
            sAinv=sAinv,
            sQk=sQk,
            sH_raw=sH_raw,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
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
            tidx,
            warp_idx=warp_idx,
            tmem_hold=tmem_hold,
            scale=scale,
            sV=sV,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sO=sO,
            sH_raw=sH_raw,
            mS_init=mS_init,
            mS_out=mS_out,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.mma_warp_id:
        _mma0_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tmem_hold=tmem_hold,
            sQ=sQ,
            sK=sK,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.mma_cg1_warp_id:
        _mma1_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tmem_hold=tmem_hold,
            sQ=sQ,
            sK=sK,
            sK_trans=sK_trans,
            sAinv=sAinv,
            sQk=sQk,
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
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            mSched=mSched,
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
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            tidx=tidx,
            mGate=mGate,
            mBeta=mBeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sO_raw=sO_raw,
            sH_raw=sH_raw,
            desc_o_base=desc_o_base,
            desc_h_base=desc_h_base,
            sSched=sSched,
            bars=bars,
        )


@dataclass
class GdnCfg:
    """Per-compile GDN kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA, state flags) are the
    ``cute.compile`` cache keys; the rest is derived from the module-global
    ``CFG`` constants.  ``_host`` stamps the shape-derived fields at trace
    time.  Passed ``cfg``-first (a ``cutlass.Constexpr``) into ``_host`` /
    ``_kernel`` and every warp body.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    is_GQA: bool
    use_initial_state: bool
    store_final_state: bool
    enable_h: bool
    enable_o: bool
    split_k: bool = False
    log_gate: bool = False
    dyn_sched: bool = False
    sched_stages: int = CFG.SMEM_SCHED_STAGES

    # --- fixed constants stamped from CFG by build_cfg ---
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    mma_warp_id: int = CFG.MMA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    mma_cg1_warp_id: int = CFG.MMA_CG1_WARP_ID
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # --- named barrier slots (ids 1-4; 0 is the CTA-wide sync) ---
    tmem_alloc_barrier_id: int = 1
    tmem_alloc_barrier_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0
    init_state_store_barrier_id: int = 4
    init_state_store_barrier_threads: int = 0

    # --- SMEM / TMEM stage counts + TMEM column offsets ---
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_ainv_stages: int = CFG.SMEM_AINV_STAGES
    smem_qk_stages: int = CFG.SMEM_QK_STAGES
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_h_stages: int = 1
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    smem_beta_stages: int = CFG.SMEM_BETA_STAGES
    tmem_kv_acc_stages: int = CFG.TMEM_KV_ACC_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES
    tmem_state_inp_stages: int = CFG.TMEM_STATE_INP_STAGES
    tmem_cg0_acc_stages: int = CFG.TMEM_CG0_ACC_STAGES
    tmem_cg1_acc_stages: int = CFG.TMEM_CG1_ACC_STAGES
    tmem_state_offset: int = 0
    tmem_q_state_offset: int = 0
    tmem_state_inp_offset: int = 0
    tmem_cg0_acc_offset: int = 0
    tmem_cg1_acc_offset: int = 0
    tmem_inp_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # --- stamped by _host at trace time (shape-derived) ---
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    ainv_cosize: int = 0
    qk_cosize: int = 0
    o_cosize: int = 0
    h_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_o_bytes: int = 0
    n_heads_out: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool = True,
    enable_h: bool = False,
    enable_o: bool = True,
    split_k: bool = False,
    log_gate: bool = False,
    dyn_sched: bool = False,
) -> GdnCfg:
    """Build the per-compile ``GdnCfg`` (io_dtype ∈ {Float16, BFloat16};
    acc is always Float32).  Fills the derived thread / barrier counts and
    the TMEM column offsets."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = GdnCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        is_GQA=is_GQA,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_h=enable_h,
        enable_o=enable_o,
        split_k=split_k,
        log_gate=log_gate,
        dyn_sched=dyn_sched,
    )
    cfg.smem_h_stages = 1
    if enable_h:
        # trim K/V lookahead so the 32 KB H buffer fits
        cfg.smem_k_stages = 3
        cfg.smem_v_stages = 2
    if not use_initial_state:
        # fund the lightweight warps from CG1 (peeled chunk carries more cursors)
        cfg.num_regs_compute_group_1 = 232
        cfg.num_regs_other = 48
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1)
    # both MMA issuer warps join the TMEM alloc barrier
    cfg.tmem_alloc_barrier_threads = cfg.threads_per_warp * (2 + n_cg0 + n_cg1)
    cfg.inverse_barrier_threads = cfg.threads_per_warp * n_cg0
    cfg.init_state_store_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.tmem_state_offset = 0
    cfg.tmem_q_state_offset = cfg.tmem_state_offset + cfg.tmem_kv_acc_stages * 128
    cfg.tmem_state_inp_offset = cfg.tmem_q_state_offset + cfg.tmem_q_state_acc_stages * 64
    cfg.tmem_cg0_acc_offset = cfg.tmem_state_inp_offset + cfg.tmem_state_inp_stages * 64
    cfg.tmem_cg1_acc_offset = cfg.tmem_cg0_acc_offset + cfg.tmem_cg0_acc_stages * 64
    cfg.tmem_inp_offset = cfg.tmem_cg1_acc_offset + cfg.tmem_cg1_acc_stages * 64
    return cfg


def get_workspace_size(B: int, HQ: int, HV: int):
    HO = HQ if HQ >= HV else HV
    return CFG.BYTES_PER_TENSORMAP * (5 * B * HO) + 128


# ---------------------------------------------------------------------------


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


def _cutlass_state_dtype(dtype):
    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float32" in name:
        return cutlass.Float32
    raise ValueError(f"Unsupported state dtype {dtype}, expected float32 or bfloat16")


@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_h: bool,
    enable_o: bool,
    split_k: bool,
    log_gate: bool,
    dyn_sched: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    state_dtype,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_h: bool,
    enable_o: bool,
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
    o_cute,
    cu_seqlens_cute,
    s_in_cute,
    s_out_cute,
    work_items_cute=None,
    work_count_cute=None,
    sched_ctr_cute=None,
    checkpoint_every_n_tokens,
    scale,
    workspace_cute,
    stream,
):
    """JIT-compile the chunked GDN prefill kernel for one static config."""
    cfg = build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sm,
        is_GQA=is_GQA,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_h=enable_h,
        enable_o=enable_o,
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
        o_cute,
        cu_seqlens_cute,
        s_in_cute,
        s_out_cute,
        work_items_cute,
        work_count_cute,
        sched_ctr_cute,
        checkpoint_every_n_tokens,
        scale,
        workspace_cute,
        stream,
        options="--enable-tvm-ffi --opt-level 3",
    )


def _stream_capturing(stream) -> bool:
    """True when ``stream`` is inside CUDA-graph capture."""
    from cuda.bindings import runtime as _rt

    err, status = _rt.cudaStreamIsCapturing(int(stream))
    return int(err) == 0 and int(status) != 0


def chunk_gdn_sm100(
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
    output_h=None,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    log_gate: bool = False,
    *,
    workspace,
    stream,
) -> None:
    """Execute the Blackwell chunked GDN prefill kernel (THD / varlen entry).

    All tensors are contiguous, DLPack-compatible CUDA tensors on the same
    device.  Compile-cache-and-replay: the kernel is compiled once per static
    config (dtypes, head counts, state flags) and replayed afterwards.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate — raw linear
            alpha, or the natural-log decay when ``log_gate``
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated,
            or None to skip the O output path entirely (state/H-only run;
            requires output_h or output_state)
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: emit an H entry every N tokens (0 = off)
        output_h: ``(total_h, HO, DK, DK)`` bfloat16, or None.  H[j] is the
            state after ``(j + 1) * N`` tokens, STRICTLY BEFORE the sequence
            end -- the end-of-sequence state is only ``output_state`` (S,
            fp32-capable).  With ``N == B_T`` this is the per-chunk state
            series the backward pass consumes.
        work_items: ``(max_items, 6)`` int32 split-K work-item table from
            ``common/split_k.py``, or None for the one-tile-per-(b,h)
            layout.  With a table, each item computes chunks
            ``[cstart, wend)`` and writes O/H only for ``[wstart, wend)``.
        work_count: ``(1,)`` int32 device-side item count (required with
            work_items)
        log_gate: ``gate`` holds natural-log decay values; the gate warp
            skips its log2 (rescales by 1/ln2) instead of exponentiating
            upstream
        workspace: ``(>= get_workspace_size(B, HQ, HV) // 8,)`` int64,
            128-byte aligned; holds the per-(b,h) TMA descriptors (contents
            managed here — reuse the same buffer across calls)
        stream: CUDA stream handle (``cudaStream_t`` as an int)
    """
    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint_every_n_tokens must be non-negative")
    if checkpoint_every_n_tokens > 0:
        if checkpoint_every_n_tokens % CFG.B_T != 0:
            raise ValueError(f"checkpoint_every_n_tokens must be a multiple of the chunk " f"size ({CFG.B_T}), got {checkpoint_every_n_tokens}")
        if output_h is None:
            raise ValueError("output_h must be provided when checkpoint_every_n_tokens > 0")
        if str(output_h.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
            raise ValueError(f"output_h dtype must match the io dtype (fp32 state belongs to " f"output_state): got {output_h.dtype} with io {q.dtype}")
    elif output_h is not None:
        raise ValueError("output_h must be None when checkpoint_every_n_tokens == 0")
    HQ = q.shape[1]
    HV = v.shape[1]
    DK = q.shape[2]
    B = cu_seqlens.shape[0] - 1
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_h = checkpoint_every_n_tokens > 0
    enable_o = output is not None
    split_k = work_items is not None
    dyn_sched = sched_ctr is not None
    if not enable_o and not (enable_h or store_final_state):
        raise ValueError("output=None requires output_h or output_state")
    if split_k:
        if work_count is None:
            raise ValueError("work_count is required with work_items")
        if enable_h and checkpoint_every_n_tokens != CFG.B_T:
            raise ValueError(f"split-K H checkpoints require checkpoint_every_n_tokens == {CFG.B_T}, got {checkpoint_every_n_tokens}")
    elif work_count is not None:
        raise ValueError("work_count must be None without work_items")
    io_dtype = _cutlass_io_dtype(q.dtype)

    if initial_state is not None:
        state_dtype_src = initial_state.dtype
    elif output_state is not None:
        state_dtype_src = output_state.dtype
    else:
        state_dtype_src = None
    state_dtype = _cutlass_state_dtype(state_dtype_src) if state_dtype_src is not None else cutlass.Float32

    ws_words = get_workspace_size(B, HQ, HV) // 8
    if workspace.shape[0] < ws_words:
        raise ValueError(f"workspace too small: need {ws_words} int64 words, " f"got {workspace.shape[0]}")
    if _data_ptr(workspace) % 128 != 0:
        raise ValueError("workspace must be 128-byte aligned")
    cu_stream = cuda.CUstream(int(stream))

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_dtype_src),
        HQ,
        k.shape[1],
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_h,
        enable_o,
        split_k,
        log_gate,
        dyn_sched,
    )

    if "compiled" not in cache:
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        o_cute = None
        if enable_o:
            o_cute = from_dlpack(output, assumed_align=16)
            o_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(initial_state, assumed_align=16)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(output_state, assumed_align=16)
            s_out_cute.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)

        workspace_cute = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()

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
            state_dtype,
            is_GQA,
            use_initial_state,
            store_final_state,
            enable_h,
            enable_o,
            split_k,
            log_gate,
            dyn_sched,
            num_sm=_device_sm_count(),
            q_cute=q_cute,
            k_cute=k_cute,
            v_cute=v_cute,
            gate_cute=gate_cute,
            beta_cute=beta_cute,
            o_cute=o_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            s_in_cute=s_in_cute,
            s_out_cute=s_out_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            sched_ctr_cute=sched_ctr_cute,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            scale=scale,
            workspace_cute=workspace_cute,
            stream=cu_stream,
        )

    compiled = cache["compiled"]

    # desc key: cu object identity + _version so address reuse forces a rebuild
    desc_key = (
        _data_ptr(q),
        _data_ptr(k),
        _data_ptr(v),
        _data_ptr(output) if enable_o else 0,
        _data_ptr(output_h) if enable_h else 0,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(output.shape) if enable_o else (),
        tuple(output_h.shape) if enable_h else (),
        int(B),
        int(checkpoint_every_n_tokens) if enable_h else (),
    )
    cu_versions = (getattr(cu_seqlens, "_version", 0),)
    if (
        cache.get("desc_key") != desc_key
        or cache.get("desc_cu") is not cu_seqlens
        or cache.get("desc_cu_versions") != cu_versions
        or cache.get("desc_workspace") is not workspace
        or _stream_capturing(stream)
    ):
        if "build_descs" not in cache:
            q_bc = from_dlpack(q, assumed_align=16)
            q_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            k_bc = from_dlpack(k, assumed_align=16)
            k_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            v_bc = from_dlpack(v, assumed_align=16)
            v_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            o_bc = None
            if enable_o:
                o_bc = from_dlpack(output, assumed_align=16)
                o_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
            cu_bc = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
            s_bc = None
            cu_ckpt_bc = None
            if enable_h:
                s_bc = from_dlpack(output_h, assumed_align=16)
                s_bc.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2, 3), divisibility=1)
            ws_bc = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
            cache["build_descs"] = cute.compile(
                _build_descs,
                io_dtype,
                CFG.B_T,
                q_bc,
                k_bc,
                v_bc,
                o_bc,
                cu_bc,
                s_bc,
                cutlass.Int32(checkpoint_every_n_tokens if enable_h else 1),
                ws_bc,
                cu_stream,
                options="--enable-tvm-ffi",
            )
        cache["build_descs"](
            q,
            k,
            v,
            output,
            cu_seqlens,
            output_h,
            checkpoint_every_n_tokens if enable_h else 1,
            workspace,
            cu_stream,
        )
        cache["desc_key"] = desc_key
        cache["desc_cu"] = cu_seqlens
        cache["desc_cu_versions"] = cu_versions
        cache["desc_workspace"] = workspace

    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        work_items,
        work_count,
        sched_ctr,
        checkpoint_every_n_tokens,
        scale,
        workspace,
        cu_stream,
    )
