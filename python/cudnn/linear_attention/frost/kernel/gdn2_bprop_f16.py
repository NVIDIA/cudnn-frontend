"""Chunked Gated DeltaNet v2 (GDN-2) BPROP kernel for Blackwell SM100/SM103
(Cutlass DSL), BT=16 tiling with a per-key-channel decay.  Framework-neutral
entry ``chunk_gdn2_bwd_sm100``.

Algorithm overview (per chunk c, iterated c = NT-1 .. 0; within-chunk
log2-domain gate cumsum G[t,d], eG = 2^G, eGl = 2^G[BT-1]):
  Inputs : Q/K[BT,DK], V/dO[BT,DV], S = checkpoint[c-1] (state ENTERING chunk c, KV),
           Gate[BT,DK], Beta[BT,DK] (per-key erase), W[BT,DV] (per-value write)
  State  : dH[DV,DK] (state gradient, fp32 TMEM, accumulated backward)

  Operands (WG0, prefill recompute): K_decay = eG.(Beta.K) (erase key, Beta
  folded), K_inv = K/eG, K_restore = (eGl/eG).K, Q_decay = eG.Q, diag(eGl).

  Forward recompute: T_inv = (I + strict_tril(K_decay@K_inv^T))^-1 (register Neumann,
  Beta pre-folded); A = tril_incl(Q_decay@K_inv^T); Y = W.V - S^T K_decay; U = T_inv@Y.

  Backward math:
  dU  = K_restore@dH + A^T@dO (Q_decay carries scale, so A does too)
  dY  = T_inv^T@dU
  dV  = W.dY                          dW_out = V.dY      (elementwise, WG1)
  dA  = tril_incl(dO@U^T) (unscaled)  dM = dY@U^T        dM_strict = +strict(dM)
  dQ  = eG.scale.(dO@S^T + dA@K_inv)
  dK  = Beta.eG.dK_decay + dK_inv/eG + (eGl/eG).dK_restore  where (sign-flipped parts)
        dK_decay part = dY@S^T + dM_strict@K_inv          (= -dK_decay)
        dK_inv part = dA^T@(scale.Q_decay) - dM_strict^T@K_decay   (= dK_inv; one TMEM
                acc, the minus rides the staged -dM_strict tile)
        dK_restore part = U@dH^T                     (= +dK_restore)
  dBeta[t,d] = k_n.eG.dK_decay = -k_n.eG.dK_decay part   (per-channel, WG2)
  dGate[t,d] = q_n.dQ_pre + Beta.dBeta + k_n.(dK_inv_part/eG
            - (eGl/eG).dK_restore_part)
  dGate_last[d] = eGl.sum_v(dH.S) + sum_t k_n.(eGl/eG).dK_restore_part
  dGate = suffix-sum(dGate + dGate_last at row BT-1) (WG2 in-register reverse cumsum)
  dH <- diag-GEMM(eGl).dH + (scale.Q_decay)^T@dO - K_decay^T@dY

ABI: state_checkpoints `[total_checkpoints, HO, DK, DV]` (KV, v contiguous - the GDN checkpoint layout) io
dtype, the plain per-chunk series with NO initial-state slot (entry `c
- 1` = state entering chunk c >= 1; chunk 0 seeds from `initial_state`); beta `[T, HO, DK]` / w `[T, HO, DV]` io dtype
(post-sigmoid); dq/dk/dv io at HO heads; dgate `[T, HO, DK]` fp32 (natural-log
gate domain); dbeta/dw io dtype like beta/w; d_initial_state / d_final_state
fp32 `[N, HO, DK, DV]` (K-major, matching the prefill states).

Warp assignments (16 warps = 512 threads):
  warps 0-3  : WG0 - Gate prefix scan + decay/restore operands (all chunks)
  warps 4-7  : WG1 - value-side TMEM staging, restages, dstate capture, dV/dW_out
  warps 8-11 : WG2 - dQ/dK part drain, dGate/dBeta assembly, reverse cumsum
  warp  12   : super-MMA - register KK/A/dA/dM + Neumann inverse
  warp  13   : tcgen05-MMA - the backward schedule
  warp  14   : TMA load - Q/K/V/Gate/dO/state(checkpoint) loads + Beta/W tiles
  warp  15   : epilogue - dQ/dK/dV/dGate/dBeta/dW_out TMA stores
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Optional, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMS, ORDER_THREADS, decode_work_item, order_body
from ..common.host import get_dtype
from cudnn.frost.buffers import current_device_id, data_ptr
from cudnn.frost.device import multiprocessor_count
from ..common.thd import TENSOR_MAP_QWORDS, emit_copy_desc, emit_checkpoint_seq_descs, emit_seq_descs
from .gdn2_bprop_config import CFG

from cudnn.frost.tile_dsl.barrier import (
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.mma import mma_ss, mma_step, mma_ts_step
from cudnn.frost.tile_dsl.swizzle import swizzle_lin_S, swizzle_xor_128b
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from cudnn.frost.tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    movmatrix_16b,
    mul_f16x2,
    opaque_f32_zero,
    sub_f16x2,
)

LOG2_E: float = 1.4426950408889634


DEFAULT_GATE_LOWER_BOUND: float = -5.0


L2_NORM_EPS: float = 1.0e-12


class Gdn2BwdBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring.  Consumers
    track ``(idx, phase)`` inline; the producer tag selects the arrive
    lowering (``TMA_LOAD``/``MMA_COMMIT``/``THREAD``).

    Buffers read by both the MMA warp and a compute/warp group carry mixed
    arrive counts (one MMA commit + N thread arrivers) so the producer only
    reuses the slot once every reader is done."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_done: MBarrier
    mb_do_mma_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_w_ready: MBarrier
    mb_w_done: MBarrier
    mb_state_ready: MBarrier
    mb_state_done: MBarrier
    mb_state_cg0_done: MBarrier
    mb_state_inp_ready: MBarrier
    mb_state_inp_done: MBarrier
    mb_state_inp_cg2_done: MBarrier

    mb_k_decay_inv_ready: MBarrier
    mb_q_decay_k_restore_ready: MBarrier
    mb_decay_done: MBarrier

    mb_t_inv_ready: MBarrier
    mb_a_ready: MBarrier
    mb_da_ready: MBarrier
    mb_dm_ready: MBarrier
    mb_a_done: MBarrier
    mb_t_inv_done: MBarrier
    mb_da_done: MBarrier
    mb_dm_done: MBarrier

    mb_state_k_acc_ready: MBarrier
    mb_y_inp_ready: MBarrier
    mb_u_acc_ready: MBarrier
    mb_u_smem_ready: MBarrier
    mb_du_acc_ready: MBarrier
    mb_du_inp_ready: MBarrier
    mb_dy_acc_ready: MBarrier
    mb_neg_dy_inp_ready: MBarrier
    mb_dy_smem_ready: MBarrier
    mb_dy_smem_done: MBarrier
    mb_dstate_acc_ready: MBarrier
    mb_dstate_inp_ready: MBarrier
    mb_dstate_smem_ready: MBarrier
    mb_dstate_smem_done: MBarrier
    mb_dstate_smem_cg2_done: MBarrier

    mb_dq_acc_ready: MBarrier
    mb_dk_decay_part_acc_ready: MBarrier
    mb_dk_inv_part_acc_ready: MBarrier
    mb_dk_restore_part_acc_ready: MBarrier
    mb_dqk_acc_done: MBarrier

    mb_qk_raw_ready: MBarrier
    mb_qk_raw_done: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier
    mb_dgate_tmastg_ready: MBarrier
    mb_dgate_tmastg_done: MBarrier
    mb_db_tmastg_ready: MBarrier
    mb_db_tmastg_done: MBarrier
    mb_dwo_tmastg_ready: MBarrier
    mb_dwo_tmastg_done: MBarrier
    mb_dstate0_acc_stored: MBarrier
    mb_tmem_done: MBarrier

    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_gdn2_bwd_bars(cfg) -> Gdn2BwdBars:
    """Bars factory.  MUST be called from inside ``kernel`` (allocates the
    mbarrier rings in SMEM ahead of the data buffers)."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0 = len(cfg.compute_group_0_warp_ids) * WARP
    CG2 = len(cfg.compute_group_2_warp_ids) * WARP
    CG1 = len(cfg.compute_group_1_warp_ids) * WARP
    MMA = 1

    return Gdn2BwdBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0 + CG2, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0 + CG2, producer=Producer.THREAD),
        mb_do_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_do_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=WARP, producer=Producer.THREAD),
        mb_do_mma_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_v_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1, producer=Producer.THREAD),
        mb_w_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_w_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1, producer=Producer.THREAD),
        mb_state_ready=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_state_done=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_state_cg0_done=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=CG0, producer=Producer.THREAD),
        mb_state_inp_ready=MBarrier(alloc(2), stages=2, init_count=CG0, producer=Producer.THREAD),
        mb_state_inp_done=MBarrier(alloc(2), stages=2, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_state_inp_cg2_done=MBarrier(alloc(2), stages=2, init_count=CG2, producer=Producer.THREAD),
        mb_k_decay_inv_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_q_decay_k_restore_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_decay_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_da_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dm_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_t_inv_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_da_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dm_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_state_k_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_y_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_u_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_du_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_du_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dy_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_neg_dy_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dy_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dy_smem_done=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_smem_done=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_smem_cg2_done=MBarrier(alloc(1), stages=1, init_count=CG2, producer=Producer.THREAD),
        mb_dq_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_decay_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_inv_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_restore_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dqk_acc_done=MBarrier(alloc(1), stages=1, init_count=CG2, producer=Producer.THREAD),
        mb_qk_raw_ready=MBarrier(alloc(cfg.tmem_qk_raw_stages), stages=cfg.tmem_qk_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_qk_raw_done=MBarrier(alloc(cfg.tmem_qk_raw_stages), stages=cfg.tmem_qk_raw_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dq_tmastg_ready=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dq_tmastg_done=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dk_tmastg_ready=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dk_tmastg_done=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dv_tmastg_ready=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=CG1, producer=Producer.THREAD),
        mb_dv_tmastg_done=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dgate_tmastg_ready=MBarrier(alloc(cfg.smem_dgate_stages), stages=cfg.smem_dgate_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dgate_tmastg_done=MBarrier(alloc(cfg.smem_dgate_stages), stages=cfg.smem_dgate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_db_tmastg_ready=MBarrier(alloc(cfg.smem_db_stages), stages=cfg.smem_db_stages, init_count=CG2, producer=Producer.THREAD),
        mb_db_tmastg_done=MBarrier(alloc(cfg.smem_db_stages), stages=cfg.smem_db_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dwo_tmastg_ready=MBarrier(alloc(cfg.smem_dwo_stages), stages=cfg.smem_dwo_stages, init_count=CG1, producer=Producer.THREAD),
        mb_dwo_tmastg_done=MBarrier(alloc(cfg.smem_dwo_stages), stages=cfg.smem_dwo_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dstate0_acc_stored=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1 + CG2, producer=Producer.THREAD),
        mb_sched_ready=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=1, producer=Producer.THREAD),
        mb_sched_done=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=15, producer=Producer.THREAD),
    )


# ---- Dynamic tile scheduler ------------------------------------------------------


@cute.jit
def sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas):
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
def sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas):
    """Consumer side: read the TMA warp's published next tile."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_ready[sched_state.idx].wait(sched_state.phase)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_done[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


# ---- Warp bodies -----------------------------------------------------------------


@cute.jit
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    sK_inv_raw,
    sQ_decay_raw,
    sDo_raw,
    sU_raw,
    sIntermediate_raw,
    sDq_raw,
    sDk_raw,
    sDv_raw,
    sDgate_raw,
    sDb_raw,
    sDwOut_raw,
    desc_dq_base,
    desc_dk_base,
    desc_dv_base,
    desc_dgate_base,
    desc_db_base,
    desc_dwo_base,
    bars,
) -> None:
    """Epilogue warp role (warp 15): register-MMA A/dA tiles and the
    gradient TMA stores, in chunk order with a one-behind store ladder."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    # ---- ldmatrix/stmatrix lane decode -------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + stsm_col_coord, bbits=1, mbase=3, sshift=3)
    row_lo = lane // 4
    row_hi = row_lo + cutlass.Int32(8)

    # hoisted tril bitmask: bit i = row >= col for accum index i
    tril_incl_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_incl_mask = tril_incl_mask | (cutlass.Int32(1 << accum_idx) if row_coord >= col_coord else cutlass.Int32(0))
    raw_index = PipelineState.start(phase=0)
    u_index = PipelineState.start(phase=0)
    gbase = cutlass.Int32(0)

    sDq_tma = SmemTile(
        base=sDq_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDk_tma = SmemTile(
        base=sDk_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dk_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDv_tma = SmemTile(
        base=sDv_raw,
        elems_per_stage=(cfg.b_t * cfg.d_v),
        stages=cfg.smem_dv_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDgate_tma = SmemTile(
        base=sDgate_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dgate_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 32),
        tma_granu_elems=32,
        tma_subtile_stride_elems=cfg.b_t * 32,
    )
    sDb_tma = SmemTile(
        base=sDb_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_db_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDwOut_tma = SmemTile(
        base=sDwOut_raw,
        elems_per_stage=(cfg.b_t * cfg.d_v),
        stages=cfg.smem_dwo_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    dgate_index = PipelineState.start(phase=0)
    db_index = PipelineState.start(phase=0)
    dwo_index = PipelineState.start(phase=0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        if elect_one:
            desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_db_slot = (desc_db_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dwo_slot = (desc_dwo_base + slot).tospace(cutlass.AddressSpace.generic)
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)
            tma_tensormap_acquire(desc_dgate_slot)
            tma_tensormap_acquire(desc_db_slot)
            tma_tensormap_acquire(desc_dwo_slot)
        sk_nt = cend - wstart
        pend_start = cutlass.Int32(0)
        pend_writes = cutlass.Boolean(False)
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t
            writes = chunk_idx < wend
            gc = gbase + rev_idx
            decay_stage = gc % cfg.smem_decay_stages
            intermediate_stage = gc % cfg.smem_intermediate_stages
            raw_stage = raw_index.idx
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sDo_ptr = sDo_raw.data_ptr() + raw_stage * (cfg.d_v * cfg.b_t)
            sIntermediate_ptr = sIntermediate_raw.data_ptr() + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            # ---- A = tril_incl(Q_decay @ K_inv^T) --------------------------------
            bars.mb_a_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)
            bars.mb_q_decay_k_restore_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            a_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sQ_decay_ptr + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    a_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = a_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            nvvm.stmatrix(
                sIntermediate_ptr + stsm_idx,
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

            # ---- dA = tril_incl(dO @ U^T) ----------------------------------------
            bars.mb_u_smem_ready.wait(u_index.phase)
            u_index = advance(u_index, 1)
            da_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                da_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_v // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sDo_ptr + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sU_raw.data_ptr() + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    da_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            # fence: the dO/U ldmatrix reads must complete before this release
            # licenses the TMA reload (sDo)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_do_done[raw_stage].arrive()
            for accum_idx in cutlass.range_constexpr(8):
                da_acc[accum_idx] = da_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            bars.mb_da_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)
            nvvm.stmatrix(
                sIntermediate_ptr + 2 * (cfg.b_t * cfg.b_t) + stsm_idx,
                [
                    fp32_to_fp16(da_acc[0], da_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[2], da_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[4], da_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[6], da_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[intermediate_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

            # ---- dQ/dK/dGate/dBeta/dV/dW_out: previous chunk, one-behind store ladder ----
            if rev_idx > 0:
                bars.mb_dq_tmastg_ready[dq_index.idx].wait(dq_index.phase)
                if pend_writes:
                    desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
                    dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDq_tma[dq_index.idx], dq_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dk_tmastg_ready[dk_index.idx].wait(dk_index.phase)
                if pend_writes:
                    desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
                    dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDk_tma[dk_index.idx], dk_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dgate_tmastg_ready[dgate_index.idx].wait(dgate_index.phase)
                if pend_writes:
                    desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
                    dgate_slice = tma_slice_runtime_desc(desc_dgate_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDgate_tma[dgate_index.idx], dgate_slice, acquire=False)
                    tma_store_commit()
                bars.mb_db_tmastg_ready[db_index.idx].wait(db_index.phase)
                if pend_writes:
                    desc_db_slot = (desc_db_base + slot).tospace(cutlass.AddressSpace.generic)
                    db_slice = tma_slice_runtime_desc(desc_db_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDb_tma[db_index.idx], db_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dv_tmastg_ready[dv_index.idx].wait(dv_index.phase)
                if pend_writes:
                    desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
                    dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDv_tma[dv_index.idx], dv_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dwo_tmastg_ready[dwo_index.idx].wait(dwo_index.phase)
                if pend_writes:
                    desc_dwo_slot = (desc_dwo_base + slot).tospace(cutlass.AddressSpace.generic)
                    dwo_slice = tma_slice_runtime_desc(desc_dwo_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDwOut_tma[dwo_index.idx], dwo_slice, acquire=False)
                    tma_store_commit()
                tma_store_wait(5)
                bars.mb_dq_tmastg_done[dq_index.idx].arrive()
                tma_store_wait(4)
                bars.mb_dk_tmastg_done[dk_index.idx].arrive()
                tma_store_wait(3)
                bars.mb_dgate_tmastg_done[dgate_index.idx].arrive()
                tma_store_wait(2)
                bars.mb_db_tmastg_done[db_index.idx].arrive()
                tma_store_wait(1)
                bars.mb_dv_tmastg_done[dv_index.idx].arrive()
                tma_store_wait(0)
                bars.mb_dwo_tmastg_done[dwo_index.idx].arrive()
                dq_index = advance(dq_index, cfg.smem_dq_stages)
                dk_index = advance(dk_index, cfg.smem_dk_stages)
                dgate_index = advance(dgate_index, cfg.smem_dgate_stages)
                db_index = advance(db_index, cfg.smem_db_stages)
                dv_index = advance(dv_index, cfg.smem_dv_stages)
                dwo_index = advance(dwo_index, cfg.smem_dwo_stages)
            pend_start = chunk_start
            pend_writes = writes

        # ---- tile tail: drain the last chunk's dQ/dK/dGate/dBeta/dV/dW_out -------
        if sk_nt > 0:
            bars.mb_dq_tmastg_ready[dq_index.idx].wait(dq_index.phase)
            if pend_writes:
                desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
                dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDq_tma[dq_index.idx], dq_slice, acquire=False)
                tma_store_commit()
            bars.mb_dk_tmastg_ready[dk_index.idx].wait(dk_index.phase)
            if pend_writes:
                desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
                dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDk_tma[dk_index.idx], dk_slice, acquire=False)
                tma_store_commit()
            bars.mb_dgate_tmastg_ready[dgate_index.idx].wait(dgate_index.phase)
            if pend_writes:
                desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
                dgate_slice = tma_slice_runtime_desc(desc_dgate_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDgate_tma[dgate_index.idx], dgate_slice, acquire=False)
                tma_store_commit()
            bars.mb_db_tmastg_ready[db_index.idx].wait(db_index.phase)
            if pend_writes:
                desc_db_slot = (desc_db_base + slot).tospace(cutlass.AddressSpace.generic)
                db_slice = tma_slice_runtime_desc(desc_db_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDb_tma[db_index.idx], db_slice, acquire=False)
                tma_store_commit()
            bars.mb_dv_tmastg_ready[dv_index.idx].wait(dv_index.phase)
            if pend_writes:
                desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
                dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDv_tma[dv_index.idx], dv_slice, acquire=False)
                tma_store_commit()
            bars.mb_dwo_tmastg_ready[dwo_index.idx].wait(dwo_index.phase)
            if pend_writes:
                desc_dwo_slot = (desc_dwo_base + slot).tospace(cutlass.AddressSpace.generic)
                dwo_slice = tma_slice_runtime_desc(desc_dwo_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDwOut_tma[dwo_index.idx], dwo_slice, acquire=False)
                tma_store_commit()
            tma_store_wait(5)
            bars.mb_dq_tmastg_done[dq_index.idx].arrive()
            tma_store_wait(4)
            bars.mb_dk_tmastg_done[dk_index.idx].arrive()
            tma_store_wait(3)
            bars.mb_dgate_tmastg_done[dgate_index.idx].arrive()
            tma_store_wait(2)
            bars.mb_db_tmastg_done[db_index.idx].arrive()
            tma_store_wait(1)
            bars.mb_dv_tmastg_done[dv_index.idx].arrive()
            tma_store_wait(0)
            bars.mb_dwo_tmastg_done[dwo_index.idx].arrive()
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dgate_index = advance(dgate_index, cfg.smem_dgate_stages)
            db_index = advance(db_index, cfg.smem_db_stages)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            dwo_index = advance(dwo_index, cfg.smem_dwo_stages)

        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def super_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    sK_decay_raw,
    sK_inv_raw,
    sU_raw,
    sDy_raw,
    sIntermediate_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): builds the Neumann T_inv and
    strict-tril dM staging tiles, in chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    sdy_index = PipelineState.start(phase=0)

    # ---- ldmatrix lane decode ----------------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + stsm_col_coord, bbits=1, mbase=3, sshift=3)
    row_lo = lane // 4
    row_hi = row_lo + cutlass.Int32(8)

    # hoisted tril bitmasks: bit i = row > col / row == col for accum index i
    tril_strict_mask = cutlass.Int32(0)
    eye_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_strict_mask = tril_strict_mask | (cutlass.Int32(1 << accum_idx) if row_coord > col_coord else cutlass.Int32(0))
        eye_mask = eye_mask | (cutlass.Int32(1 << accum_idx) if row_coord == col_coord else cutlass.Int32(0))

    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        sk_nt = cend - wstart
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            gc = gbase + rev_idx
            decay_stage = gc % cfg.smem_decay_stages
            intermediate_stage = gc % cfg.smem_intermediate_stages
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = sIntermediate_raw.data_ptr() + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            bars.mb_t_inv_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)

            # ---- KK = K_decay @ K_inv^T ------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)
            kk_lhs_row = lhs_row_coord
            kk_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sK_decay_ptr + a_seg * (cfg.b_t * 64) + kk_lhs_row * 64 + swizzle_xor_128b(kk_lhs_row, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    kk_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- L = tril(KK, -1) ------------------------------------------------
            l_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                lower = kk_acc[accum_idx] if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                l_regs[accum_idx] = lower
            l_a0 = fp32_to_fp16(l_regs[0], l_regs[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_regs[4], l_regs[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_regs[6], l_regs[7], dtype=cfg.io_dtype)
            l_values = cutlass.Vector.from_elements((l_a0, l_a1, l_a2, l_a3), cutlass.Int32).bitcast(cfg.io_dtype).to(cutlass.Float32)

            tinv_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                eye = cutlass.Float32(1.0) if (eye_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                tinv_acc[accum_idx] = eye - l_values[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(l_a0), movmatrix_16b(l_a1), movmatrix_16b(l_a2), movmatrix_16b(l_a3)
            for _round in cutlass.range_constexpr(3):
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

            nvvm.stmatrix(
                sIntermediate_ptr + 1 * (cfg.b_t * cfg.b_t) + stsm_idx,
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

            # ---- dM = dY @ U^T ---------------------------------------------------
            bars.mb_dm_done[intermediate_stage].wait(((gc // cfg.smem_intermediate_stages) + 1) % 2)
            bars.mb_dy_smem_ready.wait(sdy_index.phase)
            sdy_index = advance(sdy_index, 1)
            dm_acc = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                dm_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_v // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sDy_raw.data_ptr() + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sU_raw.data_ptr() + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    dm_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            dm_strict_regs = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for accum_idx in cutlass.range_constexpr(8):
                dm_strict_regs[accum_idx] = dm_acc[accum_idx] if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            w0 = fp32_to_fp16(dm_strict_regs[0], dm_strict_regs[1], dtype=cfg.io_dtype)
            w1 = fp32_to_fp16(dm_strict_regs[2], dm_strict_regs[3], dtype=cfg.io_dtype)
            w2 = fp32_to_fp16(dm_strict_regs[4], dm_strict_regs[5], dtype=cfg.io_dtype)
            w3 = fp32_to_fp16(dm_strict_regs[6], dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(sIntermediate_ptr + 3 * (cfg.b_t * cfg.b_t) + stsm_idx, [w0, w1, w2, w3], nvvm.MMALayout.ROW, shape=nvvm.StoreShape.M8N8)
            nw0 = fp32_to_fp16(-dm_strict_regs[0], -dm_strict_regs[1], dtype=cfg.io_dtype)
            nw1 = fp32_to_fp16(-dm_strict_regs[2], -dm_strict_regs[3], dtype=cfg.io_dtype)
            nw2 = fp32_to_fp16(-dm_strict_regs[4], -dm_strict_regs[5], dtype=cfg.io_dtype)
            nw3 = fp32_to_fp16(-dm_strict_regs[6], -dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(sIntermediate_ptr + 4 * (cfg.b_t * cfg.b_t) + stsm_idx, [nw0, nw1, nw2, nw3], nvvm.MMALayout.ROW, shape=nvvm.StoreShape.M8N8)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_ready[intermediate_stage].arrive()
        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    tmem_hold,
    sState_alt,
    sK_decay_lead16,
    sK_inv_amaj,
    sK_restore_lead16,
    sDo_lead16,
    sDo_amaj,
    sQ_decay_trans,
    sK_decay_trans,
    sU_lead16,
    sDy_lead16,
    sDstate_alt,
    sIntermediate,
    sState_scale_diag,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): issues every tcgen05 GEMM and owns
    the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_hold, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_hold.load()
    bpe = cfg.io_dtype.width // 8

    # ---- chunk-invariant GEMM descriptors ----------------------------------------
    idesc_mv_nt = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    idesc_state_k_at = nvvm.Tcgen05InstrDesc.build(
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
        idesc=idesc_state_k_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_dvinter_desc = MmaDesc(
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
    bmm_du_at_desc = MmaDesc(
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
    bmm_dstate_q_at_desc = MmaDesc(
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
    bmm_qk_ts_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_mv_nt,
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
    bmm_qk_ts_t_desc = MmaDesc(
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
    idesc_diag = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=16,
        m_dim=cfg.d_v,
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
    idesc_dstate_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dstate_k_desc = MmaDesc(
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
    idesc_state_ts = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_state_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_state_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_ts_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_state_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dstate_at_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dstate_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_dgrad_ts_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_dgp,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dgrad_at_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dgp_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp_at_t = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_dgrad_at_t_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dgp_at_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    state_index = PipelineState.start(phase=0)
    y_inp_index = PipelineState.start(phase=0)
    dstate_inp_index = PipelineState.start(phase=0)
    du_inp_index = PipelineState.start(phase=0)
    neg_dy_index = PipelineState.start(phase=0)
    u_smem_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    parts_done_index = PipelineState.start(phase=1)

    do_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    op_seg = (cfg.b_t * cfg.d_k * (cfg.io_dtype.width // 8)) >> 4
    intermediate_seg = (cfg.intermediate_tiles * cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    intermediate_slot = (cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    diag_seg = ((cfg.d_k // 16) * 256 * (cfg.io_dtype.width // 8)) >> 4
    d_do_amaj0 = sDo_amaj[0].desc()
    d_qd_trans0 = sQ_decay_trans[0].desc()
    d_kd_trans0 = sK_decay_trans[0].desc()
    d_ki_amaj0 = sK_inv_amaj[0].desc()
    d_int0 = sIntermediate[0].desc()
    d_kd_lead0 = sK_decay_lead16[0].desc()
    d_do_lead0 = sDo_lead16[0].desc()
    d_kr_lead0 = sK_restore_lead16[0].desc()
    d_diag0 = sState_scale_diag[0].desc()
    d_dstate_alt0 = sDstate_alt[0].desc()
    d_u_lead0 = sU_lead16[0].desc()
    d_dy_lead0 = sDy_lead16[0].desc()
    assert cfg.smem_state_stages == 1
    d_state_alt0 = sState_alt[0].desc()
    dstate0_index = PipelineState.start(phase=0)

    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        sk_nt = cend - wstart
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

            # ---- stage-derived operand descriptors -------------------------------
            decay_op_off = decay_stage * op_seg
            d_do_amaj = d_do_amaj0 + raw_stage_idx * do_seg
            d_qd_trans = d_qd_trans0 + decay_op_off
            d_kd_trans = d_kd_trans0 + decay_op_off
            d_ki_amaj = d_ki_amaj0 + decay_op_off
            d_int = d_int0 + intermediate_stage * intermediate_seg
            d_int_tinv = d_int + intermediate_slot
            d_int_da = d_int + 2 * intermediate_slot
            d_int_dm = d_int + 3 * intermediate_slot
            d_int_ndm = d_int + 4 * intermediate_slot
            chunk_idx = cend - cutlass.Int32(1) - rev_idx

            # ---- state_k = state(S) @ K_decay^T ----------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(decay_phase)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                mma_ss(
                    bmm_state_k_desc,
                    d_state_alt0,
                    d_kd_lead0 + decay_op_off,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_k_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_state_k_acc_ready.arrive(cta_group=1)
                    bars.mb_state_done[state_index.idx].arrive(cta_group=1)
                state_index = advance(state_index, cfg.smem_state_stages)

            # ---- dQ inter = state(T) @ dO^T --------------------------------------
            bars.mb_dqk_acc_done.wait(parts_done_index.phase)
            parts_done_index = advance(parts_done_index, 1)
            bars.mb_state_inp_ready[gc % 2].wait((gc // 2) % 2)
            bars.mb_do_ready[raw_stage_idx].wait((gc // cfg.smem_raw_stages) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_inp_offset + (gc % 2) * (cfg.d_v // 2)), cutlass.Int8)
                b_desc = d_do_lead0 + raw_stage_idx * do_seg
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_state_ts_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_ts_desc.sps_B):
                        mma_ts_step(
                            bmm_state_ts_desc,
                            a_ptr.subview(sub * bmm_state_ts_desc.sps_B * bmm_state_ts_desc.tmem_advance_A),
                            b_desc + sub * (bmm_state_ts_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )

            # ---- dU inter = dstate_inp(T) @ K_restore ----------------------------
            bars.mb_q_decay_k_restore_ready[decay_stage].wait(decay_phase)
            if has_dstate:
                bars.mb_dstate_inp_ready.wait(dstate_inp_index.phase)
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_inp_offset), cutlass.Int8)
                b_desc = d_kr_lead0 + decay_op_off
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_dvinter_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dvinter_desc.sps_B):
                        mma_ts_step(
                            bmm_dvinter_desc,
                            a_ptr.subview(sub * bmm_dvinter_desc.sps_B * bmm_dvinter_desc.tmem_advance_A),
                            b_desc + sub * (bmm_dvinter_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )

            # ---- dstate decay = dstate_inp(T) @ diag(eGl) ------------------------
            if has_dstate:
                desc_diag = d_diag0 + decay_stage * diag_seg
                for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                    a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_inp_offset) + k_block * 8, cutlass.Int8)
                    b_desc = desc_diag.advance_start_address(k_block * 256 * 2)
                    c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset) + k_block * 16, cutlass.Float32)
                    for sub in cutlass.range_constexpr(bmm_diag_desc.num_subtiles_B):
                        for k in cutlass.range_constexpr(bmm_diag_desc.sps_B):
                            mma_ts_step(
                                bmm_diag_desc,
                                a_ptr.subview(sub * bmm_diag_desc.sps_B * bmm_diag_desc.tmem_advance_A),
                                b_desc + sub * (bmm_diag_desc.smem_subtile_B >> 4),
                                c_ptr,
                                k,
                                cutlass.Boolean(sub + k > 0),
                            )
                dstate_inp_index = advance(dstate_inp_index, 1)

            # ---- dU intra += dO^T(S) @ A -----------------------------------------
            bars.mb_a_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_du_at_desc,
                d_do_amaj,
                d_int,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )
            if elect_one:
                bars.mb_du_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate Q-term += dO^T(S) @ Q_decay ------------------------------
            mma_ss(
                bmm_dstate_q_at_desc,
                d_do_amaj,
                d_qd_trans,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )

            # ---- U = Y(T) @ T_inv ------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_phase)
            bars.mb_y_inp_ready.wait(y_inp_index.phase)
            y_inp_index = advance(y_inp_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_y_inp_offset), cutlass.Int8)
            b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_u_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_qk_ts_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_qk_ts_desc.sps_B):
                    mma_ts_step(
                        bmm_qk_ts_desc,
                        a_ptr.subview(sub * bmm_qk_ts_desc.sps_B * bmm_qk_ts_desc.tmem_advance_A),
                        b_desc + sub * (bmm_qk_ts_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_u_acc_ready.arrive(cta_group=1)
                bars.mb_do_mma_done[raw_stage_idx].arrive(cta_group=1)

            # ---- dY = dU(T) @ T_inv ----------------------------------------------
            bars.mb_du_inp_ready.wait(du_inp_index.phase)
            du_inp_index = advance(du_inp_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_inp_offset), cutlass.Int8)
            b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dy_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_qk_ts_t_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_qk_ts_t_desc.sps_B):
                    mma_ts_step(
                        bmm_qk_ts_t_desc,
                        a_ptr.subview(sub * bmm_qk_ts_t_desc.sps_B * bmm_qk_ts_t_desc.tmem_advance_A),
                        b_desc + sub * (bmm_qk_ts_t_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready.arrive(cta_group=1)
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)

            # ---- dK_restore part = dstate(S) @ U^T -------------------------------
            bars.mb_u_smem_ready.wait(u_smem_index.phase)
            u_smem_index = advance(u_smem_index, 1)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)
                mma_ss(
                    bmm_dstate_at_desc,
                    d_dstate_alt0,
                    d_u_lead0,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_restore_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_restore_part_acc_ready.arrive(cta_group=1)
                    bars.mb_dstate_smem_done.arrive(cta_group=1)

            # ---- dstate K-term += -dY(T) @ K_decay -------------------------------
            bars.mb_neg_dy_inp_ready.wait(neg_dy_index.phase)
            neg_dy_index = advance(neg_dy_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_neg_dy_inp_offset), cutlass.Int8)
            b_desc = d_kd_trans
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_dstate_k_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dstate_k_desc.sps_B):
                    mma_ts_step(
                        bmm_dstate_k_desc,
                        a_ptr.subview(sub * bmm_dstate_k_desc.sps_B * bmm_dstate_k_desc.tmem_advance_A),
                        b_desc + sub * (bmm_dstate_k_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready.arrive(cta_group=1)

            # ---- dK_inv part = scale.Q_decay^T(S) @ dA ---------------------------
            bars.mb_da_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_dgrad_at_t_desc,
                d_qd_trans,
                d_int_da,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=False,
            )

            # ---- dQ attn += K_inv^T(S) @ dA^T ------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dq_acc_ready.arrive(cta_group=1)
                bars.mb_da_done[intermediate_stage].arrive(cta_group=1)

            # ---- dK_decay part = state(T) @ dY^T ---------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dy_smem_ready.wait(gc % 2)
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_inp_offset + (gc % 2) * (cfg.d_v // 2)), cutlass.Int8)
                b_desc = d_dy_lead0
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_state_ts_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_ts_desc.sps_B):
                        mma_ts_step(
                            bmm_state_ts_desc,
                            a_ptr.subview(sub * bmm_state_ts_desc.sps_B * bmm_state_ts_desc.tmem_advance_A),
                            b_desc + sub * (bmm_state_ts_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )
            if elect_one:
                bars.mb_dy_smem_done.arrive(cta_group=1)
                bars.mb_state_inp_done[gc % 2].arrive(cta_group=1)

            # ---- dK_inv part += K_decay^T(S) @ -dM_strict ------------------------
            bars.mb_dm_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_dgrad_at_t_desc,
                d_kd_trans,
                d_int_ndm,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=True,
            )
            if elect_one:
                bars.mb_dk_inv_part_acc_ready.arrive(cta_group=1)

            # ---- dK_decay part += K_inv^T(S) @ dM_strict^T -----------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_dm,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_dm,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dk_decay_part_acc_ready.arrive(cta_group=1)
                bars.mb_dm_done[intermediate_stage].arrive(cta_group=1)
                bars.mb_decay_done[decay_stage].arrive(cta_group=1)

        # ---- tile end: WG1's dstate0 drain gates the next tile's dstate reuse ----
        bars.mb_dstate0_acc_stored.wait(dstate0_index.phase)
        dstate0_index = advance(dstate0_index, 1)
        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
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
    mSched,
    sSched,
    q_tx_bytes,
    k_tx_bytes,
    gate_tx_bytes,
    beta_tx_bytes,
    do_tx_bytes,
    v_tx_bytes,
    w_tx_bytes,
    sQ_raw,
    sK_raw,
    sV_raw,
    sGate_raw,
    sDo_raw,
    sBeta_raw,
    sW_raw,
    sState_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    desc_do_base,
    desc_beta_base,
    desc_w_base,
    desc_checkpoint_base,
    desc_initial_state_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing every
    G->S operand load."""
    elect_one = nvvm.elect_sync()

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
        tma_loads_per_tile=(cfg.d_v // 64),
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
    sW_tma = SmemTile(
        base=sW_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sState_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=(cfg.d_k * cfg.d_v),
        stages=cfg.smem_state_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.d_k * 64,
    )
    raw_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        next_tile, sched_state = sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)
        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_beta_slot = (desc_beta_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_w_slot = (desc_w_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_initial_state_slot = (desc_initial_state_base + cutlass.Int32(0)).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_beta_slot)
            tma_tensormap_acquire(desc_w_slot)
            tma_tensormap_acquire(desc_checkpoint_slot)
            if cutlass.const_expr(cfg.use_initial_state):
                tma_tensormap_acquire(desc_initial_state_slot)
        sk_nt = cend - wstart
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t

            # ---- Q load ----------------------------------------------------------
            bars.mb_q_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_q_ready[raw_index.idx].arrive(n_bytes=q_tx_bytes)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, chunk_start)
            tma_load_tile(sQ_tma[raw_index.idx], q_slice, bars.mb_q_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- K load ----------------------------------------------------------
            bars.mb_k_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_k_ready[raw_index.idx].arrive(n_bytes=k_tx_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, bars.mb_k_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Gate load -------------------------------------------------------
            bars.mb_gate_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_gate_ready[raw_index.idx].arrive(n_bytes=gate_tx_bytes)
            gate_slice = tma_slice_runtime_desc(desc_gate_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, bars.mb_gate_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Beta load -------------------------------------------------------
            bars.mb_beta_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_beta_ready[raw_index.idx].arrive(n_bytes=beta_tx_bytes)
            beta_slice = tma_slice_runtime_desc(desc_beta_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sBeta_tma[raw_index.idx], beta_slice, bars.mb_beta_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- entering state: checkpoint[c - 1], or initial_state for chunk 0 when given --
            if chunk_idx >= FIRST_STATE_CHUNK:
                state_idx = state_index.idx
                bars.mb_state_cg0_done[state_idx].wait(state_index.phase)
                bars.mb_state_done[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                if elect_one:
                    bars.mb_state_ready[state_idx].arrive(n_bytes=cfg.tma_state_bytes)
                if cutlass.const_expr(cfg.use_initial_state):
                    if chunk_idx == 0:
                        initial_state_slice = tma_slice_runtime_desc(desc_initial_state_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, batch_idx)
                        tma_load_tile(sState_tma[state_idx], initial_state_slice, bars.mb_state_ready[state_idx].smem_ptr, acquire=False)
                    else:
                        state_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx - cutlass.Int32(1), head_o)
                        tma_load_tile(sState_tma[state_idx], state_slice, bars.mb_state_ready[state_idx].smem_ptr, acquire=False)
                else:
                    state_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx - FIRST_STATE_CHUNK, head_o)
                    tma_load_tile(sState_tma[state_idx], state_slice, bars.mb_state_ready[state_idx].smem_ptr, acquire=False)

            # ---- dO load ---------------------------------------------------------
            bars.mb_do_done[raw_index.idx].wait(raw_index.phase)
            bars.mb_do_mma_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_do_ready[raw_index.idx].arrive(n_bytes=do_tx_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sDo_tma[raw_index.idx], do_slice, bars.mb_do_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- V load ----------------------------------------------------------
            bars.mb_v_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_v_ready[raw_index.idx].arrive(n_bytes=v_tx_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, bars.mb_v_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- W load ----------------------------------------------------------
            bars.mb_w_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_w_ready[raw_index.idx].arrive(n_bytes=w_tx_bytes)
            w_slice = tma_slice_runtime_desc(desc_w_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(sW_tma[raw_index.idx], w_slice, bars.mb_w_ready[raw_index.idx].smem_ptr, acquire=False)
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx = next_tile


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment."""

    if cutlass.const_expr(cfg.safe_gate):
        half = cutlass.Float32(0.5)
        sigmoid = cute.math.tanh(raw_gate * half, approx=True) * half + half
        return cfg.gate_scale_log2 * sigmoid
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
    sSched,
    lane,
    tmem_hold,
    warp_idx,
    scale,
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_raw,
    sK_raw,
    sQ_raw,
    sState_raw,
    sV_raw,
    sDo_raw,
    sBeta_raw,
    sW_raw,
    sNorm_raw,
    sK_decay_raw,
    sK_restore_raw,
    sQ_decay_raw,
    sState_scale_diag_raw,
    bars,
) -> None:
    """WG0 warp role (warps 0-3): persistent tile-scheduler loop + gate prefix
    scan and the decay/restore operand materialization into tcgen05 SMEM for
    EVERY chunk (no ping-pong: the backward pipeline is drain-bound).  Also
    stashes the per-row Q/K inverse norms for WG2's dGate assembly and copies
    H -> TMEM f16 at the chunk tail."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_hold.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_sp = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    value_dim = tmem_sp * cfg.threads_per_warp + lane
    state_copy_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
    state_index = PipelineState.start(phase=0)
    cg0_prefix_dim = cg0_warp * cfg.threads_per_warp + lane
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        sk_nt = cend - wstart
        if cutlass.const_expr(cfg.safe_gate):
            if sk_nt > 0:
                cg0_a_log_exp = cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * LOG2_E, fastmath=True)
                cg0_dt_bias_value = mDt_bias[head_idx, cg0_prefix_dim].to(cutlass.Float32)
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            gc = gbase + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = gc % cfg.smem_decay_stages
            raw_stage = gc % cfg.smem_raw_stages
            sQ_ptr = sQ_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = sK_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sBetaP_ptr = sBeta_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sWP_ptr = sW_raw.data_ptr() + raw_stage * (cfg.d_v * cfg.b_t)
            sGate_ptr = sGate_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = sK_inv_raw.data_ptr() + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = sK_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = sQ_decay_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = sK_restore_raw.data_ptr() + decay_stage * (cfg.d_k * cfg.b_t)
            sState_scale_diag_ptr = sState_scale_diag_raw.data_ptr() + decay_stage * ((cfg.d_k // 16) * 256)

            bars.mb_gate_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_q_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_k_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_beta_ready[raw_stage].wait((gc // cfg.smem_raw_stages) % 2)

            row_group_start = cg0_warp * (cfg.b_t // len(cfg.compute_group_0_warp_ids))
            lane_row_group = lane // 8
            lane_in_row_group = lane - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            g_prefix_ptr = sGate_ptr
            prefix_dim = cg0_warp * cfg.threads_per_warp + lane
            # ---- gate prefix scan: cumulative log-gate per key channel -----------
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
                    gate0 = gate_scale(
                        cfg,
                        gate0,
                    )
                    gate1 = gate_scale(
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
                pair_vec = nvvm.add_packed_f32x2(
                    cutlass.Vector.from_elements((prefix_acc, gate0), cutlass.Float32),
                    cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32),
                    ftz=False,
                    rnd="rn",
                )
                prefix0, row_pair_sum = cutlass.Float32(pair_vec[0]), cutlass.Float32(pair_vec[1])
                prefix1 = prefix_acc + row_pair_sum
                g_prefix_regs[row0] = prefix0
                g_prefix_regs[row1] = prefix1
                prefix_acc = prefix1

            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            exp_g_last = g_prefix_regs[cfg.b_t - 1]
            # ---- decay-slot guard: previous use fully consumed -------------------
            operand_done_phase = ((gc // cfg.smem_decay_stages) + 1) % 2
            bars.mb_decay_done[decay_stage].wait(operand_done_phase)

            for row in cutlass.range_constexpr(cfg.b_t):
                f32_segment = prefix_dim // 32
                f32_segment_dim = prefix_dim - f32_segment * 32
                prefix_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                (sGate_ptr + prefix_idx).store(g_prefix_regs[row])

            # ---- state-scale diag: stage exp2(g_last) decay blocks ---------------
            block = prefix_dim // cutlass.Int32(16)
            coord = prefix_dim - block * cutlass.Int32(16)
            linear_idx = block * cutlass.Int32(256) + coord * cutlass.Int32(16) + coord
            diag_idx = swizzle_lin_S(linear_idx, bbits=1, mbase=3, sshift=3)
            sState_scale_diag_ptr[diag_idx] = exp_g_last.to(cfg.io_dtype)

            # ---- raw Q/K: SMEM -> TMEM ring (channel-major, for WG2) -------------
            qk_raw_stage = gc % cfg.tmem_qk_raw_stages
            bars.mb_qk_raw_done[qk_raw_stage].wait(((gc // cfg.tmem_qk_raw_stages) + 1) % 2)
            raw_seg = prefix_dim // 64
            raw_dim = prefix_dim - raw_seg * 64
            q_raw_words = cutlass.Array(cutlass.Int32, cfg.b_t // 2, alignment=16)
            k_raw_words = cutlass.Array(cutlass.Int32, cfg.b_t // 2, alignment=16)
            for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                t0 = 2 * t2
                ridx0 = raw_seg * (cfg.b_t * 64) + t0 * 64 + swizzle_xor_128b(t0, raw_dim, elem_bytes=2)
                ridx1 = raw_seg * (cfg.b_t * 64) + (t0 + 1) * 64 + swizzle_xor_128b(t0 + 1, raw_dim, elem_bytes=2)
                q0 = (sQ_ptr + ridx0).load().to(cutlass.Float32)
                q1 = (sQ_ptr + ridx1).load().to(cutlass.Float32)
                k0 = (sK_ptr + ridx0).load().to(cutlass.Float32)
                k1 = (sK_ptr + ridx1).load().to(cutlass.Float32)
                q_raw_words[t2] = fp32_to_fp16(q0, q1, dtype=cfg.io_dtype)
                k_raw_words[t2] = fp32_to_fp16(k0, k1, dtype=cfg.io_dtype)
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(state_copy_addr + (tmem_col + cfg.tmem_qraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)), cutlass.Int8),
                q_raw_words[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(state_copy_addr + (tmem_col + cfg.tmem_kraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)), cutlass.Int8),
                k_raw_words[0 : (cfg.b_t // 2)],
            )
            nvvm.tcgen05_wait("store")
            bars.mb_qk_raw_ready[qk_raw_stage].arrive()

            nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)

            k_inv_pack = cutlass.Array(cutlass.Int32, 2 * 4, alignment=16)
            raw_q_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            raw_k_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            raw_beta_regs = cutlass.Array(cutlass.Float32, 2 * 8, alignment=16)
            # ---- optional Q/K L2-norm --------------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                qk0_lo = opaque_f32_zero()
                qk0_hi = opaque_f32_zero()
                qk1_lo = opaque_f32_zero()
                qk1_hi = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
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
                    raw_q_regs[reg_base + dim_offset] = q_val
                    raw_k_regs[reg_base + dim_offset] = k_val
                    beta_val = raw_beta_frag_f32[dim_offset]
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        half = cutlass.Float32(0.5)
                        beta_val = (cute.math.tanh(beta_val * half, approx=True) * half + half).to(cfg.io_dtype).to(cutlass.Float32)
                    raw_beta_regs[reg_base + dim_offset] = beta_val
                    if cutlass.const_expr(cfg.l2norm):
                        if cutlass.const_expr(dim_offset % 2 == 0):
                            qk0_lo, qk0_hi = ffma2(q_val, k_val, q_val, k_val, qk0_lo, qk0_hi)
                        else:
                            qk1_lo, qk1_hi = ffma2(q_val, k_val, q_val, k_val, qk1_lo, qk1_hi)

            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_done[raw_stage].arrive()
            bars.mb_k_done[raw_stage].arrive()
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
                if lane_in_row_group == 0:
                    sNorm_raw[(gc % cfg.tmem_qk_raw_stages) * (2 * cfg.b_t) + decay_row] = q_inv_norm
                    sNorm_raw[(gc % cfg.tmem_qk_raw_stages) * (2 * cfg.b_t) + cfg.b_t + decay_row] = k_inv_norm
            q_stage_norm = q_inv_norm * scale

            # ---- decay/restore operands: exp2(+-g) applied per key channel -------
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
                    exp_g_frag = (g_prefix_ptr + g_prefix_idx).load(count=4, alignment=16)
                    exp_g_last_idx = f32_segment * (cfg.b_t * 32) + (cfg.b_t - 1) * 32 + swizzle_xor_128b((cfg.b_t - 1), f32_segment_dim, elem_bytes=4)
                    exp_g_last_frag = (g_prefix_ptr + exp_g_last_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    exp_g_regs[f32_reg_base] = exp_g_frag[0]
                    exp_g_regs[f32_reg_base + 1] = exp_g_frag[1]
                    exp_g_regs[f32_reg_base + 2] = exp_g_frag[2]
                    exp_g_regs[f32_reg_base + 3] = exp_g_frag[3]
                    exp_g_last_regs[f32_reg_base] = exp_g_last_frag[0]
                    exp_g_last_regs[f32_reg_base + 1] = exp_g_last_frag[1]
                    exp_g_last_regs[f32_reg_base + 2] = exp_g_last_frag[2]
                    exp_g_last_regs[f32_reg_base + 3] = exp_g_last_frag[3]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_gate_done[raw_stage].arrive()

            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                # ---- K decay + K inv operands: exp2(g) * Beta * K / exp2(-g) * K -
                k_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
                    k_beta0, k_beta1 = fmul2(k_value0, k_value1, raw_beta_regs[raw_reg_idx0], raw_beta_regs[raw_reg_idx1])
                    k_pair = fp32_to_fp16(k_beta0, k_beta1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_decay_pack[pair_idx] = mul_f16x2(k_pair, exp_g_pair, cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    exp_neg_pair = fp32_to_fp16(exp_neg_g0, exp_neg_g1, dtype=cfg.io_dtype)
                    k_norm_pair = fp32_to_fp16(k_value0, k_value1, dtype=cfg.io_dtype)
                    k_inv_pack[dim_half * 4 + pair_idx] = mul_f16x2(k_norm_pair, exp_neg_pair, cfg.io_dtype)

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

            # ---- Q decay + K_restore operands ------------------------------------
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                q_decay_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                k_restore_pack = cutlass.Array(cutlass.Int32, 4, alignment=16)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_stage_norm, q_stage_norm)
                    q_pair = fp32_to_fp16(q_value0, q_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    q_decay_pack[pair_idx] = mul_f16x2(q_pair, exp_g_pair, cfg.io_dtype)
                    exp_g_last_pair = fp32_to_fp16(exp_g_last_regs[raw_reg_idx0], exp_g_last_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_restore_pack[pair_idx] = mul_f16x2(k_inv_pack[dim_half * 4 + pair_idx], exp_g_last_pair, cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (q_decay_pack[0], q_decay_pack[1], q_decay_pack[2], q_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                k_restore_vec = cutlass.Vector.from_elements(
                    (k_restore_pack[0], k_restore_pack[1], k_restore_pack[2], k_restore_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sQ_decay_ptr + op_idx).store(q_decay_vec, alignment=16)
                (sK_restore_ptr + op_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_decay_k_restore_ready[decay_stage].arrive()

            # ---- state copy: SMEM -> TMEM f16 ------------------------------------
            bars.mb_state_inp_done[gc % 2].wait(((gc // 2) + 1) % 2)
            bars.mb_state_inp_cg2_done[gc % 2].wait(((gc // 2) + 1) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                state_src = sState_raw.data_ptr() + state_index.idx * (cfg.d_k * cfg.d_v)
                for pl in cutlass.range_constexpr(2):
                    for g8 in cutlass.range_constexpr(8):
                        state_frag = (state_src + pl * (cfg.d_k * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, g8 * 8, elem_bytes=2)).load(
                            count=8, alignment=16
                        )
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                state_copy_addr + (tmem_col + cfg.tmem_state_inp_offset + (gc % 2) * (cfg.d_v // 2) + pl * 32 + g8 * 4), cutlass.Int8
                            ),
                            state_frag.bitcast(cutlass.Int32),
                        )
                nvvm.tcgen05_wait("store")
                bars.mb_state_cg0_done[state_index.idx].arrive()
                state_index = advance(state_index, cfg.smem_state_stages)
            bars.mb_state_inp_ready[gc % 2].arrive()
        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def compute1_warp_group(
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
    mDstate0,
    mDstate_in,
    sV_raw,
    sW_raw,
    sDo_raw,
    sU_raw,
    sDy_raw,
    sDv_raw,
    sDwOut_raw,
    sDstate_raw,
    bars,
) -> None:
    """WG1 warp role (warps 4-7): the value-side TMEM staging."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_hold.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_sp = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    ov_tok = (lane // 16) * 8 + (lane & 7)
    ov_col = ((lane // 8) & 1) * 8
    value_dim = tmem_sp * cfg.threads_per_warp + lane
    value_dim_base = tmem_sp * cfg.threads_per_warp
    cg1_tidx = warp_idx % 4 * cfg.threads_per_warp + lane

    raw_index = PipelineState.start(phase=0)
    state_k_index = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    du_acc_index = PipelineState.start(phase=0)
    dy_acc_index = PipelineState.start(phase=0)
    sdy_done_index = PipelineState.start(phase=1)
    dstate_ready_index = PipelineState.start(phase=0)
    dstate_smem_done_index = PipelineState.start(phase=1)
    dv_done_index = PipelineState.start(phase=1)
    dwo_done_index = PipelineState.start(phase=1)

    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        sk_nt = cend - wstart

        # ---- dht seeding: dH acc + dh_inp f16 + sdH ------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if sk_nt > 0:
                seed_true = cend == num_chunks_b
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
                row_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    seed_block = cutlass.Array(cutlass.Float32, 16, alignment=16)
                    for kk_i in cutlass.range_constexpr(16):
                        dval = mDstate_in[batch_idx, head_idx, sub * 16 + kk_i, value_dim].to(cutlass.Float32)
                        dval = dval if seed_true else cutlass.Float32(0.0)
                        seed_block[kk_i] = dval
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 16), cutlass.Float32),
                        seed_block[0:16],
                    )
                    seed_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                    for pc in cutlass.range_constexpr(8):
                        seed_pack[pc] = fp32_to_fp16(seed_block[2 * pc], seed_block[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 8), cutlass.Int8),
                        seed_pack[0:8],
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready.arrive()

                # ---- dht seed -> sdH: re-read dh_inp after the TMEM publish ------
                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 8), cutlass.Float32), num=8
                    )
                    for half in cutlass.range_constexpr(2):
                        d_base = sub * 16 + half * 8
                        h_pack = cutlass.Vector.from_elements(
                            (dstate_words[half * 4], dstate_words[half * 4 + 1], dstate_words[half * 4 + 2], dstate_words[half * 4 + 3]),
                            cutlass.Float32,
                        ).bitcast(cfg.io_dtype)
                        h_addr = (d_base // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                        (sDstate_raw.data_ptr() + h_addr).store(h_pack, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()

        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            gc = gbase + rev_idx
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)

            sV_ptr = sV_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            sW_ptr = sW_raw.data_ptr() + raw_index.idx * (cfg.d_v * cfg.b_t)
            row_addr_lo = tmem_row << 16
            row_addr_hi = (tmem_row + 16) << 16
            row_id0 = tmem_row + value_dim_base
            row_id1 = row_id0 + 16

            # ---- Y staging: Y = W*V - state_k -> TMEM f16 ------------------------
            bars.mb_v_ready[raw_index.idx].wait((gc // cfg.smem_raw_stages) % 2)
            bars.mb_w_ready[raw_index.idx].wait((gc // cfg.smem_raw_stages) % 2)
            projection_col_id = tmem_col + cfg.tmem_state_k_acc_offset
            input_col_id = tmem_col + cfg.tmem_y_inp_offset
            raw_v_frag0 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag1 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_w_frag0 = nvvm.ldmatrix(
                sW_ptr
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_w_frag1 = nvvm.ldmatrix(
                sW_ptr
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )

            y_inp_pack0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            y_inp_pack1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_k_acc_ready.wait(state_k_index.phase)
                state_k_index = advance(state_k_index, 1)
                state_k_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)
                state_k_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    frag_pair = (reg_idx ^ 2) * 2
                    state_k_pair = fp32_to_fp16(state_k_vec0[frag_pair], state_k_vec0[frag_pair + 1], dtype=cfg.io_dtype)
                    wv_pair = mul_f16x2(raw_w_frag0[raw_matrix], raw_v_frag0[raw_matrix], cfg.io_dtype)
                    y_inp_pack0[reg_idx ^ 2] = sub_f16x2(wv_pair, state_k_pair, cfg.io_dtype)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    frag_pair = (reg_idx ^ 2) * 2
                    state_k_pair = fp32_to_fp16(state_k_vec1[frag_pair], state_k_vec1[frag_pair + 1], dtype=cfg.io_dtype)
                    wv_pair = mul_f16x2(raw_w_frag1[raw_matrix], raw_v_frag1[raw_matrix], cfg.io_dtype)
                    y_inp_pack1[reg_idx ^ 2] = sub_f16x2(wv_pair, state_k_pair, cfg.io_dtype)
            if chunk_idx < FIRST_STATE_CHUNK:
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    y_inp_pack0[reg_idx ^ 2] = mul_f16x2(raw_w_frag0[raw_matrix], raw_v_frag0[raw_matrix], cfg.io_dtype)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    y_inp_pack1[reg_idx ^ 2] = mul_f16x2(raw_w_frag1[raw_matrix], raw_v_frag1[raw_matrix], cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + input_col_id, cutlass.Int8), y_inp_pack0[0:4])
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + input_col_id, cutlass.Int8), y_inp_pack1[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_y_inp_ready.arrive()

            # ---- dU restage: dU acc -> TMEM f16 A operand ------------------------
            bars.mb_du_acc_ready.wait(du_acc_index.phase)
            du_acc_index = advance(du_acc_index, 1)
            du_col_id = tmem_col + cfg.tmem_du_acc_offset
            du_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + du_col_id, cutlass.Float32), num=2)
            du_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + du_col_id, cutlass.Float32), num=2)

            du_pack0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            du_pack1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                du_pack0[reg_idx] = fp32_to_fp16(du_vec0[frag_pair], du_vec0[frag_pair + 1], dtype=cfg.io_dtype)
                du_pack1[reg_idx] = fp32_to_fp16(du_vec1[frag_pair], du_vec1[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + (tmem_col + cfg.tmem_du_inp_offset), cutlass.Int8), du_pack0[0:4])
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + (tmem_col + cfg.tmem_du_inp_offset), cutlass.Int8), du_pack1[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_du_inp_ready.arrive()

            # ---- U readback -> sU ------------------------------------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_index = advance(u_acc_index, 1)
            u_col_id = tmem_col + cfg.tmem_u_acc_offset
            u_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + u_col_id, cutlass.Float32), num=2)
            u_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + u_col_id, cutlass.Float32), num=2)

            u_pack0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            u_pack1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                u_pack0[reg_idx] = fp32_to_fp16(u_vec0[2 * reg_idx], u_vec0[2 * reg_idx + 1], dtype=cfg.io_dtype)
                u_pack1[reg_idx] = fp32_to_fp16(u_vec1[2 * reg_idx], u_vec1[2 * reg_idx + 1], dtype=cfg.io_dtype)
            nvvm.stmatrix(
                sU_raw.data_ptr()
                + (value_dim_base + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2),
                u_pack0.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                sU_raw.data_ptr()
                + (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2),
                u_pack1.data_ptr().load(count=4, alignment=4),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_smem_ready.arrive()

            # ---- dY readback -------------------------------------------------------
            bars.mb_dy_acc_ready.wait(dy_acc_index.phase)
            dy_acc_index = advance(dy_acc_index, 1)
            dy_col_id = tmem_col + cfg.tmem_dy_acc_offset
            dy_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + dy_col_id, cutlass.Float32), num=2)
            dy_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + dy_col_id, cutlass.Float32), num=2)

            # ---- -dY -> TMEM: A operand of the dstate K-term -----------------------
            neg_dy_regs0 = cutlass.Array(cutlass.Float32, 8, alignment=16)
            neg_dy_regs1 = cutlass.Array(cutlass.Float32, 8, alignment=16)
            for e in cutlass.range_constexpr(8):
                neg_dy_regs0[e] = -dy_vec0[e]
                neg_dy_regs1[e] = -dy_vec1[e]
            neg_dy_pack0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            neg_dy_pack1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                neg_dy_pack0[reg_idx] = fp32_to_fp16(neg_dy_regs0[frag_pair], neg_dy_regs0[frag_pair + 1], dtype=cfg.io_dtype)
                neg_dy_pack1[reg_idx] = fp32_to_fp16(neg_dy_regs1[frag_pair], neg_dy_regs1[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + (tmem_col + cfg.tmem_neg_dy_inp_offset), cutlass.Int8), neg_dy_pack0[0:4])
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + (tmem_col + cfg.tmem_neg_dy_inp_offset), cutlass.Int8), neg_dy_pack1[0:4])
            nvvm.tcgen05_wait("store")
            bars.mb_neg_dy_inp_ready.arrive()

            # ---- dY -> sdY: pack + store + publish (super dM + dV scalar operand) --
            addr_lo0 = (value_dim_base + ov_col) // 64 * (cfg.b_t * 64) + ov_tok * 64 + swizzle_xor_128b(ov_tok, (value_dim_base + ov_col) % 64, elem_bytes=2)
            addr_lo1 = (
                (value_dim_base + 16 + ov_col) // 64 * (cfg.b_t * 64)
                + ov_tok * 64
                + swizzle_xor_128b(ov_tok, (value_dim_base + 16 + ov_col) % 64, elem_bytes=2)
            )
            dy_pack0 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            dy_pack1 = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                dy_pack0[reg_idx] = fp32_to_fp16(dy_vec0[2 * reg_idx], dy_vec0[2 * reg_idx + 1], dtype=cfg.io_dtype)
                dy_pack1[reg_idx] = fp32_to_fp16(dy_vec1[2 * reg_idx], dy_vec1[2 * reg_idx + 1], dtype=cfg.io_dtype)
            bars.mb_dy_smem_done.wait(sdy_done_index.phase)
            sdy_done_index = advance(sdy_done_index, 1)
            nvvm.stmatrix(sDy_raw.data_ptr() + addr_lo0, dy_pack0.data_ptr().load(count=4, alignment=4), nvvm.MMALayout.COL, shape=nvvm.StoreShape.M8N8)
            nvvm.stmatrix(sDy_raw.data_ptr() + addr_lo1, dy_pack1.data_ptr().load(count=4, alignment=4), nvvm.MMALayout.COL, shape=nvvm.StoreShape.M8N8)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dy_smem_ready.arrive()

            # ---- scalar pass over own sdY: dV staging ----------------------------
            dv_stage = gc % cfg.smem_dv_stages
            bars.mb_dv_tmastg_done[dv_stage].wait(dv_done_index.phase)
            dv_done_index = advance(dv_done_index, cfg.smem_dv_stages)
            dwo_stage = gc % cfg.smem_dwo_stages
            bars.mb_dwo_tmastg_done[dwo_stage].wait(dwo_done_index.phase)
            dwo_done_index = advance(dwo_done_index, cfg.smem_dwo_stages)
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)
            sdv_stage_base = dv_stage * (cfg.b_t * cfg.d_v)
            sdwo_stage_base = dwo_stage * (cfg.b_t * cfg.d_v)
            c_seg = value_dim // 64
            c_dim = value_dim - c_seg * 64
            for t in cutlass.range_constexpr(cfg.b_t):
                idx = c_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, c_dim, elem_bytes=2)
                dy_v = (sDy_raw.data_ptr() + idx).load().to(cutlass.Float32)
                w_v = (sW_ptr + idx).load().to(cutlass.Float32)
                v_v = (sV_ptr + idx).load().to(cutlass.Float32)
                (sDv_raw.data_ptr() + sdv_stage_base + idx).store((w_v * dy_v).to(cfg.io_dtype))
                (sDwOut_raw.data_ptr() + sdwo_stage_base + idx).store((v_v * dy_v).to(cfg.io_dtype))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stage].arrive()
            bars.mb_dwo_tmastg_ready[dwo_stage].arrive()
            bars.mb_v_done[raw_index.idx].arrive()
            bars.mb_w_done[raw_index.idx].arrive()

            # ---- dH capture for the next -----------------------------------------
            bars.mb_dstate_acc_ready.wait(dstate_ready_index.phase)
            dstate_ready_index = advance(dstate_ready_index, 1)
            if rev_idx + cutlass.Int32(1) < sk_nt:
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
                row_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
                for sub in cutlass.range_constexpr(cfg.d_k // 32):
                    dstate_vec = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 32), cutlass.Float32), num=32
                    )
                    dstate_pack = cutlass.Array(cutlass.Int32, 16, alignment=16)
                    for pc in cutlass.range_constexpr(16):
                        dstate_pack[pc] = fp32_to_fp16(dstate_vec[2 * pc], dstate_vec[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 16), cutlass.Int8),
                        dstate_pack[0:16],
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready.arrive()

                # ---- dh_inp -> sdH: re-read after the TMEM publish ---------------
                for sub in cutlass.range_constexpr(cfg.d_k // 32):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 16), cutlass.Float32), num=16
                    )
                    for half in cutlass.range_constexpr(4):
                        d_base = sub * 32 + half * 8
                        h_pack = cutlass.Vector.from_elements(
                            (dstate_words[half * 4], dstate_words[half * 4 + 1], dstate_words[half * 4 + 2], dstate_words[half * 4 + 3]),
                            cutlass.Float32,
                        ).bitcast(cfg.io_dtype)
                        h_addr = (d_base // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                        (sDstate_raw.data_ptr() + h_addr).store(h_pack, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        # ---- tile end: dS0 drain / zero-length pass-through ----------------------
        if cutlass.const_expr(mDstate0 is not None):
            if sk_nt > 0:
                if wstart == 0:
                    row_addr = (tmem_row + tmem_sp * cfg.threads_per_warp) << 16
                    for sub in cutlass.range_constexpr(cfg.d_k // 32):
                        dstate0_vec = nvvm.tcgen05_ld(
                            "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 32), cutlass.Float32), num=32
                        )
                        for kk_i in cutlass.range_constexpr(32):
                            mDstate0[batch_idx, head_idx, sub * 32 + kk_i, value_dim] = dstate0_vec[kk_i]
            else:
                for key_dim_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for kk_i in cutlass.range_constexpr(32):
                        kd = key_dim_base + kk_i
                        if cutlass.const_expr(cfg.use_dstate_in):
                            mDstate0[batch_idx, head_idx, kd, value_dim] = mDstate_in[batch_idx, head_idx, kd, value_dim]
                        else:
                            mDstate0[batch_idx, head_idx, kd, value_dim] = cutlass.Float32(0.0)
        bars.mb_dstate0_acc_stored.arrive()
        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def compute2_warp_group(
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
    sBeta_raw,
    sGate_raw,
    sNorm_raw,
    sDq_raw,
    sDk_raw,
    sRed1_raw,
    sDstate_raw,
    sDgate_raw,
    sDb_raw,
    scale,
    bars,
) -> None:
    """WG2 warp role (warps 8-11): the gradient drain.  Each thread owns one
    key channel d for all 16 tokens: reads the dQ accumulator and the four dK
    parts from TMEM, assembles dQ/dK with the per-channel gate factors (raw
    Q/K arrive through WG0's TMEM ring), applies the in-kernel L2-norm
    backward row projection, assembles the per-channel dGate including the
    g_last terms, reverse-cumsums it in registers, stages dGate and dBeta
    (db) for the epilogue's TMA stores, and stages dQ/dK for the epilogue's
    TMA stores."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_hold.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    wg1_sp = warp_idx % 4
    channel = wg1_sp * cfg.threads_per_warp + lane
    row_addr = (tmem_row + wg1_sp * cfg.threads_per_warp) << 16
    cg2_tidx = channel

    raw_index = PipelineState.start(phase=0)
    dq_acc_index = PipelineState.start(phase=0)
    dk_decay_part_index = PipelineState.start(phase=0)
    dk_inv_part_index = PipelineState.start(phase=0)
    dk_restore_part_index = PipelineState.start(phase=0)
    dgate_last_dstate_smem_index = PipelineState.start(phase=0)
    gbase = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    SFIRST_MIN = 1 if cfg.use_initial_state else 2
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        sk_nt = cend - wstart
        for rev_idx in cutlass.range(sk_nt, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            gc = gbase + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            raw_stage = gc % cfg.smem_raw_stages
            decay_stage = gc % cfg.smem_decay_stages
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)
            sBetaP_ptr = sBeta_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = sGate_raw.data_ptr() + raw_stage * (cfg.d_k * cfg.b_t)
            writes = chunk_idx < wend

            # ---- raw q/k/beta/gate landed: CG0 publishes the decay ring only after
            # consuming them, so this wait is CG2's visibility guard ---------------
            bars.mb_k_decay_inv_ready[decay_stage].wait((gc // cfg.smem_decay_stages) % 2)

            # ---- per-channel gate factors ----------------------------------------
            f32_seg = channel // 32
            f32_dim = channel - f32_seg * 32
            f16_seg = channel // 64
            f16_dim = channel - f16_seg * 64
            eg = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            for t in cutlass.range_constexpr(cfg.b_t):
                eg[t] = (sGate_ptr + f32_seg * (cfg.b_t * 32) + t * 32 + swizzle_xor_128b(t, f32_dim, elem_bytes=4)).load()
            egl = eg[cfg.b_t - 1]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_gate_done[raw_stage].arrive()

            # ---- staged raw Q/K: TMEM ring cols for this chunk -------------------
            qk_raw_stage = gc % cfg.tmem_qk_raw_stages
            qraw_col = tmem_col + cfg.tmem_qraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)
            kraw_col = tmem_col + cfg.tmem_kraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)
            norm_base = qk_raw_stage * (2 * cfg.b_t)
            bars.mb_qk_raw_ready[qk_raw_stage].wait((gc // cfg.tmem_qk_raw_stages) % 2)

            # ---- dGate_last hdot: sum_v sdH[v, c] * S0[c, v] ---------------------
            dgate_last_val = cutlass.Float32(0.0)
            bars.mb_state_inp_ready[gc % 2].wait((gc // 2) % 2)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dgate_last_dstate_smem_index.phase)
                dgate_last_dstate_smem_index = advance(dgate_last_dstate_smem_index, 1)
                for pl in cutlass.range_constexpr(2):
                    for row_half in cutlass.range_constexpr(2):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_addr + (tmem_col + cfg.tmem_state_inp_offset + (gc % 2) * (cfg.d_v // 2) + pl * 32 + row_half * 16), cutlass.Float32
                            ),
                            num=16,
                        )
                        hacc = cutlass.Array(cutlass.Float32, 8, alignment=16)
                        for i in cutlass.range_constexpr(8):
                            hacc[i] = opaque_f32_zero()
                        for j in cutlass.range_constexpr(16):
                            v0 = pl * 64 + row_half * 32 + 2 * j
                            state_pair = cutlass.Vector.from_elements((state_vec[j],), cutlass.Float32).bitcast(cfg.io_dtype)
                            dstate_addr0 = (channel // 64) * (cfg.d_v * 64) + v0 * 64 + swizzle_xor_128b(v0, channel % 64, elem_bytes=2)
                            dstate_addr1 = (channel // 64) * (cfg.d_v * 64) + (v0 + 1) * 64 + swizzle_xor_128b(v0 + 1, channel % 64, elem_bytes=2)
                            hval0 = (sDstate_raw.data_ptr() + dstate_addr0).load().to(cutlass.Float32)
                            hval1 = (sDstate_raw.data_ptr() + dstate_addr1).load().to(cutlass.Float32)
                            hacc[(2 * j) % 8] = hacc[(2 * j) % 8] + hval0 * state_pair[0].to(cutlass.Float32)
                            hacc[(2 * j + 1) % 8] = hacc[(2 * j + 1) % 8] + hval1 * state_pair[1].to(cutlass.Float32)
                        pa0, pb0 = fadd2(hacc[0], hacc[2], hacc[4], hacc[6])
                        pa1, pb1 = fadd2(hacc[1], hacc[3], hacc[5], hacc[7])
                        part_a, part_b = fadd2(pa0, pb0, pa1, pb1)
                        dgate_last_val = dgate_last_val + (part_a + part_b)
                bars.mb_dstate_smem_cg2_done.arrive()
            bars.mb_state_inp_cg2_done[gc % 2].arrive()

            # ---- part-drain accumulators -------------------------------------------
            dq_n = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            dk_n = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            db_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            dgate_regs = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
            dgate_last_acc = cutlass.Array(cutlass.Float32, 4, alignment=16)
            for i in cutlass.range_constexpr(4):
                dgate_last_acc[i] = opaque_f32_zero()
            for t in cutlass.range_constexpr(cfg.b_t):
                dk_n[t] = cutlass.Float32(0.0)

            # ---- dK_restore part drain: (eGl/eG) scale + dGate_last k-dot ----------
            if has_dstate:
                bars.mb_dk_restore_part_acc_ready.wait(dk_restore_part_index.phase)
                dk_restore_part_index = advance(dk_restore_part_index, 1)
                dk_restore_part_vec = nvvm.tcgen05_ld(
                    "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_restore_acc_offset), cutlass.Float32), num=cfg.b_t
                )
                kr_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + kraw_col, cutlass.Float32), num=cfg.b_t // 2)
                for t in cutlass.range_constexpr(cfg.b_t):
                    dk_hat = egl * cute.math.rcp(eg[t], approx=True, ftz=True) * dk_restore_part_vec[t]
                    dk_n[t] = dk_hat
                    k_pair = cutlass.Vector.from_elements((kr_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                    k_v = k_pair[t % 2].to(cutlass.Float32)
                    if cutlass.const_expr(cfg.l2norm):
                        k_v = k_v * sNorm_raw[norm_base + cfg.b_t + t]
                    dgate_last_acc[t % 4] = dgate_last_acc[t % 4] + k_v * dk_hat

            # ---- dQ acc drain: eG.scale ---------------------------------------------
            bars.mb_dq_acc_ready.wait(dq_acc_index.phase)
            dq_acc_index = advance(dq_acc_index, 1)
            dq_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dq_acc_offset), cutlass.Float32), num=cfg.b_t)
            for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                t = 2 * t2
                es_lo, es_hi = fmul2(eg[t], eg[t + 1], scale, scale)
                dq_n[t], dq_n[t + 1] = fmul2(es_lo, es_hi, dq_vec[t], dq_vec[t + 1])

            # ---- dK_inv part drain: (dA - dM) term, 1/eG scale ----------------------
            bars.mb_dk_inv_part_acc_ready.wait(dk_inv_part_index.phase)
            dk_inv_part_index = advance(dk_inv_part_index, 1)
            dk_inv_part_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_inv_acc_offset), cutlass.Float32), num=cfg.b_t)
            for t in cutlass.range_constexpr(cfg.b_t):
                dk_n[t] = dk_n[t] + dk_inv_part_vec[t] * cute.math.rcp(eg[t], approx=True, ftz=True)

            # ---- dK_decay part drain: -eG scale, seeds dBeta and dGate --------------
            bars.mb_dk_decay_part_acc_ready.wait(dk_decay_part_index.phase)
            dk_decay_part_index = advance(dk_decay_part_index, 1)
            dk_decay_part_vec = nvvm.tcgen05_ld(
                "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_decay_acc_offset), cutlass.Float32), num=cfg.b_t
            )
            kd_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + kraw_col, cutlass.Float32), num=cfg.b_t // 2)
            for t in cutlass.range_constexpr(cfg.b_t):
                dk_decay = -eg[t] * dk_decay_part_vec[t]
                k_pair = cutlass.Vector.from_elements((kd_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                k_v = k_pair[t % 2].to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    k_v = k_v * sNorm_raw[norm_base + cfg.b_t + t]
                db_regs[t] = k_v * dk_decay
                beta_v = (sBetaP_ptr + f16_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, f16_dim, elem_bytes=2)).load().to(cutlass.Float32)
                if cutlass.const_expr(cfg.beta_sigmoid):
                    half = cutlass.Float32(0.5)
                    beta_v = (cute.math.tanh(beta_v * half, approx=True) * half + half).to(cfg.io_dtype).to(cutlass.Float32)
                dgate_regs[t] = beta_v * dk_decay
                dk_n[t] = dk_n[t] + dgate_regs[t]

            nvvm.tcgen05_wait("load")
            bars.mb_dqk_acc_done.arrive()

            # ---- dGate finalize --------------------------------------------------
            qf_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qraw_col, cutlass.Float32), num=cfg.b_t // 2)
            kf_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + kraw_col, cutlass.Float32), num=cfg.b_t // 2)
            for t in cutlass.range_constexpr(cfg.b_t):
                q_pair = cutlass.Vector.from_elements((qf_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                k_pair = cutlass.Vector.from_elements((kf_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                q_v = q_pair[t % 2].to(cutlass.Float32)
                k_v = k_pair[t % 2].to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    q_v = q_v * sNorm_raw[norm_base + t]
                    k_v = k_v * sNorm_raw[norm_base + cfg.b_t + t]
                beta_v = (sBetaP_ptr + f16_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, f16_dim, elem_bytes=2)).load().to(cutlass.Float32)
                if cutlass.const_expr(cfg.beta_sigmoid):
                    half = cutlass.Float32(0.5)
                    beta_v = (cute.math.tanh(beta_v * half, approx=True) * half + half).to(cfg.io_dtype).to(cutlass.Float32)
                dgate_regs[t] = q_v * dq_n[t] + beta_v * db_regs[t] - k_v * (dk_n[t] - dgate_regs[t])
                if cutlass.const_expr(cfg.beta_sigmoid):
                    # after dgate, which consumes db_regs pre-chain-rule
                    db_regs[t] = db_regs[t] * (beta_v - beta_v * beta_v)
            dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + ((dgate_last_acc[0] + dgate_last_acc[1]) + (dgate_last_acc[2] + dgate_last_acc[3]))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_beta_done[raw_stage].arrive()

            # ---- L2-norm backward row projection ---------------------------------
            if cutlass.const_expr(cfg.l2norm):
                for grad, qk_col, inv_off in ((dq_n, qraw_col, 0), (dk_n, kraw_col, cfg.b_t)):
                    dots = cutlass.Array(cutlass.Float32, cfg.b_t, alignment=16)
                    for half in cutlass.range_constexpr(2):
                        p_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qk_col + half * (cfg.b_t // 4), cutlass.Float32), num=cfg.b_t // 4)
                        for tt2 in cutlass.range_constexpr(cfg.b_t // 4):
                            tt = 2 * tt2
                            t = half * (cfg.b_t // 2) + tt
                            p_pair = cutlass.Vector.from_elements((p_words[tt2],), cutlass.Float32).bitcast(cfg.io_dtype)
                            gp_lo, gp_hi = fmul2(grad[t], grad[t + 1], p_pair[0].to(cutlass.Float32), p_pair[1].to(cutlass.Float32))
                            dots[t], dots[t + 1] = fmul2(gp_lo, gp_hi, sNorm_raw[norm_base + inv_off + t], sNorm_raw[norm_base + inv_off + t + 1])
                    for off in cutlass.range_constexpr(5):
                        step = cutlass.const_expr(1 << off)
                        for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                            t = 2 * t2
                            bfly_lo = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, dots[t], step, 31, kind=nvvm.Shfl.BFLY))
                            bfly_hi = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, dots[t + 1], step, 31, kind=nvvm.Shfl.BFLY))
                            dots[t], dots[t + 1] = fadd2(dots[t], dots[t + 1], bfly_lo, bfly_hi)
                    if lane == 0:
                        for t in cutlass.range_constexpr(cfg.b_t):
                            sRed1_raw[wg1_sp * cfg.b_t + t] = dots[t]
                    nvvm.barrier_cta_sync(cfg.cg2_sync_barrier_id, thread_count=cfg.cg2_threads)
                    for half in cutlass.range_constexpr(2):
                        a_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qk_col + half * (cfg.b_t // 4), cutlass.Float32), num=cfg.b_t // 4)
                        for tt2 in cutlass.range_constexpr(cfg.b_t // 4):
                            t = half * (cfg.b_t // 2) + 2 * tt2
                            a_pair = cutlass.Vector.from_elements((a_words[tt2],), cutlass.Float32).bitcast(cfg.io_dtype)
                            dot_lo, dot_hi = fadd2(sRed1_raw[t], sRed1_raw[t + 1], sRed1_raw[cfg.b_t + t], sRed1_raw[cfg.b_t + t + 1])
                            dot_lo, dot_hi = fadd2(dot_lo, dot_hi, sRed1_raw[2 * cfg.b_t + t], sRed1_raw[2 * cfg.b_t + t + 1])
                            dot_lo, dot_hi = fadd2(dot_lo, dot_hi, sRed1_raw[3 * cfg.b_t + t], sRed1_raw[3 * cfg.b_t + t + 1])
                            norm_lo = sNorm_raw[norm_base + inv_off + t]
                            norm_hi = sNorm_raw[norm_base + inv_off + t + 1]
                            grad[t] = (grad[t] - a_pair[0].to(cutlass.Float32) * norm_lo * dot_lo) * norm_lo
                            grad[t + 1] = (grad[t + 1] - a_pair[1].to(cutlass.Float32) * norm_hi * dot_hi) * norm_hi
                    nvvm.barrier_cta_sync(cfg.cg2_sync_barrier_id, thread_count=cfg.cg2_threads)

            nvvm.tcgen05_wait("load")
            bars.mb_qk_raw_done[qk_raw_stage].arrive()

            # ---- stage dQ/dK for the epilogue TMA stores -------------------------
            dq_stage = gc % cfg.smem_dq_stages
            dk_stage = gc % cfg.smem_dk_stages
            bars.mb_dq_tmastg_done[dq_stage].wait(((gc // cfg.smem_dq_stages) + 1) % 2)
            bars.mb_dk_tmastg_done[dk_stage].wait(((gc // cfg.smem_dk_stages) + 1) % 2)
            dq_base = dq_stage * (cfg.b_t * cfg.d_k)
            dk_base = dk_stage * (cfg.b_t * cfg.d_k)
            for t in cutlass.range_constexpr(cfg.b_t):
                out_idx = f16_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, f16_dim, elem_bytes=2)
                (sDq_raw.data_ptr() + dq_base + out_idx).store(dq_n[t].to(cfg.io_dtype))
                (sDk_raw.data_ptr() + dk_base + out_idx).store(dk_n[t].to(cfg.io_dtype))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_tmastg_ready[dq_stage].arrive()
            bars.mb_dk_tmastg_ready[dk_stage].arrive()

            # ---- dGate_last add --------------------------------------------------
            if has_dstate:
                if chunk_idx >= FIRST_STATE_CHUNK:
                    dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + egl * dgate_last_val

            # ---- dGate reverse cumsum --------------------------------------------
            suffix = cutlass.Float32(0.0)
            for rt in cutlass.range_constexpr(cfg.b_t):
                t = cfg.b_t - 1 - rt
                suffix = suffix + dgate_regs[t]
                dgate_regs[t] = suffix

            # ---- stage dGate + dBeta for the epilogue TMA stores -----------------
            dgate_stage = gc % cfg.smem_dgate_stages
            bars.mb_dgate_tmastg_done[dgate_stage].wait(((gc // cfg.smem_dgate_stages) + 1) % 2)
            db_stage = gc % cfg.smem_db_stages
            bars.mb_db_tmastg_done[db_stage].wait(((gc // cfg.smem_db_stages) + 1) % 2)
            for t in cutlass.range_constexpr(cfg.b_t):
                dgate_idx = f32_seg * (cfg.b_t * 32) + t * 32 + swizzle_xor_128b(t, f32_dim, elem_bytes=4)
                (sDgate_raw.data_ptr() + dgate_idx).store(dgate_regs[t])
                db_idx = f16_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, f16_dim, elem_bytes=2)
                (sDb_raw.data_ptr() + db_idx).store(db_regs[t].to(cfg.io_dtype))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dgate_tmastg_ready[dgate_stage].arrive()
            bars.mb_db_tmastg_ready[db_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        gbase += sk_nt
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()


# ---------------------------------------------------------------------------
# Host-side assembly
# ---------------------------------------------------------------------------


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_gate,
    base_do,
    base_beta,
    base_w,
    base_dq,
    base_dk,
    base_dv,
    base_dgate,
    base_dwo,
    base_dbo,
    base_checkpoint,
    base_initial_state,
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dwo: cute.Tensor,
    dbo: cute.Tensor,
    state_checkpoints: cute.Tensor,
    initial_state: cute.Tensor | None,
    n_batch: cutlass.Int32,
    q_rs: cutlass.Int32,
    k_rs: cutlass.Int32,
    v_rs: cutlass.Int32,
    g_rs: cutlass.Int32,
    do_rs: cutlass.Int32,
    beta_rs: cutlass.Int32,
    w_rs: cutlass.Int32,
    dq_rs: cutlass.Int32,
    dk_rs: cutlass.Int32,
    dv_rs: cutlass.Int32,
    dgate_rs: cutlass.Int32,
    dwo_rs: cutlass.Int32,
    dbo_rs: cutlass.Int32,
    checkpoint_rs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_ws.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_ws.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub2 = cute.make_tensor(desc_ws.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub3 = cute.make_tensor(desc_ws.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub4 = cute.make_tensor(desc_ws.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub5 = cute.make_tensor(desc_ws.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub6 = cute.make_tensor(desc_ws.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub7 = cute.make_tensor(desc_ws.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub8 = cute.make_tensor(desc_ws.iterator + 8 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub9 = cute.make_tensor(desc_ws.iterator + 9 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub10 = cute.make_tensor(desc_ws.iterator + 10 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub11 = cute.make_tensor(desc_ws.iterator + 11 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub12 = cute.make_tensor(desc_ws.iterator + 12 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub13 = cute.make_tensor(desc_ws.iterator + 13 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub14 = cute.make_tensor(desc_ws.iterator + 14 * arr_words, cute.make_layout((cutlass.Int32(TENSOR_MAP_QWORDS),), stride=(1,)))

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
            emit_seq_descs(base_gate, sub3, cu_seqlens, gate, n_batch, g_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        if nvvm.elect_sync():
            emit_seq_descs(base_do, sub4, cu_seqlens, do, n_batch, do_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 5:
        if nvvm.elect_sync():
            emit_seq_descs(base_beta, sub5, cu_seqlens, beta, n_batch, beta_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 6:
        if nvvm.elect_sync():
            emit_seq_descs(base_w, sub6, cu_seqlens, w, n_batch, w_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 7:
        if nvvm.elect_sync():
            emit_seq_descs(base_dq, sub7, cu_seqlens, dq, n_batch, dq_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 8:
        if nvvm.elect_sync():
            emit_seq_descs(base_dk, sub8, cu_seqlens, dk, n_batch, dk_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 9:
        if nvvm.elect_sync():
            emit_seq_descs(base_dv, sub9, cu_seqlens, dv, n_batch, dv_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 10:
        if nvvm.elect_sync():
            emit_seq_descs(base_dgate, sub10, cu_seqlens, dgate, n_batch, dgate_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 11:
        if nvvm.elect_sync():
            emit_seq_descs(base_dwo, sub11, cu_seqlens, dwo, n_batch, dwo_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 12:
        if nvvm.elect_sync():
            emit_seq_descs(base_dbo, sub12, cu_seqlens, dbo, n_batch, dbo_rs, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 13:
        if nvvm.elect_sync():
            emit_checkpoint_seq_descs(base_checkpoint, sub13, cu_seqlens, state_checkpoints, n_batch, checkpoint_rs, checkpoint_every_n, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(initial_state is not None):
        if widx == 14:
            if nvvm.elect_sync():
                emit_copy_desc(base_initial_state, sub14)
                nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def prologue_kernel(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_do: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_beta: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_w: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dq: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dk: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dv: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dgate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dwo: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dbo: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_initial_state: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dwo: cute.Tensor,
    dbo: cute.Tensor,
    state_checkpoints: cute.Tensor,
    initial_state: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mSched: cute.Tensor | None,
    n_batch: cutlass.Int32,
    q_rs: cutlass.Int32,
    k_rs: cutlass.Int32,
    v_rs: cutlass.Int32,
    g_rs: cutlass.Int32,
    do_rs: cutlass.Int32,
    beta_rs: cutlass.Int32,
    w_rs: cutlass.Int32,
    dq_rs: cutlass.Int32,
    dk_rs: cutlass.Int32,
    dv_rs: cutlass.Int32,
    dgate_rs: cutlass.Int32,
    dwo_rs: cutlass.Int32,
    dbo_rs: cutlass.Int32,
    checkpoint_rs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
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
        n_heads_out = cutlass.Int32(gate.shape[1])
        order_body(
            order_gen,
            has_sched,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMS,
            tidx,
            n_heads_out,
            n_heads_out * n_batch,
            cu_seqlens,
            mStaging,
            mCount,
            mWorkItems,
            mSched,
            sKey,
            sIdx,
            sSpread,
        )
    build_descs_body(
        widx,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_do,
        base_beta,
        base_w,
        base_dq,
        base_dk,
        base_dv,
        base_dgate,
        base_dwo,
        base_dbo,
        base_checkpoint,
        base_initial_state,
        desc_ws,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        do,
        beta,
        w,
        dq,
        dk,
        dv,
        dgate,
        dwo,
        dbo,
        state_checkpoints,
        initial_state,
        n_batch,
        q_rs,
        k_rs,
        v_rs,
        g_rs,
        do_rs,
        beta_rs,
        w_rs,
        dq_rs,
        dk_rs,
        dv_rs,
        dgate_rs,
        dwo_rs,
        dbo_rs,
        checkpoint_rs,
        checkpoint_every_n,
    )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dwo: cute.Tensor,
    dbo: cute.Tensor,
    state_checkpoints: cute.Tensor,
    initial_state: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    sched_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when this kernel is
    the table's first consumer) and build the 15 per-(batch, head)
    TMA-descriptor arrays into ``tensormap_workspace``."""
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

    q_headed = cute.make_tensor(q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0])))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])))
    do_headed = cute.make_tensor(do.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, do.stride[1], do.stride[0])))
    beta_headed = cute.make_tensor(beta.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, beta.stride[1], beta.stride[0])))
    w_headed = cute.make_tensor(w.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, w.stride[1], w.stride[0])))
    dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dq.stride[1], dq.stride[0])))
    dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dk.stride[1], dk.stride[0])))
    dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, dv.stride[1], dv.stride[0])))
    dgate_headed = cute.make_tensor(dgate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dgate.stride[1], dgate.stride[0])))
    dwo_headed = cute.make_tensor(dwo.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, dwo.stride[1], dwo.stride[0])))
    dbo_headed = cute.make_tensor(dbo.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dbo.stride[1], dbo.stride[0])))

    swz = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(q_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_do = cuda.create_tensor_map_tiled_from_view(do_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_beta = cuda.create_tensor_map_tiled_from_view(beta_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_w = cuda.create_tensor_map_tiled_from_view(w_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dq = cuda.create_tensor_map_tiled_from_view(dq_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dk = cuda.create_tensor_map_tiled_from_view(dk_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dv = cuda.create_tensor_map_tiled_from_view(dv_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dgate = cuda.create_tensor_map_tiled_from_view(dgate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dwo = cuda.create_tensor_map_tiled_from_view(dwo_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dbo = cuda.create_tensor_map_tiled_from_view(dbo_headed, box_dims=(granu, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)

    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (d_v, d_k, state_checkpoints.shape[0], ho),
            stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
        ),
    )
    base_checkpoint = cuda.create_tensor_map_tiled_from_view(checkpoint_view, box_dims=(64, d_k, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swz)
    base_initial_state = base_checkpoint
    if cutlass.const_expr(initial_state is not None):
        initial_state_view = cute.make_tensor(
            initial_state.iterator,
            cute.make_layout(
                (d_v, d_k, ho, batch_size),
                stride=(initial_state.stride[3], initial_state.stride[2], initial_state.stride[1], initial_state.stride[0]),
            ),
        )
        base_initial_state = cuda.create_tensor_map_tiled_from_view(initial_state_view, box_dims=(64, d_k, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swz)

    prologue_kernel(
        run_order,
        order_gen,
        has_sched,
        b_t,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_do,
        base_beta,
        base_w,
        base_dq,
        base_dk,
        base_dv,
        base_dgate,
        base_dwo,
        base_dbo,
        base_checkpoint,
        base_initial_state,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        do,
        beta,
        w,
        dq,
        dk,
        dv,
        dgate,
        dwo,
        dbo,
        state_checkpoints,
        initial_state,
        work_item_staging,
        work_count,
        work_items,
        sched_all,
        cutlass.Int32(batch_size),
        cutlass.Int32(q.stride[0]),
        cutlass.Int32(k.stride[0]),
        cutlass.Int32(v.stride[0]),
        cutlass.Int32(gate.stride[0]),
        cutlass.Int32(do.stride[0]),
        cutlass.Int32(beta.stride[0]),
        cutlass.Int32(w.stride[0]),
        cutlass.Int32(dq.stride[0]),
        cutlass.Int32(dk.stride[0]),
        cutlass.Int32(dv.stride[0]),
        cutlass.Int32(dgate.stride[0]),
        cutlass.Int32(dwo.stride[0]),
        cutlass.Int32(dbo.stride[0]),
        cutlass.Int32(state_checkpoints.stride[0]),
        cutlass.Int32(b_t),
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    state_checkpoints: cute.Tensor,
    mState_init: cute.Tensor | None,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    cu_seqlens: cute.Tensor,
    d_initial_state: cute.Tensor | None,
    d_final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    sched_ctr: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    scale: cutlass.Float32,
    stream,
) -> None:
    num_sequences = cu_seqlens.shape[0] - 1

    # ---- launch ------------------------------------------------------------------
    n_desc = num_sequences
    grid_shape = (cfg.max_active_clusters, 1, 1)
    kernel(
        cfg,
        tensormap_workspace,
        n_desc,
        cu_seqlens,
        a_log,
        dt_bias,
        dgate,
        dbeta,
        dw,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        sched_ctr,
        scale,
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
    cu_seqlens: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mDgate: cute.Tensor,
    mDb: cute.Tensor,
    mDw_out: cute.Tensor,
    mDstate0: cute.Tensor | None,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
    scale: cutlass.Float32,
) -> None:
    """BT=16 GDN-2 backward kernel (persistent, 16 warps)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    assert mDgate.element_type == cutlass.Float32
    assert mDb.element_type == cfg.io_dtype and mDw_out.element_type == cfg.io_dtype

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_beta_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_w_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(7) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(8) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(9) * arr_words
    desc_dgate_base = desc_base_words + cutlass.Int32(10) * arr_words
    desc_dwo_base = desc_base_words + cutlass.Int32(11) * arr_words
    desc_db_base = desc_base_words + cutlass.Int32(12) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(13) * arr_words
    desc_initial_state_base = desc_base_words + cutlass.Int32(14) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_gdn2_bwd_bars(cfg)
    tmem_hold = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    STATE_ALT_LEAD = cfg.d_v * 128

    # sub-bank split: tcgen05-descriptor operands low, generic-client buffers high
    sRed1_raw = cutlass.Array(cutlass.Float32, 4 * cfg.b_t, space=SMEM, alignment=64)
    sNorm_raw = cutlass.Array(cutlass.Float32, cfg.tmem_qk_raw_stages * 2 * cfg.b_t, space=SMEM, alignment=64)
    sState_raw = cutlass.Array(cfg.io_dtype, cfg.state_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDstate_raw = cutlass.Array(cfg.io_dtype, cfg.d_k * cfg.d_v, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_decay_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_inv_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_restore_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_decay_raw = cutlass.Array(cfg.io_dtype, cfg.operand_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sState_scale_diag_raw = cutlass.Array(cfg.io_dtype, cfg.diag_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sIntermediate_raw = cutlass.Array(cfg.io_dtype, cfg.intermediate_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDo_raw = cutlass.Array(cfg.io_dtype, cfg.raw_v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sBeta_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    # sub-bank fill: high group starts at the 128KB midpoint
    smem_bank_fill = cutlass.Array(cfg.io_dtype, 1024 // bpe, space=SMEM, alignment=cfg.buffer_align_bytes)
    sQ_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sK_raw = cutlass.Array(cfg.io_dtype, cfg.raw_qk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sV_raw = cutlass.Array(cfg.io_dtype, cfg.raw_v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sGate_raw = cutlass.Array(cutlass.Float32, cfg.raw_gate_cosize, space=SMEM, alignment=1024)
    sW_raw = cutlass.Array(cfg.io_dtype, cfg.raw_v_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sU_raw = cutlass.Array(cfg.io_dtype, cfg.b_t * cfg.d_v, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDy_raw = cutlass.Array(cfg.io_dtype, cfg.b_t * cfg.d_v, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDq_raw = cutlass.Array(cfg.io_dtype, cfg.dq_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDk_raw = cutlass.Array(cfg.io_dtype, cfg.dk_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDv_raw = cutlass.Array(cfg.io_dtype, cfg.dv_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDgate_raw = cutlass.Array(cutlass.Float32, cfg.dgate_cosize, space=SMEM, alignment=1024)
    sDwOut_raw = cutlass.Array(cfg.io_dtype, cfg.dwo_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)
    sDb_raw = cutlass.Array(cfg.io_dtype, cfg.db_cosize, space=SMEM, alignment=cfg.buffer_align_bytes)

    sState_alt = SmemTile(
        base=sState_raw.data_ptr().toint(),
        elems_per_stage=((cfg.state_cosize) // (cfg.smem_state_stages)) * bpe,
        stages=cfg.smem_state_stages,
        leading_byte_offset=STATE_ALT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_decay_lead16 = SmemTile(
        base=sK_decay_raw.data_ptr().toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_restore_lead16 = SmemTile(
        base=sK_restore_raw.data_ptr().toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_lead16 = SmemTile(
        base=sDo_raw.data_ptr().toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bpe,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_amaj = SmemTile(
        base=sDo_raw.data_ptr().toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bpe,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sU_lead16 = SmemTile(
        base=sU_raw.data_ptr().toint(),
        elems_per_stage=((cfg.b_t * cfg.d_v) // (1)) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDy_lead16 = SmemTile(
        base=sDy_raw.data_ptr().toint(),
        elems_per_stage=((cfg.b_t * cfg.d_v) // (1)) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDstate_alt = SmemTile(
        base=sDstate_raw.data_ptr().toint(),
        elems_per_stage=((cfg.d_k * cfg.d_v) // (1)) * bpe,
        stages=1,
        leading_byte_offset=STATE_ALT_LEAD,
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
    sK_inv_amaj = SmemTile(
        base=sK_inv_raw.data_ptr().toint(),
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
    sState_scale_diag = SmemTile(
        base=sState_scale_diag_raw,
        elems_per_stage=((cfg.d_k // 16) * 256),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * 16 * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
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
    gate_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * 4)
    beta_tx_bytes = cutlass.const_expr(cfg.d_k * cfg.b_t * bpe)
    do_tx_bytes = cutlass.const_expr(cfg.d_v * cfg.b_t * bpe)
    v_tx_bytes = cutlass.const_expr(cfg.d_v * cfg.b_t * bpe)
    w_tx_bytes = cutlass.const_expr(cfg.d_v * cfg.b_t * bpe)

    elect_one = nvvm.elect_sync()
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
                bars.mb_do_done[stage].init()
                bars.mb_do_mma_done[stage].init()
                bars.mb_v_ready[stage].init()
                bars.mb_v_done[stage].init()
                bars.mb_w_ready[stage].init()
                bars.mb_w_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_state_stages):
                bars.mb_state_ready[stage].init()
                bars.mb_state_done[stage].init()
                bars.mb_state_cg0_done[stage].init()
            for stage in cutlass.range_constexpr(2):
                bars.mb_state_inp_ready[stage].init()
                bars.mb_state_inp_done[stage].init()
                bars.mb_state_inp_cg2_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_ready.init()
            bars.mb_y_inp_ready.init()
            bars.mb_u_acc_ready.init()
            bars.mb_u_smem_ready.init()
            bars.mb_du_acc_ready.init()
            bars.mb_du_inp_ready.init()
            bars.mb_dy_acc_ready.init()
            bars.mb_neg_dy_inp_ready.init()
            bars.mb_dy_smem_ready.init()
            bars.mb_dy_smem_done.init()
            bars.mb_dstate_acc_ready.init()
            bars.mb_dstate_inp_ready.init()
            bars.mb_dstate_smem_ready.init()
            bars.mb_dstate_smem_done.init()
            bars.mb_dstate_smem_cg2_done.init()
            bars.mb_dq_acc_ready.init()
            bars.mb_dk_decay_part_acc_ready.init()
            bars.mb_dk_inv_part_acc_ready.init()
            bars.mb_dk_restore_part_acc_ready.init()
            bars.mb_dqk_acc_done.init()
            bars.mb_dstate0_acc_stored.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.super_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_ready[stage].init()
                bars.mb_q_decay_k_restore_ready[stage].init()
                bars.mb_decay_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.tmem_qk_raw_stages):
                bars.mb_qk_raw_ready[stage].init()
                bars.mb_qk_raw_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_a_ready[stage].init()
                bars.mb_da_ready[stage].init()
                bars.mb_dm_ready[stage].init()
                bars.mb_a_done[stage].init()
                bars.mb_t_inv_done[stage].init()
                bars.mb_da_done[stage].init()
                bars.mb_dm_done[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_dq_stages):
                bars.mb_dq_tmastg_ready[stage].init()
                bars.mb_dq_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dk_stages):
                bars.mb_dk_tmastg_ready[stage].init()
                bars.mb_dk_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dgate_stages):
                bars.mb_dgate_tmastg_ready[stage].init()
                bars.mb_dgate_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_db_stages):
                bars.mb_db_tmastg_ready[stage].init()
                bars.mb_db_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dv_stages):
                bars.mb_dv_tmastg_ready[stage].init()
                bars.mb_dv_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dwo_stages):
                bars.mb_dwo_tmastg_ready[stage].init()
                bars.mb_dwo_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.sched_stages):
                bars.mb_sched_ready[stage].init()
                bars.mb_sched_done[stage].init()
    diag_zero = cfg.io_dtype(0.0)
    for diag_idx in cutlass.range(tidx, cfg.diag_cosize, cfg.threads_per_cta, unroll=1):
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
            mSched,
            sSched,
            q_tx_bytes,
            k_tx_bytes,
            gate_tx_bytes,
            beta_tx_bytes,
            do_tx_bytes,
            v_tx_bytes,
            w_tx_bytes,
            sQ_raw,
            sK_raw,
            sV_raw,
            sGate_raw,
            sDo_raw,
            sBeta_raw,
            sW_raw,
            sState_raw,
            desc_q_base,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
            desc_do_base,
            desc_beta_base,
            desc_w_base,
            desc_checkpoint_base,
            desc_initial_state_base,
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
            sSched,
            lane,
            sK_decay_raw,
            sK_inv_raw,
            sU_raw,
            sDy_raw,
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
            sSched,
            tmem_hold,
            sState_alt,
            sK_decay_lead16,
            sK_inv_amaj,
            sK_restore_lead16,
            sDo_lead16,
            sDo_amaj,
            sQ_decay_trans,
            sK_decay_trans,
            sU_lead16,
            sDy_lead16,
            sDstate_alt,
            sIntermediate,
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
            sSched,
            lane,
            sK_inv_raw,
            sQ_decay_raw,
            sDo_raw,
            sU_raw,
            sIntermediate_raw,
            sDq_raw,
            sDk_raw,
            sDv_raw,
            sDgate_raw,
            sDb_raw,
            sDwOut_raw,
            desc_dq_base,
            desc_dk_base,
            desc_dv_base,
            desc_dgate_base,
            desc_db_base,
            desc_dwo_base,
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
            sSched,
            lane,
            tmem_hold,
            warp_idx,
            scale,
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_raw,
            sK_raw,
            sQ_raw,
            sState_raw,
            sV_raw,
            sDo_raw,
            sBeta_raw,
            sW_raw,
            sNorm_raw,
            sK_decay_raw,
            sK_restore_raw,
            sQ_decay_raw,
            sState_scale_diag_raw,
            bars,
        )
    elif warp_idx >= cfg.compute_group_2_warp_ids[0] and warp_idx <= cfg.compute_group_2_warp_ids[-1]:
        compute2_warp_group(
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
            sBeta_raw,
            sGate_raw,
            sNorm_raw,
            sDq_raw,
            sDk_raw,
            sRed1_raw,
            sDstate_raw,
            sDgate_raw,
            sDb_raw,
            scale,
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
            sSched,
            lane,
            tmem_hold,
            warp_idx,
            mDstate0,
            mDstate_in,
            sV_raw,
            sW_raw,
            sDo_raw,
            sU_raw,
            sDy_raw,
            sDv_raw,
            sDwOut_raw,
            sDstate_raw,
            bars,
        )


@dataclass
class Gdn2BwdCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``)."""

    io_dtype: Type[cutlass.Numeric]
    use_dstate_in: bool
    use_dstate0: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    use_initial_state: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    dyn_sched: bool = False
    sched_stages: int = 8

    # ---- fixed constants stamped from CFG by build_cfg ---------------------------
    compute_group_0_warp_ids: tuple = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_2_warp_ids: tuple = CFG.COMPUTE_GROUP_2_WARP_IDS
    compute_group_1_warp_ids: tuple = CFG.COMPUTE_GROUP_1_WARP_IDS
    super_mma_warp_id: int = CFG.SUPER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -------------------
    cg0_sync_barrier_id: int = 1
    cg0_threads: int = 0
    cg2_sync_barrier_id: int = 2
    cg2_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    tmem_user_threads: int = 0
    cg1_sync_barrier_id: int = 4
    cg1_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets --------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_state_stages: int = CFG.SMEM_STATE_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_dq_stages: int = CFG.SMEM_DQ_STAGES
    smem_dk_stages: int = CFG.SMEM_DK_STAGES
    smem_dgate_stages: int = CFG.SMEM_DGATE_STAGES
    smem_db_stages: int = CFG.SMEM_DB_STAGES
    smem_dv_stages: int = CFG.SMEM_DV_STAGES
    smem_dwo_stages: int = CFG.SMEM_DWO_STAGES
    intermediate_tiles: int = 5
    tmem_dstate_acc_offset: int = 0
    tmem_dstate_inp_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_du_acc_offset: int = 0
    tmem_dy_acc_offset: int = 0
    tmem_dq_acc_offset: int = 0
    tmem_dk_decay_acc_offset: int = 0
    tmem_dk_inv_acc_offset: int = 0
    tmem_dk_restore_acc_offset: int = 0
    tmem_qk_raw_stages: int = 4
    tmem_qraw_inp_offset: int = 0
    tmem_kraw_inp_offset: int = 0
    tmem_y_inp_offset: int = 0
    tmem_du_inp_offset: int = 0
    tmem_neg_dy_inp_offset: int = 0
    tmem_state_inp_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- buffer cosizes / TMA bytes stamped by build_cfg -------------------------
    raw_qk_cosize: int = 0
    raw_v_cosize: int = 0
    raw_gate_cosize: int = 0
    operand_cosize: int = 0
    diag_cosize: int = 0
    intermediate_cosize: int = 0
    state_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dgate_cosize: int = 0
    db_cosize: int = 0
    dv_cosize: int = 0
    dwo_cosize: int = 0
    tma_state_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    use_initial_state: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    dyn_sched: bool = False,
) -> Gdn2BwdCfg:
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = Gdn2BwdCfg(
        io_dtype=io_dtype,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        use_initial_state=use_initial_state,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        dyn_sched=dyn_sched,
    )
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.cg0_threads = len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp
    cfg.cg2_threads = len(cfg.compute_group_2_warp_ids) * cfg.threads_per_warp
    cfg.cg1_threads = len(cfg.compute_group_1_warp_ids) * cfg.threads_per_warp
    cfg.tmem_user_threads = (
        1 + len(cfg.compute_group_2_warp_ids) + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)
    ) * cfg.threads_per_warp

    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dstate_inp_offset = cfg.d_k
    cfg.tmem_state_inp_offset = cfg.tmem_dstate_inp_offset + cfg.d_k // 2
    cfg.tmem_state_k_acc_offset = cfg.tmem_state_inp_offset + cfg.d_v
    cfg.tmem_u_acc_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    cfg.tmem_du_acc_offset = cfg.tmem_u_acc_offset + cfg.b_t
    # dY overwrites the state_k slot: WG1's Y staging consumes state_k
    # before the dY = dU @ T_inv MMA writes (du_inp chain), and the dY
    # readback precedes state_k(c+1) = state @ K_decay^T via
    # neg_dy_ready -> the -dY @ K_decay dstate MMA -> in-order MMA
    cfg.tmem_dy_acc_offset = cfg.tmem_state_k_acc_offset
    cfg.tmem_dq_acc_offset = cfg.tmem_du_acc_offset + cfg.b_t
    cfg.tmem_dk_decay_acc_offset = cfg.tmem_dq_acc_offset + cfg.b_t
    cfg.tmem_dk_inv_acc_offset = cfg.tmem_dk_decay_acc_offset + cfg.b_t
    cfg.tmem_dk_restore_acc_offset = cfg.tmem_dk_inv_acc_offset + cfg.b_t
    cfg.tmem_y_inp_offset = cfg.tmem_dk_restore_acc_offset + cfg.b_t
    # -dY overwrites the y_inp slot: U = Y @ T_inv consumed Y before the dY
    # block runs (u_acc_ready wait), and y_inp(c+1) is gated by
    # state_k_acc_ready(c+1) whose commit covers the -dY @ K_decay MMA (c)
    cfg.tmem_neg_dy_inp_offset = cfg.tmem_y_inp_offset
    cfg.tmem_du_inp_offset = cfg.tmem_y_inp_offset + cfg.b_t // 2
    cfg.tmem_qraw_inp_offset = cfg.tmem_du_inp_offset + cfg.b_t // 2
    cfg.tmem_kraw_inp_offset = cfg.tmem_qraw_inp_offset + cfg.tmem_qk_raw_stages * (cfg.b_t // 2)
    assert cfg.tmem_kraw_inp_offset + cfg.tmem_qk_raw_stages * (cfg.b_t // 2) <= 512

    cfg.raw_qk_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.raw_v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.raw_gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.operand_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k
    cfg.diag_cosize = cfg.smem_decay_stages * (cfg.d_k // 16) * 256
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * cfg.intermediate_tiles * cfg.b_t * cfg.b_t
    cfg.state_cosize = cfg.smem_state_stages * cfg.d_k * cfg.d_v
    cfg.dq_cosize = cfg.smem_dq_stages * cfg.b_t * cfg.d_k
    cfg.dk_cosize = cfg.smem_dk_stages * cfg.b_t * cfg.d_k
    cfg.dgate_cosize = cfg.smem_dgate_stages * cfg.b_t * cfg.d_k
    cfg.db_cosize = cfg.smem_db_stages * cfg.b_t * cfg.d_k
    cfg.dv_cosize = cfg.smem_dv_stages * cfg.b_t * cfg.d_v
    cfg.dwo_cosize = cfg.smem_dwo_stages * cfg.b_t * cfg.d_v
    cfg.tma_state_bytes = cfg.d_k * cfg.d_v * (io_dtype.width // 8)
    return cfg


TENSORMAP_DESC_ARRAYS = 14  # per-batch runtime TMA descriptors: Q, K, V, Gate, dO, Beta, W, Checkpoint, dQ, dK, dV, dGate, dW_out, dBeta
TENSORMAP_STATIC_SLOTS = 1  # initial_state


# ---- Torch adapter / host-side compilation ---------------------------------------


@lru_cache(maxsize=None)
def get_compiled_cache(
    io_dtype_str: str,
    cu_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    beta_sigmoid: bool,
    use_initial_state: bool,
    dyn_sched: bool,
    order_in_prologue: bool,
    order_gen: bool,
    has_sched: bool,
):
    return {}


def chunk_gdn2_bwd_sm100(
    q,
    k,
    v,
    gate,
    beta,
    w,
    do,
    state_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    dw,
    cu_seqlens,
    scale: float,
    *,
    initial_state=None,
    d_initial_state=None,
    d_final_state=None,
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    sched_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    tensormap_workspace,
    stream,
) -> None:
    """Execute the Blackwell BT=16 chunked GDN-2 backward kernel.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO, DK)`` fp32 per-channel decay.  Natural-log
            unless ``safe_gate``, which applies the safe-gate transform
            ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``
        beta: ``(total_tokens, HO, DK)`` io dtype per-key erase.  Post-sigmoid,
            or logits when ``use_beta_sigmoid``
        w: ``(total_tokens, HO, DV)`` io dtype post-sigmoid per-value write
        do: ``(total_tokens, HO, DV)`` io dtype
        state_checkpoints: ``(total_checkpoints, HO, DK, DV)`` io dtype (KV, v contiguous - the GDN
            checkpoint layout), the PLAIN per-chunk series with no initial-state
            slot: sequence-local entry ``c - 1`` is the state ENTERING chunk c >= 1
            of sequence b; chunk 0 seeds from ``initial_state``
        dq/dk/dv: io dtype at ``HO = max(HQ, HV)`` heads, pre-allocated
        dgate: ``(total_tokens, HO, DK)`` fp32 (dL/d ln alpha; ``safe_gate``
            leaves it in the transformed gate space), pre-allocated
        dbeta: ``(total_tokens, HO, DK)`` io dtype (post-sigmoid space, or
            wrt the raw logits under ``beta_sigmoid``), pre-allocated
        dw: ``(total_tokens, HO, DV)`` io dtype, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        scale: attention scale factor
        initial_state: ``(num_seqs, HO, DK, DV)`` io dtype (KV) - the state
            entering chunk 0 (engine-provided zeros when the graph has none)
        d_initial_state: fp32 ``(num_seqs, HO, DK, DV)`` OUT (dL/dS0), or None
        d_final_state: fp32 ``(num_seqs, HO, DK, DV)`` IN (dL/d final state)
        use_qk_l2norm_in_kernel: q/k arrive raw; the kernel normalizes for the
            recompute math and chains the L2-norm backward into dq/dk
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO, DK)`` float32, safe-gate channel bias (None = 0)
        use_beta_sigmoid: ``beta`` holds logits; sigmoid in-kernel
        work_items/work_count: split-K table (``common/split_k.py``, REQUIRED;
            an uncut table row is the whole (b, h) sequence); each item
            computes chunks ``[wstart, cend)`` backward and writes
            gradients only for ``[wstart, wend)``
        sched_ctr: ``(2,)`` int32 zeroed scratch enabling the dynamic
            (work-stealing) tile scheduler
        tensormap_workspace: ``tensormap_workspace_bytes(module, B)`` bytes,
            128-byte aligned, for the per-(batch, head) TMA-descriptor
            arrays (tail chunks clip/zero-fill in hardware)
    """
    HQ = q.shape[1]
    HK = k.shape[1]
    HV = v.shape[1]
    HO = max(HQ, HV)
    use_dstate_in = d_final_state is not None
    use_dstate0 = d_initial_state is not None
    use_initial_state = initial_state is not None
    if work_items is None or work_count is None:
        raise ValueError("work_items/work_count are required (the split-table stage builds them for every launch)")
    dyn_sched = sched_ctr is not None
    order_gen = work_item_scratch is None
    if order_in_prologue and sched_all is None:
        raise ValueError("order_in_prologue requires sched_all (the prologue zeroes both consumers' sched rings)")
    for name, t in (("state_checkpoints", state_checkpoints), ("beta", beta), ("w", w), ("dbeta", dbeta), ("dw", dw)) + (
        (("initial_state", initial_state),) if use_initial_state else ()
    ):
        if str(t.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
            raise ValueError(f"{name} dtype must match the io dtype: got {t.dtype} with io {q.dtype}")
    for name, hh in (("HQ", HQ), ("HK", HK), ("HV", HV)):
        if HO % hh != 0:
            raise ValueError(f"{name}={hh} must divide {HO}")
    B = cu_seqlens.shape[0] - 1
    gate_scale_log2 = gate_lower_bound * LOG2_E

    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if not safe_gate:
        a_log = None
        dt_bias = None

    cu_stream = cuda_driver.CUstream(int(stream))
    cache = get_compiled_cache(
        str(q.dtype),
        str(cu_seqlens.dtype),
        HQ,
        HK,
        HV,
        use_dstate_in,
        use_dstate0,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid,
        use_initial_state,
        dyn_sched,
        order_in_prologue,
        order_gen,
        sched_all is not None,
    )

    if "compiled" not in cache:
        io_dtype = get_dtype(q.dtype)
        cfg = build_cfg(
            io_dtype,
            use_dstate_in=use_dstate_in,
            use_dstate0=use_dstate0,
            l2norm=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            use_initial_state=use_initial_state,
            q_ratio=HO // HQ,
            k_ratio=HO // HK,
            v_ratio=HO // HV,
            n_heads_out=HO,
            max_active_clusters=multiprocessor_count(current_device_id()),
            dyn_sched=dyn_sched,
        )

        dstate0_cute = None
        if use_dstate0:
            dstate0_cute = from_dlpack(d_initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate_in_cute = None
        if use_dstate_in:
            dstate_in_cute = from_dlpack(d_final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        wi_cute = from_dlpack(work_items, assumed_align=16)
        wi_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        wc_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        sc_cute = None
        if dyn_sched:
            sc_cute = from_dlpack(sched_ctr, assumed_align=4).mark_layout_dynamic()

        tensormap_ws_cute = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()

        state_checkpoints_cute = from_dlpack(state_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=len(state_checkpoints.shape) - 1)
        initial_state_cute = (
            from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=len(initial_state.shape) - 1) if use_initial_state else None
        )
        a_log_cute = from_dlpack(a_log, assumed_align=4) if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=16) if dt_bias is not None else None
        dgate_cute = from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=len(dgate.shape) - 1)
        dbeta_cute = from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=len(dbeta.shape) - 1)
        dw_cute = from_dlpack(dw, assumed_align=16).mark_layout_dynamic(leading_dim=len(dw.shape) - 1)
        cache["compiled"] = cute.compile(
            host,
            cfg,
            state_checkpoints_cute,
            initial_state_cute,
            a_log_cute,
            dt_bias_cute,
            dgate_cute,
            dbeta_cute,
            dw_cute,
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            dstate0_cute,
            dstate_in_cute,
            wi_cute,
            wc_cute,
            sc_cute,
            tensormap_ws_cute,
            scale,
            cu_stream,
            options="--enable-tvm-ffi --opt-level 2",
        )

    if "prologue" not in cache:
        io_dtype = get_dtype(q.dtype)
        q_pl = from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        k_pl = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        v_pl = from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        gate_pl = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        do_pl = from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        beta_pl = from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        w_pl = from_dlpack(w, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dq_pl = from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dk_pl = from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dv_pl = from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dgate_pl = from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dwo_pl = from_dlpack(dw, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        dbo_pl = from_dlpack(dbeta, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        state_checkpoints_pl = from_dlpack(state_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        initial_state_pl = from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3) if use_initial_state else None
        cu_pl = from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic()
        ws_pl = from_dlpack(tensormap_workspace, assumed_align=128).mark_layout_dynamic()
        staging_pl = None
        if not order_gen:
            staging_pl = from_dlpack(work_item_scratch, assumed_align=16)
            staging_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_pl = from_dlpack(work_items, assumed_align=16)
        work_items_pl.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_pl = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        sched_all_pl = None
        if sched_all is not None:
            sched_all_pl = from_dlpack(sched_all, assumed_align=4).mark_layout_dynamic()
        cache["prologue"] = cute.compile(
            prologue,
            io_dtype,
            CFG.B_T,
            order_in_prologue,
            order_gen,
            sched_all is not None,
            q_pl,
            k_pl,
            v_pl,
            gate_pl,
            do_pl,
            beta_pl,
            w_pl,
            dq_pl,
            dk_pl,
            dv_pl,
            dgate_pl,
            dwo_pl,
            dbo_pl,
            state_checkpoints_pl,
            initial_state_pl,
            cu_pl,
            staging_pl,
            work_count_pl,
            work_items_pl,
            sched_all_pl,
            ws_pl,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["prologue"](
        q,
        k,
        v,
        gate,
        do,
        beta,
        w,
        dq,
        dk,
        dv,
        dgate,
        dw,
        dbeta,
        state_checkpoints,
        initial_state,
        cu_seqlens,
        work_item_scratch if not order_gen else None,
        work_count,
        work_items,
        sched_all,
        tensormap_workspace,
        cu_stream,
    )
    cache["compiled"](
        state_checkpoints,
        initial_state,
        a_log,
        dt_bias,
        dgate,
        dbeta,
        dw,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        scale,
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
    w,
    do,
    state_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    dw,
    cu_seqlens,
    initial_state,
    d_initial_state,
    d_final_state,
    work_items,
    work_count,
    sched_ctr,
    sched_all,
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
    cu_stream = cuda_driver.CUstream(int(stream))
    cache["prologue"](
        q,
        k,
        v,
        gate,
        do,
        beta,
        w,
        dq,
        dk,
        dv,
        dgate,
        dw,
        dbeta,
        state_checkpoints,
        initial_state,
        cu_seqlens,
        work_item_scratch,
        work_count,
        work_items,
        sched_all,
        tensormap_workspace,
        cu_stream,
    )
    cache["compiled"](
        state_checkpoints,
        initial_state,
        a_log,
        dt_bias,
        dgate,
        dbeta,
        dw,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        scale,
        cu_stream,
    )
