# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST SM100 D128 BF16 NVFP4 QAT dQ kernel (role-swapped twin of the dK/dV kernel).

Two CTAs collaborate on a 256-Q by 128-KV tile and loop over KV. Per KV tile the
leader issues three cta_group::2 MMAs: S = Q·Kᵀ, dP = dO·Vᵀ and dQ += dS·K, where
dS is recomputed from the same fake-quantized Q/K/V, LSE and delta as the dK/dV
kernel. Nothing is materialized: dQ accumulates in TMEM over all KV tiles and is
stored once per Q tile, so the result is bitwise deterministic and the only
workspace is the fake-quantized Q/K/V plus delta.

Softmax lanes are Q rows, so LSE and delta are per-lane scalars. dS keeps the
FP32 straight-through P (no probability fake quantization on this path) and
folds attn_scale before its BF16 store, exactly like the dK/dV kernel.

Layouts: Q/K/V BSHD (quantizer outputs), dO and dQ caller BHSD (TMA strides).
"""

from typing import Callable, Tuple

import cuda.bindings.driver as _cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from dataclasses import dataclass
from typing import NamedTuple

from cudnn.frost.tile_dsl.barrier import PipelineState, advance, cga_arrive, cga_wait, MBarrier, Producer, Scope, wait, arrive_expect_tx
from cudnn.frost.tile_dsl.scheduler import Sched, read_tile_id_arrive
from cudnn.frost.tile_dsl.mma import mma_ss
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile

from ._frost_kernel import _boot_tile, _decode_tile_payload

# ============================================================================
# Config — fixed: BF16, D128, dense, cga2, 12 warps.
# ============================================================================


@dataclass(frozen=True)
class DqCfg:
    TILE_Q: int = 128  # q rows per CTA (pair covers 256)
    TILE_KV: int = 128  # kv per iteration
    D: int = 128
    BPE: int = 2
    CTA_MMA: int = 2
    CGA_M: int = 2
    STAGES_KV: int = 2  # K / V / K_dq rings
    TILE_K_HW: int = 16
    SWZ_BYTES: int = 128
    SOFTMAX_WARPGROUPS: int = 2
    SOFTMAX_WG_WARPS: int = 4
    SCHEDULER_STAGES: int = 2
    # setmaxnreg: 8*(S-168) == 4*(168-R) around the compiled 168 for 12 warps.
    SOFTMAX_REGS: int = 216
    OTHER_REGS: int = 72
    TOTAL_WARPS: int = 12
    THREADS_PER_CTA: int = 12 * 32
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11
    SOFTMAX_LANES: int = 256  # 8 warps
    SOFTMAX_WG_LANES: int = 128


CFG = DqCfg()
assert 8 * (CFG.SOFTMAX_REGS - 168) == 4 * (168 - CFG.OTHER_REGS), "setmaxnreg inc/dec must balance around 168"
assert CFG.SOFTMAX_REGS % 8 == 0 and CFG.OTHER_REGS % 8 == 0

STORAGE_DTYPE = cutlass.BFloat16
OUT_STORAGE_DTYPE = cutlass.BFloat16
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2
CGA_SIZE = CFG.CGA_M
_LOG2E = 1.4426950408889634

# --- SMEM geometry (elements) ---------------------------------------------
_KV_PER_CTA = CFG.TILE_KV // CFG.CTA_MMA  # 64 — K/V N-split rows per CTA
_D_PER_CTA = CFG.D // CFG.CTA_MMA  # 64 — K_dq N-split cols per CTA
qBufferElems = CFG.TILE_Q * CFG.D  # 128 q × 128 d (A of S / dP)
kBufferElems = _KV_PER_CTA * CFG.D  # 64 kv × 128 d (B of S / dP)
kdqBufferElems = CFG.TILE_KV * _D_PER_CTA  # 128 kv × 64 d (B of dQ, BT)
dSBufferElems = CFG.TILE_Q * CFG.TILE_KV  # 128 q × 128 kv (A of dQ)
dQBufferElems = CFG.TILE_Q * CFG.D  # 128 q × 128 d (store staging, aliases Q)

qTmaBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
kdqTmaBytes = kdqBufferElems * CFG.BPE * CFG.CTA_MMA

GRANU = CFG.SWZ_BYTES // CFG.BPE  # 64 elems per 128 B swizzle row
TMA_D_ITERS = CFG.D // GRANU  # 2 sub-tiles along d
TMA_KV_ITERS = CFG.TILE_KV // GRANU  # 2 sub-tiles along kv (dS)
SMEM_LAYOUT_SWZ128 = 2
STRIDE_BYTE_OFFSET = 8 * CFG.SWZ_BYTES
LEADING_BYTE_OFFSET_BT = CFG.TILE_KV * CFG.SWZ_BYTES  # MN-major B: 64-col block stride
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
_SMX_CHUNK = CFG.TILE_KV // CFG.SOFTMAX_WARPGROUPS  # 64 kv cols per wg
_DQ_CHUNK = CFG.D // CFG.SOFTMAX_WARPGROUPS  # 64 d cols per wg
_Q_BLOCK_ROWS = CFG.TILE_Q * CFG.CTA_MMA  # 256 q per cluster tile


@dataclass(frozen=True)
class TmemLayout:
    TOTAL_COLS: int = 512
    S_OFF: int = 0
    dP_OFF: int = 128
    dQ_OFF: int = 256


LAYOUT = TmemLayout()

ONE_LANE = 1
MMA_COMMIT_ARRIVES = 1
SOFTMAX_LANES = CFG.SOFTMAX_LANES
SOFT_X_CTA_MMA = SOFTMAX_LANES * CFG.CTA_MMA  # 512
# leader: 8 softmax + MMA + TMALDG + TMASTG = 11; follower: 10 (quiet MMA) -> 21.
READ_TILE_ARRIVERS_TOT = 21


class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_do_full: object
    mb_do_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object
    mb_kdq_full: object
    mb_kdq_empty: object
    mb_s_acc_full: object
    mb_s_acc_empty: object
    mb_dp_full: object
    mb_dp_empty: object
    mb_ds_ready: object
    mb_ds_mma_done: object
    mb_dq_ready: object
    mb_dq_acc_empty: object
    mb_dq_stg_full: object
    mb_dq_stg_empty: object
    mb_tmem_dealloc: object


def _make_bars():
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    S = CFG.STAGES_KV
    return Bars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_do_full=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_empty=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(S), stages=S, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(S), stages=S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(S), stages=S, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(S), stages=S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_kdq_full=MBarrier(_alloc(S), stages=S, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_kdq_empty=MBarrier(_alloc(S), stages=S, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_s_acc_full=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        # S / dP slots are single-buffer; every softmax lane of BOTH CTAs frees them on the leader.
        mb_s_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dp_full=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dp_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_ds_ready=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_ds_mma_done=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dq_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=Producer.MMA_COMMIT),
        mb_dq_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_dq_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_dq_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFT_X_CTA_MMA, producer=Producer.THREAD),
    )


# ============================================================================
# Kernel entry
# ============================================================================


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],  # Q   (A of S; M-split 128 q × 128 d per CTA)
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],  # dO  (A of dP; caller BHSD)
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],  # K   (B of S; N-split 64 kv × 128 d per CTA)
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],  # V   (B of dP)
    tma_kdq_desc: cutlass.GridConstant[tmap.TensorMap],  # K   (B of dQ; d-split 128 kv × 64 d per CTA, BT)
    tma_dq_desc: cutlass.GridConstant[tmap.TensorMap],  # dQ  -> out [B,H,S_q,d] BF16
    lse_tensor: cute.Tensor,  # [B, H, S_q] FP32 natural log
    delta_tensor: cute.Tensor,  # [B, H, S_q] FP32 raw rowsum(O·dO)
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    n_qh_grid: cutlass.Int32,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- SMEM ---------------------------------------------------------------
    sQ_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_raw = cutlass.Array(STORAGE_DTYPE, qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sKdq_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kdqBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdS_raw = cutlass.Array(STORAGE_DTYPE, dSBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # dQ BF16 store staging aliases Q (dead once the tile's last MMA committed).
    sdQ_raw = cutlass.Array(sQ_raw.data_ptr(), dQBufferElems, dtype=OUT_STORAGE_DTYPE)

    sQ = SmemTile(
        desc_version=0,
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_D_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=CFG.TILE_Q * GRANU,
    )
    sdO = SmemTile(
        desc_version=0,
        base=sdO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_D_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=CFG.TILE_Q * GRANU,
    )
    sK = SmemTile(
        desc_version=0,
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_D_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=_KV_PER_CTA * GRANU,
    )
    sV = SmemTile(
        desc_version=0,
        base=sV_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_D_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=_KV_PER_CTA * GRANU,
    )
    # K dQ-view — B of dQ (BT=true, 128 kv × 64 d per CTA), same geometry as the dK kernel's Q dK-view.
    sKdq = SmemTile(
        desc_version=0,
        base=sKdq_raw,
        elems_per_stage=kdqBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_BT,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=1,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=CFG.TILE_KV * GRANU,
    )
    # dS [q × kv] — A of dQ (K-major: kv contiguous within 64-col slabs).
    sdS = SmemTile(
        desc_version=0,
        base=sdS_raw,
        elems_per_stage=dSBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_KV_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=CFG.TILE_Q * GRANU,
    )
    sdQ = SmemTile(
        desc_version=0,
        base=sdQ_raw,
        elems_per_stage=dQBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_D_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=CFG.TILE_Q * GRANU,
    )

    bars = _make_bars()
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    cta_id_x = cute.arch.block_idx_in_cluster()
    cta_in_pair = cta_id_x & cutlass.Int32(1)
    leader_cta_id = cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)
    partner_cta_id = cta_id_x ^ cutlass.Int32(1)
    is_leader = cta_in_pair == cutlass.Int32(0)

    if warp_idx == 0:
        if nvvm.elect_sync():
            bars.mb_q_full.init()
            bars.mb_q_empty.init()
            bars.mb_do_full.init()
            bars.mb_do_empty.init()
            for s in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[s].init()
                bars.mb_k_empty[s].init()
                bars.mb_v_full[s].init()
                bars.mb_v_empty[s].init()
                bars.mb_kdq_full[s].init()
                bars.mb_kdq_empty[s].init()
            # Leader-waited barriers count both CTAs' softmax lanes; the follower's copies are unused.
            LEADER_INIT = cutlass.Int32(arith.select(is_leader.ir_value(), cutlass.Int32(SOFT_X_CTA_MMA).ir_value(), cutlass.Int32(ONE_LANE).ir_value()))
            bars.mb_s_acc_full.init()
            bars.mb_s_acc_empty.init(override_count=LEADER_INIT)
            bars.mb_dp_full.init()
            bars.mb_dp_empty.init(override_count=LEADER_INIT)
            bars.mb_ds_ready.init(override_count=LEADER_INIT)
            bars.mb_ds_mma_done.init()
            bars.mb_dq_ready.init()
            bars.mb_dq_acc_empty.init(override_count=LEADER_INIT)
            bars.mb_dq_stg_full.init()
            bars.mb_dq_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    cga_arrive()
    cga_wait()

    mcast_mask = cutlass.Int32(3) << leader_cta_id
    tma_mcast_mask = cutlass.Int16(1) << cta_id_x
    is_cga_first_cta = cta_id_x == cutlass.Int32(0)

    if warp_idx < cutlass.Int32(CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            warp_idx,
            tmem_ptr_i32,
            bars,
            sched,
            sdS_raw,
            sdQ_raw,
            lse_tensor,
            delta_tensor,
            seqlen_q,
            seqlen_kv,
            attn_scale,
            attn_scale_log2e,
            head_base,
            cta_in_pair,
            leader_cta_id,
            partner_cta_id,
        )
    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if is_leader:
            _mma_warp(sQ, sdO, sK, sV, sKdq, sdS, tmem_ptr_i32, bars, sched, seqlen_q, seqlen_kv, mcast_mask)
        else:
            _mma_warp_quiet(tmem_ptr_i32, bars)
    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_kdq_desc.get_ptr())
        _tmaldg_warp(
            tma_q_desc,
            tma_do_desc,
            tma_k_desc,
            tma_v_desc,
            tma_kdq_desc,
            sQ,
            sdO,
            sK,
            sV,
            sKdq,
            bars,
            sched,
            seqlen_q,
            seqlen_kv,
            qh_per_kh,
            head_base,
            is_leader,
            cta_in_pair,
            tma_mcast_mask,
        )
    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dq_desc.get_ptr())
        _tmastg_warp(tma_dq_desc, sdQ, bars, sched, seqlen_q, seqlen_kv, cta_in_pair, head_base)
    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _scheduler_warp(sched, is_cga_first_cta)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ============================================================================
# Warp bodies
# ============================================================================


@cute.jit
def _n_kv_tiles(seqlen_kv):
    return seqlen_kv // cutlass.Int32(CFG.TILE_KV)


@cute.jit
def _softmax_warp_group(
    warp_idx,
    tmem_ptr_i32,
    bars,
    sched,
    sdS_raw,
    sdQ_raw,
    lse_tensor,
    delta_tensor,
    seqlen_q,
    seqlen_kv,
    attn_scale,
    attn_scale_log2e,
    head_base,
    cta_in_pair,
    leader_cta_id,
    partner_cta_id,
) -> None:
    """8 compute warps (2 wg x 4), lane = q row (128 per CTA), wg owns a 64-kv-col half.

    Per kv-iter: S -> P = exp2(scale·log2e·S - lse·log2e) ; dP -> dS = (scale·dP - scale·delta)·P
    -> BF16 -> sdS slab (wg = kv half) -> mb_ds_ready (leader).  S/dP slots are freed on the
    leader by every lane after the tmem load (mb_s_acc_empty / mb_dp_empty).
    Per q-tile: dQ TMEM -> BF16 -> sdQ (wg = d half) -> mb_dq_stg_full ; mb_dq_acc_empty.
    """
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    lse_arr = cutlass.make_array_view(lse_tensor)
    delta_arr = cutlass.make_array_view(delta_tensor)

    q_super_idx, head_idx, batch_idx = _boot_tile(sched)
    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw & cutlass.Int32(127)
    wg_id = warp_idx // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    kv_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()
    dp_full_state = PipelineState.start()
    ds_mma_done_state = PipelineState.start(phase=1)
    dq_ready_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        tmem_base = tmem_ptr_i32.load()
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        q_abs = q_super_idx * cutlass.Int32(_Q_BLOCK_ROWS) + cta_in_pair * cutlass.Int32(CFG.TILE_Q) + tid_in_wg
        lse_log2 = lse_arr[batch_idx, full_head, q_abs] * cutlass.Float32(_LOG2E)
        delta_scaled = delta_arr[batch_idx, full_head, q_abs] * attn_scale
        lse_vec = cutlass.Vector.from_elements(tuple(lse_log2 for _ in range(_SMX_CHUNK)), cutlass.Float32)
        delta_vec = cutlass.Vector.from_elements(tuple(delta_scaled for _ in range(_SMX_CHUNK)), cutlass.Float32)
        n_kv = _n_kv_tiles(seqlen_kv)

        for kv_iter in cutlass.range(0, n_kv, 1, unroll=1):
            # ---- softmax: S -> P (regs) ; free the S slot ----
            bars.mb_s_acc_full.wait(s_full_state.phase)
            s_full_state = advance(s_full_state, 1)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + kv_half_off, num_elems=_SMX_CHUNK, ld_num=64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.mb_s_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            chunk_P = cute.math.exp2(reg_S.vec * attn_scale_log2e - lse_vec, fastmath=True)

            # ---- dSoftmax: dP -> dS ; free the dP slot ----
            bars.mb_dp_full.wait(dp_full_state.phase)
            dp_full_state = advance(dp_full_state, 1)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + kv_half_off, num_elems=_SMX_CHUNK, ld_num=64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            bars.mb_dp_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)
            chunk_dS = (reg_dP.vec * attn_scale - delta_vec) * chunk_P
            chunk_dS_f16 = chunk_dS.to(STORAGE_DTYPE)

            # ---- dS -> SMEM slab (A of dQ); slot free once dQ[i-1] committed ----
            bars.mb_ds_mma_done.wait(ds_mma_done_state.phase)
            ds_mma_done_state = advance(ds_mma_done_state, 1)
            (sdS_raw.subview(wg_id * cutlass.Int32(CFG.TILE_Q * GRANU) + tid_in_wg * cutlass.Int32(GRANU))).data_ptr().store_swizzled(
                chunk_dS_f16, alignment=128, swizzle=P_SMEM_SWIZZLE
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ds_ready.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        # ---- dQ epilogue (per q-tile): TMEM -> BF16 -> sdQ (aliases Q) ----
        bars.mb_dq_ready.wait(dq_ready_state.phase)
        dq_ready_state = advance(dq_ready_state, 1)
        reg_dQ = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dQ_OFF) + wg_id * cutlass.Int32(_DQ_CHUNK), num_elems=_DQ_CHUNK, ld_num=64)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        dQ_bf16 = reg_dQ.vec.to(OUT_STORAGE_DTYPE)
        (sdQ_raw.subview(wg_id * cutlass.Int32(CFG.TILE_Q * GRANU) + tid_in_wg * cutlass.Int32(GRANU))).data_ptr().store_swizzled(
            dQ_bf16, alignment=128, swizzle=P_SMEM_SWIZZLE
        )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_dq_stg_full.arrive()
        bars.mb_dq_acc_empty.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        q_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.arrive()
    bars.mb_tmem_dealloc.arrive_on_peer(partner_cta_id)


@cute.jit
def _mma_warp(sQ, sdO, sK, sV, sKdq, sdS, tmem_ptr_i32, bars, sched, seqlen_q, seqlen_kv, mcast_mask) -> None:
    """Leader MMA warp: per kv-iter  S = Q·Kᵀ ; dP = dO·Vᵀ ; dQ += dS·K  (all cta_group::2)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=False)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    kdq_full_state = PipelineState.start()
    s_empty_state = PipelineState.start(phase=1)
    dp_empty_state = PipelineState.start(phase=1)
    ds_ready_state = PipelineState.start()
    dq_empty_state = PipelineState.start(phase=0)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))
    tmem_dQ = tmem_raw.subview(cutlass.Int32(LAYOUT.dQ_OFF))

    # S / dP: A = Q or dO (M = q, K-major), B = K or V (N = kv, K-major), K = d.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.TILE_KV, m_dim=CFG.TILE_Q * CFG.CTA_MMA, k_dim=1
    )
    qk_desc = MmaDesc(
        M=CFG.TILE_Q * CFG.CTA_MMA,
        N=CFG.TILE_KV,
        K=CFG.D,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # dQ: A = dS (M = q, K-major over kv), B = K dQ-view (N = d, MN-major), K = kv.
    idesc_dq = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=STORAGE_DTYPE, b_dtype=STORAGE_DTYPE, n_dim=CFG.D, m_dim=CFG.TILE_Q * CFG.CTA_MMA, a_major=0, b_major=1, k_dim=1
    )
    dq_desc = MmaDesc(
        M=CFG.TILE_Q * CFG.CTA_MMA,
        N=CFG.D,
        K=CFG.TILE_KV,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=True,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_dq,
        kind=MMA_KIND,
    )

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        n_kv = _n_kv_tiles(seqlen_kv)

        bars.mb_q_full.wait(q_full_state.phase)
        bars.mb_do_full.wait(do_full_state.phase)
        desc_Q = sQ[0].desc()
        desc_dO = sdO[0].desc()

        for kv_iter in cutlass.range(0, n_kv, 1, unroll=1):
            # ----- S = Q·K[i]ᵀ -----
            bars.mb_s_acc_empty.wait(s_empty_state.phase)
            s_empty_state = advance(s_empty_state, 1)
            bars.mb_k_full[k_full_state.idx].wait(k_full_state.phase)
            mma_ss(qk_desc, desc_Q, sK[k_full_state.idx].desc(), tmem_S, accumulate=False)
            elect_p = nvvm.elect_sync()
            bars.mb_s_acc_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[k_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            k_full_state = advance(k_full_state, CFG.STAGES_KV)

            # ----- dP = dO·V[i]ᵀ -----
            bars.mb_dp_empty.wait(dp_empty_state.phase)
            dp_empty_state = advance(dp_empty_state, 1)
            bars.mb_v_full[v_full_state.idx].wait(v_full_state.phase)
            mma_ss(qk_desc, desc_dO, sV[v_full_state.idx].desc(), tmem_dP, accumulate=False)
            elect_p = nvvm.elect_sync()
            bars.mb_dp_full.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[v_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            v_full_state = advance(v_full_state, CFG.STAGES_KV)

            # ----- dQ += dS[i]·K[i] -----
            bars.mb_ds_ready.wait(ds_ready_state.phase)
            ds_ready_state = advance(ds_ready_state, 1)
            bars.mb_kdq_full[kdq_full_state.idx].wait(kdq_full_state.phase)
            mma_ss(dq_desc, sdS[0].desc(), sKdq[kdq_full_state.idx].desc(), tmem_dQ, accumulate=(kv_iter > cutlass.Int32(0)))
            elect_p = nvvm.elect_sync()
            bars.mb_ds_mma_done.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_kdq_empty[kdq_full_state.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kdq_full_state = advance(kdq_full_state, CFG.STAGES_KV)

        # dQ complete -> epilogue; wait for the drain before the next tile's first MMA.
        elect_p = nvvm.elect_sync()
        bars.mb_dq_ready.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_q_empty.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        bars.mb_do_empty.arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        q_full_state = advance(q_full_state, 1)
        do_full_state = advance(do_full_state, 1)
        bars.mb_dq_acc_empty.wait(dq_empty_state.phase)
        dq_empty_state = advance(dq_empty_state, 1)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars) -> None:
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=False)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _tmastg_warp(tma_dq_desc, sdQ, bars, sched, seqlen_q, seqlen_kv, cta_in_pair, head_base) -> None:
    tma_dq = GmemTileTma(tma_dq_desc)
    q_super_idx, head_idx, batch_idx = _boot_tile(sched)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    dq_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        q_row_base = q_super_idx * cutlass.Int32(_Q_BLOCK_ROWS) + cta_in_pair * cutlass.Int32(CFG.TILE_Q)

        bars.mb_dq_stg_full.wait(dq_full_state.phase)
        dq_full_state = advance(dq_full_state, 1)
        tma_store_tile(sdQ[0], tma_dq(cutlass.Int32(0), head_idx + head_base, q_row_base, batch_idx))
        tma_store_commit()
        tma_store_wait()
        if nvvm.elect_sync():
            bars.mb_dq_stg_empty.arrive()

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        q_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _tmaldg_warp(
    tma_q_desc,
    tma_do_desc,
    tma_k_desc,
    tma_v_desc,
    tma_kdq_desc,
    sQ,
    sdO,
    sK,
    sV,
    sKdq,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    qh_per_kh,
    head_base,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
) -> None:
    tma_q = GmemTileTma(tma_q_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_v = GmemTileTma(tma_v_desc)
    tma_kdq = GmemTileTma(tma_kdq_desc)

    q_super_idx, head_idx, batch_idx = _boot_tile(sched)
    full_head = cute.arch.make_warp_uniform(head_idx + head_base)
    kv_head = cute.arch.make_warp_uniform(full_head // qh_per_kh)
    Q_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_Q)
    KV_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(_KV_PER_CTA)
    KDQ_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(_D_PER_CTA)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    kdq_empty_state = PipelineState.start(phase=1)
    # sdQ aliases Q: the previous tile's dQ store must finish before reloading Q.
    dq_storage_empty_state = PipelineState.start(phase=1)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        q_row_base = q_super_idx * cutlass.Int32(_Q_BLOCK_ROWS) + Q_ROW_OFFSET_PEER
        n_kv = _n_kv_tiles(seqlen_kv)

        # ---- Q + dO — one-shot per q-tile (M-split q, full d) ----
        bars.mb_dq_stg_empty.wait(dq_storage_empty_state.phase)
        dq_storage_empty_state = advance(dq_storage_empty_state, 1)
        bars.mb_q_empty.wait(q_empty_state.phase)
        q_empty_state = advance(q_empty_state, 1)
        if is_leader:
            if nvvm.elect_sync():
                bars.mb_q_full.arrive(n_bytes=qTmaBytes)
        tma_load_tile(
            sQ[0], tma_q(cutlass.Int32(0), full_head, q_row_base, batch_idx), bars.mb_q_full.smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mcast_mask
        )

        bars.mb_do_empty.wait(do_empty_state.phase)
        do_empty_state = advance(do_empty_state, 1)
        if is_leader:
            if nvvm.elect_sync():
                bars.mb_do_full.arrive(n_bytes=qTmaBytes)
        tma_load_tile(
            sdO[0], tma_do(cutlass.Int32(0), full_head, q_row_base, batch_idx), bars.mb_do_full.smem_ptr, cta_group=CFG.CTA_MMA, mcast_mask=tma_mcast_mask
        )

        # ---- K / V / K_dq — per kv-iter ----
        for kv_iter in cutlass.range(0, n_kv, 1, unroll=1):
            kv_row_base = kv_iter * cutlass.Int32(CFG.TILE_KV)

            bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_k_full[k_empty_state.idx].arrive(n_bytes=kTmaBytes)
            tma_load_tile(
                sK[k_empty_state.idx],
                tma_k(cutlass.Int32(0), kv_head, kv_row_base + KV_ROW_OFFSET_PEER, batch_idx),
                bars.mb_k_full[k_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            k_empty_state = advance(k_empty_state, CFG.STAGES_KV)

            bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_v_full[v_empty_state.idx].arrive(n_bytes=kTmaBytes)
            tma_load_tile(
                sV[v_empty_state.idx],
                tma_v(cutlass.Int32(0), kv_head, kv_row_base + KV_ROW_OFFSET_PEER, batch_idx),
                bars.mb_v_full[v_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            v_empty_state = advance(v_empty_state, CFG.STAGES_KV)

            bars.mb_kdq_empty[kdq_empty_state.idx].wait(kdq_empty_state.phase)
            if is_leader:
                if nvvm.elect_sync():
                    bars.mb_kdq_full[kdq_empty_state.idx].arrive(n_bytes=kdqTmaBytes)
            tma_load_tile(
                sKdq[kdq_empty_state.idx],
                tma_kdq(KDQ_COL_OFFSET_PEER, kv_head, kv_row_base, batch_idx),
                bars.mb_kdq_full[kdq_empty_state.idx].smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )
            kdq_empty_state = advance(kdq_empty_state, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        q_super_idx, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    # P15: drain the cga2 _empty rings outside the persistent loop.
    bars.mb_q_empty.wait(q_empty_state.phase)
    bars.mb_do_empty.wait(do_empty_state.phase)
    for _s in cutlass.range_constexpr(CFG.STAGES_KV):
        bars.mb_k_empty[k_empty_state.idx].wait(k_empty_state.phase)
        k_empty_state = advance(k_empty_state, CFG.STAGES_KV)
        bars.mb_v_empty[v_empty_state.idx].wait(v_empty_state.phase)
        v_empty_state = advance(v_empty_state, CFG.STAGES_KV)
        bars.mb_kdq_empty[kdq_empty_state.idx].wait(kdq_empty_state.phase)
        kdq_empty_state = advance(kdq_empty_state, CFG.STAGES_KV)


@cute.jit
def _scheduler_warp(sched, is_cga_first_cta) -> None:
    """Persistent tile scheduler (try_cancel protocol), no stats prefetch."""
    state = PipelineState.start()
    is_valid = cutlass.Int32(1)
    while is_valid > cutlass.Int32(0):
        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync() and is_cga_first_cta:
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        validity = (sched.tile_id_smem.subview(state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid = validity & cutlass.Int32(1)
        state = advance(state, CFG.SCHEDULER_STAGES)


# ============================================================================
# Host
# ============================================================================


@cute.jit
def _host(
    q_tensor: cute.Tensor,  # [B, S_q, H, d]  BF16 (quantizer output, BSHD)
    do_tensor: cute.Tensor,  # [B, S_q, H, d]  BF16 logical view over caller BHSD
    k_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 (BSHD)
    v_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 (BSHD)
    dq_tensor: cute.Tensor,  # [B, S_q, H, d]  BF16 logical view over caller BHSD
    lse_tensor: cute.Tensor,  # [B, H, S_q] FP32
    delta_tensor: cute.Tensor,  # [B, H, S_q] FP32 raw
    problem_size: Tuple[int, int, int, int, int, int],
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    stream: _cuda_driver.CUstream,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK = problem_size
    stride_order = (3, 2, 1, 0)
    swz = tmap.TensorMapSwizzle.s128b
    l2 = tmap.TensorMapL2Promotion.l2_128b
    q_box = (1, CFG.TILE_Q, 1, GRANU)  # 128 q rows × 64 d per sub-tile
    kv_box = (1, _KV_PER_CTA, 1, GRANU)  # 64 kv rows × 64 d
    kdq_box = (1, CFG.TILE_KV, 1, GRANU)  # 128 kv rows × 64 d (per-CTA d half)
    tma_q_desc = tmap.create_tensor_map_tiled_from_view(q_tensor, box_dims=q_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(do_tensor, box_dims=q_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(k_tensor, box_dims=kv_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(v_tensor, box_dims=kv_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)
    tma_kdq_desc = tmap.create_tensor_map_tiled_from_view(k_tensor, box_dims=kdq_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)
    tma_dq_desc = tmap.create_tensor_map_tiled_from_view(dq_tensor, box_dims=q_box, stride_order=stride_order, swizzle=swz, l2_promotion=l2)

    q_blocks = (SQ + _Q_BLOCK_ROWS - 1) // _Q_BLOCK_ROWS
    grid_shape = (q_blocks * CFG.CGA_M, QH_CHUNK, B)
    _kernel(
        tma_q_desc,
        tma_do_desc,
        tma_k_desc,
        tma_v_desc,
        tma_kdq_desc,
        tma_dq_desc,
        lse_tensor,
        delta_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        cutlass.Int32(QH_CHUNK),
    ).launch(grid=grid_shape, block=[CFG.THREADS_PER_CTA, 1, 1], cluster=(CFG.CGA_M, 1, 1), stream=stream)


def compile(b: int, qh: int, kh: int, sq: int, skv: int, qh_chunk: int = 0) -> Callable:
    """Compile the dQ kernel for fixed plan-time shapes (same contract as the dK/dV kernel)."""
    if b != 1 or qh != kh or sq != skv or sq <= 0 or sq % 256:
        raise ValueError("SM100 QAT requires B=1, MHA, and equal positive 256-aligned lengths")
    if qh_chunk == 0:
        qh_chunk = qh
    if qh_chunk <= 0 or qh <= 0 or qh % qh_chunk:
        raise ValueError("head chunk must be a positive divisor of H")
    d = CFG.D
    fake_q = cute.runtime.make_fake_compact_tensor(STORAGE_DTYPE, (b, sq, qh, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fake_k = cute.runtime.make_fake_compact_tensor(STORAGE_DTYPE, (b, skv, kh, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    fake_v = cute.runtime.make_fake_compact_tensor(STORAGE_DTYPE, (b, skv, kh, d), stride_order=(3, 2, 1, 0), assumed_align=16)
    bhsd = (sq * qh * d, d, sq * d, 1)
    fake_do = cute.runtime.make_fake_tensor(STORAGE_DTYPE, (b, sq, qh, d), bhsd, assumed_align=16)
    fake_dq = cute.runtime.make_fake_tensor(OUT_STORAGE_DTYPE, (b, sq, qh, d), bhsd, assumed_align=16)
    fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_delta = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    return cute.compile(
        _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dq,
        fake_lse,
        fake_delta,
        (b, qh, kh, sq, skv, qh_chunk),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _cuda_driver.CUstream(0),
        options="--enable-tvm-ffi",
    )
