# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""pre-upstream DSL qwen prefill SDPA kernel (d_qk=d_v=256, FP8 E4M3/E5M2).

Qwen pipeline: TILES_Q=1, single softmax wg, Q∪O SMEM alias with two
extra barriers (mb_q_o_alias TMASTG→TMA bootstrap-armed; mb_tmastg_go
TMA→TMASTG empty-mainloop cover).  Q·K(i+1) → S·V(i) lookahead MMA
stream with two parity S_acc TMEM slots; P_cast aliases each slot's
tail.  Bootstrap-only-lo-parity correction arrive on mb_bmm2_ready.

THD / varlen (``CFG.THD_VARLEN=1``) supported (E4M3/E5M2) via the shared
mechanism (FP8 element-addressed, no block-scale SF; same path as f16); dense
byte-identical.  Public-API fp8 THD glue is a follow-up (THD-aware quantizer).
"""

import os
import sys
from functools import lru_cache
from typing import Callable, Optional, Tuple


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.primitives import vote_sync, VoteSync
from cutlass._mlir.dialects import arith

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental.cuda import tensor_map as tmap
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from dataclasses import dataclass
from typing import NamedTuple

# Config comes from the FROST template loader, NOT an env var: the pre-upstream
# kernels picked a flavor with an env var + a sdpa_config_<flavor> module, which the
# FROST engine contract forbids ("no environment variables for configuration" --
# parameters travel as typed dataclasses). The loader injects
# FROST_TEMPLATE_PARAMS before this body runs; the default keeps a plain
# `import` usable as a standalone driver.
from cudnn.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256

PARAMS: TemplateParams = globals().get("FROST_TEMPLATE_PARAMS", TemplateParams())
CFG, _TMA = make_cfg_d256(PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE
# decision point, wired into every construction below rather than repeated as a
# per-tile literal.  A version-0 descriptor's ``start_address`` is 14 bits = a
# 256 KiB window; Rubin raises the per-CTA SMEM cap to 327 KiB, so an operand
# buffer at or above 256 KiB wraps to offset 0 and the MMA multiplies whatever
# sits at the bottom of SMEM.  This flavor's buffers all stay below the line.
#
# Do NOT re-literal this at a call site: the d512 MXFP8 sibling shipped NaN on
# 100% of cells because its scale-factor tiles were declared UNDER a comment
# claiming "every operand tile here carries desc_version=1" -- without the
# kwarg.  A single constant makes that class of drift impossible, and
# test_sm107_descriptor_version_matches_the_smem_budget asserts it.
DESC_VERSION: int = 0
Cfg = type(CFG)
TMA_QK_ITERS = _TMA.QK_ITERS
TMA_VO_ITERS = _TMA.VO_ITERS
TMA_QK_GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
TMA_VO_GRANU_ELEMS = _TMA.VO_GRANU_ELEMS

# Rubin's dense-FP8 MMA runs K=64 per instruction, and every
# ``Tcgen05InstrDesc.build`` site in this body hardcodes the matching idesc
# ``k_dim=1``.  ``config_sm107.tile_k_hw()`` derives the 64; this guard is the
# tripwire for the pairing, which is arch-OPPOSITE (Blackwell wants k_dim=0 with
# TILE_K_HW=32) and fails SILENTLY -- a mismatch scrambles rows of the
# accumulator rather than raising (rules/mma-tma-matrix.md S1).
if CFG.TILE_K_HW_BMM1 != 64 or CFG.TILE_K_HW_BMM2 != 64:
    raise ValueError(f"{__name__}: this body's idesc k_dim=1 requires TILE_K_HW=64 on Rubin; " f"got BMM1={CFG.TILE_K_HW_BMM1} BMM2={CFG.TILE_K_HW_BMM2}")

if CFG.DTYPE_QKV not in (0, 1):
    raise ValueError(f"prefill_sdpa_d256_fp8: DTYPE_QKV must be 0 (E4M3) or 1 (E5M2); " f"got {CFG.DTYPE_QKV}.  Use prefill_sdpa_d256_f16.py for BF16/FP16.")

TMA_O_GRANU_ELEMS_HOST = CFG.O_SWZ_BYTES // CFG.BPE_O
TMA_O_ITERS_HOST = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES

from cudnn.frost.tile_dsl.barrier import (
    PipelineState,
    advance,
    cga_arrive,
    cga_wait,
    # `wait` (free fn) — still used for sched.mb_* (Sched not in Bars).
    wait,
)
from cudnn.frost.tile_dsl.scheduler import (
    Sched,
    scheduler_warp_loop,
    scheduler_warp_loop_persistent,
    read_tile_id_arrive,
    read_clc_payload,
    SCHED_NATURAL,
    SCHED_LPT,
    SCHED_LPT_L2,
)
from cudnn.frost.tile_dsl.pointwise import (
    tmem_load_max_reduction_tile,
    row_reduction_pair,
    row_max_reduction,
    vec_scale_pair,
)
from cudnn.frost.tile_dsl.regtile import RegTile, vec_concat
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts_step
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma, tma_slice_runtime_desc
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.mask import (
    apply_mask_chunk,
    MASK_NONE,
    MASK_PADDED,
    MASK_CAUSAL,
    MASK_SWA,
)

if CFG.DTYPE_QKV == 0:
    STORAGE_DTYPE = cutlass.Float8E4M3FN
    P_STORAGE_DTYPE = cutlass.Float8E4M3FN
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4
elif CFG.DTYPE_QKV == 1:
    STORAGE_DTYPE = cutlass.Float8E5M2
    P_STORAGE_DTYPE = cutlass.Float8E5M2
    MMA_KIND = nvvm.Tcgen05MMAKind.F8F6F4

# DTYPE_O independent of DTYPE_QKV — FP8 input may write BF16/FP16 O so a
# downstream consumer skips a dequant.  BPE_O ∈ {1, 2}; epilogue already casts
# via .to(OUT_STORAGE_DTYPE) + store_swizzled (dtype-generic).  The Q∪O SMEM
# alias is byte-sized for max(Q@BPE, O@BPE_O) so BF16 O doesn't overflow it.
if CFG.DTYPE_O == 0:
    OUT_STORAGE_DTYPE = cutlass.Float8E4M3FN
elif CFG.DTYPE_O == 1:
    OUT_STORAGE_DTYPE = cutlass.Float8E5M2
elif CFG.DTYPE_O == 2:
    OUT_STORAGE_DTYPE = cutlass.BFloat16
elif CFG.DTYPE_O == 3:
    OUT_STORAGE_DTYPE = cutlass.Float16
else:
    raise ValueError(f"prefill_sdpa_d256_fp8: DTYPE_O={CFG.DTYPE_O} not supported " f"(expected 0=E4M3 / 1=E5M2 / 2=BF16 / 3=FP16)")


from cudnn.sdpa.fwd.kernels._common_blackwell import (
    D256Bars as Bars,
    KvLoopBounds,
    make_d256_bars,
    compute_kv_loop_bounds,
    lpt_tile_coords,
    make_sdpa_helpers,
)

CGA_SIZE = CFG.CGA_M * CFG.CGA_N

CTA_GROUP_KIND = nvvm.CTAGroup.CTA_2 if CFG.CTA_MMA == 2 else nvvm.CTAGroup.CTA_1

qBufferElems = CFG.TILE_M * CFG.TILE_K
kBufferElems = CFG.TILE_N * CFG.TILE_K // CFG.CTA_MMA
vBufferElems = CFG.TILE_O * CFG.TILE_N // CFG.CTA_MMA
oBufferElems = CFG.TILE_M * CFG.TILE_O
qoAliasBytes = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)

qTmaTransactionBytes = qBufferElems * CFG.BPE * CFG.CTA_MMA
kTmaTransactionBytes = kBufferElems * CFG.BPE * CFG.CTA_MMA
vTmaTransactionBytes = vBufferElems * CFG.BPE * CFG.CTA_MMA

N_O_CHUNKS = (CFG.TILE_O * CFG.BPE_O + 127) // 128

CGA_TILE_M = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA

# lpt_q_tiles_in_cga_units=True is REQUIRED under any non-NATURAL scheduler:
# the LPT linearization needs q_tiles in CGA units (n_q_supers // CTA_MMA).
# Without it the decode walks a row range CTA_MMA times too large, no tile is
# ever claimed, and the kernel writes NOTHING -- cosine 0.0000 at every shape.
# It is a NO-OP under SCHED_NATURAL (that decode branch never reads q_tiles),
# so restoring it cannot change today's shipped path. Every SM100 kernel passes
# it; the SM107 port dropped it on most flavors. Verified on d256 f16.
_sdpa_h = make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)
_decode_initial = _sdpa_h.decode_initial
_decode_payload = _sdpa_h.decode_payload
_bounds_for_tile = _sdpa_h.bounds_for_tile
_resolve_seqlen_kv = _sdpa_h.resolve_seqlen_kv
# Q-side per-sequence length.  Under THD `seqlen_q` is the PACKED TOTAL, so
# every BOTTOM_RIGHT diagonal needs the sequence's own S_q_b instead.
_resolve_seqlen_q = _sdpa_h.resolve_seqlen_q

# THD / varlen — shared helpers (FP8 is element-addressed like f16; per-tensor
# dequant scalars, no block-scale SF).  Gated by CFG.THD_VARLEN (folds out).
from cudnn.sdpa.fwd.kernels.thd_helpers import build_thd_meta_o_descs_kernel as _build_thd_meta_o_descs_kernel, TENSOR_MAP_QWORDS, THD_SETUP_THREADS

_TENSOR_MAP_QWORDS = TENSOR_MAP_QWORDS
_dispatch_decode_initial = _sdpa_h.dispatch_decode_initial
_dispatch_decode_payload = _sdpa_h.dispatch_decode_payload
_thd_tma_offsets = _sdpa_h.thd_tma_offsets


@dataclass(frozen=True)
class KernelTmemLayout:
    """Column offsets for the qwen 2-parity-slot SDPA pipeline (FP8 d=256).

    P aliases the TAIL of each parity's S_acc slot.  Stats parked at col
    544 (DSL-only — outside S_acc/O range) vs C++ col 512.
    """

    TOTAL_COLS: int = 576

    S_ACC_EVEN_OFF: int = 0
    S_ACC_ODD_OFF: int = 128

    P_EVEN_OFF: int = 96
    P_ODD_OFF: int = 224

    O_OFF: int = 256

    STATS_OFF: int = 544


LAYOUT = KernelTmemLayout()


_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
SMEM_LAYOUT_Q = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_K = _SWZ_ENUM[CFG.K_SWZ_BYTES]
SMEM_LAYOUT_V = _SWZ_ENUM[CFG.V_SWZ_BYTES]
SMEM_LAYOUT_O = _SWZ_ENUM[CFG.O_SWZ_BYTES]
SMEM_LAYOUT_QKO = SMEM_LAYOUT_Q

_O_SWZ_B = {128: 3, 64: 2, 32: 1}[CFG.O_SWZ_BYTES]
_O_SMEM_SWIZZLE = cutlass.Swizzle(_O_SWZ_B, 4, 3)

LEADING_BYTE_OFFSET_QK = 0
STRIDE_BYTE_OFFSET_QK = 8 * CFG.Q_SWZ_BYTES

# leading_byte_offset = 0 when (TILE_O/CTA_MMA)/8 <= 8 else TILE_N*V_SWZ_BYTES
_CORE_MATRIX_ROWS = 8
_V_PC_COLS = CFG.TILE_O // CFG.CTA_MMA
LEADING_BYTE_OFFSET_PV = 0 if (_V_PC_COLS // _CORE_MATRIX_ROWS) <= 8 else CFG.TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_PV = 8 * CFG.V_SWZ_BYTES

# Derived, never a literal: NUM_KPHASES_PV must track the SAME k-step size
# the MmaDesc below is built with, or the BMM2 k-loop and the descriptor's
# num_k_steps disagree and half of V's K is silently dropped.
_MMA_K_FP8 = CFG.TILE_K_HW_BMM2
NUM_KPHASES_PV = CFG.TILE_N // _MMA_K_FP8
NUM_KPHASES_PV_PER_CHUNK = NUM_KPHASES_PV // CFG.N_BMM2_CHUNKS


# === Kernel entry ===


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_o_desc: cutlass.GridConstant[tmap.TensorMap],
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_q_supers: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    # FROST's quantized ABI: the four 1-element fp32 scales arrive as DEVICE
    # TENSORS and the amax of O is an OUTPUT.  The pre-upstream contract pre-folded the scales on
    # the host and produced no amax -- that gap is what this closes.  Template:
    # the shipped sm107/prefill_d128_fp8.py sibling (same lineage).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
) -> None:

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # Device-scale fold: every thread loads the same four 1-element fp32 scales
    # (identical addresses -> L2 broadcast, negligible) and folds them into the
    # softmax scale and the output scale, exactly as the shipped d128 sibling.
    _dsc_q = cutlass.Float32(cutlass.make_array_view(descale_q_t)[0])
    _dsc_k = cutlass.Float32(cutlass.make_array_view(descale_k_t)[0])
    _dsc_v = cutlass.Float32(cutlass.make_array_view(descale_v_t)[0])
    _scl_o = cutlass.Float32(cutlass.make_array_view(scale_o_t)[0])
    scale_softmax_log2 = scale_softmax_log2 * _dsc_q * _dsc_k
    o_scale_fused = o_scale_fused * _dsc_v * _scl_o

    # Q∪O alias byte-sized to max(Q@BPE, O@BPE_O) — STORAGE_DTYPE is FP8 (1 B)
    # so the element count == byte count.  BF16/FP16 O (BPE_O=2) would overflow
    # a qBufferElems-only allocation, so size for the larger of the two views.
    _QO_ALIAS_ELEMS = max(qBufferElems * CFG.BPE, oBufferElems * CFG.BPE_O)
    sQO_raw = cutlass.Array(STORAGE_DTYPE, _QO_ALIAS_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    sK_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * kBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_KV * vBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    # OUT_STORAGE_DTYPE view of the Q∪O backing — element offsets in the
    # epilogue store then advance in BPE_O units (BF16/FP16 O).  No-op recast
    # for FP8 output (same dtype).
    sO_raw = cutlass.Array(sQO_raw.data_ptr(), shape=oBufferElems, dtype=OUT_STORAGE_DTYPE)

    # sQ / sO share Q∪O backing; separate SmemTile wrappers for per-site TMA params
    sQ = SmemTile(
        base=sQO_raw,
        elems_per_stage=qBufferElems,
        stages=1,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=kBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_QK,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QKO,
        tma_loads_per_tile=TMA_QK_ITERS,
        tma_granu_elems=TMA_QK_GRANU_ELEMS,
        tma_subtile_stride_elems=(CFG.TILE_N // CFG.CTA_MMA) * TMA_QK_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=vBufferElems,
        stages=CFG.STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_PV,
        stride_byte_offset=STRIDE_BYTE_OFFSET_PV,
        layout=SMEM_LAYOUT_V,
        tma_loads_per_tile=TMA_VO_ITERS // CFG.CTA_MMA,
        tma_granu_elems=TMA_VO_GRANU_ELEMS,
        tma_subtile_stride_elems=CFG.TILE_N * TMA_VO_GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=oBufferElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_O,
        tma_loads_per_tile=TMA_O_ITERS_HOST,
        tma_granu_elems=TMA_O_GRANU_ELEMS_HOST,
        tma_subtile_stride_elems=CFG.TILE_M * TMA_O_GRANU_ELEMS_HOST,
        desc_version=DESC_VERSION,
    )

    bars = make_d256_bars(CFG, N_O_CHUNKS=N_O_CHUNKS)

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

    READ_TILE_ARRIVERS_TOTAL = CFG.READ_TILE_ARRIVERS

    if warp_idx == 0:
        if nvvm.elect_sync():
            # Init counts baked into make_d256_bars(...); kernel calls .init()
            # per stage.  range_constexpr keeps loop vars Python int.
            bars.mb_q_full.init()
            bars.mb_q_o_alias.init()
            bars.mb_tmastg_go.init()
            for p in cutlass.range_constexpr(2):
                bars.mb_bmm1_done[p].init()
                bars.mb_bmm2_done[p].init()
                for c in cutlass.range_constexpr(CFG.N_BMM2_CHUNKS):
                    bars.mb_bmm2_ready[p * CFG.N_BMM2_CHUNKS + c].init()
            bars.mb_stat_full.init()
            bars.mb_stat_empty.init()
            for chunk in cutlass.range_constexpr(N_O_CHUNKS):
                bars.mb_o_full[chunk].init()
            bars.mb_o_empty.init()
            for ks in cutlass.range_constexpr(CFG.STAGES_KV):
                bars.mb_k_full[ks].init()
                bars.mb_k_empty[ks].init()
                bars.mb_v_full[ks].init()
                bars.mb_v_empty[ks].init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOTAL)
            bars.mb_empty_mainloop.init()
            bars.mb_tmem_dealloc.init()

            # bootstrap mb_q_o_alias so first TMA wait passes
            bars.mb_q_o_alias.arrive()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # P4 cluster fence after mbar init for cga2 cross-CTA arrives
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        cga_arrive()
        cga_wait()

    cta_id_x = cute.arch.block_idx_in_cluster() if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    cta_in_pair = (cta_id_x & cutlass.Int32(1)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    leader_cta_id = (cta_id_x & cutlass.Int32(~1 & 0xFFFFFFFF)) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    mcast_mask = (cutlass.Int32(3) << leader_cta_id) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int32(0)
    tma_mcast_mask = (cutlass.Int16(1) << cta_in_pair) if cutlass.const_expr(CFG.CTA_MMA == 2) else cutlass.Int16(0)
    is_leader = cta_in_pair == cutlass.Int32(0)

    # Warp layout: 0..3 softmax wg, 4..7 correction, 8 MMA, 9 TMALDG, 10 TMASTG, 11 sched.
    if warp_idx >= CFG.SOFTMAX_WG0_BASE and warp_idx < CFG.SOFTMAX_WG0_BASE + CFG.SOFTMAX_WG_WARPS:
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            scale_log2=scale_softmax_log2,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
        )

    elif warp_idx >= CFG.CORR_WARP_BASE and warp_idx < CFG.CORR_WARP_BASE + CFG.CORRECTION_WARPS:
        nvvm.setmaxregister(CFG.CORRECTION_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _correction_warp_group(
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            sO=sO,
            tmem_ptr_i32=tmem_ptr_i32,
            tidx=tidx,
            bars=bars,
            sched=sched,
            lse_tensor=lse_tensor,
            sinks_tensor=sinks_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            leader_cta_id=leader_cta_id,
            cta_in_pair=cta_in_pair,
            cta_id_x=cta_id_x,
            o_scale_fused=o_scale_fused,
            amax_o_tensor=amax_o_tensor,
        )

    elif warp_idx == CFG.MMA_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        if cutlass.const_expr(CFG.CTA_MMA == 2):
            if is_leader:
                _mma_warp_group(
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    sQ=sQ,
                    sK=sK,
                    sV=sV,
                    tmem_ptr_i32=tmem_ptr_i32,
                    bars=bars,
                    sched=sched,
                    seq_kv_lens_tensor=seq_kv_lens_tensor,
                    n_q_supers=n_q_supers,
                    n_qh=n_qh,
                    n_batch=n_batch,
                    mcast_mask=mcast_mask,
                    cta_in_pair=cta_in_pair,
                )
            else:
                _mma_warp_quiet(tmem_ptr_i32, bars)
        else:
            _mma_warp_group(
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                sQ=sQ,
                sK=sK,
                sV=sV,
                tmem_ptr_i32=tmem_ptr_i32,
                bars=bars,
                sched=sched,
                seq_kv_lens_tensor=seq_kv_lens_tensor,
                n_q_supers=n_q_supers,
                n_qh=n_qh,
                n_batch=n_batch,
                mcast_mask=mcast_mask,
                cta_in_pair=cta_in_pair,
            )

    elif warp_idx == CFG.TMALDG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            o_desc_words=o_desc_words,
            sQ=sQ,
            sK=sK,
            sV=sV,
            bars=bars,
            sched=sched,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            qh_per_kh=qh_per_kh,
            is_leader=is_leader,
            cta_in_pair=cta_in_pair,
            tma_mcast_mask=tma_mcast_mask,
        )

    elif warp_idx == CFG.TMASTG_WARP_ID:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _tmastg_warp_group(
            tma_o_desc=tma_o_desc,
            sO=sO,
            bars=bars,
            sched=sched,
            n_q_supers=n_q_supers,
            n_qh=n_qh,
            n_batch=n_batch,
            cta_in_pair=cta_in_pair,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            o_desc_words=o_desc_words,
        )

    else:  # warp_idx == CFG.SCHED_WARP_ID
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: persistent grid + the device claim counter in the 4B+4
            # metadata's last two slots.  The dense CLC path hands out tiles
            # against a grid that is the persistent cluster count, not the work
            # list, so every tile->unit mapping is off without this.
            scheduler_warp_loop_persistent(
                sched,
                CFG.SCHEDULER_STAGES,
                is_cga_first_cta,
                seq_kv_lens_tensor,
                cutlass.Int32(4) * n_batch + cutlass.Int32(3),
                cutlass.Int32(4) * n_batch + cutlass.Int32(2),
                CGA_SIZE,
                CFG.CGA_M,
            )
        else:
            scheduler_warp_loop(sched, CFG.SCHEDULER_STAGES, is_cga_first_cta, CGA_SIZE)


# === TMA-LDG warp group ===


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_k_desc,
    tma_v_desc,
    o_desc_words,
    sQ,
    sK,
    sV,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    qh_per_kh,
    is_leader,
    cta_in_pair,
    tma_mcast_mask,
):
    """Qwen TMA-LDG warp: single Q load gated on mb_q_o_alias, fires
    mb_tmastg_go each tile end (covers empty-mainloop branch)."""

    q_o_alias_phase = cutlass.Int32(0)
    kv_state = PipelineState.start(phase=1)  # bootstrap pre-armed at phase 1

    tma_q = GmemTileTma(tma_q_desc)
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: K/V ride the setup kernel's packed-total-CLAMPED runtime
        # descriptors (o_desc_words slots n_batch+1 / n_batch+2).  The plan-time
        # descriptors describe the buffer's CAPACITY, so the last sequence's
        # tile-tail would read caller-owned bytes past cu_k[B]; a NaN there wipes
        # the tile through BMM2 (0 * NaN == NaN).  Clamped, those rows are
        # TMA-OOB and land as EXACT ZEROS.
        _k_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(1)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        _v_rt_ptr = (o_desc_words.iterator.raw_ptr() + (n_batch + cutlass.Int32(2)) * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tma_k = lambda *coords: tma_slice_runtime_desc(_k_rt_ptr, *coords)  # noqa: E731
        tma_v = lambda *coords: tma_slice_runtime_desc(_v_rt_ptr, *coords)  # noqa: E731
    else:
        tma_k = GmemTileTma(tma_k_desc)
        tma_v = GmemTileTma(tma_v_desc)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
    q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
    q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    K_ROW_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_N // CFG.CTA_MMA)
    V_COL_OFFSET_PEER = cta_in_pair * cutlass.Int32(CFG.TILE_O // CFG.CTA_MMA)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        # Run unconditionally — covers both normal and empty-mainloop branches
        bars.mb_q_o_alias.wait(q_o_alias_phase)
        q_o_alias_phase = q_o_alias_phase ^ cutlass.Int32(1)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            # Empty mainloop — skip Q/K/V loads; mb_tmastg_go still fires at end-of-iter
            pass
        else:
            # P9 — leader-only arrive_expect_tx under cga2 (TMA bytes routed to leader mbar)
            if cutlass.const_expr(CFG.CTA_MMA == 2):
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
            else:
                bars.mb_q_full.arrive(n_bytes=qTmaTransactionBytes, pred=nvvm.elect_sync())
            tma_load_tile(
                sQ[0],
                tma_q(cutlass.Int32(0), head_idx, q_row_base + q_seq_off, tma_batch),
                bars.mb_q_full.smem_ptr,
                cta_group=CFG.CTA_MMA,
                mcast_mask=tma_mcast_mask,
            )

            for kv_loop in cutlass.range(kv_left, kv_right, 1, unroll=1):
                kv_row_base = kv_loop * cutlass.Int32(CFG.TILE_N)

                bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_k_full[kv_state.idx].arrive(n_bytes=kTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sK[kv_state.idx],
                    tma_k(cutlass.Int32(0), kv_head_idx, kv_row_base + K_ROW_OFFSET_PEER + kv_seq_off, tma_batch),
                    bars.mb_k_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                # V split along d_v under cga2 via V_COL_OFFSET_PEER
                bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
                if cutlass.const_expr(CFG.CTA_MMA == 2):
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=is_leader & nvvm.elect_sync())
                else:
                    bars.mb_v_full[kv_state.idx].arrive(n_bytes=vTmaTransactionBytes, pred=nvvm.elect_sync())
                tma_load_tile(
                    sV[kv_state.idx],
                    tma_v(V_COL_OFFSET_PEER, kv_head_idx, kv_row_base + kv_seq_off, tma_batch),
                    bars.mb_v_full[kv_state.idx].smem_ptr,
                    cta_group=CFG.CTA_MMA,
                    mcast_mask=tma_mcast_mask,
                )

                kv_state = advance(kv_state, CFG.STAGES_KV)

        # mb_tmastg_go fires on both normal and empty-mainloop tiles
        if nvvm.elect_sync():
            bars.mb_tmastg_go.arrive()
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        kv_head_idx = cute.arch.make_warp_uniform(head_idx // qh_per_kh)
        q_row_base = cute.arch.make_warp_uniform(q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M))
        q_seq_off, kv_seq_off, tma_batch = _thd_tma_offsets(seq_kv_lens_tensor, batch_idx, n_batch)
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        if cutlass.const_expr(CFG.MASK_FLAGS != 0):
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right

    # cga2 drain — trailing empty mbar arrives from leader's multicast commits
    if cutlass.const_expr(CFG.CTA_MMA == 2):
        for _ks in cutlass.range_constexpr(CFG.STAGES_KV):
            bars.mb_k_empty[kv_state.idx].wait(kv_state.phase)
            bars.mb_v_empty[kv_state.idx].wait(kv_state.phase)
            kv_state = advance(kv_state, CFG.STAGES_KV)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


# === TMA-STG warp group ===


@cute.jit
def _tmastg_warp_group(
    tma_o_desc,
    sO,
    bars,
    sched,
    n_q_supers,
    n_qh,
    n_batch,
    cta_in_pair,
    seq_kv_lens_tensor,
    o_desc_words,
):
    """Qwen TMA-STG warp: waits mb_tmastg_go before per-chunk mb_o_full
    (covers empty-mainloop); fires mb_q_o_alias at end-of-tile to release
    Q∪O for the next persistent tile."""

    tmastg_go_phase = cutlass.Int32(0)
    o_full_phase = cutlass.Int32(0)

    tma_o = GmemTileTma(tma_o_desc)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        bars.mb_tmastg_go.wait(tmastg_go_phase)
        tmastg_go_phase = tmastg_go_phase ^ cutlass.Int32(1)

        for chunk in cutlass.range_constexpr(N_O_CHUNKS):
            bars.mb_o_full[chunk].wait(o_full_phase)

        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)

        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: store through this batch's pre-built descriptor (seq extent
            # = S_q_b → box past S_q_b OOB-clipped).  q_row_coord seq-local; batch→0.
            # DEAD unit (batch_idx == n_batch, the over-launched persistent grid's
            # sentinel): no O rows exist and descriptor slot n_batch is never
            # built, so skip only the store -- the barrier protocol still runs.
            if batch_idx < n_batch:
                o_desc_ptr = (o_desc_words.iterator.raw_ptr() + batch_idx * cutlass.Int32(_TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
                o_slice = tma_slice_runtime_desc(o_desc_ptr, cutlass.Int32(0), head_idx, q_row_coord, cutlass.Int32(0))
                tma_store_tile(sO[0], o_slice)
        else:
            tma_store_tile(
                sO[0],
                tma_o(cutlass.Int32(0), head_idx, q_row_coord, batch_idx),
            )
        tma_store_commit()
        tma_store_wait(0)

        bars.mb_o_empty.arrive()
        if nvvm.elect_sync():
            bars.mb_q_o_alias.arrive()

        o_full_phase = o_full_phase ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


# === MMA warp group (+ quiet warp for cga2 non-leader) ===


@cute.jit
def _mma_warp_quiet(tmem_ptr_i32, bars):
    """Quiet MMA-warp body for cga2 non-leader: alloc TMEM, no MMA, dealloc."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _mma_warp_group(
    seqlen_q,
    seqlen_kv,
    sQ,
    sK,
    sV,
    tmem_ptr_i32,
    bars,
    sched,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    mcast_mask,
    cta_in_pair,
):
    """Qwen MMA warp: Q*K(i+1) → S*V(i) lookahead stream.
    Prologue Q*K(lo); mainloop kv=lo..hi-2 runs Q*K(kv+1) then chunked
    S*V(kv) gated on bmm2_ready; epilogue S*V(hi-1).
    """
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND, is_exclusive=True)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    nvvm.barrier_cta_arrive(2, 32 * (CFG.CORRECTION_WARPS + 1))

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_N,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        k_dim=1,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=CFG.TILE_O,
        m_dim=CFG.TILE_M * CFG.CTA_MMA,
        b_major=1,
        k_dim=1,
    )
    bmm1_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_N,
        K=CFG.TILE_K,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM1,
        btranspose=False,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    bmm2_desc = MmaDesc(
        M=CFG.TILE_M * CFG.CTA_MMA,
        N=CFG.TILE_O,
        K=CFG.TILE_N,
        bpe_a=CFG.BPE,
        bpe_b=CFG.BPE,
        tile_k_hw=CFG.TILE_K_HW_BMM2,
        btranspose=True,
        k_subtile=CFG.V_SWZ_BYTES // CFG.BPE,
        cta_group=CFG.CTA_MMA,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    desc_Q = sQ[0].desc()

    if cutlass.const_expr(CFG.MASK_FLAGS == 0):
        kv_left = cutlass.Int32(0)
        kv_right = seqlen_kv // cutlass.Int32(CFG.TILE_N)
    else:
        q_super_idx, _hd, batch_idx = _dispatch_decode_initial(
            sched.bidx_init,
            sched.bidy_init,
            sched.bidz_init,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds_init = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
        kv_left = bounds_init.left
        kv_right = bounds_init.right

    q_full_phase = cutlass.Int32(0)
    kv_state_K = PipelineState.start(phase=0)
    kv_state_V = PipelineState.start(phase=0)
    # bit p = next-wait phase for that parity
    bmm2_ready_phase_pair = cutlass.Int32(0)
    empty_mainloop_phase = cutlass.Int32(0)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if cutlass.const_expr(CFG.MASK_FLAGS != 0) and (kv_right <= kv_left):
            bars.mb_empty_mainloop.wait(empty_mainloop_phase)
            empty_mainloop_phase = empty_mainloop_phase ^ cutlass.Int32(1)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[0].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
        else:
            bars.mb_q_full.wait(q_full_phase)
            q_full_phase = q_full_phase ^ cutlass.Int32(1)

            # Prologue: Q*K(lo) → S_acc[lo&1]
            lo_parity_runtime = kv_left & cutlass.Int32(1)
            parity_lo_is_even = lo_parity_runtime == cutlass.Int32(0)
            tmem_S_acc_lo_addr = cutlass.Int32(
                arith.select(
                    parity_lo_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                )
            )

            bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
            desc_K = sK[kv_state_K.idx].desc()
            mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_lo_addr)))
            elect_p = nvvm.elect_sync()
            bars.mb_bmm1_done[lo_parity_runtime].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

            # Mainloop kv=lo..hi-2: Q*K(kv+1) then S*V(kv)
            k_per_chunk = NUM_KPHASES_PV_PER_CHUNK

            for kv_loop in cutlass.range(kv_left, kv_right - cutlass.Int32(1), 1, unroll=1):
                parity_cur_rt = kv_loop & cutlass.Int32(1)
                parity_next_rt = (kv_loop + cutlass.Int32(1)) & cutlass.Int32(1)
                cur_is_even = parity_cur_rt == cutlass.Int32(0)
                next_is_even = parity_next_rt == cutlass.Int32(0)

                tmem_S_acc_next_addr = cutlass.Int32(
                    arith.select(
                        next_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value(),
                    )
                )
                tmem_P_cur_addr = cutlass.Int32(
                    arith.select(
                        cur_is_even.ir_value(),
                        cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                        cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                    )
                )
                bmm2_ready_phase_cur = (bmm2_ready_phase_pair >> parity_cur_rt) & cutlass.Int32(1)

                # Q*K(kv+1) → S_acc[parity_next]
                bars.mb_k_full[kv_state_K.idx].wait(kv_state_K.phase)
                desc_K = sK[kv_state_K.idx].desc()
                mma_ss(bmm1_desc, desc_Q, desc_K, (tmem_raw.subview(tmem_S_acc_next_addr)))
                elect_p = nvvm.elect_sync()
                bars.mb_bmm1_done[parity_next_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_k_empty[kv_state_K.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                kv_state_K = advance(kv_state_K, CFG.STAGES_KV)

                # S*V(kv) — chunked k-loop with bmm2_ready gates
                bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
                desc_V = sV[kv_state_V.idx].desc()

                # scaleC=False on iter lo (first k-step overwrites O), else accumulate
                scaleC = cutlass.Boolean(kv_loop != kv_left)
                accum_b2 = scaleC
                for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                    if k % k_per_chunk == 0:
                        chunk_id = k // k_per_chunk
                        bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_cur)
                    mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_cur_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                    accum_b2 = cutlass.Boolean(True)
                elect_p = nvvm.elect_sync()
                bars.mb_bmm2_done[parity_cur_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
                bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_cur_rt)
                kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

            # Epilogue: S*V(hi-1) → O
            kv_last = kv_right - cutlass.Int32(1)
            parity_last_rt = kv_last & cutlass.Int32(1)
            last_is_even = parity_last_rt == cutlass.Int32(0)
            tmem_P_last_addr = cutlass.Int32(
                arith.select(
                    last_is_even.ir_value(),
                    cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(),
                    cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value(),
                )
            )
            bmm2_ready_phase_last = (bmm2_ready_phase_pair >> parity_last_rt) & cutlass.Int32(1)

            bars.mb_v_full[kv_state_V.idx].wait(kv_state_V.phase)
            desc_V = sV[kv_state_V.idx].desc()
            n_kv_eff = kv_right - kv_left
            scaleC_epi = cutlass.Boolean(n_kv_eff != cutlass.Int32(1))
            accum_b2 = scaleC_epi
            for k in cutlass.range_constexpr(NUM_KPHASES_PV):
                if k % k_per_chunk == 0:
                    chunk_id = k // k_per_chunk
                    bars.mb_bmm2_ready[parity_last_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS) + cutlass.Int32(chunk_id)].wait(bmm2_ready_phase_last)
                mma_ts_step(bmm2_desc, (tmem_raw.subview(tmem_P_last_addr)), desc_V, (tmem_raw.subview(cutlass.Int32(LAYOUT.O_OFF))), k, accum_b2)
                accum_b2 = cutlass.Boolean(True)
            elect_p = nvvm.elect_sync()
            bars.mb_bmm2_done[parity_last_rt].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bars.mb_v_empty[kv_state_V.idx].arrive(mcast_mask=mcast_mask, cta_group=CFG.CTA_MMA, pred=elect_p)
            bmm2_ready_phase_pair = bmm2_ready_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)
            kv_state_V = advance(kv_state_V, CFG.STAGES_KV)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        if cutlass.const_expr(CFG.MASK_FLAGS == 0):
            _nq, _nh, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            is_valid_tile = nxt_v & cutlass.Int32(1)
        else:
            nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
            nxt_q = cute.arch.make_warp_uniform(nxt_q)
            nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
            nxt_v = cute.arch.make_warp_uniform(nxt_v)
            q_super_idx, _hd, batch_idx = _dispatch_decode_payload(
                nxt_q,
                nxt_hb,
                cta_in_pair,
                n_q_supers,
                n_qh,
                n_batch,
                seq_kv_lens_tensor,
            )
            is_valid_tile = nxt_v & cutlass.Int32(1)
            eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
            eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
            bounds_next = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)
            kv_left = bounds_next.left
            kv_right = bounds_next.right
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


# === Softmax warp group (single wg — qwen TILES_Q=1) ===


@cute.jit
def _softmax_warp_group(
    seqlen_q,
    seqlen_kv,
    scale_log2: cutlass.Float32,
    tmem_ptr_i32,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
):
    """Qwen softmax warp: single wg, parity-keyed (kv&1) S_acc/P selectors,
    publishes (total_max, total_sum) in end-of-tile epilogue."""
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    bmm1_done_phase_pair = cutlass.Int32(0)  # bit p = next-wait phase for parity p
    stat_empty_phase = cutlass.Int32(1)  # bootstrap pre-armed at phase 1
    epilogue_state = cutlass.Int32(1)

    NEG_INF = cutlass.Float32(-3.4028235e38)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)

    tid_in_wg = cute.arch.thread_idx()[0] - cutlass.Int32(CFG.SOFTMAX_WG0_BASE * 32)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        total_max = NEG_INF
        total_sum = cutlass.Vector.from_elements(
            (cutlass.Float32(0.0), cutlass.Float32(0.0)),
            cutlass.Float32,
        )
        q_row_coord = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M)
        q_abs = q_row_coord + tid_in_wg

        bars.mb_o_empty.wait(epilogue_state)
        bars.mb_stat_empty.wait(stat_empty_phase)
        stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        # Body inlined per-segment so the DSL tracer can dispatch through
        # cutlass.range without hitting the closure check.
        CHUNK = 64
        P_COLS_PER_CHUNK = CHUNK // 4  # fp8 packed 4:1 into FP32 cells
        N_CHUNKS = CFG.N_BMM2_CHUNKS  # 2 at TILE_N=128, 1 at TILE_N=64
        RESCALE_THRESHOLD = cutlass.Float32(CFG.RESCALE_THRESHOLD)

        if cutlass.const_expr(CFG.MASK_FLAGS == MASK_NONE):
            for kv_loop in cutlass.range(bounds.left, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)

                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)

                # Fast path: fused TMEM load + HW row-max
                reg_S_tile, current_max_unscaled = tmem_load_max_reduction_tile(
                    s_addr_base,
                    num_elems=CFG.TILE_N,
                )
                reg_S = RegTile(reg_S_tile.vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()

                reg_S = reg_S * scale_log2 - new_total_max

                # Chunk 0 (always emitted); chunk 1 N_CHUNKS==2 only (TILE_N=128)
                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_pack = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_pack)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_pack = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_pack)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair

                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
        else:
            for kv_loop in cutlass.range(bounds.left, bounds.unmasked_lo, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                chunks_S = [
                    apply_mask_chunk(
                        raw_chunks[c],
                        q_abs,
                        kv_col_base + cutlass.Int32(c * CHUNK),
                        eff_seqlen_kv,
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=CHUNK,
                        bottom_right=CFG.BOTTOM_RIGHT,
                        causal_diag=causal_diag,
                        window_right=CFG.WINDOW_RIGHT,
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_pack = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_pack)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_pack = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_pack)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_lo, bounds.unmasked_hi, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                reg_S_tile, current_max_unscaled = tmem_load_max_reduction_tile(
                    s_addr_base,
                    num_elems=CFG.TILE_N,
                )
                reg_S = RegTile(reg_S_tile.vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_pack = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_pack)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_pack = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_pack)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)
            for kv_loop in cutlass.range(bounds.unmasked_hi, bounds.right, 1, unroll=1):
                parity_rt = kv_loop & cutlass.Int32(1)
                parity_is_even = parity_rt == cutlass.Int32(0)
                s_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.S_ACC_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.S_ACC_ODD_OFF).ir_value())
                )
                p_off_rt = cutlass.Int32(
                    arith.select(parity_is_even.ir_value(), cutlass.Int32(LAYOUT.P_EVEN_OFF).ir_value(), cutlass.Int32(LAYOUT.P_ODD_OFF).ir_value())
                )
                bmm1_phase = (bmm1_done_phase_pair >> parity_rt) & cutlass.Int32(1)
                bars.mb_bmm1_done[parity_rt].wait(bmm1_phase)
                bmm1_done_phase_pair = bmm1_done_phase_pair ^ (cutlass.Int32(1) << parity_rt)
                tmem_base = tmem_ptr_i32.load()
                s_addr_base = tmem_base + s_off_rt
                p_addr_base = tmem_base + p_off_rt
                stats_addr = tmem_base + cutlass.Int32(LAYOUT.STATS_OFF)
                kv_col_base = kv_loop * cutlass.Int32(CFG.TILE_N)
                raw_chunks = [
                    nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(s_addr_base + cutlass.Int32(c * CHUNK), cutlass.Float32), num=CHUNK) for c in range(N_CHUNKS)
                ]
                # Bottom-right causal: runtime SKV-SQ diagonal offset (folds out when
                # CFG.BOTTOM_RIGHT is 0 — top-left masking is unchanged).
                causal_diag = eff_seqlen_kv - eff_seqlen_q if cutlass.const_expr(CFG.BOTTOM_RIGHT) else None
                chunks_S = [
                    apply_mask_chunk(
                        raw_chunks[c],
                        q_abs,
                        kv_col_base + cutlass.Int32(c * CHUNK),
                        eff_seqlen_kv,
                        CFG.WINDOW_LEFT,
                        CFG.MASK_FLAGS,
                        N=CHUNK,
                        bottom_right=CFG.BOTTOM_RIGHT,
                        causal_diag=causal_diag,
                        window_right=CFG.WINDOW_RIGHT,
                    )
                    for c in range(N_CHUNKS)
                ]
                chunks_max = [row_max_reduction(chunks_S[c]) for c in range(N_CHUNKS)]
                reg_S_vec = vec_concat(chunks_S)
                current_max_unscaled = chunks_max[0]
                for m in chunks_max[1:]:
                    current_max_unscaled = cute.math.max(current_max_unscaled, m)
                reg_S = RegTile(reg_S_vec, size=CFG.TILE_N)
                current_max = current_max_unscaled * scale_log2

                old_total_max = total_max
                is_first = total_max == NEG_INF
                update_cond = is_first | ((current_max - total_max) > RESCALE_THRESHOLD)
                total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
                exp_input = cutlass.Float32(arith.select(is_first.ir_value(), NEG_INF.ir_value(), (old_total_max - total_max).ir_value()))
                alpha = cute.math.exp2(exp_input, fastmath=True)
                new_total_max = total_max
                alpha_vec = cutlass.Vector.from_elements((alpha,), cutlass.Float32)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr, cutlass.Float32), alpha_vec)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_stat_full.arrive()
                reg_S = reg_S * scale_log2 - new_total_max

                chunk_S_0 = reg_S[0:CHUNK].vec
                chunk_P_0 = cute.math.exp2(chunk_S_0, fastmath=True)
                hoisted_sum = row_reduction_pair(chunk_P_0)
                chunk_P_0_pack = chunk_P_0.to(STORAGE_DTYPE)
                nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base, cutlass.Float32), chunk_P_0_pack)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(0)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                deferred_P_1 = None
                if cutlass.const_expr(N_CHUNKS == 2):
                    chunk_S_1 = reg_S[CHUNK : 2 * CHUNK].vec
                    deferred_P_1 = cute.math.exp2(chunk_S_1, fastmath=True)
                    chunk_P_1_pack = deferred_P_1.to(STORAGE_DTYPE)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(p_addr_base + cutlass.Int32(P_COLS_PER_CHUNK), cutlass.Float32), chunk_P_1_pack)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
                    bars.mb_bmm2_ready[parity_rt * cutlass.Int32(N_CHUNKS) + cutlass.Int32(1)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

                new_p_sum_pair = hoisted_sum
                if cutlass.const_expr(N_CHUNKS == 2):
                    new_p_sum_pair = new_p_sum_pair + row_reduction_pair(deferred_P_1)
                alpha_pair = cutlass.Vector.from_elements((alpha, alpha), cutlass.Float32)
                total_sum = total_sum * alpha_pair + new_p_sum_pair
                bars.mb_stat_empty.wait(stat_empty_phase)
                stat_empty_phase = stat_empty_phase ^ cutlass.Int32(1)

        # End-of-tile: publish final (total_max, total_sum)
        total_sum_scalar = total_sum[0] + total_sum[1]
        stats_addr_epi = tmem_ptr_i32.load() + cutlass.Int32(LAYOUT.STATS_OFF)
        stats_vec_epi = cutlass.Vector.from_elements((total_max, total_sum_scalar), cutlass.Float32)
        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32), stats_vec_epi)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
        bars.mb_stat_full.arrive()

        # LSE write deferred to correction warp epilogue (qwen-specific)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        nxt_q = cute.arch.make_warp_uniform(nxt_q)
        nxt_hb = cute.arch.make_warp_uniform(nxt_hb)
        nxt_v = cute.arch.make_warp_uniform(nxt_v)
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)


# === Correction warp group ===


@cute.jit
def _correction_warp_group(
    seqlen_q,
    seqlen_kv,
    sO,
    tmem_ptr_i32,
    tidx,
    bars,
    sched,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor,
    n_q_supers,
    n_qh,
    n_batch,
    leader_cta_id,
    cta_in_pair,
    cta_id_x,
    o_scale_fused,
    amax_o_tensor,
):
    """Qwen correction warp: bootstrap-only-lo-parity bmm2_ready arrive,
    per-parity bmm2_done_phase_pair, LSE+sink fold lives here (not softmax),
    P14 catch-up flip on bmm2_done_phase_pair after the epilogue wait."""
    nvvm.barrier_cta_sync(barrier_id=2, thread_count=32 * (CFG.CORRECTION_WARPS + 1))

    tid_raw = cute.arch.thread_idx()[0]
    tid_in_wg = tid_raw - cutlass.Int32(CFG.CORR_WARP_BASE * 32)

    bmm2_done_phase_pair = cutlass.Int32(0)  # bit p = next-wait phase for parity p
    stat_mbar_state = cutlass.Int32(0)
    # bootstrap pre-armed at phase 1 so first wait passes immediately
    epilogue_state = cutlass.Int32(1)

    q_super_idx, head_idx, batch_idx = _dispatch_decode_initial(
        sched.bidx_init,
        sched.bidy_init,
        sched.bidz_init,
        cta_in_pair,
        n_q_supers,
        n_qh,
        n_batch,
        seq_kv_lens_tensor,
    )
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()

    eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
    eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
    bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)

    O_CHUNK = 16
    N_CHUNKS_O = CFG.TILE_O // O_CHUNK
    TMA_O_ITERS_LOCAL = (CFG.TILE_O * CFG.BPE_O) // CFG.O_SWZ_BYTES
    D_BLOCK_SIZE = CFG.TILE_O // TMA_O_ITERS_LOCAL
    TMA_O_GRANU_ELEMS_LOCAL = CFG.TILE_M * D_BLOCK_SIZE

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)

        if bounds.right > bounds.left:
            # Bootstrap arrive on ONE parity (qwen-specific — NOT both)
            lo_parity_rt = bounds.left & cutlass.Int32(1)
            bars.mb_bmm2_ready[lo_parity_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            # Iter-0 stat_full consume (no rescale — softmax(lo) just signaled)
            bars.mb_stat_full.wait(stat_mbar_state)
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)
        else:
            # Empty-mainloop: MMA waits mb_empty_mainloop then fires bmm2_done[0]
            bars.mb_empty_mainloop.arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

        for kv_loop in cutlass.range(bounds.left + cutlass.Int32(1), bounds.right, 1, unroll=1):
            parity_prev_rt = (kv_loop - cutlass.Int32(1)) & cutlass.Int32(1)
            parity_cur_rt = kv_loop & cutlass.Int32(1)
            tmem_base_iter = tmem_ptr_i32.load()

            bars.mb_stat_full.wait(stat_mbar_state)

            stats_addr = tmem_base_iter + cutlass.Int32(LAYOUT.STATS_OFF)
            stats_vec = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            alpha = stats_vec[0]

            alpha_is_one = alpha == cutlass.Float32(1.0)
            all_alpha_one = vote_sync(0xFFFFFFFF, alpha_is_one, VoteSync.ALL)

            bars.mb_stat_empty.arrive()

            bmm2_done_phase_prev = (bmm2_done_phase_pair >> parity_prev_rt) & cutlass.Int32(1)
            bars.mb_bmm2_done[parity_prev_rt].wait(bmm2_done_phase_prev)
            bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_prev_rt)

            if ~all_alpha_one:
                for chunk_idx in cutlass.range_constexpr(N_CHUNKS_O):
                    o_addr = tmem_base_iter + cutlass.Int32(LAYOUT.O_OFF + chunk_idx * O_CHUNK)
                    o_chunk = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                        num=O_CHUNK,
                    )
                    o_scaled = vec_scale_pair(o_chunk, alpha, O_CHUNK)
                    nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(o_addr, cutlass.Float32), o_scaled)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)

            bars.mb_bmm2_ready[parity_cur_rt * cutlass.Int32(CFG.N_BMM2_CHUNKS)].arrive(leader_cta_id=leader_cta_id, cta_group=CFG.CTA_MMA)

            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        # Epilogue: always run (stores garbage on empty-mainloop — TMASTG ignores)
        tmem_base_epi = tmem_ptr_i32.load()

        # Pre-declare for DSL if-staging — names used after conditional must
        # be bound on every path
        total_max_scaled = cutlass.Float32(0.0)
        total_sum = cutlass.Float32(0.0)
        if bounds.right > bounds.left:
            bars.mb_stat_full.wait(stat_mbar_state)
            stats_addr_epi = tmem_base_epi + cutlass.Int32(LAYOUT.STATS_OFF)
            stats_vec_epi = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(stats_addr_epi, cutlass.Float32),
                num=2,
            )
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            total_max_scaled = stats_vec_epi[0]
            total_sum = stats_vec_epi[1]
            bars.mb_stat_empty.arrive()
            stat_mbar_state = stat_mbar_state ^ cutlass.Int32(1)

        # Sink fold + LSE compute (qwen: corr writes LSE)
        LN2 = cutlass.Float32(0.6931471805599453)
        total_max_nat = total_max_scaled * LN2
        lse_val = cutlass.Float32(0.0)
        inv_sum = cutlass.Float32(0.0)
        if cutlass.const_expr(CFG.HAS_SINK):
            sinks_arr = cutlass.make_array_view(sinks_tensor)
            sink_logit = sinks_arr[head_idx]
            new_max = cute.math.max(total_max_nat, sink_logit)
            scale = cute.math.exp(total_max_nat - new_max, fastmath=True)
            new_sum = total_sum * scale + cute.math.exp(sink_logit - new_max, fastmath=True)
            lse_val = new_max + cute.math.log(new_sum, fastmath=True)
            inv_sum = (scale * o_scale_fused) / new_sum
        else:
            lse_val = total_max_nat + cute.math.log(cute.math.max(total_sum, cutlass.Float32(1e-30)), fastmath=True)
            inv_sum = o_scale_fused / cute.math.max(total_sum, cutlass.Float32(1e-30))

        # --- empty KV range (zero-length sequence under the padding mask) ---
        # bounds.right <= bounds.left means the mainloop never ran, so
        # total_max/total_sum are still 0 and the 1e-30 denominator floor above
        # turns LSE into log(1e-30) = -69.08 and inv_sum into o_scale/1e-30,
        # which overflows to +inf.  O is then (TMEM residue) * inf.
        #
        # The residue is genuinely NaN, not merely stale: the suite poisons TMEM
        # before execute on purpose (poison_tmem_before_execute), so O must be
        # zeroed with a SELECT and never with a multiply by zero -- NaN * 0 is
        # NaN (rules/frost-gotchas.md, the empty-reduction-axis row).
        _kv_empty = bounds.right <= bounds.left
        if cutlass.const_expr(not CFG.HAS_SINK):
            # With a sink the row still has mass (the sink logit itself), and the
            # branch above already yields LSE = sink_logit correctly because
            # total_sum == 0 kills its other term.  Without one, an empty row has
            # no mass at all and LSE is -inf.
            lse_val = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(float("-inf")).ir_value(), lse_val.ir_value()))

        q_row_global = q_super_idx * cutlass.Int32(CFG.TILES_Q * CFG.TILE_M) + tid_in_wg
        if cutlass.const_expr(CFG.THD_VARLEN):
            # THD: sequence-local row; LSE packed [1,QH,T] → [0, head, cu_q[b]+local].
            _cu = cutlass.make_array_view(seq_kv_lens_tensor)
            _cu_q_b = cutlass.Int32(_cu[n_batch + batch_idx])
            _s_q_b = cutlass.Int32(_cu[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < _s_q_b:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    # Written in the CALLER's layout, picked by the STATIC rank compile()
                    # baked in: token-major rank-2 [T, QH] (the DEFAULT) or head-major
                    # rank-3 [1, QH, head_stride].  Serving only the rank-3 arm
                    # transposes every LSE on the common path.
                    if cutlass.const_expr(len(lse_tensor.shape) == 2):
                        lse_row = lse_arr[_cu_q_b + q_row_global, :]
                        lse_row[head_idx] = lse_val
                    else:
                        lse_row = lse_arr[cutlass.Int32(0), head_idx, :]
                        lse_row[_cu_q_b + q_row_global] = lse_val
        else:
            if cutlass.const_expr(lse_tensor is not None):
                if q_row_global < seqlen_q:
                    lse_arr = cutlass.make_array_view(lse_tensor)
                    lse_row = lse_arr[batch_idx, head_idx, :]
                    lse_row[q_row_global] = lse_val

        parity_last_rt = cutlass.Int32(0)
        if bounds.right > bounds.left:
            parity_last_rt = (bounds.right - cutlass.Int32(1)) & cutlass.Int32(1)
        bmm2_done_phase_last = (bmm2_done_phase_pair >> parity_last_rt) & cutlass.Int32(1)
        bars.mb_bmm2_done[parity_last_rt].wait(bmm2_done_phase_last)
        # P14 catch-up flip — bmm2_done_phase_pair ^= (1u<<parity_last) AFTER epilogue wait
        bmm2_done_phase_pair = bmm2_done_phase_pair ^ (cutlass.Int32(1) << parity_last_rt)

        # Epilogue store: block granularity = 64 elems (4 × O_CHUNK) so we
        # can fire mb_o_full[block/2] every 2 blocks
        O_EPI_BLK = 64
        N_BLOCKS_EPI = CFG.TILE_O // O_EPI_BLK
        CHUNKS_PER_BLK = O_EPI_BLK // O_CHUNK

        sO_base = sO[0].base
        epi_o_full_block_idx = 0

        # amax_o = max over VALID rows of |o| (the fp32 pre-cast output).  A
        # padded/dead row holds garbage, so the atomic is gated the same way
        # the LSE write above is -- an ungated max would silently inflate the
        # graph's Amax_O output.
        _amax_o_ptr = Pointer(amax_o_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
        _amax_o_local = cutlass.Float32(0.0)
        # Row validity gates BOTH the Stats write and the amax atomic.  Under THD
        # `seqlen_q` is the PACKED TOTAL, so comparing against it lets rows that
        # belong to a LATER sequence -- and every row of a dead unit from the
        # over-launched persistent grid -- count as valid.  atomicMax only grows,
        # so those rows silently inflate the graph's Amax_O while O itself stays
        # correct (sdpa-invariants.md S5).  Use the sequence's own S_q_b.
        if cutlass.const_expr(CFG.THD_VARLEN):
            _cu_rv = cutlass.make_array_view(seq_kv_lens_tensor)
            _cu_q_b_rv = cutlass.Int32(_cu_rv[n_batch + batch_idx])
            _s_q_b_rv = cutlass.Int32(_cu_rv[n_batch + batch_idx + cutlass.Int32(1)]) - _cu_q_b_rv
            _row_valid = (q_row_global < _s_q_b_rv) & (batch_idx < n_batch)
        else:
            _row_valid = q_row_global < seqlen_q

        for block_idx in cutlass.range_constexpr(N_BLOCKS_EPI):
            for sub in cutlass.range_constexpr(CHUNKS_PER_BLK):
                chunk_idx_total = block_idx * CHUNKS_PER_BLK + sub
                o_addr = tmem_base_epi + cutlass.Int32(LAYOUT.O_OFF + chunk_idx_total * O_CHUNK)
                o_chunk = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(o_addr, cutlass.Float32),
                    num=O_CHUNK,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                o_scaled = o_chunk * inv_sum
                # SELECT the zero for an empty KV range -- see _kv_empty above.
                # This also keeps the NaN out of the amax fold, which would
                # otherwise poison the graph's Amax_O for every other row.
                _o_elems = []
                for _i in cutlass.range_constexpr(O_CHUNK):
                    _e = cutlass.Float32(arith.select(_kv_empty.ir_value(), cutlass.Float32(0.0).ir_value(), o_scaled[_i].ir_value()))
                    _amax_o_local = cute.math.max(_amax_o_local, cute.math.max(_e, -_e))
                    _o_elems.append(_e)
                o_out = cutlass.Vector.from_elements(tuple(_o_elems), cutlass.Float32).to(OUT_STORAGE_DTYPE)

                col_offset_const = (chunk_idx_total * O_CHUNK) % D_BLOCK_SIZE
                block_offset_const = ((chunk_idx_total * O_CHUNK) // D_BLOCK_SIZE) * TMA_O_GRANU_ELEMS_LOCAL
                smem_offset = cutlass.Int32(block_offset_const + col_offset_const) + tid_in_wg * cutlass.Int32(D_BLOCK_SIZE)
                smem_ptr = sO_base.subview(smem_offset).data_ptr()

                if block_idx == 0 and sub == 0:
                    bars.mb_o_empty.wait(epilogue_state)
                smem_ptr.store_swizzled(o_out, alignment=64, swizzle=_O_SMEM_SWIZZLE)

            # Fire one mb_o_full per TMA-O store chunk.  There are N_O_CHUNKS
            # chunks across N_BLOCKS_EPI epilogue blocks; BF16/FP16 O doubles
            # the chunk count (BPE_O=2) so _BLOCKS_PER_OCHUNK drops 2→1 and we
            # fire every block instead of every other one — keeps the TMA-STG
            # consumer's N_O_CHUNKS waits balanced (FP8 path unchanged).
            _BLOCKS_PER_OCHUNK = N_BLOCKS_EPI // N_O_CHUNKS
            fire_now = (block_idx + 1) % _BLOCKS_PER_OCHUNK == 0
            if cutlass.const_expr(fire_now):
                # fence_proxy needed before TMA reads SMEM written by store_swizzled
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_full[block_idx // _BLOCKS_PER_OCHUNK].arrive()

        if _row_valid:
            nvvm.atomicrmw(nvvm.AtomicOp.MAX, _amax_o_ptr, _amax_o_local.bitcast(cutlass.Int32))

        epilogue_state = epilogue_state ^ cutlass.Int32(1)

        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_q, nxt_hb, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        q_super_idx, head_idx, batch_idx = _dispatch_decode_payload(
            nxt_q,
            nxt_hb,
            cta_in_pair,
            n_q_supers,
            n_qh,
            n_batch,
            seq_kv_lens_tensor,
        )
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)
        eff_seqlen_kv = _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, seqlen_kv)
        eff_seqlen_q = _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, seqlen_q, n_batch)
        bounds = _bounds_for_tile(q_super_idx, eff_seqlen_q, eff_seqlen_kv, cta_in_pair)

    if cutlass.const_expr(CFG.CTA_MMA == 2):
        peer_cta = cta_id_x ^ cutlass.Int32(1)
        bars.mb_tmem_dealloc.arrive_on_peer(peer_cta)
    bars.mb_tmem_dealloc.arrive()


# === Host launcher ===


@cute.jit
def _host(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    v_tensor: cute.Tensor,
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    sinks_tensor: cute.Tensor,
    seq_kv_lens_tensor: cute.Tensor,
    o_desc_words: cute.Tensor,
    problem_size: Tuple[int, int, int, int, int, int],
    scale_softmax_log2: cutlass.Float32,
    o_scale_fused: cutlass.Float32,
    n_thd_units: cutlass.Int32,
    # FROST's quantized ABI: the four 1-element fp32 scales arrive as DEVICE
    # TENSORS and the amax of O is an OUTPUT.  The pre-upstream contract pre-folded the scales on
    # the host and produced no amax -- that gap is what this closes.  Template:
    # the shipped sm107/prefill_d128_fp8.py sibling (same lineage).
    descale_q_t: cute.Tensor,
    descale_k_t: cute.Tensor,
    descale_v_t: cute.Tensor,
    scale_o_t: cute.Tensor,
    amax_o_tensor: cute.Tensor,
    # FROST passes the dense padded-Q trim tensor POSITIONALLY on every call.
    # These kernels have no Q-trim epilogue (Capabilities.dense_seq_q_trim=False)
    # so it is accepted and unused -- but the SLOT is mandatory: without it the
    # adapter's 12th positional lands on `stream` and execute dies with
    # "got multiple values for argument 'stream'".
    seq_q_lens_tensor: Optional[cute.Tensor] = None,
    # FROST plans must run on the caller's stream (engine contract; there is a
    # dedicated stream-respect test).  Threaded exactly as the shipped
    # sm107/prefill_d128_fp8.py sibling does.
    # THD device metadata build: the CALLER's Q/KV length tensors, (B,) lengths
    # or (B+1,) cu prefix sums per side via thd_lens_form.  Consumed only by the
    # setup kernel, which writes the metadata buffer device-side.
    thd_q_lens_tensor: Optional[cute.Tensor] = None,
    thd_kv_lens_tensor: Optional[cute.Tensor] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    stream: _cuda_driver.CUstream = None,
) -> None:
    B, QH, KH, SQ, SKV, _ = problem_size
    if cutlass.const_expr(CFG.THD_VARLEN):
        # Packed token totals are RUNTIME values: the adapter passes 0 in the
        # problem_size seq slots by THD contract.  Without this the kernel runs
        # at SQ = SKV = 0 -- descriptor extents, loop bounds and the tile count
        # all collapse, and the output is uniformly wrong with no crash.
        SQ = q_tensor.shape[1]
        SKV = k_tensor.shape[1]

    _O_GRANU_ELEMS = CFG.O_SWZ_BYTES // CFG.BPE_O
    qk_box_q = (1, CFG.TILE_M, 1, TMA_QK_GRANU_ELEMS)
    qk_box_k = (1, CFG.TILE_N // CFG.CTA_MMA, 1, TMA_QK_GRANU_ELEMS)
    vo_box_v = (1, CFG.TILE_N, 1, TMA_VO_GRANU_ELEMS)
    vo_box_o = (1, CFG.TILE_M, 1, _O_GRANU_ELEMS)
    stride_order = (3, 2, 1, 0)

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    tma_q_desc = tmap.create_tensor_map_tiled_from_view(
        q_tensor,
        box_dims=qk_box_q,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(
        k_tensor,
        box_dims=qk_box_k,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.K_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(
        v_tensor,
        box_dims=vo_box_v,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.V_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    tma_o_desc = tmap.create_tensor_map_tiled_from_view(
        o_tensor,
        box_dims=vo_box_o,
        stride_order=stride_order,
        swizzle=_tma_swz(CFG.O_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )

    rows_per_cluster = CFG.TILES_Q * CFG.TILE_M * CFG.CTA_MMA
    q_clusters = (SQ + rows_per_cluster - 1) // rows_per_cluster
    grid_q_supers = q_clusters * CFG.CTA_MMA
    q_supers = grid_q_supers
    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: build the per-batch O descriptor array, then launch the exact
        # flat batch-outermost grid (n_thd_units host-computed); grid_x = units*CGA_M.
        _build_thd_meta_o_descs_kernel(
            o_tensor,
            tma_o_desc,
            tma_k_desc,
            tma_v_desc,
            o_desc_words,
            seq_kv_lens_tensor,
            thd_q_lens_tensor,
            thd_kv_lens_tensor,
            thd_lens_form,
            cutlass.Int32(QH),
            cutlass.Int32(B),
            cutlass.Int32(o_tensor.stride[1]),
            cutlass.Int32(CGA_TILE_M),
            n_thd_units,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_thd_units * cutlass.Int32(CFG.CGA_M), cutlass.Int32(1), cutlass.Int32(1))
    else:
        grid_shape = (grid_q_supers, QH, B) if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_NATURAL) else (grid_q_supers * QH * B, 1, 1)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        tma_o_desc,
        lse_tensor,
        sinks_tensor,
        seq_kv_lens_tensor,
        o_desc_words,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(q_supers),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        scale_softmax_log2,
        o_scale_fused,
        descale_q_t,
        descale_k_t,
        descale_v_t,
        scale_o_t,
        amax_o_tensor,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(CFG.CTA_MMA, 1, 1),
        stream=stream,
    )


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    b: int = 1,
    qh: int = 1,
    kh: int = 1,
    sq: int = 256,
    skv: int = 128,
    d_qk: int = CFG.TILE_K,
    d_v: int = CFG.TILE_O,
    has_lse: bool = True,
    # The f16 THD compile key carries the caller's DECLARED packed strides
    # (api_dsl._thd_compile_kwargs) and the ragged-Stats layout.
    q_stride: Optional[tuple] = None,
    k_stride: Optional[tuple] = None,
    v_stride: Optional[tuple] = None,
    o_stride: Optional[tuple] = None,
    lse_head_major: bool = False,
    lse_head_stride: int = 0,
    lse_stride: Optional[tuple] = None,
) -> Callable:
    """Compile a kernel with ALL dims concrete (pins TMA descriptor strides).

    THD/varlen: q/k/v/o/lse PACKED with batch dim 1; ``b`` = logical batch."""
    # THD/varlen is NOT ported for this kernel yet.  The setup-kernel call site
    # below still speaks the pre-upstream 7-arg contract, while FROST's
    # thd_helpers.build_thd_meta_o_descs_kernel takes 14 args and a different
    # metadata layout (4B+4 with batch_remap + a claim counter, vs the 3B+2
    # here), and the scheduler decode differs to match.  Raise here rather than
    # let it fail as an arity error deep in the trace -- and so no engine row
    # can advertise thd=True for this kernel and appear to work.  The dense
    # path is unaffected: CFG.THD_VARLEN is 0 and every THD branch folds out.
    # ---- FROST adapter ABI ------------------------------------------------
    # lower_dsl_prefill calls EVERY kernel with the full forward signature.
    # This body carries only what the port brought over, so anything
    # it cannot honor RAISES rather than being silently ignored: a raise here
    # means the engine's Capabilities row is lying, which is the failure we
    # want loud.  (Capabilities: lse_optional=False, no strided Stats.)
    if lse_stride is not None:
        raise NotImplementedError(f"{__name__}: strided Stats not ported (contiguous [B, H, S] only)")
    if d_qk > CFG.TILE_K or d_v > CFG.TILE_O or d_qk <= 0 or d_v <= 0:
        raise ValueError(f"{__name__}: envelope is 0 < d_qk <= {CFG.TILE_K}, 0 < d_v <= {CFG.TILE_O}; " f"got ({d_qk}, {d_v})")
    _fake_batch = 1 if CFG.THD_VARLEN else b
    if CFG.THD_VARLEN:
        # Dynamic packed token totals: a new packed total RE-BINDS the same
        # artifact instead of minting one per shape.
        sq = cute.sym_int(divisibility=1)
        skv = cute.sym_int(divisibility=1)

    def _fake_bshd(shape, stride, dtype=STORAGE_DTYPE, bpe=CFG.BPE):
        """BSHD fake tensor, compact or at the caller's DECLARED strides.

        The head dim must be innermost-contiguous, and the seq/head global
        strides feed TMA so they obey the 16-byte global-stride rule.  Under THD
        the fake binds the token stride for the extent-1 batch dim, as _thd_view
        does at runtime: tokens * token_stride is never stepped and overflows
        the int32 stride slot on long packed KV (GitHub #980)."""
        if stride is None:
            return cute.runtime.make_fake_compact_tensor(dtype, shape, stride_order=(3, 2, 1, 0), assumed_align=16)
        if stride[3] != 1:
            raise ValueError(f"declared stride {stride}: the head dim must be innermost-contiguous (stride[3] == 1)")
        for axis in (1, 2):
            if (stride[axis] * bpe) % 16 != 0:
                raise ValueError(f"declared stride {stride} axis {axis} must be a 16-byte multiple at BPE={bpe} (TMA global-stride rule)")
        if CFG.THD_VARLEN:
            # Extent-1 batch dim: bind the token stride, as _thd_view does at
            # runtime -- T * token_stride is never stepped and overflows the int32
            # stride slot on long packed KV with wide tokens (GitHub #980).
            return cute.runtime.make_fake_tensor(dtype, shape, (stride[1], stride[1], stride[2], stride[3]), assumed_align=16)
        return cute.runtime.make_fake_tensor(dtype, shape, tuple(stride), assumed_align=16)

    fake_q = _fake_bshd((_fake_batch, sq, qh, d_qk), q_stride)
    fake_k = _fake_bshd((_fake_batch, skv, kh, d_qk), k_stride)
    fake_v = _fake_bshd((_fake_batch, skv, kh, d_v), v_stride)
    fake_o = _fake_bshd((_fake_batch, sq, qh, d_v), o_stride, dtype=OUT_STORAGE_DTYPE, bpe=CFG.BPE_O)
    # has_lse=False (no Stats output): the LSE argument is None-specialized and
    # the store is compiled out entirely -- no dummy buffer exists at any level,
    # which is what lets the dense graph report get_workspace_size() == 0.
    # Mirrors the shipped sm107/prefill_d128_fp8.py.
    if not has_lse:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride require has_lse=True")
        fake_lse = None
    elif CFG.THD_VARLEN:
        # Packed ragged Stats in the CALLER's declared layout; the epilogue store
        # branches on the STATIC rank, so the layout is fully encoded here.
        if lse_head_major:
            _lse_hs = lse_head_stride if lse_head_stride else sq
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1, qh, _lse_hs), stride_order=(2, 1, 0), assumed_align=4)
        else:
            if lse_head_stride:
                raise ValueError("lse_head_stride is head-major-only (token-major (T, H) is compact)")
            fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (sq, qh), stride_order=(1, 0), assumed_align=4)
    else:
        if lse_head_major or lse_head_stride:
            raise ValueError("lse_head_major / lse_head_stride are THD-only (dense LSE is compact (B, H, Sq))")
        fake_lse = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (_fake_batch, qh, sq), stride_order=(2, 1, 0), assumed_align=16)
    fake_sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (qh,),
        stride_order=(0,),
        assumed_align=16,
    )
    _skv_len = (4 * b + 4) if CFG.THD_VARLEN else b
    fake_seq_kv_lens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_skv_len,),
        stride_order=(0,),
        assumed_align=16,
    )
    _odesc_len = ((b + 3) * _TENSOR_MAP_QWORDS) if CFG.THD_VARLEN else 1
    fake_o_desc = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (_odesc_len,),
        stride_order=(0,),
        assumed_align=16,
    )

    # The four 1-element fp32 device scales + the amax_o output slot; the
    # compiled artifact must declare them or the adapter's positional call
    # cannot bind (FROST quantized ABI).
    def _fake_scale():
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (1,),
            stride_order=(0,),
            assumed_align=4,
        )

    fake_amax_o = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        stride_order=(0,),
        assumed_align=16,
    )
    # Caller Q/KV length tensors, read only by the setup kernel.  DYNAMIC extent:
    # the accepted forms differ in length ((B,) vs (B+1,)) and which one arrives
    # rides the runtime thd_lens_form bitmask.
    if CFG.THD_VARLEN:
        fake_thd_q_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_kv_lens = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(divisibility=1),), stride_order=(0,), assumed_align=4)
        fake_thd_lens_form = cutlass.Int32(0)
    else:
        fake_thd_q_lens = fake_thd_kv_lens = fake_thd_lens_form = None
    return cute.compile(
        _host,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        fake_sinks,
        fake_seq_kv_lens,
        fake_o_desc,
        (b, qh, kh, 0, 0, 0) if CFG.THD_VARLEN else (b, qh, kh, sq, skv, 0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        _fake_scale(),
        fake_amax_o,
        None,  # seq_q_lens_tensor: accepted-but-unused slot (no Q-trim epilogue)
        fake_thd_q_lens,
        fake_thd_kv_lens,
        fake_thd_lens_form,
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


def _main():
    """Minimal CLI for compile-check / perf bring-up."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--hq", type=int, default=1)
    parser.add_argument("--hk", type=int, default=1)
    parser.add_argument("--sq", type=int, default=256)
    parser.add_argument("--skv", type=int, default=128)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--iters", type=int, default=0)
    args = parser.parse_args()

    print(f"[d256_fp8] compile b={args.b} qh={args.hq} kh={args.hk} " f"sq={args.sq} skv={args.skv}", flush=True)
    fn = compile(args.b, args.hq, args.hk, args.sq, args.skv)
    print(f"[d256_fp8] compile OK: {fn}", flush=True)
    if args.validate:
        print("[d256_fp8] --validate: run via the test driver, not this CLI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
